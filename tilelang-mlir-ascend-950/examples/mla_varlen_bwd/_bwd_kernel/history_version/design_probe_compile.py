"""Stage 1 设计期编译探针（mla_varlen_bwd / _bwd_kernel，packed 分支）。

无设备（仅前端 + 编译）：`ASCEND_RT_VISIBLE_DEVICES=""`。
必须在**仓根之外**运行（仓根 `tilelang/` 是 dev root，会遮蔽 site-packages 的
0.1.15 私有 TVM）：

    cd /tmp && ASCEND_RT_VISIBLE_DEVICES="" \
        /workspace/venv-ascend950/bin/python \
        /workspace/tilelang-mlir-ascend-950/examples/mla_varlen_bwd/_bwd_kernel/history_version/design_probe_compile.py

探针清单（对应 DESIGN.md §3.5.4；dense 版已验证核心不重复探）：

  P1 `packed_skeleton`      — 动态 `total_q/total_kv` + **动态 1-D grid**
      `H * (total_kv//BN + B)` + 逐 request 运行时偏移装载 + 5 个 NT GEMM +
      `dual_copy` 2:1 + `sid` + store-atomic + 块尾整 tile 写回（静态式解码）。
      隔离「动态 grid + 混合 kernel + 运行时偏移」这一未组合过的形态。
  P2 `packed_full`          — P1 + 融合式 (前缀和 + decode) 串行扫描 + `live`
      谓词整块包裹 + head-major `lse/delta` + 运行期 mask + 块尾**双分支**
      （整 tile 快路 / 逐 lane 守卫慢路）。
  P3 `store_symbolic_rows`  — 块尾/原子写回用**符号长度**区段（`nr` 运行期）
      替代双分支。
  P4 `l0c_clear`            — `T.clear` 作用于 L0C 累加器（空 query 循环的
      防御性初始化）+ 单 GEMM 链，最小形态。
  P5 `static_grid_live`     — **静态化方案**：grid 取编译期上界
      `H * (TK_MAX//BN + B)`，size 仍为 `T.dynamic`，`live` 谓词丢弃越界 tile。

注：`prim_func` 内的循环必须写在被 AST 变换的函数体里，故计算体用 `@T.macro`
（与源 `GroupTiling` 的写法一致），不能用普通 Python 函数。
"""

import traceback

import tilelang
import tilelang.ascend.language as T

# ---------------------------------------------------------------------------
# 共同几何（编译期常量）：batch 进编译期（变长组的组数），
# T_q/T_kv 运行期动态（T.dynamic），D_qk=192 / D_v=128 / 64 全为 16 的倍数。
# ---------------------------------------------------------------------------
B, H, DQK, DV = 3, 2, 192, 128
BN, BM = 64, 64
HN = BN // 2  # dual_copy 2:1 的每 AIV 半区
DT = "float16"
SCALE = DQK**-0.5


@T.macro
def _decode(cu_kv, tile, req, n0l, cnt, live):
    """融合式 (前缀和 + decode)：一次 O(B) 串行扫描得出 request 与块内 key 起点。

    语义等价于 `GroupTiling.cumsum_offsets` + `GroupTiling.decode`
    （`s_cum[g+1]-s_cum[g] = ceil(len_kv/BN)`，`row = (tile - s_cum[g]) * BN`），
    但去掉了源实现的 warp shuffle 前缀扫描（NPU 无 warp）与共享块表。
    """
    req[0] = T.int32(0)
    n0l[0] = T.int32(0)
    cnt[0] = T.int32(0)
    for g in T.serial(B):
        ntg = (cu_kv[g + 1] - cu_kv[g] + T.int32(BN - 1)) // T.int32(BN)
        if (tile >= cnt[0]) & (tile < cnt[0] + ntg):
            req[0] = g
            n0l[0] = (tile - cnt[0]) * T.int32(BN)
        cnt[0] = cnt[0] + ntg
    live[0] = T.if_then_else(tile < cnt[0], T.int32(1), T.int32(0))


@T.macro
def _body(
    q, k, v, dout, lse, delta, dq_acc, dk, dv,
    k_s, v_s, kT_s, q_s, qT_s, do_s, doT_s, pT_s, dsT_s, ds_s,
    st_c, dpt_c, dq_c, dk_c, dv_c,
    st_u, dp_u, p16_u, dsT_u, ds_u, lse_u, dlt_u,
    tail_u, tailv_u, out_k, out_v,
    h, tile, sid, n0_dyn, lq, lkv, shift, kv_start, q_start,
    l0c_clear, symbolic_store,
):
    """一个 (request, key 块, head) 的全部计算（key-stationary，dense 已验证结构）。"""
    koff = kv_start + n0_dyn
    T.copy(k[koff : koff + BN, h, :], k_s)
    T.copy(v[koff : koff + BN, h, :], v_s)
    T.copy(k[koff : koff + BN, h, :], kT_s, transpose=True)
    if l0c_clear:
        T.clear(dk_c)
        T.clear(dv_c)

    loop_st = T.max(0, T.floordiv(n0_dyn - shift, BM))
    loop_ed = T.ceildiv(lq, BM)
    for kq in T.Pipelined(loop_st, loop_ed, num_stages=1):
        m0 = kq * BM
        qoff = q_start + m0
        T.copy(q[qoff : qoff + BM, h, :], q_s)
        T.copy(dout[qoff : qoff + BM, h, :], do_s)
        T.copy(q[qoff : qoff + BM, h, :], qT_s, transpose=True)
        T.copy(dout[qoff : qoff + BM, h, :], doT_s, transpose=True)
        # head-major lse/delta: [H, T_q]
        T.copy(lse[h, qoff : qoff + BM], lse_u)
        T.copy(delta[h, qoff : qoff + BM], dlt_u)

        T.gemm(k_s, q_s, st_c, transpose_B=True, clear_accum=True)
        T.gemm(v_s, do_s, dpt_c, transpose_B=True, clear_accum=True)
        T.dual_copy(st_c, st_u)
        T.dual_copy(dpt_c, dp_u)

        # A2 块级特化：谓词全为运行期量（len_kv/len_q/shift 来自 cu_seqlens）
        fast = (n0_dyn + BN <= lkv) & (m0 + BM <= lq) & (n0_dyn + BN - 1 <= m0 + shift)
        if fast:
            with T.SimtVF(threads=128):
                for i, j in T.Parallel(HN, BM):
                    p = T.exp(st_u[i, j] * SCALE - lse_u[j])
                    p16_u[i, j] = T.cast(p, DT)
                    dsv = p * (dp_u[i, j] - dlt_u[j])
                    dsT_u[i, j] = T.cast(dsv, DT)
                    ds_u[j, i] = T.cast(dsv, DT)
        else:
            with T.SimtVF(threads=128):
                for i, j in T.Parallel(HN, BM):
                    gi = n0_dyn + T.cast(sid, "int32") * HN + i
                    gj = m0 + j
                    vis = (gi < lkv) & (gj < lq) & (gi <= gj + shift)
                    p = T.exp(st_u[i, j] * SCALE - lse_u[j])
                    pv = T.if_then_else(vis, p, T.cast(0, "float32"))
                    p16_u[i, j] = T.cast(pv, DT)
                    dsv = pv * (dp_u[i, j] - dlt_u[j])
                    dsT_u[i, j] = T.cast(dsv, DT)
                    ds_u[j, i] = T.cast(dsv, DT)

        T.dual_copy(p16_u, pT_s)
        T.dual_copy(dsT_u, dsT_s)
        T.dual_copy(ds_u, ds_s)
        T.gemm(pT_s, doT_s, dv_c, transpose_B=True, clear_accum=(kq == loop_st))
        T.gemm(dsT_s, qT_s, dk_c, transpose_B=True, clear_accum=(kq == loop_st))
        T.gemm(ds_s, kT_s, dq_c, transpose_B=True, clear_accum=True)
        if symbolic_store:
            nq = T.max(0, T.min(BM, q_start + lq - qoff))
            T.set_atomic("add", "float32")
            T.copy(dq_c[:nq, :], dq_acc[qoff : qoff + nq, h, :])
            T.set_atomic_none()
        else:
            # 整 tile 原子写回：越过 len_q 的行 dS 恒 0（掩码臂 gj<len_q），
            # 落到下一 request 的 dq_acc 上是「原子加精确 0」，数值无害
            T.set_atomic("add", "float32")
            T.copy(dq_c, dq_acc[qoff : qoff + BM, h, :])
            T.set_atomic_none()

    T.dual_copy(dk_c, tail_u)
    T.dual_copy(dv_c, tailv_u)
    with T.SimtVF(threads=128):
        for i, d in T.Parallel(HN, DQK):
            out_k[i, d] = T.cast(tail_u[i, d] * SCALE, DT)
        for i, d in T.Parallel(HN, DV):
            out_v[i, d] = T.cast(tailv_u[i, d], DT)

    own = koff + T.cast(sid, "int32") * HN
    kv_end = kv_start + lkv
    if symbolic_store:
        nr = T.max(0, T.min(HN, kv_end - own))
        T.copy(out_k[:nr, :], dk[own : own + nr, h, :])
        T.copy(out_v[:nr, :], dv[own : own + nr, h, :])
    else:
        # 双分支：内部半块整 tile 快路；每个 request 的尾半块走逐 lane 守卫
        if own + HN <= kv_end:
            T.copy(out_k, dk[own : own + HN, h, :])
            T.copy(out_v, dv[own : own + HN, h, :])
        else:
            with T.SimtVF(threads=128):
                for i, d in T.Parallel(HN, DQK):
                    if own + i < kv_end:
                        dk[own + i, h, d] = out_k[i, d]
                for i, d in T.Parallel(HN, DV):
                    if own + i < kv_end:
                        dv[own + i, h, d] = out_v[i, d]


def _alloc():
    """L1 / L0C / UB 布局（与 dense 已验证形态逐字相同，UB 为 dual_copy 半区）。"""
    d = {}
    d["k_s"] = T.alloc_l1((BN, DQK), DT)
    d["v_s"] = T.alloc_l1((BN, DV), DT)
    d["kT_s"] = T.alloc_l1((DQK, BN), DT)
    d["q_s"] = T.alloc_l1((BM, DQK), DT)
    d["qT_s"] = T.alloc_l1((DQK, BM), DT)
    d["do_s"] = T.alloc_l1((BM, DV), DT)
    d["doT_s"] = T.alloc_l1((DV, BM), DT)
    d["pT_s"] = T.alloc_l1((BN, BM), DT)
    d["dsT_s"] = T.alloc_l1((BN, BM), DT)
    d["ds_s"] = T.alloc_l1((BM, BN), DT)
    d["st_c"] = T.alloc_l0c((BN, BM), "float32")
    d["dpt_c"] = T.alloc_l0c((BN, BM), "float32")
    d["dq_c"] = T.alloc_l0c((BM, DQK), "float32")
    d["dk_c"] = T.alloc_l0c((BN, DQK), "float32")
    d["dv_c"] = T.alloc_l0c((BN, DV), "float32")
    d["st_u"] = T.alloc_shared((HN, BM), "float32")
    d["dp_u"] = T.alloc_shared((HN, BM), "float32")
    d["p16_u"] = T.alloc_shared((HN, BM), DT)
    d["dsT_u"] = T.alloc_shared((HN, BM), DT)
    d["ds_u"] = T.alloc_shared((BM, HN), DT)
    d["lse_u"] = T.alloc_shared((BM,), "float32")
    d["dlt_u"] = T.alloc_shared((BM,), "float32")
    d["tail_u"] = T.alloc_shared((HN, DQK), "float32")
    d["tailv_u"] = T.alloc_shared((HN, DV), "float32")
    d["out_k"] = T.alloc_shared((HN, DQK), DT)
    d["out_v"] = T.alloc_shared((HN, DV), DT)
    return d


def _args(d):
    return (
        d["k_s"], d["v_s"], d["kT_s"], d["q_s"], d["qT_s"], d["do_s"], d["doT_s"],
        d["pT_s"], d["dsT_s"], d["ds_s"], d["st_c"], d["dpt_c"], d["dq_c"],
        d["dk_c"], d["dv_c"], d["st_u"], d["dp_u"], d["p16_u"], d["dsT_u"],
        d["ds_u"], d["lse_u"], d["dlt_u"], d["tail_u"], d["tailv_u"],
        d["out_k"], d["out_v"],
    )


# ---------------------------------------------------------------------------
# P1: 动态 grid + 运行时偏移（静态式解码）
# ---------------------------------------------------------------------------
def build_skeleton():
    total_q = T.dynamic("total_q")
    total_kv = T.dynamic("total_kv")
    ntiles = T.floordiv(total_kv, BN) + B
    grid = ntiles * H

    @T.prim_func
    def main(
        q: T.Tensor((total_q, H, DQK), DT),
        k: T.Tensor((total_kv, H, DQK), DT),
        v: T.Tensor((total_kv, H, DV), DT),
        dout: T.Tensor((total_q, H, DV), DT),
        lse: T.Tensor((H, total_q), "float32"),
        delta: T.Tensor((H, total_q), "float32"),
        cu_q: T.Tensor((B + 1,), "int32"),
        cu_kv: T.Tensor((B + 1,), "int32"),
        dq_acc: T.Tensor((total_q, H, DQK), "float32"),
        dk: T.Tensor((total_kv, H, DQK), DT),
        dv: T.Tensor((total_kv, H, DV), DT),
    ):
        with T.Kernel(grid) as bid:
            tile = bid // H  # 静态除数
            h = bid % H
            sid = T.call_extern("int64", "asc_get_sub_block_id")
            d = _alloc()
            b_dyn = tile // ntiles
            n0_dyn = (tile % ntiles) * BN
            q_start = cu_q[b_dyn]
            kv_start = cu_kv[b_dyn]
            lq = cu_q[b_dyn + 1] - q_start
            lkv = cu_kv[b_dyn + 1] - kv_start
            _body(
                q, k, v, dout, lse, delta, dq_acc, dk, dv, *_args(d),
                h, tile, sid, n0_dyn, lq, lkv, lkv - lq, kv_start, q_start,
                False, False,
            )

    return main


# ---------------------------------------------------------------------------
# P2: 融合式 decode + live 谓词 + 双分支块尾
# ---------------------------------------------------------------------------
def build_full(l0c_clear=False, symbolic_store=False):
    total_q = T.dynamic("total_q")
    total_kv = T.dynamic("total_kv")
    ntiles = T.floordiv(total_kv, BN) + B
    grid = ntiles * H

    @T.prim_func
    def main(
        q: T.Tensor((total_q, H, DQK), DT),
        k: T.Tensor((total_kv, H, DQK), DT),
        v: T.Tensor((total_kv, H, DV), DT),
        dout: T.Tensor((total_q, H, DV), DT),
        lse: T.Tensor((H, total_q), "float32"),
        delta: T.Tensor((H, total_q), "float32"),
        cu_q: T.Tensor((B + 1,), "int32"),
        cu_kv: T.Tensor((B + 1,), "int32"),
        dq_acc: T.Tensor((total_q, H, DQK), "float32"),
        dk: T.Tensor((total_kv, H, DQK), DT),
        dv: T.Tensor((total_kv, H, DV), DT),
    ):
        with T.Kernel(grid) as bid:
            tile = bid // H
            h = bid % H
            sid = T.call_extern("int64", "asc_get_sub_block_id")
            d = _alloc()
            req = T.alloc_local([1], "int32")
            n0l = T.alloc_local([1], "int32")
            cnt = T.alloc_local([1], "int32")
            live = T.alloc_local([1], "int32")
            _decode(cu_kv, tile, req, n0l, cnt, live)
            if live[0] == T.int32(1):
                q_start = cu_q[req[0]]
                kv_start = cu_kv[req[0]]
                lq = cu_q[req[0] + 1] - q_start
                lkv = cu_kv[req[0] + 1] - kv_start
                _body(
                    q, k, v, dout, lse, delta, dq_acc, dk, dv, *_args(d),
                    h, tile, sid, n0l[0], lq, lkv, lkv - lq, kv_start, q_start,
                    l0c_clear, symbolic_store,
                )

    return main


# ---------------------------------------------------------------------------
# P4: T.clear 作用于 L0C（最小链：1 个 NT GEMM）
# ---------------------------------------------------------------------------
def build_l0c_clear():
    total_kv = T.dynamic("total_kv")
    ntiles = T.floordiv(total_kv, BN) + B
    grid = ntiles * H

    @T.prim_func
    def main(
        k: T.Tensor((total_kv, H, DQK), DT),
        q: T.Tensor((total_kv, H, DQK), DT),
        cu_kv: T.Tensor((B + 1,), "int32"),
        dk: T.Tensor((total_kv, H, DQK), DT),
    ):
        with T.Kernel(grid) as bid:
            tile = bid // H
            h = bid % H
            req = T.alloc_local([1], "int32")
            n0l = T.alloc_local([1], "int32")
            cnt = T.alloc_local([1], "int32")
            live = T.alloc_local([1], "int32")
            dsT_s = T.alloc_l1((BN, BM), DT)
            qT_s = T.alloc_l1((DQK, BM), DT)
            dk_c = T.alloc_l0c((BN, DQK), "float32")
            tail_u = T.alloc_shared((HN, DQK), "float32")
            out_k = T.alloc_shared((HN, DQK), DT)
            _decode(cu_kv, tile, req, n0l, cnt, live)
            if live[0] == T.int32(1):
                kv_start = cu_kv[req[0]]
                koff = kv_start + n0l[0]
                T.copy(k[koff : koff + BN, h, 0:BM], dsT_s)
                T.copy(q[koff : koff + BM, h, :], qT_s, transpose=True)
                T.clear(dk_c)  # 空 query 循环的防御性初始化
                T.gemm(dsT_s, qT_s, dk_c, transpose_B=True, clear_accum=False)
                T.dual_copy(dk_c, tail_u)
                with T.SimtVF(threads=128):
                    for i, dd in T.Parallel(HN, DQK):
                        out_k[i, dd] = T.cast(tail_u[i, dd] * SCALE, DT)
                T.copy(out_k, dk[koff : koff + HN, h, :])

    return main


# ---------------------------------------------------------------------------
# P5: 静态化方案（编译期 grid 上界 + live 谓词）
# ---------------------------------------------------------------------------
TK_MAX = 4096  # 编译期上界（host 契约：total_kv <= TK_MAX）


def build_static_grid():
    total_q = T.dynamic("total_q")
    total_kv = T.dynamic("total_kv")
    grid = (TK_MAX // BN + B) * H  # 纯静态

    @T.prim_func
    def main(
        q: T.Tensor((total_q, H, DQK), DT),
        k: T.Tensor((total_kv, H, DQK), DT),
        v: T.Tensor((total_kv, H, DV), DT),
        dout: T.Tensor((total_q, H, DV), DT),
        lse: T.Tensor((H, total_q), "float32"),
        delta: T.Tensor((H, total_q), "float32"),
        cu_q: T.Tensor((B + 1,), "int32"),
        cu_kv: T.Tensor((B + 1,), "int32"),
        dq_acc: T.Tensor((total_q, H, DQK), "float32"),
        dk: T.Tensor((total_kv, H, DQK), DT),
        dv: T.Tensor((total_kv, H, DV), DT),
    ):
        with T.Kernel(grid) as bid:
            tile = bid // H
            h = bid % H
            sid = T.call_extern("int64", "asc_get_sub_block_id")
            d = _alloc()
            req = T.alloc_local([1], "int32")
            n0l = T.alloc_local([1], "int32")
            cnt = T.alloc_local([1], "int32")
            live = T.alloc_local([1], "int32")
            _decode(cu_kv, tile, req, n0l, cnt, live)
            if live[0] == T.int32(1):
                q_start = cu_q[req[0]]
                kv_start = cu_kv[req[0]]
                lq = cu_q[req[0] + 1] - q_start
                lkv = cu_kv[req[0] + 1] - kv_start
                _body(
                    q, k, v, dout, lse, delta, dq_acc, dk, dv, *_args(d),
                    h, tile, sid, n0l[0], lq, lkv, lkv - lq, kv_start, q_start,
                    False, False,
                )

    return main


# ---------------------------------------------------------------------------
# 运行器
# ---------------------------------------------------------------------------
PROBES = [
    ("P1 packed_skeleton", build_skeleton),
    ("P2 packed_full", build_full),
    ("P3 store_symbolic_rows", lambda: build_full(symbolic_store=True)),
    ("P4 l0c_clear", build_l0c_clear),
    ("P5 static_grid_live", build_static_grid),
    ("P2b full+l0c_clear", lambda: build_full(l0c_clear=True)),
]

MARKS = (
    "ascend_gemm_l1",
    "asc_copy_l0c2ub",
    "ascend_nd2nz_scatter",
    "asc_copy_l0c2gm",
    "asc_set_atomic_add_float",
    "__mix__",
    "asc_get_sub_block_id",
    "asc_fill_l1",
    "asc_copy_gm2l1_nd2nz",
    "asc_copy_gm2l1_dn2nz",
    "asc_copy_ub2gm",
)


def main():
    results = []
    for name, build in PROBES:
        try:
            fn = build()
            src = tilelang.compile(fn, target="ascend").get_kernel_source()
            marks = sorted({kw for kw in MARKS if kw in src})
            results.append((name, "COMPILE_OK", f"{len(src)} B; marks={marks}"))
        except Exception as exc:  # noqa: BLE001 - 探针必须逐项报告，不中断
            last = traceback.format_exc().strip().splitlines()[-1]
            results.append(
                (name, "COMPILE_FAIL", f"{type(exc).__name__}: {str(exc)[:500]} || {last[:200]}")
            )

    print()
    for name, status, detail in results:
        print(f"[{status}] {name} -- {detail}", flush=True)
    ok = sum(1 for _, s, _ in results if s == "COMPILE_OK")
    print(f"\n[PROBE_SUMMARY] {ok}/{len(results)} COMPILE_OK", flush=True)


if __name__ == "__main__":
    main()
