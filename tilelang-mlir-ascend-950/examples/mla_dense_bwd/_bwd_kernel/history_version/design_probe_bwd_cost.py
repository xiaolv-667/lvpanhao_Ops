"""Stage 1 设计期代价探针（revision v2 / cube-bound）：标定内层迭代各阶段单价。

D-3 探针（≤2 个、每个 ≤10min）：只测**未知常数**，不实现完整算子。

被测对象 = `_bwd_kernel` 的 (key 块 x query 块) 内层迭代。v1 实测 T_op 455.3 ms，
msprof 口径 cube 8.3% / vector 73.5%；2 113 536 次迭代 / 28 核 ⇒ 6.03 µs/迭代。
本探针把一次迭代拆成若干段的单价（vector epilogue 3 种写法 / UB→L1 / GM→L1 /
L0C→UB / 5 个 GEMM / 整链），用来判定「6.03 µs 花在哪一段」，以及放大 tile 后的可达值。

运行（必须在仓根之外，否则 dev-root 的 tilelang/ 遮蔽 site-packages 0.1.15）：

    cd /tmp && source /workspace/tilelang-mlir-ascend-950/.agents/platforms/toolchains/ascend/env-950pr.sh \
      && TMPDIR=/workspace /workspace/venv-ascend950/bin/python \
         /workspace/tilelang-mlir-ascend-950/examples/mla_dense_bwd/_bwd_kernel/history_version/design_probe_bwd_cost.py

每个变体：grid=28（一核一块），块内 R 次迭代，perf_counter + npu.synchronize 计时，
取 3 次最小值；单价 = elapsed / R（每核每次迭代）。

模式（stage 开关，末尾统一跑 3 个 GEMM + 块尾写回，保证循环体不被 DCE）：
  mte2       4 个 GM→L1（q/do/qT/doT）+ lse/delta
  l0c2ub     2 个 L0C→UB dual_copy
  gemm5      5 个 NT GEMM
  ub2l1      3 个 UB→L1 dual_copy
  epi0       合并写 epilogue（p16 + dsT，无转置）
  epi1       epi0 + `ds_u[j,i]` 跨步写（v1 现形态）
  epi2       epi0 + 第二趟（跨步读 dsT_u / 合并写 ds_u）
  chain      v1 内层整链（含跨步转置写）
  chain_notr整链但省掉跨步转置写（= 转置免费时的上界）
  u2l1_tr    仅编译可行性：半区 UB -> L1 `transpose=True`
"""

import functools
import sys
import time

import tilelang
import tilelang.ascend.language as T
import torch
import torch_npu  # noqa: F401

DT = "bfloat16"
H, SQ, SKV = 8, 2048, 2048
DQK, DV = 192, 128
SCALE = DQK**-0.5
GRID = 28          # 物理 Cube 核数（platform core_source: cube_core_cnt=28）
R = 400            # 块内迭代次数


@functools.lru_cache(maxsize=None)
def build(mode, nk, mq):
    hn = nk // 2
    nch = SQ // mq
    load = mode in ("chain", "chain_notr", "mte2")
    g2 = mode in ("chain", "chain_notr", "gemm5")
    l2u = mode in ("chain", "chain_notr", "l0c2ub")
    epi = mode in ("chain", "chain_notr", "epi0", "epi1", "epi2")
    u2l1 = mode in ("chain", "chain_notr", "ub2l1")
    g3 = mode in ("chain", "chain_notr")

    @T.prim_func
    def main(
        q: T.Tensor((SQ, H, DQK), DT),          # type: ignore
        k: T.Tensor((SKV, H, DQK), DT),         # type: ignore
        v: T.Tensor((SKV, H, DV), DT),          # type: ignore
        do: T.Tensor((SQ, H, DV), DT),          # type: ignore
        lse: T.Tensor((H, SQ), "float32"),      # type: ignore
        delta: T.Tensor((H, SQ), "float32"),    # type: ignore
        dq_acc: T.Tensor((SQ, H, DQK), "float32"),   # type: ignore
        dk: T.Tensor((SKV, H, DQK), DT),        # type: ignore
        dv: T.Tensor((SKV, H, DV), DT),         # type: ignore
    ):
        with T.Kernel(GRID) as bid:
            h = bid % H
            n0 = ((bid // H) % 4) * nk

            k_s = T.alloc_l1((nk, DQK), DT)
            v_s = T.alloc_l1((nk, DV), DT)
            kT_s = T.alloc_l1((DQK, nk), DT)
            q_s = T.alloc_l1((mq, DQK), DT)
            qT_s = T.alloc_l1((DQK, mq), DT)
            do_s = T.alloc_l1((mq, DV), DT)
            doT_s = T.alloc_l1((DV, mq), DT)
            pT_s = T.alloc_l1((nk, mq), DT)
            dsT_s = T.alloc_l1((nk, mq), DT)
            ds_s = T.alloc_l1((mq, nk), DT)
            st_c = T.alloc_l0c((nk, mq), "float32")
            dpt_c = T.alloc_l0c((nk, mq), "float32")
            dq_c = T.alloc_l0c((mq, DQK), "float32")
            dk_c = T.alloc_l0c((nk, DQK), "float32")
            dv_c = T.alloc_l0c((nk, DV), "float32")
            st_u = T.alloc_shared((hn, mq), "float32")
            dp_u = T.alloc_shared((hn, mq), "float32")
            p16_u = T.alloc_shared((hn, mq), DT)
            dsT_u = T.alloc_shared((hn, mq), DT)
            ds_u = T.alloc_shared((mq, hn), DT)
            lse_u = T.alloc_shared((mq,), "float32")
            dlt_u = T.alloc_shared((mq,), "float32")
            tail_u = T.alloc_shared((hn, DQK), "float32")
            tailv_u = T.alloc_shared((hn, DV), "float32")
            out_k = T.alloc_shared((hn, DQK), DT)
            out_v = T.alloc_shared((hn, DV), DT)

            T.copy(k[n0 : n0 + nk, h, :], k_s)
            T.copy(v[n0 : n0 + nk, h, :], v_s)
            T.copy(k[n0 : n0 + nk, h, :], kT_s, transpose=True)
            T.copy(q[0:mq, h, :], q_s)
            T.copy(do[0:mq, h, :], do_s)
            T.copy(q[0:mq, h, :], qT_s, transpose=True)
            T.copy(do[0:mq, h, :], doT_s, transpose=True)
            T.copy(lse[h, 0:mq], lse_u)
            T.copy(delta[h, 0:mq], dlt_u)
            # epi-only 模式：先给 st_u/dp_u 一份数据，避免读到未初始化缓冲
            if mode in ("epi0", "epi1", "epi2"):
                T.dual_copy(st_c, st_u)
                T.dual_copy(dpt_c, dp_u)

            for kq in T.Pipelined(0, R, num_stages=1):
                m0 = (kq % nch) * mq
                if load:
                    T.copy(q[m0 : m0 + mq, h, :], q_s)
                    T.copy(do[m0 : m0 + mq, h, :], do_s)
                    T.copy(q[m0 : m0 + mq, h, :], qT_s, transpose=True)
                    T.copy(do[m0 : m0 + mq, h, :], doT_s, transpose=True)
                    T.copy(lse[h, m0 : m0 + mq], lse_u)
                    T.copy(delta[h, m0 : m0 + mq], dlt_u)
                if g2:
                    T.gemm(k_s, q_s, st_c, transpose_B=True, clear_accum=True)
                    T.gemm(v_s, do_s, dpt_c, transpose_B=True, clear_accum=True)
                if l2u:
                    T.dual_copy(st_c, st_u)
                    T.dual_copy(dpt_c, dp_u)
                if epi:
                    if mode in ("chain", "chain_notr", "epi0"):
                        with T.SimtVF(threads=128):
                            for i, j in T.Parallel(hn, mq):
                                p = T.exp(st_u[i, j] * SCALE - lse_u[j])
                                p16_u[i, j] = T.cast(p, DT)
                                dsv = p * (dp_u[i, j] - dlt_u[j])
                                dsT_u[i, j] = T.cast(dsv, DT)
                                if mode == "chain":
                                    ds_u[j, i] = T.cast(dsv, DT)
                    elif mode == "epi1":
                        with T.SimtVF(threads=128):
                            for i, j in T.Parallel(hn, mq):
                                p = T.exp(st_u[i, j] * SCALE - lse_u[j])
                                p16_u[i, j] = T.cast(p, DT)
                                dsv = p * (dp_u[i, j] - dlt_u[j])
                                dsT_u[i, j] = T.cast(dsv, DT)
                                ds_u[j, i] = T.cast(dsv, DT)
                    else:  # epi2
                        with T.SimtVF(threads=128):
                            for i, j in T.Parallel(hn, mq):
                                p = T.exp(st_u[i, j] * SCALE - lse_u[j])
                                p16_u[i, j] = T.cast(p, DT)
                                dsT_u[i, j] = T.cast(
                                    p * (dp_u[i, j] - dlt_u[j]), DT)
                        with T.SimtVF(threads=128):
                            for j, i in T.Parallel(mq, hn):
                                ds_u[j, i] = dsT_u[i, j]
                if u2l1:
                    T.dual_copy(p16_u, pT_s)
                    T.dual_copy(dsT_u, dsT_s)
                    T.dual_copy(ds_u, ds_s)
                if g3:
                    T.gemm(pT_s, doT_s, dv_c, transpose_B=True, clear_accum=(kq == 0))
                    T.gemm(dsT_s, qT_s, dk_c, transpose_B=True, clear_accum=(kq == 0))
                    T.gemm(ds_s, kT_s, dq_c, transpose_B=True, clear_accum=True)
                    T.set_atomic("add", "float32")
                    T.copy(dq_c, dq_acc[m0 : m0 + mq, h, :])
                    T.set_atomic_none()

            # 收尾（保证循环体活：pv 输出 -> GEMM -> dk/dv 写回）
            if not g3:
                T.dual_copy(p16_u, pT_s)
                T.dual_copy(dsT_u, dsT_s)
                T.dual_copy(ds_u, ds_s)
                T.gemm(pT_s, doT_s, dv_c, transpose_B=True, clear_accum=False)
                T.gemm(dsT_s, qT_s, dk_c, transpose_B=True, clear_accum=False)
                T.gemm(ds_s, kT_s, dq_c, transpose_B=True, clear_accum=False)
            T.dual_copy(dk_c, tail_u)
            T.dual_copy(dv_c, tailv_u)
            with T.SimtVF(threads=128):
                for i, d in T.Parallel(hn, DQK):
                    out_k[i, d] = T.cast(tail_u[i, d] * SCALE, DT)
                for i, d in T.Parallel(hn, DV):
                    out_v[i, d] = T.cast(tailv_u[i, d], DT)
            T.copy(out_k, dk[n0 : n0 + hn, h, :])
            T.copy(out_v, dv[n0 : n0 + hn, h, :])

    return main


def build_u2l1_tr(nk=64, mq=64):
    """编译可行性：半区 UB (nk/2,mq) -> L1 (mq,nk/2) 的 `transpose=True` 搬运。"""
    hn = nk // 2

    @T.prim_func
    def main(x: T.Tensor((nk, mq), DT), y: T.Tensor((mq, nk), DT)):  # type: ignore
        with T.Kernel(2) as bid:
            src = T.alloc_shared((hn, mq), DT)
            dst = T.alloc_l1((mq, hn), DT)
            out = T.alloc_shared((mq, hn), DT)
            T.copy(x[bid * hn : bid * hn + hn, :], src)
            T.copy(src, dst, transpose=True)
            T.copy(dst, out)
            T.copy(out, y[:, bid * hn : bid * hn + hn])

    return main


def run(mode, nk, mq, dev="npu:0"):
    try:
        fn = tilelang.compile(build(mode, nk, mq), target="ascend")
    except Exception as exc:  # noqa: BLE001 - compile feasibility is a result
        return f"COMPILE_FAIL {type(exc).__name__}: {str(exc)[:180]}"

    g = torch.Generator().manual_seed(7)

    def rnd(shape, dt=DT):
        t = torch.randn(shape, generator=g, dtype=torch.float32)
        return t.to(torch.bfloat16 if dt == DT else torch.float32).to(dev)

    q = rnd((SQ, H, DQK))
    k = rnd((SKV, H, DQK))
    v = rnd((SKV, H, DV))
    do = rnd((SQ, H, DV))
    lse = (torch.randn((H, SQ), generator=g) * 0.1 - 2.0).to(torch.float32).to(dev)
    delta = (torch.randn((H, SQ), generator=g) * 0.1).to(torch.float32).to(dev)
    dq = torch.zeros((SQ, H, DQK), dtype=torch.float32, device=dev)
    dk = torch.zeros((SKV, H, DQK), dtype=torch.bfloat16, device=dev)
    dv = torch.zeros((SKV, H, DV), dtype=torch.bfloat16, device=dev)
    args = (q, k, v, do, lse, delta, dq, dk, dv)
    try:
        for _ in range(2):
            fn(*args)
        torch.npu.synchronize()
        best = float("inf")
        for _ in range(3):
            t0 = time.perf_counter()
            fn(*args)
            torch.npu.synchronize()
            best = min(best, time.perf_counter() - t0)
    except Exception as exc:  # noqa: BLE001
        return f"RUN_FAIL {type(exc).__name__}: {str(exc)[:180]}"
    return f"{best / R * 1e6:8.3f} us/iter  (R={R}, wall {best * 1e3:8.2f} ms)"


def compile_only(name, builder):
    try:
        tilelang.compile(builder(), target="ascend")
        return "COMPILE_OK"
    except Exception as exc:  # noqa: BLE001
        return f"COMPILE_FAIL {type(exc).__name__}: {str(exc)[:220]}"


MODES = [
    ("mte2", 64, 64),
    ("l0c2ub", 64, 64),
    ("gemm5", 64, 64),
    ("ub2l1", 64, 64),
    ("epi0", 64, 64),
    ("epi1", 64, 64),
    ("epi2", 64, 64),
    ("chain_notr", 64, 64),
    ("chain", 64, 64),
    ("mte2", 128, 48),
    ("gemm5", 128, 48),
    ("chain_notr", 128, 48),
    ("chain", 128, 48),
]

if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else None
    print(f"{'mode':12s} {'nk':>4s} {'mq':>4s}  result", flush=True)
    for mode, nk, mq in MODES:
        if only and only not in mode:
            continue
        print(f"{mode:12s} {nk:4d} {mq:4d}  {run(mode, nk, mq)}", flush=True)
    if not only or "u2l1" in only:
        print(f"{'u2l1_tr':12s} {64:4d} {64:4d}  {compile_only('u2l1_tr', build_u2l1_tr)}", flush=True)
