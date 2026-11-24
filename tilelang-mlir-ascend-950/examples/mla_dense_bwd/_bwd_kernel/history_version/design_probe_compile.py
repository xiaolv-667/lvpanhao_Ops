"""Stage 1 设计期编译探针（revision v1）：证明 §3.3 的修正形态可编译。

无设备（仅前端 + 编译）：`ASCEND_RT_VISIBLE_DEVICES=""`。
必须在**仓根之外**的目录运行（仓根 `tilelang/` 是 dev root，会遮蔽 site-packages
的 0.1.15 私有 TVM）：

    cd /tmp && ASCEND_RT_VISIBLE_DEVICES="" \
        /workspace/venv-ascend950/bin/python \
        /workspace/tilelang-mlir-ascend-950/examples/mla_dense_bwd/_bwd_kernel/history_version/design_probe_compile.py

探针清单（对应 DESIGN.md §3.5.4）：
  A  `_bwd_kernel_primary`  — 5 个全 extent NT GEMM + 整 tile L0C→UB + SimtVF(128)
      + UB→L1 + store-atomic + 块尾 cast/写回（**不含** T.Vector / sid）。COMPILE_OK。
  B  `_bwd_kernel_tail_oob` — S 非整除（96）时全 extent 装载/写回的越界尾部，
      检查生成码中的 clamp + fill，以及动态 `clear_accum=(kq == loop_st)`。
  C  `_bwd_kernel_a2_branch` — A2 块级特化的**运行时**分支（两支各一套 epilogue）。
"""

import tilelang
import tilelang.ascend.language as T

# --------------------------------------------------------------------------
# A: primary 形态（block 64x64，S = 128 整除）
# --------------------------------------------------------------------------
B, H, SQ, SKV = 1, 2, 128, 128
DQK, DV = 192, 128
BN, BM = 64, 64
DT = "float16"
SCALE = DQK**-0.5
NTP = (SKV + BN - 1) // BN


def build_primary():
    @T.prim_func
    def main(
        q: T.Tensor((B * SQ, H, DQK), DT),
        k: T.Tensor((B * SKV, H, DQK), DT),
        v: T.Tensor((B * SKV, H, DV), DT),
        do: T.Tensor((B * SQ, H, DV), DT),
        lse: T.Tensor((B, H, SQ), "float32"),
        delta: T.Tensor((B, H, SQ), "float32"),
        dq_acc: T.Tensor((B * SQ, H, DQK), "float32"),
        dk: T.Tensor((B * SKV, H, DQK), DT),
        dv: T.Tensor((B * SKV, H, DV), DT),
    ):
        with T.Kernel(H * B * NTP) as bid:
            h = bid // (B * NTP)
            tile = bid % (B * NTP)
            b = tile // NTP
            n0 = (tile % NTP) * BN
            k_s = T.alloc_l1((BN, DQK), DT)
            v_s = T.alloc_l1((BN, DV), DT)
            kT_s = T.alloc_l1((DQK, BN), DT)
            q_s = T.alloc_l1((BM, DQK), DT)
            qT_s = T.alloc_l1((DQK, BM), DT)
            do_s = T.alloc_l1((BM, DV), DT)
            doT_s = T.alloc_l1((DV, BM), DT)
            pT_s = T.alloc_l1((BN, BM), DT)
            dsT_s = T.alloc_l1((BN, BM), DT)
            ds_s = T.alloc_l1((BM, BN), DT)
            st_c = T.alloc_l0c((BN, BM), "float32")
            dpt_c = T.alloc_l0c((BN, BM), "float32")
            dq_c = T.alloc_l0c((BM, DQK), "float32")
            dk_c = T.alloc_l0c((BN, DQK), "float32")
            dv_c = T.alloc_l0c((BN, DV), "float32")
            st_u = T.alloc_shared((BN, BM), "float32")
            dp_u = T.alloc_shared((BN, BM), "float32")
            p16_u = T.alloc_shared((BN, BM), DT)
            dsT_u = T.alloc_shared((BN, BM), DT)
            ds_u = T.alloc_shared((BM, BN), DT)
            lse_u = T.alloc_shared((BM,), "float32")
            dlt_u = T.alloc_shared((BM,), "float32")
            tail_u = T.alloc_shared((BN, DQK), "float32")
            tailv_u = T.alloc_shared((BN, DV), "float32")
            out_k = T.alloc_shared((BN, DQK), DT)
            out_v = T.alloc_shared((BN, DV), DT)

            T.copy(k[b * SKV + n0 : b * SKV + n0 + BN, h, :], k_s)
            T.copy(v[b * SKV + n0 : b * SKV + n0 + BN, h, :], v_s)
            T.copy(k[b * SKV + n0 : b * SKV + n0 + BN, h, :], kT_s, transpose=True)

            shift = SKV - SQ
            loop_st = T.max(0, (n0 - shift) // BM)
            for kq in T.Pipelined(loop_st, (SQ + BM - 1) // BM, num_stages=1):
                m0 = kq * BM
                T.copy(q[b * SQ + m0 : b * SQ + m0 + BM, h, :], q_s)
                T.copy(do[b * SQ + m0 : b * SQ + m0 + BM, h, :], do_s)
                T.copy(q[b * SQ + m0 : b * SQ + m0 + BM, h, :], qT_s, transpose=True)
                T.copy(do[b * SQ + m0 : b * SQ + m0 + BM, h, :], doT_s, transpose=True)
                T.copy(lse[b, h, m0 : m0 + BM], lse_u)
                T.copy(delta[b, h, m0 : m0 + BM], dlt_u)

                T.gemm(k_s, q_s, st_c, transpose_B=True, clear_accum=True)
                T.gemm(v_s, do_s, dpt_c, transpose_B=True, clear_accum=True)
                T.copy(st_c, st_u)
                T.copy(dpt_c, dp_u)
                with T.SimtVF(threads=128):
                    for i, j in T.Parallel(BN, BM):
                        gi = n0 + i
                        gj = m0 + j
                        vis = (gi < SKV) & (gj < SQ) & (gi <= gj + shift)
                        p = T.exp(st_u[i, j] * SCALE - lse_u[j])
                        pv = T.if_then_else(vis, p, T.cast(0, "float32"))
                        p16_u[i, j] = T.cast(pv, DT)
                        dsv = pv * (dp_u[i, j] - dlt_u[j])
                        dsT_u[i, j] = T.cast(dsv, DT)
                        ds_u[j, i] = T.cast(dsv, DT)
                T.copy(p16_u, pT_s)
                T.copy(dsT_u, dsT_s)
                T.copy(ds_u, ds_s)
                T.gemm(pT_s, doT_s, dv_c, transpose_B=True, clear_accum=(kq == loop_st))
                T.gemm(dsT_s, qT_s, dk_c, transpose_B=True, clear_accum=(kq == loop_st))
                T.gemm(ds_s, kT_s, dq_c, transpose_B=True, clear_accum=True)
                T.set_atomic("add", "float32")
                T.copy(dq_c, dq_acc[b * SQ + m0 : b * SQ + m0 + BM, h, :])
                T.set_atomic_none()

            T.copy(dk_c, tail_u)
            T.copy(dv_c, tailv_u)
            with T.SimtVF(threads=128):
                for i, d in T.Parallel(BN, DQK):
                    out_k[i, d] = T.cast(tail_u[i, d] * SCALE, DT)
                for i, d in T.Parallel(BN, DV):
                    out_v[i, d] = T.cast(tailv_u[i, d], DT)
            T.copy(out_k, dk[b * SKV + n0 : b * SKV + n0 + BN, h, :])
            T.copy(out_v, dv[b * SKV + n0 : b * SKV + n0 + BN, h, :])

    return main


# --------------------------------------------------------------------------
# B: 非整除尾部（S = 96）+ 动态 clear_accum
# --------------------------------------------------------------------------
SB = 96
NTP_B = (SB + BN - 1) // BN


def build_tail_oob():
    @T.prim_func
    def main(
        q: T.Tensor((B * SB, H, DQK), DT),
        k: T.Tensor((B * SB, H, DQK), DT),
        v: T.Tensor((B * SB, H, DV), DT),
        lse: T.Tensor((B, H, SB), "float32"),
        dk: T.Tensor((B * SB, H, DQK), DT),
    ):
        with T.Kernel(H * B * NTP_B) as bid:
            h = bid // (B * NTP_B)
            tile = bid % (B * NTP_B)
            b = tile // NTP_B
            n0 = (tile % NTP_B) * BN
            k_s = T.alloc_l1((BN, DQK), DT)
            v_s = T.alloc_l1((BN, DV), DT)
            q_s = T.alloc_l1((BM, DQK), DT)
            qT_s = T.alloc_l1((DQK, BM), DT)
            dsT_s = T.alloc_l1((BN, BM), DT)
            dk_c = T.alloc_l0c((BN, DQK), "float32")
            tail_u = T.alloc_shared((BN, DQK), "float32")
            out_k = T.alloc_shared((BN, DQK), DT)
            T.copy(k[b * SB + n0 : b * SB + n0 + BN, h, :], k_s)
            T.copy(v[b * SB + n0 : b * SB + n0 + BN, h, :], v_s)
            loop_st = T.max(0, (n0 - 0) // BM)
            for kq in T.Pipelined(loop_st, (SB + BM - 1) // BM, num_stages=1):
                m0 = kq * BM
                T.copy(q[b * SB + m0 : b * SB + m0 + BM, h, :], q_s)
                T.copy(q[b * SB + m0 : b * SB + m0 + BM, h, :], qT_s, transpose=True)
                T.gemm(dsT_s, qT_s, dk_c, transpose_B=True, clear_accum=(kq == loop_st))
            T.copy(dk_c, tail_u)
            with T.SimtVF(threads=128):
                for i, d in T.Parallel(BN, DQK):
                    out_k[i, d] = T.cast(tail_u[i, d] * SCALE, DT)
            T.copy(out_k, dk[b * SB + n0 : b * SB + n0 + BN, h, :])

    return main


# --------------------------------------------------------------------------
# C: A2 运行时块级分支（两支各一套整 tile epilogue）
# --------------------------------------------------------------------------
def build_a2_branch():
    @T.prim_func
    def main(
        k: T.Tensor((B * SKV, H, DQK), DT),
        lse: T.Tensor((B, H, SQ), "float32"),
        dk: T.Tensor((B * SKV, H, DQK), DT),
    ):
        with T.Kernel(H * B * NTP) as bid:
            h = bid // (B * NTP)
            tile = bid % (B * NTP)
            b = tile // NTP
            n0 = (tile % NTP) * BN
            st_u = T.alloc_shared((BN, BM), "float32")
            dp_u = T.alloc_shared((BN, BM), "float32")
            p16_u = T.alloc_shared((BN, BM), DT)
            dsT_u = T.alloc_shared((BN, BM), DT)
            ds_u = T.alloc_shared((BM, BN), DT)
            lse_u = T.alloc_shared((BM,), "float32")
            dlt_u = T.alloc_shared((BM,), "float32")
            out_k = T.alloc_shared((BN, DQK), DT)
            shift = SKV - SQ
            loop_st = T.max(0, (n0 - shift) // BM)
            for kq in T.Pipelined(loop_st, (SQ + BM - 1) // BM, num_stages=1):
                m0 = kq * BM
                T.copy(lse[b, h, m0 : m0 + BM], lse_u)
                T.copy(lse[b, h, m0 : m0 + BM], dlt_u)
                if (n0 + BN <= SKV) & (m0 + BM <= SQ) & (n0 + BN - 1 <= m0 + shift):
                    with T.SimtVF(threads=128):
                        for i, j in T.Parallel(BN, BM):
                            p = T.exp(st_u[i, j] * SCALE - lse_u[j])
                            p16_u[i, j] = T.cast(p, DT)
                            dsv = p * (dp_u[i, j] - dlt_u[j])
                            dsT_u[i, j] = T.cast(dsv, DT)
                            ds_u[j, i] = T.cast(dsv, DT)
                else:
                    with T.SimtVF(threads=128):
                        for i, j in T.Parallel(BN, BM):
                            gi = n0 + i
                            gj = m0 + j
                            vis = (gi < SKV) & (gj < SQ) & (gi <= gj + shift)
                            p = T.exp(st_u[i, j] * SCALE - lse_u[j])
                            pv = T.if_then_else(vis, p, T.cast(0, "float32"))
                            p16_u[i, j] = T.cast(pv, DT)
                            dsv = pv * (dp_u[i, j] - dlt_u[j])
                            dsT_u[i, j] = T.cast(dsv, DT)
                            ds_u[j, i] = T.cast(dsv, DT)
            with T.SimtVF(threads=128):
                for i, d in T.Parallel(BN, DQK):
                    out_k[i, d] = T.cast(out_k[i, d], DT)
            T.copy(out_k, dk[b * SKV + n0 : b * SKV + n0 + BN, h, :])

    return main


def run(name, builder, dump=None):
    try:
        kern = tilelang.compile(builder(), target="ascend")
        src = kern.get_kernel_source()
        print(f"COMPILE_OK {name} source_bytes={len(src)} "
              f"gemm_l1={src.count('ascend_gemm_l1')} "
              f"l0c2ub={src.count('asc_copy_l0c2ub')} "
              f"nd2nz_scatter={src.count('ascend_nd2nz_scatter')}")
        if dump:
            with open(dump, "w", encoding="utf-8") as fh:
                fh.write(src)
    except Exception as exc:  # noqa: BLE001
        print(f"COMPILE_FAIL {name} {type(exc).__name__}: "
              f"{str(exc)[:600]}".replace("\n", " "))


if __name__ == "__main__":
    run("A_primary", build_primary, "/tmp/opencode/probe_A.txt")
    run("B_tail_oob", build_tail_oob, "/tmp/opencode/probe_B.txt")
    run("C_a2_branch", build_a2_branch, "/tmp/opencode/probe_C.txt")
