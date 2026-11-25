"""Launch-cost floor probe for the EngramGateConvFwd geometry (Stage 4 diagnostic).

Measures the platform `Task Duration(us)` of kernels that keep the *same launch
geometry* as the tuned operator (grid = 48 blocks, d = 768, bf16) but do
strictly less work, so the fixed per-launch / per-block cost of this operator
can be separated from its compute cost.

Variants (single launch, same 12-tensor signature / out_idx as the operator):
  kind=0  grid=48, one 1536 B row load + two 1536 B row stores per block
  kind=1  grid=48, 29 row loads (the real kernel issues 27 + the prologue ones)
  kind=2  kind=1 + the 4 fp32 row-statistic stores inside a SimdVF block
  kind=3  grid=1,  one row load + two stores   (pure single-block launch cost)

Driven by `bench_one.py` (same msopprof protocol) through the operator-compatible
factory signature `_engram_gate_conv_fwd_kernel(M, seq_len, d, eps, dtype)`; the
variant is selected with the FLOOR_KIND environment variable.

run: FLOOR_KIND=1 python bench_one.py --module floor_probe.py --workload s96
"""

import os

import tilelang
import tilelang.ascend.language as T

M, S, D, RC, NCH = 3, 96, 768, 6, 16
GRID = M * NCH
DT = "bfloat16"
FP32 = "float32"

_KIND = int(os.environ.get("FLOOR_KIND", "0"))


def _build(kind: int, grid: int):
    rows = 9

    @T.prim_func
    def main(
        H: T.Tensor((M, S, D), DT),
        K: T.Tensor((M, S, D), DT),
        V: T.Tensor((M, S, D), DT),
        WH: T.Tensor((D,), DT),
        WV: T.Tensor((D,), DT),
        CW: T.Tensor((4, D), DT),
        Y: T.Tensor((M, S, D), DT),
        VHAT: T.Tensor((M, S, D), DT),
        Alpha: T.Tensor((M, S), FP32),
        RH: T.Tensor((M, S), FP32),
        RK: T.Tensor((M, S), FP32),
        RV: T.Tensor((M, S), FP32),
    ):
        with T.Kernel(grid) as bx:
            b = bx // NCH if grid > 1 else 0
            c = bx % NCH if grid > 1 else 0
            t0 = c * RC
            h_ub = T.alloc_shared((D,), DT)
            k_ub = T.alloc_shared((D,), DT)
            v_ub = T.alloc_shared((D,), DT)
            y_ub = T.alloc_shared((D,), DT)
            st = T.alloc_shared((4, 8), FP32)
            h_win = T.alloc_shared((RC + 3, D), DT)
            k_win = T.alloc_shared((RC + 3, D), DT)
            v_win = T.alloc_shared((RC + 3, D), DT)
            T.annotate_buffer_versions(
                {h_ub: 1, k_ub: 1, v_ub: 1, y_ub: 1, st: 1, h_win: 1, k_win: 1, v_win: 1}
            )
            T.copy(H[b, t0, :], h_ub[:D])
            if kind == 1:
                for i in T.serial(rows):
                    T.copy(K[b, t0, :], k_ub[:D])
                    T.copy(V[b, t0, :], v_ub[:D])
                    T.copy(H[b, t0, :], h_ub[:D])
            if kind >= 4:
                # the hoisted-window structure: one (rc+3, d) copy per tensor, or
                # the owned/halo split the causal guard needs (6 copies)
                T.copy(H[b, t0 : t0 + RC, :], h_win[3 : 3 + RC, :D])
                T.copy(K[b, t0 : t0 + RC, :], k_win[3 : 3 + RC, :D])
                T.copy(V[b, t0 : t0 + RC, :], v_win[3 : 3 + RC, :D])
                if kind == 5:
                    if t0 >= 3:
                        T.copy(H[b, t0 - 3 : t0, :], h_win[0:3, :D])
                        T.copy(K[b, t0 - 3 : t0, :], k_win[0:3, :D])
                        T.copy(V[b, t0 - 3 : t0, :], v_win[0:3, :D])
            if kind >= 2:
                with T.SimdVF():
                    full = T.simd.pset(32)
                    one = T.simd.pset(32, "PAT_VL1")
                    for m in T.serial(4):
                        T.simd.vsts(st[m, 0], T.simd.vdup(T.float32(1), FP32, full), one, extent=1)
            T.copy(h_ub[:D], Y[b, t0, :])
            T.copy(y_ub[:D], VHAT[b, t0, :])
            if kind >= 2:
                for m, out in enumerate((Alpha, RH, RK, RV)):
                    T.copy(st[m, 0:1], out[b, t0 : t0 + 1])

    return main


def _engram_gate_conv_fwd_kernel(M_, seq_len, d, eps, dtype):
    """Operator-compatible factory for the probe (kind from FLOOR_KIND)."""
    grid = 1 if _KIND == 3 else GRID
    return tilelang.compile(_build(_KIND, grid), target="ascend", out_idx=[6, 7, 8, 9, 10, 11])
