"""Generate perf_opt/engram_gate_conv_bwd_round1_n3.py from the round-0 best.

Single optimization point (n3 / O-15), launch #2 only: the ordered reduction
sweeps the core rows *inside* the column-chunk loop, so a block with N column
chunks runs N dependent ``vadd`` chains one after the other.  Hoisting the
column loop out gives N independent chains that interleave (ILP = N), which is
what the reduction launch is limited by at d=768 (2 chunks -> 48 serial adds
twice).

The per-column summation order (ascending core id, one accumulator per column) is
untouched, so the three weight gradients stay bit-identical.

Shapes: ``width`` is a multiple of ``VL`` and <= 128 for every supported ``d``
(<= 1024), so the specialised paths are ``nch == 1`` and ``nch == 2``; a generic
serial path is kept for anything wider.
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
OPT = os.path.dirname(HERE)
BASE = os.path.join(OPT, "engram_gate_conv_bwd_round0_best.py")
OUT = os.path.join(OPT, "engram_gate_conv_bwd_round1_n3.py")

s = open(BASE).read()

i0 = s.index("""            with T.SimdVF():
                full = T.simd.pset(32)
                T.simd.mem_bar("VST_VLD")
                for ch in T.serial(width // VL):""")
i1 = s.index("""            if g == 0:
                T.copy(tot[:cols], DWH[off : off + cols])""")

new = '''            nch_r = width // VL
            with T.SimdVF():
                full = T.simd.pset(32)
                T.simd.mem_bar("VST_VLD")
                # O-15: one accumulator per column chunk, row loop outermost, so the
                # chains of a multi-chunk block interleave (ILP = nch_r).  Column
                # order per output element is unchanged (ascending core id).
                if nch_r == 1:
                    valid0 = T.simd.update_mask(T.min(VL, T.max(0, cols)), 32)
                    run0 = T.simd.alloc_local((1,), FP32)
                    run0[0] = T.simd.vdup(T.float32(0), FP32, full)
                    for row in T.serial(cores):
                        run0[0] = T.simd.vadd(
                            run0[0], T.simd.vld(part[row, 0]), valid0
                        )
                    T.simd.vsts(tot[0], run0[0], full)
                if nch_r == 2:
                    valid0 = T.simd.update_mask(T.min(VL, T.max(0, cols)), 32)
                    valid1 = T.simd.update_mask(T.min(VL, T.max(0, cols - VL)), 32)
                    run0 = T.simd.alloc_local((1,), FP32)
                    run1 = T.simd.alloc_local((1,), FP32)
                    run0[0] = T.simd.vdup(T.float32(0), FP32, full)
                    run1[0] = T.simd.vdup(T.float32(0), FP32, full)
                    for row in T.serial(cores):
                        run0[0] = T.simd.vadd(
                            run0[0], T.simd.vld(part[row, 0]), valid0
                        )
                        run1[0] = T.simd.vadd(
                            run1[0], T.simd.vld(part[row, VL]), valid1
                        )
                    T.simd.vsts(tot[0], run0[0], full)
                    T.simd.vsts(tot[VL], run1[0], full)
                if nch_r > 2:  # generic fallback (not reached for d <= 1024)
                    for ch in T.serial(nch_r):
                        col = ch * VL
                        valid = T.simd.update_mask(T.min(VL, T.max(0, cols - col)), 32)
                        run = T.simd.alloc_local((1,), FP32)
                        run[0] = T.simd.vdup(T.float32(0), FP32, full)
                        for row in T.serial(cores):
                            run[0] = T.simd.vadd(
                                run[0], T.simd.vld(part[row, col]), valid
                            )
                        T.simd.vsts(tot[col], run[0], full)

'''

s = s[:i0] + new + s[i1:]
with open(OUT, "w") as f:
    f.write(s)
print(f"wrote {OUT}")
