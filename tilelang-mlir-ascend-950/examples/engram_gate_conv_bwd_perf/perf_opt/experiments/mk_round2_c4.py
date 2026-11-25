"""Generate perf_opt/engram_gate_conv_bwd_round2_c4.py from the r1_c2 base.

Single optimization point (r2_c4 / O-10): fuse the dconv_w partial accumulation
into the conv-recompute chunk sweep, so the four nrm ring taps are loaded once
and the pc read-modify-write happens in the same iteration.
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
OPT = os.path.dirname(HERE)
BASE = os.path.join(OPT, "engram_gate_conv_bwd_round1_c2.py")
OUT = os.path.join(OPT, "engram_gate_conv_bwd_round2_c4.py")

s = open(BASE).read()
i0 = s.index(
    """                # ---- (b) d_silu of row pds = pn (conv recompute + logit deriva.)
                if i >= LEAD:
                    pds = t0 + i - LEAD  # == pn
                    if pds < seq_len:
                        with T.SimdVF():"""
)
i1 = s.index(
    """                # ---- (c) value / gate backward for token tp = t0 + i - 6 ------
"""
)

IN = " " * 36
IH = " " * 36
body = []
body.append(
    """                # ---- (b) d_silu of row pds = pn (conv recompute + logit deriva.)
                # r2_c4 (O-10): the four nrm ring rows consumed by the conv
                # recompute are exactly the four the dconv_w partial accumulates, so
                # the two sweeps are fused -- the taps stay in registers and the pc
                # read-modify-write runs in the same chunk iteration.  One chunk
                # sweep and one VST_VLD barrier per row disappear; only the halo rows
                # (owned by the next run) take the compute-only path.
                if i >= LEAD:
                    pds = t0 + i - LEAD  # == pn
                    if pds < seq_len:
                        if i < rc + LEAD:  # this run owns the row -> accumulate
                            with T.SimdVF():
                                # The row window was vector-stored above (and by the
                                # three previous iterations): vector store -> vector
                                # load needs an explicit barrier.
                                T.simd.mem_bar("VST_VLD")
                                full = T.simd.pset(32)
                                one = T.simd.vdup(T.float32(1), FP32, full)
                                for ch in T.serial(nchunks):
                                    mask = T.simd.update_mask(T.min(VL, d - ch * VL), 32)
"""
)
for p in range(4):
    body.append(f"{IN}n{p} = T.simd.alloc_local((1,), FP32)\n")
for p in range(4):
    body.append(
        f"{IN}n{p}[0] = T.simd.vld(nrmw[T.floormod(i - LEAD + {p}, KS), ch * VL])\n"
    )
body.append(
    f"""{IN}conv = T.simd.alloc_local((1,), FP32)
{IN}conv[0] = T.simd.vdup(T.float32(0), FP32, full)
"""
)
for p in range(4):
    body.append(
        f"""{IN}conv[0] = T.simd.vadd(
{IN}    conv[0],
{IN}    T.simd.vmul(n{p}[0], T.simd.vld(cwf[{p}, ch * VL]), mask),
{IN}    full,
{IN})
"""
    )
body.append(
    f"""{IN}sig = T.simd.vdiv(
{IN}    one,
{IN}    T.simd.vadd(
{IN}        one,
{IN}        T.simd.vexp(T.simd.vneg(conv[0], mask), mask),
{IN}        mask,
{IN}    ),
{IN}    mask,
{IN})
{IN}deriv = T.simd.vadd(
{IN}    sig,
{IN}    T.simd.vmul(
{IN}        T.simd.vmul(conv[0], sig, mask),
{IN}        T.simd.vsub(one, sig, mask),
{IN}        mask,
{IN}    ),
{IN}    mask,
{IN})
{IN}ds_v = T.simd.alloc_local((1,), FP32)
{IN}ds_v[0] = T.simd.vmul(T.simd.vld(dyw[slot, ch * VL]), deriv, mask)
{IN}T.simd.vsts(dsw[slot, ch * VL], ds_v[0], full)
"""
)
for p in range(4):
    body.append(
        f"""{IN}T.simd.vsts(
{IN}    pc[{p}, ch * VL],
{IN}    T.simd.vadd(
{IN}        T.simd.vld(pc[{p}, ch * VL]),
{IN}        T.simd.vmul(ds_v[0], n{p}[0], mask),
{IN}        full,
{IN}    ),
{IN}    full,
{IN})
"""
    )
body.append(
    """                    if i >= rc + LEAD:  # halo row owned by the next run
                        if pds < seq_len:
                            with T.SimdVF():
                                T.simd.mem_bar("VST_VLD")
                                full = T.simd.pset(32)
                                one = T.simd.vdup(T.float32(1), FP32, full)
                                for ch in T.serial(nchunks):
                                    mask = T.simd.update_mask(T.min(VL, d - ch * VL), 32)
"""
)
body.append(
    f"""{IH}conv_h = T.simd.alloc_local((1,), FP32)
{IH}conv_h[0] = T.simd.vdup(T.float32(0), FP32, full)
"""
)
for p in range(4):
    body.append(
        f"""{IH}conv_h[0] = T.simd.vadd(
{IH}    conv_h[0],
{IH}    T.simd.vmul(
{IH}        T.simd.vld(nrmw[T.floormod(i - LEAD + {p}, KS), ch * VL]),
{IH}        T.simd.vld(cwf[{p}, ch * VL]),
{IH}        mask,
{IH}    ),
{IH}    full,
{IH})
"""
    )
body.append(
    f"""{IH}sig_h = T.simd.vdiv(
{IH}    one,
{IH}    T.simd.vadd(
{IH}        one,
{IH}        T.simd.vexp(T.simd.vneg(conv_h[0], mask), mask),
{IH}        mask,
{IH}    ),
{IH}    mask,
{IH})
{IH}deriv_h = T.simd.vadd(
{IH}    sig_h,
{IH}    T.simd.vmul(
{IH}        T.simd.vmul(conv_h[0], sig_h, mask),
{IH}        T.simd.vsub(one, sig_h, mask),
{IH}        mask,
{IH}    ),
{IH}    mask,
{IH})
{IH}T.simd.vsts(
{IH}    dsw[slot, ch * VL],
{IH}    T.simd.vmul(T.simd.vld(dyw[slot, ch * VL]), deriv_h, mask),
{IH}    full,
{IH})
"""
)
body.append(
    """                    if pds >= seq_len:  # halo past the tail: zero slot, no compute
                        with T.SimdVF():
                            full = T.simd.pset(32)
                            zero = T.simd.vdup(T.float32(0), FP32, full)
                            for ch in T.serial(nchunks):
                                T.simd.vsts(dsw[slot, ch * VL], zero, full)

"""
)

out = s[:i0] + "".join(body) + s[i1:]
with open(OUT, "w") as f:
    f.write(out)
print(f"wrote {OUT}: {len(out.splitlines())} lines")
