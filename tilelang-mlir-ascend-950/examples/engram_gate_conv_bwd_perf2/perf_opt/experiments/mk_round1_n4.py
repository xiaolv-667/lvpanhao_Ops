"""Generate perf_opt/engram_gate_conv_bwd_round1_n4.py from the round-0 best.

Single optimization point (n4 / O-14): use the vector unit's fused instruction
forms instead of the separate op chains they replace, with no change in
arithmetic (the rounding sequence is identical, so the six outputs stay
bit-identical):

  * ``vexpdif(zero, conv)`` = ``exp(0 - conv)``  -- one instruction instead of
    ``vneg`` + ``vexp`` for the sigmoid argument (``0 - x == -x`` exactly);
  * the conv chain starts at the first tap instead of ``0 + t0``
    (``conv[0] = vdup(0)`` + ``vadd`` disappear; ``0 + a == a`` exactly);
  * ``vaxpy``  (``dst = src * scalar + dst``) replaces the two
    ``ph += (uh*hf)*rh`` / ``ph += (uk*kf)*rk`` mul/add pairs in pass C1.
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
OPT = os.path.dirname(HERE)
BASE = os.path.join(OPT, "engram_gate_conv_bwd_round0_best.py")
OUT = os.path.join(OPT, "engram_gate_conv_bwd_round1_n4.py")

s = open(BASE).read()


def rep(buf, old, new, tag, count=1):
    global s
    assert buf is s
    n = s.count(old)
    assert n == count, (tag, n)
    s = s.replace(old, new)
    print("  patched:", tag)


# ------------------------------------------------------------------ (b): exp
rep(
    s,
    """                            full = T.simd.pset(32)
                            one = T.simd.vdup(T.float32(1), FP32, full)
                            for ch in T.serial(nchunks):""",
    """                            full = T.simd.pset(32)
                            one = T.simd.vdup(T.float32(1), FP32, full)
                            zero = T.simd.vdup(T.float32(0), FP32, full)
                            for ch in T.serial(nchunks):""",
    "b_zero",
)
rep(
    s,
    """                                sig = T.simd.vdiv(
                                    one,
                                    T.simd.vadd(
                                        one,
                                        T.simd.vexp(T.simd.vneg(conv[0], mask), mask),
                                        mask,
                                    ),
                                    mask,
                                )""",
    """                                # O-14: exp(0 - conv) in one instruction (0 - x == -x)
                                sig = T.simd.vdiv(
                                    one,
                                    T.simd.vadd(
                                        one,
                                        T.simd.vexpdif(zero, conv[0], mask),
                                        mask,
                                    ),
                                    mask,
                                )""",
    "b_expdif",
)

# ------------------------------------------------- (b): conv starts at tap 0
rep(
    s,
    """                                conv = T.simd.alloc_local((1,), FP32)
                                conv[0] = T.simd.vdup(T.float32(0), FP32, full)
                                for p in T.unroll(KS):
                                    conv[0] = T.simd.vadd(
                                        conv[0],
                                        T.simd.vmul(
                                            T.simd.vld(
                                                nrm_all[i - LEAD + p, ch * VL]
                                            ),
                                            T.simd.vld(cwf[p, ch * VL]),
                                            mask,
                                        ),
                                        full,
                                    )""",
    """                                # O-14: start the chain at tap 0 (0 + a == a)
                                conv = T.simd.alloc_local((1,), FP32)
                                conv[0] = T.simd.vmul(
                                    T.simd.vld(nrm_all[i - LEAD, ch * VL]),
                                    T.simd.vld(cwf[0, ch * VL]),
                                    mask,
                                )
                                for p in T.unroll(KS - 1):
                                    conv[0] = T.simd.vadd(
                                        conv[0],
                                        T.simd.vmul(
                                            T.simd.vld(
                                                nrm_all[i - LEAD + p + 1, ch * VL]
                                            ),
                                            T.simd.vld(cwf[p + 1, ch * VL]),
                                            mask,
                                        ),
                                        full,
                                    )""",
    "b_conv_start",
)

# NOTE: the ``vaxpy`` (dst = src*scalar + dst) form for the ph partial was tried
# here too and rejected: the device compilation fails for it in this kernel
# ("Ascend device compilation failed"), so O-14 keeps only the two pure
# instruction substitutions below.

with open(OUT, "w") as f:
    f.write(s)
print(f"wrote {OUT}")
