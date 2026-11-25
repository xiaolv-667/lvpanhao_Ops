"""O-9: hoist chunk-invariant per-token vector expressions out of the chunk sweeps.

Applied to a stage-4 variant file in place (used for r2_c3 and for the merged
r3_c6 candidate):
  * pass B:  ``coef = rrms_v^3 * (1/d)``   (one per token, not one per chunk lane)
  * pass C2: ``rh3d`` / ``rk3d``            (one per token, not one per chunk lane)

    python mk_hoist_invariants.py <variant.py>
"""

import sys


def rep(s, old, new, tag):
    n = s.count(old)
    assert n == 1, (tag, n)
    return s.replace(old, new)


def main():
    path = sys.argv[1]
    s = open(path).read()

    s = rep(
        s,
        """                            da = T.simd.alloc_local((1,), FP32)
                            da[0] = T.simd.vdup(T.float32(0), FP32, full)
                            for ch in T.serial(nchunks):
                                mask = T.simd.update_mask(T.min(VL, d - ch * VL), 32)
                                z = T.simd.vld(zraw[T.floormod(i - LEAD, KS), ch * VL])
                                # O2 + O1: (rrms_v^3 * (1/d)) folded per chunk lane
                                coef = T.simd.vmuls(
                                    T.simd.vmul(T.simd.vmul(rv, rv, full), rv, full),
                                    inv_d,
                                    full,
                                )
""",
        """                            da = T.simd.alloc_local((1,), FP32)
                            da[0] = T.simd.vdup(T.float32(0), FP32, full)
                            # O-9: (rrms_v^3 * (1/d)) is one per token, not one per
                            # chunk lane: hoisted into a register before the sweep.
                            coef = T.simd.alloc_local((1,), FP32)
                            coef[0] = T.simd.vmuls(
                                T.simd.vmul(T.simd.vmul(rv, rv, full), rv, full),
                                inv_d,
                                full,
                            )
                            for ch in T.serial(nchunks):
                                mask = T.simd.update_mask(T.min(VL, d - ch * VL), 32)
                                z = T.simd.vld(zraw[T.floormod(i - LEAD, KS), ch * VL])
""",
        "coef",
    )
    s = rep(
        s,
        """                                    T.simd.vmul(coef, z, mask), dot_b, mask""",
        """                                    T.simd.vmul(coef[0], z, mask), dot_b, mask""",
        "coef_use",
    )

    s = rep(
        s,
        """                            for ch in T.serial(nchunks):
                                mask = T.simd.update_mask(T.min(VL, d - ch * VL), 32)
                                hf = T.simd.vcvt(
                                    T.simd.vld(hstg[widx_c, ch * VL], dist="UNPK_B16"),
                                    FP32,
                                    mask,
                                    part=0,
                                )
                                kf = T.simd.vcvt(
                                    T.simd.vld(kstg[widx_c, ch * VL], dist="UNPK_B16"),
                                    FP32,
                                    mask,
                                    part=0,
                                )
                                w = T.simd.vld(whf[ch * VL])
                                hn = T.simd.vld(hnf[ch * VL])
                                kn = T.simd.vld(knf[ch * VL])
                                # O2 + O1: rrms^3 * (1/d) folded per chunk lane
                                rh3d = T.simd.vmuls(
                                    T.simd.vmul(T.simd.vmul(rh, rh, full), rh, full),
                                    inv_d,
                                    full,
                                )
                                rk3d = T.simd.vmuls(
                                    T.simd.vmul(T.simd.vmul(rk, rk, full), rk, full),
                                    inv_d,
                                    full,
                                )
""",
        """                            # O-9: rrms^3 * (1/d) is one per token, hoisted out of
                            # the chunk sweep into registers.
                            rh3d = T.simd.alloc_local((1,), FP32)
                            rh3d[0] = T.simd.vmuls(
                                T.simd.vmul(T.simd.vmul(rh, rh, full), rh, full),
                                inv_d,
                                full,
                            )
                            rk3d = T.simd.alloc_local((1,), FP32)
                            rk3d[0] = T.simd.vmuls(
                                T.simd.vmul(T.simd.vmul(rk, rk, full), rk, full),
                                inv_d,
                                full,
                            )
                            for ch in T.serial(nchunks):
                                mask = T.simd.update_mask(T.min(VL, d - ch * VL), 32)
                                hf = T.simd.vcvt(
                                    T.simd.vld(hstg[widx_c, ch * VL], dist="UNPK_B16"),
                                    FP32,
                                    mask,
                                    part=0,
                                )
                                kf = T.simd.vcvt(
                                    T.simd.vld(kstg[widx_c, ch * VL], dist="UNPK_B16"),
                                    FP32,
                                    mask,
                                    part=0,
                                )
                                w = T.simd.vld(whf[ch * VL])
                                hn = T.simd.vld(hnf[ch * VL])
                                kn = T.simd.vld(knf[ch * VL])
""",
        "rh3d",
    )
    s = rep(
        s,
        """                                        T.simd.vmul(rh3d, hf, mask), dhb, mask""",
        """                                        T.simd.vmul(rh3d[0], hf, mask), dhb, mask""",
        "rh3d_use",
    )
    s = rep(
        s,
        """                                        T.simd.vmul(rk3d, kf, mask), dkb, mask""",
        """                                        T.simd.vmul(rk3d[0], kf, mask), dkb, mask""",
        "rk3d_use",
    )
    with open(path, "w") as f:
        f.write(s)
    print(f"hoisted chunk-invariant expressions in {path}")


if __name__ == "__main__":
    main()
