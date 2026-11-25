"""Build round-3 branch c7: the DESIGN.md 1.6.3 two-launch alternative (备选).

DESIGN.md 1.6.3 freezes the single-launch + halo-recompute structure (主选) and
asks for an A/B against the two-segment launch (备选), whose only structural
advantage is that it needs no halo recompute: launch A materialises ``vhat`` and
the fp32 row statistics, launch B reads four neighbour rows of ``vhat`` from GM.
The design's own flip threshold is ``alt < main * 0.9``.

This branch implements that alternative in its best measured form (the lessons of
round 1/2 applied to it as well -- bulk 2-D copies instead of per-row DMAs, 2-D
output stores, folded conv weights):

    launch A  grid = M * nch, rc owned rows per block, no halo:
              one (rc, d) copy per input tensor, gate statistics, vhat + the
              four fp32 row scalars out.
    launch B  grid = M * nch, rc owned rows per block:
              one (rc + 3, d) bulk copy of the vhat window (causal zero fill at
              the sequence start), one (rc, d) copy of v, the rv/alpha row
              scalars, then per output row 4 taps x (mul r_v, mul cwv, add).

    python mk_round3_c7.py
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "engram_gate_conv_round2_c5.py")
DST = os.path.join(os.path.dirname(HERE), "engram_gate_conv_round3_c7.py")

START = "@functools.lru_cache(maxsize=64)\n"
END = '    return tilelang.compile(main, target="ascend", out_idx=[6, 7, 8, 9, 10, 11])\n'

NEW = '''@functools.lru_cache(maxsize=64)
def _engram_gate_conv_fwd_kernel(M: int, seq_len: int, d: int, eps: float, dtype: str):
    """Two-launch alternative (DESIGN.md 1.6.3 备选): gate launch + conv launch."""
    if dtype not in ELEM_BYTES:
        raise ValueError(f"dtype must be float16 or bfloat16, got {dtype!r}")
    if M <= 0 or seq_len <= 0 or d <= 0:
        raise ValueError("M, seq_len and d must be positive")
    if d > MAX_HIDDEN_DIM:
        raise ValueError(f"d={d} exceeds the hidden-dim refusal of {MAX_HIDDEN_DIM}")

    geo = geometry(M, seq_len, d)
    dw = geo["DW"]
    nchunks = geo["nchunks"]
    rc = geo["RC"]
    nch = geo["nch"]
    grid = geo["grid"]
    if seq_len % rc:
        raise AssertionError(f"RC={rc} does not divide seq_len={seq_len}")
    fp32 = "float32"
    inv_d = 1.0 / float(d)
    inv_sqrt_d = 1.0 / (d**0.5)
    taps = KS

    # ------------------------------------------------------------------ launch A
    @T.prim_func
    def gate(
        H: T.Tensor((M, seq_len, d), dtype),
        K: T.Tensor((M, seq_len, d), dtype),
        V: T.Tensor((M, seq_len, d), dtype),
        WH: T.Tensor((d,), dtype),
        WV: T.Tensor((d,), dtype),
        VHAT: T.Tensor((M, seq_len, d), dtype),
        Alpha: T.Tensor((M, seq_len), fp32),
        RH: T.Tensor((M, seq_len), fp32),
        RK: T.Tensor((M, seq_len), fp32),
        RV: T.Tensor((M, seq_len), fp32),
    ):
        with T.Kernel(grid) as bx:
            b = bx // nch
            c = bx % nch
            t0 = c * rc

            h_ub = T.alloc_shared((rc, dw), dtype)
            k_ub = T.alloc_shared((rc, dw), dtype)
            v_ub = T.alloc_shared((rc, dw), dtype)
            vh_ub = T.alloc_shared((rc, dw), dtype)
            wh_ub = T.alloc_shared((dw,), dtype)
            wv_ub = T.alloc_shared((dw,), dtype)
            w2h_ub = T.alloc_shared((dw,), fp32)
            st = T.alloc_shared((MEMBERS, rc, STATS_STRIDE), fp32)
            T.annotate_buffer_versions(
                {
                    h_ub: 1,
                    k_ub: 1,
                    v_ub: 1,
                    vh_ub: 1,
                    wh_ub: 1,
                    wv_ub: 1,
                    w2h_ub: 1,
                    st: 1,
                }
            )

            # one (rc, d) copy per input tensor: no halo rows are needed here
            T.copy(WH, wh_ub[:d])
            T.copy(WV, wv_ub[:d])
            T.copy(H[b, t0 : t0 + rc, :], h_ub[:, :d])
            T.copy(K[b, t0 : t0 + rc, :], k_ub[:, :d])
            T.copy(V[b, t0 : t0 + rc, :], v_ub[:, :d])
            with T.SimdVF():
                full_p = T.simd.pset(32)
                for ch in T.serial(nchunks):
                    mask = T.simd.update_mask(T.min(VL, d - ch * VL), 32)
                    wv = T.simd.vcvt(T.simd.vld(wh_ub[ch * VL], dist="UNPK_B16"), fp32, part=0)
                    T.simd.vsts(w2h_ub[ch * VL], T.simd.vmul(wv, wv, mask))

            for i in T.serial(rc):
                with T.SimdVF():
                    full = T.simd.pset(32)
                    one = T.simd.pset(32, "PAT_VL1")
                    one_f = T.simd.vdup(T.float32(1), fp32, full)
                    zero_f = T.simd.vdup(T.float32(0), fp32, full)
                    hsq = T.simd.alloc_local((1,), fp32)
                    ksq = T.simd.alloc_local((1,), fp32)
                    hk = T.simd.alloc_local((1,), fp32)
                    vsq = T.simd.alloc_local((1,), fp32)
                    alpha_v = T.simd.alloc_local((1,), fp32)
                    rv_v = T.simd.alloc_local((1,), fp32)
                    hsq[0] = zero_f
                    ksq[0] = zero_f
                    hk[0] = zero_f
                    for ch in T.serial(nchunks):
                        mask = T.simd.update_mask(T.min(VL, d - ch * VL), 32)
                        hf = T.simd.vcvt(
                            T.simd.vld(h_ub[i, ch * VL], dist="UNPK_B16"), fp32, part=0
                        )
                        kf = T.simd.vcvt(
                            T.simd.vld(k_ub[i, ch * VL], dist="UNPK_B16"), fp32, part=0
                        )
                        w2 = T.simd.vld(w2h_ub[ch * VL])
                        hsq[0] = T.simd.vadd(hsq[0], T.simd.vmul(hf, hf, mask), full)
                        ksq[0] = T.simd.vadd(ksq[0], T.simd.vmul(kf, kf, mask), full)
                        hk[0] = T.simd.vadd(
                            hk[0], T.simd.vmul(T.simd.vmul(hf, kf, mask), w2, mask), full
                        )
                    hmean = T.simd.vadd(
                        T.simd.vmul(
                            T.simd.vdupv(T.simd.vcadd(hsq[0], full)),
                            T.simd.vdup(T.float32(inv_d), fp32, full),
                            full,
                        ),
                        T.simd.vdup(T.float32(eps), fp32, full),
                        full,
                    )
                    rh_v = T.simd.vdiv(one_f, T.simd.vsqrt(hmean, full), full)
                    kmean = T.simd.vadd(
                        T.simd.vmul(
                            T.simd.vdupv(T.simd.vcadd(ksq[0], full)),
                            T.simd.vdup(T.float32(inv_d), fp32, full),
                            full,
                        ),
                        T.simd.vdup(T.float32(eps), fp32, full),
                        full,
                    )
                    rk_v = T.simd.vdiv(one_f, T.simd.vsqrt(kmean, full), full)
                    logit = T.simd.vmul(
                        T.simd.vmul(T.simd.vdupv(T.simd.vcadd(hk[0], full)), rh_v, full),
                        T.simd.vmul(rk_v, T.simd.vdup(T.float32(inv_sqrt_d), fp32, full), full),
                        full,
                    )
                    alpha_v[0] = T.simd.vdiv(
                        one_f,
                        T.simd.vadd(one_f, T.simd.vexp(T.simd.vneg(logit, full), full), full),
                        full,
                    )
                    T.simd.vsts(st[0, i, 0], alpha_v[0], one, extent=1)
                    T.simd.vsts(st[1, i, 0], rh_v, one, extent=1)
                    T.simd.vsts(st[2, i, 0], rk_v, one, extent=1)
                    vsq[0] = zero_f
                    for ch in T.serial(nchunks):
                        mask = T.simd.update_mask(T.min(VL, d - ch * VL), 32)
                        vf = T.simd.vcvt(
                            T.simd.vld(v_ub[i, ch * VL], dist="UNPK_B16"), fp32, part=0
                        )
                        narrowed = T.simd.vcvt(
                            T.simd.vmul(vf, alpha_v[0], mask),
                            dtype,
                            round="ROUND_R",
                            sat=False,
                            part=0,
                        )
                        T.simd.vsts(vh_ub[i, ch * VL], narrowed, full, dist="PK_B32")
                        back = T.simd.vcvt(narrowed, fp32, part=0)
                        vsq[0] = T.simd.vadd(vsq[0], T.simd.vmul(back, back, mask), full)
                    vmean = T.simd.vadd(
                        T.simd.vmul(
                            T.simd.vdupv(T.simd.vcadd(vsq[0], full)),
                            T.simd.vdup(T.float32(inv_d), fp32, full),
                            full,
                        ),
                        T.simd.vdup(T.float32(eps), fp32, full),
                        full,
                    )
                    rv_v[0] = T.simd.vdiv(one_f, T.simd.vsqrt(vmean, full), full)
                    T.simd.vsts(st[3, i, 0], rv_v[0], one, extent=1)

            T.copy(vh_ub[:, :d], VHAT[b, t0 : t0 + rc, :])
            T.copy(st[0, :, 0:1], Alpha[b, t0 : t0 + rc])
            T.copy(st[1, :, 0:1], RH[b, t0 : t0 + rc])
            T.copy(st[2, :, 0:1], RK[b, t0 : t0 + rc])
            T.copy(st[3, :, 0:1], RV[b, t0 : t0 + rc])

    # ------------------------------------------------------------------ launch B
    @T.prim_func
    def conv(
        VHAT: T.Tensor((M, seq_len, d), dtype),
        RV: T.Tensor((M, seq_len), fp32),
        ALPHA: T.Tensor((M, seq_len), fp32),
        V: T.Tensor((M, seq_len, d), dtype),
        WV: T.Tensor((d,), dtype),
        CW: T.Tensor((taps, d), dtype),
        Y: T.Tensor((M, seq_len, d), dtype),
    ):
        with T.Kernel(grid) as bx:
            b = bx // nch
            c = bx % nch
            t0 = c * rc

            vh_win = T.alloc_shared((rc + 3, dw), dtype)
            v_win = T.alloc_shared((rc, dw), dtype)
            y_ub = T.alloc_shared((rc, dw), dtype)
            wv_ub = T.alloc_shared((dw,), dtype)
            cw_ub = T.alloc_shared((taps, dw), dtype)
            cwv_ub = T.alloc_shared((taps, dw), fp32)
            rvb = T.alloc_shared((rc + 3, STATS_STRIDE), fp32)
            ab = T.alloc_shared((rc, STATS_STRIDE), fp32)
            T.annotate_buffer_versions(
                {
                    vh_win: 1,
                    v_win: 1,
                    y_ub: 1,
                    wv_ub: 1,
                    cw_ub: 1,
                    cwv_ub: 1,
                    rvb: 1,
                    ab: 1,
                }
            )

            T.copy(WV, wv_ub[:d])
            T.copy(CW, cw_ub[:, :d])
            T.copy(V[b, t0 : t0 + rc, :], v_win[:, :d])
            T.copy(ALPHA[b, t0 : t0 + rc], ab[:, 0:1])
            if t0 >= 3:
                T.copy(VHAT[b, t0 - 3 : t0, :], vh_win[0:3, :d])
                T.copy(RV[b, t0 - 3 : t0], rvb[0:3, 0:1])
            else:
                # causal boundary: the rows before the sequence start are zeros
                with T.SimdVF():
                    zfull = T.simd.pset(32)
                    zone = T.simd.pset(32, "PAT_VL1")
                    zvec = T.simd.vdup(T.float32(0), fp32, zfull)
                    zb = T.simd.vcvt(zvec, dtype, round="ROUND_R", sat=False, part=0)
                    for hz in T.serial(3):
                        for ch in T.serial(nchunks):
                            T.simd.vsts(vh_win[hz, ch * VL], zb, zfull, dist="PK_B32")
                        T.simd.vsts(rvb[hz, 0], zvec, zone, extent=1)
                for hz in range(3):
                    rr = t0 - 3 + hz
                    if rr >= 0:
                        T.copy(VHAT[b, rr, :], vh_win[hz, :d])
                        T.copy(RV[b, rr : rr + 1], rvb[hz, 0:1])
            T.copy(VHAT[b, t0 : t0 + rc, :], vh_win[3 : 3 + rc, :d])
            T.copy(RV[b, t0 : t0 + rc], rvb[3 : 3 + rc, 0:1])
            with T.SimdVF():
                full_p = T.simd.pset(32)
                for p in T.serial(taps):
                    for ch in T.serial(nchunks):
                        mask = T.simd.update_mask(T.min(VL, d - ch * VL), 32)
                        cf = T.simd.vcvt(
                            T.simd.vld(cw_ub[p, ch * VL], dist="UNPK_B16"), fp32, part=0
                        )
                        vf = T.simd.vcvt(
                            T.simd.vld(wv_ub[ch * VL], dist="UNPK_B16"), fp32, part=0
                        )
                        T.simd.vsts(cwv_ub[p, ch * VL], T.simd.vmul(cf, vf, mask))

            for i in T.serial(rc):
                with T.SimdVF():
                    full = T.simd.pset(32)
                    one_f = T.simd.vdup(T.float32(1), fp32, full)
                    zero_f = T.simd.vdup(T.float32(0), fp32, full)
                    acc = T.simd.alloc_local((1,), fp32)
                    rl = [T.simd.alloc_local((1,), fp32) for _ in range(taps)]
                    al = T.simd.alloc_local((1,), fp32)
                    T.simd.mem_bar("VST_VLD")
                    al[0] = T.simd.vld(ab[i, 0], dist="BRC_B32")
                    for p in range(taps):  # build-time unroll: 4 tap scalars
                        rl[p][0] = T.simd.vld(rvb[i + p, 0], dist="BRC_B32")
                    for ch in T.serial(nchunks):
                        mask = T.simd.update_mask(T.min(VL, d - ch * VL), 32)
                        acc[0] = zero_f
                        for p in T.unroll(taps):
                            gf = T.simd.vcvt(
                                T.simd.vld(vh_win[i + p, ch * VL], dist="UNPK_B16"), fp32, part=0
                            )
                            acc[0] = T.simd.vadd(
                                acc[0],
                                T.simd.vmul(
                                    T.simd.vmul(gf, rl[p][0], mask),
                                    T.simd.vld(cwv_ub[p, ch * VL]),
                                    mask,
                                ),
                                full,
                            )
                        silu = T.simd.vdiv(
                            acc[0],
                            T.simd.vadd(one_f, T.simd.vexp(T.simd.vneg(acc[0], full), full), full),
                            full,
                        )
                        vf = T.simd.vcvt(
                            T.simd.vld(v_win[i, ch * VL], dist="UNPK_B16"), fp32, part=0
                        )
                        y = T.simd.vadd(silu, T.simd.vmul(vf, al[0], mask), mask)
                        T.simd.vsts(
                            y_ub[i, ch * VL],
                            T.simd.vcvt(y, dtype, round="ROUND_R", sat=False, part=0),
                            full,
                            dist="PK_B32",
                        )
            T.copy(y_ub[:, :d], Y[b, t0 : t0 + rc, :])

    ka = tilelang.compile(gate, target="ascend", out_idx=[5, 6, 7, 8, 9])
    kb = tilelang.compile(conv, target="ascend", out_idx=[6])

    def run(H, k, v, wh, wv, cw):
        vhat, alpha, rh, rk, rv = ka(H, k, v, wh, wv)
        y = kb(vhat, rv, alpha, v, wv, cw)
        return y, vhat, alpha, rh, rk, rv

    return run
'''


def main():
    with open(SRC, encoding="utf-8") as f:
        text = f.read()
    start = text.index(START)
    end = text.index(END) + len(END)
    out = text[:start] + NEW + text[end:]
    with open(DST, "w", encoding="utf-8") as f:
        f.write(out)
    print("wrote", DST, len(out.splitlines()), "lines")


if __name__ == "__main__":
    main()
