"""Design-phase compile probe (D-3) for the Engram GateConv *backward* design.

No device work: ``tilelang.compile(..., target="ascend")`` only lowers and calls
the BishengIR compiler.  Run with the card hidden:

    cd <task dir>
    source ../../../.agents/platforms/toolchains/ascend/env-950pr.sh
    ASCEND_RT_VISIBLE_DEVICES= /workspace/venv-ascend950/bin/python history_version/design_probe_compile.py

What it validates — the constructs DESIGN.md 3.3 depends on that neither the
knowledge base nor the verified 950DT sibling (`TileOPs-ascend950
.../sequence_modeling/gate_conv_bwd.py`) nor the in-repo forward probe
(`examples/engram_gate_conv/_engram_gate_conv_fwd_kernel/history_version/design_probe_compile.py`)
already cover:

  B1  a *fused* backward: one kernel per core chunk that (a) recomputes
      conv_out / d_silu for a lead of +3 rows, (b) keeps the four rolling fp32
      windows (`nrm`, `ds`, `dy`) in UB keyed by ``floormod(position, 4)``, and
      (c) runs the whole value/gate backward for the *lagged* token in the same
      iteration — the source's three launches collapsed into one;
  B2  out-of-batch halo rows handled by *uniform scalar guards* (an out-of-range
      row is never loaded, its window slot is written to 0), expressed as two
      complementary ``if`` statements because the traced frontend has no scalar
      ``else`` -- the source's ``T.if_then_else(..., 0)`` semantics without a
      per-element select and without the ``0 * inf`` hazard of a clamp+multiply;
  B3  per-core fp32 weight-gradient partials accumulated in UB across the whole
      chunk and stored once with a *dynamic* second index
      (``T.copy(pc[p, :], PART[2 + p, bx, :])``);
  B4  per-batch scalar strips ``ALPHA/RH/RK/RV`` staged once
      (``T.copy(ALPHA[b, :], sc[0, :])``) and read per row through a
      dynamic-address ``BRC_B32`` vector load;
  B5  the ordered reduction kernel: 6 groups x column tiles, one 6-way static
      ``if`` chain (dynamic group never indexes a buffer), running sum over the
      core axis held in one fp32 register in ascending core order;
  B6  the tail geometries: prime ``seq`` (chunk does not divide seq), ``seq``
      smaller than the halo, and ``d`` not a multiple of the 64-lane width.

Each case prints COMPILE OK / FAIL so a failure localises to a construct.
"""

import functools
import sys
import traceback

import tilelang
import tilelang.ascend.language as T

VL = 64      # fp32 lanes per vector register (fp16 chunk = 64 elements)
KS = 4       # conv taps
CORES = 56   # Ascend950PR_9579 vector cores (statectl platform hardware.observed_cores)
FP32 = "float32"


def geometry(M, seq, d):
    """Build-time geometry of DESIGN.md 5.2: chunk length and launch grid."""
    dw = -(-d // VL) * VL
    nch = min(seq, max(1, CORES // M))
    rc = -(-seq // nch)
    nch = -(-seq // rc)
    return {"dw": dw, "nchunks": dw // VL, "RC": rc, "nch": nch, "grid": M * nch}


@T.macro
def _zero_words(buf, words):
    """Zero complete 32-bit words with SIMD (same helper as the 950DT sibling)."""
    bits = T.view(buf, (words,), dtype="uint32")
    full = T.simd.pset(32)
    zero = T.simd.vdup(T.uint32(0), "uint32", full)
    for chunk in T.serial(T.ceildiv(words, 64)):
        count = T.min(64, words - chunk * 64)
        valid = T.simd.update_mask(count, 32)
        T.simd.vsts(bits[chunk * 64], zero, valid, extent=count)


@functools.lru_cache(maxsize=16)
def build_fused(M, seq, d, dtype):
    """Fused GateConv backward: dH/dk/dv + per-core weight-gradient partials."""
    geo = geometry(M, seq, d)
    dw, nchunks, rc, nch, grid = geo["dw"], geo["nchunks"], geo["RC"], geo["nch"], geo["grid"]
    dim = float(d)
    inv_d = 1.0 / dim            # DESIGN 1.6.1 O1: /d  -> * (1/d)
    inv_sqrt_d = 1.0 / (d ** 0.5)  # DESIGN 1.6.1 O1: /sqrt(d) -> * (1/sqrt(d))
    lead = KS - 1                # rows of d_silu lead the fused loop must carry

    @T.prim_func
    def main(
        DY: T.Tensor((M, seq, d), dtype),
        H: T.Tensor((M, seq, d), dtype),
        K: T.Tensor((M, seq, d), dtype),
        V: T.Tensor((M, seq, d), dtype),
        WH: T.Tensor((d,), dtype),
        WV: T.Tensor((d,), dtype),
        CW: T.Tensor((KS, d), dtype),
        VHAT: T.Tensor((M, seq, d), dtype),
        ALPHA: T.Tensor((M, seq), FP32),
        RH: T.Tensor((M, seq), FP32),
        RK: T.Tensor((M, seq), FP32),
        RV: T.Tensor((M, seq), FP32),
        DH: T.Tensor((M, seq, d), dtype),
        DK: T.Tensor((M, seq, d), dtype),
        DV: T.Tensor((M, seq, d), dtype),
        PART: T.Tensor((2 + KS, grid, dw), FP32),
    ):
        with T.Kernel(grid) as bx:
            b = bx // nch
            c = bx % nch
            t0 = c * rc

            # --- staging rows (input dtype), static slot: T.copy destinations ---
            vrow = T.alloc_shared((dw,), dtype)
            drow = T.alloc_shared((dw,), dtype)
            hrow = T.alloc_shared((dw,), dtype)
            krow = T.alloc_shared((dw,), dtype)
            vvrow = T.alloc_shared((dw,), dtype)
            whs = T.alloc_shared((dw,), dtype)
            wvs = T.alloc_shared((dw,), dtype)
            cws = T.alloc_shared((KS, dw), dtype)

            # --- fp32 rolling windows keyed by floormod(position, KS) ---
            nrmw = T.alloc_shared((KS, dw), FP32)   # vhat_norm  (DESIGN 1.6.1 O5)
            dsw = T.alloc_shared((KS, dw), FP32)    # d_silu
            dyw = T.alloc_shared((KS, dw), FP32)    # dY

            # --- fp32 constant tables (converted once per launch) ---
            whf = T.alloc_shared((dw,), FP32)
            wvf = T.alloc_shared((dw,), FP32)
            cwf = T.alloc_shared((KS, dw), FP32)

            # --- per-core weight-gradient partials (B3) ---
            pc = T.alloc_shared((KS, dw), FP32)
            ph = T.alloc_shared((dw,), FP32)
            pv = T.alloc_shared((dw,), FP32)

            # --- per-token scratch (full row resident: the row dots precede use) ---
            gn = T.alloc_shared((dw,), FP32)
            gz = T.alloc_shared((dw,), FP32)
            hnf = T.alloc_shared((dw,), FP32)
            knf = T.alloc_shared((dw,), FP32)

            # --- per-batch scalar strips (B4) ---
            sc = T.alloc_shared((4, seq), FP32)

            T.annotate_buffer_versions(
                {
                    vrow: 1, drow: 1, hrow: 1, krow: 1, vvrow: 1,
                    whs: 1, wvs: 1, cws: 1,
                    nrmw: 1, dsw: 1, dyw: 1,
                    whf: 1, wvf: 1, cwf: 1,
                    pc: 1, ph: 1, pv: 1,
                    gn: 1, gz: 1, hnf: 1, knf: 1, sc: 1,
                }
            )

            # ---- zero partials and any tail lane beyond d -------------------
            with T.SimdVF():
                full = T.simd.pset(32)
                zero = T.simd.vdup(0.0, FP32, full)
                for ch in T.serial(nchunks):
                    T.simd.vsts(ph[ch * VL], zero, full)
                    T.simd.vsts(pv[ch * VL], zero, full)
                    for p in T.serial(KS):
                        T.simd.vsts(pc[p, ch * VL], zero, full)

            # ---- tables: fp16 GM -> fp16 UB -> fp32 UB (masked) --------------
            T.copy(WH, whs[:d])
            T.copy(WV, wvs[:d])
            T.copy(CW, cws[:, :d])
            T.copy(ALPHA[b, :], sc[0, :])
            T.copy(RH[b, :], sc[1, :])
            T.copy(RK[b, :], sc[2, :])
            T.copy(RV[b, :], sc[3, :])
            with T.SimdVF():
                full = T.simd.pset(32)
                for ch in T.serial(nchunks):
                    mask = T.simd.update_mask(T.min(VL, d - ch * VL), 32)
                    T.simd.vsts(
                        whf[ch * VL],
                        T.simd.vcvt(T.simd.vld(whs[ch * VL], dist="UNPK_B16"), FP32,
                                    mask, part=0),
                        full,
                    )
                    T.simd.vsts(
                        wvf[ch * VL],
                        T.simd.vcvt(T.simd.vld(wvs[ch * VL], dist="UNPK_B16"), FP32,
                                    mask, part=0),
                        full,
                    )
                    for p in T.serial(KS):
                        T.simd.vsts(
                            cwf[p, ch * VL],
                            T.simd.vcvt(T.simd.vld(cws[p, ch * VL], dist="UNPK_B16"), FP32,
                                        mask, part=0),
                            full,
                        )

            # =====================================================================
            # fused loop: iteration i carries one new d_silu row (lead = +3) and
            # runs the value/gate backward for the token that is 3 rows behind.
            # =====================================================================
            for i in T.serial(rc + 2 * lead):
                pn = t0 - lead + i                      # normalised-vhat row
                nslot = T.floormod(pn, KS)
                # B2: uniform scalar branch — an out-of-batch row is *never read*
                # (the source's T.if_then_else semantics); the slot is zeroed so
                # the causal taps read exact 0.  Two complementary guards avoid a
                # scalar ``else`` in the traced frontend.
                if pn >= 0:
                    if pn < seq:
                        T.copy(VHAT[b, pn, :], vrow[:d])
                        T.copy(DY[b, pn, :], drow[:d])
                        with T.SimdVF():
                            full = T.simd.pset(32)
                            rv_p = T.simd.vld(sc[3, pn], "BRC_B32")   # B4: dynamic BRC read
                            for ch in T.serial(nchunks):
                                mask = T.simd.update_mask(T.min(VL, d - ch * VL), 32)
                                z = T.simd.vcvt(T.simd.vld(vrow[ch * VL], dist="UNPK_B16"),
                                                FP32, mask, part=0)
                                dyv = T.simd.vcvt(T.simd.vld(drow[ch * VL], dist="UNPK_B16"),
                                                  FP32, mask, part=0)
                                # O5: vhat_norm materialised once per row, 4 taps reuse it
                                nrm = T.simd.vmul(
                                    T.simd.vmul(z, rv_p, mask),
                                    T.simd.vld(wvf[ch * VL]), mask,
                                )
                                T.simd.vsts(nrmw[nslot, ch * VL], nrm, full)
                                T.simd.vsts(dyw[nslot, ch * VL], dyv, full)
                    if pn >= seq:
                        with T.SimdVF():
                            full = T.simd.pset(32)
                            zero = T.simd.vdup(0.0, FP32, full)
                            for ch in T.serial(nchunks):
                                T.simd.vsts(nrmw[nslot, ch * VL], zero, full)
                                T.simd.vsts(dyw[nslot, ch * VL], zero, full)
                if pn < 0:
                    with T.SimdVF():
                        full = T.simd.pset(32)
                        zero = T.simd.vdup(0.0, FP32, full)
                        for ch in T.serial(nchunks):
                            T.simd.vsts(nrmw[nslot, ch * VL], zero, full)
                            T.simd.vsts(dyw[nslot, ch * VL], zero, full)

                # ---- d_silu for row pds = t0 + i - lead ---------------------
                if i >= lead:
                    pds = t0 + i - lead
                    dslot = T.floormod(pds, KS)
                    if pds < seq:
                        with T.SimdVF():
                            full = T.simd.pset(32)
                            one = T.simd.vdup(1.0, FP32, full)
                            for ch in T.serial(nchunks):
                                mask = T.simd.update_mask(T.min(VL, d - ch * VL), 32)
                                conv = T.simd.alloc_local((1,), FP32)
                                conv[0] = T.simd.vdup(0.0, FP32, full)
                                for p in T.unroll(KS):
                                    conv[0] = T.simd.vadd(
                                        conv[0],
                                        T.simd.vmul(
                                            T.simd.vld(nrmw[T.floormod(pds - lead + p, KS),
                                                            ch * VL]),
                                            T.simd.vld(cwf[p, ch * VL]),
                                            mask,
                                        ),
                                        full,
                                    )
                                sig = T.simd.vdiv(
                                    one,
                                    T.simd.vadd(
                                        one,
                                        T.simd.vexp(T.simd.vneg(conv[0], mask), mask), mask),
                                    mask,
                                )
                                deriv = T.simd.vadd(
                                    sig,
                                    T.simd.vmul(
                                        T.simd.vmul(conv[0], sig, mask),
                                        T.simd.vsub(one, sig, mask),
                                        mask,
                                    ),
                                    mask,
                                )
                                T.simd.vsts(
                                    dsw[dslot, ch * VL],
                                    T.simd.vmul(T.simd.vld(dyw[dslot, ch * VL]), deriv, mask),
                                    full,
                                )
                    if pds >= seq:
                        with T.SimdVF():
                            full = T.simd.pset(32)
                            zero = T.simd.vdup(0.0, FP32, full)
                            for ch in T.serial(nchunks):
                                T.simd.vsts(dsw[dslot, ch * VL], zero, full)
                    # ---- dconv_w partial: only the chunk that owns the row ----
                    if i < rc + lead:
                        if pds < seq:
                            with T.SimdVF():
                                full = T.simd.pset(32)
                                for ch in T.serial(nchunks):
                                    mask = T.simd.update_mask(T.min(VL, d - ch * VL), 32)
                                    for p in T.unroll(KS):
                                        T.simd.vsts(
                                            pc[p, ch * VL],
                                            T.simd.vadd(
                                                T.simd.vld(pc[p, ch * VL]),
                                                T.simd.vmul(
                                                    T.simd.vld(dsw[dslot, ch * VL]),
                                                    T.simd.vld(
                                                        nrmw[T.floormod(pds - lead + p, KS),
                                                             ch * VL]),
                                                    mask,
                                                ),
                                                full,
                                            ),
                                            full,
                                        )

                # ---- value / gate backward for token tp = pn ---------------- (B1c)
                if i >= 2 * lead:
                    tp = t0 + i - 2 * lead
                    T.copy(H[b, tp, :], hrow[:d])
                    T.copy(K[b, tp, :], krow[:d])
                    T.copy(V[b, tp, :], vvrow[:d])
                    with T.SimdVF():
                        full = T.simd.pset(32)
                        one = T.simd.vdup(1.0, FP32, full)
                        rv = T.simd.vld(sc[3, tp], "BRC_B32")
                        rh = T.simd.vld(sc[1, tp], "BRC_B32")
                        rk = T.simd.vld(sc[2, tp], "BRC_B32")
                        al = T.simd.vld(sc[0, tp], "BRC_B32")
                        rv3d = T.simd.alloc_local((1,), FP32)   # O2: rv^3 * (1/d) once
                        rv3d[0] = T.simd.vmuls(
                            T.simd.vmul(T.simd.vmul(rv, rv, full), rv, full), inv_d, full
                        )
                        # pass A: conv transpose -> d_vnorm, dot_v, drms_w_v partial
                        dot = T.simd.alloc_local((1,), FP32)
                        dot[0] = T.simd.vdup(0.0, FP32, full)
                        for ch in T.serial(nchunks):
                            mask = T.simd.update_mask(T.min(VL, d - ch * VL), 32)
                            dn = T.simd.alloc_local((1,), FP32)
                            dn[0] = T.simd.vdup(0.0, FP32, full)
                            for p in T.unroll(KS):
                                dn[0] = T.simd.vadd(
                                    dn[0],
                                    T.simd.vmul(
                                        T.simd.vld(cwf[p, ch * VL]),
                                        T.simd.vld(dsw[T.floormod(tp + lead - p, KS),
                                                       ch * VL]),
                                        mask,
                                    ),
                                    full,
                                )
                            T.simd.vsts(gn[ch * VL], dn[0], full)
                            zd = T.simd.vmul(T.simd.vld(nrmw[T.floormod(tp, KS), ch * VL]),
                                             dn[0], mask)            # O3 CSE
                            dot[0] = T.simd.vadd(
                                dot[0], T.simd.vmul(zd, T.simd.vld(wvf[ch * VL]), mask), full
                            )
                            T.simd.vsts(
                                pv[ch * VL],
                                T.simd.vadd(T.simd.vld(pv[ch * VL]),
                                            T.simd.vmul(zd, rv, mask), full),
                                full,
                            )
                        dot_b = T.simd.vdupv(T.simd.vcadd(dot[0], full), full)
                        # pass B: dvhat -> dv output and dalpha
                        da = T.simd.alloc_local((1,), FP32)
                        da[0] = T.simd.vdup(0.0, FP32, full)
                        for ch in T.serial(nchunks):
                            mask = T.simd.update_mask(T.min(VL, d - ch * VL), 32)
                            z = T.simd.vld(nrmw[T.floormod(tp, KS), ch * VL])
                            wvz = T.simd.vmul(T.simd.vld(wvf[ch * VL]),
                                              T.simd.vld(gn[ch * VL]), mask)
                            direct = T.simd.vmul(rv, wvz, mask)
                            corr = T.simd.vmul(
                                T.simd.vmul(rv3d[0], z, mask), dot_b, mask
                            )
                            g = T.simd.vadd(
                                T.simd.vsub(direct, corr, mask),
                                T.simd.vld(dyw[T.floormod(tp, KS), ch * VL]),
                                mask,
                            )
                            T.simd.vsts(gz[ch * VL], g, full)
                            vv = T.simd.vcvt(T.simd.vld(vvrow[ch * VL], dist="UNPK_B16"),
                                             FP32, mask, part=0)
                            da[0] = T.simd.vadd(da[0], T.simd.vmul(g, vv, mask), full)
                            T.simd.vsts(
                                hrow[ch * VL],
                                T.simd.vcvt(T.simd.vmul(al, g, mask), dtype,
                                            round="ROUND_R", sat=False, mask=mask, part=0),
                                full,
                                dist="PK_B32",
                            )
                        da_b = T.simd.vdupv(T.simd.vcadd(da[0], full), full)
                        ddot = T.simd.vmul(
                            T.simd.vmul(da_b, al, full),
                            T.simd.vmul(T.simd.vsub(one, al, full),
                                        T.simd.vdup(inv_sqrt_d, FP32, full), full),
                            full,
                        )
                        # pass C1: gate norms, the two row dots and drms_w_h partial
                        dh_dot = T.simd.alloc_local((1,), FP32)
                        dk_dot = T.simd.alloc_local((1,), FP32)
                        dh_dot[0] = T.simd.vdup(0.0, FP32, full)
                        dk_dot[0] = T.simd.vdup(0.0, FP32, full)
                        for ch in T.serial(nchunks):
                            mask = T.simd.update_mask(T.min(VL, d - ch * VL), 32)
                            hf = T.simd.vcvt(T.simd.vld(hrow[ch * VL], dist="UNPK_B16"),
                                             FP32, mask, part=0)
                            kf = T.simd.vcvt(T.simd.vld(krow[ch * VL], dist="UNPK_B16"),
                                             FP32, mask, part=0)
                            w = T.simd.vld(whf[ch * VL])
                            hn = T.simd.vmul(T.simd.vmul(hf, rh, mask), w, mask)
                            kn = T.simd.vmul(T.simd.vmul(kf, rk, mask), w, mask)
                            T.simd.vsts(hnf[ch * VL], hn, full)
                            T.simd.vsts(knf[ch * VL], kn, full)
                            uh = T.simd.vmul(ddot, kn, mask)     # u_h = ddot * k_norm
                            uk = T.simd.vmul(ddot, hn, mask)
                            ph_v = T.simd.vadd(
                                T.simd.vmul(T.simd.vmul(uh, hf, mask), rh, mask),
                                T.simd.vmul(T.simd.vmul(uk, kf, mask), rk, mask),
                                mask,
                            )
                            T.simd.vsts(
                                ph[ch * VL],
                                T.simd.vadd(T.simd.vld(ph[ch * VL]), ph_v, full),
                                full,
                            )
                            dh_dot[0] = T.simd.vadd(
                                dh_dot[0], T.simd.vmul(T.simd.vmul(w, uh, mask), hf, mask), full
                            )
                            dk_dot[0] = T.simd.vadd(
                                dk_dot[0], T.simd.vmul(T.simd.vmul(w, uk, mask), kf, mask), full
                            )
                        dhb = T.simd.vdupv(T.simd.vcadd(dh_dot[0], full), full)
                        dkb = T.simd.vdupv(T.simd.vcadd(dk_dot[0], full), full)
                        # pass C2: dH / dk outputs
                        for ch in T.serial(nchunks):
                            mask = T.simd.update_mask(T.min(VL, d - ch * VL), 32)
                            hf = T.simd.vcvt(T.simd.vld(hrow[ch * VL], dist="UNPK_B16"),
                                             FP32, mask, part=0)
                            kf = T.simd.vcvt(T.simd.vld(krow[ch * VL], dist="UNPK_B16"),
                                             FP32, mask, part=0)
                            w = T.simd.vld(whf[ch * VL])
                            hn = T.simd.vld(hnf[ch * VL])
                            kn = T.simd.vld(knf[ch * VL])
                            rh3d = T.simd.vmuls(T.simd.vmul(T.simd.vmul(rh, rh, full), rh, full),
                                                inv_d, full)
                            rk3d = T.simd.vmuls(T.simd.vmul(T.simd.vmul(rk, rk, full), rk, full),
                                                inv_d, full)
                            dh = T.simd.vsub(
                                T.simd.vmul(rh, T.simd.vmul(w, T.simd.vmul(ddot, kn, mask),
                                                             mask), mask),
                                T.simd.vmul(T.simd.vmul(rh3d, hf, mask), dhb, mask),
                                mask,
                            )
                            dkk = T.simd.vsub(
                                T.simd.vmul(rk, T.simd.vmul(w, T.simd.vmul(ddot, hn, mask),
                                                             mask), mask),
                                T.simd.vmul(T.simd.vmul(rk3d, kf, mask), dkb, mask),
                                mask,
                            )
                            T.simd.vsts(
                                krow[ch * VL],
                                T.simd.vcvt(dh, dtype, round="ROUND_R", sat=False,
                                            mask=mask, part=0),
                                full,
                                dist="PK_B32",
                            )
                            T.simd.vsts(
                                vvrow[ch * VL],
                                T.simd.vcvt(dkk, dtype, round="ROUND_R", sat=False,
                                            mask=mask, part=0),
                                full,
                                dist="PK_B32",
                            )
                    T.copy(hrow[:d], DH[b, tp, :])
                    T.copy(krow[:d], DK[b, tp, :])
                    T.copy(vvrow[:d], DV[b, tp, :])

            # ---- per-core partials -> GM workspace (B3) ---------------------
            for p in T.serial(KS):
                T.copy(pc[p, :], PART[2 + p, bx, :])
            T.copy(ph[:], PART[0, bx, :])
            T.copy(pv[:], PART[1, bx, :])

    return tilelang.compile(main, target="ascend", out_idx=[12, 13, 14, 15])


@functools.lru_cache(maxsize=16)
def build_reduce(cores, dw, d):
    """Ordered reduction of the per-core partials (DESIGN 6.2, second launch)."""
    tiles = min(9, max(1, -(-dw // (VL * 9))))
    width = -(-(-(-dw // tiles)) // VL) * VL
    tiles = -(-dw // width)

    @T.prim_func
    def main(
        PART: T.Tensor((2 + KS, cores, dw), FP32),
        DWH: T.Tensor((d,), FP32),
        DWV: T.Tensor((d,), FP32),
        DCW: T.Tensor((KS, d), FP32),
    ):
        with T.Kernel((2 + KS) * tiles) as bx:
            g = bx // tiles
            off = (bx % tiles) * width
            cols = T.min(width, dw - off)
            part = T.alloc_shared((cores, width), FP32)
            tot = T.alloc_shared((width,), FP32)
            T.annotate_buffer_versions({part: 1, tot: 1})
            if g == 0:
                T.copy(PART[0, :, off:off + cols], part[:, :cols])
            elif g == 1:
                T.copy(PART[1, :, off:off + cols], part[:, :cols])
            elif g == 2:
                T.copy(PART[2, :, off:off + cols], part[:, :cols])
            elif g == 3:
                T.copy(PART[3, :, off:off + cols], part[:, :cols])
            elif g == 4:
                T.copy(PART[4, :, off:off + cols], part[:, :cols])
            else:
                T.copy(PART[5, :, off:off + cols], part[:, :cols])
            with T.SimdVF():
                full = T.simd.pset(32)
                for ch in T.serial(width // VL):
                    col = ch * VL
                    valid = T.simd.update_mask(T.min(VL, T.max(0, cols - col)), 32)
                    run = T.simd.alloc_local((1,), FP32)
                    run[0] = T.simd.vdup(0.0, FP32, full)
                    for row in T.serial(cores):
                        run[0] = T.simd.vadd(run[0], T.simd.vld(part[row, col]), valid)
                    T.simd.vsts(tot[col], run[0], full)
            if g == 0:
                T.copy(tot[:cols], DWH[off:off + cols])
            elif g == 1:
                T.copy(tot[:cols], DWV[off:off + cols])
            elif g == 2:
                T.copy(tot[:cols], DCW[0, off:off + cols])
            elif g == 3:
                T.copy(tot[:cols], DCW[1, off:off + cols])
            elif g == 4:
                T.copy(tot[:cols], DCW[2, off:off + cols])
            else:
                T.copy(tot[:cols], DCW[3, off:off + cols])

    return tilelang.compile(main, target="ascend", out_idx=[1, 2, 3])


CASES = [
    ("train-s96", 3, 96, 768, "bfloat16"),
    ("train-s64", 2, 64, 512, "float16"),
    ("train-s128", 1, 128, 256, "bfloat16"),
    ("prime-401", 3, 401, 256, "bfloat16"),
    ("s512", 2, 512, 256, "float16"),
    ("short-1", 2, 1, 256, "float16"),
    ("pad-d192", 1, 16, 192, "bfloat16"),
]


def main():
    bad = 0
    for name, M, seq, d, dt in CASES:
        geo = geometry(M, seq, d)
        try:
            fn = build_fused(M, seq, d, dt)
            print(f"[{name}] fused M={M} seq={seq} d={d} {dt} geo={geo} COMPILE OK", flush=True)
            del fn
        except Exception as exc:  # noqa: BLE001 - the probe reports and continues
            bad += 1
            print(f"[{name}] fused M={M} seq={seq} d={d} COMPILE FAIL: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc(limit=4)
        try:
            fn = build_reduce(geo["grid"], geo["dw"], d)
            print(f"[{name}] reduce cores={geo['grid']} dw={geo['dw']} COMPILE OK", flush=True)
            del fn
        except Exception as exc:  # noqa: BLE001
            bad += 1
            print(f"[{name}] reduce cores={geo['grid']} dw={geo['dw']} COMPILE FAIL: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc(limit=4)
    print("PROBE_RESULT:", "ALL_OK" if bad == 0 else f"{bad} FAILED")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
