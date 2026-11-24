"""Design-phase compile probe (D-3) for the Engram GateConv forward design.

No device work: ``tilelang.compile(..., target="ascend")`` only lowers and calls
the BishengIR compiler.  Run with the card hidden:

    cd <task dir>
    source ../../../.agents/platforms/toolchains/ascend/env-950pr.sh
    ASCEND_RT_VISIBLE_DEVICES= /workspace/venv-ascend950/bin/python history_version/design_probe_compile.py

What it validates — the constructs DESIGN.md 3.3 depends on that the knowledge
base does not cover yet.  The 950DT sibling ``gate_conv_fwd.py`` covers the leaf
forms (``if`` inside ``T.unroll`` inside ``T.SimdVF``, ``vld``+``vcvt``+``vsts``
64-element chunk chain, ``alloc_local`` scalar registers); this probe covers the
*fused single-launch* nesting that the design introduces:

  P1  one-dimensional ``T.Kernel(grid)`` + ``T.serial(RC + 3)`` row loop whose
      body carries a *runtime* guard ``if r >= 0:`` around three ``T.copy``s
      (causal halo skip) and ``if i >= 3:`` around the conv/SiLU/store block;
  P2  the same guards *inside* one ``with T.SimdVF():`` frame;
  P3  dynamic rolling-window slot index ``slot[i % 4, ch * 64]`` as vld/vsts base
      (the fused replacement for the source's second launch);
  P4  fp32 full-vector ``vsts`` (default ``NORM_B32``), dtype ``vsts`` with
      ``dist="PK_B32"``, single-lane scalar ``vsts`` with ``extent=1``;
  P5  loop-carried fp32 accumulators in ``T.simd.alloc_local((1,), "float32")``
      accumulated over the chunk loop, reduced with ``vcadd`` and reused for the
      whole row (3 gate statistics + rrms_v);
  P6  batched strided scalar write-back
      ``T.copy(st[0, 0:RC, 0:1], Alpha[b, t0:t0+RC])`` plus per-row dtype
      write-back ``T.copy(vh_ub[:d], VHAT[b, r, :])``;
  P7  the geometry rule of DESIGN.md 5.2 on four manifest shapes, a prime
      ``seq_len`` (falls back to ``RC = 1``), and a ``d % 64 != 0`` shape
      (exercises the ``update_mask`` tail-lane path);
  P8  ``T.annotate_buffer_versions`` pinning the rolling window to one version.

Each case prints COMPILE OK / FAIL, so a failure localises to a construct.
"""

import functools
import sys
import traceback

import tilelang
import tilelang.ascend.language as T

VL = 64     # elements per fp32 vector register (= fp16 chunk, UNPK_B16/PK_B32)
KS = 4      # conv taps
CORES = 56  # Ascend950PR_9579 vector cores (statectl platform hardware.observed_cores)


def divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def geometry(M, seq, d):
    """Build-time geometry: tile rule and core mapping of DESIGN.md 5.2/5.5.

    ``RC`` always divides ``seq`` (so every chunk owns exactly ``RC`` rows and the
    kernel has no tail code path).  When the smallest admissible divisor would
    starve the core grid, fall back to ``RC = 1``, which always divides.
    """
    dw = -(-d // VL) * VL
    rows = M * seq
    target = max(1, -(-rows // CORES))
    cands = [d for d in divisors(seq) if d >= target]
    rc = min(cands) if cands else 1
    if rc > 1 and M * (seq // rc) < CORES // 2:
        rc = 1
    nch = seq // rc
    return {
        "DW": dw,
        "nchunks": dw // VL,
        "RC": rc,
        "nch": nch,
        "nfull": nch,
        "nlast": 0,
        "grid": M * nch,
    }


@functools.lru_cache(maxsize=8)
def build(M, seq, d, dtype, eps):
    geo = geometry(M, seq, d)
    dw, nchunks, rc, nch, nfull, nlast, grid = (
        geo["DW"],
        geo["nchunks"],
        geo["RC"],
        geo["nch"],
        geo["nfull"],
        geo["nlast"],
        geo["grid"],
    )
    fp32 = "float32"
    dim = float(d)
    inv_sqrt_d = 1.0 / (d**0.5)

    @T.prim_func
    def main(
        H: T.Tensor((M, seq, d), dtype),
        K: T.Tensor((M, seq, d), dtype),
        V: T.Tensor((M, seq, d), dtype),
        WH: T.Tensor((d,), dtype),
        WV: T.Tensor((d,), dtype),
        CW: T.Tensor((KS, d), dtype),
        Y: T.Tensor((M, seq, d), dtype),
        VHAT: T.Tensor((M, seq, d), dtype),
        Alpha: T.Tensor((M, seq), fp32),
        RH: T.Tensor((M, seq), fp32),
        RK: T.Tensor((M, seq), fp32),
        RV: T.Tensor((M, seq), fp32),
    ):
        with T.Kernel(grid) as bx:
            b = bx // nch
            c = bx % nch
            t0 = c * rc

            h_ub = T.alloc_shared((dw,), dtype)
            k_ub = T.alloc_shared((dw,), dtype)
            v_ub = T.alloc_shared((dw,), dtype)
            y_ub = T.alloc_shared((dw,), dtype)
            vh_ub = T.alloc_shared((dw,), dtype)
            wh_ub = T.alloc_shared((dw,), dtype)
            wv_ub = T.alloc_shared((dw,), dtype)
            cw_ub = T.alloc_shared((KS, dw), dtype)
            w2h_ub = T.alloc_shared((dw,), fp32)
            cwv_ub = T.alloc_shared((KS, dw), fp32)
            slot = T.alloc_shared((KS, dw), fp32)
            st = T.alloc_shared((4, rc + 3, 8), fp32)
            T.annotate_buffer_versions(
                {
                    slot: 1,
                    w2h_ub: 1,
                    cwv_ub: 1,
                    y_ub: 1,
                    vh_ub: 1,
                    st: 1,
                    h_ub: 1,
                    k_ub: 1,
                    v_ub: 1,
                    wh_ub: 1,
                    wv_ub: 1,
                    cw_ub: 1,
                }
            )

            # ---------- prologue: w_h^2 (fp32) and conv_w * rms_w_v (fp32) ------
            # SIMD_VF blocks cannot touch GM, so the three parameter tensors are
            # staged into UB first (T.copy is outside the frame).
            T.copy(WH, wh_ub[:d])
            T.copy(WV, wv_ub[:d])
            T.copy(CW, cw_ub[:, :d])
            with T.SimdVF():
                # causal halo slots start at zero (probe finding: T.clear on this
                # 12 KB fp32 buffer lowers to a 1536-iteration scalar float2 loop;
                # three vector chunk-loops are ~36 vsts instead).
                full_p = T.simd.pset(32)
                zero_p = T.simd.vdup(T.float32(0), fp32, full_p)
                for h in T.serial(KS - 1):
                    for ch in T.serial(nchunks):
                        T.simd.vsts(slot[h, ch * VL], zero_p, mask=full_p)
                for ch in T.serial(nchunks):
                    mask = T.simd.update_mask(T.min(VL, d - ch * VL), 32)
                    wv = T.simd.vcvt(T.simd.vld(wh_ub[ch * VL], dist="UNPK_B16"), fp32, part=0)
                    T.simd.vsts(w2h_ub[ch * VL], T.simd.vmul(wv, wv, mask))
                for p in T.serial(KS):
                    for ch in T.serial(nchunks):
                        mask = T.simd.update_mask(T.min(VL, d - ch * VL), 32)
                        cf = T.simd.vcvt(T.simd.vld(cw_ub[p, ch * VL], dist="UNPK_B16"), fp32, part=0)
                        vf = T.simd.vcvt(T.simd.vld(wv_ub[ch * VL], dist="UNPK_B16"), fp32, part=0)
                        T.simd.vsts(cwv_ub[p, ch * VL], T.simd.vmul(cf, vf, mask))

            # ---------- fused row pipeline --------------------------------------
            for i in T.serial(rc + 3):
                r = t0 - 3 + i
                if r >= 0:
                    T.copy(H[b, r, :], h_ub[:d])
                    T.copy(K[b, r, :], k_ub[:d])
                    T.copy(V[b, r, :], v_ub[:d])
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
                    if r >= 0:
                        hsq[0] = zero_f
                        ksq[0] = zero_f
                        hk[0] = zero_f
                        for ch in T.serial(nchunks):
                            mask = T.simd.update_mask(T.min(VL, d - ch * VL), 32)
                            hf = T.simd.vcvt(
                                T.simd.vld(h_ub[ch * VL], dist="UNPK_B16"), fp32, part=0
                            )
                            kf = T.simd.vcvt(
                                T.simd.vld(k_ub[ch * VL], dist="UNPK_B16"), fp32, part=0
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
                                T.simd.vdup(T.float32(1.0 / dim), fp32, full),
                                full,
                            ),
                            T.simd.vdup(T.float32(eps), fp32, full),
                            full,
                        )
                        rh_v = T.simd.vdiv(one_f, T.simd.vsqrt(hmean, full), full)
                        kmean = T.simd.vadd(
                            T.simd.vmul(
                                T.simd.vdupv(T.simd.vcadd(ksq[0], full)),
                                T.simd.vdup(T.float32(1.0 / dim), fp32, full),
                                full,
                            ),
                            T.simd.vdup(T.float32(eps), fp32, full),
                            full,
                        )
                        rk_v = T.simd.vdiv(one_f, T.simd.vsqrt(kmean, full), full)
                        logit = T.simd.vmul(
                            T.simd.vmul(T.simd.vdupv(T.simd.vcadd(hk[0], full)), rh_v, full),
                            T.simd.vmul(
                                rk_v, T.simd.vdup(T.float32(inv_sqrt_d), fp32, full), full
                            ),
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
                                T.simd.vld(v_ub[ch * VL], dist="UNPK_B16"), fp32, part=0
                            )
                            narrowed = T.simd.vcvt(
                                T.simd.vmul(vf, alpha_v[0], mask),
                                dtype,
                                round="ROUND_R",
                                sat=False,
                                part=0,
                            )
                            T.simd.vsts(vh_ub[ch * VL], narrowed, full, dist="PK_B32")
                            back = T.simd.vcvt(narrowed, fp32, part=0)
                            vsq[0] = T.simd.vadd(vsq[0], T.simd.vmul(back, back, mask), full)
                        vmean = T.simd.vadd(
                            T.simd.vmul(
                                T.simd.vdupv(T.simd.vcadd(vsq[0], full)),
                                T.simd.vdup(T.float32(1.0 / dim), fp32, full),
                                full,
                            ),
                            T.simd.vdup(T.float32(eps), fp32, full),
                            full,
                        )
                        rv_v[0] = T.simd.vdiv(one_f, T.simd.vsqrt(vmean, full), full)
                        T.simd.vsts(st[3, i, 0], rv_v[0], one, extent=1)
                        for ch in T.serial(nchunks):
                            back = T.simd.vcvt(
                                T.simd.vld(vh_ub[ch * VL], dist="UNPK_B16"), fp32, part=0
                            )
                            T.simd.vsts(
                                slot[i % 4, ch * VL], T.simd.vmul(back, rv_v[0]), full
                            )
                    if i >= 3:
                        for ch in T.serial(nchunks):
                            mask = T.simd.update_mask(T.min(VL, d - ch * VL), 32)
                            acc = T.simd.alloc_local((1,), fp32)
                            acc[0] = zero_f
                            for p in T.unroll(KS):
                                acc[0] = T.simd.vadd(
                                    acc[0],
                                    T.simd.vmul(
                                        T.simd.vld(slot[(i - 3 + p) % 4, ch * VL]),
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
                                T.simd.vld(v_ub[ch * VL], dist="UNPK_B16"), fp32, part=0
                            )
                            y = T.simd.vadd(silu, T.simd.vmul(vf, alpha_v[0], mask), mask)
                            T.simd.vsts(
                                y_ub[ch * VL],
                                T.simd.vcvt(y, dtype, round="ROUND_R", sat=False, part=0),
                                full,
                                dist="PK_B32",
                            )
                if i >= 3:
                    T.copy(vh_ub[:d], VHAT[b, r, :])
                    T.copy(y_ub[:d], Y[b, r, :])

            # ---------- batched scalar write-back -------------------------------
            T.copy(st[0, 3 : 3 + rc, 0:1], Alpha[b, t0 : t0 + rc])
            T.copy(st[1, 3 : 3 + rc, 0:1], RH[b, t0 : t0 + rc])
            T.copy(st[2, 3 : 3 + rc, 0:1], RK[b, t0 : t0 + rc])
            T.copy(st[3, 3 : 3 + rc, 0:1], RV[b, t0 : t0 + rc])

    return tilelang.compile(main, target="ascend", out_idx=[6, 7, 8, 9, 10, 11])


CASES = [
    ("train-s96", 3, 96, 768),
    ("train-s64", 2, 64, 512),
    ("train-s128", 1, 128, 256),
    ("train-s32", 1, 32, 256),
    ("prime-95", 1, 95, 256),
    ("pad-d196", 1, 8, 196),
]


def main():
    bad = 0
    for name, M, seq, d in CASES:
        try:
            fn = build(M, seq, d, "bfloat16", 1e-6)
            print(f"[{name}] M={M} seq={seq} d={d} geo={geometry(M, seq, d)} COMPILE OK", flush=True)
            del fn
        except Exception as exc:  # noqa: BLE001 - the probe reports and continues
            bad += 1
            print(f"[{name}] M={M} seq={seq} d={d} COMPILE FAIL: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc(limit=4)
    print("PROBE_RESULT:", "ALL_OK" if bad == 0 else f"{bad} FAILED")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
