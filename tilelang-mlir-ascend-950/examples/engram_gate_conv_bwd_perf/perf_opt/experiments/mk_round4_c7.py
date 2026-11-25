"""Generate perf_opt/engram_gate_conv_bwd_round4_c7.py from the r3_c6 base.

Single optimization point (r4_c7 / O-12): the dconv_w partial accumulation leaves
the row loop and becomes a chunk-major sweep with its four accumulators held in
registers across the owned rows.  Mechanism:

  * the row loop currently does ``pc[p] = vld(pc[p]) + ds*nrm[p]; vsts(pc[p])``
    for all four taps of every owned row (4 loads + 4 stores per row-chunk);
  * chunk-major, ``pc[p]`` is loaded once per chunk and stored once per chunk
    (8 memory ops per chunk instead of 8 per owned row), i.e. -8 memory ops per
    owned (row, chunk) and the per-chunk ``cwf`` loads are amortised over the
    rows;
  * it needs the nrm / d_silu rows to stay resident for the block instead of a
    4-slot ring, which this variant allocates (``nrm_all`` / ``dsw_all``); the
    ring slot expressions disappear with it.

The per-lane addition order is unchanged (owned rows ascending, taps 0..3 per
row), so the result stays bit-identical to the base.
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
OPT = os.path.dirname(HERE)
BASE = os.path.join(OPT, "engram_gate_conv_bwd_round3_c6.py")
OUT = os.path.join(OPT, "engram_gate_conv_bwd_round4_c7.py")

s = open(BASE).read()


def rep(buf, old, new, tag=""):
    global s
    assert buf is s
    n = s.count(old)
    assert n == 1, (tag, n)
    s = s.replace(old, new)
    print("  patched:", tag)


# ---------------------------------------------------------------- buffer decls
rep(
    s,
    """            # --- fp32 four-slot ring windows, slot = floormod(iteration, KS)
            nrmw = T.alloc_shared((KS, dw), FP32)  # vhat_norm (DESIGN 1.6.1 O5)
            zraw = T.alloc_shared((KS, dw), FP32)  # raw vhat (see pass A)
            dyw = T.alloc_shared((KS, dw), FP32)  # dY
            dsw = T.alloc_shared((KS, dw), FP32)  # d_silu
""",
    """            # --- fp32 ring windows for the raw vhat / dY rows, slot = floormod(i, KS)
            zraw = T.alloc_shared((KS, dw), FP32)  # raw vhat (see pass A)
            dyw = T.alloc_shared((KS, dw), FP32)  # dY
            # r4_c7: vhat_norm and d_silu stay resident for the whole block (window
            # row ``w`` holds token ``t0 - LEAD + w``, resp. ``t0 + w``), which is what
            # lets the dconv_w partial run chunk-major below.
            nrm_all = T.alloc_shared((rc + 2 * LEAD, dw), FP32)  # vhat_norm
            dsw_all = T.alloc_shared((rc + LEAD, dw), FP32)  # d_silu
""",
    "decls",
)

rep(
    s,
    """                    nrmw: 1,
                    zraw: 1,
                    dyw: 1,
                    dsw: 1,
""",
    """                    nrm_all: 1,
                    zraw: 1,
                    dyw: 1,
                    dsw_all: 1,
""",
    "versions",
)

# ------------------------------------------------------------------- (a) store
s = s.replace("nrmw[slot, ch * VL]", "nrm_all[i, ch * VL]")
print("  (a) store/zero -> nrm_all[i]")

# --------------------------------------------------------------- (b) taps/store
rep(
    s,
    """                                            T.simd.vld(
                                                nrmw[T.floormod(i - LEAD + p, KS), ch * VL]
                                            ),""",
    """                                            T.simd.vld(
                                                nrm_all[i - LEAD + p, ch * VL]
                                            ),""",
    "b_taps",
)
rep(
    s,
    """                                T.simd.vsts(
                                    dsw[slot, ch * VL],
                                    T.simd.vmul(
                                        T.simd.vld(dyw[slot, ch * VL]), deriv, mask
                                    ),
                                    full,
                                )
""",
    """                                T.simd.vsts(
                                    dsw_all[i - LEAD, ch * VL],
                                    T.simd.vmul(
                                        T.simd.vld(dyw[slot, ch * VL]), deriv, mask
                                    ),
                                    full,
                                )
""",
    "b_store",
)
rep(
    s,
    """                        with T.SimdVF():
                            full = T.simd.pset(32)
                            zero = T.simd.vdup(T.float32(0), FP32, full)
                            for ch in T.serial(nchunks):
                                T.simd.vsts(dsw[slot, ch * VL], zero, full)
""",
    """                        with T.SimdVF():
                            full = T.simd.pset(32)
                            zero = T.simd.vdup(T.float32(0), FP32, full)
                            for ch in T.serial(nchunks):
                                T.simd.vsts(dsw_all[i - LEAD, ch * VL], zero, full)
""",
    "b_zero",
)

# ------------------------------------------------- drop the in-loop dconv block
i0 = s.index("""                    # ---- dconv_w partial: only the run that owns the row -----
""")
i1 = s.index("""                # ---- (c) value / gate backward for token tp = t0 + i - 6 ------
""")
s = s[:i0] + s[i1:]
print("  removed in-loop dconv block")

# --------------------------------------------- add the chunk-major dconv sweep
rep(
    s,
    """            # -------- per-core partials -> GM workspace (DESIGN 0.6 R1) ---------
""",
    """            # -------- r4_c7: dconv_w partial, chunk-major ------------------------
            # ``pc[p, j] = sum over owned rows r of d_silu[r, j] * vhat_norm[r + p, j]``
            # accumulated in four registers per chunk (loaded/stored once per chunk
            # instead of once per row).  Rows ascend and taps run 0..3 per row, which
            # is the base accumulation order, so the partials stay bit-identical.
            with T.SimdVF():
                full = T.simd.pset(32)
                for ch in T.serial(nchunks):
                    mask = T.simd.update_mask(T.min(VL, d - ch * VL), 32)
                    q0 = T.simd.alloc_local((1,), FP32)
                    q1 = T.simd.alloc_local((1,), FP32)
                    q2 = T.simd.alloc_local((1,), FP32)
                    q3 = T.simd.alloc_local((1,), FP32)
                    q0[0] = T.simd.vld(pc[0, ch * VL])
                    q1[0] = T.simd.vld(pc[1, ch * VL])
                    q2[0] = T.simd.vld(pc[2, ch * VL])
                    q3[0] = T.simd.vld(pc[3, ch * VL])
                    for r in T.serial(rc):
                        dsv = T.simd.alloc_local((1,), FP32)
                        dsv[0] = T.simd.vld(dsw_all[r, ch * VL])
                        q0[0] = T.simd.vadd(
                            q0[0],
                            T.simd.vmul(
                                dsv[0], T.simd.vld(nrm_all[r, ch * VL]), mask
                            ),
                            full,
                        )
                        q1[0] = T.simd.vadd(
                            q1[0],
                            T.simd.vmul(
                                dsv[0], T.simd.vld(nrm_all[r + 1, ch * VL]), mask
                            ),
                            full,
                        )
                        q2[0] = T.simd.vadd(
                            q2[0],
                            T.simd.vmul(
                                dsv[0], T.simd.vld(nrm_all[r + 2, ch * VL]), mask
                            ),
                            full,
                        )
                        q3[0] = T.simd.vadd(
                            q3[0],
                            T.simd.vmul(
                                dsv[0], T.simd.vld(nrm_all[r + 3, ch * VL]), mask
                            ),
                            full,
                        )
                    T.simd.vsts(pc[0, ch * VL], q0[0], full)
                    T.simd.vsts(pc[1, ch * VL], q1[0], full)
                    T.simd.vsts(pc[2, ch * VL], q2[0], full)
                    T.simd.vsts(pc[3, ch * VL], q3[0], full)

            # -------- per-core partials -> GM workspace (DESIGN 0.6 R1) ---------
""",
    "dconv sweep",
)

# ------------------------------------------------- stage-2 d_silu tap indices
rep(
    s,
    """                                            T.simd.vld(
                                                dsw[T.floormod(i - p, KS), ch * VL]
                                            ),""",
    """                                            T.simd.vld(
                                                dsw_all[i - LEAD - p, ch * VL]
                                            ),""",
    "passA_taps",
)

# ------------------------------------- drop the zraw / dyw fp32 rings (UB budget)
# The two 4-slot fp32 rings of the raw vhat / dY rows are redundant now that the
# bulk dtype staging window is resident: those rows are converted at the point of
# use instead.  This is what keeps the resident nrm / d_silu windows inside the UB
# budget (see the footprint check).
rep(
    s,
    """            # --- fp32 ring windows for the raw vhat / dY rows, slot = floormod(i, KS)
            zraw = T.alloc_shared((KS, dw), FP32)  # raw vhat (see pass A)
            dyw = T.alloc_shared((KS, dw), FP32)  # dY
""",
    "",
    "drop_rings",
)
rep(
    s,
    """                    zraw: 1,
                    dyw: 1,
""",
    "",
    "drop_ring_versions",
)
rep(
    s,
    """                                T.simd.vsts(zraw[slot, ch * VL], z, full)
                                T.simd.vsts(dyw[slot, ch * VL], dyv, full)
""",
    "",
    "a_ring_stores",
)
rep(
    s,
    """                                dyv = T.simd.vcvt(
                                    T.simd.vld(dstg[i, ch * VL], dist="UNPK_B16"),
                                    FP32,
                                    mask,
                                    part=0,
                                )
""",
    "",
    "a_dyv",
)
rep(
    s,
    """                            for ch in T.serial(nchunks):
                                T.simd.vsts(nrm_all[i, ch * VL], zero, full)
                                T.simd.vsts(zraw[slot, ch * VL], zero, full)
                                T.simd.vsts(dyw[slot, ch * VL], zero, full)
""",
    """                            for ch in T.serial(nchunks):
                                T.simd.vsts(nrm_all[i, ch * VL], zero, full)
""",
    "a_zero1",
)
rep(
    s,
    """                        for ch in T.serial(nchunks):
                            T.simd.vsts(nrm_all[i, ch * VL], zero, full)
                            T.simd.vsts(zraw[slot, ch * VL], zero, full)
                            T.simd.vsts(dyw[slot, ch * VL], zero, full)
""",
    """                        for ch in T.serial(nchunks):
                            T.simd.vsts(nrm_all[i, ch * VL], zero, full)
""",
    "a_zero2",
)
rep(
    s,
    """                                    T.simd.vmul(
                                        T.simd.vld(dyw[slot, ch * VL]), deriv, mask
                                    ),
""",
    """                                    T.simd.vmul(
                                        T.simd.vcvt(
                                            T.simd.vld(
                                                dstg[i, ch * VL], dist="UNPK_B16"
                                            ),
                                            FP32,
                                            mask,
                                            part=0,
                                        ),
                                        deriv,
                                        mask,
                                    ),
""",
    "b_dy",
)
rep(
    s,
    """                                zd = T.simd.vmul(
                                    T.simd.vld(
                                        zraw[T.floormod(i - LEAD, KS), ch * VL]
                                    ),
                                    dn[0],
                                    mask,
                                )
""",
    """                                zd = T.simd.vmul(
                                    T.simd.vcvt(
                                        T.simd.vld(
                                            vstg[i - LEAD, ch * VL], dist="UNPK_B16"
                                        ),
                                        FP32,
                                        mask,
                                        part=0,
                                    ),
                                    dn[0],
                                    mask,
                                )
""",
    "passA_z",
)
rep(
    s,
    """                                z = T.simd.vld(zraw[T.floormod(i - LEAD, KS), ch * VL])
""",
    """                                z = T.simd.vcvt(
                                    T.simd.vld(
                                        vstg[i - LEAD, ch * VL], dist="UNPK_B16"
                                    ),
                                    FP32,
                                    mask,
                                    part=0,
                                )
""",
    "pass_z",
)
rep(
    s,
    """                                    T.simd.vld(
                                        dyw[T.floormod(i - LEAD, KS), ch * VL]
                                    ),
""",
    """                                    T.simd.vcvt(
                                        T.simd.vld(
                                            dstg[i - LEAD, ch * VL], dist="UNPK_B16"
                                        ),
                                        FP32,
                                        mask,
                                        part=0,
                                    ),
""",
    "passB_dy",
)

# ------------------------------------------------------------------ UB footprint
rep(
    s,
    """    fp32_rows = 4 * 6 + 7  # nrmw/zraw/dyw/dsw/cwf/pc + ph/pv/gn/hnf/knf/whf/wvf
""",
    """    # nrm_all(rc+6) + dsw_all(rc+3) + cwf/pc (2*KS) + ph/pv/gn/hnf/knf/whf/wvf
    fp32_rows = (rc + 2 * LEAD) + (rc + LEAD) + 2 * KS + 7
""",
    "fp32_rows",
)

with open(OUT, "w") as f:
    f.write(s)
print(f"wrote {OUT}")
