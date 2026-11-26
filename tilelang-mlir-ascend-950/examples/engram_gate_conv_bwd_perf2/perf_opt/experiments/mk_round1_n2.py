"""Generate perf_opt/engram_gate_conv_bwd_round1_n2.py from the round-0 best.

Single optimization point (n2 / O-13): the per-iteration row loop is split into
three phases so the convolution recompute and the ``dconv_w`` partial become ONE
chunk-major sweep that shares the four causal taps, keeps the four conv weights
and the four accumulators in registers, and needs no ``floormod`` ring
arithmetic.

  phase 1  ``vhat_norm`` per window row          (row-major,  1 sweep / row)
  phase 2  ``d_silu`` + ``dconv_w`` partial       (CHUNK-major, fused)
  phase 3  value / gate backward per token        (token-major, 4 passes)

Cheaper than the interleaved loop because
  * ``cwf`` is loaded once per chunk instead of once per (row, chunk);
  * the ``dconv_w`` partial reuses the nrm taps the conv sweep already holds
    (a second sweep used to load them again) and keeps its accumulators in
    registers across the owned rows;
  * every index is a plain row offset (no ring ``floormod``).

Every per-lane accumulation order is unchanged (conv taps 0..3, ``dconv_w`` owned
rows ascending with taps 0..3, stage-2 passes A/B/C1/C2 in order), so the six
outputs stay bit-identical to the round-0 kernel.
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
OPT = os.path.dirname(HERE)
BASE = os.path.join(OPT, "engram_gate_conv_bwd_round0_best.py")
OUT = os.path.join(OPT, "engram_gate_conv_bwd_round1_n2.py")

s = open(BASE).read()

i0 = s.index(
    """            # =================================================================
            # Fused loop: iteration i publishes d_silu for row pn = t0 - 3 + i"""
)
i1 = s.index(
    """            # -------- r1_c2: bulk store of the dH / dk / dv window ----------------
"""
)
body = s[i0:i1]
c0 = body.index(
    """                # ---- (c) value / gate backward for token tp = t0 + i - 6 ------
"""
)

# ---- rebuild the (c) block as phase 3: drop the ``if i >= 2*LEAD`` level and the
# ``i``-derived indices, and de-indent its body by one level.
raw = body[c0:]
lines = raw.splitlines()
assert lines[1].strip() == "if i >= 2 * LEAD:", lines[1]
out_lines = [
    "                # ---- (c) value / gate backward for token tp = t0 + k ----------",
    "                tp = t0 + k",
]
for line in lines[2:]:
    if line.strip() == "tp = t0 + i - 2 * LEAD":
        continue
    line = line.replace("# r1_c2: the H / k / v rows are already resident; the window",
                        "# the H / k / v rows are already resident; the window")
    line = line.replace("# slot of token ``tp`` is ``i - 2*LEAD`` (``tp - t0``).",
                        "# slot of token ``tp`` is ``k`` (``tp - t0``).")
    out_lines.append(line[4:] if line.startswith("    ") else line)
stage2 = "\n".join(out_lines).rstrip() + "\n"
assert "if tp < seq_len:" in stage2 and "widx_c = k" not in stage2
stage2 = stage2.replace("widx_c = i - 2 * LEAD", "widx_c = k")
# every remaining ``i``-derived index becomes a plain ``k`` offset:
#   token tp = t0 + k, so the staging slot of tp is W(tp) = (tp - t0) + LEAD = k + LEAD
#   and pass A's d_silu taps are d_silu[tp + KS-1-p] -> k + LEAD - p.
stage2 = stage2.replace("dsw_all[i - LEAD - p, ch * VL]", "dsw_all[k + LEAD - p, ch * VL]")
stage2 = stage2.replace("vstg[i - LEAD, ch * VL]", "vstg[k + LEAD, ch * VL]")
stage2 = stage2.replace("dstg[i - LEAD, ch * VL]", "dstg[k + LEAD, ch * VL]")
assert " i " not in stage2.replace(" i = ", " "), [l for l in stage2.splitlines() if " i " in l][:3]

new = '''            # =================================================================
            # r1_n2 / O-13: three-phase structure (see the module docstring).
            # =================================================================

            # ---- phase 1: vhat_norm of every window row (index i = row - t0 + LEAD)
            for i in T.serial(nwin):
                pn = t0 - LEAD + i
                if pn >= 0:
                    if pn < seq_len:
                        with T.SimdVF():
                            full = T.simd.pset(32)
                            rv_p = T.simd.vld(sc[3, pn], dist="BRC_B32")
                            for ch in T.serial(nchunks):
                                mask = T.simd.update_mask(T.min(VL, d - ch * VL), 32)
                                z = T.simd.vcvt(
                                    T.simd.vld(vstg[i, ch * VL], dist="UNPK_B16"),
                                    FP32,
                                    mask,
                                    part=0,
                                )
                                T.simd.vsts(
                                    nrm_all[i, ch * VL],
                                    T.simd.vmul(
                                        T.simd.vmul(z, rv_p, mask),
                                        T.simd.vld(wvf[ch * VL]),
                                        mask,
                                    ),
                                    full,
                                )
                    if pn >= seq_len:  # past the batch tail: zero row, no compute
                        with T.SimdVF():
                            full = T.simd.pset(32)
                            zero = T.simd.vdup(T.float32(0), FP32, full)
                            for ch in T.serial(nchunks):
                                T.simd.vsts(nrm_all[i, ch * VL], zero, full)
                if pn < 0:  # before the batch head: zero row, no compute
                    with T.SimdVF():
                        full = T.simd.pset(32)
                        zero = T.simd.vdup(T.float32(0), FP32, full)
                        for ch in T.serial(nchunks):
                            T.simd.vsts(nrm_all[i, ch * VL], zero, full)

            # ---- phase 2: d_silu + dconv_w partial, chunk-major and fused.
            # Rows past the batch tail are pre-zeroed (the causal taps of phase 3
            # must read an exact 0), then only the in-range rows are computed.
            with T.SimdVF():
                full = T.simd.pset(32)
                zero = T.simd.vdup(T.float32(0), FP32, full)
                for r in T.serial(rc + LEAD):
                    for ch in T.serial(nchunks):
                        T.simd.vsts(dsw_all[r, ch * VL], zero, full)
            with T.SimdVF():
                # phase 1 vector-stored nrm_all and the prologue zeroed pc:
                # vector store -> vector load needs an explicit barrier.
                T.simd.mem_bar("VST_VLD")
                full = T.simd.pset(32)
                one = T.simd.vdup(T.float32(1), FP32, full)
                for ch in T.serial(nchunks):
                    mask = T.simd.update_mask(T.min(VL, d - ch * VL), 32)
                    # the four conv weights and the four dconv_w accumulators live
                    # in registers for the whole row sweep of this chunk
                    c0 = T.simd.alloc_local((1,), FP32)
                    c1 = T.simd.alloc_local((1,), FP32)
                    c2 = T.simd.alloc_local((1,), FP32)
                    c3 = T.simd.alloc_local((1,), FP32)
                    c0[0] = T.simd.vld(cwf[0, ch * VL])
                    c1[0] = T.simd.vld(cwf[1, ch * VL])
                    c2[0] = T.simd.vld(cwf[2, ch * VL])
                    c3[0] = T.simd.vld(cwf[3, ch * VL])
                    q0 = T.simd.alloc_local((1,), FP32)
                    q1 = T.simd.alloc_local((1,), FP32)
                    q2 = T.simd.alloc_local((1,), FP32)
                    q3 = T.simd.alloc_local((1,), FP32)
                    q0[0] = T.simd.vld(pc[0, ch * VL])
                    q1[0] = T.simd.vld(pc[1, ch * VL])
                    q2[0] = T.simd.vld(pc[2, ch * VL])
                    q3[0] = T.simd.vld(pc[3, ch * VL])
                    for r in T.serial(rc + LEAD):
                        if t0 + r < seq_len:
                            n0 = T.simd.alloc_local((1,), FP32)
                            n1 = T.simd.alloc_local((1,), FP32)
                            n2 = T.simd.alloc_local((1,), FP32)
                            n3 = T.simd.alloc_local((1,), FP32)
                            n0[0] = T.simd.vld(nrm_all[r, ch * VL])
                            n1[0] = T.simd.vld(nrm_all[r + 1, ch * VL])
                            n2[0] = T.simd.vld(nrm_all[r + 2, ch * VL])
                            n3[0] = T.simd.vld(nrm_all[r + 3, ch * VL])
                            conv = T.simd.alloc_local((1,), FP32)
                            conv[0] = T.simd.vdup(T.float32(0), FP32, full)
                            conv[0] = T.simd.vadd(
                                conv[0], T.simd.vmul(n0[0], c0[0], mask), full
                            )
                            conv[0] = T.simd.vadd(
                                conv[0], T.simd.vmul(n1[0], c1[0], mask), full
                            )
                            conv[0] = T.simd.vadd(
                                conv[0], T.simd.vmul(n2[0], c2[0], mask), full
                            )
                            conv[0] = T.simd.vadd(
                                conv[0], T.simd.vmul(n3[0], c3[0], mask), full
                            )
                            sig = T.simd.vdiv(
                                one,
                                T.simd.vadd(
                                    one,
                                    T.simd.vexp(T.simd.vneg(conv[0], mask), mask),
                                    mask,
                                ),
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
                            dsv = T.simd.alloc_local((1,), FP32)
                            dsv[0] = T.simd.vmul(
                                T.simd.vcvt(
                                    T.simd.vld(
                                        dstg[r + LEAD, ch * VL], dist="UNPK_B16"
                                    ),
                                    FP32,
                                    mask,
                                    part=0,
                                ),
                                deriv,
                                mask,
                            )
                            T.simd.vsts(dsw_all[r, ch * VL], dsv[0], full)
                            if r < rc:  # only the run that owns the row accumulates
                                q0[0] = T.simd.vadd(
                                    q0[0], T.simd.vmul(dsv[0], n0[0], mask), full
                                )
                                q1[0] = T.simd.vadd(
                                    q1[0], T.simd.vmul(dsv[0], n1[0], mask), full
                                )
                                q2[0] = T.simd.vadd(
                                    q2[0], T.simd.vmul(dsv[0], n2[0], mask), full
                                )
                                q3[0] = T.simd.vadd(
                                    q3[0], T.simd.vmul(dsv[0], n3[0], mask), full
                                )
                    T.simd.vsts(pc[0, ch * VL], q0[0], full)
                    T.simd.vsts(pc[1, ch * VL], q1[0], full)
                    T.simd.vsts(pc[2, ch * VL], q2[0], full)
                    T.simd.vsts(pc[3, ch * VL], q3[0], full)

            # ---- phase 3: value / gate backward, one token per step
            for k in T.serial(rc):
'''

s = s[:i0] + new + stage2 + s[i1:]

# the old chunk-major dconv sweep is folded into phase 2 now
d0 = s.index(
    """            # -------- r4_c7: dconv_w partial, chunk-major ------------------------"""
)
d1 = s.index(
    """            # -------- per-core partials -> GM workspace (DESIGN 0.6 R1) ---------
"""
)
s = s[:d0] + s[d1:]

with open(OUT, "w") as f:
    f.write(s)
print(f"wrote {OUT}: {len(s.splitlines())} lines")
