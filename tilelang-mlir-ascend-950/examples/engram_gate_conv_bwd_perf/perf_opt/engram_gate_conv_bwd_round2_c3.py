# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Engram GateConv full backward for Ascend 950PR (SIMD / AIV).

Two launches, following DESIGN.md 0.6 / 1.4 / 3.3:

  launch #1  one fused compute kernel, one AIV per run of ``RC`` consecutive
             tokens of one batch row.  Iteration ``i`` of the run (a) loads row
             ``pn = t0 - 3 + i`` of ``vhat`` / ``dY`` and publishes
             ``vhat_norm`` / raw ``vhat`` / ``dY`` into four-slot UB ring
             windows, (b) recomputes the causal width-4 convolution and
             ``d_silu`` for the row it just loaded and accumulates the
             ``dconv_w`` partial of the rows it owns, (c) runs the whole
             value/gate backward (inverse convolution, RMSNorm(value) backward,
             gate backward, the two RMSNorm(hidden) halves) for the token that
             is three rows behind.  ``d_silu``, ``dvhat`` and ``dY`` never
             touch GM.
  launch #2  ordered reduction of the per-core fp32 partials
             ``PART[6, G, dw]`` summed in ascending core order, so the three
             weight gradients are deterministic.  No atomics, no zero launch.

Six outputs: ``dH``, ``dk``, ``dv`` (input dtype) and the float32
``drms_w_h``, ``drms_w_v``, ``dconv_w``.

Two toolchain rules this kernel has to respect (both have a repro under the
sibling forward task ``examples/engram_gate_conv/_engram_gate_conv_fwd_kernel/
repro/``): a vector store to UB is only visible to a later vector load of the
same buffer after ``T.simd.mem_bar("VST_VLD")``, and a scalar broadcast must be
read back through ``vld(..., dist="BRC_B32")``.

Deviation from DESIGN.md 3.3, implementation-level only, no semantic change:
the ring slot is keyed on the iteration counter ``i`` (``floormod(i, 4)``)
rather than on the row index, so every slot expression is provably
non-negative -- the row index ``pn = t0 - 3 + i`` is negative for the first
three iterations of the first run and ``floormod`` of a negative value is the
one index this kernel must not depend on.  Slot of row ``pn`` is
``floormod(i, 4)``, slot of row ``pn - k`` is ``floormod(i - k, 4)``.
A second deviation: the raw ``vhat`` row is kept in its own ring window
(``zraw``) instead of being recovered from ``vhat_norm`` (see the note at
pass A).

run: python _engram_gate_conv_bwd_kernel.py --level all
"""

import argparse
import functools
import time

import torch
import torch.nn.functional as F

import tilelang
import tilelang.ascend.language as T
import torch_npu  # noqa: F401  # registers torch.npu

# ---------------------------------------------------------------------------
# Constants (DESIGN.md 5.2 / 5.5 / 8.2)
# ---------------------------------------------------------------------------
VL = 64  # fp32 vector register width; also the 16-bit chunk mapped by UNPK_B16/PK_B32
KS = 4  # causal depthwise convolution taps
LEAD = KS - 1  # rows by which d_silu leads the token whose gradients are due
CORES = 56  # Ascend950PR_9579 vector cores (statectl platform observed_cores.vector)
# DESIGN.md 9.2 R-1: the fused structure holds 152 B per SIMD-padded channel plus
# 16 B per sequence position.  d <= 1024 leaves >= 35 % of the 248 KiB UB free at
# seq_len = 512; the manifest's d <= 4096 needs the staged fallback (DESIGN 1.6.4
# B1) which is not implemented here, so the refusal below must stay at 1024.
MAX_HIDDEN_DIM = 1024
UB_BYTES = 253952  # Ascend950PR_9579.ini [AICoreSpec] ub_size
FP32 = "float32"
ELEM_BYTES = {"float16": 2, "bfloat16": 2}

OUT_NAMES = ("dH", "dk", "dv", "drms_w_h", "drms_w_v", "dconv_w")


# ---------------------------------------------------------------------------
# 1. Golden (PyTorch CPU reference, ported from
#    examples/TileOPs/tileops/workloads/sequence_modeling.py)
# ---------------------------------------------------------------------------
def _rmsnorm(x, w, eps=1e-6):
    """fp32 statistics + fp32 normalisation; returns (normed, rrms)."""
    x_f = x.float()
    rrms = (x_f**2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    return x_f * rrms * w.float(), rrms.squeeze(-1)


def golden_engram_gate_conv_bwd(
    dY, H, k, v, rms_w_h, rms_w_v, conv_w, vhat, alpha, rrms_h, rrms_k, rrms_v, eps=1e-6
):
    """fp32 autograd reference (the source op's own oracle path).

    ``vhat / alpha / rrms_*`` are the forward's saved tensors and are
    deliberately unused: the reference differentiates the float32 forward over
    the six differentiable inputs, so it is independent of every NPU structure
    (no window, no per-core partial, no ordered reduction).

    Deliberate difference from the kernel: the kernel consumes the forward's
    *rounded* ``vhat`` (the op contract), while this reference re-derives the
    unrounded fp32 path.  DESIGN.md 8.1/9.2 R-3 quantifies that gap at 8.9e-3
    (fp16) / 5.4e-2 (bf16) of the field scale, i.e. far inside the op tolerance
    (_tol_bwd: fp16 0.2/0.2, bf16 0.3/0.2).
    """
    H_ag = H.float().detach().requires_grad_(True)
    k_ag = k.float().detach().requires_grad_(True)
    v_ag = v.float().detach().requires_grad_(True)
    w_h_ag = rms_w_h.float().detach().requires_grad_(True)
    w_v_ag = rms_w_v.float().detach().requires_grad_(True)
    cw_ag = conv_w.float().detach().requires_grad_(True)

    h_norm, _ = _rmsnorm(H_ag, w_h_ag, eps)  # H and k share rms_w_h
    k_norm, _ = _rmsnorm(k_ag, w_h_ag, eps)

    dot = (h_norm * k_norm).sum(dim=-1, keepdim=True)
    alpha_ag = torch.sigmoid(dot / (H.shape[-1] ** 0.5))
    v_hat_ag = alpha_ag * v_ag
    v_hat_norm, _ = _rmsnorm(v_hat_ag, w_v_ag, eps)

    v_perm = v_hat_norm.permute(0, 2, 1)  # (M, d, T)
    v_padded = F.pad(v_perm, (KS - 1, 0))  # causal left pad, dilation 1
    cw_expanded = cw_ag.T.unsqueeze(1)  # depthwise (d, 1, KS)
    conv_out = F.conv1d(v_padded, cw_expanded, groups=H.shape[-1]).permute(0, 2, 1)
    y = F.silu(conv_out) + v_hat_ag

    with torch.autograd.set_multithreading_enabled(False):
        y.backward(dY.float())

    return (
        H_ag.grad.to(H.dtype),
        k_ag.grad.to(k.dtype),
        v_ag.grad.to(v.dtype),
        w_h_ag.grad,
        w_v_ag.grad,
        cw_ag.grad,
    )


# ---------------------------------------------------------------------------
# 2. Build-time geometry (DESIGN.md 5.2 / 5.5) and UB budget (DESIGN.md 4.5)
# ---------------------------------------------------------------------------
def geometry(M, seq_len, d):
    """Chunk length and launch grid: ``grid = M * nch <= 56`` by construction."""
    dw = -(-d // VL) * VL
    nch = min(seq_len, max(1, CORES // M))
    rc = -(-seq_len // nch)
    nch = -(-seq_len // rc)
    return {"dw": dw, "nchunks": dw // VL, "RC": rc, "nch": nch, "grid": M * nch}


def reduce_layout(dw, d):
    """Column tiling of launch #2: ``6 * tiles <= 54`` cores, ``width`` a VL multiple."""
    tiles = min(9, max(1, -(-dw // (VL * 9))))
    width = -(-(-(-dw // tiles)) // VL) * VL
    tiles = -(-dw // width)
    return width, tiles


def _ub_footprint(dw, elem, seq_len, rc):
    """152 B per channel + 16 B per position (DESIGN.md 9.2 R-1 recomputed).

    r1_c1: the vhat / dY stream is staged as a ``(rc + 2*LEAD, dw)`` dtype window
    per block instead of one row each, so the dtype rows grow with ``rc``.
    """
    fp32_rows = 4 * 6 + 7  # nrmw/zraw/dyw/dsw/cwf/pc + ph/pv/gn/hnf/knf/whf/wvf
    # vhat/dY windows + H/k/v + dH/dk/dv windows + whs/wvs + cws
    dtype_rows = 2 * (rc + 2 * LEAD) + 6 * rc + 2 + 4
    seq_pitch = max(8, -(-seq_len // 8) * 8)
    return fp32_rows * dw * 4 + dtype_rows * dw * elem + 4 * seq_pitch * 4


def _check(M, seq_len, d, dtype):
    if dtype not in ELEM_BYTES:
        raise ValueError(f"dtype must be float16 or bfloat16, got {dtype!r}")
    if M <= 0 or seq_len <= 0 or d <= 0:
        raise ValueError("M, seq_len and d must be positive")
    if d > MAX_HIDDEN_DIM:
        raise ValueError(
            f"d={d} exceeds the hidden-dim refusal of {MAX_HIDDEN_DIM} "
            "(the fused kernel holds 152 B per padded channel; DESIGN 9.2 R-1)"
        )
    geo = geometry(M, seq_len, d)
    footprint = _ub_footprint(geo["dw"], ELEM_BYTES[dtype], seq_len, geo["RC"])
    if footprint * 1.1 > UB_BYTES:
        raise ValueError(
            f"UB footprint {footprint} B x1.1 exceeds {UB_BYTES} B "
            f"(d={d}, seq_len={seq_len}, RC={geo['RC']})"
        )
    return geo


# ---------------------------------------------------------------------------
# 3. Launch #1: fused compute kernel (DESIGN.md 1.4 / 3.3 / 6.2)
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=32)
def _build_fused(M: int, seq_len: int, d: int, dtype: str):
    """Compile the fused backward for one ``(M, seq_len, d, dtype)`` geometry."""
    geo = _check(M, seq_len, d, dtype)
    dw = geo["dw"]
    nchunks = geo["nchunks"]
    rc = geo["RC"]
    nch = geo["nch"]
    grid = geo["grid"]
    # 32 B granularity for the strip rows: both the pitch and the alignment of
    # every row must be a multiple of 8 fp32 elements (seq_len = 401 -> 408).
    seq_pitch = max(8, -(-seq_len // 8) * 8)
    dim = float(d)
    inv_d = 1.0 / dim  # DESIGN 1.6.1 O1: /d -> * (1/d)
    inv_sqrt_d = 1.0 / (d**0.5)  # DESIGN 1.6.1 O1: /sqrt(d) -> * (1/sqrt(d))

    @T.prim_func
    def main(
        DY: T.Tensor((M, seq_len, d), dtype),
        H: T.Tensor((M, seq_len, d), dtype),
        K: T.Tensor((M, seq_len, d), dtype),
        V: T.Tensor((M, seq_len, d), dtype),
        WH: T.Tensor((d,), dtype),
        WV: T.Tensor((d,), dtype),
        CW: T.Tensor((KS, d), dtype),
        VHAT: T.Tensor((M, seq_len, d), dtype),
        ALPHA: T.Tensor((M, seq_len), FP32),
        RH: T.Tensor((M, seq_len), FP32),
        RK: T.Tensor((M, seq_len), FP32),
        RV: T.Tensor((M, seq_len), FP32),
        DH: T.Tensor((M, seq_len, d), dtype),
        DK: T.Tensor((M, seq_len, d), dtype),
        DV: T.Tensor((M, seq_len, d), dtype),
        PART: T.Tensor((2 + KS, grid, dw), FP32),
    ):
        with T.Kernel(grid) as bx:
            b = bx // nch
            c = bx % nch
            t0 = c * rc

            # --- staging rows (input dtype): T.copy destinations are static rows
            # r1_c1 (O-1 bulk window load): the vhat / dY stream is loaded as ONE
            # 2-D window per block instead of one row per iteration (24 single-row
            # descriptors -> 2 window descriptors + edge guards).  Window row
            # `widx` holds token `t0 - LEAD + widx`, i.e. widx == the iteration
            # index `i`; the three rows past the block's own range are the causal
            # halo that `rc`+`LEAD` needs anyway.
            nwin = rc + 2 * LEAD
            vstg = T.alloc_shared((nwin, dw), dtype)  # vhat window
            dstg = T.alloc_shared((nwin, dw), dtype)  # dY window
            # r1_c2 (O-2 bulk window, (c) group): the per-token H / k / v loads and
            # the dH / dk / dv stores become one 2-D window each (18 + 18 single-row
            # descriptors -> 3 + 3).  Window row ``widx_c`` holds token ``t0 + widx_c``.
            hstg = T.alloc_shared((rc, dw), dtype)  # H window
            kstg = T.alloc_shared((rc, dw), dtype)  # k window
            vvstg = T.alloc_shared((rc, dw), dtype)  # v window
            dhstg = T.alloc_shared((rc, dw), dtype)  # dH output window
            dkstg = T.alloc_shared((rc, dw), dtype)  # dk output window
            dvstg = T.alloc_shared((rc, dw), dtype)  # dv output window
            whs = T.alloc_shared((dw,), dtype)
            wvs = T.alloc_shared((dw,), dtype)
            cws = T.alloc_shared((KS, dw), dtype)

            # --- fp32 four-slot ring windows, slot = floormod(iteration, KS)
            nrmw = T.alloc_shared((KS, dw), FP32)  # vhat_norm (DESIGN 1.6.1 O5)
            zraw = T.alloc_shared((KS, dw), FP32)  # raw vhat (see pass A)
            dyw = T.alloc_shared((KS, dw), FP32)  # dY
            dsw = T.alloc_shared((KS, dw), FP32)  # d_silu

            # --- fp32 constant tables (converted once per core)
            whf = T.alloc_shared((dw,), FP32)
            wvf = T.alloc_shared((dw,), FP32)
            cwf = T.alloc_shared((KS, dw), FP32)

            # --- per-core weight-gradient partials (DESIGN 0.6 R1)
            pc = T.alloc_shared((KS, dw), FP32)  # dconv_w
            ph = T.alloc_shared((dw,), FP32)  # drms_w_h
            pv = T.alloc_shared((dw,), FP32)  # drms_w_v

            # --- per-token scratch, full row resident (row dots precede their use)
            gn = T.alloc_shared((dw,), FP32)  # d_vnorm
            hnf = T.alloc_shared((dw,), FP32)  # h_norm
            knf = T.alloc_shared((dw,), FP32)  # k_norm

            # --- per-batch scalar strips (alpha / rrms_h / rrms_k / rrms_v)
            # Row pitch is padded to 32 B: a shorter strip (seq_len < 8) or a strip
            # whose length is not 8 fp32 elements (seq_len = 401) puts a row on a
            # 4 B boundary and the kernel faults on device.
            sc = T.alloc_shared((4, seq_pitch), FP32)

            # One version each: the ring is carried across iterations on purpose and
            # every other buffer is written then read inside the enclosing step.
            T.annotate_buffer_versions(
                {
                    vstg: 1,
                    dstg: 1,
                    hstg: 1,
                    kstg: 1,
                    vvstg: 1,
                    dhstg: 1,
                    dkstg: 1,
                    dvstg: 1,
                    whs: 1,
                    wvs: 1,
                    cws: 1,
                    nrmw: 1,
                    zraw: 1,
                    dyw: 1,
                    dsw: 1,
                    whf: 1,
                    wvf: 1,
                    cwf: 1,
                    pc: 1,
                    ph: 1,
                    pv: 1,
                    gn: 1,
                    hnf: 1,
                    knf: 1,
                    sc: 1,
                }
            )

            # -------- prologue: zero the partials, materialise the fp32 tables --
            with T.SimdVF():
                full = T.simd.pset(32)
                zero = T.simd.vdup(T.float32(0), FP32, full)
                for ch in T.serial(nchunks):
                    T.simd.vsts(ph[ch * VL], zero, full)
                    T.simd.vsts(pv[ch * VL], zero, full)
                    for p in T.serial(KS):
                        T.simd.vsts(pc[p, ch * VL], zero, full)
            # A SIMD_VF block may not read GM, so every parameter is staged to UB first.
            T.copy(WH, whs[:d])
            T.copy(WV, wvs[:d])
            T.copy(CW, cws[:, :d])
            T.copy(ALPHA[b, :], sc[0, :seq_len])
            T.copy(RH[b, :], sc[1, :seq_len])
            T.copy(RK[b, :], sc[2, :seq_len])
            T.copy(RV[b, :], sc[3, :seq_len])

            # -------- r1_c1: bulk load of the vhat / dY window -------------------
            # Rows with a global index outside [0, seq_len) are never loaded (the
            # source's ``T.if_then_else(..., 0)`` semantics): the consumer that
            # needs them (the nrm ring slot of iteration ``i``) is zero filled by
            # the unchanged guards below, so the untouched window slots are never
            # read.  The head case only exists for the block that owns the batch
            # start, so it is a block-level scalar guard; the tail case uses a
            # clamped extent.
            if t0 >= LEAD:
                T.copy(VHAT[b, t0 - LEAD : t0, :], vstg[0:LEAD, :d])
                T.copy(DY[b, t0 - LEAD : t0, :], dstg[0:LEAD, :d])
            else:
                for hz in range(LEAD):
                    hrr = t0 - LEAD + hz
                    if hrr >= 0:
                        T.copy(VHAT[b, hrr, :], vstg[hz, :d])
                        T.copy(DY[b, hrr, :], dstg[hz, :d])
            if t0 + rc + LEAD <= seq_len:
                T.copy(
                    VHAT[b, t0 : t0 + rc + LEAD, :],
                    vstg[LEAD : LEAD + rc + LEAD, :d],
                )
                T.copy(
                    DY[b, t0 : t0 + rc + LEAD, :],
                    dstg[LEAD : LEAD + rc + LEAD, :d],
                )
            else:
                nr = T.min(rc + LEAD, seq_len - t0)
                T.copy(VHAT[b, t0 : t0 + nr, :], vstg[LEAD : LEAD + nr, :d])
                T.copy(DY[b, t0 : t0 + nr, :], dstg[LEAD : LEAD + nr, :d])

            # -------- r1_c2: bulk load of the H / k / v window (own rows only) ----
            if t0 + rc <= seq_len:
                T.copy(H[b, t0 : t0 + rc, :], hstg[:, :d])
                T.copy(K[b, t0 : t0 + rc, :], kstg[:, :d])
                T.copy(V[b, t0 : t0 + rc, :], vvstg[:, :d])
            else:
                nc = seq_len - t0  # the last run of a batch is short
                T.copy(H[b, t0 : t0 + nc, :], hstg[0:nc, :d])
                T.copy(K[b, t0 : t0 + nc, :], kstg[0:nc, :d])
                T.copy(V[b, t0 : t0 + nc, :], vvstg[0:nc, :d])
            with T.SimdVF():
                # The partials were vector-stored above and are read-modify-written
                # below: the store has to be made visible to the first vector load.
                T.simd.mem_bar("VST_VLD")
                full = T.simd.pset(32)
                for ch in T.serial(nchunks):
                    mask = T.simd.update_mask(T.min(VL, d - ch * VL), 32)
                    T.simd.vsts(
                        whf[ch * VL],
                        T.simd.vcvt(
                            T.simd.vld(whs[ch * VL], dist="UNPK_B16"), FP32, mask, part=0
                        ),
                        full,
                    )
                    T.simd.vsts(
                        wvf[ch * VL],
                        T.simd.vcvt(
                            T.simd.vld(wvs[ch * VL], dist="UNPK_B16"), FP32, mask, part=0
                        ),
                        full,
                    )
                    for p in T.serial(KS):
                        T.simd.vsts(
                            cwf[p, ch * VL],
                            T.simd.vcvt(
                                T.simd.vld(cws[p, ch * VL], dist="UNPK_B16"),
                                FP32,
                                mask,
                                part=0,
                            ),
                            full,
                        )

            # =================================================================
            # Fused loop: iteration i publishes d_silu for row pn = t0 - 3 + i
            # and runs the value/gate backward for token tp = pn - 3.
            # =================================================================
            for i in T.serial(rc + 2 * LEAD):
                pn = t0 - LEAD + i
                slot = T.floormod(i, KS)

                # ---- (a) vhat_norm / raw vhat / dY of row pn into the ring -----
                # Out-of-batch rows are never loaded (the source's
                # T.if_then_else(..., 0) semantics) and their slot is zeroed, so a
                # causal tap reads an exact 0 and no 0 * inf can appear.  Two
                # complementary guards because the traced frontend has no scalar
                # else.
                if pn >= 0:
                    if pn < seq_len:
                        # r1_c1: the row already sits in the bulk window at slot
                        # ``i`` (widx == i); no per-row DMA and no GM read.
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
                                dyv = T.simd.vcvt(
                                    T.simd.vld(dstg[i, ch * VL], dist="UNPK_B16"),
                                    FP32,
                                    mask,
                                    part=0,
                                )
                                T.simd.vsts(zraw[slot, ch * VL], z, full)
                                T.simd.vsts(dyw[slot, ch * VL], dyv, full)
                                # O5: vhat_norm materialised once per row, reused by
                                # the four convolution taps and by dconv_w.
                                T.simd.vsts(
                                    nrmw[slot, ch * VL],
                                    T.simd.vmul(
                                        T.simd.vmul(z, rv_p, mask),
                                        T.simd.vld(wvf[ch * VL]),
                                        mask,
                                    ),
                                    full,
                                )
                    if pn >= seq_len:  # past the batch tail: zero slot, no load
                        with T.SimdVF():
                            full = T.simd.pset(32)
                            zero = T.simd.vdup(T.float32(0), FP32, full)
                            for ch in T.serial(nchunks):
                                T.simd.vsts(nrmw[slot, ch * VL], zero, full)
                                T.simd.vsts(zraw[slot, ch * VL], zero, full)
                                T.simd.vsts(dyw[slot, ch * VL], zero, full)
                if pn < 0:  # before the batch head: zero slot, no load
                    with T.SimdVF():
                        full = T.simd.pset(32)
                        zero = T.simd.vdup(T.float32(0), FP32, full)
                        for ch in T.serial(nchunks):
                            T.simd.vsts(nrmw[slot, ch * VL], zero, full)
                            T.simd.vsts(zraw[slot, ch * VL], zero, full)
                            T.simd.vsts(dyw[slot, ch * VL], zero, full)

                # ---- (b) d_silu of row pds = pn (conv recompute + logit deriva.)
                if i >= LEAD:
                    pds = t0 + i - LEAD  # == pn
                    if pds < seq_len:
                        with T.SimdVF():
                            # The row window was vector-stored above (and by the
                            # three previous iterations): vector store -> vector
                            # load needs an explicit barrier.
                            T.simd.mem_bar("VST_VLD")
                            full = T.simd.pset(32)
                            one = T.simd.vdup(T.float32(1), FP32, full)
                            for ch in T.serial(nchunks):
                                mask = T.simd.update_mask(T.min(VL, d - ch * VL), 32)
                                conv = T.simd.alloc_local((1,), FP32)
                                conv[0] = T.simd.vdup(T.float32(0), FP32, full)
                                for p in T.unroll(KS):
                                    conv[0] = T.simd.vadd(
                                        conv[0],
                                        T.simd.vmul(
                                            T.simd.vld(
                                                nrmw[T.floormod(i - LEAD + p, KS), ch * VL]
                                            ),
                                            T.simd.vld(cwf[p, ch * VL]),
                                            mask,
                                        ),
                                        full,
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
                                T.simd.vsts(
                                    dsw[slot, ch * VL],
                                    T.simd.vmul(
                                        T.simd.vld(dyw[slot, ch * VL]), deriv, mask
                                    ),
                                    full,
                                )
                    if pds >= seq_len:  # halo past the tail: zero slot, no compute
                        with T.SimdVF():
                            full = T.simd.pset(32)
                            zero = T.simd.vdup(T.float32(0), FP32, full)
                            for ch in T.serial(nchunks):
                                T.simd.vsts(dsw[slot, ch * VL], zero, full)
                    # ---- dconv_w partial: only the run that owns the row -----
                    if i < rc + LEAD:  # pds = t0 + i - 3 < t0 + RC
                        if pds < seq_len:
                            with T.SimdVF():
                                # d_silu was just stored and pc is read-modify-written
                                # across iterations: barrier before the loads.
                                T.simd.mem_bar("VST_VLD")
                                full = T.simd.pset(32)
                                for ch in T.serial(nchunks):
                                    mask = T.simd.update_mask(T.min(VL, d - ch * VL), 32)
                                    ds_v = T.simd.vld(dsw[slot, ch * VL])
                                    for p in T.unroll(KS):
                                        T.simd.vsts(
                                            pc[p, ch * VL],
                                            T.simd.vadd(
                                                T.simd.vld(pc[p, ch * VL]),
                                                T.simd.vmul(
                                                    ds_v,
                                                    T.simd.vld(
                                                        nrmw[
                                                            T.floormod(i - LEAD + p, KS),
                                                            ch * VL,
                                                        ]
                                                    ),
                                                    mask,
                                                ),
                                                full,
                                            ),
                                            full,
                                        )

                # ---- (c) value / gate backward for token tp = t0 + i - 6 ------
                if i >= 2 * LEAD:
                    tp = t0 + i - 2 * LEAD
                    if tp < seq_len:  # the last run is short: guard the whole step
                        # r1_c2: the H / k / v rows are already resident; the window
                        # slot of token ``tp`` is ``i - 2*LEAD`` (``tp - t0``).
                        widx_c = i - 2 * LEAD
                        with T.SimdVF():
                            T.simd.mem_bar("VST_VLD")
                            full = T.simd.pset(32)
                            one = T.simd.vdup(T.float32(1), FP32, full)
                            rv = T.simd.vld(sc[3, tp], dist="BRC_B32")
                            rh = T.simd.vld(sc[1, tp], dist="BRC_B32")
                            rk = T.simd.vld(sc[2, tp], dist="BRC_B32")
                            al = T.simd.vld(sc[0, tp], dist="BRC_B32")
                            # ---- pass A: inverse convolution -> d_vnorm, dot_v,
                            #      drms_w_v partial -----------------------------
                            # O3 keeps z * d_vnorm; z is the *raw* vhat, exactly as
                            # in DESIGN 1.3 (dot_v = sum vhat*w_v*g, drms_w_v =
                            # sum g*vhat*rrms_v, correction = rrms_v^3*vhat*dot_v/d).
                            # vhat_norm carries the rrms_v * w_v factors and must not
                            # stand in for it.
                            dot = T.simd.alloc_local((1,), FP32)
                            dot[0] = T.simd.vdup(T.float32(0), FP32, full)
                            for ch in T.serial(nchunks):
                                mask = T.simd.update_mask(T.min(VL, d - ch * VL), 32)
                                dn = T.simd.alloc_local((1,), FP32)
                                dn[0] = T.simd.vdup(T.float32(0), FP32, full)
                                for p in T.unroll(KS):
                                    dn[0] = T.simd.vadd(
                                        dn[0],
                                        T.simd.vmul(
                                            T.simd.vld(cwf[p, ch * VL]),
                                            T.simd.vld(
                                                dsw[T.floormod(i - p, KS), ch * VL]
                                            ),
                                            mask,
                                        ),
                                        full,
                                    )
                                T.simd.vsts(gn[ch * VL], dn[0], full)
                                zd = T.simd.vmul(
                                    T.simd.vld(
                                        zraw[T.floormod(i - LEAD, KS), ch * VL]
                                    ),
                                    dn[0],
                                    mask,
                                )
                                dot[0] = T.simd.vadd(
                                    dot[0],
                                    T.simd.vmul(zd, T.simd.vld(wvf[ch * VL]), mask),
                                    full,
                                )
                                T.simd.vsts(
                                    pv[ch * VL],
                                    T.simd.vadd(
                                        T.simd.vld(pv[ch * VL]),
                                        T.simd.vmul(zd, rv, mask),
                                        full,
                                    ),
                                    full,
                                )
                            dot_b = T.simd.vdupv(T.simd.vcadd(dot[0], full), full)
                            # ---- pass B: dvhat -> dv output and dalpha ----------
                            T.simd.mem_bar("VST_VLD")
                            da = T.simd.alloc_local((1,), FP32)
                            da[0] = T.simd.vdup(T.float32(0), FP32, full)
                            # r2_c3 (O-9): (rrms_v^3 * (1/d)) is one per token, not one
                            # per chunk lane: hoisted into a register before the sweep.
                            coef = T.simd.alloc_local((1,), FP32)
                            coef[0] = T.simd.vmuls(
                                T.simd.vmul(T.simd.vmul(rv, rv, full), rv, full),
                                inv_d,
                                full,
                            )
                            for ch in T.serial(nchunks):
                                mask = T.simd.update_mask(T.min(VL, d - ch * VL), 32)
                                z = T.simd.vld(zraw[T.floormod(i - LEAD, KS), ch * VL])
                                direct = T.simd.vmul(
                                    rv,
                                    T.simd.vmul(
                                        T.simd.vld(wvf[ch * VL]),
                                        T.simd.vld(gn[ch * VL]),
                                        mask,
                                    ),
                                    mask,
                                )
                                corr = T.simd.vmul(
                                    T.simd.vmul(coef[0], z, mask), dot_b, mask
                                )
                                g = T.simd.vadd(
                                    T.simd.vsub(direct, corr, mask),
                                    T.simd.vld(
                                        dyw[T.floormod(i - LEAD, KS), ch * VL]
                                    ),
                                    mask,
                                )
                                T.simd.vsts(
                                    dvstg[widx_c, ch * VL],
                                    T.simd.vcvt(
                                        T.simd.vmul(al, g, mask),
                                        dtype,
                                        round="ROUND_R",
                                        sat=False,
                                        mask=mask,
                                        part=0,
                                    ),
                                    full,
                                    dist="PK_B32",
                                )
                                vv = T.simd.vcvt(
                                    T.simd.vld(vvstg[widx_c, ch * VL], dist="UNPK_B16"),
                                    FP32,
                                    mask,
                                    part=0,
                                )
                                da[0] = T.simd.vadd(
                                    da[0], T.simd.vmul(g, vv, mask), full
                                )
                            da_b = T.simd.vdupv(T.simd.vcadd(da[0], full), full)
                            # ddot = dalpha * alpha * (1 - alpha) * (1/sqrt(d))
                            ddot = T.simd.vmul(
                                T.simd.vmul(da_b, al, full),
                                T.simd.vmul(
                                    T.simd.vsub(one, al, full),
                                    T.simd.vdup(T.float32(inv_sqrt_d), FP32, full),
                                    full,
                                ),
                                full,
                            )
                            # ---- pass C1: gate norms, the two row dots and the
                            #      drms_w_h partial ---------------------------------
                            T.simd.mem_bar("VST_VLD")
                            dh_dot = T.simd.alloc_local((1,), FP32)
                            dk_dot = T.simd.alloc_local((1,), FP32)
                            dh_dot[0] = T.simd.vdup(T.float32(0), FP32, full)
                            dk_dot[0] = T.simd.vdup(T.float32(0), FP32, full)
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
                                hn = T.simd.vmul(T.simd.vmul(hf, rh, mask), w, mask)
                                kn = T.simd.vmul(T.simd.vmul(kf, rk, mask), w, mask)
                                T.simd.vsts(hnf[ch * VL], hn, full)
                                T.simd.vsts(knf[ch * VL], kn, full)
                                # O4: u_h / u_k formed once, reused by the dots,
                                # the outputs and the drms_w_h partial.
                                uh = T.simd.vmul(ddot, kn, mask)
                                uk = T.simd.vmul(ddot, hn, mask)
                                T.simd.vsts(
                                    ph[ch * VL],
                                    T.simd.vadd(
                                        T.simd.vld(ph[ch * VL]),
                                        T.simd.vadd(
                                            T.simd.vmul(
                                                T.simd.vmul(uh, hf, mask), rh, mask
                                            ),
                                            T.simd.vmul(
                                                T.simd.vmul(uk, kf, mask), rk, mask
                                            ),
                                            mask,
                                        ),
                                        full,
                                    ),
                                    full,
                                )
                                dh_dot[0] = T.simd.vadd(
                                    dh_dot[0],
                                    T.simd.vmul(T.simd.vmul(w, uh, mask), hf, mask),
                                    full,
                                )
                                dk_dot[0] = T.simd.vadd(
                                    dk_dot[0],
                                    T.simd.vmul(T.simd.vmul(w, uk, mask), kf, mask),
                                    full,
                                )
                            dhb = T.simd.vdupv(T.simd.vcadd(dh_dot[0], full), full)
                            dkb = T.simd.vdupv(T.simd.vcadd(dk_dot[0], full), full)
                            # ---- pass C2: dH / dk outputs -----------------------
                            T.simd.mem_bar("VST_VLD")
                            # r2_c3 (O-9): rrms^3 * (1/d) is one per token, hoisted
                            # out of the chunk sweep into registers.
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
                                dh = T.simd.vsub(
                                    T.simd.vmul(
                                        rh,
                                        T.simd.vmul(
                                            w,
                                            T.simd.vmul(ddot, kn, mask),
                                            mask,
                                        ),
                                        mask,
                                    ),
                                    T.simd.vmul(
                                        T.simd.vmul(rh3d[0], hf, mask), dhb, mask
                                    ),
                                    mask,
                                )
                                dkk = T.simd.vsub(
                                    T.simd.vmul(
                                        rk,
                                        T.simd.vmul(
                                            w,
                                            T.simd.vmul(ddot, hn, mask),
                                            mask,
                                        ),
                                        mask,
                                    ),
                                    T.simd.vmul(
                                        T.simd.vmul(rk3d[0], kf, mask), dkb, mask
                                    ),
                                    mask,
                                )
                                T.simd.vsts(
                                    dhstg[widx_c, ch * VL],
                                    T.simd.vcvt(
                                        dh,
                                        dtype,
                                        round="ROUND_R",
                                        sat=False,
                                        mask=mask,
                                        part=0,
                                    ),
                                    full,
                                    dist="PK_B32",
                                )
                                T.simd.vsts(
                                    dkstg[widx_c, ch * VL],
                                    T.simd.vcvt(
                                        dkk,
                                        dtype,
                                        round="ROUND_R",
                                        sat=False,
                                        mask=mask,
                                        part=0,
                                    ),
                                    full,
                                    dist="PK_B32",
                                )

            # -------- r1_c2: bulk store of the dH / dk / dv window ----------------
            if t0 + rc <= seq_len:
                T.copy(dhstg[:, :d], DH[b, t0 : t0 + rc, :])
                T.copy(dkstg[:, :d], DK[b, t0 : t0 + rc, :])
                T.copy(dvstg[:, :d], DV[b, t0 : t0 + rc, :])
            else:
                nc = seq_len - t0
                T.copy(dhstg[0:nc, :d], DH[b, t0 : t0 + nc, :])
                T.copy(dkstg[0:nc, :d], DK[b, t0 : t0 + nc, :])
                T.copy(dvstg[0:nc, :d], DV[b, t0 : t0 + nc, :])

            # -------- per-core partials -> GM workspace (DESIGN 0.6 R1) ---------
            for p in T.serial(KS):
                T.copy(pc[p, :], PART[2 + p, bx, :])
            T.copy(ph[:], PART[0, bx, :])
            T.copy(pv[:], PART[1, bx, :])

    return tilelang.compile(main, target="ascend", out_idx=[12, 13, 14, 15])


# ---------------------------------------------------------------------------
# 4. Launch #2: ordered reduction of the per-core partials (DESIGN.md 0.6 R1)
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=32)
def _build_reduce(cores: int, dw: int, d: int):
    """Sum ``PART[6, cores, dw]`` in ascending core order; deterministic."""
    width, tiles = reduce_layout(dw, d)

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
            # Clamp the output extent to the real d: the partials are dw wide but the
            # outputs are d wide, and d is not always a multiple of VL (d = 100).
            cols = T.min(width, d - off)
            part = T.alloc_shared((cores, width), FP32)
            tot = T.alloc_shared((width,), FP32)
            T.annotate_buffer_versions({part: 1, tot: 1})
            if g == 0:
                T.copy(PART[0, :, off : off + cols], part[:, :cols])
            elif g == 1:
                T.copy(PART[1, :, off : off + cols], part[:, :cols])
            elif g == 2:
                T.copy(PART[2, :, off : off + cols], part[:, :cols])
            elif g == 3:
                T.copy(PART[3, :, off : off + cols], part[:, :cols])
            elif g == 4:
                T.copy(PART[4, :, off : off + cols], part[:, :cols])
            else:
                T.copy(PART[5, :, off : off + cols], part[:, :cols])
            with T.SimdVF():
                full = T.simd.pset(32)
                T.simd.mem_bar("VST_VLD")
                for ch in T.serial(width // VL):
                    col = ch * VL
                    valid = T.simd.update_mask(T.min(VL, T.max(0, cols - col)), 32)
                    run = T.simd.alloc_local((1,), FP32)
                    run[0] = T.simd.vdup(T.float32(0), FP32, full)
                    # The row loop *is* the determinism contract: ascending core id.
                    for row in T.serial(cores):
                        run[0] = T.simd.vadd(
                            run[0], T.simd.vld(part[row, col]), valid
                        )
                    T.simd.vsts(tot[col], run[0], full)
            if g == 0:
                T.copy(tot[:cols], DWH[off : off + cols])
            elif g == 1:
                T.copy(tot[:cols], DWV[off : off + cols])
            elif g == 2:
                T.copy(tot[:cols], DCW[0, off : off + cols])
            elif g == 3:
                T.copy(tot[:cols], DCW[1, off : off + cols])
            elif g == 4:
                T.copy(tot[:cols], DCW[2, off : off + cols])
            else:
                T.copy(tot[:cols], DCW[3, off : off + cols])

    return tilelang.compile(main, target="ascend", out_idx=[1, 2, 3])


# ---------------------------------------------------------------------------
# 5. Op entry point: one callable returning the six gradients in op order
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=32)
def _engram_gate_conv_bwd_kernel(M: int, seq_len: int, d: int, dtype: str):
    """Compile both launches and return ``run(*inputs) -> 6 gradients``."""
    geo = geometry(M, seq_len, d)
    fused = _build_fused(M, seq_len, d, dtype)
    reduce_fn = _build_reduce(geo["grid"], geo["dw"], d)

    def run(dY, H, k, v, rms_w_h, rms_w_v, conv_w, vhat, alpha, rrms_h, rrms_k, rrms_v):
        dh, dk, dv, part = fused(
            dY, H, k, v, rms_w_h, rms_w_v, conv_w, vhat, alpha, rrms_h, rrms_k, rrms_v
        )
        dwh, dwv, dcw = reduce_fn(part)
        return [dh, dk, dv, dwh, dwv, dcw]

    return run


# ---------------------------------------------------------------------------
# 6. Case runner (golden on the CPU, kernel on the NPU, six outputs compared)
# ---------------------------------------------------------------------------
TOL_BWD = {"float16": (0.2, 0.2), "bfloat16": (0.3, 0.2)}  # _tol_bwd, per output
RESULTS = []  # (level, name, ok, detail)
MAXDIFF = {name: 0.0 for name in OUT_NAMES}


def _record(level, name, ok, detail=""):
    RESULTS.append((level, name, ok, detail))


def _rand(shape, dtype, seed, scale=1.0):
    gen = torch.Generator().manual_seed(seed)
    return (torch.randn(shape, generator=gen, dtype=torch.float32) * scale).to(dtype)


def _make_inputs(M, seq_len, d, dtype, seed, pattern=None, manifest=False):
    """The op's twelve inputs on the CPU, with the workload's saved-state recipe.

    The five saved intermediates are regenerated from the fp32 forward so that
    ``vhat`` carries the forward's rounding (the op contract); ``alpha`` and the
    three ``rrms`` are the forward's own statistics.
    """
    torch_dtype = getattr(torch, dtype)
    H = _rand((M, seq_len, d), torch_dtype, seed)
    k = _rand((M, seq_len, d), torch_dtype, seed + 1, 0.1)
    v = _rand((M, seq_len, d), torch_dtype, seed + 2, 0.1)
    if manifest:
        wh = torch.ones(d, dtype=torch_dtype)
        wv = torch.ones(d, dtype=torch_dtype)
    else:
        wh = _rand((d,), torch_dtype, seed + 3, 0.1) + 1.0
        wv = _rand((d,), torch_dtype, seed + 4, 0.1) + 1.0
    cw = _rand((KS, d), torch_dtype, seed + 5, 0.02)
    dY = _rand((M, seq_len, d), torch_dtype, seed + 6, 0.1)

    if pattern == "zeros":
        H.zero_()
        k.zero_()
        v.zero_()
        dY.zero_()
        cw.zero_()
    elif pattern == "zero-dy":
        dY.zero_()
    elif pattern == "nonfinite-tail":
        # The repository's nonfinite case: the last row of dY is +-inf, so every
        # dconv_w tap of that row is infinite.  The tail rows past seq_len must
        # add nothing (not zero times an infinite gradient).
        dY[0, -1, :] = float("inf")
    elif pattern is not None:
        raise ValueError(f"pattern: unknown {pattern!r}")

    h_norm, rrms_h = _rmsnorm(H, wh)
    k_norm, rrms_k = _rmsnorm(k, wh)
    dot = (h_norm * k_norm).sum(dim=-1, keepdim=True)
    alpha = torch.sigmoid(dot / (d**0.5))
    v_hat = alpha * v.float()
    _, rrms_v = _rmsnorm(v_hat.to(torch_dtype), wv)
    return (
        dY,
        H,
        k,
        v,
        wh,
        wv,
        cw,
        v_hat.to(torch_dtype),
        alpha.squeeze(-1).float(),
        rrms_h.float(),
        rrms_k.float(),
        rrms_v.float(),
    )


def _diff(got, gold):
    a, b = got.float(), gold.float()
    diff = (a - b).abs()
    diff = torch.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)
    nan_mismatch = int(((a != a) != (b != b)).sum())
    return (float(diff.max()) if diff.numel() else 0.0), nan_mismatch


def run_case(
    level,
    name,
    M,
    seq_len,
    d,
    dtype,
    seed=0,
    pattern=None,
    manifest=False,
    repeat=1,
    nonfinite_positions=False,
):
    """Build one geometry, run it on the NPU, compare all six outputs."""
    geo = geometry(M, seq_len, d)
    per_out = {}
    elapsed = float("nan")
    try:
        inputs = _make_inputs(M, seq_len, d, dtype, seed, pattern, manifest)
        gold = golden_engram_gate_conv_bwd(*inputs)

        fn = _engram_gate_conv_bwd_kernel(M, seq_len, d, dtype)
        args = [t.to("npu") for t in inputs]
        if repeat > 1:  # warm up, then measure and keep the first result
            fn(*args)
            torch.npu.synchronize()
        start = time.perf_counter()
        got = fn(*args)
        torch.npu.synchronize()
        elapsed = (time.perf_counter() - start) * 1e3
        got = [g.cpu() for g in got]

        problems = []
        if repeat > 1:
            # Two launches of one compiled kernel with identical inputs must agree
            # bit for bit: a UB store/load visibility slip is run-to-run random.
            again = [g.cpu() for g in fn(*args)]
            for idx, out_name in enumerate(OUT_NAMES):
                if not torch.equal(got[idx], again[idx]):
                    dmax, _ = _diff(got[idx], again[idx])
                    problems.append(
                        f"{out_name}: not bitwise reproducible across runs "
                        f"(max_diff={dmax:.3e})"
                    )

        if nonfinite_positions:
            # Only the sign/NaN positions are compared (the reference multiplies
            # inf by the same factors but its reduction order differs).
            for kind in (torch.isposinf, torch.isneginf, torch.isnan):
                if not torch.equal(kind(got[5]), kind(gold[5])):
                    problems.append(f"dconv_w: {kind.__name__} positions differ")
            nonfinite = int((~torch.isfinite(gold[5])).sum())
            per_out["dconv_w_nonfinite"] = float(nonfinite)
            if nonfinite == 0:  # the case would be vacuous
                problems.append("dconv_w: the reference has no non-finite tap")
        else:
            atol, rtol = TOL_BWD[dtype]
            for idx, out_name in enumerate(OUT_NAMES):
                if got[idx].shape != gold[idx].shape:
                    problems.append(
                        f"{out_name}: got {tuple(got[idx].shape)}, "
                        f"want {tuple(gold[idx].shape)}"
                    )
                    continue
                want_dtype = gold[idx].dtype
                if got[idx].dtype != want_dtype:
                    problems.append(
                        f"{out_name}: dtype {got[idx].dtype} != {want_dtype}"
                    )
                    continue
                diff, nan_mismatch = _diff(got[idx], gold[idx])
                per_out[out_name] = diff
                MAXDIFF[out_name] = max(MAXDIFF[out_name], diff)
                if nan_mismatch:
                    problems.append(f"{out_name}: {nan_mismatch} NaN positions differ")
                try:
                    torch.testing.assert_close(
                        got[idx], gold[idx], atol=atol, rtol=rtol, equal_nan=True
                    )
                except AssertionError:
                    problems.append(f"{out_name}: max_diff={diff:.3e} > atol={atol:g}")
            if pattern == "zero-dy":
                # dY = 0 makes every output exactly zero (each one is linear in dY).
                for idx, out_name in enumerate(OUT_NAMES):
                    worst = float(got[idx].abs().max())
                    if worst != 0.0:
                        problems.append(f"{out_name}: expected zero, got {worst:.3e}")
        if problems:
            raise AssertionError("; ".join(problems))
    except AssertionError as exc:
        _record(level, name, False, f"{str(exc)[:400]}")
        return False
    except (ValueError, RuntimeError) as exc:  # reported, never swallowed
        _record(level, name, False, f"{type(exc).__name__}: {str(exc)[:400]}")
        return False
    _record(level, name, True)
    shown = " ".join(f"{n}={per_out[n]:.2e}" for n in per_out)
    print(
        f"       M={M} S={seq_len} d={d} {dtype} RC={geo['RC']} grid={geo['grid']} "
        f"nchunks={geo['nchunks']} {shown} launch={elapsed:.3f} ms",
        flush=True,
    )
    return True


def _finish(level, strict):
    rows = [r for r in RESULTS if r[0] == level]
    failed = [r for r in rows if not r[2]]
    print(
        f"[{level}] {len(rows) - len(failed)}/{len(rows)} cases passed | "
        + " ".join(f"{n}_max_diff={MAXDIFF[n]:.3e}" for n in OUT_NAMES),
        flush=True,
    )
    if failed and strict:
        for row in failed:
            print(f"[{level}][FAIL] {row[1]}: {row[3]}", flush=True)
        raise AssertionError(f"[{level}] {len(failed)} case(s) failed")
    if failed:
        for row in failed:
            print(f"[{level}][WARN] {row[1]}: {row[3]}", flush=True)


# ---------------------------------------------------------------------------
# 7. Layered cases
# ---------------------------------------------------------------------------
def run_L0():
    """Gate: the four manifest geometries, the tail-lane path, seq_len < 4, the
    short last run, prime seq_len, zero gradients and bitwise determinism."""
    # 1-4: the manifest workloads (train-s32 / train-s64 / train-s128 / train-s96).
    run_case("L0", "L0-1 train-s32-fp16", 1, 32, 256, "float16", seed=101)
    # RC = 3 and 22 * 3 > 64: the last run owns one token, so step (c)'s tp guard
    # and the zero d_silu halo are exercised.
    run_case("L0", "L0-2 train-s64-fp16", 2, 64, 512, "float16", seed=102)
    run_case("L0", "L0-3 train-s128-bf16", 1, 128, 256, "bfloat16", seed=103)
    # Largest manifest d (768). RC = 6: the four-slot ring rolls over many times.
    run_case("L0", "L0-4 train-s96-bf16", 3, 96, 768, "bfloat16", seed=104)
    # d = 100 -> dw = 128, tail masks 64/36: masked lanes, /d with the real d.
    run_case("L0", "L0-5 d100-tail-lane-fp16", 1, 16, 100, "float16", seed=105)
    # seq_len < 4: only the p = 3 tap is valid at t = 0.
    run_case("L0", "L0-6 seq1-fp16", 2, 1, 256, "float16", seed=106)
    # 401 is prime: RC = 23, the last run owns 10 tokens.
    run_case("L0", "L0-7 seq401-prime-bf16", 3, 401, 256, "bfloat16", seed=107)
    # One chunk per batch row (nch = seq_len, grid = tokens).
    run_case("L0", "L0-8 seq8-single-chunk-fp16", 1, 8, 256, "float16", seed=108)
    # dY = 0: every gradient is exactly zero.
    run_case(
        "L0", "L0-9 zero-dy-fp16", 1, 16, 256, "float16", seed=109, pattern="zero-dy"
    )
    # Two launches of the same compiled kernel must agree bit for bit.
    run_case("L0", "L0-10 deterministic-bf16", 3, 96, 768, "bfloat16", seed=110, repeat=2)
    _finish("L0", strict=True)


def run_L1():
    """Cross dtype / geometry coverage, multi-batch mapping and repeat runs."""
    run_case("L1", "L1-1 train-s32-bf16", 1, 32, 256, "bfloat16", seed=201, repeat=2)
    run_case("L1", "L1-2 train-s96-fp16", 3, 96, 768, "float16", seed=202, repeat=2)
    run_case("L1", "L1-3 train-s64-bf16", 2, 64, 512, "bfloat16", seed=203)
    run_case("L1", "L1-4 train-s128-fp16", 1, 128, 256, "float16", seed=204)
    run_case("L1", "L1-5 s512-fp16", 2, 512, 256, "float16", seed=205, repeat=2)
    run_case("L1", "L1-6 s16-bf16", 2, 16, 256, "bfloat16", seed=206)
    _finish("L1", strict=True)


def run_L2():
    """The manifest workloads with the manifest input recipe (informative)."""
    run_case("L2", "L2-1 train-s32", 1, 32, 256, "float16", seed=301, manifest=True)
    run_case("L2", "L2-2 train-s64", 2, 64, 512, "float16", seed=302, manifest=True)
    run_case("L2", "L2-3 train-s128", 1, 128, 256, "bfloat16", seed=303, manifest=True)
    run_case("L2", "L2-4 train-s96", 3, 96, 768, "bfloat16", seed=304, manifest=True)
    _finish("L2", strict=False)


def run_boundary():
    """Prime seq_len, d = 1, the d upper bound, d = 130 and the nonfinite case."""
    # 95 is prime: RC = 2, nch = 48.
    run_case("boundary", "B-1 seq95-prime-bf16", 1, 95, 256, "bfloat16", seed=401)
    # A single valid lane: mask = update_mask(1).
    run_case("boundary", "B-2 d1-single-lane-fp16", 1, 4, 1, "float16", seed=402)
    # The largest supported d (152 B per channel * 1024 + scalar strip).
    run_case("boundary", "B-3 d1024-upper-bound-fp16", 1, 64, 1024, "float16", seed=403)
    # d = 130 -> dw = 192, tail mask 64/64/2.
    run_case("boundary", "B-4 d130-tail-lane-bf16", 1, 2, 130, "bfloat16", seed=404)
    # dY[0, -1, :] = inf: the dconv_w taps of that row are infinite while the
    # rows past the tail contribute nothing (never loaded, zero slot).
    run_case(
        "boundary",
        "B-5 dconv-nonfinite-tail-fp16",
        1,
        9,
        256,
        "float16",
        seed=405,
        pattern="nonfinite-tail",
        nonfinite_positions=True,
    )
    _finish("boundary", strict=False)


def _verdict():
    failed = [r for r in RESULTS if not r[2] and r[0] in ("L0", "L1")]
    if failed:
        print(f"[PRECISION_FAIL] {len(failed)} blocking case(s) failed", flush=True)
    else:
        print(
            "[PRECISION_PASS] " + " ".join(f"{n}={MAXDIFF[n]:.3e}" for n in OUT_NAMES),
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--level", default="L0", choices=["L0", "L1", "L2", "boundary", "all"]
    )
    args, _ = parser.parse_known_args()
    levels = {"L0": run_L0, "L1": run_L1, "L2": run_L2, "boundary": run_boundary}
    chosen = list(levels.values()) if args.level == "all" else [levels[args.level]]
    started = time.perf_counter()
    for run in chosen:
        run()
    print(f"[TIME] {time.perf_counter() - started:.1f} s for level={args.level}", flush=True)
    _verdict()


if __name__ == "__main__":
    main()
