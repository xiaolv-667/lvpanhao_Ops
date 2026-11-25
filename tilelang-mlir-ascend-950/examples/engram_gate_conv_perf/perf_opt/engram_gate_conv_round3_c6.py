# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Engram GateConv forward (post-projection fusion) for Ascend 950PR (SIMD / AIV).

Single AIV launch: every core owns ``RC`` consecutive rows of one ``m`` plus a
three row causal halo, keeps one convolution input row per pipeline row in UB
(``rc + 3`` rows, affine indices) and recomputes the halo rows with the same
instruction sequence (so the recomputation is bit identical to the owning core).
The gate statistics, the ``vhat`` rounding round trip, the ``rms_w_v`` folded
convolution weights and the ``alpha * v`` residual follow DESIGN.md 0.6 / 3.3 /
8.1.

Two toolchain rules this kernel has to respect (both with a repro under
``repro/``): a vector store to UB is only visible to a later vector load of the
same buffer after ``T.simd.mem_bar("VST_VLD")``, and the rounded ``vhat`` is
re-derived from the DMA-filled ``v_ub`` rather than read back from the 16-bit
row buffer.

Six outputs: ``Y``, ``vhat`` (input dtype) and ``alpha``, ``rrms_h``,
``rrms_k``, ``rrms_v`` (float32).

run: python _engram_gate_conv_fwd_kernel.py --level all
"""

import argparse
import functools
import time

import tilelang
import tilelang.ascend.language as T
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401  # registers torch.npu

# ---------------------------------------------------------------------------
# Constants (DESIGN.md 5.2 / 5.5 / 8.2)
# ---------------------------------------------------------------------------
VL = 64  # fp32 vector register width; also the 16-bit chunk mapped by UNPK_B16/PK_B32
KS = 4  # causal depthwise convolution taps
CORES = 56  # Ascend950PR_9579 vector cores (statectl platform observed_cores.vector)
MAX_HIDDEN_DIM = 4096  # same refusal as the source kernel and the 950DT port
UB_BYTES = 253952  # Ascend950PR_9579.ini [AICoreSpec] ub_size
MIN_GRID = CORES // 2  # below this the RC divisor rule falls back to RC = 1

MEMBERS = 4  # alpha, rrms_h, rrms_k, rrms_v
STATS_STRIDE = 8  # fp32 slots per st row: 32 B row pitch for the single-lane store
ELEM_BYTES = {"float16": 2, "bfloat16": 2}

ATOL = {"float16": 1e-2, "bfloat16": 2e-2}  # Y / vhat (DESIGN.md 8.2)
RTOL = {"float16": 1e-2, "bfloat16": 2e-2}
SCALAR_ATOL = 1e-3  # alpha / rrms_* (float32)
SCALAR_RTOL = 1e-3

OUT_NAMES = ("Y", "vhat", "alpha", "rrms_h", "rrms_k", "rrms_v")


# ---------------------------------------------------------------------------
# 1. Golden (PyTorch CPU reference, ported from
#    examples/TileOPs/tileops/workloads/sequence_modeling.py)
# ---------------------------------------------------------------------------
def _rmsnorm(x, w, eps=1e-6):
    """fp32 statistics + fp32 normalisation; returns (normed, rrms)."""
    x_f = x.float()
    rrms = (x_f**2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    return x_f * rrms * w.float(), rrms.squeeze(-1)


def golden_engram_gate_conv_fwd(H, k, v, rms_w_h, rms_w_v, conv_w, eps=1e-6):
    """Reference semantics (source operator, independent of the NPU structure).

    Rounding points that matter:
      * ``vhat = fl_dtype(alpha * v)`` is what ``rrms_v`` is measured on,
      * the convolution consumes that rounded ``vhat``,
      * ``Y`` adds back the *unrounded* ``alpha * v`` residual.
    """
    d = H.shape[2]
    h_norm, rrms_h = _rmsnorm(H, rms_w_h, eps)  # H and k share rms_w_h
    k_norm, rrms_k = _rmsnorm(k, rms_w_h, eps)

    dot = (h_norm * k_norm).sum(dim=-1, keepdim=True)
    alpha = torch.sigmoid(dot / (d**0.5))

    v_hat = alpha * v.float()
    v_hat_norm, rrms_v = _rmsnorm(v_hat.to(H.dtype), rms_w_v, eps)

    v_perm = v_hat_norm.float().permute(0, 2, 1)  # (M, d, T)
    v_padded = F.pad(v_perm, (KS - 1, 0))  # causal left pad, dilation 1
    conv_w_expanded = conv_w.float().T.unsqueeze(1)  # depthwise (d, 1, KS)
    conv_out = F.conv1d(v_padded, conv_w_expanded, groups=d).permute(0, 2, 1)

    Y = F.silu(conv_out) + v_hat.float()

    return (
        Y.to(H.dtype),
        v_hat.to(H.dtype),
        alpha.squeeze(-1).float(),
        rrms_h.float(),
        rrms_k.float(),
        rrms_v.float(),
    )


# ---------------------------------------------------------------------------
# 2. Build-time geometry (DESIGN.md 5.2) and UB budget (DESIGN.md 4.5)
# ---------------------------------------------------------------------------
def _divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def geometry(M, seq_len, d):
    """Tile rule of DESIGN.md 5.2: RC always divides seq_len (no tail code path)."""
    dw = -(-d // VL) * VL
    rows = M * seq_len
    target = max(1, -(-rows // CORES))
    cands = [dd for dd in _divisors(seq_len) if dd >= target]
    rc = min(cands) if cands else 1
    if rc > 1 and M * (seq_len // rc) < MIN_GRID:
        rc = 1
    nch = seq_len // rc
    return {"DW": dw, "nchunks": dw // VL, "RC": rc, "nch": nch, "grid": M * nch}


def _ub_footprint(dw, elem, rc):
    """7 dtype rows + cw (4) + w2h + cwv (4) + slot (rc+3 rows) + st."""
    return (
        (rc + 3) * dw * 4  # slot (fp32 row window, one row per pipeline row)
        + 4 * dw * 4  # cwv_ub (fp32 folded conv weights)
        + dw * 4  # w2h_ub (fp32)
        + 7 * dw * elem  # h/k/v/vh/y/wh/wv
        + 4 * dw * elem  # cw_ub
        + MEMBERS * (rc + 3) * STATS_STRIDE * 4  # st
    )


# ---------------------------------------------------------------------------
# 3. Kernel
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=64)
def _engram_gate_conv_fwd_kernel(M: int, seq_len: int, d: int, eps: float, dtype: str):
    """Compile one specialised kernel for a (M, seq_len, d, eps, dtype) geometry."""
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
    if seq_len % rc:  # invariant of the tile rule: kept as a build-time assertion
        raise AssertionError(f"RC={rc} does not divide seq_len={seq_len}")
    elem = ELEM_BYTES[dtype]
    footprint = _ub_footprint(dw, elem, rc)
    if footprint * 1.2 > UB_BYTES:
        raise ValueError(f"UB footprint {footprint} B x1.2 exceeds {UB_BYTES} B (d={d}, RC={rc})")

    # d is a multiple of the fp32 vector width: every lane mask is the full
    # 64-lane pattern, so the per-chunk mask build folds into the hoisted
    # predicate register (build-time decision, tail path keeps update_mask).
    full_lanes = (d % VL == 0)

    fp32 = "float32"
    dim_f = float(d)
    inv_d = 1.0 / dim_f
    inv_sqrt_d = 1.0 / (d**0.5)
    taps = KS

    @T.prim_func
    def main(
        H: T.Tensor((M, seq_len, d), dtype),
        K: T.Tensor((M, seq_len, d), dtype),
        V: T.Tensor((M, seq_len, d), dtype),
        WH: T.Tensor((d,), dtype),
        WV: T.Tensor((d,), dtype),
        CW: T.Tensor((taps, d), dtype),
        Y: T.Tensor((M, seq_len, d), dtype),
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

            # (rc + 3, dw) windows: the whole per-block row window (owned rows plus
            # the 3 causal halo rows) in one buffer per input tensor.
            h_ub = T.alloc_shared((rc + 3, dw), dtype)
            k_ub = T.alloc_shared((rc + 3, dw), dtype)
            v_ub = T.alloc_shared((rc + 3, dw), dtype)
            y_ub = T.alloc_shared((rc + 3, dw), dtype)
            vh_ub = T.alloc_shared((rc + 3, dw), dtype)
            wh_ub = T.alloc_shared((dw,), dtype)
            wv_ub = T.alloc_shared((dw,), dtype)
            cw_ub = T.alloc_shared((taps, dw), dtype)
            w2h_ub = T.alloc_shared((dw,), fp32)
            cwv_ub = T.alloc_shared((taps, dw), fp32)
            slot = T.alloc_shared((rc + 3, dw), fp32)
            st = T.alloc_shared((MEMBERS, rc + 3, STATS_STRIDE), fp32)
            # One version each: the ring window is carried across rows on purpose and
            # the row buffers are written then read inside the same row.
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

            # -------- prologue: w_h^2 and conv_w * rms_w_v, once per core --------
            # A SIMD_VF block may not read GM, so every parameter is staged to UB
            # first (the T.copy calls stay outside the frame).
            T.copy(WH, wh_ub[:d])
            T.copy(WV, wv_ub[:d])
            T.copy(CW, cw_ub[:, :d])
            with T.SimdVF():
                # The three causal halo rows start at zero.  A vector store is used
                # instead of T.clear: T.clear on a multi-KB fp32 UB buffer lowers to
                # a scalar float2 loop (DESIGN.md 9.2).
                full_p = T.simd.pset(32)
                zero_p = T.simd.vdup(T.float32(0), fp32, full_p)
                for hz in T.serial(3):
                    for ch in T.serial(nchunks):
                        T.simd.vsts(slot[hz, ch * VL], zero_p, mask=full_p)
                for ch in T.serial(nchunks):
                    mask = full_p if full_lanes else T.simd.update_mask(T.min(VL, d - ch * VL), 32)
                    wv = T.simd.vcvt(T.simd.vld(wh_ub[ch * VL], dist="UNPK_B16"), fp32, part=0)
                    T.simd.vsts(w2h_ub[ch * VL], T.simd.vmul(wv, wv, mask))
                for p in T.serial(taps):
                    for ch in T.serial(nchunks):
                        mask = full_p if full_lanes else T.simd.update_mask(T.min(VL, d - ch * VL), 32)
                        cf = T.simd.vcvt(
                            T.simd.vld(cw_ub[p, ch * VL], dist="UNPK_B16"), fp32, part=0
                        )
                        vf = T.simd.vcvt(T.simd.vld(wv_ub[ch * VL], dist="UNPK_B16"), fp32, part=0)
                        T.simd.vsts(cwv_ub[p, ch * VL], T.simd.vmul(cf, vf, mask))

            # -------- window load: 6 bulk copies, no DMA inside the row loop ------
            # Only the block that owns the start of a sequence (t0 < 3) can have
            # halo rows that do not exist; those slots are zero filled, which is
            # the causal zero boundary of the reference.  Every other block loads
            # its 3 halo rows with one 3-row copy per tensor.
            if t0 < 3:
                with T.SimdVF():
                    zfull = T.simd.pset(32)
                    zb = T.simd.vcvt(
                        T.simd.vdup(T.float32(0), fp32, zfull),
                        dtype,
                        round="ROUND_R",
                        sat=False,
                        part=0,
                    )
                    for hz in T.serial(3):
                        for ch in T.serial(nchunks):
                            T.simd.vsts(h_ub[hz, ch * VL], zb, zfull, dist="PK_B32")
                            T.simd.vsts(k_ub[hz, ch * VL], zb, zfull, dist="PK_B32")
                            T.simd.vsts(v_ub[hz, ch * VL], zb, zfull, dist="PK_B32")
                for hz in range(3):  # build-time unroll over the 3 halo slots
                    rr = t0 - 3 + hz
                    if rr >= 0:
                        T.copy(H[b, rr, :], h_ub[hz, :d])
                        T.copy(K[b, rr, :], k_ub[hz, :d])
                        T.copy(V[b, rr, :], v_ub[hz, :d])
            else:
                T.copy(H[b, t0 - 3 : t0, :], h_ub[0:3, :d])
                T.copy(K[b, t0 - 3 : t0, :], k_ub[0:3, :d])
                T.copy(V[b, t0 - 3 : t0, :], v_ub[0:3, :d])
            T.copy(H[b, t0 : t0 + rc, :], h_ub[3 : 3 + rc, :d])
            T.copy(K[b, t0 : t0 + rc, :], k_ub[3 : 3 + rc, :d])
            T.copy(V[b, t0 : t0 + rc, :], v_ub[3 : 3 + rc, :d])

            # -------- row pipeline: RC owned rows + 3 halo rows ------------------
            for i in T.serial(rc + 3):
                r = t0 - 3 + i
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
                    if True:  # the head rows are zero filled, so every row is computed
                        # Three row statistics in one scan: sum h^2, sum k^2 and
                        # sum (h*k)*w_h^2 (the gate dot with the r_h/r_k factors
                        # factored out, DESIGN.md 1.6.1 A1).
                        hsq[0] = zero_f
                        ksq[0] = zero_f
                        hk[0] = zero_f
                        for ch in T.serial(nchunks):
                            mask = full if full_lanes else T.simd.update_mask(T.min(VL, d - ch * VL), 32)
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
                                hk[0],
                                T.simd.vmul(T.simd.vmul(hf, kf, mask), w2, mask),
                                full,
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
                        # vhat = fl_dtype(alpha*v): the dtype round trip feeds both
                        # the output row and the rrms_v statistic (DESIGN.md 0.1).
                        vsq[0] = zero_f
                        for ch in T.serial(nchunks):
                            mask = full if full_lanes else T.simd.update_mask(T.min(VL, d - ch * VL), 32)
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
                        # r_v is already a lane-broadcast fp32 vector: use the
                        # register directly instead of the store -> BRC_B32 reload
                        # round trip and its VST_VLD barrier (branch r1_c2).
                        rvb = rv_v[0]
                        # g = vhat * r_v is computed once per row and reused by all
                        # four taps (DESIGN.md 1.6.1 A3).  The rounded vhat is
                        # re-derived from the DMA-filled v_ub instead of being read
                        # back from the 16-bit row buffer: a 16-bit row written with
                        # dist="PK_B32" only reads back through dist="UNPK_B16" after
                        # an explicit mem_bar, so the recomputation (the same
                        # deterministic expression as the first pass) keeps this pass
                        # free of a vector read-after-write dependency.
                        # The window keeps rc + 3 rows and is indexed by the row
                        # counter -- the same ring of four rows the design describes,
                        # written without a modular buffer index.
                        for ch in T.serial(nchunks):
                            mask = full if full_lanes else T.simd.update_mask(T.min(VL, d - ch * VL), 32)
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
                            back = T.simd.vcvt(narrowed, fp32, part=0)
                            T.simd.vsts(slot[i, ch * VL], T.simd.vmul(back, rvb, full), full)
                    if i >= 3:
                        # The own-window read below depends on vector stores issued
                        # earlier in this kernel (the halo rows and this row's g).
                        # Vector store -> vector load visibility is not implicit
                        # here, so the window needs an explicit memory barrier
                        # (without it the results are run-to-run nondeterministic).
                        T.simd.mem_bar("VST_VLD")
                        # Owned row: 4 tap convolution, SiLU, unrounded residual.
                        for ch in T.serial(nchunks):
                            mask = full if full_lanes else T.simd.update_mask(T.min(VL, d - ch * VL), 32)
                            acc = T.simd.alloc_local((1,), fp32)
                            acc[0] = zero_f
                            for p in T.unroll(taps):
                                acc[0] = T.simd.vadd(
                                    acc[0],
                                    T.simd.vmul(
                                        T.simd.vld(slot[i - 3 + p, ch * VL]),
                                        T.simd.vld(cwv_ub[p, ch * VL]),
                                        mask,
                                    ),
                                    full,
                                )
                            silu = T.simd.vdiv(
                                acc[0],
                                T.simd.vadd(
                                    one_f,
                                    T.simd.vexp(T.simd.vneg(acc[0], full), full),
                                    full,
                                ),
                                full,
                            )
                            vf = T.simd.vcvt(
                                T.simd.vld(v_ub[i, ch * VL], dist="UNPK_B16"), fp32, part=0
                            )
                            y = T.simd.vadd(silu, T.simd.vmul(vf, alpha_v[0], mask), mask)
                            T.simd.vsts(
                                y_ub[i, ch * VL],
                                T.simd.vcvt(y, dtype, round="ROUND_R", sat=False, part=0),
                                full,
                                dist="PK_B32",
                            )

            # -------- owned rows: one 2-D store per output, not one per row ------
            # The owned rows are contiguous in GM, so a single (rc, d) region copy
            # replaces rc per-row copies, which each drained PIPE_MTE3 before the
            # next store (TRAP-950-rowstore-mte3-drain; branch r1_c3 measured the
            # MTE3 busy ratio drop 0.131 -> 0.033).
            T.copy(y_ub[3 : 3 + rc, :d], Y[b, t0 : t0 + rc, :])
            T.copy(vh_ub[3 : 3 + rc, :d], VHAT[b, t0 : t0 + rc, :])

            # -------- four fp32 row scalars: one strided copy per member --------
            T.copy(st[0, 3 : 3 + rc, 0:1], Alpha[b, t0 : t0 + rc])
            T.copy(st[1, 3 : 3 + rc, 0:1], RH[b, t0 : t0 + rc])
            T.copy(st[2, 3 : 3 + rc, 0:1], RK[b, t0 : t0 + rc])
            T.copy(st[3, 3 : 3 + rc, 0:1], RV[b, t0 : t0 + rc])

    return tilelang.compile(main, target="ascend", out_idx=[6, 7, 8, 9, 10, 11])


# ---------------------------------------------------------------------------
# 4. Case runner (golden on CPU, kernel on the NPU, six outputs compared)
# ---------------------------------------------------------------------------
_RESULTS = []  # (level, name, ok, detail)
_MAXDIFF = {name: 0.0 for name in OUT_NAMES}


def _record(level, name, ok, detail=""):
    _RESULTS.append((level, name, ok, detail))


def _rand(shape, dtype, seed, scale=1.0):
    gen = torch.Generator().manual_seed(seed)
    return (torch.randn(shape, generator=gen, dtype=torch.float32) * scale).to(dtype)


def _make_inputs(M, seq_len, d, dtype, seed, special=None, manifest=False):
    """Build the six inputs on the CPU (``manifest`` uses the workload recipe)."""
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

    if special == "zeros":
        H.zero_()
        k.zero_()
        v.zero_()
    elif special == "zero-rows":
        # Exactly zero rows: alpha = 0.5, rrms = eps**-0.5, vhat = 0, conv = 0.
        for row in (1, seq_len - 1):
            H[:, row, :] = 0
            k[:, row, :] = 0
            v[:, row, :] = 0
    elif special == "non-finite":
        # inf in v and NaN in H / one norm weight.  inf in H or k is covered by
        # the B-4 boundary case instead (the folded gate dot reaches inf*0 = NaN
        # on a different path than the reference's h_norm * k_norm).
        H[0, 0, 0] = float("nan")
        v[0, 0, 1] = float("inf")
        v[0, 1, 2] = float("-inf")
        v[0, min(2, seq_len - 1), 3] = float("nan")
        wh[0] = float("nan")
    elif special == "inf-hidden":
        H[0, 0, 0] = float("inf")
        k[0, 1, 1] = float("-inf")
        v[0, 2, 2] = float("nan")
    elif special == "nan-only":
        H[0, 1, 0] = float("nan")
        v[0, 0, 1] = float("nan")
    elif special is not None:
        raise ValueError(f"special: unknown pattern {special!r}")
    return H, k, v, wh, wv, cw


def _diff(got, gold):
    """Largest absolute difference in float space, plus the NaN mask mismatch."""
    a, b = got.float(), gold.float()
    diff = (a - b).abs()
    diff = torch.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)
    nan_mismatch = int(((a != a) != (b != b)).sum())
    return (float(diff.max()) if diff.numel() else 0.0), nan_mismatch


def run_case(level, name, M, seq_len, d, dtype, seed=0, special=None, manifest=False, repeat=1):
    """Build one geometry, run it on the NPU, compare all six outputs."""
    geo = geometry(M, seq_len, d)
    per_out = {}
    elapsed = float("nan")
    try:
        H, k, v, wh, wv, cw = _make_inputs(M, seq_len, d, dtype, seed, special, manifest)
        gold = golden_engram_gate_conv_fwd(H, k, v, wh, wv, cw, 1e-6)

        args = [t.to("npu") for t in (H, k, v, wh, wv, cw)]
        fn = _engram_gate_conv_fwd_kernel(M, seq_len, d, 1e-6, dtype)
        start = time.perf_counter()
        got = fn(*args)
        torch.npu.synchronize()
        elapsed = (time.perf_counter() - start) * 1e3
        if repeat > 1:  # out_idx allocates fresh outputs, so repeats are independent
            start = time.perf_counter()
            for _ in range(repeat):
                fn(*args)
            torch.npu.synchronize()
            elapsed = (time.perf_counter() - start) * 1e3 / repeat
        got = tuple(g.cpu() for g in got)

        problems = []
        for idx, out_name in enumerate(OUT_NAMES):
            if got[idx].shape != gold[idx].shape or got[idx].dtype != gold[idx].dtype:
                problems.append(
                    f"{out_name}: got {tuple(got[idx].shape)}/{got[idx].dtype}, "
                    f"want {tuple(gold[idx].shape)}/{gold[idx].dtype}"
                )
                continue
            diff, nan_mismatch = _diff(got[idx], gold[idx])
            per_out[out_name] = diff
            _MAXDIFF[out_name] = max(_MAXDIFF[out_name], diff)
            if nan_mismatch:
                problems.append(f"{out_name}: {nan_mismatch} NaN positions differ")
            if out_name in ("Y", "vhat"):
                atol, rtol = ATOL[dtype], RTOL[dtype]
            else:
                atol, rtol = SCALAR_ATOL, SCALAR_RTOL
            try:
                torch.testing.assert_close(
                    got[idx], gold[idx], atol=atol, rtol=rtol, equal_nan=True
                )
            except AssertionError:
                problems.append(f"{out_name}: max_diff={diff:.3e} > atol={atol:g}")
        if problems:
            raise AssertionError("; ".join(problems))
    except AssertionError as exc:
        _record(level, name, False, f"{str(exc)[:300]}")
        return False
    except (ValueError, RuntimeError) as exc:  # reported, never swallowed
        _record(level, name, False, f"{type(exc).__name__}: {str(exc)[:300]}")
        return False
    _record(level, name, True)
    shown = " ".join(f"{n}={per_out[n]:.2e}" for n in OUT_NAMES)
    print(
        f"       M={M} S={seq_len} d={d} {dtype} RC={geo['RC']} grid={geo['grid']} "
        f"nchunks={geo['nchunks']} {shown} launch={elapsed:.3f} ms",
        flush=True,
    )
    return True


def _finish(level, strict):
    rows = [r for r in _RESULTS if r[0] == level]
    failed = [r for r in rows if not r[2]]
    print(
        f"[{level}] {len(rows) - len(failed)}/{len(rows)} cases passed | "
        + " ".join(f"{n}_max_diff={_MAXDIFF[n]:.3e}" for n in OUT_NAMES),
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
# 5. Layered cases
# ---------------------------------------------------------------------------
def run_L0():
    """Gate: the four manifest workloads, the tail-lane path, seq_len < 3, zero
    rows and non-finite propagation."""
    run_case("L0", "L0-1 train-s32-fp16", 1, 32, 256, "float16", seed=1)
    run_case("L0", "L0-2 train-s32-bf16", 1, 32, 256, "bfloat16", seed=2)
    run_case("L0", "L0-3 train-s64-fp16", 2, 64, 512, "float16", seed=3)
    run_case("L0", "L0-4 train-s128-bf16", 1, 128, 256, "bfloat16", seed=4)
    run_case("L0", "L0-5 train-s96-bf16", 3, 96, 768, "bfloat16", seed=5)
    # Tail lane path: d = 196 -> DW = 256, masks 64/64/64/4.
    run_case("L0", "L0-6 d196-tail-lane-bf16", 1, 8, 196, "bfloat16", seed=6)
    # seq_len < 3: only the p=3 tap contributes, slot[0..2] stay zero.
    run_case("L0", "L0-7 seq1-fp16", 1, 1, 64, "float16", seed=7)
    run_case("L0", "L0-8 seq2-fp16", 1, 2, 64, "float16", seed=8)
    run_case("L0", "L0-9 zero-rows-fp16", 1, 8, 64, "float16", seed=9, special="zero-rows")
    run_case("L0", "L0-10 nonfinite-bf16", 1, 8, 64, "bfloat16", seed=10, special="non-finite")
    run_case("L0", "L0-11 nan-only-fp16", 1, 8, 64, "float16", seed=11, special="nan-only")
    _finish("L0", strict=True)


def run_L1():
    """Cross dtype / geometry coverage and the multi-batch tile mapping."""
    run_case("L1", "L1-1 train-s128-fp16", 1, 128, 256, "float16", seed=21, repeat=3)
    run_case("L1", "L1-2 train-s96-fp16", 3, 96, 768, "float16", seed=22, repeat=3)
    run_case("L1", "L1-3 train-s64-bf16", 2, 64, 512, "bfloat16", seed=23, repeat=3)
    run_case("L1", "L1-4 three-batch-s64-bf16", 3, 64, 512, "bfloat16", seed=24, repeat=3)
    run_case("L1", "L1-5 train-s32-bf16", 1, 32, 256, "bfloat16", seed=25, repeat=3)
    _finish("L1", strict=True)


def run_L2():
    """The manifest workloads with the manifest input recipe (informative)."""
    run_case("L2", "L2-1 train-s32", 1, 32, 256, "float16", seed=31, manifest=True, repeat=3)
    run_case("L2", "L2-2 train-s64", 2, 64, 512, "float16", seed=32, manifest=True, repeat=3)
    run_case("L2", "L2-3 train-s128", 1, 128, 256, "bfloat16", seed=33, manifest=True, repeat=3)
    run_case("L2", "L2-4 train-s96", 3, 96, 768, "bfloat16", seed=34, manifest=True, repeat=3)
    _finish("L2", strict=False)


def run_boundary():
    """Non-divisible d, prime seq_len (RC = 1 oversubscription), zero, non-finite."""
    # d = 200 -> DW = 256, tail mask 64/64/64/8.
    run_case("boundary", "B-1 d200-tail-lane", 1, 8, 200, "bfloat16", seed=41)
    # seq_len = 95 is prime: RC = 1 and grid = 95 > 56 (two hardware waves).
    run_case("boundary", "B-2 seq95-prime", 1, 95, 256, "bfloat16", seed=42)
    # All-zero inputs: alpha = 0.5, rrms = eps**-0.5, Y = 0.
    run_case("boundary", "B-3 all-zero-fp16", 1, 16, 128, "float16", seed=43, special="zeros")
    # An inf in H / k reaches the gate dot through a different path than the
    # reference (folded h*k*w^2); reported as evidence, not as a gate.
    run_case("boundary", "B-4 inf-hidden-bf16", 1, 8, 64, "bfloat16", seed=44, special="inf-hidden")
    # d = 1: a single valid lane, mask = update_mask(1).
    run_case("boundary", "B-5 d1-single-lane", 1, 4, 1, "float16", seed=45)
    _finish("boundary", strict=False)


def _verdict():
    failed = [r for r in _RESULTS if not r[2] and r[0] in ("L0", "L1")]
    if failed:
        print(f"[PRECISION_FAIL] {len(failed)} blocking case(s) failed", flush=True)
    else:
        print(
            "[PRECISION_PASS] " + " ".join(f"{n}={_MAXDIFF[n]:.3e}" for n in OUT_NAMES),
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0", choices=["L0", "L1", "L2", "boundary", "all"])
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
