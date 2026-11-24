# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""`_engram_step_kernel` on Ascend 950PR (SIMT): launch 2 of 2 of EngramDecodeFwdOp.

One decode step of everything that reduces over ``d``: ``kv[0]`` and ``h_t`` are
RMS-normalized with the shared ``rms_w_h``, their dot product gives the gate scalar
``alpha``, the gate scales ``kv[1]`` before the third RMSNorm, and that normalized
row is both the new last history slot and the ``conv_w[w-1]`` term of a ``w``-tap
dilated causal convolution; SiLU plus the ``alpha * v`` residual is ``y_t``.

Implementation follows DESIGN.md (frozen, REVIEW.md passed):

* One core per batch row (``T.Kernel(batch)``), one lane per element
  (``T.SimtVF(threads=min(512, tile))`` + ``T.Parallel(tile, coalesced_width=1)``);
  the four row statistics (``S_h``, ``S_k``, ``P = sum h*k*w_h^2``, ``S_v``) come
  from four concurrent ``T.alloc_reducer(op="sum")`` epochs in a single pass
  (DESIGN.md 0.6 R2/R3, 1.6.1 A1/A2).
* Two row passes only (the source structure has 15 elementwise passes): phase 1
  produces the statistics, phase 2 the three outputs, with the scalar chain between
  them -- that boundary is inherent (DESIGN.md 1.6.0 R2), so ``alpha``/``rv`` are
  carried in ``(1,)`` fragments.
* ``kv`` is read as fp32 and normalized directly, with no ``dtype -> fp32`` round
  trip (DESIGN.md 0.5 #7 / 1.6.1 B1: the round trip is a semantics deviation,
  measured at 1.56e-02 absolute).  The cache left shift belongs to
  ``_engram_project_kernel``; this launch only writes
  ``new_conv_state[bx, max_conv_len-1, :]`` (DESIGN.md 0.5 #11).
* ``x * (1/d)`` / ``dot * (1/sqrt(d))`` folded constants (DESIGN.md 1.6.1 A6).
* SiLU keeps the source form ``c * (1/(1+exp(-c)))``; the one-division variant
  ``c/(1+exp(-c))`` was measured and rejected (DESIGN.md 1.6.1 B0).
* ``tile == d_padded`` for every power-of-two padded width, so the trailer predicate
  is not generated at all; a non-power-of-two ``d_padded`` (e.g. ``d=600`` ->
  ``d_padded=768``, ``tile=1024``) takes the masked branch (DESIGN.md 6.4 / E8).

Cross-launch contract: ``kv`` is fp32 ``(2, batch, d_padded)`` produced by launch 1;
``d_padded = align_up(d, 256)`` and the host zero-fills the padded columns, which is
what lets the row reduction run over the full padded width while the norms divide by
``d`` (DESIGN.md 1.6.1 A0).

run: python _engram_step_kernel.py --level L0|L1|L2|boundary|all
"""

import argparse
import functools
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401 -- registers torch.npu
import tilelang
import tilelang.ascend.language as T

# Offsets the row of every tensor along the last axis; 256 keeps the trailing axis a
# multiple of the 16-element vector granularity (host contract of the source op).
ALIGNMENT = 256

# DESIGN.md 8.2 == examples/TileOPs/tests/ops/test_engram.py::_tol_decode.
TOL = {"float16": (0.05, 0.05), "bfloat16": (0.1, 0.05)}

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT / "examples" / "TileOPs") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "examples" / "TileOPs"))

# The torch reference of the source repository is the semantic ground truth
# (DESIGN.md 8.1).  ``golden_engram_decode`` below re-implements that semantics with
# torch ops directly, and every case cross-checks the two references bit for bit, so
# the golden cannot drift from the reference it was derived from.
from tileops.workloads.sequence_modeling import engram_decode_step_torch  # noqa: E402

_DIFFS = []


# ---------------------------------------------------------------------------
# Golden (PyTorch CPU reference, from the operator semantics only)
# ---------------------------------------------------------------------------
def _rmsnorm_torch(x, weight, eps):
    """RMSNorm over the last axis in fp32, mirroring the shape the reference uses."""
    x_f = torch.unsqueeze(x.float(), 1)
    rrms = torch.rsqrt(torch.add(torch.mean(torch.square(x_f), dim=-1, keepdim=True), eps))
    return torch.squeeze(torch.mul(torch.mul(x_f, rrms), weight.float()), 1)


def golden_engram_decode(e_t, h_t, conv_state, W_K, W_V, rms_w_h, rms_w_v, conv_w,
                         max_conv_len, dilation, eps=1e-6):
    """Whole-operator reference: ``y_t`` and the full ``max_conv_len``-row history.

    Implemented directly with torch ops (fp32 arithmetic, only the two returned tensors
    are rounded to the input dtype), following the source semantics of DESIGN.md 0.1:
    the two projections, three RMSNorms (denominator ``d``, shared ``rms_w_h`` for h and
    k), the sigmoid gate, the ``w``-tap dilated causal convolution, SiLU plus the
    ``alpha * v`` residual, and a cache append that always yields ``max_conv_len`` rows
    with this step's normalized value last.

    ``conv_state`` may hold fewer than ``max_conv_len`` rows; the host left padding is
    part of this reference, so a partial cache needs no preprocessing by the caller.
    """
    batch, d = h_t.shape
    w = conv_w.shape[0]
    cache_len = conv_state.shape[1]

    # Launch-1 outputs, recomputed here so the golden stands on its own: fp32 and
    # without any rounding back to the input dtype (DESIGN.md 0.5 #7).
    k = torch.matmul(e_t.float(), W_K.float())
    v = torch.matmul(e_t.float(), W_V.float())

    h_norm = _rmsnorm_torch(h_t, rms_w_h, eps)
    k_norm = _rmsnorm_torch(k, rms_w_h, eps)
    dot = torch.sum(torch.mul(h_norm, k_norm), dim=-1, keepdim=True)
    alpha = torch.sigmoid(torch.div(dot, float(d) ** 0.5))
    v_hat = torch.mul(alpha, v)
    v_hat_norm = _rmsnorm_torch(v_hat, rms_w_v, eps)

    # Host contract: left-pad the cache to max_conv_len; the taps read its fp32 view.
    state_pad = max_conv_len - cache_len
    padded_history = (
        F.pad(conv_state, (0, 0, state_pad, 0)) if state_pad > 0 else conv_state
    )
    padded_state = padded_history.float()

    conv_out = torch.zeros(batch, d, dtype=torch.float32, device=h_t.device)
    for p in range(w - 1):
        state_idx = max_conv_len - (w - 1 - p) * dilation
        if 0 <= state_idx < max_conv_len:
            conv_out = conv_out + torch.mul(
                conv_w[p].float().unsqueeze(0), padded_state[:, state_idx, :]
            )
    conv_out = conv_out + torch.mul(
        conv_w[w - 1].float().unsqueeze(0), v_hat_norm
    )

    # Drop the oldest slot and append this step: always max_conv_len rows.
    new_conv_state = torch.cat(
        [
            padded_history[:, 1:, :],
            v_hat_norm.unsqueeze(1).to(conv_state.dtype),
        ],
        dim=1,
    )

    y_t = torch.add(F.silu(conv_out), v_hat)
    return y_t.to(h_t.dtype), new_conv_state


def host_kv(e_t, W_K, W_V, d_pad):
    """Step-kernel ``kv`` input: the fp32 projection of launch 1 (``out_idx=[]``).

    Same expression as the reference (``e_t.float() @ W.float()``), fp32 and without
    any rounding back to the input dtype (DESIGN.md 0.5 #7), zero-padded on the last
    axis to ``d_pad``.
    """
    k = e_t.float() @ W_K.float()
    v = e_t.float() @ W_V.float()
    pad = d_pad - k.shape[1]
    if pad:
        k = F.pad(k, (0, pad))
        v = F.pad(v, (0, pad))
    return torch.stack([k, v], dim=0)


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


@functools.lru_cache(maxsize=None)
def _build_step(batch, d, d_padded, max_conv_len, conv_kernel_size, dilation, eps,
                dtype_str, threads):
    """Everything that reduces over ``d``, one core per batch row.

    ``tile`` is the row width of the lane-parallel region (``>= d_padded``) and
    ``threads`` the lane count, so each lane owns ``tile // threads`` elements.
    """
    w = conv_kernel_size
    tile = _next_pow2(d_padded)
    exact = tile == d_padded
    inv_d = 1.0 / float(d)
    inv_sqrt_d = 1.0 / math.sqrt(float(d))

    @T.prim_func
    def engram_decode_step(
        h_t: T.Tensor((batch, d_padded), dtype_str),
        kv: T.Tensor((2, batch, d_padded), "float32"),
        conv_state: T.Tensor((batch, max_conv_len, d_padded), dtype_str),
        rms_w_h: T.Tensor((d_padded,), dtype_str),
        rms_w_v: T.Tensor((d_padded,), dtype_str),
        conv_w: T.Tensor((w, d_padded), dtype_str),
        y_t: T.Tensor((batch, d_padded), dtype_str),
        new_conv_state: T.Tensor((batch, max_conv_len, d_padded), dtype_str),
    ):
        with T.Kernel(batch) as bx:
            h_ub = T.alloc_shared((d_padded,), dtype_str)
            k_ub = T.alloc_shared((d_padded,), "float32")
            v_ub = T.alloc_shared((d_padded,), "float32")
            wh_ub = T.alloc_shared((d_padded,), dtype_str)
            wv_ub = T.alloc_shared((d_padded,), dtype_str)
            cw_ub = T.alloc_shared((w, d_padded), dtype_str)
            # Zero rows when conv_kernel_size == 1: legal here and never read.
            tap_ub = T.alloc_shared((w - 1, d_padded), dtype_str)
            y_ub = T.alloc_shared((d_padded,), dtype_str)
            vn_ub = T.alloc_shared((d_padded,), dtype_str)

            T.copy(h_t[bx, :], h_ub)
            T.copy(kv[0, bx, :], k_ub)
            T.copy(kv[1, bx, :], v_ub)
            T.copy(rms_w_h, wh_ub)
            T.copy(rms_w_v, wv_ub)
            T.copy(conv_w, cw_ub)
            # Tap p is the value from (w-1-p)*dilation steps ago; against the cache as
            # it is on entry that is row max_conv_len - (w-1-p)*dilation, which the
            # shape rule max_conv_len >= dilation*(w-1) keeps in [0, max_conv_len-1].
            for p in range(w - 1):
                T.copy(
                    conv_state[bx, max_conv_len - (w - 1 - p) * dilation, :],
                    tap_ub[p, :],
                )

            with T.SimtVF(threads=threads):
                vf = T.alloc_fragment((tile,), "float32")
                vnf = T.alloc_fragment((tile,), "float32")
                cof = T.alloc_fragment((tile,), "float32")

                acc_h = T.alloc_reducer((1,), "float32", op="sum")
                acc_k = T.alloc_reducer((1,), "float32", op="sum")
                acc_p = T.alloc_reducer((1,), "float32", op="sum")
                acc_v = T.alloc_reducer((1,), "float32", op="sum")
                T.reducer_init(acc_h)
                T.reducer_init(acc_k)
                T.reducer_init(acc_p)
                T.reducer_init(acc_v)

                # Phase 1: one pass, four independent accumulation chains.  Each chain
                # keeps the source term order, so only the tree shape differs.
                if exact:
                    for i in T.Parallel(tile, coalesced_width=T.int32(1)):
                        vf[i] = v_ub[i]
                        T.reducer_update(
                            acc_h[0],
                            T.cast(h_ub[i], "float32") * T.cast(h_ub[i], "float32"),
                        )
                        T.reducer_update(acc_k[0], k_ub[i] * k_ub[i])
                        T.reducer_update(
                            acc_p[0],
                            T.cast(h_ub[i], "float32")
                            * k_ub[i]
                            * T.cast(wh_ub[i], "float32")
                            * T.cast(wh_ub[i], "float32"),
                        )
                        T.reducer_update(acc_v[0], v_ub[i] * v_ub[i])
                else:
                    # Trailer lanes contribute nothing; the norm denominators stay d.
                    for i in T.Parallel(tile, coalesced_width=T.int32(1)):
                        hv = T.if_then_else(
                            i < d_padded, T.cast(h_ub[i], "float32"), T.float32(0)
                        )
                        kk = T.if_then_else(i < d_padded, k_ub[i], T.float32(0))
                        vv = T.if_then_else(i < d_padded, v_ub[i], T.float32(0))
                        ww = T.if_then_else(
                            i < d_padded, T.cast(wh_ub[i], "float32"), T.float32(0)
                        )
                        vf[i] = vv
                        T.reducer_update(acc_h[0], hv * hv)
                        T.reducer_update(acc_k[0], kk * kk)
                        T.reducer_update(acc_p[0], hv * kk * ww * ww)
                        T.reducer_update(acc_v[0], vv * vv)

                sh = T.alloc_fragment((1,), "float32")
                sk = T.alloc_fragment((1,), "float32")
                sp = T.alloc_fragment((1,), "float32")
                sv = T.alloc_fragment((1,), "float32")
                T.finalize_reducer(acc_h, sh)
                T.finalize_reducer(acc_k, sk)
                T.finalize_reducer(acc_p, sp)
                T.finalize_reducer(acc_v, sv)

                # Scalar chain, once per row: gate from the hoisted dot product
                # (alpha = sigmoid(rh*rk*P/sqrt(d)), A2) and the variance hoisted out
                # of the gate (sum (alpha*v)^2 = alpha^2 * sum v^2, A3).
                alpha = T.alloc_fragment((1,), "float32")
                rv = T.alloc_fragment((1,), "float32")
                for i in T.Parallel(1):
                    alpha[i] = 1.0 / (
                        1.0
                        + T.exp(
                            -(T.rsqrt(sh[0] * inv_d + eps)
                              * T.rsqrt(sk[0] * inv_d + eps)
                              * sp[0] * inv_sqrt_d)
                        )
                    )
                    rv[i] = T.rsqrt(alpha[0] * alpha[0] * sv[0] * inv_d + eps)

                # Phase 2: one pass for the new history slot, the dilated convolution
                # and y_t; av = alpha*v is the same fp32 expression in both uses (A4).
                if exact:
                    for i in T.Parallel(tile, coalesced_width=T.int32(1)):
                        vnf[i] = alpha[0] * vf[i] * rv[0] * T.cast(wv_ub[i], "float32")
                        cof[i] = T.cast(cw_ub[w - 1, i], "float32") * vnf[i]
                        for p in range(w - 1):
                            cof[i] = cof[i] + T.cast(cw_ub[p, i], "float32") * T.cast(
                                tap_ub[p, i], "float32"
                            )
                        y_ub[i] = T.cast(
                            cof[i] * (1.0 / (1.0 + T.exp(-cof[i])))
                            + alpha[0] * vf[i],
                            dtype_str,
                        )
                        vn_ub[i] = T.cast(vnf[i], dtype_str)
                else:
                    for i in T.Parallel(tile, coalesced_width=T.int32(1)):
                        if i < d_padded:
                            vnf[i] = alpha[0] * vf[i] * rv[0] * T.cast(
                                wv_ub[i], "float32"
                            )
                            cof[i] = T.cast(cw_ub[w - 1, i], "float32") * vnf[i]
                            for p in range(w - 1):
                                cof[i] = cof[i] + T.cast(
                                    cw_ub[p, i], "float32"
                                ) * T.cast(tap_ub[p, i], "float32")
                            y_ub[i] = T.cast(
                                cof[i] * (1.0 / (1.0 + T.exp(-cof[i])))
                                + alpha[0] * vf[i],
                                dtype_str,
                            )
                            vn_ub[i] = T.cast(vnf[i], dtype_str)

            T.copy(y_ub, y_t[bx, :])
            T.copy(vn_ub, new_conv_state[bx, max_conv_len - 1, :])

    return tilelang.compile(engram_decode_step, target="ascend", out_idx=[])


def _engram_step_kernel(batch, d, d_padded, max_conv_len, conv_kernel_size, dilation,
                        eps, dtype):
    """Harness-shaped factory: returns ``_func(threads)`` like the reference module.

    ``threads`` defaults to ``min(512, tile)`` (DESIGN.md 5.6); the explicit parameter
    exists for the Stage 4 ``{128, 256, 512}`` A/B (DESIGN.md 1.6.3 (ii)).
    """

    def _func(threads=None):
        if threads is None:
            threads = default_config(d_padded)["threads"]
        return _build_step(
            batch, d, d_padded, max_conv_len, conv_kernel_size, dilation, float(eps),
            dtype, threads,
        )

    return _func


def default_config(d_padded):
    """Frozen Stage 3 launch configuration (DESIGN.md 5.6)."""
    return {"threads": min(512, _next_pow2(d_padded))}


# ---------------------------------------------------------------------------
# Precision comparing
# ---------------------------------------------------------------------------
def _align_up(value, multiple=ALIGNMENT):
    return (value + multiple - 1) // multiple * multiple


def _dtype_name(dtype):
    return str(dtype).replace("torch.", "")


def _pad_last(t, width):
    """Zero-fill the last axis up to ``width`` (the host contract, A0)."""
    pad = width - t.shape[-1]
    return F.pad(t, (0, pad)) if pad > 0 else t


def _bits_equal(got, want):
    """Bitwise equality, so -0.0 / inf / NaN payloads are compared too."""
    if got.shape != want.shape:
        return False
    return torch.equal(got.contiguous().view(torch.int16), want.contiguous().view(torch.int16))


def _sentinel(batch, max_conv_len, d_pad, dtype):
    """Prefill of ``new_conv_state``: distinct per (row, column), finite, exactly
    representable, so any write to a row this launch does not own shows up."""
    row = torch.arange(max_conv_len, dtype=torch.float32).view(1, max_conv_len, 1)
    col = torch.arange(d_pad, dtype=torch.float32).view(1, 1, d_pad)
    pattern = 0.01 * (row + 1.0) * (col + 1.0)
    return pattern.to(dtype).expand(batch, max_conv_len, d_pad).contiguous()


def _prepare(batch, d, max_conv_len, conv_kernel_size, dilation, dtype, cache_len,
             d_pad, d_mem, seed, weight_scale, state_scale, conv_scale, h_scale):
    """Inputs for one step, unpadded (golden) and padded (launch), from one seed."""
    d_mem = d if d_mem is None else d_mem
    torch.manual_seed(seed)
    e_t = torch.randn(batch, d_mem, dtype=dtype) * 0.1
    h_t = torch.randn(batch, d, dtype=dtype) * h_scale
    conv_state = torch.randn(batch, cache_len, d, dtype=dtype) * state_scale
    w_k = torch.randn(d_mem, d, dtype=dtype) * 0.02
    w_v = torch.randn(d_mem, d, dtype=dtype) * 0.02
    rms_w_h = (1.0 + weight_scale * torch.randn(d)).to(dtype)
    rms_w_v = (1.0 + weight_scale * torch.randn(d)).to(dtype)
    conv_w = torch.randn(conv_kernel_size, d, dtype=dtype) * conv_scale

    dev = "npu"
    full_state = (
        F.pad(conv_state, (0, 0, max_conv_len - cache_len, 0))
        if cache_len < max_conv_len
        else conv_state
    )
    launch = {
        "h": _pad_last(h_t, d_pad).to(dev),
        "kv": host_kv(e_t, w_k, w_v, d_pad).to(dev),
        "state": _pad_last(full_state, d_pad).to(dev),
        "wh": _pad_last(rms_w_h, d_pad).to(dev),
        "wv": _pad_last(rms_w_v, d_pad).to(dev),
        "cw": _pad_last(conv_w, d_pad).to(dev),
    }
    ref_args = (e_t, h_t, conv_state, w_k, w_v, rms_w_h, rms_w_v, conv_w)
    return launch, ref_args


def _run_case(batch, d, max_conv_len, conv_kernel_size, dilation, dtype, tag,
              cache_len=None, d_pad=None, d_mem=None, eps=1e-6, seed=0, threads=None,
              weight_scale=0.1, state_scale=0.1, conv_scale=0.02, h_scale=1.0,
              launches=3):
    """Compile, launch ``launches`` times and check one shape.

    Assertions (DESIGN.md 8.1, step-only split):
      1. ``y_t[:, :d]`` matches the reference within the op tolerance;
      2. ``new_conv_state[:, mcl-1, :d]`` matches the reference likewise;
      3. every other history row is bit-identical to the prefill, i.e. this launch
         writes the last slot and nothing else (the shift belongs to launch 1);
      4. the whole ``d_pad`` row was written and the padded columns are exactly 0;
      5. repeated launches of one compiled kernel agree bitwise (determinism).
    """
    d_pad = _align_up(d) if d_pad is None else d_pad
    cache_len = max_conv_len if cache_len is None else cache_len
    # Upstream shape rule: it is what allows the kernel to carry no tap-index guard
    # (DESIGN.md 1.6.2 #6).  Break it and the tap read goes negative -- silently, see
    # repro/TRAP-950-step-tap-guard.py.
    assert max_conv_len >= dilation * (conv_kernel_size - 1), (
        f"[{tag}] max_conv_len={max_conv_len} < dilation*(w-1)="
        f"{dilation * (conv_kernel_size - 1)}"
    )
    dtype_str = _dtype_name(dtype)
    atol, rtol = TOL[dtype_str]
    lane = default_config(d_pad)["threads"] if threads is None else threads

    launch, ref_args = _prepare(
        batch, d, max_conv_len, conv_kernel_size, dilation, dtype, cache_len, d_pad,
        d_mem, seed, weight_scale, state_scale, conv_scale, h_scale,
    )
    y_ref, state_ref = golden_engram_decode(*ref_args, max_conv_len, dilation, eps)

    # The golden is a direct torch implementation; the repository reference must agree
    # with it bit for bit, otherwise the two semantics have drifted apart.
    y_alt, state_alt = engram_decode_step_torch(*ref_args, max_conv_len, dilation, eps)
    assert _bits_equal(y_alt, y_ref) and _bits_equal(state_alt, state_ref), (
        f"[{tag}] golden_engram_decode disagrees with engram_decode_step_torch"
    )

    sentinel = _sentinel(batch, max_conv_len, d_pad, dtype)
    y_buf = torch.full((batch, d_pad), float("nan"), dtype=dtype, device="npu")
    state_buf = sentinel.clone().to("npu")

    kernel = _build_step(
        batch, d, d_pad, max_conv_len, conv_kernel_size, dilation, float(eps),
        dtype_str, lane,
    )
    snaps = []
    t0 = time.perf_counter()
    for _ in range(launches):
        kernel(launch["h"], launch["kv"], launch["state"], launch["wh"], launch["wv"],
               launch["cw"], y_buf, state_buf)
        torch.npu.synchronize()
        snaps.append((y_buf.cpu().clone(), state_buf.cpu().clone()))
    elapsed = time.perf_counter() - t0

    # 5) determinism: the same compiled kernel on the same input must not drift.
    for k in range(1, launches):
        assert _bits_equal(snaps[k][0], snaps[0][0]), f"[{tag}] y_t differs on launch {k}"
        assert _bits_equal(snaps[k][1], snaps[0][1]), (
            f"[{tag}] new_conv_state differs on launch {k}"
        )

    got_y, got_state = snaps[-1]
    last = max_conv_len - 1

    # 1) y_t.
    dy = (got_y[:, :d].float() - y_ref.float()).abs().max().item()
    torch.testing.assert_close(got_y[:, :d].float(), y_ref.float(), atol=atol, rtol=rtol)

    # 2) the new history slot.
    ds = (
        got_state[:, last, :d].float() - state_ref[:, last, :d].float()
    ).abs().max().item()
    torch.testing.assert_close(
        got_state[:, last, :d].float(), state_ref[:, last, :d].float(), atol=atol, rtol=rtol
    )

    # 3) no other history row moved (sentinel intact, bit for bit).
    if max_conv_len > 1:
        assert _bits_equal(got_state[:, :last, :], sentinel[:, :last, :]), (
            f"[{tag}] new_conv_state rows [0, {last}) were written by this launch"
        )

    # 4) the full padded row was produced, and the zero-filled columns are exactly 0.
    assert not torch.isnan(got_y).any(), f"[{tag}] y_t row is not fully written"
    if d_pad > d:
        assert bool((got_y[:, d:] == 0).all()), f"[{tag}] padded y_t columns are not zero"

    _DIFFS.append((tag, dy, ds))
    print(
        f"[{tag}] PASS b={batch} d={d} d_pad={d_pad} mcl={max_conv_len} "
        f"L={cache_len} w={conv_kernel_size} dl={dilation} {dtype_str} "
        f"threads={lane} grid={batch} y_diff={dy:.3e} state_diff={ds:.3e} "
        f"det={launches}x time={elapsed:.2f}s"
    )
    return dy, ds


# ---------------------------------------------------------------------------
# Hierarchical testing
# ---------------------------------------------------------------------------
def run_L0():
    """Gate: the manifest shapes plus every structural edge of the row kernel."""
    # serving-b1: smallest full cache, fp16, grid = 1
    _run_case(1, 256, 12, 4, 3, torch.float16, "L0-serving-b1")
    # decode-s1: max_conv_len == dilation*(w-1), so tap 0 hits history row 0
    _run_case(1, 256, 9, 4, 3, torch.float16, "L0-decode-s1-tap0")
    # serving-b4: d = 512, threads = 512, the widest lane grid
    _run_case(4, 512, 20, 4, 5, torch.float16, "L0-serving-b4")
    # serving-b8: bf16 output path, 8 logical cores
    _run_case(8, 256, 18, 4, 3, torch.bfloat16, "L0-serving-b8")
    # partial cache, host left padding (L < max_conv_len)
    _run_case(2, 256, 12, 4, 3, torch.float16, "L0-partial-cache-L5", cache_len=5)
    # first step: empty cache
    _run_case(1, 256, 9, 4, 3, torch.float16, "L0-first-step-L0", cache_len=0)
    # cache boundaries: L = 1 and L = max_conv_len - 1
    _run_case(1, 256, 12, 4, 3, torch.float16, "L0-L1", cache_len=1)
    _run_case(2, 256, 12, 4, 3, torch.float16, "L0-Lmcl-1", cache_len=11)
    # conv_kernel_size == 1: zero-row tap allocation, empty tap loop
    _run_case(1, 256, 1, 1, 1, torch.float16, "L0-one-tap-mcl1")
    _run_case(2, 256, 4, 1, 1, torch.float16, "L0-one-tap-history")
    # d = 1 and d = 1024 (both padded widths are powers of two)
    _run_case(2, 1, 5, 4, 1, torch.float16, "L0-d1")
    _run_case(2, 1024, 10, 4, 3, torch.float16, "L0-d1024")
    # non power of two d: d_pad = 768, tile = 1024 -> masked trailer path
    _run_case(2, 600, 12, 4, 3, torch.float16, "L0-d600-masked-tail")


def run_L1():
    """Representative shapes beyond the gate."""
    # fixture batch-17: grid = 17, still one wave on 56 vector cores
    _run_case(17, 256, 12, 4, 3, torch.float16, "L1-batch17")
    # the same shape at the other two lane counts (DESIGN.md 1.6.3 (ii) A/B)
    _run_case(4, 512, 20, 4, 5, torch.float16, "L1-b4-threads256", threads=256)
    _run_case(4, 512, 20, 4, 5, torch.float16, "L1-b4-threads128", threads=128)
    # bf16 at the widest row
    _run_case(2, 512, 16, 4, 5, torch.bfloat16, "L1-bf16-d512")
    # full-magnitude inputs: exercises the SiLU curvature (the fixture scales keep
    # conv_out in its near-linear region)
    _run_case(4, 256, 12, 4, 3, torch.float16, "L1-large-magnitude",
              weight_scale=0.5, state_scale=1.0, conv_scale=1.0, h_scale=2.0)
    # wider memory dimension, so host_kv really reduces over d_mem
    _run_case(2, 512, 20, 4, 5, torch.float16, "L1-dmem1024", d_mem=1024)
    # rolling cache over 12 steps, cache empty -> full -> rotating
    _run_multistep(2, 256, 12, 4, 3, torch.float16, "L1-multistep", steps=12)


def run_L2():
    """Larger scale than the gate; recorded, non-blocking."""
    try:
        _run_case(64, 512, 20, 4, 5, torch.float16, "L2-b64-d512")
        _run_case(17, 512, 20, 4, 5, torch.bfloat16, "L2-b17-d512-bf16")
    except Exception as exc:  # noqa: BLE001 - L2 records without blocking
        print(f"[L2] WARN (recorded, non-blocking): {type(exc).__name__}: {exc}")


def run_boundary():
    """Boundary: masked trailer, both lane counts, and non-finite payloads."""
    try:
        # non power of two d at bf16 and at the lower lane count
        _run_case(2, 600, 12, 4, 3, torch.bfloat16, "Boundary-d600-bf16")
        _run_case(2, 600, 12, 4, 3, torch.float16, "Boundary-d600-threads256",
                  threads=256)
        # d just above a power of two -> tile doubles, 3 of 4 lanes are trailer
        _run_case(1, 257, 12, 4, 3, torch.float16, "Boundary-d257")
        # 4 elements per lane
        _run_case(1, 256, 12, 4, 3, torch.float16, "Boundary-threads64", threads=64)
    except Exception as exc:  # noqa: BLE001 - boundary records without blocking
        print(f"[Boundary] WARN (recorded, non-blocking): {type(exc).__name__}: {exc}")

    _run_nonfinite()


def _run_nonfinite():
    """Non-finite and signed-zero payloads must propagate as the reference does.

    Compared by class (NaN / +inf / -inf / signed zero) rather than by bit pattern:
    NaN payloads are not part of the operator contract (DESIGN.md 8.2 last row).
    """
    batch, d, d_pad, mcl, w, dilation = 2, 256, 256, 10, 4, 3
    dtype = torch.float16
    launch, ref_args = _prepare(batch, d, mcl, w, dilation, dtype, mcl, d_pad, None, 3,
                                0.1, 0.1, 0.02, 1.0)
    e_t, h_t, conv_state, w_k, w_v, rms_w_h, rms_w_v, conv_w = ref_args
    conv_state = conv_state.clone()
    conv_state[0, 0, 0] = float("nan")
    conv_state[0, 1, 1] = float("inf")
    conv_state[1, 2, 2] = float("-inf")
    conv_state[1, 3, 3] = -0.0
    conv_state[0, 4, 5] = float("nan")
    # re-derive the launch tensors from the mutated history
    launch["state"] = _pad_last(conv_state, d_pad).to("npu")
    y_ref, state_ref = golden_engram_decode(
        e_t, h_t, conv_state, w_k, w_v, rms_w_h, rms_w_v, conv_w, mcl, dilation, 1e-6
    )

    sentinel = _sentinel(batch, mcl, d_pad, dtype)
    y_buf = torch.full((batch, d_pad), float("nan"), dtype=dtype, device="npu")
    state_buf = sentinel.clone().to("npu")
    kernel = _build_step(batch, d, d_pad, mcl, w, dilation, 1e-6, "float16", 256)
    kernel(launch["h"], launch["kv"], launch["state"], launch["wh"], launch["wv"],
           launch["cw"], y_buf, state_buf)
    torch.npu.synchronize()
    got_y = y_buf.cpu()
    got_state = state_buf.cpu()

    def _class(t):
        """0 finite non-zero, 1 NaN, 2 +inf, 3 -inf, 4 +0, 5 -0."""
        one = torch.tensor(1, dtype=torch.int8)
        out = torch.zeros(t.shape, dtype=torch.int8)
        out = torch.where(torch.isnan(t), one, out)
        out = torch.where(torch.isinf(t) & (t > 0), one + 1, out)
        out = torch.where(torch.isinf(t) & (t < 0), one + 2, out)
        zero = (t == 0) & ~torch.isnan(t) & ~torch.isinf(t)
        return torch.where(zero, one + 3 + torch.signbit(t).to(torch.int8), out)

    dy = (got_y[:, :d].float() - y_ref.float()).abs()
    finite = torch.isfinite(y_ref)
    bad = int((_class(got_y[:, :d]) != _class(y_ref))[~finite].sum().item())
    class_total = int((~finite).sum().item())
    last = mcl - 1
    bad_state = int((_class(got_state[:, last, :d]) != _class(state_ref[:, last, :d])).sum().item())
    _DIFFS.append(("Boundary-nonfinite", dy[finite].max().item(), bad_state))
    print(
        f"[Boundary-nonfinite] classes y_ref={class_total} mismatched={bad} "
        f"state_mismatched={bad_state} finite_y_diff={dy[finite].max().item():.3e}"
    )
    assert bad_state == 0, "non-finite classes differ in the new history slot"


def _run_multistep(batch, d, max_conv_len, conv_kernel_size, dilation, dtype, tag,
                   steps=12):
    """Roll the cache for ``steps`` decode steps; check y and the new slot each step.

    The host owns the shift (launch 1 does it in the operator), so the cache handed
    to this launch is always ``max_conv_len`` rows: it starts empty, fills, then
    rotates.  ``y_t`` and the appended row are compared every step.
    """
    d_pad = _align_up(d)
    dtype_str = _dtype_name(dtype)
    atol, rtol = TOL[dtype_str]
    dev = "npu"
    torch.manual_seed(7)
    w_k = torch.randn(d, d, dtype=dtype) * 0.02
    w_v = torch.randn(d, d, dtype=dtype) * 0.02
    rms_w_h = (1.0 + 0.1 * torch.randn(d)).to(dtype)
    rms_w_v = (1.0 + 0.1 * torch.randn(d)).to(dtype)
    conv_w = torch.randn(conv_kernel_size, d, dtype=dtype) * 0.02
    kernel = _build_step(batch, d, d_pad, max_conv_len, conv_kernel_size, dilation,
                         1e-6, dtype_str, default_config(d_pad)["threads"])
    wh = _pad_last(rms_w_h, d_pad).to(dev)
    wv = _pad_last(rms_w_v, d_pad).to(dev)
    cw = _pad_last(conv_w, d_pad).to(dev)

    history = torch.zeros(batch, 0, d, dtype=dtype)
    worst_y, worst_s = 0.0, 0.0
    for step in range(steps):
        e_t = torch.randn(batch, d, dtype=dtype) * 0.1
        h_t = torch.randn(batch, d, dtype=dtype)
        cache_len = min(step, max_conv_len)
        cache = history[:, history.shape[1] - cache_len:, :] if cache_len else history
        y_ref, state_ref = golden_engram_decode(
            e_t, h_t, cache, w_k, w_v, rms_w_h, rms_w_v, conv_w, max_conv_len, dilation
        )
        if step == 0:
            y_alt, state_alt = engram_decode_step_torch(
                e_t, h_t, cache, w_k, w_v, rms_w_h, rms_w_v, conv_w, max_conv_len,
                dilation,
            )
            assert _bits_equal(y_alt, y_ref) and _bits_equal(state_alt, state_ref), (
                f"[{tag}] golden_engram_decode disagrees with engram_decode_step_torch"
            )
        full = (
            F.pad(cache, (0, 0, max_conv_len - cache_len, 0))
            if cache_len < max_conv_len
            else cache
        )
        y_buf = torch.full((batch, d_pad), float("nan"), dtype=dtype, device=dev)
        state_buf = torch.zeros(batch, max_conv_len, d_pad, dtype=dtype, device=dev)
        kernel(
            _pad_last(h_t, d_pad).to(dev),
            host_kv(e_t, w_k, w_v, d_pad).to(dev),
            _pad_last(full, d_pad).to(dev),
            wh, wv, cw, y_buf, state_buf,
        )
        torch.npu.synchronize()
        got_y = y_buf.cpu()
        got_state = state_buf.cpu()
        torch.testing.assert_close(got_y[:, :d].float(), y_ref.float(), atol=atol, rtol=rtol)
        last = max_conv_len - 1
        torch.testing.assert_close(
            got_state[:, last, :d].float(), state_ref[:, last, :d].float(),
            atol=atol, rtol=rtol,
        )
        worst_y = max(worst_y, (got_y[:, :d].float() - y_ref.float()).abs().max().item())
        worst_s = max(
            worst_s,
            (got_state[:, last, :d].float() - state_ref[:, last, :d].float()).abs().max().item(),
        )
        history = torch.cat([history, state_ref[:, last:last + 1, :]], dim=1)

    _DIFFS.append((tag, worst_y, worst_s))
    print(
        f"[{tag}] PASS b={batch} d={d} mcl={max_conv_len} steps={steps} {dtype_str} "
        f"y_diff={worst_y:.3e} state_diff={worst_s:.3e}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--level", default="L0", choices=["L0", "L1", "L2", "boundary", "all"]
    )
    args, _ = parser.parse_known_args()

    t0 = time.perf_counter()
    if args.level == "L0":
        run_L0()
    elif args.level == "L1":
        run_L1()
    elif args.level == "L2":
        run_L2()
    elif args.level == "boundary":
        run_boundary()
    else:
        run_L0()
        run_L1()
        run_L2()
        run_boundary()

    worst_y = max((dy for _, dy, _ in _DIFFS), default=0.0)
    worst_s = max((ds for _, _, ds in _DIFFS), default=0.0)
    print(
        f"cases={len(_DIFFS)} max_y_diff={worst_y:.3e} max_state_diff={worst_s:.3e} "
        f"total={time.perf_counter() - t0:.1f}s"
    )
    print("\033[92mAll check passed!\033[0m")
    print("[PRECISION_PASS]")


if __name__ == "__main__":
    main()
