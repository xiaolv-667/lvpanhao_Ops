"""Workload definitions for the sequence_modeling op family.

Ported from the GPU TileOPs ``workloads/sequence_modeling/engram.py`` --
EngramGateConvFwdOp, EngramGateConvBwdOp and EngramDecodeFwdOp subsets only.

The PyTorch reference of the forward fusion (``_rmsnorm`` and
``engram_gate_conv_fwd_torch``) is ported together with the workload because the
GPU workload delegates its ``ref_program`` to it.  The backward workload reuses
``_rmsnorm`` to regenerate the forward's saved intermediates and delegates its
``ref_program`` to ``ref_engram_gate_conv_bwd`` (autograd through the float32
forward).  The decode workload delegates its ``ref_program`` to
``engram_decode_step_torch`` (via ``_rmsnorm_decode``), the fp32 reference of one
single-token step.

Adaptation point W1: ``device`` is resolved from :func:`get_device_backend`
instead of the GPU ``workloads.device.run_device()`` helper.
"""

import torch
import torch.nn.functional as F

from tileops.device import get_device_backend
from tileops.workloads.workload_base import FixtureBase, WorkloadBase

__all__ = [
    "CONV_KERNEL_SIZE",
    "EngramDecodeFixture",
    "EngramDecodeWorkload",
    "EngramGateConvBwdFixture",
    "EngramGateConvBwdWorkload",
    "EngramGateConvFwdFixture",
    "EngramGateConvFwdWorkload",
    "engram_decode_step_torch",
    "engram_gate_conv_fwd_torch",
    "ref_engram_gate_conv_bwd",
]

# Taps of the depthwise gate convolution, which the forward and the backward
# must read the same way.
CONV_KERNEL_SIZE = 4


class EngramGateConvFwdFixture(FixtureBase):
    @classmethod
    def get_params(cls):
        import pytest

        return [
            (
                "M, seq_len, d, dtype",
                [
                    # GPU row 1 -- the manifest's train-s32 smoke workload.
                    pytest.param(
                        1,
                        32,
                        256,
                        torch.float16,
                        marks=[pytest.mark.smoke, pytest.mark.packaging],
                    ),
                    # GPU row 2 (bf16 counterpart of the smoke shape).
                    pytest.param(1, 32, 256, torch.bfloat16, marks=pytest.mark.smoke),
                    # GPU rows 3-4.
                    pytest.param(2, 64, 512, torch.float16, marks=pytest.mark.full),
                    pytest.param(2, 16, 256, torch.bfloat16, marks=pytest.mark.full),
                    # Manifest workloads train-s128 / train-s96.
                    pytest.param(1, 128, 256, torch.bfloat16, marks=pytest.mark.full),
                    pytest.param(3, 96, 768, torch.bfloat16, marks=pytest.mark.full),
                ],
            ),
        ]


class EngramGateConvFwdWorkload(WorkloadBase):
    """Inputs of the post-projection Engram GateConv fusion.

    Ported from the GPU ``EngramGateConvFwdWorkload``: ``H`` is a full-scale
    random draw while ``k`` and ``v`` are scaled by ``0.1``, both RMSNorm weights
    are ones, and the four conv taps are drawn at ``0.02``.  The GPU-only
    ``tune`` dimension is gone (T3).
    """

    def __init__(self, M, seq_len, d, dtype, eps=1e-6):
        self.M = M
        self.seq_len = seq_len
        self.d = d
        self.dtype = dtype
        self.eps = eps

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        M, seq_len, d = self.M, self.seq_len, self.d
        device = get_device_backend().name
        H = torch.randn(M, seq_len, d, dtype=self.dtype, device=device)
        k = torch.randn(M, seq_len, d, dtype=self.dtype, device=device) * 0.1
        v = torch.randn(M, seq_len, d, dtype=self.dtype, device=device) * 0.1
        rms_w_h = torch.ones(d, dtype=self.dtype, device=device)
        rms_w_v = torch.ones(d, dtype=self.dtype, device=device)
        conv_w = torch.randn(CONV_KERNEL_SIZE, d, dtype=self.dtype, device=device) * 0.02
        return H, k, v, rms_w_h, rms_w_v, conv_w

    def ref_program(self, H, k, v, rms_w_h, rms_w_v, conv_w):
        return engram_gate_conv_fwd_torch(H, k, v, rms_w_h, rms_w_v, conv_w, self.eps)


def _rmsnorm(x, w, eps=1e-6):
    """Returns (normed, rrms)."""
    x_f = x.float()
    rrms = (x_f**2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    normed = x_f * rrms * w.float()
    return normed, rrms.squeeze(-1)


def engram_gate_conv_fwd_torch(H, k, v, rms_w_h, rms_w_v, conv_w, eps=1e-6):
    """PyTorch reference for Engram GateConv forward."""
    M, T, d = H.shape

    h_norm, rrms_h = _rmsnorm(H, rms_w_h, eps)
    k_norm, rrms_k = _rmsnorm(k, rms_w_h, eps)

    dot = (h_norm * k_norm).sum(dim=-1, keepdim=True)
    alpha = torch.sigmoid(dot / (d**0.5))

    v_hat = alpha * v.float()

    v_hat_norm, rrms_v = _rmsnorm(v_hat.to(H.dtype), rms_w_v, eps)

    v_perm = v_hat_norm.float().permute(0, 2, 1)
    v_padded = F.pad(v_perm, (conv_w.shape[0] - 1, 0))
    conv_w_expanded = conv_w.float().T.unsqueeze(1)
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


class EngramGateConvBwdFixture(FixtureBase):
    @classmethod
    def get_params(cls):
        import pytest

        return [
            (
                "M, seq_len, d, dtype",
                [
                    # GPU row 1 -- the manifest's train-s32 smoke workload.
                    pytest.param(
                        1,
                        32,
                        256,
                        torch.float16,
                        marks=[pytest.mark.smoke, pytest.mark.packaging],
                    ),
                    # GPU row 2 (bf16 counterpart of the smoke shape).
                    pytest.param(1, 32, 256, torch.bfloat16, marks=pytest.mark.smoke),
                    # GPU rows 3-4.
                    pytest.param(2, 64, 512, torch.float16, marks=pytest.mark.full),
                    pytest.param(2, 16, 256, torch.bfloat16, marks=pytest.mark.full),
                    pytest.param(2, 512, 256, torch.float16, marks=pytest.mark.full),
                    # 1203 tokens: the capped run count lengthens dconv_w runs to 10
                    # tokens, the last run is short, and some runs cross a batch
                    # boundary.
                    pytest.param(3, 401, 256, torch.bfloat16, marks=pytest.mark.full),
                    # 8 tokens: one dconv_w run, stored without the atomic add.
                    pytest.param(1, 8, 256, torch.float16, marks=pytest.mark.full),
                    # One-token sequences: every dconv_w source row but one is conv
                    # padding.
                    pytest.param(2, 1, 256, torch.float16, marks=pytest.mark.full),
                ],
            ),
        ]


class EngramGateConvBwdWorkload(WorkloadBase):
    """Inputs of the Engram GateConv backward.

    Ported from the GPU ``EngramGateConvBwdWorkload``: the forward inputs are
    drawn exactly as the forward workload draws them (``H`` full scale, ``k`` and
    ``v`` at ``0.1``, both RMSNorm weights ones, the four conv taps at ``0.02``),
    ``dY`` is a fresh ``0.1``-scaled draw, and the five saved intermediates are
    regenerated from the *reference* forward so they carry its rounding — in
    particular ``vhat`` is the forward's rounded tensor, not a recomputed one.
    The GPU-only ``tune`` dimension is gone (T3).
    """

    def __init__(self, M, seq_len, d, dtype, eps=1e-6):
        self.M = M
        self.seq_len = seq_len
        self.d = d
        self.dtype = dtype
        self.eps = eps

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        """Generate inputs including saved intermediates from a reference forward."""
        M, seq_len, d = self.M, self.seq_len, self.d
        device = get_device_backend().name
        H = torch.randn(M, seq_len, d, dtype=self.dtype, device=device)
        k = torch.randn(M, seq_len, d, dtype=self.dtype, device=device) * 0.1
        v = torch.randn(M, seq_len, d, dtype=self.dtype, device=device) * 0.1
        rms_w_h = torch.ones(d, dtype=self.dtype, device=device)
        rms_w_v = torch.ones(d, dtype=self.dtype, device=device)
        conv_w = torch.randn(CONV_KERNEL_SIZE, d, dtype=self.dtype, device=device) * 0.02
        dY = torch.randn(M, seq_len, d, dtype=self.dtype, device=device) * 0.1

        # Compute saved intermediates with the same normalization as the reference.
        h_norm, rrms_h = _rmsnorm(H, rms_w_h, self.eps)
        k_norm, rrms_k = _rmsnorm(k, rms_w_h, self.eps)
        dot = (h_norm * k_norm).sum(dim=-1, keepdim=True)
        alpha = torch.sigmoid(dot / (d**0.5))
        v_hat = alpha * v.float()
        _, rrms_v = _rmsnorm(v_hat.to(self.dtype), rms_w_v, self.eps)

        vhat = v_hat.to(self.dtype)
        alpha_squeezed = alpha.squeeze(-1).float()
        rrms_h = rrms_h.float()
        rrms_k = rrms_k.float()
        rrms_v = rrms_v.float()

        return (
            dY,
            H,
            k,
            v,
            rms_w_h,
            rms_w_v,
            conv_w,
            vhat,
            alpha_squeezed,
            rrms_h,
            rrms_k,
            rrms_v,
        )

    def ref_program(
        self, dY, H, k, v, rms_w_h, rms_w_v, conv_w, vhat, alpha, rrms_h, rrms_k, rrms_v
    ):
        return ref_engram_gate_conv_bwd(
            dY,
            H,
            k,
            v,
            rms_w_h,
            rms_w_v,
            conv_w,
            vhat,
            alpha,
            rrms_h,
            rrms_k,
            rrms_v,
            self.eps,
        )


def ref_engram_gate_conv_bwd(
    dY, H, k, v, rms_w_h, rms_w_v, conv_w, vhat, alpha, rrms_h, rrms_k, rrms_v, eps=1e-6
):
    """PyTorch reference backward via autograd.

    ``vhat``, ``alpha`` and the three ``rrms`` tensors are the forward's saved
    values and are *not* used by this oracle: it differentiates the float32
    forward over the six differentiable inputs.  They stay in the signature so the
    reference takes the same argument list as the op.
    """
    M, T, d = H.shape

    H_ag = H.float().detach().requires_grad_(True)
    k_ag = k.float().detach().requires_grad_(True)
    v_ag = v.float().detach().requires_grad_(True)
    w_h_ag = rms_w_h.float().detach().requires_grad_(True)
    w_v_ag = rms_w_v.float().detach().requires_grad_(True)
    cw_ag = conv_w.float().detach().requires_grad_(True)

    h_norm, _ = _rmsnorm(H_ag, w_h_ag, eps)
    k_norm, _ = _rmsnorm(k_ag, w_h_ag, eps)

    dot = (h_norm * k_norm).sum(dim=-1, keepdim=True)
    alpha_ag = torch.sigmoid(dot / (d**0.5))
    v_hat_ag = alpha_ag * v_ag
    v_hat_norm, _ = _rmsnorm(v_hat_ag, w_v_ag, eps)

    v_perm = v_hat_norm.permute(0, 2, 1)
    v_padded = F.pad(v_perm, (conv_w.shape[0] - 1, 0))
    cw_expanded = cw_ag.T.unsqueeze(1)
    conv_out = F.conv1d(v_padded, cw_expanded, groups=d).permute(0, 2, 1)
    Y_ag = F.silu(conv_out) + v_hat_ag

    # On this thread, not autograd's engine thread: a benchmark timing this
    # reference cannot attribute kernels launched where its iteration id was
    # never pushed.  Same gradients either way.
    with torch.autograd.set_multithreading_enabled(False):
        Y_ag.backward(dY.float())

    return (
        H_ag.grad.to(H.dtype),
        k_ag.grad.to(H.dtype),
        v_ag.grad.to(v.dtype),
        w_h_ag.grad,
        w_v_ag.grad,
        cw_ag.grad,
    )


# ---------------------------------------------------------------------------
# Decode (EngramDecodeFwdOp): one single-token step
# ---------------------------------------------------------------------------


class EngramDecodeFixture(FixtureBase):
    @classmethod
    def get_params(cls):
        import pytest

        return [
            (
                "batch, d_mem, d, max_conv_len, conv_kernel_size, dilation, dtype",
                [
                    # GPU row 1 -- the manifest's decode-s1 smoke workload.
                    pytest.param(
                        1,
                        512,
                        256,
                        12,
                        4,
                        3,
                        torch.float16,
                        marks=[pytest.mark.smoke, pytest.mark.packaging],
                    ),
                    # GPU row 2 (bf16 counterpart of the smoke shape).
                    pytest.param(1, 512, 256, 12, 4, 3, torch.bfloat16, marks=pytest.mark.smoke),
                    # GPU row 3: d_mem 519 is not a multiple of the MMA K tile, so the
                    # projection kernel's predicated tail is exercised.
                    pytest.param(17, 519, 256, 12, 4, 3, torch.float16, marks=pytest.mark.full),
                    # GPU row 4: the minimum legal cache -- one tap, no dilation.
                    pytest.param(1, 512, 256, 1, 1, 1, torch.float16, marks=pytest.mark.full),
                    # GPU row 5 (the GPU fixture autotuned this row; ``tune`` is gone, T3).
                    pytest.param(4, 1024, 512, 20, 4, 5, torch.float16, marks=pytest.mark.full),
                    # GPU row 6 -- the manifest's serving-b8 workload.
                    pytest.param(8, 512, 256, 18, 4, 3, torch.bfloat16, marks=pytest.mark.full),
                ],
            ),
        ]


class EngramDecodeWorkload(WorkloadBase):
    """Inputs of one Engram decode step.

    Ported from the GPU ``EngramDecodeWorkload``: ``e_t``, ``h_t`` and
    ``conv_state`` are ``0.1``-scaled draws (``h_t`` full scale), both projections
    are drawn at ``0.02``, both RMSNorm weights are ones and the conv taps are
    drawn at ``0.02``.  ``conv_len`` is how many history steps ``conv_state``
    holds (a full cache by default); a shorter one exercises the left padding.
    The GPU-only ``tune`` dimension is gone (T3).
    """

    def __init__(
        self,
        batch,
        d_mem,
        d,
        max_conv_len,
        conv_kernel_size,
        dilation,
        dtype,
        eps=1e-6,
        conv_len=None,
    ):
        self.batch = batch
        self.d_mem = d_mem
        self.d = d
        self.max_conv_len = max_conv_len
        self.conv_kernel_size = conv_kernel_size
        self.dilation = dilation
        self.dtype = dtype
        self.eps = eps
        # The history steps conv_state holds; a full cache by default.
        self.conv_len = max_conv_len if conv_len is None else conv_len

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        batch, d_mem, d = self.batch, self.d_mem, self.d
        device = get_device_backend().name
        e_t = torch.randn(batch, d_mem, dtype=self.dtype, device=device) * 0.1
        h_t = torch.randn(batch, d, dtype=self.dtype, device=device)
        conv_state = torch.randn(batch, self.conv_len, d, dtype=self.dtype, device=device) * 0.1
        W_K = torch.randn(d_mem, d, dtype=self.dtype, device=device) * 0.02
        W_V = torch.randn(d_mem, d, dtype=self.dtype, device=device) * 0.02
        rms_w_h = torch.ones(d, dtype=self.dtype, device=device)
        rms_w_v = torch.ones(d, dtype=self.dtype, device=device)
        conv_w = torch.randn(self.conv_kernel_size, d, dtype=self.dtype, device=device) * 0.02
        return e_t, h_t, conv_state, W_K, W_V, rms_w_h, rms_w_v, conv_w

    def ref_program(self, e_t, h_t, conv_state, W_K, W_V, rms_w_h, rms_w_v, conv_w):
        y_ref, state_ref = engram_decode_step_torch(
            e_t,
            h_t,
            conv_state,
            W_K,
            W_V,
            rms_w_h,
            rms_w_v,
            conv_w,
            self.max_conv_len,
            self.dilation,
            self.eps,
        )
        return y_ref, state_ref


def _rmsnorm_decode(x, w, eps=1e-6):
    x_f = x.float()
    rrms = (x_f**2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    return (x_f * rrms * w.float()), rrms


def engram_decode_step_torch(
    e_t,
    h_t,
    conv_state,
    W_K,
    W_V,
    rms_w_h,
    rms_w_v,
    conv_w,
    max_conv_len,
    dilation,
    eps=1e-6,
):
    """PyTorch reference for a single decode step with dilated causal conv.

    ``conv_state`` holds the last ``L`` steps (``L <= max_conv_len``, as the
    manifest shape rule requires) and is left-padded to ``max_conv_len`` before the
    taps are read; ``new_conv_state`` is that *padded* history shifted by one row
    with this step's normalized value appended, so it is ``max_conv_len`` long
    whether or not the cache was full on entry.  Arithmetic is fp32; only the
    returned tensors are rounded to the input dtype.
    """
    B, d = h_t.shape
    w = conv_w.shape[0]
    L = conv_state.shape[1]

    k = e_t.float() @ W_K.float()
    v = e_t.float() @ W_V.float()

    h_norm, _ = _rmsnorm_decode(h_t.unsqueeze(1), rms_w_h)
    k_norm, _ = _rmsnorm_decode(k.unsqueeze(1), rms_w_h)
    h_norm = h_norm.squeeze(1)
    k_norm = k_norm.squeeze(1)

    dot = (h_norm * k_norm).sum(dim=-1, keepdim=True)
    alpha = torch.sigmoid(dot / (d**0.5))
    v_hat = alpha * v

    v_hat_norm, _ = _rmsnorm_decode(v_hat.unsqueeze(1), rms_w_v)
    v_hat_norm = v_hat_norm.squeeze(1)

    # Left-pad the history once, in the input dtype: the taps read its fp32 view and
    # the cache update shifts the same tensor, so both see one padded layout.
    state_pad = max_conv_len - L
    padded_history = F.pad(conv_state, (0, 0, state_pad, 0)) if state_pad > 0 else conv_state
    padded_state = padded_history.float()

    conv_out = torch.zeros(B, d, device=h_t.device)
    for p in range(w - 1):
        state_idx = max_conv_len - (w - 1 - p) * dilation
        if 0 <= state_idx < max_conv_len:
            conv_out += conv_w[p].float().unsqueeze(0) * padded_state[:, state_idx, :]
    conv_out += conv_w[w - 1].float().unsqueeze(0) * v_hat_norm

    # Drop the oldest slot (the leading cache-padding row when L < max_conv_len) and
    # append this step: the result is always max_conv_len rows, with the most recent
    # ``max_conv_len - 1`` history steps kept to the left of the new one.
    new_conv_state = torch.cat(
        [
            padded_history[:, 1:, :],
            v_hat_norm.unsqueeze(1).to(conv_state.dtype),
        ],
        dim=1,
    )

    y_t = F.silu(conv_out) + v_hat
    return y_t.to(h_t.dtype), new_conv_state
