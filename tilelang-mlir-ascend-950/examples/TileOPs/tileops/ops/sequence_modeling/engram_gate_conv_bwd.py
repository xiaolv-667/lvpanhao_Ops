"""Engram GateConv backward operator — NPU adaptation.

Given ``dY`` and the forward's saved intermediates (``vhat``, ``alpha``,
``rrms_h``, ``rrms_k``, ``rrms_v``), this op computes the gradients of the fused
gate and convolution::

    RMSNorm gating -> causal DWConv1D -> SiLU + residual   (backward)

The caller is responsible for the projection backward (GEMM)::

    dE   = dk @ W_K^T + dv @ W_V^T
    dW_K = E^T @ dk
    dW_V = E^T @ dv

Adaptation from GPU (TileOPs) to NPU:

- ``forward``: the ``t.contiguous()`` pass over the twelve inputs and the
  shape-derived kernel dispatch are preserved unchanged (O6).  The GPU
  ``kernel_for(key, lambda: EngramGateConvCall(...))`` call-spec machinery is
  replaced by the NPU ``_get_kernel`` cache over explicit dims (the NPU project
  has no ``_kernels/call_spec`` layer).  Device and dtype checks on the inputs
  use ``backend.is_device_tensor`` (O1).
- ``tune`` parameter removed (O3); the kernel constructor takes
  ``(M, seq_len, d, eps, dtype, config=None)``.
- ``eval_roofline``: the GPU formula is re-implemented inline as plain
  arithmetic (the NPU project has no ``tileops.perf`` module; roofline
  arithmetic is device-agnostic).
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from tileops.device import get_device_backend
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.sequence_modeling import EngramGateConvBwdKernel
from tileops.ops.op_base import Op

__all__ = ["EngramGateConvBwdOp"]

# Taps of the depthwise gate convolution, as the prim_func declares it.
CONV_KERNEL_SIZE = 4

# Refusal declared for this migration: the fused kernel pads ``d`` to a
# 256-element boundary and keeps the whole padded row in fp32 fragments.
MAX_D = 4096

# The per-position tensors the forward saved are fp32, whatever the input dtype.
_SAVED_FP32 = ("alpha", "rrms_h", "rrms_k", "rrms_v")


class EngramGateConvBwdOp(Op):
    """Engram GateConv backward operator.

    The batch size, sequence length, hidden size and dtype come with each call;
    only the RMSNorm epsilon is fixed at construction.

    Args:
        eps: RMSNorm epsilon (default 1e-6).
        kernel_map: Optional override for kernel dispatch.
    """

    def __init__(
        self,
        eps: float = 1e-6,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ):
        self.eps = float(eps)
        self.M = None
        self.seq_len = None
        self.d = None
        self.dtype = None
        self.dispatch_kernel(kernel_map)
        self._kernel_cache: Dict[tuple, Kernel] = {}
        self.kernel = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"engram_gate_conv_bwd": EngramGateConvBwdKernel}

    def _get_kernel(
        self,
        M: int,
        seq_len: int,
        d: int,
        dtype: torch.dtype,
        device_index: int | None,
    ) -> Kernel:
        key = (M, seq_len, d, self.eps, dtype, device_index)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = self.kernel_map["engram_gate_conv_bwd"](
                M,
                seq_len,
                d,
                self.eps,
                dtype,
                device_index=device_index,
            )
        return self._kernel_cache[key]

    def forward(
        self,
        dY: torch.Tensor,
        H: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        rms_w_h: torch.Tensor,
        rms_w_v: torch.Tensor,
        conv_w: torch.Tensor,
        vhat: torch.Tensor,
        alpha: torch.Tensor,
        rrms_h: torch.Tensor,
        rrms_k: torch.Tensor,
        rrms_v: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Propagate ``dY`` back through the fused gate and convolution.

        Args:
            dY: (M, seq_len, d) — gradient of output Y.
            H: (M, seq_len, d) — hidden states (forward input).
            k: (M, seq_len, d) — key projection (forward input).
            v: (M, seq_len, d) — value projection (forward input).
            rms_w_h: (d,) — RMSNorm weight for H and k.
            rms_w_v: (d,) — RMSNorm weight for vhat.
            conv_w: (4, d) — conv weights.
            vhat: (M, seq_len, d) — saved from forward (its rounded value).
            alpha: (M, seq_len) fp32 — saved from forward.
            rrms_h: (M, seq_len) fp32 — saved from forward.
            rrms_k: (M, seq_len) fp32 — saved from forward.
            rrms_v: (M, seq_len) fp32 — saved from forward.

        Returns:
            Tuple ``(dH, dk, dv, drms_w_h, drms_w_v, dconv_w)``:
                dH:       (M, seq_len, d) — gradient w.r.t. H.
                dk:       (M, seq_len, d) — gradient w.r.t. the key projection.
                dv:       (M, seq_len, d) — gradient w.r.t. the value projection.
                drms_w_h: (d,) fp32 — gradient w.r.t. the H/k RMSNorm weight.
                drms_w_v: (d,) fp32 — gradient w.r.t. the vhat RMSNorm weight.
                dconv_w:  (4, d) fp32 — gradient w.r.t. the conv weights.
        """
        backend = get_device_backend()
        named = {
            "dY": dY,
            "H": H,
            "k": k,
            "v": v,
            "rms_w_h": rms_w_h,
            "rms_w_v": rms_w_v,
            "conv_w": conv_w,
            "vhat": vhat,
            "alpha": alpha,
            "rrms_h": rrms_h,
            "rrms_k": rrms_k,
            "rrms_v": rrms_v,
        }
        for name, tensor in named.items():
            if not backend.is_device_tensor(tensor):
                raise ValueError(
                    f"{name} must be a {backend.name} tensor, got device {tensor.device}"
                )

        if dY.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                f"dY must be float16 or bfloat16 (manifest declares no other dtype), "
                f"got {dY.dtype}"
            )
        for name in ("H", "k", "v", "rms_w_h", "rms_w_v", "conv_w", "vhat"):
            if named[name].dtype != dY.dtype:
                raise ValueError(f"{name} must share dY's dtype {dY.dtype}, got {named[name].dtype}")
        for name in _SAVED_FP32:
            if named[name].dtype != torch.float32:
                raise ValueError(
                    f"{name} is the forward's saved scalar, which is float32, "
                    f"got {named[name].dtype}"
                )

        if dY.ndim != 3:
            raise ValueError("dY must have shape [M, seq_len, d]")
        M, seq_len, d = dY.shape
        for name in ("H", "k", "v", "vhat"):
            if named[name].shape != dY.shape:
                raise ValueError(
                    f"{name} must have the same shape as dY, "
                    f"got {tuple(named[name].shape)} vs {tuple(dY.shape)}"
                )
        if rms_w_h.shape != (d,) or rms_w_v.shape != (d,):
            raise ValueError(
                f"rms_w_h / rms_w_v must have shape ({d},), got "
                f"{tuple(rms_w_h.shape)} / {tuple(rms_w_v.shape)}"
            )
        if conv_w.ndim != 2 or conv_w.shape != (CONV_KERNEL_SIZE, d):
            raise ValueError(
                f"conv_w must have shape [{CONV_KERNEL_SIZE}, {d}], got {tuple(conv_w.shape)}"
            )
        for name in _SAVED_FP32:
            if named[name].shape != (M, seq_len):
                raise ValueError(
                    f"{name} must have shape [{M}, {seq_len}], got {tuple(named[name].shape)}"
                )

        if M <= 0 or seq_len <= 0 or d <= 0:
            raise ValueError(f"M, seq_len and d must be positive, got {M}, {seq_len}, {d}")
        if d > MAX_D:
            raise ValueError(f"d must be <= {MAX_D}, got {d}")

        inputs = tuple(
            t.contiguous()
            for t in (dY, H, k, v, rms_w_h, rms_w_v, conv_w, vhat, alpha, rrms_h, rrms_k, rrms_v)
        )

        self.M = M
        self.seq_len = seq_len
        self.d = d
        self.dtype = dY.dtype
        self.kernel = self._get_kernel(M, seq_len, d, dY.dtype, dY.device.index)

        return self.kernel(*inputs)

    def eval_roofline(self) -> tuple[int, int]:
        """Return ``(flops, bytes)`` for this op instance.

        ``flops`` is the GPU manifest formula, copied verbatim::

            M * (seq_len * (45 * d + 10)
                 + 6 * d * (seq_len + max(seq_len - 1, 0) + max(seq_len - 2, 0)
                            + max(seq_len - 3, 0)))

        ``bytes`` is derived from the declared signature: read the five
        dtype-shaped input tensors (dY, H, k, v, vhat), the two RMSNorm weights,
        the four conv taps and the four fp32 saved per-position tensors; write
        dH/dk/dv in the input dtype, the two fp32 RMSNorm-weight gradients and
        the fp32 conv-weight gradient.
        """
        if self.M is None or self.dtype is None:
            raise RuntimeError(
                f"{type(self).__name__}.eval_roofline() requires a prior forward() "
                "call to bind dynamic input shape"
            )
        M = int(self.M)
        seq_len = int(self.seq_len)
        d = int(self.d)
        elem_bytes = self.dtype.itemsize
        taps = CONV_KERNEL_SIZE

        flops = M * (
            seq_len * (45 * d + 10)
            + 6 * d * (seq_len + max(seq_len - 1, 0) + max(seq_len - 2, 0) + max(seq_len - 3, 0))
        )
        nbytes = (
            5 * M * seq_len * d * elem_bytes  # dY, H, k, v, vhat
            + 2 * d * elem_bytes  # rms_w_h, rms_w_v
            + taps * d * elem_bytes  # conv_w
            + 4 * M * seq_len * 4  # alpha, rrms_h, rrms_k, rrms_v (fp32)
            + 3 * M * seq_len * d * elem_bytes  # dH, dk, dv
            + 2 * d * 4  # drms_w_h, drms_w_v (fp32)
            + taps * d * 4  # dconv_w (fp32)
        )
        return int(flops), int(nbytes)
