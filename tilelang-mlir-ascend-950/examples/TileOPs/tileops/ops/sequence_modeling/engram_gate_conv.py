"""Engram GateConv forward operator (post-projection fusion) — NPU adaptation.

Assumes ``k = E @ W_K`` and ``v = E @ W_V`` have been computed externally via
standard GEMM.  This op fuses the remaining memory-bound stages:

    RMSNorm gating -> causal DWConv1D -> SiLU + residual

Returns ``Y`` plus the intermediates the backward saves:
``vhat, alpha, rrms_h, rrms_k, rrms_v``.

Adaptation from GPU (TileOPs) to NPU:

- ``forward``: the ``t.contiguous()`` pass over the six inputs and the
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
from tileops.kernels.sequence_modeling import EngramGateConvFwdKernel
from tileops.ops.op_base import Op

__all__ = ["EngramGateConvFwdOp"]

# Taps of the depthwise gate convolution, as the prim_func declares it.
CONV_KERNEL_SIZE = 4

# Refusal declared for this migration: the fused kernel pads ``d`` to a
# 256-element boundary and keeps the whole padded row in fp32 fragments.
MAX_D = 4096


class EngramGateConvFwdOp(Op):
    """Engram GateConv forward operator (post-projection fusion).

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
        return {"engram_gate_conv_fwd": EngramGateConvFwdKernel}

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
            self._kernel_cache[key] = self.kernel_map["engram_gate_conv_fwd"](
                M,
                seq_len,
                d,
                self.eps,
                dtype,
            )
        return self._kernel_cache[key]

    def forward(
        self,
        H: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        rms_w_h: torch.Tensor,
        rms_w_v: torch.Tensor,
        conv_w: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Run the fused gate and convolution.

        Args:
            H: (M, seq_len, d) — hidden states from preceding layers.
            k: (M, seq_len, d) — key projection output (E @ W_K).
            v: (M, seq_len, d) — value projection output (E @ W_V).
            rms_w_h: (d,) — RMSNorm weight for H and k.
            rms_w_v: (d,) — RMSNorm weight for gated value (conv input).
            conv_w: (4, d) — depthwise causal conv1d weights.

        Returns:
            Tuple ``(Y, vhat, alpha, rrms_h, rrms_k, rrms_v)``:
                Y:      (M, seq_len, d) — output to add as residual to H.
                vhat:   (M, seq_len, d) — saved for backward.
                alpha:  (M, seq_len) fp32 — scalar gate, saved for backward.
                rrms_h: (M, seq_len) fp32 — RMSNorm reciprocal rms of H.
                rrms_k: (M, seq_len) fp32 — RMSNorm reciprocal rms of k.
                rrms_v: (M, seq_len) fp32 — RMSNorm reciprocal rms of v_hat.
        """
        backend = get_device_backend()
        named = {
            "H": H,
            "k": k,
            "v": v,
            "rms_w_h": rms_w_h,
            "rms_w_v": rms_w_v,
            "conv_w": conv_w,
        }
        for name, tensor in named.items():
            if not backend.is_device_tensor(tensor):
                raise ValueError(
                    f"{name} must be a {backend.name} tensor, got device {tensor.device}"
                )

        if H.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                f"H must be float16 or bfloat16 (manifest declares no other dtype), got {H.dtype}"
            )
        for name in ("k", "v", "rms_w_h", "rms_w_v", "conv_w"):
            if named[name].dtype != H.dtype:
                raise ValueError(f"{name} must share H's dtype {H.dtype}, got {named[name].dtype}")

        if H.ndim != 3:
            raise ValueError("H must have shape [M, seq_len, d]")
        M, seq_len, d = H.shape
        for name in ("k", "v"):
            if named[name].shape != H.shape:
                raise ValueError(
                    f"{name} must have the same shape as H, "
                    f"got {tuple(named[name].shape)} vs {tuple(H.shape)}"
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

        if M <= 0 or seq_len <= 0 or d <= 0:
            raise ValueError(f"M, seq_len and d must be positive, got {M}, {seq_len}, {d}")
        if d > MAX_D:
            raise ValueError(f"d must be <= {MAX_D}, got {d}")

        inputs = tuple(t.contiguous() for t in (H, k, v, rms_w_h, rms_w_v, conv_w))

        self.M = M
        self.seq_len = seq_len
        self.d = d
        self.dtype = H.dtype
        self.kernel = self._get_kernel(M, seq_len, d, H.dtype, H.device.index)

        return self.kernel(*inputs)

    def eval_roofline(self) -> tuple[int, int]:
        """Return ``(flops, bytes)`` for this op instance.

        ``flops`` is the GPU manifest formula, copied verbatim::

            M * (seq_len * (21 * d + 14)
                 + 2 * d * (seq_len + max(seq_len - 1, 0) + max(seq_len - 2, 0)
                            + max(seq_len - 3, 0)))

        ``bytes`` is derived from the declared signature: read the six inputs
        (H, k, v, the two RMSNorm weights and the four conv taps) and write the
        two dtype-shaped outputs plus the four fp32 per-position tensors.
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
            seq_len * (21 * d + 14)
            + 2 * d * (seq_len + max(seq_len - 1, 0) + max(seq_len - 2, 0) + max(seq_len - 3, 0))
        )
        nbytes = (
            3 * M * seq_len * d * elem_bytes  # H, k, v
            + 2 * d * elem_bytes  # rms_w_h, rms_w_v
            + taps * d * elem_bytes  # conv_w
            + 2 * M * seq_len * d * elem_bytes  # Y, vhat
            + 4 * M * seq_len * 4  # alpha, rrms_h, rrms_k, rrms_v (fp32)
        )
        return int(flops), int(nbytes)
