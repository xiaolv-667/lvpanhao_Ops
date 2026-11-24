"""Engram fused single-token decode operator — NPU adaptation.

One decode step of the Engram stack, fused end to end::

    k = e_t @ W_K,  v = e_t @ W_V                       # projection
    alpha = sigmoid(dot(RMSNorm(h_t, w_h), RMSNorm(k, w_h)) / sqrt(d))
    v_hat = alpha * v
    vhat_n = RMSNorm(v_hat, w_v)
    conv_out = dilated-causal-DWConv1D(vhat_n, conv_state, conv_w)
    y_t = SiLU(conv_out) + v_hat

and returns ``y_t`` together with the updated ``new_conv_state`` (the shifted
history plus ``vhat_n``, always ``max_conv_len`` long).

Adaptation from GPU (TileOPs) to NPU:

- ``forward``: the ``t.contiguous()`` pass over the eight inputs and the
  shape-derived kernel dispatch are preserved unchanged (O6).  The GPU
  ``kernel_for("engram_decode", lambda: EngramDecodeCall(...))`` call-spec
  machinery is replaced by the NPU ``_get_kernel`` cache over explicit dims (the
  NPU project has no ``_kernels/call_spec`` layer).  The kernel is cached on this
  op's own geometry, so any ``conv_state`` length up to ``max_conv_len`` runs
  against the same compiled kernel.
- Device and dtype checks on the inputs use ``backend.is_device_tensor`` (O1).
- ``target`` parameter removed (O3): the NPU project has no target selection, and
  the kernel constructor takes
  ``(batch, d_mem, d, max_conv_len, conv_kernel_size, dilation, eps, dtype,
  config=None)``.
- ``eval_roofline``: the GPU formula is re-implemented inline as plain arithmetic
  (the NPU project has no ``tileops.perf`` module; roofline arithmetic is
  device-agnostic).
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from tileops.device import get_device_backend
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.sequence_modeling import EngramDecodeKernel
from tileops.ops.op_base import Op

__all__ = ["EngramDecodeFwdOp"]


class EngramDecodeFwdOp(Op):
    """Engram fused decode operator — single-token inference.

    Fuses GEMV projection + gating + dilated conv + SiLU.  It compiles for
    ``max_conv_len`` and accepts any ``conv_state`` length up to it without
    recompiling; the batch size, memory and hidden sizes and dtype come with each
    call.

    Args:
        max_conv_len: Max conv cache capacity (compile-time); also the fixed length
            of ``new_conv_state``.
        conv_kernel_size: Number of conv taps w (model param, e.g. 4).
        dilation: Dilation factor δ (model param, e.g. max N-gram order).
        eps: RMSNorm epsilon (default 1e-6).
        kernel_map: Optional override for kernel dispatch.
    """

    def __init__(
        self,
        max_conv_len: int,
        conv_kernel_size: int,
        dilation: int,
        eps: float = 1e-6,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ):
        if max_conv_len <= 0:
            raise ValueError(f"max_conv_len must be positive, got {max_conv_len}")
        if conv_kernel_size <= 0:
            raise ValueError(f"conv_kernel_size must be positive, got {conv_kernel_size}")
        if dilation <= 0:
            raise ValueError(f"dilation must be positive, got {dilation}")
        min_cache = dilation * (conv_kernel_size - 1)
        if max_conv_len < min_cache:
            raise ValueError(
                f"max_conv_len ({max_conv_len}) must be >= "
                f"dilation * (conv_kernel_size - 1) = {min_cache}"
            )
        self.max_conv_len = int(max_conv_len)
        self.conv_kernel_size = int(conv_kernel_size)
        self.dilation = int(dilation)
        self.eps = float(eps)
        # Bound by forward(); the roofline needs the per-call geometry.
        self.batch = None
        self.d_mem = None
        self.d = None
        self.conv_len = None
        self.dtype = None
        self.dispatch_kernel(kernel_map)
        self._kernel_cache: Dict[tuple, Kernel] = {}
        self.kernel = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"engram_decode": EngramDecodeKernel}

    def _get_kernel(
        self,
        batch: int,
        d_mem: int,
        d: int,
        dtype: torch.dtype,
        device_index: int | None,
    ) -> Kernel:
        """Return the cached kernel for this call's geometry.

        ``max_conv_len``, ``conv_kernel_size``, ``dilation`` and ``eps`` are fixed at
        construction; ``conv_state.shape[1]`` is deliberately not part of the key —
        a shorter history is left-padded inside the kernel.
        """
        key = (batch, d_mem, d, dtype, device_index)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = self.kernel_map["engram_decode"](
                batch,
                d_mem,
                d,
                self.max_conv_len,
                self.conv_kernel_size,
                self.dilation,
                self.eps,
                dtype,
            )
        return self._kernel_cache[key]

    def forward(
        self,
        e_t: torch.Tensor,
        h_t: torch.Tensor,
        conv_state: torch.Tensor,
        W_K: torch.Tensor,
        W_V: torch.Tensor,
        rms_w_h: torch.Tensor,
        rms_w_v: torch.Tensor,
        conv_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one decode token through the fused projection, gate and convolution.

        Args:
            e_t: (batch, d_mem) — gathered N-gram embedding for current token.
            h_t: (batch, d) — hidden state for current token.
            conv_state: (batch, L, d) — conv history, ``L <= max_conv_len``.
                Left-padded to ``max_conv_len`` internally if ``L < max_conv_len``.
            W_K: (d_mem, d) — key projection weight.
            W_V: (d_mem, d) — value projection weight.
            rms_w_h: (d,) — RMSNorm weight for h and k.
            rms_w_v: (d,) — RMSNorm weight for v_hat.
            conv_w: (conv_kernel_size, d) — depthwise conv weights.

        Returns:
            Tuple ``(y_t, new_conv_state)``:
                y_t:            (batch, d) — output to add as residual.
                new_conv_state: (batch, max_conv_len, d) — updated state for next step.
        """
        backend = get_device_backend()
        named = {
            "e_t": e_t,
            "h_t": h_t,
            "conv_state": conv_state,
            "W_K": W_K,
            "W_V": W_V,
            "rms_w_h": rms_w_h,
            "rms_w_v": rms_w_v,
            "conv_w": conv_w,
        }
        for name, tensor in named.items():
            if not backend.is_device_tensor(tensor):
                raise ValueError(
                    f"{name} must be a {backend.name} tensor, got device {tensor.device}"
                )

        if e_t.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                f"e_t must be float16 or bfloat16 (manifest declares no other dtype), "
                f"got {e_t.dtype}"
            )
        for name in ("h_t", "conv_state", "W_K", "W_V", "rms_w_h", "rms_w_v", "conv_w"):
            if named[name].dtype != e_t.dtype:
                raise ValueError(
                    f"{name} must share e_t's dtype {e_t.dtype}, got {named[name].dtype}"
                )

        if e_t.ndim != 2:
            raise ValueError("e_t must have shape [batch, d_mem]")
        batch, d_mem = e_t.shape
        if h_t.ndim != 2 or h_t.shape[0] != batch:
            raise ValueError(f"h_t must have shape [{batch}, d], got {tuple(h_t.shape)}")
        d = h_t.shape[1]
        if conv_state.ndim != 3 or conv_state.shape[0] != batch or conv_state.shape[2] != d:
            raise ValueError(
                f"conv_state must have shape [{batch}, L, {d}], got {tuple(conv_state.shape)}"
            )
        conv_len = conv_state.shape[1]
        for name in ("W_K", "W_V"):
            if named[name].shape != (d_mem, d):
                raise ValueError(
                    f"{name} must have shape [{d_mem}, {d}], got {tuple(named[name].shape)}"
                )
        for name in ("rms_w_h", "rms_w_v"):
            if named[name].shape != (d,):
                raise ValueError(f"{name} must have shape ({d},), got {tuple(named[name].shape)}")
        if conv_w.ndim != 2 or conv_w.shape != (self.conv_kernel_size, d):
            raise ValueError(
                f"conv_w must have shape [{self.conv_kernel_size}, {d}], got {tuple(conv_w.shape)}"
            )

        if batch <= 0 or d_mem <= 0 or d <= 0:
            raise ValueError(f"batch, d_mem and d must be positive, got {batch}, {d_mem}, {d}")
        if conv_len < 0 or conv_len > self.max_conv_len:
            raise ValueError(
                f"conv_state length L must be <= max_conv_len ({self.max_conv_len}), got {conv_len}"
            )

        inputs = tuple(
            t.contiguous() for t in (e_t, h_t, conv_state, W_K, W_V, rms_w_h, rms_w_v, conv_w)
        )

        self.batch = batch
        self.d_mem = d_mem
        self.d = d
        self.conv_len = conv_len
        self.dtype = e_t.dtype
        self.kernel = self._get_kernel(batch, d_mem, d, e_t.dtype, e_t.device.index)

        return self.kernel(*inputs)

    def eval_roofline(self) -> tuple[int, int]:
        """Return ``(flops, bytes)`` for this op instance.

        ``flops`` is the GPU manifest formula, copied verbatim::

            4 * batch * d_mem * d + batch * (16 * d + conv_kernel_size * 2 * d) + 20 * batch

        ``bytes`` is derived from the declared signature (the GPU entry declares
        none): read the eight inputs — ``e_t``, ``h_t``, the ``L``-long
        ``conv_state``, the two projections, the two RMSNorm weights and the conv
        taps — and write ``y_t`` plus the ``max_conv_len``-long
        ``new_conv_state``, all in the input dtype.
        """
        if self.batch is None or self.dtype is None:
            raise RuntimeError(
                f"{type(self).__name__}.eval_roofline() requires a prior forward() "
                "call to bind dynamic input shape"
            )
        batch = int(self.batch)
        d_mem = int(self.d_mem)
        d = int(self.d)
        conv_len = int(self.conv_len)
        elem_bytes = self.dtype.itemsize
        taps = self.conv_kernel_size

        flops = 4 * batch * d_mem * d + batch * (16 * d + taps * 2 * d) + 20 * batch
        nbytes = (
            batch * d_mem * elem_bytes  # e_t
            + batch * d * elem_bytes  # h_t
            + batch * conv_len * d * elem_bytes  # conv_state (L steps)
            + 2 * d_mem * d * elem_bytes  # W_K, W_V
            + 2 * d * elem_bytes  # rms_w_h, rms_w_v
            + taps * d * elem_bytes  # conv_w
            + batch * d * elem_bytes  # y_t
            + batch * self.max_conv_len * d * elem_bytes  # new_conv_state
        )
        return int(flops), int(nbytes)
