"""Shared expert MLP op (NPU) — the shared expert of an MoE layer.

``output = down(silu(hidden @ gate.T) * (hidden @ up.T))``, with the gate and
up rows stacked in ``w_gate_up[2S, H]`` and ``w_down[H, S]``.

Adaptation from GPU (TileOPs) to NPU:

- O1: ``x.is_cuda`` device checks become
  ``get_device_backend().is_device_tensor(x)``.
- O2/O3: the GPU op resolves a target and a kernel per call
  (``kernel_types`` + ``self.kernel_for("shared_expert_mlp", lambda:
  SharedExpertMLPCall(...))``); the NPU op takes its kernel map directly and
  caches one kernel per call geometry in ``_kernel_cache`` (the kernel
  constructor takes ``(num_tokens, hidden_size, ffn_size, dtype, config=None,
  device_index=None)``).  There is no ``tune`` parameter (K8/O3).
- O4: ``from tileops.utils import get_sm_version`` is not imported -- the op
  has no CUDA SM branching, and the kernel's ``roof_key``
  (``tensor_core_roof``, the GPU's per-dtype tensor-core peak pick) has no NPU
  counterpart here; ``eval_roofline`` is plain arithmetic.
- O5: no ``compile_boundary.register_instance`` (not used on NPU).
- O6: the op-specific flow is preserved unchanged -- shapes and dtype come from
  each call, the three tensors are passed through in order, and the output is
  ``[T, H]`` in the input dtype.  The GPU manifest ``shape_rules``
  (``H > 0``, ``S > 0``) are checked where the GPU checks them, at build; the
  NPU project has no manifest codegen, so they are reproduced explicitly.
- ``eval_roofline`` is ported from the GPU manifest entry: ``flops =
  6 * T * S * H + 6 * T * S`` (two GEMM contractions plus the gated
  elementwise), and the byte term is the traffic the GPU derives from the
  signature -- every declared input read once, every declared output written
  once.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from tileops.device import get_device_backend
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.moe.shared_expert_mlp import SharedExpertMLPKernel
from tileops.ops.op_base import Op

__all__ = ["SharedExpertMLPFwdOp"]

# The signature declares the token dtype as float16 or bfloat16.
_TOKEN_DTYPES = (torch.float16, torch.bfloat16)


class SharedExpertMLPFwdOp(Op):
    """Dense gated MLP: ``down(silu(hidden @ gate.T) * (hidden @ up.T))``.

    ``w_gate_up`` stacks the gate rows over the up rows.  Both GEMMs accumulate
    in float32 on the GPU; the gated activation is rounded to the input dtype
    before the down projection.

    Shapes and dtype are taken from each call: ``hidden_states`` is ``[T, H]``,
    ``w_gate_up`` is ``[2S, H]`` and ``w_down`` is ``[H, S]``, so ``T``, ``H``
    and ``S`` are read from the tensors rather than from the constructor.

    Args:
        kernel_map: Optional override for kernel dispatch.
    """

    def __init__(self, kernel_map: Optional[Dict[str, Kernel]] = None) -> None:
        # The call geometry of the last forward(), for eval_roofline().
        self._geometry: Optional[Tuple[int, int, int]] = None
        self._kernel_cache: Dict[tuple, Kernel] = {}
        self.dispatch_kernel(kernel_map)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"shared_expert_mlp": SharedExpertMLPKernel}

    def _validate(
        self,
        hidden_states: torch.Tensor,
        w_gate_up: torch.Tensor,
        w_down: torch.Tensor,
    ) -> Tuple[int, int, int]:
        """Check the tensor contract the GPU op reads from the manifest signature.

        Returns ``(T, H, S)``.  The signature declares three contiguous 2-D
        tensors of one token dtype (float16 or bfloat16), with ``w_gate_up``
        twice as tall as the expert is wide and ``w_down`` its transpose-shaped
        counterpart.  The value range and the ``contiguous`` clause are the
        caller's obligation on the GPU too; a non-contiguous tensor is refused
        here rather than copied, because the copy would silently land in the
        measured path.
        """
        backend = get_device_backend()
        named = (
            ("hidden_states", hidden_states),
            ("w_gate_up", w_gate_up),
            ("w_down", w_down),
        )
        for name, tensor in named:
            if not backend.is_device_tensor(tensor):
                raise ValueError(
                    f"{type(self).__name__} expects {backend.name} tensors; "
                    f"{name} is on {tensor.device}"
                )
            if tensor.dim() != 2:
                raise ValueError(f"{name} must be 2-D; got shape {tuple(tensor.shape)}")
            if not tensor.is_contiguous():
                raise ValueError(
                    f"{name} must be contiguous (the manifest signature declares "
                    f"contiguous: true); got strides {tuple(tensor.stride())}"
                )
            if tensor.dtype not in _TOKEN_DTYPES:
                raise ValueError(f"{name} dtype {tensor.dtype} must be float16 or bfloat16")
        if not (hidden_states.dtype == w_gate_up.dtype == w_down.dtype):
            raise ValueError(
                "hidden_states, w_gate_up and w_down must share one dtype; got "
                f"{hidden_states.dtype}, {w_gate_up.dtype}, {w_down.dtype}"
            )

        num_tokens, hidden_size = hidden_states.shape
        ffn_size = w_down.shape[1]
        if w_gate_up.shape != (2 * ffn_size, hidden_size):
            raise ValueError(
                "w_gate_up must be [2 * S, H] = "
                f"({2 * ffn_size}, {hidden_size}) for hidden_states "
                f"{tuple(hidden_states.shape)} and w_down {tuple(w_down.shape)}; "
                f"got {tuple(w_gate_up.shape)}"
            )
        if w_down.shape[0] != hidden_size:
            raise ValueError(
                f"w_down must be [H, S] with H = {hidden_size}; got {tuple(w_down.shape)}"
            )
        if hidden_size <= 0 or ffn_size <= 0:
            raise ValueError(
                f"{type(self).__name__}: shape rules 'H > 0' and 'S > 0' must hold; "
                f"got H={hidden_size}, S={ffn_size}"
            )
        return num_tokens, hidden_size, ffn_size

    def _get_or_create_kernel(
        self,
        num_tokens: int,
        hidden_size: int,
        ffn_size: int,
        dtype: torch.dtype,
    ) -> Kernel:
        """Return the kernel for a call geometry, building and caching it on first use."""
        key = (num_tokens, hidden_size, ffn_size, dtype)
        kernel = self._kernel_cache.get(key)
        if kernel is None:
            kernel_cls = self.kernel_map["shared_expert_mlp"]
            kernel = kernel_cls(
                num_tokens=num_tokens,
                hidden_size=hidden_size,
                ffn_size=ffn_size,
                dtype=dtype,
                device_index=None,
            )
            self._kernel_cache[key] = kernel
        self.kernel = kernel
        return kernel

    def forward(
        self,
        hidden_states: torch.Tensor,
        w_gate_up: torch.Tensor,
        w_down: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the shared expert to every token.

        Args:
            hidden_states: ``[T, H]``, float16 or bfloat16.
            w_gate_up: ``[2S, H]``, the gate rows then the up rows.
            w_down: ``[H, S]``.

        Returns:
            ``[T, H]`` in the dtype of ``hidden_states``.
        """
        num_tokens, hidden_size, ffn_size = self._validate(hidden_states, w_gate_up, w_down)
        kernel = self._get_or_create_kernel(num_tokens, hidden_size, ffn_size, hidden_states.dtype)
        self._geometry = (num_tokens, hidden_size, ffn_size)
        self.dtype = hidden_states.dtype
        return kernel(hidden_states, w_gate_up, w_down)

    def eval_roofline(self) -> Tuple[int, int]:
        """Return ``(flops, bytes)`` for the last call's geometry.

        Ported from the GPU manifest entry:

        - ``flops = 6 * T * S * H + 6 * T * S``: the gate/up GEMM
          (``2 * T * 2S * H``), the down GEMM (``2 * T * H * S``) and the
          gated elementwise over ``T * S`` elements.
        - bytes: the traffic the GPU derives from the signature -- every
          declared input read once, every declared output written once, all in
          the token dtype: ``2 * T * H`` for ``hidden_states`` and the output,
          ``2 * S * H`` for ``w_gate_up`` and ``S * H`` for ``w_down``.
        """
        if self._geometry is None:
            raise RuntimeError(
                f"{type(self).__name__}.eval_roofline() requires a prior forward() "
                "call to bind the call geometry (T, H, S)"
            )
        num_tokens, hidden_size, ffn_size = self._geometry
        flops = 6 * num_tokens * ffn_size * hidden_size + 6 * num_tokens * ffn_size
        moved = 2 * num_tokens * hidden_size + 3 * ffn_size * hidden_size
        return flops, moved * self.dtype.itemsize
