"""MoE permute-align op (spec-conformant interface).

Routes tokens to experts and pads each expert's route count to ``block_size``,
producing the three index arrays a MoE grouped GEMM consumes.

Adaptation from GPU (TileOPs) to NPU:

- O1: CUDA device checks (``x.is_cuda``) become
  ``get_device_backend().is_device_tensor(x)``.
- O2/O3: the GPU op takes a ``target`` and resolves the kernel per call through
  ``kernel_for("permute_align", lambda: PermuteAlignCall(...))``; the NPU op
  takes its kernel map directly and caches one kernel per routed count in
  ``_kernel_cache`` (the kernel constructor takes
  ``(numel, num_experts, block_size, config=None, device_index=None)``).  There
  is no ``tune`` parameter (K8/O3).
- O5: no ``compile_boundary.register_instance`` (not used on NPU).
- O6: the op-specific flow is preserved unchanged -- the constructor params, the
  pass-through of each call's routed extents, and the three index outputs.  The
  GPU validity checks come from the manifest ``shape_rules``; the NPU project
  has no manifest codegen, so they are reproduced explicitly.
- ``eval_roofline`` is ported from the GPU manifest entry: ``flops = 0`` (no
  arithmetic), and the byte term is the traffic the GPU derives from the
  signature -- the routed ids read once and the three int32 outputs written
  once each.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch

from tileops.device import get_device_backend
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.moe.moe_permute_align import MoEPermuteAlignKernel
from tileops.ops.op_base import Op

__all__ = ["MoEPermuteAlignFwdOp"]

_INDEX_BYTES = 4  # every tensor this op touches is an int32 index array


class MoEPermuteAlignFwdOp(Op):
    """Route tokens to experts and pad each expert's token count to block_size.

    Takes ``topk_ids`` and produces the three index arrays required by MoE
    grouped GEMM: sorted token indices, per-block expert ids, and the total
    padded token count.

    With ``numel = T * K`` routes and ``count[e]`` of them on expert ``e``, the result is:

    - ``num_tokens_post_pad``: the sum over experts of ``count[e]`` rounded up to
      ``block_size``.
    - ``sorted_token_ids[:num_tokens_post_pad]``: one range per expert, in expert order, of
      that rounded-up length. The first ``count[e]`` slots of the range hold the indices of
      the routes on ``e``, each once, in an order that is unspecified and may differ
      between calls on the same input; the rest of the range holds ``numel``.
    - ``expert_ids[:num_tokens_post_pad // block_size]``: the owner of each ``block_size``
      block of those ranges.
    - Past those lengths, every entry of ``sorted_token_ids`` is ``numel`` and every entry
      of ``expert_ids`` lies outside ``[0, num_experts)``.

    Args:
        num_experts: Number of experts.
        block_size: GEMM tile size (M dimension); default 64.
        kernel_map: Optional override for kernel dispatch.
    """

    def __init__(
        self,
        num_experts: int,
        block_size: int = 64,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ) -> None:
        # The manifest shape_rules, checked where the GPU checked them (at build).
        if num_experts <= 0:
            raise ValueError(
                f"{type(self).__name__}: shape rule 'num_experts > 0' does not hold "
                f"(got {num_experts})"
            )
        if block_size <= 0:
            raise ValueError(
                f"{type(self).__name__}: shape rule 'block_size > 0' does not hold "
                f"(got {block_size})"
            )
        self.num_experts = num_experts
        self.block_size = block_size
        # The routed extents are taken from each call.
        self._numel: Optional[int] = None
        self._kernel_cache: Dict[int, Kernel] = {}
        self.dispatch_kernel(kernel_map)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"permute_align": MoEPermuteAlignKernel}

    def _validate(self, topk_ids: torch.Tensor) -> None:
        """Check the tensor contract the GPU op reads from the manifest signature.

        The signature declares one input, ``topk_ids``, as a 2-D int32 array of
        expert indices in ``[0, num_experts)``.  The value range is the caller's
        obligation on the GPU too (its ``requires`` clause is checked only by the
        opt-in ``validate_inputs`` path); checking it here would stall the device
        on every call, so the op does not.
        """
        backend = get_device_backend()
        if not backend.is_device_tensor(topk_ids):
            raise ValueError(
                f"{type(self).__name__} expects a {backend.name} tensor; "
                "move topk_ids to the device backend first"
            )
        if topk_ids.dtype is not torch.int32:
            raise ValueError(f"topk_ids dtype {topk_ids.dtype} must be int32")
        if topk_ids.dim() != 2:
            raise ValueError(
                f"topk_ids must be (T, K); got shape {tuple(topk_ids.shape)}"
            )

    def _get_or_create_kernel(self, numel: int) -> Kernel:
        """Return the kernel for *numel* routes, building and caching it on first use."""
        kernel = self._kernel_cache.get(numel)
        if kernel is None:
            kernel_cls = self.kernel_map["permute_align"]
            kernel = kernel_cls(numel, self.num_experts, self.block_size)
            self._kernel_cache[numel] = kernel
        self.kernel = kernel
        return kernel

    def forward(self, topk_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run permute-align.

        Args:
            topk_ids: [T, K] int32 expert indices (0-indexed).

        Returns:
            sorted_token_ids: [T * K + (num_experts + 1) * (block_size - 1)] int32
            expert_ids:       [ceil_div(that, block_size)] int32
            num_tokens_post_pad: [1] int32
        """
        self._validate(topk_ids)
        numel = topk_ids.numel()
        kernel = self._get_or_create_kernel(numel)
        assert self.kernel is not None
        self._numel = numel
        return kernel(topk_ids.contiguous())

    def eval_roofline(self) -> tuple[int, int]:
        """Return ``(flops, bytes)`` for the last call's routed extents.

        Ported from the GPU manifest entry: ``flops = 0`` (the op re-maps the
        routed ids; it computes nothing on them), and the byte term is the
        traffic the GPU derives from the signature -- every declared input read
        once and every declared output written once, all int32:

        - ``topk_ids``: ``numel`` entries,
        - ``sorted_token_ids``: ``padded`` entries,
        - ``expert_ids``: ``ceil_div(padded, block_size)`` entries,
        - ``num_tokens_post_pad``: one entry.
        """
        if self._numel is None:
            raise RuntimeError(
                f"{type(self).__name__}.eval_roofline() requires a prior forward() "
                "call to bind the routed extents T * K"
            )
        numel = int(self._numel)
        padded = numel + (self.num_experts + 1) * (self.block_size - 1)
        blocks = math.ceil(padded / self.block_size)
        return 0, (numel + padded + blocks + 1) * _INDEX_BYTES
