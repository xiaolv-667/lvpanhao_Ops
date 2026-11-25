"""MoE token-to-expert alignment kernel using TileLang (NPU-adapted).

Converts ``topk_ids`` into the three index arrays a MoE grouped GEMM consumes::

  sorted_token_ids  [P]                - route indices grouped by expert, each
                                         expert's routes first in its padded
                                         range, numel elsewhere
  expert_ids        [ceil_div(P, bs)]  - the owner of each block of those ranges,
                                         num_experts past the padded length
  num_tokens_post_pad  [1]             - total padded route count

The kernel picks its path when it is built: a call with fewer than 1024 routes
and at most one warp of experts takes one fused small-batch kernel; any other
call takes the align kernel and, when it has routes, the scatter kernel after it.

Adaptation summary (GPU -> NPU):

  **Part A -- TileLang kernel functions** (extracted + imported):
    The GPU TileLang kernel factories (``_make_align_kernel``,
    ``_make_scatter_kernel``, ``_make_small_batch_kernel``) are extracted from
    the GPU repo via ``extract_tl_kernel.py`` and imported as-is.  They serve
    as the reference for the NPU kernel component to reimplement for
    ``target="npuir"``.  K1-K4 adaptations (decorator, grid/sync semantics,
    ``threads`` removal, padding strategy) are handled by the NPU kernel
    component during re-implementation: the extracted align factory's inner
    callable still takes ``threads``, while the NPU shape takes no launch
    parameter.  The module constants the factories read (``WARP_LANES``,
    ``WARP_SHUFFLE_STAGES``) are defined in the extracted file itself, so the
    ``moe_permute_align/`` package is self-contained.

  **Part B -- Kernel class** (fully ported):
    K5: ``supported_archs = None`` (was ``[80, 86, 89, 90]``).
    K8: ``autotune_configs`` / ``autotune()`` / ``tune`` removed -- heuristic
        config selection only (``init_config(config)``).
    K9: ``threads`` removed from ``default_config`` and from the compiled
        program's argument list (GPU default was ``threads=MAX_BLOCK_THREADS``,
        and the class rejected a thread count outside ``[32, 1024]``; the
        backend chooses its block width on NPU, so the class carries no launch
        parameter and no such check).
    GPU ``entry_for`` (the ``PermuteAlignCall`` / ``Entry`` build cache) and
    ``Kernel._require_devices`` have no NPU counterpart: the op resolves the
    geometry from each call and validates the tensors itself, and the class
    caches its compiled programs per instance (it is built per ``numel``).
    ``dtype`` does not apply -- every tensor this kernel reads or writes is an
    int32 index array whose element type the grouped-GEMM index contract fixes,
    so the class has no free element type to select.  The GPU kernel class says
    the same.

  Stage 5 note: the imported factories are the NPU (``target="ascend"``)
  implementations produced by the conductor pipeline and integrated under
  ``moe_permute_align_kernel/``.  Each one is a single-level factory:
  ``factory(numel, num_experts, block_size) -> builder()``, and the built
  program takes the NPU-shaped argument list (no ``threads``).  The three
  kernels use different programming modes (align ``simd``, scatter and
  small-batch ``simt``), so ``ascend_mode`` is ``None`` -- the mode is fixed
  per kernel module, not per class.
"""

import math
from typing import Any, Optional

import torch

from tileops.kernels.kernel_base import Kernel

# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# Default policy: the perf_opt source becomes active once tuned drop-in
# kernels (same factory signatures) land at
# moe_permute_align_kernel/perf_opt/{func}.py and pass their L0/L1
# regression; the baseline (Stage 3) source is active otherwise.
# ``pytest tests/ops/`` and ``pytest benchmarks/ops/`` dispatch through
# whichever source is active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
# from .moe_permute_align_kernel import (
#     _make_align_kernel,
#     _make_scatter_kernel,
#     _make_small_batch_kernel,
# )
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
# Active: Stage 4 tuned sources (measured 650.67 -> 30.67 us T_op on
# kimi-k2-t4096, 12/12 manifest workloads improved; see
# examples/moe_permute_align_perf/perf_opt/opt_log.md).
from .moe_permute_align_kernel.perf_opt._make_align_kernel import _make_align_kernel
from .moe_permute_align_kernel.perf_opt._make_scatter_kernel import _make_scatter_kernel
from .moe_permute_align_kernel.perf_opt._make_small_batch_kernel import _make_small_batch_kernel

__all__ = ["MoEPermuteAlignKernel"]

# Fewer routes than this, with at most a warp of experts, take the small-batch
# kernel (GPU parity: the threshold and the warp bound are the GPU's).  The warp
# bound mirrors ``WARP_LANES`` in the extracted file, which defines it for the
# kernel functions it emits.
_SMALL_NUMEL_THRESHOLD = 1024
_WARP_LANES = 32


class MoEPermuteAlignKernel(Kernel):
    """MoE token permutation and alignment kernel.

    Converts ``topk_ids`` into the three index arrays required by MoE grouped GEMM.

    Args:
        numel: Total number of (token, expert) assignments = total_tokens * top_k.
        num_experts: Number of experts.
        block_size: GEMM tile size (M dimension).
        config: Optional config dict.  The GPU entry carried ``threads`` (the
            align kernel's threads per block); K9 removes it, so the heuristic
            config is empty and the backend chooses the block width.
        device_index: The device this kernel is built for.

    Note:
        ``dtype`` does not apply.  Every tensor this kernel reads or writes is an
        int32 index array whose element type is fixed by the MoE grouped GEMM
        index contract, so there is no free element type to select.

    Example:
        ```python linenums="1"
        kernel = MoEPermuteAlignKernel(numel=32, num_experts=8, block_size=16)
        sorted_ids, expert_ids, num_post_pad = kernel(topk_ids)
        ```
    """

    ascend_mode = None

    # K5: [80, 86, 89, 90] (CUDA SM) -> None (all architectures).
    supported_archs: Optional[list] = None

    def __init__(
        self,
        numel: int,
        num_experts: int,
        block_size: int,
        config: Optional[dict] = None,
        *,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.numel = numel
        self.num_experts = num_experts
        self.block_size = block_size
        self.device_index = device_index

        # Build the factory callables (they do not compile yet -- compilation is
        # deferred to forward()).  The imported functions are the integrated NPU
        # (``target="ascend"``) factories from ``moe_permute_align_kernel/``.
        self._small_batch_fn = (
            _make_small_batch_kernel(numel, num_experts, block_size)
            if numel < _SMALL_NUMEL_THRESHOLD and num_experts <= _WARP_LANES
            else None
        )
        if self._small_batch_fn is None:
            self._align_fn = _make_align_kernel(numel, num_experts, block_size)
            # The scatter kernel takes at least one route.
            self._scatter_fn = (
                _make_scatter_kernel(numel, num_experts, block_size) if numel else None
            )
        else:
            self._align_fn = None
            self._scatter_fn = None

        # Compiled programs, built on the first call of the selected path.
        self._small_program: Optional[Any] = None
        self._align_program: Optional[Any] = None
        self._scatter_program: Optional[Any] = None

        self.init_config(config)

    @property
    def default_config(self) -> dict:
        """No launch parameter: K9 removes the GPU's ``threads``.

        Fewer threads only added rounds to the align kernel's count, scan and
        fill loops on the GPU; on NPU the backend picks the block width, so the
        heuristic config is empty.
        """
        return {}

    def forward(
        self, topk_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the alignment kernel.

        Args:
            topk_ids: [total_tokens, top_k] int32 tensor of expert indices (0-indexed).

        Returns:
            sorted_token_ids: [max_num_tokens_padded] int32
            expert_ids:       [num_blocks] int32
            num_tokens_post_pad: [1] int32
        """
        assert topk_ids.dtype == torch.int32, "topk_ids must be int32"
        assert topk_ids.numel() == self.numel, (
            f"Expected numel={self.numel}, got {topk_ids.numel()}"
        )

        flat = topk_ids.flatten().contiguous()
        max_padded = self.numel + (self.num_experts + 1) * (self.block_size - 1)
        max_num_blocks = math.ceil(max_padded / self.block_size)
        dev = flat.device

        sorted_token_ids = torch.empty(max_padded, dtype=torch.int32, device=dev)
        expert_ids = torch.empty(max_num_blocks, dtype=torch.int32, device=dev)
        num_tokens_post_pad = torch.empty(1, dtype=torch.int32, device=dev)

        if self._small_batch_fn is not None:
            self._compiled_small()(flat, sorted_token_ids, expert_ids, num_tokens_post_pad)
        else:
            cumsum = torch.empty(self.num_experts + 1, dtype=torch.int32, device=dev)
            align, scatter = self._compiled_align()
            align(flat, sorted_token_ids, expert_ids, num_tokens_post_pad, cumsum)
            if scatter is not None:
                scatter(flat, sorted_token_ids, cumsum)

        return sorted_token_ids, expert_ids, num_tokens_post_pad

    def _compiled_small(self) -> Any:
        """The small-batch program, which takes no launch parameter (GPU ``_compile_small_batch``)."""
        if self._small_program is None:
            self._small_program = self._small_batch_fn()
        return self._small_program

    def _compiled_align(self) -> tuple[Any, Optional[Any]]:
        """The align program and the scatter program, ``None`` without routes.

        GPU ``_compile_align(cfg)`` passed ``cfg["threads"]``; K9 drops it, so the
        align program takes no launch parameter.
        """
        if self._align_program is None:
            self._align_program = self._align_fn()
            self._scatter_program = self._scatter_fn() if self._scatter_fn is not None else None
        return self._align_program, self._scatter_program
