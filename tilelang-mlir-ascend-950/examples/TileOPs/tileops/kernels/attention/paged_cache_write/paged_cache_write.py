"""Paged KV cache write kernel using TileLang (NPU-adapted).

Scatters token ``n``'s k and v rows into the pool row ``slot_mapping[n]``
names, in place, leaving every row no slot names untouched::

  k_cache[slot_mapping[n]] = k[n] / k_scale     (16-bit cache: plain copy)
  v_cache[slot_mapping[n]] = v[n] / v_scale

One block owns a tile of tokens and a column tile of their rows.  A token
whose slot is ``-1`` is skipped; every other token reads its slot once and
writes the same columns of its key and value rows, so the guard and the
slot load are paid once for the two stores.  An FP8 cache divides by the
scale and uses a saturating cast that preserves NaN.

Adaptation summary (GPU -> NPU):

  **Part A -- TileLang kernel functions** (extracted + imported):
    The GPU TileLang kernel factory (``_paged_cache_write_kernel``) is
    extracted from the GPU repo via ``extract_tl_kernel.py`` and imported
    as-is.  It serves as the reference for the NPU kernel component to
    reimplement for ``target="npuir"``.  K1-K4 adaptations (decorator,
    grid/sync semantics, ``threads`` removal, padding strategy) are
    handled by the NPU kernel component during re-implementation: the
    extracted factory's inner callable still takes
    ``(block_rows, block_cols, threads)``, while the NPU shape is
    ``(block_rows, block_cols)``.

  **Part B -- Kernel class** (fully ported):
    K5: ``supported_archs = None`` (was the GPU arch list).
    K8: ``autotune_configs`` / ``autotune_supply_prog`` / ``tune`` removed
        -- heuristic config selection only (``init_config(config)``).
    K9: ``threads`` removed from ``default_config``, the class constructor
        and the ``forward`` dispatch (GPU default was ``threads=128``).
    E10: no ``@torch.library.custom_op`` wrapper -- the GPU kernel class
        has none either; ``forward`` calls the imported factory directly.

  Dropped GPU decoration: the GPU file applies
  ``@target_cache(maxsize=32)`` to the factory.  The extraction script
  captures the function body from its ``def`` line, so the decorator is
  not part of the extracted file and the NPU import has no unresolved
  name; the NPU side caches the compiled callable per config in
  ``forward`` instead.

  Pre-Stage-3 note: the imported factory is still the GPU (CUDA target)
  implementation.  The dispatch below passes the NPU-shaped argument list
  (no ``threads``); it becomes callable once the NPU kernel component
  rewrites the factory for ``target="npuir"`` and drops ``threads`` from
  the callable signature (K3).  Running ``forward`` before that rewrite
  fails -- this is the expected scaffold state.
"""

from typing import Any, Optional

import torch

from tileops.kernels.kernel_base import Kernel

# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# Default policy: the perf_opt source becomes active once tuned drop-in
# kernels (same factory signatures) land at
# paged_cache_write_kernel/perf_opt/{func}.py and pass their L0/L1
# regression; the baseline (Stage 3) source is active otherwise.
# ``pytest tests/ops/`` and ``pytest benchmarks/ops/`` dispatch through
# whichever source is active here.
# ---------------------------------------------------------------------------
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
# Staged row-tile DMA load + a scattered store path chosen by cache width, with
# the tile resolved per call by ``pick_config`` (see ``forward``).  Measured on
# the migrated benchmark (msprof Task Duration, one launch per call): prefill
# 8192x1024 811.5 -> 48.1 us, qwen3-235b fp8 270.6 -> 37.7 us, decode 16.1 ->
# 4.5 us, all byte-exact.  ``paged_cache_write_kernel/perf_opt/`` carries the
# tuned module; the Stage 3 module below stays untouched.
from .paged_cache_write_kernel.perf_opt._paged_cache_write_kernel import (
    _paged_cache_write_kernel,
    pick_config as _tuned_pick_config,
)
# --- baseline (Stage 3) ------------------------------------------------------
# from .paged_cache_write_kernel import _paged_cache_write_kernel
# _tuned_pick_config = None  # uncomment with the line above to toggle back

__all__ = ["PagedKVCacheWriteKernel"]

# Tiling thresholds measured with the framework harness (10 cases).
# A whole-row column segment measured 8% faster for the 8192-token prefill but
# 4.4x slower for 64-token decode: the row segment removes the column split, so
# the grid shrinks to ``tokens / block_rows`` blocks and small shapes lose the
# vector-core parallelism.  Only switch once the block count can carry a row.
_FULL_ROW_MIN_TOKENS = 6144
_FULL_ROW_MAX = 1024


class PagedKVCacheWriteKernel(Kernel):
    """One pass over the tokens, a row tile and a column tile per block.

    Args:
        heads_kv: Key/value heads one token's row holds.
        dim: Elements per head in a row.
        num_pages: Pages in the pool.
        page_size: Rows in a page.
        dtype: Torch dtype of the rows the call passes in (float16 or
            bfloat16).
        cache_dtype: Torch dtype of the page pool; ``float8_e4m3fn``
            quantizes on write.
        config: Tile sizes (``block_rows``, ``block_cols``), or ``None``
            for ``default_config``.
    """

    ascend_mode = None

    # K5: GPU arch list -> None (all architectures).
    supported_archs: Optional[list] = None

    def __init__(
        self,
        heads_kv: int,
        dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        cache_dtype: torch.dtype,
        config: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.heads_kv, self.dim = heads_kv, dim
        self.num_pages, self.page_size = num_pages, page_size
        self.dtype, self.cache_dtype = dtype, cache_dtype
        self.row_width = heads_kv * dim
        self.physical_tokens = num_pages * page_size
        self.quantized = cache_dtype == torch.float8_e4m3fn
        # Build the factory callable (does not compile yet -- compilation is
        # deferred to forward()).  The imported function is the GPU
        # implementation; the NPU component rewrites it for target="npuir".
        self.kernel = _paged_cache_write_kernel(
            heads_kv,
            dim,
            self.physical_tokens,
            self.dtype_str,
            self.dtype_to_str(cache_dtype),
        )
        self._compiled_fn: Optional[Any] = None
        self._compiled_config: Optional[tuple[int, int]] = None
        self.init_config(config)
        # A caller-supplied config is honoured verbatim; only the default is
        # retiled per call (see ``forward``).
        self._user_config = config is not None

    @property
    def default_config(self) -> dict:
        # A modest row tile exposes parallelism for small token counts; a
        # 128-column tile keeps contiguous stores while masking narrower
        # rows.  K9: no ``threads``.
        return {"block_rows": 16, "block_cols": 128}

    def forward(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        k_pages: torch.Tensor,
        v_pages: torch.Tensor,
        slot_mapping: torch.Tensor,
        k_scale: Optional[torch.Tensor] = None,
        v_scale: Optional[torch.Tensor] = None,
    ) -> None:
        """Write the rows the slots name into both pools.

        Args:
            k: New keys, ``(tokens, heads_kv, dim)``.
            v: New values, the same shape and dtype.
            k_pages: Key page pool, ``(num_pages, page_size, heads_kv, dim)``,
                mutated in place.
            v_pages: Value page pool, same shape, mutated in place.
            slot_mapping: ``int64`` ``(tokens,)``; entry ``n`` is the
                flattened pool row token ``n`` writes, or ``-1`` to skip it.
            k_scale: ``float32`` ``(1,)`` divisor of the stored keys;
                required exactly when the cache is float8_e4m3fn.
            v_scale: The same divisor for the stored values.
        """
        # A call of no tokens writes nothing, and a zero-extent grid is not a
        # legal one; the launch is skipped here instead (GPU parity).
        if k.shape[0] == 0:
            return
        # A slot indexes the page pool as one flat run of rows, so the kernel
        # reads the four-dimensional tensors as rows of ``row_width`` elements.
        args = [
            k.view(k.shape[0], self.row_width),
            v.view(k.shape[0], self.row_width),
            k_pages.view(self.physical_tokens, self.row_width),
            v_pages.view(self.physical_tokens, self.row_width),
            slot_mapping,
        ]
        if self.quantized:
            args += [k_scale, v_scale]
        # K9: the compiled callable takes (block_rows, block_cols); ``threads``
        # is dropped.  Cache the compiled callable per config.
        # A caller-supplied config is honoured verbatim.  Otherwise the tuned
        # module resolves the tile from the call's geometry: its row tile
        # depends on the token count (a dynamic extent) and its column tile on
        # the row width, so the choice has to happen here, per call.
        if self._user_config or _tuned_pick_config is None:
            block_cols = self.config["block_cols"]
            if not self._user_config and k.shape[0] >= _FULL_ROW_MIN_TOKENS:
                block_cols = min(_FULL_ROW_MAX, -(-self.row_width // 128) * 128)
            cfg = (self.config["block_rows"], block_cols)
        else:
            cfg = _tuned_pick_config(
                k.shape[0],
                self.heads_kv,
                self.dim,
                self.physical_tokens,
                self.dtype_str,
                self.dtype_to_str(self.cache_dtype),
            )
            # keep the kernel's published config equal to the tile that runs, so
            # reports/benchmarks that read ``kernel.config`` record the truth
            self.config = {"block_rows": cfg[0], "block_cols": cfg[1]}
        if self._compiled_fn is None or self._compiled_config != cfg:
            self._compiled_fn = self.kernel(*cfg)
            self._compiled_config = cfg
        self._compiled_fn(*args)
