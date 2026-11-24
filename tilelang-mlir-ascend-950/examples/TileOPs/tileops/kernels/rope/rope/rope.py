"""Rotary position embedding forward kernels (NPU-adapted).

A kernel receives cos/sin tables and rotates each token's heads at a table row::

    y = x * cos + rotate(x) * sin

- ``RoPENeoxPositionIdsKernel``: NeoX halves, ``(x[k], x[k + rotary_dim / 2])``.
- ``RoPENonNeoxKernel``: RoFormer (GPT-J) adjacent pairs, ``(x[2k], x[2k + 1])``.
- ``RoPENeoxStagedRowsKernel`` / ``RoPENonNeoxStagedRowsKernel``: the same two
  conventions with each token's row staged in shared memory (the GPU "staged
  rows" programs, taken where a head copies columns).

Both conventions rotate the query and the key at the row each token's position
names, in one launch, rotate the leading ``rotary_dim`` columns of each head and
copy the rest, and take the token count from the call: one built kernel serves
every count.

Adaptation summary (GPU -> NPU):

  **Part A -- TileLang kernel functions** (extracted + imported):
    The GPU TileLang kernel factory (``_rope_positions_kernel``) and the module
    helpers it calls (``_head_units`` / ``_staged_sides`` / the ``_rotate_pair``
    ``T.macro``) are extracted from the GPU repo via ``extract_tl_kernel.py`` and
    imported as-is.  They serve as the reference for the NPU kernel component to
    reimplement for the selected target.  K1-K4 (decorator, grid/sync
    semantics, ``threads`` removal, padding strategy) are handled by the NPU
    kernel component during re-implementation.

  **Part B -- Kernel class** (fully ported):
    K5:  ``supported_archs = None`` (was ``[80, 86, 89, 90]``).
    K8:  autotune removed -- heuristic ``default_config`` only, no ``tune``
         argument to ``init_config``.
    K9:  ``threads`` removed from ``default_config`` and the factory call.
    K11: the GPU per-block ``threads * units`` product is collapsed into a
         single ``units_per_block`` (this is not an elementwise op, but its
         factory callable also carries the CUDA block decomposition).

  **GPU machinery not ported** (no NPU counterpart): the ``CallSpec`` /
    ``KernelInterface`` / ``Entry`` protocol, ``entry_for``, ``refusal`` /
    ``preferred_over`` static kernel selection, and the shared-memory /
    resident-thread device terms (``device_smem_budget``, ``threads_per_sm``,
    ``sm_count``) that computed the staged wave.  The Op keeps the *rule* those
    terms fed: a call whose heads copy no column takes the plain variant, one
    whose heads copy columns takes the staged variant (``_WAVE_ROWS = 0``
    declares no device bound).  Both variants compile the same program here, so
    the choice cannot change the result.

  Pre-Stage-3 note: the imported factory is still the GPU (CUDA target)
    implementation whose inner callable takes ``(threads, units, staged)``.
    ``_compile`` passes the NPU-shaped argument list ``(units_per_block,
    staged)``; it binds once the NPU kernel component rewrites the factory for
    the selected target and drops ``threads`` from the callable signature (K3,
    K9, K11).  Running ``forward`` before that rewrite fails -- this is the
    expected scaffold state.
"""

import math
from typing import Optional

import torch

from tileops.kernels.kernel_base import Kernel

# Part A: GPU TileLang kernel factory + helpers extracted by
# extract_tl_kernel.py (pattern B) from
# {gpu_repo_root}/tileops/_kernels/rope.py.
# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# Default policy: the perf_opt source becomes active once tuned drop-in
# kernels (same factory signatures) land at
# .rope_kernel/perf_opt/{func}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
# and ``pytest benchmarks/ops/`` dispatch through whichever source is
# active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
from .rope_kernel import _rope_positions_kernel
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
# from .rope_kernel.perf_opt._rope_positions_kernel import _rope_positions_kernel

__all__ = [
    "RoPENeoxPositionIdsKernel",
    "RoPENeoxStagedRowsKernel",
    "RoPENonNeoxKernel",
    "RoPENonNeoxStagedRowsKernel",
]

_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

# Widest vectorized global access the kernels plan for: 128 bits.  The GPU
# source takes it from ``tileops._kernels.constants.VECTOR_ACCESS_BYTES``; the
# NPU project has no such module, so the device-agnostic constant is defined
# here (the value is identical).
_VECTOR_ACCESS_BYTES = 16


def _head_units(
    neox: bool, head_size: int, rotary_dim: int, rot_vec: int, tail_vec: int
) -> tuple[int, int, int]:
    """A unit's width in columns, and a head's rotating and copying unit counts.

    Integration glue: the GPU source takes this arithmetic from the extracted
    kernel module, while the Stage 3 NPU product folds it into the factory
    (``rot_vec`` / ``tail_vec`` / ``row_vec`` carry no NPU meaning there).  It
    only feeds the wrapper's occupancy hint, which no longer changes the
    compiled program, so the arithmetic is kept here unchanged rather than
    re-exported from the kernel module.
    """
    width = rot_vec if neox else max(rot_vec, 2)
    rotating = (rotary_dim // 2 if neox else rotary_dim) // width
    return width, rotating, (head_size - rotary_dim) // tail_vec


class _RoPEPositionsKernel(Kernel):
    """The rotation at each token's position, one convention per subclass.

    Each thread moves whole vectors: the widest power of two up to 16 bytes
    dividing every column offset its unit starts at.

    Args:
        query_heads: Heads of the query (``query.shape[-1] // head_size``).
        key_heads: Heads of the key, 0 for a call without a key.
        head_size: Columns per head.
        rotary_dim: Leading columns of each head that rotate; even, at most
            ``head_size``; the rest are copied.
        max_position: Rows of the cos/sin tables (positions index them).
        dtype: Data type of query, key and the tables.
        config: Optional kernel configuration override
            (``{"units_per_block": <int>}``).
        position_dtype: ``torch.int32`` or ``torch.int64``, the dtype of
            ``positions``.
        offset_dtype: ``torch.int32`` / ``torch.int64`` / None, the dtype of the
            row ``offsets`` (None for a call without offsets).
        device_index: Device the kernel is built for (unused by the NPU
            toolchain's kernel cache).
    """

    # K5: [80, 86, 89, 90] (CUDA SM) -> None (all architectures).
    supported_archs: Optional[list] = None

    # NPU programming mode this wrapper's integrated kernel source requires.
    # The ascend Stage 3 product declares no Developer/Expert form (the platform
    # rule sets ``ascend_mode = None``), so ``Kernel.__call__`` runs ``forward``
    # unscoped.  Declared once on the class that subclasses ``Kernel`` directly
    # and inherited by the four convention/staging variants, so the Stage 5 gate
    # finds exactly one Kernel class carrying exactly one ``ascend_mode``.
    ascend_mode = None

    SUPPORTED_DTYPES = _FLOAT_DTYPES
    _NEOX: bool
    # True for the "staged rows" programs (one token's query row or key row in
    # shared memory); False for the plain positions programs.
    _STAGED: bool = False
    # GPU: ``sm_count * threads_per_sm``, the device's resident threads, which
    # decided between a unit and several a thread.  The NPU execution model has
    # no CUDA threads; 0 means "not declared", so ``_shape`` stays narrow.  The
    # hint never changes the block: the design fixes one ``units_per_block``.
    _resident_threads: int = 0

    def __init__(
        self,
        query_heads: int,
        key_heads: int,
        head_size: int,
        rotary_dim: int,
        max_position: int,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        *,
        position_dtype: torch.dtype = torch.int64,
        offset_dtype: Optional[torch.dtype] = None,
        device_index: Optional[int] = None,
    ):
        super().__init__()
        if self.SUPPORTED_DTYPES is not None and dtype not in self.SUPPORTED_DTYPES:
            supported = ", ".join(str(dt) for dt in self.SUPPORTED_DTYPES)
            raise ValueError(
                f"{self.__class__.__name__} only supports dtypes [{supported}], got {dtype}"
            )
        if rotary_dim <= 0 or rotary_dim % 2 != 0 or rotary_dim > head_size:
            raise ValueError("rotary_dim must be positive, even, and <= head_size")
        if query_heads <= 0 or key_heads < 0:
            raise ValueError(
                "query_heads must be positive and key_heads not negative, got "
                f"{query_heads} and {key_heads}"
            )
        if max_position <= 0:
            raise ValueError(f"max_position must be positive, got {max_position}")
        if position_dtype not in (torch.int32, torch.int64):
            raise ValueError(f"position_dtype must be int32 or int64, got {position_dtype}")
        if offset_dtype not in (None, torch.int32, torch.int64):
            raise ValueError(f"offset_dtype must be int32, int64 or None, got {offset_dtype}")
        self.query_heads = query_heads
        self.key_heads = key_heads
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position = max_position
        self.dtype = dtype
        self.position_dtype = position_dtype
        self.offset_dtype = offset_dtype
        self.device_index = device_index
        lanes = _VECTOR_ACCESS_BYTES // dtype.itemsize
        span = rotary_dim // 2 if self._NEOX else rotary_dim
        rot_vec = math.gcd(lanes, head_size, span)
        tail_vec = math.gcd(lanes, head_size, rotary_dim)
        _, rotating, copying = _head_units(self._NEOX, head_size, rotary_dim, rot_vec, tail_vec)
        self._token_units = (query_heads + key_heads) * (rotating + copying)
        # The extracted GPU factory (CUDA target); see the module docstring's
        # pre-Stage-3 note.  ``_compile`` calls it with the NPU-shaped argument
        # list once the NPU kernel component rewrites it for the selected backend.
        self.kernel = _rope_positions_kernel(
            query_heads,
            key_heads,
            head_size,
            rotary_dim,
            max_position,
            self.dtype_str,
            self.dtype_to_str(position_dtype),
            None if offset_dtype is None else self.dtype_to_str(offset_dtype),
            self._NEOX,
            rot_vec,
            tail_vec,
            math.gcd(lanes, head_size),
        )
        self._programs: dict[tuple, object] = {}
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        """Return the default config (K9/K11: ``threads * units`` collapsed).

        GPU: ``{"threads": 128, "units": 1 if float32 else 4}`` (128 or 512 units
        a block).  NPU: the design (Stage 1, reviewed at Stage 2) fixes **one**
        block size for the migrated kernel -- 2048 units per block, i.e. 16
        elements per lane and 4096 elements per block -- so the config carries
        ``{"units_per_block": 2048}`` for every dtype and both the Stage 3 kernel
        and this integration-time wrapper run the same block count.  Stage 3 may
        override it only by also changing this default.
        """
        return {"units_per_block": 2048}

    def _shape(self, tokens: int) -> bool:
        """Whether a call of *tokens* tokens is wide (a GPU occupancy hint only).

        GPU: a call whose units fill at most half the device's resident threads
        is bound by the latency of one unit, so each thread takes one; a larger
        one keeps several in flight a thread.  The formula is preserved but the
        hint no longer changes the NPU program: the design fixes a single block
        size (see :attr:`default_config`), so both values compile the same
        program.  With no declared resident-thread count, it is always narrow.
        """
        return 2 * tokens * self._token_units <= self._resident_threads

    def _compile(self, cfg: dict, wide: bool):
        """The program ``forward`` launches, compiled from *cfg*.

        GPU: ``self.kernel(threads, units, staged)`` -- a block holds
        ``threads * units`` units of one token, ``units`` a thread.  NPU
        (K3/K9/K11): the CUDA ``threads`` dimension is gone and the product
        collapses into ``units_per_block``; the second argument selects the
        shared-memory (staged) variant.

        *wide* is accepted for parity with the GPU scheduling hook but does not
        change the program: both of its values compile the same ``units_per_block``
        block (the design's single block size), so a stale occupancy figure can
        never move Stage 5 onto a different block count.
        """
        return self.kernel(cfg["units_per_block"], self._STAGED)

    def _program(self, cfg: dict, wide: bool):
        """``_compile`` cached per (units_per_block, wide) -- the GPU factory's
        ``@target_cache(maxsize=32)`` equivalent."""
        key = (cfg.get("units_per_block"), wide, self._STAGED)
        if key not in self._programs:
            self._programs[key] = self._compile(cfg, wide)
        return self._programs[key]

    @staticmethod
    def _aligned(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Return *tensor* based at a ``_VECTOR_ACCESS_BYTES`` boundary.

        Integration glue for the NPU memory contract: the kernel's per-lane
        global accesses are vectorized, so its operands must start on a vector
        boundary.  ``contiguous()`` keeps a view's storage offset, so a caller's
        ``torch.empty(n + 1)[1:]`` reaches the kernel 2 bytes in and the device
        rejects the launch (ACL device error 507035, measured: fp16 element
        offsets 1, 2, 3 and 10 fault while 4, 8, 12 stay correct -- an 8-byte
        minimum that the 16-byte vector width covers).  A misaligned input is
        copied once here (values, shape and dtype unchanged, so the result is
        identical); an aligned input, or None, is returned untouched and costs
        nothing.  The kernel is not involved, so its program is unaffected.
        """
        if tensor is None or tensor.data_ptr() % _VECTOR_ACCESS_BYTES == 0:
            return tensor
        return tensor.clone()

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: Optional[torch.Tensor],
        cos: torch.Tensor,
        sin: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Rotate each token's leading rotary columns; nothing is written in place.

        Args:
            positions: ``[*S]`` in ``position_dtype``, contiguous.
            query: ``[*S, query_heads * head_size]`` in ``dtype``, contiguous.
                A base address that is not ``_VECTOR_ACCESS_BYTES``-aligned (a
                contiguous view into a larger buffer, say) is copied once; see
                :meth:`_aligned`.
            key: ``[*S, key_heads * head_size]``, or None; with ``key_heads`` 0
                it may still be present, and empty.
            cos: The cosine table, ``(max_position, rotary_dim // 2)``.
            sin: The sine table, the same shape and dtype.
            offsets: ``[*S]`` added to ``positions``, or None.

        Returns:
            ``(query_out, key_out)``: new contiguous tensors shaped and typed as
            ``query`` and ``key``; ``key_out`` is None where ``key`` is.
        """
        if positions.numel() == 0:
            return torch.empty_like(query), None if key is None else torch.empty_like(key)
        rows = self._aligned(positions.view(-1))
        program = self._program(self.config, self._shape(rows.numel()))
        query_rows = self._aligned(query).view(rows.numel(), query.shape[-1])
        if not self.key_heads and offsets is None:
            # A key of no heads is built into the query-only program.
            query_out = program(rows, query_rows, cos, sin).view(query.shape)
            return query_out, None if key is None else torch.empty_like(key)
        tensors = (rows, query_rows, self._aligned(key).view(rows.numel(), key.shape[-1]))
        if offsets is not None:
            tensors += (self._aligned(offsets.view(-1)),)
        query_out, key_out = program(*tensors, cos, sin)
        return query_out.view(query.shape), key_out.view(key.shape)


class RoPENeoxPositionIdsKernel(_RoPEPositionsKernel):
    """GPT-NeoX rotation of each token's heads at the table row its position names.

    Pair ``k`` becomes ``(x[k] * cos[k] - x[k + h] * sin[k], x[k + h] * cos[k] +
    x[k] * sin[k])`` with ``h = rotary_dim / 2``.
    """

    _NEOX = True


class RoPENonNeoxKernel(_RoPEPositionsKernel):
    """RoFormer (GPT-J) adjacent-pair rotation of each token's heads at its position's row.

    Reference: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding".
    """

    _NEOX = False


class RoPENeoxStagedRowsKernel(_RoPEPositionsKernel):
    """GPT-NeoX rotation with each token's query row and key row staged in shared memory.

    GPU: a block writes its row back only after the whole row has arrived and
    rotated, and a call takes the staged program only where its heads copy
    columns (``rotary_dim < head_size``) and its rows outnumber the staged blocks
    the device could hold at once (an upper bound from shared memory and
    threads).  The shared-memory budget, the SM count and the threads-per-SM
    terms that decided this on CUDA are device facts with no NPU analogue; the
    ported Op keeps the copy-columns half of the rule and declares no wave bound
    (``_WAVE_ROWS = 0``), and Stage 3 sets the NPU staging policy.  Both staged
    and plain variants compile the same program for the selected backend.
    """

    _NEOX = True
    _STAGED = True


class RoPENonNeoxStagedRowsKernel(_RoPEPositionsKernel):
    """RoFormer (GPT-J) rotation with each token's rows staged in shared memory.

    See :class:`RoPENeoxStagedRowsKernel` for the GPU staging rule and its NPU
    re-design note.
    """

    _NEOX = False
    _STAGED = True
