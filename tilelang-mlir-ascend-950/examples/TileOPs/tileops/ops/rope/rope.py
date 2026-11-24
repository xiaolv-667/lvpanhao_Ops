"""Rotary position embedding operator.

The op holds a cos/sin cache computed from its construction parameters and
rotates ``query`` and ``key`` at the rows ``positions`` names, as vLLM's
``RotaryEmbedding.forward_native(positions, query, key)`` does:

- **RoPEFwdOp**: standard frequencies ``base ** (-2k / rotary_dim)``

``query`` and ``key`` hold each token's heads flattened, ``head_size`` columns a
head; the first ``rotary_dim`` columns of each head rotate and the rest are
copied.

Adaptation from GPU (TileOPs) to NPU:

- ``x.is_cuda``-style device checks -> ``backend.is_device_tensor(x)`` (O1).
- ``tune`` parameter removed (O3); the kernel constructor takes
  ``(query_heads, key_heads, head_size, rotary_dim, max_position, dtype,
  config=None, position_dtype=..., offset_dtype=None, device_index=None)``.
- ``Target`` / ``kernel_for`` / the ``CallSpec`` kernel-selection protocol
  removed: the convention (NeoX / GPT-J) plus the GPU staging rule select one
  key of ``default_kernel_map`` through :meth:`_kernel_key` (plain
  ``rope_{neox,gptj}`` where the heads copy no column, the
  ``rope_*_staged_rows`` variants otherwise); all four GPU
  ``source.kernel_map`` keys are reachable.
- ``param_state(self, signature)`` (GPU manifest codegen) replaced by
  :meth:`_RotaryEmbeddingOp._param_state`, a snapshot of the constructor params
  the cache reads; the cache is recomputed when that snapshot changes.
- ``validate_inputs``: the GPU signature machinery's position-range check is
  implemented directly in :meth:`_RotaryEmbeddingOp._validate_positions_range`.
- Forward flow (validate -> contiguous inputs -> cache -> kernel dispatch ->
  output) preserved unchanged (O6); ``eval_roofline`` inlines the GPU
  ``tileops.perf.formulas.rope_positions_roofline`` (the NPU project has no
  tileops.perf module; roofline arithmetic is device-agnostic).
"""

from __future__ import annotations

from typing import ClassVar, Dict, Mapping, Optional

import torch

from tileops.device import get_device_backend
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.rope import (
    RoPENeoxPositionIdsKernel,
    RoPENeoxStagedRowsKernel,
    RoPENonNeoxKernel,
    RoPENonNeoxStagedRowsKernel,
)
from tileops.ops.op_base import Op

__all__ = ["RoPEFwdOp"]

# The interface of each rotation convention.
_NEOX, _GPTJ = "neox", "gptj"


class _RotaryEmbeddingOp(Op):
    """The rotation every RoPE op shares: cos/sin cache rows read at each token's position.

    A subclass gives its inverse frequencies, its cache rows and its magnitude
    scale.  The cache is computed in float32, rounded to the ``dtype`` parameter
    and then to the query's dtype, as vLLM's cache is.

    Args:
        kernel_map: Optional override for kernel dispatch (see
            ``default_kernel_map``).
    """

    # GPU ``kernel_types``: the four variants the GPU manifest's
    # ``source.kernel_map`` declares (NeoX / GPT-J, plain / staged rows).  These
    # keys are the ONLY dispatch keys: ``_KERNEL_KEYS`` maps a call's
    # (convention, staged) pair onto exactly one of them, so every key below is
    # reachable and a call can never ask for a key the map lacks.
    _KERNEL_MAP: ClassVar[Mapping[str, type[Kernel]]] = {
        "rope_neox_positions": RoPENeoxPositionIdsKernel,
        "rope_gptj": RoPENonNeoxKernel,
        "rope_neox_staged_rows": RoPENeoxStagedRowsKernel,
        "rope_gptj_staged_rows": RoPENonNeoxStagedRowsKernel,
    }
    # (convention, staged) -> manifest kernel key (the GPU ``preferred_over``
    # relation: each staged variant stands in for its convention's plain one).
    _KERNEL_KEYS: ClassVar[Mapping[tuple[str, bool], str]] = {
        (_NEOX, False): "rope_neox_positions",
        (_GPTJ, False): "rope_gptj",
        (_NEOX, True): "rope_neox_staged_rows",
        (_GPTJ, True): "rope_gptj_staged_rows",
    }
    # GPU ``_RoPEStagedRowsKernel`` staged a call only where its rows outnumbered
    # the staged blocks one device wave holds (`sm_count * per_sm`, from shared
    # memory and threads).  The NPU project declares no such device bound (0):
    # every call whose heads copy columns may take the staged variant, and Stage 3
    # sets the real staging policy for the selected backend.
    _WAVE_ROWS: ClassVar[int] = 0
    # Constructor params the cos/sin cache reads (GPU ``param_state(sig)``).
    _cache_params: ClassVar[tuple[str, ...]] = (
        "rotary_dim",
        "max_position_embeddings",
        "base",
        "dtype",
    )

    head_size: int
    rotary_dim: int
    is_neox_style: bool
    dtype: torch.dtype

    def _start(self, kernel_map: Optional[Dict[str, Kernel]] = None) -> None:
        """Install the kernel map and reset the cache / kernel caches."""
        self._freq_cache: dict[tuple, tuple] = {}
        self._kernel_cache: dict[tuple, Kernel] = {}
        self._positions: Optional[torch.Tensor] = None
        self._query_heads = 0
        self._key_heads = 0
        self._query_cols = 0
        self._key_cols = 0
        self._elem_dtype: Optional[torch.dtype] = None
        self.dispatch_kernel(kernel_map)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return dict(self._KERNEL_MAP)

    def _inv_freq(self) -> torch.Tensor:
        """The ``rotary_dim // 2`` inverse frequencies, float32 on the CPU."""
        return 1.0 / (
            self.base
            ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim)
        )

    def _cache_rows(self) -> int:
        """The rows of the cos/sin cache, which positions index."""
        return self.max_position_embeddings

    def _mscale(self) -> float:
        """The magnitude every cos and sin value is scaled by."""
        return 1.0

    def _param_state(self) -> tuple:
        """Snapshot the constructor params the cache reads.

        A container param is copied into a tuple, so editing it in place after
        the snapshot leaves it unequal and the next call recomputes the cache.
        """
        state = []
        for name in self._cache_params:
            value = getattr(self, name, None)
            if isinstance(value, (list, tuple)):
                value = tuple(value)
            state.append((type(value), value))
        return tuple(state)

    def _tables(self, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, ...]:
        """The cos and sin tables, ``(cache rows, rotary_dim // 2)`` in *dtype* on *device*.

        They are computed again once a parameter changes.
        """
        key = (dtype, device)
        sources = self._param_state()
        held = self._freq_cache.get(key)
        if held is None or held[0] != sources:
            t = torch.arange(self._cache_rows(), dtype=torch.float32)
            freqs = torch.outer(t, self._inv_freq())
            mscale = self._mscale()
            tables = tuple(
                (table * mscale).to(self.dtype).to(device=device, dtype=dtype)
                for table in (freqs.cos(), freqs.sin())
            )
            held = self._freq_cache[key] = (sources, tables)
        return held[1]

    def _validate(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: Optional[torch.Tensor],
        backend,
    ) -> None:
        """Check devices, the ``[*S]`` positions and the heads-flattened query / key (O1)."""
        for name, tensor in (("positions", positions), ("query", query), ("key", key)):
            if tensor is not None and not backend.is_device_tensor(tensor):
                raise ValueError(
                    f"{name} must be a {backend.name} tensor, got device {tensor.device}"
                )
        if positions.ndim < 1:
            raise ValueError("positions must have at least one dimension, [*S]")
        if query.shape[:-1] != positions.shape:
            raise ValueError("query must have shape [*S, head_size * Hq], S the positions' shape")
        if query.shape[-1] % self.head_size != 0:
            raise ValueError("query's last dimension must be a multiple of head_size")
        if key is not None:
            if key.shape[:-1] != positions.shape:
                raise ValueError("key must have shape [*S, head_size * Hk], S the positions' shape")
            if key.shape[-1] % self.head_size != 0:
                raise ValueError("key's last dimension must be a multiple of head_size")

    def _validate_positions_range(self, positions: torch.Tensor) -> None:
        """Check on the host that every position lies in the cache (``validate_inputs``).

        This synchronizes the device; it is off by default.
        """
        if not self.validate_inputs or positions.numel() == 0:
            return
        rows = self._cache_rows()
        if int(positions.min()) < 0 or int(positions.max()) >= rows:
            raise ValueError(f"positions must be in [0, {rows})")

    def _staged_for(self, tokens: int, key_heads: int) -> bool:
        """Whether this call takes the "staged rows" variant (GPU ``refusal`` / ``_shape``).

        GPU: a call takes the staged program only where its heads copy columns
        (``rotary_dim < head_size``; at ``rotary_dim == head_size`` the staged
        kernel ``refusal`` is "the heads copy no columns") and its rows (a query
        row and a key row a token) outnumber one wave of staged blocks.  The wave
        bound is a device fact; :attr:`_WAVE_ROWS` declares none, so no row count
        excludes staging here.  Both staged and plain variants compile the same
        program for the selected backend (the design's single block size), so
        this choice cannot change the result.
        """
        if self.rotary_dim == self.head_size:
            return False
        sides = 2 if key_heads else 1
        return tokens * sides > self._WAVE_ROWS

    def _kernel_key(self, tokens: int, key_heads: int) -> str:
        """The ``default_kernel_map`` key of a call of *tokens* tokens.

        The convention (NeoX / GPT-J) and the staging rule select exactly one of
        the four keys the GPU manifest's ``source.kernel_map`` declares.
        """
        convention = _NEOX if self.is_neox_style else _GPTJ
        return self._KERNEL_KEYS[(convention, self._staged_for(tokens, key_heads))]

    def _get_kernel(
        self,
        tokens: int,
        query_heads: int,
        key_heads: int,
        dtype: torch.dtype,
        position_dtype: torch.dtype,
        offset_dtype: Optional[torch.dtype],
    ) -> Kernel:
        """The kernel of this call's convention, staging variant and shapes (cached)."""
        key = (
            self._kernel_key(tokens, key_heads),
            query_heads,
            key_heads,
            dtype,
            position_dtype,
            offset_dtype,
        )
        if key not in self._kernel_cache:
            self._kernel_cache[key] = self.kernel_map[key[0]](
                query_heads,
                key_heads,
                self.head_size,
                self.rotary_dim,
                self._cache_rows(),
                dtype,
                position_dtype=position_dtype,
                offset_dtype=offset_dtype,
                device_index=None,
            )
        return self._kernel_cache[key]

    def _forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: Optional[torch.Tensor],
        offsets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """``query`` and ``key`` with each token's heads rotated at its cache row."""
        backend = get_device_backend()
        self._validate(positions, query, key, backend)
        self._validate_positions_range(positions)
        positions, query = positions.contiguous(), query.contiguous()
        key = None if key is None else key.contiguous()
        offsets = None if offsets is None else offsets.contiguous()
        cos, sin = self._tables(query.dtype, query.device)
        query_heads = query.shape[-1] // self.head_size
        key_heads = 0 if key is None else key.shape[-1] // self.head_size
        kernel = self._get_kernel(
            int(positions.numel()),
            query_heads,
            key_heads,
            query.dtype,
            positions.dtype,
            None if offsets is None else offsets.dtype,
        )
        # Bind the roofline's facts (read by eval_roofline after forward).
        self._positions = positions
        self._query_heads, self._key_heads = query_heads, key_heads
        self._query_cols = int(query.shape[-1])
        self._key_cols = 0 if key is None else int(key.shape[-1])
        self._elem_dtype = query.dtype
        return kernel(positions, query, key, cos, sin, offsets)

    def forward(
        self, positions: torch.Tensor, query: torch.Tensor, key: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Rotate ``query`` and ``key`` at ``positions``.

        Args:
            positions: Each token's position, ``[*S]``, ``int32`` or ``int64``,
                in ``[0, cache rows)``.
            query: ``[*S, head_size * Hq]``, each token's heads flattened.
            key: ``[*S, head_size * Hk]``, or None.

        Returns:
            ``(query_out, key_out)``, shaped as ``query`` and ``key``;
            ``key_out`` is None where ``key`` is.
        """
        return self._forward(positions, query, key)

    def eval_roofline(self) -> tuple[int, int]:
        """Return ``(flops, bytes)`` for this op instance.

        Inlined from the GPU ``tileops.perf.formulas.rope_positions_roofline``:
        three flops per rotated value of query and key.  Query and key are read
        and written whole, positions are read, and only the distinct cache rows
        the positions name are read (cos and sin), in the query's dtype.

        The distinct-row count reads the positions back to the host; roofline
        evaluation happens outside the timed region.
        """
        if self._positions is None or self._elem_dtype is None:
            raise RuntimeError(
                f"{type(self).__name__}.eval_roofline() requires a prior forward() "
                "call to bind dynamic input shape"
            )
        tokens = int(self._positions.numel())
        elem_bytes = self._elem_dtype.itemsize
        rotated = tokens * self.rotary_dim * (self._query_heads + self._key_heads)
        flops = 3 * rotated
        rows_read = int(torch.unique(self._positions).numel())
        nbytes = (
            2 * tokens * (self._query_cols + self._key_cols) * elem_bytes
            + tokens * self._positions.element_size()
            + rows_read * self.rotary_dim * elem_bytes
        )
        return int(flops), int(nbytes)


class RoPEFwdOp(_RotaryEmbeddingOp):
    """RoPE with standard frequencies, either rotation convention (vLLM ``RotaryEmbedding``).

    ``is_neox_style`` rotates the two halves of each head's rotary columns
    against each other; otherwise each adjacent pair rotates (GPT-J).

    Positions in the cache are the caller's obligation, checked only with
    ``validate_inputs=True``; unchecked, a position outside the cache reads its
    nearest row, row 0 or the last.

    Args:
        head_size: Columns per head.
        rotary_dim: Leading columns of each head that rotate; even, at most
            ``head_size``.
        max_position_embeddings: Rows of the cos/sin cache.
        base: Frequency base.
        is_neox_style: Rotate halves (NeoX) rather than adjacent pairs (GPT-J).
        dtype: The cos/sin cache's dtype, before it takes the query's.
        init_cache: Must be True: the op computes its own cache.
        validate_inputs: Check on the host that every position lies in the cache
            before the kernel runs; this synchronizes the device.
        kernel_map: Optional override for kernel dispatch.
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
        init_cache: bool = True,
        *,
        validate_inputs: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ):
        if head_size <= 0:
            raise ValueError(f"head_size must be positive, got {head_size}")
        if rotary_dim <= 0 or rotary_dim % 2 != 0 or rotary_dim > head_size:
            raise ValueError("rotary_dim must be positive, even, and <= head_size")
        if max_position_embeddings <= 0:
            raise ValueError(
                f"max_position_embeddings must be positive, got {max_position_embeddings}"
            )
        if not base > 0:
            raise ValueError(f"base must be positive, got {base}")
        if not init_cache:
            raise ValueError("init_cache must be True: the op computes its own cache")
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype
        self.init_cache = init_cache
        self.validate_inputs = validate_inputs
        self._start(kernel_map)
