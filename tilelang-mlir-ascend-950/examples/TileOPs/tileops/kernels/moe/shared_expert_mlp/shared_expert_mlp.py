"""Shared expert MLP kernel (NPU-adapted wrapper + Kernel class).

The shared expert of an MoE layer is a dense gated MLP over every token::

    output = down(silu(hidden @ gate.T) * (hidden @ up.T))       [T, H]

with the gate and up rows stacked in one ``w_gate_up`` of ``[2S, H]``
(``gate = w_gate_up[:S]``, ``up = w_gate_up[S:]``) and ``w_down`` of
``[H, S]``.

Adaptation summary (GPU -> NPU):

  **Part A -- TileLang kernel functions** (extracted + imported):
    ``_gemm_basic_kernel``, ``splitk_reduce_kernel`` and
    ``_silu_mul_fused_kernel`` are extracted from the GPU repo via
    ``extract_tl_kernel.py`` into the sibling
    ``_shared_expert_mlp_fwd_kernels.py`` and imported as-is.  They are the
    GPU (CUDA-target) implementations and serve as the reference for the NPU
    kernel component to reimplement for the selected NPU target.  K1-K4
    adaptations (decorator/target, grid/sync semantics, ``threads`` removal,
    padding/masking) are handled by that component during re-implementation:
    the extracted factories' inner callables still take ``threads``, while the
    NPU shape takes no launch parameter.

    The GPU kernel class is a *decomposer*: it owns no fused TileLang program
    and dispatches each GEMM to the generic dense-GEMM library
    (``GEMMCpAsyncKernel`` / ``GEMMTMAKernel`` / ``GEMMTemplate``), so the
    extraction's transitive closure is the whole SM90 GEMM zoo.  The extracted
    file keeps the kernels this decomposition actually runs and documents the
    curation; see its module docstring.

  **Part B -- Kernel class** (fully ported):
    K5: ``supported_archs = None`` (was ``[80, 86, 89, 90]``).
    K8: no autotune policy (the GPU class had none either) -- heuristic
        config selection only, ``init_config(config)`` without ``tune``.
    K9: ``threads`` removed from ``default_config`` and from every child
        program's argument list.
    ``entry_for`` (the GPU ``SharedExpertMLPCall`` / ``Entry`` build cache) has
    no NPU counterpart: the op resolves the geometry from each call, and the
    class builds its child programs per instance, rebuilt when ``config``
    changes.
    The GPU SM90 branch of ``_decompose`` -- the dense template, the fused
    ``silu_and_mul`` epilogue for ``512 <= T <= 2048``, the 192/224/256-wide
    template tiles chosen by wave fill, and the ``template_min_m`` threshold --
    is architecture specialization with no NPU counterpart.  The NPU
    decomposition keeps the shape of the non-SM90 path (two dense GEMMs plus
    the activation) and the decode split-K, which carries over: at ``T <= 32``
    the M dimension is far shorter than a cube tile, so a longer K pass is what
    the GPU buys back there too.

  Stage 5 (integration) note: the imported programs are the NPU (``target=
  "ascend"``, ``ascend_mode = None``) reimplementations landing in the sibling
  ``shared_expert_mlp_kernel/`` package, and ``forward`` passes the NPU-shaped
  argument list (no ``threads``).  The glue layer owns the decompose decision
  only: it hands the GEMM the effective M tile (``_tile_m``) and derives the
  split-K count from the GEMM's exact-tile precondition (``_split_k``), so the
  matching ``splitk_reduce_kernel`` is always built for the same split.
"""

from typing import Any, Optional, Tuple

import torch

from tileops.kernels.kernel_base import Kernel

# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# Default policy: the perf_opt source becomes active once tuned drop-in
# kernels (same factory signatures) land at
# .shared_expert_mlp_kernel/perf_opt/{func}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
# and ``pytest benchmarks/ops/`` dispatch through whichever source is
# active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
from .shared_expert_mlp_kernel import (
    _gemm_basic_kernel,
    _silu_mul_fused_kernel,
    splitk_reduce_kernel,
)
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
# from .shared_expert_mlp_kernel.perf_opt._gemm_basic_kernel import _gemm_basic_kernel
# from .shared_expert_mlp_kernel.perf_opt._silu_mul_fused_kernel import _silu_mul_fused_kernel
# from .shared_expert_mlp_kernel.perf_opt.splitk_reduce_kernel import splitk_reduce_kernel

__all__ = ["SharedExpertMLPKernel"]

# GPU parity: a contraction is split over K only in the decode band, where the
# token count is a small fraction of an M tile and the split is what fills the
# cube.  The GPU gates it at 32 tokens (``_decompose``: ``num_tokens == 32``
# with ``split_k > 1``, and ``small_m_splitk_config`` on the down GEMM).
_SPLIT_K_MAX_TOKENS = 32
# Bounded so every slice keeps K tiles loaded behind one accumulator.
_SPLIT_K_MAX_SLICES = 4
# The split-K reduction maps one fp32 output element per element of a block of
# ``elems_per_cta`` (the GPU default).
_SPLIT_K_ELEMS_PER_CTA = 1024
# The smallest M tile the backend GEMM can hold (its static assert wants a
# multiple of the 16-row fractal).
_MIN_BLOCK_M = 16


def _tile_m(m: int, block_m: int) -> int:
    """The M tile the dense GEMM actually builds for *m* rows.

    Mirrors the kernel's own ``_clamp_block_m`` (``min(block_m,
    max(16, ceil16(m)))``) so the split-K decision below is made against the
    tile the GEMM really uses: the harness default ``block_m = 128`` is clamped
    down for the decode band (``m = 1/32``), and a tile wider than ``m`` would
    make the exact-tile split-K precondition unsatisfiable.
    """
    return min(block_m, max(_MIN_BLOCK_M, -(-m // _MIN_BLOCK_M) * _MIN_BLOCK_M))


class SharedExpertMLPKernel(Kernel):
    """Shared expert MLP producing ``[T, H]``.

    Inputs are ``hidden[T, H]``, concatenated ``w_gate_up[2F, H]`` (gate rows
    then up rows) and ``w_down[H, F]``.

    NPU adaptations:

    - K5: ``supported_archs = None`` (all architectures; was
      ``[80, 86, 89, 90]``).
    - K8: heuristic config selection only; ``init_config(config)`` takes no
      ``tune`` argument.
    - K9: ``threads`` removed from ``default_config`` and from every child
      program's argument list (K11 does not apply: this op has no ``npt`` --
      its activation kernel is a tiled 2-D map, not the flat
      ``T.Parallel(threads, npt)`` form).

    Args:
        num_tokens: Token count ``T``.
        hidden_size: Model width ``H``.
        ffn_size: Expert width ``S`` (the gate/up half-width).
        dtype: Data type of the inputs and the output (float16 or bfloat16).
        config: Optional dict with "block_m", "block_n", "block_k" and
            "num_stages".
        device_index: The device this kernel is built for.

    Example:
        ```python linenums="1"
        kernel = SharedExpertMLPKernel(num_tokens=32, hidden_size=256, ffn_size=128)
        out = kernel(hidden, w_gate_up, w_down)
        ```
    """

    ascend_mode = None

    # K5: [80, 86, 89, 90] (CUDA SM) -> None (all architectures).
    supported_archs: Optional[list] = None

    def __init__(
        self,
        num_tokens: int,
        hidden_size: int,
        ffn_size: int,
        dtype: torch.dtype = torch.bfloat16,
        config: Optional[dict] = None,
        *,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.hidden_size = hidden_size
        self.ffn_size = ffn_size
        self.dtype = dtype
        self.device_index = device_index

        # The child programs, built on first use and rebuilt when the config
        # changes (the GPU class cached them through ``Kernel.program`` keyed by
        # the config dict; the NPU base has no such cache).
        self._program_key: Optional[tuple] = None
        self._programs: Optional[tuple] = None
        self._silu_program: Optional[Any] = None

        self.init_config(config)

    @property
    def default_config(self) -> dict:
        """Default tiling config (the GPU defaults minus the CUDA ``threads``).

        K9: ``threads`` -- a CUDA execution detail -- is dropped; the tiling
        knobs ``block_m`` / ``block_n`` / ``block_k`` / ``num_stages`` keep the
        GPU values.  The GPU ``template_min_m`` threshold is dropped with the
        SM90 template it gates.
        """
        return {
            "block_m": 128,
            "block_n": 256,
            "block_k": 64,
            "num_stages": 3,
        }

    def forward(
        self, hidden: torch.Tensor, w_gate_up: torch.Tensor, w_down: torch.Tensor
    ) -> torch.Tensor:
        """Run the expert: gate/up GEMM, gated activation, down GEMM.

        K9: ``threads`` is not passed to any child program.

        Args:
            hidden: ``[T, H]`` token matrix.
            w_gate_up: ``[2S, H]``, the gate rows then the up rows.
            w_down: ``[H, S]``.

        Returns:
            ``[T, H]`` in the dtype of ``hidden``.
        """
        gate_up_program, down_program, fused, activated = self._programs_for_config()
        if fused is None:
            gate_up = gate_up_program(hidden, w_gate_up)
            if not activated:
                gate_up = self._silu_mul(gate_up)
        else:
            gate_up = fused(hidden, w_gate_up)

        return down_program(gate_up, w_down)

    # -- decomposition ----------------------------------------------------- #

    def _programs_for_config(self) -> tuple:
        """The child programs for the current config, rebuilt when it changes."""
        key = tuple(sorted(self.config.items()))
        if self._program_key != key:
            self._programs = self._decompose(self.config)
            self._silu_program = None
            self._program_key = key
        return self._programs

    def _decompose(self, cfg: dict) -> Tuple[Any, Any, Optional[Any], bool]:
        """The child programs *cfg* decomposes the expert into.

        GPU parity (``SharedExpertMLPKernel._decompose``): returns
        ``(gate_up, down, fused, activated)`` -- the gate/up GEMM
        (``[T, H] x [2S, H]^T``), the down GEMM (``[T, S] x [H, S]^T``), the
        gate/up GEMM with the activation fused into its epilogue when that
        replaces the separate activation pass, and whether gate/up's output is
        already activated.  ``fused`` is ``None`` when the pair is not
        replaced.

        The NPU decomposition serves gate/up with the plain GEMM unless it runs
        split over K, in which case the split-K reduction fuses
        ``silu_and_mul`` (the GPU's ``splitk_reduce_kernel`` epilogue) and the
        separate activation pass disappears with it.
        """
        num_tokens, hidden_size, ffn_size = self.num_tokens, self.hidden_size, self.ffn_size
        gate_up_split = self._split_k(num_tokens, 2 * ffn_size, hidden_size, cfg)
        down_split = self._split_k(num_tokens, hidden_size, ffn_size, cfg)

        gate_up = None
        fused = None
        if gate_up_split == 1:
            gate_up = self._gemm_program(num_tokens, 2 * ffn_size, hidden_size, cfg)
        else:
            fused = self._gemm_program(
                num_tokens,
                2 * ffn_size,
                hidden_size,
                cfg,
                split_k=gate_up_split,
                activation="silu_and_mul",
            )
        down = self._gemm_program(num_tokens, hidden_size, ffn_size, cfg, split_k=down_split)
        return gate_up, down, fused, fused is not None

    def _split_k(self, m: int, n: int, k: int, cfg: dict) -> int:
        """K slices for an ``m x n`` output over a contraction of *k*.

        One slice except in the decode band (``m <= _SPLIT_K_MAX_TOKENS``),
        where the GPU splits K and the split is what fills the cube there.

        A split-K GEMM keeps the source's exact-tile precondition -- every
        slice must get whole K tiles and the M/N tiles must be exact, with
        ``m >= 16`` -- so any geometry that cannot honour it degrades to one
        slice instead of raising.  The M tile is the *effective* tile
        (``_tile_m``), not the config value: ``block_m = 128`` is clamped to the
        decode-band geometry.  The caller builds the matching
        ``splitk_reduce_kernel`` from the returned count, so the choice made
        here is the one the reduction sees.

        The slice count stays at the Stage 3-validated one
        (``_SPLIT_K_MAX_SLICES`` whole K tiles, each slice an equally long
        chain) rather than the smallest divisor of the K tile count: the dense
        GEMM's split path was validated at 4 slices and at 1, and a 2- or
        3-slice chain would be a shape no stage has measured.
        """
        block_m = _tile_m(m, cfg["block_m"])
        block_n = cfg["block_n"]
        block_k, k_eff = self._k_tile(cfg, k)
        if m > _SPLIT_K_MAX_TOKENS or m < _MIN_BLOCK_M:
            return 1
        if m % block_m or n % block_n:
            return 1
        k_tiles = k_eff // block_k
        if k_tiles < _SPLIT_K_MAX_SLICES or k_tiles % _SPLIT_K_MAX_SLICES:
            return 1
        return _SPLIT_K_MAX_SLICES

    def _k_tile(self, cfg: dict, k: int) -> Tuple[int, int]:
        """``(block_k, k_eff)`` for a contraction of *k*.

        The dense GEMM's K tail is not allowed (DESIGN.md 5.2 rule 2): ``block_k``
        must divide ``k``.  The design delegates that resolution to the host
        (DESIGN.md 5.4 rule 2): take the first K tile that divides *k* from the
        configured one down the ``64 / 32 / 16`` ladder the source uses, and when
        not even 16 divides *k* the host pads the contraction up to the next
        multiple of 16 with zeros.  Zeros are exact here -- every padded lane
        contributes nothing to the product -- so the padding changes the tile
        geometry, never the result.

        Returns the tile to build with and the K extent to build for
        (``k_eff >= k``); the GEMM wrapper pads the operands when they differ.
        """
        for cand in (cfg["block_k"], 64, 32, _MIN_BLOCK_M):
            if k % cand == 0:
                return cand, k
        k_eff = -(-k // _MIN_BLOCK_M) * _MIN_BLOCK_M
        for cand in (cfg["block_k"], 64, 32, _MIN_BLOCK_M):
            if k_eff % cand == 0:
                return cand, k_eff
        return _MIN_BLOCK_M, k_eff  # unreachable: k_eff is a multiple of 16

    def _gemm_program(
        self,
        m: int,
        n: int,
        k: int,
        cfg: dict,
        split_k: int = 1,
        activation: str = "none",
    ) -> Any:
        """A dense ``C = A @ B^T`` program for an ``m x n`` output over ``k``.

        ``trans_b=True`` for both of the expert's GEMMs: ``w_gate_up`` and
        ``w_down`` are stored ``[n, k]``.  A split-K program lands its partials
        in fp32 and reduces them into the output through
        ``splitk_reduce_kernel`` (the GPU ``GEMMCpAsyncKernel`` split path).

        The M tile handed to the GEMM is the effective one (``_tile_m``): the
        config's ``block_m`` is the prefill tile and is clamped to the call's
        token count, which is what ``_split_k`` assumed when it chose the slice
        count.  The K tile comes from ``_k_tile``; when it pads the contraction
        (``k % 16 != 0``, which no legal ``block_k`` divides) the two operands
        are zero-extended along K -- the K axis is the last axis of both
        ``A[m, k]`` and ``B[n, k]`` -- so the padded lanes add nothing.
        """
        block_k, k_eff = self._k_tile(cfg, k)
        program = _gemm_basic_kernel(m, n, k_eff, False, True, self.dtype_str)(
            _tile_m(m, cfg["block_m"]),
            cfg["block_n"],
            block_k,
            cfg["num_stages"],
            split_k,
        )
        if k_eff != k:
            pad = k_eff - k
            unpadded = program

            def program(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                return unpadded(
                    torch.nn.functional.pad(a, (0, pad)),
                    torch.nn.functional.pad(b, (0, pad)),
                )

        mainloop = program
        if split_k == 1:
            return mainloop

        reduce_program = splitk_reduce_kernel(split_k, m, n, self.dtype_str, activation)(
            _SPLIT_K_ELEMS_PER_CTA
        )
        out_n = n // 2 if activation == "silu_and_mul" else n

        def gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            partial = mainloop(a, b)
            out = torch.empty((m, out_n), dtype=self.dtype, device=a.device)
            reduce_program(partial, out)
            return out

        return gemm

    def _silu_mul(self, gate_up: torch.Tensor) -> torch.Tensor:
        """Apply ``silu(gate) * up`` to the ``[T, 2S]`` gate/up output."""
        if self._silu_program is None:
            self._silu_program = _silu_mul_fused_kernel(
                self.num_tokens, self.ffn_size, self.dtype_str
            )(self.config["block_m"], self.config["block_n"])
        return self._silu_program(gate_up)
