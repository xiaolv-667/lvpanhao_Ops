"""Chain speculative sampling kernel (NPU scaffold).

Verifies each request's ``N`` draft tokens and draws the token after the
accepted prefix, over one read of one probability row:

  draft token i is accepted while ``u_i * draft < target`` at its id;
  at the first rejection a token is drawn from
  ``normalize(max(0, target - draft))`` of that position, and the rest of the
  row is ``-1``; when every draft is accepted a bonus token is drawn from
  target row ``N``.  ``num_accepted`` is the accepted prefix length, the bonus
  excluded.

Official-aligned interface (matches ``flashinfer.sampling.chain_speculative_sampling``):

Inputs:
  draft_probs:      [B, N, V]      float32   draft model probabilities
  draft_token_ids:  [B, N]         int32     draft tokens, each in [0, V)
  target_probs:     [B, N + 1, V]  float32   target model probabilities
  seed:             [1]            int64     Philox seed
  offset:           [1]            int64     Philox offset
Outputs:
  output_token_ids: [B, N + 1]     int32     accepted drafts, the drawn token, then -1
  num_accepted:     [B]            int32     accepted prefix length (bonus excluded)

Adaptation summary (GPU -> NPU):

  **Part A -- TileLang kernel functions** (extracted + imported):
    The GPU TileLang kernel factory (``_chain_speculative_sampling_kernel``)
    is extracted from the GPU repo via ``extract_tl_kernel.py`` (resolution
    pattern A) and imported as-is; its ``T.macro`` helpers and the
    ``@tilelang.jit`` builder it returns live inside it, and the helpers it
    calls are appended to the extracted file.  It serves as the reference for
    the NPU kernel component to reimplement for ``target="npuir"``.  K1-K4
    adaptations (decorator, grid/sync semantics, ``threads``, padding) are
    handled by the NPU kernel component during re-implementation.  The GPU
    ``@target_cache(maxsize=32)`` build-target wrapper is not part of the
    extracted factory (the script captures a function from its ``def`` line),
    so the package stays self-contained; the compiled callable is cached here
    per instance instead.

  **Part B -- Kernel class** (fully ported):
    K5: ``supported_archs = None`` (was ``[80, 86, 89, 90]``).
    K6: the shared-memory guard reads ``device_smem_budget`` from
      ``tileops.utils.tiling`` (the GPU read the SM-keyed
      ``BLOCK_SHARED_BYTES_OPT_IN`` table).
    K8: ``autotune_configs`` / ``tune`` absent in the GPU kernel class too;
      heuristic config selection only (``init_config(config)``).
    K9: the GPU kernel carries no ``threads`` config key (``default_config``
      is empty; the launch follows from the call).  The factory still takes
      ``threads`` as a *builder* argument -- K3's removal applies to the NPU
      re-implementation, so the scaffold passes the GPU plan from ``_plan``.
    The GPU ``self.program(self._compile)`` launcher and ``KernelInterface``
    machinery have no NPU counterpart; ``forward`` calls the built program
    directly and caches it (the paged-cache-write / mamba pattern).

  ``ascend_mode`` is intentionally left ``None`` (defer to the ambient
  environment): the programming-mode choice (``simd`` default) belongs to the
  NPU kernel component.

  Pre-Stage-3 note: the imported factory is still the GPU (CUDA target)
  implementation -- it compiles a CUDA-target program with
  ``@tilelang.jit(compile_flags=csrc_include("streaming_load.h"))`` and calls
  the CUDA-only device helpers ``tileops::load16*`` / ``__umulhi``.  Running
  ``forward`` before the NPU rewrite fails; this is the expected scaffold
  state.
"""

from typing import Any, ClassVar, Optional

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.utils.tiling import align_up, device_smem_budget

# Part A: GPU TileLang kernel factory extracted by extract_tl_kernel.py
# (pattern A) from {gpu_repo_root}/tileops/_kernels/sampling/chain_speculative_sampling.py
# -- root /workspace/gpu_mirror_950pr.
# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# Default policy: the perf_opt source becomes active once tuned drop-in
# kernels (same factory signatures) land at
# .chain_speculative_sampling_kernel/perf_opt/{func}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
# and ``pytest benchmarks/ops/`` dispatch through whichever source is
# active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
from .chain_speculative_sampling_kernel import _chain_speculative_sampling_kernel
# --- perf_opt (Stage 4 tuned) ------------------------------------------------
# from .chain_speculative_sampling_kernel.perf_opt._chain_speculative_sampling_kernel import _chain_speculative_sampling_kernel

__all__ = ["ChainSpeculativeSamplingFwdKernel"]

# Bytes of one 16-byte row vector.
_VECTOR_ACCESS_BYTES = 16
# Lanes of one GPU warp (``tileops/_utils/utils.py:WARP_LANES``), which the
# shared-memory guard below counts the descent's scratch in.
_WARP_LANES = 32


def _vector_width(vocab: int, itemsize: int) -> int:
    """Elements of a 16-byte vector, or 1 where a row's bytes are not a whole number of them.

    Ported from the GPU ``row_tiles.vector_width`` (pure arithmetic, device
    agnostic); kept local so the launch plan does not depend on the extracted
    (GPU) module.
    """
    if vocab * itemsize % _VECTOR_ACCESS_BYTES == 0:
        return _VECTOR_ACCESS_BYTES // itemsize
    return 1


class ChainSpeculativeSamplingFwdKernel(Kernel):
    """Verify each request's draft chain and draw the token after it.

    The chain test is ``2 N`` scalars a request, so the launch moves one row:
    the residual ``max(0, target - draft)`` at the position the accepted prefix
    stops at, the draft read as zero at the bonus position.  A fold pass sums
    that row into per-(slot, warp) entries on chip, and the inverse-CDF search
    that follows descends those entries instead of the row, re-reading only the
    vectors the winning warp holds.

    The uniforms are a function of the seed, the offset, the draw index and the
    row index alone, and the acceptance test is the reference's float32
    ``u * draft < target``, so ``num_accepted`` and the accepted prefix are the
    reference's exactly.  The token after the prefix follows the same
    distribution but need not be the reference's index.

    Args:
        batch: Requests in the call (``B``).
        num_draft: Draft tokens per request (``N``).
        vocab: Tokens in a probability row (``V``).
        dtype: Torch dtype of the probability tensors (``float32``).
        config: Unused; the launch follows from the call (GPU parity).
        device_index: Device the call runs on, for the shared-memory query.
    """

    ascend_mode = None

    # K5: [80, 86, 89, 90] (CUDA SM) -> None (all architectures).
    supported_archs: Optional[list] = None

    # GPU launch heuristics, preserved: a thread stages ``_PACE`` vectors of each
    # row before folding any of them, and a 512-thread block of them leaves an
    # SM room for one resident block; a batch that already gives the device more
    # blocks than it has workers takes the narrower 256-thread block instead.
    _THREADS: ClassVar[int] = 512
    _CROWDED_THREADS: ClassVar[int] = 256
    _PACE: ClassVar[int] = 4
    # The GPU plan's ``sm_count`` input on the selected platform: the
    # Ascend950PR_9579 worker count (28 cube / 56 vector cores, from the
    # platform's ``[SoCInfo]`` tables).  The NPU kernel component re-derives the
    # launch plan for the target it implements; this keeps the GPU structure
    # complete until then.
    _BLOCKS: ClassVar[int] = 56

    @classmethod
    def _plan(cls, vocab: int, batch: int, blocks: int) -> tuple[int, int, int, int]:
        """A row's vector width, a block's threads, how many blocks share a row, and its leaves.

        A row whose bytes are a whole number of 16-byte vectors is folded a vector at a
        time; any other row is folded weight by weight. A row is split only while the batch
        leaves blocks idle, and never past its own vectors.
        """
        vec = _vector_width(vocab, torch.float32.itemsize)
        threads = cls._CROWDED_THREADS if batch >= blocks else cls._THREADS
        row_tiles = max(1, -(-(vocab // vec) // threads))
        parts = max(1, min(row_tiles, blocks // max(batch, 1)))
        return vec, threads, parts, -(-row_tiles // parts) * (threads // _WARP_LANES)

    def __init__(
        self,
        batch: int,
        num_draft: int,
        vocab: int,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        device_index: int | None = None,
    ) -> None:
        super().__init__()
        self.batch = batch
        self.num_draft = num_draft
        self.vocab = vocab
        self.dtype = dtype
        self.device_index = device_index

        self.vec, self._threads, self._parts, self._leaves = self._plan(
            vocab, batch, self._BLOCKS
        )
        # GPU parity: a call whose elements exceed int32 indexing is refused
        # before anything is built (the GPU ``refusal`` check, shape half).
        elements = batch * (num_draft + 1) * vocab
        if elements > 2**31 - 1:
            raise ValueError(
                f"chain speculative sampling indexes elements with int32, and "
                f"B * (N + 1) * V = {elements}"
            )
        reason = self._refusal()
        if reason is not None:
            raise ValueError(reason)

        # Build the factory callable (does not compile yet -- compilation is
        # deferred to forward()).  The imported function is the GPU
        # implementation; the NPU component rewrites it for the NPU target.
        self.kernel = _chain_speculative_sampling_kernel(
            batch,
            num_draft,
            vocab,
            self.vec,
            self._threads,
            self._parts,
            self._PACE,
        )
        self._compiled_fn: Optional[Any] = None
        self.init_config(config)

    def _refusal(self) -> str | None:
        """Why this call cannot run, or ``None``.

        One float64 entry per (slot, warp) of a block's chunk must fit the
        device's shared budget less the scratch the descent takes -- the GPU
        check, with the budget read through the device backend (K6).
        """
        budget = device_smem_budget(self.device_index) - (
            align_up(8 * self._parts, 16)
            + 3 * 8 * _WARP_LANES  # tail, lane, prefix
            + 2 * 16  # total, left
            + align_up(4 * max(self.num_draft, 1), 16)  # verdict
            + 4 * 16  # pick, token_uniform, owner_part, warp_at
        )
        if align_up(8 * self._leaves, 16) > budget:
            return f"folds a row into {self._leaves} shared float64 entries, past {budget // 8}"
        return None

    @property
    def default_config(self) -> dict:
        # GPU parity: the launch follows from the call, no config keys.
        return {}

    def forward(
        self,
        draft_probs: torch.Tensor,
        draft_token_ids: torch.Tensor,
        target_probs: torch.Tensor,
        seed: torch.Tensor,
        offset: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Verify the draft chains and draw the token after each accepted prefix.

        Args:
            draft_probs: ``[B, N, V]`` float32 draft model probabilities.
            draft_token_ids: ``[B, N]`` int32 draft tokens, each in ``[0, V)``.
            target_probs: ``[B, N + 1, V]`` float32 target model probabilities.
            seed: ``[1]`` int64 Philox seed.
            offset: ``[1]`` int64 Philox offset.

        Returns:
            ``output_token_ids``, ``[B, N + 1]`` int32: the accepted drafts, the
            drawn token, then ``-1``; and ``num_accepted``, ``[B]`` int32.
        """
        device = target_probs.device
        # The row split's cross-block totals, one float64 entry per (row, part).
        partial = torch.empty(self.batch * self._parts, dtype=torch.float64, device=device)
        output_token_ids = torch.empty(
            self.batch, self.num_draft + 1, dtype=torch.int32, device=device
        )
        num_accepted = torch.empty(self.batch, dtype=torch.int32, device=device)
        if self._compiled_fn is None:
            self._compiled_fn = self.kernel()
        self._compiled_fn(
            draft_probs.view(-1),
            draft_token_ids.view(-1),
            target_probs.view(-1),
            seed,
            offset,
            partial,
            output_token_ids.view(-1),
            num_accepted,
        )
        return output_token_ids, num_accepted
