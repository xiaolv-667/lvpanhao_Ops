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
    calls are appended to the extracted file.  In the integrated package this
    role is taken by the Stage 3 NPU re-implementation: same factory name,
    ``ascend`` target, ``T.SimtVF`` row loop and ``grid = min(B, 56)``.  The GPU
    ``@target_cache(maxsize=32)`` build-target wrapper was never part of the
    extracted factory (the script captures a function from its ``def`` line),
    so the package stays self-contained; the compiled callable is cached here
    per instance instead.

  **Part B -- Kernel class** (fully ported):
    K5: ``supported_archs = None`` (was ``[80, 86, 89, 90]``).
    K6: the GPU shared-memory guard (``device_smem_budget`` /
      ``BLOCK_SHARED_BYTES_OPT_IN``) is dropped: the NPU kernel closes each
      row inside one core's UB and allocates no cross-block scratch.
    K8: ``autotune_configs`` / ``tune`` absent in the GPU kernel class too;
      heuristic config selection only (``init_config(config)``).
    K9: the GPU kernel carries no ``threads`` config key (``default_config``
      is empty; the launch follows from the call).  The NPU factory derives
      ``tile`` / ``threads`` / ``blocks`` from ``(batch, num_draft, vocab)``
      itself, so no launch plan is passed from here.
    The GPU ``self.program(self._compile)`` launcher and ``KernelInterface``
    machinery have no NPU counterpart; ``forward`` calls the compiled program
    directly and caches it (the paged-cache-write / mamba pattern).

  ``ascend_mode`` is intentionally left ``None`` (defer to the ambient
  environment): the programming-mode choice (``simd`` default) belongs to the
  NPU kernel component.

  Stage 5 note: the integrated factory is the NPU re-implementation
  (``ascend`` target, ``T.SimtVF`` row loop, ``grid = min(B, 56)``).  Its
  signature is ``(batch, num_draft, vocab) -> compiled callable``; the
  compiled callable takes the flattened probability rows, the seed/offset as
  int32 word pairs, and the caller-allocated int32 outputs (``out_idx=[]``).
  The GPU launch-plan fields and the ``partial`` cross-block buffer of the
  scaffold are gone with it.
"""

from typing import Any, Optional

import torch

from tileops.kernels.kernel_base import Kernel

# Part A: NPU TileLang kernel factory (integrated Stage 3 product; the GPU
# factory it replaces was extracted by extract_tl_kernel.py -- pattern A --
# from {gpu_repo_root}/tileops/_kernels/sampling/chain_speculative_sampling.py,
# root /workspace/gpu_mirror_950pr).
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
        device_index: Device the call runs on (the op's kernel-cache key).
    """

    ascend_mode = None

    # K5: [80, 86, 89, 90] (CUDA SM) -> None (all architectures).
    supported_archs: Optional[list] = None

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

        # GPU parity: a call whose elements exceed int32 indexing is refused
        # before anything is built (the GPU ``refusal`` check, shape half).
        elements = batch * (num_draft + 1) * vocab
        if elements > 2**31 - 1:
            raise ValueError(
                f"chain speculative sampling indexes elements with int32, and "
                f"B * (N + 1) * V = {elements}"
            )

        # Build the factory callable (does not compile yet -- compilation is
        # deferred to forward()).  The NPU factory derives its own launch plan
        # (tile, lanes, grid = min(B, 56)) from these three dims.
        self.kernel = _chain_speculative_sampling_kernel(batch, num_draft, vocab)
        self._compiled_fn: Optional[Any] = None
        self.init_config(config)

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
        output_token_ids = torch.empty(
            self.batch, self.num_draft + 1, dtype=torch.int32, device=device
        )
        num_accepted = torch.empty(self.batch, dtype=torch.int32, device=device)
        if self._compiled_fn is None:
            self._compiled_fn = self.kernel
        self._compiled_fn(
            draft_probs.contiguous().view(-1),
            draft_token_ids.contiguous().view(-1),
            target_probs.contiguous().view(-1),
            # The kernel takes the Philox state as its two int32 words.
            seed.contiguous().view(torch.int32).view(-1),
            offset.contiguous().view(torch.int32).view(-1),
            output_token_ids.view(-1),
            num_accepted,
        )
        return output_token_ids, num_accepted
