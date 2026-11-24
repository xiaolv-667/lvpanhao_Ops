"""The chain speculative sampling op (NPU).

Verifies each request's ``N`` draft tokens in order and draws the token after
the accepted prefix:

  draft token i is accepted with probability ``min(1, target / draft)`` at its
  id, as FlashInfer's ``chain_speculative_sampling`` tests
  ``u * draft < target``; at the first rejection a token is drawn from
  ``normalize(max(0, target - draft))`` of that position and the rest of the row
  is ``-1``; when every draft is accepted a bonus token is drawn from target row
  ``N``.  ``num_accepted`` is the accepted prefix length, the bonus excluded.
  Every probability row finite, non-negative and normalized, and each draft
  token of positive draft probability, is the caller's obligation.

The draws are a function of the Philox state ``(seed, offset)``: the same pair
and the same inputs give the same result.  Which stream a pair selects belongs
to the implementation, so two implementations need not draw the same tokens
from one pair.

Adaptation from GPU (TileOPs) to NPU:

- O1: the GPU op had no device check of its own (the manifest call harness
  checked); ``forward`` checks each input with
  ``backend.is_device_tensor`` and validates the shapes and dtypes the
  signature fixes.  These are metadata checks only -- no device-side read, so a
  benchmarked call keeps one launch.
- O2: nothing hard-codes ``"cuda"``; the tensors' device is used as given.
- O3/O5: the ``tune`` parameter and ``compile_boundary.register_instance`` are
  absent from the GPU op as well; the NPU kernel constructor takes
  ``(batch, num_draft, vocab, dtype, config=None, device_index=None)``.
- O6: the flow is preserved -- ``contiguous()`` on every input, the dims read
  off ``draft_probs``, the kernel dispatched on those dims, the two int32
  tensors returned as the kernel built them.  The GPU
  ``kernel_for(key, lambda: SamplingCall(...))`` call-spec machinery (which
  carries the dtype / vocab / num_draft / sm_count record and the
  ``KernelInterface`` dispatch) has no NPU counterpart; it is replaced by the
  ``_get_kernel`` cache over the same facts.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from tileops.device import get_device_backend
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.sampling import ChainSpeculativeSamplingFwdKernel
from tileops.ops.op_base import Op

__all__ = ["ChainSpeculativeSamplingFwdOp"]


class ChainSpeculativeSamplingFwdOp(Op):
    """Chain speculative sampling: verifies each request's ``N`` draft tokens in order.

    Draft token ``i`` is accepted with probability ``min(1, target / draft)`` at
    its id.  At the first rejection a token is drawn from
    ``normalize(max(0, target - draft))`` of that position and the rest of the
    row is ``-1``; when every draft is accepted a bonus token is drawn from
    target row ``N``.  ``num_accepted`` is the accepted prefix length, the bonus
    excluded.

    The in-tree kernel takes the stream ``tileops/workloads/sampling.py``
    states: uniform ``j`` of a row is the first output word of Philox4x32-10
    keyed by ``(seed, offset)`` and counted by ``(j, row)``, draws ``0`` to
    ``N - 1`` testing the drafts and draw ``N`` placing the token.  The counter
    names no launch fact and the acceptance test is the reference's own float32
    comparison, so ``num_accepted`` and the accepted prefix are the reference's
    exactly, whatever the batch and whatever grid the call takes.

    Args:
        kernel_map: Optional override for kernel dispatch.
    """

    def __init__(self, kernel_map: Optional[Dict[str, Kernel]] = None):
        self.dispatch_kernel(kernel_map)
        self._kernel_cache: Dict[tuple, Kernel] = {}
        self.kernel = None
        self.batch: Optional[int] = None
        self.num_draft: Optional[int] = None
        self.vocab: Optional[int] = None
        self.dtype: Optional[torch.dtype] = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"chain_speculative_sampling": ChainSpeculativeSamplingFwdKernel}

    def _get_kernel(
        self,
        batch: int,
        num_draft: int,
        vocab: int,
        dtype: torch.dtype,
        device_index: int | None,
    ) -> Kernel:
        key = (batch, num_draft, vocab, dtype, device_index)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = self.kernel_map["chain_speculative_sampling"](
                batch,
                num_draft,
                vocab,
                dtype,
                device_index=device_index,
            )
        return self._kernel_cache[key]

    def _validate(self, draft_probs, draft_token_ids, target_probs, seed, offset):
        """Check the device, dtypes and shapes the signature fixes (metadata only)."""
        backend = get_device_backend()
        named = {
            "draft_probs": draft_probs,
            "draft_token_ids": draft_token_ids,
            "target_probs": target_probs,
            "seed": seed,
            "offset": offset,
        }
        for name, tensor in named.items():
            if not backend.is_device_tensor(tensor):
                raise ValueError(f"{name} must be a {backend.name} tensor, got {tensor.device}")
        if draft_probs.dtype != torch.float32 or target_probs.dtype != torch.float32:
            raise ValueError(
                "draft_probs and target_probs must be float32, got "
                f"{draft_probs.dtype} and {target_probs.dtype}"
            )
        if draft_token_ids.dtype != torch.int32:
            raise ValueError(f"draft_token_ids must be int32, got {draft_token_ids.dtype}")
        if seed.dtype != torch.int64 or offset.dtype != torch.int64:
            raise ValueError(
                f"seed and offset must be int64, got {seed.dtype} and {offset.dtype}"
            )
        if draft_probs.ndim != 3:
            raise ValueError("draft_probs must have shape [B, N, V]")
        batch, num_draft, vocab = draft_probs.shape
        if num_draft <= 0:
            raise ValueError(f"N must be positive, got {num_draft}")
        if tuple(draft_token_ids.shape) != (batch, num_draft):
            raise ValueError(
                "draft_token_ids must have shape [B, N] = "
                f"{(batch, num_draft)}, got {tuple(draft_token_ids.shape)}"
            )
        if tuple(target_probs.shape) != (batch, num_draft + 1, vocab):
            raise ValueError(
                "target_probs must have shape [B, N + 1, V] = "
                f"{(batch, num_draft + 1, vocab)}, got {tuple(target_probs.shape)}"
            )
        if seed.numel() != 1 or offset.numel() != 1:
            raise ValueError("seed and offset must hold one element each")

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
        self._validate(draft_probs, draft_token_ids, target_probs, seed, offset)
        draft_probs = draft_probs.contiguous()
        draft_token_ids = draft_token_ids.contiguous()
        target_probs = target_probs.contiguous()
        seed = seed.contiguous()
        offset = offset.contiguous()
        batch, num_draft, vocab = draft_probs.shape

        self.batch = batch
        self.num_draft = num_draft
        self.vocab = vocab
        self.dtype = draft_probs.dtype
        self.kernel = self._get_kernel(
            batch, num_draft, vocab, draft_probs.dtype, draft_probs.device.index
        )
        return self.kernel(draft_probs, draft_token_ids, target_probs, seed, offset)

    def eval_roofline(self) -> tuple[int, int]:
        """Return ``(flops, bytes)`` for this op instance.

        The GPU manifest formula, copied verbatim: the work depends on where the
        chain stops, which the random draws decide, so both counts are the
        cheaper of the two extreme outcomes --

          all accepted: ``2 N + 3 V + 1`` (N ratio tests, a draw from target row
          N at 3 per token and 1 per row);
          rejected at the first: ``5 V + 3`` (one test, the residual at 2 per
          token, a draw from it).

        ``bytes`` adds the once-per-call tensors (seed, offset,
        ``output_token_ids``, ``num_accepted``) to the per-row traffic of the
        same cheaper extreme: ``12 N + 4 V`` bytes all accepted (N ids, 2 N
        float32 probabilities, one float32 target row), ``4 + 8 V`` bytes
        rejected at the first (one id, the two rows).
        """
        if self.batch is None or self.dtype is None:
            raise RuntimeError(
                f"{type(self).__name__}.eval_roofline() requires a prior forward() "
                "call to bind dynamic input shape"
            )
        batch = int(self.batch)
        num_draft = int(self.num_draft)
        vocab = int(self.vocab)
        elem = self.dtype.itemsize
        index = torch.int32.itemsize

        flops = batch * min(2 * num_draft + 3 * vocab + 1, 5 * vocab + 3)
        nbytes = (2 * 8 + index * batch * (num_draft + 1) + index * batch) + batch * min(
            3 * index * num_draft + elem * vocab, index + 2 * elem * vocab
        )
        return int(flops), int(nbytes)
