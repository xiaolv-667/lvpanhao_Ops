"""Correctness tests for the sampling family (NPU).

Adapted from TileOPs ``tests/ops/test_sampling.py`` -- the
``ChainSpeculativeSamplingFwdOp`` subset only (the shared GPU file also covers
TopKMask / MinPMask / TopPMask / TopKTopPMask / SamplingFromProbs, which are not
migrated).

The GPU file compares the random ops by support, structure and distribution
through ``workloads.numerics.Custom``; the NPU project has no such machinery, so
this file checks the contract directly:

- the reference is reproducible for a pinned ``(seed, offset)`` pair -- the
  fixed-stream contract a seeded draw has to keep;
- ``num_accepted`` is the reference's exactly, so the accepted prefix is too
  (the in-tree kernel draws the reference's Philox stream and applies the same
  float32 acceptance test);
- structure and support: ``-1`` exactly after the drawn token, the drawn token
  carries positive weight (the residual of its position, or target row ``N``
  after a whole chain), and an accepted draft is one its target row can draw;
- the kernel is reproducible: two calls on one input agree bit for bit;
- one many-row case checks the distribution of the accepted length and of the
  drawn token at each position (the GPU distribution probe).

NPU adaptations (T1-T4): manual tensor creation uses the device backend (T1);
the GPU tests carry no ``torch.cuda`` availability guard to remove (T2);
``tune`` / autotune try-except do not apply (T3/T4).
"""

import pytest
import torch

from tileops.device import get_device_backend
from tileops.ops.sampling.chain_speculative_sampling import ChainSpeculativeSamplingFwdOp
from tileops.testing.test_base import exact_compare, record_check_result
from tileops.workloads.sampling import (
    OFFSET,
    SEED,
    ChainSpeculativeSamplingFwdFixture,
    ChainSpeculativeSamplingFwdWorkload,
    assert_chain_distribution,
    verify_chain_outputs,
)


def _run(op, *inputs):
    """One call, without autograd bookkeeping: both outputs are integer tensors."""
    with torch.no_grad():
        return op(*inputs)


@ChainSpeculativeSamplingFwdFixture
def test_chain_speculative_sampling_fwd(batch: int, num_draft: int, vocab: int) -> None:
    """The accepted prefix and ``num_accepted`` are the reference's, at every launch shape.

    The uniforms and the float32 ``u * draft < target`` test are the reference's,
    and neither names a launch fact, so the two agree on how far each chain got
    however the row is split.  The token after the prefix follows the same
    distribution but is not required to be the reference's index.
    """
    test = ChainSpeculativeSamplingFwdWorkload(batch, num_draft, vocab)
    inputs = test.gen_inputs()

    # The reference is a function of the pinned (seed, offset) pair alone.
    ref_tokens, ref_accepted = test.ref_program(*inputs)
    again_tokens, again_accepted = test.ref_program(*inputs)
    exact_compare(again_tokens, ref_tokens)
    exact_compare(again_accepted, ref_accepted)

    op = ChainSpeculativeSamplingFwdOp()
    out_tokens, out_accepted = _run(op, *inputs)
    record_check_result(op, (out_tokens, out_accepted), (ref_tokens, ref_accepted))

    # The accepted length is the reference's exactly (the accepted prefix of the
    # output is checked against the draft ids below, and the reference's is the
    # same prefix).
    exact_compare(out_accepted, ref_accepted)
    verify_chain_outputs(
        out_tokens, out_accepted, inputs[0], inputs[1], inputs[2]
    )

    # A second call on the same input returns the same draws.
    next_tokens, next_accepted = _run(op, *inputs)
    exact_compare(next_tokens, out_tokens)
    exact_compare(next_accepted, out_accepted)


def test_chain_speculative_sampling_draws_from_the_residual() -> None:
    """65536 requests over 16 tokens with two drafts: the GPU distribution probe.

    Each draft id is drawn from its draft row, so the accepted prefix follows
    ``min(draft, target)`` of its position, the drawn token the residual of the
    position the chain stopped at, and the accepted length the per-position
    acceptance probabilities.
    """
    n, num_draft, vocab = 65536, 2, 16
    device = get_device_backend().name
    draft = torch.rand(num_draft, vocab, device=device) ** 3
    draft /= draft.sum(-1, keepdim=True)
    target = torch.rand(num_draft + 1, vocab, device=device) ** 3
    target /= target.sum(-1, keepdim=True)
    draft_ids = torch.multinomial(draft, n, replacement=True).T.to(torch.int32).contiguous()
    inputs = (
        draft.expand(n, num_draft, vocab).contiguous(),
        draft_ids,
        target.expand(n, num_draft + 1, vocab).contiguous(),
        torch.tensor([SEED], dtype=torch.int64, device=device),
        torch.tensor([OFFSET], dtype=torch.int64, device=device),
    )
    op = ChainSpeculativeSamplingFwdOp()
    out_tokens, out_accepted = _run(op, *inputs)
    assert_chain_distribution(out_tokens, out_accepted, draft_ids, draft, target)


@pytest.mark.smoke
def test_chain_speculative_sampling_probabilities_off_a_vector_boundary() -> None:
    """Probabilities starting one element past a 16-byte boundary sample as their aligned copy.

    The draws are seeded, so the aligned and the shifted call have to agree bit
    for bit; the re-reads of the probability rows are what a row longer than a
    16-byte vector is folded from.
    """
    batch, num_draft, vocab = 2, 2, 64
    test = ChainSpeculativeSamplingFwdWorkload(batch, num_draft, vocab)
    inputs = list(test.gen_inputs())
    op = ChainSpeculativeSamplingFwdOp()
    want_tokens, want_accepted = _run(op, *inputs)

    device = get_device_backend().name
    # The two float32 probability tensors, moved one element (4 bytes) past the
    # start of a fresh allocation.
    for i in (0, 2):
        moved = torch.empty(inputs[i].numel() + 1, dtype=inputs[i].dtype, device=device)
        inputs[i] = moved[1:].copy_(inputs[i].reshape(-1)).view(inputs[i].shape)
        assert inputs[i].storage_offset() == 1

    got_tokens, got_accepted = _run(op, *inputs)
    exact_compare(got_tokens, want_tokens)
    exact_compare(got_accepted, want_accepted)


@pytest.mark.smoke
def test_chain_speculative_sampling_rejects_integer_probs() -> None:
    """A first input of integer dtype is refused before anything is built."""
    test = ChainSpeculativeSamplingFwdWorkload(2, 2, 64)
    inputs = list(test.gen_inputs())
    inputs[0] = inputs[0].to(torch.int32)
    with pytest.raises(ValueError, match="float32"):
        ChainSpeculativeSamplingFwdOp()(*inputs)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
