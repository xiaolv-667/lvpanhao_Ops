"""Workload and reference for the sampling op family (NPU).

Ported from TileOPs ``workloads/sampling.py`` -- the ``ChainSpeculativeSampling``
subset only (the shared GPU file also defines the logit-filter and
token-draw references and workloads -- TopKMask / MinPMask / TopPMask /
TopKTopPMask / SamplingFromProbs -- which are not migrated).

Adaptation point W1: ``device`` is resolved from :func:`get_device_backend`
instead of the hard-coded ``"cuda"`` of the GPU code; every reference below is
integer / float tensor arithmetic, so it runs on any device unchanged.

The reference draws its uniforms from Philox4x32-10 keyed by ``seed`` and
counted by ``(draw, row, offset)``, written in integer tensor arithmetic so
that the draws are the same on every device.  The in-tree chain speculative
sampling kernel takes its acceptance uniforms from this same stream, so
``num_accepted`` and the accepted prefix are the reference's exactly; the token
after the prefix follows the same distribution but need not be the reference's
index.  ``SEED`` / ``OFFSET`` are pinned by the workload so that one call's
draws are reproducible: a fixed ``(seed, offset)`` pair gives the same result
for the same inputs.
"""

import torch

from tileops.device import get_device_backend
from tileops.workloads.workload_base import FixtureBase, WorkloadBase

__all__ = [
    "OFFSET",
    "SEED",
    "ChainSpeculativeSamplingFwdFixture",
    "ChainSpeculativeSamplingFwdWorkload",
    "assert_chain_distribution",
    "chain_speculative_sampling",
    "philox_uniform",
    "verify_chain_outputs",
]

_MASK = 0xFFFFFFFF
# Philox4x32 round multipliers and Weyl key increments (Salmon et al., SC'11).
_M0, _M1 = 0xD2511F53, 0xCD9E8D57
_W0, _W1 = 0x9E3779B9, 0xBB67AE85

# The Philox state every workload call passes: the pair the GPU correctness
# tests use, pinned so a call is reproducible (a fixed pair draws the same
# stream, so the reference and the kernel can be compared bit for bit).
SEED = 1234
OFFSET = 7


def _mulhilo(a: torch.Tensor, m: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The high and low 32-bit words of ``a * m`` for 32-bit ``a`` and ``m``, without int64 overflow."""
    low = (a & 0xFFFF) * m
    high = (a >> 16) * m
    mid = low + ((high & 0xFFFF) << 16)
    return ((high >> 16) + (mid >> 32)) & _MASK, mid & _MASK


def philox_uniform(seed: torch.Tensor, offset: torch.Tensor, rows: int, draws: int) -> torch.Tensor:
    """``[rows, draws]`` float32 uniforms in ``[0, 1)``, a function of ``seed`` and ``offset`` alone.

    Draw ``j`` of row ``r`` is the first output word of Philox4x32-10 with key
    ``(seed_lo, seed_hi)`` and counter ``(j, r, offset_lo, offset_hi)``, its top 24 bits
    scaled by ``2**-24``.
    """
    device = seed.device
    s, o = seed.reshape(()), offset.reshape(()).to(device)
    shape = (rows, draws)
    c0 = torch.arange(draws, device=device).expand(shape)
    c1 = torch.arange(rows, device=device)[:, None].expand(shape)
    c2 = (o & _MASK).expand(shape)
    c3 = ((o >> 32) & _MASK).expand(shape)
    k0, k1 = s & _MASK, (s >> 32) & _MASK
    for step in range(10):
        if step:
            k0, k1 = (k0 + _W0) & _MASK, (k1 + _W1) & _MASK
        hi0, lo0 = _mulhilo(c0, _M0)
        hi1, lo1 = _mulhilo(c2, _M1)
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
    return (c0 >> 8).to(torch.float32) * 2.0**-24


def _draw(weights: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Per row, the index an inverse-CDF draw at ``u * total`` lands on; never a zero weight.

    ``weights`` is ``[B, V]``, non-negative with a positive row total; ``u`` is ``[B]`` in ``[0, 1)``.
    """
    batch, vocab = weights.shape
    cdf = weights.double().cumsum(-1)
    point = u.double() * cdf[:, -1]
    index = torch.searchsorted(cdf, point[:, None], right=True).clamp(max=vocab - 1)
    # A draw on a zero weight, which cumsum rounding allows, moves to the next positive one.
    positive = weights > 0
    columns = torch.arange(vocab, device=weights.device).expand(batch, vocab)
    next_positive = torch.where(positive, columns, vocab).flip(-1).cummin(-1).values.flip(-1)
    chosen = next_positive.gather(1, index)[:, 0]
    last_positive = vocab - 1 - positive.flip(-1).int().argmax(-1)
    return torch.where(chosen == vocab, last_positive, chosen).to(torch.int32)


def chain_speculative_sampling(
    draft_probs: torch.Tensor,
    draft_token_ids: torch.Tensor,
    target_probs: torch.Tensor,
    seed: torch.Tensor,
    offset: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Verify each row's draft chain; draws ``0..N-1`` test acceptance, draw ``N`` the token.

    Draft ``i`` is accepted while ``u_i * draft < target`` at its id. The token after the
    accepted prefix is drawn from ``max(0, target - draft)`` at that position, or from target
    row ``N`` when every draft is accepted; later positions are ``-1``.
    """
    batch, n, vocab = draft_probs.shape
    ids = draft_token_ids.view(batch, n).to(draft_probs.device, torch.long)
    target = target_probs.view(batch, n + 1, vocab)
    u = philox_uniform(seed, offset, batch, n + 1)
    d = draft_probs.gather(2, ids[..., None])[..., 0]
    q = target[:, :n].gather(2, ids[..., None])[..., 0]
    num = (u[:, :n] * d < q).int().cumprod(1).sum(1)
    padded_draft = torch.cat([draft_probs, torch.zeros_like(draft_probs[:, :1])], 1)
    at = num.long()[:, None, None].expand(batch, 1, vocab)
    residual = (target.gather(1, at) - padded_draft.gather(1, at))[:, 0].clamp_min(0)
    token = _draw(residual, u[:, n]).long()
    position = torch.arange(n + 1, device=ids.device)[None]
    drafts = torch.cat([ids, torch.full_like(ids[:, :1], -1)], 1)
    stop = num.long()[:, None]
    tokens = torch.where(position < stop, drafts, torch.where(position == stop, token[:, None], -1))
    return tokens.to(torch.int32), num.to(torch.int32)


def _assert_follows(samples: torch.Tensor, probs: torch.Tensor) -> None:
    """Each index's count within ``6 sigma + 5`` of its expectation; a zero-probability index
    never drawn. The ``+ 5`` covers indices expected fewer than a few times."""
    n = samples.numel()
    count = torch.bincount(samples.long(), minlength=probs.numel()).double()
    p = probs.double()
    bound = 6 * (n * p * (1 - p)).sqrt() + 5 * (p > 0)
    assert ((count - n * p).abs() <= bound).all(), (count, n * p)


def verify_chain_outputs(
    tokens: torch.Tensor,
    num_accepted: torch.Tensor,
    draft_probs: torch.Tensor,
    draft_token_ids: torch.Tensor,
    target_probs: torch.Tensor,
) -> None:
    """Check one call's structure against the contract the op promises, exactly.

    The accepted prefix is the draft ids, ``-1`` sits exactly after the drawn
    token, the drawn token carries positive weight (the residual of its
    position, or target row ``N`` after a whole chain), and an accepted draft is
    one its position's target row can draw.  ``num_accepted`` is the accepted
    prefix length, the bonus excluded.

    Ported from the GPU ``ChainSpeculativeSamplingWorkload.verification``
    ``validate`` step; the cross-implementation accepted-length comparison of
    that step is subsumed here by comparing ``num_accepted`` with the reference
    exactly (the kernel draws the reference's stream).
    """
    batch, num_draft, vocab = draft_probs.shape
    rows = torch.arange(batch, device=tokens.device)
    assert tokens.dtype == torch.int32
    assert num_accepted.dtype == torch.int32
    assert ((tokens >= -1) & (tokens < vocab)).all()
    assert ((num_accepted >= 0) & (num_accepted <= num_draft)).all()
    positions = torch.arange(num_draft + 1, device=tokens.device)[None, :]
    # ``-1`` exactly after the drawn token, which sits at the accepted length.
    assert torch.equal(tokens == -1, positions > num_accepted[:, None])
    # The accepted prefix is the drafts.
    prefix = positions[:, :num_draft] < num_accepted[:, None]
    assert torch.equal(tokens[:, :num_draft][prefix], draft_token_ids[prefix])
    # An accepted draft is one its target row can draw.
    support = target_probs[:, :num_draft].gather(-1, draft_token_ids.long()[..., None])[..., 0]
    assert (support[prefix] > 0).all(), "accepted a draft the target cannot draw"
    # The drawn token carries weight of the residual it was drawn from, or of
    # target row ``N`` when the whole chain was accepted.
    padded = torch.cat([draft_probs, torch.zeros_like(draft_probs[:, :1])], 1)
    at = num_accepted.long()
    weights = (target_probs[rows, at] - padded[rows, at]).clamp_min(0)
    assert (weights[rows, tokens[rows, at].long()] > 0).all()


def assert_chain_distribution(
    tokens: torch.Tensor,
    num_accepted: torch.Tensor,
    draft_token_ids: torch.Tensor,
    draft: torch.Tensor,
    target: torch.Tensor,
) -> None:
    """Check one call's distribution when every row draws from the same two rows.

    ``draft`` is ``[N, V]`` and ``target`` ``[N + 1, V]`` -- one distribution
    shared by every row -- and ``tokens`` / ``num_accepted`` / ``draft_token_ids``
    are that many rows of draws.  Each position's tokens follow the weight the
    op promises: the accepted prefix follows ``min(draft, target)`` of its
    position, the drawn token the residual ``max(0, target - draft)`` of the
    position the chain stopped at (or target row ``N`` after a whole chain), the
    first token target row ``0``, and the accepted length the per-position
    acceptance probabilities.

    Ported from the GPU ``_assert_verifies_chains``; the accepted-length
    distribution is computed for any ``N`` (the GPU version spelled out the
    ``N = 2`` case).
    """
    assert tokens.dtype == torch.int32
    assert num_accepted.dtype == torch.int32
    num_draft = draft.shape[0]
    position = torch.arange(num_draft + 1, device=tokens.device)[None]
    prefix = position[:, :-1] < num_accepted[:, None]
    assert torch.equal(tokens[:, :-1][prefix], draft_token_ids[prefix])
    assert (tokens[position > num_accepted[:, None]] == -1).all()
    for stop in range(num_draft + 1):
        weights = target[stop] if stop == num_draft else (target[stop] - draft[stop]).clamp_min(0)
        drawn = tokens[num_accepted == stop, stop]
        assert drawn.numel(), f"no row stopped at {stop}"
        _assert_follows(drawn, weights / weights.sum())
    _assert_follows(tokens[:, 0], target[0])
    for step in range(num_draft):
        kept = torch.minimum(draft[step], target[step])
        accepted = draft_token_ids[num_accepted > step, step]
        assert accepted.numel(), f"no row accepted draft {step}"
        _assert_follows(accepted, kept / kept.sum())
    # Accepted lengths: the chain passes position ``i`` with probability
    # ``sum(min(draft, target))`` of it.
    share = torch.minimum(draft, target[:num_draft]).sum(-1).double()
    survives = torch.ones((), dtype=torch.float64, device=tokens.device)
    want = []
    for step in range(num_draft):
        want.append(survives * (1 - share[step]))
        survives = survives * share[step]
    want.append(survives)
    _assert_follows(num_accepted, torch.stack(want))


class ChainSpeculativeSamplingFwdFixture(FixtureBase):
    @classmethod
    def get_params(cls):
        import pytest

        return [
            (
                "batch, num_draft, vocab",
                [
                    # Smoke: two requests, two drafts, a 64-token vocabulary.
                    pytest.param(
                        2,
                        2,
                        64,
                        marks=[pytest.mark.smoke, pytest.mark.packaging],
                    ),
                    # Full: the GPU launch-boundary cases -- a row split across
                    # blocks, a row one block holds, a row whose bytes are not a
                    # whole number of 16-byte vectors, and a chain longer than
                    # the block, whose uniforms one thread each takes over
                    # several rounds.
                    pytest.param(2, 3, 151936, marks=pytest.mark.full),
                    pytest.param(8, 3, 4099, marks=pytest.mark.full),
                    pytest.param(4, 3, 64, marks=pytest.mark.full),
                    pytest.param(4, 600, 64, marks=pytest.mark.full),
                    pytest.param(300, 3, 32000, marks=pytest.mark.full),
                ],
            ),
        ]


class ChainSpeculativeSamplingFwdWorkload(WorkloadBase):
    """Inputs of one chain speculative sampling call.

    Ported from the GPU ``ChainSpeculativeSamplingWorkload``: the draft and
    target rows are the softmax of standard normal logits over the vocabulary,
    the draft ids are drawn from each draft row (so every accepted draft is one
    its target row can draw with positive probability), and ``seed`` /
    ``offset`` are the pinned Philox pair.  The prob dtype is ``float32`` and
    the ids ``int32``, as the signature fixes.
    """

    def __init__(self, batch: int, num_draft: int, vocab: int):
        if num_draft <= 0:
            raise ValueError(f"num_draft must be positive, got {num_draft}")
        self.batch = batch
        self.num_draft = num_draft
        self.vocab = vocab
        self.dtype = torch.float32

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        device = get_device_backend().name
        b, n, v = self.batch, self.num_draft, self.vocab
        draft_probs = torch.randn(b, n, v, dtype=self.dtype, device=device).softmax(-1)
        target_probs = torch.randn(b, n + 1, v, dtype=self.dtype, device=device).softmax(-1)
        draft_token_ids = torch.multinomial(
            draft_probs.reshape(-1, v), 1
        ).view(b, n).to(torch.int32).contiguous()
        seed = torch.tensor([SEED], dtype=torch.int64, device=device)
        offset = torch.tensor([OFFSET], dtype=torch.int64, device=device)
        return draft_probs, draft_token_ids, target_probs, seed, offset

    def ref_program(
        self,
        draft_probs: torch.Tensor,
        draft_token_ids: torch.Tensor,
        target_probs: torch.Tensor,
        seed: torch.Tensor,
        offset: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return chain_speculative_sampling(draft_probs, draft_token_ids, target_probs, seed, offset)
