# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Chain speculative sampling (ascend, Stage 4 perf_opt candidate): kernel + golden + tests.

Same contract, same draws, same tiling as the Stage 3 kernel
(``tile = min(8192, ceil(V/128)*128)``, ``threads = 128``, ``blocks = min(B, 56)``,
one row a core): every fp32 grouping that decides a bit is unchanged --

  * the fold's per-thread partial sums (``T.Parallel(tile)`` maps the same
    elements to the same threads in the same order) and the block all-reduce,
  * the coarse prefix over tile sums (serial, ascending),
  * the per-lane sums of the crossing tile (lane ``l`` owns
    ``[l*per_lane, (l+1)*per_lane)``, ascending) and the element walk,

so ``num_accepted`` and the drawn token are the baseline's exactly.  What
changes is how the same arithmetic is spelled:

  1. **Fold**: the draft row is read under one branch per *row* instead of a
     select per *element* (the per-element select compiles to a branch inside
     the unrolled element loop and costs 3.1x the whole fold), and the tail tile
     -- the only tile with a per-element in-row predicate -- is peeled off the
     tile loop.  Two reducers are allocated because one allocation carries one
     reducer epoch.
  2. **Crossing tile**: recomputed once, as before, and the fallback index is
     answered by walking lanes backwards instead of by a second reducer pass
     over the whole tile (same value: the largest positive index of the tile).
  3. **Chain test**: unchanged (Philox4x32-10, the reference's float32
     acceptance test).

``_KNOBS`` selects the spelling of each stage; the shipped defaults are the
measured winners.  The factory signature, the launch shape and the output
contract match Stage 3 so the wrapper's import switch is a drop-in.
"""

import argparse
import functools
import pathlib as _pathlib
import time

import tilelang
import tilelang.ascend.language as T
import torch
import torch_npu  # noqa: F401  -- registers torch.npu

# The Philox state every workload call passes (pinned by the manifest, so one
# call's draws are reproducible and the kernel can be compared bit for bit).
SEED = 1234
OFFSET = 7

_MASK = 0xFFFFFFFF
# Philox4x32 round multipliers and Weyl key increments (Salmon et al., SC'11).
_M0, _M1 = 0xD2511F53, 0xCD9E8D57
_W0, _W1 = 0x9E3779B9, 0xBB67AE85
_ROUNDS = 10
_UNIFORM_BITS = 24
_UNIFORM_SCALE = 2.0 ** -_UNIFORM_BITS

_THREADS = 128
_TILE_CAP = 8192
_MAX_BLOCKS = 56  # vector cores of Ascend950PR_9579 (runtime probe, see run_L0)
_F32 = "float32"

# Stage spellings.  ``fold``: how the residual fold masks the draft row;
# ``recomp``: how the crossing tile is re-read; ``walk``: how the fallback index
# is answered when no lane of the crossing tile carries the draw point.
_KNOBS = {
    "fold": "hoist",    # "select" (Stage 3) | "hoist" | "mul"
    "recomp": "branch",  # "branch" (Stage 3) | "flat"
    "walk": "lane",     # "tile" (Stage 3) | "lane"
    "resid": "mul",     # residual spelling in the crossing tile: "select" | "mul"
    "chain": "local",   # Philox state: "shared" (Stage 3) | "local"
}


# ---------------------------------------------------------------------------
# 1. Golden: torch CPU reference, ported from the op's correctness oracle
#    (``tileops/workloads/sampling.py``).  Device-agnostic integer / float
#    tensor arithmetic; no tiling, no streaming, no fp32 hierarchy.
# ---------------------------------------------------------------------------
def golden_mulhilo(a: torch.Tensor, m: int):
    """The high and low 32-bit words of ``a * m``, without int64 overflow."""
    low = (a & 0xFFFF) * m
    high = (a >> 16) * m
    mid = low + ((high & 0xFFFF) << 16)
    return ((high >> 16) + (mid >> 32)) & _MASK, mid & _MASK


def golden_philox_uniform(seed: torch.Tensor, offset: torch.Tensor, rows: int, draws: int):
    """``[rows, draws]`` float32 uniforms in ``[0, 1)``, a function of ``(seed, offset)``.

    Draw ``j`` of row ``r`` is the first output word of Philox4x32-10 with key
    ``(seed_lo, seed_hi)`` and counter ``(j, r, offset_lo, offset_hi)``, its top
    24 bits scaled by ``2**-24``.
    """
    device = seed.device
    s, o = seed.reshape(()), offset.reshape(()).to(device)
    shape = (rows, draws)
    c0 = torch.arange(draws, device=device).expand(shape)
    c1 = torch.arange(rows, device=device)[:, None].expand(shape)
    c2 = (o & _MASK).expand(shape)
    c3 = ((o >> 32) & _MASK).expand(shape)
    k0, k1 = s & _MASK, (s >> 32) & _MASK
    for step in range(_ROUNDS):
        if step:
            k0, k1 = (k0 + _W0) & _MASK, (k1 + _W1) & _MASK
        hi0, lo0 = golden_mulhilo(c0, _M0)
        hi1, lo1 = golden_mulhilo(c2, _M1)
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
    return (c0 >> 8).to(torch.float32) * _UNIFORM_SCALE


def golden_draw(weights: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Per row, the index an inverse-CDF draw at ``u * total`` lands on; never a zero weight.

    ``weights`` is ``[B, V]``, non-negative with a positive row total; ``u`` is
    ``[B]`` in ``[0, 1)``.  The reference accumulates in float64, left to right.
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


def golden_chain_speculative_sampling(
    draft_probs: torch.Tensor,
    draft_token_ids: torch.Tensor,
    target_probs: torch.Tensor,
    seed: torch.Tensor,
    offset: torch.Tensor,
):
    """Verify each row's draft chain; draws ``0..N-1`` test acceptance, draw ``N`` the token."""
    batch, n, vocab = draft_probs.shape
    ids = draft_token_ids.view(batch, n).to(draft_probs.device, torch.long)
    target = target_probs.view(batch, n + 1, vocab)
    u = golden_philox_uniform(seed, offset, batch, n + 1)
    d = draft_probs.gather(2, ids[..., None])[..., 0]
    q = target[:, :n].gather(2, ids[..., None])[..., 0]
    num = (u[:, :n] * d < q).int().cumprod(1).sum(1)
    padded_draft = torch.cat([draft_probs, torch.zeros_like(draft_probs[:, :1])], 1)
    at = num.long()[:, None, None].expand(batch, 1, vocab)
    residual = (target.gather(1, at) - padded_draft.gather(1, at))[:, 0].clamp_min(0)
    token = golden_draw(residual, u[:, n]).long()
    position = torch.arange(n + 1, device=ids.device)[None]
    drafts = torch.cat([ids, torch.full_like(ids[:, :1], -1)], 1)
    stop = num.long()[:, None]
    tokens = torch.where(
        position < stop, drafts, torch.where(position == stop, token[:, None], -1)
    )
    return tokens.to(torch.int32), num.to(torch.int32)


# ---------------------------------------------------------------------------
# 2. Contract validators (ported from the same oracle): structure and
#    distribution, never the reference's drawn index.
# ---------------------------------------------------------------------------
def _assert_follows(samples: torch.Tensor, probs: torch.Tensor) -> None:
    """Each index's count within ``6 sigma + 5`` of its expectation; a zero-probability
    index is never drawn.  The ``+ 5`` covers indices expected fewer than a few times."""
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
    one its position's target row can draw.
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
    are that many rows of draws.
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


# ---------------------------------------------------------------------------
# 3. Kernel: one core a row, a tile fold with a block reducer, a hierarchical
#    locate (tile -> lane -> element) and a scalar emit.
# ---------------------------------------------------------------------------
def _mulhilo(a, m):
    """``(hi, lo)`` of the 32-bit product ``a * m`` from four 16x16 products.

    Ascend has no ``__umulhi`` and the int64 spelling of the source
    ``(int64(a) * m) >> 32`` is a signed overflow on this backend (wrong high
    word, see the design's trap record), so the high word is rebuilt from 32-bit
    words only: ``a*m = p3*2^32 + (p1+p2)*2^16 + p0``.
    """
    ah = a >> T.uint32(16)
    al = a & T.uint32(0xFFFF)
    mh, ml = m >> 16, m & 0xFFFF
    p0 = al * T.uint32(ml)
    p1 = ah * T.uint32(ml)
    p2 = al * T.uint32(mh)
    p3 = ah * T.uint32(mh)
    t = (p1 & T.uint32(0xFFFF)) + (p2 & T.uint32(0xFFFF)) + (p0 >> T.uint32(16))
    lo = (p0 & T.uint32(0xFFFF)) | ((t & T.uint32(0xFFFF)) << T.uint32(16))
    hi = p3 + (p1 >> T.uint32(16)) + (p2 >> T.uint32(16)) + (t >> T.uint32(16))
    return hi, lo


def _residual(draft, target, at_t, at_d, n_s, num_draft, vocab, base, i):
    """``max(0, target[at_t + base + i] - (n < N ? draft[at_d + base + i] : 0))``.

    The tail tile and the tail of the vocabulary are zero-filled by predicate
    (sum-neutral).  The index stays the plain affine ``base + i``: the backend
    guards every GM load with ``0 <= addr && addr < total`` (0 when the guard
    fails), so an out-of-row lane reads nothing.
    """
    i0 = base + i
    return T.if_then_else(
        i0 < vocab,
        T.max(
            target[at_t[0] + i0]
            - T.if_then_else(n_s[0] < num_draft, draft[at_d[0] + i0], T.float32(0)),
            T.float32(0),
        ),
        T.float32(0),
    )


def _plan(batch, vocab):
    """Compile-time window: a whole lane-quantised row, one row a core."""
    tile = min(_TILE_CAP, -(-vocab // _THREADS) * _THREADS)
    return tile, _THREADS, min(batch, _MAX_BLOCKS)


def _build(batch, num_draft, vocab, tile, threads, blocks, knobs):
    fold_mode = knobs["fold"]
    recomp_mode = knobs["recomp"]
    walk_mode = knobs["walk"]
    resid_mode = knobs.get("resid", "select")
    chain_mode = knobs.get("chain", "shared")

    ntile = -(-vocab // tile)
    per_core = -(-batch // blocks)
    per_lane = tile // threads
    ndraw = num_draft + 1
    rows = batch * ndraw
    full_tiles = vocab // tile  # tiles wholly inside the row
    tail = vocab - full_tiles * tile  # elements of the last tile, 0 when exact
    peel = fold_mode != "select"  # the peeled fold spelling ("hoist"/"mul")

    @T.prim_func
    def main(
        draft_probs: T.Tensor((batch * num_draft * vocab,), _F32),
        draft_token_ids: T.Tensor((batch * num_draft,), "int32"),
        target_probs: T.Tensor((rows * vocab,), _F32),
        seed_w: T.Tensor((2,), "int32"),
        offset_w: T.Tensor((2,), "int32"),
        output_token_ids: T.Tensor((rows,), "int32"),
        num_accepted: T.Tensor((batch,), "int32"),
    ) -> None:
        with T.Kernel(blocks) as bx:
            r_ub = T.alloc_shared((tile,), _F32)
            lane_sum = T.alloc_shared((threads,), _F32)
            tile_sum = T.alloc_shared((ntile,), _F32)
            ids = T.alloc_shared((num_draft,), "int32")
            n_s = T.alloc_shared((1,), "int32")  # accepted prefix length
            at_t = T.alloc_shared((1,), "int32")  # target row base
            at_d = T.alloc_shared((1,), "int32")  # draft row base
            u_tok = T.alloc_shared((1,), _F32)  # uniform of draw N
            tok = T.alloc_shared((1,), "int32")  # drawn token
            cw = T.alloc_shared((4,), "uint32")  # Philox counter words
            kw = T.alloc_shared((2,), "uint32")  # Philox key words (Weyl-stepped)
            bw = T.alloc_shared((2,), "uint32")  # Philox round scratch
            point = T.alloc_shared((1,), _F32)  # u_N * total
            cum = T.alloc_shared((1,), _F32)  # running inclusive prefix
            left = T.alloc_shared((1,), _F32)  # point minus the level's prefix
            rem = T.alloc_shared((1,), _F32)  # point minus the chosen lane's prefix
            pbase = T.alloc_shared((1,), "int32")  # first element of the chosen lane
            pick = T.alloc_shared((1,), "int32")  # crossing tile
            lpick = T.alloc_shared((1,), "int32")  # crossing lane
            hit = T.alloc_shared((1,), "int32")  # a level crossed
            lastt = T.alloc_shared((1,), "int32")  # last tile of positive mass
            zfac = T.alloc_shared((1,), _F32)  # 1.0 while the draft row is read


            for rr in T.serial(per_core):
                line = bx + rr * blocks
                if line < batch:
                    # One VF region a row: the reducers and the lane domain are
                    # opened and closed inside the row.
                    with T.SimtVF(threads=threads):
                        # One reducer carries one epoch, so the peeled full-tile
                        # loop and the peeled tail tile get one allocation each.
                        if (not peel) or (full_tiles > 0):
                            can = T.alloc_reducer((1,), _F32, op="sum")
                            s = T.alloc_fragment((1,), _F32)
                        if peel and (tail > 0):
                            cant = T.alloc_reducer((1,), _F32, op="sum")
                            st = T.alloc_fragment((1,), _F32)
                        if walk_mode == "tile":
                            mxr = T.alloc_reducer((1,), _F32, op="max")
                            m = T.alloc_fragment((1,), _F32)
                        # Register-resident copies: the fold reads these once an
                        # element, and a UB scalar load in the innermost loop
                        # costs an extra op an element.
                        at_tl = T.alloc_local((1,), "int32")
                        at_dl = T.alloc_local((1,), "int32")
                        zl = T.alloc_local((1,), _F32)
                        nl = T.alloc_local((1,), "int32")
                        if chain_mode == "local":
                            cwl = T.alloc_local((4,), "uint32")
                            kwl = T.alloc_local((2,), "uint32")
                            bwl = T.alloc_local((2,), "uint32")
                            u_tokl = T.alloc_local((1,), _F32)

                        # ---- stage A: chain test (scalar; lanes run in lockstep) ----
                        if chain_mode == "local":
                            # The Philox state lives in registers: as UB scalars
                            # every write is a shared-memory store and the backend
                            # inserts a barrier after it (about fifty a row), which
                            # dominates this stage.
                            for j in T.Parallel(num_draft):
                                ids[j] = draft_token_ids[line * num_draft + j]
                            nl[0] = num_draft
                            for j in T.serial(ndraw):
                                cwl[0] = T.cast(j, "uint32")
                                cwl[1] = T.cast(line, "uint32")
                                cwl[2] = T.reinterpret(offset_w[0], "uint32")
                                cwl[3] = T.reinterpret(offset_w[1], "uint32")
                                kwl[0] = T.reinterpret(seed_w[0], "uint32")
                                kwl[1] = T.reinterpret(seed_w[1], "uint32")
                                for step in T.serial(_ROUNDS):
                                    if step > 0:
                                        kwl[0] = kwl[0] + T.uint32(_W0)
                                        kwl[1] = kwl[1] + T.uint32(_W1)
                                    hi0, lo0 = _mulhilo(cwl[0], _M0)
                                    hi1, lo1 = _mulhilo(cwl[2], _M1)
                                    bwl[0] = hi1 ^ cwl[1] ^ kwl[0]
                                    bwl[1] = hi0 ^ cwl[3] ^ kwl[1]
                                    cwl[1] = lo1
                                    cwl[3] = lo0
                                    cwl[0] = bwl[0]
                                    cwl[2] = bwl[1]
                                if j == num_draft:
                                    u_tokl[0] = T.cast(
                                        cwl[0] >> T.uint32(32 - _UNIFORM_BITS), _F32
                                    ) * T.float32(_UNIFORM_SCALE)
                                else:
                                    if (
                                        T.cast(cwl[0] >> T.uint32(32 - _UNIFORM_BITS), _F32)
                                        * T.float32(_UNIFORM_SCALE)
                                        * draft_probs[(line * num_draft + j) * vocab + ids[j]]
                                        >= target_probs[(line * ndraw + j) * vocab + ids[j]]
                                    ):
                                        nl[0] = T.min(nl[0], j)
                            at_tl[0] = (line * ndraw + nl[0]) * vocab
                            at_dl[0] = (line * num_draft + T.min(nl[0], num_draft - 1)) * vocab
                            zl[0] = T.if_then_else(
                                nl[0] < num_draft, T.float32(1.0), T.float32(0.0)
                            )
                            at_t[0] = at_tl[0]
                            at_d[0] = at_dl[0]
                            zfac[0] = zl[0]
                            n_s[0] = nl[0]
                            u_tok[0] = u_tokl[0]
                        else:
                            for j in T.Parallel(num_draft):
                                ids[j] = draft_token_ids[line * num_draft + j]
                            n_s[0] = num_draft
                            for j in T.serial(ndraw):
                                kw[0] = T.reinterpret(seed_w[0], "uint32")
                                kw[1] = T.reinterpret(seed_w[1], "uint32")
                                cw[0] = T.cast(j, "uint32")
                                cw[1] = T.cast(line, "uint32")
                                cw[2] = T.reinterpret(offset_w[0], "uint32")
                                cw[3] = T.reinterpret(offset_w[1], "uint32")
                                for step in T.serial(_ROUNDS):
                                    if step > 0:
                                        kw[0] = kw[0] + T.uint32(_W0)
                                        kw[1] = kw[1] + T.uint32(_W1)
                                    hi0, lo0 = _mulhilo(cw[0], _M0)
                                    hi1, lo1 = _mulhilo(cw[2], _M1)
                                    bw[0] = hi1 ^ cw[1] ^ kw[0]
                                    bw[1] = hi0 ^ cw[3] ^ kw[1]
                                    cw[1] = lo1
                                    cw[3] = lo0
                                    cw[0] = bw[0]
                                    cw[2] = bw[1]
                                if j == num_draft:
                                    u_tok[0] = T.cast(
                                        cw[0] >> T.uint32(32 - _UNIFORM_BITS), _F32
                                    ) * T.float32(_UNIFORM_SCALE)
                                else:
                                    if (
                                        T.cast(cw[0] >> T.uint32(32 - _UNIFORM_BITS), _F32)
                                        * T.float32(_UNIFORM_SCALE)
                                        * draft_probs[(line * num_draft + j) * vocab + ids[j]]
                                        >= target_probs[(line * ndraw + j) * vocab + ids[j]]
                                    ):
                                        n_s[0] = T.min(n_s[0], j)
                            at_t[0] = (line * ndraw + n_s[0]) * vocab
                            at_d[0] = (line * num_draft + T.min(n_s[0], num_draft - 1)) * vocab
                            zfac[0] = T.if_then_else(
                                n_s[0] < num_draft, T.float32(1.0), T.float32(0.0)
                            )

                        # The fold reads the row bases, the draft mask and the
                        # accepted length once an element: keep them in registers.
                        if chain_mode != "local":
                            at_tl[0] = at_t[0]
                            at_dl[0] = at_d[0]
                            zl[0] = zfac[0]
                            nl[0] = n_s[0]

                        # ---- stage B: fold the residual row, one sum a tile ----
                        if peel:
                            if full_tiles > 0:
                                for t in T.serial(full_tiles):
                                    T.reducer_init(can)
                                    if fold_mode == "hoist":
                                        # One branch a tile decides whether the
                                        # draft row is read; a select per element
                                        # compiles to a branch inside the
                                        # unrolled element loop and costs 3x the
                                        # whole fold.  Both arms add the same
                                        # values in the same order, so the tile
                                        # sums are unchanged bit for bit.
                                        if nl[0] < num_draft:
                                            for i in T.Parallel(tile):
                                                T.reducer_update(
                                                    can[0],
                                                    T.max(
                                                        target_probs[at_tl[0] + t * tile + i]
                                                        - draft_probs[at_dl[0] + t * tile + i],
                                                        T.float32(0),
                                                    ),
                                                )
                                        else:
                                            for i in T.Parallel(tile):
                                                T.reducer_update(
                                                    can[0],
                                                    T.max(
                                                        target_probs[at_tl[0] + t * tile + i],
                                                        T.float32(0),
                                                    ),
                                                )
                                    else:  # "mul": one multiply instead of a select
                                        for i in T.Parallel(tile):
                                            T.reducer_update(
                                                can[0],
                                                T.max(
                                                    target_probs[at_tl[0] + t * tile + i]
                                                    - zl[0]
                                                    * draft_probs[at_dl[0] + t * tile + i],
                                                    T.float32(0),
                                                ),
                                            )
                                    T.finalize_reducer(can, s)
                                    if T.get_thread_binding() == 0:
                                        tile_sum[t] = s[0]
                            if tail > 0:
                                # The tail tile is the only one with a per-element
                                # in-row predicate; zero fill leaves the sums
                                # unchanged.
                                T.reducer_init(cant)
                                for i in T.Parallel(tile):
                                    T.reducer_update(
                                        cant[0],
                                        T.if_then_else(
                                            full_tiles * tile + i < vocab,
                                            T.max(
                                                target_probs[
                                                    at_tl[0] + full_tiles * tile + i
                                                ]
                                                - zl[0]
                                                * draft_probs[
                                                    at_dl[0] + full_tiles * tile + i
                                                ],
                                                T.float32(0),
                                            ),
                                            T.float32(0),
                                        ),
                                    )
                                T.finalize_reducer(cant, st)
                                if T.get_thread_binding() == 0:
                                    tile_sum[ntile - 1] = st[0]
                        else:
                            # The Stage 3 spelling: one select per element.
                            for t in T.serial(ntile):
                                T.reducer_init(can)
                                for i in T.Parallel(tile):
                                    T.reducer_update(
                                        can[0],
                                        _residual(
                                            draft_probs, target_probs, at_t, at_d, n_s,
                                            num_draft, vocab, t * tile, i,
                                        ),
                                    )
                                T.finalize_reducer(can, s)
                                if T.get_thread_binding() == 0:
                                    tile_sum[t] = s[0]

                        # ---- stage C: locate the drawn token ----
                        point[0] = T.float32(0)
                        for t in T.serial(ntile):
                            point[0] = point[0] + tile_sum[t]
                        point[0] = u_tok[0] * point[0]
                        lastt[0] = 0
                        for t in T.serial(ntile):
                            if tile_sum[t] > T.float32(0):
                                lastt[0] = t
                        pick[0] = ntile - 1
                        left[0] = point[0]
                        cum[0] = T.float32(0)
                        hit[0] = 0
                        for t in T.serial(ntile):
                            if hit[0] == 0:
                                if cum[0] + tile_sum[t] > point[0]:
                                    pick[0] = t
                                    hit[0] = 1
                                    left[0] = point[0] - cum[0]
                                cum[0] = cum[0] + tile_sum[t]
                        if hit[0] == 0:
                            # The point reached the whole row's mass: no tile
                            # crosses.  Scan the last tile of positive mass and
                            # let the walk fail so the fallback answers.
                            pick[0] = lastt[0]
                            left[0] = cum[0] + T.float32(1.0)

                        # Recompute the crossing tile: the tile index stays a
                        # serial loop variable (a single pass at a runtime base
                        # aborts the device task on some vocabularies -- repro/
                        # TRAP-950-dynamic-base-tile-pass.py).
                        if recomp_mode == "branch":
                            for t in T.serial(ntile):
                                if t == pick[0]:
                                    for i in T.Parallel(tile):
                                        r_ub[i] = _residual(
                                            draft_probs, target_probs, at_t, at_d, n_s,
                                            num_draft, vocab, t * tile, i,
                                        )
                        else:  # "flat": one pass at the runtime base
                            for i in T.Parallel(tile):
                                r_ub[i] = _residual(
                                    draft_probs, target_probs, at_t, at_d, n_s,
                                    num_draft, vocab, pick[0] * tile, i,
                                )

                        if walk_mode == "tile":
                            # Stage 3: the fallback index arrives from a reducer
                            # over the whole crossing tile.
                            T.reducer_init(mxr)
                            for i in T.Parallel(tile):
                                T.reducer_update(
                                    mxr[0],
                                    T.if_then_else(
                                        r_ub[i] > T.float32(0),
                                        T.cast(pick[0] * tile + i, _F32),
                                        T.float32(-1),
                                    ),
                                )
                            T.finalize_reducer(mxr, m)

                        for l in T.Parallel(threads):
                            part = T.alloc_local((1,), _F32)
                            part[0] = T.float32(0)
                            for k in T.serial(per_lane):
                                part[0] = part[0] + r_ub[l * per_lane + k]
                            lane_sum[l] = part[0]

                        lpick[0] = threads - 1
                        cum[0] = T.float32(0)
                        rem[0] = left[0]
                        hit[0] = 0
                        for l in T.serial(threads):
                            if hit[0] == 0:
                                if cum[0] + lane_sum[l] > left[0]:
                                    lpick[0] = l
                                    rem[0] = left[0] - cum[0]
                                    hit[0] = 1
                                cum[0] = cum[0] + lane_sum[l]

                        # Element walk of the chosen lane: the first positive
                        # element whose inclusive prefix passes the remaining
                        # point (a zero weight never crosses, and the tolerance
                        # rule is ``prefix > point``, i.e. strict).
                        pbase[0] = lpick[0] * per_lane
                        tok[0] = -1
                        for k in T.serial(per_lane):
                            if (tok[0] < 0) & (r_ub[pbase[0] + k] > rem[0]):
                                tok[0] = pick[0] * tile + pbase[0] + k
                            rem[0] = rem[0] - r_ub[pbase[0] + k]
                        if tok[0] < 0:
                            if walk_mode == "lane":
                                # No lane carried the point (rounding window): the
                                # answer is the tile's last positive element, which
                                # is in the last lane that carries mass.
                                for l in T.serial(threads):
                                    if lane_sum[threads - 1 - l] > T.float32(0):
                                        pbase[0] = (threads - 1 - l) * per_lane
                                        tok[0] = pick[0] * tile + pbase[0]
                                        for k in T.serial(per_lane):
                                            if r_ub[pbase[0] + k] > T.float32(0):
                                                tok[0] = pick[0] * tile + pbase[0] + k
                            else:
                                tok[0] = T.cast(m[0], "int32")
                            if tok[0] < 0:  # a zero-mass row: the reference's clamp
                                tok[0] = vocab - 1
                        if tok[0] >= vocab:  # the tail of the last tile is zero-filled
                            tok[0] = vocab - 1

                        # ---- stage D: emit ----
                        if T.get_thread_binding() == 0:
                            for j in T.serial(ndraw):
                                output_token_ids[line * ndraw + j] = T.if_then_else(
                                    j < nl[0],
                                    draft_token_ids[line * num_draft + T.min(j, num_draft - 1)],
                                    T.if_then_else(j == nl[0], tok[0], T.int32(-1)),
                                )
                            num_accepted[line] = nl[0]

    return main


@functools.lru_cache(maxsize=32)
def _chain_speculative_sampling_kernel(batch, num_draft, vocab):
    """Frozen factory: ``(batch, num_draft, vocab)`` -> compiled callable.

    Outputs are allocated by the caller and passed in (``out_idx=[]``), as the
    source operator's wrapper does.
    """
    tile, threads, blocks = _plan(batch, vocab)
    return tilelang.compile(
        _build(batch, num_draft, vocab, tile, threads, blocks, _KNOBS),
        target="ascend",
        out_idx=[],
    )


# ---------------------------------------------------------------------------
# 4. Launch + graded comparison
# ---------------------------------------------------------------------------
_RESULTS = []  # (level, name, ok, detail)
_MAX_DIFF = [0.0]  # max |num_accepted - reference| -- the bit-exact contract
_TOKEN_DIFF = [0]  # drawn-token disagreements with the reference (tolerance contract)
_ELAPSED = [0.0]


def _record(level, name, ok, detail=""):
    _RESULTS.append((level, name, ok, detail))
    print(f"[{level}][{'PASS' if ok else 'FAIL'}] {name}"
          f"{(' -- ' + detail) if detail and not ok else ''}", flush=True)


def _finish(level, strict):
    rows = [r for r in _RESULTS if r[0] == level]
    failed = [r for r in rows if not r[2]]
    print(f"[{level}] {len(rows) - len(failed)}/{len(rows)} cases passed, "
          f"num_accepted max_diff={_MAX_DIFF[0]:.6g}, "
          f"drawn-token mismatches={_TOKEN_DIFF[0]}", flush=True)
    if failed and strict:
        raise AssertionError(f"[{level}] {len(failed)} case(s) failed")


def _validate(draft_probs, draft_token_ids, target_probs, seed, offset):
    """The wrapper's contract check (the kernel never sees a rejected call)."""
    batch, num_draft, vocab = draft_probs.shape
    if draft_probs.dtype is not torch.float32 or target_probs.dtype is not torch.float32:
        raise ValueError(
            f"dtype: probabilities must be float32, got "
            f"{draft_probs.dtype} / {target_probs.dtype}"
        )
    if tuple(target_probs.shape) != (batch, num_draft + 1, vocab):
        raise ValueError(
            f"shape: target_probs {tuple(target_probs.shape)} is not "
            f"{(batch, num_draft + 1, vocab)}"
        )
    if draft_token_ids.dtype is not torch.int32:
        raise ValueError(f"dtype: draft_token_ids must be int32, got {draft_token_ids.dtype}")
    if seed.dtype is not torch.int64 or offset.dtype is not torch.int64:
        raise ValueError("dtype: seed and offset must be int64")
    if batch * (num_draft + 1) * vocab > 2**31 - 1:
        raise ValueError("shape: the flat index of a probability row does not fit int32")


def _launch(draft_probs, draft_token_ids, target_probs, seed, offset, out_tokens, out_acc):
    """One call with the frozen wrapper's contract (flat views, outputs passed in)."""
    batch, num_draft, vocab = draft_probs.shape
    _validate(draft_probs, draft_token_ids, target_probs, seed, offset)
    fn = _chain_speculative_sampling_kernel(batch, num_draft, vocab)
    fn(
        draft_probs.contiguous().view(-1),
        draft_token_ids.contiguous().view(-1),
        target_probs.contiguous().view(-1),
        seed.view(torch.int32).reshape(-1),
        offset.view(torch.int32).reshape(-1),
        out_tokens.view(-1),
        out_acc.view(-1),
    )


def _run(draft_probs, draft_token_ids, target_probs, seed, offset):
    """One call into fresh outputs (device tensors in, device tensors out)."""
    batch, num_draft, _ = draft_probs.shape
    out_tokens = torch.empty(batch, num_draft + 1, dtype=torch.int32, device=draft_probs.device)
    out_acc = torch.empty(batch, dtype=torch.int32, device=draft_probs.device)
    _launch(draft_probs, draft_token_ids, target_probs, seed, offset, out_tokens, out_acc)
    return out_tokens, out_acc


def _make_inputs(batch, num_draft, vocab, gen_seed, mode="random"):
    """Deterministic inputs on the host (CPU).

    The manifest workload generator: draft and target rows are the softmax of
    standard normal logits, and each draft id is drawn from its own draft row (so
    every accepted draft is one its target row can draw).  ``mode`` forces a
    control-flow path: all accepted (``target == draft``), rejected at the first
    position, or rejected in the middle of the chain.
    """
    gen = torch.Generator().manual_seed(gen_seed)
    draft = torch.randn(batch, num_draft, vocab, generator=gen, dtype=torch.float32).softmax(-1)
    target = torch.randn(
        batch, num_draft + 1, vocab, generator=gen, dtype=torch.float32
    ).softmax(-1)
    ids = (
        torch.multinomial(draft.reshape(-1, vocab), 1, generator=gen)
        .view(batch, num_draft)
        .to(torch.int32)
    )
    if mode == "all_accepted":
        target[:, :num_draft] = draft
    elif mode == "reject_first":
        target[:, 0].scatter_(1, ids[:, :1].long(), 0.0)
    elif mode == "reject_mid":
        mid = num_draft // 2
        target[:, :mid] = draft[:, :mid]
        target[:, mid].scatter_(1, ids[:, mid : mid + 1].long(), 0.0)
    elif mode == "zero_residual":
        # target == draft everywhere and the first id's target probability is
        # zero: the residual row is all zeros -- the contract-external corner the
        # reference answers with its ``last_positive`` clamp (``vocab - 1``).
        target[:, :num_draft] = draft
        target[:, 0].scatter_(1, ids[:, :1].long(), 0.0)
    elif mode == "all_rejected":
        # Every acceptance test fails, so the fold reads the draft row on every
        # row (the branch the fold hoists).
        target[:, :num_draft] = draft
        target[:, :num_draft].scatter_(2, ids.long()[..., None], 0.0)
    elif mode != "random":
        raise ValueError(f"unknown input mode {mode!r}")
    seed = torch.tensor([SEED], dtype=torch.int64)
    offset = torch.tensor([OFFSET], dtype=torch.int64)
    return draft.contiguous(), ids.contiguous(), target.contiguous(), seed, offset


def _shifted_views(draft, target, device):
    """The same values one float32 word past a 16-byte boundary (``storage_offset=1``).

    The manifest fixture hands the op views like these, so the kernel must not
    assume the row base is 16-byte aligned.
    """
    out = []
    for x in (draft, target):
        host = torch.empty(x.numel() + 1, dtype=x.dtype)
        host[1:] = x.reshape(-1)
        view = host[1:].view(x.shape)
        assert view.storage_offset() == 1 and view.is_contiguous()
        dev = torch.empty(x.numel() + 1, dtype=x.dtype, device=device)
        dev[1:].copy_(view.reshape(-1))
        shifted = dev[1:].view(x.shape)
        assert shifted.storage_offset() == 1
        out.append(shifted)
    return out[0], out[1]


def run_case(
    level,
    name,
    batch,
    num_draft,
    vocab,
    gen_seed,
    mode="random",
    expect_n=None,
    repeat=1,
    shift=False,
    support=True,
):
    """Generate, run, and compare one shape.

    Asserts, in the contract's own order: ``num_accepted`` equals the reference's
    bit for bit, the emitted row is structurally the contract's, and a second
    call on the same input returns the same draws.
    """
    try:
        draft, ids, target, seed, offset = _make_inputs(batch, num_draft, vocab, gen_seed, mode)
        ref_tokens, ref_acc = golden_chain_speculative_sampling(draft, ids, target, seed, offset)
        device = "npu"
        if shift:
            dd, dt = _shifted_views(draft, target, device)
        else:
            dd, dt = draft.to(device), target.to(device)
        di, ds, do = ids.to(device), seed.to(device), offset.to(device)
        ref_hit = (ref_acc.long() == 0).sum().item() if batch else 0
        t0 = time.time()
        got_tokens, got_acc = _run(dd, di, dt, ds, do)
        torch.npu.synchronize()
        elapsed = (time.time() - t0) * 1e3
        _ELAPSED[0] += elapsed
        gt, ga = got_tokens.cpu(), got_acc.cpu()

        gap = int((ga.long() - ref_acc.long()).abs().max().item()) if batch else 0
        _MAX_DIFF[0] = max(_MAX_DIFF[0], float(gap))
        if gap:
            bad = (ga != ref_acc).nonzero().flatten()[:8].tolist()
            raise AssertionError(
                f"num_accepted differs at rows {bad} (max |diff| {gap}); "
                f"got {ga[bad].tolist()} vs ref {ref_acc[bad].tolist()}"
            )
        if expect_n is not None and not bool((ref_acc == expect_n).all()):
            raise AssertionError(
                f"the reference did not stop at {expect_n}: {ref_acc.unique().tolist()}"
            )
        if support:
            verify_chain_outputs(gt, ga, draft, ids, target)
        else:
            # Contract-external corner: no positive residual mass at all, so the
            # "drawn token carries weight" clause of the contract cannot hold.
            assert ((gt >= -1) & (gt < vocab)).all()
            assert bool((ga == expect_n).all()), f"expected n={expect_n}"

        rows = torch.arange(batch)
        drawn = gt[rows, ga.long()]
        ref_drawn = ref_tokens[rows, ref_acc.long()]
        _TOKEN_DIFF[0] += int((drawn != ref_drawn).sum())

        for _ in range(repeat - 1):
            again_tokens, again_acc = _run(dd, di, dt, ds, do)
            torch.npu.synchronize()
            if not (torch.equal(again_tokens.cpu(), gt) and torch.equal(again_acc.cpu(), ga)):
                raise AssertionError("a second call on the same input differs (not reproducible)")

        detail = (
            f"n=[{int(ga.min())},{int(ga.max())}] stops@0={ref_hit}/{batch} "
            f"drawn!=ref={int((drawn != ref_drawn).sum())} {elapsed:.2f} ms"
        )
        print(
            f"       B={batch} N={num_draft} V={vocab} mode={mode} shift={shift} "
            f"n_accepted[min,max]=[{int(ga.min())},{int(ga.max())}] "
            f"{detail}",
            flush=True,
        )
    except AssertionError as exc:
        _record(level, name, False, str(exc)[:400])
        return False
    except (RuntimeError, ValueError) as exc:
        _record(level, name, False, f"{type(exc).__name__}: {str(exc)[:300]}")
        return False
    _record(level, name, True, detail)
    return True


def run_distribution_case(level, name, rows, num_draft, vocab, gen_seed):
    """``rows`` requests sharing one (draft, target) pair: the distribution probe."""
    try:
        gen = torch.Generator().manual_seed(gen_seed)
        draft = torch.rand(num_draft, vocab, generator=gen, dtype=torch.float32) ** 3
        draft /= draft.sum(-1, keepdim=True)
        target = torch.rand(num_draft + 1, vocab, generator=gen, dtype=torch.float32) ** 3
        target /= target.sum(-1, keepdim=True)
        ids = (
            torch.multinomial(draft, rows, replacement=True, generator=gen)
            .T.to(torch.int32)
            .contiguous()
        )
        seed = torch.tensor([SEED], dtype=torch.int64)
        offset = torch.tensor([OFFSET], dtype=torch.int64)
        ref_tokens, ref_acc = golden_chain_speculative_sampling(
            draft.expand(rows, num_draft, vocab).contiguous(), ids,
            target.expand(rows, num_draft + 1, vocab).contiguous(), seed, offset,
        )
        device = "npu"
        args = (
            draft.expand(rows, num_draft, vocab).contiguous().to(device),
            ids.to(device),
            target.expand(rows, num_draft + 1, vocab).contiguous().to(device),
            seed.to(device),
            offset.to(device),
        )
        t0 = time.time()
        got_tokens, got_acc = _run(*args)
        torch.npu.synchronize()
        elapsed = (time.time() - t0) * 1e3
        _ELAPSED[0] += elapsed
        gt, ga = got_tokens.cpu(), got_acc.cpu()
        gap = int((ga.long() - ref_acc.long()).abs().max().item())
        _MAX_DIFF[0] = max(_MAX_DIFF[0], float(gap))
        assert gap == 0, f"num_accepted differs on {int((ga != ref_acc).sum())} rows"
        rows_idx = torch.arange(rows)
        _TOKEN_DIFF[0] += int((gt[rows_idx, ga.long()] != ref_tokens[rows_idx, ref_acc.long()]).sum())
        assert_chain_distribution(gt, ga, ids, draft, target)
    except AssertionError as exc:
        _record(level, name, False, str(exc)[:400])
        return False
    _record(level, name, True, f"rows={rows} {elapsed:.1f} ms, structure+distribution pass")
    return True


# ---------------------------------------------------------------------------
# 5. Layered cases
# ---------------------------------------------------------------------------
def run_L0():
    """Gate: the manifest workloads, the control-flow paths, and reproducibility."""
    props = torch.npu.get_device_properties(torch.npu.current_device())
    print(f"[L0] device {props.name} cube={props.cube_core_num} "
          f"vector={props.vector_core_num} (plan folds onto {_MAX_BLOCKS})", flush=True)
    assert props.vector_core_num == _MAX_BLOCKS, (
        f"the plan's core count {_MAX_BLOCKS} is not this device's vector core count "
        f"{props.vector_core_num}"
    )
    run_case("L0", "L0-1 smoke (2,2,64)", 2, 2, 64, 11, repeat=2)
    run_case("L0", "L0-2 (4,3,64) n in 0..3 reachable", 4, 3, 64, 12, repeat=2)
    run_case("L0", "L0-3 all accepted (4,3,64)", 4, 3, 64, 13, mode="all_accepted", expect_n=3)
    run_case("L0", "L0-4 rejected at the first draft (4,3,64)", 4, 3, 64, 14,
             mode="reject_first", expect_n=0)
    run_case("L0", "L0-5 rejected mid chain (4,5,64)", 4, 5, 64, 15, mode="reject_mid", expect_n=2)
    run_case("L0", "L0-6 B=1 (1,2,64)", 1, 2, 64, 16, repeat=2)
    run_case("L0", "L0-7 llama-8b-b16-n4 (16,4,128256)", 16, 4, 128256, 21)
    run_case("L0", "L0-8 llama-8b-b64-n4 (64,4,128256)", 64, 4, 128256, 22)
    run_case("L0", "L0-9 ds-v3-mtp-n1 (64,1,129280)", 64, 1, 129280, 23)
    run_case("L0", "L0-10 ds-v3-mtp-n3 (64,3,129280)", 64, 3, 129280, 24)
    run_case("L0", "L0-11 qwen3-235b-eagle-n5 (32,5,151936)", 32, 5, 151936, 25)
    run_case("L0", "L0-12 llama-8b-b17-n8 (17,8,128256)", 17, 8, 128256, 26)
    run_case("L0", "L0-13 llama2-7b-b256-n1 (256,1,32000)", 256, 1, 32000, 27)
    _finish("L0", strict=True)


def run_L1():
    """Representative shapes: odd V, many tiles, per-core rows, a long chain."""
    run_case("L1", "L1-1 V not a multiple of 4 (8,3,4099)", 8, 3, 4099, 31, repeat=2)
    run_case("L1", "L1-2 19 tiles (2,3,151936)", 2, 3, 151936, 32)
    run_case("L1", "L1-3 per-core 6 rows (300,3,32000)", 300, 3, 32000, 33)
    run_case("L1", "L1-4 N=600 scalar chain (4,600,64)", 4, 600, 64, 34)
    run_case("L1", "L1-5 V exactly one tile (16,2,8192)", 16, 2, 8192, 35)
    run_case("L1", "L1-6 V one past a tile (16,2,8193)", 16, 2, 8193, 36)
    run_distribution_case("L1", "L1-7 distribution probe (65536,2,16)", 65536, 2, 16, 37)
    _finish("L1", strict=True)


def run_L2():
    """Stress: informative, non-blocking."""
    run_case("L2", "L2-1 B=256 with N=3 (256,3,32000)", 256, 3, 32000, 41)
    run_case("L2", "L2-2 tiny batch, big V (1,4,151936)", 1, 4, 151936, 42)
    run_case("L2", "L2-3 blocks full at B=56 (56,5,4099)", 56, 5, 4099, 43)
    run_case("L2", "L2-4 all accepted, big V (2,4,32000)", 2, 4, 32000, 44,
             mode="all_accepted", expect_n=4)
    run_case("L2", "L2-5 first reject, big V (2,4,32000)", 2, 4, 32000, 45,
             mode="reject_first", expect_n=0)
    _finish("L2", strict=False)


def run_boundary():
    """Boundaries: alignment, a one-token vocabulary, last-column ids, dtype guard."""
    run_case("boundary", "B-1 storage_offset=1 (2,2,64)", 2, 2, 64, 51, shift=True, repeat=2)
    run_case("boundary", "B-2 storage_offset=1, odd V (3,3,4099)", 3, 3, 4099, 52, shift=True)
    try:
        # The same logical call, once aligned and once one word past a 16-byte
        # boundary: both outputs (and both accepted lengths) agree bit for bit.
        draft, ids, target, seed, offset = _make_inputs(2, 2, 64, 56)
        ref_tokens, ref_acc = golden_chain_speculative_sampling(draft, ids, target, seed, offset)
        dev = "npu"
        di, ds, do = ids.to(dev), seed.to(dev), offset.to(dev)
        aligned = _run(draft.to(dev), di, target.to(dev), ds, do)
        sd, st = _shifted_views(draft, target, dev)
        shifted = _run(sd, di, st, ds, do)
        torch.npu.synchronize()
        for got, name in ((aligned, "aligned"), (shifted, "shifted")):
            gap = int((got[1].cpu().long() - ref_acc.long()).abs().max().item())
            _MAX_DIFF[0] = max(_MAX_DIFF[0], float(gap))
            assert gap == 0, f"{name}: num_accepted differs from the reference"
        assert torch.equal(aligned[0].cpu(), shifted[0].cpu()), "tokens depend on alignment"
        assert torch.equal(aligned[1].cpu(), shifted[1].cpu()), "accepted lengths differ"
        rows = torch.arange(2)
        _TOKEN_DIFF[0] += int(
            (aligned[0].cpu()[rows, aligned[1].cpu().long()]
             != ref_tokens[rows, ref_acc.long()]).sum()
        )
    except (AssertionError, ValueError, RuntimeError) as exc:
        _record("boundary", "B-2b aligned vs storage_offset=1, same call", False,
                f"{type(exc).__name__}: {str(exc)[:300]}")
    else:
        _record("boundary", "B-2b aligned vs storage_offset=1, same call", True, "bit exact")
    run_case("boundary", "B-3 one-token vocabulary (4,2,1)", 4, 2, 1, 53,
             support=False, expect_n=2)
    run_case("boundary", "B-3b zero residual row (4,3,64)", 4, 3, 64, 55,
             mode="zero_residual", expect_n=0, support=False)
    try:
        # The last column of every row: the row-fold and locate read the tail.
        draft, ids, target, seed, offset = _make_inputs(2, 3, 64, 54)
        ids = torch.full((2, 3), 63, dtype=torch.int32)
        ref_tokens, ref_acc = golden_chain_speculative_sampling(draft, ids, target, seed, offset)
        args = (draft.to("npu"), ids.to("npu"), target.to("npu"), seed.to("npu"), offset.to("npu"))
        got_tokens, got_acc = _run(*args)
        torch.npu.synchronize()
        gt, ga = got_tokens.cpu(), got_acc.cpu()
        gap = int((ga.long() - ref_acc.long()).abs().max().item())
        _MAX_DIFF[0] = max(_MAX_DIFF[0], float(gap))
        assert gap == 0, "num_accepted differs with the draft id on the last column"
        verify_chain_outputs(gt, ga, draft, ids, target)
        rows = torch.arange(2)
        _TOKEN_DIFF[0] += int((gt[rows, ga.long()] != ref_tokens[rows, ref_acc.long()]).sum())
    except AssertionError as exc:
        _record("boundary", "B-4 draft id on the last column (2,3,64)", False, str(exc)[:400])
    else:
        _record("boundary", "B-4 draft id on the last column (2,3,64)", True, "pass")
    try:
        bad = torch.randint(0, 8, (2, 2, 64)).to("npu")
        _run(bad, torch.zeros(2, 2, dtype=torch.int32, device="npu"),
             torch.rand(2, 3, 64, device="npu"),
             torch.tensor([SEED], dtype=torch.int64, device="npu"),
             torch.tensor([OFFSET], dtype=torch.int64, device="npu"))
    except ValueError as exc:
        assert "float32" in str(exc), str(exc)
        _record("boundary", "B-5 int dtype rejected by the wrapper", True, "ValueError")
    except AssertionError as exc:
        _record("boundary", "B-5 int dtype rejected by the wrapper", False, str(exc)[:300])
    else:
        _record("boundary", "B-5 int dtype rejected by the wrapper", False,
                "an int32 probability input was accepted")
    _finish("boundary", strict=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0", choices=["L0", "L1", "L2", "boundary", "all", "none"])
    parser.add_argument("--case", help="one benchmark workload label: run it once for profiling")
    parser.add_argument("--knobs", help="override stage spellings, e.g. fold=hoist,walk=lane")
    parser.add_argument("--timeit", type=int, default=0, help="time the --case call this many times")
    args, _ = parser.parse_known_args()
    if args.knobs:
        for item in args.knobs.split(","):
            key, value = item.split("=")
            _KNOBS[key] = value
    if args.case:
        return _run_case_once(args.case, args.timeit)
    levels = {"L0": run_L0, "L1": run_L1, "L2": run_L2, "boundary": run_boundary}
    chosen = levels.values() if args.level == "all" else [levels[args.level]]
    for run in chosen:
        run()
    print(f"[PRECISION_PASS] num_accepted_max_diff={_MAX_DIFF[0]:.6g} "
          f"drawn_token_mismatches={_TOKEN_DIFF[0]} device_ms={_ELAPSED[0]:.2f}")


def _workload_shape(label):
    """``(batch, num_draft, vocab)`` of one benchmark workload, read from the manifest.

    The benchmark parametrises ``(batch, num_draft, vocab)`` out of
    ``load_workloads("ChainSpeculativeSamplingFwdOp")``; reading the same rows
    here keeps a profiled case identical to the benchmark's.
    """
    import sys

    root = _pathlib.Path(__file__).resolve().parents[2] / "TileOPs"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from tileops.manifest import load_workloads

    for workload in load_workloads("ChainSpeculativeSamplingFwdOp"):
        if workload.get("label") == label:
            batch, num_draft, vocab = tuple(workload["draft_probs_shape"])
            return batch, num_draft, vocab
    raise ValueError(f"no benchmark workload labelled {label!r}")


def _run_case_once(label, timeit_iters=0):
    """One workload, one steady call: the entry msprof profiles."""
    batch, num_draft, vocab = _workload_shape(label)
    mode = "random"
    draft, ids, target, seed, offset = _make_inputs(batch, num_draft, vocab, 24, mode)
    dev = "npu"
    dd, di, dt = draft.to(dev), ids.to(dev), target.to(dev)
    ds, do = seed.to(dev), offset.to(dev)
    out_tokens, out_acc = _run(dd, di, dt, ds, do)
    torch.npu.synchronize()
    if timeit_iters:
        for _ in range(3):
            _run(dd, di, dt, ds, do)
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(timeit_iters):
            _run(dd, di, dt, ds, do)
        torch.npu.synchronize()
        us = (time.perf_counter() - t0) / timeit_iters * 1e6
        print(f"case={label} B={batch} N={num_draft} V={vocab} knobs={_KNOBS} "
              f"wall={us:.1f} us", flush=True)
        return 0
    print(f"case={label} B={batch} N={num_draft} V={vocab} knobs={_KNOBS}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
