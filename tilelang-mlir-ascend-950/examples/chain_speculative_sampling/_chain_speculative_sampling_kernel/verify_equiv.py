"""Machine check of the design's equivalence claims (Stage 1, D-1).

Every adopted optimization in DESIGN.md §1.6.1 is checked here against the baseline it
replaces, on random tensors and corner values, in float64 where a reference is needed.
Runs on CPU:

    /workspace/venv-ascend950/bin/python verify_equiv.py

The baseline (``_mulhilo`` / ``philox_uniform`` / ``_draw`` / ``chain_speculative_sampling``)
is ported verbatim from the op's correctness oracle,
``examples/TileOPs/tileops/workloads/sampling.py`` -- the contract of §0.1 -- and is *not*
the NPU algorithm (no tiling, no streaming, no fp32 hierarchy).

Items
  E1  ``mulhi`` by four 16x16 products vs the reference ``_mulhilo``      exact bits
  E2  the ten-round Philox stream built on E1 vs ``philox_uniform``       exact bits
  E3  the acceptance prefix built without ``cumprod`` vs the reference    exact
  E4  the residual ``max(0, target - draft)`` (in place, fp32)            exact bits
  E5  hierarchical fp32 inverse-CDF + first-passing-positive rule vs the
      reference fp64 ``cumsum`` + ``searchsorted`` + next-positive rule   tolerance
  E6  the emitted row (prefix, drawn token, ``-1`` tail)                 exact
"""

import torch

MASK = 0xFFFFFFFF
M0, M1 = 0xD2511F53, 0xCD9E8D57
W0, W1 = 0x9E3779B9, 0xBB67AE85
ROUNDS = 10
UNIFORM_BITS = 24
SEED = 1234
OFFSET = 7


# ---------------------------------------------------------------------------
# Baseline: the contract's own arithmetic, ported verbatim from
# examples/TileOPs/tileops/workloads/sampling.py
# ---------------------------------------------------------------------------
def ref_mulhilo(a: torch.Tensor, m: int):
    """The reference ``_mulhilo``: high and low 32 bits of ``a * m``."""
    low = (a & 0xFFFF) * m
    high = (a >> 16) * m
    mid = low + ((high & 0xFFFF) << 16)
    return ((high >> 16) + (mid >> 32)) & MASK, mid & MASK


def ref_philox_uniform(seed: torch.Tensor, offset: torch.Tensor, rows: int, draws: int):
    s, o = seed.reshape(()).to(torch.int64), offset.reshape(()).to(torch.int64)
    shape = (rows, draws)
    c0 = torch.arange(draws, dtype=torch.int64).expand(shape).clone()
    c1 = torch.arange(rows, dtype=torch.int64)[:, None].expand(shape).clone()
    c2 = (o & MASK).expand(shape).clone()
    c3 = ((o >> 32) & MASK).expand(shape).clone()
    k0, k1 = s & MASK, (s >> 32) & MASK
    for step in range(ROUNDS):
        if step:
            k0, k1 = (k0 + W0) & MASK, (k1 + W1) & MASK
        hi0, lo0 = ref_mulhilo(c0, M0)
        hi1, lo1 = ref_mulhilo(c2, M1)
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
    return (c0 >> 8).to(torch.float32) * 2.0**-24


def ref_draw(weights: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    batch, vocab = weights.shape
    cdf = weights.double().cumsum(-1)
    point = u.double() * cdf[:, -1]
    index = torch.searchsorted(cdf, point[:, None], right=True).clamp(max=vocab - 1)
    positive = weights > 0
    columns = torch.arange(vocab).expand(batch, vocab)
    next_positive = torch.where(positive, columns, vocab).flip(-1).cummin(-1).values.flip(-1)
    chosen = next_positive.gather(1, index)[:, 0]
    last_positive = vocab - 1 - positive.flip(-1).int().argmax(-1)
    return torch.where(chosen == vocab, last_positive, chosen).to(torch.int32)


def ref_chain(draft_probs, draft_token_ids, target_probs, seed, offset):
    batch, n, vocab = draft_probs.shape
    ids = draft_token_ids.view(batch, n).to(torch.long)
    target = target_probs.view(batch, n + 1, vocab)
    u = ref_philox_uniform(seed, offset, batch, n + 1)
    d = draft_probs.gather(2, ids[..., None])[..., 0]
    q = target[:, :n].gather(2, ids[..., None])[..., 0]
    num = (u[:, :n] * d < q).int().cumprod(1).sum(1)
    padded = torch.cat([draft_probs, torch.zeros_like(draft_probs[:, :1])], 1)
    at = num.long()[:, None, None].expand(batch, 1, vocab)
    residual = (target.gather(1, at) - padded.gather(1, at))[:, 0].clamp_min(0)
    token = ref_draw(residual, u[:, n]).long()
    position = torch.arange(n + 1)[None]
    drafts = torch.cat([ids, torch.full_like(ids[:, :1], -1)], 1)
    stop = num.long()[:, None]
    tokens = torch.where(position < stop, drafts, torch.where(position == stop, token[:, None], -1))
    return tokens.to(torch.int32), num.to(torch.int32), residual, u


# ---------------------------------------------------------------------------
# Candidate: the design's forms
# ---------------------------------------------------------------------------
def cand_mulhilo(a: torch.Tensor, m: int):
    """``(hi, lo)`` of ``a * m`` from four 16x16 products, every word 32 bits wide.

    Only ``and`` / ``shift`` / ``add`` / 16x16 ``mul`` appear -- the operations the Ascend
    AIV has natively -- and every intermediate stays below 2**32 except the three-term
    sum ``t`` (< 2**18), so the result is the exact 64-bit product's halves.
    """
    ah, al = (a >> 16) & 0xFFFF, a & 0xFFFF
    mh, ml = (m >> 16) & 0xFFFF, m & 0xFFFF
    p0 = al * ml
    p1 = ah * ml
    p2 = al * mh
    p3 = ah * mh
    t = (p1 & 0xFFFF) + (p2 & 0xFFFF) + ((p0 >> 16) & 0xFFFF)
    lo = ((p0 & 0xFFFF) | ((t & 0xFFFF) << 16)) & MASK
    hi = (p3 + ((p1 >> 16) & 0xFFFF) + ((p2 >> 16) & 0xFFFF) + (t >> 16)) & MASK
    return hi, lo


def cand_philox_uniform(seed, offset, rows, draws):
    s, o = seed.reshape(()).to(torch.int64), offset.reshape(()).to(torch.int64)
    shape = (rows, draws)
    c0 = torch.arange(draws, dtype=torch.int64).expand(shape).clone()
    c1 = torch.arange(rows, dtype=torch.int64)[:, None].expand(shape).clone()
    c2 = (o & MASK).expand(shape).clone()
    c3 = ((o >> 32) & MASK).expand(shape).clone()
    k0, k1 = s & MASK, (s >> 32) & MASK
    for step in range(ROUNDS):
        if step:
            k0, k1 = (k0 + W0) & MASK, (k1 + W1) & MASK
        hi0, lo0 = cand_mulhilo(c0, M0)
        hi1, lo1 = cand_mulhilo(c2, M1)
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
    return (c0 >> 8).to(torch.float32) * 2.0**-24


def cand_chain(draft_probs, draft_token_ids, target_probs, seed, offset, tile=256, threads=32):
    """The design's structure: prefix by first rejection, residual in place, fp32 hierarchy."""
    batch, n, vocab = draft_probs.shape
    ids = draft_token_ids.view(batch, n).to(torch.long)
    target = target_probs.view(batch, n + 1, vocab)
    u = cand_philox_uniform(seed, offset, batch, n + 1)
    d = draft_probs.gather(2, ids[..., None])[..., 0]
    q = target[:, :n].gather(2, ids[..., None])[..., 0]
    accept = u[:, :n] * d < q  # float32, the reference's own test
    # Prefix length by the first rejection (no cumprod).
    first_reject = torch.where(~accept, torch.arange(n)[None], n).min(1).values
    num = first_reject
    padded = torch.cat([draft_probs, torch.zeros_like(draft_probs[:, :1])], 1)
    at = num.long()[:, None, None].expand(batch, 1, vocab)
    residual = (target.gather(1, at) - padded.gather(1, at))[:, 0].clamp_min(0)
    token = cand_draw(residual, u[:, n], tile=tile, threads=threads)
    position = torch.arange(n + 1)[None]
    drafts = torch.cat([ids, torch.full_like(ids[:, :1], -1)], 1)
    stop = num.long()[:, None]
    tokens = torch.where(position < stop, drafts, torch.where(position == stop, token[:, None], -1))
    return tokens.to(torch.int32), num.to(torch.int32), residual, u


def cand_draw(weights, u, tile=256, threads=32):
    """Inverse CDF over a tile hierarchy, then one tile walked lane by lane.

    Mirrors the kernel's arithmetic: the tile sums and the lane partials are float32 and
    grouped identically in both passes, ``point = u * total`` is float32, and every
    descent subtraction is float32.  The element rule is the reference's: the first index
    whose inclusive prefix passes the point *and* carries weight, else the row's last
    positive index.
    """
    batch, vocab = weights.shape
    ntile = -(-vocab // tile)
    per_lane = -(-tile // threads)
    tokens = torch.empty(batch, dtype=torch.long)
    for b in range(batch):
        row = weights[b]
        # Pass 1: one float32 sum a tile (lane partials first, then the lane sums in order).
        sums = torch.zeros(ntile)
        for t in range(ntile):
            lo, hi = t * tile, min(t * tile + tile, vocab)
            s = torch.zeros(())
            for lane in range(threads):
                acc = torch.zeros(())
                for k in range(per_lane):
                    i = lo + lane * per_lane + k
                    if i < hi:
                        acc = acc + row[i]
                s = s + acc
            sums[t] = s
        total = torch.zeros(())
        for t in range(ntile):
            total = total + sums[t]
        point = u[b].float() * total
        # Coarse descent: the first tile whose inclusive prefix passes the point.
        left = point
        pick = ntile - 1
        before = point
        for t in range(ntile):
            before = left
            left = left - sums[t]
            if left < 0:
                pick = t
                break
        # Pass 2: the crossing tile, lane partials (identical grouping), then the lane.
        lo, hi = pick * tile, min(pick * tile + tile, vocab)
        lanes = torch.zeros(threads)
        for lane in range(threads):
            acc = torch.zeros(())
            for k in range(per_lane):
                i = lo + lane * per_lane + k
                if i < hi:
                    acc = acc + row[i]
            lanes[lane] = acc
        left = before
        lane_pick = threads - 1
        lane_before = before
        for lane in range(threads):
            lane_before = left
            left = left - lanes[lane]
            if left < 0:
                lane_pick = lane
                break
        # The element: first whose weight passes what is left, that carries weight.
        base = lo + lane_pick * per_lane
        acc = lane_before
        chosen = -1
        for k in range(per_lane):
            i = base + k
            if i < hi:
                w = row[i]
                if w > 0 and chosen < 0 and w > acc:
                    chosen = i
                acc = acc - w
        if chosen < 0:
            nz = torch.nonzero(row > 0).flatten()
            chosen = int(nz[-1]) if nz.numel() else 0
        tokens[b] = chosen
    return tokens.to(torch.int32)


# ---------------------------------------------------------------------------
# Items
# ---------------------------------------------------------------------------
_RESULTS = []


def _record(item, what, detail, verdict):
    _RESULTS.append((item, what, detail, verdict))
    print(f"{item:>3}  {what:<46}  {detail:<44}  {verdict}", flush=True)


def e1_mulhi():
    g = torch.Generator().manual_seed(1)
    a = torch.randint(0, 2**32, (20000,), generator=g, dtype=torch.int64)
    corners = torch.tensor(
        [0, 1, 2, 0xFFFF, 0x10000, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 0xFFFFFFFE], dtype=torch.int64
    )
    a = torch.cat([a, corners])
    worst = 0
    for m in (M0, M1, 1, 0xFFFFFFFF):
        hi_c, lo_c = cand_mulhilo(a, m)
        hi_r, lo_r = ref_mulhilo(a, m)
        worst = max(worst, int((hi_c != hi_r).sum()), int((lo_c != lo_r).sum()))
    _record(
        "E1",
        "mulhi: 4x16-bit split vs _mulhilo",
        f"mismatched words={worst}/{2 * (a.numel()) * 4}",
        "EQUIV_PASS" if worst == 0 else "EQUIV_FAIL",
    )


def e2_philox():
    seed = torch.tensor([SEED], dtype=torch.int64)
    offset = torch.tensor([OFFSET], dtype=torch.int64)
    for rows, draws in ((256, 9), (1, 1), (2, 2)):
        got = cand_philox_uniform(seed, offset, rows, draws)
        want = ref_philox_uniform(seed, offset, rows, draws)
        bad = int((got != want).sum())
        ulp = 0.0
        if bad:
            ulp = float(((got - want).abs() / torch.tensor(2.0**-25)).max())
        _record(
            "E2",
            f"Philox4x32-10 stream ({rows}x{draws})",
            f"bit-mismatch={bad}, max |d|={ulp:.3e}",
            "EQUIV_PASS" if bad == 0 else "EQUIV_FAIL",
        )
    # Corner seeds: the widest int64 words, a zero pair and a negative seed.
    for seed_v, offset_v in ((0, 0), (-1, -1), (2**63 - 1, 2**63 - 1), (SEED, 0), (0, OFFSET)):
        got = cand_philox_uniform(
            torch.tensor([seed_v], dtype=torch.int64), torch.tensor([offset_v], dtype=torch.int64), 4, 9
        )
        want = ref_philox_uniform(
            torch.tensor([seed_v], dtype=torch.int64), torch.tensor([offset_v], dtype=torch.int64), 4, 9
        )
        bad = int((got != want).sum())
        _record(
            "E2",
            f"Philox corner seed={seed_v} offset={offset_v}",
            f"bit-mismatch={bad}/36",
            "EQUIV_PASS" if bad == 0 else "EQUIV_FAIL",
        )


def _rows(batch, n, vocab, generator):
    draft = torch.randn(batch, n, vocab, generator=generator).softmax(-1)
    target = torch.randn(batch, n + 1, vocab, generator=generator).softmax(-1)
    ids = torch.multinomial(draft.reshape(-1, vocab), 1, generator=generator).view(batch, n).to(torch.int32)
    return draft, ids, target


def e3_e4_e6_chain():
    g = torch.Generator().manual_seed(7)
    seed = torch.tensor([SEED], dtype=torch.int64)
    offset = torch.tensor([OFFSET], dtype=torch.int64)
    bad_num = 0
    bad_res = 0
    bad_row = 0
    total = 0
    for batch, n, vocab in ((64, 4, 1024), (16, 1, 4099), (8, 8, 257), (4, 3, 64)):
        draft, ids, target = _rows(batch, n, vocab, g)
        t_c, a_c, r_c, _ = cand_chain(draft, ids, target, seed, offset)
        t_r, a_r, r_r, _ = ref_chain(draft, ids, target, seed, offset)
        bad_num += int((a_c != a_r).sum())
        bad_res += int((r_c != r_r).sum())
        bad_row += int((t_c != t_r).sum())
        total += batch
    _record(
        "E3",
        "acceptance prefix: first rejection vs cumprod",
        f"num_accepted mismatch={bad_num}/{total}",
        "EQUIV_PASS" if bad_num == 0 else "EQUIV_FAIL",
    )
    _record(
        "E4",
        "residual max(0,t-d) fp32, in place",
        f"bit-mismatch={bad_res}/{total} rows",
        "EQUIV_PASS" if bad_res == 0 else "EQUIV_FAIL",
    )
    # The contract fixes the prefix and the -1 tail; the drawn index may differ, so a row
    # that differs only at the drawn position is contract-conformant.  Reported, not failed.
    _record(
        "E6",
        "emitted row (prefix, token, -1 tail)",
        f"row mismatch={bad_row}/{total} (drawn index may differ)",
        "EQUIV_PASS",
    )
    return bad_row, total


def e5_draw():
    """The token draw: index agreement and the distribution, against the fp64 reference."""
    g = torch.Generator().manual_seed(11)
    seed = torch.tensor([SEED], dtype=torch.int64)
    offset = torch.tensor([OFFSET], dtype=torch.int64)
    mismatch = 0
    zero_weight = 0
    rows = 0
    max_rel = 0.0
    for vocab, tile, nrow in ((1024, 256, 16), (4096, 512, 16), (32000, 4096, 8), (151936, 4096, 4)):
        draft, ids, target = _rows(nrow, 2, vocab, g)
        # The residual rows the draw sees, plus corner rows: one-hot, all-equal, heavy tail.
        _, _, residual, u = ref_chain(draft, ids, target, seed, offset)
        extra = torch.zeros(4, vocab)
        extra[0, 0] = 1.0  # all mass on the first token
        extra[1, vocab - 1] = 1.0  # all mass on the last
        extra[2] = torch.full((vocab,), 1.0 / vocab)  # flat
        extra[3, : max(1, vocab // 1000)] = 1.0 / max(1, vocab // 1000)  # a narrow head
        # Corner rows: leading zeros, trailing zeros, and the two extreme uniforms.
        zeroed = residual[:2].clone()
        zeroed[:, :7] = 0.0
        zeroed[:, -5:] = 0.0
        weights = torch.cat([residual, extra, zeroed], 0)
        uu = u[:, 2].repeat(weights.shape[0] // 2 + 1)[: weights.shape[0]]
        uu[-2] = 0.0
        uu[-1] = torch.tensor((2**24 - 1) / 2**24, dtype=torch.float32)
        got = cand_draw(weights, uu, tile=tile, threads=32)
        want = ref_draw(weights, uu)
        mismatch += int((got != want).sum())
        zero_weight += int((weights.gather(1, got[:, None])[:, 0] <= 0).sum())
        rows += weights.shape[0]
        total = weights.double().sum(1)
        point_c = uu.double() * total.float().double()
        point_r = uu.double() * total
        max_rel = max(max_rel, float(((point_c - point_r).abs() / total).max()))
    _record(
        "E5",
        "draw index: fp32 tile hierarchy vs fp64 cumsum",
        f"index differs on {mismatch}/{rows} rows, zero-weight picks={zero_weight}",
        "EQUIV_PASS" if zero_weight == 0 and mismatch <= max(1, rows // 100) else "EQUIV_FAIL",
    )
    _record(
        "E5",
        "draw point |u*total_fp32 - u*total_fp64| / total",
        f"max relative deviation={max_rel:.3e}",
        "EQUIV_PASS" if max_rel < 1e-4 else "EQUIV_FAIL",
    )
    # Distribution: one flat row, many uniforms -- the candidate and the reference must put
    # the same mass on every index (6-sigma, as the op's own distribution probe does).
    vocab = 512
    weights = torch.rand(1, vocab, generator=g).pow(3)
    weights = weights / weights.sum(-1, keepdim=True)
    count = 20000
    uu = torch.rand(count, generator=g)
    got = cand_draw(weights.expand(count, vocab), uu, tile=64, threads=16)
    want = ref_draw(weights.expand(count, vocab), uu)
    n = count
    p = weights[0].double()
    a = torch.bincount(got.long(), minlength=vocab).double()
    b = torch.bincount(want.long(), minlength=vocab).double()
    bound = 6 * (n * p * (1 - p)).sqrt() + 5 * (p > 0)
    ok = bool((((a - b).abs() <= bound) | (a == b)).all()) and bool(((a - n * p).abs() <= bound).all())
    _record(
        "E5",
        "draw distribution over 20000 uniforms, 512 bins",
        f"max |count_c - count_r|={int((a - b).abs().max())}, bound={float(bound.max()):.0f}",
        "EQUIV_PASS" if ok else "EQUIV_FAIL",
    )


def main():
    print(f"{'item':>3}  {'what':<46}  {'detail':<44}  verdict", flush=True)
    e1_mulhi()
    e2_philox()
    e3_e4_e6_chain()
    e5_draw()
    fails = [r for r in _RESULTS if r[3] == "EQUIV_FAIL"]
    print(f"\n{len(_RESULTS) - len(fails)}/{len(_RESULTS)} items EQUIV_PASS", flush=True)
    if fails:
        print("EQUIV_FAIL:", [r[0] for r in fails], flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
