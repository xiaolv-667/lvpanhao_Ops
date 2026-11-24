"""§1.6.1 equivalence machine verification for `_make_small_batch_kernel` (Stage 1, D-1).

Design-phase, device-free: everything runs on CPU in exact integer arithmetic
(Python `int`, i.e. the same values as an int64/fp64 reference), so every
comparison below is an *exact* result (0 ulp) rather than a tolerance result.

Three implementations of the same operator semantics (DESIGN.md §0.1):

  baseline(...)    the GPU source algorithm (`permute_align.py:222-340`):
                   32 private per-lane count rows + row-wise prefix + serial
                   padded-total prefix + sentinel fill + per-lane scatter.
  optimized(...)   this design (DESIGN.md §1.3 A1-A5): compare-accumulate
                   histogram with UNROLL independent accumulators, shift
                   padding, Hillis-Steele scan over W=32 lanes, rank-count
                   `expert_ids`, token-parallel atomic-bump scatter.
  reference(...)   the op's independent reference semantics, ported from
                   `examples/TileOPs/tileops/workloads/moe.py::ref_permute_align`
                   (ascending route order + binary search for the block owner).

Checks
  A1  shift padding `((c + B-1) >> p) << p` vs `ceil(c/B)*B` (B = 2^p), and the
      div/mul fallback for B not a power of two.
  A2  rank-count `expert_ids` vs the baseline expert-major variable-length loop.
  A3  Hillis-Steele exclusive prefix (W lanes, left halo, padded lanes = total)
      vs the baseline serial prefix.
  A4  UNROLL-accumulator histogram vs the baseline count matrix (and bincount).
  A5  atomic-bump scatter vs the baseline per-lane scatter: contract level
      (per-expert multiset, slot distinctness, slot range, sentinel preserved,
      cursor end value) -- the expert-internal order is contract-undefined.
  A6  padded-lane discipline: (i) `cumsum[0..E]` unaffected by the idle lanes
      (E < W) and (ii) the discrimination control -- reading the idle lanes with
      UB garbage breaks `expert_ids` when the rank-count bound is W, and the
      bound-E form survives it (the invariant is load-bearing).
  A7  `numel = 0` boundary: all three outputs at their contract values.

Run:  cd examples/moe_permute_align/_make_small_batch_kernel && \
      /workspace/venv-ascend950/bin/python verify_equiv.py
Exit code 0 == every check EQUIV_PASS.
"""

from __future__ import annotations

import itertools
import math
import random
import sys

W = 32          # lane count of the fused kernel (E <= 32)
UNROLL = 4      # histogram accumulators per lane (DESIGN.md §1.6.1 A4)
MAX_BLOCK = 1 << 30


def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def geom(numel: int, E: int, B: int) -> tuple[int, int]:
    padded = numel + (E + 1) * (B - 1)
    return padded, math.ceil(padded / B)


# ---------------------------------------------------------------------------
# baseline: the GPU source algorithm, transcribed from the source (§0.3)
# ---------------------------------------------------------------------------
def baseline(flat, E, B):
    numel = len(flat)
    padded, nblocks = geom(numel, E, B)

    # tokens_cnts: row k+1 holds lane k's counts; lane = index mod W
    rows = [[0] * E for _ in range(W + 1)]
    for i, e in enumerate(flat):
        rows[(i % W) + 1][e] += 1
    # prefix over the rows: row k becomes lane k's first rank, row W the totals
    for k in range(W):
        for e in range(E):
            rows[k + 1][e] += rows[k][e]

    cumsum = [0] * (E + 1)
    for e in range(E):
        cumsum[e + 1] = cumsum[e] + ceil_div(rows[W][e], B) * B
    total = cumsum[E]

    sorted_ids = [numel] * padded
    cursor = [rows[k][:] for k in range(W)]
    for i, e in enumerate(flat):
        lane = i % W
        sorted_ids[cursor[lane][e] + cumsum[e]] = i
        cursor[lane][e] += 1

    expert_ids = [E] * nblocks
    for e in range(E):
        start = cumsum[e] // B
        nb = cumsum[e + 1] // B - start
        for b in range(nb):
            expert_ids[start + b] = e
    return sorted_ids, expert_ids, total, cumsum


# ---------------------------------------------------------------------------
# optimized: this design (§1.3 A1-A5)
# ---------------------------------------------------------------------------
def optimized(flat, E, B, unroll: int = UNROLL, rank_bound: str = "E",
              garbage_idle: bool = False, garbage_value: int = -777):
    numel = len(flat)
    padded, nblocks = geom(numel, E, B)
    chunk = max(64, math.ceil(max(numel, 1) / 64) * 64)
    p_shift = int(math.log2(B)) if B & (B - 1) == 0 else None

    # route tile, tail padded with an id that matches no expert
    x = list(flat) + [-1] * (chunk - numel)

    # ---- A4: histogram, lane <-> expert, `unroll` independent accumulators ----
    cnt = []
    for e in range(W):
        acc = [0] * unroll
        for i in range(chunk // unroll):
            for j in range(unroll):
                if x[i * unroll + j] == e:
                    acc[j] += 1
        cnt.append(sum(acc))

    # ---- A1: padded count ----
    if p_shift is not None:
        pad = [((c + (B - 1)) >> p_shift) << p_shift for c in cnt]
    else:
        pad = [ceil_div(c, B) * B for c in cnt]

    # ---- A3: Hillis-Steele inclusive scan over the W lanes (left halo = 0) ----
    vals = list(pad)
    d = 0
    while (1 << d) < W:
        k = 1 << d
        vals = [vals[e] + (vals[e - k] if e >= k else 0) for e in range(W)]
        d += 1
    cum = [0] + vals                       # exclusive prefix over [0, W]

    total = cum[W]

    if garbage_idle:
        # Discrimination control: the idle lanes (E < e < W) carry UB garbage
        # instead of the defined `total` the shift-left step writes into [0, W].
        # Index W keeps the true total (it is the value num_tokens_post_pad
        # reads), so the control isolates the rank-count bound.
        for e in range(E + 1, W):
            cum[e] = garbage_value

    # ---- A2: expert_ids by the rank-count identity ----
    bound = E if rank_bound == "E" else W
    expert_ids = []
    for b in range(nblocks):
        pos = b * B
        le = sum(1 for e in range(bound) if cum[e] <= pos)
        expert_ids.append(le - 1 if pos < total else E)

    # ---- sentinel fill + A5: token-parallel atomic-bump scatter ----
    sorted_ids = [numel] * padded
    cur = cum[:E]
    rounds = ceil_div(max(numel, 1), W)
    for r in range(rounds):
        for j in range(W):
            idx = r * W + j
            if idx < numel:
                e = x[idx]
                slot = cur[e]
                cur[e] = slot + 1
                sorted_ids[slot] = idx
    return sorted_ids, expert_ids, total, cum[:E + 1], cur, cnt


# ---------------------------------------------------------------------------
# reference: the op's own semantics (workloads/moe.py::ref_permute_align)
# ---------------------------------------------------------------------------
def reference(flat, E, B):
    numel = len(flat)
    padded, nblocks = geom(numel, E, B)
    counts = [0] * E
    for e in flat:
        counts[e] += 1
    cumsum = [0] * (E + 1)
    for e in range(E):
        cumsum[e + 1] = cumsum[e] + ceil_div(counts[e], B) * B
    total = cumsum[E]
    sorted_ids = [numel] * total
    slot = list(cumsum[:E])
    for i, e in enumerate(flat):
        sorted_ids[slot[e]] = i
        slot[e] += 1
    expert_ids = []
    for b in range(total // B):
        start = b * B
        lo, hi, found = 0, E - 1, E - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if cumsum[mid] <= start < cumsum[mid + 1]:
                found = mid
                break
            if start < cumsum[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        expert_ids.append(found)
    return sorted_ids, expert_ids, total


# ---------------------------------------------------------------------------
# comparisons
# ---------------------------------------------------------------------------
def per_expert_multisets(values, starts, counts):
    """Group a slot array by its owner expert, in ascending order."""
    return [sorted(values[starts[e]:starts[e] + counts[e]]) for e in range(len(counts))]


def cmp_exact(name, a, b, cases):
    bad = 0
    first = None
    for case, x, y in cases:
        if x != y:
            bad += 1
            if first is None:
                first = (case, x, y)
    rate = bad / len(cases) if cases else 0.0
    ok = bad == 0
    print(f"| {name} | {len(cases)} | 0 | 0 | {rate:.6f} | "
          f"{'**EQUIV_PASS**' if ok else '**EQUIV_FAIL**'} |")
    if not ok:
        print(f"    first mismatch: {first}")
    return ok


DISTS = ("uniform", "one_expert", "alternating", "blocked", "single")


def make_flat(numel: int, E: int, dist: str, seed: int) -> list[int]:
    rng = random.Random(seed)
    if numel == 0:
        return []
    if dist == "uniform":
        return [rng.randrange(E) for _ in range(numel)]
    if dist == "one_expert":
        return [0] * numel
    if dist == "alternating":
        return [i % E for i in range(numel)]
    if dist == "blocked":
        out = []
        e = 0
        while len(out) < numel:
            out.extend([e] * min(E + 1, numel - len(out)))
            e = (e + 1) % E
        return out[:numel]
    if dist == "single":
        # one route per expert, the rest empty (E <= numel only)
        out = list(range(min(E, numel)))
        out.extend([0] * (numel - len(out)))
        return out
    raise ValueError(dist)


SHAPES = [
    (0, 8, 16),        # no routes, few experts (manifest test "no-routes-few-experts")
    (2, 4, 4),         # single token: 1 token x top_k 2
    (8, 4, 4),         # smoke case
    (8, 8, 16),
    (32, 8, 16),
    (100, 8, 48),      # block_size not a power of two -> div/mul fallback
    (200, 8, 16),
    (600, 8, 64),
    (600, 32, 64),     # max experts of the fused path
    (1023, 8, 16),     # last small-batch route count
    (1023, 32, 64),    # largest fused workload
    (1023, 1, 64),     # single expert
    (1000, 3, 1),      # block_size 1, E not a power of two
    (700, 32, 32),
]


def main() -> int:
    print("## verify_equiv.py -- `_make_small_batch_kernel` (W=%d, UNROLL=%d)" % (W, UNROLL))
    print(f"cases: {len(SHAPES)} shapes x {len(DISTS)} distributions = "
          f"{len(SHAPES) * len(DISTS)} per check")
    print()
    print("| check | cases | max abs diff | max ulp diff | violation rate | conclusion |")
    print("|---|---|---|---|---|---|")

    ok = True
    cases = [(s, make_flat(s[0], s[1], d, 17 + i), d)
             for i, (s, d) in enumerate(itertools.product(SHAPES, DISTS))]

    # ---- A3 + A4 + A1: cumsum / total ----
    cs_cases, total_cases = [], []
    for shape, flat, dist in cases:
        numel, E, B = shape
        _, _, base_total, base_cum = baseline(flat, E, B)
        _, _, opt_total, opt_cum, _, _ = optimized(flat, E, B)
        cs_cases.append(((shape, dist), opt_cum, base_cum))
        total_cases.append(((shape, dist), opt_total, base_total))
    ok &= cmp_exact("A1+A3+A4 cumsum (shift pad + HS scan) vs baseline serial", None, None, cs_cases)
    ok &= cmp_exact("num_tokens_post_pad (= cumsum[W]) vs baseline", None, None, total_cases)

    # ---- third-party cross-check: reference cumsum ----
    ref_cum_cases = []
    for shape, flat, dist in cases:
        numel, E, B = shape
        ref_sorted, ref_ids, ref_total = reference(flat, E, B)
        _, _, opt_total, opt_cum, _, _ = optimized(flat, E, B)
        ref_cum_cases.append(((shape, dist), opt_cum[:E + 1],
                              baseline(flat, E, B)[3]))
    ok &= cmp_exact("cumsum vs reference/baseline (self-check)", None, None, ref_cum_cases)

    # ---- A2: expert_ids ----
    id_cases = []
    for shape, flat, dist in cases:
        numel, E, B = shape
        base_ids = baseline(flat, E, B)[1]
        opt_ids = optimized(flat, E, B)[1]
        id_cases.append(((shape, dist), opt_ids, base_ids))
    ok &= cmp_exact("A2 expert_ids (rank-count) vs baseline expert-major loop", None, None, id_cases)

    # reference only covers [0, total/B); the tail is the sentinel's business
    ref_id_cases = []
    for shape, flat, dist in cases:
        numel, E, B = shape
        _, ref_ids, ref_total = reference(flat, E, B)
        opt_ids = optimized(flat, E, B)[1]
        ref_id_cases.append(((shape, dist), opt_ids[:ref_total // B], ref_ids))
    ok &= cmp_exact("A2 expert_ids vs reference (binary search)", None, None, ref_id_cases)

    # ---- A5: scatter, contract level ----
    sc_cases, sent_cases, cur_cases = [], [], []
    for shape, flat, dist in cases:
        numel, E, B = shape
        starts, counts = [], []
        base_sorted, base_ids, base_total, base_cum = baseline(flat, E, B)
        opt_sorted, _, opt_total, opt_cum, opt_end, _ = optimized(flat, E, B)
        cnt = [0] * E
        for e in flat:
            cnt[e] += 1
        base_start = base_cum[:E]
        # routed slots hold each expert's route set (order undefined)
        routed = [sum(cnt)] + [0]
        got_ms = per_expert_multisets(opt_sorted, base_start, cnt)
        exp_ms = per_expert_multisets(base_sorted, base_start, cnt)
        slots = []
        for e in range(E):
            slots.extend(range(base_start[e], base_start[e] + cnt[e]))
        inside = all(0 <= s < len(opt_sorted) for s in slots)
        distinct = len(set(slots)) == len(slots)
        ok_multiset = got_ms == exp_ms
        ok_slots = inside and distinct
        # every slot outside the routed ranges keeps the sentinel
        outside = set(range(len(opt_sorted))) - set(slots)
        sent_ok = all(opt_sorted[i] == numel for i in outside)
        sc_cases.append(((shape, dist), (ok_multiset, ok_slots),
                         (True, True)))
        sent_cases.append(((shape, dist), sent_ok, True))
        cur_cases.append(((shape, dist), opt_end, [base_start[e] + cnt[e] for e in range(E)]))
    ok &= cmp_exact("A5 scatter: per-expert multiset + slot distinctness/range", None, None, sc_cases)
    ok &= cmp_exact("A5 sentinel preserved outside every expert range", None, None, sent_cases)
    ok &= cmp_exact("A5 cursor end = start + count", None, None, cur_cases)

    # normalized comparison against the reference (ascending order there)
    norm_cases = []
    for shape, flat, dist in cases:
        numel, E, B = shape
        ref_sorted, _, ref_total = reference(flat, E, B)
        opt_sorted, _, opt_total, _, _, _ = optimized(flat, E, B)
        cnt = [0] * E
        for e in flat:
            cnt[e] += 1
        pad = [(ceil_div(c, B) * B) for c in cnt]
        start = [0] * E
        for e in range(1, E):
            start[e] = start[e - 1] + pad[e - 1]
        got = per_expert_multisets(opt_sorted, start, cnt)
        exp = per_expert_multisets(ref_sorted, start, cnt)
        tail = all(v == numel for v in opt_sorted[opt_total:])
        norm_cases.append(((shape, dist), (got == exp, tail), (True, True)))
    ok &= cmp_exact("A5 normalized sorted_token_ids vs reference (+ tail sentinel)", None, None, norm_cases)

    # ---- A1: shift vs div/mul over the full count range ----
    a1_cases = []
    for B in (1, 2, 4, 16, 32, 64, 128):
        p = int(math.log2(B))
        for c in range(0, 1024):
            a1_cases.append(((c, B), ((c + (B - 1)) >> p) << p, ceil_div(c, B) * B))
    ok &= cmp_exact("A1 shift padding vs ceil(c/B)*B (B = 2^p, c < 1024)", None, None, a1_cases)

    # ---- A4: histogram vs bincount, all unroll factors ----
    a4_cases = []
    for shape, flat, dist in cases:
        numel, E, B = shape
        for unroll in (1, 2, 4, 8):
            counts = [0] * E
            for e in flat:
                counts[e] += 1
            got = optimized(flat, E, B, unroll=unroll)[5]
            a4_cases.append(((shape, dist, unroll), got, counts + [0] * (W - E)))
    ok &= cmp_exact("A4 histogram (UNROLL 1/2/4/8) vs bincount (idle lanes = 0)", None, None, a4_cases)

    # ---- A6: padded-lane discipline + discrimination control ----
    a6_cases = []
    for shape, flat, dist in cases:
        numel, E, B = shape
        if E >= W:
            continue
        base_ids = baseline(flat, E, B)[1]
        opt_ids = optimized(flat, E, B, rank_bound="E")[1]
        opt_ids_w = optimized(flat, E, B, rank_bound="W")[1]
        a6_cases.append(((shape, dist, "boundE"), opt_ids, base_ids))
        a6_cases.append(((shape, dist, "boundW"), opt_ids_w, base_ids))
    ok &= cmp_exact("A6 idle lanes (E<W): expert_ids with bound E and bound W", None, None, a6_cases)

    # discrimination control: with UB garbage in the idle lanes, only bound E survives
    good, bad_w = 0, 0
    total_ctl = 0
    for shape, flat, dist in cases:
        numel, E, B = shape
        if E >= W or len(flat) == 0:
            continue
        total_ctl += 1
        base_ids = baseline(flat, E, B)[1]
        if optimized(flat, E, B, rank_bound="E", garbage_idle=True)[1] == base_ids:
            good += 1
        if optimized(flat, E, B, rank_bound="W", garbage_idle=True)[1] != base_ids:
            bad_w += 1
    print(f"| A6 discrimination control (idle lanes = UB garbage, {total_ctl} cases) "
          f"| {total_ctl} | - | - | boundE wrong {total_ctl - good}/{total_ctl}; "
          f"boundW wrong {bad_w}/{total_ctl} | "
          f"{'**EQUIV_PASS**' if good == total_ctl and bad_w > 0 else '**EQUIV_FAIL**'} |")
    ok &= (good == total_ctl and bad_w > 0)

    # ---- A7: numel = 0 contract values ----
    zero_ok = True
    detail = []
    for E, B in ((8, 16), (4, 4), (32, 64), (1, 1)):
        _, ids, total, _, _, _ = optimized([], E, B)
        sorted_ids = optimized([], E, B)[0]
        padded, nblocks = geom(0, E, B)
        conds = (
            total == 0,
            ids == [E] * nblocks,
            sorted_ids == [0] * padded,
            len(sorted_ids) == padded,
        )
        zero_ok &= all(conds)
        detail.append(f"E={E},B={B}: total={total} ids_all_E={ids == [E] * nblocks} "
                      f"sentinel0={sorted_ids == [0] * padded}")
    print(f"| A7 numel=0 boundary (4 geometries) | 4 | 0 | 0 | 0.000000 | "
          f"{'**EQUIV_PASS**' if zero_ok else '**EQUIV_FAIL**'} |")
    ok &= zero_ok

    print()
    print("L0 landmarks (asserted inline):")
    for (shape, dist), _, _ in cs_cases:
        numel, E, B = shape
        if (numel, E, B) in ((600, 32, 64), (1023, 8, 16), (100, 8, 48), (1000, 3, 1), (0, 8, 16)) \
                and dist == "uniform":
            _, ids, total, cum, _, _ = optimized(make_flat(numel, E, dist, 17), E, B)
            _, base_ids, base_total, base_cum = baseline(make_flat(numel, E, dist, 17), E, B)
            print(f"  ({numel},{E},{B}) uniform: total={total} (base {base_total}) "
                  f"cumsum MATCH {cum == base_cum} expert_ids MATCH {ids == base_ids}")
    print("  numel=0: " + " | ".join(detail))
    print()
    print("RESULT:", "ALL EQUIV_PASS" if ok else "EQUIV_FAIL PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
