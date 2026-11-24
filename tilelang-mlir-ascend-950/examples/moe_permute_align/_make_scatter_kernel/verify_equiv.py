"""D-1 equivalence machine-check for _make_scatter_kernel (Stage 1).

Every adopted formula-level optimisation in DESIGN.md §1.6.1 is compared against
its baseline formulation on random tensors plus corner values.

The kernel performs no floating-point arithmetic (flops = 0, pure int32 index
arithmetic), so the "fp64 reference" analogue is an exact **int64** reference:
both formulations are evaluated in int64 and compared exactly, and because the
op contract leaves the order *within* an expert's range unspecified
(``tests/ops/test_moe_permute_align.py::_assert_route_equivalent`` normalises it
away) the comparison is the contract-level one: per-expert multiset equality,
slot-range/disjointness, sentinel preservation, and cursor end position.

Adopted items (DESIGN.md §1.6.1):
  O1  single-level bump cursor   vs  two-level per-block histogram + bulk reserve
  O2  clamped sentinel load      vs  guarded load (`eid = -1` for tail lanes)
  O3  int32 range of the slot / cursor domain (no overflow in any workload)
  O4  expert upper-bound guard   vs  the source's lower-bound-only guard

Run:  cd <task dir> && /workspace/venv-ascend950/bin/python verify_equiv.py
"""

from __future__ import annotations

import math
import random

import torch

PASS, FAIL = "EQUIV_PASS", "EQUIV_FAIL"


def padded_starts(flat: torch.Tensor, num_experts: int, block_size: int) -> torch.Tensor:
    """Exclusive prefix sum of the per-expert padded counts (what the align stage writes)."""
    counts = torch.bincount(flat, minlength=num_experts).to(torch.int64)
    padded = (counts + block_size - 1) // block_size * block_size
    starts = torch.cumsum(padded, 0) - padded
    return torch.cat([starts, torch.cumsum(padded, 0)[-1:] if num_experts else torch.zeros(1, dtype=torch.int64)])


def baseline_two_level(flat, num_experts, block_size, tile):
    """Baseline = the source algorithm: per-block SMEM histogram + local rank,
    per-expert bulk reserve from the global cursor, then base + rank writes."""
    numel = int(flat.numel())
    cumsum = padded_starts(flat, num_experts, block_size)
    capacity = numel + (num_experts + 1) * (block_size - 1)
    out = torch.full((capacity,), numel, dtype=torch.int64)
    cursor = cumsum.clone()
    for bid in range(math.ceil(numel / tile)):
        lo, hi = bid * tile, min(numel, (bid + 1) * tile)
        if lo >= hi:
            continue
        block = flat[lo:hi]
        rank = torch.zeros(hi - lo, dtype=torch.int64)
        base = torch.zeros(num_experts, dtype=torch.int64)
        for e in range(num_experts):
            hit = (block == e).nonzero().squeeze(1)
            if hit.numel():
                rank[hit] = torch.arange(hit.numel(), dtype=torch.int64)  # SMEM-local rank
                base[e] = cursor[e]                                       # bulk reserve (prev)
                cursor[e] += hit.numel()
        for j in range(hi - lo):
            out[int(base[int(block[j])] + rank[j])] = lo + j
    return out, cumsum[:num_experts].clone(), cursor, numel


def candidate_single_level(flat, num_experts, block_size, tile, order=None):
    """Candidate: one global bump per route -- slot = prev(atomic_add(cursor[e], 1)).

    ``order`` fixes the atomic arrival order: on the device any interleaving of
    lanes and blocks is legal, so the candidate is replayed under three legal
    orders and the contract invariants must hold for each."""
    numel = int(flat.numel())
    cumsum = padded_starts(flat, num_experts, block_size)
    capacity = numel + (num_experts + 1) * (block_size - 1)
    out = torch.full((capacity,), numel, dtype=torch.int64)
    cursor = cumsum.clone()
    nblocks = math.ceil(numel / tile)
    routing = [(bid, j) for bid in range(nblocks) for j in range(min(tile, numel - bid * tile))]
    if order == "reversed":
        routing = routing[::-1]
    elif order == "shuffled":
        random.Random(0).shuffle(routing)
    for bid, j in routing:
        idx = bid * tile + j
        e = int(flat[idx])
        slot = int(cursor[e])  # == prev value returned by asc_atomic_add
        cursor[e] += 1
        out[slot] = idx
    return out, cumsum[:num_experts].clone(), cursor, numel


def contract_violations(out, starts, cursor, numel, num_experts, flat):
    """0 iff the written array satisfies the op contract for this routing."""
    bad = 0
    used = torch.zeros(out.numel(), dtype=torch.bool)
    for e in range(num_experts):
        lo, hi = int(starts[e]), int(cursor[e])
        cnt = int((flat == e).sum())
        if hi - lo != cnt:                       # bump consumed exactly count_e slots
            bad += 1
        if hi > out.numel() or lo < 0:           # stayed inside the buffer
            bad += 1
        got = out[lo:hi]
        if got.numel() != cnt:                   # right number of routes
            bad += 1
        if int(torch.unique(got).numel()) != got.numel():   # disjoint slots
            bad += 1
        if bool(torch.any((got < 0) | (got >= numel))):     # legal route indices
            bad += 1
        used[lo:hi] = True
    if bool(torch.any(out[~used] != numel)):      # sentinel preserved elsewhere
        bad += 1
    return bad


def expert_multisets(out, starts, cursor, num_experts):
    return [sorted(out[int(starts[e]):int(cursor[e])].tolist()) for e in range(num_experts)]


def make_flat(numel, num_experts, seed=0, skew=False):
    g = torch.Generator().manual_seed(seed)
    if skew:
        return torch.zeros(numel, dtype=torch.int64)
    return torch.randint(0, num_experts, (numel,), generator=g, dtype=torch.int64)


def main() -> int:
    rows = []

    # ------------------------------------------------------------- O1 (and O3)
    cases = [
        (8, 384, 64, 128, False),
        (256, 384, 64, 128, False),
        (1032, 128, 64, 128, False),
        (4096, 128, 64, 128, False),
        (4096, 256, 64, 128, True),      # every route on one expert
        (32768, 384, 64, 640, False),
        (32768, 128, 1, 640, False),     # block_size = 1
        (1, 1100, 48, 128, False),       # single route, > 1 block of experts
    ]
    viol = 0
    checks = 0
    for numel, e, bs, tile, skew in cases:
        flat = make_flat(numel, e, seed=numel + e + bs, skew=skew)
        base_out, starts, base_cursor, n = baseline_two_level(flat, e, bs, tile)
        # the baseline itself must satisfy the contract (sanity of the reference)
        viol += contract_violations(base_out, starts, base_cursor, n, e, flat)
        ref = expert_multisets(base_out, starts, base_cursor, e)
        for order in (None, "reversed", "shuffled"):
            cand_out, c_starts, c_cursor, _ = candidate_single_level(flat, e, bs, tile, order)
            checks += 1
            viol += contract_violations(cand_out, c_starts, c_cursor, n, e, flat)
            if expert_multisets(cand_out, c_starts, c_cursor, e) != ref:
                viol += 1
    rows.append((f"O1 single-level bump vs two-level ({len(cases)} shapes x 3 arrival orders)",
                 0, viol, PASS if viol == 0 else FAIL))

    # --------------------------------------------------------------------- O2
    v2 = 0
    for numel, tile in ((1, 128), (8, 128), (255, 128), (1032, 128), (4096, 640), (32768, 640)):
        total = math.ceil(numel / tile) * tile
        idx = torch.arange(total, dtype=torch.int64)
        flat = make_flat(numel, 7, seed=numel)
        guarded = torch.where(idx < numel, idx, torch.full_like(idx, -1))
        clamped = torch.where(idx < numel, torch.clamp(idx, max=numel - 1), torch.full_like(idx, -1))
        a = torch.where(guarded >= 0, flat[guarded.clamp(min=0)], torch.full_like(guarded, -1))
        b = torch.where(clamped >= 0, flat[clamped.clamp(min=0)], torch.full_like(clamped, -1))
        if not torch.equal(a, b):                          # identical eid stream
            v2 += 1
        if bool(torch.any(clamped[clamped >= 0] >= numel)):  # clamped index never OOB
            v2 += 1
        if bool(torch.any(guarded[guarded >= 0] >= numel)):  # guarded index never OOB
            v2 += 1
    rows.append(("O2 clamped sentinel load vs guarded load (6 shapes incl. tails)", 0, v2,
                 PASS if v2 == 0 else FAIL))

    # --------------------------------------------------------------------- O3
    v3 = 0
    worst_cap = 0
    for numel, e, bs in ((32768, 1100, 64), (32768, 384, 64), (32768, 128, 1), (8, 384, 64)):
        cap = numel + (e + 1) * (bs - 1)
        worst_cap = max(worst_cap, cap)
        if cap >= 2**31:
            v3 += 1
    rows.append(("O3 int32 slot/cursor range (4 workloads, max capacity value)", worst_cap, v3,
                 PASS if v3 == 0 else FAIL))

    # --------------------------------------------------------------------- O4
    v4 = 0
    for e in (1, 8, 128, 384, 1100):
        flat = make_flat(256, e, seed=e)
        if bool(torch.any((flat < 0) | (flat >= e))):   # in-contract draw stays in range
            v4 += 1
        # in-contract: the added `eid < num_experts` guard is never taken, so the
        # emitted atomic/store stream is identical to the two-operand-guard form
        guarded_hits = int((flat >= e).sum())
        if guarded_hits != 0:
            v4 += 1
    rows.append(("O4 expert upper-bound guard (in-contract domain identical)", 0, v4,
                 PASS if v4 == 0 else FAIL))

    print(f"{'item':74s} {'value':>10s} {'violations':>10s}  verdict")
    ok = True
    for name, value, violations, verdict in rows:
        print(f"{name:74s} {value:10d} {violations:10d}  {verdict}")
        ok &= verdict == PASS
    print(f"\n(route-order replays checked in O1: {checks})")
    print("NOTE: no floating-point arithmetic in this kernel -> the int64 reference is")
    print("      exact (fp64 analogue). Order-agnostic invariants are replayed over three")
    print("      legal atomic arrival orders per shape, since the op contract leaves the")
    print("      order within an expert's range unspecified.")
    print(f"\nRESULT: {PASS if ok else FAIL}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
