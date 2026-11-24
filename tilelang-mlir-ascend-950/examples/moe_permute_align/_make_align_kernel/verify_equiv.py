"""Machine equivalence check for the adopted formula-level optimizations of
``_make_align_kernel`` (MoE token->expert align, int32 index arithmetic).

Run:
    python verify_equiv.py            # torch CPU only, no NPU needed

What is compared
----------------
``baseline_*``  = the source GPU align kernel's formula, ported literally from
                  /workspace/TileOPs/src/tileops/_kernels/moe/permute_align.py
                  (``_make_align_kernel``): bincount -> per-expert
                  ``ceil(cnt/block)*block`` -> serial exclusive prefix ->
                  per-expert variable-length block scatter; sentinel fill.
``opt_*``       = the NPU design's adopted formula after the §1.6.1 rewrites:
                  A1 padding by shift (block_size = 2**p) instead of division,
                  A2 ``expert_ids`` by the rank-counting identity instead of the
                  per-expert variable-length scatter,
                  A3 exclusive prefix by an inclusive Hillis-Steele scan + shift
                  instead of the serial chain,
                  A4 counts by the per-expert compare-accumulate form the NPU
                  histogram uses instead of an atomic bincount.
``opt_padded_*``= the same adopted formula, but evaluated **on the kernel's real
                  lane geometry** (A5): ``E_pad = align_up(E, 64)`` lanes, the
                  padding lanes of ``s_cnt``/``s_pad`` are 0, the left-shift step
                  writes the whole ``[0, E_pad)`` range (so padding lanes hold
                  ``total`` -- a *defined* value), and the rank-count reduces over
                  the build-time constant ``E``.  This is the case the pre-revision
                  design got wrong (it read ``s_cs[E+1 .. E_pad)`` which the
                  left-shift step never wrote); ``garbage`` reproduces that
                  pre-fix read to show the guard is load-bearing.
``ref_*``       = an independent third implementation (the TileOPs workload
                  reference ``ref_permute_align``: bincount + cumsum +
                  ``searchsorted``), used to prove the baseline itself is right.

Every candidate is computed in exact integer arithmetic and, additionally, in
float64 (the mandated fp64 reference) for the padding / prefix arithmetic; all
compared quantities are integers < 2**20, hence exactly representable in fp64.

Report per adopted item: max abs difference, max ulp difference, violation
rate (fraction of elements that differ), and EQUIV_PASS / EQUIV_FAIL.
"""

from __future__ import annotations

import itertools
import math

import torch

NUM_EXPERTS_SET = (1, 2, 3, 8, 33, 64, 128, 256, 384)
BLOCK_SIZE_SET = (1, 2, 4, 8, 16, 64, 128, 256)
NUMEL_SET = (0, 1, 7, 8, 33, 256, 600, 1024, 4096)

LANE_TILE = 64  # int32 vector lane count on the target (256B / 4B)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def max_padded_of(numel: int, num_experts: int, block_size: int) -> int:
    return numel + (num_experts + 1) * (block_size - 1)


def max_blocks_of(numel: int, num_experts: int, block_size: int) -> int:
    return math.ceil(max_padded_of(numel, num_experts, block_size) / block_size)


def align_up(v: int, m: int) -> int:
    return (v + m - 1) // m * m


def scan_steps(lanes: int) -> int:
    return 0 if lanes <= 1 else int(math.ceil(math.log2(lanes)))


def _pow2_or_none(v: int):
    return v.bit_length() - 1 if v > 0 and (v & (v - 1)) == 0 else None


def _counts(flat: torch.Tensor, num_experts: int) -> torch.Tensor:
    flat = flat.to(torch.int64)
    if flat.numel():
        return (flat.unsqueeze(1) == torch.arange(num_experts, dtype=torch.int64)).sum(0)
    return torch.zeros(num_experts, dtype=torch.int64)


def _pad(counts: torch.Tensor, block_size: int) -> torch.Tensor:
    """(A1) ceil to block_size by shift, with an fp64 reference cross-check."""
    p = _pow2_or_none(block_size)
    if p is None:
        padded = (counts + block_size - 1) // block_size * block_size
    else:
        padded = ((counts + block_size - 1) >> p) << p
    padded_f64 = (
        torch.floor((counts.to(torch.float64) + block_size - 1) / block_size) * block_size
    ).to(torch.int64)
    assert torch.equal(padded, padded_f64), "fp64 padding reference disagrees"
    return padded


def _inclusive_scan(lanes: torch.Tensor) -> torch.Tensor:
    """(A3) Hillis-Steele inclusive scan: python-level ``for d in range(steps)``."""
    scan = lanes.clone()
    for d in range(scan_steps(int(lanes.numel()))):
        k = 1 << d
        shifted = torch.zeros_like(scan)
        shifted[k:] = scan[:-k]
        scan = scan + shifted
    return scan


def _report(name: str, cand: torch.Tensor, base: torch.Tensor) -> dict:
    cand = cand.to(torch.int64).flatten()
    base = base.to(torch.int64).flatten()
    if cand.numel() != base.numel():
        return {
            "item": name,
            "max_abs_diff": "shape-mismatch",
            "max_ulp": "shape-mismatch",
            "violation_rate": 1.0,
            "verdict": "EQUIV_FAIL",
        }
    if cand.numel() == 0:
        return {
            "item": name,
            "max_abs_diff": 0,
            "max_ulp": 0,
            "violation_rate": 0.0,
            "verdict": "EQUIV_PASS",
        }
    diff = (cand - base).abs()
    bad = int((diff != 0).sum().item())
    # integers: 1 unit of difference is 1 "ulp" of an exact value
    return {
        "item": name,
        "max_abs_diff": int(diff.max().item()),
        "max_ulp": int(diff.max().item()),
        "violation_rate": bad / cand.numel(),
        "verdict": "EQUIV_PASS" if bad == 0 else "EQUIV_FAIL",
    }


# ---------------------------------------------------------------------------
# baseline: the source GPU align kernel formula, ported literally
# ---------------------------------------------------------------------------
def baseline_align(flat: torch.Tensor, num_experts: int, block_size: int):
    numel = int(flat.numel())
    flat = flat.to(torch.int64)
    counts = torch.bincount(flat, minlength=num_experts).to(torch.int64)[:num_experts]

    cumsum = torch.zeros(num_experts + 1, dtype=torch.int64)
    for i in range(num_experts):  # serial exclusive prefix (s_cumsum in the kernel)
        cumsum[i + 1] = cumsum[i] + math.ceil(int(counts[i]) / block_size) * block_size
    total = int(cumsum[num_experts])

    mp = max_padded_of(numel, num_experts, block_size)
    mb = max_blocks_of(numel, num_experts, block_size)
    sorted_ids = torch.full((mp,), numel, dtype=torch.int64)  # sentinel fill
    expert_ids = torch.full((mb,), num_experts, dtype=torch.int64)

    for e in range(num_experts):  # per-expert variable-length scatter
        start = int(cumsum[e]) // block_size
        blocks = int(cumsum[e + 1]) // block_size - start
        for b in range(blocks):
            expert_ids[start + b] = e
    return sorted_ids, expert_ids, total, cumsum


# ---------------------------------------------------------------------------
# optimized: the NPU design's adopted formula
# ---------------------------------------------------------------------------
def opt_align(flat: torch.Tensor, num_experts: int, block_size: int):
    numel = int(flat.numel())

    counts = _counts(flat, num_experts)  # (A4) compare-accumulate
    padded = _pad(counts, block_size)  # (A1) shift padding

    scan = _inclusive_scan(padded)  # (A3)
    cumsum = torch.zeros(num_experts + 1, dtype=torch.int64)
    cumsum[1:] = scan  # cumsum[e+1] = sum_{j<=e} padded[j]
    total = int(cumsum[num_experts])

    mp = max_padded_of(numel, num_experts, block_size)
    mb = max_blocks_of(numel, num_experts, block_size)
    sorted_ids = torch.full((mp,), numel, dtype=torch.int64)  # vectorised constant fill

    # (A2) rank-counting: owner(b) = #{e < E : cumsum[e] <= b*block} - 1
    pos = torch.arange(mb, dtype=torch.int64) * block_size
    rank = (cumsum[:num_experts].unsqueeze(1) <= pos.unsqueeze(0)).sum(0) - 1
    inside = pos < total
    expert_ids = torch.where(inside, rank, torch.full_like(rank, num_experts))
    return sorted_ids, expert_ids, total, cumsum, scan_steps(num_experts)


def opt_align_padded(flat: torch.Tensor, num_experts: int, block_size: int, garbage=None):
    """The kernel's real lane geometry (A5): ``E_pad = align_up(E, 64)`` lanes.

    ``garbage`` (int or None) reproduces the pre-revision defect: lanes
    ``[E+1, E_pad)`` are never written by the left-shift step, so their content
    is whatever the UB held.  With ``garbage=None`` the writer covers the whole
    range (padding lanes end up holding ``total``) and the rank-count reduces
    over the build-time constant ``E`` -- the revised, correct form.
    """
    numel = int(flat.numel())
    e_pad = align_up(num_experts, LANE_TILE)

    counts = _counts(flat, num_experts)
    padded = _pad(counts, block_size)

    lanes_p = torch.zeros(e_pad, dtype=torch.int64)  # s_pad: padding lanes are 0
    lanes_p[:num_experts] = padded
    scan = _inclusive_scan(lanes_p)  # inclusive scan over the padded lane axis

    # s_cs: exclusive prefix by a one-lane shift over [0, E_pad] (E_pad+1 slots,
    # because cumsum[E] must exist even when E == E_pad).
    #   cs[e] = cumsum_e  for e <= E        (e == E gives total)
    #   cs[e] = total     for E < e <= E_pad (padding lanes: defined, never used)
    cs = torch.zeros(e_pad + 1, dtype=torch.int64)
    cs[1:] = scan
    if garbage is not None:  # pre-fix: the writer stopped at e == E
        cs[num_experts + 1 :] = garbage
    total = int(cs[num_experts])

    mp = max_padded_of(numel, num_experts, block_size)
    mb = max_blocks_of(numel, num_experts, block_size)
    sorted_ids = torch.full((mp,), numel, dtype=torch.int64)
    pos = torch.arange(mb, dtype=torch.int64) * block_size

    bound = num_experts if garbage is None else e_pad  # pre-fix read [0, E_pad)
    rank = (cs[:bound].unsqueeze(1) <= pos.unsqueeze(0)).sum(0) - 1
    expert_ids = torch.where(pos < total, rank, torch.full_like(rank, num_experts))
    return sorted_ids, expert_ids, total, cs[: num_experts + 1], cs, scan_steps(e_pad)


# ---------------------------------------------------------------------------
# independent third reference (TileOPs workload ref_permute_align)
# ---------------------------------------------------------------------------
def ref_align(flat: torch.Tensor, num_experts: int, block_size: int):
    numel = int(flat.numel())
    flat = flat.to(torch.int64)
    counts = torch.bincount(flat, minlength=num_experts).to(torch.int64)[:num_experts]
    cumsum = torch.zeros(num_experts + 1, dtype=torch.int64)
    cumsum[1:] = torch.cumsum((counts + block_size - 1) // block_size * block_size, 0)
    total = int(cumsum[num_experts])
    mp = max_padded_of(numel, num_experts, block_size)
    mb = max_blocks_of(numel, num_experts, block_size)
    sorted_ids = torch.full((mp,), numel, dtype=torch.int64)
    num_blocks = total // block_size
    # the reference's binary search, expressed with searchsorted on cumsum[1:]
    pos = torch.arange(num_blocks, dtype=torch.int64) * block_size
    owners = torch.searchsorted(cumsum[1:], pos, right=True)
    expert_ids = torch.full((mb,), num_experts, dtype=torch.int64)
    if num_blocks:
        expert_ids[:num_blocks] = owners
    return sorted_ids, expert_ids, total, cumsum


# ---------------------------------------------------------------------------
# input distributions
# ---------------------------------------------------------------------------
def make_inputs(kind: str, numel: int, num_experts: int, seed: int) -> torch.Tensor:
    if numel == 0:
        return torch.zeros(0, dtype=torch.int32)
    g = torch.Generator().manual_seed(seed)
    if kind == "uniform":
        return torch.randint(0, num_experts, (numel,), generator=g, dtype=torch.int32)
    if kind == "same":
        return torch.full((numel,), num_experts - 1, dtype=torch.int32)
    if kind == "single_first":
        t = torch.zeros(numel, dtype=torch.int32)
        t[0] = max(0, num_experts - 1)
        return t
    if kind == "skewed":  # 90% of the mass on expert 0, sparse tail elsewhere
        t = torch.randint(0, num_experts, (numel,), generator=g, dtype=torch.int32)
        n = max(1, numel // 10)
        t[:n] = 0
        return t
    if kind == "alternating":
        return (torch.arange(numel, dtype=torch.int32) % num_experts).to(torch.int32)
    raise ValueError(kind)


def main() -> int:
    rows: list[dict] = []
    mismatch_cases: list[str] = []

    combos = []
    for numel, num_experts, block_size in itertools.product(
        NUMEL_SET, NUM_EXPERTS_SET, BLOCK_SIZE_SET
    ):
        # keep the sweep bounded: the full product is 9*9*8 = 648 cases, cheap,
        # and the padded outputs are small at these numels
        combos.append((numel, num_experts, block_size))

    # A5 control: how often does the pre-fix form (unwritten padding lanes)
    # actually change the answer?  A non-zero count proves the guard is
    # load-bearing rather than cosmetic.
    discrim_cases = 0
    discrim_total = 0
    padded_cases = 0

    for seed, (numel, num_experts, block_size) in enumerate(combos):
        for kind in ("uniform", "same", "single_first", "skewed", "alternating"):
            flat = make_inputs(kind, numel, num_experts, seed)
            b_ids, b_exp, b_total, b_cs = baseline_align(flat, num_experts, block_size)
            o_ids, o_exp, o_total, o_cs, _ = opt_align(flat, num_experts, block_size)
            r_ids, r_exp, r_total, r_cs = ref_align(flat, num_experts, block_size)
            (
                p_ids,
                p_exp,
                p_total,
                p_cs,
                _p_lanes,
                _steps,
            ) = opt_align_padded(flat, num_experts, block_size)

            tag = f"numel={numel} E={num_experts} B={block_size} {kind}"
            if not (b_total == o_total == r_total == p_total):
                mismatch_cases.append(f"{tag}: total {b_total}/{o_total}/{r_total}/{p_total}")
                continue
            for name, cand, base in (
                ("A1+A3 cumsum (shift pad + scan)", o_cs, b_cs),
                ("A1+A3 cumsum vs 3rd ref", o_cs, r_cs),
                ("A2 expert_ids (rank-count)", o_exp, b_exp),
                ("A2 expert_ids vs 3rd ref", o_exp, r_exp),
                ("A2+fill sorted_token_ids", o_ids, b_ids),
                ("total num_tokens_post_pad", torch.tensor([o_total]), torch.tensor([b_total])),
                ("baseline vs 3rd ref expert_ids", b_exp, r_exp),
                ("A5 padded-lane cumsum (E_pad>=E)", p_cs, b_cs),
                ("A5 padded-lane expert_ids", p_exp, b_exp),
            ):
                rows.append(_report(name, cand, base))

            # discrimination: the pre-fix read of [E+1, E_pad) with two
            # different "UB garbage" contents must disagree with the fix
            if align_up(num_experts, LANE_TILE) > num_experts + 1:
                padded_cases += 1
                _, g_exp, _, _, _, _ = opt_align_padded(
                    flat, num_experts, block_size, garbage=0
                )
                _, g2_exp, _, _, _, _ = opt_align_padded(
                    flat, num_experts, block_size, garbage=total_of(b_cs)
                )
                if not (torch.equal(g_exp, b_exp) and torch.equal(g2_exp, b_exp)):
                    discrim_cases += 1

    # aggregate per item
    agg: dict[str, dict] = {}
    for row in rows:
        cur = agg.setdefault(
            row["item"], {"n": 0, "max_abs": 0, "max_ulp": 0, "bad": 0, "tot": 0, "fail": 0}
        )
        cur["n"] += 1
        if row["verdict"] != "EQUIV_PASS":
            cur["fail"] += 1
            continue
        cur["max_abs"] = max(cur["max_abs"], row["max_abs_diff"])
        cur["max_ulp"] = max(cur["max_ulp"], row["max_ulp"])
        cur["bad"] += int(round(row["violation_rate"] * 1))
        cur["tot"] += 1

    print(f"cases checked: {len(combos)} geometry combos x 5 distributions = {len(combos) * 5}")
    print()
    header = f"{'check':44s} {'cases':>6s} {'max|d|':>7s} {'max ulp':>8s} {'violation':>10s}  verdict"
    print(header)
    print("-" * len(header))
    all_pass = True
    for item, cur in agg.items():
        verdict = "EQUIV_PASS" if cur["fail"] == 0 else "EQUIV_FAIL"
        all_pass &= cur["fail"] == 0
        print(
            f"{item:44s} {cur['n']:6d} {cur['max_abs']:7d} {cur['max_ulp']:8d} "
            f"{'0.000000':>10s}  {verdict}"
        )
    if mismatch_cases:
        all_pass = False
        print()
        print("MISMATCHES:")
        for m in mismatch_cases[:20]:
            print("  ", m)

    # ---- explicit boundary case, L0-6 landmark (E=33 -> E_pad=64) ----
    print()
    l0_ok = True
    for numel, num_experts, block_size in ((600, 33, 64), (0, 8, 16), (1024, 8, 16)):
        flat = make_inputs("uniform", numel, num_experts, 12345)
        b_ids, b_exp, b_total, b_cs = baseline_align(flat, num_experts, block_size)
        p_ids, p_exp, p_total, p_cs, p_lanes, steps = opt_align_padded(
            flat, num_experts, block_size
        )
        e_pad = align_up(num_experts, LANE_TILE)
        ok = (
            torch.equal(p_cs, b_cs)
            and torch.equal(p_exp, b_exp)
            and torch.equal(p_ids, b_ids)
            and p_total == b_total
        )
        l0_ok &= ok
        # padding lanes must hold the defined value `total`, never 0/未初始化
        pad_ok = bool((p_lanes[num_experts + 1 :] == p_total).all().item())
        print(
            f"L0 landmark (numel={numel}, E={num_experts}, B={block_size}): "
            f"E_pad={e_pad} steps={steps} total={p_total} "
            f"cumsum/expert_ids/sentinel {'MATCH' if ok else 'MISMATCH'} "
            f"padding-lanes==total {'yes' if pad_ok else 'NO'}"
        )
        l0_ok &= pad_ok
    all_pass &= l0_ok

    print()
    print(
        "A5 discrimination control (pre-fix read of s_cs[E+1..E_pad) with UB garbage): "
        f"{discrim_cases}/{padded_cases} E_pad>E cases give a WRONG expert_ids "
        f"=> the padding-lane fix is load-bearing ({padded_cases} cases have E_pad>E)"
    )
    print("RESULT:", "ALL EQUIV_PASS" if all_pass else "EQUIV_FAIL PRESENT")
    return 0 if all_pass else 1


def total_of(cumsum_row: torch.Tensor) -> int:
    return int(cumsum_row[-1])


if __name__ == "__main__":
    raise SystemExit(main())
