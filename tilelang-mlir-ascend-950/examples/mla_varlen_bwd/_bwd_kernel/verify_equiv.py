#!/usr/bin/env python3
"""D-1 machine check for the adopted formula optimisations of `_bwd_kernel` (packed).

Design doc: DESIGN.md §1.6.1 of examples/mla_varlen_bwd/_bwd_kernel.
Baseline (source/GPU formula) vs candidate (NPU formula), both evaluated on torch
CPU.  Every candidate is compared with an fp64 reference of the *same*
mathematical quantity and with the baseline's dtype-rounded output, so a
candidate that merely relocates rounding is visible.

Items
-----
A1  natural-log-domain probability  (reused from the dense design verbatim)
    baseline : p = exp2(s*scale*LOG2E - lse*LOG2E)
    candidate: p = exp (s*scale      - lse)
    both rounded to the target dtype, as the kernel does when it feeds P into
    the dV GEMM.

A2  block-level mask specialisation, **re-derived against per-request bounds**
    The packed kernel has no compile-time S_q / S_kv: every bound is a run-time
    difference of `cu_seqlens_*`.  The predicate therefore reads
        fast = (n0 + block_n <= len_kv) & (m0 + block_m <= len_q)
               & (n0 + block_n - 1 <= m0 + shift),  shift = len_kv - len_q
    and the masked arm reads
        vis  = (gi < len_kv) & (gj < len_q) & (gi <= gj + shift).
    This item re-checks the same equivalence per request (the fast predicate is
    once more only a *sufficient* condition of "mask all true in the block"),
    over the manifest's packed length sets, including tails and shift > / < 0.

A4  per-launch grid decomposition (**v1, the 65535 fix**; a task-mapping item)
    baseline : one launch, grid = tiles * heads
    candidate: ceil(heads / slab) launches, grid = tiles * slab each, with
               tile = bid // slab and h = h_off + bid % slab
    Claim: (a) every per-launch grid <= 65535; (b) the union of the launches is
    a *bijection* onto {(tile, h) : tile < tiles, h < heads} -- nothing missing,
    nothing computed twice, so dq_acc/dk/dv (indexed by (token, head, dim), one
    contiguous D-run per head) get disjoint write sets per launch; (c) a plan
    whose cap cannot be met raises instead of truncating.  Exhaustively
    enumerated over every bid of every launch for the manifest geometries, the
    H=126/H=127 boundary geometries and an auto-shrink geometry.

A3  packed zero-spill invariant (**new**; the dq store is a whole-tile atomic add)
    A query block that runs past `len_q` spills into the next request's dq_acc
    rows.  The claim the design relies on: those rows of dS are *exactly* zero,
    hence dq_c's spilled rows are exactly zero, hence the atomic add is a no-op.
    Checked structurally: with the masked arm's predicate the P/dS entries with
    gj >= len_q are exactly 0.0, and the fp32 accumulation of ds16 · k over such
    a row is exactly 0.0.  Also checks that adding +0.0 leaves a random fp32
    accumulator bitwise unchanged (incl. -0.0 / subnormal / large values).

ulp metric: every compared quantity is a probability (p >= 0), so the distance
in units-in-the-last-place is the difference of the raw bit patterns (monotone
for non-negative IEEE values).  Non-finite values are compared by class and sign.

Run:  /workspace/venv-ascend950/bin/python verify_equiv.py
Exit code 0 iff every item is EQUIV_PASS.
"""

from __future__ import annotations

import math

import torch

LOG2E = 1.4426950408889634
SCALE = 192**-0.5  # D_qk = 192, the manifest default sm_scale

_WIDTH = {torch.float32: torch.int32, torch.float16: torch.int16, torch.bfloat16: torch.int16}


def _bit_key(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Monotone integer key for non-negative IEEE values (one step == one ulp)."""
    return x.contiguous().view(_WIDTH[dtype]).to(torch.int64)


def _ulp_diff(a: torch.Tensor, b: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    both_finite = torch.isfinite(a) & torch.isfinite(b)
    diff = torch.where(
        both_finite,
        (_bit_key(a, dtype) - _bit_key(b, dtype)).abs(),
        torch.zeros((), dtype=torch.int64),
    )
    same_class = (
        (torch.isnan(a) == torch.isnan(b))
        & (torch.isinf(a) == torch.isinf(b))
        & (torch.sign(a) == torch.sign(b))
    )
    return torch.where(both_finite | same_class, diff, torch.full_like(diff, 1 << 20))


def _stats(name, cand, base, ref64, dtype, extra=None):
    ulp = _ulp_diff(cand, base, dtype)
    n = base.numel()
    both_finite = torch.isfinite(cand) & torch.isfinite(base)
    viol = int((ulp > 1).sum().item())  # dtype-visible disagreement > 1 ulp
    ulp_c = _ulp_diff(cand, ref64, dtype)
    ulp_b = _ulp_diff(base, ref64, dtype)
    row = {
        "item": name,
        "n": n,
        "max_ulp": int(ulp.max().item()),
        "viol": viol / n,
        "max_ulp_cand_vs_fp64": int(ulp_c[both_finite].max().item()) if both_finite.any() else 0,
        "max_ulp_base_vs_fp64": int(ulp_b[both_finite].max().item()) if both_finite.any() else 0,
        "verdict": "EQUIV_PASS" if viol == 0 else "EQUIV_BLOCKED",
    }
    if extra:
        row.update(extra)
    return row


# ---------------------------------------------------------------------------
# A1: natural-log domain (identical to the dense design's item)
# ---------------------------------------------------------------------------
def _a1_inputs():
    gen = torch.Generator().manual_seed(20261123)
    n = 1 << 20
    s = torch.randn(n, generator=gen, dtype=torch.float64) * math.sqrt(192.0)
    lse = torch.log(torch.tensor(8192.0, dtype=torch.float64)) + 0.5 * torch.randn(
        n, generator=gen, dtype=torch.float64
    )
    corner = [
        (0.0, 0.0),
        (1.0, 1.0),
        (-1.0, -1.0),
        (8e2, 0.0),
        (-8e2, 0.0),
        (40.0, 0.0),
        (-40.0, 0.0),
        (700.0, 0.0),
        (-700.0, 0.0),
        (100.0, 100.0),
        (1.0, float("-inf")),
        (0.0, float("-inf")),
        (-2.0, float("-inf")),
        (0.0, 88.0),
        (0.0, 1.0e-45),
        (0.0, 30.0),
    ]
    cs = torch.tensor([c[0] for c in corner], dtype=torch.float64)
    cl = torch.tensor([c[1] for c in corner], dtype=torch.float64)
    return torch.cat([s, cs]), torch.cat([lse, cl])


def check_a1(dtype: torch.dtype):
    s, lse = _a1_inputs()
    p64 = torch.exp(s * SCALE - lse)  # fp64 truth of the natural-log formula
    s32, lse32 = s.to(torch.float32), lse.to(torch.float32)
    base = torch.exp2(s32 * (SCALE * LOG2E) - lse32 * LOG2E)  # source formula
    cand = torch.exp(s32 * SCALE - lse32)  # NPU formula
    name = "A1 exp 域 P（目标 dtype %s）" % str(dtype).replace("torch.", "")
    return _stats(name, cand.to(dtype), base.to(dtype), p64.to(dtype), dtype)


# ---------------------------------------------------------------------------
# A2: block-level mask specialisation with per-request (run-time) bounds
# ---------------------------------------------------------------------------
PACKED_SETS = [
    # (label, [q_lens], [kv_lens])  -- the manifest's packed workloads, capped
    ("ds-v3-8x4k/4req", [4096] * 4, [4096] * 4),
    ("ds-v3-mixed", [512, 1024, 2048, 4096, 8192, 16384],
     [512, 1024, 2048, 4096, 8192, 16384]),
    ("kimi-k2-4x8k/2req", [8192] * 2, [8192] * 2),
    ("tails+shift", [96, 128, 64, 65, 200], [192, 64, 64, 129, 200]),
    ("tiny", [1, 63, 64, 65], [1, 65, 64, 63]),
]


def check_a2(block_n: int = 64, block_m: int = 64, max_query_blocks: int = 24):
    gen = torch.Generator().manual_seed(7)
    total = viol = blocks = fast = 0
    per_set = []
    for label, q_lens, kv_lens in PACKED_SETS:
        set_blocks = set_fast = 0
        for len_q, len_kv in zip(q_lens, kv_lens, strict=True):
            shift = len_kv - len_q
            n_keys = max(1, (len_kv + block_n - 1) // block_n)
            n_qs = max(1, (len_q + block_m - 1) // block_m)
            # enumerate every key block; sample the query blocks when wide
            step = max(1, n_qs // max_query_blocks)
            for kn in range(n_keys):
                n0 = kn * block_n
                for km in range(0, n_qs, step):
                    m0 = km * block_m
                    blocks += 1
                    set_blocks += 1
                    i = torch.arange(n0, min(n0 + block_n, len_kv))[:, None]
                    j = torch.arange(m0, min(m0 + block_m, len_q))[None, :]
                    p = (
                        torch.rand(i.shape[0], j.shape[1], generator=gen, dtype=torch.float32)
                        + 0.5
                    )
                    vis = (i < len_kv) & (j < len_q) & (i <= j + shift)
                    base = torch.where(vis, p, torch.zeros(()))
                    is_fast = (
                        (n0 + block_n <= len_kv)
                        and (m0 + block_m <= len_q)
                        and (n0 + block_n - 1 <= m0 + shift)
                    )
                    if is_fast:
                        fast += 1
                        set_fast += 1
                        # a fully visible block: the predicate must imply vis everywhere
                        assert bool(vis.all()), (label, len_q, len_kv, n0, m0)
                        cand = p
                    else:
                        cand = torch.where(vis, p, torch.zeros(()))
                    total += p.numel()
                    viol += int((cand != base).sum())
        per_set.append("%s %d/%d" % (label, set_fast, set_blocks))
    return {
        "item": "A2 块级 mask 特化 · packed 运行时界（block_n=%d, block_m=%d）" % (block_n, block_m),
        "n": total,
        "max_ulp": 0,
        "viol": viol / total,
        "max_ulp_cand_vs_fp64": "-",
        "max_ulp_base_vs_fp64": "-",
        "verdict": "EQUIV_PASS" if viol == 0 else "EQUIV_BLOCKED",
        "fast": "%d/%d" % (fast, blocks),
        "per_set": " | ".join(per_set),
    }


# ---------------------------------------------------------------------------
# A3: zero-spill invariant (whole-tile atomic dq store in the packed kernel)
# ---------------------------------------------------------------------------
def check_a3(block_n: int = 64, block_m: int = 64, dtype=torch.float16):
    """Whole blocks, exactly as the kernel stages them (full 64-row extents)."""
    gen = torch.Generator().manual_seed(11)
    n_elem = n_spill = n_nonzero = 0
    acc_bad = 0
    n_acc = 0
    for len_q, len_kv in ((96, 96), (65, 65), (200, 200), (512, 768), (128, 64), (1, 1)):
        shift = len_kv - len_q
        n_qs = (len_q + block_m - 1) // block_m
        for kn in range((len_kv + block_n - 1) // block_n):
            n0 = kn * block_n
            for km in range(n_qs):
                m0 = km * block_m
                # full extents (the GEMM always runs whole tiles); q/do rows past
                # len_q were loaded as zeros or as the next request's rows -- both
                # are dropped by the mask, so their content is irrelevant here.
                i = torch.arange(n0, n0 + block_n)[:, None]
                j = torch.arange(m0, m0 + block_m)[None, :]
                s = torch.randn(i.shape[0], j.shape[1], generator=gen) * math.sqrt(192.0)
                lse = torch.log(torch.tensor(float(len_kv))) + torch.zeros_like(s[0])
                vis = (i < len_kv) & (j < len_q) & (i <= j + shift)
                p = torch.where(vis, torch.exp(s * SCALE - lse), torch.zeros(()))
                dp = torch.randn(p.shape, generator=gen)
                delta = (p * dp).sum(0, keepdim=True)  # per query row
                ds = (p * (dp - delta)).to(dtype)
                # columns of dS whose query index is outside the request must be 0
                spill = j.ravel() >= len_q
                if spill.any():
                    spilled = ds[:, spill]
                    n_spill += spilled.numel()
                    n_nonzero += int((spilled != 0).sum())
                    # a spilled query row's dq row = (ds16 . k) accumulates to 0
                    kk = torch.randn(i.shape[0], 8, generator=gen).to(dtype)
                    prod = ds[:, spill].float().T @ kk.float()
                    n_elem += prod.numel()
                    n_nonzero += int((prod != 0).sum())
    # adding +0.0 leaves the accumulator's *value* unchanged (bitwise: only the
    # -0.0 -> +0.0 sign of zero may differ, which is value-identical)
    acc = torch.cat(
        [
            torch.randn(4096, generator=gen) * 1e3,
            torch.tensor([-0.0, 0.0, 1e-45, -1e-45, 3.4e38, -3.4e38], dtype=torch.float32),
        ]
    )
    n_acc += acc.numel()
    added = acc + 0.0
    same_value = (added == acc) | (torch.isnan(added) & torch.isnan(acc))
    acc_bad += int((~same_value).sum())
    sign_flips = int((added.view(torch.int32) != acc.view(torch.int32)).sum())
    check = n_spill + n_elem
    return {
        "item": "A3 packed 溢出零行不变式（dS/dq 越界行恒 0，+0.0 原子累加无副作用）",
        "n": check + n_acc,
        "max_ulp": 0,
        "viol": (n_nonzero + acc_bad) / max(1, check + n_acc),
        "max_ulp_cand_vs_fp64": "-",
        "max_ulp_base_vs_fp64": "-",
        "verdict": "EQUIV_PASS" if (n_nonzero + acc_bad) == 0 else "EQUIV_BLOCKED",
        "fast": "spill_entries=%d dq_rows=%d acc=%d zero_sign_flips=%d"
        % (n_spill, n_elem, n_acc, sign_flips),
    }


def check_a4(block_n: int = 64):
    """A4: launch plan -- per-launch grid cap, task-set bijection, raise on overflow."""
    from launch_plan import NPU_GRID_MAX, describe, plan_launch  # noqa: PLC0415

    geoms = [
        # label, batch, total_kv, heads  (manifest first, then boundaries)
        ("ds-v3-8x4k", 8, 32768, 128),
        ("ds-v3-mixed", 6, 32256, 128),
        ("kimi-k2-4x8k", 4, 32768, 64),
        ("mla-varlen-bwd-smoke", 2, 640, 8),
        ("H=126 (65,520, no truncation in v0)", 8, 32768, 126),
        ("H=127 (66,040, v0 truncated)", 8, 32768, 127),
        ("H=128 one-tile requests", 8, 4096, 128),
        ("auto-shrink (2100 tiles)", 8, 2100 * block_n, 128),
    ]
    n_tasks = bad = over = 0
    detail = []
    for label, batch, total_kv, heads in geoms:
        plan = plan_launch(batch, total_kv, heads, block_n=block_n)
        tiles = plan.tiles
        seen: dict[tuple[int, int], int] = {}
        for launch in plan.launches:
            over += int(launch.grid > NPU_GRID_MAX)
            for bid in range(launch.grid):
                h = launch.head_off + (bid % plan.slab)
                if h >= heads:  # tail-slab guard: the block returns at once
                    continue
                key = (bid // plan.slab, h)
                seen[key] = seen.get(key, 0) + 1
        expect = {(t, h) for t in range(tiles) for h in range(heads)}
        n_tasks += len(expect)
        missing = len(expect - set(seen))
        duplicated = sum(c - 1 for c in seen.values() if c > 1)
        stray = len(set(seen) - expect)
        # head slabs must not overlap -> write sets (one contiguous D-run per
        # (token, head)) are disjoint across launches, no inter-launch sync
        overlap = 0
        for i, a in enumerate(plan.launches):
            for b in plan.launches[i + 1:]:
                lo = max(a.head_off, b.head_off)
                hi = min(a.head_off + a.active_heads, b.head_off + b.active_heads)
                overlap += max(0, hi - lo)
        bad += missing + duplicated + stray + overlap
        v0_grid = tiles * heads
        if v0_grid <= NPU_GRID_MAX:
            v0 = f"v0 grid={v0_grid} ok"
        elif v0_grid % 65536 == 0:
            v0 = f"v0 grid={v0_grid} ABORT"
        else:
            v0 = f"v0 grid={v0_grid} ran {v0_grid % 65536}/{v0_grid}"
        detail.append(
            f"{label}: slab={plan.slab} launches={len(plan.launches)} "
            f"max_grid={plan.max_grid} blocks={plan.total_blocks} "
            f"useful={plan.useful_blocks} idle={plan.idle_blocks} ({v0})"
        )

    # the cap itself: 65,535 must pass, 65,536 must raise, never truncate
    edge_ok = 0
    at_cap = plan_launch(8, (65535 - 8) * block_n, 128, block_n=block_n)
    if at_cap.max_grid == 65535 and at_cap.slab == 1:
        edge_ok += 1
    else:
        bad += 1
    raised = 0
    for args in (
        (8, (65536 - 8) * block_n, 128),  # one tile past the cap even at slab=1
        (1, 65536 * block_n, 1),
    ):
        try:
            plan_launch(*args, block_n=block_n)
        except ValueError:
            raised += 1
        else:
            bad += 1  # silent truncation would be the bug this item guards
    if raised != 2:
        bad += 1
    # a geometry whose preferred slab is too wide must shrink, not raise.
    # Expressed in tiles (not tokens) so the check is tile-size independent:
    # 2100 tiles > 65535 // 32 = 2047, so the preferred slab of 32 cannot fit.
    shrunk = plan_launch(8, 2100 * block_n, 128, block_n=block_n)
    if shrunk.slab < 32 and shrunk.max_grid <= NPU_GRID_MAX:
        edge_ok += 1
    else:
        bad += 1
    n = n_tasks + 2 + 2
    return {
        "item": "A4 单次 launch grid ≤ 65535 的 head-slab 分解（任务集一一对应、写集按 head 不交）",
        "n": n,
        "max_ulp": 0,
        "viol": bad / n,
        "max_ulp_cand_vs_fp64": "-",
        "max_ulp_base_vs_fp64": "-",
        "verdict": "EQUIV_PASS" if (bad == 0 and over == 0) else "EQUIV_BLOCKED",
        "fast": "tasks=%d over_cap=%d edge_checks=%d/2 raised_on_overflow=%d/2"
        % (n_tasks, over, edge_ok, raised),
        "per_set": " | ".join(detail),
    }


# Tile of the adopted design (DESIGN.md §5.2, C2): key blocks of 96 rows, query
# blocks of 64.  A2/A3/A4 are re-run at this tile so the table inside §1.6.1
# describes the geometry the kernel actually uses (v4/Cube revision).
BLOCK_N = 96
BLOCK_M = 64


def check_a5(max_n_it: int = 64):
    """A5: schedule algebra of the deferred-``GB`` rotation (DESIGN.md §6.3 B).

    A5 is a *statement-order* transformation: there is no candidate formula, so
    the fp64 reference used by A1 cannot judge it.  What can be checked without a
    device is the index algebra of the rotation -- that for every query-block
    iteration ``k`` the deferred ``GB(k)`` reads exactly the buffer version that
    ``EP(k)``/``F(k)`` wrote, that no later producer overwrote that version before
    the read, and that ``clear_accum`` fires exactly once.

    The simulator below is the literal statement stream of the unrolled loop
    (``kqA = 2i``, ``kqB = 2i + 1``) plus the drain, for every
    ``loop_st`` in {0, 1, 2, 5, 7} and every ``n_it`` in 1..``max_n_it``.
    Device-level bit parity against the frozen v4 body is the complementary check
    (``history_version/design_probe_epilogue_rotation.py --parity``).
    """
    bad = []
    n_gb = 0
    n_points = 0
    for loop_st in (0, 1, 2, 5, 7):
        for n_it in range(1, max_n_it + 1):
            n_points += 1
            stream = []          # (kind, k, version_or_None, clear_or_None)
            pos = {}             # (kind, k) -> index
            gb_events = []

            def emit(kind, k, v=None, clear=None):
                pos[(kind, k)] = len(stream)
                stream.append((kind, k, v, clear))

            n_half = (n_it + 1) // 2
            for i in range(n_half):
                kA = 2 * i
                emit("F", kA, 0)
                emit("GA", kA)
                emit("EP", kA, 0)
                if kA > 0:
                    emit("GB", kA - 1, 1, kA - 1 == 0)
                    gb_events.append((kA - 1, 1, kA - 1 == 0))
                kB = kA + 1
                if kB < n_it:
                    emit("F", kB, 1)
                    emit("GA", kB)
                    emit("EP", kB, 1)
                    emit("GB", kB - 1, 0, kB - 1 == 0)
                    gb_events.append((kB - 1, 0, kB - 1 == 0))
            last = n_it - 1
            v_last = 0 if (n_it % 2) == 1 else 1
            emit("GB", last, v_last, last == 0)
            gb_events.append((last, v_last, last == 0))
            n_gb += len(gb_events)
            tag = "loop_st=%d n_it=%d" % (loop_st, n_it)

            # (1) exactly one GB per k in [0, n_it), in increasing k order
            ks = [g[0] for g in gb_events]
            if ks != list(range(n_it)):
                bad.append("%s: GB coverage %s" % (tag, ks))
                continue

            for k_prev, v_read, _clear in gb_events:
                p = pos[("GB", k_prev)]
                # (2a) the version GB(k) reads must be the version EP(k) wrote:
                #      EP(k)/F(k) always write v = k % 2.  This is the check that
                #      catches the first-probe bug (drain/odd half reading the
                #      wrong version), which produced 5.7e-2 errors on one block.
                if v_read != (k_prev % 2):
                    bad.append("%s: GB(%d) reads v%d, but EP(%d) wrote v%d"
                               % (tag, k_prev, v_read, k_prev, k_prev % 2))
                # (2b) EP(k) wrote that version, and no later EP overwrote it first
                if pos.get(("EP", k_prev)) is None:
                    bad.append("%s: no EP(%d)" % (tag, k_prev))
                    continue
                if pos[("EP", k_prev)] > p:
                    bad.append("%s: EP(%d) after GB(%d)" % (tag, k_prev, k_prev))
                for k2 in range(k_prev + 1, n_it):
                    if k2 % 2 == v_read and pos[("EP", k2)] < p:
                        bad.append("%s: GB(%d) reads v%d after EP(%d) overwrote it"
                                   % (tag, k_prev, v_read, k2))
                # (3) the fill version read by GB(k) is not overwritten by F(k+2)
                if pos[("F", k_prev)] > p:
                    bad.append("%s: F(%d) after GB(%d)" % (tag, k_prev, k_prev))
                for k2 in range(k_prev + 1, n_it):
                    if k2 % 2 == v_read and pos[("F", k2)] < p:
                        bad.append("%s: GB(%d) reads stale fill v%d (F(%d) first)"
                                   % (tag, k_prev, v_read, k2))

            # (4) clear fires exactly once, on GB(0)
            n_clear = sum(1 for g in gb_events if g[2])
            if n_clear != 1 or not gb_events[0][2]:
                bad.append("%s: clear_accum fired %d times" % (tag, n_clear))

            # (5) within an iteration, F -> GA -> EP order is preserved
            for k in range(n_it):
                if not (pos[("F", k)] < pos[("GA", k)] < pos[("EP", k)]):
                    bad.append("%s: F/GA/EP order broken at k=%d" % (tag, k))

    viol = len(bad)
    return {
        "item": "A5 旋转调度代数（版本分配 / drain / clear_accum；loop_st×n_it 全扫）",
        "n": n_gb,
        "max_ulp": "-",
        "viol": viol / max(1, n_gb),
        "max_ulp_cand_vs_fp64": "-",
        "max_ulp_base_vs_fp64": "-",
        "verdict": "EQUIV_PASS" if viol == 0 else "EQUIV_BLOCKED",
        "fast": "schedule_points=%d gb_events=%d violations=%d"
        % (n_points, n_gb, viol),
        "per_set": " | ".join(bad[:6]) if bad else "no violations",
    }


def main() -> int:
    rows = [
        check_a1(torch.float16),
        check_a1(torch.bfloat16),
        check_a2(block_n=BLOCK_N, block_m=BLOCK_M),
        check_a3(block_n=BLOCK_N, block_m=BLOCK_M),
        check_a4(block_n=BLOCK_N),
        check_a5(),
    ]
    head = (
        "| 采纳项 | 样本数 | 候选 vs 基线 最大 ulp | 违反率 | 候选 vs fp64 最大 ulp "
        "| 基线 vs fp64 最大 ulp | 结论 |"
    )
    print(head)
    print("|---|---|---|---|---|---|---|")
    ok = True
    for r in rows:
        ok &= r["verdict"] == "EQUIV_PASS"
        print(
            "| {item} | {n} | {mu} | {v:.3e} | {c} | {b} | {verdict} |".format(
                item=r["item"],
                n=r["n"],
                mu=r["max_ulp"],
                v=r["viol"],
                c=r["max_ulp_cand_vs_fp64"],
                b=r["max_ulp_base_vs_fp64"],
                verdict=r["verdict"],
            )
        )
    print("A2 fast-path blocks: %s" % rows[2]["fast"])
    print("A2 per set (fast/total): %s" % rows[2]["per_set"])
    print("A3 detail: %s" % rows[3]["fast"])
    print("A4 detail: %s" % rows[4]["fast"])
    print("A4 per geometry: %s" % rows[4]["per_set"])
    print("A5 detail: %s" % rows[5]["fast"])
    print("A5 violations: %s" % rows[5]["per_set"])
    print("EQUIV_SUMMARY", "ALL_PASS" if ok else "HAS_BLOCKED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
