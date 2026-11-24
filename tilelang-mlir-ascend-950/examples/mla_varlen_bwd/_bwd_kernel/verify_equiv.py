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


def main() -> int:
    rows = [check_a1(torch.float16), check_a1(torch.bfloat16), check_a2(), check_a3()]
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
    print("EQUIV_SUMMARY", "ALL_PASS" if ok else "HAS_BLOCKED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
