#!/usr/bin/env python3
"""D-1 machine check for the adopted formula optimisations of _bwd_kernel.

Design doc: DESIGN.md §1.6.1.  Baseline (source/GPU formula) vs candidate
(NPU formula), both evaluated on torch CPU.  Each candidate is compared with an
fp64 reference of the *same* mathematical quantity and with the baseline's
dtype-rounded output, so a candidate that merely relocates rounding is visible.

Items
-----
A1  natural-log-domain probability
    baseline : p = exp2(s*scale*LOG2E - lse*LOG2E)
    candidate: p = exp (s*scale      - lse)
    both then rounded to the target dtype (float16 / bfloat16), as the kernel
    does when it feeds P into the dV GEMM.

A2  block-level mask specialisation
    baseline : p = select(mask, p, 0) on every (key, query) block
    candidate: p unchanged when the block is provably fully visible
               (n0 + block_n - 1 < S_kv and m0 + block_m - 1 < S_q and
                n0 + block_n - 1 <= m0 + shift), else the same select.

ulp metric: every compared quantity is a probability (p >= 0), so the
distance in units-in-the-last-place is the difference of the raw bit patterns
(monotone for non-negative IEEE values), divided by nothing — one bit step is
one ulp.  Non-finite values are compared by class and sign.

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
    diff = torch.where(both_finite, (_bit_key(a, dtype) - _bit_key(b, dtype)).abs(),
                       torch.zeros((), dtype=torch.int64))
    # same non-finite class (inf vs inf, nan vs nan) counts as 0 ulp, otherwise
    # a large but finite marker is reported
    same_class = (torch.isnan(a) == torch.isnan(b)) & (torch.isinf(a) == torch.isinf(b)) & (
        torch.sign(a) == torch.sign(b)
    )
    diff = torch.where(both_finite | same_class, diff, torch.full_like(diff, 1 << 20))
    return diff


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


def _a1_inputs():
    gen = torch.Generator().manual_seed(20261123)
    n = 1 << 20
    # realistic: s = q.k over D_qk=192 with unit-normal entries -> N(0, 192)
    s = torch.randn(n, generator=gen, dtype=torch.float64) * math.sqrt(192.0)
    # a row lse of 8192 scores: log(sum exp(.)) ~ log(S) + O(1)
    lse = torch.log(torch.tensor(8192.0, dtype=torch.float64)) + 0.5 * torch.randn(
        n, generator=gen, dtype=torch.float64
    )
    corner = [
        # (s, lse)  -- extreme exponents, empty row, exact cancellation, subnormal
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


def check_a2(block_n: int = 64, block_m: int = 64):
    gen = torch.Generator().manual_seed(7)
    total = viol = blocks = fast = 0
    for s_q, s_kv in ((512, 512), (8192, 8192), (96, 192), (128, 64), (64, 64), (3072, 3072)):
        shift = s_kv - s_q
        for n0 in range(0, s_kv, block_n):
            for m0 in range(0, s_q, block_m):
                blocks += 1
                i = torch.arange(n0, min(n0 + block_n, s_kv))[:, None]
                j = torch.arange(m0, min(m0 + block_m, s_q))[None, :]
                p = torch.rand(i.shape[0], j.shape[1], generator=gen, dtype=torch.float32) + 0.5
                base = torch.where(i <= j + shift, p, torch.zeros(()))
                if (n0 + block_n - 1 < s_kv) and (m0 + block_m - 1 < s_q) and (n0 + block_n - 1 <= m0 + shift):
                    fast += 1
                    cand = p  # block-level fast path, no select emitted
                else:
                    cand = torch.where(i <= j + shift, p, torch.zeros(()))
                total += p.numel()
                viol += int((cand != base).sum())
    return {
        "item": "A2 块级 mask 特化（block_n=%d, block_m=%d）" % (block_n, block_m),
        "n": total,
        "max_ulp": 0,
        "viol": viol / total,
        "max_ulp_cand_vs_fp64": "-",
        "max_ulp_base_vs_fp64": "-",
        "verdict": "EQUIV_PASS" if viol == 0 else "EQUIV_BLOCKED",
        "fast": "%d/%d" % (fast, blocks),
    }


def main() -> int:
    rows = [check_a1(torch.float16), check_a1(torch.bfloat16), check_a2()]
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
                item=r["item"], n=r["n"], mu=r["max_ulp"], v=r["viol"],
                c=r["max_ulp_cand_vs_fp64"], b=r["max_ulp_base_vs_fp64"], verdict=r["verdict"],
            )
        )
    print("A2 fast-path blocks: %s" % rows[2]["fast"])
    print("EQUIV_SUMMARY", "ALL_PASS" if ok else "HAS_BLOCKED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
