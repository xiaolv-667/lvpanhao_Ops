"""Design-time equivalence machine check (D-1) for `_ssd_chunk_coupling_kernel`.

Every *adopted* mathematical-equivalence item of DESIGN.md §1.6.1 is compared
against the baseline expression it replaces, on random tensors **and** corner
values (``±inf``, ``nan``, ``±0``, subnormals, max finite), with an fp64
reference computed on CPU by torch and then rounded to the target storage dtype
(fp16 / bf16) along each implementation's own rounding path.

Items
-----
E1 (adopted)   masked cast fused into one select-write
               vs  baseline "write the full fp32 product, then zero the upper
               triangle with a second pass".
E2 (adopted)   causal predicate strength reduction
               ``(s0 + j) <= (l0 + i)``  ->  ``(j - i) <= (l0 - s0)``
               (integer identity; also covers the AIV sub-block row offset).
E3 (refuted)   masked *product* ``v * 1[s <= l]`` instead of select -- kept only
               as boundary evidence (``nan * 0 == nan`` breaks the "upper
               triangle is exactly 0" contract), NOT adopted.

Run (CPU only, no device):

    cd examples/ssd_chunk_coupling/_ssd_chunk_coupling_kernel
    /workspace/venv-ascend950/bin/python verify_equiv.py
"""

import json
import math

import torch

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16}
ULP_EPS = {"float16": 2.0**-10, "bfloat16": 2.0**-7}


def causal_mask(q):
    """lower triangle incl. diagonal (keep s <= l)."""
    idx = torch.arange(q)
    return idx[None, :] <= idx[:, None]


def _bits(t):
    """integer view of a 2-byte float tensor, for ulp distances."""
    return t.view(torch.int16).to(torch.int32)


def compare(ref, got, name, dtype):
    """exact + ulp comparison of two equally-shaped tensors of `dtype`."""
    both_nan = torch.isnan(ref) & torch.isnan(got)
    same = (ref == got) | both_nan
    viol = int((~same).sum().item())
    total = int(same.numel())
    finite = torch.isfinite(ref) & torch.isfinite(got)
    if finite.any():
        max_ulp = int((_bits(ref)[finite] - _bits(got)[finite]).abs().max().item())
        max_abs = float((ref[finite].float() - got[finite].float()).abs().max().item())
    else:
        max_ulp, max_abs = 0, 0.0
    return {
        "item": name,
        "dtype": dtype,
        "elements": total,
        "violations": viol,
        "violation_rate": viol / max(total, 1),
        "max_ulp": max_ulp,
        "max_abs": max_abs,
        "verdict": "EQUIV_PASS" if viol == 0 else "EQUIV_FAIL",
    }


def reference_acc(c, b):
    """fp64 reference of the mathematical product cb = C @ B^T per (chunk, group)."""
    return torch.einsum("bclgn,bcsgn->bcgls", c, b)


def e1_masked_cast_fused(dtype_str):
    """E1: fused select+cast write vs baseline two-pass write-then-zero."""
    dt = DTYPES[dtype_str]
    torch.manual_seed(0)
    rows = []
    # (a) random, (b) corner values incl. inf/nan/subnormal, (c) garbage acc.
    bs, nc, g, q, n = 1, 2, 2, 8, 16
    cases = {
        "random": None,
        "corner": None,
        "garbage": None,
    }
    c = torch.zeros(bs, nc * q, g, n, dtype=torch.float64)
    b = torch.zeros(bs, nc * q, g, n, dtype=torch.float64)
    for case in cases:
        if case == "random":
            c.normal_()
            b.normal_()
        elif case == "corner":
            # deterministic spread of the corner values torch can represent.
            torch.manual_seed(7)
            c = torch.randn(bs, nc * q, g, n, dtype=torch.float64)
            b = torch.randn(bs, nc * q, g, n, dtype=torch.float64)
            c.view(-1)[0] = math.inf
            c.view(-1)[1] = -math.inf
            c.view(-1)[2] = math.nan
            c.view(-1)[3] = 0.0
            c.view(-1)[4] = -0.0
            c.view(-1)[5] = 5.0e-324          # subnormal in fp64
            c.view(-1)[6] = 65504.0           # fp16 max finite
            b.view(-1)[0] = math.nan
            b.view(-1)[1] = math.inf
            b.view(-1)[2] = -math.inf
            b.view(-1)[7] = 3.0e-324
        else:  # garbage: the accumulator of a fully-upper tile is never written
            c.fill_(math.nan)
            b.fill_(math.inf)
        acc = reference_acc(
            c.reshape(bs, nc, q, g, n), b.reshape(bs, nc, q, g, n)
        )                                             # (bs, nc, g, q, q) fp64
        mask = causal_mask(q)
        # baseline: materialise everything, then zero the upper triangle.
        base = acc.to(dt).clone()
        base[:, :, :, ~mask] = torch.zeros((), dtype=dt)
        # candidate: single fused select+cast, upper triangle never materialised.
        cand = torch.where(mask, acc, torch.zeros((), dtype=torch.float64)).to(dt)
        rows.append(compare(base, cand, f"E1/{case}", dtype_str))
    agg = {
        "item": "E1",
        "dtype": dtype_str,
        "elements": sum(r["elements"] for r in rows),
        "violations": sum(r["violations"] for r in rows),
        "max_ulp": max(r["max_ulp"] for r in rows),
        "max_abs": max(r["max_abs"] for r in rows),
    }
    agg["violation_rate"] = agg["violations"] / max(agg["elements"], 1)
    agg["verdict"] = "EQUIV_PASS" if agg["violations"] == 0 else "EQUIV_FAIL"
    return agg, rows


def e2_mask_index_identity():
    """E2: (s0 + j) <= (l0 + i)  ==  (j - i) <= (l0 - s0)  over the tile grid."""
    bad = 0
    total = 0
    for bl in (16, 32, 64, 128):
        for l0 in range(0, 256, bl):
            for s0 in range(0, 256, bl):
                i = torch.arange(bl // 2).view(-1, 1)     # AIV row (half tile)
                j = torch.arange(bl).view(1, -1)          # column
                for sid in (0, 1):
                    lhs = (s0 + j) <= (l0 + sid * (bl // 2) + i)
                    rhs = (j - i) <= (l0 + sid * (bl // 2) - s0)
                    total += int(lhs.numel())
                    bad += int((lhs != rhs).sum().item())
    return {
        "item": "E2",
        "dtype": "int",
        "elements": total,
        "violations": bad,
        "violation_rate": bad / max(total, 1),
        "max_ulp": 0,
        "max_abs": 0.0,
        "verdict": "EQUIV_PASS" if bad == 0 else "EQUIV_FAIL",
    }


def e4_masked_tile_skip(dtype_str):
    """E4: skipping the K accumulation of a fully-masked tile is exact.

    The whole tile is upper (no element satisfies s <= l), so its output must be
    exactly 0 regardless of the accumulator value.  Baseline = compute from real
    inputs then zero the tile; candidate = arbitrary garbage accumulator then the
    same select mask.
    """
    dt = DTYPES[dtype_str]
    q = 16
    torch.manual_seed(11)
    c = torch.randn(1, 1, q, 1, q, dtype=torch.float64)
    b = torch.randn(1, 1, q, 1, q, dtype=torch.float64)
    acc = reference_acc(c, b)                      # real accumulator (baseline)
    mask = torch.zeros((q, q), dtype=torch.bool)   # fully-masked tile
    base = torch.where(mask, acc, torch.zeros((), dtype=torch.float64)).to(dt)
    garbage = torch.empty((q, q), dtype=torch.float64)
    flat = garbage.view(-1)
    flat[0::4] = math.nan
    flat[1::4] = math.inf
    flat[2::4] = -0.0
    flat[3::4] = 4.9e-324                          # subnormal
    cand = torch.where(mask, garbage, torch.zeros((), dtype=torch.float64)).to(dt)
    return compare(base.reshape(-1), cand.reshape(-1), f"E4/{dtype_str}", dtype_str)


def e3_mask_by_product():
    """E3 (NOT adopted): v * 1[s<=l] loses the exact-zero contract for nan/inf."""
    q, n = 8, 16
    c = torch.full((1, 1, q, 1, n), math.nan, dtype=torch.float64)
    b = torch.full((1, 1, q, 1, n), 1.0, dtype=torch.float64)
    acc = reference_acc(c, b)
    mask = causal_mask(q)
    base = torch.where(mask, acc, torch.zeros((), dtype=torch.float64))
    prod = acc * mask.to(torch.float64)
    viol = int((~((prod == base) | (torch.isnan(prod) & torch.isnan(base)))).sum().item())
    return {
        "item": "E3",
        "dtype": "float64",
        "elements": int(base.numel()),
        "violations": viol,
        "note": "nan*0 = nan -> upper triangle is not exactly 0; refuted, not adopted",
    }


def main():
    table, detail = [], []
    for dtype_str in DTYPES:
        agg, rows = e1_masked_cast_fused(dtype_str)
        table.append(agg)
        detail.extend(rows)
        table.append(e4_masked_tile_skip(dtype_str))
    table.append(e2_mask_index_identity())
    refuted = e3_mask_by_product()
    for row in detail + table:
        print(json.dumps(row), flush=True)
    print(json.dumps({"item": "E3(refuted)", **refuted}), flush=True)
    failed = [r["item"] for r in table if r.get("verdict") != "EQUIV_PASS"]
    print(json.dumps({"adopted_items": [r["item"] for r in table],
                      "failed": failed,
                      "summary": "all adopted items EQUIV_PASS" if not failed
                      else "adopted item failed: " + ",".join(failed)}))


if __name__ == "__main__":
    main()
