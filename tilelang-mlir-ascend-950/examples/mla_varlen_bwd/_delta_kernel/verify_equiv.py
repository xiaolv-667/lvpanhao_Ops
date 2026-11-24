"""Equivalence machine check for the packed _delta_kernel design (D-1, Stage 1).

Companion of ``examples/mla_dense_bwd/_delta_kernel/verify_equiv.py``: A1/A2/A2b/A3
are the same three formula-level optimizations (the packed function computes the
identical per-row dot product, so the same arguments and the same numbers apply),
and **A4 is re-derived for this function** -- the head-major packed write-back,
whose destination address and extent are dynamic.

Reference: an fp64 exact computation; the compared forms are emulated in float32
step by step (each binary op rounds to float32, which is what the device does), so
a grouping / materialization change shows up in the numbers instead of being
masked by fp64 headroom.

Adopted items under test
------------------------
A1  fused register multiply (no fp32 product tile materialized in UB)
        baseline : p = fl32(o) * fl32(do) -> stored to an fp32 tile -> summed
        candidate: same product, consumed straight from the register
    Expected difference: exactly 0 ulp (the product is rounded to fp32 in both
    forms; the only thing that moves is where the value lives).

A2  pairwise tree reduction order (vcadd over 64 fp32 lanes) instead of a
    sequential running sum -- emulating the *implemented* grouping (DESIGN.md
    3.1 steps 4-6 / 3.3): one vld of 128 16-bit lanes, vcvt PART_EVEN and
    PART_ODD, two vmul, one vadd (lane k holds p[2k] + p[2k+1]) and one vcadd
    (6-stage pairwise over the 64 lanes).
        baseline : acc += p[d] for d = 0..D-1   (sequential, D roundings)
        candidate: vcadd(p[0::2]*... + p[1::2]*...)   [implementation grouping]
    Expected difference: tolerance-level only (fp32 grouping), no systematic bias.
    A2b keeps the *alternative* two-reduction grouping quoted in dense v1 so both
    groups stay on record; the adopted implementation is A2.

A3  batched strided output store from a 32-byte-row staging buffer instead of one
    scalar store per row (index-mapping check of the compacting step)
        baseline : out[i] written directly, i = 0..R-1
        candidate: values land in staging[i, 0] of an (R, 8) buffer, then column 0
                   is gathered into out[R]
    Expected difference: 0 (pure data movement).

A4  *(packed-only)* head-major compressed write-back with the dynamic extent
        baseline : delta[h, t0 + i] = s[t0 + i] for i < n, n = min(R, total - t0)
                   (one scalar store per valid row)
        candidate: staging[R, 8] column 0 -> one compacting copy into
                   delta[h, t0 : t0 + n]
    Checks: (a) bit-exact equality of the two forms over the whole (heads, total)
    tensor; (b) coverage -- the candidate writes exactly the valid positions and
    nothing else; (c) every tile has n >= 1 (t0 < total always holds);
    (d) diagnostic: how many entries a *static* R-extent store would touch outside
    the head if the platform did not clamp it to the last axis (the reason the
    explicit clamp is kept; see DESIGN.md 3.4 / N3).

Run:
    cd examples/mla_varlen_bwd/_delta_kernel
    /workspace/venv-ascend950/bin/python verify_equiv.py

Prints one row per (item, case) plus the summary table embedded in DESIGN.md
1.6.1.  Exit code 0 == every adopted item passed.
"""

from __future__ import annotations

import math

import torch

# Practical tolerance of the L0 comparison (DESIGN.md 8.2): atol + rtol*|ref|.
TOL_ATOL = 1e-4
TOL_RTOL = 1e-4
# Equivalence criterion used for the verdict: a reduction of D float32 terms
# carries an error proportional to sum|p_d| (not to |sum p_d|, which cancels), so
# the adopted form passes when |cand - ref| <= RTOL_SCALED * sum|p_d| + ATOL_FLOOR.
# 1e-5 leaves ~12x margin over the fp32 bound eps*sqrt(D) ~ 8e-7 for D = 128.
RTOL_SCALED = 1e-5
ATOL_FLOOR = 1e-6
SEED = 20261123
R_TILE = 64  # row tile of DESIGN.md 5.2


# ---------------------------------------------------------------------------
# elementary fp32 / fp64 helpers (explicit rounding, no fused ops)
# ---------------------------------------------------------------------------
def f32(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float32)


def pairwise_tree(x: torch.Tensor) -> torch.Tensor:
    """Pairwise (binary tree) sum over the last axis, emulated in float32."""
    cur = x
    while cur.shape[-1] > 1:
        if cur.shape[-1] % 2:
            pad = torch.zeros_like(cur[..., :1])
            cur = torch.cat([cur, pad], dim=-1)
        cur = cur[..., 0::2] + cur[..., 1::2]  # one fp32 rounding per stage
    return cur[..., 0]


def sequential_sum(x: torch.Tensor) -> torch.Tensor:
    acc = torch.zeros(x.shape[:-1], dtype=torch.float32)
    for d in range(x.shape[-1]):
        acc = acc + x[..., d]
    return acc


def impl_grouping(x: torch.Tensor, vw: int = 128) -> torch.Tensor:
    """The device grouping of DESIGN.md 3.1 steps 3-6, emulated in float32.

    One `vld` covers `vw` 16-bit lanes; `vcvt` PART_EVEN / PART_ODD split them into
    two fp32x64 vectors holding the even and the odd lanes; the two `vmul` products
    are added lane-wise (so lane k holds p[2k] + p[2k+1], one fp32 rounding), and
    `vcadd` runs a 6-stage pairwise tree over the 64 lanes.  Columns past the
    valid width are zero (UB pad), which is what the design clears.
    """
    n = x.shape[-1]
    if n < vw:
        x = torch.cat([x, torch.zeros(*x.shape[:-1], vw - n, dtype=x.dtype)], dim=-1)
    nvec = (x.shape[-1] + vw - 1) // vw
    part = None
    for v in range(nvec):
        blk = x[..., v * vw : (v + 1) * vw]
        if blk.shape[-1] < vw:
            blk = torch.cat(
                [blk, torch.zeros(*blk.shape[:-1], vw - blk.shape[-1], dtype=blk.dtype)], dim=-1
            )
        cur = blk[..., 0::2] + blk[..., 1::2]  # vadd(vmul(even), vmul(odd))
        part = cur if part is None else part + cur  # nvec > 1: build-time chained vadd
    return pairwise_tree(part)  # vcadd over the 64 partial sums


def ulp_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    """Max ULP distance between two finite float32 tensors (0.0 if identical)."""
    a = a.to(torch.float32).contiguous()
    b = b.to(torch.float32).contiguous()
    finite = torch.isfinite(a) & torch.isfinite(b)
    if not bool(finite.any()):
        return 0.0
    ai = a.view(torch.int32).to(torch.int64)[finite]
    bi = b.view(torch.int32).to(torch.int64)[finite]
    # map IEEE bit patterns to a monotone integer key so +x and -x compare sanely
    ai = torch.where(ai < 0, torch.tensor(-(1 << 31), dtype=torch.int64) - ai, ai)
    bi = torch.where(bi < 0, torch.tensor(-(1 << 31), dtype=torch.int64) - bi, bi)
    return float((ai - bi).abs().max().item())


def violation_rate(cand: torch.Tensor, ref: torch.Tensor, atol: float, rtol: float) -> float:
    """Fraction of entries outside |cand - ref| <= atol + rtol * |ref|."""
    c = cand.to(torch.float64)
    r = ref.to(torch.float64)
    ok = torch.isclose(c, r, atol=atol, rtol=rtol, equal_nan=True)
    return float((~ok).float().mean().item())


def violation_rate_scaled(
    cand: torch.Tensor, o: torch.Tensor, do: torch.Tensor
) -> float:
    """Fraction of entries outside |cand - ref| <= RTOL_SCALED*sum|p| + ATOL_FLOOR."""
    p = (o.to(torch.float64) * do.to(torch.float64)).abs()
    ref = (o.to(torch.float64) * do.to(torch.float64)).sum(-1)
    scale = p.sum(-1)
    err = (cand.to(torch.float64) - ref).abs()
    ok = err <= RTOL_SCALED * scale + ATOL_FLOOR
    cf = cand.to(torch.float64)
    ok = ok | (torch.isnan(cf) & torch.isnan(ref)) | (cf == ref)
    return float((~ok).float().mean().item())


def nonfinite_mismatch(a: torch.Tensor, b: torch.Tensor) -> int:
    fa, fb = torch.isfinite(a), torch.isfinite(b)
    return int((fa != fb).sum().item())


# ---------------------------------------------------------------------------
# input cases for A1/A2/A2b/A3 (identical product, identical cases as dense)
# ---------------------------------------------------------------------------
def make_cases():
    """(name, o, do) rows, D wide, on which A1/A2/A2b/A3 are compared."""
    g = torch.Generator().manual_seed(SEED)
    cases = []

    def quantize(x, dtype):
        return x.to(dtype).to(torch.float32)  # exact: 16-bit -> fp32

    for dtype, tag in ((torch.float16, "fp16"), (torch.bfloat16, "bf16")):
        for D in (128, 127):
            o = quantize(torch.randn(256, D, generator=g), dtype)
            do = quantize(torch.randn(256, D, generator=g), dtype)
            cases.append((f"random/{tag}/D{D}", o, do))

        # large magnitudes: products up to ~1e4, sums up to ~1e5
        o = quantize(torch.randn(64, 128, generator=g) * 512, dtype)
        do = quantize(torch.randn(64, 128, generator=g) * 512, dtype)
        cases.append((f"large/{tag}", o, do))

        # tiny / subnormal magnitudes
        o = quantize(torch.randn(64, 128, generator=g) * 1e-5, dtype)
        do = quantize(torch.randn(64, 128, generator=g) * 1e-5, dtype)
        cases.append((f"tiny/{tag}", o, do))

        # corner rows: zeros, +-inf, nan, alternating signs, constant rows
        o = torch.zeros(6, 128)
        do = torch.zeros(6, 128)
        o[1, 0] = math.inf
        do[1, 0] = 1.0
        o[2, 3] = -math.inf
        do[2, 3] = 2.0
        o[3, 5] = math.nan
        do[3, 5] = 1.0
        o[4] = torch.tensor([1.0, -1.0] * 64)
        do[4] = torch.tensor([1.0, -1.0] * 64)
        o[5] = 1.0
        do[5] = 1.0
        o[0, 7] = math.inf  # inf * 0 -> nan, both forms must agree
        do[0, 7] = 0.0
        cases.append((f"corner/{tag}", o, do))

        # single-element reduction (padded device path)
        o = quantize(torch.randn(32, 1, generator=g), dtype)
        do = quantize(torch.randn(32, 1, generator=g), dtype)
        cases.append((f"D1/{tag}", o, do))

    return cases


# ---------------------------------------------------------------------------
# the adopted items
# ---------------------------------------------------------------------------
def item_a1_fused(o: torch.Tensor, do: torch.Tensor):
    prod_tile = f32(o) * f32(do)  # materialized fp32 tile
    base = sequential_sum(prod_tile)
    cand = sequential_sum(f32(o) * f32(do))  # same product, register-resident
    return base, cand


def item_a2_tree(o: torch.Tensor, do: torch.Tensor):
    """Implementation grouping: interleaved PART_EVEN/PART_ODD add, then vcadd."""
    prod = f32(o) * f32(do)
    base = sequential_sum(prod)
    cand = impl_grouping(prod)
    return base, cand


def item_a2_alt(o: torch.Tensor, do: torch.Tensor):
    """dense v1's quoted grouping (two independent vcadds) -- kept for the record."""
    prod = f32(o) * f32(do)
    base = sequential_sum(prod)
    if prod.shape[-1] <= 64:
        cand = pairwise_tree(prod)
    else:
        cand = pairwise_tree(prod[..., :64]) + pairwise_tree(prod[..., 64:])
    return base, cand


def item_a3_store(o: torch.Tensor, do: torch.Tensor):
    """Staging-buffer layout: (R, 8) 32-byte rows, column 0 compacted to out[R]."""
    prod = f32(o) * f32(do)
    sums = impl_grouping(prod)
    R = sums.shape[0]
    direct = sums.clone()  # baseline: one scalar store per row
    staging = torch.zeros(R, 8, dtype=torch.float32)
    for i in range(R):
        staging[i, 0] = sums[i]  # vsts with PAT_VL1 / extent = 1
    compact = staging[:, 0].contiguous()  # strided UB -> contiguous GM copy
    return direct, compact


ITEMS = (
    ("A1", "fused register multiply (no fp32 product tile)", item_a1_fused),
    ("A2", "vcadd grouping as implemented (interleaved halves + 1 vcadd)", item_a2_tree),
    ("A2b", "alternative grouping quoted in dense v1 (2 vcadd + scalar add)", item_a2_alt),
    ("A3", "batched strided output store (staging -> compact)", item_a3_store),
)


# ---------------------------------------------------------------------------
# A4: packed head-major write-back (re-derived for this function)
# ---------------------------------------------------------------------------
def make_packed_cases():
    """(name, o, do) for the write-back check: shapes that exercise every n."""
    g = torch.Generator().manual_seed(SEED + 1)
    cases = []
    # total_q values: exact multiple of R, +1, +7, prime, smaller than R, 1, empty
    for total, heads, tag in (
        (128, 4, "T128-H4-full"),
        (256, 1, "T256-H1-full"),
        (65, 4, "T65-H4-rag1"),
        (200, 8, "T200-H8-rag8"),
        (1, 1, "T1-H1-single"),
        (63, 128, "T63-H128-rag63"),
        (4096, 128, "T4096-H128-many"),
        (0, 4, "T0-H4-empty"),
    ):
        o = torch.randn(total, 128, generator=g).to(torch.float16).to(torch.float32)
        do = torch.randn(total, 128, generator=g).to(torch.float16).to(torch.float32)
        if total >= 3:
            o[0, 0] = math.inf      # non-finite values must travel through the
            do[0, 0] = 0.0          # staging buffer unchanged (inf*0 -> nan)
            o[1, 3] = math.nan
            do[1, 3] = 1.0
            o[total - 1, 7] = -math.inf  # ... including on the ragged row
            do[total - 1, 7] = 3.0
        cases.append((tag, o, do, heads, total))
    return cases


def item_a4_packed_write(o, do, heads, total, R=R_TILE):
    """(baseline, candidate, stats) of the head-major compacting write-back."""
    try:
        sums = impl_grouping(f32(o) * f32(do))  # (total,) fp32
    except Exception:  # noqa: BLE001 - empty tensor: keep the shape logic
        sums = torch.zeros(total, dtype=torch.float32)
    direct = torch.zeros(heads, total, dtype=torch.float32)
    cand = torch.zeros(heads, total, dtype=torch.float32)
    mask = torch.zeros(heads, total, dtype=torch.bool)
    ntiles = ((total + R - 1) // R) * heads
    stats = {"tiles": ntiles, "min_n": None, "spill_static": 0, "bad_tile": 0}
    for it in range(ntiles):
        h = it % heads          # tile index -> head (kernel: h = it % heads)
        t0 = (it // heads) * R  # tile index -> row block (kernel: t0 = (it // heads) * R)
        n = min(R, total - t0)
        if n <= 0:
            stats["bad_tile"] += 1
            continue
        stats["min_n"] = n if stats["min_n"] is None else min(stats["min_n"], n)
        # baseline: one scalar store per valid row
        for i in range(n):
            direct[h, t0 + i] = sums[t0 + i]
        # candidate: staging[i, 0] (vsts) then one compacting copy of [0:n)
        staging = torch.zeros(R, 8, dtype=torch.float32)
        for i in range(n):
            staging[i, 0] = sums[t0 + i]
        cand[h, t0 : t0 + n] = staging[:n, 0]
        mask[h, t0 : t0 + n] = True
        # diagnostic: what a static R-extent store would touch outside the head
        stats["spill_static"] += R - n
    return direct, cand, mask, stats


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def main() -> int:
    rows = []
    failed = False

    for item, _desc, fn in ITEMS:
        for name, o, do in make_cases():
            base, cand = fn(o, do)
            ref = (o.to(torch.float64) * do.to(torch.float64)).sum(-1)  # exact
            u_ref = ulp_diff(cand, ref)
            u_pair = ulp_diff(cand, base)
            u_base = ulp_diff(base, ref)
            vr = violation_rate(cand, ref, TOL_ATOL, TOL_RTOL)
            vs = violation_rate_scaled(cand, o, do)
            nf = nonfinite_mismatch(cand, base)
            ok = (vs == 0.0) and (nf == 0)
            failed |= not ok
            rows.append((item, name, u_ref, u_pair, u_base, vr, vs, nf,
                         "EQUIV_PASS" if ok else "EQUIV_FAIL"))

    # --- A4: packed write-back (bit-exact index mapping + coverage) ---
    a4 = []
    for name, o, do, heads, total in make_packed_cases():
        direct, cand, mask, stats = item_a4_packed_write(o, do, heads, total)
        # (a) both forms agree bit-for-bit, (b) the tile stream covers exactly the
        # valid domain, (c) nothing outside the domain was written
        # bit-exact: compare IEEE bit patterns so NaN == NaN counts as equal
        same = bool(torch.equal(direct.view(torch.int32), cand.view(torch.int32)))
        covered = bool(mask.all())
        clean = bool(torch.equal(cand[~mask], torch.zeros_like(cand[~mask])))
        ok = same and covered and clean and stats["bad_tile"] == 0
        failed |= not ok
        a4.append((name, heads, total, stats, same, clean, ok))
        print(
            f"A4  {name:24} heads={heads:4} T={total:6} tiles={stats['tiles']:6} "
            f"min_n={stats['min_n']} spill_static={stats['spill_static']:6} "
            f"bit_equal={same} covered={covered} clean_outside={clean} "
            f"{'EQUIV_PASS' if ok else 'EQUIV_FAIL'}",
            flush=True,
        )

    print("\n=== equivalence table (torch CPU, fp64 reference) ===")
    header = (
        f"{'item':5} {'case':22} {'ulp(cand,ref)':>14} {'ulp(cand,base)':>15} "
        f"{'viol_std':>9} {'viol_scale':>11} {'nf_mis':>7}  verdict"
    )
    print(header)
    for r in rows:
        print(
            f"{r[0]:5} {r[1]:22} {r[2]:14.1f} {r[3]:15.1f} "
            f"{r[5]:9.4f} {r[6]:11.4f} {r[7]:7d}  {r[8]}"
        )

    # per-item aggregate (what DESIGN.md 1.6.1 embeds)
    print("\n=== per-item summary ===")
    print(f"{'item':5} {'cases':>6} {'max ulp(cand,ref)':>18} {'max violation':>14}  verdict")
    for item, _desc, _fn in ITEMS:
        sel = [r for r in rows if r[0] == item]
        max_ulp = max(r[2] for r in sel)
        max_vr = max(r[6] for r in sel)
        verdict = "EQUIV_PASS" if all(r[8] == "EQUIV_PASS" for r in sel) else "EQUIV_FAIL"
        print(f"{item:5} {len(sel):6d} {max_ulp:18.1f} {max_vr:14.4f}  {verdict}")
    a4_verdict = "EQUIV_PASS" if all(r[6] for r in a4) else "EQUIV_FAIL"
    print(f"{'A4':5} {len(a4):6d} {'0.0 (bit-exact)':>18} {0.0:14.4f}  {a4_verdict}")

    print("\nall items pass" if not failed else "\nsome items failed")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
