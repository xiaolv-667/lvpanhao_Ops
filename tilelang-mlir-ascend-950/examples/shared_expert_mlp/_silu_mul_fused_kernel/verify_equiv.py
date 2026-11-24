"""D-1 machine check for ``_silu_mul_fused_kernel``'s adopted formula optimizations.

Compares the **baseline** formula (the source kernel's literal
``g * sigmoid(g) * u``, with ``sigmoid`` evaluated as one reciprocal then two
multiplies) against the **candidate** adopted in DESIGN.md 1.6.1 A1
(``(g*u) / (1+exp(-g))``), plus the rejected C2 variant
(``g*u*exp(g) / (1+exp(g))``) that the report claims breaks on overflow.

Both formulas are replayed through the *same* fp32 rounding path the kernel
runs (every intermediate rounded to float32, one final rounding to the output
dtype), and both are additionally compared against an fp64 reference.  The
judgement is on the **output dtype** (float16 / bfloat16), which is what the
operator returns.

Also measured: the two golden forms a developer may pick -- fp32-upcast
(``silu(g.float())*u.float()`` -> dtype, the source kernel's dtype semantics)
vs torch's native-dtype ``silu`` -- so the L0 tolerance covers either.

Run (torch CPU only):

    cd examples/shared_expert_mlp/_silu_mul_fused_kernel
    /workspace/venv-ascend950/bin/python verify_equiv.py
"""

import math

import torch
import torch.nn.functional as F

TORCH_DTYPE = {"float16": torch.float16, "bfloat16": torch.bfloat16}
ATOL = RTOL = 1e-2  # moe_verification(1): elementwise atol = rtol = 1e-2


# ---------------------------------------------------------------------------
# formulas, each with an explicit fp32 rounding path
# ---------------------------------------------------------------------------
def _f32(x):
    return x.float()


def baseline(g64, u64):
    """Source literal: ``p = g * (1/(1+exp(-g))); y = p * u`` in fp32."""
    g, u = _f32(g64), _f32(u64)
    t = torch.exp(-g)              # fp32 exp(-g)
    d = 1.0 + t                    # fp32 add
    s = 1.0 / d                    # fp32 reciprocal (one division)
    p = g * s                      # fp32 mul
    return p * u                   # fp32 mul


def candidate(g64, u64):
    """DESIGN 1.6.1 A1: ``num = g*u; y = num / (1+exp(-g))`` in fp32."""
    g, u = _f32(g64), _f32(u64)
    t = torch.exp(-g)
    d = 1.0 + t
    num = g * u
    return num / d


def rejected_c2(g64, u64):
    """Rejected C2: ``g*u*exp(g) / (1+exp(g))`` (overflow -> inf/inf)."""
    g, u = _f32(g64), _f32(u64)
    e = torch.exp(g)
    return (g * u) * e / (1.0 + e)


def ref64(g64, u64):
    """fp64 reference of the operator's mathematics."""
    g, u = g64.double(), u64.double()
    return (g * u) / (1.0 + torch.exp(-g))


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def _ordered(x, dtype):
    """Monotonic integer key: neighbour difference == ulp distance."""
    if dtype == torch.bfloat16:
        i = x.view(torch.int16).to(torch.int32)
    else:
        i = x.view(torch.int16).to(torch.int32)
    return torch.where(i < 0, -32768 - i, i)


def _dtype_round(x, dtype):
    return x.to(dtype)


def compare(a_f32, b_f32, dtype):
    """Metrics of ``a`` vs ``b`` after rounding both to *dtype*."""
    a, b = _dtype_round(a_f32, dtype), _dtype_round(b_f32, dtype)
    finite = torch.isfinite(a) & torch.isfinite(b)
    nan_a, nan_b = torch.isnan(a), torch.isnan(b)
    inf_a, inf_b = torch.isinf(a), torch.isinf(b)
    mism = (nan_a != nan_b) | (inf_a != inf_b) | (inf_a & inf_b & (a != b))
    ulp = torch.zeros_like(a, dtype=torch.float32)
    if finite.any():
        ulp[finite] = (
            (_ordered(a[finite], dtype) - _ordered(b[finite], dtype)).abs().float()
        )
    n = a.numel()
    diff_frac = float((ulp > 0).sum()) / n
    bad_frac = float((ulp > 1).sum()) / n
    rel = torch.zeros_like(a, dtype=torch.float32)
    if finite.any():
        denom = b[finite].float().abs().clamp_min(1e-30)
        rel[finite] = ((a[finite].float() - b[finite].float()).abs() / denom)
    return {
        "max_ulp": float(ulp.max()) if n else 0.0,
        "frac_any_ulp": diff_frac,
        "frac_gt1_ulp": bad_frac,
        "max_rel": float(rel.max()) if n else 0.0,
        "nonfinite_mismatch": int(mism.sum()),
        "n": n,
    }


def tolerance_violations(a_f32, ref_f32, dtype):
    """Elementwise atol/rtol violations of ``a`` vs the fp64 reference."""
    a = _dtype_round(a_f32, dtype)
    r = _dtype_round(ref_f32, dtype)
    finite = torch.isfinite(r) & torch.isfinite(a)
    if not finite.any():
        return 0.0
    err = (a[finite].float() - r[finite].float()).abs()
    bound = ATOL + RTOL * r[finite].float().abs()
    return float((err > bound).sum()) / a.numel()


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------
def _cases(dtype_name, generator):
    """(name, g fp64, u fp64) rows for one dtype."""
    dt = TORCH_DTYPE[dtype_name]
    rows = []

    def q(x):
        """Round to the input dtype, as the operator receives it."""
        return x.to(dt).double()

    g = q(torch.randn(1 << 14, generator=generator))
    u = q(torch.randn(1 << 14, generator=generator))
    rows.append(("random", g, u))

    g = q(torch.randn(1 << 13, generator=generator) * 12.0)
    u = q(torch.randn(1 << 13, generator=generator) * 1e3)
    rows.append(("wide-range", g, u))

    corner = torch.tensor(
        [
            0.0, -0.0, 1e-8, -1e-8, 1e-38, -1e-38, 6e-8, -6e-8,
            1.0, -1.0, 10.0, -10.0, 88.0, -88.0, 100.0, -100.0, 1000.0, -1000.0,
            float("inf"), float("-inf"), float("nan"),
            torch.finfo(dt).max, -torch.finfo(dt).max,
            torch.finfo(dt).tiny, -torch.finfo(dt).tiny,
        ],
        dtype=torch.float64,
    )
    u_corner = torch.tensor(
        [
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            0.0, 1.0, -1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
        ],
        dtype=torch.float64,
    )
    rows.append(("corner", q(corner), q(u_corner)))

    # dead zone: |g| large negative -> exp(-g) overflows to +inf in fp32
    g = q(torch.linspace(-200.0, -80.0, 4096, dtype=torch.float64))
    u = q(torch.ones(4096, dtype=torch.float64))
    rows.append(("exp-overflow", g, u))
    return rows


def main():
    generator = torch.Generator().manual_seed(20261123)
    table = []
    failures = 0
    for dtype_name in ("bfloat16", "float16"):
        dt = TORCH_DTYPE[dtype_name]
        for name, g64, u64 in _cases(dtype_name, generator):
            base = baseline(g64, u64)
            cand = candidate(g64, u64)
            c2 = rejected_c2(g64, u64)
            ref = ref64(g64, u64)
            m = compare(cand, base, dt)
            m2 = compare(c2, base, dt)
            viol = tolerance_violations(cand, ref, dt)
            ok = (
                m["max_ulp"] <= 1
                and m["frac_gt1_ulp"] <= 1e-3
                and m["nonfinite_mismatch"] == 0
                and viol == 0.0
            )
            failures += 0 if ok else 1
            table.append(
                (
                    dtype_name,
                    name,
                    m["max_ulp"],
                    m["frac_gt1_ulp"],
                    m["max_rel"],
                    m["nonfinite_mismatch"],
                    viol,
                    m2["max_ulp"],
                    m2["nonfinite_mismatch"],
                    "EQUIV_PASS" if ok else "EQUIV_FAIL",
                )
            )

    # golden form: fp32 upcast (source semantics) vs torch native dtype
    for dtype_name in ("bfloat16", "float16"):
        dt = TORCH_DTYPE[dtype_name]
        g = torch.randn(1 << 14, generator=generator).to(dt)
        u = torch.randn(1 << 14, generator=generator).to(dt)
        upcast = (F.silu(g.float()) * u.float()).to(dt)
        native = F.silu(g) * u
        m = compare(upcast.float(), native.float(), dt)
        viol = tolerance_violations(upcast.float(), native.float(), dt)
        print(
            f"golden-form {dtype_name}: upcast-vs-native max_ulp={m['max_ulp']:.0f} "
            f"frac_gt1_ulp={m['frac_gt1_ulp']:.2e} max_rel={m['max_rel']:.3e} "
            f"tol_violations={viol:.2e}",
            flush=True,
        )

    header = (
        "| dtype | case | max_ulp | frac_gt1_ulp | max_rel | nonfinite_mismatch "
        "| tol_viol | C2 max_ulp | C2 nonfinite_mismatch | verdict |"
    )
    print(header)
    print("|" + "---|" * 10)
    for row in table:
        print(
            f"| {row[0]} | {row[1]} | {row[2]:.0f} | {row[3]:.2e} | {row[4]:.3e} | "
            f"{row[5]} | {row[6]:.2e} | {row[7]:.0f} | {row[8]} | **{row[9]}** |"
        )
    print("EQUIV_RESULT:", "ALL_PASS" if failures == 0 else f"{failures} FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
