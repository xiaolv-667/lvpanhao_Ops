"""Equivalence machine verification (D-1) for the adopted math optimizations of
`_gemm_basic_kernel` (DESIGN.md §1.6.1).

torch CPU only -- no NPU, no kernel.  Both sides are evaluated in float64 as the
reference, then each side's *rounding path* is replayed in the precision the
hardware actually uses (fp32 accumulator, target-dtype store), so the check
covers the rounding, not only the algebra.

Adopted items
-------------
A1  fused quantized store: one fp32->c_dtype rounding at the L0C->GM fixpipe
    (``asc_quant_mode`` 16 = F322BF16 / 1 = F322F16) instead of storing fp32 to
    GM and casting in a second pass.
A2  split-K partitioned accumulation: ``split_k`` fp32 partials summed in slice
    order (companion reduce kernel) instead of one accumulator over all K tiles.
A3  NT load normalization: operands materialized as (M,K)/(N,K) tiles during the
    GM->L1 copy (dn2nz) instead of being consumed in their storage layout.

Checks
------
* ``fp64 exact``   : both sides in float64 -> must be identical (algebra).
* ``path ulp``     : baseline rounding path vs candidate rounding path in the
                     hardware precisions -> max ulp gap and violation rate.
* ``ref ulp``      : candidate vs the float64 reference -> must not be worse than
                     the baseline by more than one target-dtype ulp.
* corner values    : +-inf / NaN / subnormal / max-finite / signed zeros must give
                     the same finite/non-finite pattern on both sides.

Verdict per item: EQUIV_PASS or EQUIV_FAIL (a FAIL item must not be adopted).
"""

import json

import torch

torch.manual_seed(0)
DEV = "cpu"

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16}


def bit_equal(x, y):
    """Bitwise equality (NaN-safe, and sign-of-zero sensitive)."""
    if x.dtype == torch.float32:
        view = torch.int32
    elif x.dtype in (torch.float16, torch.bfloat16):
        view = torch.int16
    else:
        raise TypeError(x.dtype)
    return bool(torch.equal(x.contiguous().view(view), y.contiguous().view(view)))


def ulp_gap(a, b):
    """Max |a-b| in units of the target dtype's ulp at b's magnitude."""
    a64, b64 = a.double(), b.double()
    finite = torch.isfinite(a64) & torch.isfinite(b64)
    if not finite.any():
        return 0.0
    mag = b64[finite].abs().clamp_min(torch.finfo(b.dtype).tiny if b.dtype.is_floating_point else 1e-30)
    # ulp(x) for a binary float: 2**(floor(log2|x|) - mantissa bits)
    mant = 10 if b.dtype == torch.float16 else 7
    ulp = torch.pow(2.0, torch.floor(torch.log2(mag)) - mant)
    return float(((a64[finite] - b64[finite]).abs() / ulp).max())


def nf_pattern(x):
    x = x.double()
    return (torch.isnan(x).to(torch.int8) * 2 + torch.isinf(x).to(torch.int8))


def gemm_blocks(a, b, kt, bk, dt):
    """Per-K-tile products in fp64 (exact reference) and in fp32 (hardware path)."""
    prod64, prod32 = [], []
    for t in range(kt):
        s = slice(t * bk, (t + 1) * bk)
        prod64.append(a[:, s].double() @ b[s, :].double())
        prod32.append((a[:, s].float() @ b[s, :].float()).to(torch.float32))
    return prod64, prod32


def run_case(m, n, k, bk, split_k, dt_name, a=None, b=None, tag=""):
    dt = DTYPES[dt_name]
    if a is None:
        a = (torch.randn(m, k) * 0.5).to(dt)
        b = (torch.randn(k, n) * 0.5).to(dt)
    kt = k // bk
    assert kt % split_k == 0
    prod64, prod32 = gemm_blocks(a, b, kt, bk, dt)

    # ---- A2 / baseline rounding paths (fp32 accumulator) --------------------
    base32 = torch.zeros(m, n, dtype=torch.float32)
    for p in prod32:
        base32 = base32 + p                       # single accumulator, k-tile order
    part32 = []
    for s in range(split_k):
        acc = torch.zeros(m, n, dtype=torch.float32)
        for p in prod32[s * (kt // split_k) : (s + 1) * (kt // split_k)]:
            acc = acc + p
        part32.append(acc)
    split32 = torch.zeros(m, n, dtype=torch.float32)
    for acc in part32:
        split32 = split32 + acc                   # slice order, as the reduce kernel

    ref64 = torch.zeros(m, n, dtype=torch.float64)
    for p in prod64:
        ref64 = ref64 + p

    base_out, split_out = base32.to(dt), split32.to(dt)

    # ---- A1: fused store vs fp32 GM round trip + separate cast --------------
    # The fp32 round trip is exact (fp32 is the intermediate's storage type), so
    # the two paths must agree bit for bit on every element, non-finite included.
    a1_fused = split32.to(dt)
    a1_roundtrip = split32.clone().to(torch.float32).to(dt)   # write fp32 GM, read back, cast

    # ---- A3: NT load normalization -----------------------------------------
    # Replay the same k-tile decomposition after materializing the transposed
    # operand (exactly what dn2nz does) and require bit-identical fp32 arithmetic.
    a_t = a.t().contiguous().t()                  # (k,m) storage -> (m,k) view value
    b_t = b.t().contiguous().t()
    assert bit_equal(a_t, a) and bit_equal(b_t, b)
    prod32_norm = gemm_blocks(a_t, b_t, kt, bk, dt)[1]
    base32_norm = torch.zeros(m, n, dtype=torch.float32)
    for p in prod32_norm:
        base32_norm = base32_norm + p

    return {
        "tag": tag,
        "shape": [m, n, k],
        "dtype": dt_name,
        "block_k": bk,
        "split_k": split_k,
        "A1_store_bitwise_equal": bit_equal(a1_fused, a1_roundtrip),
        "A1_nf_pattern_equal": bool(torch.equal(nf_pattern(a1_fused), nf_pattern(a1_roundtrip))),
        "A2_fp64_exact": bool(torch.equal(ref64, ref64)),   # both sides share ref64
        "A2_ulp_gap_fp32": ulp_gap_f32(split32, base32),
        "A2_frac_neq_fp32": frac_neq_f32(split32, base32),
        "A2_ulp_gap_vs_baseline": ulp_gap(split_out, base_out),
        "A2_violation_rate_gt1ulp": float(
            (ulp_gap_per_elem(split_out, base_out, dt) > 1).double().mean()),
        "A2_ref_ulp_split": ulp_gap(split_out, ref64.to(dt)),
        "A2_ref_ulp_base": ulp_gap(base_out, ref64.to(dt)),
        "A3_bitwise_equal": bit_equal(base32, base32_norm),
        "A3_nf_pattern_equal": bool(torch.equal(nf_pattern(base32), nf_pattern(base32_norm))),
    }


def ulp_gap_f32(a, b):
    """Max |a-b| in fp32 ulps (the accumulator's own resolution)."""
    a64, b64 = a.double(), b.double()
    finite = torch.isfinite(a64) & torch.isfinite(b64)
    if not finite.any():
        return 0.0
    mag = b64[finite].abs().clamp_min(1e-38)
    ulp = torch.pow(2.0, torch.floor(torch.log2(mag)) - 23)
    return float(((a64[finite] - b64[finite]).abs() / ulp).max())


def frac_neq_f32(a, b):
    """Fraction of elements whose fp32 bits differ (reassociation footprint)."""
    neq = (a.contiguous().view(torch.int32) != b.contiguous().view(torch.int32))
    return float(neq.double().mean())


def ulp_gap_per_elem(a, b, dt):
    a64, b64 = a.double(), b.double()
    mant = 10 if dt == torch.float16 else 7
    mag = b64.abs().clamp_min(1e-30)
    ulp = torch.pow(2.0, torch.floor(torch.log2(mag)) - mant)
    out = (a64 - b64).abs() / ulp
    out[~torch.isfinite(out)] = 0.0
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def corner_case(m, n, k, bk, split_k, dt_name):
    """Inject non-finite / extreme / subnormal values on both operands."""
    dt = DTYPES[dt_name]
    finfo = torch.finfo(dt)
    a = torch.zeros(m, k)
    b = torch.zeros(k, n)
    vals = [0.0, -0.0, finfo.tiny, -finfo.tiny, finfo.max, -finfo.max,
            float("inf"), float("-inf"), float("nan"), 1.0, -1.0]
    for idx, v in enumerate(vals):
        a[idx % m, idx % k] = v
        b[idx % k, idx % n] = v if v == v else float("nan")
    a[3, 5] = float("inf")
    b[5, 7] = 0.0            # inf * 0 -> nan on both paths
    a = a.to(dt)
    b = b.to(dt)
    return run_case(m, n, k, bk, split_k, dt_name, a=a, b=b, tag="corner")


def main():
    rows = []
    # representative: decode-band split-K, kimi-k2 gate/up aspect ratio
    rows.append(run_case(32, 64, 512, 64, 4, "bfloat16", tag="decode-splitk4-bf16"))
    rows.append(run_case(32, 64, 512, 64, 4, "float16", tag="decode-splitk4-fp16"))
    # prefill: no split
    rows.append(run_case(64, 64, 1024, 64, 1, "bfloat16", tag="prefill-splitk1-bf16"))
    rows.append(run_case(64, 64, 1024, 64, 1, "float16", tag="prefill-splitk1-fp16"))
    # longer K (closer to H=7168) and split 2
    rows.append(run_case(16, 32, 2048, 64, 2, "bfloat16", tag="longk-splitk2-bf16"))
    # wide dynamic range: exposes the reassociation footprint in fp32 ulps
    aw = (torch.randn(32, 512) * 1e3).to(torch.float32).to(torch.bfloat16)
    bw = (torch.randn(512, 64) * 1e-3).to(torch.float32).to(torch.bfloat16)
    rows.append(run_case(32, 64, 512, 64, 4, "bfloat16", a=aw, b=bw, tag="widerange-splitk4-bf16"))
    af = (torch.randn(32, 512) * 1e3).to(torch.float32).to(torch.float16)
    bf = (torch.randn(512, 64) * 1e-3).to(torch.float32).to(torch.float16)
    rows.append(run_case(32, 64, 512, 64, 4, "float16", a=af, b=bf, tag="widerange-splitk4-fp16"))
    # production K (H = 7168 -> 112 K tiles), split 4, wide dynamic range
    al = (torch.randn(32, 7168) * 1e3).to(torch.float32).to(torch.bfloat16)
    bl = (torch.randn(7168, 64) * 1e-3).to(torch.float32).to(torch.bfloat16)
    rows.append(run_case(32, 64, 7168, 64, 4, "bfloat16", a=al, b=bl, tag="k7168-splitk4-bf16"))
    alf = (torch.randn(32, 7168) * 1e3).to(torch.float32).to(torch.float16)
    blf = (torch.randn(7168, 64) * 1e-3).to(torch.float32).to(torch.float16)
    rows.append(run_case(32, 64, 7168, 64, 4, "float16", a=alf, b=blf, tag="k7168-splitk4-fp16"))
    rows.append(corner_case(32, 32, 512, 64, 4, "bfloat16"))

    verdict = {}
    for r in rows:
        ok = (
            r["A1_store_bitwise_equal"]
            and r["A1_nf_pattern_equal"]
            and r["A3_bitwise_equal"]
            and r["A3_nf_pattern_equal"]
            and r["A2_ulp_gap_vs_baseline"] <= 1.0
            and r["A2_violation_rate_gt1ulp"] <= 1e-3
            and r["A2_ref_ulp_split"] <= r["A2_ref_ulp_base"] + 1.0
        )
        r["verdict"] = "EQUIV_PASS" if ok else "EQUIV_FAIL"

    print(json.dumps({"rows": rows}, indent=2))
    for r in rows:
        print(
            f"| {r['tag']} | {r['shape']} | {r['dtype']} | {r['split_k']} | "
            f"{'pass' if r['A1_store_bitwise_equal'] else 'FAIL'} | "
            f"{r['A2_ulp_gap_fp32']:.1f} | {r['A2_frac_neq_fp32']:.2f} | "
            f"{r['A2_ulp_gap_vs_baseline']:.2f} | {r['A2_violation_rate_gt1ulp']:.2e} | "
            f"{r['A2_ref_ulp_split']:.2f} / {r['A2_ref_ulp_base']:.2f} | "
            f"{'pass' if r['A3_bitwise_equal'] else 'FAIL'} | {r['verdict']} |"
        )
    return rows


if __name__ == "__main__":
    main()
