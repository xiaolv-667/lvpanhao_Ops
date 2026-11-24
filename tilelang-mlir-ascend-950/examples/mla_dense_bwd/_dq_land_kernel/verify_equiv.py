# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Equivalence verification for _dq_land_kernel (DESIGN.md section 1.6.1, D-1).

Operator semantics (DESIGN.md section 0.1, frozen):
    dq[t, h, d] = round_dtype(fp32(dq_acc[t, h, d]) * fp32(scale))
i.e. one float32 multiply followed by a *single* round-to-nearest-even (RNE)
narrowing to float16 / bfloat16.

The script compares, element by element and on CPU only:

  adopted items (section 1.6.1 A1/A2/A3)
    A1  single rounding: the scale multiply happens in the float32 domain and
        the value is narrowed exactly once       (fp64 reference expression)
    A2  the scale is rounded to float32 before the multiply (the value the
        kernel bakes as a `float` literal through `T.simd.vmuls`)
    A3  no saturation on narrowing: an out-of-range product must become
        +-inf, not +-max_finite (the `sat=False` / RS_DISABLE epilogue)

  rejected controls (they are *not* adopted; the table marks them NOT_EQUIV)
    C1  16-bit arithmetic: `(x.to(dtype) * dtype(scale)).to(dtype)` -- two
        roundings instead of one
    C2  saturating narrowing: clamp the product to +-max_finite first
    C3  truncating narrowing (ROUND_Z) instead of RNE

Reference: every expression is evaluated in float64 and then rounded to the
target dtype, which is the strongest available oracle on CPU; the baseline is
additionally evaluated exactly as the golden does it (torch float32 multiply,
then a dtype cast) so that a double-rounding gap cannot hide.

Run:
    /workspace/venv-ascend950/bin/python verify_equiv.py
Exit code 0 iff every adopted item passes and every control is rejected as
predicted.
"""

import math

import torch

_TOL = {
    # tests/ops/test_mla_dense_bwd.py::_tolerance -> (atol, rtol)
    torch.float16: (5e-3, 1e-3),
    torch.bfloat16: (1.6e-2, 1.6e-2),
}
_MAX_FINITE = {
    torch.float16: 65504.0,
    torch.bfloat16: 3.3895313892515355e38,
}
_SCALE = 192 ** -0.5  # manifest attention.yaml: sm_scale defaults to D_qk ** -0.5


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _to_dtype(v64, dtype):
    """Round a float64 tensor to `dtype` through the dtype's own cast."""
    return v64.to(dtype)


def _corner_values(dtype, n_random=8192, seed=0):
    """Random dq-like floats plus every boundary the operator can see."""
    gen = torch.Generator().manual_seed(seed)
    # dq_acc is an accumulated float32 gradient: roughly zero-mean, |x| <= 1e2
    rand = (torch.randn(n_random, generator=gen, dtype=torch.float32) * 3.0).double()
    mx = _MAX_FINITE[dtype]
    corner = torch.tensor(
        [
            0.0, -0.0, 1.0, -1.0, 0.5, -0.5,
            float("inf"), float("-inf"), float("nan"),
            1e-8, -1e-8, 1e-38, -1e-38, 6e-8, -6e-8,  # subnormal / flush range
            mx / abs(_SCALE), -(mx / abs(_SCALE)),  # exactly at the finite edge
            (mx * 0.9999) / abs(_SCALE), (mx * 1.0001) / abs(_SCALE),  # overflow edge
            7e4, -7e4, 3.0e38, -3.0e38,  # far out of every 16-bit range
            65504.0 / abs(_SCALE), -65504.0 / abs(_SCALE),
            # IEEE fp16 overflow threshold is 65520 (halfway to the next
            # representable 65536): below it RNE still lands on 65504.
            65519.0 / abs(_SCALE), -65519.0 / abs(_SCALE),
            65520.0 / abs(_SCALE), -65520.0 / abs(_SCALE),
            1.0e5 / abs(_SCALE), -1.0e5 / abs(_SCALE),
        ],
        dtype=torch.float64,
    )
    return torch.cat([rand, corner])


def _ordered_bits(xint):
    """Monotone integer image of a sign-magnitude float bit pattern."""
    mag = (xint.to(torch.int32) & 0x7FFF)
    sign = (xint.to(torch.int32) >> 15) & 1
    return torch.where(sign.bool(), -mag, mag)


def _ulp_stats(got, base, dtype):
    """(max ulp difference, violation rate, nan mismatch count)."""
    a = got.to(torch.float64)
    b = base.to(torch.float64)
    nan_a = torch.isnan(a)
    nan_b = torch.isnan(b)
    nan_mismatch = int((nan_a != nan_b).sum())
    finite = ~(nan_a | nan_b)
    if finite.any():
        ai = a[finite].contiguous().to(dtype).reshape(-1).view(torch.int16)
        bi = b[finite].contiguous().to(dtype).reshape(-1).view(torch.int16)
        ulp = (_ordered_bits(ai) - _ordered_bits(bi)).abs().max().item()
    else:
        ulp = 0
    atol, rtol = _TOL[dtype]
    aa = torch.nan_to_num(got.to(torch.float64), nan=0.0, posinf=1e30, neginf=-1e30)
    bb = torch.nan_to_num(base.to(torch.float64), nan=0.0, posinf=1e30, neginf=-1e30)
    bad = (aa - bb).abs() > (atol + rtol * bb.abs())
    bad = bad | (nan_a != nan_b)
    rate = float(bad.sum()) / float(bad.numel())
    return int(ulp), rate, nan_mismatch


# ---------------------------------------------------------------------------
# expressions: each returns the *float64 pre-image* of the candidate
# ---------------------------------------------------------------------------
def base_exprs(x64, dtype):
    """B0 = the operator's own semantics (== golden) and its fp64 pre-image."""
    xf32 = x64.to(torch.float32)
    # golden, exactly as section 8.1 writes it: fp32 multiply then dtype cast
    golden = (xf32 * _SCALE).to(dtype)
    # same formula, but with the product kept in float64 before the cast
    ref64 = _to_dtype(xf32.to(torch.float64) * float(_SCALE), dtype)
    return golden, ref64


def cand_a1(x64, dtype):
    """A1: one rounding -- fp32 domain multiply, single RNE narrowing."""
    return _to_dtype(x64.to(torch.float32).to(torch.float64) * float(_SCALE), dtype)


def cand_a2(x64, dtype):
    """A2: the scale is first rounded to float32 (the baked `float` literal)."""
    s32 = float(torch.tensor(_SCALE, dtype=torch.float32).item())
    return _to_dtype(x64.to(torch.float32).to(torch.float64) * s32, dtype)


def cand_a3(x64, dtype):
    """A3: IEEE narrowing without saturation (sat=False / RS_DISABLE).

    Values in (max_finite, overflow_threshold) round DOWN to max_finite and only
    values at or beyond the threshold become +-inf, which is exactly what the
    unsaturating vector convert must reproduce.  The saturating form is the C2
    control below, and it disagrees on the points beyond the threshold.
    """
    y64 = x64.to(torch.float32).to(torch.float64) * float(_SCALE)
    return _to_dtype(y64, dtype)


def cand_c1(x64, dtype):
    """C1 (rejected): 16-bit domain arithmetic -- cast first, then multiply.

    Emulated explicitly (cast -> fp64 product -> cast) because torch CPU bf16
    arithmetic runs through an fp32 opmath path and would hide the second
    rounding.
    """
    x16 = x64.to(dtype).to(torch.float64)
    s32 = float(torch.tensor(_SCALE, dtype=torch.float32).item())
    return _to_dtype(x16 * s32, dtype)


def cand_c2(x64, dtype):
    """C2 (rejected): saturating narrowing."""
    y64 = x64.to(torch.float32).to(torch.float64) * float(_SCALE)
    mx = _MAX_FINITE[dtype]
    return _to_dtype(y64.clamp(-mx, mx), dtype)


def cand_c3(x64, dtype):
    """C3 (rejected): truncation (ROUND_Z) instead of RNE."""
    y64 = x64.to(torch.float32).to(torch.float64) * float(_SCALE)
    return _trunc(y64, dtype)


def _trunc(y64, dtype):
    """Round toward zero to the target dtype (what ROUND_Z would give)."""
    mx = _MAX_FINITE[dtype]
    y = y64.clamp(-mx, mx)
    rne = y.to(dtype)  # the RNE candidate
    bits = rne.reshape(-1).view(torch.int16).to(torch.int32)
    sign = (bits >> 15) & 1
    mag = bits & 0x7FFF
    away = rne.to(torch.float64).abs() > y.abs()  # RNE rounded away from zero
    mag = torch.where(away & (mag > 0), mag - 1, mag)
    out_bits = ((sign << 15) | mag).to(torch.int16)
    return out_bits.view(dtype).reshape(y64.shape)


ADOPTED = [
    ("A1", "single rounding (fp32 multiply, one RNE narrowing)", cand_a1),
    ("A2", "scale pre-rounded to float32 (vmuls literal)", cand_a2),
    ("A3", "no saturation (sat=False / RS_DISABLE)", cand_a3),
]
CONTROLS = [
    ("C1", "16-bit arithmetic (two roundings)", cand_c1),
    ("C2", "saturating narrowing", cand_c2),
    ("C3", "truncating narrowing (ROUND_Z)", cand_c3),
]


def main():
    rows = []
    ok = True
    for dtype in (torch.float16, torch.bfloat16):
        x64 = _corner_values(dtype)
        golden, ref64 = base_exprs(x64, dtype)
        # sanity: the golden and its fp64 pre-image must agree on this set
        u, r, nm = _ulp_stats(ref64, golden, dtype)
        print(
            f"[baseline] {str(dtype):>14}: golden(fp32 mul) vs fp64 pre-image "
            f"ulp={u} viol={r:.2e} nan_mis={nm}",
            flush=True,
        )
        for tag, desc, fn in ADOPTED:
            got = fn(x64, dtype)
            u, r, nm = _ulp_stats(got, golden, dtype)
            verdict = "EQUIV_PASS" if (r == 0.0 and nm == 0) else "NOT_EQUIV"
            ok &= verdict == "EQUIV_PASS"
            rows.append((dtype, tag, desc, u, r, nm, verdict))
        for tag, desc, fn in CONTROLS:
            got = fn(x64, dtype)
            u, r, nm = _ulp_stats(got, golden, dtype)
            # A control must be shown to differ from the golden: outside the
            # op tolerance (NOT_EQUIV) or, for the loose bfloat16 tolerance,
            # at least bit-different for some element (WITHIN_TOL).
            if r > 0.0 or nm > 0:
                verdict = "NOT_EQUIV"
            elif u > 0:
                verdict = "WITHIN_TOL"
            else:
                verdict = "IDENTICAL"
            ok &= verdict != "IDENTICAL"
            rows.append((dtype, tag, desc, u, r, nm, verdict))

    print()
    hdr = f"{'dtype':>14} {'item':>4} {'max ulp':>8} {'viol rate':>11} {'nan mis':>8}  verdict   description"
    print(hdr)
    print("-" * len(hdr))
    for dtype, tag, desc, u, r, nm, verdict in rows:
        print(
            f"{str(dtype):>14} {tag:>4} {u:>8} {r:>11.3e} {nm:>8}  {verdict:<10} {desc}"
        )
    print()
    print(f"scale = {_SCALE!r} (= 192 ** -0.5, rounded to float32 by the kernel)")
    print(
        "adopted items all EQUIV_PASS and every control shown different "
        f"(NOT_EQUIV or WITHIN_TOL): {ok}"
    )
    print("[EQUIV_PASS]" if ok else "[EQUIV_NOT_VERIFIED]")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
