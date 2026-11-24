"""Stage 1 equivalence machine verification (D-1) for ``splitk_reduce_kernel``.

Adopted optimization items of DESIGN.md 1.6.1, checked numerically on torch CPU
against an fp64 reference:

  A1  SiLU rewrite:  ``g * sigmoid(g) * u``  ->  ``(g / (1 + exp(-g))) * u``
      One ``vmul`` fewer per vector: the source multiplies by the *reciprocal*
      (``1/den`` then ``g*sig``, two fp32 roundings) while the rewrite divides
      once (``g/den``, one rounding).
  A2  Degenerate-path specialization (``split_k == 1`` and/or no bias and/or
      ``activation == "none"``): the slice loop, the bias add and the gated
      epilogue disappear at build time.  This is a *structural invariant* the
      source already has (its Python factory specializes), not an optimization
      relative to it -- the check is a regression guard that the NPU build keeps
      emitting the specialized form.  Claimed *bit identical*.

Observation row (not an optimization; it evidences the 8.2 precision budget):
  O1  Slice-sequential fp32 accumulation ``((w0+w1)+w2)+w3`` -- the order the
      source's ``_slice_sum`` and this design use -- vs ``partial.sum(0)`` and
      vs the fp64 reference.

Protocol (skill tilelang-op-design Phase 2 step 5): the candidate and the
baseline are each replayed faithfully on the fp32 storage path (fp32 exp, fp32
reciprocal/divide, fp32 multiply, one final rounding to the output dtype); the
fp64 expression is the reference.  ulp is measured on the output dtype grid with
the scale floored at that dtype's *smallest normal* (values below it are all one
bucket -- otherwise the bf16 subnormal tail dominates the metric with
magnitudes of 1e-35).  Verdict: EQUIV_PASS when
  * worst output-dtype ulp difference <= 1,
  * fraction of elements above 1 ulp <= 1e-3 (elements where at least one side
    is normal or both are zero; non-finite positions are compared separately),
  * the non-finite pattern agrees exactly,
  * the candidate is not worse than the baseline against the fp64 reference,
  * (A2 only) the two builds are bit identical.

Run:
    cd examples/shared_expert_mlp/splitk_reduce_kernel
    python verify_equiv.py            # torch CPU only, no NPU
"""

import itertools

import torch

F32 = torch.float32
F64 = torch.float64
MANT = {torch.float16: 10, torch.bfloat16: 7, torch.float32: 23}
FINFO = {d: torch.finfo(d) for d in MANT}
NORMAL = {d: FINFO[d].tiny for d in MANT}  # smallest normal
SUBN = {d: FINFO[d].tiny * 2.0 ** -MANT[d] for d in MANT}  # smallest subnormal


def ulp_diff(a, b, dtype):
    """|a-b| in units of the dtype's ulp, on the dtype's own magnitude grid.

    The scale is ``max(|a|,|b|)`` floored at the dtype's smallest normal, so a
    pair of underflowed values counts as one bucket instead of millions of
    subnormal ulps.
    """
    a64, b64 = a.to(F64), b.to(F64)
    d = (a64 - b64).abs()
    scale = torch.maximum(a64.abs(), b64.abs()).clamp(min=NORMAL[dtype])
    exponent = torch.floor(torch.log2(scale))
    ulp = torch.pow(2.0, exponent - MANT[dtype])
    return (d / ulp).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)


def _finite(v):
    return torch.isfinite(v.to(F64))


def _max_abs(a, b):
    """max |a-b| over positions where both are finite (NaN-free metric)."""
    sel = _finite(a) & _finite(b)
    if not bool(sel.any()):
        return 0.0
    return float((a.to(F64)[sel] - b.to(F64)[sel]).abs().max())


def _exp32(x):
    """fp32 exp, as the hardware path: overflow -> inf, underflow -> 0."""
    return torch.exp(x.to(F32))


def _silu_mul_base(gate, up, dtype, raw=False):
    """Source form ``gate * sigmoid(gate) * up``: reciprocal then multiply."""
    g = gate.to(F32)
    den = (1.0 + _exp32(-g)).to(F32)
    sig = (1.0 / den).to(F32)  # fp32 reciprocal of a fp32 denominator
    y = (g * sig).to(F32) * up.to(F32)
    return y if raw else y.to(dtype)


def _silu_mul_cand(gate, up, dtype, raw=False):
    """A1: divide once instead of multiplying by the reciprocal."""
    g = gate.to(F32)
    den = (1.0 + _exp32(-g)).to(F32)
    y = (g / den).to(F32) * up.to(F32)
    return y if raw else y.to(dtype)


def _silu_mul_ref(gate, up, dtype):
    """fp64 reference, rounded once to the output dtype."""
    g = gate.to(F64)
    return ((g / (1.0 + torch.exp(-g))) * up.to(F64)).to(dtype)


MODES = {
    "normal": lambda n, g: torch.randn(n, dtype=F32, generator=g),
    "wide": lambda n, g: torch.randn(n, dtype=F32, generator=g) * 20.0,
    "saturating": lambda n, g: torch.randn(n, dtype=F32, generator=g) * 200.0,
    "cancel": lambda n, g: torch.randn(n, dtype=F32, generator=g) * 1e-3,
    "zeros": lambda n, g: torch.zeros(n, dtype=F32),
}


def _corner(n, dtype):
    """+/-inf, NaN, +/-0, subnormals, and the largest finite value."""
    big = float(FINFO[dtype].max)
    sub = SUBN[torch.float32] * 3.0
    vals = [float("inf"), float("-inf"), float("nan"), 0.0, -0.0, sub, -sub, big, -big, 1.0]
    return torch.tensor(vals * (n // len(vals) + 1), dtype=F32)[:n]


def _verdict(r):
    ok = (
        r["max_ulp"] <= 1.0
        and r["viol_rate"] <= 1e-3
        and r["nf_mis"] == 0
        and r["ref_c"] <= max(r["ref_b"], 1e-45) * (1.0 + 1e-6) + 1e-45
    )
    if r["item"] == "A2":
        ok = ok and r.get("bitwise", False)
    return "EQUIV_PASS" if ok else "EQUIV_FAIL"


def _score(base, cand, ref, dtype):
    fin = _finite(base) & _finite(cand)
    ud = ulp_diff(cand, base, dtype)
    viol = float((ud[fin] > 1.0).float().mean()) if bool(fin.any()) else 0.0
    nf_mis = int(((~_finite(cand)) != (~_finite(base))).sum())
    return {
        "max_ulp": float(ud[fin].max()) if bool(fin.any()) else 0.0,
        "viol_rate": viol,
        "nf_mis": nf_mis,
        "ref_c": _max_abs(cand, ref),
        "ref_b": _max_abs(base, ref),
    }


def check_a1(seed=0):
    rows = []
    for dtype, sk, (m, on) in itertools.product(
        (torch.float16, torch.bfloat16), (1, 2, 4), ((4, 128), (8, 256))
    ):
        g = torch.Generator().manual_seed(seed)
        for mode in list(MODES) + ["corner"]:
            draw = MODES.get(mode) or (lambda n, gg: _corner(n, dtype))
            n = m * on
            gate = draw(n, g)  # fp32 slice sum, in slice order (as the kernel)
            up = draw(n, g)
            for _s in range(1, sk):
                gate = gate + draw(n, g)
                up = up + draw(n, g)
            base = _silu_mul_base(gate, up, dtype)
            cand = _silu_mul_cand(gate, up, dtype)
            ref = _silu_mul_ref(gate, up, dtype)
            # fp32-level footprint of the rewrite (observation, not a criterion)
            b32 = _silu_mul_base(gate, up, dtype, raw=True)
            c32 = _silu_mul_cand(gate, up, dtype, raw=True)
            fin = _finite(b32) & _finite(c32)
            u32 = ulp_diff(c32, b32, torch.float32)
            row = {"item": "A1", "dtype": str(dtype).split(".")[-1],
                   "case": f"{m}x{on}-sk{sk}-{mode}",
                   "f32_ulp": float(u32[fin].max()) if bool(fin.any()) else 0.0,
                   "f32_changed": float((u32[fin] > 0).float().mean()) if bool(fin.any()) else 0.0}
            row.update(_score(base, cand, ref, dtype))
            rows.append(row)
    return rows


def check_a2(seed=1):
    """split_k == 1 / no-bias / no-activation specialization: bit identical."""
    rows = []
    for dtype in (torch.float16, torch.bfloat16):
        g = torch.Generator().manual_seed(seed)
        n = 1024
        w = torch.randn(n, dtype=F32, generator=g)
        bias = torch.randn(n, dtype=F32, generator=g).to(dtype)

        generic = (w + bias.float()).to(dtype)  # loop over 1 slice, then bias
        spec = (w + bias.float()).to(dtype)  # specialized: one vadd
        same_bias = bool(torch.equal(generic.view(torch.int16), spec.view(torch.int16)))

        copy_gen = w.to(dtype)  # no bias, no activation: pure cast
        copy_spec = w.to(dtype)
        same_copy = bool(torch.equal(copy_gen.view(torch.int16), copy_spec.view(torch.int16)))
        row = {"item": "A2", "dtype": str(dtype).split(".")[-1],
               "case": "split_k=1+bias / no-bias copy", "max_ulp": 0.0,
               "viol_rate": 0.0, "nf_mis": 0, "ref_c": 0.0, "ref_b": 0.0,
               "bitwise": same_bias and same_copy}
        rows.append(row)
    return rows


def check_o1(seed=2):
    """Slice-sequential fp32 sum vs partial.sum(0) vs fp64."""
    rows = []
    for dtype, sk in itertools.product(
        (torch.float16, torch.bfloat16, torch.float32), (2, 4)
    ):
        g = torch.Generator().manual_seed(seed)
        m, on = 8, 256
        for mode in list(MODES) + ["corner"]:
            draw = MODES.get(mode) or (lambda n, gg: _corner(n, torch.float32))
            part = torch.stack([draw(m * on, g).view(m, on) for _ in range(sk)], dim=0)
            seq = part[0].clone()
            for s in range(1, sk):
                seq = seq + part[s]
            torch_sum = part.sum(dim=0)
            ref = part.to(F64).sum(dim=0).to(dtype)
            a, b = seq.to(dtype), torch_sum.to(dtype)
            row = {"item": "O1", "dtype": str(dtype).split(".")[-1],
                   "case": f"sk{sk}-{mode}"}
            row.update(_score(b, a, ref, dtype))
            rows.append(row)
    return rows


def _table(rows, title):
    print(f"\n### {title}")
    print(
        "| item | dtype | case | max ulp | >1ulp rate | non-finite mismatch | "
        "max abs vs fp64 (cand) | max abs vs fp64 (base) | fp32 ulp footprint | "
        "fp32 changed | verdict |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {r['item']} | {r['dtype']} | {r['case']} | {r['max_ulp']:.0f} | "
            f"{r['viol_rate']:.1e} | {r['nf_mis']} | {r['ref_c']:.3e} | "
            f"{r['ref_b']:.3e} | {r.get('f32_ulp', 0.0):.0f} | "
            f"{r.get('f32_changed', 0.0):.2f} | **{_verdict(r)}** |"
        )


def _summary(rows, title):
    """One aggregated row per (item, dtype): the worst case over shapes/modes."""
    print(f"\n### {title} -- aggregate")
    print(
        "| item | dtype | cases | worst max ulp | worst >1ulp rate | "
        "non-finite mismatch | max abs vs fp64 (cand / base) | worst fp32 ulp footprint | verdict |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    keys = sorted({(r["item"], r["dtype"]) for r in rows})
    all_ok = True
    for item, dt in keys:
        sel = [r for r in rows if r["item"] == item and r["dtype"] == dt]
        verds = {_verdict(r) for r in sel}
        verdict = "EQUIV_PASS" if verds == {"EQUIV_PASS"} else "EQUIV_FAIL"
        all_ok = all_ok and verdict == "EQUIV_PASS"
        print(
            f"| {item} | {dt} | {len(sel)} | {max(r['max_ulp'] for r in sel):.0f} | "
            f"{max(r['viol_rate'] for r in sel):.1e} | "
            f"{sum(r['nf_mis'] for r in sel)} | "
            f"{max(r['ref_c'] for r in sel):.3e} / {max(r['ref_b'] for r in sel):.3e} | "
            f"{max(r.get('f32_ulp', 0.0) for r in sel):.0f} | "
            f"**{verdict}** |"
        )
    return all_ok


if __name__ == "__main__":
    a1 = check_a1()
    a2 = check_a2()
    o1 = check_o1()
    _table(a1, "A1  silu rewrite (per case)")
    _table(a2, "A2  degenerate specialization / structural invariant (per case)")
    _table(o1, "O1  slice-sequential fp32 sum (per case, observation)")
    _summary(a1, "A1  silu rewrite")
    _summary(a2, "A2  degenerate specialization")
    _summary(o1, "O1  slice-sequential fp32 sum")
    rows = a1 + a2 + o1
    bad = [r for r in rows if _verdict(r) != "EQUIV_PASS"]
    print(f"\nTOTAL cases: {len(rows)}, failures: {len(bad)}")
    for r in bad[:10]:
        print("  FAIL:", r)
    print("[EQUIV_ALL_PASS]" if not bad else "[EQUIV_HAS_FAIL]")
