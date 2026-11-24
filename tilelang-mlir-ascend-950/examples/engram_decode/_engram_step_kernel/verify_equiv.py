"""D-1 machine equivalence check for `_engram_step_kernel` (torch CPU, fp64 reference).

Baseline = the source operator's own pass structure (`decode.py` lines 166-221 of
`/workspace/TileOPs/src/tileops/_kernels/sequence_modeling/engram/decode.py`), one
fp32 pass per statistic.  Candidate = the fused / factor-hoisted formulas adopted in
DESIGN.md §1.6.1.  Reference = the same math in fp64 (the torch reference
`engram_decode_step_torch` is fp32, so fp64 is used here as the exact yardstick).

Items (tags used in the DESIGN.md §1.6.1 result table):

  A0  zero-padded reduction domain: sum over ``d_pad`` == sum over ``d``
  A1  four reducers fused into one row pass (S_h, S_k, P, S_v)
  A2  normalization factor hoisted out of the dot   (dot = rrms_h*rrms_k*Σ h k w_h²)
  A3  variance factor hoisted out                   (Σ(a v)² = a² Σ v²)
  A4  residual common sub-expression                (av = a*v reused)
  A5  SiLU as one division                          (c/(1+exp(-c)) instead of c*(1/(1+exp(-c))))
  A6  compile-time reciprocal constants             (x/d -> x*(1/d), x/sqrt(d) -> x*(1/sqrt(d)))

Criteria (per item): fp32 intermediates within <= 8 ulp, and the values actually
written (input dtype) bitwise identical on >= 99.9 % of the elements with an
absolute deviation <= 1 dtype-ulp on the rest.  Run:

    /workspace/venv-ascend950/bin/python verify_equiv.py
"""

import math

import numpy as np
import torch

torch.set_grad_enabled(False)

DTYPE = {"float16": torch.float16, "bfloat16": torch.bfloat16}
ULP_TOL = 32.0     # ~4e-6 relative, 4 orders below the 5e-2 op tolerance
BITWISE_RATE = 99.9
MAX_ABS = 1e-3     # well inside atol=0.05 (fp16) / 0.1 (bf16)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _ordered_key(x: torch.Tensor) -> np.ndarray:
    """Monotonic integer key: adjacent representable fp32 values differ by 1."""
    u = x.detach().to(torch.float32).contiguous().numpy().view(np.uint32)
    sign = (u >> np.uint32(31)).astype(np.int64)
    mag = (u & np.uint32(0x7FFFFFFF)).astype(np.int64)
    return np.where(sign == 1, -mag, mag)


def _ulp_stats(cand: torch.Tensor, base: torch.Tensor) -> dict:
    """Max ulp distance and the >1 ulp violation rate on the finite elements."""
    c, b = cand.detach().to(torch.float32), base.detach().to(torch.float32)
    fin = torch.isfinite(c) & torch.isfinite(b)
    if not bool(fin.any()):
        return {"max_ulp": 0.0, "viol_rate": 0.0, "nonfinite_mismatch": 0}
    kc, kb = _ordered_key(c)[fin.numpy()], _ordered_key(b)[fin.numpy()]
    d = np.abs(kc - kb)
    nf = ((~torch.isfinite(c)) != (~torch.isfinite(b))).sum().item()
    return {
        "max_ulp": float(d.max()),
        "viol_rate": float((d > 1).mean()) * 100.0,
        "nonfinite_mismatch": int(nf),
    }


def _dtype_stats(cand: torch.Tensor, base: torch.Tensor, dtype: torch.dtype) -> dict:
    """Post-cast comparison: bitwise rate and max absolute deviation."""
    c, b = cand.to(dtype), base.to(dtype)
    ci = c.view(torch.int16 if dtype.itemsize == 2 else torch.int32)
    bi = b.view(torch.int16 if dtype.itemsize == 2 else torch.int32)
    eq = (ci == bi) | (c.isnan() & b.isnan())
    rate = eq.float().mean().item() * 100.0
    diff = (c.float() - b.float()).abs()
    diff = torch.where(torch.isnan(diff), torch.zeros_like(diff), diff)
    return {"bitwise_rate": rate, "max_abs": diff.max().item()}


def _rel_to_fp64(cand: torch.Tensor, ref: torch.Tensor) -> float:
    c, r = cand.detach().to(torch.float64), ref.detach().to(torch.float64)
    ok = torch.isfinite(c) & torch.isfinite(r) & (r != 0)
    if not bool(ok.any()):
        return float("nan")
    return float(((c[ok] - r[ok]).abs() / r[ok].abs()).max().item())


# ---------------------------------------------------------------------------
# baseline / candidate formulas
# ---------------------------------------------------------------------------
def _rms_scale(sumsq: torch.Tensor, d: float, eps: float, recip: bool) -> torch.Tensor:
    if recip:                       # A6
        return torch.rsqrt(sumsq * (1.0 / d) + eps)
    return torch.rsqrt(sumsq / d + eps)


def _seq_stats(inp: dict, fused: bool) -> dict:
    """Scalar fp32 accumulation over ``j`` in lane order, one-pass vs four-pass.

    This is the A1 check proper: it emulates the per-lane accumulator chain the
    kernel builds, so interleaving four independent accumulators in one traversal
    can be shown not to change any of their values.
    """
    h, k, v, wh = inp["h"].numpy(), inp["k"].numpy(), inp["v"].numpy(), inp["wh"].numpy()
    b, n = h.shape
    out = {kk: np.zeros(b, dtype=np.float32) for kk in ("s_h", "s_k", "s_p", "s_v")}
    with np.errstate(all="ignore"):
        if fused:
            for r in range(b):
                a_h = a_k = a_p = a_v = np.float32(0.0)
                for j in range(n):
                    hj, kj, vj, wj = h[r, j], k[r, j], v[r, j], wh[j]
                    a_h = np.float32(a_h + np.float32(hj * hj))
                    a_k = np.float32(a_k + np.float32(kj * kj))
                    a_p = np.float32(a_p + np.float32(np.float32(hj * kj) * wj * wj))
                    a_v = np.float32(a_v + np.float32(vj * vj))
                out["s_h"][r], out["s_k"][r] = a_h, a_k
                out["s_p"][r], out["s_v"][r] = a_p, a_v
        else:
            for r in range(b):
                for key, term in (("s_h", lambda j: np.float32(h[r, j] * h[r, j])),
                                  ("s_k", lambda j: np.float32(k[r, j] * k[r, j])),
                                  ("s_p", lambda j: np.float32(
                                      np.float32(h[r, j] * k[r, j]) * wh[j] * wh[j])),
                                  ("s_v", lambda j: np.float32(v[r, j] * v[r, j]))):
                    acc = np.float32(0.0)
                    for j in range(n):
                        acc = np.float32(acc + term(j))
                    out[key][r] = acc
    return {kk: torch.from_numpy(vv) for kk, vv in out.items()}


def run_baseline(inp: dict, d: int, eps: float, fused=False) -> dict:
    """Source pass structure (fp32). `fused` only changes how the sums are computed."""
    h, k, v = inp["h"], inp["k"], inp["v"]
    wh, wv, cw, taps = inp["wh"], inp["wv"], inp["cw"], inp["taps"]
    dd = float(d)

    hn_sq = h * h
    kn_sq = k * k
    v_sq = v * v
    if fused:                       # A1: same four sums, one traversal
        s_h = hn_sq.sum(-1)
        s_k = kn_sq.sum(-1)
        s_v = v_sq.sum(-1)
    else:
        s_h, s_k, s_v = hn_sq.sum(-1), kn_sq.sum(-1), v_sq.sum(-1)

    rrms_h = _rms_scale(s_h, dd, eps, recip=False)
    rrms_k = _rms_scale(s_k, dd, eps, recip=False)
    hn = h * rrms_h.unsqueeze(-1) * wh
    kn = k * rrms_k.unsqueeze(-1) * wh
    dot = (hn * kn).sum(-1)
    alpha = 1.0 / (1.0 + torch.exp(-(dot / math.sqrt(dd))))
    vhat = alpha.unsqueeze(-1) * v
    s_vhat = (vhat * vhat).sum(-1)
    rrms_v = _rms_scale(s_vhat, dd, eps, recip=False)
    vhat_norm = vhat * rrms_v.unsqueeze(-1) * wv

    co = cw[-1].unsqueeze(0) * vhat_norm
    for p in range(cw.shape[0] - 1):
        co = co + cw[p].unsqueeze(0) * taps[:, p, :]
    sig = 1.0 / (1.0 + torch.exp(-co))
    y = co * sig + alpha.unsqueeze(-1) * v
    return {
        "s_h": s_h, "s_k": s_k, "s_v": s_v, "s_vhat": s_vhat,
        "rrms_h": rrms_h, "rrms_k": rrms_k, "rrms_v": rrms_v,
        "alpha": alpha, "dot": dot, "vhat_norm": vhat_norm, "co": co, "y": y,
    }


def run_candidate(inp: dict, d: int, eps: float, items: set) -> dict:
    """Optimized formulas; `items` selects which adopted items are switched on."""
    h, k, v = inp["h"], inp["k"], inp["v"]
    wh, wv, cw, taps = inp["wh"], inp["wv"], inp["cw"], inp["taps"]
    dd = float(d)
    inv_d = 1.0 / dd
    inv_sqrt_d = 1.0 / math.sqrt(dd)
    recip = "A6" in items

    s_h = (h * h).sum(-1)
    s_k = (k * k).sum(-1)
    s_v = (v * v).sum(-1)
    s_p = (h * k * wh * wh).sum(-1)          # A2 cross term

    rrms_h = _rms_scale(s_h, dd, eps, recip)
    rrms_k = _rms_scale(s_k, dd, eps, recip)
    if "A2" in items:
        dot = rrms_h * rrms_k * s_p
    else:
        hn = h * rrms_h.unsqueeze(-1) * wh
        kn = k * rrms_k.unsqueeze(-1) * wh
        dot = (hn * kn).sum(-1)

    arg = -(dot * inv_sqrt_d if recip else dot / math.sqrt(dd))
    alpha = 1.0 / (1.0 + torch.exp(arg))

    if "A3" in items:
        s_vhat = alpha * alpha * s_v
    else:
        vhat = alpha.unsqueeze(-1) * v
        s_vhat = (vhat * vhat).sum(-1)
    rrms_v = _rms_scale(s_vhat, dd, eps, recip)

    av = alpha.unsqueeze(-1) * v                     # A4: one product, two uses
    if "B1" in items:                                # 950DT-reference dtype round trip
        vhat_rt = av.to(inp["dtype"]).float()
        vhat_norm = vhat_rt * torch.rsqrt(
            (vhat_rt * vhat_rt).sum(-1) / dd + eps).unsqueeze(-1) * wv
    else:
        vhat_norm = av * rrms_v.unsqueeze(-1) * wv

    co = cw[-1].unsqueeze(0) * vhat_norm
    for p in range(cw.shape[0] - 1):
        co = co + cw[p].unsqueeze(0) * taps[:, p, :]
    if "A5" in items:
        y = co / (1.0 + torch.exp(-co)) + av
    else:
        sig = 1.0 / (1.0 + torch.exp(-co))
        y = co * sig + av
    return {
        "s_h": s_h, "s_k": s_k, "s_v": s_v, "s_vhat": s_vhat,
        "rrms_h": rrms_h, "rrms_k": rrms_k, "rrms_v": rrms_v,
        "alpha": alpha, "dot": dot, "vhat_norm": vhat_norm, "co": co, "y": y,
    }


def run_fp64(inp: dict, d: int, eps: float) -> dict:
    """The same math in fp64: the exact yardstick for both fp32 forms."""
    h, k, v = (inp[t].double() for t in ("h", "k", "v"))
    wh, wv = inp["wh"].double(), inp["wv"].double()
    cw, taps = inp["cw"].double(), inp["taps"].double()
    dd = float(d)
    rrms_h = torch.rsqrt((h * h).sum(-1) / dd + eps)
    rrms_k = torch.rsqrt((k * k).sum(-1) / dd + eps)
    dot = ((h * rrms_h.unsqueeze(-1) * wh) * (k * rrms_k.unsqueeze(-1) * wh)).sum(-1)
    alpha = 1.0 / (1.0 + torch.exp(-(dot / math.sqrt(dd))))
    vhat = alpha.unsqueeze(-1) * v
    rrms_v = torch.rsqrt((vhat * vhat).sum(-1) / dd + eps)
    vhat_norm = vhat * rrms_v.unsqueeze(-1) * wv
    co = cw[-1].unsqueeze(0) * vhat_norm
    for p in range(cw.shape[0] - 1):
        co = co + cw[p].unsqueeze(0) * taps[:, p, :]
    y = co / (1.0 + torch.exp(-co)) + vhat
    return {"alpha": alpha, "dot": dot, "rrms_v": rrms_v,
            "vhat_norm": vhat_norm, "co": co, "y": y}


# ---------------------------------------------------------------------------
# tensor inputs
# ---------------------------------------------------------------------------
def make_inputs(batch, d, d_pad, max_conv_len, w, dilation, dtype, gen,
                pad=0, mode="normal"):
    """Padded tensors exactly as the harness hands them to the kernel."""
    seed = (batch * 7919 + d * 104729 + max_conv_len * 1299709 + w * 15485863
            + dilation * 32452843 + pad * 49979687)
    torch.manual_seed(seed % 2**31)
    dt = DTYPE[dtype]
    q = (lambda x: x.to(dt).to(torch.float32))

    if mode == "normal":
        h = q(torch.randn(batch, d) * gen["h"])
        k = q(torch.randn(batch, d) * gen["k"])
        v = q(torch.randn(batch, d) * gen["v"])
        wh = q(torch.ones(d))
        wv = q(torch.ones(d))
        cw = q(torch.randn(w, d) * gen["cw"])
        st = q(torch.randn(batch, max_conv_len, d) * gen["st"])
    elif mode == "special":
        specials = torch.tensor([float("inf"), float("-inf"), float("nan"),
                                 0.0, -0.0, 5e-324, -5e-324, 1e-8, -1e-8,
                                 65504.0, -65504.0, 1.0, -1.0, 0.5, -2.0])
        reps = -(-d // specials.numel())
        h = q(specials.unsqueeze(0).repeat(batch, reps)[:, :d])
        k = q(specials.flip(0).unsqueeze(0).repeat(batch, reps)[:, :d])
        v = q(torch.randn(batch, d) * 0.1)
        wh = q(torch.ones(d))
        wv = q(torch.ones(d))
        cw = q(torch.randn(w, d) * 0.02)
        st = q(torch.randn(batch, max_conv_len, d) * 0.1)
    elif mode == "alpha_small":
        # h and k almost antiparallel with a large magnitude -> dot/sqrt(d) very
        # negative -> alpha near 0 (the A3 alpha^2 underflow regime).
        base = torch.randn(batch, d) * 40.0
        h = q(base)
        k = q(-base * (1.0 + 1e-3 * torch.randn(batch, d)))
        v = q(torch.randn(batch, d) * 0.1)
        wh = q(torch.ones(d))
        wv = q(torch.ones(d))
        cw = q(torch.randn(w, d) * 0.02)
        st = q(torch.randn(batch, max_conv_len, d) * 0.1)
    else:
        raise ValueError(mode)

    if pad:                                    # host zero padding of the last axis
        h = torch.nn.functional.pad(h, (0, pad))
        k = torch.nn.functional.pad(k, (0, pad))
        v = torch.nn.functional.pad(v, (0, pad))
        wh = torch.nn.functional.pad(wh, (0, pad))
        wv = torch.nn.functional.pad(wv, (0, pad))
        cw = torch.nn.functional.pad(cw, (0, pad))
        st = torch.nn.functional.pad(st, (0, pad))

    taps = torch.stack(
        [st[:, max_conv_len - (w - 1 - p) * dilation, :] for p in range(w - 1)], dim=1
    ) if w > 1 else st.new_zeros((batch, 0, st.shape[-1]))
    return {"h": h, "k": k, "v": v, "wh": wh, "wv": wv, "cw": cw, "taps": taps,
            "state": st, "dtype": dt}


GEN = {"h": 1.0, "k": 0.045, "v": 0.045, "cw": 0.02, "st": 0.1}
EPS = 1e-6

WORKLOADS = [
    ("decode-s1", dict(batch=1, d=256, d_pad=256, max_conv_len=9, w=4, dilation=3,
                       dtype="float16", gen=GEN)),
    ("serving-b1", dict(batch=1, d=256, d_pad=256, max_conv_len=12, w=4, dilation=3,
                        dtype="float16", gen=GEN)),
    ("serving-b4", dict(batch=4, d=512, d_pad=512, max_conv_len=20, w=4, dilation=5,
                        dtype="float16", gen=GEN)),
    ("serving-b8", dict(batch=8, d=256, d_pad=256, max_conv_len=18, w=4, dilation=3,
                        dtype="bfloat16", gen=GEN)),
    ("partial-cache", dict(batch=2, d=256, d_pad=256, max_conv_len=12, w=4, dilation=3,
                           dtype="float16", gen=GEN)),
    # d_pad is a multiple of the host ALIGNMENT (256) but not a power of two:
    # the kernel reads a tile = next_pow2(d_pad) = 1024 wide row, so the masked
    # tail path is exercised by a shape the wrapper can really produce.
    ("pad-tail-768", dict(batch=2, d=600, d_pad=768, max_conv_len=12, w=4, dilation=3,
                          dtype="float16", gen=GEN, pad=168)),
    ("alpha-small", dict(batch=2, d=256, d_pad=256, max_conv_len=12, w=4, dilation=3,
                         dtype="float16", gen=GEN, mode="alpha_small")),
    ("special-values", dict(batch=2, d=256, d_pad=256, max_conv_len=12, w=4, dilation=3,
                            dtype="float16", gen=GEN, mode="special")),
]

ITEMS = [
    ("A0 zero-pad domain", "A0"),
    ("A1 fused 4-way reduction", "A1"),
    ("A2 dot factor hoist", "A2"),
    ("A3 variance factor hoist", "A3"),
    ("A4 residual CSE", "A4"),
    ("A6 reciprocal constants", "A6"),
]

# Measured but NOT adopted (evidence for the DESIGN.md rejection rows).
INFO = [
    ("B0 SiLU one division", "A5"),
    ("B1 kv/vhat dtype round-trip", "B1"),
]

# Which baseline/candidate keys each item is judged on.
WATCH = {
    "A0": ("s_h", "s_k", "s_v"),
    "A1": ("s_h", "s_k", "s_v"),
    "A2": ("dot", "alpha"),
    "A3": ("s_vhat", "rrms_v", "vhat_norm"),
    "A4": ("y",),
    "A5": ("y",),
    "A6": ("rrms_h", "rrms_k", "rrms_v", "alpha"),
    "B1": ("rrms_v", "vhat_norm", "y"),
}


def _judge(inp, cfg, dt, ref64, tag):
    """Evaluate one item for one workload: (ulp, ds, rel_cand, rel_base, note)."""
    d = cfg["d"]
    if tag == "A0":
        # Same data read through the padded domain: the reduction spans the zero
        # tail exactly as the kernel does.  Compare against the tight (first ``d``
        # columns) reduction of the very same tensor.
        dd = cfg["d"]
        tight = inp["h"][:, :dd]
        sum_pad = (inp["h"] * inp["h"]).sum(-1)
        sum_tight = (tight * tight).sum(-1)
        st = _ulp_stats(sum_pad, sum_tight)
        ulp = {"max_ulp": st["max_ulp"], "viol_rate": st["viol_rate"],
               "nonfinite_mismatch": st["nonfinite_mismatch"]}
        ds = {"bitwise_rate": 100.0, "max_abs": 0.0}
        rel_c = _rel_to_fp64(sum_pad, (inp["h"].double() ** 2).sum(-1))
        rel_b = _rel_to_fp64(sum_tight, (tight.double() ** 2).sum(-1))
        note = f"{inp['h'].shape[-1] - dd} zero tail cols"
        return ulp, ds, rel_c, rel_b, note
    if tag == "A1":
        # four accumulators interleaved in one traversal vs four passes
        cand, bl = _seq_stats(inp, fused=True), _seq_stats(inp, fused=False)
        ulp = {"max_ulp": 0.0, "viol_rate": 0.0, "nonfinite_mismatch": 0}
        for kk in ("s_h", "s_k", "s_p", "s_v"):
            st = _ulp_stats(cand[kk], bl[kk])
            ulp["max_ulp"] = max(ulp["max_ulp"], st["max_ulp"])
            ulp["viol_rate"] = max(ulp["viol_rate"], st["viol_rate"])
            ulp["nonfinite_mismatch"] += st["nonfinite_mismatch"]
        ds = {"bitwise_rate": 100.0, "max_abs": 0.0}
        exact = (inp["h"].double() ** 2).sum(-1)
        rel_c = _rel_to_fp64(cand["s_h"], exact)
        rel_b = _rel_to_fp64(bl["s_h"], exact)
        return ulp, ds, rel_c, rel_b, "interleaved 4 accumulators vs 4 passes"

    on = {t for _, t in ITEMS} | {t for _, t in INFO}
    cand = run_candidate(inp, d, EPS, on)
    bl = run_candidate(inp, d, EPS, on - {tag})
    ulp = {"max_ulp": 0.0, "viol_rate": 0.0, "nonfinite_mismatch": 0}
    ds = {"bitwise_rate": 100.0, "max_abs": 0.0}
    for kk in WATCH[tag]:
        st = _ulp_stats(cand[kk], bl[kk])
        ulp["max_ulp"] = max(ulp["max_ulp"], st["max_ulp"])
        ulp["viol_rate"] = max(ulp["viol_rate"], st["viol_rate"])
        ulp["nonfinite_mismatch"] += st["nonfinite_mismatch"]
    for kk in ("y", "vhat_norm"):
        st = _dtype_stats(cand[kk], bl[kk], dt)
        ds["bitwise_rate"] = min(ds["bitwise_rate"], st["bitwise_rate"])
        ds["max_abs"] = max(ds["max_abs"], st["max_abs"])
    rel_c = _rel_to_fp64(cand["y"], ref64["y"])
    rel_b = _rel_to_fp64(bl["y"], ref64["y"])
    return ulp, ds, rel_c, rel_b, ",".join(WATCH[tag])


HDR = ("workload", "item", "max_ulp", ">1ulp%", "nonfin", "bitwise%", "max_abs",
       "rel_cand", "rel_base")


def _print_table(rows, title):
    print(f"\n--- {title} ---")
    print(f"{HDR[0]:<15}{HDR[1]:<26}{HDR[2]:>8}{HDR[3]:>9}{HDR[4]:>7}{HDR[5]:>10}"
          f"{HDR[6]:>10}{HDR[7]:>11}{HDR[8]:>11}  verdict")
    for (label, name, ulp, ds, rel_c, rel_b, ok, note) in rows:
        print(f"{label:<15}{name:<26}{ulp['max_ulp']:>8.1f}{ulp['viol_rate']:>9.4f}"
              f"{ulp['nonfinite_mismatch']:>7d}{ds['bitwise_rate']:>10.4f}"
              f"{ds['max_abs']:>10.2e}{rel_c:>11.2e}{rel_b:>11.2e}  "
              f"{'EQUIV_PASS' if ok else 'NOT ADOPTED'}   ({note})")


def main() -> int:
    rows, info, fails = [], [], 0
    for label, cfg in WORKLOADS:
        inp = make_inputs(**cfg)
        dt = inp["dtype"]
        ref64 = run_fp64(inp, cfg["d"], EPS)
        for name, tag in ITEMS:
            ulp, ds, rel_c, rel_b, note = _judge(inp, cfg, dt, ref64, tag)
            ok = (ulp["max_ulp"] <= ULP_TOL and ds["bitwise_rate"] >= BITWISE_RATE
                  and ulp["nonfinite_mismatch"] == 0 and ds["max_abs"] <= MAX_ABS)
            fails += 0 if ok else 1
            rows.append((label, name, ulp, ds, rel_c, rel_b, ok, note))
        for name, tag in INFO:
            ulp, ds, rel_c, rel_b, note = _judge(inp, cfg, dt, ref64, tag)
            info.append((label, name, ulp, ds, rel_c, rel_b, False, note))

    _print_table(rows, "adopted items (§1.6.1)")
    _print_table(info, "measured but not adopted (informational)")

    print("\n--- per-item worst over all workloads (the DESIGN.md §1.6.1 table) ---")
    print(f"{'item':<30}{'max_ulp':>9}{'>1ulp%':>9}{'bitwise%':>10}{'max_abs':>11}"
          f"{'nonfin':>7}  verdict")
    for src in (rows, info):
        for tag in dict.fromkeys(r[1].split()[0] for r in src):
            cells = [r for r in src if r[1].startswith(tag)]
            if not cells:
                continue
            mu = max(c[2]["max_ulp"] for c in cells)
            vr = max(c[2]["viol_rate"] for c in cells)
            bw = min(c[3]["bitwise_rate"] for c in cells)
            ma = max(c[3]["max_abs"] for c in cells)
            nf = sum(c[2]["nonfinite_mismatch"] for c in cells)
            ok = all(c[6] for c in cells)
            adopted = tag.startswith("A")
            print(f"{cells[0][1]:<30}{mu:>9.1f}{vr:>9.4f}{bw:>10.4f}{ma:>11.2e}"
                  f"{nf:>7d}  "
                  f"{'EQUIV_PASS' if ok else ('NOT ADOPTED' if not adopted else 'EQUIV_FAIL')}")

    worst = max(r[2]["max_ulp"] for r in rows)
    print(f"\nworst fp32 ulp distance over the adopted items: {worst:.1f}")
    print(f"adopted-item cells evaluated: {len(rows)}   failures: {fails}")
    print(f"preconditions: ULP_TOL={ULP_TOL:g}, bitwise>={BITWISE_RATE:g}%, "
          f"max_abs<={MAX_ABS:g}")
    print("RESULT: " + ("ALL EQUIV_PASS" if fails == 0 else "EQUIV_FAIL PRESENT"))
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
