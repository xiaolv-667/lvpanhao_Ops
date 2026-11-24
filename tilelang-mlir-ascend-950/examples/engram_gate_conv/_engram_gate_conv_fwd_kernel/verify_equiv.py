"""Design-phase equivalence check (D-1) for the Engram GateConv forward design.

Compares each formula-level optimisation adopted in ``DESIGN.md`` section 1.6.1
against the unoptimised baseline of the *same* semantics (the torch reference of
``examples/TileOPs/tileops/workloads/sequence_modeling.py``), on random tensors
and on corner patterns, with an fp64 reference as truth.  torch CPU only.

Items (see DESIGN.md 1.6.1):
  A1  gate dot fused with the two RMSNorm normalisations:
          base: dot = sum( (h*rh*w) * (k*rk*w) )
          cand: dot = rh*rk*sum( h*k*w*w )
  A2  rms_w_v folded into the conv weights (once per launch):
          base: normed = vhat_r * rrms_v * w_v ; conv_out = sum_p conv_w * normed
          cand: g = vhat_r * rrms_v ; cwv = conv_w * w_v ; conv_out = sum_p cwv * g
  A3  one normalised-conv-input row per source row (rolling window reuse) instead
      of recomputing it once per tap:
          base: normed recomputed for every (output row, tap) pair
          cand: g stored once and reused by the four taps that read the row
  A4  sqrt(d) division folded into a scalar multiplier of the raw dot:
          base: alpha = sigmoid(dot / sqrt(d))
          cand: alpha = sigmoid(dot * (1/sqrt(d)))
  ALL the four items together, all six outputs, against the dtype tolerance the
      op's own test uses (tests/ops/test_engram.py: fp16 atol=rtol=0.1,
      bf16 atol=0.2/rtol=0.1).

Verdict per item: EQUIV_PASS when no element violates the op tolerance and the
fp64-relative error is at fp32-rounding level; EQUIV_FAIL otherwise.

Run:  /workspace/venv-ascend950/bin/python verify_equiv.py
"""

from __future__ import annotations

import itertools
import math

import torch
import torch.nn.functional as F

CONV_KERNEL_SIZE = 4
EPS = 1e-6
TOL = {
    torch.float16: (0.1, 0.1),
    torch.bfloat16: (0.2, 0.1),
}
# fp32 intermediates (alpha, rrms_*) keep the manifest fp32 standard.
PF32_ATOL, PF32_RTOL = 1e-4, 1e-4


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _ulp(x: torch.Tensor) -> torch.Tensor:
    """Spacing of the fp32 grid around ``x`` (abs, NaN -> 0)."""
    x = x.float()
    up = torch.nextafter(x, torch.full_like(x, math.inf))
    d = (up - x).abs()
    return torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)


def _spacing(x: torch.Tensor) -> torch.Tensor:
    """Spacing of the *evaluated* dtype grid (fp16/bf16/fp32) around ``x``."""
    up = torch.nextafter(x, torch.full_like(x, math.inf))
    d = (up - x).abs()
    return torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)


def _stats(got, ref, atol, rtol):
    """(max dtype-ulp of the evaluated tensor, max |diff|, max tolerance budget, nan mismatch).

    ``budget = |got-ref| / (atol + rtol*|ref|)`` — values <= 1 pass the op's own
    tolerance.  ulp is measured on the evaluated dtype grid (fp16/bf16 for the two
    tensor outputs, fp32 for the four row scalars), so a cancellation element with
    a tiny |ref| no longer produces a meaningless five-digit ulp count.
    """
    nan_mismatch = int(((got != got) != (ref != ref)).sum())
    both = torch.isfinite(got) & torch.isfinite(ref)
    diff = (got - ref).abs()
    ulp = _spacing(ref)
    ulp_diff = torch.where(both & (ulp > 0), diff / ulp.clamp_min(1e-45), torch.zeros_like(diff))
    ulp_diff = torch.nan_to_num(ulp_diff, nan=0.0, posinf=0.0, neginf=0.0)
    tol = atol + rtol * ref.abs()
    budget = torch.where(both, diff / tol, torch.zeros_like(diff))
    budget = torch.nan_to_num(budget, nan=0.0, posinf=0.0, neginf=0.0)
    return (
        float(ulp_diff.max()) if ulp_diff.numel() else 0.0,
        float(diff[both].max()) if bool(both.any()) else 0.0,
        float(budget.max()) if budget.numel() else 0.0,
        nan_mismatch,
    )


def _rmsnorm(x, w, eps=EPS):
    """fp32 RMSNorm: returns (normed, rrms). Mirrors the torch reference."""
    xf = x.float()
    rrms = (xf**2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    return xf * rrms * w.float(), rrms.squeeze(-1)


def _conv_out(normed, cw):
    """Causal width-4 depthwise conv, fp32, identical to the torch reference."""
    M, T, d = normed.shape
    v_perm = normed.permute(0, 2, 1)
    padded = F.pad(v_perm, (CONV_KERNEL_SIZE - 1, 0))
    cw_expanded = cw.float().T.unsqueeze(1)
    return F.conv1d(padded, cw_expanded, groups=d).permute(0, 2, 1)


# ---------------------------------------------------------------------------
# baseline (unoptimised) and candidates
# ---------------------------------------------------------------------------
def baseline(H, k, v, rms_w_h, rms_w_v, conv_w, part="all"):
    """fp32 arithmetic of the source semantics; ``part`` isolates one item."""
    h_norm, rrms_h = _rmsnorm(H, rms_w_h)
    k_norm, rrms_k = _rmsnorm(k, rms_w_h)
    dot = (h_norm * k_norm).sum(dim=-1, keepdim=True)
    # baseline keeps the vector division by sqrt(d)
    alpha = torch.sigmoid(dot / (H.shape[-1] ** 0.5))
    v_hat = alpha * v.float()
    v_hat_dt = v_hat.to(H.dtype)
    v_hat_norm, rrms_v = _rmsnorm(v_hat_dt, rms_w_v)
    out = _conv_out(v_hat_norm, conv_w)
    y = F.silu(out) + v_hat
    return {
        "Y": y.to(H.dtype),
        "vhat": v_hat_dt,
        "alpha": alpha.squeeze(-1).float(),
        "rrms_h": rrms_h.float(),
        "rrms_k": rrms_k.float(),
        "rrms_v": rrms_v.float(),
        "dot": dot,
        "conv_out": out,
    }


def candidate(H, k, v, rms_w_h, rms_w_v, conv_w, part="all"):
    """The four adopted items; ``part`` enables only one of them."""
    d = H.shape[-1]
    use_a1 = part in ("all", "A1")
    use_a2 = part in ("all", "A2")
    use_a3 = part in ("all", "A3")
    use_a4 = part in ("all", "A4")

    hf, kf, wf = H.float(), k.float(), rms_w_h.float()
    if use_a1:
        # A1: normalisation folded into the dot.
        sum_h = (hf**2).mean(dim=-1, keepdim=True).add(EPS)
        sum_k = (kf**2).mean(dim=-1, keepdim=True).add(EPS)
        rh = sum_h.rsqrt()
        rk = sum_k.rsqrt()
        dot_raw = (hf * kf * (wf**2)).sum(dim=-1, keepdim=True)
        dot = dot_raw * rh * rk
        rrms_h = rh.squeeze(-1).float()
        rrms_k = rk.squeeze(-1).float()
    else:
        h_norm, rrms_h = _rmsnorm(H, rms_w_h)
        k_norm, rrms_k = _rmsnorm(k, rms_w_h)
        dot = (h_norm * k_norm).sum(dim=-1, keepdim=True)
        dot = dot.squeeze(-1).unsqueeze(-1)

    scale = 1.0 / (d**0.5) if use_a4 else None
    if use_a4:
        alpha = torch.sigmoid(dot * scale)
    else:
        alpha = torch.sigmoid(dot / (d**0.5))

    v_hat = alpha * v.float()
    v_hat_dt = v_hat.to(H.dtype)

    # rrms_v always from the *rounded* vhat (golden semantics), fp32 statistics.
    rrms_v = (v_hat_dt.float() ** 2).mean(dim=-1, keepdim=True).add(EPS).rsqrt()

    wv = rms_w_v.float()
    cw = conv_w.float()
    if use_a2:
        # A2: rms_w_v folded into the depthwise conv weights (once per launch)
        cwv = cw * wv.unsqueeze(0)
        _conv_in = v_hat_dt.float() * rrms_v  # (M, T, d), fp32, folded form
    else:
        cwv = cw
        _conv_in = v_hat_dt.float() * rrms_v * wv

    M, T, dd = _conv_in.shape
    conv = torch.zeros((M, T, dd), dtype=torch.float32)
    for p in range(CONV_KERNEL_SIZE):
        idx = torch.arange(T) - (CONV_KERNEL_SIZE - 1) + p
        valid = idx >= 0
        if use_a3:
            # A3: one stored conv-input row, reused by the four taps
            gathered = _conv_in[:, idx.clamp_min(0), :] * valid.view(1, T, 1)
        else:
            # baseline: the normalised row is recomputed for every (row, tap) pair
            g = v_hat_dt.float()[:, idx.clamp_min(0), :] * valid.view(1, T, 1)
            rv = rrms_v[:, idx.clamp_min(0), :] * valid.view(1, T, 1)
            gathered = g * rv * wv if not use_a2 else g * rv
        conv += cwv[p].view(1, 1, dd) * gathered
    y = F.silu(conv) + v_hat

    return {
        "Y": y.to(H.dtype),
        "vhat": v_hat_dt,
        "alpha": alpha.squeeze(-1).float(),
        "rrms_h": rrms_h.float().squeeze(-1),
        "rrms_k": rrms_k.float().squeeze(-1),
        "rrms_v": rrms_v.squeeze(-1).float(),
        "dot": dot,
        "conv_out": conv,
    }


def reference64(H, k, v, rms_w_h, rms_w_v, conv_w, dtype):
    """fp64 truth of the operator semantics, then rounded to the output dtype."""
    h64, k64, v64 = H.double(), k.double(), v.double()
    w64, wv64, cw64 = rms_w_h.double(), rms_w_v.double(), conv_w.double()
    d = h64.shape[-1]

    def rms(x, w):
        rr = (x**2).mean(dim=-1, keepdim=True).add(EPS).rsqrt()
        return x * rr * w, rr.squeeze(-1)

    h_norm, rh = rms(h64, w64)
    k_norm, rk = rms(k64, w64)
    dot = (h_norm * k_norm).sum(-1, keepdim=True)
    alpha = torch.sigmoid(dot / d**0.5)
    v_hat = alpha * v64
    # the reference defines the conv chain on the dtype-rounded vhat
    v_hat_dt = v_hat.to(dtype).double()
    v_hat_norm, rv = rms(v_hat_dt, wv64)
    M, T, dd = v_hat_norm.shape
    perm = v_hat_norm.permute(0, 2, 1)
    padded = F.pad(perm, (CONV_KERNEL_SIZE - 1, 0))
    conv = F.conv1d(padded, cw64.T.unsqueeze(1), groups=dd).permute(0, 2, 1)
    y = F.silu(conv) + v_hat
    return {
        "Y": y.to(dtype),
        "vhat": v_hat.to(dtype),
        "alpha": alpha.squeeze(-1),
        "rrms_h": rh,
        "rrms_k": rk,
        "rrms_v": rv,
        "conv_out": conv,
    }


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------
def make_case(M, T, d, dtype, pattern="random", seed=0):
    gen = torch.Generator().manual_seed(seed)
    H = torch.randn(M, T, d, generator=gen, dtype=torch.float32)
    k = torch.randn(M, T, d, generator=gen) * 0.1
    v = torch.randn(M, T, d, generator=gen) * 0.1
    rms_w_h = torch.ones(d)
    rms_w_v = torch.ones(d)
    conv_w = torch.randn(CONV_KERNEL_SIZE, d, generator=gen) * 0.02
    if pattern == "zeros":
        H.zero_()
        k.zero_()
        v.zero_()
        conv_w.zero_()
    elif pattern == "large":
        H.mul_(60.0)
        k.mul_(8.0)
        v.mul_(8.0)
    elif pattern == "tiny":
        H.mul_(1e-4)
        k.mul_(1e-4)
        v.mul_(1e-4)
    elif pattern == "ones_w":
        rms_w_h.fill_(1.0)
        rms_w_v.fill_(1.0)
    elif pattern == "nonfinite":
        H[0, 0, 0] = float("inf")
        k[0, 0, 0] = 0.0
        H[0, min(1, T - 1), 1] = float("nan")
        v[0, min(1, T - 1), 1] = float("nan")
    H, k = H.to(dtype), k.to(dtype)
    v = v.to(dtype)
    rms_w_h, rms_w_v = rms_w_h.to(dtype), rms_w_v.to(dtype)
    conv_w = conv_w.to(dtype)
    return H, k, v, rms_w_h, rms_w_v, conv_w


GEOMETRIES = [
    (1, 32, 256),   # manifest train-s32
    (2, 64, 512),   # manifest train-s64
    (1, 128, 256),  # manifest train-s128
    (3, 96, 768),   # manifest train-s96
    (1, 16, 192),   # d not a multiple of 128 -> tail-lane path
    (1, 8, 128),    # minimal geometry
]
PATTERNS = ["random", "zeros", "large", "tiny", "nonfinite", "ones_w"]
DTYPES = [torch.float16, torch.bfloat16]


def run_item(item, dt, only_shape=None):
    worst_ulp = 0.0
    worst_budget = 0.0
    worst_abs = 0.0
    nf_total = 0
    n_cases = 0
    worst_rel = 0.0
    fields = ["Y", "vhat", "alpha", "rrms_h", "rrms_k", "rrms_v"] if item == "ALL" else None
    item_fields = {
        "A1": ["alpha"],
        "A2": ["Y"],
        "A3": ["Y"],
        "A4": ["alpha"],
    }
    for (M, T, d), pat in itertools.product(GEOMETRIES, PATTERNS):
        if only_shape is not None and only_shape != (M, T, d):
            continue
        args = make_case(M, T, d, dt, pat)
        base = baseline(*args)
        cand = candidate(*args, part=item)
        ref = reference64(*args, dtype=dt)
        keys = fields or item_fields[item]
        for key in keys:
            atol, rtol = TOL[dt] if key in ("Y", "vhat") else (PF32_ATOL, PF32_RTOL)
            u, mx, budget, nfm = _stats(cand[key], base[key], atol, rtol)
            u2, mx2, budget2, _ = _stats(base[key], ref[key], atol, rtol)
            worst_ulp = max(worst_ulp, u)
            worst_abs = max(worst_abs, mx)
            worst_budget = max(worst_budget, budget)
            nf_total += nfm
            n_cases += 1
            # fp64-relative error of the candidate, scale-referenced (max |row|)
            c, r = cand[key].double(), ref[key].double()
            fin = torch.isfinite(c) & torch.isfinite(r)
            if fin.any():
                scale = r[fin].abs().max().clamp_min(1e-12)
                worst_rel = max(worst_rel, float(((c[fin] - r[fin]).abs().max() / scale)))
    return {
        "item": item,
        "dtype": str(dt).replace("torch.", ""),
        "fields": n_cases,
        "max_ulp_vs_base": worst_ulp,
        "max_abs_vs_base": worst_abs,
        "max_rel_vs_fp64": worst_rel,
        "max_budget": worst_budget,
        "nonfinite_mismatch": nf_total,
    }


def main():
    rows = []
    for dt in DTYPES:
        for item in ("A1", "A2", "A3", "A4", "ALL"):
            rows.append(run_item(item, dt))
    hdr = (
        "| 采纳项 | dtype | 对照字段数 | 最大 ulp(cand, base) | 最大 abs(cand, base) "
        "| 尺度相对误差(vs fp64) | 最大容差预算占用 | 非有限不一致 | 结论 |"
    )
    print(hdr)
    print("|---|---|---|---|---|---|---|---|---|")
    verdicts = {}
    for r in rows:
        ok = r["max_budget"] <= 1.0 and r["nonfinite_mismatch"] == 0
        verdict = "EQUIV_PASS" if ok else "EQUIV_FAIL"
        verdicts[(r["item"], r["dtype"])] = verdict
        print(
            f"| {r['item']} | {r['dtype']} | {r['fields']} | {r['max_ulp_vs_base']:.0f} "
            f"| {r['max_abs_vs_base']:.3e} | {r['max_rel_vs_fp64']:.3e} "
            f"| {r['max_budget']:.4f} | {r['nonfinite_mismatch']} | **{verdict}** |"
        )
    print("\nper-geometry detail (train-s96, bf16, random):")
    args = make_case(3, 96, 768, torch.bfloat16, "random")
    base = baseline(*args)
    cand = candidate(*args, part="all")
    ref = reference64(*args, dtype=torch.bfloat16)
    for key in ("Y", "vhat", "alpha", "rrms_h", "rrms_k", "rrms_v"):
        atol, rtol = TOL[torch.bfloat16] if key in ("Y", "vhat") else (PF32_ATOL, PF32_RTOL)
        u, mx, budget, _ = _stats(cand[key], base[key], atol, rtol)
        u2, mx2, budget2, _ = _stats(base[key], ref[key], atol, rtol)
        print(
            f"  {key:8s} ulp(cand,base)={u:6.0f} max|d|={mx:.3e} budget={budget:.4f} "
            f"| baseline ulp(base,fp64)={u2:6.0f} max|d|={mx2:.3e} budget={budget2:.4f}"
        )
    bad = [k for k, v in verdicts.items() if v == "EQUIV_FAIL"]
    print("\nFAILED:", bad if bad else "none")


if __name__ == "__main__":
    main()
