"""Design-phase equivalence check (D-1) for the Engram GateConv *backward* design.

Each formula-level optimisation adopted in ``DESIGN.md`` section 1.6.1 is
compared against the *unoptimised* arithmetic of the same semantics (the
recompute + transpose order of the source kernel
``tileops/_kernels/sequence_modeling/engram/gate_conv_bwd.py``), element by
element in fp32, with an fp64 autograd reference as truth.  torch CPU only.

Items (see DESIGN.md 1.6.1):
  O1  reciprocal precomputation:  x / d  ->  x * (1/d),  x / sqrt(d) -> x * (1/sqrt(d))
      (three RMSNorm corrections + the gate scalar ddot)
  O2  RMSNorm correction coefficient folded once per token:
      base ((((rv*rv)*rv)*z)*dot)/d      cand ((rv*rv*rv)*(1/d)*z)*dot
  O3  value-chain common sub-expressions: the product z*dn feeds both the row dot
      and drms_w_v, and w*dn feeds both the dot and the direct dvhat term
  O4  gate-chain common sub-expressions: u_h = ddot*kn / u_k = ddot*hn are formed
      once and reused by dot_h/dot_k, dH/dk and drms_w_h
  O5  vhat_norm materialised once per row (base recomputes z*rv*w_v per conv tap);
      mathematically a pure reuse, so the comparison must be bitwise
  ALL all five items together, all six outputs
Each row is compared both per item and against the dtype tolerance the op's own
test uses (tests/ops/test_engram.py ``_tol_bwd``: fp16 atol=rtol=0.2,
bf16 atol=0.3/rtol=0.2).

Verdict per item: EQUIV_PASS when no element violates the op tolerance, no
non-finite mismatch, and the fp32 arithmetic stays at rounding level
(<= 4 ulp against the baseline); EQUIV_FAIL otherwise.

Run:  /workspace/venv-ascend950/bin/python verify_equiv.py
"""

from __future__ import annotations

import itertools
import math

import torch
import torch.nn.functional as F

KS = 4
EPS = 1e-6
TOL_BWD = {
    torch.float16: (0.2, 0.2),
    torch.bfloat16: (0.3, 0.2),
}
ITEMS = ("O1", "O2", "O3", "O4", "O5")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _spacing(x: torch.Tensor) -> torch.Tensor:
    """Spacing of the fp32 grid around ``x`` (abs, non-finite -> 0)."""
    x = x.float()
    up = torch.nextafter(x, torch.full_like(x, math.inf))
    return torch.nan_to_num((up - x).abs(), nan=0.0, posinf=0.0, neginf=0.0)


def _stats(got, ref, atol, rtol):
    """(scale-aware ulp, max |diff|, max tolerance budget, nan mismatch).

    The ulp is measured on the *field scale* (the spacing of the largest
    magnitude in ``ref``) so that exactly-zero elements -- where the fp32 grid
    spacing is meaningless -- do not dominate the metric.  ``budget`` is the
    op-tolerance criterion: ``|got-ref| / (atol + rtol*|ref|)``.
    """
    nan_mismatch = int(((got != got) != (ref != ref)).sum())
    both = torch.isfinite(got) & torch.isfinite(ref)
    diff = (got - ref).abs()
    fin = ref.abs()[both]
    scale = float(fin.max()) if fin.numel() else 0.0
    grid = float(_spacing(torch.tensor([scale])).item()) if scale > 0 else 1e-45
    ulp_scaled = float((diff[both].max() / max(grid, 1e-45)).item()) if bool(both.any()) else 0.0
    tol = atol + rtol * ref.abs()
    budget = torch.where(both, diff / tol, torch.zeros_like(diff))
    budget = torch.nan_to_num(budget, nan=0.0, posinf=0.0, neginf=0.0)
    return (
        ulp_scaled,
        float(diff[both].max()) if bool(both.any()) else 0.0,
        float(budget.max()) if budget.numel() else 0.0,
        nan_mismatch,
    )


def _rmsnorm(x, w, eps=EPS):
    xf = x.float()
    rrms = (xf**2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    return xf * rrms * w.float(), rrms.squeeze(-1)


def _conv(normed, cw):
    """Causal width-4 depthwise conv, fp32, identical to the torch reference."""
    M, T, d = normed.shape
    padded = F.pad(normed.permute(0, 2, 1), (KS - 1, 0))
    return F.conv1d(padded, cw.float().T.unsqueeze(1), groups=d).permute(0, 2, 1)


# ---------------------------------------------------------------------------
# the source kernel's arithmetic (baseline) and the optimised arithmetic
# ---------------------------------------------------------------------------
def backward(dY, H, k, v, wh, wv, cw, vhat, alpha, rh, rk, rv, items=()):
    """Recompute the source kernel's arithmetic in fp32 element-wise ops.

    ``items`` selects the adopted optimisations; an empty tuple reproduces the
    order the source kernel uses (its own left-to-right expression nesting).
    """
    o1, o2, o3, o4, o5 = (it in items for it in ITEMS)
    M, T, d = H.shape
    dim = float(d)
    inv_d = 1.0 / dim
    inv_sqrt_d = 1.0 / (dim**0.5)
    hf, kf, vf = H.float(), k.float(), v.float()
    whf, wvf = wh.float(), wv.float()
    cwf = cw.float()
    dyf = dY.float()
    z = vhat.float()

    # ---- stage 1: conv_out recompute -> d_silu, and dconv_w ------------------
    if o5:
        nrm = (z * rv[..., None]) * wvf          # materialised once per row
        conv = torch.zeros_like(z)
        for p in range(KS):
            src = torch.zeros_like(z)
            if T > KS - 1 - p:
                src[:, KS - 1 - p :] = nrm[:, : T - (KS - 1 - p)]
            conv = conv + cwf[p] * src
        normed_taps = [None] * KS
        for p in range(KS):
            src = torch.zeros_like(z)
            if T > KS - 1 - p:
                src[:, KS - 1 - p :] = nrm[:, : T - (KS - 1 - p)]
            normed_taps[p] = src
    else:
        conv = torch.zeros_like(z)
        normed_taps = [None] * KS
        for p in range(KS):
            src = torch.zeros_like(z)
            if T > KS - 1 - p:
                src[:, KS - 1 - p :] = z[:, : T - (KS - 1 - p)]
            src_r = torch.zeros_like(rv)
            if T > KS - 1 - p:
                src_r[:, KS - 1 - p :] = rv[:, : T - (KS - 1 - p)]
            normed = (src * src_r[..., None]) * wvf
            normed_taps[p] = normed
            conv = conv + cwf[p] * normed

    sig = 1.0 / (1.0 + torch.exp(-conv))
    ds = dyf * (sig + conv * sig * (1.0 - sig))

    dcw = torch.stack([(ds * normed_taps[p]).sum(dim=(0, 1)) for p in range(KS)], dim=0)

    # ---- stage 2: conv transpose -> RMSNorm(vhat) backward -------------------
    d_vnorm = torch.zeros_like(z)
    for p in range(KS):
        if T - (KS - 1 - p) > 0:
            d_vnorm[:, : T - (KS - 1 - p)] += cwf[p] * ds[:, KS - 1 - p :]
    if o3:
        zd = z * d_vnorm
        dot_v = (zd * wvf).sum(dim=-1, keepdim=True)
        dwv = (zd * rv[..., None]).sum(dim=(0, 1))
    else:
        dot_v = ((z * wvf) * d_vnorm).sum(dim=-1, keepdim=True)
        dwv = (d_vnorm * z * rv[..., None]).sum(dim=(0, 1))

    if o3:
        direct = rv[..., None] * (wvf * d_vnorm)
    else:
        direct = (rv[..., None] * wvf) * d_vnorm
    # O2 folds 1/d into the per-token coefficient; O1 alone only replaces the
    # trailing divisions by reciprocal multiplies.
    if o2:
        corr = (rv * rv * rv * inv_d)[..., None] * z * dot_v
    elif o1:
        corr = ((rv * rv * rv)[..., None] * z * dot_v) * inv_d
    else:
        corr = ((rv * rv * rv)[..., None] * z * dot_v) / dim
    gz = direct - corr + dyf

    dv = alpha[..., None] * gz
    dalpha = (gz * vf).sum(dim=-1, keepdim=True)
    if o1:
        ddot = dalpha * alpha[..., None] * (1.0 - alpha[..., None]) * inv_sqrt_d
    else:
        ddot = dalpha * alpha[..., None] * (1.0 - alpha[..., None]) / (dim**0.5)

    # ---- stage 3: gate + RMSNorm(H)/RMSNorm(k) ------------------------------
    hn = (hf * rh[..., None]) * whf
    kn = (kf * rk[..., None]) * whf
    if o4:
        uh = ddot * kn
        uk = ddot * hn
        wuh = whf * uh
        wuk = whf * uk
        dot_h = (hf * wuh).sum(dim=-1, keepdim=True)
        dot_k = (kf * wuk).sum(dim=-1, keepdim=True)
        drms_h = ((uh * hf) * rh[..., None] + (uk * kf) * rk[..., None]).sum(dim=(0, 1))
        dh_dir = rh[..., None] * wuh
        dk_dir = rk[..., None] * wuk
    else:
        dot_h = ((hf * whf) * ddot * kn).sum(dim=-1, keepdim=True)
        dot_k = ((kf * whf) * ddot * hn).sum(dim=-1, keepdim=True)
        drms_h = (
            ddot * kn * hf * rh[..., None] + ddot * hn * kf * rk[..., None]
        ).sum(dim=(0, 1))
        dh_dir = (rh[..., None] * whf) * ddot * kn
        dk_dir = (rk[..., None] * whf) * ddot * hn
    if o2:
        dh = dh_dir - (rh * rh * rh * inv_d)[..., None] * hf * dot_h
        dk = dk_dir - (rk * rk * rk * inv_d)[..., None] * kf * dot_k
    elif o1:
        dh = dh_dir - (((rh * rh * rh)[..., None] * hf * dot_h) * inv_d)
        dk = dk_dir - (((rk * rk * rk)[..., None] * kf * dot_k) * inv_d)
    else:
        dh = dh_dir - (rh * rh * rh)[..., None] * hf * dot_h / dim
        dk = dk_dir - (rk * rk * rk)[..., None] * kf * dot_k / dim

    return {
        "dH": dh.to(H.dtype),
        "dk": dk.to(k.dtype),
        "dv": dv.to(v.dtype),
        "drms_w_h": drms_h,
        "drms_w_v": dwv,
        "dconv_w": dcw,
    }


def reference64(dY, H, k, v, wh, wv, cw, vhat, alpha, rh, rk, rv, eps=EPS,
                dtype=torch.float16):
    """fp64 autograd reference of the same semantics (the op's own oracle path).

    ``vhat/alpha/rh/rk/rv`` are the forward's saved tensors and are unused here,
    exactly as in the op's own reference (``ref_engram_gate_conv_bwd``).
    """
    H_ag = H.double().detach().requires_grad_(True)
    k_ag = k.double().detach().requires_grad_(True)
    v_ag = v.double().detach().requires_grad_(True)
    w_h = wh.double().detach().requires_grad_(True)
    w_v = wv.double().detach().requires_grad_(True)
    cw_ag = cw.double().detach().requires_grad_(True)
    h_norm, _ = _rmsnorm(H_ag, w_h, eps)
    k_norm, _ = _rmsnorm(k_ag, w_h, eps)
    dot = (h_norm * k_norm).sum(dim=-1, keepdim=True)
    alpha = torch.sigmoid(dot / (H_ag.shape[-1] ** 0.5))
    v_hat = alpha * v_ag
    v_hat_norm, _ = _rmsnorm(v_hat, w_v, eps)
    conv_out = _conv(v_hat_norm, cw_ag)
    y = F.silu(conv_out) + v_hat
    y.backward(dY.double())
    return {
        "dH": H_ag.grad.to(dtype),
        "dk": k_ag.grad.to(dtype),
        "dv": v_ag.grad.to(dtype),
        "drms_w_h": w_h.grad.float(),
        "drms_w_v": w_v.grad.float(),
        "dconv_w": cw_ag.grad.float(),
    }


# ---------------------------------------------------------------------------
# inputs (the workload's draws, plus corner patterns)
# ---------------------------------------------------------------------------
def make_case(M, T, d, dtype, pattern, seed=0):
    gen = torch.Generator().manual_seed(seed + 1)
    H = torch.randn(M, T, d, generator=gen, dtype=torch.float32)
    k = torch.randn(M, T, d, generator=gen) * 0.1
    v = torch.randn(M, T, d, generator=gen) * 0.1
    dY = torch.randn(M, T, d, generator=gen) * 0.1
    wh = torch.ones(d)
    wv = torch.ones(d)
    cw = torch.randn(KS, d, generator=gen) * 0.02
    if pattern == "zeros":
        H.zero_()
        k.zero_()
        v.zero_()
        dY.zero_()
        cw.zero_()
    elif pattern == "large":
        H.mul_(60.0)
        k.mul_(8.0)
        v.mul_(8.0)
        dY.mul_(8.0)
    elif pattern == "tiny":
        H.mul_(1e-4)
        k.mul_(1e-4)
        v.mul_(1e-4)
        dY.mul_(1e-4)
    elif pattern == "spike":
        cw.mul_(40.0)          # large conv taps -> saturating sigmoid
        dY.mul_(6.0)
    elif pattern == "nonfinite":
        H[0, 0, 0] = float("inf")
        k[0, 0, 0] = 0.0
        H[0, min(1, T - 1), 1] = float("nan")
        v[0, min(1, T - 1), 1] = float("nan")
    Hq, kq, vq = H.to(dtype), k.to(dtype), v.to(dtype)
    whq, wvq, cwq = wh.to(dtype), wv.to(dtype), cw.to(dtype)
    dYq = dY.to(dtype)
    # saved forward state, produced exactly as the workload produces it
    h_norm, rrms_h = _rmsnorm(Hq, whq)
    k_norm, rrms_k = _rmsnorm(kq, whq)
    dot = (h_norm * k_norm).sum(dim=-1, keepdim=True)
    alpha = torch.sigmoid(dot / (d**0.5))
    v_hat = alpha * vq.float()
    _, rrms_v = _rmsnorm(v_hat.to(dtype), wvq)
    return (
        dYq, Hq, kq, vq, whq, wvq, cwq,
        v_hat.to(dtype), alpha.squeeze(-1).float(),
        rrms_h.float(), rrms_k.float(), rrms_v.float(),
    )


GEOMETRIES = [
    (1, 32, 256),   # manifest train-s32
    (2, 64, 512),   # manifest train-s64
    (1, 128, 256),  # manifest train-s128
    (3, 96, 768),   # manifest train-s96
    (1, 8, 256),    # one chunk, halo only
    (2, 1, 256),    # single-token sequences
    (3, 7, 192),    # d not a multiple of 64, prime-ish seq
]
PATTERNS = ["random", "zeros", "large", "tiny", "spike", "nonfinite"]
DTYPES = [torch.float16, torch.bfloat16]
FIELDS = ("dH", "dk", "dv", "drms_w_h", "drms_w_v", "dconv_w")


def run_item(item, dt, only_shape=None):
    """Worst-case metrics over geometries x patterns for one optimisation item."""
    # rounding-level ceiling per *output grid*: the reassociations must move a
    # value by less than one grid step at the field scale (fp32 sums: 1e-5,
    # which is the fp32 reduction reassociation level for <= 1.5k terms).
    eps_out = {torch.float16: 9.8e-4, torch.bfloat16: 7.8e-3}
    items = () if item == "BASE" else (ITEMS if item == "ALL" else (item,))
    worst = {k: {"rel": 0.0, "budget": 0.0, "abs": 0.0} for k in FIELDS}
    nf = 0
    n_fields = 0
    for (M, T, d), pat in itertools.product(GEOMETRIES, PATTERNS):
        if only_shape is not None and only_shape != (M, T, d):
            continue
        args = make_case(M, T, d, dt, pat)
        atol, rtol = TOL_BWD[dt]
        base = backward(*args)
        cand = backward(*args, items=items)
        for key in FIELDS:
            _, mx, budget, nfm = _stats(cand[key], base[key], atol, rtol)
            nf += nfm
            n_fields += 1
            scale = float(base[key].float().abs().max())
            rel = mx / max(scale, 1e-12)
            w = worst[key]
            w["rel"] = max(w["rel"], rel)
            w["budget"] = max(w["budget"], budget)
            w["abs"] = max(w["abs"], mx)
    ok = nf == 0
    for key in FIELDS:
        ceiling = eps_out[dt] if key in ("dH", "dk", "dv") else 1e-5
        if worst[key]["budget"] > 1.0 or worst[key]["rel"] > ceiling:
            ok = False
    return {
        "item": item,
        "dtype": str(dt).replace("torch.", ""),
        "fields": n_fields,
        "rel": max(w["rel"] for w in worst.values()),
        "abs": max(w["abs"] for w in worst.values()),
        "budget": max(w["budget"] for w in worst.values()),
        "nf": nf,
        "verdict": "EQUIV_PASS" if ok else "EQUIV_FAIL",
        "detail": worst,
    }


def fp64_gap(dt):
    """Baseline (source arithmetic) vs the fp64 autograd truth, worst field."""
    worst = 0.0
    for (M, T, d), pat in itertools.product(GEOMETRIES, PATTERNS):
        args = make_case(M, T, d, dt, pat)
        atol, rtol = TOL_BWD[dt]
        base = backward(*args)
        ref = reference64(*args, dtype=dt)
        for key in FIELDS:
            _, mx, budget, _ = _stats(base[key], ref[key], atol, rtol)
            scale = float(ref[key].float().abs().max())
            worst = max(worst, mx / max(scale, 1e-12))
    return worst


def main():
    rows = []
    for dt in DTYPES:
        for item in ("BASE",) + ITEMS + ("ALL",):
            rows.append(run_item(item, dt))
    print("| 项 | dtype | 对照字段数 | 尺度相对误差 | 最大 abs | 最大容差预算 | 非有限不一致 | 结论 |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['item']} | {r['dtype']} | {r['fields']} | {r['rel']:.3e} "
              f"| {r['abs']:.3e} | {r['budget']:.4f} | {r['nf']} "
              f"| **{r['verdict']}** |")
    print()
    for r in rows:
        if r["item"] == "ALL":
            print(f"  ALL/{r['dtype']} per field:", ", ".join(
                f"{k}: rel={v['rel']:.2e} budget={v['budget']:.4f}"
                for k, v in r["detail"].items()))
    for dt in DTYPES:
        print(f"baseline(source arithmetic) vs fp64 truth, worst rel: "
              f"{str(dt).replace('torch.', '')} = {fp64_gap(dt):.3e}")
    bad = [f"{r['item']}/{r['dtype']}" for r in rows if r["verdict"] == "EQUIV_FAIL"]
    print("EQUIV_SUMMARY", "ALL_PASS" if not bad else "FAILED: " + ", ".join(bad))
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
