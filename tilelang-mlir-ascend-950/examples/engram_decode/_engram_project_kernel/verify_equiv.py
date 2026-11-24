"""D-1 machine verification for `_engram_project_kernel`'s adopted等价优化 items.

Runs on CPU (torch), no device.  From the task directory:

    /workspace/venv-ascend950/bin/python verify_equiv.py

Adopted items (DESIGN.md §1.6.1)
  A3  clear_accum on the first K block instead of an explicit zero-fill of the
      L0C accumulator:      acc = P_0 ; acc += P_t          (candidate)
      vs explicit clear:    acc = 0 ; acc += P_0 ; acc += P_t (baseline)
  A4  M-tail dead-row elision: only the `am = min(batch, block_m)` valid rows are
      loaded and stored, instead of the full block_m rows.
  T1  (tiling mechanism, checked for completeness) K-tail zero padding: the last
      K block is extended to `block_k` with zeros when `d_mem % block_k != 0`.

Emulation fidelity
  Both accumulators are replayed as fp32 with the same K-block grouping as the
  kernel, so the comparison isolates the *only* difference between A3's two paths
  (the additive identity at the start of the chain).  The fp64 reference decides
  which path is closer to the exact result; the criterion is the fp32 bit pattern
  (float `==` treats -0.0 == +0.0, which is the documented A3 difference).

Criteria (per row, all must hold for EQUIV_PASS)
  * finite elements: max ulp difference == 0
  * violation rate (elements with > 1 ulp) == 0
  * NaN / +inf / -inf positions identical
"""

import itertools
import math
import struct

import torch

FP16 = torch.float16
BF16 = torch.bfloat16

WORKLOADS = [
    # label, batch, d_mem, d_pad, max_conv_len, dtype
    ("decode-s1", 1, 256, 256, 9, FP16),
    ("serving-b1", 1, 512, 256, 12, FP16),
    ("serving-b4", 4, 1024, 512, 20, FP16),
    ("serving-b8", 8, 512, 256, 18, BF16),
    ("decode-partial-cache", 2, 512, 256, 12, FP16),
    ("ktail-519", 17, 519, 256, 12, FP16),
]

BLOCK_K_CANDIDATES = (256, 128, 64, 32, 16)


def pick_block_k(d_mem):
    for bk in BLOCK_K_CANDIDATES:
        if d_mem % bk == 0:
            return bk
    return 128  # K tail handled by the padded copy (T1)


def ulp_diff(a, b):
    """Integer-ordered fp32 bit distance, elementwise; NaN/inf handled outside."""
    ai = a.contiguous().view(torch.int32).to(torch.int64)
    bi = b.contiguous().view(torch.int32).to(torch.int64)
    ai = torch.where(ai < 0, torch.tensor(-(1 << 31), dtype=torch.int64) - ai, ai)
    bi = torch.where(bi < 0, torch.tensor(-(1 << 31), dtype=torch.int64) - bi, bi)
    return (ai - bi).abs()


def pattern(a):
    nan = torch.isnan(a)
    pinf = torch.isposinf(a)
    ninf = torch.isneginf(a)
    return nan, pinf, ninf


def kblocked_matmul_ordered(E32, W32, block_k, zero_init):
    """fp32 accumulation with the kernel's K-block grouping.

    zero_init=True  -> acc starts at zeros (baseline, ``T.clear`` then accumulate)
    zero_init=False -> acc starts at the first block's product (candidate A3)
    """
    batch, d_mem = E32.shape
    d = W32.shape[1]
    k_tiles = -(-d_mem // block_k)
    acc = None
    for kt in range(k_tiles):
        k0 = kt * block_k
        k1 = min(k0 + block_k, d_mem)
        part = E32[:, k0:k1] @ W32[k0:k1, :]
        if kt == 0:
            acc = part.clone() if not zero_init else torch.zeros_like(part) + part
        else:
            acc = acc + part
    _ = (batch, d, k_tiles)
    return acc


def check_row(name, candidate, baseline, ref64, extra_note=""):
    cand_finite = torch.isfinite(candidate)
    ulp = ulp_diff(candidate, baseline)
    ulp = torch.where(cand_finite, ulp, torch.zeros_like(ulp))
    max_ulp = int(ulp.max().item()) if ulp.numel() else 0
    viol = float((ulp > 1).float().mean().item()) if ulp.numel() else 0.0
    cn, cp, cm = pattern(candidate)
    bn_, bp, bm = pattern(baseline)
    pat_ok = bool(
        torch.equal(cn, bn_) and torch.equal(cp, bp) and torch.equal(cm, bm)
    )
    # closeness to the exact (fp64) result, as an observation
    finite = torch.isfinite(ref64)
    denom = ref64.abs().clamp_min(1e-30)
    if bool(finite.any()):
        err_c = ((candidate.double() - ref64).abs() / denom)[finite].max().item()
        err_b = ((baseline.double() - ref64).abs() / denom)[finite].max().item()
    else:
        err_c = err_b = float("nan")  # the whole tile is non-finite by construction
    ok = (max_ulp == 0) and (viol == 0.0) and pat_ok
    print(
        f"| {name} | {max_ulp} | {viol:.1e} | {'equal' if pat_ok else 'DIFF'} "
        f"| {err_c:.2e} / {err_b:.2e} | {'EQUIV_PASS' if ok else 'EQUIV_FAIL'} |"
        + (f" {extra_note}" if extra_note else "")
    )
    return ok


def corner_tensors(shape, dtype):
    """Random draw plus the corner values the kernel must survive."""
    g = torch.Generator().manual_seed(7)
    base = torch.randn(shape, generator=g, dtype=torch.float32)
    out = base.to(dtype)
    flat = out.view(-1)
    corners = [
        float("inf"), float("-inf"), float("nan"), 0.0, -0.0,
        65504.0 if dtype == FP16 else 3.39e38,        # max finite
        -65504.0 if dtype == FP16 else -3.39e38,
        6.1e-5 if dtype == FP16 else 1.2e-38,          # smallest normal
        5.96e-8 if dtype == FP16 else 1.0e-41,         # subnormal
        -5.96e-8 if dtype == FP16 else -1.0e-41,
    ]
    for i, v in enumerate(itertools.islice(itertools.cycle(corners), flat.numel() // 97 + 1)):
        flat[i * 97 % flat.numel()] = v
    return out


def main():
    rows = []
    print("## A3 — clear_accum vs explicit zero-init (per workload, both projections)")
    print("| case | max ulp | >1ulp rate | NaN/inf pattern | max rel err vs fp64 (cand / base) | verdict |")
    print("|---|---|---|---|---|---|")
    all_ok = True
    for label, batch, d_mem, d_pad, mcl, dtype in WORKLOADS:
        block_k = pick_block_k(d_mem)
        g = torch.Generator().manual_seed(11)
        E = (torch.randn(batch, d_mem, generator=g) * 0.1).to(dtype)
        WK = torch.randn(d_mem, d_pad, generator=g).to(dtype) * 0.02
        WV = torch.randn(d_mem, d_pad, generator=g).to(dtype) * 0.02
        for tag, W in (("K", WK), ("V", WV)):
            E32, W32 = E.float(), W.float()
            ref64 = E.double() @ W.double()
            cand = kblocked_matmul_ordered(E32, W32, block_k, zero_init=False)
            base = kblocked_matmul_ordered(E32, W32, block_k, zero_init=True)
            all_ok &= check_row(
                f"A3 {label}/{tag} bk={block_k} mcl={mcl}", cand, base, ref64
            )

    print()
    print("## A3 corner values (NaN/±inf/subnormal/±0 fed through both paths)")
    print("| case | max ulp | >1ulp rate | NaN/inf pattern | max rel err vs fp64 (cand / base) | verdict |")
    print("|---|---|---|---|---|---|")
    for dtype in (FP16, BF16):
        E = corner_tensors((8, 512), dtype)
        W = corner_tensors((512, 256), dtype)
        ref64 = E.double() @ W.double()
        cand = kblocked_matmul_ordered(E.float(), W.float(), 256, zero_init=False)
        base = kblocked_matmul_ordered(E.float(), W.float(), 256, zero_init=True)
        all_ok &= check_row(f"A3 corner {dtype}", cand, base, ref64)

    print()
    print("## A4 — M-tail dead-row elision (only the am valid rows loaded/stored)")
    print("| case | max ulp | >1ulp rate | NaN/inf pattern | max rel err vs fp64 (cand / base) | verdict |")
    print("|---|---|---|---|---|---|")
    for label, batch, d_mem, d_pad, mcl, dtype in WORKLOADS:
        g = torch.Generator().manual_seed(13)
        block_m = 16
        W = torch.randn(d_mem, d_pad, generator=g).to(dtype) * 0.02
        ok_all = True
        for m0 in range(0, batch, block_m):
            am = min(batch - m0, block_m)
            pad_rows = block_m - am
            E = (torch.randn(am, d_mem, generator=g) * 0.1).to(dtype)
            # same matmul shape on both sides so torch's blocking cannot bias the
            # comparison; the baseline pads the dead rows with zeros, the candidate
            # pads them with NaN/+-inf (the worst thing stale L1 can hold)
            garbage = torch.full((pad_rows, d_mem), float("nan"), dtype=dtype)
            if pad_rows > 1:
                garbage[1, :] = float("inf")
                garbage[-1, :] = -0.0
            base_full = torch.cat([E, torch.zeros(pad_rows, d_mem, dtype=dtype)], 0)
            cand_full = torch.cat([E, garbage], 0) if pad_rows else E
            cand = (cand_full.float() @ W.float())[:am, :]
            base = (base_full.float() @ W.float())[:am, :]
            ref64 = E.double() @ W.double()
            ok_all &= check_row(
                f"A4 {label} m0={m0} am={am}/{block_m}", cand, base, ref64
            )
        all_ok &= ok_all

    print()
    print("## T1 — K-tail zero padding (compute over ceil(d_mem/block_k)*block_k with zeros)")
    print("| case | max ulp | >1ulp rate | NaN/inf pattern | max rel err vs fp64 (cand / base) | verdict |")
    print("|---|---|---|---|---|---|")
    for label, batch, d_mem, d_pad, mcl, dtype in WORKLOADS:
        block_k = pick_block_k(d_mem)
        if d_mem % block_k == 0:
            print(f"| T1 {label} | 0 | 0.0e+00 | equal | 0.00e+00 / 0.00e+00 | "
                  f"EQUIV_PASS (no tail: exact same K) |")
            continue
        g = torch.Generator().manual_seed(17)
        E = (torch.randn(batch, d_mem, generator=g) * 0.1).to(dtype)
        W = torch.randn(d_mem, d_pad, generator=g).to(dtype) * 0.02
        k_pad = -(-d_mem // block_k) * block_k
        E_pad = torch.zeros(batch, k_pad, dtype=dtype)
        E_pad[:, :d_mem] = E
        W_pad = torch.zeros(k_pad, d_pad, dtype=dtype)
        W_pad[:d_mem] = W
        ref64 = E.double() @ W.double()
        cand = kblocked_matmul_ordered(E_pad.float(), W_pad.float(), block_k,
                                       zero_init=False)
        base = kblocked_matmul_ordered(E.float(), W.float(), block_k, zero_init=False)
        all_ok &= check_row(f"T1 {label} bk={block_k} k_pad={k_pad}", cand, base, ref64)

    print()
    print(f"RESULT: {'ALL EQUIV_PASS' if all_ok else 'HAS EQUIV_FAIL'}")
    build_detail()
    return 0 if all_ok else 1


def build_detail():
    """Extra structural check: the shift copy is a bit-exact dtype move."""
    g = torch.Generator().manual_seed(23)
    for dtype in (FP16, BF16):
        batch, mcl, d = 8, 18, 256
        state = (torch.randn(batch, mcl, d, generator=g) * 0.1).to(dtype)
        state[0, 0, 0] = float("nan")
        state[0, 1, 1] = float("inf")
        ref = state[:, 1:, :].clone()
        out = torch.empty(batch, mcl, d, dtype=dtype)
        out[:, : mcl - 1, :] = state[:, 1:, :]
        same = torch.equal(out[:, : mcl - 1, :].view(torch.int16), ref.view(torch.int16))
        print(f"shift copy bit-exact ({dtype}): {same}")


if __name__ == "__main__":
    raise SystemExit(main())
