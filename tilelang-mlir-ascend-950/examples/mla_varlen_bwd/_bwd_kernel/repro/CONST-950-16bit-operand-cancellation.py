# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""CONST-950-16bit-operand-cancellation: the 16-bit GEMM-operand rounding of a
flash-attention backward kernel is *not* bounded by the output dtype's
tolerance on strongly cancelling cells.

What it shows (measured, 950PR / CANN 9.2.1 / TileLang ascend 0.1.15):

The backward pass casts ``P`` and ``dS`` to the *input* dtype before the three
output GEMMs (``dV = P^T dO``, ``dK = scale dS^T q``, ``dQ = dS k``), because the
GEMM operands live in 16-bit L1 -- the source GPU kernel does the same.  For a
cell whose true value is a small difference of many O(1) terms, that rounding is
amplified by ``sum|terms| / |value|`` and can exceed ``atol`` even though the
kernel is correct.  Diagnosing it needs a *second* reference that repeats the
same rounding, not a looser tolerance:

    ideal = fp64 reference with fp64 probabilities
    truth = reference that casts P and dS to the input dtype (source semantics)
    kernel == truth (to the output ulp) while kernel != ideal (beyond atol)

The arithmetic is dtype-agnostic; bfloat16 (8 mantissa bits) is what trips it in
practice -- fp16 (11 bits) stays inside 5e-3 on the same geometry.  The same
effect is why a bf16 operator can report a handful of cells outside a 5e-3 atol
over 10^8 compared cells and still be exact.

Usage (device 0, from the task directory, environment loaded):
    /workspace/venv-ascend950/bin/python repro/CONST-950-16bit-operand-cancellation.py
Exit code 0 means every assertion below held.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

import _bwd_kernel as impl  # noqa: E402

BF16 = "bfloat16"
RTOL = impl._RTOL[BF16]
DQ_ATOL = impl._ATOL / 192**-0.5


def cells_outside(got, ref, atol, rtol):
    a, e = got.float(), ref.float()
    gap = torch.nan_to_num((a - e).abs(), nan=0.0, posinf=0.0, neginf=0.0)
    bad = int((gap > atol + rtol * e.abs()).sum())
    return bad, float(gap.max())


def main():
    # Request 0 is empty (len_q = 0) so request 1 starts at q row 0; request 1 is
    # a single 64-key block -- the smallest geometry that exposes the effect.
    q_lens, kv_lens, heads, seed = [0, 64], [64, 64], 2, 10
    q, k, v, do, cu_q, cu_kv = impl._build_packed(q_lens, kv_lens, heads, 192, 128,
                                                  BF16, seed)
    lse, delta = impl.forward_stats(q, k, v, do, cu_q, cu_kv, slab_heads=2)
    dq32, dk, dv, oob, _ = impl._execute(q, k, v, do, lse, delta, cu_q, cu_kv, True,
                                         192**-0.5, BF16)
    assert oob == 0, f"unexpected out-of-range store: {oob} cell(s)"

    ideal = impl.golden_bwd_kernel(q, k, v, do, cu_q, cu_kv, scale=192**-0.5,
                                   dtype=torch.bfloat16, slab_heads=2, quantize=False)
    truth = impl.golden_bwd_kernel(q, k, v, do, cu_q, cu_kv, scale=192**-0.5,
                                   dtype=torch.bfloat16, slab_heads=2, quantize=True)
    # Request 1 lives at q rows [0, 64) and k/v rows [64, 128).
    shape_dv, shape_dk = (64, heads, 128), (64, heads, 192)
    cases = (
        ("dk", dk[64:128], ideal[1][64:128].reshape(shape_dk), truth[1][64:128].reshape(shape_dk), impl._ATOL),
        ("dv", dv[64:128], ideal[2][64:128].reshape(shape_dv), truth[2][64:128].reshape(shape_dv), impl._ATOL),
        ("dq32", dq32[0:64], ideal[3][0:64].reshape(shape_dk), truth[3][0:64].reshape(shape_dk), DQ_ATOL),
    )

    # (1) A handful of cells miss the ideal reference ...
    bad_ideal = 0
    for tag, got, ref_ideal, _, atol in cases:
        bad, gap = cells_outside(got, ref_ideal, atol, RTOL)
        bad_ideal += bad
        print(f"    {tag}: {bad} cell(s) outside the ideal atol, max gap {gap:.3e}", flush=True)
    assert 0 < bad_ideal < 5, f"expected 1..4 cells outside the ideal atol, got {bad_ideal}"

    # (2) ... while the source-semantics reference (same 16-bit operand rounding)
    # is matched on every cell of every tensor.
    for tag, got, _, ref_truth, atol in cases:
        bad, gap = cells_outside(got, ref_truth, atol, RTOL)
        assert bad == 0, f"{tag}: {bad} cell(s) outside the second reference (max {gap:.3e})"

    # (3) Away from those few cells the kernel is inside its own atol: the max
    # residual against the ideal stays below 8x the atol of that tensor.
    for tag, got, ref_ideal, _, atol in cases:
        gap = float((got.float() - ref_ideal.float()).abs().max())
        print(f"    {tag}: max gap to the ideal {gap:.3e} (atol {atol:.3e})", flush=True)
        assert gap <= 8 * atol, f"{tag}: gap to the ideal {gap:.3e} exceeds 8x atol"

    # (4) The cell that misses the tolerance is a genuine cancellation:
    # sum|terms| >> |value|.  Only cells outside their own tolerance count here
    # (a dq32 residual inside atol is ordinary fp32 accumulation noise).
    shapes, candidates = {}, []
    for tag, got, ref_ideal, _, atol in cases:
        shapes[tag] = (got.shape, ref_ideal)
        g = (got.float() - ref_ideal.float()).abs().reshape(-1)
        t = (atol + RTOL * ref_ideal.float().abs()).reshape(-1)
        for flat in torch.nonzero(g > t).reshape(-1).tolist():
            candidates.append((float(g[flat]), tag, flat))
    assert candidates, "no cell exceeded its tolerance -- nothing to classify"
    candidates.sort(reverse=True)
    gap, tag, flat = candidates[0]
    (_, _, nd), ref_ideal = shapes[tag]
    nh = shapes[tag][0][1]
    row, head, dim = flat // (nh * nd), (flat // nd) % nh, flat % nd
    qf = q[0:64].float()
    kf, vf = k[64:128].float(), v[64:128].float()
    dof = do[0:64].float()
    s = torch.einsum("shd,nhd->hsn", qf, kf).mul_(192**-0.5)
    pos = torch.arange(64)
    s.masked_fill_(pos[None, :] > pos[:, None], float("-inf"))
    p = torch.exp(s - lse[:, 0:64][:, None]).nan_to_num(0.0)
    dp = torch.einsum("shd,nhd->hsn", dof, vf)
    ds = p * (dp - (p * dp).sum(-1, keepdim=True))
    if tag == "dk":
        terms = ds[head, :, row] * qf[:, head, dim] * 192**-0.5
    elif tag == "dv":
        terms = p[head, :, row] * dof[:, head, dim]
    else:  # dq: dQ[m, :] = sum_n dS[m, n] k[n, :]
        terms = ds[head, row, :] * kf[:, head, dim]
    magnitude = float(terms.abs().sum())
    value = abs(float(ref_ideal[row, head, dim]))
    ratio = magnitude / max(value, 1e-12)
    print(f"worst {tag} cell (row={row}, head={head}, dim={dim}): "
          f"ideal={value:.6f}, sum|terms|={magnitude:.3f} -> cancellation {ratio:.0f}x, "
          f"gap to ideal {gap:.2e}", flush=True)
    assert ratio > 10.0, f"expected a cancelling cell, ratio {ratio:.1f}"

    print("[CONST-950-16bit-operand-cancellation] ALL_ASSERTIONS_HOLD: "
          f"{bad_ideal} cell(s) miss the ideal atol, 0 miss the 16-bit-cast reference",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
