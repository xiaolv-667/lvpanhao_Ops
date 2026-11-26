"""Stage 5 non-silent check of ``ds-v3-b2-8k`` through the *integrated* TileOPs op.

The Stage 5 requirement is that the 4-D batch-view wrapper (DESIGN.md v3 0.2 /
5.4, per-request dk/dv store window) survives integration: with a 3-D packed
view request ``b``'s tail key block would overwrite request ``b+1``'s first key
rows.  This script therefore

1. builds B=2/S=8192/H=128/D_qk=192/D_v=128 bf16 inputs (production shape),
2. produces ``o``/``lse`` with the same chunked float32 forward the development
   ``--level big`` probe uses (``_stats_npu`` shape; recomputed here only to
   also return ``o``, no golden is rewritten),
3. runs ``MLADenseBwdOp`` (the integrated op -> integrated wrapper -> integrated
   ``_bwd_kernel``) and times the whole call,
4. asserts the gradients are non-zero/finite for BOTH requests, that request
   1's first 64 key rows are populated (cross-request overwrite detector), and
5. compares a spread of ``(request, head)`` slices against the development
   fp64 ``golden_head`` reference (reused, not rewritten).

Usage:
    python check_b2s8k_op.py [--heads 0,127] [--reps 2]
"""

from __future__ import annotations

import argparse
import importlib.util
import time

import torch
import torch_npu  # noqa: F401  (registers the npu device)

_TASK_KERNEL = (
    "/workspace/tilelang-mlir-ascend-950/examples/mla_dense_bwd/_bwd_kernel/_bwd_kernel.py"
)
_spec = importlib.util.spec_from_file_location("bwd_dev_kernel", _TASK_KERNEL)
bk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bk)

from tileops.ops.attention.mla_dense_bwd import MLADenseBwdOp  # noqa: E402

B, S, H, DQK, DV = 2, 8192, 128, 192, 128
ATOL = bk._ATOL  # dev-stage DESIGN 8.2 floor (5e-3), stricter than the op test
RTOL = bk._RTOL["bfloat16"]
ATOL_SUITE = max(5e-3, RTOL)  # tests/ops/test_mla_dense_bwd.py:_tolerance(bf16)


def forward_stats(q, k, v, dout, is_causal=True, chunk=4096):
    """Chunked float32 forward on the NPU: ``(o, lse)`` for every head.

    Same per-head/per-chunk loop as ``bk._stats_npu`` (the development
    ``--level big`` statistics path), extended to also return ``o``, because the
    integrated op recomputes ``delta = rowsum(o * do)`` from the ``o`` it is
    given -- so ``o`` and ``lse`` must come from the same forward pass.
    """
    seq_q, dim_qk = q.shape[1], q.shape[-1]
    seq_kv = k.shape[1]
    scale = dim_qk**-0.5
    shift = seq_kv - seq_q
    batch, heads = q.shape[0], q.shape[2]
    dev = q.device
    lse = torch.empty(batch, heads, seq_q, dtype=torch.float32, device=dev)
    o = torch.empty(batch, seq_q, heads, v.shape[-1], dtype=q.dtype, device=dev)
    key_pos = torch.arange(seq_kv, device=dev)
    for b in range(batch):
        for h in range(heads):
            kk = k[b, :, h, :]
            vv = v[b, :, h, :]
            for m0 in range(0, seq_q, chunk):
                m1 = min(m0 + chunk, seq_q)
                s = torch.matmul(q[b, m0:m1, h, :], kk.t()).float() * scale
                if is_causal:
                    q_pos = torch.arange(m0, m1, device=dev) + shift
                    s = s.masked_fill(key_pos[None, :] > q_pos[:, None], float("-inf"))
                row = torch.logsumexp(s, dim=-1)
                p = torch.exp(s - row[:, None]).nan_to_num(0.0)
                lse[b, h, m0:m1] = row
                o[b, m0:m1, h, :] = torch.matmul(p.to(v.dtype), vv)
    return o, lse.contiguous()


def golden_head_given_stats(q_bh, k_bh, v_bh, do_bh, scale, lse_bh, delta_bh,
                            chunk=1024, round_inputs=False, dtype=None):
    """fp64 reference for ONE ``(request, head)`` slice with the *given* ``lse``/``delta``.

    ``bk.golden_head`` recomputes ``lse``/``delta`` in fp64 from q/k/do, while the
    integrated op receives ``lse`` from the forward pass and recomputes
    ``delta = rowsum(o * do)`` from that forward's ``o``.  Comparing the kernel
    against a reference that re-derives those two statistics therefore mixes the
    kernel's error with the statistics' precision; this variant holds them fixed
    at the values the kernel actually consumes, so the comparison isolates the
    kernel/wrapper.  Same chunked structure as ``golden_head``.
    """
    seq_q, dim_qk = q_bh.shape
    seq_kv, dim_v = k_bh.shape[0], v_bh.shape[-1]
    shift = seq_kv - seq_q
    qf, kf, vf, dof = (t.double() for t in (q_bh, k_bh, v_bh, do_bh))
    dq = torch.zeros(seq_q, dim_qk, dtype=torch.float64)
    dk = torch.zeros(seq_kv, dim_qk, dtype=torch.float64)
    dv = torch.zeros(seq_kv, dim_v, dtype=torch.float64)
    key_pos = torch.arange(seq_kv)
    for m0 in range(0, seq_q, chunk):
        m1 = min(m0 + chunk, seq_q)
        s = torch.einsum("sd,nd->sn", qf[m0:m1], kf) * scale
        if True:
            q_pos = torch.arange(m0, m1) + shift
            s = s.masked_fill(key_pos[None, :] > q_pos[:, None], float("-inf"))
        p = torch.exp(s - lse_bh[m0:m1].double()[:, None]).nan_to_num(0.0)
        dp = torch.einsum("sd,nd->sn", dof[m0:m1], vf)
        ds = p * (dp - delta_bh[m0:m1].double()[:, None])
        if round_inputs:
            p = p.to(dtype).to(torch.float64)
            ds = ds.to(dtype).to(torch.float64)
        dv += torch.einsum("sn,sd->nd", p, dof[m0:m1])
        dk += torch.einsum("sn,sd->nd", ds, qf[m0:m1])
        dq[m0:m1] = torch.einsum("sn,nd->sd", ds, kf)
    return dq * scale, dk * scale, dv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heads", default="0,127")
    parser.add_argument("--reps", type=int, default=2)
    args = parser.parse_args()
    heads_check = tuple(int(x) for x in args.heads.split(",") if x != "")
    torch.npu.set_device(0)
    dev = "npu"
    dt = torch.bfloat16
    ok = True

    print(f"[chk] B={B} S={S} H={H} D={DQK}/{DV} bf16 causal, heads_check={heads_check}",
          flush=True)
    t0 = time.perf_counter()
    q = torch.randn((B, S, H, DQK), dtype=torch.float32, device=dev).to(dt)
    k = torch.randn((B, S, H, DQK), dtype=torch.float32, device=dev).to(dt)
    v = torch.randn((B, S, H, DV), dtype=torch.float32, device=dev).to(dt)
    do = torch.randn((B, S, H, DV), dtype=torch.float32, device=dev).to(dt)
    o, lse = forward_stats(q, k, v, do, True)
    torch.npu.synchronize()
    print(f"[chk] inputs + forward stats in {time.perf_counter() - t0:.1f} s", flush=True)

    op = MLADenseBwdOp(is_causal=True)
    t0 = time.perf_counter()
    dq, dk, dv = op(do, q, k, v, o, lse)
    torch.npu.synchronize()
    print(f"[chk] first op call (compile + run) {time.perf_counter() - t0:.1f} s", flush=True)

    best = float("inf")
    for _ in range(args.reps):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        op(do, q, k, v, o, lse)
        torch.npu.synchronize()
        best = min(best, time.perf_counter() - t0)
    print(f"[chk] op T_op = {best * 1e3:.3f} ms (min of {args.reps}, whole call, 4 kernels)",
          flush=True)

    # ---- non-silent / per-request structure ------------------------------
    for name, t in (("dq", dq), ("dk", dk), ("dv", dv)):
        finite = bool(torch.isfinite(t).all())
        f32 = t.float()
        per_req = [
            f"b{b}: {int((f32[b] != 0).sum())}/{f32[b].numel()} nz, "
            f"max|.|={f32[b].abs().max().item():.3e}"
            for b in range(B)
        ]
        print(f"[chk] {name}: finite={finite} | " + " | ".join(per_req), flush=True)
        if not finite:
            ok = False
        for b in range(B):
            nz = int((f32[b] != 0).sum())
            if nz < f32[b].numel() // 100:
                ok = False
                print(f"[chk][FAIL] {name} request {b} is mostly zero ({nz} non-zero)", flush=True)

    for name, t, dim in (("dk", dk, DQK), ("dv", dv, DV)):
        victim = t[1, 0:64].float()
        zeros = int((victim == 0).sum())
        print(f"[chk] request 1 key 0..63 {name}: zeros {zeros}/{victim.numel()}, "
              f"max|.|={victim.abs().max().item():.3e} "
              f"(request 0 same rows max|.|={t[0, 0:64].float().abs().max().item():.3e})",
              flush=True)
        if zeros > victim.numel() // 100:
            ok = False
            print(f"[chk][FAIL] request 1 key rows are mostly zero -> cross-request overwrite",
                  flush=True)

    # ---- sliced fp64 reference (reused development golden) ---------------
    worst = {"dq": 0.0, "dk": 0.0, "dv": 0.0}
    over_rounded = {"dq": 0, "dk": 0, "dv": 0}
    over_fp64 = {"dq": 0, "dk": 0, "dv": 0}
    worst_given = {"dq": 0.0, "dk": 0.0, "dv": 0.0}
    over_given = {"dq": 0, "dk": 0, "dv": 0}
    over_given_suite = {"dq": 0, "dk": 0, "dv": 0}
    delta_ref = (o.float() * do.float()).sum(-1)  # the op's own delta, (B, S, H)
    scale = DQK**-0.5
    for b in range(B):
        for h in heads_check:
            t0 = time.perf_counter()
            args_h = (q[b, :, h, :].cpu(), k[b, :, h, :].cpu(),
                      v[b, :, h, :].cpu(), do[b, :, h, :].cpu())
            dq_t, dk_t, dv_t, _, _ = bk.golden_head(*args_h, scale, True)
            r_dq, r_dk, r_dv, _, _ = bk.golden_head(*args_h, scale, True,
                                                   round_inputs=True, dtype=dt)
            g = golden_head_given_stats(*args_h, scale, lse[b, h].cpu(), delta_ref[b, :, h].cpu())
            gr = golden_head_given_stats(*args_h, scale, lse[b, h].cpu(), delta_ref[b, :, h].cpu(),
                                        round_inputs=True, dtype=dt)
            got = {
                "dq": dq[b, :, h, :].detach().cpu().double(),
                "dk": dk[b, :, h, :].detach().cpu().double(),
                "dv": dv[b, :, h, :].detach().cpu().double(),
            }
            rows = []
            for key, ref, rref, gref, grref in (
                ("dq", dq_t, r_dq, g[0], gr[0]),
                ("dk", dk_t, r_dk, g[1], gr[1]),
                ("dv", dv_t, r_dv, g[2], gr[2]),
            ):
                gap = (got[key] - ref).abs()
                thr = ATOL + RTOL * ref.abs()
                worst[key] = max(worst[key], float(gap.max()))
                over_fp64[key] += int((gap > thr).sum())
                over_rounded[key] += int(((got[key] - rref).abs() > (ATOL + RTOL * rref.abs())).sum())
                ggap = (got[key] - gref).abs()
                gthr = ATOL + RTOL * gref.abs()
                ratio = (ggap / gthr)
                worst_given[key] = max(worst_given[key], float(ggap.max()))
                if key == "dq" and b == 0:
                    big = gref.abs() >= 0.1
                    print(f"[chk]   dq ratio stats b={b} h={h}: max ratio {float(ratio.max()):.2f} "
                          f"({int((ratio > 1).sum())}/{ratio.numel()} over) | "
                          f"|ref|>=0.1: max ratio {float(ratio[big].max()):.2f} "
                          f"({int((ratio[big] > 1).sum())}/{int(big.sum())} over) | "
                          f"max|ref|={float(gref.abs().max()):.2e}", flush=True)
                    if int((ratio > 1).sum()):
                        flat = ratio.flatten().topk(min(5, ratio.numel()))
                        for val, flat_idx in zip(flat.values.tolist(), flat.indices.tolist()):
                            if val <= 1:
                                break
                            r, c = divmod(flat_idx, ratio.shape[1])
                            print(f"[chk]     dq[{r},{c}]: got={float(got[key][r, c]):.5f} "
                                  f"ref={float(gref[r, c]):.5f} gap={float(ggap[r, c]):.3e} "
                                  f"thr={float(gthr[r, c]):.3e} ratio={val:.2f}", flush=True)
                suite_thr = ATOL_SUITE + RTOL * gref.abs()
                over_given_suite[key] += int(((ggap > suite_thr)
                                              | ((got[key] - grref).abs()
                                                 > (ATOL_SUITE + RTOL * grref.abs()))).sum())
                bad = (ggap > gthr) | ((got[key] - grref).abs() > (ATOL + RTOL * grref.abs()))
                n_bad = int(bad.sum())
                over_given[key] += n_bad
                if n_bad:
                    idx = bad.nonzero()[:3].tolist()
                    detail = [
                        (tuple(i), float(got[key][tuple(i)]), float(gref[tuple(i)]),
                         float(ggap[tuple(i)]), float(gthr[tuple(i)]))
                        for i in idx
                    ]
                    print(f"[chk]   given-stats outliers {key} b={b} h={h}: {detail}", flush=True)
                rows.append(f"{key} fp64={float(gap.max()):.3e} given={float(ggap.max()):.3e}")
            print(f"[chk] b={b} h={h:3d}: {' '.join(rows)} "
                  f"(golden {time.perf_counter() - t0:.1f} s)", flush=True)
    print(f"[chk] elements over (atol+rtol|ref|) vs the fp64 golden that re-derives lse: "
          f"{over_fp64} (same-rounding {over_rounded})", flush=True)
    print(f"[chk] elements over, reference holding the op's own lse/delta: "
          f"dev atol {ATOL:g} + rtol {RTOL:g} -> {over_given} | "
          f"op-test atol {ATOL_SUITE:g} + rtol {RTOL:g} -> {over_given_suite}", flush=True)
    for key in ("dq", "dk", "dv"):
        if over_given_suite[key]:
            ok = False
            print(f"[chk][FAIL] {key}: {over_given_suite[key]} element(s) outside the "
                  f"op-test tolerance with the same lse/delta the kernel consumes", flush=True)

    print(f"[chk] worst |gap| dq={worst['dq']:.3e} dk={worst['dk']:.3e} dv={worst['dv']:.3e} "
          f"| given-stats dq={worst_given['dq']:.3e} dk={worst_given['dk']:.3e} "
          f"dv={worst_given['dv']:.3e}", flush=True)
    print(f"[chk] RESULT {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
