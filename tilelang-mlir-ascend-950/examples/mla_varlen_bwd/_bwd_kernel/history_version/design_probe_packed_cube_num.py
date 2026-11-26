#!/usr/bin/env python3
"""Stage-1 numeric prototype: packed MLA backward on the dense-cube (C2) core.

Same body as `design_probe_packed_cube.py` (compile probe) but with a
per-geometry factory and a torch reference, so the packed-specific deltas
(fused decode, head-major stats, per-request write-back, head slab) are
checked numerically on device -- not just compiled.

Reference (fp32, per request, bottom-right causal), deliberately *not* the
kernel's blocking:
    s   = scale * q @ k^T
    vis = (n <= m + shift) & (n < lkv) & (m < lq)
    lse[m] = log(sum_n exp(scale*s)) over visible n     (-inf when none)
    p   = exp(scale*s - lse) * vis
    dp  = do @ v^T
    delta[m] = sum_n p*dp          (== rowsum(o*do) when o is the forward output)
    dV = p^T @ do ;  dK = scale * dS^T @ q ;  dQ = dS @ k   (dq unscaled)
Gradients are rounded to the output dtype once, as the workload does.

Run (device, single card):
    cd examples/mla_varlen_bwd/_bwd_kernel
    TMPDIR=/workspace /workspace/venv-ascend950/bin/python \
        history_version/design_probe_packed_cube_num.py
"""

import math

import torch
import torch_npu  # noqa: F401
import tilelang
import tilelang.ascend.language as T

DQK, DV = 192, 128
BN, BM = 96, 64
HN = BN // 2
SCALE = DQK**-0.5


def build(batch, heads, dim_qk, dim_v, dt, slab):
    """Packed kernel.  ``slab`` heads per launch (== heads -> single launch)."""
    total_q = T.dynamic("total_q")
    total_kv = T.dynamic("total_kv")
    grid = (T.floordiv(total_kv, BN) + batch) * slab

    @T.macro
    def decode(cu_kv, tile, req, n0l, cnt, live):
        req[0] = T.int32(0)
        n0l[0] = T.int32(0)
        cnt[0] = T.int32(0)
        for g in T.serial(batch):
            ntg = (cu_kv[g + 1] - cu_kv[g] + T.int32(BN - 1)) // T.int32(BN)
            if (tile >= cnt[0]) & (tile < cnt[0] + ntg):
                req[0] = g
                n0l[0] = (tile - cnt[0]) * T.int32(BN)
            cnt[0] = cnt[0] + ntg
        live[0] = T.if_then_else(tile < cnt[0], T.int32(1), T.int32(0))

    @T.macro
    def body(
        q, k, v, dout, lse, delta, dq_acc, dk, dv,
        k_s, v_s, kT_s, q_s, qT_s, do_s, doT_s, pT_s, dsT_s,
        a_k, a_t, b_lo, b_hi, st_c, dpt_c, dq_c, dk_c, dv_c,
        st_u, dp_u, p16_u, dsT_u, lse_u, dlt_u, tail_u, tailv_u, out_k, out_v,
        h, sid, n0_dyn, lq, lkv, shift, kv_start, q_start,
    ):
        koff = kv_start + n0_dyn
        T.copy(k[koff : koff + BN, h, :], k_s)
        T.copy(v[koff : koff + BN, h, :], v_s)
        T.copy(k[koff : koff + BN, h, :], kT_s, transpose=True)
        loop_st = T.max(0, T.floordiv(n0_dyn - shift, BM))
        loop_ed = T.max(T.ceildiv(lq, BM), loop_st + 1)
        for kq in T.Pipelined(loop_st, loop_ed, num_stages=2):
            m0 = kq * BM
            qoff = q_start + m0
            T.copy(q[qoff : qoff + BM, h, :], q_s)
            T.copy(dout[qoff : qoff + BM, h, :], do_s)
            T.copy(q[qoff : qoff + BM, h, :], qT_s, transpose=True)
            T.copy(dout[qoff : qoff + BM, h, :], doT_s, transpose=True)
            T.copy(lse[h, qoff : qoff + BM], lse_u)
            T.copy(delta[h, qoff : qoff + BM], dlt_u)

            T.copy(k_s, a_k)
            T.copy(q_s, b_lo)
            T.gemm(a_k, b_lo, st_c, transpose_B=True, clear_accum=True)
            T.copy(v_s, a_k[0:BN, 0:dim_v])
            T.copy(do_s, b_lo[0:BM, 0:dim_v])
            T.gemm(a_k[0:BN, 0:dim_v], b_lo[0:BM, 0:dim_v], dpt_c,
                   transpose_B=True, clear_accum=True)
            T.dual_copy(st_c, st_u)
            T.dual_copy(dpt_c, dp_u)

            fast = (n0_dyn + BN <= lkv) & (m0 + BM <= lq) & (n0_dyn + BN - 1 <= m0 + shift)
            if fast:
                with T.SimtVF(threads=128):
                    for i, j in T.Parallel(HN, BM):
                        p = T.exp(st_u[i, j] * SCALE - lse_u[j])
                        p16_u[i, j] = T.cast(p, dt)
                        dsT_u[i, j] = T.cast(p * (dp_u[i, j] - dlt_u[j]), dt)
            else:
                with T.SimtVF(threads=128):
                    for i, j in T.Parallel(HN, BM):
                        gi = n0_dyn + T.cast(sid, "int32") * HN + i
                        gj = m0 + j
                        vis = (gi < lkv) & (gj < lq) & (gi <= gj + shift)
                        p = T.exp(st_u[i, j] * SCALE - lse_u[j])
                        pv = T.if_then_else(vis, p, T.cast(0, "float32"))
                        p16_u[i, j] = T.cast(pv, dt)
                        dsT_u[i, j] = T.cast(pv * (dp_u[i, j] - dlt_u[j]), dt)

            T.dual_copy(p16_u, pT_s)
            T.dual_copy(dsT_u, dsT_s)
            T.copy(pT_s, a_k[0:BN, 0:BM])
            T.copy(doT_s, b_hi[0:dim_v, 0:BM])
            T.gemm(a_k[0:BN, 0:BM], b_hi[0:dim_v, 0:BM], dv_c,
                   transpose_B=True, clear_accum=(kq == loop_st))
            T.copy(dsT_s, a_k[0:BN, 0:BM])
            T.copy(qT_s, b_hi[0:dim_qk, 0:BM])
            T.gemm(a_k[0:BN, 0:BM], b_hi[0:dim_qk, 0:BM], dk_c,
                   transpose_B=True, clear_accum=(kq == loop_st))
            T.copy(dsT_s, a_t)
            T.copy(kT_s, b_hi)
            T.gemm(a_t, b_hi, dq_c, transpose_A=True, transpose_B=True, clear_accum=True)
            T.set_atomic("add", "float32")
            T.copy(dq_c, dq_acc[qoff : qoff + BM, h, :])
            T.set_atomic_none()

        T.dual_copy(dk_c, tail_u)
        T.dual_copy(dv_c, tailv_u)
        with T.SimtVF(threads=128):
            for i, d in T.Parallel(HN, dim_qk):
                out_k[i, d] = T.cast(tail_u[i, d] * SCALE, dt)
            for i, d in T.Parallel(HN, dim_v):
                out_v[i, d] = T.cast(tailv_u[i, d], dt)
        own = koff + T.cast(sid, "int32") * HN
        kv_end = kv_start + lkv
        if own + HN <= kv_end:
            T.copy(out_k, dk[own : own + HN, h, :])
            T.copy(out_v, dv[own : own + HN, h, :])
        else:
            with T.SimtVF(threads=128):
                for i, d in T.Parallel(HN, dim_qk):
                    if own + i < kv_end:
                        dk[own + i, h, d] = out_k[i, d]
                for i, d in T.Parallel(HN, dim_v):
                    if own + i < kv_end:
                        dv[own + i, h, d] = out_v[i, d]

    @T.prim_func
    def main(
        q: T.Tensor((total_q, heads, dim_qk), dt),
        k: T.Tensor((total_kv, heads, dim_qk), dt),
        v: T.Tensor((total_kv, heads, dim_v), dt),
        dout: T.Tensor((total_q, heads, dim_v), dt),
        lse: T.Tensor((heads, total_q), "float32"),
        delta: T.Tensor((heads, total_q), "float32"),
        cu_q: T.Tensor((batch + 1,), "int32"),
        cu_kv: T.Tensor((batch + 1,), "int32"),
        h_off: T.Tensor((1,), "int32"),
        dq_acc: T.Tensor((total_q, heads, dim_qk), "float32"),
        dk: T.Tensor((total_kv, heads, dim_qk), dt),
        dv: T.Tensor((total_kv, heads, dim_v), dt),
    ):
        with T.Kernel(grid) as bid:
            tile = bid // slab                  # static divisor
            h = h_off[0] + bid % slab           # run-time origin + compile-time slot
            sid = T.call_extern("int64", "asc_get_sub_block_id")
            k_s = T.alloc_l1((BN, dim_qk), dt)
            v_s = T.alloc_l1((BN, dim_v), dt)
            kT_s = T.alloc_l1((dim_qk, BN), dt)
            q_s = T.alloc_l1((BM, dim_qk), dt)
            qT_s = T.alloc_l1((dim_qk, BM), dt)
            do_s = T.alloc_l1((BM, dim_v), dt)
            doT_s = T.alloc_l1((dim_v, BM), dt)
            pT_s = T.alloc_l1((BN, BM), dt)
            dsT_s = T.alloc_l1((BN, BM), dt)
            a_k = T.alloc_l0a((BN, dim_qk), dt)
            a_t = T.alloc_l0a((BN, BM), dt)
            b_lo = T.alloc_l0b((BM, dim_qk), dt)
            b_hi = T.alloc_l0b((dim_qk, BN), dt)
            st_c = T.alloc_l0c((BN, BM), "float32")
            dpt_c = T.alloc_l0c((BN, BM), "float32")
            dq_c = T.alloc_l0c((BM, dim_qk), "float32")
            dk_c = T.alloc_l0c((BN, dim_qk), "float32")
            dv_c = T.alloc_l0c((BN, dim_v), "float32")
            st_u = T.alloc_shared((HN, BM), "float32")
            dp_u = T.alloc_shared((HN, BM), "float32")
            p16_u = T.alloc_shared((HN, BM), dt)
            dsT_u = T.alloc_shared((HN, BM), dt)
            lse_u = T.alloc_shared((BM,), "float32")
            dlt_u = T.alloc_shared((BM,), "float32")
            tail_u = T.alloc_shared((HN, dim_qk), "float32")
            tailv_u = T.alloc_shared((HN, dim_v), "float32")
            out_k = T.alloc_shared((HN, dim_qk), dt)
            out_v = T.alloc_shared((HN, dim_v), dt)

            req = T.alloc_local([1], "int32")
            n0l = T.alloc_local([1], "int32")
            cnt = T.alloc_local([1], "int32")
            live = T.alloc_local([1], "int32")
            if h < heads:                          # tail-slab guard
                decode(cu_kv, tile, req, n0l, cnt, live)
                if live[0] == T.int32(1):
                    q_start = cu_q[req[0]]
                    kv_start = cu_kv[req[0]]
                    lq = cu_q[req[0] + 1] - q_start
                    lkv = cu_kv[req[0] + 1] - kv_start
                    body(
                        q, k, v, dout, lse, delta, dq_acc, dk, dv,
                        k_s, v_s, kT_s, q_s, qT_s, do_s, doT_s, pT_s, dsT_s,
                        a_k, a_t, b_lo, b_hi, st_c, dpt_c, dq_c, dk_c, dv_c,
                        st_u, dp_u, p16_u, dsT_u, lse_u, dlt_u, tail_u, tailv_u,
                        out_k, out_v,
                        h, sid, n0l[0], lq, lkv, lkv - lq, kv_start, q_start,
                    )

    return tilelang.compile(main, target="ascend")


# ---------------------------------------------------------------------------
def reference(q, k, v, dout, lens_q, lens_kv, heads, dtype):
    """fp32 per-request reference; returns (dk, dv, dq32, lse, delta) packed."""
    T_q = sum(lens_q)
    T_kv = sum(lens_kv)
    dk = torch.zeros(T_kv, heads, DQK, dtype=torch.float32)
    dv = torch.zeros(T_kv, heads, DV, dtype=torch.float32)
    dq = torch.zeros(T_q, heads, DQK, dtype=torch.float32)
    lse = torch.full((heads, T_q), float("-inf"))
    delta = torch.zeros(heads, T_q)
    oq = okv = 0
    for lq, lkv in zip(lens_q, lens_kv):
        sl, skv = slice(oq, oq + lq), slice(okv, okv + lkv)
        shift = lkv - lq
        m = torch.arange(lq)[:, None]
        n = torch.arange(lkv)[None, :]
        vis = n <= m + shift
        for h in range(heads):
            qb = q[sl, h].float()            # (lq, DQK)
            kb = k[skv, h].float()           # (lkv, DQK)
            vb = v[skv, h].float()
            dob = dout[sl, h].float()
            s = (qb @ kb.T) * SCALE
            sm = torch.where(vis, s, torch.full_like(s, float("-inf")))
            row_lse = torch.logsumexp(sm, dim=1)
            lse[h, sl] = row_lse
            p = torch.where(vis, torch.exp(s - row_lse[:, None]), torch.zeros(()))
            dp = dob @ vb.T
            dlt = (p * dp).sum(1)
            delta[h, sl] = dlt
            ds = p * (dp - dlt[:, None])
            p16 = p.to(dtype).float()
            ds16 = ds.to(dtype).float()
            dv[skv, h] = p16.T @ dob
            dk[skv, h] = SCALE * (ds16.T @ qb)
            dq[sl, h] = ds16 @ kb
        oq += lq
        okv += lkv
    return (
        dk.to(dtype), dv.to(dtype), dq,
        lse.contiguous(), delta.contiguous(),
    )


def run_case(name, lens_q, lens_kv, heads, dtype_name, seed=0, slab=None):
    dt = getattr(torch, dtype_name)
    batch = len(lens_q)
    T_q = sum(lens_q)
    T_kv = sum(lens_kv)
    g = torch.Generator().manual_seed(seed)
    q = (torch.randn(T_q, heads, DQK, generator=g) * 0.5).to(dt)
    k = (torch.randn(T_kv, heads, DQK, generator=g) * 0.5).to(dt)
    v = (torch.randn(T_kv, heads, DV, generator=g) * 0.5).to(dt)
    dout = (torch.randn(T_q, heads, DV, generator=g) * 0.5).to(dt)

    ref_dk, ref_dv, ref_dq, lse, delta = reference(
        q, k, v, dout, lens_q, lens_kv, heads, dt)

    cu_q = torch.tensor([0] + list(torch.cumsum(torch.tensor(lens_q), 0).tolist()),
                        dtype=torch.int32)
    cu_kv = torch.tensor([0] + list(torch.cumsum(torch.tensor(lens_kv), 0).tolist()),
                         dtype=torch.int32)
    dev = "npu:0"
    to = lambda t: t.to(dev)  # noqa: E731
    dq_acc = torch.zeros(T_q, heads, DQK, dtype=torch.float32, device=dev)
    dk = torch.zeros(T_kv, heads, DQK, dtype=dt, device=dev)
    dv = torch.zeros(T_kv, heads, DV, dtype=dt, device=dev)

    slab = slab or heads
    fn = build(batch, heads, DQK, DV, dtype_name, slab)
    h_offs = torch.tensor([[i * slab] for i in range((heads + slab - 1) // slab)],
                          dtype=torch.int32, device=dev)
    for i in range(h_offs.shape[0]):
        fn(to(q), to(k), to(v), to(dout), to(lse), to(delta), to(cu_q), to(cu_kv),
           h_offs[i], dq_acc, dk, dv)
    torch.npu.synchronize()

    got_dk, got_dv, got_dq = dk.cpu().float(), dv.cpu().float(), dq_acc.cpu()
    rk, rv, rq = ref_dk.float(), ref_dv.float(), ref_dq
    out = {}
    for tag, got, ref in (("dk", got_dk, rk), ("dv", got_dv, rv), ("dq", got_dq, rq)):
        denom = ref.abs().max().item() or 1.0
        out[tag] = (got - ref).abs().max().item() / denom
    # per-request worst relative error: catches a tail block leaking into the
    # next request (the failure mode a whole-tensor max can hide)
    oq = okv = 0
    worst = 0.0
    for lq, lkv in zip(lens_q, lens_kv):
        sl, skv = slice(oq, oq + lq), slice(okv, okv + lkv)
        for got, ref in ((got_dk[skv], rk[skv]), (got_dv[skv], rv[skv]),
                         (got_dq[sl], rq[sl])):
            worst = max(worst, (got - ref).abs().max().item() / (ref.abs().max().item() or 1.0))
        oq += lq
        okv += lkv
    ok = out["dk"] < 3e-2 and out["dv"] < 3e-2 and out["dq"] < 3e-2 and worst < 3e-2
    print(
        f"[{'PASS' if ok else 'FAIL'}] {name:28s} q={lens_q} kv={lens_kv} H={heads} "
        f"{dtype_name}: "
        f"dk={out['dk']:.3e} dv={out['dv']:.3e} dq={out['dq']:.3e} per_request_worst={worst:.3e}",
        flush=True,
    )
    return ok


def iterations(lens_q, lens_kv, heads, bn=BN, bm=BM):
    """Analytic iteration count of the adopted tile (DESIGN.md 1.6.0 R3)."""
    tot = 0
    for lq, lkv in zip(lens_q, lens_kv):
        shift = lkv - lq
        n_q = -(-lq // bm)
        for kn in range(-(-lkv // bn)):
            n0 = kn * bn
            st = max(0, (n0 - shift) // bm)
            tot += max(0, max(n_q, st + 1) - st)
    return tot * heads


def time_case(lens_q, lens_kv, heads, dtype_name="bfloat16", reps=3):
    """Kernel duration / analytic iteration count for the packed Cube kernel."""
    dt = getattr(torch, dtype_name)
    batch, T_q, T_kv = len(lens_q), sum(lens_q), sum(lens_kv)
    g = torch.Generator().manual_seed(1)
    q = (torch.randn(T_q, heads, DQK, generator=g) * 0.5).to(dt)
    k = (torch.randn(T_kv, heads, DQK, generator=g) * 0.5).to(dt)
    v = (torch.randn(T_kv, heads, DV, generator=g) * 0.5).to(dt)
    dout = (torch.randn(T_q, heads, DV, generator=g) * 0.5).to(dt)
    lse = torch.randn(heads, T_q, generator=g).abs() * 2
    delta = torch.randn(heads, T_q, generator=g)
    cu_q = torch.tensor([0] + list(torch.cumsum(torch.tensor(lens_q), 0).tolist()),
                        dtype=torch.int32)
    cu_kv = torch.tensor([0] + list(torch.cumsum(torch.tensor(lens_kv), 0).tolist()),
                         dtype=torch.int32)
    dev = "npu:0"
    to = lambda t: t.to(dev)  # noqa: E731
    dq_acc = torch.zeros(T_q, heads, DQK, dtype=torch.float32, device=dev)
    dk = torch.zeros(T_kv, heads, DQK, dtype=dt, device=dev)
    dv = torch.zeros(T_kv, heads, DV, dtype=dt, device=dev)
    fn = build(batch, heads, DQK, DV, dtype_name, heads)
    h_off = torch.zeros(1, dtype=torch.int32, device=dev)
    # all operands resident once: the H2D copies must not be inside the timing
    dq_, dk_, dv_ = to(q), to(k), to(v)
    do_, lse_, del_, cuq_, cukv_ = to(dout), to(lse), to(delta), to(cu_q), to(cu_kv)

    def once():
        fn(dq_, dk_, dv_, do_, lse_, del_, cuq_, cukv_, h_off, dq_acc, dk, dv)

    once()
    torch.npu.synchronize()
    best = float("inf")
    for _ in range(reps):
        s0, s1 = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        s0.record()
        once()
        s1.record()
        torch.npu.synchronize()
        best = min(best, s0.elapsed_time(s1))
    it = iterations(lens_q, lens_kv, heads)
    print(
        f"[TIME] q={lens_q} kv={lens_kv} H={heads} iters={it} "
        f"T={best:.3f} ms  per-iter={best*1e3/it*28:.3f} us (x28 cores)  "
        f"per-iter/allcores={best*1e3/it:.4f} us",
        flush=True,
    )
    return best


def main():
    import sys

    if "--time" in sys.argv:
        time_case([4096, 4096], [4096, 4096], 16)
        time_case([512, 4096], [512, 4096], 16)
        return 0
    cases = [
        ("exact 2 key blocks", [192, 192], [192, 192], 2, "bfloat16"),
        ("ragged + tail blocks", [128, 320], [128, 320], 2, "bfloat16"),
        ("len_q > len_kv (shift<0, lse=-inf)", [320, 128], [128, 320], 2, "bfloat16"),
        ("len_q < len_kv (shift>0)", [128, 192], [320, 384], 2, "bfloat16"),
        ("single key block", [96, 96], [96, 96], 2, "float16"),
        # head slab: heads=3, slab=2 -> 2 launches, last one half empty
        ("head slab 3/2 (tail slab)", [192, 320], [192, 320], 3, "bfloat16"),
    ]
    ok = True
    for name, lq, lkv, heads, dt in cases:
        ok &= run_case(name, lq, lkv, heads, dt)
    ok &= run_case("slab<heads explicit", [192, 320], [192, 320], 3, "bfloat16", slab=1)
    print("\n[PROTO_SUMMARY]", "ALL_PASS" if ok else "HAS_FAILURE", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
