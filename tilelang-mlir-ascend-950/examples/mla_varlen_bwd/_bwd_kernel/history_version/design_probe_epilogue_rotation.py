#!/usr/bin/env python3
"""Stage-1 probe (v5 revision): deferred-gradient-GEMM rotation vs the current body.

Question this probe answers with device/compile evidence, not by argument:

  1. Can the query loop be *hand rotated* so that ``GB(k) = {dV, dK, dQ}`` runs one
     iteration late -- beside ``EP(k+1)`` instead of behind it -- without relying on
     ``T.Pipelined``'s automatic buffer versioning?  (The generated code rewrites
     *every* access of a versioned buffer with the *current* iteration index, so a
     hand-lagged consumer would read the wrong version; the fills therefore have to
     be versioned explicitly.)
  2. Does that rotation change the L0C / L0A / L0B / UB budget?  (Design claim: no
     -- ``st_c``/``dpt_c``/``st_u``/``dp_u``/``p16_u``/``dsT_u`` all stay single
     version; only the six per-iteration L1 fill buffers and ``pT_s``/``dsT_s``
     need an explicit 2x.)
  3. Does the generated code cut the cross-unit edge, i.e. do the AIC's ``GB(k-1)``
     L1->L0 copies no longer wait on the AIV's *current* epilogue?
  4. Is it numerically equal to the validated body, and what is its per-iteration
     price relative to the production ``num_stages=3`` body?

Modes
  base : the production body (``T.Pipelined(num_stages=3)``, v4 DESIGN).
  rot  : the candidate.  ``T.serial`` loop unrolled by 2 (``kqA = loop_st + 2i``,
         ``kqB = kqA + 1``); each half body is
             fills(v) -> GA(kq) -> [AIV] EP(kq) -> GB(kq-1) with the other version
         so that ``GB`` of the previous iteration is issued while the AIV runs the
         current epilogue.  Halves use statically named buffers (version A / B), so
         no buffer is indexed by a runtime parity value and the epilogue can be
         instantiated twice instead of four times.

Run (device, single card, cwd = operator dir):
    TMPDIR=/workspace /workspace/venv-ascend950/bin/python \
        history_version/design_probe_epilogue_rotation.py --compile
    TMPDIR=/workspace /workspace/venv-ascend950/bin/python \
        history_version/design_probe_epilogue_rotation.py --numeric
    TMPDIR=/workspace /workspace/venv-ascend950/bin/python \
        history_version/design_probe_epilogue_rotation.py --time
"""

import sys

import torch
import tilelang
import tilelang.ascend.language as T

DQK, DV = 192, 128
BN, BM = 96, 64
HN = BN // 2
SCALE = DQK**-0.5


# ---------------------------------------------------------------------------
# fp32 per-request reference (identical to design_probe_packed_cube_num.py)
# ---------------------------------------------------------------------------
def reference(q, k, v, dout, lens_q, lens_kv, heads, dtype):
    T_q, T_kv = sum(lens_q), sum(lens_kv)
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
            qb = q[sl, h].float()
            kb = k[skv, h].float()
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
    return dk.to(dtype), dv.to(dtype), dq, lse.contiguous(), delta.contiguous()


# ---------------------------------------------------------------------------
def build(batch, heads, dim_qk, dim_v, dt, slab, mode="rot", stages=3):
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
    def fills(q, dout, lse, delta, q_s, do_s, qT_s, doT_s, lse_u, dlt_u, h, qoff):
        T.copy(q[qoff:qoff + BM, h, :], q_s)
        T.copy(dout[qoff:qoff + BM, h, :], do_s)
        T.copy(q[qoff:qoff + BM, h, :], qT_s, transpose=True)
        T.copy(dout[qoff:qoff + BM, h, :], doT_s, transpose=True)
        T.copy(lse[h, qoff:qoff + BM], lse_u)
        T.copy(delta[h, qoff:qoff + BM], dlt_u)

    @T.macro
    def ga(k_s, v_s, q_s, do_s, a_k, b_lo, st_c, dpt_c, st_u, dp_u):
        T.copy(k_s, a_k)
        T.copy(q_s, b_lo)
        T.gemm(a_k, b_lo, st_c, transpose_B=True, clear_accum=True)
        T.copy(v_s, a_k[0:BN, 0:dim_v])
        T.copy(do_s, b_lo[0:BM, 0:dim_v])
        T.gemm(a_k[0:BN, 0:dim_v], b_lo[0:BM, 0:dim_v], dpt_c,
               transpose_B=True, clear_accum=True)
        T.dual_copy(st_c, st_u)
        T.dual_copy(dpt_c, dp_u)

    @T.macro
    def epilogue(sid, st_u, dp_u, p16_u, dsT_u, lse_u, dlt_u, pT_dst, dsT_dst,
                 n0_dyn, m0, lq, lkv, shift):
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
        T.dual_copy(p16_u, pT_dst)
        T.dual_copy(dsT_u, dsT_dst)

    @T.macro
    def gb(pT_p, dsT_p, doT_p, qT_p, kT_s, a_k, a_t, b_hi,
           dq_c, dk_c, dv_c, dq_acc, qoff_p, h, clear_prev):
        T.copy(pT_p, a_k[0:BN, 0:BM])
        T.copy(doT_p, b_hi[0:dim_v, 0:BM])
        T.gemm(a_k[0:BN, 0:BM], b_hi[0:dim_v, 0:BM], dv_c,
               transpose_B=True, clear_accum=clear_prev)
        T.copy(dsT_p, a_k[0:BN, 0:BM])
        T.copy(qT_p, b_hi[0:dim_qk, 0:BM])
        T.gemm(a_k[0:BN, 0:BM], b_hi[0:dim_qk, 0:BM], dk_c,
               transpose_B=True, clear_accum=clear_prev)
        T.copy(dsT_p, a_t)
        # kT must be reloaded into b_hi AFTER doT/qT (DESIGN.md 9.2)
        T.copy(kT_s, b_hi)
        T.gemm(a_t, b_hi, dq_c, transpose_A=True, transpose_B=True, clear_accum=True)
        T.set_atomic("add", "float32")
        T.copy(dq_c, dq_acc[qoff_p:qoff_p + BM, h, :])
        T.set_atomic_none()

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
            tile = bid // slab
            h = h_off[0] + bid % slab
            sid = T.call_extern("int64", "asc_get_sub_block_id")
            # ---- persistent L1 ----
            k_s = T.alloc_l1((BN, dim_qk), dt)
            v_s = T.alloc_l1((BN, dim_v), dt)
            kT_s = T.alloc_l1((dim_qk, BN), dt)
            # ---- per-iteration L1: version A / B ----
            q_s0 = T.alloc_l1((BM, dim_qk), dt)
            qT_s0 = T.alloc_l1((dim_qk, BM), dt)
            do_s0 = T.alloc_l1((BM, dim_v), dt)
            doT_s0 = T.alloc_l1((dim_v, BM), dt)
            lse_u0 = T.alloc_shared((BM,), "float32")
            dlt_u0 = T.alloc_shared((BM,), "float32")
            pT_s0 = T.alloc_l1((BN, BM), dt)
            dsT_s0 = T.alloc_l1((BN, BM), dt)
            q_s1 = T.alloc_l1((BM, dim_qk), dt)
            qT_s1 = T.alloc_l1((dim_qk, BM), dt)
            do_s1 = T.alloc_l1((BM, dim_v), dt)
            doT_s1 = T.alloc_l1((dim_v, BM), dt)
            lse_u1 = T.alloc_shared((BM,), "float32")
            dlt_u1 = T.alloc_shared((BM,), "float32")
            pT_s1 = T.alloc_l1((BN, BM), dt)
            dsT_s1 = T.alloc_l1((BN, BM), dt)
            # ---- L0: single version (unchanged) ----
            a_k = T.alloc_l0a((BN, dim_qk), dt)
            a_t = T.alloc_l0a((BN, BM), dt)
            b_lo = T.alloc_l0b((BM, dim_qk), dt)
            b_hi = T.alloc_l0b((dim_qk, BN), dt)
            # ---- L0C: single version (unchanged) ----
            st_c = T.alloc_l0c((BN, BM), "float32")
            dpt_c = T.alloc_l0c((BN, BM), "float32")
            dq_c = T.alloc_l0c((BM, dim_qk), "float32")
            dk_c = T.alloc_l0c((BN, dim_qk), "float32")
            dv_c = T.alloc_l0c((BN, dim_v), "float32")
            # ---- UB half tiles: single version (unchanged) ----
            st_u = T.alloc_shared((HN, BM), "float32")
            dp_u = T.alloc_shared((HN, BM), "float32")
            p16_u = T.alloc_shared((HN, BM), dt)
            dsT_u = T.alloc_shared((HN, BM), dt)
            tail_u = T.alloc_shared((HN, dim_qk), "float32")
            tailv_u = T.alloc_shared((HN, dim_v), "float32")
            out_k = T.alloc_shared((HN, dim_qk), dt)
            out_v = T.alloc_shared((HN, dim_v), dt)

            req = T.alloc_local([1], "int32")
            n0l = T.alloc_local([1], "int32")
            cnt = T.alloc_local([1], "int32")
            live = T.alloc_local([1], "int32")
            if h < heads:
                decode(cu_kv, tile, req, n0l, cnt, live)
                if live[0] == T.int32(1):
                    q_start = cu_q[req[0]]
                    kv_start = cu_kv[req[0]]
                    lq = cu_q[req[0] + 1] - q_start
                    lkv = cu_kv[req[0] + 1] - kv_start
                    shift = lkv - lq
                    koff = kv_start + n0l[0]
                    T.copy(k[koff:koff + BN, h, :], k_s)
                    T.copy(v[koff:koff + BN, h, :], v_s)
                    T.copy(k[koff:koff + BN, h, :], kT_s, transpose=True)
                    loop_st = T.max(0, T.floordiv(n0l[0] - shift, BM))
                    loop_ed = T.max(T.ceildiv(lq, BM), loop_st + 1)

                    if mode == "base":
                        for kq in T.Pipelined(loop_st, loop_ed, num_stages=stages):
                            qoff = q_start + kq * BM
                            fills(q, dout, lse, delta, q_s0, do_s0, qT_s0, doT_s0,
                                  lse_u0, dlt_u0, h, qoff)
                            ga(k_s, v_s, q_s0, do_s0, a_k, b_lo, st_c, dpt_c, st_u, dp_u)
                            epilogue(sid, st_u, dp_u, p16_u, dsT_u, lse_u0, dlt_u0,
                                     pT_s0, dsT_s0, n0l[0], kq * BM, lq, lkv, shift)
                            gb(pT_s0, dsT_s0, doT_s0, qT_s0, kT_s, a_k, a_t, b_hi,
                               dq_c, dk_c, dv_c, dq_acc, qoff, h, kq == loop_st)
                    else:
                        # unrolled by 2: half A holds version A, half B version B.
                        # GB(kq-1) reads the *other* half's buffers, which were
                        # written by the previous half body (one epilogue earlier).
                        n_it = loop_ed - loop_st
                        n_half = (n_it + 1) // 2
                        for i in T.serial(n_half):
                            kqA = loop_st + i * 2
                            qoffA = q_start + kqA * BM
                            fills(q, dout, lse, delta, q_s0, do_s0, qT_s0, doT_s0,
                                  lse_u0, dlt_u0, h, qoffA)
                            ga(k_s, v_s, q_s0, do_s0, a_k, b_lo, st_c, dpt_c, st_u, dp_u)
                            epilogue(sid, st_u, dp_u, p16_u, dsT_u, lse_u0, dlt_u0,
                                     pT_s0, dsT_s0, n0l[0], kqA * BM, lq, lkv, shift)
                            if kqA > loop_st:
                                gb(pT_s1, dsT_s1, doT_s1, qT_s1, kT_s, a_k, a_t, b_hi,
                                   dq_c, dk_c, dv_c, dq_acc, qoffA - BM, h,
                                   kqA - 1 == loop_st)
                            kqB = kqA + 1
                            if kqB < loop_ed:
                                qoffB = q_start + kqB * BM
                                fills(q, dout, lse, delta, q_s1, do_s1, qT_s1, doT_s1,
                                      lse_u1, dlt_u1, h, qoffB)
                                ga(k_s, v_s, q_s1, do_s1, a_k, b_lo, st_c, dpt_c, st_u, dp_u)
                                epilogue(sid, st_u, dp_u, p16_u, dsT_u, lse_u1, dlt_u1,
                                         pT_s1, dsT_s1, n0l[0], kqB * BM, lq, lkv, shift)
                                gb(pT_s0, dsT_s0, doT_s0, qT_s0, kT_s, a_k, a_t, b_hi,
                                   dq_c, dk_c, dv_c, dq_acc, qoffB - BM, h,
                                   kqB - 1 == loop_st)
                        # drain: body(loop_ed) is virtual, so GB(last) is always
                        # pending.  Its version is the one written for kq = last:
                        # A when the iteration count is odd, B when it is even.
                        last = loop_ed - 1
                        if (n_it % 2) == 1:
                            gb(pT_s0, dsT_s0, doT_s0, qT_s0, kT_s, a_k, a_t, b_hi,
                               dq_c, dk_c, dv_c, dq_acc, q_start + last * BM, h,
                               last == loop_st)
                        else:
                            gb(pT_s1, dsT_s1, doT_s1, qT_s1, kT_s, a_k, a_t, b_hi,
                               dq_c, dk_c, dv_c, dq_acc, q_start + last * BM, h,
                               last == loop_st)

                    # ---- block tail: dk = scale * cast(dk_c), dv = cast(dv_c) ----
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
                        T.copy(out_k, dk[own:own + HN, h, :])
                        T.copy(out_v, dv[own:own + HN, h, :])
                    else:
                        with T.SimtVF(threads=128):
                            for i, d in T.Parallel(HN, dim_qk):
                                if own + i < kv_end:
                                    dk[own + i, h, d] = out_k[i, d]
                            for i, d in T.Parallel(HN, dim_v):
                                if own + i < kv_end:
                                    dv[own + i, h, d] = out_v[i, d]

    return tilelang.compile(main, target="ascend")


# ---------------------------------------------------------------------------
def make_inputs(lens_q, lens_kv, heads, dtype_name, seed=0):
    dt = getattr(torch, dtype_name)
    T_q, T_kv = sum(lens_q), sum(lens_kv)
    g = torch.Generator().manual_seed(seed)
    q = (torch.randn(T_q, heads, DQK, generator=g) * 0.5).to(dt)
    k = (torch.randn(T_kv, heads, DQK, generator=g) * 0.5).to(dt)
    v = (torch.randn(T_kv, heads, DV, generator=g) * 0.5).to(dt)
    dout = (torch.randn(T_q, heads, DV, generator=g) * 0.5).to(dt)
    ref_dk, ref_dv, ref_dq, lse, delta = reference(q, k, v, dout, lens_q, lens_kv, heads, dt)
    cu_q = torch.tensor([0] + list(torch.cumsum(torch.tensor(lens_q), 0).tolist()),
                        dtype=torch.int32)
    cu_kv = torch.tensor([0] + list(torch.cumsum(torch.tensor(lens_kv), 0).tolist()),
                         dtype=torch.int32)
    return dict(dt=dt, batch=len(lens_q), T_q=T_q, T_kv=T_kv, heads=heads,
                q=q, k=k, v=v, dout=dout, lse=lse, delta=delta, cu_q=cu_q, cu_kv=cu_kv,
                ref_dk=ref_dk, ref_dv=ref_dv, ref_dq=ref_dq)


def run_case(name, lens_q, lens_kv, heads, dtype_name="bfloat16", slab=None, mode="rot"):
    inp = make_inputs(lens_q, lens_kv, heads, dtype_name)
    dev = "npu:0"
    to = lambda t: t.to(dev)  # noqa: E731
    dt = inp["dt"]
    dq_acc = torch.zeros(inp["T_q"], heads, DQK, dtype=torch.float32, device=dev)
    dk = torch.zeros(inp["T_kv"], heads, DQK, dtype=dt, device=dev)
    dv = torch.zeros(inp["T_kv"], heads, DV, dtype=dt, device=dev)
    slab = slab or heads
    fn = build(inp["batch"], heads, DQK, DV, dtype_name, slab, mode=mode)
    n_launch = (heads + slab - 1) // slab
    h_offs = torch.tensor([[i * slab] for i in range(n_launch)], dtype=torch.int32, device=dev)
    for i in range(n_launch):
        fn(to(inp["q"]), to(inp["k"]), to(inp["v"]), to(inp["dout"]), to(inp["lse"]),
           to(inp["delta"]), to(inp["cu_q"]), to(inp["cu_kv"]), h_offs[i], dq_acc, dk, dv)
    torch.npu.synchronize()

    got = {"dk": dk.cpu().float(), "dv": dv.cpu().float(), "dq": dq_acc.cpu()}
    ref = {"dk": inp["ref_dk"].float(), "dv": inp["ref_dv"].float(), "dq": inp["ref_dq"]}
    out, worst = {}, 0.0
    for tag in ("dk", "dv", "dq"):
        denom = ref[tag].abs().max().item() or 1.0
        out[tag] = (got[tag] - ref[tag]).abs().max().item() / denom
    oq = okv = 0
    for lq, lkv in zip(lens_q, lens_kv):
        sl, skv = slice(oq, oq + lq), slice(okv, okv + lkv)
        for tag in ("dk", "dv"):
            rr = ref[tag][skv]
            worst = max(worst, (got[tag][skv] - rr).abs().max().item()
                        / (rr.abs().max().item() or 1.0))
        rr = ref["dq"][sl]
        worst = max(worst, (got["dq"][sl] - rr).abs().max().item()
                    / (rr.abs().max().item() or 1.0))
        oq += lq
        okv += lkv
    ok = all(v < 3e-2 for v in out.values()) and worst < 3e-2
    print(f"[{'PASS' if ok else 'FAIL'}] {mode:4s} {name:26s} q={lens_q} kv={lens_kv} "
          f"H={heads} slab={slab} {dtype_name}: dk={out['dk']:.3e} dv={out['dv']:.3e} "
          f"dq={out['dq']:.3e} per_request_worst={worst:.3e}", flush=True)
    return ok


def iterations(lens_q, lens_kv, heads, bn=BN, bm=BM):
    tot = 0
    for lq, lkv in zip(lens_q, lens_kv):
        shift = lkv - lq
        n_q = -(-lq // bm)
        for kn in range(-(-lkv // bn)):
            n0 = kn * bn
            st = max(0, (n0 - shift) // bm)
            tot += max(0, max(n_q, st + 1) - st)
    return tot * heads


def time_case(lens_q, lens_kv, heads, mode="rot", dtype_name="bfloat16", reps=3):
    inp = make_inputs(lens_q, lens_kv, heads, dtype_name)
    dev = "npu:0"
    to = lambda t: t.to(dev)  # noqa: E731
    dt = inp["dt"]
    dq_acc = torch.zeros(inp["T_q"], heads, DQK, dtype=torch.float32, device=dev)
    dk = torch.zeros(inp["T_kv"], heads, DQK, dtype=dt, device=dev)
    dv = torch.zeros(inp["T_kv"], heads, DV, dtype=dt, device=dev)
    fn = build(inp["batch"], heads, DQK, DV, dtype_name, heads, mode=mode)
    h_off = torch.zeros(1, dtype=torch.int32, device=dev)
    q_, k_, v_, do_ = to(inp["q"]), to(inp["k"]), to(inp["v"]), to(inp["dout"])
    lse_, del_, cuq_, cukv_ = to(inp["lse"]), to(inp["delta"]), to(inp["cu_q"]), to(inp["cu_kv"])

    def once():
        fn(q_, k_, v_, do_, lse_, del_, cuq_, cukv_, h_off, dq_acc, dk, dv)

    once()
    torch.npu.synchronize()
    best = float("inf")
    for _ in range(reps):
        e0, e1 = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        e0.record()
        once()
        e1.record()
        torch.npu.synchronize()
        best = min(best, e0.elapsed_time(e1))
    it = iterations(lens_q, lens_kv, heads)
    print(f"[TIME] {mode:4s} q={lens_q} kv={lens_kv} H={heads} iters={it} "
          f"T={best:.3f} ms  per-iter={best*1e3/it*28:.3f} us (x28 cores)", flush=True)
    return best


def parity_case(name, lens_q, lens_kv, heads, dtype_name="bfloat16", slab=None):
    """v4 body vs rotated body on identical inputs: dk/dv must be bit-exact.

    ``dq_acc`` is an fp32 store-atomic reduction across key tiles, so its arrival
    order is not fixed (the frozen design already documents this); it is therefore
    compared with the design's fp32 tolerance, and its run-to-run spread within one
    implementation is reported for scale.
    """
    inp = make_inputs(lens_q, lens_kv, heads, dtype_name)
    dev = "npu:0"
    to = lambda t: t.to(dev)  # noqa: E731
    dt = inp["dt"]
    slab = slab or heads
    n_launch = (heads + slab - 1) // slab
    h_offs = torch.tensor([[i * slab] for i in range(n_launch)], dtype=torch.int32, device=dev)
    res = {}
    for mode in ("base", "rot"):
        fn = build(inp["batch"], heads, DQK, DV, dtype_name, slab, mode=mode)
        dq_acc = torch.zeros(inp["T_q"], heads, DQK, dtype=torch.float32, device=dev)
        dk = torch.zeros(inp["T_kv"], heads, DQK, dtype=dt, device=dev)
        dv = torch.zeros(inp["T_kv"], heads, DV, dtype=dt, device=dev)
        for i in range(n_launch):
            fn(to(inp["q"]), to(inp["k"]), to(inp["v"]), to(inp["dout"]), to(inp["lse"]),
               to(inp["delta"]), to(inp["cu_q"]), to(inp["cu_kv"]), h_offs[i], dq_acc, dk, dv)
        torch.npu.synchronize()
        res[mode] = (dk.clone(), dv.clone(), dq_acc.clone())
    b, r = res["base"], res["rot"]
    dk_bit = torch.equal(b[0], r[0])
    dv_bit = torch.equal(b[1], r[1])
    dq_delta = (b[2] - r[2]).abs().max().item()
    dq_ref = b[2].abs().max().item() or 1.0
    ok = dk_bit and dv_bit and (dq_delta / dq_ref) < 1e-3
    print(f"[{'PASS' if ok else 'FAIL'}] parity {name:24s} q={lens_q} kv={lens_kv} H={heads} "
          f"slab={slab}: dk_bit_exact={dk_bit} dv_bit_exact={dv_bit} "
          f"dq_max_abs={dq_delta:.3e} (rel {dq_delta / dq_ref:.2e})", flush=True)
    return ok


def compile_only():
    print("[COMPILE] base (T.Pipelined num_stages=3) ...", flush=True)
    build(2, 8, DQK, DV, "bfloat16", 8, mode="base")
    print("[COMPILE] base OK", flush=True)
    print("[COMPILE] rot (T.serial + explicit 2 versions + deferred GB) ...", flush=True)
    build(2, 8, DQK, DV, "bfloat16", 8, mode="rot")
    print("[COMPILE] rot OK", flush=True)
    return 0


def main():
    if "--compile" in sys.argv:
        return compile_only()
    if "--parity" in sys.argv:
        ok = True
        ok &= parity_case("exact 2 key blocks", [192, 192], [192, 192], 2)
        ok &= parity_case("ragged + tail", [128, 320], [128, 320], 2)
        ok &= parity_case("shift<0 + slab=1", [320, 128], [128, 320], 2, slab=1)
        ok &= parity_case("head slab 3/2", [192, 320], [192, 320], 3, slab=2)
        ok &= parity_case("4k x2, H=8", [4096, 4096], [4096, 4096], 8)
        print("\n[PARITY_SUMMARY]", "ALL_PASS" if ok else "HAS_FAILURE", flush=True)
        return 0 if ok else 1
    if "--time" in sys.argv:
        time_case([4096, 4096], [4096, 4096], 16, mode="base")
        time_case([4096, 4096], [4096, 4096], 16, mode="rot")
        return 0
    ok = True
    ok &= run_case("exact 2 key blocks", [192, 192], [192, 192], 2, mode="rot")
    ok &= run_case("ragged + tail blocks", [128, 320], [128, 320], 2, mode="rot")
    ok &= run_case("len_q > len_kv (shift<0)", [320, 128], [128, 320], 2, mode="rot")
    ok &= run_case("head slab 3/2 (tail slab)", [192, 320], [192, 320], 3, slab=2, mode="rot")
    print("\n[PROTO_SUMMARY]", "ALL_PASS" if ok else "HAS_FAILURE", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
