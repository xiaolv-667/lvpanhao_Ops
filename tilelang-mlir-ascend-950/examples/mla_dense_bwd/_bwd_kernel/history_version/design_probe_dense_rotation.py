#!/usr/bin/env python3
"""Stage-1 probe (revision v4 = "A5 rotation on the dense body").

Question this probe answers with compile/device evidence, not by argument:

  1. Can the *dense* query loop (4-D batch views, static ``S_q/S_kv``, no head
     slab, no ``cu_seqlens`` decode) be hand rotated so that
     ``GB(k) = {dV, dK, dQ}`` runs one iteration late -- beside ``EP(k+1)``
     instead of behind it -- using ``T.serial`` + a 2-tile static unroll and
     statically named buffer versions (no ``T.Pipelined`` versioning)?
  2. Does that rotation keep L0C / L0A / L0B / UB unchanged (only the six
     per-iteration L1 fill buffers + ``pT_s``/``dsT_s`` get a second copy)?
  3. Is it numerically equal to the frozen body (``dk``/``dv`` bit-exact, ``dq``
     within the fp32 store-atomic tolerance) on the shapes the dense operator
     actually runs, *including* ``B=2`` and the non-causal mask?
  4. What is its per-iteration price relative to the frozen
     ``T.Pipelined(num_stages=2)`` body on a dense production-like geometry?

Modes
  base : the frozen body of DESIGN v3 (``T.Pipelined(num_stages=2)``,
         single-version per-iteration buffers).
  rot  : the candidate. ``T.serial(n_half)`` with ``kqA = loop_st + 2i``,
         ``kqB = kqA + 1``; each half body is
             F(kq) -> GA(kq) -> EP(kq) -> GB(kq-1)   (kq-1 reads the other version)
         and a drain issues ``GB(loop_ed-1)`` with the version selected by the
         parity of ``n_it``.

Run (device, single card, cwd = operator dir):
    TMPDIR=/workspace /workspace/venv-ascend950/bin/python \
        history_version/design_probe_dense_rotation.py --compile
    TMPDIR=/workspace /workspace/venv-ascend950/bin/python \
        history_version/design_probe_dense_rotation.py --numeric
    TMPDIR=/workspace /workspace/venv-ascend950/bin/python \
        history_version/design_probe_dense_rotation.py --parity
    TMPDIR=/workspace /workspace/venv-ascend950/bin/python \
        history_version/design_probe_dense_rotation.py --time
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
# fp32 per-request reference (same arithmetic as DESIGN.md 8.1 golden)
# ---------------------------------------------------------------------------
def reference(q, k, v, dout, s_q, s_kv, heads, dtype, is_causal=True):
    """q:(B,S_q,H,DQK) k,v:(B,S_kv,H,*), bottom-right causal, fp32内部.

    ``p`` and ``ds`` are rounded to ``dtype`` *before* the three gradient GEMMs
    because that is what the kernel does (the L1 tiles are the input dtype).
    ``dq`` returned is ``dq_acc`` (= dq / SCALE), the kernel's actual output.
    """
    bsz = q.shape[0]
    shift = s_kv - s_q
    dk = torch.zeros(bsz, s_kv, heads, DQK, dtype=torch.float32)
    dv = torch.zeros(bsz, s_kv, heads, DV, dtype=torch.float32)
    dq = torch.zeros(bsz, s_q, heads, DQK, dtype=torch.float32)
    lse = torch.full((bsz, heads, s_q), float("-inf"))
    delta = torch.zeros(bsz, heads, s_q)
    m = torch.arange(s_q)[:, None]
    n = torch.arange(s_kv)[None, :]
    vis = (n <= m + shift) if is_causal else torch.ones(s_q, s_kv, dtype=torch.bool)
    for b in range(bsz):
        for h in range(heads):
            qb = q[b, :, h].float()
            kb = k[b, :, h].float()
            vb = v[b, :, h].float()
            dob = dout[b, :, h].float()
            s = (qb @ kb.T) * SCALE
            sm = torch.where(vis, s, torch.full_like(s, float("-inf")))
            row_lse = torch.logsumexp(sm, dim=1)
            lse[b, h] = row_lse
            p = torch.where(vis, torch.exp(s - row_lse[:, None]), torch.zeros(()))
            dp = dob @ vb.T
            dlt = (p * dp).sum(1)
            delta[b, h] = dlt
            ds = p * (dp - dlt[:, None])
            p16 = p.to(dtype).float()
            ds16 = ds.to(dtype).float()
            dv[b, :, h] = p16.T @ dob
            dk[b, :, h] = SCALE * (ds16.T @ qb)
            dq[b, :, h] = ds16 @ kb
    return dk.to(dtype), dv.to(dtype), dq, lse.contiguous(), delta.contiguous()


# ---------------------------------------------------------------------------
def build(batch, s_q, s_kv, heads, dim_qk, dim_v, dt, mode="rot",
          stages=2, is_causal=True):
    nt = -(-s_kv // BN)                     # key tiles per request (static)
    shift = s_kv - s_q                      # compile-time constant (dense)
    grid = heads * batch * nt

    @T.macro
    def fills(q, dout, lse, delta, q_s, do_s, qT_s, doT_s, lse_u, dlt_u, b, h, m0):
        T.copy(q[b, m0:m0 + BM, h, :], q_s)
        T.copy(dout[b, m0:m0 + BM, h, :], do_s)
        T.copy(q[b, m0:m0 + BM, h, :], qT_s, transpose=True)
        T.copy(dout[b, m0:m0 + BM, h, :], doT_s, transpose=True)
        T.copy(lse[b, h, m0:m0 + BM], lse_u)
        T.copy(delta[b, h, m0:m0 + BM], dlt_u)

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
                 n0, m0):
        # fast path: block provably fully visible (A2 of DESIGN 1.6.1)
        fast = (n0 + BN <= s_kv) & (m0 + BM <= s_q)
        if is_causal:
            fast = fast & (n0 + BN - 1 <= m0 + shift)
        if fast:
            with T.SimtVF(threads=128):
                for i, j in T.Parallel(HN, BM):
                    p = T.exp(st_u[i, j] * SCALE - lse_u[j])
                    p16_u[i, j] = T.cast(p, dt)
                    dsT_u[i, j] = T.cast(p * (dp_u[i, j] - dlt_u[j]), dt)
        else:
            with T.SimtVF(threads=128):
                for i, j in T.Parallel(HN, BM):
                    gi = n0 + T.cast(sid, "int32") * HN + i
                    gj = m0 + j
                    vis = (gi < s_kv) & (gj < s_q)
                    if is_causal:
                        vis = vis & (gi <= gj + shift)
                    p = T.exp(st_u[i, j] * SCALE - lse_u[j])
                    pv = T.if_then_else(vis, p, T.cast(0, "float32"))
                    p16_u[i, j] = T.cast(pv, dt)
                    dsT_u[i, j] = T.cast(pv * (dp_u[i, j] - dlt_u[j]), dt)
        T.dual_copy(p16_u, pT_dst)
        T.dual_copy(dsT_u, dsT_dst)

    @T.macro
    def gb(pT_p, dsT_p, doT_p, qT_p, kT_s, a_k, a_t, b_hi,
           dq_c, dk_c, dv_c, dq_acc, b, m0, h, clear_prev):
        T.copy(pT_p, a_k[0:BN, 0:BM])
        T.copy(doT_p, b_hi[0:dim_v, 0:BM])
        T.gemm(a_k[0:BN, 0:BM], b_hi[0:dim_v, 0:BM], dv_c,
               transpose_B=True, clear_accum=clear_prev)
        T.copy(dsT_p, a_k[0:BN, 0:BM])
        T.copy(qT_p, b_hi[0:dim_qk, 0:BM])
        T.gemm(a_k[0:BN, 0:BM], b_hi[0:dim_qk, 0:BM], dk_c,
               transpose_B=True, clear_accum=clear_prev)
        T.copy(dsT_p, a_t)
        # kT must be reloaded into b_hi AFTER doT/qT (DESIGN 9.2)
        T.copy(kT_s, b_hi)
        T.gemm(a_t, b_hi, dq_c, transpose_A=True, transpose_B=True, clear_accum=True)
        T.set_atomic("add", "float32")
        T.copy(dq_c, dq_acc[b, m0:m0 + BM, h, :])
        T.set_atomic_none()

    @T.prim_func
    def main(
        q: T.Tensor((batch, s_q, heads, dim_qk), dt),
        k: T.Tensor((batch, s_kv, heads, dim_qk), dt),
        v: T.Tensor((batch, s_kv, heads, dim_v), dt),
        dout: T.Tensor((batch, s_q, heads, dim_v), dt),
        lse: T.Tensor((batch, heads, s_q), "float32"),
        delta: T.Tensor((batch, heads, s_q), "float32"),
        dq_acc: T.Tensor((batch, s_q, heads, dim_qk), "float32"),
        dk: T.Tensor((batch, s_kv, heads, dim_qk), dt),
        dv: T.Tensor((batch, s_kv, heads, dim_v), dt),
    ):
        with T.Kernel(grid) as bid:
            h = bid // (batch * nt)
            tile = bid % (batch * nt)
            b = tile // nt
            n0 = (tile % nt) * BN
            sid = T.call_extern("int64", "asc_get_sub_block_id")
            # ---- persistent L1 ----
            k_s = T.alloc_l1((BN, dim_qk), dt)
            v_s = T.alloc_l1((BN, dim_v), dt)
            kT_s = T.alloc_l1((dim_qk, BN), dt)
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
            # ---- per-iteration L1 + lse/delta: version A / B ----
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

            T.copy(k[b, n0:n0 + BN, h, :], k_s)
            T.copy(v[b, n0:n0 + BN, h, :], v_s)
            T.copy(k[b, n0:n0 + BN, h, :], kT_s, transpose=True)

            loop_st = T.max(0, T.floordiv(n0 - shift, BM)) if is_causal else 0
            loop_ed = -(-s_q // BM)

            if mode == "base":
                for kq in T.Pipelined(loop_st, loop_ed, num_stages=stages):
                    m0 = kq * BM
                    fills(q, dout, lse, delta, q_s0, do_s0, qT_s0, doT_s0,
                          lse_u0, dlt_u0, b, h, m0)
                    ga(k_s, v_s, q_s0, do_s0, a_k, b_lo, st_c, dpt_c, st_u, dp_u)
                    epilogue(sid, st_u, dp_u, p16_u, dsT_u, lse_u0, dlt_u0,
                             pT_s0, dsT_s0, n0, m0)
                    gb(pT_s0, dsT_s0, doT_s0, qT_s0, kT_s, a_k, a_t, b_hi,
                       dq_c, dk_c, dv_c, dq_acc, b, m0, h, kq == loop_st)
            else:
                n_it = loop_ed - loop_st
                n_half = (n_it + 1) // 2
                for i in T.serial(n_half):
                    kqA = loop_st + i * 2
                    fills(q, dout, lse, delta, q_s0, do_s0, qT_s0, doT_s0,
                          lse_u0, dlt_u0, b, h, kqA * BM)
                    ga(k_s, v_s, q_s0, do_s0, a_k, b_lo, st_c, dpt_c, st_u, dp_u)
                    epilogue(sid, st_u, dp_u, p16_u, dsT_u, lse_u0, dlt_u0,
                             pT_s0, dsT_s0, n0, kqA * BM)
                    if kqA > loop_st:
                        gb(pT_s1, dsT_s1, doT_s1, qT_s1, kT_s, a_k, a_t, b_hi,
                           dq_c, dk_c, dv_c, dq_acc, b, (kqA - 1) * BM, h,
                           kqA - 1 == loop_st)
                    kqB = kqA + 1
                    if kqB < loop_ed:
                        fills(q, dout, lse, delta, q_s1, do_s1, qT_s1, doT_s1,
                              lse_u1, dlt_u1, b, h, kqB * BM)
                        ga(k_s, v_s, q_s1, do_s1, a_k, b_lo, st_c, dpt_c, st_u, dp_u)
                        epilogue(sid, st_u, dp_u, p16_u, dsT_u, lse_u1, dlt_u1,
                                 pT_s1, dsT_s1, n0, kqB * BM)
                        gb(pT_s0, dsT_s0, doT_s0, qT_s0, kT_s, a_k, a_t, b_hi,
                           dq_c, dk_c, dv_c, dq_acc, b, (kqB - 1) * BM, h,
                           kqB - 1 == loop_st)
                # drain: body(loop_ed) is virtual -> GB(last) always pending.
                last = loop_ed - 1
                if (n_it % 2) == 1:
                    gb(pT_s0, dsT_s0, doT_s0, qT_s0, kT_s, a_k, a_t, b_hi,
                       dq_c, dk_c, dv_c, dq_acc, b, last * BM, h, last == loop_st)
                else:
                    gb(pT_s1, dsT_s1, doT_s1, qT_s1, kT_s, a_k, a_t, b_hi,
                       dq_c, dk_c, dv_c, dq_acc, b, last * BM, h, last == loop_st)

            # ---- block tail: dk = scale * cast(dk_c), dv = cast(dv_c) ----
            T.dual_copy(dk_c, tail_u)
            T.dual_copy(dv_c, tailv_u)
            with T.SimtVF(threads=128):
                for i, d in T.Parallel(HN, dim_qk):
                    out_k[i, d] = T.cast(tail_u[i, d] * SCALE, dt)
                for i, d in T.Parallel(HN, dim_v):
                    out_v[i, d] = T.cast(tailv_u[i, d], dt)
            # 4-D batch view -> the OOB clamp row bound is this request's s_kv
            kown = n0 + T.cast(sid, "int32") * HN
            T.copy(out_k, dk[b, kown:kown + HN, h, :])
            T.copy(out_v, dv[b, kown:kown + HN, h, :])

    return tilelang.compile(main, target="ascend")


# ---------------------------------------------------------------------------
def make_inputs(batch, s_q, s_kv, heads, dtype_name, seed=0, is_causal=True):
    dt = getattr(torch, dtype_name)
    g = torch.Generator().manual_seed(seed)
    q = (torch.randn(batch, s_q, heads, DQK, generator=g) * 0.5).to(dt)
    k = (torch.randn(batch, s_kv, heads, DQK, generator=g) * 0.5).to(dt)
    v = (torch.randn(batch, s_kv, heads, DV, generator=g) * 0.5).to(dt)
    dout = (torch.randn(batch, s_q, heads, DV, generator=g) * 0.5).to(dt)
    ref_dk, ref_dv, ref_dq, lse, delta = reference(q, k, v, dout, s_q, s_kv, heads,
                                                   dt, is_causal=is_causal)
    return dict(dt=dt, batch=batch, s_q=s_q, s_kv=s_kv, heads=heads, q=q, k=k, v=v,
                dout=dout, lse=lse, delta=delta, ref_dk=ref_dk, ref_dv=ref_dv,
                ref_dq=ref_dq)


def run_case(name, batch, s_q, s_kv, heads, dtype_name="bfloat16",
             mode="rot", is_causal=True):
    inp = make_inputs(batch, s_q, s_kv, heads, dtype_name, is_causal=is_causal)
    dev = "npu:0"
    to = lambda t: t.to(dev)  # noqa: E731
    dt = inp["dt"]
    dq_acc = torch.zeros(batch, s_q, heads, DQK, dtype=torch.float32, device=dev)
    dk = torch.zeros(batch, s_kv, heads, DQK, dtype=dt, device=dev)
    dv = torch.zeros(batch, s_kv, heads, DV, dtype=dt, device=dev)
    fn = build(batch, s_q, s_kv, heads, DQK, DV, dtype_name, mode=mode,
               is_causal=is_causal)
    fn(to(inp["q"]), to(inp["k"]), to(inp["v"]), to(inp["dout"]), to(inp["lse"]),
       to(inp["delta"]), dq_acc, dk, dv)
    torch.npu.synchronize()

    got = {"dk": dk.cpu().float(), "dv": dv.cpu().float(), "dq": dq_acc.cpu()}
    ref = {"dk": inp["ref_dk"].float(), "dv": inp["ref_dv"].float(), "dq": inp["ref_dq"]}
    out, worst = {}, 0.0
    for tag in ("dk", "dv", "dq"):
        denom = ref[tag].abs().max().item() or 1.0
        out[tag] = (got[tag] - ref[tag]).abs().max().item() / denom
    for bb in range(batch):
        for tag in ("dk", "dv", "dq"):
            rr = ref[tag][bb]
            worst = max(worst, (got[tag][bb] - rr).abs().max().item()
                        / (rr.abs().max().item() or 1.0))
    ok = all(v < 3e-2 for v in out.values()) and worst < 3e-2
    print(f"[{'PASS' if ok else 'FAIL'}] {mode:4s} {name:30s} B={batch} S={s_q}/{s_kv} "
          f"H={heads} causal={is_causal} {dtype_name}: dk={out['dk']:.3e} "
          f"dv={out['dv']:.3e} dq={out['dq']:.3e} per_batch_worst={worst:.3e}",
          flush=True)
    return ok


def iterations(batch, s_q, s_kv, heads, bn=BN, bm=BM, is_causal=True):
    shift = s_kv - s_q
    n_q = -(-s_q // bm)
    tot = 0
    for kn in range(-(-s_kv // bn)):
        n0 = kn * bn
        st = max(0, (n0 - shift) // bm) if is_causal else 0
        tot += max(0, max(n_q, st + 1) - st)
    return tot * heads * batch


def time_case(batch, s_q, s_kv, heads, mode="rot", dtype_name="bfloat16", reps=3,
              is_causal=True):
    inp = make_inputs(batch, s_q, s_kv, heads, dtype_name, is_causal=is_causal)
    dev = "npu:0"
    to = lambda t: t.to(dev)  # noqa: E731
    dt = inp["dt"]
    dq_acc = torch.zeros(batch, s_q, heads, DQK, dtype=torch.float32, device=dev)
    dk = torch.zeros(batch, s_kv, heads, DQK, dtype=dt, device=dev)
    dv = torch.zeros(batch, s_kv, heads, DV, dtype=dt, device=dev)
    fn = build(batch, s_q, s_kv, heads, DQK, DV, dtype_name, mode=mode,
               is_causal=is_causal)
    q_, k_, v_, do_ = to(inp["q"]), to(inp["k"]), to(inp["v"]), to(inp["dout"])
    lse_, del_ = to(inp["lse"]), to(inp["delta"])

    def once():
        fn(q_, k_, v_, do_, lse_, del_, dq_acc, dk, dv)

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
    it = iterations(batch, s_q, s_kv, heads, is_causal=is_causal)
    print(f"[TIME] {mode:4s} B={batch} S={s_q} H={heads} iters={it} "
          f"T={best:.3f} ms  per-iter={best*1e3/it*28:.3f} us (x28 cores)", flush=True)
    return best


def parity_case(name, batch, s_q, s_kv, heads, dtype_name="bfloat16", is_causal=True):
    """frozen body vs rotated body on identical inputs: dk/dv must be bit-exact.

    ``dq_acc`` is an fp32 store-atomic reduction across key tiles, so its arrival
    order is not fixed (DESIGN 8.2 already documents this); it is compared with
    the design's fp32 tolerance and its run-to-run spread is reported for scale.
    """
    inp = make_inputs(batch, s_q, s_kv, heads, dtype_name, is_causal=is_causal)
    dev = "npu:0"
    to = lambda t: t.to(dev)  # noqa: E731
    dt = inp["dt"]
    res = {}
    for mode in ("base", "rot"):
        fn = build(batch, s_q, s_kv, heads, DQK, DV, dtype_name, mode=mode,
                   is_causal=is_causal)
        dq_acc = torch.zeros(batch, s_q, heads, DQK, dtype=torch.float32, device=dev)
        dk = torch.zeros(batch, s_kv, heads, DQK, dtype=dt, device=dev)
        dv = torch.zeros(batch, s_kv, heads, DV, dtype=dt, device=dev)
        fn(to(inp["q"]), to(inp["k"]), to(inp["v"]), to(inp["dout"]), to(inp["lse"]),
           to(inp["delta"]), dq_acc, dk, dv)
        torch.npu.synchronize()
        res[mode] = (dk.clone(), dv.clone(), dq_acc.clone())
    b, r = res["base"], res["rot"]
    dk_bit = torch.equal(b[0], r[0])
    dv_bit = torch.equal(b[1], r[1])
    dq_delta = (b[2] - r[2]).abs().max().item()
    dq_ref = b[2].abs().max().item() or 1.0
    ok = dk_bit and dv_bit and (dq_delta / dq_ref) < 1e-3
    print(f"[{'PASS' if ok else 'FAIL'}] parity {name:28s} B={batch} S={s_q}/{s_kv} "
          f"H={heads} causal={is_causal}: dk_bit_exact={dk_bit} dv_bit_exact={dv_bit} "
          f"dq_max_abs={dq_delta:.3e} (rel {dq_delta / dq_ref:.2e})", flush=True)
    return ok


def compile_only():
    print("[COMPILE] base (T.Pipelined num_stages=2) ...", flush=True)
    build(2, 512, 512, 4, DQK, DV, "bfloat16", mode="base")
    print("[COMPILE] base OK", flush=True)
    print("[COMPILE] rot (T.serial + explicit 2 versions + deferred GB) ...", flush=True)
    build(2, 512, 512, 4, DQK, DV, "bfloat16", mode="rot")
    print("[COMPILE] rot OK", flush=True)
    return 0


def main():
    if "--compile" in sys.argv:
        return compile_only()
    if "--parity" in sys.argv:
        ok = True
        ok &= parity_case("1 key block", 1, 96, 96, 1)
        ok &= parity_case("2 key blocks (n_it=2)", 1, 128, 128, 1)
        ok &= parity_case("tail blocks B=1", 1, 320, 320, 2)
        ok &= parity_case("4-D B=2 + tail", 2, 512, 512, 4)
        ok &= parity_case("4-D B=2 non-causal", 2, 512, 512, 4, is_causal=False)
        print("\n[PARITY_SUMMARY]", "ALL_PASS" if ok else "HAS_FAILURE", flush=True)
        return 0 if ok else 1
    if "--time" in sys.argv:
        time_case(2, 4096, 4096, 16, mode="base")
        time_case(2, 4096, 4096, 16, mode="rot")
        time_case(2, 8192, 8192, 8, mode="base")
        time_case(2, 8192, 8192, 8, mode="rot")
        return 0
    ok = True
    ok &= run_case("1 key block / 1 iter", 1, 96, 96, 1)
    ok &= run_case("2 key blocks / n_it=2", 1, 128, 128, 1)
    ok &= run_case("tail blocks", 1, 320, 320, 2)
    ok &= run_case("4-D B=2 + tail", 2, 512, 512, 4)
    ok &= run_case("4-D B=2 non-causal", 2, 512, 512, 4, is_causal=False)
    print("\n[PROTO_SUMMARY]", "ALL_PASS" if ok else "HAS_FAILURE", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
