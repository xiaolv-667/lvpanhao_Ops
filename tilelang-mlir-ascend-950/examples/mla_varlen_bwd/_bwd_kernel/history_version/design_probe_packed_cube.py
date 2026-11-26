#!/usr/bin/env python3
"""Stage-1 compile probe: **packed MLA backward on the dense-cube (C2) core**.

Design doc: DESIGN.md §0.8 / §0.6 R9 / §3.5.4 (revision v4, Cube).

The dense design (`examples/mla_dense_bwd`) is already device-verified with the
explicit-L0 5-GEMM core.  The packed kernel adds four things that are *not* in
the dense body, and this probe isolates exactly those on top of that core:

  * run-time (dynamic) `total_q` / `total_kv` and a **dynamic 1-D grid**
    (`(total_kv // 96 + B) * SLAB`), plus the head-slab launch decomposition
    (`tile = bid // SLAB`, `h = h_off[0] + bid % SLAB`, tail-slab guard);
  * the fused O(B) prefix+decode scan and the `live` predicate;
  * head-major `lse/delta` addressing (`[H, T_q]`);
  * per-request write-back (`R7`: whole-half fast path / per-lane guarded tail).

Everything else is copied from the verified dense body: 5 GEMMs all on the
explicit L0 path, `trans_A` reuse of the single `dsT_s` tile, two-tile merged
epilogue, half-region `dual_copy` 2:1, store-atomic `dq_acc`, `kT` reloaded
into `b_hi` last.

No device is required (compile only):

    cd examples/mla_varlen_bwd/_bwd_kernel
    TMPDIR=/workspace ASCEND_RT_VISIBLE_DEVICES="" \
      /workspace/venv-ascend950/bin/python history_version/design_probe_packed_cube.py
"""

import traceback

import tilelang
import tilelang.ascend.language as T

B, H, DQK, DV = 3, 3, 192, 128   # H=3 / SLAB=2: deliberately non-divisible
BN, BM = 96, 64
HN = BN // 2                      # 48 key rows per AIV (dual_copy 2:1)
SLAB = 2                          # compile-time head slab
DT = "float16"
SCALE = DQK**-0.5


@T.macro
def _decode(cu_kv, tile, req, n0l, cnt, live):
    """Fused running-prefix decode (replaces GroupTiling.cumsum_offsets+decode)."""
    req[0] = T.int32(0)
    n0l[0] = T.int32(0)
    cnt[0] = T.int32(0)
    for g in T.serial(B):
        ntg = (cu_kv[g + 1] - cu_kv[g] + T.int32(BN - 1)) // T.int32(BN)
        if (tile >= cnt[0]) & (tile < cnt[0] + ntg):
            req[0] = g
            n0l[0] = (tile - cnt[0]) * T.int32(BN)
        cnt[0] = cnt[0] + ntg
    live[0] = T.if_then_else(tile < cnt[0], T.int32(1), T.int32(0))


@T.macro
def _body(
    q, k, v, dout, lse, delta, dq_acc, dk, dv,
    k_s, v_s, kT_s, q_s, qT_s, do_s, doT_s, pT_s, dsT_s,
    a_k, a_t, b_lo, b_hi, st_c, dpt_c, dq_c, dk_c, dv_c,
    st_u, dp_u, p16_u, dsT_u, lse_u, dlt_u, tail_u, tailv_u, out_k, out_v,
    h, sid, n0_dyn, lq, lkv, shift, kv_start, q_start, stages,
):
    koff = kv_start + n0_dyn
    T.copy(k[koff : koff + BN, h, :], k_s)
    T.copy(v[koff : koff + BN, h, :], v_s)
    T.copy(k[koff : koff + BN, h, :], kT_s, transpose=True)

    loop_st = T.max(0, T.floordiv(n0_dyn - shift, BM))
    loop_ed = T.max(T.ceildiv(lq, BM), loop_st + 1)   # len_q == 0 must not skip
    for kq in T.Pipelined(loop_st, loop_ed, num_stages=stages):
        m0 = kq * BM
        qoff = q_start + m0
        T.copy(q[qoff : qoff + BM, h, :], q_s)
        T.copy(dout[qoff : qoff + BM, h, :], do_s)
        T.copy(q[qoff : qoff + BM, h, :], qT_s, transpose=True)
        T.copy(dout[qoff : qoff + BM, h, :], doT_s, transpose=True)
        T.copy(lse[h, qoff : qoff + BM], lse_u)          # head-major
        T.copy(delta[h, qoff : qoff + BM], dlt_u)

        # ---- 2 L0 GEMMs, cleared every iteration -------------------------
        T.copy(k_s, a_k)
        T.copy(q_s, b_lo)
        T.gemm(a_k, b_lo, st_c, transpose_B=True, clear_accum=True)
        T.copy(v_s, a_k[0:BN, 0:DV])
        T.copy(do_s, b_lo[0:BM, 0:DV])
        T.gemm(a_k[0:BN, 0:DV], b_lo[0:BM, 0:DV], dpt_c,
               transpose_B=True, clear_accum=True)
        T.dual_copy(st_c, st_u)
        T.dual_copy(dpt_c, dp_u)

        # ---- vector epilogue: exactly two fp16 output tiles ---------------
        fast = (n0_dyn + BN <= lkv) & (m0 + BM <= lq) & (n0_dyn + BN - 1 <= m0 + shift)
        if fast:
            with T.SimtVF(threads=128):
                for i, j in T.Parallel(HN, BM):
                    p = T.exp(st_u[i, j] * SCALE - lse_u[j])
                    p16_u[i, j] = T.cast(p, DT)
                    dsT_u[i, j] = T.cast(p * (dp_u[i, j] - dlt_u[j]), DT)
        else:
            with T.SimtVF(threads=128):
                for i, j in T.Parallel(HN, BM):
                    gi = n0_dyn + T.cast(sid, "int32") * HN + i
                    gj = m0 + j
                    vis = (gi < lkv) & (gj < lq) & (gi <= gj + shift)
                    p = T.exp(st_u[i, j] * SCALE - lse_u[j])
                    pv = T.if_then_else(vis, p, T.cast(0, "float32"))
                    p16_u[i, j] = T.cast(pv, DT)
                    dsT_u[i, j] = T.cast(pv * (dp_u[i, j] - dlt_u[j]), DT)

        T.dual_copy(p16_u, pT_s)
        T.dual_copy(dsT_u, dsT_s)
        # ---- 3 L0 GEMMs: dV/dK accumulate, dQ reuses dsT via trans_A -------
        T.copy(pT_s, a_k[0:BN, 0:BM])
        T.copy(doT_s, b_hi[0:DV, 0:BM])
        T.gemm(a_k[0:BN, 0:BM], b_hi[0:DV, 0:BM], dv_c,
               transpose_B=True, clear_accum=(kq == loop_st))
        T.copy(dsT_s, a_k[0:BN, 0:BM])
        T.copy(qT_s, b_hi[0:DQK, 0:BM])
        T.gemm(a_k[0:BN, 0:BM], b_hi[0:DQK, 0:BM], dk_c,
               transpose_B=True, clear_accum=(kq == loop_st))
        T.copy(dsT_s, a_t)                 # L1 -> L0A transposing copy (MTE1)
        T.copy(kT_s, b_hi)                 # kT reloaded into b_hi LAST
        T.gemm(a_t, b_hi, dq_c, transpose_A=True, transpose_B=True, clear_accum=True)
        # ---- dQ: whole-tile store-atomic; spilled rows are exactly 0 (A3) --
        T.set_atomic("add", "float32")
        T.copy(dq_c, dq_acc[qoff : qoff + BM, h, :])
        T.set_atomic_none()

    # ---- block tail: dk = scale*cast(dk_c), dv = cast(dv_c) --------------
    T.dual_copy(dk_c, tail_u)
    T.dual_copy(dv_c, tailv_u)
    with T.SimtVF(threads=128):
        for i, d in T.Parallel(HN, DQK):
            out_k[i, d] = T.cast(tail_u[i, d] * SCALE, DT)
        for i, d in T.Parallel(HN, DV):
            out_v[i, d] = T.cast(tailv_u[i, d], DT)
    # R7: per-request isolation (packed has no 4-D batch view)
    own = koff + T.cast(sid, "int32") * HN
    kv_end = kv_start + lkv
    if own + HN <= kv_end:
        T.copy(out_k, dk[own : own + HN, h, :])
        T.copy(out_v, dv[own : own + HN, h, :])
    else:
        with T.SimtVF(threads=128):
            for i, d in T.Parallel(HN, DQK):
                if own + i < kv_end:
                    dk[own + i, h, d] = out_k[i, d]
            for i, d in T.Parallel(HN, DV):
                if own + i < kv_end:
                    dv[own + i, h, d] = out_v[i, d]


def _alloc():
    d = {}
    d["k_s"] = T.alloc_l1((BN, DQK), DT)
    d["v_s"] = T.alloc_l1((BN, DV), DT)
    d["kT_s"] = T.alloc_l1((DQK, BN), DT)
    d["q_s"] = T.alloc_l1((BM, DQK), DT)
    d["qT_s"] = T.alloc_l1((DQK, BM), DT)
    d["do_s"] = T.alloc_l1((BM, DV), DT)
    d["doT_s"] = T.alloc_l1((DV, BM), DT)
    d["pT_s"] = T.alloc_l1((BN, BM), DT)
    d["dsT_s"] = T.alloc_l1((BN, BM), DT)
    d["a_k"] = T.alloc_l0a((BN, DQK), DT)
    d["a_t"] = T.alloc_l0a((BN, BM), DT)
    d["b_lo"] = T.alloc_l0b((BM, DQK), DT)
    d["b_hi"] = T.alloc_l0b((DQK, BN), DT)
    d["st_c"] = T.alloc_l0c((BN, BM), "float32")
    d["dpt_c"] = T.alloc_l0c((BN, BM), "float32")
    d["dq_c"] = T.alloc_l0c((BM, DQK), "float32")
    d["dk_c"] = T.alloc_l0c((BN, DQK), "float32")
    d["dv_c"] = T.alloc_l0c((BN, DV), "float32")
    d["st_u"] = T.alloc_shared((HN, BM), "float32")
    d["dp_u"] = T.alloc_shared((HN, BM), "float32")
    d["p16_u"] = T.alloc_shared((HN, BM), DT)
    d["dsT_u"] = T.alloc_shared((HN, BM), DT)
    d["lse_u"] = T.alloc_shared((BM,), "float32")
    d["dlt_u"] = T.alloc_shared((BM,), "float32")
    d["tail_u"] = T.alloc_shared((HN, DQK), "float32")
    d["tailv_u"] = T.alloc_shared((HN, DV), "float32")
    d["out_k"] = T.alloc_shared((HN, DQK), DT)
    d["out_v"] = T.alloc_shared((HN, DV), DT)
    return d


def _args(d):
    return (
        d["k_s"], d["v_s"], d["kT_s"], d["q_s"], d["qT_s"], d["do_s"], d["doT_s"],
        d["pT_s"], d["dsT_s"], d["a_k"], d["a_t"], d["b_lo"], d["b_hi"],
        d["st_c"], d["dpt_c"], d["dq_c"], d["dk_c"], d["dv_c"],
        d["st_u"], d["dp_u"], d["p16_u"], d["dsT_u"], d["lse_u"], d["dlt_u"],
        d["tail_u"], d["tailv_u"], d["out_k"], d["out_v"],
    )


def build(stages=2, slab=SLAB, no_slab=False):
    total_q = T.dynamic("total_q")
    total_kv = T.dynamic("total_kv")
    ntiles = T.floordiv(total_kv, BN) + B
    grid = ntiles * (H if no_slab else slab)

    @T.prim_func
    def main(
        q: T.Tensor((total_q, H, DQK), DT),
        k: T.Tensor((total_kv, H, DQK), DT),
        v: T.Tensor((total_kv, H, DV), DT),
        dout: T.Tensor((total_q, H, DV), DT),
        lse: T.Tensor((H, total_q), "float32"),
        delta: T.Tensor((H, total_q), "float32"),
        cu_q: T.Tensor((B + 1,), "int32"),
        cu_kv: T.Tensor((B + 1,), "int32"),
        h_off: T.Tensor((1,), "int32"),
        dq_acc: T.Tensor((total_q, H, DQK), "float32"),
        dk: T.Tensor((total_kv, H, DQK), DT),
        dv: T.Tensor((total_kv, H, DV), DT),
    ):
        with T.Kernel(grid) as bid:
            d = _alloc()
            sid = T.call_extern("int64", "asc_get_sub_block_id")
            if no_slab:
                tile = bid // H
                h = bid % H
            else:
                tile = bid // slab                    # static divisor
                h = h_off[0] + bid % slab             # run-time origin + slot
            if h < H:                                 # tail-slab guard
                req = T.alloc_local([1], "int32")
                n0l = T.alloc_local([1], "int32")
                cnt = T.alloc_local([1], "int32")
                live = T.alloc_local([1], "int32")
                _decode(cu_kv, tile, req, n0l, cnt, live)
                if live[0] == T.int32(1):
                    q_start = cu_q[req[0]]
                    kv_start = cu_kv[req[0]]
                    lq = cu_q[req[0] + 1] - q_start
                    lkv = cu_kv[req[0] + 1] - kv_start
                    _body(
                        q, k, v, dout, lse, delta, dq_acc, dk, dv, *_args(d),
                        h, sid, n0l[0], lq, lkv, lkv - lq, kv_start, q_start, stages,
                    )

    return main


PROBES = [
    ("P7 packed_cube_stages2", lambda: build(stages=2)),
    ("P8 packed_cube_stages1", lambda: build(stages=1)),
    ("P9 packed_cube_no_slab", lambda: build(stages=2, no_slab=True)),
]

MARKS = (
    "asc_mmad",
    "asc_copy_l12l0a",
    "asc_copy_l12l0b",
    "asc_copy_l12l0a_transpose",
    "asc_copy_l0c2ub",
    "ascend_nd2nz_scatter",
    "asc_copy_l0c2gm",
    "asc_set_atomic_add_float",
    "__mix__",
    "asc_get_sub_block_id",
    "asc_fill_l1",
    "asc_copy_gm2l1_nd2nz",
    "asc_copy_gm2l1_dn2nz",
    "asc_copy_ub2gm",
)


def main():
    results = []
    for name, build_fn in PROBES:
        try:
            fn = build_fn()
            src = tilelang.compile(fn, target="ascend").get_kernel_source()
            counts = {kw: src.count(kw) for kw in MARKS if kw in src}
            results.append((name, "COMPILE_OK", f"{len(src)} B; marks={counts}"))
            with open(f"/tmp/opencode/probe_{name.split()[0]}_device.asc", "w") as fh:
                fh.write(src)
        except Exception as exc:  # noqa: BLE001 - a probe must report, not abort
            last = traceback.format_exc().strip().splitlines()[-1]
            results.append(
                (name, "COMPILE_FAIL", f"{type(exc).__name__}: {str(exc)[:400]} || {last[:200]}")
            )
    print()
    for name, status, detail in results:
        print(f"[{status}] {name} -- {detail}", flush=True)
    ok = sum(1 for _, s, _ in results if s == "COMPILE_OK")
    print(f"\n[PROBE_SUMMARY] {ok}/{len(results)} COMPILE_OK", flush=True)


if __name__ == "__main__":
    main()
