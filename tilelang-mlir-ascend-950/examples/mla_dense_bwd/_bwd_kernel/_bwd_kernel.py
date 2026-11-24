# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""_bwd_kernel (ascend): MLA prefill backward main pass (MLADenseBwdOp).

Given the forward tensors ``q, k, v`` and ``dout`` plus the per-query row
statistics ``lse`` (natural log) and ``delta = rowsum(o * dout)``, this launch
recomputes the attention probabilities and emits

    dV[n, :] = sum_m P[m, n] dout[m, :]
    dK[n, :] = scale * sum_m dS[m, n] q[m, :]      (fp32 accumulate, cast on write)
    dQ[m, :] = sum_n dS[m, n] k[n, :]              (fp32 in-place accumulate in
                                                    ``dq_acc``; ``_dq_land_kernel``
                                                    applies ``scale`` and the cast)

with ``P = exp(scale * q.k - lse)``, ``dP = dout.v``, ``dS = P * (dP - delta)``
and FA3 bottom-right causal alignment (query ``m`` sees key ``n <= m + shift``,
``shift = S_kv - S_q``).

Structure (key stationary, one key block per core, DESIGN.md 0.6 R1-R5):

* one-dimensional ``T.Kernel(grid)``, ``grid = H * B * ceil(S_kv / 64)``;
* ``k/v`` (and the transposed ``kT``) stay resident in L1 for the whole query
  loop; ``dk/dv`` accumulate in L0C across the loop;
* five NT GEMMs (``C = A(M,K) . B(N,K)^T``) because the L1 GEMM path only
  supports ``trans_A=False, trans_B=True``, whole-allocation extents only;
* the ``P``/``dS`` epilogue runs on the vector side from the L0C accumulators.

Two runtime facts measured on this toolchain shape the vector side and differ
from the frozen DESIGN.md wording (see RETROSPECTIVE.md stage 3):

1. ``T.copy(L0C -> UB)`` (``dual_dst_mode=0``) only lands in one AIV's buffer;
   the whole-tile epilogue of DESIGN.md 3.3 therefore reads stale data on the
   other AIV (probe P1: 99.2% mismatch).  The documented fallback is used
   instead: every L0C<->UB crossing is a ``T.dual_copy`` 2:1 split, so each AIV
   owns half of the tile (``_HN = 32`` key rows) in its own UB bank and the L1
   tiles stay whole (``pT_s/dsT_s/ds_s`` are still ``(64, 64)`` operands for the
   following GEMMs).
2. The half tile makes the key index AIV dependent, which the mask needs, so
   ``sid = asc_get_sub_block_id()`` is bound once outside the SIMT region and
   captured into every ``T.SimtVF`` (``T.Vector(vector=2) as sid`` remains
   uncompilable, ``InternalError ... dup sid``).

Content layout:

1. ``golden_bwd_kernel`` -- independent fp64 torch reference (materialises P).
2. ``forward_stats`` -- the ``lse`` / ``delta`` the upstream launches provide.
3. ``_geometry`` / ``_bwd_kernel`` -- the frozen kernel factory.
4. ``run_case`` -- launch on the NPU and compare with the golden.
5. ``run_L0`` / ``run_L1`` / ``run_L2`` / ``run_boundary`` / ``main``.
"""

import argparse
import functools
import math
import time

import tilelang
import tilelang.ascend.language as T
import torch
import torch_npu  # noqa: F401  -- registers torch.npu

_IN_DTYPES = ("float16", "bfloat16")
_TORCH_DTYPE = {"float16": torch.float16, "bfloat16": torch.bfloat16}

_BN = 64  # key rows of one block (DESIGN.md 5.2)
_BM = 64  # query rows of one block
_HN = _BN // 2  # key rows per AIV (dual_copy 2:1 split)
_OOB_PAD = _BN  # sentinel rows appended to every output allocation
_ATOL = 5e-3  # DESIGN.md 8.2 (max(5e-3, reference_tolerance))
_RTOL = {"float16": 1e-3, "bfloat16": 1.6e-2}
_DEFAULT_DTYPE = "bfloat16"
_SENTINEL = -12345.0


# ---------------------------------------------------------------------------
# 1. Golden: independent fp64 reference (no re-implementation of the tiling)
# ---------------------------------------------------------------------------
def golden_bwd_kernel(q, k, v, dout, scale=None, is_causal=True, dtype=None, slab_heads=4):
    """Materialised fp64 reference of the backward pass (bottom-right causal).

    ``q`` ``(B, S_q, H, D_qk)``; ``k`` ``(B, S_kv, H, D_qk)``; ``v, dout``
    ``(B, S, H, D_v)``.  The probabilities and ``delta`` are recomputed here
    from ``q, k, v, dout`` -- the ``lse`` / ``delta`` tensors the kernel consumes
    are never read, so this reference shares no assumption with the tiling,
    the L0C accumulators, the dual split or the atomic ``dq`` accumulation.
    Returns ``(dq, dk, dv)`` rounded once to ``dtype`` (same contract as the
    source ``ref_program``) plus ``dq_acc_true = dq / scale``, the float32
    intermediate this kernel writes into ``dq_acc``.
    """
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    dtype = dtype or q.dtype
    batch, seq_q, heads, dim_qk = q.shape
    seq_kv, dim_v = k.shape[1], v.shape[-1]
    shift = seq_kv - seq_q
    qf, kf, vf, dof = (t.detach().to(torch.float64) for t in (q, k, v, dout))
    dq = torch.zeros(batch, seq_q, heads, dim_qk, dtype=torch.float64)
    dk = torch.zeros(batch, seq_kv, heads, dim_qk, dtype=torch.float64)
    dv = torch.zeros(batch, seq_kv, heads, dim_v, dtype=torch.float64)
    key_pos = torch.arange(seq_kv)
    q_pos = torch.arange(seq_q) + shift
    hidden = key_pos[None, :] > q_pos[:, None] if is_causal else None
    for b in range(batch):
        for h0 in range(0, heads, slab_heads):
            hs = slice(h0, min(h0 + slab_heads, heads))
            s = torch.einsum("shd,nhd->hsn", qf[b, :, hs], kf[b, :, hs]) * scale
            if hidden is not None:
                s = s.masked_fill(hidden[None], float("-inf"))
            lse = s.logsumexp(-1, keepdim=True)
            p = torch.exp(s - lse).nan_to_num(0.0)  # rows that see no key -> 0
            dp = torch.einsum("shd,nhd->hsn", dof[b, :, hs], vf[b, :, hs])
            ds = p * (dp - (p * dp).sum(-1, keepdim=True))
            dv[b, :, hs] += torch.einsum("hsn,shd->nhd", p, dof[b, :, hs])
            dk[b, :, hs] += torch.einsum("hsn,shd->nhd", ds, qf[b, :, hs])
            dq[b, :, hs] = torch.einsum("hsn,nhd->shd", ds, kf[b, :, hs])
    dq, dk = dq * scale, dk * scale
    dq_acc_true = (dq / scale).to(torch.float32)
    return dq.to(dtype), dk.to(dtype), dv.to(dtype), dq_acc_true


def forward_stats(q, k, v, dout, is_causal=True, slab_heads=4):
    """The ``(lse, delta)`` pair the upstream launches hand to this kernel.

    ``lse`` is the natural-log log-sum-exp of the (bottom-right masked) scores,
    ``-inf`` for a query row that sees no key; ``delta = rowsum(o * dout)`` with
    ``o = P @ v``.  Computed in float32 from the 16-bit inputs, i.e. exactly what
    a float32-accumulating forward pass produces (independently of the golden,
    which recomputes both quantities in float64 from scratch).
    """
    batch, seq_q, heads, dim_qk = q.shape
    seq_kv, dim_v = k.shape[1], v.shape[-1]
    scale = dim_qk**-0.5
    shift = seq_kv - seq_q
    qf, kf, vf, dof = (t.detach().float() for t in (q, k, v, dout))
    lse = torch.empty(batch, heads, seq_q, dtype=torch.float32)
    delta = torch.empty(batch, heads, seq_q, dtype=torch.float32)
    key_pos = torch.arange(seq_kv)
    q_pos = torch.arange(seq_q) + shift
    hidden = key_pos[None, :] > q_pos[:, None] if is_causal else None
    for b in range(batch):
        for h0 in range(0, heads, slab_heads):
            hs = slice(h0, min(h0 + slab_heads, heads))
            s = torch.einsum("shd,nhd->hsn", qf[b, :, hs], kf[b, :, hs]) * scale
            if hidden is not None:
                s = s.masked_fill(hidden[None], float("-inf"))
            row_lse = s.logsumexp(-1)  # (h, s); -inf when nothing is visible
            p = torch.exp(s - row_lse[..., None]).nan_to_num(0.0)
            # Head-first matmuls (no einsum label ambiguity on size-1 axes).
            vs = vf[b, :, hs].permute(1, 0, 2)  # (h, n, dv)
            dos = dof[b, :, hs].permute(1, 0, 2)  # (h, s, dv)
            o = torch.matmul(p, vs)  # (h, s, dv)
            lse[b, hs] = row_lse
            delta[b, hs] = (o * dos).sum(-1)
    return lse.contiguous(), delta.contiguous()


# ---------------------------------------------------------------------------
# 2. Kernel
# ---------------------------------------------------------------------------
def _geometry(batch, heads, dim_qk, dim_v, seq_q, seq_kv, is_causal):
    """Build-time geometry; every extent is a compile-time constant."""
    ntiles = (seq_kv + _BN - 1) // _BN
    return {
        "ntiles": ntiles,
        "grid": heads * batch * ntiles,
        "loop_ed": (seq_q + _BM - 1) // _BM,
        "shift": seq_kv - seq_q,
        "total_q": batch * seq_q,
        "total_kv": batch * seq_kv,
        "is_causal": bool(is_causal),
    }


@functools.lru_cache(maxsize=64)
def _bwd_kernel(
    batch, heads, dim_qk, dim_v, dense_seqlen_q, dense_seqlen_kv, is_causal, scale, dtype,
    a2_fast=True,
):
    """Frozen factory, signature-compatible with the source ``_bwd_kernel``.

    ``batch, heads, dim_qk, dim_v, dense_seqlen_q, dense_seqlen_kv, is_causal,
    scale, dtype`` is exactly the source order, so the harness keeps its call
    site.  ``dense_seqlen_q == 0`` selects the packed layout in the source and is
    out of this kernel's scope (``ValueError``).  ``is_causal``, ``scale`` and
    ``a2_fast`` are build-time constants; ``a2_fast=False`` forces the masked
    epilogue arm everywhere and exists only for the arm-equivalence check (P6).
    """
    seq_q, seq_kv = dense_seqlen_q, dense_seqlen_kv
    if dtype not in _IN_DTYPES:
        raise ValueError(f"dtype: {dtype!r} is not one of {_IN_DTYPES}")
    if dense_seqlen_q <= 0:
        raise ValueError(
            f"shape: dense_seqlen_q must be positive for MLADenseBwdOp (got {dense_seqlen_q}); "
            "the packed layout is out of this kernel's scope"
        )
    if min(batch, heads, dim_qk, dim_v, seq_q, seq_kv) <= 0:
        raise ValueError(
            "shape: batch/heads/dim_qk/dim_v/seq_q/seq_kv must all be positive, got "
            f"{batch}/{heads}/{dim_qk}/{dim_v}/{seq_q}/{seq_kv}"
        )
    if dim_qk % 32 or dim_v % 16:
        raise ValueError(f"shape: dim_qk ({dim_qk}) must divide 32, dim_v ({dim_v}) must divide 16")

    geo = _geometry(batch, heads, dim_qk, dim_v, seq_q, seq_kv, is_causal)
    ntiles, grid, loop_ed = geo["ntiles"], geo["grid"], geo["loop_ed"]
    shift, total_q, total_kv = geo["shift"], geo["total_q"], geo["total_kv"]
    dt = dtype

    @T.prim_func
    def _bwd_main(
        q: T.Tensor((total_q, heads, dim_qk), dt),  # type: ignore
        k: T.Tensor((total_kv, heads, dim_qk), dt),  # type: ignore
        v: T.Tensor((total_kv, heads, dim_v), dt),  # type: ignore
        dout: T.Tensor((total_q, heads, dim_v), dt),  # type: ignore
        lse: T.Tensor((batch, heads, seq_q), "float32"),  # type: ignore
        delta: T.Tensor((batch, heads, seq_q), "float32"),  # type: ignore
        cu_q: T.Tensor((batch + 1,), "int32"),  # type: ignore
        cu_kv: T.Tensor((batch + 1,), "int32"),  # type: ignore
        dq_acc: T.Tensor((total_q, heads, dim_qk), "float32"),  # type: ignore
        dk: T.Tensor((total_kv, heads, dim_qk), dt),  # type: ignore
        dv: T.Tensor((total_kv, heads, dim_v), dt),  # type: ignore
    ) -> None:
        with T.Kernel(grid) as bid:
            h = bid // (batch * ntiles)
            tile = bid % (batch * ntiles)
            b = tile // ntiles
            n0 = (tile % ntiles) * _BN
            # Sub-block id of the AIV running this code (0/1).  Bound outside the
            # SIMT region because asc_get_sub_block_id() is not callable inside a
            # __simt_vf__ body; the value is then captured as a VF argument and
            # is the same id the dual copies split with.
            sid = T.call_extern("int64", "asc_get_sub_block_id")

            # ---- L1: key resident + per-iteration operands -------------------
            k_s = T.alloc_l1((_BN, dim_qk), dt)
            v_s = T.alloc_l1((_BN, dim_v), dt)
            kT_s = T.alloc_l1((dim_qk, _BN), dt)
            q_s = T.alloc_l1((_BM, dim_qk), dt)
            qT_s = T.alloc_l1((dim_qk, _BM), dt)
            do_s = T.alloc_l1((_BM, dim_v), dt)
            doT_s = T.alloc_l1((dim_v, _BM), dt)
            pT_s = T.alloc_l1((_BN, _BM), dt)
            dsT_s = T.alloc_l1((_BN, _BM), dt)
            ds_s = T.alloc_l1((_BM, _BN), dt)
            # ---- L0C accumulators ------------------------------------------
            st_c = T.alloc_l0c((_BN, _BM), "float32")
            dpt_c = T.alloc_l0c((_BN, _BM), "float32")
            dq_c = T.alloc_l0c((_BM, dim_qk), "float32")
            dk_c = T.alloc_l0c((_BN, dim_qk), "float32")
            dv_c = T.alloc_l0c((_BN, dim_v), "float32")
            # ---- UB: one half tile per AIV (dual_copy 2:1) ------------------
            st_u = T.alloc_shared((_HN, _BM), "float32")
            dp_u = T.alloc_shared((_HN, _BM), "float32")
            p16_u = T.alloc_shared((_HN, _BM), dt)
            dsT_u = T.alloc_shared((_HN, _BM), dt)
            ds_u = T.alloc_shared((_BM, _HN), dt)
            lse_u = T.alloc_shared((_BM,), "float32")
            dlt_u = T.alloc_shared((_BM,), "float32")
            tail_u = T.alloc_shared((_HN, dim_qk), "float32")
            tailv_u = T.alloc_shared((_HN, dim_v), "float32")
            out_k = T.alloc_shared((_HN, dim_qk), dt)
            out_v = T.alloc_shared((_HN, dim_v), dt)

            koff = b * seq_kv + n0
            # Whole-extent loads: the tail rows outside [0, S_kv) are clamped by
            # the OOB pass and filled with zero; the mask arm drops them later.
            T.copy(k[koff : koff + _BN, h, :], k_s)
            T.copy(v[koff : koff + _BN, h, :], v_s)
            T.copy(k[koff : koff + _BN, h, :], kT_s, transpose=True)

            loop_st = T.max(0, (n0 - shift) // _BM) if geo["is_causal"] else 0
            # num_stages=1 pins every per-iteration buffer to a single version, so
            # the explicit T.annotate_buffer_versions({buf: 1}) of DESIGN.md 3.2
            # step 10 (which only steers the Persistent/multi-stage scheduler) is
            # not needed; a multi-version loop is a Stage 4 decision.
            for kq in T.Pipelined(loop_st, loop_ed, num_stages=1):
                m0 = kq * _BM
                qoff = b * seq_q + m0
                T.copy(q[qoff : qoff + _BM, h, :], q_s)
                T.copy(dout[qoff : qoff + _BM, h, :], do_s)
                T.copy(q[qoff : qoff + _BM, h, :], qT_s, transpose=True)
                T.copy(dout[qoff : qoff + _BM, h, :], doT_s, transpose=True)
                T.copy(lse[b, h, m0 : m0 + _BM], lse_u)
                T.copy(delta[b, h, m0 : m0 + _BM], dlt_u)

                T.gemm(k_s, q_s, st_c, transpose_B=True, clear_accum=True)
                T.gemm(v_s, do_s, dpt_c, transpose_B=True, clear_accum=True)
                # L0C -> UB: each AIV receives its own 32 key rows of the tile.
                T.dual_copy(st_c, st_u)
                T.dual_copy(dpt_c, dp_u)

                fast = (n0 + _BN <= seq_kv) & (m0 + _BM <= seq_q)
                if geo["is_causal"]:
                    fast = fast & (n0 + _BN - 1 <= m0 + shift)
                if a2_fast:
                    if fast:
                        # Fast arm: the whole tile is visible and fully in range.
                        with T.SimtVF(threads=128):
                            for i, j in T.Parallel(_HN, _BM):
                                p = T.exp(st_u[i, j] * scale - lse_u[j])
                                p16_u[i, j] = T.cast(p, dt)
                                dsv = p * (dp_u[i, j] - dlt_u[j])
                                dsT_u[i, j] = T.cast(dsv, dt)
                                ds_u[j, i] = T.cast(dsv, dt)
                    else:
                        with T.SimtVF(threads=128):
                            for i, j in T.Parallel(_HN, _BM):
                                gi = n0 + T.cast(sid, "int32") * _HN + i
                                gj = m0 + j
                                vis = (gi < seq_kv) & (gj < seq_q)
                                if geo["is_causal"]:
                                    vis = vis & (gi <= gj + shift)
                                p = T.exp(st_u[i, j] * scale - lse_u[j])
                                pv = T.if_then_else(vis, p, T.cast(0, "float32"))
                                p16_u[i, j] = T.cast(pv, dt)
                                dsv = pv * (dp_u[i, j] - dlt_u[j])
                                dsT_u[i, j] = T.cast(dsv, dt)
                                ds_u[j, i] = T.cast(dsv, dt)
                else:
                    with T.SimtVF(threads=128):
                        for i, j in T.Parallel(_HN, _BM):
                            gi = n0 + T.cast(sid, "int32") * _HN + i
                            gj = m0 + j
                            vis = (gi < seq_kv) & (gj < seq_q)
                            if geo["is_causal"]:
                                vis = vis & (gi <= gj + shift)
                            p = T.exp(st_u[i, j] * scale - lse_u[j])
                            pv = T.if_then_else(vis, p, T.cast(0, "float32"))
                            p16_u[i, j] = T.cast(pv, dt)
                            dsv = pv * (dp_u[i, j] - dlt_u[j])
                            dsT_u[i, j] = T.cast(dsv, dt)
                            ds_u[j, i] = T.cast(dsv, dt)

                # UB -> L1: M-split for the key-major tiles, N-split for dS.
                T.dual_copy(p16_u, pT_s)
                T.dual_copy(dsT_u, dsT_s)
                T.dual_copy(ds_u, ds_s)
                T.gemm(pT_s, doT_s, dv_c, transpose_B=True, clear_accum=(kq == loop_st))
                T.gemm(dsT_s, qT_s, dk_c, transpose_B=True, clear_accum=(kq == loop_st))
                T.gemm(ds_s, kT_s, dq_c, transpose_B=True, clear_accum=True)
                # dQ accumulates across cores -> hardware store-atomic into fp32.
                T.set_atomic("add", "float32")
                T.copy(dq_c, dq_acc[qoff : qoff + _BM, h, :])
                T.set_atomic_none()

            # ---- block tail: dk = scale * cast(dk_c), dv = cast(dv_c) ------
            T.dual_copy(dk_c, tail_u)
            T.dual_copy(dv_c, tailv_u)
            with T.SimtVF(threads=128):
                for i, d in T.Parallel(_HN, dim_qk):
                    out_k[i, d] = T.cast(tail_u[i, d] * scale, dt)
                for i, d in T.Parallel(_HN, dim_v):
                    out_v[i, d] = T.cast(tailv_u[i, d], dt)
            koff_own = koff + T.cast(sid, "int32") * _HN
            T.copy(out_k, dk[koff_own : koff_own + _HN, h, :])
            T.copy(out_v, dv[koff_own : koff_own + _HN, h, :])

    return tilelang.compile(_bwd_main, target="ascend")


def _launch(batch, heads, dim_qk, dim_v, seq_q, seq_kv, is_causal, scale, dtype, a2_fast=True):
    """Compiled kernel for one geometry (same argument order as the factory)."""
    return _bwd_kernel(
        batch, heads, dim_qk, dim_v, seq_q, seq_kv, is_causal, scale, dtype, a2_fast
    )


# ---------------------------------------------------------------------------
# 3. Launch + comparison with the golden
# ---------------------------------------------------------------------------
_RESULTS = []  # (level, name, ok, detail)
_MAX_DIFF = [0.0]


def _record(level, name, ok, detail=""):
    _RESULTS.append((level, name, ok, detail))
    tag = "PASS" if ok else "FAIL"
    print(f"[{level}][{tag}] {name}{(' -- ' + detail) if detail and not ok else ''}", flush=True)


def _randn(shape, dtype_name, seed):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=gen, dtype=torch.float32).to(_TORCH_DTYPE[dtype_name])


def _max_diff(got, gold):
    a, b = got.float(), gold.float()
    diff = (a - b).abs()
    diff = torch.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)
    nan_mismatch = int(((a != a) != (b != b)).sum())
    nonfinite = int((~torch.isfinite(a)).sum())
    return (float(diff.max()) if diff.numel() else 0.0), nan_mismatch, nonfinite


def run_case(
    level,
    name,
    batch,
    seq_q,
    seq_kv,
    heads,
    dtype=_DEFAULT_DTYPE,
    is_causal=True,
    seed=0,
    special=None,
    repeat=1,
    a2_fast=True,
    dim_qk=192,
    dim_v=128,
    scale=None,
):
    """Build one geometry, run it on the NPU and compare all three gradients."""
    diff = float("nan")
    scale = dim_qk**-0.5 if scale is None else scale
    try:
        q = _randn((batch, seq_q, heads, dim_qk), dtype, seed)
        k = _randn((batch, seq_kv, heads, dim_qk), dtype, seed + 1)
        v = _randn((batch, seq_kv, heads, dim_v), dtype, seed + 2)
        dout = _randn((batch, seq_q, heads, dim_v), dtype, seed + 3)
        if special == "zero_dout":
            dout.zero_()
        elif special == "tiny_q":
            q = (q.float() * 0.05).to(q.dtype)
        elif special is not None and special != "atom_prefill":
            raise ValueError(f"special: unknown pattern {special!r}")

        dq_true, dk_true, dv_true, dq_acc_true = golden_bwd_kernel(
            q, k, v, dout, scale=scale, is_causal=is_causal, dtype=q.dtype, slab_heads=2
        )
        lse, delta = forward_stats(q, k, v, dout, is_causal=is_causal, slab_heads=2)

        dev = "npu"
        q_dev = q.reshape(batch * seq_q, heads, dim_qk).contiguous().to(dev)
        k_dev = k.reshape(batch * seq_kv, heads, dim_qk).contiguous().to(dev)
        v_dev = v.reshape(batch * seq_kv, heads, dim_v).contiguous().to(dev)
        do_dev = dout.reshape(batch * seq_q, heads, dim_v).contiguous().to(dev)
        lse_dev = lse.to(dev)
        delta_dev = delta.to(dev)
        cu_q = (torch.arange(batch + 1, dtype=torch.int32) * seq_q).to(dev)
        cu_kv = (torch.arange(batch + 1, dtype=torch.int32) * seq_kv).to(dev)
        # Every output allocation carries _OOB_PAD sentinel rows behind it: the
        # kernel receives a leading view, so any write past T_kv / T_q lands in
        # the sentinel area and is caught (probe P2).
        pad = _OOB_PAD
        dq_base = torch.full((batch * seq_q + pad, heads, dim_qk), _SENTINEL,
                            dtype=torch.float32, device=dev)
        dq_acc = dq_base[0 : batch * seq_q]
        if special == "atom_prefill":
            # Store-atomic contract (probe P5): a pre-filled buffer must receive
            # the increment on top of its content, never a plain store.
            dq_acc.fill_(2.0)
        else:
            dq_acc.zero_()
        dk_base = torch.full((batch * seq_kv + pad, heads, dim_qk), _SENTINEL,
                             dtype=_TORCH_DTYPE[dtype], device=dev)
        dv_base = torch.full((batch * seq_kv + pad, heads, dim_v), _SENTINEL,
                             dtype=_TORCH_DTYPE[dtype], device=dev)
        dk_dev = dk_base[0 : batch * seq_kv]
        dv_dev = dv_base[0 : batch * seq_kv]

        fn = _launch(
            batch, heads, dim_qk, dim_v, seq_q, seq_kv, is_causal, scale, dtype, a2_fast
        )
        args = (q_dev, k_dev, v_dev, do_dev, lse_dev, delta_dev, cu_q, cu_kv, dq_acc, dk_dev, dv_dev)
        start = time.perf_counter()
        fn(*args)
        torch.npu.synchronize()
        elapsed = (time.perf_counter() - start) * 1e3
        # Read the results before any timed repeat: dq_acc is an in-out store
        # atomic accumulator, so re-launching would add into the same buffer.
        got_dq32 = dq_acc.cpu()
        got_dk = dk_dev.cpu()
        got_dv = dv_dev.cpu()
        oob = 0
        for base, rows in ((dq_base, batch * seq_q), (dk_base, batch * seq_kv), (dv_base, batch * seq_kv)):
            oob += int((base[rows:] != _SENTINEL).sum())
        if repeat > 1:
            start = time.perf_counter()
            for _ in range(repeat):
                fn(*args)
            torch.npu.synchronize()
            elapsed = (time.perf_counter() - start) * 1e3 / repeat
        if oob:
            raise AssertionError(f"{oob} cell(s) written outside the valid range (OOB store)")

        got_dq = (got_dq32 * scale).to(_TORCH_DTYPE[dtype])
        if special == "atom_prefill":
            got_dq32 = got_dq32 - 2.0
            got_dq = (got_dq32 * scale).to(_TORCH_DTYPE[dtype])
        dk_true = dk_true.reshape(batch * seq_kv, heads, dim_qk)
        dv_true = dv_true.reshape(batch * seq_kv, heads, dim_v)
        dq_true = dq_true.reshape(batch * seq_q, heads, dim_qk)
        dq_acc_true = dq_acc_true.reshape(batch * seq_q, heads, dim_qk)

        d_dk, nan_dk, nf_dk = _max_diff(got_dk, dk_true)
        d_dv, nan_dv, nf_dv = _max_diff(got_dv, dv_true)
        d_dq, nan_dq, nf_dq = _max_diff(got_dq32, dq_acc_true)
        diff = max(d_dk, d_dv, d_dq)
        _MAX_DIFF[0] = max(_MAX_DIFF[0], diff)
        bad = nan_dk + nan_dv + nan_dq
        if bad:
            raise AssertionError(f"{bad} NaN position(s) disagree with the golden (max_diff={diff:.3e})")
        nonfinite = nf_dk + nf_dv + nf_dq
        rtol = _RTOL[dtype]
        torch.testing.assert_close(got_dk, dk_true, atol=_ATOL, rtol=rtol, equal_nan=True)
        torch.testing.assert_close(got_dv, dv_true, atol=_ATOL, rtol=rtol, equal_nan=True)
        torch.testing.assert_close(got_dq, dq_true, atol=_ATOL, rtol=rtol, equal_nan=True)
        # float32 dq_acc intermediate against its own (scale-free) reference.  The
        # relative tolerance is the *input dtype's* one: dS is cast to the input
        # dtype before the dQ GEMM (source semantics), so the fp32 accumulator
        # carries that rounding even though the golden reference is fp64.
        tol_dq = _ATOL / scale
        torch.testing.assert_close(got_dq32, dq_acc_true, atol=tol_dq, rtol=rtol, equal_nan=True)
    except AssertionError as exc:
        _record(level, name, False, f"max_diff={diff:.3e} {str(exc)[:300]}")
        return False
    except (ValueError, RuntimeError) as exc:  # noqa: BLE001 - reported, never swallowed
        _record(level, name, False, f"{type(exc).__name__}: {str(exc)[:300]}")
        return False
    _record(level, name, True)
    print(
        f"       B={batch} S_q={seq_q} S_kv={seq_kv} H={heads} D={dim_qk}/{dim_v} {dtype} "
        f"causal={int(is_causal)} grid={heads * batch * ((seq_kv + _BN - 1) // _BN)} "
        f"max_diff={diff:.3e} (dk {d_dk:.2e} dv {d_dv:.2e} dq {d_dq:.2e}) "
        f"launch={elapsed:.3f} ms (nonfinite cells {nonfinite})",
        flush=True,
    )
    return True


def _finish(level, strict):
    rows = [r for r in _RESULTS if r[0] == level]
    failed = [r for r in rows if not r[2]]
    print(
        f"[{level}] {len(rows) - len(failed)}/{len(rows)} cases passed, max_diff={_MAX_DIFF[0]:.6g}",
        flush=True,
    )
    if failed and strict:
        raise AssertionError(f"[{level}] {len(failed)} case(s) failed")
    if failed:
        print(f"[{level}][WARN] {len(failed)} non-blocking case(s) failed", flush=True)


# ---------------------------------------------------------------------------
# 4. Layered cases
# ---------------------------------------------------------------------------
def _case_contract(level, name):
    """Factory rejects contracts it cannot serve, and caches per geometry."""
    problems = []
    for args, why in (
        ((1, 4, 192, 128, 64, 64, True, 192**-0.5, "float32"), "float32 inputs"),
        ((1, 4, 192, 128, 0, 64, True, 192**-0.5, "float16"), "dense_seqlen_q=0 (packed)"),
        ((0, 4, 192, 128, 64, 64, True, 192**-0.5, "float16"), "batch=0"),
        ((1, 4, 100, 128, 64, 64, True, 192**-0.5, "float16"), "dim_qk not a multiple of 32"),
    ):
        try:
            _bwd_kernel(*args)
            problems.append(f"{why}: no error raised")
        except ValueError:
            pass
    first = _bwd_kernel(1, 2, 192, 128, 64, 64, True, 192**-0.5, "float16")
    again = _bwd_kernel(1, 2, 192, 128, 64, 64, True, 192**-0.5, "float16")
    if first is not again:
        problems.append("the compiled program is not cached per (geometry, dtype)")
    if problems:
        _record(level, name, False, "; ".join(problems))
        return False
    _record(level, name, True)
    print("       4 invalid contracts rejected, program cached per (geometry, dtype)", flush=True)
    return True


def _case_golden_naive(level, name):
    """Self-check the golden against an explicit per-pair loop (tiny geometry)."""
    problems = []
    for seq_q, seq_kv, is_causal in ((6, 8, True), (8, 4, True), (5, 5, False)):
        q = _randn((1, seq_q, 2, 8), "float16", 61)
        k = _randn((1, seq_kv, 2, 8), "float16", 62)
        v = _randn((1, seq_kv, 2, 6), "float16", 63)
        do = _randn((1, seq_q, 2, 6), "float16", 64)
        scale = 8**-0.5
        dq, dk, dv, _ = golden_bwd_kernel(
            q, k, v, do, scale=scale, is_causal=is_causal, dtype=torch.float64
        )
        shift = seq_kv - seq_q
        qf, kf, vf, dof = (t.double() for t in (q, k, v, do))
        r_dq = torch.zeros_like(dq)
        r_dk = torch.zeros_like(dk)
        r_dv = torch.zeros_like(dv)
        for b in range(1):
            for h in range(2):
                for m in range(seq_q):
                    scores = []
                    for n in range(seq_kv):
                        s = float((qf[b, m, h] * kf[b, n, h]).sum()) * scale
                        visible = n < seq_kv and (not is_causal or n <= m + shift)
                        scores.append(s if visible else None)
                    vis = [s for s in scores if s is not None]
                    lse = math.log(sum(math.exp(s) for s in vis)) if vis else None
                    p = torch.zeros(seq_kv, dtype=torch.float64)
                    if lse is not None:
                        for n, s in enumerate(scores):
                            if s is not None:
                                p[n] = math.exp(s - lse)
                    dp = torch.tensor(
                        [float((dof[b, m, h] * vf[b, n, h]).sum()) for n in range(seq_kv)],
                        dtype=torch.float64,
                    )
                    delta = float((p * dp).sum())
                    ds = p * (dp - delta)
                    r_dv[b, :, h] += p[:, None] * dof[b, m, h][None, :]
                    r_dk[b, :, h] += ds[:, None] * qf[b, m, h][None, :]
                    r_dq[b, m, h] = (ds[:, None] * kf[b, :, h]).sum(0) * scale
        r_dk = r_dk * scale
        for got, ref, tag in ((dq, r_dq, "dq"), (dk, r_dk, "dk"), (dv, r_dv, "dv")):
            gap = (got - ref).abs().max().item()
            if gap > 1e-12:
                problems.append(f"S={seq_q}/{seq_kv} causal={int(is_causal)} {tag}: {gap:.3e}")
    if problems:
        _record(level, name, False, "; ".join(problems))
        return False
    _record(level, name, True)
    print("       golden == explicit per-pair loop on 3 geometries (1e-12)", flush=True)
    return True


def _case_atomic_contract(level, name):
    """P5: store-atomic adds onto the caller's buffer instead of overwriting it."""
    ok = run_case(level, name + "-prefilled", 1, 64, 64, 1, "float16", seed=91,
                  special="atom_prefill")
    ok = run_case(level, name + "-zeroed", 1, 64, 64, 1, "float16", seed=91) and ok
    return ok


def run_L0():
    """Gate: manifest smoke, D_qk=192, bottom-right both shifts, -inf rows, no causal."""
    run_case("L0", "L0-1 bf16-manifest-smoke", 1, 512, 512, 8, "bfloat16", seed=1)
    run_case("L0", "L0-2 fp16-manifest-smoke", 1, 512, 512, 8, "float16", seed=2)
    run_case("L0", "L0-3 bf16-non-causal", 1, 512, 512, 8, "bfloat16", seed=3, is_causal=False)
    run_case("L0", "L0-4 bf16-tail-S96", 1, 96, 96, 2, "bfloat16", seed=4)
    run_case("L0", "L0-5 bf16-shift-pos-96x192", 1, 96, 192, 2, "bfloat16", seed=5)
    run_case("L0", "L0-6 bf16-shift-neg-128x64", 1, 128, 64, 2, "bfloat16", seed=6)
    run_case("L0", "L0-7 fp16-single-block", 1, 64, 64, 1, "float16", seed=7)
    run_case("L0", "L0-8 fp16-tail-S65", 1, 65, 65, 1, "float16", seed=8)
    _case_golden_naive("L0", "L0-9 golden-vs-naive-loop")
    _case_contract("L0", "L0-10 refuses-an-invalid-contract")
    _case_atomic_contract("L0", "L0-11 store-atomic")
    # P6: the fast arm and the masked arm must agree cell for cell.
    _case_arm_equivalence("L0", "L0-12 a2-arm-equivalence")
    _finish("L0", strict=True)


def _case_arm_equivalence(level, name):
    """P6: force the masked arm and compare against the specialised build."""
    problems = []
    for seq_q, seq_kv, dtype in ((128, 128, "bfloat16"), (96, 192, "bfloat16")):
        outs = []
        for a2_fast in (True, False):
            q = _randn((1, seq_q, 2, 192), dtype, 51)
            k = _randn((1, seq_kv, 2, 192), dtype, 52)
            v = _randn((1, seq_kv, 2, 128), dtype, 53)
            do = _randn((1, seq_q, 2, 128), dtype, 54)
            lse, delta = forward_stats(q, k, v, do, slab_heads=2)
            dev = "npu"
            dq = torch.zeros(seq_q, 2, 192, dtype=torch.float32, device=dev)
            dk = torch.zeros(seq_kv, 2, 192, dtype=_TORCH_DTYPE[dtype], device=dev)
            dv = torch.zeros(seq_kv, 2, 128, dtype=_TORCH_DTYPE[dtype], device=dev)
            fn = _launch(1, 2, 192, 128, seq_q, seq_kv, True, 192**-0.5, dtype, a2_fast)
            fn(q.reshape(seq_q, 2, 192).to(dev), k.reshape(seq_kv, 2, 192).to(dev),
               v.reshape(seq_kv, 2, 128).to(dev), do.reshape(seq_q, 2, 128).to(dev),
               lse.to(dev), delta.to(dev),
               torch.zeros(2, dtype=torch.int32, device=dev),
               torch.zeros(2, dtype=torch.int32, device=dev), dq, dk, dv)
            torch.npu.synchronize()
            outs.append((dq.cpu(), dk.cpu(), dv.cpu()))
        for idx, tag in enumerate(("dq", "dk", "dv")):
            a, b = outs[0][idx].float(), outs[1][idx].float()
            gap = (a - b).abs().max().item()
            # dk/dv are deterministic (one owner per key block); dq goes through
            # store-atomic accumulation, whose cross-core order is unspecified.
            if tag == "dq":
                if gap > 1e-5:
                    problems.append(f"S={seq_q}/{seq_kv} {tag}: arms differ (max {gap:.3e})")
            elif not torch.equal(outs[0][idx], outs[1][idx]):
                problems.append(f"S={seq_q}/{seq_kv} {tag}: arms differ (max {gap:.3e})")
    if problems:
        _record(level, name, False, "; ".join(problems))
        return False
    _record(level, name, True)
    print(
        "       fast arm == masked arm: dk/dv bitwise, dq <= 1e-5 (atomic order) on 2 geometries",
        flush=True,
    )
    return True


def run_L1():
    """Representative shapes, both dtypes, both causal settings."""
    run_case("L1", "L1-1 bf16-smoke-1024", 1, 1024, 1024, 4, "bfloat16", seed=11, repeat=2)
    run_case("L1", "L1-2 fp16-smoke-1024", 1, 1024, 1024, 4, "float16", seed=12, repeat=2)
    run_case("L1", "L1-3 bf16-two-batches", 2, 512, 512, 4, "bfloat16", seed=13, repeat=2)
    run_case("L1", "L1-4 bf16-bottom-right", 1, 512, 768, 4, "bfloat16", seed=14, repeat=2)
    run_case("L1", "L1-5 fp16-non-causal-1024", 1, 1024, 1024, 4, "float16", seed=15,
             is_causal=False, repeat=2)
    run_case("L1", "L1-6 bf16-single-head", 1, 768, 768, 1, "bfloat16", seed=16, repeat=2)
    _finish("L1", strict=True)


def run_L2():
    """Larger informative geometries (golden is O(H*S^2) on CPU -> capped)."""
    run_case("L2", "L2-1 bf16-2k", 1, 2048, 2048, 2, "bfloat16", seed=21, repeat=1)
    run_case("L2", "L2-2 fp16-1536", 1, 1536, 1536, 2, "float16", seed=22, repeat=1)
    run_case("L2", "L2-3 bf16-b2-1024", 2, 1024, 1024, 4, "bfloat16", seed=23, repeat=1)
    run_case("L2", "L2-4 bf16-2k-shift", 1, 1024, 2048, 2, "bfloat16", seed=24, repeat=1)
    _finish("L2", strict=False)


def run_boundary():
    """Sub-tile sizes, tails, both shifts, zero gradients, tiny scores."""
    run_case("boundary", "B-1 S=1", 1, 1, 1, 1, "bfloat16", seed=31)
    run_case("boundary", "B-2 S=63 tail-63", 1, 63, 63, 2, "bfloat16", seed=32)
    run_case("boundary", "B-3 S=65 tail-1", 1, 65, 65, 2, "float16", seed=33)
    run_case("boundary", "B-4 S=129 tail-1", 1, 129, 129, 2, "bfloat16", seed=34)
    run_case("boundary", "B-5 S=200 tail-8", 1, 200, 200, 4, "bfloat16", seed=35)
    run_case("boundary", "B-6 shift+ 65x129", 1, 65, 129, 2, "bfloat16", seed=36)
    run_case("boundary", "B-7 shift- 192x64", 1, 192, 64, 2, "bfloat16", seed=37)
    run_case("boundary", "B-8 zero-dout", 1, 96, 96, 2, "float16", seed=38, special="zero_dout")
    run_case("boundary", "B-9 tiny-scores", 1, 128, 128, 2, "bfloat16", seed=39, special="tiny_q")
    run_case("boundary", "B-10 non-causal-shift+", 1, 96, 192, 2, "bfloat16", seed=40,
             is_causal=False)
    run_case("boundary", "B-11 B=3 H=1", 3, 128, 128, 1, "float16", seed=41)
    _finish("boundary", strict=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0", choices=["L0", "L1", "L2", "boundary", "all"])
    args, _ = parser.parse_known_args()
    levels = {"L0": run_L0, "L1": run_L1, "L2": run_L2, "boundary": run_boundary}
    chosen = list(levels.values()) if args.level == "all" else [levels[args.level]]
    started = time.perf_counter()
    for run in chosen:
        run()
    print(f"[TIME] {time.perf_counter() - started:.1f} s for level={args.level}", flush=True)
    print(f"[PRECISION_PASS] max_diff={_MAX_DIFF[0]:.6g}", flush=True)


if __name__ == "__main__":
    main()
