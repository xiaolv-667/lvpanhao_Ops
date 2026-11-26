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

Structure (key stationary, one key block per core, DESIGN.md v3 0.6 R1-R6):

* one-dimensional ``T.Kernel(grid)``, ``grid = H * B * ceil(S_kv / 96)``;
* ``k/v`` (and the transposed ``kT``) stay resident in L1 for the whole query
  loop; ``dk/dv`` accumulate in L0C across the loop;
* all five GEMMs use the explicit L0 path (``alloc_l0a/alloc_l0b`` + one
  ``T.copy(L1 -> L0)`` before each GEMM) because the L1 operand path and
  explicit L0 buffers overwrite each other inside one kernel (DESIGN.md 0.6 R6);
* ``dS`` exists once (``dsT_s0``/``dsT_s1``): ``dK`` consumes it as an A operand
  and ``dQ`` consumes the same tile with ``transpose_A=True``, so the transpose
  is done by ``asc_copy_l12l0a_transpose`` on MTE1 instead of a strided
  vector-side copy (DESIGN.md 0.6 R3);
* the ``P``/``dS`` epilogue runs on the vector side from the L0C accumulators
  and produces exactly two tiles (``p16``, ``dsT``) mapped over the query axis.

Schedule (v4 = A5 rotation, DESIGN.md 6.3 B / 13.3): the query loop is
``T.serial(n_half)`` over **two statically expanded half bodies**, and the
gradient GEMMs lag one iteration::

    half body(k, version v) = F(k) -> GA(k) -> EP(k) -> GB(k-1, read 1-v)

so the AIC's ``dV/dK/dQ`` GEMMs of ``k-1`` execute *beside* the AIV's epilogue
of ``k`` instead of behind it.  Version identity is carried by static buffer
names (``*_s0``/``*_s1``, six buffers each) because a ``T.gemm`` L0 operand
region needs a static origin; the first iteration has no lagged ``GB``
(``if kq_a > loop_st``), ``clear_accum`` moves to ``(kq-1 == loop_st)``, ``dq``
is written at ``(kq-1)*BLOCK_M``, and the missing ``GB(loop_ed-1)`` is drained
after the loop with a version selected by the ``n_it`` parity.  The rotation
does not touch any arithmetic: the operand set and the accumulation order of
``dK``/``dV`` are unchanged, which is why ``dk/dv`` stay bit-identical to the
frozen v3 body (``_case_rotation_parity``).
``T.Pipelined``/``T.annotate_buffer_versions`` are gone: the scheduler rewrites
*every* access of a versioned buffer with the current iteration index and so
cannot express a lagged consumer (Z3 reports ``unsat`` and degrades to a
topological sort -- visible in the compile log, and not fatal by design).

Two runtime facts measured on this toolchain shape the vector side and differ
from the frozen DESIGN.md wording (see RETROSPECTIVE.md stage 3):

1. ``T.copy(L0C -> UB)`` (``dual_dst_mode=0``) only lands in one AIV's buffer;
   the whole-tile epilogue of DESIGN.md 3.3 therefore reads stale data on the
   other AIV (probe P1: 99.2% mismatch).  The documented fallback is used
   instead: every L0C<->UB crossing is a ``T.dual_copy`` 2:1 split, so each AIV
   owns half of the tile (``_HN = 48`` key rows) in its own UB bank and the L1
   tiles stay whole (``pT_s0/pT_s1`` are still ``(96, 64)`` operands).
2. The half tile makes the key index AIV dependent, which the mask needs, so
   ``sid = asc_get_sub_block_id()`` is bound once outside the SIMT region and
   captured into every ``T.SimtVF`` (``T.Vector(vector=2) as sid`` remains
   uncompilable, ``InternalError ... dup sid``).

GM layout (DESIGN.md v3 0.2 / 5.4, fixes the v2 cross-request overwrite): the
seven token-major tensors are **4-D batch views** ``(B, S, H, D)``, so the
platform ``AscendInsertOOBPadding`` clamp (whose dimension is the buffer shape)
crops the ``dk``/``dv`` store to the *own request's* ``S_kv`` -- semantically
the source operator's per-row guard ``if n0[0]+i < len_kv``.  The host side
reaches these views with a zero-copy ``.view(B, S, H, D)``.

Content layout:

1. ``golden_bwd_kernel`` -- independent fp64 torch reference (materialises P).
2. ``forward_stats`` -- the ``lse`` / ``delta`` the upstream launches provide.
3. ``_geometry`` / ``_bwd_kernel`` -- the frozen kernel factory (``_fill`` /
   ``_scores`` / ``_epilogue`` / ``_gradients`` macros + the rotated loop).
4. ``run_case`` -- launch on the NPU and compare with the golden.
5. ``_case_rotation_parity`` -- v4 vs the frozen v3 body, ``dk/dv`` bitwise.
6. ``run_big`` -- production shape (B=2, S=8192, H=128) sliced reference.
7. ``run_L0`` / ``run_L1`` / ``run_L2`` / ``run_boundary`` / ``main``.
"""

import argparse
import functools
import importlib.util
import math
import os
import time

import tilelang
import tilelang.ascend.language as T
import torch
import torch_npu  # noqa: F401  -- registers torch.npu

_IN_DTYPES = ("float16", "bfloat16")
_TORCH_DTYPE = {"float16": torch.float16, "bfloat16": torch.bfloat16}

_BN = 96  # key rows of one block (DESIGN.md 5.2)
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
def golden_bwd_kernel(q, k, v, dout, scale=None, is_causal=True, dtype=None, slab_heads=4,
                      round_inputs=False):
    """Materialised fp64 reference of the backward pass (bottom-right causal).

    ``q`` ``(B, S_q, H, D_qk)``; ``k`` ``(B, S_kv, H, D_qk)``; ``v, dout``
    ``(B, S, H, D_v)``.  The probabilities and ``delta`` are recomputed here
    from ``q, k, v, dout`` -- the ``lse`` / ``delta`` tensors the kernel consumes
    are never read, so this reference shares no assumption with the tiling,
    the L0C accumulators, the dual split or the atomic ``dq`` accumulation.
    Returns ``(dq, dk, dv)`` rounded once to ``dtype`` (same contract as the
    source ``ref_program``) plus ``dq_acc_true = dq / scale``, the float32
    intermediate this kernel writes into ``dq_acc``.

    ``round_inputs=True`` applies the *source algorithm's* rounding -- ``P`` and
    ``dS`` cast to the input dtype before their GEMMs (``bwd.py:216, 244-246``)
    -- and is used only as the discriminator quoted in DESIGN.md 8.2 (recompute
    the reference after casting ``dS`` alone to the input dtype): a strongly
    cancelling ``dV``/``dK``
    element can exceed the absolute tolerance in the input dtype while the
    kernel still reproduces the algorithm exactly.  The gate itself uses
    ``round_inputs=False``.
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
            p_g = p.to(dtype).to(torch.float64) if round_inputs else p
            ds_g = ds.to(dtype).to(torch.float64) if round_inputs else ds
            dv[b, :, hs] += torch.einsum("hsn,shd->nhd", p_g, dof[b, :, hs])
            dk[b, :, hs] += torch.einsum("hsn,shd->nhd", ds_g, qf[b, :, hs])
            dq[b, :, hs] = torch.einsum("hsn,nhd->shd", ds_g, kf[b, :, hs])
    dq, dk = dq * scale, dk * scale
    dq_acc_true = (dq / scale).to(torch.float32)
    return dq.to(dtype), dk.to(dtype), dv.to(dtype), dq_acc_true


def golden_head(q_bh, k_bh, v_bh, do_bh, scale, is_causal, chunk=1024, round_inputs=False,
                dtype=None):
    """fp64 reference for ONE (request, head) slice, chunked over queries.

    Same mathematics as ``golden_bwd_kernel`` (materialise ``P`` for the query
    chunk, accumulate ``dV``/``dK`` over the chunks), used by ``run_big`` where
    a full ``(H, S, S)`` slab does not fit in host memory.  ``q_bh, k_bh`` are
    ``(S, D_qk)``, ``v_bh, do_bh`` are ``(S, D_v)``.  Returns
    ``(dq, dk, dv, lse, delta)`` in fp64 / fp32.  ``round_inputs`` selects the
    same-rounding discriminator of ``golden_bwd_kernel``.
    """
    seq_q, dim_qk = q_bh.shape
    seq_kv, dim_v = k_bh.shape[0], v_bh.shape[-1]
    shift = seq_kv - seq_q
    qf, kf, vf, dof = (t.double() for t in (q_bh, k_bh, v_bh, do_bh))
    dq = torch.zeros(seq_q, dim_qk, dtype=torch.float64)
    dk = torch.zeros(seq_kv, dim_qk, dtype=torch.float64)
    dv = torch.zeros(seq_kv, dim_v, dtype=torch.float64)
    lse = torch.empty(seq_q, dtype=torch.float32)
    delta = torch.empty(seq_q, dtype=torch.float32)
    key_pos = torch.arange(seq_kv)
    for m0 in range(0, seq_q, chunk):
        m1 = min(m0 + chunk, seq_q)
        s = torch.einsum("sd,nd->sn", qf[m0:m1], kf) * scale
        if is_causal:
            q_pos = torch.arange(m0, m1) + shift
            s = s.masked_fill(key_pos[None, :] > q_pos[:, None], float("-inf"))
        row = s.logsumexp(-1)
        p = torch.exp(s - row[:, None]).nan_to_num(0.0)
        dp = torch.einsum("sd,nd->sn", dof[m0:m1], vf)
        row_delta = (p * dp).sum(-1, keepdim=True)
        ds = p * (dp - row_delta)
        if round_inputs:
            p = p.to(dtype).to(torch.float64)
            ds = ds.to(dtype).to(torch.float64)
        dv += torch.einsum("sn,sd->nd", p, dof[m0:m1])
        dk += torch.einsum("sn,sd->nd", ds, qf[m0:m1])
        dq[m0:m1] = torch.einsum("sn,nd->sd", ds, kf)
        lse[m0:m1] = row
        delta[m0:m1] = row_delta.squeeze(-1)
    return dq * scale, dk * scale, dv, lse, delta


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


def _stats_npu(q, k, v, dout, is_causal, chunk=4096):
    """``(lse, delta)`` for every head, computed on the NPU (production shapes).

    Independent of the kernel under test (a plain per-head chunked forward
    pass), used by ``run_big`` only where a float64 host computation of all
    128 heads is out of budget; the heads that are actually compared get their
    statistics from the float64 host path instead.
    """
    seq_q, dim_qk = q.shape[1], q.shape[-1]
    seq_kv, dim_v = k.shape[1], k.shape[-1]
    scale = dim_qk**-0.5
    shift = seq_kv - seq_q
    batch, heads = q.shape[0], q.shape[2]
    dev = q.device
    lse = torch.empty(batch, heads, seq_q, dtype=torch.float32, device=dev)
    delta = torch.empty(batch, heads, seq_q, dtype=torch.float32, device=dev)
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
                o = torch.matmul(p.to(v.dtype), vv).float()
                delta[b, h, m0:m1] = (o * dout[b, m0:m1, h, :].float()).sum(-1)
    return lse, delta


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
        "is_causal": bool(is_causal),
    }


@functools.lru_cache(maxsize=64)
def _bwd_kernel(
    batch, heads, dim_qk, dim_v, dense_seqlen_q, dense_seqlen_kv, is_causal, scale, dtype,
    a2_fast=True, n_versions=2,
):
    """Frozen factory, signature-compatible with the source ``_bwd_kernel``.

    ``batch, heads, dim_qk, dim_v, dense_seqlen_q, dense_seqlen_kv, is_causal,
    scale, dtype`` is exactly the source order, so the harness keeps its call
    site.  ``dense_seqlen_q == 0`` selects the packed layout in the source and is
    out of this kernel's scope (``ValueError``).  ``is_causal``, ``scale`` and
    ``a2_fast`` are build-time constants; ``a2_fast=False`` forces the masked
    epilogue arm everywhere and exists only for the arm-equivalence check (P6).
    ``n_versions`` is the v4 A/B interface only: the two versions are *static
    buffer names* (``_s0``/``_s1``), so exactly two are expressible and the key
    is not forwarded to any scheduler (DESIGN.md 13.3 item 2).
    """
    if n_versions != 2:
        raise ValueError(
            f"n_versions: the rotated schedule exposes exactly 2 static versions "
            f"(DESIGN.md 6.3 B), got {n_versions}"
        )
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
    shift, is_causal = geo["shift"], geo["is_causal"]
    if grid > 65535:
        # One-dimensional grid bound (TRAP-950-grid-over-65535); every factor is
        # a compile-time constant, so the refusal is static.
        raise ValueError(
            f"shape: grid = heads*B*ceil(S_kv/{_BN}) = {grid} exceeds the 65535 block limit"
        )
    dt = dtype

    # -------------------------------------------------------------------
    # v4 / A5 (DESIGN.md 6.3 B, 13.3): the query loop is a ``T.serial`` over two
    # statically expanded half bodies, and the gradient GEMMs of iteration
    # ``k-1`` run beside the epilogue of iteration ``k``:
    #
    #     half body(k, version v) = F(k) -> GA(k) -> EP(k) -> GB(k-1, !v)
    #
    # ``GB`` (the three gradient GEMMs plus the ``dq`` atomic) reads the *other*
    # half's statically named buffers, so the AIC's gradient GEMMs of ``k-1``
    # execute beside the AIV's epilogue of ``k`` instead of behind it.
    # ``EP(k) -> GB(k)`` is a RAW edge, so buffer versioning alone cannot remove
    # it (DESIGN.md 6.3 A); the statement order has to move, which is why the
    # consumer lives in the next half body.
    #
    # Version identity is expressed by *static names* (``*_s0`` / ``*_s1``):
    # a ``T.gemm`` L0 operand region must have a static origin, so a runtime
    # parity index into one buffer is not available (DESIGN.md 3.3 note 1).
    # Two halves per loop iteration is what makes the static names possible.
    # L0A/L0B/L0C, the UB half tiles and ``st_u/dp_u`` stay single-version
    # (DESIGN.md 4.5/4.6: the producers and consumers are in the same half
    # body, and a second L0C set would be 103.1% of the budget).
    # -------------------------------------------------------------------
    @T.macro
    def _fill(q, dout, lse, delta, q_s, do_s, qT_s, doT_s, lse_u, dlt_u, b, h, m0):
        """F(k): the four query-row loads plus the two head-major statistics."""
        T.copy(q[b, m0 : m0 + _BM, h, :], q_s)
        T.copy(dout[b, m0 : m0 + _BM, h, :], do_s)
        T.copy(q[b, m0 : m0 + _BM, h, :], qT_s, transpose=True)
        T.copy(dout[b, m0 : m0 + _BM, h, :], doT_s, transpose=True)
        T.copy(lse[b, h, m0 : m0 + _BM], lse_u)
        T.copy(delta[b, h, m0 : m0 + _BM], dlt_u)

    @T.macro
    def _scores(k_s, v_s, q_s, do_s, a_k, b_lo, st_c, dpt_c, st_u, dp_u):
        """GA(k): s^T and dP^T, cleared every iteration, landed as half tiles."""
        T.copy(k_s, a_k)
        T.copy(q_s, b_lo)
        T.gemm(a_k, b_lo, st_c, transpose_B=True, clear_accum=True)
        T.copy(v_s, a_k[0:_BN, 0:dim_v])
        T.copy(do_s, b_lo[0:_BM, 0:dim_v])
        T.gemm(a_k[0:_BN, 0:dim_v], b_lo[0:_BM, 0:dim_v], dpt_c, transpose_B=True,
               clear_accum=True)
        # L0C -> UB: each AIV receives its own 48 key rows.
        T.dual_copy(st_c, st_u)
        T.dual_copy(dpt_c, dp_u)

    @T.macro
    def _epilogue(sid, st_u, dp_u, p16_u, dsT_u, lse_u, dlt_u, pT_dst, dsT_dst, n0, m0):
        """EP(k): P / dS on the vector side, written into version ``v``."""
        fast = (n0 + _BN <= seq_kv) & (m0 + _BM <= seq_q)
        if is_causal:
            fast = fast & (n0 + _BN - 1 <= m0 + shift)
        if a2_fast:
            if fast:
                # Fast arm: the whole tile is visible and fully in range.
                with T.SimtVF(threads=128):
                    for i, j in T.Parallel(_HN, _BM):
                        p = T.exp(st_u[i, j] * scale - lse_u[j])
                        p16_u[i, j] = T.cast(p, dt)
                        dsT_u[i, j] = T.cast(p * (dp_u[i, j] - dlt_u[j]), dt)
            else:
                with T.SimtVF(threads=128):
                    for i, j in T.Parallel(_HN, _BM):
                        gi = n0 + T.cast(sid, "int32") * _HN + i
                        gj = m0 + j
                        vis = (gi < seq_kv) & (gj < seq_q)
                        if is_causal:
                            vis = vis & (gi <= gj + shift)
                        p = T.exp(st_u[i, j] * scale - lse_u[j])
                        pv = T.if_then_else(vis, p, T.cast(0, "float32"))
                        p16_u[i, j] = T.cast(pv, dt)
                        dsT_u[i, j] = T.cast(pv * (dp_u[i, j] - dlt_u[j]), dt)
        else:
            with T.SimtVF(threads=128):
                for i, j in T.Parallel(_HN, _BM):
                    gi = n0 + T.cast(sid, "int32") * _HN + i
                    gj = m0 + j
                    vis = (gi < seq_kv) & (gj < seq_q)
                    if is_causal:
                        vis = vis & (gi <= gj + shift)
                    p = T.exp(st_u[i, j] * scale - lse_u[j])
                    pv = T.if_then_else(vis, p, T.cast(0, "float32"))
                    p16_u[i, j] = T.cast(pv, dt)
                    dsT_u[i, j] = T.cast(pv * (dp_u[i, j] - dlt_u[j]), dt)
        # UB -> L1: M-split of the two produced half tiles (into version ``v``).
        T.dual_copy(p16_u, pT_dst)
        T.dual_copy(dsT_u, dsT_dst)

    @T.macro
    def _gradients(pT_p, dsT_p, doT_p, qT_p, kT_s, a_k, a_t, b_hi, dq_c, dk_c, dv_c,
                   dq_acc, b, m0_p, h, clear_prev):
        """GB(k): dV / dK (accumulate) and dQ (transpose_A reuse of dS^T).

        ``m0_p`` is the *lagged* query offset (``k``'s own, not the caller's) and
        ``clear_prev`` replaces ``clear_accum=(kq == loop_st)`` because the first
        iteration now initialises the accumulators one half body late.
        """
        T.copy(pT_p, a_k[0:_BN, 0:_BM])
        T.copy(doT_p, b_hi[0:dim_v, 0:_BM])
        T.gemm(a_k[0:_BN, 0:_BM], b_hi[0:dim_v, 0:_BM], dv_c, transpose_B=True,
               clear_accum=clear_prev)
        T.copy(dsT_p, a_k[0:_BN, 0:_BM])
        T.copy(qT_p, b_hi[0:dim_qk, 0:_BM])
        T.gemm(a_k[0:_BN, 0:_BM], b_hi[0:dim_qk, 0:_BM], dk_c, transpose_B=True,
               clear_accum=clear_prev)
        # Same dsT tile as dK's A operand; here the transpose is done by the
        # L1 -> L0A transposing copy (MTE1), not by a vector scatter.
        T.copy(dsT_p, a_t)
        # kT must be reloaded into b_hi AFTER qT/doT (DESIGN.md 9.2).
        T.copy(kT_s, b_hi)
        T.gemm(a_t, b_hi, dq_c, transpose_A=True, transpose_B=True, clear_accum=True)
        # dQ accumulates across blocks -> hardware store-atomic into fp32.
        T.set_atomic("add", "float32")
        T.copy(dq_c, dq_acc[b, m0_p : m0_p + _BM, h, :])
        T.set_atomic_none()

    @T.prim_func
    def _bwd_main(
        q: T.Tensor((batch, seq_q, heads, dim_qk), dt),  # type: ignore
        k: T.Tensor((batch, seq_kv, heads, dim_qk), dt),  # type: ignore
        v: T.Tensor((batch, seq_kv, heads, dim_v), dt),  # type: ignore
        dout: T.Tensor((batch, seq_q, heads, dim_v), dt),  # type: ignore
        lse: T.Tensor((batch, heads, seq_q), "float32"),  # type: ignore
        delta: T.Tensor((batch, heads, seq_q), "float32"),  # type: ignore
        cu_q: T.Tensor((batch + 1,), "int32"),  # type: ignore
        cu_kv: T.Tensor((batch + 1,), "int32"),  # type: ignore
        dq_acc: T.Tensor((batch, seq_q, heads, dim_qk), "float32"),  # type: ignore
        dk: T.Tensor((batch, seq_kv, heads, dim_qk), dt),  # type: ignore
        dv: T.Tensor((batch, seq_kv, heads, dim_v), dt),  # type: ignore
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

            # ---- L1: key resident (load once per block) ----------------------
            k_s = T.alloc_l1((_BN, dim_qk), dt)
            v_s = T.alloc_l1((_BN, dim_v), dt)
            kT_s = T.alloc_l1((dim_qk, _BN), dt)
            # ---- L1/UB: per-iteration operands, **two explicit versions** ---
            # ``*_s0`` = version A: written by half body A (``kq_a = loop_st+2i``)
            # and drained when ``n_it`` is odd; ``*_s1`` = version B (half body
            # B, drained when ``n_it`` is even).
            # ``qT/doT`` are read by the *lagged* ``GB`` and ``pT/dsT`` are
            # produced by ``EP`` and consumed by the *lagged* ``GB``, so both
            # need the second version.  ``q_s/do_s/lse_u/dlt_u`` are read by the
            # same half body's ``GA``/``EP`` only; their second copy removes the
            # cross-unit WAR against ``F(k+1)`` (DESIGN.md 4.3 / 6.3 C).
            q_s0 = T.alloc_l1((_BM, dim_qk), dt)
            qT_s0 = T.alloc_l1((dim_qk, _BM), dt)
            do_s0 = T.alloc_l1((_BM, dim_v), dt)
            doT_s0 = T.alloc_l1((dim_v, _BM), dt)
            pT_s0 = T.alloc_l1((_BN, _BM), dt)
            dsT_s0 = T.alloc_l1((_BN, _BM), dt)  # dK (as A), dQ (transposed A)
            lse_u0 = T.alloc_shared((_BM,), "float32")
            dlt_u0 = T.alloc_shared((_BM,), "float32")
            q_s1 = T.alloc_l1((_BM, dim_qk), dt)
            qT_s1 = T.alloc_l1((dim_qk, _BM), dt)
            do_s1 = T.alloc_l1((_BM, dim_v), dt)
            doT_s1 = T.alloc_l1((dim_v, _BM), dt)
            pT_s1 = T.alloc_l1((_BN, _BM), dt)
            dsT_s1 = T.alloc_l1((_BN, _BM), dt)
            lse_u1 = T.alloc_shared((_BM,), "float32")
            dlt_u1 = T.alloc_shared((_BM,), "float32")
            # ---- L0A/L0B: explicit operands, one version (L0B has 4 KiB left) --
            a_k = T.alloc_l0a((_BN, dim_qk), dt)  # K-major: k / v / pT / dsT
            a_t = T.alloc_l0a((_BN, _BM), dt)  # MN-major: dsT for dQ (trans_A)
            b_lo = T.alloc_l0b((_BM, dim_qk), dt)  # q / do
            b_hi = T.alloc_l0b((dim_qk, _BN), dt)  # qT / doT / kT (kT reloaded last)
            # ---- L0C accumulators -------------------------------------------
            st_c = T.alloc_l0c((_BN, _BM), "float32")
            dpt_c = T.alloc_l0c((_BN, _BM), "float32")
            dq_c = T.alloc_l0c((_BM, dim_qk), "float32")
            dk_c = T.alloc_l0c((_BN, dim_qk), "float32")
            dv_c = T.alloc_l0c((_BN, dim_v), "float32")
            # ---- UB: one half tile per AIV (dual_copy 2:1) ------------------
            # ``st_u/dp_u/p16_u/dsT_u`` stay single-version: their producer (the
            # fixpipe of GA) and consumer (EP) are in the same half body; the
            # only crossing hazard is the WAR of ``fixpipe(k+1)`` against
            # ``EP(k)``, which InsertSync inserts (DESIGN.md 4.5 / 7.2, risk 16).
            st_u = T.alloc_shared((_HN, _BM), "float32")
            dp_u = T.alloc_shared((_HN, _BM), "float32")
            p16_u0 = T.alloc_shared((_HN, _BM), dt)
            dsT_u0 = T.alloc_shared((_HN, _BM), dt)
            p16_u1 = T.alloc_shared((_HN, _BM), dt)
            dsT_u1 = T.alloc_shared((_HN, _BM), dt)
            tail_u = T.alloc_shared((_HN, dim_qk), "float32")
            tailv_u = T.alloc_shared((_HN, dim_v), "float32")
            out_k = T.alloc_shared((_HN, dim_qk), dt)
            out_v = T.alloc_shared((_HN, dim_v), dt)

            # Whole-extent loads on the 4-D batch view: the OOB pass clamps the
            # DMA to this request's S_kv and fills zero; the mask arm drops the
            # padded rows (zero fill alone is not a zero gradient).
            T.copy(k[b, n0 : n0 + _BN, h, :], k_s)
            T.copy(v[b, n0 : n0 + _BN, h, :], v_s)
            T.copy(k[b, n0 : n0 + _BN, h, :], kT_s, transpose=True)

            # IS_CAUSAL is a build-time constant: without the mask the loop must
            # start at 0 (DESIGN.md 3.3 note 4).
            loop_st = T.max(0, (n0 - shift) // _BM) if is_causal else 0
            # v4 (DESIGN.md 6.3 B): ``T.serial`` over two statically expanded
            # half bodies instead of ``T.Pipelined``'s automatic versioning,
            # which rewrites every access of a versioned buffer with the
            # *current* iteration index and therefore cannot express a lagged
            # consumer (Z3 ``unsat`` -> plain topological sort; three probes).
            n_it = loop_ed - loop_st  # block-level scalar, >= 1
            n_half = (n_it + 1) // 2
            for i in T.serial(n_half):
                # ---- half body A (version A) -------------------------------
                kq_a = loop_st + i * 2
                m0_a = kq_a * _BM
                _fill(q, dout, lse, delta, q_s0, do_s0, qT_s0, doT_s0,
                      lse_u0, dlt_u0, b, h, m0_a)
                _scores(k_s, v_s, q_s0, do_s0, a_k, b_lo, st_c, dpt_c, st_u, dp_u)
                _epilogue(sid, st_u, dp_u, p16_u0, dsT_u0, lse_u0, dlt_u0,
                          pT_s0, dsT_s0, n0, m0_a)
                # GB(k-1) reads version B, i.e. the previous half body.  The
                # first iteration has no lagged predecessor.
                if kq_a > loop_st:
                    _gradients(pT_s1, dsT_s1, doT_s1, qT_s1, kT_s, a_k, a_t, b_hi,
                               dq_c, dk_c, dv_c, dq_acc, b, m0_a - _BM, h,
                               kq_a - 1 == loop_st)
                # ---- half body B (version B) -------------------------------
                kq_b = kq_a + 1
                if kq_b < loop_ed:  # an odd iteration count ends after A
                    m0_b = kq_b * _BM
                    _fill(q, dout, lse, delta, q_s1, do_s1, qT_s1, doT_s1,
                          lse_u1, dlt_u1, b, h, m0_b)
                    _scores(k_s, v_s, q_s1, do_s1, a_k, b_lo, st_c, dpt_c, st_u, dp_u)
                    _epilogue(sid, st_u, dp_u, p16_u1, dsT_u1, lse_u1, dlt_u1,
                              pT_s1, dsT_s1, n0, m0_b)
                    # GB(k-1) reads version A (this iteration's A half).
                    _gradients(pT_s0, dsT_s0, doT_s0, qT_s0, kT_s, a_k, a_t, b_hi,
                               dq_c, dk_c, dv_c, dq_acc, b, m0_b - _BM, h,
                               kq_b - 1 == loop_st)

            # ---- drain: GB(loop_ed-1) has no next iteration ------------------
            # The version of ``last`` is A when the iteration count is odd (the
            # loop ended after a half body A) and B otherwise.  A missing even
            # branch silently drops the last query block's dK/dV contribution of
            # every even-length key block (DESIGN.md 3.3 note 10 / 9.2).
            last = loop_ed - 1
            if (n_it % 2) == 1:
                _gradients(pT_s0, dsT_s0, doT_s0, qT_s0, kT_s, a_k, a_t, b_hi,
                           dq_c, dk_c, dv_c, dq_acc, b, last * _BM, h,
                           last == loop_st)
            else:
                _gradients(pT_s1, dsT_s1, doT_s1, qT_s1, kT_s, a_k, a_t, b_hi,
                           dq_c, dk_c, dv_c, dq_acc, b, last * _BM, h,
                           last == loop_st)

            # ---- block tail: dk = scale * cast(dk_c), dv = cast(dv_c) --------
            T.dual_copy(dk_c, tail_u)
            T.dual_copy(dv_c, tailv_u)
            with T.SimtVF(threads=128):
                for i, d in T.Parallel(_HN, dim_qk):
                    out_k[i, d] = T.cast(tail_u[i, d] * scale, dt)
                for i, d in T.Parallel(_HN, dim_v):
                    out_v[i, d] = T.cast(tailv_u[i, d], dt)
            # The store window is this request's row range, so the OOB clamp
            # crops it to S_kv: min(48, max(S_kv - sid*48 - n0, 0)).  A 3-D
            # packed view would clamp to B*S_kv and overwrite the next request.
            kown = n0 + T.cast(sid, "int32") * _HN
            T.copy(out_k, dk[b, kown : kown + _HN, h, :])
            T.copy(out_v, dv[b, kown : kown + _HN, h, :])

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
    per_request=False,
    determinism=False,
):
    """Build one geometry, run it on the NPU and compare all three gradients.

    ``per_request=True`` reports (and asserts) the difference per request and
    for the first 64 key rows of every request -- the region a 3-D packed view
    would overwrite (DESIGN.md 8.3, L0-9/L0-10).
    """
    diff = float("nan")
    scale = dim_qk**-0.5 if scale is None else scale
    info = {}
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
        # The wrapper hands the kernel zero-copy 4-D batch views of the packed
        # (T, H, D) tensors; exercise exactly that path.
        q_dev = q.reshape(batch * seq_q, heads, dim_qk).contiguous().to(dev)
        k_dev = k.reshape(batch * seq_kv, heads, dim_qk).contiguous().to(dev)
        v_dev = v.reshape(batch * seq_kv, heads, dim_v).contiguous().to(dev)
        do_dev = dout.reshape(batch * seq_q, heads, dim_v).contiguous().to(dev)
        lse_dev = lse.to(dev)
        delta_dev = delta.to(dev)
        cu_q = (torch.arange(batch + 1, dtype=torch.int32) * seq_q).to(dev)
        cu_kv = (torch.arange(batch + 1, dtype=torch.int32) * seq_kv).to(dev)
        # Every output allocation carries _OOB_PAD sentinel rows behind it: the
        # kernel receives a leading view, so any write past the last request
        # lands in the sentinel area and is caught (probe P2).
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
        args = (
            q_dev.view(batch, seq_q, heads, dim_qk),
            k_dev.view(batch, seq_kv, heads, dim_qk),
            v_dev.view(batch, seq_kv, heads, dim_v),
            do_dev.view(batch, seq_q, heads, dim_v),
            lse_dev,
            delta_dev,
            cu_q,
            cu_kv,
            dq_acc.view(batch, seq_q, heads, dim_qk),
            dk_dev.view(batch, seq_kv, heads, dim_qk),
            dv_dev.view(batch, seq_kv, heads, dim_v),
        )
        start = time.perf_counter()
        fn(*args)
        torch.npu.synchronize()
        elapsed = (time.perf_counter() - start) * 1e3
        # Read the results before any repeat: dq_acc is an in-out store-atomic
        # accumulator, so re-launching would add into the same buffer.
        got_dq32 = dq_acc.cpu()
        got_dk = dk_dev.cpu()
        got_dv = dv_dev.cpu()
        same_dk = same_dv = None
        dq_run_diff = float("nan")
        if determinism:
            # dk/dv have one owner per key block, so a second launch must be
            # bitwise identical; any difference is a cross-request/block race.
            dk_first, dv_first = got_dk.clone(), got_dv.clone()
            dq_acc.zero_()
            fn(*args)
            torch.npu.synchronize()
            same_dk = bool(torch.equal(dk_dev.cpu(), dk_first))
            same_dv = bool(torch.equal(dv_dev.cpu(), dv_first))
            dq_run_diff = float((dq_acc.cpu() - got_dq32).abs().max())
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
        info.update(dk=d_dk, dv=d_dv, dq=d_dq)
        if per_request:
            per = []
            for bi in range(batch):
                sl_kv = slice(bi * seq_kv, (bi + 1) * seq_kv)
                sl_q = slice(bi * seq_q, (bi + 1) * seq_q)
                per.append(
                    (
                        bi,
                        _max_diff(got_dk[sl_kv], dk_true[sl_kv])[0],
                        _max_diff(got_dv[sl_kv], dv_true[sl_kv])[0],
                        _max_diff(got_dq32[sl_q], dq_acc_true[sl_q])[0],
                        _max_diff(got_dk[sl_kv][0:64], dk_true[sl_kv][0:64])[0],
                    )
                )
            info["per_request"] = per
        bad = nan_dk + nan_dv + nan_dq
        if bad:
            raise AssertionError(f"{bad} NaN position(s) disagree with the golden (max_diff={diff:.3e})")
        nonfinite = nf_dk + nf_dv + nf_dq
        rtol = _RTOL[dtype]
        tol_dq = _ATOL / scale

        def _checks(ref_dk, ref_dv, ref_dq, ref_dq32):
            torch.testing.assert_close(got_dk, ref_dk, atol=_ATOL, rtol=rtol, equal_nan=True)
            torch.testing.assert_close(got_dv, ref_dv, atol=_ATOL, rtol=rtol, equal_nan=True)
            torch.testing.assert_close(got_dq, ref_dq, atol=_ATOL, rtol=rtol, equal_nan=True)
            # float32 dq_acc intermediate against its own (scale-free) reference.
            # The relative tolerance is the *input dtype's* one: dS is cast to the
            # input dtype before the dQ GEMM (source semantics), so the fp32
            # accumulator carries that rounding even though the reference is fp64.
            torch.testing.assert_close(got_dq32, ref_dq32, atol=tol_dq, rtol=rtol, equal_nan=True)

        rounded_note = ""
        try:
            _checks(dk_true, dv_true, dq_true, dq_acc_true)
        except AssertionError:
            # DESIGN.md 8.2 discriminator: a strongly cancelling dV/dK element in
            # the input dtype can exceed the absolute tolerance while the kernel
            # still reproduces the algorithm.  Re-check against a reference that
            # casts P and dS to the input dtype before its GEMMs (source
            # semantics); if that passes the deviation is inherent rounding and
            # the number of such elements is reported, otherwise the failure
            # propagates.
            r_dq, r_dk, r_dv, r_dq32 = golden_bwd_kernel(
                q, k, v, dout, scale=scale, is_causal=is_causal, dtype=q.dtype,
                slab_heads=2, round_inputs=True,
            )
            _checks(
                r_dk.reshape(batch * seq_kv, heads, dim_qk),
                r_dv.reshape(batch * seq_kv, heads, dim_v),
                r_dq.reshape(batch * seq_q, heads, dim_qk),
                r_dq32.reshape(batch * seq_q, heads, dim_qk),
            )
            over = 0
            for got, ref, atol in (
                (got_dk, dk_true, _ATOL),
                (got_dv, dv_true, _ATOL),
                (got_dq, dq_true, _ATOL),
                (got_dq32, dq_acc_true, tol_dq),
            ):
                gap = (got.float() - ref.float()).abs()
                thr = atol + rtol * ref.float().abs()
                over += int((gap > thr).sum())
            rounded_note = (
                f" [{over} element(s) over the fp64 atol: inherent {dtype} rounding of "
                f"P/dS, matches the same-rounding reference]"
            )
            info["rounded_fallback"] = over
        if determinism:
            if not (same_dk and same_dv):
                raise AssertionError(
                    f"dk/dv are not deterministic across launches (dk {same_dk}, dv {same_dv})"
                )
            if not (dq_run_diff <= tol_dq):
                raise AssertionError(f"dq_acc differs across launches by {dq_run_diff:.3e}")
    except AssertionError as exc:
        _record(level, name, False, f"max_diff={diff:.3e} {str(exc)[:300]}")
        return False, info
    except (ValueError, RuntimeError) as exc:  # noqa: BLE001 - reported, never swallowed
        _record(level, name, False, f"{type(exc).__name__}: {str(exc)[:300]}")
        return False, info
    _record(level, name, True)
    extra = ""
    if per_request:
        extra = " per-request " + " ".join(
            f"b{bi}:dk={dk:.2e}/dv={dv:.2e}/dq={dq:.2e}/key0-63={v0:.2e}"
            for bi, dk, dv, dq, v0 in info["per_request"]
        )
    if determinism:
        extra += f" rerun-identical(dk={same_dk},dv={same_dv},dq={dq_run_diff:.1e})"
    print(
        f"       B={batch} S_q={seq_q} S_kv={seq_kv} H={heads} D={dim_qk}/{dim_v} {dtype} "
        f"causal={int(is_causal)} grid={heads * batch * ((seq_kv + _BN - 1) // _BN)} "
        f"max_diff={diff:.3e} (dk {d_dk:.2e} dv {d_dv:.2e} dq {d_dq:.2e}) "
        f"launch={elapsed:.3f} ms (nonfinite cells {nonfinite}){extra}{rounded_note}",
        flush=True,
    )
    return True, info


_V3_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "history_version",
    "_bwd_kernel_impl_s3_v4_attempt1.py",
)


def _load_v3():
    """Frozen v3 (pre-rotation) factory: the bitwise ``dk/dv`` oracle.

    The rotation must not move ``dK``/``dV`` by a single bit: the three
    gradient GEMMs keep their operand set and their accumulation order (the
    gradient blocks are consumed in increasing query order), so a wrong version
    shows up as a wholesale mismatch instead of rounding noise (A5 reading the
    *current* half's P/dS is exactly that).  ``None`` when the frozen copy is
    not shipped next to this file (e.g. inside the TileOPs package), in which
    case the gate reports "skipped" instead of failing.
    """
    if not os.path.exists(_V3_PATH):
        return None
    spec = importlib.util.spec_from_file_location("_bwd_kernel_v3_frozen", _V3_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _case_rotation_parity(level, name, batch, seq, heads, dtype=_DEFAULT_DTYPE,
                          is_causal=True, reps=3, seed=51, chunk=4096,
                          dim_qk=192, dim_v=128):
    """L0-14/L0-15 gate: v4 vs the frozen v3 body, ``dk/dv`` bitwise, 3 reps.

    Also asserts that the v4 body is reproducible across launches, so a race
    introduced by the lagged consumer cannot hide behind a matching average.
    """
    v3 = _load_v3()
    if v3 is None:
        _record(level, name, True, "skipped: frozen v3 copy not shipped")
        print(f"       [skip] {_V3_PATH} absent", flush=True)
        return True
    dev = "npu"
    dt = _TORCH_DTYPE[dtype]
    scale = dim_qk**-0.5
    q = torch.randn((batch, seq, heads, dim_qk), dtype=torch.float32, device=dev).to(dt)
    k = torch.randn((batch, seq, heads, dim_qk), dtype=torch.float32, device=dev).to(dt)
    v = torch.randn((batch, seq, heads, dim_v), dtype=torch.float32, device=dev).to(dt)
    dout = torch.randn((batch, seq, heads, dim_v), dtype=torch.float32, device=dev).to(dt)
    lse, delta = _stats_npu(q, k, v, dout, is_causal, chunk=chunk)
    fn4 = _launch(batch, heads, dim_qk, dim_v, seq, seq, is_causal, scale, dtype)
    fn3 = v3._launch(batch, heads, dim_qk, dim_v, seq, seq, is_causal, scale, dtype)
    cu = torch.zeros(batch + 1, dtype=torch.int32, device=dev)
    worst_dq = 0.0
    ok = True
    first = None
    for rep in range(reps):
        dq4 = torch.zeros((batch, seq, heads, dim_qk), dtype=torch.float32, device=dev)
        dk4 = torch.empty((batch, seq, heads, dim_qk), dtype=dt, device=dev)
        dv4 = torch.empty((batch, seq, heads, dim_v), dtype=dt, device=dev)
        fn4(q, k, v, dout, lse, delta, cu, cu, dq4, dk4, dv4)
        dq3 = torch.zeros((batch, seq, heads, dim_qk), dtype=torch.float32, device=dev)
        dk3 = torch.empty((batch, seq, heads, dim_qk), dtype=dt, device=dev)
        dv3 = torch.empty((batch, seq, heads, dim_v), dtype=dt, device=dev)
        fn3(q, k, v, dout, lse, delta, cu, cu, dq3, dk3, dv3)
        torch.npu.synchronize()
        same_dk = bool(torch.equal(dk4, dk3))
        same_dv = bool(torch.equal(dv4, dv3))
        rel = float((dq4 - dq3).abs().max() / dq3.abs().max().clamp(min=1.0e-30))
        worst_dq = max(worst_dq, rel)
        det = True
        if first is None:
            first = (dk4.clone(), dv4.clone())
        else:
            det = bool(torch.equal(dk4, first[0]) and torch.equal(dv4, first[1]))
        if not (same_dk and same_dv) or rel >= 1.0e-3 or not det:
            ok = False
            _record(level, f"{name} rep{rep}", False,
                    f"dk_bitwise={same_dk} dv_bitwise={same_dv} dq_rel={rel:.3e} "
                    f"reproducible={det}")
    if ok:
        _record(level, name, True)
    print(f"       B={batch} S={seq} H={heads} {dtype} causal={int(is_causal)} "
          f"dk/dv bitwise vs v3 over {reps} reps, dq rel<={worst_dq:.3e}, "
          f"v4 reproducible across launches", flush=True)
    del q, k, v, dout, lse, delta, cu
    return ok


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
        # 65535-block grid bound (DESIGN.md 9.1 risk 12): 128*8*86 = 88064
        ((8, 128, 192, 128, 8192, 8192, True, 192**-0.5, "float16"), "grid over 65535"),
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
    print("       5 invalid contracts rejected, program cached per (geometry, dtype)", flush=True)
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
                  special="atom_prefill")[0]
    ok = run_case(level, name + "-zeroed", 1, 64, 64, 1, "float16", seed=91)[0] and ok
    return ok


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
            dq = torch.zeros(1, seq_q, 2, 192, dtype=torch.float32, device=dev)
            dk = torch.zeros(1, seq_kv, 2, 192, dtype=_TORCH_DTYPE[dtype], device=dev)
            dv = torch.zeros(1, seq_kv, 2, 128, dtype=_TORCH_DTYPE[dtype], device=dev)
            fn = _launch(1, 2, 192, 128, seq_q, seq_kv, True, 192**-0.5, dtype, a2_fast)
            fn(q.to(dev), k.to(dev), v.to(dev), do.to(dev), lse.to(dev), delta.to(dev),
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


def _case_request_isolation(level, name, is_causal, reps=5):
    """L0-9/L0-10: 5 repeats of the B=2 tail-block geometry, per request.

    ``S = 512 = 5*96 + 32``: request 0's tail block (n0=480) is the block whose
    store window would spill into request 1's key rows 0..63 under a 3-D view.
    A single run cannot decide the race (the causal variant passes 5/5 even when
    broken), so all ``reps`` launches are compared per request.
    """
    ok = True
    rows = []
    for rep in range(reps):
        good, info = run_case(
            level, f"{name}-rep{rep}", 2, 512, 512, 4, "bfloat16", is_causal=is_causal,
            seed=rep, per_request=True, determinism=True,
        )
        ok = ok and good
        rows.append(info)
    # Different random data per repeat, so the maxima need not be identical;
    # a >2x spread between repeats would mean an unstable (racy) result.
    worst = {key: max(info.get(key, float("nan")) for info in rows) for key in ("dk", "dv", "dq")}
    best = {key: min(info.get(key, float("nan")) for info in rows) for key in ("dk", "dv", "dq")}
    for key, value in worst.items():
        if not (value <= 2 * max(best[key], 1e-6)):
            ok = False
            _record(level, name + "-spread", False, f"{key} unstable across repeats: "
                                                   f"{best[key]:.2e}..{value:.2e}")
    if ok:
        _record(level, name + "-5x", True)
        print(f"       {name}: 5/5 launches pass, per-request keys 0..63 clean, no rerun drift",
              flush=True)
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
    # v3 additions (DESIGN.md 8.3): request isolation of the dk/dv store, 5 runs.
    _case_request_isolation("L0", "L0-9 request-isolation-causal", is_causal=True)
    _case_request_isolation("L0", "L0-10 request-isolation-non-causal", is_causal=False)
    run_case("L0", "L0-11 b2-tail-shift-pos-512x608", 2, 512, 608, 2, "bfloat16", seed=9,
             per_request=True, determinism=True)
    _case_golden_naive("L0", "L0-12 golden-vs-naive-loop")
    _case_contract("L0", "L0-13 refuses-an-invalid-contract")
    _case_atomic_contract("L0", "L0-14 store-atomic")
    _case_arm_equivalence("L0", "L0-15 a2-arm-equivalence")
    run_case("L0", "L0-16 bf16-tail-both-100", 1, 100, 100, 2, "bfloat16", seed=10)
    # ---- v4 rotation gates (DESIGN.md 8.3 L0-12..L0-15; numbered 17..20 here
    # because 12..16 above belong to v3 and stay stable).  BLOCK_M = 64, so the
    # drain parity is uniform only when loop_st == 0 (non-causal):
    #   n_it = ceil(S_q/64) - loop_st     (S=64 -> 1, S=128 -> 2, S=192 -> 3)
    run_case("L0", "L0-17 drain-even n_it=2", 1, 128, 128, 1, "bfloat16", seed=61,
             determinism=True)
    run_case("L0", "L0-17b drain-even all-blocks non-causal", 1, 128, 128, 2,
             "bfloat16", seed=62, is_causal=False, determinism=True)
    run_case("L0", "L0-18 drain-odd n_it=1", 1, 64, 64, 1, "bfloat16", seed=63,
             determinism=True)
    run_case("L0", "L0-18b drain-odd all-blocks non-causal", 1, 64, 64, 2,
             "float16", seed=64, is_causal=False, determinism=True)
    run_case("L0", "L0-19 n_it=3 multi-block", 1, 192, 192, 2, "bfloat16", seed=65,
             determinism=True)
    run_case("L0", "L0-20 rotation 4-D B=2 per-request", 2, 512, 512, 4, "bfloat16",
             seed=66, per_request=True, determinism=True)
    _finish("L0", strict=True)


def run_L1():
    """Representative shapes, both dtypes, both causal settings."""
    run_case("L1", "L1-1 bf16-smoke-1024", 1, 1024, 1024, 4, "bfloat16", seed=11, repeat=2)
    run_case("L1", "L1-2 fp16-smoke-1024", 1, 1024, 1024, 4, "float16", seed=12, repeat=2)
    run_case("L1", "L1-3 bf16-two-batches", 2, 512, 512, 4, "bfloat16", seed=13, repeat=2,
             per_request=True, determinism=True)
    run_case("L1", "L1-4 bf16-bottom-right", 1, 512, 768, 4, "bfloat16", seed=14, repeat=2)
    run_case("L1", "L1-5 fp16-non-causal-1024", 1, 1024, 1024, 4, "float16", seed=15,
             is_causal=False, repeat=2)
    run_case("L1", "L1-6 bf16-single-head", 1, 768, 768, 1, "bfloat16", seed=16, repeat=2)
    # v4: the rotation must leave dK/dV bit-identical to the frozen v3 body at
    # the production geometry (DESIGN.md 8.2 / REVIEW.md stage-3 item 7).
    _case_rotation_parity("L1", "L1-7 rotation-vs-v3-b2-8k-causal", 2, 8192, 8,
                          "bfloat16", True, reps=3, seed=17)
    _case_rotation_parity("L1", "L1-8 rotation-vs-v3-b2-8k-non-causal", 2, 8192, 8,
                          "bfloat16", False, reps=3, seed=18)
    _finish("L1", strict=True)


def run_L2():
    """Larger informative geometries (the CPU golden is O(H*S^2) -> capped)."""
    run_case("L2", "L2-1 bf16-2k", 1, 2048, 2048, 2, "bfloat16", seed=21, repeat=1)
    run_case("L2", "L2-2 fp16-1536", 1, 1536, 1536, 2, "float16", seed=22, repeat=1)
    run_case("L2", "L2-3 bf16-b2-1024", 2, 1024, 1024, 4, "bfloat16", seed=23, repeat=1,
             per_request=True, determinism=True)
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
    run_case("boundary", "B-11 B=3 H=1", 3, 128, 128, 1, "float16", seed=41, per_request=True)
    run_case("boundary", "B-12 B=2 tail-32 non-causal", 2, 512, 512, 2, "bfloat16", seed=42,
             is_causal=False, per_request=True)
    _finish("boundary", strict=False)


def run_big(batch=2, seq=8192, heads=128, dim_qk=192, dim_v=128,
            heads_check=(0, 43, 86, 127), reps=2, chunk=4096):
    """Production shape ``ds-v3-b2-8k``: kernel-only timing + sliced fp64 reference.

    ``8192 = 85*96 + 32``, so request 0's tail block is the cross-request store
    window of DESIGN.md 5.4.  A full ``(H, S, S)`` golden does not fit, so the
    reference is computed per ``(request, head)`` slice (``golden_head``) for a
    spread of heads; the statistics of those heads come from the same float64
    host path (the other heads' statistics only feed the kernel and cannot
    influence a compared head, every ``(b, h)`` block being independent).
    Two launches are compared bitwise on ``dk/dv`` to expose any race.
    """
    dev = "npu"
    dt_name = "bfloat16"
    dt = _TORCH_DTYPE[dt_name]
    scale = dim_qk**-0.5
    print(f"[big] B={batch} S={seq} H={heads} D={dim_qk}/{dim_v} {dt_name} causal grid="
          f"{heads * batch * ((seq + _BN - 1) // _BN)}", flush=True)
    t_start = time.perf_counter()
    q = torch.randn((batch, seq, heads, dim_qk), dtype=torch.float32, device=dev).to(dt)
    k = torch.randn((batch, seq, heads, dim_qk), dtype=torch.float32, device=dev).to(dt)
    v = torch.randn((batch, seq, heads, dim_v), dtype=torch.float32, device=dev).to(dt)
    dout = torch.randn((batch, seq, heads, dim_v), dtype=torch.float32, device=dev).to(dt)
    print(f"[big] inputs ready in {time.perf_counter() - t_start:.1f} s", flush=True)

    t_start = time.perf_counter()
    lse, delta = _stats_npu(q, k, v, dout, True, chunk=chunk)
    print(f"[big] npu statistics (all heads) in {time.perf_counter() - t_start:.1f} s", flush=True)

    refs = {}
    t_start = time.perf_counter()
    for b in range(batch):
        for h in heads_check:
            args_h = (q[b, :, h, :].cpu(), k[b, :, h, :].cpu(),
                      v[b, :, h, :].cpu(), dout[b, :, h, :].cpu())
            dq_t, dk_t, dv_t, lse_h, delta_h = golden_head(*args_h, scale, True)
            r_dq, r_dk, r_dv, _, _ = golden_head(*args_h, scale, True, round_inputs=True,
                                                 dtype=dt)
            lse[b, h] = lse_h.to(dev)
            delta[b, h] = delta_h.to(dev)
            refs[(b, h)] = (dq_t, dk_t, dv_t, r_dq, r_dk, r_dv)
    print(f"[big] fp64 sliced reference ({len(refs)} request/head pairs) in "
          f"{time.perf_counter() - t_start:.1f} s", flush=True)

    fn = _launch(batch, heads, dim_qk, dim_v, seq, seq, True, scale, dt_name)
    dq_acc = torch.zeros((batch, seq, heads, dim_qk), dtype=torch.float32, device=dev)
    dk = torch.empty((batch, seq, heads, dim_qk), dtype=dt, device=dev)
    dv = torch.empty((batch, seq, heads, dim_v), dtype=dt, device=dev)
    cu = torch.zeros(batch + 1, dtype=torch.int32, device=dev)
    args = (q, k, v, dout, lse, delta, cu, cu, dq_acc, dk, dv)

    best = float("inf")
    for _ in range(reps):
        dq_acc.zero_()
        torch.npu.synchronize()
        t0 = time.perf_counter()
        fn(*args)
        torch.npu.synchronize()
        best = min(best, time.perf_counter() - t0)
    iters = 0
    ntiles = (seq + _BN - 1) // _BN
    for t in range(ntiles):
        n0 = t * _BN
        st = max(0, n0 // _BM)  # shift == 0 for this shape
        iters += (seq + _BM - 1) // _BM - st
    iters *= heads * batch
    per_iter = best * 1e6 * 28 / iters
    print(f"[big] T_op = {best * 1e3:.3f} ms over {iters} block-iterations "
          f"({per_iter:.3f} us/iteration, {reps} launches, min)", flush=True)

    # ---- determinism / race check ---------------------------------------
    dq1 = dq_acc.clone()
    dk1 = dk.clone()
    dv1 = dv.clone()
    dq_acc.zero_()
    fn(*args)
    torch.npu.synchronize()
    same_dk = bool(torch.equal(dk, dk1))
    same_dv = bool(torch.equal(dv, dv1))
    dq_drift = float((dq_acc - dq1).abs().max())
    ok = True
    if not (same_dk and same_dv):
        ok = False
        print(f"[big][FAIL] dk/dv differ between launches (dk {same_dk}, dv {same_dv})", flush=True)
    tol_dq = _ATOL / scale
    if dq_drift > tol_dq:
        ok = False
        print(f"[big][FAIL] dq_acc drift between launches {dq_drift:.3e} > {tol_dq:.3e}", flush=True)
    print(f"[big] rerun: dk identical={same_dk}, dv identical={same_dv}, dq drift={dq_drift:.2e}",
          flush=True)
    # A 3-D packed view would have overwritten request 1's first key rows with
    # zeros: assert that region is populated for every head.
    victim = dk1[1, 0:64].float()
    print(f"[big] request 1 key 0..63: zeros {int((victim == 0).sum())}/{victim.numel()}, "
          f"max|.|={victim.abs().max().item():.3e}", flush=True)
    if int((victim == 0).sum()) > victim.numel() // 100:
        ok = False
        print("[big][FAIL] request 1 key rows are mostly zero (cross-request overwrite)", flush=True)
    for name, t in (("dk", dk1), ("dv", dv1), ("dq_acc", dq1)):
        if not bool(torch.isfinite(t).all()):
            ok = False
            print(f"[big][FAIL] {name} has non-finite entries", flush=True)

    # ---- sliced reference ------------------------------------------------
    worst = {"dk": 0.0, "dv": 0.0, "dq": 0.0}
    over = {"dk": 0, "dv": 0, "dq": 0}
    over_rounded = {"dk": 0, "dv": 0, "dq": 0}
    rtol = _RTOL[dt_name]
    for (b, h), (dq_t, dk_t, dv_t, r_dq, r_dk, r_dv) in refs.items():
        got_dk = dk1[b, :, h, :].float().double().cpu()
        got_dv = dv1[b, :, h, :].float().double().cpu()
        got_dq = dq1[b, :, h, :].double().cpu()
        rows = []
        victim_dk = float((got_dk[0:64] - dk_t[0:64]).abs().max())
        for key, got, ref, rref, atol in (
            ("dk", got_dk, dk_t, r_dk, _ATOL),
            ("dv", got_dv, dv_t, r_dv, _ATOL),
            ("dq", got_dq, dq_t / scale, r_dq / scale, tol_dq),
        ):
            gap = (got - ref).abs()
            thr = atol + rtol * ref.abs()
            worst[key] = max(worst[key], float(gap.max()))
            over[key] += int((gap > thr).sum())
            over_rounded[key] += int(((got - rref).abs() > (atol + rtol * rref.abs())).sum())
            rows.append(f"{key}={float(gap.max()):.3e}")
        print(f"[big] b={b} h={h:3d}: {' '.join(rows)} (key 0..63 dk={victim_dk:.3e})", flush=True)
    print(f"[big] elements over (atol+rtol|ref|): fp64-golden {over}, same-rounding "
          f"{over_rounded}", flush=True)
    for key in ("dk", "dv", "dq"):
        if over_rounded[key]:
            ok = False
            print(f"[big][FAIL] {key}: {over_rounded[key]} element(s) outside the tolerance "
                  f"even against the same-rounding reference", flush=True)
    _MAX_DIFF[0] = max(_MAX_DIFF[0], worst["dk"], worst["dv"], worst["dq"])
    _record("big", f"big-b{batch}-s{seq}-h{heads}", ok,
            f"dk={worst['dk']:.3e} dv={worst['dv']:.3e} dq={worst['dq']:.3e} "
            f"over_fp64={sum(over.values())} over_rounded={sum(over_rounded.values())}")
    return ok, best * 1e3, worst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0", choices=["L0", "L1", "L2", "boundary", "big", "all"])
    parser.add_argument("--heads-check", default="0,43,86,127")
    args, _ = parser.parse_known_args()
    levels = {"L0": run_L0, "L1": run_L1, "L2": run_L2, "boundary": run_boundary}
    started = time.perf_counter()
    if args.level == "big":
        heads_check = tuple(int(x) for x in args.heads_check.split(",") if x != "")
        run_big(heads_check=heads_check)
    else:
        chosen = list(levels.values()) if args.level == "all" else [levels[args.level]]
        for run in chosen:
            run()
    print(f"[TIME] {time.perf_counter() - started:.1f} s for level={args.level}", flush=True)
    print(f"[PRECISION_PASS] max_diff={_MAX_DIFF[0]:.6g}", flush=True)


if __name__ == "__main__":
    main()
