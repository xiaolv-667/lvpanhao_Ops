# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""_bwd_kernel (ascend): MLA prefill backward main pass (MLAVarlenBwdOp, packed/THD).

Given the packed forward tensors ``q, k, v`` and ``dout`` plus the per-query row
statistics ``lse`` (natural log, head-major) and ``delta = rowsum(o * dout)``,
this launch recomputes the attention probabilities and emits

    dV[n, :] = sum_m P[m, n] dout[m, :]
    dK[n, :] = scale * sum_m dS[m, n] q[m, :]      (fp32 accumulate, cast on write)
    dQ[m, :] = sum_n dS[m, n] k[n, :]              (fp32 in-place accumulate in
                                                    ``dq_acc``; ``_dq_land_kernel``
                                                    applies ``scale`` and the cast)

with ``P = exp(scale * q.k - lse)``, ``dP = dout.v``, ``dS = P * (dP - delta)``
and FA3 bottom-right causal alignment (request-local query ``m`` sees key
``n <= m + shift``, ``shift = len_kv - len_q``); every request only sees its own
keys, the packed offsets come from ``cu_seqlens_q/kv``.

Structure (key stationary, packed; DESIGN.md 1.4 / 3.3, reuse of the verified
dense core with the three packed differences of DESIGN.md 11.2):

* one-dimensional ``T.Kernel((total_kv // 96 + B) * slab)`` -- a *dynamic* upper
  bound on the key blocks, launched once per head slab; ``tile = bid // slab`` /
  ``h = h_off[0] + bid % slab`` keeps the divisor static, ``h_off`` is a one
  element int32 tensor and ``if h < H`` makes a tail-slab block return after one
  4 B scalar read (DESIGN.md 5.6; per-launch grid <= 65,535, see section 0);
* the fused ``(request, n0)`` decode is an O(B) serial scan over the prefix
  offsets (running prefix sum + first match), and ``live = tile < cnt`` guards
  the whole body -- no warp scan, no cross-block flag, no shared block table;
* ``k/v`` (and the transposed ``kT``) stay resident in L1 for the whole query
  loop; ``dk/dv`` accumulate in L0C across the loop;
* five GEMMs on the **explicit L0 path** (``alloc_l0a/l0b/l0c`` + manual L1->L0
  ``T.copy``), all ``(trans_A, trans_B) = (False, True)`` except ``dQ``'s
  ``(True, True)``: the single ``dS^T`` L1 tile serves ``dK`` as its A operand and
  ``dQ`` as a *transposed* A operand (the transpose is an MTE1
  ``asc_copy_l12l0a_transpose``), exactly the source's ``transpose_A=True`` reuse.
  The vector side therefore emits two half tiles (``p16``, ``dsT``) and nothing
  else -- the v3 ``ds_u[j, i]`` cross-strided scatter (63% of the iteration) is
  gone.  L0A/L0B are whole-allocation extents with 0-origin static slices;
* the ``P``/``dS`` epilogue runs on the vector side from the L0C accumulators;
* ``dq`` is written with a hardware store-atomic over the whole tile (rows past
  ``len_q`` carry exactly 0, see DESIGN.md 1.6.1 A3); ``dk/dv`` use the
  two-branch write (whole half tile inside the request / per-lane guard on the
  tail) because the platform OOB clamp only knows the tensor extent, not the
  request boundary.

Two runtime facts measured on this toolchain (dense Stage 3, re-verified here)
shape the vector side and differ from a literal reading of DESIGN.md:

1. ``T.copy(L0C -> UB)`` (``dual_dst_mode=0``) only lands in one AIV's buffer;
   the whole-tile epilogue therefore reads stale data on the other AIV (probe
   P1: 99.2% mismatch).  Every L0C<->UB crossing is a ``T.dual_copy`` 2:1 split
   instead, so each AIV owns half of the tile (``_HN = 48`` key rows) in its own
   UB bank while the L1 tiles stay whole (``pT_s/dsT_s`` are still ``(96, 64)``
   operands for the following GEMMs).
2. The half tile makes the key index AIV dependent, which the mask needs, so
   ``sid = asc_get_sub_block_id()`` is bound once outside the SIMT region and
   captured into every ``T.SimtVF`` (``T.Vector(vector=2) as sid`` remains
   uncompilable, ``InternalError ... dup sid``).

Content layout:

1. ``golden_bwd_kernel`` -- independent packed fp64 torch reference.
2. ``forward_stats`` -- the ``lse`` / ``delta`` the upstream launches provide.
3. ``_bwd_kernel`` -- the frozen kernel factory (source-compatible signature).
4. ``run_case`` -- launch on the NPU and compare with the golden.
5. ``run_L0`` / ``run_L1`` / ``run_L2`` / ``run_boundary`` / ``main``.
"""

import argparse
import functools
import math
import time
from dataclasses import dataclass

import tilelang
import tilelang.ascend.language as T
import torch
import torch_npu  # noqa: F401  -- registers torch.npu

try:  # the module form of the launch plan (L0-20 drift check); absent in the port
    import launch_plan as _launch_plan_module
except ImportError:  # pragma: no cover - the inlined copy is authoritative
    _launch_plan_module = None

_IN_DTYPES = ("float16", "bfloat16")
_TORCH_DTYPE = {"float16": torch.float16, "bfloat16": torch.bfloat16}

# ---------------------------------------------------------------------------
# 0. Launch plan of the slabbed kernel: **per-launch grid <= 65535**.
#
# ``T.Kernel(grid)`` carries the block index in 16 bits on Ascend950PR.  A grid
# that is a multiple of 65,536 aborts (error code 507035); the dangerous form --
# measured on 2026-11-24 -- is a grid whose remainder is non-zero: it does *not*
# abort, it silently executes only ``grid % 65536`` blocks and leaves the rest of
# the output untouched (DS-V3-8x4k: 66,560 -> 1,024 blocks, requests 1..7 zero).
#
# The packed kernel's logical block count ``heads * (total_kv // 96 + batch)``
# depends on the **run-time** token count, so no compile-time check can catch it:
# the host plans and asserts before launching, and splits the head axis into
# slabs (one launch per slab):
#
#     grid(launch i) = tiles * slab          (``slab`` is a compile-time constant)
#     tile = bid // slab,  h = h_off[i] + bid % slab
#
# ``plan_launch`` / ``assert_launch_plan`` / ``describe`` and the dataclasses
# below are **inlined verbatim** from ``launch_plan.py`` (DESIGN.md 5.6 g): the
# kernel file must stay self-contained for the TileOPs port, and L0-20 asserts
# field-by-field that the inlined copy and the module agree on every geometry.
# ---------------------------------------------------------------------------
NPU_GRID_MAX = 65535  # 16-bit block index, see TRAP-950-grid-over-65535
HEAD_SLAB_DEFAULT = 32  # DESIGN.md 5.6 (b): at most 4 launches for the manifest
BLOCK_N = 96  # key rows per block (DESIGN.md 5.2; half tile 48 = 3*16 for nd2nz)


@dataclass(frozen=True)
class Launch:
    """One ``T.Kernel(...)`` invocation of the slabbed kernel."""

    index: int
    head_off: int  # value of the ``h_off`` tensor (head origin of this launch)
    active_heads: int  # heads this launch may touch (``h < heads`` guard)
    grid: int  # (total_kv // 96 + batch) * slab
    idle_blocks: int  # blocks whose ``h`` lands past ``heads`` and return at once


@dataclass(frozen=True)
class Plan:
    tiles: int  # tile_upper_bound(total_kv) = total_kv // 96 + batch (per head)
    slab: int  # compile-time head slab fed to the kernel factory
    launches: tuple
    total_blocks: int  # sum of the per-launch grids (all launches)
    useful_blocks: int  # heads * tiles -- the task set the algorithm needs
    idle_blocks: int

    @property
    def max_grid(self):
        return max(launch.grid for launch in self.launches)


def tile_upper_bound(total_kv, batch, block_n=BLOCK_N):
    """`GroupTiling.tile_upper_bound`: sum_b ceil(len_kv_b / block_n) <= this."""
    return total_kv // block_n + batch


def plan_launch(batch, total_kv, heads, *, block_n=BLOCK_N,
                head_slab=HEAD_SLAB_DEFAULT):
    """Plan the launches of one operator call and assert the grid cap.

    ``head_slab`` is the *preferred* slab (a compile-time constant of the kernel);
    the returned ``slab`` is the effective one, shrunk to ``65535 // tiles`` when
    the token count is large enough to need it.  ``ValueError`` -- never a silent
    truncation -- when even ``slab == 1`` would exceed the cap.
    """
    if batch <= 0:
        raise ValueError(f"shape: batch must be positive, got {batch}")
    if heads <= 0:
        raise ValueError(f"shape: heads must be positive, got {heads}")
    if total_kv < 0:
        raise ValueError(f"shape: total_kv must be non-negative, got {total_kv}")
    if block_n <= 0:
        raise ValueError(f"shape: block_n must be positive, got {block_n}")
    if block_n != BLOCK_N:
        # REVIEW 7-6: the plan and the compiled artifact must share one constant.
        raise ValueError(
            f"grid: block_n {block_n} != the module constant BLOCK_N {BLOCK_N}; "
            "the launch plan and the kernel factory cannot disagree"
        )
    if head_slab < 1:
        raise ValueError(f"grid: head_slab must be >= 1, got {head_slab}")

    tiles = tile_upper_bound(total_kv, batch, block_n)
    room = NPU_GRID_MAX // tiles  # max heads per launch at this token count
    slab = min(int(head_slab), heads, room)
    if slab < 1:
        raise ValueError(
            "grid: one head slab would need "
            f"tile_upper_bound={tiles} blocks > the {NPU_GRID_MAX}-block per-launch "
            f"grid cap (total_kv={total_kv}, batch={batch}, block_n={block_n}); "
            "split the token axis (smaller block_n is not allowed to be the fix) or "
            "run the operator in token chunks"
        )

    grid = tiles * slab
    if grid > NPU_GRID_MAX:  # defensive: unreachable by construction
        raise ValueError(f"grid: per-launch grid {grid} exceeds {NPU_GRID_MAX}")

    launches = []
    off = 0
    while off < heads:
        active = min(slab, heads - off)
        launches.append(
            Launch(
                index=len(launches),
                head_off=off,
                active_heads=active,
                grid=grid,
                idle_blocks=grid - tiles * active,
            )
        )
        off += slab

    plan = Plan(
        tiles=tiles,
        slab=slab,
        launches=tuple(launches),
        total_blocks=sum(launch.grid for launch in launches),
        useful_blocks=tiles * heads,
        idle_blocks=sum(launch.idle_blocks for launch in launches),
    )
    assert_launch_plan(plan, heads=heads)
    return plan


def assert_launch_plan(plan, *, heads):
    """Hard pre-launch assertion: cap honoured, heads covered exactly once."""
    for launch in plan.launches:
        if launch.grid > NPU_GRID_MAX:
            raise ValueError(
                f"grid: launch {launch.index} needs {launch.grid} blocks > {NPU_GRID_MAX}"
            )
        if launch.active_heads < 1 or launch.head_off + launch.active_heads > heads:
            raise ValueError(
                f"grid: launch {launch.index} covers heads "
                f"[{launch.head_off}, {launch.head_off + launch.active_heads}) "
                f"outside [0, {heads})"
            )
    covered = sum(launch.active_heads for launch in plan.launches)
    if covered != heads:
        raise ValueError(f"grid: head slabs cover {covered} heads, expected {heads}")


def describe(plan):
    """One-line summary for test logs."""
    per_launch = ", ".join(
        f"h[{launch.head_off}:{launch.head_off + launch.active_heads}) grid={launch.grid} "
        f"idle={launch.idle_blocks}"
        for launch in plan.launches
    )
    return (
        f"tiles={plan.tiles} slab={plan.slab} launches={len(plan.launches)} "
        f"max_grid={plan.max_grid} blocks={plan.total_blocks} "
        f"useful={plan.useful_blocks} idle={plan.idle_blocks} | {per_launch}"
    )


_BN = BLOCK_N  # alias: the tile shapes of the kernel body (single source above)
_BM = 64  # query rows of one block
_HN = _BN // 2  # key rows per AIV (dual_copy 2:1 split; 48 % 16 == 0 for nd2nz)
_STAGES = 2  # query-chunk loop pipeline depth (Round 2 sweep)
_QCHUNK = 2  # 64-row query blocks per GM fill (1 = frozen design)
_MF = _BM * _QCHUNK  # query rows per fill
_OOB_PAD = _BN  # sentinel rows appended to every output allocation
_ATOL = 5e-3  # DESIGN.md 8.2 (max(5e-3, reference_tolerance))
_RTOL = {"float16": 1e-3, "bfloat16": 1.6e-2}
_RTOL_ACC = 1e-3  # DESIGN.md 8.2: float32 ``dq_acc`` relative tolerance
_DEFAULT_DTYPE = "bfloat16"
_SENTINEL = -12345.0


def factory_config():
    """The **single source** of the kernel's build-time knobs (DESIGN.md 5.6 g).

    ``block_n`` is defined here and nowhere else: the launch plan and the kernel
    factory both read it, and the wrapper's post-assert compares plan, artifact
    metadata and this config (L0-20 / G5 (6)).  ``head_slab`` is only a
    *preference* -- the effective slab is ``plan.slab``, which shrinks when the
    run-time token count demands it.
    """
    return {
        "block_n": BLOCK_N,
        "block_m": _BM,
        "num_stages": _STAGES,
        "head_slab": HEAD_SLAB_DEFAULT,
    }


# ---------------------------------------------------------------------------
# 1. Golden: independent reference (materialised P, no re-implementation of the
#    tiling).  Memory-lean: the accumulation dtype, input casting and score
#    plane are all per-request/per-slab, and the three gradients are written
#    straight to their (request, head-slab) destination, so no full-size fp64
#    accumulator exists (the CPU reference must fit in the 32 GiB cgroup even
#    for the 16384-token manifest request).
# ---------------------------------------------------------------------------
def _golden_acc(max_len):
    """Accumulation dtype of the reference.

    float64 while the materialised score plane stays small; float32 for the very
    long manifest requests, where float64 planes would exceed the container
    memory.  float32 accumulation of <= D-term dot products is 3+ orders below
    the 5e-3 tolerance.
    """
    return torch.float64 if max_len <= 8192 else torch.float32


def golden_bwd_kernel(q, k, v, dout, cu_seqlens_q, cu_seqlens_kv, scale=None,
                      is_causal=True, dtype=None, slab_heads=4, quantize=False):
    """Materialised packed reference of the backward pass (request by request).

    ``q`` ``(T_q, H, D_qk)``; ``k`` ``(T_kv, H, D_qk)``; ``v, dout``
    ``(T_kv|T_q, H, D_v)``; ``cu_seqlens_*`` are the packed request offsets.  The
    probabilities and ``delta`` are recomputed here from ``q, k, v, dout`` -- the
    ``lse`` / ``delta`` tensors the kernel consumes are never read, so this
    reference shares no assumption with the block enumeration, the L0C
    accumulators, the dual split, the request-boundary write branches or the
    atomic ``dq`` accumulation.  Requests are walked one by one and sliced by
    ``cu_seqlens``, i.e. exactly the semantics of the source ``ref_program``.

    Returns ``(dq, dk, dv)`` rounded once to ``dtype`` plus
    ``dq_acc_true = dq / scale``, the float32 intermediate this kernel writes
    into ``dq_acc`` (``_dq_land_kernel`` applies ``scale`` and the cast).

    ``quantize=True`` is the *source-semantics* variant: the probabilities and
    ``dS`` are rounded to the input dtype before the three output contractions,
    which is what the kernel (and the source GPU kernel) must do because the GEMM
    operands live in 16-bit L1.  It is the second reference for the strongly
    cancelling cells where the ideal and a 16-bit-operand kernel legitimately
    differ by more than the tolerance.
    """
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    tdt = dtype if isinstance(dtype, torch.dtype) else q.dtype
    total_q, heads, dim_qk = q.shape
    total_kv, dim_v = k.shape[0], v.shape[-1]
    bq, bkv = cu_seqlens_q.tolist(), cu_seqlens_kv.tolist()
    span = max([bq[i + 1] - bq[i] for i in range(len(bq) - 1)] +
               [bkv[i + 1] - bkv[i] for i in range(len(bkv) - 1)] + [0])
    acc = _golden_acc(span)
    dk = torch.empty(total_kv, heads, dim_qk, dtype=tdt)
    dv = torch.empty(total_kv, heads, dim_v, dtype=tdt)
    dq = torch.empty(total_q, heads, dim_qk, dtype=tdt)
    dq_acc = torch.empty(total_q, heads, dim_qk, dtype=torch.float32)
    for b in range(len(bq) - 1):
        q0, q1, k0, k1 = bq[b], bq[b + 1], bkv[b], bkv[b + 1]
        len_q, len_kv = q1 - q0, k1 - k0
        shift = len_kv - len_q
        q_pos = torch.arange(len_q) + shift
        key_pos = torch.arange(len_kv)
        hidden = key_pos[None, :] > q_pos[:, None] if is_causal else None
        qf = q[q0:q1].detach().to(acc)
        kf = k[k0:k1].detach().to(acc)
        vf = v[k0:k1].detach().to(acc)
        dof = dout[q0:q1].detach().to(acc)
        for h0 in range(0, heads, slab_heads):  # slab the heads: O(H*len_q*len_kv)
            hs = slice(h0, min(h0 + slab_heads, heads))
            # Score plane: one big buffer, reused in place for P and then dS.
            s = torch.einsum("shd,nhd->hsn", qf[:, hs], kf[:, hs]).mul_(scale)
            if hidden is not None:
                s.masked_fill_(hidden[None], float("-inf"))
            lse = s.logsumexp(-1, keepdim=True)
            p = s.sub_(lse).exp_().nan_to_num_(0.0)  # rows that see no key -> 0
            dp = torch.einsum("shd,nhd->hsn", dof[:, hs], vf[:, hs])
            # dV uses P; keep it here (no cast) unless the source-semantics
            # reference is requested.
            dv_loc = torch.einsum("hsn,shd->nhd", p.to(tdt).to(acc) if quantize else p,
                                  dof[:, hs])
            ds = p.mul_(dp.sub_(torch.einsum("hsn,hsn->hs", p, dp)[..., None]))
            if quantize:
                ds = ds.to(tdt).to(acc)
            dk_loc = torch.einsum("hsn,shd->nhd", ds, qf[:, hs]).mul_(scale)
            dq_loc = torch.einsum("hsn,nhd->shd", ds, kf[:, hs])  # = dq / scale
            dv[k0:k1, hs] = dv_loc.to(tdt)
            dk[k0:k1, hs] = dk_loc.to(tdt)
            dq[q0:q1, hs] = (dq_loc * scale).to(tdt)
            dq_acc[q0:q1, hs] = dq_loc.to(torch.float32)
            del s, p, dp, ds, dv_loc, dk_loc, dq_loc
        del qf, kf, vf, dof
    return dq, dk, dv, dq_acc


def forward_stats(q, k, v, dout, cu_seqlens_q, cu_seqlens_kv, is_causal=True,
                  slab_heads=4):
    """The packed ``(lse, delta)`` pair the upstream launches hand to this kernel.

    ``lse`` is the natural-log log-sum-exp of the (bottom-right masked) scores,
    ``-inf`` for a query row that sees no key; ``delta = rowsum(o * dout)`` with
    ``o = P @ v``.  Both are **head-major** ``(H, T_q)`` float32, i.e. the packed
    layout of the source factory (``stat_shape = (heads, total_q)``), computed in
    float32 from the 16-bit inputs: exactly what a float32-accumulating forward
    pass produces, and independent of the golden (which recomputes both).
    """
    total_q, heads, dim_qk = q.shape
    scale = dim_qk**-0.5
    # Device-agnostic: the same function serves the CPU golden path and the
    # on-device statistics of the large grid-cap regressions.
    dev = q.device
    lse = torch.full((heads, total_q), float("-inf"), dtype=torch.float32, device=dev)
    delta = torch.zeros(heads, total_q, dtype=torch.float32, device=dev)
    bq, bkv = cu_seqlens_q.tolist(), cu_seqlens_kv.tolist()
    for b in range(len(bq) - 1):
        q0, q1, k0, k1 = bq[b], bq[b + 1], bkv[b], bkv[b + 1]
        len_q, len_kv = q1 - q0, k1 - k0
        if len_q == 0:
            continue
        shift = len_kv - len_q
        q_pos = torch.arange(len_q, device=dev) + shift
        key_pos = torch.arange(len_kv, device=dev)
        hidden = key_pos[None, :] > q_pos[:, None] if is_causal else None
        qf = q[q0:q1].detach().float()
        kf = k[k0:k1].detach().float()
        vf = v[k0:k1].detach().float()
        dof = dout[q0:q1].detach().float()
        for h0 in range(0, heads, slab_heads):
            hs = slice(h0, min(h0 + slab_heads, heads))
            s = torch.einsum("shd,nhd->hsn", qf[:, hs], kf[:, hs]).mul_(scale)
            if hidden is not None:
                s.masked_fill_(hidden[None], float("-inf"))
            row_lse = s.logsumexp(-1)  # (h, s); -inf when nothing is visible
            p = s.sub_(row_lse[..., None]).exp_().nan_to_num_(0.0)
            # Head-first matmuls (no einsum label ambiguity on size-1 axes).
            vs = vf[:, hs].permute(1, 0, 2)  # (h, n, dv)
            dos = dof[:, hs].permute(1, 0, 2)  # (h, s, dv)
            o = torch.matmul(p, vs)  # (h, s, dv)
            lse[hs, q0:q1] = row_lse
            delta[hs, q0:q1] = (o * dos).sum(-1)
            del s, p, o
        del qf, kf, vf, dof
    return lse.contiguous(), delta.contiguous()


# ---------------------------------------------------------------------------
# 2. Kernel
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=64)
def _bwd_kernel(
    batch, heads, dim_qk, dim_v, dense_seqlen_q, dense_seqlen_kv, is_causal, scale,
    dtype, a2_fast=True, head_slab=HEAD_SLAB_DEFAULT,
):
    """Frozen factory, signature-compatible with the source ``_bwd_kernel``.

    ``batch, heads, dim_qk, dim_v, dense_seqlen_q, dense_seqlen_kv, is_causal,
    scale, dtype`` is exactly the source order, so the harness keeps its call
    site.  ``dense_seqlen_q == 0`` selects the packed layout in the source; a
    positive value belongs to ``MLADenseBwdOp`` and is out of this kernel's scope
    (``ValueError``).  ``dense_seqlen_kv`` is unused on the packed branch (the
    key extent comes from the tensor itself), exactly as in the source factory.
    ``is_causal``, ``scale`` and ``a2_fast`` are build-time constants;
    ``a2_fast=False`` forces the masked epilogue arm everywhere and exists only
    for the arm-equivalence check.

    ``head_slab`` (DESIGN.md 5.6 b) is the **compile-time** head slab of one
    launch: the program's grid is ``(total_kv // block_n + batch) * head_slab``
    and the per-launch head origin arrives as the run-time ``h_off`` tensor, so
    one compiled program serves every slab.  It must be ``plan.slab`` -- the
    caller (``build_kernel``) plans first and post-asserts plan == artifact.
    """
    if dtype not in _IN_DTYPES:
        raise ValueError(f"dtype: {dtype!r} is not one of {_IN_DTYPES}")
    if dense_seqlen_q > 0:
        raise ValueError(
            f"shape: dense_seqlen_q must be 0 for MLAVarlenBwdOp (got {dense_seqlen_q}); "
            "the dense layout is out of this kernel's scope"
        )
    if min(batch, heads, dim_qk, dim_v) <= 0:
        raise ValueError(
            "shape: batch/heads/dim_qk/dim_v must all be positive, got "
            f"{batch}/{heads}/{dim_qk}/{dim_v}"
        )
    if dim_qk % 32 or dim_v % 16:
        raise ValueError(f"shape: dim_qk ({dim_qk}) must divide 32, dim_v ({dim_v}) must divide 16")
    cfg = factory_config()
    block_n = cfg["block_n"]
    # The tile shapes of the body are compiled from ``_BN``; the plan's block_n is
    # ``cfg["block_n"]``.  They must be the same object of truth (DESIGN.md 5.6 g)
    # and the half-tile split of the dual copies must stay inside block_n.
    if block_n != _BN or block_n != 2 * _HN:
        raise ValueError(
            f"grid: cfg block_n ({block_n}) != the compiled tile constant "
            f"(_BN={_BN}, half tile _HN={_HN}); refusing to build"
        )
    if head_slab < 1:
        raise ValueError(f"grid: head_slab must be >= 1, got {head_slab}")
    if head_slab > heads:
        raise ValueError(f"grid: head_slab ({head_slab}) cannot exceed heads ({heads})")

    total_q = T.dynamic("total_q")
    total_kv = T.dynamic("total_kv")
    dt = dtype
    # Dynamic upper bound of the key-block count (source ``tile_upper_bound``):
    # sum_b ceil(len_kv_b / 64) <= total_kv // 64 + batch.  Tiles past the real
    # count are dropped by ``live`` and issue no DMA at all.  The head axis is the
    # compile-time slab, so this launch's grid stays <= 65,535 for a planned
    # geometry (``plan_launch`` / ``assert_launch_consistency``).
    grid = (total_kv // block_n + batch) * head_slab

    @T.prim_func
    def _bwd_main(
        q: T.Tensor((total_q, heads, dim_qk), dt),  # type: ignore
        k: T.Tensor((total_kv, heads, dim_qk), dt),  # type: ignore
        v: T.Tensor((total_kv, heads, dim_v), dt),  # type: ignore
        dout: T.Tensor((total_q, heads, dim_v), dt),  # type: ignore
        lse: T.Tensor((heads, total_q), "float32"),  # type: ignore
        delta: T.Tensor((heads, total_q), "float32"),  # type: ignore
        cu_q: T.Tensor((batch + 1,), "int32"),  # type: ignore
        cu_kv: T.Tensor((batch + 1,), "int32"),  # type: ignore
        dq_acc: T.Tensor((total_q, heads, dim_qk), "float32"),  # type: ignore
        dk: T.Tensor((total_kv, heads, dim_qk), dt),  # type: ignore
        dv: T.Tensor((total_kv, heads, dim_v), dt),  # type: ignore
        h_off: T.Tensor((1,), "int32"),  # type: ignore  -- run-time head origin
    ) -> None:
        with T.Kernel(grid) as bid:
            tile = bid // head_slab  # static divisor (DESIGN.md 1.6.3 candidate 7)
            # Run-time head origin of this launch + compile-time slot: one
            # compiled program serves every slab (DESIGN.md 5.6 b).
            h = h_off[0] + bid % head_slab  # inner head -> L2 row reuse
            # Sub-block id of the AIV running this code (0/1).  Bound outside the
            # SIMT region because asc_get_sub_block_id() is not callable inside a
            # __simt_vf__ body; the value is then captured as a VF argument and is
            # the same id the dual copies split with.
            sid = T.call_extern("int64", "asc_get_sub_block_id")

            # ---- L1: key resident + per-iteration operands -------------------
            k_s = T.alloc_l1((_BN, dim_qk), dt)
            v_s = T.alloc_l1((_BN, dim_v), dt)
            kT_s = T.alloc_l1((dim_qk, _BN), dt)
            q_s = T.alloc_l1((_MF, dim_qk), dt)
            qT_s = T.alloc_l1((dim_qk, _MF), dt)
            do_s = T.alloc_l1((_MF, dim_v), dt)
            doT_s = T.alloc_l1((dim_v, _MF), dt)
            pT_s = T.alloc_l1((_BN, _BM), dt)
            dsT_s = T.alloc_l1((_BN, _BM), dt)  # dK (as A) and dQ (transposed A)
            # ---- L0A/L0B: explicit operands, one version (L0B has 4 KiB left) --
            a_k = T.alloc_l0a((_BN, dim_qk), dt)  # K-major: k / v / pT / dsT
            a_t = T.alloc_l0a((_BN, _BM), dt)  # MN-major: dsT for dQ (trans_A)
            b_lo = T.alloc_l0b((_BM, dim_qk), dt)  # q / do
            b_hi = T.alloc_l0b((dim_qk, _BN), dt)  # qT / doT / kT (kT reloaded last)
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
            lse_u = T.alloc_shared((_MF,), "float32")
            dlt_u = T.alloc_shared((_MF,), "float32")
            tail_u = T.alloc_shared((_HN, dim_qk), "float32")
            tailv_u = T.alloc_shared((_HN, dim_v), "float32")
            out_k = T.alloc_shared((_HN, dim_qk), dt)
            out_v = T.alloc_shared((_HN, dim_v), dt)
            # ---- block-level scalars (fused running-prefix decode) ----------
            cnt = T.alloc_local([1], "int32")
            req = T.alloc_local([1], "int32")
            n0_l = T.alloc_local([1], "int32")
            live = T.alloc_local([1], "int32")

            # ---- tail-slab guard (DESIGN.md 5.6 b) -----------------------
            # A block whose head lands past ``heads`` returns here after one
            # 4 B ``h_off`` scalar read and the block-index arithmetic: no DMA,
            # no ``cu_seqlens`` read (P6 device code: ``if (h < H) { ... }``).
            if h < heads:
                # Fused (prefix sum + first-match decode), O(batch) serial scans:
                # cnt ends at the real tile count of the packed batch, req/n0 hold
                # the first request whose tile range contains ``tile``.
                cnt[0] = 0
                req[0] = 0
                n0_l[0] = 0
                for g in T.serial(batch):
                    nt = (cu_kv[g + 1] - cu_kv[g] + _BN - 1) // _BN
                    if (tile >= cnt[0]) & (tile < cnt[0] + nt):
                        req[0] = g
                        n0_l[0] = (tile - cnt[0]) * _BN
                    cnt[0] = cnt[0] + nt
                live[0] = T.if_then_else(tile < cnt[0], 1, 0)

                if live[0] == 1:
                    n0 = n0_l[0]
                    q_start = cu_q[req[0]]
                    kv_start = cu_kv[req[0]]
                    len_q = cu_q[req[0] + 1] - q_start
                    len_kv = cu_kv[req[0] + 1] - kv_start
                    shift = len_kv - len_q
                    koff = kv_start + n0

                    # Whole-extent loads: the tail rows outside [0, T_kv) are clamped
                    # by the OOB pass and filled with zero; the mask arm drops them.
                    T.copy(k[koff : koff + _BN, h, :], k_s)
                    T.copy(v[koff : koff + _BN, h, :], v_s)
                    T.copy(k[koff : koff + _BN, h, :], kT_s, transpose=True)

                    loop_st = T.max(0, (n0 - shift) // _BM) if is_causal else 0
                    # max(..., loop_st + 1) keeps the loop non-empty for the
                    # degenerate request len_q == 0 (T.clear cannot touch L0C, so
                    # dk_c/dv_c must be initialised by a first clear_accum iteration;
                    # that iteration is fully masked and yields exact 0).
                    loop_ed = T.max(T.ceildiv(len_q, _BM), loop_st + 1)
                    # ---- query-chunk loop: one GM fill per 2 64-row blocks ----
                    # The fill granularity is what this round changes: a long request
                    # re-reads its q/do suffix once per key block, and the per-fill
                    # DRAM round trip is what makes the per-iteration price grow with
                    # the request length.  `_QCHUNK` query blocks share one descriptor
                    # (and one lse/delta run), so the round-trip count per query row
                    # drops by `_QCHUNK`; the GEMM tile, the task mapping, the grid and
                    # the slab are untouched.
                    cst = loop_st // _QCHUNK
                    ced = T.max(T.ceildiv(loop_ed, _QCHUNK), cst + 1)
                    for kqc in T.Pipelined(cst, ced, num_stages=_STAGES):
                        m0 = kqc * _MF
                        qoff = q_start + m0
                        T.copy(q[qoff : qoff + _MF, h, :], q_s)
                        T.copy(dout[qoff : qoff + _MF, h, :], do_s)
                        T.copy(q[qoff : qoff + _MF, h, :], qT_s, transpose=True)
                        T.copy(dout[qoff : qoff + _MF, h, :], doT_s, transpose=True)
                        T.copy(lse[h, qoff : qoff + _MF], lse_u)  # head-major run
                        T.copy(delta[h, qoff : qoff + _MF], dlt_u)
                        for sub in range(_QCHUNK):
                            mb = sub * _BM
                            m1 = m0 + mb
                            qo = qoff + mb
                            clr = (kqc == cst) if sub == 0 else False
                            # ---- s^T and dP^T: two L0 GEMMs, cleared every sub-step ----
                            T.copy(k_s, a_k)
                            T.copy(q_s[mb : mb + _BM, :], b_lo)
                            T.gemm(a_k, b_lo, st_c, transpose_B=True, clear_accum=True)
                            T.copy(v_s, a_k[0:_BN, 0:dim_v])
                            T.copy(do_s[mb : mb + _BM, :], b_lo[0:_BM, 0:dim_v])
                            T.gemm(a_k[0:_BN, 0:dim_v], b_lo[0:_BM, 0:dim_v], dpt_c,
                                   transpose_B=True, clear_accum=True)
                            # L0C -> UB: each AIV receives its own 48 key rows.
                            T.dual_copy(st_c, st_u)
                            T.dual_copy(dpt_c, dp_u)

                            fast = (n0 + _BN <= len_kv) & (m1 + _BM <= len_q)
                            if is_causal:
                                fast = fast & (n0 + _BN - 1 <= m1 + shift)
                            if a2_fast:
                                if fast:
                                    # Fast arm: the whole tile is visible and in range.
                                    with T.SimtVF(threads=128):
                                        for i, j in T.Parallel(_HN, _BM):
                                            p = T.exp(st_u[i, j] * scale - lse_u[mb + j])
                                            p16_u[i, j] = T.cast(p, dt)
                                            dsT_u[i, j] = T.cast(
                                                p * (dp_u[i, j] - dlt_u[mb + j]), dt)
                                else:
                                    with T.SimtVF(threads=128):
                                        for i, j in T.Parallel(_HN, _BM):
                                            gi = n0 + T.cast(sid, "int32") * _HN + i
                                            gj = m1 + j
                                            vis = (gi < len_kv) & (gj < len_q)
                                            if is_causal:
                                                vis = vis & (gi <= gj + shift)
                                            p = T.exp(st_u[i, j] * scale - lse_u[mb + j])
                                            pv = T.if_then_else(vis, p, T.cast(0, "float32"))
                                            p16_u[i, j] = T.cast(pv, dt)
                                            dsT_u[i, j] = T.cast(
                                                pv * (dp_u[i, j] - dlt_u[mb + j]), dt)
                            else:
                                with T.SimtVF(threads=128):
                                    for i, j in T.Parallel(_HN, _BM):
                                        gi = n0 + T.cast(sid, "int32") * _HN + i
                                        gj = m1 + j
                                        vis = (gi < len_kv) & (gj < len_q)
                                        if is_causal:
                                            vis = vis & (gi <= gj + shift)
                                        p = T.exp(st_u[i, j] * scale - lse_u[mb + j])
                                        pv = T.if_then_else(vis, p, T.cast(0, "float32"))
                                        p16_u[i, j] = T.cast(pv, dt)
                                        dsT_u[i, j] = T.cast(
                                            pv * (dp_u[i, j] - dlt_u[mb + j]), dt)

                            # UB -> L1: M-split of the two produced half tiles.
                            T.dual_copy(p16_u, pT_s)
                            T.dual_copy(dsT_u, dsT_s)
                            # ---- dV / dK (accumulate) and dQ (transpose_A reuse of dsT) ---
                            T.copy(pT_s, a_k[0:_BN, 0:_BM])
                            T.copy(doT_s[0:dim_v, mb : mb + _BM], b_hi[0:dim_v, 0:_BM])
                            T.gemm(a_k[0:_BN, 0:_BM], b_hi[0:dim_v, 0:_BM], dv_c,
                                   transpose_B=True, clear_accum=clr)
                            T.copy(dsT_s, a_k[0:_BN, 0:_BM])
                            T.copy(qT_s[0:dim_qk, mb : mb + _BM], b_hi[0:dim_qk, 0:_BM])
                            T.gemm(a_k[0:_BN, 0:_BM], b_hi[0:dim_qk, 0:_BM], dk_c,
                                   transpose_B=True, clear_accum=clr)
                            # Same dsT tile as dK's A operand; the transpose is done by
                            # the L1 -> L0A transposing copy (MTE1), not by a vector
                            # scatter (v3's 63% hotspot, DESIGN.md 0.6 R3).
                            T.copy(dsT_s, a_t)
                            # kT must be reloaded into b_hi AFTER qT/doT (DESIGN.md 9.2).
                            T.copy(kT_s, b_hi)
                            T.gemm(a_t, b_hi, dq_c, transpose_A=True, transpose_B=True,
                                   clear_accum=True)
                            # dQ accumulates across cores -> hardware store-atomic into
                            # fp32.  Rows/columns past this request carry exactly 0
                            # (DESIGN.md 1.6.1 A3), so the neighbour request's dq_acc
                            # rows are left untouched; the tensor end is clamped by the
                            # platform.
                            T.set_atomic("add", "float32")
                            T.copy(dq_c, dq_acc[qo : qo + _BM, h, :])
                            T.set_atomic_none()

                    # ---- block tail: dk = scale * cast(dk_c), dv = cast(dv_c) ----
                    T.dual_copy(dk_c, tail_u)
                    T.dual_copy(dv_c, tailv_u)
                    with T.SimtVF(threads=128):
                        for i, d in T.Parallel(_HN, dim_qk):
                            out_k[i, d] = T.cast(tail_u[i, d] * scale, dt)
                        for i, d in T.Parallel(_HN, dim_v):
                            out_v[i, d] = T.cast(tailv_u[i, d], dt)
                    own = koff + T.cast(sid, "int32") * _HN  # this AIV's first key row
                    kv_end = kv_start + len_kv
                    if own + _HN <= kv_end:
                        # Fast path: the whole half tile lives inside the request.
                        T.copy(out_k, dk[own : own + _HN, h, :])
                        T.copy(out_v, dv[own : own + _HN, h, :])
                    else:
                        # Tail path (at most one per request): per-lane guard on the
                        # request boundary -- the platform clamp only knows T_kv, so a
                        # whole-tile store here would zero the next request's rows.
                        with T.SimtVF(threads=128):
                            for i, d in T.Parallel(_HN, dim_qk):
                                if own + i < kv_end:
                                    dk[own + i, h, d] = out_k[i, d]
                            for i, d in T.Parallel(_HN, dim_v):
                                if own + i < kv_end:
                                    dv[own + i, h, d] = out_v[i, d]

    fn = tilelang.compile(_bwd_main, target="ascend")
    # Metadata of *this* artifact, read back by
    # ``assert_launch_consistency`` (DESIGN.md 5.6 g (3)): the grid the
    # host plans and the grid this program really launches must agree.
    fn.head_slab = head_slab
    fn.block_n = block_n
    return fn


def _launch(batch, heads, dim_qk, dim_v, is_causal, scale, dtype, a2_fast=True,
            head_slab=HEAD_SLAB_DEFAULT):
    """Compiled kernel for one packed geometry and one compile-time head slab.

    The two ``dense_seqlen_*`` slots are 0: on this branch the query/key extents
    are ``T.dynamic`` (taken from the tensors), exactly like the source packed
    call site.
    """
    return _bwd_kernel(
        batch, heads, dim_qk, dim_v, 0, 0, is_causal, scale, dtype, a2_fast, head_slab
    )


def _factory_meta(fn):
    """Metadata of a compiled artifact: the geometry it was really built with."""
    return {"head_slab": fn.head_slab, "block_n": fn.block_n}


def assert_launch_consistency(plan, meta, *, batch, total_kv, block_n):
    """Post-assert that the plan and the compiled artifact describe one grid.

    DESIGN.md 5.6 g (3): the *plan* is the only source of ``head_slab`` for the
    factory, so this recomputes the real per-launch grid from the artifact's own
    ``(block_n, head_slab)`` and compares it field by field.  Without it a
    ``block_n`` mismatch is invisible, e.g. plan at ``block_n=64`` for
    ``total_kv=131,072, batch=8`` gives ``slab=31, max_grid=63,736`` while a
    ``block_n=32`` build launches ``(131,072 // 32 + 8) * 31 = 127,224`` blocks
    and is silently truncated again.  ``ValueError`` -- never a silent pass.
    """
    real_grid = (total_kv // block_n + batch) * meta["head_slab"]
    if real_grid != plan.max_grid:
        raise ValueError(
            f"grid: plan/compile mismatch -- kernel grid {real_grid} != plan.max_grid "
            f"{plan.max_grid} (batch={batch}, total_kv={total_kv}, block_n={block_n}, "
            f"head_slab={meta['head_slab']})"
        )
    if meta["head_slab"] != plan.slab:
        raise ValueError(
            f"grid: kernel factory got head_slab={meta['head_slab']}, plan.slab="
            f"{plan.slab} (total_kv={total_kv}, batch={batch}, heads in plan)"
        )
    if meta["block_n"] != block_n:
        raise ValueError(
            f"grid: kernel factory got block_n={meta['block_n']}, plan block_n="
            f"{block_n}; the plan and the artifact must share one constant source"
        )


def build_kernel(batch, heads, dim_qk, dim_v, is_causal, scale, dtype, total_kv,
                 a2_fast=True, head_slab=None):
    """Plan -> compile -> post-assert of one operator call (DESIGN.md 5.6 g).

    ``total_kv`` is a host-side shape value, so the plan costs O(1); the head
    slab is ``plan.slab`` (never ``cfg["head_slab"]`` -- the plan may have shrunk
    it for a long token count), and ``cfg["block_n"]`` is the single source of the
    key-block size for both the plan and the factory.
    """
    cfg = factory_config()
    preferred = cfg["head_slab"] if head_slab is None else int(head_slab)
    plan = plan_launch(batch, total_kv, heads, block_n=cfg["block_n"],
                       head_slab=preferred)
    fn = _launch(batch, heads, dim_qk, dim_v, is_causal, scale, dtype, a2_fast,
                 head_slab=plan.slab)
    assert_launch_consistency(plan, _factory_meta(fn), batch=batch,
                              total_kv=total_kv, block_n=cfg["block_n"])
    return fn, plan, cfg


# ---------------------------------------------------------------------------
# 3. Launch + comparison with the golden
# ---------------------------------------------------------------------------
_RESULTS = []  # (level, name, ok, detail)
_MAX_DIFF = [0.0]
_ULP = [0.0]  # max k_ulp seen by the dq_acc diagnostics (never a criterion)


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


def _offsets(lens):
    out = [0]
    for value in lens:
        out.append(out[-1] + value)
    return out


def _slab(max_len):
    """Head slab of the fp64 golden: keeps each materialised score plane bounded."""
    if max_len <= 2048:
        return 4
    if max_len <= 8192:
        return 2
    return 1


def _build_packed(q_lens, kv_lens, heads, dim_qk, dim_v, dtype_name, seed, special=None):
    """Packed ``(q, k, v, dout, cu_q, cu_kv)`` CPU tensors for one request vector."""
    def pack(lens, tail, seed0):
        parts = [_randn((length,) + tail, dtype_name, seed0 + i) for i, length in enumerate(lens)]
        if not parts:
            return torch.zeros((0,) + tail, dtype=_TORCH_DTYPE[dtype_name])
        return torch.cat(parts, 0)

    q = pack(q_lens, (heads, dim_qk), seed)
    k = pack(kv_lens, (heads, dim_qk), seed + 11)
    v = pack(kv_lens, (heads, dim_v), seed + 22)
    dout = pack(q_lens, (heads, dim_v), seed + 33)
    if special == "zero_dout":
        dout.zero_()
    elif special == "tiny_q":
        q = (q.float() * 0.05).to(q.dtype)
    elif special is not None and special != "atom_prefill":
        raise ValueError(f"special: unknown pattern {special!r}")
    cu_q = torch.tensor(_offsets(q_lens), dtype=torch.int32)
    cu_kv = torch.tensor(_offsets(kv_lens), dtype=torch.int32)
    return q, k, v, dout, cu_q, cu_kv


def _run_dev(q, k, v, dout, lse, delta, cu_q, cu_kv, is_causal, scale, dtype,
             a2_fast=True, prefill=None, repeat=1, head_slab=None, keep_device=False):
    """Launch every planned head slab on the NPU; results stay on the device.

    The launch plan is computed from the run-time ``total_kv`` and asserted
    against the compiled artifact (``build_kernel``); one ``h_off`` int32 scalar
    per launch carries the head origin, so a single program serves all slabs.

    ``dq_acc`` is zeroed (or ``prefill``-ed) **once, before the first launch**:
    it is a store-atomic accumulator and the slabs own disjoint head ranges
    (DESIGN.md 5.6 d item 4), so a per-launch clear would drop the earlier
    slabs' contribution.  ``dk``/``dv`` are assignments with one owner per
    ``(tile, head)`` and need no ordering.

    Every output allocation carries ``_OOB_PAD`` sentinel rows behind it and the
    kernel receives a leading view, so any store past ``T_q`` / ``T_kv`` lands in
    the sentinel area and is reported (``oob``) instead of corrupting the heap.
    """
    total_q, heads, dim_qk = q.shape
    total_kv, dim_v = k.shape[0], v.shape[-1]
    batch = cu_q.numel() - 1
    tdt = _TORCH_DTYPE[dtype]
    dev = "npu"
    q_dev = q.contiguous().to(dev)
    k_dev = k.contiguous().to(dev)
    v_dev = v.contiguous().to(dev)
    do_dev = dout.contiguous().to(dev)
    lse_dev = lse.to(dev)
    delta_dev = delta.to(dev)
    cu_q_dev = cu_q.to(dev)
    cu_kv_dev = cu_kv.to(dev)
    pad = _OOB_PAD
    dq_base = torch.full((total_q + pad, heads, dim_qk), _SENTINEL,
                         dtype=torch.float32, device=dev)
    dq_acc = dq_base[0:total_q]
    if prefill is None:
        dq_acc.zero_()
    else:
        dq_acc.fill_(prefill)
    dk_base = torch.full((total_kv + pad, heads, dim_qk), _SENTINEL, dtype=tdt, device=dev)
    dv_base = torch.full((total_kv + pad, heads, dim_v), _SENTINEL, dtype=tdt, device=dev)
    dk_dev = dk_base[0:total_kv]
    dv_dev = dv_base[0:total_kv]

    fn, plan, cfg = build_kernel(batch, heads, dim_qk, dim_v, is_causal, scale, dtype,
                                 total_kv, a2_fast=a2_fast, head_slab=head_slab)
    h_off = torch.tensor([launch.head_off for launch in plan.launches],
                         dtype=torch.int32, device=dev)
    args = (q_dev, k_dev, v_dev, do_dev, lse_dev, delta_dev, cu_q_dev, cu_kv_dev,
            dq_acc, dk_dev, dv_dev)

    def launch_all():
        for index in range(len(plan.launches)):
            fn(*args, h_off[index:index + 1])

    start = time.perf_counter()
    launch_all()
    torch.npu.synchronize()
    elapsed = (time.perf_counter() - start) * 1e3
    oob = 0
    for base, rows in ((dq_base, total_q), (dk_base, total_kv), (dv_base, total_kv)):
        oob += int((base[rows:] != _SENTINEL).sum())
    # Read the results before any timed repeat: dq_acc is an in-out store-atomic
    # accumulator, so re-launching into the same buffer would add a second time.
    if keep_device:
        got_dq32, got_dk, got_dv = dq_acc, dk_dev, dv_dev
    else:
        got_dq32, got_dk, got_dv = dq_acc.cpu(), dk_dev.cpu(), dv_dev.cpu()
    if repeat > 1:
        start = time.perf_counter()
        for _ in range(repeat):
            launch_all()
        torch.npu.synchronize()
        elapsed = (time.perf_counter() - start) * 1e3 / repeat
    return {
        "dq_acc": got_dq32,
        "dk": got_dk,
        "dv": got_dv,
        "oob": oob,
        "elapsed": elapsed,
        "plan": plan,
        "cfg": cfg,
        "batch": batch,
        "total_q": total_q,
        "total_kv": total_kv,
        "heads": heads,
        "dim_qk": dim_qk,
        "dim_v": dim_v,
    }


def _execute(q, k, v, dout, lse, delta, cu_q, cu_kv, is_causal, scale, dtype,
             a2_fast=True, prefill=None, repeat=1, head_slab=None):
    """``_run_dev`` with the three gradients copied back to the host."""
    out = _run_dev(q, k, v, dout, lse, delta, cu_q, cu_kv, is_causal, scale, dtype,
                   a2_fast=a2_fast, prefill=prefill, repeat=repeat,
                   head_slab=head_slab)
    return (out["dq_acc"], out["dk"], out["dv"], out["oob"], out["elapsed"],
            out["plan"])


def _check(got, ref, atol, rtol):
    """``(ok, exceeded_cells, max_gap)`` of one tensor against one reference."""
    a, e = got.float(), ref.float()
    if a.numel() == 0:
        return True, 0, 0.0
    gap = torch.nan_to_num((a - e).abs(), nan=0.0, posinf=0.0, neginf=0.0)
    bad = int((gap > atol + rtol * e.abs()).sum())
    return bad == 0, bad, float(gap.max())


def _segment_report(got, ref, atol, rtol, cu, batch):
    """Per-request ``(max_diff, ok)`` list; used to name a polluting request."""
    rows = []
    offs = cu.tolist()
    for b in range(batch):
        lo, hi = offs[b], offs[b + 1]
        if hi <= lo:
            rows.append((0.0, True))
            continue
        a, e = got[lo:hi].float(), ref[lo:hi].float()
        gap = torch.nan_to_num((a - e).abs(), nan=0.0, posinf=0.0, neginf=0.0)
        bad = int((gap > atol + rtol * e.abs()).sum())
        rows.append((float(gap.max()), bad == 0))
    return rows


def _ulp_stats(got, ref):
    """``(max_gap, k_ulp)`` with ``k_ulp = max|d| / ulp(max|ref|)`` -- diagnostic.

    DESIGN.md 8.2 (5): the fp32 ``dq_acc`` is accumulated by store-atomic adds, so
    the arrival order of the key blocks is unspecified and two runs cannot be
    bitwise equal.  ``k_ulp`` answers "is the jitter still a couple of ulps of the
    tensor scale"; it is **not** a pass criterion: ``k_ulp > 8`` only warns.  The
    element-wise normalization ``max_i |d_i| / ulp(|ref_i|)`` is deliberately not
    used: cells whose ideal value is ~0 make it explode (a 1 ulp of the tensor
    scale is then thousands of "ulps" of that cell) without carrying information.
    """
    a, e = got.float(), ref.float()
    if a.numel() == 0:
        return 0.0, 0.0
    gap = torch.nan_to_num((a - e).abs(), nan=0.0, posinf=0.0, neginf=0.0)
    max_ref = max(float(e.abs().max()), 1e-30)
    ulp = 2.0 ** (math.floor(math.log2(max_ref)) - 23)  # fp32 ulp at max|ref|
    return float(gap.max()), float(gap.max()) / ulp


def _check_plan(plan, batch, total_kv, heads, expect):
    """Assert the planned launch set of one case (DESIGN.md 8.3 / 8.4 support)."""
    got = {
        "tiles": plan.tiles,
        "slab": plan.slab,
        "launches": len(plan.launches),
        "max_grid": plan.max_grid,
        "idle": plan.idle_blocks,
        "useful": plan.useful_blocks,
    }
    for key, want in expect.items():
        if got[key] != want:
            raise AssertionError(
                f"launch plan: {key}={got[key]} != expected {want} "
                f"(batch={batch}, total_kv={total_kv}, heads={heads}) | {describe(plan)}"
            )
    if plan.max_grid > NPU_GRID_MAX:
        raise AssertionError(f"launch plan: max_grid {plan.max_grid} > {NPU_GRID_MAX}")


def run_case(
    level,
    name,
    q_lens,
    kv_lens,
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
    head_slab=None,
    expect=None,
):
    """Build one packed geometry, run it on the NPU and compare all three gradients."""
    diff = float("nan")
    scale = dim_qk**-0.5 if scale is None else scale
    batch = len(q_lens)
    tdt = _TORCH_DTYPE[dtype]
    try:
        if batch != len(kv_lens):
            raise ValueError(f"shape: q_lens/kv_lens length mismatch ({batch} vs {len(kv_lens)})")
        q, k, v, dout, cu_q, cu_kv = _build_packed(
            q_lens, kv_lens, heads, dim_qk, dim_v, dtype, seed, special
        )
        slab = _slab(max(max(q_lens), max(kv_lens)))
        dq_true, dk_true, dv_true, dq_acc_true = golden_bwd_kernel(
            q, k, v, dout, cu_q, cu_kv, scale=scale, is_causal=is_causal, dtype=tdt,
            slab_heads=slab,
        )
        lse, delta = forward_stats(q, k, v, dout, cu_q, cu_kv, is_causal=is_causal,
                                   slab_heads=slab)

        prefill = 2.0 if special == "atom_prefill" else None
        got_dq32, got_dk, got_dv, oob, elapsed, plan = _execute(
            q, k, v, dout, lse, delta, cu_q, cu_kv, is_causal, scale, dtype,
            a2_fast=a2_fast, prefill=prefill, repeat=repeat, head_slab=head_slab,
        )
        if expect:
            _check_plan(plan, batch, sum(kv_lens), heads, expect)
        if oob:
            raise AssertionError(f"{oob} cell(s) written outside the valid range (OOB store)")
        if prefill is not None:
            got_dq32 = got_dq32 - prefill  # proves add-on-top-of-buffer semantics

        got_dq = (got_dq32 * scale).to(tdt)
        d_dk, nan_dk, nf_dk = _max_diff(got_dk, dk_true)
        d_dv, nan_dv, nf_dv = _max_diff(got_dv, dv_true)
        d_dq, nan_dq, nf_dq = _max_diff(got_dq32, dq_acc_true)
        diff = max(d_dk, d_dv, d_dq)
        _MAX_DIFF[0] = max(_MAX_DIFF[0], diff)
        bad = nan_dk + nan_dv + nan_dq
        if bad:
            raise AssertionError(
                f"{bad} NaN position(s) disagree with the golden (max_diff={diff:.3e})"
            )
        nonfinite = nf_dk + nf_dv + nf_dq
        rtol = _RTOL[dtype]
        # float32 dq_acc intermediate against its own (scale-free) reference.
        # DESIGN.md 8.2 is the single tolerance source: dq_acc uses
        # atol = 5e-3 / scale (~6.9e-2 = 565 fp32 ulp at |dq_acc| ~ 1.05e3) and
        # rtol 1e-3, while the 16-bit outputs use the input dtype's rtol.  No
        # fixed absolute threshold may be tighter than that table (8.2 2).
        tol_dq = _ATOL / scale
        rtol_acc = _RTOL_ACC
        note = ""
        ref_dk, ref_dv, ref_acc = dk_true, dv_true, dq_acc_true
        try:
            torch.testing.assert_close(got_dk, dk_true, atol=_ATOL, rtol=rtol, equal_nan=True)
            torch.testing.assert_close(got_dv, dv_true, atol=_ATOL, rtol=rtol, equal_nan=True)
            torch.testing.assert_close(got_dq, dq_true, atol=_ATOL, rtol=rtol, equal_nan=True)
            torch.testing.assert_close(got_dq32, dq_acc_true, atol=tol_dq, rtol=rtol_acc,
                                       equal_nan=True)
        except AssertionError:
            # Second opinion: the kernel (like the source GPU kernel) rounds P and
            # dS to the input dtype before the three output contractions because
            # the GEMM operands are 16-bit.  A handful of strongly cancelling
            # cells can therefore sit outside atol of the fp64 ideal.  Accept the
            # case only if (a) it matches that source-semantics reference and
            # (b) the number of such cells is a negligible fraction.
            _, dk_q, dv_q, acc_q = golden_bwd_kernel(
                q, k, v, dout, cu_q, cu_kv, scale=scale, is_causal=is_causal,
                dtype=tdt, slab_heads=slab, quantize=True,
            )
            primary = (
                ("dk", got_dk, dk_true, _ATOL, rtol),
                ("dv", got_dv, dv_true, _ATOL, rtol),
                ("dq", got_dq, dq_true, _ATOL, rtol),
                ("dq32", got_dq32, dq_acc_true, tol_dq, rtol_acc),
            )
            second = (
                ("dk", got_dk, dk_q, _ATOL, rtol),
                ("dv", got_dv, dv_q, _ATOL, rtol),
                ("dq32", got_dq32, acc_q, tol_dq, rtol_acc),
            )
            bad_total = sum(_check(g, r, a, rr)[1] for _, g, r, a, rr in primary)
            numel = sum(g.numel() for _, g, _, _, _ in primary)
            sec_bad = sum(_check(g, r, a, rr)[1] for _, g, r, a, rr in second)
            allowance = max(1, numel // 20000)  # <= 5e-5 of the compared cells
            if sec_bad or bad_total > allowance:
                raise
            detail = ", ".join(
                f"{tag} {cnt}x{gap:.2e}" for tag, _, cnt, gap in
                ((t,) + _check(g, r, a, rr) for t, g, r, a, rr in primary) if cnt
            )
            note = (f" {bad_total}/{numel} cell(s) outside the fp64 atol ({detail}): "
                    "16-bit-operand boundary, matches the source-semantics reference")
            ref_dk, ref_dv, ref_acc = dk_q, dv_q, acc_q
        if per_request:
            faults = []
            for tag, got, ref, cu, atol, rr in (
                ("dk", got_dk, ref_dk, cu_kv, _ATOL, rtol),
                ("dv", got_dv, ref_dv, cu_kv, _ATOL, rtol),
                ("dq", got_dq32, ref_acc, cu_q, tol_dq, rtol_acc),
            ):
                for b, (seg_diff, ok) in enumerate(
                    _segment_report(got, ref, atol, rr, cu, batch)
                ):
                    if not ok:
                        faults.append(f"request {b} {tag} max_diff={seg_diff:.3e}")
            if faults:
                raise AssertionError("per-request mismatch: " + "; ".join(faults))
            note += " per-request-clean"
    except AssertionError as exc:
        _record(level, name, False, f"max_diff={diff:.3e} {str(exc)[:400]}")
        return False
    except (ValueError, RuntimeError) as exc:  # noqa: BLE001 - reported, never swallowed
        _record(level, name, False, f"{type(exc).__name__}: {str(exc)[:400]}")
        return False
    _record(level, name, True)
    print(
        f"       q={q_lens} kv={kv_lens} H={heads} D={dim_qk}/{dim_v} {dtype} "
        f"causal={int(is_causal)} T_q={sum(q_lens)} T_kv={sum(kv_lens)} "
        f"plan[{describe(plan)}] max_diff={diff:.3e} "
        f"(dk {d_dk:.2e} dv {d_dv:.2e} dq {d_dq:.2e}) launch={elapsed:.3f} ms "
        f"(nonfinite cells {nonfinite}){note}",
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
# 4. Probe-style cases
# ---------------------------------------------------------------------------
def _case_contract(level, name):
    """Factory rejects contracts it cannot serve, and caches per geometry+slab."""
    problems = []
    scale = 192**-0.5
    for args, why in (
        ((2, 4, 192, 128, 0, 0, True, scale, "float32"), "float32 inputs"),
        ((2, 4, 192, 128, 64, 64, True, scale, "float16"), "dense_seqlen_q=64"),
        ((0, 4, 192, 128, 0, 0, True, scale, "float16"), "batch=0"),
        ((2, 4, 100, 128, 0, 0, True, scale, "float16"), "dim_qk not a multiple of 32"),
        # the head slab is a compile-time launch parameter: an out-of-range value
        # would change the grid silently, so the factory refuses it (DESIGN 5.6 b)
        ((2, 2, 192, 128, 0, 0, True, scale, "float16", True, 0), "head_slab=0"),
        ((2, 2, 192, 128, 0, 0, True, scale, "float16", True, 4), "head_slab > heads"),
    ):
        try:
            _bwd_kernel(*args)
            problems.append(f"{why}: no error raised")
        except ValueError:
            pass
    first = _bwd_kernel(2, 2, 192, 128, 0, 0, True, scale, "float16", True, 2)
    again = _bwd_kernel(2, 2, 192, 128, 0, 0, True, scale, "float16", True, 2)
    if first is not again:
        problems.append("the compiled program is not cached per (geometry, dtype, slab)")
    other = _bwd_kernel(2, 2, 192, 128, 0, 0, True, scale, "float16", True, 1)
    if other is first:
        problems.append("a different head_slab returned the same compiled program")
    if problems:
        _record(level, name, False, "; ".join(problems))
        return False
    _record(level, name, True)
    print("       6 invalid contracts rejected (incl. head_slab), program cached per "
          "(geometry, dtype, slab), slab 1 != slab 2 program", flush=True)
    return True


def _case_golden_naive(level, name):
    """Self-check the golden against an explicit per-pair loop (tiny packed batch)."""
    problems = []
    for q_lens, kv_lens, is_causal in (([6, 8], [8, 4], True), ([5, 3], [5, 7], False)):
        q, k, v, do, cu_q, cu_kv = _build_packed(q_lens, kv_lens, 2, 8, 6, "float16", 61)
        scale = 8**-0.5
        dq, dk, dv, _ = golden_bwd_kernel(q, k, v, do, cu_q, cu_kv, scale=scale,
                                          is_causal=is_causal, dtype=torch.float64)
        qf, kf, vf, dof = (t.double() for t in (q, k, v, do))
        r_dq = torch.zeros_like(dq)
        r_dk = torch.zeros_like(dk)
        r_dv = torch.zeros_like(dv)
        bq, bkv = cu_q.tolist(), cu_kv.tolist()
        for bidx in range(len(bq) - 1):
            q0, k0 = bq[bidx], bkv[bidx]
            len_q, len_kv = bq[bidx + 1] - q0, bkv[bidx + 1] - k0
            shift = len_kv - len_q
            for h in range(2):
                for m in range(len_q):
                    scores = []
                    for n in range(len_kv):
                        s = float((qf[q0 + m, h] * kf[k0 + n, h]).sum()) * scale
                        visible = not is_causal or n <= m + shift
                        scores.append(s if visible else None)
                    vis = [s for s in scores if s is not None]
                    lse = math.log(sum(math.exp(s) for s in vis)) if vis else None
                    p = torch.zeros(len_kv, dtype=torch.float64)
                    if lse is not None:
                        for n, s in enumerate(scores):
                            if s is not None:
                                p[n] = math.exp(s - lse)
                    dp = torch.tensor(
                        [float((dof[q0 + m, h] * vf[k0 + n, h]).sum()) for n in range(len_kv)],
                        dtype=torch.float64,
                    )
                    ds = p * (dp - float((p * dp).sum()))
                    r_dv[k0:k0 + len_kv, h] += p[:, None] * dof[q0 + m, h][None, :]
                    r_dk[k0:k0 + len_kv, h] += ds[:, None] * qf[q0 + m, h][None, :]
                    r_dq[q0 + m, h] = (ds[:, None] * kf[k0:k0 + len_kv, h]).sum(0) * scale
        r_dk = r_dk * scale
        for got, ref, tag in ((dq, r_dq, "dq"), (dk, r_dk, "dk"), (dv, r_dv, "dv")):
            gap = (got - ref).abs().max().item() if ref.numel() else 0.0
            if gap > 1e-12:
                problems.append(f"q={q_lens}/kv={kv_lens} causal={int(is_causal)} {tag}: {gap:.3e}")
    if problems:
        _record(level, name, False, "; ".join(problems))
        return False
    _record(level, name, True)
    print("       golden == explicit per-pair loop on 2 packed geometries (1e-12)", flush=True)
    return True


def _case_atomic_contract(level, name, q_lens=(96, 128), kv_lens=(96, 128), heads=2,
                          expect=None):
    """Store-atomic adds onto the caller's buffer instead of overwriting it.

    The prefilled arm proves "add on top of the caller's buffer", the zeroed arm is
    the plain result; the prefilled output has the prefill subtracted once (it is
    added exactly once, before the first slab launch) and is then compared with the
    zeroed one under the 8.2 tolerance -- a bitwise ``== prefill`` is not a legal
    criterion for a store-atomic accumulator (DESIGN.md 8.2 (1)).
    """
    ok = run_case(
        level, name + "-prefilled", list(q_lens), list(kv_lens), heads, "float16",
        seed=91, special="atom_prefill", per_request=True, expect=expect,
    )
    ok = run_case(level, name + "-zeroed", list(q_lens), list(kv_lens), heads, "float16",
                  seed=91, per_request=True, expect=expect) and ok
    return ok


def _case_arm_equivalence(level, name):
    """Force the masked arm and compare against the specialised build.

    DESIGN.md 8.3 L0-12: the two arms produce identical ``p`` element-wise (A2,
    violation rate 0) and the atomic arrival order is the same inside one arm, so
    ``dk/dv`` must be **bitwise** equal, while ``dq_acc`` is judged with the 8.2
    dtype tolerance (atol 6.9e-2 + rtol 1e-3 * |ref|) -- never with a fixed
    absolute threshold (the v1 ``dq <= 1e-5`` was 0.08 fp32 ulp, see 12.6 item 1).
    ``k_ulp`` is printed as a diagnostic; ``> 8`` warns but does not fail.
    """
    problems = []
    for q_lens, kv_lens, dtype in (([128, 160], [128, 160], "bfloat16"),
                                   ([96], [192], "bfloat16")):
        outs = []
        scale = 192**-0.5
        tol_acc = _ATOL / scale
        q, k, v, do, cu_q, cu_kv = _build_packed(q_lens, kv_lens, 2, 192, 128, dtype, 51)
        lse, delta = forward_stats(q, k, v, do, cu_q, cu_kv, slab_heads=2)
        for a2_fast in (True, False):
            got = _execute(q, k, v, do, lse, delta, cu_q, cu_kv, True, scale, dtype,
                           a2_fast=a2_fast)
            if got[3]:
                problems.append(f"q={q_lens} a2_fast={a2_fast}: {got[3]} OOB cell(s)")
            outs.append((got[0], got[1], got[2]))
        # idx 0 is the fp32 dq_acc accumulator (dq before the landing kernel).
        for idx, tag in ((0, "dq_acc"), (1, "dk"), (2, "dv")):
            a, b = outs[0][idx], outs[1][idx]
            gap, k_global = _ulp_stats(a, b)
            if tag == "dq_acc":
                bad = _check(a, b, tol_acc, _RTOL_ACC)[1]
                if bad:
                    problems.append(
                        f"q={q_lens}/{kv_lens} {tag}: arms differ on {bad} cell(s) "
                        f"(max {gap:.3e}, k_ulp {k_global:.2f})"
                    )
                if k_global > 8.0:
                    diff = (a.float() - b.float()).abs().flatten()
                    top = torch.topk(diff, min(10, diff.numel()))
                    print(
                        f"       [WARN] q={q_lens}/{kv_lens} {tag}: k_ulp={k_global:.2f} > 8 "
                        f"(diagnostic only) top gaps {[round(float(x), 6) for x in top.values]}",
                        flush=True,
                    )
                _ULP[0] = max(_ULP[0], k_global)
            elif not torch.equal(a, b):
                problems.append(f"q={q_lens}/{kv_lens} {tag}: arms differ (max {gap:.3e})")
    if problems:
        _record(level, name, False, "; ".join(problems))
        return False
    _record(level, name, True)
    print(
        "       fast arm == masked arm: dk/dv bitwise, dq_acc within 8.2 tolerance "
        f"(k_ulp max {_ULP[0]:.2f}, diagnostic) on 2 geometries",
        flush=True,
    )
    return True


# ---------------------------------------------------------------------------
# 4b. Grid-cap regressions (DESIGN.md 8.4 G1-G5, 8.3 L0-20 ... L0-23)
#
# Why a *kernel-vs-kernel* reference and not the fp64 golden: at H=128/L=4096 the
# reference's score plane (128 x 4096^2) does not fit the container, and -- more
# importantly -- a reference that walks the same 66,560-block launch would be
# polluted by the same truncation.  The reference is therefore the *same kernel*
# with **one request per launch** (B=1 -> grid 2,080, three decades below the cap),
# compared against the rows of the full run (DESIGN.md 8.4 preamble / G1).
# ---------------------------------------------------------------------------
_MANIFEST_8X4K_TFLOPS = 14.34  # manifest roofline of ds-v3-8x4k (B=8,H=128,L=4096)
# With block_n=96 the 8x4k launch is 44,672 blocks (4 x 11,168) and the cube
# design projects 153-165 ms; the v3 Vector kernel measured 488 ms.  A truncated
# launch (the 65,536 block-index wrap) finishes in ~20 ms, so 120 ms is the floor.
_GRID_MAX_MS = 120.0
_TFLOPS_MAX = 200.0  # ~2x the projected ~90 TFLOPS; anything above is not real


def _acc_report(got, ref, atol, rtol):
    """``(bad_cells, max_gap, k_ulp)`` under the DESIGN 8.2 tolerance form."""
    gap = torch.nan_to_num((got.float() - ref.float()).abs(), nan=0.0, posinf=0.0,
                           neginf=0.0)
    bad = int((gap > atol + rtol * ref.float().abs()).sum())
    _, k_ulp = _ulp_stats(got, ref)
    return bad, float(gap.max()), k_ulp


def _case_launch_plan(level, name):
    """L0-20 / G5: overflow raises, the legal bound passes, plan == artifact."""
    problems = []
    # (1)(2) even slab == 1 cannot fit -> ValueError, never a silent truncation.
    # block_n=96: tiles = total_kv // 96 + batch, so the raise happens at
    # 65_529 tiles (8 * 96 * 65536/8 ... see the two anchors below).
    for args in ((8, 6_290_688, 128), (1, 65_536 * 96, 1)):
        try:
            plan_launch(*args)
            problems.append(f"plan_launch{args} returned instead of raising")
        except ValueError as exc:
            if "tile_upper_bound" not in str(exc):
                problems.append(f"plan_launch{args}: message lacks tile_upper_bound: {exc}")
    # (3) the legal upper bound itself: tiles == 65,535 -> slab 1, grid 65,535
    try:
        plan = plan_launch(8, 6_290_592, 128)
        if (plan.slab, plan.max_grid) != (1, 65_535):
            problems.append(f"upper bound: (slab, max_grid) = {(plan.slab, plan.max_grid)}")
    except ValueError as exc:
        problems.append(f"upper bound raised: {exc}")
    # (4) a long token count shrinks the slab instead of overflowing
    # (201,600 tokens -> tiles 2,108 -> slab 31, 5 launches, grid 65,348)
    try:
        plan = plan_launch(8, 201_600, 128)
        if (plan.slab, plan.max_grid, len(plan.launches)) != (31, 65_348, 5):
            problems.append(f"auto-shrink: (slab, max_grid, launches) = "
                            f"{(plan.slab, plan.max_grid, len(plan.launches))}")
    except ValueError as exc:
        problems.append(f"auto-shrink raised: {exc}")
    # (5) drift: the inlined copy and launch_plan.py agree field by field.  The
    # module is the unique source in the task directory; the TileOPs port ships
    # only the inlined copy, so a missing module is a warning, not a failure.
    if _launch_plan_module is None:
        print("       [WARN] launch_plan.py not importable here: the drift check is "
              "skipped (the inlined copy is authoritative in the port)", flush=True)
    else:
        for geometry in ((8, 32_768, 128), (6, 32_256, 128), (4, 32_768, 64),
                         (2, 640, 8), (8, 201_600, 128), (1, 1_024, 64)):
            mine, theirs = plan_launch(*geometry), _launch_plan_module.plan_launch(*geometry)
            for field in ("tiles", "slab", "total_blocks", "useful_blocks", "idle_blocks"):
                if getattr(mine, field) != getattr(theirs, field):
                    problems.append(f"{geometry}: {field} {getattr(mine, field)} != "
                                    f"{getattr(theirs, field)}")
            if len(mine.launches) != len(theirs.launches):
                problems.append(f"{geometry}: {len(mine.launches)} launches != "
                                f"{len(theirs.launches)}")
            for lm, lt in zip(mine.launches, theirs.launches):
                for field in ("index", "head_off", "active_heads", "grid", "idle_blocks"):
                    if getattr(lm, field) != getattr(lt, field):
                        problems.append(f"{geometry}: launch {field} "
                                        f"{getattr(lm, field)} != {getattr(lt, field)}")
    # (6) the post-assert: plan, factory metadata and cfg share one source
    scale = 192**-0.5
    fn, plan, cfg = build_kernel(1, 64, 192, 128, True, scale, "float16", 1_024)
    meta = _factory_meta(fn)
    if (meta["head_slab"], meta["block_n"]) != (plan.slab, cfg["block_n"]):
        problems.append(f"build_kernel meta {meta} disagrees with plan.slab={plan.slab} "
                        f"cfg block_n={cfg['block_n']}")
    if (plan.tiles, plan.slab, plan.max_grid, len(plan.launches)) != (11, 32, 352, 2):
        problems.append(f"multi-launch geometry: {describe(plan)}")
    # 6a: a real artifact built for a different slab must be rejected
    plan16 = plan_launch(1, 1_024, 64, head_slab=16)
    try:
        assert_launch_consistency(plan16, meta, batch=1, total_kv=1_024,
                                  block_n=cfg["block_n"])
        problems.append("slab mismatch (plan 16 vs artifact 32) passed the post-assert")
    except ValueError:
        pass
    # 6b: DESIGN.md 5.6 g counter-example -- plan at block_n=96, artifact at 48:
    # (131,072 // 48 + 8) * 32 = 87,616 >> 65,535 while the plan says 43,936.
    stale = plan_launch(8, 131_072, 128)
    try:
        assert_launch_consistency(stale, {"head_slab": stale.slab, "block_n": 48},
                                  batch=8, total_kv=131_072, block_n=48)
        problems.append("block_n mismatch (plan 64 vs artifact 32) passed the post-assert")
    except ValueError:
        pass
    if problems:
        _record(level, name, False, "; ".join(problems)[:600])
        return False
    _record(level, name, True)
    print("       2 overflow raises, bound 65,535 legal, auto-shrink slab=31/65,348, "
          "inlined plan == launch_plan.py on 6 geometries, post-assert rejects 2 "
          "deliberate mismatches", flush=True)
    return True


def _one_request(q, k, v, dout, lse, delta, cu_q, cu_kv, b):
    """The packed inputs of request ``b`` as their own B=1 call (offsets from 0)."""
    q0, q1 = int(cu_q[b]), int(cu_q[b + 1])
    k0, k1 = int(cu_kv[b]), int(cu_kv[b + 1])
    return {
        "q": q[q0:q1], "k": k[k0:k1], "v": v[k0:k1], "dout": dout[q0:q1],
        "lse": lse[:, q0:q1].contiguous(), "delta": delta[:, q0:q1].contiguous(),
        "cu_q": torch.tensor([0, q1 - q0], dtype=torch.int32),
        "cu_kv": torch.tensor([0, k1 - k0], dtype=torch.int32),
        "rows": (q0, q1), "kv_rows": (k0, k1),
    }


def _split_reference(q, k, v, dout, lse, delta, cu_q, cu_kv, is_causal, scale, dtype,
                     a2_fast=True, head_slab=None, tag="G1"):
    """One launch set per request: the grid stays three decades below the cap."""
    parts = []
    for b in range(int(cu_q.numel()) - 1):
        one = _one_request(q, k, v, dout, lse, delta, cu_q, cu_kv, b)
        res = _run_dev(one["q"], one["k"], one["v"], one["dout"], one["lse"], one["delta"],
                       one["cu_q"], one["cu_kv"], is_causal, scale, dtype,
                       a2_fast=a2_fast, head_slab=head_slab, keep_device=True)
        res["rows"], res["kv_rows"] = one["rows"], one["kv_rows"]
        parts.append(res)
        print(f"       [{tag}] ref request {b}: rows {one['rows']} kv {one['kv_rows']} "
              f"plan[{describe(res['plan'])}]", flush=True)
    return parts


def _per_head_report(full, parts, scale, dtype, atol, rtol):
    """V5: ``dq_acc`` sliced per (request, head) -- every head must be clean.

    The store-atomic accumulator is the only output whose value depends on the
    *arrival order* of the key blocks, so a slab/head mapping bug shows up as one
    head that never received its blocks (all-zero or a large gap) while the whole
    tensor still looks acceptable.  Returns ``(bad_heads, worst_head, worst_gap,
    heads)``; ``bad_heads`` counts heads with any cell outside the 8.2 tolerance.
    """
    heads = full["heads"]
    tdt = _TORCH_DTYPE[dtype]
    bad_heads, worst_h, worst_gap = 0, -1, 0.0
    for b, part in enumerate(parts):
        q0, q1 = part["rows"]
        got_all = full["dq_acc"][q0:q1]
        ref_all = part["dq_acc"]
        for h in range(heads):
            gap = float(torch.nan_to_num((got_all[:, h].float() - ref_all[:, h].float()).abs(),
                                         nan=0.0, posinf=0.0, neginf=0.0).max())
            bad = int((torch.nan_to_num((got_all[:, h].float() - ref_all[:, h].float()).abs(),
                                        nan=0.0, posinf=0.0, neginf=0.0)
                       > atol + rtol * ref_all[:, h].float().abs()).sum())
            if gap > worst_gap:
                worst_gap, worst_h = gap, h
            if bad:
                bad_heads += 1
    return bad_heads, worst_h, worst_gap, heads


def _compare_split(level, name, full, parts, scale, dtype, energy=True, per_head=False):
    """Full-run rows vs the per-request reference: G1 steps (3) and (4).

    ``dk/dv`` are **bitwise** compared (assignment writes, one owner per
    ``(tile, head)``); ``dq_acc``/``dq`` use the DESIGN 8.2 tolerance and print
    ``k_ulp`` as a diagnostic.  ``energy`` additionally requires every head of
    every request to carry non-zero gradient: a truncated launch leaves whole
    requests (or heads) exactly zero, which no tolerance discussion can hide.
    """
    problems = []
    tdt = _TORCH_DTYPE[dtype]
    tol_acc = _ATOL / scale
    rtol = _RTOL[dtype]
    heads = full["heads"]
    worst = {"dk": 0.0, "dv": 0.0, "dq_acc": 0.0, "dq": 0.0}
    rows_out = []
    for b, part in enumerate(parts):
        q0, q1 = part["rows"]
        k0, k1 = part["kv_rows"]
        row = {"request": b, "rows": (q0, q1)}
        for tag in ("dk", "dv"):
            got, ref = full[tag][k0:k1], part[tag]
            gap = float(torch.nan_to_num((got.float() - ref.float()).abs(),
                                         nan=0.0, posinf=0.0, neginf=0.0).max())
            row[tag] = gap
            worst[tag] = max(worst[tag], gap)
            if not torch.equal(got, ref):
                problems.append(f"request {b} {tag}: not bitwise identical (max {gap:.3e})")
        for tag, got, ref, atol, rrel in (
            ("dq_acc", full["dq_acc"][q0:q1], part["dq_acc"], tol_acc, _RTOL_ACC),
            ("dq", (full["dq_acc"][q0:q1] * scale).to(tdt),
             (part["dq_acc"] * scale).to(tdt), _ATOL, rtol),
        ):
            bad, gap, k_ulp = _acc_report(got, ref, atol, rrel)
            row[tag] = gap
            row[tag + "_k_ulp"] = k_ulp
            worst[tag] = max(worst[tag], gap)
            _ULP[0] = max(_ULP[0], k_ulp)
            if bad:
                problems.append(f"request {b} {tag}: {bad} cell(s) outside "
                                f"atol {atol:.3e} + rtol {rrel:.1e}|ref| "
                                f"(max {gap:.3e}, k_ulp {k_ulp:.2f})")
        if energy:
            for tensor, gname, lo, hi in ((full["dk"], "dk", k0, k1),
                                          (full["dv"], "dv", k0, k1),
                                          (full["dq_acc"], "dq", q0, q1)):
                sums = tensor[lo:hi].abs().sum(dim=(0, 2))  # (heads,)
                empty = int((sums == 0).sum())
                if empty:
                    problems.append(f"request {b}: {empty} of {heads} {gname} head(s) "
                                    "are all zero (launch truncated?)")
        rows_out.append(row)
        print(f"       [{name}] req {b} rows {q0}:{q1}: dk {row['dk']:.3e} (bitwise) "
              f"dv {row['dv']:.3e} (bitwise) dq_acc {row['dq_acc']:.3e} "
              f"(k_ulp {row['dq_acc_k_ulp']:.2f}) dq {row['dq']:.3e}", flush=True)
    if problems:
        _record(level, name, False, "; ".join(problems)[:600])
        return False, worst, rows_out
    if per_head:
        bad_heads, worst_h, worst_gap, heads = _per_head_report(
            full, parts, scale, dtype, _ATOL / scale, _RTOL_ACC)
        if bad_heads:
            problems.append(f"per-head dq_acc: {bad_heads}/{heads} head(s) outside 8.2 "
                            f"(worst head {worst_h}, max {worst_gap:.3e})")
        else:
            print(f"       [{name}] per-head dq_acc: {heads} heads x "
                  f"{len(parts)} request(s) clean (worst head {worst_h}, "
                  f"max {worst_gap:.3e})", flush=True)
    if problems:
        _record(level, name, False, "; ".join(problems)[:600])
        return False, worst, rows_out
    _record(level, name, True)
    print(f"       [{name}] {len(parts)} request(s): dk/dv bitwise, max_abs_err "
          f"dk {worst['dk']:.3e} dv {worst['dv']:.3e} dq_acc {worst['dq_acc']:.3e} "
          f"dq {worst['dq']:.3e}, all heads non-zero", flush=True)
    return True, worst, rows_out


def _case_grid_cap_split(level, name, q_lens, kv_lens, heads, dtype, seed, tag, expect):
    """G1/G2: the manifest geometry at full H, split per request for the reference."""
    total_kv = sum(kv_lens)
    batch = len(q_lens)
    scale = 192**-0.5
    q, k, v, dout, cu_q, cu_kv = _build_packed(q_lens, kv_lens, heads, 192, 128, dtype, seed)
    lse, delta = forward_stats(q, k, v, dout, cu_q, cu_kv, is_causal=True, slab_heads=1)
    try:
        full = _run_dev(q, k, v, dout, lse, delta, cu_q, cu_kv, True, scale, dtype,
                        keep_device=True)
        _check_plan(full["plan"], batch, total_kv, heads, expect)
        print(f"       [{tag}] full plan[{describe(full['plan'])}] "
              f"launch={full['elapsed']:.3f} ms oob={full['oob']}", flush=True)
        if full["oob"]:
            raise AssertionError(f"{full['oob']} cell(s) written outside the valid range")
        parts = _split_reference(q, k, v, dout, lse, delta, cu_q, cu_kv, True, scale,
                                 dtype, tag=tag)
        ok, worst, rows = _compare_split(level, name, full, parts, scale, dtype,
                                         per_head=heads >= 32)
        if tag == "G1":
            _G1_ROWS.clear()
            _G1_ROWS.extend(rows)
            # G4 needs a *warm* measurement: the first launch of a freshly compiled
            # program carries the one-time device upload (~145 ms on this stack) and
            # would bias the tripwire.  The second launch set is also REVIEW's
            # "repeat 2x" check: dk/dv are assignments (bitwise), dq_acc is a
            # store-atomic accumulator (8.2 tolerance + k_ulp diagnostic).
            warm = _run_dev(q, k, v, dout, lse, delta, cu_q, cu_kv, True, scale, dtype,
                            keep_device=True)
            _G4_ELAPSED[0] = min(full["elapsed"], warm["elapsed"])
            _G4_PLAN[0] = warm["plan"]
            msgs = []
            for tag_ in ("dk", "dv"):
                if not torch.equal(full[tag_], warm[tag_]):
                    msgs.append(f"repeat {tag_}: not bitwise identical")
            bad, gap, k_ulp = _acc_report(full["dq_acc"], warm["dq_acc"],
                                          _ATOL / scale, _RTOL_ACC)
            print(f"       [{tag}] repeat: dk/dv bitwise, dq_acc max {gap:.3e} "
                  f"k_ulp {k_ulp:.2f} ({bad} cell(s) outside 8.2), "
                  f"elapsed {full['elapsed']:.2f} / {warm['elapsed']:.2f} ms",
                  flush=True)
            if bad:
                msgs.append(f"repeat dq_acc: {bad} cell(s) outside 8.2 (max {gap:.3e})")
            del warm
            if msgs:
                _record(level, name + "-repeat", False, "; ".join(msgs))
                return False
            _record(level, name + "-repeat", True)
        return ok
    finally:
        del q, k, v, dout, lse, delta
        try:
            del full
        except UnboundLocalError:
            pass


def _case_head_slab_mapping(level, name, q_lens, kv_lens, heads, dtype, seed, slabs,
                            expect_tail, tag):
    """G3 / L0-21: the same geometry under different head slabs must agree.

    ``dk/dv`` are assignment writes with a single owner per ``(tile, head)``, so a
    different slab partitioning cannot change a bit; ``dq_acc`` only differs in
    atomic arrival order (8.2 tolerance + ``k_ulp`` diagnostic).  ``expect_tail``
    pins the tail-slab plan (``idle`` blocks) of every slab.
    """
    scale = 192**-0.5
    total_kv = sum(kv_lens)
    batch = len(q_lens)
    q, k, v, dout, cu_q, cu_kv = _build_packed(q_lens, kv_lens, heads, 192, 128, dtype, seed)
    lse, delta = forward_stats(q, k, v, dout, cu_q, cu_kv, is_causal=True, slab_heads=1)
    runs = {}
    problems = []
    for slab in slabs:
        res = _run_dev(q, k, v, dout, lse, delta, cu_q, cu_kv, True, scale, dtype,
                       head_slab=slab, keep_device=True)
        plan = res["plan"]
        print(f"       [{tag}] head_slab={slab} plan[{describe(plan)}] "
              f"launch={res['elapsed']:.3f} ms oob={res['oob']}", flush=True)
        if res["oob"]:
            problems.append(f"head_slab={slab}: {res['oob']} OOB cell(s)")
        if plan.slab not in expect_tail:
            problems.append(f"head_slab={slab}: plan.slab {plan.slab} not in "
                            f"{sorted(expect_tail)}")
        elif (len(plan.launches), plan.idle_blocks) != expect_tail[plan.slab]:
            problems.append(f"head_slab={slab}: (launches, idle) = "
                            f"{(len(plan.launches), plan.idle_blocks)} != "
                            f"{expect_tail[plan.slab]}")
        runs[slab] = res
    base = runs[slabs[0]]
    for slab in slabs[1:]:
        other = runs[slab]
        for tag_ in ("dk", "dv"):
            if not torch.equal(base[tag_], other[tag_]):
                gap = float((base[tag_].float() - other[tag_].float()).abs().max())
                problems.append(f"slab {slabs[0]} vs {slab} {tag_}: not bitwise "
                                f"identical (max {gap:.3e})")
        bad, gap, k_ulp = _acc_report(base["dq_acc"], other["dq_acc"],
                                      _ATOL / scale, _RTOL_ACC)
        _ULP[0] = max(_ULP[0], k_ulp)
        print(f"       [{tag}] slab {slabs[0]} vs {slab}: dk/dv bitwise, dq_acc max "
              f"{gap:.3e} k_ulp {k_ulp:.2f} ({bad} cell(s) outside 8.2)", flush=True)
        if bad:
            problems.append(f"slab {slabs[0]} vs {slab} dq_acc: {bad} cell(s) outside "
                            f"8.2 (max {gap:.3e}, k_ulp {k_ulp:.2f})")
    parts = _split_reference(q, k, v, dout, lse, delta, cu_q, cu_kv, True, scale, dtype,
                             tag=tag)
    ok, worst, rows = _compare_split(level, name + "-per-request", base, parts, scale, dtype)
    if problems or not ok:
        _record(level, name, False, "; ".join(problems)[:600] or "per-request mismatch")
        return False
    _record(level, name, True)
    print(f"       [{tag}] H={heads}: slabs {slabs} agree (dk/dv bitwise, dq_acc under "
          f"8.2), tail plan {expect_tail}, {heads} heads non-zero per request", flush=True)
    return True


def _case_tripwire(level, name):
    """G4: the full 8x4k run must take >= 120 ms -- a truncated one takes ~20 ms."""
    ms, plan = _G4_ELAPSED[0], _G4_PLAN[0]
    if plan is None:
        _record(level, name, False, "G1 did not run: no latency to check")
        return False
    tflops = _MANIFEST_8X4K_TFLOPS / (ms / 1e3)
    problems = []
    if plan.max_grid > NPU_GRID_MAX:
        problems.append(f"max_grid {plan.max_grid} > {NPU_GRID_MAX}")
    if ms < _GRID_MAX_MS:
        problems.append(f"latency {ms:.2f} ms < {_GRID_MAX_MS:.0f} ms: the manifest "
                        f"workload cannot run that fast (silent truncation)")
    if tflops > _TFLOPS_MAX:
        problems.append(f"achieved {tflops:.0f} TFLOPS > {_TFLOPS_MAX:.0f}: impossible "
                        "for this kernel (the 20.7 ms / 691 TFLOPS report was this bug)")
    if problems:
        _record(level, name, False, "; ".join(problems))
        return False
    _record(level, name, True)
    slow = " (slower than the 400 ms band -- environment, not correctness)" if ms > 400 else ""
    print(f"       latency {ms:.2f} ms, achieved {tflops:.1f} TFLOPS "
          f"(<= {_TFLOPS_MAX:.0f}), max_grid {plan.max_grid} <= {NPU_GRID_MAX}{slow}",
          flush=True)
    if ms > 400:
        print(f"       [WARN] {name}: {ms:.2f} ms is above the 153-165 ms projection "
              "band (recorded, not a failure)", flush=True)
    return True


_G1_ROWS = []  # per-request max_abs_err table of G1 (printed by the level summary)
_G4_ELAPSED = [0.0]
_G4_PLAN = [None]


# ---------------------------------------------------------------------------
# 5. Layered cases
# ---------------------------------------------------------------------------
def run_L0():
    """Gate: manifest workloads, both dtypes, both shifts, -inf rows, packed tails."""
    # 0a-0d: block-shape / packed-geometry separators (REVIEW 6-1 + Stage 3 item 2).
    # 0a is exactly two key blocks (192 = 2*96) with aligned queries, so a
    # tile/transpose/half-region error cannot hide behind a mask arm; 0b is one
    # key block with a single query block (dq spill + R7 tail at once); 0c is the
    # query tail (100 % 64 = 36) + key tail (100 % 96 = 4) + cross-request dq
    # spill in one geometry; 0d walks the R7 slow path at len_kv in
    # {1,47,48,49,95} -- every half-tile boundary.
    run_case("L0", "L0-0a exact-2-key-blocks 192", [192], [192], 2, "bfloat16", seed=100,
             per_request=True)
    run_case("L0", "L0-0b one-key-block 96 fp16", [96], [96], 2, "float16", seed=101,
             per_request=True)
    run_case("L0", "L0-0c qtail+ktail+spill 100/205", [100, 205], [100, 205], 2,
             "bfloat16", seed=102, per_request=True)
    run_case("L0", "L0-0d slow-path len_kv 1/47/48/49/95", [1, 47, 48, 49, 95],
             [1, 47, 48, 49, 95], 2, "bfloat16", seed=103, per_request=True)
    # 1-2: manifest ``mla-varlen-bwd-smoke`` (H=8 as declared), one tail request.
    run_case("L0", "L0-1 manifest-smoke bf16", [256, 384], [256, 384], 8, "bfloat16",
             seed=1, per_request=True)
    run_case("L0", "L0-2 manifest-smoke fp16", [256, 384], [256, 384], 8, "float16",
             seed=2, per_request=True)
    # 3: non-causal -> loop_st == 0 and the A2 fast arm everywhere.
    run_case("L0", "L0-3 non-causal bf16", [512, 96], [512, 96], 4, "bfloat16",
             seed=3, is_causal=False, per_request=True)
    # 4-5: bottom-right shift, both signs (shift<0 makes head query rows see no key).
    run_case("L0", "L0-4 shift>0 96x192", [96], [192], 2, "bfloat16", seed=4)
    run_case("L0", "L0-5 shift<0 128x64", [128], [64], 2, "bfloat16", seed=5,
             per_request=True)
    # 6: tiny / unaligned requests in one packed batch (len 1, len 65, len 200).
    run_case("L0", "L0-6 tiny-unaligned", [65, 1, 200], [65, 1, 200], 2, "bfloat16",
             seed=6, per_request=True)
    # 7-8: single block / minimum size (R6 half-region dual_copy + sid, R7 slow path).
    run_case("L0", "L0-7 single-block", [64], [64], 1, "float16", seed=7)
    run_case("L0", "L0-8 minimal 1x1", [1], [1], 1, "float16", seed=8)
    # 9: request pollution (B>1, different lengths, request 0 has a 44-row tail).
    run_case("L0", "L0-9 request-pollution", [300, 512], [300, 512], 2, "bfloat16",
             seed=9, per_request=True)
    run_case("L0", "L0-10 unequal-tails", [300], [700], 2, "bfloat16", seed=10)
    # 11: store-atomic contract, 12: A2 arm equivalence.
    _case_atomic_contract("L0", "L0-11 store-atomic")
    # 11b (v2): the *multi-launch* form of the same contract -- H=64 with the
    # default slab 32 gives 2 launches (tiles=17, grid=544, idle=0), which is the
    # only case that can tell "cleared once" from "cleared per launch": dq_acc is
    # prefilled once, so a per-launch clear would lose the first slab's dq.
    _case_atomic_contract("L0", "L0-11b store-atomic-multi-launch", q_lens=(1024,),
                          kv_lens=(1024,), heads=64,
                          expect={"tiles": 11, "slab": 32, "launches": 2,
                                  "max_grid": 352, "idle": 0})
    _case_arm_equivalence("L0", "L0-12 a2-arm-equivalence")
    # 24: the tail-slab guard itself at a small scale (H=40, slab 32 -> the second
    # launch has 8 active and 24 idle blocks; the idle blocks must return without
    # touching any gradient).
    run_case("L0", "L0-24 tail-slab-guard H=40", [128], [128], 40, "bfloat16", seed=40,
             per_request=True, expect={"tiles": 2, "slab": 32, "launches": 2,
                                       "max_grid": 64, "idle": 48})
    # 13: degenerate request len_q == 0 (R7) and an extreme short/long pair.
    run_case("L0", "L0-13 len_q=0 request", [0, 64], [64, 64], 2, "bfloat16", seed=13,
             per_request=True)
    run_case("L0", "L0-14 len_q-1-vs-1024", [1, 1024], [1024, 1], 2, "bfloat16", seed=14,
             per_request=True)
    _case_golden_naive("L0", "L0-15 golden-vs-naive-loop")
    _case_contract("L0", "L0-16 refuses-an-invalid-contract")
    # 17-19: the three remaining manifest workloads: declared B / lengths / D and
    # dtype, with H reduced (the memory-lean CPU reference must fit the 32 GiB
    # cgroup; the kernel work per block is independent of H, which only scales the
    # block count).  Tails, request mixture and the packed decode are unchanged.
    run_case("L0", "L0-17 manifest-ds-v3-8x4k H=32/128", [4096] * 8, [4096] * 8, 32,
             "bfloat16", seed=17, per_request=True,
             expect={"tiles": 349, "slab": 32, "launches": 1, "max_grid": 11_168})
    run_case("L0", "L0-18 manifest-kimi-k2-4x8k H=16/64", [8192] * 4, [8192] * 4, 16,
             "bfloat16", seed=18, per_request=True,
             expect={"tiles": 345, "slab": 16, "launches": 1, "max_grid": 5_520})
    run_case("L0", "L0-19 manifest-ds-v3-mixed H=16/128",
             [512, 1024, 2048, 4096, 8192, 16384], [512, 1024, 2048, 4096, 8192, 16384],
             16, "bfloat16", seed=19, per_request=True,
             expect={"tiles": 342, "slab": 16, "launches": 1, "max_grid": 5_472})
    # 20: the launch plan itself (host only, no device work): overflow raises, the
    # legal bound passes, the inlined copy == launch_plan.py, post-assert rejects
    # mismatches (DESIGN.md 8.4 G5 (1)-(6)).
    _case_launch_plan("L0", "L0-20 launch-plan-assertions")
    _finish("L0", strict=True)


def run_grid_cap():
    """Grid-cap regressions at the real manifest H (DESIGN.md 8.4 G1-G4, L0-21-23).

    These need the full 32k-token packed tensors, so they live in their own level:
    ``--level L0`` keeps the CPU-golden gate, ``--level G`` runs the decisive
    boundary regressions against the per-request reference.
    """
    # G1 (decisive): ds-v3-8x4k at H=128 -- grid 66,560 without the fix.
    _case_grid_cap_split("G", "G1 ds-v3-8x4k-split H=128",
                         [4096] * 8, [4096] * 8, 128, "bfloat16", 8, "G1",
                         {"tiles": 349, "slab": 32, "launches": 4,
                          "max_grid": 11_168, "idle": 0})
    # G2: ds-v3-mixed at H=128 -- 65,280 in v0, i.e. 255 blocks of margin.
    _case_grid_cap_split("G", "G2 ds-v3-mixed-split H=128",
                         [512, 1024, 2048, 4096, 8192, 16384],
                         [512, 1024, 2048, 4096, 8192, 16384], 128, "bfloat16", 6, "G2",
                         {"tiles": 342, "slab": 32, "launches": 4,
                          "max_grid": 10_944, "idle": 0})
    # G3 / L0-21 / L0-22: the H=126 (healthy) vs H=127 (truncated) boundary, plus the
    # head-slab mapping invariance (slab 32 vs 63 vs 126 on the same H=126 data).
    _case_head_slab_mapping("G", "G3a H=126 slab-mapping", [4096] * 8, [4096] * 8, 126,
                            "bfloat16", 26, (32, 63, 126),
                            {32: (4, 698), 63: (2, 0), 126: (1, 0)}, "G3a")
    _case_head_slab_mapping("G", "G3b H=127 tail-slab", [4096] * 8, [4096] * 8, 127,
                            "bfloat16", 27, (32,),
                            {32: (4, 349)}, "G3b")
    # G4: the throughput tripwire of the same full run (truncation only makes it
    # faster: the bug showed 20.7 ms / 691 TFLOPS, the truth is ~488 ms / ~29).
    _case_tripwire("G", "G4 throughput-tripwire")
    if _G1_ROWS:
        print("[G] G1 per-request max_abs_err table", flush=True)
        for row in _G1_ROWS:
            print(f"       request {row['request']} rows {row['rows']}: "
                  f"dk {row['dk']:.3e} dv {row['dv']:.3e} dq_acc {row['dq_acc']:.3e} "
                  f"dq {row['dq']:.3e} k_ulp {row['dq_acc_k_ulp']:.2f}", flush=True)
    _finish("G", strict=True)


def run_L1():
    """Representative shapes, both dtypes, both causal settings, multi-request."""
    run_case("L1", "L1-1 bf16 two-request-1024-1536", [1024, 1536], [1024, 1536], 2,
             "bfloat16", seed=41, repeat=2, per_request=True)
    run_case("L1", "L1-2 fp16 single-1024", [1024], [1024], 2, "float16", seed=42,
             repeat=2)
    run_case("L1", "L1-3 bf16 three-requests-grow", [512, 768, 1024], [512, 768, 1024], 2,
             "bfloat16", seed=43, repeat=2, per_request=True)
    run_case("L1", "L1-4 bf16 bottom-right-512x768", [512], [768], 2, "bfloat16", seed=44,
             repeat=2)
    run_case("L1", "L1-5 fp16 non-causal-1024", [1024], [1024], 2, "float16", seed=45,
             is_causal=False, repeat=2)
    run_case("L1", "L1-6 bf16 single-head-768", [768], [768], 1, "bfloat16", seed=46,
             repeat=2)
    # L1-7: the manifest smoke geometry (B=2, lens 256/384, H=8) -> tiles=12, slab=8,
    # 96 blocks in **one** launch: the degenerate single-launch path of the slabbed
    # kernel must stay bit-identical to the pre-slab form.
    run_case("L1", "L1-7 manifest-smoke-single-launch", [256, 384], [256, 384], 8,
             "bfloat16", seed=47, repeat=2, per_request=True,
             expect={"tiles": 8, "slab": 8, "launches": 1, "max_grid": 64})
    # L1-8: a two-launch geometry (H=64, slab 32) at a representative length.
    run_case("L1", "L1-8 bf16 multi-launch-2048-H64", [2048], [2048], 64, "bfloat16",
             seed=48, repeat=2, expect={"tiles": 22, "slab": 32, "launches": 2,
                                        "max_grid": 704})
    _finish("L1", strict=True)


def run_L2():
    """Larger informative geometries (golden is O(H*len_q*len_kv) on CPU -> capped)."""
    run_case("L2", "L2-1 bf16 2048", [2048], [2048], 2, "bfloat16", seed=61)
    run_case("L2", "L2-2 bf16 packed-1024-2048", [1024, 2048], [1024, 2048], 2,
             "bfloat16", seed=62, per_request=True)
    run_case("L2", "L2-3 fp16 1536", [1536], [1536], 2, "float16", seed=63)
    run_case("L2", "L2-4 bf16 non-causal-1024x2048", [1024], [2048], 2, "bfloat16",
             seed=64, is_causal=False)
    _finish("L2", strict=False)


def run_boundary():
    """Sub-tile sizes, tails, both shifts, zero gradients, degenerate requests."""
    run_case("boundary", "B-1 len=1", [1], [1], 1, "bfloat16", seed=81)
    run_case("boundary", "B-2 len=63 tail-63", [63], [63], 1, "bfloat16", seed=82)
    run_case("boundary", "B-3 len=65 tail-1", [65], [65], 1, "float16", seed=83)
    run_case("boundary", "B-4 len=129 tail-1", [129], [129], 1, "bfloat16", seed=84)
    run_case("boundary", "B-5 len=200 tail-8", [200], [200], 1, "bfloat16", seed=85)
    run_case("boundary", "B-6 shift+ 65x129", [65], [129], 1, "bfloat16", seed=86)
    run_case("boundary", "B-7 shift- 192x64", [192], [64], 1, "bfloat16", seed=87)
    run_case("boundary", "B-8 zero-dout", [96], [96], 1, "float16", seed=88,
             special="zero_dout")
    run_case("boundary", "B-9 tiny-scores", [128], [128], 1, "bfloat16", seed=89,
             special="tiny_q")
    run_case("boundary", "B-10 non-causal-shift+", [96], [192], 1, "bfloat16", seed=90,
             is_causal=False)
    run_case("boundary", "B-11 len_q=0 in the middle", [128, 0, 64], [128, 64, 64], 2,
             "bfloat16", seed=91, per_request=True)
    run_case("boundary", "B-12 B=3 H=1 fp16", [128, 64, 96], [128, 64, 96], 1, "float16",
             seed=92, per_request=True)
    _finish("boundary", strict=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0",
                        choices=["L0", "L1", "L2", "boundary", "G", "all"])
    args, _ = parser.parse_known_args()
    levels = {"L0": run_L0, "L1": run_L1, "L2": run_L2, "boundary": run_boundary,
              "G": run_grid_cap}
    chosen = list(levels.values()) if args.level == "all" else [levels[args.level]]
    started = time.perf_counter()
    for run in chosen:
        run()
    print(f"[TIME] {time.perf_counter() - started:.1f} s for level={args.level}", flush=True)
    print(f"[PRECISION_PASS] max_diff={_MAX_DIFF[0]:.6g}", flush=True)


if __name__ == "__main__":
    main()
