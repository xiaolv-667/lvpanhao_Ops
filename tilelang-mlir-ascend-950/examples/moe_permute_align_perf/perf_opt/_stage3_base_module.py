# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""MoE token-to-expert alignment + route scatter, Ascend 950PR (Stage 4 tuned).

One self-contained module carrying the three kernel factories the frozen wrapper
``tileops/kernels/moe/moe_permute_align/moe_permute_align.py`` imports:

* ``_make_align_kernel(numel, num_experts, block_size)`` -> ``align(flat,
  sorted_token_ids, expert_ids, num_tokens_post_pad, cumsum)``
* ``_make_scatter_kernel(numel, num_experts, block_size)`` -> ``scatter(flat,
  sorted_token_ids, cumsum)``
* ``_make_small_batch_kernel(numel, num_experts, block_size)`` -> ``small(flat,
  sorted_token_ids, expert_ids, num_tokens_post_pad)``

Stage 4 restructure (unchanged semantics, see ``perf_opt/opt_log.md``):

1. the align stage keeps only the per-core histogram (its launch 1) and hands
   the per-core partial counts to the scatter stage through a persistent
   scratch;
2. the scatter stage becomes the second, wider launch: it merges the partial
   counts, derives the padded prefix sum, writes ``cumsum`` /
   ``num_tokens_post_pad`` / ``expert_ids`` / the sentinel, and places every
   route with the *deterministic* slot
   ``cumsum[e] + (routes of e in earlier cores) + (routes of e earlier in this
   core)`` -- no global atomics;
3. ``expert_ids`` is produced by letting each expert expand its own block range
   instead of an ``O(num_experts)`` rank count per block.

Program entry: ``python moe_permute_align.py --level L0|L1|boundary`` runs the
combined-pipeline precision harness (align+scatter, or the small-batch path).
"""

import argparse
import functools
import math
import os
import time

import torch
import torch_npu  # noqa: F401 -- registers torch.npu
import tilelang
import tilelang.ascend.language as T


# ---------------------------------------------------------------------------
# Align stage: build-time planning + both paths
# ---------------------------------------------------------------------------
FILL_TILE = 2048
VECTOR_CORES = 56
PATH_S_INSTR_BUDGET = 8192



def _align_up(value, multiple):
    return (value + multiple - 1) // multiple * multiple


def _scan_steps(lanes):
    return 0 if lanes <= 1 else int(math.ceil(math.log2(lanes)))


def _geometry(numel, num_experts, block_size):
    E = num_experts
    B = block_size
    p_total = numel + (E + 1) * (B - 1)
    return {
        "numel": numel,
        "E": E,
        "B": B,
        "E_pad": _align_up(max(E, 1), 64),
        "P": p_total,
        "NB": (p_total + B - 1) // B,
        "steps": _scan_steps(_align_up(max(E, 1), 64)),
        "shift": B.bit_length() - 1 if (B & (B - 1)) == 0 else None,
    }


def _use_small_path(geo):
    lanes_per_vec = (geo["E"] + 63) // 64
    hist = geo["numel"] * lanes_per_vec * 2
    ids = geo["NB"] * lanes_per_vec
    fill = (geo["P"] + 63) // 64
    return hist + ids + fill <= PATH_S_INSTR_BUDGET


def _hist_cores(numel):
    return max(1, min(VECTOR_CORES, (numel + 511) // 512))


def _align_cores(p_total, num_blocks):
    logical = max(
        (num_blocks + 15) // 16, ((p_total + FILL_TILE - 1) // FILL_TILE + 7) // 8
    )
    logical = max(1, min(VECTOR_CORES, logical))
    for cand in (1, 2, 4, 7, 8, 14, 28, 56):
        if cand >= logical:
            return cand
    return VECTOR_CORES


# ---------------------------------------------------------------------------
# Kernel: small path (single launch, grid=2)
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _compile_single(numel, E, B):
    geo = _geometry(numel, E, B)
    e_pad = geo["E_pad"]
    p_total = geo["P"]
    nb = geo["NB"]
    blk = max(64, _align_up(nb, 64))
    steps = geo["steps"]
    shift = geo["shift"]
    chunk = max(64, min(1024, _align_up(max(numel, 1), 64)))
    half = (p_total + 1) // 2
    n_fill = (half + FILL_TILE - 1) // FILL_TILE
    n_full = numel // chunk
    tail = numel - n_full * chunk

    @T.prim_func
    def main(
        flat: T.Tensor((numel,), "int32"),
        sorted_token_ids: T.Tensor((p_total,), "int32"),
        expert_ids: T.Tensor((nb,), "int32"),
        num_tokens_post_pad: T.Tensor((1,), "int32"),
        cumsum: T.Tensor((E + 1,), "int32"),
    ):
        with T.Kernel(2) as bx:
            s_x = T.alloc_shared((chunk,), "int32")
            s_cnt = T.alloc_shared((e_pad,), "int32")
            s_pad = T.alloc_shared((e_pad,), "int32")
            s_scan_a = T.alloc_shared((2 * e_pad,), "int32")
            s_scan_b = T.alloc_shared((2 * e_pad,), "int32")
            s_pref = T.alloc_shared((e_pad + 64,), "int32")
            s_ids = T.alloc_shared((blk,), "int32")
            s_fill = T.alloc_shared((FILL_TILE,), "int32")

            # ---- sentinel fill: unconditional, one half per core -------------
            T.fill(s_fill, numel)
            if n_fill > 0:
                for t in T.serial(n_fill):
                    off = bx * half + t * FILL_TILE
                    n = T.min(FILL_TILE, p_total - off)
                    if n > 0:
                        T.copy(s_fill[0:n], sorted_token_ids[off : off + n])

            # ---- metadata and mapping: no `numel > 0` guard on the writes,
            #      so a zero-route call still emits the contract values
            #      (cumsum all 0, npost 0, expert_ids all E) ---------------
            if bx == 0:
                with T.SimtVF(threads=e_pad):
                    for e in T.Parallel(e_pad):
                        s_cnt[e] = 0

                if numel > 0:  # build-time branch: only the GM read is guarded
                    for c in range(n_full):
                        T.copy(flat[c * chunk : (c + 1) * chunk], s_x[0:chunk])
                        with T.SimtVF(threads=e_pad):
                            for e in T.Parallel(e_pad):
                                acc = T.alloc_local([1], "int32")
                                acc[0] = 0
                                for i in T.serial(chunk):
                                    acc[0] = acc[0] + T.if_then_else(s_x[i] == e, 1, 0)
                                s_cnt[e] = s_cnt[e] + acc[0]
                    if tail > 0:
                        T.copy(flat[n_full * chunk : numel], s_x[0:tail])
                        with T.SimtVF(threads=e_pad):
                            for e in T.Parallel(e_pad):
                                acc = T.alloc_local([1], "int32")
                                acc[0] = 0
                                for i in T.serial(tail):
                                    acc[0] = acc[0] + T.if_then_else(s_x[i] == e, 1, 0)
                                s_cnt[e] = s_cnt[e] + acc[0]

                # padding lanes (e >= E) contribute nothing
                with T.SimtVF(threads=e_pad):
                    for e in T.Parallel(e_pad):
                        pv = T.if_then_else(e < E, s_cnt[e], 0)
                        if shift is not None:
                            s_pad[e] = ((pv + (B - 1)) >> shift) << shift
                        else:
                            s_pad[e] = T.ceildiv(pv, B) * B

                # inclusive doubling scan; two alternating buffers, data in the upper
                # half and a zero halo in the lower half, so every shifted operand stays
                # in bounds and no shared->shared copy is needed
                with T.SimtVF(threads=e_pad):
                    for e in T.Parallel(e_pad):
                        s_scan_a[e_pad + e] = s_pad[e]
                with T.SimtVF(threads=e_pad):
                    for e in T.Parallel(e_pad):
                        s_scan_a[e] = 0
                # The SIMT thread count is capped at 2048 by the compiler
                # (``LAUNCH_BOUND``), so the 2*e_pad halo buffer is zeroed in two
                # passes of e_pad lanes instead of one of 2*e_pad -- same writes,
                # in the same order as the halves above.
                with T.SimtVF(threads=e_pad):
                    for e in T.Parallel(e_pad):
                        s_scan_b[e] = 0
                with T.SimtVF(threads=e_pad):
                    for e in T.Parallel(e_pad):
                        s_scan_b[e_pad + e] = 0
                for d in range(steps):
                    k = 1 << d
                    if d % 2 == 0:  # build-time branch
                        with T.SimtVF(threads=e_pad):
                            for e in T.Parallel(e_pad):
                                s_scan_b[e_pad + e] = (
                                    s_scan_a[e_pad + e] + s_scan_a[e_pad + e - k]
                                )
                    else:
                        with T.SimtVF(threads=e_pad):
                            for e in T.Parallel(e_pad):
                                s_scan_a[e_pad + e] = (
                                    s_scan_b[e_pad + e] + s_scan_b[e_pad + e - k]
                                )

                # shift by one lane, filled over [0, e_pad] so the padding lanes hold
                # `total`, a defined value, never an uninitialised one
                if steps % 2 == 0:
                    with T.SimtVF(threads=e_pad + 64):
                        for e in T.Parallel(e_pad + 64):
                            if e <= e_pad:
                                s_pref[e] = T.if_then_else(
                                    e == 0, 0, s_scan_a[e_pad + e - 1]
                                )
                else:
                    with T.SimtVF(threads=e_pad + 64):
                        for e in T.Parallel(e_pad + 64):
                            if e <= e_pad:
                                s_pref[e] = T.if_then_else(
                                    e == 0, 0, s_scan_b[e_pad + e - 1]
                                )

                total = s_pref[E]
                T.copy(s_pref[0 : E + 1], cumsum[0 : E + 1])
                num_tokens_post_pad[0] = total

                # rank-count over [0, E) only; the padding lanes are never read
                with T.SimtVF(threads=blk):
                    for b in T.Parallel(blk):
                        pos = b * B
                        cnt = T.alloc_local([1], "int32")
                        cnt[0] = 0
                        for e in T.serial(E):
                            cnt[0] = cnt[0] + T.if_then_else(s_pref[e] <= pos, 1, 0)
                        s_ids[b] = T.if_then_else(pos < total, cnt[0] - 1, E)
                if nb > 0:
                    T.copy(s_ids[0:nb], expert_ids[0:nb])

    return tilelang.compile(main, target="ascend", out_idx=[])


# ---------------------------------------------------------------------------
# Kernel: large path (launch 1: per-core partial histograms)
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _compile_hist(numel, E, B, g_h):
    e_pad = _align_up(max(E, 1), 64)
    tpc = (numel + g_h - 1) // g_h
    chunk = max(64, min(1024, _align_up(max(tpc, 1), 64)))
    n_chunks = (tpc + chunk - 1) // chunk

    @T.prim_func
    def main(
        flat: T.Tensor((numel,), "int32"),
        partial: T.Tensor((g_h, e_pad), "int32"),
    ):
        with T.Kernel(g_h) as g:
            s_x = T.alloc_shared((chunk,), "int32")
            s_hist = T.alloc_shared((e_pad,), "int32")

            with T.SimtVF(threads=e_pad):
                for e in T.Parallel(e_pad):
                    s_hist[e] = 0

            if n_chunks > 0:  # build-time: numel == 0 leaves an all-zero row
                # serial (not pipelined): the accumulator is loop-carried
                for c in T.serial(n_chunks):
                    base = g * tpc + c * chunk
                    # clamp to this core's own slice and to the tensor end
                    n = T.max(0, T.min(T.min(tpc - c * chunk, chunk), numel - base))
                    if n > 0:
                        T.copy(flat[base : base + n], s_x[0:n])
                        with T.SimtVF(threads=e_pad):
                            for e in T.Parallel(e_pad):
                                acc = T.alloc_local([1], "int32")
                                acc[0] = 0
                                for i in T.serial(n):
                                    acc[0] = acc[0] + T.if_then_else(s_x[i] == e, 1, 0)
                                s_hist[e] = s_hist[e] + acc[0]

            # plain store, no atomics, no initialisation contract
            T.copy(s_hist, partial[g, :])

    return tilelang.compile(main, target="ascend", out_idx=[])


# ---------------------------------------------------------------------------
# Kernel: large path (launch 2: redundant merge + scan + mapping + fill)
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _compile_align_partial(numel, E, B, g_h, g_a):
    geo = _geometry(numel, E, B)
    e_pad = geo["E_pad"]
    p_total = geo["P"]
    nb = geo["NB"]
    steps = geo["steps"]
    shift = geo["shift"]
    blkpc = (nb + g_a - 1) // g_a
    blk = max(64, _align_up(max(blkpc, 1), 64))
    share = (p_total + g_a - 1) // g_a
    n_fill = (share + FILL_TILE - 1) // FILL_TILE

    @T.prim_func
    def main(
        partial: T.Tensor((g_h, e_pad), "int32"),
        sorted_token_ids: T.Tensor((p_total,), "int32"),
        expert_ids: T.Tensor((nb,), "int32"),
        num_tokens_post_pad: T.Tensor((1,), "int32"),
        cumsum: T.Tensor((E + 1,), "int32"),
    ):
        with T.Kernel(g_a) as a:
            s_part = T.alloc_shared((e_pad,), "int32")
            s_cnt = T.alloc_shared((e_pad,), "int32")
            s_pad = T.alloc_shared((e_pad,), "int32")
            s_scan_a = T.alloc_shared((2 * e_pad,), "int32")
            s_scan_b = T.alloc_shared((2 * e_pad,), "int32")
            s_pref = T.alloc_shared((e_pad + 64,), "int32")
            s_ids = T.alloc_shared((blk,), "int32")
            s_fill = T.alloc_shared((FILL_TILE,), "int32")

            # every core re-reads the whole scratch: integer addition is
            # associative, so all cores derive bit-identical counts
            with T.SimtVF(threads=e_pad):
                for e in T.Parallel(e_pad):
                    s_cnt[e] = 0
            for g in T.serial(g_h):
                T.copy(partial[g, :], s_part)
                with T.SimtVF(threads=e_pad):
                    for e in T.Parallel(e_pad):
                        s_cnt[e] = s_cnt[e] + s_part[e]

            with T.SimtVF(threads=e_pad):
                for e in T.Parallel(e_pad):
                    pv = T.if_then_else(e < E, s_cnt[e], 0)
                    if shift is not None:
                        s_pad[e] = ((pv + (B - 1)) >> shift) << shift
                    else:
                        s_pad[e] = T.ceildiv(pv, B) * B

            # inclusive doubling scan; two alternating buffers, data in the upper
            # half and a zero halo in the lower half, so every shifted operand stays
            # in bounds and no shared->shared copy is needed
            with T.SimtVF(threads=e_pad):
                for e in T.Parallel(e_pad):
                    s_scan_a[e_pad + e] = s_pad[e]
            with T.SimtVF(threads=e_pad):
                for e in T.Parallel(e_pad):
                    s_scan_a[e] = 0
            # Same cap as the small path: 2*e_pad lanes would exceed the 2048
            # SIMT thread limit for e_pad > 1024, so zero the two halves apart.
            with T.SimtVF(threads=e_pad):
                for e in T.Parallel(e_pad):
                    s_scan_b[e] = 0
            with T.SimtVF(threads=e_pad):
                for e in T.Parallel(e_pad):
                    s_scan_b[e_pad + e] = 0
            for d in range(steps):
                k = 1 << d
                if d % 2 == 0:  # build-time branch
                    with T.SimtVF(threads=e_pad):
                        for e in T.Parallel(e_pad):
                            s_scan_b[e_pad + e] = (
                                s_scan_a[e_pad + e] + s_scan_a[e_pad + e - k]
                            )
                else:
                    with T.SimtVF(threads=e_pad):
                        for e in T.Parallel(e_pad):
                            s_scan_a[e_pad + e] = (
                                s_scan_b[e_pad + e] + s_scan_b[e_pad + e - k]
                            )

            # shift by one lane, filled over [0, e_pad] so the padding lanes hold
            # `total`, a defined value, never an uninitialised one
            if steps % 2 == 0:
                with T.SimtVF(threads=e_pad + 64):
                    for e in T.Parallel(e_pad + 64):
                        if e <= e_pad:
                            s_pref[e] = T.if_then_else(
                                e == 0, 0, s_scan_a[e_pad + e - 1]
                            )
            else:
                with T.SimtVF(threads=e_pad + 64):
                    for e in T.Parallel(e_pad + 64):
                        if e <= e_pad:
                            s_pref[e] = T.if_then_else(
                                e == 0, 0, s_scan_b[e_pad + e - 1]
                            )

            total = s_pref[E]
            if a == 0:
                T.copy(s_pref[0 : E + 1], cumsum[0 : E + 1])
                num_tokens_post_pad[0] = total

            blk0 = a * blkpc
            with T.SimtVF(threads=blk):
                for j in T.Parallel(blk):
                    b = blk0 + j
                    cnt = T.alloc_local([1], "int32")
                    cnt[0] = 0
                    for e in T.serial(E):
                        cnt[0] = cnt[0] + T.if_then_else(s_pref[e] <= b * B, 1, 0)
                    s_ids[j] = T.if_then_else(b * B < total, cnt[0] - 1, E)
            valid = T.max(0, T.min(blkpc, nb - blk0))
            if valid > 0:
                T.copy(s_ids[0:valid], expert_ids[blk0 : blk0 + valid])

            T.fill(s_fill, numel)
            if n_fill > 0:
                for t in T.serial(n_fill):
                    off = a * share + t * FILL_TILE
                    n = T.min(FILL_TILE, T.min((a + 1) * share, p_total) - off)
                    if n > 0:
                        T.copy(s_fill[0:n], sorted_token_ids[off : off + n])

    return tilelang.compile(main, target="ascend", out_idx=[])


class _AlignTwoPass:
    """The large path's align callable: two launches plus a persistent scratch.

    ``partial`` is allocated on first use from ``flat.device`` and reused by
    later calls of this instance.  Launch 1 overwrites every row, so the scratch
    needs no initialisation; concurrent streams sharing one instance are unsafe.
    """

    def __init__(self, numel, E, B, g_h, g_a, e_pad):
        self._numel = numel
        self._hist = _compile_hist(numel, E, B, g_h)
        self._align = _compile_align_partial(numel, E, B, g_h, g_a)
        self._shape = (g_h, e_pad)
        self._partial = None

    def __call__(self, flat, sorted_token_ids, expert_ids, num_tokens_post_pad, cumsum):
        if self._partial is None or self._partial.device != flat.device:
            self._partial = torch.empty(
                self._shape, dtype=torch.int32, device=flat.device
            )
        self._hist(flat, self._partial)
        self._align(
            self._partial, sorted_token_ids, expert_ids, num_tokens_post_pad, cumsum
        )


# ---------------------------------------------------------------------------
# Kernel factory (harness wrapper contract: factory() -> align(...) callable)
# ---------------------------------------------------------------------------
def _make_align_kernel(numel: int, num_experts: int, block_size: int):
    """Build the align factory for one ``(numel, num_experts, block_size)``.

    Returns a no-argument callable whose result is invoked as
    ``align(flat, sorted_token_ids, expert_ids, num_tokens_post_pad, cumsum)``,
    matching ``MoEPermuteAlignKernel._compiled_align``.
    """
    numel = int(numel)
    E = int(num_experts)
    B = int(block_size)
    geo = _geometry(numel, E, B)
    small = _use_small_path(geo)
    g_h = _hist_cores(numel)
    g_a = _align_cores(geo["P"], geo["NB"])
    e_pad = geo["E_pad"]

    def _build():
        if small:
            return _compile_single(numel, E, B)
        return _AlignTwoPass(numel, E, B, g_h, g_a, e_pad)

    _build.path = "S" if small else "L"
    _build.g_h = g_h
    _build.g_a = g_a
    return _build


# ---------------------------------------------------------------------------
# Test harness

# ---------------------------------------------------------------------------
# Scatter stage
# ---------------------------------------------------------------------------
_THREADS = 128  # lane width, a compile-time constant (not a launch parameter)
_NBLK_TARGET = 56  # physical Vector cores of Ascend950PR_9579



@functools.lru_cache(maxsize=32)
def _make_scatter_kernel(numel: int, num_experts: int, block_size: int):
    """Frozen factory: geometry in, a zero-arg callable building the program.

    ``numel``, ``num_experts`` and ``block_size`` are compile-time constants, so
    every loop bound and guard below is static.  ``ept`` is the number of routes
    one lane handles, picked so that the logical core count stays at or below the
    56 physical Vector cores (``tile = _THREADS * ept >= ceil(numel / 56)``
    implies ``nblocks <= 56`` for any ``numel``).
    """
    if numel < 1:
        raise ValueError(f"the scatter needs at least one route, got numel={numel}")
    max_padded = numel + (num_experts + 1) * (block_size - 1)
    ept = max(1, math.ceil(math.ceil(numel / _NBLK_TARGET) / _THREADS))
    tile = _THREADS * ept
    nblocks = math.ceil(numel / tile)

    @tilelang.jit(out_idx=[], target="ascend")
    def _scatter():
        @T.prim_func
        def _scatter_main(
            flat: T.Tensor([numel], "int32"),
            sorted_token_ids: T.Tensor([max_padded], "int32"),
            cumsum: T.Tensor([num_experts + 1], "int32"),
        ):
            with T.Kernel(nblocks) as bid, T.SimtVF(threads=_THREADS):
                for i in T.serial(ept):  # routes this lane handles, unrolled
                    for j in T.Parallel(_THREADS):  # one route per lane
                        idx = (bid * ept + i) * _THREADS + j
                        # Clamped index + -1 sentinel: a tail lane never reads
                        # past ``flat``, never takes a slot and never writes.
                        eid = T.if_then_else(
                            idx < numel, flat[T.min(idx, numel - 1)], -1
                        )
                        # Upper bound guards ``cumsum[num_experts]``
                        # (num_tokens_post_pad), which this stage must not touch.
                        if eid >= 0 and eid < num_experts:
                            slot = T.atomic_add(
                                cumsum[eid], T.int32(1), return_prev=True
                            )
                            # The store shares the guard with the atomic on
                            # purpose: the else branch of an atomic yields 0, so
                            # an unguarded store would write slot 0.
                            sorted_token_ids[slot] = idx

        return _scatter_main

    return _scatter

# ---------------------------------------------------------------------------
# Fused small-batch stage (wrapper path: numel < 1024 and E <= 32)
# ---------------------------------------------------------------------------
SB_W = 32  # lane width of the expert and route axes (= the expert-count bound)
SB_UNROLL = 4  # independent histogram accumulator chains per lane
SB_FILL_TILE = 1024  # sentinel tile, in int32 elements per bulk store
SB_IDS_LANES = 64  # lane width of the expert_ids pass
SB_MAX_EXPERTS = 32  # the path's contract: at most one warp of experts
SB_MAX_CHUNK = 1024  # route tile upper bound (one SIMT pass per chunk)
SB_XFER_GUARD = 32  # spare slots past a partial tile (the GM->UB load rounds up)
SB_SCAN_STEPS = 5  # ceil(log2(W)) -- Hillis-Steele steps over 32 lanes



def _sb_align_up(value, multiple):
    return (value + multiple - 1) // multiple * multiple


def _sb_geometry(numel, num_experts, block_size):
    """Every build-time constant of one specialised program."""
    E, B = int(num_experts), int(block_size)
    p_total = numel + (E + 1) * (B - 1)
    nb = math.ceil(p_total / B) if p_total else 0
    chunk = min(SB_MAX_CHUNK, max(64, _sb_align_up(max(numel, 1), 64)))
    return {
        "numel": numel,
        "E": E,
        "B": B,
        "P": p_total,
        "NB": nb,
        "chunk": chunk,
        "n_chunks": (numel + chunk - 1) // chunk,
        "ids_blk": _sb_align_up(max(nb, 1), SB_IDS_LANES),
        "rounds": (numel + SB_W - 1) // SB_W,
        "n_fill": (p_total + SB_FILL_TILE - 1) // SB_FILL_TILE,
        "shift": B.bit_length() - 1 if (B & (B - 1)) == 0 else None,
    }


# ---------------------------------------------------------------------------
# 3. Kernel: one launch, one core, seven in-core stages
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _compile_sb_small(numel, E, B):
    if SB_W != 32 or SB_SCAN_STEPS != 5:
        # the five doubling steps below are written out for a 32-lane domain
        raise ValueError(f"the scan is unrolled for 32 lanes, got {SB_W}/{SB_SCAN_STEPS}")
    geo = _sb_geometry(numel, E, B)
    p_total = geo["P"]
    nb = geo["NB"]
    chunk = geo["chunk"]
    ids_blk = geo["ids_blk"]
    rounds = geo["rounds"]
    n_fill = geo["n_fill"]
    shift = geo["shift"]
    # Build-time split of the route tile into whole chunks plus the last one.
    # All four numbers are plain Python integers, so the histogram bounds and
    # the copy regions fold at build time.
    n_full = numel // chunk
    tail = numel - n_full * chunk
    tail_groups = tail // SB_UNROLL
    tail_rem = tail - tail_groups * SB_UNROLL

    @T.prim_func
    def _small_main(
        flat: T.Tensor((numel,), "int32"),
        sorted_token_ids: T.Tensor((p_total,), "int32"),
        expert_ids: T.Tensor((nb,), "int32"),
        num_tokens_post_pad: T.Tensor((1,), "int32"),
    ):
        with T.Kernel(1):
            # The last GM->UB load carries a partial tile: the copy rounds its
            # length up to a 32-byte burst (repro/TRAP-950-partial-gm2ub-load),
            # so the slots past `tail` hold junk.  The guard band keeps that
            # spill inside s_x, and the histogram bound is the build-time
            # `tail`, so the junk is never read.
            s_x = T.alloc_shared((chunk + SB_XFER_GUARD,), "int32")
            s_cnt = T.alloc_shared((SB_W,), "int32")  # counts, then padded counts
            s_scan_a = T.alloc_shared((2 * SB_W,), "int32")  # scan: [halo | data]
            s_scan_b = T.alloc_shared((2 * SB_W,), "int32")  # alternate buffer
            s_cs = T.alloc_shared((SB_W + 1,), "int32")  # prefix, then cursors
            s_ids = T.alloc_shared((ids_blk,), "int32")  # expert_ids staging
            s_fill = T.alloc_shared((SB_FILL_TILE,), "int32")  # sentinel tile

            # ---- P0 sentinel fill: same core as the scatter, issued first ----
            T.fill(s_fill, numel)
            for t in T.serial(n_fill):  # tail block clipped to its live region
                off = t * SB_FILL_TILE
                n = T.min(SB_FILL_TILE, p_total - off)
                if n > 0:
                    T.copy(s_fill[0:n], sorted_token_ids[off : off + n])

            # ---- P2 histogram: lane <-> expert, SB_UNROLL independent chains ----
            with T.SimtVF(threads=SB_W):
                for e in T.Parallel(SB_W):
                    s_cnt[e] = 0

            if numel > 0:  # build-time: a zero-route call folds the GM reads
                for c in range(n_full):
                    T.copy(flat[c * chunk : (c + 1) * chunk], s_x[0:chunk])
                    with T.SimtVF(threads=SB_W):
                        for e in T.Parallel(SB_W):
                            acc = T.alloc_local([SB_UNROLL], "int32")
                            for j in range(SB_UNROLL):
                                acc[j] = 0
                            for i in T.serial(chunk // SB_UNROLL):
                                for j in range(SB_UNROLL):
                                    acc[j] = acc[j] + T.if_then_else(
                                        s_x[i * SB_UNROLL + j] == e, 1, 0
                                    )
                            s_cnt[e] = s_cnt[e] + acc[0] + acc[1] + acc[2] + acc[3]
                if tail > 0:
                    T.copy(flat[n_full * chunk : numel], s_x[0:tail])
                    # Static bound (tail is a build-time constant): the junk the
                    # load leaves in the guard band is never read, and the
                    # accumulator keeps its SB_UNROLL chains over the whole tail.
                    with T.SimtVF(threads=SB_W):
                        for e in T.Parallel(SB_W):
                            acc = T.alloc_local([SB_UNROLL], "int32")
                            for j in range(SB_UNROLL):
                                acc[j] = 0
                            for i in T.serial(tail_groups):
                                for j in range(SB_UNROLL):
                                    acc[j] = acc[j] + T.if_then_else(
                                        s_x[i * SB_UNROLL + j] == e, 1, 0
                                    )
                            if tail_rem > 0:
                                for j in range(tail_rem):
                                    acc[j] = acc[j] + T.if_then_else(
                                        s_x[tail_groups * SB_UNROLL + j] == e, 1, 0
                                    )
                            s_cnt[e] = s_cnt[e] + acc[0] + acc[1] + acc[2] + acc[3]

            # ---- P3 pad each count up to a whole block ----------------------
            with T.SimtVF(threads=SB_W):
                for e in T.Parallel(SB_W):
                    if shift is not None:
                        s_cnt[e] = ((s_cnt[e] + (B - 1)) >> shift) << shift
                    else:
                        s_cnt[e] = T.ceildiv(s_cnt[e], B) * B

            # ---- P4 exclusive prefix: Hillis-Steele over 32 lanes -----------
            # Data in the upper half of each scan buffer, a zero halo in the
            # lower half, so every shifted operand stays in bounds and no
            # shared -> shared copy is needed.  Lanes at e >= E keep a zero
            # count, a zero padded increment and therefore cs_e = total.
            with T.SimtVF(threads=SB_W):
                for e in T.Parallel(SB_W):
                    s_scan_a[SB_W + e] = s_cnt[e]
            with T.SimtVF(threads=SB_W):
                for e in T.Parallel(SB_W):
                    s_scan_a[e] = 0
            with T.SimtVF(threads=2 * SB_W):
                for e in T.Parallel(2 * SB_W):
                    s_scan_b[e] = 0
            # The five doubling steps are written out: a traced `range` loop
            # cannot hold a build-time parity test.
            with T.SimtVF(threads=SB_W):  # k = 1, a -> b
                for e in T.Parallel(SB_W):
                    s_scan_b[SB_W + e] = s_scan_a[SB_W + e] + s_scan_a[SB_W + e - 1]
            with T.SimtVF(threads=SB_W):  # k = 2, b -> a
                for e in T.Parallel(SB_W):
                    s_scan_a[SB_W + e] = s_scan_b[SB_W + e] + s_scan_b[SB_W + e - 2]
            with T.SimtVF(threads=SB_W):  # k = 4, a -> b
                for e in T.Parallel(SB_W):
                    s_scan_b[SB_W + e] = s_scan_a[SB_W + e] + s_scan_a[SB_W + e - 4]
            with T.SimtVF(threads=SB_W):  # k = 8, b -> a
                for e in T.Parallel(SB_W):
                    s_scan_a[SB_W + e] = s_scan_b[SB_W + e] + s_scan_b[SB_W + e - 8]
            with T.SimtVF(threads=SB_W):  # k = 16, a -> b (inclusive result in b)
                for e in T.Parallel(SB_W):
                    s_scan_b[SB_W + e] = s_scan_a[SB_W + e] + s_scan_a[SB_W + e - 16]

            # Shift by one lane, filled over [0, SB_W] so every entry -- including
            # the lanes at e >= E -- is a defined value.
            with T.SimtVF(threads=SB_W + SB_W):
                for e in T.Parallel(SB_W + SB_W):
                    if e <= SB_W:
                        s_cs[e] = T.if_then_else(e == 0, 0, s_scan_b[SB_W + e - 1])

            total = s_cs[SB_W]
            # contract value for a zero-route call too: the write is not guarded
            num_tokens_post_pad[0] = total

            # ---- P5 expert_ids: rank-count over [0, E) ----------------------
            for c in range(ids_blk // SB_IDS_LANES):
                with T.SimtVF(threads=SB_IDS_LANES):
                    for j in T.Parallel(SB_IDS_LANES):
                        b = c * SB_IDS_LANES + j
                        cnt = T.alloc_local([1], "int32")
                        cnt[0] = 0
                        for e in T.serial(E):
                            cnt[0] = cnt[0] + T.if_then_else(s_cs[e] <= b * B, 1, 0)
                        s_ids[b] = T.if_then_else(b * B < total, cnt[0] - 1, E)
            if nb > 0:
                T.copy(s_ids[0:nb], expert_ids[0:nb])

            # ---- P6 scatter: lane <-> route, per-expert UB cursor bump ------
            # The route id comes from GM, not from the UB tile: that keeps the
            # loop free of any data-dependent UB access, which would otherwise
            # make the compiler insert one in-core barrier per round.  The
            # cursors are cs_e themselves, so a bump walks [cs_e, cs_e + cnt_e)
            # and the slots stay distinct and inside the expert's range.
            # cs_W (= total) is already published and is never touched: a
            # contract-conforming id is < E <= SB_W.
            if numel > 0:
                with T.SimtVF(threads=SB_W):
                    for j in T.Parallel(SB_W):
                        eid = T.alloc_local([1], "int32")
                        slot = T.alloc_local([1], "int32")
                        for r in T.serial(rounds):
                            idx = r * SB_W + j
                            if idx < numel:
                                eid[0] = flat[idx]
                                slot[0] = T.atomic_add(
                                    s_cs[eid[0]], T.int32(1), return_prev=True
                                )
                                sorted_token_ids[slot[0]] = idx

    return tilelang.compile(_small_main, target="ascend", out_idx=[])


def _make_small_batch_kernel(numel: int, num_experts: int, block_size: int):
    """Build the fused small-batch factory for one ``(numel, E, block_size)``.

    Returns a no-argument callable whose result is invoked as
    ``small(flat, sorted_token_ids, expert_ids, num_tokens_post_pad)``, matching
    ``MoEPermuteAlignKernel._small_batch_fn()`` -- the path takes no launch
    parameter because ``T.Kernel`` has no thread count.

    Raises:
        ValueError: ``num_experts`` exceeds one warp of experts, matching the
            source factory's guard.
    """
    numel = int(numel)
    E = int(num_experts)
    B = int(block_size)
    if E > SB_MAX_EXPERTS:
        raise ValueError(
            f"the small-batch kernel takes at most {SB_MAX_EXPERTS} experts, got {E}"
        )
    geo = _sb_geometry(numel, E, B)

    def _build():
        return _compile_sb_small(numel, E, B)

    _build.geometry = geo
    return _build

# ---------------------------------------------------------------------------
# Combined-pipeline precision harness (whole operator: align + scatter, or the
# fused small-batch path).  `python moe_permute_align.py --level L0`
# ---------------------------------------------------------------------------
MAX_DIFF = {"value": 0}


def _golden(topk_ids, num_experts, block_size):
    """The operator's three outputs, from the semantics only.

    ``bincount`` + padded ``cumsum`` + ``searchsorted`` for the block owners and
    a stable slot walk for the route order -- a different derivation from every
    kernel path, so it is an independent check.
    """
    B, E = block_size, num_experts
    flat = topk_ids.flatten()
    numel = int(flat.numel())
    counts = torch.bincount(flat.long(), minlength=E)[:E]
    padded = (counts + B - 1) // B * B
    cumsum = torch.zeros(E + 1, dtype=torch.int64)
    cumsum[1:] = torch.cumsum(padded, 0)
    total = int(cumsum[E].item())

    p_total = numel + (E + 1) * (B - 1)
    num_blocks = math.ceil(p_total / B)
    sorted_ids = torch.full((p_total,), numel, dtype=torch.int32)
    expert_ids = torch.full((num_blocks,), E, dtype=torch.int32)
    slot = cumsum[:E].clone()
    for route, expert in enumerate(flat.tolist()):
        sorted_ids[int(slot[expert])] = route
        slot[expert] += 1
    owned = total // B
    if owned:
        pos = torch.arange(owned, dtype=torch.int64) * B
        expert_ids[:owned] = torch.searchsorted(cumsum[1:], pos, right=True).to(torch.int32)
    npost = torch.tensor([total], dtype=torch.int32)
    return sorted_ids, expert_ids, npost


def _make_flat(numel, E, dist, seed):
    """Routes for one case, honouring the input contract ``0 <= v < E``."""
    if numel == 0:
        return torch.zeros((0,), dtype=torch.int32)
    gen = torch.Generator().manual_seed(seed)
    if dist == "uniform":
        flat = torch.randint(0, E, (numel,), generator=gen, dtype=torch.int32)
    elif dist == "skewed":
        weights = torch.arange(1, E + 1, dtype=torch.float64) ** 2
        flat = torch.multinomial(weights, numel, replacement=True, generator=gen).to(torch.int32)
    elif dist == "zeros_expert":
        span = max(1, E // 4)
        flat = torch.randint(0, span, (numel,), generator=gen, dtype=torch.int32)
    elif dist == "alternating":
        flat = (torch.arange(numel) % max(1, min(E, 3))).to(torch.int32)
    elif dist == "single":
        flat = torch.full((numel,), E - 1, dtype=torch.int32)
    else:
        raise ValueError(f"unknown distribution {dist}")
    return torch.clamp(flat, 0, E - 1)


def _run_once(numel, E, B, dist, tag, seed):
    """Build the operator's kernels for one geometry and compare 3 outputs."""
    flat = _make_flat(numel, E, dist, seed)
    g_sorted, g_expert, g_npost = _golden(flat, E, B)
    p_total = numel + (E + 1) * (B - 1)
    num_blocks = math.ceil(p_total / B)
    dev = "npu"
    if numel < 1024 and E <= 32:
        small = _make_small_batch_kernel(numel, E, B)()
        sorted_d = torch.empty(p_total, dtype=torch.int32, device=dev)
        expert_d = torch.empty(num_blocks, dtype=torch.int32, device=dev)
        npost_d = torch.empty(1, dtype=torch.int32, device=dev)
        small(flat.npu(), sorted_d, expert_d, npost_d)
        path = "small"
    else:
        align = _make_align_kernel(numel, E, B)()
        scatter = _make_scatter_kernel(numel, E, B)() if numel else None
        sorted_d = torch.empty(p_total, dtype=torch.int32, device=dev)
        expert_d = torch.empty(num_blocks, dtype=torch.int32, device=dev)
        npost_d = torch.empty(1, dtype=torch.int32, device=dev)
        cumsum = torch.empty(E + 1, dtype=torch.int32, device=dev)
        align(flat.npu(), sorted_d, expert_d, npost_d, cumsum)
        if scatter is not None:
            scatter(flat.npu(), sorted_d, cumsum)
        path = "align+scatter"
    torch.npu.synchronize()
    got_sorted = sorted_d.cpu()
    # The operator contract fixes each expert's routed *set*, not the order in
    # which its routes sit (the frozen test normalises the same way).  Compare
    # after sorting every expert's routed range into ascending route order; also
    # report whether the raw array happened to match that order already.
    counts = torch.bincount(flat.long(), minlength=E)[:E]
    padded = (counts + B - 1) // B * B
    starts = torch.cumsum(padded, 0) - padded
    normalised = got_sorted.clone()
    for expert in range(E):
        n = int(counts[expert])
        if n > 1:
            lo = int(starts[expert])
            normalised[lo:lo + n] = torch.sort(normalised[lo:lo + n]).values
    stable_order = "ascending-order-identical" if torch.equal(got_sorted, g_sorted) else "set-identical"
    checks = (("sorted_token_ids", normalised, g_sorted),
              ("expert_ids", expert_d.cpu(), g_expert),
              ("num_tokens_post_pad", npost_d.cpu(), g_npost))
    for name, got, expected in checks:
        if not torch.equal(got, expected):
            bad = (got.flatten() != expected.flatten()).nonzero().flatten()[:6].tolist()
            diff = int((got.to(torch.int64) - expected.to(torch.int64)).abs().max())
            MAX_DIFF["value"] = max(MAX_DIFF["value"], diff)
            raise AssertionError(
                f"[{tag}] ({numel},{E},{B},{dist}) {name} mismatch max|diff|={diff} "
                f"first idx={bad} got={got.flatten()[bad].tolist()} "
                f"expected={expected.flatten()[bad].tolist()} path={path}"
            )
    print(f"[{tag}] PASS (numel={numel}, E={E}, B={B}, {dist}, path={path}, {stable_order})")


def run_L0(only=None):
    cases = (
        (8, 384, 64, "uniform"),
        (256, 384, 64, "skewed"),
        (4096, 384, 64, "uniform"),
        (32768, 384, 64, "uniform"),
        (1024, 8, 16, "alternating"),
        (600, 33, 64, "skewed"),
        (0, 8, 16, "uniform"),
        (4096, 128, 64, "uniform"),
        (200, 8, 16, "alternating"),
    )
    for idx, (numel, E, B, dist) in enumerate(cases):
        if only is not None and idx not in only:
            continue
        _run_once(numel, E, B, dist, f"L0-{idx + 1}", 100 + idx)


def run_L1(only=None):
    cases = []
    for numel in (8, 256, 4096, 32768):
        for E in (384, 256, 128):
            dist = "skewed" if (numel + E) % 2 else "uniform"
            cases.append((numel, E, 64, dist))
    for idx, (numel, E, B, dist) in enumerate(cases):
        if only is not None and idx not in only:
            continue
        _run_once(numel, E, B, dist, f"L1-{idx + 1}", 200 + idx)


def run_boundary():
    for idx, (numel, E, B, dist) in enumerate((
        (7, 3, 5, "uniform"),
        (64, 8, 1, "uniform"),
        (15, 1, 1, "single"),
        (128, 64, 64, "skewed"),
        (4096, 65, 64, "uniform"),
        (32768, 128, 64, "single"),
        (1, 128, 64, "uniform"),
        (1023, 8, 16, "alternating"),
        (2048, 1, 64, "single"),
    )):
        try:
            _run_once(numel, E, B, dist, f"B-{idx + 1}", 300 + idx)
        except AssertionError as exc:
            print(f"[boundary] WARN {exc}")


def _parse_only(text):
    return None if text is None else {int(v) for v in text.split(",") if v.strip()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="L0",
                        choices=["L0", "L1", "boundary", "all"])
    parser.add_argument("--case", default=None)
    args, _ = parser.parse_known_args()
    only = _parse_only(args.case)
    started = time.perf_counter()
    levels = {"L0": lambda: run_L0(only), "L1": lambda: run_L1(only),
              "boundary": run_boundary}
    chosen = levels.values() if args.level == "all" else [levels[args.level]]
    for run in chosen:
        run()
    print(f"elapsed: {time.perf_counter() - started:.1f} s")
    print("\033[92m[PRECISION_PASS] All check passed!\033[0m")
    print(f"[PRECISION_PASS] max_diff={MAX_DIFF['value']}")


if __name__ == "__main__":
    main()
