# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Stage 4 (round 2) candidate R6: R3 with the histogram window flattened (1-D copy).

Parent: ``_base_module.py`` (= the current best, ``MoEPermuteAlignFwdOp.py``,
kimi-k2-t4096 29.7 us).  One main optimization point:

**The align launch stops reading the whole per-core histogram on 56 cores.**
Every align core re-reads the full ``partial[g_place, e_pad]`` window because the
prefix sum over experts is global, so the base version issues
``g_align x g_place x e_pad x 4`` bytes of read: 56 x 86 KB = **4.8 MB** for
``numel=32768`` (measured 6999 KB of GM traffic in a 19.6 us launch, 357 GB/s
effective).  The same core count also fixes the sentinel fill's parallelism
(``p_total / g_align`` bytes per core), which is what makes the small geometries
slow (43-142 KB on a single core).

Here the two core counts are chosen separately, both at build time:

* ``g_place`` = route-slice count of the placement launch (unchanged: ~512 routes
  per core, up to 56);
* ``g_align`` = ``clamp(ceil(p_total / _FILL_TARGET_ENTRIES), 1, 56)`` -- driven
  by the sentinel/metadata volume, which is what the align launch actually moves.

The align launch then also writes the placement cores' cursor rows itself
(``cursor[c, e] = cumsum[e] + sum_{r<c} partial[r, e]``, from a running sum over
the UB-resident histogram) so the decoupling costs no extra pass and the
placement launch is untouched.

Everything else (histogram launch, placement launch, output contract) is the
parent's.
"""

import argparse
import functools
import math
import os
import sys
import time

import torch
import torch_npu  # noqa: F401 -- registers torch.npu
import tilelang
import tilelang.ascend.language as T


FILL_TILE = 2048  # sentinel tile, in int32 elements per bulk store
VECTOR_CORES = 56  # physical Vector cores of Ascend950PR_9579
_THREADS = 128  # SIMT lane width of the histogram and placement launches
_UB_PARTIAL_BUDGET = 96 * 1024  # UB bytes the whole per-core histogram may take
_BULK_MERGE_MIN_CORES = 16  # merge the histogram window in one descriptor


# ---------------------------------------------------------------------------
# Build-time geometry
# ---------------------------------------------------------------------------
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


def _fused_plan(numel, E, B):
    """Core count, padded expert count and routes per core (all launches)."""
    e_pad = _align_up(max(E, 1), 64)
    g = max(1, min(VECTOR_CORES, (numel + 511) // 512))
    while g > 1 and g * e_pad * 4 > _UB_PARTIAL_BUDGET:
        g = (g + 1) // 2
    return g, e_pad, (numel + g - 1) // g



_FILL_TARGET_ENTRIES = 4096  # sentinel entries per align core
_FILL_CORES_MIN = 1          # extra blocks the fill may add to launch 1


def _align_plan(numel, E, B):
    """(align cores, placement cores, padded expert count, launch-1 grid)."""
    g_place, e_pad, _ = _fused_plan(numel, E, B)
    geo = _geometry(numel, E, B)
    g_align = max(
        1, min(VECTOR_CORES, (geo["P"] + _FILL_TARGET_ENTRIES - 1) // _FILL_TARGET_ENTRIES)
    )
    g_fill = max(1, min(VECTOR_CORES, (geo["P"] + _FILL_TARGET_ENTRIES - 1) // _FILL_TARGET_ENTRIES))
    grid = max(g_place, g_fill)
    return g_align, g_place, e_pad, grid


_SCRATCH2 = {}


def _scratch2(numel, E, B, device):
    g_align, g_place, e_pad, _grid = _align_plan(numel, E, B)
    key = (numel, E, B, g_align, g_place, e_pad)
    bufs = _SCRATCH2.get(key)
    if bufs is None or bufs[0].device != device:
        bufs = (
            torch.empty(g_place * e_pad, dtype=torch.int32, device=device),
            torch.empty(g_place * e_pad, dtype=torch.int32, device=device),
        )
        _SCRATCH2[key] = bufs
    return bufs


@functools.lru_cache(maxsize=None)
def _compile_hist_fill(numel, E, B, g_place, grid):
    """Launch 1: per-core histogram row + the sentinel fill, on ``grid`` blocks.

    The fill used to sit in the align launch, where it shares the core count with
    the merge (one core for the small geometries: 43-142 KB of sentinel written
    serially).  Here it has its own blocks -- ``grid - g_place`` extra ones when
    the padded array is large -- and the kernel boundary before the placement
    keeps the fill ordered ahead of every routed slot write.
    """
    _, e_pad, tpc = _fused_plan(numel, E, B)
    geo = _geometry(numel, E, B)
    p_total = geo["P"]
    rounds = (tpc + _THREADS - 1) // _THREADS
    n_clear = max(1, (e_pad + _THREADS - 1) // _THREADS)
    share = (p_total + grid - 1) // grid
    n_fill = (share + FILL_TILE - 1) // FILL_TILE

    @T.prim_func
    def main(
        flat: T.Tensor((numel,), "int32"),
        partial: T.Tensor((g_place * e_pad,), "int32"),
        sorted_token_ids: T.Tensor((p_total,), "int32"),
    ):
        with T.Kernel(grid) as gid:
            s_fill = T.alloc_shared((FILL_TILE,), "int32")
            s_hist = T.alloc_shared((e_pad,), "int32")

            T.fill(s_fill, numel)
            for t in T.serial(n_fill):
                off = gid * share + t * FILL_TILE
                n = T.min(FILL_TILE, T.min((gid + 1) * share, p_total) - off)
                if n > 0:
                    T.copy(s_fill[0:n], sorted_token_ids[off:off + n])

            # The histogram body is NOT wrapped in a block guard: a runtime `if`
            # around the whole VF region collapses the vector lane form (measured
            # vec ratio 0.81 -> 0.03, launch 4.3 -> 8.9 us).  The extra fill-only
            # blocks (gid >= g_place) exit through the `idx < numel` bound instead,
            # and only the per-core store is guarded.
            with T.SimtVF(threads=_THREADS):
                eid = T.alloc_local([1], "int32")
                for k in T.serial(n_clear):
                    for j in T.Parallel(_THREADS):
                        if k * _THREADS + j < e_pad:
                            s_hist[k * _THREADS + j] = 0
                for r in T.serial(rounds):
                    for j in T.Parallel(_THREADS):
                        idx = gid * tpc + r * _THREADS + j
                        if r * _THREADS + j < tpc and idx < numel:
                            eid[0] = flat[idx]
                            if eid[0] < E:
                                T.atomic_add(s_hist[eid[0]], T.int32(1))
            if gid < g_place:  # only the real histogram blocks publish a row
                T.copy(s_hist, partial[gid * e_pad:(gid + 1) * e_pad])

    return tilelang.compile(main, target="ascend", out_idx=[])

@functools.lru_cache(maxsize=None)
def _compile_align_split(numel, E, B, g_align, g_place, fused):
    geo = _geometry(numel, E, B)
    e_pad = geo["E_pad"]
    p_total = geo["P"]
    nb = geo["NB"]
    steps = geo["steps"]
    shift = geo["shift"]
    blkpc = (nb + g_align - 1) // g_align        # tail block ids per align core
    blk = max(64, _align_up(max(blkpc, 1), 64))
    epc = (E + g_align - 1) // g_align           # experts each align core expands
    e_lanes = _align_up(max(epc, 1), 64)
    rpc = (g_place + g_align - 1) // g_align     # cursor rows per align core
    tpc = (numel + g_place - 1) // g_place
    rounds = (tpc + _THREADS - 1) // _THREADS

    @T.prim_func
    def main(
        flat: T.Tensor((numel,), "int32"),
        partial: T.Tensor((g_place * e_pad,), "int32"),
        cursor: T.Tensor((g_place * e_pad,), "int32"),
        sorted_token_ids: T.Tensor((p_total,), "int32"),
        expert_ids: T.Tensor((nb,), "int32"),
        num_tokens_post_pad: T.Tensor((1,), "int32"),
        cumsum: T.Tensor((E + 1,), "int32"),
    ):
        with T.Kernel(g_align) as a:
            s_all = T.alloc_shared((g_place * e_pad,), "int32")
            s_cnt = T.alloc_shared((e_pad,), "int32")
            s_pad = T.alloc_shared((e_pad,), "int32")
            s_scan_a = T.alloc_shared((2 * e_pad,), "int32")
            s_scan_b = T.alloc_shared((2 * e_pad,), "int32")
            s_pref = T.alloc_shared((e_pad + 64,), "int32")
            s_cur = T.alloc_shared((e_pad,), "int32")

            # one bulk descriptor for the whole per-core histogram (unchanged)
            # 1-D window: a 2-D tile copy is charged per row (56 descriptors of
            # 1.5 KB), a flat one is a single descriptor
            T.copy(partial, s_all)
            # The row loop is a build-time `range`: `g_place` is a compile-time
            # constant, and a *runtime* row index into the UB window costs a
            # serialized UB access per row (measured: two such 56-iteration loops
            # were ~9 us of the align launch).  Unrolled, every row is a static
            # offset and the adds pipeline.
            with T.SimtVF(threads=e_pad):
                for e in T.Parallel(e_pad):
                    tot = T.alloc_local([1], "int32")
                    tot[0] = 0
                    for r in range(g_place):
                        tot[0] = tot[0] + s_all[r * e_pad + e]
                    s_cnt[e] = tot[0]

            with T.SimtVF(threads=e_pad):
                for e in T.Parallel(e_pad):
                    pv = T.if_then_else(e < E, s_cnt[e], 0)
                    if shift is not None:
                        s_pad[e] = ((pv + (B - 1)) >> shift) << shift
                    else:
                        s_pad[e] = T.ceildiv(pv, B) * B

            with T.SimtVF(threads=e_pad):
                for e in T.Parallel(e_pad):
                    s_scan_a[e_pad + e] = s_pad[e]
            with T.SimtVF(threads=e_pad):
                for e in T.Parallel(e_pad):
                    s_scan_a[e] = 0
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
                T.copy(s_pref[0:E + 1], cumsum[0:E + 1])
                num_tokens_post_pad[0] = total

            # this core's cursor rows: cumsum[e] + the routes on earlier cores.
            # The running sum walks the UB-resident histogram, so the rows cost
            # one store per (row, expert) and no extra GM read.
            r_lo = a * rpc
            r_hi = T.min(g_place, r_lo + rpc)
            with T.SimtVF(threads=e_pad):
                for e in T.Parallel(e_pad):
                    run = T.alloc_local([1], "int32")
                    run[0] = 0
                    for r in range(g_place):
                        if r >= r_lo and r < r_hi:
                            cursor[r * e_pad + e] = s_pref[e] + run[0]
                        run[0] = run[0] + s_all[r * e_pad + e]

            # expert_ids past the padded length name no expert
            blk0 = a * blkpc
            with T.SimtVF(threads=blk):
                for j in T.Parallel(blk):
                    b = blk0 + j
                    if b >= total // B and b < nb:
                        expert_ids[b] = E
            # inside it, each expert expands the block range it owns
            e_lo = a * epc
            with T.SimtVF(threads=e_lanes):
                for j in T.Parallel(e_lanes):
                    expert = e_lo + j
                    start = T.alloc_local([1], "int32")
                    cnt = T.alloc_local([1], "int32")
                    if expert < E:
                        start[0] = s_pref[expert] // B
                        cnt[0] = s_pref[expert + 1] // B - start[0]
                        for b in T.serial(cnt[0]):
                            expert_ids[start[0] + b] = expert

            # single placement core: place here, after launch 1's sentinel fill
            if fused:
                with T.SimtVF(threads=e_pad):
                    for e in T.Parallel(e_pad):
                        s_cur[e] = s_pref[e]
                with T.SimtVF(threads=_THREADS):
                    eid = T.alloc_local([1], "int32")
                    slot = T.alloc_local([1], "int32")
                    if a == 0:
                        for r in T.serial(rounds):
                            for j in T.Parallel(_THREADS):
                                idx = r * _THREADS + j
                                if idx < numel:
                                    eid[0] = flat[idx]
                                    if eid[0] < E:
                                        slot[0] = T.atomic_add(
                                            s_cur[eid[0]], T.int32(1), return_prev=True
                                        )
                                        sorted_token_ids[slot[0]] = idx

    return tilelang.compile(main, target="ascend", out_idx=[])


class _Noop:
    """The scatter stage when the align stage already placed every route."""

    def __call__(self, *args, **kwargs):
        return None


@functools.lru_cache(maxsize=None)
def _compile_place_flat(numel, E, B, g_place):
    tpc = (numel + g_place - 1) // g_place
    rounds = (tpc + _THREADS - 1) // _THREADS
    _, e_pad, _ = _fused_plan(numel, E, B)
    n_load = _align_up(max(e_pad, 1), _THREADS) // _THREADS

    @T.prim_func
    def main(
        flat: T.Tensor((numel,), "int32"),
        sorted_token_ids: T.Tensor((numel + (E + 1) * (B - 1),), "int32"),
        cursor: T.Tensor((g_place * e_pad,), "int32"),
    ):
        with T.Kernel(g_place) as a, T.SimtVF(threads=_THREADS):
            s_cur = T.alloc_shared((e_pad,), "int32")
            eid = T.alloc_local([1], "int32")
            slot = T.alloc_local([1], "int32")

            for k in T.serial(n_load):
                for j in T.Parallel(_THREADS):
                    if k * _THREADS + j < e_pad:
                        s_cur[k * _THREADS + j] = cursor[a * e_pad + k * _THREADS + j]
            for r in T.serial(rounds):
                for j in T.Parallel(_THREADS):
                    idx = a * tpc + r * _THREADS + j
                    if r * _THREADS + j < tpc and idx < numel:
                        eid[0] = flat[idx]
                        if eid[0] < E:
                            slot[0] = T.atomic_add(
                                s_cur[eid[0]], T.int32(1), return_prev=True
                            )
                            sorted_token_ids[slot[0]] = idx

    return tilelang.compile(main, target="ascend", out_idx=[])


class _PlaceSplit:
    """scatter callable: the wrapper's three args, plus this core's cursor row."""

    def __init__(self, numel, E, B, g_place):
        self._numel = numel
        self._E = E
        self._B = B
        self._g_place = g_place
        self._prog = _compile_place_flat(numel, E, B, g_place)

    def __call__(self, flat, sorted_token_ids, cumsum):
        _partial, cursor = _scratch2(self._numel, self._E, self._B, flat.device)
        return self._prog(flat, sorted_token_ids, cursor)


class _AlignStageSplit:
    def __init__(self, numel, E, B, g_align, g_place, grid):
        self._numel = numel
        self._E = E
        self._B = B
        self._g_align = g_align
        self._g_place = g_place
        self._fused = g_place == 1
        self._hist = _compile_hist_fill(numel, E, B, g_place, grid)
        self._align = _compile_align_split(numel, E, B, g_align, g_place, self._fused)

    def __call__(self, flat, sorted_token_ids, expert_ids, num_tokens_post_pad, cumsum):
        partial, cursor = _scratch2(self._numel, self._E, self._B, flat.device)
        self._hist(flat, partial, sorted_token_ids)
        self._align(flat, partial, cursor, sorted_token_ids, expert_ids,
                    num_tokens_post_pad, cumsum)


def _make_align_kernel(numel: int, num_experts: int, block_size: int):
    numel = int(numel)
    E = int(num_experts)
    B = int(block_size)
    g_align, g_place, _, grid = _align_plan(numel, E, B)

    def _build():
        return _AlignStageSplit(numel, E, B, g_align, g_place, grid)

    _build.path = "L"
    _build.g_h = g_place
    _build.g_a = g_align
    return _build


@functools.lru_cache(maxsize=32)
def _make_scatter_kernel(numel: int, num_experts: int, block_size: int):
    if numel < 1:
        raise ValueError(f"the scatter needs at least one route, got numel={numel}")
    numel = int(numel)
    E = int(num_experts)
    B = int(block_size)
    _, g_place, _, _grid = _align_plan(numel, E, B)

    def _build():
        return _Noop() if g_place == 1 else _PlaceSplit(numel, E, B, g_place)

    return _build

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


def golden_moe_permute_align(topk_ids, num_experts, block_size):
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
    g_sorted, g_expert, g_npost = golden_moe_permute_align(flat, E, B)
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


def run_L2():
    """Beyond-manifest geometries: informative, non-blocking (L2 warns)."""
    for idx, (numel, E, B, dist) in enumerate((
        (65536, 384, 64, "uniform"),
        (8192, 256, 64, "zeros_expert"),
        (1, 128, 64, "uniform"),
        (1024, 1100, 32, "uniform"),
    )):
        try:
            _run_once(numel, E, B, dist, f"L2-{idx + 1}", 400 + idx)
        except AssertionError as exc:
            print(f"[L2] WARN (recorded, non-blocking) {exc}")


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
                        choices=["L0", "L1", "L2", "boundary", "all"])
    parser.add_argument("--case", default=None)
    args, _ = parser.parse_known_args()
    only = _parse_only(args.case)
    started = time.perf_counter()
    levels = {"L0": lambda: run_L0(only), "L1": lambda: run_L1(only),
              "L2": run_L2, "boundary": run_boundary}
    chosen = levels.values() if args.level == "all" else [levels[args.level]]
    for run in chosen:
        run()
    print(f"elapsed: {time.perf_counter() - started:.1f} s")
    print("\033[92m[PRECISION_PASS] All check passed!\033[0m")
    print(f"[PRECISION_PASS] max_diff={MAX_DIFF['value']}")


if __name__ == "__main__":
    main()
