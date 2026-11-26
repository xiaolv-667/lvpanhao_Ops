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

import functools
import math
import os
import sys

import torch
import torch_npu  # noqa: F401 -- registers torch.npu
import tilelang
import tilelang.ascend.language as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _base_module import (  # noqa: E402
    FILL_TILE,
    VECTOR_CORES,
    _align_up,
    _fused_plan,
    _geometry,
    _make_small_batch_kernel,
    _scratch,
    _THREADS,
)

_FILL_TARGET_ENTRIES = 8192  # sentinel entries per align core
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
