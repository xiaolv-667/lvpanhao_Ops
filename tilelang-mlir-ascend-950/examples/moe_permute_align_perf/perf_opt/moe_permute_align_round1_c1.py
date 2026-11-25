# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Stage 4 Round 1 candidate C1: deterministic slot allocation (no global atomics).

Derived from the current best (`moe_permute_align.py`, the Stage 3 source).  One
main optimization point:

**C1 -- the scatter stage stops taking a global atomic per route.**  The Stage 3
scatter advances a GM bump cursor once per route
(``slot = atomic_add(cumsum[eid], 1, prev)``), measured at 586.0 us for 32768
routes / 384 experts (per-route GM atomic, ``probes/probe_mechanisms.py`` case
``gmcur``).  Here each route's slot comes from the per-core histogram the align
stage already merges:

    slot = cumsum[e] + (routes of e on earlier cores)   # from the partial rows
                    + (routes of e already placed here) # UB cursor

so the per-route cost is one on-chip cursor bump (5 cyc/route, probe ``ubatom``)
plus the store the contract needs anyway (probe ``rand``: 6.7 us for the same
32768 scattered GM stores).

The align stage keeps both of its launches and gains one output (same
mechanism): each core writes its *slot cursor row* into the shared scratch next
to the partial histogram it already reads.

Structure per call (unchanged launch count vs. Stage 3):

* launch 1 ``hist``   : per-core partial histogram;
* launch 2 ``align``  : merge partials, padded prefix sum, ``cumsum`` /
  ``num_tokens_post_pad`` / ``expert_ids``, sentinel fill, and this core's
  cursor row;
* launch 3 ``scatter``: place every route of this core's slice with the
  deterministic slot above.

The pair hands data over through a module-level scratch keyed by
``(numel, E, B, g, e_pad)`` -- the same lifetime contract Stage 3's
``_AlignTwoPass._partial`` already had, now shared with the scatter stage.
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
from _stage3_base_module import (  # noqa: E402  (current best = Stage 3 source)
    FILL_TILE,
    VECTOR_CORES,
    _align_up,
    _compile_hist,
    _compile_single,
    _geometry,
    _make_small_batch_kernel,
    _use_small_path,
)

_THREADS = 128
_UB_PARTIAL_BUDGET = 96 * 1024  # bytes of UB the whole partial matrix may take


# ---------------------------------------------------------------------------
# Geometry shared by the histogram, the align-merge and the placement launches
# ---------------------------------------------------------------------------
def _fused_plan(numel, E, B):
    e_pad = _align_up(max(E, 1), 64)
    g = max(1, min(VECTOR_CORES, (numel + 511) // 512))
    while g > 1 and g * e_pad * 4 > _UB_PARTIAL_BUDGET:
        g = (g + 1) // 2
    return g, e_pad, (numel + g - 1) // g


_SCRATCH = {}


def _scratch(numel, E, B, device):
    """(partial rows, per-core slot cursors) for one geometry and device."""
    g, e_pad, _ = _fused_plan(numel, E, B)
    key = (numel, E, B, g, e_pad)
    bufs = _SCRATCH.get(key)
    if bufs is None or bufs[0].device != device:
        bufs = (
            torch.empty((g, e_pad), dtype=torch.int32, device=device),
            torch.empty((g, e_pad), dtype=torch.int32, device=device),
        )
        _SCRATCH[key] = bufs
    return bufs


# ---------------------------------------------------------------------------
# Launch 2 (align): Stage 3's merge + scan + metadata, plus the cursor row
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _compile_align_cur(numel, E, B, g):
    geo = _geometry(numel, E, B)
    e_pad = geo["E_pad"]
    p_total = geo["P"]
    nb = geo["NB"]
    steps = geo["steps"]
    shift = geo["shift"]
    blkpc = (nb + g - 1) // g
    blk = max(64, _align_up(max(blkpc, 1), 64))
    share = (p_total + g - 1) // g
    n_fill = (share + FILL_TILE - 1) // FILL_TILE

    @T.prim_func
    def main(
        partial: T.Tensor((g, e_pad), "int32"),
        cursor: T.Tensor((g, e_pad), "int32"),
        sorted_token_ids: T.Tensor((p_total,), "int32"),
        expert_ids: T.Tensor((nb,), "int32"),
        num_tokens_post_pad: T.Tensor((1,), "int32"),
        cumsum: T.Tensor((E + 1,), "int32"),
    ):
        with T.Kernel(g) as a:
            s_part = T.alloc_shared((e_pad,), "int32")
            s_cnt = T.alloc_shared((e_pad,), "int32")
            s_own = T.alloc_shared((e_pad,), "int32")
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
            with T.SimtVF(threads=e_pad):
                for e in T.Parallel(e_pad):
                    s_own[e] = 0
            for r in T.serial(g):
                T.copy(partial[r, :], s_part)
                with T.SimtVF(threads=e_pad):
                    for e in T.Parallel(e_pad):
                        s_cnt[e] = s_cnt[e] + s_part[e]
                        s_own[e] = s_own[e] + T.if_then_else(r < a, s_part[e], 0)

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

            # this core's slot cursor: expert range start + routes before it
            # (per-lane stores, not a UB->UB copy: TRAP-950-shared-shift-copy)
            with T.SimtVF(threads=e_pad):
                for e in T.Parallel(e_pad):
                    cursor[a, e] = s_pref[e] + s_own[e]

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
                T.copy(s_ids[0:valid], expert_ids[blk0:blk0 + valid])

            T.fill(s_fill, numel)
            if n_fill > 0:
                for t in T.serial(n_fill):
                    off = a * share + t * FILL_TILE
                    n = T.min(FILL_TILE, T.min((a + 1) * share, p_total) - off)
                    if n > 0:
                        T.copy(s_fill[0:n], sorted_token_ids[off:off + n])

    return tilelang.compile(main, target="ascend", out_idx=[])


# ---------------------------------------------------------------------------
# Launch 3 (scatter): deterministic slot from the cursor row
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _compile_place(numel, E, B, g):
    tpc = (numel + g - 1) // g
    rounds = (tpc + _THREADS - 1) // _THREADS
    _, e_pad, _ = _fused_plan(numel, E, B)
    n_load = _align_up(max(e_pad, 1), _THREADS) // _THREADS

    @T.prim_func
    def main(
        flat: T.Tensor((numel,), "int32"),
        sorted_token_ids: T.Tensor((numel + (E + 1) * (B - 1),), "int32"),
        cursor: T.Tensor((g, e_pad), "int32"),
    ):
        with T.Kernel(g) as a, T.SimtVF(threads=_THREADS):
            s_cur = T.alloc_shared((e_pad,), "int32")
            eid = T.alloc_local([1], "int32")
            slot = T.alloc_local([1], "int32")

            for k in T.serial(n_load):
                for j in T.Parallel(_THREADS):
                    if k * _THREADS + j < e_pad:
                        s_cur[k * _THREADS + j] = cursor[a, k * _THREADS + j]
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


class _AlignTwoPassCur:
    """align callable: Stage 3's two launches, with the cursor row added."""

    def __init__(self, numel, E, B, g):
        self._numel = numel
        self._E = E
        self._B = B
        self._g = g
        self._hist = _compile_hist(numel, E, B, g)
        self._align = _compile_align_cur(numel, E, B, g)

    def __call__(self, flat, sorted_token_ids, expert_ids, num_tokens_post_pad, cumsum):
        partial, cursor = _scratch(self._numel, self._E, self._B, flat.device)
        self._hist(flat, partial)
        self._align(partial, cursor, sorted_token_ids, expert_ids,
                    num_tokens_post_pad, cumsum)


def _make_align_kernel(numel: int, num_experts: int, block_size: int):
    numel = int(numel)
    E = int(num_experts)
    B = int(block_size)
    g, _, _ = _fused_plan(numel, E, B)

    def _build():
        return _AlignTwoPassCur(numel, E, B, g)

    _build.path = "L"
    _build.g_h = g
    _build.g_a = g
    return _build


class _Place:
    """scatter callable: takes the wrapper's three args, adds the cursor row."""

    def __init__(self, numel, E, B, g):
        self._numel = numel
        self._E = E
        self._B = B
        self._prog = _compile_place(numel, E, B, g)

    def __call__(self, flat, sorted_token_ids, cumsum):
        _partial, cursor = _scratch(self._numel, self._E, self._B, flat.device)
        return self._prog(flat, sorted_token_ids, cursor)


@functools.lru_cache(maxsize=32)
def _make_scatter_kernel(numel: int, num_experts: int, block_size: int):
    """Scatter factory: deterministic placement (no global atomics)."""
    if numel < 1:
        raise ValueError(f"the scatter needs at least one route, got numel={numel}")
    numel = int(numel)
    E = int(num_experts)
    B = int(block_size)
    g, _, _ = _fused_plan(numel, E, B)

    def _build():
        return _Place(numel, E, B, g)

    return _build
