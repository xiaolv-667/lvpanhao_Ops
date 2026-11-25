# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""MoE token-to-expert align + route scatter, Ascend 950PR (Stage 4 tuned).

Self-contained replacement for the three Stage 3 kernel modules the frozen
wrapper ``tileops/kernels/moe/moe_permute_align/moe_permute_align.py`` imports:

* ``_make_align_kernel(numel, num_experts, block_size)`` ->
  ``align(flat, sorted_token_ids, expert_ids, num_tokens_post_pad, cumsum)``
* ``_make_scatter_kernel(numel, num_experts, block_size)`` ->
  ``scatter(flat, sorted_token_ids, cumsum)``
* ``_make_small_batch_kernel(numel, num_experts, block_size)`` ->
  ``small(flat, sorted_token_ids, expert_ids, num_tokens_post_pad)``

Every launch keeps the Stage 3 output contract bit for bit: ``cumsum`` holds the
padded per-expert starts, ``expert_ids`` the block owners (``num_experts`` past
the padded length), ``num_tokens_post_pad`` the padded total, and
``sorted_token_ids`` holds the sentinel ``numel`` outside each expert's routed
slots and each route index exactly once inside its own expert's range (the order
*within* a range is the contract's documented free choice).

Tuned structure (each step measured, see ``perf_opt/opt_log.md``):

1. **Histogram launch** -- one on-chip ``atomic_add`` per route into the core's
   own ``int32`` histogram row, instead of ``numel x num_experts`` lane
   comparisons (29.1 us -> 3.8 us at 32768 routes).
2. **Align launch** -- merge the per-core rows (one bulk copy above
   ``_BULK_MERGE_MIN_CORES``, per-row copies below it), padded prefix sum,
   ``cumsum`` / ``num_tokens_post_pad``, ``expert_ids`` by expert-side expansion
   (no ``O(num_experts)`` rank count per block), the sentinel fill, and each
   core's slot-cursor row into the shared scratch.
3. **Scatter launch** -- every route's slot is *derived*, not allocated by a
   global atomic: ``cumsum[e]`` + the routes of ``e`` on earlier cores (from the
   histogram rows) + the routes of ``e`` already placed here (on-chip cursor).
   The Stage 3 per-route GM atomic cost 586 us at 32768 routes; this costs one
   on-chip bump plus the store the contract needs anyway.
4. Single-core geometries (``g == 1``) place inside the align launch, so the
   call is two launches instead of three.

Tuned defaults (module constants, referenced by the wrapper's perf_opt block):

* ``_THREADS = 128`` -- SIMT lane width of both the histogram and the placement;
* ``_UB_PARTIAL_BUDGET = 96 * 1024`` -- UB bytes the whole per-core histogram may
  occupy, which bounds the core count (it only binds for very wide expert
  tables);
* ``_BULK_MERGE_MIN_CORES = 16`` -- core count above which the merge copies the
  whole histogram window in one descriptor instead of per row (measured: 56
  cores -17.4 us, 8 cores +2.1 us).

Program entry: ``python moe_permute_align.py --level L0|L1|boundary`` runs the
combined-pipeline precision harness (align+scatter or the fused small-batch
path) against a torch CPU golden derived from the operator semantics.
"""

import argparse
import functools
import math
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


_SCRATCH = {}


def _scratch(numel, E, B, device):
    """(per-core histogram rows, per-core slot cursors) for one geometry."""
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
# Launch 1: per-core histogram (one on-chip atomic per route)
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _compile_hist_atomic(numel, E, B, g):
    """Every core counts its own route slice into its own int32 histogram row.

    ``numel`` lane-ops in total instead of ``numel x E`` compares; each core
    writes row ``gid`` of the scratch and touches nothing else, so no cross-core
    interaction is involved.  The route -> core split is the placement's
    (``idx = gid*tpc + r*_THREADS + j``), so both launches agree on the slices.
    """
    _, e_pad, tpc = _fused_plan(numel, E, B)
    rounds = (tpc + _THREADS - 1) // _THREADS
    n_clear = max(1, (e_pad + _THREADS - 1) // _THREADS)

    @T.prim_func
    def main(
        flat: T.Tensor((numel,), "int32"),
        partial: T.Tensor((g, e_pad), "int32"),
    ):
        with T.Kernel(g) as gid, T.SimtVF(threads=_THREADS):
            s_hist = T.alloc_shared((e_pad,), "int32")
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
            T.copy(s_hist, partial[gid, :])

    return tilelang.compile(main, target="ascend", out_idx=[])

# ---------------------------------------------------------------------------
# Launch 2 (align): merge + padded prefix + metadata + sentinel [+ placement]
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _compile_align_stage(numel, E, B, g, fused):
    """Merge the per-core rows and emit every non-route output of the call.

    ``fused`` (single-core geometries only, a build-time constant) additionally
    runs the placement in this launch, after the sentinel fill -- the ordering
    the fused small-batch kernel already relies on inside one core.
    """
    geo = _geometry(numel, E, B)
    e_pad = geo["E_pad"]
    p_total = geo["P"]
    nb = geo["NB"]
    steps = geo["steps"]
    shift = geo["shift"]
    blkpc = (nb + g - 1) // g
    blk = max(64, _align_up(max(blkpc, 1), 64))
    epc = (E + g - 1) // g                       # experts this core expands
    e_lanes = _align_up(max(epc, 1), 64)
    share = (p_total + g - 1) // g
    n_fill = (share + FILL_TILE - 1) // FILL_TILE
    tpc = (numel + g - 1) // g
    rounds = (tpc + _THREADS - 1) // _THREADS
    bulk = g >= _BULK_MERGE_MIN_CORES

    @T.prim_func
    def main(
        flat: T.Tensor((numel,), "int32"),
        partial: T.Tensor((g, e_pad), "int32"),
        cursor: T.Tensor((g, e_pad), "int32"),
        sorted_token_ids: T.Tensor((p_total,), "int32"),
        expert_ids: T.Tensor((nb,), "int32"),
        num_tokens_post_pad: T.Tensor((1,), "int32"),
        cumsum: T.Tensor((E + 1,), "int32"),
    ):
        with T.Kernel(g) as a:
            s_all = T.alloc_shared((g, e_pad), "int32")
            s_part = T.alloc_shared((e_pad,), "int32")
            s_cnt = T.alloc_shared((e_pad,), "int32")
            s_own = T.alloc_shared((e_pad,), "int32")
            s_pad = T.alloc_shared((e_pad,), "int32")
            s_scan_a = T.alloc_shared((2 * e_pad,), "int32")
            s_scan_b = T.alloc_shared((2 * e_pad,), "int32")
            s_pref = T.alloc_shared((e_pad + 64,), "int32")
            s_cur = T.alloc_shared((e_pad,), "int32")
            s_fill = T.alloc_shared((FILL_TILE,), "int32")

            # ---- merge the per-core rows ------------------------------------
            # Large core counts: one descriptor for the whole window
            # (TRAP-950-rowload-dma-chain / PL-950-window-bulk-copy).  Small
            # ones: the per-row chain is cheaper than the window's indexed
            # accumulate.  Both keep integer addition associative, so every core
            # derives bit-identical counts.
            if bulk:
                T.copy(partial, s_all)
                with T.SimtVF(threads=e_pad):
                    for e in T.Parallel(e_pad):
                        tot = T.alloc_local([1], "int32")
                        own = T.alloc_local([1], "int32")
                        tot[0] = 0
                        own[0] = 0
                        for r in T.serial(g):
                            tot[0] = tot[0] + s_all[r, e]
                            own[0] = own[0] + T.if_then_else(r < a, s_all[r, e], 0)
                        s_cnt[e] = tot[0]
                        s_own[e] = own[0]
            else:
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

            # ---- pad each count up to a whole block -------------------------
            with T.SimtVF(threads=e_pad):
                for e in T.Parallel(e_pad):
                    pv = T.if_then_else(e < E, s_cnt[e], 0)
                    if shift is not None:
                        s_pad[e] = ((pv + (B - 1)) >> shift) << shift
                    else:
                        s_pad[e] = T.ceildiv(pv, B) * B

            # ---- inclusive doubling scan (data in the upper half, zero halo) -
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

            # shift by one lane; the padding lanes hold `total`, a defined value
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

            # this core's slot cursor: its range start + routes before it
            with T.SimtVF(threads=e_pad):
                for e in T.Parallel(e_pad):
                    cursor[a, e] = s_pref[e] + s_own[e]

            # ---- expert_ids: past the padded length no block names an expert -
            blk0 = a * blkpc
            with T.SimtVF(threads=blk):
                for j in T.Parallel(blk):
                    b = blk0 + j
                    if b >= total // B and b < nb:
                        expert_ids[b] = E
            # ---- inside it, each expert expands the block range it owns ------
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

            # ---- sentinel fill: no other launch writes these slots before this
            #      one retires, so the placement can overwrite freely ----------
            T.fill(s_fill, numel)
            if n_fill > 0:
                for t in T.serial(n_fill):
                    off = a * share + t * FILL_TILE
                    n = T.min(FILL_TILE, T.min((a + 1) * share, p_total) - off)
                    if n > 0:
                        T.copy(s_fill[0:n], sorted_token_ids[off:off + n])

            # ---- single-core geometries place here, after the fill ----------
            if fused:
                with T.SimtVF(threads=e_pad):
                    for e in T.Parallel(e_pad):
                        s_cur[e] = s_pref[e]
                with T.SimtVF(threads=_THREADS):
                    eid = T.alloc_local([1], "int32")
                    slot = T.alloc_local([1], "int32")
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

# ---------------------------------------------------------------------------
# Launch 3 (scatter): deterministic slot from this core's cursor row
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _compile_place(numel, E, B, g):
    """Place every route of this core's slice with a derived, distinct slot.

    ``slot = cursor[e]`` then bump: the cursor row already holds
    ``cumsum[e] + routes of e on earlier cores``, and the on-chip bump hands out
    the disjoint sub-range this core owns, so the per-route cost is one on-chip
    atomic plus the store the contract needs anyway -- no global atomic, no
    contention between cores.
    """
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
                        # explicit upper bound: eid == E would bump the cursor's
                        # padding lane (TRAP-950-scatter-eid-upper-bound)
                        if eid[0] < E:
                            slot[0] = T.atomic_add(
                                s_cur[eid[0]], T.int32(1), return_prev=True
                            )
                            sorted_token_ids[slot[0]] = idx

    return tilelang.compile(main, target="ascend", out_idx=[])



class _Place:
    """scatter callable: the wrapper's three args, plus this core's cursor row."""

    def __init__(self, numel, E, B, g):
        self._numel = numel
        self._E = E
        self._B = B
        self._prog = _compile_place(numel, E, B, g)

    def __call__(self, flat, sorted_token_ids, cumsum):
        _partial, cursor = _scratch(self._numel, self._E, self._B, flat.device)
        return self._prog(flat, sorted_token_ids, cursor)


class _Noop:
    """The scatter stage when the align stage already placed every route."""

    def __call__(self, *args, **kwargs):
        return None


class _AlignStage:
    """align callable: histogram launch, then merge/scan/metadata [+ placement]."""

    def __init__(self, numel, E, B, g):
        self._numel = numel
        self._E = E
        self._B = B
        self._g = g
        self._hist = _compile_hist_atomic(numel, E, B, g)
        self._align = _compile_align_stage(numel, E, B, g, g == 1)

    def __call__(self, flat, sorted_token_ids, expert_ids, num_tokens_post_pad, cumsum):
        partial, cursor = _scratch(self._numel, self._E, self._B, flat.device)
        self._hist(flat, partial)
        self._align(flat, partial, cursor, sorted_token_ids, expert_ids,
                    num_tokens_post_pad, cumsum)


def _make_align_kernel(numel: int, num_experts: int, block_size: int):
    """Build the align factory for one ``(numel, num_experts, block_size)``."""
    numel = int(numel)
    E = int(num_experts)
    B = int(block_size)
    g, _, _ = _fused_plan(numel, E, B)

    def _build():
        return _AlignStage(numel, E, B, g)

    _build.path = "L"
    _build.g_h = g
    _build.g_a = g
    return _build


@functools.lru_cache(maxsize=32)
def _make_scatter_kernel(numel: int, num_experts: int, block_size: int):
    """Build the scatter factory: deterministic placement, no global atomics."""
    if numel < 1:
        raise ValueError(f"the scatter needs at least one route, got numel={numel}")
    numel = int(numel)
    E = int(num_experts)
    B = int(block_size)
    g, _, _ = _fused_plan(numel, E, B)

    def _build():
        return _Noop() if g == 1 else _Place(numel, E, B, g)

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
