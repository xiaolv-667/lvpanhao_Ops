# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Stage 4 Round 2 candidate C6: single-core geometries fused into one launch.

Parent: ``moe_permute_align_round1_merged.py`` (current best = C1+C3+C4).  One
main optimization point: when the plan hands the whole call to **one** core
(``numel <= 512``, i.e. six of the twelve manifest workloads), the scalar
evidence of the placement no longer needs a separate launch.

With ``g == 1`` the placement runs inside the align stage's own launch, after
the sentinel fill -- the ordering the fused small-batch kernel already uses in
one core -- so the call is two launches (histogram, then everything else)
instead of three.  The scatter factory returns a no-op callable for those
geometries (the frozen wrapper always calls it once).

This is the only difference from the merged Round 1 candidate.
"""

import functools
import os
import sys

import torch
import torch_npu  # noqa: F401 -- registers torch.npu
import tilelang
import tilelang.ascend.language as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _stage3_base_module import (  # noqa: E402,F401
    FILL_TILE,
    _align_up,
    _geometry,
    _make_small_batch_kernel,
)
from moe_permute_align_round1_c1 import _THREADS, _scratch  # noqa: E402
from moe_permute_align_round1_c3 import _Place, _fused_plan  # noqa: E402
from moe_permute_align_round1_c4 import _compile_hist_atomic  # noqa: E402


@functools.lru_cache(maxsize=None)
def _compile_align_fused(numel, E, B, g, fused):
    geo = _geometry(numel, E, B)
    e_pad = geo["E_pad"]
    p_total = geo["P"]
    nb = geo["NB"]
    steps = geo["steps"]
    shift = geo["shift"]
    blkpc = (nb + g - 1) // g
    blk = max(64, _align_up(max(blkpc, 1), 64))
    epc = (E + g - 1) // g
    e_lanes = _align_up(max(epc, 1), 64)
    share = (p_total + g - 1) // g
    n_fill = (share + FILL_TILE - 1) // FILL_TILE
    tpc = (numel + g - 1) // g
    rounds = (tpc + _THREADS - 1) // _THREADS

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
            s_part = T.alloc_shared((e_pad,), "int32")
            s_cnt = T.alloc_shared((e_pad,), "int32")
            s_own = T.alloc_shared((e_pad,), "int32")
            s_pad = T.alloc_shared((e_pad,), "int32")
            s_scan_a = T.alloc_shared((2 * e_pad,), "int32")
            s_scan_b = T.alloc_shared((2 * e_pad,), "int32")
            s_pref = T.alloc_shared((e_pad + 64,), "int32")
            s_ids = T.alloc_shared((blk,), "int32")
            s_cur = T.alloc_shared((e_pad,), "int32")
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
                    if b >= total // B and b < nb:
                        expert_ids[b] = E
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

            T.fill(s_fill, numel)
            if n_fill > 0:
                for t in T.serial(n_fill):
                    off = a * share + t * FILL_TILE
                    n = T.min(FILL_TILE, T.min((a + 1) * share, p_total) - off)
                    if n > 0:
                        T.copy(s_fill[0:n], sorted_token_ids[off:off + n])

            # single-core geometry: the whole placement runs in this launch, after
            # the sentinel fill (the ordering the fused small-batch kernel uses)
            if fused:  # build-time: only a single-core geometry places here
                with T.SimtVF(threads=e_pad):
                    for e in T.Parallel(e_pad):
                        s_cur[e] = s_pref[e]
            with T.SimtVF(threads=_THREADS):
                eid = T.alloc_local([1], "int32")
                slot = T.alloc_local([1], "int32")
                if fused:
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



class _Noop:
    """The scatter stage when the align stage already placed every route."""

    def __call__(self, *args, **kwargs):
        return None


class _AlignFusedAtomic:
    """align callable: atomic histogram, then one fused merge/scan/ids/place."""

    def __init__(self, numel, E, B, g):
        self._numel = numel
        self._E = E
        self._B = B
        self._g = g
        self._hist = _compile_hist_atomic(numel, E, B, g)
        self._align = _compile_align_fused(numel, E, B, g, g == 1)

    def __call__(self, flat, sorted_token_ids, expert_ids, num_tokens_post_pad, cumsum):
        partial, cursor = _scratch(self._numel, self._E, self._B, flat.device)
        self._hist(flat, partial)
        self._align(flat, partial, cursor, sorted_token_ids, expert_ids,
                    num_tokens_post_pad, cumsum)


def _make_align_kernel(numel: int, num_experts: int, block_size: int):
    numel = int(numel)
    E = int(num_experts)
    B = int(block_size)
    g, _, _ = _fused_plan(numel, E, B)

    def _build():
        return _AlignFusedAtomic(numel, E, B, g)

    _build.path = "L"
    _build.g_h = g
    _build.g_a = g
    return _build


@functools.lru_cache(maxsize=32)
def _make_scatter_kernel(numel: int, num_experts: int, block_size: int):
    if numel < 1:
        raise ValueError(f"the scatter needs at least one route, got numel={numel}")
    numel = int(numel)
    E = int(num_experts)
    B = int(block_size)
    g, _, _ = _fused_plan(numel, E, B)

    def _build():
        return _Noop() if g == 1 else _Place(numel, E, B, g)

    return _build
