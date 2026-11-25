# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Stage 4 Round 1 candidate C4: histogram by per-route on-chip atomic.

Parent: ``moe_permute_align_round1_c1.py`` (current best).  One main
optimization point: the histogram stops paying ``numel x E`` comparisons.

Stage 3 (and C1) build the per-core counts with an expert-lane compare /
accumulate loop -- ``for i in T.serial(n): acc += (s_x[i] == e)`` over ``e_pad``
lanes -- which is ``numel * ceil(E/64)`` vector ops and the dominant cost of the
histogram launch (29.1 us for 32768 routes, probe ``histcmp``; 11.6 us even with
8 independent accumulator chains, probe ``histunroll``).  Here each route does
one on-chip ``atomic_add`` into the core's own histogram instead: ``numel``
lane-ops total (probe ``histatom2``: 3.8 us for the same 32768 routes, 7.7x).

Correctness: integer ``+1`` per route into that route's bin is the definition of
the count, and each core writes only its own histogram row, so no cross-core
interaction is involved.  The route -> core partition is the placement's
(``idx = a*tpc + r*128 + j``), so the two launches keep agreeing on which core
owns which route.
"""

import functools
import os
import sys

import torch
import torch_npu  # noqa: F401 -- registers torch.npu
import tilelang
import tilelang.ascend.language as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _stage3_base_module import _make_small_batch_kernel  # noqa: E402
from moe_permute_align_round1_c1 import (  # noqa: E402
    _THREADS,
    _AlignTwoPassCur,
    _Place,
    _fused_plan,
    _scratch,
)


@functools.lru_cache(maxsize=None)
def _compile_hist_atomic(numel, E, B, g):
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


class _AlignAtomicHist(_AlignTwoPassCur):
    """C1's align callable with the atomic-histogram launch 1."""

    def __init__(self, numel, E, B, g):
        super().__init__(numel, E, B, g)
        self._hist = _compile_hist_atomic(numel, E, B, g)


def _make_align_kernel(numel: int, num_experts: int, block_size: int):
    numel = int(numel)
    E = int(num_experts)
    B = int(block_size)
    g, _, _ = _fused_plan(numel, E, B)

    def _build():
        return _AlignAtomicHist(numel, E, B, g)

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
        return _Place(numel, E, B, g)

    return _build
