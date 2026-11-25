# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Stage 4 Round 1 candidate C2: the DESIGN's 备选 B (two-level slot reserve).

The frozen scatter ``DESIGN.md`` §1.6.3 leaves exactly this alternative for Stage
4 to decide (its 实验裁决计划 T2): keep the Stage 3 align stage untouched and give
the scatter the source operator's two-level structure --

1. per-block histogram in UB (``rank = atomic_add(s_cnt[eid], 1, prev)``),
2. one batched GM reserve per (block, expert)
   (``base = atomic_add(cumsum[e], s_cnt[e], prev)``),
3. write ``sorted[base + rank] = idx`` in a second pass.

Global atomics drop from ``numel`` to ``sum_blocks min(tile, E)`` and hot experts
are no longer hammered by every lane.  One main optimization point; the align
stage (both launches) is the Stage 3 source verbatim.
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
from _stage3_base_module import *  # noqa: E402,F401,F403  (Stage 3 align + factories)
from _stage3_base_module import (  # noqa: E402
    _NBLK_TARGET,
    _THREADS,
    _make_align_kernel,
    _make_small_batch_kernel,
)


@functools.lru_cache(maxsize=32)
def _make_scatter_kernel(numel: int, num_experts: int, block_size: int):
    """Two-level scatter: block histogram + batched per-expert global reserve."""
    if numel < 1:
        raise ValueError(f"the scatter needs at least one route, got numel={numel}")
    max_padded = numel + (num_experts + 1) * (block_size - 1)
    ept = max(1, math.ceil(math.ceil(numel / _NBLK_TARGET) / _THREADS))
    tile = _THREADS * ept
    nblocks = math.ceil(numel / tile)
    e_rounds = math.ceil(num_experts / _THREADS)

    @tilelang.jit(out_idx=[], target="ascend")
    def _scatter():
        @T.prim_func
        def _scatter_main(
            flat: T.Tensor([numel], "int32"),
            sorted_token_ids: T.Tensor([max_padded], "int32"),
            cumsum: T.Tensor([num_experts + 1], "int32"),
        ):
            with T.Kernel(nblocks) as bid, T.SimtVF(threads=_THREADS):
                s_cnt = T.alloc_shared((num_experts,), "int32")
                s_base = T.alloc_shared((num_experts,), "int32")
                s_eid = T.alloc_shared((tile,), "int32")
                s_rnk = T.alloc_shared((tile,), "int32")
                eid = T.alloc_local([1], "int32")

                for i in T.serial(e_rounds):
                    for j in T.Parallel(_THREADS):
                        e = i * _THREADS + j
                        if e < num_experts:
                            s_cnt[e] = 0

                # pass 1: this block's histogram, ranked per route
                for i in T.serial(ept):
                    for j in T.Parallel(_THREADS):
                        idx = bid * tile + i * _THREADS + j
                        eid[0] = T.if_then_else(
                            idx < numel, flat[T.min(idx, numel - 1)], -1
                        )
                        s_eid[i * _THREADS + j] = eid[0]
                        if eid[0] >= 0 and eid[0] < num_experts:
                            s_rnk[i * _THREADS + j] = T.atomic_add(
                                s_cnt[eid[0]], T.int32(1), return_prev=True
                            )

                # pass 2: one global reserve per non-empty (block, expert)
                for i in T.serial(e_rounds):
                    for j in T.Parallel(_THREADS):
                        e = i * _THREADS + j
                        if e < num_experts:
                            if s_cnt[e] > 0:
                                s_base[e] = T.atomic_add(
                                    cumsum[e], s_cnt[e], return_prev=True
                                )

                # pass 3: write with base + rank
                for i in T.serial(ept):
                    for j in T.Parallel(_THREADS):
                        idx = bid * tile + i * _THREADS + j
                        if idx < numel:
                            eid[0] = s_eid[i * _THREADS + j]
                            if eid[0] >= 0 and eid[0] < num_experts:
                                sorted_token_ids[s_base[eid[0]] + s_rnk[i * _THREADS + j]] = idx

        return _scatter_main

    return _scatter
