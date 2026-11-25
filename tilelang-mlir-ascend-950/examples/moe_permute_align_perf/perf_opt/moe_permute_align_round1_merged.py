# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Stage 4 Round 1 merged candidate: C1 + C3 + C4.

Three independently measured mechanisms, each a Round 1 branch on its own:

* **C1** (`moe_permute_align_round1_c1.py`) -- the scatter's per-route GM atomic
  becomes a deterministic slot from the per-core histogram plus an on-chip
  cursor bump;
* **C3** (`moe_permute_align_round1_c3.py`) -- ``expert_ids`` by expert-side
  expansion instead of an ``O(num_experts)`` rank count per block;
* **C4** (`moe_permute_align_round1_c4.py`) -- the histogram by per-route
  on-chip atomic instead of ``numel x E`` lane comparisons.

The three touch disjoint parts of the pipeline (placement launch, block-owner
loop, histogram launch), so this file is their composition and nothing else:
the align stage is C3's, with C4's histogram kernel swapped in, and the scatter
stage is C1's placement.
"""

import functools
import os
import sys

import torch
import torch_npu  # noqa: F401 -- registers torch.npu
import tilelang
import tilelang.ascend.language as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _stage3_base_module import _make_small_batch_kernel  # noqa: E402,F401
from moe_permute_align_round1_c3 import (  # noqa: E402
    _AlignExpand,
    _Place,
    _THREADS,
    _compile_align_expand,
    _fused_plan,
    _scratch,
)
from moe_permute_align_round1_c4 import _compile_hist_atomic  # noqa: E402


class _AlignExpandAtomic(_AlignExpand):
    """C3's align callable with C4's histogram launch."""

    def __init__(self, numel, E, B, g):
        super().__init__(numel, E, B, g)
        self._hist = _compile_hist_atomic(numel, E, B, g)


def _make_align_kernel(numel: int, num_experts: int, block_size: int):
    numel = int(numel)
    E = int(num_experts)
    B = int(block_size)
    g, _, _ = _fused_plan(numel, E, B)

    def _build():
        return _AlignExpandAtomic(numel, E, B, g)

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
