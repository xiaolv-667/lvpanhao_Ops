# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Stage 4 tuned (perf_opt) kernels for ``MoEPermuteAlignFwdOp``.

Drop-in replacement for the Stage 3 kernel package one level up: the same three
factories with the same signatures, so the frozen wrapper's
baseline/perf_opt import switch block only swaps the source it imports from.

The tuned implementation lives in ``moe_permute_align.py`` of this package (one
self-contained module, copied verbatim from the Stage 4 **round 2** task
directory ``examples/moe_permute_align_perf2/perf_opt/MoEPermuteAlignFwdOp.py``;
round 1 came from ``examples/moe_permute_align_perf/perf_opt``, kept in
``history_version/*_pre_round2.py``).  The three ``_make_*_kernel`` modules only
re-export its factories, because the switch block imports one factory per module
(Stage 3 layout parity).  Round 2 re-exports the same three factories plus
``_THREADS`` / ``_UB_PARTIAL_BUDGET`` / ``_BULK_MERGE_MIN_CORES`` -- no re-export
change was needed.
"""

from .moe_permute_align import (
    _BULK_MERGE_MIN_CORES,
    _THREADS,
    _UB_PARTIAL_BUDGET,
    _make_align_kernel,
    _make_scatter_kernel,
    _make_small_batch_kernel,
)

__all__ = [
    "_make_align_kernel",
    "_make_scatter_kernel",
    "_make_small_batch_kernel",
    "_THREADS",
    "_UB_PARTIAL_BUDGET",
    "_BULK_MERGE_MIN_CORES",
]
