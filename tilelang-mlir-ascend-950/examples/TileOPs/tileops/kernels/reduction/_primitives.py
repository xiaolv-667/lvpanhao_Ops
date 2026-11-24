"""Compatibility exports for the existing NPUIR reduction helpers."""

from tileops.toolchains.npuir.reduction import (
    DEFAULT_ALIGNMENT,
    MAX_SINGLE_TILE_COLS,
    SHARED_MEMORY_BUDGET_BYTES,
    UB_SAFETY_RESERVE_BYTES,
    align_up,
    compute_tile_n,
    device_smem_budget,
    make_cumulative_scan,
    make_reduce_epilogue,
    make_softmax_epilogue,
    make_welford_update,
    tune_by_forward,
    ub_slab_units,
)

__all__ = [
    "DEFAULT_ALIGNMENT",
    "MAX_SINGLE_TILE_COLS",
    "SHARED_MEMORY_BUDGET_BYTES",
    "UB_SAFETY_RESERVE_BYTES",
    "align_up",
    "compute_tile_n",
    "device_smem_budget",
    "make_cumulative_scan",
    "make_reduce_epilogue",
    "make_softmax_epilogue",
    "make_welford_update",
    "tune_by_forward",
    "ub_slab_units",
]
