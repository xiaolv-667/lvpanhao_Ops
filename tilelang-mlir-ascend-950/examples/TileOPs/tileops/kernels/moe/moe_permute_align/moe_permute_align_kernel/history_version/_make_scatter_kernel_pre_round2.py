# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Stage 4 tuned `_make_scatter_kernel` (drop-in re-export).

The scatter stage: deterministic slot placement from the per-core cursor row.

The implementation is the single self-contained module next to this file; this
module exists only because the wrapper's switch block imports one factory per
module (Stage 3 package layout).
"""

from .moe_permute_align import _make_scatter_kernel

__all__ = ["_make_scatter_kernel"]
