# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""paged_cache_write Stage 4 variant r1_c1: row-DMA staging, one token per block.

Single main change against the Stage 3 elementwise scatter: all GM traffic moves
to ``T.copy`` descriptors (see ``_pcw_kernel_dma``).  This branch keeps the
Stage 3 task mapping (one row per block, both pools in the same block) so the
gain is attributable to the copy path alone.

Content: kernel factory + tile policy; the golden and the layered precision
cases come from ``_pcw_tests`` (the same byte-exact cases as Stage 3).
"""

from _pcw_kernel_dma import make_factory
from _pcw_tests import main as _run_tests

_ROWS = 1
_SPLIT = False
_STORE = "dma"
_THREADS = 128

_paged_cache_write_kernel = make_factory(_SPLIT, _STORE, _THREADS)
_DEFAULT_CONFIG = {"block_rows": 16, "block_cols": 128}


def pick_config(tokens, heads_kv, dim, physical_tokens, in_dtype, cache_dtype):
    """One token per block, the whole row in one DMA."""
    return _ROWS, heads_kv * dim


if __name__ == "__main__":
    _run_tests(globals())
