# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""paged_cache_write Stage 4 variant r1_c3: per-dtype store path, one block per tile.

Single main change against r1_c2: the store path follows the cache width.  The
16-bit build keeps the vector store (no MTE3 drain per row); the fp8 build goes
back to the DMA store because a one-byte element cannot be widened per lane
(measured: lane stores 157 us vs 45 us for the same fp8 workload).
"""

from _pcw_kernel_dma import make_factory
from _pcw_tests import main as _run_tests

_SPLIT = False
_STORE = "auto"
_THREADS = 128

_paged_cache_write_kernel = make_factory(_SPLIT, _STORE, _THREADS)
_DEFAULT_CONFIG = {"block_rows": 16, "block_cols": 128}


def pick_config(tokens, heads_kv, dim, physical_tokens, in_dtype, cache_dtype):
    row_width = heads_kv * dim
    if cache_dtype == "float8_e4m3fn":
        rows = max(1, min(16, -(-tokens // 256)))
    else:
        rows = max(1, min(32, -(-tokens // 128)))
    if row_width * rows < 256:
        rows = max(rows, -(-256 // row_width))
    return rows, row_width


if __name__ == "__main__":
    _run_tests(globals())
