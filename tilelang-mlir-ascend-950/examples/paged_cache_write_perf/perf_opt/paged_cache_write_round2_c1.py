# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""paged_cache_write Stage 4 variant r2_c1: merged winner (round-1 branches combined).

Round 1 kept every branch to one change against the Stage 3 kernel:
  r1_c1  row-tile DMA load + per-row DMA store, one token per block
  r1_c2  row-tile DMA load + per-lane vector store
  r1_c3  per-dtype store path (16-bit lane, fp8 DMA)
  r1_c4  r1_c3 split by pool (rejected: slower on both large cases)
  r1_c5  r1_c3 with 256 SimtVF lanes (prefill 61.5 -> 53.5 us)
  r1_c6  r1_c3 with 512 SimtVF lanes (prefill 61.5 -> 49.5 us, fp8 41.4 -> 37.9 us)

This merge keeps r1_c3's structure and adds the two measured policies: the
SimtVF lane count follows the tile size (512 lanes once a tile holds 8192
elements, else 128), and the row-tile height follows the cache width (fp8 rows
are half the bytes, so its step is 4x smaller).  No branch is left behind: every
workload keeps the configuration its own round-1 measurement picked.

Content: kernel factory + tile policy; the golden and the layered precision
cases come from ``_pcw_tests``.
"""

from _pcw_kernel_dma import make_factory
from _pcw_tests import main as _run_tests

_SPLIT = False
_STORE = "auto"
_THREADS = 0  # auto: 512 lanes for wide tiles, 128 otherwise

_paged_cache_write_kernel = make_factory(_SPLIT, _STORE, _THREADS)
_DEFAULT_CONFIG = {"block_rows": 16, "block_cols": 128}


def pick_config(tokens, heads_kv, dim, physical_tokens, in_dtype, cache_dtype):
    """Row-tile height per cache width, whole row per DMA."""
    row_width = heads_kv * dim
    step = 128 if cache_dtype == "float8_e4m3fn" else 32
    rows = max(1, min(32, -(-tokens // step)))
    if cache_dtype == "float8_e4m3fn":
        rows = min(rows, 16)
    # a tail block that is mostly padding buys nothing: keep at least four
    # blocks' worth of rows per block at the smallest token counts
    rows = min(rows, max(1, tokens // 4))
    # the fp8 conversion pass needs two elements per lane
    if cache_dtype == "float8_e4m3fn":
        rows = max(rows, -(-256 // row_width))
    return rows, row_width


if __name__ == "__main__":
    _run_tests(globals())
