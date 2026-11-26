# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""paged_cache_write Stage 4 (round 2) variant m2: m1 with 1024 SimtVF lanes.

Single main change against the round-1 winner (`paged_cache_write_perf/perf_opt`):
the quantised build compiles with ``tl.enable_auto_schedule=False`` and states the
two dependencies it actually has (MTE2->V before the conversion reads the staged
tile, V->MTE3 before the stores read the converted tile) plus one per-block
PIPE_MTE3 drain.  The auto-scheduler used to drain the store pipe before *every*
UB->GM row copy (a same-pipe dependency it cannot disprove), which made the fp8
case store-issue bound: 8192 scattered 512 B rows at ~0.16-0.25 us of core time
each.  Distinct non-negative slots are the op's contract, so two row stores never
target the same row and need no ordering between them.

The 16-bit build is untouched (auto-schedule, lane store): it already runs at
~1.4 TB/s, i.e. at the HBM ceiling.

Content: kernel factory + tile policy; the golden and the layered precision
cases come from ``_pcw_tests``.
"""

from _pcw_kernel_dma import make_factory
from _pcw_tests import main as _run_tests

_SPLIT = False
_STORE = "auto"
_THREADS = 1024
_SYNC = "manual"

_paged_cache_write_kernel = make_factory(_SPLIT, _STORE, _THREADS, _SYNC)
_DEFAULT_CONFIG = {"block_rows": 16, "block_cols": 128}


def pick_config(tokens, heads_kv, dim, physical_tokens, in_dtype, cache_dtype):
    """Row-tile height per cache width, whole row per DMA (round-1 policy)."""
    row_width = heads_kv * dim
    step = 128 if cache_dtype == "float8_e4m3fn" else 32
    rows = max(1, min(32, -(-tokens // step)))
    if cache_dtype == "float8_e4m3fn":
        rows = min(rows, 16)
    rows = min(rows, max(1, tokens // 4))
    if cache_dtype == "float8_e4m3fn":
        rows = max(rows, -(-256 // row_width))
    return rows, row_width


if __name__ == "__main__":
    _run_tests(globals())
