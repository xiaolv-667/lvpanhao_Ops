# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""paged_cache_write Stage 4 (round 2) variant m1: manual sync on the fp8 store path.

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
_THREADS = 0  # auto: 1024 lanes (fp8, wide tile) / 512 / 128
_SYNC = "manual"

_paged_cache_write_kernel = make_factory(_SPLIT, _STORE, _THREADS, _SYNC)
_DEFAULT_CONFIG = {"block_rows": 16, "block_cols": 128}


def pick_config(tokens, heads_kv, dim, physical_tokens, in_dtype, cache_dtype):
    """Row-tile height, whole row per DMA (round-2 policy).

    Both builds tile ``tokens / 32`` rows per block; the quantised build allows
    a taller tile (48 vs 32) because its stores no longer pay the per-row MTE3
    drain, so a block amortises its fixed cost over more rows (measured: fp8
    4096 tokens rows 16/32/48/64 -> 20.7 / 20.2 / 19.5 / 23.5 us; fp8-fp16
    1024 tokens rows 8/16/32/48 -> 10.3 / 8.5 / 7.6 / 9.5 us).
    """
    row_width = heads_kv * dim
    quantized = cache_dtype == "float8_e4m3fn"
    rows = max(1, -(-tokens // 32))
    rows = min(rows, 48 if quantized else 32)
    rows = min(rows, max(1, tokens // 4))
    if quantized:
        # the conversion pass needs two elements per lane
        rows = max(rows, -(-256 // row_width))
    return rows, row_width


if __name__ == "__main__":
    _run_tests(globals())
