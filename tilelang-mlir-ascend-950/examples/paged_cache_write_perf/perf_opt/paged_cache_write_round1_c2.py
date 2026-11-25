# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""paged_cache_write Stage 4 variant r1_c2: DMA load + vector (per-lane) store.

Single main change against r1_c1: the scattered store leaves the DMA path.  The
scheduler drains the MTE3 pipe before every UB->GM copy, so a row-sized DMA
store costs a fixed ~0.3 us of core time; a ``T.Parallel`` store from UB to the
scattered GM row is issued by the vector pipe with no such drain.

Content: kernel factory + tile policy; the golden and the layered precision
cases come from ``_pcw_tests``.
"""

from _pcw_kernel_dma import make_factory
from _pcw_tests import main as _run_tests

_SPLIT = False
_STORE = "lane"
_THREADS = 128

_paged_cache_write_kernel = make_factory(_SPLIT, _STORE, _THREADS)
_DEFAULT_CONFIG = {"block_rows": 16, "block_cols": 128}


def pick_config(tokens, heads_kv, dim, physical_tokens, in_dtype, cache_dtype):
    """One block per row tile, the whole row staged."""
    return 8, heads_kv * dim


if __name__ == "__main__":
    _run_tests(globals())
