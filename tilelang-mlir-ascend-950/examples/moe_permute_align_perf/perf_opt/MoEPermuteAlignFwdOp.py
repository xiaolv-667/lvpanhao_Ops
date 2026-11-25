# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Stage 4 tuned ``MoEPermuteAlignFwdOp`` kernels (artifact alias).

The tuned implementation lives in the sibling module
``moe_permute_align.py`` (the file ``statectl`` records as
``kernel_opt_py_path``); this module re-exports its three factories so the
deliverable is also importable by the operator name.  Both files are the same
program: nothing is defined here beyond the re-export.

Factories (the frozen wrapper's contract):

* ``_make_align_kernel(numel, num_experts, block_size)``
* ``_make_scatter_kernel(numel, num_experts, block_size)``
* ``_make_small_batch_kernel(numel, num_experts, block_size)``
"""

import importlib.util
import pathlib

_PATH = pathlib.Path(__file__).with_name("moe_permute_align.py")
_SPEC = importlib.util.spec_from_file_location("moe_permute_align_tuned", _PATH)
_IMPL = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_IMPL)

_make_align_kernel = _IMPL._make_align_kernel
_make_scatter_kernel = _IMPL._make_scatter_kernel
_make_small_batch_kernel = _IMPL._make_small_batch_kernel

# tuned defaults, exposed for the wrapper's perf_opt switch block
_THREADS = _IMPL._THREADS
_UB_PARTIAL_BUDGET = _IMPL._UB_PARTIAL_BUDGET
_BULK_MERGE_MIN_CORES = _IMPL._BULK_MERGE_MIN_CORES

__all__ = [
    "_make_align_kernel",
    "_make_scatter_kernel",
    "_make_small_batch_kernel",
    "_THREADS",
    "_UB_PARTIAL_BUDGET",
    "_BULK_MERGE_MIN_CORES",
]


if __name__ == "__main__":
    raise SystemExit(_IMPL.main())
