"""Stage-4 (perf_opt) kernel sources for the ``engram_decode`` migration.

Drop-in replacement for the two Stage-3 kernel modules in the parent package.  A
decode step is **two launches**, so the tuned sources ship as one module that
exposes the same three factories/configs the wrapper's dispatch reads:

* ``ALIGNMENT``             -- host row-padding granularity (256, unchanged);
* ``_engram_project_kernel``-- launch 1 factory, same ``(batch, d_mem, d_padded,
  max_conv_len, dtype) -> _func(block_m, block_n, block_k, num_stages)`` shape;
* ``_engram_step_kernel``   -- launch 2 factory, same ``(batch, d, d_padded,
  max_conv_len, conv_kernel_size, dilation, eps, dtype) -> _func(threads)`` shape;
* ``project_default_config``/``step_default_config`` -- the launch configs, so the
  wrapper's ``perf_opt`` branch references factories and configs in pairs.

Baseline (Stage 3) stays importable from the parent package
(``from .engram_decode_kernel import ...``) for A/B and rollback; the wrapper's
``baseline`` branch of the import switch block is a pure source swap.

Measured effect of this swap (msopprof kernel-only Task Duration, T_op = the two
launches summed): serving-b8 9.466 -> 6.385 us (-32.5%); serving-b4 / serving-b1 /
decode-partial-cache / decode-s1 emit **byte-identical device code** to the
baseline, so they are unchanged by construction.  See
``examples/engram_decode_perf/perf_opt/opt_log.md``.
"""

from .EngramDecodeFwdOp import (
    ALIGNMENT,
    _engram_project_kernel,
    _engram_step_kernel,
    project_default_config,
    step_default_config,
)

__all__ = [
    "ALIGNMENT",
    "_engram_project_kernel",
    "_engram_step_kernel",
    "project_default_config",
    "step_default_config",
]
