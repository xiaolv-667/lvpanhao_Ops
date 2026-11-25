"""Deliverable alias for the Stage 4 tuned kernel.

The canonical tuned file is ``perf_opt/chain_speculative_sampling.py`` (the
task's ``kernel_opt_py_path``, and the file every record in
``perf_records.jsonl`` was measured on).  This module re-exports it under the
operator's name, so ``perf_opt/ChainSpeculativeSamplingFwdOp.py`` is importable
and runnable (``--level L0|L1|L2|boundary|all``, ``--case <workload>``); the two
files cannot drift because only the canonical one holds the kernel.
"""

try:  # imported as part of the kernel package
    from . import chain_speculative_sampling as _core
except ImportError:  # imported as a plain module (tuning helpers, --level entry)
    import chain_speculative_sampling as _core

_chain_speculative_sampling_kernel = _core._chain_speculative_sampling_kernel
_KNOBS = _core._KNOBS
golden_chain_speculative_sampling = _core.golden_chain_speculative_sampling
golden_philox_uniform = _core.golden_philox_uniform
golden_draw = _core.golden_draw
verify_chain_outputs = _core.verify_chain_outputs
assert_chain_distribution = _core.assert_chain_distribution
run_L0 = _core.run_L0
run_L1 = _core.run_L1
run_L2 = _core.run_L2
run_boundary = _core.run_boundary
main = _core.main

__all__ = [
    "_chain_speculative_sampling_kernel",
    "golden_chain_speculative_sampling",
    "verify_chain_outputs",
    "assert_chain_distribution",
    "run_L0",
    "run_L1",
    "run_L2",
    "run_boundary",
]


def __getattr__(name):
    """Everything else (private helpers used by the tuning scripts) comes from the core."""
    return getattr(_core, name)


if __name__ == "__main__":
    raise SystemExit(main())
