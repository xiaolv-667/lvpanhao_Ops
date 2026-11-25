"""Stage-4 (perf_opt) kernel sources for the ``engram_gate_conv`` migration.

Drop-in replacements for the Stage-3 kernel modules in the parent package: same
factory name, same signature, same six-output contract, so the wrapper's
``perf_opt`` branch of the import switch block is a pure source swap.  Baseline
(Stage 3) stays importable from the parent package for A/B and rollback.
"""

from ._engram_gate_conv_fwd_kernel import _engram_gate_conv_fwd_kernel

__all__ = [
    "_engram_gate_conv_fwd_kernel",
]
