"""Stage-4 (perf_opt) kernel sources for the ``engram_gate_conv_bwd`` migration.

Drop-in replacement for the Stage-3 kernel module in the parent package: same
factory name (``_engram_gate_conv_bwd_kernel``), same signature
``(M, seq_len, d, dtype)``, same six-output contract and the same
``lru_cache`` shape specialisation, so the wrapper's ``perf_opt`` branch of the
import switch block is a pure source swap.  Baseline (Stage 3) stays importable
from the parent package for A/B and rollback.

Source of truth: ``examples/engram_gate_conv_bwd_perf/perf_opt/engram_gate_conv_bwd.py``
(byte-identical, sha256 65c3133c38ec86ba26db72e90d620c2c4cac8c079b04e1a211fde0b11668ec81);
the tuning evidence is in that directory's ``opt_log.md``.
"""

from ._engram_gate_conv_bwd_kernel import _engram_gate_conv_bwd_kernel

__all__ = [
    "_engram_gate_conv_bwd_kernel",
]
