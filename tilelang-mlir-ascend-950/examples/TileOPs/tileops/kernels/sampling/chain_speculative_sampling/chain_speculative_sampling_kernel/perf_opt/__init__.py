"""Stage 4 tuned kernels for chain_speculative_sampling (perf_opt drop-in sources).

Built by the Stage 4 optimizer; the wrapper beside this package selects between
the Stage 3 baseline (``.._chain_speculative_sampling_kernel``) and these sources
through its import switch block.  Each module keeps the Stage 3 factory
signature, so the switch is a pure import swap.
"""

from ._chain_speculative_sampling_kernel import _chain_speculative_sampling_kernel

__all__ = [
    "_chain_speculative_sampling_kernel",
]
