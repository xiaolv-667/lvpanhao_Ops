"""Stage 4 Round 2 tuned kernels (multi-core split) for chain_speculative_sampling.

The wrapper beside this package selects between the Stage 3 baseline
(``.._chain_speculative_sampling_kernel``) and these sources through its import
switch block.  ``ChainSpeculativeSamplingFwdOp`` holds the kernel;
``_chain_speculative_sampling_kernel`` is the module name the switch imports and
re-exports it, so both spellings resolve to the same factory and the Stage 3
factory signature ``(batch, num_draft, vocab) -> callable`` is unchanged.
"""

from ._chain_speculative_sampling_kernel import _chain_speculative_sampling_kernel

__all__ = [
    "_chain_speculative_sampling_kernel",
]
