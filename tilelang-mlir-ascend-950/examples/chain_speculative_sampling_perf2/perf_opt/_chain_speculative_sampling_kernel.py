"""Adoption alias: the module name the op wrapper's perf_opt switch imports.

The Round 2 tuned kernel lives in ``ChainSpeculativeSamplingFwdOp.py`` next to
this file (same Stage 3 factory signature ``(batch, num_draft, vocab) ->
callable``, same seven-argument call contract).  Adopting it is a copy of these
two files into ``…/chain_speculative_sampling_kernel/perf_opt/`` plus the
wrapper's switch-block flip::

    from .chain_speculative_sampling_kernel.perf_opt._chain_speculative_sampling_kernel \
        import _chain_speculative_sampling_kernel
"""

try:  # imported as part of the kernel package
    from . import ChainSpeculativeSamplingFwdOp as _core
except ImportError:  # imported as a plain module by the tuning helpers
    import ChainSpeculativeSamplingFwdOp as _core

_chain_speculative_sampling_kernel = _core._chain_speculative_sampling_kernel
main = _core.main

__all__ = ["_chain_speculative_sampling_kernel"]


def __getattr__(name):
    return getattr(_core, name)
