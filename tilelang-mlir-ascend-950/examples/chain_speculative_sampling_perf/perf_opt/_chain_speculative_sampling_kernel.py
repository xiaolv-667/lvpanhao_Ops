"""Drop-in module name the op wrapper's perf_opt import switch expects.

The tuned kernel lives in ``chain_speculative_sampling.py`` next to this file
(same Stage 3 factory signature ``(batch, num_draft, vocab) -> callable``); this
module only re-exports it under the name the wrapper imports::

    from .chain_speculative_sampling_kernel.perf_opt._chain_speculative_sampling_kernel \
        import _chain_speculative_sampling_kernel
"""

try:  # imported as part of the kernel package
    from . import chain_speculative_sampling as _core
except ImportError:  # imported as a plain module by the tuning helpers
    import chain_speculative_sampling as _core

_chain_speculative_sampling_kernel = _core._chain_speculative_sampling_kernel
main = _core.main

__all__ = ["_chain_speculative_sampling_kernel"]


def __getattr__(name):
    return getattr(_core, name)
