"""Stage 4 tuned (`perf_opt`) kernels for paged_cache_write.

Re-exports the drop-in tuned module integrated from
``examples/paged_cache_write_perf/perf_opt/paged_cache_write.py``.  The factory
signature is unchanged
(``_paged_cache_write_kernel(heads_kv, dim, physical_tokens, in_dtype, cache_dtype)``
returning a callable ``(rows, cols) -> compiled callable``); the module also
exports the tuned tile policy ``pick_config``, which the wrapper's
``forward`` resolves the per-call tile through.
"""

from ._paged_cache_write_kernel import _paged_cache_write_kernel, pick_config

__all__ = ["_paged_cache_write_kernel", "pick_config"]
