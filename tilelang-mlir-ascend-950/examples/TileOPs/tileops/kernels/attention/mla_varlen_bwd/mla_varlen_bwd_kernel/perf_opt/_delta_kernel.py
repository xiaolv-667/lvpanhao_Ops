"""Stage 4 drop-in for the delta launch: unchanged, re-exported.

``_delta_kernel`` is not touched by the Stage 4 tuning (its cost is 1.4 ms of a
200–690 ms call, see ``perf_opt/opt_log.md`` §1.1), so the tuned package
re-exports the Stage 3 module instead of duplicating it.
"""

from .._delta_kernel import _delta_kernel

__all__ = ["_delta_kernel"]
