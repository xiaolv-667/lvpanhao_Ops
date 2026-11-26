"""Stage 4 drop-in for the dQ landing launch: unchanged, re-exported.

``_dq_land_kernel`` is not touched by the Stage 4 tuning (3.3 ms of a 200–690 ms
call, see ``perf_opt/opt_log.md`` §1.1), so the tuned package re-exports the
Stage 3 module instead of duplicating it.
"""

from .._dq_land_kernel import _dq_land_kernel

__all__ = ["_dq_land_kernel"]
