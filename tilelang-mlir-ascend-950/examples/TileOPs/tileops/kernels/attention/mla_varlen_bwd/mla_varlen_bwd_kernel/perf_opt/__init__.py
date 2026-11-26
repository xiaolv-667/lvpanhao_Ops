"""Stage 4 (tuned) drop-in for the MLAVarlenBwdOp kernel factories — A5 v5 generation.

``perf_opt/_bwd_kernel.py`` is the v5 Stage 4 winner
(`examples/mla_varlen_bwd_perf3/perf_opt/opt_log.md`): the A5 loop unchanged
(same two-half-body ``T.serial`` expansion, same lagged ``GB`` + drain, same
``_STAGES = 2`` contract, same tile/grid/slab), with three measured additions:

* ``st_u``/``dp_u`` are double-versioned per half body (UB +24,576 B), which
  removes the WAR that made half body B wait for half body A's epilogue;
* the static half-body rotation is V=3 (``*_s0/_s1/_s2``, L1 417,792 B = 79.7%)
  with a 3-way drain, keeping one more fill in flight;
* the six head-major ``lse``/``delta`` runs are issued at the iteration top.

Measured (msopprof kernel ``Task Duration``, ``T_op`` = one call's launches):
ds-v3-mixed 441.7 -> 402.4 ms (-8.9%), ds-v3-8x4k 139.0 -> 135.4 (-2.5%),
kimi-k2-4x8k flat.  ``dk``/``dv`` bitwise identical to the v5 baseline, ``dq_acc``
inside DESIGN.md 8.2; L0 29/29, L1 8/8.

The wrapper's ``baseline`` / ``perf_opt`` switch block moves **both** imports, so
that the launch-plan helpers and the compiled artifact keep one source of truth:

    from .mla_varlen_bwd_kernel.perf_opt._bwd_kernel import _bwd_kernel
    from .mla_varlen_bwd_kernel.perf_opt._delta_kernel import _delta_kernel
    from .mla_varlen_bwd_kernel.perf_opt._dq_land_kernel import _dq_land_kernel
    from .mla_varlen_bwd_kernel.perf_opt._bwd_kernel import (
        build_kernel, describe, factory_config, plan_launch,
    )

``block_n`` / ``head_slab`` / the grid assertions are unchanged, so the staged
plan and the artifact stay consistent (``assert_launch_consistency`` re-checks
both fields at build time): ``plan_launch(8, 32768, 128)`` is still
``tiles=349 slab=32 launches=4 max_grid=11168 idle=0``.
"""

from ._bwd_kernel import _bwd_kernel
from ._delta_kernel import _delta_kernel
from ._dq_land_kernel import _dq_land_kernel

__all__ = ["_bwd_kernel", "_delta_kernel", "_dq_land_kernel"]
