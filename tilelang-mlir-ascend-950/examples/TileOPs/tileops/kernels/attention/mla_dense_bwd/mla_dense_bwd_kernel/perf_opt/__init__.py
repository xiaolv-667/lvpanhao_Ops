"""Stage 4 (tuned) drop-in for the MLADenseBwdOp kernel factories.

``perf_opt/_bwd_kernel.py`` is the Stage 4 winner of the A5-rotation kernel
(md5 ``dbc75010b909da4468ce272c16e3f827``): byte-identical to the Stage 5 v4
main pass (``../../mla_dense_bwd_kernel/_bwd_kernel.py``) except three edits —

1. ``st_u``/``dp_u`` get one pair per statically expanded half body
   (``st_u0/dp_u0`` for half A, ``st_u1/dp_u1`` for half B), which removes the
   WAR edge ``fixpipe(k+1) -> vector read of EP(k)`` (+24,576 B UB = 60.9%,
   DESIGN.md 4.5 / 7.2 risk 16);
2. a dedicated L0A buffer ``a_p`` holds ``dV``'s A operand (``pT``) so its
   ``L1 -> L0A`` copy is not serialised behind the ``dK``/``dQ`` mads that share
   ``a_k`` (L0A 36,864 + 12,288 + 12,288 = 61,440 B = 93.75%);
3. ``_gradients`` therefore takes one extra argument (``a_p``).

Measured on the task platform (msopprof Task Duration, ``T_op`` = one call's six
launches): ds-v3-b2-8k 98,986.0 -> **94,169.7 us** (-4.87%, cube MFU 38.32% ->
40.28%), ds-v3-4k -3.17% / -3.25%, kimi-k2-8k -4.32%, ds-v2-lite-3k -0.66%; no
regression.  ``dk``/``dv`` stay **bit-identical** to v4, ``dq_acc`` only drifts
with the store-atomic arrival order.  Factory signature (including
``n_versions=2``), grid, the 4-D batch views and every contract are unchanged.

The delta and dQ-landing launches are re-exported from the Stage 3 modules
unchanged (they are 0.2-1.7% of a call):

    from .mla_dense_bwd_kernel.perf_opt._bwd_kernel import _bwd_kernel
    from .mla_dense_bwd_kernel.perf_opt._delta_kernel import _delta_kernel
    from .mla_dense_bwd_kernel.perf_opt._dq_land_kernel import _dq_land_kernel
"""

from ._bwd_kernel import _bwd_kernel
from ._delta_kernel import _delta_kernel
from ._dq_land_kernel import _dq_land_kernel

__all__ = ["_bwd_kernel", "_delta_kernel", "_dq_land_kernel"]
