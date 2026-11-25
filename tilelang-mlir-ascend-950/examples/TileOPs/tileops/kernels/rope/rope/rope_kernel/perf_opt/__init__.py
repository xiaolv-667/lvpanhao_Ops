"""Stage 4 tuned (perf_opt) drop-in for ``_rope_positions_kernel``.

``_rope_positions_kernel.py`` is the Stage 4 tuned kernel (identical bytes to
the task artifact ``examples/rope_perf/perf_opt/rope.py``, sha256
``0d34fc39a6686453628ff902c82521b00d965ab0f479761526391883273a0bff``).  It keeps
the Stage 3 factory contract -- ``_rope_positions_kernel(query_heads, key_heads,
head_size, rotary_dim, max_position, dtype, position_dtype, offset_dtype, neox,
rot_vec, tail_vec, row_vec)`` returns ``(units_per_block, staged)`` -> the
compiled program -- so the wrapper's switch block can import it in place of the
Stage 3 module without any other change.
"""

from ._rope_positions_kernel import _rope_positions_kernel

__all__ = ["_rope_positions_kernel"]
