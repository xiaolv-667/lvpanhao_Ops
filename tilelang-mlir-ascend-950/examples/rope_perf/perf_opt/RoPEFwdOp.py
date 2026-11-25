"""Alias module for the Stage 4 tuned RoPE forward kernel.

The tuned, self-contained implementation lives in ``rope.py`` next to this file
(it is the artifact the task state tracks as ``kernel_opt_py_path``).  This alias
only re-exports it under the operator's name so that
``perf_opt/RoPEFwdOp.py`` resolves to the same factory:

    from perf_opt.RoPEFwdOp import _rope_positions_kernel

Same factory signature as the Stage 3 module
(``_rope_positions_kernel(query_heads, key_heads, head_size, rotary_dim,
max_position, dtype, position_dtype, offset_dtype, neox, rot_vec, tail_vec,
row_vec)`` -> ``(units_per_block, staged)`` -> compiled program), so it is a
drop-in for ``rope_kernel/_rope_positions_kernel.py``.  Run the layered
precision suite through the alias too:

    python RoPEFwdOp.py --level L0
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from rope import (  # noqa: E402,F401  -- re-export the tuned module
    _TAB_VEC,
    _TILE_ELEMS,
    _STEP_BIG,
    _STEP_SMALL,
    _SUB_DIV,
    _THREADS,
    _UNITS_PER_BLOCK,
    _build,
    _launch,
    _rope_positions_kernel,
    golden_rope_positions,
    main,
    rope_tables,
)

__all__ = [
    "_rope_positions_kernel",
    "golden_rope_positions",
    "rope_tables",
    "_build",
    "_launch",
    "_UNITS_PER_BLOCK",
]

if __name__ == "__main__":
    sys.exit(main())
