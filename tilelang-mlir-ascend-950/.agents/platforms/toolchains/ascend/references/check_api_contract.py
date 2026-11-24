"""REF-950-api-contract: front-end checks, no device execution.
Verified: TileLang ascend 0.1.15 / CANN 9.2 SDK, 2026-11-19.
"""

import inspect
import json
import tilelang.ascend.language as T
from tvm import tirx

checks = []


def rejects(name, action, exception):
    try:
        action()
    except exception as error:
        checks.append(
            {"case": name, "exception": type(error).__name__, "message": str(error)}
        )
    else:
        raise AssertionError(name)


rejects(
    "Kernel_threads_parameter",
    lambda: inspect.signature(T.Kernel).bind(1, threads=32),
    TypeError,
)
rejects("SimtVF_four_dimensions", lambda: T.SimtVF([1, 1, 1, 1]), ValueError)
rejects("SimtVF_zero_threads", lambda: T.SimtVF(0), ValueError)
rejects(
    "whole_copy_different_counts",
    lambda: T.copy(
        tirx.decl_buffer((8,), "float32"), tirx.decl_buffer((16,), "float32")
    ),
    ValueError,
)
rejects(
    "dual_copy_L0C_dtype_change",
    lambda: T.dual_copy(
        tirx.decl_buffer((16, 32), "float32", scope="shared.l0c"),
        tirx.decl_buffer((8, 32), "float16", scope="shared"),
    ),
    ValueError,
)
rejects("barrier_non_string", lambda: T.ascend_pipe_barrier(1), TypeError)
rejects("set_flag_non_string", lambda: T.ascend_set_flag(1, 0), TypeError)
rejects(
    "buffer_versions_zero",
    lambda: T.annotate_buffer_versions({tirx.decl_buffer((8,), "float32"): 0}),
    ValueError,
)
print(json.dumps(checks, indent=2))
