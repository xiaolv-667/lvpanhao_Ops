"""pytest plugin: run the published op tests against a ``perf_opt`` candidate.

``tests/ops/test_fused_moe_shared_expert.py`` builds ``SharedExpertMLPFwdOp()``
with its default kernel map, i.e. the Stage 3 package the wrapper still points
at.  The op exposes a ``kernel_map`` override, so this plugin swaps the kernel
class at session start -- the test file, the op and the wrapper are untouched.

Usage (from ``examples/TileOPs``)::

    PYTHONPATH=<perf_opt>/harness TILEOPS_PERF_OPT_IMPL=<perf_opt>/SharedExpertMLPFwdOp.py \
        python -m pytest tests/ops/test_fused_moe_shared_expert.py \
        -p perf_opt_plugin -q
"""

import importlib.util
import os
import sys


def _load(path):
    spec = importlib.util.spec_from_file_location("perf_opt_candidate", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["perf_opt_candidate"] = module
    spec.loader.exec_module(module)
    return module


def pytest_configure(config):
    path = os.environ.get("TILEOPS_PERF_OPT_IMPL")
    if not path:
        raise RuntimeError("TILEOPS_PERF_OPT_IMPL is not set")
    module = _load(os.path.abspath(path))
    from tileops.ops.moe.shared_expert_mlp import SharedExpertMLPFwdOp

    tuned = {"shared_expert_mlp": module.SharedExpertMLPKernel}
    SharedExpertMLPFwdOp.default_kernel_map = property(lambda self: dict(tuned))
    print(f"[perf_opt_plugin] {path} -> SharedExpertMLPFwdOp kernel map")
