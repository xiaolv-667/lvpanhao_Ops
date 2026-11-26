"""Implementation loader + workload table for the Stage 4 capture harness.

Two implementation kinds are supported so the baseline (Stage 3 / TileOPs
package) and any ``perf_opt`` candidate are measured through the exact same
child process, inputs and launch loop:

* ``baseline`` -- ``tileops.kernels.moe.shared_expert_mlp.shared_expert_mlp.
  SharedExpertMLPKernel`` from the (read-only) migration repo.
* ``<path.py>`` -- a ``perf_opt`` candidate module exposing
  ``build_kernel(num_tokens, hidden_size, ffn_size, dtype_str, config)`` that
  returns a callable ``(hidden, w_gate_up, w_down) -> out``.

The workload table mirrors the cases ``benchmarks/ops/bench_fused_moe.py::
test_shared_expert_mlp_bench`` expands for ``SharedExpertMLPFwdOp`` (the
manifest rows of ``src/tileops/manifest/spec/moe.yaml``).  Nothing else is
enumerated.
"""

import importlib.util
import os
import sys

TILEOPS_DIR = "/workspace/tilelang-mlir-ascend-950/examples/TileOPs"

# label -> (T, H, S, dtype)
WORKLOADS = {
    "kimi-k2-t1": (1, 7168, 18432, "bfloat16"),
    "kimi-k2-t512": (512, 7168, 18432, "bfloat16"),
    "ds-v3-t32-bf16": (32, 7168, 2048, "bfloat16"),
    "ds-v3-t32-fp16": (32, 7168, 2048, "float16"),
    "ds-v3-t4096": (4096, 7168, 2048, "bfloat16"),
    "glm45-t32": (32, 5120, 1536, "bfloat16"),
}


def _ensure_tileops_on_path():
    if TILEOPS_DIR not in sys.path:
        sys.path.insert(0, TILEOPS_DIR)


def load_module(path):
    """Import a standalone python file by path."""
    name = "cand_" + os.path.basename(path).replace(".py", "")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build_impl(impl, workload, config=None):
    """Return ``(callable(hidden, w_gate_up, w_down), info)`` for *workload*."""
    T, H, S, dtype_str = WORKLOADS[workload]
    if impl == "baseline":
        _ensure_tileops_on_path()
        from tileops.kernels.moe.shared_expert_mlp.shared_expert_mlp import (
            SharedExpertMLPKernel,
        )

        kernel = SharedExpertMLPKernel(T, H, S, getattr(__import__("torch"), dtype_str))
        if config:
            kernel.init_config(config)
        return (lambda a, b, c: kernel(a, b, c)), {"impl": "baseline", "config": dict(kernel.config)}

    module = load_module(impl)
    kernel = module.build_kernel(T, H, S, dtype_str, config or {})
    return kernel, {"impl": impl, "config": dict(getattr(kernel, "resolved_config", {}))}
