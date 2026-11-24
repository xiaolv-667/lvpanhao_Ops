"""Manifest-driven benchmark for SharedExpertMLPFwdOp (NPU).

Adapted from the GPU TileOPs ``benchmarks/ops/bench_fused_moe.py``, which is
shared with ``FusedMoEFwdOp`` and ``FusedMoESharedExpertFwdOp``; only
``test_shared_expert_mlp_bench`` is ported.

Baselines:

- The GPU file compares against QuACK (``quack.gemm_act`` + ``quack.gemm``).
  QuACK is not available on NPU, so that baseline is dropped.
- The ``torch-cublas`` baseline -- the same expert built from
  ``F.linear`` + ``silu`` -- is kept: it is a native operator, runnable on the
  NPU's torch, and the honest lower bound the fused kernel must beat.

Multi-input note: the op takes three tensors (``hidden_states``,
``w_gate_up``, ``w_down``), so ``workloads_to_params`` cannot be used (its
single-input workload contract raises ``KeyError``).  ``_manifest_params``
reads the manifest workload rows instead and expands each row as
``(hidden_states_shape, w_gate_up_shape, w_down_shape, dtype)``.

T3/T4: no ``tune`` knob and no autotune try/except (the NPU op has neither).
"""

import pytest
import torch
import torch.nn.functional as F

from tileops.benchmark.benchmark_base import BenchmarkReport, ManifestBenchmark
from tileops.manifest import load_workloads
from tileops.ops.moe.shared_expert_mlp import SharedExpertMLPFwdOp
from tileops.workloads.moe import SharedExpertMLPWorkload

_SHARED_EXPERT_OP = "SharedExpertMLPFwdOp"


def _manifest_params(workloads: list[dict]) -> list:
    """Parametrize the manifest workload rows (multi-input op).

    Each row carries all three input shapes and one ``dtypes`` list; every
    dtype case becomes its own param, labelled ``{row label}-{dtype}``.
    """
    params = []
    for workload in workloads:
        hidden_shape = tuple(workload["hidden_states_shape"])
        gate_up_shape = tuple(workload["w_gate_up_shape"])
        down_shape = tuple(workload["w_down_shape"])
        label = workload.get("label", f"t{hidden_shape[0]}-h{hidden_shape[1]}")
        for dtype_name in workload["dtypes"]:
            params.append(
                pytest.param(
                    hidden_shape,
                    gate_up_shape,
                    down_shape,
                    getattr(torch, dtype_name),
                    id=f"{label}-{dtype_name}",
                )
            )
    return params


def _torch_shared_expert(
    hidden: torch.Tensor, w_gate_up: torch.Tensor, w_down: torch.Tensor
) -> torch.Tensor:
    """The same expert built from native torch ops (the GPU ``torch_fn`` baseline)."""
    gate, up = F.linear(hidden, w_gate_up).chunk(2, -1)
    return F.linear(F.silu(gate) * up, w_down)


@pytest.mark.parametrize(
    "hidden_shape, gate_up_shape, down_shape, dtype",
    _manifest_params(load_workloads(_SHARED_EXPERT_OP)),
)
def test_shared_expert_mlp_bench(
    hidden_shape: tuple[int, ...],
    gate_up_shape: tuple[int, ...],
    down_shape: tuple[int, ...],
    dtype: torch.dtype,
) -> None:
    """Time the shared expert against the native torch baseline on one workload row."""
    num_tokens, hidden_size = hidden_shape
    ffn_size = down_shape[1]
    test = SharedExpertMLPWorkload(num_tokens, hidden_size, ffn_size, dtype)
    inputs = test.gen_inputs()

    op = SharedExpertMLPFwdOp()
    bm = ManifestBenchmark(_SHARED_EXPERT_OP, op, test)
    # The op's roofline is read from the op, so it must be called first.
    result = bm.profile(op, *inputs)

    # One params dict for both tags, so the two rows of the report line up
    # under the same label.
    case = {
        "label": f"{num_tokens}x{hidden_size}x{ffn_size}",
        "dtype": dtype,
        "hidden_states_shape": list(hidden_shape),
        "w_gate_up_shape": list(gate_up_shape),
        "w_down_shape": list(down_shape),
    }
    BenchmarkReport.record(op, case, result, tag="tileops")

    result_baseline = bm.profile(_torch_shared_expert, *inputs)
    BenchmarkReport.record(_SHARED_EXPERT_OP, case, result_baseline, tag="torch")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
