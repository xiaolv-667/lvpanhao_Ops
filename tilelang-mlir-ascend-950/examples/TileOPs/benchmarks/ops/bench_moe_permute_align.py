"""Manifest-driven benchmark for MoEPermuteAlignFwdOp (NPU).

Adapted from TileOPs ``benchmarks/ops/bench_moe_permute_align.py`` (the GPU
file covers only this op).

- The GPU Triton baseline (an adaptation of SGLang's 4-stage
  ``moe_align_block_size``) and the vLLM baseline
  (``vllm._custom_ops.moe_align_block_size``) are not ported: Triton and vLLM
  are not available on NPU, and neither is a runnable performance baseline
  here.  The GPU file's ``case.inputs`` / ``case.arguments`` parametrization is
  replaced by the NPU manifest's workload rows, which carry the same shapes
  and op params.
- T3/T4: no ``tune`` policy knob and no autotune try/except (NPU uses
  heuristic config selection only).

Single-input note: the op takes one tensor, ``topk_ids``, so
``workloads_to_params`` applies; ``num_experts`` and ``block_size`` come from
each workload row as the op params.
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import (
    BenchmarkReport,
    ManifestBenchmark,
    workloads_to_params,
)
from tileops.ops.moe.permute_align import MoEPermuteAlignFwdOp
from tileops.workloads.moe import MoEPermuteAlignWorkload

_PERMUTE_ALIGN_OP = "MoEPermuteAlignFwdOp"


@pytest.mark.parametrize(
    "shape, dtype, op_params",
    workloads_to_params(_PERMUTE_ALIGN_OP, include_extra=True),
)
def test_permute_align_bench(shape: tuple, dtype: torch.dtype, op_params: dict) -> None:
    tokens, top_k = shape
    test = MoEPermuteAlignWorkload(
        tokens, top_k, op_params["num_experts"], op_params["block_size"], dtype
    )
    inputs = test.gen_inputs()

    op = MoEPermuteAlignFwdOp(**op_params)
    bm = ManifestBenchmark(_PERMUTE_ALIGN_OP, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
