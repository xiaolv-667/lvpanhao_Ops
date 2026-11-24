"""Manifest-driven benchmark for MLADenseBwdOp (NPU).

Adapted from the ``test_mla_dense_bwd_bench`` function of TileOPs
``benchmarks/ops/bench_mla.py``.  The GPU file is shared with the MLA decode /
paged / varlen forward / varlen backward benchmarks (none of those ops is
migrated yet), so only the dense backward case is ported.

- Baselines: the GPU case compares against FA3
  (``flash_attn_interface._flash_attn_backward``) and, when FA3 is absent,
  against ``torch`` SDPA autograd at equal query/key lengths.  FA3 is not
  available on NPU, and the SDPA fallback is the GPU workaround for a
  query/key width (192) that differs from the value width (128) -- a path
  the NPU SDPA backend does not promise.  Neither is ported, so the
  benchmark records the TileOPs latency only (the same call as
  ``bench_paged_cache_write.py``, which drops the unavailable vLLM /
  FlashInfer baselines).
- ``o`` and ``lse`` come from the workload's float32 forward and are made
  once outside the timed call, in the layout FA3 takes, so no forward and no
  transpose is timed (GPU parity).
- T3/T4: the ``tune`` policy knob (``op.request_tune()``) and the autotune
  try/except are removed -- NPU uses heuristic config selection only.

Multi-input note: the op takes 6 tensors, so the single-input
``workloads_to_params`` contract does not apply.  ``_manifest_params`` reads
``load_workloads`` directly -- the multi-input pattern of
``bench_lerp_tensor.py`` -- over this op's geometry workload schema; the
first workload is the smoke case.
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import BenchmarkReport, ManifestBenchmark
from tileops.manifest import load_workloads
from tileops.ops.attention.mla_dense_bwd import MLADenseBwdOp
from tileops.workloads.attention import MLADenseBwdWorkload

_MLA_DENSE_BWD_OP = "MLADenseBwdOp"


def _manifest_params(workloads: list[dict]) -> list:
    """Params from the manifest's geometry workload schema.

    Each workload carries ``B``/``S_q``/``S_kv``/``H``/``D_qk``/``D_v`` plus
    ``is_causal`` (and optionally ``sm_scale``); every dtype in ``dtypes``
    becomes one benchmark case.
    """
    params = []
    for idx, workload in enumerate(workloads):
        label = workload.get("label", "manifest")
        marks = pytest.mark.smoke if idx == 0 else pytest.mark.full
        for dtype_name in workload["dtypes"]:
            dtype = getattr(torch, dtype_name)
            params.append(
                pytest.param(
                    workload["B"],
                    workload["S_q"],
                    workload["S_kv"],
                    workload["H"],
                    workload["D_qk"],
                    workload["D_v"],
                    workload.get("is_causal", False),
                    workload.get("sm_scale"),
                    dtype,
                    id=f"{label}-{dtype_name}",
                    marks=marks,
                )
            )
    return params


@pytest.mark.parametrize(
    "batch, seqlen_q, seqlen_kv, heads, dim_qk, dim_v, is_causal, sm_scale, dtype",
    _manifest_params(load_workloads(_MLA_DENSE_BWD_OP)),
)
def test_mla_dense_bwd_bench(
    batch: int,
    seqlen_q: int,
    seqlen_kv: int,
    heads: int,
    dim_qk: int,
    dim_v: int,
    is_causal: bool,
    sm_scale,
    dtype: torch.dtype,
) -> None:
    test = MLADenseBwdWorkload(
        batch, seqlen_q, seqlen_kv, heads, dim_qk, dim_v, dtype, is_causal, sm_scale
    )
    inputs = test.gen_inputs()

    op = MLADenseBwdOp(is_causal=is_causal, sm_scale=sm_scale)
    bm = ManifestBenchmark(_MLA_DENSE_BWD_OP, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
