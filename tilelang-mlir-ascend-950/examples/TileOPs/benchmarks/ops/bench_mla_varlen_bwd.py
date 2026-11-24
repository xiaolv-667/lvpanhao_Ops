"""Manifest-driven benchmark for MLAVarlenBwdOp (NPU).

Adapted from the ``test_mla_varlen_bwd_bench`` function of TileOPs
``benchmarks/ops/bench_mla.py``.  The GPU file is shared with the MLA decode /
paged / varlen forward / dense backward benchmarks (the dense backward is
migrated separately, see ``bench_mla_dense_bwd.py``), so only the packed
backward case is ported.

- Baselines: the GPU case compares against FA3
  (``flash_attn_interface._flash_attn_backward``, available only when
  ``optional_baseline`` finds it).  FA3 is not available on NPU, and the
  reference is the accuracy-only float32 torch backward in
  ``MLAVarlenBwdWorkload.ref_program``, which the skill keeps out of the
  benchmark.  So the benchmark records the TileOPs latency only (the same call
  as ``bench_mla_dense_bwd.py`` and ``bench_paged_cache_write.py``, which drop
  the unavailable FA3 / vLLM / FlashInfer baselines).
- ``o`` and ``lse`` come from the workload's float32 forward and are made once
  outside the timed call, in the head-major layout FA3 takes, so no forward and
  no transpose is timed (GPU parity).
- T3/T4: the ``tune`` policy knob (``op.request_tune()``) and the autotune
  try/except are removed -- NPU uses heuristic config selection only.

Multi-input note: the op takes 8 tensors (6 numeric + both offset vectors), so
the single-input ``workloads_to_params`` contract does not apply.
``_manifest_params`` reads ``load_workloads`` directly -- the multi-input
pattern of ``bench_lerp_tensor.py`` -- over this op's packed geometry workload
schema (``q_lens`` / ``kv_lens`` + head geometry); the first workload is the
smoke case.

The packed geometry in the manifest is deliberately extreme on ``ds-v3-mixed``
(512 .. 16384 rows in one call): the per-request tile enumeration is what
balances the 56 vector / 28 cube cores there, so the benchmark keeps that case
as the load-balance reference.
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import BenchmarkReport, ManifestBenchmark
from tileops.manifest import load_workloads
from tileops.ops.attention.mla_varlen_bwd import MLAVarlenBwdOp
from tileops.workloads.attention import MLAVarlenBwdWorkload

_MLA_VARLEN_BWD_OP = "MLAVarlenBwdOp"


def _manifest_params(workloads: list[dict]) -> list:
    """Params from the manifest's packed-geometry workload schema.

    Each workload carries ``q_lens``/``kv_lens`` (the packed request lengths,
    which also fix ``T_q``/``T_kv``), ``H``/``D_qk``/``D_v`` plus ``is_causal``
    (and optionally ``sm_scale``); every dtype in ``dtypes`` becomes one
    benchmark case.
    """
    params = []
    for idx, workload in enumerate(workloads):
        label = workload.get("label", "manifest")
        marks = pytest.mark.smoke if idx == 0 else pytest.mark.full
        for dtype_name in workload["dtypes"]:
            dtype = getattr(torch, dtype_name)
            params.append(
                pytest.param(
                    list(workload["q_lens"]),
                    list(workload["kv_lens"]),
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
    "q_lens, kv_lens, heads, dim_qk, dim_v, is_causal, sm_scale, dtype",
    _manifest_params(load_workloads(_MLA_VARLEN_BWD_OP)),
)
def test_mla_varlen_bwd_bench(
    q_lens: list,
    kv_lens: list,
    heads: int,
    dim_qk: int,
    dim_v: int,
    is_causal: bool,
    sm_scale,
    dtype: torch.dtype,
) -> None:
    test = MLAVarlenBwdWorkload(q_lens, kv_lens, heads, dim_qk, dim_v, dtype, is_causal, sm_scale)
    inputs = test.gen_inputs()

    op = MLAVarlenBwdOp(is_causal=is_causal, sm_scale=sm_scale)
    bm = ManifestBenchmark(_MLA_VARLEN_BWD_OP, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
