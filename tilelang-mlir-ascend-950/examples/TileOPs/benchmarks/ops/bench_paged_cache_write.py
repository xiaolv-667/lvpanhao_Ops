"""Manifest-driven benchmark for PagedKVCacheWriteFwdOp (NPU).

Adapted from TileOPs ``benchmarks/ops/bench_paged_cache_write.py`` (the
GPU file covers only this op).

- The GPU vLLM / FlashInfer baselines are not ported: neither library is
  available on NPU, and the GPU "torch-ref" implementation is the accuracy
  oracle (the workload ``ref_program``, already kept in the correctness
  tests), not a performance baseline -- so it is omitted here entirely
  rather than recorded as a benchmark row.
- T3/T4: the ``tune`` policy knob and the autotune try/except are removed
  (NPU uses heuristic config selection only).

Multi-input note: the op takes 4 tensors plus the slot index (and the two
optional scales), so the single-input ``workloads_to_params`` contract
does not apply.  ``_manifest_params`` reads ``load_workloads`` directly --
the multi-input pattern of ``bench_lerp_tensor.py`` -- over this op's
``k_shape`` / ``cache_shape`` / ``dtype_combos`` workload schema; the
first workload is the smoke case.
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import BenchmarkReport, ManifestBenchmark
from tileops.manifest import load_workloads
from tileops.ops.attention.paged_cache_write import PagedKVCacheWriteFwdOp
from tileops.workloads.attention import PagedKVCacheWriteWorkload

_PAGED_WRITE_OP = "PagedKVCacheWriteFwdOp"


def _manifest_params(workloads: list[dict]) -> list:
    """Params from the manifest's multi-input workload schema.

    Each workload carries ``k_shape``, ``cache_shape`` and the (T, KV)
    ``dtype_combos``; every combination becomes one benchmark case.
    """
    params = []
    for idx, workload in enumerate(workloads):
        k_shape = tuple(workload["k_shape"])
        cache_shape = tuple(workload["cache_shape"])
        label = workload.get("label", "manifest")
        marks = pytest.mark.smoke if idx == 0 else pytest.mark.full
        for combo in workload["dtype_combos"]:
            dtype = getattr(torch, combo["T"])
            cache_dtype = getattr(torch, combo["KV"])
            params.append(
                pytest.param(
                    k_shape,
                    cache_shape,
                    dtype,
                    cache_dtype,
                    id=f"{label}-{combo['T']}-{combo['KV']}",
                    marks=marks,
                )
            )
    return params


@pytest.mark.parametrize(
    "k_shape, cache_shape, dtype, cache_dtype", _manifest_params(load_workloads(_PAGED_WRITE_OP))
)
def test_paged_cache_write_bench(
    k_shape: tuple[int, ...], cache_shape: tuple[int, ...], dtype: torch.dtype, cache_dtype
) -> None:
    tokens, heads_kv, dim = k_shape
    num_pages, page_size = cache_shape[0], cache_shape[1]
    test = PagedKVCacheWriteWorkload(
        tokens, heads_kv, dim, page_size, num_pages, dtype, cache_dtype
    )
    inputs = test.gen_inputs()

    # Eager geometry: the benchmark asks for the roofline of this case, and
    # the written-slot count is resolved from the slots after profiling.
    op = PagedKVCacheWriteFwdOp(
        heads_kv=heads_kv,
        dim=dim,
        num_pages=num_pages,
        page_size=page_size,
        dtype=dtype,
        cache_dtype=cache_dtype,
    )
    bm = ManifestBenchmark(_PAGED_WRITE_OP, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
