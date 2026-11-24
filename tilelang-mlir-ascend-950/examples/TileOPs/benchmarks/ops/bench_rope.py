"""Manifest-driven benchmark for RoPEFwdOp (NPU).

Adapted from TileOPs ``benchmarks/ops/bench_rope.py`` -- ``test_rope_bench``'s
``RoPEFwdOp`` case only (the shared GPU file also benchmarks
RoPELlama31FwdOp / YaRNFwdOp / LongRoPEFwdOp / RoPECosSinFwdOp, which are not
migrated).

NPU adaptations:

- T3: no ``tune`` parametrization dimension (autotune is removed on NPU).
- The GPU baselines (vLLM's rotary kernel, ``torch.compile``'d reference) are
  dropped: vLLM is not available on the NPU toolchain, and the GPU file's
  ``torch-ref`` is an accuracy-only reference (``rope_reference`` over
  independently computed cos/sin tables) -- it stays in
  ``tests/ops/test_rope.py`` and is not profiled or recorded here.

Multi-input note: ``RoPEFwdOp`` has 3 tensor inputs (positions / query / key,
the key nullable), so ``workloads_to_params`` cannot be used (its single-input
workload contract raises ``KeyError`` for multi-input ops).  ``_to_params``
reads the manifest workloads itself (the same pattern as
``benchmarks/ops/bench_ada_layer_norm.py``).
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import BenchmarkReport, ManifestBenchmark
from tileops.manifest import load_workloads
from tileops.ops.rope.rope import RoPEFwdOp
from tileops.workloads.rope import RoPEFwdWorkload

_ROPE_FWD_OP_NAME = "RoPEFwdOp"

_POSITION_DTYPES = {"int32": torch.int32, "int64": torch.int64}


def _to_params(workloads):
    params = []
    for w in workloads:
        key_shape = w.get("key_shape")
        position_dtype = _POSITION_DTYPES[w.get("position_dtype", "int64")]
        for dtype_str in w["dtypes"]:
            dtype = getattr(torch, dtype_str)
            params.append(
                pytest.param(
                    tuple(w["positions_shape"]),
                    tuple(w["query_shape"]),
                    None if key_shape is None else tuple(key_shape),
                    w["head_size"],
                    w["rotary_dim"],
                    w["max_position_embeddings"],
                    w["base"],
                    w["is_neox_style"],
                    dtype,
                    position_dtype,
                    id=f"{w.get('label', 'rope')}-{dtype_str}",
                )
            )
    return params


@pytest.mark.parametrize(
    "positions_shape, query_shape, key_shape, head_size, rotary_dim, "
    "max_position_embeddings, base, is_neox_style, dtype, position_dtype",
    _to_params(load_workloads(_ROPE_FWD_OP_NAME)),
)
def test_rope_fwd_bench(
    positions_shape: tuple,
    query_shape: tuple,
    key_shape: tuple | None,
    head_size: int,
    rotary_dim: int,
    max_position_embeddings: int,
    base: float,
    is_neox_style: bool,
    dtype: torch.dtype,
    position_dtype: torch.dtype,
) -> None:
    test = RoPEFwdWorkload(
        positions_shape,
        query_shape,
        key_shape,
        head_size,
        rotary_dim,
        max_position_embeddings,
        base,
        is_neox_style,
        dtype,
        position_dtype,
    )
    inputs = test.gen_inputs()  # positions, query, key (key may be None)

    op = RoPEFwdOp(
        head_size=head_size,
        rotary_dim=rotary_dim,
        max_position_embeddings=max_position_embeddings,
        base=base,
        is_neox_style=is_neox_style,
        dtype=dtype,
    )
    bm = ManifestBenchmark(_ROPE_FWD_OP_NAME, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
