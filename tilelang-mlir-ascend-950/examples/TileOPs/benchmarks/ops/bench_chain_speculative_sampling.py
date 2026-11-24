"""Benchmark for the chain speculative sampling op (NPU).

Adapted from TileOPs ``benchmarks/ops/bench_chain_speculative_sampling.py``.
Workload shapes come from the ops manifest; roofline FLOP and byte counts are
the op's ``eval_roofline()`` formula (kept device-agnostic).

Multi-input note: ``ChainSpeculativeSamplingFwdOp`` has five tensor inputs, so
``workloads_to_params`` cannot be used (its single-input workload contract
raises ``KeyError`` for multi-input ops).  ``_bench_params`` reads the manifest
workload rows instead and collapses each to ``(batch, num_draft, vocab)`` -- the
row's three ``_shape`` keys name the same dims.

Baselines: the GPU file measures a ``torch.compile`` reference, FlashInfer's
``sampling.chain_speculative_sampling`` and vLLM's rejection sampler.  None of
them is available on NPU, and the torch reference is the correctness golden --
per the add-npu-op skill (S6) an accuracy reference is not profiled, and its
``profile``/``record`` calls are omitted rather than left commented out.  The
benchmark records the TileOPs kernel only.
"""

import pytest
import torch

from tileops.benchmark.benchmark_base import BenchmarkBase, BenchmarkReport
from tileops.manifest import load_workloads
from tileops.ops.sampling.chain_speculative_sampling import ChainSpeculativeSamplingFwdOp
from tileops.workloads.sampling import ChainSpeculativeSamplingFwdWorkload

_OP = "ChainSpeculativeSamplingFwdOp"


class ChainSpeculativeSamplingFwdBenchmark(
    BenchmarkBase[ChainSpeculativeSamplingFwdWorkload]
):
    """Latency / TFLOPS / bandwidth of one call, against the manifest roofline."""

    def calculate_flops(self):
        t = self.workload
        b, n, v = t.batch, t.num_draft, t.vocab
        # GPU manifest roofline (verbatim): the cheaper of the two chain extremes.
        return float(b * min(2 * n + 3 * v + 1, 5 * v + 3))

    def calculate_memory(self):
        t = self.workload
        b, n, v = t.batch, t.num_draft, t.vocab
        elem = torch.tensor([], dtype=t.dtype).element_size()
        index = torch.int32.itemsize
        # GPU manifest roofline (verbatim): the tensors every call touches once
        # (seed, offset, output_token_ids, num_accepted) plus the per-row
        # traffic of the cheaper extreme.
        fixed = 2 * 8 + index * b * (n + 1) + index * b
        chain = min(3 * index * n + elem * v, index + 2 * elem * v)
        return float(fixed + b * chain)


def _bench_params(workloads: list[dict]) -> list:
    """Parametrize ``(batch, num_draft, vocab)`` from the manifest workloads.

    Each row declares its inputs' shapes and the probability element type the
    signature fixes (``float32``); the dims are read off ``draft_probs_shape``
    and the other two are checked against it.  The first row is the smoke case.
    """
    params = []
    for idx, w in enumerate(workloads):
        draft = tuple(w["draft_probs_shape"])
        ids = tuple(w["draft_token_ids_shape"])
        target = tuple(w["target_probs_shape"])
        batch, num_draft, vocab = draft
        if ids != (batch, num_draft):
            raise ValueError(f"workload {w.get('label', w)!r}: draft_token_ids_shape {ids}")
        if target != (batch, num_draft + 1, vocab):
            raise ValueError(f"workload {w.get('label', w)!r}: target_probs_shape {target}")
        label = w.get("label", "x".join(str(dim) for dim in draft))
        for dtype_name in w["dtypes"]:
            if dtype_name != "float32":
                raise ValueError(
                    f"workload {label!r}: the op's probability rows are float32, got {dtype_name}"
                )
            marks = pytest.mark.smoke if idx == 0 else pytest.mark.full
            params.append(
                pytest.param(
                    batch,
                    num_draft,
                    vocab,
                    id=f"{label}-{dtype_name}",
                    marks=marks,
                )
            )
    return params


_BENCH_PARAMS = _bench_params(load_workloads(_OP))


@pytest.mark.parametrize("batch, num_draft, vocab", _BENCH_PARAMS)
def test_chain_speculative_sampling_bench(
    batch: int, num_draft: int, vocab: int
) -> None:
    test = ChainSpeculativeSamplingFwdWorkload(batch, num_draft, vocab)
    inputs = test.gen_inputs()  # draft_probs, draft_token_ids, target_probs, seed, offset

    bm = ChainSpeculativeSamplingFwdBenchmark(test)
    op = ChainSpeculativeSamplingFwdOp()
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
