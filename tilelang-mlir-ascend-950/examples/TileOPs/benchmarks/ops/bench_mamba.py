"""Performance benchmarks for the mamba (Mamba-2 / SSD) family (NPU).

Adapted from TileOPs ``benchmarks/ops/bench_mamba.py`` -- SSDChunkScanFwdOp
and SSDChunkCouplingFwdOp subsets only (the shared GPU file also covers
DaCumsum / SSDChunkState / SSDStatePassing / SSDDecode / Mamba2Fwd, which are
not migrated).

NPU adaptations (T3): ``tune`` parametrization dimension removed.  The GPU
``mamba_ssm`` Triton baselines (``_chunk_scan_fwd``, ``_bmm_chunk_fwd``) are
not available on NPU, so the benchmarks record the TileOPs kernel only.  The
GPU ``torch``/``torch-compile`` baselines are the correctness goldens
(``*_ref``), i.e. accuracy references -- per the add-npu-op skill (S6) they
are not profiled here; they stay in the correctness tests.

Multi-input note: ``SSDChunkScanFwdOp`` has 6 and ``SSDChunkCouplingFwdOp`` 2
tensor inputs, so ``workloads_to_params`` cannot be used (its single-input
workload contract raises ``KeyError`` for multi-input ops).  Custom
parametrization helpers (``_SSD_CHUNK_SCAN_FWD_BENCH_PARAMS``, ported from the
GPU bench_mamba.py, and ``_ssd_chunk_coupling_bench_params``, read from the
manifest workloads) drive the benchmarks instead.
"""

from typing import Optional

import pytest
import torch

from tileops.benchmark.benchmark_base import BenchmarkBase, BenchmarkReport
from tileops.manifest import load_workloads
from tileops.ops.mamba.ssd_chunk_coupling import SSDChunkCouplingFwdOp
from tileops.ops.mamba.ssd_chunk_scan import SSDChunkScanFwdOp
from tileops.testing.mamba2_reference import ssd_chunk_scan_fwd_ref
from tileops.workloads.mamba import SSDChunkCouplingFwdWorkload, SSDChunkScanFwdWorkload


class SSDChunkScanFwdBenchmark(BenchmarkBase[SSDChunkScanFwdWorkload]):
    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        b, c, L, h, p, n = (
            t.batch,
            t.num_chunks,
            t.chunk_len,
            t.n_heads,
            t.d_head,
            t.d_state,
        )
        # History path: C @ prev_states per token
        #   b * c * L * h matmuls of shape (1, n) x (n, p) -> 2*n*p FLOPs each
        history_flops = b * c * L * h * 2 * n * p
        # Intra-chunk path: lower-triangular lcb @ x
        #   b * c * h causal GEMMs of size (L, L) x (L, p) -> L*(L+1)/2 * 2*p FLOPs each
        diag_flops = b * c * h * (L * (L + 1) // 2) * 2 * p
        return float(history_flops + diag_flops)

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        b, c, L, h, p, n, g = (
            t.batch,
            t.num_chunks,
            t.chunk_len,
            t.n_heads,
            t.d_head,
            t.d_state,
            t.n_groups,
        )
        S = c * L
        elem = torch.tensor([], dtype=t.dtype).element_size()
        # Reads (input dtype): x + cb + C + prev_states + dt
        reads = (
            b * S * h * p  # x          [B, S, H, P]
            + b * c * g * L * L  # cb         [B, C, G, L, L]
            + b * S * g * n  # C          [B, S, G, N]
            + b * c * h * p * n  # prev_states [B, C, H, P, N]
            + b * h * c * L  # dt         [B, H, C, L]
        ) * elem
        # Reads (float32): dA_cumsum [B, H, C, L]
        reads += b * h * c * L * 4
        # Writes (float32): out [B, S, H, P]
        writes = b * S * h * p * 4
        return float(reads + writes)


# Benchmark parameters
#
# Model-to-shape mapping (Mamba-2 defaults):
#   n_heads = d_model / 32,  head_dim = 64,  d_state = 128,  chunk_len = 256
#   num_chunks = seq_len // chunk_len  (chunk_len=256: 2k->8, 4k->16, 32k->128)
#   n_groups = 1 (Mamba-2 standard)
#
#   130M -> n_heads=24   370M -> n_heads=32   780M -> n_heads=48
#   1.3B -> n_heads=64   2.7B -> n_heads=80
#
# Schema: (batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype)
_SSD_CHUNK_SCAN_FWD_BENCH_PARAMS = [
    # ── unit-scale ──
    pytest.param(1, 2, 64, 4, 64, 32, 1, torch.float16, id="b1-c2-L64-h4-p64-n32-fp16"),
    pytest.param(2, 4, 64, 8, 64, 64, 2, torch.float16, id="b2-c4-L64-h8-p64-n64-fp16"),
    pytest.param(1, 2, 128, 4, 128, 32, 1, torch.bfloat16, id="b1-c2-L128-h4-p128-n32-bf16"),
    pytest.param(2, 2, 64, 4, 64, 32, 2, torch.bfloat16, id="b2-c2-L64-h4-p64-n32-bf16"),
    # ── 130M (n_heads=24) ──
    pytest.param(1, 16, 256, 24, 64, 128, 1, torch.float16, id="latency-130m-4k"),
    pytest.param(8, 16, 256, 24, 64, 128, 1, torch.float16, id="serving-130m-4k"),
    pytest.param(4, 128, 256, 24, 64, 128, 1, torch.float16, id="longctx-130m-32k"),
    # ── 2.7B (n_heads=80) ──
    pytest.param(1, 16, 256, 80, 64, 128, 1, torch.float16, id="latency-2p7b-4k"),
    pytest.param(4, 16, 256, 80, 64, 128, 1, torch.float16, id="serving-2p7b-4k"),
    pytest.param(2, 128, 256, 80, 64, 128, 1, torch.float16, id="longctx-2p7b-32k"),
    pytest.param(4, 8, 256, 80, 64, 128, 1, torch.float16, id="throughput-2p7b-2k"),
]


@pytest.mark.parametrize(
    "batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype",
    _SSD_CHUNK_SCAN_FWD_BENCH_PARAMS,
)
def test_ssd_chunk_scan_fwd_bench(
    batch: int,
    num_chunks: int,
    chunk_len: int,
    n_heads: int,
    d_head: int,
    d_state: int,
    n_groups: int,
    dtype: torch.dtype,
) -> None:
    test = SSDChunkScanFwdWorkload(
        batch,
        num_chunks,
        chunk_len,
        n_heads,
        d_head,
        d_state,
        n_groups,
        dtype,
    )
    bm = SSDChunkScanFwdBenchmark(test)
    inputs = test.gen_inputs()  # x, cb, dA_cumsum, C, prev_states, dt

    # ── TileOPs kernel ──
    op = SSDChunkScanFwdOp()
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    # ── PyTorch reference baseline ──
    def torch_ref(x, cb, dA_cumsum, C, prev_states, dt):
        return ssd_chunk_scan_fwd_ref(x, cb, dA_cumsum, C, prev_states, dt, n_groups)

    result_bl = bm.profile(torch_ref, *inputs)
    # BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


class SSDChunkCouplingFwdBenchmark(BenchmarkBase[SSDChunkCouplingFwdWorkload]):
    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        # GPU manifest roofline (verbatim): Bsz * NC * G * chunk_size * chunk_size * N.
        # The causal mask is already folded in -- only the lower triangle is
        # priced, hence no factor 2 for the contraction.
        return float(
            t.batch * t.num_chunks * t.n_groups * t.chunk_len * t.chunk_len * t.d_state
        )

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        elem = torch.tensor([], dtype=t.dtype).element_size()
        S = t.num_chunks * t.chunk_len
        # Reads: C + B   [B, S, G, N] each (input dtype)
        reads = 2 * t.batch * S * t.n_groups * t.d_state * elem
        # Writes: cb     [B, NC, G, Q, Q] (input dtype)
        writes = t.batch * t.num_chunks * t.n_groups * t.chunk_len * t.chunk_len * elem
        return float(reads + writes)


def _ssd_chunk_coupling_bench_params(workloads: list[dict]) -> list:
    """Parametrize dims from the manifest workloads (multi-input op).

    Each row carries ``C_shape`` / ``B_shape`` / ``chunk_size`` / ``dtypes``;
    both operands share one shape, so the row collapses to
    ``(batch, num_chunks, chunk_len, n_groups, d_state, dtype)``.  The GPU
    bench is manifest-driven too (``bench.cases(SSDChunkCouplingFwdOp)``).
    """
    params = []
    for idx, w in enumerate(workloads):
        c_shape = tuple(w["C_shape"])
        if tuple(w["B_shape"]) != c_shape:
            raise ValueError(f"workload {w.get('label', w)!r}: C_shape and B_shape must match")
        batch, seq_len, n_groups, d_state = c_shape
        chunk_len = int(w["chunk_size"])
        if seq_len % chunk_len != 0:
            raise ValueError(
                f"workload {w.get('label', w)!r}: seq_len {seq_len} is not divisible by "
                f"chunk_size {chunk_len}"
            )
        num_chunks = seq_len // chunk_len
        label = w.get("label", "x".join(str(dim) for dim in c_shape))
        for dtype_name in w["dtypes"]:
            dtype = getattr(torch, dtype_name)
            marks = pytest.mark.smoke if idx == 0 else pytest.mark.full
            params.append(
                pytest.param(
                    batch,
                    num_chunks,
                    chunk_len,
                    n_groups,
                    d_state,
                    dtype,
                    id=f"{label}-{dtype_name}",
                    marks=marks,
                )
            )
    return params


_SSD_CHUNK_COUPLING_FWD_BENCH_PARAMS = _ssd_chunk_coupling_bench_params(
    load_workloads("SSDChunkCouplingFwdOp")
)


@pytest.mark.parametrize(
    "batch, num_chunks, chunk_len, n_groups, d_state, dtype",
    _SSD_CHUNK_COUPLING_FWD_BENCH_PARAMS,
)
def test_ssd_chunk_coupling_fwd_bench(
    batch: int,
    num_chunks: int,
    chunk_len: int,
    n_groups: int,
    d_state: int,
    dtype: torch.dtype,
) -> None:
    test = SSDChunkCouplingFwdWorkload(
        batch, num_chunks, chunk_len, n_groups, d_state, dtype
    )
    bm = SSDChunkCouplingFwdBenchmark(test)
    inputs = test.gen_inputs()  # C, B

    # ── TileOPs kernel ──
    op = SSDChunkCouplingFwdOp(chunk_len)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
