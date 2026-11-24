"""Performance benchmarks for the Engram ops (NPU).

Adapted from the GPU TileOPs ``benchmarks/ops/bench_engram.py`` —
EngramGateConvFwdOp, EngramGateConvBwdOp and EngramDecodeFwdOp subsets.

The GPU file is manifest-driven (``bench.cases(<Op>)``) and compares against a
torch baseline: ``torch-ref`` plus a ``torch-compile`` variant for the forward,
and the autograd reference for the backward.  All of them are the correctness
goldens (``engram_gate_conv_fwd_torch`` / ``ref_engram_gate_conv_bwd``), i.e.
accuracy references, so per the add-npu-op skill (S6) they stay in the
correctness tests and are not profiled here; the benchmarks record the TileOPs
kernel only.

NPU adaptations (T3): ``tune`` parametrization removed (the GPU file forced
``_TUNE = True`` — autotuning is a bench-run policy, not a workload property).

Multi-input note: every op here has more than one tensor input (6 for the
forward, 12 for the backward, 8 for the decode step), so ``workloads_to_params``
cannot be used (its single-input workload contract raises ``KeyError`` for
multi-input ops).  The ``_*_bench_params`` helpers read the manifest workloads
and collapse each row to the op's constructor plus workload arguments.

Decode note: one decode step is **two** device launches (the projection + cache
shift, then the norms/gate/conv/SiLU step).  The benchmark times the op call
itself, so the recorded latency covers the complete step rather than either
launch alone; both launches stay part of the measurement.

Roofline: FLOP and byte counts are the manifest formulas, which each op's
``eval_roofline()`` reproduces.
"""

from typing import Optional

import pytest
import torch

from tileops.benchmark.benchmark_base import BenchmarkBase, BenchmarkReport
from tileops.manifest import load_workloads
from tileops.ops.sequence_modeling.engram_decode import EngramDecodeFwdOp
from tileops.ops.sequence_modeling.engram_gate_conv import EngramGateConvFwdOp
from tileops.ops.sequence_modeling.engram_gate_conv_bwd import EngramGateConvBwdOp
from tileops.workloads.sequence_modeling import (
    EngramDecodeWorkload,
    EngramGateConvBwdWorkload,
    EngramGateConvFwdWorkload,
)


class EngramGateConvFwdBenchmark(BenchmarkBase[EngramGateConvFwdWorkload]):
    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        # GPU manifest roofline (verbatim): per position three RMSNorms, the gate
        # dot and sigmoid, vhat, SiLU and the residual (21d + 14), plus the causal
        # width-4 depthwise convolution over the taps inside the input (2d each).
        taps_in_input = (
            t.seq_len + max(t.seq_len - 1, 0) + max(t.seq_len - 2, 0) + max(t.seq_len - 3, 0)
        )
        return float(t.M * (t.seq_len * (21 * t.d + 14) + 2 * t.d * taps_in_input))

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        M, seq_len, d = t.M, t.seq_len, t.d
        elem = torch.tensor([], dtype=t.dtype).element_size()
        # Reads: H, k, v [M, seq_len, d] + rms_w_h, rms_w_v [d] + conv_w [4, d].
        reads = 3 * M * seq_len * d * elem + 2 * d * elem + 4 * d * elem
        # Writes: Y, vhat [M, seq_len, d] in the input dtype.
        writes = 2 * M * seq_len * d * elem
        # Writes: alpha, rrms_h, rrms_k, rrms_v [M, seq_len] fp32.
        writes += 4 * M * seq_len * 4
        return float(reads + writes)


def _engram_gate_conv_fwd_bench_params(workloads: list[dict]) -> list:
    """Parametrize dims from the manifest workloads (multi-input op).

    Each row carries ``H_shape`` / ``k_shape`` / ``v_shape`` (all
    ``[M, seq_len, d]``), ``rms_w_h_shape`` / ``rms_w_v_shape`` (``[d]``),
    ``conv_w_shape`` (``[4, d]``), ``eps`` and ``dtypes``.
    """
    params = []
    for idx, w in enumerate(workloads):
        h_shape = tuple(w["H_shape"])
        if len(h_shape) != 3:
            raise ValueError(f"workload {w.get('label', w)!r}: H_shape must be [M, seq_len, d]")
        M, seq_len, d = h_shape
        for name in ("k_shape", "v_shape"):
            if tuple(w[name]) != h_shape:
                raise ValueError(
                    f"workload {w.get('label', w)!r}: {name} must match H_shape {h_shape}"
                )
        for name in ("rms_w_h_shape", "rms_w_v_shape"):
            if tuple(w[name]) != (d,):
                raise ValueError(f"workload {w.get('label', w)!r}: {name} must be [{d}]")
        if tuple(w["conv_w_shape"]) != (4, d):
            raise ValueError(f"workload {w.get('label', w)!r}: conv_w_shape must be [4, {d}]")

        eps = float(w.get("eps", 1e-6))
        label = w.get("label", "x".join(str(dim) for dim in h_shape))
        for dtype_name in w["dtypes"]:
            dtype = getattr(torch, dtype_name)
            marks = pytest.mark.smoke if idx == 0 else pytest.mark.full
            params.append(
                pytest.param(
                    M,
                    seq_len,
                    d,
                    eps,
                    dtype,
                    id=f"{label}-{dtype_name}",
                    marks=marks,
                )
            )
    return params


_ENGRAM_GATE_CONV_FWD_BENCH_PARAMS = _engram_gate_conv_fwd_bench_params(
    load_workloads("EngramGateConvFwdOp")
)


@pytest.mark.parametrize("M, seq_len, d, eps, dtype", _ENGRAM_GATE_CONV_FWD_BENCH_PARAMS)
def test_engram_gate_conv_fwd_bench(
    M: int,
    seq_len: int,
    d: int,
    eps: float,
    dtype: torch.dtype,
) -> None:
    test = EngramGateConvFwdWorkload(M, seq_len, d, dtype, eps)
    bm = EngramGateConvFwdBenchmark(test)
    inputs = test.gen_inputs()  # H, k, v, rms_w_h, rms_w_v, conv_w

    # ── TileOPs kernel ──
    op = EngramGateConvFwdOp(eps)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")


class EngramGateConvBwdBenchmark(BenchmarkBase[EngramGateConvBwdWorkload]):
    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        # GPU manifest roofline (verbatim): per position the normalized-value
        # recomputes, SiLU, RMSNorm, the gate and dot backward (45d + 10), plus
        # three convolution passes (recompute, weight and input gradients) over
        # the taps inside the input (2d each).
        taps_in_input = (
            t.seq_len + max(t.seq_len - 1, 0) + max(t.seq_len - 2, 0) + max(t.seq_len - 3, 0)
        )
        return float(t.M * (t.seq_len * (45 * t.d + 10) + 6 * t.d * taps_in_input))

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        M, seq_len, d = t.M, t.seq_len, t.d
        elem = torch.tensor([], dtype=t.dtype).element_size()
        # Reads: dY, H, k, v, vhat [M, seq_len, d] + rms_w_h, rms_w_v [d]
        # + conv_w [4, d].
        reads = 5 * M * seq_len * d * elem + 2 * d * elem + 4 * d * elem
        # Reads: alpha, rrms_h, rrms_k, rrms_v [M, seq_len] fp32.
        reads += 4 * M * seq_len * 4
        # Writes: dH, dk, dv [M, seq_len, d] in the input dtype.
        writes = 3 * M * seq_len * d * elem
        # Writes: drms_w_h, drms_w_v [d] and dconv_w [4, d], all fp32.
        writes += 2 * d * 4 + 4 * d * 4
        return float(reads + writes)


def _engram_gate_conv_bwd_bench_params(workloads: list[dict]) -> list:
    """Parametrize dims from the manifest workloads (multi-input op).

    Each row carries ``dY_shape`` / ``H_shape`` / ``k_shape`` / ``v_shape`` /
    ``vhat_shape`` (all ``[M, seq_len, d]``), ``rms_w_h_shape`` /
    ``rms_w_v_shape`` (``[d]``), ``conv_w_shape`` (``[4, d]``), the four fp32
    ``[M, seq_len]`` saved-intermediate shapes, ``eps`` and ``dtypes``.
    """
    params = []
    for idx, w in enumerate(workloads):
        dy_shape = tuple(w["dY_shape"])
        if len(dy_shape) != 3:
            raise ValueError(f"workload {w.get('label', w)!r}: dY_shape must be [M, seq_len, d]")
        M, seq_len, d = dy_shape
        for name in ("H_shape", "k_shape", "v_shape", "vhat_shape"):
            if tuple(w[name]) != dy_shape:
                raise ValueError(
                    f"workload {w.get('label', w)!r}: {name} must match dY_shape {dy_shape}"
                )
        for name in ("rms_w_h_shape", "rms_w_v_shape"):
            if tuple(w[name]) != (d,):
                raise ValueError(f"workload {w.get('label', w)!r}: {name} must be [{d}]")
        if tuple(w["conv_w_shape"]) != (4, d):
            raise ValueError(f"workload {w.get('label', w)!r}: conv_w_shape must be [4, {d}]")
        for name in ("alpha_shape", "rrms_h_shape", "rrms_k_shape", "rrms_v_shape"):
            if tuple(w[name]) != (M, seq_len):
                raise ValueError(
                    f"workload {w.get('label', w)!r}: {name} must be [{M}, {seq_len}]"
                )

        eps = float(w.get("eps", 1e-6))
        label = w.get("label", "x".join(str(dim) for dim in dy_shape))
        for dtype_name in w["dtypes"]:
            dtype = getattr(torch, dtype_name)
            marks = pytest.mark.smoke if idx == 0 else pytest.mark.full
            params.append(
                pytest.param(
                    M,
                    seq_len,
                    d,
                    eps,
                    dtype,
                    id=f"{label}-{dtype_name}",
                    marks=marks,
                )
            )
    return params


_ENGRAM_GATE_CONV_BWD_BENCH_PARAMS = _engram_gate_conv_bwd_bench_params(
    load_workloads("EngramGateConvBwdOp")
)


@pytest.mark.parametrize("M, seq_len, d, eps, dtype", _ENGRAM_GATE_CONV_BWD_BENCH_PARAMS)
def test_engram_gate_conv_bwd_bench(
    M: int,
    seq_len: int,
    d: int,
    eps: float,
    dtype: torch.dtype,
) -> None:
    test = EngramGateConvBwdWorkload(M, seq_len, d, dtype, eps)
    bm = EngramGateConvBwdBenchmark(test)
    inputs = test.gen_inputs()  # dY, H, k, v, rms_w_h, rms_w_v, conv_w, vhat, alpha, ...

    # ── TileOPs kernel ──
    op = EngramGateConvBwdOp(eps)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")



class EngramDecodeBenchmark(BenchmarkBase[EngramDecodeWorkload]):
    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        # GPU manifest roofline (verbatim): the two projections (2 per multiply-add
        # each), then per batch row the RMSNorm/gate/SiLU arithmetic (16d) and the
        # conv taps (2d each) plus a flat 20 per row.
        return float(
            4 * t.batch * t.d_mem * t.d
            + t.batch * (16 * t.d + t.conv_kernel_size * 2 * t.d)
            + 20 * t.batch
        )

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        batch, d_mem, d = t.batch, t.d_mem, t.d
        taps, conv_len, max_conv_len = t.conv_kernel_size, t.conv_len, t.max_conv_len
        elem = torch.tensor([], dtype=t.dtype).element_size()
        # Reads: e_t [batch, d_mem] + h_t [batch, d] + conv_state [batch, L, d]
        # + W_K, W_V [d_mem, d] + rms_w_h, rms_w_v [d] + conv_w [taps, d].
        reads = (
            batch * d_mem * elem
            + batch * d * elem
            + batch * conv_len * d * elem
            + 2 * d_mem * d * elem
            + 2 * d * elem
            + taps * d * elem
        )
        # Writes: y_t [batch, d] + new_conv_state [batch, max_conv_len, d].
        writes = batch * d * elem + batch * max_conv_len * d * elem
        return float(reads + writes)


def _engram_decode_bench_params(workloads: list[dict]) -> list:
    """Parametrize the decode geometry from the manifest workloads (multi-input op).

    Each row carries ``e_t_shape`` (``[batch, d_mem]``), ``h_t_shape``
    (``[batch, d]``), ``conv_state_shape`` (``[batch, L, d]``), ``W_K_shape`` /
    ``W_V_shape`` (``[d_mem, d]``), ``rms_w_h_shape`` / ``rms_w_v_shape`` (``[d]``),
    ``conv_w_shape`` (``[conv_kernel_size, d]``), the op params
    (``max_conv_len``, ``conv_kernel_size``, ``dilation``, ``eps``) and ``dtypes``.
    """
    params = []
    for idx, w in enumerate(workloads):
        e_t_shape = tuple(w["e_t_shape"])
        h_t_shape = tuple(w["h_t_shape"])
        conv_state_shape = tuple(w["conv_state_shape"])
        if len(e_t_shape) != 2 or len(h_t_shape) != 2 or len(conv_state_shape) != 3:
            raise ValueError(
                f"workload {w.get('label', w)!r}: expected e_t/h_t 2-D and conv_state 3-D"
            )
        batch, d_mem = e_t_shape
        if h_t_shape[0] != batch:
            raise ValueError(f"workload {w.get('label', w)!r}: h_t_shape must be [{batch}, d]")
        d = h_t_shape[1]
        if conv_state_shape[:1] != (batch,) or conv_state_shape[2] != d:
            raise ValueError(
                f"workload {w.get('label', w)!r}: conv_state_shape must be [{batch}, L, {d}]"
            )
        conv_len = conv_state_shape[1]
        for name in ("W_K_shape", "W_V_shape"):
            if tuple(w[name]) != (d_mem, d):
                raise ValueError(f"workload {w.get('label', w)!r}: {name} must be [{d_mem}, {d}]")
        for name in ("rms_w_h_shape", "rms_w_v_shape"):
            if tuple(w[name]) != (d,):
                raise ValueError(f"workload {w.get('label', w)!r}: {name} must be [{d}]")

        max_conv_len = int(w["max_conv_len"])
        conv_kernel_size = int(w["conv_kernel_size"])
        dilation = int(w["dilation"])
        if tuple(w["conv_w_shape"]) != (conv_kernel_size, d):
            raise ValueError(
                f"workload {w.get('label', w)!r}: conv_w_shape must be [{conv_kernel_size}, {d}]"
            )
        if max_conv_len < dilation * (conv_kernel_size - 1):
            raise ValueError(
                f"workload {w.get('label', w)!r}: max_conv_len must be >= "
                f"dilation * (conv_kernel_size - 1)"
            )
        if not 0 <= conv_len <= max_conv_len:
            raise ValueError(
                f"workload {w.get('label', w)!r}: conv_state length must be <= max_conv_len"
            )

        eps = float(w.get("eps", 1e-6))
        label = w.get("label", f"{batch}x{d_mem}x{d}")
        for dtype_name in w["dtypes"]:
            dtype = getattr(torch, dtype_name)
            marks = pytest.mark.smoke if idx == 0 else pytest.mark.full
            params.append(
                pytest.param(
                    batch,
                    d_mem,
                    d,
                    max_conv_len,
                    conv_kernel_size,
                    dilation,
                    conv_len,
                    eps,
                    dtype,
                    id=f"{label}-{dtype_name}",
                    marks=marks,
                )
            )
    return params


_ENGRAM_DECODE_BENCH_PARAMS = _engram_decode_bench_params(load_workloads("EngramDecodeFwdOp"))


@pytest.mark.parametrize(
    "batch, d_mem, d, max_conv_len, conv_kernel_size, dilation, conv_len, eps, dtype",
    _ENGRAM_DECODE_BENCH_PARAMS,
)
def test_engram_decode_bench(
    batch: int,
    d_mem: int,
    d: int,
    max_conv_len: int,
    conv_kernel_size: int,
    dilation: int,
    conv_len: int,
    eps: float,
    dtype: torch.dtype,
) -> None:
    test = EngramDecodeWorkload(
        batch,
        d_mem,
        d,
        max_conv_len,
        conv_kernel_size,
        dilation,
        dtype,
        eps,
        conv_len,
    )
    bm = EngramDecodeBenchmark(test)
    inputs = test.gen_inputs()  # e_t, h_t, conv_state, W_K, W_V, rms_w_h, rms_w_v, conv_w

    # ── TileOPs kernel (both launches of the step are inside this call) ──
    op = EngramDecodeFwdOp(max_conv_len, conv_kernel_size, dilation, eps)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
