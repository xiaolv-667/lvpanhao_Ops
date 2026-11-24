"""Ascend sparse MLA (DeepSeek Sparse Attention) decode forward kernel.

One task owns 64 query heads of one ``(batch, query row)`` pair.  The AIC runs the
two GEMMs of every KV step (QK over ``dim + dim_tail``, PV over ``dim``) while the
two AIVs own 32 heads each: they turn the score tile into online-softmax
probabilities and rescale/accumulate the running PV numerator.

KV rows are consumed as *dense* blocks of ``KEY_TILE`` rows plus a per-column
membership mask built from ``indices``: a column whose KV row no selected index
names is floored in the softmax, so the dense contraction accumulates exactly the
sparse sum.  With ``topk == seq_len_kv`` -- the decode workload -- the dense form
performs no work the sparse form would skip; when ``topk < seq_len_kv`` it pays
``seq_len_kv / topk`` times the arithmetic, which is why the scalar gather kernel
below stays as the fallback for shapes the dense form does not serve.
"""

import functools
from typing import ClassVar, Optional

import tilelang
import tilelang.ascend.language as T
import torch
from tilelang.ascend.language import simd as S

from tileops.backend import register_kernel_type
from tileops.kernels.ascend.attention.nsa_common import calibrated_cube_blocks
from tileops.kernels.attention.call_spec import DSADecodeCall, SparseMLADecodeFwdInterface
from tileops.kernels.kernel_base import Kernel

__all__ = ["AscendDSADecodeKernel"]

# KV rows one pipelined step consumes; also the score tile's column count and the
# width of one deinterleaved SIMD score row. The Cube's PV operand is (dim,
# KEY_TILE) and the SIMD packing path writes one such row per store.
KEY_TILE = 128
# Query heads one task owns. The two AIVs split them, so each AIV keeps
# HEAD_TILE // 2 score/PV rows in its own UB.
HEAD_TILE = 64
# KV tile of the scalar fallback: small enough that one AI core holds the gather.
SCALAR_KEY_TILE = 32
# The most negative finite float32 used as the softmax floor. A score added to it
# cannot overflow to -inf, and exp(floor - floor) == 1 keeps an all-masked block
# from producing NaN.
SOFTMAX_FLOOR = -1.0e30
# Columns one finalize pass normalizes and stores; two SIMD vectors wide.
RESULT_TILE = 128
# Floats one SIMD load covers.
VECTOR_LANES = 64


def _make_cube_kernel(
    batch: int,
    seq_len: int,
    seq_len_kv: int,
    heads: int,
    dim: int,
    tail_dim: int,
    topk: int,
    dtype: str,
    scale: float,
    num_blocks: int,
    q_start_index_s: int,
    stride_kv: int,
    is_causal: bool,
):
    """Dense-block Cube+Vector kernel; one task is 64 heads of one query row."""
    dt = dim + tail_dim
    rows = HEAD_TILE // 2
    key_tile = KEY_TILE
    assert key_tile == 2 * VECTOR_LANES, "the score row is two SIMD vectors wide"
    blocks = seq_len_kv // key_tile
    head_tiles = heads // HEAD_TILE
    tasks = batch * seq_len * head_tiles
    steps = (tasks + num_blocks - 1) // num_blocks
    # One padding vector keeps the per-row state vectors inside the SIMD store width.
    state_width = max(rows, VECTOR_LANES)
    bias_len = ((seq_len_kv + VECTOR_LANES - 1) // VECTOR_LANES) * VECTOR_LANES
    dim_vectors = dim // VECTOR_LANES
    # The PV numerator is accumulated one half of `dim` at a time: two half-width
    # accumulators replace one full-width pair and free 64 KB of UB.
    half_vectors = dim_vectors // 2
    half_dim = dim // 2
    result_vectors = RESULT_TILE // VECTOR_LANES

    @T.macro
    def build_bias(indices, bias, b, s, hkv, limit):
        """Floor every column, then clear the columns a selected index names."""
        with T.SimtVF(threads=VECTOR_LANES):
            for column in T.Parallel(bias_len):
                bias[column] = T.float32(SOFTMAX_FLOOR)
        with T.SimtVF(threads=VECTOR_LANES):
            for slot in T.serial(topk):
                index = indices[b, s, hkv, slot]
                # An unused slot holds seq_len_kv, so clamp before addressing; the
                # invalid branch rewrites the floor it just read.
                target = T.max(0, T.min(index, seq_len_kv - 1))
                bias[target] = T.if_then_else(
                    (index >= 0) & (index <= limit), T.float32(0.0), bias[target]
                )

    @T.macro
    def clear_state(accumulator, maximum, length):
        with T.SimdVF(latency=142):
            full = S.pset(32, "PAT_ALL")
            zero = S.vdup(T.float32(0.0), "float32", full)
            floor = S.vdup(T.float32(SOFTMAX_FLOOR), "float32", full)
            S.vsts(maximum[0], floor, full, "NORM_B32")
            S.vsts(length[0], zero, full, "NORM_B32")
            for row in T.serial(rows):
                for chunk in T.serial(dim_vectors):
                    S.vsts(
                        accumulator[row, chunk * VECTOR_LANES],
                        zero,
                        full,
                        "NORM_B32",
                    )

    @T.macro
    def softmax_block(
        scores,
        bias_tile,
        maximum,
        old_maximum,
        alpha,
        block_sum,
        probabilities,
        length,
    ):
        """Score + mask -> online softmax; leaves probabilities as 16-bit rows."""
        with T.SimdVF(latency=735):
            full = S.pset(32, "PAT_ALL")
            one = S.pset(32, "PAT_VL1")
            zero = S.vdup(T.float32(0.0), "float32", full)
            S.vsts(old_maximum[0], S.vld(maximum[0]), full, "NORM_B32")
            S.mem_bar("VST_VLD")
            for row in T.serial(rows):
                score_low = S.vld(scores[row, 0])
                score_high = S.vld(scores[row, VECTOR_LANES])
                bias_low = S.vld(bias_tile[0])
                bias_high = S.vld(bias_tile[VECTOR_LANES])
                previous = S.vld(old_maximum[row], "BRC_B32")
                scaled_low = S.vadd(S.vmuls(score_low, scale, full), bias_low, full)
                scaled_high = S.vadd(S.vmuls(score_high, scale, full), bias_high, full)
                row_max = S.vmax(
                    previous,
                    S.vdupv(S.vcmax(S.vmax(scaled_low, scaled_high, full), full), full),
                    full,
                )
                S.vsts(maximum[row], row_max, one, "ONEPT_B32", extent=1)
                masked_low = S.vcmps(bias_low, T.float32(0.0), full, op="lt")
                masked_high = S.vcmps(bias_high, T.float32(0.0), full, op="lt")
                probability_low = S.vsel(
                    zero, S.vexpdif(scaled_low, row_max, full), masked_low
                )
                probability_high = S.vsel(
                    zero, S.vexpdif(scaled_high, row_max, full), masked_high
                )
                total = S.vcadd(
                    S.vadd(probability_low, probability_high, full), full
                )
                S.vsts(block_sum[row], total, one, "ONEPT_B32", extent=1)
                S.vsts(
                    probabilities[row, 0],
                    S.vcvt(probability_low, dtype, full, round="ROUND_R", sat=False),
                    full,
                    dist="PK_B32",
                    extent=VECTOR_LANES,
                )
                S.vsts(
                    probabilities[row, VECTOR_LANES],
                    S.vcvt(probability_high, dtype, full, round="ROUND_R", sat=False),
                    full,
                    dist="PK_B32",
                    extent=VECTOR_LANES,
                )
            S.mem_bar("VST_VLD")
            factors = S.vexpdif(S.vld(old_maximum[0]), S.vld(maximum[0]), full)
            S.vsts(alpha[0], factors, full, "NORM_B32")
            rescaled = S.vmul(S.vld(length[0]), factors, full)
            S.vsts(
                length[0],
                S.vadd(rescaled, S.vld(block_sum[0]), full),
                full,
                "NORM_B32",
            )

    @T.macro
    def accumulate_output(accumulator, partial, alpha, base):
        with T.SimdVF(latency=318):
            full = S.pset(32, "PAT_ALL")
            for row in T.serial(rows):
                factor = S.vld(alpha[row], "BRC_B32")
                for chunk in T.serial(half_vectors):
                    column = base + chunk * VECTOR_LANES
                    previous = S.vld(accumulator[row, column])
                    current = S.vld(partial[row, chunk * VECTOR_LANES])
                    S.vsts(
                        accumulator[row, column],
                        S.vadd(S.vmul(factor, previous, full), current, full),
                        full,
                        "NORM_B32",
                    )

    @T.macro
    def normalize_tile(accumulator, length, result, base):
        """Normalize one RESULT_TILE-wide column tile and cast it to the output dtype."""
        with T.SimdVF(latency=400):
            full = S.pset(32, "PAT_ALL")
            zero = S.vdup(T.float32(0.0), "float32", full)
            one = S.vdup(T.float32(1.0), "float32", full)
            for row in T.serial(rows):
                denominator = S.vld(length[row], "BRC_B32")
                positive = S.vcmps(denominator, T.float32(0.0), full, op="gt")
                safe = S.vsel(denominator, one, positive)
                for chunk in T.serial(result_vectors):
                    values = S.vdiv(
                        S.vld(accumulator[row, base + chunk * VECTOR_LANES]),
                        safe,
                        full,
                        precision="exact",
                    )
                    values = S.vsel(zero, values, positive)
                    converted = S.vcvt(values, dtype, full, sat=False)
                    S.vsts(
                        result[row, chunk * VECTOR_LANES],
                        converted,
                        full,
                        dist="PK_B32",
                        extent=VECTOR_LANES,
                    )

    @T.prim_func
    def main(
        q: T.Tensor((batch, seq_len, heads, dt), dtype),
        kv: T.Tensor((batch, seq_len_kv, 1, dt), dtype),
        indices: T.Tensor((batch, seq_len, 1, topk), "int32"),
        out: T.Tensor((batch, seq_len, heads, dim), dtype),
    ):
        with T.MixedKernel(num_blocks, sids=2) as (bx, sid):
            q_l1 = T.alloc_l1((HEAD_TILE, dt), dtype)
            k_l1 = T.alloc_l1((key_tile, dt), dtype)
            v_l1 = T.alloc_l1((dim, key_tile), dtype)
            p_l1 = T.alloc_l1((HEAD_TILE, key_tile), dtype)
            score_l0c = T.alloc_l0c((HEAD_TILE, key_tile), "float32")
            out_l0c = T.alloc_l0c((HEAD_TILE, half_dim), "float32")
            scores = T.alloc_shared((rows, key_tile), "float32")
            probabilities = T.alloc_shared((rows, key_tile), dtype)
            bias = T.alloc_shared((bias_len,), "float32")
            bias_tile = T.alloc_shared((key_tile,), "float32")
            partial = T.alloc_shared((rows, half_dim), "float32")
            accumulator = T.alloc_shared((rows, dim), "float32")
            result = T.alloc_shared((rows, RESULT_TILE), dtype)
            maximum = T.alloc_shared((state_width,), "float32")
            old_maximum = T.alloc_shared((state_width,), "float32")
            length = T.alloc_shared((state_width,), "float32")
            alpha = T.alloc_shared((state_width,), "float32")
            block_sum = T.alloc_shared((state_width,), "float32")
            # L1 holds one Q tile plus the KV step's operands; only the V tile --
            # the one the PV GEMM waits on longest -- is double buffered, because
            # two versions of all three operands exceed the 512 KB L1 and the
            # hardware faults on the overflow instead of the scheduler noticing.
            T.annotate_buffer_versions(
                {
                    q_l1: 1,
                    k_l1: 1,
                    v_l1: 2,
                    p_l1: 1,
                    score_l0c: 2,
                    out_l0c: 1,
                    scores: 2,
                    probabilities: 2,
                    partial: 2,
                }
            )
            for step in T.serial(steps):
                task = step * num_blocks + bx
                if task < tasks:
                    slab = task % head_tiles
                    rest = task // head_tiles
                    s = rest % seq_len
                    b = rest // seq_len
                    head_start = slab * HEAD_TILE
                    # Compressed-causal reach of this query row, clamped to the rows kv holds.
                    limit = (
                        T.min(
                            (q_start_index_s + s + 1 - stride_kv) // stride_kv,
                            seq_len_kv - 1,
                        )
                        if is_causal
                        else seq_len_kv - 1
                    )
                    T.copy(q[b, s, head_start : head_start + HEAD_TILE, :], q_l1)
                    build_bias(indices, bias, b, s, 0, limit)
                    clear_state(accumulator, maximum, length)
                    for block in T.Pipelined(
                        blocks,
                        num_stages=2,
                        annotations={"enable_offset": True},
                    ):
                        T.copy(kv[b, block * key_tile : (block + 1) * key_tile, 0, :], k_l1)
                        T.copy(
                            kv[b, block * key_tile : (block + 1) * key_tile, 0, 0:dim],
                            v_l1,
                            transpose=True,
                        )
                        T.copy(
                            bias[block * key_tile : (block + 1) * key_tile],
                            bias_tile,
                        )
                        T.gemm(q_l1, k_l1, score_l0c, transpose_B=True, clear_accum=True)
                        T.dual_copy(score_l0c, scores)
                        softmax_block(
                            scores,
                            bias_tile,
                            maximum,
                            old_maximum,
                            alpha,
                            block_sum,
                            probabilities,
                            length,
                        )
                        T.dual_copy(probabilities[0:rows, :], p_l1)
                        for half_idx in range(2):
                            T.gemm(
                                p_l1,
                                v_l1[half_idx * half_dim : (half_idx + 1) * half_dim, :],
                                out_l0c,
                                transpose_B=True,
                                clear_accum=True,
                            )
                            T.dual_copy(out_l0c, partial)
                            accumulate_output(accumulator, partial, alpha, half_idx * half_dim)
                    for tile in range(dim // RESULT_TILE):
                        normalize_tile(accumulator, length, result, tile * RESULT_TILE)
                        T.dual_copy(
                            result,
                            out[
                                b,
                                s,
                                head_start : head_start + HEAD_TILE,
                                tile * RESULT_TILE : (tile + 1) * RESULT_TILE,
                            ],
                        )

    return main


@functools.lru_cache(maxsize=64)
def _compile_cube_kernel(
    batch: int,
    seq_len: int,
    seq_len_kv: int,
    heads: int,
    dim: int,
    tail_dim: int,
    topk: int,
    dtype: str,
    scale: float,
    num_blocks: int,
    q_start_index_s: int,
    stride_kv: int,
    is_causal: bool,
):
    return tilelang.compile(
        _make_cube_kernel(
            batch,
            seq_len,
            seq_len_kv,
            heads,
            dim,
            tail_dim,
            topk,
            dtype,
            scale,
            num_blocks,
            q_start_index_s,
            stride_kv,
            is_causal,
        ),
        target="ascend",
        out_idx=-1,
    )


def _make_dsa_decode_kernel(
    batch: int,
    seq_len: int,
    seq_len_kv: int,
    heads: int,
    heads_kv: int,
    dim: int,
    tail_dim: int,
    topk: int,
    dtype: str,
    scale: float,
    num_blocks: int,
    q_start_index_s: int = 0,
    stride_kv: int = 1,
    is_causal: bool = True,
):
    """Scalar fallback: one query head row per AI core, KV rows gathered by index."""
    assert topk % SCALAR_KEY_TILE == 0, "topk must be a multiple of the key tile"
    assert num_blocks > 0
    dt = dim + tail_dim
    group = heads // heads_kv
    tiles = topk // SCALAR_KEY_TILE
    tasks = batch * seq_len * heads
    grid = min(num_blocks, tasks)
    steps = (tasks + grid - 1) // grid

    @T.prim_func
    def main(
        q: T.Tensor((batch, seq_len, heads, dt), dtype),
        kv: T.Tensor((batch, seq_len_kv, heads_kv, dt), dtype),
        indices: T.Tensor((batch, seq_len, heads_kv, topk), "int32"),
        out: T.Tensor((batch, seq_len, heads, dim), dtype),
    ):
        with T.Kernel(grid) as bx:
            qv = T.alloc_shared((dt,), dtype)
            kt = T.alloc_shared((SCALAR_KEY_TILE, dt), dtype)
            prev = T.alloc_shared((SCALAR_KEY_TILE,), "int32")
            acc = T.alloc_shared((dim,), "float32")
            mx = T.alloc_shared((1,), "float32")
            sm = T.alloc_shared((1,), "float32")

            for step in T.serial(steps):
                task = step * grid + bx
                if task < tasks:
                    h = task % heads
                    rest = task // heads
                    s = rest % seq_len
                    b = rest // seq_len
                    hkv = h // group
                    # Compressed-causal reach of this query row, clamped to the rows kv holds.
                    limit = (
                        T.min(
                            (q_start_index_s + s + 1 - stride_kv) // stride_kv,
                            seq_len_kv - 1,
                        )
                        if is_causal
                        else seq_len_kv - 1
                    )

                    T.copy(q[b, s, h, :], qv)
                    with T.SimtVF(threads=64):
                        for d in T.Parallel(dim):
                            acc[d] = 0.0
                        for i in T.Parallel(1):
                            mx[i] = -T.infinity("float32")
                            sm[i] = 0.0

                    for t in T.serial(tiles):
                        for j in T.serial(SCALAR_KEY_TILE):
                            idxv = indices[b, s, hkv, t * SCALAR_KEY_TILE + j]
                            prev[j] = idxv
                            row = T.max(0, T.min(idxv, seq_len_kv - 1))
                            T.copy(kv[b, row, hkv, :], kt[j, :])

                        with T.SimtVF(threads=64):
                            score = T.alloc_fragment((SCALAR_KEY_TILE,), "float32")
                            mx_new = T.alloc_fragment((1,), "float32")
                            rescale = T.alloc_fragment((1,), "float32")
                            tsum = T.alloc_fragment((1,), "float32")
                            mred = T.alloc_reducer((1,), "float32", op="max")
                            sred = T.alloc_reducer((1,), "float32", op="sum")
                            for j in T.Parallel(SCALAR_KEY_TILE):
                                score[j] = 0.0
                            for d in T.serial(dt):
                                for j in T.Parallel(SCALAR_KEY_TILE):
                                    score[j] = score[j] + T.cast(qv[d], "float32") * T.cast(
                                        kt[j, d], "float32"
                                    )
                            for j in T.Parallel(SCALAR_KEY_TILE):
                                score[j] = T.if_then_else(
                                    (prev[j] >= 0) & (prev[j] <= limit),
                                    score[j] * scale,
                                    -T.infinity("float32"),
                                )
                            T.reducer_init(mred)
                            for j in T.Parallel(SCALAR_KEY_TILE):
                                T.reducer_update(mred[0], score[j])
                            T.finalize_reducer(mred, mx_new)
                            for i in T.Parallel(1):
                                mx_new[i] = T.max(mx_new[i], mx[i])
                                rescale[i] = T.exp(mx[i] - mx_new[i])
                            for d in T.Parallel(dim):
                                acc[d] = acc[d] * rescale[0]
                            T.reducer_init(sred)
                            for j in T.Parallel(SCALAR_KEY_TILE):
                                score[j] = T.exp(score[j] - mx_new[0])
                                T.reducer_update(sred[0], score[j])
                            T.finalize_reducer(sred, tsum)
                            for j in T.serial(SCALAR_KEY_TILE):
                                for d in T.Parallel(dim):
                                    acc[d] = acc[d] + score[j] * T.cast(kt[j, d], "float32")
                            for i in T.Parallel(1):
                                sm[i] = sm[i] * rescale[i] + tsum[i]
                                mx[i] = mx_new[i]

                    with T.SimtVF(threads=64):
                        for d in T.Parallel(dim):
                            out[b, s, h, d] = T.cast(
                                T.if_then_else(sm[0] > 0, acc[d] / sm[0], 0.0), dtype
                            )

    return tilelang.compile(main, target="ascend", out_idx=-1)


class AscendDSADecodeKernel(Kernel, SparseMLADecodeFwdInterface):
    """Ascend sparse MLA forward: dense Cube+Vector blocks, scalar gather fallback."""

    devices: ClassVar[frozenset[str]] = frozenset({"npu"})

    @classmethod
    def refusal(cls, call: DSADecodeCall) -> Optional[str]:
        """The calls this implementation serves."""
        if call.dim != 512 or call.tail_dim not in (0, 64):
            return "requires dim 512 and tail_dim 0 or 64"
        if call.kv_group != 1:
            return "requires a single KV group"
        if call.topk % SCALAR_KEY_TILE != 0:
            return f"requires topk a multiple of {SCALAR_KEY_TILE}"
        if call.dtype not in (torch.float16, torch.bfloat16):
            return "requires float16 or bfloat16"
        if call.seq_len_kv <= 0 or call.batch <= 0 or call.seq_len <= 0:
            return "requires positive shapes"
        return None

    @classmethod
    def entry_for(cls, call: DSADecodeCall):
        scale = call.sm_scale if call.sm_scale is not None else (call.dim + call.tail_dim) ** -0.5
        return call, lambda: cls(
            call,
            scale,
            call.device.index if call.device is not None else None,
        )

    @staticmethod
    def _cube_serves(call: DSADecodeCall) -> bool:
        """Whether the dense Cube kernel covers this call's shapes."""
        return (
            call.heads % HEAD_TILE == 0
            and 0 < call.topk
            and call.seq_len_kv % KEY_TILE == 0
        )

    def __init__(self, call: DSADecodeCall, scale: float, device_index: Optional[int] = None) -> None:
        super().__init__(device_index=device_index)
        self.call = call
        self.scale = scale
        self._kernel = None

    def forward(self, q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        if self._kernel is None:
            dtype = "float16" if self.call.dtype == torch.float16 else "bfloat16"
            if self._cube_serves(self.call):
                self._kernel = _compile_cube_kernel(
                    self.call.batch,
                    self.call.seq_len,
                    self.call.seq_len_kv,
                    self.call.heads,
                    self.call.dim,
                    self.call.tail_dim,
                    self.call.topk,
                    dtype,
                    self.scale,
                    calibrated_cube_blocks(),
                    self.call.q_start_index_s,
                    self.call.kv_stride,
                    self.call.is_causal,
                )
            else:
                num_blocks = min(self.call.batch * self.call.seq_len * self.call.heads, 256)
                self._kernel = _make_dsa_decode_kernel(
                    self.call.batch,
                    self.call.seq_len,
                    self.call.seq_len_kv,
                    self.call.heads,
                    self.call.kv_group,
                    self.call.dim,
                    self.call.tail_dim,
                    self.call.topk,
                    dtype,
                    self.scale,
                    num_blocks,
                    self.call.q_start_index_s,
                    self.call.kv_stride,
                    self.call.is_causal,
                )
        return self._kernel(q, kv, indices)


register_kernel_type("DSADecodeWithKVCacheFwdOp", "ascend", AscendDSADecodeKernel)
