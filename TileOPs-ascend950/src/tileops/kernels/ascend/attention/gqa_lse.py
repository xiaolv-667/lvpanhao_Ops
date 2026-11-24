"""Ascend dense GQA forward that also reports each query row's log-sum-exp.

One Ascend AI core processes one query-head row: it walks the KV sequence in
tiles under the call's causal mask and window, runs an online softmax in FP32,
and writes both the attention output and the natural-log log-sum-exp of the
row's logits.  This is the correctness-first implementation of the op; tiling
for the Cube engine is left to the optimization stage.
"""

from typing import ClassVar, Optional

import tilelang
import tilelang.ascend.language as T
import torch

from tileops.backend import register_kernel_type
from tileops.kernels.attention.call_spec import (
    AttentionCall,
    GQADenseWithLSEFwdInterface,
)
from tileops.kernels.kernel_base import Kernel

__all__ = ["AscendGQADenseWithLSEKernel"]


def _make_gqa_lse_kernel(
    batch: int,
    seq_len_q: int,
    seq_len_kv: int,
    heads: int,
    heads_kv: int,
    dim: int,
    dtype: str,
    scale: float,
    softcap: float,
    window_size_left: int,
    window_size_right: int,
    is_causal: bool,
    num_blocks: int,
):
    """One query-head row per AI core; the KV sequence is walked in dense tiles."""
    # KV rows per tile. The shared K/V tile's row is ``dim`` elements, so 32B alignment
    # requires ``dim`` to be a multiple of 16 for a 16-bit dtype, which ``refusal`` names.
    # Measured on the llama-8b-short shape, 64- and 128-row tiles ran about 3x slower, so
    # the tile stays at the width where the row-wise dot product reads best.
    key_tile = 32
    group = heads // heads_kv
    tiles = (seq_len_kv + key_tile - 1) // key_tile
    tasks = batch * seq_len_q * heads
    grid = min(num_blocks, tasks)
    steps = (tasks + grid - 1) // grid

    @T.prim_func
    def main(
        q: T.Tensor((batch, seq_len_q, heads, dim), dtype),
        k: T.Tensor((batch, seq_len_kv, heads_kv, dim), dtype),
        v: T.Tensor((batch, seq_len_kv, heads_kv, dim), dtype),
        out: T.Tensor((batch, seq_len_q, heads, dim), dtype),
        lse: T.Tensor((batch, heads, seq_len_q), "float32"),
    ):
        with T.Kernel(grid) as bx:
            qv = T.alloc_shared((dim,), dtype)
            kt = T.alloc_shared((key_tile, dim), dtype)
            vt = T.alloc_shared((key_tile, dim), dtype)
            acc = T.alloc_shared((dim,), "float32")
            mx = T.alloc_shared((1,), "float32")
            sm = T.alloc_shared((1,), "float32")

            for step in T.serial(steps):
                task = step * grid + bx
                if task < tasks:
                    h = task % heads
                    rest = task // heads
                    s = rest % seq_len_q
                    b = rest // seq_len_q
                    hkv = h // group
                    # Query row s sits at KV position p, bottom-right aligned with the KV
                    # sequence; the causal mask and both windows are read against it.
                    p = s + seq_len_kv - seq_len_q
                    lo = p - window_size_left if window_size_left >= 0 else 0
                    hi = p if is_causal else seq_len_kv - 1
                    if window_size_right >= 0:
                        hi = T.min(hi, p + window_size_right)

                    T.copy(q[b, s, h, :], qv)
                    with T.SimtVF(threads=64):
                        for d in T.Parallel(dim):
                            acc[d] = 0.0
                        for i in T.Parallel(1):
                            mx[i] = -T.infinity("float32")
                            sm[i] = 0.0

                    for t in T.serial(tiles):
                        for j in T.serial(key_tile):
                            # A tile past the end of the KV sequence reads the last row and
                            # is masked below, so the copy stays in bounds.
                            row = T.max(0, T.min(t * key_tile + j, seq_len_kv - 1))
                            T.copy(k[b, row, hkv, :], kt[j, :])
                            T.copy(v[b, row, hkv, :], vt[j, :])

                        with T.SimtVF(threads=64):
                            score = T.alloc_fragment((key_tile,), "float32")
                            mx_new = T.alloc_fragment((1,), "float32")
                            rescale = T.alloc_fragment((1,), "float32")
                            tsum = T.alloc_fragment((1,), "float32")
                            mred = T.alloc_reducer((1,), "float32", op="max")
                            sred = T.alloc_reducer((1,), "float32", op="sum")
                            for j in T.Parallel(key_tile):
                                score[j] = 0.0
                            for d in T.serial(dim):
                                for j in T.Parallel(key_tile):
                                    score[j] = score[j] + T.cast(qv[d], "float32") * T.cast(
                                        kt[j, d], "float32"
                                    )
                            for j in T.Parallel(key_tile):
                                kpos = t * key_tile + j
                                z = score[j] * scale
                                if softcap > 0:
                                    z = softcap * T.tanh(z / softcap)
                                score[j] = T.if_then_else(
                                    (kpos < seq_len_kv) & (kpos >= lo) & (kpos <= hi),
                                    z,
                                    -T.infinity("float32"),
                                )
                            T.reducer_init(mred)
                            for j in T.Parallel(key_tile):
                                T.reducer_update(mred[0], score[j])
                            T.finalize_reducer(mred, mx_new)
                            for i in T.Parallel(1):
                                mx_new[i] = T.max(mx_new[i], mx[i])
                                # A tile with no visible key leaves the running max at
                                # -inf; the neutral value keeps the rescale at 0 instead
                                # of exp(-inf - -inf), which is NaN and would poison the
                                # accumulator for every later tile.
                                mx_new[i] = T.if_then_else(
                                    mx_new[i] == -T.infinity("float32"), 0.0, mx_new[i]
                                )
                                rescale[i] = T.exp(mx[i] - mx_new[i])
                            for d in T.Parallel(dim):
                                acc[d] = acc[d] * rescale[0]
                            T.reducer_init(sred)
                            for j in T.Parallel(key_tile):
                                score[j] = T.exp(score[j] - mx_new[0])
                                T.reducer_update(sred[0], score[j])
                            T.finalize_reducer(sred, tsum)
                            for j in T.serial(key_tile):
                                for d in T.Parallel(dim):
                                    acc[d] = acc[d] + score[j] * T.cast(vt[j, d], "float32")
                            for i in T.Parallel(1):
                                sm[i] = sm[i] * rescale[i] + tsum[i]
                                mx[i] = mx_new[i]

                    with T.SimtVF(threads=64):
                        for d in T.Parallel(dim):
                            out[b, s, h, d] = T.cast(
                                T.if_then_else(sm[0] > 0, acc[d] / sm[0], 0.0), dtype
                            )
                        for i in T.Parallel(1):
                            # Natural log, as flash-attn returns it; a row that saw no key
                            # has an empty sum and reports -inf.
                            lse[b, h, s] = T.if_then_else(
                                sm[0] > 0,
                                mx[0] + T.log(sm[0]),
                                -T.infinity("float32"),
                            )

    return tilelang.compile(main, target="ascend")


class AscendGQADenseWithLSEKernel(Kernel, GQADenseWithLSEFwdInterface):
    """Ascend dense GQA forward with log-sum-exp: one query head row per AI core."""

    devices: ClassVar[frozenset[str]] = frozenset({"npu"})

    @classmethod
    def refusal(cls, call: AttentionCall) -> Optional[str]:
        """The calls this implementation serves."""
        if call.dtype not in (torch.float16, torch.bfloat16):
            return "requires float16 or bfloat16"
        if call.dim <= 0 or call.dim % 16 != 0:
            return "requires a head dimension that is a positive multiple of 16"
        if call.heads_kv <= 0 or call.heads % call.heads_kv != 0:
            return "requires H_kv > 0 and H divisible by H_kv"
        if call.batch <= 0 or call.max_seqlen_q <= 0 or call.seqlen_kv <= 0:
            return "requires positive shapes"
        if call.is_fp8:
            return "does not serve FP8"
        if call.fuse_rope:
            return "does not serve fused RoPE"
        return None

    @classmethod
    def entry_for(cls, call: AttentionCall):
        scale = call.sm_scale if call.sm_scale is not None else call.dim**-0.5
        args = (
            call.batch,
            call.max_seqlen_q,
            call.seqlen_kv,
            call.heads,
            call.heads_kv,
            call.dim,
            scale,
            call.softcap,
            call.window_size_left,
            call.window_size_right,
            call.is_causal,
        )
        device_index = call.device.index if call.device is not None else None
        return (args, call.dtype, device_index), lambda: cls(
            *args, dtype=call.dtype, device_index=device_index
        )

    def __init__(
        self,
        batch: int,
        seq_len_q: int,
        seq_len_kv: int,
        heads: int,
        heads_kv: int,
        dim: int,
        scale: float,
        softcap: float,
        window_size_left: int,
        window_size_right: int,
        is_causal: bool,
        dtype: torch.dtype = torch.bfloat16,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.batch = batch
        self.seq_len_q = seq_len_q
        self.seq_len_kv = seq_len_kv
        self.heads = heads
        self.heads_kv = heads_kv
        self.dim = dim
        self.scale = scale
        self.softcap = softcap
        self.window_size_left = window_size_left
        self.window_size_right = window_size_right
        self.is_causal = is_causal
        self.dtype = dtype
        self._kernel = None

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._require_devices(q=q, k=k, v=v)
        if self._kernel is None:
            # One block per query-head row, capped at the physical core count so the
            # scheduler can overlap rows; the task loop covers the remainder.
            num_blocks = min(self.batch * self.seq_len_q * self.heads, 256)
            self._kernel = _make_gqa_lse_kernel(
                self.batch,
                self.seq_len_q,
                self.seq_len_kv,
                self.heads,
                self.heads_kv,
                self.dim,
                "float16" if self.dtype == torch.float16 else "bfloat16",
                self.scale,
                self.softcap,
                self.window_size_left,
                self.window_size_right,
                self.is_causal,
                num_blocks,
            )
        out = torch.empty(
            (self.batch, self.seq_len_q, self.heads, self.dim), device=q.device, dtype=q.dtype
        )
        lse = torch.empty(
            (self.batch, self.heads, self.seq_len_q), device=q.device, dtype=torch.float32
        )
        self._kernel(q, k, v, out, lse)
        return out, lse


register_kernel_type("GQADenseWithLSEFwdOp", "ascend", AscendGQADenseWithLSEKernel)
