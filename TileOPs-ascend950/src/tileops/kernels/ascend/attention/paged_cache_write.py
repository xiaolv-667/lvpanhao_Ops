"""Scatter new K/V rows into the page pool their slots name, on Ascend 950.

One AI core owns a short run of tokens. For each token it reads the slot once, and when
that slot names a row it moves the token's key row and value row through the Unified
Buffer into the page pools: a DMA in, an optional FP8 encode in the vector registers, and
a DMA back out to the row the slot names. Both stores address ``slot_mapping[n]``, so the
pages a call writes need not be consecutive, need not be new, and need not belong to one
request. A token whose slot is ``-1`` reads nothing and writes nothing, which leaves a
row no slot names holding the bytes it held.

The token count is a dynamic extent, so one build serves every decode step: the grid is
``ceildiv(tokens, _ROWS_PER_CORE)`` and each core's inner loop is a compile-time run of
``_ROWS_PER_CORE`` tokens, its tail guarded by ``token < tokens``.

An FP8 cache stores ``k / k_scale`` and ``v / v_scale``. The division is FP32, and the
clamp is explicit: the backend's cast alone maps finite overflow to NaN instead of
saturating, so the range is bounded first, and a NaN quotient is carried through the
clamp untouched. Rounding is the cast's own round-to-nearest-ties-to-even, which the
contract asks for; a negative zero keeps its sign. NaN reaches the cache as the
canonical E4M3 NaN code ``0x7F``.
"""

import functools
from typing import ClassVar, Optional

import tilelang
import tilelang.ascend.language as T
import torch

from tileops.backend import register_kernel_type
from tileops.kernels.attention.call_spec import (
    PagedKVCacheWriteCall,
    PagedKVCacheWriteFwdInterface,
)
from tileops.kernels.kernel_base import Kernel

__all__ = ["AscendPagedKVCacheWriteKernel"]

# The largest finite magnitude E4M3FN holds; an FP8 cache saturates here.
_FP8_MAX = 448.0
# Tokens one AI core writes. Measured on the site device: 8 is the best value for a decode
# step and within 10% of the best for the largest prefill, while 1, 16, 32 and 64 lose more
# than they gain at one end or the other (see the work-per-core sweep in the handover notes).
_ROWS_PER_CORE = 8
# A SIMT region runs two elements per thread where it can, and never more than a full
# vector; the widest row one token has is charged one pass of this many lanes.
_MAX_THREADS = 128


def _threads(row_width: int) -> int:
    """Threads one token's row conversion uses: two elements per thread where possible."""
    return min(_MAX_THREADS, max(1, row_width // 2))


@functools.lru_cache(maxsize=64)
def build_paged_cache_write(row_width: int, physical_tokens: int, in_dtype: str, cache_dtype: str):
    """Build the write for one pool geometry and dtype pair.

    ``row_width`` is one token's key or value row flattened to ``heads_kv * dim``
    elements, and ``physical_tokens`` is the page pool flattened to ``num_pages *
    page_size`` rows: the slot is an index into that flat run in both the CUDA contract
    and this program, so the four-dimensional tensors arrive here as rows.
    """
    tokens = T.dynamic("tokens")
    quantized = cache_dtype == "float8_e4m3fn"
    threads = _threads(row_width)
    blocks = T.ceildiv(tokens, _ROWS_PER_CORE)

    if quantized:

        @T.prim_func
        def main(
            K: T.Tensor((tokens, row_width), in_dtype),
            V: T.Tensor((tokens, row_width), in_dtype),
            KPages: T.Tensor((physical_tokens, row_width), cache_dtype),
            VPages: T.Tensor((physical_tokens, row_width), cache_dtype),
            Slots: T.Tensor((tokens,), "int64"),
            KScale: T.Tensor((1,), "float32"),
            VScale: T.Tensor((1,), "float32"),
        ):
            with T.Kernel(blocks) as bx:
                k_row = T.alloc_shared((row_width,), in_dtype)
                v_row = T.alloc_shared((row_width,), in_dtype)
                k_code = T.alloc_shared((row_width,), cache_dtype)
                v_code = T.alloc_shared((row_width,), cache_dtype)
                # One scalar load per core rather than one per vector instruction.
                k_div = T.alloc_shared((1,), "float32")
                v_div = T.alloc_shared((1,), "float32")
                T.copy(KScale, k_div)
                T.copy(VScale, v_div)
                for r in T.serial(_ROWS_PER_CORE):
                    token = bx * _ROWS_PER_CORE + r
                    if token < tokens:
                        slot = Slots[token]
                        if slot >= 0:
                            T.copy(K[token, :], k_row)
                            T.copy(V[token, :], v_row)
                            # One store per vector body: two FP8 stores in one SIMT region
                            # defeat the vectoriser, and a scalar E4M3 store writes 0.
                            with T.SimtVF(threads=threads):
                                for j in T.Parallel(row_width):
                                    value = T.cast(k_row[j], "float32") / k_div[0]
                                    k_code[j] = T.cast(
                                        T.if_then_else(
                                            T.isnan(value),
                                            value,
                                            T.min(
                                                T.max(value, T.float32(-_FP8_MAX)),
                                                T.float32(_FP8_MAX),
                                            ),
                                        ),
                                        cache_dtype,
                                    )
                            with T.SimtVF(threads=threads):
                                for j in T.Parallel(row_width):
                                    value = T.cast(v_row[j], "float32") / v_div[0]
                                    v_code[j] = T.cast(
                                        T.if_then_else(
                                            T.isnan(value),
                                            value,
                                            T.min(
                                                T.max(value, T.float32(-_FP8_MAX)),
                                                T.float32(_FP8_MAX),
                                            ),
                                        ),
                                        cache_dtype,
                                    )
                            T.copy(k_code, KPages[slot, :])
                            T.copy(v_code, VPages[slot, :])

        return tilelang.compile(main, target="ascend", out_idx=[])

    @T.prim_func
    def main(
        K: T.Tensor((tokens, row_width), in_dtype),
        V: T.Tensor((tokens, row_width), in_dtype),
        KPages: T.Tensor((physical_tokens, row_width), cache_dtype),
        VPages: T.Tensor((physical_tokens, row_width), cache_dtype),
        Slots: T.Tensor((tokens,), "int64"),
    ):
        with T.Kernel(blocks) as bx:
            k_row = T.alloc_shared((row_width,), in_dtype)
            v_row = T.alloc_shared((row_width,), in_dtype)
            for r in T.serial(_ROWS_PER_CORE):
                token = bx * _ROWS_PER_CORE + r
                if token < tokens:
                    slot = Slots[token]
                    if slot >= 0:
                        # The rows travel through the Unified Buffer untouched, so the
                        # stored bytes are the input bytes.
                        T.copy(K[token, :], k_row)
                        T.copy(V[token, :], v_row)
                        T.copy(k_row, KPages[slot, :])
                        T.copy(v_row, VPages[slot, :])

    return tilelang.compile(main, target="ascend", out_idx=[])


class AscendPagedKVCacheWriteKernel(Kernel, PagedKVCacheWriteFwdInterface):
    """Write each token's key and value rows into the page-pool row its slot names."""

    devices: ClassVar[frozenset[str]] = frozenset({"npu"})
    supported_archs = None

    @classmethod
    def refusal(cls, call: PagedKVCacheWriteCall) -> Optional[str]:
        if call.dtype not in (torch.float16, torch.bfloat16):
            return "requires float16 or bfloat16 rows"
        if call.cache_dtype not in (torch.float16, torch.bfloat16, torch.float8_e4m3fn):
            return "requires a float16, bfloat16 or float8_e4m3fn page pool"
        if call.cache_dtype is not torch.float8_e4m3fn and call.cache_dtype != call.dtype:
            return "requires a page pool of the call's own dtype, or float8_e4m3fn"
        if min(call.heads_kv, call.dim, call.page_size, call.num_pages) <= 0:
            return "requires positive head, dimension, page-size and page counts"
        if call.row_width < 2:
            return "requires a row of at least two elements"
        return super().refusal(call)

    def __init__(self, call: PagedKVCacheWriteCall) -> None:
        reason = self.refusal(call)
        if reason is not None:
            raise ValueError(reason)
        super().__init__(device_index=call.device.index if call.device is not None else None)
        self.call = call
        self.in_dtype, self.cache_dtype = call.dtype, call.cache_dtype
        self.row_width, self.physical_tokens = call.row_width, call.physical_tokens
        self.quantized = call.quantized

    def forward(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        k_pages: torch.Tensor,
        v_pages: torch.Tensor,
        slot_mapping: torch.Tensor,
        k_scale: Optional[torch.Tensor] = None,
        v_scale: Optional[torch.Tensor] = None,
    ) -> None:
        """Write the rows the slots name into both pools; see the interface for the contract."""
        self._require_devices(k=k, v=v, k_pages=k_pages, v_pages=v_pages, slot_mapping=slot_mapping)
        # A call of no tokens writes nothing, and a zero-extent program is not a legal one,
        # so it is answered here rather than compiled.
        tokens = k.shape[0]
        if tokens == 0:
            return
        if k.dtype != self.in_dtype or v.dtype != self.in_dtype:
            raise ValueError("k and v must hold the specialization's row dtype")
        if k_pages.dtype != self.cache_dtype or v_pages.dtype != self.cache_dtype:
            raise ValueError("the page pools must hold the specialization's cache dtype")
        if slot_mapping.dtype != torch.int64:
            raise ValueError("slot_mapping must be int64")
        if k.shape[1:] != (self.call.heads_kv, self.call.dim) or v.shape != k.shape:
            raise ValueError("k and v must hold (tokens, heads_kv, dim) rows")
        if k_pages.shape[1:] != (self.call.page_size, self.call.heads_kv, self.call.dim):
            raise ValueError("k_pages must hold (num_pages, page_size, heads_kv, dim)")
        if v_pages.shape != k_pages.shape:
            raise ValueError("k_pages and v_pages must be one page pool's shape")
        if slot_mapping.numel() != tokens:
            raise ValueError("slot_mapping holds one row index per token")
        if self.quantized and (k_scale is None or v_scale is None):
            raise ValueError("a float8_e4m3fn page pool requires both scales")
        if not self.quantized and (k_scale is not None or v_scale is not None):
            raise ValueError("only a float8_e4m3fn page pool takes scales")
        # The slot indexes the pool as one flat run of rows, so both pools are handed over
        # as rows of row_width elements; the views write through to the pools themselves.
        args = [
            k.view(tokens, self.row_width),
            v.view(tokens, self.row_width),
            k_pages.view(self.physical_tokens, self.row_width),
            v_pages.view(self.physical_tokens, self.row_width),
            slot_mapping,
        ]
        if self.quantized:
            args += [k_scale, v_scale]
        build_paged_cache_write(
            self.row_width,
            self.physical_tokens,
            self.dtype_to_str(self.in_dtype),
            self.dtype_to_str(self.cache_dtype),
        )(*args)


register_kernel_type("PagedKVCacheWriteFwdOp", "ascend", AscendPagedKVCacheWriteKernel)
