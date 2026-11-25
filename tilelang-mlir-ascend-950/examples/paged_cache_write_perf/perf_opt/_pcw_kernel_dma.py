# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""DMA-staged paged_cache_write kernel template (Stage 4 experiments).

Stage 3 moved every GM byte with per-lane vector accesses.  This template keeps
the same scatter semantics but changes where the traffic goes through:

* load: a block's source rows are contiguous, so the whole row tile is one
  ``T.copy`` DMA (platform constant ``CONST-950-dma-vs-lane-gm``: 1.49 TB/s over
  descriptors against 0.74 TB/s per-lane on 950PR);
* convert (fp8 only): one vector pass per pool over the staged UB tile;
* store: two selectable paths.
  - ``store="dma"``: one ``T.copy`` per row to the row ``slot_mapping`` names.
    Destination regions keep a symbolic row extent (``slot : slot + one`` with
    ``one = min(nr, 1)``): a literal single-row region is collapsed into a 1-D
    descriptor and the dynamic row index is then rejected
    (``CopyNode::MakeIndices: Index 1 out of bounds 1``).
  - ``store="lane"``: one ``T.Parallel(rows, cols)`` pass per pool storing from
    UB straight to the scattered GM row (16 B/lane, predicated by ``slot >= 0``).
    The scheduler drains the MTE3 pipe before every UB->GM copy
    (``asc_sync_pipe(PIPE_MTE3)``, a same-pipe dependency), so a row-sized DMA
    costs a fixed ~0.3 us of core time; the vector store path has no such drain
    and wins as soon as the store is smaller than a few KiB.

``rows`` rows per block, ``split`` puts one tensor per block (2x the blocks, one
read + one write stream each), ``threads`` is the SimtVF lane count.
"""

import functools

import tilelang
import tilelang.ascend.language as T

_FP8_E4M3FN = "float8_e4m3fn"
_FP8_MAX = 448.0
_IN_DTYPES = ("float16", "bfloat16")


def make_factory(split, store, threads=128):
    """``(geometry, in_dtype, cache_dtype) -> (rows, cols) -> fn``."""

    def _factory(heads_kv, dim, physical_tokens, in_dtype, cache_dtype):
        if in_dtype not in _IN_DTYPES:
            raise ValueError(
                f"dtype: input dtype {in_dtype!r} is not one of {_IN_DTYPES}"
            )
        if cache_dtype != _FP8_E4M3FN and cache_dtype != in_dtype:
            raise ValueError(
                f"dtype: cache dtype {cache_dtype!r} is neither {_FP8_E4M3FN!r} "
                f"nor the input dtype {in_dtype!r}"
            )

        def _func(block_rows, block_cols):
            return _build(
                heads_kv,
                dim,
                physical_tokens,
                in_dtype,
                cache_dtype,
                int(block_cols),
                int(block_rows),
                bool(split),
                str(store),
                int(threads),
            )

        return _func

    return _factory


@functools.lru_cache(maxsize=256)
def _build(heads_kv, dim, physical_tokens, in_dtype, cache_dtype, cols, rows, split, store,
           threads):
    row_width = heads_kv * dim
    quantized = cache_dtype == _FP8_E4M3FN
    cols = cols if (row_width % cols == 0 and cols <= row_width) else row_width
    nchunks = row_width // cols
    tokens = T.dynamic("tokens")
    slot_dtype = "int32" if physical_tokens < 2**31 else "int64"
    sides = 2 if split else 1
    # fp8 stores are one byte wide: the vector path cannot widen them, so the
    # quantised build keeps the DMA store (a 512 B row segment) while a 16-bit
    # tile big enough to fill the lanes stores 16 B/lane straight to GM with no
    # MTE3 drain per row (measured: lane wins from a 4096-element tile up).
    if store == "auto":
        store_path = "dma" if (quantized or rows * cols < 4096) else "lane"
    else:
        store_path = store
    if threads <= 0:
        # Wide tiles fill more lanes; small ones keep 128 so every lane still
        # holds a vector's worth of elements.
        threads = 512 if rows * cols >= 8192 else 128
    item = 1 if quantized else 2
    pad = 32 // item
    cols_pad = ((cols + pad - 1) // pad) * pad

    @T.macro
    def one(X, XPages, Slots, scale, rb, cb):
        """Stage this block's rows of one tensor, convert them, scatter them."""
        row0 = rb * rows
        nr = T.min(rows, tokens - row0)
        one_row = T.min(nr, 1)
        c0 = cb * cols
        x_ub = T.alloc_shared((rows, cols_pad), in_dtype)
        T.copy(X[row0 : row0 + nr, c0 : c0 + cols], x_ub[0:nr, 0:cols])
        if store_path == "dma":
            if quantized:
                x8_ub = T.alloc_shared((rows, cols_pad), cache_dtype)
                with T.SimtVF(threads=threads):
                    for i, c in T.Parallel(rows, cols):
                        x8_ub[i, c] = T.cast(
                            T.clamp(
                                T.cast(x_ub[i, c], "float32") / scale, -_FP8_MAX, _FP8_MAX
                            ),
                            cache_dtype,
                        )
                for i in T.unroll(rows):
                    if row0 + i < tokens:
                        slot = T.cast(Slots[row0 + i], slot_dtype)
                        if slot >= 0:
                            T.copy(
                                x8_ub[i : i + 1, 0:cols],
                                XPages[slot : slot + one_row, c0 : c0 + cols],
                            )
            else:
                for i in T.unroll(rows):
                    if row0 + i < tokens:
                        slot = T.cast(Slots[row0 + i], slot_dtype)
                        if slot >= 0:
                            T.copy(
                                x_ub[i : i + 1, 0:cols],
                                XPages[slot : slot + one_row, c0 : c0 + cols],
                            )
        else:
            slot_ub = T.alloc_shared((rows,), "int32")
            with T.SimtVF(threads=threads):
                for i in T.Parallel(rows):
                    slot_ub[i] = T.cast(
                        Slots[T.min(row0 + i, tokens - 1)], slot_dtype
                    )
                if quantized:
                    for i, c in T.Parallel(rows, cols):
                        slot = slot_ub[i]
                        if (row0 + i < tokens) and (slot >= 0):
                            XPages[slot, c0 + c] = T.cast(
                                T.clamp(
                                    T.cast(x_ub[i, c], "float32") / scale,
                                    -_FP8_MAX,
                                    _FP8_MAX,
                                ),
                                cache_dtype,
                            )
                else:
                    for i, c in T.Parallel(rows, cols):
                        slot = slot_ub[i]
                        if (row0 + i < tokens) and (slot >= 0):
                            XPages[slot, c0 + c] = x_ub[i, c]

    if quantized:

        @T.prim_func
        def _fp8(
            K: T.Tensor((tokens, row_width), in_dtype),
            V: T.Tensor((tokens, row_width), in_dtype),
            KPages: T.Tensor((physical_tokens, row_width), cache_dtype),
            VPages: T.Tensor((physical_tokens, row_width), cache_dtype),
            Slots: T.Tensor((tokens,), "int64"),
            KScale: T.Tensor((1,), "float32"),
            VScale: T.Tensor((1,), "float32"),
        ) -> None:
            with T.Kernel(T.ceildiv(tokens, rows) * nchunks * sides) as bx:
                if split:
                    if bx % sides == 0:
                        one(K, KPages, Slots, KScale[0], (bx // sides) // nchunks,
                            (bx // sides) % nchunks)
                    else:
                        one(V, VPages, Slots, VScale[0], (bx // sides) // nchunks,
                            (bx // sides) % nchunks)
                else:
                    one(K, KPages, Slots, KScale[0], bx // nchunks, bx % nchunks)
                    one(V, VPages, Slots, VScale[0], bx // nchunks, bx % nchunks)

        return tilelang.compile(_fp8, target="ascend", out_idx=[])

    @T.prim_func
    def _16bit(
        K: T.Tensor((tokens, row_width), in_dtype),
        V: T.Tensor((tokens, row_width), in_dtype),
        KPages: T.Tensor((physical_tokens, row_width), cache_dtype),
        VPages: T.Tensor((physical_tokens, row_width), cache_dtype),
        Slots: T.Tensor((tokens,), "int64"),
    ) -> None:
        with T.Kernel(T.ceildiv(tokens, rows) * nchunks * sides) as bx:
            if split:
                if bx % sides == 0:
                    one(K, KPages, Slots, K, (bx // sides) // nchunks,
                        (bx // sides) % nchunks)
                else:
                    one(V, VPages, Slots, V, (bx // sides) // nchunks,
                        (bx // sides) % nchunks)
            else:
                one(K, KPages, Slots, K, bx // nchunks, bx % nchunks)
                one(V, VPages, Slots, V, bx // nchunks, bx % nchunks)

    return tilelang.compile(_16bit, target="ascend", out_idx=[])
