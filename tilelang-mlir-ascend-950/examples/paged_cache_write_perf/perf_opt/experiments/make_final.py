"""Assemble the final self-contained perf_opt module from the tuned parts.

The delivered file must be self-contained (golden + kernel + layered tests), so
it is assembled from the two files the experiments ran with:
``_pcw_tests.py`` (the byte-exact harness, itself the Stage 3 test section) and
the kernel body below (the r2_c1 merged winner).
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
PERF_OPT = os.path.dirname(HERE)

DOC = '''# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Paged KV cache write (ascend) -- Stage 4 tuned (perf_opt).

Scatter new K/V rows into the pool rows ``slot_mapping`` names, in place:

    k_pages[slot_mapping[n]] = k[n]                          (16-bit cache)
    k_pages[slot_mapping[n]] = clamp(k[n] / k_scale, -448, 448) -> e4m3

A token whose slot is ``-1`` is skipped and no row is touched unless a slot
names it.  The arithmetic, the dtypes and the byte-level result are the Stage 3
contract; only the memory path changed.

Stage 4 change (elementwise scatter -> staged DMA + vector scatter):

* The Stage 3 kernel moved every GM byte with per-lane vector accesses
  (``K[token, col]`` read and ``KPages[slot, col]`` written element by element).
  On 950PR a descriptor copy runs at 1.49 TB/s against 0.74 TB/s per-lane
  (platform constant ``CONST-950-dma-vs-lane-gm``), so the reads were the first
  loss: the source rows of a block are contiguous, and the whole row tile now
  arrives in UB with one ``T.copy`` per pool.
* The stores are scattered (one row per destination), and the scheduler drains
  the MTE3 pipe before every UB->GM copy (``asc_sync_pipe(PIPE_MTE3)``), which
  costs a fixed ~0.3 us of core time per row-sized DMA.  The 16-bit build
  therefore stores 16 B/lane straight from UB to the scattered GM row (one
  ``T.Parallel`` pass per pool, ``slot >= 0`` as the lane predicate); the fp8
  build cannot widen a one-byte element per lane (measured 157 us vs 45 us on
  the same workload) and keeps the per-row DMA.
* The fp8 conversion runs as one vector pass per pool over the staged tile,
  ``cast_f32 -> /scale -> clamp(-448, 448) -> cast_e4m3`` exactly as before.
* Two measured policies pick the tile per call: the row-tile height follows the
  cache width (an fp8 row is half the bytes, so its step is 4x smaller), and the
  SimtVF lane count follows the tile size (512 lanes from an 8192-element tile
  up, else 128).

Measured on ``bench_paged_cache_write.py``'s cases (msprof ``Task Duration(us)``,
one launch per call, 3 samples, median):

    llama-8b-prefill   811.5 -> 46.2 us   (1.45 TB/s, 91% of nominal HBM)
    qwen3-235b-fp8     270.6 -> 37.8 us
    qwen3-235b-fp8-fp16 79.7 -> 15.6 us
    gpt-oss-120b        26.6 ->  6.3 us
    llama-8b-decode     16.1 ->  4.5 us
    odd-tokens           8.2 ->  2.8 us

Content layout:

1. ``golden_paged_cache_write`` -- torch CPU reference (source ``ref_program``).
2. ``_paged_cache_write_kernel`` -- the frozen factory; returns a callable
   ``(rows, cols) -> compiled callable``; ``pick_config`` is the tuned tile
   policy the wrapper's default config resolves through.
3. ``run_case`` / ``_write`` -- launch + exact comparison of both page pools.
4. ``run_L0`` / ``run_L1`` / ``run_L2`` / ``run_boundary`` / ``main``.
"""

import argparse
import functools
import time

import tilelang
import tilelang.ascend.language as T
import torch
import torch_npu  # noqa: F401  -- registers torch.npu

_FP8_E4M3FN = "float8_e4m3fn"
_FP8_MAX = 448.0
_IN_DTYPES = ("float16", "bfloat16")
_THREADS = 128
_DEFAULT_BLOCK_ROWS = 16
_DEFAULT_BLOCK_COLS = 128

_TORCH_DTYPE = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float8_e4m3fn": torch.float8_e4m3fn,
}
_DTYPE_NAME = {v: k for k, v in _TORCH_DTYPE.items()}


'''

KERNEL = '''
# ---------------------------------------------------------------------------
# 2. Kernel: staged row-tile DMA, vector (16-bit) or DMA (fp8) scatter
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=256)
def _build(heads_kv, dim, physical_tokens, in_dtype, cache_dtype, cols, rows):
    """Compile one specialised program; cached per geometry and tile shape."""
    row_width = heads_kv * dim
    quantized = cache_dtype == _FP8_E4M3FN
    cols = cols if (row_width % cols == 0 and cols <= row_width) else row_width
    nchunks = row_width // cols
    tokens = T.dynamic("tokens")
    slot_dtype = "int32" if physical_tokens < 2**31 else "int64"
    # fp8 stores are one byte wide: the vector path cannot widen them, so the
    # quantised build keeps the DMA store (a 512 B row segment) while a 16-bit
    # tile big enough to fill the lanes stores straight to GM with no MTE3 drain
    # per row (measured: the lane path wins from a 4096-element tile up).
    store_dma = quantized or rows * cols < 4096
    # Wide tiles fill more lanes; small ones keep 128 so every lane still holds
    # a vector's worth of elements.
    threads = 512 if rows * cols >= 8192 else _THREADS
    item = 1 if quantized else 2
    pad = 32 // item
    cols_pad = ((cols + pad - 1) // pad) * pad

    @T.macro
    def one(X, XPages, Slots, scale, rb, cb):
        """Stage this block's rows of one tensor, convert them, scatter them."""
        row0 = rb * rows
        nr = T.min(rows, tokens - row0)
        one_row = T.min(nr, 1)  # provably 1, but never folded into a literal
        c0 = cb * cols
        x_ub = T.alloc_shared((rows, cols_pad), in_dtype)
        T.copy(X[row0 : row0 + nr, c0 : c0 + cols], x_ub[0:nr, 0:cols])
        if store_dma:
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
                    slot_ub[i] = T.cast(Slots[T.min(row0 + i, tokens - 1)], slot_dtype)
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
            with T.Kernel(T.ceildiv(tokens, rows) * nchunks) as bx:
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
        with T.Kernel(T.ceildiv(tokens, rows) * nchunks) as bx:
            one(K, KPages, Slots, K, bx // nchunks, bx % nchunks)
            one(V, VPages, Slots, V, bx // nchunks, bx % nchunks)

    return tilelang.compile(_16bit, target="ascend", out_idx=[])


def _paged_cache_write_kernel(heads_kv, dim, physical_tokens, in_dtype, cache_dtype):
    """Frozen factory: geometry and dtypes in, ``(block_rows, block_cols)`` out."""
    if in_dtype not in _IN_DTYPES:
        raise ValueError(f"dtype: input dtype {in_dtype!r} is not one of {_IN_DTYPES}")
    if cache_dtype != _FP8_E4M3FN and cache_dtype != in_dtype:
        raise ValueError(
            f"dtype: cache dtype {cache_dtype!r} is neither {_FP8_E4M3FN!r} "
            f"nor the input dtype {in_dtype!r}"
        )

    def _func(block_rows, block_cols):
        return _build(
            heads_kv, dim, physical_tokens, in_dtype, cache_dtype, block_cols, block_rows
        )

    return _func


def pick_config(tokens, heads_kv, dim, physical_tokens, in_dtype, cache_dtype):
    """Tuned tile policy: row-tile height per cache width, whole row per DMA."""
    row_width = heads_kv * dim
    step = 128 if cache_dtype == _FP8_E4M3FN else 32
    rows = max(1, min(32, -(-tokens // step)))
    if cache_dtype == _FP8_E4M3FN:
        rows = min(rows, 16)
    # a tail block that is mostly padding buys nothing: keep four blocks' worth
    # of rows at the smallest token counts
    rows = min(rows, max(1, tokens // 4))
    # the fp8 conversion pass needs two elements per lane
    if cache_dtype == _FP8_E4M3FN:
        rows = max(rows, -(-256 // row_width))
    return rows, row_width


'''


def section(path, start, end=None):
    text = open(path).read()
    i = text.index(start)
    j = len(text) if end is None else text.index(end)
    return text[i:j]


def main():
    tests = os.path.join(PERF_OPT, "_pcw_tests.py")
    golden = section(tests, "# 1. Golden", "# 3. Launch")
    harness = section(tests, "# 3. Launch", "def main():")
    harness = harness.replace("KERNEL_FACTORY()", "_paged_cache_write_kernel")
    harness = harness.replace("pick_config(", "pick_config(")
    tail = '''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--level", default="L0", choices=["L0", "L1", "L2", "boundary", "all"]
    )
    args, _ = parser.parse_known_args()
    levels = {
        "L0": run_L0,
        "L1": run_L1,
        "L2": run_L2,
        "boundary": run_boundary,
    }
    chosen = levels.values() if args.level == "all" else [levels[args.level]]
    for run in chosen:
        run()
    print(f"[PRECISION_PASS] max_diff={_MAX_DIFF[0]:.6g}")


if __name__ == "__main__":
    main()
'''
    out = DOC + golden.rstrip() + "\n\n" + KERNEL.lstrip("\n") + harness.rstrip() + tail
    body = out + "\n"
    path = os.path.join(PERF_OPT, "paged_cache_write.py")
    with open(path, "w") as f:
        f.write(body)
    print(f"wrote {path} ({len(body.splitlines())} lines)")
    # a byte-identical alias under the operator's class name, for callers that
    # look the kernel up by op name
    alias = os.path.join(PERF_OPT, "PagedKVCacheWriteFwdOp.py")
    with open(alias, "w") as f:
        f.write(body)
    print(f"wrote {alias}")


if __name__ == "__main__":
    main()
