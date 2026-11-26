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
"""Paged KV cache write (ascend) -- Stage 4 tuned, round 2 (perf_opt).

Scatter new K/V rows into the pool rows ``slot_mapping`` names, in place:

    k_pages[slot_mapping[n]] = k[n]                          (16-bit cache)
    k_pages[slot_mapping[n]] = clamp(k[n] / k_scale, -448, 448) -> e4m3

A token whose slot is ``-1`` is skipped and no row is touched unless a slot
names it.  Arithmetic, dtypes and byte-level results are the Stage 3 contract;
only the memory path changed.

Round 1 (staged DMA): the source rows of a block are contiguous, so the whole
row tile arrives in UB with one ``T.copy`` per pool instead of per-lane reads
(950PR: 1.49 TB/s over descriptors vs 0.74 TB/s per-lane).  The scattered store
stays row-granular -- it writes the rows ``slot_mapping`` names.

Round 2 (manual sync on the quantised store path): the auto-scheduler drained
the store pipe before *every* UB->GM row copy (``asc_sync_pipe(PIPE_MTE3)``, a
same-pipe dependency it cannot disprove), so 8192 scattered 512 B rows cost
~0.16-0.25 us of core time each and the fp8 case was store-issue bound.  The
quantised build now compiles with ``tl.enable_auto_schedule=False`` and states
the only two dependencies the kernel has -- MTE2->V before the conversion reads
the staged tile, V->MTE3 before the stores read the converted tile -- plus one
per-block PIPE_MTE3 drain.  Distinct non-negative slots are the op contract, so
two row stores never target the same row and need no ordering between them.
The 16-bit build stays on auto-schedule with a per-lane (16 B) store: it already
runs at the HBM ceiling.

Measured on the migrated benchmark (framework msprof protocol: warm_up=5,
launch_count=10, median of the measured launches):

    qwen3-235b-fp8      31.8 -> 19.4 us   (1.64x, MBU 25% -> 41%)
    qwen3-235b-fp8-fp16 13.1 ->  7.3 us   (1.79x)
    llama-8b-prefill    43.5 -> 43.9 us   (unchanged, at ~1.4 TB/s)
    gpt-oss-120b / decode / odd-tokens    unchanged (16-bit path untouched)

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
# 2. Kernel: staged row-tile DMA, manual sync on the fp8 scatter, lane store
#    for 16-bit
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
    if quantized and rows * cols >= 16384:
        threads = 1024
    else:
        threads = 512 if rows * cols >= 8192 else _THREADS
    item = 1 if quantized else 2
    pad = 32 // item
    cols_pad = ((cols + pad - 1) // pad) * pad
    # Manual sync for the quantised DMA-store path (see the module docstring).
    manual = quantized and store_dma

    @T.macro
    def one(X, XPages, Slots, scale, rb, cb, eid):
        """Stage this block's rows of one tensor, convert them, scatter them."""
        row0 = rb * rows
        nr = T.min(rows, tokens - row0)
        one_row = T.min(nr, 1)  # provably 1, but never folded into a literal
        c0 = cb * cols
        x_ub = T.alloc_shared((rows, cols_pad), in_dtype)
        T.copy(X[row0 : row0 + nr, c0 : c0 + cols], x_ub[0:nr, 0:cols])
        if manual:
            T.ascend_set_flag("MTE2_V", eid)
            T.ascend_wait_flag("MTE2_V", eid)
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
                if manual:
                    T.ascend_set_flag("V_MTE3", eid)
                    T.ascend_wait_flag("V_MTE3", eid)
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
                one(K, KPages, Slots, KScale[0], bx // nchunks, bx % nchunks, 0)
                one(V, VPages, Slots, VScale[0], bx // nchunks, bx % nchunks, 1)
                if manual:
                    # one drain per block so the rows are on GM before it ends
                    T.ascend_pipe_barrier("PIPE_MTE3")

        if manual:
            return tilelang.compile(
                _fp8,
                target="ascend",
                out_idx=[],
                pass_configs={"tl.enable_auto_schedule": False},
            )
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
            one(K, KPages, Slots, K, bx // nchunks, bx % nchunks, 0)
            one(V, VPages, Slots, V, bx // nchunks, bx % nchunks, 1)

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
    """Row-tile height, whole row per DMA.

    Both builds tile ``tokens / 32`` rows per block; the quantised build allows a
    taller tile (48 vs 32) because its stores no longer pay the per-row MTE3
    drain, so a block amortises its fixed cost over more rows (measured on the
    fp8 case: rows 16/32/48/64 -> 20.7 / 20.2 / 19.5 / 23.5 us).
    """
    row_width = heads_kv * dim
    quantized = cache_dtype == _FP8_E4M3FN
    rows = max(1, -(-tokens // 32))
    rows = min(rows, 48 if quantized else 32)
    rows = min(rows, max(1, tokens // 4))
    if quantized:
        # the conversion pass needs two elements per lane
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
    path = os.path.join(PERF_OPT, "PagedKVCacheWriteFwdOp.py")
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
