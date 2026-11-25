"""Floor probes: how fast can this device move the RoPE payload at all?

Two programs over the *same* payload shape as the measured cases (query
``[T, Hq*D]`` + key ``[T, Hk*D]`` read once and written once), with no rotation
at all -- they bound the op from below:

* ``KIND=lane``  -- per-lane ``T.Parallel`` copy (the access shape the Stage 3
  baseline uses: 8-byte per-lane GM accesses, one read pass + one write pass).
* ``KIND=dma``   -- ``T.copy`` GM->UB tile staging + ``T.copy`` UB->GM.

Measured with ``experiments/bench_one.py`` (one msprof capture per call):

    KIND=dma python perf_opt/experiments/bench_one.py --module perf_opt/experiments/p_copy.py \
        --workload llama3-8b-b2-2k-float32 --no-l0
"""

import os

import tilelang
import tilelang.ascend.language as T
import torch
import torch_npu  # noqa: F401  -- registers torch.npu

_THREADS = 256
_UNITS_PER_BLOCK = 2048
_TILE_ELEMS = 4096
KIND = os.environ.get("KIND", "dma")
_VEC = int(os.environ.get("VEC", "2"))  # explicit per-lane vector width (lane mode)

_DT = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def _copy_kernel(cols, dtype, bt):
    tokens = T.dynamic("num_tokens")
    nrb = T.ceildiv(tokens, bt)

    if KIND == "lane":

        @T.prim_func
        def main(X: T.Tensor((tokens, cols), dtype), Y: T.Tensor((tokens, cols), dtype)) -> None:
            with T.Kernel(nrb) as bx:
                with T.SimtVF(threads=_THREADS):
                    for i, c in T.Parallel(bt, cols):
                        if bx * bt + i < tokens:
                            Y[bx * bt + i, c] = X[bx * bt + i, c]

    elif KIND == "vec":
        # explicit per-lane vector width: one T.vectorized run of _VEC per lane
        nrun = cols // _VEC

        @T.prim_func
        def main(X: T.Tensor((tokens, cols), dtype), Y: T.Tensor((tokens, cols), dtype)) -> None:
            with T.Kernel(nrb) as bx:
                with T.SimtVF(threads=_THREADS):
                    for i, r in T.Parallel(bt, nrun):
                        if bx * bt + i < tokens:
                            for v in T.vectorized(_VEC):
                                Y[bx * bt + i, r * _VEC + v] = X[bx * bt + i, r * _VEC + v]

    else:

        @T.prim_func
        def main(X: T.Tensor((tokens, cols), dtype), Y: T.Tensor((tokens, cols), dtype)) -> None:
            with T.Kernel(nrb) as bx:
                x_ub = T.alloc_shared((bt, cols), dtype)
                T.copy(X[bx * bt : bx * bt + bt, :], x_ub)
                T.copy(x_ub, Y[bx * bt : bx * bt + bt, :])

    return tilelang.compile(main, target="ascend", out_idx=[1])


def _rope_positions_kernel(
    query_heads, key_heads, head_size, rotary_dim, max_position, dtype,
    position_dtype, offset_dtype, neox, rot_vec, tail_vec, row_vec,
):
    """Same factory signature as the Stage 3 module so the harness is unchanged."""
    dim = head_size
    q_width = query_heads * dim
    k_width = key_heads * dim

    def _program(units_per_block, staged):
        progs = []
        for width in (q_width, k_width):
            if not width:
                progs.append(None)
                continue
            bt = max(1, min(int(units_per_block) // width, _TILE_ELEMS // width))
            progs.append(_copy_kernel(width, dtype, bt))
        return tuple(progs)

    return _program


def _build(query_heads, key_heads, head_size, rotary_dim, max_position, dtype,
           position_dtype, offset_dtype, neox, units_per_block=None, staged=False):
    fn = _rope_positions_kernel(query_heads, key_heads, head_size, rotary_dim,
                                max_position, dtype, position_dtype, offset_dtype,
                                neox, 1, 1, 1)
    return fn(units_per_block or _UNITS_PER_BLOCK, staged)


def _launch(fn, positions, query, key, cos, sin, offsets, key_heads):
    q_prog, k_prog = fn
    q_out = q_prog(query)
    if key_heads and k_prog is not None:
        return q_out, k_prog(key)
    return q_out, None
