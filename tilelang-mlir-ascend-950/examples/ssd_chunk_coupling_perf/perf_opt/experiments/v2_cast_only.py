"""v2 (probe, not a winner): same AIV path as the baseline but WITHOUT the mask.

``co[i, j] = cast(cv[i, j])`` -- no ``if_then_else``.  Numerically wrong above
the diagonal; measures how much of the 4.1 us/block AIV epilogue is the
divergent predicate versus the SIMT loop itself.
"""

import functools
import os
import sys

import tilelang
import tilelang.ascend.language as T
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _harness  # noqa: E402

base = _harness.load_base()
_tile_shape = base._tile_shape
DUMP = os.environ.get("SSD_CC_DUMP_SRC")


@functools.lru_cache(maxsize=32)
def _ssd_chunk_coupling_kernel(batch, num_chunks, n_groups, chunk_len, d_state, dtype):
    BL, BK = _tile_shape(chunk_len, d_state)
    Q, N, G = chunk_len, d_state, n_groups
    S = num_chunks * Q
    nlt = Q // BL
    grid = batch * num_chunks * n_groups * nlt * nlt
    k_trips = -(-N // BK)
    c_shape = (batch, S, G, N)
    cb_shape = (batch, num_chunks, G, Q, Q)

    @T.prim_func
    def main(
        C_mat: T.Tensor(c_shape, dtype),
        B_mat: T.Tensor(c_shape, dtype),
        cb: T.Tensor(cb_shape, dtype),
    ):
        with T.Kernel(grid) as bid:
            bcg = bid // (nlt * nlt)
            rem = bid % (nlt * nlt)
            bb = bcg // (num_chunks * G)
            cg = bcg % (num_chunks * G)
            bc = cg // G
            bg = cg % G
            l0 = (rem // nlt) * BL
            s0 = (rem % nlt) * BL

            cl1 = T.alloc_l1((BL, BK), dtype)
            bl1 = T.alloc_l1((BL, BK), dtype)
            x0 = T.alloc_l0a((BL, BK), dtype)
            y0 = T.alloc_l0b((BL, BK), dtype)
            acc = T.alloc_l0c((BL, BL), "float32")
            cv = T.alloc_shared((BL // 2, BL), "float32")
            co = T.alloc_shared((BL // 2, BL), dtype)
            T.annotate_buffer_versions({cl1: 2, bl1: 2})

            trips = T.if_then_else(s0 <= l0 + BL - 1, k_trips, 0)
            for kk in T.Pipelined(trips, num_stages=2):
                n0 = kk * BK
                nk = T.min(N - n0, BK)
                T.copy(C_mat[bb, bc * Q + l0 : bc * Q + l0 + BL, bg, n0 : n0 + nk], cl1[:, :nk])
                T.copy(B_mat[bb, bc * Q + s0 : bc * Q + s0 + BL, bg, n0 : n0 + nk], bl1[:, :nk])
                T.copy(cl1[:, :nk], x0[:, :nk])
                T.copy(bl1[:, :nk], y0[:, :nk])
                T.gemm(x0[:, :nk], y0[:, :nk], acc, transpose_B=True, clear_accum=kk == 0)

            T.dual_copy(acc, cv)
            with T.SimtVF(threads=128):
                for i, j in T.Parallel(BL // 2, BL):
                    co[i, j] = T.cast(cv[i, j], dtype)
            T.dual_copy(co, cb[bb, bc, bg, l0 : l0 + BL, s0 : s0 + BL])

    fn = tilelang.compile(main, target="ascend", out_idx=-1)
    if DUMP:
        src = fn.get_kernel_source() if hasattr(fn, "get_kernel_source") else None
        if src:
            with open(DUMP, "w") as f:
                f.write(src)
    return fn


_launch = _ssd_chunk_coupling_kernel
golden_ssd_chunk_coupling = base.golden_ssd_chunk_coupling

if __name__ == "__main__":
    _harness.main(base)
