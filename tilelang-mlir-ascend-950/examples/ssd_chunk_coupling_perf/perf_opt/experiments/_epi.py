"""Kernel-factory template shared by the Stage-4 experiment modules.

Only the epilogue (last section of ``main``) differs between experiments; the
task decode, K pipeline and tile geometry are the frozen Stage-3 form so that
every measured delta is attributable to one change.
"""

import functools
import os
import sys

import tilelang
import tilelang.ascend.language as T
import torch

EPILOGUE_BASELINE = "baseline"
EPILOGUE_MASK_MUL = "mask_mul"
EPILOGUE_THREADS64 = "threads64"
EPILOGUE_CLASS_UL = "class_ul"
EPILOGUE_CLASS_UL_STRIP = "class_ul_strip"


def build(base, epilogue, bl_ladder=None, force_trips=False):
    _tile_shape = base._tile_shape

    @functools.lru_cache(maxsize=32)
    def _factory(batch, num_chunks, n_groups, chunk_len, d_state, dtype):
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
                zr = T.alloc_shared((BL // 2, BL), dtype)
                T.annotate_buffer_versions({cl1: 2, bl1: 2})

                trips = k_trips if force_trips else T.if_then_else(
                    s0 <= l0 + BL - 1, k_trips, 0
                )
                for kk in T.Pipelined(trips, num_stages=2):
                    n0 = kk * BK
                    nk = T.min(N - n0, BK)
                    T.copy(C_mat[bb, bc * Q + l0 : bc * Q + l0 + BL, bg, n0 : n0 + nk], cl1[:, :nk])
                    T.copy(B_mat[bb, bc * Q + s0 : bc * Q + s0 + BL, bg, n0 : n0 + nk], bl1[:, :nk])
                    T.copy(cl1[:, :nk], x0[:, :nk])
                    T.copy(bl1[:, :nk], y0[:, :nk])
                    T.gemm(x0[:, :nk], y0[:, :nk], acc, transpose_B=True, clear_accum=kk == 0)

                sid = T.call_extern("int64", "asc_get_sub_block_id")
                thr = T.cast(sid, "int32") * (BL // 2)
                # predicate right-hand side, strength reduced (DESIGN 1.6.1 O2)
                cdiag = l0 + thr - s0

                if epilogue == EPILOGUE_BASELINE:
                    T.dual_copy(acc, cv)
                    with T.SimtVF(threads=128):
                        for i, j in T.Parallel(BL // 2, BL):
                            co[i, j] = T.cast(
                                T.if_then_else((j - i) <= cdiag, cv[i, j], T.float32(0.0)),
                                dtype,
                            )
                    T.dual_copy(co, cb[bb, bc, bg, l0 : l0 + BL, s0 : s0 + BL])

                elif epilogue == EPILOGUE_MASK_MUL:
                    T.dual_copy(acc, cv)
                    with T.SimtVF(threads=128):
                        for i, j in T.Parallel(BL // 2, BL):
                            co[i, j] = T.cast(
                                cv[i, j] * T.cast((j - i) <= cdiag, "float32"), dtype
                            )
                    T.dual_copy(co, cb[bb, bc, bg, l0 : l0 + BL, s0 : s0 + BL])

                elif epilogue == EPILOGUE_THREADS64:
                    T.dual_copy(acc, cv)
                    with T.SimtVF(threads=64):
                        for i, j in T.Parallel(BL // 2, BL):
                            co[i, j] = T.cast(
                                T.if_then_else((j - i) <= cdiag, cv[i, j], T.float32(0.0)),
                                dtype,
                            )
                    T.dual_copy(co, cb[bb, bc, bg, l0 : l0 + BL, s0 : s0 + BL])

                elif epilogue == EPILOGUE_CLASS_UL:
                    # U: strictly upper tile -> exact zero block, no GEMM, no mask.
                    # L: strictly lower tile -> one Cube fixpipe store (fp32->dtype).
                    # D: diagonal -> unchanged Stage-3 SIMT mask epilogue.
                    if s0 > l0 + BL - 1:
                        with T.SimtVF(threads=128):
                            for i, j in T.Parallel(BL // 2, BL):
                                zr[i, j] = T.cast(T.float32(0.0), dtype)
                        T.dual_copy(zr, cb[bb, bc, bg, l0 : l0 + BL, s0 : s0 + BL])
                    else:
                        if l0 > s0:
                            T.copy(acc, cb[bb, bc, bg, l0 : l0 + BL, s0 : s0 + BL])
                        else:
                            T.dual_copy(acc, cv)
                            with T.SimtVF(threads=128):
                                for i, j in T.Parallel(BL // 2, BL):
                                    co[i, j] = T.cast(
                                        T.if_then_else(
                                            (j - i) <= cdiag, cv[i, j], T.float32(0.0)
                                        ),
                                        dtype,
                                    )
                            T.dual_copy(co, cb[bb, bc, bg, l0 : l0 + BL, s0 : s0 + BL])
                else:
                    raise ValueError(epilogue)

        return tilelang.compile(main, target="ascend", out_idx=-1)

    return _factory
