"""Row-panel kernel: one block per ``(b, c, g)`` chunk, all Q*Q tiles inside.

Motivation (measured, ``profiles/experiments``): with one output tile per block
the AIV epilogue waits for that block's own Cube phase, so a block costs
``aic + aiv`` instead of ``max(aic, aiv)``; and ``C``/``B`` row blocks are
re-read 1.5x.

This variant is specialised to ``nlt = Q // BL == 2`` (all manifest full
workloads: Q=256 -> BL=128).  One block loads the whole ``C``/``B`` chunk panel
into L1 once (1.0x read amplification, 128 KB for Q=256/N=128), then runs the
three lower-triangular tiles with two alternating L0C accumulators so the next
tile's GEMM can overlap the previous tile's AIV epilogue:

    tile (0,0) D -> tile (1,0) L -> tile (1,1) D -> tile (0,1) U (exact zeros)

``D`` keeps the Stage-3 exact ``select`` mask (``dual_copy`` L0C->UB, SIMT
mask+cast, ``dual_copy`` UB->GM); ``L`` is one Cube fixpipe store; ``U`` is a
zero block.  Other geometries (``nlt != 2``) fall back to the frozen Stage-3
factory so their numerics are untouched.
"""

import functools
import os
import sys

import tilelang
import tilelang.ascend.language as T
import torch


def build(base, zh=0, mask_form="select", u_mode="fill"):
    """``zh``: height of the zero UB buffer used for the strictly-upper tile
    (0 = whole ``BL/2`` half, as in the Stage-3 fill).  ``mask_form`` selects the
    diagonal-tile mask spelling: ``select`` (Stage-3 exact form), ``selhalf``
    (select in the output dtype) or ``mul`` (timing probe only -- not exact for
    non-finite accumulators, never adopted)."""
    _tile_shape = base._tile_shape
    fallback = base._ssd_chunk_coupling_kernel

    @functools.lru_cache(maxsize=32)
    def _factory(batch, num_chunks, n_groups, chunk_len, d_state, dtype):
        BL, _ = _tile_shape(chunk_len, d_state)
        if chunk_len // BL != 2:
            return fallback(batch, num_chunks, n_groups, chunk_len, d_state, dtype)

        Q, N, G = chunk_len, d_state, n_groups
        S = num_chunks * Q
        grid = batch * num_chunks * n_groups
        c_shape = (batch, S, G, N)
        cb_shape = (batch, num_chunks, G, Q, Q)
        HB = BL // 2

        @T.prim_func
        def main(
            C_mat: T.Tensor(c_shape, dtype),
            B_mat: T.Tensor(c_shape, dtype),
            cb: T.Tensor(cb_shape, dtype),
        ):
            with T.Kernel(grid) as bid:
                bb = bid // (num_chunks * G)
                cg = bid % (num_chunks * G)
                bc = cg // G
                bg = cg % G

                cl1 = T.alloc_l1((Q, N), dtype)
                bl1 = T.alloc_l1((Q, N), dtype)
                x0 = T.alloc_l0a((BL, N), dtype)
                y0 = T.alloc_l0b((BL, N), dtype)
                a0 = T.alloc_l0c((BL, BL), "float32")
                a1 = T.alloc_l0c((BL, BL), "float32")
                cv = T.alloc_shared((HB, BL), "float32")
                co = T.alloc_shared((HB, BL), dtype)
                ZH_ = zh if zh else HB
                zq = T.alloc_shared((ZH_, BL), dtype)

                T.copy(C_mat[bb, bc * Q : bc * Q + Q, bg, 0:N], cl1)
                T.copy(B_mat[bb, bc * Q : bc * Q + Q, bg, 0:N], bl1)
                sid = T.call_extern("int64", "asc_get_sub_block_id")
                cdiag = T.cast(sid, "int32") * HB

                # ---------------- tile (0, 0): diagonal -> masked --------------
                T.copy(cl1[0:BL, :], x0)
                T.copy(bl1[0:BL, :], y0)
                T.gemm(x0, y0, a0, transpose_B=True, clear_accum=True)
                T.dual_copy(a0, cv)
                with T.SimtVF(threads=128):
                    for i, j in T.Parallel(HB, BL):
                        if mask_form == "mul":
                            co[i, j] = T.cast(
                                cv[i, j] * T.cast((j - i) <= cdiag, "float32"), dtype
                            )
                        elif mask_form == "selhalf":
                            co[i, j] = T.if_then_else(
                                (j - i) <= cdiag, T.cast(cv[i, j], dtype), T.cast(0.0, dtype)
                            )
                        else:
                            co[i, j] = T.cast(
                                T.if_then_else((j - i) <= cdiag, cv[i, j], T.float32(0.0)),
                                dtype,
                            )
                T.dual_copy(co, cb[bb, bc, bg, 0:BL, 0:BL])

                # ---------------- tile (1, 0): strictly lower -> fixpipe --------
                T.copy(cl1[BL : 2 * BL, :], x0)
                T.copy(bl1[0:BL, :], y0)
                T.gemm(x0, y0, a1, transpose_B=True, clear_accum=True)
                T.copy(a1, cb[bb, bc, bg, BL : 2 * BL, 0:BL])

                # ---------------- tile (1, 1): diagonal -> masked --------------
                T.copy(cl1[BL : 2 * BL, :], x0)
                T.copy(bl1[BL : 2 * BL, :], y0)
                T.gemm(x0, y0, a0, transpose_B=True, clear_accum=True)
                T.dual_copy(a0, cv)
                with T.SimtVF(threads=128):
                    for i, j in T.Parallel(HB, BL):
                        if mask_form == "mul":
                            co[i, j] = T.cast(
                                cv[i, j] * T.cast((j - i) <= cdiag, "float32"), dtype
                            )
                        elif mask_form == "selhalf":
                            co[i, j] = T.if_then_else(
                                (j - i) <= cdiag, T.cast(cv[i, j], dtype), T.cast(0.0, dtype)
                            )
                        else:
                            co[i, j] = T.cast(
                                T.if_then_else((j - i) <= cdiag, cv[i, j], T.float32(0.0)),
                                dtype,
                            )
                T.dual_copy(co, cb[bb, bc, bg, BL : 2 * BL, BL : 2 * BL])

                # ---------------- tile (0, 1): strictly upper -> exact zeros ---
                if u_mode == "clear":
                    T.clear(a1)
                    T.copy(a1, cb[bb, bc, bg, 0:BL, BL : 2 * BL])
                elif not zh:
                    with T.SimtVF(threads=128):
                        for i, j in T.Parallel(HB, BL):
                            co[i, j] = T.cast(T.float32(0.0), dtype)
                    T.dual_copy(co, cb[bb, bc, bg, 0:BL, BL : 2 * BL])
                else:
                    with T.SimtVF(threads=128):
                        for i, j in T.Parallel(ZH_, BL):
                            zq[i, j] = T.cast(T.float32(0.0), dtype)
                    for z in range(BL // (2 * ZH_)):
                        T.dual_copy(
                            zq,
                            cb[bb, bc, bg, z * 2 * ZH_ : (z + 1) * 2 * ZH_, BL : 2 * BL],
                        )

        return tilelang.compile(main, target="ascend", out_idx=-1)

    return _factory
