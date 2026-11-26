"""``diag64``+``band`` variant (single spelling, no switches): 128x128 diagonal bands decomposed into 64x64 sub-tiles.

Per chunk (Q=256, BL=128) the lower-triangular tile set is

    (0,0) diagonal band   -> 64x64 sub-tiles (0,0) masked, (1,0) fixpipe,
                             (1,1) masked, (0,1) exact zeros
    (1,0) strictly lower  -> one 128x128 fixpipe store (unchanged)
    (1,1) diagonal band   -> same decomposition as (0,0)
    (0,1) strictly upper  -> exact zeros (unchanged)

Only the two 64x64 corner sub-tiles per band run the per-element mask, so the
masked element count per chunk halves (Q*BL -> Q*BL/2) and two of the four
previously masked 64x64 quadrants leave the AIV entirely (one fixpipe, one zero
store).  GM reads stay at one C/B panel per chunk; the 64-row GEMMs read their
operands from the already resident L1 panel.
"""

import functools
import os
import sys

import tilelang
import tilelang.ascend.language as T
import torch


def build(base, zh=8):
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
        HB = BL // 2        # 64 rows per AIV half
        SB = BL // 2        # 64: sub-tile edge
        SH = SB // 2        # 32 rows per AIV for a 64x64 sub-tile

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

                cl1a = T.alloc_l1((BL, N), dtype)
                cl1b = T.alloc_l1((BL, N), dtype)
                bl1a = T.alloc_l1((BL, N), dtype)
                bl1b = T.alloc_l1((BL, N), dtype)
                x0 = T.alloc_l0a((BL, N), dtype)
                y0 = T.alloc_l0b((BL, N), dtype)
                x1 = T.alloc_l0a((SB, N), dtype)
                y1 = T.alloc_l0b((SB, N), dtype)
                a1 = T.alloc_l0c((BL, BL), "float32")
                b0 = T.alloc_l0c((SB, SB), "float32")
                b1 = T.alloc_l0c((SB, SB), "float32")
                b2 = T.alloc_l0c((SB, SB), "float32")
                cvh = T.alloc_shared((SH, SB), "float32")
                coh = T.alloc_shared((SH, SB), dtype)
                zq = T.alloc_shared((zh, BL), dtype)
                zc = T.alloc_shared((zh, SB), dtype)

                # only the first half of each panel is needed to start band 0;
                # the second half loads while band 0 computes.
                T.copy(C_mat[bb, bc * Q : bc * Q + BL, bg, 0:N], cl1a)
                T.copy(B_mat[bb, bc * Q : bc * Q + BL, bg, 0:N], bl1a)
                sid = T.call_extern("int64", "asc_get_sub_block_id")
                cd = T.cast(sid, "int32") * SH      # corner keep boundary

                # ================= diagonal band at rows/cols [0, BL) ========
                # (0,0) corner 64x64 -> masked
                T.copy(cl1a[0:SB, :], x1)
                T.copy(bl1a[0:SB, :], y1)
                T.gemm(x1, y1, b0, transpose_B=True, clear_accum=True)
                T.dual_copy(b0, cvh)
                with T.SimtVF(threads=128):
                    for i, j in T.Parallel(SH, SB):
                        coh[i, j] = T.cast(
                            T.reinterpret(
                                T.bitwise_and(
                                    T.reinterpret(cvh[i, j], "int32"),
                                    -T.cast((j - i) <= cd, "int32"),
                                ),
                                "float32",
                            ),
                            dtype,
                        )
                T.dual_copy(coh, cb[bb, bc, bg, 0:SB, 0:SB])
                # (1,0) 64x64 strictly lower -> fixpipe
                T.copy(cl1a[SB : 2 * SB, :], x1)
                T.copy(bl1a[0:SB, :], y1)
                T.gemm(x1, y1, b1, transpose_B=True, clear_accum=True)
                T.copy(b1, cb[bb, bc, bg, SB : 2 * SB, 0:SB])
                # (1,1) corner 64x64 -> masked
                T.copy(cl1a[SB : 2 * SB, :], x1)
                T.copy(bl1a[SB : 2 * SB, :], y1)
                T.gemm(x1, y1, b2, transpose_B=True, clear_accum=True)
                T.dual_copy(b2, cvh)
                with T.SimtVF(threads=128):
                    for i, j in T.Parallel(SH, SB):
                        coh[i, j] = T.cast(
                            T.reinterpret(
                                T.bitwise_and(
                                    T.reinterpret(cvh[i, j], "int32"),
                                    -T.cast((j - i) <= cd, "int32"),
                                ),
                                "float32",
                            ),
                            dtype,
                        )
                T.dual_copy(coh, cb[bb, bc, bg, SB : 2 * SB, SB : 2 * SB])
                # (0,1) 64x64 strictly upper -> exact zeros
                with T.SimtVF(threads=128):
                    for i, j in T.Parallel(zh, SB):
                        zc[i, j] = T.cast(T.float32(0.0), dtype)
                for z in range(SB // (2 * zh)):
                    T.dual_copy(zc, cb[bb, bc, bg, z * 2 * zh : (z + 1) * 2 * zh, SB : 2 * SB])

                T.copy(C_mat[bb, bc * Q + BL : bc * Q + 2 * BL, bg, 0:N], cl1b)
                T.copy(B_mat[bb, bc * Q + BL : bc * Q + 2 * BL, bg, 0:N], bl1b)

                # ================= tile (1, 0): strictly lower 128 -> fixpipe =
                T.copy(cl1b, x0)
                T.copy(bl1a, y0)
                T.gemm(x0, y0, a1, transpose_B=True, clear_accum=True)
                T.copy(a1, cb[bb, bc, bg, BL : 2 * BL, 0:BL])

                # ================= diagonal band at rows/cols [BL, 2*BL) =====
                T.copy(cl1b[0:SB, :], x1)
                T.copy(bl1b[0:SB, :], y1)
                T.gemm(x1, y1, b0, transpose_B=True, clear_accum=True)
                T.dual_copy(b0, cvh)
                with T.SimtVF(threads=128):
                    for i, j in T.Parallel(SH, SB):
                        coh[i, j] = T.cast(
                            T.reinterpret(
                                T.bitwise_and(
                                    T.reinterpret(cvh[i, j], "int32"),
                                    -T.cast((j - i) <= cd, "int32"),
                                ),
                                "float32",
                            ),
                            dtype,
                        )
                T.dual_copy(coh, cb[bb, bc, bg, BL : BL + SB, BL : BL + SB])

                T.copy(cl1b[SB : 2 * SB, :], x1)
                T.copy(bl1b[0:SB, :], y1)
                T.gemm(x1, y1, b1, transpose_B=True, clear_accum=True)
                T.copy(b1, cb[bb, bc, bg, BL + SB : 2 * BL, BL : BL + SB])

                T.copy(cl1b[SB : 2 * SB, :], x1)
                T.copy(bl1b[SB : 2 * SB, :], y1)
                T.gemm(x1, y1, b2, transpose_B=True, clear_accum=True)
                T.dual_copy(b2, cvh)
                with T.SimtVF(threads=128):
                    for i, j in T.Parallel(SH, SB):
                        coh[i, j] = T.cast(
                            T.reinterpret(
                                T.bitwise_and(
                                    T.reinterpret(cvh[i, j], "int32"),
                                    -T.cast((j - i) <= cd, "int32"),
                                ),
                                "float32",
                            ),
                            dtype,
                        )
                T.dual_copy(coh, cb[bb, bc, bg, BL + SB : 2 * BL, BL + SB : 2 * BL])

                with T.SimtVF(threads=128):
                    for i, j in T.Parallel(zh, SB):
                        zc[i, j] = T.cast(T.float32(0.0), dtype)
                for z in range(SB // (2 * zh)):
                    T.dual_copy(
                        zc, cb[bb, bc, bg, BL + z * 2 * zh : BL + (z + 1) * 2 * zh, BL + SB : 2 * BL]
                    )

                # ================= tile (0, 1): strictly upper 128 -> zeros ==
                with T.SimtVF(threads=128):
                    for i, j in T.Parallel(zh, BL):
                        zq[i, j] = T.cast(T.float32(0.0), dtype)
                for z in range(BL // (2 * zh)):
                    T.dual_copy(zq, cb[bb, bc, bg, z * 2 * zh : (z + 1) * 2 * zh, BL : 2 * BL])

        return tilelang.compile(main, target="ascend", out_idx=-1)

    return _factory
