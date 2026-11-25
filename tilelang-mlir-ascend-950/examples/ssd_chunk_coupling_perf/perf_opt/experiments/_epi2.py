"""Round-2 epilogue experiments: block-class specialised, strip-decomposed.

Tile classes of one ``(b, c, g)`` chunk (``Q``, tile ``BL``):

* **U** strictly upper  (``s0 > l0 + BL - 1``)  -> exact zeros, no GEMM
* **L** strictly lower  (``l0 > s0``)           -> ``cast(acc)``, one Cube fixpipe
* **D** diagonal        (``l0 == s0``)          -> lower triangle kept, upper exact 0

For **D** the tile is cut into ``BL/SH`` horizontal strips of ``SH`` rows.  Each
strip owns three *disjoint* GM rectangles, so the Cube fixpipe and the AIV never
write the same element (no cross-engine write ordering needed):

* ``rows[SH k, +SH) x cols[0, SH k)``        -> Cube fixpipe ``cast(acc)``  (keep)
* ``rows[SH k, +SH) x cols[SH(k+1), BL)``    -> AIV store from a zero buffer
* ``rows[SH k, +SH) x cols[SH k, SH(k+1))``  -> AIV SIMT masked ``SH x SH`` corner

Only the diagonal corners (``BL*SH`` elements instead of ``BL*BL``) run through
the per-element mask, and the mask keeps the exact ``select`` semantics of the
Stage-3 kernel (no ``x * 0``).
"""

import functools
import os
import sys

import tilelang
import tilelang.ascend.language as T
import torch

MODE_FILL_UL = "fill_ul"           # U: clear+dual_copy (round-1 refinement only)
MODE_STRIP = "strip"               # U clear, L fixpipe, D strip decomposition
MODE_STRIP_NODIAGFIX = "strip_nd"  # as MODE_STRIP but D keeps the plain SIMT mask


def build(base, mode, strip=16):
    _tile_shape = base._tile_shape
    SH = strip

    @functools.lru_cache(maxsize=32)
    def _factory(batch, num_chunks, n_groups, chunk_len, d_state, dtype):
        BL, BK = _tile_shape(chunk_len, d_state)
        if BL % SH or (BL // 2) % SH:
            raise ValueError(f"BL={BL} not compatible with strip={SH}")
        Q, N, G = chunk_len, d_state, n_groups
        S = num_chunks * Q
        nlt = Q // BL
        grid = batch * num_chunks * n_groups * nlt * nlt
        k_trips = -(-N // BK)
        c_shape = (batch, S, G, N)
        cb_shape = (batch, num_chunks, G, Q, Q)
        HB = BL // 2
        NS = BL // SH      # strips per tile
        NSH = HB // SH     # strips per AIV half

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
                cv = T.alloc_shared((HB, BL), "float32")
                co = T.alloc_shared((HB, BL), dtype)
                zs = T.alloc_shared((SH, BL), dtype)
                mc = T.alloc_shared((HB, SH), dtype)
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

                sid = T.call_extern("int64", "asc_get_sub_block_id")
                hb = T.cast(sid, "int32") * HB      # AIV-local row base inside tile
                cdiag = l0 + hb - s0

                if s0 > l0 + BL - 1:
                    # ---- U: exact zero block (the K loop ran 0 trips) ------------
                    T.clear(co)
                    T.dual_copy(co, cb[bb, bc, bg, l0 : l0 + BL, s0 : s0 + BL])
                else:
                    if l0 > s0:
                        # ---- L: whole tile, Cube fixpipe cast --------------------
                        T.copy(acc, cb[bb, bc, bg, l0 : l0 + BL, s0 : s0 + BL])
                    else:
                        # ---- D: strip decomposition ------------------------------
                        T.clear(zs)
                        T.dual_copy(acc, cv)
                        if mode == MODE_STRIP_NODIAGFIX:
                            with T.SimtVF(threads=128):
                                for i, j in T.Parallel(HB, BL):
                                    co[i, j] = T.cast(
                                        T.if_then_else((j - i) <= cdiag, cv[i, j], T.float32(0.0)),
                                        dtype,
                                    )
                            T.dual_copy(co, cb[bb, bc, bg, l0 : l0 + BL, s0 : s0 + BL])
                        else:
                            # (a) strictly-below rectangles: Cube fixpipe, all rows
                            for k in range(1, NS):
                                T.copy(
                                    acc[k * SH : (k + 1) * SH, 0 : k * SH],
                                    cb[bb, bc, bg,
                                       l0 + k * SH : l0 + (k + 1) * SH,
                                       s0 : s0 + k * SH],
                                )
                            # (b) per-AIV diagonal-corner values (masked, exact)
                            with T.SimtVF(threads=128):
                                for i, t in T.Parallel(HB, SH):
                                    mc[i, t] = T.cast(
                                        T.if_then_else(
                                            t <= i % SH,
                                            cv[i, (i // SH) * SH + t],
                                            T.float32(0.0),
                                        ),
                                        dtype,
                                    )
                            # (c) strictly-above zero rectangles + corner, own rows
                            for j in range(NSH):
                                w = BL - hb - SH * (j + 1)
                                # the last strip of AIV 1 keeps its whole row, so
                                # the zero rectangle is empty there.
                                if w > 0:
                                    T.copy(
                                        zs[0:SH, 0:w],
                                        cb[bb, bc, bg,
                                           l0 + hb + j * SH : l0 + hb + (j + 1) * SH,
                                           s0 + hb + SH * (j + 1) : s0 + BL],
                                    )
                                T.copy(
                                    mc[j * SH : (j + 1) * SH, 0:SH],
                                    cb[bb, bc, bg,
                                       l0 + hb + j * SH : l0 + hb + (j + 1) * SH,
                                       s0 + hb + j * SH : s0 + hb + (j + 1) * SH],
                                )

        return tilelang.compile(main, target="ascend", out_idx=-1)

    return _factory
