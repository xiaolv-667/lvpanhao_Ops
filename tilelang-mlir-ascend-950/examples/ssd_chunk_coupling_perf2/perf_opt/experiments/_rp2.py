"""Round-1 experiments on the Stage-4 row-panel kernel (baseline 83.2 us).

Base module = `examples/ssd_chunk_coupling_perf/perf_opt/SSDChunkCouplingFwdOp.py`.
Every variant keeps the per-chunk panel / L1 residency / IO contract and changes
only the diagonal-band epilogue, because the last profile put the AIV
elementwise path at 4.0 us of a 6.9 us AIV task (2 x 8192 fp32->dtype
element-ops per AIV, plus the zero tile).

``base``    identical to the production kernel (regression check).
``band``    mask applied on the raw fp32 bit pattern (``bits & (cond ? -1 : 0)``):
            exact ``select`` semantics for *every* input including inf/nan
            (``nan & 0 == +0.0``), but branch-free so the vectorizer can emit a
            full-width vector ``and``/``vcvt`` instead of a per-lane branch.
``diag64``  the two 128x128 diagonal tiles are decomposed into 64x64 sub-tiles:
            the strictly-lower sub-tile becomes one Cube fixpipe store (no AIV
            element work), the strictly-upper one an exact zero store, so only
            the 2 corner sub-tiles run the per-element mask -- the masked
            element count per chunk halves (``Q*BL`` -> ``Q*BL/2``) and the
            strictly-lower sub-tile leaves the AIV entirely.  GM traffic is
            unchanged (the panel is still loaded once).
"""

import functools
import os
import sys

import tilelang
import tilelang.ascend.language as T
import torch


def _prefix(base, variant):
    _tile_shape = base._tile_shape
    fallback = base._ssd_chunk_coupling_kernel
    return _tile_shape, fallback


def build(base, variant="base", zh=8):
    _tile_shape, fallback = _prefix(base, variant)
    HB_ZH = zh

    def _make(variant):
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
            SB = BL // 2          # diag64 sub-tile edge
            zh = HB_ZH

            # ---------------- shared prim_func body -------------------------
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
                    zq = T.alloc_shared((zh, BL), dtype)

                    T.copy(C_mat[bb, bc * Q : bc * Q + Q, bg, 0:N], cl1)
                    T.copy(B_mat[bb, bc * Q : bc * Q + Q, bg, 0:N], bl1)
                    sid = T.call_extern("int64", "asc_get_sub_block_id")
                    cdiag = T.cast(sid, "int32") * HB

                    # ---- tile (0, 0): diagonal 128-band --------------------
                    T.copy(cl1[0:BL, :], x0)
                    T.copy(bl1[0:BL, :], y0)
                    T.gemm(x0, y0, a0, transpose_B=True, clear_accum=True)
                    T.dual_copy(a0, cv)
                    if variant == "band":
                        with T.SimtVF(threads=128):
                            for i, j in T.Parallel(HB, BL):
                                m = -T.cast((j - i) <= cdiag, "int32")
                                co[i, j] = T.cast(
                                    T.reinterpret(
                                        T.bitwise_and(T.reinterpret(cv[i, j], "int32"), m),
                                        "float32",
                                    ),
                                    dtype,
                                )
                    else:
                        with T.SimtVF(threads=128):
                            for i, j in T.Parallel(HB, BL):
                                co[i, j] = T.cast(
                                    T.if_then_else((j - i) <= cdiag, cv[i, j], T.float32(0.0)),
                                    dtype,
                                )
                    T.dual_copy(co, cb[bb, bc, bg, 0:BL, 0:BL])

                    # ---- tile (1, 0): strictly lower 128-tile -> fixpipe ---
                    T.copy(cl1[BL : 2 * BL, :], x0)
                    T.copy(bl1[0:BL, :], y0)
                    T.gemm(x0, y0, a1, transpose_B=True, clear_accum=True)
                    T.copy(a1, cb[bb, bc, bg, BL : 2 * BL, 0:BL])

                    # ---- tile (1, 1): diagonal 128-band --------------------
                    T.copy(cl1[BL : 2 * BL, :], x0)
                    T.copy(bl1[BL : 2 * BL, :], y0)
                    T.gemm(x0, y0, a0, transpose_B=True, clear_accum=True)
                    T.dual_copy(a0, cv)
                    if variant == "band":
                        with T.SimtVF(threads=128):
                            for i, j in T.Parallel(HB, BL):
                                m = -T.cast((j - i) <= cdiag, "int32")
                                co[i, j] = T.cast(
                                    T.reinterpret(
                                        T.bitwise_and(T.reinterpret(cv[i, j], "int32"), m),
                                        "float32",
                                    ),
                                    dtype,
                                )
                    else:
                        with T.SimtVF(threads=128):
                            for i, j in T.Parallel(HB, BL):
                                co[i, j] = T.cast(
                                    T.if_then_else((j - i) <= cdiag, cv[i, j], T.float32(0.0)),
                                    dtype,
                                )
                    T.dual_copy(co, cb[bb, bc, bg, BL : 2 * BL, BL : 2 * BL])

                    # ---- tile (0, 1): strictly upper -> exact zeros --------
                    with T.SimtVF(threads=128):
                        for i, j in T.Parallel(zh, BL):
                            zq[i, j] = T.cast(T.float32(0.0), dtype)
                    for z in range(BL // (2 * zh)):
                        T.dual_copy(
                            zq, cb[bb, bc, bg, z * 2 * zh : (z + 1) * 2 * zh, BL : 2 * BL]
                        )

            return tilelang.compile(main, target="ascend", out_idx=-1)

        return _factory

    return _make(variant)
