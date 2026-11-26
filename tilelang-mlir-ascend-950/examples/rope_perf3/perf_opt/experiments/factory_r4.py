# Tuned tiling knobs of the Stage 4 kernel (module-level so the wrapper's
# baseline/perf_opt switch block can reference them, and so splice-time
# experiment branches can override them).
#   _TILE_ELEMS -- elements one sub-tile may hold (UB envelope; the compiler
#                  counts every multi-buffered version separately).
#   _STEP_BIG   -- rows one block walks when the call splits into few column
#                  chunks (a flat geometry needs taller blocks to fill 56 cores).
#   _STEP_SMALL -- rows one block walks when a row block splits into many chunks
#                  (the chunks already provide the parallelism, and the decode
#                  calls must keep enough blocks).
#   _SUB_DIV    -- sub-tile rows target: bt = step_target // _SUB_DIV (capped by
#                  the UB envelope and the wrapper's pair budget).
#   _TAB_VEC    -- per-lane vector width of the cos/sin row gather.
#   _ROT_VEC    -- per-lane vector width of the in-place rotate pass (1 = let the
#                  mapping pick; the SIMT backend caps it at 8 B anyway).
_TILE_ELEMS = 4096
_STEP_BIG = 32
_STEP_SMALL = 16
_SUB_DIV = 4
_TAB_VEC = 4
_ROT_VEC = 1
_NUM_STAGES = 3
_BUDGET_MULT = 1
_SIMD_ROT = 1     # rotate with T.simd.* vector registers (fp32/16-bit, NeoX)
_SIMD_VL = 64     # fp32 elements per vector register (256 B)
_SKIP_TABLE = globals().get("_SKIP_TABLE", 0)     # diagnostic: cos/sin = constants
_HPC_CAP = globals().get("_HPC_CAP", 4096)        # wide-layout column-chunk cap (elements)
_HPC_CAP_NARROW = globals().get("_HPC_CAP_NARROW", 1024)  # narrow-layout cap
_TOK_WIDE = globals().get("_TOK_WIDE", 2048)      # tokens >= this -> wide layout
_WIDE_TILE_BYTES = globals().get("_WIDE_TILE_BYTES", 16384)  # wide sub-tile byte budget


def _rope_positions_kernel(
    query_heads: int,
    key_heads: int,
    head_size: int,
    rotary_dim: int,
    max_position: int,
    dtype: str,
    position_dtype: str,
    offset_dtype: str | None,
    neox: bool,
    rot_vec: int,
    tail_vec: int,
    row_vec: int,
):
    """The frozen factory: ``(units_per_block, staged)`` returns the compiled program.

    Candidate ``c5`` (Stage 4 tuned): one UB buffer per block walks ``nsub``
    sub-tiles of ``bt`` rows in a two-stage ``T.Pipelined`` loop (``T.copy``
    GM->UB -> one in-place rotate pass -> ``T.copy`` UB->GM), and the cos/sin
    gather for the whole block span is hoisted out of that loop and moved
    ``_TAB_VEC`` columns per lane.

    Structure comes from the measurements in ``perf_opt/opt_log.md``:

    * GM traffic over ``T.copy`` instead of per-lane vector accesses: the same
      payload costs 322 us per-lane (8 B) / 245 us (16 B) vs 124 us over the DMA
      path on 950PR, so the tile must not be moved by the vector pipe.
    * One UB buffer rotated in place (Stage 3 used two buffers and five UB
      passes for one read + one write), and the copied tail columns need no
      handling at all because they are never touched.
    * Bigger blocks: the per-block fixed cost dominates small tiles (a 4-row
      block: 292 us, 16 rows: 206 us, 32 rows: 150 us, 64 rows: 142 us on the
      llama3-8b-b2-2k case), but a block taller than the call's row count would
      starve the 56 vector cores on the decode cases, so the row step is chosen
      from the number of column chunks: few chunks -> taller blocks.
    * The table gather (two guarded 4 B per-lane GM reads with an int64 row
      scale) is the largest single remaining cost when done per sub-tile;
      hoisting it to the block and widening it to 4 columns per lane takes the
      dominant case from 195.7 us to 150.4 us.

    ``units_per_block`` stays the wrapper's pair budget (2048) and is used only
    as a floor for the sub-tile pair count; the tiling itself follows the tuned
    constants above.  ``rot_vec`` / ``tail_vec`` / ``row_vec`` selected CUDA
    access widths and are accepted but unused here; ``staged`` accepts both
    values and compiles this one program.
    """
    if rotary_dim <= 0 or rotary_dim % 2 or rotary_dim > head_size:
        raise ValueError("rotary_dim must be positive, even and <= head_size")
    if query_heads <= 0 or key_heads < 0:
        raise ValueError("query_heads must be positive and key_heads not negative")
    if max_position <= 0:
        raise ValueError("max_position must be positive")
    if dtype not in _TORCH_DTYPE:
        raise ValueError(f"dtype: {dtype!r} is not one of {tuple(_TORCH_DTYPE)}")
    if position_dtype not in ("int32", "int64"):
        raise ValueError(f"position_dtype: {position_dtype!r} is not int32 or int64")
    if offset_dtype not in (None, "int32", "int64"):
        raise ValueError(f"offset_dtype: {offset_dtype!r} is not int32, int64 or None")
    if offset_dtype is not None and not key_heads:
        raise ValueError("offsets: an offsets call also carries a key")

    dim, rotary = head_size, rotary_dim
    half = rotary // 2

    # A 2D UB buffer's row pitch must be a whole number of 32 B sectors, or the
    # device rejects the ``T.copy`` with error 507035 (no compile diagnostic).
    # ``head_size``/``rotary_dim`` may be odd or non-pow2 (65 / 24 / 2), so the
    # staging buffers are padded to the sector while every copy and every lane
    # access still works on the logical ``cols`` / ``half`` extent.
    _sector = 32 // _TORCH_DTYPE[dtype].itemsize

    def _pad(n):
        return max(_sector, ((n + _sector - 1) // _sector) * _sector)

    def _vf_threads(extent):
        """Threads for the in-place rotate VF: the largest power of two <= 256 that
        divides the pass's lane count.

        The rotate pass reads and writes one UB buffer, so the pipeline puts a
        shared-memory sync into it; such a sync can only be lowered when every
        participating warp is fully active, i.e. when the lane count is a whole
        number of warps.  Benchmark geometries have 512..2048 lanes per pass and
        keep 256 threads; odd ones (one key head with rotary_dim 24 -> 48 lanes)
        would otherwise leave a half warp and fail to compile (ThreadSync
        "participating threads are not warp-uniform").
        """
        t = _THREADS
        while t > 1 and extent % t:
            t //= 2
        return t

    def _hpc(heads, cap):
        return max(d for d in range(1, min(max(1, cap // dim), heads) + 1) if heads % d == 0)

    def _layout(cap):
        hq = _hpc(query_heads, cap)
        hk = _hpc(key_heads, cap) if key_heads else 0
        return hq, hq * dim, hk, hk * dim, query_heads // hq, (key_heads // hk if key_heads else 0)

    hpc_q, cols_q, hpc_k, cols_k, nch_q, nch_k = _layout(_HPC_CAP)          # wide
    hpc_qn, cols_qn, hpc_kn, cols_kn, nch_qn, nch_kn = _layout(_HPC_CAP_NARROW)
    cols_max = max(cols_q, cols_k, cols_qn, cols_kn)
    q_width, k_width = query_heads * dim, key_heads * dim
    has_off = offset_dtype is not None
    out_idx = [6, 7] if has_off else [5, 6] if key_heads else [4]
    tvec = _TAB_VEC if (_TAB_VEC > 1 and half % _TAB_VEC == 0) else 1
    rvec = _ROT_VEC if (neox and _ROT_VEC > 1 and half % _ROT_VEC == 0) else 1

    def _program(units_per_block, staged):
        # Rows one block walks: a row block splits into ``nch_q + nch_k`` column
        # chunks, so a geometry with few chunks needs taller blocks to fill the
        # 56 vector cores -- and one with many chunks must keep the blocks small
        # or the decode calls starve.
        units = nch_q + nch_k
        step_target = _STEP_SMALL if units > 5 else _STEP_BIG
        # sub-tile height follows the narrow layout: it is the layout every call can
        # take, so a wide-only geometry must not shrink the shared row step.
        ub_rows = max(1, _TILE_ELEMS // max(cols_qn, cols_kn))
        budget_rows = max(1, int(units_per_block) * _BUDGET_MULT // (hpc_qn * half))
        bt = max(1, min(step_target // _SUB_DIV, ub_rows, max(budget_rows, 1)))
        nsub = max(1, (step_target + bt - 1) // bt)
        step = bt * nsub  # rows one block walks (shared by both layouts)
        # The wide layout takes one whole token row per sub-tile (a single
        # contiguous DMA), so its sub-tile is as tall as a 16 KiB tile allows.
        item_bytes = _TORCH_DTYPE[dtype].itemsize
        bt_w = max(1, min(step, _WIDE_TILE_BYTES // max(1, max(cols_q, cols_k) * item_bytes)))
        nsub_w = max(1, step // bt_w)
        wide = _TORCH_DTYPE[dtype].itemsize == 4
        simd_rot = bool(_SIMD_ROT) and neox and (half % _SIMD_VL == 0)
        cols_pad_w = _pad(max(cols_q, cols_k))
        cols_pad_n = _pad(max(cols_qn, cols_kn))
        cols_pad_w = _pad(max(cols_q, cols_k))
        cols_pad_n = _pad(max(cols_qn, cols_kn))
        cols_pad = _pad(cols_max)
        half_pad = _pad(half)
        tokens = T.dynamic("num_tokens")
        nrb = T.ceildiv(tokens, step)
        # Two column-chunk layouts of the same tiling: the wide one (one chunk a
        # side, cos/sin gathered once a row block, longest DMA rows) needs rows to
        # fill 56 cores, the narrow one keeps the decode calls wide in blocks.
        use_wide = bool(_TOK_WIDE) and _TORCH_DTYPE[dtype].itemsize == 4
        nblocks = (T.if_then_else(tokens >= _TOK_WIDE,
                                  nrb * (nch_q + nch_k),
                                  nrb * (nch_qn + nch_kn))
                   if use_wide else nrb * (nch_qn + nch_kn))

        @T.macro
        def side(X, Y, POS, OFF, COS, SIN, row_ub, cs_ub, sn_ub, x_ub, hpc, cols, rthreads, bt_s, nsub_s, rb, cb):
            """One tensor's column chunk at ``cb`` over ``nsub`` sub-tiles from ``rb``."""
            c0 = cb * cols
            base = rb * step
            with T.SimtVF(threads=_THREADS):
                for i in T.Parallel(step):
                    r = T.min(base + i, tokens - 1)
                    seen = T.cast(POS[r], "int64")
                    if has_off:
                        seen = seen + T.cast(OFF[r], "int64")
                    row_ub[i] = T.cast(T.max(0, T.min(seen, max_position - 1)), "int32")
                if _SKIP_TABLE:
                    for i, k in T.Parallel(step, half):
                        cs_ub[i, k] = T.cast(1.0, dtype)
                    for i, k in T.Parallel(step, half):
                        sn_ub[i, k] = T.cast(0.0, dtype)
                elif tvec > 1:
                    for i, kk in T.Parallel(step, half // tvec):
                        for v in T.vectorized(tvec):
                            cs_ub[i, kk * tvec + v] = COS[row_ub[i], kk * tvec + v]
                    for i, kk in T.Parallel(step, half // tvec):
                        for v in T.vectorized(tvec):
                            sn_ub[i, kk * tvec + v] = SIN[row_ub[i], kk * tvec + v]
                else:
                    for i, k in T.Parallel(step, half):
                        cs_ub[i, k] = COS[row_ub[i], k]
                    for i, k in T.Parallel(step, half):
                        sn_ub[i, k] = SIN[row_ub[i], k]
            for s in T.Pipelined(nsub_s, num_stages=_NUM_STAGES):
                row0 = base + s * bt_s
                nr = T.min(bt_s, tokens - row0)
                T.copy(X[row0 : row0 + nr, c0 : c0 + cols], x_ub[0:nr, 0:cols])
                if simd_rot:
                    with T.SimdVF():
                        full = T.simd.pset(32)
                        for i in T.serial(bt_s):
                            for ch in T.serial(half // _SIMD_VL):
                                if wide:
                                    cs = T.simd.vld(cs_ub[s * bt_s + i, ch * _SIMD_VL])
                                    sn = T.simd.vld(sn_ub[s * bt_s + i, ch * _SIMD_VL])
                                else:
                                    cs = T.simd.vcvt(T.simd.vld(cs_ub[s * bt_s + i, ch * _SIMD_VL],
                                                                dist="UNPK_B16"), "float32", part=0)
                                    sn = T.simd.vcvt(T.simd.vld(sn_ub[s * bt_s + i, ch * _SIMD_VL],
                                                                dist="UNPK_B16"), "float32", part=0)
                                for h in T.serial(hpc):
                                    lo = h * dim + ch * _SIMD_VL
                                    hi = lo + half
                                    if wide:
                                        x_lo = T.simd.vld(x_ub[i, lo])
                                        x_hi = T.simd.vld(x_ub[i, hi])
                                    else:
                                        x_lo = T.simd.vcvt(T.simd.vld(x_ub[i, lo], dist="UNPK_B16"),
                                                           "float32", part=0)
                                        x_hi = T.simd.vcvt(T.simd.vld(x_ub[i, hi], dist="UNPK_B16"),
                                                           "float32", part=0)
                                    r_lo = T.simd.vsub(T.simd.vmul(x_lo, cs), T.simd.vmul(x_hi, sn))
                                    r_hi = T.simd.vadd(T.simd.vmul(x_hi, cs), T.simd.vmul(x_lo, sn))
                                    if wide:
                                        T.simd.vsts(x_ub[i, lo], r_lo)
                                        T.simd.vsts(x_ub[i, hi], r_hi)
                                    else:
                                        T.simd.vsts(x_ub[i, lo],
                                                    T.simd.vcvt(r_lo, dtype, part=0, sat=False),
                                                    mask=full, dist="PK_B32")
                                        T.simd.vsts(x_ub[i, hi],
                                                    T.simd.vcvt(r_hi, dtype, part=0, sat=False),
                                                    mask=full, dist="PK_B32")
                else:
                    with T.SimtVF(threads=rthreads):
                        if rvec > 1:
                            for i, h, kk in T.Parallel(bt, hpc, half // rvec):
                                for v in T.vectorized(rvec):
                                    k = kk * rvec + v
                                    lo = h * dim + k
                                    hi = lo + half
                                    cs = T.cast(cs_ub[s * bt_s + i, k], "float32")
                                    sn = T.cast(sn_ub[s * bt_s + i, k], "float32")
                                    x_lo = T.cast(x_ub[i, lo], "float32")
                                    x_hi = T.cast(x_ub[i, hi], "float32")
                                    x_ub[i, lo] = T.cast(x_lo * cs - x_hi * sn, dtype)
                                    x_ub[i, hi] = T.cast(x_hi * cs + x_lo * sn, dtype)
                        else:
                            for i, p in T.Parallel(bt, hpc * half):
                                h = p // half
                                k = p - h * half
                                lo = h * dim + (k if neox else 2 * k)
                                hi = lo + (half if neox else 1)
                                cs = T.cast(cs_ub[s * bt_s + i, k], "float32")
                                sn = T.cast(sn_ub[s * bt_s + i, k], "float32")
                                x_lo = T.cast(x_ub[i, lo], "float32")
                                x_hi = T.cast(x_ub[i, hi], "float32")
                                x_ub[i, lo] = T.cast(x_lo * cs - x_hi * sn, dtype)
                                x_ub[i, hi] = T.cast(x_hi * cs + x_lo * sn, dtype)
                T.copy(x_ub[0:nr, 0:cols], Y[row0 : row0 + nr, c0 : c0 + cols])

        @T.macro
        def run(XQ, YQ, XK, YK, POS, OFF, COS, SIN, row_ub, cs_ub, sn_ub, x_ub, bx,
                nchq, nchk, hpcq, colsq, hpck, colsk, bt_r, nsub_r, nchq_r=None, nchk_r=None):
            """Send one block to the query or the key side of one layout."""
            if bx < nrb * nchq:
                side(XQ, YQ, POS, OFF, COS, SIN, row_ub, cs_ub, sn_ub, x_ub,
                     hpcq, colsq, _vf_threads(bt_r * hpcq * half), bt_r, nsub_r,
                     bx // (nchq_r or nchq), bx % (nchq_r or nchq))
            else:
                t = bx - nrb * nchq
                side(XK, YK, POS, OFF, COS, SIN, row_ub, cs_ub, sn_ub, x_ub,
                     hpck, colsk, _vf_threads(bt_r * hpck * half), bt_r, nsub_r,
                     t // nchk, t % nchk)

        if has_off:

            @T.prim_func
            def main(
                positions: T.Tensor((tokens,), position_dtype),
                query: T.Tensor((tokens, q_width), dtype),
                key: T.Tensor((tokens, k_width), dtype),
                offsets: T.Tensor((tokens,), offset_dtype),
                cos_table: T.Tensor((max_position, half), dtype),
                sin_table: T.Tensor((max_position, half), dtype),
                query_out: T.Tensor((tokens, q_width), dtype),
                key_out: T.Tensor((tokens, k_width), dtype),
            ) -> None:
                with T.Kernel(nblocks) as bx:
                    x_ub_w = T.alloc_shared((bt_w, cols_pad_w), dtype) if use_wide else None
                    x_ub_n = T.alloc_shared((bt, cols_pad_n), dtype)
                    cs_ub = T.alloc_shared((step, half_pad), dtype)
                    sn_ub = T.alloc_shared((step, half_pad), dtype)
                    row_ub = T.alloc_shared((step,), "int32")
                    if use_wide and tokens >= _TOK_WIDE:
                        run(query, query_out, key, key_out, positions, offsets,
                            cos_table, sin_table, row_ub, cs_ub, sn_ub, x_ub_w, bx,
                            nch_q, nch_k, hpc_q, cols_q, hpc_k, cols_k, bt_w, nsub_w)
                    else:
                        run(query, query_out, key, key_out, positions, offsets,
                            cos_table, sin_table, row_ub, cs_ub, sn_ub, x_ub_n, bx,
                            nch_qn, nch_kn, hpc_qn, cols_qn, hpc_kn, cols_kn, bt, nsub)

        elif key_heads:

            @T.prim_func
            def main(
                positions: T.Tensor((tokens,), position_dtype),
                query: T.Tensor((tokens, q_width), dtype),
                key: T.Tensor((tokens, k_width), dtype),
                cos_table: T.Tensor((max_position, half), dtype),
                sin_table: T.Tensor((max_position, half), dtype),
                query_out: T.Tensor((tokens, q_width), dtype),
                key_out: T.Tensor((tokens, k_width), dtype),
            ) -> None:
                with T.Kernel(nblocks) as bx:
                    x_ub_w = T.alloc_shared((bt_w, cols_pad_w), dtype) if use_wide else None
                    x_ub_n = T.alloc_shared((bt, cols_pad_n), dtype)
                    cs_ub = T.alloc_shared((step, half_pad), dtype)
                    sn_ub = T.alloc_shared((step, half_pad), dtype)
                    row_ub = T.alloc_shared((step,), "int32")
                    if use_wide and tokens >= _TOK_WIDE:
                        run(query, query_out, key, key_out, positions, positions,
                            cos_table, sin_table, row_ub, cs_ub, sn_ub, x_ub_w, bx,
                            nch_q, nch_k, hpc_q, cols_q, hpc_k, cols_k, bt_w, nsub_w)
                    else:
                        run(query, query_out, key, key_out, positions, positions,
                            cos_table, sin_table, row_ub, cs_ub, sn_ub, x_ub_n, bx,
                            nch_qn, nch_kn, hpc_qn, cols_qn, hpc_kn, cols_kn, bt, nsub)

        else:

            @T.prim_func
            def main(
                positions: T.Tensor((tokens,), position_dtype),
                query: T.Tensor((tokens, q_width), dtype),
                cos_table: T.Tensor((max_position, half), dtype),
                sin_table: T.Tensor((max_position, half), dtype),
                query_out: T.Tensor((tokens, q_width), dtype),
            ) -> None:
                with T.Kernel(nblocks) as bx:
                    x_ub_w = T.alloc_shared((bt_w, cols_pad_w), dtype) if use_wide else None
                    x_ub_n = T.alloc_shared((bt, cols_pad_n), dtype)
                    cs_ub = T.alloc_shared((step, half_pad), dtype)
                    sn_ub = T.alloc_shared((step, half_pad), dtype)
                    row_ub = T.alloc_shared((step,), "int32")
                    if use_wide and tokens >= _TOK_WIDE:
                        run(query, query_out, query, query_out, positions, positions,
                            cos_table, sin_table, row_ub, cs_ub, sn_ub, x_ub_w, bx,
                            nch_q, 1, hpc_q, cols_q, hpc_q, cols_q, bt_w, nsub_w, nch_q, nch_k)
                    else:
                        run(query, query_out, query, query_out, positions, positions,
                            cos_table, sin_table, row_ub, cs_ub, sn_ub, x_ub_n, bx,
                            nch_qn, 1, hpc_qn, cols_qn, hpc_qn, cols_qn, bt, nsub)

        return tilelang.compile(main, target="ascend", out_idx=out_idx)

    return _program


