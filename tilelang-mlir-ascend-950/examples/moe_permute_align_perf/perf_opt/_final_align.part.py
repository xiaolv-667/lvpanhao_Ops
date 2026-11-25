# ---------------------------------------------------------------------------
# Launch 2 (align): merge + padded prefix + metadata + sentinel [+ placement]
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _compile_align_stage(numel, E, B, g, fused):
    """Merge the per-core rows and emit every non-route output of the call.

    ``fused`` (single-core geometries only, a build-time constant) additionally
    runs the placement in this launch, after the sentinel fill -- the ordering
    the fused small-batch kernel already relies on inside one core.
    """
    geo = _geometry(numel, E, B)
    e_pad = geo["E_pad"]
    p_total = geo["P"]
    nb = geo["NB"]
    steps = geo["steps"]
    shift = geo["shift"]
    blkpc = (nb + g - 1) // g
    blk = max(64, _align_up(max(blkpc, 1), 64))
    epc = (E + g - 1) // g                       # experts this core expands
    e_lanes = _align_up(max(epc, 1), 64)
    share = (p_total + g - 1) // g
    n_fill = (share + FILL_TILE - 1) // FILL_TILE
    tpc = (numel + g - 1) // g
    rounds = (tpc + _THREADS - 1) // _THREADS
    bulk = g >= _BULK_MERGE_MIN_CORES

    @T.prim_func
    def main(
        flat: T.Tensor((numel,), "int32"),
        partial: T.Tensor((g, e_pad), "int32"),
        cursor: T.Tensor((g, e_pad), "int32"),
        sorted_token_ids: T.Tensor((p_total,), "int32"),
        expert_ids: T.Tensor((nb,), "int32"),
        num_tokens_post_pad: T.Tensor((1,), "int32"),
        cumsum: T.Tensor((E + 1,), "int32"),
    ):
        with T.Kernel(g) as a:
            s_all = T.alloc_shared((g, e_pad), "int32")
            s_part = T.alloc_shared((e_pad,), "int32")
            s_cnt = T.alloc_shared((e_pad,), "int32")
            s_own = T.alloc_shared((e_pad,), "int32")
            s_pad = T.alloc_shared((e_pad,), "int32")
            s_scan_a = T.alloc_shared((2 * e_pad,), "int32")
            s_scan_b = T.alloc_shared((2 * e_pad,), "int32")
            s_pref = T.alloc_shared((e_pad + 64,), "int32")
            s_cur = T.alloc_shared((e_pad,), "int32")
            s_fill = T.alloc_shared((FILL_TILE,), "int32")

            # ---- merge the per-core rows ------------------------------------
            # Large core counts: one descriptor for the whole window
            # (TRAP-950-rowload-dma-chain / PL-950-window-bulk-copy).  Small
            # ones: the per-row chain is cheaper than the window's indexed
            # accumulate.  Both keep integer addition associative, so every core
            # derives bit-identical counts.
            if bulk:
                T.copy(partial, s_all)
                with T.SimtVF(threads=e_pad):
                    for e in T.Parallel(e_pad):
                        tot = T.alloc_local([1], "int32")
                        own = T.alloc_local([1], "int32")
                        tot[0] = 0
                        own[0] = 0
                        for r in T.serial(g):
                            tot[0] = tot[0] + s_all[r, e]
                            own[0] = own[0] + T.if_then_else(r < a, s_all[r, e], 0)
                        s_cnt[e] = tot[0]
                        s_own[e] = own[0]
            else:
                with T.SimtVF(threads=e_pad):
                    for e in T.Parallel(e_pad):
                        s_cnt[e] = 0
                with T.SimtVF(threads=e_pad):
                    for e in T.Parallel(e_pad):
                        s_own[e] = 0
                for r in T.serial(g):
                    T.copy(partial[r, :], s_part)
                    with T.SimtVF(threads=e_pad):
                        for e in T.Parallel(e_pad):
                            s_cnt[e] = s_cnt[e] + s_part[e]
                            s_own[e] = s_own[e] + T.if_then_else(r < a, s_part[e], 0)

            # ---- pad each count up to a whole block -------------------------
            with T.SimtVF(threads=e_pad):
                for e in T.Parallel(e_pad):
                    pv = T.if_then_else(e < E, s_cnt[e], 0)
                    if shift is not None:
                        s_pad[e] = ((pv + (B - 1)) >> shift) << shift
                    else:
                        s_pad[e] = T.ceildiv(pv, B) * B

            # ---- inclusive doubling scan (data in the upper half, zero halo) -
            with T.SimtVF(threads=e_pad):
                for e in T.Parallel(e_pad):
                    s_scan_a[e_pad + e] = s_pad[e]
            with T.SimtVF(threads=e_pad):
                for e in T.Parallel(e_pad):
                    s_scan_a[e] = 0
            with T.SimtVF(threads=e_pad):
                for e in T.Parallel(e_pad):
                    s_scan_b[e] = 0
            with T.SimtVF(threads=e_pad):
                for e in T.Parallel(e_pad):
                    s_scan_b[e_pad + e] = 0
            for d in range(steps):
                k = 1 << d
                if d % 2 == 0:  # build-time branch
                    with T.SimtVF(threads=e_pad):
                        for e in T.Parallel(e_pad):
                            s_scan_b[e_pad + e] = (
                                s_scan_a[e_pad + e] + s_scan_a[e_pad + e - k]
                            )
                else:
                    with T.SimtVF(threads=e_pad):
                        for e in T.Parallel(e_pad):
                            s_scan_a[e_pad + e] = (
                                s_scan_b[e_pad + e] + s_scan_b[e_pad + e - k]
                            )

            # shift by one lane; the padding lanes hold `total`, a defined value
            if steps % 2 == 0:
                with T.SimtVF(threads=e_pad + 64):
                    for e in T.Parallel(e_pad + 64):
                        if e <= e_pad:
                            s_pref[e] = T.if_then_else(
                                e == 0, 0, s_scan_a[e_pad + e - 1]
                            )
            else:
                with T.SimtVF(threads=e_pad + 64):
                    for e in T.Parallel(e_pad + 64):
                        if e <= e_pad:
                            s_pref[e] = T.if_then_else(
                                e == 0, 0, s_scan_b[e_pad + e - 1]
                            )

            total = s_pref[E]
            if a == 0:
                T.copy(s_pref[0:E + 1], cumsum[0:E + 1])
                num_tokens_post_pad[0] = total

            # this core's slot cursor: its range start + routes before it
            with T.SimtVF(threads=e_pad):
                for e in T.Parallel(e_pad):
                    cursor[a, e] = s_pref[e] + s_own[e]

            # ---- expert_ids: past the padded length no block names an expert -
            blk0 = a * blkpc
            with T.SimtVF(threads=blk):
                for j in T.Parallel(blk):
                    b = blk0 + j
                    if b >= total // B and b < nb:
                        expert_ids[b] = E
            # ---- inside it, each expert expands the block range it owns ------
            e_lo = a * epc
            with T.SimtVF(threads=e_lanes):
                for j in T.Parallel(e_lanes):
                    expert = e_lo + j
                    start = T.alloc_local([1], "int32")
                    cnt = T.alloc_local([1], "int32")
                    if expert < E:
                        start[0] = s_pref[expert] // B
                        cnt[0] = s_pref[expert + 1] // B - start[0]
                        for b in T.serial(cnt[0]):
                            expert_ids[start[0] + b] = expert

            # ---- sentinel fill: no other launch writes these slots before this
            #      one retires, so the placement can overwrite freely ----------
            T.fill(s_fill, numel)
            if n_fill > 0:
                for t in T.serial(n_fill):
                    off = a * share + t * FILL_TILE
                    n = T.min(FILL_TILE, T.min((a + 1) * share, p_total) - off)
                    if n > 0:
                        T.copy(s_fill[0:n], sorted_token_ids[off:off + n])

            # ---- single-core geometries place here, after the fill ----------
            if fused:
                with T.SimtVF(threads=e_pad):
                    for e in T.Parallel(e_pad):
                        s_cur[e] = s_pref[e]
                with T.SimtVF(threads=_THREADS):
                    eid = T.alloc_local([1], "int32")
                    slot = T.alloc_local([1], "int32")
                    for r in T.serial(rounds):
                        for j in T.Parallel(_THREADS):
                            idx = a * tpc + r * _THREADS + j
                            if r * _THREADS + j < tpc and idx < numel:
                                eid[0] = flat[idx]
                                if eid[0] < E:
                                    slot[0] = T.atomic_add(
                                        s_cur[eid[0]], T.int32(1), return_prev=True
                                    )
                                    sorted_token_ids[slot[0]] = idx

    return tilelang.compile(main, target="ascend", out_idx=[])
