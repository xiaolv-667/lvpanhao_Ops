# ---------------------------------------------------------------------------
# Launch 1: per-core histogram (one on-chip atomic per route)
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _compile_hist_atomic(numel, E, B, g):
    """Every core counts its own route slice into its own int32 histogram row.

    ``numel`` lane-ops in total instead of ``numel x E`` compares; each core
    writes row ``gid`` of the scratch and touches nothing else, so no cross-core
    interaction is involved.  The route -> core split is the placement's
    (``idx = gid*tpc + r*_THREADS + j``), so both launches agree on the slices.
    """
    _, e_pad, tpc = _fused_plan(numel, E, B)
    rounds = (tpc + _THREADS - 1) // _THREADS
    n_clear = max(1, (e_pad + _THREADS - 1) // _THREADS)

    @T.prim_func
    def main(
        flat: T.Tensor((numel,), "int32"),
        partial: T.Tensor((g, e_pad), "int32"),
    ):
        with T.Kernel(g) as gid, T.SimtVF(threads=_THREADS):
            s_hist = T.alloc_shared((e_pad,), "int32")
            eid = T.alloc_local([1], "int32")

            for k in T.serial(n_clear):
                for j in T.Parallel(_THREADS):
                    if k * _THREADS + j < e_pad:
                        s_hist[k * _THREADS + j] = 0
            for r in T.serial(rounds):
                for j in T.Parallel(_THREADS):
                    idx = gid * tpc + r * _THREADS + j
                    if r * _THREADS + j < tpc and idx < numel:
                        eid[0] = flat[idx]
                        if eid[0] < E:
                            T.atomic_add(s_hist[eid[0]], T.int32(1))
            T.copy(s_hist, partial[gid, :])

    return tilelang.compile(main, target="ascend", out_idx=[])
