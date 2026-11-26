# ---------------------------------------------------------------------------
# Launch 3 (scatter): deterministic slot from this core's cursor row
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _compile_place(numel, E, B, g):
    """Place every route of this core's slice with a derived, distinct slot.

    ``slot = cursor[e]`` then bump: the cursor row already holds
    ``cumsum[e] + routes of e on earlier cores``, and the on-chip bump hands out
    the disjoint sub-range this core owns, so the per-route cost is one on-chip
    atomic plus the store the contract needs anyway -- no global atomic, no
    contention between cores.
    """
    tpc = (numel + g - 1) // g
    rounds = (tpc + _THREADS - 1) // _THREADS
    _, e_pad, _ = _fused_plan(numel, E, B)
    n_load = _align_up(max(e_pad, 1), _THREADS) // _THREADS

    @T.prim_func
    def main(
        flat: T.Tensor((numel,), "int32"),
        sorted_token_ids: T.Tensor((numel + (E + 1) * (B - 1),), "int32"),
        cursor: T.Tensor((g, e_pad), "int32"),
    ):
        with T.Kernel(g) as a, T.SimtVF(threads=_THREADS):
            s_cur = T.alloc_shared((e_pad,), "int32")
            eid = T.alloc_local([1], "int32")
            slot = T.alloc_local([1], "int32")

            for k in T.serial(n_load):
                for j in T.Parallel(_THREADS):
                    if k * _THREADS + j < e_pad:
                        s_cur[k * _THREADS + j] = cursor[a, k * _THREADS + j]
            for r in T.serial(rounds):
                for j in T.Parallel(_THREADS):
                    idx = a * tpc + r * _THREADS + j
                    if r * _THREADS + j < tpc and idx < numel:
                        eid[0] = flat[idx]
                        # explicit upper bound: eid == E would bump the cursor's
                        # padding lane (TRAP-950-scatter-eid-upper-bound)
                        if eid[0] < E:
                            slot[0] = T.atomic_add(
                                s_cur[eid[0]], T.int32(1), return_prev=True
                            )
                            sorted_token_ids[slot[0]] = idx

    return tilelang.compile(main, target="ascend", out_idx=[])
