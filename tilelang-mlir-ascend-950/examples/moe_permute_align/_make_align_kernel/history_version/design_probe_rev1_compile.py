"""Design-time compile-only probe (revision 1) — D-3.

Static (no device) buildability check of the **revised** §3.3 pseudocode forms from
DESIGN.md v1:

  * path S skeleton with the ``bx == 0`` body *unconditional* (numel == 0 no longer
    skips the metadata writes; only the build-time ``if numel > 0`` wraps the
    GM-read chunk loop),
  * per-lane accumulator via ``T.alloc_local([1], "int32")`` instead of
    ``T.alloc_var(...)[0]``,
  * Hillis-Steele scan unrolled with python ``for d in range(N_STEPS)`` instead of a
    ``while`` inside ``@T.prim_func``,
  * left shift written over ``[0, E_pad]`` (padding lanes hold ``total``) and the
    rank-count reduce bounded by the build-time constant ``E``,
  * path L launch-2 (partial merge + same scan/rank-count) skeleton.

Run (no NPU needed, no kernel is launched):
    cd examples/moe_permute_align/_make_align_kernel
    source .agents/platforms/toolchains/ascend/env-950pr.sh
    ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python \
        history_version/design_probe_rev1_compile.py [S33|S0|L384]
"""

from __future__ import annotations

import math
import sys

import tilelang
import tilelang.ascend.language as T

FILL_TILE = 2048


def _align_up(v, m):
    return (v + m - 1) // m * m


def _steps(lanes):
    return 0 if lanes <= 1 else int(math.ceil(math.log2(lanes)))


def build_single(numel: int, E: int, B: int):
    """Path S: 1 launch, grid=2 (mirrors DESIGN.md §3.3 path S)."""
    e_pad = _align_up(E, 64)
    chunk = max(64, min(1024, _align_up(numel, 64)))
    p_total = numel + (E + 1) * (B - 1)
    nb = math.ceil(p_total / B)
    blk = _align_up(nb, 64)
    n_steps = _steps(e_pad)
    p_shift = B.bit_length() - 1 if (B & (B - 1)) == 0 else None
    half = (p_total + 1) // 2
    n_fill = (half + FILL_TILE - 1) // FILL_TILE
    n_chunk = (numel + chunk - 1) // chunk

    @T.prim_func
    def main(
        flat: T.Tensor((numel,), "int32"),
        sorted_ids: T.Tensor((p_total,), "int32"),
        expert_ids: T.Tensor((nb,), "int32"),
        npost: T.Tensor((1,), "int32"),
        cumsum: T.Tensor((E + 1,), "int32"),
    ):
        with T.Kernel(2) as bx:
            s_x = T.alloc_shared((chunk,), "int32")
            s_cnt = T.alloc_shared((e_pad,), "int32")
            s_pad = T.alloc_shared((e_pad,), "int32")
            s_scan = T.alloc_shared((2 * e_pad,), "int32")
            s_cs = T.alloc_shared((e_pad + 64,), "int32")
            s_ids = T.alloc_shared((blk,), "int32")
            s_fill = T.alloc_shared((FILL_TILE,), "int32")

            # ---- sentinel fill: unconditional, two slices (numel == 0 -> zeros)
            T.fill(s_fill, numel)
            for t in T.serial(n_fill):
                off = bx * half + t * FILL_TILE
                n = T.min(FILL_TILE, p_total - off)
                if n > 0:
                    T.copy(s_fill[0:n], sorted_ids[off : off + n])

            # ---- metadata + mapping: NO `numel > 0` guard on the writes
            if bx == 0:
                with T.SimtVF(threads=e_pad):
                    for e in T.Parallel(e_pad):
                        s_cnt[e] = 0

                if numel > 0:  # build-time python branch: only the GM read
                    for c in T.Pipelined(n_chunk, num_stages=2):
                        base = c * chunk
                        n = T.min(chunk, numel - base)
                        T.copy(flat[base : base + n], s_x[0:n])
                        with T.SimtVF(threads=e_pad):
                            for e in T.Parallel(e_pad):
                                acc = T.alloc_local([1], "int32")
                                acc[0] = 0
                                for i in T.serial(n):
                                    acc[0] = acc[0] + T.if_then_else(s_x[i] == e, 1, 0)
                                s_cnt[e] = s_cnt[e] + acc[0]

                with T.SimtVF(threads=e_pad):
                    for e in T.Parallel(e_pad):
                        pv = T.if_then_else(e < E, s_cnt[e], 0)
                        s_pad[e] = (
                            (((pv + (B - 1)) >> p_shift) << p_shift)
                            if p_shift is not None
                            else T.ceildiv(pv, B) * B
                        )

                with T.SimtVF(threads=e_pad):
                    for e in T.Parallel(e_pad):
                        s_scan[e_pad + e] = s_pad[e]
                    for e in T.Parallel(e_pad):
                        s_scan[e] = 0

                for d in range(n_steps):  # python-level unroll, never `while`
                    k = 1 << d
                    with T.SimtVF(threads=e_pad):
                        for e in T.Parallel(e_pad):
                            s_cs[e] = s_scan[e_pad + e] + s_scan[e_pad + e - k]
                    T.copy(s_cs[0:e_pad], s_scan[e_pad : 2 * e_pad])

                # left shift over [0, E_pad]: padding lanes become `total`
                with T.SimtVF(threads=e_pad + 64):
                    for e in T.Parallel(e_pad + 64):
                        if e <= e_pad:
                            s_cs[e] = T.if_then_else(e == 0, 0, s_scan[e_pad + e - 1])

                total = s_cs[E]
                T.copy(s_cs[0 : E + 1], cumsum[0 : E + 1])
                npost[0] = total

                # rank-count bounded by the build-time constant E (not E_pad)
                with T.SimtVF(threads=blk):
                    for b in T.Parallel(blk):
                        pos = b * B
                        cnt = T.alloc_local([1], "int32")
                        cnt[0] = 0
                        for e in T.serial(E):
                            cnt[0] = cnt[0] + T.if_then_else(s_cs[e] <= pos, 1, 0)
                        s_ids[b] = T.if_then_else(pos < total, cnt[0] - 1, E)
                T.copy(s_ids[0:nb], expert_ids[0:nb])

    return main


def build_align_partial(numel: int, E: int, B: int, g_h: int, g_a: int):
    """Path L launch-2 (mirrors DESIGN.md §3.3 align_kernel)."""
    e_pad = _align_up(E, 64)
    p_total = numel + (E + 1) * (B - 1)
    nb = math.ceil(p_total / B)
    blkpc = math.ceil(nb / g_a)
    blk = _align_up(blkpc, 64)
    n_steps = _steps(e_pad)
    p_shift = B.bit_length() - 1 if (B & (B - 1)) == 0 else None
    share = (p_total + g_a - 1) // g_a
    n_fill = (share + FILL_TILE - 1) // FILL_TILE

    @T.prim_func
    def main(
        partial: T.Tensor((g_h, e_pad), "int32"),
        sorted_ids: T.Tensor((p_total,), "int32"),
        expert_ids: T.Tensor((nb,), "int32"),
        npost: T.Tensor((1,), "int32"),
        cumsum: T.Tensor((E + 1,), "int32"),
    ):
        with T.Kernel(g_a) as a:
            s_part = T.alloc_shared((e_pad,), "int32")
            s_cnt = T.alloc_shared((e_pad,), "int32")
            s_pad = T.alloc_shared((e_pad,), "int32")
            s_scan = T.alloc_shared((2 * e_pad,), "int32")
            s_cs = T.alloc_shared((e_pad + 64,), "int32")
            s_ids = T.alloc_shared((blk,), "int32")
            s_fill = T.alloc_shared((FILL_TILE,), "int32")

            with T.SimtVF(threads=e_pad):
                for e in T.Parallel(e_pad):
                    s_cnt[e] = 0
            for g in T.serial(g_h):
                T.copy(partial[g, :], s_part)
                with T.SimtVF(threads=e_pad):
                    for e in T.Parallel(e_pad):
                        s_cnt[e] = s_cnt[e] + s_part[e]

            with T.SimtVF(threads=e_pad):
                for e in T.Parallel(e_pad):
                    pv = T.if_then_else(e < E, s_cnt[e], 0)
                    s_pad[e] = (
                        (((pv + (B - 1)) >> p_shift) << p_shift)
                        if p_shift is not None
                        else T.ceildiv(pv, B) * B
                    )
            with T.SimtVF(threads=e_pad):
                for e in T.Parallel(e_pad):
                    s_scan[e_pad + e] = s_pad[e]
                for e in T.Parallel(e_pad):
                    s_scan[e] = 0
            for d in range(n_steps):
                k = 1 << d
                with T.SimtVF(threads=e_pad):
                    for e in T.Parallel(e_pad):
                        s_cs[e] = s_scan[e_pad + e] + s_scan[e_pad + e - k]
                T.copy(s_cs[0:e_pad], s_scan[e_pad : 2 * e_pad])
            with T.SimtVF(threads=e_pad + 64):
                for e in T.Parallel(e_pad + 64):
                    if e <= e_pad:
                        s_cs[e] = T.if_then_else(e == 0, 0, s_scan[e_pad + e - 1])

            total = s_cs[E]
            if a == 0:
                T.copy(s_cs[0 : E + 1], cumsum[0 : E + 1])
                npost[0] = total

            blk0 = a * blkpc
            with T.SimtVF(threads=blk):
                for j in T.Parallel(blk):
                    b = blk0 + j
                    cnt = T.alloc_local([1], "int32")
                    cnt[0] = 0
                    for e in T.serial(E):
                        cnt[0] = cnt[0] + T.if_then_else(s_cs[e] <= b * B, 1, 0)
                    s_ids[j] = T.if_then_else(b * B < total, cnt[0] - 1, E)
            valid = T.max(0, T.min(blkpc, nb - blk0))
            if valid > 0:
                T.copy(s_ids[0:valid], expert_ids[blk0 : blk0 + valid])

            T.fill(s_fill, numel)
            for t in T.serial(n_fill):
                off = a * share + t * FILL_TILE
                n = T.min(FILL_TILE, T.min((a + 1) * share, p_total) - off)
                if n > 0:
                    T.copy(s_fill[0:n], sorted_ids[off : off + n])

    return main


CASES = {
    "S33": ("path S, E=33 -> E_pad=64 (L0-6 geometry)", lambda: build_single(600, 33, 64)),
    "S0": ("path S, numel=0 (L0-7 geometry)", lambda: build_single(0, 8, 16)),
    "L384": (
        "path L launch-2, E=384 (max workload)",
        lambda: build_align_partial(32768, 384, 64, 56, 56),
    ),
}


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    ok = True
    for key, (desc, fn) in CASES.items():
        if which not in ("all", key):
            continue
        try:
            tilelang.compile(fn(), target="ascend", out_idx=-1)
            print(f"[{key}] COMPILE_OK   {desc}")
        except Exception as exc:  # noqa: BLE001 - probe: report and continue
            ok = False
            print(f"[{key}] COMPILE_FAIL {desc}: {type(exc).__name__}: {exc}")
    print("PROBE RESULT:", "ALL COMPILE_OK" if ok else "COMPILE_FAIL PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
