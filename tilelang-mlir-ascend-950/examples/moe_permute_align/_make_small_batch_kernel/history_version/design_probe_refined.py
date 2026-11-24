"""Design probe 2 (Stage 1, `_make_small_batch_kernel`): refined candidate forms.

Compile only: `ASCEND_RT_VISIBLE_DEVICES=""`, nothing is launched.  Probe 1
(`design_probe_forms.py`) established: the per-lane UB-indexed staging form
inserts `asc_syncthreads()` **inside** the per-token loop (aliasing guard), and
`T.cumsum` has no ascend implementation.  This probe compiles the refined
primary (V1) and the atomic-free alternative (V3) skeletons plus the fill forms.

Checks
  M1  V1 primary, `(numel=600, E=8, B=64)`, pass 2 reads the route tile from
      UB (`s_x`) + per-expert UB `atomic_add` cursor bump.
  M2  V1 primary, `(numel=1023, E=32, B=64)`.
  M3  V1 primary, `numel=0` boundary.
  M4  V3 alternative, `(numel=600, E=8, B=64)`: both passes per-lane
      (lane<->expert), no atomics (index-ascending expert order).
  M5  V1 with pass 2 reading GM `flat` directly (no UB read-write aliasing in
      the round loop) -- the form chosen for §3.3.
  M6  M5 at the max shape `(numel=1023, E=32, B=64)`.
  M7  sentinel fill form B: `T.Parallel(FILL_TILE)` constant write + tiled
      `T.copy` (code size vs `T.fill`'s unrolled element stores).
  M8  region fill of the route-tile tail: `T.fill(s_x[numel:chunk], -1)`.
  M9-M11  FINAL skeleton `grid = 1` (sentinel fill as a same-core pre-pass) at
      `(600,32,64)`, `(1023,8,16)` and the `numel=0` boundary.  The fill writes
      slots the scatter also writes and there is no cross-core barrier, so the
      fill cannot live on another core (that is why the source's two-CTA split
      becomes a same-core pre-pass).

Run (compile only, no card):
    cd examples/moe_permute_align/_make_small_batch_kernel && bash -c \
      'source /workspace/tilelang-mlir-ascend-950/.agents/platforms/toolchains/ascend/env-950pr.sh; \
       ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python \
       history_version/design_probe_refined.py'
"""

import math
import os
import traceback

os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "")

import tilelang  # noqa: E402
import tilelang.ascend.language as T  # noqa: E402

W = 32          # lane count of the fused path (E <= 32 experts)
UNROLL = 4      # independent histogram accumulators per lane
FILL_TILE = 1024
FILL_LANES = 256


def _geom(numel, num_experts, block):
    padded = numel + (num_experts + 1) * (block - 1)
    nblocks = math.ceil(padded / block)
    chunk = max(64, math.ceil(max(numel, 1) / 64) * 64)
    return padded, nblocks, chunk


def v1(numel, num_experts, block, atomic_pass2=True, pass2_gm=False,
       fill_per_core=8192):
    """The candidate NPU algorithm of §3.3 (V1 when atomic_pass2, else V3)."""
    padded, nblocks, chunk = _geom(numel, num_experts, block)
    blk = math.ceil(max(nblocks, 1) / 64) * 64
    g_fill = max(1, min(55, math.ceil(max(padded, 1) / fill_per_core)))
    grid = 1 + g_fill
    share = math.ceil(max(padded, 1) / g_fill)
    shift = int(math.log2(block)) if block & (block - 1) == 0 else None
    rounds = math.ceil(max(numel, 1) / W)

    @T.prim_func
    def main(
        flat: T.Tensor((max(numel, 1),), "int32"),
        sorted_token_ids: T.Tensor((max(padded, 1),), "int32"),
        expert_ids: T.Tensor((max(nblocks, 1),), "int32"),
        num_tokens_post_pad: T.Tensor((1,), "int32"),
    ) -> None:
        with T.Kernel(grid) as bx:
            s_x = T.alloc_shared((chunk,), "int32")
            s_cnt = T.alloc_shared((W,), "int32")
            s_scan = T.alloc_shared((2 * W,), "int32")
            s_cs = T.alloc_shared((W + 1,), "int32")
            s_ids = T.alloc_shared((blk,), "int32")
            s_fill = T.alloc_shared((FILL_TILE,), "int32")

            # ---- sentinel fill: cores 1..g_fill own disjoint slices ----
            if bx >= 1:
                with T.SimtVF(threads=FILL_LANES):
                    for i in T.Parallel(FILL_TILE):
                        s_fill[i] = numel
                for t in T.serial(math.ceil(share / FILL_TILE)):
                    off = (bx - 1) * share + t * FILL_TILE
                    n = T.min(FILL_TILE, (bx - 1) * share + share - off)
                    n = T.min(n, padded - off)
                    if n > 0:
                        T.copy(s_fill[0:n], sorted_token_ids[off:off + n])

            if bx == 0:
                # ---- load the routes once; tail padded with a non-matching id ----
                T.fill(s_x[numel:chunk], T.int32(-1))
                if numel > 0:
                    T.copy(flat[0:numel], s_x[0:numel])

                # ---- pass 1: histogram, lane <-> expert, UNROLL accumulators ----
                with T.SimtVF(threads=W):
                    for e in T.Parallel(W):
                        acc = T.alloc_local([UNROLL], "int32")
                        for j in range(UNROLL):
                            acc[j] = 0
                        for i in T.serial(chunk // UNROLL):
                            for j in range(UNROLL):
                                acc[j] = acc[j] + T.if_then_else(
                                    s_x[i * UNROLL + j] == e, 1, 0
                                )
                        s_cnt[e] = acc[0] + acc[1] + acc[2] + acc[3]

                # ---- padded count (B a power of two: shift) ----
                with T.SimtVF(threads=W):
                    for e in T.Parallel(W):
                        if shift is not None:
                            s_cnt[e] = ((s_cnt[e] + (block - 1)) >> shift) << shift
                        else:
                            s_cnt[e] = T.ceildiv(s_cnt[e], block) * block

                # ---- HS inclusive scan over W lanes, left halo = 0 ----
                with T.SimtVF(threads=W):
                    for e in T.Parallel(W):
                        s_scan[W + e] = s_cnt[e]
                    for e in T.Parallel(W):
                        s_scan[e] = 0
                for d in range(int(math.log2(W))):
                    k = 1 << d
                    with T.SimtVF(threads=W):
                        for e in T.Parallel(W):
                            s_cs[e] = s_scan[W + e] + s_scan[W + e - k]
                    T.copy(s_cs[0:W], s_scan[W:2 * W])
                # exclusive prefix, written over the full [0, W]
                with T.SimtVF(threads=W + 32):
                    for e in T.Parallel(W + 32):
                        if e <= W:
                            s_cs[e] = T.if_then_else(e == 0, 0, s_scan[W + e - 1])
                num_tokens_post_pad[0] = s_cs[W]

                # ---- expert_ids by the rank-count identity ----
                with T.SimtVF(threads=blk):
                    for b in T.Parallel(blk):
                        cnt = T.alloc_local([1], "int32")
                        cnt[0] = 0
                        for e in T.serial(num_experts):
                            cnt[0] = cnt[0] + T.if_then_else(s_cs[e] <= b * block, 1, 0)
                        s_ids[b] = T.if_then_else(b * block < s_cs[W], cnt[0] - 1, num_experts)
                T.copy(s_ids[0:nblocks], expert_ids[0:nblocks])

                # ---- pass 2 ----
                if atomic_pass2 and pass2_gm:
                    # token-parallel lanes, route id from GM, UB cursor bump
                    with T.SimtVF(threads=W):
                        for j in T.Parallel(W):
                            for r in T.serial(rounds):
                                idx = r * W + j
                                if idx < numel:
                                    eid = flat[idx]
                                    slot = T.atomic_add(s_cs[eid], T.int32(1), return_prev=True)
                                    sorted_token_ids[slot] = idx
                elif atomic_pass2:
                    # token-parallel lanes, route id from UB, UB cursor bump
                    with T.SimtVF(threads=W):
                        for j in T.Parallel(W):
                            for r in T.serial(rounds):
                                idx = r * W + j
                                if idx < numel:
                                    eid = s_x[idx]
                                    slot = T.atomic_add(s_cs[eid], T.int32(1), return_prev=True)
                                    sorted_token_ids[slot] = idx
                else:
                    # alternative: per-lane (lane <-> expert) counter, no atomics
                    with T.SimtVF(threads=W):
                        for e in T.Parallel(W):
                            run = T.alloc_local([1], "int32")
                            run[0] = 0
                            for i in T.serial(numel):
                                if s_x[i] == e:
                                    sorted_token_ids[s_cs[e] + run[0]] = i
                                    run[0] = run[0] + 1
                            s_cnt[e] = run[0]

    return main



def final_single_core(numel, num_experts, block):
    """The shipped skeleton: grid = 1, the sentinel fill is a same-core pre-pass.

    The fill must be ordered before the scatter, and it writes slots the scatter
    also writes: there is no cross-core barrier on this toolchain, so the fill
    and the algorithm have to live on one core (DESIGN.md §0.6 R1 / §7).
    """
    padded, nblocks, chunk = _geom(numel, num_experts, block)
    blk = math.ceil(max(nblocks, 1) / 64) * 64
    shift = int(math.log2(block)) if block & (block - 1) == 0 else None
    rounds = math.ceil(max(numel, 1) / W)

    @T.prim_func
    def main(
        flat: T.Tensor((max(numel, 1),), "int32"),
        sorted_token_ids: T.Tensor((max(padded, 1),), "int32"),
        expert_ids: T.Tensor((max(nblocks, 1),), "int32"),
        num_tokens_post_pad: T.Tensor((1,), "int32"),
    ) -> None:
        with T.Kernel(1) as bx:
            s_x = T.alloc_shared((chunk,), "int32")
            s_cnt = T.alloc_shared((W,), "int32")
            s_scan = T.alloc_shared((2 * W,), "int32")
            s_cs = T.alloc_shared((W + 1,), "int32")
            s_ids = T.alloc_shared((blk,), "int32")
            s_fill = T.alloc_shared((FILL_TILE,), "int32")

            # ---- sentinel fill (same core, bulk MTE3) ----
            with T.SimtVF(threads=FILL_LANES):
                for i in T.Parallel(FILL_TILE):
                    s_fill[i] = numel
            for t in T.serial(math.ceil(max(padded, 1) / FILL_TILE)):
                off = t * FILL_TILE
                n = T.min(FILL_TILE, padded - off)
                if n > 0:
                    T.copy(s_fill[0:n], sorted_token_ids[off:off + n])

            # ---- route tile ----
            T.fill(s_x[numel:chunk], T.int32(-1))
            if numel > 0:
                T.copy(flat[0:numel], s_x[0:numel])

            # ---- pass 1 ----
            with T.SimtVF(threads=W):
                for e in T.Parallel(W):
                    acc = T.alloc_local([UNROLL], "int32")
                    for j in range(UNROLL):
                        acc[j] = 0
                    for i in T.serial(chunk // UNROLL):
                        for j in range(UNROLL):
                            acc[j] = acc[j] + T.if_then_else(s_x[i * UNROLL + j] == e, 1, 0)
                    s_cnt[e] = acc[0] + acc[1] + acc[2] + acc[3]

            with T.SimtVF(threads=W):
                for e in T.Parallel(W):
                    if shift is not None:
                        s_cnt[e] = ((s_cnt[e] + (block - 1)) >> shift) << shift
                    else:
                        s_cnt[e] = T.ceildiv(s_cnt[e], block) * block

            with T.SimtVF(threads=W):
                for e in T.Parallel(W):
                    s_scan[W + e] = s_cnt[e]
                for e in T.Parallel(W):
                    s_scan[e] = 0
            for d in range(int(math.log2(W))):
                k = 1 << d
                with T.SimtVF(threads=W):
                    for e in T.Parallel(W):
                        s_cs[e] = s_scan[W + e] + s_scan[W + e - k]
                T.copy(s_cs[0:W], s_scan[W:2 * W])
            with T.SimtVF(threads=W + 32):
                for e in T.Parallel(W + 32):
                    if e <= W:
                        s_cs[e] = T.if_then_else(e == 0, 0, s_scan[W + e - 1])
            num_tokens_post_pad[0] = s_cs[W]

            with T.SimtVF(threads=blk):
                for b in T.Parallel(blk):
                    cnt = T.alloc_local([1], "int32")
                    cnt[0] = 0
                    for e in T.serial(num_experts):
                        cnt[0] = cnt[0] + T.if_then_else(s_cs[e] <= b * block, 1, 0)
                    s_ids[b] = T.if_then_else(b * block < s_cs[W], cnt[0] - 1, num_experts)
            T.copy(s_ids[0:nblocks], expert_ids[0:nblocks])

            # ---- pass 2: token-parallel, route id from GM, UB cursor bump ----
            with T.SimtVF(threads=W):
                for j in T.Parallel(W):
                    for r in T.serial(rounds):
                        idx = r * W + j
                        if idx < numel:
                            eid = flat[idx]
                            slot = T.atomic_add(s_cs[eid], T.int32(1), return_prev=True)
                            sorted_token_ids[slot] = idx

    return main


def fill_parallel():
    """M7: constant fill via T.Parallel + tiled copies (code size check)."""
    padded, n = 3102, FILL_TILE

    @T.prim_func
    def main(out: T.Tensor((padded,), "int32")) -> None:
        with T.Kernel(1) as bx:
            s_fill = T.alloc_shared((n,), "int32")
            with T.SimtVF(threads=FILL_LANES):
                for i in T.Parallel(n):
                    s_fill[i] = T.int32(1023)
            for t in T.serial(math.ceil(padded / n)):
                off = t * n
                m = T.min(n, padded - off)
                if m > 0:
                    T.copy(s_fill[0:m], out[off:off + m])

    return main


def fill_region():
    """M8: region fill of the route-tile tail."""
    numel, chunk = 600, 640

    @T.prim_func
    def main(flat: T.Tensor((numel,), "int32"), out: T.Tensor((chunk,), "int32")) -> None:
        with T.Kernel(1) as bx:
            s_x = T.alloc_shared((chunk,), "int32")
            T.fill(s_x[numel:chunk], T.int32(-1))
            T.copy(flat[0:numel], s_x[0:numel])
            T.copy(s_x, out)

    return main


CHECKS = [
    ("M9_final_single_core_600_32_64", lambda: final_single_core(600, 32, 64)),
    ("M10_final_single_core_1023_8_16", lambda: final_single_core(1023, 8, 16)),
    ("M11_final_single_core_numel0_8_16", lambda: final_single_core(0, 8, 16)),
    ("M1_v1_ub_pass2_600_8_64", lambda: v1(600, 8, 64)),
    ("M2_v1_ub_pass2_1023_32_64", lambda: v1(1023, 32, 64)),
    ("M3_v1_ub_pass2_numel0_8_64", lambda: v1(0, 8, 64)),
    ("M4_v3_alternative_600_8_64", lambda: v1(600, 8, 64, atomic_pass2=False)),
    ("M5_v1_gm_pass2_600_8_64", lambda: v1(600, 8, 64, pass2_gm=True)),
    ("M6_v1_gm_pass2_1023_32_64", lambda: v1(1023, 32, 64, pass2_gm=True)),
    ("M7_fill_parallel_1024", fill_parallel),
    ("M8_fill_region_tail", fill_region),
]


def dump(name, build):
    print(f"\n===== {name} =====")
    try:
        kernel = tilelang.compile(build(), target="ascend", out_idx=[])
        print("COMPILE: OK")
        src = ""
        for attr in ("get_kernel_source", "get_source"):
            if hasattr(kernel, attr):
                try:
                    src = str(getattr(kernel, attr)())
                    break
                except Exception:  # noqa: BLE001 - diagnostic only
                    pass
        print(src[:7000] if src else "(no source accessor)")
        return True
    except Exception:  # noqa: BLE001 - probe reports, never raises
        print("COMPILE: FAIL")
        traceback.print_exc(limit=5)
        return False


def main() -> int:
    print(f"tilelang: {tilelang.__file__}")
    print(f"ASCEND_RT_VISIBLE_DEVICES={os.environ.get('ASCEND_RT_VISIBLE_DEVICES')!r}")
    rc = 0
    for name, build in CHECKS:
        if not dump(name, build):
            rc += 1
    print(f"\nRESULT: {len(CHECKS) - rc}/{len(CHECKS)} COMPILE_OK")
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
