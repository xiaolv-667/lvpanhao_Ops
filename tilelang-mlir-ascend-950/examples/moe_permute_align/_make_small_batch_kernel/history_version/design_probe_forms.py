"""Design probe (Stage 1, `_make_small_batch_kernel`): compile-only form evidence.

Compile only: `ASCEND_RT_VISIBLE_DEVICES=""`, nothing is launched.  The script
lowers + compiles candidate prim_func forms for `target="ascend"` so DESIGN.md
§0.5/§3.3/§3.4 can cite real evidence (which form builds, and what it lowers
to) instead of guesses.

Checks
  K1  PRIMARY skeleton, workload A `(numel=600, E=8, B=64)`:
      grid = 1 + G_fill (core 0 = algorithm, cores 1.. = sentinel fill),
      lane<->expert pass 1 (histogram) + HS exclusive prefix + rank-count
      `expert_ids` + lane<->expert pass 2 (per-lane counter + predicated
      GM scatter store).
  K2  PRIMARY skeleton, workload B `(numel=1023, E=32, B=64)` (W=32 lanes).
  K3  PRIMARY skeleton, `numel=0` boundary (empty `flat`, algorithm still
      writes all three outputs).
  K4  ALTERNATIVE skeleton: single pass staging per-expert UB lists
      (`s_list[e, r] = i`) + one runtime-length `T.copy` per expert.
  K5  per-lane data-dependent UB (shared) indexed read/write:
      `s_cnts[p * W + eid] += 1` inside `T.Parallel`.
  K6  per-element UB (shared) atomic add with return_prev (own evidence; the
      sibling `_make_scatter_kernel` probe also reported it compiles).
  K7  negative control: `T.cumsum` on a UB int32 buffer (expects FAIL; the
      error text is cited in DESIGN.md §3.4).

Run (compile only, no card):
    cd examples/moe_permute_align/_make_small_batch_kernel && bash -c \
      'source /workspace/tilelang-mlir-ascend-950/.agents/platforms/toolchains/ascend/env-950pr.sh; \
       ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python \
       history_version/design_probe_forms.py'
"""

import math
import os
import traceback

os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "")

import tilelang  # noqa: E402
import tilelang.ascend.language as T  # noqa: E402

W = 32          # lane count = max experts of the fused path (E <= W)
FILL_TILE = 2048


def _geom(numel, num_experts, block):
    padded = numel + (num_experts + 1) * (block - 1)
    nblocks = math.ceil(padded / block)
    return padded, nblocks


def primary(numel, num_experts, block, fill_per_core=8192, dump_full=False):
    """The candidate NPU algorithm of §3.3 (primary)."""
    padded, nblocks = _geom(numel, num_experts, block)
    blk = math.ceil(nblocks / 64) * 64 if nblocks else 64
    g_fill = max(1, min(55, math.ceil(padded / fill_per_core))) if padded else 0
    grid = 1 + g_fill
    shift = int(math.log2(block)) if block & (block - 1) == 0 else None

    @T.prim_func
    def main(
        flat: T.Tensor((max(numel, 1),), "int32"),
        sorted_token_ids: T.Tensor((max(padded, 1),), "int32"),
        expert_ids: T.Tensor((max(nblocks, 1),), "int32"),
        num_tokens_post_pad: T.Tensor((1,), "int32"),
    ) -> None:
        with T.Kernel(grid) as bx:
            s_fill = T.alloc_shared((FILL_TILE,), "int32")
            s_x = T.alloc_shared((max(numel, 64),), "int32")
            s_cnt = T.alloc_shared((W,), "int32")
            s_scan = T.alloc_shared((2 * W,), "int32")
            s_cs = T.alloc_shared((W + 1,), "int32")
            s_run = T.alloc_shared((W,), "int32")
            s_ids = T.alloc_shared((blk,), "int32")

            # ---- sentinel fill: cores 1..g_fill, disjoint slices ----
            if bx >= 1:
                share = T.ceildiv(padded, g_fill) if g_fill else padded
                T.fill(s_fill, numel)
                if share > 0:
                    for t in T.serial(math.ceil(padded / FILL_TILE) if padded else 0):
                        off = (bx - 1) * share + t * FILL_TILE
                        n = T.min(FILL_TILE, T.min(share * (bx - 1) + share, padded) - off)
                        if n > 0:
                            T.copy(s_fill[0:n], sorted_token_ids[off:off + n])

            if bx == 0:
                # ---- load the routes once (UB resident) ----
                if numel > 0:
                    T.copy(flat[0:numel], s_x[0:numel])

                # ---- pass 1: histogram, lane <-> expert (private counter) ----
                with T.SimtVF(threads=W):
                    for e in T.Parallel(W):
                        acc = T.alloc_local([1], "int32")
                        acc[0] = 0
                        for i in T.serial(numel):
                            acc[0] = acc[0] + T.if_then_else(s_x[i] == e, 1, 0)
                        s_cnt[e] = acc[0]

                # ---- padded count (B a power of two: shift; else div/mul) ----
                with T.SimtVF(threads=W):
                    for e in T.Parallel(W):
                        if shift is not None:
                            s_cnt[e] = ((s_cnt[e] + (block - 1)) >> shift) << shift
                        else:
                            s_cnt[e] = T.ceildiv(s_cnt[e], block) * block

                # ---- HS inclusive scan over W lanes (left halo = 0) ----
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

                # ---- pass 2: per-lane running counter + predicated scatter ----
                with T.SimtVF(threads=W):
                    for e in T.Parallel(W):
                        run = T.alloc_local([1], "int32")
                        run[0] = 0
                        for i in T.serial(numel):
                            if s_x[i] == e:
                                sorted_token_ids[s_cs[e] + run[0]] = i
                                run[0] = run[0] + 1
                        s_run[e] = run[0]

    return main


def alternative_staged(numel, num_experts, block):
    """Single pass: stage each expert's routes in UB, then emit per expert."""
    padded, nblocks = _geom(numel, num_experts, block)
    cap = max(numel, 1)
    shift = int(math.log2(block)) if block & (block - 1) == 0 else None

    @T.prim_func
    def main(
        flat: T.Tensor((max(numel, 1),), "int32"),
        sorted_token_ids: T.Tensor((max(padded, 1),), "int32"),
    ) -> None:
        with T.Kernel(1) as bx:
            s_x = T.alloc_shared((max(numel, 64),), "int32")
            s_list = T.alloc_shared((W * cap,), "int32")
            s_cnt = T.alloc_shared((W,), "int32")
            s_scan = T.alloc_shared((2 * W,), "int32")
            s_cs = T.alloc_shared((W + 1,), "int32")

            if numel > 0:
                T.copy(flat[0:numel], s_x[0:numel])
            # one pass: count and stage (per-lane private region)
            with T.SimtVF(threads=W):
                for e in T.Parallel(W):
                    run = T.alloc_local([1], "int32")
                    run[0] = 0
                    for i in T.serial(numel):
                        if s_x[i] == e:
                            s_list[e * cap + run[0]] = i
                            run[0] = run[0] + 1
                    s_cnt[e] = run[0]
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

            # emit: one contiguous runtime-length copy per expert
            for e in T.serial(num_experts):
                n = s_cs[e + 1] - s_cs[e]
                if n > 0:
                    T.copy(s_list[e * cap:e * cap + n], sorted_token_ids[s_cs[e]:s_cs[e] + n])

    return main


def per_lane_ub_indexed():
    """K5: per-lane data-dependent UB read/write inside T.Parallel."""
    n, lanes, experts = 64, 32, 8

    @T.prim_func
    def main(
        key: T.Tensor((n,), "int32"),
        out: T.Tensor((n,), "int32"),
    ) -> None:
        with T.Kernel(1) as bx:
            s_cnts = T.alloc_shared((lanes * experts,), "int32")
            with T.SimtVF(threads=lanes):
                for p in T.Parallel(lanes):
                    s_cnts[p * experts + (p % experts)] = 0
                for p in T.Parallel(lanes):
                    for i in T.serial(n // lanes):
                        idx = p + i * lanes
                        eid = key[idx]
                        prev = s_cnts[p * experts + eid]
                        s_cnts[p * experts + eid] = prev + 1
                        out[idx] = prev

    return main


def ub_atomic():
    """K6: per-element UB atomic add returning the previous value."""
    n, experts = 64, 8

    @T.prim_func
    def main(
        key: T.Tensor((n,), "int32"),
        out: T.Tensor((n,), "int32"),
    ) -> None:
        with T.Kernel(1) as bx:
            s_count = T.alloc_shared((experts,), "int32")
            with T.SimtVF(threads=32):
                for i in T.Parallel(n):
                    out[i] = T.atomic_add(s_count[key[i]], T.int32(1), return_prev=True)

    return main


def ub_cumsum():
    """K7: negative control -- T.cumsum on a UB int32 buffer."""
    n = 64

    @T.prim_func
    def main(
        src: T.Tensor((n,), "int32"),
        out: T.Tensor((n,), "int32"),
    ) -> None:
        with T.Kernel(1) as bx:
            s_mask = T.alloc_shared((n,), "int32")
            s_rank = T.alloc_shared((n,), "int32")
            with T.SimtVF(threads=64):
                for i in T.Parallel(n):
                    s_mask[i] = src[i]
            T.cumsum(s_mask, s_rank, dim=0)
            with T.SimtVF(threads=64):
                for i in T.Parallel(n):
                    out[i] = s_rank[i]

    return main


CHECKS = [
    ("K1_primary_600_8_64", lambda: primary(600, 8, 64)),
    ("K2_primary_1023_32_64", lambda: primary(1023, 32, 64)),
    ("K3_primary_numel0_8_64", lambda: primary(0, 8, 64)),
    ("K4_alternative_staged_600_8_64", lambda: alternative_staged(600, 8, 64)),
    ("K5_per_lane_ub_indexed", per_lane_ub_indexed),
    ("K6_per_element_ub_atomic", ub_atomic),
    ("K7_ub_cumsum_negative_control", ub_cumsum),
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
