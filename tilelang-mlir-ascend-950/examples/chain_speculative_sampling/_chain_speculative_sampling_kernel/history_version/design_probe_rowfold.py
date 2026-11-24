"""Design probe #2 (Stage 1, _chain_speculative_sampling_kernel) -- the row fold and locate.

The NPU row pipeline the design proposes has these structural pieces, which no existing
source of this platform demonstrates together:

  p2a  one core per row, the row scanned in ``ceil(V / tile)`` tiles: a VF region per row
       holding the chain test, the tile fold (residual ``max(0, target - draft)`` in place,
       one reducer sum a tile into a UB array), the kernel-side coarse prefix, and the
       int32 emit -- the per-row loop is a kernel-scope ``T.serial``
  p2b  the same at a full workload size (V = 151936, tile = 4096, B = 32, N = 5) with the
       ``grid = min(B, vector cores)`` fold, to price the UB budget at the real tile width
  p2c  the tile load at a dynamic row base: ``T.copy(region, ub)`` vs the element-wise
       ``T.Parallel`` read, and a V odd row (V = 4099) whose tiles are not whole vectors

Run (no NPU, no launch):
    cd examples/chain_speculative_sampling/_chain_speculative_sampling_kernel
    source ../../../.agents/platforms/toolchains/ascend/env-950pr.sh
    ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python \
        history_version/design_probe_rowfold.py [p2a|p2b|p2c|all]
"""

import os
import sys
import traceback

os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "")

import tilelang  # noqa: E402
import tilelang.ascend.language as T  # noqa: E402

_MARKS = ("asc_copy", "DataCopy", "vld", "vsts", "reduce", "vadd", "vsub", "vmax", "atomic",
          "tl::", "asc_syncthreads")


def _report(name, fn):
    print(f"\n===== {name} =====", flush=True)
    try:
        kernel = tilelang.compile(fn, target="ascend", out_idx=[])
        print("COMPILE: OK", flush=True)
    except Exception:  # noqa: BLE001
        print("COMPILE: FAIL", flush=True)
        traceback.print_exc(limit=4)
        return None
    src = kernel.get_kernel_source()
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"probe_{name}_gen.txt")
    with open(path, "w") as fh:
        fh.write(src)
    hits = [ln.strip() for ln in src.splitlines() if any(m in ln for m in _MARKS)]
    print(f"  generated-code lines with copy/vector/reduce ops: {len(hits)}", flush=True)
    for ln in hits[:16]:
        print("   ", ln[:170], flush=True)
    return src


def _build(batch, num_draft, vocab, tile, blocks, by_copy):
    """The row pipeline skeleton: chain test, tile fold, locate, emit."""
    ntile = -(-vocab // tile)
    per_core = -(-batch // blocks)
    threads = 128

    rows = batch * (num_draft + 1)

    @T.prim_func
    def main(
        draft_probs: T.Tensor((batch * num_draft * vocab,), "float32"),
        draft_token_ids: T.Tensor((batch * num_draft,), "int32"),
        target_probs: T.Tensor((rows * vocab,), "float32"),
        seed_w: T.Tensor((2,), "int32"),
        offset_w: T.Tensor((2,), "int32"),
        output_token_ids: T.Tensor((batch * (num_draft + 1),), "int32"),
        num_accepted: T.Tensor((batch,), "int32"),
    ):
        with T.Kernel(blocks) as bx:
            t_ub = T.alloc_shared((tile,), "float32")
            r_ub = T.alloc_shared((tile,), "float32")
            tile_sum = T.alloc_shared((ntile,), "float32")
            ids = T.alloc_shared((num_draft + 1,), "int32")
            acc = T.alloc_shared((1,), "int32")
            at = T.alloc_shared((1,), "int32")
            point = T.alloc_shared((1,), "float32")
            token = T.alloc_shared((1,), "int32")
            for rr in T.serial(per_core):
                line = bx + rr * blocks
                if line < batch:
                    with T.SimtVF(threads=threads):
                        can = T.alloc_reducer((1,), "float32", op="sum")
                        s = T.alloc_fragment((1,), "float32")
                        # Chain test (draws stand in for Philox here): scalar, all lanes.
                        for j in T.Parallel(num_draft):
                            ids[j] = draft_token_ids[line * num_draft + j]
                        acc[0] = 0
                        for j in T.serial(num_draft):
                            if (acc[0] == j) & (
                                T.float32(0.25) * draft_probs[(line * num_draft + j) * vocab + ids[j]]
                                >= target_probs[(line * (num_draft + 1) + j) * vocab + ids[j]]
                            ):
                                acc[0] = j + 1
                        at[0] = (line * (num_draft + 1) + acc[0]) * vocab
                        # Tile fold: residual in place, one sum a tile.
                        for t in T.serial(ntile):
                            if by_copy:
                                T.copy(
                                    target_probs[at[0] + t * tile : at[0] + t * tile + tile],
                                    t_ub,
                                )
                            else:
                                for i in T.Parallel(tile):
                                    t_ub[i] = T.if_then_else(
                                        t * tile + i < vocab,
                                        target_probs[at[0] + T.min(t * tile + i, vocab - 1)],
                                        T.float32(0),
                                    )
                            for i in T.Parallel(tile):
                                r_ub[i] = T.if_then_else(
                                    t * tile + i < vocab,
                                    T.max(
                                        t_ub[i]
                                        - T.if_then_else(
                                            acc[0] < num_draft,
                                            draft_probs[
                                                (line * num_draft + T.min(acc[0], num_draft - 1))
                                                * vocab
                                                + T.min(t * tile + i, vocab - 1)
                                            ],
                                            T.float32(0),
                                        ),
                                        T.float32(0),
                                    ),
                                    T.float32(0),
                                )
                            T.reducer_init(can)
                            for i in T.Parallel(tile):
                                T.reducer_update(can[0], r_ub[i])
                            T.finalize_reducer(can, s)
                            if T.get_thread_binding() == 0:
                                tile_sum[t] = s[0]
                        # Coarse prefix: redundant on every lane, same answer.
                        point[0] = T.float32(0.5)
                        token[0] = ntile - 1
                        for t in T.serial(ntile):
                            point[0] = point[0] - tile_sum[t]
                            if point[0] <= T.float32(0):
                                token[0] = t
                        # Emit.
                        if T.get_thread_binding() == 0:
                            for j in T.serial(num_draft + 1):
                                output_token_ids[line * (num_draft + 1) + j] = T.if_then_else(
                                    j < acc[0],
                                    draft_token_ids[line * num_draft + T.min(j, num_draft - 1)],
                                    T.if_then_else(j == acc[0], token[0], -1),
                                )
                            num_accepted[line] = acc[0]

    return main


def make_p2a():
    return _build(batch=2, num_draft=2, vocab=300, tile=128, blocks=2, by_copy=True)


def make_p2b():
    return _build(batch=32, num_draft=5, vocab=151936, tile=4096, blocks=32, by_copy=False)


def make_p2c():
    return _build(batch=8, num_draft=3, vocab=4099, tile=1024, blocks=8, by_copy=False)



def make_p2d():
    """The grid fold: B = 256 rows over 56 cores (5 rows a core, guarded tail)."""
    return _build(batch=256, num_draft=1, vocab=32000, tile=4096, blocks=56, by_copy=False)


def make_p2e():
    """Kernel-scope ``T.copy`` of a tile at a dynamic row base (the DMA spelling)."""
    return _build(batch=8, num_draft=2, vocab=32000, tile=4096, blocks=8, by_copy=True)



def _locate_nested(vocab, tile, threads):
    """The crossing tile walked by a lane-partial level (nested Parallel over serial)."""
    per_lane = -(-tile // threads)
    ntile = -(-vocab // tile)

    @T.prim_func
    def main(
        target_probs: T.Tensor((vocab,), "float32"),
        out: T.Tensor((3,), "int32"),
    ):
        with T.Kernel(1) as bx:
            r_ub = T.alloc_shared((tile,), "float32")
            lane_sum = T.alloc_shared((threads,), "float32")
            point = T.alloc_shared((1,), "float32")
            pick = T.alloc_shared((1,), "int32")
            with T.SimtVF(threads=threads):
                # The crossing tile: residual recomputed into UB.
                for i in T.Parallel(tile):
                    r_ub[i] = T.if_then_else(i < vocab, target_probs[T.min(i, vocab - 1)], T.float32(0))
                # Lane level: one thread sums its own per_lane elements.
                for l in T.Parallel(threads):
                    part = T.alloc_local((1,), "float32")
                    part[0] = T.float32(0)
                    for k in T.serial(per_lane):
                        part[0] = part[0] + r_ub[l * per_lane + k]
                    lane_sum[l] = part[0]
                # The lane prefix and the element walk: serial, redundant on every lane.
                point[0] = T.float32(0.5)
                pick[0] = threads - 1
                for l in T.serial(threads):
                    if pick[0] == threads - 1:
                        point[0] = point[0] - lane_sum[l]
                        if point[0] < T.float32(0):
                            pick[0] = l
                if T.get_thread_binding() == 0:
                    out[0] = pick[0]
                    out[1] = 0
                    out[2] = ntile * per_lane * threads
    return main


def _locate_bisect(vocab, tile, threads):
    """The crossing tile located by masked half-window reductions (no lane partials)."""
    ntile = -(-vocab // tile)

    @T.prim_func
    def main(
        target_probs: T.Tensor((vocab,), "float32"),
        out: T.Tensor((3,), "int32"),
    ):
        with T.Kernel(1) as bx:
            r_ub = T.alloc_shared((tile,), "float32")
            lo = T.alloc_shared((1,), "int32")
            half = T.alloc_shared((1,), "int32")
            point = T.alloc_shared((1,), "float32")
            with T.SimtVF(threads=threads):
                can = T.alloc_reducer((1,), "float32", op="sum")
                s = T.alloc_fragment((1,), "float32")
                for i in T.Parallel(tile):
                    r_ub[i] = T.if_then_else(i < vocab, target_probs[T.min(i, vocab - 1)], T.float32(0))
                lo[0] = 0
                half[0] = tile
                point[0] = T.float32(0.5)
                for step in T.serial(6):
                    half[0] = half[0] // 2
                    T.reducer_init(can)
                    for i in T.Parallel(tile):
                        T.reducer_update(
                            can[0],
                            T.if_then_else(
                                (i >= lo[0]) & (i < lo[0] + half[0]), r_ub[i], T.float32(0)
                            ),
                        )
                    T.finalize_reducer(can, s)
                    if T.get_thread_binding() == 0:
                        if s[0] <= point[0]:
                            point[0] = point[0] - s[0]
                            lo[0] = lo[0] + half[0]
                if T.get_thread_binding() == 0:
                    out[0] = lo[0]
                    out[1] = step
                    out[2] = ntile * tile
    return main


def make_p2f():
    return _locate_nested(4096, 4096, 128)


def make_p2g():
    return _locate_bisect(4096, 4096, 128)


_CASES = {"p2f": make_p2f, "p2g": make_p2g, "p2a": make_p2a, "p2b": make_p2b, "p2c": make_p2c, "p2d": make_p2d, "p2e": make_p2e}


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "all"
    cases = _CASES if name == "all" else {name: _CASES[name]}
    for label, build in cases.items():
        _report(label, build())


if __name__ == "__main__":
    main()
