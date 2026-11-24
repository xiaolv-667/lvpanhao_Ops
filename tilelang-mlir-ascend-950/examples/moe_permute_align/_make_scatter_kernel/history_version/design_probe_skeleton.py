"""Design probe 2 (Stage 1, _make_scatter_kernel): skeleton + lowered-source evidence.

Compile-only ( `ASCEND_RT_VISIBLE_DEVICES=""`, nothing is launched ).

  K1b  lowered source of a per-lane data-dependent GM store (from probe 1 K1)
  K2b  lowered source of a per-element GM atomic add with return_prev (probe 1 K2)
  K6   PRIMARY skeleton: route-slice blocks + per-lane `atomic_add(cumsum[e])`
       bump allocator + per-lane `sorted[slot] = i` scatter store
  K7   FALLBACK skeleton: expert-group blocks + hand-rolled Hillis-Steele
       predicate prefix in UB (T.cumsum is NOT available for ascend) +
       per-lane scatter store -- determinism path
  K8   negative control: `T.cumsum` (already known to fail; re-confirmed here so
       the design can cite the exact error text)
"""

import math
import os
import sys
import traceback

os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "")

import tilelang  # noqa: E402
import tilelang.ascend.language as T  # noqa: E402

NUMEL = 4096
E = 384
BLOCK = 64
P = NUMEL + (E + 1) * (BLOCK - 1)


def dump(name, fn):
    print(f"\n===== {name} =====")
    try:
        kernel = tilelang.compile(fn, target="ascend", out_idx=[])
        print("COMPILE: OK")
        src = ""
        for attr in ("get_kernel_source", "get_source"):
            if hasattr(kernel, attr):
                try:
                    src = str(getattr(kernel, attr)())
                    break
                except Exception:  # noqa: BLE001 - diagnostic only
                    pass
        print(src[:6000] if src else "(no source accessor)")
        return True
    except Exception:  # noqa: BLE001 - probe reports, never raises
        print("COMPILE: FAIL")
        traceback.print_exc(limit=4)
        return False


def k1b():
    n = 256

    @T.prim_func
    def main(
        src: T.Tensor((n,), "int32"),
        pos: T.Tensor((n,), "int32"),
        out: T.Tensor((n,), "int32"),
    ) -> None:
        with T.Kernel(1) as bx, T.SimtVF(threads=64):
            for i in T.Parallel(n):
                out[pos[i]] = src[i]

    return main


def k2b():
    n, e = 256, 8

    @T.prim_func
    def main(
        key: T.Tensor((n,), "int32"),
        cursor: T.Tensor((e,), "int32"),
        prev: T.Tensor((n,), "int32"),
    ) -> None:
        with T.Kernel(1) as bx, T.SimtVF(threads=64):
            for i in T.Parallel(n):
                prev[i] = T.atomic_add(cursor[key[i]], T.int32(1), return_prev=True)

    return main


def k6(numel, num_experts, tile, threads):
    """PRIMARY: route-slice blocks, per-route GM atomic bump + scatter store."""
    nblocks = (numel + tile - 1) // tile
    padded = numel + (num_experts + 1) * (BLOCK - 1)

    @T.prim_func
    def main(
        flat: T.Tensor((numel,), "int32"),
        sorted_token_ids: T.Tensor((padded,), "int32"),
        cumsum: T.Tensor((num_experts + 1,), "int32"),
    ) -> None:
        with T.Kernel(nblocks) as bid, T.SimtVF(threads=threads):
            for j in T.Parallel(tile):
                idx = bid * tile + j
                if idx < numel:
                    eid = flat[idx]
                    slot = T.atomic_add(cumsum[eid], T.int32(1), return_prev=True)
                    sorted_token_ids[slot] = idx

    return main


def k7(numel, num_experts, group, tile, threads):
    """FALLBACK: expert-group blocks, hand-rolled predicate prefix, no atomics."""
    nblocks = (num_experts + group - 1) // group
    ntiles = (numel + tile - 1) // tile
    steps = int(math.log2(tile))
    padded = numel + (num_experts + 1) * (BLOCK - 1)

    @T.prim_func
    def main(
        flat: T.Tensor((numel,), "int32"),
        sorted_token_ids: T.Tensor((padded,), "int32"),
        cumsum: T.Tensor((num_experts + 1,), "int32"),
    ) -> None:
        with T.Kernel(nblocks) as bx, T.SimtVF(threads=threads):
            m = T.alloc_shared((tile,), "int32")
            p = T.alloc_shared((tile,), "int32")
            carry = T.alloc_var("int32")
            for g in T.serial(group):
                expert = bx * group + g
                if expert < num_experts:
                    if cumsum[expert + 1] > cumsum[expert]:
                        carry = T.int32(0)
                        for t in T.serial(ntiles):
                            for j in T.Parallel(tile):
                                pos = t * tile + j
                                m[j] = T.if_then_else(
                                    pos < numel,
                                    T.if_then_else(flat[T.min(pos, numel - 1)] == expert, T.int32(1), T.int32(0)),
                                    T.int32(0),
                                )
                            # Hillis-Steele inclusive prefix into p
                            for s in T.serial(steps):
                                stride = 1 << s
                                for j in T.Parallel(tile):
                                    p[j] = m[j] + T.if_then_else(j >= stride, m[j - stride], T.int32(0))
                                T.sync_threads()
                                for j in T.Parallel(tile):
                                    m[j] = p[j]
                                T.sync_threads()
                            for j in T.Parallel(tile):
                                pos = t * tile + j
                                if pos < numel:
                                    if flat[pos] == expert:
                                        sorted_token_ids[cumsum[expert] + carry + m[j] - 1] = pos
                            carry = carry + m[tile - 1]
                            T.sync_threads()

    return main


def k8():
    n = 64

    @T.prim_func
    def main(src: T.Tensor((n,), "int32"), out: T.Tensor((n,), "int32")) -> None:
        with T.Kernel(1) as bx, T.SimtVF(threads=64):
            s = T.alloc_shared((n,), "int32")
            d = T.alloc_shared((n,), "int32")
            for i in T.Parallel(n):
                s[i] = src[i]
            T.cumsum(s, d, dim=0)
            for i in T.Parallel(n):
                out[i] = d[i]

    return main


if __name__ == "__main__":
    print(f"tilelang: {tilelang.__file__}")
    dump("K1b_lowered_per_lane_scatter_store", k1b())
    dump("K2b_lowered_gm_atomic_retprev", k2b())
    ok6 = dump("K6_PRIMARY_skeleton", k6(NUMEL, E, tile=1024, threads=128))
    ok7 = dump("K7_FALLBACK_skeleton", k7(NUMEL, E, group=7, tile=512, threads=128))
    ok8 = dump("K8_cumsum_negative_control", k8())
    print(f"\nSUMMARY primary={ok6} fallback={ok7} cumsum={ok8}")
    sys.exit(0 if (ok6 and ok7 and not ok8) else 1)
