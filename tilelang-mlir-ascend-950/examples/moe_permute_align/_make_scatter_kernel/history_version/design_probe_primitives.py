"""Design probe 1 (Stage 1, _make_scatter_kernel): compile-only primitive availability.

No device work: this script only *lowers + compiles* candidate primitives for
target="ascend" with `ASCEND_RT_VISIBLE_DEVICES=""` so the design can cite real
evidence instead of guesses.  Nothing is launched.

Checks (one prim_func each, compiled independently):
  K1  per-lane data-dependent GM store inside T.SimtVF + T.Parallel
      (`out[pos[i]] = src[i]`)                       -- the scatter store
  K2  per-element GM atomic add with return_prev
      (`prev[i] = T.atomic_add(cursor[key[i]], 1, return_prev=True)`)
  K3  per-element UB (alloc_shared) atomic add
  K4  T.cumsum on a UB int32 buffer (predicate prefix sum)
  K5  T.set_atomic("add","int32") + T.copy(UB -> GM) store-mode accumulate

Run (compile only, no card):
    cd <task dir> && bash -c 'source .../env-950pr.sh; \
        ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python \
        history_version/design_probe_primitives.py'
"""

import os
import sys
import traceback

os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "")

import tilelang  # noqa: E402
import tilelang.ascend.language as T  # noqa: E402

N = 256
E = 8
THREADS = 64


def k1_scatter_store():
    @T.prim_func
    def main(
        src: T.Tensor((N,), "int32"),
        pos: T.Tensor((N,), "int32"),
        out: T.Tensor((N,), "int32"),
    ) -> None:
        with T.Kernel(1) as bx, T.SimtVF(threads=THREADS):
            for i in T.Parallel(N):
                out[pos[i]] = src[i]

    return main


def k2_gm_atomic():
    @T.prim_func
    def main(
        key: T.Tensor((N,), "int32"),
        cursor: T.Tensor((E,), "int32"),
        prev: T.Tensor((N,), "int32"),
    ) -> None:
        with T.Kernel(1) as bx, T.SimtVF(threads=THREADS):
            for i in T.Parallel(N):
                prev[i] = T.atomic_add(cursor[key[i]], T.int32(1), return_prev=True)

    return main


def k3_ub_atomic():
    @T.prim_func
    def main(
        key: T.Tensor((N,), "int32"),
        out: T.Tensor((N,), "int32"),
    ) -> None:
        with T.Kernel(1) as bx, T.SimtVF(threads=THREADS):
            s_count = T.alloc_shared((E,), "int32")
            for i in T.Parallel(N):
                out[i] = T.atomic_add(s_count[key[i]], T.int32(1), return_prev=True)

    return main


def k4_cumsum():
    @T.prim_func
    def main(
        src: T.Tensor((N,), "int32"),
        key: T.Tensor((N,), "int32"),
        out: T.Tensor((N,), "int32"),
    ) -> None:
        with T.Kernel(1) as bx, T.SimtVF(threads=THREADS):
            s_mask = T.alloc_shared((N,), "int32")
            s_rank = T.alloc_shared((N,), "int32")
            for i in T.Parallel(N):
                s_mask[i] = T.if_then_else(key[i] == T.int32(0), T.int32(1), T.int32(0))
            T.cumsum(s_mask, s_rank, dim=0)
            for i in T.Parallel(N):
                out[i] = s_rank[i]

    return main


def k5_store_mode_atomic():
    @T.prim_func
    def main(
        src: T.Tensor((N,), "int32"),
        out: T.Tensor((N,), "int32"),
    ) -> None:
        with T.Kernel(1) as bx, T.SimtVF(threads=THREADS):
            s = T.alloc_shared((N,), "int32")
            for i in T.Parallel(N):
                s[i] = src[i]
            T.set_atomic("add", "int32")
            T.copy(s, out)
            T.set_atomic_none()

    return main


CHECKS = [
    ("K1_per_lane_gm_scatter_store", k1_scatter_store),
    ("K2_per_element_gm_atomic", k2_gm_atomic),
    ("K3_per_element_ub_atomic", k3_ub_atomic),
    ("K4_ub_cumsum", k4_cumsum),
    ("K5_store_mode_atomic_int32", k5_store_mode_atomic),
]


def main() -> int:
    print(f"tilelang: {tilelang.__file__}")
    print(f"ASCEND_RT_VISIBLE_DEVICES={os.environ.get('ASCEND_RT_VISIBLE_DEVICES')!r}")
    rc = 0
    for name, build in CHECKS:
        print(f"\n===== {name} =====")
        try:
            fn = build()
            kernel = tilelang.compile(fn, target="ascend", out_idx=[])
            print("COMPILE: OK")
            src = None
            for attr in ("get_kernel_source", "get_source"):
                if hasattr(kernel, attr):
                    try:
                        src = getattr(kernel, attr)()
                        break
                    except Exception:  # noqa: BLE001 - diagnostic only
                        pass
            if src:
                interesting = [
                    line.strip()
                    for line in str(src).splitlines()
                    if any(t in line for t in ("Atomic", "atomic", "SetAtomic", "DataCopy", "scatter", "GetValue"))
                ]
                print("SOURCE hints:", interesting[:12] or "(none matched)")
        except Exception:  # noqa: BLE001 - probe reports, never raises
            rc = 1
            print("COMPILE: FAIL")
            traceback.print_exc(limit=6)
    return rc


if __name__ == "__main__":
    sys.exit(main())


# ---------------------------------------------------------------------------
# K9 (added after the first run): the *exact* factory form of DESIGN.md §3.3 --
# `@tilelang.jit(out_idx=[], target="ascend")` in lazy mode (function returns a
# prim_func) + serial(E PT) x parallel(THREADS) nesting + atomic + scatter.
# Tiny shape (numel=64, E=8, bs=4); compile only.
# ---------------------------------------------------------------------------
def k9_factory_form():
    import functools
    import math

    THREADS = 128

    @functools.lru_cache(maxsize=32)
    def _make_scatter_kernel(numel, num_experts, block_size):
        max_padded = numel + (num_experts + 1) * (block_size - 1)
        ept = max(1, math.ceil(math.ceil(numel / 56) / THREADS))
        tile = THREADS * ept
        nblocks = math.ceil(numel / tile)

        @tilelang.jit(out_idx=[], target="ascend")
        def _scatter():
            @T.prim_func
            def _scatter_main(
                flat: T.Tensor([numel], "int32"),
                sorted_token_ids: T.Tensor([max_padded], "int32"),
                cumsum: T.Tensor([num_experts + 1], "int32"),
            ):
                with T.Kernel(nblocks) as bid, T.SimtVF(threads=THREADS):
                    for i in T.serial(ept):
                        for j in T.Parallel(THREADS):
                            idx = (bid * ept + i) * THREADS + j
                            eid = T.if_then_else(idx < numel, flat[T.min(idx, numel - 1)], -1)
                            if 0 <= eid and eid < num_experts:
                                slot = T.atomic_add(cumsum[eid], T.int32(1), return_prev=True)
                                sorted_token_ids[slot] = idx
            return _scatter_main

        return _scatter

    return _make_scatter_kernel(64, 8, 4)()


# K9 is invoked explicitly (see DESIGN.md §11.1), e.g.
#   python -c "import sys; sys.path.insert(0, 'history_version'); \
#              import design_probe_primitives as P; k = P.k9_factory_form()"
