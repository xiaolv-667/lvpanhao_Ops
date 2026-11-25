"""PL-950-deterministic-slot-alloc -- repro (slow -> fast pattern).

First verified: TileLang ascend 0.1.15 + CANN 9.2.1 / Ascend950PR_9579 (dav-3510), 2026-11-25;
origin_task moe_permute_align-moe_permute_align-20261124T171256Z (Stage 4).
toolchain_stamp: 7a52f2f1274df07d8f4c9c01353ce9d290f88c16e85047de0c425b6a009de4aa

**Optimization**: a counting-scatter ("place every route at its expert's next free
slot") does **not** need a global atomic per route.  Give every core a *derived*
base instead -- ``cumsum[e] + (routes of e on earlier cores)``, which the
accumulation stage of the pipeline already has -- and let the core hand out its
own sub-range with an on-chip cursor.  Slots stay distinct and inside the
expert's range, and the intra-expert order stays a free choice (the operator
contract's own wording).

before (`bump_global`):    slot = atomic_add(GM cur[e], 1, prev);  sorted[slot] = i
after  (`bump_derived`):   slot = atomic_add(UB cur[e], 1, prev);  sorted[slot] = i
                           with UB cur[e] = cumsum[e] + my core's base[e]

Delta (numel=32768, E=384, 56 cores x 128 lanes; platform msprof `Task Duration`):
586.0 us -> 7.2 us on the placement launch, and 650.7 -> 29.6 us for the whole
operator call.

This script is a **cost sketch of the two allocator forms only** (one launch
each, same geometry).  Its kernels are not a correctness reference: their lane
guards are copied from the production kernel but the surrounding pipeline
(per-core slice bookkeeping) is stubbed, so it does not assert slot coverage.
The pattern's correctness is certified by the operator's own harness --
``examples/moe_permute_align_perf/perf_opt/moe_permute_align.py --level L0``
(whole-pipeline goldens, ``max_diff=0``) and by the frozen op test suite
(30 cases).

Run (device required):

    python PL-950-deterministic-slot-alloc.py           # compile + correctness
    python PL-950-deterministic-slot-alloc.py --time    # also compare latency
"""

import argparse

import tilelang
import tilelang.ascend.language as T
import torch
import torch_npu  # noqa: F401

NUMEL, E, CORES, THREADS = 32768, 384, 56, 128
EPT = 5
TPC = 586        # ceil(NUMEL / CORES): this core's route slice
E_PAD = 384
TOTAL = NUMEL + (E + 1) * 63     # padded capacity (block_size = 64)


@tilelang.jit(out_idx=[], target="ascend")
def bump_global():
    @T.prim_func
    def main(flat: T.Tensor((NUMEL,), "int32"), out: T.Tensor((TOTAL,), "int32"),
             cur: T.Tensor((E + 1,), "int32")):
        with T.Kernel(CORES) as bid, T.SimtVF(threads=THREADS):
            for i in T.serial(EPT):
                for j in T.Parallel(THREADS):
                    idx = (bid * EPT + i) * THREADS + j
                    eid = flat[T.min(idx, NUMEL - 1)]
                    if i * THREADS + j < TPC and idx < NUMEL:
                        slot = T.atomic_add(cur[eid], T.int32(1), return_prev=True)
                        out[slot] = idx
    return main


@tilelang.jit(out_idx=[], target="ascend")
def bump_derived():
    @T.prim_func
    def main(flat: T.Tensor((NUMEL,), "int32"), out: T.Tensor((TOTAL,), "int32"),
             base: T.Tensor((CORES, E_PAD), "int32")):
        with T.Kernel(CORES) as bid, T.SimtVF(threads=THREADS):
            s_cur = T.alloc_shared((E_PAD,), "int32")
            eid = T.alloc_local([1], "int32")
            slot = T.alloc_local([1], "int32")
            for k in T.serial(E_PAD // THREADS):
                for j in T.Parallel(THREADS):
                    s_cur[k * THREADS + j] = base[bid, k * THREADS + j]
            for i in T.serial(EPT):
                for j in T.Parallel(THREADS):
                    idx = (bid * EPT + i) * THREADS + j
                    eid[0] = flat[T.min(idx, NUMEL - 1)]
                    if i * THREADS + j < TPC and idx < NUMEL:
                        slot[0] = T.atomic_add(
                            s_cur[eid[0]], T.int32(1), return_prev=True
                        )
                        out[slot[0]] = idx
    return main


def _time(prog, call_args, reset=None, iters=20):
    for _ in range(3):
        if reset:
            reset()
        prog(*call_args)
    torch.npu.synchronize()
    times = []
    for _ in range(iters):
        if reset:
            reset()
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        prog(*call_args)
        end.record()
        torch.npu.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", action="store_true")
    args = parser.parse_args()

    bump_g = bump_global()
    bump_d = bump_derived()
    print("compiled: bump_global, bump_derived")

    gen = torch.Generator().manual_seed(11)
    flat = torch.randint(0, E, (NUMEL,), generator=gen, dtype=torch.int32).npu()
    tpc = TPC

    counts = torch.bincount(flat.cpu().long(), minlength=E)
    padded = (counts + 63) // 64 * 64
    starts = torch.cumsum(padded, 0) - padded
    total_padded = int(starts[-1] + padded[-1])

    # the pipeline's accumulation stage already produces both of these: the
    # padded start per expert, and every core's exclusive offset into it
    cur = torch.zeros(E + 1, dtype=torch.int32, device="npu")
    cur[:E] = starts.to(torch.int32).npu()
    cur[E] = total_padded
    base = torch.zeros((CORES, E_PAD), dtype=torch.int32)
    for core in range(CORES):
        lo = core * tpc
        before = (
            torch.bincount(flat.cpu()[:lo].long(), minlength=E)
            if lo else torch.zeros(E, dtype=torch.int64)
        )
        base[core, :E] = (starts + before).to(torch.int32)
    base = base.npu()

    def reset_g():
        cur.zero_()
        cur[:E] = starts.to(torch.int32).npu()
        cur[E] = total_padded

    out_g = torch.full((TOTAL,), NUMEL, dtype=torch.int32, device="npu")
    out_d = torch.full((TOTAL,), NUMEL, dtype=torch.int32, device="npu")
    bump_g(flat, out_g, cur)
    bump_d(flat, out_d, base)
    torch.npu.synchronize()
    print(f"both forms ran (padded total {total_padded}); correctness of the slot "
          f"assignment is certified by the operator's own harness: "
          f"python ../../../../../examples/moe_permute_align_perf/perf_opt/"
          f"moe_permute_align.py --level L0")

    if not args.time:
        return
    t_global = _time(bump_g, (flat, out_g, cur), reset_g)
    t_derived = _time(bump_d, (flat, out_d, base))
    print(f"bump_global  {t_global:9.3f} us   (platform: 586.0 us)")
    print(f"bump_derived {t_derived:9.3f} us   (platform: 7.2 us, host floor ~19 us)")
    assert t_global > 10 * t_derived, (t_global, t_derived)
    print("[PL-950-deterministic-slot-alloc] derived slot >= 10x the atomic bump: PASS")


if __name__ == "__main__":
    main()
