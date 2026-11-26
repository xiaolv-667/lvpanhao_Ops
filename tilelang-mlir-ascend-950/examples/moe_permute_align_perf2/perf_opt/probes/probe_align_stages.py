"""Ablation of the align launch (r6 layout) to locate its residual cost.

Builds the align kernel of ``MoEPermuteAlignFwdOp_r6.py`` for the largest
workload geometry (numel=32768, E=384, B=64; 14 align cores, 56 placement rows)
with cumulatively enabled stages and measures each with the platform protocol
(one launch, kernel `Task Duration`).

    python probe_align_stages.py            # compile + measure all stages
    python probe_align_stages.py --stage 4  # one stage only
"""

import argparse
import os
import sys

import tilelang
import tilelang.ascend.language as T
import torch
import torch_npu  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from MoEPermuteAlignFwdOp_r6 import (  # noqa: E402
    FILL_TILE,
    _align_plan,
    _align_up,
    _fused_plan,
    _geometry,
    _THREADS,
)

NUMEL, E, B = 32768, 384, 64
G_ALIGN, G_PLACE, E_PAD, _GRID = _align_plan(NUMEL, E, B)
GEO = _geometry(NUMEL, E, B)
P_TOTAL, NB, STEPS, SHIFT = GEO["P"], GEO["NB"], GEO["steps"], GEO["shift"]
BLK = max(64, _align_up(max((NB + G_ALIGN - 1) // G_ALIGN, 1), 64))
EPC = (E + G_ALIGN - 1) // G_ALIGN
E_LANES = _align_up(max(EPC, 1), 64)
RPC = (G_PLACE + G_ALIGN - 1) // G_ALIGN

FACTORIES = {}


def build(stage):
    if stage in FACTORIES:
        return FACTORIES[stage]()

    @tilelang.jit(out_idx=[], target="ascend")
    def factory():
        @T.prim_func
        def main(
            flat: T.Tensor((NUMEL,), "int32"),
            partial: T.Tensor((G_PLACE * E_PAD,), "int32"),
            cursor: T.Tensor((G_PLACE * E_PAD,), "int32"),
            sorted_token_ids: T.Tensor((P_TOTAL,), "int32"),
            expert_ids: T.Tensor((NB,), "int32"),
            num_tokens_post_pad: T.Tensor((1,), "int32"),
            cumsum: T.Tensor((E + 1,), "int32"),
        ):
            with T.Kernel(G_ALIGN) as a:
                s_all = T.alloc_shared((G_PLACE * E_PAD,), "int32")
                s_cnt = T.alloc_shared((E_PAD,), "int32")
                s_pad = T.alloc_shared((E_PAD,), "int32")
                s_scan_a = T.alloc_shared((2 * E_PAD,), "int32")
                s_scan_b = T.alloc_shared((2 * E_PAD,), "int32")
                s_pref = T.alloc_shared((E_PAD + 64,), "int32")

                T.copy(partial, s_all)
                if stage == 0:
                    with T.SimtVF(threads=E_PAD):
                        for e in T.Parallel(E_PAD):
                            cumsum[e] = s_all[e]

                if stage >= 1:  # merge
                    with T.SimtVF(threads=E_PAD):
                        for e in T.Parallel(E_PAD):
                            tot = T.alloc_local([1], "int32")
                            tot[0] = 0
                            for r in range(G_PLACE):
                                tot[0] = tot[0] + s_all[r * E_PAD + e]
                            s_cnt[e] = tot[0]
                    if stage == 1:
                        with T.SimtVF(threads=E_PAD):
                            for e in T.Parallel(E_PAD):
                                cumsum[e] = s_cnt[e]

                if stage >= 2:  # pad + scan + exclusive prefix
                    with T.SimtVF(threads=E_PAD):
                        for e in T.Parallel(E_PAD):
                            pv = T.if_then_else(e < E, s_cnt[e], 0)
                            s_pad[e] = ((pv + (B - 1)) >> SHIFT) << SHIFT
                    with T.SimtVF(threads=E_PAD):
                        for e in T.Parallel(E_PAD):
                            s_scan_a[E_PAD + e] = s_pad[e]
                    with T.SimtVF(threads=E_PAD):
                        for e in T.Parallel(E_PAD):
                            s_scan_a[e] = 0
                    with T.SimtVF(threads=E_PAD):
                        for e in T.Parallel(E_PAD):
                            s_scan_b[e] = 0
                    with T.SimtVF(threads=E_PAD):
                        for e in T.Parallel(E_PAD):
                            s_scan_b[E_PAD + e] = 0
                    for d in range(STEPS):
                        k = 1 << d
                        if d % 2 == 0:
                            with T.SimtVF(threads=E_PAD):
                                for e in T.Parallel(E_PAD):
                                    s_scan_b[E_PAD + e] = (
                                        s_scan_a[E_PAD + e] + s_scan_a[E_PAD + e - k]
                                    )
                        else:
                            with T.SimtVF(threads=E_PAD):
                                for e in T.Parallel(E_PAD):
                                    s_scan_a[E_PAD + e] = (
                                        s_scan_b[E_PAD + e] + s_scan_b[E_PAD + e - k]
                                    )
                    if STEPS % 2 == 0:
                        with T.SimtVF(threads=E_PAD + 64):
                            for e in T.Parallel(E_PAD + 64):
                                if e <= E_PAD:
                                    s_pref[e] = T.if_then_else(
                                        e == 0, 0, s_scan_a[E_PAD + e - 1]
                                    )
                    else:
                        with T.SimtVF(threads=E_PAD + 64):
                            for e in T.Parallel(E_PAD + 64):
                                if e <= E_PAD:
                                    s_pref[e] = T.if_then_else(
                                        e == 0, 0, s_scan_b[E_PAD + e - 1]
                                    )
                    total = s_pref[E]
                    if a == 0:
                        T.copy(s_pref[0:E + 1], cumsum[0:E + 1])
                        num_tokens_post_pad[0] = total

                if stage >= 3:  # cursor rows
                    r_lo = a * RPC
                    r_hi = T.min(G_PLACE, r_lo + RPC)
                    with T.SimtVF(threads=E_PAD):
                        for e in T.Parallel(E_PAD):
                            run = T.alloc_local([1], "int32")
                            run[0] = 0
                            for r in range(G_PLACE):
                                if r >= r_lo and r < r_hi:
                                    cursor[r * E_PAD + e] = s_pref[e] + run[0]
                                run[0] = run[0] + s_all[r * E_PAD + e]

                if stage >= 4:  # expert_ids
                    blkpc = (NB + G_ALIGN - 1) // G_ALIGN
                    blk0 = a * blkpc
                    with T.SimtVF(threads=BLK):
                        for j in T.Parallel(BLK):
                            b = blk0 + j
                            if b >= total // B and b < NB:
                                expert_ids[b] = E
                    e_lo = a * EPC
                    with T.SimtVF(threads=E_LANES):
                        for j in T.Parallel(E_LANES):
                            expert = e_lo + j
                            start = T.alloc_local([1], "int32")
                            cnt = T.alloc_local([1], "int32")
                            if expert < E:
                                start[0] = s_pref[expert] // B
                                cnt[0] = s_pref[expert + 1] // B - start[0]
                                for b in T.serial(cnt[0]):
                                    expert_ids[start[0] + b] = expert

        return main

    FACTORIES[stage] = factory
    return factory()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=None)
    args = ap.parse_args()

    from tileops.benchmark.msprof import bench_kernel_msprof

    os.environ.setdefault("TILEOPS_MSPROF_WARM_UP", "0")
    os.environ.setdefault("TILEOPS_MSPROF_LAUNCH_COUNT", "1")

    gen = torch.Generator().manual_seed(7)
    flat = torch.randint(0, E, (NUMEL,), generator=gen, dtype=torch.int32).npu()
    partial = torch.zeros(G_PLACE * E_PAD, dtype=torch.int32, device="npu")
    cursor = torch.zeros(G_PLACE * E_PAD, dtype=torch.int32, device="npu")
    srt = torch.full((P_TOTAL,), NUMEL, dtype=torch.int32, device="npu")
    ex = torch.zeros(NB, dtype=torch.int32, device="npu")
    npost = torch.zeros(1, dtype=torch.int32, device="npu")
    cs = torch.zeros(E + 1, dtype=torch.int32, device="npu")
    argv = (flat, partial, cursor, srt, ex, npost, cs)

    stages = [args.stage] if args.stage is not None else [0, 1, 2, 3, 4]
    print(f"geometry: g_align={G_ALIGN} g_place={G_PLACE} e_pad={E_PAD} "
          f"window={G_PLACE * E_PAD * 4 / 1024:.0f} KB/core")
    for stage in stages:
        prog = build(stage)
        # bench_kernel_msprof imports the callable by module+name, so each stage
        # gets its own module-level thunk
        name = f"align_stage{stage}"
        globals()[name] = _thunk(prog, name)
        lat, _ = bench_kernel_msprof(globals()[name], args=argv)
        print(f"stage {stage}: {lat:8.3f} us")


def _thunk(prog, name):
    def fn(*args):
        return prog(*args)

    fn.__name__ = name
    fn.__qualname__ = name
    return fn


if __name__ == "__main__":
    main()
