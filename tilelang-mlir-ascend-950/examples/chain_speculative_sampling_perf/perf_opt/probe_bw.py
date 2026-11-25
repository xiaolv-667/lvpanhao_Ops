"""Stage 4 probe: what does one read of the probability data cost?

Two spellings of the same row-sum over a flat fp32 array, same bytes and same
grid, timed on device (wall clock; diagnostic only):

  lane : per-element GM load inside ``T.Parallel`` (what the Stage 3 fold does)
  dma  : ``T.copy`` of a tile into UB, then the same sum out of UB

Usage (task dir):
  python perf_opt/probe_bw.py --n 8257536 --grid 56 --tile 8192 --per 18
"""

import argparse
import time

import tilelang
import tilelang.ascend.language as T
import torch
import torch_npu  # noqa: F401

F32 = "float32"
THREADS = 128


def build(n, grid, tile, per, mode, threads=THREADS):
    total = n

    @T.prim_func
    def main(X: T.Tensor((n,), F32), out: T.Tensor((grid,), F32)) -> None:
        with T.Kernel(grid) as bx:
            ub = T.alloc_shared((tile,), F32)
            ts = T.alloc_shared((1,), F32)
            bs = T.alloc_shared((1,), "int32")
            with T.SimtVF(threads=threads):
                can = T.alloc_reducer((1,), F32, op="sum")
                s = T.alloc_fragment((1,), F32)
                ts[0] = T.float32(0)
                bs[0] = bx * per * tile
                for t in T.serial(per):
                    base = (bx * per + t) * tile
                    base_s = bs[0] + t * tile
                    if mode == "dma":
                        T.copy(X[base : base + tile], ub)
                    T.reducer_init(can)
                    for i in T.Parallel(tile):
                        if mode == "lane":
                            T.reducer_update(can[0], X[base + i])
                        elif mode == "lane_sbase":
                            T.reducer_update(can[0], X[base_s + i])
                        elif mode == "lane_sbase_pred":
                            T.reducer_update(
                                can[0],
                                T.if_then_else(
                                    t * tile + i < total, X[base_s + i], T.float32(0)
                                ),
                            )
                        elif mode == "lane_pred":
                            T.reducer_update(
                                can[0],
                                T.if_then_else(
                                    t * tile + i < total, X[base + i], T.float32(0)
                                ),
                            )
                        elif mode == "store_sbase":
                            ub[i] = X[base_s + i]
                            T.reducer_update(can[0], ub[i])
                        elif mode == "store_pred":
                            ub[i] = T.if_then_else(
                                t * tile + i < total, X[base_s + i], T.float32(0)
                            )
                            T.reducer_update(can[0], ub[i])
                        else:
                            T.reducer_update(can[0], ub[i])
                    T.finalize_reducer(can, s)
                    ts[0] = ts[0] + s[0]
                if T.get_thread_binding() == 0:
                    out[bx] = ts[0]

    return tilelang.compile(main, target="ascend", out_idx=[])


def timeit(fn, args, warmup=5, iters=20):
    for _ in range(warmup):
        fn(*args)
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=8257536)
    p.add_argument("--grid", type=int, default=56)
    p.add_argument("--tile", type=int, default=8192)
    p.add_argument("--per", type=int, default=18)
    p.add_argument("--threads", type=int, default=THREADS)
    p.add_argument("--modes", default="lane,dma")
    args = p.parse_args()

    n = args.grid * args.per * args.tile
    x = torch.rand(n, dtype=torch.float32, device="npu")
    out = torch.zeros(args.grid, dtype=torch.float32, device="npu")
    gb = n * 4 / 1e9
    print(f"n={n} grid={args.grid} tile={args.tile} per={args.per} bytes={gb*1e3:.2f} MB", flush=True)
    for mode in args.modes.split(","):
        fn = build(n, args.grid, args.tile, args.per, mode, args.threads)
        us = timeit(fn, (x, out))
        print(
            f"  {mode:6s} threads={args.threads:4d} tile={args.tile:6d}: {us:8.1f} us  "
            f"{gb/us*1e6:7.1f} GB/s aggregate  {gb/us*1e6/args.grid:6.2f} GB/s per core",
            flush=True,
        )


if __name__ == "__main__":
    main()
