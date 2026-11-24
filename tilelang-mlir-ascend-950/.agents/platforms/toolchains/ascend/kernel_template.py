"""Ascend backend example with layered checks, using elementwise absolute value."""

import argparse
import functools

import tilelang
import tilelang.ascend.language as T
import torch
import torch_npu  # noqa: F401 -- registers torch.npu


@functools.lru_cache
def build(n):
    tile = 256
    full, tail = divmod(n, tile)
    grid = full + bool(tail)

    @T.prim_func
    def main(X: T.Tensor((n,), "float32"), Y: T.Tensor((n,), "float32")):
        with T.Kernel(grid) as bx:
            x = T.alloc_shared((tile,), "float32")
            y = T.alloc_shared((tile,), "float32")
            if bx < full:
                start = bx * tile
                T.copy(X[start : start + tile], x)
                with T.SimtVF(threads=256):
                    for i in T.Parallel(tile):
                        y[i] = T.if_then_else(x[i] < 0, -x[i], x[i])
                T.copy(y, Y[start : start + tile])
            if tail:  # noqa: SIM102 - keep the compile-time tail test separate
                if bx == full:
                    T.copy(X[full * tile : n], x[:tail])
                    with T.SimtVF(threads=256):
                        for i in T.Parallel(tail):
                            y[i] = T.if_then_else(x[i] < 0, -x[i], x[i])
                    T.copy(y[:tail], Y[full * tile : n])

    return tilelang.compile(main, target="ascend", out_idx=-1)


def golden_abs(x):
    return torch.abs(x)


def check(n):
    fn, args = profile_workload(n)
    torch.testing.assert_close(fn(*args), golden_abs(*args))


def profile_workload(n):
    x = torch.randn(n, dtype=torch.float32, device="npu")
    return build(n), (x,)


def run_L0():
    check(256)


def run_L1():
    check(4096)


def run_L2():
    check(65536)


def run_boundary():
    check(1)
    check(257)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    checks = {"L0": run_L0, "L1": run_L1, "L2": run_L2, "boundary": run_boundary}
    parser.add_argument("--level", choices=[*checks, "all"], default="all")
    level = parser.parse_args().level
    for run in checks.values() if level == "all" else [checks[level]]:
        run()
    print("[PRECISION_PASS]")
