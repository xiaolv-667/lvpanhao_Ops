"""TRAP-950-partial-gm2ub-load: a GM->UB copy rounds its length up.
First verified: TileLang ascend 0.1.15 + CANN 9.2.1 / Ascend950PR_9579 (dav-3510), 2026-11-23; origin_task batch2-20261123.

Probe: load ``n`` int32 from GM into a UB tile that was pre-filled with a
canary, then store the whole tile back out.  Whatever the load wrote beyond
``n`` is visible as canary loss.  Run from the task directory with the ascend
env sourced and one NPU visible:

    /workspace/venv-ascend950/bin/python repro/TRAP-950-partial-gm2ub-load.py
"""

import tilelang
import tilelang.ascend.language as T
import torch
import torch_npu  # noqa: F401 - registers torch.npu

CANARY = -999
TILE = 64


def build(n):
    @T.prim_func
    def main(X: T.Tensor((n,), "int32"), Y: T.Tensor((TILE,), "int32")):
        with T.Kernel(1):
            s = T.alloc_shared((TILE,), "int32")
            with T.SimtVF(threads=32):
                for i in T.Parallel(TILE):
                    s[i] = CANARY
            T.copy(X[0:n], s[0:n])
            T.copy(s, Y)

    return tilelang.compile(main, target="ascend", out_idx=[])


def main():
    for n in (1, 2, 7, 8, 9, 16, 23, 24, 1000 % 64):
        if n < 1:
            continue
        x = torch.full((n,), 7, dtype=torch.int32, device="npu")
        y = torch.full((TILE,), -1, dtype=torch.int32, device="npu")
        build(n)(x, y)
        torch.npu.synchronize()
        out = y.cpu().tolist()
        spilled = [i for i in range(TILE) if i >= n and out[i] != CANARY]
        # reading past ``x`` is undefined but the whole 32-byte burst lands:
        print(
            f"n={n:4d} -> positions past n overwritten: {len(spilled)} "
            f"({spilled[:8]}...) values={[out[i] for i in spilled[:8]]}"
        )
        assert out[:n] == [7] * n, f"payload wrong for n={n}: {out[:n]}"
        if n % 8 == 0:
            assert not spilled, f"exact multiple of 8 still spilled: {spilled}"
        else:
            assert len(spilled) == 8 - n % 8, (
                f"n={n} spilled {len(spilled)} slots, expected {8 - n % 8}: {spilled}"
            )
    print("TRAP-950-partial-gm2ub-load: the load occupies ceil(n/8)*8 int32 slots")
    print("[PASS] reproduced")


if __name__ == "__main__":
    main()
