# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Ascend 950: four constructs one rope-shaped SIMT kernel needs, in one tiny program.
First verified: TileLang ascend 0.1.15 + CANN 9.2.1 / Ascend950PR_9579 (dav-3510), 2026-11-23; origin_task batch2-20261123.

Every fact below was required by ``examples/rope/_rope_positions_kernel`` and had
no operator-level precedent on this toolchain (the design listed the row index as
risk R1 and the branch/ABI facts as unverified):

1. ``@T.macro`` takes **buffers and Python ints** as arguments (the rope body is
   traced twice, once per side, so it cannot be inlined by hand without drift).
2. A **UB-resident scalar buffer value indexes a GM row**: ``cs[i, k] =
   TBL[row_ub[i], k]`` with ``row_ub`` written by an earlier ``T.Parallel`` pass.
   The only prior precedent was a GM value used as a GM row index
   (``paged_cache_write``: ``slot = T.cast(Slots[token], ...)``).
3. A **block-level runtime if/else opens ``T.SimtVF`` in each branch** and the two
   branches emit different static tile widths (this is how one launch serves both
   query and key).
4. ``out_idx`` with a **``T.dynamic`` extent** allocates and returns the output
   with the caller's actual token count.

Asserts the numeric result exactly (atol=rtol=0) including the row tail block, so
a silent mis-mapping of the folded grid would fail rather than pass quietly.

Run from any directory after sourcing ``.agents/platforms/toolchains/ascend/env-950pr.sh``:
    /workspace/venv-ascend950/bin/python repro/CONST-950-ub-value-as-gm-row-index.py
"""

import torch
import torch_npu  # noqa: F401  -- registers torch.npu
import tilelang
import tilelang.ascend.language as T

_THREADS = 256
P, HALF, BT, COLS = 8, 8, 4, 16  # table rows, table width, rows a block, columns a chunk
NCH = 2  # 2 chunks of 16 columns = 32 columns a side


def build():
    tokens = T.dynamic("num_tokens")
    nrb = T.ceildiv(tokens, BT)
    nq = nrb * NCH
    nblocks = nq + nrb * NCH

    # Fact 1: buffers and ints as macro arguments.
    @T.macro
    def side(X, Y, POS, TBL, row_ub, cs_ub, x_ub, y_ub, cbase, rb, cb):
        c0 = cbase + cb * COLS
        row0 = rb * BT
        for i in T.Parallel(BT):
            r = T.min(row0 + i, tokens - 1)
            row_ub[i] = T.cast(T.max(0, T.min(T.cast(POS[r], "int64"), P - 1)), "int32")
        # Fact 2: the UB value indexes a GM row.
        for i, k in T.Parallel(BT, HALF):
            cs_ub[i, k] = TBL[row_ub[i], k]
        for i, c in T.Parallel(BT, COLS):
            if row0 + i < tokens:
                x_ub[i, c] = X[row0 + i, c0 + c]
        for i, c in T.Parallel(BT, COLS):
            if row0 + i < tokens:
                h = c // HALF
                k = c - h * HALF
                Y[row0 + i, c0 + c] = x_ub[i, c] * cs_ub[i, k]

    @T.prim_func
    def main(
        POS: T.Tensor((tokens,), "int32"),
        X: T.Tensor((tokens, 2 * NCH * COLS), "float32"),
        TBL: T.Tensor((P, HALF), "float32"),
        Y: T.Tensor((tokens, 2 * NCH * COLS), "float32"),
    ) -> None:
        with T.Kernel(nblocks) as bx:
            # Every buffer is allocated before the branch: a conditional
            # allocation is not legal on this backend.
            x_ub = T.alloc_shared((BT, COLS), "float32")
            y_ub = T.alloc_shared((BT, COLS), "float32")
            cs_ub = T.alloc_shared((BT, HALF), "float32")
            row_ub = T.alloc_shared((BT,), "int32")
            # Fact 3: a runtime branch, each side with its own thread domain.
            if bx < nq:
                with T.SimtVF(threads=_THREADS):
                    side(X, Y, POS, TBL, row_ub, cs_ub, x_ub, y_ub, 0, bx // NCH, bx % NCH)
            else:
                with T.SimtVF(threads=_THREADS):
                    t = bx - nq
                    side(
                        X,
                        Y,
                        POS,
                        TBL,
                        row_ub,
                        cs_ub,
                        x_ub,
                        y_ub,
                        NCH * COLS,
                        t // NCH,
                        t % NCH,
                    )

    # Fact 4: callee-allocated output with a dynamic extent.
    return tilelang.compile(main, target="ascend", out_idx=[3])


def main():
    fn = build()
    tokens = 10  # not a multiple of BT: the last row block is a tail block
    gen = torch.Generator().manual_seed(0)
    pos = torch.tensor([7, -3, 0, 5, 9, 1, 2, 6, 4, 3], dtype=torch.int32)  # clamp too
    x = torch.randn((tokens, 2 * NCH * COLS), generator=gen, dtype=torch.float32)
    tbl = torch.randn((P, HALF), generator=gen, dtype=torch.float32)
    rows = pos.long().clamp(0, P - 1)
    gold = x * tbl[rows].repeat(1, 8)  # 8 half-blocks across 2 * NCH * COLS columns
    y = fn(pos.npu(), x.npu(), tbl.npu())
    torch.npu.synchronize()
    got = y.cpu()
    assert tuple(got.shape) == (tokens, 2 * NCH * COLS), got.shape
    torch.testing.assert_close(got, gold, atol=0, rtol=0)
    print(f"[CONST-950-ub-value-as-gm-row-index] shape={tuple(got.shape)} exact match")
    print("[PASS]")


if __name__ == "__main__":
    main()
