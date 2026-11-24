"""TRAP-950-fp8-scalar-store: a scalar e4m3 GM store writes 0 on this backend.
First verified: TileLang ascend 0.1.15 + CANN 9.2.1 / Ascend950PR_9579 (dav-3510), 2026-11-23; origin_task batch2-20261123.

Measured 2026-11-23 on ascend-950pr-9579 (dav-3510), TileLang ascend 0.1.15,
CANN 9.2.1, torch 2.13.0 / torch_npu 2.13.0.rc1.

Claim under test
----------------
A direct fp8_e4m3fn store to GM is correct only when the store is vectorised.
Two independent ways of forcing a *scalar* (one element per lane) e4m3 store
both write 0 instead of the value:

1. A single-store body whose parallel extent is one element per thread
   (``block_rows * cols == threads``).
2. A body with two fp8 stores (KPages and VPages) in the same ``T.Parallel``
   loop: the vectoriser gives up, both stores lower to scalar e4m3 stores and
   both write 0.

The 16-bit (float16/bfloat16) direct store with two stores in one loop body is
unaffected, so the trap is specific to e4m3 scalar stores.

Workaround verified here
------------------------
Split the two pools into two ``T.Parallel`` loops inside the same
``T.SimtVF`` block: each loop then has a single fp8 store, is vectorised, and
writes the exact value.

Run
---
    cd <task_dir> && source .agents/platforms/toolchains/ascend/env-950pr.sh
    /workspace/venv-ascend950/bin/python repro/TRAP-950-fp8-scalar-store.py
"""

import torch
import torch_npu  # noqa: F401
import tilelang
import tilelang.ascend.language as T

FP8 = "float8_e4m3fn"
LIM = 448.0
THREADS = 128
RW = 64  # row width of every variant below


def build(variant, block_rows, block_cols, threads=THREADS, rw=RW):
    cols = block_cols if (rw % block_cols == 0 and block_cols <= rw) else rw
    nchunks = (rw + cols - 1) // cols
    tokens = T.dynamic("tokens")
    nblocks = T.ceildiv(tokens, block_rows) * nchunks

    def quantize(x):
        return T.cast(T.clamp(T.cast(x, "float32") / 1.25, -LIM, LIM), FP8)

    if variant == "one_store":

        @T.prim_func
        def main(
            K: T.Tensor((tokens, rw), "float16"),
            KP: T.Tensor((8, rw), FP8),
            Slots: T.Tensor((tokens,), "int64"),
        ) -> None:
            with T.Kernel(nblocks) as bx, T.SimtVF(threads=threads):
                rb = bx // nchunks
                cb = bx % nchunks
                for i, c in T.Parallel(block_rows, cols):
                    token = rb * block_rows + i
                    if token < tokens:
                        slot = T.cast(Slots[token], "int32")
                        if slot >= 0:
                            KP[slot, cb * cols + c] = quantize(K[token, cb * cols + c])

    elif variant == "two_stores_one_loop":

        @T.prim_func
        def main(
            K: T.Tensor((tokens, rw), "float16"),
            V: T.Tensor((tokens, rw), "float16"),
            KP: T.Tensor((8, rw), FP8),
            VP: T.Tensor((8, rw), FP8),
            Slots: T.Tensor((tokens,), "int64"),
        ) -> None:
            with T.Kernel(nblocks) as bx, T.SimtVF(threads=threads):
                rb = bx // nchunks
                cb = bx % nchunks
                for i, c in T.Parallel(block_rows, cols):
                    token = rb * block_rows + i
                    if token < tokens:
                        slot = T.cast(Slots[token], "int32")
                        if slot >= 0:
                            KP[slot, cb * cols + c] = quantize(K[token, cb * cols + c])
                            VP[slot, cb * cols + c] = quantize(V[token, cb * cols + c])

    else:  # two_loops

        @T.prim_func
        def main(
            K: T.Tensor((tokens, rw), "float16"),
            V: T.Tensor((tokens, rw), "float16"),
            KP: T.Tensor((8, rw), FP8),
            VP: T.Tensor((8, rw), FP8),
            Slots: T.Tensor((tokens,), "int64"),
        ) -> None:
            with T.Kernel(nblocks) as bx, T.SimtVF(threads=threads):
                rb = bx // nchunks
                cb = bx % nchunks
                for i, c in T.Parallel(block_rows, cols):
                    token = rb * block_rows + i
                    if token < tokens:
                        slot = T.cast(Slots[token], "int32")
                        if slot >= 0:
                            KP[slot, cb * cols + c] = quantize(K[token, cb * cols + c])
                for i, c in T.Parallel(block_rows, cols):
                    token = rb * block_rows + i
                    if token < tokens:
                        slot = T.cast(Slots[token], "int32")
                        if slot >= 0:
                            VP[slot, cb * cols + c] = quantize(V[token, cb * cols + c])

    return tilelang.compile(main, target="ascend", out_idx=[])


def run(variant, block_rows, block_cols):
    per_thread = (
        block_rows
        * (block_cols if (RW % block_cols == 0 and block_cols <= RW) else RW)
        / THREADS
    )
    gen = torch.Generator().manual_seed(5)
    k = torch.randn((8, RW), generator=gen, dtype=torch.float32).to(torch.float16)
    v = torch.randn((8, RW), generator=gen, dtype=torch.float32).to(torch.float16)
    sentinel = 7.0
    kp = (
        torch.full((8, RW), sentinel, dtype=torch.float32).to(torch.float8_e4m3fn).npu()
    )
    vp = (
        torch.full((8, RW), sentinel, dtype=torch.float32).to(torch.float8_e4m3fn).npu()
    )
    slots = torch.arange(8, dtype=torch.int64).npu()
    gold = torch.clamp(k.float() / 1.25, -LIM, LIM).to(torch.float8_e4m3fn)
    gold_v = torch.clamp(v.float() / 1.25, -LIM, LIM).to(torch.float8_e4m3fn)

    fn = build(variant, block_rows, block_cols)
    if variant == "one_store":
        fn(k.npu(), kp, slots)
    else:
        fn(k.npu(), v.npu(), kp, vp, slots)
    torch.npu.synchronize()

    k_ok = torch.equal(kp.cpu(), gold)
    v_ok = torch.equal(vp.cpu(), gold_v) if variant != "one_store" else True
    return per_thread, k_ok, v_ok, kp[0, 0].item()


def main():
    results = {}
    for variant, brows, bcols in (
        ("one_store", 16, 64),  # 8 elements per thread -> vectorised
        ("one_store", 2, 64),  # 1 element per thread  -> scalar
        ("two_stores_one_loop", 16, 64),  # two fp8 stores -> scalar fallback
        ("two_loops", 16, 64),  # split loops -> vectorised
    ):
        per_thread, k_ok, v_ok, first = run(variant, brows, bcols)
        results[(variant, brows)] = (per_thread, k_ok, v_ok)
        print(
            f"{variant:20s} block={brows}x{bcols} per_thread={per_thread:5.1f} "
            f"k_ok={k_ok} v_ok={v_ok} kp[0,0]={first:.4f}"
        )

    # 8 elements per thread with a single store: correct.
    assert results[("one_store", 16)] == (8.0, True, True), (
        "vectorised single store must be exact"
    )
    # 1 element per thread: the scalar e4m3 store writes 0, not the value.
    per_thread, k_ok, _ = results[("one_store", 2)]
    assert per_thread == 1.0 and not k_ok, "scalar e4m3 store is expected to write 0"
    # Two fp8 stores in one loop body: same scalar fallback, both pools wrong.
    assert results[("two_stores_one_loop", 16)] == (8.0, False, False), (
        "two stores must lose the values"
    )
    # Two loops: restored.
    assert results[("two_loops", 16)] == (8.0, True, True), "split loops must be exact"
    print(
        "TRAP-950-fp8-scalar-store: assertions hold "
        "(scalar e4m3 store writes 0; one loop per pool restores exactness)"
    )


if __name__ == "__main__":
    main()
