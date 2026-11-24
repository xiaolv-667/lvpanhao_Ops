"""CONST-950-vcvt-narrow-store-pack-b32: narrow vcvt needs PK_B32 + a b32 mask.
First verified: TileLang ascend 0.1.15 / CANN 9.2.1 runtime, Ascend950PR_9579.

Trap. `T.simd.vcvt(src_f32x64, "float16"|"bfloat16", part=0, sat=False)` returns a
128-lane 16-bit register whose 64 results occupy the *even* lanes (one value per
32-bit word).  A dense store must therefore pack 32-bit words down to their low
half: `dist="PK_B32"`.  `dist="NORM_B16"` writes every other slot (the odd
lanes), and `dist="PK_B16"` packs at the wrong granularity; both compile fine
and silently leave half of the row unwritten.

Mask. The pack store's predicate is a 32-bit-lane pattern.  A 16-bit pattern
(`pset(16, "PAT_VL64")`) enables only 32 of the 64 32-bit lanes, so exactly half
of the slots stay untouched -- still bit-correct where it writes, which is why
a test that reuses one process can read stale-but-right data and pass.

Detection trick used here: pre-fill the destination buffer with a sentinel
inside the kernel, so an unwritten slot cannot be confused with a correct one
(UB content survives between launches in one process).

Run: cd examples/mla_dense_bwd/_dq_land_kernel
     source ../../../.agents/platforms/toolchains/ascend/env-950pr.sh
     /workspace/venv-ascend950/bin/python repro/CONST-950-vcvt-narrow-store-pack-b32.py
"""

import tilelang
import tilelang.ascend.language as T
import torch
import torch_npu  # noqa: F401

DQ, ROWS = 64, 4
SCALE = 0.07216878364870322
SENTINEL = 1234.5


def build(dist, mask_bits):
    @T.prim_func
    def main(
        x: T.Tensor((ROWS, DQ), "float32"),
        sent: T.Tensor((ROWS, DQ), "float16"),
        y: T.Tensor((ROWS, DQ), "float16"),
    ):
        with T.Kernel(1):
            acc = T.alloc_shared((ROWS, DQ), "float32")
            out = T.alloc_shared((ROWS, DQ), "float16")
            T.copy(x[:, :], acc[0:ROWS, 0:DQ])
            T.copy(sent[:, :], out[0:ROWS, 0:DQ])  # sentinel fill, same buffer
            with T.SimdVF():
                m = T.simd.pset(mask_bits, "PAT_ALL" if mask_bits == 32 else "PAT_VL64")
                for i in T.serial(ROWS):
                    v = T.simd.vcvt(
                        T.simd.vmuls(T.simd.vld(acc[i, 0]), SCALE),
                        "float16",
                        part=0,
                        sat=False,
                    )
                    T.simd.vsts(out[i, 0], v, m, dist, extent=DQ)
            T.copy(out[0:ROWS, 0:DQ], y[:, :])

    return tilelang.compile(main, target="ascend", out_idx=[2])


def probe(dist, mask_bits):
    torch.manual_seed(0)
    x = torch.arange(ROWS * DQ, dtype=torch.float32).reshape(ROWS, DQ) / 7.0
    sent = torch.full((ROWS, DQ), SENTINEL, dtype=torch.float16)
    gold = (x * SCALE).to(torch.float16)
    got = build(dist, mask_bits)(x.npu(), sent.npu())
    torch.npu.synchronize()
    got = got.cpu()
    return (
        int((got.view(torch.int16) == gold.view(torch.int16)).sum()),
        int((got.view(torch.int16) == sent.view(torch.int16)).sum()),
        got.numel(),
    )


def main():
    ok, sent, total = probe("PK_B32", 32)
    print(f"[PK_B32 mask=32] correct {ok}/{total}, still sentinel {sent}")
    assert ok == total and sent == 0, "the documented dense form must fill every slot"

    ok, sent, total = probe("PK_B32", 16)
    print(f"[PK_B32 mask=16] correct {ok}/{total}, still sentinel {sent}")
    assert ok == total // 2 and sent == total // 2, "a 16-bit mask must halve the store"

    ok, sent, total = probe("NORM_B16", 16)
    print(f"[NORM_B16 mask=16] correct {ok}/{total}, still sentinel {sent}")
    assert ok < total // 4, "NORM_B16 of a narrowed register must not be dense"
    print("[CONST-950-vcvt-narrow-store-pack-b32] OK")


if __name__ == "__main__":
    main()
