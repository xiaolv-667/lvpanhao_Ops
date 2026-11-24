"""CONST-950-vcvt-rs-disable-ieee: f32->f16 narrowing is IEEE; `sat` is inert.
First verified: TileLang ascend 0.1.15 / CANN 9.2.1 runtime, Ascend950PR_9579.

Two documentation traps around `T.simd.vcvt(src_f32x64, "float16", sat=...)`:

1. The installed docstring says `sat=False` means "No sat; out-of-range values
   wrap".  On this device it does not wrap: `RS_DISABLE` keeps IEEE semantics,
   so the fp16 overflow threshold 65520 (halfway to the unrepresentable 65536)
   gives +Inf while 65519 still rounds to 65504 -- the same as torch's
   `.to(torch.float16)`.
2. `sat=True` (`RS_ENABLE`) is not a clamp either for this conversion pair: the
   measured output is identical to `sat=False`, i.e. also +Inf past 65520.  The
   flag reaches the device code (the two builds emit different instructions) but
   does not change the result here.

Consequence for a `cast(fp32(x) * scale)` operator: both settings reproduce a
torch golden bit for bit on this toolchain, and `sat=False` is the safer choice
to declare because it states the intended IEEE semantics.

Measured (scale = 192 ** -0.5, operand pre-divided by the scale so the float32
product sits exactly on the edge):

    product edge | sat=False | sat=True | torch fp16
             65504 |     65504 |    65504 |      65504
             65519 |     65504 |    65504 |      65504
             65520 |       inf |      inf |        inf
             1e5   |       inf |      inf |        inf

Run: cd examples/mla_dense_bwd/_dq_land_kernel
     source ../../../.agents/platforms/toolchains/ascend/env-950pr.sh
     /workspace/venv-ascend950/bin/python repro/CONST-950-vcvt-rs-disable-ieee.py
"""

import tilelang
import tilelang.ascend.language as T
import torch
import torch_npu  # noqa: F401

DQ, ROWS = 64, 1
SCALE = 0.07216878364870322
EDGES = [65504.0, 65519.0, 65520.0, 1.0e5]  # in the *product* domain


def build(sat):
    @T.prim_func
    def main(
        x: T.Tensor((ROWS, DQ), "float32"),
        y: T.Tensor((ROWS, DQ), "float16"),
    ):
        with T.Kernel(1):
            acc = T.alloc_shared((ROWS, DQ), "float32")
            out = T.alloc_shared((ROWS, DQ), "float16")
            T.copy(x[:, :], acc[0:ROWS, 0:DQ])
            with T.SimdVF():
                m = T.simd.pset(32, "PAT_ALL")
                v = T.simd.vcvt(
                    T.simd.vmuls(T.simd.vld(acc[0, 0]), SCALE),
                    "float16",
                    part=0,
                    sat=sat,
                )
                T.simd.vsts(out[0, 0], v, m, "PK_B32", extent=DQ)
            T.copy(out[0:ROWS, 0:DQ], y[:, :])

    return tilelang.compile(main, target="ascend", out_idx=[1])


def run(sat):
    # Operand = edge / scale so that the float32 product is the edge itself.
    vals = [e / SCALE for e in EDGES] + [0.0] * (DQ - len(EDGES))
    x = torch.tensor([vals], dtype=torch.float32)
    got = build(sat)(x.npu())
    torch.npu.synchronize()
    return [got.cpu()[0, k].item() for k in range(len(EDGES))]


def main():
    off = run(False)
    on = run(True)
    print("product edge |  sat=False  |   sat=True  |  torch(fp16)")
    for edge, a, b in zip(EDGES, off, on):
        print(
            f"{edge:>12.0f} | {a:>11} | {b:>11} | "
            f"{torch.tensor(edge).to(torch.float16).item():>10}"
        )
    assert off[0] == 65504.0 and off[1] == 65504.0, "below the threshold must stay finite"
    assert off[2] == float("inf") and off[3] == float("inf"), "past 65520 must be +inf"
    assert on == off, "sat must not change the f32->f16 result on this device"
    print("[CONST-950-vcvt-rs-disable-ieee] OK")


if __name__ == "__main__":
    main()
