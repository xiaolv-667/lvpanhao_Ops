"""CONST-950-vcvt-sat-false-overflow-inf: narrowed ``vcvt(sat=False)`` overflows to inf.

First verified: TileLang ascend 0.1.15 / CANN 9.2.1 runtime, Ascend950PR_9579.

Question this settles. ``T.simd.vcvt``'s module docstring says ``sat=False``
means "no saturation; out-of-range values wrap", while the PyTorch golden of a
16-bit elementwise kernel rounds out-of-range values to ``+/-inf`` (IEEE RNE).
If the device really wrapped, a kernel whose output can exceed the dtype range
(``silu(g) * u`` with a large ``u``) would differ from the golden only for
inputs that random data never produces -- a silent correctness hole.

Measured answer on this toolchain: **``sat=False`` rounds to IEEE ``inf``, it
does not wrap.**  f32 -> f16 and f32 -> bf16 both match ``Tensor.to(dtype)``
bit for bit across the rounding boundary (65519 stays finite, 65520 becomes
inf) and for much larger magnitudes (1e5, 1e30, +/-inf, NaN).  Therefore the
golden form ``(silu(g) * u).to(dtype)`` is the correct reference for the
narrowing step and no clamping is needed.

Layout: one 64-lane chunk per row, ``vld`` (fp32) -> ``vcvt(ROUND_R,
sat=False, part=0)`` -> ``vsts(dist="PK_B32")``, i.e. exactly the negative path
of CONST-950-vcvt-narrow-store-pack-b32.

Run: cd examples/shared_expert_mlp/_silu_mul_fused_kernel
     source ../../../.agents/platforms/toolchains/ascend/env-950pr.sh
     ASCEND_RT_VISIBLE_DEVICES=0 /workspace/venv-ascend950/bin/python repro/CONST-950-vcvt-sat-false-overflow-inf.py
"""

import tilelang
import tilelang.ascend.language as T
import torch
import torch_npu  # noqa: F401  # registers torch.npu

ROWS, DQ = 2, 64

# Row 0: the fp16 overflow regime, row 1: values that must stay unchanged.
FP16_VALUES = [
    65504.0,  # fp16 max, finite
    65519.0,  # just below the round-to-inf threshold, finite
    65520.0,  # threshold: rounds to inf
    65600.0,
    1e5,
    1e30,
    -65504.0,
    -65520.0,
    -1e30,
    1.0,
    0.0,
    -0.0,
]
BF16_VALUES = [
    3.3895e38,  # ~bf16 max, finite
    3.3960e38,  # just below the bf16 round-to-inf threshold
    3.4000e38,  # rounds to inf
    3.5e38,
    1e39,
    -3.3895e38,
    -3.4000e38,
    -1e39,
    1.0,
    0.0,
]


def build(dtype):
    @T.prim_func
    def main(
        x: T.Tensor((ROWS, DQ), "float32"),
        y: T.Tensor((ROWS, DQ), dtype),  # type: ignore[valid-type]
    ):
        with T.Kernel(1):
            acc = T.alloc_shared((ROWS, DQ), "float32")
            out = T.alloc_shared((ROWS, DQ), dtype)
            T.copy(x[:, :], acc[0:ROWS, 0:DQ])
            with T.SimdVF():
                full = T.simd.pset(32)  # 32-bit lane mask: PK_B32 packs words
                for r in T.serial(ROWS):
                    T.simd.vsts(
                        out[r, 0],
                        T.simd.vcvt(
                            T.simd.vld(acc[r, 0]), dtype, round="ROUND_R", sat=False, part=0
                        ),
                        full,
                        dist="PK_B32",
                    )
            T.copy(out[0:ROWS, 0:DQ], y[:, :])

    return tilelang.compile(main, target="ascend", out_idx=[1])


def probe(dtype_str, torch_dtype, row0, row1):
    x = torch.zeros(ROWS, DQ, dtype=torch.float32)
    x[0, : len(row0)] = torch.tensor(row0, dtype=torch.float32)
    x[1, : len(row1)] = torch.tensor(row1, dtype=torch.float32)
    want = x.to(torch_dtype)  # IEEE RNE, out of range -> +/-inf
    got = build(dtype_str)(x.npu())
    torch.npu.synchronize()
    got = got.cpu()

    same = got.view(torch.int16) == want.view(torch.int16)
    finite_where_inf = int((want == float("inf")).sum())
    wrapped = int((torch.isinf(want) & torch.isfinite(got)).sum())
    print(
        f"[{dtype_str}] bit-exact {int(same.sum())}/{same.numel()} | "
        f"golden inf slots={finite_where_inf} | wrapped-to-finite on device={wrapped}"
    )
    for j in range(len(row0)):
        print(
            f"    x={float(x[0, j]):>12.6g}  golden={float(want[0, j]):>12.6g}  "
            f"device={float(got[0, j]):>12.6g}"
        )
    assert bool(same.all()), "sat=False must match Tensor.to(dtype) bit for bit"
    assert wrapped == 0, "sat=False must round to inf, not wrap"
    assert finite_where_inf > 0, "the case must actually exercise the overflow"


def main():
    probe("float16", torch.float16, FP16_VALUES, FP16_VALUES[::-1])
    probe("bfloat16", torch.bfloat16, BF16_VALUES, BF16_VALUES[::-1])
    print("[CONST-950-vcvt-sat-false-overflow-inf] OK: sat=False -> IEEE inf (no wrap)")


if __name__ == "__main__":
    main()
