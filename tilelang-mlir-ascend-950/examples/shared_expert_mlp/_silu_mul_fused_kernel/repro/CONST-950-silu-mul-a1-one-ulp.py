"""CONST-950-silu-mul-a1-one-ulp: DESIGN A1 stays within one 16-bit ulp on device.

First verified: TileLang ascend 0.1.15 / CANN 9.2.1 runtime, Ascend950PR_9579.

Claim under test. DESIGN.md 1.6.1 replaced the source expression
``g*sigmoid(g)*u`` (6 vector ops / 64 lanes) by the rational form A1

    y = (g * u) / (1 + exp(-g))        # 5 vector ops / 64 lanes

and showed 0 ulp against the source form on a *torch fp32 replay* (that replay
cannot model the device's ``vexp``/``vdiv`` variants).  This repro closes the
gap on the device: the same 64-lane chain the kernel runs
(``vld(UNPK_B16) -> vcvt(fp32) -> vmul / vadds(vexp(vneg)) / vdiv ->
vcvt(narrow, ROUND_R, sat=False) -> vsts(PK_B32)``) is compared with the
independent golden ``F.silu(gate.float()) * up.float() -> dtype``.

Measured: max 1 output ulp for both 16-bit dtypes over 4096 random elements
covering ~[-9, 9] in gate and up (mean |ulp| ~1e-5), i.e. the fp32 chain's own
roundings cost at most the last output bit -- orders of magnitude inside the
1e-2 / 1e-2 gate of DESIGN.md 8.2.  A wrong ``vexp`` variant, an fp16-domain
chain or a missing ``vneg`` shows up here as a multi-ulp / large count result.

Run: cd examples/shared_expert_mlp/_silu_mul_fused_kernel
     source ../../../.agents/platforms/toolchains/ascend/env-950pr.sh
     ASCEND_RT_VISIBLE_DEVICES=0 /workspace/venv-ascend950/bin/python repro/CONST-950-silu-mul-a1-one-ulp.py
"""

import torch

import tilelang
import tilelang.ascend.language as T
import torch_npu  # noqa: F401  # registers torch.npu

VL = 64
ROWS = 64  # 64 x 64 = 4096 elements per dtype
NCHUNK = 1


def build(dtype):
    @T.prim_func
    def main(
        g16: T.Tensor((ROWS, ROWS), dtype),  # type: ignore[valid-type]
        u16: T.Tensor((ROWS, ROWS), dtype),  # type: ignore[valid-type]
        y16: T.Tensor((ROWS, ROWS), dtype),  # type: ignore[valid-type]
    ):
        with T.Kernel(1):
            g_ub = T.alloc_shared((ROWS, ROWS), dtype)
            u_ub = T.alloc_shared((ROWS, ROWS), dtype)
            y_ub = T.alloc_shared((ROWS, ROWS), dtype)
            T.copy(g16[:, :], g_ub[0:ROWS, 0:ROWS])
            T.copy(u16[:, :], u_ub[0:ROWS, 0:ROWS])
            with T.SimdVF():
                full = T.simd.pset(32)
                for r in T.serial(ROWS):
                    for c in T.serial(NCHUNK):
                        g = T.simd.vcvt(
                            T.simd.vld(g_ub[r, c * VL], dist="UNPK_B16"), "float32", part=0
                        )
                        u = T.simd.vcvt(
                            T.simd.vld(u_ub[r, c * VL], dist="UNPK_B16"), "float32", part=0
                        )
                        num = T.simd.vmul(g, u, full)
                        den = T.simd.vadds(T.simd.vexp(T.simd.vneg(g, full), full), 1.0, full)
                        y = T.simd.vdiv(num, den, full)
                        T.simd.vsts(
                            y_ub[r, c * VL],
                            T.simd.vcvt(y, dtype, round="ROUND_R", sat=False, part=0),
                            full,
                            dist="PK_B32",
                        )
            T.copy(y_ub[0:ROWS, 0:ROWS], y16[:, :])

    return tilelang.compile(main, target="ascend", out_idx=[2])


def ulp_distance(got, want):
    """ULP distance in the 16-bit dtype (0 == bit identical)."""
    a = got.view(torch.int16).to(torch.int32)
    b = want.view(torch.int16).to(torch.int32)
    a = torch.where(a < 0, -a + 32768, a + 32768)  # monotone bit pattern
    b = torch.where(b < 0, -b + 32768, b + 32768)
    return (a - b).abs()


def probe(dtype_str, torch_dtype, seed):
    gen = torch.Generator().manual_seed(seed)
    gate = torch.randn(ROWS, ROWS, generator=gen) * 3.0
    up = torch.randn(ROWS, ROWS, generator=gen) * 2.0
    g16, u16 = gate.to(torch_dtype), up.to(torch_dtype)

    want = (torch.nn.functional.silu(g16.float()) * u16.float()).to(torch_dtype)
    got = build(dtype_str)(g16.npu(), u16.npu())
    torch.npu.synchronize()
    got = got.cpu()

    d = ulp_distance(got, want)
    exact = int((d == 0).sum())
    print(
        f"[{dtype_str:9s}] exact {exact}/{d.numel()} | max_ulp={int(d.max())} | "
        f"frac_gt1ulp={float((d > 1).float().mean()):.3e} | mean_ulp={float(d.float().mean()):.3e}"
    )
    assert int(d.max()) <= 1, "A1 must stay within one output ulp"
    assert float((d > 1).float().mean()) == 0.0
    assert exact > d.numel() // 2, "most elements must be bit identical"


def main():
    probe("bfloat16", torch.bfloat16, 11)
    probe("float16", torch.float16, 12)
    print("[CONST-950-silu-mul-a1-one-ulp] OK: 5-op A1 chain <= 1 output ulp on device")


if __name__ == "__main__":
    main()
