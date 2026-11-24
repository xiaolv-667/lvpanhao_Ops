# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Stage 3 probe: device semantics of `T.simd.vcvt` narrowing with sat=False.

Question (DESIGN.md 9.1 R1): does `vcvt(f32 -> f16/bf16, round="ROUND_R",
sat=False)` reproduce torch's `.to(dtype)` bit-for-bit, i.e. single RNE and
IEEE overflow (65520 -> +-inf), or does the "no sat = wrap" wording of the
installed docstring mean something else?

Form: the frozen kernel shape of DESIGN.md 3.3 (one-dimensional grid, UB
acc/out, `T.SimdVF` with `T.serial` row/segment loops written inside the
prim_func body), on a static geometry so that only the conversion semantics
vary.  Input = the corner set of `verify_equiv.py` (pre-divided by scale, so
the products sit exactly on the fp16/bf16 boundaries) plus random values.

Run (task dir, locked env):
    source ../../../.agents/platforms/toolchains/ascend/env-950pr.sh
    /workspace/venv-ascend950/bin/python history_version/stage3_probe_vcvt.py
"""

import os

os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0")

import tilelang  # noqa: E402
import tilelang.ascend.language as T  # noqa: E402
import torch  # noqa: E402
import torch_npu  # noqa: F401,E402

DQ, SEG, R, H, TQ = 192, 64, 64, 2, 64
RATIO = 0.07216878364870322  # 192 ** -0.5
MAX_FINITE = {"float16": 65504.0, "bfloat16": 3.3895313892515355e38}


def make(sat):
    @T.prim_func
    def main(
        dq_acc: T.Tensor((TQ, H, DQ), "float32"),  # type: ignore
        dq: T.Tensor((TQ, H, DQ), "float16"),  # type: ignore
    ) -> None:
        with T.Kernel(2) as bx:
            acc = T.alloc_shared((R, DQ), "float32")
            out = T.alloc_shared((R, DQ), "float16")
            if bx == 0:
                for it in T.serial(1):
                    T.copy(dq_acc[0:R, 0, :], acc[0:R, 0:DQ])
                    with T.SimdVF():
                        m = T.simd.pset(16, "PAT_VL64")
                        for i in T.serial(R):
                            for s in T.serial(3):
                                v = T.simd.vcvt(
                                    T.simd.vmuls(T.simd.vld(acc[i, s * SEG]), RATIO),
                                    "float16",
                                    part=0,
                                    sat=sat,
                                )
                                T.simd.vsts(out[i, s * SEG], v, m, "NORM_B16", extent=SEG)
                    T.copy(out[0:R, 0:DQ], dq[0:R, 0, :])
            else:
                for it in T.serial(1):
                    T.copy(dq_acc[0:R, 1, :], acc[0:R, 0:DQ])
                    with T.SimdVF():
                        m = T.simd.pset(16, "PAT_VL64")
                        for i in T.serial(R):
                            for s in T.serial(3):
                                v = T.simd.vcvt(
                                    T.simd.vmuls(T.simd.vld(acc[i, s * SEG]), RATIO),
                                    "float16",
                                    part=0,
                                    sat=sat,
                                )
                                T.simd.vsts(out[i, s * SEG], v, m, "NORM_B16", extent=SEG)
                    T.copy(out[0:R, 0:DQ], dq[0:R, 1, :])

    return tilelang.compile(main, target="ascend", out_idx=[1])


def corner_values():
    mx = MAX_FINITE["float16"]
    s = RATIO
    vals = [
        0.0, -0.0, 1.0, -1.0, float("inf"), float("-inf"), float("nan"),
        1e-8, -1e-8, 1e-38, -1e-38,
        mx / s, -mx / s,
        65519.0 / s, -65519.0 / s,
        65520.0 / s, -65520.0 / s,
        7e4 / s, -7e4 / s, 1.0e5 / s, -1.0e5 / s, 3.0e38, -3.0e38,
    ]
    return torch.tensor(vals, dtype=torch.float32), vals


def bits_ok(got, gold):
    a = got.reshape(-1).view(torch.int16).to(torch.int32)
    b = gold.reshape(-1).view(torch.int16).to(torch.int32)
    return (a & 0xFFFF) == (b & 0xFFFF)


def report(tag, kernel, x, corners):
    got = kernel(x.to("npu"))
    torch.npu.synchronize()
    got = got.cpu()
    gold = (x.float() * RATIO).to(torch.float16)
    same = bits_ok(got, gold)
    nbad = int((~same).sum())
    print(f"[{tag}] bit-mismatch {nbad}/{same.numel()}", flush=True)
    flat = x.reshape(-1)[: len(corners)]
    gflat = got.reshape(-1)[: len(corners)]
    dflat = gold.reshape(-1)[: len(corners)]
    for v, g, d in zip(corners, gflat.tolist(), dflat.tolist()):
        mark = "" if (float("nan") == v) and (g != g) else ""
        ok = (g == d) or (g != g and d != d)
        if not ok:
            print(f"    corner {v!r:>22} -> dev {g!r:>10} golden {d!r:>10}  MISMATCH {mark}", flush=True)
    return nbad


def main():
    x = torch.randn((TQ, H, DQ), generator=torch.Generator().manual_seed(0)) * 3.0
    corners, vals = corner_values()
    flat = x.reshape(-1)
    flat[: len(corners)] = corners
    for sat in (False, True):
        kernel = make(sat)
        src = kernel.get_kernel_source()
        hits = [k for k in ("vlds_norm", "vmuls", "vcvt", "vsts_norm") if k in src]
        print(f"[sat={sat}] source hits {hits}", flush=True)
        report(f"sat={sat}", kernel, x, vals)


if __name__ == "__main__":
    main()
