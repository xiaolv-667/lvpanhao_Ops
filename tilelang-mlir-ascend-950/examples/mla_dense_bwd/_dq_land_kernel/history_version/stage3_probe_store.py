# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Stage 3 probe #3: store distribution + saturation mode for f32 -> 16-bit.

Probe #2 showed `vsts(..., "NORM_B16")` of a `vcvt(f32->f16, part=0)` result
writes every other 16-bit slot (values land in the *even* lanes of the b16
register).  This probe compares the store distributions and the saturation
flags against torch's `.to(dtype)` on the corner set of `verify_equiv.py`.

Run (task dir, locked env):
    source ../../../.agents/platforms/toolchains/ascend/env-950pr.sh
    /workspace/venv-ascend950/bin/python history_version/stage3_probe_store.py
"""

import os

os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0")

import tilelang  # noqa: E402
import tilelang.ascend.language as T  # noqa: E402
import torch  # noqa: E402
import torch_npu  # noqa: F401,E402

DQ, SEG, R = 192, 64, 64
RATIO = 0.07216878364870322
MAX_FINITE = {"float16": 65504.0, "bfloat16": 3.3895313892515355e38}
TORCH = {"float16": torch.float16, "bfloat16": torch.bfloat16}


def make(dtype, dist, sat, extent):
    @T.prim_func
    def main(
        x: T.Tensor((R, DQ), "float32"),  # type: ignore
        y: T.Tensor((R, DQ), dtype),  # type: ignore
    ) -> None:
        with T.Kernel(1) as bx:
            acc = T.alloc_shared((R, DQ), "float32")
            out = T.alloc_shared((R, DQ), dtype)
            T.copy(x[:, :], acc[0:R, 0:DQ])
            with T.SimdVF():
                m = T.simd.pset(16, "PAT_VL64")
                for i in T.serial(R):
                    for s in T.serial(3):
                        v = T.simd.vcvt(
                            T.simd.vmuls(T.simd.vld(acc[i, s * SEG]), RATIO),
                            dtype,
                            part=0,
                            sat=sat,
                        )
                        T.simd.vsts(out[i, s * SEG], v, m, dist, extent=extent)
            T.copy(out[0:R, 0:DQ], y[:, :])

    return tilelang.compile(main, target="ascend", out_idx=[1])


def corner_tensor():
    s = RATIO
    vals = []
    for name in ("float16", "bfloat16"):
        mx = MAX_FINITE[name]
        vals += [
            0.0, -0.0, 1.0, -1.0,
            float("inf"), float("-inf"), float("nan"),
            1e-8, -1e-8, 1e-38, -1e-38,
            mx / s, -mx / s,
            mx * 0.9999 / s, -mx * 0.9999 / s,
            mx * 1.0001 / s, -mx * 1.0001 / s,
        ]
    return torch.tensor(vals, dtype=torch.float32), vals


def run(dtype, dist, sat, extent, x, vals):
    kernel = make(dtype, dist, sat, extent)
    got = kernel(x.to("npu"))
    torch.npu.synchronize()
    got = got.cpu()
    gold = (x * RATIO).to(TORCH[dtype])
    a = got.reshape(-1).view(torch.int16).to(torch.int32) & 0xFFFF
    b = gold.reshape(-1).view(torch.int16).to(torch.int32) & 0xFFFF
    nbad = int((a != b).sum())
    print(f"[{dtype} dist={dist} extent={extent} sat={sat}] bit-mismatch {nbad}/{a.numel()}", flush=True)
    if nbad <= 64:
        flat_got = got.reshape(-1)[: len(vals)].tolist()
        flat_gold = gold.reshape(-1)[: len(vals)].tolist()
        for v, g, d in zip(vals, flat_got, flat_gold):
            ok = (g == d) or (g != g and d != d)
            if not ok:
                print(f"    corner {v!r:>24} dev {g!r:>12} golden {d!r:>12}", flush=True)
    else:
        print("    got [0:8] ", got.reshape(-1)[:8].tolist(), flush=True)
        print("    gold[0:8] ", gold.reshape(-1)[:8].tolist(), flush=True)
    return nbad


def main():
    x = torch.randn((R, DQ), generator=torch.Generator().manual_seed(0)) * 3.0
    corners, vals = corner_tensor()
    x.reshape(-1)[: len(corners)] = corners
    for dtype in ("float16", "bfloat16"):
        for dist, extent in (("NORM_B16", 64), ("PK_B16", 64), ("PK_B16", None)):
            for sat in (False, True):
                run(dtype, dist, sat, extent, x, vals)


if __name__ == "__main__":
    main()
