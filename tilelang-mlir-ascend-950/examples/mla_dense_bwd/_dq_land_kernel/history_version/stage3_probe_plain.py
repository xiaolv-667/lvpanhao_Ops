# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Stage 3 probe #2: minimal single-core elementwise multiply + narrow."""

import os

os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0")

import tilelang  # noqa: E402
import tilelang.ascend.language as T  # noqa: E402
import torch  # noqa: E402
import torch_npu  # noqa: F401,E402

DQ, SEG, R = 192, 64, 64
RATIO = 0.07216878364870322


def make_plain():
    @T.prim_func
    def main(
        x: T.Tensor((R, DQ), "float32"),  # type: ignore
        y: T.Tensor((R, DQ), "float16"),  # type: ignore
    ) -> None:
        with T.Kernel(1) as bx:
            acc = T.alloc_shared((R, DQ), "float32")
            out = T.alloc_shared((R, DQ), "float16")
            T.copy(x[:, :], acc[0:R, 0:DQ])
            with T.SimdVF():
                m = T.simd.pset(16, "PAT_VL64")
                for i in T.serial(R):
                    for s in T.serial(3):
                        v = T.simd.vcvt(
                            T.simd.vmuls(T.simd.vld(acc[i, s * SEG]), RATIO),
                            "float16",
                            part=0,
                            sat=False,
                        )
                        T.simd.vsts(out[i, s * SEG], v, m, "NORM_B16", extent=SEG)
            T.copy(out[0:R, 0:DQ], y[:, :])

    return tilelang.compile(main, target="ascend", out_idx=[1])


def main():
    torch.manual_seed(0)
    x = torch.randn((R, DQ)) * 3.0
    k = make_plain()
    got = k(x.to("npu"))
    torch.npu.synchronize()
    got = got.cpu()
    gold = (x * RATIO).to(torch.float16)
    a = got.reshape(-1).view(torch.int16).to(torch.int32) & 0xFFFF
    b = gold.reshape(-1).view(torch.int16).to(torch.int32) & 0xFFFF
    print("bit mismatch", int((a != b).sum()), "/", a.numel())
    print("got  [0,:6]", got.reshape(-1)[:6].tolist())
    print("gold [0,:6]", gold.reshape(-1)[:6].tolist())
    got60 = got.reshape(-1)[60:70].tolist()
    gold60 = gold.reshape(-1)[60:70].tolist()
    print("got  [60:70]", got60)
    print("gold [60:70]", gold60)
    print("abs max diff", float((got.float() - gold.float()).abs().max()))


if __name__ == "__main__":
    main()
