# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Stage 3 probe #4: exact lane mapping of vcvt(f32->f16) + vsts.

Input is an index ramp so every stored 16-bit word can be traced back to its
source lane: `x[i, d] = i * 1000 + d`, `gold = (x * RATIO).to(float16)`.
Prints, for each output slot, the source lane it carries (or `--` when the
slot holds something that is not a golden value).

Run (task dir, locked env):
    source ../../../.agents/platforms/toolchains/ascend/env-950pr.sh
    /workspace/venv-ascend950/bin/python history_version/stage3_probe_map.py
"""

import os

os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0")

import tilelang  # noqa: E402
import tilelang.ascend.language as T  # noqa: E402
import torch  # noqa: E402
import torch_npu  # noqa: F401,E402

DQ, SEG = 192, 64
RATIO = 0.07216878364870322


def make(rows, dist, mask_kind):
    @T.prim_func
    def main(
        x: T.Tensor((rows, DQ), "float32"),  # type: ignore
        y: T.Tensor((rows, DQ), "float16"),  # type: ignore
    ) -> None:
        with T.Kernel(1) as bx:
            acc = T.alloc_shared((rows, DQ), "float32")
            out = T.alloc_shared((rows, DQ), "float16")
            T.copy(x[:, :], acc[0:rows, 0:DQ])
            with T.SimdVF():
                m = T.simd.pset(16, mask_kind)
                for i in T.serial(rows):
                    for s in T.serial(3):
                        v = T.simd.vcvt(
                            T.simd.vmuls(T.simd.vld(acc[i, s * SEG]), RATIO),
                            "float16",
                            part=0,
                            sat=False,
                        )
                        T.simd.vsts(out[i, s * SEG], v, m, dist, extent=SEG)
            T.copy(out[0:rows, 0:DQ], y[:, :])

    return tilelang.compile(main, target="ascend", out_idx=[1])


def analyse(got, gold, rows):
    g = got.reshape(-1).view(torch.int16).to(torch.int32) & 0xFFFF
    d = gold.reshape(-1).view(torch.int16).to(torch.int32) & 0xFFFF
    index = {}
    for k, v in enumerate(d.tolist()):
        index.setdefault(v, k)
    mapping = [index.get(v, None) for v in g.tolist()]
    nbad = int((g != d).sum())
    print(f"    bit mismatch {nbad}/{g.numel()}; mapping[0:64] =")
    print("   ", mapping[0:64])
    print(f"    mapping[64:128] =")
    print("   ", mapping[64:128])
    print(f"    mapping[128:192] =")
    print("   ", mapping[128:192])
    print(f"    mapping[192:256] (row1) =")
    print("   ", mapping[192:256])


def main():
    rows = 2
    x = torch.arange(rows * DQ, dtype=torch.float32).reshape(rows, DQ) * 1.0
    gold = (x * RATIO).to(torch.float16)
    for dist in ("NORM_B16", "PK_B16"):
        for mask_kind in ("PAT_VL64", "PAT_ALL", "PAT_VL128"):
            try:
                k = make(rows, dist, mask_kind)
                got = k(x.to("npu"))
                torch.npu.synchronize()
            except Exception as exc:  # noqa: BLE001
                print(f"[{dist}/{mask_kind}] FAILED {type(exc).__name__}: {str(exc)[:160]}", flush=True)
                continue
            print(f"[{dist}/{mask_kind}]", flush=True)
            analyse(got.cpu(), gold, rows)


if __name__ == "__main__":
    main()
