# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Stage 3 probe #5: store distribution for a narrowing vcvt, exhaustive search.

`vcvt(f32x64 -> 16-bit, part=0)` places the 64 converted values in the *even*
16-bit lanes of the destination register (probe #4).  A dense store therefore
has to pack 32-bit words down to their low half; `dist="PK_B32"`
(`asc_storealign_pack`) is the documented distribution for that.

This probe compares every plausible (dist, mask) pair on an index ramp against
torch's own `.to(dtype)`, for both 16-bit targets.

Run (task dir, locked env):
    source ../../../.agents/platforms/toolchains/ascend/env-950pr.sh
    /workspace/venv-ascend950/bin/python history_version/stage3_probe_store2.py
"""

import os

os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0")

import tilelang  # noqa: E402
import tilelang.ascend.language as T  # noqa: E402
import torch  # noqa: E402
import torch_npu  # noqa: F401,E402

DQ, SEG = 192, 64
RATIO = 0.07216878364870322
TORCH = {"float16": torch.float16, "bfloat16": torch.bfloat16}


def make(rows, dtype, dist, mask_bits, mask_kind, extent):
    @T.prim_func
    def main(
        x: T.Tensor((rows, DQ), "float32"),  # type: ignore
        y: T.Tensor((rows, DQ), dtype),  # type: ignore
    ) -> None:
        with T.Kernel(1) as bx:
            acc = T.alloc_shared((rows, DQ), "float32")
            out = T.alloc_shared((rows, DQ), dtype)
            T.copy(x[:, :], acc[0:rows, 0:DQ])
            with T.SimdVF():
                m = T.simd.pset(mask_bits, mask_kind)
                for i in T.serial(rows):
                    for s in T.serial(3):
                        v = T.simd.vcvt(
                            T.simd.vmuls(T.simd.vld(acc[i, s * SEG]), RATIO),
                            dtype,
                            part=0,
                            sat=False,
                        )
                        T.simd.vsts(out[i, s * SEG], v, m, dist, extent=extent)
            T.copy(out[0:rows, 0:DQ], y[:, :])

    return tilelang.compile(main, target="ascend", out_idx=[1])


def corners():
    s = RATIO
    vals = []
    for name in ("float16", "bfloat16"):
        mx = 65504.0 if name == "float16" else 3.3895313892515355e38
        vals += [
            0.0, -0.0, 1.0, -1.0, float("inf"), float("-inf"), float("nan"),
            1e-8, -1e-8, 1e-38, -1e-38,
            mx / s, -mx / s, mx * 0.9999 / s, -mx * 0.9999 / s,
            mx * 1.0001 / s, -mx * 1.0001 / s,
        ]
    return torch.tensor(vals, dtype=torch.float32), vals


def main():
    rows = 4
    x = torch.arange(rows * DQ, dtype=torch.float32).reshape(rows, DQ)
    corners_t, vals = corners()
    x.reshape(-1)[: len(corners_t)] = corners_t
    gold_by_dtype = {d: (x * RATIO).to(TORCH[d]) for d in TORCH}
    combos = [
        ("PK_B32", 32, "PAT_ALL", 64),
        ("PK_B32", 16, "PAT_VL64", 64),
        ("PK_B32", 32, "PAT_ALL", None),
        ("NORM_B16", 16, "PAT_ALL", 64),
        ("NORM_B16", 32, "PAT_ALL", 64),
        ("PK_B16", 32, "PAT_ALL", 64),
    ]
    for dtype in ("float16", "bfloat16"):
        gold = gold_by_dtype[dtype]
        g = gold.reshape(-1).view(torch.int16).to(torch.int32) & 0xFFFF
        for dist, bits, kind, extent in combos:
            try:
                k = make(rows, dtype, dist, bits, kind, extent)
                got = k(x.to("npu"))
                torch.npu.synchronize()
            except Exception as exc:  # noqa: BLE001
                print(f"[{dtype} {dist} m{bits}/{kind} ext={extent}] FAIL {type(exc).__name__}: {str(exc)[:120]}", flush=True)
                continue
            got = got.cpu().reshape(-1)
            a = got.view(torch.int16).to(torch.int32) & 0xFFFF
            nbad = int((a != g).sum())
            print(f"[{dtype} {dist} m{bits}/{kind} ext={extent}] bit-mismatch {nbad}/{a.numel()}", flush=True)
            if nbad:
                idx = (a != g).nonzero().reshape(-1)[:6].tolist()
                for j in idx:
                    print(f"      slot {j}: dev {got[j].item()!r} gold {gold.reshape(-1)[j].item()!r}", flush=True)
            else:
                cf = [(v, got[j].item(), gold.reshape(-1)[j].item()) for j, v in enumerate(vals)]
                print("      all corners bit-exact:", all(
                    (gv == dv) or (gv != gv and dv != dv) for _, gv, dv in cf), flush=True)


if __name__ == "__main__":
    main()
