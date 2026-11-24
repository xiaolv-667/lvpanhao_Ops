# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Stage 3 probe #9: stale-proof dense-store sweep (sentinel pre-fill).

Earlier probes disagreed because UB content survives between launches in one
process: a store that leaves slots unwritten still "reads back" correctly when
a previous launch happened to write that data.  This probe pre-fills the `out`
buffer *inside* the kernel with a sentinel, then runs the narrowing store, and
counts how many of the 64 slots per segment still hold the sentinel.

Run (task dir, locked env):
    source ../../../.agents/platforms/toolchains/ascend/env-950pr.sh
    /workspace/venv-ascend950/bin/python history_version/stage3_probe_sentinel.py
"""

import os

os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0")

import tilelang  # noqa: E402
import tilelang.ascend.language as T  # noqa: E402
import torch  # noqa: E402
import torch_npu  # noqa: F401,E402

DQ, SEG, ROWS = 64, 64, 4
SCALE = 0.07216878364870322
SENT = 1234.5


def build(dist, mask_bits, mask_kind, extent):
    @T.prim_func
    def main(
        x: T.Tensor((ROWS, DQ), "float32"),  # type: ignore
        sent: T.Tensor((ROWS, DQ), "float16"),  # type: ignore
        y: T.Tensor((ROWS, DQ), "float16"),  # type: ignore
    ) -> None:
        with T.Kernel(1) as bx:
            acc = T.alloc_shared((ROWS, DQ), "float32")
            out = T.alloc_shared((ROWS, DQ), "float16")
            T.copy(x[:, :], acc[0:ROWS, 0:DQ])
            T.copy(sent[:, :], out[0:ROWS, 0:DQ])  # sentinel fill, same buffer
            with T.SimdVF():
                m = T.simd.pset(mask_bits, mask_kind)
                for i in T.serial(ROWS):
                    v = T.simd.vcvt(
                        T.simd.vmuls(T.simd.vld(acc[i, 0]), SCALE),
                        "float16",
                        part=0,
                        sat=False,
                    )
                    T.simd.vsts(out[i, 0], v, m, dist, extent=extent)
            T.copy(out[0:ROWS, 0:DQ], y[:, :])

    return tilelang.compile(main, target="ascend", out_idx=[2])


def run(tag, dist, mask_bits, mask_kind, extent):
    torch.manual_seed(0)
    x = torch.arange(ROWS * DQ, dtype=torch.float32).reshape(ROWS, DQ) / 7.0
    sent = torch.full((ROWS, DQ), SENT, dtype=torch.float16)
    gold = (x * SCALE).to(torch.float16)
    try:
        k = build(dist, mask_bits, mask_kind, extent)
        got = k(x.to("npu"), sent.to("npu"))
        torch.npu.synchronize()
        got = got.cpu()
    except Exception as exc:  # noqa: BLE001
        print(f"[{tag}] FAIL {type(exc).__name__}: {str(exc)[:150]}", flush=True)
        return
    n_sent = int((got.view(torch.int16) == sent.view(torch.int16)).sum())
    n_ok = int((got.view(torch.int16) == gold.view(torch.int16)).sum())
    print(
        f"[{tag}] correct {n_ok}/{got.numel()}  still-sentinel {n_sent}  "
        f"other {got.numel() - n_ok - n_sent}",
        flush=True,
    )
    src = [ln.strip() for ln in k.get_kernel_source().splitlines() if "vsts" in ln]
    for ln in src[:2]:
        print("     ", ln[:190], flush=True)


def main():
    combos = [
        ("PK_B32 m32/PAT_ALL ext64", "PK_B32", 32, "PAT_ALL", 64),
        ("PK_B32 m32/PAT_ALL extNone", "PK_B32", 32, "PAT_ALL", None),
        ("PK_B32 m16/PAT_VL64 ext64", "PK_B32", 16, "PAT_VL64", 64),
        ("PK_B32 m32/PAT_VL64 ext64", "PK_B32", 32, "PAT_VL64", 64),
        ("PK_B16 m16/PAT_VL64 ext64", "PK_B16", 16, "PAT_VL64", 64),
        ("NORM_B16 m16/PAT_VL64 ext64", "NORM_B16", 16, "PAT_VL64", 64),
        ("NORM_B16 m32/PAT_ALL ext64", "NORM_B16", 32, "PAT_ALL", 64),
    ]
    for tag, dist, bits, kind, extent in combos:
        run(tag, dist, bits, kind, extent)


if __name__ == "__main__":
    main()
