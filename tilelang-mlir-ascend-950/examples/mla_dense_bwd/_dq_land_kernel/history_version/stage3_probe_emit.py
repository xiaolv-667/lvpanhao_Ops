# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Stage 3 probe #8: why does the production form store only 32 of 64 lanes?

Compiles three forms of the same arithmetic and prints each generated SIMD
body plus a ramp check, so the difference between the static 2D probe and the
production kernel is isolated:

  a  static (4, 192)  2D tensor, whole-buffer copies
  b  static (64, 192) 2D tensor, whole-buffer copies
  c  production form  (dynamic total, persistent grid, pipelined tiles)

Run (task dir, locked env):
    source ../../../.agents/platforms/toolchains/ascend/env-950pr.sh
    /workspace/venv-ascend950/bin/python history_version/stage3_probe_emit.py
"""

import os

os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0")

import tilelang  # noqa: E402
import tilelang.ascend.language as T  # noqa: E402
import torch  # noqa: E402
import torch_npu  # noqa: F401,E402

DQ, SEG = 192, 64
SCALE = DQ ** -0.5


def make_static(rows, dist):
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
                m = T.simd.pset(16, "PAT_VL64")
                for i in T.serial(rows):
                    for s in T.serial(3):
                        v = T.simd.vcvt(
                            T.simd.vmuls(T.simd.vld(acc[i, s * SEG]), SCALE),
                            "float16",
                            part=0,
                            sat=False,
                        )
                        T.simd.vsts(out[i, s * SEG], v, m, dist, extent=SEG)
            T.copy(out[0:rows, 0:DQ], y[:, :])

    return tilelang.compile(main, target="ascend", out_idx=[1])


def show_vf(tag, kernel):
    lines = kernel.get_kernel_source().splitlines()
    keep = [ln for ln in lines if "vsts" in ln or "vcvt" in ln or "vlds" in ln]
    print(f"[{tag}] instructions:", flush=True)
    for ln in keep[:6]:
        print("   ", ln.strip()[:220], flush=True)


def ramp(rows):
    d = torch.arange(DQ, dtype=torch.float32)
    x = torch.zeros((rows, DQ), dtype=torch.float32)
    for t in range(rows):
        x[t, :] = t * 4096 + d
    return x


def check(tag, kernel, x):
    got = kernel(x.to("npu"))
    torch.npu.synchronize()
    got = got.cpu()
    gold = (x * SCALE).to(torch.float16)
    nb = int((got.view(torch.int16) != gold.view(torch.int16)).sum())
    print(f"[{tag}] mismatch {nb}/{got.numel()}", flush=True)
    show_vf(tag, kernel)
    if nb:
        g = gold[0].view(torch.int16).tolist()
        idx = {int(v): k for k, v in enumerate(g)}
        print("    row0 mapping:", [idx.get(int(v), None) for v in got[0].view(torch.int16).tolist()], flush=True)


def main():
    for rows in (4, 64):
        k = make_static(rows, "PK_B32")
        check(f"static rows={rows} PK_B32", k, ramp(rows))


if __name__ == "__main__":
    main()
