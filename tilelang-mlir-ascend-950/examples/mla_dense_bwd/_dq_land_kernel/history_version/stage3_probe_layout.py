# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Stage 3 probe #6: where do the real kernel's outputs differ?

Runs the production kernel form (dynamic total, persistent grid, pipelined
tile loop) on a ramp input and reports the mismatch structure per row / head /
column, for the 2-version pipeline and for the no-version control.

Run (task dir, locked env):
    source ../../../.agents/platforms/toolchains/ascend/env-950pr.sh
    /workspace/venv-ascend950/bin/python history_version/stage3_probe_layout.py
"""

import os

os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0")

import _dq_land_kernel as m  # noqa: E402
import torch  # noqa: E402
import torch_npu  # noqa: F401,E402

TQ, H, DQ = 512, 8, 192
SCALE = DQ ** -0.5


def analyse(tag, fn, x):
    got = fn(x.to("npu"))
    torch.npu.synchronize()
    got = got.cpu()
    gold = (x * SCALE).to(torch.float16)
    bad = ~((got.view(torch.int16) == gold.view(torch.int16)) | (got.isnan() & gold.isnan()))
    n = int(bad.sum())
    print(f"[{tag}] mismatch {n}/{bad.numel()}", flush=True)
    if not n:
        return
    per_row = bad.any(-1).any(-1).nonzero().reshape(-1).tolist()
    per_head = [int(bad[:, h, :].sum()) for h in range(H)]
    per_col = bad.any(0).any(0).nonzero().reshape(-1).tolist()
    print(f"    bad rows  ({len(per_row)}): {per_row[:20]}{' ...' if len(per_row) > 20 else ''}", flush=True)
    print(f"    bad heads: {per_head}", flush=True)
    print(f"    bad cols  ({len(per_col)}): {per_col[:20]}{' ...' if len(per_col) > 20 else ''}", flush=True)
    t, h, d = [int(v) for v in bad.nonzero()[0]]
    print(f"    first bad (t={t},h={h},d={d}) dev={got[t, h, d].item()!r} gold={gold[t, h, d].item()!r} x={x[t, h, d].item()!r}", flush=True)
    # value mapping: which source (t,d) does the device value correspond to?
    flat_gold = gold.view(torch.int16).reshape(-1).to(torch.int32)
    idx = {}
    for k, v in enumerate(flat_gold.tolist()):
        idx.setdefault(v, []).append(k)
    hits = idx.get(int(got.view(torch.int16)[t, h, d]) & 0xFFFF, [])
    srcs = [((k // (H * DQ)), (k // DQ) % H, k % DQ) for k in hits[:4]]
    print(f"    device value equals golden at {srcs}", flush=True)


def main():
    x = torch.randn((TQ, H, DQ), generator=torch.Generator().manual_seed(1)) * 3.0
    analyse("versions=2", m._dq_land_kernel(H, DQ, SCALE, "float16"), x)
    analyse("versions=1", m._dq_land_single_version(H, DQ, SCALE, "float16"), x)
    src = m._dq_land_kernel(H, DQ, SCALE, "float16").get_kernel_source()
    print("---- generated source (head) ----", flush=True)
    print("\n".join(src.splitlines()[:40]), flush=True)


if __name__ == "__main__":
    main()
