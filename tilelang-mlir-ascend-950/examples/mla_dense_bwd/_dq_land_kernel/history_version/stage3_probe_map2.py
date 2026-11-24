# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Stage 3 probe #7: exact slot mapping of the production kernel (ramp input).

Ramp input ``x[t, h, d] = t * 4096 + d`` makes every golden element unique, so
each output slot can be traced back to the (t, d) it carries.

Run (task dir, locked env):
    source ../../../.agents/platforms/toolchains/ascend/env-950pr.sh
    /workspace/venv-ascend950/bin/python history_version/stage3_probe_map2.py
"""

import os

os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0")

import _dq_land_kernel as m  # noqa: E402
import torch  # noqa: E402
import torch_npu  # noqa: F401,E402

TQ, H, DQ = 128, 2, 192
SCALE = DQ ** -0.5


def mapping_for(tag, fn, x):
    got = fn(x.to("npu"))
    torch.npu.synchronize()
    got = got.cpu()
    gold = (x * SCALE).to(torch.float16)
    g = gold[0, 0, :].view(torch.int16).tolist()
    d = got[0, 0, :].view(torch.int16).tolist()
    idx = {int(v): k for k, v in enumerate(g)}
    print(f"[{tag}] row0 head0 slot -> golden column", flush=True)
    print("   ", [idx.get(int(v), None) for v in d], flush=True)
    print("    raw dev  ", [got[0, 0, k].item() for k in range(0, 8)], flush=True)
    print("    raw gold ", [gold[0, 0, k].item() for k in range(0, 8)], flush=True)
    nb = sum(1 for v in d if int(v) not in idx)
    print(f"    slots holding a foreign value: {nb}/192", flush=True)


def main():
    d = torch.arange(DQ, dtype=torch.float32)
    x = torch.zeros((TQ, H, DQ), dtype=torch.float32)
    for t in range(TQ):
        x[t, 0, :] = t * 4096 + d
        x[t, 1, :] = t * 4096 + d + 2048
    mapping_for("real kernel", m._dq_land_kernel(H, DQ, SCALE, "float16"), x)


if __name__ == "__main__":
    main()
