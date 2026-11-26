"""One operator call of one bwd variant (arm script for ab_test.py).

    python run_once.py --module <variant.py> --workload s96

Loads the module, compiles the factory for the workload geometry, then runs
exactly one operator call (two fused/reduce launches).
"""

import argparse
import importlib.util
import os
import sys

import torch
import torch_npu  # noqa: F401

WORKLOADS = {
    "s32": (1, 32, 256, "float16"),
    "s64": (2, 64, 512, "float16"),
    "s128": (1, 128, 256, "bfloat16"),
    "s96": (3, 96, 768, "bfloat16"),
}
_DT = {"float16": torch.float16, "bfloat16": torch.bfloat16}


def load_module(path):
    name = "arm_variant_" + os.path.basename(path).replace(".py", "").replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True)
    ap.add_argument("--workload", default="s96")
    ap.add_argument("--runs", type=int, default=1)
    args = ap.parse_args()
    torch.npu.set_device(0)
    mod = load_module(os.path.abspath(args.module))
    M, seq_len, d, dt = WORKLOADS[args.workload]
    t = _DT[dt]
    H = torch.randn(M, seq_len, d, dtype=t, device="npu")
    k = torch.randn(M, seq_len, d, dtype=t, device="npu") * 0.1
    v = torch.randn(M, seq_len, d, dtype=t, device="npu") * 0.1
    wh = torch.ones(d, dtype=t, device="npu")
    wv = torch.ones(d, dtype=t, device="npu")
    cw = torch.randn(4, d, dtype=t, device="npu") * 0.02
    dY = torch.randn(M, seq_len, d, dtype=t, device="npu") * 0.1
    h_f = H.float()
    rrms_h = (h_f**2).mean(dim=-1, keepdim=True).add(1e-6).rsqrt()
    h_norm = h_f * rrms_h
    k_f = k.float()
    rrms_k = (k_f**2).mean(dim=-1, keepdim=True).add(1e-6).rsqrt()
    k_norm = k_f * rrms_k
    dot = (h_norm * k_norm).sum(dim=-1, keepdim=True)
    alpha = torch.sigmoid(dot / (d**0.5))
    vh = (alpha * v.float()).to(t)
    rrms_v = (vh.float() ** 2).mean(dim=-1, keepdim=True).add(1e-6).rsqrt()
    args_in = (dY, H, k, v, wh, wv, cw, vh, alpha.squeeze(-1).float(),
               rrms_h.squeeze(-1).float(), rrms_k.squeeze(-1).float(),
               rrms_v.squeeze(-1).float())
    fn = mod._engram_gate_conv_bwd_kernel(M, seq_len, d, dt)
    torch.npu.synchronize()
    for _ in range(args.runs):
        fn(*args_in)
    torch.npu.synchronize()


if __name__ == "__main__":
    main()
