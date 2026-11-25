"""Child process of the capture harness: build one implementation, run it.

Run under ``msopprof`` by ``capture.py``.  Shapes come from the workload table
in ``impls.py``; the inputs are seeded so every candidate sees the same data.
"""

import argparse
import json
import os
import sys

import torch
import torch_npu  # noqa: F401 -- registers torch.npu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from impls import WORKLOADS, build_impl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", default="baseline")
    ap.add_argument("--workload", required=True)
    ap.add_argument("--warm-up", type=int, default=0)
    ap.add_argument("--launch-count", type=int, default=1)
    ap.add_argument("--config", default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = json.loads(args.config) if args.config else {}
    T, H, S, dtype_str = WORKLOADS[args.workload]
    dtype = getattr(torch, dtype_str)
    torch.manual_seed(args.seed)
    hidden = torch.randn(T, H, dtype=dtype).npu()
    w_gate_up = torch.randn(2 * S, H, dtype=dtype).npu()
    w_down = torch.randn(H, S, dtype=dtype).npu()

    fn, info = build_impl(args.impl, args.workload, cfg)
    if os.environ.get("CAPTURE_PRINT_INFO"):
        print("[capture]", json.dumps(info, default=str), flush=True)

    for _ in range(args.warm_up):
        fn(hidden, w_gate_up, w_down)
    torch.npu.synchronize()
    for _ in range(args.launch_count):
        fn(hidden, w_gate_up, w_down)
    torch.npu.synchronize()


if __name__ == "__main__":
    main()
