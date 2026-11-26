"""Screening probe: wall-clock sweep of GEMM tiles (not a reported metric).

Compilation dominates a sweep, so this runs every config in one process and
times each with ``perf_counter`` around N launches after a warm-up launch.  The
numbers are a *screening* signal only; every config that matters is re-measured
through ``capture.py`` (msopprof Task Duration) before it is recorded in
``perf_records.jsonl``.

Usage::

    python probe.py --m 512 --n 36864 --k 7168 --dtype bfloat16 \
        --configs '[{"block_m":256,"block_n":256,"block_k":64,"num_stages":4}]'
"""

import argparse
import json
import os
import statistics
import sys
import time

import torch
import torch_npu  # noqa: F401 -- registers torch.npu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import importlib.util  # noqa: E402


def load_module(path):
    name = "probe_cand"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", required=True)
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--configs", required=True, help="JSON list of config dicts")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--what", default="gemm", choices=["gemm", "op"])
    ap.add_argument("--T", type=int, default=0)
    ap.add_argument("--H", type=int, default=0)
    ap.add_argument("--S", type=int, default=0)
    args = ap.parse_args()

    module = load_module(args.impl)
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(0)
    configs = json.loads(args.configs)
    results = []
    for cfg in configs:
        try:
            if args.what == "gemm":
                fn = module._gemm_call(args.m, args.n, args.k, dict(
                    block_m=cfg["block_m"], block_n=cfg["block_n"],
                    block_k=cfg["block_k"], num_stages=cfg["num_stages"],
                    split_k=cfg.get("split_k"),
                ), args.dtype)
                a = torch.randn(args.m, args.k, dtype=dtype).npu()
                b = torch.randn(args.n, args.k, dtype=dtype).npu()
                args3 = (a, b)
            else:
                kernel = module.SharedExpertMLPKernel(
                    args.T, args.H, args.S, dtype, cfg)
                a = torch.randn(args.T, args.H, dtype=dtype).npu()
                b = torch.randn(2 * args.S, args.H, dtype=dtype).npu()
                c = torch.randn(args.H, args.S, dtype=dtype).npu()
                args3 = (a, b, c)
                fn = kernel
            fn(*args3)
            torch.npu.synchronize()
            samples = []
            for _ in range(args.reps):
                t0 = time.perf_counter()
                out = fn(*args3)
                torch.npu.synchronize()
                samples.append((time.perf_counter() - t0) * 1e6)
            tflops = 2.0 * (args.m * args.n * args.k if args.what == "gemm"
                            else 2 * args.T * args.S * args.H + 6 * args.T * args.S)
            med = statistics.median(samples)
            results.append((cfg, med, tflops / med / 1e6))
        except Exception as exc:  # noqa: BLE001 - a rejected tile is data
            print(f"  {cfg} -> REJECTED {type(exc).__name__}: {str(exc)[:160]}")
            results.append((cfg, float("inf"), 0.0))
    for cfg, med, tf in sorted(results, key=lambda item: item[1]):
        print(f"  {cfg} -> {med:9.2f} us  {tf:7.1f} TF/s")
    _ = out


if __name__ == "__main__":
    main()
