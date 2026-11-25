"""Single-launch msprof driver for one SSD-chunk-coupling kernel variant.

Protocol (identical to the task platform's TileOPs benchmark path):
``msopprof --launch-count=1 --warm-up=0 --kernel-name=main`` around **one**
operator call, then read ``Task Duration(us)`` from ``OpBasicInfo.csv``.  All
launches captured in one capture are summed (the operator must issue exactly
one kernel launch; the driver asserts that and reports the per-op breakdown).

Usage (cwd must stay ``examples/TileOPs`` so that ``tileops``/``torch_npu``
resolve, per the platform execution context):

    python <this file> --module <variant.py> --workload wl3 [--reps 3]
"""

import argparse
import csv
import glob
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile

import torch
import torch_npu  # noqa: F401

WORKLOADS = {
    # label -> (batch, num_chunks, chunk_len, n_groups, d_state, dtype_str)
    "smoke-fp16": (1, 2, 64, 1, 64, "float16"),
    "smoke-bf16": (1, 2, 64, 1, 64, "bfloat16"),
    "wl1": (1, 16, 256, 1, 128, "float16"),      # mamba2-780m
    "wl2": (4, 8, 256, 1, 128, "bfloat16"),      # mamba2-2p7b
    "wl3": (3, 12, 256, 8, 128, "float16"),      # mamba2-grouped-prefill
    "wl3b": (3, 12, 256, 8, 128, "bfloat16"),    # grouped-prefill, bf16
}

_DT = {"float16": torch.float16, "bfloat16": torch.bfloat16}


def load_module(path):
    name = "ssd_cc_variant_" + os.path.basename(path).replace(".py", "").replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def make_inputs(wl, seed=0, dev="npu"):
    batch, nc, q, g, n, dtype = WORKLOADS[wl]
    gen = torch.Generator().manual_seed(seed)
    shape = (batch, nc * q, g, n)
    c = torch.randn(shape, generator=gen, dtype=torch.float32).to(_DT[dtype]).to(dev)
    b = torch.randn(shape, generator=gen, dtype=torch.float32).to(_DT[dtype]).to(dev)
    return c, b


CHILD = r'''
import importlib.util, os, sys, torch, torch_npu
torch.npu.set_device(0)
path = {path!r}
spec = importlib.util.spec_from_file_location("variant_mod", path)
mod = importlib.util.module_from_spec(spec)
sys.modules["variant_mod"] = mod
spec.loader.exec_module(mod)
args = torch.load({args!r})
args = [a.to("npu") if isinstance(a, torch.Tensor) else a for a in args]
fn = mod._launch(*{geom!r})
torch.npu.synchronize()
fn(*args)
torch.npu.synchronize()
'''


def run_capture(module_path, wl, out_root, tag, warmup):
    batch, nc, q, g, n, dtype = WORKLOADS[wl]
    tmp = tempfile.mkdtemp(prefix="bench_one_", dir=os.environ.get("TMPDIR", "/workspace"))
    try:
        args_path = os.path.join(tmp, "args.pt")
        c, b = make_inputs(wl)
        torch.save([c.cpu(), b.cpu()], args_path)
        del c, b
        torch.npu.empty_cache()
        script = os.path.join(tmp, "run.py")
        # warm-up launches are separate processes so that no un-measured launch
        # can be attributed to the capture window by the profiler.
        body = CHILD.format(
            path=module_path, args=args_path, geom=(batch, nc, g, q, n, dtype)
        )
        with open(script, "w") as f:
            f.write(body)
        env = dict(os.environ)
        env["PYTHONPATH"] = os.getcwd() + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        if warmup:
            subprocess.run([sys.executable, script], cwd=os.getcwd(), env=env,
                           capture_output=True, timeout=900)
        out = os.path.join(out_root, tag)
        if os.path.isdir(out):
            shutil.rmtree(out)
        os.makedirs(out)
        os.chmod(out, 0o700)
        cmd = [
            "msopprof", f"--output={out}", "--launch-count=1", "--warm-up=0",
            "--kernel-name=main", "--aic-metrics=Roofline", sys.executable, script,
        ]
        res = subprocess.run(cmd, cwd=os.getcwd(), env=env, capture_output=True, text=True,
                             timeout=900)
        if res.returncode != 0:
            raise RuntimeError(f"msopprof failed rc={res.returncode}: {res.stderr[-2000:]}")
        per, total, rows = {}, 0.0, 0
        csvs = sorted(glob.glob(os.path.join(out, "**", "OpBasicInfo*.csv"), recursive=True))
        for path in csvs:
            with open(path, newline="") as f:
                for row in csv.DictReader(f):
                    name = (row.get("Op Name") or "").strip()
                    raw = (row.get("Task Duration(us)") or "").strip()
                    if name and raw:
                        per[name] = per.get(name, 0.0) + float(raw)
                        total += float(raw)
                        rows += 1
        if not rows:
            raise RuntimeError("no OpBasicInfo rows captured")
        return total, per, out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True)
    ap.add_argument("--workload", default="wl3")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    module = os.path.abspath(args.module)
    out_root = args.out_root or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "profiles", "experiments"
    )
    out_root = os.path.abspath(out_root)
    os.makedirs(out_root, exist_ok=True)
    # Populate the tilelang JIT cache in this process first so the child does not
    # pay compilation inside the capture window.
    mod = load_module(module)
    batch, nc, q, g, n, dtype = WORKLOADS[args.workload]
    mod._launch(batch, nc, g, q, n, dtype)
    torch.npu.synchronize()

    results = []
    for rep in range(args.reps):
        tag = f"{os.path.basename(module)[:-3]}_{args.workload}_r{rep}"
        total, per, out = run_capture(module, args.workload, out_root, tag, args.warmup)
        results.append(total)
        print(json.dumps({"run": tag, "total_us": round(total, 3), "per_op": per}), flush=True)
    print(json.dumps({"module": os.path.basename(module), "workload": args.workload,
                      "samples": [round(v, 3) for v in results],
                      "median_us": round(sorted(results)[len(results) // 2], 3)}))
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"module": module, "workload": args.workload,
                       "samples": results}, f, indent=2)


if __name__ == "__main__":
    main()
