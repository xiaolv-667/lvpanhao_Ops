"""Single-call msprof driver for one EngramGateConvFwd kernel variant (Stage 4).

Protocol (task platform, `measurement.raw_evidence_field = msprof_raw_path`):
``msopprof --launch-count=1 --warm-up=0 --kernel-name=main --aic-metrics=Roofline``
around **exactly one** operator call, then read ``Task Duration(us)`` from
``OpBasicInfo*.csv``.  One capture = one operator call (the kernel issues a
single ``main_kernel`` launch; the driver asserts that and reports per-op).

The ~10 us scale of this operator makes a single capture noisy, so every
reported number is the **median of ``--reps`` independent capture processes**
(each process = warm-up process + one profiled process).  The sample list is
always printed so the spread stays visible.

Usage (cwd must be ``examples/TileOPs`` per the platform execution context):

    python <this file> --module <variant.py> --workload s96 [--reps 3]
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

# label -> (M, seq_len, d, dtype_str)   (tileops.manifest:EngramGateConvFwdOp)
WORKLOADS = {
    "s32": (1, 32, 256, "float16"),    # train-s32  (benchmark idx 0 -> smoke)
    "s64": (2, 64, 512, "float16"),    # train-s64
    "s128": (1, 128, 256, "bfloat16"),  # train-s128
    "s96": (3, 96, 768, "bfloat16"),   # train-s96 (dominant)
}

_DT = {"float16": torch.float16, "bfloat16": torch.bfloat16}


def load_module(path):
    name = "engram_variant_" + os.path.basename(path).replace(".py", "").replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def make_inputs(wl, dtype="cpu"):
    """Manifest input recipe (workloads.sequence_modeling.EngramGateConvFwdWorkload)."""
    M, seq_len, d, dt = WORKLOADS[wl]
    t = _DT[dt]
    H = torch.randn(M, seq_len, d, dtype=t)
    k = torch.randn(M, seq_len, d, dtype=t) * 0.1
    v = torch.randn(M, seq_len, d, dtype=t) * 0.1
    rms_w_h = torch.ones(d, dtype=t)
    rms_w_v = torch.ones(d, dtype=t)
    conv_w = torch.randn(4, d, dtype=t) * 0.02
    return H, k, v, rms_w_h, rms_w_v, conv_w


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
fn = mod._engram_gate_conv_fwd_kernel(*{geom!r})
torch.npu.synchronize()
fn(*args)
torch.npu.synchronize()
'''


def run_capture(module_path, wl, out_root, tag, warmup, kernel_name="main", launch_count=1):
    M, seq_len, d, dt = WORKLOADS[wl]
    tmp = tempfile.mkdtemp(prefix="bench_one_", dir=os.environ.get("TMPDIR", "/workspace"))
    try:
        args_path = os.path.join(tmp, "args.pt")
        torch.save(list(make_inputs(wl)), args_path)
        torch.npu.empty_cache()
        script = os.path.join(tmp, "run.py")
        # warm-up launches run in their own process so that no un-measured launch
        # can land inside the profiled capture window.
        body = CHILD.format(path=module_path, args=args_path, geom=(M, seq_len, d, 1e-6, dt))
        with open(script, "w") as f:
            f.write(body)
        env = dict(os.environ)
        env["PYTHONPATH"] = os.getcwd() + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        if warmup:
            subprocess.run([sys.executable, script], cwd=os.getcwd(), env=env,
                           capture_output=True, timeout=1800)
        out = os.path.join(out_root, tag)
        if os.path.isdir(out):
            shutil.rmtree(out)
        os.makedirs(out)
        os.chmod(out, 0o700)
        cmd = ["msopprof", f"--output={out}", f"--launch-count={launch_count}", "--warm-up=0"]
        if kernel_name:
            cmd.append(f"--kernel-name={kernel_name}")
        cmd += ["--aic-metrics=Roofline", sys.executable, script]
        res = subprocess.run(cmd, cwd=os.getcwd(), env=env, capture_output=True, text=True,
                             timeout=1800)
        if res.returncode != 0:
            raise RuntimeError(f"msopprof failed rc={res.returncode}: {res.stderr[-3000:]}")
        per, total, rows = {}, 0.0, 0
        for path in sorted(glob.glob(os.path.join(out, "**", "OpBasicInfo*.csv"), recursive=True)):
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
    ap.add_argument("--workload", default="s96")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--launch-count", type=int, default=1,
                    help="measured launches per capture; 2 for a two-launch candidate")
    ap.add_argument("--kernel-name", default="main",
                    help="msopprof kernel-name filter (substring match); use 'kernel' "
                         "for multi-launch candidates")
    args = ap.parse_args()
    module = os.path.abspath(args.module)
    out_root = os.path.abspath(args.out_root or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "profiles", "experiments"))
    os.makedirs(out_root, exist_ok=True)

    # Populate the tilelang JIT cache in this process first so no child process
    # pays compilation inside its capture window.
    mod = load_module(module)
    M, seq_len, d, dt = WORKLOADS[args.workload]
    mod._engram_gate_conv_fwd_kernel(M, seq_len, d, 1e-6, dt)
    torch.npu.synchronize()

    results = []
    for rep in range(args.reps):
        tag = f"{os.path.basename(module)[:-3]}_{args.workload}_r{rep}"
        total, per, out = run_capture(
            module, args.workload, out_root, tag, args.warmup, args.kernel_name,
            args.launch_count
        )
        results.append(total)
        print(json.dumps({"run": tag, "total_us": round(total, 3), "per_op": per}), flush=True)
    med = sorted(results)[len(results) // 2]
    print(json.dumps({"module": os.path.basename(module), "workload": args.workload,
                      "samples": [round(v, 3) for v in results], "median_us": round(med, 3)}))
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"module": module, "workload": args.workload, "samples": results,
                       "median_us": med}, f, indent=2)


if __name__ == "__main__":
    main()
