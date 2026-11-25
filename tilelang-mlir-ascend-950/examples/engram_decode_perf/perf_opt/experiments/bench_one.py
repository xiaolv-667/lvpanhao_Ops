"""Single-call msopprof driver for one EngramDecode kernel pair (Stage 4).

Protocol (task platform `measurement`): `msopprof --launch-count=N --warm-up=0
--kernel-name=kernel --aic-metrics=Roofline` around **exactly one** operator call,
then sum `Task Duration(us)` over the captured `OpBasicInfo*.csv` rows.  One decode
step is two launches (`engram_project_kernel` + `engram_decode_step_kernel`), so
`T_op` = the sum of both, and the per-op dict is the breakdown.

The ~5-10 us scale of this operator makes a single capture noisy, so every reported
number is the **median of `--reps` independent capture processes** (each process =
one warm-up process + one profiled process).  Samples are always printed.

Usage (cwd must be `examples/TileOPs` per the platform execution context):

    python <this file> --workload serving-b4 --reps 3 \
        [--project <project kernel module>] [--step <step kernel module>]
    # omit --project/--step for the frozen Stage-3 baseline pair
"""

import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

import torch  # noqa: F401  (only used by the parent for args staging / cache warmup)

# label -> (batch, d_mem, d, max_conv_len, w, dilation, conv_len, eps, dtype)
# exactly `tileops.manifest:EngramDecodeFwdOp` (see workload_inventory.json)
WORKLOADS = {
    # idx 0 of the benchmark parametrization -> pytest.mark.smoke
    "decode-s1": (1, 256, 256, 9, 4, 3, 9, 1e-6, "float16"),
    "serving-b1": (1, 512, 256, 12, 4, 3, 12, 1e-6, "float16"),
    "serving-b4": (4, 1024, 512, 20, 4, 5, 20, 1e-6, "float16"),
    "serving-b8": (8, 512, 256, 18, 4, 3, 18, 1e-6, "bfloat16"),
    "decode-partial-cache": (2, 512, 256, 12, 4, 3, 5, 1e-6, "float16"),
}

BASELINE_DIR = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "TileOPs", "tileops", "kernels", "sequence_modeling",
    "engram_decode", "engram_decode_kernel",
))


CHILD = r'''
import importlib.util, os, sys, torch, torch_npu
torch.npu.set_device(0)
cwd = {cwd!r}
os.chdir(cwd)
sys.path.insert(0, cwd)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


proj = _load({proj!r}, "proj_mod")
step = _load({step!r}, "step_mod")

batch, d_mem, d, mcl, w, dil, conv_len, eps, dt = {geom!r}
torch.manual_seed(0)
dtype = getattr(torch, dt)
d_pad = (d + proj.ALIGNMENT - 1) // proj.ALIGNMENT * proj.ALIGNMENT

# --- inputs exactly as EngramDecodeWorkload.gen_inputs + wrapper forward ---------
dev = "npu"
e_t = torch.randn(batch, d_mem, dtype=dtype, device=dev) * 0.1
h_t = torch.randn(batch, d, dtype=dtype, device=dev)
conv_state = torch.randn(batch, conv_len, d, dtype=dtype, device=dev) * 0.1
W_K = torch.randn(d_mem, d, dtype=dtype, device=dev) * 0.02
W_V = torch.randn(d_mem, d, dtype=dtype, device=dev) * 0.02
rms_w_h = torch.ones(d, dtype=dtype, device=dev)
rms_w_v = torch.ones(d, dtype=dtype, device=dev)
conv_w = torch.randn(w, d, dtype=dtype, device=dev) * 0.02

# host-side padding of the wrapper (kept OUTSIDE the measured call)
import torch.nn.functional as F
if conv_len < mcl:
    conv_state = F.pad(conv_state, (0, 0, mcl - conv_len, 0))
pad = d_pad - d
if pad:
    h_t = F.pad(h_t, (0, pad))
    conv_state = F.pad(conv_state, (0, pad))
    W_K = F.pad(W_K, (0, pad))
    W_V = F.pad(W_V, (0, pad))
    rms_w_h = F.pad(rms_w_h, (0, pad))
    rms_w_v = F.pad(rms_w_v, (0, pad))
    conv_w = F.pad(conv_w, (0, pad))

cfg = proj.project_default_config(d_pad) if hasattr(proj, "project_default_config") \
    else dict(proj.default_config(d_pad))
threads = step.step_default_config(d_pad)["threads"] if hasattr(step, "step_default_config") \
    else step.default_config(d_pad)["threads"]
override = {override!r}
if override:
    cfg.update(override)
if {threads_override!r}:
    threads = {threads_override!r}
project = proj._engram_project_kernel(batch, d_mem, d_pad, mcl, dt)(
    cfg["block_m"], cfg["block_n"], cfg["block_k"], cfg["num_stages"])
st = step._engram_step_kernel(batch, d, d_pad, mcl, w, dil, eps, dt)(threads)

kv = torch.empty((2, batch, d_pad), dtype=torch.float32, device=dev)
y_t = torch.empty((batch, d_pad), dtype=dtype, device=dev)
new_conv_state = torch.empty((batch, mcl, d_pad), dtype=dtype, device=dev)
torch.npu.synchronize()

project(e_t, W_K, W_V, conv_state, kv, new_conv_state)
st(h_t, kv, conv_state, rms_w_h, rms_w_v, conv_w, y_t, new_conv_state)
torch.npu.synchronize()
'''


def run_capture(proj_path, step_path, wl, out_root, tag, warmup, launch_count, kernel_name,
                cfg_override=None, threads_override=None):
    geom = WORKLOADS[wl]
    tmp = tempfile.mkdtemp(prefix="bench_one_", dir=os.environ.get("TMPDIR", "/workspace"))
    try:
        cwd = os.getcwd()
        script = os.path.join(tmp, "run.py")
        with open(script, "w") as f:
            f.write(CHILD.format(cwd=cwd, proj=proj_path, step=step_path, geom=geom,
                                 override=cfg_override, threads_override=threads_override))
        env = dict(os.environ)
        env["PYTHONPATH"] = cwd + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        if warmup:
            # warm-up launches run in their own process so nothing un-measured can
            # land inside the profiled capture window
            r = subprocess.run([sys.executable, script], cwd=cwd, env=env,
                               capture_output=True, text=True, timeout=3600)
            if r.returncode != 0:
                raise RuntimeError(f"warmup failed rc={r.returncode}: {r.stderr[-4000:]}")
        out = os.path.join(out_root, tag)
        if os.path.isdir(out):
            shutil.rmtree(out)
        os.makedirs(out)
        os.chmod(out, 0o700)
        cmd = ["msopprof", f"--output={out}", f"--launch-count={launch_count}", "--warm-up=0"]
        if kernel_name:
            cmd.append(f"--kernel-name={kernel_name}")
        cmd += ["--aic-metrics=Roofline", sys.executable, script]
        res = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=3600)
        if res.returncode != 0:
            raise RuntimeError(f"msopprof failed rc={res.returncode}: {res.stderr[-4000:]}")
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
    ap.add_argument("--project", default=os.path.join(
        BASELINE_DIR, "_engram_project_kernel.py"))
    ap.add_argument("--step", default=os.path.join(BASELINE_DIR, "_engram_step_kernel.py"))
    ap.add_argument("--workload", default="serving-b4", choices=sorted(WORKLOADS))
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--proj-cfg", default="", help="block_m,block_n,block_k,num_stages override")
    ap.add_argument("--step-threads", type=int, default=0, help="override the step lane count")
    ap.add_argument("--launch-count", type=int, default=2,
                    help="measured launches per capture (2 = project + step)")
    ap.add_argument("--kernel-name", default="engram",
                    help="msopprof kernel-name filter (prefix match: 'engram' catches both "
                         "engram_project_kernel and engram_decode_step_kernel)")
    args = ap.parse_args()

    proj_path = os.path.abspath(args.project)
    step_path = os.path.abspath(args.step)
    out_root = os.path.abspath(args.out_root or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "profiles", "experiments"))
    os.makedirs(out_root, exist_ok=True)

    cfg_override = None
    if args.proj_cfg:
        bm, bn, bk, ns = (int(v) for v in args.proj_cfg.split(","))
        cfg_override = {"block_m": bm, "block_n": bn, "block_k": bk, "num_stages": ns}
    threads_override = args.step_threads or None
    base = args.tag or (os.path.basename(proj_path)[:-3] + "+" + os.path.basename(step_path)[:-3])
    results, breakdowns = [], []
    for rep in range(args.reps):
        tag = f"{base}_{args.workload}_r{rep}"
        total, per, _out = run_capture(proj_path, step_path, args.workload, out_root, tag,
                                       args.warmup, args.launch_count, args.kernel_name)
        results.append(total)
        breakdowns.append(per)
        print(json.dumps({"run": tag, "total_us": round(total, 3),
                          "per_op": {k: round(v, 3) for k, v in per.items()}}), flush=True)
    s = sorted(results)
    med = s[len(s) // 2] if len(s) % 2 else 0.5 * (s[len(s) // 2 - 1] + s[len(s) // 2])
    # per-op medians
    per_med = {}
    for key in breakdowns[0]:
        vals = sorted(b.get(key, 0.0) for b in breakdowns)
        per_med[key] = round(vals[len(vals) // 2] if len(vals) % 2
                             else 0.5 * (vals[len(vals) // 2 - 1] + vals[len(vals) // 2]), 3)
    print(json.dumps({"project": os.path.basename(proj_path),
                      "step": os.path.basename(step_path),
                      "workload": args.workload,
                      "samples": [round(v, 3) for v in results],
                      "median_us": round(med, 3), "per_op_median": per_med}))
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"project": proj_path, "step": step_path, "workload": args.workload,
                       "samples": results, "median_us": med, "per_op_median": per_med,
                       "per_op_samples": breakdowns}, f, indent=2)


if __name__ == "__main__":
    main()
