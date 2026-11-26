"""Experiment batch runner for the paged_cache_write Stage 4 loop (T-4).

One invocation measures one candidate (a factory module) over the requested
workloads with the platform protocol (``msopprof`` single launch, ``Task
Duration(us)``) and appends one ``perf_records.jsonl`` line per workload --
append-only, never rewriting earlier lines.

    python perf_opt/experiments/run_experiments.py \
        --module perf_opt/paged_cache_write_round1_c1.py \
        --candidate-id r1_c1 --parent-id baseline-stage3 --round 1 \
        --phase candidate --reps 3 --l0

The platform evidence block (hardware_id / toolchain_id / platform_config /
platform_runtime) is read from the task state through ``statectl platform
--task-dir`` so every row carries the exact recorded values.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PERF_OPT = os.path.dirname(HERE)
TASK_DIR = os.path.dirname(PERF_OPT)
sys.path.insert(0, HERE)

import bench_one  # noqa: E402

TUNE = [
    "llama-8b-decode-bfloat16-bfloat16",
    "llama-8b-decode-float16-float16",
    "llama-8b-prefill-bfloat16-bfloat16",
    "qwen3-235b-fp8-bfloat16-float8_e4m3fn",
    "gpt-oss-120b-bfloat16-bfloat16",
    "odd-tokens-and-page-float16-float16",
    "qwen3-235b-fp8-fp16-float16-float8_e4m3fn",
]
SMOKE = [
    "paged-write-smoke-float16",
    "paged-write-smoke-bfloat16",
    "paged-write-smoke-bfloat16-float8_e4m3fn",
]

# Stage 3 module the baseline records point at (read-only upstream artifact).
BASELINE_MODULE = (
    "/workspace/tilelang-mlir-ascend-950/examples/TileOPs/tileops/kernels/attention/"
    "paged_cache_write/paged_cache_write_kernel/_paged_cache_write_kernel.py"
)
BASELINE_ARTIFACT = "../../TileOPs/tileops/kernels/attention/paged_cache_write/paged_cache_write_kernel/_paged_cache_write_kernel.py"
KERNEL_ID = (
    "tileops/kernels/attention/paged_cache_write/paged_cache_write_kernel/"
    "_paged_cache_write_kernel.py::_paged_cache_write_kernel::main"
)

# The frozen wrapper's tile policy (paged_cache_write.py::forward) for the
# baseline module, which has no ``pick_config`` of its own.
_WRAPPER_FULL_ROW_MIN_TOKENS = 6144
_WRAPPER_FULL_ROW_MAX = 1024


STATE_TOOLS = "/workspace/tilelang-mlir-ascend-950/.agents/tools"


def platform_evidence():
    out = subprocess.run(
        [sys.executable, os.path.join(STATE_TOOLS, "statectl.py"),
         "platform", "--task-dir", TASK_DIR],
        capture_output=True, text=True, env=dict(os.environ), check=True,
    )
    data = json.loads(out.stdout)
    sel = data["platform"]
    return {
        "hardware_id": sel["hardware_id"],
        "toolchain_id": sel["toolchain_id"],
        "platform_config": sel["platform_config"],
        "platform_runtime": sel["runtime"],
    }


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def baseline_config(wid, mod):
    """Tile the *frozen wrapper* would pick for the Stage 3 module."""
    g = bench_one.geometry(wid)
    rw = g["heads_kv"] * g["dim"]
    if g["tokens"] >= _WRAPPER_FULL_ROW_MIN_TOKENS:
        cols = min(_WRAPPER_FULL_ROW_MAX, -(-rw // 128) * 128)
        return (16, cols)
    return (16, 128)


def run_l0(module):
    r = subprocess.run([sys.executable, module, "--level", "L0"],
                       cwd=os.path.dirname(module), capture_output=True, text=True,
                       timeout=7200)
    log_dir = os.path.join(PERF_OPT, "logs", "l0")
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, os.path.basename(module)[:-3] + ".log"), "w") as f:
        f.write(r.stdout + r.stderr)
    ok = r.returncode == 0 and "[PRECISION_PASS]" in r.stdout
    print(f"[L0] {os.path.basename(module)} rc={r.returncode} ok={ok}")
    if not ok:
        print(r.stdout[-4000:], r.stderr[-2000:])
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default=None, help="variant factory module (default: Stage 3)")
    ap.add_argument("--candidate-id", required=True)
    ap.add_argument("--parent-id", default=None)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--phase", required=True,
                    choices=["baseline", "candidate", "merged", "final"])
    ap.add_argument("--workloads", default="tune")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--l0", action="store_true", help="run the module's --level L0 first")
    ap.add_argument("--l0-module", default=None)
    ap.add_argument("--l0-log", default=None,
                    help="log of an earlier --level L0 run of this artifact")
    ap.add_argument("--config", default=None, help="block_rows:block_cols override")
    ap.add_argument("--configs", default=None,
                    help="per-workload override, e.g. 'wid=32:1024,wid2=16:512'")
    ap.add_argument("--baseline-policy", action="store_true",
                    help="use the frozen wrapper's tile policy (Stage 3 module)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--records", default=os.path.join(PERF_OPT, "perf_records.jsonl"))
    args = ap.parse_args()

    module = os.path.abspath(args.module or BASELINE_MODULE)
    artifact = (BASELINE_ARTIFACT if module == BASELINE_MODULE
                else os.path.basename(module))
    if args.workloads == "tune":
        wids = TUNE
    elif args.workloads == "smoke":
        wids = SMOKE
    elif args.workloads == "all":
        wids = TUNE + SMOKE
    else:
        wids = args.workloads.split(",")

    # l0_pass is only claimed from a run this script made (or a log it is
    # pointed at); a variant with no L0 evidence records False.
    l0_ok = False
    if args.l0:
        l0_ok = run_l0(os.path.abspath(args.l0_module or module))
        if not l0_ok:
            print("[L0] FAILED -- measurements skipped")
            return 1
    elif args.l0_log:
        with open(args.l0_log) as f:
            text = f.read()
        l0_ok = "[PRECISION_PASS]" in text and "case(s) failed" not in text
        print(f"[L0] from {args.l0_log}: pass={l0_ok}")
    if not l0_ok and not args.no_record:
        print("[L0] no passing evidence -- records would claim l0_pass=False; "
              "pass --l0 or --l0-log")

    mod = bench_one.load_module(module)
    evidence = platform_evidence()
    tag = args.tag or args.candidate_id
    rows_out = []
    per_wl = {}
    if args.configs:
        for item in args.configs.split(","):
            name, cfg = item.split("=")
            per_wl[name.strip()] = tuple(int(v) for v in cfg.split(":"))
    for wid in wids:
        if wid in per_wl:
            cfg = per_wl[wid]
        elif args.config:
            br, bc = (int(v) for v in args.config.split(":"))
            cfg = (br, bc)
        elif args.baseline_policy:
            cfg = baseline_config(wid, mod)
        else:
            cfg = bench_one.block_config(mod, wid)
        g = bench_one.geometry(wid)
        mod._paged_cache_write_kernel(
            g["heads_kv"], g["dim"], g["num_pages"] * g["page_size"],
            g["dtype"], g["cache_dtype"])(*cfg)
        samples, per_op = [], None
        out_root = os.path.join(PERF_OPT, "profiles", "experiments")
        for rep in range(args.reps):
            out_dir = os.path.join(out_root, f"{tag}_{wid}_r{rep}")
            total, rows = bench_one.run_capture(module, wid, out_dir, "BasicInfo", cfg,
                                                warmup=(rep == 0))
            samples.append(total)
            per_op = {r[1]: float(r[2]["Task Duration(us)"]) for r in rows}
            if len(rows) != 1:
                print(f"[WARN] {wid}: {len(rows)} captured rows (expected one launch)")
        med = sorted(samples)[len(samples) // 2]
        spread = (max(samples) - min(samples)) / med * 100 if med else 0.0
        rec = {
            **evidence,
            "round": args.round,
            "candidate_id": args.candidate_id,
            "parent_id": args.parent_id,
            "kernel_id": KERNEL_ID,
            "workload_id": wid,
            "phase": args.phase,
            "artifact_path": artifact,
            "artifact_sha256": sha256(module),
            "duration_us": round(med, 3),
            "l0_pass": bool(l0_ok),
            "msprof_raw_path": os.path.relpath(
                os.path.join(out_root, f"{tag}_{wid}_r0"), PERF_OPT),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "note": (f"config={cfg[0]}x{cfg[1]} samples={[round(s, 3) for s in samples]} "
                     f"spread={spread:.2f}% ops={sorted(per_op)}"),
        }
        rows_out.append(rec)
        print(json.dumps({"workload": wid, "config": list(cfg),
                          "median_us": round(med, 3),
                          "samples": [round(s, 3) for s in samples],
                          "spread_pct": round(spread, 2)}), flush=True)
        if not args.no_record:
            with open(args.records, "a") as f:
                f.write(json.dumps(rec, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
