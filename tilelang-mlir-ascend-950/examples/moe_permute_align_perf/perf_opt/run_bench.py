"""Stage 4 measurement driver (platform protocol: msprof kernel `Task Duration`).

Runs the frozen TileOPs benchmark
``benchmarks/ops/bench_moe_permute_align.py`` (one msprof capture per call, as
``ManifestBenchmark.profile`` does) against either the Stage 3 baseline source or
a candidate file injected through ``inject_site/sitecustomize.py``, then reports
per-workload ``t_op_us`` = the sum of that call's kernel ``Task Duration`` (the
platform's declared metric, ``mfu_baseline_v1.json`` policy ``t_op``).

Usage (workdir = ``examples/TileOPs``, after sourcing the platform env):

    python perf_opt/run_bench.py --tag r1_c1 --candidate perf_opt/x.py [--repeat 3]
    python perf_opt/run_bench.py --tag r0_base                      # baseline

Every invocation also copies the platform report directory into
``perf_opt/profiles/<tag>/`` so the raw msprof CSVs stay as evidence.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2] / "TileOPs"   # .../examples/TileOPs
TASK = Path(__file__).resolve().parent              # .../perf_opt
BENCH = "benchmarks/ops/bench_moe_permute_align.py"
OP = "MoEPermuteAlignFwdOp"


def _env(candidate, args):
    env = dict(os.environ)
    parts = [str(TASK / "inject_site")]
    if env.get("PYTHONPATH"):
        parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(parts)
    env["TMPDIR"] = env.get("TMPDIR", "/workspace")
    env["TILELANG_CACHE_DIR"] = env.get(
        "TILELANG_CACHE_DIR", "/tmp/opencode/tl-cache-opt-moe"
    )
    # t_op (sum of kernel `Task Duration`) is only attached to the result when
    # the profiler output is kept; keep it under this tag's profile directory.
    env["TILEOPS_MSPROF_KEEP_OUTPUT"] = "1"
    env["TILEOPS_MSPROF_ARTIFACT_ROOT"] = str(TASK / "profiles" / args.tag / "msprof")
    if candidate:
        env["MOE_ALIGN_CANDIDATE"] = str(Path(candidate).resolve())
    else:
        env.pop("MOE_ALIGN_CANDIDATE", None)
    return env


def _case_rows(report):
    data = json.loads(report.read_text())
    rows = {}
    for rec in data["records"]:
        prof = rec["result"].get("msprof") or {}
        rows[rec["label"]] = {
            "t_op_us": prof.get("t_op_us"),
            "breakdown": prof.get("t_op_breakdown"),
            "latency_us": rec["result"]["latency_us"],
            "launch_dim": prof.get("sample_count"),
            "artifact_dir": prof.get("artifact_dir") or rec["result"].get("artifact_dir"),
        }
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--candidate", default=None)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--only", default=None, help="comma separated case labels")
    ap.add_argument("--pytest-args", default="-q")
    args = ap.parse_args()

    log_dir = TASK / "logs" / args.tag
    log_dir.mkdir(parents=True, exist_ok=True)
    prof_dir = TASK / "profiles" / args.tag
    prof_dir.mkdir(parents=True, exist_ok=True)
    per_run = []
    for rep in range(args.repeat):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        report = prof_dir / f"bench_run{rep}_{stamp}.json"
        report_env = prof_dir / f"bench_run{rep}_{stamp}.log"
        env = _env(args.candidate, args)
        env["TILEOPS_BENCHMARK_REPORT_PATH"] = str(report_env)
        cmd = [sys.executable, "-m", "pytest", BENCH, *args.pytest_args.split()]
        if args.only:
            cmd += ["-k", " or ".join(args.only.split(","))]
        log = log_dir / f"bench_run{rep}_{stamp}.log"
        with log.open("w") as handle:
            proc = subprocess.run(cmd, cwd=REPO, env=env,
                                  stdout=handle, stderr=subprocess.STDOUT)
        if proc.returncode != 0 or not report.is_file():
            print(f"[bench] pytest failed (run {rep}), see {log}")
            print(log.read_text()[-4000:])
            return 1
        per_run.append(_case_rows(report))
        print(f"[bench] run {rep}: {report}")

    labels = sorted(per_run[0])
    print(f"\n{'case':22s} " + "  ".join(f"run{i}" for i in range(len(per_run))) + "   median   breakdown")
    summary = {}
    for label in labels:
        vals = [run[label]["t_op_us"] for run in per_run]
        med = statistics.median(vals)
        summary[label] = {
            "t_op_us": med,
            "runs": vals,
            "breakdown": per_run[0][label]["breakdown"],
            "latency_us": per_run[0][label]["latency_us"],
        }
        print(f"{label:22s} " + "  ".join(f"{v:7.2f}" for v in vals)
              + f"   {med:8.2f}   {per_run[0][label]['breakdown']}")
    out = TASK / "profiles" / args.tag / "summary.json"
    out.write_text(json.dumps({
        "tag": args.tag,
        "candidate": args.candidate,
        "repeats": args.repeat,
        "cases": summary,
    }, indent=1))
    print(f"\n[bench] summary -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
