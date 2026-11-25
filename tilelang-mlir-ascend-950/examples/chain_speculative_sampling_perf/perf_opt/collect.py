"""Platform latency collection (msopprof, kernel-only Task Duration) for one impl.

Runs the task platform's profiler over ``<impl> --case <workload>``, parses the
``OpBasicInfo`` CSVs it writes and reports the median Task Duration in
microseconds -- the metric every record in ``perf_records.jsonl`` uses.

Usage (task dir, after sourcing the platform env):
  python perf_opt/collect.py --impl perf_opt/ChainSpeculativeSamplingFwdOp.py \
      --label ds-v3-mtp-n3 --out perf_opt/profiles/round2/final/ds-v3-mtp-n3 \
      --launch-count 15 --warm-up 5 --knobs fold=hoist

``--impl baseline`` profiles the Stage 3 kernel file instead.
"""

import argparse
import csv
import json
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

BASELINE = (
    Path(__file__).resolve().parents[2]
    / "TileOPs/tileops/kernels/sampling/chain_speculative_sampling"
    / "chain_speculative_sampling_kernel/_chain_speculative_sampling_kernel.py"
)
TILEOPS_ROOT = Path(__file__).resolve().parents[2] / "TileOPs"


def samples(out_dir):
    """Every ``(op_name, task_duration_us)`` row the profiler wrote."""
    rows = []
    for capture in sorted(Path(out_dir).glob("OPPROF_*")):
        for path in sorted(capture.rglob("OpBasicInfo*.csv")):
            with path.open() as handle:
                for row in csv.DictReader(handle):
                    value = row.get("Task Duration(us)")
                    if value:
                        rows.append((row.get("Op Name") or row.get("OpName") or "", float(value)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", default="perf_opt/ChainSpeculativeSamplingFwdOp.py")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--launch-count", type=int, default=15)
    ap.add_argument("--warm-up", type=int, default=5)
    ap.add_argument("--knobs", default="")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.chmod(out, 0o700)

    if args.impl == "baseline":
        script = Path(__file__).resolve().parent / "baseline_case.py"
        extra = []
        script_args = ["--case", args.label]
    else:
        script = Path(args.impl).resolve()
        extra = []
        script_args = ["--case", args.label]

    cmd = [
        "msopprof",
        f"--output={out}",
        f"--launch-count={args.launch_count}",
        f"--warm-up={args.warm_up}",
        "--aic-metrics=Roofline",
        sys.executable,
        str(script),
        *extra,
        *script_args,
    ]
    if args.knobs:
        cmd.extend(["--knobs", args.knobs])
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{TILEOPS_ROOT}:{env.get('PYTHONPATH', '')}"
    log_dir = Path(__file__).resolve().parent / "logs" / out.parent.name
    log_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1800)
    (log_dir / f"{out.name}.log").write_text(
        "# " + " ".join(cmd) + "\n" + proc.stdout + "\n" + proc.stderr
    )
    if proc.returncode != 0:
        print(f"PROFILER_FAIL rc={proc.returncode} log={log_dir / (out.name + '.log')}")
        print(proc.stderr[-2000:])
        return 1
    rows = samples(out)
    if not rows:
        print(f"NO_ROWS dir={out}")
        return 1
    by_op = {}
    for name, value in rows:
        by_op.setdefault(name, []).append(value)
    payload = {
        "workload_id": args.label,
        "impl": str(script),
        "knobs": args.knobs,
        "raw_profile_dir": str(out),
        "ops": {name: {"n": len(v), "median_us": round(statistics.median(v), 3),
                       "min_us": round(min(v), 3), "max_us": round(max(v), 3)}
                for name, v in by_op.items()},
    }
    total = sum(entry["median_us"] for entry in payload["ops"].values())
    payload["task_duration_us"] = round(total, 3)
    print(json.dumps(payload, indent=1))
    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
