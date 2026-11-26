"""Platform capture driver: one operator call under msopprof, per-kernel sums.

Protocol (task platform ruleset; ``msopprof`` semantics measured on this host):

* the child (``launch.py``) performs exactly **one** operator call
  (``warm_up=0``), matching the policy the MFU baseline used
  (``examples/batch2/msprof_sum.py``: "one profiled operator call per capture");
* ``msopprof --launch-count=<max launches per call>`` captures every kernel
  launch of that call; one ``OPPROF_*/<Op>/<i>/OpBasicInfo*.csv`` row is one
  launch and ``Task Duration(us)`` is the only duration used;
* ``T_op`` = **sum of all captured launches** of the call -- never a single
  kernel standing in for the operator, never a median over pooled launches
  (the two ``gemm_basic_kernel`` launches differ 2x, so pooling would be wrong);
* ``--repeats R`` re-runs the whole capture in R fresh processes and reports the
  per-position median, so a run is not decided by one noisy sample.

Usage::

    python capture.py --impl baseline --workload kimi-k2-t512 --tag round0-baseline
"""

import argparse
import csv
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PERF_OPT = HERE.parent
PROFILE_ROOT = PERF_OPT / "profiles"
LOG_ROOT = PERF_OPT / "logs"
AIC_METRICS = (
    "BasicInfo,PipeUtilization,ArithmeticUtilization,Memory,MemoryUB,MemoryL0,"
    "L2Cache,ResourceConflictRatio"
)
LAUNCH_SLOTS = 8


def _clean_tmp():
    for path in Path("/tmp").glob("tileops_msprof_*"):
        shutil.rmtree(path, ignore_errors=True)


def _index(path):
    match = re.search(r"/(\d+)/OpBasicInfo", str(path))
    return int(match.group(1)) if match else -1


def parse_outputs(output_dir):
    """Ordered ``[(op_name, duration_us, csv_dir)]`` for one captured call."""
    rows = []
    for path in Path(output_dir).rglob("OpBasicInfo*.csv"):
        with path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                name = (row.get("Op Name") or "").strip()
                raw = (row.get("Task Duration(us)") or "").strip()
                if name and raw:
                    try:
                        rows.append((_index(path), name, float(raw), path.parent))
                    except ValueError:
                        continue
    rows.sort()
    return [(name, dur, parent) for _, name, dur, parent in rows]


_PIPE_KEYS = (
    "aic_cube_ratio",
    "aic_mte1_ratio",
    "aic_mte2_ratio",
    "aic_fixpipe_ratio",
    "aic_mte2_active_bw",
    "aiv_vec_ratio",
)


def parse_pipe_metrics(csv_dir):
    """Mean per-pipe ratios of one launch (diagnostic only, not a metric)."""
    out = {}
    paths = sorted(Path(csv_dir).glob("PipeUtilization_*.csv"))
    if not paths:
        return out
    sums = {key: [] for key in _PIPE_KEYS}
    with paths[0].open(newline="") as fh:
        for row in csv.DictReader(fh):
            for column, raw in row.items():
                if column is None:
                    continue
                key = column.split("(")[0].strip()
                if key not in sums:
                    continue
                raw = (raw or "").strip()
                if not raw or raw == "NA":
                    continue
                try:
                    sums[key].append(float(raw))
                except ValueError:
                    continue
    for key, values in sums.items():
        if values:
            out[key] = round(sum(values) / len(values), 4)
    return out


def run_once(args, raw_dir):
    log_dir = LOG_ROOT / args.tag
    msprof = shutil.which("msopprof")
    if msprof is None:
        raise SystemExit("msopprof not found in PATH")
    cmd = [
        msprof,
        f"--output={raw_dir}",
        f"--launch-count={LAUNCH_SLOTS}",
        "--warm-up=0",
        f"--aic-metrics={AIC_METRICS}",
        sys.executable,
        str(HERE / "launch.py"),
        "--impl",
        args.impl,
        "--workload",
        args.workload,
        "--warm-up",
        "0",
        "--launch-count",
        "1",
        "--seed",
        str(args.seed),
    ]
    if args.config:
        cmd += ["--config", args.config]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(HERE), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    started = time.time()
    with (log_dir / f"{args.workload}.log").open("w") as log:
        log.write(" ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    return raw_dir, proc.returncode, time.time() - started


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", default="baseline")
    ap.add_argument("--workload", required=True)
    ap.add_argument("--tag", required=True, help="profile_stage/candidate identifier")
    ap.add_argument("--config", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--reuse", action="store_true", help="keep an existing capture")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    out_root = PROFILE_ROOT / args.tag / args.workload
    (LOG_ROOT / args.tag).mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "summary.json"

    if not (args.reuse and summary_path.exists()):
        runs = []
        for repeat in range(args.repeats):
            raw_dir = out_root / f"raw_run{repeat}"
            if raw_dir.exists():
                shutil.rmtree(raw_dir)
            raw_dir.mkdir(parents=True, exist_ok=True)
            os.chmod(raw_dir, 0o700)
            _clean_tmp()
            code, wall = 0, 0.0
            cmd_dir, code, wall = run_once(args, raw_dir)
            _clean_tmp()
            if code != 0:
                raise SystemExit(f"[capture] msopprof exited {code}; see {LOG_ROOT/args.tag}")
            launches = parse_outputs(raw_dir)
            if not launches:
                raise SystemExit(f"[capture] no OpBasicInfo rows under {raw_dir}")
            runs.append(
                {
                    "repeat": repeat,
                    "raw_profile_dir": str(raw_dir),
                    "wall_s": round(wall, 1),
                    "launches": [
                        {
                            "pos": pos,
                            "op_name": name,
                            "duration_us": dur,
                            "pipes": parse_pipe_metrics(parent),
                        }
                        for pos, (name, dur, parent) in enumerate(launches)
                    ],
                    "t_op_us": sum(d for _, d, _ in launches),
                }
            )

        npos = max(len(run["launches"]) for run in runs)
        per_position = []
        for pos in range(npos):
            entries = [run["launches"][pos] for run in runs if len(run["launches"]) > pos]
            durs = [entry["duration_us"] for entry in entries]
            pipes = {}
            for key in _PIPE_KEYS:
                values = [entry["pipes"].get(key) for entry in entries]
                values = [v for v in values if v is not None]
                if values:
                    pipes[key] = round(statistics.median(values), 4)
            per_position.append(
                {
                    "pos": pos,
                    "op_name": entries[0]["op_name"],
                    "median_us": statistics.median(durs),
                    "all_us": durs,
                    "pipes": pipes,
                }
            )
        summary = {
            "impl": args.impl,
            "workload": args.workload,
            "tag": args.tag,
            "note": args.note,
            "config": json.loads(args.config) if args.config else {},
            "repeats": args.repeats,
            "t_op_us": sum(entry["median_us"] for entry in per_position),
            "t_op_all_us": [run["t_op_us"] for run in runs],
            "launches": per_position,
            "runs": runs,
        }
        with summary_path.open("w") as fh:
            json.dump(summary, fh, indent=2)

    summary = json.loads(summary_path.read_text())
    print(
        f"[{summary['tag']}] {summary['workload']} impl={summary['impl']} "
        f"cfg={summary['config']} repeats={summary['repeats']}"
    )
    for entry in summary["launches"]:
        pipes = entry["pipes"]
        extra = ""
        if pipes:
            extra = (
                f" cube={pipes.get('aic_cube_ratio', float('nan')):.3f}"
                f" mte2={pipes.get('aic_mte2_ratio', float('nan')):.3f}"
                f" mte2bw={pipes.get('aic_mte2_active_bw', float('nan')):.0f}GB/s"
                f" mte1={pipes.get('aic_mte1_ratio', float('nan')):.3f}"
            )
        print(
            f"  #{entry['pos']} {entry['op_name']:<24} "
            f"{entry['median_us']:9.3f} us (all={['%.1f' % v for v in entry['all_us']]}){extra}"
        )
    print(f"  T_op = {summary['t_op_us']:.3f} us   runs={summary['t_op_all_us']}")


if __name__ == "__main__":
    main()
