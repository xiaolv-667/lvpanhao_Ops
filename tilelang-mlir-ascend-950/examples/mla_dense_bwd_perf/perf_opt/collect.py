"""Platform latency collection for MLA-dense-bwd (kernel Task Duration(us)).

Runs the task platform's profiler over one call of ``mla_dense_case.py`` and
reports, per captured op, the summed ``Task Duration(us)`` plus the summed
``T_op`` -- the metric every record in ``perf_records.jsonl`` uses.  ``T_op`` is
what the production report calls ``t_op_us``: one operator call launches
``Range`` + ``ZerosLike`` + ``_delta_main`` + ``_bwd_main`` + ``_dq_land_main``,
and the sum of the per-op samples is one call's kernel time (same convention as
``bench_kernel_msprof``).

The profiled subprocess only *loads* its inputs (generated once into a cache
file): input generation inside the profiled process fills the capture window
with ``StatelessNormal`` / forward kernels, which is why the phase split exists.

Usage (cwd ``examples/TileOPs``, platform python):
    python ../mla_dense_bwd_perf/perf_opt/collect.py --impl baseline \
        --case ds-v3-b2-8k-bfloat16 --out .../perf_opt/profiles/round0/b2-8k_baseline

``--impl`` is ``baseline`` / ``op`` / a candidate kernel module path.
``--metrics full`` switches to the task ruleset's diagnostic metric set
(PipeUtilization / ArithmeticUtilization / Memory / L0 / L2 ...) instead of the
benchmark ``Roofline`` set, for pipe-busy attribution.
"""

import argparse
import csv
import importlib
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TILEOPS_ROOT = HERE.parents[1] / "TileOPs"
# The venv carries an editable install of another TileOPs checkout; every module
# of this runner must resolve ``tileops`` from the migrated tree instead.
if str(TILEOPS_ROOT) not in sys.path:
    sys.path.insert(0, str(TILEOPS_ROOT))
INPUT_CACHE = Path("/workspace/mla_dense_bwd_opt/inputs")
FULL_METRICS = (
    "BasicInfo,PipeUtilization,ArithmeticUtilization,Memory,MemoryUB,MemoryL0,"
    "L2Cache,ResourceConflictRatio"
)
PIPE_METRICS = "BasicInfo,PipeUtilization"


def ensure_inputs(case_id, custom=""):
    """Path of the cached input file of one case (generated on first use)."""
    INPUT_CACHE.mkdir(parents=True, exist_ok=True)
    tag = case_id if not custom else "custom_" + case_id
    path = INPUT_CACHE / f"{tag}.pt"
    if path.is_file():
        return path
    cmd = [sys.executable, str(HERE / "mla_dense_case.py"), "--case", case_id,
           "--save-inputs", str(path)]
    if custom:
        cmd += ["--custom", custom]
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{TILEOPS_ROOT}:{env.get('PYTHONPATH', '')}"
    print("[inputs] generating", tag, flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=3600)
    if proc.returncode != 0:
        raise SystemExit(f"input generation failed:\n{proc.stdout[-2000:]}\n"
                         f"{proc.stderr[-3000:]}")
    return path


def read_op_rows(out_dir):
    """Every ``(op_name, task_duration_us)`` row the profiler wrote."""
    rows = []
    for path in sorted(Path(out_dir).glob("**/OpBasicInfo*.csv")):
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                value = row.get("Task Duration(us)")
                if value is None or not value.strip():
                    continue
                rows.append((row.get("Op Name", "").strip(), float(value)))
    return rows


def case_kwargs(case_id, custom=""):
    """Geometry kwargs of one case (manifest sweep or ``--custom``)."""
    if custom:
        raw = json.loads(custom)
        seq = int(raw["seq_q"])
        return dict(batch=int(raw["batch"]), seq_q=seq,
                    seq_kv=int(raw.get("seq_kv", seq)), heads=int(raw["heads"]),
                    dim_qk=int(raw.get("dim_qk", 192)), dim_v=int(raw.get("dim_v", 128)))
    from tileops.manifest import load_workloads

    for workload in load_workloads("MLADenseBwdOp"):
        for dtype_name in workload["dtypes"]:
            if f"{workload['label']}-{dtype_name}" == case_id:
                return dict(batch=int(workload["B"]), seq_q=int(workload["S_q"]),
                            seq_kv=int(workload["S_kv"]), heads=int(workload["H"]),
                            dim_qk=int(workload["D_qk"]), dim_v=int(workload["D_v"]))
    raise SystemExit(f"--case {case_id}: not a benchmark case")


def launches_per_call(case_id, custom=""):
    """Kernel launches of one operator call: Range + ZerosLike + 3 kernels."""
    from mla_dense_case import LAUNCHES_PER_CALL

    case_kwargs(case_id, custom)
    return LAUNCHES_PER_CALL


def iteration_count(case_id, custom=""):
    """Block-iteration count of one case: sum over (request, key tile).

    ``iters = sum_b sum_kn (ceil(S_q/64) - loop_st)`` with
    ``loop_st = max(0, (96*kn - shift)//64)`` -- the DESIGN.md 1.6.0 R3 formula,
    recomputed here so a per-iteration price can be read off ``T_op``.
    """
    geo = case_kwargs(case_id, custom)
    seq_q, seq_kv, heads = geo["seq_q"], geo["seq_kv"], geo["heads"]
    shift = seq_kv - seq_q
    total = 0
    for kn in range(-(-seq_kv // 96)):
        loop_st = max(0, (96 * kn - shift) // 64)
        loop_ed = max(-(-seq_q // 64), loop_st + 1)
        total += max(0, loop_ed - loop_st)
    return total * heads * geo["batch"]


def read_pipe_ratios(out_dir, op_match="_bwd_main"):
    """Per-core-type pipe busy metrics of the main pass.

    ``PipeUtilization`` rows are one per (block, sub_block); ``sub_block_id``
    starts with ``cube`` for the AIC rows and ``vector`` for the AIV rows, so the
    two are averaged separately.  Besides the per-block means, the ``*_share``
    fields give the true busy fraction of the core type's own time
    (``sum(pipe time) / sum(core time)``), which is what a busy-fraction claim
    must use when the block durations are heterogeneous.
    """
    groups = {"cube": {}, "vector": {}}
    files = [p for p in sorted(Path(out_dir).glob("**/PipeUtilization*.csv"))
             if op_match in str(p)]
    for path in files:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or []
            sums, counts = {}, {}
            for row in reader:
                sub = (row.get("sub_block_id") or "").strip()
                group = "cube" if sub.startswith("cube") else (
                    "vector" if sub.startswith("vector") else None)
                if group is None:
                    continue
                for key in fields:
                    if key.endswith("_ratio") or key.endswith("_time(us)") \
                            or key.endswith("_active_bw(GB/s)"):
                        try:
                            value = float(row[key])
                        except (TypeError, ValueError):
                            continue
                        bucket = (group, key)
                        sums[bucket] = sums.get(bucket, 0.0) + value
                        counts[bucket] = counts.get(bucket, 0) + 1
        for (group, key), total in sums.items():
            groups[group][key] = round(total / max(counts[(group, key)], 1), 6)
        for group, core_key in (("cube", "aic_time(us)"), ("vector", "aiv_time(us)")):
            total_core = sum(total for (g, k), total in sums.items()
                             if g == group and k == core_key)
            if total_core <= 0:
                continue
            for (g, k), total in sums.items():
                if g == group and k.endswith("_time(us)"):
                    groups[group][k.replace("_time(us)", "_share")] = round(
                        total / total_core, 6)
    return {group: values for group, values in groups.items() if values}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", default="baseline")
    parser.add_argument("--case", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--launch-count", type=int, default=1)
    parser.add_argument("--warm-up", type=int, default=5)
    parser.add_argument("--metrics", default="roofline",
                        choices=["roofline", "full", "pipe"])
    parser.add_argument("--custom", default="")
    parser.add_argument("--json", default="")
    parser.add_argument("--no-capture", action="store_true",
                        help="only re-parse an existing --out directory")
    args = parser.parse_args()

    out = Path(args.out).resolve()
    log_dir = HERE / "logs" / out.parent.name
    log_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_capture:
        out.mkdir(parents=True, exist_ok=True)
        os.chmod(out, 0o700)
        inputs = ensure_inputs(args.case, args.custom)
        metrics = {"roofline": "Roofline", "full": FULL_METRICS,
                   "pipe": PIPE_METRICS}[args.metrics]
        per_call = launches_per_call(args.case, args.custom)
        # The profiled process runs one setup call (compile) + ``warm-up`` calls
        # + one measured call; the profiler skips a whole number of calls and
        # captures exactly one (``--launch-count`` counts kernel launches).
        prof_warm = (1 + args.warm_up) * per_call
        prof_capture = per_call
        cmd = [
            "msopprof",
            f"--output={out}",
            f"--launch-count={prof_capture}",
            f"--warm-up={prof_warm}",
            f"--aic-metrics={metrics}",
            sys.executable,
            str(HERE / "mla_dense_case.py"),
            "--impl", args.impl,
            "--inputs", str(inputs),
            "--case", args.case,
            "--warm-up", str(args.warm_up),
            "--launch-count", "1",
        ]
        env = dict(os.environ)
        env["PYTHONPATH"] = f"{TILEOPS_ROOT}:{env.get('PYTHONPATH', '')}"
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=7200)
        (log_dir / f"{out.name}.log").write_text(
            "# " + " ".join(cmd) + "\n" + proc.stdout + "\n--- stderr ---\n" + proc.stderr
        )
        if proc.returncode != 0:
            print(f"PROFILER_FAIL rc={proc.returncode} "
                  f"log={log_dir / (out.name + '.log')}")
            print(proc.stderr[-3000:])
            return 1
    rows = read_op_rows(out)
    if not rows:
        print(f"NO_ROWS dir={out}")
        return 1
    by_op = {}
    for name, value in rows:
        by_op.setdefault(name, []).append(value)
    payload = {
        "impl": args.impl,
        "case": args.case,
        "launch_count": args.launch_count,
        "warm_up": args.warm_up,
        "metrics": args.metrics,
        "artifact": str(Path(args.impl).resolve()) if args.impl not in
        ("baseline", "op") else args.impl,
        "raw_profile_dir": str(out),
        "ops": {
            name: {
                "n": len(values),
                "sum_us": round(sum(values), 3),
                "median_us": round(statistics.median(values), 3),
                "min_us": round(min(values), 3),
                "max_us": round(max(values), 3),
            }
            for name, values in by_op.items()
        },
    }
    # One operator call = one window, so the call's kernel time is the SUM of
    # every captured sample (the production report's ``t_op_us`` is the same sum).
    payload["task_duration_us"] = round(sum(value for _, value in rows), 3)
    payload["per_call_launches"] = launches_per_call(args.case, args.custom)
    payload["captured_samples"] = sum(entry["n"] for entry in payload["ops"].values())
    payload["window_ok"] = payload["captured_samples"] == payload["per_call_launches"]
    payload["pipe"] = read_pipe_ratios(out)
    payload["iterations"] = iteration_count(args.case, args.custom)
    main_us = sum(entry["sum_us"] for name, entry in payload["ops"].items()
                  if "_bwd_main" in name)
    payload["main_us"] = round(main_us, 3)
    payload["us_per_iteration"] = round(
        main_us * 28.0 / payload["iterations"], 4) if payload["iterations"] else 0.0
    print(json.dumps(payload, indent=1))
    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
