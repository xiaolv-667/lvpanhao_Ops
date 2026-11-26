"""Platform latency collection for MLA-varlen-bwd (kernel Task Duration(us)).

Runs the task platform's profiler over one call of ``mla_bwd_case.py`` and reports,
per captured op, the median ``Task Duration(us)`` plus the summed ``T_op`` -- the
metric every record in ``perf_records.jsonl`` uses.  ``T_op`` is what the
production report calls ``t_op_us``: one operator call launches
``_delta_main`` + N x ``_bwd_main`` + ``_dq_land_main`` + ``ZerosLike``, and the
sum of the per-op medians is one call's kernel time (same convention as
``bench_kernel_msprof``).

The profiled subprocess only *loads* its inputs (generated once into a cache
file): input generation inside the profiled process fills the capture window
with ``StatelessNormal`` kernels, which was observed and is why the phase split
exists.

Usage (task dir, platform python):
    python perf_opt/collect.py --impl baseline --case ds-v3-mixed-bfloat16 \
        --out perf_opt/profiles/baseline/ds-v3-mixed-bfloat16

``--impl`` is ``baseline`` / ``op`` / a candidate kernel module path.
``--metrics full`` switches to the task ruleset's diagnostic metric set
(PipeUtilization / ArithmeticUtilization / Memory / L0 / L2 ...) instead of the
benchmark ``Roofline`` set, for pipe-busy attribution.
"""

import argparse
import csv
import importlib
import importlib.util
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
INPUT_CACHE = Path("/workspace/mla_bwd_opt/inputs")
FULL_METRICS = (
    "BasicInfo,PipeUtilization,ArithmeticUtilization,Memory,MemoryUB,MemoryL0,"
    "L2Cache,ResourceConflictRatio"
)


def ensure_inputs(case_id, custom=""):
    """Path of the cached input file of one case (generated on first use)."""
    INPUT_CACHE.mkdir(parents=True, exist_ok=True)
    path = INPUT_CACHE / f"{case_id}.pt"
    if path.is_file():
        return path
    cmd = [sys.executable, str(HERE / "mla_bwd_case.py"), "--case", case_id,
           "--save-inputs", str(path)]
    if custom:
        cmd += ["--custom", custom]
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{TILEOPS_ROOT}:{env.get('PYTHONPATH', '')}"
    print("[inputs] generating", case_id, flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1800)
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


def case_geometry(case_id, custom=""):
    """``(q_lens, kv_lens, heads)`` of one case (manifest sweep or ``--custom``)."""
    if custom:
        raw = json.loads(custom)
        return (list(raw["q_lens"]), list(raw.get("kv_lens", raw["q_lens"])),
                int(raw["heads"]))
    from tileops.manifest import load_workloads

    for workload in load_workloads("MLAVarlenBwdOp"):
        for dtype_name in workload["dtypes"]:
            if f"{workload['label']}-{dtype_name}" == case_id:
                return (list(workload["q_lens"]), list(workload["kv_lens"]),
                        int(workload["H"]))
    raise SystemExit(f"--case {case_id}: not a benchmark case")


def impl_helpers(impl):
    """The launch-plan module of the impl under test (candidate or baseline).

    The head-slab count is a property of the *measured* kernel, so the capture
    window must be derived from its own ``plan_launch`` -- using the baseline's
    plan silently captures a fraction of a call when the candidate changes the
    slab (observed: a 16-launch candidate measured as 7 launches).
    """
    baseline = importlib.import_module(
        "tileops.kernels.attention.mla_varlen_bwd"
        ".mla_varlen_bwd_kernel._bwd_kernel"
    )
    if impl in ("baseline", "op"):
        return baseline
    path = Path(impl).resolve()
    if not path.is_file():
        return baseline
    spec = importlib.util.spec_from_file_location("collect_candidate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module if hasattr(module, "plan_launch") else baseline


def launches_per_call(case_id, custom="", impl="baseline"):
    """Kernel launches of one operator call: delta + N slabs + ZerosLike + land.

    The profiler's ``--warm-up``/``--launch-count`` count *kernel launches* in the
    process (measured: ``--launch-count=10`` captured exactly 10 launches), so
    one call is captured by skipping a whole number of calls first.
    """
    q_lens, kv_lens, heads = case_geometry(case_id, custom)
    helpers = impl_helpers(impl)
    plan = helpers.plan_launch(len(q_lens), sum(kv_lens), heads)
    return len(plan.launches) + 3


def iteration_count(case_id, custom=""):
    """Mixed-shape iteration count of one case: sum over (request, key block).

    ``iters = sum_b sum_kn (ceil(len_q/64) - loop_st)`` with
    ``loop_st = max(0, (96*kn - shift)//64)`` -- the DESIGN.md 1.6.0 R3 formula,
    recomputed here so a per-iteration price can be read off ``T_op``.
    """
    q_lens, kv_lens, heads = case_geometry(case_id, custom)
    total = 0
    for len_q, len_kv in zip(q_lens, kv_lens):
        shift = len_kv - len_q
        for kn in range(-(-len_kv // 96)):
            loop_st = max(0, (96 * kn - shift) // 64)
            loop_ed = max(-(-len_q // 64), loop_st + 1)
            total += max(0, loop_ed - loop_st)
    return total * heads


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
    parser.add_argument("--launch-count", type=int, default=10)
    parser.add_argument("--warm-up", type=int, default=5)
    parser.add_argument("--metrics", default="roofline", choices=["roofline", "full"])
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
        metrics = "Roofline" if args.metrics == "roofline" else FULL_METRICS
        per_call = launches_per_call(args.case, args.custom, args.impl)
        # One profiled process issues **exactly one** operator call, so the
        # capture window is its whole launch set regardless of the head-slab
        # count: the compile cache is warmed first by an unprofiled run of the
        # same case (``--no-setup`` skips the untimed setup call inside the
        # profiled process).  Deriving the window from a *warm-up multiple* of a
        # guessed launch count was wrong the moment a candidate changed the slab
        # (a 19-launch call measured as 26 samples).
        warm_env = dict(os.environ)
        warm_env["PYTHONPATH"] = f"{TILEOPS_ROOT}:{warm_env.get('PYTHONPATH', '')}"
        subprocess.run(
            [sys.executable, str(HERE / "mla_bwd_case.py"), "--impl", args.impl,
             "--inputs", str(inputs), "--case", args.case, "--no-setup"],
            capture_output=True, text=True, env=warm_env, timeout=3600,
        )
        prof_warm = 0
        prof_capture = per_call
        cmd = [
            "msopprof",
            f"--output={out}",
            f"--launch-count={prof_capture}",
            f"--warm-up={prof_warm}",
            f"--aic-metrics={metrics}",
            sys.executable,
            str(HERE / "mla_bwd_case.py"),
            "--impl", args.impl,
            "--inputs", str(inputs),
            "--case", args.case,
            "--warm-up", "0",
            "--launch-count", "1",
            "--no-setup",
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
    # every captured sample (the production report's ``t_op_us`` is the same
    # sum: ``_bwd_main_kernel`` contributes N slab launches, not one).
    payload["task_duration_us"] = round(sum(value for _, value in rows), 3)
    payload["per_call_launches"] = launches_per_call(args.case, args.custom, args.impl)
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
