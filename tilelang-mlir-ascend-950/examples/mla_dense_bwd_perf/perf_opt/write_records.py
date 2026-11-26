"""Append one measurement row per (candidate, workload) to perf_records.jsonl.

Append-only: existing lines are never rewritten or deleted.  Values are read
from the collector's JSON so nothing is retyped; ``duration_us`` is the call's
summed kernel Task Duration (the production report's ``t_op_us``).

    python perf_opt/write_records.py --round 0 --candidate-id baseline-stage3 \
        --artifact perf_opt/MLADenseBwdOp_baseline_stage3.py --phase baseline \
        --l0-pass true --capture profiles/round0/ds-v3-b2-8k-bfloat16_baseline.json \
        [--note "..."] [--parent-id ...]
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
KERNEL_ID = ("tileops/kernels/attention/mla_dense_bwd/"
             "mla_dense_bwd_kernel::_bwd_kernel+_delta_kernel+_dq_land_kernel")
PLATFORM_CONFIG = {
    "hardware": "0f2c8d59d2fce13995a3266d66e34d18dc2b602d7533779b6455af426b727f1c",
    "toolchain": "02257b6fc82378310c1d827b0cb2ecddac03b18f2fdcd3c19072baf29cae0078",
}
PLATFORM_RUNTIME = {
    "device": {"cores": {"cube": 28, "vector": 56}, "name": "Ascend950PR_9579"},
    "toolchain_stamp": {
        "cann": "CANN9.2.1 /usr/local/Ascend/cann", "compiler": "12.3.0",
        "raw": "7a52f2f1274df07d8f4c9c01353ce9d290f88c16e85047de0c425b6a009de4aa",
        "runtime_opp": "CANN9.2.1 /usr/local/Ascend/cann/lib64",
        "tilelang": "0.1.15", "tilelang_commit": None,
    },
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--parent-id", default=None)
    parser.add_argument("--artifact", required=True, help="path of the measured kernel file")
    parser.add_argument("--phase", required=True,
                        choices=["baseline", "candidate", "merged", "final"])
    parser.add_argument("--l0-pass", required=True)
    parser.add_argument("--capture", required=True, help="collector JSON (relative to perf_opt/)")
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    payload = json.loads((HERE / args.capture).read_text())
    artifact = Path(args.artifact)
    if not artifact.is_absolute():
        artifact = (HERE.parent / artifact).resolve()
    rel_artifact = str(artifact.relative_to(HERE.parent))
    row = {
        "round": args.round,
        "candidate_id": args.candidate_id,
        "parent_id": args.parent_id,
        "kernel_id": KERNEL_ID,
        "workload_id": payload["case"],
        "phase": args.phase,
        "artifact_path": rel_artifact,
        "artifact_sha256": sha256(artifact),
        "duration_us": payload["task_duration_us"],
        "l0_pass": args.l0_pass.lower() == "true",
        "per_op_us": {name: entry["sum_us"] for name, entry in payload["ops"].items()},
        "main_us": payload["main_us"],
        "iterations": payload["iterations"],
        "us_per_iteration": payload["us_per_iteration"],
        "captured_samples": payload["captured_samples"],
        "launch_count_per_call": payload["per_call_launches"],
        "pipe": payload["pipe"],
        "implementation": payload["impl"],
        "t_op_from": ("sum of every captured launch of one call: Range x2 + ZerosLike "
                      "+ _delta_main + _bwd_main + _dq_land_main"),
        "msprof_raw_path": payload["raw_profile_dir"],
        "measurement": ("msopprof --aic-metrics=PipeUtilization, Task Duration(us), "
                        "one call per capture window"),
        "note": args.note,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hardware_id": "ascend-950pr-9579",
        "toolchain_id": "ascend",
        "platform_config": PLATFORM_CONFIG,
        "platform_runtime": PLATFORM_RUNTIME,
    }
    out = HERE / "perf_records.jsonl"
    with out.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=False) + "\n")
    print(f"appended {row['candidate_id']} / {row['workload_id']} "
          f"T_op={row['duration_us']} us -> {out}")


if __name__ == "__main__":
    main()
