"""Append measured records to ``perf_opt/perf_records.jsonl`` (append-only).

One row per (candidate, workload) measurement.  ``duration_us`` is ``T_op``: the
sum of every kernel launch of one operator call, read from the msopprof capture
that ``capture.py`` produced for that ``--tag``/``--workload``.  The per-kernel
breakdown stays in the capture ``summary.json`` and in ``opt_log.md``.

Rows are never rewritten: this script only ever opens the file in append mode.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

PERF_OPT = Path(__file__).resolve().parent.parent
TASK_DIR = PERF_OPT.parent
STATE = json.loads((TASK_DIR / ".stage_state.json").read_text())
RECORDS = PERF_OPT / "perf_records.jsonl"
WORKLOADS = [
    "kimi-k2-t1",
    "kimi-k2-t512",
    "ds-v3-t32-bf16",
    "ds-v3-t32-fp16",
    "ds-v3-t4096",
    "glm45-t32",
]
KERNEL_ID = "shared_expert_mlp"  # the op, decomposed into its child kernels


def sha256(path):
    with open(path, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["baseline", "candidate", "merged", "final"])
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--candidate-id", required=True)
    ap.add_argument("--parent-id", default="")
    ap.add_argument("--tag", required=True, help="capture tag holding the summaries")
    ap.add_argument("--artifact", required=True, help="measured kernel/impl file")
    ap.add_argument("--l0-pass", default="true", choices=["true", "false"])
    ap.add_argument("--workloads", default=" ".join(WORKLOADS))
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    artifact = Path(args.artifact).resolve()
    digest = sha256(artifact)
    rows = []
    for workload in args.workloads.split():
        summary_path = PERF_OPT / "profiles" / args.tag / workload / "summary.json"
        if not summary_path.exists():
            raise SystemExit(f"missing capture summary: {summary_path}")
        summary = json.loads(summary_path.read_text())
        rows.append(
            {
                "round": args.round,
                "candidate_id": args.candidate_id,
                "parent_id": args.parent_id,
                "kernel_id": KERNEL_ID,
                "workload_id": workload,
                "phase": args.phase,
                "artifact_path": str(artifact),
                "artifact_sha256": digest,
                "duration_us": round(summary["t_op_us"], 3),
                "l0_pass": args.l0_pass == "true",
                "kernel_breakdown_us": {
                    entry["op_name"] + f"#{entry['pos']}": round(entry["median_us"], 3)
                    for entry in summary["launches"]
                },
                "t_op_runs_us": [round(v, 3) for v in summary.get("t_op_all_us", [])],
                "msprof_raw_path": str(
                    PERF_OPT / "profiles" / args.tag / workload / "raw_run0"
                ),
                "measurement": "msopprof Task Duration(us), 1 operator call per capture, "
                               "repeats median; T_op = sum of every launch",
                "note": args.note,
                "timestamp": __import__("datetime").datetime.utcnow().isoformat() + "Z",
                "hardware_id": STATE["hardware_id"],
                "toolchain_id": STATE["toolchain_id"],
                "platform_config": STATE["platform_config"],
                "platform_runtime": STATE["platform_runtime"],
            }
        )
    with RECORDS.open("a") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    for row in rows:
        print(f"+ {row['phase']}/r{row['round']} {row['candidate_id']} "
              f"{row['workload_id']}: {row['duration_us']} us")
    print(f"appended {len(rows)} rows to {RECORDS}")
    _ = os, sys


if __name__ == "__main__":
    main()
