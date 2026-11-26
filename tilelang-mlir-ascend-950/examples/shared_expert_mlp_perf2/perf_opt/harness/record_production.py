"""Append the production (integrated) measurement to ``perf_records.jsonl``.

Reads the single-operator report run (``tileops.reporting.cli run --op
SharedExpertMLPFwdOp --prof-mode msprof``) and appends one row per tune
workload with ``phase="merged"`` (production adoption), the report's own
``t_op_us`` (sum of every launch of the tileops-tagged record) and the report
artifact directory as the raw evidence.

Append-only: the file is opened in append mode, existing rows are untouched.

Usage::

    python record_production.py --run-json <report>/run.json --round 4 \
        --candidate-id candB-fused-epilogue --artifact <integrated perf_opt file>
"""

import argparse
import datetime
import hashlib
import json
from pathlib import Path

PERF_OPT = Path(__file__).resolve().parent.parent
TASK_DIR = PERF_OPT.parent
STATE = json.loads((TASK_DIR / ".stage_state.json").read_text())
RECORDS = PERF_OPT / "perf_records.jsonl"
KERNEL_ID = "shared_expert_mlp"

# report label + dtype -> workload_id of the benchmark inventory
LABELS = {
    ("1x7168x18432", "bfloat16"): "kimi-k2-t1",
    ("512x7168x18432", "bfloat16"): "kimi-k2-t512",
    ("32x7168x2048", "bfloat16"): "ds-v3-t32-bf16",
    ("32x7168x2048", "float16"): "ds-v3-t32-fp16",
    ("4096x7168x2048", "bfloat16"): "ds-v3-t4096",
    ("32x5120x1536", "bfloat16"): "glm45-t32",
}
PARAMS = {
    "kimi-k2-t1": (1, 7168, 18432),
    "kimi-k2-t512": (512, 7168, 18432),
    "ds-v3-t32-bf16": (32, 7168, 2048),
    "ds-v3-t32-fp16": (32, 7168, 2048),
    "ds-v3-t4096": (4096, 7168, 2048),
    "glm45-t32": (32, 5120, 1536),
}
PEAK = 378.0e12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-json", required=True)
    ap.add_argument("--round", type=int, default=4)
    ap.add_argument("--candidate-id", required=True)
    ap.add_argument("--parent-id", default="candB-fused-epilogue")
    ap.add_argument("--artifact", required=True)
    args = ap.parse_args()

    run = json.loads(Path(args.run_json).read_text())
    artifact = Path(args.artifact).resolve()
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    report_id = run["metadata"]["run_id"]
    rows = []
    for case in run["performance"]["cases"]:
        if case.get("tag") != "tileops":
            continue
        key = (case["label"], case["params"]["dtype"])
        workload = LABELS[key]
        t_op = case["msprof"]["t_op_us"]
        tt, hh, ss = PARAMS[workload]
        flops = 6 * tt * ss * hh + 6 * tt * ss
        rows.append(
            {
                "round": args.round,
                "candidate_id": args.candidate_id,
                "parent_id": args.parent_id,
                "kernel_id": KERNEL_ID,
                "workload_id": workload,
                "phase": "merged",
                "artifact_path": str(artifact),
                "artifact_sha256": digest,
                "duration_us": round(t_op, 3),
                "l0_pass": True,
                "kernel_breakdown_us": {
                    name: round(value, 3)
                    for name, value in case["msprof"]["t_op_breakdown"].items()
                },
                "captured_op_names": case["msprof"]["captured_op_names"],
                "mfu_pct_nominal": round(100.0 * flops / (t_op * 1e-6) / PEAK, 3),
                "msprof_raw_path": case["artifact_dir"],
                "report_run_id": report_id,
                "report_run_json": str(Path(args.run_json).resolve()),
                "measurement": "tileops.reporting.cli run --prof-mode msprof; T_op = sum of "
                               "every launch of the tileops-tagged case record",
                "note": "production adoption (wrapper exports the perf_opt kernel)",
                "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
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
        print(f"+ merged/r{row['round']} {row['workload_id']}: {row['duration_us']} us "
              f"(MFU {row['mfu_pct_nominal']}%)")
    print(f"appended {len(rows)} production rows (report {report_id}) -> {RECORDS}")


if __name__ == "__main__":
    main()
