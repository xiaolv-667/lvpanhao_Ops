"""Write perf_opt/perf_records.jsonl (append-only) from the capture JSONs.

One line per (candidate, workload) platform measurement.  Values are read from
the collector's JSON so nothing is retyped: ``task_duration_us`` is the call's
summed kernel Task Duration (the production report's ``t_op_us``).
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASK = HERE.parent
KERNEL_ID = ("tileops/kernels/attention/mla_varlen_bwd/"
             "mla_varlen_bwd_kernel::_bwd_kernel+_delta_kernel+_dq_land_kernel")
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

# (round, candidate_id, parent_id, artifact, capture_json, phase, l0_pass, note)
ROWS = [
    (0, "baseline-stage3", None, "MLAVarlenBwdOp_baseline_stage3.py",
     "profiles/round0/ds-v3-8x4k-bfloat16_baseline.json", "baseline", True,
     "frozen Stage 3 cube design (byte copy of the integrated _bwd_kernel.py, md5 4262fe4b); l0_pass carries the Stage 3 gate (integration_log.md: 12/12) plus this round's parity reference run (verify_parity.py loads this module as the reference)"),
    (0, "baseline-stage3", None, "MLAVarlenBwdOp_baseline_stage3.py",
     "profiles/round0/ds-v3-8x4k-float16_baseline.json", "baseline", True, ""),
    (0, "baseline-stage3", None, "MLAVarlenBwdOp_baseline_stage3.py",
     "profiles/round0/ds-v3-mixed-bfloat16_baseline.json", "baseline", True, ""),
    (0, "baseline-stage3", None, "MLAVarlenBwdOp_baseline_stage3.py",
     "profiles/round0/kimi-k2-4x8k-bfloat16_baseline.json", "baseline", True, ""),
    (1, "r1-c1-nt-operands", "baseline-stage3", "MLAVarlenBwdOp_round1_c1.py",
     "profiles/round1/ds-v3-8x4k-bfloat16_c1.json", "candidate", True,
     "drop qT/doT/kT: dV/dK/dQ take the natural (K,N) operand; dk/dv bitwise vs baseline"),
    (1, "r1-c1-nt-operands", "baseline-stage3", "MLAVarlenBwdOp_round1_c1.py",
     "profiles/round1/ds-v3-mixed-bfloat16_c1.json", "candidate", True, ""),
    (1, "r1-c3-stages3", "baseline-stage3", "MLAVarlenBwdOp_round1_c3.py",
     "profiles/round1/ds-v3-8x4k-bfloat16_c3.json", "candidate", True,
     "query-loop num_stages 2 -> 3; dk/dv bitwise vs baseline"),
    (1, "r1-c3-stages3", "baseline-stage3", "MLAVarlenBwdOp_round1_c3.py",
     "profiles/round1/ds-v3-mixed-bfloat16_c3.json", "candidate", True, ""),
    (1, "r1-c3-stages3", "baseline-stage3", "MLAVarlenBwdOp_round1_c3.py",
     "profiles/round2/kimi-k2-4x8k-bfloat16_c3.json", "candidate", True, ""),
    (1, "r1-c3-stages3", "baseline-stage3", "MLAVarlenBwdOp_round1_c3.py",
     "profiles/round2/ds-v3-8x4k-float16_c3.json", "candidate", True, ""),
    (1, "r1-c4-stages4", "baseline-stage3", "MLAVarlenBwdOp_round1_c4.py",
     "profiles/round2/ds-v3-8x4k-bfloat16_c4.json", "candidate", True,
     "num_stages 2 -> 4"),
    (1, "r1-c4-stages4", "baseline-stage3", "MLAVarlenBwdOp_round1_c4.py",
     "profiles/round2/ds-v3-mixed-bfloat16_c4.json", "candidate", True, ""),
    (1, "r1-c5-stages3-nt", "baseline-stage3", "MLAVarlenBwdOp_round1_c5.py",
     "profiles/round2/ds-v3-8x4k-bfloat16_c5.json", "candidate", True,
     "c1 + c3 combined"),
    (1, "r1-c5-stages3-nt", "baseline-stage3", "MLAVarlenBwdOp_round1_c5.py",
     "profiles/round2/ds-v3-mixed-bfloat16_c5.json", "candidate", True, ""),
    (1, "r1-c6-stages4-nt", "baseline-stage3", "MLAVarlenBwdOp_round1_c6.py",
     "profiles/round2/ds-v3-8x4k-bfloat16_c6.json", "candidate", True,
     "c1 + num_stages 4"),
    (1, "r1-c6-stages4-nt", "baseline-stage3", "MLAVarlenBwdOp_round1_c6.py",
     "profiles/round2/ds-v3-mixed-bfloat16_c6.json", "candidate", True, ""),
    (1, "r1-c7-stages5-nt", "baseline-stage3", "MLAVarlenBwdOp_round1_c7.py",
     "profiles/round2/ds-v3-8x4k-bfloat16_c7.json", "candidate", True,
     "c1 + num_stages 5"),
    (1, "r1-c7-stages5-nt", "baseline-stage3", "MLAVarlenBwdOp_round1_c7.py",
     "profiles/round2/ds-v3-mixed-bfloat16_c7.json", "candidate", True, ""),
    (2, "final-c3-stages3", "r1-c3-stages3", "MLAVarlenBwdOp.py",
     "profiles/final/ds-v3-8x4k-bfloat16_final.json", "final", True,
     "winner merged = r1-c3-stages3 (identical bytes), re-measured on the final file"),
    (2, "final-c3-stages3", "r1-c3-stages3", "MLAVarlenBwdOp.py",
     "profiles/final/ds-v3-8x4k-float16_final.json", "final", True, ""),
    (2, "final-c3-stages3", "r1-c3-stages3", "MLAVarlenBwdOp.py",
     "profiles/final/ds-v3-mixed-bfloat16_final.json", "final", True, ""),
    (2, "final-c3-stages3", "r1-c3-stages3", "MLAVarlenBwdOp.py",
     "profiles/final/kimi-k2-4x8k-bfloat16_final.json", "final", True, ""),
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out = HERE / "perf_records.jsonl"
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = []
    for rnd, cand, parent, artifact, capture, phase, l0, note in ROWS:
        payload = json.loads((HERE / capture).read_text())
        artifact_path = HERE / artifact
        rows = [
            {
                "round": rnd, "candidate_id": cand, "parent_id": parent,
                "kernel_id": KERNEL_ID,
                "workload_id": payload["case"],
                "phase": phase,
                "artifact_path": artifact,
                "artifact_sha256": sha256(artifact_path),
                "duration_us": payload["task_duration_us"],
                "l0_pass": l0,
                "per_op_us": {name: entry["sum_us"]
                              for name, entry in payload["ops"].items()},
                "main_us": payload["main_us"],
                "iterations": payload["iterations"],
                "us_per_iteration": payload["us_per_iteration"],
                "captured_samples": payload["captured_samples"],
                "launch_count_per_call": payload["per_call_launches"],
                "pipe": payload["pipe"],
                "implementation": payload["impl"],
                "t_op_from": ("sum of every captured launch of one call: "
                              "_delta_main + N x _bwd_main + _dq_land_main + ZerosLike"),
                "msprof_raw_path": capture.replace(".json", ""),
                "measurement": "msopprof --aic-metrics=full, Task Duration(us), "
                               "one call per capture window",
                "note": note,
                "timestamp": stamp,
                "hardware_id": "ascend-950pr-9579",
                "toolchain_id": "ascend",
                "platform_config": PLATFORM_CONFIG,
                "platform_runtime": PLATFORM_RUNTIME,
            }
        ]
        lines.extend(json.dumps(row, sort_keys=False) for row in rows)
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} records -> {out}")


if __name__ == "__main__":
    main()
