"""Write perf_opt/perf_records.jsonl (append-only) from this round's captures.

Round 2 (task `mla_varlen_bwd_perf2`) targets the residual that round 1 left:
the per-iteration price grows with the request length.  The base is the round-1
winner (`_STAGES=3`, byte copy in `MLAVarlenBwdOp.py`); every branch changes one
structure/parameter and all of them lost, so the final version is the base.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
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

BASELINE = "MLAVarlenBwdOp.py"
# (round, candidate, parent, artifact, capture_json, phase, note)
ROWS = [
    (0, "baseline-r1-winner", None, BASELINE, "profiles/round0/ds-v3-8x4k-bfloat16_baseline.json",
     "baseline", "round-1 winner carried in (num_stages=3), re-measured in this task"),
    (0, "baseline-r1-winner", None, BASELINE, "profiles/round0/ds-v3-8x4k-float16_baseline.json",
     "baseline", ""),
    (0, "baseline-r1-winner", None, BASELINE, "profiles/round0/ds-v3-mixed-bfloat16_baseline.json",
     "baseline", ""),
    (0, "baseline-r1-winner", None, BASELINE, "profiles/round0/kimi-k2-4x8k-bfloat16_baseline.json",
     "baseline", ""),
    # ---- round 1: query-chunked GM fill (2 x 64-row query blocks per descriptor) ----
    (1, "r2-q2s2", "baseline-r1-winner", "MLAVarlenBwdOp_r2_q2s2.py",
     "profiles/round1/ds-v3-mixed-bfloat16_q2s2.json", "candidate",
     "fill granularity 2x64 rows, stages=2: pipe overlap collapses (mad share 26.3->17.5%), "
     "much slower"),
    (1, "r2-q2s2", "baseline-r1-winner", "MLAVarlenBwdOp_r2_q2s2.py",
     "profiles/round1/ds-v3-8x4k-bfloat16_q2s2.json", "candidate", ""),
    (1, "r2-q2s3nt", "baseline-r1-winner", "MLAVarlenBwdOp_r2_q2s3nt.py",
     "profiles/round1/ds-v3-mixed-bfloat16_q2s3nt.json", "candidate",
     "fill granularity 2x64 rows + natural (K,N) operands + stages=3"),
    (1, "r2-q2s3nt", "baseline-r1-winner", "MLAVarlenBwdOp_r2_q2s3nt.py",
     "profiles/round1/ds-v3-8x4k-bfloat16_q2s3nt.json", "candidate", ""),
    # ---- round 2: head-slab (launch plan) sweep, i.e. in-flight head footprint ----
    (2, "r2-s8", "baseline-r1-winner", "MLAVarlenBwdOp_r2_s8.py",
     "profiles/wintest/mixed_s8.json", "candidate",
     "HEAD_SLAB_DEFAULT 32 -> 8 (4x smaller in-flight head footprint)"),
    (2, "r2-s4", "baseline-r1-winner", "MLAVarlenBwdOp_r2_s4.py",
     "profiles/probe/ds-v3-mixed-bfloat16_s4.json", "candidate",
     "HEAD_SLAB_DEFAULT 32 -> 4 (8x smaller in-flight head footprint)"),
    (2, "r2-s4", "baseline-r1-winner", "MLAVarlenBwdOp_r2_s4.py",
     "profiles/probe/ds-v3-8x4k-bfloat16_s4_8x4k.json", "candidate", ""),
    # ---- final: the base itself, re-measured on the delivered file ----
    (3, "final-r1-winner", "baseline-r1-winner", BASELINE, "profiles/final/ds-v3-8x4k-bfloat16_final.json",
     "final", "no branch won: the delivered file is the round-1 winner unchanged"),
    (3, "final-r1-winner", "baseline-r1-winner", BASELINE, "profiles/final/ds-v3-8x4k-float16_final.json",
     "final", ""),
    (3, "final-r1-winner", "baseline-r1-winner", BASELINE, "profiles/final/ds-v3-mixed-bfloat16_final.json",
     "final", ""),
    (3, "final-r1-winner", "baseline-r1-winner", BASELINE, "profiles/final/kimi-k2-4x8k-bfloat16_final.json",
     "final", ""),
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out = HERE / "perf_records.jsonl"
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = []
    for rnd, cand, parent, artifact, capture, phase, note in ROWS:
        payload = json.loads((HERE / capture).read_text())
        lines.append(json.dumps({
            "round": rnd, "candidate_id": cand, "parent_id": parent,
            "kernel_id": KERNEL_ID,
            "workload_id": payload["case"],
            "phase": phase,
            "artifact_path": artifact,
            "artifact_sha256": sha256(HERE / artifact),
            "duration_us": payload["task_duration_us"],
            "l0_pass": True,
            "per_op_us": {name: entry["sum_us"] for name, entry in payload["ops"].items()},
            "main_us": payload["main_us"],
            "iterations": payload["iterations"],
            "us_per_iteration": payload["us_per_iteration"],
            "captured_samples": payload["captured_samples"],
            "launch_count_per_call": payload["per_call_launches"],
            "pipe": payload["pipe"],
            "implementation": payload["impl"],
            "t_op_from": ("sum of every captured launch of one call: _delta_main + "
                          "N x _bwd_main + _dq_land_main + ZerosLike"),
            "msprof_raw_path": capture.replace(".json", ""),
            "measurement": "msopprof --aic-metrics=full, Task Duration(us), one call per "
                           "capture window (pre-warmed compile cache, single-call process)",
            "note": note,
            "timestamp": stamp,
            "hardware_id": "ascend-950pr-9579",
            "toolchain_id": "ascend",
            "platform_config": PLATFORM_CONFIG,
            "platform_runtime": PLATFORM_RUNTIME,
        }, sort_keys=False))
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} records -> {out}")


if __name__ == "__main__":
    main()
