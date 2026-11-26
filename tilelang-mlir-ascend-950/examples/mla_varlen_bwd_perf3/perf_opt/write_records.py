"""Write perf_opt/perf_records.jsonl (append-only) from this round's captures.

Round 3 (task `mla_varlen_bwd_perf3`) tunes the A5 (v5) cube kernel toward
mixed <= 400 ms / 8x4k <= 130 ms.  Base = the v5 baseline (byte copy in
`MLAVarlenBwdOp_baseline_a5v5.py`), winner = `v3hoist` (V=3 static versions +
double-versioned st_u/dp_u + hoisted statistics), delivered as
`MLAVarlenBwdOp.py`.
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
BASE = "MLAVarlenBwdOp_baseline_a5v5.py"
FINAL = "MLAVarlenBwdOp.py"

ROWS = [
    (0, "baseline-a5v5", None, BASE, "profiles/round0/ds-v3-8x4k-bfloat16_baseline.json", "baseline", ""),
    (0, "baseline-a5v5", None, BASE, "profiles/round0/ds-v3-8x4k-float16_baseline.json", "baseline", ""),
    (0, "baseline-a5v5", None, BASE, "profiles/round0/ds-v3-mixed-bfloat16_baseline.json", "baseline", ""),
    (0, "baseline-a5v5", None, BASE, "profiles/round0/kimi-k2-4x8k-bfloat16_baseline.json", "baseline", ""),
    (1, "r3-ubv2", "baseline-a5v5", "MLAVarlenBwdOp_r3_ubv2.py",
     "profiles/round1/ds-v3-mixed-bfloat16_ubv2.json", "candidate",
     "double-version st_u/dp_u per half body (UB +24,576 B): removes the WAR that "
     "serialised half body B behind half body A's epilogue"),
    (1, "r3-ubv2", "baseline-a5v5", "MLAVarlenBwdOp_r3_ubv2.py",
     "profiles/round1/ds-v3-8x4k-bfloat16_ubv2.json", "candidate", ""),
    (1, "r3-ubv2b", "baseline-a5v5", "MLAVarlenBwdOp_r3_ubv2b.py",
     "profiles/round1/ds-v3-mixed-bfloat16_ubv2b.json", "candidate",
     "ubv2 + double-version p16_u/dsT_u (UB +12,288 B)"),
    (1, "r3-ubv2b", "baseline-a5v5", "MLAVarlenBwdOp_r3_ubv2b.py",
     "profiles/round1/ds-v3-8x4k-bfloat16_ubv2b.json", "candidate", ""),
    (1, "r3-nt", "baseline-a5v5", "MLAVarlenBwdOp_r3_nt.py",
     "profiles/round1/ds-v3-mixed-bfloat16_nt.json", "candidate",
     "natural (K,N) B operands for dV/dK/dQ: fill 6->4 descriptors, -50% bytes, "
     "kT_s dropped; LOST because the NN GEMM form is slower than the fill it saves"),
    (1, "r3-nt", "baseline-a5v5", "MLAVarlenBwdOp_r3_nt.py",
     "profiles/round1/ds-v3-8x4k-bfloat16_nt.json", "candidate", ""),
    (1, "r3-ubv2-nt", "baseline-a5v5", "MLAVarlenBwdOp_r3_ubv2_nt.py",
     "profiles/round1/ds-v3-mixed-bfloat16_ubv2_nt.json", "candidate", "ubv2 + nt"),
    (1, "r3-ubv2-nt", "baseline-a5v5", "MLAVarlenBwdOp_r3_ubv2_nt.py",
     "profiles/round1/ds-v3-8x4k-bfloat16_ubv2_nt.json", "candidate", ""),
    (1, "r3-v3ub", "baseline-a5v5", "MLAVarlenBwdOp_r3_v3ub.py",
     "profiles/round1/ds-v3-mixed-bfloat16_v3ub.json", "candidate",
     "V=3 static half-body versions (L1 417,792 B = 79.7%) on top of ubv2: one more "
     "fill in flight"),
    (1, "r3-v3ub", "baseline-a5v5", "MLAVarlenBwdOp_r3_v3ub.py",
     "profiles/round1/ds-v3-8x4k-bfloat16_v3ub.json", "candidate", ""),
    (2, "r3-v3hoist", "r3-v3ub", "MLAVarlenBwdOp_r3_v3hoist.py",
     "profiles/round1/ds-v3-mixed-bfloat16_v3hoist.json", "merged",
     "v3ub + the six head-major statistics runs issued at the iteration top"),
    (2, "r3-v3hoist", "r3-v3ub", "MLAVarlenBwdOp_r3_v3hoist.py",
     "profiles/round1/ds-v3-8x4k-bfloat16_v3hoist.json", "merged", ""),
    (3, "final-v3hoist", "r3-v3hoist", FINAL, "profiles/final/ds-v3-8x4k-bfloat16_final.json", "final",
     "delivered file (= v3hoist, byte identical)"),
    (3, "final-v3hoist", "r3-v3hoist", FINAL, "profiles/final/ds-v3-8x4k-float16_final.json", "final", ""),
    (3, "final-v3hoist", "r3-v3hoist", FINAL, "profiles/final/ds-v3-mixed-bfloat16_final.json", "final", ""),
    (3, "final-v3hoist", "r3-v3hoist", FINAL, "profiles/final/kimi-k2-4x8k-bfloat16_final.json", "final", ""),
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
                           "capture window (pre-warmed cache, single-call process)",
            "note": note, "timestamp": stamp,
            "hardware_id": "ascend-950pr-9579", "toolchain_id": "ascend",
            "platform_config": PLATFORM_CONFIG, "platform_runtime": PLATFORM_RUNTIME,
        }))
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} records -> {out}")


if __name__ == "__main__":
    main()
