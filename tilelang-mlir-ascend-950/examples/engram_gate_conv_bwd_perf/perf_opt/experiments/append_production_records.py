"""Append the production-adoption rows to perf_opt/perf_records.jsonl (append-only).

Rows: ``phase=merged`` / ``candidate_id=production-wrapped``, measured through the
flipped wrapper by the single-operator report chain
(``tileops.reporting.cli run --op EngramGateConvBwdOp --prof-mode msprof``,
run-id ``20261124_231932_732777_EngramGateConvBwdOp``).

``duration_us`` is the report's ``msprof.t_op_us`` (the platform T_op: sum of the
two kernel launches of one operator call).  ``samples_us`` holds the two
per-launch durations of that one capture (launch #2, launch #1), i.e. the
components of the recorded sum -- not two independent T_op samples.

    python append_production_records.py
"""

import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))   # perf_opt/experiments
OPT = os.path.dirname(HERE)                          # perf_opt
TASK = os.path.dirname(OPT)                          # examples/engram_gate_conv_bwd_perf

KERNEL_ID = (
    "tileops/kernels/sequence_modeling/engram_gate_conv_bwd/engram_gate_conv_bwd_kernel"
    "::_engram_gate_conv_bwd_kernel::main"
)
DEPLOYED = (
    "../../TileOPs/tileops/kernels/sequence_modeling/engram_gate_conv_bwd/"
    "engram_gate_conv_bwd_kernel/perf_opt/_engram_gate_conv_bwd_kernel.py"
)
RUN_ID = "20261124_231932_732777_EngramGateConvBwdOp"
TS = "2026-11-24T23:22:00Z"

state = json.load(open(os.path.join(TASK, ".stage_state.json")))
EV = {
    "hardware_id": state["hardware_id"],
    "toolchain_id": state["toolchain_id"],
    "platform_config": state["platform_config"],
    "platform_runtime": state["platform_runtime"],
}

# workload -> (t_op_us, [launch2_us, launch1_us])  from the report's run.json
ROWS = [
    ("train-s96-bfloat16", 12.393, [2.590, 9.803]),
    ("train-s64-float16", 8.809, [2.716, 6.093]),
    ("train-s128-bfloat16", 7.139, [2.266, 4.873]),
]


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    path = os.path.join(OPT, "perf_records.jsonl")
    digest = sha256(os.path.join(OPT, DEPLOYED))
    out = []
    for wl, t_op, launches in ROWS:
        rec = {
            "round": 5,
            "candidate_id": "production-wrapped",
            "parent_id": "final-chunkmajor",
            "kernel_id": KERNEL_ID,
            "workload_id": wl,
            "phase": "merged",
            "artifact_path": DEPLOYED,
            "artifact_sha256": digest,
            "duration_us": t_op,
            "l0_pass": True,
            "msprof_raw_path": (
                f"../../TileOPs/reports/tileops/{RUN_ID}/artifacts/msprof/"
            ),
            "samples_us": launches,
            "launch_count": 2,
            "timestamp": TS,
            "note": (
                "wrapped (wrapper perf_opt switch ACTIVE) via the single-operator report "
                f"chain, run-id {RUN_ID}; duration_us = msprof.t_op_us (sum of both "
                "launches), samples_us = the two per-launch durations of that capture"
            ),
        }
        rec.update(EV)
        out.append(rec)
    with open(path, "a", encoding="utf-8") as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"appended {len(out)} rows to {path} (deployed sha256 {digest[:12]}…)")
    for rec in out:
        print(f"  {rec['workload_id']:<20} T_op={rec['duration_us']:>7.3f} us "
              f"launches={rec['samples_us']}")


if __name__ == "__main__":
    main()
