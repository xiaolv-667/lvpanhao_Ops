"""Emit perf_opt/perf_records.jsonl for the perf2 round (append-only contract).

One row per msopprof measurement of one candidate file on one benchmark
workload.  ``duration_us`` is the platform T_op for this operator: the SUM of the
two `main_kernel` launches of one operator call (``--launch-count=2``, one
capture = one call), median over the independent capture processes in
``samples_us``.

    python emit_records.py
"""

import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))  # perf_opt/experiments
OPT = os.path.dirname(HERE)                         # perf_opt
TASK = os.path.dirname(OPT)                         # examples/engram_gate_conv_bwd_perf2

KERNEL_ID = (
    "tileops/kernels/sequence_modeling/engram_gate_conv_bwd/engram_gate_conv_bwd_kernel"
    "::_engram_gate_conv_bwd_kernel::main"
)
BASE = "../../engram_gate_conv_bwd_perf/perf_opt/engram_gate_conv_bwd.py"
TS = "2026-11-25T19:10:00Z"

state = json.load(open(os.path.join(TASK, ".stage_state.json")))
EV = {
    "hardware_id": state["hardware_id"],
    "toolchain_id": state["toolchain_id"],
    "platform_config": state["platform_config"],
    "platform_runtime": state["platform_runtime"],
}

# (round, candidate_id, parent_id, phase, artifact rel-to-perf_opt, workload,
#  samples_us, raw profile dir rel-to-perf_opt, note)
ROWS = [
    (0, "baseline-perf1-final", None, "baseline", BASE, "train-s96-bfloat16",
     [13.110, 12.994, 12.876],
     "profiles/experiments/engram_gate_conv_bwd_round0_best_s96_r0",
     "read-only baseline = the adopted Stage-4 file of examples/engram_gate_conv_bwd_perf "
     "(sha256 65c3133c...), 3 independent capture processes; launch1 10.04 + launch2 2.95 us"),
    (0, "baseline-perf1-final", None, "baseline", BASE, "train-s64-float16",
     [8.895, 8.990, 8.929],
     "profiles/experiments/engram_gate_conv_bwd_round0_best_s64_r0",
     "read-only baseline, 3 independent capture processes"),
    (0, "baseline-perf1-final", None, "baseline", BASE, "train-s128-bfloat16",
     [7.873, 7.300, 7.754],
     "profiles/experiments/engram_gate_conv_bwd_round0_best_s128_r0",
     "read-only baseline, 3 independent capture processes"),

    (1, "r1_n1-chunk-unroll", "baseline-perf1-final", "candidate",
     "engram_gate_conv_bwd_round1_n1.py", "train-s96-bfloat16",
     [31.284, 31.611, 30.860],
     "profiles/experiments/engram_gate_conv_bwd_round1_n1_s96_r0",
     "O-13: build-time unroll of the 12 chunk loops -> launch1 10.04 -> 28.2 us "
     "(instruction-cache thrash); REJECTED"),
    (1, "r1_n2-phase-split", "baseline-perf1-final", "candidate",
     "engram_gate_conv_bwd_round1_n2.py", "train-s96-bfloat16",
     [21.331, 21.096, 21.753],
     "profiles/experiments/engram_gate_conv_bwd_round1_n2_s96_r0",
     "O-13': three-phase split + fused chunk-major conv/dconv_w sweep with 12 live "
     "registers -> aiv_vec_time 6.17 -> 14.83 us (register spilling); REJECTED"),
    (1, "r1_n3-reduce-ilp", "baseline-perf1-final", "candidate",
     "engram_gate_conv_bwd_round1_n3.py", "train-s96-bfloat16",
     [13.200, 12.393, 12.813],
     "profiles/experiments/engram_gate_conv_bwd_round1_n3_s96_r0",
     "O-15: launch-2 row loop hoisted out of the column-chunk loop (nch independent "
     "chains) -> launch2 2.95 -> 2.70 us, its vector chain 0.150 -> 0.085 us, "
     "block 1.368 -> 1.258 us; ACCEPTED (bit-identical)"),
    (1, "r1_n4-fused-forms", "baseline-perf1-final", "candidate",
     "engram_gate_conv_bwd_round1_n4.py", "train-s96-bfloat16",
     [13.102, 13.132, 13.294],
     "profiles/experiments/engram_gate_conv_bwd_round1_n4_s96_r0",
     "O-14: vexpdif(zero,conv) instead of vneg+vexp and the conv chain started at tap 0 "
     "(-324 vector instructions) -> no measurable change (the kernel is not "
     "issue-slot-bound); REJECTED as a separate branch, superseded"),

    (2, "final-reduce-ilp", "r1_n3-reduce-ilp", "final",
     "EngramGateConvBwdOp.py", "train-s96-bfloat16",
     [12.972, 13.074, 12.530],
     "profiles/experiments/engram_gate_conv_bwd_s96_r0",
     "final perf_opt file (== r1_n3 bytes), 3 independent capture processes"),
    (2, "final-reduce-ilp", "r1_n3-reduce-ilp", "final",
     "EngramGateConvBwdOp.py", "train-s64-float16",
     [8.541, 9.069, 9.028],
     "profiles/experiments/engram_gate_conv_bwd_s64_r0",
     "final perf_opt file, 3 independent capture processes"),
    (2, "final-reduce-ilp", "r1_n3-reduce-ilp", "final",
     "EngramGateConvBwdOp.py", "train-s128-bfloat16",
     [7.474, 7.218, 7.194],
     "profiles/experiments/engram_gate_conv_bwd_s128_r0",
     "final perf_opt file, 3 independent capture processes"),
]


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def main():
    out = []
    for rnd, cid, pid, phase, art, wl, samples, raw, note in ROWS:
        path = os.path.join(OPT, art)
        rec = {
            "round": rnd,
            "candidate_id": cid,
            "parent_id": pid,
            "kernel_id": KERNEL_ID,
            "workload_id": wl,
            "phase": phase,
            "artifact_path": art,
            "artifact_sha256": sha256(path),
            "duration_us": round(median(samples), 3),
            "l0_pass": True,
            "msprof_raw_path": raw,
            "samples_us": samples,
            "launch_count": 2,
            "timestamp": TS,
            "note": note,
        }
        rec.update(EV)
        out.append(rec)
    path = os.path.join(OPT, "perf_records.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {path}: {len(out)} rows")
    for rec in out:
        print(f"  r{rec['round']} {rec['phase']:<9} {rec['candidate_id']:<22} "
              f"{rec['workload_id']:<18} {rec['duration_us']:>7.3f} us")


if __name__ == "__main__":
    main()
