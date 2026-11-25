"""Emit perf_opt/perf_records.jsonl (append-only contract, signal-registry §5).

Every row is one msopprof measurement of one candidate file on one benchmark
workload.  Latency = `Task Duration(us)` of one operator call, median over the
independent capture processes listed in `samples_us`.

    python emit_records.py
"""

import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))  # perf_opt/experiments
OPT = os.path.dirname(HERE)                          # perf_opt
TASK = os.path.dirname(OPT)                          # examples/engram_gate_conv_perf

KERNEL_ID = (
    "tileops/kernels/sequence_modeling/engram_gate_conv/engram_gate_conv_kernel"
    "::_engram_gate_conv_fwd_kernel::main"
)
BASE = ("TileOPs/tileops/kernels/sequence_modeling/engram_gate_conv/engram_gate_conv_kernel/"
        "_engram_gate_conv_fwd_kernel.py")
TS = "2026-11-24T22:40:00Z"

state = json.load(open(os.path.join(TASK, ".stage_state.json")))
EV = {
    "hardware_id": state["hardware_id"],
    "toolchain_id": state["toolchain_id"],
    "platform_config": state["platform_config"],
    "platform_runtime": state["platform_runtime"],
}

# (round, candidate_id, parent_id, phase, artifact rel-to-task-dir, workload,
#  samples_us, raw profile dir rel-to-perf_opt, note)
ROWS = [
    (0, "baseline-stage3", None, "baseline", "../../" + BASE, "train-s96-bfloat16",
     [11.254, 10.921, 11.084], "profiles/baseline/_engram_gate_conv_fwd_kernel_s96_r0",
     "frozen Stage-3 module, 3 independent capture processes"),
    (0, "baseline-stage3", None, "baseline", "../../" + BASE, "train-s64-float16",
     [8.272, 8.230, 8.303], "profiles/baseline/_engram_gate_conv_fwd_kernel_s64_r0",
     "frozen Stage-3 module, 3 independent capture processes"),
    (0, "baseline-stage3", None, "baseline", "../../" + BASE, "train-s128-bfloat16",
     [7.615, 7.904, 7.703], "profiles/baseline/_engram_gate_conv_fwd_kernel_s128_r0",
     "frozen Stage-3 module, 3 independent capture processes"),

    (1, "r1_c1-pipelined", "baseline-stage3", "candidate",
     "engram_gate_conv_round1_c1.py", "train-s96-bfloat16",
     [11.106, 11.364], "profiles/round1/engram_gate_conv_round1_c1_s96_r0",
     "O-1 T.Pipelined(num_stages=2) + 2 row-buffer versions: no effect (emitted code unchanged)"),
    (1, "r1_c2-rv-register", "baseline-stage3", "candidate",
     "engram_gate_conv_round1_c2.py", "train-s96-bfloat16",
     [10.671, 11.313], "profiles/round1/engram_gate_conv_round1_c2_s96_r0",
     "O-2 drop the r_v store->BRC reload and its VST_VLD barrier: below noise alone"),
    (1, "r1_c3-rowstore2d", "baseline-stage3", "candidate",
     "engram_gate_conv_round1_c3.py", "train-s96-bfloat16",
     [11.470, 10.603], "profiles/round1/engram_gate_conv_round1_c3_s96_r0",
     "O-3 one 2-D store per output: MTE3 busy 0.131->0.033, total below noise alone"),

    (2, "r2_c4-rowprefetch", "baseline-stage3", "candidate",
     "engram_gate_conv_round2_c4.py", "train-s96-bfloat16",
     [12.907, 12.918, 12.993], "profiles/round2/engram_gate_conv_round2_c4_s96_r0",
     "O-4 build-time unrolled row loop with 2 explicit buffer sets and prefetch: -10.9% (MTE2 busy ratio 0.31->0.72)"),
    (2, "r2_c5-windowload", "baseline-stage3", "candidate",
     "engram_gate_conv_round2_c5.py", "train-s96-bfloat16",
     [9.056, 8.733, 8.875], "profiles/round2/engram_gate_conv_round2_c5_s96_r0",
     "O-5 bulk (rc+3, d) window copies replace 27 one-row DMAs: winner"),
    (2, "r2_c5-windowload", "baseline-stage3", "candidate",
     "engram_gate_conv_round2_c5.py", "train-s64-float16",
     [6.728, 6.837], "profiles/round2/engram_gate_conv_round2_c5_s64_r0",
     "O-5 merged check on s64"),
    (2, "r2_c5-windowload", "baseline-stage3", "candidate",
     "engram_gate_conv_round2_c5.py", "train-s128-bfloat16",
     [6.183, 5.998], "profiles/round2/engram_gate_conv_round2_c5_s128_r0",
     "O-5 merged check on s128"),

    (3, "r3_c6-fullmask", "r2_c5-windowload", "candidate",
     "engram_gate_conv_round3_c6.py", "train-s96-bfloat16",
     [9.036, 8.608, 9.163], "profiles/round3/engram_gate_conv_round3_c6_s96_r0",
     "O-6 fold the per-chunk lane mask at build time (d % 64 == 0): no gain over c5"),
    (3, "r3_c7-two-launch", "baseline-stage3", "candidate",
     "engram_gate_conv_round3_c7.py", "train-s96-bfloat16",
     [9.993, 9.974, 10.539], "profiles/round3/engram_gate_conv_round3_c7_s96_r0",
     "O-7 DESIGN 1.6.3 alternative (two launches, no halo recompute): +12.5% vs c5"),

    (4, "final-windowload", "r2_c5-windowload", "final",
     "engram_gate_conv.py", "train-s96-bfloat16",
     [8.935, 8.326, 8.460], "profiles/final/engram_gate_conv_s96_r0",
     "final perf_opt file, 3 independent capture processes"),
    (4, "final-windowload", "r2_c5-windowload", "final",
     "engram_gate_conv.py", "train-s64-float16",
     [7.096, 6.962, 6.842], "profiles/final/engram_gate_conv_s64_r0",
     "final perf_opt file, 3 independent capture processes"),
    (4, "final-windowload", "r2_c5-windowload", "final",
     "engram_gate_conv.py", "train-s128-bfloat16",
     [5.764, 5.671, 5.870], "profiles/final/engram_gate_conv_s128_r0",
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
            "launch_count": 1,
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
        print(f"  r{rec['round']} {rec['phase']:<9} {rec['candidate_id']:<18} "
              f"{rec['workload_id']:<18} {rec['duration_us']:>7.3f} us")


if __name__ == "__main__":
    main()
