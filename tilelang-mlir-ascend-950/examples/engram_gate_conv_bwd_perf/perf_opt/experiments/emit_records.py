"""Emit perf_opt/perf_records.jsonl (append-only contract, signal-registry §5).

One row per msopprof measurement of one candidate file on one benchmark workload.
``duration_us`` is the platform `T_op` for this operator: the SUM of the two
`main_kernel` launches of one operator call (`--launch-count=2`, one capture =
one call), median over the independent capture processes listed in `samples_us`
(even counts: mean of the two middle samples).

    python emit_records.py            # writes the file (first generation)
    python emit_records.py --append   # append further rows, never rewrite

Rows are ordered chronologically; rounds 0-4 are the tuning rounds and the last
three rows are the `phase=final` records of the delivered file.
"""

import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))  # perf_opt/experiments
OPT = os.path.dirname(HERE)                          # perf_opt
TASK = os.path.dirname(OPT)                          # examples/engram_gate_conv_bwd_perf

KERNEL_ID = (
    "tileops/kernels/sequence_modeling/engram_gate_conv_bwd/engram_gate_conv_bwd_kernel"
    "::_engram_gate_conv_bwd_kernel::main"
)
BASE = (
    "../../TileOPs/tileops/kernels/sequence_modeling/engram_gate_conv_bwd/"
    "engram_gate_conv_bwd_kernel/_engram_gate_conv_bwd_kernel.py"
)
TS = "2026-11-24T23:20:00Z"

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
    (0, "baseline-stage3", None, "baseline", BASE, "train-s96-bfloat16",
     [14.776, 15.135, 14.401],
     "profiles/experiments/_engram_gate_conv_bwd_kernel_s96_r0",
     "frozen Stage-3 module (sha256 0d5bb90b...), 3 independent capture processes; "
     "T_op = launch1 + launch2 (11.6 + 3.2 us), grid 48 + 36"),
    (0, "baseline-stage3", None, "baseline", BASE, "train-s64-float16",
     [10.591, 10.789, 10.603],
     "profiles/experiments/_engram_gate_conv_bwd_kernel_s64_r0",
     "frozen Stage-3 module, 3 independent capture processes"),
    (0, "baseline-stage3", None, "baseline", BASE, "train-s128-bfloat16",
     [9.777, 8.891, 9.663],
     "profiles/experiments/_engram_gate_conv_bwd_kernel_s128_r0",
     "frozen Stage-3 module, 3 independent capture processes"),

    (1, "r1_c1-vhatdy-window", "baseline-stage3", "candidate",
     "engram_gate_conv_bwd_round1_c1.py", "train-s96-bfloat16",
     [13.906, 14.030, 13.784],
     "profiles/experiments/engram_gate_conv_bwd_round1_c1_s96_r0",
     "O-1: vhat / dY row chain -> one 2-D window per block (24 -> 2 descriptors); "
     "MTE2 busy 5.5 -> 3.1 us"),
    (1, "r1_c1-vhatdy-window", "baseline-stage3", "candidate",
     "engram_gate_conv_bwd_round1_c1.py", "train-s64-float16",
     [9.818, 10.032, 9.506],
     "profiles/experiments/engram_gate_conv_bwd_round1_c1_s64_r0", "O-1 on s64"),
    (1, "r1_c1-vhatdy-window", "baseline-stage3", "candidate",
     "engram_gate_conv_bwd_round1_c1.py", "train-s128-bfloat16",
     [8.064, 8.193, 8.236],
     "profiles/experiments/engram_gate_conv_bwd_round1_c1_s128_r0", "O-1 on s128"),
    (1, "r1_c2-window-store", "baseline-stage3", "candidate",
     "engram_gate_conv_bwd_round1_c2.py", "train-s96-bfloat16",
     [14.107, 14.200, 13.691],
     "profiles/experiments/engram_gate_conv_bwd_round1_c2_s96_r0",
     "O-2: adds the (c)-group H/k/v window loads + dH/dk/dv window stores "
     "(36 -> 6 descriptors); MTE2 3.1 -> 1.0 us, MTE3 1.15 -> 0.35 us, vec_wait 0.68 -> 0.11"),
    (1, "r1_c2-window-store", "baseline-stage3", "candidate",
     "engram_gate_conv_bwd_round1_c2.py", "train-s64-float16",
     [9.839, 9.604, 9.526],
     "profiles/experiments/engram_gate_conv_bwd_round1_c2_s64_r0", "O-2 on s64"),
    (1, "r1_c2-window-store", "baseline-stage3", "candidate",
     "engram_gate_conv_bwd_round1_c2.py", "train-s128-bfloat16",
     [8.094, 8.103, 7.755],
     "profiles/experiments/engram_gate_conv_bwd_round1_c2_s128_r0", "O-2 on s128"),

    (2, "r2_c3-hoist-invariant", "r1_c2-window-store", "candidate",
     "engram_gate_conv_bwd_round2_c3.py", "train-s96-bfloat16",
     [13.838, 13.487, 14.269],
     "profiles/experiments/engram_gate_conv_bwd_round2_c3_s96_r0",
     "O-9: hoist rrms^3/d out of the chunk sweeps; aiv_vec_time unchanged "
     "(6.318 us) -> the compiler already CSE'd it, gain is inside the noise"),
    (2, "r2_c4-fused-dconv", "r1_c2-window-store", "candidate",
     "engram_gate_conv_bwd_round2_c4.py", "train-s96-bfloat16",
     [14.622, 14.181, 14.166],
     "profiles/experiments/engram_gate_conv_bwd_round2_c4_s96_r0",
     "O-10: fuse the dconv_w partial into the conv sweep with the taps in registers; "
     "aiv_vec_time 6.318 -> 6.683 us (register pressure) -> rejected"),
    (2, "r2_c5-reduce-tiling", "r1_c2-window-store", "candidate",
     "engram_gate_conv_bwd_round2_c5.py", "train-s96-bfloat16",
     [13.907, 13.343, 13.663],
     "profiles/experiments/engram_gate_conv_bwd_round2_c5_s96_r0",
     "O-11: launch-2 column tiling to 6*tiles<=54 blocks (12 -> 36 blocks of 128 "
     "columns); launch2 3.2 -> 2.9 us, per-core copy bandwidth 31 -> 55 GB/s"),

    (3, "r3_c6-merged", "r2_c5-reduce-tiling", "merged",
     "engram_gate_conv_bwd_round3_c6.py", "train-s96-bfloat16",
     [13.358, 13.234, 13.589],
     "profiles/experiments/engram_gate_conv_bwd_round3_c6_s96_r0",
     "merge of r2_c5 + r2_c3 (the latter is instruction-neutral), re-measured on the "
     "whole workload set"),
    (3, "r3_c6-merged", "r2_c5-reduce-tiling", "merged",
     "engram_gate_conv_bwd_round3_c6.py", "train-s64-float16",
     [9.145, 9.688, 8.884],
     "profiles/experiments/engram_gate_conv_bwd_round3_c6_s64_r0", "merge check on s64"),
    (3, "r3_c6-merged", "r2_c5-reduce-tiling", "merged",
     "engram_gate_conv_bwd_round3_c6.py", "train-s128-bfloat16",
     [7.660, 7.103, 7.716],
     "profiles/experiments/engram_gate_conv_bwd_round3_c6_s128_r0", "merge check on s128"),

    (4, "r4_c7-dconv-chunkmajor", "r3_c6-merged", "candidate",
     "engram_gate_conv_bwd_round4_c7.py", "train-s96-bfloat16",
     [13.131, 13.045, 12.964],
     "profiles/experiments/engram_gate_conv_bwd_round4_c7_s96_r0",
     "O-12: dconv_w partial chunk-major with the 4 accumulators in registers "
     "(nrm/d_silu rows block-resident, the raw vhat/dY fp32 rings dropped)"),

    (5, "final-chunkmajor", "r4_c7-dconv-chunkmajor", "final",
     "engram_gate_conv_bwd.py", "train-s96-bfloat16",
     [13.676, 12.623, 12.784],
     "profiles/experiments/engram_gate_conv_bwd_s96_r0",
     "final perf_opt file, 3 independent capture processes"),
    (5, "final-chunkmajor", "r4_c7-dconv-chunkmajor", "final",
     "engram_gate_conv_bwd.py", "train-s64-float16",
     [8.826, 9.641, 9.111],
     "profiles/experiments/engram_gate_conv_bwd_s64_r0",
     "final perf_opt file, 3 independent capture processes"),
    (5, "final-chunkmajor", "r4_c7-dconv-chunkmajor", "final",
     "engram_gate_conv_bwd.py", "train-s128-bfloat16",
     [7.244, 7.393, 7.601],
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
    append = "--append" in sys.argv[1:]
    rows = ROWS if not append else ROWS[-3:]
    out = []
    for rnd, cid, pid, phase, art, wl, samples, raw, note in ROWS if not append else rows:
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
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"{'appended' if append else 'wrote'} {path}: {len(out)} rows")
    for rec in out:
        print(f"  r{rec['round']} {rec['phase']:<9} {rec['candidate_id']:<24} "
              f"{rec['workload_id']:<18} {rec['duration_us']:>7.3f} us")


if __name__ == "__main__":
    main()
