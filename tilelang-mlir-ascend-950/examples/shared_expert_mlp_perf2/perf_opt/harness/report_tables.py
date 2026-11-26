"""Render the opt_log tables from ``perf_records.jsonl`` and the captures.

Every number printed here comes from a msopprof capture (``T_op`` = sum of the
per-launch medians) or from the MFU policy constants of
``examples/batch2/mfu_baseline_v1.json`` -- nothing is typed in by hand.

Usage: ``python report_tables.py [--round-tags round1-candA=..., ...]``
"""

import argparse
import json
from pathlib import Path

PERF_OPT = Path(__file__).resolve().parent.parent
RECORDS = PERF_OPT / "perf_records.jsonl"
PEAK_BF16 = 378.0e12  # examples/batch2/mfu_baseline_v1.json cube_tflops.bfloat16
PEAK_FP16 = 378.0e12
DTYPE_PEAK = {"bfloat16": PEAK_BF16, "float16": PEAK_FP16}


def flops(T, H, S):
    """Manifest roofline: 6*T*S*H + 6*T*S (moe.yaml SharedExpertMLPFwdOp)."""
    return 6 * T * S * H + 6 * T * S


def rows():
    out = []
    for line in RECORDS.read_text().splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", default="")
    args = ap.parse_args()
    inventory = json.loads((PERF_OPT / "workload_inventory.json").read_text())
    params = {w["workload_id"]: w["params"] for w in inventory["workloads"]}
    dtypes = {w["workload_id"]: w["dtype"] for w in inventory["workloads"]}
    by_candidate = {}
    for row in rows():
        by_candidate.setdefault(row["candidate_id"], {})[row["workload_id"]] = row
    order = list(params)
    print("| workload | dtype | FLOPs | roofline us | " +
          " | ".join(f"{c}" for c in by_candidate) + " |")
    print("|---|---|---:|---:|" + "---:|" * len(by_candidate))
    for workload in order:
        p = params[workload]
        f = flops(p["T"], p["H"], p["S"])
        ideal = f / DTYPE_PEAK[dtypes[workload]] * 1e6
        cells = []
        for cand in by_candidate:
            row = by_candidate[cand].get(workload)
            if row is None:
                cells.append("n/a")
            else:
                tflops = f / (row["duration_us"] * 1e-6) / 1e12
                mfu = 100.0 * f / (row["duration_us"] * 1e-6) / DTYPE_PEAK[
                    dtypes[workload]]
                cells.append(f"{row['duration_us']:.1f} us / {mfu:.1f}%")
        print(f"| {workload} | {dtypes[workload]} | {f/1e9:.1f}e9 | {ideal:.1f} | "
              + " | ".join(cells) + " |")
    print()
    print("| workload | " + " | ".join(
        f"{c} vs baseline" for c in by_candidate if c != "baseline") + " |")
    base = by_candidate.get("baseline", {})
    others = [c for c in by_candidate if c != "baseline"]
    print("|---|" + "---:|" * len(others))
    for workload in order:
        cells = []
        for cand in others:
            row = by_candidate[cand].get(workload)
            if row is None or workload not in base:
                cells.append("n/a")
            else:
                delta = 100.0 * (row["duration_us"] / base[workload]["duration_us"] - 1)
                cells.append(f"{delta:+.1f}%")
        print(f"| {workload} | " + " | ".join(cells) + " |")
    _ = args


if __name__ == "__main__":
    main()
