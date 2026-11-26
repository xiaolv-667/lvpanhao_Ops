"""Interleaved A/B/A/B measurement of two kernel variants (T-2 protocol).

``PERF_RECORDS`` / winner decisions for differences below the noise threshold must
come from an interleaved multi-run comparison, not from two single captures.  The
platform's ``ab_test.py`` wraps each arm in ``msprof op --launch-count N``; this
task's measurement contract is *one capture = one operator call*, so this runner
implements the same interleaved protocol on top of ``bench_one`` (fresh process
per call, one ``msopprof --launch-count=1 --warm-up=0`` capture per call) and
alternates the arm order per pair to cancel slow drift.

    python ab_interleaved.py --a-module base.py --b-module cand.py \
        --workload llama3-8b-b2-2k-float32 --pairs 4
"""

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_one  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-module", required=True)
    ap.add_argument("--b-module", required=True)
    ap.add_argument("--workload", default="llama3-8b-b2-2k-float32")
    ap.add_argument("--pairs", type=int, default=4)
    ap.add_argument("--out-root", default=None)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    perf_opt = os.path.dirname(here)
    out_root = os.path.abspath(args.out_root or os.path.join(perf_opt, "profiles", "ab"))
    arms = {"a": os.path.abspath(args.a_module), "b": os.path.abspath(args.b_module)}
    # resolve each arm's default units_per_block (the wrapper's default_config value)
    upb = {}
    for arm, path in arms.items():
        upb[arm] = bench_one.load_module(path)._UNITS_PER_BLOCK
    samples = {"a": [], "b": []}
    warm = {"a": True, "b": True}
    for pair in range(args.pairs):
        order = ("a", "b") if pair % 2 == 0 else ("b", "a")
        for arm in order:
            out_dir = os.path.join(out_root, f"pair{pair}_{arm}")
            total, rows = bench_one.run_capture(
                arms[arm], args.workload, out_dir, "BasicInfo", upb[arm], False,
                warmup=warm[arm],
            )
            warm[arm] = False
            assert len(rows) == 1, rows
            samples[arm].append(round(total, 3))
            print(json.dumps({"pair": pair, "arm": arm, "us": round(total, 3)}), flush=True)

    a_med = statistics.median(samples["a"])
    b_med = statistics.median(samples["b"])
    delta = (a_med - b_med) / a_med * 100.0
    wins = sum(1 for x, y in zip(samples["a"], samples["b"]) if y < x)
    summary = {
        "a": os.path.basename(arms["a"]), "b": os.path.basename(arms["b"]),
        "workload": args.workload,
        "a_samples": samples["a"], "b_samples": samples["b"],
        "a_median": a_med, "b_median": b_med,
        "delta_pct": round(delta, 2), "b_wins_of": f"{wins}/{args.pairs}",
    }
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
