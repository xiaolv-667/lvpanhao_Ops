#!/usr/bin/env python3
"""Sum per-kernel msprof durations into a whole-operator time (T_op).

Why: the TileOPs benchmark attributes a single kernel name to each case.  An
NPU operator that launches several kernels per call (plus torch-side device
ops such as ZerosLike/Range) is then reported as ``invalid`` (unfiltered
capture) or, worse, as *valid but incomplete* when the auto-derived name
filters out the other launches.  This script reads the raw msprof artifact
that the reporter already stored and sums one representative duration per
captured operator, producing T_op for MFU/MU accounting.

Usage:
    python msprof_sum.py --run-json <run.json> [--json-out out.json]
                         [--include-host-ops] [--exclude-regex REGEX]

Policy (documented in the acceptance note):
    * every kernel launched by the operator is included;
    * torch-side device ops (ZerosLike*, Range*) are included by default
      because they are part of the operator's forward;
    * host-padding ops (PadV3*) are excluded by default.

The harness profiles exactly one operator call per capture (0 warm-up +
1 measured launch), so **every sample is one kernel launch inside that
call**: T_op is the sum of all samples (including several launches that
share one kernel name, e.g. the two `main_kernel` launches of
`EngramGateConvBwdOp`).  The reporter's own ``latency_us`` is kept next to
T_op so the difference is visible.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

HOST_PAD = re.compile(r"^Pad", re.I)
HOST_AUX = re.compile(r"^(ZerosLike|Range|Fill|Cast)", re.I)


def find_csvs(root: Path):
    return sorted(root.rglob("OpBasicInfo*.csv"))


def read_durations(root: Path) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for path in find_csvs(root):
        try:
            with path.open(newline="") as fh:
                for row in csv.DictReader(fh):
                    name = (row.get("Op Name") or "").strip()
                    raw = (row.get("Task Duration(us)") or "").strip()
                    if not name or not raw:
                        continue
                    out.setdefault(name, []).append(float(raw))
        except (OSError, ValueError):
            continue
    return out


def resolve_dir(rel: str | None) -> Path | None:
    if not rel:
        return None
    candidates = [Path(rel)]
    for base in (Path("/workspace/tilelang-mlir-ascend-950"),
                 Path(__file__).resolve().parents[2]):
        candidates.append(base / rel)
    for c in candidates:
        if c.exists():
            return c
    return None


def case_table(run_json: Path, include_host: bool, exclude_re: re.Pattern | None):
    run = json.loads(run_json.read_text())
    rows = []
    for case in (run.get("performance") or {}).get("cases", []):
        art = resolve_dir(case.get("artifact_dir"))
        durations = read_durations(art) if art else {}
        picked = {}
        for name, samples in durations.items():
            if exclude_re and exclude_re.search(name):
                continue
            if not include_host and HOST_PAD.search(name):
                continue
            # one call is profiled per capture: sum every launch of this name
            picked[name] = float(sum(samples))
        t_op = sum(picked.values())
        rows.append(
            {
                "case": case.get("label"),
                "reported_us": case.get("latency_us"),
                "profiler_valid": case.get("profiler_valid"),
                "invalid_reason": case.get("invalid_reason"),
                "captured": sorted(durations),
                "breakdown_us": dict(sorted(picked.items(), key=lambda kv: -kv[1])),
                "t_op_us": round(t_op, 3) if picked else None,
                "artifact": str(art) if art else None,
                # Prefer the acceptance record (manifest FLOPs/Bytes). The
                # ``performance_tops``/``bandwidth_tb_s`` fallback only exists
                # for run.json files produced before the MFU/MU layer.
                "flops": (case.get("acceptance") or {}).get("roofline_flops")
                if (case.get("acceptance") or {}).get("roofline_flops") is not None
                else (case.get("performance_tops") or 0)
                * 1e12
                * (case.get("latency_us") or 0)
                * 1e-6,
                "bytes": (case.get("acceptance") or {}).get("roofline_bytes")
                if (case.get("acceptance") or {}).get("roofline_bytes") is not None
                else (case.get("bandwidth_tb_s") or 0)
                * 1e12
                * (case.get("latency_us") or 0)
                * 1e-6,
            }
        )
    return run.get("operator"), rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-json", required=True)
    ap.add_argument("--json-out")
    ap.add_argument("--include-host-ops", action="store_true",
                    help="also include Pad* device ops")
    ap.add_argument("--exclude-regex")
    args = ap.parse_args(argv)

    exclude = re.compile(args.exclude_regex) if args.exclude_regex else None
    op, rows = case_table(Path(args.run_json), args.include_host_ops, exclude)

    print(f"operator: {op}")
    print(f"run.json: {args.run_json}")
    print(f"{'case':28s} {'reported':>10s} {'T_op':>10s} {'ops':>3s}  breakdown")
    for r in rows:
        t = f"{r['t_op_us']:.3f}" if r["t_op_us"] is not None else "N/A"
        rep = f"{r['reported_us']:.3f}" if r["reported_us"] is not None else "N/A"
        bd = ", ".join(f"{k}={v:.3f}" for k, v in r["breakdown_us"].items())
        print(f"{str(r['case'])[:28]:28s} {rep:>10s} {t:>10s} {len(r['captured']):>3d}  {bd}")
        if r["t_op_us"] is None:
            print(f"    MISSING RAW TRACE ({r['artifact']})")

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps({"operator": op, "run_json": args.run_json, "cases": rows},
                       ensure_ascii=False, indent=1)
        )
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
