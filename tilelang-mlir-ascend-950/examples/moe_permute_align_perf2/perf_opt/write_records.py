"""Append one round of platform measurements to ``perf_opt/perf_records.jsonl``.

Append-only: existing rows are never rewritten.  One row per (kernel_id,
workload_id, phase), fields per ``_shared/standards/signal-registry.md`` §5 plus
the platform evidence the Stage 4 gate reconciles against
(hardware_id / toolchain_id / platform_config / platform_runtime).

    python write_records.py --tag r1_c1 --phase candidate --round 1 \
        --candidate-id c1_deterministic_slot --parent-id stage3_baseline \
        --artifact perf_opt/moe_permute_align_round1_c1.py [--l0-pass]

``--tag`` names a directory under ``perf_opt/profiles`` holding ``summary.json``
(written by ``run_bench.py``) and the raw msprof captures.
"""

import argparse
import hashlib
import json
import pathlib
import sys
from datetime import datetime, timezone

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
KERNEL_ID = "perf_opt/MoEPermuteAlignFwdOp.py::MoEPermuteAlignKernel"
TOOLS = REPO / ".agents" / "tools"


def evidence():
    sys.path.insert(0, str(TOOLS))
    import platform_context

    state = json.loads((HERE.parent / ".stage_state.json").read_text())
    return platform_context.resolve_platform(state=state).evidence()


def sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--phase", required=True,
                    choices=["baseline", "candidate", "merged", "final"])
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--candidate-id", required=True)
    ap.add_argument("--parent-id", default=None)
    ap.add_argument("--artifact", required=True, help="repo-relative measured file")
    ap.add_argument("--l0-pass", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    art = pathlib.Path(args.artifact)
    if not art.is_absolute():
        art = REPO / art
    digest = sha256(art)
    tag_dir = HERE / "profiles" / args.tag
    summary = json.loads((tag_dir / "summary.json").read_text())
    raw = {}
    for bench in sorted(tag_dir.glob("bench_run*.json")) + sorted(
            tag_dir.glob("report_benchmark.json")):
        for rec in json.loads(bench.read_text())["records"]:
            art_dir = rec["result"].get("artifact_dir")
            if art_dir:
                raw.setdefault(rec["label"], art_dir)
    ev = evidence()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    rows = []
    for label, case in sorted(summary["cases"].items()):
        rows.append({
            "round": args.round,
            "candidate_id": args.candidate_id,
            "parent_id": args.parent_id,
            "kernel_id": KERNEL_ID,
            "workload_id": label,
            "phase": args.phase,
            # absolute: the Stage 4 gate resolves this against the records file's
            # own directory, so a repo-relative path would double the prefix
            "artifact_path": str(art),
            "artifact_sha256": digest,
            "duration_us": round(float(case["t_op_us"]), 4),
            "l0_pass": bool(args.l0_pass),
            "msprof_raw_path": raw.get(label, str(tag_dir.relative_to(REPO))),
            "timestamp": stamp,
            **ev,
        })
    if args.dry_run:
        print(json.dumps(rows, indent=1)[:4000])
        return
    with (HERE / "perf_records.jsonl").open("a") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(f"appended {len(rows)} rows ({args.tag} / {args.phase} / {args.candidate_id})")


if __name__ == "__main__":
    main()
