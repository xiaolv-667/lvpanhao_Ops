"""kb_stale_check.py -- toolchain-staleness detection for KB entries (E-5).

"Entries are auto-downgraded to pending re-verification after toolchain
changes" was a prose-only convention; this tool makes it mechanical. It never
modifies library files -- it only reports.

Current stamp detection is delegated to the platform adapter. Override it
with --stamp when checking a recorded toolchain.

Explicit fingerprints are compared directly. Historical version formats are
interpreted by the selected toolchain; unavailable evidence remains unknown. Use --include-unknown to list those entries.

Usage:
  python3 .agents/tools/kb_stale_check.py [--stamp "tilelang X + toolchain Y"]
                                          [--include-unknown] [--json]
Exit 0 always (advisory); consumes the JSON "stale" list in conductor start /
optimizer Phase 0 as the pre-task re-verification checklist. Repro-backed
entries can be mechanically re-verified with:
  python3 .agents/tools/repro_runner.py --filter stale
"""

import argparse
import glob
import json
import os
import re
import sys

from pathlib import Path

from platform_context import (
    add_platform_args,
    get_platform,
    get_adapter,
    knowledge_entry_applies,
    run_with_platform,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *([".."] * 2)))


def knowledge_globs():
    pattern_dir = get_platform().toolchain["knowledge"]["patterns"]
    return [
        os.path.join(REPO_ROOT, pattern_dir, "*.md"),
        os.path.join(REPO_ROOT, ".agents", "evolution", "queue.md"),
    ]


def current_stamp() -> dict:
    platform = get_platform()
    recorded = (platform.runtime or {}).get("toolchain_stamp")
    return recorded or platform.adapter.toolchain_stamp(
        REPO_ROOT, active=platform.hardware is not None
    )


def parse_entries(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        text = f.read()
    rel = os.path.relpath(path, REPO_ROOT)
    entries = []
    parts = re.split(r"(?m)^---\s*$", text)
    for i in range(1, len(parts) - 1, 2):
        fm_raw = parts[i]
        m = re.search(r"^toolchain_stamp:[ \t]*(.+)$", fm_raw, re.M) or re.search(
            r"^toolchain:[ \t]*(.+)$", fm_raw, re.M
        )
        scope = dict(re.findall(r"(?m)^([a-z_]+):[ \t]*(.*)$", fm_raw))
        if not knowledge_entry_applies(scope):
            continue
        mid = re.search(r"^id:\s*(\S+)", fm_raw, re.M)
        if m and mid:
            entries.append(
                {"id": mid.group(1), "toolchain": m.group(1).strip(), "path": rel}
            )
    # queue.md VP entries carry '- toolchain_stamp:'
    for m in re.finditer(r"(?m)^## (VP-\d{4}-\d{4})\s*$([\s\S]*?)(?=^## |\Z)", text):
        fields = dict(re.findall(r"(?m)^- ([a-z_]+):[ \t]*(.*)$", m.group(2)))
        if not knowledge_entry_applies({**fields, "source": "queue"}):
            continue
        mm = re.search(r"^- toolchain_stamp:\s*(.+)$", m.group(2), re.M)
        if mm:
            entries.append(
                {"id": m.group(1), "toolchain": mm.group(1).strip(), "path": rel}
            )
    return entries


def classify(entry_stamp: str, current: dict) -> str:
    return get_adapter().classify_stamp(entry_stamp, current)


def check_entries(current: dict, *, include_unknown: bool = False) -> dict:
    stale, unknown = [], []
    for pattern in knowledge_globs():
        for path in sorted(glob.glob(pattern)):
            if not os.path.isfile(path):
                continue
            for e in parse_entries(path):
                verdict = classify(e["toolchain"], current)
                if verdict == "stale":
                    stale.append({**e, "verdict": "stale"})
                elif verdict == "unknown":
                    unknown.append({**e, "verdict": "unknown"})

    report = {
        "current_stamp": current["raw"],
        "checked": "pattern-library topic files + queue.md",
        "stale_count": len(stale),
        "stale": stale,
        "unknown_count": len(unknown),
        "note": "stale = recorded toolchain version differs from current; "
        "re-verify repro-backed entries via: "
        "repro_runner.py --filter stale",
    }
    if include_unknown:
        report["unknown"] = unknown
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--stamp",
        default=None,
        help="override current stamp, e.g. 'tilelang 0.1.2+abcdef123 + CANN 8.5.0'",
    )
    ap.add_argument("--include-unknown", action="store_true")
    ap.add_argument("--json", action="store_true")
    add_platform_args(ap)
    args = ap.parse_args()
    return run_with_platform(ap, args, run, Path.cwd())


def run(args) -> int:
    current = {"raw": args.stamp} if args.stamp else current_stamp()

    report = check_entries(current, include_unknown=args.include_unknown)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"current stamp: {current['raw']}")
        print(f"stale entries: {report['stale_count']}")
        for e in report["stale"]:
            print(f"  [stale] {e['id']}  ({e['path']})  stamp: {e['toolchain'][:80]}")
        print(
            f"unknown (missing or unrecognized version stamp): {report['unknown_count']} "
            f"(--include-unknown to list)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
