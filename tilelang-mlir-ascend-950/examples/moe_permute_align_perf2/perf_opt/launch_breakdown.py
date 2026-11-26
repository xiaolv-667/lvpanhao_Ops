"""Per-launch `Task Duration` breakdown for one measured tag.

``benchmark.json`` only carries the per-call sum (``t_op_us``); the split across
the call's launches comes from the msprof ``OpBasicInfo`` CSVs kept under
``profiles/<tag>/msprof/<artifact>``.  Prints one row per workload with every
captured launch, so the iteration log can attribute a call to its launches.

    python launch_breakdown.py r0_base
"""

import csv
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent


def _mean(pattern, column):
    total, count = 0.0, 0
    for path in pattern.parent.glob(pattern.name):
        with path.open() as handle:
            for row in csv.DictReader(handle):
                raw = (row.get(column) or "").strip()
                try:
                    total += float(raw)
                    count += 1
                except ValueError:
                    continue
    return total / count if count else 0.0


def _mem(pattern):
    rd = wr = 0.0
    for path in pattern.parent.glob(pattern.name):
        with path.open() as handle:
            for row in csv.DictReader(handle):
                for key, target in (("read_main_memory_datas(KB)", "r"),
                                    ("write_main_memory_datas(KB)", "w")):
                    raw = (row.get(key) or "").strip()
                    try:
                        value = float(raw)
                    except ValueError:
                        continue
                    if target == "r":
                        rd += value
                    else:
                        wr += value
    return rd, wr


def main():
    tag = sys.argv[1]
    tag_dir = HERE / "profiles" / tag
    case_art = {}
    for bench in sorted(tag_dir.glob("bench_run*.json")):
        for rec in json.loads(bench.read_text())["records"]:
            art = rec["result"].get("artifact_dir")
            if art:
                case_art.setdefault(rec["label"], art)
    for label, art in sorted(case_art.items()):
        rows = []
        for info in sorted(pathlib.Path(art).rglob("OpBasicInfo*.csv")):
            capture = info.parent
            with info.open() as handle:
                for row in csv.DictReader(handle):
                    name = row["Op Name"].strip()
                    dur = float(row["Task Duration(us)"])
                    ratio = _mean(capture / "ArithmeticUtilization_*.csv", "aiv_vec_ratio")
                    rd, wr = _mem(capture / "Memory_*.csv")
                    rows.append((name, dur, ratio, rd, wr))
        total = sum(r[1] for r in rows)
        parts = " + ".join(
            f"{name}={dur:.2f}(vec {ratio:.2f}, {rd + wr:.0f}KB)" for name, dur, ratio, rd, wr in rows
        )
        gbs = sum(r[3] + r[4] for r in rows) * 1024 / (total * 1e-6) / 1e9 if total else 0
        print(f"{label:22s} sum={total:8.2f}  bw={gbs:6.1f}GB/s   {parts}")


if __name__ == "__main__":
    main()
