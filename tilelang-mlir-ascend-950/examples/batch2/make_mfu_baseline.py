#!/usr/bin/env python3
"""Freeze the current (pre-optimization) MFU baseline into a versioned record.

Reads the measured table (``mfu_table.json``), the peak policy
(``peaks.json`` / ``compute_units.json``) and the per-operator report paths,
then writes:

- ``examples/batch2/mfu_baseline_v1.json``  machine-readable snapshot
- ``docs/950pr-mfu-baseline-v1.md``         human-readable record

The snapshot is intentionally labelled "baseline, no Stage 4 optimization".
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import date
from pathlib import Path

BATCH2 = Path(__file__).resolve().parent
ROOT = BATCH2.parents[1]
TILEOPS = ROOT / "examples" / "TileOPs"
REPORTING = TILEOPS / "tileops" / "reporting"

TARGET_PCT = 90.0


def load(path: Path):
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    rows = load(BATCH2 / "mfu_table.json")
    peaks = load(REPORTING / "peaks.json")
    units = load(REPORTING / "compute_units.json")
    summary = {e["op"]: e for e in load(BATCH2 / "integration_summary.json")}
    unit_map = {k: v for k, v in units.items() if not k.startswith("_")}

    by_op: dict[str, list[dict]] = {}
    for row in rows:
        by_op.setdefault(row["operator"], []).append(row)

    operators = []
    for op, cases in sorted(by_op.items()):
        valid = [c for c in cases if not c.get("invalid") and c.get("t_op_us")]
        with_arith = [c for c in valid if (c.get("flops") or 0) > 0]
        if with_arith:
            rep = max(with_arith, key=lambda c: c.get("flops") or 0)
        elif valid:
            rep = max(valid, key=lambda c: c.get("bytes") or 0)
        else:
            rep = None
        mfus = [c["mfu_pct"] for c in valid if c.get("mfu_pct") is not None]
        operators.append(
            {
                "operator": op,
                "compute_unit": unit_map.get(op, {}).get("unit"),
                "compute_unit_evidence": unit_map.get(op, {}).get("evidence"),
                "dominant_case": rep.get("case") if rep else None,
                "dominant_dtype": rep.get("dtype") if rep else None,
                "dominant_ai": round(rep["ai"], 3) if rep and rep.get("ai") is not None else None,
                "dominant_t_op_us": rep.get("t_op_us") if rep else None,
                "dominant_achieved_tflops": (
                    round(rep["achieved_tflops"], 2)
                    if rep and rep.get("achieved_tflops") is not None
                    else None
                ),
                "mfu_pct": round(rep["mfu_pct"], 3) if rep and rep.get("mfu_pct") is not None else None,
                "mfu_ceiling_pct": (
                    round(rep["mfu_ceiling_pct"], 3)
                    if rep and rep.get("mfu_ceiling_pct") is not None
                    else None
                ),
                "share_of_ceiling_pct": (
                    round(100.0 * rep["mfu_pct"] / rep["mfu_ceiling_pct"], 1)
                    if rep and rep.get("mfu_pct") is not None and rep.get("mfu_ceiling_pct")
                    else None
                ),
                "mfu_max_pct": round(max(mfus), 3) if mfus else None,
                "cases_with_mfu": len(mfus),
                "cases_meeting_target": sum(1 for v in mfus if v >= TARGET_PCT),
                "report_run": summary.get(op, {}).get("rj"),
            }
        )

    snapshot = {
        "schema": "tileops-mfu-baseline-v1",
        "label": "pre-optimization baseline (no Stage 4)",
        "date": str(date.today()),
        "hardware": {
            "device": "Ascend950PR_9579 (28 Cube / 56 Vector, 128 GiB)",
            "cube_tflops": peaks["peaks"]["cube_tflops"],
            "cube_tflops_measured": peaks["peaks"].get("cube_tflops_measured"),
            "vector_tflops": peaks["peaks"]["vector_tflops"],
            "bandwidth_tbs": peaks["peaks"]["bandwidth_tbs"],
            "bandwidth_tbs_measured": peaks["peaks"].get("bandwidth_tbs_measured"),
        },
        "policy": {
            "metric": "MFU",
            "formula": "MFU = FLOPs / (T_op * peak_of_executing_unit)",
            "t_op": "sum of every kernel launch of one operator call",
            "target_percent": TARGET_PCT,
            "mu": "diagnostic only, never decides pass/fail",
            "invalid": "achieved > peak*1.01 -> invalid, excluded",
        },
        "source": {
            "mfu_table": str((BATCH2 / "mfu_table.json").relative_to(ROOT)),
            "mfu_table_sha256": sha256(BATCH2 / "mfu_table.json"),
            "peaks_sha256": sha256(REPORTING / "peaks.json"),
            "compute_units_sha256": sha256(REPORTING / "compute_units.json"),
        },
        "operators": operators,
        "cases": rows,
    }
    (BATCH2 / "mfu_baseline_v1.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=1) + "\n"
    )

    # ---- Markdown record -------------------------------------------------
    lines = [
        "# 950PR MFU 基线记录 v1（未优化）",
        "",
        f"> 日期：{date.today()} ｜ 硬件：Ascend950PR_9579（28 Cube / 56 Vector，128 GiB）",
        "> 状态：**Stage 4 优化前的基线**；本次只记录，不做任何优化。",
        "",
        "## 口径",
        "",
        "```",
        "MFU = FLOPs / (T_op × 执行单元峰值)",
        "T_op = 一次算子调用内所有 kernel 的耗时之和（同 stream）",
        "```",
        "",
        "| 项 | 取值 |",
        "|---|---|",
        "| 峰值（提交主口径） | Cube bf16/fp16 **378**、fp8/int8 756、fp32 189；Vector bf16/fp16 **47**、fp32 23；HBM **1.6 TB/s** |",
        "| 实测峰值（附表） | Cube 370.5（8192³ matmul）、HBM 1.424（512 MiB copy） |",
        "| 执行单元 | 由 msprof `ArithmeticUtilization` 实测判定（`compute_units.json`） |",
        "| 目标 | 90%（sparse CV 50% / CV 80% / others 90%） |",
        "| MU | 仅诊断，不参与达标 |",
        "| invalid | `achieved > peak × 1.01`，不参与统计 |",
        "",
        "## 11 个算子的 MFU（主导 case）",
        "",
        "| 算子 | 单元 | 主导 case | AI | T_op | 实测 TFLOP/s | **MFU** | 上限 | MFU/上限 | MFU max | 达标 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|:--:|",
    ]

    def f(value, digits=2, suffix=""):
        return "-" if value is None else f"{value:.{digits}f}{suffix}"

    total_mfu_cases = sum(o["cases_with_mfu"] for o in operators)

    for row in operators:
        met_txt = (
            f"{row['cases_meeting_target']}/{row['cases_with_mfu']}"
            if row["cases_with_mfu"]
            else "n/a"
        )
        lines.append(
            f"| `{row['operator']}` | {row['compute_unit'] or '-'} | "
            f"{row['dominant_case'] or '-'} | {f(row['dominant_ai'], 1)} | "
            f"{f(row['dominant_t_op_us'], 1)} µs | {f(row['dominant_achieved_tflops'], 1)} | "
            f"**{f(row['mfu_pct'], 2, '%')}** | {f(row['mfu_ceiling_pct'], 2, '%')} | "
            f"{f(row['share_of_ceiling_pct'], 1, '%')} | {f(row['mfu_max_pct'], 2, '%')} | "
            f"{met_txt} |"
        )

    lines += [
        "",
        "说明：",
        "",
        "- **主导 case** = 该算子计算量（FLOPs）最大的 workload；MoE 的 FLOPs=0，取流量最大者，MFU 记为 n/a；",
        "- `MFU 上限 = AI / (单元峰值 / 1.6 TB/s)`：访存受限的形状，MFU 被带宽锁死，低不是实现问题；",
        "- `MFU max` = 该算子所有有效 case 中的最高值；`达标` = MFU ≥ 90% 的 case 数；",
        f"- 当前 **0/11 算子、0/{total_mfu_cases} 个有 MFU 的 case 达到 90%**；"
        "最接近的是 MLADense（67.0%）与 SharedExpert（50.0%）。",
        "",
        "## 附录：逐 case（69 行）",
        "",
        "| 算子 | case | dtype | 单元 | T_op µs | MFU % | 上限 % | 达标 |",
        "|---|---|---|---|---:|---:|---:|:--:|",
    ]
    for r in sorted(rows, key=lambda x: (x["operator"], str(x["case"]), str(x.get("dtype")))):
        mfu = r.get("mfu_pct")
        status = "invalid" if r.get("invalid") else (
            "yes" if mfu is not None and mfu >= TARGET_PCT else ("no" if mfu is not None else "n/a")
        )
        lines.append(
            f"| `{r['operator']}` | {r['case']} | {r.get('dtype') or '-'} | {r.get('unit') or '-'} | "
            f"{f(r.get('t_op_us'), 1)} | {f(mfu, 2)} | {f(r.get('mfu_ceiling_pct'), 2)} | {status} |"
        )

    lines += [
        "",
        "## 溯源",
        "",
        f"- 数据表：`examples/batch2/mfu_table.json`（sha256 `{snapshot['source']['mfu_table_sha256'][:16]}…`）",
        "- 峰值策略：`tileops/reporting/peaks.json`、`compute_units.json`",
        "- 报告：`examples/TileOPs/reports/tileops/`（6 个 2026-11-24 重跑 + 5 个既有通过）",
        "- 逐 case CSV：`examples/batch2/mfu_cases.csv`；算子级 CSV：`mfu_operators.csv`",
        "",
        "> 本记录为优化前基线；后续 Stage 4 的结果应另写 v2，不改写本文件。",
    ]
    (ROOT / "docs" / "950pr-mfu-baseline-v1.md").write_text("\n".join(lines) + "\n")

    # CSV mirror (operators) for convenience
    with (BATCH2 / "mfu_baseline_operators.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(operators[0].keys()))
        writer.writeheader()
        writer.writerows(operators)

    print(f"wrote {BATCH2 / 'mfu_baseline_v1.json'}")
    print(f"wrote {ROOT / 'docs' / '950pr-mfu-baseline-v1.md'}")


if __name__ == "__main__":
    main()
