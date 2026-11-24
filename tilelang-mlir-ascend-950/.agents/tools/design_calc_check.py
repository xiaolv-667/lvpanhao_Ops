"""design_calc_check.py -- mechanical arithmetic checks on DESIGN.md (D-5).

Best-effort mechanical re-computation of the machine-checkable parts of a
Stage 1 design document. Output is an advisory report for the Stage 2
reviewer (dimension 2/3/8 support) -- it is NOT a gate: any check that
cannot be parsed from the document is reported as "skip" with the reason.

Checks:
  1. ub_budget   -- sum the buffer sizes declared in the memory plan section
                    (mixed B/KB/MB units are normalized; VP-2026-0009) and
                    compare against the platform budget and report its
                    auto-multi-buffer inflation allowance.
  2. l0c_budget  -- for GEMM designs, block_M * block_N * accum_bytes must
                    fit the platform accumulator budget.
  3. core_split  -- recompute logical cores ceil(M/bm) * ceil(N/bn) from the
                    numbers the design declares and cross-check the declared
                    logical-core count.
  4. r3_metrics  -- the research complexity table must cover the four
                    required metrics (FLOPs / access bytes / scan passes /
                    intermediate buffer peak).

Usage:
  python3 .agents/tools/design_calc_check.py --design examples/{p}/{op}/DESIGN.md
Exit code 0 always (advisory); parse the JSON "checks" list.
"""

import argparse
import json
import math
import re

from pathlib import Path

from platform_context import (
    add_platform_args,
    get_adapter,
    get_platform,
    run_with_platform,
)

UNIT_KB = {"b": 1.0 / 1024, "kb": 1.0, "mb": 1024.0}


def _section(text: str, marker: str) -> str:
    """Return the body of the first heading containing the marker."""
    lines = text.splitlines()
    body, inside, depth = [], False, 0
    for ln in lines:
        m = re.match(r"^(#{1,6})\s+(.*)$", ln)
        if m:
            if inside and len(m.group(1)) <= depth:
                break
            if marker in m.group(2):
                inside, depth = True, len(m.group(1))
                continue
        if inside:
            body.append(ln)
    return "\n".join(body)


SIZE_CHAIN_RE = re.compile(r"((?:\d+(?:\.\d+)?\s*\+\s*)+\d+(?:\.\d+)?)\s*(B|KB|MB)\b")
SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(B|KB|MB)\b")
TOTAL_MARK = re.compile(r"合计|总占用|total", re.I)
CAPACITY_MARK = re.compile(r"容量|佐证|参考|≤")


def _parse_sizes(s: str) -> list:
    """Parse size expressions in a string; chains like '16 + 16 + 8KB'
    count as one summed entry (mixed byte-unit trap, VP-2026-0009)."""
    out = []
    covered = []
    for m in SIZE_CHAIN_RE.finditer(s):
        total = sum(float(x) for x in re.findall(r"\d+(?:\.\d+)?", m.group(1)))
        out.append(total * UNIT_KB[m.group(2).lower()])
        covered.append(m.span())
    for m in SIZE_RE.finditer(s):
        if any(a <= m.start() < b for a, b in covered):
            continue
        out.append(float(m.group(1)) * UNIT_KB[m.group(2).lower()])
    return out


def check_ub_budget(text: str, budget=None) -> dict:
    """Sum per-level buffer tables in the memory plan and compare against
    the level capacity; recompute any declared 合计 row."""
    sec = _section(text, "4.5") or _section(text, "内存") or ""
    budget = budget or get_platform().design_budget()
    table_levels = "|".join(map(re.escape, budget["table_levels"]))
    heading_levels = "|".join(map(re.escape, budget["heading_levels"]))
    inflation_level = budget["inflation_level"]
    level = None
    buckets = {}
    declared_totals = {}
    for ln in sec.splitlines():
        m = re.search(rf"\*\*\s*({table_levels})\b", ln) or re.search(
            rf"^#{{2,6}}.*\b({heading_levels})\b", ln
        )
        if m:
            level = m.group(1).upper()
            buckets.setdefault(level, [])
            continue
        if not ln.lstrip().startswith("|"):
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        joined = " ".join(cells)
        if set(joined.replace(" ", "")) <= {"-", " "}:
            continue
        if not any(re.match(r"^[a-zA-Z_]", c) or "缓冲" in c for c in cells):
            continue  # header rows
        if level is None:
            continue
        if TOTAL_MARK.search(joined) and not CAPACITY_MARK.search(joined):
            vals = _parse_sizes(joined)
            if vals:
                declared_totals[level] = sum(vals)
            continue
        vals = _parse_sizes(joined)
        if vals:
            buckets[level].extend(vals)
    inflation = budget["ub_inflation"]
    checks = []
    for lv, sizes in buckets.items():
        if not sizes:
            continue
        total = sum(sizes)
        cap = budget["capacity_kb"].get(lv)
        detail = (
            f"{lv}: {len(sizes)} buffer entries sum = {total:.1f} KB"
            + (
                f" (declared 合计 {declared_totals.get(lv):.1f} KB, "
                f"recompute matches: "
                f"{abs(declared_totals.get(lv, total) - total) < 1.0})"
                if lv in declared_totals
                else ""
            )
            + (
                f"; capacity = {cap} KB"
                if cap is not None
                else "; capacity is unverified for the selected hardware"
            )
            + (
                f"; x{inflation} auto-multi-buffer inflation budget = "
                f"{cap / inflation:.0f} KB"
                if lv == inflation_level and cap is not None and inflation is not None
                else ""
            )
        )
        status = "skip" if cap is None else ("fail" if total > cap else "pass")
        checks.append(
            {
                "check": f"{lv.lower()}_budget",
                "status": status,
                "detail": detail,
                "within_inflation_budget": (
                    total <= cap / inflation
                    if lv == inflation_level
                    and cap is not None
                    and inflation is not None
                    else None
                ),
            }
        )
    if not checks:
        return {
            "check": "ub_budget",
            "status": "skip",
            "detail": "no per-level buffer tables parsed from the memory "
            "plan (advisory checker; verify manually)",
        }
    return checks


def check_l0c_budget(text: str, budget=None) -> dict:
    adapter = get_adapter()
    section = re.search(
        r"^#{2,6}[ \t]+3\.3\.?[ \t]+[^\n]*\n(.*?)(?=^#{2,6}[ \t]+|\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    pseudocode = section.group(1) if section else text
    if not adapter.has_matrix_path(pseudocode):
        return {
            "check": "l0c_budget",
            "status": "skip",
            "detail": adapter.matrix_path_required_message(),
        }
    budget = budget or get_platform().design_budget()
    accumulator = budget["accumulator"]
    level, dtype = accumulator["level"], accumulator["dtype"]
    sec = _section(text, "Tiling") or text
    bm = bn = None
    m = re.search(r"\b(?:block_?M|bm)\s*[=×x*]\s*(\d+)", sec, re.I)
    if m:
        bm = int(m.group(1))
    m = re.search(r"\b(?:block_?N|bn)\s*[=×x*]\s*(\d+)", sec, re.I)
    if m:
        bn = int(m.group(1))
    if bm is None or bn is None:
        # try "bm x bn" adjacent pattern
        m = re.search(
            rf"\b(\d{{2,4}})\s*[x×*]\s*(\d{{2,4}})\b.*?"
            rf"(?:{re.escape(level)}|累加|accum|{re.escape(dtype)})",
            text,
        )
        if m:
            bm, bn = int(m.group(1)), int(m.group(2))
    if bm is None or bn is None:
        return {
            "check": "l0c_budget",
            "status": "skip",
            "detail": "no block_M x block_N values parsed",
        }
    need_kb = bm * bn * accumulator["bytes"] / 1024.0
    capacity = budget["capacity_kb"].get(level)
    if capacity is None:
        return {
            "check": "l0c_budget",
            "status": "skip",
            "detail": f"{level} capacity is unverified for the selected hardware",
        }
    ok = need_kb <= capacity
    return {
        "check": "l0c_budget",
        "status": "pass" if ok else "fail",
        "detail": (
            f"block {bm}x{bn} x {dtype} accum = {need_kb:.1f} KB vs {level} {capacity:.0f} KB"
        ),
    }


def check_core_split(text: str) -> dict:
    sec = _section(text, "Tiling") or _section(text, "5") or text
    nums = {}
    for key, pats in (
        ("M", [r"\bM\s*=\s*(\d+)", r"seq.*?(\d{3,6})\b"]),
        ("N", [r"\bN\s*=\s*(\d+)"]),
        ("bm", [r"\b(?:bm|block_?m)\s*=\s*(\d+)"]),
        ("bn", [r"\b(?:bn|block_?n)\s*=\s*(\d+)"]),
    ):
        for p in pats:
            m = re.search(p, sec, re.I)
            if m:
                nums[key] = int(m.group(1))
                break
    declared = None
    m = re.search(r"逻辑核数[^\d]*(\d+)", sec) or re.search(
        r"逻辑核数.*?=\s*(\d+)", sec
    )
    if m:
        declared = int(m.group(1))
    if not all(k in nums for k in ("M", "bm")):
        return {
            "check": "core_split",
            "status": "skip",
            "detail": f"insufficient parsed dims {nums}; declared logical "
            f"cores = {declared}",
        }
    n = nums.get("N", 1)
    bn = nums.get("bn", 1)
    logical = math.ceil(nums["M"] / nums["bm"]) * math.ceil(n / bn)
    if declared is None:
        return {
            "check": "core_split",
            "status": "pass",
            "detail": f"recomputed logical cores = ceil({nums['M']}/"
            f"{nums['bm']}) x ceil({n}/{bn}) = {logical} "
            f"(design declares no explicit count)",
        }
    ok = declared == logical
    return {
        "check": "core_split",
        "status": "pass" if ok else "fail",
        "detail": f"recomputed ceil({nums['M']}/{nums['bm']}) x "
        f"ceil({n}/{bn}) = {logical} vs declared {declared}",
    }


def check_r3_metrics(text: str) -> dict:
    sec = _section(text, "1.6.0") or text
    metrics = {
        "FLOPs": bool(re.search(r"FLOPs?|浮点", sec, re.I)),
        "access_bytes": bool(re.search(r"访存|Bytes|字节", sec)),
        "scan_passes": bool(re.search(r"扫描遍数|遍数|扫描次数", sec)),
        "buffer_peak": bool(re.search(r"中间缓冲|缓冲峰值|buffer", sec, re.I)),
    }
    missing = [k for k, v in metrics.items() if not v]
    if missing:
        return {
            "check": "r3_metrics",
            "status": "fail",
            "detail": f"complexity table missing metrics: {missing}",
        }
    return {
        "check": "r3_metrics",
        "status": "pass",
        "detail": "all four R3 metrics present "
        "(FLOPs/access bytes/scan passes/buffer peak)",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--design", required=True, help="path to DESIGN.md")
    ap.add_argument("--workload", help="measurement workload key for design budget")
    add_platform_args(ap)
    args = ap.parse_args()
    return run_with_platform(ap, args, run, Path(args.design).resolve().parent)


def run(args) -> int:
    try:
        with open(args.design, encoding="utf-8") as f:
            text = f.read()
    except OSError as e:
        print(json.dumps({"error": str(e)}))
        return 0
    budget = get_platform().design_budget(args.workload)
    ub = check_ub_budget(text, budget)
    if not isinstance(ub, list):
        ub = [ub]
    report = {
        "design": args.design,
        "capacity_constants": {
            "ub_kb": budget["capacity_kb"].get(budget["inflation_level"]),
            "ub_inflation": budget["ub_inflation"],
            "l0c_kb": budget["capacity_kb"].get(budget["accumulator"]["level"]),
            "source": budget["source"],
        },
        "checks": [
            *ub,
            check_l0c_budget(text, budget),
            check_core_split(text),
            check_r3_metrics(text),
        ],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
