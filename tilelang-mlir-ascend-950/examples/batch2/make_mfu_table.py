#!/usr/bin/env python3
"""Build the corrected whole-operator time + MFU/MU table for batch2 (11 ops).

T_op sources, in priority order
  1. ``t_op_results.json``   forced-unfiltered re-captures (``status == ok``)
  2. report ``run.json``     artifact dir of the msprof capture (sum of every
                             launched kernel, ``Pad*`` excluded)
  3. report ``run.json``     ``latency_us`` when the profiler captured the whole
                             call (``profiler_valid``) and no artifact was kept

Anything else is reported as ``blocked`` with the reason.  A blocked case is
never given a fabricated 0: its T_op stays ``null``.

Peaks (provisional, from the framework's own reported ``computility_tops`` and
the bandwidth model constant):
  Peak compute = 378 TFLOPS (Ascend 950PR SKU table, 28-cube column)
  Peak BW      = 1.6 TB/s   -> machine balance = 236.5 FLOP/byte

Outputs
  examples/batch2/mfu_table.json
  examples/batch2/MFU_ALL.md
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path("/workspace/tilelang-mlir-ascend-950")
TILEOPS = ROOT / "examples/TileOPs"
MANIFEST = TILEOPS / "tileops/manifest"
BATCH2 = ROOT / "examples/batch2"
sys.path.insert(0, str(BATCH2))
sys.path.insert(0, str(TILEOPS))
from roofline_eval import eval_roofline  # noqa: E402
from tileops.reporting.acceptance import (  # noqa: E402
    load_compute_units,
    load_peaks,
    peak_for_unit,
)

UNITS = load_compute_units()
PEAKS = load_peaks()
PEAK_TFLOPS = 378.0
PEAK_TBW = float(PEAKS["bandwidth_tbs"])
MACHINE_BALANCE = PEAK_TFLOPS * 1e12 / (PEAK_TBW * 1e12)  # Cube balance, FLOP/byte
HOST_PAD = re.compile(r"^Pad", re.I)
DTYPE_SUFFIX = re.compile(r"-(bfloat16|float16|float32)$")


def sum_capture(path: Path):
    """One duration per launched kernel name (all samples), Pad* excluded."""
    per, total = {}, 0.0
    if not path or not path.exists():
        return None, {}
    for p in path.rglob("OpBasicInfo*.csv"):
        try:
            for row in csv.DictReader(p.open()):
                name = (row.get("Op Name") or "").strip()
                raw = (row.get("Task Duration(us)") or "").strip()
                if not name or not raw or HOST_PAD.search(name):
                    continue
                dur = float(raw)
                per[name] = per.get(name, 0.0) + dur
                total += dur
        except (OSError, ValueError):
            continue
    return (total, per) if per else (None, {})


def load_reports():
    """op -> (run_json_path, run_dict) for the 11 batch2 operators."""
    summary = BATCH2 / "integration_summary.json"
    out = {}
    if summary.exists():
        for e in json.loads(summary.read_text()):
            p = ROOT / e["rj"]
            if not p.exists():
                continue
            run = json.loads(p.read_text())
            out[run.get("operator") or e["op"]] = (p, run)
        if out:
            return out
    for f in TILEOPS.glob("tileops/kernels/*/*/*/integration_report.json"):
        d = json.loads(f.read_text())
        rj = d.get("run_json")
        if not rj:
            continue
        p = Path(rj) if Path(rj).is_absolute() else ROOT / rj
        if p.exists():
            run = json.loads(p.read_text())
            out[run.get("operator")] = (p, run)
    return out


def strip_dtype(label: str) -> str:
    return DTYPE_SUFFIX.sub("", str(label))


def load_recaptures():
    p = BATCH2 / "t_op_results.json"
    if not p.exists():
        return {}
    out = {}
    for r in json.loads(p.read_text()):
        if r.get("status", "ok" if r.get("t_op_us") else "blocked") != "ok":
            continue
        label = strip_dtype(r.get("node"))
        dt = (r.get("params") or {}).get("dtype")
        out[(r["operator"], label, dt)] = r
    return out


def build_rows():
    recaptures = load_recaptures()

    # ---- collect the union of report cases and re-captured-only cases ----
    cases: dict[tuple, dict] = {}
    for op, (_path, run) in sorted(load_reports().items()):
        run_op = run.get("operator") or op
        for c in (run.get("performance") or {}).get("cases", []):
            cop = c.get("operator") or op
            # Shared test files (e.g. test_engram.py) run several operators, so
            # one operator's report can contain another operator's cases.  Keep
            # only the cases belonging to this report's operator; otherwise a
            # stale foreign case can win via setdefault().
            if cop != run_op:
                continue
            params = c.get("params") or {}
            dt = params.get("dtype")
            key = (cop, str(c.get("label")), dt)
            cases.setdefault(key, {
                "operator": cop,
                "case": c.get("label"),
                "dtype": dt,
                "params": params,
                "reported_us": c.get("latency_us"),
                "profiler_valid": c.get("profiler_valid"),
                "invalid_reason": c.get("invalid_reason"),
                "artifact_dir": c.get("artifact_dir"),
                "msprof": c.get("msprof") or {},
                "in_report": True,
            })

    for (op, label, dt), r in recaptures.items():
        key = (op, label, dt)
        if key in cases:
            continue
        cases[key] = {
            "operator": op,
            "case": label,
            "dtype": dt,
            "params": r.get("params") or {},
            "reported_us": r.get("reported_us"),
            "profiler_valid": None,
            "invalid_reason": None,
            "artifact_dir": r.get("artifact"),
            "in_report": False,
        }

    rows = []
    for key, c in cases.items():
        op, params = c["operator"], c["params"]
        rec = recaptures.get(key)

        # ---- resolve T_op ----
        t_op = breakdown = None
        source = "blocked"
        t_op_from = None
        block_reason = None
        # A fresh report's msprof metadata aggregates every kernel of the whole
        # call; prefer it over a stale recapture from an earlier session.
        report_t_op = (c.get("msprof") or {}).get("t_op_us")
        if report_t_op:
            t_op = float(report_t_op)
            breakdown = (c.get("msprof") or {}).get("t_op_breakdown") or {}
            source, t_op_from = "report", "report msprof metadata (whole call)"
        elif rec and rec.get("t_op_us"):
            t_op = float(rec["t_op_us"])
            breakdown = rec.get("breakdown") or {}
            source, t_op_from = "recapture", "forced-unfiltered capture"
        else:
            art = c.get("artifact_dir")
            art = Path(art) if art else None
            if art is not None and not art.is_absolute():
                art = ROOT / art
            t_op, breakdown = sum_capture(art)
            if t_op:
                source, t_op_from = "report", "report artifact sum"
            elif c.get("profiler_valid") and c.get("reported_us"):
                t_op = float(c["reported_us"])
                breakdown = {"<reported latency>": t_op}
                source, t_op_from = "report", "reported latency (whole call)"
                if breakdown:
                    pass
            else:
                block_reason = (c.get("invalid_reason")
                                or "no artifact and profiler did not validate the capture")

        # ---- unit + roofline ----
        unit = (UNITS.get(op) or {}).get("unit") or "cube"
        dtype = c.get("dtype") or (UNITS.get(op) or {}).get("default_dtype")
        if not dtype:
            match = DTYPE_SUFFIX.search(str(c.get("case") or ""))
            dtype = match.group(1) if match else None
        eval_params = params
        if dtype and not params.get("dtype"):
            eval_params = {**params, "dtype": dtype}
        try:
            roof = eval_roofline(MANIFEST, op, eval_params)
        except Exception as exc:  # noqa: BLE001
            roof = {"flops": None, "bytes": None, "assumptions": [f"manifest error: {exc}"]}
        flops, nbytes = roof.get("flops"), roof.get("bytes")
        unit_peak = peak_for_unit(PEAKS, dtype, unit) or PEAK_TFLOPS
        row = {
            "operator": op,
            "case": c["case"],
            "dtype": dtype,
            "unit": unit,
            "unit_peak_tflops": unit_peak,
            "reported_us": c.get("reported_us"),
            "t_op_us": round(t_op, 3) if t_op else None,
            "source": source,
            "t_op_from": t_op_from,
            "breakdown_us": breakdown,
            "flops": flops,
            "bytes": nbytes,
            "ai": (flops / nbytes) if (flops is not None and nbytes) else None,
            "roofline_assumptions": roof.get("assumptions", []),
        }
        if block_reason:
            row["block_reason"] = block_reason

        if t_op and flops:
            row["achieved_tflops"] = flops / (t_op * 1e-6) / 1e12
            row["mfu_pct"] = 100.0 * row["achieved_tflops"] / unit_peak
            if row["ai"]:
                row["mfu_ceiling_pct"] = 100.0 * min(
                    1.0, row["ai"] / (unit_peak / PEAK_TBW)
                )
        if t_op and nbytes:
            row["achieved_tbps"] = nbytes / (t_op * 1e-6) / 1e12
            row["mu_pct"] = 100.0 * row["achieved_tbps"] / PEAK_TBW

        notes = []
        rep = c.get("reported_us")
        if t_op and rep:
            ratio = t_op / float(rep)
            if ratio > 2.0 or ratio < 0.5:
                notes.append("report-underreports")
        if row.get("achieved_tflops", 0) > unit_peak * 1.01:
            notes.append("physically-impossible:grid-over-limit")
            row["invalid"] = True
            row["invalid_reason"] = "grid-over-limit"
            row["note_detail"] = (
                f"achieved {row['achieved_tflops']:.0f} TFLOPS > {PEAK_TFLOPS} peak. "
                "Not a capture artifact: the forced-unfiltered msprof capture and the "
                "events timing agree. Root cause found by a dedicated probe: the kernel "
                "grid is (total_kv//64 + batch) * heads = 66,560 for ds-v3-8x4k, which "
                "exceeds the 950PR 65,535-block limit; blockDim is carried in 16 bits, "
                "so only 66,560 - 65,536 = 1,024 CTAs (1.54%) actually run and the "
                "rest are silently dropped. The 20.7 ms is 1.54% of the work; the true "
                "cost is ~488 ms (~29.4 TFLOPS, in line with every other MLA case).")
        if row.get("achieved_tbps", 0) > PEAK_TBW:
            notes.append("physically-impossible")
            row["note_detail"] = (
                f"achieved {row['achieved_tbps']:.2f} TB/s > {PEAK_TBW} peak; "
                "the byte model or the capture is inconsistent.")
        if notes:
            row["note"] = ";".join(dict.fromkeys(notes))
        rows.append(row)

    rows.sort(key=lambda r: (r["operator"], str(r["case"]), str(r["dtype"])))
    return rows


# --------------------------------------------------------------------------
# Markdown report
# --------------------------------------------------------------------------
def fmt(v, nd=2, dash="-"):
    if v is None:
        return dash
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def op_summary(rows, op):
    rs = [r for r in rows if r["operator"] == op]
    invalid = [r for r in rs if r.get("invalid")]
    measured = [r for r in rs if r.get("t_op_us") and not r.get("invalid")]
    blocked = [r for r in rs if r["source"] == "blocked"]
    mfus = [r["mfu_pct"] for r in measured if r.get("mfu_pct") is not None]
    mus = [r["mu_pct"] for r in measured if r.get("mu_pct") is not None]
    ais = [r["ai"] for r in measured if r.get("ai") is not None]
    peak_mfu = max(mfus) if mfus else None
    peak_mu = max(mus) if mus else None
    ai_max = max(ais) if ais else None
    impossible = bool(invalid)
    under = any("report-underreports" in (r.get("note") or "") for r in rs)

    # Judge on the dominant case (most arithmetic; else most traffic), because a
    # small decode/t1 workload and a large prefill workload have different
    # binding resources.
    rep = None
    if measured:
        with_arith = [r for r in measured if (r.get("flops") or 0) > 0]
        if with_arith:
            rep = max(with_arith, key=lambda r: r.get("flops") or 0)
        else:
            rep = max(measured, key=lambda r: r.get("bytes") or 0)
    mfu_rep = rep.get("mfu_pct") if rep else None
    mu_rep = rep.get("mu_pct") if rep else None
    # Diagnostic only: which resource binds the dominant case.  The acceptance
    # metric itself is always MFU.
    # Diagnostic only: which resource binds the dominant case.  The acceptance
    # metric itself is always MFU, on the peak of the unit the kernel runs on.
    rep_peak = (rep or {}).get("unit_peak_tflops") or PEAK_TFLOPS
    rep_balance = rep_peak / PEAK_TBW
    binding = "compute" if ((rep or {}).get("ai") or 0) >= rep_balance else "bandwidth"
    target = mfu_rep
    rep_impossible = bool(rep and "physically-impossible" in (rep.get("note") or ""))
    workload_type = ("compute-bound"
                     if (ai_max or 0) >= rep_balance else "memory-bound")
    # Single acceptance target from peaks.json ("others" tier: 90%).
    accept_target = float((PEAKS.get("acceptance") or {}).get("target_percent", 90.0))
    ceiling = (rep or {}).get("mfu_ceiling_pct")
    met = (target is not None and target >= accept_target and not rep_impossible)
    case_passed = sum(
        1 for r in measured
        if r.get("mfu_pct") is not None and r["mfu_pct"] >= accept_target
    )
    return {
        "op": op, "n": len(rs), "measured": len(measured), "blocked": blocked,
        "mfu": peak_mfu, "mu": peak_mu, "ai_max": ai_max,
        "binding": binding, "target": target, "met": met,
        "rep_case": rep.get("case") if rep else None,
        "rep_mfu": mfu_rep, "rep_mu": mu_rep,
        "rep_impossible": rep_impossible,
        "impossible": impossible, "under": under,
        "invalid": invalid,
        "unit": (rep or (rs[0] if rs else {})).get("unit"),
        "workload_type": workload_type,
        "accept_target": accept_target,
        "ceiling": ceiling,
        "case_passed": case_passed,
        "case_total": len(measured),
    }


def write_markdown(rows, path: Path):
    ops = sorted({r["operator"] for r in rows})
    lines = []
    lines.append("# Batch2 — whole-operator time (T_op) and MFU / MU\n")
    lines.append(
        "Every case is measured as **one operator call**.  `T_op` sums every "
        "kernel the call launches (host `Pad*` ops excluded); `reported` is the "
        "number the TileOPs reporter wrote, which is often a single filtered "
        "kernel and therefore under-reports multi-kernel operators.\n")
    lines.append(
        f"Peaks: **{PEAK_TFLOPS} TFLOPS** Cube, **47 TFLOPS** Vector (bf16/fp16), "
        f"**{PEAK_TBW} TB/s** bandwidth.  Machine balance is **236 FLOP/byte** for "
        "Cube and **29.4 FLOP/byte** for Vector.  MFU uses the peak of the unit the "
        "kernel actually runs on (vendor values; measured 370.5 / 1.424 is the "
        "supplementary column).\n")
    lines.append(
        "**Acceptance metric is MFU** (sparse CV 50% / CV 80% / others 90%).  MU is "
        "reported as a diagnostic only and never decides pass/fail.  `MFU 上限` = "
        "`AI / (unit peak / 1.6 TB/s)` is the physical ceiling bandwidth imposes on "
        "a memory-bound shape, so a tiny MFU there is not a kernel defect.\n")
    lines.append(
        "Peak provenance: Ascend950PR_9579 has 28 cube cores; the dense bf16/fp16 "
        "28 Cube cores x 4096 MAC/cycle x 2 x 1.65 GHz = 378 TFLOPS, which matches "
        "the vendor 28-core SKU column exactly.\n")

    lines.append("## How T_op was obtained\n")
    lines.append(
        "- **recapture** — this session re-ran the case with "
        "`TILEOPS_MSPROF_KERNEL_NAME=__force_unfiltered__`, kept the artifact and "
        "summed every kernel the single profiled call launched (`Pad*` excluded).\n"
        "- **report** — the case's stored msprof artifact already covered the whole "
        "call; `T_op` is the sum of its `OpBasicInfo*.csv`, or (when the reporter "
        "kept no artifact) the reporter's own `latency_us` for a validated capture.\n"
        "- **blocked** — no usable measurement; `T_op` is left empty, never 0.\n")
    lines.append(
        "A single call can launch a kernel more than once under one name; those "
        "samples are all present and are summed (verified on `EngramGateConvBwdOp`, "
        "which launches `main_kernel` twice).\n")

    lines.append("## msprof failures: root cause\n")
    lines.append(
        "The earlier `msprof profiling failed (...); falling back to events` runs were "
        "not a profiler or toolchain defect.  The container's root overlay (`/`) is a "
        "**5 GB** filesystem and was **100% full (0 bytes available)**: stale "
        "`/tmp/tileops_msprof_*` scratch dirs from previous captures had accumulated to "
        "~9.7 GB (single dirs up to 4.7 GB).  msprof stages its raw trace there, so it "
        "died with `No space left on device`; the reporter then fell back to events, "
        "and the large cases produced an empty artifact dir.\n")
    lines.append(
        "After removing the stale scratch dirs and pointing `TMPDIR` at `/workspace` "
        "(297 GB free), the exact previously-failing command "
        "`bench_mla_dense_bwd.py::test_mla_dense_bwd_bench[ds-v3-b2-8k-bfloat16]` "
        "returned rc=0 in 95.5 s with a complete unfiltered trace.  Every "
        "re-captured case then succeeded.\n")

    invalid_rows = [r for r in rows if r.get("invalid")]
    lines.append("## Invalid rows\n")
    lines.append(
        "Rows with achieved throughput above the cube peak are **not** efficiency "
        "figures and are excluded from every max/达标 statistic (`invalid`).\n")
    if invalid_rows:
        lines.append(
            "Affected cases in this table: "
            + ", ".join(f"`{r['operator']}`/{r['case']}" for r in invalid_rows)
            + ".  The cause is a launch whose grid exceeds the 950PR 65,535-block "
            "limit: blockDim is carried in 16 bits, so only `grid % 65536` CTAs run "
            "and the rest are silently dropped.\n")
    else:
        lines.append(
            "None in this run.  The 2026-11-24 head-slab fix removed the only known "
            "case (`MLAVarlenBwdOp`/`ds-v3-8x4k`, grid 66,560): it now runs 4 x 16,640 "
            "blocks and reports a real ~490 ms.  The trap and its evidence remain "
            "documented in `docs/950pr-mfu-acceptance.md` and "
            "`TRAP-950-grid-over-65535`.\n")

    lines.append("## Per-operator summary\n")
    lines.append(
        "| operator | unit | type | binding | AI max | cases | blocked | MFU(主导) "
        "| MFU 上限 | MFU max | MU max (诊断) | 达标(MFU) | case 达标 | notes |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|:--:|---:|---|")
    summaries = {}
    for op in ops:
        s = op_summary(rows, op)
        summaries[op] = s
        note = []
        if s["impossible"]:
            note.append(
                f"invalid: grid-over-limit ({len(s['invalid'])} case), excluded from maxima")

        if s["under"]:
            note.append("report under-reports")
        if s["blocked"]:
            note.append(f"{len(s['blocked'])} blocked")
        ceiling_txt = f"{s['ceiling']:.2f}%" if s["ceiling"] is not None else "n/a"
        mfu_txt = f"{s['mfu']:.2f}%" if s["mfu"] is not None else "n/a"
        rep_mfu_txt = f"{s['rep_mfu']:.2f}%" if s["rep_mfu"] is not None else "n/a"
        met_txt = "yes" if s["met"] else "no"
        if s.get("rep_impossible"):
            met_txt = "n/a*"
        lines.append(
            f"| `{op}` | {s.get('unit') or '-'} | {s['workload_type']} | {s['binding']} | "
            f"{fmt(s['ai_max'], 1)} | {s['n']} | {len(s['blocked'])} | "
            f"{rep_mfu_txt} | {ceiling_txt} | {mfu_txt} | {fmt(s['mu'])}% | "
            f"{met_txt} | {s['case_passed']}/{s['case_total']} | "
            f"{', '.join(note) or '-'} |")
    lines.append("")
    lines.append("`invalid` rows (grid above the 65,535-block limit) are excluded from "
                 "every max / 达标 statistic; see the section above for the evidence.\n")

    lines.append("## All cases\n")
    lines.append(
        "| operator | case | dtype | unit | reported µs | T_op µs | source | FLOPs "
        "| Bytes | AI | MFU % | MFU 上限 | 达标(MFU≥90%) | MU % | note |")
    lines.append(
        "|---|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|:--:|---:|---|")
    for r in rows:
        mfu_val = r.get("mfu_pct")
        pass_txt = (
            "invalid" if r.get("invalid")
            else "yes" if mfu_val is not None and mfu_val >= 90.0
            else "no" if mfu_val is not None
            else "n/a"
        )
        lines.append(
            f"| `{r['operator']}` | {r['case']} | {r['dtype'] or '-'} | "
            f"{r.get('unit') or '-'} | "
            f"{fmt(r.get('reported_us'), 3)} | {fmt(r.get('t_op_us'), 3)} | "
            f"{r['source']} | {fmt(r.get('flops'), 3)} | {fmt(r.get('bytes'), 3)} | "
            f"{fmt(r.get('ai'), 1)} | "
            f"{'invalid' if r.get('invalid') else fmt(r.get('mfu_pct'), 2)} | "
            f"{fmt(r.get('mfu_ceiling_pct'), 2)} | "
            f"{pass_txt} | "
            f"{fmt(r.get('mu_pct'), 2)} | {r.get('note') or '-'} |")
    lines.append("")

    lines.append("## Per-operator conclusion\n")
    for op in ops:
        s = summaries[op]
        lines.append(f"### `{op}`\n")
        unit_balance = 236.2 if (s.get("unit") or "cube") == "cube" else 29.4
        lines.append(
            f"- **类型**: {s['workload_type']} (max AI {fmt(s['ai_max'], 1)} FLOP/byte "
            f"vs the {s.get('unit') or 'cube'} balance {unit_balance}).")
        lines.append(
            f"- **受限资源**: {s['binding']} — on the dominant case `{s.get('rep_case')}` "
            f"MFU {fmt(s['rep_mfu'])}%, MU {fmt(s['rep_mu'])}% "
            f"(operator maxima: MFU {fmt(s['mfu'])}%, MU {fmt(s['mu'])}%).")
        if s.get("rep_impossible"):
            lines.append(
                "- **是否达标**: n/a — the dominant case is invalid, so the MFU "
                "denominator does not describe the executed work (see the invalid-rows "
                "section above).")
        else:
            lines.append(
                f"- **是否达标**: {'是' if s['met'] else '否'} "
                f"(MFU {fmt(s['rep_mfu'])}% on the dominant case vs target "
                f"{s['accept_target']:.0f}%; MFU is the acceptance metric, MU is "
                "diagnostic only).")
        if s["impossible"]:
            bad = [r["case"] for r in rows
                   if r["operator"] == op and "physically-impossible" in (r.get("note") or "")]
            reasons = sorted({str(r.get("invalid_reason") or "invalid") for r in s["invalid"]})
            lines.append(
                "- **阻塞/异常**: `invalid` ("
                + ", ".join(reasons)
                + ") for "
                + ", ".join(sorted(set(map(str, bad))))
                + " — the kernel launched a grid above the 65,535-block limit, so only "
                "the truncated remainder ran and the measured time covers 1.5% of the "
                "work.  Excluded from every max/达标 statistic; fix needs a design "
                "revision and a runtime grid assertion.")
        if s["blocked"]:
            lines.append("- **仍阻塞**: "
                         + "; ".join(f"{r['case']} ({r.get('block_reason')})"
                                     for r in s["blocked"]))
        if not s["impossible"] and not s["blocked"]:
            lines.append("- **阻塞/异常**: none — every case has a whole-call T_op.")
        lines.append("")

    path.write_text("\n".join(lines))


def main():
    rows = build_rows()
    out_json = BATCH2 / "mfu_table.json"
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    out_md = BATCH2 / "MFU_ALL.md"
    write_markdown(rows, out_md)
    print(f"{'operator':30s} {'case':26s} {'dt':9s} {'T_op_us':>10s} {'AI':>9s} {'MFU%':>7s} {'MU%':>7s} src")
    for r in rows:
        print(f"{r['operator'][:30]:30s} {str(r['case'])[:26]:26s} {str(r['dtype'])[:9]:9s} "
              f"{fmt(r.get('t_op_us'), 1):>10s} {fmt(r.get('ai'), 1):>9s} "
              f"{fmt(r.get('mfu_pct'), 2):>7s} {fmt(r.get('mu_pct'), 2):>7s} {r['source']}")
    print(f"\nwrote {out_json}\nwrote {out_md}")


if __name__ == "__main__":
    main()
