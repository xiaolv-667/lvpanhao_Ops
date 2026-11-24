"""The existing TileOPs scaffold and integration contract, shared by providers."""

from __future__ import annotations

import ast
import json
import os
import re
from pathlib import Path


def _read_text(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except (OSError, UnicodeDecodeError):
        return None


def _fail(rule_id: str, file: str, message: str) -> dict:
    return {"rule_id": rule_id, "file": file, "message": message}


_warn = _fail


def meta_lint(meta_path: str, repo_root: str):
    """Stage 0 gate: .migration_meta.json schema + TileOPs 7-file scaffold."""
    failures, warnings = [], []
    raw = _read_text(meta_path)
    if raw is None or not raw.strip():
        failures.append(
            _fail("S0-META", meta_path, ".migration_meta.json 不存在或为空")
        )
        return failures, warnings
    try:
        meta = json.loads(raw)
    except json.JSONDecodeError as exc:
        failures.append(
            _fail("S0-META", meta_path, f".migration_meta.json 非法 JSON：{exc}")
        )
        return failures, warnings

    if not isinstance(meta, dict):
        return [
            _fail("S0-META", meta_path, ".migration_meta.json 必须是 JSON 对象")
        ], warnings
    for field in ("op_slug", "family"):
        if not isinstance(meta.get(field), str) or not meta[field].strip():
            failures.append(
                _fail(
                    "S0-META-FIELD",
                    meta_path,
                    f".migration_meta.json 缺有效字段 {field}",
                )
            )
    funcs = meta.get("extracted_functions")
    if (
        not isinstance(funcs, list)
        or not funcs
        or any(not isinstance(name, str) or not name.strip() for name in funcs)
    ):
        failures.append(
            _fail("S0-META-FUNC", meta_path, "extracted_functions 必须是非空函数名数组")
        )
    if failures:
        return failures, warnings

    family = meta.get("family")
    op_slug = meta.get("op_slug")
    if family and op_slug:
        checks = [
            ("S0-SCAFFOLD", f"tileops/manifest/{family}.yaml"),
            ("S0-SCAFFOLD", f"tileops/workloads/{family}.py"),
            ("S0-SCAFFOLD", f"tileops/kernels/{family}/{op_slug}/{op_slug}.py"),
        ]
        for test_slug in [meta.get("test_slug")] if meta.get("test_slug") else []:
            checks.append(("S0-SCAFFOLD", f"tests/ops/test_{test_slug}.py"))
        for bench_slug in [meta.get("bench_slug")] if meta.get("bench_slug") else []:
            checks.append(("S0-SCAFFOLD", f"benchmarks/ops/bench_{bench_slug}.py"))
        for rule_id, rel in checks:
            abs_path = os.path.join(repo_root, "examples/TileOPs", rel)
            if not os.path.exists(abs_path):
                failures.append(
                    _fail(
                        rule_id,
                        abs_path,
                        f"TileOPs 脚手架文件缺失：examples/TileOPs/{rel}",
                    )
                )
    return failures, warnings


WRAPPER_PERF_IMPORT_RE = re.compile(r"^\s*#\s*(?:from|import)\s+.*perf_opt", re.M)


def integration_lint(migration_state: dict, repo_root: str, *, integration_mode):
    """Stage 5 gate: integration package, wrapper, and single-op report.

    ``migration_state`` is the parsed .migration_state.json (needs meta_path,
    family, op_slug, functions).
    """
    failures, warnings = [], []
    family = migration_state.get("family")
    op_slug = migration_state.get("op_slug")
    meta_path = migration_state.get("meta_path")
    if not (family and op_slug):
        failures.append(
            _fail("S5-STATE", str(meta_path), ".migration_state.json 缺 family/op_slug")
        )
        return failures, warnings
    pkg_dir = os.path.join(
        repo_root,
        f"examples/TileOPs/tileops/kernels/{family}/{op_slug}/{op_slug}_kernel",
    )
    if not os.path.isdir(pkg_dir):
        failures.append(_fail("S5-PKG", pkg_dir, "集成包目录不存在"))
        return failures, warnings
    for required in ("__init__.py", "integration_log.md", "integration_report.json"):
        if not os.path.exists(os.path.join(pkg_dir, required)):
            failures.append(
                _fail(
                    "S5-PKG", os.path.join(pkg_dir, required), f"集成包缺少 {required}"
                )
            )
    for func in migration_state.get("functions") or {}:
        for suffix in (".py", "_DESIGN.md"):
            path = os.path.join(pkg_dir, f"{func}{suffix}")
            if not os.path.exists(path):
                failures.append(
                    _fail("S5-FUNCS", path, f"集成包缺少函数 {func} 的 {suffix} 文件")
                )
    wrapper = os.path.join(
        repo_root, f"examples/TileOPs/tileops/kernels/{family}/{op_slug}/{op_slug}.py"
    )
    wtext = _read_text(wrapper)
    source_modes = set()
    for func in migration_state.get("functions") or {}:
        kernel_path = os.path.join(pkg_dir, f"{func}.py")
        kernel_text = _read_text(kernel_path)
        if kernel_text is None:
            continue
        try:
            source_modes.add(integration_mode(kernel_path))
        except (SystemExit, ValueError, SyntaxError) as exc:
            failures.append(_fail("S5-ASCEND-MODE", kernel_path, str(exc)))
    if len(source_modes) > 1:
        failures.append(
            _fail(
                "S5-ASCEND-MODE",
                pkg_dir,
                "同一 Kernel class 集成了不同编程模式，需逐调用限定模式",
            )
        )
    if wtext is None:
        failures.append(_fail("S5-WRAPPER", wrapper, "wrapper 文件不存在或不可读"))
    else:
        try:
            wrapper_tree = ast.parse(wtext)
            classes = [
                node
                for node in wrapper_tree.body
                if isinstance(node, ast.ClassDef)
                and any(
                    isinstance(base, ast.Name) and base.id == "Kernel"
                    for base in node.bases
                )
            ]
            class_modes = [
                stmt.value.value
                for cls in classes
                for stmt in cls.body
                if isinstance(stmt, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "ascend_mode"
                    for target in stmt.targets
                )
                and isinstance(stmt.value, ast.Constant)
            ]
            if (
                len(classes) != 1
                or len(class_modes) != 1
                or set(class_modes) != source_modes
            ):
                failures.append(
                    _fail(
                        "S5-ASCEND-MODE",
                        wrapper,
                        "wrapper 的 Kernel.ascend_mode 与集成 kernel 模式不一致",
                    )
                )
        except SyntaxError:
            failures.append(
                _fail(
                    "S5-ASCEND-MODE",
                    wrapper,
                    "wrapper Python 语法错误，无法检查编程模式",
                )
            )
        if "perf_opt" not in wtext:
            failures.append(
                _fail(
                    "S5-WRAPPER-SWITCH",
                    wrapper,
                    "wrapper 未生成 baseline/perf_opt 双 import 切换块（未检测到 perf_opt 引用）",
                )
            )
        elif not WRAPPER_PERF_IMPORT_RE.search(wtext):
            warnings.append(
                _warn(
                    "S5-WRAPPER-SWITCH",
                    wrapper,
                    "wrapper 含 perf_opt 引用但未检测到注释态的 perf_opt import（切换块形态请复核）",
                )
            )
    report_ref_path = os.path.join(pkg_dir, "integration_report.json")
    report_ref_text = _read_text(report_ref_path)
    if report_ref_text is None:
        return failures, warnings
    try:
        report_ref = json.loads(report_ref_text)
    except json.JSONDecodeError as exc:
        failures.append(
            _fail("S5-REPORT-REF", report_ref_path, f"报告引用不是合法 JSON：{exc}")
        )
        return failures, warnings
    run_rel = report_ref.get("run_json") if isinstance(report_ref, dict) else None
    if not isinstance(run_rel, str) or not run_rel.endswith("/run.json"):
        failures.append(
            _fail(
                "S5-REPORT-REF",
                report_ref_path,
                "run_json 必须是仓库相对的报告 run.json 路径",
            )
        )
        return failures, warnings
    report_root = os.path.realpath(
        os.path.join(repo_root, "examples", "TileOPs", "reports", "tileops")
    )
    run_path = os.path.realpath(os.path.join(repo_root, run_rel))
    if (
        not run_rel.startswith("examples/TileOPs/reports/tileops/")
        or os.path.commonpath((report_root, run_path)) != report_root
    ):
        failures.append(
            _fail(
                "S5-REPORT-REF", report_ref_path, "run_json 路径不在 TileOPs 报告目录内"
            )
        )
        return failures, warnings
    run_text = _read_text(run_path)
    if run_text is None:
        failures.append(_fail("S5-REPORT", run_path, "报告 run.json 不存在或不可读"))
        return failures, warnings
    try:
        run = json.loads(run_text)
    except json.JSONDecodeError as exc:
        failures.append(
            _fail("S5-REPORT", run_path, f"报告 run.json 不是合法 JSON：{exc}")
        )
        return failures, warnings
    if not isinstance(run, dict):
        failures.append(_fail("S5-REPORT", run_path, "报告 run.json 顶层必须为对象"))
        return failures, warnings

    meta_file = (
        meta_path
        if os.path.isabs(str(meta_path))
        else os.path.join(repo_root, str(meta_path))
    )
    try:
        meta = json.loads(_read_text(meta_file) or "")
    except (json.JSONDecodeError, TypeError):
        meta = None
    if not isinstance(meta, dict) or not meta.get("op_name"):
        failures.append(
            _fail("S5-REPORT-META", meta_file, "迁移元数据缺少 manifest 算子名 op_name")
        )
        return failures, warnings
    if run.get("operator") != meta["op_name"]:
        failures.append(
            _fail("S5-REPORT-OP", run_path, "报告算子与迁移元数据 op_name 不一致")
        )
    metadata = run.get("metadata")
    summary = run.get("summary")
    if not isinstance(metadata, dict) or not isinstance(summary, dict):
        failures.append(_fail("S5-REPORT", run_path, "报告缺少 metadata/summary 对象"))
        return failures, warnings
    expected_test = (
        meta.get("test_path") or f"tests/ops/test_{meta.get('test_slug')}.py"
    )
    expected_bench = (
        meta.get("bench_path") or f"benchmarks/ops/bench_{meta.get('bench_slug')}.py"
    )
    if (
        metadata.get("test_file") != expected_test
        or metadata.get("benchmark_file") != expected_bench
    ):
        failures.append(
            _fail(
                "S5-REPORT-TARGET",
                run_path,
                "报告 test/benchmark 目标与迁移元数据不一致",
            )
        )
    if metadata.get("prof_mode_requested") != "msprof":
        failures.append(_fail("S5-REPORT-PROF", run_path, "报告未请求 msprof 模式"))
    correctness_passed = summary.get("correctness_passed") is True
    benchmark_requested = summary.get("benchmark_requested") is True
    benchmark_passed = summary.get("benchmark_passed") is True
    status = run.get("status")
    correctness_tests = summary.get("correctness_tests")
    if (
        not correctness_passed
        or not isinstance(correctness_tests, int)
        or isinstance(correctness_tests, bool)
        or correctness_tests < 1
    ):
        failures.append(
            _fail("S5-REPORT-TEST", run_path, "报告缺少通过的全量正确性用例")
        )
    if not benchmark_requested:
        failures.append(_fail("S5-REPORT-BENCH", run_path, "报告未实际运行 benchmark"))
    if status not in ("passed", "partial") or (status == "passed") != benchmark_passed:
        failures.append(
            _fail("S5-REPORT-STATUS", run_path, "报告状态与 benchmark 结论不一致")
        )
    if status == "partial" and correctness_passed and benchmark_requested:
        warnings.append(
            _warn(
                "S5-REPORT-BENCH",
                run_path,
                "正确性通过，但 benchmark 未得到有效结果；仅记录，不阻断集成",
            )
        )
    for name in ("report.md", "report.html"):
        path = os.path.join(os.path.dirname(run_path), name)
        if not os.path.isfile(path):
            failures.append(_fail("S5-REPORT-FILE", path, f"缺少 {name}"))
    sources = [wrapper, os.path.join(pkg_dir, "__init__.py")]
    sources.extend(
        os.path.join(pkg_dir, f"{func}.py")
        for func in migration_state.get("functions") or {}
    )
    if all(os.path.isfile(path) for path in sources) and os.path.getmtime(
        run_path
    ) < max(os.path.getmtime(path) for path in sources):
        failures.append(
            _fail(
                "S5-REPORT-STALE",
                run_path,
                "报告早于当前 wrapper/kernel 集成文件，需重新运行",
            )
        )
    return failures, warnings


def migration_mode(source_repo, operator):
    root = Path(source_repo)
    directories = ("tileops/manifest", "tests/ops", "benchmarks/ops")
    if not all((root / directory).is_dir() for directory in directories):
        return "plain"
    key = re.compile(f"^{re.escape(operator)}:", re.M)
    return (
        "harness"
        if any(
            key.search(_read_text(str(path)) or "")
            for path in (root / directories[0]).glob("*.yaml")
        )
        else "plain"
    )
