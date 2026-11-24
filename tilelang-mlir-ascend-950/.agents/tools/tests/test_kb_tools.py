"""Tests for kb_lint.py (K-4 + ED-F) and kb_search.py (E-1/K-3)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOLS = REPO_ROOT / ".agents" / "tools"


def run_tool(name: str, *args: str) -> tuple[int, dict | str]:
    if not any(arg in args for arg in ("--hardware", "--toolchain", "--task-dir")):
        args = (*args, "--hardware", "ascend-910b2c", "--toolchain", "npuir")
    proc = subprocess.run(
        [sys.executable, str(TOOLS / name), *args],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(REPO_ROOT),
    )
    return proc.returncode, proc.stdout


def test_kb_lint_real_library_is_clean():
    rc, out = run_tool("kb_lint.py")
    assert rc == 0, out


def test_kb_lint_detects_missing_repro_and_bad_enum(tmp_path):
    pl = (
        tmp_path
        / ".agents"
        / "skills"
        / "tilelang-op-optimize"
        / "references"
        / "pattern-library"
    )
    pl.mkdir(parents=True)
    (pl / "INDEX.md").write_text("# index\n", encoding="utf-8")
    (pl / "traps-runtime.md").write_text(
        "---\n"
        "id: TRAP-test-entry\n"
        "kind: trap\n"
        "status: bogus\n"  # KB-FM-ENUM
        "origin_task: task-x\n"
        "toolchain: tilelang 0.1.2+abcdef123 + CANN 8.5.0 2026-09-10\n"
        "repro: repro/TRAP-test-entry.py\n"  # KB-REPRO-EXISTS (missing)
        "---\n"
        "### test entry\nbody\n",
        encoding="utf-8",
    )
    rc, out = run_tool("kb_lint.py", "--files", str(pl / "traps-runtime.md"))
    assert rc == 1
    assert "KB-FM-ENUM" in out
    assert "KB-REPRO-EXISTS" in out


def test_kb_lint_repro_must_not_reference_workspace(tmp_path):
    pl = (
        tmp_path
        / ".agents"
        / "skills"
        / "tilelang-op-optimize"
        / "references"
        / "pattern-library"
    )
    (pl / "repro").mkdir(parents=True)
    (pl / "INDEX.md").write_text("# index\n", encoding="utf-8")
    repro = pl / "repro" / "TRAP-coupled.py"
    repro.write_text(
        '"""[repro] TRAP-coupled\n\nFirst verified: 2026-09-10 test\n"""\n'
        "open('examples/proj/op/kernel.py')\n",
        encoding="utf-8",
    )
    (pl / "traps-runtime.md").write_text(
        "---\n"
        "id: TRAP-coupled\n"
        "kind: trap\n"
        "status: verified\n"
        "origin_task: task-x\n"
        "toolchain: tilelang 0.1.2+abcdef123 + CANN 8.5.0 2026-09-10\n"
        "repro: repro/TRAP-coupled.py\n"
        "---\n"
        "### coupled entry\nbody\n",
        encoding="utf-8",
    )
    rc, out = run_tool(
        "kb_lint.py", "--root", str(tmp_path), "--files", str(pl / "traps-runtime.md")
    )
    assert rc == 1
    assert "KB-REPRO-DECOUPLE" in out


def test_kb_search_returns_relevant_entries():
    rc, out = run_tool("kb_search.py", "fp16 golden opmath 精度", "--json")
    assert rc == 0
    data = json.loads(out)
    assert data["results"], "expected hits for fp16 golden opmath"
    ids = [r["id"] for r in data["results"]]
    assert any("lerp" in i or "opmath" in i or "fp16" in i for i in ids), ids


def test_kb_search_covers_all_knowledge_sources():
    rc, out = run_tool("kb_search.py", "--index")
    assert rc == 0
    assert "TRAP-load-nd2nz-strided" in out  # pattern-library traps
    assert "BP_run_state_bimodality" in out  # bottleneck-patterns
    assert "ALG-attention" in out  # algorithm-candidates
    assert "CG-2026-0004" in out  # capability-gaps
    assert "VP-2026-0042" in out  # queue (decided entry)


def test_kb_lint_flags_task_workspace_repro_in_queue(tmp_path):
    pl = (
        tmp_path
        / ".agents"
        / "skills"
        / "tilelang-op-optimize"
        / "references"
        / "pattern-library"
    )
    ev = tmp_path / ".agents" / "evolution"
    (pl / "repro").mkdir(parents=True)
    ev.mkdir(parents=True)
    (pl / "INDEX.md").write_text("# index\n", encoding="utf-8")
    (ev / "queue.md").write_text(
        "# queue\n## Pending\n"
        "## VP-2026-0099\n"
        "- type: P\n"
        "- title: t\n"
        "- repro: python examples/foo/bar.py --level all\n"
        "- status: pending\n"
        "- confirmations: 1/2\n"
        "## Decided\n",
        encoding="utf-8",
    )
    rc, out = run_tool("kb_lint.py", "--root", str(tmp_path))
    assert rc == 1
    assert "KB-QUEUE-REPRO" in out


def test_kb_lint_queue_inline_python_command_is_exempt(tmp_path):
    ev = tmp_path / ".agents" / "evolution"
    ev.mkdir(parents=True)
    (ev / "queue.md").write_text(
        "# queue\n## Pending\n"
        "## VP-2026-0098\n"
        "- type: R\n"
        "- repro: python3 -c \"import re; re.finditer('examples/x', s)\"\n"
        "- status: pending\n"
        "- confirmations: -/-\n"
        "## Decided\n",
        encoding="utf-8",
    )
    rc, out = run_tool("kb_lint.py", "--root", str(tmp_path))
    assert "KB-QUEUE-REPRO" not in out


def test_kb_stale_check_reports_and_never_modifies():
    rc, out = run_tool("kb_stale_check.py", "--json")
    assert rc == 0
    data = json.loads(out)
    assert "current_stamp" in data and "stale" in data
    # advisory tool: exit 0 even with stale entries


def test_stale_check_and_runner_share_saved_commit(tmp_path, monkeypatch, capsys):
    from argparse import Namespace
    from dataclasses import replace

    monkeypatch.syspath_prepend(str(TOOLS))
    import kb_stale_check
    import repro_runner
    from platform_context import resolve_platform, using_platform

    topic = tmp_path / "topic.md"
    topic.write_text(
        "---\nid: KB-matching\ntoolchain: tilelang 0.1.2+abcdef123\n---\n"
        "---\nid: KB-different\ntoolchain: tilelang 0.1.2+fedcba987\n---\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(kb_stale_check, "knowledge_globs", lambda: [str(topic)])
    platform = resolve_platform("ascend-910b3", "npuir")
    stamp = {
        "raw": "tilelang 0.1.2+abcdef123 + CANN 9.0.0",
        "tilelang_commit": None,
    }
    args = Namespace(stamp=None, include_unknown=True, json=True)

    for commit, stale, unknown_count in (
        (None, set(), 2),
        ("abcdef1234", {"KB-different"}, 0),
    ):
        runtime = {"toolchain_stamp": {**stamp, "tilelang_commit": commit}}
        assert platform.adapter.classify_stamp(
            stamp["raw"], runtime["toolchain_stamp"]
        ) == ("unknown" if commit is None else "fresh")
        with using_platform(replace(platform, runtime=runtime)):
            assert kb_stale_check.run(args) == 0
            report = json.loads(capsys.readouterr().out)
            assert report["unknown_count"] == unknown_count
            assert {entry["id"] for entry in report["stale"]} == stale
            assert repro_runner.stale_ids() == stale
