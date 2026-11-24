"""Non-NPU checks for the Stage 4 experiment runner template."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest


TEMPLATE = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "tilelang-op-optimize"
    / "references"
    / "run_experiments_template.py"
)


def _load_runner():
    spec = importlib.util.spec_from_file_location("stage4_runner_template", TEMPLATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_uses_only_inventory_tune_workloads(tmp_path, monkeypatch):
    runner = _load_runner()
    inventory = {
        "workloads": [
            {"kernel_id": "k1", "workload_id": "large", "kind": "tune"},
            {"kernel_id": "k1", "workload_id": "smoke-1m", "kind": "smoke"},
            {"kernel_id": "k2", "workload_id": "other", "kind": "tune"},
        ],
    }
    path = tmp_path / "workload_inventory.json"
    path.write_text(json.dumps(inventory), encoding="utf-8")
    monkeypatch.setattr(runner, "KERNEL_ID", "k1")
    monkeypatch.setattr(runner, "WORKLOAD_INVENTORY", str(path))
    assert runner.tune_workload_ids() == ["large"]
    assert "/" not in runner.safe_component("../smoke;rm -rf")
    assert (
        runner.build_bench_cmd("kernel.py", "shape with spaces")[-1]
        == "shape with spaces"
    )
    monkeypatch.setattr(sys, "argv", ["run_experiments.py", "--workload", "smoke-1m"])
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 2


def test_candidate_cannot_run_all_workloads(tmp_path, monkeypatch):
    runner = _load_runner()
    path = tmp_path / "workload_inventory.json"
    path.write_text(
        json.dumps(
            {
                "workloads": [
                    {"kernel_id": "k1", "workload_id": "shape-a", "kind": "tune"},
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "KERNEL_ID", "k1")
    monkeypatch.setattr(runner, "WORKLOAD_INVENTORY", str(path))
    monkeypatch.setattr(
        sys, "argv", ["run_experiments.py", "--phase", "candidate", "--all"]
    )
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 2


def test_runner_records_are_accepted_by_platform_gate(tmp_path, monkeypatch):
    from dataclasses import replace
    from test_statectl import _single_workload_artifacts, make_repo, op_dir, sc

    runner = _load_runner()
    from platform_context import resolve_platform

    repo = make_repo(tmp_path)
    task = op_dir(repo)
    context = resolve_platform("ascend-910b3", "npuir")
    runtime = {"device": {"name": "Ascend910B3"}, "toolchain_stamp": {"raw": "fixture"}}
    assert (
        sc(
            "init",
            "--dir",
            str(task),
            "--project",
            "proj",
            "--op",
            "t",
            "--scenario",
            "optimize",
            "--hardware",
            "ascend-910b3",
            "--toolchain",
            "npuir",
        )[0]
        == 0
    )
    state_file = task / ".stage_state.json"
    state = json.loads(state_file.read_text())
    _single_workload_artifacts(task)
    perf_dir = task / "perf_opt"
    records = perf_dir / "perf_records.jsonl"
    records.unlink()
    monkeypatch.chdir(perf_dir)
    monkeypatch.setattr(
        type(context), "verified", lambda self: replace(self, runtime=runtime)
    )
    monkeypatch.setattr(runner, "KERNEL_ID", "t.py::main")
    monkeypatch.setattr(runner, "run_l0", lambda *args: True)
    durations = iter((120.5, 100.0))
    monkeypatch.setattr(
        runner,
        "run_msprof",
        lambda *args: {
            "duration_us": next(durations),
            "msprof_raw_path": args[-1],
            "n": 1,
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_experiments.py", "--phase", "baseline", "--all", "--task-dir", str(task)],
    )
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 2 and not records.exists()

    state_file.write_text(json.dumps({**state, "platform_runtime": runtime}))
    for phase, candidate in (("baseline", "baseline"), ("final", "merged-best")):
        monkeypatch.setattr(
            runner,
            "EXPERIMENTS",
            [
                {
                    "file": "t.py",
                    "opt_id": candidate,
                    "parent_id": None if phase == "baseline" else "baseline",
                }
            ],
        )
        monkeypatch.setattr(
            sys,
            "argv",
            ["run_experiments.py", "--phase", phase, "--all", "--task-dir", str(task)],
        )
        assert runner.main() == 0
    rc, result = sc("gate", "4", "--dir", str(task))
    assert rc == 0, result

    rows = [json.loads(line) for line in records.read_text().splitlines()]
    rows[-1]["toolchain_id"] = "other"
    records.write_text("".join(json.dumps(row) + "\n" for row in rows))
    rc, result = sc("gate", "4", "--dir", str(task))
    assert rc == 1
    assert "S4-PERF-RECORDS-PLATFORM" in {f["rule_id"] for f in result["failures"]}
