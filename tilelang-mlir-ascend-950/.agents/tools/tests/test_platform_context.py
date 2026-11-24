"""Independent model selection, task pinning and old-state compatibility."""

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import platform_context as pc
from design_calc_check import check_ub_budget
from test_statectl import (
    _make_harness_repo,
    _ready_stage5_gate,
    _write_stage5_report,
    sc,
)


def test_same_toolchain_uses_model_specific_budget():
    b2c = pc.resolve_platform("ascend-910b2c", "npuir")
    b3 = pc.resolve_platform("ascend-910b3", "npuir")
    assert type(b2c.adapter) is type(b3.adapter)
    assert b2c.adapter.precision_command("op.py", "L0") == b3.adapter.precision_command(
        "op.py", "L0"
    )
    assert b2c.adapter.profiler_args(
        "main", "output", 15, 5
    ) == b3.adapter.profiler_args("main", "output", 15, 5)
    design = "## 4.5 内存\n**UB**\n| buffer | 128KB |\n"
    assert check_ub_budget(design, b2c.design_budget())[0]["status"] == "pass"
    assert b3.design_budget()["ub_inflation"] is None
    assert pc.resolve_platform(state={}).design_budget()["ub_inflation"] == 1.7


def test_measurements_match_hardware_toolchain_stamp_and_workload(monkeypatch):
    source = pc.read_json
    measurement = {
        "hardware_id": "ascend-910b2c",
        "toolchain_id": "npuir",
        "toolchain_stamp": "fixture-build",
        "workload": "fixture",
        "metric": "ub_inflation_factor",
        "unit": "ratio",
        "value": 1.25,
        "source": "fixture",
    }
    monkeypatch.setattr(
        pc,
        "read_json",
        lambda path: (
            {"measurements": [measurement]}
            if path.name == "index.json"
            else source(path)
        ),
    )
    b2c = pc.resolve_platform("ascend-910b2c", "npuir")
    b3 = pc.resolve_platform("ascend-910b3", "npuir")
    assert b2c.measurements("fixture-build", "fixture") == [measurement]
    assert b2c.measurements("other-build", "fixture") == []
    assert b2c.measurements("fixture-build", "other-workload") == []
    assert b3.measurements("fixture-build", "fixture") == []


def test_probe_rejects_different_device_or_runtime(monkeypatch):
    context = pc.resolve_platform("ascend-910b3", "npuir")
    monkeypatch.setattr(
        type(context.adapter), "probe_device", lambda self: {"name": "Ascend910B2C"}
    )
    with pytest.raises(pc.PlatformError, match="active device reports"):
        context.probe()
    monkeypatch.setattr(
        type(context.adapter),
        "probe_device",
        lambda self: {"name": "Ascend910B3", "cores": {"cube": 20, "vector": 40}},
    )
    monkeypatch.setattr(
        type(context.adapter),
        "toolchain_stamp",
        lambda *a, **k: {
            "raw": "current",
            "tilelang": "1",
            "cann": "9",
            "compiler": "fixture",
        },
    )
    context = context.verified()
    assert context.runtime["device"]["cores"]["vector"] == 40
    monkeypatch.setattr(
        type(context.adapter),
        "toolchain_stamp",
        lambda *a, **k: {
            "raw": "updated",
            "tilelang": "1",
            "cann": "9",
            "compiler": "fixture",
        },
    )
    with pytest.raises(pc.PlatformError, match="differ from the task record"):
        context.probe()


def test_migration_children_inherit_selection(tmp_path):
    op_dir = tmp_path / "op"
    assert (
        sc(
            "migration",
            "init",
            "--dir",
            str(op_dir),
            "--op-slug",
            "op",
            "--family",
            "test",
            "--meta-path",
            "meta.json",
            "--functions",
            "func",
            "--hardware",
            "ascend-910b3",
            "--toolchain",
            "npuir",
        )[0]
        == 0
    )
    child = op_dir / "func"
    args = (
        "init",
        "--dir",
        str(child),
        "--project",
        "op",
        "--op",
        "func",
        "--scenario",
        "migration",
        "--migration-mode",
        "harness",
        "--task-dir",
        str(op_dir),
    )
    assert sc(*args, "--hardware", "ascend-910b2c")[0] == 1
    assert sc(*args)[0] == 0
    saved = json.loads((child / ".stage_state.json").read_text())
    aggregate = json.loads((op_dir / ".migration_state.json").read_text())
    assert saved["hardware_id"] == aggregate["hardware_id"] == "ascend-910b3"
    assert saved["toolchain_id"] == aggregate["toolchain_id"] == "npuir"
    assert saved["platform_config"] == aggregate["platform_config"]
    code, result = sc("platform", "--task-dir", str(child))
    assert code == 0, result
    assert result["platform"]["hardware_id"] == saved["hardware_id"]
    assert result["platform"]["toolchain_id"] == saved["toolchain_id"]
    assert (
        sc("platform", "--task-dir", str(child), "--hardware", "ascend-910b2c")[0] == 1
    )


def test_harness_scaffold_and_aggregate_state_handoff(tmp_path):
    repo, op_dir, meta = _make_harness_repo(tmp_path)
    directory = ("--dir", str(op_dir))
    platform = ("--hardware", "ascend-910b3", "--toolchain", "npuir")
    assert (
        sc(
            "init",
            *directory,
            "--project",
            "mop",
            "--op",
            "mop",
            "--scenario",
            "migration",
            "--migration-mode",
            "harness",
            "--stage-plan",
            "0",
            *platform,
        )[0]
        == 0
    )
    assert sc("start", "0", *directory)[0] == 0
    original_meta = meta.read_text()
    meta.write_text(
        json.dumps({**json.loads(original_meta), "extracted_functions": "funcA"})
    )
    rc, result = sc("gate", "0", *directory, "--meta", str(meta.relative_to(repo)))
    assert rc == 1 and "S0-META-FUNC" in {f["rule_id"] for f in result["failures"]}
    meta.write_text(original_meta)
    for relative in (
        "tileops/manifest/reduction.yaml",
        "tileops/workloads/reduction.py",
        "tileops/kernels/reduction/mop/mop.py",
        "tests/ops/test_mop.py",
        "benchmarks/ops/bench_mop.py",
    ):
        path = repo / "examples/TileOPs" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# scaffold\n")
    meta_arg = str(meta.relative_to(repo))
    assert sc("complete", "0", *directory, "--meta", meta_arg)[0] == 0
    assert (
        sc(
            "migration",
            "init",
            *directory,
            "--op-slug",
            "mop",
            "--family",
            "reduction",
            "--meta-path",
            str(meta),
            "--functions",
            "funcA",
            "funcB",
            *platform,
        )[0]
        == 0
    )
    aggregate_path = op_dir / ".migration_state.json"
    assert pc.task_state_path(op_dir) == aggregate_path
    assert sc("gate", "0", *directory)[0] == 0
    rc, result = sc("gate", "5", "--migration-dir", str(op_dir), *platform)
    assert rc == 1 and "S5-PKG" in {f["rule_id"] for f in result["failures"]}


def test_wrong_task_directory_and_malformed_state_are_rejected(tmp_path):
    task = tmp_path / "task"
    task.mkdir()
    selected = pc.resolve_platform("ascend-910b3", "npuir").selection
    path = task / ".stage_state.json"
    path.write_text(json.dumps(selected))
    code, result = sc("platform", "--task-dir", str(task / "typo"))
    assert code == 1 and result["errors"][0]["code"] == "E-PLATFORM"
    path.write_text(json.dumps({**selected, "platform_runtime": []}))
    code, result = sc("platform", "--task-dir", str(task))
    assert code == 1 and "invalid platform_runtime" in result["errors"][0]["message"]


def test_stage5_gate_reads_explicit_aggregate_and_report(tmp_path):
    repo, directory, package, _ = _ready_stage5_gate(tmp_path)
    _write_stage5_report(repo, package)
    path = directory / ".migration_state.json"
    selection = pc.resolve_platform("ascend-910b3", "npuir").selection
    path.write_text(json.dumps({**json.loads(path.read_text()), **selection}))
    args = ("gate", "5", "--migration-dir", str(directory))
    rc, result = sc(*args, "--hardware", "ascend-910b3", "--toolchain", "npuir")
    assert rc == 0, result
    rc, result = sc(*args, "--hardware", "ascend-910b2c", "--toolchain", "npuir")
    assert rc == 1 and result["errors"][0]["code"] == "E-PLATFORM"


def test_simulator_execution_context_preserves_platform_selection(monkeypatch):
    import shlex
    import sys

    monkeypatch.delenv("IS_SIMULATOR_ENV", raising=False)
    context = pc.resolve_platform("ascend-950dt-9582", "ascend")
    selection = context.selection
    assert context.prompt()["execution"]["precision_command"] == [
        sys.executable,
        "<kernel>",
        "--level",
        "<level>",
    ]

    monkeypatch.setenv("IS_SIMULATOR_ENV", "true")
    monkeypatch.setenv("soc_version", "Ascend950DT")
    simulated = pc.resolve_platform("ascend-950dt-9582", "ascend")
    execution = simulated.prompt()["execution"]
    assert simulated.selection == selection
    argv = [sys.executable, "/tmp/probe code.py", "--level", "L0"]
    launched = simulated.adapter.execution_command(argv, output_dir="profiles/probe")
    assert launched[:8] == [
        "cannsim",
        "record",
        "-s",
        "Ascend950DT",
        "-n",
        "0",
        "-o",
        "profiles/probe",
    ]
    assert shlex.split(launched[-1]) == argv
    assert (
        pc.resolve_platform("ascend-910b3", "npuir").adapter.execution_command(argv)
        == argv
    )
    assert execution["mode"] == "simulator"
    assert execution["working_directory"] == "<task_dir>"
    assert execution["prototype_command"][:8] == [
        "cannsim",
        "record",
        "-s",
        "Ascend950DT",
        "-n",
        "0",
        "-o",
        "<output_dir>",
    ]
    assert shlex.split(execution["prototype_command"][-1]) == [
        sys.executable,
        "<script>",
    ]
    assert shlex.split(execution["precision_command"][-1]) == [
        sys.executable,
        "<kernel>",
        "--level",
        "<level>",
    ]
    assert (
        pc.resolve_platform("ascend-910b3", "npuir").prompt()["execution"]["mode"]
        == "device"
    )
    assert pc.resolve_platform(state={}).prompt()["execution"]["mode"] == "device"


def test_child_can_reload_simulator_launch_instructions(tmp_path, monkeypatch):
    monkeypatch.setenv("IS_SIMULATOR_ENV", "true")
    monkeypatch.setenv("soc_version", "Ascend950DT")
    task = tmp_path / "task"
    task.mkdir()
    selection = pc.resolve_platform("ascend-950dt-9582", "ascend").selection
    (task / ".stage_state.json").write_text(json.dumps(selection))
    code, result = sc("platform", "--task-dir", str(task))
    assert code == 0, result
    execution = result["platform"]["execution"]
    assert execution["mode"] == "simulator"
    assert execution["prototype_command"][0] == "cannsim"
    assert execution["precision_command"][2:4] == ["-s", "Ascend950DT"]


def test_platform_selection_has_no_implicit_default():
    with pytest.raises(pc.PlatformError, match="toolchain explicitly"):
        pc.resolve_platform()
    with pytest.raises(pc.PlatformError, match="toolchain explicitly"):
        pc.resolve_platform("ascend-910b3")
    with pytest.raises(pc.PlatformError, match="hardware model explicitly"):
        pc.resolve_platform(toolchain="ascend")
    with pytest.raises(pc.PlatformError, match="no active task platform"):
        pc.get_platform()
    historical = pc.resolve_platform(state={})
    assert historical.compatibility is not None
    with pc.using_platform(historical):
        assert pc.get_adapter() is historical.adapter


def test_knowledge_scopes_keep_legacy_facts_out_of_new_toolchains():
    old = pc.resolve_platform("ascend-910b2c", "npuir")
    new = pc.resolve_platform("ascend-950dt-9582", "ascend")
    historical_fact = {"source": "queue", "type": "D"}
    process_proposal = {"source": "queue", "type": "R", "toolchain_id": "*"}
    with pc.using_platform(old):
        assert pc.knowledge_entry_applies(historical_fact)
        assert pc.knowledge_entry_applies({"device": "910B2C"})
        assert not pc.knowledge_entry_applies({"device": "Ascend910B3"})
    with pc.using_platform(new):
        assert not pc.knowledge_entry_applies(historical_fact)
        assert not pc.knowledge_entry_applies({"source": "queue", "type": "R"})
        assert pc.knowledge_entry_applies(process_proposal)
        assert pc.knowledge_entry_applies(
            {"source": "queue", "type": "D", "toolchain_id": "ascend"}
        )
        assert not pc.knowledge_entry_applies({"toolchain_id": "npuir"})


def test_native_fingerprint_change_is_stale_and_missing_stamp_is_unknown():
    adapter = pc.resolve_platform("ascend-950dt-9582", "ascend").adapter
    current = {"raw": "a" * 64, "tilelang_commit": None}
    assert adapter.classify_stamp("a" * 64, current) == "fresh"
    assert adapter.classify_stamp("b" * 64, current) == "stale"
    assert adapter.classify_stamp("版本戳缺失", current) == "unknown"


@pytest.mark.parametrize(
    ("hardware", "toolchain", "device_name"),
    [
        ("ascend-910b3", "npuir", "Ascend910B3"),
        ("ascend-950dt-9582", "ascend", "Ascend950DT_9582"),
    ],
)
def test_platform_cli_explicit_detected_and_saved_hardware(
    tmp_path, monkeypatch, capsys, hardware, toolchain, device_name
):
    import statectl

    device = {"name": device_name, "cores": {"cube": 32, "vector": 64}}
    monkeypatch.setattr(pc.TOOLCHAINS[toolchain], "probe_device", lambda self: device)
    monkeypatch.setattr(
        pc.PlatformContext, "check_environment", lambda self: {"raw": "fixture-build"}
    )

    def invoke(*args):
        code = statectl.main(list(args))
        return code, json.loads(capsys.readouterr().out)

    for args in [
        ("--hardware", hardware, "--toolchain", toolchain),
        ("--hardware", hardware, "--toolchain", toolchain, "--probe"),
        ("--toolchain", toolchain, "--probe"),
        ("--toolchain", toolchain, "--probe", "--check-environment"),
    ]:
        code, result = invoke("platform", *args)
        assert code == 0, result
        assert result["platform"]["hardware_id"] == hardware
        assert result["platform"]["toolchain_id"] == toolchain

    for args in [("--probe",), ("--hardware", hardware, "--probe")]:
        code, result = invoke("platform", *args)
        assert (
            code == 1
            and "select a toolchain explicitly" in result["errors"][0]["message"]
        )

    (tmp_path / ".git").mkdir()
    task = tmp_path / "examples" / "task"
    code, result = invoke(
        "init",
        "--dir",
        str(task),
        "--project",
        "fixture",
        "--op",
        "op",
        "--scenario",
        "new_op",
        "--toolchain",
        toolchain,
        "--probe",
    )
    assert code == 0, result
    state_path = task / ".stage_state.json"
    saved_bytes = state_path.read_bytes()
    saved = json.loads(saved_bytes)
    device = {"name": "DifferentDevice"}
    code, result = invoke("platform", "--task-dir", str(task))
    assert code == 0, result
    for key in ("hardware_id", "toolchain_id", "platform_config"):
        assert result["platform"][key] == saved[key]

    other_toolchain = "ascend" if toolchain == "npuir" else "npuir"
    code, result = invoke(
        "platform", "--task-dir", str(task), "--toolchain", other_toolchain, "--probe"
    )
    assert (
        code == 1
        and "differs from the task selection" in result["errors"][0]["message"]
    )
    code, result = invoke("platform", "--task-dir", str(task), "--probe")
    assert code == 1 and "active device reports" in result["errors"][0]["message"]
    assert state_path.read_bytes() == saved_bytes


def test_registered_new_model_needs_no_workflow_branch(monkeypatch):
    import copy

    profiles = pc.hardware_profiles()
    model = copy.deepcopy(profiles["ascend-950dt-9582"])
    model["hardware_id"] = "future-model"
    model["device_names"] = ["FutureDevice"]
    model["capacity_bytes"]["UB"] = 327680
    monkeypatch.setattr(
        pc, "hardware_profiles", lambda: {**profiles, "future-model": model}
    )
    adapter = pc.TOOLCHAINS["ascend"]
    original = adapter.workflow_context
    monkeypatch.setattr(
        adapter,
        "workflow_context",
        lambda self: {
            **original(self),
            "supported_hardware": [
                *original(self)["supported_hardware"],
                "future-model",
            ],
        },
    )
    selected = pc.resolve_platform("future-model", "ascend")
    assert selected.design_budget()["capacity_bytes"]["UB"] == 327680
    assert pc.resolve_platform(state=selected.selection).selection == selected.selection


@pytest.mark.parametrize("query_fails", [False, True])
def test_native_device_query_keeps_cli_stdout_clean(monkeypatch, capfd, query_fails):
    import ctypes
    from types import SimpleNamespace

    def properties(_):
        print("vendor Python log")
        ctypes.CDLL(None).printf(b"vendor C log\n")
        if query_fails:
            raise RuntimeError("query failed")
        return SimpleNamespace(
            name="FixtureDevice", cube_core_num=32, vector_core_num=64
        )

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            npu=SimpleNamespace(
                current_device=lambda: 0, get_device_properties=properties
            )
        ),
    )
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    adapter = pc.TOOLCHAINS["ascend"]()
    if query_fails:
        with pytest.raises(RuntimeError, match="query failed"):
            adapter.probe_device()
    else:
        assert adapter.probe_device()["cores"] == {"cube": 32, "vector": 64}
    print("protocol stdout restored")
    assert capfd.readouterr().out == "protocol stdout restored\n"


def test_catalogue_extension_does_not_invalidate_saved_task(monkeypatch):
    context = pc.resolve_platform("ascend-910b3", "npuir")
    saved = context.selection
    original = type(context.adapter).workflow_context
    monkeypatch.setattr(
        type(context.adapter),
        "workflow_context",
        lambda self: {
            **original(self),
            "supported_hardware": original(self)["supported_hardware"]
            + ["future-model"],
        },
    )
    assert pc.resolve_platform(state=saved).selection == saved
    monkeypatch.setattr(
        type(context.adapter),
        "workflow_context",
        lambda self: {**original(self), "ruleset": "changed-rules.md"},
    )
    with pytest.raises(pc.PlatformError, match="configuration changed"):
        pc.resolve_platform(state=saved)


def test_pre_fix_tasks_keep_recorded_evidence(monkeypatch):
    context = pc.resolve_platform("ascend-950dt-9582", "ascend")
    saved = {
        "hardware_id": "ascend-950dt-9582",
        "toolchain_id": "ascend",
        "platform_config": {
            "hardware": next(iter(context.hardware["compatible_revisions"])),
            "toolchain": next(iter(context.toolchain["compatible_revisions"])),
        },
    }
    assert pc.resolve_platform(state=saved).selection == saved
    changed = dict(
        context.hardware, capacity_bytes={**context.hardware["capacity_bytes"], "UB": 1}
    )
    monkeypatch.setattr(
        pc, "hardware_profiles", lambda: {changed["hardware_id"]: changed}
    )
    with pytest.raises(pc.PlatformError, match="configuration changed"):
        pc.resolve_platform(state=saved)


@pytest.mark.parametrize(
    "model,toolchain,profiler",
    [("ascend-910b3", "npuir", "msprof"), ("ascend-950dt-9582", "ascend", "msopprof")],
)
@pytest.mark.parametrize("entry", ["saved_task", "explicit", "explicit_with_old_task"])
def test_task_context_reaches_child_and_restores_environment(
    tmp_path, monkeypatch, model, toolchain, profiler, entry
):
    import argparse
    import os
    import subprocess

    tileops = Path(__file__).resolve().parents[3] / "examples/TileOPs"
    monkeypatch.syspath_prepend(str(tileops))
    monkeypatch.setenv("PYTHONPATH", str(tileops))
    monkeypatch.delenv("TILEOPS_TASK_DIR", raising=False)
    monkeypatch.delenv(pc.PLATFORM_ENV, raising=False)
    context = pc.resolve_platform(model, toolchain)
    if entry == "saved_task":
        (tmp_path / ".stage_state.json").write_text(json.dumps(context.selection))
        context = pc.command_platform(
            argparse.Namespace(task_dir=str(tmp_path), hardware=None, toolchain=None)
        )
    elif entry == "explicit_with_old_task":
        old = tmp_path / "old_task"
        old.mkdir()
        old_context = pc.resolve_platform(
            "ascend-910b3" if toolchain == "ascend" else "ascend-950dt-9582",
            "npuir" if toolchain == "ascend" else "ascend",
        )
        (old / ".stage_state.json").write_text(json.dumps(old_context.selection))
        monkeypatch.setenv("TILEOPS_TASK_DIR", str(old))
        monkeypatch.setenv(pc.PLATFORM_ENV, json.dumps(old_context.evidence()))
    previous = {
        key: os.environ.get(key) for key in ("TILEOPS_TASK_DIR", pc.PLATFORM_ENV)
    }
    probe = """import json
from tileops.utils.task_platform import get_task_platform, shared_memory_budget
from tileops.benchmark.msprof import _task_adapter
context = get_task_platform()
print(json.dumps({
    "hardware": context.hardware["hardware_id"],
    "toolchain": context.toolchain["toolchain_id"],
    "budget": shared_memory_budget(),
    "profiler": _task_adapter().profiler,
}))
"""
    with pc.using_platform(context):
        from tileops.utils.task_platform import get_task_platform

        assert get_task_platform().selection == context.selection
        result = subprocess.check_output(
            [sys.executable, "-c", probe], cwd=tmp_path, text=True
        )
        cli = json.loads(
            subprocess.check_output(
                [
                    sys.executable,
                    str(pc.ROOT / ".agents/tools/statectl.py"),
                    "platform",
                ],
                cwd=tmp_path,
                text=True,
            )
        )
        assert cli["platform"]["hardware_id"] == model
        assert cli["platform"]["toolchain_id"] == toolchain
        assert json.loads(result) == {
            "hardware": model,
            "toolchain": toolchain,
            "budget": context.hardware["capacity_bytes"]["UB"],
            "profiler": profiler,
        }
    assert {key: os.environ.get(key) for key in previous} == previous


@pytest.mark.parametrize(
    "model", ["ascend-910b2c", "ascend-910b3", "ascend-950dt-9582"]
)
@pytest.mark.parametrize("level", ["L0A", "L0B"])
def test_original_operand_memory_budget_is_enforced(model, level):
    context = pc.resolve_platform(
        model, "ascend" if model == "ascend-950dt-9582" else "npuir"
    )
    for size, status in ((64, "pass"), (128, "fail")):
        checks = check_ub_budget(
            f"## 4.5 内存\n**{level}**\n| buffer | {size}KB |\n",
            context.design_budget(),
        )
        assert checks[0]["status"] == status, checks


def test_native_blockscaled_gemm_has_accumulator_budget():
    from design_calc_check import check_l0c_budget

    context = pc.resolve_platform("ascend-950dt-9582", "ascend")
    design = "block_M = 512\nblock_N = 512\nT.gemm_blockscaled(A, B, C)"
    with pc.using_platform(context):
        checks = check_l0c_budget(design, context.design_budget())
    assert checks["status"] == "fail", checks


def test_first_child_pins_migration_runtime(tmp_path, monkeypatch):
    import argparse
    import statectl

    parent = tmp_path / "op"
    selection = pc.resolve_platform("ascend-910b3", "npuir").selection
    parent.mkdir()
    aggregate_path = parent / ".migration_state.json"
    aggregate_path.write_text(json.dumps({**selection, "phase": "DEV_LOOP"}))
    children = [parent / "f1", parent / "f2"]
    for child in children:
        assert (
            sc(
                "init",
                "--dir",
                str(child),
                "--project",
                "op",
                "--op",
                child.name,
                "--scenario",
                "migration",
                "--migration-mode",
                "harness",
                "--stage-plan",
                "3",
                "--task-dir",
                str(parent),
            )[0]
            == 0
        )
    runtime = {"device": {"name": "Ascend910B3"}, "toolchain_stamp": {"raw": "build-A"}}
    monkeypatch.setattr(pc.PlatformContext, "probe", lambda self: dict(runtime))
    assert statectl.cmd_start(argparse.Namespace(dir=str(children[0]), stage=3)) == 0
    assert pc.read_json(aggregate_path)["platform_runtime"] == runtime
    runtime["toolchain_stamp"] = {"raw": "build-B"}
    with pytest.raises(pc.PlatformError, match="migration aggregate"):
        statectl.cmd_start(argparse.Namespace(dir=str(children[1]), stage=3))
    assert pc.read_json(children[1] / ".stage_state.json")["stage_status"] == {}


def test_knowledge_search_uses_the_common_model_scope(monkeypatch, capsys):
    import argparse
    import kb_search

    context = pc.resolve_platform("ascend-950dt-9582", "ascend")
    entry = {
        "source": "pattern-library",
        "id": "fixture",
        "path": "fixture.md",
        "hardware_id": "ascend-950dt-9582",
        "device": "ascend-950dt-9582",
        "toolchain_id": "ascend",
        "facets": {},
    }
    monkeypatch.setattr(kb_search, "build_index", lambda: [entry])
    with pc.using_platform(context):
        assert pc.knowledge_entry_applies(entry)
        assert kb_search.run(argparse.Namespace(index=True, json=True)) == 0
    assert "fixture" in capsys.readouterr().out


def test_malformed_config_fingerprint_is_rejected():
    context = pc.resolve_platform("ascend-910b3", "npuir")
    saved = {
        **context.selection,
        "platform_config": {"hardware": [], "toolchain": "invalid"},
    }
    with pytest.raises(pc.PlatformError, match="configuration changed"):
        pc.resolve_platform(state=saved)


@pytest.mark.parametrize("level", ["L0A", "L0B"])
def test_native_operand_memory_heading_is_checked(level):
    context = pc.resolve_platform("ascend-950dt-9582", "ascend")
    checks = check_ub_budget(
        f"## 4.5 内存\n### {level}\n| buffer | 128KB |\n",
        context.design_budget(),
    )
    assert any(check["status"] == "fail" for check in checks), checks
