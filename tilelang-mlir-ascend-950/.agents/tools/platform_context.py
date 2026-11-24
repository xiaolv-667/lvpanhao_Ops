"""Resolve hardware and toolchain separately and pin their selection per task."""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

from toolchains import TOOLCHAINS, compatibility_profile
from toolchains.adapter import ToolchainAdapter

ROOT = Path(__file__).resolve().parents[2]
PLATFORMS = ROOT / ".agents/platforms"
STATE_FILES = (".migration_state.json", ".stage_state.json")
ACTIVE = ContextVar("task_platform", default=None)
PLATFORM_ENV = "TILELANG_TASK_PLATFORM"


class PlatformError(ValueError):
    """Invalid, incomplete or inconsistent platform selection."""


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlatformError(f"cannot read {path}: {exc}") from exc


def probe_device(adapter):
    try:
        return adapter.probe_device()
    except (ImportError, OSError, RuntimeError, AttributeError) as exc:
        raise PlatformError(f"device probe failed: {exc}") from exc


def config_revision(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def hardware_profiles():
    return {
        path.stem: read_json(path)
        for path in sorted((PLATFORMS / "hardware").glob("*.json"))
    }


def identify_device(device, profiles):
    for hardware_id, profile in profiles.items():
        if device["name"] in profile["device_names"]:
            return hardware_id
    raise PlatformError(f"unrecognized device model: {device['name']}")


@dataclass(frozen=True)
class PlatformContext:
    adapter: ToolchainAdapter
    toolchain: dict
    hardware: dict | None
    compatibility: dict | None = None
    runtime: dict | None = None
    task_dir: Path | None = None
    pinned_config: dict | None = None

    @property
    def selection(self):
        if self.hardware is None:
            return {}
        return {
            "hardware_id": self.hardware["hardware_id"],
            "toolchain_id": self.toolchain["toolchain_id"],
            "platform_config": self.pinned_config
            or {
                "hardware": effective_revision(self.hardware),
                "toolchain": effective_revision(self.toolchain),
            },
        }

    def measurements(self, stamp=None, workload=None):
        if self.hardware is None:
            return []
        stamp = stamp or (self.runtime or {}).get("toolchain_stamp", {}).get("raw")
        if not stamp:
            return []
        records = read_json(PLATFORMS / "measurements/index.json")["measurements"]
        return [
            record
            for record in records
            if record["hardware_id"] == self.hardware["hardware_id"]
            and record["toolchain_id"] == self.toolchain["toolchain_id"]
            and record["toolchain_stamp"] == stamp
            and (workload is None or record["workload"] == workload)
        ]

    def design_budget(self, workload=None):
        if self.compatibility is not None:
            budget = self.compatibility["design_budget"]
        else:
            multipliers = [
                item["value"]
                for item in self.measurements(workload=workload)
                if workload is not None
                and item["metric"] == "ub_inflation_factor"
                and item["unit"] == "ratio"
            ]
            budget = {
                **self.toolchain["design_rules"],
                "capacity_bytes": self.hardware["capacity_bytes"],
                "ub_inflation_factor": multipliers[0]
                if len(multipliers) == 1
                else None,
                "source": self.hardware["capacity_source"],
            }
        return {
            **budget,
            "capacity_kb": {
                name: value / 1024 for name, value in budget["capacity_bytes"].items()
            },
            "ub_inflation": budget["ub_inflation_factor"],
        }

    def prompt(self):
        if self.compatibility is not None:
            return {
                **self.compatibility,
                "compatibility_mode": True,
                "execution": self.adapter.execution_context(),
            }
        return {
            **self.toolchain,
            **self.selection,
            "hardware": self.hardware,
            "design_budget": self.design_budget(),
            "cost_measurements": self.measurements(),
            "runtime": self.runtime,
            "execution": self.adapter.execution_context(),
        }

    def verified(self):
        if self.hardware is None:
            return self
        return replace(self, runtime=self.probe())

    def evidence(self):
        if self.hardware is None:
            return {}
        return {**self.selection, "platform_runtime": self.runtime}

    def check_environment(self):
        try:
            return self.adapter.check_environment(str(ROOT), hardware=self.hardware)
        except (OSError, RuntimeError) as exc:
            raise PlatformError(str(exc)) from exc

    def probe(self):
        if self.hardware is None:
            return {"observed_cores": self.adapter.device_cores()}
        device = probe_device(self.adapter)
        if device["name"] not in self.hardware["device_names"]:
            raise PlatformError(
                f"expected {self.hardware['hardware_id']}, active device reports {device['name']}"
            )
        stamp = self.check_environment()
        result = {"device": device, "toolchain_stamp": stamp}
        if self.runtime is not None and self.runtime != result:
            raise PlatformError(
                "active device capabilities or toolchain version differ from the task record"
            )
        return result


def legacy_context():
    compatibility = compatibility_profile()
    adapter = TOOLCHAINS[compatibility["backend"]]()
    return PlatformContext(
        adapter,
        adapter.workflow_context(),
        None,
        compatibility,
    )


def resolve_platform(hardware=None, toolchain=None, *, detect=False, state=None):
    if state is not None:
        if not isinstance(state, dict):
            raise PlatformError("task state must be a JSON object")
        saved = (state.get("hardware_id"), state.get("toolchain_id"))
        runtime = state.get("platform_runtime")
        if runtime is not None and (
            not isinstance(runtime, dict)
            or not isinstance(runtime.get("device"), dict)
            or not isinstance(runtime.get("toolchain_stamp"), dict)
            or not isinstance(runtime["toolchain_stamp"].get("raw"), str)
        ):
            raise PlatformError("invalid platform_runtime in task state")
        if saved == (None, None):
            if any(
                key in state
                for key in (
                    "hardware_id",
                    "toolchain_id",
                    "platform_config",
                    "platform_runtime",
                )
            ):
                raise PlatformError(
                    "task state must contain both hardware_id and toolchain_id"
                )
            if hardware or toolchain:
                raise PlatformError(
                    "this task uses the legacy compatibility configuration"
                )
            return legacy_context()
        if (hardware and hardware != saved[0]) or (toolchain and toolchain != saved[1]):
            raise PlatformError(
                "requested hardware or toolchain differs from the task selection"
            )
        hardware, toolchain = saved
        if not hardware or not toolchain:
            raise PlatformError(
                "task state must contain both hardware_id and toolchain_id"
            )
    if not toolchain:
        raise PlatformError(
            "select a toolchain explicitly or supply an existing task directory"
        )
    if not hardware and not detect:
        raise PlatformError(
            "select a hardware model explicitly or supply an existing task directory"
        )
    if toolchain not in TOOLCHAINS:
        raise PlatformError(f"unknown toolchain: {toolchain}")
    adapter = TOOLCHAINS[toolchain]()
    toolchain_config = adapter.workflow_context()
    profiles = hardware_profiles()
    if hardware is None:
        hardware = identify_device(probe_device(adapter), profiles)
    if hardware not in profiles:
        raise PlatformError(f"unknown hardware model: {hardware}")
    if hardware not in toolchain_config["supported_hardware"]:
        raise PlatformError(f"{toolchain} is not configured for {hardware}")
    context = PlatformContext(
        adapter,
        toolchain_config,
        profiles[hardware],
        runtime=state.get("platform_runtime") if state is not None else None,
    )
    if state is not None:
        saved_config = state.get("platform_config")
        if not compatible_config(context, saved_config):
            raise PlatformError(
                "hardware or toolchain configuration changed since task initialization"
            )
        context = replace(context, pinned_config=saved_config)
    return context


def effective_revision(config):
    """Exclude catalogue and compatibility metadata from execution rules."""
    return config_revision(
        {
            key: value
            for key, value in config.items()
            if key not in {"supported_hardware", "compatible_revisions"}
        }
    )


def compatible_config(context, saved):
    """Accept identical rules or an explicitly equivalent configuration revision."""
    if (
        not isinstance(saved, dict)
        or set(saved) != {"hardware", "toolchain"}
        or any(not isinstance(value, str) for value in saved.values())
    ):
        return False
    return all(
        saved[key] == effective_revision(config)
        or config.get("compatible_revisions", {}).get(saved[key])
        == effective_revision(config)
        for key, config in (
            ("hardware", context.hardware),
            ("toolchain", context.toolchain),
        )
    )


def task_state_path(directory, *, exact=False, state_files=STATE_FILES):
    path = Path(directory).resolve()
    directories = (path,) if exact else (path, *path.parents)
    for parent in directories:
        for name in state_files:
            candidate = parent / name
            if candidate.is_file():
                return candidate
        if (parent / ".git").exists():
            break
    return None


def add_platform_args(parser, *, task_dir=True):
    parser.add_argument("--hardware", help="hardware model ID")
    parser.add_argument("--toolchain", help="toolchain ID")
    if task_dir:
        parser.add_argument("--task-dir", help="directory containing the task state")


def command_platform(args, directory=None):
    inferred = task_state_path(directory) if directory else None
    explicit = task_state_path(args.task_dir, exact=True) if args.task_dir else None
    if args.task_dir and explicit is None:
        raise PlatformError(f"no task state found at {args.task_dir}")
    path = explicit or inferred
    if not (path or args.hardware or args.toolchain):
        inherited = get_platform(required=False)
        if inherited is not None:
            return inherited
    platform = resolve_platform(
        args.hardware, args.toolchain, state=read_json(path) if path else None
    )
    if explicit and inferred and explicit != inferred:
        inferred_platform = resolve_platform(state=read_json(inferred))
        if inferred_platform.evidence() != platform.evidence():
            raise PlatformError(
                "selected task platform differs from the artifact's task"
            )
    return replace(platform, task_dir=path.parent if path else None)


def get_platform(*, required=True):
    context = ACTIVE.get()
    if context is None:
        task_dir = os.environ.get("TILEOPS_TASK_DIR")
        inherited = os.environ.get(PLATFORM_ENV)
        if task_dir:
            context = command_platform(
                SimpleNamespace(task_dir=task_dir, hardware=None, toolchain=None)
            )
        elif inherited is not None:
            try:
                state = json.loads(inherited)
            except json.JSONDecodeError as exc:
                raise PlatformError(f"invalid inherited platform: {exc}") from exc
            context = resolve_platform(state=state)
    if context is None and required:
        raise PlatformError(
            "no active task platform; select hardware and toolchain or supply a task directory"
        )
    return context


def run_with_platform(parser, args, run, directory=None, *, verify=False):
    try:
        platform = command_platform(args, directory)
        if verify:
            platform = platform.verified()
    except (PlatformError, ImportError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    with using_platform(platform):
        return run(args)


def get_adapter():
    return get_platform().adapter


@contextmanager
def using_platform(context):
    token = ACTIVE.set(context)
    environment = {
        "TILEOPS_TASK_DIR": str(context.task_dir)
        if context.task_dir is not None
        else None,
        PLATFORM_ENV: json.dumps(context.evidence()),
    }
    previous = {key: os.environ.get(key) for key in environment}
    try:
        for key, value in environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        ACTIVE.reset(token)


def knowledge_entry_applies(entry: dict) -> bool:
    """Apply explicit scope; unscoped historical entries retain their original domain."""
    platform = get_platform()
    if platform.hardware is None:
        return True
    hardware = entry.get("hardware_id") or entry.get("device")
    toolchain = entry.get("toolchain_id")
    if (
        hardware
        and hardware != platform.hardware["hardware_id"]
        and not any(
            name.lower().endswith(hardware.lower())
            for name in platform.hardware["device_names"]
        )
    ):
        return False
    if toolchain and toolchain not in {platform.toolchain["toolchain_id"], "*"}:
        return False
    if entry.get("source") != "queue" or toolchain:
        return True
    # Entries predating explicit scope belong to the original compatibility domain.
    if platform.toolchain["toolchain_id"] != legacy_context().toolchain["toolchain_id"]:
        return False
    stamp = entry.get("toolchain_stamp", "")
    for model in hardware_profiles().values():
        if any(name.lower() in stamp.lower() for name in model["device_names"]):
            return model["hardware_id"] == platform.hardware["hardware_id"]
    return True
