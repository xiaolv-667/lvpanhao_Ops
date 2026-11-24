"""TileLang ascend backend and the locked ARM64 compiler/runtime environment."""

import ast
import contextlib
import csv
import ctypes
import hashlib
import io
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import tempfile
from importlib import metadata
from pathlib import Path

from .tileops_contract import integration_lint, meta_lint
from .adapter import ToolchainAdapter
from .cann_profile import benchmark_profile_result

ROOT = Path(__file__).resolve().parents[3]
PROFILE = ROOT / ".agents/platforms/toolchains/ascend/profile.json"


def file_digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class AscendToolchain(ToolchainAdapter):
    profiler = "msopprof"
    raw_profile_field = "msprof_raw_path"
    measurement_label = "msopprof 口径证据（Task Duration）"
    duration_label = "Task Duration"

    def workflow_context(self):
        profile = json.loads(PROFILE.read_text(encoding="utf-8"))
        profile["environment_lock_revision"] = file_digest(
            ROOT / profile["environment_lock"]
        )
        return profile

    def shared_memory_budget(self, device=None):
        raise RuntimeError(
            "this toolchain requires a task with an explicit hardware model"
        )

    def has_matrix_path(self, design):
        return bool(re.search(r"^[ \t]*T\.gemm(?:_blockscaled)?\s*\(", design, re.M))

    def matrix_path_required_message(self):
        return "no Ascend Cube GEMM call found in the design pseudocode"

    def design_modes(self):
        return r"SIMD|SIMT|Cube|混合", "SIMD/SIMT/Cube/混合"

    def core_evidence(self):
        return "get_device_properties", "② torch.npu.get_device_properties 实查记录"

    def probe_device(self):
        # The simulator driver also writes through C stdio; keep CLI JSON clean.
        sys.stdout.flush()
        saved_stdout = os.dup(1)
        with tempfile.TemporaryFile() as capture:
            try:
                os.dup2(capture.fileno(), 1)
                with contextlib.redirect_stdout(io.StringIO()):
                    import torch
                    import torch_npu  # noqa: F401 -- registers torch.npu

                    properties = torch.npu.get_device_properties(
                        torch.npu.current_device()
                    )
            finally:
                ctypes.CDLL(None).fflush(None)
                os.dup2(saved_stdout, 1)
                os.close(saved_stdout)
        return {
            "name": properties.name,
            "cores": {
                "cube": properties.cube_core_num,
                "vector": properties.vector_core_num,
            },
        }

    def device_cores(self):
        return self.probe_device()["cores"]

    def kernel_checks(self, tree, path, prefix):
        nodes = list(ast.walk(tree))
        ascend_namespace = any(
            (
                isinstance(node, ast.Import)
                and any(n.name == "tilelang.ascend.language" for n in node.names)
            )
            or (
                isinstance(node, ast.ImportFrom)
                and node.module == "tilelang.ascend"
                and any(n.name == "language" for n in node.names)
            )
            for node in nodes
        )
        entries = [
            node
            for node in nodes
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "tilelang"
            and node.func.attr in {"compile", "jit"}
        ]
        failures = []
        if not ascend_namespace:
            failures.append(
                {
                    "rule_id": f"{prefix}-ASCEND-LANGUAGE",
                    "file": path,
                    "message": "使用 tilelang.ascend.language 声明 Ascend kernel",
                }
            )
        if not entries or any(
            not any(
                kw.arg == "target"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value == "ascend"
                for kw in entry.keywords
            )
            for entry in entries
        ):
            failures.append(
                {
                    "rule_id": f"{prefix}-JIT",
                    "file": path,
                    "message": 'tilelang.compile 或 tilelang.jit 必须使用 target="ascend"',
                }
            )
        return failures, []

    def kernel_entry_checks(self, tree, path, prefix):
        nodes = list(ast.walk(tree))
        failures = []
        if not any(
            isinstance(node, ast.Constant) and node.value == "--level" for node in nodes
        ):
            failures.append(
                {
                    "rule_id": f"{prefix}-MAIN-LEVEL",
                    "file": path,
                    "message": "main 入口缺少 --level 参数",
                }
            )
        if not any(
            isinstance(node, ast.Attribute) and node.attr == "prim_func"
            for node in nodes
        ):
            failures.append(
                {
                    "rule_id": f"{prefix}-PRIM-FUNC",
                    "file": path,
                    "message": "缺少 Ascend T.prim_func",
                }
            )
        return failures, []

    def precision_command(self, kernel, level):
        return [sys.executable, kernel, "--level", level]

    def execution_command(self, command, *, output_dir=None):
        if os.environ.get("IS_SIMULATOR_ENV") != "true":
            return super().execution_command(command, output_dir=output_dir)
        return [
            "cannsim",
            "record",
            "-s",
            os.environ["soc_version"],  # noqa: SIM112 -- vendor variable
            "-n",
            "0",
            "-o",
            output_dir or tempfile.mkdtemp(prefix="tilelang-simulator-"),
            shlex.join(command),
        ]

    def execution_context(self):
        context = super().execution_context()
        if os.environ.get("IS_SIMULATOR_ENV") != "true":
            return context
        return {
            **context,
            "mode": "simulator",
            "instructions": (
                "先切换到任务目录，脚本和 kernel 用绝对路径，output_dir 每次新建；命令数组最后一项作为单一参数整体引用。"
                "此启动方式适用于精度验证、设计期原型和任何创建 NPU 张量的探针。"
                "所有仿真命令顺序执行，不并发启动模型进程，避免 CPU 和内存争用。"
                "cannsim 是命令行入口，不能用 python -m cannsim。"
                "仅导入 API/检查签名且不执行设备计算时，可直接在任务目录运行 Python。"
                "调用方按任务预算设置外层超时；仿真数据不得当作真机性能。"
            ),
        }

    def profiler_args(self, kernel, output, launch_count, warm_up):
        command = [
            self.profiler,
            f"--output={output}",
            f"--launch-count={launch_count}",
            f"--warm-up={warm_up}",
            "--aic-metrics=BasicInfo,PipeUtilization,ArithmeticUtilization,Memory,"
            "MemoryUB,MemoryL0,L2Cache,ResourceConflictRatio",
        ]
        if kernel:
            command.append(f"--kernel-name={kernel}")
        return command

    def benchmark_profiler_args(self, kernel, output, launch_count, warm_up):
        command = self.profiler_args(kernel, output, launch_count, warm_up)
        command = [arg for arg in command if not arg.startswith("--aic-metrics=")]
        if kernel is None:
            command = [arg for arg in command if not arg.startswith("--kernel-name=")]
        return [*command, "--aic-metrics=Roofline"]

    def benchmark_profile_result(self, output):
        return benchmark_profile_result(output)

    def profile_samples(self, output):
        samples = []
        for capture in sorted(Path(output).glob("OPPROF_*")):
            for path in sorted(capture.rglob("OpBasicInfo*.csv")):
                with path.open() as handle:
                    for row in csv.DictReader(handle):
                        if row.get("Task Duration(us)"):
                            samples.append(float(row["Task Duration(us)"]))
        return samples

    def integration_mode(self, kernel_path):
        path = Path(kernel_path)
        failures, _ = self.kernel_checks(ast.parse(path.read_text()), str(path), "S5")
        if failures:
            raise SystemExit(f"[error] {failures}")
        return None

    def toolchain_stamp(self, repo_root, *, active=False):
        stamp, _ = self._environment(repo_root)
        return stamp

    def check_environment(self, repo_root, *, hardware=None):
        stamp, errors = self._environment(repo_root, hardware=hardware)
        if errors:
            raise RuntimeError("Ascend toolchain mismatch: " + " | ".join(errors))
        return stamp

    def _environment(self, repo_root, *, hardware=None):
        profile = self.workflow_context()
        lock_path = Path(repo_root) / profile["environment_lock"]
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        errors = []
        if (
            hardware is not None
            and os.environ.get("ASCEND_NPU_ARCH") != hardware["architecture"]
        ):
            errors.append(
                f"ASCEND_NPU_ARCH: expected {hardware['architecture']}, got {os.environ.get('ASCEND_NPU_ARCH')}"
            )
        cxx = os.environ.get("CXX", "g++")
        try:
            compiler = subprocess.check_output(
                [cxx, "-dumpfullversion"], text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            compiler = None
        for name, actual, expected in (
            ("machine", platform.machine(), lock["machine"]),
            ("python", platform.python_version(), lock["python"]),
            ("CXX", compiler, lock["cxx_version"]),
        ):
            if actual != expected:
                errors.append(f"{name}: expected {expected}, found {actual}")

        packages = {}
        for name, expected in lock["packages"].items():
            try:
                actual = metadata.version(name)
            except metadata.PackageNotFoundError:
                actual = None
            packages[name] = actual
            if actual != expected:
                errors.append(f"{name}: expected {expected}, found {actual}")

        files = {}
        for item in lock["files"]:
            name, package = item["name"], item.get("package")
            try:
                if package:
                    path = Path(
                        metadata.distribution(package).locate_file(item["path"])
                    )
                else:
                    env_key = item.get("root_env", "ASCEND_HOME_PATH")
                    root = os.environ.get(env_key)
                    if not root:
                        errors.append(f"{name}: {env_key} is unset")
                        continue
                    path = Path(root) / item["path"]
                files[name] = file_digest(path)
                if files[name] != item["sha256"]:
                    errors.append(f"{name}: fingerprint differs from {lock['profile']}")
            except (OSError, metadata.PackageNotFoundError) as exc:
                errors.append(f"{name}: {exc}")
        evidence = {
            "lock": file_digest(lock_path),
            "packages": packages,
            "files": files,
            "compiler": compiler,
        }
        return {
            "raw": hashlib.sha256(
                json.dumps(evidence, sort_keys=True).encode()
            ).hexdigest(),
            "tilelang": packages["tilelang"],
            "tilelang_commit": None,
            "cann": lock["compiler_sdk"] if not errors else None,
            "runtime_opp": lock["runtime_opp"] if not errors else None,
            "compiler": compiler,
        }, errors

    def scaffold_checks(self, meta_path, repo_root):
        return meta_lint(meta_path, repo_root)

    def integration_checks(self, migration_state, repo_root):
        return integration_lint(
            migration_state, repo_root, integration_mode=self.integration_mode
        )
