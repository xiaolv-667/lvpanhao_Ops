"""Toolchain rules and tools for the existing NPUIR workflow."""

from __future__ import annotations

import ast
import csv
import contextlib
import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from .adapter import ToolchainAdapter
from .cann_profile import benchmark_profile_result
from .tileops_contract import _fail, meta_lint, integration_lint

ROOT = Path(__file__).resolve().parents[3]
PROFILE = ROOT / ".agents/platforms/toolchains/npuir/profile.json"
PROFILER = "msprof"
DURATION_FIELD = "Task Duration(us)"
METRICS = (
    "BasicInfo,PipeUtilization,ArithmeticUtilization,Memory,MemoryUB,"
    "MemoryL0,L2Cache,ResourceConflictRatio"
)


class NpuirToolchain(ToolchainAdapter):
    profiler = PROFILER
    duration_label = "Task Duration"

    @property
    def raw_profile_field(self) -> str:
        return self.workflow_context()["measurement"]["raw_evidence_field"]

    measurement_label = "msprof 口径证据（Task Duration，唯一 kernel 时延口径）"

    def workflow_context(self) -> dict:
        return json.loads(PROFILE.read_text(encoding="utf-8"))

    def has_matrix_path(self, design: str) -> bool:
        return bool(
            re.search(
                r"^[ \t]*(?:(?:T\.gemm|(?:T\.)?(?:load_nd2nz|store_fixpipe))[ \t]*\("
                r"""|(?:with[ \t]+)?(?:T\.)?Scope\(\s*["']Cube["']\s*\))""",
                design,
                re.MULTILINE,
            )
        )

    def matrix_path_required_message(self) -> str:
        return (
            "no Cube-path call (T.gemm/load_nd2nz/store_fixpipe/"
            "Scope(Cube)) found in the design pseudocode"
        )

    def design_modes(self) -> tuple[str, str]:
        return r"[Dd]eveloper|[Ee]xpert|混合", "Developer/Expert/混合"

    def core_evidence(self) -> tuple[str, str]:
        return (
            "get_aicore_num",
            "② 物理核数依据（NPUUtils.get().get_aicore_num() 实查记录）",
        )

    def probe_device(self) -> dict:
        with contextlib.redirect_stdout(io.StringIO()):
            import torch
            import torch_npu  # noqa: F401 -- registers torch.npu

            properties = torch.npu.get_device_properties(torch.npu.current_device())
        return {
            "name": properties.name,
            "cores": {
                "cube": properties.cube_core_num,
                "vector": properties.vector_core_num,
            },
        }

    def device_cores(self) -> dict:
        from tilelang.utils import NPUUtils

        count = NPUUtils.get().get_aicore_num()
        return {"cube": count, "vector": count * 2, "mixed": count}

    def shared_memory_budget(self, device=None):
        from tileops.toolchains.npuir.device import shared_memory_budget

        return shared_memory_budget(device)

    def precision_command(self, kernel: str, level: str) -> list[str]:
        return [sys.executable, kernel, "--level", level]

    def profiler_args(self, kernel, output, launch_count, warm_up) -> list[str]:
        return [
            PROFILER,
            "op",
            f"--kernel-name={kernel}",
            f"--output={output}",
            f"--launch-count={launch_count}",
            f"--warm-up={warm_up}",
            "--dump=off",
            f"--aic-metrics={METRICS}",
        ]

    def benchmark_profiler_args(self, kernel, output, launch_count, warm_up):
        command = self.profiler_args(kernel, output, launch_count, warm_up)
        command = [arg for arg in command if not arg.startswith("--aic-metrics=")]
        if kernel is None:
            command = [arg for arg in command if not arg.startswith("--kernel-name=")]
        return [*command, "--aic-metrics=Roofline"]

    def benchmark_profile_result(self, output):
        return benchmark_profile_result(output)

    def profile_samples(self, output) -> list[float]:
        samples = []
        for capture in sorted(Path(output).glob("OPPROF_*")):
            paths = sorted(capture.glob("*/*/OpBasicInfo_*.csv"))
            # A single launch writes an unnumbered CSV at the capture root.
            if not paths:
                paths = list(capture.glob("OpBasicInfo.csv"))
            for path in paths:
                with path.open() as handle:
                    for row in csv.DictReader(handle):
                        if row.get(DURATION_FIELD):
                            samples.append(float(row[DURATION_FIELD]))
        return samples

    def scaffold_checks(self, meta_path: str, repo_root: str):
        return meta_lint(meta_path, repo_root)

    def integration_checks(self, migration_state: dict, repo_root: str):
        return integration_lint(
            migration_state, repo_root, integration_mode=self.integration_mode
        )

    def integration_mode(self, kernel_path):
        return read_integration_mode(Path(kernel_path))

    def classify_stamp(self, entry_stamp, current):
        verdict = super().classify_stamp(entry_stamp, current)
        if verdict != "unknown":
            return verdict
        commits = re.findall(r"\b([0-9a-f]{7,40})\b", entry_stamp or "")
        commit = (
            current["tilelang_commit"]
            if "tilelang_commit" in current
            else (
                re.findall(r"\b([0-9a-f]{7,40})\b", current.get("raw", "")) or [None]
            )[0]
        )
        if not commits or not commit:
            return "unknown"
        return (
            "fresh"
            if any(c.startswith(commit) or commit.startswith(c) for c in commits)
            else "stale"
        )

    def toolchain_stamp(self, repo_root: str, *, active: bool = False) -> dict:
        return detect_current_stamp(repo_root, active=active)

    def kernel_checks(self, tree, path: str, prefix: str):
        failures, warnings = [], []
        backend = self.workflow_context()["backend"]
        jit_found = False
        for _dec, has_target, target_val in _jit_decorators(tree):
            jit_found = True
            if has_target and target_val != backend:
                failures.append(
                    _fail(
                        f"{prefix}-JIT",
                        path,
                        f'@tilelang.jit target="{target_val}"，必须为 target="{backend}"',
                    )
                )
            elif not has_target:
                warnings.append(
                    _warn(
                        f"{prefix}-JIT",
                        path,
                        f'@tilelang.jit 未显式声明 target="{backend}"',
                    )
                )
        if not jit_found:
            failures.append(
                _fail(f"{prefix}-JIT", path, "未找到 @tilelang.jit 装饰的 kernel 函数")
            )

        return failures, warnings

    def kernel_entry_checks(self, tree, path: str, prefix: str):
        failures, warnings = [], []
        # --- main with --level ---
        has_level = any(
            isinstance(node, ast.Constant) and node.value == "--level"
            for node in ast.walk(tree)
        )
        if not has_level:
            failures.append(
                _fail(
                    f"{prefix}-MAIN-LEVEL",
                    path,
                    "main 入口缺少 --level 参数（L0/all 分层测试入口）",
                )
            )

        # --- T.prim_func usage ---
        has_prim = any(
            (isinstance(node, ast.Attribute) and node.attr == "prim_func")
            or (isinstance(node, ast.Name) and node.id == "prim_func")
            for node in ast.walk(tree)
        )
        if not has_prim:
            warnings.append(
                _warn(
                    f"{prefix}-PRIM-FUNC",
                    path,
                    "未检测到 T.prim_func（请确认 kernel 结构）",
                )
            )
        return failures, warnings


_warn = _fail


def _jit_decorators(tree: ast.AST):
    """Yield (call_node, has_target, target_value) for tilelang.jit decorators."""
    tilelang_aliases, jit_names = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                if item.name == "tilelang":
                    tilelang_aliases.add(item.asname or "tilelang")
        elif isinstance(node, ast.ImportFrom) and node.module == "tilelang":
            for item in node.names:
                if item.name == "jit":
                    jit_names.add(item.asname or "jit")

    def is_jit(func) -> ast.expr | None:
        for dec in func.decorator_list:
            target = None
            if isinstance(dec, ast.Call):
                target = dec.func
            elif isinstance(
                dec, (ast.Attribute, ast.Name)
            ):  # bare @tilelang.jit / @jit
                target = dec
            if target is None:
                continue
            if isinstance(target, ast.Attribute) and target.attr == "jit":
                if isinstance(target.value, ast.Name) and (
                    target.value.id in tilelang_aliases
                ):
                    return dec
            elif isinstance(target, ast.Name) and target.id in jit_names:
                return dec
        return None

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            dec = is_jit(node)
            if dec is not None:
                if isinstance(dec, ast.Call):
                    for kw in dec.keywords:
                        if kw.arg == "target":
                            val = getattr(kw.value, "value", None)
                            yield dec, True, val
                            break
                    else:
                        yield dec, False, None
                else:
                    yield dec, False, None


# ---------------------------------------------------------------------------
# Toolchain version evidence for historical knowledge entries


def detect_current_stamp(repo_root: str, *, active: bool = False) -> dict:
    stamp = {"tilelang": None, "tilelang_commit": None, "cann": None, "raw": ""}
    source_root = repo_root
    try:
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            import tilelang  # noqa: PLC0415
        stamp["tilelang"] = getattr(tilelang, "__version__", None)
        if active:
            package_root = Path(tilelang.__file__).resolve().parent.parent
            source_root = package_root if (package_root / ".git").exists() else None
    except Exception:  # noqa: BLE001 -- advisory tool
        if active:
            source_root = None
    if source_root is not None:
        try:
            head = subprocess.run(
                ["git", "rev-parse", "--short=10", "HEAD"],
                cwd=source_root,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if head.returncode == 0:
                stamp["tilelang_commit"] = head.stdout.strip()
        except Exception:  # noqa: BLE001
            pass
    ascend_home = os.environ.get("ASCEND_HOME_PATH", "")
    candidates = [
        Path(ascend_home) / ".." / "version.cfg"
        if ascend_home
        else Path("/../version.cfg"),
        Path("/usr/local/Ascend/ascend-toolkit/latest/version.cfg"),
    ]
    if active and ascend_home:
        home = Path(ascend_home)
        candidates = [
            home / "version.cfg",
            home / f"{os.uname().machine}-linux/ascend_toolkit_install.info",
            *candidates,
        ]
    for candidate in candidates:
        try:
            text = candidate.read_text()
        except OSError:
            continue
        match = re.search(r"Version\s*=\s*(\S+)", text, re.I if active else 0)
        if match:
            stamp["cann"] = match.group(1)
            break
    if active:
        try:
            compiler = subprocess.run(
                ["bishengir-compile", "--version"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            ).stdout
            stamp["compiler"] = next(
                (line.strip() for line in compiler.splitlines() if line.strip()), None
            )
        except (OSError, subprocess.SubprocessError):
            stamp["compiler"] = None
    stamp["raw"] = "tilelang {} (commit {}) + CANN {}".format(
        stamp["tilelang"] or "?", stamp["tilelang_commit"] or "?", stamp["cann"] or "?"
    )
    if active:
        stamp["raw"] += " + " + (stamp["compiler"] or "unknown bishengir-compile")
    return stamp


def read_integration_mode(kernel_src: Path) -> str:
    """Read the verified source mode, accepting legacy literal setdefault calls."""
    tree = ast.parse(kernel_src.read_text(encoding="utf-8"))
    declared: list[str] = []
    defaults: list[str] = []
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "ASCEND_MODE"
                for target in node.targets
            )
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            declared.append(node.value.value)
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if (
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "setdefault"
            and isinstance(call.func.value, ast.Attribute)
            and call.func.value.attr == "environ"
            and len(call.args) >= 2
            and isinstance(call.args[0], ast.Constant)
            and call.args[0].value == "TILELANG_ASCEND_MODE"
        ):
            value = call.args[1]
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                defaults.append(value.value)
            elif isinstance(value, ast.Name) and value.id == "ASCEND_MODE" and declared:
                defaults.append(declared[-1])
    modes = {mode.capitalize() for mode in declared + defaults}
    if not modes or modes - {"Developer", "Expert"} or len(modes) != 1:
        raise SystemExit(
            f"[error] {kernel_src}: missing or conflicting ASCEND_MODE / "
            "TILELANG_ASCEND_MODE declaration; expected one Developer or Expert mode"
        )
    return modes.pop()
