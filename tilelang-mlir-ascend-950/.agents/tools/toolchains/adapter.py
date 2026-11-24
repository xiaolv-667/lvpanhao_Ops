"""Toolchain operations used by the agent workflow."""

from __future__ import annotations

from abc import ABC, abstractmethod
import sys
import re


class ToolchainAdapter(ABC):
    @property
    @abstractmethod
    def profiler(self) -> str:
        """Return the profiler executable used by the platform."""

    @property
    @abstractmethod
    def raw_profile_field(self) -> str:
        """Return the existing record field for native profiler output."""

    @property
    @abstractmethod
    def measurement_label(self) -> str:
        """Return the platform's human-readable measurement evidence label."""

    @property
    @abstractmethod
    def duration_label(self) -> str:
        """Return the latency label used in the existing report contract."""

    @abstractmethod
    def workflow_context(self) -> dict:
        """Return the platform configuration for agent prompts."""

    @abstractmethod
    def has_matrix_path(self, design: str) -> bool:
        """Recognize a design that uses matrix accumulation."""

    @abstractmethod
    def matrix_path_required_message(self) -> str:
        """Explain why the matrix budget check was skipped."""

    @abstractmethod
    def design_modes(self) -> tuple[str, str]:
        """Return the design mode matcher and its diagnostic label."""

    @abstractmethod
    def core_evidence(self) -> tuple[str, str]:
        """Return the required device query marker and its diagnostic label."""

    @abstractmethod
    def probe_device(self) -> dict:
        """Read the active device name and runtime-reported core counts."""

    @abstractmethod
    def device_cores(self) -> dict:
        """Query core counts using the legacy compatibility convention."""

    @abstractmethod
    def kernel_checks(self, tree, path: str, prefix: str):
        """Return platform-specific kernel failures and warnings."""

    @abstractmethod
    def kernel_entry_checks(self, tree, path: str, prefix: str):
        """Check the platform kernel entrypoint and primitive syntax."""

    @abstractmethod
    def precision_command(self, kernel: str, level: str) -> list[str]:
        """Build the command for the existing precision runner."""

    @abstractmethod
    def profiler_args(self, kernel, output, launch_count, warm_up) -> list[str]:
        """Build the platform profiler command prefix."""

    @abstractmethod
    def profile_samples(self, output) -> list[float]:
        """Read samples from the profiler's native output."""

    @abstractmethod
    def benchmark_profiler_args(
        self, kernel, output, launch_count, warm_up
    ) -> list[str]:
        """Build the framework latency-collection command."""

    @abstractmethod
    def benchmark_profile_result(self, output) -> tuple[list[float], str, list[str]]:
        """Read latency samples in microseconds and captured operator identities."""

    def classify_stamp(self, entry_stamp: str, current: dict) -> str:
        """Compare an explicitly recorded fingerprint; legacy formats are backend-owned."""
        current_stamp = current.get("raw", "")
        if not (
            re.fullmatch(r"[0-9a-f]{64}", entry_stamp or "")
            and re.fullmatch(r"[0-9a-f]{64}", current_stamp)
        ):
            return "unknown"
        return "fresh" if entry_stamp == current_stamp else "stale"

    @abstractmethod
    def integration_mode(self, kernel_path) -> str | None:
        """Validate an integrated kernel and return its wrapper mode."""

    @abstractmethod
    def shared_memory_budget(self, device=None) -> int:
        """Query the framework memory budget for historical tasks without a model."""

    def execution_command(
        self, command: list[str], *, output_dir: str | None = None
    ) -> list[str]:
        """Build a runnable command for the selected execution environment."""
        return list(command)

    def execution_context(self) -> dict:
        """Return the platform launch instructions passed to every stage."""
        return {
            "mode": "device",
            "working_directory": "<task_dir>",
            "precision_command": self.execution_command(
                self.precision_command("<kernel>", "<level>"), output_dir="<output_dir>"
            ),
            "prototype_command": self.execution_command(
                [sys.executable, "<script>"], output_dir="<output_dir>"
            ),
        }

    def check_environment(self, repo_root: str, *, hardware=None) -> dict:
        """Return the stamp after checking the active toolchain."""
        stamp = self.toolchain_stamp(repo_root, active=True)
        if not all(stamp.get(key) for key in ("tilelang", "cann", "compiler")):
            backend = self.workflow_context()["backend"].upper()
            raise RuntimeError(
                f"active {backend} toolchain version could not be verified"
            )
        return stamp

    @abstractmethod
    def toolchain_stamp(self, repo_root: str, *, active: bool = False) -> dict:
        """Read the platform's current toolchain stamp."""

    def migration_mode(self, source_repo: str, operator: str) -> str:
        """Classify the source project using its existing integration contract."""
        from .tileops_contract import migration_mode

        return migration_mode(source_repo, operator)

    @abstractmethod
    def scaffold_checks(self, meta_path: str, repo_root: str):
        """Check the existing migration scaffold contract."""

    @abstractmethod
    def integration_checks(self, migration_state: dict, repo_root: str):
        """Check the existing integration contract."""
