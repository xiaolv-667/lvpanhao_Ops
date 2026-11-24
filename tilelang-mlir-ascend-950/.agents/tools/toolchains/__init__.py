"""Toolchain implementations used by the agent workflow."""

import json
from pathlib import Path

from .ascend import AscendToolchain
from .npuir import NpuirToolchain

TOOLCHAINS = {"npuir": NpuirToolchain, "ascend": AscendToolchain}


def compatibility_profile():
    """The registered platform for tasks created before platform selection."""
    path = (
        Path(__file__).resolve().parents[2]
        / "platforms/compatibility/legacy-npuir.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))
