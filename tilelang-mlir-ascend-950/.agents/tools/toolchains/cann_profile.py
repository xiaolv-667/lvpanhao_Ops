"""Load the TileOPs-owned CANN decoder without importing device runtimes."""

import importlib.util
from pathlib import Path

_PATH = (
    Path(__file__).resolve().parents[3]
    / "examples/TileOPs/tileops/toolchains/cann_profile.py"
)
_SPEC = importlib.util.spec_from_file_location("_tileops_cann_profile", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
benchmark_profile_result = _MODULE.benchmark_profile_result
