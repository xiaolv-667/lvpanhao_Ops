"""Shared loader: expose the frozen Stage-3 baseline module as ``base``.

Every experiment module in this directory loads the *unmodified* baseline
kernel module by path, then overrides only the kernel factory it wants to
compare.  The baseline file itself is never edited (Stage 4 constraint).
"""

import importlib.util
import os
import sys

BASE_PATH = os.environ.get(
    "SSD_CC_BASE",
    "/workspace/tilelang-mlir-ascend-950/examples/TileOPs/tileops/kernels/mamba/"
    "ssd_chunk_coupling/ssd_chunk_coupling_kernel/_ssd_chunk_coupling_kernel.py",
)


def load_base():
    spec = importlib.util.spec_from_file_location("ssd_cc_base", BASE_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ssd_cc_base"] = mod
    spec.loader.exec_module(mod)
    return mod


def install(base, factory):
    """Point the baseline harness (run_L0/L1/bench) at a new factory."""
    base._ssd_chunk_coupling_kernel = factory
    base._launch = factory


def main(base):
    base.main()
