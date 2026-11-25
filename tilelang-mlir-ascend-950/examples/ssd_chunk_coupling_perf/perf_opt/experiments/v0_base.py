"""v0: the frozen Stage-3 baseline, re-exported unchanged (driver sanity check)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _harness  # noqa: E402

base = _harness.load_base()
_launch = base._launch
_ssd_chunk_coupling_kernel = base._ssd_chunk_coupling_kernel
golden_ssd_chunk_coupling = base.golden_ssd_chunk_coupling

if __name__ == "__main__":
    _harness.main(base)
