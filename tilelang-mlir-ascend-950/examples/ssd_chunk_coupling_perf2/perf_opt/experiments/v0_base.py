"""Experiment v0_base."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _harness  # noqa: E402
import _rp2 as B  # noqa: E402

base = _harness.load_base()
_ssd_chunk_coupling_kernel = B.build(base, variant="base")
_launch = _ssd_chunk_coupling_kernel
golden_ssd_chunk_coupling = base.golden_ssd_chunk_coupling

if __name__ == "__main__":
    _harness.main(base)
