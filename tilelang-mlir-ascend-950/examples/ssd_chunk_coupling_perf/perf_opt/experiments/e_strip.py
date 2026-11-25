"""Experiment STRIP: U clear + L fixpipe + D strip-decomposed epilogue."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _epi2  # noqa: E402
import _harness  # noqa: E402

STRIP = 16

base = _harness.load_base()
_ssd_chunk_coupling_kernel = _epi2.build(base, _epi2.MODE_STRIP, strip=STRIP)
_launch = _ssd_chunk_coupling_kernel
golden_ssd_chunk_coupling = base.golden_ssd_chunk_coupling

if __name__ == "__main__":
    _harness.main(base)
