"""Experiment p1_mask_mul: epilogue = EPILOGUE_MASK_MUL."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _epi  # noqa: E402
import _harness  # noqa: E402

base = _harness.load_base()
_ssd_chunk_coupling_kernel = _epi.build(base, _epi.EPILOGUE_MASK_MUL)
_launch = _ssd_chunk_coupling_kernel
golden_ssd_chunk_coupling = base.golden_ssd_chunk_coupling

if __name__ == "__main__":
    _harness.main(base)
