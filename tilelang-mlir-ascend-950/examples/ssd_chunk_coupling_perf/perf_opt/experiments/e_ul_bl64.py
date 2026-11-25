"""Experiment: class-U/L specialisation (as e_ul) with BL=64 output tiles.

``_tile_shape`` is overridden so that the block tile is 64x64 instead of
128x128: the diagonal-crossing element count is ``Q * BL`` per chunk, so halving
BL halves the per-element masked work, at the cost of more tile re-reads.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _epi  # noqa: E402
import _harness  # noqa: E402

BL = int(os.environ.get("SSD_CC_BL", "64"))

base = _harness.load_base()
_base_tile_shape = base._tile_shape


def _tile_shape(chunk_len, d_state):
    _, block_n = _base_tile_shape(chunk_len, d_state)
    return BL, block_n


base._tile_shape = _tile_shape
_ssd_chunk_coupling_kernel = _epi.build(base, _epi.EPILOGUE_CLASS_UL)
_launch = _ssd_chunk_coupling_kernel
golden_ssd_chunk_coupling = base.golden_ssd_chunk_coupling

if __name__ == "__main__":
    _harness.main(base)
