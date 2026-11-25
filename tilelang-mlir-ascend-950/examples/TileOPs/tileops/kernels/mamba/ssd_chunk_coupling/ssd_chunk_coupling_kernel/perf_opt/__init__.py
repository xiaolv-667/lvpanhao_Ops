"""Stage 4 perf_opt drop-in for the mamba-2 chunk-coupling kernel.

Exposes the same factory symbol the op wrapper's baseline/perf_opt switch
block imports, so the two import lines stay interchangeable:

    # --- baseline (Stage 3) ----------------------------------------------------
    from .ssd_chunk_coupling_kernel import _ssd_chunk_coupling_kernel
    # --- perf_opt (Stage 4 tuned) ----------------------------------------------
    # from .ssd_chunk_coupling_kernel.perf_opt._ssd_chunk_coupling_kernel import _ssd_chunk_coupling_kernel

Tuned structure: one block per ``(batch, chunk, group)`` row panel, C/B chunk
panel resident in L1 (1.0x read amplification), two alternating L0C
accumulators so the next tile's GEMM overlaps the previous tile's AIV epilogue,
and a tile-class-specialised epilogue (Cube fixpipe for strictly-lower tiles,
exact zero store for strictly-upper tiles, exact ``select`` mask for the
diagonal tiles only).  ``Q // BL != 2`` geometries fall back to the frozen
Stage-3 kernel, so their code and numerics are unchanged.
"""

from ._ssd_chunk_coupling_kernel import (  # noqa: F401
    TUNED_DEFAULT_CONFIG,
    TUNED_ROW_PANEL,
    TUNED_ZERO_SLICE,
    _ssd_chunk_coupling_kernel,
    golden_ssd_chunk_coupling,
)

__all__ = [
    "_ssd_chunk_coupling_kernel",
    "golden_ssd_chunk_coupling",
    "TUNED_DEFAULT_CONFIG",
    "TUNED_ROW_PANEL",
    "TUNED_ZERO_SLICE",
]
