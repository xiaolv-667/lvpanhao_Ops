"""Stage 4 (perf_opt) tuned kernel for SharedExpertMLPFwdOp.

Single self-contained module: the op-level ``SharedExpertMLPKernel`` (same
constructor and call shape as the Stage 3 class) with the Stage 4 tile
(256x256x128, num_stages 3) and the gated activation fused into the gate/up
GEMM's Cube + Vector epilogue.  Adopted by ``shared_expert_mlp.py``; the Stage 3
modules stay in the parent package as the rollback source.
"""

from .SharedExpertMLPFwdOp import (
    TUNED_BLOCK_K,
    TUNED_BLOCK_M,
    TUNED_BLOCK_N,
    TUNED_NUM_STAGES,
    TUNED_SPLIT_K,
    SharedExpertMLPKernel,
)

__all__ = [
    "SharedExpertMLPKernel",
    "TUNED_BLOCK_M",
    "TUNED_BLOCK_N",
    "TUNED_BLOCK_K",
    "TUNED_NUM_STAGES",
    "TUNED_SPLIT_K",
]
