"""MoE permute-align kernel package (NPU).

Re-exports the kernel class so the family package can import it by package name.
"""

from tileops.kernels.moe.moe_permute_align.moe_permute_align import MoEPermuteAlignKernel

__all__ = ["MoEPermuteAlignKernel"]
