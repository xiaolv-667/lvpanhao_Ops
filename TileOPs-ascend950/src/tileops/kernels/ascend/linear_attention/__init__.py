"""Ascend linear-attention kernels."""

from tileops.kernels.ascend.linear_attention.gdn_fwd import AscendGDNFwdKernel
from tileops.kernels.ascend.linear_attention.gla_chunk_fwd import AscendGLAChunkFwdKernel

__all__ = [
    "AscendGLAChunkFwdKernel",
    "AscendGDNFwdKernel",
]
