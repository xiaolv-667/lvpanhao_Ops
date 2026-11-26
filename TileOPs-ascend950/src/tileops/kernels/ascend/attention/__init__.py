"""Ascend implementations of attention kernel interfaces."""

from tileops.kernels.ascend.attention.dsa_decode import AscendDSADecodeKernel
from tileops.kernels.ascend.attention.gqa_lse import AscendGQADenseWithLSEKernel
from tileops.kernels.ascend.attention.nsa_compressed_varlen import (
    AscendNSACompressedFwdVarlenKernel,
)
from tileops.kernels.ascend.attention.nsa_topk_varlen import AscendNSATopKVarlenKernel
from tileops.kernels.ascend.attention.nsa_varlen_unified import AscendNSAFwdVarlenKernel
from tileops.kernels.ascend.attention.paged_cache_write import AscendPagedKVCacheWriteKernel

__all__ = [
    "AscendDSADecodeKernel",
    "AscendGQADenseWithLSEKernel",
    "AscendNSACompressedFwdVarlenKernel",
    "AscendNSAFwdVarlenKernel",
    "AscendNSATopKVarlenKernel",
    "AscendPagedKVCacheWriteKernel",
]
