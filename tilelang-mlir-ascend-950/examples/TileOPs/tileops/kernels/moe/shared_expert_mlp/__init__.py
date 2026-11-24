"""Shared expert MLP kernel package (NPU).

Re-exports the kernel class so the family package can import it by package name.
"""

from tileops.kernels.moe.shared_expert_mlp.shared_expert_mlp import SharedExpertMLPKernel

__all__ = ["SharedExpertMLPKernel"]
