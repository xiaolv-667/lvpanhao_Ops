"""Mamba-2 causal per-chunk ``C @ B^T`` coupling (CB) matrix operator.

Computes ``cb[b,c,g,l,s] = sum_n C[b,c*Q+l,g,n] * B[b,c*Q+s,g,n]`` with causal
masking (``cb[l,s] = 0`` when ``s > l``).

Adaptation from GPU (TileOPs) to NPU:

- ``forward``: ``C.contiguous()`` / ``B.contiguous()`` and the shape-derived
  kernel dispatch are preserved unchanged (O6); the GPU
  ``kernel_for(key, lambda: ...Call)`` call-spec machinery is replaced by the
  NPU ``_get_kernel`` cache over explicit dims (the NPU project has no
  ``_kernels/call_spec`` layer).  A device check on the inputs uses
  ``backend.is_device_tensor`` (O1); the GPU op had no device check of its own.
- ``tune`` parameter removed (O3); the kernel constructor takes
  ``(batch, num_chunks, n_groups, chunk_len, d_state, dtype, config=None)``.
- ``eval_roofline``: the GPU ``roof_key``/``tensor_core_roof`` split is a GPU
  perf-model detail; the manifest formula
  ``Bsz * NC * G * chunk_size * chunk_size * N`` is re-implemented inline as
  plain arithmetic (the NPU project has no tileops.perf module; roofline
  arithmetic is device-agnostic).
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from tileops.device import get_device_backend
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.mamba import SSDChunkCouplingFwdKernel
from tileops.ops.op_base import Op

__all__ = ["SSDChunkCouplingFwdOp"]


class SSDChunkCouplingFwdOp(Op):
    """Causal per-chunk ``C @ B^T`` coupling (CB) matrix producer operator.

    Computes ``cb[b,c,g,l,s] = sum_n C[b,c*Q+l,g,n] * B[b,c*Q+s,g,n]`` with
    causal masking (``cb[l,s] = 0`` if ``s > l``).

    Args:
        chunk_size: Chunk length (Q); ``S`` must be divisible by it.
        kernel_map: Optional override for kernel dispatch.
    """

    def __init__(
        self,
        chunk_size: int,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.chunk_size = int(chunk_size)
        self.batch = None
        self.num_chunks = None
        self.n_groups = None
        self.d_state = None
        self.dtype = None
        self.dispatch_kernel(kernel_map)
        self._kernel_cache: Dict[tuple, Kernel] = {}
        self.kernel = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"ssd_chunk_coupling_fwd": SSDChunkCouplingFwdKernel}

    def _get_kernel(
        self,
        batch: int,
        num_chunks: int,
        n_groups: int,
        d_state: int,
        dtype: torch.dtype,
        device_index: int | None,
    ) -> Kernel:
        key = (
            batch,
            num_chunks,
            self.chunk_size,
            n_groups,
            d_state,
            dtype,
            device_index,
        )
        if key not in self._kernel_cache:
            self._kernel_cache[key] = self.kernel_map["ssd_chunk_coupling_fwd"](
                batch,
                num_chunks,
                n_groups,
                self.chunk_size,
                d_state,
                dtype,
            )
        return self._kernel_cache[key]

    def forward(
        self,
        C: torch.Tensor,
        B: torch.Tensor,
    ) -> torch.Tensor:
        """Run the causal coupling GEMM.

        Args:
            C: (batch, seq_len, n_groups, d_state)  fp16/bf16
            B: (batch, seq_len, n_groups, d_state)  fp16/bf16

        Returns:
            cb: (batch, seq_len // chunk_size, n_groups, chunk_size, chunk_size)
                in the input dtype, zero above each chunk's diagonal.
        """
        backend = get_device_backend()
        if not backend.is_device_tensor(C):
            raise ValueError(f"C must be a {backend.name} tensor, got device {C.device}")
        if not backend.is_device_tensor(B):
            raise ValueError(f"B must be a {backend.name} tensor, got device {B.device}")
        if C.ndim != 4:
            raise ValueError("C must have shape [batch, seq_len, n_groups, d_state]")
        if B.shape != C.shape:
            raise ValueError(
                f"B must have the same shape as C, got {tuple(B.shape)} vs {tuple(C.shape)}"
            )
        if C.dtype != B.dtype:
            raise ValueError(f"C and B must share a dtype, got {C.dtype} vs {B.dtype}")

        batch, seq_len, n_groups, d_state = C.shape
        if seq_len % self.chunk_size != 0:
            raise ValueError(
                f"seq_len ({seq_len}) must be divisible by chunk_size ({self.chunk_size})"
            )
        num_chunks = seq_len // self.chunk_size

        self.batch = batch
        self.num_chunks = num_chunks
        self.n_groups = n_groups
        self.d_state = d_state
        self.dtype = C.dtype
        self.kernel = self._get_kernel(
            batch,
            num_chunks,
            n_groups,
            d_state,
            C.dtype,
            C.device.index,
        )

        C = C.contiguous()
        B = B.contiguous()

        return self.kernel(C, B)

    def eval_roofline(self) -> tuple[int, int]:
        """Return ``(flops, bytes)`` for this op instance.

        ``flops`` is the GPU manifest formula, copied verbatim:
        ``Bsz * NC * G * chunk_size * chunk_size * N`` (the causal mask is
        already folded in -- only the lower triangle is priced).  ``bytes`` is
        derived from the declared signature: read C and B, write cb.
        """
        if self.batch is None or self.dtype is None:
            raise RuntimeError(
                f"{type(self).__name__}.eval_roofline() requires a prior forward() "
                "call to bind dynamic input shape"
            )
        batch = int(self.batch)
        num_chunks = int(self.num_chunks)
        n_groups = int(self.n_groups)
        d_state = int(self.d_state)
        chunk_len = int(self.chunk_size)
        elem_bytes = self.dtype.itemsize

        seq_len = num_chunks * chunk_len
        flops = batch * num_chunks * n_groups * chunk_len * chunk_len * d_state
        nbytes = (
            2 * batch * seq_len * n_groups * d_state * elem_bytes
            + batch * num_chunks * n_groups * chunk_len * chunk_len * elem_bytes
        )
        return int(flops), int(nbytes)
