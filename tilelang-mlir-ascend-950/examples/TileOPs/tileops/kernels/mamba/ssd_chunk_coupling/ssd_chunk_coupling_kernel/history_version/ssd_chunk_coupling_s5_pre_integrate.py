"""Mamba-2 causal per-chunk ``C @ B^T`` coupling (CB) matrix kernel (NPU-adapted).

Produces the lower triangle of each ``(chunk, group)`` coupling matrix:

  cb[b, c, g, l, s] = sum_n C[b, c*Q + l, g, n] * B[b, c*Q + s, g, n]   for s <= l
  cb[b, c, g, l, s] = 0                                                 for s > l

Official-aligned interface (matches ``_bmm_chunk_fwd`` in mamba_ssm):

Inputs:
  C_mat: [B, S, G, N]   dtype (fp16/bf16)   seqlen-fused, group-owned
  B_mat: [B, S, G, N]   dtype (fp16/bf16)   seqlen-fused, group-owned
Output:
  cb:    [B, NC, G, Q, Q]  dtype            NC = S // Q, causal (lower-triangular)

Adaptation summary (GPU -> NPU):

  **Part A -- TileLang kernel functions** (extracted + imported):
    The GPU TileLang kernel factory (``_ssd_chunk_coupling_kernel``) is
    extracted from the GPU repo via ``extract_tl_kernel.py`` and imported
    as-is.  It serves as the reference for the NPU kernel component to
    reimplement for ``target="npuir"``.  K1-K4 adaptations (decorator,
    grid/sync semantics, ``threads`` removal, padding strategy) are handled
    by the NPU kernel component during re-implementation.  The GPU
    ``@target_cache`` build-target wrapper is not part of the extracted
    factory, so the package stays self-contained.

  **Part B -- custom_op wrapper + Kernel class** (fully ported):
    K5: ``supported_archs = None`` (was ``[80, 86, 89, 90]``).
    K7: ``custom_op("npub::ssd_chunk_coupling_fwd")`` (was the GPU
        ``self.program(...)`` launch path; there is no ``entry_for`` /
        ``KernelInterface`` machinery on the NPU side, so the wrapper is the
        single launch surface).
    K8: ``autotune_configs`` / ``tune`` removed; heuristic config selection
        only (``init_config(config)``).
    K9: ``threads`` removed from the wrapper signature, from
        ``default_config`` and from the dispatch (GPU default was
        ``threads=128``).

  ``ascend_mode`` is intentionally left ``None`` (defer to the ambient
  environment): the programming-mode choice belongs to the NPU kernel
  component, which decides Developer vs Expert together with the Cube tiling
  design.

  Pre-Stage-3 note: the imported factory is still the GPU (CUDA target)
  implementation whose inner callable takes ``(block_l, block_s, block_n,
  threads)``.  The dispatch below passes the NPU-shaped argument list (no
  ``threads``); it becomes callable once the NPU kernel component rewrites the
  factory for ``target="npuir"`` and drops ``threads`` from the callable
  signature (K3).  Running ``forward`` before that rewrite fails -- this is the
  expected scaffold state.
"""

from typing import Optional

import torch

from tileops.kernels.kernel_base import Kernel

# Part A: GPU TileLang kernel factory extracted by extract_tl_kernel.py
# (pattern A) from {gpu_repo_root}/tileops/_kernels/mamba/ssd_chunk_coupling.py.
from ._ssd_chunk_coupling_kernels import _ssd_chunk_coupling_kernel

__all__ = ["SSDChunkCouplingFwdKernel"]


# ---------------------------------------------------------------------------
# custom_op wrapper (K7: npub:: wrapper replaces the GPU ``self.program``
# launch path; K9: threads removed)
# ---------------------------------------------------------------------------


@torch.library.custom_op("npub::ssd_chunk_coupling_fwd", mutates_args=())
def _ssd_chunk_coupling_fwd_wrapped(
    batch: int,
    num_chunks: int,
    n_groups: int,
    chunk_len: int,
    d_state: int,
    dtype: str,
    block_l: int,
    block_s: int,
    block_n: int,
    C_mat: torch.Tensor,
    B_mat: torch.Tensor,
) -> torch.Tensor:
    # GPU callable signature (pre-Stage-3): (block_l, block_s, block_n,
    # threads).  The NPU-shaped call below omits `threads` (K3/K9); it binds
    # 1:1 once the NPU kernel component rewrites the factory for
    # target="npuir".
    return _ssd_chunk_coupling_kernel(
        batch,
        num_chunks,
        n_groups,
        chunk_len,
        d_state,
        dtype,
    )(block_l, block_s, block_n)(C_mat, B_mat)


@_ssd_chunk_coupling_fwd_wrapped.register_fake
def _(
    batch: int,
    num_chunks: int,
    n_groups: int,
    chunk_len: int,
    d_state: int,
    dtype: str,
    block_l: int,
    block_s: int,
    block_n: int,
    C_mat: torch.Tensor,
    B_mat: torch.Tensor,
) -> torch.Tensor:
    # output: [B, NC, G, Q, Q] in the input dtype
    return C_mat.new_empty(
        (batch, num_chunks, n_groups, chunk_len, chunk_len),
        dtype=C_mat.dtype,
    )


# ---------------------------------------------------------------------------
# Kernel class (K5/K8/K9 adaptations)
# ---------------------------------------------------------------------------


class SSDChunkCouplingFwdKernel(Kernel):
    """Mamba-2 causal per-chunk ``C @ B^T`` coupling (CB) matrix kernel.

    Computes ``cb[b,c,g,l,s] = sum_n C[b,c*Q+l,g,n] * B[b,c*Q+s,g,n]`` with
    causal masking (``cb[l,s] = 0`` when ``s > l``).

    NPU adaptations:
      K5: ``supported_archs = None`` (all architectures; GPU pinned CUDA
          SM ints).
      K8: autotune removed -- heuristic config selection only;
          ``init_config(config)`` takes no ``tune`` argument.
      K9: ``threads`` removed from ``default_config`` and from the
          ``forward`` dispatch (GPU default was ``threads=128``).
    """

    ascend_mode = None

    # K5: [80, 86, 89, 90] (CUDA SM) -> None (all architectures).
    supported_archs: Optional[list] = None

    def __init__(
        self,
        batch: int,
        num_chunks: int,
        n_groups: int,
        chunk_len: int,
        d_state: int,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        device_index: int | None = None,
    ) -> None:
        super().__init__()
        self.batch = batch
        self.num_chunks = num_chunks
        self.n_groups = n_groups
        self.chunk_len = chunk_len
        self.d_state = d_state
        self.dtype = dtype
        self.device_index = device_index

        self.kernel = _ssd_chunk_coupling_kernel(
            batch,
            num_chunks,
            n_groups,
            chunk_len,
            d_state,
            self.dtype_str,
        )
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        # GPU default minus ``threads`` (K9): the per-block element count is a
        # backend detail, resolved by the NPU kernel component.
        return {
            "block_l": 64,
            "block_s": 64,
            "block_n": 64,
        }

    def forward(
        self,
        C_mat: torch.Tensor,
        B_mat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            C_mat: [B, S, G, N]  dtype
            B_mat: [B, S, G, N]  dtype

        Returns:
            cb: [B, NC, G, Q, Q]  dtype
        """
        # K9: no `threads` in the dispatch.
        return _ssd_chunk_coupling_fwd_wrapped(
            self.batch,
            self.num_chunks,
            self.n_groups,
            self.chunk_len,
            self.d_state,
            self.dtype_str,
            self.config["block_l"],
            self.config["block_s"],
            self.config["block_n"],
            C_mat.contiguous(),
            B_mat.contiguous(),
        )
