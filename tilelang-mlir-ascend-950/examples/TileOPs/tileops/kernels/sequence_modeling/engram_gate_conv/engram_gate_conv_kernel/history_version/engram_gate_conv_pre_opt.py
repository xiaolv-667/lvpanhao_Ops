"""Engram GateConv forward kernel (post-projection fusion) — NPU adaptation.

Assumes the projection (``k = E @ W_K``, ``v = E @ W_V``) is done externally via
GEMM.  This kernel fuses the remaining memory-bound operations::

    h_norm = RMSNorm(H, w_h)                                   # (M, T, d)
    k_norm = RMSNorm(k, w_h)                                   # (M, T, d)
    alpha  = sigmoid(dot(h_norm, k_norm) / sqrt(d))             # (M, T)
    v_hat  = alpha * v                                          # (M, T, d)
    v_hat_norm = RMSNorm(v_hat, w_v)                            # (M, T, d)
    conv_out = CausalDWConv1D(v_hat_norm, conv_w, kernel=4)     # (M, T, d)
    Y = SiLU(conv_out) + v_hat                                  # (M, T, d)

Saved for backward: ``alpha, rrms_h, rrms_k, rrms_v`` — all fp32 ``(M, T)``.

Adaptation summary (GPU -> NPU):

  **Part A — TileLang kernel functions** (extracted + imported):
    The single GPU TileLang kernel factory
    (``_engram_gate_conv_fwd_kernel``) was extracted from
    ``{gpu_repo_root}/tileops/_kernels/sequence_modeling/engram/gate_conv_fwd.py``
    by ``extract_tl_kernel.py`` (pattern A) and is imported as-is.  It serves as
    the reference for the NPU kernel component to reimplement for
    ``target="npuir"``.  K1-K4 and K11 adaptations (decorator, grid/sync
    semantics, ``threads`` removal, padding strategy) belong to that
    re-implementation.  Two names the factory reads are not emitted by the
    extractor and were appended to the extracted file so the package is
    self-contained: the ``ALIGNMENT`` constant (emitted) and the ``align_up``
    helper (appended verbatim from the GPU ``tileops/_kernels/tiling.py``).  The
    GPU ``@target_cache`` build-target decorator is not part of the extracted
    factory either, which is what keeps the package free of GPU build plumbing.

  **Part B — custom_op wrapper + Kernel class** (fully ported):
    K5: ``supported_archs = None`` (was ``[80, 86, 89, 90]``).
    K7: ``custom_op("npub::engram_gate_conv_fwd")`` replaces the GPU
        ``self.program(...)`` launch path — the NPU project has no
        ``Entry``/``KernelInterface``/``target_cache`` machinery, so the
        wrapper is the single launch surface.  The ``register_fake`` body returns
        all six outputs with their declared shapes and dtypes.
    K8: ``autotune_configs`` removed; heuristic config selection only
        (``init_config(config)``).
    K9: ``threads`` removed from the wrapper signature, from ``default_config``
        and from the dispatch (the GPU default was ``threads=128``).  Nothing
        backend-meaningful remains in the config, so ``default_config`` is empty
        and the factory is invoked with no launch-parameter level at all.

  ``ascend_mode`` is intentionally left ``None`` (defer to the ambient
  environment): the ascend backend carries no Developer/Expert declaration.

  Stage-3 note: the NPU kernel component (``_engram_gate_conv_fwd_kernel``,
  ``target="ascend"``) collapses the GPU factory's two call levels into one --
  the factory itself returns the JIT-compiled kernel, so the dispatch is
  ``_engram_gate_conv_fwd_kernel(M, seq_len, d, eps, dtype)(H, k, v, ...)``
  with no launch-parameter level (K3/K9).  Its prim_func also takes the
  *unpadded* row width ``d`` (tail lanes are masked inside the kernel,
  DESIGN.md 0.6 R2), so the GPU-era host-side ``F.pad`` + trim round trip is
  gone from ``forward``: the six input tensors and the two dtype-shaped outputs
  travel with zero host copies.
"""

from typing import Optional

import torch

from tileops.kernels.kernel_base import Kernel

# Part A: GPU TileLang kernel factory extracted by extract_tl_kernel.py
# (pattern A) from {gpu_repo_root}/tileops/_kernels/sequence_modeling/engram/
# gate_conv_fwd.py; reimplemented for ``target="ascend"`` by the NPU kernel
# component, which took over the GPU ``ALIGNMENT`` padding contract internally
# (UB row buffers are ``DW = align_up(d, 64)`` wide, global IO keeps ``d``).
# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# Default policy: the perf_opt source becomes active once tuned drop-in
# kernels (same factory signatures) land at
# .engram_gate_conv_kernel/perf_opt/{func}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
# and ``pytest benchmarks/ops/`` dispatch through whichever source is
# active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
from .engram_gate_conv_kernel import _engram_gate_conv_fwd_kernel

# --- perf_opt (Stage 4 tuned) ------------------------------------------------
# from .engram_gate_conv_kernel.perf_opt._engram_gate_conv_fwd_kernel import _engram_gate_conv_fwd_kernel

__all__ = ["EngramGateConvFwdKernel"]


# ---------------------------------------------------------------------------
# custom_op wrapper (K7: npub:: wrapper replaces the GPU ``self.program``
# launch path; K9: threads removed)
# ---------------------------------------------------------------------------


@torch.library.custom_op("npub::engram_gate_conv_fwd", mutates_args=())
def _engram_gate_conv_fwd_wrapped(
    M: int,
    seq_len: int,
    d: int,
    eps: float,
    dtype_str: str,
    H: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rms_w_h: torch.Tensor,
    rms_w_v: torch.Tensor,
    conv_w: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    # Stage 3 factory returns the compiled kernel directly (no launch-parameter
    # level, K3/K9), so the six inputs are passed straight into it.
    # ``out_idx=[6, 7, 8, 9, 10, 11]`` on the factory: the six outputs are
    # allocated by the JIT wrapper, so only the six inputs are passed.
    return _engram_gate_conv_fwd_kernel(M, seq_len, d, eps, dtype_str)(
        H, k, v, rms_w_h, rms_w_v, conv_w
    )


@_engram_gate_conv_fwd_wrapped.register_fake
def _(
    M: int,
    seq_len: int,
    d: int,
    eps: float,
    dtype_str: str,
    H: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rms_w_h: torch.Tensor,
    rms_w_v: torch.Tensor,
    conv_w: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    # Y, vhat: (M, seq_len, d) in the input dtype; alpha, rrms_h, rrms_k,
    # rrms_v: (M, seq_len) fp32.
    Y = H.new_empty((M, seq_len, d), dtype=H.dtype)
    vhat = H.new_empty((M, seq_len, d), dtype=H.dtype)
    alpha = H.new_empty((M, seq_len), dtype=torch.float32)
    rrms_h = H.new_empty((M, seq_len), dtype=torch.float32)
    rrms_k = H.new_empty((M, seq_len), dtype=torch.float32)
    rrms_v = H.new_empty((M, seq_len), dtype=torch.float32)
    return Y, vhat, alpha, rrms_h, rrms_k, rrms_v


# ---------------------------------------------------------------------------
# Kernel class (K5/K8/K9 adaptations)
# ---------------------------------------------------------------------------


class EngramGateConvFwdKernel(Kernel):
    """Engram GateConv forward kernel (post-projection fusion).

    NPU adaptations:
      K5: ``supported_archs = None`` (all architectures; GPU pinned CUDA SM
          ints).
      K8: autotune removed — heuristic config selection only;
          ``init_config(config)`` takes no ``tune`` argument.
      K9: ``threads`` removed from ``default_config`` and from the dispatch
          (GPU default was ``threads=128``); nothing backend-meaningful is left
          in the config.
    """

    ascend_mode = None

    # K5: [80, 86, 89, 90] (CUDA SM) -> None (all architectures).
    supported_archs: Optional[list] = None

    def __init__(
        self,
        M: int,
        seq_len: int,
        d: int,
        eps: float,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        device_index: int | None = None,
    ) -> None:
        super().__init__()
        self.M = M
        self.seq_len = seq_len
        self.d = d
        self.eps = eps
        self.dtype = dtype
        self.device_index = device_index
        self.kernel = _engram_gate_conv_fwd_kernel(
            M,
            seq_len,
            d,
            eps,
            self.dtype_str,
        )
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        # GPU default was {"threads": 128}; K9 drops ``threads`` and the kernel
        # has no other launch knob, so the config is empty.
        return {}

    def forward(
        self,
        H: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        rms_w_h: torch.Tensor,
        rms_w_v: torch.Tensor,
        conv_w: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Run the fused gate + conv forward pass.

        Args:
            H: (M, seq_len, d) — hidden states from preceding layers.
            k: (M, seq_len, d) — key projection output (E @ W_K).
            v: (M, seq_len, d) — value projection output (E @ W_V).
            rms_w_h: (d,) — RMSNorm weight for H and k.
            rms_w_v: (d,) — RMSNorm weight for the gated value (conv input).
            conv_w: (4, d) — depthwise causal conv1d weights.

        Returns:
            Tuple ``(Y, vhat, alpha, rrms_h, rrms_k, rrms_v)``.
        """
        # K9: no `threads` in the dispatch.  The prim_func consumes the
        # unpadded row width `d` (tail lanes masked in-kernel), so the GPU-era
        # host-side pad / trim round trip is not reproduced here.
        return _engram_gate_conv_fwd_wrapped(
            self.M,
            self.seq_len,
            self.d,
            self.eps,
            self.dtype_str,
            H,
            k,
            v,
            rms_w_h,
            rms_w_v,
            conv_w,
        )
