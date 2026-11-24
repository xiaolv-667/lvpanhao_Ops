"""Engram GateConv backward kernel — NPU adaptation.

Given ``dY`` (the gradient of the fused output) plus the forward's saved
intermediates (``vhat``, ``alpha``, ``rrms_h``, ``rrms_k``, ``rrms_v``) and the
forward inputs (``H``, ``k``, ``v``, the two RMSNorm weights and the four conv
taps), the kernel produces::

    dH, dk, dv          -- gradients w.r.t. the projection outputs and H
    drms_w_h, drms_w_v  -- gradients w.r.t. the RMSNorm weights
    dconv_w             -- gradient w.r.t. the conv weights   [4, d]

All three weight gradients accumulate over every token in fp32.

The GPU implementation stages the work in four launches, which the extracted
factory keeps as four ``T.Kernel`` regions of one prim_func:

    Pass 1:  SiLU + residual backward -> buffer ``d_silu``
    Pass 1a: reduce ``dconv_w`` from the buffered ``d_silu``, one run of tokens a CTA
    Pass 1b: conv transpose -> RMSNorm(vhat) backward -> ``dvhat``
    Pass 2:  gate backward -> RMSNorm(H) / RMSNorm(k) backward -> ``dH``, ``dk``, ``dv``

Adaptation summary (GPU -> NPU):

  **Part A — TileLang kernel functions** (extracted, then reimplemented):
    The single GPU TileLang kernel factory
    (``_engram_gate_conv_bwd_kernel``) was extracted from
    ``{gpu_repo_root}/tileops/_kernels/sequence_modeling/engram/gate_conv_bwd.py``
    by ``extract_tl_kernel.py`` (pattern A) as the reference; the NPU kernel
    component then reimplemented it for ``target="ascend"`` (K1-K4/K11:
    decorator, grid/sync semantics, ``threads`` removal, padding strategy) and
    that implementation is what ``_engram_gate_conv_bwd_kernel`` names here.
    The extracted GPU source ``_engram_gate_conv_bwd_kernels.py`` (with the
    ``CONV_KERNEL_SIZE`` constant and the ``align_up`` helper it needed) stays
    in the package only as the migration record; the NPU factory needs neither.
    The GPU ``@target_cache`` build-target decorator is not part of the
    extracted factory either, which is what keeps the package free of GPU build
    plumbing.

  **Part B — custom_op wrapper + Kernel class** (fully ported):
    K5: ``supported_archs = None`` (was ``[80, 86, 89, 90]``).
    K6: the ``dconv_w``-run heuristic took ``sm_count`` from the GPU call spec;
        the kernel component now derives the run length and launch grid itself
        from the shape plus the AI-core count, so the wrapper carries no
        ``run_len``/``_core_count`` glue (Stage 5 removed both).
    K7: ``custom_op("npub::engram_gate_conv_bwd")`` replaces the GPU
        ``self.program(...)`` launch path — the NPU project has no
        ``Entry``/``KernelInterface``/``target_cache`` machinery, so the wrapper
        is the single launch surface.
    K8: ``autotune_configs`` removed; heuristic config selection only
        (``init_config(config)``).
    K9: ``threads`` removed from the wrapper signature, from ``default_config``
        and from the dispatch (the GPU default was ``threads=128``).  Nothing
        backend-meaningful remains in the config, so ``default_config`` is empty
        and the factory's inner callable is invoked with no launch parameter.

  ``ascend_mode`` is intentionally left ``None`` (defer to the ambient
  environment): the ascend backend carries no Developer/Expert declaration.

  Stage-5 note (contract of the integrated Stage 3 kernel):
    The NPU kernel component (``_engram_gate_conv_bwd_kernel``,
    ``target="ascend"``) collapses the GPU factory's two call levels into one —
    the factory itself compiles both launches and returns ``run(*inputs) -> 6
    gradients``, so the dispatch is
    ``_engram_gate_conv_bwd_kernel(M, seq_len, d, dtype)(dY, H, ...)`` with no
    launch-parameter level and no ``run_len`` argument (the kernel re-derives
    the ``dconv_w`` run length and the launch grid from the shape).  Its
    prim_funcs also take the *unpadded* row width ``d`` (the UB rows are
    ``DW = align_up(d, 64)`` wide inside the kernel, global IO keeps ``d``), so
    the GPU-era host-side ``F.pad`` + trim round trip is gone from ``forward``.

    The declared refusal is the kernel's real bound, ``d <= 1024``
    (``MAX_HIDDEN_DIM``): the fused structure holds 152 B per padded channel and
    ``d <= 1024`` keeps >= 35 % of the UB free at ``seq_len = 512`` (DESIGN.md
    9.2 R-1).  The manifest's ``d <= 4096`` needs the staged fallback of DESIGN
    1.6.4 B1, which is not implemented, so this wrapper does **not** carry the
    manifest bound.  ``tileops/ops/`` still declares 4096 (out of Stage 5
    scope); the effective bound is ``min(4096, 1024) = 1024``, enforced here.
"""

from typing import Optional

import torch

from tileops.kernels.kernel_base import Kernel

# Part A: the NPU TileLang kernel factory, reimplemented for ``target="ascend"``
# by the NPU kernel component and integrated into ``engram_gate_conv_bwd_kernel/``
# (Stage 5).  The integrated factory supersedes the extracted GPU (CUDA target)
# source ``_engram_gate_conv_bwd_kernels.py``, which stays in the package only as
# the migration record: it declares the GPU ``ALIGNMENT`` padding contract and
# the ``CONV_KERNEL_SIZE`` constant, neither of which the NPU factory exposes
# (it derives ``DW = align_up(d, 64)`` internally and keeps ``KS`` private).
# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# Default policy: the perf_opt source becomes active once tuned drop-in
# kernels (same factory signatures) land at
# .engram_gate_conv_bwd_kernel/perf_opt/{func}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
# and ``pytest benchmarks/ops/`` dispatch through whichever source is
# active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
from .engram_gate_conv_bwd_kernel import _engram_gate_conv_bwd_kernel

# --- perf_opt (Stage 4 tuned) ------------------------------------------------
# from .engram_gate_conv_bwd_kernel.perf_opt._engram_gate_conv_bwd_kernel import (
#     _engram_gate_conv_bwd_kernel,
# )

__all__ = ["EngramGateConvBwdKernel"]

# Refusal declared for this migration (the kernel's real bound, not the
# manifest's 4096): DESIGN.md 9.2 R-1 sizes the fused structure at 152 B per
# SIMD-padded channel, and d <= 1024 keeps >= 35 % of the 248 KiB UB free at
# seq_len = 512.  The 4096 bound needs the staged fallback (DESIGN 1.6.4 B1),
# which is not implemented.
MAX_HIDDEN_DIM = 1024


def _refusal(dtype: torch.dtype, M: int, seq_len: int, d: int) -> Optional[str]:
    """The GPU ``refusal()`` contract, inlined for the NPU port.

    Returns the reason to refuse the call, or ``None`` when it is admissible.
    The bound is the integrated kernel's own (``MAX_HIDDEN_DIM``), because
    beyond it the kernel's ``_check`` raises anyway -- refusing here converts
    that into one message at construction time, before anything is compiled.
    """
    if dtype not in (torch.float16, torch.bfloat16):
        return f"runs float16 and bfloat16, got {dtype}"
    if M <= 0 or seq_len <= 0 or d <= 0:
        return f"M, seq_len and d must be positive, got {M}, {seq_len}, {d}"
    if d > MAX_HIDDEN_DIM:
        return f"d must be <= {MAX_HIDDEN_DIM}, got {d}"
    return None


# ---------------------------------------------------------------------------
# custom_op wrapper (K7: npub:: wrapper replaces the GPU ``self.program``
# launch path; K9: threads removed)
# ---------------------------------------------------------------------------


@torch.library.custom_op("npub::engram_gate_conv_bwd", mutates_args=())
def _engram_gate_conv_bwd_wrapped(
    M: int,
    seq_len: int,
    d: int,
    dtype_str: str,
    dY: torch.Tensor,
    H: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rms_w_h: torch.Tensor,
    rms_w_v: torch.Tensor,
    conv_w: torch.Tensor,
    vhat: torch.Tensor,
    alpha: torch.Tensor,
    rrms_h: torch.Tensor,
    rrms_k: torch.Tensor,
    rrms_v: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    # Stage 5: the integrated factory compiles both launches and returns
    # ``run(*inputs) -> 6 gradients``; there is no launch-parameter level and no
    # ``run_len`` argument (K3/K9) -- the kernel re-derives the ``dconv_w`` run
    # length from the shape.  The fused launch declares ``out_idx=[12, 13, 14,
    # 15]`` (dh, dk, dv, PART) and the reduction ``out_idx=[1, 2, 3]`` (DWH, DWV,
    # DCW); the factory absorbs the PART workspace, so only the twelve inputs are
    # passed and the six gradients come back.
    results = _engram_gate_conv_bwd_kernel(M, seq_len, d, dtype_str)(
        dY,
        H,
        k,
        v,
        rms_w_h,
        rms_w_v,
        conv_w,
        vhat,
        alpha,
        rrms_h,
        rrms_k,
        rrms_v,
    )
    return tuple(results[:6])


@_engram_gate_conv_bwd_wrapped.register_fake
def _(
    M: int,
    seq_len: int,
    d: int,
    dtype_str: str,
    dY: torch.Tensor,
    H: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rms_w_h: torch.Tensor,
    rms_w_v: torch.Tensor,
    conv_w: torch.Tensor,
    vhat: torch.Tensor,
    alpha: torch.Tensor,
    rrms_h: torch.Tensor,
    rrms_k: torch.Tensor,
    rrms_v: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    # dH, dk, dv: (M, seq_len, d) in the input dtype; drms_w_h, drms_w_v:
    # (d,) fp32; dconv_w: (4, d) fp32.
    dH = dY.new_empty((M, seq_len, d), dtype=dY.dtype)
    dk = dY.new_empty((M, seq_len, d), dtype=dY.dtype)
    dv = dY.new_empty((M, seq_len, d), dtype=dY.dtype)
    drms_w_h = dY.new_empty((d,), dtype=torch.float32)
    drms_w_v = dY.new_empty((d,), dtype=torch.float32)
    dconv_w = dY.new_empty((conv_w.shape[0], d), dtype=torch.float32)
    return dH, dk, dv, drms_w_h, drms_w_v, dconv_w


# ---------------------------------------------------------------------------
# Kernel class (K5/K6/K8/K9 adaptations)
# ---------------------------------------------------------------------------


class EngramGateConvBwdKernel(Kernel):
    """Engram GateConv backward kernel; the module docstring lists its passes.

    NPU adaptations:
      K5: ``supported_archs = None`` (all architectures; GPU pinned CUDA SM
          ints).
      K6: the ``dconv_w`` run length is derived by the integrated kernel from
          the shape and the AI-core count (the GPU wrapper sized it from the
          call spec's ``sm_count``), so the wrapper carries no ``run_len``.
      K8: autotune removed — heuristic config selection only;
          ``init_config(config)`` takes no ``tune`` argument.
      K9: ``threads`` removed from ``default_config`` and from the dispatch
          (GPU default was ``threads=128``); nothing backend-meaningful is left
          in the config.
      Refusal: the GPU ``refusal()`` contract is inlined as ``_refusal()`` with
          the kernel's real bound ``d <= MAX_HIDDEN_DIM`` (1024), not the
          manifest's 4096 (see the module docstring's Stage-5 note).
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
        reason = _refusal(dtype, M, seq_len, d)
        if reason is not None:
            raise ValueError(reason)
        self.M = M
        self.seq_len = seq_len
        self.d = d
        self.eps = eps
        self.dtype = dtype
        self.device_index = device_index
        # Stage 5: the integrated factory compiles both launches for this
        # geometry and returns ``run(*inputs) -> 6 gradients``.  No launch
        # parameter level (K3/K9) and no ``run_len`` (K6: shape- and
        # AI-core-derived inside the kernel).
        self.kernel = _engram_gate_conv_bwd_kernel(
            M,
            seq_len,
            d,
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
        dY: torch.Tensor,
        H: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        rms_w_h: torch.Tensor,
        rms_w_v: torch.Tensor,
        conv_w: torch.Tensor,
        vhat: torch.Tensor,
        alpha: torch.Tensor,
        rrms_h: torch.Tensor,
        rrms_k: torch.Tensor,
        rrms_v: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Run the fused gate + conv backward pass; the six gradients come back as-is.

        Args:
            dY: (M, seq_len, d) — gradient of the fused output.
            H: (M, seq_len, d) — hidden states (forward input).
            k: (M, seq_len, d) — key projection (forward input).
            v: (M, seq_len, d) — value projection (forward input).
            rms_w_h: (d,) — RMSNorm weight for H and k.
            rms_w_v: (d,) — RMSNorm weight for vhat.
            conv_w: (4, d) — depthwise causal conv1d weights.
            vhat: (M, seq_len, d) — the forward's rounded gated value.
            alpha: (M, seq_len) fp32 — saved scalar gate.
            rrms_h: (M, seq_len) fp32 — reciprocal RMS of H.
            rrms_k: (M, seq_len) fp32 — reciprocal RMS of k.
            rrms_v: (M, seq_len) fp32 — reciprocal RMS of vhat.

        Returns:
            Tuple ``(dH, dk, dv, drms_w_h, drms_w_v, dconv_w)``.
        """
        # Stage 5: the integrated kernel takes the *unpadded* row width ``d``
        # (UB rows are ``DW = align_up(d, 64)`` wide inside the kernel and the
        # tail lanes are masked, DESIGN.md 0.6 R2), so there is no host-side
        # padding and no trim on the way out.
        # K9: no `threads` in the dispatch.
        results = list(
            _engram_gate_conv_bwd_wrapped(
                self.M,
                self.seq_len,
                self.d,
                self.dtype_str,
                dY,
                H,
                k,
                v,
                rms_w_h,
                rms_w_v,
                conv_w,
                vhat,
                alpha,
                rrms_h,
                rrms_k,
                rrms_v,
            )
        )
        return tuple(results)
