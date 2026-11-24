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

  **Part A — TileLang kernel functions** (extracted + imported):
    The single GPU TileLang kernel factory
    (``_engram_gate_conv_bwd_kernel``) was extracted from
    ``{gpu_repo_root}/tileops/_kernels/sequence_modeling/engram/gate_conv_bwd.py``
    by ``extract_tl_kernel.py`` (pattern A) and is imported as-is.  It serves as
    the reference for the NPU kernel component to reimplement for
    ``target="npuir"``.  K1-K4 and K11 adaptations (decorator, grid/sync
    semantics, ``threads`` removal, padding strategy) belong to that
    re-implementation.  Two dependencies the factory reads were appended to the
    extracted file so the package is self-contained: the module-level constant
    ``CONV_KERNEL_SIZE`` and the ``align_up`` helper (both emitted/appended
    verbatim from the GPU repo).  The GPU ``@target_cache`` build-target
    decorator is not part of the extracted factory either, which is what keeps
    the package free of GPU build plumbing.

  **Part B — custom_op wrapper + Kernel class** (fully ported):
    K5: ``supported_archs = None`` (was ``[80, 86, 89, 90]``).
    K6: the ``dconv_w``-run heuristic took ``sm_count`` from the GPU call spec;
        on NPU the block count comes from the device's AI-core count
        (``_core_count``) and the ``tile_n``-style arithmetic stays in
        ``tileops.utils.tiling``.  The run length is a shape-derived heuristic
        only — the kernel component re-derives it.
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
  environment): the programming-mode choice belongs to the NPU kernel component.

  Pre-Stage-3 note: the imported factory is still the GPU (CUDA target)
  implementation whose inner callable takes ``threads``.  The dispatch below
  passes the NPU-shaped argument list — ``_engram_gate_conv_bwd_kernel(...)()``,
  i.e. no launch parameter (K3/K9).  It becomes callable once the NPU kernel
  component rewrites the factory for ``target="npuir"``; running ``forward``
  before that rewrite fails, which is the expected scaffold state.
"""

from typing import Optional

import torch
import torch.nn.functional as F

from tileops.device import get_device_backend
from tileops.kernels.kernel_base import Kernel
from tileops.utils.tiling import align_up

# Part A: GPU TileLang kernel factory extracted by extract_tl_kernel.py
# (pattern A) from {gpu_repo_root}/tileops/_kernels/sequence_modeling/engram/
# gate_conv_bwd.py.  ``ALIGNMENT`` is the row padding the prim_func declares.
from ._engram_gate_conv_bwd_kernels import (
    ALIGNMENT,
    CONV_KERNEL_SIZE,
    _engram_gate_conv_bwd_kernel,
)

__all__ = ["EngramGateConvBwdKernel"]


def _core_count(device_index: Optional[int] = None) -> int:
    """AI-core count the ``dconv_w`` reduction runs are sized to (K6).

    The GPU version took ``sm_count`` from the call spec.  On NPU the block count
    is the device's AI-core count, read from the backend that owns the query;
    when the property is unavailable the heuristic degrades to a single block
    instead of failing.
    """
    try:
        props = get_device_backend().get_device_properties(device_index)
    except Exception:  # a missing query must not break construction
        return 1
    for attr in ("cube_core_num", "ai_core_count", "aicore_count", "multi_processor_count"):
        value = getattr(props, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    return 1


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
    run_len: int,
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
    # GPU callable signature (pre-Stage-3): ``_func(threads)``; the NPU-shaped
    # call below takes no launch parameter (K3/K9) and binds 1:1 once the NPU
    # kernel component rewrites the factory for target="npuir".
    # ``out_idx=[12, 13, 14, 15, 16, 17, 18, 19]`` on the factory: eight outputs
    # are allocated by the JIT wrapper (the six declared gradients plus the two
    # ``dvhat`` scratch tensors), so only the twelve inputs are passed and the
    # scratch tensors are dropped here.
    results = _engram_gate_conv_bwd_kernel(M, seq_len, d, dtype_str, run_len)(
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
    run_len: int,
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
      K6: the ``dconv_w`` run length is sized to the device's AI-core count
          instead of the GPU ``sm_count``.
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
        self.d_padded = align_up(d, ALIGNMENT)
        self.run_len = self._run_len(M * seq_len, _core_count(device_index))
        self.kernel = _engram_gate_conv_bwd_kernel(
            M,
            seq_len,
            d,
            self.dtype_str,
            self.run_len,
        )
        self.init_config(config)

    @staticmethod
    def _run_len(tokens: int, core_count: int) -> int:
        """Tokens a dconv_w run holds; the reduction runs one CTA a (tap, run).

        The target run count is ``ceil(tokens / 8)``, capped at
        ``4 * core_count / KS`` so the reduction launches at most 4 CTAs per core.
        The run length is ``ceil(tokens / target)``; the reduction launches
        ``ceil(tokens / run_len)`` runs, the last of which may be shorter, and
        none is empty.  A run of about 8 rows spreads the d-wide atomic add that
        ends each CTA over the rows it reads.  (GPU: the same heuristic with the
        device's SM count as ``core_count``.)
        """
        target_run = 8
        ctas_per_core = 4
        runs = max(
            1, min(-(-tokens // target_run), ctas_per_core * core_count // CONV_KERNEL_SIZE)
        )
        return -(-tokens // runs)

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
        """Run the fused gate + conv backward pass; outputs are trimmed to ``d``.

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
        pad = self.d_padded - self.d
        if pad:
            dY = F.pad(dY, (0, pad))
            H = F.pad(H, (0, pad))
            k = F.pad(k, (0, pad))
            v = F.pad(v, (0, pad))
            vhat = F.pad(vhat, (0, pad))
            rms_w_h = F.pad(rms_w_h, (0, pad))
            rms_w_v = F.pad(rms_w_v, (0, pad))
            conv_w = F.pad(conv_w, (0, pad))
        # K9: no `threads` in the dispatch.
        results = list(
            _engram_gate_conv_bwd_wrapped(
                self.M,
                self.seq_len,
                self.d,
                self.dtype_str,
                self.run_len,
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
        if pad:
            results[0] = results[0][:, :, : self.d]
            results[1] = results[1][:, :, : self.d]
            results[2] = results[2][:, :, : self.d]
            results[3] = results[3][: self.d]
            results[4] = results[4][: self.d]
            results[5] = results[5][:, : self.d]
        return tuple(results)
