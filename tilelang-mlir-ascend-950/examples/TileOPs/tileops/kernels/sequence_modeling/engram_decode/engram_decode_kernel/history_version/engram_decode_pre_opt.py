"""Engram fused single-token decode kernel — NPU adaptation.

One decode step of the Engram stack, end to end::

    k = e_t @ W_K                                              # (batch, d)
    v = e_t @ W_V                                              # (batch, d)
    h_norm = RMSNorm(h_t, rms_w_h)                             # (batch, d)
    k_norm = RMSNorm(k,   rms_w_h)                             # (batch, d)
    alpha  = sigmoid(dot(h_norm, k_norm) / sqrt(d))             # (batch,)
    v_hat  = alpha * v                                          # (batch, d)
    vhat_n = RMSNorm(v_hat, rms_w_v)                            # (batch, d)
    new_conv_state = shift(conv_state) ++ vhat_n                # (batch, max_conv_len, d)
    conv_out[t] = sum_p conv_w[p] * new_conv_state[-1 - (w-1-p)*dilation]
    y_t = SiLU(conv_out) + v_hat                                # (batch, d)

``conv_state`` holds the last ``L`` steps (``L <= max_conv_len``) and is left-padded
to ``max_conv_len`` internally, so a caller may pass any length up to the cache
capacity without a recompile.

Adaptation summary (GPU -> NPU):

  **Part A — TileLang kernel functions** (Stage 3, integrated):
    The GPU decode step is *two* TileLang launches.  Both were extracted from
    ``{gpu_repo_root}/tileops/_kernels/sequence_modeling/engram/decode.py`` into
    ``_engram_decode_fwd_kernels.py`` as the porting reference (the projection
    factory by ``extract_tl_kernel.py`` pattern A, the step factory appended by
    hand), and the Stage 3 NPU rewrites now live in the same package under
    ``engram_decode_kernel/``:

    * ``_engram_project_kernel`` — both projections (Cube GEMM over ``d``) plus the
      cache shift, writing ``kv`` as ``[k; v]`` in fp32;
    * ``_engram_step_kernel`` — the RMSNorms, gate, cache append, dilated causal
      conv and SiLU, one block per batch row (reduces over all of ``d``).

    Both are compiled with ``target="ascend"`` (K1-K4/K11: 1-D grid, mix-mode
    promotion, no ``threads`` on the project launch, ``ALIGNMENT = 256`` padding
    passed in as ``d_padded``).

  **Part B — custom_op wrapper + Kernel class** (fully ported):
    K5: ``supported_archs = None`` (was ``[80, 86, 89, 90]``).
    K7: ``custom_op("npub::engram_decode")`` replaces the GPU
        ``self.program(self._compile)`` launch path — the NPU project has no
        ``Entry``/``KernelInterface``/``target_cache`` machinery, so the wrapper is
        the single launch surface and it issues **both** launches in order
        (projection + cache shift, then the reduce-over-d step).
    K8: ``autotune_configs`` / ``autotune()`` / ``tune_jit_kernel`` removed;
        heuristic config selection only (``init_config(config)`` without ``tune``).
    K9: ``threads`` and ``step_threads`` removed from ``default_config`` (the GPU
        defaults were 128 for both).  The project tiling (``block_m``, ``block_n``,
        ``block_k``, ``num_stages``) is taken from the integrated project kernel's
        ``default_config(d_padded)`` — ``block_k = 256`` / ``num_stages = 2`` is the
        Ascend L0B solve, not the GPU ``_projection_block_k(d_mem)`` / 4-stage pair
        — and the step lane count is taken from the integrated step kernel's
        ``default_config(d_padded)["threads"]`` (``min(512, tile)``).

  ``ascend_mode`` is intentionally left ``None`` (defer to the ambient environment):
  the ascend backend has no Developer/Expert mode declaration, and this package has
  exactly one Kernel class.
"""

from typing import Optional

import torch
import torch.nn.functional as F

from tileops.kernels.kernel_base import Kernel
from tileops.utils.tiling import align_up

# Part A: Stage 3 NPU TileLang kernel factories, integrated into
# ``engram_decode_kernel/`` by ``integrate_kernel.py``.  ``ALIGNMENT`` (256) is the
# row padding both prim_funcs declare; ``d_padded = align_up(d, ALIGNMENT)``.
# ---------------------------------------------------------------------------
# Kernel source selection: baseline vs perf_opt (Stage 4 tuned)
#
# Exactly one source block below is active; toggle by swapping the comment.
# Default policy: the perf_opt source becomes active once tuned drop-in
# kernels (same factory signatures) land at
# .engram_decode_kernel/perf_opt/{func}.py and pass their L0/L1 regression;
# the baseline (Stage 3) source is active otherwise.  ``pytest tests/ops/``
# and ``pytest benchmarks/ops/`` dispatch through whichever source is
# active here.
# ---------------------------------------------------------------------------
# --- baseline (Stage 3) ------------------------------------------------------
from .engram_decode_kernel import ALIGNMENT, _engram_project_kernel, _engram_step_kernel

# --- frozen launch config: each integrated module's own ``default_config`` ----
# Read from the integrated modules so the wrapper cannot drift from the constants
# they were verified with:
#   * project: ``{"block_m": 16, "block_n": min(64, d_pad), "block_k": 256,
#     "num_stages": 2}`` (Stage 3 DESIGN.md 5.6; the GPU-derived
#     ``block_k=_projection_block_k(d_mem)`` / ``num_stages=4`` pair does not apply
#     to the Ascend ``block_n`` x ``block_k`` L0B solve);
#   * step: ``{"threads": min(512, next_pow2(d_pad))}``, the SIMT lane count.
# A Stage 4 perf_opt adoption that changes either tiling must repoint these two
# imports together with the factory imports above.
from .engram_decode_kernel._engram_project_kernel import default_config as _project_default_config
from .engram_decode_kernel._engram_step_kernel import default_config as _step_default_config

# --- perf_opt (Stage 4 tuned) ------------------------------------------------
# from .engram_decode_kernel.perf_opt._engram_project_kernel import ALIGNMENT
# from .engram_decode_kernel.perf_opt._engram_project_kernel import _engram_project_kernel
# from .engram_decode_kernel.perf_opt._engram_step_kernel import _engram_step_kernel

__all__ = ["EngramDecodeKernel"]


# ---------------------------------------------------------------------------
# custom_op wrapper (K7: npub:: wrapper replaces the GPU ``self.program`` launch
# path and issues both launches; K9: threads/step_threads removed)
# ---------------------------------------------------------------------------


@torch.library.custom_op("npub::engram_decode", mutates_args=())
def _engram_decode_wrapped(
    batch: int,
    d_mem: int,
    d: int,
    d_padded: int,
    max_conv_len: int,
    conv_kernel_size: int,
    dilation: int,
    eps: float,
    dtype_str: str,
    block_m: int,
    block_n: int,
    block_k: int,
    num_stages: int,
    e_t: torch.Tensor,
    h_t: torch.Tensor,
    conv_state: torch.Tensor,
    W_K: torch.Tensor,
    W_V: torch.Tensor,
    rms_w_h: torch.Tensor,
    rms_w_v: torch.Tensor,
    conv_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Launch 1 of 2: the two projections and the cache shift.  The integrated Stage 3
    # callable signature is ``_func(block_m, block_n, block_k, num_stages)`` -- no
    # thread count on this backend.  ``out_idx=[]`` on the factory: every output is
    # allocated here and passed in.
    project = _engram_project_kernel(batch, d_mem, d_padded, max_conv_len, dtype_str)(
        block_m, block_n, block_k, num_stages
    )
    # Launch 2 of 2: the norms, gate, cache append, dilated conv and SiLU.  The only
    # launch parameter is the SIMT lane count, read from the integrated step module's
    # frozen config (``min(512, next_pow2(d_padded))``); passing it explicitly keeps
    # one place to A/B it later.
    step = _engram_step_kernel(
        batch, d, d_padded, max_conv_len, conv_kernel_size, dilation, eps, dtype_str
    )(_step_default_config(d_padded)["threads"])

    # ``kv`` is the cross-launch fp32 staging buffer for [k; v].
    kv = torch.empty((2, batch, d_padded), dtype=torch.float32, device=e_t.device)
    y_t = torch.empty((batch, d_padded), dtype=e_t.dtype, device=e_t.device)
    new_conv_state = torch.empty(
        (batch, max_conv_len, d_padded), dtype=e_t.dtype, device=e_t.device
    )
    project(e_t, W_K, W_V, conv_state, kv, new_conv_state)
    step(h_t, kv, conv_state, rms_w_h, rms_w_v, conv_w, y_t, new_conv_state)
    return y_t, new_conv_state


@_engram_decode_wrapped.register_fake
def _(
    batch: int,
    d_mem: int,
    d: int,
    d_padded: int,
    max_conv_len: int,
    conv_kernel_size: int,
    dilation: int,
    eps: float,
    dtype_str: str,
    block_m: int,
    block_n: int,
    block_k: int,
    num_stages: int,
    e_t: torch.Tensor,
    h_t: torch.Tensor,
    conv_state: torch.Tensor,
    W_K: torch.Tensor,
    W_V: torch.Tensor,
    rms_w_h: torch.Tensor,
    rms_w_v: torch.Tensor,
    conv_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # y_t: (batch, d) in the input dtype; new_conv_state: (batch, max_conv_len, d).
    y_t = e_t.new_empty((batch, d_padded), dtype=e_t.dtype)
    new_conv_state = e_t.new_empty((batch, max_conv_len, d_padded), dtype=e_t.dtype)
    return y_t, new_conv_state


# ---------------------------------------------------------------------------
# Kernel class (K5/K8/K9 adaptations)
# ---------------------------------------------------------------------------


class EngramDecodeKernel(Kernel):
    """Engram fused decode kernel — full single-token pipeline.

    Two launches per call: the projections and the cache shift split over ``d``,
    then the RMSNorm gating, dilated causal conv and SiLU activation, which reduce
    over all of ``d``, one thread block per batch row.

    NPU adaptations:
      K5: ``supported_archs = None`` (all architectures; GPU pinned CUDA SM ints).
      K8: autotune removed — heuristic config selection only; ``init_config`` takes
          no ``tune`` argument.
      K9: ``threads`` / ``step_threads`` removed from ``default_config`` and from
          the dispatch (GPU defaults were 128 for both); the remaining entries are
          tiling heuristics.

    Args:
        batch: batch size (compile-time).
        d_mem: memory embedding dimension.
        d: model hidden dimension.
        max_conv_len: max conv cache capacity (compile-time).
        conv_kernel_size: number of conv taps w (model param, e.g. 4).
        dilation: dilation factor δ (model param, e.g. max N-gram order).
        eps: RMSNorm epsilon.
        dtype: data type.
        config: optional tiling-config override.
        device_index: optional device index (kept for the NPU kernel-constructor
            contract).
    """

    ascend_mode = None

    # K5: [80, 86, 89, 90] (CUDA SM) -> None (all architectures).
    supported_archs: Optional[list] = None

    def __init__(
        self,
        batch: int,
        d_mem: int,
        d: int,
        max_conv_len: int,
        conv_kernel_size: int,
        dilation: int,
        eps: float,
        dtype: torch.dtype,
        config: Optional[dict] = None,
        device_index: int | None = None,
    ) -> None:
        super().__init__()
        self.batch = batch
        self.d_mem = d_mem
        self.d = d
        self.max_conv_len = max_conv_len
        self.conv_kernel_size = conv_kernel_size
        self.dilation = dilation
        self.eps = eps
        self.dtype = dtype
        self.device_index = device_index
        self.d_padded = align_up(d, ALIGNMENT)

        min_cache = dilation * (conv_kernel_size - 1)
        if max_conv_len < min_cache:
            raise ValueError(
                f"max_conv_len ({max_conv_len}) must be >= "
                f"dilation * (conv_kernel_size - 1) = {min_cache}"
            )

        self.step_jit = _engram_step_kernel(
            batch,
            d,
            self.d_padded,
            max_conv_len,
            conv_kernel_size,
            dilation,
            eps,
            self.dtype_str,
        )
        self.kernel = _engram_project_kernel(
            batch,
            d_mem,
            self.d_padded,
            max_conv_len,
            self.dtype_str,
        )
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        # Stage 3 frozen tiling, read straight from the integrated project kernel so
        # the wrapper cannot drift from the constants it was verified with.  The GPU
        # default was {"block_m": 16, "block_n": min(64, d_padded), "block_k":
        # _projection_block_k(d_mem), "num_stages": 4, "threads": 128,
        # "step_threads": 128}; on this backend the L0B budget fixes
        # ``block_k = 256`` at ``block_n = 64`` / fp16 / 2 stages (DESIGN.md 5.6), and
        # the two thread counts live in ``block_k``-independent places: the step
        # kernel's own ``default_config(d_padded)["threads"]`` (min(512, tile)) is
        # passed at dispatch, and the project kernel takes no thread count.
        return _project_default_config(self.d_padded)

    def forward(
        self,
        e_t: torch.Tensor,
        h_t: torch.Tensor,
        conv_state: torch.Tensor,
        W_K: torch.Tensor,
        W_V: torch.Tensor,
        rms_w_h: torch.Tensor,
        rms_w_v: torch.Tensor,
        conv_w: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Run one fused decode step, applying and trimming the alignment padding the
        prim_funcs require.

        Args:
            e_t: (batch, d_mem) — gathered N-gram embedding of the current token.
            h_t: (batch, d) — hidden state of the current token.
            conv_state: (batch, L, d) — conv history, ``L <= max_conv_len``;
                left-padded to ``max_conv_len`` here when it is shorter.
            W_K: (d_mem, d) — key projection weight.
            W_V: (d_mem, d) — value projection weight.
            rms_w_h: (d,) — RMSNorm weight for ``h_t`` and the key projection.
            rms_w_v: (d,) — RMSNorm weight for the gated value.
            conv_w: (conv_kernel_size, d) — depthwise conv weights.

        Returns:
            Tuple ``(y_t, new_conv_state)``:
                y_t:            (batch, d) — output to add as a residual.
                new_conv_state: (batch, max_conv_len, d) — updated history.
        """
        state_pad = self.max_conv_len - conv_state.shape[1]
        if state_pad > 0:
            conv_state = F.pad(conv_state, (0, 0, state_pad, 0))
        pad = self.d_padded - self.d
        if pad:
            h_t = F.pad(h_t, (0, pad))
            conv_state = F.pad(conv_state, (0, pad))
            W_K = F.pad(W_K, (0, pad))
            W_V = F.pad(W_V, (0, pad))
            rms_w_h = F.pad(rms_w_h, (0, pad))
            rms_w_v = F.pad(rms_w_v, (0, pad))
            conv_w = F.pad(conv_w, (0, pad))
        cfg = self.config
        y_t, new_conv_state = _engram_decode_wrapped(
            self.batch,
            self.d_mem,
            self.d,
            self.d_padded,
            self.max_conv_len,
            self.conv_kernel_size,
            self.dilation,
            self.eps,
            self.dtype_str,
            cfg["block_m"],
            cfg["block_n"],
            cfg["block_k"],
            cfg["num_stages"],
            e_t,
            h_t,
            conv_state,
            W_K,
            W_V,
            rms_w_h,
            rms_w_v,
            conv_w,
        )
        if pad:
            return y_t[:, : self.d], new_conv_state[:, :, : self.d]
        return y_t, new_conv_state
