"""Ascend MoE fused top-k routing.

Each AI core owns a strided set of tokens.  For one token it stages the router row in the
Unified Buffer, scores it in float32 (softmax or sigmoid) on the vector engine, applies the
optional correction bias and grouped-routing mask, then selects the ``top_k`` experts with a
serial argmax whose strict comparison keeps ties on the lower expert index.  Weights are the
unbiased scores, renormalised and scaled on the way out.

A call without a correction bias passes a zero row, so one compiled program serves both call
shapes; the compile-time ``with_correction_bias`` flag still selects the single-best group
score the unbiased grouped-routing reference uses.
"""

import functools
from typing import ClassVar, Optional

import tilelang
import tilelang.ascend.language as T
import torch

from tileops.backend import register_kernel_type
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.moe.call_spec import FusedTopKCall, FusedTopKFwdInterface

__all__ = ["AscendFusedTopKKernel"]

_SCORING_FUNCS = ("softmax", "sigmoid")


@functools.lru_cache(maxsize=64)
def _fused_topk_kernel(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    scoring_func: str,
    dtype: str,
    with_correction_bias: bool,
    renormalize: bool,
    num_expert_group: int,
    topk_group: int,
    routed_scaling_factor: float,
):
    """Build one routing kernel for a fixed shape and semantic configuration."""
    assert scoring_func in _SCORING_FUNCS
    # Unified-buffer rows are 32-byte aligned: pad the row to a vector of the input dtype.
    align = 16 if dtype in ("float16", "bfloat16") else 8
    experts_pad = (num_experts + align - 1) // align * align
    groups_pad = max(8, (num_expert_group + 7) // 8 * 8)
    k_pad = max(8, (top_k + 7) // 8 * 8)
    group_size = num_experts // num_expert_group
    blocks = min(num_tokens, 64)
    steps = (num_tokens + blocks - 1) // blocks
    # The scoring fragment's layout needs the thread count to divide its extent.
    threads = min(256, experts_pad)
    while experts_pad % threads:
        threads -= 1
    scaled = routed_scaling_factor != 1.0
    grouped = num_expert_group > 1
    softmax = scoring_func == "softmax"
    neg_inf = -T.infinity("float32")

    @T.prim_func
    def main(
        gating_output: T.Tensor((num_tokens, num_experts), dtype),
        correction_bias: T.Tensor((num_experts,), "float32"),
        topk_weights: T.Tensor((num_tokens, top_k), "float32"),
        topk_ids: T.Tensor((num_tokens, top_k), "int32"),
    ):
        with T.Kernel(blocks) as bx:
            row = T.alloc_shared((experts_pad,), dtype)
            weight = T.alloc_shared((experts_pad,), "float32")
            select = T.alloc_shared((experts_pad,), "float32")
            group_score = T.alloc_shared((groups_pad,), "float32")
            group_kept = T.alloc_shared((groups_pad,), "int32")
            chosen_w = T.alloc_shared((k_pad,), "float32")

            for step in T.serial(steps):
                token = step * blocks + bx
                if token < num_tokens:
                    T.copy(gating_output[token, :], row[:num_experts])

                    # Score the row in float32 on the vector engine.  ``weight`` holds the
                    # unbiased score the output carries; ``select`` adds the correction bias.
                    with T.SimtVF(threads=threads):
                        logit = T.alloc_fragment((experts_pad,), "float32")
                        for e in T.Parallel(experts_pad):
                            logit[e] = T.if_then_else(
                                e < num_experts, T.cast(row[e], "float32"), neg_inf
                            )
                        if softmax:
                            row_max = T.alloc_fragment((1,), "float32")
                            T.reduce_max(logit, row_max, dim=0)
                            for e in T.Parallel(experts_pad):
                                logit[e] = T.exp(logit[e] - row_max[0])
                            row_sum = T.alloc_fragment((1,), "float32")
                            T.reduce_sum(logit, row_sum, dim=0)
                            for e in T.Parallel(num_experts):
                                weight[e] = logit[e] / row_sum[0]
                                select[e] = weight[e] + correction_bias[e]
                        else:
                            for e in T.Parallel(num_experts):
                                weight[e] = 1.0 / (1.0 + T.exp(-logit[e]))
                                select[e] = weight[e] + correction_bias[e]

                    # Grouped routing: score each group, keep the ``topk_group`` best.
                    if grouped:
                        for g in T.serial(num_expert_group):
                            best1 = T.alloc_var("float32", init=neg_inf)
                            best2 = T.alloc_var("float32", init=neg_inf)
                            for e in T.serial(group_size):
                                v = select[g * group_size + e]
                                if v > best1:
                                    best2 = best1
                                    best1 = v
                                else:
                                    if v > best2:
                                        best2 = v
                            if with_correction_bias:
                                group_score[g] = best1 + best2
                            else:
                                group_score[g] = best1
                        for g in T.serial(num_expert_group):
                            group_kept[g] = 0
                        for _ in T.serial(topk_group):
                            pick = T.alloc_var("int32", init=0)
                            pick_val = T.alloc_var("float32", init=neg_inf)
                            for g in T.serial(num_expert_group):
                                if group_kept[g] == 0:
                                    if group_score[g] > pick_val:
                                        pick = g
                                        pick_val = group_score[g]
                            group_kept[pick] = 1
                        for e in T.serial(num_experts):
                            if group_kept[e // group_size] == 0:
                                select[e] = neg_inf

                    # Top-k by serial argmax; strict ``>`` leaves ties on the lower index.
                    # Selected weights stay in the Unified Buffer and are written once, so
                    # the kernel never reads back a value it has just written.
                    selected_sum = T.alloc_var("float32", init=0.0)
                    for k in T.serial(top_k):
                        best = T.alloc_var("float32", init=neg_inf)
                        best_idx = T.alloc_var("int32", init=0)
                        for e in T.serial(num_experts):
                            if select[e] > best:
                                best = select[e]
                                best_idx = e
                        topk_ids[token, k] = best_idx
                        chosen_w[k] = weight[best_idx]
                        selected_sum = selected_sum + weight[best_idx]
                        select[best_idx] = neg_inf

                    factor = T.alloc_var("float32", init=1.0)
                    if renormalize:
                        if scaled:
                            factor = T.cast(routed_scaling_factor, "float32") / selected_sum
                        else:
                            factor = 1.0 / selected_sum
                    else:
                        if scaled:
                            factor = T.cast(routed_scaling_factor, "float32")
                    for k in T.serial(top_k):
                        topk_weights[token, k] = chosen_w[k] * factor

    return main


@functools.lru_cache(maxsize=64)
def _compile_fused_topk_kernel(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    scoring_func: str,
    dtype: str,
    with_correction_bias: bool,
    renormalize: bool,
    num_expert_group: int,
    topk_group: int,
    routed_scaling_factor: float,
):
    return tilelang.compile(
        _fused_topk_kernel(
            num_tokens,
            num_experts,
            top_k,
            scoring_func,
            dtype,
            with_correction_bias,
            renormalize,
            num_expert_group,
            topk_group,
            routed_scaling_factor,
        ),
        target="ascend",
    )


class AscendFusedTopKKernel(Kernel, FusedTopKFwdInterface):
    """MoE fused top-k routing on Ascend 950, one token at a time per AI core."""

    devices: ClassVar[frozenset[str]] = frozenset({"npu"})

    @classmethod
    def refusal(cls, call: FusedTopKCall) -> Optional[str]:
        if call.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return "requires float16, bfloat16 or float32 logits"
        if call.top_k <= 0 or call.top_k > call.num_experts:
            return "requires 0 < top_k <= num_experts"
        if call.num_expert_group < 1 or call.num_experts % call.num_expert_group:
            return "requires num_expert_group to divide num_experts"
        if not 1 <= call.topk_group <= call.num_expert_group:
            return "requires 1 <= topk_group <= num_expert_group"
        if call.with_correction_bias and call.scoring_func != "sigmoid":
            return "requires sigmoid scoring with a correction bias"
        return super().refusal(call)

    @classmethod
    def entry_for(cls, call: FusedTopKCall):
        index = call.device.index if call.device is not None else None
        args = (
            call.num_tokens,
            call.num_experts,
            call.top_k,
            call.scoring_func,
            call.dtype,
            call.with_correction_bias,
            call.renormalize,
            call.num_expert_group,
            call.topk_group,
            call.routed_scaling_factor,
        )
        return (args, index), lambda: cls(*args, device_index=index)

    def __init__(
        self,
        num_tokens: int,
        num_experts: int,
        top_k: int,
        scoring_func: str = "softmax",
        dtype: torch.dtype = torch.float32,
        with_correction_bias: bool = False,
        renormalize: bool = False,
        num_expert_group: int = 1,
        topk_group: int = 1,
        routed_scaling_factor: float = 1.0,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.top_k = top_k
        self.scoring_func = scoring_func
        self.dtype = dtype
        self.with_correction_bias = with_correction_bias
        self.renormalize = renormalize
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.routed_scaling_factor = float(routed_scaling_factor)
        self._zero_bias: Optional[torch.Tensor] = None
        self._compiled = _compile_fused_topk_kernel(
            num_tokens,
            num_experts,
            top_k,
            scoring_func,
            self.dtype_str,
            with_correction_bias,
            renormalize,
            num_expert_group,
            topk_group,
            self.routed_scaling_factor,
        )

    def _bias(self, correction_bias: Optional[torch.Tensor], device: torch.device):
        if correction_bias is not None:
            return correction_bias.contiguous()
        if self._zero_bias is None or self._zero_bias.device != device:
            self._zero_bias = torch.zeros(
                (self.num_experts,), dtype=torch.float32, device=device
            )
        return self._zero_bias

    def forward(
        self,
        gating_output: torch.Tensor,
        correction_bias: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._require_devices(gating_output=gating_output, correction_bias=correction_bias)
        device = gating_output.device
        topk_weights = torch.empty(
            (self.num_tokens, self.top_k), dtype=torch.float32, device=device
        )
        topk_ids = torch.empty((self.num_tokens, self.top_k), dtype=torch.int32, device=device)
        self._compiled(
            gating_output.contiguous(), self._bias(correction_bias, device), topk_weights, topk_ids
        )
        return topk_weights, topk_ids


register_kernel_type("FusedTopKFwdOp", "ascend", AscendFusedTopKKernel)
