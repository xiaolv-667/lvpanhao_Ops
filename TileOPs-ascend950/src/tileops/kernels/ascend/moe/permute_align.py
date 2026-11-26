"""Block-aligned MoE token routing for Ascend 950, using TileLang's Ascend dialect.

One AI core owns the whole call: it counts the routes per expert, scans the padded counts,
names each block's owner and writes every route index into its expert's segment. The route
count and expert count are compile-time facts, so the metadata arrays are sized once and no
cross-core scratch or barrier is needed. A route reads as ``num_routes`` where an expert's
segment is padded and a block past the padded total names no expert.
"""

import functools
import math
from typing import ClassVar

import tilelang
import tilelang.ascend.language as T
import torch

from tileops.backend import register_kernel_type
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.moe.call_spec import PermuteAlignCall, PermuteAlignFwdInterface

__all__ = ["AscendMoEPermuteAlignKernel"]

# Rows one Unified Buffer tile holds: the sentinel fill writes ``_FILL_TILE`` slots per copy
# and the two route passes stage ``_STAGE_TILE`` routes per copy.
_FILL_TILE = 2048
_STAGE_TILE = 1024
# Experts one build serves. The three Unified Buffer metadata arrays hold one entry per
# expert plus the scan's total, so a larger count would not fit the buffer.
_MAX_EXPERTS = 4096


def make_align_kernel(num_routes: int, num_experts: int, block_size: int):
    """Sort the flat route list by expert, padding each expert's run to ``block_size``.

    ``s_cnt`` counts the routes per expert, ``s_pref`` holds their padded exclusive prefix
    sum, and ``s_live`` is the cursor each route advances inside its own expert's segment.
    Routes are visited in ascending index order, so a segment holds its routes in that order
    and two calls on one input return the same arrays.
    """
    # Worst-case padded length, and the blocks that length spans.
    max_padded = num_routes + (num_experts + 1) * (block_size - 1)
    num_blocks = math.ceil(max_padded / block_size)
    # The scan writes one entry past the last expert; one vector row of slack keeps the
    # metadata arrays at a fixed multiple of the vector width.
    expert_pad = max(64, math.ceil((num_experts + 1) / 64) * 64)
    stage = max(64, min(_STAGE_TILE, math.ceil(max(num_routes, 1) / 64) * 64))
    full_tiles = num_routes // stage
    tail = num_routes - full_tiles * stage
    fill_tiles = math.ceil(max_padded / _FILL_TILE)

    @T.prim_func
    def main(
        flat: T.Tensor((num_routes,), "int32"),
        sorted_token_ids: T.Tensor((max_padded,), "int32"),
        expert_ids: T.Tensor((num_blocks,), "int32"),
        num_tokens_post_pad: T.Tensor((1,), "int32"),
    ):
        with T.Kernel(1):
            s_cnt = T.alloc_shared((expert_pad,), "int32")
            s_pref = T.alloc_shared((expert_pad,), "int32")
            s_live = T.alloc_shared((expert_pad,), "int32")
            s_route = T.alloc_shared((stage,), "int32")
            s_fill = T.alloc_shared((_FILL_TILE,), "int32")

            for expert in T.serial(expert_pad):
                s_cnt[expert] = 0
                s_live[expert] = 0

            # A slot no route reaches reads as ``num_routes``.
            T.fill(s_fill, num_routes)
            for tile in T.serial(fill_tiles):
                base = tile * _FILL_TILE
                count = T.min(_FILL_TILE, max_padded - base)
                T.copy(s_fill[0:count], sorted_token_ids[base : base + count])

            for tile in T.serial(full_tiles):
                T.copy(flat[tile * stage : (tile + 1) * stage], s_route)
                for lane in T.serial(stage):
                    expert = s_route[lane]
                    s_cnt[expert] = s_cnt[expert] + 1
            if tail > 0:
                T.copy(flat[full_tiles * stage : num_routes], s_route[0:tail])
                for lane in T.serial(tail):
                    expert = s_route[lane]
                    s_cnt[expert] = s_cnt[expert] + 1

            # Each expert's run starts on a block boundary, so ``s_pref`` is the exclusive
            # padded prefix sum and its last entry the padded route count.
            total = T.alloc_var("int32", init=0)
            for expert in T.serial(num_experts):
                s_pref[expert] = total
                total = total + T.ceildiv(s_cnt[expert], block_size) * block_size
            s_pref[num_experts] = total
            num_tokens_post_pad[0] = total

            # A block past the padded total names no expert.
            for blk in T.serial(num_blocks):
                expert_ids[blk] = num_experts
            for expert in T.serial(num_experts):
                first = s_pref[expert] // block_size
                stop = s_pref[expert + 1] // block_size
                for blk in T.serial(stop - first):
                    expert_ids[first + blk] = expert

            for tile in T.serial(full_tiles):
                T.copy(flat[tile * stage : (tile + 1) * stage], s_route)
                for lane in T.serial(stage):
                    expert = s_route[lane]
                    slot = s_pref[expert] + s_live[expert]
                    s_live[expert] = s_live[expert] + 1
                    sorted_token_ids[slot] = tile * stage + lane
            if tail > 0:
                T.copy(flat[full_tiles * stage : num_routes], s_route[0:tail])
                for lane in T.serial(tail):
                    expert = s_route[lane]
                    slot = s_pref[expert] + s_live[expert]
                    s_live[expert] = s_live[expert] + 1
                    sorted_token_ids[slot] = full_tiles * stage + lane

    return main


@functools.lru_cache(maxsize=None)
def _compile(num_routes: int, num_experts: int, block_size: int):
    return tilelang.compile(
        make_align_kernel(num_routes, num_experts, block_size),
        target="ascend",
        out_idx=[],
    )


class AscendMoEPermuteAlignKernel(Kernel, PermuteAlignFwdInterface):
    """Route each flat route index into its expert's block-aligned segment."""

    devices: ClassVar[frozenset[str]] = frozenset({"npu"})
    supported_archs = None

    @classmethod
    def refusal(cls, call: PermuteAlignCall) -> str | None:
        if call.num_routes < 0:
            return "requires a non-negative route count"
        if call.num_experts <= 0 or call.block_size <= 0:
            return "requires positive expert and block counts"
        if call.num_experts > _MAX_EXPERTS:
            return f"supports at most {_MAX_EXPERTS} experts"
        return super().refusal(call)

    def __init__(self, call: PermuteAlignCall) -> None:
        reason = self.refusal(call)
        if reason is not None:
            raise ValueError(reason)
        super().__init__(device_index=call.device.index if call.device is not None else None)
        self.call = call
        self._align = _compile(call.num_routes, call.num_experts, call.block_size)

    def forward(self, topk_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._require_devices(topk_ids=topk_ids)
        c = self.call
        if topk_ids.dtype != torch.int32:
            raise ValueError("topk_ids must be int32")
        if topk_ids.numel() != c.num_routes:
            raise ValueError(
                f"topk_ids holds {topk_ids.numel()} routes, the specialization {c.num_routes}"
            )
        flat = topk_ids.contiguous().view(-1)
        device = flat.device
        max_padded = c.num_routes + (c.num_experts + 1) * (c.block_size - 1)
        sorted_token_ids = torch.empty(max_padded, dtype=torch.int32, device=device)
        expert_ids = torch.empty(
            math.ceil(max_padded / c.block_size), dtype=torch.int32, device=device
        )
        num_tokens_post_pad = torch.empty(1, dtype=torch.int32, device=device)
        self._align(flat, sorted_token_ids, expert_ids, num_tokens_post_pad)
        return sorted_token_ids, expert_ids, num_tokens_post_pad


register_kernel_type("MoEPermuteAlignFwdOp", "ascend", AscendMoEPermuteAlignKernel)
