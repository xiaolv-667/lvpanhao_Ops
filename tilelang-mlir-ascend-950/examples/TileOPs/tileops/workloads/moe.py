"""Workload definitions for the moe op family (NPU).

Ported from the GPU TileOPs ``workloads/moe.py``: the
``MoEPermuteAlignWorkload`` class with its ``ref_permute_align`` helper, and
the ``SharedExpertMLPWorkload`` class with its ``ref_shared_expert`` /
``gated_activation`` helpers (the other moe workloads -- FusedTopK,
MoEPrePermute, MoEPostPermute, MoEGroupedGEMM, MoEExpertMLP,
FusedMoESharedExpert and the rest -- belong to ops that are not migrated yet).

The GPU workloads are ``CallWorkload``s: geometry, params and the value
distribution all come from the manifest row, and ``gen_inputs`` materializes
them.  The NPU manifest is data-only, so these classes take the geometry and
the op params directly and draw the same distributions here
(``topk_ids(T, K, num_experts)``: K distinct random experts per row; the
shared expert's ``randn`` tensors scaled by 0.1 / 0.02).

The only adaptation (W1) is device placement: the tensors are built on the
device resolved through :func:`tileops.device.get_device_backend` instead
of the GPU's hard-coded ``"cuda"``.  ``moe_close`` is the GPU
``_moe_close``, with its normalized-squared-error check (GPU
``workloads/numerics.py``) inlined because the NPU project has no numerics
module.
"""

import math

import torch

from tileops.device import get_device_backend
from tileops.workloads.workload_base import WorkloadBase

__all__ = [
    "MoEPermuteAlignWorkload",
    "SharedExpertMLPWorkload",
    "gated_activation",
    "moe_close",
    "ref_permute_align",
    "ref_shared_expert",
]


def ref_permute_align(
    topk_ids: torch.Tensor, block_size: int, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-Python reference for permute_align (ported verbatim from the GPU workload).

    Returns the routed ranges only: ``sorted_token_ids`` is
    ``num_tokens_post_pad`` long, ``expert_ids`` holds one owner per block of
    those ranges, and ``num_tokens_post_pad`` is a one-element tensor.  The
    caller pads both arrays out to the manifest's full lengths.
    """
    numel = topk_ids.numel()
    flat = topk_ids.flatten().tolist()

    counts = [0] * num_experts
    for eid in flat:
        counts[eid] += 1

    cumsum = [0] * (num_experts + 1)
    for i in range(num_experts):
        padded = math.ceil(counts[i] / block_size) * block_size
        cumsum[i + 1] = cumsum[i] + padded

    total_padded = cumsum[num_experts]
    sorted_token_ids = [numel] * total_padded

    slot = list(cumsum[:-1])
    for flat_idx, eid in enumerate(flat):
        sorted_token_ids[slot[eid]] = flat_idx
        slot[eid] += 1

    num_blocks = total_padded // block_size
    expert_ids_list = []
    for b in range(num_blocks):
        block_start = b * block_size
        lo, hi = 0, num_experts - 1
        eid = num_experts - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if cumsum[mid] <= block_start < cumsum[mid + 1]:
                eid = mid
                break
            elif block_start < cumsum[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        expert_ids_list.append(eid)

    device = topk_ids.device
    return (
        torch.tensor(sorted_token_ids, dtype=torch.int32, device=device),
        torch.tensor(expert_ids_list, dtype=torch.int32, device=device),
        torch.tensor([total_padded], dtype=torch.int32, device=device),
    )


class MoEPermuteAlignWorkload(WorkloadBase):
    """The routing ids of one ``MoEPermuteAlignFwdOp`` call.

    Args:
        tokens: Total routed tokens ``T`` (the first axis of ``topk_ids``).
        top_k: Routes per token ``K`` (its second axis).
        num_experts: Experts the routes select from.
        block_size: GEMM tile height each expert's routes are padded to.
        dtype: Dtype of ``topk_ids``; the index contract fixes int32.
    """

    def __init__(
        self,
        tokens: int,
        top_k: int,
        num_experts: int,
        block_size: int,
        dtype: torch.dtype = torch.int32,
    ) -> None:
        self.tokens, self.top_k = tokens, top_k
        self.num_experts, self.block_size = num_experts, block_size
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        """K distinct random experts per row, as the GPU generator draws them.

        ``argsort`` of one uniform draw per (token, expert) pair gives each row a
        random permutation, whose first K entries are K distinct experts in
        ``[0, num_experts)`` -- the distribution the GPU manifest's
        ``topk_ids(T, K, num_experts)`` generator samples.  Drawing the
        permutation instead of sampling K of E keeps the host out of the loop
        at the prefill token counts.
        """
        device = get_device_backend().name
        if not 0 < self.top_k <= self.num_experts:
            raise ValueError(
                f"topk_ids needs 0 < K <= E, got K={self.top_k}, E={self.num_experts}"
            )
        draw = torch.rand(self.tokens, self.num_experts, dtype=torch.float32, device=device)
        ids = torch.argsort(draw, dim=1)[:, : self.top_k]
        return (ids.to(self.dtype).contiguous(),)

    def ref_program(self, topk_ids: torch.Tensor):
        """Routes in ascending index order within each expert's range, padded with ``numel``.

        Blocks past the padded length hold ``num_experts``, one of the values that
        name no expert.  The GPU workload padded with ``torch.nn.functional.pad``;
        the filler is built on the device here so the reference needs no pad op.
        """
        ids, experts, count = ref_permute_align(topk_ids, self.block_size, self.num_experts)
        capacity = topk_ids.numel() + (self.num_experts + 1) * (self.block_size - 1)
        blocks = math.ceil(capacity / self.block_size)
        padded_ids = torch.full(
            (capacity - ids.numel(),), topk_ids.numel(), dtype=torch.int32, device=ids.device
        )
        padded_experts = torch.full(
            (blocks - experts.numel(),), self.num_experts, dtype=torch.int32, device=ids.device
        )
        return torch.cat((ids, padded_ids)), torch.cat((experts, padded_experts)), count


def gated_activation(gate_up: torch.Tensor, activation: str = "silu_and_mul") -> torch.Tensor:
    """The gated activation over the last axis, half as wide, in the input's dtype.

    GPU parity (``workloads/moe.py:gated_activation``), restricted to the
    ``silu_and_mul`` the shared expert computes: ``gate_up[..., :n]`` holds the
    gate, ``gate_up[..., n:]`` the up half, so ``silu(gate) * up``.  The
    ``swigluoai`` / ``gelu_and_mul`` branches belong to the routed-expert
    workloads, which are not migrated yet.
    """
    if activation != "silu_and_mul":
        raise ValueError(f"the shared expert computes silu_and_mul, got {activation!r}")
    gate, up = gate_up.chunk(2, dim=-1)
    return torch.nn.functional.silu(gate) * up


def ref_shared_expert(
    hidden: torch.Tensor, gate_up: torch.Tensor, down: torch.Tensor
) -> torch.Tensor:
    """Shared expert MLP in FP32, narrowed once to the token dtype.

    Both products run in float32 and the gated activation is rounded once, at
    the end -- the oracle the GPU workload uses (``ref_shared_expert``).
    """
    act = gated_activation(hidden.float() @ gate_up.float().T, "silu_and_mul")
    return (act @ down.float().T).to(hidden.dtype)


def _assert_normalized_error(got: torch.Tensor, expected: torch.Tensor, bound: float) -> None:
    """Bound squared error / combined energy, with identical nonfinite values.

    Ported from the GPU ``workloads/numerics.py:assert_normalized_error``: not
    an rtol or atol -- those compare element by element, this one ratios the
    error over the whole output, so a systematically small error cannot pass by
    landing under a per-element tolerance.  Reduced a chunk at a time so the
    float64 temporaries stay small at the prefill token counts.
    """
    assert got.shape == expected.shape and got.dtype == expected.dtype
    error = torch.zeros((), device=got.device, dtype=torch.float64)
    energy = torch.zeros_like(error)
    chunk = 1 << 20
    for a, b in zip(got.reshape(-1).split(chunk), expected.reshape(-1).split(chunk), strict=True):
        a, b = a.double(), b.double()
        finite = torch.isfinite(a)
        assert torch.equal(finite, torch.isfinite(b)), "nonfinite mask mismatch"
        torch.testing.assert_close(
            a.masked_fill(finite, 0), b.masked_fill(finite, 0), rtol=0, atol=0, equal_nan=True
        )
        a, b = a.masked_fill(~finite, 0), b.masked_fill(~finite, 0)
        error += ((a - b) ** 2).sum()
        energy += (a * a + b * b).sum()
    assert error <= bound * energy, f"normalized squared error: {error / energy}"


def moe_close(
    got: torch.Tensor,
    expected: torch.Tensor,
    tol: float = 1e-2,
    bound: float = 1e-4,
) -> None:
    """Elementwise ``atol = rtol = tol`` and normalized squared error at most *bound*.

    The GPU ``_moe_close``: the elementwise band catches per-entry faults, the
    normalized error catches a whole output that is uniformly off -- and, with
    the tolerance above the dtype's rounding, it is what lets an fp16/bf16
    kernel accumulate in the input dtype instead of float32.
    """
    torch.testing.assert_close(got, expected, rtol=tol, atol=tol)
    _assert_normalized_error(got, expected, bound)


class SharedExpertMLPWorkload(WorkloadBase):
    """Tokens and the stacked gate/up and down weights of one
    ``SharedExpertMLPFwdOp`` call.

    Args:
        num_tokens: Token count ``T`` (the first axis of ``hidden_states``).
        hidden_size: Model width ``H``.
        ffn_size: Expert width ``S`` (the gate/up half-width).
        dtype: Token dtype of all three tensors.
    """

    def __init__(
        self,
        num_tokens: int,
        hidden_size: int,
        ffn_size: int,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.num_tokens = num_tokens
        self.hidden_size = hidden_size
        self.ffn_size = ffn_size
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``randn`` tokens and weights, scaled as the GPU workload scales them.

        ``hidden`` is scaled by 0.1 and both weights by 0.02: small scales keep
        fp16 accumulation over ``H = 7168`` finite.
        """
        device = get_device_backend().name
        hidden = torch.randn(self.num_tokens, self.hidden_size, dtype=self.dtype, device=device)
        w_gate_up = torch.randn(
            2 * self.ffn_size, self.hidden_size, dtype=self.dtype, device=device
        )
        w_down = torch.randn(self.hidden_size, self.ffn_size, dtype=self.dtype, device=device)
        return hidden.mul_(0.1), w_gate_up.mul_(0.02), w_down.mul_(0.02)

    def ref_program(
        self,
        hidden_states: torch.Tensor,
        w_gate_up: torch.Tensor,
        w_down: torch.Tensor,
    ) -> torch.Tensor:
        """``down(silu(gate) * up)`` in fp32, cast to the hidden dtype."""
        return ref_shared_expert(hidden_states, w_gate_up, w_down)
