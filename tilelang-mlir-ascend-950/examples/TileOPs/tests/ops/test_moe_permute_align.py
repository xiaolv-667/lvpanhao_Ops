"""Correctness tests for MoEPermuteAlignFwdOp (NPU).

Verifies that the op routes tokens to experts and pads each expert's slot count
to the GEMM ``block_size`` boundary.

Adapted from the GPU TileOPs ``tests/ops/test_moe_permute_align.py`` (the GPU
file covers only this op).  Reference: SGLang ``moe_align_block_size``.

Adaptations (T1-T4):

- T1: device placement for the manually built skewed input uses
  ``get_device_backend().name`` instead of ``run_device()``.
- T3: the ``tuned mode`` test is removed -- the NPU op has no ``tune`` /
  ``request_tune`` (heuristic config selection only).
- T4: no autotune try/except (there is no autotune).
- The GPU manifest-validation error (``CheckError`` on a nonpositive
  ``num_experts`` / ``block_size``) is a ``ValueError`` on NPU: the NPU project
  has no manifest codegen, so the op raises it directly.
- ``TestBase.check`` is not used directly: the GPU workload overrode
  ``verification()`` because the route order *within* an expert's range is
  unspecified, and the NPU ``TestBase`` has no verification hook.  The
  comparison below is that hook, ported: ``num_tokens_post_pad`` and the block
  owners are compared exactly, every slot past the padded length must read
  ``numel``, and ``sorted_token_ids`` is compared after sorting each expert's
  routed slots into module order.
"""

import math

import pytest
import torch

from tileops.device import get_device_backend
from tileops.ops.moe.permute_align import MoEPermuteAlignFwdOp
from tileops.testing.test_base import record_check_result
from tileops.workloads.moe import MoEPermuteAlignWorkload


def _routed_slots(
    topk_ids: torch.Tensor, num_experts: int, block_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Each route's slot when every expert's routes are in ascending index order.

    A route to expert ``e`` ranked ``r`` among the routes on ``e`` sits at
    ``starts[e] + r``, where ``starts`` is the exclusive prefix sum of the
    per-expert padded counts.  Returns ``(slot, starts[expert])``, the slot and
    the start of the range holding it, both in flattened routing order.
    """
    flat = topk_ids.flatten().long()
    counts = torch.bincount(flat, minlength=num_experts)
    padded = (counts + block_size - 1) // block_size * block_size
    starts = torch.cumsum(padded, 0) - padded
    rank = torch.arange(flat.numel()) - (torch.cumsum(counts, 0) - counts).repeat_interleave(counts)
    group_start = starts.repeat_interleave(counts)
    return group_start + rank, group_start


def _assert_route_equivalent(
    got, expected, topk_ids: torch.Tensor, num_experts: int, block_size: int
) -> None:
    """Exact up to route order within each expert and the value past the padded length.

    The comparison runs on the host: it is an exact, value-only check of index
    arrays that the tests keep small, so it spends no NPU op surface on sorts
    and gathers the kernel under test does not need.
    """
    tokens, experts, count = (t.cpu() for t in got)
    ref_tokens, ref_experts, ref_count = (t.cpu() for t in expected)
    assert torch.equal(count, ref_count), "num_tokens_post_pad differs"

    blocks = int(ref_count.item()) // block_size
    assert torch.equal(experts[:blocks], ref_experts[:blocks]), "a block's owner differs"
    tail = experts[blocks:]
    assert ((tail < 0) | (tail >= num_experts)).all(), "a block past the end names an expert"

    # Each expert's routed slots hold its routes in an unspecified order: sort
    # the whole array by (expert, route index) and compare with the reference,
    # which lists them in ascending index order.
    slots, group_start = _routed_slots(topk_ids.cpu(), num_experts, block_size)
    key = torch.arange(tokens.numel())
    key[slots] = group_start
    order = torch.sort(tokens, stable=True).indices
    order = order[torch.sort(key[order], stable=True).indices]
    assert torch.equal(tokens[order], ref_tokens), "sorted_token_ids differs"


def _check(test: MoEPermuteAlignWorkload, op: MoEPermuteAlignFwdOp, *inputs) -> None:
    """Run *op*, record the result for the pytest report, and compare as above."""
    outputs_ref = test.ref_program(*inputs)
    with torch.no_grad():
        outputs = op(*inputs)
    record_check_result(op, outputs, outputs_ref)
    _assert_route_equivalent(
        outputs, outputs_ref, inputs[0], test.num_experts, test.block_size
    )


@pytest.mark.parametrize(
    "total_tokens, top_k, num_experts, block_size",
    [
        pytest.param(
            4,
            2,
            4,
            4,
            marks=[pytest.mark.smoke, pytest.mark.packaging],
            id="tiny-bs4",
        ),
        pytest.param(16, 2, 8, 16, marks=pytest.mark.full, id="small-bs16"),
        pytest.param(1024, 8, 64, 128, marks=pytest.mark.full, id="large-bs128"),
        pytest.param(1, 2, 4, 4, marks=pytest.mark.full, id="single-token"),
        # top_k=1: each token is routed to exactly one expert
        pytest.param(8, 1, 4, 4, marks=pytest.mark.full, id="top-k-1"),
        # A few hundred routes over 8, 32 and 33 experts.
        pytest.param(100, 2, 8, 16, marks=pytest.mark.full, id="sb-numel200"),
        pytest.param(100, 6, 32, 64, marks=pytest.mark.full, id="sb-numel600-maxexp"),
        pytest.param(100, 6, 33, 64, marks=pytest.mark.full, id="experts-33"),
        # dispatch boundary: numel=1023 (last small-batch) vs numel=1024 (first large-batch)
        pytest.param(1023, 1, 8, 16, marks=pytest.mark.full, id="routes-1023"),
        pytest.param(512, 2, 8, 16, marks=pytest.mark.full, id="lb-boundary-1024"),
        # 1024 experts, then 1100.
        pytest.param(64, 8, 1024, 64, marks=pytest.mark.full, id="experts-1024"),
        pytest.param(64, 8, 1100, 64, marks=pytest.mark.full, id="experts-past-threads"),
        # No routes, over many experts and over few.
        pytest.param(0, 8, 128, 64, marks=pytest.mark.full, id="no-routes"),
        pytest.param(0, 8, 8, 16, marks=pytest.mark.full, id="no-routes-few-experts"),
        # One route over many experts.
        pytest.param(1, 1, 128, 64, marks=pytest.mark.full, id="one-route"),
        # Every route on the only expert.
        pytest.param(2048, 1, 1, 64, marks=pytest.mark.full, id="one-expert"),
        # A block height of one, then one that is not a power of two.
        pytest.param(64, 8, 128, 1, marks=pytest.mark.full, id="bs1"),
        pytest.param(64, 8, 128, 48, marks=pytest.mark.full, id="bs-not-pow2"),
        # 1032 routes, which no power of two above 8 divides.
        pytest.param(129, 8, 128, 64, marks=pytest.mark.full, id="routes-1032"),
        # 512 experts, then 513.
        pytest.param(64, 8, 512, 64, marks=pytest.mark.full, id="experts-512"),
        pytest.param(64, 8, 513, 64, marks=pytest.mark.full, id="experts-513"),
    ],
)
def test_permute_align_op(
    total_tokens: int, top_k: int, num_experts: int, block_size: int
) -> None:
    """Routed ids become expert-grouped, block-padded index arrays."""
    test = MoEPermuteAlignWorkload(total_tokens, top_k, num_experts, block_size)
    op = MoEPermuteAlignFwdOp(num_experts, block_size)
    _check(test, op, *test.gen_inputs())


@pytest.mark.parametrize(
    "total_tokens, top_k, num_experts, block_size",
    [
        pytest.param(32, 4, 8, 16, marks=pytest.mark.smoke, id="few-experts"),
        pytest.param(256, 8, 384, 64, marks=pytest.mark.full, id="many-experts"),
    ],
)
def test_permute_align_skewed_distribution(
    total_tokens: int, top_k: int, num_experts: int, block_size: int
) -> None:
    """Every route on expert 0, over few experts and over many."""
    topk_ids = torch.zeros(
        (total_tokens, top_k), dtype=torch.int32, device=get_device_backend().name
    )
    op = MoEPermuteAlignFwdOp(num_experts, block_size)
    test = MoEPermuteAlignWorkload(total_tokens, top_k, num_experts, block_size)
    _check(test, op, topk_ids)


@pytest.mark.parametrize(
    "num_experts, block_size, rule",
    [
        pytest.param(0, 64, "num_experts > 0", marks=pytest.mark.smoke, id="no-experts"),
        pytest.param(8, 0, "block_size > 0", marks=pytest.mark.full, id="empty-block"),
    ],
)
def test_permute_align_rejects_nonpositive_params(
    num_experts: int, block_size: int, rule: str
) -> None:
    """The op refuses an expert count or block height below one when it is built."""
    with pytest.raises(ValueError, match=rule):
        MoEPermuteAlignFwdOp(num_experts, block_size)


@pytest.mark.parametrize(
    "shape, dtype, match",
    [
        pytest.param((4, 2), torch.float32, "int32", marks=pytest.mark.smoke, id="float-input"),
        pytest.param((4, 2), torch.int64, "int32", marks=pytest.mark.full, id="int64-input"),
        pytest.param((8,), torch.int32, "T, K", marks=pytest.mark.full, id="one-dim-input"),
    ],
)
def test_permute_align_rejects_an_invalid_tensor_contract(
    shape: tuple[int, ...], dtype: torch.dtype, match: str
) -> None:
    """A non-int32 or non-2-D ``topk_ids`` is refused before the kernel is built."""
    topk_ids = torch.zeros(shape, dtype=dtype, device=get_device_backend().name)
    with pytest.raises(ValueError, match=match):
        MoEPermuteAlignFwdOp(num_experts=8, block_size=16)(topk_ids)


@pytest.mark.smoke
def test_permute_align_dispatches_to_its_kernel() -> None:
    """The first call builds the permute-align kernel for the call's routed count."""
    test = MoEPermuteAlignWorkload(16, 2, 8, 16)
    op = MoEPermuteAlignFwdOp(num_experts=8, block_size=16)
    op(*test.gen_inputs())
    assert op.kernel.__class__.__name__ == "MoEPermuteAlignKernel"


@pytest.mark.smoke
def test_permute_align_eval_roofline() -> None:
    """No arithmetic; the byte term is the index traffic the signature declares."""
    total_tokens, top_k, num_experts, block_size = 16, 2, 8, 16
    test = MoEPermuteAlignWorkload(total_tokens, top_k, num_experts, block_size)
    op = MoEPermuteAlignFwdOp(num_experts=num_experts, block_size=block_size)
    op(*test.gen_inputs())

    numel = total_tokens * top_k
    padded = numel + (num_experts + 1) * (block_size - 1)
    blocks = math.ceil(padded / block_size)
    flops, mem_bytes = op.eval_roofline()
    assert flops == 0
    assert mem_bytes == (numel + padded + blocks + 1) * 4


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
