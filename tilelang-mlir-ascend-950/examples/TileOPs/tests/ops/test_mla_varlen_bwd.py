"""Correctness tests for MLAVarlenBwdOp (NPU).

Adapted from TileOPs ``tests/ops/test_mla_bwd.py`` -- the packed (varlen)
subset only.  The GPU file is shared with ``MLADenseBwdOp`` (already migrated,
see ``test_mla_dense_bwd.py``) and ``MLAVarlenFwdOp`` (not migrated yet), so
``MLADenseBwdFixture`` / ``test_mla_dense_bwd_op`` and
``test_mla_varlen_fwd_feeds_the_varlen_bwd`` (which needs ``MLAVarlenFwdOp`` and
``mla_varlen_inputs``) are intentionally not ported.

Adaptations (T1-T4):

- T2: the GPU module carries ``pytestmark = pytest.mark.sm90`` (both MLA
  backward kernels are SM90 only) and its oracle is
  ``flash_attn_interface._flash_attn_backward``.  Neither applies on NPU:
  the arch marker is dropped, and the oracle is the float32 torch backward
  reference ``MLAVarlenBwdWorkload.ref_program`` (device-agnostic, computed
  from ``q``/``k``/``v``/``do`` and the ``cu_seqlens`` offsets alone).
- The GPU ``verification()`` hook (``Exact`` with an atol floor of ``5e-3``,
  the dtype ``rtol``, and a ``first-input-zeroed`` control) has no NPU
  ``TestBase`` counterpart, so the same tolerance is passed to ``check()``
  here.  The zeroed-input control is not ported (the NPU ``TestBase`` has no
  controls hook).
- T1 is not needed: no input tensor is created by hand here.

The op supports float16/bfloat16 only (manifest ``T``), so the smoke cases are
bf16 + fp16 (one also ``packaging``) plus a tiny packed case, rather than the
usual fp32/fp16/bf16 trio.

``lse`` is head-major ``(H, T_q)`` for packed requests -- the layout FA3 returns
and ``MLAVarlenFwdOp`` produces on GPU -- which the last test pins down by
rejecting the token-major ``(T_q, H)`` transpose.
"""

from itertools import accumulate

import pytest
import torch

from tileops.ops.attention.mla_varlen_bwd import MLAVarlenBwdOp
from tileops.testing.test_base import TestBase
from tileops.workloads.attention import MLAVarlenBwdWorkload
from tileops.workloads.workload_base import FixtureBase

# GPU ``workloads/numerics.py:reference_tolerance`` for the two supported dtypes;
# ``verification()`` used ``atol = max(5e-3, tolerance)`` and ``rtol = tolerance``.
_RTOL = {torch.float16: 1e-3, torch.bfloat16: 1.6e-2}
_ATOL_FLOOR = 5e-3


def _tolerance(dtype: torch.dtype) -> tuple[float, float]:
    rtol = _RTOL[dtype]
    return max(_ATOL_FLOOR, rtol), rtol


class MLAVarlenBwdTest(MLAVarlenBwdWorkload, TestBase):
    """The workload is its own oracle: float32 gradients from q/k/v/do and the offsets."""


class MLAVarlenBwdFixture(FixtureBase):
    PARAMS = [
        (
            "q_lens, kv_lens, is_causal, sm_scale, dtype",
            [
                pytest.param(
                    [256, 384],
                    [256, 384],
                    True,
                    None,
                    torch.bfloat16,
                    marks=[pytest.mark.smoke],
                    id="smoke-varlen-bwd-bf16",
                ),
                pytest.param(
                    [256, 384],
                    [256, 384],
                    True,
                    None,
                    torch.float16,
                    marks=[pytest.mark.smoke, pytest.mark.packaging],
                    id="smoke-varlen-bwd-fp16",
                ),
                # A single short request: the packed row offsets are trivial but the
                # GroupTiling path is the same.
                pytest.param(
                    [64],
                    [64],
                    True,
                    None,
                    torch.bfloat16,
                    marks=[pytest.mark.smoke],
                    id="smoke-varlen-bwd-single-request",
                ),
                # Request lengths that no tile divides, one shorter than any tile.
                pytest.param(
                    [200, 17, 91],
                    [200, 17, 91],
                    True,
                    None,
                    torch.bfloat16,
                    marks=[pytest.mark.full],
                    id="full-varlen-bwd-ragged",
                ),
                # Fewer queries than keys in one request: its mask aligns to the bottom right.
                pytest.param(
                    [128, 256],
                    [320, 256],
                    True,
                    None,
                    torch.bfloat16,
                    marks=[pytest.mark.full],
                    id="full-varlen-bwd-q128-kv320",
                ),
                # More queries than keys: the first rows of that request see no key and get
                # zero gradients.
                pytest.param(
                    [320, 64],
                    [128, 64],
                    True,
                    None,
                    torch.bfloat16,
                    marks=[pytest.mark.full],
                    id="full-varlen-bwd-q320-kv128",
                ),
                pytest.param(
                    [256, 384],
                    [256, 384],
                    False,
                    None,
                    torch.bfloat16,
                    marks=[pytest.mark.full],
                    id="full-varlen-bwd-noncausal",
                ),
                # A caller's score scale in place of D_qk ** -0.5.
                pytest.param(
                    [256, 384],
                    [256, 384],
                    True,
                    0.05,
                    torch.bfloat16,
                    marks=[pytest.mark.full],
                    id="full-varlen-bwd-sm-scale",
                ),
            ],
        ),
    ]


@MLAVarlenBwdFixture
def test_mla_varlen_bwd_op(q_lens, kv_lens, is_causal, sm_scale, dtype) -> None:
    # DeepSeek-V3's decompressed head: D_qk 192, D_v 128.
    test = MLAVarlenBwdTest(q_lens, kv_lens, 4, 192, 128, dtype, is_causal, sm_scale)
    op = MLAVarlenBwdOp(is_causal=is_causal, sm_scale=sm_scale)
    atol, rtol = _tolerance(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@pytest.mark.smoke
def test_mla_varlen_bwd_general_kernel() -> None:
    """A head other than D_qk 192, D_v 128 runs the general kernel."""
    test = MLAVarlenBwdTest([256, 128], [256, 128], 4, 128, 64, torch.bfloat16)
    op = MLAVarlenBwdOp(is_causal=True)
    atol, rtol = _tolerance(torch.bfloat16)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@pytest.mark.smoke
def test_mla_varlen_bwd_validate_inputs_accepts_prefix_offsets() -> None:
    """``validate_inputs=True`` reads the offsets on the host and finds them well formed."""
    test = MLAVarlenBwdTest([64, 32], [64, 32], 4, 192, 128, torch.bfloat16)
    op = MLAVarlenBwdOp(is_causal=True, validate_inputs=True)
    atol, rtol = _tolerance(torch.bfloat16)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@pytest.mark.smoke
def test_mla_varlen_bwd_rejects_a_token_major_lse() -> None:
    """``lse`` is head-major ``(H, T_q)``, as FA3 returns it for packed requests, not
    token-major ``(T_q, H)``."""
    test = MLAVarlenBwdTest([64, 32], [64, 32], 4, 192, 128, torch.bfloat16)
    inputs = list(test.gen_inputs())
    inputs[5] = inputs[5].T.contiguous()
    with pytest.raises(ValueError, match="lse"):
        MLAVarlenBwdOp(is_causal=True)(*inputs)


@pytest.mark.smoke
def test_mla_varlen_bwd_eval_roofline() -> None:
    """The roofline sums each request's visible scores, an independent formulation.

    The visible key count of query ``i`` of a request with ``q`` queries and ``kv``
    keys is ``max(0, min(i + kv - q + 1, kv))`` -- query ``i`` sits at key position
    ``i + kv - q`` and sees the keys up to it -- which is the per-request form of
    the GPU ``visible_score_rows`` used by ``eval_roofline``.
    The geometry and offsets are installed as ``forward`` would leave them -- no
    device tensors are needed.
    """
    heads, dim_qk, dim_v = 4, 192, 128
    q_lens, kv_lens = [200, 17, 91], [320, 17, 40]
    for is_causal in (True, False):
        op = MLAVarlenBwdOp(is_causal=is_causal)
        op._geometry = (heads, dim_qk, dim_v)
        # ``eval_roofline`` reads packed *offsets*, an exclusive prefix sum.
        op._offsets = (
            torch.tensor([0, *accumulate(q_lens)], dtype=torch.int32),
            torch.tensor([0, *accumulate(kv_lens)], dtype=torch.int32),
        )
        op.dtype = torch.bfloat16
        flops, mem_bytes = op.eval_roofline()

        scores = rows = 0
        for q_len, kv_len in zip(q_lens, kv_lens, strict=False):
            if is_causal:
                # Query i sits at key position i + kv - q and sees keys 0..that position.
                visible = [
                    max(0, min(i + kv_len - q_len + 1, kv_len))
                    for i in range(q_len)
                ]
                scores += sum(visible)
                rows += sum(1 for n in visible if n > 0)
            else:
                scores += q_len * kv_len
                rows += q_len
        per_score = 2 * (3 * dim_qk + 2 * dim_v) + 5
        expected_flops = heads * (scores * per_score + rows * 2 * dim_v)
        elem = torch.bfloat16.itemsize
        total_q, total_kv = sum(q_lens), sum(kv_lens)
        expected_bytes = (
            2
            * (
                total_q * heads * dim_qk  # q / dq
                + total_q * heads * dim_v  # o / do
                + total_kv * heads * dim_qk  # k / dk
                + total_kv * heads * dim_v  # v / dv
            )
            * elem
            + heads * total_q * 4  # lse is float32
            + 2 * (len(q_lens) + 1) * 4  # both cu_seqlens are int32
        )
        assert flops == expected_flops, f"flops {flops} != {expected_flops}"
        assert mem_bytes == expected_bytes, f"bytes {mem_bytes} != {expected_bytes}"


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
