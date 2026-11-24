"""Correctness tests for MLADenseBwdOp (NPU).

Adapted from TileOPs ``tests/ops/test_mla_bwd.py`` -- the fixed-length
(``MLADenseBwdOp``) subset only.  The GPU file is shared with
``MLAVarlenBwdOp`` and ``MLAVarlenFwdOp`` (neither is migrated yet), so
``MLAVarlenBwdFixture`` / ``test_mla_varlen_bwd_op`` /
``test_mla_varlen_bwd_rejects_a_token_major_lse`` /
``test_mla_varlen_fwd_feeds_the_varlen_bwd`` and their imports
(``MLAVarlenFwdOp``, ``mla_varlen_inputs``) are intentionally not ported.

Adaptations (T1-T4):

- T2: the GPU module carries ``pytestmark = pytest.mark.sm90`` (both MLA
  backward kernels are SM90 only) and its oracle is
  ``flash_attn_interface._flash_attn_backward``.  Neither applies on NPU:
  the arch marker is dropped, and the oracle is the float32 torch backward
  reference ``MLADenseBwdWorkload.ref_program`` (device-agnostic, computed
  from ``q``/``k``/``v``/``do`` alone).
- The GPU ``verification()`` hook (``Exact`` with an atol floor of ``5e-3``,
  the dtype ``rtol``, and a ``first-input-zeroed`` control) has no NPU
  ``TestBase`` counterpart, so the same tolerance is passed to
  ``check()`` here.  The zeroed-input control is not ported (the NPU
  ``TestBase`` has no controls hook).
- T1 is not needed: no tensor is created by hand here.

The op supports float16/bfloat16 only (manifest ``T``), so the smoke trio is
bf16 + fp16 (one also ``packaging``) rather than the usual fp32/fp16/bf16.
"""

import pytest
import torch

from tileops.ops.attention.mla_dense_bwd import MLADenseBwdOp
from tileops.testing.test_base import TestBase
from tileops.workloads.attention import MLADenseBwdWorkload
from tileops.workloads.workload_base import FixtureBase

# GPU ``workloads/numerics.py:reference_tolerance`` for the two supported dtypes;
# ``verification()`` used ``atol = max(5e-3, tolerance)`` and ``rtol = tolerance``.
_RTOL = {torch.float16: 1e-3, torch.bfloat16: 1.6e-2}
_ATOL_FLOOR = 5e-3


def _tolerance(dtype: torch.dtype) -> tuple[float, float]:
    rtol = _RTOL[dtype]
    return max(_ATOL_FLOOR, rtol), rtol


class MLADenseBwdTest(MLADenseBwdWorkload, TestBase):
    """The workload is its own oracle: float32 gradients from q/k/v/do alone."""


class MLADenseBwdFixture(FixtureBase):
    PARAMS = [
        (
            "batch, seqlen_q, seqlen_kv, is_causal, sm_scale, dtype",
            [
                pytest.param(
                    2,
                    256,
                    256,
                    True,
                    None,
                    torch.bfloat16,
                    marks=[pytest.mark.smoke],
                    id="smoke-bwd-bf16",
                ),
                pytest.param(
                    2,
                    256,
                    256,
                    True,
                    None,
                    torch.float16,
                    marks=[pytest.mark.smoke, pytest.mark.packaging],
                    id="smoke-bwd-fp16",
                ),
                # A length that no power-of-two tile divides.
                pytest.param(
                    1,
                    200,
                    200,
                    True,
                    None,
                    torch.bfloat16,
                    marks=[pytest.mark.full],
                    id="full-bwd-s200",
                ),
                # Fewer queries than keys: the causal mask aligns to the bottom right.
                pytest.param(
                    1,
                    128,
                    320,
                    True,
                    None,
                    torch.bfloat16,
                    marks=[pytest.mark.full],
                    id="full-bwd-q128-kv320",
                ),
                # More queries than keys: the first rows see no key and get zero gradients.
                pytest.param(
                    1,
                    320,
                    128,
                    True,
                    None,
                    torch.bfloat16,
                    marks=[pytest.mark.full],
                    id="full-bwd-q320-kv128",
                ),
                pytest.param(
                    1,
                    256,
                    256,
                    False,
                    None,
                    torch.bfloat16,
                    marks=[pytest.mark.full],
                    id="full-bwd-noncausal",
                ),
                # A caller's score scale in place of D_qk ** -0.5.
                pytest.param(
                    1,
                    256,
                    256,
                    True,
                    0.05,
                    torch.bfloat16,
                    marks=[pytest.mark.full],
                    id="full-bwd-sm-scale",
                ),
            ],
        ),
    ]


@MLADenseBwdFixture
def test_mla_dense_bwd_op(batch, seqlen_q, seqlen_kv, is_causal, sm_scale, dtype) -> None:
    # DeepSeek-V3's decompressed head: D_qk 192, D_v 128.
    test = MLADenseBwdTest(
        batch, seqlen_q, seqlen_kv, 4, 192, 128, dtype, is_causal, sm_scale
    )
    op = MLADenseBwdOp(is_causal=is_causal, sm_scale=sm_scale)
    atol, rtol = _tolerance(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@pytest.mark.smoke
def test_mla_dense_bwd_general_kernel() -> None:
    """A head other than D_qk 192, D_v 128 runs the general kernel."""
    test = MLADenseBwdTest(1, 256, 256, 4, 128, 64, torch.bfloat16)
    op = MLADenseBwdOp(is_causal=True)
    atol, rtol = _tolerance(torch.bfloat16)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@pytest.mark.smoke
def test_mla_dense_bwd_rejects_a_token_major_lse() -> None:
    """``lse`` is ``(B, H, S_q)``, as FA3 returns it, not the token-major layout."""
    inputs = list(MLADenseBwdWorkload(1, 64, 64, 4, 192, 128, torch.bfloat16).gen_inputs())
    inputs[5] = inputs[5].transpose(1, 2).contiguous()
    with pytest.raises(ValueError, match="lse"):
        MLADenseBwdOp(is_causal=True)(*inputs)


@pytest.mark.smoke
def test_mla_dense_bwd_eval_roofline() -> None:
    """The roofline counts the visible scores of a bottom-right causal mask.

    The visible key count of query ``i`` is ``min(i + S_kv - S_q + 1, S_kv)``
    summed over the queries, which is an independent formulation of the
    GPU ``visible_score_rows`` used by ``eval_roofline``.  The geometry is
    installed as ``forward`` would leave it -- no device tensors are needed.
    """
    batch, heads, dim_qk, dim_v, seqlen_q, seqlen_kv = 2, 4, 192, 128, 320, 128
    for is_causal in (True, False):
        op = MLADenseBwdOp(is_causal=is_causal)
        op._geometry = (batch, heads, dim_qk, dim_v, seqlen_q, seqlen_kv)
        op.dtype = torch.bfloat16
        flops, mem_bytes = op.eval_roofline()

        visible = [
            max(0, min(i + seqlen_kv - seqlen_q + 1, seqlen_kv)) if is_causal else seqlen_kv
            for i in range(seqlen_q)
        ]
        scores, rows = sum(visible), sum(1 for n in visible if n > 0)
        per_score = 2 * (3 * dim_qk + 2 * dim_v) + 5
        expected_flops = batch * heads * (scores * per_score + rows * 2 * dim_v)
        elem = torch.bfloat16.itemsize
        expected_bytes = (
            2
            * (
                batch * seqlen_q * heads * dim_qk  # q / dq
                + batch * seqlen_q * heads * dim_v  # o / do
                + batch * seqlen_kv * heads * dim_qk  # k / dk
                + batch * seqlen_kv * heads * dim_v  # v / dv
            )
            * elem
            + batch * heads * seqlen_q * 4  # lse is float32
        )
        assert flops == expected_flops, f"flops {flops} != {expected_flops}"
        assert mem_bytes == expected_bytes, f"bytes {mem_bytes} != {expected_bytes}"


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
