"""Correctness tests for the Engram ops (NPU).

Adapted from the GPU TileOPs ``tests/ops/test_engram.py`` —
EngramGateConvFwdOp, EngramGateConvBwdOp and EngramDecodeFwdOp subsets.

Forward: all six outputs are compared, including the three intermediate
quantities the backward saves (``alpha``, ``rrms_h``, ``rrms_k``) and
``rrms_v``: the four are fp32 ``(M, seq_len)`` tensors, so ``TestBase.check``
walks them one by one.

Backward: the three input gradients are compared in the input dtype and the
three weight gradients in fp32; the weight gradients accumulate over every
token, so their last bits depend on the reduction order and are compared within
the manifest tolerance, each gradient on its own.

Decode: **both** outputs are compared — ``y_t`` and the complete updated
``new_conv_state``.  A wrong history shift is the most common decode bug and it
never shows up in ``y_t`` of the same step, so the state is checked slot by slot
(including the pure-shift prefix) and one test walks a growing cache over more
steps than the cache is long.

NPU adaptations (T1-T4):

- T1: manual tensor creation uses the device backend (``_device()``) instead of
  hard-coded ``device="cuda"``.
- T2: no ``torch.cuda`` availability guard in the GPU Engram tests — nothing to
  remove.  The GPU ``run_device()`` helper becomes ``get_device_backend().name``.
- T3: ``tune`` parametrization dimension removed.
- T4: no try/except for "No configurations to tune" in these tests.

Tolerances: the GPU workloads' ``verification()`` returned ``Exact`` with
``atol=0.1, rtol=0.1`` (fp16) / ``atol=0.2, rtol=0.1`` (bf16) for the forward,
``atol=0.2, rtol=0.2`` (fp16) / ``atol=0.3, rtol=0.2`` (bf16) for the backward,
and ``atol=0.05, rtol=0.05`` (fp16) / ``atol=0.1, rtol=0.05`` (bf16) for the
decode step; the same bounds are applied to every output of each op here.
"""

import pytest
import torch

from tileops.device import get_device_backend
from tileops.ops.sequence_modeling.engram_decode import EngramDecodeFwdOp
from tileops.ops.sequence_modeling.engram_gate_conv import EngramGateConvFwdOp
from tileops.ops.sequence_modeling.engram_gate_conv_bwd import EngramGateConvBwdOp
from tileops.testing.test_base import TestBase, allclose_compare, record_check_result
from tileops.workloads.sequence_modeling import (
    EngramDecodeFixture,
    EngramDecodeWorkload,
    EngramGateConvBwdFixture,
    EngramGateConvBwdWorkload,
    EngramGateConvFwdFixture,
    EngramGateConvFwdWorkload,
    engram_decode_step_torch,
    engram_gate_conv_fwd_torch,
)


def _device() -> str:
    return get_device_backend().name


def _tol(dtype: torch.dtype) -> dict:
    # GPU ``EngramGateConvFwdWorkload.verification``: fp16 -> 0.1, bf16 -> 0.2.
    return {"atol": 0.1, "rtol": 0.1} if dtype == torch.float16 else {"atol": 0.2, "rtol": 0.1}


def _tol_decode(dtype: torch.dtype) -> dict:
    # GPU ``EngramDecodeWorkload.verification``: fp16 -> 0.05, bf16 -> 0.1,
    # ``rtol=0.05`` for both.
    return {"atol": 0.05, "rtol": 0.05} if dtype == torch.float16 else {"atol": 0.1, "rtol": 0.05}


def _tol_bwd(dtype: torch.dtype) -> dict:
    # GPU ``EngramGateConvBwdWorkload.verification``: fp16 -> 0.2, bf16 -> 0.3,
    # ``rtol=0.2`` for both.
    return {"atol": 0.2, "rtol": 0.2} if dtype == torch.float16 else {"atol": 0.3, "rtol": 0.2}


class EngramGateConvFwdTest(EngramGateConvFwdWorkload, TestBase):
    pass


@EngramGateConvFwdFixture
def test_engram_gate_conv_fwd(M, seq_len, d, dtype):
    test = EngramGateConvFwdTest(M, seq_len, d, dtype)
    op = EngramGateConvFwdOp()
    inputs = test.gen_inputs()
    tol = _tol(dtype)
    test.check(op, *inputs, **tol)


@pytest.mark.smoke
def test_engram_gate_conv_fwd_outputs_are_fresh():
    """A cached kernel keeps the allocation contract: every call returns six new tensors."""
    M, seq_len, d, dtype = 1, 32, 256, torch.float16
    test = EngramGateConvFwdTest(M, seq_len, d, dtype)
    op = EngramGateConvFwdOp()
    inputs = test.gen_inputs()
    tol = _tol(dtype)

    first = op(*inputs)
    second = op(*inputs)
    assert len(first) == 6
    for a, b in zip(first, second, strict=True):
        assert a.data_ptr() != b.data_ptr(), "a cached launch must allocate fresh outputs"
        allclose_compare(a, b, **tol)
    # The first call's outputs must not have been written in place.
    allclose_compare(first[0], test.ref_program(*inputs)[0], **tol)


@pytest.mark.smoke
def test_engram_gate_conv_fwd_matches_each_output():
    """Each of the six outputs is compared against the torch reference on its own."""
    M, seq_len, d, dtype = 2, 16, 256, torch.bfloat16
    test = EngramGateConvFwdTest(M, seq_len, d, dtype)
    op = EngramGateConvFwdOp()
    inputs = test.gen_inputs()
    ref = test.ref_program(*inputs)
    out = op(*inputs)
    assert len(out) == len(ref) == 6
    names = ("Y", "vhat", "alpha", "rrms_h", "rrms_k", "rrms_v")
    for name, actual, expected in zip(names, out, ref, strict=True):
        assert actual.shape == expected.shape, name
        assert actual.dtype == expected.dtype, name
        allclose_compare(actual, expected, **_tol(dtype))


@pytest.mark.smoke
def test_engram_gate_conv_fwd_noncontiguous():
    """EngramGateConvFwdOp must handle non-contiguous inputs (forward takes .contiguous())."""
    M, seq_len, d, dtype = 1, 16, 256, torch.float16
    device = _device()
    H_full = torch.randn(M, seq_len * 2, d, dtype=dtype, device=device)
    k_full = torch.randn(M, seq_len * 2, d, dtype=dtype, device=device)
    v_full = torch.randn(M, seq_len * 2, d, dtype=dtype, device=device)
    H, k, v = H_full[:, ::2, :], k_full[:, ::2, :], v_full[:, ::2, :]
    assert not H.is_contiguous()
    rms_w_h = torch.ones(d, dtype=dtype, device=device)
    rms_w_v = torch.ones(d, dtype=dtype, device=device)
    conv_w = torch.randn(4, d, dtype=dtype, device=device) * 0.02

    ref = engram_gate_conv_fwd_torch(H, k, v, rms_w_h, rms_w_v, conv_w)
    out = EngramGateConvFwdOp()(H, k, v, rms_w_h, rms_w_v, conv_w)
    for actual, expected in zip(out, ref, strict=True):
        allclose_compare(actual, expected, **_tol(dtype))


@pytest.mark.smoke
def test_engram_gate_conv_fwd_rejects_unsupported_dtype():
    """The manifest declares float16 | bfloat16 only; fp32 must be refused."""
    M, seq_len, d = 1, 8, 256
    device = _device()
    dtype = torch.float32
    H = torch.randn(M, seq_len, d, dtype=dtype, device=device)
    k = torch.randn(M, seq_len, d, dtype=dtype, device=device)
    v = torch.randn(M, seq_len, d, dtype=dtype, device=device)
    rms_w_h = torch.ones(d, dtype=dtype, device=device)
    rms_w_v = torch.ones(d, dtype=dtype, device=device)
    conv_w = torch.randn(4, d, dtype=dtype, device=device)
    with pytest.raises(ValueError, match="float16 or bfloat16"):
        EngramGateConvFwdOp()(H, k, v, rms_w_h, rms_w_v, conv_w)


@pytest.mark.full
def test_engram_gate_conv_fwd_rejects_wide_d():
    """Rows wider than 4096 are refused rather than compiled (declared refusal)."""
    M, seq_len, d, dtype = 1, 4, 4352, torch.float16
    device = _device()
    H = torch.randn(M, seq_len, d, dtype=dtype, device=device)
    k = torch.randn(M, seq_len, d, dtype=dtype, device=device)
    v = torch.randn(M, seq_len, d, dtype=dtype, device=device)
    rms_w_h = torch.ones(d, dtype=dtype, device=device)
    rms_w_v = torch.ones(d, dtype=dtype, device=device)
    conv_w = torch.randn(4, d, dtype=dtype, device=device)
    with pytest.raises(ValueError, match="d must be <= 4096"):
        EngramGateConvFwdOp()(H, k, v, rms_w_h, rms_w_v, conv_w)


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_engram_gate_conv_fwd_roofline(dtype: torch.dtype):
    """``eval_roofline`` reproduces the manifest formula for the bound shapes."""
    M, seq_len, d = 1, 32, 256
    test = EngramGateConvFwdTest(M, seq_len, d, dtype)
    op = EngramGateConvFwdOp()
    inputs = test.gen_inputs()
    with torch.no_grad():
        out = op(*inputs)
    record_check_result(op, out, test.ref_program(*inputs))

    flops, nbytes = op.eval_roofline()
    expected_flops = M * (
        seq_len * (21 * d + 14)
        + 2 * d * (seq_len + max(seq_len - 1, 0) + max(seq_len - 2, 0) + max(seq_len - 3, 0))
    )
    elem = torch.tensor([], dtype=dtype).element_size()
    expected_bytes = (
        3 * M * seq_len * d * elem
        + 2 * d * elem
        + 4 * d * elem
        + 2 * M * seq_len * d * elem
        + 4 * M * seq_len * 4
    )
    assert flops == expected_flops
    assert nbytes == expected_bytes


# ---------------------------------------------------------------------------
# Backward (EngramGateConvBwdOp)
# ---------------------------------------------------------------------------


class EngramGateConvBwdTest(EngramGateConvBwdWorkload, TestBase):
    pass


@EngramGateConvBwdFixture
def test_engram_gate_conv_bwd(M, seq_len, d, dtype):
    test = EngramGateConvBwdTest(M, seq_len, d, dtype)
    op = EngramGateConvBwdOp()
    inputs = test.gen_inputs()
    test.check(op, *inputs, **_tol_bwd(dtype))
    run1 = [o.clone() for o in op(*inputs)]
    run2 = [o.clone() for o in op(*inputs)]
    for i, name in ((0, "dH"), (1, "dk"), (2, "dv")):
        max_err = (run1[i].float() - run2[i].float()).abs().max()
        assert torch.equal(run1[i], run2[i]), (
            f"{name} non-deterministic across runs (data race): max_err={max_err:.4e}"
        )


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(float("inf"), marks=pytest.mark.smoke),
        pytest.param(-float("inf"), marks=pytest.mark.full),
    ],
)
def test_engram_gate_conv_bwd_dconv_w_keeps_nonfinite_like_torch(value) -> None:
    # 9 tokens split into two dconv_w runs of 5: the last step of the second run lies
    # past the last token and must add nothing, not zero times an infinite gradient.
    test = EngramGateConvBwdTest(1, 9, 256, torch.float16)
    op = EngramGateConvBwdOp()
    inputs = list(test.gen_inputs())
    inputs[0][0, -1, :] = value
    actual = op(*inputs)[-1]
    expected = test.ref_program(*inputs)[-1]
    for kind in (torch.isposinf, torch.isneginf, torch.isnan):
        assert torch.equal(kind(actual), kind(expected)), kind.__name__


@pytest.mark.smoke
def test_engram_gate_conv_bwd_matches_each_output():
    """Each of the six outputs is compared against the torch reference on its own.

    The three weight gradients accumulate over every token in fp32, so they are
    asserted one by one: an error in one must not be hidden by the others.
    """
    M, seq_len, d, dtype = 2, 16, 256, torch.bfloat16
    test = EngramGateConvBwdTest(M, seq_len, d, dtype)
    op = EngramGateConvBwdOp()
    inputs = test.gen_inputs()
    ref = test.ref_program(*inputs)
    out = op(*inputs)
    assert len(out) == len(ref) == 6
    names = ("dH", "dk", "dv", "drms_w_h", "drms_w_v", "dconv_w")
    dtypes = (dtype, dtype, dtype, torch.float32, torch.float32, torch.float32)
    for name, expected_dtype, actual, expected in zip(
        names, dtypes, out, ref, strict=True
    ):
        assert actual.shape == expected.shape, name
        assert actual.dtype == expected_dtype, name
        assert expected.dtype == expected_dtype, name
        allclose_compare(actual, expected, **_tol_bwd(dtype))


@pytest.mark.smoke
def test_engram_gate_conv_bwd_weight_grads_are_fp32():
    """The three weight gradients are fp32 even though the inputs are fp16."""
    M, seq_len, d, dtype = 1, 8, 256, torch.float16
    test = EngramGateConvBwdTest(M, seq_len, d, dtype)
    op = EngramGateConvBwdOp()
    dH, dk, dv, drms_w_h, drms_w_v, dconv_w = op(*test.gen_inputs())
    assert dH.dtype == dk.dtype == dv.dtype == dtype
    assert drms_w_h.dtype == drms_w_v.dtype == dconv_w.dtype == torch.float32
    assert drms_w_h.shape == drms_w_v.shape == (d,)
    assert dconv_w.shape == (4, d)


@pytest.mark.smoke
def test_engram_gate_conv_bwd_noncontiguous():
    """EngramGateConvBwdOp must handle non-contiguous inputs (forward takes .contiguous())."""
    M, seq_len, d, dtype = 1, 16, 256, torch.float16
    device = _device()
    dY_full = torch.randn(M, seq_len * 2, d, dtype=dtype, device=device)
    H_full = torch.randn(M, seq_len * 2, d, dtype=dtype, device=device)
    k_full = torch.randn(M, seq_len * 2, d, dtype=dtype, device=device)
    v_full = torch.randn(M, seq_len * 2, d, dtype=dtype, device=device)
    dY, H, k, v = dY_full[:, ::2, :], H_full[:, ::2, :], k_full[:, ::2, :], v_full[:, ::2, :]
    assert not dY.is_contiguous()
    rms_w_h = torch.ones(d, dtype=dtype, device=device)
    rms_w_v = torch.ones(d, dtype=dtype, device=device)
    conv_w = torch.randn(4, d, dtype=dtype, device=device) * 0.02

    # Saved intermediates from the reference forward, so the call is well posed.
    ref_fwd = engram_gate_conv_fwd_torch(H, k, v, rms_w_h, rms_w_v, conv_w)
    vhat, alpha, rrms_h, rrms_k, rrms_v = ref_fwd[1:]

    inputs = (dY, H, k, v, rms_w_h, rms_w_v, conv_w, vhat, alpha, rrms_h, rrms_k, rrms_v)
    ref = EngramGateConvBwdTest(M, seq_len, d, dtype).ref_program(*inputs)
    out = EngramGateConvBwdOp()(*inputs)
    for actual, expected in zip(out, ref, strict=True):
        allclose_compare(actual, expected, **_tol_bwd(dtype))


@pytest.mark.smoke
def test_engram_gate_conv_bwd_rejects_unsupported_dtype():
    """The manifest declares float16 | bfloat16 only; fp32 must be refused."""
    M, seq_len, d = 1, 8, 256
    test = EngramGateConvBwdTest(M, seq_len, d, torch.float16)
    inputs = test.gen_inputs()
    dY_f32 = inputs[0].float()
    with pytest.raises(ValueError, match="float16 or bfloat16"):
        EngramGateConvBwdOp()(dY_f32, *inputs[1:])


@pytest.mark.smoke
def test_engram_gate_conv_bwd_rejects_non_fp32_saved_scalars():
    """``alpha`` and the three ``rrms`` tensors are the forward's fp32 outputs."""
    M, seq_len, d, dtype = 1, 8, 256, torch.float16
    test = EngramGateConvBwdTest(M, seq_len, d, dtype)
    inputs = list(test.gen_inputs())
    inputs[8] = inputs[8].to(dtype)  # alpha
    with pytest.raises(ValueError, match="float32"):
        EngramGateConvBwdOp()(*inputs)


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_engram_gate_conv_bwd_roofline(dtype: torch.dtype):
    """``eval_roofline`` reproduces the manifest formula for the bound shapes."""
    M, seq_len, d = 1, 32, 256
    test = EngramGateConvBwdTest(M, seq_len, d, dtype)
    op = EngramGateConvBwdOp()
    inputs = test.gen_inputs()
    out = op(*inputs)
    record_check_result(op, out, test.ref_program(*inputs))

    flops, nbytes = op.eval_roofline()
    expected_flops = M * (
        seq_len * (45 * d + 10)
        + 6 * d * (seq_len + max(seq_len - 1, 0) + max(seq_len - 2, 0) + max(seq_len - 3, 0))
    )
    elem = torch.tensor([], dtype=dtype).element_size()
    expected_bytes = (
        5 * M * seq_len * d * elem  # dY, H, k, v, vhat
        + 2 * d * elem  # rms_w_h, rms_w_v
        + 4 * d * elem  # conv_w
        + 4 * M * seq_len * 4  # alpha, rrms_h, rrms_k, rrms_v
        + 3 * M * seq_len * d * elem  # dH, dk, dv
        + 2 * d * 4  # drms_w_h, drms_w_v
        + 4 * d * 4  # dconv_w
    )
    assert flops == expected_flops
    assert nbytes == expected_bytes


# ---------------------------------------------------------------------------
# Decode (EngramDecodeFwdOp): one single-token step
# ---------------------------------------------------------------------------


class EngramDecodeTest(EngramDecodeWorkload, TestBase):
    pass


@EngramDecodeFixture
def test_engram_decode(batch, d_mem, d, max_conv_len, conv_kernel_size, dilation, dtype):
    test = EngramDecodeTest(batch, d_mem, d, max_conv_len, conv_kernel_size, dilation, dtype)
    op = EngramDecodeFwdOp(max_conv_len, conv_kernel_size, dilation)
    inputs = test.gen_inputs()
    # ``check`` walks both outputs: y_t and the full new_conv_state.
    test.check(op, *inputs, **_tol_decode(dtype))


@pytest.mark.smoke
def test_engram_decode_matches_each_output():
    """``y_t`` and ``new_conv_state`` are compared against the reference separately."""
    batch, d_mem, d, max_conv_len, conv_kernel_size, dilation = 2, 256, 256, 12, 4, 3
    dtype = torch.float16
    test = EngramDecodeTest(batch, d_mem, d, max_conv_len, conv_kernel_size, dilation, dtype)
    op = EngramDecodeFwdOp(max_conv_len, conv_kernel_size, dilation)
    inputs = test.gen_inputs()
    ref_y, ref_state = test.ref_program(*inputs)
    y_t, new_conv_state = op(*inputs)

    assert y_t.shape == ref_y.shape == (batch, d)
    assert y_t.dtype == ref_y.dtype == dtype
    assert new_conv_state.shape == ref_state.shape == (batch, max_conv_len, d)
    assert new_conv_state.dtype == ref_state.dtype == dtype
    allclose_compare(y_t, ref_y, **_tol_decode(dtype))
    allclose_compare(new_conv_state, ref_state, **_tol_decode(dtype))


@pytest.mark.smoke
def test_engram_decode_new_conv_state_history():
    """Every slot of ``new_conv_state`` is checked, not just the appended one.

    ``conv_state`` is filled with a value that is distinct per (batch row, slot), so a
    row shifted by the wrong offset, a row that is only half written, or an unshifted
    copy cannot alias the reference.  The first ``max_conv_len - 1`` slots are a pure
    dtype copy of the shifted history (no arithmetic), so they must match bit for bit;
    the last slot is this step's normalized value and is compared numerically.
    """
    batch, d_mem, d, max_conv_len, conv_kernel_size, dilation = 2, 256, 256, 12, 4, 3
    dtype = torch.float16
    device = _device()
    test = EngramDecodeTest(batch, d_mem, d, max_conv_len, conv_kernel_size, dilation, dtype)
    e_t, h_t, _, W_K, W_V, rms_w_h, rms_w_v, conv_w = test.gen_inputs()

    # Distinct magnitudes per (row, slot): 0.01 * (slot + 1) * (row + 1).
    slot_scale = torch.arange(1, max_conv_len + 1, dtype=torch.float32, device=device)
    row_scale = torch.arange(1, batch + 1, dtype=torch.float32, device=device)
    conv_state = (
        (0.01 * row_scale[:, None, None] * slot_scale[None, :, None])
        .expand(batch, max_conv_len, d)
        .to(dtype)
        .contiguous()
    )

    op = EngramDecodeFwdOp(max_conv_len, conv_kernel_size, dilation)
    y_t, new_conv_state = op(e_t, h_t, conv_state, W_K, W_V, rms_w_h, rms_w_v, conv_w)
    ref_y, ref_state = test.ref_program(e_t, h_t, conv_state, W_K, W_V, rms_w_h, rms_w_v, conv_w)

    assert new_conv_state.shape == (batch, max_conv_len, d)
    assert torch.equal(new_conv_state[:, :-1, :], conv_state[:, 1:, :]), (
        "new_conv_state history is not the shifted input cache"
    )
    allclose_compare(y_t, ref_y, **_tol_decode(dtype))
    allclose_compare(new_conv_state, ref_state, **_tol_decode(dtype))


@pytest.mark.smoke
def test_engram_decode_partial_cache():
    """A history shorter than ``max_conv_len`` is left-padded, and the new state is full."""
    batch, d_mem, d, max_conv_len, conv_kernel_size, dilation, conv_len = (
        2,
        512,
        256,
        12,
        4,
        3,
        5,
    )
    dtype = torch.float16
    test = EngramDecodeTest(
        batch,
        d_mem,
        d,
        max_conv_len,
        conv_kernel_size,
        dilation,
        dtype,
        conv_len=conv_len,
    )
    inputs = test.gen_inputs()
    assert inputs[2].shape == (batch, conv_len, d)
    op = EngramDecodeFwdOp(max_conv_len, conv_kernel_size, dilation)
    y_t, new_conv_state = op(*inputs)
    ref_y, ref_state = test.ref_program(*inputs)

    # One compiled kernel serves both cache lengths, and the output state is always full.
    assert new_conv_state.shape == ref_state.shape == (batch, max_conv_len, d)
    allclose_compare(y_t, ref_y, **_tol_decode(dtype))
    allclose_compare(new_conv_state, ref_state, **_tol_decode(dtype))


@pytest.mark.smoke
def test_engram_decode_multi_step():
    """Multi-step decode with a growing conv_state and a dilated conv.

    The cache starts empty, grows one step at a time, then rotates once it is full --
    every step checks ``y_t`` *and* the whole updated history.
    """
    B, d_mem, d = 2, 256, 256
    conv_kernel_size = 4
    dilation = 3
    max_conv_len = dilation * (conv_kernel_size - 1)  # = 9, minimum required
    dtype = torch.float16
    eps = 1e-6

    torch.manual_seed(123)
    W_K = torch.randn(d_mem, d, dtype=dtype, device=_device()) * 0.02
    W_V = torch.randn(d_mem, d, dtype=dtype, device=_device()) * 0.02
    rms_w_h = torch.ones(d, dtype=dtype, device=_device())
    rms_w_v = torch.ones(d, dtype=dtype, device=_device())
    conv_w = torch.randn(conv_kernel_size, d, dtype=dtype, device=_device()) * 0.02

    op = EngramDecodeFwdOp(max_conv_len, conv_kernel_size, dilation)

    conv_state = torch.zeros(B, 0, d, dtype=dtype, device=_device())
    conv_state_ref = conv_state.clone()

    num_steps = max_conv_len + 8  # past the growing phase
    for step in range(num_steps):
        e_t = torch.randn(B, d_mem, dtype=dtype, device=_device()) * 0.1
        h_t = torch.randn(B, d, dtype=dtype, device=_device())

        y_op, conv_state = op(e_t, h_t, conv_state, W_K, W_V, rms_w_h, rms_w_v, conv_w)
        y_ref, conv_state_ref = engram_decode_step_torch(
            e_t,
            h_t,
            conv_state_ref,
            W_K,
            W_V,
            rms_w_h,
            rms_w_v,
            conv_w,
            max_conv_len,
            dilation,
            eps,
        )

        y_err = (y_op.float() - y_ref.float()).abs().max().item()
        ref_len = conv_state_ref.shape[1]
        op_state_valid = conv_state[:, -ref_len:, :]
        s_err = (op_state_valid.float() - conv_state_ref.float()).abs().max().item()

        assert y_err < 0.1, f"Step {step}: y max_err={y_err:.6f}"
        assert s_err < 0.05, f"Step {step}: state max_err={s_err:.6f}"


@pytest.mark.smoke
def test_engram_decode_noncontiguous():
    """EngramDecodeFwdOp must handle non-contiguous inputs (forward takes .contiguous())."""
    batch, d_mem, d, max_conv_len, conv_kernel_size, dilation = 1, 256, 256, 12, 4, 3
    dtype = torch.float16
    device = _device()
    e_full = torch.randn(batch, d_mem * 2, dtype=dtype, device=device) * 0.1
    h_full = torch.randn(batch, d * 2, dtype=dtype, device=device)
    cs_full = torch.randn(batch, max_conv_len, d * 2, dtype=dtype, device=device) * 0.1
    e_t, h_t, conv_state = e_full[:, ::2], h_full[:, ::2], cs_full[:, :, ::2]
    assert not e_t.is_contiguous()
    assert not conv_state.is_contiguous()
    W_K = torch.randn(d_mem, d, dtype=dtype, device=device) * 0.02
    W_V = torch.randn(d_mem, d, dtype=dtype, device=device) * 0.02
    rms_w_h = torch.ones(d, dtype=dtype, device=device)
    rms_w_v = torch.ones(d, dtype=dtype, device=device)
    conv_w = torch.randn(conv_kernel_size, d, dtype=dtype, device=device) * 0.02

    test = EngramDecodeTest(batch, d_mem, d, max_conv_len, conv_kernel_size, dilation, dtype)
    ref = test.ref_program(e_t, h_t, conv_state, W_K, W_V, rms_w_h, rms_w_v, conv_w)
    out = EngramDecodeFwdOp(max_conv_len, conv_kernel_size, dilation)(
        e_t, h_t, conv_state, W_K, W_V, rms_w_h, rms_w_v, conv_w
    )
    for actual, expected in zip(out, ref, strict=True):
        allclose_compare(actual, expected, **_tol_decode(dtype))


@pytest.mark.smoke
def test_engram_decode_rejects_unsupported_dtype():
    """The manifest declares float16 | bfloat16 only; fp32 must be refused."""
    batch, d_mem, d, max_conv_len, conv_kernel_size, dilation = 1, 256, 256, 12, 4, 3
    test = EngramDecodeTest(
        batch, d_mem, d, max_conv_len, conv_kernel_size, dilation, torch.float16
    )
    inputs = test.gen_inputs()
    e_t_f32 = inputs[0].float()
    with pytest.raises(ValueError, match="float16 or bfloat16"):
        EngramDecodeFwdOp(max_conv_len, conv_kernel_size, dilation)(e_t_f32, *inputs[1:])


@pytest.mark.smoke
def test_engram_decode_rejects_too_small_cache():
    """``max_conv_len`` must reach the oldest conv tap (manifest shape rule)."""
    with pytest.raises(ValueError, match=r"dilation \* \(conv_kernel_size - 1\)"):
        EngramDecodeFwdOp(max_conv_len=8, conv_kernel_size=4, dilation=3)


@pytest.mark.full
def test_engram_decode_rejects_overlong_conv_state():
    """``L <= max_conv_len`` (manifest shape rule): a longer history is refused."""
    batch, d_mem, d, max_conv_len, conv_kernel_size, dilation = 1, 256, 256, 12, 4, 3
    dtype = torch.float16
    test = EngramDecodeTest(
        batch, d_mem, d, max_conv_len, conv_kernel_size, dilation, dtype, conv_len=max_conv_len + 1
    )
    inputs = test.gen_inputs()
    with pytest.raises(ValueError, match="must be <= max_conv_len"):
        EngramDecodeFwdOp(max_conv_len, conv_kernel_size, dilation)(*inputs)


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_engram_decode_roofline(dtype: torch.dtype):
    """``eval_roofline`` reproduces the manifest formula for the bound shapes."""
    batch, d_mem, d, max_conv_len, conv_kernel_size, dilation = 1, 512, 256, 12, 4, 3
    test = EngramDecodeTest(batch, d_mem, d, max_conv_len, conv_kernel_size, dilation, dtype)
    op = EngramDecodeFwdOp(max_conv_len, conv_kernel_size, dilation)
    inputs = test.gen_inputs()
    with torch.no_grad():
        out = op(*inputs)
    record_check_result(op, out, test.ref_program(*inputs))

    flops, nbytes = op.eval_roofline()
    expected_flops = (
        4 * batch * d_mem * d + batch * (16 * d + conv_kernel_size * 2 * d) + 20 * batch
    )
    elem = torch.tensor([], dtype=dtype).element_size()
    conv_len = max_conv_len
    expected_bytes = (
        batch * d_mem * elem  # e_t
        + batch * d * elem  # h_t
        + batch * conv_len * d * elem  # conv_state
        + 2 * d_mem * d * elem  # W_K, W_V
        + 2 * d * elem  # rms_w_h, rms_w_v
        + conv_kernel_size * d * elem  # conv_w
        + batch * d * elem  # y_t
        + batch * max_conv_len * d * elem  # new_conv_state
    )
    assert flops == expected_flops
    assert nbytes == expected_bytes


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
