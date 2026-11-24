"""Correctness tests for SharedExpertMLPFwdOp (NPU).

The shared expert of an MoE layer: a dense gated MLP,
``down(silu(hidden @ gate.T) * (hidden @ up.T))``, with the gate and up rows
stacked in ``w_gate_up[2S, H]``.

Adapted from the GPU TileOPs ``tests/ops/test_fused_moe_shared_expert.py``,
which is shared with ``FusedMoESharedExpertFwdOp`` (the upper-level op that
calls this one).  Only this op's cases are ported:
``test_the_shared_expert_matches_its_reference``,
``test_the_shared_expert_runs_the_dense_template_past_its_threshold`` (as
``test_shared_expert_mlp_runs_its_shape_rows``, its SM90 template gate dropped)
and ``test_the_shared_expert_splits_k_at_32_tokens``.  The GPU cases that
assert on SM90 internals are not portable and are dropped:

- ``test_a_published_threshold_decomposes_the_next_call_again`` and
  ``test_a_published_tile_runs_as_the_basic_gemm_it_states`` read the GPU
  kernel's build cache (``kernel._generation.programs``) and compare against
  ``GEMMTMAKernel._MODAL_CONFIG`` / ``GEMMCall`` -- the NPU project has no
  generic GEMM library to assert on.
- the ``get_sm_version(...) != 90`` guards are SM90 gates; the NPU kernel has
  no template branch, so the shape rows run unconditionally here (the case is
  named ``test_shared_expert_mlp_runs_its_shape_rows`` for that reason).
- every ``FusedMoESharedExpertFwdOp`` case (basic / none / swigluoai refusal /
  tp / kernel replacement / grouped routing) belongs to the upper-level op,
  which is not migrated.

Adaptations (T1-T4):

- T1: manual tensor creation uses ``get_device_backend().name`` instead of the
  GPU's ``run_device()``.
- T2: the CUDA-availability / SM90 skip guards are removed.
- T3/T4: no ``tune`` parametrization and no autotune try/except (the NPU op
  has neither).

The GPU workload overrode ``verification()`` with the shared
``moe_verification(1)`` policy: elementwise ``atol = rtol = 1e-2`` plus a
normalized squared error bound of ``1e-4``.  ``TestBase`` has no verification
hook on NPU, so that policy is passed as the ``compare`` callable instead
(``tileops.workloads.moe.moe_close``).  Its negative control ("gate and up
rows swapped") is covered here by ``test_shared_expert_mlp_keeps_the_gate_up_row_order``.
"""

import pytest
import torch

from tileops.device import get_device_backend
from tileops.manifest import load_workloads
from tileops.ops.moe.shared_expert_mlp import SharedExpertMLPFwdOp
from tileops.testing import TestBase
from tileops.workloads.moe import SharedExpertMLPWorkload, moe_close

_SHARED_EXPERT_OP = "SharedExpertMLPFwdOp"


def _device() -> str:
    return get_device_backend().name


def _check(workload: SharedExpertMLPWorkload, op: SharedExpertMLPFwdOp, *inputs) -> None:
    """Run *op* against the fp32 reference and apply the MoE comparison policy."""
    TestBase.check(workload, op, *inputs, compare=moe_close)


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.bfloat16, marks=[pytest.mark.smoke, pytest.mark.packaging]),
        pytest.param(torch.float16, marks=pytest.mark.smoke),
    ],
)
def test_shared_expert_mlp_matches_its_reference(dtype: torch.dtype) -> None:
    """A small expert matches the fp32 reference in both token dtypes."""
    workload = SharedExpertMLPWorkload(32, 256, 128, dtype)
    _check(workload, SharedExpertMLPFwdOp(), *workload.gen_inputs())


@pytest.mark.parametrize(
    "tokens, hidden, ffn, dtype",
    [
        # A few waves of M tiles with the last row and column tiles ragged.
        pytest.param(600, 256, 7384, torch.float16, marks=pytest.mark.smoke, id="ragged"),
        pytest.param(1024, 256, 512, torch.bfloat16, marks=pytest.mark.smoke, id="one-wave"),
        pytest.param(4096, 256, 512, torch.bfloat16, marks=pytest.mark.full, id="separate"),
        # More than one wave of M tiles.
        pytest.param(4096, 256, 2048, torch.bfloat16, marks=pytest.mark.full, id="two-waves"),
        # An expert narrower than the model width.
        pytest.param(512, 512, 256, torch.float16, marks=pytest.mark.full, id="narrow-expert"),
        # An expert width that is not a multiple of the GEMM tile width.
        pytest.param(512, 1024, 1020, torch.bfloat16, marks=pytest.mark.full, id="narrow-tiles"),
    ],
)
def test_shared_expert_mlp_runs_its_shape_rows(
    tokens: int, hidden: int, ffn: int, dtype: torch.dtype
) -> None:
    """The GPU shape rows, over one and several M-tile waves and ragged tiles."""
    workload = SharedExpertMLPWorkload(tokens, hidden, ffn, dtype)
    _check(workload, SharedExpertMLPFwdOp(), *workload.gen_inputs())


@pytest.mark.parametrize(
    "hidden, ffn",
    [
        pytest.param(2048, 1408, marks=pytest.mark.smoke, id="wide-expert"),
        pytest.param(1024, 6144, marks=pytest.mark.full, id="narrow-model"),
    ],
)
def test_shared_expert_mlp_splits_k_at_32_tokens(hidden: int, ffn: int) -> None:
    """At 32 tokens a long contraction runs split over K (the decode band)."""
    workload = SharedExpertMLPWorkload(32, hidden, ffn, torch.bfloat16)
    _check(workload, SharedExpertMLPFwdOp(), *workload.gen_inputs())


def _manifest_rows() -> list:
    """The manifest workloads as ``(T, H, S, dtype)`` params, one per dtype case."""
    params = []
    for workload in load_workloads(_SHARED_EXPERT_OP):
        num_tokens, hidden_size = workload["hidden_states_shape"]
        ffn_size = workload["w_down_shape"][1]
        label = workload.get("label", f"t{num_tokens}-h{hidden_size}-s{ffn_size}")
        for dtype_name in workload["dtypes"]:
            params.append(
                pytest.param(
                    num_tokens,
                    hidden_size,
                    ffn_size,
                    getattr(torch, dtype_name),
                    marks=pytest.mark.full,
                    id=f"{label}-{dtype_name}",
                )
            )
    return params


@pytest.mark.parametrize("tokens, hidden, ffn, dtype", _manifest_rows())
def test_shared_expert_mlp_matches_the_manifest_workloads(
    tokens: int, hidden: int, ffn: int, dtype: torch.dtype
) -> None:
    """The declared workload rows (Kimi-K2 / DeepSeek-V3 / GLM-4.5) match the reference."""
    workload = SharedExpertMLPWorkload(tokens, hidden, ffn, dtype)
    _check(workload, SharedExpertMLPFwdOp(), *workload.gen_inputs())


@pytest.mark.smoke
def test_shared_expert_mlp_keeps_the_gate_up_row_order() -> None:
    """Swapping the gate and up halves changes the result (the GPU's negative control)."""
    dtype = torch.bfloat16
    workload = SharedExpertMLPWorkload(16, 256, 128, dtype)
    hidden, w_gate_up, w_down = workload.gen_inputs()
    op = SharedExpertMLPFwdOp()
    with torch.no_grad():
        out = op(hidden, w_gate_up, w_down)

    gate, up = w_gate_up.chunk(2, dim=0)
    swapped = torch.cat((up, gate), dim=0)
    with torch.no_grad():
        out_swapped = op(hidden, swapped, w_down)

    reference = workload.ref_program(hidden, w_gate_up, w_down)
    with pytest.raises(AssertionError):
        moe_close(out_swapped, reference)
    moe_close(out, reference)
    assert out.shape == (16, 256)
    assert out.dtype == dtype
    assert out.device == hidden.device


@pytest.mark.parametrize(
    "case, match",
    [
        pytest.param("float32", "float16 or bfloat16", marks=pytest.mark.smoke, id="fp32"),
        pytest.param("gate_up", r"w_gate_up must be", marks=pytest.mark.full, id="gate-up"),
        pytest.param("down", r"w_down must be", marks=pytest.mark.full, id="down"),
        pytest.param("contiguous", "contiguous", marks=pytest.mark.full, id="contiguous"),
        pytest.param("rank", "must be 2-D", marks=pytest.mark.full, id="rank"),
    ],
)
def test_shared_expert_mlp_rejects_an_invalid_tensor_contract(case: str, match: str) -> None:
    """A dtype, shape, layout or rank outside the signature is refused before the kernel."""
    device = _device()
    num_tokens, hidden_size, ffn_size = 8, 64, 32
    dtype = torch.bfloat16 if case != "float32" else torch.float32
    hidden = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    w_gate_up = torch.randn(2 * ffn_size, hidden_size, dtype=dtype, device=device)
    w_down = torch.randn(hidden_size, ffn_size, dtype=dtype, device=device)

    if case == "gate_up":
        w_gate_up = torch.randn(ffn_size, hidden_size, dtype=dtype, device=device)
    elif case == "down":
        # A wrong row count, so the fault is *only* ``w_down``'s: ``(H, 2S)``
        # would also be an illegal ``w_gate_up`` at ``H == 2S`` (and S is read
        # from ``w_down``'s column count), which would make this case assert on
        # whichever shape check the op happens to run first.
        w_down = torch.randn(2 * hidden_size, ffn_size, dtype=dtype, device=device)
    elif case == "contiguous":
        w_gate_up = torch.randn(hidden_size, 2 * ffn_size, dtype=dtype, device=device).t()
    elif case == "rank":
        hidden = torch.randn(num_tokens, dtype=dtype, device=device)

    with pytest.raises(ValueError, match=match):
        SharedExpertMLPFwdOp()(hidden, w_gate_up, w_down)


@pytest.mark.smoke
def test_shared_expert_mlp_dispatches_to_its_kernel() -> None:
    """The first call builds the shared-expert kernel for the call's geometry."""
    workload = SharedExpertMLPWorkload(16, 256, 128, torch.bfloat16)
    op = SharedExpertMLPFwdOp()
    op(*workload.gen_inputs())
    assert op.kernel.__class__.__name__ == "SharedExpertMLPKernel"

    # One kernel per call geometry: a second token count is a second build.
    other = SharedExpertMLPWorkload(32, 256, 128, torch.bfloat16)
    op(*other.gen_inputs())
    assert len(op._kernel_cache) == 2
    assert op.kernel.num_tokens == 32


@pytest.mark.smoke
def test_shared_expert_mlp_eval_roofline() -> None:
    """Two GEMM contractions plus the gated elementwise; the signature's traffic."""
    num_tokens, hidden_size, ffn_size = 16, 256, 128
    dtype = torch.bfloat16
    workload = SharedExpertMLPWorkload(num_tokens, hidden_size, ffn_size, dtype)
    op = SharedExpertMLPFwdOp()
    op(*workload.gen_inputs())

    flops, mem_bytes = op.eval_roofline()
    assert flops == (6 * num_tokens * ffn_size * hidden_size + 6 * num_tokens * ffn_size)
    assert mem_bytes == (2 * num_tokens * hidden_size + 3 * ffn_size * hidden_size) * dtype.itemsize


@pytest.mark.smoke
def test_shared_expert_mlp_roofline_requires_a_call() -> None:
    """The roofline needs a call to bind T / H / S."""
    with pytest.raises(RuntimeError, match="prior forward"):
        SharedExpertMLPFwdOp().eval_roofline()


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
