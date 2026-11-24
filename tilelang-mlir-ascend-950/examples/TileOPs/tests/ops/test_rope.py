"""Correctness tests for RoPEFwdOp (NPU).

Adapted from TileOPs ``tests/ops/test_rope.py`` -- the ``RoPEFwdOp`` subset
only (the shared GPU file also covers RoPELlama31FwdOp / YaRNFwdOp /
LongRoPEFwdOp / RoPECosSinFwdOp, which are not migrated).

The op rotates ``query`` and ``key`` at the cache rows ``positions`` names, as
vLLM's ``RotaryEmbedding.forward_native(positions, query, key)`` does; it is
checked against the independent reference of the vLLM formulas in
:mod:`tileops.workloads.rope`.

NPU adaptations (T1-T4):

- T1: manual tensor creation uses the device backend (``_device()``) instead of
  hard-coded ``device="cuda"``.
- T2: no ``@pytest.mark.skipif(not torch.cuda.is_available(), ...)`` guards in
  the GPU file -- nothing to remove.  The CUDA-graph replay case
  (``test_rope_graph_replay``) is dropped: it captures a ``torch.cuda.CUDAGraph``
  and asserts nothing about the rotation's semantics that the position tests do
  not.
- T3/T4: no ``tune`` parametrization, no autotune try-except (autotune is
  removed on NPU) -- the GPU ``test_rope_runs_after_a_tuning_request`` case is
  therefore dropped.
- The GPU "tune-first / tune-after" and vLLM cross-checks are dropped: vLLM is
  not available on the NPU toolchain, and the reference of
  :mod:`tileops.workloads.rope` is the independent ground truth here.
- The GPU tests that assert *which* program a call takes (the "staged rows"
  cases, chosen from CUDA shared-memory and resident-thread facts) keep their
  correctness content only; the NPU program choice is Stage 3's decision.
"""

import pytest
import torch

from tileops.device import get_device_backend
from tileops.ops.rope.rope import RoPEFwdOp
from tileops.testing.test_base import TestBase
from tileops.workloads.rope import RoPEFwdFixture, RoPEFwdWorkload


def _device() -> str:
    return get_device_backend().name


def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-5, 1e-5
    if dtype == torch.float16:
        return 1e-3, 1e-3
    return 1.6e-2, 1.6e-2


class RoPEFwdTest(RoPEFwdWorkload, TestBase):
    """The workload is the test: its ``ref_program`` is the vLLM-formula reference."""


def _workload(
    tokens: int,
    query_heads: int,
    key_heads: int | None,
    *,
    head_size: int = 64,
    rotary_dim: int = 64,
    is_neox_style: bool = True,
    dtype: torch.dtype = torch.bfloat16,
    max_position_embeddings: int = 4096,
    base: float = 10000.0,
    position_dtype: torch.dtype = torch.int64,
    batch: int | None = None,
) -> RoPEFwdTest:
    """A ``RoPEFwdOp`` workload of *tokens* tokens (optionally *batch* sequences),
    ``key_heads`` None for a query-only call."""
    positions_shape = (tokens,) if batch is None else (batch, tokens)
    query_shape = (*positions_shape, query_heads * head_size)
    key_shape = None if key_heads is None else (*positions_shape, key_heads * head_size)
    return RoPEFwdTest(
        positions_shape,
        query_shape,
        key_shape,
        head_size,
        rotary_dim,
        max_position_embeddings,
        base,
        is_neox_style,
        dtype,
        position_dtype,
    )


def _op(workload: RoPEFwdTest, **changes) -> RoPEFwdOp:
    """The op of *workload*'s parameters, with *changes* applied."""
    params = {**workload.params(), **changes}
    return RoPEFwdOp(**params)


@RoPEFwdFixture
def test_rope_fwd(
    positions_shape,
    query_shape,
    key_shape,
    head_size,
    rotary_dim,
    is_neox_style,
    dtype,
) -> None:
    """Manifest-shaped calls against the reference (smoke: fp32/fp16/bf16)."""
    test = RoPEFwdTest(
        positions_shape,
        query_shape,
        key_shape,
        head_size,
        rotary_dim,
        4096,
        10000.0,
        is_neox_style,
        dtype,
    )
    atol, rtol = _tolerances(dtype)
    test.check(_op(test), *test.gen_inputs(), atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_rope_matches_the_reference_at_positions_past_0(dtype: torch.dtype) -> None:
    """Decode at position p > 0: the rows the positions name, not rows 0..S-1."""
    test = _workload(37, 4, 2, dtype=dtype)
    _, query, key = test.gen_inputs()
    positions = torch.arange(1000, 1037, device=_device(), dtype=torch.int64)
    atol, rtol = _tolerances(dtype)
    test.check(_op(test), positions, query, key, atol=atol, rtol=rtol)


@pytest.mark.parametrize("neox", [True, False], ids=["neox", "gptj"])
@pytest.mark.parametrize(
    ("head_size", "rotary_dim"),
    [
        pytest.param(64, 64, marks=pytest.mark.smoke, id="full"),
        pytest.param(64, 32, marks=pytest.mark.full, id="partial"),
        pytest.param(65, 32, marks=pytest.mark.full, id="odd-head"),
    ],
)
def test_rope_rotates_the_leading_columns_in_either_convention(
    neox: bool, head_size: int, rotary_dim: int
) -> None:
    """An odd head size leaves no column offset on a vector boundary: scalar accesses."""
    dtype = torch.float16
    test = _workload(
        64, 8, 2, head_size=head_size, rotary_dim=rotary_dim, is_neox_style=neox, dtype=dtype
    )
    _, query, key = test.gen_inputs()
    positions = torch.arange(4000, 4064, device=_device(), dtype=torch.int64).flip(0)
    atol, rtol = _tolerances(dtype)
    test.check(_op(test), positions, query, key, atol=atol, rtol=rtol)


@pytest.mark.smoke
def test_rope_takes_batched_positions_and_a_query_alone() -> None:
    """Batched ``[*S]`` positions with no key: the key output is None."""
    dtype = torch.bfloat16
    test = _workload(16, 3, None, dtype=dtype, batch=2)
    positions = torch.randint(0, 4096, (2, 16), device=_device(), dtype=torch.int64)
    _, query, _ = test.gen_inputs()
    got_q, got_k = _op(test)(positions, query)
    ref_q, _ = test.ref_program(positions, query, None)
    assert got_k is None
    atol, rtol = _tolerances(dtype)
    torch.testing.assert_close(got_q, ref_q, atol=atol, rtol=rtol)


@pytest.mark.smoke
@pytest.mark.parametrize("neox", [True, False], ids=["neox", "gptj"])
@pytest.mark.parametrize(
    ("dtype", "bad"),
    [(torch.int32, 8), (torch.int64, 2**32 + 1), (torch.int64, -(2**32) + 1)],
    ids=["int32", "int64-past-int32", "int64-negative"],
)
def test_rope_validate_inputs_rejects_a_position_outside_the_cache(
    neox: bool, dtype: torch.dtype, bad: int
) -> None:
    """``validate_inputs=True`` checks the positions on the host before the kernel runs."""
    test = _workload(
        2,
        1,
        None,
        is_neox_style=neox,
        dtype=torch.float16,
        max_position_embeddings=8,
        position_dtype=dtype,
    )
    positions = torch.zeros(2, device=_device(), dtype=dtype)
    positions[-1] = bad
    query = torch.randn(2, 64, device=_device(), dtype=torch.float16)
    op = _op(test, validate_inputs=True)
    with pytest.raises(ValueError, match=r"^positions must be in \[0, 8\)"):
        op(positions, query)


@pytest.mark.smoke
@pytest.mark.parametrize("neox", [True, False], ids=["neox", "gptj"])
def test_rope_unchecked_position_outside_the_cache_reads_its_nearest_row(neox: bool) -> None:
    """Unchecked, a position outside the cache clamps to row 0 or the last row."""
    dtype = torch.float16
    test = _workload(3, 2, 1, is_neox_style=neox, dtype=dtype, max_position_embeddings=8)
    positions = torch.tensor([-1, 8, 3], device=_device())
    _, query, key = test.gen_inputs()
    got_q, got_k = _op(test)(positions, query, key)
    ref_q, ref_k = test.ref_program(positions.clamp(0, 7), query, key)
    atol, rtol = _tolerances(dtype)
    torch.testing.assert_close(got_q, ref_q, atol=atol, rtol=rtol)
    torch.testing.assert_close(got_k, ref_k, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    ("neox", "dtype", "head_size", "rotary_dim", "heads", "tokens"),
    [
        pytest.param(
            False, torch.float16, 64, 24, (32, 8), 16384, marks=pytest.mark.smoke, id="gptj"
        ),
        pytest.param(
            False, torch.float16, 65, 32, (3, 1), 16384, marks=pytest.mark.full, id="scalar-gptj"
        ),
        pytest.param(
            True, torch.float16, 64, 24, (1100, 8), 512, marks=pytest.mark.full, id="block-limit"
        ),
    ],
)
def test_rope_heads_copying_columns(
    neox: bool, dtype: torch.dtype, head_size: int, rotary_dim: int, heads, tokens: int
) -> None:
    """Heads with copied columns past ``rotary_dim``, in both conventions.

    A 65-column head moves single columns; a 1100-head query row is wider than any
    single tiling unit.
    """
    test = _workload(
        tokens,
        heads[0],
        heads[1],
        head_size=head_size,
        rotary_dim=rotary_dim,
        is_neox_style=neox,
        dtype=dtype,
    )
    positions, query, key = test.gen_inputs()
    positions = positions.flip(0) % 4096
    atol, rtol = _tolerances(dtype)
    test.check(_op(test), positions, query, key, atol=atol, rtol=rtol)


@pytest.mark.smoke
def test_rope_rotates_a_query_alone_with_copied_columns() -> None:
    """A NeoX query of 80-column heads rotating 32, alone, at int32 positions."""
    dtype = torch.bfloat16
    test = _workload(16384, 32, None, head_size=80, rotary_dim=32, dtype=dtype)
    positions = torch.randint(0, 4096, (16384,), device=_device(), dtype=torch.int32)
    _, query, _ = test.gen_inputs()
    got_q, got_k = _op(test)(positions, query)
    ref_q, _ = test.ref_program(positions, query, None)
    assert got_k is None
    atol, rtol = _tolerances(dtype)
    torch.testing.assert_close(got_q, ref_q, atol=atol, rtol=rtol)


@pytest.mark.smoke
def test_rope_heads_copying_columns_read_positions_outside_the_cache() -> None:
    """A small call whose heads copy columns; positions outside the cache clamp."""
    dtype = torch.float32
    test = _workload(5, 64, 1, head_size=64, rotary_dim=24, is_neox_style=False, dtype=dtype)
    rows = test.max_position_embeddings
    positions = torch.tensor([-1, rows, 3, rows - 1, 1000], device=_device())
    _, query, key = test.gen_inputs()
    got_q, got_k = _op(test)(positions, query, key)
    ref_q, ref_k = test.ref_program(positions.clamp(0, rows - 1), query, key)
    atol, rtol = _tolerances(dtype)
    torch.testing.assert_close(got_q, ref_q, atol=atol, rtol=rtol)
    torch.testing.assert_close(got_k, ref_k, atol=atol, rtol=rtol)


@pytest.mark.smoke
def test_rope_takes_a_key_of_no_heads() -> None:
    """A 0-column key comes back empty, shaped as the key, and the query rotates."""
    dtype = torch.float16
    test = _workload(5, 3, 0, dtype=dtype)
    positions, query, key = test.gen_inputs()
    assert key.shape[-1] == 0
    got_q, got_k = _op(test)(positions, query, key)
    assert got_k.shape == key.shape
    ref_q, _ = test.ref_program(positions, query, None)
    atol, rtol = _tolerances(dtype)
    torch.testing.assert_close(got_q, ref_q, atol=atol, rtol=rtol)


@pytest.mark.smoke
@pytest.mark.parametrize("neox", [True, False], ids=["neox", "gptj"])
def test_rope_query_off_a_vector_boundary(neox: bool) -> None:
    """A contiguous query starting one element past a 16-byte boundary matches the reference."""
    dtype = torch.float16
    test = _workload(8192, 8, None, is_neox_style=neox, dtype=dtype)
    positions = torch.randint(0, 4096, (8192,), device=_device(), dtype=torch.int64)
    _, aligned, _ = test.gen_inputs()
    query = torch.empty(aligned.numel() + 1, device=_device(), dtype=dtype)[1:]
    query = query.view(aligned.shape).copy_(aligned)
    got_q, _ = _op(test)(positions, query)
    ref_q, _ = test.ref_program(positions, aligned, None)
    atol, rtol = _tolerances(dtype)
    torch.testing.assert_close(got_q, ref_q, atol=atol, rtol=rtol)


@pytest.mark.smoke
def test_rope_kernel_map_resolves_every_variant() -> None:
    """Each of the four GPU ``source.kernel_map`` keys is hit by the dispatch.

    The GPU staging rule (ported into ``_kernel_key``) chooses between a
    convention's plain "positions" variant and its shared-memory "staged rows"
    variant: a call whose heads copy no column (``rotary_dim == head_size``)
    takes the plain one, a call with copied columns the staged one (the GPU wave
    bound is declared 0 on NPU, so no row count excludes staging).  Every key
    resolves to its declared class, so the first ``forward`` cannot raise
    ``KeyError``; both values of the ``wide`` hint and both ``_STAGED`` values
    compile the same program (the design's single block size).
    """
    from tileops.kernels.rope import (
        RoPENeoxPositionIdsKernel,
        RoPENeoxStagedRowsKernel,
        RoPENonNeoxKernel,
        RoPENonNeoxStagedRowsKernel,
    )

    expected = {
        "rope_neox_positions": RoPENeoxPositionIdsKernel,
        "rope_gptj": RoPENonNeoxKernel,
        "rope_neox_staged_rows": RoPENeoxStagedRowsKernel,
        "rope_gptj_staged_rows": RoPENonNeoxStagedRowsKernel,
    }
    cases = [
        # (is_neox_style, head_size, rotary_dim, key_heads, expected key)
        (True, 128, 128, 8, "rope_neox_positions"),  # heads copy no column
        (True, 80, 32, 0, "rope_neox_staged_rows"),  # copied columns, query only
        (False, 256, 256, 16, "rope_gptj"),  # heads copy no column
        (False, 256, 64, 16, "rope_gptj_staged_rows"),  # copied columns
    ]
    hit = set()
    for neox, head_size, rotary_dim, key_heads, expected_key in cases:
        op = RoPEFwdOp(
            head_size=head_size,
            rotary_dim=rotary_dim,
            max_position_embeddings=4096,
            base=10000.0,
            is_neox_style=neox,
            dtype=torch.bfloat16,
        )
        assert dict(op.default_kernel_map) == expected
        assert sorted(op.kernel_map) == sorted(expected)
        assert op._kernel_key(64, key_heads) == expected_key
        assert op.kernel_map[expected_key] is expected[expected_key]
        kernel = op._get_kernel(
            64,
            2,
            key_heads,
            torch.bfloat16,
            torch.int64,
            None,
        )
        assert type(kernel) is expected[expected_key]
        # The design's single block size reaches every variant unchanged.
        assert kernel.config == {"units_per_block": 2048}
        assert kernel._STAGED is (expected_key.endswith("_staged_rows"))
        hit.add(expected_key)
    assert hit == set(expected)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
