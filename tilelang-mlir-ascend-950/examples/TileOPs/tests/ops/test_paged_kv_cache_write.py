"""Correctness tests for PagedKVCacheWriteFwdOp (NPU).

Adapted from TileOPs ``tests/ops/test_paged_kv_cache_write.py`` (the GPU
file covers only this op).  The op scatters k/v rows into two page pools
in place and returns ``None``; every assertion is therefore made on the
written pools, exactly (the GPU workload's ``verification()`` returned
``Exact(rtol=0, atol=0)``: a named row is written byte-exactly and a row
no slot names must still hold what it held).

Adaptations (T1-T4):

- T1: device placement for the manually built saturation input uses the
  tensor already produced by the workload (itself W1-adapted).
- T3: the ``tune`` param and the ``bf16-tuned`` case are removed (NPU
  uses heuristic config selection only).
- Not ported: the GPU ``torch.compile`` contract test
  (``test_paged_kv_cache_write_compiles_cold_to_its_operator``) and its
  ``isolated_dynamo`` / ``register_compile_contract`` wiring -- the NPU
  project has no compile-contract harness.
- The GPU smoke triple is fp32/fp16/bf16; this op's manifest dtype set is
  ``float16 | bfloat16`` only (an fp32 cache write is not a contract
  case), so the first params are the fp16, bf16 and both FP8-cache cases,
  and the first one also carries ``packaging``.

``TestBase.check`` is not used directly: it cannot consume a ``None``
return (``_to_tuple`` rejects it).  ``_check_exact`` runs the op, exposes
the two mutated pools through ``paged_cache_write_result`` (the GPU
helper), records the result for the pytest report, and compares both
pools with zero tolerance.
"""

import pytest
import torch

from tileops.ops.attention.paged_cache_write import PagedKVCacheWriteFwdOp
from tileops.testing.test_base import TestBase, record_check_result
from tileops.workloads.attention import (
    PagedKVCacheWriteWorkload,
    paged_cache_write_result,
)


class PagedKVCacheWriteTest(PagedKVCacheWriteWorkload, TestBase):
    """The workload as a test case: the reference and its policy come from the workload."""


def _host_if_fp8(tensor):
    """float8 has no device-side isclose on this stack (aclnnIsClose, error 161002).

    The comparison is still byte-exact; only the side it runs on changes.
    """
    return tensor.cpu() if tensor.dtype is torch.float8_e4m3fn else tensor


def _check_exact(test, op, *inputs):
    """Run *op* in place and compare both written pools byte-exactly."""
    outputs_ref = test.ref_program(*inputs)
    outputs = paged_cache_write_result(op, *inputs)
    record_check_result(op, outputs, outputs_ref)
    for output, output_ref in zip(outputs, outputs_ref, strict=True):
        torch.testing.assert_close(
            _host_if_fp8(output), _host_if_fp8(output_ref), atol=0, rtol=0, equal_nan=True
        )


@pytest.mark.parametrize(
    "tokens,heads_kv,dim,page_size,num_pages,dtype,cache_dtype,slots",
    [
        pytest.param(
            5,
            2,
            64,
            7,
            3,
            torch.float16,
            torch.float16,
            [20, -1, 0, 6, 13],
            marks=[pytest.mark.smoke, pytest.mark.packaging],
            id="fp16-odd-page-skip-and-page-edges",
        ),
        pytest.param(
            4,
            1,
            32,
            4,
            2,
            torch.bfloat16,
            torch.bfloat16,
            [7, 0, 3, 4],
            marks=pytest.mark.smoke,
            id="bf16-one-kv-head-last-row",
        ),
        pytest.param(
            3,
            2,
            64,
            16,
            2,
            torch.bfloat16,
            torch.float8_e4m3fn,
            [1, -1, 31],
            marks=pytest.mark.smoke,
            id="fp8-bf16-scaled",
        ),
        pytest.param(
            2,
            4,
            128,
            16,
            4,
            torch.float16,
            torch.float8_e4m3fn,
            [63, 0],
            marks=pytest.mark.smoke,
            id="fp8-fp16-page-boundary",
        ),
        pytest.param(
            64,
            8,
            128,
            64,
            8,
            torch.bfloat16,
            torch.bfloat16,
            None,
            marks=pytest.mark.full,
            id="bf16-decode-step-distinct-slots",
        ),
        pytest.param(
            4096,
            8,
            128,
            16,
            256,
            torch.float16,
            torch.float16,
            None,
            marks=pytest.mark.full,
            id="fp16-prefill-sweep",
        ),
    ],
)
def test_paged_kv_cache_write_scatters_the_named_rows(
    tokens, heads_kv, dim, page_size, num_pages, dtype, cache_dtype, slots
):
    """Every row a slot names holds the new k and v; the rest of both pools is untouched."""
    workload = PagedKVCacheWriteTest(
        tokens, heads_kv, dim, page_size, num_pages, dtype, cache_dtype, slots
    )
    inputs = workload.gen_inputs()
    op = PagedKVCacheWriteFwdOp()
    _check_exact(workload, op, *inputs)


@pytest.mark.smoke
def test_paged_kv_cache_write_saturates_fp8_overflow():
    """FP8 preserves NaN, saturates overflow/infinities, and rounds ties to even."""
    workload = PagedKVCacheWriteTest(1, 1, 16, 2, 1, torch.float16, torch.float8_e4m3fn, [1])
    inputs = workload.gen_inputs()
    k, v, _, _, _, k_scale, v_scale = inputs
    values = torch.tensor(
        [
            1000,
            -1000,
            448,
            -448,
            1.2,
            -1.2,
            1.0625,
            1.1875,
            float("nan"),
            float("inf"),
            -float("inf"),
            0,
            -0.0,
            1,
            -1,
            0.5,
        ],
        dtype=k.dtype,
        device=k.device,
    )
    k.copy_(values.reshape_as(k))
    v.copy_(-values.reshape_as(v))
    k_scale.fill_(1)
    v_scale.fill_(1)
    op = PagedKVCacheWriteFwdOp()
    _check_exact(workload, op, *inputs)


@pytest.mark.smoke
def test_paged_kv_cache_write_skips_every_token():
    """All-negative slots leave both sentinel-filled page pools unchanged."""
    workload = PagedKVCacheWriteTest(3, 2, 64, 7, 4, torch.bfloat16, slots=[-1, -1, -1])
    k, v, k_cache, v_cache, slots, k_scale, v_scale = workload.gen_inputs()
    k_cache.fill_(3.0)
    v_cache.fill_(-2.0)
    before = (k_cache.clone(), v_cache.clone())
    op = PagedKVCacheWriteFwdOp()
    assert op(k, v, k_cache, v_cache, slots, k_scale, v_scale) is None
    assert torch.equal(k_cache, before[0]) and torch.equal(v_cache, before[1])
    _check_exact(workload, op, k, v, before[0], before[1], slots, k_scale, v_scale)


@pytest.mark.smoke
def test_paged_kv_cache_write_of_no_tokens_writes_nothing():
    """A call with no token writes neither pool; the caches it is handed are non-empty."""
    workload = PagedKVCacheWriteTest(0, 2, 64, 16, 2, torch.bfloat16)
    k, v, k_cache, v_cache, slots, k_scale, v_scale = workload.gen_inputs()
    before = (k_cache.clone(), v_cache.clone())
    op = PagedKVCacheWriteFwdOp()
    assert op(k, v, k_cache, v_cache, slots, k_scale, v_scale) is None
    assert torch.equal(k_cache, before[0]) and torch.equal(v_cache, before[1])


@pytest.mark.parametrize(
    "invalid",
    [
        pytest.param("scale-missing", marks=pytest.mark.smoke),
        pytest.param("scale-present", marks=pytest.mark.smoke),
        pytest.param("input-dtype", marks=pytest.mark.full),
        pytest.param("cache-dtype", marks=pytest.mark.full),
    ],
)
def test_paged_kv_cache_write_refuses_an_invalid_tensor_contract(invalid):
    """Unsupported input/cache dtypes and incompatible scale presence are rejected."""
    cache_dtype = torch.float8_e4m3fn if invalid == "scale-missing" else torch.bfloat16
    workload = PagedKVCacheWriteTest(2, 2, 64, 16, 2, torch.bfloat16, cache_dtype)
    inputs = list(workload.gen_inputs())
    if invalid == "scale-missing":
        inputs[5] = inputs[6] = None
    elif invalid == "scale-present":
        inputs[5] = torch.ones(1, dtype=torch.float32, device=inputs[0].device)
        inputs[6] = inputs[5].clone()
    elif invalid == "input-dtype":
        inputs[0], inputs[1] = inputs[0].float(), inputs[1].float()
    else:
        inputs[2], inputs[3] = inputs[2].float(), inputs[3].float()
    with pytest.raises(ValueError, match="dtype" if invalid.endswith("dtype") else None):
        PagedKVCacheWriteFwdOp()(*inputs)


@pytest.mark.smoke
def test_paged_kv_cache_write_dispatches_to_its_kernel():
    """The first call builds the paged cache write kernel for the call's geometry."""
    workload = PagedKVCacheWriteTest(3, 2, 64, 7, 2, torch.float16, slots=[1, 0, 6])
    op = PagedKVCacheWriteFwdOp()
    op(*workload.gen_inputs())
    assert op.kernel.__class__.__name__ == "PagedKVCacheWriteKernel"


@pytest.mark.smoke
def test_paged_kv_cache_write_eval_roofline():
    """Only the tokens with a slot count; FP8 adds the scale/saturate FLOPs."""
    heads_kv, dim, page_size = 2, 64, 4
    row = heads_kv * dim
    tokens, num_pages = 4, 2
    slots = [7, 0, -1, 3]
    written = 3

    for cache_dtype in (torch.bfloat16, torch.float8_e4m3fn):
        workload = PagedKVCacheWriteTest(
            tokens, heads_kv, dim, page_size, num_pages, torch.bfloat16, cache_dtype, slots
        )
        op = PagedKVCacheWriteFwdOp()
        op(*workload.gen_inputs())
        flops, mem_bytes = op.eval_roofline()
        expected_flops = 4 * written * row if cache_dtype is torch.float8_e4m3fn else 0
        in_elem_bytes = torch.bfloat16.itemsize
        cache_elem_bytes = cache_dtype.itemsize
        expected_bytes = 2 * written * row * (in_elem_bytes + cache_elem_bytes)
        expected_bytes += tokens * 8  # slot_mapping (int64) is read once
        if cache_dtype is torch.float8_e4m3fn:
            expected_bytes += 8  # k_scale + v_scale
        assert flops == expected_flops, (
            f"flops {flops} != {expected_flops} (cache {cache_dtype})"
        )
        assert mem_bytes == expected_bytes, (
            f"bytes {mem_bytes} != {expected_bytes} (cache {cache_dtype})"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
