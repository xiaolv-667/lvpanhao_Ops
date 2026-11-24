"""Correctness tests for the mamba (Mamba-2 / SSD) family (NPU).

Adapted from TileOPs ``tests/ops/test_mamba.py`` -- SSDChunkScanFwdOp and
SSDChunkCouplingFwdOp subsets only (the shared GPU file also covers DaCumsum /
SSDChunkState / SSDStatePassing / SSDDecode / Mamba2Fwd, which are not
migrated).

NPU adaptations (T1-T4):

- T1: manual tensor creation uses the device backend instead of hard-coded
  ``device="cuda"``.
- T2: no ``torch.cuda`` availability guard in the GPU SSDChunkScan /
  SSDChunkCoupling tests -- nothing to remove.  The GPU
  ``run_device()`` / ``compare_outputs`` helpers become
  ``get_device_backend().name`` / ``allclose_compare``.
- T3: ``tune`` parametrization dimension removed (the GPU ``tune=True`` row
  duplicated the first smoke row once the dimension was dropped).
- T4: no try/except for "No configurations to tune" in these tests.
"""

import pytest
import torch

from tileops.device import get_device_backend
from tileops.ops.mamba.ssd_chunk_coupling import SSDChunkCouplingFwdOp
from tileops.ops.mamba.ssd_chunk_scan import SSDChunkScanFwdOp
from tileops.testing.mamba2_reference import (
    ssd_chunk_coupling_fwd_ref,
    ssd_chunk_scan_fwd_ref,
)
from tileops.testing.test_base import TestBase, allclose_compare, record_check_result
from tileops.workloads.mamba import (
    SSDChunkCouplingFwdFixture,
    SSDChunkCouplingFwdWorkload,
    SSDChunkScanFwdFixture,
    SSDChunkScanFwdWorkload,
)

# GPU ``coupling_verification()`` = ``Exact(atol=1e-3, rtol=1e-3)``.
COUPLING_ATOL = 1e-3
COUPLING_RTOL = 1e-3



class SSDChunkScanFwdTest(SSDChunkScanFwdWorkload, TestBase):
    def ref_program(self, x, cb, dA_cumsum, C, prev_states, dt):
        return ssd_chunk_scan_fwd_ref(x, cb, dA_cumsum, C, prev_states, dt, self.n_groups)


@SSDChunkScanFwdFixture
def test_ssd_chunk_scan_fwd(
    batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype
):
    test = SSDChunkScanFwdTest(
        batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype
    )
    op = SSDChunkScanFwdOp()
    inputs = test.gen_inputs()
    atol = 1e-3 if dtype == torch.float16 else 2e-3
    rtol = 1e-5
    test.check(op, *inputs, atol=atol, rtol=rtol)


class SSDChunkCouplingFwdTest(SSDChunkCouplingFwdWorkload, TestBase):
    def ref_program(self, C, B):
        return ssd_chunk_coupling_fwd_ref(C, B, self.num_chunks, self.chunk_len, self.dtype)


@SSDChunkCouplingFwdFixture
def test_ssd_chunk_coupling_fwd(batch, num_chunks, chunk_len, n_groups, d_state, dtype):
    test = SSDChunkCouplingFwdTest(batch, num_chunks, chunk_len, n_groups, d_state, dtype)
    op = SSDChunkCouplingFwdOp(chunk_len)
    C_mat, B_mat = test.gen_inputs()
    ref = test.ref_program(C_mat, B_mat)
    with torch.no_grad():
        out = op(C_mat, B_mat)
    record_check_result(op, out, ref)
    allclose_compare(out, ref, atol=COUPLING_ATOL, rtol=COUPLING_RTOL)
    # A cached executable must keep the same output allocation contract on
    # later calls: the second call returns a fresh tensor and leaves the first
    # one untouched.
    next_ref = ssd_chunk_coupling_fwd_ref(-C_mat, B_mat, num_chunks, chunk_len, dtype)
    allclose_compare(op(-C_mat, B_mat), next_ref, atol=COUPLING_ATOL, rtol=COUPLING_RTOL)
    allclose_compare(out, ref, atol=COUPLING_ATOL, rtol=COUPLING_RTOL)


@pytest.mark.smoke
def test_ssd_chunk_coupling_fwd_noncontiguous():
    """SSDChunkCouplingFwdOp must handle non-contiguous inputs."""
    batch, num_chunks, chunk_len, n_groups, d_state = 1, 2, 64, 1, 64
    dtype = torch.float16
    seq_len = num_chunks * chunk_len
    device = get_device_backend().name
    C_full = torch.randn(batch, seq_len * 2, n_groups, d_state, dtype=dtype, device=device)
    B_full = torch.randn(batch, seq_len * 2, n_groups, d_state, dtype=dtype, device=device)
    C_mat = C_full[:, ::2, :, :]
    B_mat = B_full[:, ::2, :, :]
    assert not C_mat.is_contiguous()
    assert not B_mat.is_contiguous()
    ref = ssd_chunk_coupling_fwd_ref(
        C_mat.contiguous(), B_mat.contiguous(), num_chunks, chunk_len, dtype
    )
    out = SSDChunkCouplingFwdOp(chunk_len)(C_mat, B_mat)
    allclose_compare(out, ref, atol=COUPLING_ATOL, rtol=COUPLING_RTOL)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
