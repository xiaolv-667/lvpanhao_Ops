# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""CONST-950-mmad-ktail-nonaligned: a single-tile Cube GEMM is exact for a K
First verified: TileLang ascend 0.1.15 + CANN 9.2.1 / Ascend950PR_9579 (dav-3510), 2026-11-23; origin_task batch2-20261123.
contraction that is not a multiple of the 16-lane Cube granularity.

Claim under test (device evidence, not documentation): ``asc_mmad`` takes the
*effective* K of the L0 operand regions and masks the lanes past it, so a GEMM
whose contraction length is 1, 7, 17 or a 256-block plus a 7 remainder needs no
host-side zero padding and no elementwise predicate.  The L1/L0 allocations stay
``block_k=256`` wide; only the copy regions and the ``T.gemm`` arguments carry the
remainder.

Why it matters: ``knowledge/gemm-tiles.md`` verified the (17,19,35) case, i.e. a
remainder inside a *multi-block* K loop (32+3).  It did not cover a contraction
shorter than one block (17, 1) nor a 256-blocked loop with a 7 remainder (263),
which is what an unaligned ``d_mem`` (519 -> 256+256+7) produces.

Method: one block, ``m=16``, ``n=64``, fp16 operands, fp32 output, reference is
the quantised-input fp32 matmul on CPU.  Threshold atol=rtol=1e-3.

run (from the task directory, environment loaded):
    /workspace/venv-ascend950/bin/python repro/CONST-950-mmad-ktail-nonaligned.py
"""

import functools

import torch
import torch_npu  # noqa: F401 -- registers torch.npu
import tilelang
import tilelang.ascend.language as T

M = 16
N = 64
BLOCK_K = 256
ATOL = 1e-3
RTOL = 1e-3


@functools.lru_cache(maxsize=None)
def build(k_total, dtype_str):
    """``C = A @ B`` for one tile; K is cut into ``BLOCK_K`` blocks, tail carried
    by the region extents."""
    k_tiles = -(-k_total // BLOCK_K)

    @T.prim_func
    def main(
        A: T.Tensor((M, k_total), dtype_str),
        B: T.Tensor((k_total, N), dtype_str),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(1) as bx:
            a_l1 = T.alloc_l1((M, BLOCK_K), dtype_str)
            b_l1 = T.alloc_l1((N, BLOCK_K), dtype_str)
            a_l0 = T.alloc_l0a((M, BLOCK_K), dtype_str)
            b_l0 = T.alloc_l0b((N, BLOCK_K), dtype_str)
            res = T.alloc_l0c((M, N), "float32")
            for kt in T.Pipelined(k_tiles, num_stages=1):
                k0 = kt * BLOCK_K
                ak = T.min(k_total - k0, BLOCK_K)
                T.copy(A[0:M, k0 : k0 + ak], a_l1[0:M, 0:ak])
                T.copy(B[k0 : k0 + ak, 0:N], b_l1[0:N, 0:ak], transpose=True)
                T.copy(a_l1[0:M, 0:ak], a_l0[0:M, 0:ak])
                T.copy(b_l1[0:N, 0:ak], b_l0[0:N, 0:ak])
                T.gemm(
                    a_l0[0:M, 0:ak],
                    b_l0[0:N, 0:ak],
                    res[0:M, 0:N],
                    transpose_B=True,
                    clear_accum=kt == 0,
                )
            T.copy(res[0:M, 0:N], C)

    return tilelang.compile(main, target="ascend", out_idx=[])


def check(k_total, dtype=torch.float16):
    torch.manual_seed(2)
    a = torch.randn(M, k_total, dtype=dtype)
    b = torch.randn(k_total, N, dtype=dtype)
    ref = a.float() @ b.float()
    c = torch.empty(M, N, dtype=torch.float32, device="npu")
    build(k_total, str(dtype).replace("torch.", ""))(a.to("npu"), b.to("npu"), c)
    torch.npu.synchronize()
    diff = (c.cpu() - ref).abs().max().item()
    torch.testing.assert_close(c.cpu(), ref, atol=ATOL, rtol=RTOL)
    print(f"K={k_total:4d} (k_blocks={-(-k_total // BLOCK_K)}, tail={k_total - (k_total // BLOCK_K) * BLOCK_K or BLOCK_K:3d}) "
          f"-> PASS max_diff={diff:.3e}")


if __name__ == "__main__":
    # shorter than one block, unaligned; 16-aligned control; 256 + 7 remainder
    for k in (1, 7, 16, 17, 263, 519):
        check(k)
    print("RESULT: all K tails exact -- asc_mmad masks the lanes past effective K")
