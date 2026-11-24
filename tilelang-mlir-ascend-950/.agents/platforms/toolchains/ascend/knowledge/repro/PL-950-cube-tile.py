"""PL-950-cube-tile: one dense tile with explicit L0 operands.
First verified: 2026-11-19; TileLang ascend 0.1.15, CANN 9.2 SDK / 9.1 simulator.
"""

import json
import torch
import torch_npu  # noqa: F401
import tilelang
import tilelang.ascend.language as T


def build(dtype):
    @T.prim_func
    def main(
        A: T.Tensor((16, 32), dtype),
        B: T.Tensor((32, 16), dtype),
        C: T.Tensor((16, 16), "float32"),
    ):
        with T.Kernel(1):
            a1 = T.alloc_l1((16, 32), dtype)
            b1 = T.alloc_l1((32, 16), dtype)
            a0 = T.alloc_l0a((16, 32), dtype)
            b0 = T.alloc_l0b((32, 16), dtype)
            c0 = T.alloc_l0c((16, 16), "float32")
            T.copy(A, a1)
            T.copy(B, b1)
            T.copy(a1, a0)
            T.copy(b1, b0)
            T.gemm(a0, b0, c0, clear_accum=True)
            T.copy(c0, C)

    return tilelang.compile(main, target="ascend", out_idx=-1)


def main():
    torch.manual_seed(0)
    for dtype in ("float16", "bfloat16"):
        a = torch.randn(16, 32).to(getattr(torch, dtype))
        b = torch.randn(32, 16).to(getattr(torch, dtype))
        expected = a.float() @ b.float()
        result = build(dtype)(a.to("npu"), b.to("npu")).cpu()
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)
        print(
            json.dumps(
                {
                    "dtype": dtype,
                    "shape": [16, 16, 32],
                    "max_abs_error": (result - expected).abs().max().item(),
                    "passed": True,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
