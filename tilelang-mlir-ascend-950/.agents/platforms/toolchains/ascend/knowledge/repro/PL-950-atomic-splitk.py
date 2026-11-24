"""PL-950-atomic-splitk. First verified: TileLang ascend 0.1.15 / CANN 9.2 SDK + 9.1 model."""

import json
import torch
import torch_npu  # noqa: F401
import tilelang
import tilelang.ascend.language as T


def build(dtype):
    @T.prim_func
    def main(
        A: T.Tensor((16, 64), dtype),
        B: T.Tensor((64, 16), dtype),
        C: T.Tensor((16, 16), "float32"),
    ):
        with T.Kernel(2) as bid:
            a1 = T.alloc_l1((16, 32), dtype)
            b1 = T.alloc_l1((32, 16), dtype)
            a0 = T.alloc_l0a((16, 32), dtype)
            b0 = T.alloc_l0b((32, 16), dtype)
            c0 = T.alloc_l0c((16, 16), "float32")
            T.copy(A[:, bid * 32 : (bid + 1) * 32], a1)
            T.copy(B[bid * 32 : (bid + 1) * 32, :], b1)
            T.copy(a1, a0)
            T.copy(b1, b0)
            T.gemm(a0, b0, c0, clear_accum=True)
            T.set_atomic("add", "float32")
            T.copy(c0, C)
            T.set_atomic_none()

    return tilelang.compile(main, target="ascend")


def main():
    torch.manual_seed(6)
    for dtype in ("float16", "bfloat16"):
        dt = getattr(torch, dtype)
        a, b = torch.randn(16, 64).to(dt), torch.randn(64, 16).to(dt)
        result = torch.zeros(16, 16, device="npu", dtype=torch.float32)
        build(dtype)(a.to("npu"), b.to("npu"), result)
        result = result.cpu()
        expected = a.float() @ b.float()
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)
        print(
            json.dumps(
                {
                    "dtype": dtype,
                    "blocks": 2,
                    "max_abs_error": (result - expected).abs().max().item(),
                    "passed": True,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
