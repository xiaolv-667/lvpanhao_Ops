"""PL-950-mixed-epilogue. First verified: TileLang ascend 0.1.15 / CANN 9.2 SDK + 9.1 model."""

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
            cv = T.alloc_shared((8, 16), "float32")
            T.copy(A, a1)
            T.copy(B, b1)
            T.copy(a1, a0)
            T.copy(b1, b0)
            T.gemm(a0, b0, c0, clear_accum=True)
            T.dual_copy(c0, cv)
            with T.SimtVF(threads=64):
                for i, j in T.Parallel(8, 16):
                    cv[i, j] = T.max(cv[i, j], 0)
            T.dual_copy(cv, C)

    return tilelang.compile(main, target="ascend", out_idx=-1)


def main():
    torch.manual_seed(4)
    for dtype in ("float16", "bfloat16"):
        dt = getattr(torch, dtype)
        a, b = torch.randn(16, 32).to(dt), torch.randn(32, 16).to(dt)
        expected = (a.float() @ b.float()).relu()
        result = build(dtype)(a.to("npu"), b.to("npu")).cpu()
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)
        print(
            json.dumps(
                {
                    "dtype": dtype,
                    "max_abs_error": (result - expected).abs().max().item(),
                    "passed": True,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
