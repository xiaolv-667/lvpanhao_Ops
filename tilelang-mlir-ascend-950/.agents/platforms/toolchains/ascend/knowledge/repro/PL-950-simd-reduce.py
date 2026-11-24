"""PL-950-simd-reduce. First verified: TileLang ascend 0.1.15 / CANN 9.2 SDK + 9.1 model."""

import json
import torch
import torch_npu  # noqa: F401
import tilelang
import tilelang.ascend.language as T


def build(n):
    @T.prim_func
    def main(
        X: T.Tensor((2, n), "float32"),
        S: T.Tensor((2,), "float32"),
        M: T.Tensor((2,), "float32"),
    ):
        with T.Kernel(1):
            x = T.alloc_shared((2, 64), "float32")
            sm = T.alloc_shared((2, 8), "float32")
            mx = T.alloc_shared((2, 8), "float32")
            T.copy(X, x[:, :n])
            with T.SimdVF():
                one = T.simd.pset(32, "PAT_VL1")
                for row in T.serial(2):
                    mask = T.simd.update_mask(n, 32)
                    value = T.simd.vld(x[row, 0])
                    total = T.simd.vcadd(value, mask)
                    T.simd.vsts(sm[row, 0], total, one, extent=1)
                for row in T.serial(2):
                    mask = T.simd.update_mask(n, 32)
                    value = T.simd.vld(x[row, 0])
                    maximum = T.simd.vcmax(value, mask)
                    T.simd.vsts(mx[row, 0], maximum, one, extent=1)
            for row in T.serial(2):
                T.copy(sm[row, :1], S[row : row + 1])
                T.copy(mx[row, :1], M[row : row + 1])

    return tilelang.compile(main, target="ascend", out_idx=[1, 2])


def main():
    for n in (64, 63):
        x = torch.linspace(-3, -1, 2 * n).reshape(2, n)
        sm, mx = build(n)(x.to("npu"))
        torch.testing.assert_close(sm.cpu(), x.sum(1), atol=1e-4, rtol=1e-5)
        torch.testing.assert_close(mx.cpu(), x.max(1).values, atol=0, rtol=0)
        print(
            json.dumps(
                {
                    "n": n,
                    "sum_error": (sm.cpu() - x.sum(1)).abs().max().item(),
                    "passed": True,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
