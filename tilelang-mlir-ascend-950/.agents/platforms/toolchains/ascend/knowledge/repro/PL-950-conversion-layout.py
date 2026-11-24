"""PL-950-conversion-layout. First verified: TileLang ascend 0.1.15 / CANN 9.2 SDK + 9.1 model."""

import json
import torch
import torch_npu  # noqa: F401
import tilelang
import tilelang.ascend.language as T


def cast_kernel(dtype):
    @T.prim_func
    def main(X: T.Tensor((17,), "float32"), Y: T.Tensor((17,), dtype)):
        with T.Kernel(1):
            x = T.alloc_shared((32,), "float32")
            y = T.alloc_shared((32,), dtype)
            T.copy(X[:17], x[:17])
            with T.SimtVF(threads=32):
                for i in T.Parallel(17):
                    y[i] = T.cast(x[i], dtype)
            T.copy(y[:17], Y[:17])

    return tilelang.compile(main, target="ascend", out_idx=-1)


def transpose_kernel(dtype):
    @T.prim_func
    def main(
        X: T.Tensor((3, 17), dtype),
        B: T.Tensor((17,), dtype),
        Y: T.Tensor((17, 3), dtype),
    ):
        with T.Kernel(1):
            x = T.alloc_shared((3, 32), dtype)
            b = T.alloc_shared((32,), dtype)
            y = T.alloc_shared((17, 16), dtype)
            T.copy(X, x[:3, :17])
            T.copy(B, b[:17])
            with T.SimtVF(threads=64):
                for i, j in T.Parallel(3, 17):
                    y[j, i] = T.cast(x[i, j], "float32") + T.cast(b[j], "float32")
            T.copy(y[:17, :3], Y)

    return tilelang.compile(main, target="ascend", out_idx=-1)


def main():
    values = [
        0.0,
        -0.0,
        float("inf"),
        -float("inf"),
        float("nan"),
        1e-38,
        -1e-38,
        1e-8,
        -1e-8,
        65504.0,
        65520.0,
        -65520.0,
        1.00390625,
        1.01171875,
        2.0,
        -2.0,
        0.5,
    ]
    x = torch.tensor(values, dtype=torch.float32)
    for dtype in ("float32", "float16", "bfloat16"):
        dt = getattr(torch, dtype)
        result = cast_kernel(dtype)(x.to("npu")).cpu()
        expected = x.to(dt)
        torch.testing.assert_close(result, expected, atol=0, rtol=0, equal_nan=True)
        assert torch.equal(torch.signbit(result[:2]), torch.signbit(expected[:2]))
        print(
            json.dumps({"case": "cast_special", "dtype": dtype, "passed": True}),
            flush=True,
        )
        a = torch.arange(51).reshape(3, 17).to(dt) / 4
        b = torch.arange(17).to(dt) / 8
        expected = (a.float() + b.float()).to(dt).T.contiguous()
        result = transpose_kernel(dtype)(a.to("npu"), b.to("npu")).cpu()
        torch.testing.assert_close(result, expected, atol=0, rtol=0)
        print(
            json.dumps(
                {
                    "case": "broadcast_transpose",
                    "dtype": dtype,
                    "shape": [3, 17],
                    "passed": True,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
