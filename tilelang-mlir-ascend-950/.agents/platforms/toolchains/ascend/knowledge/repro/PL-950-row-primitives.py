"""PL-950-row-primitives. First verified: 2026-11-19; TileLang ascend 0.1.15, CANN 9.2 SDK / 9.1 simulator."""

import argparse
import json
import torch
import torch_npu  # noqa: F401
import tilelang
import tilelang.ascend.language as T


def build(n, dtype, mode):
    width = 64

    @T.prim_func
    def main(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            x = T.alloc_shared((width,), dtype)
            y = T.alloc_shared((width,), "float32")
            T.copy(X[:n], x[:n])
            with T.SimtVF(threads=64):
                xf = T.alloc_fragment((width,), "float32")
                total = T.alloc_reducer((1,), "float32", op="sum")
                maximum = T.alloc_reducer((1,), "float32", op="max")
                T.reducer_init(total)
                T.reducer_init(maximum)
                for i in T.Parallel(width):
                    xf[i] = T.if_then_else(i < n, T.cast(x[i], "float32"), 0)
                    T.reducer_update(
                        maximum[0], T.if_then_else(i < n, xf[i], -T.infinity("float32"))
                    )
                mx = T.alloc_fragment((1,), "float32")
                T.finalize_reducer(maximum, mx)
                for i in T.Parallel(width):
                    if mode == "softmax":
                        xf[i] = T.if_then_else(i < n, T.exp(xf[i] - mx[0]), 0)
                    if mode == "rmsnorm":
                        T.reducer_update(total[0], xf[i] * xf[i])
                    else:
                        T.reducer_update(total[0], xf[i])
                sm = T.alloc_fragment((1,), "float32")
                T.finalize_reducer(total, sm)
                for i in T.Parallel(width):
                    if i < n:
                        if mode == "softmax":
                            y[i] = xf[i] / sm[0]
                        elif mode == "rmsnorm":
                            y[i] = xf[i] / T.sqrt(sm[0] / n + 1e-5)
                        elif mode == "reduce":
                            y[i] = sm[0]
                        else:
                            y[i] = xf[i] + mx[0]
            T.copy(y[:n], Y[:n])

    return tilelang.compile(main, target="ascend", out_idx=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["all", "broadcast", "reduce", "softmax", "rmsnorm"],
        default="all",
    )
    args = parser.parse_args()
    modes = (
        ("broadcast", "reduce", "softmax", "rmsnorm")
        if args.mode == "all"
        else (args.mode,)
    )
    for mode in modes:
        for dtype in ("float32", "float16", "bfloat16"):
            for n in (64, 63):
                x = torch.linspace(-3, 2, n).to(getattr(torch, dtype))
                xf = x.float()
                if mode == "softmax":
                    expected = torch.softmax(xf, dim=0)
                elif mode == "rmsnorm":
                    expected = xf / torch.sqrt(xf.square().mean() + 1e-5)
                elif mode == "reduce":
                    expected = xf.sum().expand(n)
                else:
                    expected = xf + xf.max()
                result = build(n, dtype, mode)(x.to("npu")).cpu()
                torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)
                print(
                    json.dumps(
                        {
                            "mode": mode,
                            "dtype": dtype,
                            "n": n,
                            "max_abs_error": (result - expected).abs().max().item(),
                            "passed": True,
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
