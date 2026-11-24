"""PL-950-softmax-boundaries. First verified: TileLang ascend 0.1.15 / CANN 9.2 SDK + 9.1 model."""

import json
import torch
import torch_npu  # noqa: F401
import tilelang
import tilelang.ascend.language as T


def build():
    @T.prim_func
    def main(X: T.Tensor((64,), "float32"), Y: T.Tensor((64,), "float32")):
        with T.Kernel(1):
            x = T.alloc_shared((64,), "float32")
            y = T.alloc_shared((64,), "float32")
            T.copy(X, x)
            with T.SimtVF(threads=64):
                xf = T.alloc_fragment((64,), "float32")
                mx = T.alloc_reducer((1,), "float32", op="max")
                sm = T.alloc_reducer((1,), "float32", op="sum")
                T.reducer_init(mx)
                T.reducer_init(sm)
                for i in T.Parallel(64):
                    xf[i] = x[i]
                    T.reducer_update(mx[0], xf[i])
                maximum = T.alloc_fragment((1,), "float32")
                T.finalize_reducer(mx, maximum)
                for i in T.Parallel(64):
                    xf[i] = T.exp(xf[i] - maximum[0])
                    T.reducer_update(sm[0], xf[i])
                total = T.alloc_fragment((1,), "float32")
                T.finalize_reducer(sm, total)
                for i in T.Parallel(64):
                    y[i] = xf[i] / total[0]
            T.copy(y, Y)

    return tilelang.compile(main, target="ascend", out_idx=-1)


def main():
    cases = {
        "large_finite": torch.linspace(-1000, 1000, 64),
        "constant": torch.full((64,), -10000.0),
        "negative_inf": torch.full((64,), -float("inf")),
    }
    for name, value in [("positive_inf", float("inf")), ("nan", float("nan"))]:
        x = torch.zeros(64)
        x[0] = value
        cases[name] = x
    fn = build()
    for name, x in cases.items():
        result = fn(x.to("npu")).cpu()
        expected = torch.softmax(x, dim=0)
        torch.testing.assert_close(
            result, expected, atol=1e-6, rtol=1e-5, equal_nan=True
        )
        print(
            json.dumps(
                {"case": name, "nan_count": int(result.isnan().sum()), "passed": True}
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
