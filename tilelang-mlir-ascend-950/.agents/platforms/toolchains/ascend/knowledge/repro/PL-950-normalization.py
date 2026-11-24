"""PL-950-normalization. First verified: TileLang ascend 0.1.15 / CANN 9.2 SDK + 9.1 model."""

import json
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401
import tilelang
import tilelang.ascend.language as T


def build(n, dtype, mode):
    @T.prim_func
    def main(
        X: T.Tensor((2, n), dtype),
        G: T.Tensor((n,), dtype),
        B: T.Tensor((n,), dtype),
        Y: T.Tensor((2, n), dtype),
    ):
        with T.Kernel(2) as row:
            x = T.alloc_shared((64,), dtype)
            g = T.alloc_shared((64,), dtype)
            b = T.alloc_shared((64,), dtype)
            y = T.alloc_shared((64,), dtype)
            T.copy(X[row, :n], x[:n])
            T.copy(G, g[:n])
            T.copy(B, b[:n])
            with T.SimtVF(threads=64):
                xf = T.alloc_fragment((64,), "float32")
                total = T.alloc_reducer((1,), "float32", op="sum")
                square = T.alloc_reducer((1,), "float32", op="sum")
                T.reducer_init(total)
                T.reducer_init(square)
                for i in T.Parallel(64):
                    xf[i] = T.if_then_else(i < n, T.cast(x[i], "float32"), 0)
                    T.reducer_update(total[0], xf[i])
                sm = T.alloc_fragment((1,), "float32")
                T.finalize_reducer(total, sm)
                for i in T.Parallel(64):
                    if mode == "layernorm":
                        xf[i] = T.if_then_else(i < n, xf[i] - sm[0] / n, 0)
                    T.reducer_update(square[0], xf[i] * xf[i])
                sq = T.alloc_fragment((1,), "float32")
                T.finalize_reducer(square, sq)
                for i in T.Parallel(64):
                    if i < n:
                        if mode == "layernorm":
                            y[i] = xf[i] / T.sqrt(sq[0] / n + 1e-5) * T.cast(
                                g[i], "float32"
                            ) + T.cast(b[i], "float32")
                        else:
                            y[i] = (
                                xf[i]
                                / T.sqrt(sq[0] / n + 1e-5)
                                * T.cast(g[i], "float32")
                            )
            T.copy(y[:n], Y[row, :n])

    return tilelang.compile(main, target="ascend", out_idx=-1)


def main():
    torch.manual_seed(1)
    for mode in ("layernorm", "rmsnorm"):
        for dtype in ("float32", "float16", "bfloat16"):
            n = 63
            dt = getattr(torch, dtype)
            x, g, b = (
                torch.randn(2, n).to(dt),
                torch.randn(n).to(dt),
                torch.randn(n).to(dt),
            )
            x[1].fill_(2)
            if mode == "layernorm":
                expected = F.layer_norm(
                    x.float(), (n,), g.float(), b.float(), eps=1e-5
                ).to(dt)
            else:
                expected = F.rms_norm(x.float(), (n,), g.float(), eps=1e-5).to(dt)
            result = build(n, dtype, mode)(x.to("npu"), g.to("npu"), b.to("npu")).cpu()
            atol, rtol = (
                (1e-5, 1e-5)
                if dtype == "float32"
                else ((2e-3, 2e-3) if dtype == "float16" else (2e-2, 2e-2))
            )
            torch.testing.assert_close(result, expected, atol=atol, rtol=rtol)
            print(json.dumps(dict(mode=mode, dtype=dtype, passed=True)), flush=True)


if __name__ == "__main__":
    main()
