"""PL-950-attention. First verified: TileLang ascend 0.1.15 / CANN 9.2 SDK + 9.1 model."""

import json
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401
import tilelang
import tilelang.ascend.language as T


def build(dtype, causal):
    @T.prim_func
    def main(
        Q: T.Tensor((3, 16), dtype),
        K: T.Tensor((5, 16), dtype),
        V: T.Tensor((5, 16), dtype),
        O: T.Tensor((3, 16), "float32"),
    ):
        with T.Kernel(3) as row:
            q = T.alloc_shared((16,), dtype)
            k = T.alloc_shared((5, 16), dtype)
            v = T.alloc_shared((5, 16), dtype)
            o = T.alloc_shared((16,), "float32")
            T.copy(Q[row, :], q)
            T.copy(K, k)
            T.copy(V, v)
            with T.SimtVF(threads=32):
                score = T.alloc_fragment((8,), "float32")
                mx = T.alloc_reducer((1,), "float32", op="max")
                sm = T.alloc_reducer((1,), "float32", op="sum")
                T.reducer_init(mx)
                T.reducer_init(sm)
                T.clear(score)
                for d in T.serial(16):
                    for j in T.Parallel(8):
                        if j < 5:
                            score[j] = score[j] + T.cast(q[d], "float32") * T.cast(
                                k[j, d], "float32"
                            )
                for j in T.Parallel(8):
                    score[j] = T.if_then_else(
                        j < 5 and (not causal or j <= row),
                        score[j] * 0.25,
                        -T.infinity("float32"),
                    )
                    T.reducer_update(mx[0], score[j])
                maximum = T.alloc_fragment((1,), "float32")
                T.finalize_reducer(mx, maximum)
                for j in T.Parallel(8):
                    score[j] = T.exp(score[j] - maximum[0])
                    T.reducer_update(sm[0], score[j])
                total = T.alloc_fragment((1,), "float32")
                T.finalize_reducer(sm, total)
                for j in T.Parallel(8):
                    score[j] = score[j] / total[0]
                out = T.alloc_fragment((16,), "float32")
                T.clear(out)
                for j in T.serial(5):
                    for d in T.Parallel(16):
                        out[d] = out[d] + score[j] * T.cast(v[j, d], "float32")
                for d in T.Parallel(16):
                    o[d] = out[d]
            T.copy(o, O[row, :])

    return tilelang.compile(main, target="ascend", out_idx=-1)


def main():
    torch.manual_seed(3)
    for dtype in ("float32", "float16", "bfloat16"):
        for causal in (False, True):
            dt = getattr(torch, dtype)
            q, k, v = [torch.randn(n, 16).to(dt) for n in (3, 5, 5)]
            expected = F.scaled_dot_product_attention(
                q.float()[None, None],
                k.float()[None, None],
                v.float()[None, None],
                is_causal=causal,
            )[0, 0]
            result = build(dtype, causal)(q.to("npu"), k.to("npu"), v.to("npu")).cpu()
            torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)
            print(
                json.dumps(
                    {
                        "dtype": dtype,
                        "causal": causal,
                        "shape": [3, 5, 16],
                        "max_abs_error": (result - expected).abs().max().item(),
                        "passed": True,
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
