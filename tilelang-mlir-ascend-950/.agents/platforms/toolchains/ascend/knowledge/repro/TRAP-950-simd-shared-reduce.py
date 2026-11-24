"""TRAP-950-simd-shared-reduce. First verified: TileLang ascend 0.1.15 / CANN 9.2 SDK + 9.1 model."""

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
            sm = T.alloc_shared((2,), "float32")
            mx = T.alloc_shared((2,), "float32")
            T.copy(X, x[:, :n])
            with T.SimdVF():
                for i, j in T.Parallel(2, 64):
                    if j >= n:
                        x[i, j] = 0
                T.reduce_sum(x, sm, dim=1)
                for i, j in T.Parallel(2, 64):
                    if j >= n:
                        x[i, j] = -T.infinity("float32")
                T.reduce_max(x, mx, dim=1)
            T.copy(sm, S)
            T.copy(mx, M)

    return tilelang.compile(main, target="ascend", out_idx=[1, 2])


def main():
    fn = build(64)
    for label, x in [
        ("ramp", torch.linspace(-3, -1, 128).reshape(2, 64)),
        ("constant", torch.full((2, 64), -2.0)),
    ]:
        sm, _ = fn(x.to("npu"))
        actual, expected = sm.cpu(), x.sum(1)
        assert not torch.allclose(actual, expected, atol=1e-4, rtol=1e-5)
        print(
            json.dumps(
                {
                    "case": label,
                    "actual": actual.tolist(),
                    "expected": expected.tolist(),
                    "expected_mismatch": True,
                    "passed": True,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
