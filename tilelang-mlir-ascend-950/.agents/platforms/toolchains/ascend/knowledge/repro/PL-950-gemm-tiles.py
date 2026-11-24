"""PL-950-gemm-tiles. First verified: TileLang ascend 0.1.15 / CANN 9.2 SDK + 9.1 model."""

import json
import torch
import torch_npu  # noqa: F401
import tilelang
import tilelang.ascend.language as T


def build(m, n, k, dtype, trans_b, stages):
    bshape = (n, k) if trans_b else (k, n)
    grid_n = (n + 15) // 16

    @T.prim_func
    def main(
        A: T.Tensor((m, k), dtype),
        B: T.Tensor(bshape, dtype),
        C: T.Tensor((m, n), "float32"),
    ):
        with T.Kernel(((m + 15) // 16) * grid_n) as bid:
            mi, ni = bid // grid_n * 16, bid % grid_n * 16
            am, an = T.min(m - mi, 16), T.min(n - ni, 16)
            a1 = T.alloc_l1((16, 32), dtype)
            b1 = T.alloc_l1((16, 32), dtype)
            a0 = T.alloc_l0a((16, 32), dtype)
            b0 = T.alloc_l0b((16, 32), dtype)
            c0 = T.alloc_l0c((16, 16), "float32")
            T.annotate_buffer_versions({a1: stages, b1: stages, a0: stages, b0: stages})
            for step in T.Pipelined(T.ceildiv(k, 32), num_stages=stages):
                begin = step * 32
                ak = T.min(k - begin, 32)
                T.copy(A[mi : mi + am, begin : begin + ak], a1[:am, :ak])
                if trans_b:
                    T.copy(B[ni : ni + an, begin : begin + ak], b1[:an, :ak])
                else:
                    T.copy(
                        B[begin : begin + ak, ni : ni + an],
                        b1[:an, :ak],
                        transpose=True,
                    )
                T.copy(a1[:am, :ak], a0[:am, :ak])
                T.copy(b1[:an, :ak], b0[:an, :ak])
                T.gemm(
                    a0[:am, :ak],
                    b0[:an, :ak],
                    c0[:am, :an],
                    transpose_B=True,
                    clear_accum=step == 0,
                )
            T.copy(c0[:am, :an], C[mi : mi + am, ni : ni + an])

    return tilelang.compile(main, target="ascend", out_idx=-1)


def main():
    torch.manual_seed(2)
    for dtype in ("float16", "bfloat16"):
        for trans_b in (False, True):
            for m, n, k, stages in ((16, 16, 64, 1), (17, 19, 35, 2)):
                a = torch.randn(m, k).to(getattr(torch, dtype))
                logical_b = torch.randn(k, n).to(getattr(torch, dtype))
                b = logical_b.T.contiguous() if trans_b else logical_b
                expected = a.float() @ logical_b.float()
                result = build(m, n, k, dtype, trans_b, stages)(
                    a.to("npu"), b.to("npu")
                ).cpu()
                torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)
                print(
                    json.dumps(
                        {
                            "dtype": dtype,
                            "shape": [m, n, k],
                            "trans_b": trans_b,
                            "stages": stages,
                            "max_abs_error": (result - expected).abs().max().item(),
                            "passed": True,
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
