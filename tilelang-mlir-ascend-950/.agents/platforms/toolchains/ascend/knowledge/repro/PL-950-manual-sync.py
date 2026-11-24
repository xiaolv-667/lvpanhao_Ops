"""PL-950-manual-sync. First verified: TileLang ascend 0.1.15 / CANN 9.2 SDK + 9.1 model."""

import json
import torch
import torch_npu  # noqa: F401
import tilelang
import tilelang.ascend.language as T


def build():
    @T.prim_func
    def main(
        A: T.Tensor((16, 32), "float16"),
        B: T.Tensor((32, 16), "float16"),
        C: T.Tensor((16, 16), "float32"),
    ):
        with T.Kernel(1):
            with T.Cube():
                a1 = T.alloc_l1((16, 32), "float16")
                b1 = T.alloc_l1((32, 16), "float16")
                a0 = T.alloc_l0a((16, 32), "float16")
                b0 = T.alloc_l0b((32, 16), "float16")
                c0 = T.alloc_l0c((16, 16), "float32")
                T.copy(A, a1)
                T.copy(B, b1)
                T.ascend_set_flag("MTE2_MTE1", 0)
                T.ascend_wait_flag("MTE2_MTE1", 0)
                T.copy(a1, a0)
                T.copy(b1, b0)
                T.ascend_set_flag("MTE1_M", 0)
                T.ascend_wait_flag("MTE1_M", 0)
                T.gemm(a0, b0, c0, clear_accum=True)
                T.ascend_set_flag("M_FIX", 0)
                T.ascend_wait_flag("M_FIX", 0)
                T.copy(c0, C)
                T.ascend_pipe_barrier("PIPE_ALL")

    return tilelang.compile(
        main,
        target="ascend",
        out_idx=-1,
        pass_configs={"tl.enable_auto_schedule": False},
    )


def main():
    torch.manual_seed(5)
    a, b = torch.randn(16, 32).half(), torch.randn(32, 16).half()
    result = build()(a.to("npu"), b.to("npu")).cpu()
    expected = a.float() @ b.float()
    torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)
    print(
        json.dumps(
            {
                "case": "manual_cube_flags",
                "max_abs_error": (result - expected).abs().max().item(),
                "passed": True,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
