"""Stage 1 设计期探针 2（revision v2）：L0 路径的 `transpose_A=True` / `transpose_B=False`。

背景：`ascend/op/gemm/gemm_mad.py` 的 layout 推断支持四种转置组合（A: trans_A=False→K-major,
trans_A=True→MN-major；B: trans_B=True→K-major, trans_B=False→MN-major），但 `lower` 的
L1 路径 assert 只允许 NT；已核验 repro（PL-950-gemm-tiles.py）只覆盖 NT + 显式 L0 路径。

本探针回答两件事（编译 + 运行期数值，小 shape）：
  Q1  `dQ = A(dS^T,(K=Nk,M=Mq)) . B(k^T,(N=Dqk,K=Nk))`，`transpose_A=True, transpose_B=True`
      —— 若可用，则 **dS 不需要在向量侧转置**（省掉实测 3.5 µs/迭代的跨步 scatter 写）。
  Q2  `dK = A(dS^T,(Nk,K=Mq)) . B(q,(K=Mq,N=Dqk))`，`transpose_A=False, transpose_B=False`
      —— 若可用，则 **不需要 qT 的转置装载**（MTE2 每 query 行从 1280 B 降到 640 B）。

运行（仓根之外）：

    cd /tmp && source /workspace/tilelang-mlir-ascend-950/.agents/platforms/toolchains/ascend/env-950pr.sh \
      && TMPDIR=/workspace /workspace/venv-ascend950/bin/python \
         /workspace/tilelang-mlir-ascend-950/examples/mla_dense_bwd/_bwd_kernel/history_version/design_probe_l0_gemm.py
"""

import time

import tilelang
import tilelang.ascend.language as T
import torch
import torch_npu  # noqa: F401

DT = "bfloat16"
NK, MQ, DQK = 64, 64, 192   # (key 块, query 块, D_qk)


def build_tn():
    """Q1: C(Mq,Dqk) = A(Nk,Mq)^T . B(Dqk,Nk)^T  -> dQ（trans_A=True, trans_B=True）."""

    @T.prim_func
    def main(
        a: T.Tensor((NK, MQ), DT),        # type: ignore  # dS^T
        b: T.Tensor((DQK, NK), DT),       # type: ignore  # k^T
        c: T.Tensor((MQ, DQK), "float32"),  # type: ignore
    ):
        with T.Kernel(1) as _:
            a1 = T.alloc_l1((NK, MQ), DT)
            b1 = T.alloc_l1((DQK, NK), DT)
            a0 = T.alloc_l0a((NK, MQ), DT)
            b0 = T.alloc_l0b((DQK, NK), DT)
            c0 = T.alloc_l0c((MQ, DQK), "float32")
            T.copy(a, a1)
            T.copy(b, b1)
            T.copy(a1, a0)
            T.copy(b1, b0)
            T.gemm(a0, b0, c0, transpose_A=True, transpose_B=True, clear_accum=True)
            T.copy(c0, c)

    return main


def build_nn():
    """Q2: C(Nk,Dqk) = A(Nk,Mq) . B(Mq,Dqk)  -> dK（trans_A=False, trans_B=False）."""

    @T.prim_func
    def main(
        a: T.Tensor((NK, MQ), DT),        # type: ignore  # dS^T
        b: T.Tensor((MQ, DQK), DT),       # type: ignore  # q（ND，未转置）
        c: T.Tensor((NK, DQK), "float32"),  # type: ignore
    ):
        with T.Kernel(1) as _:
            a1 = T.alloc_l1((NK, MQ), DT)
            b1 = T.alloc_l1((MQ, DQK), DT)
            a0 = T.alloc_l0a((NK, MQ), DT)
            b0 = T.alloc_l0b((MQ, DQK), DT)
            c0 = T.alloc_l0c((NK, DQK), "float32")
            T.copy(a, a1)
            T.copy(b, b1)
            T.copy(a1, a0)
            T.copy(b1, b0)
            T.gemm(a0, b0, c0, transpose_A=False, transpose_B=False, clear_accum=True)
            T.copy(c0, c)

    return main


def build_nt_l0():
    """对照：已核验形态 NT（trans_B=True），确认 L0 路径本身可用."""

    @T.prim_func
    def main(
        a: T.Tensor((NK, MQ), DT),        # type: ignore
        b: T.Tensor((DQK, NK), DT),       # type: ignore
        c: T.Tensor((NK, DQK), "float32"),  # type: ignore
    ):
        with T.Kernel(1) as _:
            a1 = T.alloc_l1((NK, MQ), DT)
            b1 = T.alloc_l1((DQK, NK), DT)
            a0 = T.alloc_l0a((NK, MQ), DT)
            b0 = T.alloc_l0b((DQK, NK), DT)
            c0 = T.alloc_l0c((NK, DQK), "float32")
            T.copy(a, a1)
            T.copy(b, b1)
            T.copy(a1, a0)
            T.copy(b1, b0)
            T.gemm(a0, b0, c0, transpose_A=False, transpose_B=True, clear_accum=True)
            T.copy(c0, c)

    return main


def run(name, builder, ref):
    try:
        fn = tilelang.compile(builder(), target="ascend")
    except Exception as exc:  # noqa: BLE001
        return f"COMPILE_FAIL {type(exc).__name__}: {str(exc)[:200]}"
    try:
        want = ref[1]
        c = torch.zeros(want.shape, dtype=torch.float32, device="npu")
        fn(*[t.to("npu") for t in ref[0]], c)
        torch.npu.synchronize()
        got = c.cpu().float()
    except Exception as exc:  # noqa: BLE001
        return f"RUN_FAIL {type(exc).__name__}: {str(exc)[:200]}"
    err = (got - want).abs().max().item()
    scale = want.abs().max().item() + 1e-9
    rel = err / scale
    verdict = "PASS" if rel < 5e-2 else "FAIL"   # bf16 输入 + fp32 累加
    return f"{verdict}  max_abs_err={err:.3e}  rel={rel:.2e}  |ref|max={scale:.3f}"


if __name__ == "__main__":
    torch.manual_seed(5)
    a = torch.randn(NK, MQ).to(torch.bfloat16)          # dS^T
    kt = torch.randn(DQK, NK).to(torch.bfloat16)        # k^T
    q = torch.randn(MQ, DQK).to(torch.bfloat16)         # q
    # Q1: C[m,d] = sum_nk a[nk,m] * kt[d,nk]
    q1_ref = a.float().T @ kt.float().T
    # Q2: C[nk,d] = sum_m a[nk,m] * q[m,d]
    q2_ref = a.float() @ q.float()
    # NT 对照: trans_A=False, trans_B=True 时 C[m,n]=sum_k a[m,k]*b[n,k]，
    # 取 a=dS^T (Nk,Mq)、b=kt (Dqk,Nk) 得 C[nk,d]=sum_m a[nk,m]*kt[d,m]
    nt_ref = a.float() @ kt.float().T

    print("Q1 trans_A=True / trans_B=False-ability:", flush=True)
    print("  dQ  (trans_A=True , trans_B=True ) :", run("tn", build_tn, ([a, kt], q1_ref)), flush=True)
    print("  dK  (trans_A=False, trans_B=False) :", run("nn", build_nn, ([a, q], q2_ref)), flush=True)
    print("  NT  (trans_A=False, trans_B=True ) :", run("nt", build_nt_l0, ([a, kt], nt_ref)), flush=True)

    # 单元价：q1/q2 的单次 GEMM 时延（同 28 核并发、无循环）
    for name, builder, args in (("tn", build_tn, (a, kt)), ("nn", build_nn, (a, q))):
        try:
            fn = tilelang.compile(builder(), target="ascend")
            cshape = (MQ, DQK) if name == "tn" else (NK, DQK)
            c = torch.zeros(cshape, dtype=torch.float32, device="npu")
            dev = [t.to("npu") for t in args] + [c]
            fn(*dev)
            torch.npu.synchronize()
            t0 = time.perf_counter()
            for _ in range(20):
                fn(*dev)
            torch.npu.synchronize()
            print(f"  {name} launch: {(time.perf_counter() - t0) / 20 * 1e6:.1f} us/launch (含 launch 地板)",
                  flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  {name} launch: skipped ({type(exc).__name__})", flush=True)
