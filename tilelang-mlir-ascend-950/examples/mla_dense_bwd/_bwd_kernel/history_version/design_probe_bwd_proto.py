"""Stage 1 设计期原型（revision v2 / cube-bound）：C2 结构的可行性 + 数值 + 单价。

与 v1（`_bwd_kernel.py`）的结构差异（DESIGN.md §0.6 R3/R6、§1.6.0 C2）：
  1. **5 个 GEMM 全部走显式 L0 路径**（`alloc_l0a/l0b` + 每次 GEMM 前一次 `T.copy(L1→L0)`）：
     L1 操作数路径与显式 L0 缓冲在同一 kernel 内混用会互相覆盖 L0A/L0B（实测 dk/dv 错 4.8e-1）。
  2. **dQ 用 `transpose_A=True`** 直接消费 `dsT (Nk,Mq)`：向量侧不再产生转置副本 `ds (Mq,Nk)`，
     省掉实测 3.75 µs/迭代的跨步 scatter 写；转置由编译器的 `asc_copy_l12l0a_transpose` 落在 MTE1。
  3. 向量 epilogue 只产 2 个**合并写** tile（`p16`,`dsT`），lane = query 轴。
  4. tile `Nk=96, Mq=64`（L0C 221184 B = 84.4%），每 query 行的 1280 B 填充摊到 96 个 key 行。

实测（2026-11-25，ascend-950pr-9579 / tilelang 0.1.15）：S=384 数值 PASS（causal 与非 causal），
S=2048/H=8 时 3.464 µs/迭代（num_stages=1）、3.439（=2，无收益）。

小 shape 端到端跑通 + 与 fp64 golden 比对（不跑大 shape）。

运行（仓根之外）：

    cd /tmp && source /workspace/tilelang-mlir-ascend-950/.agents/platforms/toolchains/ascend/env-950pr.sh \
      && TMPDIR=/workspace /workspace/venv-ascend950/bin/python \
         /workspace/tilelang-mlir-ascend-950/examples/mla_dense_bwd/_bwd_kernel/history_version/design_probe_bwd_proto.py
"""

import functools
import time

import tilelang
import tilelang.ascend.language as T
import torch
import torch_npu  # noqa: F401

DT = "bfloat16"
DQK, DV = 192, 128
NK, MQ = 96, 64          # key / query 块
HN = NK // 2             # 每个 AIV 的 key 行数（dual_copy 2:1）
STAGES = 1               # 先钉 1（正确性）；再试 2（重叠）


@functools.lru_cache(maxsize=None)
def build(batch, heads, seq_q, seq_kv, is_causal, stages):
    ntiles = (seq_kv + NK - 1) // NK
    grid = heads * batch * ntiles
    shift = seq_kv - seq_q
    scale = DQK**-0.5
    tq, tkv = batch * seq_q, batch * seq_kv

    @T.prim_func
    def main(
        q: T.Tensor((tq, heads, DQK), DT),          # type: ignore
        k: T.Tensor((tkv, heads, DQK), DT),         # type: ignore
        v: T.Tensor((tkv, heads, DV), DT),          # type: ignore
        do: T.Tensor((tq, heads, DV), DT),          # type: ignore
        lse: T.Tensor((batch, heads, seq_q), "float32"),    # type: ignore
        delta: T.Tensor((batch, heads, seq_q), "float32"),  # type: ignore
        dq_acc: T.Tensor((tq, heads, DQK), "float32"),      # type: ignore
        dk: T.Tensor((tkv, heads, DQK), DT),        # type: ignore
        dv: T.Tensor((tkv, heads, DV), DT),         # type: ignore
    ):
        with T.Kernel(grid) as bid:
            h = bid // (batch * ntiles)
            tile = bid % (batch * ntiles)
            b = tile // ntiles
            n0 = (tile % ntiles) * NK
            sid = T.call_extern("int64", "asc_get_sub_block_id")

            # ---- L1: key 侧常驻 -------------------------------------------
            k_s = T.alloc_l1((NK, DQK), DT)
            v_s = T.alloc_l1((NK, DV), DT)
            kT_s = T.alloc_l1((DQK, NK), DT)
            # ---- L1: 每迭代 -----------------------------------------------
            q_s = T.alloc_l1((MQ, DQK), DT)
            qT_s = T.alloc_l1((DQK, MQ), DT)
            do_s = T.alloc_l1((MQ, DV), DT)
            doT_s = T.alloc_l1((DV, MQ), DT)
            pT_s = T.alloc_l1((NK, MQ), DT)
            dsT_s = T.alloc_l1((NK, MQ), DT)      # dK 用：K-major
            dsM_s = T.alloc_l1((NK, MQ), DT)      # dQ 用：MN-major（trans_A=True）
            # ---- L0（五个 GEMM 全走显式 L0 路径）--------------------------
            a_k = T.alloc_l0a((NK, DQK), DT)      # trans_A=False：k / v / pT / dsT
            a_t = T.alloc_l0a((NK, MQ), DT)       # trans_A=True ：dsT（dQ）
            b_lo = T.alloc_l0b((MQ, DQK), DT)     # q / do
            b_hi = T.alloc_l0b((DQK, NK), DT)     # qT / doT / kT
            # ---- L0C 累加器 ------------------------------------------------
            st_c = T.alloc_l0c((NK, MQ), "float32")
            dpt_c = T.alloc_l0c((NK, MQ), "float32")
            dq_c = T.alloc_l0c((MQ, DQK), "float32")
            dk_c = T.alloc_l0c((NK, DQK), "float32")
            dv_c = T.alloc_l0c((NK, DV), "float32")
            # ---- UB（每 AIV 半区）-----------------------------------------
            st_u = T.alloc_shared((HN, MQ), "float32")
            dp_u = T.alloc_shared((HN, MQ), "float32")
            p16_u = T.alloc_shared((HN, MQ), DT)
            dsT_u = T.alloc_shared((HN, MQ), DT)
            lse_u = T.alloc_shared((MQ,), "float32")
            dlt_u = T.alloc_shared((MQ,), "float32")
            tail_u = T.alloc_shared((HN, DQK), "float32")
            tailv_u = T.alloc_shared((HN, DV), "float32")
            out_k = T.alloc_shared((HN, DQK), DT)
            out_v = T.alloc_shared((HN, DV), DT)

            koff = b * seq_kv + n0
            T.copy(k[koff : koff + NK, h, :], k_s)
            T.copy(v[koff : koff + NK, h, :], v_s)
            T.copy(k[koff : koff + NK, h, :], kT_s, transpose=True)


            loop_st = T.max(0, (n0 - shift) // MQ) if is_causal else 0
            loop_ed = (seq_q + MQ - 1) // MQ
            for kq in T.Pipelined(loop_st, loop_ed, num_stages=stages):
                m0 = kq * MQ
                qoff = b * seq_q + m0
                T.copy(q[qoff : qoff + MQ, h, :], q_s)
                T.copy(do[qoff : qoff + MQ, h, :], do_s)
                T.copy(q[qoff : qoff + MQ, h, :], qT_s, transpose=True)
                T.copy(do[qoff : qoff + MQ, h, :], doT_s, transpose=True)
                T.copy(lse[b, h, m0 : m0 + MQ], lse_u)
                T.copy(delta[b, h, m0 : m0 + MQ], dlt_u)

                T.copy(k_s, a_k)
                T.copy(q_s, b_lo)
                T.gemm(a_k, b_lo, st_c, transpose_B=True, clear_accum=True)
                T.copy(v_s, a_k[0:NK, 0:DV])
                T.copy(do_s, b_lo[0:MQ, 0:DV])
                T.gemm(a_k[0:NK, 0:DV], b_lo[0:MQ, 0:DV], dpt_c,
                       transpose_B=True, clear_accum=True)
                T.dual_copy(st_c, st_u)
                T.dual_copy(dpt_c, dp_u)

                fast = (n0 + NK <= seq_kv) & (m0 + MQ <= seq_q)
                if is_causal:
                    fast = fast & (n0 + NK - 1 <= m0 + shift)
                if fast:
                    with T.SimtVF(threads=128):
                        for i, j in T.Parallel(HN, MQ):
                            p = T.exp(st_u[i, j] * scale - lse_u[j])
                            p16_u[i, j] = T.cast(p, DT)
                            dsT_u[i, j] = T.cast(p * (dp_u[i, j] - dlt_u[j]), DT)
                else:
                    with T.SimtVF(threads=128):
                        for i, j in T.Parallel(HN, MQ):
                            gi = n0 + T.cast(sid, "int32") * HN + i
                            gj = m0 + j
                            vis = (gi < seq_kv) & (gj < seq_q)
                            if is_causal:
                                vis = vis & (gi <= gj + shift)
                            p = T.exp(st_u[i, j] * scale - lse_u[j])
                            pv = T.if_then_else(vis, p, T.cast(0, "float32"))
                            p16_u[i, j] = T.cast(pv, DT)
                            dsT_u[i, j] = T.cast(pv * (dp_u[i, j] - dlt_u[j]), DT)

                T.dual_copy(p16_u, pT_s)
                T.dual_copy(dsT_u, dsT_s)
                T.copy(pT_s, a_k[0:NK, 0:MQ])
                T.copy(doT_s, b_hi[0:DV, 0:MQ])
                T.gemm(a_k[0:NK, 0:MQ], b_hi[0:DV, 0:MQ], dv_c,
                       transpose_B=True, clear_accum=(kq == loop_st))
                T.copy(dsT_s, a_k[0:NK, 0:MQ])
                T.copy(qT_s, b_hi[0:DQK, 0:MQ])
                T.gemm(a_k[0:NK, 0:MQ], b_hi[0:DQK, 0:MQ], dk_c,
                       transpose_B=True, clear_accum=(kq == loop_st))
                T.copy(dsT_s, a_t)
                T.copy(kT_s, b_hi)
                T.gemm(a_t, b_hi, dq_c, transpose_A=True, transpose_B=True, clear_accum=True)
                T.set_atomic("add", "float32")
                T.copy(dq_c, dq_acc[qoff : qoff + MQ, h, :])
                T.set_atomic_none()

            T.dual_copy(dk_c, tail_u)
            T.dual_copy(dv_c, tailv_u)
            with T.SimtVF(threads=128):
                for i, d in T.Parallel(HN, DQK):
                    out_k[i, d] = T.cast(tail_u[i, d] * scale, DT)
                for i, d in T.Parallel(HN, DV):
                    out_v[i, d] = T.cast(tailv_u[i, d], DT)
            kown = koff + T.cast(sid, "int32") * HN
            T.copy(out_k, dk[kown : kown + HN, h, :])
            T.copy(out_v, dv[kown : kown + HN, h, :])

    return main


def golden(q, k, v, do, scale, is_causal):
    """fp64 物化参考（bottom-right causal）。"""
    qf, kf, vf, dof = (t.double() for t in (q, k, v, do))
    B, Sq, H, D = q.shape
    Skv, Dv = k.shape[1], v.shape[-1]
    shift = Skv - Sq
    kp = torch.arange(Skv)
    qp = torch.arange(Sq) + shift
    hid = kp[None, :] > qp[:, None] if is_causal else None
    dq = torch.zeros(B, Sq, H, D, dtype=torch.float64)
    dk = torch.zeros(B, Skv, H, D, dtype=torch.float64)
    dv = torch.zeros(B, Skv, H, Dv, dtype=torch.float64)
    for b in range(B):
        for h0 in range(0, H, 2):
            hs = slice(h0, min(h0 + 2, H))
            s = torch.einsum("shd,nhd->hsn", qf[b, :, hs], kf[b, :, hs]) * scale
            if hid is not None:
                s = s.masked_fill(hid[None], float("-inf"))
            lse = s.logsumexp(-1, keepdim=True)
            p = torch.exp(s - lse).nan_to_num(0.0)
            dp = torch.einsum("shd,nhd->hsn", dof[b, :, hs], vf[b, :, hs])
            ds = p * (dp - (p * dp).sum(-1, keepdim=True))
            dv[b, :, hs] += torch.einsum("hsn,shd->nhd", p, dof[b, :, hs])
            dk[b, :, hs] += torch.einsum("hsn,shd->nhd", ds, qf[b, :, hs])
            dq[b, :, hs] = torch.einsum("hsn,nhd->shd", ds, kf[b, :, hs])
    return dq * scale, dk * scale, dv


def run(batch, seq_q, seq_kv, heads, is_causal=True, stages=1, seed=0):
    scale = DQK**-0.5
    g = torch.Generator().manual_seed(seed)
    def rnd(shape):
        return torch.randn(shape, generator=g).to(torch.bfloat16)
    q = rnd((batch, seq_q, heads, DQK))
    k = rnd((batch, seq_kv, heads, DQK))
    v = rnd((batch, seq_kv, heads, DV))
    do = rnd((batch, seq_q, heads, DV))
    # lse/delta from fp32 forward
    qf, kf, vf, dof = (t.float() for t in (q, k, v, do))
    shift = seq_kv - seq_q
    hid = torch.arange(seq_kv)[None, :] > (torch.arange(seq_q) + shift)[:, None] if is_causal else None
    lse = torch.empty(batch, heads, seq_q)
    delta = torch.empty(batch, heads, seq_q)
    for b in range(batch):
        for h in range(heads):
            s = (qf[b, :, h] @ kf[b, :, h].T) * scale
            if hid is not None:
                s = s.masked_fill(hid, float("-inf"))
            row = s.logsumexp(-1)
            p = torch.exp(s - row[:, None]).nan_to_num(0.0)
            lse[b, h], delta[b, h] = row, (p @ vf[b, :, h] * dof[b, :, h]).sum(-1)
    dq_t, dk_t, dv_t = golden(q, k, v, do, scale, is_causal)

    fn = tilelang.compile(build(batch, heads, seq_q, seq_kv, is_causal, stages), target="ascend")
    dev = "npu"
    qd = q.reshape(batch * seq_q, heads, DQK).to(dev)
    kd = k.reshape(batch * seq_kv, heads, DQK).to(dev)
    vd = v.reshape(batch * seq_kv, heads, DV).to(dev)
    dod = do.reshape(batch * seq_q, heads, DV).to(dev)
    lsed, dltd = lse.to(dev), delta.to(dev)
    dqd = torch.zeros(batch * seq_q, heads, DQK, dtype=torch.float32, device=dev)
    dkd = torch.zeros(batch * seq_kv, heads, DQK, dtype=torch.bfloat16, device=dev)
    dvd = torch.zeros(batch * seq_kv, heads, DV, dtype=torch.bfloat16, device=dev)
    args = (qd, kd, vd, dod, lsed, dltd, dqd, dkd, dvd)
    fn(*args)
    torch.npu.synchronize()
    got_dq = (dqd.cpu() * scale).to(torch.bfloat16).float()
    got_dk = dkd.cpu().float()
    got_dv = dvd.cpu().float()
    ref_dq = dq_t.reshape(batch * seq_q, heads, DQK).to(torch.bfloat16).float()
    ref_dk = dk_t.reshape(batch * seq_kv, heads, DQK).to(torch.bfloat16).float()
    ref_dv = dv_t.reshape(batch * seq_kv, heads, DV).to(torch.bfloat16).float()
    e = [float((a - b).abs().max()) for a, b in ((got_dq, ref_dq), (got_dk, ref_dk), (got_dv, ref_dv))]
    ok = all(x < 6e-2 for x in e)   # bf16 + 强抵消：设计期宽判据（Stage 3 用 5e-3/1.6e-2 正式判）
    # 计时（先读结果再计时）
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        fn(*args)
        torch.npu.synchronize()
        best = min(best, time.perf_counter() - t0)
    ntiles = (seq_kv + NK - 1) // NK
    iters = sum(min(seq_q // MQ, max(0, (m0 := ((t % ntiles) * NK)) // MQ + 1)) for t in range(ntiles))
    iters = 0
    for t in range(ntiles):
        n0 = t * NK
        st = max(0, (n0 - shift) // MQ) if is_causal else 0
        iters += (seq_q + MQ - 1) // MQ - st
    tot_iter = iters * heads * batch
    return ok, e, best * 1e3, tot_iter


if __name__ == "__main__":
    # 1) 数值（小 shape）：causal / 非 causal 各一次
    for causal in (True, False):
        ok, e, ms, it = run(1, 384, 384, 4, causal, 1)
        print(f"[{'PASS' if ok else 'FAIL'}] causal={causal} T={ms:7.3f} ms for {it} block-iterations  "
              f"err dq/dk/dv = {e[0]:.2e}/{e[1]:.2e}/{e[2]:.2e}", flush=True)
    # 2) 单价（中等 shape，不跑生产 shape）：stages=1 / 2
    for stages in (1, 2):
        ok, e, ms, it = run(1, 2048, 2048, 8, True, stages)
        print(f"[{'PASS' if ok else 'FAIL'}] S=2048 stages={stages} T={ms:8.3f} ms  iters={it}  "
              f"per-iter={ms * 1e3 * 28 / max(it, 1):6.3f} us  err={e[0]:.2e}", flush=True)
