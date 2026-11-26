"""Stage 1 revision v3 探针：`dk/dv` 块尾写回的 request 隔离（DESIGN.md §5.4 / §8.3 L0-9）。

目的：证明 v2（3-D packed 视图 `(T,H,D)`）的块尾 `dk/dv` 写回在 **B≥2 且 S_kv % 96 ≠ 0** 时会
写到**下一个 request 的行**，而 4-D batch 视图 `(B,S,H,D)` 由平台 OOB clamp 按 request 裁剪后不再越界。

两个变体只差 GM 张量的签名与寻址（C2 结构完全相同：5 个显式 L0 GEMM、`trans_A` 的 dQ、
tile 96/64、半区 dual_copy + 2-tile 合并写）：
  layout="3d"：`k/v/dk/dv : (B*S, H, D)`，写回 `dk[koff + sid*HN : …]`（koff = b*S_kv + n0）
               → OOB clamp 的裁剪维度是**整张张量** `B*S_kv`
  layout="4d"：`k/v/dk/dv : (B, S, H, D)`，写回 `dk[b, n0 + sid*HN : …]`
               → OOB clamp 的裁剪维度是**本 request 的** `S_kv`

判据（每个变体各跑 5 次，B=2, S=512, H=4，512 = 5×96 + 32）：
  - 逐 request 比对 golden；重点看 **request 1 的 key 0…63**（被 request 0 的尾块 AIV0/1 越界写入的行）
  - 4-D 变体必须 5/5 逐元素一致；3-D 变体在因果掩码下尾块迭代数最少（常先完成而侥幸正确），
    非因果掩码下两块迭代数相同 ⇒ 竞态暴露

运行（仓根之外；平台 `execution.prototype_command` 的解释器无 torch/tilelang，见 DESIGN.md §3.5.4）：

    cd /tmp && source /workspace/tilelang-mlir-ascend-950/.agents/platforms/toolchains/ascend/env-950pr.sh \
      && TMPDIR=/workspace /workspace/venv-ascend950/bin/python \
         /workspace/tilelang-mlir-ascend-950/examples/mla_dense_bwd/_bwd_kernel/history_version/design_probe_b1_tailrace.py
"""

import functools

import tilelang
import tilelang.ascend.language as T
import torch
import torch_npu  # noqa: F401

DT = "bfloat16"
DQK, DV = 192, 128
NK, MQ = 96, 64
HN = NK // 2


def _build_3d(batch, heads, seq_q, seq_kv, is_causal, stages=1):
    ntiles = (seq_kv + NK - 1) // NK
    grid = heads * batch * ntiles
    shift = seq_kv - seq_q
    scale = DQK**-0.5
    tq, tkv = batch * seq_q, batch * seq_kv

    @T.prim_func
    def main(
        q: T.Tensor((tq, heads, DQK), DT),  # type: ignore
        k: T.Tensor((tkv, heads, DQK), DT),  # type: ignore
        v: T.Tensor((tkv, heads, DV), DT),  # type: ignore
        do: T.Tensor((tq, heads, DV), DT),  # type: ignore
        lse: T.Tensor((batch, heads, seq_q), "float32"),  # type: ignore
        delta: T.Tensor((batch, heads, seq_q), "float32"),  # type: ignore
        dq_acc: T.Tensor((tq, heads, DQK), "float32"),  # type: ignore
        dk: T.Tensor((tkv, heads, DQK), DT),  # type: ignore
        dv: T.Tensor((tkv, heads, DV), DT),  # type: ignore
    ):
        with T.Kernel(grid) as bid:
            h = bid // (batch * ntiles)
            tile = bid % (batch * ntiles)
            b = tile // ntiles
            n0 = (tile % ntiles) * NK
            sid = T.call_extern("int64", "asc_get_sub_block_id")

            k_s = T.alloc_l1((NK, DQK), DT)
            v_s = T.alloc_l1((NK, DV), DT)
            kT_s = T.alloc_l1((DQK, NK), DT)
            q_s = T.alloc_l1((MQ, DQK), DT)
            qT_s = T.alloc_l1((DQK, MQ), DT)
            do_s = T.alloc_l1((MQ, DV), DT)
            doT_s = T.alloc_l1((DV, MQ), DT)
            pT_s = T.alloc_l1((NK, MQ), DT)
            dsT_s = T.alloc_l1((NK, MQ), DT)
            dsM_s = T.alloc_l1((NK, MQ), DT)
            a_k = T.alloc_l0a((NK, DQK), DT)
            a_t = T.alloc_l0a((NK, MQ), DT)
            b_lo = T.alloc_l0b((MQ, DQK), DT)
            b_hi = T.alloc_l0b((DQK, NK), DT)
            st_c = T.alloc_l0c((NK, MQ), "float32")
            dpt_c = T.alloc_l0c((NK, MQ), "float32")
            dq_c = T.alloc_l0c((MQ, DQK), "float32")
            dk_c = T.alloc_l0c((NK, DQK), "float32")
            dv_c = T.alloc_l0c((NK, DV), "float32")
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
            # ---- 唯一的写回差异：行偏移用整张张量的全局行号 ----
            kown = koff + T.cast(sid, "int32") * HN
            T.copy(out_k, dk[kown : kown + HN, h, :])
            T.copy(out_v, dv[kown : kown + HN, h, :])

    return main


def _build_4d(batch, heads, seq_q, seq_kv, is_causal, stages=1):
    ntiles = (seq_kv + NK - 1) // NK
    grid = heads * batch * ntiles
    shift = seq_kv - seq_q
    scale = DQK**-0.5

    @T.prim_func
    def main(  # noqa: F811
        q: T.Tensor((batch, seq_q, heads, DQK), DT),  # type: ignore
        k: T.Tensor((batch, seq_kv, heads, DQK), DT),  # type: ignore
        v: T.Tensor((batch, seq_kv, heads, DV), DT),  # type: ignore
        do: T.Tensor((batch, seq_q, heads, DV), DT),  # type: ignore
        lse: T.Tensor((batch, heads, seq_q), "float32"),  # type: ignore
        delta: T.Tensor((batch, heads, seq_q), "float32"),  # type: ignore
        dq_acc: T.Tensor((batch, seq_q, heads, DQK), "float32"),  # type: ignore
        dk: T.Tensor((batch, seq_kv, heads, DQK), DT),  # type: ignore
        dv: T.Tensor((batch, seq_kv, heads, DV), DT),  # type: ignore
    ):
        with T.Kernel(grid) as bid:
            h = bid // (batch * ntiles)
            tile = bid % (batch * ntiles)
            b = tile // ntiles
            n0 = (tile % ntiles) * NK
            sid = T.call_extern("int64", "asc_get_sub_block_id")

            k_s = T.alloc_l1((NK, DQK), DT)
            v_s = T.alloc_l1((NK, DV), DT)
            kT_s = T.alloc_l1((DQK, NK), DT)
            q_s = T.alloc_l1((MQ, DQK), DT)
            qT_s = T.alloc_l1((DQK, MQ), DT)
            do_s = T.alloc_l1((MQ, DV), DT)
            doT_s = T.alloc_l1((DV, MQ), DT)
            pT_s = T.alloc_l1((NK, MQ), DT)
            dsT_s = T.alloc_l1((NK, MQ), DT)
            dsM_s = T.alloc_l1((NK, MQ), DT)
            a_k = T.alloc_l0a((NK, DQK), DT)
            a_t = T.alloc_l0a((NK, MQ), DT)
            b_lo = T.alloc_l0b((MQ, DQK), DT)
            b_hi = T.alloc_l0b((DQK, NK), DT)
            st_c = T.alloc_l0c((NK, MQ), "float32")
            dpt_c = T.alloc_l0c((NK, MQ), "float32")
            dq_c = T.alloc_l0c((MQ, DQK), "float32")
            dk_c = T.alloc_l0c((NK, DQK), "float32")
            dv_c = T.alloc_l0c((NK, DV), "float32")
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

            T.copy(k[b, n0 : n0 + NK, h, :], k_s)
            T.copy(v[b, n0 : n0 + NK, h, :], v_s)
            T.copy(k[b, n0 : n0 + NK, h, :], kT_s, transpose=True)

            loop_st = T.max(0, (n0 - shift) // MQ) if is_causal else 0
            loop_ed = (seq_q + MQ - 1) // MQ
            for kq in T.Pipelined(loop_st, loop_ed, num_stages=stages):
                m0 = kq * MQ
                T.copy(q[b, m0 : m0 + MQ, h, :], q_s)
                T.copy(do[b, m0 : m0 + MQ, h, :], do_s)
                T.copy(q[b, m0 : m0 + MQ, h, :], qT_s, transpose=True)
                T.copy(do[b, m0 : m0 + MQ, h, :], doT_s, transpose=True)
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
                T.copy(dq_c, dq_acc[b, m0 : m0 + MQ, h, :])
                T.set_atomic_none()

            T.dual_copy(dk_c, tail_u)
            T.dual_copy(dv_c, tailv_u)
            with T.SimtVF(threads=128):
                for i, d in T.Parallel(HN, DQK):
                    out_k[i, d] = T.cast(tail_u[i, d] * scale, DT)
                for i, d in T.Parallel(HN, DV):
                    out_v[i, d] = T.cast(tailv_u[i, d], DT)
            # ---- 唯一的写回差异：request 内的行偏移（OOB clamp 按本 request 裁剪）----
            kown = n0 + T.cast(sid, "int32") * HN
            T.copy(out_k, dk[b, kown : kown + HN, h, :])
            T.copy(out_v, dv[b, kown : kown + HN, h, :])

    return main


@functools.lru_cache(maxsize=None)
def build(batch, heads, seq_q, seq_kv, is_causal, layout, stages=1):
    fn = _build_3d if layout == "3d" else _build_4d
    return tilelang.compile(fn(batch, heads, seq_q, seq_kv, is_causal, stages), target="ascend")


def golden(q, k, v, do, scale, is_causal, slab=4):
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
        for h0 in range(0, H, slab):
            hs = slice(h0, min(h0 + slab, H))
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


def make_inputs(batch, seq_q, seq_kv, heads, is_causal, seed):
    scale = DQK**-0.5
    g = torch.Generator().manual_seed(seed)

    def rnd(shape):
        return torch.randn(shape, generator=g).to(torch.bfloat16)

    q = rnd((batch, seq_q, heads, DQK))
    k = rnd((batch, seq_kv, heads, DQK))
    v = rnd((batch, seq_kv, heads, DV))
    do = rnd((batch, seq_q, heads, DV))
    qf, kf, vf, dof = (t.float() for t in (q, k, v, do))
    shift = seq_kv - seq_q
    hid = (torch.arange(seq_kv)[None, :] > (torch.arange(seq_q) + shift)[:, None]
           if is_causal else None)
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
    return q, k, v, do, lse, delta


def run_once(batch, seq_q, seq_kv, heads, is_causal, layout, seed=0, stages=1):
    scale = DQK**-0.5
    q, k, v, do, lse, delta = make_inputs(batch, seq_q, seq_kv, heads, is_causal, seed)
    dq_t, dk_t, dv_t = golden(q, k, v, do, scale, is_causal)
    fn = build(batch, heads, seq_q, seq_kv, is_causal, layout, stages)
    dev = "npu"
    if layout == "3d":
        args = (
            q.reshape(batch * seq_q, heads, DQK).to(dev),
            k.reshape(batch * seq_kv, heads, DQK).to(dev),
            v.reshape(batch * seq_kv, heads, DV).to(dev),
            do.reshape(batch * seq_q, heads, DV).to(dev),
            lse.to(dev), delta.to(dev),
            torch.zeros(batch * seq_q, heads, DQK, dtype=torch.float32, device=dev),
            torch.zeros(batch * seq_kv, heads, DQK, dtype=torch.bfloat16, device=dev),
            torch.zeros(batch * seq_kv, heads, DV, dtype=torch.bfloat16, device=dev),
        )
    else:
        args = (
            q.to(dev), k.to(dev), v.to(dev), do.to(dev), lse.to(dev), delta.to(dev),
            torch.zeros(batch, seq_q, heads, DQK, dtype=torch.float32, device=dev),
            torch.zeros(batch, seq_kv, heads, DQK, dtype=torch.bfloat16, device=dev),
            torch.zeros(batch, seq_kv, heads, DV, dtype=torch.bfloat16, device=dev),
        )
    fn(*args)
    torch.npu.synchronize()
    dq_g, dk_g, dv_g = args[6].cpu().float(), args[7].cpu().float(), args[8].cpu().float()
    if layout == "3d":
        dq_g = dq_g.reshape(batch, seq_q, heads, DQK)
        dk_g = dk_g.reshape(batch, seq_kv, heads, DQK)
        dv_g = dv_g.reshape(batch, seq_kv, heads, DV)
    out = {}
    for name, got, ref in (("dq", dq_g * scale, dq_t), ("dk", dk_g, dk_t), ("dv", dv_g, dv_t)):
        diff = (got.double() - ref.double()).abs()
        out[name] = float(diff.max())
        for b in range(batch):
            out[f"{name}_b{b}"] = float(diff[b].max())
    victim = slice(0, 64)   # request 1 的 key 0…63：被 request 0 尾块越界写入的行
    out["dk_victim"] = float((dk_g[1, victim].double() - dk_t[1, victim].double()).abs().max())
    out["dv_victim"] = float((dv_g[1, victim].double() - dv_t[1, victim].double()).abs().max())
    return out


def time_iters(batch, seq_q, seq_kv, heads, layout, is_causal=True, reps=3, stages=1):
    """单价（µs/迭代）：确认 4-D 签名不改变 C2 的迭代成本。"""
    import time
    scale = DQK**-0.5
    q, k, v, do, lse, delta = make_inputs(batch, seq_q, seq_kv, heads, is_causal, 0)
    fn = build(batch, heads, seq_q, seq_kv, is_causal, layout, stages)
    dev = "npu"
    if layout == "3d":
        args = (q.reshape(batch * seq_q, heads, DQK).to(dev),
                k.reshape(batch * seq_kv, heads, DQK).to(dev),
                v.reshape(batch * seq_kv, heads, DV).to(dev),
                do.reshape(batch * seq_q, heads, DV).to(dev),
                lse.to(dev), delta.to(dev),
                torch.zeros(batch * seq_q, heads, DQK, dtype=torch.float32, device=dev),
                torch.zeros(batch * seq_kv, heads, DQK, dtype=torch.bfloat16, device=dev),
                torch.zeros(batch * seq_kv, heads, DV, dtype=torch.bfloat16, device=dev))
    else:
        args = (q.to(dev), k.to(dev), v.to(dev), do.to(dev), lse.to(dev), delta.to(dev),
                torch.zeros(batch, seq_q, heads, DQK, dtype=torch.float32, device=dev),
                torch.zeros(batch, seq_kv, heads, DQK, dtype=torch.bfloat16, device=dev),
                torch.zeros(batch, seq_kv, heads, DV, dtype=torch.bfloat16, device=dev))
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(*args)
        torch.npu.synchronize()
        best = min(best, time.perf_counter() - t0)
    ntiles = (seq_kv + NK - 1) // NK
    shift = seq_kv - seq_q
    iters = 0
    for t in range(ntiles):
        st = max(0, (t * NK - shift) // MQ) if is_causal else 0
        iters += (seq_q + MQ - 1) // MQ - st
    return best * 1e3, iters * heads * batch


if __name__ == "__main__":
    B, S, H = 2, 512, 4          # 512 = 5*96 + 32 → request 0 的尾块越界 64 行进入 request 1
    for causal in (True, False):
        for layout in ("3d", "4d"):
            worst = {"dk_victim": 0.0, "dv_victim": 0.0, "dk": 0.0, "dv": 0.0}
            fails = 0
            for rep in range(5):
                r = run_once(B, S, S, H, causal, layout, seed=rep)
                for key in worst:
                    worst[key] = max(worst[key], r[key])
                bad = max(r["dk"], r["dv"]) > 6e-2
                fails += int(bad)
                print(
                    f"  rep{rep} {layout} causal={causal}: "
                    f"dk={r['dk']:.2e} dv={r['dv']:.2e} dq={r['dq']:.2e} | "
                    f"victim(r1 key0..63) dk={r['dk_victim']:.2e} dv={r['dv_victim']:.2e} "
                    f"-> {'FAIL' if bad else 'ok'}",
                    flush=True,
                )
            print(
                f"[{'FAIL' if fails else 'PASS'}] layout={layout} causal={causal} "
                f"坏run数={fails}/5  最差: dk={worst['dk']:.2e} dv={worst['dv']:.2e} "
                f"victim_dk={worst['dk_victim']:.2e} victim_dv={worst['dv_victim']:.2e}",
                flush=True,
            )

    # 4-D + num_stages=2 的数值回归（§4.8/§6.3 选定配置）
    for causal in (True, False):
        for rep in range(2):
            r = run_once(B, S, S, H, causal, "4d", seed=rep, stages=2)
            bad = max(r["dk"], r["dv"]) > 6e-2
            print(f"  [stages=2] 4d causal={causal} rep{rep}: dk={r['dk']:.2e} dv={r['dv']:.2e} "
                  f"victim_dk={r['dk_victim']:.2e} -> {'FAIL' if bad else 'PASS'}", flush=True)

    # 单价对照：4-D 签名是否改变 C2 的迭代成本（S=2048, H=8, causal）
    for layout, stages in (("3d", 1), ("4d", 1), ("4d", 2)):
        ms, it = time_iters(1, 2048, 2048, 8, layout, stages=stages)
        print(
            f"[cost] layout={layout} stages={stages} S=2048 H=8 causal: {ms:.3f} ms / {it} 块迭代 "
            f"= {ms * 1e3 * 28 / it:.3f} µs/迭代（28 核归一）",
            flush=True,
        )
