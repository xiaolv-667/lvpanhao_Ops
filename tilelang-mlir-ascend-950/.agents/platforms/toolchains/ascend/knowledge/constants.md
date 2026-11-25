# Ascend950PR_9579 实测代价常数（Stage 4 回填）

条目按 [knowledge-entry-schema.md](../../../../skills/_shared/standards/knowledge-entry-schema.md) 登记；
本文件只记**真机实测**常数，不引用型号 INI 的频率字段换算值。

---
id: CONST-950-gemm-mte2-burst
kind: constant
status: verified
origin_task: shared_expert_mlp-SharedExpertMLPFwdOp-20261124T132107Z (Stage 4)
toolchain: TileLang ascend 0.1.15 + CANN 9.2.1, 2026-11-24
toolchain_stamp: 7a52f2f1274df07d8f4c9c01353ce9d290f88c16e85047de0c425b6a009de4aa
toolchain_id: ascend
hardware_id: ascend-950pr-9579
repro: repro/PL-950-gemm-l1-fill.py
---

# bf16 L1 操作数 GEMM 的 MTE2 有效带宽取决于每行 DMA 段长度

被测算子：`C = A[m,k] @ B[n,k]^T`，A/B 均按 `(rows, k)` 存放，`transpose_B=True` 的
L1 操作数路径（`ascend_gemm_l1` 模板 + `T.Pipelined` 多版本 L1），fp32 L0C 累加，
`m=512, n=36864, k=7168, bf16`，`block_m=256, block_n=256, num_stages=3`。

| block_k | 每行 DMA 段 | `aic_mte2_active_bw` | gate/up 时延 |
|---:|---:|---:|---:|
| 64 | 128 B | 70 GB/s/core | 1201.6 µs |
| 128 | 256 B | 101–105 GB/s/core | 811.8 µs |
| 256 | 512 B | — | 818.9 µs（`num_stages=2`，L1 恰好 512 KiB） |

- 聚合口径：28 Cube 核 × 101–105 GB/s ≈ **2.7–2.9 TB/s** 的 L2→L1 填充带宽（MTE2 满载时）。
- 对照常数：HBM 有效带宽 **1.42–1.48 TB/s**（`kimi-k2-t1`：权重 792 MB / 535.9 µs = 1.48 TB/s，
  权重只读一遍、无复用）。
- 对照常数：bf16 Cube 有效峰值 **356–359 TF/s**（256×256 tile、k=18432 的 down GEMM，
  `aic_cube_ratio=0.978` × 367.6 µs = 358 µs，而 378 TF/s 的名义下界是 357 µs）。

边界：结论限于 bf16/fp16 L1 操作数 NT GEMM + 本工具链的 `ascend_gemm_l1` 模板；`block_k=256`
只测到「不比 128 快」，未验证更大 K 块（L1 预算不允许 `block_k ≥ 192` 配 `num_stages ≥ 3`）。
`aic_mte2_active_bw` 是 PipeUtilization CSV 的逐 block 字段，本表取其块间均值。

---

id: CONST-950-kernel-launch-floor-48blk
kind: constant
status: verified
origin_task: engram_gate_conv-engram_gate_conv-20261124T171256Z (Stage 4)
toolchain: TileLang ascend 0.1.15 + CANN 9.2.1, 2026-11-24
toolchain_stamp: 7a52f2f1274df07d8f4c9c01353ce9d290f88c16e85047de0c425b6a009de4aa
toolchain_id: ascend
hardware_id: ascend-950pr-9579
repro: repro/TRAP-950-rowload-dma-chain.py
provenance: 溯源见 origin_task；`engram_gate_conv_perf/perf_opt/opt_log.md` §1.3/§2/§4，
  `perf_opt/experiments/floor_probe.py` kind=0，原始采集 `perf_opt/profiles/floor/*`
apis: [T.Kernel, T.copy]
dtype: [bfloat16]
---

# 小算子 launch 地板：48 块一维 grid ≈ 2.7–2.8 µs（Task Duration 口径）

单 launch、`T.Kernel(48)`、d=768、bf16，每块只做 1 条行装载 + 2 条行落盘：msopprof
`Task Duration(us)` = **2.72 / 2.75 / 2.83 µs**（三次独立采集，另一次 3.44 µs 为同分布上沿）。
这既是该几何的**地板**，也是「多一次 launch 的代价」：两段 launch 的结构天然多付 ≈2.7 µs。

用途：判「两段 launch 备选是否可能胜出」。本算子的设计层不确定常数①（DESIGN 假设 2–5 µs）由此锚定为
2.7 µs，两段结构因此在 `train-s96` 上实测 9.99 µs vs 单 launch 8.46 µs（+18%），备选被否。

边界：地板随 grid 与 dtype 变化未系统扫描；`grid=1` 的同一探针给出 5.7–6.0 µs（与 grid=48 反向），
说明 1-D grid 的调度形态在极小 grid 下另有开销，**不要**用 grid=1 的 kernel 测地板。
口径是同任务平台的 kernel-only Task Duration，不是端到端时延，也不是 NPU event。
