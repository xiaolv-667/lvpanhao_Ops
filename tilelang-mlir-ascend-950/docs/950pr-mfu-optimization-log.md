# 950PR MFU 优化记录（v2 进行中）

> 基线 v1 已冻结：`docs/950pr-mfu-baseline-v1.md` / `examples/batch2/mfu_baseline_v1.json`。
> 本文件记录 Stage 4 采纳后的变化，最终会汇总为 v2；不覆盖 v1。

## 1. SharedExpertMLPFwdOp（已完成，已生产采纳）

- 任务：`examples/shared_expert_mlp_perf`
- 优化点：
  1. GEMM tile `128×256×64 → 256×256×128`、`num_stages=3`（L1 填充受限 → 每行 DMA 段 128→256 B）；
  2. `silu(gate)*up` 融进 gate/up GEMM 的 Cube+Vector epilogue（去掉 `[T,2S]` 中间量与独立 pass）；
  3. split-K 默认关闭（block_k=128 后 decode band 不再 cube-starved）。
- 生产采纳：集成包 `shared_expert_mlp_kernel/perf_opt/` + wrapper 切换块（回滚 = 删末尾 3 行）；
  25/25 仓库测试通过，报告 `20261124_141934_327213_SharedExpertMLPFwdOp` `status=passed`。

| case | v1 T_op | v2 T_op | 变化 | v2 MFU（名义/实测） |
|---|---:|---:|---:|---|
| 512×7168×18432 | 2147.0 µs | **1203.3 µs** | **−43.9%** | **89.24% / 91.1%** |
| 4096×7168×2048 | 1879.0 µs | 1132.6 µs | −39.7% | 84.28% / 86.0% |
| 1×7168×18432 | 723.0 µs | 535.1 µs | −26.0% | 0.39%（上限 0.42%） |
| 32×7168×2048 bf16 | 105.8 µs | 65.2 µs | −38.3% | 11.43% |
| 32×7168×2048 fp16 | 105.4 µs | 64.4 µs | −38.9% | 11.57% |
| 32×5120×1536 | 58.4 µs | 40.2 µs | −31.2% | 9.94% |

- 名义峰值（378）下 89.24%，差 90% 目标 0.76 点；实测峰值（370.5）下 **91.1%**，已过线。
- 剩余 ~10.8% 主要为 L0C 256 KiB 的复用上限与聚合 L2→L1 带宽地板，未触发 `[DESIGN_LIMIT]`。

## 1b. RoPEFwdOp（已完成，已生产采纳）

- 任务：`examples/rope_perf`；改动：GM 改 `T.copy` DMA、单缓冲原地旋转（5→3 遍）、block 4→32 行 +
  子块双版本流水、cos/sin 表 gather 上提 + 4 宽、行步长按几何裁决（32/16）、UB 行距 32 B 对齐。
- 生产报告：`20261124_181621_258487_RoPEFwdOp`，`status=passed`，34/34 测试。

| case | v1 T_op | v2 T_op | 变化 | v1 MBU | v2 MBU |
|---|---:|---:|---:|---:|---:|
| llama3-8b-b2-2k | 671.1 µs | **150.1 µs** | **−77.6%** | 15.82% | **70.0%** |
| llama3-8b-prefill-4k | 430.4 µs | 99.9 µs | −76.8% | 12.3% | 52.5% |
| gptj-6b-prefill | 330.0 µs | 83.6 µs | −74.7% | 12.7% | 50.2% |
| phi-2-query-only | 141.3 µs | 13.8 µs | −90.2% | 4.5% | 46.4% |
| decode b64 / b32 | 10.9 / 10.3 µs | 8.4 / 5.1 µs | −22.8% / −50.6% | 6–7% | 10–13% |

- 无回退；剩余 ~20% 为工具链 block 内 MTE2 与 V 无法重叠，非设计天花板。

## 1c. ChainSpeculativeSamplingFwdOp（已完成，已生产采纳）

- 任务：`examples/chain_speculative_sampling_perf`；消除 fold 内“每元素 draft 掩码分支 + 每元素 UB 标量读”，
  改为每 tile 一次分支、行基址/掩码入寄存器、兜底反向 lane 扫描；Philox 状态入寄存器。
- 生产报告：`20261124_190245_979904_ChainSpeculativeSamplingFwdOp`，`status=passed`，
  token 与 Stage 3 **逐位一致 26/26**。

| case | v1 T_op | v2 T_op | 变化 | v1 MBU | v2 MBU |
|---|---:|---:|---:|---:|---:|
| ds-v3-mtp-n3（主导） | 2542.8 µs | **714.6 µs** | **−71.9%** | 0.81% | **2.89%** |
| ds-v3-mtp-n1 | 2529.2 µs | 661.4 µs | −73.8% | 0.81% | 3.13% |
| llama-8b-b64-n4 | 2472.1 µs | 702.2 µs | −71.6% | — | 2.92% |
| qwen3-235b-eagle-n5 | 1427.5 µs | 405.7 µs | −71.6% | — | 3.00% |
| llama2-7b-b256-n1 | 1666.3 µs | 629.4 µs | −62.2% | — | 3.25% |

- 无回退；剩余受“逐位一致”约束（`T.simd` 会改变 fp32 求和顺序），多核拆 tile 属下一轮候选（未发 DESIGN_LIMIT）。

## 1d. PagedKVCacheWriteFwdOp（已完成，已生产采纳）

- 任务：`examples/paged_cache_write_perf`；改动：逐 lane 元素 scatter 改为 block-staged DMA
  （每连续行 tile 一次 `T.copy`）+ 按元素宽度选 store 路径（16-bit 逐 lane 谓词 / fp8 逐行 DMA）+ 按 tile 选 lane 数；
  wrapper 增加 `pick_config(tokens, heads_kv, dim, physical_tokens, dtype, cache_dtype)` 路由。
- 生产报告：`20261124_211651_753746_PagedKVCacheWriteFwdOp`，`status=passed`，15/15 正确性，逐字节一致。

| case | v1 T_op | v2 T_op | 变化 |
|---|---:|---:|---:|
| qwen3-235b-fp8 | 267.7 µs | **32.5 µs** | **−87.9%** |
| llama-8b-prefill | 884.7 µs | **41.8 µs** | **−95.3%**（1.40 TB/s） |
| qwen3-235b-fp8-fp16 | 78.9 µs | 13.3 µs | −83.1% |
| decode bf16/fp16 | 15.2 / 15.4 µs | 3.1 / 3.0 µs | −80% |

- **口径更新**：模型字节/T_op 可以超过 1.6 TB/s（输入跨 launch 驻 L2），不再判 `invalid`，
  改记 `bandwidth-above-hbm: L2-resident`；`invalid` 只保留给“算力超过峰值”（grid 截断类）。
  该规则已加测试（37 passed）。

## 1e. EngramGateConvFwdOp（已完成，已生产采纳）

- 任务：`examples/engram_gate_conv_perf`；改动：整窗 `(rc+3,d)` 批量 copy 取代行循环内 27 条单行 DMA、
  输出行整块 2-D 落盘、`r_v` 直用去掉每行 `mem_bar`。
- 生产报告：`20261124_222322_139293_EngramGateConvFwdOp`，`status=passed`，13/13。

| case | v1 T_op | v2 T_op | 变化 |
|---|---:|---:|---:|
| train-s96（主导） | 11.0 µs | **8.8 µs** | **−20%** |
| train-s64 | 8.2 µs | 6.6 µs | −20% |
| train-s128 | 7.6 µs | 5.6 µs | −26% |

- 该量级（~9 µs）的 launch 地板实测 2.7 µs，剩余空间有限；未触发 `[DESIGN_LIMIT]`。
- 顺带修了 `make_mfu_table` 的跨算子污染：共享测试文件会让一个算子的报告带出其他算子 case，
  旧逻辑用 `setdefault` 会让旧 case 抢先；现在只接受 `operator` 与报告一致的 case。

## 1f. EngramGateConvBwdOp（已完成，已生产采纳）

- 任务：`examples/engram_gate_conv_bwd_perf`；改动：60 条逐行 DMA → 8 条整窗 2-D copy、reduce kernel 重新分块；
  两次 launch 合并被 API 证据否决。
- 生产报告：`20261124_231932_732777_EngramGateConvBwdOp`，`status=passed`，17/17，逐位不变。

| case | v1 T_op | v2 T_op | 变化 |
|---|---:|---:|---:|
| train-s96（主导，2 launch 求和） | 15.0 µs | **12.4 µs** | **−18.3%** |
| train-s64 | 10.3 µs | 8.8 µs | −14.8% |
| train-s128 | 8.7 µs | 7.1 µs | −18.1% |

- 未达 −20%（launch 地板 ~2.5 µs + reduce launch ~1.3 µs 为硬底），未触发 `[DESIGN_LIMIT]`；
- 边界：整窗暂存使 `d` 较大时 `seq_len` 上界下降（d=768 → 336），manifest/bench 全在包络内。

## 1g. EngramDecodeFwdOp（已完成，已生产采纳）

- 任务：`examples/engram_decode_perf`；改动：cache shift 从“每块对全部 batch 串行 gm2ub→sync→ub2gm”
  改为 batch>n_tiles 时每 batch 行一个块、一条整行窗口 copy。
- 生产报告：`20261125_005026_893589_EngramDecodeFwdOp`，`status=passed`，16/16。

| case | v1 T_op | v2 T_op | 变化 |
|---|---:|---:|---:|
| serving-b8 | 9.5 µs | **6.1 µs** | **−35%** |
| serving-b4（主导） | 8.2 µs | 8.2 µs | 持平（device 码逐字节未变） |
| serving-b1 | 5.7 µs | 5.7 µs | 持平 |
| decode-partial-cache | 8.9 µs | 8.9 µs | 持平 |

- 主导 b4 的 69% 是两次 launch 固定地板（实测 2.7/2.9/2.9 µs），已到设计地板；
  其余 case 的 device 代码与基线逐字节相同（构造性保证不回退）。

## 1h. MoEPermuteAlignFwdOp（已完成，已生产采纳）

- 任务：`examples/moe_permute_align_perf`；改动：scatter 的“每路由一次 GM 原子”→ 可推导槽位
  （每核直方图 + 排他偏移 + UB 游标）、`expert_ids` 改 expert 侧展开、直方图改每路由一次 UB 原子、
  partial 归并整窗 copy（g≥16）、g==1 几何合并 launch。
- 生产报告：`20261125_023838_308793_MoEPermuteAlignFwdOp`，`status=passed`，30/30，35 例逐位一致。

| case | v1 T_op | v2 T_op | 变化 |
|---|---:|---:|---:|
| kimi-k2-t4096（主导） | 650.7 µs | **29.7 µs** | **−95.4%（21.9×）** |
| ds-v3-t4096 | 934.3 µs | 27.7 µs | −97.0% |
| qwen3-235b-t4096 | 1732.5 µs | 27.4 µs | −98.4% |
| t512 组 | 130–250 µs | 16.8–19.2 µs | −86~−93% |
| t32 组 | 38–43 µs | 12.9–14.1 µs | −65~−68% |
| t1 组 | 15–27 µs | 11.6–12.5 µs | −22~−52% |

- 12/12 提升，无回退；GM 原子 ≈1465 cyc/路由 vs UB ≈5 cyc，是本次结构性收益来源。

## 2. SSDChunkCouplingFwdOp（已完成，已生产采纳）

- 任务：`examples/ssd_chunk_coupling_perf`
- 基线：`mamba2-grouped-prefill` 171.0 µs / MFU 30.06%（vector）/ MBU 27.59%；roofline ≈ 51.4 µs。
- 改动：每 tile 一块 → 每 chunk 一个 `(b,c,g)` 行面板（1152→288 块）、C/B 整 panel 一次进 L1、
  双 L0C 累加器交替覆盖 GEMM/AIV epilogue、tile 类分派 epilogue（严格下/上三角/对角分开）、
  上三角零块用 `BL/8` 零缓冲 `dual_copy`。
- 生产报告：`20261124_151020_937387_SSDChunkCouplingFwdOp`，`status=passed`，8/8 正确性。

| case | v1 T_op | v2 T_op | 变化 | v1 MFU | v2 MFU |
|---|---:|---:|---:|---:|---:|
| mamba2-grouped-prefill | 171.0 µs | **83.2 µs** | **−51.3%** | 30.06% | **61.79%** |
| mamba2-2p7b | 22.9 µs | 16.0 µs | −30.0% | 24.9% | 35.61% |
| mamba2-780m | 12.6 µs | 8.5 µs | −32.7% | 22.6% | 33.58% |

- roofline 达成 30.1% → 62.0%；GM 流量 93.0 → 74.6 MB（0.54 → 0.90 TB/s）。
- **`[DESIGN_LIMIT]`**：剩余 ~38% 为设计/带宽地板（L0C/行面板复用上限），已生成
  `perf_opt/perf_feedback.md`。处置：**暂按附录补记归档**（不改写 DESIGN），是否回设计修订待定。
- 已知非本次问题：`SSDChunkScanFwdOp` 的 `PassConfigKey.TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION`
  在 tilelang 0.1.15 不存在，导致 `tests/ops/test_mamba.py` 整文件 4 个 scan 用例失败（A/B 已证与本采纳无关）。

## 3. MLA dense / varlen（排队，需设计修订）

- 当前 MFU：dense 67.01%、varlen 47.05%（vector）；换 Cube 的理论空间约 8×。
- 这不是 Stage 4 参数调优能拿到的，需 Stage 1 设计修订；等 SSD 释放设备后开始。

## 4. 其他算子

- RoPE / Paged / ChainSpec / Engram / MoE：按“双低”诊断（T_op 与 roofline 差 6–2900×）后续再做；
  小 shape 的收益受 launch 地板限制，优先考虑融合/减少 launch。

## 口径提醒

- MFU 分母：Cube 378 / Vector 47（bf16/fp16），目标 90%；实测峰值为 370.5 / 1.424（附表）。
- `T_op` = 一次算子调用内所有 kernel 之和；多 launch 采集必须“一次采集只跑一次调用”。
- 逐 case 数据：`examples/batch2/MFU_ALL.md`、`mfu_cases.csv`；v1 冻结件在 `mfu_baseline_v1.json`。

## 5. 处置记录（2026-11-24）

- **SharedExpert**：接受 89.24%（名义）/ 91.1%（实测）为结果，不再追 0.76 点名义差距；
  生产已采纳，回滚点见 wrapper 末尾的 re-export。
- **SSDChunkCoupling**：`[DESIGN_LIMIT]` 按**归档**处理（保留 61.79%）；`perf_feedback.md` 留档，
  不回设计修订。若后续要做，需换行面板/数据布局，属新任务。
- **其余算子**：按 `rope → chain_spec → paged_cache_write → engram_fwd → engram_bwd → engram_decode → moe`
  顺序连续做 Stage 4；每个完成后生产采纳并刷新报告。
- **MLA dense/varlen**：Stage 4 参数调优空间有限（已在 Vector 上 47–67%），换 Cube 属设计修订，
  待上面 7 个完成后再单独排期。
