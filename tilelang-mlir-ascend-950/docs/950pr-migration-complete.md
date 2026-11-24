# 950PR 算子迁移完成状态（本仓库）

> 范围：11 个算子在当前仓库（`tilelang-mlir-ascend-950`）内完成迁移、集成、真机正确性与 MFU 统计。
> 交付仓 `TileOPs-ascend950` **不在本阶段范围**（相关试点已停止）。
> 日期：2026-11-24。硬件：Ascend950PR_9579（28 Cube / 56 Vector，128 GiB）。

## 1. 算子清单与落点

每个算子两部分：`examples/{slug}/`（设计/内核/复盘 + 迁移状态）与
`examples/TileOPs/tileops/kernels/{family}/{slug}/`（集成包：wrapper + kernel + 测试入口）。

| # | 算子 | family | 聚合状态 | 集成 | 最新报告 |
|--:|---|---|---|---|---|
| 1 | PagedKVCacheWriteFwdOp | attention | DONE | pass | `reports/tileops/*PagedKVCacheWriteFwdOp` |
| 2 | RoPEFwdOp | positional | DONE | pass | `*RoPEFwdOp` |
| 3 | MLADenseBwdOp | attention | DONE | pass | `*MLADenseBwdOp` |
| 4 | MLAVarlenBwdOp | attention | DONE | pass（2026-11-24 重集成） | `20261124_070330_693136_MLAVarlenBwdOp` |
| 5 | SSDChunkCouplingFwdOp | mamba | DONE | pass | `*SSDChunkCouplingFwdOp` |
| 6 | EngramGateConvFwdOp | sequence_modeling | DONE | pass | `*EngramGateConvFwdOp` |
| 7 | EngramGateConvBwdOp | sequence_modeling | DONE | pass | `*EngramGateConvBwdOp` |
| 8 | EngramDecodeFwdOp | sequence_modeling | DONE | pass | `*EngramDecodeFwdOp` |
| 9 | SharedExpertMLPFwdOp | moe | DONE | pass | `*SharedExpertMLPFwdOp` |
| 10 | ChainSpeculativeSamplingFwdOp | sampling | DONE | pass | `*ChainSpeculativeSamplingFwdOp` |
| 11 | MoEPermuteAlignFwdOp | moe | DONE | pass | `*MoEPermuteAlignFwdOp` |

## 2. 正确性

### 2.1 集成正确性（小 shape，真机）

11/11 通过，合计 **281 个用例**（Engram 三算子共享 `tests/ops/test_engram.py`，按报告计 46×3）：

| 算子 | 用例 | 通过 |
|---|---:|---:|
| PagedKVCacheWrite | 15 | 15 |
| RoPE | 34 | 34 |
| MLADenseBwd | 10 | 10 |
| MLAVarlenBwd | 12 | 12 |
| SSDChunkCoupling | 8 | 8 |
| EngramGateConvFwd | 46 | 46 |
| EngramGateConvBwd | 46 | 46 |
| EngramDecode | 46 | 46 |
| SharedExpertMLP | 25 | 25 |
| ChainSpecSampling | 9 | 9 |
| MoEPermuteAlign | 30 | 30 |

### 2.2 主导 workload 正确性（`examples/batch2/WORKLOAD_CORRECTNESS.md`）

13 个 case，全部在主导 shape 上真机比对：

- **10 个 pass**：Engram×3、MoE（2 case，逐位）、PagedKV（逐位）、RoPE、SSD、ChainSpec、SharedExpert；
- **3 个 MLA 反向 fail（1 ulp 尾部）**：dense / varlen mixed / varlen 8x4k，
  超差 14 / 36 / 44 个元素（1.3e-8–2.0e-8 比例），全部 `k_ulp ≤ 1.0`；
  - 切片自洽（dk/dv 逐位、dq ≤9.8e-4、无死 head、blocks=useful）证明**不是静默少算**；
  - 按 DESIGN §8.1 golden（P/dS 进 GEMM 前舍入）参考后 **0 超差**；
  - 待定：按 `k_ulp ≤ 1` 判 pass，或严格记 fail（见 §6）。

### 2.3 本次修复的关键正确性 bug

`MLAVarlenBwdOp` `_bwd_kernel` grid = 66,560 > 65,535 → 16 位 blockDim 静默只跑 1,024 个 CTA。
修复：head-slab 4×16,640 + `h_off` + 尾 slab 守卫 + 启动前后断言。修复后 8x4k 为
**490.6 ms / 29.2 TFLOPS**（原伪值 20.8 ms / 690 TFLOPS）。

## 3. 性能（MFU，按执行单元）

口径：`MFU = FLOPs / (T_op × 单元峰值)`；Cube 378 / Vector 47（bf16/fp16），目标 90%。
完整 69 行见 `examples/batch2/MFU_ALL.md` 与 `mfu_cases.csv`。

| 算子 | 单元 | 主导 case | AI | T_op | MFU | 上限 | MFU max |
|---|---|---|---|---:|---:|---:|---:|---:|
| MLADenseBwd | vector | ds-v3-b2-8k | 2666.7 | 455.3 ms | **67.01%** | 100% | 67.05% |
| MLAVarlenBwd | vector | ds-v3-mixed | 3611.0 | 1728.7 ms | **47.05%** | 100% | 64.84% |
| SharedExpertMLP | cube | 512×7168×18432 | 502.8 | 2147.0 µs | **50.02%** | 100% | 50.80% |
| SSDChunkCoupling | vector | grouped-prefill | 32.0 | 171.0 µs | **30.06%** | 100% | 30.06% |
| EngramGateConvBwd | vector | train-s96 | 4.3 | 15.0 µs | **2.15%** | 14.5% | 2.15% |
| EngramDecode | vector | serving-b4 | 3.7 | 8.7 µs | **2.06%** | 12.6% | 2.06% |
| EngramGateConvFwd | vector | train-s96 | 2.9 | 11.0 µs | **1.23%** | 9.8% | 1.23% |
| RoPE | vector | llama3-8b-b2-2k | 0.4 | 671.1 µs | **0.41%** | 2.6% | 0.41% |
| PagedKVCacheWrite | vector | qwen3-235b-fp8 | 0.7 | 267.7 µs | **0.07%** | 2.3% | 0.07% |
| ChainSpecSampling | vector | ds-v3-mtp-n3 | 0.7 | 2542.8 µs | **0.04%** | 5.2% | 0.06% |
| MoEPermuteAlign | vector | kimi-k2-t4096 | 0.0 | 643.6 µs | n/a（FLOPs=0） | — | — |

无算子达到 90% 目标；最接近的是 MLA dense（67%）、SharedExpert（50%）。

2026-11-24 复核：6 个原 `partial` 报告已按新采集规则重跑，全部 `status=passed`、`invalid_profile_count=0`、`invalid_acceptance_count=0`、fallback=0；11/11 报告通过。
MLA 反向在 Vector 上跑到 47–65%，真问题是要把矩阵乘迁回 Cube（理论 ~8×），不是 tiling。

## 4. 本次框架改动（都在本仓库）

- MFU 为唯一验收指标，按执行单元选分母（`compute_units.json`），MU 仅诊断；
- `T_op` 为整算子所有 kernel 之和；`invalid`（>峰值）直接判 benchmark 失败；
- 多 kernel 单次调用在带 `t_op_us` 时视为有效采集；
- 报告列：`Latency | T_op | Acceptance | ... | Achieved`，逐 case 与算子级表；
- 删除旧 roofline ratio 链路（`msprof_roofline.py` 等），manifest roofline 只作 FLOPs/Bytes 来源。

## 5. 产物索引

| 产物 | 路径 |
|---|---|
| 逐 case 数据 | `examples/batch2/mfu_cases.csv`（69 行） |
| 算子级数据 | `examples/batch2/mfu_operators.csv`（11 行） |
| 全量表 | `examples/batch2/MFU_ALL.md` |
| 主导 shape 正确性 | `examples/batch2/WORKLOAD_CORRECTNESS.md` + `.json` |
| 验证脚本 | `examples/batch2/verify_workload_shapes.py` |
| 验收口径 | `docs/950pr-mfu-acceptance.md` |
| varlen 修复证据 | `examples/mla_varlen_bwd/_bwd_kernel/DESIGN.md`、`REPRO`、`integration_log.md` |

## 6. 未完成 / 待定

1. **MLA 1-ulp 尾部定性**：按 `k_ulp ≤ 1` 判 pass，还是严格记 fail；不影响 kernel。
2. **Stage 4 性能优化**：按约定本批不做；MLA→Cube、SharedExpert、SSD 是三个主要方向。
3. **交付仓**：本轮不处理；已停止相关试点。
4. **Vector 峰值实测标定**：目前用厂商标称 47 TFLOPS（Cube 已标定 370.5）。
5. **fp32/fp8 Cube 峰值**：按 bf16 比例折算，标 `estimated`。
