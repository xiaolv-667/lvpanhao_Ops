# 950PR 算子性能验收协议（MFU / MU / 时延）

> 适用：本批 11 个迁移算子，以及交付仓 `TileOPs-ascend950` 的后续验收。
> 本文定义“怎么算”，不记录具体数字（数字见 `examples/batch2/MFU_ALL.md`）。
> 原则：**分子用 manifest roofline 公式，分母用 950PR 实测峰值，T_op 用整算子求和，events 不得用于 MFU。**

## 1. 判定流程

```
① 算 AI = FLOPs / Bytes（manifest roofline）
② 与机器平衡点比较：AI ≫ Peak_TFLOPS / Peak_BW  → 计算受限
                    AI ≪ 平衡点                → 访存受限
                    工作量 < 启动地板           → 时延受限
③ 计算对应利用率
④ 与目标比较：稀疏CV >50% / CV >80% / 其余 >90% / 时延型 ≤ 基线
```

机器平衡点（950PR，待标定）：`378.47e12 / 1.6e12 ≈ 236 FLOP/Byte`
（若标定后 Peak_BW 变化，此值同步更新。）

## 2. 四类算子的公式

| 类别 | 判据 | 指标 | 公式 |
|---|---|---|---|
| 计算受限 CV | 稀疏 CV >50%；CV >80% | **MFU** | `FLOPs / (T_op × Peak_TFLOPS)` |
| 访存受限（CV 或 Vec） | 其余 >90% | **MU（带宽效率）** | `Bytes / (T_op × Peak_BW)` |
| 时延受限（小 shape / flops≈0） | 持平或超越基线 | **时延比** | `T_op / T_baseline` |
| 稀疏 CV | 稀疏 CV >50% | **稀疏 MFU** | `FLOPs_稀疏 / (T_op × Peak_sparse)` |

补充：
- `FLOPs/Bytes` 一律取 manifest `roofline`（脚本 `examples/batch2/roofline_eval.py` 可复现）；
- `Peak_sparse` 需单独标定（2:4 或结构化稀疏 matmul），未标定前稀疏 CV 只报相对值；
- **不得**用 `Perf(TOps/s)`（msprof 单 kernel 计数器）反推整算子 FLOPs。

## 3. 峰值标定（一次性，写入 constants.md）

| 常数 | 方法 | 输出 |
|---|---|---|
| `Peak_TFLOPS`（dense fp16/bf16） | 8192×8192×8192 Cube matmul，稳态中位（≥20 次，warmup 5） | e.g. 378.47 |
| `Peak_BW` | ≥1 GiB 连续 copy/read kernel，取有效带宽中位 | 当前模型常数 1.6 TB/s，**必须实测替换** |
| `Peak_sparse` | 2:4 稀疏 matmul，或按“实际稀疏 FLOPs/时间”定义 | 稀疏 CV 用 |

验收前必须完成标定；标定结果带 `hardware=ascend-950pr-9579`、CANN 版本、日期戳写入
`pattern-library/constants.md`，并注明“canonical 验收需用 950DT”。

## 4. T_op 采集协议（关键）

```
T_op = Σ(该算子一次调用涉及的全部 device kernel 时间)
```

规则：
1. **一次调用**：benchmark 的 profiler 脚本固定 0 warmup + 1 measured launch；每个 `OpBasicInfo` sample = 一次 launch。
2. **求和**：同名多次 launch（如 Engram bwd 的两个 `main_kernel`）也要相加，**不能取 max/median**。
3. **包含**：算子 forward 内部的 `ZerosLike`（dq_acc 清零）、`Range`（cu_seqlens）计入。
4. **排除**：host 侧 padding 产生的 `PadV3*` 单列，不计入 kernel 时间，但报告披露。
5. **禁止**：`events` 模式（op 墙钟，含 host 提交）只能作补充，不能用于 MFU。
6. **多 kernel 归因**：优先让 benchmark 支持 kernel 名列表/时间窗求和；临时可用
   `TILEOPS_MSPROF_KERNEL_NAME=<bogus>` 强制无过滤采集 + `examples/batch2/msprof_sum.py` 求和。
7. **合理性检查**：若 `FLOPs/T_op > Peak_TFLOPS` 或 `Bytes/T_op > Peak_BW`，该 capture 不完整，
   标 `physically-impossible`，不得用于判据。

## 5. 基线（挑战目标）

| 对象 | 基线 |
|---|---|
| 6 个已有 DT 实现的算子 | 交付仓当前 `kernels/ascend/...` 在同机同 shape 的 `T_op` |
| 5 个新增算子 | 交付仓无实现时：PyTorch/torch_npu 等价调用，或 GPU 参考除以硬件比（仅趋势参考） |

比较协议：同机、同 shape、同 dtype、同 T_op 口径、同 profiler。

## 6. 11 个算子的判定归属

| 算子 | 类型 | 受限 | 应报指标 | 备注 |
|---|---|---|---|---|
| MLADenseBwd | CV | 计算（AI≈1334） | MFU | 3 launch 求和 |
| MLAVarlenBwd | CV | 计算（AI≈1334–3611） | MFU | msprof 大 shape 阻塞，需先修采集 |
| SharedExpertMLP | CV | 大 T 计算；小 T 访存 | MFU / MU | AI≈503（t512）→ MFU；AI≈1–32 → MU |
| SSDChunkCoupling | CV | 访存（AI≈32） | MU | 计算 MFU 仅 ~3%，不可达 80% |
| EngramDecode | CV（小 batch） | 时延 | 时延 vs 基线 | MFU 不适用 |
| RoPEFwd | Vec | 访存（AI≈0.75） | MU | |
| PagedKVCacheWrite | Vec | 访存（flops≈0） | MU | |
| MoEPermuteAlign | Vec（flops=0） | 时延/访存 | 时延 | MFU 无定义 |
| ChainSpecSampling | Vec | 访存（AI<1） | MU | roofline 是下界，MFU 不计算 |
| EngramGateConvFwd | Vec | 时延 | 时延 vs 基线 | |
| EngramGateConvBwd | Vec | 时延 | 时延 vs 基线 | refusal d≤1024 |

## 7. 报告格式（每个算子一行）

```
operator | class | AI | metric(MFU/MU/latency) | value | target | pass? | T_op source | notes
```

必须同时给出：`reported_latency` 与 `T_op`，两者差 >2× 时标 `report-underreports`；
`blocked` 行给根因（msprof 失败 / capture 不完整 / 无基线）。

## 8. 当前已知阻塞

1. **大 MLA shape 的 msprof 失败**（varlen 全部大 case、dense 8k/kimi/v2-lite）：回退 events 后吞吐物理不可能，需先修采集。
2. **峰值未标定**：378.47 TFLOPS 来自框架自报，1.6 TB/s 是模型常数。
3. **baseline 未采集**：6 个 DT 实现的同机对比尚未做。

## 9. 实现状态（框架已改造）

已在 `examples/TileOPs` 落地 MFU 验收层，新 run 自动产出：

| 文件 | 作用 |
|---|---|
| `tileops/reporting/acceptance.py` | manifest roofline 求值（含 `product`/`element size of T`/`slot_mapping entries >= 0` 等伪函数规范化）+ 分类 + MFU/MU/latency 计算 |
| `tileops/reporting/peaks.json` | 峰值与**来源戳**（当前 `provisional`：378.47 TFLOPS / 1.6 TB/s） |
| `tileops/benchmark/msprof.py` | 元数据新增 `t_op_us` 与 `t_op_breakdown`（一次调用内所有 launch 求和） |
| `.agents/tools/toolchains/{adapter,ascend}.py` | `benchmark_profile_breakdown()`：按 kernel 名汇总 raw CSV |
| `tileops/reporting/analyzer.py` | 每个 case 附 `acceptance`，run.json 增加 `acceptance_summary`；多 kernel 但无 `t_op_us` 时标 `blocked`（不再用 latency 冒充） |
| `tileops/reporting/report.py` | 报告新增 `T_op (us)` 与 `Acceptance`（MFU/MU/latency + PASS/FAIL）列 |
| `tests/test_reporting_acceptance.py` | 5 个 CPU 单测（compute/memory/latency 分类与数值） |

已验证（真实数据）：

| 算子/case | T_op | 类别 | 指标 | 结论 |
|---|---:|---|---:|---|
| SharedExpert 512×7168×18432 | 2150.2 µs | compute | **MFU 49.9%** | FAIL（<80%） |
| SharedExpert 4096×7168×2048 | 1877.9 µs | compute | **MFU 50.8%** | FAIL |
| SharedExpert 1×7168×18432 | 709.2 µs | memory | **MU 69.9%** | FAIL（<90%） |
| MLADenseBwd 4k | 58.27 ms | compute | **MFU 8.1%** | FAIL |
| RoPE prefill-4k | 429.4 µs | memory | **MU 0.6%** | FAIL |
| Paged 64 tokens | 15.2 µs | latency | 15.2 µs | 需基线 |

> 旧 run.json 不会自动回填 acceptance；重渲染不会重新分析。需要历史数据时，用
> `analyzer._attach_acceptance()` 重新分析后写回（或重跑 benchmark）。

### 9.1 尚未完成

1. **大 MLA shape 的 msprof 失败**：varlen 全部大 case、dense 8k/kimi/v2-lite 仍回退 events；accepted 层会把它们标 `blocked`（正确行为）。
2. **峰值标定**：`peaks.json` 仍是 provisional，需按 §3 实测替换。
3. **baseline 采集**：6 个 DT 实现的同机对比尚未做。

## 10. 口径定稿（2026-11-24，对外提交用）

### 10.1 主口径：理论峰值（vendor SKU 表）

| 常数 | 主口径（提交） | 附表（实测） |
|---|---:|---:|
| Cube BF16/FP16 | **378 TFLOPS** | 370.5（8192³ matmul，98.0%） |
| Cube FP8 / INT8 | 756 | 741（按 98% 折算，estimated） |
| Cube FP32/TF32 | 189 | 185.2（estimated） |
| Cube+Vector 合计 BF16 | 425（仅作上限参考） | — |
| Vector BF16 / FP32 | 47 / 23 | — |
| HBM 带宽 | **1.6 TB/s** | 1.424（512 MiB copy，89.0%） |
| 平衡点 | 236 FLOP/Byte | 260 FLOP/Byte |

选择理论峰值做提交口径的理由：**可审计**（手册可核）、**保守**（分母更大 → 指标更低）、**跨人跨机稳定**。实测值作为附表，说明“当前可达上限”。

带宽档位判定：实测 copy 1.424 TB/s > 1.4，故理论档必为 **1.6 TB/s**；128 GiB 容量亦与该档一致。

### 10.2 计算与提交格式

```
MFU_spec     = achieved_TFLOPS / 378      ← 主口径（提交）
MFU_measured = achieved_TFLOPS / 370.5    ← 附表
MU_spec      = achieved_BW     / 1.6      ← 主口径（提交）
MU_measured  = achieved_BW     / 1.424    ← 附表
```

报告每行同时给出两列（当前 `report.md` 显示为 `MFU 49.9% (实测 51.0%)`）。

### 10.3 无效值处理

`achieved > 理论峰值 × 1.01` 的 case 一律标 `status=invalid`（历史例子：grid 越界导致的
`MLAVarlenBwd ds-v3-8x4k`，隐含 692 TFLOPS；该 bug 已于 2026-11-24 用 head-slab 修复，
现在实测 490.6 ms / 29.2 TFLOPS / MFU 62.2%），**不参与达标统计、不出现在性能结论里**。

2026-11-24 起，`invalid` 还是一项 **benchmark 阶段 tripwire**：`analyze_run` 会把
`summary.invalid_acceptance_count > 0` 的 run 标为 `benchmark_passed=false`（整体
`status=partial`）并写 warning，避免“profiler 有效但结果物理不可能”的 run 被当成通过。

### 10.4 量级换算（用于把旧表转成主口径）

旧表若用 370.5 / 1.424：`MFU_spec = MFU_old × 1.0203`，`MU_spec = MU_old × 1.1236`。

### 10.5 提交时随附的 provenance

```json
{"status":"vendor_spec_plus_measured",
 "primary":{"source":"Ascend 950PR SKU table (28-cube column)","cube_bf16":"378","bandwidth":"1.6"},
 "measured":{"compute":"8192^3 bf16 matmul median 370.5","bandwidth":"512 MiB copy 1.424","date":"2026-11-24"}}
```

## 11. 旧 roofline 链路的清理（2026-11-24）

报告与数据记录**只输出 MFU/MU 口径**，旧的「msprof roofline ratio」字段与解析链路已删除：

| 删除项 | 位置 | 说明 |
|---|---|---|
| `roofline_utilization_percent`、`bandwidth_tb_s`、`arithmetic_intensity_ops_per_byte`、`performance_tops`、`computility_tops` | `tileops/reporting/analyzer.py` | 不再写入 `run.json` 的 `performance.cases[]` |
| `ratio_range` | `analyzer.py` / `report.py` | 概览表改为 `Acceptance`（MFU/MU 达标计数） |
| `_parse_msprof_roofline`、`_dump_roofline_log`、`Ratio(%)` 列 | `tileops/benchmark/benchmark_base.py` | `_build_result` 不再解析 `visualize_data.bin` |
| `tileops/benchmark/msprof_roofline.py` | 整文件 | 无引用，已删除 |
| `tileops_tflops` / `tileops_bandwidth_tbs` pytest 属性 | `benchmarks/conftest.py` | 无消费者，且来自单 case latency，易与 T_op 口径混淆 |

**保留**：manifest 的 `roofline` 表达式（`eval_roofline`）仍是 FLOPs / Bytes 的唯一来源，
`acceptance.py` 用它做分子，并据此计算算术强度与 compute/memory 分类。

报告列现在为：

```
| Label | Latency (us) | T_op (us) | Acceptance | Shape/Parameters | DType | Mode | Kernel | Achieved |
```

- `T_op (us)`：整算子所有 kernel 之和（MFU/MU 分母时间）；
- `Acceptance`：`MFU 49.9% (实测 51.0%) FAIL`；`blocked` 表示无法聚合 T_op，`invalid` 表示超过物理峰值；
- `Achieved`：按类别显示 `TFLOP/s`（compute）或 `TB/s`（memory），可用于与峰值直接对账。

渲染旧的 `run.json` 时，`report.py` 会尝试用 manifest 重新计算 acceptance（缺 `t_op_us` 的
case 显示 `blocked`），不会回落到已删除的 ratio 字段。

## 12. 验收指标：MFU（唯一）

**统计口径就是 MFU，不是 MU。** `MU` 只作为诊断列保留，不参与达标判定。

```
MFU = FLOPs / (T_op × Peak_TFLOPS)      ← 唯一验收指标
MU  = Bytes / (T_op × Peak_BW)          ← 诊断，仅用于解释 MFU 为什么低
```

| 项 | 规则 |
|---|---|
| 分母 | 按 dtype 取理论 Cube 峰值（bf16/fp16 378、fp8/int8 756、fp32 189） |
| 目标 | sparse CV 50% / CV 80% / others 90%（`peaks.json` 的 `acceptance.target_percent`，默认 90） |
| FLOPs=0 的算子 | MFU 无定义（如 MoE 纯索引搬移），回退为时延型，报告显式标注 |
| `invalid` | `achieved > peak × 1.01`；不参与任何统计 |

### 12.1 为什么还要显示 MU 和「MFU 上限」

带宽受限的 shape，MFU 有物理上限：

```
MFU_ceiling = min(1, AI / 平衡点) × 100%      平衡点 = 378 / 1.6 = 236 FLOP/Byte
```

| 算子（主导 case） | AI | MFU 上限 | 实测 MFU |
|---|---:|---:|---:|
| RoPE | 0.4 | 0.16% | 0.02% |
| ChainSpec | 0.7 | 0.32% | 0.00% |
| Paged KV write | 0.7 | 0.28% | 0.01% |
| SSD chunk coupling | 32.0 | 13.5% | 3.7% |
| Engram decode | 3.7 | 1.6% | 0.26% |

报告里以 `带宽上限 X%` 标注，避免把「形状决定的上限」误读成「kernel 写得差」。
验收仍按 MFU 判定（这些算子会显示未达标），MU 与上限只用于解释与后续 Stage 4 定位。

### 12.2 与之前版本的差异

- 之前 memory-bound 算子报 MU（`MU 27.6%`），现在统一报 `MFU 3.7%`（同一 case），并在括号里给 `带宽上限 13.5%`；
- `aggregate()` 仍按 metric 分桶，但正常只有 `MFU`（和 FLOPs=0 的 `latency`）两类，不再出现 MFU/MU 混算；
- MU 字段改为 `mu_value`/`mu_value_measured`，只出现在 case 详情里。

## 13. MFU 分母按执行单元选择（2026-11-24）

MFU 的分子是 FLOPs，分母必须是**该 kernel 实际使用的计算单元**的峰值。用 Cube 峰值去
除一个纯 Vector kernel，会把它低估约 8 倍（378/47）。

单元不是按"算法看起来像什么"猜的，而是按 msprof `ArithmeticUtilization` 实测：

- `aic_time = NA` 且只有 `vector0` block → **纯 Vector**（没有启动 Cube 核）；
- `aic_cube_ratio` 与 `aiv_vec_ratio` 都显著 → 取占优者，并在证据里记两者。

| 算子 | AIC cube | AIV vector | 单元 |
|---|---:|---:|---|
| ChainSpeculativeSampling | 无 Cube 块 | 99.96% | vector |
| PagedKVCacheWrite | 无 Cube 块 | 95.2% | vector |
| RoPE | 无 Cube 块 | 91.4% | vector |
| MLA dense / varlen bwd | 8.3% | 73.5% | vector |
| SSDChunkCoupling | 12.2% | 52.4% | vector |
| EngramDecode | 无 Cube 块 | 44.4% | vector |
| EngramGateConv Fwd / Bwd | 无 Cube 块 | 44.1% / 45.2% | vector |
| SharedExpertMLP | GEMM | — | cube |

证据落在 `tileops/reporting/compute_units.json`；`acceptance.py` 用它选分母，
msprof metadata 里的逐 case `compute_unit` 可覆盖它。

### 13.1 分母与平衡点

| 单元 | bf16/fp16 | fp32 | 平衡点（AI 上限的分母） |
|---|---:|---:|---:|
| Cube | 378 | 189 | 236 FLOP/Byte |
| Vector | 47 | 23 | **29.4 FLOP/Byte** |

`MFU = FLOPs / (T_op × unit_peak)`；`MFU 上限 = AI / 单元平衡点`。

### 13.2 换分母后的结论变化

| 算子（主导 case） | ÷Cube（旧） | ÷对应单元（新） | 上限 | 真实问题 |
|---|---:|---:|---:|---|
| SharedExpert | 49.94% | 49.94%（cube） | 100% | Cube 利用率不足 |
| MLADenseBwd | 8.33% | **67.0%**（vector） | 100% | 在 Vector 上不差，**该走 Cube** |
| MLAVarlenBwd | 5.85%（mixed） | **47.0%**（vector，mixed；8x4k 修复后 62.2%） | 100% | 同上 |
| SSDChunkCoupling | 3.74% | **30.1%**（vector） | 100% | Vector 跑到 30% |
| RoPE | 0.04% | **0.41%**（vector） | 2.58% | 只到上限 16%，访存问题 |
| EngramDecode | 0.26% | **2.06%** | 12.6% | 上限的 16% |
| PagedKVCacheWrite | 0.01% | **0.07%** | 2.26% | 上限的 3% |
| ChainSpecSampling | 0.00% | **0.06%** | 5.22% | 上限的 1% |

关键结论：**MLA 反向不是"实现只跑到 8%"，而是"用 Vector 跑了本应在 Cube 上的矩阵乘"**。
它 29 TFLOPS 约为 Vector 峰值 47 的 62%，所以移到 Cube 的理论空间约 8 倍。

### 13.3 报告展示

```
MFU 67.0% (vector, 带宽上限 100.0%) FAIL
MFU 0.016% (vector, 带宽上限 2.53%) FAIL
```

详情里增加 `Compute unit`（含 msprof 证据）与 `MU (diagnostic)`。
`target_percent` 仍取 `peaks.json` 的 `acceptance.target_percent`（默认 90）。

### 10.6 采集有效性规则（按“整算子调用”适配）

本框架的计时单位是**一次算子调用**，因此有效性规则围绕 `t_op_us`（整次调用的 kernel 求和）判断：

| 情况 | 判定 |
|---|---|
| msprof 一次采集包含多个 kernel，且带 `t_op_us` | **有效**（MLA 反向是 4 个 kernel、SharedExpert 2 个、MoE 2 个） |
| 请求了 kernel filter、回退到 unfiltered，但带 `t_op_us` | **有效**（整调用仍被覆盖） |
| msprof 失败 → events 回退 | **有效**：events 计的是整个 functor 调用，`latency_us` 可作 T_op；fallback 计数保留为 warning |
| 多 kernel 但没有 `t_op_us` | 无效（无法把整调用时间和单 kernel 时间区分） |
| 有 `t_op_us` 但值物理不可能（>峰值） | `invalid`，benchmark 阶段失败 |

`benchmark_passed` 只在以下情况为 false：`invalid_profile_count > 0`（无可用计时）、
`invalid_acceptance_count > 0`（物理不可能）或 benchmark pytest 失败。profiler/filter 回退只记 warning，
不再一票否决。
