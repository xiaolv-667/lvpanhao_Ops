# 本批 11 个算子迁移与优化可行性方案

更新：2026-11-23。配套文档：`MIGRATION_PLAN_batch2.md`（逐算子现状与用例）、`MIGRATION_LOOP.md`（每轮执行流程）。

## 0. 总体判断

| 目标 | 可行性 | 依据 |
|---|---|---|
| 功能迁移：11 个算子在本机 NPU 上跑通各自 manifest 的全部 case | **可行，把握较大** | 6 个已有 Ascend 实现；3 个接口现成、只缺 kernel；2 个 MLA 反向在 main 里 spec/Op/接口/测试齐全，可拷 |
| 在本机判定「性能达标」 | **当前不可行** | 本机是 28 核 950PR 样片、无 `npu-guard` 与 clock-config，数字属开发口径；MFU 达标判定需 canonical 32 核 950DT |
| 性能优化本身（定位并消掉瓶颈） | **可行，但结论要分档** | 瓶颈可用 roofline 差额 + 分阶段计时定位；"是否算达标"不能在本机下结论 |
| 在现有硬件上做出可复现、可比较的性能记录 | **可行** | 同设备、同轮、同协议自建基线即可相对比较（`--verify-only` 出正确性，性能另测） |

**一句话结论：先做"功能全绿"，再做"本机相对性能"，最后才谈"达标"。三步判据完全不同，混在一起就会得出假的结论。**

## 1. 可行性分级（逐算子）

"轮次"= 一个「改 → 测 → 记录」循环，是工程量估算，不是承诺。

| # | 算子 | 起点 | 功能可行性 | 主要风险 | 性能天花板判断 | 估算轮次 |
|---:|---|---|---|---|---|---|
| 1 | PagedKVCacheWriteFwdOp | B 只缺 kernel | **高** | 无输出只能比对 mutated；`slot_mapping` int64；FP8 饱和 cast 且是除以 scale；`PS=7/NP=5` 奇数页 | 带宽/散射型，理论上可接近峰值带宽 | 2–3 |
| 2 | MoEPermuteAlignFwdOp | B 只缺 kernel | **高** | 整型 histogram + 跨 expert 前缀和 + scatter；无效 id 桶参与 padding；`num_experts` 128→384 动态 | `flops=0`，无 MFU 概念，纯时延 | 2–3 |
| 3 | SharedExpertMLPFwdOp | B 只缺 kernel | **中高** | 两段 GEMM；T=4096/S=18432 时 `[T,2S]` 中间量约 288 MiB；`S=18432` 的 N 尾块 | Cube 型，T≥512 有较大空间；T=1/32 受限 | 3–5 |
| 4 | EngramGateConvFwdOp | A 已有实现 | **高** | 6 个输出含 3 个 fp32 中间量；`refusal` 限 `d ≤ 4096` | 小 shape，latency 型，MFU 无意义 | 1–2（验收） |
| 5 | EngramGateConvBwdOp | A 已有实现 | **高** | fp32 权重梯度跨 token 归约顺序影响末位；3 个权重梯度都要断言 | 同上 | 1–2（验收） |
| 6 | EngramDecodeFwdOp | A 已有实现 | **高** | 必须同时比对 `y_t` 与 `new_conv_state`；两次 launch 都要计入计时 | 纯 latency（batch 1–8） | 1–2（验收） |
| 7 | RoPEFwdOp | A 已有实现 | **高** | 无 `refusal`（含 2 个 fp32 用例）；牵动 Llama31/YaRN/LongRoPE 等 5 个算子 | 带宽型，看有效带宽 | 1–2（验收） |
| 8 | SSDChunkCouplingFwdOp | A 已有实现 | **高** | 因果掩码方向（保留 `s ≤ l` 下三角）；可并行维度小（G 1–8、NC 8–16） | Cube 型，受 grid 规模限制 | 2–3（验收+调优） |
| 9 | ChainSpeculativeSamplingFwdOp | A 已有实现 | **中** | 必须固定 seed 才能逐位比对；V 到 15 万，带宽主导；与 SamplingFromProbs 共用 `_DrawKernel` | 带宽型，roofline 是下界 | 2–3（验收+调优） |
| 10 | MLADenseBwdOp | C 先移植 | **中** | 移植 6 处（含 roofline 路径 `_perf`→`perf`）；`D_qk=192` 非 2 的幂；bottom-right 对齐；NPU 无 flash-attn 参考 | 计算密集，天花板最高，但也是本批最难 | 6–10 |
| 11 | MLAVarlenBwdOp | C 先移植 | **中低** | 同 10，外加 `cu_seqlens` 偏移写回、请求长度 512→16384 的负载不均 | 同 10 | 6–10 |

### 1.1 三条起点的可行性差异

- **A 类（6 个，已有实现）**：不确定性最低。真正的未知只有两个 —— "这活是否已经有人验收过"、"现有实现距 roofline 多远"。属于**验收 + 定向调优**，不是重写。
- **B 类（3 个，只缺 kernel）**：接口、用例、参考实现全现成，属**确定性工作**，风险集中在各自的数值/边界细节（见上表）。
- **C 类（2 个 MLA 反向）**：唯一需要"先补规格"的一类，且有**参考实现缺失**这个结构性风险。建议放在最后，且不承诺性能目标。

### 1.2 性能目标的现实性

按 manifest roofline 的口径，11 个算子可分三类，**不能用同一把尺子量**：

| 类型 | 算子 | 该看什么 | 不该看什么 |
|---|---|---|---|
| Cube/计算密集 | SharedExpertMLPFwd、SSDChunkCouplingFwd、MLADenseBwd、MLAVarlenBwd | MFU 与 roofline 差额 | — |
| 带宽型 | RoPEFwd、PagedKVCacheWriteFwd、ChainSpeculativeSamplingFwd | 有效带宽 / roofline bytes | MFU（flops 太小或无意义） |
| 时延型（小 shape） | 三个 Engram、MoEPermuteAlignFwd | 绝对时延、launch 次数 | MFU（MFU 只有百分之零点几属正常） |

把时延型算子按 "MFU > 90%" 去考核，只会得出错误结论 —— 这一点必须在验收口径里先写清楚。

## 2. 关键风险与对策

| # | 风险 | 影响 | 概率 | 对策 |
|---|---|---|---|---|
| R1 | 拿不到 canonical 口径（无 `npu-guard`、无 clock-config、非 32 核 950DT） | 高：性能"达标"无法判定 | 已发生 | 报告分两栏：**开发口径数字** 与 **待 canonical 复核**；本阶段只承诺"相对基线与 roofline 趋势"，不承诺达标 |
| R2 | 参考实现依赖 `float64`，NPU 无 `DT_DOUBLE` | 高：正确性反馈直接失效 | 中 | 参考实现放 **CPU** 算（CPU 支持 double），只把结果拷到 NPU 比对；或改用 fp32 分块参考 |
| R3 | MLA 反向的 `ref_api` 是 flash-attn，NPU 上不存在 | 高：C 类整体受阻 | 已确定 | 自建参考：先用 CPU double 复算 `_flash_attn_backward` 语义，验证前向一致后再验反向；不引入 flash-attn 依赖 |
| R4 | 编译不稳定（`bishengir` SIGSEGV / 超时 / 属性名问题） | 中：迭代变慢、假失败 | 中 | 用现成 shim（`TILEOPS_SYNC_ATTR_SHIM=1` + `-p tileops_shim` + `TILEOPS_COMPILE_RETRIES/TIMEOUT`），并把 `.shims-a5` 前置到 `PATH`；每个实验独立 `TILELANG_CACHE_DIR` |
| R5 | 改共享 kernel 文件波及已验收算子 | 中：静默破坏 | 中 | 每轮照 `MIGRATION_LOOP.md` §5 的回归范围表全跑，不看上次记录 |
| R6 | 卡资源：撞卡或分到 Alarm/性能异常卡 | 中：性能数字被污染 | 中 | 固定卡；跑前确认卡状态；性能记录里写明卡号 |
| R7 | 把时延型算子按 MFU 考核 | 中：结论错误 | 中 | 按 §1.2 的分类分别设判据 |
| R8 | 为"跑绿"而改断言/加 skip/放宽容差 | 高：验收失效 | 低 | 明确列为禁止手段（见 `MIGRATION_LOOP.md` §9） |

## 3. 阶段计划

判据只要不满足，就不进下一阶段。

| 阶段 | 内容 | 产出 | 退出判据 | 估算 |
|---|---|---|---|---|
| **P0 准备** | 环境自检（venv / shim / a5 / cache）、盘点 6 个已有实现的归属与状态、确认卡 | 环境自检记录 + 归属结论 | 能稳定跑起任一已有算子的 NPU 测试；卡已锁定 | 0.5–1 轮 |
| **P1 功能补齐** | 3 个 B 类 kernel：建议顺序 PagedKVCacheWrite → MoEPermuteAlign → SharedExpertMLP（由易到难） | 3 个 `kernels/ascend/<family>/*.py` + 包登记 | 三者 manifest 全 case passed | 7–11 轮 |
| **P2 MLA 移植** | main → 分支移植 spec/Op/接口/workload/测试，同步改 roofline 路径与命名；确定 NPU 侧参考方案 | 可 import 的 Op + 能跑的参考 | 参考实现能在 NPU 上产出可比结果（R3 解决） | 3–5 轮 |
| **P3 MLA kernel** | 先 Dense 后 Varlen；`D_qk=192`、bottom-right 对齐、`lse` 自然对数、varlen 偏移写回 | 2 个 Ascend kernel | 全 case passed | 6–10 轮 |
| **P4 性能基线** | 11 个算子各出一份同设备同轮记录，算 roofline 差额并排序 | 一张"差额 vs 实现档次"的表 | 能指名"哪几个值得调、为什么" | 3–5 轮 |
| **P5 定向优化** | 按 P4 排序取差额最大的 3–5 个动刀（优先 Cube 型） | 优化前后对比 | 差额缩小且回归全绿 | 视 P4 结论 |
| **P6 canonical 验收** | 等 clock-config 与 32 核 950DT，按 `ascend950-profiler-locked-v1` 重测 | 正式 receipt | `receipt.json` 为 `passed` 且 `verify_only=false` | 依赖硬件 |

**建议先做 P0 + P1 的第一个算子**：一轮就能把流程问题全部暴露，比先啃 MLA 划算得多。

## 4. 资源需求

| 项 | 需求 | 说明 |
|---|---|---|
| NPU | 至少 1 张固定卡，专用 | 撞卡会污染性能；本机 0/1/3 曾 Alarm、5 号性能异常、7 号不存在 |
| 环境 | `xiangmu/.venvs/ascend950` + `.shims-py` + `.shims-a5` | 已在库里，无需新建 |
| 磁盘 | 每个实验独立编译缓存；profiler raw trace 需预留 | raw trace 体积可能很大，别在评审前删 |
| canonical 设备 | 32 核 950DT + 可用 clock-config + host npu-guard | 仅 P6 需要；没有则 P6 挂起 |

## 5. 降级与停止条件（什么时候承认"这个目标当前做不到"）

1. **硬件口径长期拿不到** → 性能目标降级为"相对基线与 roofline 趋势"，**不再声称达标**，报告里明确写"待 canonical 复核"。
2. **单算子连续 3 轮无进展** → 停止盲试，出分析报告：查了什么、证据、两个假设、下一步建议。
3. **MLA 参考实现始终建不起来** → C 类停在"已移植 + 未验证"，不进入 P5 优化，也不计入性能结论。
4. **本机出现无法解释的 device error（如 507035）且非对齐问题** → 先退到最小 shape 复现，再决定是否降级该算子。

## 6. 立即可做的三件事

1. **环境自检**：激活 venv、`pip install -e . --no-deps`、前置 shim，跑通任一已有算子的 NPU 测试。
2. **盘点 6 个已有实现**：谁提交的、是否已验收、`refusal` 边界各是什么 —— 这决定 A 类是"调优"还是"补验收"。
3. **样板轮**：用 `PagedKVCacheWriteFwdOp` 跑通「写 kernel → 登记 → NPU 测试 → 记性能」整圈（命令见 `MIGRATION_LOOP.md` §10）。

## 7. 需要你决策的四项

| # | 决策 | 影响 |
|---|---|---|
| 1 | 本批固定用哪张卡 | 决定性能记录是否可信、是否与其它子任务冲突 |
| 2 | 6 个已有实现的归属与验收状态 | 决定 P4/P5 是"调优"还是"先补验收" |
| 3 | MLA 参考实现路线（CPU double vs NPU fp32 分块） | 决定 P2 能否按期退出 |
| 4 | 是否接受把判据拆成"功能 / 相对性能 / 达标"三档 | 决定本阶段能承诺什么、报告怎么写 |

---
本文件是可行性判断与计划，不含代码改动。逐算子用例与现状见 `MIGRATION_PLAN_batch2.md`，每轮执行细节见 `MIGRATION_LOOP.md`。上一批的历史文档已按你的要求删除，本文件不依赖它们。
