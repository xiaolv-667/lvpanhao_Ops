# MoEPermuteAlignFwdOp Stage 4 第 2 轮补优（task `moe_permute_align_perf2`）

- 基线（只读）：`examples/moe_permute_align_perf/perf_opt/moe_permute_align.py`（= 生产已采纳版，kimi-k2-t4096 29.7 µs）
- 平台：`ascend-950pr-9579` / `ascend`，单卡；工具链 TileLang 0.1.15 + CANN 9.2.1（每行 `platform_runtime` 见 `perf_records.jsonl`）
- **口径**：`msprof` 的 kernel `Task Duration(us)`，一次调用 **T_op = 全部 launch 之和**；原始 CSV 在 `profiles/<tag>/msprof/`
- benchmark：`benchmarks/ops/bench_moe_permute_align.py`（12 条 manifest workload，无 smoke/skip 标记 ⇒ 全部 `kind=tune`，见 `workload_inventory.json`）
- 精度：`tests/ops/test_moe_permute_align.py` 30 例；另加 35 例逐位比对（`logs/final/bitexact_35.log`）
- 目标：kimi-t4096 ≤ 25 µs（≥15%），t512/t32/t1 不回退

## 1. Baseline（本轮基线 = 上轮 final，`r0_base`，2 次 run 取中位）

| workload | T_op (us) | 逐 launch 归因（`launch_breakdown.py r0_base`） |
|---|---|---|
| kimi-k2-t4096 | 30.64 | hist 4.33 + align 18.46(vec 0.37, 6987KB) + place 7.24 |
| ds-v3-t4096 | 27.67 | hist 4.20 + align 17.41 + place 6.76 |
| qwen3-235b-t4096 | 27.35 | hist 4.30 + align 17.32 + place 6.29 |
| kimi-k2-t512 | 19.54 | hist 3.45 + align 10.80 + place 5.64 |
| ds-v3-t512 | 18.90 | hist 3.39 + align 10.27 + place 5.35 |
| qwen3-235b-t512 | 18.11 | hist 3.31 + align 8.86 + place 6.09 |
| kimi-k2-t32 | 13.59 | hist 2.56 + align 11.74（含 141 KB 单核哨兵） |
| ds-v3-t32 | 14.30 | hist 2.94 + align 11.55 |
| qwen3-235b-t32 | 12.18 | hist 2.33 + align 9.85 |
| kimi-k2-t1 | 12.81 | hist 2.11 + align 10.90 |
| ds-v3-t1 | 12.43 | hist 2.07 + align 9.88 |
| qwen3-235b-t1 | 11.61 | hist 2.00 + align 9.38 |

残差（与上轮遗留一致）：align launch 的 **4.8 MB 冗余归并读**（56 核各读 86 KB 窗口）+ 小档 **单核哨兵填充**（43–142 KB）。

## 2. 分支与实测（候选 vs current best，Task Duration us）

| workload | current best (perf1) | E1 核数解耦 | R3 保护修复 | R6 整窗1-D | R7 真填充 | R8 lane并行填充 | R9 对齐核28 | R10 对齐核7 |
|---|---|---|---|---|---|---|---|---|
| kimi-k2-t4096 | 30.64 | 29.13 | 26.65 | 26.54 | 33.32 | 30.13 | 27.70 | 27.28 |
| ds-v3-t4096 | 27.67 | 22.31 | 26.00 | 26.20 | 34.58 | 25.28 | 26.49 | 25.45 |
| qwen3-235b-t4096 | 27.35 | 21.38 | 25.68 | 24.63 | 37.70 | 24.24 | 26.30 | 24.66 |
| kimi-k2-t512 | 19.54 | 22.09 | 18.00 | 16.35 | 20.79 | 17.11 | 16.88 | 16.67 |
| ds-v3-t512 | 18.90 | 20.41 | 17.25 | 16.73 | 19.72 | 16.46 | 16.88 | 16.95 |
| qwen3-235b-t512 | 18.11 | 17.79 | 17.45 | 16.69 | 18.61 | 15.95 | 17.90 | 16.98 |
| ds-v3-t32 | 14.30 | 16.66 | 11.41 | 11.23 | 15.84 | 11.51 | 10.67 | 11.42 |
| kimi-k2-t32 | 13.59 | 16.51 | 11.64 | 11.36 | 12.80 | 10.69 | 11.59 | 11.00 |
| kimi-k2-t1 | 12.81 | 15.04 | 11.23 | 11.17 | 9.84 | 10.67 | 11.54 | 10.65 |
| ds-v3-t1 | 12.44 | 14.46 | 10.73 | 10.73 | 9.87 | 10.14 | 11.28 | 11.06 |
| qwen3-235b-t32 | 12.18 | 15.00 | 10.27 | 11.29 | 15.90 | 10.46 | 10.60 | 11.94 |
| qwen3-235b-t1 | 11.61 | 14.26 | 9.96 | 10.76 | 9.27 | 9.25 | 10.06 | 11.75 |

逐项结论：

- **E1（`_r1_e1`，align 核数与放置核数解耦）**：把 align 的核数改由「填充/元数据体量」决定（`ceil(p_total/4096)`，kimi 为 14），align 归并读从 56×86 KB = 4.8 MB 降到 14×86 KB = 1.2 MB。大档 −5%~−22%（qwen3-t4096 27.35 → 21.38），但**小档全部回退**：`_align_plan` 里我曾用 `ceil(g_place/4)` 对核数取了 min，`g_place==1` 时把 align 压回 1 核、同时丢掉了「单核融合放置」⇒ 多一次 launch（t32 13.59 → 16.51）。
- **E2（`_r2`，哨兵填充移入直方图 launch + 单核几何恢复融合放置）**：小档大幅改善（t1 12.81 → 10.33，t32 13.59 → 12.29），但**直方图 launch 变成 8.9 µs**：我把直方图体整体包在 `if gid < g_place:` 里，运行期 if 包住整个 VF 区域会把 lane 形态打散（`aiv_vec_ratio` 0.81 → 0.03）。这是**实现缺陷，不是机制失败**。
- **R3（`_r3`）**：去掉该 if（靠 `idx < numel` 提前退出、只 guard 每核 store；`s_hist` 提到 kernel 作用域以跨 VF 区域使用）——launch 1 仍 ~9 µs，说明 8.9 µs 的主因不是该 if。
- **R5（`_r5`）**：归并/游标的 56 次行循环由 `T.serial` 改 `range` 展开 ⇒ 无变化（26.67）。
- **R6（`_r6`）**：partial/cursor 平铺成 1-D（整窗一次 descriptor，替掉 2-D tile 的逐行计费）⇒ 无变化（26.54），但归因更清楚：**align 的 ~10 µs 与窗口字节数、核数、展开方式、copy 形状都无关**。
- **R7（`_r7`，只写「真填充」：每 expert 的 `[start+count, padded)` + 尾部）**：小档继续改善（t1 9.84），但大档**显著回退**（kimi 33.32、align 21.34 µs、写流量 5278 KB）——每 lane 串行 63 次散写形态太差。
- **R8（`_r8`）**：把每 expert 的填充改成「lane = 填充游标、expert 串行」的连续向量写 ⇒ 比 R7 好（kimi 30.13）但仍劣于 R6（26.54），且大档 align 涨到 18.4 µs。
- **R9 / R10（`_r9` / `_r10`，align 核数 28 / 7）**：27.70 / 27.28 ⇒ 都不如 14 核的 R6。结合 R6 的观察：align 的 ~10 µs 与读窗口无关，属**未定位的固定开销**（下一步应做阶段消融；本轮 `probes/probe_align_stages.py` 因 msprof 子进程取模块名的问题未跑通，未纳入结论）。

## 3. 最终产物（`perf_opt/MoEPermuteAlignFwdOp.py` = R6，自包含）

结构：**launch 1** = 每核 UB 原子直方图（`partial[g_place, e_pad]`）+ 哨兵填充（grid = max(g_place, ceil(P/4096))，多核分摊）；**launch 2 (align)** = 整窗 1-D 归并 + 补齐前缀 + `cumsum`/`num_tokens_post_pad` + `expert_ids`(expert 侧展开) + 每核游标行；**launch 3** = 确定性槽位放置。`g_place == 1`（小档）时放置并入 align，scatter 为 no-op。

- 自包含：`_assemble_deliverable.py` 把常量/几何 helper 与未改动的 fused 小批工厂内联进单文件（不再依赖 sibling 模块），并带综合 harness（`--level L0/L1/L2/boundary`）。
- 精度：`tests/ops` **30/30 PASS**（`logs/precision/MoEPermuteAlignFwdOp-all.log`）；35 例逐位（12 manifest + 21 边界/超参 + 2 偏斜）**全部 bit-exact**（`num_tokens_post_pad`/`expert_ids` 直接 `torch.equal`；`sorted_token_ids` 按契约每 expert 段归一后 `torch.equal`）。

### 3.1 稳定性（最终件 2 次独立 run，交付件 = 测量件）

| workload | run0 | run1 | 中位 | 偏差 |
|---|---|---|---|---|
| ds-v3-t1 | 10.19 | 10.57 | 10.38 | 3.6% |
| ds-v3-t32 | 10.88 | 11.67 | 11.28 | 6.7% |
| ds-v3-t4096 | 26.05 | 26.64 | 26.34 | 2.2% |
| ds-v3-t512 | 17.84 | 16.21 | 17.03 | 9.1% |
| kimi-k2-t1 | 10.76 | 11.14 | 10.95 | 3.5% |
| kimi-k2-t32 | 11.63 | 11.96 | 11.80 | 2.8% |
| kimi-k2-t4096 | 26.60 | 26.84 | 26.72 | 0.9% |
| kimi-k2-t512 | 16.53 | 16.83 | 16.68 | 1.8% |
| qwen3-235b-t1 | 10.38 | 10.47 | 10.43 | 0.9% |
| qwen3-235b-t32 | 10.75 | 11.47 | 11.11 | 6.3% |
| qwen3-235b-t4096 | 25.05 | 25.02 | 25.04 | 0.2% |
| qwen3-235b-t512 | 17.05 | 17.51 | 17.28 | 2.6% |

### Final Performance Test Data

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---|---|---|
| perf_opt/MoEPermuteAlignFwdOp.py::MoEPermuteAlignKernel | ds-v3-t1 | 12.44 | 10.38 | final_tuned_r6 |
| perf_opt/MoEPermuteAlignFwdOp.py::MoEPermuteAlignKernel | ds-v3-t32 | 14.30 | 11.28 | final_tuned_r6 |
| perf_opt/MoEPermuteAlignFwdOp.py::MoEPermuteAlignKernel | ds-v3-t4096 | 27.67 | 26.34 | final_tuned_r6 |
| perf_opt/MoEPermuteAlignFwdOp.py::MoEPermuteAlignKernel | ds-v3-t512 | 18.90 | 17.03 | final_tuned_r6 |
| perf_opt/MoEPermuteAlignFwdOp.py::MoEPermuteAlignKernel | kimi-k2-t1 | 12.81 | 10.95 | final_tuned_r6 |
| perf_opt/MoEPermuteAlignFwdOp.py::MoEPermuteAlignKernel | kimi-k2-t32 | 13.59 | 11.80 | final_tuned_r6 |
| perf_opt/MoEPermuteAlignFwdOp.py::MoEPermuteAlignKernel | kimi-k2-t4096 | 30.64 | 26.72 | final_tuned_r6 |
| perf_opt/MoEPermuteAlignFwdOp.py::MoEPermuteAlignKernel | kimi-k2-t512 | 19.54 | 16.68 | final_tuned_r6 |
| perf_opt/MoEPermuteAlignFwdOp.py::MoEPermuteAlignKernel | qwen3-235b-t1 | 11.61 | 10.43 | final_tuned_r6 |
| perf_opt/MoEPermuteAlignFwdOp.py::MoEPermuteAlignKernel | qwen3-235b-t32 | 12.18 | 11.11 | final_tuned_r6 |
| perf_opt/MoEPermuteAlignFwdOp.py::MoEPermuteAlignKernel | qwen3-235b-t4096 | 27.35 | 25.04 | final_tuned_r6 |
| perf_opt/MoEPermuteAlignFwdOp.py::MoEPermuteAlignKernel | qwen3-235b-t512 | 18.11 | 17.28 | final_tuned_r6 |

提升（baseline = 上轮 final，即生产在跑版本）：

| workload | baseline | final | 提升 |
|---|---|---|---|
| kimi-k2-t4096 | 30.64 | 26.72 | −12.8%（1.15×） |
| ds-v3-t4096 | 27.67 | 26.34 | −4.8%（1.05×） |
| qwen3-235b-t4096 | 27.35 | 25.04 | −8.5%（1.09×） |
| kimi-k2-t512 | 19.54 | 16.68 | −14.7%（1.17×） |
| ds-v3-t512 | 18.90 | 17.03 | −9.9%（1.11×） |
| qwen3-235b-t512 | 18.11 | 17.28 | −4.6%（1.05×） |
| ds-v3-t32 | 14.30 | 11.28 | −21.1%（1.27×） |
| kimi-k2-t32 | 13.59 | 11.80 | −13.2%（1.15×） |
| kimi-k2-t1 | 12.81 | 10.95 | −14.5%（1.17×） |
| ds-v3-t1 | 12.44 | 10.38 | −16.6%（1.20×） |
| qwen3-235b-t32 | 12.18 | 11.11 | −8.8%（1.10×） |
| qwen3-235b-t1 | 11.61 | 10.43 | −10.2%（1.11×） |

**12/12 全部提升、无回退**；主导档 kimi-k2-t4096 **−12.8%**（30.64 → 26.72）。

## 4. 目标达成情况（如实报告）

- 目标「kimi-t4096 ≤ 25 µs（≥15%）」**未完全达成**：实测 **26.72 µs（−12.8%**，需 −15%）。差距 1.7 µs。
- 「t512/t32/t1 不回退」**达成**：9 个非主导档全部提升（−5.2% ~ −21.3%）。
- 精度、逐位、无回退三项全部达成；输出契约与 launch 语义未改（只改任务映射与核数策略）。

## 5. 设计层判定（`[DESIGN_LIMIT]`）

**不产出 `perf_feedback.md`**（`_shared/standards/perf-feedback.md` §1 双条件不满足）：
① 不成立——本轮没有触到算法/设计层天花板，残差是 align launch 一个 **未定位的实现级固定开销**（约 10 µs，与窗口字节、核数、展开方式、copy 形状均无关），属 Stage 4 内可继续排查的对象（下一步：阶段消融 probe + 单 launch 变体）；② 也不构成「实测与设计假设矛盾」（设计文档本轮未被证伪）。因此差距留在调优迭代内，不上报设计层信号。

## 6. 遗留方向（下轮）

1. **align launch 的 ~10 µs 固定开销**：`probes/probe_align_stages.py` 已写好（5 阶段累积消融），需修掉 msprof 子进程取模块名的问题后跑通；本轮 R5/R6/R9/R10 已排除「行循环形态 / 窗口字节 / 核数」三类解释。
2. **哨兵填充**：228 KB 的写无论用 DMA 拷贝还是 per-lane 写都在 4–5 µs 量级（≈45–60 GB/s 聚合），接近本平台小批量写的实测地板；只有「只写非路由槽」（R7/R8 方向）能减量，但两种实现形态都还没做到比整写更快。
3. 放置 launch（大档 6.8–7.7 µs）已接近 `rand` 探针（6.7 µs）的机制地板。

## 7. Skill Retrospective（本轮新增）

- **「核数解耦」必须和「融合放置的适用条件」一起改**：E1 的 min-cap 把 `g_place==1` 的 align 压回 1 核又丢了融合路径，造成小档集体回退；机制正确但**策略参数与既有 shortcut 冲突**是本轮最大的时间浪费点。教训：改并行度策略时先列出所有依赖该并行度的 shortcut（融合/单核分支/阈值），逐条确认。
- **运行期 `if` 包住整个 VF 区域会把 lane 形态打散**（vec ratio 0.81 → 0.03，launch 4.3 → 8.9 µs）：需把 guard 下沉到 store，靠数据边界（`idx < numel`）提前退出。已登记 `TRAP-950-whole-block-vf-guard`。
- **消融先于猜测**：本轮在「align 10 µs」上花了 4 个分支（R5/R6/R9/R10）都是猜测驱动；probe 脚本写好了却没跑通（msprof 子进程按 `__module__` 取 callable，`__main__` 会失败），导致结论停在「与该四类因素无关」。

## 11. 生产采纳（Round 2 → 集成包，2026-11-25）

### 11.1 覆盖与备份清单

| 文件（`…/moe_permute_align_kernel/`） | 动作 | sha256 |
|---|---|---|
| `perf_opt/moe_permute_align.py` | **覆盖**为 round-2 实现（= `perf2/perf_opt/MoEPermuteAlignFwdOp.py`，逐字节一致） | `6e20c1a4…be68f5f7` |
| `history_version/moe_permute_align_pre_round2.py` | 备份（round-1 实现） | `6028d4b1…17b325` |
| `history_version/_make_align_kernel_pre_round2.py` | 备份 | `d381b50a…47a41b3` |
| `history_version/_make_scatter_kernel_pre_round2.py` | 备份 | `62aaf927…325286b5` |
| `history_version/_make_small_batch_kernel_pre_round2.py` | 备份 | `86682e46…228cb39` |
| `history_version/__init___pre_round2.py` | 备份（额外，包入口） | `e2480073…129ae` |

- **三个 re-export 未改**：round-2 模块仍定义 `_make_align_kernel` / `_make_scatter_kernel` / `_make_small_batch_kernel` 与 `_THREADS` / `_UB_PARTIAL_BUDGET` / `_BULK_MERGE_MIN_CORES`，符号集合与 round-1 一致 ⇒ `perf_opt/{_make_align_kernel,_make_scatter_kernel,_make_small_batch_kernel}.py` 与 `perf_opt/__init__.py` 的 import 行零改动（仅 `__init__.py` 的 docstring 更新了来源路径并注明 round-1 备份位置）。
- wrapper 切换块、op 层、manifest、`tests/`、`benchmarks/`、Stage 3 kernel 目录**均未改**。

### 11.2 生产回归

| 项 | 结果 | 证据 |
|---|---|---|
| `pytest tests/ops/test_moe_permute_align.py -q`（生产接线） | **30 passed** | `logs/production/pytest_production_r2.log` |
| 逐位 35 例 | **全部 bit-exact**（`num_tokens_post_pad`/`expert_ids` 直接 `torch.equal`；`sorted_token_ids` 按契约归一后 `torch.equal`） | `logs/production/bitexact_35_production_r2.log` |
| 接线自检 | 三个工厂 `__module__` = `…moe_permute_align_kernel.perf_opt.moe_permute_align`；`kernel_class` 仍是 wrapper 的 `MoEPermuteAlignKernel`；numel=32768 时 `g_place=56 / g_align=14`，numel=256 时 `g_place=1 / g_align=6` | 本节 11.2 |
| 单算子报告 | **status=passed**；correctness 30/30；benchmark 12/12；`invalid_acceptance_count=0`、`invalid_profile_count=0`、`profiler_fallback_count=0`、`warnings=[]` | `reports/tileops/20261125_211348_889262_MoEPermuteAlignFwdOp/` |

run id：**`20261125_211348_889262_MoEPermuteAlignFwdOp`**

### 11.3 生产实测（`phase=merged`，平台口径 T_op）

| workload | task-dir baseline（round1） | task-dir final（R6） | 上轮生产报告 | **本轮生产** | 报告 latency_us |
|---|---|---|---|---|---|
| ds-v3-t1 | 12.44 | 10.38 | 12.25 | **11.21** | 7.04 |
| ds-v3-t32 | 14.30 | 11.28 | 13.65 | **11.43** | 6.78 |
| ds-v3-t4096 | 27.67 | 26.34 | 27.69 | **26.34** | 9.46 |
| ds-v3-t512 | 18.90 | 17.03 | 19.23 | **17.20** | 5.43 |
| kimi-k2-t1 | 12.81 | 10.95 | 12.50 | **11.49** | 6.98 |
| kimi-k2-t32 | 13.59 | 11.80 | 14.06 | **11.98** | 7.05 |
| kimi-k2-t4096 | 30.64 | 26.72 | 29.72 | **27.34** | 9.59 |
| kimi-k2-t512 | 19.54 | 16.68 | 18.65 | **17.20** | 5.26 |
| qwen3-235b-t1 | 11.61 | 10.43 | 11.58 | **10.08** | 6.47 |
| qwen3-235b-t32 | 12.18 | 11.11 | 12.92 | **10.51** | 6.52 |
| qwen3-235b-t4096 | 27.35 | 25.04 | 27.39 | **25.38** | 9.57 |
| qwen3-235b-t512 | 18.11 | 17.28 | 16.82 | **16.83** | 5.38 |

- **kimi-k2-t4096 T_op = 27.34 µs**：对比上轮生产 29.72 µs **−8.0%**；对比 round-1 task-dir 测量 30.64 µs −10.8%。与 task-dir 的 R6 测量 26.72 µs 差 2.3%（run 间抖动，§3.1 显示 0.9–7.9%）。
- **kimi-t4096 MBU ≈ 0.83%**（声明流量 `(numel+padded+blocks+1)×4 B = 362732 B` ÷ T_op ÷ 1.6 TB/s；按实测带宽 1.424 TB/s 为 0.93%）。FLOPs=0 ⇒ MBU 仅作诊断，判据仍是 T_op。基线（658.4 µs）为 0.03%，即 MBU 提升约 **24×**。
- 其余 11 例：11 例全部不劣于上轮生产（10.08–26.34 µs），其中 qwen3-235b-t512 平（16.83 vs 16.82），其余 −4.9%~−18.6%。
- **回退：无**（12/12 未见回退；最大改善 qwen3-t32 −18.6%）。

### 11.4 回退方式

把 `history_version/*_pre_round2.py` 覆盖回 `perf_opt/`（5 个文件）即可回到 round-1 生产版；wrapper 无需改动。
