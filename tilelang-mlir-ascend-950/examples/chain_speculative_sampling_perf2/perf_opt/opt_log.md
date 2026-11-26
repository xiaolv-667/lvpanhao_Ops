# ChainSpeculativeSamplingFwdOp — Stage 4 Round 2 调优日志（多核拆行/拆 tile）

- 算子 / kernel：`ChainSpeculativeSamplingFwdOp` / `_chain_speculative_sampling_kernel`
- kernel_id：`examples/chain_speculative_sampling_perf/perf_opt/chain_speculative_sampling.py::_chain_speculative_sampling_kernel::main`
- **本轮基线（只读）**：`examples/chain_speculative_sampling_perf/perf_opt/chain_speculative_sampling.py`（Round 1 达标版本，sha256 `5b2b5a78…`，生产 T_op n3 = 714.6 µs）
- 本轮产物：`perf_opt/ChainSpeculativeSamplingFwdOp.py`（sha256 `5da0811bcb4d7808…`），同工厂签名 `(batch, num_draft, vocab) -> callable`、同 7 实参调用契约、`out_idx=[]`
- 平台：`ascend-950pr-9579` / `ascend`，CANN 9.2.1，TileLang ascend 0.1.15，单卡（56 Vector core），`TMPDIR=/workspace`
- 测量口径：**msprof（`msopprof --aic-metrics=Roofline`）的 kernel Task Duration(us)**（本轮一次调用是 3 段 launch，三个 `@T.prim_func` 都命名 `main` → msprof 的 `Op Name` 统一为 `main_kernel`，T_op 按平台适配器的 `benchmark_profile_breakdown` 逐 op 名求和 = 三段之和；报告 `run.json` 的 `acceptance.t_op_us` 即此值，而 `Latency` 列是三次 launch 的 median，采纳验收须看 `t_op_us`），`--launch-count 15 --warm-up 5`；T_op = 一次调用全部 launch 求和（本轮每个 case 3 个 kernel）。原始证据：`perf_opt/profiles/**`（字段 `msprof_raw_path`），逐测量记录：`perf_opt/perf_records.jsonl`。文中的 wall-clock 仅用于分支粗筛（已标注）。

## 1. Performance Test Data（baseline = Round 1 达标版本，round 0）

| kernel_id | workload_id | candidate_id | target_kernel_name | captured_op_name | task_duration_us | profile_status | raw_profile_dir |
|---|---|---|---|---|---:|---|---|
| `_chain_speculative_sampling_kernel` | llama-8b-b16-n4-float32 | baseline_round1_tuned | `main_kernel` | `main_kernel` | 339.511 | valid | `perf_opt/profiles/round0/base/llama-8b-b16-n4` |
| `_chain_speculative_sampling_kernel` | llama-8b-b64-n4-float32 | baseline_round1_tuned | `main_kernel` | `main_kernel` | 696.646 | valid | `perf_opt/profiles/round0/base/llama-8b-b64-n4` |
| `_chain_speculative_sampling_kernel` | ds-v3-mtp-n1-float32 | baseline_round1_tuned | `main_kernel` | `main_kernel` | 657.434 | valid | `perf_opt/profiles/round0/base/ds-v3-mtp-n1` |
| `_chain_speculative_sampling_kernel` | ds-v3-mtp-n3-float32 | baseline_round1_tuned | `main_kernel` | `main_kernel` | 711.449 | valid | `perf_opt/profiles/round0/base/ds-v3-mtp-n3` |
| `_chain_speculative_sampling_kernel` | qwen3-235b-eagle-n5-float32 | baseline_round1_tuned | `main_kernel` | `main_kernel` | 404.502 | valid | `perf_opt/profiles/round0/base/qwen3-235b-eagle-n5` |
| `_chain_speculative_sampling_kernel` | llama-8b-b17-n8-float32 | baseline_round1_tuned | `main_kernel` | `main_kernel` | 349.964 | valid | `perf_opt/profiles/round0/base/llama-8b-b17-n8` |
| `_chain_speculative_sampling_kernel` | llama2-7b-b256-n1-float32 | baseline_round1_tuned | `main_kernel` | `main_kernel` | 659.042 | valid | `perf_opt/profiles/round0/base/llama2-7b-b256-n1` |

复测与生产报告 `20261124_190245_979904` 一致（338.8 / 702.2 / 661.4 / 714.6 / 405.7 / 347.9 / 629.4），偏差 ≤1.1%（`llama2-7b-b256-n1` 偏低 4.7%，长尾散布）。**workload inventory**：benchmark 展开 8 例 = 1 smoke + 7 tune(`perf_opt/workload_inventory.json`)；smoke 只做精度回归，无 skipped。

## 2. Iteration Log

### Round 1 — 现象与候选点

**现象（Round 1 达标版本的 profile 分解，n3 711.4 µs）**：fold 占 81%，其内层每核吞吐约 4.9 GB/s（2 数组 × 8192 元素/tile），而 `perf_opt/probe_bw.py` 已实测「逐元素读 + reducer 每核 9.5–10.0 GB/s，且与 grid 无关（16/28/56 核分别 10.0/9.9/9.5）」→ fold 的**每核成本固定**，唯一空间是把 `B×ntile` 个 tile 任务均摊到 56 核：n3 的 `per_core=2` 使关键路径 32 tile 而均值 18.3 tile（1.75x 上界），`B<56` 的 b16/b17 更甚（1 核 1 行 = 16 tile，仅用 16/17 核）。

**结构选择**：`拆分 = 3 段 launch`，因为链测试（Philox）必须产出 `n_s` 才能定 fold 的行基址，逐 tile-group 复制链测试不可行；而 Roofline 口径是「一次调用全部 launch 求和」，3 段是合法的 T_op 组成。

| 段 | grid | 内容 |
|---|---|---|
| `chain_kernel` | `min(B,56)` | Philox4x32-10 + `u*draft < target` → `n_s[B]`、`u_N[B]` 写 workspace |
| `fold_kernel` | `B*parts` | 一个 block 一个 (row, tile-group)，`chunk` 个 tile 的残差和 → `tile_sum[B*ntile]` |
| `locate_kernel` | `min(B,56)` | 读 `tile_sum` 行 → 粗层 tile 前缀 → 命中 tile 重算 → lane 和 → 元素走查 → 写出 |

**逐位一致性论证**：tile 的累加分组只由「tile 内元素→lane 映射 + 块内 `AscendAllReduce` 树 + 排序 tile 前缀」决定，与前缀由哪个 block 拥有该 tile 无关（fold 生成码实测 `t*8192 + i*128 + threadIdx.x`，与 Round 1 逐字相同）；`n_s` / `u_N` / `tile_sum` 经 GM workspace 传递不改数值；因此被抽 token 与 `num_accepted` 逐位不变。

**候选分支（同一 base 派生，msprof，dominant `ds-v3-mtp-n3`）**

| 分支 | 主要优化点 | T_op µs | vs baseline | L0 | 结论 |
|---|---|---:|---:|---|---|
| b0 | current best（融合版，Round 1） | 711.4 | 1.00x | PASS | 基准 |
| b1 | `split=k3, parts=4`（tile 拆 4 组） | 380.0 | 1.87x | PASS | 有效 |
| b2 | `split=k3, parts=auto(8)` | **356.7** | **1.99x** | PASS | **采纳** |
| b3 | `split=k3, parts=8, ctl=shared`（locate 状态留在 UB 标量） | 394.4 | 1.80x | PASS | rollback（−9%） |

**候选 vs current best（Round 1 对比表：平台时延 / 利用率 / 访存 / L0）**

| workload | candidate | T_op µs | vs best | 有效带宽 GB/s | MBU % | 关键 per-op µs | L0 |
|---|---|---:|---:|---:|---:|---|---|
| ds-v3-mtp-n3 | current best（融合） | 711.4 | 1.00x | 46.5 | 2.91 | — | PASS |
| ds-v3-mtp-n3 | b1 parts=4 | 380.0 | 1.87x | 87.1 | 5.44 | fold 276 / locate 90 | PASS |
| ds-v3-mtp-n3 | **b2 parts=8** | **356.7** | **1.99x** | 92.8 | 5.80 | chain 13.8 / fold 256.0 / locate 85.3 | PASS |
| ds-v3-mtp-n3 | b3 ctl=shared | 394.4 | 1.80x | 83.9 | 5.24 | locate 121 | PASS |

### Round 2 — 现象、候选点、分支

**现象（b2 的 profile）**：locate 段 85.3 µs（24%），wall 消融显示它**每行约 47 µs**，而其中约 36 µs 来自「128 步 lane 前缀 + 64 步元素走查 + 19 步粗层前缀」用 **UB 标量** 表达时编译器在每次写后插入 `asc_syncthreads()`（`probe` 生成码实测每迭代 2 个 barrier）。

| 分支 | 主要优化点 | 结果（msprof / wall） | L0 | 结论 |
|---|---|---:|---|---|
| r2-a | locate 的前缀/走查状态全部改**寄存器局部**（`ctl=local`） | msprof 394.4 → **356.7**；wall locate 93.3 → 55.0（n3）、49.4 → 28.5（b16） | PASS | **采纳** |
| r2-b | 命中 tile 重算拼写 branch / peel（compile-time tile 下标）/ flat（运行期基址） | 三者 locate 均 ≈93 µs（wall）→ 非热点；保留 branch（flat 命中 `TRAP-950-dynamic-base-tile-pass`） | PASS | 保持 branch |
| r2-c | `parts` 取 ntile 的**因子**（避免运行期 tile 守卫；`T.Parallel` 区域在守卫为假时仍付入场成本） | qwen3：parts=8(wall 318) / 4(365) / 2(484) / **1(189)**；fused 222 → qwen3 取 parts=1 | PASS | 采纳因子规则 |
| r2-d | fold 的 `t < ntile` 守卫在 `parts*chunk == ntile` 时**消除**（compile-time 恒真） | n3 与 b16 的 fold 随之使用无守卫路径 | PASS | 采纳 |
| r2-e | 修正 auto 规则：`B*ntile < 112` 或 `B*parts > 65535`（设备 grid 上限）时回退融合版 | L1-7 `(65536,2,16)` 曾因 fold grid 65536 > 65535 触发 **507035 vector core exception**；修复后 L1 7/7 | PASS | 采纳（防御） |

**候选 vs current best（Round 2 对比表）**

| workload | candidate | T_op µs | vs best | 有效带宽 GB/s | MBU % | 关键 per-op µs | L0 |
|---|---|---:|---:|---:|---:|---|---|
| ds-v3-mtp-n3 | current best（b2） | 356.7 | 1.00x | 92.8 | 5.80 | 13.8 / 256.0 / 85.3 | PASS |
| ds-v3-mtp-n3 | r2-a（ctl=local） | **356.7** | 1.00x（同候选，locate 85.3 → 55 wall） | 92.8 | 5.80 | 13.8 / 256.0 / 85.3 | PASS |
| ds-v3-mtp-n3 | r2-b peel / flat | 358–393 | ≈1.00x | — | — | locate 不变 | PASS |
| qwen3 | r2-c parts=1 | 377.4 | 1.07x（vs 融合 404.5） | 51.5 | 3.22 | 12.3 / 318.4 / 46.7 | PASS |

**合并非回退检查**：最终版对全部 7 个 tune workload 复测（§Final）均优于 baseline，无回退；smoke `chain-sampling-smoke` 走融合路径（`B*ntile=2 < 112`），精度通过。

**平区裁决（<5%）**：n3 的 parts=4 vs parts=8 差 6.5%（382.8 vs 357.9）> 阈值，直接采纳；`ctl` 的 9% 同理。b16 的 parts=8 vs 16（wall 97.6 vs 105.1，7%）用交错多 run A/B（`perf_opt/ab_parts.py`，4 run，run 间离散 <1%）裁决取 8。

**终止条件**：success —— 主导 `ds-v3-mtp-n3` **711.4 → 356.7 µs（1.99x）**，满足本轮「≤500 µs（1.43x）」目标；全部 tune workload 不回退。

## 3. Autotune Log

未使用 autotune。搜索空间只有 `parts`（tile 组数）与 `fold/walk/ctl` 拼写；`parts` 按「ntile 的因子 + block 数 ≤512」的实测拐点规则编译期选定（见 `_split_parts` 注释中的实测数字），不引入运行期搜索。tile 大小、threads、ub 布局、归约分组仍由逐位一致约束冻结。

## Final Performance Test Data

最终候选 `perf_opt_split_k3`（= `perf_opt/ChainSpeculativeSamplingFwdOp.py`，sha256 `5da0811bcb4d7808…`），全部 workload 在同一采集配置（`--launch-count 15 --warm-up 5`，msprof）下重新测得：

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---:|---:|---|
| examples/chain_speculative_sampling_perf/perf_opt/chain_speculative_sampling.py::_chain_speculative_sampling_kernel::main | llama-8b-b16-n4-float32 | 339.511 | 141.230 | perf_opt_split_k3 |
| examples/chain_speculative_sampling_perf/perf_opt/chain_speculative_sampling.py::_chain_speculative_sampling_kernel::main | llama-8b-b64-n4-float32 | 696.646 | 393.134 | perf_opt_split_k3 |
| examples/chain_speculative_sampling_perf/perf_opt/chain_speculative_sampling.py::_chain_speculative_sampling_kernel::main | ds-v3-mtp-n1-float32 | 657.434 | 308.731 | perf_opt_split_k3 |
| examples/chain_speculative_sampling_perf/perf_opt/chain_speculative_sampling.py::_chain_speculative_sampling_kernel::main | ds-v3-mtp-n3-float32 | 711.449 | 356.676 | perf_opt_split_k3 |
| examples/chain_speculative_sampling_perf/perf_opt/chain_speculative_sampling.py::_chain_speculative_sampling_kernel::main | qwen3-235b-eagle-n5-float32 | 404.502 | 377.383 | perf_opt_split_k3 |
| examples/chain_speculative_sampling_perf/perf_opt/chain_speculative_sampling.py::_chain_speculative_sampling_kernel::main | llama-8b-b17-n8-float32 | 349.964 | 152.670 | perf_opt_split_k3 |
| examples/chain_speculative_sampling_perf/perf_opt/chain_speculative_sampling.py::_chain_speculative_sampling_kernel::main | llama2-7b-b256-n1-float32 | 659.042 | 402.606 | perf_opt_split_k3 |

### Final 利用率与带宽（manifest roofline bytes 折算，非时延口径）

| workload | bytes MB | speedup | final 有效带宽 GB/s | final MBU % | 计划（parts / blocks） |
|---|---:|---:|---:|---:|---|
| llama-8b-b16-n4 | 8.21 | 2.40x | 58.1 | 3.63 | 8 / 128 |
| llama-8b-b64-n4 | 32.84 | 1.77x | 83.5 | 5.22 | 8 / 512 |
| ds-v3-mtp-n1 | 33.10 | 2.13x | 107.2 | 6.70 | 8 / 512 |
| **ds-v3-mtp-n3** | 33.10 | **1.99x** | 92.8 | **5.80** | 8 / 512 |
| qwen3-235b-eagle-n5 | 19.45 | 1.07x | 51.5 | 3.22 | 1 / 32 |
| llama-8b-b17-n8 | 8.72 | 2.29x | 57.1 | 3.57 | 8 / 136 |
| llama2-7b-b256-n1 | 32.77 | 1.64x | 81.4 | 5.09 | 2 / 512 |

### Final 精度回归

- 候选自带分层精度（本文件）：**L0 13/13、L1 7/7、boundary 7/7**，`num_accepted_max_diff=0`、`drawn_token_mismatches=0`（`perf_opt/logs/round1/`）。
- 逐位对拍 `perf_opt/verify_exact.py`（本轮候选 vs Round 1 达标 kernel）：**26/26 BIT-EXACT**（`output_token_ids` 与 `num_accepted` 全 0 差异；含 benchmark 7 shape、L0/L1/边界 shape、4 种控制流、`storage_offset=1`、奇数 vocab、1-token vocab）。
- smoke `chain-sampling-smoke`：走融合路径，精度通过（`precision_pass: true`）；无 skipped。

## 剩余空间（如实记录，未采纳）

1. **locate 段仍是第二大项**（n3 85 µs / b256-n1 194 µs，占 24%/48%）。b256-n1 的 `ntile=4` 使「命中 tile 重算」占整行 25%，lane 和又要扫 8192 个 UB 元素；要再降只能让 fold 段顺带产出每 tile 的 lane 和（需把残差落 UB，实测该存写往返在 Round 1 里是 2.4x 成本）→ 收益不确定，未做。
2. **fold 段每核 4.9 GB/s** vs `probe_bw` 单数组 9.5 GB/s：差在「2 数组 + `max` + 归约」的指令数，而 `float2` 向量读会改变元素→lane 映射即改变求和分组 → 逐位一致约束下不可用（Round 1 已记录）。
3. qwen3（`ntile=19` 质数）只能取 parts=1，本轮相对融合版仅 1.07x；若允许改 tile 大小（8192→非 2 的幂）可让 19 变为可整除，但那会改求和分组 → 不可行。

## Skill Retrospective

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| 测量学 | 同一 kernel 的 wall-clock 在**不同进程间**差异可达 1.4x（n3 fused：403 / 518 / 569 µs 三个进程），而**同进程交错 A/B 的 run 间离散 <1%**。跨进程只能用 msprof 定案 | 本文件 §2 的各次 fused 读数；`ab_parts.py` 的 runs 列表 | `profile-collection.md` 增「跨进程 wall 不可比，分支排序须同进程交错；最终值用平台口径」 | P |
| 平台约束 | 拆多 block 后 grid 上限（65535）成为**正确性**问题而非性能问题：`B*parts` 在 65536 行的分布探针上越界，报的是 `vector core exception 507035`（与 `TRAP-950-grid-over-65535` 同族但错误码相同于 UB 对齐故障，易误诊） | L1-7 `(65536,2,16)` 崩溃与修复 | `traps-950pr-execution.md` 的 TRAP-950-grid-over-65535 补一条「同一 507035 码也由 grid 越界触发，先查 grid 再查 UB」 | D |
| 编译器行为 | **`T.Parallel` 区域在运行期守卫为假时仍付入场成本**（~5 µs/次）：fold 的 `if t < ntile` 与 locate 的 `if t == pick` 同族；因此「按 tile 分组」必须让分组数整除 tile 数 | qwen3 parts=8（24 slot / 19 tile）fold 262 µs vs parts=1 142 µs | `iteration-diagnosis.md` 增「区域入场成本」判据 | P |

### 价值点提案（vp_type + 证据三件套）

| title | vp_type | evidence | repro | toolchain_stamp |
|---|---|---|---|---|
| **P：三 kernel 拆分（chain｜fold×parts｜locate）+ 惰性 workspace 保持只读 wrapper 契约**：把「一行一核」的 fold 拆成「一个 block 一个 (row, tile-group)」，tile 和经 GM workspace 传给 locate；工厂返回闭包，首次调用按 device 惰性分配 workspace，故 7 实参契约不变、T_op 仍是一次调用全部 launch 之和。实测 n3 711→358 µs（1.99x）、b16 339→142（2.40x）、b256 659→407（1.62x） | P | `perf_records.jsonl` round1（`split_parts4` 382.8 / `perf_opt_split_k3` 357.9 vs baseline 711.4）；per-op 分解 chain 13.6 / fold 260.0 / locate 85.3 | `perf_opt/ChainSpeculativeSamplingFwdOp.py`（`_SplitKernel` + `_build_chain/_build_fold/_build_locate`） | TileLang ascend 0.1.15 / CANN 9.2.1 / Ascend950PR_9579 / 2026-11-25 |
| **D：locate 的「前缀/走查用 UB 标量」每步带 barrier，128 步 lane 前缀 ≈36 µs/行**：改成寄存器局部后 locate 93.3→55.0 µs（n3 wall）、49.4→28.5（b16），T_op 393→358 | D | `ab_kernels`/`ctl` A/B（`perf_opt/logs`），msprof `split_ctl_shared` 393.0 vs final 357.9 | `perf_opt/ChainSpeculativeSamplingFwdOp.py` 的 `ctl_mode`（"shared" vs "local" 两臂） | 同上 |
| **D：每核逐元素吞吐与 grid 无关（16/28/56 核 → 10.0/9.9/9.5 GB/s/核）**，故「拆核」收益 ≈ 关键路径 tile 数之比；这条常数用于本轮 `parts` 决策与上界估计 | D | `perf_opt/probe_bw.py --grid 16/28/56`（33.5/34.0/34.9 MB → 209.5/122.6/65.3 µs wall） | `perf_opt/probe_bw.py`（需从本任务复制；Round 1 的同名探针） | 同上 |
| **D：fold grid 越界与 UB 故障同错误码 507035**（`B*parts > 65535` 时 `vector core exception`），拆分方案必须显式夹紧 grid | D | L1-7 崩溃日志与 `_MAX_GRID` 修复后 L1 7/7 | `perf_opt/ChainSpeculativeSamplingFwdOp.py::_MAX_GRID` 与 `run_L1` 的 L1-7 | 同上 |

## 交付摘要

- 主导 case `ds-v3-mtp-n3`：**711.4 → 356.7 µs（1.99x，本轮）；相对 Stage 3 原始 2542.8 µs 为 7.13x**；MBU 2.91% → 5.80%；满足本轮目标 ≤500 µs
- 其他 6 例：1.07x–2.40x（qwen3 仅 1.07x，因 `ntile=19` 质数只能 parts=1；仍不回退；b64-n4 1.77x、b256-n1 1.64x、n1 2.13x、b16-n4 2.40x、b17-n8 2.29x）
- 逐位一致：26/26（vs Round 1 达标 kernel）；候选 L0 13/13、L1 7/7、boundary 7/7
- 中止原因：success；`[DESIGN_LIMIT]`：无（剩余空间是可实现的实现层候选，已在「剩余空间」列出）
- 已知边界：`B*ntile < 112`（如 smoke）或 `B*parts > 65535` 时自动回退融合版；依赖 host 侧惰性 workspace（每 device 一份，n3 为 4.6 KB）

---

# 生产采纳（Production Adoption，Round 2）

**日期**：2026-11-25 ｜ 平台 `ascend-950pr-9579` / `ascend`，单卡串行，`TMPDIR=/workspace`

## 采纳文件清单

| 类型 | 路径 | sha256 |
|---|---|---|
| 备份（Round 1 落地 kernel） | `…/chain_speculative_sampling_kernel/history_version/_chain_speculative_sampling_kernel_pre_round2.py` | `5b2b5a782ea3c13b…` |
| 备份（Round 1 `__init__`） | `…/chain_speculative_sampling_kernel/history_version/__init___pre_round2.py` | `cdbdad1d18e23d8a…` |
| 落地 kernel（本轮） | `…/chain_speculative_sampling_kernel/perf_opt/ChainSpeculativeSamplingFwdOp.py` | `5da0811bcb4d7808…`（= 本任务 `perf_opt/ChainSpeculativeSamplingFwdOp.py`，逐字节相同） |
| 落地别名（wrapper 切换块导入名） | `…/perf_opt/_chain_speculative_sampling_kernel.py` | `d8c8f724c0ce0ac9…` |
| 落地包 `__init__`（本轮） | `…/perf_opt/__init__.py` | `efb67709cad073f6…` |

- `ChainSpeculativeSamplingFwdOp_pre_round2.py` **不存在**（Round 1 未落地该文件名），故未备份该文件。
- wrapper 切换块未改动（Round 1 已指向 perf_opt）：`from .chain_speculative_sampling_kernel.perf_opt._chain_speculative_sampling_kernel import _chain_speculative_sampling_kernel`；运行期核对 `wrapper._chain_speculative_sampling_kernel.__module__ == "…perf_opt.ChainSpeculativeSamplingFwdOp"`，且 `perf_opt` 包级再导出与 `ChainSpeculativeSamplingFwdOp` 内工厂是同一对象。
- 未改 op/manifest/test/bench，未触碰 `/workspace/TileOPs-ascend950`。

## 生产回归

| 回归 | 结果 |
|---|---|
| `cd examples/TileOPs && TILEOPS_TASK_DIR=…/examples/chain_speculative_sampling python -m pytest tests/ops/test_sampling.py -q` | **9 passed**（3.39 s，`perf_opt/logs/production/pytest_sampling.log`） |
| 逐位对拍（部署对：perf_opt 模块 vs 集成包 Stage 3 包） | **26/26 BIT-EXACT**（`logs/production/verify_exact_prod.log`） |
| 逐位对拍（vs Round 1 达标 kernel） | **26/26 BIT-EXACT**（`logs/production/verify_exact_vs_round1.log`） |

## 生产报告（msprof，run_id `20261125_171428_406993_ChainSpeculativeSamplingFwdOp`）

`run.json`：`status=passed`、`correctness_passed=true`（9/9，`Avg Max Abs Error = 0.00e+00`）、`benchmark_passed=true`（7/7）、`msprof_case_count=7`、`invalid_profile_count=0`、`invalid_acceptance_count=0`、`profiler_fallback_count=0`、`kernel_filter_fallback_count=0`、`pass_rate=1.0`。

| workload | T_op µs（本轮生产） | 上轮生产 µs | 本轮提升 | Stage 3 原始 µs | 对原始提升 | GB/s | **MU(MBU) %** | profile |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| llama-8b-b16-n4 | 140.2 | 338.8 | 2.42x | 1245.8 | 8.89x | 58.6 | 3.66 | valid |
| llama-8b-b64-n4 | 393.6 | 702.2 | 1.78x | 2472.1 | 6.28x | 83.4 | 5.21 | valid |
| ds-v3-mtp-n1 | 313.1 | 661.4 | 2.11x | 2529.2 | 8.08x | 105.7 | 6.61 | valid |
| **ds-v3-mtp-n3** | **346.1** | **714.6** | **2.07x** | **2542.8** | **7.35x** | 95.6 | **5.98** | valid |
| qwen3-235b-eagle-n5 | 374.3 | 405.7 | 1.08x | 1427.5 | 3.81x | 52.0 | 3.25 | valid |
| llama-8b-b17-n8 | 155.4 | 347.9 | 2.24x | 1259.7 | 8.11x | 56.1 | 3.51 | valid |
| llama2-7b-b256-n1 | 399.6 | 629.4 | 1.58x | 1666.3 | 4.17x | 82.0 | 5.13 | valid |

- 主导 case 与预期一致：**T_op 346.1 µs（Stage 4 末次采集 356.7，差 −3.0%/采集噪声）/ MU 5.98%（预期 ≈5.8%）**，远非 714.6 µs。
- **报告口径核对**：本轮一次调用是 3 段 launch，三个 `@T.prim_func` 同名 `main` → `captured_op_names = ["main_kernel"]`、`sample_count = 3`，`t_op_breakdown = {"main_kernel": 346.09}`（平台按 op 名求和）。表内 `Latency` 列（86.2 µs）是三次 launch 的**中位数**，**验收判据用 `acceptance.t_op_us`（346.1）**；`mfu_ceiling_percent` 5.22%（访存型，`arithmetic_intensity 0.75`），MFU 口径的 FAIL 属类别对齐后的诊断。
- 报告证据：`examples/TileOPs/reports/tileops/20261125_171428_406993_ChainSpeculativeSamplingFwdOp/{run.json,report.md,benchmark.md,report.html}`；逐 case 原始 msprof 目录见 `perf_records.jsonl` 的 `msprof_raw_path`（`phase=merged`、round=2、candidate_id `perf_opt_split_k3_production`）。

## 一致性说明

- 生产 T_op 与本任务 `phase=final`（同一文件实测：141.2 / 393.1 / 308.7 / 356.7 / 377.4 / 152.7 / 402.6）逐 case 偏差 ≤3.0%（n3 −3.0%、n1 −1.4%、b16 −0.7%、b64 +0.1%、qwen3 −0.8%、b17 +1.8%、b256 −0.7%），排序与结论不变。
- 回退：**无**（7/7 优于上轮生产版本，qwen3 也 +8%）。
- 环境清理：运行前后各清理 `/tmp/tileops_msprof_*`（现为 0）；单卡串行；未触碰 `/workspace/TileOPs-ascend950`。
- 采纳后建议（下一轮可选）：命中 tile 重算与 lane 和的 `locate` 段仍占 24%（n3）/48%（b256-n1），若要再降需让 fold 段顺带产出每 tile lane 和（代价是残差落 UB 的往返，Round 1 实测 2.4x）→ 收益不确定，本轮未做。
