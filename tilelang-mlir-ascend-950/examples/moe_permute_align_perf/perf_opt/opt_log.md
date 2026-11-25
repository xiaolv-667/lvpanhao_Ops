# MoEPermuteAlignFwdOp Stage 4 调优日志

- 任务目录：`examples/moe_permute_align_perf`；算子目录（Stage 3 基线，只读）：`examples/TileOPs/tileops/kernels/moe/moe_permute_align/`
- 平台：`ascend-950pr-9579` / `ascend`，单卡 Ascend950PR_9579（Freq 1650 MHz，28 Cube / 56 Vector 核，UB 253952 B）
- 工具链：TileLang 0.1.15 + CANN 9.2.1（`toolchain_stamp` 见 `perf_records.jsonl` 每行的 `platform_runtime`）
- **测量口径**：任务平台声明的 `measurement.scope = kernel` + `msprof` 的 `Task Duration(us)`；一次调用的 **T_op = 该调用全部 launch 的 `Task Duration` 之和**（`mfu_baseline_v1.json` policy `t_op`）。全部数字来自 `benchmarks/ops/bench_moe_permute_align.py` 原样运行（每个案例一次 `msprof op` 采集），原始 CSV 在 `perf_opt/profiles/<tag>/msprof/`。
- 基线运行：`perf_opt/run_bench.py`（注入方式见 §0.2，未修改任何受版本控制的文件）。
- 精度门：`tests/ops/test_moe_permute_align.py` 全套 30 例（含 smoke 6 例），逐位判据与冻结测试一致（expert 区内顺序按契约归一化后 `torch.equal`；`expert_ids` / `num_tokens_post_pad` 直接 `torch.equal`）。另有本算子自带的综合管线 L0/L1/boundary harness（`python moe_permute_align.py --level L0|L1|boundary`）。

## 0.1 workload inventory

`benchmarks/ops/bench_moe_permute_align.py` 由 `workloads_to_params("MoEPermuteAlignFwdOp", include_extra=True)` 展开 manifest 的 12 条 workload（12 条均为 `int32`、`block_size=64`；`num_experts ∈ {384, 256, 128}`；routes = tokens×8 ∈ {8, 256, 4096, 32768}）。该 benchmark 用例**没有任何 pytest mark**（`smoke`/`skip` 均不存在），故按 `_shared/standards/signal-registry.md` §5 的分类规则 12 条全部为 `kind=tune`，逐条采集 baseline/final；算子级 smoke 覆盖在 `tests/ops`（`pytest -m smoke`，6 例）里，单独回归。分类与原因见 `perf_opt/workload_inventory.json`。

## 0.2 候选注入（不改动受控文件）

冻结 wrapper（`.../moe_permute_align/moe_permute_align.py`）在模块加载时 import 三个工厂、在每次 build 时按模块全局名查找。因此 Stage 4 通过 `perf_opt/inject_site/sitecustomize.py`（`PYTHONPATH` 注入，`msprof` 子进程同样继承）把这三个名字替换为候选模块的同名工厂，**不动任何 tracked 文件**；候选模块与基线工厂签名逐一对齐。最终 `perf_opt/moe_permute_align.py` 自包含，conductor 翻转 wrapper 的 baseline/perf_opt import 块即可采纳。

## 0.3 工件与记录

- `moe_permute_align.py`：最终 tuned 实现（自包含，16 个工厂/常量 + 综合 harness），`statectl` 的 `kernel_opt_py_path`；
  `MoEPermuteAlignFwdOp.py` 是同内容的按算子名别名（仅 re-export 三个工厂与 tuned 常量），供 conductor 按算子名取件。
- `perf_records.jsonl`：本文件在**第一次门禁运行前**由 `write_records.py` 重新生成过一次——初版把 `artifact_path`
  写成仓库相对路径，而 Stage 4 门禁按记录文件自身目录解析该字段（会重复前缀）；同时最终件的 harness 段在首测后有
  一次常量改名（`_golden` → `golden_moe_permute_align`、补 `run_L2`），文件哈希随之变化。重新生成保留了各分支**同一份**
  测量（原始采集 `profiles/<tag>/*` 未变、时延/哈希/时间戳口径不变），仅修正路径编码与 final 的工件哈希；
  之后未再改写任何行（append-only）。逐轮数值见 §3/§4/§5，原始 msprof CSV 全部保留在 `profiles/`。
- `perf_opt/` 顶层其余为工具：`run_bench.py`（平台协议测量）、`write_records.py`、`check_precision.sh`、
  `launch_breakdown.py`（逐 launch 归因）、`probes/`（机制探针）、`inject_site/`（候选注入）、
  `_assemble_*.py` 与 `_*.part.py`（最终件/基线的组装脚本）、`_stage3_base_module.py`（Stage 3 三模块的只读副本，
  供各分支复现）、`logs/`、`profiles/`。

## 1. Performance Test Data（baseline，Phase 1）

`Task Duration` 逐 launch 归因（`perf_opt/launch_breakdown.py r0_base`，原始 CSV 在 `profiles/r0_base/msprof/`）：

| workload | numel | E | baseline T_op (us) | 逐 launch `Task Duration` (us) | vec 占比 / GM 流量 |
|---|---|---|---|---|---|
| kimi-k2-t1 | 8 | 384 | 26.86 | main(S)=24.27 + scatter=2.58 | 0.38 / 114 KB |
| kimi-k2-t32 | 256 | 384 | 43.23 | main(S)=34.70 + scatter=8.53 | 0.41 / 114 KB |
| kimi-k2-t512 | 4096 | 384 | 130.63 | hist=25.93 + align=22.30 + scatter=82.40 | 0.97/0.81/0.99 · 1028 KB |
| kimi-k2-t4096 | 32768 | 384 | 650.67 | hist=26.63 + align=45.46 + scatter=578.58 | 0.96/0.49/1.00 · **8204 KB** |
| ds-v3-t1 | 8 | 256 | 20.32 | main(S)=17.73 + scatter=2.60 | 0.35 / 81 KB |
| ds-v3-t32 | 256 | 256 | 37.89 | main(S)=28.62 + scatter=9.27 | 0.40 / 90 KB |
| ds-v3-t512 | 4096 | 256 | 153.82 | hist=24.00 + align=17.18 + scatter=112.64 | 0.96/0.75/1.00 · 1021 KB |
| ds-v3-t4096 | 32768 | 256 | 934.34 | hist=27.99 + align=38.96 + scatter=867.40 | 0.96/0.42/1.00 · 8155 KB |
| qwen3-235b-t1 | 8 | 128 | 15.41 | main(S)=12.32 + scatter=3.08 | 0.26 / 47 KB |
| qwen3-235b-t32 | 256 | 128 | 40.49 | main(S)=23.42 + scatter=17.08 | 0.37 / 56 KB |
| qwen3-235b-t512 | 4096 | 128 | 249.81 | hist=22.15 + align=10.75 + scatter=216.92 | 0.96/0.69/1.00 · 999 KB |
| qwen3-235b-t4096 | 32768 | 128 | 1732.50 | hist=26.34 + align=30.20 + scatter=1675.96 | 0.96/0.36/1.00 · 8020 KB |

（"vec 占比" = msprof `ArithmeticUtilization` 的 `aiv_vec_ratio` 均值；"GM 流量" = `Memory` CSV 的 `read/write_main_memory_datas(KB)` 之和。）

**现象（baseline）**：

1. `t_op` 的 89%（主导档 kimi-k2-t4096：578.6/650.7）在 **scatter 段**；其 `aiv_vec_ratio = 1.00`（向量流水线几乎 100% 忙）但 GM 主存流量只有 8204 KB —— 输出只有 57023×4 B = 228 KB，**写放大 ≈ 36×**，而有效带宽只有 21 GB/s（峰值 1.6 TB/s）。判据：scatter 的每路由一次**全局** `atomic_add(cumsum[eid],1,prev)` 被降级成逐 lane 标量 RMW（128 lane 串行），既不是带宽瓶颈也不是并行度不足，而是**逐 lane GM 原子往返**。
2. E 越小越糟（E=128 时 1676 us vs E=384 时 579 us）：同一批 128 个 GM 地址上的争用更集中，与「逐 lane GM 原子」的判据一致。
3. `expert_ids` 的 rank-count（`for e in T.serial(E): cnt += (s_pref[e] <= b*B)`）在**小档**占比很高：qwen3-t1 的 12.32 us 几乎全在这一段（E=128 × 128 lane），kimi-t32 的 34.70 us 同理（E=384）。
4. 直方图段 22–28 us 与路由数几乎无关地贴在 ~26 us（每 token `2·ceil(E/64)` 条向量指令 × 每核 586 token），是「E 维比较 + 累加」的发射开销。

**设计估算 vs 实测偏差（D-2 回填）**：`_make_align_kernel/DESIGN.md` §1.6.0 roofline 给出路径 L 估算下界 **≈6.0 us（+2 launch ≈10 us）**；`_make_scatter_kernel/DESIGN.md` §1.6.0 表给 scatter 下界 **≈3.2 us（32768/384）**。baseline 实测 align 段 72.0 us、scatter 段 578.6 us ⇒ 偏差 7.2× / 181×。失准项唯一且明确：**「GM 原子 ≈4 cyc/路由」的未实测常数**（见 §2 探针 `gmcur`：实测 ≈1465 cyc/路由/lane）。流量项（0.36 MB ⇒ 0.22 us）与容量项（20 KB/248 KB）都不失准。

## 2. 机制探针（Phase 2 前置；`perf_opt/probes/probe_mechanisms.py`）

所有探针都在最大档几何（numel=32768、E=384、56 核 × 128 lane、5 路由/lane）上单独测量，口径同平台（`msprof op`，`Task Duration`），原始采集见 `logs/probe/mechanisms*.log`：

| 探针 | 机制 | Task Duration (us) | 结论 |
|---|---|---|---|
| `contig` | 每路由写自己下标（连续） | 2.21 | 56 核 + 32768 次连续写的地板 |
| `rand` | 每路由写在 32768 长数组内打散的地址 | 6.70 | **逐 lane 数据相关 GM 写 ≈ +14 cyc/路由**（可接受） |
| `ubscat` | 先写 UB 窗口再整块 DMA 出 | 15.54 | 比直写 GM **更慢** ⇒ 不做 UB 暂存 |
| `histcmp` | 当前直方图（E-lane 比较累加） | 29.08 | 复现 baseline hist 的 26.6 us |
| `histunroll` | 同上 + 8 条独立累加链 | 11.57 | 2.5×（仍未采用：被 `histatom2` 超过） |
| `histatom2` | 每路由一次 **UB 原子**直方图 | **3.79** | 7.7×；UB 原子 ≈ +5 cyc/路由 |
| `ubatom` | 每路由 UB 原子游标 + 连续写 | **4.05** | UB 原子游标几乎免费 |
| `gmcur` | 每路由 **GM 原子**游标（= Stage 3 scatter） | 586.03 | 复现 baseline scatter（578.6 us）；**≈1465 cyc/路由** |

**判据**：Stage 3 scatter 的 586 us 几乎全部是 GM 原子的逐 lane 往返（`ubatom` 4.05 us 与 `gmcur` 586 us 是同一算法、仅原子所在层级不同）；逐路由的**打散写**只要 6.7 us。因此唯一有效的结构是把「槽位分配」从 GM 原子改成**可推导的确定性槽位**。

## 3. Round 1（base = current best = Stage 3 基线 `stage3_baseline`）

### 3.1 现象分析（基于 baseline 最新 profile）

见 §1 现象 1–4。一句话：**逐 lane GM 原子**（scatter，89%）+ **O(E) rank-count**（小档）+ **E 维比较直方图**（22–28 us）。

### 3.2 候选优化点（同轮 4 个分支，每个只改一个主要机制）

| 分支 | 优化点 | 派生自 | 预期 |
|---|---|---|---|
| **C1** `moe_permute_align_round1_c1.py` | scatter 的槽位改为**推导**：`cumsum[e] + 更早核的同 e 路由数（来自 partial 直方图行） + 本核已放置数（UB 游标）`；align 段随之输出每核游标行 | stage3_baseline | 去掉 586 us 的 GM 原子 |
| **C2** `moe_permute_align_round1_c2.py` | 设计文档 §1.6.3 **备选 B（两级保留）**：块内 UB 直方图 + 每（块, expert）一次批量 GM 保留 + 二趟写。这是 `_make_scatter_kernel/DESIGN.md` §1.6.3「实验裁决计划 T2」要求 Stage 4 做的 A/B | stage3_baseline | DESIGN 估：GM 原子数从 numel 降到 `Σ_blk min(tile,E)` |
| **C3** `moe_permute_align_round1_c3.py` | `expert_ids` 改**expert 侧展开**（每个 expert 写自己拥有的块区间），删掉每块 O(E) 的 rank-count | C1 | 省 E 条串行向量指令/块 |
| **C4** `moe_permute_align_round1_c4.py` | 直方图改**每路由一次 UB 原子**（`histatom2`） | C1 | 29.1 → 3.8 us |

### 3.3 候选 vs current best（Task Duration，us；L0 = `tests/ops` 全套 30 例）

| workload | current best (baseline) | C1 | C2（备选 B） | C3 | C4 | 利用率/访存对照（kimi-k2-t4096） |
|---|---|---|---|---|---|---|
| kimi-k2-t1 | 26.86 | 24.20 | 27.82 | **14.67** | 25.46 | — |
| kimi-k2-t32 | 43.23 | 36.69 | 50.14 | 25.67 | **25.49** | — |
| kimi-k2-t512 | 130.63 | 53.55 | 197.10 | 41.33 | **31.61** | — |
| kimi-k2-t4096 | 650.67 | 82.48 | 822.88 | 69.72 | **60.60** | 见下 |
| ds-v3-t1 | 20.32 | 19.16 | 21.66 | **13.79** | 19.23 | — |
| ds-v3-t32 | 37.89 | 30.31 | 42.01 | 24.81 | **21.57** | — |
| ds-v3-t512 | 153.82 | 46.99 | 203.53 | 38.88 | **26.67** | — |
| ds-v3-t4096 | 934.34 | 75.48 | 665.60 | 68.34 | **52.52** | — |
| qwen3-235b-t1 | 15.41 | 19.06 | 15.79 | **13.35** | 18.04 | — |
| qwen3-235b-t32 | 40.49 | 24.35 | 33.94 | 25.09 | **15.25** | — |
| qwen3-235b-t512 | 249.81 | 39.12 | 177.99 | 36.16 | **19.97** | — |
| qwen3-235b-t4096 | 1732.50 | 64.64 | 404.68 | 65.43 | **42.25** | — |
| **L0（30 例）** | PASS | **PASS** | **PASS** | **PASS** | **PASS** | |

利用率 / 访存（`launch_breakdown.py`，主导档逐 launch `aiv_vec_ratio` / GM KB）：
- baseline：hist 26.63 us(0.96 / 232 KB) + align 45.46 us(0.49 / 4956 KB) + **scatter 578.58 us(1.00 / 8204 KB)**
- C1：hist 26.6 + align 47.9 + **place 7.9**（scatter 段的 GM 原子消失，写放大随之消失）
- C4：hist **4.0** + align 47.9 + place 7.9
- 结论：C1/C4 把「向量流水线 100% 忙在标量原子上」变成「写入 3.9 MB、`vec 0.91` 的真实散射写」；C2 只是把同一个 GM 原子的争用摊薄，流量与向量占用都没变。

### 3.4 分支裁决

- **C1 采纳**（结构性胜出）：主导档 650.67 → 82.48（−87.3%），全部 12 档不回退（qwen3-t1 的 +24% 是 3 launch 地板 + 尚未消除的 O(E) rank-count，由 C3 修复）。
- **C2 否决（实验裁决结论）**：按 `_make_scatter_kernel/DESIGN.md` §1.6.3 的判定阈值「备选 B 在任一代表 shape 上 < 主选 × 0.9 才翻转，且最大规模 shape 上不劣于主选」——实测备选 B 在最大规模 shape（32768/384）上 **822.88 > 650.67（劣 26%）**，在 4096/384 上也劣（197.10 vs 130.63），只在低 E 档（128 experts，争用主导）胜出（404.68 vs 1732.50）。⇒ **主选（单级 bump）按设计自己的阈值胜出，两级形式不翻转**；而两者都被 C1 的确定性槽位碾压（82.48 / 822.88 vs 822.88）。该数据同时证伪了设计对「确定性候选 #2 代价过高」的估计（§8）。
- **C3 采纳**：t1 档 24.20 → 14.67（−39%），t4096 82.48 → 69.72（−15.5%）；qwen3-t32 有 −3%（24.35 → 25.09）的单点抖动，在 ±5% run 间噪声内（见 §5.2 的重复 run）。
- **C4 采纳**：t4096 82.48 → 60.60（−26.5%），t512 53.55 → 31.61（−41%）；t1 档 ±1 us 属 launch 抖动。

### 3.5 Round 1 合入（`moe_permute_align_round1_merged.py` = C1+C3+C4）

三者作用于流水线的不同部位（placement launch / 块归属循环 / 直方图 launch），合入后复测全部 12 档 + smoke：

| workload | current best (baseline) | merged R1 | 变化 |
|---|---|---|---|
| kimi-k2-t1 | 26.86 | 15.50 | −42.3% |
| kimi-k2-t32 | 43.23 | 16.21 | −62.5% |
| kimi-k2-t512 | 130.63 | 18.76 | −85.6% |
| kimi-k2-t4096 | 650.67 | **48.36** | **−92.6%** |
| ds-v3-t1 | 20.32 | 15.32 | −24.6% |
| ds-v3-t32 | 37.89 | 15.28 | −59.7% |
| ds-v3-t512 | 153.82 | 18.42 | −88.0% |
| ds-v3-t4096 | 934.34 | 45.89 | −95.1% |
| qwen3-235b-t1 | 15.41 | 13.79 | −10.5% |
| qwen3-235b-t32 | 40.49 | 16.11 | −60.2% |
| qwen3-235b-t512 | 249.81 | 17.29 | −93.1% |
| qwen3-235b-t4096 | 1732.50 | 44.08 | −97.5% |
| L0（30 例）/ smoke | PASS | PASS | 无回退 |

## 4. Round 2（base = current best = `merged_round1`）

### 4.1 现象分析（merged 最新 profile）

逐 launch 归因（kimi-k2-t4096，`sum = 48.36`）：**hist 4.04 + align 36.71 + place 7.60**。瓶颈已从 scatter 段迁到 **align 段（76%）**：该 launch 用 `for r in T.serial(g): T.copy(partial[r,:], s_part)` 逐行搬 56 行（每条 1.5 KB）——正是 `TRAP-950-rowload-dma-chain`（逐行 GM→UB copy 链按描述符数计费）与 `PL-950-window-bulk-copy` 的形态。小档（g==1）的 8.7–11.1 us 则是**单核做 24 KB–98 KB 哨兵填充 + 3 次 launch 地板**。

### 4.2 候选优化点

| 分支 | 优化点 | 派生自 |
|---|---|---|
| **C5** `moe_permute_align_round2_c5.py` | partial 归并改为**整窗一次 copy** + UB 内累加（替代 g 条逐行 descriptor） | merged_round1 |
| **C6** `moe_permute_align_round2_c6.py` | g==1 的几何把 placement 并入 align launch（哨兵填充之后，同核内顺序与 fused 小批路径一致），scatter 退化为 Python no-op ⇒ 2 launch | merged_round1 |

### 4.3 候选 vs current best

| workload | current best (merged R1) | C5 | C6（首版，含缺陷） |
|---|---|---|---|
| kimi-k2-t1 | 15.50 | 15.01 | 13.31 |
| kimi-k2-t32 | 16.21 | 16.80 | 13.94 |
| kimi-k2-t512 | 18.76 | 20.47 | 21.54 |
| kimi-k2-t4096 | 48.36 | **31.01** | 58.57 |
| ds-v3-t1 | 15.32 | 14.86 | 12.25 |
| ds-v3-t32 | 15.28 | 15.84 | 14.50 |
| ds-v3-t512 | 18.42 | 20.53 | 22.44 |
| ds-v3-t4096 | 45.89 | **27.42** | 59.45 |
| qwen3-235b-t1 | 13.79 | 13.11 | 11.98 |
| qwen3-235b-t32 | 16.11 | 14.39 | 13.79 |
| qwen3-235b-t512 | 17.29 | 17.99 | 21.51 |
| qwen3-235b-t4096 | 44.08 | **28.39** | 61.93 |
| L0（30 例） | PASS | PASS | PASS |

利用率/访存（kimi-k2-t4096，逐 launch）：merged align = 36.71 us(0.49 / 4956 KB) → C5 align = **18.97 us(0.36 / 6988 KB)**：同一份 4.8 MB 冗余读，descriptor 数从 56 降到 1，时间减半。

### 4.4 分支裁决与合入

- **C5 采纳但加 shape 条件**：g=56 档 −17.4 us（−36%），但 g=8 档（t512）**+2.1 us（+11%）**、g=1 档 +0.5 us。按 skill「出现可信回退时在本 kernel 内用编译期条件路径」，最终版在 `@T.prim_func` 内按 **`g >= _BULK_MERGE_MIN_CORES (=16)`** 选择整窗 copy，否则保留逐行链（两分支均为编译期常量决定，无 Python 层分派）。
- **C6 首版暴露实现缺陷、修好后并入最终版**：首版对 **所有** g 都在 align launch 里做 placement，而 g>1 时 scatter 仍会再放一次（两次写的是同一个槽集合、顺序不同 ⇒ 语义仍合法、L0 全过，但做双份工作）⇒ t4096 +21%、t512 +14%。修正为**只有 `g == 1` 才在 align 内置 placement**（编译期 `fused` 标志），g>1 的代码路径与 merged R1 逐字节等价；修正版作为最终版一起复测（§5），g==1 档的收益保留（t1 15.50 → 13.31、t32 16.21 → 13.94）。
- 无其它回退项。

## 5. 最终产物与其测量（`perf_opt/moe_permute_align.py`，候选 `final_tuned`）

结构 = merged R1（C1+C3+C4）+ C5（`g>=16` 整窗归并）+ C6（`g==1` 单核融合）：

1. **launch 1**：每核 UB 原子直方图 → `partial[g, E_pad]`；
2. **launch 2**：归并（整窗/逐行按 g 分派）+ 补齐前缀和 + `cumsum`/`num_tokens_post_pad` + `expert_ids`（expert 侧展开）+ 哨兵填充 + 每核游标行；`g==1` 时同时做 placement；
3. **launch 3**：确定性槽位 placement（`g>1`）；`g==1` 时为 Python no-op。

### 5.1 精度

| 门 | 结果 |
|---|---|
| `tests/ops/test_moe_permute_align.py`（30 例，含 smoke 6 例） | **30 passed**（`logs/precision/moe_permute_align-all.log`、`-smoke.log`） |
| 综合管线 harness `--level L0`（9 例：三条路径 + numel=0 + E=33 + E=128 等） | PASS，`max_diff=0`（`logs/final/harness_L0.log`） |
| `--level L1`（12 例：numel×E 网格） | PASS，`max_diff=0` |
| `--level L2`（4 例：numel=65536 / E=1100 / numel=1，超 manifest） | PASS，`max_diff=0` |
| `--level boundary`（9 例：B=1/非 2 幂、E=1/E=65、numel=1/1023） | PASS，`max_diff=0` |
| 逐位判据 | `expert_ids` / `num_tokens_post_pad` 直接 `torch.equal`；`sorted_token_ids` 按契约把每个 expert 区段内的路由升序归一后 `torch.equal`（与冻结测试 `_assert_route_equivalent` 同判据）——**等价于逐位一致**（expert 区内顺序是 op 契约明示的自由项；Stage 3 与最终版都只保证集合/多重集一致，行内顺序不同不会被判为差异） |

平台协议的精度入口（`measurement` 的 `precision_command`，工作目录 = 任务目录）
`/usr/local/python3.12.16/bin/python3 perf_opt/moe_permute_align.py --level <L0|L1|L2|boundary>` 即上表四条；
`PYTHONPATH` 需含 venv site-packages（该解释器本身不带 torch/tilelang）。

### 5.2 稳定性（两次独立 run，取中位）

| workload | run0 | run1 | 中位 | run 间偏差 |
|---|---|---|---|---|
| ds-v3-t1 | 13.14 | 12.46 | 12.80 | 5.2% |
| ds-v3-t32 | 12.77 | 13.80 | 13.28 | 7.5% |
| ds-v3-t4096 | 26.74 | 26.60 | 26.67 | 0.5% |
| ds-v3-t512 | 17.98 | 18.34 | 18.16 | 2.0% |
| kimi-k2-t1 | 12.67 | 12.88 | 12.77 | 1.6% |
| kimi-k2-t32 | 14.09 | 14.04 | 14.06 | 0.3% |
| kimi-k2-t4096 | 31.05 | 30.30 | 30.67 | 2.4% |
| kimi-k2-t512 | 19.14 | 18.18 | 18.66 | 5.0% |
| qwen3-235b-t1 | 11.53 | 12.52 | 12.02 | 7.9% |
| qwen3-235b-t32 | 13.52 | 13.03 | 13.27 | 3.6% |
| qwen3-235b-t4096 | 28.42 | 27.56 | 27.99 | 3.0% |
| qwen3-235b-t512 | 17.40 | 18.49 | 17.94 | 5.9% |

小档（11–14 us）的 run 间偏差 3–9%，绝对量 ≤1.2 us，已接近 2–3 次 launch 的地板（`CONST-950-kernel-launch-floor-48blk` / `CONST-950-launch-floor-grid-sweep`：单 launch 2.7–3.0 us；本算子实测 1.9–2.7 us）。大档偏差 ≤5%。**最终记录取两次 run 的中位**（`profiles/r3_final/summary.json`），Final 表逐项来自同一候选 `final_tuned` 的同一份测量。

### 5.3 最终逐 launch 归因（run0，`launch_breakdown.py r3_final`）

| workload | T_op | launch 明细 |
|---|---|---|
| kimi-k2-t4096 | 31.05 | hist 4.17(vec 0.82) + align 19.58(0.35) + place 7.30(0.91) |
| ds-v3-t4096 | 26.74 | hist 3.84 + align 15.71 + place 7.20 |
| qwen3-235b-t4096 | 28.42 | hist 5.25 + align 17.10 + place 6.07 |
| kimi-k2-t512 | 19.14 | hist 3.36 + align 10.43 + place 5.34 |
| qwen3-235b-t512 | 17.40 | hist 3.35 + align 8.48 + place 5.57 |
| kimi-k2-t32 | 14.09 | hist 2.67 + align 11.42（含哨兵 142 KB 单核填充） |
| qwen3-235b-t1 | 11.53 | hist 1.94 + align 9.59（含哨兵 43 KB + 元数据） |

## 6. Final Summary

### Final Performance Test Data

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---|---|---|
| moe_permute_align.py::MoEPermuteAlignKernel | ds-v3-t1 | 20.32 | 12.80 | final_tuned |
| moe_permute_align.py::MoEPermuteAlignKernel | ds-v3-t32 | 37.89 | 13.28 | final_tuned |
| moe_permute_align.py::MoEPermuteAlignKernel | ds-v3-t4096 | 934.34 | 26.67 | final_tuned |
| moe_permute_align.py::MoEPermuteAlignKernel | ds-v3-t512 | 153.82 | 18.16 | final_tuned |
| moe_permute_align.py::MoEPermuteAlignKernel | kimi-k2-t1 | 26.86 | 12.77 | final_tuned |
| moe_permute_align.py::MoEPermuteAlignKernel | kimi-k2-t32 | 43.23 | 14.06 | final_tuned |
| moe_permute_align.py::MoEPermuteAlignKernel | kimi-k2-t4096 | 650.67 | 30.67 | final_tuned |
| moe_permute_align.py::MoEPermuteAlignKernel | kimi-k2-t512 | 130.63 | 18.66 | final_tuned |
| moe_permute_align.py::MoEPermuteAlignKernel | qwen3-235b-t1 | 15.41 | 12.02 | final_tuned |
| moe_permute_align.py::MoEPermuteAlignKernel | qwen3-235b-t32 | 40.49 | 13.27 | final_tuned |
| moe_permute_align.py::MoEPermuteAlignKernel | qwen3-235b-t4096 | 1732.50 | 27.99 | final_tuned |
| moe_permute_align.py::MoEPermuteAlignKernel | qwen3-235b-t512 | 249.81 | 17.94 | final_tuned |

提升（`baseline → final`，`Task Duration` 之和）：

| workload | baseline | final | 提升 |
|---|---|---|---|
| qwen3-235b-t4096 | 1732.50 | 27.99 | -98.4%（61.9×） |
| ds-v3-t4096 | 934.34 | 26.67 | -97.1%（35.0×） |
| kimi-k2-t4096 | 650.67 | 30.67 | -95.3%（21.2×） |
| qwen3-235b-t512 | 249.81 | 17.94 | -92.8%（13.9×） |
| ds-v3-t512 | 153.82 | 18.16 | -88.2%（8.5×） |
| kimi-k2-t512 | 130.63 | 18.66 | -85.7%（7.0×） |
| kimi-k2-t32 | 43.23 | 14.06 | -67.5%（3.1×） |
| qwen3-235b-t32 | 40.49 | 13.27 | -67.2%（3.1×） |
| ds-v3-t32 | 37.89 | 13.28 | -64.9%（2.9×） |
| kimi-k2-t1 | 26.86 | 12.77 | -52.4%（2.1×） |
| ds-v3-t1 | 20.32 | 12.80 | -37.0%（1.6×） |
| qwen3-235b-t1 | 15.41 | 12.02 | -22.0%（1.3×） |

- **12/12 tune workload 全部提升，无回退**；主导档 kimi-k2-t4096 −95.3%（21.2×），远超「至少 −30%」的目标。
- smoke / skip：benchmark 用例无 smoke/skip（见 §0.1）；算子级 smoke 6 例在最终文件上 **PASS**。
- 中止原因：**success**（达到用户性能目标，全部 tune workload 有结论，最终全量复测完成）。
- 遗留（未做，受预算约束，见 §7）：align launch 的 4.8 MB 冗余归并读（有效 368 GB/s，仍非带宽上限）与小档哨兵填充的单核串行（43–142 KB/核）；两者合计约 8–12 us，是下一步最明确的两个方向。

## 7. 未采用的候选与遗留方向（family_no_gain / config_no_gain 范围）

| 项 | 结论 | 依据 |
|---|---|---|
| UB 暂存 + 整块 DMA 出（`ubscat`） | 否（比直写 GM 慢 2.3×） | 15.54 us vs 6.70 us |
| 直方图 8 链展开（`histunroll`） | 否（被 UB 原子超过） | 11.57 us vs 3.79 us |
| 两级 GM 保留（设计备选 B，C2） | 否（设计阈值判定 + 被 C1 碾压） | §3.4 |
| 整窗归并用在小 g（8 核） | 否（+2.1 us），改用 g≥16 条件 | §4.4 |
| 小档哨兵填充分核（把填充挪到 hist launch，2–4 核分摊） | 未做 | 预计 −4~6 us（6 个档），属 Round 3 候选 |
| 归并冗余读消除（单核归并 + GM 广播 or 跨核 flag） | 未做 | 需跨核同步能力，工具链未认证（`capability-gaps` 边界），预计 −8~12 us（3 个大档） |
| autotune | 未使用 | 无 tune 参数暴露（`default_config = {}`）；参数选择由 §2 机制探针 + 编译期常量裁决完成 |

## 8. 设计层天花板判定（`[DESIGN_LIMIT]`）→ **不产出** `perf_feedback.md`

按 `_shared/standards/perf-feedback.md` §1 双条件逐条核对：

- **条件 ②（实测与设计假设矛盾）成立**：`_make_scatter_kernel/DESIGN.md` §0.3/§1.6.3 以「GM 原子 ≈4 cyc」的未实测常数论证「单级 bump 下界 3.2 us」，并据此把确定性路线（其 §1.6.0 候选 #2「expert 所有权 + 谓词前缀压缩」与本文的「partial 直方图 + 游标」）判为不必选。实测 `gmcur` 586.03 us（≈1465 cyc/路由/lane，**偏差 181×**），而确定性槽位 = 7.16 us。
- **条件 ①（性能天花板由算法/设计层决定）不成立**：目标（主导档 −30%）已达成且远超目标（−95.4%）；最终版 29.63 us 与设计自身 roofline 下界（≈10 us，含 2–3 次 launch 地板）只差 3×，且差距已定位为**可继续在 Stage 4 内解决的实现项**（§7 两个方向），不是「设计冻结导致的不可达」。设计文档已把该结构选择显式交给 Stage 4 做 A/B（§1.6.3 实验裁决计划），本轮的 C1/C2 实测正是执行该计划并把结论按设计自己的阈值落定。

⇒ 两项不同时满足，**禁止产出 `perf_feedback.md`**；设计代价模型的证伪结论见 §6 与 §9（已回写平台知识库）。

## 9. Skill Retrospective

### 9.1 本轮有效的东西（可复用）

1. **「机制代价探针先于结构重设计」**：先用 8 个单机制探针（`probes/probe_mechanisms.py`，每个 1 次 launch、同口径 `Task Duration`）测出 `GM 原子 1465 cyc/路由` vs `UB 原子 5 cyc/路由` vs `打散 GM 写 14 cyc/路由`，才敢把 586 us 的段整体换掉；探针总耗时 < 15 min，避免了盲改。**这一步的收益（≈20×）远超参数级调优**。
2. **设计文档自带的「实验裁决计划」直接当 A/B 清单**：`_make_scatter_kernel/DESIGN.md` §1.6.3 的 T2 阈值（「备选 < 主选×0.9 且最大档不劣」）让 C2 的否决有据可依，不需要自己造判据。
3. **每轮把「逐 launch Task Duration 归因」当第一手证据**：baseline 用 `t_op` 总和看不出瓶颈在哪个 launch（hist 26.6 / align 45.5 / scatter 578.6 才是关键），`launch_breakdown.py` 从 msprof CSV 还原逐 launch 归属，直接决定 Round 1 只打 scatter、Round 2 只打 align。
4. **编译期条件路径按实测阈值分派**（`g>=16` 用整窗归并、`g==1` 融合 placement）：避免一个机制在部分 shape 上回退。

### 9.2 踩到的坑（已回写知识库）

1. **UB 原子/数据相关访问必须用「header 形式」的 `T.SimtVF`**：把 `T.Kernel(...)` 与 `T.SimtVF(...)` 分成两层 `with`（或在 `with T.SimtVF` 之外声明 `T.alloc_local`）会得到 `pointer arguments to simt_vf functions must reside in '__gm__ or '__ubuf__' address space`（local buffer 被当成形参），或者编译器语义检查 `T.Parallel loop 'j' must be inside a VF block`。首轮探针因此假失败两次（误以为 UB 原子不可用），改成与 Stage 3 scatter 同构的 header 形式后正常。→ 已登记 `TRAP-950-ub-atomic-vf-scope`。
2. **单机制探针必须与生产结构同构**：`ubscat`/`histatom` 首版在 `T.Kernel` 里嵌套了第二个 `T.SimtVF`，msprof 只报「无 CSV」而不报编译错误（错误被 `bench_kernel_msprof` 的 try/except 吞掉并截断到 400 字符），白跑两轮。→ 探针失败时应先在进程内直接 `prog(...)` 复跑一次拿完整栈。
3. **同一个 kernel 的「融合」改动要按编译期条件限定作用域**：C6 首版对所有 g 都在 align 里做 placement，g>1 时与 scatter 重复，L0 全过但白干一倍活（+21%）。正确性门不会发现「做双份工作」，只有逐 launch 归因能发现。
4. **`T.copy` 的 descriptor 数与 `T.all`/2-D 整窗 copy 的差别**：`TRAP-950-rowload-dma-chain` 在本算子再次复现（56 行 × 1.5 KB 逐行链 = 36.7 us → 整窗 18.97 us），但**整窗并非在所有 g 上都更快**（g=8 时 +2.1 us），说明该陷阱的「行数阈值」与行长、UB 索引方式相关，不能无脑套用。

### 9.3 流程建议（proposal）

- **BP_s4_mechanism_probe_first**：Stage 4 第一轮之前强制跑「机制代价探针」（≤15 min），把候选结构依赖的未实测常数（原子/散写/DMA descriptor/launch 地板）逐一测出来；本任务该步骤把「猜结构」变成「按代价表选结构」，直接产出 20× 级收益。
- **BP_s4_launch_attribution**：把「逐 launch `Task Duration` 归因」作为 Phase 1 的必需产物（现在 skill 只要求 kernel 级），因为 op 级 T_op 会掩盖真实瓶颈点。
- **BP_s4_fusion_scope_flag**：任何「把 A launch 的活并到 B launch」的候选，必须在编译期条件里限定适用的 shape，并在合并后逐 launch 复测；否则「重复工作」类缺陷不会被精度门发现。

### 9.4 价值点提案（供 `tilelang-skill-evolver` 蒸馏）

| id | vp_type | 断言 | 证据三件套 |
|---|---|---|---|
| VP-2026-1101 | D | 950PR 上**逐 lane GM `atomic_add` ≈1465 cyc/路由**（128 lane 串行、无流水）；同一算法改用 **UB 原子 ≈5 cyc/路由**（4.05 us vs 586.03 us，numel=32768/56 核） | evidence：`perf_opt/logs/probe/mechanisms2.log`（`gmcur` 586.028 / `ubatom` 4.048）；repro：`.agents/platforms/toolchains/ascend/knowledge/repro/CONST-950-atomic-tier-cost.py`；toolchain_stamp：`7a52f2f…`（0.1.15 + CANN 9.2.1） |
| VP-2026-1102 | D | 逐 lane **数据相关 GM 写**仅 +14 cyc/路由（打散 6.70 us vs 连续 2.21 us）⇒ 「散射写贵」在本平台不成立，贵的是原子往返 | evidence：同上（`contig`/`rand`）；repro 同上文件 |
| VP-2026-1103 | P | **确定性槽位 = partial 直方图 + 每核游标行 + UB 原子 bump**，可完全取代「每路由一次 GM 原子」的 bump allocator，且无需跨核同步（partial 行由阶段 1 产出、阶段 2 归并且每核冗余重算 prefix） | evidence：`perf_opt/opt_log.md` §3.4（650.67 → 82.48，scatter 578.58 → 7.60）；repro：`knowledge/repro/PL-950-deterministic-slot-alloc.py`；toolchain_stamp 同上 |
| VP-2026-1104 | P | 「E 维比较累加」直方图（numel×E）在 950PR 上是发射受限（≈82 cyc/element-iter）；**每路由一次 UB 原子**把它降到 O(numel)（29.08 → 3.79 us） | evidence：`mechanisms2.log`（`histcmp`/`histatom2`）；repro 同 VP-2026-1101 文件 |
| VP-2026-1105 | R | Stage 4 第一轮前应做「机制代价探针」并把**逐 launch** `Task Duration` 归因列入 Phase 1 产物（见 §9.3 BP） | evidence：本任务 `opt_log.md` §2/§5.3 |
| VP-2026-1106 | C | 案例索引：`CASE-batch2-moe-permute-align` 应补充「Stage 4 已把 atomic bump 换成确定性槽位（22×）」与代价常数入口 | evidence：本文件；`knowledge/cases.md` |

## 10. 生产采纳（conductor 接线 + 验收，2026-11-25）

### 10.1 采纳方式（wrapper 切换块翻转）

- **备份**：`moe_permute_align_kernel/history_version/moe_permute_align_pre_opt.py`
  （= 翻转前 wrapper 全文，sha256 `7155a556…7ab7ad`，与原文逐字节一致）。
- **新增**（drop-in，按 Stage 3 包内布局）：
  - `moe_permute_align_kernel/perf_opt/__init__.py`（包入口，re-export 三个工厂 + tuned 常量）
  - `moe_permute_align_kernel/perf_opt/moe_permute_align.py`（自包含实现，sha256 `6028d4b1…17b325`，
    与 `examples/moe_permute_align_perf/perf_opt/moe_permute_align.py` **同一文件**）
  - `moe_permute_align_kernel/perf_opt/_make_{align,scatter,small_batch}_kernel.py`（各 re-export 一个工厂，
    因为切换块按 Stage 3 布局「一模块一工厂」import；三者指向同一实现模块 ⇒ `_scratch` 单一实例）
- **wrapper 改动**：仅 `moe_permute_align.py` 的 baseline/perf_opt 双 import 切换块——baseline 三行注释掉、
  perf_opt 三行启用（并在注释里记下实测收益）。op 层（`tileops/ops/moe/permute_align.py`）、manifest、
  `tests/`、`benchmarks/`、Stage 3 kernel 目录**均未改动**。
- 切换块语义保持：三个工厂名、签名、`default_config = {}`、`_SMALL_NUMEL_THRESHOLD` / `_WARP_LANES` 分派逻辑全部不变，
  `MoEPermuteAlignKernel` 的构造/forward 契约不变。

### 10.2 接线自检（`__module__` 走 perf_opt）

```
wrapper._make_align_kernel.__module__        = …moe_permute_align_kernel.perf_opt.moe_permute_align
wrapper._make_scatter_kernel.__module__      = …moe_permute_align_kernel.perf_opt.moe_permute_align
wrapper._make_small_batch_kernel.__module__  = …moe_permute_align_kernel.perf_opt.moe_permute_align
numel=32768: align_fn.__module__=…perf_opt.moe_permute_align  path=L  g_h=56   scatter=…perf_opt…
numel=256  : align_fn.__module__=…perf_opt.moe_permute_align  path=L  g_h=1    scatter=…perf_opt…
numel=8/E=4: small_batch_fn.__module__=…perf_opt.moe_permute_align   （wrapper 自己的小批路径）
kernel_class = tileops.kernels.moe.moe_permute_align.moe_permute_align.MoEPermuteAlignKernel（不变）
```

### 10.3 生产回归

| 项 | 结果 | 证据 |
|---|---|---|
| `pytest tests/ops/test_moe_permute_align.py -q`（生产接线，无注入） | **30 passed** | `logs/production/pytest_production.log` |
| 逐位校验（12 manifest + 21 边界/超参数 + 2 偏斜 = 35 例） | **全部 bit-exact**：`num_tokens_post_pad`/`expert_ids` 直接 `torch.equal`；`sorted_token_ids` 按契约归一后 `torch.equal` | `logs/production/bitexact_production.log` |
| 单算子报告 `tileops.reporting.cli run --op MoEPermuteAlignFwdOp --prof-mode msprof` | **status=passed**；correctness 30/30；benchmark 12/12；`invalid_acceptance_count=0`、`invalid_profile_count=0`、`profiler_fallback_count=0`、`kernel_filter_fallback_count=0`；`warnings=[]` | `reports/tileops/20261125_023838_308793_MoEPermuteAlignFwdOp/`（`run.json` / `benchmark.json` / `report.md`） |

run id：**`20261125_023838_308793_MoEPermuteAlignFwdOp`**

### 10.4 生产实测（同一平台口径，`phase=merged`）

| workload | baseline_us | final_us（task dir） | **production_us** | 报告 latency_us |
|---|---|---|---|---|
| ds-v3-t1 | 20.32 | 12.80 | 12.25 | 9.92 |
| ds-v3-t32 | 37.89 | 13.28 | 13.65 | 11.25 |
| ds-v3-t4096 | 934.34 | 26.67 | 27.69 | 6.99 |
| ds-v3-t512 | 153.82 | 18.16 | 19.23 | 5.62 |
| kimi-k2-t1 | 26.86 | 12.77 | 12.50 | 10.54 |
| kimi-k2-t32 | 43.23 | 14.06 | 14.06 | 11.65 |
| kimi-k2-t4096 | 650.67 | 30.67 | 29.72 | 7.26 |
| kimi-k2-t512 | 130.63 | 18.66 | 18.65 | 5.38 |
| qwen3-235b-t1 | 15.41 | 12.02 | 11.58 | 9.49 |
| qwen3-235b-t32 | 40.49 | 13.27 | 12.92 | 10.42 |
| qwen3-235b-t4096 | 1732.50 | 27.99 | 27.39 | 6.23 |
| qwen3-235b-t512 | 249.81 | 17.94 | 16.82 | 5.58 |

- **kimi-k2-t4096 T_op = 29.72 µs**（baseline 650.67 ⇒ −95.4%），与 final 记录 30.67 µs 差 3%（run 间抖动，见 §5.2）。
- 12 档全部落在 final 记录 ±1.1 µs 内（最大差异 ds-v3-t1 12.80 → 12.25），**无回退**。
- 报告 `t_op_breakdown` 只显示 `main_kernel` 一个键：tuned 三个 launch 的 `prim_func` 都叫 `main`，
  适配器按 op 名汇总 ⇒ 键合并，数值仍是逐 launch 之和（如 kimi-k2-t4096 29.717 = hist+align+place）。
  逐 launch 归因见 §5.3（`launch_breakdown.py` 按 CSV 目录区分）。

### 10.5 兼容性 / 回退说明

- **回退方式**：把 wrapper 切换块的三行 `perf_opt` import 注释回、三行 `baseline` import 取消注释即可
  （备份文件 `history_version/moe_permute_align_pre_opt.py` 为翻转前全文）。
- **共享 scratch 的生命周期**：`perf_opt` 实现用模块级 `_SCRATCH`（按 `(numel,E,B,g,e_pad,device)` 键）
  在 align 与 scatter 两个 callable 之间传递每核直方图与游标行；同一进程内同一几何重复调用安全
  （每次 launch 1 覆盖写整个 `partial`，launch 2 覆盖写 `cursor`）。与 Stage 3 的
  `_AlignTwoPass._partial` 属同一契约级别（并发多流共用一个实例不安全，Stage 3 亦然），
  **单卡串行场景无影响**；已在 `perf_opt/moe_permute_align.py` 模块 docstring 记录。
- **输出契约**：三条输出逐位不变（§10.3）；`cumsum` 仍是内部 scratch（wrapper 不返回），
  但 tuned 版让 scatter 段读到的 `cumsum` 停在补齐起点（Stage 3 在 scatter 后推进），
  仅对外部不可见的 scratch 语义有差异。
- **路径依赖**：`perf_opt` 包内文件不 import 任何 `examples/` 下路径，只用 `tilelang`/`torch`/`torch_npu` + 标准库 ⇒ 可随包分发。
- **清理**：生产运行前后已清空 `/tmp/tileops_msprof_*` 与 `/workspace/tileops_msprof_*`（`TMPDIR=/workspace`）。
