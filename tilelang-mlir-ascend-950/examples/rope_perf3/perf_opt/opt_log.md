# `RoPEFwdOp` — Stage 4 调优日志（round 3：88.4% → 90% MBU 的尝试与实测边界）

- 任务：`examples/rope_perf3`（scenario=optimize, mode=full；平台 `ascend-950pr-9579` / `ascend`，单卡）
- 基线（只读）：`examples/rope_perf2/perf_opt/RoPEFwdOp.py`（round-2 产物，sha256 `162d7ed3…`）
- 设计：`examples/rope/_rope_positions_kernel/DESIGN.md`；生产集成包：`examples/TileOPs/tileops/kernels/rope/`
- **交付**：`perf_opt/RoPEFwdOp.py` = **与 round-2 基线逐字节相同**（sha256 `162d7ed3…`）。
  本轮构建并实测了 4 类候选（见 §3），**没有一个在 warm 生产 report 里超过 round-2**，
  故按任务约定「实测 HBM 边界」如实收束，不采纳更复杂的实现（详见 §4）。
- 口径：`msopprof --launch-count=1 --warm-up=0 --kernel-name=main` 包住**一次**算子调用，读 `OpBasicInfo*.csv`
  的 `Task Duration(us)`；本算子每次调用 1 个 launch。驱动 `perf_opt/experiments/bench_one.py`。
  MBU = roofline bytes / T_op / 1.6 TB/s。
- 环境：`TMPDIR=/workspace`、`TILELANG_CACHE_DIR=/tmp/opencode/tl-cache-opt-rope3`、cwd = 任务目录，
  `PYTHONPATH` 补 `examples/TileOPs`。

---

## 0. Phase 0 — 起点与目标

- 目标：主导 `llama3-8b-b2-2k` 从 **120.1 µs / 88.41% MBU** 到 **≤118 µs / ≥90%**（1.7% 空间）；
  prefill ≥70%、decode 不回退、34/34 契约不变。
- 起点逐块画像（round-2 产物，`profiles/baseline/base_r2_r1`）：`aiv_time 9.77 µs/块`、`aiv_vec_ratio 0.53`、
  `aiv_mte2_ratio 0.68` → **DMA 已是主项**，向量遍 5.2 µs 部分重叠。
- 先做**代价分解**（诊断分支 `r1-skip-table-probe`：cos/sin 换成常量，**不是合法 kernel**，只用于定位）：
  主导 case 113.1–113.4 µs（vs 基线 124.9）→ **cos/sin gather ≈ 11.5 µs（9%）**。
  该项由「每块 2 次相依赖的 GM 往返（positions → cos/sin）」构成，与块数成正比（640 块）。

## 1. 本轮测量方法学（重要）

本轮差异只有 1.7%，我采用**交错 A/B**（`experiments/ab_interleaved.py`，A/B 交替、每臂独立进程、每次采集一次调用；
平台的 `ab_test.py` 用 `msprof op --launch-count N`，与本任务的「一次采集一次调用」契约不同，故自建同协议 runner）。

**同时发现：本自建 harness 对本算子的「表 gather」偏冷**——单次调用进程里 cos/sin 表未驻留 L2，
导致 **窄布局（全表 gather 更多）在 harness 中被高估代价**。证据：同一 round-2 产物
harness 126.6 µs vs 生产 report 120.1 µs（+5.4%），而「宽布局」harness 121.3 µs / 生产 120.4 µs（+0.7%）。
→ **生产 report 才是裁决口径**；harness 只用于方向筛选（§5 的 VP-10）。

## 2. Phase 1 — Workload inventory 与 baseline

7 个 benchmark 案例，全部 `kind=tune`（无 smoke/skip），见 `workload_inventory.json`。

| workload | baseline（本轮 harness） | 生产 report（round-2，`20261125_123729`） |
|---|---:|---:|
| llama3-8b-b2-2k-float32 | 124.911 | **120.106**（88.4%） |
| llama3-8b-prefill-4k-bfloat16 | 78.379 | 76.060（69.8%） |
| llama3-8b-prefill-4k-float16 | 78.152 | 75.321（70.5%） |
| gptj-6b-prefill-float16 | 80.276 | 78.350 |
| phi-2-query-only-bfloat16 | 13.865 | 12.546 |
| llama3-8b-decode-b64-bfloat16 | 6.985 | 6.132 |
| gptj-6b-decode-b32-float16 | 5.980 | 4.967 |

## 3. Iteration Log

### Round 1 — 宽列块（单列 chunk）与表 gather 代价（诊断）

**候选优化点**：每个 row block 只切 **1 个列 chunk**（`cols = ���部 head`），使
①cos/sin 每行块只 gather 2 次（而非 4+1 次）、②每行 DMA 从 4 KB 变 16 KB 连续段、③块数 640 → 256。

| 候选 vs current best | workload | T_op(µs) | 说明 |
|---|---|---:|---|
| current best（round-2） | llama3-8b-b2-2k | 124.911 | — |
| `r1-skip-table-probe`（诊断） | llama3-8b-b2-2k | **113.287** | 仅定位：表 gather ≈ 11.5 µs |
| `r1-wide-hpc4096` | llama3-8b-b2-2k | **122.993** | 交错 A/B：126.57 → **121.30（−4.2%，3/3 胜）** |
| `r1-wide-hpc4096` | gptj-6b-prefill | 61.269 | −24% |
| `r1-wide-hpc4096` | **llama3-8b-decode-b64** | **12.665** | **回退 +81%**（256 块布局下 decode 只剩 4 块） |
| `r1-wide-hpc4096` | **gptj-6b-decode-b32** | **15.531** | **回退 +160%**（2 块） |

结论：宽列块对 prefill 有效但**饿死 decode**（块数不足）→ 需要「按 token 数选布局」。
同时实测块规模方向仍然无效：`step 64` 124.7、`step 16` 124.1（round-2 结构上，均无收益）。

### Round 2 — 双布局（按 token 数选择，运行期条件在 `@T.prim_func` 内）

**候选优化点**：同一 kernel 内两条布局路径 —— `tokens ≥ 2048` 走宽列块（1 chunk/侧），否则走窄列块（4+1 chunk/侧）；
grid 由 `T.if_then_else(tokens >= 2048, n_wide, n_narrow)` 决定（运行期表达式，不新增 Python 分派）。

| 候选 vs current best | workload | T_op(µs) | 备注 |
|---|---|---:|---|
| `r2-dual-layout` | llama3-8b-b2-2k | 120.633 | harness −3.4% |
| `r2-dual-layout` | **llama3-8b-decode-b64** | **11.425** | **仍回退 +64%**（虽走窄路径：两条内联路径使 7 µs 小 kernel 的二进制翻倍） |
| `r2-dual-layout` | gptj-6b-prefill | 61.067 | −24% |

结论：双路径的**代码体积**代价吃掉了 decode 的收益 → 两条路径不能共存于 16-bit 程序。

### Round 3 — 宽布局仅限 4-byte dtype（含按布局的子块高度）

**候选优化点**：`use_wide` 只在 **4-byte + NeoX + `half·itemsize == 256`**（即 llama 风格 fp32）时发射，
16-bit 程序保持单路径（decode 不受影响）；子块高度按布局给定（宽路径 1 整行/子块 → 16 KB 连续 DMA）。

| 候选 vs current best | workload | T_op(µs) | 备注 |
|---|---|---:|---|
| `r3-global-wide` | llama3-8b-b2-2k | 121.300 | harness −3.2% |
| `r3-global-wide` | llama3-8b-decode-b64 | 7.022 | 无回退 ✓ |
| `r3-global-wide` | llama3-8b-prefill-4k-bf16 | 79.517 | 无回退 ✓ |
| `r4-perlayout-tile` | llama3-8b-b2-2k | **121.259** | 最终候选形态 |

**精度**：该形态 L0 13/13、L1 7/7、L2 3/3 通过，但 boundary 出现 1 例失败
（`B-3b odd-head-65 gptj`：fp32+gptj+尾块，128/8580 元素错）→ 把宽路径限制到
`neox 且 half·itemsize == 256` 后 boundary 11/11 恢复；随后专门补测了
fp32-NeoX 的尾块（tokens=33/100）与分支边界（tokens=2048）均逐位正确（`max_diff=0`）。

### Round 4 — 生产 report 裁决（关键否定结果）

把 Round-3 形态装入 `…/rope_kernel/perf_opt/_rope_positions_kernel.py`，跑单算子 report（run-id
**`20261126_085009_745759_RoPEFwdOp`**，`status=passed`、correctness 34/34、`invalid_*`=0）：

| case | 宽布局候选（生产） | round-2 基线（生产） | Δ |
|---|---:|---:|---:|
| **llama3-8b-b2-2k** | **120.421** | 120.106 | **+0.26%（无收益）** |
| llama3-8b-prefill-4k bf16 | 76.951 | 76.060 | +1.2% |
| llama3-8b-prefill-4k fp16 | 75.518 | 75.321 | +0.3% |
| gptj-6b-prefill | 78.320 | 78.350 | −0.04% |
| phi-2-query-only | 13.073 | 12.546 | +4.2% |
| llama3-8b-decode-b64 | 5.920 | 6.132 | −3.5% |
| gptj-6b-decode-b32 | 4.499 | 4.967 | −9.4% |

**裁决**：harness 里 −4.2% 的宽布局收益在 warm 生产路径上**完全消失**（§1 的冷/热表解释），
其余差异都在 ±4% 噪声内。→ **不采纳**（不为 0 收益引入双路径复杂度与尾块风险），
`perf_opt/RoPEFwdOp.py` 保持 round-2 内核；生产包已还原（sha `162d7ed3…` 校验通过）。

### 中止条件

`plateau`（更严格说是「实测边界」）：本轮 4 类候选（宽列块、双布局、按 dtype 限定的宽布局、按布局子块高度）
在**权威的 warm 生产口径**上均未超过 round-2（最大 +0.26% / 最小 −0.04%），
且主导 case 120.1 µs = **1.415 TB/s = 实测 HBM 峰值（1.42–1.49 TB/s）的 95–99.6%**。
继续微调只会落在噪声内 → 按任务要求如实报告，不硬凑。

## 4. Final Performance Test Data

交付件 `perf_opt/RoPEFwdOp.py` 与 round-2 基线**逐字节相同**（sha256 `162d7ed3…`），
故 Final 表 baseline 与 final 为同一候选的同一组测量（harness，2–3 次独立采集）。

### Final Performance Test Data

| kernel_id | workload_id | baseline_us | final_us | candidate_id |
|---|---|---:|---:|---|
| tileops/kernels/rope/rope/rope_kernel/_rope_positions_kernel.py::_rope_positions_kernel::main | llama3-8b-b2-2k-float32 | 124.911 | 124.911 | final-round2-kept |
| tileops/kernels/rope/rope/rope_kernel/_rope_positions_kernel.py::_rope_positions_kernel::main | llama3-8b-prefill-4k-bfloat16 | 78.379 | 78.379 | final-round2-kept |
| tileops/kernels/rope/rope/rope_kernel/_rope_positions_kernel.py::_rope_positions_kernel::main | llama3-8b-prefill-4k-float16 | 78.152 | 78.152 | final-round2-kept |
| tileops/kernels/rope/rope/rope_kernel/_rope_positions_kernel.py::_rope_positions_kernel::main | gptj-6b-prefill-float16 | 80.276 | 80.276 | final-round2-kept |
| tileops/kernels/rope/rope/rope_kernel/_rope_positions_kernel.py::_rope_positions_kernel::main | phi-2-query-only-bfloat16 | 13.865 | 13.865 | final-round2-kept |
| tileops/kernels/rope/rope/rope_kernel/_rope_positions_kernel.py::_rope_positions_kernel::main | llama3-8b-decode-b64-bfloat16 | 6.985 | 6.985 | final-round2-kept |
| tileops/kernels/rope/rope/rope_kernel/_rope_positions_kernel.py::_rope_positions_kernel::main | gptj-6b-decode-b32-float16 | 5.980 | 5.980 | final-round2-kept |

### 4.1 与目标对照（生产 report 口径）

| 指标 | round-2（= 本轮交付） | 本轮目标 | 结论 |
|---|---:|---:|---|
| llama3-8b-b2-2k T_op | **120.1 µs** | ≤118 µs | 差 1.8%（未达） |
| llama3-8b-b2-2k MBU | **88.4%** | ≥90% | 差 1.6 pt（未达） |
| prefill-4k MBU | 69.8 / 70.5% | ≥70% | 达标（fp16）/ 边界（bf16） |
| decode | 6.13 / 4.97 µs | 不回退 | 达标 |

### 4.2 精度（交付件）

L0 **13/13**（4.77e-07）、L1 **7/7**（0）、L2 **3/3**（0）、boundary **11/11**（2.38e-07）；
生产路径 `pytest tests/ops/test_rope.py -q` → **34 passed**（`logs/production/pytest_test_rope.log`）；
op 层 drop-in 10/10。容差、dtype、输出契约未改。

### 4.3 为何 90% 不可达（量化）

- 120.1 µs = **1.415 TB/s**；平台实测峰值 1.424 TB/s（`mfu_baseline_v1.json` 的 bf16 `copy_` 512 MiB）
  与我方纯 `T.copy` 探针 1.49 TB/s → 当前实现是实测峰值的 **95.0–99.4%**。
- 90% MBU（118 µs）需 **1.44 TB/s** = 平台实测峰值的 101%（对 1.424）或 96.6%（对 1.49）；
  名义 1.6 TB/s 在这台机器上不可达（仪器化测量已三次复现 1.42–1.49）。
- 单块预算：640 块 / 56 核 → 每核 11.4 块；每块 256 KB（in+out）在 1.49 TB/s 聚合下需 9.6 µs，
  实测 10.5 µs → 非重叠开销仅 0.9 µs/块（表 gather + 旋转遍残差 + 同步）。

## 5. Skill Retrospective

### 5.1 流程发现

1. **自建 harness 与本算子生产口径存在系统性偏差（冷表）**：单次调用进程未预热 cos/sin 表，
   使「表 gather 多」的实现被系统性高估 4–5%。教训：**<5% 的候选裁决必须走生产 report**（或先在生产路径
   做 warm-up 注入），否则会采纳「harness 有效、生产无效」的结构（本轮差点如此）。建议写入 skill 的
   Phase 2 测量守则。
2. **双布局（运行期 token 分支）在小 kernel 上不可行**：两条内联路径让 7 µs 级 decode kernel 的二进制翻倍，
   +64% 退化。属于「构建期选择代码形态」的边界经验。
3. 95% 以上本就已经在实测带宽峰值附近时，参数级扫描（stage/step/子块/布局）全部落在噪声内 ——
   与 skill 的 plateau 判据一致，本轮据此收束。

### 5.2 价值点提案

| id | vp_type | 内容 | 证据三件套 |
|---|---|---|---|
| VP-10 | D | RoPE 类（表驱动逐元素）kernel 的 harness 冷热偏差：同一内核 harness 126.6 vs 生产 report 120.1（+5.4%）；差异全部来自 cos/sin 表的 L2 驻留。**裁决必须用生产 report** | evidence：§1 与 §3 Round 4 表、`profiles/experiments/r1_notable_r0`（113.3 = 去表代价）；repro：`perf_opt/experiments/bench_one.py` + `ab_interleaved.py`；toolchain_stamp 7a52f2f1… |
| VP-11 | D | 表 gather 在**冷**路径上 ≈ 11.5 µs / 9%（640 块，每块 2 次相依赖 GM 往返）；在 warm 生产路径上该代价被 L2 吸收，宽列块（gather 次数 −60%）**不产生**生产收益（120.42 vs 120.11） | evidence：§3 Round 4；repro：`experiments/r1_hpc4096.py`（宽）vs `baseline_r2.py` |
| VP-12 | D | 950PR 上本算子的实测边界：120.1 µs = 1.415 TB/s = 实测峰值 1.42–1.49 的 95–99%；90% 名义 MBU 需 1.44 TB/s，超出平台实测能力 | evidence：`mfu_baseline_v1.json`（1.424）、本轮纯 `T.copy` 探针（1.49）、三次 report 复现（120.1/120.4） |
| VP-13 | R | 建议 skill 增加：「当 current best 已达实测带宽 ≥95% 时，Phase 2 应先做一次『生产口径确认』再启动结构探索」，避免在 harness 偏差上构建无效结构 | evidence：本轮 Round 1–4 全过程 |

### 5.3 回写 pattern-library

本轮拟新增 `CONST-950-harness-warm-cold-table-gap`（自建 harness 与生产 report 的冷/热表偏差，含本算子的
双向实测）。**本轮未完成写入**（repro 需要在生产 harness 上做 warm 对照，属 Stage 5 报告链路），
记为待补；证据以 VP-10/VP-11 为准，由终态 evolver 按 D 类合入。

## 6. 生产采纳状态

- `perf_opt/RoPEFwdOp.py`（sha `162d7ed3…`）= round-2 内核，与生产包现状**一致**，无需翻转任何 wrapper。
- 本轮为裁决曾把宽布局候选装入 `…/rope_kernel/perf_opt/_rope_positions_kernel.py` 跑 report
  （run-id `20261126_085009_745759_RoPEFwdOp`），**随后已还原**（还原后 sha256 `162d7ed3…` 校验通过）。
- `perf_records.jsonl`：baseline 7 / candidate 13 / final 7 / merged 7（merged = 上述生产 report 的宽布局实测）。
- 未修改任何基线 kernel / op / manifest / test / bench / 集成包；`/tmp/tileops_msprof_*` 已清。

## 7. 原始证据索引

- 基线：`profiles/baseline/{base_r2_r0,r1,base_r2b_r*,base_gpre_r*,base_gdec_r*,base_dec_b64_r*}`
- 候选：`profiles/r1/*`（宽列块 + 去表诊断）、`profiles/r2/*`（双布局）、`profiles/r3/*`（按 dtype 限定）、`profiles/r4/*`（按布局子块）
- A/B 交错：`profiles/ab/pair{0,1,2}_{a,b}`（日志见 `logs/`；JSON 摘要：baseline 126.569 → candidate 121.304，−4.16%，3/3）
- 交付件：`profiles/final/final_<workload>_r*`
- 精度：`logs/precision/final_{L0,L1,L2,boundary}.log`、`logs/production/{pytest_test_rope.log,report_run.log}`
