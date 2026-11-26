# opt_log.md — `SSDChunkCouplingFwdOp` Stage 4 第二轮调优（83.2 µs → 55.0 µs）

- 任务：`examples/ssd_chunk_coupling_perf2`（scenario=optimize, mode=full，平台 `ascend-950pr-9579` / `ascend`）
- **只读基线（本轮输入）**：`examples/ssd_chunk_coupling_perf/perf_opt/SSDChunkCouplingFwdOp.py`
  （Stage-4 v1 行面板内核，sha256 `353dc196…615a`，生产报告 `20261124_151020` 实测 83.188 µs / MFU 61.79% / MBU 56.72%）
- 产物：`perf_opt/SSDChunkCouplingFwdOp.py`（v2，自包含，工厂签名不变，可直接替换 op-dir `perf_opt/_ssd_chunk_coupling_kernel.py`）
- 设计：`examples/ssd_chunk_coupling/_ssd_chunk_coupling_kernel/DESIGN.md`
- **测量口径**：`msopprof` kernel-only `Task Duration(us)`，`--launch-count=1 --warm-up=0 --aic-metrics=Roofline`，
  一次采集 = 一次算子调用（该算子每次调用只产生 1 个 kernel `main_kernel`，`bench_one.py` 逐行留 `per_op`
  证明 T_op = 全部 launch 之和）。工作目录 `examples/TileOPs`，`TMPDIR=/workspace`，
  `TILELANG_CACHE_DIR=/tmp/opencode/tl-cache-opt-ssd2`。

---

## 0. Phase 0 — 上下文与首轮必查项

- 工具链 / 硬件戳与上一轮一致（tilelang 0.1.15、CANN 9.2.1、cube 28 / vector 56）。
- 知识库条目在上一轮已逐条同口径重验（`knowledge/INDEX.md`、`traps-950pr-store-copy.md`、
  `gemm-tiles.md`、`execution-patterns.md`）；本轮新增实证见 §4 O-7~O-10，不需要重新推翻任何旧结论。
- **首轮必查项 1（向量化轴与布局）**：DESIGN §1.6.3 的布局决策在上一轮已按实测修订为
  「tile 类分派 epilogue」；本轮继续在该决策内细化 → 对角 band 用 64×64 子 tile，使
  **严格下三角子 tile 完全离开 AIV（fixpipe）**，把掩码元数从 `Q·BL` 降到 `Q·BL/2`。
- **首轮必查项 2（陷阱版本戳）**：沿用上一轮的实证（`CheckCompactL0Region` 零 origin 约束）
  —— 正因为 64×64 累加器本身零 origin，fixpipe 才能用于子 tile；这是本轮把对角 band 拆小的**前提**。
- **首轮必查项 3（DESIGN §1.6.4 裁决）**：上一轮已裁决（主选 trip-skip 胜出）；本轮不再重做。
- 上轮 `[DESIGN_LIMIT]` 的处置：上一轮的 `perf_feedback.md` 明确「未断言 51.4 µs 不可达」，
  本轮实测 55.0 µs（93.4% roofline）进一步印证该结论方向正确，**无需修正附录**；
  本轮**不产生**新的 `perf_feedback.md`（无设计层天花板，见 §5）。

## 1. Phase 1 — Inventory 与 Baseline

Inventory = `workload_inventory.json`（5 个参数化案例：3 tune + 2 smoke，来源同上一轮
`benchmarks/ops/bench_mamba.py::test_ssd_chunk_coupling_fwd_bench`）。
smoke（Q=64 → `Q//BL == 1`）走冻结的 Stage-3 回落工厂，代码未变，只做精度回归。

| kernel_id | workload_id | candidate | task_duration_us | 生产报告交叉核对 |
|---|---|---:|---:|---|
| coupling | mamba2-grouped-prefill-fp16 | baseline-v1-rowpanel | **84.284**（样本 83.256 / 84.284） | 83.188 ✓ |
| coupling | mamba2-780m-fp16 | baseline-v1-rowpanel | 8.472 | 8.504 ✓ |
| coupling | mamba2-2p7b-bf16 | baseline-v1-rowpanel | 16.284 | 16.038 ✓ |

排序：wl3（84.3 µs，占三案例总时 77%）→ wl2（16.3）→ wl1（8.5）。三个 tune workload 全部有结论。

**基线画像（`profiles/baseline/v0_base_wl3_r1`，288 块）**

| 资源 | 每 task | busy 合计 | 占 wall |
|---|---:|---:|---:|
| AIC | 5.151 µs | 53.0 µs | 63% |
| AIV | 6.949 µs | 71.5 µs | 85% ← 真瓶颈 |
| AIV vec | 4.001 µs | 41.2 µs | 掩码/零块逐元素 |
| AIV mte3 | 0.860 µs | 8.8 µs | UB→GM |
| GM 流量 | 75.5 MB（1.0R+1W，已无放大） | — | 0.90 TB/s（峰值 1.424 的 63%） |

结论：流量已到最小值，**不是带宽受限而是 AIV 逐元素 + 段前等 AIC 的停等**。

---

## 2. Iteration Log

### Round 1（从 baseline-v1 派生，每个分支只改一个主要点，wl3 上评测）

| candidate_id | 主要优化点（单一） | T_op (µs) | AIV vec/task | L0 | 结论 |
|---|---|---:|---:|---|---|
| current best → baseline-v1-rowpanel | — | 84.284 | 4.001 | pass | — |
| c-band | 掩码改**原始 fp32 位模式按位与**（`bits & (cond ? -1 : 0)`），语义与 `select` 逐位等价（`nan & 0 = +0.0`），但无分支 → 向量化 | 80.444 | — | 9/9 | 采纳（-4.6%） |
| c-diag64 | 对角 128-band 拆成 64×64 子 tile：**严格下三角→fixpipe（无 AIV）**、严格上三角→精确零块，只有 2 个对角子 tile 走掩码 | 73.985 | 2.06 | 9/9 | 采纳（+ -8%） |
| c-d64band | diag64 + band 组合 | **67.341** | 2.06 | 9/9 | 采纳（合计 -20.1%） |

c-diag64 的收益来源（实测）：掩码元数 `Q·BL → Q·BL/2`，AIV vec 4.001 → 2.058，AIC busy 53.0 → 47.5 µs；
代价：mte1（L1→L0 次数 3→7）0.569 → 0.760 µs、fixpipe 0.963 → 1.021 µs、AIV mte3 0.86 → 1.42 µs（store 变多变小）。

> `perf_records.jsonl` 收录口径：本轮所有候选（c-band / c-diag64 / c-d64band / c-pf）都跑过 L0，
> 全部入账；无「数值故意错误」的诊断探针，故无上一轮那类受控排除。

### Round 2（从 c-d64band 派生：消除段前停等）

| candidate_id | 主要优化点（单一） | T_op (µs) | AIV/task | AIC/task | L0 | 结论 |
|---|---|---:|---:|---:|---|---|
| current best → c-d64band | — | 67.341 | 5.499 | 4.62 | pass | — |
| **c-pf** | **面板按半幅拆成 4 个 L1 buffer**，第二半在 band0 之后才发搬入 → 缓冲区级依赖让尾部装入与 band0 计算重叠，消掉每块开头的 AIV 停等 | **55.545** | 4.453 | 3.576 | 9/9 | **采纳（-17.5%）** |

c-pf 的收益来源（实测，`profiles/experiments/r2_pf_wl3_r0`）：

| 指标 | c-d64band (67.3) | c-pf (55.5) | 说明 |
|---|---:|---:|---|
| AIC/task | 4.620 | 3.576 | AIC busy 47.5 → **36.8 µs** |
| AIV/task | 5.499 | 4.453 | AIV busy 56.6 → **45.8 µs** |
| AIV vec/task | 2.058 | 2.049 | 不变（元素量未变） |
| AIV mte3/task | 1.417 | 1.220 | 不变区间 |
| AIV 停等 | ≈1.7 | **≈1.1** | 面板首半 64 KB 即可启动 band0（原为 128 KB 全装完才开工） |

### 3. 参数邻域 / 未采纳方向（同一结构，仅一个参数变）

| 变体 | 参数 | T_op (µs) | 说明 |
|---|---|---:|---|
| r2_pf (zh=8) | 零块切片 8 行 | **55.545** | 采纳 |
| e_rp_z4/z16/z32（上一轮同族扫描） | 零块切片 4/16/32 行 | 83.0 / 84.0 / 85.2（v1 结构下） | 8 行为最优，本轮沿用 |
| 加宽 band 预取为 4 个四分幅 | 16 KB 首段 | 未采纳 | 首段 32 KB 已足够 band0 启动，收益边际 |

### 4. 关键实证（本轮新增，供 knowledge 回写）

- **O-7（P，正例，最大收益）**：**把"整 panel 一次装入"改成"按计算顺序分半装入"**，
  让首个 tile 只需首半即可启动 → 消除每块开头的 `AIC 装入 → AIV 首 epilogue` 停等。
  wl3 67.3 → 55.5 µs（-17.5%），AIV 停等 ≈1.7 → ≈1.1 µs，AIC busy 47.5 → 36.8 µs。
  机制：L1 分成 4 个独立 buffer（`cl1a/cl1b/bl1a/bl1b`）后，auto-schedule 的同步是**缓冲区级**的，
  于是 `cl1b/bl1b` 的 mte2 可与 `cl1a/bl1a` 的 mte1/mmad/双 copy 并行；若仍用两个整 panel buffer
  则同步粒度覆盖整块，装入无法与计算重叠。
- **O-8（P，正例）**：对角 band 用 **64×64 零 origin 累加器**后，严格下三角子 tile 可走 Cube fixpipe
  直存、完全离开 AIV；配合严格上三角子 tile 的零块，掩码元数减半（`Q·BL → Q·BL/2`）。前提是
  `CheckCompactL0Region` 只允许零 origin 的 L0C 源（上一轮 TRAP 实证），64×64 累加器天然满足。
- **O-9（D，常数）**：fp32 掩码用**位模式按位与**（`T.reinterpret(cv,"int32") & (-cast(cond,"int32"))`）
  与 `select` 逐位等价（`nan&0 = +0.0`、`inf&0 = +0.0`），且无分支 → 向量化生效，wl3 84.3 → 80.4 µs
  （同一结构下掩码段从"逐 lane 分支"变为"向量 and + vcvt"）。**这是唯一已知的"零掩码也精确"的快速写法**，
  优于被否决的 `x * mask`（`nan*0 = nan`）。
- **O-10（D，常数）**：本算子 AIV 逐元素段（SIMT VF）实测 ≈4.1–4.4 元素/cycle/AIV（无分支、float2 +
  半精度窄化）；有分支时降到 ≈2.5。故"元数 × 是否分支"是两条独立的优化轴。

### 5. 剩余空间与设计层判定（Phase 3）

| 指标 | 值 |
|---|---|
| final T_op（wl3） | **55.042 µs**（样本 54.585 / 55.042 / 55.206） |
| GM 流量 | 75.5 MB（= roofline_bytes，1.0R + 1W，无放大、无冗余） |
| 实测聚合带宽 | **1.372 TB/s = 实测峰值 1.424 TB/s 的 96.4%** |
| roofline 达成（理想 51.4 µs） | **93.4%** |
| MBU（75.5 MB / (T_op × 1.6 TB/s)） | **85.8%**（目标 ≥65% ✓，基线 56.72%） |
| MFU（vector, 47 TFLOP/s） | **≈93%**（基线 30.06% → 61.79% → 93.4%） |
| AIV busy / AIC busy | 45.8 µs（83%）/ 36.8 µs（67%） |

**结论：本算子在当前结构下已到硬件带宽屋顶**（1.372 / 1.424 = 96.4%；理想 51.4 µs 对应 1.47 TB/s，
高于实测峰值 1.424）。剩余 3.6 µs 需减少 GM 流量才能取得，而流量已等于 roofline_bytes（下三角输出必须写、
上三角必须写 0、C/B 各读一遍），**无设计层可回收空间** → 本轮**不产生** `perf_feedback.md`，
`[DESIGN_LIMIT]` 不再成立（上一轮的反馈已在 `examples/ssd_chunk_coupling_perf/perf_opt/perf_feedback.md` 留档）。

---

## 5. Final Summary

最终候选 `final-d64band-pf`：对角 band 64×64 子 tile 分解 + 位掩码 + 分半面板预取；
`Q // BL != 2` 的几何（smoke Q=64、Q=96、Q=128、Q=32）回落到冻结的 Stage-3 工厂，代码与数值不变。

### Final Performance Test Data

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---:|---:|---|
| tileops/kernels/mamba/ssd_chunk_coupling/ssd_chunk_coupling_kernel::_ssd_chunk_coupling_kernel::main | mamba2-grouped-prefill-fp16 | 84.284 | 55.042 | final-d64band-pf |
| tileops/kernels/mamba/ssd_chunk_coupling/ssd_chunk_coupling_kernel::_ssd_chunk_coupling_kernel::main | mamba2-780m-fp16 | 8.472 | 6.041 | final-d64band-pf |
| tileops/kernels/mamba/ssd_chunk_coupling/ssd_chunk_coupling_kernel::_ssd_chunk_coupling_kernel::main | mamba2-2p7b-bf16 | 16.284 | 11.234 | final-d64band-pf |

提升：wl3 **-34.7%（1.53x）**、wl1 -28.7%（1.40x）、wl2 -31.0%（1.45x）。
（对生产报告同口径 83.188 / 8.504 / 16.038 则为 1.51x / 1.41x / 1.43x。）
final 全部来自同一最终文件、同一 `candidate_id`，逐 workload 3 次采集取中位；
其余样本 wl3 54.585 / 55.206、wl1 6.063 / 6.038、wl2 11.285 / 10.716。
额外信息性采集（非 manifest 案例，不入 `perf_records.jsonl`）：grouped-prefill bf16 55.07 µs。

### 精度回归（容差未改：`_ATOL=_RTOL=1e-3` + 单 ulp 允许）

| 层 | 结果 | 日志 |
|---|---|---|
| L0 | **9/9 pass**，max_diff=0.125（bf16 1 ulp，与基线一致） | `logs/final/precision_all.log` |
| L1 | **8/8 pass**，max_diff=0.125 | 同上 |
| L2 | **4/4 pass** | 同上 |
| boundary | **10/10 pass**（zero-inputs、large-magnitude、noncontiguous、K 尾块、q96/bl32） | 同上 |

语义保持：对角子 tile 仍用**精确 select 等价**（位掩码，非 `x*0`），因此"上三角精确 0"契约对
`nan/inf/-0/subnormal` 输入仍然成立（`L0-3/L0-7` 的 `upper_nonzero=0` 与 `B-6 zero-inputs` 通过）。

### 回退 case

**无。** smoke-fp16 2.81→2.55/2.81 µs、smoke-bf16 2.78→2.64/2.78 µs（走 Stage-3 回落路径，代码未变，
差异在采集噪声内）；三个 tune workload 全部改善。

### 中止条件

`success`：达成用户目标（grouped-prefill ≤70 µs，实测 **55.0 µs**；MBU ≥65%，实测 **85.8%**），
三个 tune workload 全部有结论、winner 已合入单一工厂、smoke 精度通过、无回退。
实验用量：4 个有效候选 + 1 个参数邻域（≤ `max_experiments=30`），2 轮迭代（≤ `max_rounds=10`）。

### 生产采纳说明（本轮不改 TileOPs 树）

按要求本轮改动只写新 task 的 `perf_opt/`。`perf_opt/SSDChunkCouplingFwdOp.py` 是**自包含 drop-in**：
工厂符号、签名（`(batch, num_chunks, n_groups, chunk_len, d_state, dtype)`）、
`TUNED_ZERO_SLICE / TUNED_DEFAULT_CONFIG` 常量与 v1 完全一致，因此生产采纳仍是一次机械替换：
把本文件覆盖 `tileops/kernels/mamba/ssd_chunk_coupling/ssd_chunk_coupling_kernel/perf_opt/_ssd_chunk_coupling_kernel.py`
即可（wrapper 的双 import 切换块已指向该路径，无需再改 wrapper）。

---

## 5b. 生产采纳（round-2 kernel 覆盖 perf_opt → 集成包单算子报告验收）

### 采纳操作

| 项 | 值 |
|---|---|
| 备份（被替换的 v1） | `…/ssd_chunk_coupling_kernel/history_version/_ssd_chunk_coupling_kernel_pre_round2.py`，sha256 `353dc196…615a`（与覆盖前逐字节一致，diff 为空） |
| 覆盖目标 | `…/ssd_chunk_coupling_kernel/perf_opt/_ssd_chunk_coupling_kernel.py` |
| 覆盖源 | `examples/ssd_chunk_coupling_perf2/perf_opt/SSDChunkCouplingFwdOp.py`，sha256 `491dab80…c8b2`（覆盖后目标文件同哈希） |
| wrapper | **未改**（`from .ssd_chunk_coupling_kernel.perf_opt._ssd_chunk_coupling_kernel import _ssd_chunk_coupling_kernel` 已指向该路径） |
| `perf_opt/__init__.py` | **未改**（导出的 `_ssd_chunk_coupling_kernel / golden_ssd_chunk_coupling / TUNED_DEFAULT_CONFIG / TUNED_ROW_PANEL / TUNED_ZERO_SLICE` 在新文件中全部存在，已 AST 核对） |
| 未改动 | Stage 3 kernel 目录（`_ssd_chunk_coupling_kernel.py`、`__init__.py`）、op / manifest / test / bench |
| 回滚 | 用上述备份覆盖回 `perf_opt/_ssd_chunk_coupling_kernel.py` 即可 |

接线验证（运行时）：`SSDChunkCouplingFwdKernel(3,12,8,256,128,fp16)` 的 `kernel` 为 `JITKernel`；
wrapper 模块内 `_ssd_chunk_coupling_kernel.__module__` = `…perf_opt._ssd_chunk_coupling_kernel`，
其 `__file__` 的 sha256 = `491dab80…c8b2`（即 v2）；`TUNED_ROW_PANEL=True / TUNED_ZERO_SLICE=8 /
TUNED_DEFAULT_CONFIG={64,64,64}` 与 wrapper 的 `config` 一致。

### 生产回归（容差未改）

| 命令 | 结果 |
|---|---|
| `pytest tests/ops/test_mamba.py -q -k ssd_chunk_coupling` | **8 passed, 4 deselected**（`logs/production/pytest_coupling.log`） |
| `perf_opt/_ssd_chunk_coupling_kernel.py --level all` | **L0 9/9、L1 8/8、L2 4/4、boundary 10/10**，`0 FAIL`；max_diff **0.125**（bf16 1 ulp，与采纳前逐层一致）；`L0-3/L0-7 upper_nonzero=0`、`B-6 zero-inputs` 通过 → 精确零契约保持（`logs/production/precision_all.log`） |

### 集成包单算子报告（生产路径验收）

- run-id：**`20261125_183730_853046_SSDChunkCouplingFwdOp`**
- 产物：`examples/TileOPs/reports/tileops/20261125_183730_853046_SSDChunkCouplingFwdOp/{run.json,report.md,report.html}`
- **status = `passed`**；correctness 8/8 failed 0；`invalid_profile_count=0`、`kernel_filter_fallback_count=0`、
  `profiler_fallback_count=0`；每 case `captured_op_names=['main_kernel']`、`sample_count=1`、
  `t_op_us` = 全部 launch 之和（本例单 launch）。

| label | Stage-3 base | round-1 生产 | **round-2 生产 T_op** | vs round-1 | vs Stage-3 | **MFU** | **MBU** |
|---|---:|---:|---:|---:|---:|---:|---:|
| mamba2-780m | 12.643 | 8.504 | **6.160** | 1.38x | 2.05x | 46.36% | 42.56% |
| mamba2-2p7b | 22.910 | 16.038 | **11.808** | 1.36x | 1.94x | 48.37% | 44.40% |
| **mamba2-grouped-prefill** | 171.000 | 83.188 | **54.664** | **1.52x** | **3.13x** | **94.03%** | **86.32%** |

主导案例 54.66 µs（预期 ≈55 ✓）、MFU **94.03%**、MBU **86.32%**；实测聚合带宽
75.5 MB / 54.664 µs = **1.381 TB/s = 本机实测峰值 1.424 TB/s 的 97.0%**，
roofline（理想 51.4 µs）达成 **94.0%**。另两例较 round-1 继续改善 1.36–1.38x，**无回退**。
结构化记录：`perf_records.jsonl` 末尾 3 行（`phase=merged`, `production=true`, `round=4`，含 `run_id`、
`acceptance_mfu_percent`、`acceptance_mbu_percent` 与 `msprof_raw_path`）。

---

## 6. Skill Retrospective（按 vp_type 标注 + 证据三件套）

| # | vp_type | 结论 | evidence | repro | toolchain_stamp |
|---|---|---|---|---|---|
| 1 | **P** | **「整块一次装入」是混合 kernel 的隐藏停等来源**：把 L1 装入按计算顺序拆成多个独立 buffer，auto-schedule 的同步随即变成 buffer 级，尾部装入可与前段计算重叠（本任务 -17.5%，AIV 停等 1.7→1.1 µs） | `perf_records.jsonl` c-d64band 67.341 vs c-pf 55.545；`profiles/experiments/r1_d64_band_wl3_r2` vs `r2_pf_wl3_r0` 的 `aic_time/aiv_time` 分项 | `perf_opt/experiments/_d64.py`（整 panel）vs `_d64b.py`（分半）+ `_d64_final.py` | tilelang 0.1.15 / CANN 9.2.1 / a010a959… |
| 2 | **P** | 三角/掩码类算子：**把对角 band 拆到"最小零 origin 累加器"尺寸**，可让严格下三角子 tile 完全离开 AIV（fixpipe），掩码元数随子 tile 边长线性下降（`Q·BL`） | c-diag64 73.985 vs c-band 80.444；AIV vec 4.001→2.058 | `experiments/_d64.py` | 同上 |
| 3 | **D** | **位模式掩码**（`reinterpret(int32) & -cast(cond,int32)` → `reinterpret(float32)`）是与 `select` 逐位等价（`nan&0=+0.0`）且可向量化的零掩码写法，优于 `x*mask`（`nan*0=nan` 被证伪） | c-band 80.444 vs baseline 84.284；L0-3/L0-7 `upper_nonzero=0`、B-6 zero-inputs 通过 | `experiments/r1_band.py` + `_rp2.py` | 同上 |
| 4 | **D** | 950PR SIMT VF 逐元素段吞吐：**无分支 ≈4.1–4.4 元素/cycle/AIV**（float2 + 窄化），**有分支 ≈2.5**。元数与分支是两条独立优化轴 | 本轮 + 上一轮 profile（`aiv_vec_time/8192`） | `experiments/bench_one.py` 采集的 `PipeUtilization.csv` | 同上 |
| 5 | **R** | 流程建议：Stage 4 的"剩余空间判定"应显式比较 **实测聚合带宽 / 实测峰值带宽**（而非只比名义 roofline）——本任务 96.4% 即说明已到屋顶，可据此正当收束而不触发 `[DESIGN_LIMIT]` | 本日志 §5 | — | — |

流程问题记录：
- `bench_one.py` 的单次采集噪声约 ±1.5%（wl3 54.6–55.2 µs），故 winner 判定用 3 次采集中位；
  <5% 的候选差异（如 band vs baseline 的 4.6%）已在同结构下多次复现后再采纳。
