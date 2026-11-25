# opt_log.md — `SSDChunkCouplingFwdOp` Stage 4 性能调优日志

- 任务：`examples/ssd_chunk_coupling_perf`（scenario=optimize, mode=full, 平台 `ascend-950pr-9579` / `ascend`）
- 基线（只读）：`tileops/kernels/mamba/ssd_chunk_coupling/ssd_chunk_coupling_kernel/_ssd_chunk_coupling_kernel.py`
  （sha256 `47359c72e1f4c384b8fb6b7ea316e9ff9348cba4cc8488cabfcf4db7e8653882`）
- 设计：`examples/ssd_chunk_coupling/_ssd_chunk_coupling_kernel/DESIGN.md`（867 行，Stage 2 通过）
- 产物：`perf_opt/SSDChunkCouplingFwdOp.py`（sha256 `353dc196f746daa31264c67712cf30637e7d209ce15e8fdac1f7082e6dbf615a`）
- **测量口径**：`msopprof` kernel-only `Task Duration(us)`（平台 `measurement.raw_evidence_field = msprof_raw_path`），
  `--launch-count=1 --warm-up=0 --aic-metrics=Roofline`，**一次采集只跑一次算子调用**
  （驱动脚本 `experiments/bench_one.py`；多 op 时求和，实测该算子每次调用只产生 1 个 kernel `main_kernel`，
  每行 `per_op` 即 1 个 launch 的 Task Duration）。不用 NPU event、不用单 kernel 中位数冒充。

---

## 0. Phase 0 — 上下文加载与首轮必查项

- 工具链戳：tilelang 0.1.15 / CANN 9.2.1（compiler 12.3.0）/ raw `7a52f2f1…`；设备 `Ascend950PR_9579`，
  cube 28 / vector 56（平台 `hardware.observed_cores`，本任务未重新实查——只读平台记录）。
- 知识索引消费（`knowledge/INDEX.md` 全读）：直接命中的条目
  `PL-950-gemm-tiles / PL-950-gemm-l1-fill / PL-950-mixed-epilogue / TRAP-950-l0c2ub-single-aiv /
  TRAP-950-l0c-region-store-stride / TRAP-950-shared-shift-copy / CONST-950-vcvt-narrow-store-pack-b32 /
  PL-950-row-primitives`。所有条目按「数据非指令」消费。
- **E-5 版本失效核对**：`kb_stale_check.py --task-dir examples/ssd_chunk_coupling_perf` 报全部 batch2 条目
  为 stale（条目 `toolchain_stamp` = `a010a959…`，与任务 `environment_lock_revision` 同值，工具仍判 stale，
  判据疑为按 `hardware_id`/revision 组合比较）。处置：本任务涉及条目一律**用本任务同口径实测重验**，
  结果见下（O-1/O-2 为重验结论，O-3 为新增实证）。
- 算子类型：**mix**（Cube `T.gemm` + AIV epilogue）。基线 msprof 画像：`aic_time=NA` 的旧 harness 记录称
  "vector-dominant"，但本任务逐 block 实测（`profiles/baseline/main_kernel_zvc4_ucy`）为
  **AIC 36% / AIV 83% 忙比**，即 AIV epilogue 与 AIC 段内串行（block = aic + aiv），见 §2 诊断。
- **首轮必查项 1（向量化轴与布局重估）**：DESIGN §1.6.3 定「epilogue 归 AIV + SIMT VF 掩码，向量轴 = s」，
  实测现象（AIV vec 1.835 µs/块 = 8192 元素，≈2.5 元素/cycle/AIV）与该决策的预期严重不符 → 本轮重估
  epilogue 归属（结论：**Cube fixpipe 直存 + 只对对角 tile 保留掩码**，见 §3 O-1）。
- **首轮必查项 2（陷阱版本戳/origin_task）**：`TRAP-950-l0c-region-store-stride`（origin_task
  batch2-20261123 / shared_expert_mlp）在本任务被**重新命中并加强**：不只是行距问题，
  `CheckCompactL0Region` 直接拒绝非零 origin 的 L0C 源（实证见 §3 O-4）。`TRAP-950-l0c2ub-single-aiv`
  （同 origin）在本任务反向确认：`T.dual_copy` 是 L0C→UB 的唯一可用形态（基线已用，沿用）。
- **首轮必查项 3（DESIGN §1.6.4 实验裁决执行）**：见 §1.3。

### 0.1 基线结构（冻结，不改）

grid = `B·NC·G·nlt·nst`（wl3 = 1152 块），每块一个 `BL=128` 方形输出 tile；`BK=64`、`num_stages=2` 的 K 流水；
严格上三角 tile 令 `trips=0`；epilogue = `dual_copy(acc→cv)` + `SIMT VF 掩码+cast` + `dual_copy(co→GM)`。

---

## 1. Phase 1 — Workload Inventory 与 Baseline

### 1.1 Inventory

来源：`benchmarks/ops/bench_mamba.py::test_ssd_chunk_coupling_fwd_bench`，`_SSD_CHUNK_COUPLING_BENCH_PARAMS`
由 `tileops.manifest:SSDChunkCouplingFwdOp` 展开 = **5 个参数化案例**，完整清单见 `workload_inventory.json`。

| workload_id | shape (C=B) | chunk | dtype | kind |
|---|---|---|---|---|
| ssd-chunk-coupling-smoke-fp16 | [1,128,1,64] | 64 | fp16 | smoke（manifest label 含 smoke + pytest `smoke` 标记） |
| ssd-chunk-coupling-smoke-bf16 | [1,128,1,64] | 64 | bf16 | smoke |
| mamba2-780m-fp16 | [1,4096,1,128] | 256 | fp16 | **tune** |
| mamba2-2p7b-bf16 | [4,2048,1,128] | 256 | bf16 | **tune** |
| mamba2-grouped-prefill-fp16 | [3,3072,8,128] | 256 | fp16 | **tune** |

smoke 只做精度回归（不做性能采集/调优）；无 skip 案例。kernel_id 唯一：
`tileops/kernels/mamba/ssd_chunk_coupling/ssd_chunk_coupling_kernel::_ssd_chunk_coupling_kernel::main`
（captured op name 恒为 `main_kernel`，Block Dim 1152 / 64 / 128，每次调用 1 个 launch）。

### 1.2 Baseline（round 0，每 workload median of 2 captures）

| kernel_id | workload_id | candidate | task_duration_us | profile |
|---|---|---|---:|---|
| coupling | mamba2-grouped-prefill-fp16 | baseline-v3 | **173.076** | `profiles/baseline/v0_base_wl3_r1` |
| coupling | mamba2-780m-fp16 | baseline-v3 | 13.208 | `profiles/baseline/v0_base_wl1_r1` |
| coupling | mamba2-2p7b-bf16 | baseline-v3 | 23.041 | `profiles/baseline/v0_base_wl2_r1` |

交叉核对：Stage 5 历史报告（`reports/tileops/20261123_135638_847395_SSDChunkCouplingFwdOp/run.json`，
同口径单次采集）为 171.060 / 12.643 / 22.910 µs —— 与本轮复现在 0.4–1.3% 内一致。
`examples/batch2/mfu_baseline_v1.json` 记录的主导案例 T_op 171.0 µs、MFU(vector) 30.06%。

**排序（Phase 1 实测后）**：wl3 (173.1 µs，占三案例总时的 82%) → wl2 (23.0) → wl1 (13.2)。
排序只影响先后，三个 tune workload 全部有结论。

### 1.3 设计估算 vs 实测（D-2 回填）与 DESIGN §1.6.4 实验裁决

DESIGN §1.6.0 估算行：wl3 下界 ≈ 44.6 µs（按 1.6 TB/s 未实测假设的流量项）+ 启动代价。
实测 baseline 173.1 µs ⇒ **偏差 3.9x**，失准项为**流量（1.5R 重读 + strided 128 B burst 效率）与
AIV epilogue 发射**两项，而非容量项。

**DESIGN §1.6.4 实验裁决计划执行结果**（主选 trip-skip vs 备选 B 固定 trips）：

| 项 | 值 |
|---|---|
| 备选 B 实现 | `experiments/e_altB.py`（`trips = k_trips` 恒定，上三角 tile 算了再丢弃） |
| 主选（baseline-v3） | 173.076 µs |
| 备选 B | **180.913 µs**（样本 180.626 / 180.913），L0 9/9 通过 |
| 判定阈值 | 备选 B < 主选 × 0.95 → 翻转；实测比值 1.045 |
| **裁决** | **主选（trip-skip）胜出，保留**；反解"上三角 tile 的 GEMM+搬入代价"≈ 7.8 µs（占 wl3 4.5%），与流量项 18.9 MB ÷1.6 TB/s ≈ 11.8 µs 同量级 ✓（该 7.8 µs 现由 row-panel 结构完全消除，因为上三角 tile 不再发射 GEMM） |

---

## 2. Round 1 — 现象诊断（基于 baseline 逐 block profile）

`profiles/baseline/main_kernel_zvc4_ucy`（PipeUtilization 全 1152 块聚合）：

| 指标 | 值 | 解读 |
|---|---:|---|
| AIC busy 总计 | 1152 × 1.513 / 28 = **62.3 µs**（36% of 173） | 不饱和 |
| AIC 分项/块 | mte2 0.760 / cube 0.233 / mte1 0.200 / scalar 0.186 / fixpipe 0.185 | GM→L1 最大但仅占 1.5 µs |
| AIV busy 总计 | 2304 × 3.482 / 56 = **143.3 µs**（83% of 173） | **真瓶颈** |
| AIV 分项/task | vec 1.835 / mte3 0.309 / scalar 0.166，其余 ≈1.17 为等 AIC 的 wait | 掩码向量段 + 段内串行 |
| GM 流量 | 读 55.3 MB（1.5R）+ 写 37.7 MB = 93.0 MB → **0.54 TB/s** | 仅为实测峰值 1.424 TB/s 的 38% |

结论：**不是带宽受限，是 (a) AIV 逐元素掩码吞吐 ~2.5 元素/cycle/AIV、(b) 同块 AIC 与 AIV 串行**。
两个候选方向由此派生。

---

## 3. Iteration Log

### Round 1（从 baseline-v3 派生 4 个分支，每个只改一个主要点）

| candidate_id | 主要优化点（单一） | T_op (wl3, µs) | AIV vec/task | AIC busy% / AIV busy% | L0 | 结论 |
|---|---:|---:|---:|---|---|
| **current best → baseline-v3** | — | 173.076 | 1.835 | 36% / 83% | pass | — |
| c-p1-mask-mul | 掩码改 `x·cast(cond)` 让向量化生效 | 155.440 | — | — | **不采纳** | Δ -10%；语义不等价（`nan·0=nan` 破坏"上三角精确 0"契约，DESIGN §1.6.1 E3 已证伪），仅作诊断 |
| c-v2-cast-only | 去掉掩码只留 cast（诊断上限） | 137.304 | — | — | **不采纳** | 给出"cast 本身 1.15 µs/块"的代价，说明瓶颈不只是谓词 |
| c-ul | **epilogue 按 tile 类分派**：严格下三角→Cube fixpipe 直存；严格上三角→零块；对角→保留原掩码 | **136.577** | 1.055 | 47% / 76% | pass (9/9) | **采纳**（-21.1%） |
| c-ul-bl64 | BL=64 tile（对角元素量减半） | 311.826 | — | — | **fail** (L0-6) | 否决：块数 2880 使每块固定代价主导，且 L0 失败 → 整档 BL 排除 |

c-ul 关键发现：把"无需掩码的 tile"从 AIV 逐元素段里拿掉，等价于 `v2_cast_only` 的全部收益（136.6 vs 137.3），
说明**剩余成本全部来自对角 tile 的掩码段 + AIC/AIV 串行**。

> **`perf_records.jsonl` 收录口径说明（append-only 的受控例外，留痕）**：gate 4 的
> `S4-PERF-RECORDS-SCHEMA` 要求每条记录 `l0_pass is True`，因此**数值故意不正确或 L0 未通过的诊断探针**
> （`c-p1-mask-mul` 155.440、`c-v2-cast-only` 137.304、`c-ul-bl64` 311.826）不在 JSONL 中，其测量值、
> 原始采集目录与否决理由完整保留在本表与 §3.2 / 第 2 节邻域表中。JSONL 只承载可作为 candidate/winner 的
> 分支（baseline 3 行 + c-ul + c-rp + c-rp-z8 + final 3 行 = 9 行）。

### Round 2（从 c-ul 派生，改结构：row panel + 双累加器）

| candidate_id | 主要优化点（单一） | T_op (wl3, µs) | AIV busy% | AIC busy% | L0 | 结论 |
|---|---:|---:|---:|---:|---|---|
| current best → c-ul | — | 136.577 | 76% | 47% | pass | — |
| c-rp | **一块 = 一个 (b,c,g) 行面板**：C/B 整 panel 一次进 L1（1.0R），3 个三角 tile 块内串行、两个 L0C 累加器交替（下一 tile 的 GEMM 覆盖上一 tile 的 AIV epilogue） | **90.204** | 86% | 58% | pass (9/9) + L1/L2/boundary 全绿 | **采纳**（-34.0%） |
| c-rp-z8 | 上三角零块由 `BL/8` 行零缓冲多次 `dual_copy` 生成（原为整 `BL/2` 行 SIMT fill，1.4 µs/块） | **82.541** | 86% | 63% | 与最终文件同一代码路径（final L0 9/9） | **采纳**（-8.5%） |

c-rp 的收益拆分（用 v1_fixpipe_floor 计量）：纯 Cube、无 AIV 段的对照 = 86.8 µs；c-rp 的 90.2 µs
已基本等于该地板，说明 **AIV epilogue 的串行代价已被双累加器覆盖掉**。

#### Round 2 参数邻域（同一结构，only 一个参数变，L0 未逐点复跑——同一代码路径，最终文件覆盖）

| 变体 | 参数 | T_op (wl3, µs) |
|---|---|---:|
| e_rp | 零缓冲 = BL/2 行（= 原 fill） | 90.204 |
| e_rp_z32 | 零缓冲 32 行 | 85.166 |
| e_rp_z16 | 零缓冲 16 行 | 83.991 |
| **e_rp_z8** | **零缓冲 8 行（采纳）** | **82.541** |
| e_rp_z4 | 零缓冲 4 行（copy 次数 ×16） | 83.025 |
| e_rp_selhalf | 掩码在输出 dtype 上 select | 97.755（回退，否决） |
| e_rp_mul | `x·mask` 乘法掩码 | 78.709（更快但**语义不等价**，不采纳） |
| e_strip_nd | 对角 tile 走 16 行条带（fixpipe 分段） | 382.536（回退，否决，见 O-4） |
| e_strip / e_strip32 | 同上 strip=16/32 | 编译失败（L0C 源 region 必须零 origin） |

### 4. 关键实证（本轮新沉淀，供 knowledge 回写）

- **O-1（PL，正例）**：`T.copy(l0c_acc, gm_region)`（全 tile、零 origin）在 mixed kernel 里由 **Cube fixpipe**
  执行 fp32→dtype 直存，`aic_fixpipe_time ≈ 0.39 µs/128×128 tile`，比"L0C→UB→AIV cast→UB→GM"路径
  （vec 1.835 + mte3 0.309）便宜 **5.5x**。适用边界：**只能写"无需逐元素修改"的 tile**。
- **O-2（PL，正例）**：`dual_copy` 的 2:1 约束是**目的 region 的整轴 extent = 源 buffer 整轴 extent × 2**；
  想用小零缓冲多次搬运必须让目的 region 为 `2×ZH` 行（本任务 ZH=8 → 8 次 16 行搬运）。
- **O-3（TRAP 加强）**：`CheckCompactL0Region` 拒绝**非零 origin** 的 L0C 源
  （`Split/non-zero-origin L0 tiles are not supported`），所以"按行条带切 L0C 上三角"不可实现；
  尾轴截断（`acc[0:BL, 0:W]`）形态未在本任务验证。
- **O-4（TRAP）**：`T.clear/T.fill` 对 `shared.l0c` 直接报 `Unsupported scope shared.l0c`；
  对 `shared`(UB) 可用但代价高（在 c-rp 上把 8192 元素 fill 换 `T.clear` 使 wl3 从 90 µs 退化到 383 µs）。
- **O-5（PL，正例）**：同一 block 内**双 L0C 累加器 + Python 级展开的 tile 序列**即可让 auto-schedule 把
  tile i 的 AIV epilogue 与 tile i+1 的 GEMM 重叠，无需手写 flag（block 时间从 `aic+aiv` 变 `max(aic,aiv)`）。
- **O-6（CONST）**：SIMT VF 逐元素段吞吐实测 ≈ **2.5 元素/cycle/AIV**（分支掩码）与
  **4.3 元素/cycle/AIV**（无分支 cast，codegen 为 `float2/half2`）；`threads=128` 正确，`threads=64` 使
  wl3 从 173 µs 退化到 243 µs。

---

## 5. Final Summary

最终候选：`final-rp-z8` = `perf_opt/SSDChunkCouplingFwdOp.py`
（row panel + 双 L0C 累加器 + tile 类分派 epilogue + 8 行零缓冲）；与基准同一工厂签名、
`Q // BL != 2` 的几何（smoke Q=64、Q=96、Q=128、Q=32）**回落到冻结的 Stage-3 工厂**，代码与数值不变。

### Final Performance Test Data

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---:|---:|---|
| tileops/kernels/mamba/ssd_chunk_coupling/ssd_chunk_coupling_kernel::_ssd_chunk_coupling_kernel::main | mamba2-grouped-prefill-fp16 | 173.076 | 82.919 | final-rp-z8 |
| tileops/kernels/mamba/ssd_chunk_coupling/ssd_chunk_coupling_kernel::_ssd_chunk_coupling_kernel::main | mamba2-780m-fp16 | 13.208 | 8.411 | final-rp-z8 |
| tileops/kernels/mamba/ssd_chunk_coupling/ssd_chunk_coupling_kernel::_ssd_chunk_coupling_kernel::main | mamba2-2p7b-bf16 | 23.041 | 16.550 | final-rp-z8 |

提升幅度：wl3 **-52.1%（2.09x）**（`profiles/final/SSDChunkCouplingFwdOp_wl3_r1`）、wl1 -36.3%（1.57x）
（`profiles/final/SSDChunkCouplingFwdOp_wl1_r1`）、wl2 -28.2%（1.39x）（`profiles/final/SSDChunkCouplingFwdOp_wl2_r1`）。
final 全部来自同一最终文件、同一 `candidate_id`，逐 workload 2 次采集取中位；另一采集样本：
wl3 82.507、wl1 8.366、wl2 15.962。

**主导案例指标（升幅最大项）**

| 指标 | baseline | final | 说明 |
|---|---:|---:|---|
| T_op (mamba2-grouped-prefill fp16) | 171.06 µs（历史同口径） | **82.5–82.9 µs** | 2.06–2.09x |
| MFU(vector, 47 TFLOP/s) | 30.06% | **≈62%** | achieved 14.13 → ≈29.3 TFLOP/s |
| roofline 达成（理想 51.4 µs） | 30.1% | **62.3%** | `examples/batch2/mfu_baseline_v1.json` 口径 |
| GM 流量 | 93.0 MB（1.5R+1W） | **74.6 MB（1.0R+1W）** | row panel 消除 tile 重读 |
| 实测带宽 | 0.54 TB/s | **0.90 TB/s**（峰值 1.424 的 63%） | |
| AIV busy | 143.3 µs（83%） | 71.1 µs（86%） | AIV vec 45.2→41.2 µs |
| AIC busy | 62.3 µs（36%） | 52.5 µs（63%） | 块数 1152→288 |

**回退 case**：无。smoke-fp16 2.762→2.769 µs（+0.3%，噪声内，且 smoke 走 Stage-3 回落路径，代码未变）；
smoke-bf16 2.81→2.687 µs（-4.4%）。额外信息性采集（非 manifest 案例，不入 perf_records）：
grouped-prefill bf16 82.78 µs。

### 精度回归

| 层 | 结果 |
|---|---|
| L0 | **9/9 pass**，max_diff=0.125（bf16 1 ulp，与 Stage-3 基线完全一致） |
| L1 | **8/8 pass**，max_diff=0.125 |
| L2 | **4/4 pass**，max_diff=0.125 |
| boundary | **10/10 pass**（含 zero-inputs、large-magnitude、noncontiguous、K 尾块、q96/bl32） |

容差未改（`_ATOL=_RTOL=1e-3` + 单 ulp 允许，与 Stage-3 相同）。

### 中止条件

`success`：主导案例超出"降 20–30%"目标（实降 52%），三个 tune workload 全部有结论、winner 全部核内合并、
smoke 精度通过、无回退。实验用量：8 个有效候选分支 + 6 个参数邻域/诊断探针（≤ `max_experiments=30`），
3 轮迭代（≤ `max_rounds=10`）。

### 设计层天花板判定（Phase 3）

剩余差距：82.5 vs roofline 51.4 µs（62%）。剩余限制为 (a) 对角 tile 的 AIV 逐元素掩码（41.2 µs AIV vec，
受 SIMT VF 吞吐 2.5–4.3 元素/cycle/AIV 限制）与 (b) 0.90 TB/s 的带宽利用率。
其中 (a) 直接来自 **DESIGN §1.6.3 布局决策 #5（epilogue 归 AIV，SIMT VF 掩码）**，且本轮实测已推翻
DESIGN §5.5「不做核内串行/persistent」与 §1.6.3 #4「fixpipe 直存必须二次遍历清零」两条否决依据
（实测 2.07x）。判定结果见 `perf_opt/perf_feedback.md`。

---

## 5b. 生产采纳（wrapper 翻转 → 集成包单算子报告验收）

### 采纳操作（唯一允许的 wrapper 修改：切换块注释机械翻转）

| 项 | 值 |
|---|---|
| wrapper | `examples/TileOPs/tileops/kernels/mamba/ssd_chunk_coupling/ssd_chunk_coupling.py` |
| 备份 | `ssd_chunk_coupling_kernel/history_version/ssd_chunk_coupling_pre_opt.py`（sha256 `c069f5afa08c0e17…b8f0342`，与翻转前逐字节一致） |
| 翻转后 | sha256 `3c84e4a26120995e…`；`diff` 仅 2 行（baseline import 加 `#`、perf_opt import 去 `#`），无其他改动 |
| 激活源 | `from .ssd_chunk_coupling_kernel.perf_opt._ssd_chunk_coupling_kernel import _ssd_chunk_coupling_kernel` |
| 回滚 | 把上述两行注释换回即可（备份文件已留档） |
| 默认参数配对 | **无需**：`perf_opt` 的 `TUNED_DEFAULT_CONFIG = {block_l:64, block_s:64, block_n:64}` 与 wrapper `default_config` 完全相同（运行时已断言相等），故切换块内无成对默认参数需改 |
| 未改动 | Stage 3 kernel 目录（`_ssd_chunk_coupling_kernel.py`、`__init__.py`）、op / manifest / test / bench 全部未动 |

接线验证（运行时）：`SSDChunkCouplingFwdKernel(3,12,8,256,128,fp16)` 的 `kernel` 为 `JITKernel`，
wrapper 模块内 `_ssd_chunk_coupling_kernel.__module__` ==
`tileops.kernels.mamba.ssd_chunk_coupling.ssd_chunk_coupling_kernel.perf_opt._ssd_chunk_coupling_kernel`；
`SSDChunkCouplingFwdOp(256).default_kernel_map["ssd_chunk_coupling_fwd"].__module__` == 集成 wrapper。
**op 层无需适配**（`SSDChunkCouplingFwdKernel.__init__/forward` 的签名与 factory 调用形状不变）。

### 生产回归（`tests/ops/test_mamba.py`，容差未改）

| 命令 | 结果 |
|---|---|
| `pytest tests/ops/test_mamba.py -q -k ssd_chunk_coupling` | **8 passed, 4 deselected** |
| `pytest tests/ops/test_mamba.py -q`（整文件） | 8 passed, **4 failed** —— 全部是 `test_ssd_chunk_scan_fwd[*]`，属**另一个算子** `SSDChunkScanFwdOp` 的既有失败：`AttributeError: type object 'PassConfigKey' has no attribute 'TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION'`（其 perf_opt 使用的 pass_config 键在本工具链 tilelang 0.1.15 中不存在）。**A/B 已证与本次采纳无关**：把 wrapper 还原成 pre-opt 备份后同样 4 failed（同 4 个 case id），恢复翻转后仍 8 coupling passed |
| 分层精度（`perf_opt/SSDChunkCouplingFwdOp.py --level all`） | L0 9/9、L1 8/8、L2 4/4、boundary 10/10，max_diff=0.125（bf16 1 ulp，与基线一致）；日志 `logs/final/precision_L0_L1_L2_boundary.log` |

> 既有失败仅记录、不在本次范围内修复（跨算子，需另一任务处理 `SSDChunkScanFwdOp` 的 pass_config 兼容）。

### 集成包单算子报告（生产路径验收）

- run-id：**`20261124_151020_937387_SSDChunkCouplingFwdOp`**
- 产物：`examples/TileOPs/reports/tileops/20261124_151020_937387_SSDChunkCouplingFwdOp/{run.json,report.md,report.html}`
- 命令（工作目录 `examples/TileOPs`）：
  `TILEOPS_TASK_DIR=examples/ssd_chunk_coupling TMPDIR=/workspace python -m tileops.reporting.cli run --op SSDChunkCouplingFwdOp --prof-mode msprof --pytest-arg=-k --pytest-arg=ssd_chunk_coupling`
  （`--pytest-arg=-k …` 用于把 correctness 限定到本算子；不加时整文件会把上面 4 个无关既有失败计入 → `status=failed`，属 harness 选择器问题，非得数问题）
- **status = `passed`**；correctness 8/8、failed 0；`invalid_profile_count=0`、`kernel_filter_fallback_count=0`、`profiler_fallback_count=0`；
  每个 case `captured_op_count=1`、`sample_count=1`、`t_op_us` = 全部 launch 之和（本例单 launch）。

| label | baseline T_op (µs) | **生产 T_op (µs)** | 变化 | baseline MFU | **生产 MFU** |
|---|---:|---:|---:|---:|---:|
| mamba2-780m | 12.643 | **8.504** | -32.7% | 22.6% | **33.58%** |
| mamba2-2p7b | 22.910 | **16.038** | -30.0% | 24.9% | **35.61%** |
| mamba2-grouped-prefill | 171.000 | **83.188** | **-51.3%** | 30.06% | **61.79%** |

主导案例 83.19 µs（预期 82–83 ✓）、MFU 61.79%（预期≈62.7%，差 1.5%，为单次采集波动，与 Stage 4
`82.919 µs / 62.7%` 一致）。三案例全部改善，**无回退**。结构化记录：`perf_records.jsonl` 末尾 3 行
（`phase=merged`, `production=true`, `round=4`，携带 `run_id` 与 `msprof_raw_path`）。

---

## 6. Skill Retrospective（按 vp_type 标注 + 证据三件套）

| # | vp_type | 结论 | evidence | repro | toolchain_stamp |
|---|---|---|---|---|---|
| 1 | **P** | 「同块 AIC/AIV 串行」这类 epilogue 与 GEMM 同块串行的算子，**双片上累加器 + Python 级展开的多 tile 序列**即可让 auto-schedule 重叠，不必手写跨核 flag（block 时间 aic+aiv → max）。已在 ssd_chunk_coupling 生效 136.6→90.2 µs | `perf_records.jsonl` c-ul / c-rp 两行 + `profiles/experiments/e_ul_wl3_r0` vs `e_rp_wl3_r1` 的 `aic_time/aiv_time` 分项 | `perf_opt/experiments/_rp.py`（最小 before/after：`_epi.py` `EPILOGUE_CLASS_UL` 单 tile 版 vs `_rp.py` 双累加器版） | tilelang 0.1.15 / CANN 9.2.1 / a010a959… |
| 2 | **D** | Cube fixpipe 全 tile 直存（fp32→fp16）比 L0C→UB→AIV cast→GM 便宜 5.5x（0.39 vs 2.14 µs/128×128 tile） | `profiles/experiments/v1_fixpipe_floor_wl3_r1`（86.81 µs 全 fixpipe）vs `v0_base_wl3_r1`（173.08） | `experiments/v1_fixpipe_floor.py` + `_epi.py` | 同上 |
| 3 | **D** | L0C 源 region 必须零 origin（`CheckCompactL0Region`），行条带式 L0C 上三角不可实现；`T.clear` 对 `shared.l0c` 不支持、对 UB 极慢（8k 元素 fill 使 90→383 µs） | `experiments/e_strip.py` 编译错误原文；`e_strip_nd` 382.5 µs | `experiments/_epi2.py`（含被拒形态） | 同上 |
| 4 | **R** | 流程建议：Stage 4 Phase 0「首轮必查项」应加一条**「上/下三角或掩码类算子，先按"是否需要逐元素修改"给 tile 分类，再决定 epilogue 归属引擎」**——本任务按此一步就拿到 21%，而按 DESIGN §1.6.3 的原决策会在 AIV 段浪费 2x | 本日志 §2/§3；DESIGN §1.6.3 决策表 | — | — |
| 5 | **C** | 案例索引建议：`CASE-ssd-chunk-coupling-rowpanel`（同族 ssd_chunk_scan 的 persistent 任务流水 + 本算子的行面板 + 双累加器），供 mamba/SSD 族后续任务直接参照 | 本 opt_log + `perf_feedback.md` | — | — |

流程问题记录：
- `kb_stale_check.py` 把与任务同 `toolchain_stamp` 的全部条目判为 stale，无法用于"哪些结论需要重验"的
  机械筛选（本任务只能靠逐条同口径实测重验）。
- harness `bench_kernel_msprof` 实际发出 `--launch-count=10 --warm-up=5`，而生成的脚本只调用算子 **1 次**；
  两者语义不一致（依赖 profiler 容错才得到 1 个样本）。本任务用 `experiments/bench_one.py` 显式固定为
  `--launch-count=1 --warm-up=0`，并在每行记录 `per_op` 以证明"一次调用 = 一个 launch"。
