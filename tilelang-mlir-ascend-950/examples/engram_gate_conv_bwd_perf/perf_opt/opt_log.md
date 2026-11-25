# opt_log.md — `EngramGateConvBwdOp` Stage 4 性能调优日志

- 任务：`examples/engram_gate_conv_bwd_perf`（scenario=optimize, mode=full, 平台 `ascend-950pr-9579` / `ascend`，单卡 0）
- 基线（只读）：`tileops/kernels/sequence_modeling/engram_gate_conv_bwd/engram_gate_conv_bwd_kernel/_engram_gate_conv_bwd_kernel.py`
  （sha256 `0d5bb90bba68eda7ea10db09c0e06b27eeb3717bcd61e990ea1d34af6db13d22`，与 `perf_records.jsonl`
  round 0 记录的 `artifact_sha256` 一致）
- 设计：`examples/engram_gate_conv_bwd/_engram_gate_conv_bwd_kernel/DESIGN.md`
- 产物：`perf_opt/engram_gate_conv_bwd.py`（= `perf_opt/EngramGateConvBwdOp.py`，两者字节相同，
  sha256 `65c3133c38ec86ba26db72e90d620c2c4cac8c079b04e1a211fde0b11668ec81`）
- **测量口径**：任务平台 `measurement` 规定的 `msopprof` kernel-only `Task Duration(us)`
  （`raw_evidence_field = msprof_raw_path`），一次采集只跑一次算子调用。本算子**一次调用 2 个 launch**
  （融合计算 kernel + 有序归约 kernel，两者同名 `main_kernel`，train-s96 的 Block Dim 分别为 48 / 36），
  平台主指标 `T_op` 按 `mfu_baseline_v1.json` 的 `policy.t_op` 定义 = **一次调用内所有 kernel launch 之和**，
  因此 `--launch-count=2` 一次采到两个 launch 并**求和**；驱动 `perf_opt/experiments/bench_one.py`。
- 本算子 T_op 在 8–15 µs 量级、单次采集波动大（实测 ±3–5%，见 §1.2 的样本），**每个数值取 3 次独立采集
  进程的中位数**，样本一律写入 `perf_records.jsonl` 的 `samples_us`。不使用 NPU event、不用端到端耗时。
- 主指标：`msprof_task_duration`（Task Duration(us)，一次算子调用的全部 launch 之和）。
- 解读器逐 block 指标用 `profiles/experiments/<tag>/…/PipeUtilization|ResourceConflictRatio|Memory_*.csv`
  （48 或 36 个 block 的中位数与最大值），脚本 `perf_opt/experiments/` 下的 helper（`pp` 逻辑内联在日志的数据源里）。

---

## 0. Phase 0 — 上下文加载与首轮必查项

- 工具链戳：tilelang 0.1.15 / CANN 9.2.1（compiler 12.3.0）/ raw `7a52f2f1…`；设备 `Ascend950PR_9579`，
  cube 28 / vector 56（平台 `hardware.observed_cores`；本任务用 `torch.npu.get_device_properties` 复核
  `vector_core_num=56`）。
- 知识索引消费（`knowledge/INDEX.md` 全读）：直接命中 `CASE-batch2-engram-gate-conv`（同算子族）、
  **`PL-950-window-bulk-copy`** / **`TRAP-950-rowload-dma-chain`**（origin_task
  `engram_gate_conv-…-20261124T171256Z`，同族 forward 的 Stage 4 实测）、`TRAP-950-rowstore-mte3-drain`、
  `TRAP-950-simd-ub-store-load`（VST→VLD 需 `mem_bar`，本 kernel 的 `mem_bar("VST_VLD")` 即此）、
  `PL-950-simd-reduce`、`CONST-950-kernel-launch-floor-48blk`（2.7 µs @ grid=48）。
  条目按「数据非指令」消费。
- **E-5 版本失效核对**：条目 `toolchain_stamp` 与当前一致（`7a52f2f1…`，同日同工具链），无需重验；
  本任务反过来为同族条目提供了**装载侧 / 落盘侧双向整窗化**在第二个算子上的独立复现（§6.3）。
- 算子类型：**vector**（纯 AIV，无 AIC block；`aiv_vec_ratio` 基线 0.48–0.64、`aic_time=NA`）。
- **首轮必查项 1（向量化轴与布局重估）**：DESIGN §1.6.3 选「I/O 原生 `[M,S,d]` + lane = 行内 `d`」。
  实测 3 条 tune workload 的 `d ∈ {512, 768}` 与 smoke `d=256` 全部 `d % 64 == 0`，尾 lane 浪费 0%；
  本轮未发现需要重估布局的证据（未改动布局）。
- **首轮必查项 2（陷阱版本戳/origin_task）**：`TRAP-950-rowload-dma-chain` 的结论「成本按描述符数计、
  不按字节计」在本算子**同向复现**：基线每块 51 条 MTE2 指令 → MTE2 忙时 5.50 µs，整窗化后 16 条 → 1.02 µs，
  而 `read_main_memory_datas` 恒为 74 KB/块（字节不变）。见 §2、§3 Round 1。
- **首轮必查项 3（DESIGN §1.6.3/§1.6.4 实验裁决三件套）**：DESIGN §1.6.4 的备选 B1 = 「staged 确定化，
  3 次 launch」（`d_silu` 落 GM），裁决依赖两个未实测常数。本任务**未实现 B1**，理由为**实测替代常数的直接
  证据**而非推理：单次 launch 的固定开销在本算子实测为 ≈1.0–1.5 µs（§2.2：两个 launch 的 kernel 时长
  减各自的 block 时长），且 PART 往返是 **884 KB 读 + 18 KB 写**（=8 B/lane，已由 DESIGN §1.6.0 R3 计入）。
  B1 相对主选多 1 次 launch（+≈1.1 µs）并多出 `d_silu` 的 GM 往返（4 B/lane/token × 2 遍 ≈ 590 KB 写 + 4×590 KB 读），
  同口径下**不可能低于主选**（主选的 halo 重算只有 `3/(RC+3)` 的 stage-1 工作，train-s96 上 ≈0.3 µs 量级）。
  因此按 DESIGN 的判据「备选 < 主选 × 0.90 才翻转」直接判定**主选保持**，未把预算花在可预期的负向实验上；
  该判断的量化依据（launch 常数 + 流量差）列在本节，属于**未实测假设的显式标注**：B1 的实测值待需要时补测。
- **launch 合并（用户点名的方向）前置核查**：`T.ascend_cross_core_set_flag/wait_flag`、`ascend_sync_inter_arrive/wait`
  均编译为 `AscendC::CrossCoreSetFlag<mode, pipe>(flagId)`——这是 **AIC↔AIV 之间**的同步原语（文档语义与
  `mode ∈ {0,1,2,4}` 的 pipe 组合），**不是 block↔block 的 barrier**；工作流规则也明确「跨 block barrier 依赖
  尚未通过该工具链认证」。另一条路（GM 原子加消掉 PART）会放弃算子契约要求的**逐位可复现性**
  （`tests/ops/test_engram.py::test_engram_gate_conv_bwd` 对 `dH/dk/dv` 逐位比对，本 kernel 的 L0-10 对六个输出逐位比对）。
  结论：**两次 launch 不能合并**，该项记为 `blocked`（证据：安装模块 `tilelang.ascend.language` 的
  `ascend_cross_core_*` docstring 实查 + 平台 ruleset 条款），并把「launch 固定开销」作为 T_op 的硬地板计入分析。

### 0.1 基线结构（冻结，不改）

两次 launch：
- launch #1（`grid = M·nch ≤ 56`，train-s96 grid=48 / RC=6 / nchunks=12）：每核一个 token-run，
  行循环 `i = 0 … RC+2·LEAD-1`，(a) 载入并发布 `vhat_norm`/raw `vhat`/`dY` 到 4 槽 UB 环窗，
  (b) 重算 width-4 因果卷积与 `d_silu` 并对自有行累加 `dconv_w` 部分和，(c) 对 3 行之前的 token 跑完整
  value/gate 反向（反卷积、RMSNorm(v) 反向、gate 反向、两半 RMSNorm(h) 反向）。`d_silu`/`dvhat`/`dY` 不落 GM。
- launch #2：把 `PART[6, G, dw]` 按核号升序求和写 3 个 fp32 权重梯度（无原子、无清零 launch）。

---

## 1. Phase 1 — Workload Inventory 与 Baseline

### 1.1 Inventory

来源：`benchmarks/ops/bench_engram.py::test_engram_gate_conv_bwd_bench`，参数由
`tileops.manifest:EngramGateConvBwdOp` 展开 = **4 个案例**，完整清单见 `workload_inventory.json`。

| workload_id | shape (M, seq_len, d) | dtype | kind |
|---|---|---|---|
| train-s32-float16 | (1, 32, 256) | fp16 | **smoke**（benchmark param idx 0 带 `pytest.mark.smoke`） |
| train-s64-float16 | (2, 64, 512) | fp16 | **tune** |
| train-s128-bfloat16 | (1, 128, 256) | bf16 | **tune** |
| train-s96-bfloat16 | (3, 96, 768) | bf16 | **tune**（主导） |

smoke 只做精度回归，不采集性能、不选 winner；无 skip 案例。kernel_id 唯一：
`.../engram_gate_conv_bwd_kernel::_engram_gate_conv_bwd_kernel::main`，两个 launch 的 captured op name 恒为
`main_kernel`（Block Dim 48/36、44/24、43/24）。

### 1.2 Baseline（round 0，各 3 次独立采集，中位数）

| kernel_id | workload_id | candidate | task_duration_us | launch 拆分 | profile |
|---|---|---|---:|---|---|
| bwd | train-s96-bfloat16 | baseline-stage3 | **14.776** | 11.58 + 3.20 | `profiles/experiments/_engram_gate_conv_bwd_kernel_s96_r0` |
| bwd | train-s64-float16 | baseline-stage3 | **10.603** | 8.10 + 2.50 | `profiles/experiments/_engram_gate_conv_bwd_kernel_s64_r0` |
| bwd | train-s128-bfloat16 | baseline-stage3 | **9.663** | 7.35 + 2.31 | `profiles/experiments/_engram_gate_conv_bwd_kernel_s128_r0` |
| bwd | train-s32-float16 (smoke) | baseline-stage3 | 不计入门禁 | – | 仅精度（L0-1/L1-1） |

交叉核对：`examples/batch2/mfu_baseline_v1.json` 记录主导案例 T_op **15.021 µs**（MFU(vector) 2.15%），
本任务同口径 3 次采集中位 14.776 µs（-1.6%，在 ±3–5% 的 run-to-run 波动内），单次采集最大 15.135。
其余两例 10.34 / 8.711 与本轮 10.603 / 9.663 一致（+2.5% / +11%，后者样本 [9.777, 8.891, 9.663] 抖动较大）。

**排序**：train-s96（288 token、d=768、14.78 µs，占 4 案例 FLOPs 的 82%）→ train-s64（10.60）→
train-s128（9.66）；smoke 不参与。排序只影响先后，三个 tune workload 全部有结论。

### 1.3 设计估算 vs 实测（D-2 回填）

DESIGN §1.6.0 的 roofline 估算对 train-s96 给出「流量项 ≈5.3 µs（未实测 1.2 TB/s 假设）+ 发射项 6–10 µs
→ **10 µs 级**」，并标注为未实测假设。实测 14.776 µs，偏差 1.48×（>1.2 但仍同量级），拆解为本轮实测：

| 组成（train-s96） | 实测 | 来源 |
|---|---:|---|
| launch 固定开销（2 次 launch） | **≈2.5 µs**（1.1 + 1.4） | kernel 时长 − 各自 block 时长：11.58−10.05 / 3.20−2.08（`profiles/…_s96_r0`） |
| launch #1 向量管道忙时 | **6.42 µs** | block 中位 `aiv_vec_time`（48 块） |
| launch #1 标量忙时 | **2.14 µs** | `aiv_scalar_time` |
| launch #1 MTE2 忙时（逐行 DMA 链） | **5.50 µs** | `aiv_mte2_time`（51 条 MTE2 指令/块） |
| launch #1 MTE3 忙时 | 1.14 µs | `aiv_mte3_time`（26 条/块） |
| launch #2 block 时长 | 2.08 µs | block 中位（读 72 KB/块，31 GB/s） |

即 DESIGN 的「单次 launch 3–5 µs」估算偏保守（实测 ≈1.1–1.5 µs），但 DESIGN **未量化的
「每行一条 DMA 描述符」代价（MTE2 5.50 µs / 51 条描述符）是最大单项浪费**——与同族 forward 的
`TRAP-950-rowload-dma-chain` 结论一致，本任务把它从装载侧+落盘侧一起消除（§3 Round 1）。

---

## 2. 诊断（Round 1 前，基线 profile）

`train-s96` msprof 逐 block（48 块，`profiles/experiments/_engram_gate_conv_bwd_kernel_s96_r0`）：

| 指标 | block 中位 | block 最大 | 备注 |
|---|---:|---:|---|
| aiv_time(us) | 10.054 | 11.170 | kernel #1 时长 11.58 = 最大 block + 发射 |
| aiv_vec_time(us) | 6.420 | 6.447 | 向量忙时（几乎与 block 等长 → 计算是主成本） |
| aiv_scalar_time(us) | 2.144 | 3.132 | 逐 chunk 迭代的地址/循环开销 |
| aiv_mte2_time(us) | **5.495** | 6.103 | **51 条描述符/块**，active bw 仅 12–15 GB/s |
| aiv_mte3_time(us) | 1.139 | 1.270 | 26 条描述符/块 |
| aiv_vec_wait_ratio | **0.873** | 1.043 | 向量几乎全程在等 MTE2/MTE3 |
| read / write main memory | 74 KB / 45 KB | – | 字节数固定，与描述符数无关 |

**判据**：Σ(pipe 忙时) = 6.42+2.14+5.50+1.14 = 15.2 µs > block 10.05 µs，管道基本不重叠；且
`aiv_vec_wait_ratio` 0.87 —— 与同族 forward 基线的形态一致（`PL-950-window-bulk-copy` 的 origin）。
**结论：先把 51+26 条单行描述符整窗化（O-1/O-2），再看向量指令数（O-9…O-12）。**

启动地板：同族 forward 已在本平台实测「48 块、1 装 + 2 落的 kernel = 2.72 µs」
（`CONST-950-kernel-launch-floor-48blk`）；本算子两次 launch 的固定开销实测 ≈1.1 / 1.4 µs（§1.3），
即 **T_op 的硬地板 ≈2.5 µs**（train-s96 的 17%）。

---

## 3. 迭代记录

每轮从**同一个 current best** 派生分支、每支只改一个主要优化点、先 L0 精度再采集。
所有实验的 stdout/stderr 在 `perf_opt/logs/round{1,2,3}/`，raw profile 在 `profiles/experiments/`。

### Round 1（base = baseline-stage3，2 支）

**候选 vs current best（current best = baseline-stage3 = 14.776 µs）**

| 分支 | 优化点（单一） | workload | Task Duration(us) | Δ vs best | aiv_time | aiv_vec_time | aiv_mte2_time | aiv_mte3_time | vec_wait | MTE2 描述符 | L0 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| r1_c1 | O-1 装载侧整窗：`vhat`/`dY` 由 24 条单行 copy → 2 条 `(RC+2·LEAD, d)` 窗口 | s96 | **13.906** | **-5.9%** | 8.866 | 6.389 | 3.117 | 1.157 | 0.674 | 51→31 | PASS |
| r1_c2 | O-2 (c) 段整窗：`H/k/v` 装载 + `dH/dk/dv` 落盘各 1 条窗口（36 → 6 条） | s96 | 14.107 | -4.5% | 8.795 | 6.318 | **1.016** | **0.349** | **0.107** | 51→16 | PASS |

> Δ vs best 以 round 0 的 14.776 为基准；两分支都优于基线，c2 在 s96 上略差于 c1（-0.8%，噪声内），
> 在 s64/s128 上更好（见下）。**<5% 平局按 T-2 走交错 A/B 协议裁决**：
> `python3 .agents/tools/ab_test.py --a-cmd "…round1_c1…" --b-cmd "…round1_c2…" --pairs 3 --launch-count 2`
> → `b_rel_change = -0.0082`、`sign_test_p = 1.0`、**verdict = tie**（JSON 落在 `profiles/ab/ab_round1{,b}/`）。
> 说明：`ab_test.py` 报的是「逐 launch 行中位数」（两个 launch 各一行 → 6.9 µs），**不是** T_op 口径，
> 只用于同仪器下的平局裁决；正式数值一律来自 `bench_one.py` 的两 launch 求和。
> **平局决胜**：c2 的描述符数严格更少（16 vs 31）、vec_wait 0.107 vs 0.674，且 s64/s128 更好 → **c2 成为 current best**。

**合并非回退检查（c2，全部 tune workload + smoke）**

| workload | baseline | r1_c2 | Δ |
|---|---:|---:|---:|
| s96 | 14.776 | 14.107 | -4.5% |
| s64 | 10.603 | 9.604 | -9.4% |
| s128 | 9.663 | 8.094 | -16.2% |
| s32（smoke，仅精度） | – | PASS | – |

**现象更新（c2 之后）**：MTE2 1.02 µs / MTE3 0.35 µs / vec_wait 0.107 → **DMA 已完全落到计算之后**，
block 时长 8.80 µs 由 `vec 6.32 + scalar 1.90` 决定（两者近似串行）。瓶颈从「DMA 串行化」转为
**向量/标量指令数（issue-bound，实测 IPC ≈1.2）**。

### Round 2（base = r1_c2 = 14.107 µs，3 支）

**候选 vs current best（current best = r1_c2 = 14.107 µs）**

| 分支 | 优化点（单一） | workload | Task Duration(us) | Δ vs best | aiv_vec_time | aiv_scalar_time | 结论 | L0 |
|---|---|---|---:|---:|---:|---:|---|---|
| r2_c3 | O-9 把逐 chunk 的 `rrms^3/d`（`coef`/`rh3d`/`rk3d`）提到 token 级寄存器 | s96 | 13.838 | -1.9% | **6.318（不变）** | 1.936 | 编译器已做 CSE → 指令数不变，收益在噪声内；**保留**（源码表达意图） | PASS |
| r2_c4 | O-10 把 `dconv_w` 部分和融进卷积 sweep（4 tap 驻留寄存器） | s96 | 14.181 | **+0.5%** | **6.683（+5.8%）** | 1.856 | 4 个 tap 局部寄存器引起寄存器压力/额外搬运 → **否决** | PASS |
| r2_c5 | O-11 launch #2 列分块取 `6·tiles ≤ 54` 的最大块数（12 → 36 块，128 列/块） | s96 | **13.663** | **-3.1%** | 6.318 | 1.990 | launch #2 2.08 → 1.38 µs block，读取 72 → 24 KB/块、单核拷贝带宽 31 → 55 GB/s；**采纳** | PASS |

- c4 的反向证据很有价值：`aiv_vec_time` 从 6.318 升到 6.683 µs，`aiv_time` 9.29（vs 8.76），
  即本 kernel 已处在**向量寄存器压力边界**上，任何「多留几个 tap 寄存器」的重构都会净亏。
  O-12（Round 4）因此只保留 4 个累加器、不再复制 tap。
- c5 的求和顺序不变（每个输出列仍是「核号升序逐行相加」），只是列 tile 更细 → 结果逐位不变。

### Round 3（base = r2_c5 = 13.663 µs，1 支：合并）

**候选 vs current best（current best = r2_c5 = 13.663 µs）**

| 分支 | 优化点（单一） | workload | Task Duration(us) | Δ vs best | aiv_time | aiv_vec | aiv_scalar | MTE2 | L0 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| r3_c6 | **合并** c5 + c3（c3 已证指令中性），在全 workload 集上复测 | s96 | **13.358** | **-2.2%** | 9.033 | 6.318 | 1.986 | 1.069 | PASS |
| r3_c6 | 合并非回退检查 | s64 | 9.145 | -2.2%（vs c2 9.604） | – | – | – | – | PASS |
| r3_c6 | 合并非回退检查 | s128 | 7.660 | -5.4%（vs c2 8.094） | – | – | – | – | PASS |

合并后可安全替代 c5：三个 tune workload 全部不退化，smoke（s32）精度 PASS（L0-1/L1-1，见 §5.2）。

### Round 4（base = r3_c6 = 13.358 µs，1 支）

**候选 vs current best（current best = r3_c6 = 13.358 µs）**

| 分支 | 优化点（单一） | workload | Task Duration(us) | Δ vs best | aiv_time | aiv_vec_time | aiv_scalar_time | MTE2/MTE3 | L0 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| r4_c7 | O-12 `dconv_w` 部分和改 chunk-major，4 个累加器驻留寄存器（`nrm`/`d_silu` 行改为 block 常驻，换掉 raw `vhat`/`dY` 的 fp32 环窗） | s96 | **13.045** | **-2.3%** | 8.395 | **6.172** | **1.507** | 1.325 / 0.583 | PASS |

- 机制：每个 owned (row, chunk) 省掉 8 条 `pc` 读写（改为每 chunk 一次），`cwf` 装载由每行一次变为每块一次；
  `aiv_scalar_time` 2.0 → 1.5 µs（chunk 迭代数 612 → 540），`aiv_vec_time` 6.32 → 6.17。
  MTE2 略升（1.07 → 1.33 µs，因为 raw `vhat`/`dY` 改为在用处转换），但仍在计算之后，收益净正。
- 逐位一致性：accumulation 顺序（owned 行升序、每行 tap 0..3）与基线相同；实测六个输出的
  max|diff| 与基线**逐字段相同**（§5.2）。
- 调试留痕：初版把反卷积 tap 写成 `d_silu[tp+p]` 而基线是 `d_silu[tp+3-p]`（相关而非卷积），
  被 s96 的 L0 抓到（dv 3.35e-1、drms_w_v 2.4e-1；其它输出仍在容差内，说明**只看 assert_close 会漏**），
  修正为 `dsw_all[i - LEAD - p]` 后与基线逐位一致。

### 中止条件

Round 4 结束后：3 个 tune workload 均有结论（`tuning_status=winner_merged`），最终文件全量复测完成。
**目标（主导案例 -20%）未达成**（实得 -13.5%），未达成原因与剩余空间量化见 §5.3；
按 skill 的终止条件归类为 **plateau**（连续 3 轮无 >3% 的结构性提升：Round 2/3/4 的净提升 3.1%/2.2%/2.3%，
且剩余方向都已实测或按规则封死）。预算未耗尽（max_rounds=10、max_experiments=30，实用 11 个实验分支）。

---

## 4. 关键方向的实测裁决

| 方向 | 结论 | 证据 |
|---|---|---|
| 单行 DMA 链 → 整窗 2-D copy（装载 + 落盘） | **采纳，最大收益** | MTE2 5.50→1.02 µs、MTE3 1.14→0.35 µs、vec_wait 0.873→0.107；44 条描述符减少 |
| launch #2 列分块加大块数 | **采纳** | block 2.08→1.38 µs、单核带宽 31→55 GB/s、总读 884 KB 不变 |
| `dconv_w` 积分块改 chunk-major（累加器驻留） | **采纳** | vec 6.32→6.17、scalar 2.0→1.5 µs |
| 「tap 驻留寄存器」的融合（r2_c4） | **否决** | vec_time 6.318→6.683（寄存器压力），总时 +0.5% |
| 逐 chunk 不变式提取（r2_c3） | **neutral（已保留）** | vec_time 不变，说明编译器已 CSE/hoist |
| **两次 launch 合并为一次** | **blocked（不可行）** | 需要 block↔block barrier 或 GM 原子：`ascend_cross_core_set/wait_flag` 的实查语义是 AIC↔AIV（`CrossCoreSetFlag<mode,pipe>`）；ruleset 明确「跨 block barrier 依赖未认证」；原子会放弃逐位确定性（`test_engram_gate_conv_bwd` 对 dH/dk/dv 逐位比对、L0-10 对六输出逐位比对） |
| DESIGN §1.6.4 备选 B1（3 launch staged） | **不实现，主选保持** | 多 1 次 launch（+≈1.1 µs）+ `d_silu` GM 往返（4 B/lane/token 写 + 4×读）> 主选的 halo 重算（≈0.3 µs）；未实测，标注为未实证假设 |
| 提高核占用（48/56 → 56/56） | **不可行** | 288 token / 每核 RC=6 → 48 个 chunk 是 `M·nch ≤ 56` 与「RC 取整」规则下的唯一解；RC=5 得 58 chunk（2 波）更差，非均匀切分不改变最大 block 的 token 数 |

---

## 5. Final Summary

### 5.1 最终候选

`perf_opt/engram_gate_conv_bwd.py` = r4_c7（O-1/O-2 整窗 DMA + O-11 归约分块 + O-12 dconv chunk-major；
O-9 作为表达保留），sha256 `65c3133c38ec86ba26db72e90d620c2c4cac8c079b04e1a211fde0b11668ec81`，
与 `perf_opt/EngramGateConvBwdOp.py` 字节相同。与 Stage-3 基线相比**语义零改动**：
门控统计量、`vhat` 的 dtype 舍入往返、卷积 4 tap 的累加顺序、反卷积 tap 的顺序、三个权重梯度的
「核内 token 升序 + 跨核核号升序」求和顺序全部保持原表达式。工厂签名
`_engram_gate_conv_bwd_kernel(M, seq_len, d, dtype)` 与默认参数未变（wrapper 无需成对引用新常量）。

### Final Performance Test Data

| kernel_id | workload_id | baseline_us | final_us | candidate_id |
|---|---|---:|---:|---|
| tileops/kernels/sequence_modeling/engram_gate_conv_bwd/engram_gate_conv_bwd_kernel::_engram_gate_conv_bwd_kernel::main | train-s96-bfloat16 | 14.776 | 12.784 | final-chunkmajor |
| tileops/kernels/sequence_modeling/engram_gate_conv_bwd/engram_gate_conv_bwd_kernel::_engram_gate_conv_bwd_kernel::main | train-s64-float16 | 10.603 | 9.111 | final-chunkmajor |
| tileops/kernels/sequence_modeling/engram_gate_conv_bwd/engram_gate_conv_bwd_kernel::_engram_gate_conv_bwd_kernel::main | train-s128-bfloat16 | 9.663 | 7.393 | final-chunkmajor |

提升：train-s96 **-13.5%**、train-s64 **-14.1%**、train-s128 **-23.5%**。三者皆无回退。
（全部数值为 3 次独立采集进程的中位数，样本见 `perf_records.jsonl` 的 `samples_us`；
最终三行的 `artifact_path` = `perf_opt/engram_gate_conv_bwd.py`，`artifact_sha256` 与文件一致。）

最终 profile 证据（train-s96，`profiles/experiments/engram_gate_conv_bwd_s96_r0`）：

| 指标 | baseline | final | 说明 |
|---|---:|---:|---|
| T_op（2 launch 求和） | 14.776 | **12.784** | -13.5% |
| launch #1 时长 / block 中位 | 11.58 / 10.05 | 10.06 / 8.63 | |
| launch #2 时长 / block 中位 | 3.20 / 2.08 | 2.72 / 1.34 | |
| aiv_vec_time | 6.420 | 6.175 | 计算未变，指令数 -4% |
| aiv_scalar_time | 2.144 | 1.815 | chunk 迭代数 -12% |
| aiv_mte2_time / 描述符数 | 5.495 / 51 | 1.368 / 16 | **-75%** |
| aiv_mte3_time / 描述符数 | 1.139 / 26 | 0.581 / 11 | -49% |
| aiv_vec_wait_ratio | 0.873 | 0.112 | |

### 5.2 smoke / skip 与精度

- smoke：`train-s32-float16`（benchmark idx 0 的 `pytest.mark.smoke`）——只做精度回归，
  最终 `precision_pass: true`（L0-1 通过）。
- skip：无。
- 最终精度（`perf_opt/logs/final/L0_L1_L2_boundary.log`，`perf_opt/engram_gate_conv_bwd.py --level all`）：
  **L0 10/10、L1 6/6、L2 4/4、boundary 5/5 → `[PRECISION_PASS]`**，
  max_diff：dH 1.221e-04、dk 9.766e-04、dv 1.953e-03、drms_w_h 3.598e-04、drms_w_v 4.453e-04、dconv_w 1.194e-02。
- **与基线逐字段相同**：同一命令跑基线模块（`perf_opt/logs/final/baseline_level_all.log`）得到
  **逐项完全相同的 max_diff 与相同的 10/6/4/5 通过数** → tuned kernel 的输出与 Stage-3 基线逐位一致。
- TileOPs 回归：`pytest tests/ops/test_engram.py -q -k gate_conv_bwd` → **17 passed, 29 deselected**
  （该路径当前仍走 wrapper 里的 Stage-3 baseline import；wrapper 的切换块按纪律未改，由 conductor 翻转）。

### 5.3 结论与未达 -20% 的量化归因

主导案例 14.776 → 12.784 µs（**-13.5%**，目标 -20% 未达成），三个 tune workload 全面下降 13–24%、无回退。
剩余 T_op 的构成（train-s96，实测）：

| 项 | µs | 占 final T_op | 可压缩性 |
|---|---:|---:|---|
| 两次 launch 的固定开销 | ≈2.5 | 19.6% | **不可压缩**（工具链未认证跨 block barrier；原子破坏确定性） |
| launch #2 的 block 工作（PART 884 KB 归约） | ≈1.3 | 10.2% | 已到带宽/描述符地板（36 块 × 24 KB，55 GB/s/核） |
| launch #1 向量管道（issue-bound，IPC ≈1.2） | 6.18 | 48.3% | 需减少指令数；**寄存器压力已成禁区**（r2_c4 实测 +5.8% vec） |
| launch #1 标量（chunk 迭代 540 × ≈5 cyc） | 1.82 | 14.2% | 需减少 chunk sweep 数 |
| 其余（DMA 1.37 + 交叠损失） | ≈1.0 | 7.7% | 已整窗化，MTE2 与计算重叠 |

即 **T_op 的地板（2.5 µs launch + 1.3 µs 归约）已占 30%**，而计算部分的可用空间（把 8.0 µs 的
vec+scalar 砍掉 30% 需要再删 ~3000 条向量/标量指令，等价于 DESIGN R3 的 84 ops/lane/token 理想值）
受限于「4 个行归约 ⇒ 每 token 4 个 chunk sweep」的算法依赖与「lane 宽 64」的硬件宽度，
本轮 11 个实验分支里能落地的部分已全部采纳（-13.5%）。

**不触发 `[DESIGN_LIMIT]`**：① 性能不足归因于**平台 launch 地板 + 未认证同步原语**（属工具链边界，
不是 DESIGN 的算法/结构假设被推翻），② 结构性加速估计远小于 2×（要把 12.78 → 6.4 才够），
两项均不满足 `_shared/standards/perf-feedback.md` §1 的触发条件，故**未产出 `perf_feedback.md`**。
DESIGN 的实测对账反而**支持**其主选结构：2 次 launch 的确定化归约是当前工具链下唯一可行路径。

---

## 6. Skill Retrospective

### 6.1 流程问题与建议（proposal，不自动改 skill）

| # | 现象 | 建议 | vp_type |
|---|---|---|---|
| 1 | **多 launch 算子（T_op = launch 之和）的 `<5%` 平局裁决工具口径不匹配**：`ab_test.py` 用「逐 launch 行中位数」，2 launch 算子得到的是 (l1+l2)/2 ≈ 6.9 µs，与平台 T_op 口径（求和）不同；虽同仪器可比，但 JSON 里的 `*_medians_us` 会被误读成 T_op | `ab_test.py` 增加 `--sum-launches`（或按 `launch_count` 求和后再取中位），或在输出里显式标注「本值不是 T_op」 | R |
| 2 | 平台 `execution.precision_command` 指向 `/usr/local/python3.12.16/bin/python3`，该解释器无 torch；实际可跑的是 `$TILEOPS_950PR_VENV/bin/python` | 平台 execution context 应解析到带 torch/torch_npu 的解释器 | R |
| 3 | 「指令数减少 ⇒ 时间下降」在本 kernel 上**两次失效**（r2_c3 编译器已 CSE、r2_c4 寄存器压力反噬），只有读 `aiv_vec_time` 才能判断 | iteration-diagnosis 增一条判据：向量算子做指令级重构前后必须比对 `aiv_vec_time`；`vec_time` 不降即视为编译器已优化/寄存器溢出，不要再调参数 | R |
| 4 | 多 launch 算子的 `--launch-count` 语义（本次 `=2` 才能采到两个 launch，`=1` 只采第一个）在 profile-collection 里没有写明 | profile-collection 补一条：**采前先用 `OpBasicInfo` 的 Block Dim / 行数确认 launch 数**，再设 `--launch-count` | R |
| 5 | 反卷积 tap 顺序（`d_silu[tp+3-p]`）这类「相关而非卷积」的细节只在 golden 上暴露，且会被宽松 rtol 掩盖（`dv` 差 0.335 仍能通过 bf16 的 `0.3/0.2`） | develop/optimize 的精度校验应记录并**比对 max_diff 的具体数值**（与基线逐字段一致才算等价），不只看 `assert_close` 是否通过 | R |

### 6.2 价值点（value point proposals）

- **VP-D1（D，实测代价常数）**：950PR 上 bwd 类「两 launch」算子的单次 launch 固定开销实测
  ≈**1.1–1.5 µs**（相对 `CONST-950-kernel-launch-floor-48blk` 的 2.7 µs 地板，后者含 block 工作），
  两 launch 结构因此固定多付 ≈2.5 µs。证据三件套：evidence = §1.3 表（kernel 时长 − block 时长 / 两个 launch）；
  repro = `perf_opt/experiments/bench_one.py` + `profiles/experiments/*/main_kernel/{0,1}`；stamp = `7a52f2f1…`。
- **VP-P3（P，模式）**：**落盘侧整窗化在「逐 token 一个输出行」的结构上同样有效**：本算子把 (c) 段
  每 token 的 3 条单行 `T.copy` 改成 `(RC, d)` 暂存 + 块末 3 条窗口写回，MTE3 忙时 1.14 → 0.35 µs、
  26 → 11 条描述符（与 `TRAP-950-rowstore-mte3-drain` 的散写结论同向，但这里是**批量写回**形态）。
  证据：§3 Round 1 的 r1_c2 行；repro 可裁 `engram_gate_conv_bwd_round1_c2.py` 的 diff。
- **VP-P4（P，反向模式）**：**寄存器压力是这套 SIMD kernel 的硬边界**：把 4 个 tap 驻留寄存器
  （r2_c4）使 `aiv_vec_time` 6.318 → 6.683 µs、总时 +0.5%；而只驻留 4 个累加器并**同时删掉 8 条
  内存操作**（r2_c7/O-12）才净赚 -2.3%。判据：增益来自「删内存操作」，不是「加寄存器复用」。
- **VP-P5（P，反向模式）**：**多 launch 归约 kernel 的列分块应按「块数最大化」而非「块宽最大化」定**：
  12 块 × 384 列 → 36 块 × 128 列，block 时长 2.08 → 1.38 µs（单核拷贝带宽 31 → 55 GB/s），
  求和顺序（核号升序）与结果逐位不变。约束仍是 `6·tiles ≤ 54`（单波）。
- **VP-C2（C，案例索引）**：本任务可作为「多 launch、T_op ~15 µs、launch 占 17%」的向量算子范式：
  ①先按 launch 拆分时间 → ②确认 launch 数不可合并（先查同步原语的真实语义）→ ③整窗化所有 DMA →
  ④再按 `aiv_vec_time` 决定是否做指令级重构（并预设寄存器压力红线）。

### 6.3 回写 pattern-library 的候选（待终态 evolver 处理）

本任务实测**复现并推广**了同族 forward 的两条条目：`PL-950-window-bulk-copy`（装载侧）与本算子的
落盘侧对偶；以及 `TRAP-950-rowload-dma-chain` 的「按描述符计费」在 51 → 16 条上的第二次独立复现
（不同算子、不同 launch 几何）。三者均已在 §3/§6.2 给出任务溯源与数值，条目本身未改写
（skill 规定回写由终态 `tilelang-skill-evolver` 蒸馏；本 agent 的例外授权仅覆盖「新模式/新常数」，
本轮无超出已有条目的新形态，故不新增条目，只在此登记候选与证据位置）。

### 6.4 其它交付核对

- 最终文件以**模块方式**被导入验证（`experiments/bench_one.py` 用 importlib 加载并调用
  `_engram_gate_conv_bwd_kernel(M, seq_len, d, dtype)`，与 wrapper 的 perf_opt 切换块同签名）。
- wrapper 的 baseline/perf_opt 切换块在 Stage 4 收束时按纪律**未改**（由生产采纳步骤按其自身纪律翻转，见 §7）。
- 未改动基线 kernel / op / manifest / test / bench；未触碰 `/workspace/TileOPs-ascend950`。

---

## 7. 生产采纳（wrapper 切换 + 单算子报告验收）

### 7.1 采纳文件清单

| 文件 | 作用 | sha256 |
|---|---|---|
| `.../engram_gate_conv_bwd_kernel/history_version/engram_gate_conv_bwd_pre_opt.py` | **wrapper 备份**（翻转前原样，可整体回滚） | `94f9df55944d1bd2c710c40a7e564fc4b1e2004145579e7f55be7228b204b7a0` |
| `.../engram_gate_conv_bwd_kernel/perf_opt/__init__.py` | perf_opt 子包导出（与父包同名符号） | — |
| `.../engram_gate_conv_bwd_kernel/perf_opt/_engram_gate_conv_bwd_kernel.py` | **drop-in 优化 kernel**（= `perf_opt/engram_gate_conv_bwd.py` 同字节） | `65c3133c38ec86ba26db72e90d620c2c4cac8c079b04e1a211fde0b11668ec81` |
| `.../engram_gate_conv_bwd/engram_gate_conv_bwd.py` | wrapper：**仅**翻转 import 切换块注释（op 层构造/forward 契约不变，无成对默认参数需翻转——两版工厂签名与 `default_config`（空 dict）相同） | 见 git diff |

- Stage-3 kernel 本体未动：`_engram_gate_conv_bwd_kernel.py` 仍是
  `0d5bb90bba68eda7ea10db09c0e06b27eeb3717bcd61e990ea1d34af6db13d22`，其 `__init__.py` 未改。
- 回滚方式：把 wrapper 切换块两行注释换回（baseline 分支从父包 `engram_gate_conv_bwd_kernel` 导入，
  Stage-3 源码未动），或用 `history_version/engram_gate_conv_bwd_pre_opt.py` 覆盖 wrapper。
- `git status` 在该算子目录下只有：`M …/engram_gate_conv_bwd.py`（切换块）、`?? …/perf_opt/`、
  `?? …/history_version/engram_gate_conv_bwd_pre_opt.py`。

### 7.2 经 wrapper 后确实走优化实现

```
factory __module__ : tileops.kernels.sequence_modeling.engram_gate_conv_bwd.engram_gate_conv_bwd_kernel.perf_opt._engram_gate_conv_bwd_kernel
factory file       : .../engram_gate_conv_bwd_kernel/perf_opt/_engram_gate_conv_bwd_kernel.py
perf_opt ACTIVE    : True
deployed sha256    : 65c3133c38ec86ba26db72e90d620c2c4cac8c079b04e1a211fde0b11668ec81
baseline still importable: tileops…engram_gate_conv_bwd_kernel._engram_gate_conv_bwd_kernel
```

（`_engram_gate_conv_bwd_kernel` 是 `functools.lru_cache` 包装对象，源码路径取 `__wrapped__`。）

### 7.3 生产回归

- `pytest tests/ops/test_engram.py -q -k gate_conv_bwd` → **17 passed, 29 deselected**（47.67 s；
  `logs/production/pytest_gate_conv_bwd.log`）。翻转前同一命令也是 17 passed（4.05 s，走 baseline），
  翻转后走 perf_opt → 同一批 17 例全过。
- `perf_opt/_engram_gate_conv_bwd_kernel.py --level all`（**部署后的那份文件**）
  → L0 10/10、L1 6/6、L2 4/4、boundary 5/5、`[PRECISION_PASS]`
  （`logs/production/production_level_all.log`）；六个输出的 max|diff| 与 Stage-3 基线
  （`logs/final/baseline_level_all.log`）**逐字段相同**：dH 1.221e-04、dk 9.766e-04、dv 1.953e-03、
  drms_w_h 3.598e-04、drms_w_v 4.453e-04、dconv_w 1.194e-02 → **精度/逐位不变**。

### 7.4 单算子报告（msprof）

```
TILEOPS_TASK_DIR=/workspace/tilelang-mlir-ascend-950/examples/engram_gate_conv_bwd \
TMPDIR=/workspace python -m tileops.reporting.cli run --op EngramGateConvBwdOp \
  --prof-mode msprof --pytest-arg=-k --pytest-arg=gate_conv_bwd
```

- **run-id：`20261124_231932_732777_EngramGateConvBwdOp`**
  （`examples/TileOPs/reports/tileops/20261124_231932_732777_EngramGateConvBwdOp/`，日志
  `logs/production/report_run.log`）
- `status=passed`；correctness 17/17；`invalid_profile_count=0`、`invalid_acceptance_count=0`、
  `profiler_fallback_count=0`、`kernel_filter_fallback_count=0`；`warnings=[]`；`msprof_case_count=3`。

| case | dtype | **T_op（本次报告，一次采集两 launch 求和）** | 报告的 `latency_us`（单 launch） | Stage-4 final（3 次中位） | 优化前报告 `20261124_092758_551541`（同链路） | `mfu_baseline_v1.json` |
|---|---|---:|---:|---:|---:|---:|
| **train-s96** | bf16 | **12.393 µs** | 9.803 | 12.784 | 15.176 / 11.727 | 15.021 |
| train-s64 | fp16 | **8.809 µs** | 6.093 | 9.111 | 10.340 / 7.040 | 10.34 |
| train-s128 | bf16 | **7.139 µs** | 4.873 | 7.393 | 8.711 / 6.132 | 8.711 |

- **同链路（同一 report 链 + 同一 T_op 口径）对比**：train-s96 **15.176 → 12.393 µs（-18.3%）**、
  train-s64 10.340 → 8.809（-14.8%）、train-s128 8.711 → 7.139（-18.1%）；
  单 launch 口径同向（11.727 → 9.803 / 7.040 → 6.093 / 6.132 → 4.873）。
  *注意*：报告的 `latency_us` 是**单个 launch** 的统计量，`t_op_us` 才是平台 T_op（两 launch 之和），
  两者不可混用；本算子任何「µs 数」比较必须同一口径。
- 每次调用仍是 **2 个 launch**（`t_op_breakdown = {main_kernel: 12.393}`，即同名两 launch 之和；
  `captured_op_count=1`、`kernel_name_resolved=main`，与本算子 `policy.t_op` 口径一致）；
  train-s32 是 smoke，报告链按标记只测 3 个 tune case。
- MFU（vector）：train-s96 **2.128% → 2.606%（+22.5%）**、s64 0.923% → 1.084%、s128 0.550% → 0.671%。
  验收 `passed=0/failed=3` 与优化前一致（`mfu_baseline_v1.json` 记录本算子 0 个 case 达标），
  `blocked_notes` 明示「memory-bound，MFU 被 HBM 带宽压在 14.29–14.47%」，**不是回退**。
- MU 诊断（非门禁）：`mu_value_measured` 8.52（s64）。

### 7.5 回写 `perf_records.jsonl`（append-only）

追加 3 行（round 5、`phase=merged`、`candidate_id=production-wrapped`、`parent_id=final-chunkmajor`），
`artifact_path` 指向**部署后**的 perf_opt 文件、`artifact_sha256` = `65c3133c…`（与文件一致），
`duration_us` = 报告的 `msprof.t_op_us`，`samples_us` = 该次采集两个 launch 的分量（不是两个 T_op 样本）。
既有 19 行逐字节未改（`diff` 校验通过），见 `experiments/append_production_records.py`。

### 7.6 兼容性与回退

- 无回退：三条 tune case 经生产链路全部改善（12.393 / 8.809 / 7.139 µs），精度门限不变、逐位不变。
- 契约无变化：`EngramGateConvBwdKernel.__init__(M, seq_len, d, eps, dtype, config, device_index)`、
  `forward` 的 12 入 6 出、`default_config`（空 dict）与 `ascend_mode=None` 均未改动；
  wrapper 的 `custom_op("npub::engram_gate_conv_bwd")` 与 fake 输出形状不变。
- perf_opt kernel 与基线工厂**同签名**（`(M, seq_len, d, dtype_str)` → 编译后 kernel，
  `out_idx=[12,13,14,15]` 与 `[1,2,3]` 相同），`lru_cache` 形状特化策略相同，`geometry()` 分核规则未改
  → 无新默认参数需要在 wrapper 成对引用（切换块内没有默认参数需要一起翻转）。
- 一个**已知的形状边界差异**（不是语义变化，源于整窗暂存与 block 常驻窗口的 UB 占用）：perf_opt 的
  `_check`/`_ub_footprint` 现在计入整窗暂存（`2·(rc+2·LEAD) + 6·rc` 行 dtype）与常驻的 `nrm`/`d_silu` 行，
  因此**大 `seq_len` 组合会比 Stage-3 更早触发 UB 预算拒绝**。实测（M=1、bf16）接受的最大 `seq_len`：

  | d | perf_opt | Stage-3 baseline |
  |---|---:|---:|
  | 256 | 1560 | 4096 |
  | 512 | 672 | 4096 |
  | 768 | 336 | 4096 |
  | 1024 | 168 | 4096 |

  manifest / benchmark / 测试覆盖的全部形状（含边界用例 `d=1024, seq=64`、`d=100`、`seq=401`、
  `(2,512,256)`、`d=768, seq=96`）均在预算内，生产回归与报告链全过；超出上述包络的组合本任务
  无人使用，如需该极端组合回滚 wrapper 即回到 Stage-3（或把整窗暂存改成按 band 分批装载，
  属后续工作）。该边界已在 `workload_inventory.json` 的 kernel 描述与本表登记。
- `/tmp/tileops_msprof_*` 已清理。

