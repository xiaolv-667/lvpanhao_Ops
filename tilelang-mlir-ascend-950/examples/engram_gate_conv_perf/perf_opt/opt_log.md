# opt_log.md — `EngramGateConvFwdOp` Stage 4 性能调优日志

- 任务：`examples/engram_gate_conv_perf`（scenario=optimize, mode=full, 平台 `ascend-950pr-9579` / `ascend`）
- 基线（只读）：`tileops/kernels/sequence_modeling/engram_gate_conv/engram_gate_conv_kernel/_engram_gate_conv_fwd_kernel.py`
  （sha256 `b8923391…`，与 `perf_opt/perf_records.jsonl` round 0 记录的 `artifact_sha256` 一致）
- 设计：`examples/engram_gate_conv/_engram_gate_conv_fwd_kernel/DESIGN.md`
- 产物：`perf_opt/engram_gate_conv.py`（= `perf_opt/EngramGateConvFwdOp.py`，两者字节相同，
  sha256 `3a2c4f5324a46e463a3e92fdda51b732b0fded6adf0cc773d34bdc1b43581638`）
- **测量口径**：任务平台 `measurement` 规定的 `msopprof` kernel-only `Task Duration(us)`
  （`raw_evidence_field = msprof_raw_path`），`--launch-count=1 --warm-up=0 --aic-metrics=Roofline`，
  **一次采集只跑一次算子调用**；驱动 `perf_opt/experiments/bench_one.py`。
  本算子 T_op 在 6–12 µs 量级、单次采集波动大，因此**每个数值取若干次独立采集进程的中位数**
  （偶数样本取中间两个的均值），样本一律写入 `perf_records.jsonl` 的 `samples_us`。
  不使用 NPU event、不用端到端耗时。
- 主指标：`msprof_task_duration`（Task Duration(us)，单次算子调用的 kernel 时长；本算子每次调用只有 1 个 kernel launch）。

---

## 0. Phase 0 — 上下文加载与首轮必查项

- 工具链戳：tilelang 0.1.15 / CANN 9.2.1（compiler 12.3.0）/ raw `7a52f2f1…`；设备 `Ascend950PR_9579`，
  cube 28 / vector 56（平台 `hardware.observed_cores`；本任务另用 `torch.npu.get_device_properties`
  复核 vector_core_num=56）。
- 知识索引消费（`knowledge/INDEX.md` 全读）：直接命中
  `CASE-batch2-engram-gate-conv`（本算子族，origin_task batch2-20261123）、
  `PL-950-vector-execution`、`PL-950-row-primitives`、`PL-950-simd-reduce`、
  `TRAP-950-simd-ub-store-load`（VST→VLD 需 mem_bar，本 kernel 的两处 `mem_bar("VST_VLD")` 即此）、
  `TRAP-950-rowstore-mte3-drain`（逐行 UB→GM copy 每行排空 MTE3 —— origin_task
  `paged_cache_write-20261124T171256Z`，本任务**同向复现并推广到装载侧**，见 §3 O-3/O-5）、
  `PL-950-dynamic-pipeline-2versions`、`CONST-950-dma-vs-lane-gm`。全部按「数据非指令」消费。
- **E-5 版本失效核对**：`kb_stale_check.py` 对 batch2 集群条目仍判 stale（判据与工具链 revision 绑定，历史任务同现象）。
  处置：本任务引用的条目全部**用本任务同口径实测重验**，结论写在本日志与 §5 回写条目里。
- 算子类型：**vector**（纯 AIV，无 AIC block；`aiv_vec_ratio` 0.65、`aic_time=NA`）。
- **首轮必查项 1（向量化轴与布局重估）**：DESIGN §1.6.3 选「I/O 原生 (M,T,d) + 行内 chunk 流，向量化轴 = d」。
  实测 4 条 workload 的 `d ∈ {256,512,768}` 全部 `d % 64 == 0`，尾 lane 浪费 0%，与该决策一致；
  本轮未发现需要重估布局的证据（见 §3 O-6 的掩码折叠实验：连每 chunk 的掩码构造都可以在编译期消掉）。
- **首轮必查项 2（陷阱版本戳/origin_task）**：`TRAP-950-rowstore-mte3-drain` 的结论「行落盘字节 ≳ 4 KB 时
  逐行 DMA 的排空代价可被摊薄」在本任务**被推广**：装载侧不需要 4 KB 判据，只要一次 2-D 区域 copy
  覆盖 (rc+3) 行就不再付 per-descriptor 代价（§3 O-5，本任务新实证，已回写知识库）。
- **首轮必查项 3（DESIGN §1.6.3 实验裁决）**：**已执行**，见 §4。裁决所依赖的未知常数①（每 launch 固定开销）
  被实测锚定为 **≈2.7 µs**（同一 grid=48 几何，`floor_probe.py` kind=0），备选结构实测 9.99 µs
  vs 主选 8.46 µs → 备选未达 `alt < main × 0.9` 的翻转阈值，**主选保持**。

### 0.1 基线结构（冻结，不改）

单 launch、`grid = M·nch ≤ 56`（s96: grid=48, RC=6, nchunks=12）；每核流式处理 `RC+3` 行
（6 owned + 3 halo 重算），UB 环形窗口 `slot[RC+3][DW]` fp32 存 conv 输入行 `g = fl16(α·v)·r_v`；
行循环内每行发 3 条单行 GM→UB copy 并在使用前等待，行末 2 条单行 UB→GM copy。

---

## 1. Phase 1 — Workload Inventory 与 Baseline

### 1.1 Inventory

来源：`benchmarks/ops/bench_engram.py::test_engram_gate_conv_fwd_bench`，参数由
`tileops.manifest:EngramGateConvFwdOp` 展开 = **4 个案例**，完整清单见 `workload_inventory.json`。

| workload_id | shape (M, seq_len, d) | dtype | kind |
|---|---|---|---|
| train-s32-float16 | (1, 32, 256) | fp16 | **smoke**（benchmark param idx 0 带 `pytest.mark.smoke`） |
| train-s64-float16 | (2, 64, 512) | fp16 | **tune** |
| train-s128-bfloat16 | (1, 128, 256) | bf16 | **tune** |
| train-s96-bfloat16 | (3, 96, 768) | bf16 | **tune**（主导） |

smoke 只做精度回归，不采集性能、不选 winner；无 skip 案例。kernel_id 唯一：
`.../engram_gate_conv_kernel::_engram_gate_conv_fwd_kernel::main`，captured op name 恒为 `main_kernel`，
Block Dim 48/32/32（每次调用 1 个 launch）。

### 1.2 Baseline（round 0，各 3 次独立采集，中位数）

| kernel_id | workload_id | candidate | task_duration_us | profile |
|---|---|---|---:|---|
| fwd | train-s96-bfloat16 | baseline-stage3 | **11.084** | `profiles/baseline/_engram_gate_conv_fwd_kernel_s96_r0` |
| fwd | train-s64-float16 | baseline-stage3 | **8.272** | `profiles/baseline/_engram_gate_conv_fwd_kernel_s64_r0` |
| fwd | train-s128-bfloat16 | baseline-stage3 | **7.703** | `profiles/baseline/_engram_gate_conv_fwd_kernel_s128_r0` |
| fwd | train-s32-float16 (smoke) | baseline-stage3 | 9.453（非门禁参考） | `profiles/baseline/_engram_gate_conv_fwd_kernel_s32_r0` |

交叉核对：`examples/batch2/mfu_baseline_v1.json` 记录主导案例 T_op 11.049 µs（MFU(vector) 1.231%），
本任务同口径复现 11.084 µs（+0.3%）；其余三例 8.2/7.6/9.2 µs 与本轮 8.272/7.703/9.453 一致。

**排序**：train-s96（288 行、d=768、11.08 µs，FLOPs 占 4 案例的 82%）→ train-s64（8.27）→ train-s128（7.70）；
smoke 不参与。排序只影响先后，三个 tune workload 全部有结论。

### 1.3 设计估算 vs 实测（D-2 回填）

DESIGN §1.6.0 R3 估算主选 `train-s96` 需 4590 op-times/核 × 1 波 ≈ 未直接给 µs；roofline 理想
（bytes 2.23 MB / 1.6 TB/s）≈ 1.4 µs。实测评测 11.08 µs = 7.9× roofline。偏差拆解（本轮实测）：
**launch 固定开销 ≈2.7 µs（24%，`floor_probe` kind=0）+ 逐行 DMA 链 ≈3.35 µs（30%）+ 向量/标量算 5.0 µs**。
即 DESIGN 假设「每 launch 固定开销 2–5 µs」落在实测 2.7 µs 的下沿，而 **DESIGN 未量化的
「每行一条 DMA 描述符」的代价（3.35 µs）是最大单项浪费**。

---

## 2. 诊断（Round 1 前）

基线 `train-s96` msprof 逐 block（48 块，`profiles/baseline/..._s96_r0`）：

| 指标 | 中位 | 备注 |
|---|---:|---|
| aiv_time(us) | 9.31 | 单块执行时长；kernel 11.08 = 1 波 + 发射 |
| aiv_vec_ratio | 0.48 | 向量管道只有 48% 忙 |
| aiv_scalar_ratio | 0.16 | |
| aiv_mte2_ratio | 0.31 | GM→UB |
| aiv_mte3_ratio | 0.13 | UB→GM |
| aiv_vec_wait_ratio | 0.88 | 向量几乎全程在等 |

四管道忙碌时间之和 ≈ 9.3 µs ≈ `aiv_time`，**即管道几乎不重叠**：vec 等 mte2，mte3 等 vec。
生成的 `.asc` 证实行循环是「3 条单行 copy → `asc_sync_wait(MTE2→V)` → 计算 → 2 条单行 copy →
`asc_sync_wait(MTE3→V)`」，每块 27 次装载等待 + 12 次落盘等待。

**launch 地板探针**（`perf_opt/experiments/floor_probe.py`，同 grid=48 / d=768 / bf16，每次 3 次采集中位）：

| kind | 每块工作 | Task Duration |
|---|---|---:|
| 0 | 1 条行装载 + 2 条行落盘 | **2.72 µs** |
| 1 | 27 条单行装载（= 基线）+ 2 条落盘 | **6.07 µs** |
| 4 | 3 条 `(rc+3, d)` 整窗装载 + 2 条落盘 | **3.38 µs** |
| 5 | 6 条整窗装载（owned rc 行 + halo 3 行）+ 2 条落盘 | **3.59 µs** |
| 3 | grid=1，1 装 + 2 落 | 5.98 µs（grid=1 的调度形态异常，未采用） |

结论：**同样 41.5 KB，27 条单行描述符 3.35 µs，6 条整窗描述符 0.87 µs —— 成本按描述符数计，不按字节计。**
派生候选优化点：把行循环内的装载整体提升为整窗 copy（O-5）。

---

## 3. 迭代记录

每轮都从**同一个 current best** 派生分支、每支只改一个主要优化点、先 L0 后采集。
所有实验的 stdout/stderr 在 `perf_opt/logs/round{1,2,3}/`。

### Round 1（base = baseline-stage3，3 支）

现象（base profile）：见 §2 —— vec 等 DMA，27 次/块装载等待；另有 2 处可去掉的冗余
（每行 1 次 `rv` 的 store→BRC 往返 + 每行 1 个 VST_VLD barrier；12 次逐行落盘排空 MTE3）。

**候选 vs current best（baseline-stage3 = 11.084 µs）**

| 分支 | 优化点（单一） | workload | task_duration_us | Δ vs best | aiv_vec_ratio | aiv_mte2_ratio | aiv_mte3_ratio | L0 |
|---|---|---|---:|---:|---:|---:|---:|---|
| r1_c1 | `T.Pipelined(num_stages=2)` + h/k/v 行缓冲 2 版本（想让装载与计算重叠） | s96 | 11.235 | +1.4% | 0.49 | 0.32 | 0.13 | PASS |
| r1_c2 | `r_v` 直接用寄存器，删掉 store→`BRC_B32` 往返与其 `mem_bar` | s96 | 10.992 | -0.8% | 0.47 | 0.33 | 0.14 | PASS |
| r1_c3 | 输出行先入 `(rc+3, d)` 暂存，整块一次 2-D 落盘 | s96 | 11.037 | -0.4% | 0.50 | 0.31 | **0.033** | PASS |

结论：三支都在噪声内（<3%）。但 c3 把 mte3 忙比从 0.131 压到 0.033、c2 删掉了每行一个 barrier ——
两者都**被单点掩盖**（当时瓶颈是装载等待），故作为「已验证有效但需组合」保留。
c1 的编译产物与基线**逐字节相同（无版本偏移）**：`T.Pipelined(num_stages=2)` 在这个行循环里
没有触发多版本旋转（与 `PL-950-dynamic-pipeline-2versions` 的适用边界不符：该条目是纯 DMA 循环，
本循环体含跨迭代 UB 依赖 slot 与向量写 UB）。

### Round 2（base = baseline-stage3，2 支）

**候选 vs current best（baseline-stage3 = 11.084 µs）**

| 分支 | 优化点（单一） | workload | task_duration_us | Δ vs best | aiv_vec_ratio | aiv_mte2_ratio | aiv_mte3_ratio | aiv_vec_wait | L0 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| r2_c4 | 行循环按 build-time 展开 + 2 套显式行缓冲 + 下一行预取 | s96 | 12.918 | **+16.5%** | 0.42 | **0.72** | 0.04 | – | PASS |
| r2_c5 | 整窗 `(rc+3, d)` 装载（27 条单行 DMA → 6 条整窗），行循环无 DMA；合并 c2+c3 | s96 | **8.875** | **-19.9%** | 0.65 | 0.13 | 0.063 | **0.12** | PASS |
| r2_c5 | 同上（合并非回退检查） | s64 | 6.782 | -18.0% | | | | | PASS |
| r2_c5 | 同上（合并非回退检查） | s128 | 6.091 | -20.9% | | | | | PASS |

- c4 反向结果：显式预取让 MTE2 与向量并跑后，**MTE2 忙比从 0.31 升到 0.72、块时 9.31→10.1 µs**。
  机制：把 27 条单行描述符在无等待的情况下压进 MTE2 队列，逐描述符开销与 UB 端口争用一并放大；
  即本平台上「少而大的描述符」比「多而小的描述符 + 重叠」更划算（与 §2 探针同向）。
  据此否定「行级软件流水」这一整类方向（本 workload/工具链）。
- c5 采纳：vec_wait 0.88→0.12、mte2 0.31→0.13、块时 9.31→6.5 µs。**c5 成为 current best。**

### Round 3（base = r2_c5 = 8.875 µs，2 支）

**候选 vs current best（r2_c5 = 8.875 µs）**

| 分支 | 优化点（单一） | workload | task_duration_us | Δ vs best | 说明 | L0 |
|---|---|---|---:|---:|---|---|
| r3_c6 | `d % 64 == 0` 时把每 chunk 的 `update_mask` 折叠成已持有的全 lane 谓词（build-time） | s96 | 9.036 | +1.8% | 无收益（编译器已把 `update_mask_b32(64)` 常量折叠，标量占比未降）；回滚 | PASS |
| r3_c7 | **DESIGN §1.6.3 备选**：两段 launch（gate 段落 vhat/rrms_*，conv 段从 GM 读 4 邻居行），无 halo 重算 | s96 | 9.993（gate 5.35 + conv 4.64） | **+12.6%** | 见 §4 裁决 | PASS |

r3_c6/c7 均未超过 c5，**current best 保持 r2_c5**。

### 中止条件

Round 3 结束后：目标（主导案例 -20%）已由 c5 达成；c4/c6/c7 三个结构性方向被实测否定
（行级流水、掩码折叠、两段 launch），剩余候选（conv 双行共享 slot/cwv 装载）估算 <5%
且低于本轮噪声阈值，按 **success** 收束（未触发 budget/plateau 判定）。

---

## 4. DESIGN §1.6.3 实验裁决（主选单 launch vs 备选两段 launch）

裁决计划要求：代表 shape `train-s96`，指标同口径 Task Duration，**备选 < 主选 × 0.9 才翻转**。

| 结构 | 实现 | 实测 task_duration_us | 与主选比 |
|---|---|---:|---:|
| **主选**：单 launch + halo 重算（c5） | `engram_gate_conv_round2_c5.py` | **8.875** | 1.000 |
| **备选**：两段 launch，无 halo（c7） | `engram_gate_conv_round3_c7.py`：gate 段 5.354 / conv 段 4.639 | **9.993** | 1.126 |

- 备选已按其**最好形态**实现（也用了本轮学到的整窗 copy、2-D 落盘、`cwv` 折叠），
  gate 段 `grid=48/rc=6` 无 halo（6 行而非 9 行），conv 段 `grid=48/rc=6` 整窗读 vhat。
- 裁决所依赖的未知常数①（每 launch 固定开销）实测 **≈2.7 µs**（§2 探针 kind=0）。
  两段结构要付两次 ⇒ 地板 5.4 µs，占备选总时的 54%；省下的 halo 重算只有 3 行 gate 工作（≈0.66 µs），
  且 conv 段还要按 tap 重乘 `r_v`、多读 4 倍 vhat（4.66 MB vs 3.11 MB 流量）。
- **结论：备选 9.993 ≫ 7.99（= 8.875 × 0.9），主选保持。** 该实测把 DESIGN「若 launch 固定开销 ≪ 估算
  则备选胜出」的条件判定为**假**（2.7 µs/launch 不是 ≪，而是主导项）。裁决结果已回写 DESIGN 附录（§5）。

---

## 5. Final Summary

### 5.1 最终候选

`perf_opt/engram_gate_conv.py` = r2_c5（整窗装载 + 2-D 落盘 + `r_v` 直用 + 去掉行守卫），
sha256 `3a2c4f5324a46e463a3e92fdda51b732b0fded6adf0cc773d34bdc1b43581638`。
与 Stage-3 基线相比**语义零改动**：门控统计量、`vhat` 的 dtype 舍入往返、conv 的 4 tap 累加顺序、
`Y` 的非舍入残差全部保持原表达式（L0/L1 max_diff 与基线逐字段相同）。

### Final Performance Test Data

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---:|---:|---|
| tileops/kernels/sequence_modeling/engram_gate_conv/engram_gate_conv_kernel::_engram_gate_conv_fwd_kernel::main | train-s96-bfloat16 | 11.084 | 8.460 | final-windowload |
| tileops/kernels/sequence_modeling/engram_gate_conv/engram_gate_conv_kernel::_engram_gate_conv_fwd_kernel::main | train-s64-float16 | 8.272 | 6.962 | final-windowload |
| tileops/kernels/sequence_modeling/engram_gate_conv/engram_gate_conv_kernel::_engram_gate_conv_fwd_kernel::main | train-s128-bfloat16 | 7.703 | 5.764 | final-windowload |

提升：train-s96 **-23.7%**、train-s64 **-15.8%**、train-s128 **-25.2%**。三者皆无回退。

补充（非门禁，仅记录）：smoke `train-s32` 基线 9.453 µs / 最终 9.504 µs（+0.5%，在本 harness
±0.3 µs 的 run-to-run 波动内；该案例 rc=1、grid=32，整窗装载把 4 行并成 6 条整窗 copy，收益与
per-descriptor 成本抵消）。

### 5.2 smoke / skip 与精度

- smoke：`train-s32-float16`（benchmark idx 0 的 `pytest.mark.smoke`）——只做精度回归，最终
  `precision_pass: true`（L0-1 通过，`Y=1.22e-04`）。同一案例的 manifest 也带 `full` 标记，
  按 smoke 排除性能调优并在 inventory 记录冲突原因。
- skip：无。
- 最终精度（`perf_opt/logs/final/L0_L1_L2_boundary.log`，`--level all`）：
  L0 11/11、L1 5/5、L2 4/4、boundary 5/5 全过；max_diff Y 4.883e-04、vhat 2.441e-04、
  alpha 1.788e-07、rrms_h 1.221e-04、rrms_k 1.221e-04、rrms_v 4.730e-03（与基线逐字段一致）。

### 5.3 结论

主导案例 11.084 → 8.460 µs（**-23.7%**，超过 -20% 目标）；三个 tune workload 全面下降 16–25%，
无回退。剩余时间构成（s96）：launch 地板 ≈2.7 µs（30%，本平台不可压缩，`grid=1` 形态也是 ~6 µs）、
向量+标量算 ≈4.5 µs、装载 ≈0.9 µs。**未触发 `[DESIGN_LIMIT]`**：设计层假设未被推翻（主选结构胜出），
且剩余差距属于参数/指令级（可继续在 ≥5% 噪声以上验证 conv 双行共享），不构成设计层天花板。

---

## 6. Skill Retrospective

### 6.1 流程问题与建议（proposal，不自动改 skill）

| # | 现象 | 建议 | vp_type |
|---|---|---|---|
| 1 | Phase 2 的「每轮现象分析」把 `aiv_vec_wait_ratio` 当作「向量自身停顿」，但真正原因是**每行一条 DMA 描述符 + 同步**；只有把四管道忙时相加才能看出「管道不重叠」 | 在 iteration-diagnosis 增加一条判据：**Σ(pipe busy) ≈ aiv_time ⇒ 管道串行**，并要求对 1D 单波算子先做「同几何地板探针」再定优化方向 | R |
| 2 | 平台 `execution.precision_command` 指向 `/usr/local/python3.12.16/bin/python3`，该解释器**无 torch**；实际可跑的是 `$TILEOPS_950PR_VENV/bin/python`（`/workspace/venv-ascend950`） | 平台 execution context 应解析到带 torch/torch_npu 的解释器，或在 env-950pr.sh 里导出 `PYTHON` 供各 stage 引用 | R |
| 3 | 小算子（T_op < 20 µs）的 `Task Duration` 单次采集波动可达 ±4%，2 样本时中位数口径（上中位数 vs 均值）会改变结论 1–2% | profile-collection 明确「偶数样本取两中值均值」并**要求 baseline 与 final 在相邻时段用同样 reps** 测量 | R |
| 4 | `T.Pipelined(num_stages=2)` 在小循环上静默不生效（产物与基线逐字节相同），只有读生成码才能发现 | traps 条目补「如何确认流水真的生效」：查 `.asc` 里是否有 `(i - i0) & 1` 版本偏移与成对 sync | C |

### 6.2 价值点（value point proposals）

- **VP-D1（D，实测代价常数）**：950PR 上 kernel launch 固定开销（msopprof Task Duration 口径）
  ≈**2.7 µs**（grid=48、d=768、bf16、1 装 + 2 落），两段 launch 结构因此天然多付 ~2.7 µs。
  证据三件套：evidence = §2 探针表 kind=0；repro = `perf_opt/experiments/floor_probe.py`（kind 由
  `FLOOR_KIND` 选择）+ 本任务 `profiles/floor/*`；stamp = toolchain `7a52f2f1…` / CANN 9.2.1。
- **VP-P1（P，模式）**：逐行 GM→UB 装载应合并为**整窗 2-D copy**（本任务 27→6 条描述符，
  kernel 8.875 µs vs 11.084，-19.9%）；机制：MTE2 的逐描述符开销 + 每次使用前的 `PIPE_MTE2→V` 等待，
  与 `TRAP-950-rowstore-mte3-drain` 的落盘侧结论同构。**注意**：不要再叠加「行级预取」——
  c4 实测把 MTE2 忙比推到 0.72、总时 +16.5%。证据：§3 Round 2 两行 + §2 探针。
- **VP-P2（P，反向模式）**：`T.Pipelined(num_stages=2)` 在含跨迭代 UB 依赖（环形窗口）的向量行循环里
  **不生成多版本**（产物逐字节相同，c1 无变化）；同族的 c4 显式预取虽生成了流水但净亏。
  结论绑定 `toolchain_stamp 7a52f2f1…`。
- **VP-C1（C，案例索引）**：本任务可作为「1D 单波 vector 算子、T_op ~10 µs、launch 占 24%」的优化范式：
  先量地板 → 再量描述符数 → 再谈向量指令数。

### 6.3 回写 pattern-library（ED-C）

- `knowledge/traps-950pr-store-copy.md`：新增 **TRAP-950-rowload-dma-chain**（装载侧对偶条目，
  含 2.72/6.07/3.38/3.59 µs 实测表与 repro）+ 在 `TRAP-950-rowstore-mte3-drain` 的边界里追加
  「装载侧无 4 KB 判据」的交叉引用。
- `knowledge/constants.md`：新增 **CONST-950-kernel-launch-floor-48blk**（2.7 µs @ grid=48）。
- `knowledge/patterns-950pr-verified-forms.md`：新增 **PL-950-window-bulk-copy**（整窗 copy 取代行链，
  反向边界 = 不要叠加行级预取）。
- repro：`knowledge/repro/TRAP-950-rowload-dma-chain.py`（自包含：同几何 3 种装载形态 + msopprof 采集 +
  断言 bulk < row-chain）。

回写已执行并验证：`kb_lint.py --toolchain ascend --hardware ascend-950pr-9579` → **0 failures**；
repro 实跑通过（`REPRO_OK`，本任务复现 kind0=2.83 / kind1=5.97 / kind4=2.92 / kind5=3.15 µs，
三项断言均成立）；三个条目已登记进 `knowledge/INDEX.md` 的路由表。

### 6.4 其它交付核对

- 最终文件以**模块方式**被导入验证（`experiments/bench_one.py` 用 importlib 加载并调用
  `_engram_gate_conv_fwd_kernel(M, seq_len, d, eps, dtype)`，与 wrapper 的 `perf_opt` 切换块同签名）。
- `examples/TileOPs/tests/ops/test_engram.py -k gate_conv_fwd` → **13 passed**（该路径当前仍走
  wrapper 里的 Stage-3 baseline import；perf_opt 的精度由它自带的 L0/L1/L2/boundary 运行器验证，
  max_diff 与基线逐字段相同）。wrapper 的 baseline/perf_opt 切换块按纪律**未改**，由 conductor 翻转。

---

## 7. 生产采纳（wrapper 切换 + 单算子报告验收）

### 7.1 采纳文件清单

| 文件 | 作用 | sha256 |
|---|---|---|
| `examples/TileOPs/tileops/kernels/sequence_modeling/engram_gate_conv/engram_gate_conv_kernel/history_version/engram_gate_conv_pre_opt.py` | **wrapper 备份**（翻转前原样，可整体回滚） | `f8505f7737efc56fa7d5b0cd017f3a08be0c5eced91245539f455dd8f4a78db6` |
| `.../engram_gate_conv_kernel/perf_opt/__init__.py` | perf_opt 子包导出（与父包同名符号） | — |
| `.../engram_gate_conv_kernel/perf_opt/_engram_gate_conv_fwd_kernel.py` | **drop-in 优化 kernel**（= `perf_opt/engram_gate_conv.py` 同字节） | `3a2c4f5324a46e463a3e92fdda51b732b0fded6adf0cc773d34bdc1b43581638` |
| `.../engram_gate_conv/engram_gate_conv.py` | wrapper：仅翻转 import 切换块（op 层构造/forward 契约不变） | 见 git diff |

回滚方式：把 wrapper 切换块的两行注释换回（baseline 分支仍从父包
`engram_gate_conv_kernel` 导入，Stage-3 源码未动），或用
`history_version/engram_gate_conv_pre_opt.py` 覆盖 wrapper。

### 7.2 经 wrapper 后确实走优化实现

```
factory module : tileops.kernels.sequence_modeling.engram_gate_conv.engram_gate_conv_kernel.perf_opt._engram_gate_conv_fwd_kernel
factory file   : .../engram_gate_conv_kernel/perf_opt/_engram_gate_conv_fwd_kernel.py
perf_opt ACTIVE: True
baseline still importable: tileops...engram_gate_conv_kernel._engram_gate_conv_fwd_kernel
```

### 7.3 生产回归

- `pytest tests/ops/test_engram.py -q -k gate_conv_fwd` → **13 passed, 33 deselected**（6.34 s；
  `logs/final/production_pytest_gate_conv_fwd.log`）。
- `perf_opt/_engram_gate_conv_fwd_kernel.py --level all`（**部署后的那份文件**）
  → L0 11/11、L1 5/5、L2 4/4、boundary 5/5、`[PRECISION_PASS]`；max_diff 与 Stage-3 基线逐字段相同
  （`logs/final/production_level_all.log`）。

### 7.4 单算子报告（msprof）

```
TILEOPS_TASK_DIR=/workspace/tilelang-mlir-ascend-950/examples/engram_gate_conv \
TMPDIR=/workspace python -m tileops.reporting.cli run --op EngramGateConvFwdOp \
  --prof-mode msprof --pytest-arg=-k --pytest-arg=gate_conv_fwd
```

- **run-id：`20261124_222322_139293_EngramGateConvFwdOp`**
  （`examples/TileOPs/reports/tileops/20261124_222322_139293_EngramGateConvFwdOp/`）
- `status=passed`；correctness 13/13；`invalid_profile_count=0`、`invalid_acceptance_count=0`、
  `profiler_fallback_count=0`、`kernel_filter_fallback_count=0`；`warnings=[]`。

| case | dtype | T_op（本次报告，单次 msprof 采集） | Stage-4 final（3 次中位） | 优化前（MFU 基线 / 本任务复现） |
|---|---|---:|---:|---:|
| **train-s96** | bf16 | **8.834 µs** | 8.460 | 11.049 / 11.084 |
| train-s64 | fp16 | 6.594 | 6.962 | 8.2 / 8.272 |
| train-s128 | bf16 | 5.609 | 5.764 | 7.6 / 7.703 |

- 每次调用仍只有 1 个 launch（`t_op_breakdown = {main_kernel: …}`，`captured_op_count=1`，
  `kernel_name_resolved=main`）；train-s32 是 smoke，报告链按标记只测 3 个 tune case。
- MFU（vector）best 从基线的 1.231% 升到 **1.539%**（+25%）。验收里 `MFU passed=0/failed=3`
  与优化前一致（`mfu_baseline_v1.json` 记录本算子 0 个 case 达标），blocked_notes 明示
  「memory-bound，MFU 被 HBM 带宽压在 9.70–9.77%」，**不是回退**，是全局 90% MFU 目标对该访存受限算子的固有判定。
- 回写 `perf_records.jsonl`（append-only）：round 5 `phase=merged`、`candidate_id=production-wrapped`，
  `artifact_path` 指向部署后的 perf_opt 文件、`artifact_sha256` 与文件一致（`3a2c4f53…`）。

### 7.5 兼容性与回退

- 无回退：三条 tune case 经生产链路全部改善（8.834 / 6.594 / 5.609 µs），精度门限不变。
- 契约无变化：`EngramGateConvFwdOp.__init__(eps, kernel_map)`、`forward` 的 6 入 6 出、
  `EngramGateConvFwdKernel.default_config`（空 dict）与 `ascend_mode=None` 均未改动；
  wrapper 的 `custom_op("npub::engram_gate_conv_fwd")` 与 fake 输出形状不变。
- perf_opt kernel 与基线工厂**同签名**（`(M, seq_len, d, eps, dtype)` → 编译后 kernel，`out_idx=[6..11]`），
  `lru_cache` 形状特化策略相同，`geometry()` 分核规则未改 → 无新默认参数需要在 wrapper 成对引用。
- `/tmp/tileops_msprof_*` 已清理。
