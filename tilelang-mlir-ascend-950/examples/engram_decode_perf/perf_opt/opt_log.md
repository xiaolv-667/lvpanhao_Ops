# opt_log.md — `EngramDecodeFwdOp` Stage 4 性能调优日志

- 任务：`examples/engram_decode_perf`（scenario=optimize, mode=full；平台 `ascend-950pr-9579` / `ascend`，单卡）
- 基线（只读）：`tileops/kernels/sequence_modeling/engram_decode/engram_decode_kernel/_engram_project_kernel.py`
  （sha256 见 `perf_records.jsonl` 的 `artifact_sha256`）与 `_engram_step_kernel.py`
- 设计：`examples/engram_decode/_engram_step_kernel/DESIGN.md`（+ `_engram_project_kernel/DESIGN.md`）
- 产物：`perf_opt/EngramDecodeFwdOp.py`（两个 launch 的工厂 + `project_default_config` /
  `step_default_config` / `ALIGNMENT`，单文件 drop-in，供 wrapper 的 perf_opt 切换块成对引用）
- **测量口径**：任务平台 `measurement` 规定的 `msopprof` kernel-only `Task Duration(us)`
  （`raw_evidence_field = msprof_raw_path`），`--launch-count=2 --warm-up=0 --kernel-name=engram
  --aic-metrics=Roofline`，**一次采集只跑一次算子调用**；`T_op = engram_project_kernel +
  engram_decode_step_kernel`（两次 launch 求和，与 `mfu_baseline_v1.json` 的 `breakdown_us` 同口径）。
  驱动 `perf_opt/experiments/bench_one.py`。不使用 NPU event、不用端到端耗时。
- **噪声口径**：本算子 T_op 5–9 µs，单次采集的 run-to-run 波动经实测（同一份代码重复采）为
  **per-launch 中位 ±8–12%、T_op 中位 ±4%**；因此每个数值取「若干次独立采集进程的中位数」，
  偶数样本取中间两个的均值，样本一律写入 `perf_records.jsonl` 的 `samples_us`；**<5% 的差异**
  另用同会话交错 A/B（§4）裁决，不据单臂单次采样下结论。
- 主指标：`msprof_task_duration`（us，两次 launch 之和）。

---

## 0. Phase 0 — 上下文加载与首轮必查项

- 工具链戳：tilelang 0.1.15 / CANN 9.2.1（compiler 12.3.0）/ raw `7a52f2f1…`；
  设备 `Ascend950PR_9579`，`torch.npu.get_device_properties(0)` 实测 cube 28 / vector 56。
- 知识索引消费（`knowledge/INDEX.md` + `kb_search`）：
  `CONST-950-kernel-launch-floor-48blk`（launch 地板 ≈2.7 µs / 1 次 launch）、
  `PL-950-window-bulk-copy`（整窗 2-D copy 取代逐行装载，-19.9%）、
  `TRAP-950-rowload-dma-chain`（描述符数计费）、`CONST-950-gemm-mte2-burst`
  （bf16 L1 操作数 GEMM 的 MTE2 有效带宽：128 B 行段 70 GB/s/core、256 B 101–105 GB/s/core，
  HBM 1.42–1.48 TB/s）、`PL-950-window-bulk-copy` 的反向边界（不要叠加行级预取）。
  全部按「数据非指令」消费。
- **E-5 版本失效核对**：`kb_stale_check.py --task-dir examples/engram_decode_perf` 把 batch2 集群条目
  整体判 stale（判据绑定 toolchain revision，历史任务同现象）；本任务引用的条目**全部用本任务同口径
  实测重验**：launch 地板复现 2.69–2.96 µs（§2.2，与 CONST-950-kernel-launch-floor-48blk 一致）。
- 算子类型：**mix（launch 1 = 1 AIC + 2 AIV，msopprof `Op Type=mix`、`Mix Block Dim=32`）+
  vector（launch 2 = 纯 AIV，`Op Type=vector`）**。
- **首轮必查项 1（向量化轴与布局重估）**：两个 kernel 的向量化轴都是「最内连续轴 `d`」，
  `d_pad ∈ {256, 512, 1024}` 全部 `% 64 == 0`，无尾 lane 浪费；`d_pad > d` 的掩码路径只在
  `d=600` 这类非 2 幂宽度出现（L1 覆盖）。**未发现需要重估布局的证据**（§2 的实测把时间
  全部归因到 launch 地板、权重流与 UB↔GM 往返，没有一条落在向量化程度上）。
- **首轮必查项 2（陷阱版本戳/origin_task）**：`CONST-950-gemm-mte2-burst`
  （origin_task `shared_expert_mlp-20261124T132107Z`）的「每行 DMA 段长决定 MTE2 带宽」在本任务
  直接命中并**被实测复核**：本 kernel 的 W 流 = 每 128 B 段（`block_n=64` × fp16）跨 1024 B 行距，
  16 块 × 98 GB/s ≈ 1.6 TB/s，与条目的 128 B/70 GB/s、256 B/101 GB/s 区间自洽（§2.3）。
  `PL-950-window-bulk-copy`（origin_task `engram_gate_conv-20261124T171256Z`）是本任务 cache shift
  重写的直接依据。
- **首轮必查项 3（DESIGN §1.6.3 实验裁决）**：`_engram_step_kernel/DESIGN.md` §1.6.3 留下
  两个未实测常数：① 单次 launch 开销「≈5 µs，**未实测假设**」；② `AscendAllReduce` 单次代价。
  本任务实测 ① = **2.69–2.96 µs**（§2.2），见 §1.3 的偏差行与 §4 的裁决结论。
  DESIGN 已否定的两条结构备选（候选 4 分 d 跨核、候选 5 与 project 段融合）在本任务**重算后仍成立**
  （§5.4），因此「主选保持」被实测固化，而不是被推翻。

### 0.1 基线结构（冻结，不改）

- **launch 1 `_engram_project_kernel`**：1 维 grid = `m_tiles · 2 · n_tiles`（K/V 各半），每块一个
  `(batch tile=16, d tile=64)` 输出块，`block_k=256`、`num_stages=2`；W 以 `transpose=True`
  （codegen = `asc_copy_gm2l1_dn2nz`）装进 L1；cache 左移写在 `which==0` 的
  `m0==0` 块里，`for b in T.serial(batch)` × (gm2ub → sync → ub2gm → sync)，每块搬
  `(mcl-1, block_n=64)` 的窄切片。
- **launch 2 `_engram_decode_step_kernel`**：`T.Kernel(batch)` 每核一行，`T.SimtVF(threads=min(512,tile))`
  + 4 条并发 `T.alloc_reducer(op="sum")`（`S_h, S_k, P, S_v`），2 个元素级 pass，tap 逐行装载，
  写 `y_t` 与 `new_conv_state` 末行。

---

## 1. Phase 1 — Workload Inventory 与 Baseline

### 1.1 Inventory

来源：`benchmarks/ops/bench_engram.py::test_engram_decode_bench`，参数由
`tileops.manifest:load_workloads("EngramDecodeFwdOp")` 展开 = **5 个案例**，完整清单见
`workload_inventory.json`。kernel_id 唯一（两次 launch 一起记一次 T_op）：
`…/engram_decode_kernel::_engram_project_kernel+_engram_step_kernel`。

| workload_id | shape（e_t / h_t / conv_state） | d_mem / d / mcl / w / dil / L | dtype | kind |
|---|---|---|---|---|
| decode-s1-float16 | (1,256) / (1,256) / (1,9,256) | 256/256/9/4/3/9 | fp16 | **smoke**（parametrization idx 0 带 `pytest.mark.smoke`） |
| serving-b1-float16 | (1,512) / (1,256) / (1,12,256) | 512/256/12/4/3/12 | fp16 | tune |
| serving-b4-float16 | (4,1024) / (4,512) / (4,20,512) | 1024/512/20/4/5/20 | fp16 | **tune（主导：T_op 最大、FLOPs 占 5 案例 88%）** |
| serving-b8-bfloat16 | (8,512) / (8,256) / (8,18,256) | 512/256/18/4/3/18 | bf16 | tune |
| decode-partial-cache-float16 | (2,512) / (2,256) / (2,5,256) | 512/256/12/4/3/**5** | fp16 | tune |

smoke 只做精度回归、不采集性能、不选 winner；无 skip 案例。

### 1.2 Baseline（round 0，各 3 次独立采集，中位数）

| workload_id | project_us | step_us | **T_op_us** | profile |
|---|---:|---:|---:|---|
| serving-b4-float16 | 4.878 | 3.418 | **8.297** | `profiles/baseline/_engram_project_kernel+_engram_step_kernel_serving-b4_r*` |
| serving-b8-bfloat16 | 6.672 | 2.952 | **9.466** | 同上 `serving-b8` |
| serving-b1-float16 | 3.133 | 2.972 | **5.989** | 同上 `serving-b1` |
| decode-partial-cache-float16 | 3.081 | 3.105 | **6.186** | 同上 `decode-partial-cache` |
| decode-s1-float16（smoke，非门禁） | 2.655 | 2.874 | 5.529 | 仅记录，不入 `perf_records.jsonl` |

交叉核对 `examples/batch2/mfu_baseline_v1.json`（`t_op_from = forced-unfiltered capture`）：
serving-b4 8.727（5.193+3.534）、serving-b8 9.544（6.292+3.252）、serving-b1 6.305、
decode-partial-cache 5.854；本任务同口径复现 8.297 / 9.466 / 5.989 / 6.186，偏差 ≤5.5%（同量级噪声内）。

**排序**：serving-b4（T_op 最大 + FLOPs 88%）→ serving-b8（9.47，第二）→ serving-partial(6.19) →
serving-b1(5.99)；排序只影响先后，四个 tune workload 全部有结论。

### 1.3 设计估算 vs 实测（D-2 回填）

DESIGN §1.6.0 给出 step 段「估算下界 ≈ 6 µs / workload」，其中**发射项 = 1 次 launch ≈ 5 µs
（文中明确标注「未实测假设」）** + 2 pass × 0.5 µs；并据此给出「整算子 = 2 次 launch ≈ 11 µs」。

| 量 | DESIGN 估算 | 本任务实测 | 偏差 |
|---|---:|---:|---|
| 单次 launch（kernel-only Task Duration） | ≈5 µs（未实测假设） | **2.69 / 2.96 / 2.89 µs**（grid=4/8/16 探针，§2.2） | **−42%（估算偏高 1.8×）** |
| step 段（1 launch + 2 pass + 4 归约） | ≈6 µs | **2.87–3.42 µs** | −45% |
| 整算子（2 launch） | ≈11 µs | **8.30 µs（b4）** | −25% |

失准项是**发射项常数**（DESIGN 已标注为未实测假设），流量项与容量项没有失准。
该常数偏高**不改变** DESIGN 的两条结构否决（§5.4 重算仍成立），因此按
`_shared/standards/perf-feedback.md` §1 的触发条件（需「结构性加速估计 > 2×」或
「实测推翻设计判定」），本任务**不产出** `perf_feedback.md`；失准常数按 ED-C 回写
`knowledge/constants.md`（§7）。

---

## 2. 诊断（Round 1 前，基于 baseline profile + 三个诊断探针）

### 2.1 逐 launch 分解

| workload | T_op | project | step | 每次调用 launch 数 |
|---|---:|---:|---:|---:|
| serving-b4 | 8.297 | 4.878 | 3.418 | 2 |
| serving-b8 | 9.466 | **6.672** | 2.952 | 2 |

**最重要的一眼**：serving-b8 的权重流量只有 b4 的 1/4（0.5 MB vs 2 MB）、grid 只有 8 vs 16，
project 却比 b4 慢 **37%** → 疑点不在 Cube/带宽，而在 cache shift。

### 2.2 launch 地板探针（`perf_opt/experiments/probe_floor.py`）

同几何最小 kernel（每块 1 条行装载 + 1 条行落盘，无其他工作），每次 3 次独立采集中位：

| grid | 行宽 | Task Duration |
|---:|---:|---:|
| 16（= b4 project grid） | 512 | **2.894 µs** |
| 8（= b8 project grid） | 512 | **2.961 µs** |
| 4（= b4/b8 step grid） | 512 | **2.690 µs** |

→ 本算子两次 launch 的**结构性地板 = 5.4–5.9 µs**，占 baseline T_op 的 65–70%。
与 `CONST-950-kernel-launch-floor-48blk`（48 块 ≈2.72–2.83 µs）一致，说明该常数对 grid 4–48 普遍成立。

### 2.3 project 段拆解（`experiments/mkprobe.py` 生成的两个诊断探针）

| 形态（serving-b4） | project_us | 差值 | 结论 |
|---|---:|---|---|
| baseline（GEMM + shift） | 4.878 | — | — |
| `probe_project_noshift`（去掉 shift 与其 AIV 代码） | **4.199** | −0.679 | shift 项 0.68 µs |
| `probe_project_tinyshift`（shift 缩成 1 条 1 行 copy，AIV 代码仍在） | **4.693** | −0.185 | **0.49 µs 是 mix(AIC+AIV) 提升代价，0.19 µs 才是 shift 字节** |
| （同一探针在 serving-b8） | 2.999 | −3.673 | **b8 的 shift 项 3.67 µs** |

b8 的 3.67 µs 归因（读 `.asc` 实测）：shift 只在 `n_tiles=4` 个块里，每块 `for b in 0..7` 串行
8 个 (gm2ub → `asc_sync_wait(MTE2,MTE3)` → ub2gm → sync)，即 **8 条依赖往返**；b4 是 8 个块 ×
4 条 → 0.68 µs。**成本按「每块串行往返条数」计，不按字节计**（与 `TRAP-950-rowload-dma-chain` /
`PL-950-window-bulk-copy` 同构，本任务把该结论从「装载行循环」推广到「按 batch 循环的搬运」）。

### 2.4 project 段剩余项的归因（为什么 b4 已无空间）

b4 project 4.878 = 地板 2.89 + **W 流 1.31** + mix 提升 0.49 + shift 字节 0.19。
W 流 = 2 MB / 16 块 / 1.31 µs = **98 GB/s/core**，正落在 `CONST-950-gemm-mte2-burst` 的
256 B 行段区间（101–105 GB/s/core），即 MTE2 满速；聚合 1.6 TB/s 已接近 HBM 有效带宽（1.42–1.48 TB/s）。

### 2.5 step 段

step = 2.69（地板）+ 0.7（工作），且对 `threads ∈ {128,256,512}` 与 tap 描述符数（3 条单行 vs
1 条整窗）都不敏感（§3 Round 2）→ 主要是 4 条 `AscendAllReduce` 的屏障链 + 2 个 pass 的固定延迟。

---

## 3. 迭代记录

每轮从**同一个 current best** 派生分支、每支只改一个主要优化点、先 L0 后采集。
所有实验 stdout/stderr 与 profile 在 `perf_opt/logs/round{1,2}/`、`perf_opt/profiles/round{1,2}/`。

### Round 1（base = baseline-stage3，4 支）

**候选 vs current best（baseline-stage3：b4 = 8.297 µs，b8 = 9.466 µs）**

| 分支 | 优化点（单一） | workload | T_op_us | project_us | step_us | Δ vs best | 利用率/访存指标 | L0 |
|---|---|---|---:|---:|---:|---:|---|---|
| r1_c1 | cache shift 迁到 `batch` 个**专用尾部块**，每块 1 条整行窗口 copy 对 | b8 | **6.650** | 3.364 | 3.286 | **−29.8%** | （Roofline 采集；mix 块 12 → 见 §3.1） | PASS |
| r1_c1 | 同上 | b4 | 8.467 | 5.073 | 3.394 | +2.0% | grid 16→20 反而变慢 | PASS |
| r1_c2 | 同上但**折进前 `batch` 个已有块**（grid 不变） | b8 | **6.422** | 3.415 | 3.007 | **−32.2%** | | PASS |
| r1_c2 | 同上 | b4 | 8.183 | 4.905 | 3.278 | −1.4% | | PASS |
| r1_c2 | 同上（合并非回退检查） | b1 | 5.891 | 3.251 | 2.790 | −1.6% | | PASS |
| r1_c2 | 同上（合并非回退检查） | partial | 6.424 | 3.266 | 3.196 | +3.8%（噪声，见 §4） | | PASS |
| r1_c3 | 在 c2 基础上再按**列**切 4 份（`batch×4` 个块，每块 128 列） | b4 | 8.395 | 4.835 | 3.560 | +1.2% | 列切分**无进一步收益** → 证实成本是每块串行往返，不是字节 | PASS |
| r1_c3 | 同上 | b8 | 6.400 | 3.396 | 3.004 | −32.4% | | PASS |
| r1_c4 | **形状自适应**：仅当 `batch > n_tiles` 用整行窗口式，否则保留基线切片式 | b4 | 8.325 | 4.888 | 3.437 | +0.3% | 与基线**逐字节同码**（§3.2） | PASS |
| r1_c4 | 同上 | b8 | **6.400** | 3.310 | 3.090 | **−32.4%** | | PASS |
| r1_c4 | 同上（合并非回退检查） | b1 / partial | 6.028 / 6.225 | 3.206 / 3.062 | 2.803 / 3.209 | +0.7% / +0.6%（同码，噪声） | | PASS |

- r1_c2/c4 采纳为 current best（b8 6.4 µs，其余不变）。c1 的专用块形态在 b4 上 +2%（多 4 个块的调度代价），
  被 c2 的结构替代；c3 的列切分证明 shift 成本不是字节主导。
- **r1_c4 = 最终形态**：`batch > n_tiles` 才切换到整行窗口式。判据来自实测机制——基线的 shift 并行度是
  `n_tiles` 个块，`batch > n_tiles` 时每块要串行 >1 条往返（b8 = 8 条/块）。

#### 3.1 codegen 证据（本任务最强的「不回退」证据）

`perf_opt/experiments/verify_codegen.py` 逐 shape 编译基线 project 与 perf_opt project，
比较生成的 `device_kernel.asc`（归一化编译器内部的 `version_counter_N` 符号名）：

| workload | batch | n_tiles | wide shift | baseline asc sha | tuned asc sha | 逐字节相同 |
|---:|---:|---:|---|---|---|---|
| decode-s1 | 1 | 4 | no | `d0cafbfd…` | `d0cafbfd…` | **True** |
| serving-b1 | 1 | 4 | no | `f7f6a485…` | `f7f6a485…` | **True** |
| serving-b4 | 4 | 8 | no | `a9e1ced4…` | `a9e1ced4…` | **True** |
| decode-partial-cache | 2 | 4 | no | `d8bdb489…` | `d8bdb489…` | **True** |
| serving-b8 | 8 | 4 | **yes** | `078966b7…` | `d3b32594…` | False（shift 区段改写） |
| L1-ktail-519-b17 | 17 | 4 | **yes** | `99b5c35a…` | `c4dfdddd…` | False |

即：**四个非 b8 案例的 device 代码与 Stage-3 基线逐字节相同**（符号名归一化后 sha256 相等），
它们的 T_op 差异只可能是测量漂移（§4 的配对 A/B 也如此显示）。

### Round 2（base = r1_c2 + baseline step，2 支）

| 分支 | 优化点（单一） | workload | T_op_us | step_us | Δ vs best | L0 |
|---|---|---|---:|---:|---:|---|
| r2_c1 | step 段：`w-1` 条单行 tap copy → 1 条连续窗口 copy（11 desc → 9） | b4 | 8.257 | 3.409 | −0.3%（噪声内） | PASS（13/13） |
| r2_c1 | 同上 | b8 | 6.132 | 2.882 | −4.0%（噪声内） | PASS |
| （配置扫描） | `threads ∈ {128,256,512}`（step 启动参数） | b4 | — | 3.370 / 3.486 / 3.428 | 全在噪声内 | 未跑（仅采集） |
| （配置扫描） | project tiling `(block_n,block_k,stages) ∈ {(64,256,2),(128,128,2),(32,256,2),(64,512,1)}` | b4 | 8.485 / 8.305 / 8.574 / 8.209 | project 4.848 / 4.789 / 4.934 / 4.898 | 全在 ±1.7% 内 | 未跑（配置不变语义） |

结论：step 的整窗 tap copy 与 threads 扫描均**无超过噪声的收益**，按「打平时用代码复杂度决胜」
**回滚**（step 段保持 Stage-3 代码）。project 的 tiling 扫描证明 b4 的 project 段**不是 tiling 受限**
而是带宽/地板受限（`block_n=32` 想用 32 块换来更细粒度反而变慢，因为行段 64 B）。

---

## 4. 平区裁决（<5% 差异的交错 A/B，T-2）

每个 case 用同会话 A/B/B/A 交错（A = 冻结基线，B = `perf_opt/EngramDecodeFwdOp.py`，
每臂 2 次独立采集进程 = 2 个样本；`profiles/ab/`）：

| workload | 臂 | project_us（2 run） | step_us（2 run） | T_op_us（2 run） | 合并中位 |
|---|---|---|---|---|---|
| serving-b4 | A | 5.072 / 5.024 | 3.464 / 3.531 | 8.537 / 8.555 | 8.546 |
| serving-b4 | **B** | 4.957 / 4.968 | 3.140 / 3.398 | 8.098 / 8.366 | **8.232（−3.7%）** |
| serving-b1 | A | 3.178 / 3.292 | 2.777 / 2.816 | 5.954 / 6.108 | 6.031 |
| serving-b1 | **B** | 3.168 / 3.130 | 3.010 / 3.083 | 6.178 / 6.213 | **6.196（+2.7%）** |

- **serving-b4**：同会话下 B 的 project 段 −1.7%、T_op −3.7%；但两者 device 代码**逐字节相同**
  （§3.1），所以这 3.7% 只能是漂移/双态（同一份基线代码在两个会话里 project 从 4.878 漂到 5.048）。
  按「<5% 未过协议不得记为 winner」，**b4 记 0.0%（不回退、也无有效增益）**。
- **serving-b1**：同一个 step kernel（未改）在两臂差 9%，说明该量级的单臂差异不可判；
  b1 记 0.0%（同码）。

---

## 5. Final Summary

### 5.1 最终候选

`perf_opt/EngramDecodeFwdOp.py`（sha256 见 `perf_records.jsonl` final 行）= Round 1 的 r1_c4
（project 段形状自适应 cache shift）+ Stage-3 原样 step 段。与基线的**语义零改动**：fp32 `kv`
跨 launch 契约、算术顺序、dtype 舍入点、history 左移的逐位 copy 全部保持；L0/L1/boundary
的 max_diff 与基线一致（§5.3）。

### Final Performance Test Data

（`perf_records.jsonl` 中 `phase=final` 的行；同一最终候选 `final-adaptive-shift`，两次 3 次采集会话合并；
最终工件 `perf_opt/engram_decode.py` 与本文件 `perf_opt/EngramDecodeFwdOp.py` 同字节，sha256
`6c647f7d…`，其两个 launch 的 `device_kernel.asc` 与下表测量时逐字节相同——新增的
golden/L2 harness 只影响 host 侧）

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---:|---:|---|
| tileops/kernels/sequence_modeling/engram_decode/engram_decode_kernel::_engram_project_kernel+_engram_step_kernel | serving-b4-float16 | 8.297 | 8.515 | final-adaptive-shift |
| tileops/kernels/sequence_modeling/engram_decode/engram_decode_kernel::_engram_project_kernel+_engram_step_kernel | serving-b8-bfloat16 | 9.466 | 6.385 | final-adaptive-shift |
| tileops/kernels/sequence_modeling/engram_decode/engram_decode_kernel::_engram_project_kernel+_engram_step_kernel | serving-b1-float16 | 5.989 | 6.094 | final-adaptive-shift |
| tileops/kernels/sequence_modeling/engram_decode/engram_decode_kernel::_engram_project_kernel+_engram_step_kernel | decode-partial-cache-float16 | 6.186 | 6.258 | final-adaptive-shift |

变化（同一表，供阅读）：serving-b8 **−32.5%**；serving-b4 +2.6%、serving-b1 +1.8%、
decode-partial-cache +1.2% 均为**测量漂移**——这三个 shape 的 device 代码逐字节等于基线（§3.1），
同会话配对 A/B 给出相反符号（b4 −3.7%，§4）。

逐 launch（final 两次会话合并中位）：

| workload | project_us | step_us | T_op_us |
|---|---:|---:|---:|
| serving-b8 | 3.354 | 3.036 | 6.385 |
| serving-b4 | 5.106 | 3.410 | 8.515 |
| serving-b1 | 3.203 | 2.888 | 6.094 |
| decode-partial-cache | 3.016 | 3.172 | 6.258 |

**为什么非 b8 案例是 +1~3% 而不是 0.0%**：它们的 `device_kernel.asc` 与基线逐字节相同
（§3.1），差异完全来自会话漂移——本任务实测同一份代码的 project 段中位在 4.79–5.12 µs
（±7%）、step 段在 2.79–3.55 µs（±12%）之间漂移；paired 交错 A/B（§4）给出相反符号（b4 −3.7%）。
按平台噪声阈值（3%）判定：**四个 tune 案例均无回退**，其中 serving-b8 有 −32.5% 的真实增益。

### 5.2 smoke / skip 与精度

- smoke：`decode-s1-float16`（parametrization idx 0 的 `pytest.mark.smoke`）——只做精度回归，
  最终 `precision_pass: true`（§5.3 的 `L0-decode-s1-tap0` 用例），**无性能记录**（inventory 记 `reason`）。
  非门禁参考：基线 5.529 µs / final 5.631 µs（同码，噪声）。
- skip：无。
- 最终精度（`perf_opt/logs/final/precision_level_all.log`，`EngramDecodeFwdOp.py --level all`）：
  **L0 6/6、L1 7/7、boundary 3/3 全过**（operator 级：两个 launch 串起来对比仓库 torch 参考
  `engram_decode_step_torch`，容差取 `tests/ops/test_engram.py::_tol_decode`）；
  max_y_diff 1.53e-05、max_state_diff 3.91e-03；每例另验 2× 重复 launch 逐位一致、
  history 左移与参考逐位相同、`d_pad>d` 的填充列恰为 0。
- 生产链路（wrapper 仍指 Stage-3 baseline，perf_opt 由 conductor 翻转）：
  `pytest tests/ops/test_engram.py -q -k decode` → **16 passed, 30 deselected**（18.85 s；
  `logs/final/production_pytest_decode.log`）。该路径当前走 wrapper 的 Stage-3 import，
  所以它验证的是「本任务没有破坏仓内契约」；perf_opt 自身的精度由上面
  `EngramDecodeFwdOp.py --level all` 的 operator 级用例验证。
- 交付命名：`perf_opt/EngramDecodeFwdOp.py`（规范名）与 `perf_opt/engram_decode.py`（同字节副本，
  sha256 都是 `0368657c5a99903f5041a558d117ff94f6ea287a009a8c4574232003d1f7551e`，
  后者对齐任务状态里的 `kernel_opt_py_path`）。

### 5.3 结论与中止原因

- **serving-b8 −32.5%**（9.466 → 6.385 µs），四个 tune 案例无回退（其中三个与基线同码）。
- **serving-b4（主导案例）没有有效增益，且已到达本设计的实测地板**：8.3 µs = 两次 launch 地板
  5.7（69%）+ W 流 1.31（MTE2 满速 98 GB/s/core、2 MB 不可压缩）+ mix 提升 0.49 + shift 字节 0.19
  + step 工作 0.7。要达 −15% 需 T_op ≤ 7.05 µs，即非地板项 ≤ 1.35 µs；而 W 流一项就 1.31 µs。
  结构上唯一能省的是「少一次 launch」（省 ≈2.7 µs），但两条备选都被已冻结的设计与 buffer 契约否掉：
  ① 单 launch 每核一行 → 权重被读 `batch` 遍；② split-K 增加块数 → 需要跨块归约，而跨 launch 的
  `kv` 由 wrapper 固定为 `(2, batch, d_pad)` fp32（恰好一次投影的量），没有空间放部分和，
  且 `ascend.batch_schedule` 跨块屏障未认证。
- **中止原因：plateau/success 混合判定**——Round 1 拿到唯一的结构性收益（b8），Round 2 的 step 分支与
  配置扫描全部落在噪声内（连续 2 轮无 >3% 提升，且剩余候选项经探针逐项归因到地板/带宽项，
  估算余量 <2%）。按 skill 的 **success**（当前 workload 无逐项数值目标时以实测最优收束）＋
  **plateau**（b4）收束；未触发 budget（max_rounds=10 / max_experiments=30 未用尽：轮次 2、
  实验分支 6）。
- **未触发 `[DESIGN_LIMIT]`**：地板归因虽在结构层，但 `perf-feedback.md` §1 的量化门槛
  （结构性加速估计 >2×，或实测直接推翻设计判定）**不成立**——本任务实测的 launch 常数（2.7 vs 设计
  假设 5 µs）只是把一个「未实测假设」锚定为更小值，DESIGN 的两条结构否决重算后依旧成立（§0 首轮必查项 3）。
  该常数已按 ED-C 回写知识库（§7），供后续任务的设计期 roofline 直接引用。

---

## 6. Skill Retrospective

### 6.1 流程问题与建议（proposal，不自动改 skill）

| # | 现象 | 建议 | vp_type |
|---|---|---|---|
| 1 | 本算子 T_op 是**两次 launch 之和**，而 skill/gate 的 `perf_records` 契约按「单 kernel」组织；本次只能把两次 launch 记成一个 kernel_id（`A+B`）。多 launch 算子（解码类、MoE 类）普遍存在 | 在 `signal-registry.md` §5 明确多 launch 算子的 `kernel_id` 写法与 `launch_count`/`t_op_from` 字段，并把「T_op = 所有 launch 之和」写成门禁口径 | R |
| 2 | 平台的 `execution.precision_command` 指向的解释器无 torch；实际必须用 `$TILEOPS_950PR_VENV/bin/python` | 同 batch2 任务的结论：平台 execution 或 `env-950pr.sh` 应导出 `PYTHON` | R |
| 3 | 本量级（T_op 5–9 µs）**同码重测**即可漂移 project ±7% / step ±12%；单臂 3 次中位仍会给出 2–4% 的假差异 | `profile-collection` 增加一条：T_op < 10 µs 且 launch 数 ≥2 时，**必须**做同会话配对 A/B，并且优先报告 product/逐 launch 中位而非 T_op；另建议提供 p50-of-p50 的批量脚本 | R |
| 4 | mix kernel 的「有 AIV 工作即付 ≈0.49 µs」这一代价在任何文档里都没有；它使「把 copy 挪到 AIV」类优化在纯 Cube kernel 上天然吃亏 | 建议在 `iteration-diagnosis` 增加「mix 提升代价」判据与探针模板（本任务 `mkprobe.py` 的 tinyshift 形态） | P |

### 6.2 价值点（value point proposals）

- **VP-D1（D，实测代价常数）**：950PR 单次 launch 的 kernel-only Task Duration 地板在 grid=4/8/16、
  行宽 512（fp16）上为 **2.69/2.96/2.89 µs**；→ 两次 launch 的算子结构性地板 5.4–5.9 µs。
  证据三件套：evidence = 本日志 §2.2；repro = `perf_opt/experiments/probe_floor.py`（+ 已回写
  `knowledge/repro/CONST-950-launch-floor-grid-sweep.py`）；stamp = `7a52f2f1…` / CANN 9.2.1。
- **VP-D2（D，实测代价常数）**：mix(AIC+AIV) 提升相对同几何纯 Cube kernel 的固定代价 ≈**0.49 µs**
  （b4 project：4.199 → 4.693，仅加 1 条 1 行 copy）；此后 AIV 侧的字节按 ≈**1.2 µs/MB**（聚合）计。
  证据：§2.3 三形态探针表 + `profiles/probe/probe_project_{noshift,tinyshift}_*`。
- **VP-P1（P，模式）**：**按 batch 串行化的搬运链应按「每块串行往返条数」重排**——把
  `for b in T.serial(batch)` × (gm2ub+wait+ub2gm) 改成「一个 batch 行一个块、一条整行窗口 copy」，
  serving-b8 的 project 段 6.672 → 3.31 µs（−50%，T_op −32.5%）。反向边界：当 `batch ≤ n_tiles` 时
  重排无收益（b4 已由 8 个块天然并行），且**按列再切分**也无收益（r1_c3 实测），即成本是往返条数而非字节。
  证据：§3 Round 1 表 + `verify_codegen.py`（同码/改码判定）＋ `.asc` 逐行对照。
- **VP-P2（P，反向模式）**：step 段的「1 条整窗 tap copy 取代 `w−1` 条单行」在本算子**无收益**
  （b4 3.409 vs 3.418、b8 2.882 vs 3.007，均 <5% 噪声内），因为该 launch 已被 2.7 µs 地板 + 4 条
  `AscendAllReduce` 屏障链支配；`threads ∈ {128,256,512}` 同样无差异。结论绑定 `7a52f2f1…`。
- **VP-C1（C，案例索引）**：本任务可作为「**多 launch 小算子**（T_op < 10 µs、launch 地板占 ≥65%）」
  的优化范式：先量每次 launch 的地板 → 再逐 launch 用「去项探针」把剩余时间分到带宽/往返/提升代价 →
  只有当余量 > 地板 2 倍时才考虑结构改写；否则诚实地以「不回退 + 局部结构收益」收束。
- **VP-R1（R，流程）**：见 §6.1 #1/#3。

---

## 7. 回写 pattern-library（ED-C）

- `knowledge/constants.md`：新增 **`CONST-950-launch-floor-grid-sweep`**（grid 4/8/16 的 launch 地板
  2.69/2.96/2.89 µs + 两次 launch 地板 5.4–5.9 µs + mix 提升 0.49 µs，附 repro 与工具链戳）。
- `knowledge/patterns-950pr-verified-forms.md`：新增 **`PL-950-serial-copy-chain-hoist`**
  （按 batch 串行化的 GM↔UB 搬运链 → 每 batch 行一个块的整行窗口 copy；b8 project −50%；
  反向边界：`batch ≤ n_tiles` 与按列切分均无收益）。
- repro：`knowledge/repro/CONST-950-launch-floor-grid-sweep.py`（自包含：同几何地板探针 + 断言
  地板区间），已实跑通过 → `grid=4/8/16 = 2.711/2.766/2.770 µs, REPRO_OK`；
  `PL-950-serial-copy-chain-hoist` 的 repro 复用同一文件（其慢→快关键改动以
  `perf_opt/experiments/mkvariant.py` 的 `SHIFT_BODY_OLD` → `SHIFT_BODY_NEW_C4` diff 表达，
  逐字节同码/改码判定用 `perf_opt/experiments/verify_codegen.py`）。
- INDEX 路由：`knowledge/INDEX.md` 新增「engram_decode 实测集群」小节，
  两个条目均有路由（`KB-INDEX-COVERED` 警告已清零）。
- **7.1 校验**：`kb_lint.py --toolchain ascend --hardware ascend-950pr-9579` → **0 failures**
  （31 warnings 均为既有条目的 repro 最小化预算提示，与本次回写无关）。

---

## 8. 机械门禁自检（`statectl gate 4`）

```
python3 .agents/tools/statectl.py gate 4 --dir examples/engram_decode_perf
-> {"pass": true, "failures": [], "warnings": [], "stage": 4}
```

已处理的门禁项（首轮自检发现并修正，供后续任务参考）：

| 规则 | 首轮失败信息 | 处置 |
|---|---|---|
| `S4-GOLDEN` | 最终文件缺 `golden*` 参考实现 | 把 step 模块的 `golden_engram_decode` + `_rmsnorm_torch` 并入最终文件（并按 L0 首个用例与仓库参考 `engram_decode_step_torch` 逐位对齐，防止两份语义漂移） |
| `S4-TEST-LEVELS` | 最终文件缺 `run_L2()` | 补 `run_L2()`（b64-d1024 / b3-d1024-bf16，非阻塞记录）并接进 `--level` |
| `S4-PERF-RECORDS-RECON` | final 未测最终 perf_opt 文件 | `artifact_path` 须**相对 `perf_opt/` 目录**（不是任务目录）解析；final 行指向状态声明的 `kernel_opt_py_path` = `perf_opt/engram_decode.py` |
| `S4-PERF-RECORDS-RECON` | Final 表缺 (`kernel_id`, `workload_id`) | Final Performance Test Data 表必须**恰好 5 列**（`kernel_id \| workload_id \| baseline_us \| final_us \| final_candidate_id`）且每行写**完整** kernel_id，不能用「同上」/省略号 |

最终工件两份同字节文件：`perf_opt/EngramDecodeFwdOp.py`（任务约定名）与
`perf_opt/engram_decode.py`（状态文件 `kernel_opt_py_path`），
sha256 `6c647f7d…`（见 `perf_records.jsonl` final 行）；两个 launch 的 `device_kernel.asc`
与本日志 §3.1 的测量版本逐字节相同。

---

## 9. 生产采纳（wrapper 切换 + 单算子报告验收）

### 9.1 采纳文件清单

| 文件 | 作用 | sha256 |
|---|---|---|
| `examples/TileOPs/tileops/kernels/sequence_modeling/engram_decode/engram_decode_kernel/history_version/engram_decode_pre_opt.py` | **wrapper 备份**（翻转前原样，可整体回滚） | `a2c67d5be74719a5da2fc24a04a260e85a4976fc2852bd26ab3a8194469828a7` |
| `…/engram_decode_kernel/perf_opt/__init__.py` | perf_opt 子包导出（`ALIGNMENT` / 两个工厂 / 两个 `*_default_config`） | — |
| `…/engram_decode_kernel/perf_opt/EngramDecodeFwdOp.py` | **drop-in 优化 kernel**（= `engram_decode_perf/perf_opt/EngramDecodeFwdOp.py` 同字节，346 行 kernel + 自带 golden/L0-L2/boundary harness） | `6c647f7d111ee97b3e6c09e8f14b28b17713ee982923a40977093a5faeb73957` |
| `…/engram_decode/engram_decode.py` | wrapper：**仅**翻转 import 切换块（op 层构造/forward 契约、`custom_op` 签名与 fake 输出不变） | 见 git diff |

- 切换块（唯一改动）：baseline 三行 + 两行 config import 置为注释，
  perf_opt 分支改为 `from .engram_decode_kernel.perf_opt import (ALIGNMENT,
  _engram_project_kernel, _engram_step_kernel)` + 成对的
  `project_default_config` / `step_default_config`。
- **回滚**：把切换块两行注释换回（baseline 分支仍从父包 `engram_decode_kernel` 导入，
  Stage-3 源码未动），或用 `history_version/engram_decode_pre_opt.py` 覆盖 wrapper。
- 若 tuned kernel 与基线默认参数不同，须在 perf_opt 里以模块级常量暴露供切换块成对引用——
  **本任务的 tiling 与基线完全相同**（`block_m=16, block_n=min(64,d_pad), block_k=256,
  num_stages=2`；step `threads=min(512,next_pow2(d_pad))`），因此切换块无需新常量。

### 9.2 经 wrapper 后确实走优化实现

```
wrapper project factory module: tileops.kernels.sequence_modeling.engram_decode.engram_decode_kernel.perf_opt.EngramDecodeFwdOp
wrapper step    factory module: tileops.kernels.sequence_modeling.engram_decode.engram_decode_kernel.perf_opt.EngramDecodeFwdOp
wrapper project config  module: tileops.kernels.sequence_modeling.engram_decode.engram_decode_kernel.perf_opt.EngramDecodeFwdOp
wrapper step    config  module: tileops.kernels.sequence_modeling.engram_decode.engram_decode_kernel.perf_opt.EngramDecodeFwdOp
perf_opt ACTIVE: True
baseline still importable: …engram_decode_kernel._engram_project_kernel | …engram_decode_kernel._engram_step_kernel
```

（`logs/production/perf_opt_active.log`；同时以 `EngramDecodeFwdOp(20,4,5,1e-6)` 真实调用一次
op 层（serving-b4 几何），输出 `y_t (4,512) fp16` / `new_conv_state (4,20,512) fp16` 正常。）

### 9.3 生产回归

- `pytest tests/ops/test_engram.py -q -k decode` → **16 passed, 30 deselected**（9.88 s；
  `logs/production/pytest_decode.log`）。
- 部署后的那份文件 `…/perf_opt/EngramDecodeFwdOp.py --level all` → **L0 6/6、L1 7/7、L2 2/2、
  boundary 3/3**（18/18），`max_y_diff=6.104e-05`、`max_state_diff=3.906e-03`，与 Stage-4 阶段的
  工件逐字段相同（`logs/production/deployed_level_all.log`）→ 逐位结果不变。
- 逐字节不变性复核（`experiments/verify_codegen.py`，本任务 §3.1）：serving-b4 / serving-b1 /
  decode-partial-cache / decode-s1 的 `device_kernel.asc` 与 Stage-3 基线**逐字节相同**；
  只有 serving-b8（`batch > n_tiles`）走整行窗口 shift。

### 9.4 单算子报告（msprof）

```
TILEOPS_TASK_DIR=/workspace/tilelang-mlir-ascend-950/examples/engram_decode TMPDIR=/workspace \
  python -m tileops.reporting.cli run --op EngramDecodeFwdOp --prof-mode msprof \
  --pytest-arg=-k --pytest-arg=decode
```

- **run-id：`20261125_005026_893589_EngramDecodeFwdOp`**
  （`examples/TileOPs/reports/tileops/20261125_005026_893589_EngramDecodeFwdOp/`）
- `status=passed`；correctness **16/16**；benchmark 4/4；
  `invalid_profile_count=0`、`invalid_acceptance_count=0`、`profiler_fallback_count=0`、
  `kernel_filter_fallback_count=0`、`warnings=[]`。

| case | dtype | 本次报告 T_op（两次 engram launch 之和） | 同期基线报告（同链路、同聚合） | Stage-4 final（3 次中位） | `mfu_baseline_v1` |
|---|---|---:|---:|---:|---:|
| serving-b8 | bf16 | **6.094** µs | 9.373 / 9.040 | 6.385 | 9.544 |
| serving-b4 | fp16 | 8.165 | 7.854 / 8.221 | 8.515 | 8.727 |
| serving-b1 | fp16 | 5.741 | 5.875 / 5.743 | 6.094 | 6.305 |
| decode-partial-cache | fp16 | 6.004（报告原始 8.891，含 host `PadV3`） | 8.723 / 8.868（同样含 `PadV3`） | 6.258 | 5.854（`forced-unfiltered`，不含 `PadV3`） |

- **serving-b8 = 6.094 µs**（期望 ≈6.4，基线 9.5）→ −35%；其余三个 case 与同期基线报告持平
  （±4% 会话漂移内），b4 8.165 落在 7.85–8.22 的基线区间内（代码逐字节未变，见 9.3）。
- `decode-partial-cache` 的报告 `t_op_us`（8.891）比 §final（6.258）高 2.887 µs，原因是该 case
  在 `Kernel.forward` 里做 host 侧 `F.pad`（`conv_len=5 < max_conv_len=12`），无 kernel 名过滤的
  采集把它一起算了进去（`captured_op_names` 里可见 `PadV3_…`）。**基线报告 20261124_090540 /
  _092758 同样捕获 3 个 kernel、同样 8.868 / 8.723**，即与本次是同口径对比，不是回退；
  `perf_records.jsonl` 的 merged 行按「两次 engram launch 之和」记录（6.004）并保留
  `report_t_op_us` 原值。
- MFU 验收 `passed=0 / failed=4` 与优化前一致（`mfu_baseline_v1.json` 记录本算子 0 个 case 达标），
  MFU best 由 2.057% 升到 **2.199%**（serving-b4）；`invalid=0`、`blocked=0`，
  acceptance 状态均为 `measured` —— 该算子是访存受限，全局 90% MFU 目标下的既有判定，不是回退。

### 9.5 兼容性与回退

- 无回退：serving-b8 −35%，其余三条持平；精度逐位不变（9.3）。
- 契约无变化：`EngramDecodeFwdOp.__init__(max_conv_len, conv_kernel_size, dilation, eps, kernel_map)`
  与 8 入 2 出 `forward` 未改；`custom_op("npub::engram_decode")` 参数表与 fake 输出形状未改；
  `EngramDecodeKernel.default_config` 返回值与基线逐字段相同；`ascend_mode=None` / `supported_archs=None` 未改。
- perf_opt kernel 与基线工厂**同签名**、同一 `lru_cache` 形状特化策略，wrapper 无需新默认参数。
- `/tmp/tileops_msprof_*` 已清理。
