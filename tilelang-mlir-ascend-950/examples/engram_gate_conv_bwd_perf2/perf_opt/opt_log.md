# opt_log.md — `EngramGateConvBwdOp` Stage 4 追加轮（perf2：在已采纳的 Stage-4 结果上再压一轮）

- 任务：`examples/engram_gate_conv_bwd_perf2`（scenario=optimize, mode=full, 平台 `ascend-950pr-9579` / `ascend`，单卡）
- **基线（只读）**：`examples/engram_gate_conv_bwd_perf/perf_opt/engram_gate_conv_bwd.py`
  （= 上一轮已生产采纳的 Stage-4 结果，sha256 `65c3133c38ec86ba26db72e90d620c2c4cac8c079b04e1a211fde0b11668ec81`；
  生产报告 run-id `20261124_231932_732777`，train-s96 T_op 12.393 µs）
- 设计：`examples/engram_gate_conv_bwd/_engram_gate_conv_bwd_kernel/DESIGN.md`
- 产物：`perf_opt/EngramGateConvBwdOp.py`（= `perf_opt/engram_gate_conv_bwd.py`，字节相同，
  sha256 `27b12cf6a3c4e4d3…`；内容 = `perf_opt/engram_gate_conv_bwd_round1_n3.py`）
- **测量口径**：任务平台 `measurement` 规定 `msopprof` kernel-only `Task Duration(us)`，一次采集一次算子调用；
  本算子一次调用 **2 个 launch**（同名 `main_kernel`），平台 `T_op` = 两 launch 之和 → `--launch-count=2` 求和；
  驱动 `perf_opt/experiments/bench_one.py`。每个数值取 **3 次独立采集进程的中位数**（本算子噪声 ±1.5%，
  样本见 `perf_records.jsonl` 的 `samples_us`）。主指标 `msprof_task_duration`。

---

## 1. 本轮目标与基线

| 项 | 值 |
|---|---|
| 目标（用户） | train-s96 **T_op ≤ 11.2 µs**（相对生产报告 12.4 至少 -9.7%，相对本任务 baseline -13.8%），其他 case 不回退 |
| baseline（本任务 round 0，= 上一轮最终版） | train-s96 **12.994**（launch1 10.04 + launch2 2.95）、train-s64 **8.929**、train-s128 **7.754** |
| final（本轮） | train-s96 **12.972**、train-s64 **9.028**、train-s128 **7.218** |
| 结论 | **未达 11.2 目标；本轮净收益在噪声内（-0.2%）。按 skill 判定 plateau，并给出下列量化地板证据。** |

同一份最终字节在两批独立采集里的样本合并（6 个）：12.393 / 12.530 / 12.813 / 12.972 / 13.074 / 13.200
→ 中位 **12.89 µs**；baseline 3 样本 12.876 / 12.994 / 13.110 → **12.99 µs**。即本轮的真实增益
（唯一被采纳的改动 O-15）在 0.1–0.2 µs 量级，被 ±1.5–3% 的 run-to-run 抖动掩盖。

---

## 2. 本轮前的瓶颈画像（baseline profile，train-s96）

| 指标 | launch #1 | launch #2 |
|---|---:|---:|
| kernel 时长 | 10.04 | 2.95 |
| 最大 block `aiv_time` | 9.469（中位 8.523） | 1.642（中位 1.368） |
| **launch 固定开销（kernel − max block）** | **≈0.57** | **≈1.43** |
| `aiv_vec_time` | 6.173 | 0.150（48 行串行 vadd 链） |
| `aiv_scalar_time` | 1.630 | 0.340 |
| `aiv_mte2_time` / `aiv_mte3_time` | 1.293 / 0.572 | 0.669 / 0.115 |
| `aiv_vec_wait_ratio` | 0.122 | 0.491 |
| 读 / 写 GM | 74 KB / 45 KB | 24 KB / 0.5 KB |

生成码实查（`device_kernel.asc`）确认：预装载已按使用点分成 4 个 event 组
（`asc_sync_notify/wait(PIPE_MTE2,PIPE_V,{3,2,1,0})`），i=0 处的等待只覆盖 vhat/dY 组（≈36 KB），
H/k/V 组（27 KB）等到 i=6 才等 —— **DMA 与计算的重叠已经做到位**，`mte2` 1.29 µs 中只有
起始 ≈0.5 µs 是暴露的（对应 `vec_wait` 的一部分）。

---

## 3. 迭代记录（本轮 1 轮 / 4 个实验分支，全部单点）

**候选 vs current best（current best = baseline-perf1-final = 12.994 µs）**

| 分支 | 优化点（单一） | workload | Task Duration(us) | Δ vs best | aiv_vec_time（launch1） | 结论 | L0 |
|---|---|---|---:|---:|---:|---|---|
| r1_n1 | O-13 把 12 个 chunk 循环改为 `T.unroll`（build-time 展开，消掉逐迭代地址/循环开销） | s96 | **31.284** | **+140.8%** | –（launch1 10.04 → 28.2） | **否决**：指令 cache 打爆（.asc 语句数 101 → ~1000），说明本 kernel 的代码尺寸已贴近 icache 上限 | PASS |
| r1_n2 | O-13' 三段化 + 卷积/`dconv_w` 融成 chunk-major sweep（12 个寄存器常驻：4 cw + 4 累加 + 4 tap） | s96 | **21.331** | **+64.2%** | **6.173 → 14.831（+140%）** | **否决**：寄存器溢出（`scalar` 反而降到 1.50、`mte2` 降到 1.09，唯独 vec 翻倍 = spill 往返） | PASS |
| r1_n3 | **O-15** launch #2：行循环提到列 chunk 循环外，nch 个累加器并行（ILP = 列 chunk 数） | s96 | **12.813** | **-1.4%** | launch2 链 0.150→0.085、block 1.368→1.258 | **采纳**（逐位不变） | PASS |
| r1_n4 | O-14 融合指令形态：`vexpdif(zero,conv)` 替代 `vneg+vexp`、卷积链从 tap0 起（省 `vdup(0)`+首个 `vadd`，共 -324 条向量指令） | s96 | 13.132 | +1.1% | 6.17（不变） | **否决**：指令数 -3% 无任何时间变化 → 本 kernel 不是 issue-slot 受限 | PASS |

- **合并非回退检查（r1_n3，全部 tune workload + smoke）**

| workload | baseline | n3/final | Δ |
|---|---:|---:|---:|
| s96 | 12.994 | 12.972 | -0.2%（launch2 -0.13 µs，被噪声掩盖） |
| s64 | 8.929 | 9.028 | +1.1%（该 shape `nch_r==1`，reduce 代码与基线逐字节相同 → 纯噪声） |
| s128 | 7.754 | 7.218 | -6.9%（同上，`nch_r==1` → 纯噪声） |
| s32（smoke，仅精度） | – | PASS | – |

- 说明：本轮的 3 个否决分支都是**结构性**尝试（循环展开 / 阶段重排+寄存器常驻 / 指令形态），
  它们把三条不同的猜测都证伪了，因此不是「同一配置重复试参」。

### 3.1 本轮新增的证伪结论（都已实测，非推理）

| 猜测 | 实测 | 结论 |
|---|---|---|
| 标量开销来自逐 chunk 迭代的地址/循环 → 展开可省 | unroll 全部 chunk 循环：launch1 10.04 → 28.2 µs | **错**：icache 是硬约束（代码 ≈2.8× 后彻底失效） |
| 把 nrm/ds 常驻 + 卷积与 dconv 融合可省重复装载 | vec 6.17 → 14.83 µs | **错**：12 个常驻寄存器即溢出 |
| 指令数 -3% 能带来时间收益 | -324 条指令 → 0 变化 | **错**：不是 issue-slot 受限（上一轮 r2_c3 的同一现象在本轮复现） |
| `vaxpy`（`dst = src*scalar + dst`）可用于 `ph` 部分和 | 设备编译失败（`Ascend device compilation failed`） | 形态不可用（已记录，见 §6 回写候选） |
| 两次 launch 可否合并（用户提示） | 实查 `ascend_cross_core_set/wait_flag`、`ascend_sync_inter_*` 均生成 `AscendC::CrossCoreSetFlag<0,pipe>`，`schedule_hint.py` 的 `PerCoreTask` 示例用 `split_id` 在**同一 AI core 的 AIV 之间**同步；GM 原子加会放弃六输出的逐位确定性（`test_engram_gate_conv_bwd` 对 dH/dk/dv 逐位比对） | **不可合并**（block↔block barrier 在本工具链无认证路径） |

---

## 4. 为什么 11.2 µs 在本平台不可达（量化地板）

train-s96 的 `T_op = 12.97` 拆解（本轮实测）：

| 组成 | µs | 占 T_op | 依据 / 可压缩性 |
|---|---:|---:|---|
| launch #1 固定开销 | 0.57 | 4.4% | kernel 10.04 − 最大 block 9.47 |
| launch #2 固定开销 | **1.43** | 11.0% | kernel 2.95 − 最大 block 1.64；同族 forward 的地板探针亦为 ≈1.3（`CONST-950-kernel-launch-floor-48blk` 的总量 2.72 减去 block 工作） |
| launch #1 向量管道 | 6.17 | 47.6% | issue/latency 混合；**-324 条指令无变化**（n4）、**+12 寄存器 → +140%**（n2）、**代码 ×2.8 → +140%**（n1） |
| launch #1 标量 | 1.63 | 12.6% | 552 个 chunk 迭代 × ≈5 cycle；减少迭代数需更大重构（n1/n2 均失败） |
| launch #1 DMA（已重叠） | 1.29（暴露 ≈0.5） | – | 已按使用点分 4 个 event 组，H/k/V 组等到 i=6 |
| launch #2 工作（884 KB 归约） | 1.37 → 1.26 | 10.5% | 24 KB/块拷贝（36 GB/s，48 行 × 512 B 步长 3072 B）+ 启动；列并行已取 `6·tiles ≤ 54` 的上限 |
| 合计 | **12.97** | | 目标 11.2 需要在上面 12.97 里再砍 1.77 µs = **两次 launch 固定量（2.0 µs）的 88%** |

即：**目标值落在「两次 launch 的固定开销 + 现有 kernel body」之下**。要达成只有两条路，都已被平台/契约封死：
① 合并 launch（跨 block barrier 未认证，GM 原子加破坏逐位确定性）；② 把 launch#1 body 从 9.9 砍到 8.1
（-18%），而本轮三条结构性路径实测全部反向（+140% / +64% / 0）。

**结论：本轮判 plateau（`stop_reason = plateau`），不出 `[DESIGN_LIMIT]`**：`perf-feedback.md` §1 的两项
触发条件均不满足——① 这不是 DESIGN 的算法/结构假设被推翻（DESIGN R1 的「2 次 launch + 有序归约」
被实测**支持**为当前工具链下唯一可行路径），② 结构性加速估计远小于 2×（去掉一个 launch 也只有 10–12%）。

---

## Final Performance Test Data

| kernel_id | workload_id | baseline_us | final_us | candidate_id |
|---|---|---:|---:|---|
| tileops/kernels/sequence_modeling/engram_gate_conv_bwd/engram_gate_conv_bwd_kernel::_engram_gate_conv_bwd_kernel::main | train-s96-bfloat16 | 12.994 | 12.972 | final-reduce-ilp |
| tileops/kernels/sequence_modeling/engram_gate_conv_bwd/engram_gate_conv_bwd_kernel::_engram_gate_conv_bwd_kernel::main | train-s64-float16 | 8.929 | 9.028 | final-reduce-ilp |
| tileops/kernels/sequence_modeling/engram_gate_conv_bwd/engram_gate_conv_bwd_kernel::_engram_gate_conv_bwd_kernel::main | train-s128-bfloat16 | 7.754 | 7.218 | final-reduce-ilp |

（全部为 3 次独立采集进程的中位数；`perf_records.jsonl` 的 final 三行 `artifact_path` = `EngramGateConvBwdOp.py`，
`artifact_sha256` 与文件一致 = `27b12cf6a3c4e4d3…`。）

### 5.1 精度（六输出逐位不变）

`perf_opt/EngramGateConvBwdOp.py --level all`（`logs/final/L0_L1_L2_boundary.log`）：
**L0 10/10、L1 6/6、L2 4/4、boundary 5/5 → `[PRECISION_PASS]`**，
max|diff| 与 baseline **逐字段相同**（dH 1.221e-04、dk 9.766e-04、dv 1.953e-03、
drms_w_h 3.598e-04、drms_w_v 4.453e-04、dconv_w 1.194e-02）。
生产包回归：`pytest tests/ops/test_engram.py -q -k gate_conv_bwd` → **17 passed, 29 deselected**
（该路径当前仍导入上一轮已采纳的 perf_opt 文件；本轮新文件尚未翻转，见 §5.3）。

### 5.2 smoke / skip

- smoke `train-s32-float16`：只做精度回归 → PASS（L0-1）。skip：无。

### 5.3 采纳方式（给 conductor）

本轮交付的是**新的 `perf_opt/EngramGateConvBwdOp.py`**（= 生产包中 perf_opt 文件的下一版）。
采纳 = 覆盖
`examples/TileOPs/tileops/kernels/sequence_modeling/engram_gate_conv_bwd/engram_gate_conv_bwd_kernel/perf_opt/_engram_gate_conv_bwd_kernel.py`
（工厂签名、`out_idx`、`lru_cache` 策略完全一致，wrapper 切换块不需要改；无新默认参数）。
由于本轮收益在噪声内，**建议**：如果以「可测收益」为准，保持生产包现状（上一轮那版）也可以——
两版对 s64/s128 的生成码逐字节相同，只有 s96 的 reduce kernel 换成了 2 条并行链（严格更少工作）。

---

## 6. Skill Retrospective

| # | 现象 | 建议 | vp_type |
|---|---|---|---|
| 1 | 「小算子改进 < 噪声」时，单轮 3 样本中位数无法区分 0.1–0.2 µs 的**确定性更少工作**（本轮 O-15 的真实增益被 ±1.5–3% 抖动吞掉） | profile-collection 增补：当候选改动可证明「指令/搬运严格更少且逐位不变」时，允许以**管道指标**（如 `aiv_vec_time`、block `aiv_time`）作为次级证据记录采纳，不要求 T_op 中位数单独显著 | R |
| 2 | 本 kernel 同时撞上 **icache 上限**（unroll +140%）与 **寄存器上限**（+12 regs +140%）两条隐形边界，而 `aiv_icache_miss_rate` 恒为 0（该列不反映真实失效） | iteration-diagnosis 增补判据：SIMD 向量 kernel 做「代码或寄存器增长」的实验前，先记录 `.asc` 语句数与常驻寄存器数；`.asc` 语句数翻倍即视为高风险（本任务：101 → ≈1000 时崩） | R |
| 3 | `aiv_icache_miss_rate` 不可信（崩到 28 µs 时仍报 0），无法作为 icache 判据 | 在 knowledge traps 里登记「不要用该列判断 icache；用 `.asc` 尺寸 + A/B 实测」 | D |
| 4 | `vaxpy` 的 API 形态（dst 必须是 `BufferLoad`）通过前端校验，但在本 kernel 的设备编译阶段失败 | traps 条目登记 `vaxpy` 在 `target=ascend` 本工具链上不可用的最小复现（`ph += (uh*hf)*rh` 形态） | D |
| 5 | 「跨 block barrier 不可用」这一结论本轮用三条独立证据（`sync.py` 生成 `CrossCoreSetFlag<0,pipe>`、`schedule_hint.py` 的 `split_id` 语义、平台 ruleset 条款）再次确认，但**没有一条**是「已尝试并失败」的实测 | 建议在 knowledge 里把该结论的边界写清：可用的 `CrossCoreSetFlag` 只覆盖同一 AI core 的 AIV/AIC；跨 grid 的 barrier 在本工具链**不存在任何 API**（而非「未认证」），以便后续任务不再重复调研 | P |

### 6.1 价值点（value point proposals）

- **VP-D2（D，实测代价常数）**：950PR 上**短 kernel 的 launch 固定开销 ≈1.3–1.6 µs**（本轮两次独立观测：
  launch#2 kernel 2.95 − block 1.64 = 1.43；launch#1 10.04 − 9.47 = 0.57，差异来自 block 数与 DMA 起始重叠）。
  两次 launch 的算子因此固定多付 ≈2.0 µs。证据：§2/§4 表 + `profiles/experiments/*/main_kernel/{0,1}`。
- **VP-P6（P，正向模式）**：多 launch 归约 kernel 的**行归约链必须与列 chunk 并行**：把行循环提到
  列 chunk 循环外（每 chunk 一个累加器）使 48 行串行链变成 nch 条并行链，launch#2 的
  `aiv_vec_time` 0.150 → 0.085 µs、block 1.368 → 1.258 µs，T_op −0.13 µs，且**求和顺序不变（逐位相同）**。
- **VP-P7（P，反向模式）**：SIMD 向量 kernel 上的两类「局部优化」在本平台是**净负**且程度惊人：
  ① build-time 展开内层 chunk 循环（101 → ≈1000 条 .asc 语句）→ **+140%**；② 把 4 个 tap + 4 个累加器
  常驻寄存器（12 个）→ `aiv_vec_time` **+140%**。判据：任何增加代码尺寸或常驻寄存器数的重构，先按
  §6.2 的 vp_type R#2 预检。
- **VP-C3（C，案例索引）**：本任务可作为「已到地板的向量算子如何量化收尾」的范式：把 T_op 拆成
  「2 × launch 固定 + launch1 body（vec/scalar/DMA）+ launch2 body」，再逐项给出可压缩性，最后用
  「目标值 < 地板」的算式证明 plateau，而不是止步于「试了几个点都不涨」。

---

## 7. 交付核对

- 未改动基线（`examples/engram_gate_conv_bwd_perf/perf_opt/*` 只读）、生产包 kernel/op/manifest/test/bench；
  未触碰 `/workspace/TileOPs-ascend950`。改动只写本任务 `perf_opt/`。
- `perf_opt/` 顶层：`EngramGateConvBwdOp.py`、`engram_gate_conv_bwd.py`（同字节）、
  `engram_gate_conv_bwd_round{0,1_n1,1_n2,1_n3,1_n4}.py`、`opt_log.md`、`perf_records.jsonl`、
  `workload_inventory.json`、`experiments/`、`logs/`、`profiles/`。
- 最终文件以模块方式被导入验证：`experiments/bench_one.py` 用 importlib 加载并调用
  `_engram_gate_conv_bwd_kernel(M, seq_len, d, dtype)`（与 wrapper 同签名）。
- `/tmp` 临时采集目录由 `bench_one.py` 自行清理；`TILELANG_CACHE_DIR=/tmp/opencode/tl-cache-opt-engramb2`。
