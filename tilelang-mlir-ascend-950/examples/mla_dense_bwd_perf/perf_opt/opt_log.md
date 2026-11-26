# MLADenseBwdOp — Stage 4 调优日志（Cube 化后定向优化）

- 算子：`MLADenseBwdOp`（dense BSHD MLA prefill backward，一次调用 3 个 kernel + 2 个框架算子）
- 基线：`examples/TileOPs/tileops/kernels/attention/mla_dense_bwd/mla_dense_bwd_kernel/_bwd_kernel.py`
  （本目录副本 `perf_opt/MLADenseBwdOp_baseline_stage3.py`，md5 `ad448672` / sha256 `5968790c…`）
- 设计：`examples/mla_dense_bwd/_bwd_kernel/DESIGN.md`（v3 Cube）
- 平台：`ascend-950pr-9579` / `ascend`，单卡（cube 28 / vector 56，L2 128 MB）；
  工具链 tilelang 0.1.15 / CANN 9.2.1（stamp `7a52f2f1…`）
- 测量口径：**msopprof Task Duration(us)**；`T_op` = 一次调用内全部 launch 求和
  （`Range ×2` + `ZerosLike` + `_delta_main` + `_bwd_main` + `_dq_land_main`），与生产报告
  `t_op_us` 同口径。采集器 `perf_opt/collect.py`（一次 capture 窗口正好覆盖一次调用，
  `window_ok=true` 逐次校验）；原始采集在 `perf_opt/profiles/`，stdout 在 `perf_opt/logs/`。
- 执行环境：cwd `examples/TileOPs`，`source perf_opt/env.sh`（内含 `TMPDIR=/workspace`、
  `TILELANG_CACHE_DIR=/tmp/opencode/tl-cache-opt-mlad`、平台 python venv）。

## 0. 环境与口径自校验

`perf_opt/collect.py` 对本任务基线的实测与生产报告 `20261125_044632_077346` 逐项一致：

| workload | 本日志 baseline `T_op` | 生产报告 `t_op_us` | 偏差 |
|---|---:|---:|---:|
| ds-v3-b2-8k-bfloat16 | 154 215.6 µs | 154 419.2 µs | −0.13% |
| ds-v3-4k-bfloat16 | 20 572.8 µs | 20 571.7 µs | +0.01% |
| ds-v3-4k-float16 | 20 570.8 µs | 20 572.4 µs | −0.01% |
| kimi-k2-8k-bfloat16 | 38 415.6 µs | 38 421.1 µs | −0.01% |
| ds-v2-lite-3k-bfloat16 | 6 292.7 µs | 6 296.7 µs | −0.06% |

`per_call_launches = 6`（实测：两次 `torch.arange` 各产生一个 `Range` 样本，`n=2`），
`--warm-up=(1+5)×6 --launch-count=6` 抓**恰好一次调用**，每行记录均 `window_ok=true`。

## 1. Performance Test Data

target kernel：`_bwd_main_kernel`（占 `T_op` 97.8–98.1%）；`T_op` 另含 `_delta_main`、
`_dq_land_main`、`ZerosLike`（`dq_acc` fp32 清零，b2-8k 954.9 µs）、`Range`（两次 arange）。

| kernel_id | workload_id | candidate_id | target_kernel_name | task_duration_us | profile_status | raw_profile_dir |
|---|---|---|---|---:|---|---|
| `_bwd_kernel`(+delta+land) | ds-v3-b2-8k-bfloat16 | baseline-stage3 | `_bwd_main_kernel` | 154 215.6 | valid | `profiles/round0/ds-v3-b2-8k-bfloat16_baseline` |
| 同上 | ds-v3-4k-bfloat16 | baseline-stage3 | `_bwd_main_kernel` | 20 572.8 | valid | `profiles/round0/ds-v3-4k-bfloat16_baseline` |
| 同上 | ds-v3-4k-float16 | baseline-stage3 | `_bwd_main_kernel` | 20 570.8 | valid | `profiles/round0/ds-v3-4k-float16_baseline` |
| 同上 | kimi-k2-8k-bfloat16 | baseline-stage3 | `_bwd_main_kernel` | 38 415.6 | valid | `profiles/round0/kimi-k2-8k-bfloat16_baseline` |
| 同上 | ds-v2-lite-3k-bfloat16 | baseline-stage3 | `_bwd_main_kernel` | 6 292.7 | valid | `profiles/round0/ds-v2-lite-3k-bfloat16_baseline` |

smoke / skip 排除（`perf_opt/workload_inventory.json`）：

- `mla-dense-bwd-smoke-bfloat16` / `-float16`：benchmark 参数化 index 0 带 `pytest.mark.smoke`，
  只做精度回归（不采性能、不选 winner）。精度见 §2.5（L0-1 / L0-2 PASS）。
- 无 `skipped` 案例（benchmark 未 skip 任何案例）。

### 1.1 baseline 的逐 workload 分解（平台测量 + 设备 pipe）

`us/it` = `main_us × 28 / iterations`（iterations 由 DESIGN §1.6.0 R3 公式复算，b2-8k = 1 420 032）；
`cube%`/`vec%` 为 **sum 口径**（`Σ pipe_time / Σ core_time`，即 `aic_*_share`，比逐块 ratio 均值可对账）。

| workload | T_op (µs) | main (µs) | iterations | us/it | cube%: mad/mte1/mte2/fix/scal | vec%: vec/mte2/mte3 | mte2 active BW |
|---|---:|---:|---:|---:|---|---|---:|
| ds-v3-b2-8k-bf16 | 154 215.6 | 150 897.5 | 1 420 032 | **2.975** | 25.8 / 22.8 / 24.2 / 19.2 / 2.8 | 45.3 / 4.1 / 3.9 | 117.5 GB/s |
| ds-v3-4k-bf16 | 20 572.8 | 19 751.5 | 180 224 | 3.069 | 25.4 / 22.7 / 21.7 / 17.8 / 2.8 | 45.9 / 4.2 / 4.0 | 124.5 GB/s |
| ds-v3-4k-fp16 | 20 570.8 | 19 751.8 | 180 224 | 3.069 | 25.4 / 22.7 / 21.7 / 17.8 / 2.8 | 45.9 / 4.2 / 3.9 | 124.7 GB/s |
| kimi-k2-8k-bf16 | 38 415.6 | 37 602.9 | 355 008 | 2.966 | 25.9 / 23.2 / 21.2 / 17.8 / 2.8 | 45.5 / 4.2 / 3.8 | 127.1 GB/s |
| ds-v2-lite-3k-bf16 | 6 292.7 | 6 014.0 | 51 200 | 3.289 | 23.9 / 19.4 / 26.4 / 17.8 / 2.5 | 43.7 / 3.9 / 3.8 | 98.5 GB/s |

b2-8k 的逐迭代管道忙时（`us/it × share`，与开发期 msprof 一致）：
mad 0.769 + mte1 0.680 + mte2 0.720 + fixpipe 0.570 + scalar 0.083 = **2.822 µs**，墙钟 2.975 ⇒
AIC 内五管道**近乎全部串行**（无重叠），与开发期定位（2.74 / 2.97）同结论。

### 1.2 设计期估算 vs 实测偏差（D-2 回填，DESIGN.md §1.6.4 F/G）

| 项 | 设计投影 | baseline 实测 | 偏差 |
|---|---:|---:|---:|
| ds-v3-b2-8k 主 pass | 40.5–50.6 ms（`0.80 + MTE1`，硬条件 MTE1 ≤ 0.20 µs）；管道级 `max` 下界 62.7 ms | **150.9 ms** | **3.0–3.7×**（对 62.7 ms 下界 2.41×） |
| MTE1 / 迭代 | ≤ 0.20 µs（硬条件） | **0.680 µs** | 3.4× |
| 单迭代墙钟 | 1.24–1.79 µs | 2.975 µs | 1.7–2.4× |

失准项：**设计把「跨迭代流水可得」当成默认可得**（§1.6.4 F/G 的 40.5–50.6 ms 以 `MTE1 ≤0.20 µs`
与管道级 `max` 重叠为前提）；实测该前提在本工具链上不成立——见 §2.3 的调度器证据与 §2.4 的
L0A 容量约束。归因与量化证据见 `perf_opt/perf_feedback.md`。

## 2. Iteration Log

### 2.0 调优排序与理由

按实测负担排序：`ds-v3-b2-8k-bf16`（154.2 ms，目标缺口最大、cube 主路径）→ `kimi-k2-8k-bf16`
（38.4 ms，H=64 的第二 regime）→ `ds-v3-4k-bf16` / `-fp16`（20.6 ms，同几何两 dtype）→
`ds-v2-lite-3k-bf16`（6.3 ms，H=16，单块迭代最少、per-iteration 固定开销占比最高）。
smoke 两个 case 不采性能。

### 2.1 Round 1 — 现象（基于 baseline 最新 profile）

1. **墙钟 ≈ 依赖链之和，不是管道之和**：b2-8k 的单迭代依赖链（由生成的 AscendC 代码逐段核对，
   见 `logs/probes/bwd_base.generated.cpp`）是
   `mad(st,dpt) 0.35 → fixpipe(L0C→UB) 0.28 → AIV 向量 epilogue 1.35 → nd2nz_scatter+ub2l1 0.10
   → mte1(L1→L0) 0.10 → mad(dv,dk,dq) 0.47 → fixpipe(dq L0C→GM 原子) 0.29 ≈ 2.94 µs`，
   实测 2.975 µs ⇒ **迭代内串行、跨迭代无旋转**。
2. **AIV epilogue 占墙钟 45%**（1.35 µs）：诊断探针（去掉两个 `exp`，数值无效仅测时口径）
   154 215.6 → 147 517.2 µs（−4.3%，即 0.13 µs/迭代）⇒ 代价主要不在 `exp`，而在
   `dual_copy(L0C→UB)` + 两次 `nd2nz_scatter`（UB→L1 的 NZ 转换）与 VF 调用/同步开销。
   `perf_opt/logs/round1/ds-v3-b2-8k-bfloat16_probe_noexp.log`。
3. **mad 在忙时已逼近峰值**：0.769 µs/迭代处理 10.24 MFLOP ⇒ 13.32 TF/s/核 = 峰值 13.5 TF/s（378/28）的 **98.7%**；
   即 5 个 GEMM 的计算量不可再压（算法等价化简已由 DESIGN §1.6.1 完成）。
4. **MTE1 是带宽受限而非可调参数**：217 088 B/迭代 @ ~300–340 GB/s = 0.60–0.72 µs，
   其中 61 440 B（28%）是 k/v 的每迭代重装；无法常驻 L0A（见 §2.4）。
5. **流水调度器在本 kernel 上不可用**：编译日志 `Z3 scheduling failed: solver returned unsat,
   falling back to topological sort`（`logs/probes/z3_dense.log`、`z3_varlen.log`——**同族 varlen 亦 unsat**），
   即跨迭代“软件流水”退化为拓扑序；`num_stages=3` 仍会让缓冲拿到 3 版本
   （`logs/probes/bwd_c1_stages3.generated.cpp` 有 12 处 `% 3` 索引），但**语句顺序不变**，
   因此只能消除“填充 WAR 等待”，无法旋转 epilogue。

→ 候选优化点（同一 base 派生，每支只改一个主要点）：

| 分支 | 优化点 | 预期 |
|---|---|---|
| c1 | `num_stages` 2→3（varlen 的 Stage 4 winner 直接移植） | 若填充 WAR 是瓶颈则应 −15% 级 |
| c3 | 去掉 qT/doT/kT 三条转置 L1 复制，dV/dK/dQ 用自然 (K,N) 操作数 | 每迭代 GM 填充 −50% |
| c4 | `annotate_buffer_versions` 给 q/do/qT/doT 两版本（显式预取） | 填充与计算重叠 |
| c5 | 拆 P/dS epilogue，dV 提前到 dP GEMM 之前 | dS 计算与 dP GEMM 重叠 |
| c8 | `num_stages=1`（关闭流水） | 验证 stages=2 是否真有效 |
| c9 | `annotate_buffer_versions` 给 AIV 产出的 pT_s/dsT_s 两版本 | 消除 AIV→AIC 的 WAR |
| c10 | 六个 per-iteration L1 tile 全部两版本 | 见 c9 的加强版 |
| c11/c12/c13/c14 | c9 + `num_stages` 3/4/5/6 | 深流水 × WAR 缓解 |
| c15 | c12 + 填充缓冲两版本 | 组合 |
| c16 | c12 + c3（自然操作数） | 组合（填充减半 + 深流水） |
| c6/c7 | `block_n` 80 / 128 原型 | tiling 灵敏度 / 预算边界 |

**候选 vs current best（baseline）对比表**（`ds-v3-b2-8k-bfloat16`，平台测量；pipe 为 sum 口径 `us/迭代`）：

| candidate | T_op (µs) | Δ | us/it | mad | mte1 | mte2 | fixpipe | AIV vec | mte2 BW | 精度 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **baseline（current best）** | **154 215.6** | — | 2.9754 | 0.769 | 0.680 | 0.720 | 0.570 | 1.349 | 117.5 | Stage 3 门禁 |
| c1 stages=3 | 154 767.3 | +0.36% | 2.9861 | 0.769 | 0.596 | 0.778 | 0.547 | 1.391 | 106.4 | parity PASS |
| c3 nt-operands | 164 806.4 | +6.87% | 3.1841 | 0.768 | 0.627 | 0.543 | 0.614 | 1.346 | 78.5 | parity PASS（dk/dv/dq 逐位） |
| c4 fill 2-ver | 154 214.1 | −0.00% | 2.9752 | 0.769 | 0.596 | 0.769 | 0.548 | 1.347 | 107.0 | parity PASS |
| c5 epilogue split | 186 734.7 | +21.1% | 3.6167 | 0.767 | 0.606 | 0.779 | 0.574 | **1.742** | 105.1 | parity PASS |
| c8 stages=1 | 157 012.9 | +1.81% | 3.0283 | 0.769 | 0.707 | 0.816 | 0.584 | 1.349 | 102.6 | 配置项 |
| c9 pT/dsT 2-ver | 153 289.6 | −0.60% | 2.9569 | 0.769 | 0.610 | 0.669 | 0.548 | 1.374 | 121.4 | parity PASS |
| c10 六缓冲 2-ver | 157 668.4 | +2.24% | 3.0412 | 0.769 | 0.729 | 0.777 | 0.544 | 1.346 | 105.5 | 配置项 |
| c11 c9+stages3 | 159 461.2 | +3.40% | 3.0749 | 0.769 | 0.692 | 0.761 | 0.604 | 1.349 | 106.7 | 配置项 |
| **c12 c9+stages4** | **152 743.1** | **−0.95%** | 2.9461 | 0.746 | 0.598 | 0.880 | 0.543 | 1.349 | 91.5 | parity PASS |
| c13 c9+stages5 | 155 179.0 | +0.62% | 2.9930 | 0.769 | 0.675 | 0.738 | 0.546 | 1.374 | 109.1 | 配置项 |
| c14 c9+stages6 | 167 307.8 | +8.49% | 3.2267 | 0.768 | 0.622 | 0.660 | 0.615 | 1.366 | 122.8 | 配置项 |
| c15 c12+fill 2-ver | 157 743.4 | +2.29% | 3.0425 | 0.769 | 0.653 | 0.773 | 0.596 | 1.346 | 107.2 | 配置项 |
| c16 c12+nt | 159 000.2 | +3.10% | 3.0665 | 0.769 | 0.636 | 0.538 | 0.595 | 1.346 | 78.5 | 配置项 |
| c6 block_n=80 | — | — | — | — | — | — | — | — | — | **编译失败**（见 §2.2） |
| c7 block_n=128 | — | — | — | — | — | — | — | — | — | **预算越界**（见 §2.2） |

**winner / rollback**：

- **c12（`annotate_buffer_versions({pT_s: 2, dsT_s: 2})` + `_STAGES = 4`）为 winner**：b2-8k −0.95%，
  其余四个 tune workload **全部不回退**（−1.29% / −1.28% / −1.58% / −3.94%，见 §2.5 合并非回退检查）。
- c9 的 −0.60% 经**交错 A/B/A/B 3 对**裁决为真（见 §2.3），c12 在其上再叠加 stages=4 得 −0.35%。
- 其余分支回退：c3/c16 的“自然操作数”在本 kernel 上**反向**（+6.9%/+3.1%，与 varlen 的 −6.1% 相反，
  原因：填充不再是墙钟项，而 nt 布局把 fixpipe 推高 8%、mte2 active BW 从 117 掉到 78 GB/s）；
  c5 的 epilogue 拆分**净增** 0.39 µs/迭代的重复 `exp` 且未换来重叠；c10/c15 的“全缓冲多版本”
  抬高 L1 压力反而变慢；c14 深到 6 段后 L1 版本不足、回退 8.5%。
- 单点证明后再组合（c11–c16）全部不叠加 ⇒ 组合均未合入。

**mechanism（c9/c12 为什么有效，幅度为什么小）**：pT_s/dsT_s 是 AIV 生产、AIC 消费的跨核缓冲，
单版本时 AIV 必须等 AIC 上一迭代的 mte1 读完才能写（WAR）。给它们两版本后该 WAR 放宽一代，
mte1 忙时由 0.680 → 0.598/0.617（−12%）、fixpipe 0.570 → 0.543；再叠 stages=4 后 mad 有效忙时
0.769 → 0.746（出现少量真实重叠）。但**所有管道的工作量本身不变**（mad 已 100% 峰值），
所以墙钟只降 1%——与 §2.1 第 1 条的“墙钟=依赖链”一致。

### 2.2 blocked 的结构候选（测量/编译证据）

| 候选 | 结论 | 证据 |
|---|---|---|
| `block_n = 80`（为 L0C 腾出 2 版本空间） | **编译失败** | `logs/round1/ds-v3-b2-8k-bfloat16_c6_bn80.log`：`nd2nz_copy.h:59 static_assert '40 % 16 == 0': ROWS must be a multiple of 16`（`_HN = _BN/2` 是 scatter 的行数） |
| `block_n = 112` | **不可行** | `_HN = 56`，56 % 16 = 8 ⇒ 同上断言；且 L0B `b_lo 24,576 + b_hi 43,008 = 67,584 > 65,536` |
| `block_n = 128` | **预算越界** | L0C `32,768×2 + 49,152 + 98,304 + 65,536 = 278,528 > 262,144`（106.2%）；L0B `24,576 + 49,152 = 73,728 > 65,536`（112.5%）。仅按行数约束可行（`_HN = 64`），故未提交编译 |
| L0C 双缓冲（`st_c`/`dpt_c` 各 2 版本）| **需改 kernel 结构**（非参数） | 现状 L0C 五累加器 221,184 B（84.4%）；直接双版本 = 270,336 B（103.1%）越界。可行形态：先把 `dpt` 并入 `st_c`（P pass 只需 `st_u`，`st_c` 排空后即可复用）→ 共享 score 累加器双版本 = 2×24,576 + dq 49,152 + dk 73,728 + dv 49,152 = **221,184 B（84.4%，恰好可行）**。但这属于 kernel 结构与语句顺序重排（跨迭代旋转 epilogue），超出 Stage 4 参数/配置调优范围，且调度器不提供该能力（§2.3） |
| 手工 `order`/`stage`（强制旋转） | **不生效** | `T.Pipelined(..., order=[0], stage=[0])` 编译通过但被忽略：`logs/probes/z3_scheduler_probe.log` 打印 `Z3 scheduler failed: solver returned unsat, falling back to topological sort` |

### 2.3 c9 的交错 A/B 裁决（T-2，<5% 差异）

`perf_opt/ab_interleaved.sh`（A/B/A、B/A/B 交替，3 对 6 次独立采集，同一平台口径 `T_op`）：

| 对 | baseline (µs) | c9 (µs) | 差 |
|---|---:|---:|---:|
| 1 | 154 316.6 | 153 421.6 | −895.0 |
| 2 | 154 304.8 | 153 369.6 | −935.2 |
| 3 | 154 369.5 | 153 399.9 | −969.6 |

中位数 154 316.6 → 153 399.9 = **−0.594%**，符号一致 6/6，臂内极差 ≤0.04%（44.9 µs）⇒ 该差异为真
（非噪声），但**远低于 3% 噪声阈值**，故仅作为 winner c12 的组件记录，不单独作为“有效提升”宣称。
（未用 `ab_test.py` 的原因：其 `--kernel-name` 包装按单 kernel 取中位数，无法对 5-launch 的
`T_op` 求和；此处以同一交错协议直接对平台口径 `T_op` 采样。）

### 2.4 Autotune Log

未使用 autotune：TileOPs 侧 `tune` 策略在本 op 已移除，工厂把 tiling 冻结为编译期常量。
本轮以手动分支完成参数扫描并全部记录在 §2.1 表内（`num_stages` 1/2/3/4/5/6 ×
版本标注 4 组 × 操作数布局 2 种 × `block_n` 3 种）。

### 2.5 精度回归（每个候选分支 + 最终版本）

| 检查 | 命令 | 结果 |
|---|---|---|
| 与 Stage 3 基线逐位对齐（c1/c3/c4/c5/c9/c12） | `perf_opt/verify_parity.py --case mla-dense-bwd-smoke-bfloat16 --cand …` | **PARITY_PASS**：`dk`/`dv` **逐位相同**；`dq_acc` 仅 store-atomic 到达序抖动（baseline 自身两次运行同样抖动），0 个单元超出 1e-3 |
| 尾块 + 双向 shift + B=2（final） | `--custom '{"batch":2,"seq_q":512,"seq_kv":608,"heads":4}'` | **PARITY_PASS**（dk/dv 逐位） |
| 4k 真形（final） | `--case ds-v3-4k-bfloat16` | **PARITY_PASS**（dk/dv 逐位） |
| fp16 smoke（final） | `--case mla-dense-bwd-smoke-float16` | **PARITY_PASS** |
| L0（27 例：smoke×2、非因果、尾块、shift 双向、request 隔离 5×2、atomic 契约、arm 等价、golden-vs-naive、契约拒绝） | `python perf_opt/MLADenseBwdOp.py --level L0` | **27/27 PASS**，max_diff 0.078373（与 Stage 3 同判据同量级），210.5 s |
| L1（6 例，两 dtype × 因果/非因果 × 多 batch） | `--level L1` | **6/6 PASS**，max_diff 0.112534，80.4 s |
| `big`（`ds-v3-b2-8k` 生产真形 B=2/S=8192/H=128，逐 request×head 切片 fp64 golden，8 对切片） | `--level big` | **PASS**（`logs/precision/big_final.log`）：dk/dv 两次 launch **逐位相同**、dq 漂移 7.15e-07；对同取整参考 **0 个元素越界**（对 fp64 golden 仅 dv 1 个元素超 atol+rtol）；request 1 的 key 0..63 非零（无跨 request 覆写）；kernel-only 2.936 µs/迭代（148.909 ms） |

日志：`perf_opt/logs/precision/{parity_*, L0_final.log, L1_final.log}`。
最终文件 `perf_opt/MLADenseBwdOp.py` 与 `MLADenseBwdOp_round1_c12.py` **字节相同**（md5 `d10538aa`），
L0/L1/parity 与 5 个 tune workload 的 `phase=final` 性能记录全部来自该文件（sha256 `4600a243…`）。

**合并非回退检查（本 kernel 全部 tune workload，c12 已合并进 final）**：

| workload | baseline (µs) | c12/final (µs) | Δ |
|---|---:|---:|---:|
| ds-v3-b2-8k-bfloat16 | 154 215.6 | 152 626.3 | **−1.03%** |
| ds-v3-4k-bfloat16 | 20 572.8 | 20 307.1 | **−1.29%** |
| ds-v3-4k-float16 | 20 570.8 | 20 318.8 | **−1.23%** |
| kimi-k2-8k-bfloat16 | 38 415.6 | 37 799.7 | **−1.60%** |
| ds-v2-lite-3k-bfloat16 | 6 292.7 | 6 044.7 | **−3.94%** |

### 2.6 终止判定（plateau + blocked）与参照结构 diff

参数/配置类候选（13 个有效分支）中**没有任何分支带来超过 3% 噪声阈值的提升**（最好的 c12 为 −0.95%），
且剩余结构候选全部因编译/容量/设计约束不可继续（§2.2）⇒ `blocked`（同时满足 plateau 的
“主要结构候选已充分验证”）。终止前做的**参照结构 diff**（T-1）：

| 参照 | 结构形态 | 与 current best 的差异 | 为何不可移植 |
|---|---|---|---|
| 同族 `MLA varlen bwd` 的 Stage 4 winner（`examples/mla_varlen_bwd_perf/perf_opt/MLAVarlenBwdOp.py`，md5 `6e2e6935`） | 与本 kernel **逐语句同构**（5 GEMM 显式 L0、same 6 L1 tile、same 5 L0C、AIV 半区 epilogue）；`num_stages=3` 得 −17.4% | 仅 stages 常量 | 已实测移植（c1/c11）：dense 上 0% / +3.4%。根因：varlen 的墙钟里 GM→L1 填充是**延迟受限**（mte2 1.09–1.22 µs/迭代 @ ~70 GB/s，占 30%），深流水直接吃掉它；dense 的填充已是 117 GB/s / 0.72 µs，墙钟由 epilogue 往返链支配（§2.1） |
| `moe/shared_expert_mlp` 的 `gemm_basic`（`annotate_buffer_versions` + Z3 流水范式） | 纯 GEMM、无 AIV epilogue、无跨核依赖 | 有 AIV↔AIC 交错与原子写 | 该范式只对“AIC 单核内 copy→gemm”链有效；本 kernel 的链跨 AIC/AIV 且被 `Z3 unsat → 拓扑序` 拒绝旋转（§2.3） |
| `examples/flash_attention/` 的 npuir SIMD 版（无显式 L0、向量化 score 路径） | 纯 Vector 实现 | 编程模式不同（本算子 Cube 化后 mad 已 100% 峰值，纯向量版 cube MFU≈0） | 会退回 Cube 化前的 3.5× 慢基线（DESIGN §11） |

未发现“同族 + 同编程模式 + 可跨核旋转”的现成参照 ⇒ 天花板结论只对**本冻结结构族**成立，
已在 `perf_feedback.md` 中如实标注该边界。

## 3. 关于测量口径的三条说明（供后续任务复用）

1. **`--launch-count` 数的是 kernel launch**：本 op 的一次调用 = `Range ×2` + `ZerosLike` +
   `_delta_main` + `_bwd_main` + `_dq_land_main` = **6** 个 launch（`test_torch.arange` 两次各一个样本）。
   按 `--warm-up=(1+W)×6 --launch-count=6` 可抓**恰好一次调用**，`window_ok` 逐次校验；
   直接沿用 `--launch-count=5` 会与调用边界错位（实测只抓到 2 个 `Range` 样本）。
2. **`T_op` 是逐 launch 求和**（与生产报告 `t_op_us` 一致），不是逐 op 取中位数。
   `ZerosLike`（fp32 `dq_acc` 清零，b2-8k 954.9 µs）与 `Range` 属一次调用的一部分，不可省略。
3. **pipe 占比用 sum 口径**：`aic_*_share = Σ pipe_time / Σ aic_time`；`*_ratio` 的逐块均值在
   因果 workload（首块 1 次迭代、末块 128 次迭代）上会被长短块权重带偏。

## 4. Final Summary

最终版本：`perf_opt/MLADenseBwdOp.py`（= Round 1 winner c12：`annotate_buffer_versions({pT_s: 2,
dsT_s: 2})` + `_STAGES = 4`），md5 `d10538aad61293e253f07428fb068aac`，
sha256 `4600a2438d8cf6d2c8a86437a9b408434480d25278fe543c9975be3317d30909`，
candidate_id `final-c12-ptver-stages4`（5 个 tune workload 的 final 记录全部来自该文件、同一 sha256）。

### Final Performance Test Data

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---:|---:|---|
| tileops/kernels/attention/mla_dense_bwd/mla_dense_bwd_kernel::_bwd_kernel+_delta_kernel+_dq_land_kernel | ds-v3-b2-8k-bfloat16 | 154215.635 | 152626.290 | final-c12-ptver-stages4 |
| tileops/kernels/attention/mla_dense_bwd/mla_dense_bwd_kernel::_bwd_kernel+_delta_kernel+_dq_land_kernel | ds-v3-4k-bfloat16 | 20572.769 | 20307.071 | final-c12-ptver-stages4 |
| tileops/kernels/attention/mla_dense_bwd/mla_dense_bwd_kernel::_bwd_kernel+_delta_kernel+_dq_land_kernel | ds-v3-4k-float16 | 20570.803 | 20318.755 | final-c12-ptver-stages4 |
| tileops/kernels/attention/mla_dense_bwd/mla_dense_bwd_kernel::_bwd_kernel+_delta_kernel+_dq_land_kernel | kimi-k2-8k-bfloat16 | 38415.571 | 37799.726 | final-c12-ptver-stages4 |
| tileops/kernels/attention/mla_dense_bwd/mla_dense_bwd_kernel::_bwd_kernel+_delta_kernel+_dq_land_kernel | ds-v2-lite-3k-bfloat16 | 6292.690 | 6044.739 | final-c12-ptver-stages4 |

final 的逐 workload 分解：

| workload | T_op (µs) | us/it | cube%: mad/mte1/mte2/fix/scal | vec%: vec/mte2/mte3 | MFU(cube 378) | 目标 | 判定 |
|---|---:|---:|---|---|---:|---|---|
| ds-v3-b2-8k-bf16 | 152 626.3 | 2.944 | 26.1 / 21.0 / 30.7 / 18.9 / 2.7 | 45.8 / 4.1 / 4.0 | 24.85%（基线 24.60%） | ≤130 ms、cube ≥32% | ❌ 未达（−1.03%，cube +0.3pt） |
| ds-v3-4k-bf16 | 20 307.1 | 3.028 | 25.8 / 20.8 / 29.4 / 17.8 / 2.8 | 46.5 / 4.4 / 4.0 | — | 同比例、不回退 | ✅ −1.29% |
| ds-v3-4k-fp16 | 20 318.8 | 3.030 | 25.8 / 20.7 / 29.3 / 17.8 / 2.8 | 46.5 / 4.4 / 4.0 | — | 同比例、不回退 | ✅ −1.23% |
| kimi-k2-8k-bf16 | 37 799.7 | 2.918 | 26.3 / 21.1 / 28.8 / 17.9 / 2.8 | 46.3 / 4.3 / 3.9 | — | 同比例、不回退 | ✅ −1.60% |
| ds-v2-lite-3k-bf16 | 6 044.7 | 3.154 | 25.0 / 21.9 / 21.7 / 17.3 / 2.9 | 46.4 / 4.1 / 4.0 | — | 同比例、不回退 | ✅ −3.94% |

**结论**：五个 tune workload **全部不回退、全部小幅提速（−1.0% ~ −3.9%）**；但
**目标（≤130 ms / cube ≥32%）未达**：`ds-v3-b2-8k` 仅从 154.2 ms 降到 152.6 ms（−1.03%），
cube 忙时占比 +0.3 pt。剩余缺口（152.6 → 130 ms 需 −15%，→ 62 ms 理论档需 −59%）**不是
tiling/参数可解**：见 §2.1 的“墙钟 = 依赖链”证据、§2.2 的容量/编译硬约束、§2.3 的调度器 unsat
证据。已按 `_shared/standards/perf-feedback.md` §2 产出 `perf_opt/perf_feedback.md`，返回 `[DESIGN_LIMIT]`。

### smoke 最终精度

`mla-dense-bwd-smoke-bfloat16` / `-float16`：L0-1 / L0-2 PASS（含在 27/27 内）+ parity `PARITY_PASS`
（dk/dv 逐位，dq_acc 1e-3 内）⇒ `precision_pass = true`。

生产真形（`ds-v3-b2-8k`，逐 request×head 切片 fp64 对照）：`--level big` **PASS**（见 §2.5 末行），
且 `run_big` 的 kernel-only 口径（148.909 ms / 2.936 µs·迭代⁻¹）与 msprof 口径
（`_bwd_main` 149 307.8 µs）一致 ⇒ 4-D 视图 / B1 契约 / 逐切片结果均未变。

### 中止原因

`blocked`：剩余结构候选分别被①编译断言（`block_n=80/112`：`nd2nz ROWS % 16`）、
②L0C/L0B 容量（`block_n=128`、L0C 双缓冲）、③工具链流水调度器 unsat（跨迭代旋转）挡死；
参数/配置维度已扫完（13 个有效分支，最优 −0.95%，无一超过 3% 噪声阈值）。

## 5. Skill Retrospective

### 流程问题

| area | issue | evidence | 建议 |
|---|---|---|---|
| 诊断口径 | 对 Cube 化算子的“管道忙时之和 ≈ 墙钟”只说明**无重叠**，不能定位**是哪条依赖卡住**；必须把生成的 AscendC 逐段读出依赖链（mad→fixpipe→AIV→mte3→mte1→mad→fixpipe）才能判定“参数不可解、需旋转”。现有 `references/iteration-diagnosis.md` 缺这一条读码方法 | §2.1 第 1 条：链长 2.94 µs vs 墙钟 2.975（管道和 2.822）——只看 pipe 会误判为“mte2 可优化”，实际 mte2 已不在链上（c4 版本化后 0%） | 在 iteration-diagnosis.md 增加“依赖链重建”步骤：用 `get_kernel_source()` 打印的代码按 `asc_sync_wait` 顺序重建链，逐段乘 pipe 单价，与墙钟对账；差值 <5% 即确认“迭代内串行” |
| 同族迁移 | 同族 varlen 的 Stage 4 winner（`num_stages=3`，−17%）在本 kernel 上为 0%——照搬会得出“该旋钮无效”的错误泛化 | §2.6 参照 diff：varlen 的收益来自“填充延迟受限”（mte2 1.09–1.22 @70 GB/s），dense 填充已 117 GB/s | pattern-library 的 “stages=3 knee” 条目应补**适用前提**：仅当 mte2 是墙钟的显著串行项（≥25% 迭代墙钟且 active BW 远低于平台可达）时成立 |
| 采集 | `msopprof --launch-count` 按 kernel launch 计数，多 launch 算子的 `warm-up` 必须按“每次调用 launch 数”整除；生产默认值会与调用边界错位 | §0/§3 第 1 条（`--launch-count=5` 只抓到 2 个 `Range` 样本，`window_ok=false`；改成 6 后逐次 true 且与生产报告差 <0.15%） | `references/profile-collection.md` 补“多 launch 算子：`per_call = 实测 launch 数`，`(1+W)×per_call` 跳过整数次调用，并以 `captured_samples == per_call` 校验” |
| 门禁/工件 | 分支被**编译断言**或**容量**挡死时也要留证（否则 gate 的“逐分支记录”只有性能分支） | §2.2：c6/c7 无性能记录，证据在 `logs/round1/*.log` 与 DESIGN 预算复算 | opt_log 增加“blocked 分支表（候选 / 结论 / 证据文件）”作为可与 perf_records 对账的固定小节 |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp |
|---|---|---|---|---|
| **Cube 化 MLA bwd（dense）的墙钟 = 迭代内依赖链**：`mad 0.35 → fixpipe 0.28 → AIV epilogue 1.35 → scatter+ub2l1 0.10 → mte1 0.10 → mad 0.47 → fixpipe(dq 原子) 0.29 ≈ 2.94 µs` vs 墙钟 2.975；AIC 五管道之和 2.822（mad 已 100% 峰值 13.5 TF/s/核）。故管道级优化（mte2 减半、nt 操作数、填充分版本）**全部无效或反向** | D | `perf_opt/perf_records.jsonl`（baseline/c1/c3/c4 行）+ `perf_opt/logs/probes/bwd_base.generated.cpp`（逐段 sync 顺序） | `python perf_opt/collect.py --impl perf_opt/MLADenseBwdOp_baseline_stage3.py --case ds-v3-b2-8k-bfloat16 --metrics pipe`（对照 c3/c4） | tilelang 0.1.15 / Ascend950PR_9579 / CANN 9.2.1 |
| **AIV epilogue 的代价不在 `exp` 而在 `nd2nz_scatter` + 跨核往返**：去掉两次 `exp` 只省 4.3%（0.13 µs/迭代）；整段 epilogue（含 2×`nd2nz_scatter` 6144 元素 + ub2l1）占墙钟 45%。把 epilogue 拆成两趟（P 趟 + dS 趟）反而 +21%（重复 `exp` 0.39 µs/迭代且调度器不重叠） | D | `logs/round1/*probe_noexp.log`（154 215.6 → 147 517.2）、`c5` 行（186 734.7） | `python perf_opt/collect.py --impl logs/probes/MLADenseBwdOp_probe_noexp.py --case ds-v3-b2-8k-bfloat16` | 同上 |
| **本工具链的 Z3 流水调度器在 “5-GEMM + AIV epilogue + 跨核 handshake” kernel 上 unsat**，静默退化为拓扑序：`num_stages`/`annotate_buffer_versions` 只改缓冲版本数、不改语句顺序 ⇒ 无法旋转 epilogue；`T.Pipelined(order=/stage=)` 也被忽略。同族 varlen kernel 亦 unsat，其 −17% 来自“版本化放宽填充 WAR”，不是旋转 | D（能力缺口） | `logs/probes/z3_dense.log`、`z3_varlen.log`、`z3_scheduler_probe.log`；c1（`% 3` 版本但 0%）与 c4（填充 2 版本但 0%）的实测 | `TILELANG_CACHE_DIR=/tmp/... python -c "…编译 _bwd_kernel(1,2,192,128,512,512,…)"`，grep `Z3` | 同上 |
| **L0/L0A/L0B 容量把 `block_n` 与“k/v 常驻 L0A”钉死**：`_HN=_BN/2` 必须 %16（scatter 断言）⇒ `_BN∈{32,96,160,…}`；`_BN=128` 超 L0C（278,528>262,144）与 L0B（73,728>65,536）；k+v 常驻 L0A 需 86,016>65,536，故每迭代 61,440 B 的 k/v 重装不可消（MTE1 ≥0.62 µs，设计要求的 ≤0.20 µs 不可达） | D | `logs/round1/*c6_bn80.log`（static_assert）、DESIGN §1.6.4 与 §2.1 第 4 条 | `python perf_opt/make_candidate.py …`（改 `_BN` 重编译即复现断言） | 同上 |
| 多 launch 算子的 Stage 4 采集配方（本次复用并修正 varlen 的 `K=launches+3`）：`per_call` 必须实测（`Range` 是两次 arange ⇒ 6 而非 5），`T_op=Σ launch`，同时落 `aic_*_share` 与 `us_per_iteration`，可直接与 DESIGN 单价对账（本例复现生产报告 <0.15%） | P | `perf_opt/collect.py`、§0 表（5 行偏差 ≤0.13%） | 同 §0 | 同上 |

### 回写 pattern-library

本轮 4 条 D 类结论已追加到平台知识库 `knowledge/constants.md`（`CONST-950-mla-bwd-dense-chain-bound`、
`CONST-950-aiv-epilogue-scatter-cost`、`CONST-950-z3-pipeline-unsat-mixkernel`、
`CONST-950-ascend-l0-capacity-mla-bwd`）与 `knowledge/capability_gaps.md`（Z3 调度器在 mix-kernel 上 unsat，
含工具链戳与 repro 命令）；最小 repro 以“同参数 `collect.py` + 三个候选文件”形式给出（见上表 repro 列），
无法脱离本 kernel 独立最小化的部分标 `repro-missing`。

### 记录规范性说明（`perf_records.jsonl`，append-only）

- 本轮共 27 行：5 baseline + 17 candidate（b2-8k 上 12 个分支 + c12 的另外 4 个 workload）+ 5 final。
  每个分支的精度状态见 §2.5；`phase=final` 的 5 行全部来自 `perf_opt/MLADenseBwdOp.py`
  （同一 sha256 `4600a243…`），与 `workload_inventory.json` 的 `final_us` 逐项一致。
- **唯一一次字段规范化（如实披露）**：5 条 `phase=final` 记录的 `artifact_path` 原写作
  `perf_opt/MLADenseBwdOp.py`（相对任务目录），而 gate 4 的 `S4-PERF-RECORDS-RECON` 按
  `signal-registry.md §5` 以 `perf_opt/` 为基准解析该字段，故规范化为 `MLADenseBwdOp.py`。
  **只改路径前缀**：`duration_us`、`artifact_sha256`（`4600a243…`）、`pipe`、`per_op_us`、`timestamp`
  一律未变；改前全文备份在 `logs/probes/perf_records.before_path_norm.jsonl`，可逐行 diff 核对。

## 6. 交付布局与采纳说明

- 最终版本：`perf_opt/MLADenseBwdOp.py`（自包含：主 pass + golden + L0/L1/L2/boundary/big 测试入口，
  `python perf_opt/MLADenseBwdOp.py --level L0|L1|L2|boundary|big|all` 可直接跑精度门禁）。
  与基准 kernel 的差异**只有两处**：模块常量 `_STAGES = 4`（原 2）与 query 循环前的一行
  `T.annotate_buffer_versions({pT_s: 2, dsT_s: 2})`；工厂签名、`_delta_kernel`/`_dq_land_kernel`
  契约、GM 4-D 视图、grid、精度口径全部不变。
- 采纳方式（由 conductor 在回归通过后翻转 wrapper 的双 import 切换块）：本目录文件可直接作为
  `mla_dense_bwd_kernel/perf_opt/_bwd_kernel.py` 落位（与 `MLADenseBwdOp.py` 字节相同）；
  `_delta_kernel` / `_dq_land_kernel` 未改动，可继续由 baseline 包提供。
- 本目录同时保留：`MLADenseBwdOp_baseline_stage3.py`（只读基线副本）、`MLADenseBwdOp_round1_c*.py`
  （16 个实验分支）、`collect.py` / `mla_dense_case.py` / `verify_parity.py` / `write_records.py` /
  `make_candidate.py` / `capture*.sh` / `ab_interleaved.sh`（采集与校验工具）、`profiles/`、`logs/`。
- 实验期内被编译/容量挡死的两个原型与两个诊断探针保留在 `logs/probes/`（不作为候选）。
