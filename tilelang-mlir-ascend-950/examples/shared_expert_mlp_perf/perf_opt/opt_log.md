# SharedExpertMLPFwdOp — Stage 4 调优日志

- 算子：`SharedExpertMLPFwdOp`（`shared_expert_mlp`，family `moe`）
  `output = down(silu(hidden @ gate.T) * (hidden @ up.T))`，`hidden[T,H]`、`w_gate_up[2S,H]`、`w_down[H,S]`，bf16/fp16 输入、fp32 累加、输出 bf16/fp16
- 平台：`ascend-950pr-9579`（28 Cube / 56 Vector），CANN 9.2.1，TileLang ascend 0.1.15，单卡串行
- 基线（只读）：`examples/TileOPs/tileops/kernels/moe/shared_expert_mlp/`（`shared_expert_mlp.py` + `shared_expert_mlp_kernel/{_gemm_basic_kernel,_silu_mul_fused_kernel,splitk_reduce_kernel}.py`）
- 产物：`perf_opt/SharedExpertMLPFwdOp.py`（自包含，最优版本）、`perf_opt/perf_records.jsonl`、`perf_opt/workload_inventory.json`、`perf_opt/profiles/`、`perf_opt/logs/`、本文件
- 口径：平台 `msopprof` 的 `Task Duration(us)`；**T_op = 一次调用内全部 kernel launch 求和**（不是单 kernel 中位数），一次采集只 profile 一次算子调用，重复 3 个独立进程取逐位置中位数（`harness/capture.py`）

---

## 1. Workload inventory 与分类

benchmark 入口 `examples/TileOPs/benchmarks/ops/bench_fused_moe.py::test_shared_expert_mlp_bench` 展开的
6 个案例（manifest `SharedExpertMLPFwdOp.workloads`），**全部无 smoke 标记 → 全部 `kind=tune`**；
不额外枚举 manifest/L0/L1 测试案例。完整清单见 `workload_inventory.json`。

| workload_id | T | H | S | dtype | kind |
|---|---:|---:|---:|---|---|
| kimi-k2-t1 | 1 | 7168 | 18432 | bfloat16 | tune |
| kimi-k2-t512 | 512 | 7168 | 18432 | bfloat16 | tune |
| ds-v3-t32-bf16 | 32 | 7168 | 2048 | bfloat16 | tune |
| ds-v3-t32-fp16 | 32 | 7168 | 2048 | float16 | tune |
| ds-v3-t4096 | 4096 | 7168 | 2048 | bfloat16 | tune |
| glm45-t32 | 32 | 5120 | 1536 | bfloat16 | tune |

`kernel_id = shared_expert_mlp`：本 operator 的一次调用由「gate/up GEMM → gated 激活 → down GEMM」
多个 launch 组成，`duration_us` 记的是这几个 launch 的 T_op；逐 launch 明细见下表与
`profiles/<tag>/<workload>/summary.json`。

**采集有效性与排序理由**（Phase 1）：每 workload 采集后核对 `OpBasicInfo` 行数 = 该调用实际
launch 数（基线 3 或 4 行、候选 2 或 3 行），`captured_op_name` 均为本算子的 TileLang kernel
（`gemm_basic_kernel` / `gemm_nt_kernel` / `gate_up_fused_kernel` / `main_kernel` /
`silu_mul_kernel`），没有 Cast/Mul 类框架小算子。调优顺序按实测负担：
`kimi-k2-t512`（2142.7 µs，占 6 案例合计 43%）→ `ds-v3-t4096`（1879.5）→ `kimi-k2-t1`（727.4，
带宽型）→ `ds-v3-t32×2`（105.9/105.4）→ `glm45-t32`（58.4）。

---

## 2. Performance Test Data（baseline，`phase=baseline`，round 0）

命令模板（日志见 `logs/round0-baseline/`、`profiles/round0-baseline/*/raw_run*/`）：

```bash
python capture.py --impl baseline --workload <workload_id> --tag round0-baseline --repeats 3
# 内部：msopprof --output=<raw> --launch-count=8 --warm-up=0 --aic-metrics=BasicInfo,PipeUtilization,... \
#         python launch.py --impl baseline --workload <id> --warm-up 0 --launch-count 1
```

| workload_id | launch | captured_op_name | Task Duration(us) | 3 次进程实测 | 该 kernel 利用率 |
|---|---|---:|---:|---|---|
| kimi-k2-t512 | #0 | gemm_basic_kernel (gate/up) | 1386.073 | 1386.1 / 1386.6 / 1384.9 | cube 0.533，mte2 0.931 |
| kimi-k2-t512 | #1 | main_kernel (silu·mul) | 53.311 | 53.3 / 53.3 / 53.4 | 纯 Vector |
| kimi-k2-t512 | #2 | gemm_basic_kernel (down) | 703.300 | 705.8 / 698.9 / 703.3 | cube 0.520，mte2 0.985 |
| **kimi-k2-t512** | | **T_op** | **2142.684** | 2145.1 / 2138.7 / 2141.6 | MFU 50.1% |
| ds-v3-t4096 | #0 | gemm_basic_kernel (gate/up) | 1224.182 | 1224.6 / 1219.0 / 1224.2 | cube 0.548，mte2 0.950 |
| ds-v3-t4096 | #1 | main_kernel | 46.710 | 46.7 / 46.7 / 46.1 | 纯 Vector |
| ds-v3-t4096 | #2 | gemm_basic_kernel (down) | 608.595 | 607.6 / 608.6 / 608.7 | cube 0.542，mte2 0.754 |
| **ds-v3-t4096** | | **T_op** | **1879.487** | 1878.9 / 1874.3 / 1879.0 | MFU 50.8% |
| kimi-k2-t1 | #0 | gemm_basic_kernel (gate/up) | 490.784 | 490.8 / 486.5 / 491.0 | cube 0.050，mte2 0.993 |
| kimi-k2-t1 | #1 | main_kernel | 3.861 | 4.2 / 3.6 / 3.9 | 纯 Vector |
| kimi-k2-t1 | #2 | gemm_basic_kernel (down) | 232.767 | 227.6 / 234.0 / 232.8 | cube 0.050，mte2 0.996 |
| **kimi-k2-t1** | | **T_op** | **727.412** | 722.6 / 724.1 / 727.7 | 权重带宽受限 |
| ds-v3-t32-bf16 | #0..#3 | gemm(gate/up) / reduce(silu·mul) / gemm(down) / reduce | 59.958 / 9.339 / 28.325 / 8.283 | ±2% | cube 0.106 / 0.101，mte2 0.955 / 0.854 |
| **ds-v3-t32-bf16** | | **T_op** | **105.905** | 106.9 / 105.9 / 105.4 | |
| ds-v3-t32-fp16 | #0..#3 | 同上 | 59.628 / 9.224 / 28.609 / 7.958 | ±2% | cube 0.106 / 0.101 |
| **ds-v3-t32-fp16** | | **T_op** | **105.419** | 104.9 / 105.9 / 104.9 | |
| glm45-t32 | #0..#3 | 同上 | 30.227 / 5.268 / 16.863 / 6.045 | ±2% | cube 0.109 / 0.096 |
| **glm45-t32** | | **T_op** | **58.403** | 58.1 / 58.4 / 59.0 | |

与 `examples/batch2/mfu_baseline_v1.json` 的已发布基线一致（2147.0 / 1879.0 / 723.0 / 105.8），
差异 < 0.7%，说明本次 T_op 口径与 MFU 基线口径相同。

**设计估算 vs 实测（D-2 偏差行）**：`_gemm_basic_kernel/DESIGN.md` §1.6.0 的「设计期估算」行
把数值下界写成「待核验——缺少 ascend-950pr-9579 的 HBM 有效带宽与 Cube 峰值吞吐实测常数」。
本阶段补齐了这三个常数（见 §7 与回报知识库的 `constants.md`）：HBM 实测 1.42 TB/s、
L2→L1 聚合 ~2.7–2.9 TB/s、单核 MTE2 有效带宽 70 GB/s（block_k=64）/ 105 GB/s（block_k=128）。
设计期唯一可核对的、与常数无关的量——tile 内复用 FLOP/B = `bm·bn/(bm+bn)` = 85.3（128×256）——
实测口径下成立的效率上限正是 `85.3 × 2.3 TB/s ≈ 196 TF/s ≈ 52% MFU`，与基线实测 50.1% 吻合。
**设计假设的偏差**：DESIGN 判定 prefill 段「算力受限」（AI=1593），实测该段在 128×256 tile 下
`aic_mte2_ratio = 0.93–0.95` 而 `aic_cube_ratio = 0.53`，即**受 L1 填充带宽（L2 带宽）限制**，
不是 Cube 受限；把 tile 提到 L0C 上限 256×256 后 cube 占比升到 0.96–0.98（§4），该偏差已在
Stage 4 内用参数级手段闭合，不需要回写设计（故不触发 `[DESIGN_LIMIT]`）。

---

## 3. Baseline 现象分析（Round 1 的输入）

`PipeUtilization` 逐 block 读数（kimi-k2-t512 gate/up，`profiles/round0-baseline/`）：

- `aic_cube_ratio ≈ 0.47–0.55`（Cube 只有一半时间在算），`aic_mte2_ratio ≈ 0.93–0.99`
  （GM→L1 几乎满负荷），`aic_mte1_ratio ≈ 0.24`，`aic_fixpipe_ratio ≈ 0.01`；
- `aic_mte2_active_bw ≈ 70–88 GB/s/core` → 28 核聚合 ~2.3 TB/s，远高于 HBM 的 1.42 TB/s，
  说明大部分读来自 L2（A panel 7.3 MB 常驻、B tile 被 mt=4 复用）；
- 单块 `aic_time ≈ 65–74 µs`，`aic_cube_time` 恒为 34.75 µs → 块内 cube 只占 47%。

判据：**GEMM 受「L1 填充（MTE2）」限制**。每 K 步的填充字节 = `(bm+bn)·bk·2`，算力 =
`2·bm·bn·bk`，故每字节复用 = `bm·bn/(bm+bn)` FLOP/B：
128×256 → 85.3；256×256 → 128（L0C 上限 `bm·bn·4 ≤ 256 KiB` 下的最大值）。据此把
「提高 tile 到 L0C 上限」与「加长 MTE2 burst（block_k）」列为本轮主候选。

`kimi-k2-t1` 是另一类：`cube_ratio 0.05`、`mte2_ratio 0.99`、权重 792 MB 只读一遍 →
纯 HBM 带宽受限（1.42 TB/s 下界 566 µs，实测 727 µs = 78%），本轮候选是 burst 长度。

---

## 4. Iteration Log

### Round 1 — 参数级：tile 与 block_k（candidate `candA-tile256x256x128-ns3`）

**候选优化点（同一 base = baseline 派生，逐项单独验证）**

| 候选 | 单点改动 | 预期机制 |
|---|---|---|
| H1 | `block_m 128→256`、`block_n 256`（L0C 顶到 256 KiB 上限） | 每字节复用 85.3 → 128 FLOP/B |
| H2 | `block_k 64→128`（L1 仍 ≤ 512 KiB，`num_stages 3`） | 每行 DMA 段 128 B → 256 B，抬 MTE2 有效带宽 |
| H3 | decode 段 `split_k=4` → `1`（去 reduce launch） | 少 1 次 launch + 少 fp32 部分和往返 |

**H1/H2 的粗搜（`harness/probe.py`，wall-clock 筛选；kimi-k2-t512，m=512/n=36864/k=7168）**

| (bm, bn, bk, ns) | gate/up µs | 说明 |
|---|---:|---|
| 128, 256, 64, 4（Stage 3 形状） | 1382.9 | 基线 |
| 192, 256, 64, 4 | 1331.2 | AI 109.7 |
| 128, 512, 64, 4 | 1218.4 | AI 102.4 |
| 256, 256, 64, 4 | 1201.6 | AI 128，H1 单独生效 −13% |
| 256, 128, 64, 4 | 1384.5 | 窄 N 无收益 |
| **256, 256, 128, 3** | **811.8** | H1+H2，−41% |
| 256, 256, 128, 4 | 811.0 | L1 恰好 512 KiB |
| 256, 256, 256, 2 | 818.9 | L1 恰好 512 KiB |
| 128, 512, 128, 3 | 816.3 | AI 102 |

同网格在其它 shape 上的择优（`down` = m=512/n=7168/k=18432；t4096 = 4096/4096/7168 与
4096/7168/2048）：`256,256,128,3` 分别为 379.4 / 739.0 / 373.5 µs，均为该 shape 最优或并列
（`128,512,128,3` 在 down 上 378.8，差 0.2%；在 t4096 gate/up 上 741.6，差 0.4%），故取
**统一配置 256×256×128 / ns=3**，不引入 shape 相关的 tile 分派。`num_stages=2` 在大 shape 上
略快 0.2%，但在 glm45 down（k=1536 只有 12 个 K tile）上退化到 56.4 µs（vs 21.8），淘汰。

**H2 的机制确认（本阶段最重要的单点发现）**：`block_k=64` 时每个 L1 填充行是 128 B，MTE2
有效带宽 70 GB/s/core、且行数翻倍；`block_k=128`（256 B/行）后 active bandwidth 升到
101–105 GB/s/core。这是 −32% 的主因（AI 只解释 −13%）。

**H3 分支（decode 段去 split-K）**

| workload | split_k=4 + reduce（Stage 3 形状，probe） | split_k=1（probe） | 结论 |
|---|---:|---:|---|
| ds-v3-t32 gate/up（bk=128,bn=256） | 36.01 + reduce 9.3 | 37.3 + silu ~3 | 单遍净胜（去掉一次 launch 与 4.2 MB fp32 部分和） |
| glm45-t32 down（bk=128,bn=256） | 19.57 + reduce 6.0 | 20.65 + 0 | 单遍净胜 |

→ 采纳「decode 段也用单遍」，`_split_k` 默认返回 1（保留显式请求路径）。

**本轮对比表（平台时延 / 利用率 / 访存 / L0）**

| workload | current best（baseline）T_op | candA T_op | 提升 | gate/up cube→mte2 | down cube→mte2 | L0 |
|---|---:|---:|---:|---|---|---|
| kimi-k2-t512 | 2142.684 | **1226.262** | −42.8% | 0.533/0.931 → 0.958/0.900 | 0.520/0.985 → 0.977/0.934 | pass |
| ds-v3-t4096 | 1879.487 | **1153.131** | −38.6% | 0.548/0.950 → 0.951/0.877 | 0.542/0.754 → 0.885/0.674 | pass |
| kimi-k2-t1 | 727.412 | **533.660** | −26.6% | 0.050/0.993 → 0.070/0.990 | 0.050/0.996 → 0.066/0.995 | pass |
| ds-v3-t32-bf16 | 105.905 | **73.643** | −30.5% | 0.106/0.955 → 0.209/0.976 | 0.101/0.854 → 0.126/0.944 | pass |
| ds-v3-t32-fp16 | 105.419 | **73.046** | −30.7% | 0.106 → 0.211 | 0.101 → 0.128 | pass |
| glm45-t32 | 58.403 | **45.696** | −21.8% | 0.109/0.933 → 0.252/0.960 | 0.096/0.810 → 0.160/0.913 | pass |

winner：candA 全 workload 取胜，直接作为 current best（无回退，无需条件路径）。
`profiles/round1-candA/`、`logs/round1_candA.log`。

### Round 2 — 结构级：gated 激活融进 gate/up GEMM 的 Cube+Vector epilogue（candidate `candB-fused-epilogue`）

**现象（基于 candA 最新 profile）**：两个 GEMM 都已逼近本结构上限
（gate/up cube 0.958、down cube 0.977），但 `kimi-k2-t512` 仍有一次 53.3 µs 的
`silu_mul_kernel`（56.6 MB 往返，1.07 TB/s），占 T_op 的 4.3%；小 shape 上它占 12%
（dst32 8.9 µs / 73.6）。→ 唯一剩下的结构性开销就是这次独立激活 pass。

**候选**：单元改动 = 一个 block 同时算 gate tile 与 up tile（两个 fp32 L0C 累加器，
`bm·bn` 减半使「每字节复用」`1/bm + 1/(2bn)` 与 256×256 单累加器持平 = 128 FLOP/B），
epilogue 用 `T.dual_copy` 把两个累加器分片进两个 AIV 的 UB，在 fp32 里算
`(g·u)/(1+exp(−g))`（DESIGN §1.6.1 A1 链），一次 `vcvt(ROUND_R)` 后 `dual_copy` 写出。
中间 `[T,2S]` 张量与独立激活 pass 消失。约束：`dual_copy` 不能推导符号 region extent →
要求 M/S tile 精确（与 Stage 3 bias epilogue 同类前置）。

**本轮对比表**

| workload | candA（current best） | candB（fused） | 提升 | gate/up cube→mte2 | L0 |
|---|---:|---:|---:|---|---|
| kimi-k2-t512 | 1226.262 | **1203.571** | −1.9% | 0.958/0.900 → 0.970/0.941 | pass |
| ds-v3-t4096 | 1153.131 | **1133.373** | −1.7% | 0.951/0.877 → 0.966/0.876 | pass |
| kimi-k2-t1 | 533.660 | 532.614（走非融合回退，m=1 非精确 tile） | −0.2% | 不变 | pass |
| ds-v3-t32-bf16 | 73.643 | **65.427** | −11.2% | 0.209/0.976 → 0.208/0.970（少一次 launch） | pass |
| ds-v3-t32-fp16 | 73.046 | **65.203** | −10.7% | 同上 | pass |
| glm45-t32 | 45.696 | **40.138** | −12.2% | 0.252/0.960 → 0.244/0.934 | pass |

**分支操作笔记（负结果留痕）**

1. 首版把融合 kernel 的 N tile 取到 `cap = 65536/(2·bm)`（小 m 时 = 256）→ 小 shape 的块数
   从 16 掉到 8，dst32 反而慢 10%（81.1 vs 73.6）、glm45 慢 21%（55.2 vs 45.7）。改回
   「N tile = 单累加器 tile 的一半」(=128) 后，块数与不融合时一致，小 shape 转为 −11%。
2. 筛选时 `bn=96` 曾给出 t512 gate_up 808 µs（比 bn=128 快 4%），但该配置下
   `nchunk = 96 // 64 = 1`，epilogue 只写前 64 列 → **静默错**。已在 `_build_gate_up_fused`
   加 `block_n % VL == 0` 的构建期断言，N tile 只取 64 的倍数（bn=64 实测 941 µs，远差于 128）。
3. 融和版本**减少一次舍入点**：gate/up 的 fp32 累加结果直接进 epilogue（不再先落 bf16），
   gated 结果仍以 bf16 写出作为 down GEMM 的 A。实现后的端到端 NSE 从 1.6e-5 降到 1e-7 量级
   （更接近 fp32 参考），输出 dtype/单次舍入契约不变。

winner：candB（全 workload 不劣，6 例中 5 例取胜、1 例持平）→ 合入 current best
（`perf_opt/SharedExpertMLPFwdOp.py`），并复测本 kernel 全部 tune workload + 全部 smoke 精度。

### Round 3 — final 复测（同一最终候选，独立进程）

`phase=final`（`profiles/final2/`，`logs/final2_capture.log`，在**冻结后的交付文件**上独立重测）与
`phase=merged`（`profiles/round2-merged/`）逐 workload 偏差 ≤ 0.5%（t1 533.2 vs 532.6 = 0.1%，
t512 1204.3 vs 1203.6 = 0.06%，其余 ≤ 0.2%），无翻转；再早一次的 final 采集（`profiles/final/`）
也落在 ±0.2% 内，说明 3 独立进程的中位数口径可分辨 1% 量级差异。全部 tune workload 的
final 记录来自同一 `candidate_id = candB-fused-epilogue` 与同一文件 SHA256。

**中止条件**：`success`（kimi-k2-t512 2142.7 → 1203.4 µs = −43.8%，MFU 50.1% → 89.2%，达到
roofline 的 89.2%；其余 5 个 workload 全部改善，无回退）。预算未用完（3 轮 / 6 个实验分支）。

### 精度回归（每次合并与最终）

| 回归 | 范围 | 结果 |
|---|---|---|
| `perf_opt/SharedExpertMLPFwdOp.py --level all` | L0 10 例（5 个 manifest + m=1 + M 尾块 + k 阶梯 + 2 个 smoke）+ L1 7 例（fp16 全覆盖、M=100/128/2048） | 17/17 pass，`[PRECISION_PASS]`，NSE ≤ 1.6e-5（限 1e-4） |
| `harness/op_regression.py` | 逐案镜像 `tests/ops/test_fused_moe_shared_expert.py`（同名 11 例 + 6 个 manifest 行，`moe_close` 策略 atol=rtol=1e-2 + NSE ≤ 1e-4） | 17/17 pass |
| **`pytest tests/ops/test_fused_moe_shared_expert.py`（用 `harness/perf_opt_plugin.py` 把 op 的 kernel_map 指向 perf_opt 版本）** | 仓库原有 25 例（含 smoke / full / manifest / reject / dispatch / roofline） | **25 passed**（容差未改） |

---

## 5. Final Summary

### Final Performance Test Data

（`phase=final`，round 3，同一最终候选，数值取自 `perf_records.jsonl` 的 `phase=final` 行）

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---:|---:|---|
| shared_expert_mlp | kimi-k2-t1 | 727.412 | 533.160 | candB-fused-epilogue |
| shared_expert_mlp | kimi-k2-t512 | 2142.684 | 1204.301 | candB-fused-epilogue |
| shared_expert_mlp | ds-v3-t32-bf16 | 105.905 | 65.145 | candB-fused-epilogue |
| shared_expert_mlp | ds-v3-t32-fp16 | 105.419 | 65.681 | candB-fused-epilogue |
| shared_expert_mlp | ds-v3-t4096 | 1879.487 | 1135.039 | candB-fused-epilogue |
| shared_expert_mlp | glm45-t32 | 58.403 | 39.934 | candB-fused-epilogue |

### Final 派生指标（提升 / MFU / roofline）

| workload_id | 提升 | MFU(baseline→final) | roofline_us | final 相对 roofline |
|---|---:|---|---:|---:|
| kimi-k2-t512 | **−43.8%** | 50.1% → **89.1%** | 1073.9 | 89.2% |
| ds-v3-t4096 | **−39.6%** | 50.8% → **84.1%** | 954.6 | 84.1% |
| kimi-k2-t1 | −26.7% | 0.3%（带宽型） | 2.1 | — |
| ds-v3-t32-bf16 | −38.5% | 7.0% → 11.4% | 7.5 | — |
| ds-v3-t32-fp16 | −37.7% | 7.1% → 11.3% | 7.5 | — |
| glm45-t32 | −31.6% | 6.8% → 10.1% | 4.0 | — |

（`kimi-k2-t1`/`ds-v3-t32`/`glm45-t32` 的 FLOPs 极小、权重流占绝对主导，MFU 天然低，
MFU 口径对它们是诊断量；对应地它们的下界是 HBM 带宽：t1 权重 792 MB → 566 µs 下界，
实测 535.9 µs 已优于「1.42 TB/s 一遍权重」的粗估下界，说明权重读取有效率 > 1.42 TB/s
（部分 tile 复用 L2）。）

### 记录修订留痕

`perf_records.jsonl` 的第一版把 `hardware_id` 写成设备名（`Ascend950PR_9579`），而平台
契约要求任务状态里的 `hardware_id`（`ascend-950pr-9579`）。修正时按同一批采集
（`profiles/<tag>/<workload>/summary.json`，数值未变）重新生成全部 24 行；旧文件留档在
`logs/perf_records_v1_hardware_id_field.jsonl`，`duration_us` 与旧文件逐行一致，没有任何
测量值被改写。

第二版修订：给交付文件补上 `run_L2()` / `run_boundary()` 分层测试入口（gate 4 的
`S4-TEST-LEVELS` 要求）后文件哈希变化，`phase=final` 的 6 行按契约在**冻结后的文件**上重新
采集（`profiles/final2/`）并替换；`baseline`/`candidate`/`merged` 18 行保持首次采集的原始
`artifact_sha256`，未重算。

### 最终逐 launch 证据（`profiles/final/`）

| workload | launch 明细（Task Duration us） | 3 次进程 |
|---|---|---|
| kimi-k2-t512 | gate_up_fused 836.244 + gemm_nt(down) 368.057 | 1203.9 / 1203.5 / 1204.7 |
| ds-v3-t4096 | gate_up_fused 763.617 + gemm_nt(down) 371.422 | 1133.6 / 1133.7 / 1135.2 |
| kimi-k2-t1 | gemm_nt(gate/up) 354.242 + silu_mul 3.850 + gemm_nt(down) 175.068 | 533.4 / 534.3 / 532.6 |
| ds-v3-t32-bf16 | gate_up_fused 44.305 + gemm_nt(down) 20.840 | 65.5 / 65.0 / 64.7 |
| ds-v3-t32-fp16 | gate_up_fused 44.2 + gemm_nt(down) 20.9 | 65.3 / 66.5 / 65.1 |
| glm45-t32 | gate_up_fused 26.980 + gemm_nt(down) 12.954 | 39.4 / 40.0 / 40.0 |

### smoke / skip 排除

- benchmark 6 个案例都不带 smoke 标记 → 无 smoke 性能项；`skipped` 为空。
- 精度用的 smoke 集合来自 `tests/ops/test_fused_moe_shared_expert.py` 的 smoke 标记，
  最终一次全量（25 例）通过，未参与性能排序。

### 结论

- 主导 case `kimi-k2-t512`：2142.7 → **1203.4 µs（−43.8%）**，MFU 50.1% → **89.2%**，
  为 roofline（1073.9 µs）的 89.2%；剩余 129 µs 的构成已量化：
  gate/up 835.8（cube 0.970，其 2×128 宽 N tile 的 mad 效率 ~88% + epilogue ~43 µs）与
  down 367.6（cube 0.978，接近 357 µs 的 cube 下界）。
- 全部 6 个 shape 都改善 26–44%，无回退；小 shape（t1/t32/glm）绝对时延下降 26–38%。
- **未触发 `[DESIGN_LIMIT]`**：达到的是「L1 操作数 NT GEMM + 256 KiB L0C 累加器」这一数据流
  的 L2 带宽地板（每字节复用上限 128 FLOP/B，聚合 L2→L1 ~2.8 TB/s），不是设计层错误；
  参数/tile 级手段已闭合设计假设偏差。

---

## 6. Autotune Log

- 空间（受 DESIGN §5.2 契约约束）：`bm ∈ {128,192,256}`、`bn ∈ {128,256,512}`、
  `bk ∈ {64,128,256}`（须整除 k）、`ns ∈ {2,3,4}`；过滤器 = `bm·bn·4 ≤ 256 KiB`（L0C）、
  `(bm+bn+bn_fused)·bk·2·ns ≤ 512 KiB`（L1）、`16 | bm,bn,bk`、`k % bk == 0`。
- 筛选：`harness/probe.py`（wall-clock，4–6 次重复取中位数）先把 9 个组合在
  m=512/n=36864/k=7168 上排序，再用同一网格覆盖 down / t4096 / t1 / t32 / glm 的代表 GEMM。
- best config：`block_m=256, block_n=256, block_k=128, num_stages=3`（融合 gate/up 的 N tile
  = 128，由 `block_n//2` 与 L0C 上限共同决定）。
- 正确性：每个候选先跑 L0（自测）再采集；winner 的平台复测见 §5。
- 未使用框架级 autotune（该 kernel 无 tune 策略），筛选与复测均为本阶段脚本。

---

## 7. 本阶段实测硬件常数（回报知识库 `constants.md`）

| 常数 | 值 | 证据 |
|---|---|---|
| 单核 MTE2 有效带宽（`block_k=64`，128 B 行段） | 70 GB/s | `profiles/round0-baseline/kimi-k2-t512` PipeUtilization |
| 单核 MTE2 有效带宽（`block_k=128`，256 B 行段） | 101–105 GB/s | `profiles/final/kimi-k2-t512` |
| 单核 MTE2 有效带宽（`block_k=128`，小 shape/bn=512） | 95–133 GB/s | `profiles/final/{ds-v3-t32,glm45}` |
| 聚合 L2→L1 带宽（MTE2 满载，28 核） | ~2.7–2.9 TB/s | 同上（bytes = `(bm+bn)·k·2 × #blocks` / T） |
| HBM 有效带宽（权重流一遍，m=1） | 1.42–1.48 TB/s | `kimi-k2-t1`：792 MB / 535.9 µs = 1.48 TB/s |
| bf16 Cube 有效峰值（256×256 tile，大 K） | 356–359 TF/s（94–95% of 378） | down GEMM：`aic_cube_ratio 0.978` × 367.6 µs vs 357 µs 下界 |
| L0C 容量上限（fp32 累加器） | 256 KiB = 131072 elt... 实际 65536 elt (bm·bn) | `_check_tiles`；256×256 恰好通过 |

---

## 8. 复现命令

```bash
cd /workspace/tilelang-mlir-ascend-950
source .agents/platforms/toolchains/ascend/env-950pr.sh
export TMPDIR=/workspace TILELANG_CACHE_DIR=/tmp/opencode/tl-cache-opt-shared
cd examples/shared_expert_mlp_perf/perf_opt

# 精度（自测 L0/L1 + 镜像 op 测试 + 仓库原测试套件指向 perf_opt）
python SharedExpertMLPFwdOp.py --level all
python harness/op_regression.py --impl SharedExpertMLPFwdOp.py
( cd /workspace/tilelang-mlir-ascend-950/examples/TileOPs && \
  PYTHONPATH=<perf_opt>/harness TILEOPS_PERF_OPT_IMPL=<perf_opt>/SharedExpertMLPFwdOp.py \
  python -m pytest tests/ops/test_fused_moe_shared_expert.py -p perf_opt_plugin -q )

# 性能（每次 = 一次算子调用的 msopprof 采集，3 个独立进程取中位数）
cd harness && TAG=final IMPL=../SharedExpertMLPFwdOp.py REPEATS=3 bash capture_all.sh
python report_tables.py          # 由 perf_records.jsonl 生成 §4/§5 的对比表
```

---

## 9. Skill Retrospective

### 9.1 流程问题（本阶段实测暴露）

1. **T-4 批处理 runner 模板与本平台 profiler 语义不匹配**：`run_experiments_template.py`
   的适配点是 `build_bench_cmd`，但其默认假设「一次 msprof 调用可采多次 launch」。本平台
   `msopprof --launch-count=N` 的真实语义是「**整进程只采前 N 次 kernel launch**」——
   实测 `--launch-count=5 --warm-up=2` 时，7 次算子调用（每次 4 个 launch）只产出 5 行，
   把两个不同 GEMM（1390 µs / 701 µs）混进同一个名字的中位数，差点得出 `gemm=1046 µs` 的
   假数。本阶段改用 `harness/capture.py`：**一次采集只 profile 一次算子调用**（0 warm-up、
   `--launch-count` = 单次调用的 launch 数上限 8），T_op = 该调用全部 launch 之和；用 3 个
   独立进程重复取逐位置中位数。建议把这条语义写进 `profile-collection.md` 与模板。
2. **`perf_records.jsonl` 的 `hardware_id` 取值歧义**：`signal-registry.md` §5 只说「取值与
   本次任务状态一致」，而状态里同时存在 `hardware_id`（`ascend-950pr-9579`）与
   `platform_runtime.device.name`（`Ascend950PR_9579`）。gate 4 的 `S4-PERF-RECORDS-PLATFORM`
   按前者机械比对，写后者会整批失败。建议在契约里直接写出 `hardware_id` 的来源字段名。
3. **`Final Performance Test Data` 表是 5 列硬契约**：`gate_lint` 只认
   `kernel_id | workload_id | baseline_us | final_us | final_candidate_id` 恰好 5 列，
   多列会被当作非数据行跳过，随后报「Final 表缺少 …」。建议在 skill 的「Final Summary」
   要求里直接给出表头模板（现在只有字段名清单）。
4. **smoke 分类在 TileOPs benchmark 里恒为空**：benchmark 的 6 个案例都没有 smoke 标记，
   而 op 层 `tests/ops/` 里有 smoke 标记——两处 smoke 语义不同（benchmark 采集 vs 精度回归）。
   本阶段按 `profile-collection.md` §1 判为全 tune，并在 inventory 里显式记
   `precision_regression`。建议在文档中明确「benchmark smoke ≠ pytest smoke」。

### 9.2 value point proposals

| id | vp_type | 结论 | evidence（证据三件套） |
|---|---|---|---|
| VP-950-GEMM-BK-BURST | **D**（实测数据 / 代价常数） | 本平台 bf16 L1 操作数 GEMM 的 MTE2 有效带宽由 **每行 DMA 段长度**决定：`block_k=64`（128 B/行）≈ 70 GB/s/core，`block_k=128`（256 B/行）≈ 101–105 GB/s/core；同一 tile 上仅此一项就使 kimi-k2-t512 gate/up 从 1201.6 → 811.8 µs（−32%），而 tile 从 128×256 提到 256×256（AI 85.3→128）只值 −13% | evidence: `profiles/round0-baseline/kimi-k2-t512/*/PipeUtilization*.csv`（aic_mte2_active_bw≈70）vs `profiles/final/kimi-k2-t512`（≈101–105）；`opt_log.md` §4 Round 1 表。repro: `.agents/platforms/toolchains/ascend/knowledge/repro/PL-950-gemm-l1-fill.py`。toolchain_stamp: `7a52f2f1274df07d8f4c9c01353ce9d290f88c16e85047de0c425b6a009de4aa`（TileLang ascend 0.1.15 + CANN 9.2.1，ascend-950pr-9579） |
| PL-950-gemm-l1-fill | **P**（模式方法） | L1 操作数 NT GEMM 的判据是「每 K 步复用 FLOP/B = `bm·bn/(bm+bn)`」；`aic_mte2_ratio≈1` 且 `aic_cube_ratio≈0.5` 时，把 tile 顶到 L0C 上限（`bm·bn·4 ≤ 256 KiB`，即 256×256）是首选；先抬 `block_k` 再抬 tile（顺序反了会只拿到 1/3 收益） | evidence: 同 VP-950-GEMM-BK-BURST + `opt_log.md` §3/§4；工具链同 stamp。repro: 同文件（两轴各自单独对照 + 组合） |
| VP-950-FUSED-GATE-EPILOGUE | **P**（模式方法，含负结果） | 把 `silu(gate)*up` 融进 gate/up GEMM 的 Cube+Vector epilogue（两个 fp32 L0C 累加器 + 双 `dual_copy`）可省掉独立激活 pass（t512 −1.9%、t32 −11%），**但 N tile 必须取单累加器的一半**（否则小 m 块数腰斩，实测 dst32 反而 +10%）；epilogue 的列块宽必须是 64 的倍数，否则 `nchunk = bn//64` 静默丢列（bn=96 曾给出「更快」的 808 µs 假象） | evidence: `profiles/round2-candB`（bn=256 小 m 退化 81.1）、`profiles/round2b-candB-bn128`（65.1）、`profiles/final`；`opt_log.md` §4 Round 2 负结果 1/2。repro: 同 PL-950-gemm-l1-fill.py 的第 3 组（fused vs unfused，自包含）。toolchain_stamp 同上 |
| VP-950-SPLITK-NOT-FREE | **D** | `block_k=128` 之后 decode 段（T=32）不再缺并行度，Stage 3 的 split-K=4 + reduce 变成净负：ds-v3-t32 gate/up 单遍 37.3 µs + 3 µs 激活 vs 分片 36.0 µs + 9.3 µs reduce；glm45 down 20.7 vs 19.6+6.0。去掉 split-K 同时把 launch 数从 4 降到 3 | evidence: `opt_log.md` §4 Round 1 H3 表；`profiles/round1-candA/ds-v3-t32-bf16`（2 launch）vs `profiles/round0-baseline/ds-v3-t32-bf16`（4 launch）。toolchain_stamp 同上 |
| R-950-PROFILE-ONE-CALL | **R**（流程规则） | 多 launch 算子的 Stage 4 采集必须「一次采集 = 一次算子调用」，用 `--launch-count ≥ 单次调用 launch 数`，且 **同名多 launch 不能用跨调用的中位数聚合**（同名 launch 可能相差 2×）；T_op 用该调用全部 launch 求和 | evidence: `opt_log.md` §9.1-1 的实测（`--launch-count=5 --warm-up=2` → 5 行、混杂 1390/701）；`profiles/round0-baseline-4x`（旧口径）与 `profiles/round0-baseline`（新口径）对照。toolchain_stamp 同上 |
| C-950-shared-expert-mlp-perf | **C**（案例索引） | 本任务：Cube GEMM + Vector 激活分解算子的 Stage 4 案例，50.1% → 89.2% MFU（−43.8% T_op），有效手段排序 = block_k burst > tile 顶 L0C > 激活融合 > 去 split-K；无效手段 = 窄 N tile、bn=96（正确性陷阱）、num_stages=2（短 K 退化） | evidence: `examples/shared_expert_mlp_perf/perf_opt/opt_log.md` + `perf_records.jsonl` |

---

## 10. 生产采纳（Stage 5 接线 + 单算子报告验收）

### 10.1 文件清单

| 动作 | 文件 |
|---|---|
| 新增（交付副本） | `examples/TileOPs/tileops/kernels/moe/shared_expert_mlp/shared_expert_mlp_kernel/perf_opt/SharedExpertMLPFwdOp.py`（sha256 `2c084887da85c325da6170f810780c30b455c89741e7230628126d1456fa4a5f`，与本阶段 `perf_opt/` 交付件逐字节相同） |
| 新增 | `…/perf_opt/__init__.py`（re-export `SharedExpertMLPKernel` 与 `TUNED_*` 常量） |
| 备份 | `…/shared_expert_mlp_kernel/history_version/shared_expert_mlp_pre_opt.py`（采纳前 wrapper 原文） |
| 修改（仅切换块 + 末尾 re-export） | `…/shared_expert_mlp/shared_expert_mlp.py` |

切换块内容：baseline 三工厂 import 保持有效（作为回滚源），新增
`from .shared_expert_mlp_kernel.perf_opt.SharedExpertMLPFwdOp import SharedExpertMLPKernel as _PerfOptSharedExpertMLPKernel`，
并在文件末尾 `SharedExpertMLPKernel = _PerfOptSharedExpertMLPKernel`。`__all__`、op 层
`from tileops.kernels.moe.shared_expert_mlp import SharedExpertMLPKernel`、manifest/test/bench 均未改。
回滚 = 删除末尾 3 行（Stage 3 类本体完好）。

> 采纳的是**算子级 class**而非三个工厂：Stage 4 的收益（256×256×128 tile、ns=3、去 split-K、
> 融合激活 epilogue）有一半在 host 分解逻辑里，只换三个工厂不会带来任何加速。

### 10.2 生产回归

```
cd examples/TileOPs && python -m pytest tests/ops/test_fused_moe_shared_expert.py -q
→ 25 passed, 14 warnings in 3.13s
```

- 容差未改（`moe_close`：atol=rtol=1e-2 + NSE ≤ 1e-4）。
- **兼容性适配：无**。perf_opt 的 `SharedExpertMLPKernel` 不是 `tileops.kernels.kernel_base.Kernel`
  的子类，但 op 层只把它当「可调用 + 有 `config`/`dtype_str`」使用（`_get_or_create_kernel`
  只做构造与缓存），25 例全部通过，未改 op。
- 注：任务给的命令行带 `--tileops-device npu --tileops-target detect`，本仓库
  `examples/TileOPs/conftest.py` 未注册这两个 option（该 CLI 形状属于另一份 TileOPs 变体），
  pytest 直接 `error: unrecognized arguments`。设备后端在本仓库按 `tileops.device` 自动探测
  （npu），故实际执行命令去掉这两个参数后运行，不改变被测路径。

### 10.3 单算子报告（集成包、生产路径）

```
TILEOPS_TASK_DIR=/workspace/tilelang-mlir-ascend-950/examples/shared_expert_mlp TMPDIR=/workspace \
python -m tileops.reporting.cli run --op SharedExpertMLPFwdOp --prof-mode msprof
→ status=passed, run_id=20261124_141934_327213_SharedExpertMLPFwdOp
```

`report.md`：correctness `100.0% (25/25)`，Avg Max Abs Error `1.97e-04`，`6` 个性能 shape；
`run.json` 中 `invalid=0`、`invalid_profile_count=0`、`invalid_acceptance_count=0`，
6 个 case 的 `invalid_reason` 均为 `null`。

**生产路径确实用到优化内核**：`benchmark.json` 的 `tileops` 记录里 `captured_op_names` 为
`['gate_up_fused_kernel', 'gemm_nt_kernel']`（t512 / t4096 / t32×3）与
`['gemm_nt_kernel', 'silu_mul_kernel']`（t1，m=1 非精确 tile 走非融合回退），
`config = {block_m:256, block_n:256, block_k:128, num_stages:3, split_k:1, fuse_silu:true}`。

| case | 报告 T_op(us) | Stage 4 final(us) | Stage 3 基线(us) | 报告 MFU(名义/实测峰值) | 提升 |
|---|---:|---:|---:|---|---:|
| 512x7168x18432 | **1203.313** | 1204.301 | 2147.0 | **89.2% / 91.1%** | **−43.9%** |
| 4096x7168x2048 | 1132.632 | 1135.039 | 1879.0 | 84.3% / 86.0% | −39.7% |
| 1x7168x18432 | 535.080 | 533.160 | 723.0 | 0.39%（带宽上限 0.42%） | −26.0% |
| 32x7168x2048 bf16 | 65.233 | 65.145 | 105.8 | 11.4%（带宽上限 13.4%） | −38.3% |
| 32x7168x2048 fp16 | 64.433 | 65.681 | 105.4 | 11.6%（带宽上限 13.4%） | −38.9% |
| 32x5120x1536 | 40.200 | 39.934 | 58.4 | 9.94%（带宽上限 13.4%） | −31.2% |

报告 Acceptance 列在 6 个 case 上都标 `FAIL`：该列是「MFU ≥ 90% 目标」的门槛，而 t512 的名义
峰值口径是 89.2%（按报告自己标定的实测峰值 370.5 TF/s 则为 91.1%），其余 5 例是权重带宽受限的
decode/small-M shape（MFU 上限 0.4%–13.4%）。`status=passed` 由正确性门槛决定，`invalid_*` 全 0，
无 profile 失效、无张量契约失效。

### 10.4 生产记录

`perf_records.jsonl` 追加 6 行 `phase=merged, round=4`（`report_run_id` 指向上面的 run-id，
`msprof_raw_path` 指向报告 artifacts），`artifact_path` 为集成包内的 perf_opt 文件（与 final 记录
同一 sha256）。stage 门禁复核：`statectl gate 4` → `pass: true`，failures 0 / warnings 0。
