# 性能反馈（[DESIGN_LIMIT]）

## 触发判定

- 设计层归因: `examples/mla_dense_bwd/_bwd_kernel/DESIGN.md`
  §1.6.4 设计期 roofline 表（F/G 两行：「改进后 `0.80 + MTE1` ⇒ 40.5–50.6 ms，**关键硬条件
  `MTE1 ≤ 0.20 µs`**」与「管道级 `max` 下界 1.24 µs ⇒ 62.7 ms」）以及 §1.6.2/§5.2 的 tiling 选型
  （`block_n = 96 / block_m = 64`，L0C 五累加器**单版本** 221,184 B = 84.4%）。该投影的前提是
  「跨迭代流水可得 + L1→L0（MTE1）可压到 0.20 µs/迭代」，两者都是**设计层假设**，不是 tiling 参数。
- 结构性加速估计: **2.43x**（依据: 本任务实测 152.6 ms vs DESIGN §1.6.4 自己给出的管道级 `max`
  下界 62.7 ms；相对其「改进档」40.5–50.6 ms 为 3.0–3.8x）。
  并且**实测与假设直接矛盾**: DESIGN 要求 `MTE1 ≤ 0.20 µs/迭代`，实测 **0.60–0.68 µs/迭代**
  （b2-8k 217 088 B/迭代 @ ~300–340 GB/s），且该量在本结构下有下界 0.62 µs
  （见「实测」第 3 条：k/v 每迭代重装 61 440 B 不可消，因为 k+v 常驻 L0A 需 86 016 B > 65 536 B）。

## 参照锚定

- 参照实现:
  1. **同族最近亲**：`examples/mla_varlen_bwd_perf/perf_opt/MLAVarlenBwdOp.py`
     （md5 `6e2e693565b238c1cec71c268cab3170`，varlen 的 Stage 4 winner，−17.4%）。
     与本 kernel **逐语句同构**（5 GEMM 显式 L0 路径、同样的 6 个 per-iteration L1 tile、
     同样的 5 个 L0C 累加器、同样的 AIV 半区 epilogue、同样的 `pT_s/dsT_s` 跨核复用），
     差异只有 3 个常量与 packed 任务映射。
  2. **工具链内的 buffer-version 流水范式**：
     `examples/TileOPs/tileops/kernels/moe/shared_expert_mlp/shared_expert_mlp_kernel/_gemm_basic_kernel.py`
     （`annotate_buffer_versions({a1: num_stages, b1: num_stages})` + `T.Pipelined`）。
  3. **同算子族的 Cube 化基线对比**：`examples/TileOPs/reports/tileops/20261125_044632_077346_MLADenseBwdOp/`
     （本任务 baseline 的生产报告，154.4 ms / cube MFU 24.57%）。
  > 未找到「同族 + 同编程模式（Cube + mix AIV epilogue）+ 已实现跨迭代旋转」的现成参照；
  > 因此本文件的天花板结论**仅对本冻结结构族成立**，置信度按 T-1 规则降级标注。
- 不可移植论证:
  1. **varlen 的 `num_stages=3` 不可移植**（实测，非推理）：本任务已实测移植（候选 c1、c11），
     在 dense 上为 **+0.36% / +3.40%**（回退）。根因：varlen 的迭代墙钟里 GM→L1 填充是
     **延迟受限**的串行项（mte2 1.09–1.22 µs/迭代，active BW ~70 GB/s，占墙钟 30%），深流水
     直接吃掉它；dense 的填充已到 117 GB/s / 0.72 µs，且**不在依赖链上**（候选 c4 显式给
     q/do/qT/doT 两版本后墙钟变化 −0.00%），dense 的墙钟由「mad → fixpipe → AIV epilogue →
     scatter/ub2l1 → mte1 → mad → fixpipe(dq 原子)」串行链支配（见「实测」第 1 条）。
  2. **`gemm_basic` 的 version 流水范式不可移植**（实测）：该范式只对「AIC 单核内 copy→gemm」
     链有效；本 kernel 的链跨 AIC/AIV（`dual_copy` L0C→UB、`nd2nz_scatter`+`ub2l1`、mte1 回读），
     且本工具链的流水调度器在该 kernel 上 **unsat 后静默退化为拓扑序**（`Z3 scheduling failed:
     solver returned unsat, falling back to topological sort`，`perf_opt/logs/probes/z3_dense.log`；
     同族 varlen 亦 unsat，`z3_varlen.log`），语句顺序不被旋转，故任何 `num_stages` /
     `annotate_buffer_versions` 组合都只能放宽 WAR 距离，不能把 epilogue 挪到下一迭代。
     手工 `T.Pipelined(..., order=/stage=)` 亦被忽略（`logs/probes/z3_scheduler_probe.log`）。
  3. **纯 Vector（`examples/flash_attention/` 的 SIMD 形态）不可移植**：Cube 化后 mad 在忙时已逼近
     峰值（0.769 µs / 10.24 MFLOP = 13.32 TF/s/核，峰值 13.5 TF/s = 378/28 的 98.7%），退回纯向量实现会回到 Cube 化前
     约 3.5× 的慢基线（DESIGN §11）。

## 假设

DESIGN.md §1.6.4（F/G 行）：

> 「改进后 `0.80 + MTE1` ⇒ 40.5–50.6 ms，**关键硬条件 `MTE1 ≤ 0.20 µs`**」
> 「管道级 `max` 下界 1.24 µs ⇒ 62.7 ms」

以及 §5.2 tiling 与 §4.6 的缓冲规划所隐含的两条前提：

- **假设 A1**：5 个 GEMM 的 AIC 管道（mad / mte1 / mte2 / fixpipe）可以通过跨迭代软件流水
  与 boost 后的 `num_stages` 相互重叠，墙钟压到 `max(pipe) + MTE1` 量级；
- **假设 A2**：`MTE1` 可以降到 0.20 µs/迭代（设计未给出机制，只作为「关键硬条件」列出）。

## 实测

平台口径：msopprof `Task Duration(us)`，一次 capture 窗口 = 一次算子调用，
`T_op` = 该调用 6 个 launch（`Range×2` + `ZerosLike` + `_delta_main` + `_bwd_main` + `_dq_land_main`）
求和（与生产报告 `t_op_us` 同口径，本节全部数字复现生产报告 ≤0.13%）。
原始采集：`examples/mla_dense_bwd_perf/perf_opt/profiles/`，机器可读记录：
`examples/mla_dense_bwd_perf/perf_opt/perf_records.jsonl`。

1. **墙钟 = 迭代内依赖链，不是管道和**（`ds-v3-b2-8k-bfloat16`，1 420 032 次块迭代）：

   | 量 | 值 |
   |---|---|
   | `T_op` / `_bwd_main` | 152 626.3 µs / 149 307.8 µs（final）；154 215.6 / 150 897.5（baseline） |
   | 迭代墙钟 | 2.944 µs（final）/ 2.975（baseline） |
   | AIC 五管道忙时之和 | 2.822 µs（baseline: mad 0.769 + mte1 0.680 + mte2 0.720 + fixpipe 0.570 + scalar 0.083） |
   | 依赖链复算（按 `logs/probes/bwd_base.generated.cpp` 的 sync 顺序逐段求值） | mad(st,dpt) 0.35 + fixpipe 0.28 + AIV epilogue 1.35 + nd2nz_scatter/ub2l1 0.10 + mte1 0.10 + mad(dv,dk,dq) 0.47 + fixpipe(dq 原子) 0.29 = **2.94 µs** |

   ⇒ 链长与墙钟相差 <1%，即**迭代内全串行、跨迭代无旋转**；`mad` 在忙时已逼近峰值
   （0.769 µs 处理 10.24 MFLOP = 13.32 TF/s/核，为峰值 13.5 TF/s = 378/28 的 98.7%），算法层已无计算量可省（DESIGN §1.6.1 已完成）。
   AIV epilogue 独占墙钟 **45%**，且其代价不在 `exp`：去掉两次 `exp` 的探针只省 4.3%
   （154 215.6 → 147 517.2 µs，`logs/round1/*probe_noexp.log`），主体是两次 `nd2nz_scatter`
   （6144 元素 NZ 转换）+ 跨核往返。

2. **A2 不成立（MTE1 无法降到 0.20 µs）**：实测 MTE1 = **0.680 µs/迭代**（baseline）、
   0.598–0.617（winner），搬运量 217 088 B/迭代，折合 ~300–340 GB/s（已是该 L1→L0 路径的
   可达带宽）。10 条 L1→L0 copy 全是随迭代变化的数据；唯一可省的 61 440 B（28%）是 k/v 重装，
   其常驻化需要 L0A 同时容纳 k(36 864) + v(24 576) + pT(12 288) + dsT(12 288) + dsT^T(12 288)
   = **98 304 B > 65 536 B（L0A 容量）**，在 `block_n=96` 下不可行；`block_n=64/48` 可放但
   迭代数按 1.5×/2× 增长，每单位工作的 MTE1 字节数不降。⇒ MTE1 的实测下界 ≈0.62 µs，
   **是设计要求的 3.1×**。

3. **A1 不成立（该结构上跨迭代流水不可得）**：
   - 13 个参数/配置分支（`num_stages` 1/2/3/4/5/6 × 版本标注 4 组 × 操作数布局 2 种）中
     **没有任何分支超过 3% 噪声阈值**；最好者 `num_stages=4 + pT/dsT 两版本` = −0.95%（152 743.1 µs），
     且经交错 A/B/A/B 3 对裁决（−0.594%，符号 6/6，臂内极差 ≤0.04%）。
   - 生成代码证据：`num_stages=3` 确实让缓冲拿到 3 个版本（`bwd_c1_stages3.generated.cpp`
     有 12 处 `% 3` 索引），但**墙钟 +0.36%**；给填充缓冲显式两版本（c4）墙钟 −0.00%。
   - 调度器证据：`Z3 scheduling failed: solver returned unsat, falling back to topological sort`
     （dense 与同族 varlen 均如此）⇒ 语句顺序不旋转；手工 `order=/stage=` 被忽略。
   - 结构性替代（把 dV/dK/dQ 移入下一迭代以与 epilogue 重叠）所需的 **L0C 双版本越界**：
     现状 221 184 B（84.4%），`st_c`/`dpt_c` 直接双版本 = 270 336 B（103.1%）；可行形态要求
     **把 dP 并入 `st_c`**（P 趟只需 `st_u`，`st_c` 排空后即可复用）→ 2×24 576 + 49 152 + 73 728
     + 49 152 = 221 184 B（84.4%，恰好可行）。该重排需要改写 kernel 结构与语句顺序（跨迭代旋转
     epilogue + 原子 dq 存储的位置），属设计层改动，Stage 4 不得擅自施加。
   - 其它 tiling 出口均被硬约束挡死（实测/容量）：`block_n=80`、`=112` 触发工具链断言
     `nd2nz_copy.h:59 static_assert 'ROWS % 16 == 0'`（`ROWS = _HN = _BN/2`）；
     `block_n=128` 超 L0C（278 528 > 262 144）与 L0B（73 728 > 65 536）。

## 影响面

- **算子族**：所有「Cube 显式 L0 + 单版本 L0C + AIV 半区 epilogue + 跨核 handshake」的
  attention 反向 kernel——即 `_bwd_kernel` 的 dense（本文件）与 varlen（`mla_varlen_bwd`）
  两个实例。它们共享同一依赖链形态与同一调度器 unsat 现象；varlen 之所以仍能拿到 −17%，
  是因为它的填充项恰好是串行墙钟的显著成分，而不是因为链被旋转。
- **shape 范围**：全部 dense workload（1–4 k 至 2×8 k、H=16/64/128）。实测 5 个 tune workload
  的 per-iteration 墙钟 2.92–3.15 µs、cube MFU 24.6–26.4%，与 shape 弱相关 ⇒ 结论对整族成立。
- **改设计预期影响**：Stage 1（修订 §1.6.4 的 roofline 与硬条件、§4.6/§5.2 的 L0C 规划，
  给出「dP 并入 score 累加器 + 双版本」的新缓冲表与旋转后的语句顺序）→ Stage 2 检视 →
  Stage 3 重写 `_bwd_kernel`（本任务估算：改缓冲表 + 循环体拆成「A 段：打分/epilogue」
  与「B 段：梯度 GEMM 滞后一迭代」+ 首末迭代的 prologue/epilogue 处理，其中 dq 原子存储需
  随 B 段后移，`clear_accum=(kq == loop_st)` 需改为首迭代 prologue 显式清零；
  预期把链压到 `max(AIC 管道组, AIV 1.35)` ≈ 1.6–2.0 µs/迭代 ⇒ 82–102 ms，cube MFU 36–45%）→
  Stage 4 复测。**该估算是外推，未在本任务实测**，按 T-1 规则标注为未实证假设。

## 反馈结论

本冻结结构族（5 GEMM 显式 L0、L0C 五累加器单版本、AIV 半区 epilogue、`block_n=96`）的性能天花板由
**迭代内跨核依赖链**决定（152.6 ms / 2.94 µs per iteration，cube MFU 24.85%），其量级由 DESIGN
§1.6.4 的两条假设（MTE1 ≤ 0.20 µs、跨迭代流水可得）共同界定，而两者实测均不成立
（MTE1 下界 0.62 µs；调度器 unsat、L0C 双版本越界 103.1%），因此设计投影的 40.5–50.6 ms
与 62.7 ms 下界在本结构内不可达（实测为其 2.4–3.8×）；Stage 4 已在参数/配置维度穷举并交付
不回退最优版本（−1.0% ~ −3.9%）。

建议路由: **附录补记**（本任务为 `optimize` 场景、`stage_plan=[4]`，无 Stage 1，故「设计修订」
路径不适用；建议把本文件的触发判定 + 实测第 1/2/3 条与「dP 并入 score 累加器 + 双版本」
缓冲表作为性能反馈附录追加到 `DESIGN.md`，供该算子族后续 optimize / 再设计任务直接引用；
若未来对该族做设计修订（新任务或 `new_op`/`migration` 场景），再按上节「影响面」的
A/B 两段重构 + 首末迭代 prologue 方案执行）。
