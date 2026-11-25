# 性能反馈（[DESIGN_LIMIT]）

## 触发判定

- **设计层归因**：`examples/ssd_chunk_coupling/_ssd_chunk_coupling_kernel/DESIGN.md`
  - **§1.6.3 布局决策 #5（阻塞级）**：epilogue 归 AIV——`L0C →(M 切分)→ UB(fp32) → SimtVF 掩码+cast → UB(dtype) → GM`；同表 **#4 明确否决** Cube fixpipe 直存（`T.copy(acc, cb)`，理由："Cube 无法施加因果掩码 ⇒ 必须第二次遍历清零（+20% 写）"）。
  - **§5.5 分核策略**：`grid = B·NC·G·nlt·nst`（wl3 = 1152 块，每块一个 128×128 tile），并明确"**不做"固定启动 28 个核 + 核内串行（persistent）**：本工具链未认证常驻式核内循环 / 跨 block barrier，且"核内串行化不摊薄启动开销反而会引入手工同步风险"。
  - **§1.6.4 主选**：tile 级 grid + 条件 trip 的 K 循环 + 统一 AIV 掩码 epilogue。
  这三项是**选型**（epilogue 归属引擎、块粒度与块内串行性、栅格组织），不是 tiling 参数：本轮已穷举可参数化维度——`BL ∈ {64,128}`、`SimtVF threads ∈ {64,128}`、6 种掩码写法（`select` / 输出 dtype select / `x·cast(cond)` / 无掩码 / 条带分区）、零缓冲高度 4/8/16/32/64——**单靠参数最多 1.26x**；2.07x 只能由结构变更得到。
- **结构性加速实测 2.07x，且与上述设计假设直接矛盾**：
  - 矛盾 1（§5.5）：实测"每块一个 `(batch, chunk, group)` 行面板、块内三角 tile 串行"（wl3 288 块）**比主选快 2.09x**，且不需要手写跨核 flag（用两个 L0C 累加器 + Python 级展开的 tile 序列即由默认 auto-schedule 重叠）。§5.5 否决 persistent 的两条理由（flag 风险、不摊薄启动开销）在本形态下均不成立。
  - 矛盾 2（§1.6.3 #4）：Cube fixpipe 直存**不需要**第二次遍历清零——按"该 tile 是否需要逐元素修改"把输出 tile 分三类后，严格下三角 tile 一次 fixpipe 直存（无 AIV），严格上三角 tile 由 AIV 写**不相交**的零 region，只有对角 tile 保留精确 select 掩码。实测该组合把 AIV vec 段从 75.5 µs 降到 45.2 µs、总时从 173.1 降到 136.6 µs（再叠加行面板到 82.9 µs）。
  - 量化依据：`examples/ssd_chunk_coupling_perf/perf_opt/perf_records.jsonl` 的 `baseline-v3`（173.076 µs）与 `c-ul`（136.577）/ `c-rp`（90.204）/ `c-rp-z8`（82.541）/ `final-rp-z8`（82.919）三阶段实测，同平台同口径。

## 参照锚定

- **参照实现**：同族算子 `examples/TileOPs/tileops/kernels/mamba/ssd_chunk_scan/ssd_chunk_scan_kernel/perf_opt/_ssd_chunk_scan_fwd_kernel.py`（Stage 4 已优化版：任务级持久流水 + 任务级 L1 驻留 + AIV subid 蛇形分片）；知识条目 `.agents/platforms/toolchains/ascend/knowledge/gemm-tiles.md` 的 `PL-950-gemm-l1-fill`（L1 填充带宽判据）与 `.agents/platforms/toolchains/ascend/references/programming.md` 的同步/flag 边界。
- **不可移植论证**（逐条结构差异 + 依据）：
  1. `ssd_chunk_scan` 的持久任务流水依赖**显式跨核 flag 配对与手工深度 2 流水**（同平台 `PL-950-manual-sync`），而 `programming.md`「同步与缓冲版本」明确"手工跨核 flag、循环复用下的复杂同步和跨 block batch barrier 仍未认证"。直接照搬即引入未认证同步形态。→ 本算子改用**块内多 tile + 双 L0C 累加器 + 默认 auto-schedule**（不手写 flag），实测生效。依据：文档条款 + 本任务实测。
  2. `PL-950-gemm-l1-fill` 的判据是"先加长 block_k、再把 tile 顶到 L0C 上限"，针对 K 远大于 M/N 的大 K GEMM；本算子 K = d_state ≤ 128 固定、L0C 需容纳 2×128×128 fp32 累加器，无法以同形态顶满。依据：条目正文 + 本任务 `ais` 实测（cube 仅 0.93 µs/块，AIC 主导项是 mte2 1.59 µs）。→ 该参照的结论**不适用**，非"未移植"。
  3. 仓库内**未找到**与 `mamba2 chunk coupling`（`tril(C·Bᵀ)` per chunk）同 shape 族的第二个实现（`grep` 全仓只有本算子与其 history_version）。故"是否还能更接近 roofline"的下述判断**置信度降级**；但"§5.5 / §1.6.3 的否决依据被实测推翻"这一结论不依赖参照，由本任务实测直接给出。
  4. 纯推理否决的标注：本反馈**未**断言"当前工具链内 51.4 µs roofline 不可达"——实测 82.9 µs 仍在改善通道上（剩余限制为 AIV 逐元素掩码吞吐与 0.90 TB/s 带宽利用率），不作不可达断言。

## 假设

被推翻/需修订的 DESIGN.md 原文要点（逐条引用）：

1. §5.5：「wl3：极大规模 —— 1152 块 = 41 波，波效率 98%，**不做"固定启动 28 个核 + 核内串行"（persistent）**……核内串行化不摊薄启动开销反而会引入手工同步风险。」
2. §1.6.3 表 #4：「epilogue 归 Cube：`T.copy(acc → cb region)` fixpipe 直存（带 fp32→dtype 量化）……**可见（probe P1：asc_copy_l0c2gm，quant_mode=1），但 Cube 无法施加因果掩码 ⇒ 必须第二次遍历清零（= §1.6.1 未采纳项，+20% 写）**……❌」。
3. §1.6.3 表 #5：采纳「epilogue 归 AIV：`dual_copy(acc→cv)` + `SimtVF` 掩码 + `dual_copy(co→GM)`」，并以"probe P5/P6/P7 实证"为其执行证据（未含吞吐代价）。
4. §1.6.0 R3/§5.5 的分核结论：每 tile 一块、无块内串行。

隐含假设（本轮实测证伪）：**"AIV SIMT VF 逐元素段的代价可以忽略 / 与 Cube 段并行"**——实测 SIMT VF 掩码段吞吐仅 2.5 元素/cycle/AIV（`profiles/baseline/main_kernel_zvc4_ucy`：8192 元素用 3284 cycle），且同块 AIC 段与 AIV 段串行（block ≈ aic + aiv），使 AIV busy 占 wall 的 83%。

## 实测

- **平台与口径**：`ascend-950pr-9579` / `ascend` 工具链（tilelang 0.1.15，CANN 9.2.1 compiler 12.3.0，runtime/OPP CANN 9.2.1）；
  设备 Ascend950PR_9579（cube 28 / vector 56）。
  测量口径 = 平台 `measurement`：`msopprof` kernel-only **`Task Duration(us)`**（`--launch-count=1 --warm-up=0 --aic-metrics=Roofline`，
  **一次采集只跑一次算子调用**；该算子每次调用只产生一个 kernel `main_kernel`，故 T_op = 该 launch 的 Task Duration）。
  原始采集：`examples/ssd_chunk_coupling_perf/perf_opt/profiles/`（baseline / experiments / final 三组 `OPPROF_*` 目录，
  含 `OpBasicInfo.csv` 与 `PipeUtilization.csv`）；结构化记录：同目录 `perf_records.jsonl`。
- **dispatch**：`perf_opt/experiments/bench_one.py --module <candidate.py> --workload <wl>`（工作目录 `examples/TileOPs`，
  环境 `source .agents/platforms/toolchains/ascend/env-950pr.sh`，`TILEOPS_TASK_DIR=examples/ssd_chunk_coupling_perf`）。
- **三阶段对比（wl3 = mamba2-grouped-prefill, B=3,S=3072,G=8,N=128,Q=256,fp16）**：

| 阶段 / 结构 | Task Duration (µs) | AIC busy | AIV busy | AIV vec | 采集 |
|---|---:|---:|---:|---:|---|
| baseline-v3：每 tile 一块（1152 块）+ 统一 AIV 掩码 | 173.076 | 62.3 µs (36%) | 143.3 µs (83%) | 75.5 µs | `profiles/baseline/v0_base_wl3_r1` |
| c-ul：每 tile 一块 + tile 类分派 epilogue | 136.577 | 64.0 µs (47%) | 103.9 µs (76%) | 43.4 µs | `profiles/experiments/e_ul_wl3_r0` |
| c-rp：行面板（288 块）+ 双 L0C 累加器 | 90.204 | 52.6 µs (58%) | 77.7 µs (86%) | 42.6 µs | `profiles/experiments/e_rp_wl3_r1` |
| **final-rp-z8**（= `perf_opt/SSDChunkCouplingFwdOp.py`） | **82.919**（另一采集 82.507） | 52.5 µs (63%) | 71.1 µs (86%) | 41.2 µs | `profiles/final/SSDChunkCouplingFwdOp_wl3_r1` |

  比值 173.076 / 82.919 = **2.09x**；对历史同口径 baseline 171.060 µs 为 **2.06x**。
- **其他 workload（无回退）**：mamba2-780m fp16 13.208 → 8.411 µs（1.57x）；mamba2-2p7b bf16 23.041 → 16.550 µs（1.39x）；
  smoke fp16 2.762 → 2.769 µs、smoke bf16 2.81 → 2.687 µs（smoke 走 Stage-3 回落路径，代码未变）。
- **精度**：`perf_opt/SSDChunkCouplingFwdOp.py` L0 9/9、L1 8/8、L2 4/4、boundary 10/10 全通过，max_diff=0.125
  （bf16 1 ulp，与 Stage-3 基线一致），容差未改。
- **否决依据的实测反驳**：DESIGN §1.6.4 的裁决计划（主选 vs 备选 B）本轮已执行：备选 B（`trips` 恒定）
  = 180.913 µs > 主选 173.076 × 1.045，主选胜出——**但两变体都建立在"每 tile 一块"这一被推翻的块粒度上**，
  行面板结构把上三角 tile 的 GEMM 完全消除（该 tile 的搬入+GEMM 实测值 ≈ 7.8 µs）。
- **被否决的替代形态（不构成未实证假设，均有实测）**：`BL=64` tile 311.8 µs 且 L0-6 失败（`e_ul_bl64`）；
  对角 tile 的 16 行条带 fixpipe 分区编译失败（`CheckCompactL0Region`：`Split/non-zero-origin L0 tiles are not
  supported`）；`T.clear/T.fill` 对 `shared`(UB) 使 wl3 从 90.2 退化到 382.5 µs、对 `shared.l0c` 不支持。

## 影响面

- **算子族**：mamba-2 / SSD 族中所有"per-chunk 三角 GEMM + 因果掩码 epilogue"的 kernel——本算子
  `SSDChunkCouplingFwdOp`（4 个 manifest 案例）、`SSDChunkScanFwdOp` 的 intra-chunk `lcb @ x` 段、
  `SSDChunkState` / decay 类同构段。
- **shape 范围**：`Q // BL == 2`（Q=256, BL=128）的几何——即本仓库全部 manifest full 案例（780m / 2p7b /
  grouped-prefill）；`Q // BL != 2`（smoke Q=64、Q=96、Q=128、Q=32）在 `perf_opt` 中回落到 Stage-3 工厂，
  若要同样受益需在 DESIGN 中把行面板推广到 `nlt ∈ {1,3}`（`nlt = 1` 时两者等价）。
- **改设计预期影响的阶段与工作量**：Stage 1 DESIGN（§1.6.3 布局决策表 #4/#5 的采纳结论、§5.5 分核方案、
  §1.6.4 裁决计划的代表形态）+ Stage 3 一次照设计重实现（单个 kernel 工厂 + L0/L1 回归，约 1 个
  kernel 文件、不改 golden / IO 契约 / dtype 路径）。**不改**数学语义（对角 tile 仍为精确 `select`，
  不用 `x·0`）与 IO 契约。

## 反馈结论

DESIGN.md 以「Cube fixpipe 无法施加因果掩码、必须二次遍历清零」与「核内多 tile 串行/persistent 有跨核同步风险、
不摊薄启动开销」为由否决的两个方案，在"按是否需逐元素修改给输出 tile 分类 + 块内多 tile 串行 + 双 L0C 累加器"
的组合下不但可行且**实测快 2.07x**（wl3 173.076 → 82.919 µs，AIV busy 143.3 → 71.1 µs，GM 流量 93.0 → 74.6 MB），
该两条否决依据应被推翻并回写设计。

建议路由: 附录补记
