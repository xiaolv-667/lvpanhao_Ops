# 性能反馈（[DESIGN_LIMIT]）— 第 2 轮：packed 长请求残差

## 触发判定

- **设计层归因**：mixed 563.8 ms 与 500 ms 目标的差额（11.3%）来自 **Cube+Vector 混合块内的
  AIC↔AIV 串行链**，其根因是 DESIGN.md §3.3/§6.3 的多缓冲决策：epilogue 的 UB 缓冲
  （`st_u/dp_u/p16_u/dsT_u`）与 L1 中转（`pT_s/dsT_s`）都是**单版本**，于是循环只有一条硬串行链
  `GEMMs(k) → L0C→UB(k) → AIV epilogue(k) → UB→L1(k) → GEMMs(k+1)`，
  §1.6.0 D 表「AIV epilogue（与之并行）~1.23 µs」这一并行假设在长请求上不成立。
  已证明非参数可解：本轮把参数/结构旋钮逐个实测排除（§实测 B/C/D），
  `block_n/block_m` 又由 L0C 容量（84%）与 `plan_launch` 断言冻结。
- **实测与设计假设矛盾 + 结构性加速 >2×**：
  ① 矛盾：§1.6.0 D 表假设 AIC 四管道忙时之和 ≈ 迭代（2.74 vs 2.97，单 AIC 内无重叠）且 AIV epilogue
     与之并行；实测每迭代 wall 4.138 µs，而 AIC 忙时 2.905、AIV 忙时 2.90（每个 AIV），
     **两个引擎各有 ~1.2 µs 同时空闲**，即 epilogue 并未与 GEMM 重叠。
  ② 结构收益上界：DESIGN §1.6.0 自述的**完全重叠下界 1.47 µs/迭代（mixed 198 ms）**
     ⇒ 相对当前 4.138 µs/迭代（563.8 ms）为 **2.85×**（> 2×），依据是设计文档自己的管道核算 + 本轮实测。

## 参照锚定

- **参照实现**：同族最近参照 `examples/mla_dense_bwd/_bwd_kernel`（同一 5-GEMM 显式 L0 结构、同一 epilogue），
  其生产实测 2.97 µs/迭代（S=2048、H=8）；本 kernel 在 len≤5376 已达 **3.11–3.14 µs/迭代**，
  即**与参照同量级**，残差只出现在参照从未覆盖的 len≥8192 区间（dense 参照的 S 上限 8192）。
  平台知识库（`kb_search`）无「packed 长请求单价」条目；`knowledge/bottlenecks.md` 明确尚无已验证的
  950 瓶颈模式。⇒ **未找到在 len=16384 packed 区间实测更优的同族参照，本反馈的天花板结论置信度降级**
  （只对「本冻结结构 + 本工具链」成立）。
- **不可移植论证**：
  1. dense 参照的静态 extent 形态在 len≥8192 未测过，不能证明该区间存在更优的冻结结构；
  2. 与参照唯一的实质差异是 packed 的「按 request 切 suffix 的循环」——而本轮同构对照证明
     差异来自**块内串行**而非 request 切分方式（同构 6×5376 快、同构 1×16384 慢）；
  3. DESIGN §1.6.0 R3/R4 淘汰 query-stationary / 两 kernel 拆分 / 物化的理由仍是**设计期推算**
     （本轮未复测），按未实证假设记录，不用作否决依据。

## 假设

`DESIGN.md` §1.6.0 D 表（设计期 roofline，per-iteration）：

> AIC 四管道忙时之和 2.74 µs ≈ 迭代 2.97 µs（**单 AIC 内无重叠**）；
> `AIV epilogue（与之并行）~1.23`；
> 完全重叠下界 `max(mad 0.80 + MTE1 0.67, MTE2 0.70, AIV 1.23) = 1.47 µs`（mixed 198 ms，cube 54%）。

以及 §3.3/§6.3：查询块循环 `num_stages=2`（本轮已提到 3），epilogue 缓冲单版本。

## 实测

口径：任务平台 `msopprof` kernel `Task Duration(us)`，`T_op` = 一次调用全部 launch 求和；
原始采集 `examples/mla_varlen_bwd_perf2/perf_opt/profiles/`，逐行 `perf_records.jsonl`。

**A. 每迭代绝对忙时（分母为公式复算的 iterations，而不是 share）**

| case | wall/iter | AIC 忙时 | AIC 五管道和 | AIV 忙时/每个 AIV | AIV 四管道和/每个 AIV |
|---|---:|---:|---:|---:|---:|
| 8x4k | 3.144 | 2.956 | 3.643 | 3.04 | 2.05 |
| 6×5376（同构） | 3.108 | 3.006 | 3.797 | 3.00 | 2.10 |
| mixed | 4.138 | 2.905 | 3.661 | 2.90 | 2.00 |
| 1×16384（同构） | 4.264 | 2.842 | 3.562 | 2.86 | 1.95 |

⇒ 工作量相同、wall 差 1.0–1.3 µs，且 AIC/AIV **同时**空闲（`Σ aic_time/28 / wall` = 0.94/0.97/0.70/0.67）。

**B. 访存类解释全部证伪**

| 实验 | 改法 | mixed `T_op` | 结论 |
|---|---|---:|---|
| slab 剂量-反应 | `HEAD_SLAB_DEFAULT` 32→8→4（L2 工作集 336→84→42 MB） | 563 742 → 559 485 → 558 621 | L2 容量无关（`aic_mte2_share` 56%→28%，wall 不变） |
| 填充粒度 | 一次填充 2×64 行（往返次数减半） | 845 089（stages=2）/ 726 963（stages=3+自然操作数） | 更差：L0/UB 单版本被两个子步串行化 |
| store-atomic | 去原子（普通 store，结果故意错） | 616 826 | 原子写不是瓶颈（更慢） |
| 循环内分支 | `a2_fast=False` 单一掩码 epilogue | 704 342 | 分支不是障碍（掩码更贵） |
| 预取深度 | `num_stages` 1/2/3/4/5 | 3 最优；4 = 574 108、5 = 593 154（第 1 轮） | 深度已到拐点 |
| 填充字节/描述符 | 自然 (K,N) 操作数（字节与描述符各减半） | 644 264（第 1 轮 c1） | mte2 忙时只 −1%（延迟受限） |

**C. 计数器自洽**：`aic_total_cycles / aic_time` 恒 1.650e3（含 752 µs 长块）⇒ 排除 profiler 对长块截断。

**D. 唯一与全部数据一致的剩余解释**：块内 AIC↔AIV 串行（单版本 UB epilogue）。
直接验证需要一个 UB epilogue 双版本的实现（本轮未做：代码体翻倍、属设计决策，见「影响面」）。

## 影响面

- 受影响算子族：所有「Cube GEMM + 每迭代一次 Vector epilogue」的 950 混合 kernel
  （MLA bwd varlen / dense、flash-attention 反向的 NPU 实现），以及任何把「epilogue 与 GEMM 并行」
  当作既成事实的设计投影。
- 受影响 shape：迭代数大的块（本算子 len≥8192；等价地 iters/block ≥ 64）。
  短请求（len≤5376）不受影响（已达 dense 参照同量级）。
- 改设计预期影响：Stage 1 的 §3.3 循环结构 / §6.3 多缓冲（UB epilogue 双版本：UB 129→166 KiB、
  L1 205→229 KiB，均有余量）+ Stage 3 的 kernel 重写（源 ~30 KB → ~60 KB，两份 body 交替）；
  收益上界 2.85×（设计自述 1.47 µs/迭代），达成后 mixed ≤198 ms、8x4k ≤70 ms。
- 需同步放弃的契约：若改用「两遍 query 切分」，需把 `dk/dv` 由赋值改为累加（原子），属算子契约变更。

## 反馈结论

mixed 的剩余 11.3% 缺口不是 tiling/参数可解（本轮 6 分支 + 3 探针全部证伪访存/原子/填充/派发类解释），
而是 DESIGN §3.3/§6.3 的**单版本 epilogue 缓冲导致的块内 AIC↔AIV 串行**；按 DESIGN 自述的完全重叠下界
（1.47 µs/迭代）该结构改造的收益上界为 **2.85×**。Stage 4 已交出冻结结构内最优版本
（mixed 563.8 ms / 8x4k 168.5 ms / kimi 159.1 ms，精度逐位不变，无回退）。

建议路由: 附录补记
