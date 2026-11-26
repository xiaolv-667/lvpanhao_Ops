# MLADenseBwdOp — Stage 4 调优日志（A5 多缓冲 / v4 旋转后定向优化）

- 算子：`MLADenseBwdOp`（dense BSHD MLA prefill backward；一次调用 3 个 kernel + 2 个框架算子）
- 基线：`examples/TileOPs/tileops/kernels/attention/mla_dense_bwd/mla_dense_bwd_kernel/_bwd_kernel.py`
  （**v4 = A5 rotation**；本目录只读副本 `perf_opt/MLADenseBwdOp_baseline_v4.py`，md5 `e0d6e9a7`）
- 设计：`examples/mla_dense_bwd/_bwd_kernel/DESIGN.md`（v4）
- 平台：`ascend-950pr-9579` / `ascend`，单卡（cube 28 / vector 56，L2 128 MB）；
  工具链 tilelang 0.1.15 / CANN 9.2.1
- 测量口径：**msopprof Task Duration(us)**；`T_op` = 一次调用内全部 launch 求和
  （`Range×2` + `ZerosLike` + `_delta_main` + `_bwd_main` + `_dq_land_main` = 6 个 launch），
  与生产报告 `t_op_us` 同口径。采集器 `perf_opt/collect.py`（`window_ok` 逐次校验 + profiler
  外 warm 编译，见 §3.3）。原始采集 `perf_opt/profiles/`，stdout `perf_opt/logs/`。

## 0. 口径自校验（vs 生产报告 `20261126_040450_369928`）

| workload | 本日志 baseline `T_op` | 生产报告 `t_op_us` | 偏差 |
|---|---:|---:|---:|
| ds-v3-b2-8k-bfloat16 | 98 986.0 µs | 99 091.7 µs | −0.11% |
| ds-v3-4k-bfloat16 | 13 610.3 µs | 13 613.7 µs | −0.03% |
| ds-v3-4k-float16 | 13 616.3 µs | 13 593.8 µs | +0.17% |
| kimi-k2-8k-bfloat16 | 24 848.0 µs | 24 860.5 µs | −0.05% |
| ds-v2-lite-3k-bfloat16 | 3 729.9 µs | 3 747.2 µs | −0.46% |

## 1. Performance Test Data（baseline，`phase=baseline`）

`us/it` = `main_us × 28 / iterations`（iterations 按 DESIGN §1.6.0 R3 复算，b2-8k = 1 420 032）；
cube%/vec% 为 sum 口径（`aic_*_share` / `aiv_*_share`）。**注意**：同一文件两次采集的 pipe 绝对量
有 ±12% 的归一化漂移（§3.4），故 cube%/vec% 只作同 run 内比较，winner 判定一律用 `T_op`。

| workload | T_op (µs) | main (µs) | us/it | AIC: mad / mte1 / mte2 / fix / scal（µs·迭代⁻¹） | AIV: vec / mte2 / mte3 |
|---|---:|---:|---:|---|---|
| ds-v3-b2-8k-bf16 | 98 986.0 | 95 645.2 | 1.8859 | 0.778 / 0.730 / 0.688 / 0.664 / 0.088（和 2.948） | **1.674（88.7%）** / 0.115 / 0.127 |
| ds-v3-4k-bf16 | 13 610.3 | 12 775.8 | 1.9849 | 0.794 / 0.756 / 0.675 / 0.639 / — | 1.716（86.4%） / 0.117 / 0.130 |
| ds-v3-4k-fp16 | 13 616.3 | — | 1.9857 | 同上（±0.5%） | 同上 |
| kimi-k2-8k-bf16 | 24 848.0 | 24 020.0 | 1.8945 | 0.777 / 0.743 / 0.633 / 0.621 / — | 1.686（89.0%） / 0.113 / 0.122 |
| ds-v2-lite-3k-bf16 | 3 729.9 | — | 1.8810 | 0.809 / 0.750 / 0.595 / 0.618 / — | 1.512（80.3%） / 0.111 / 0.116 |

**baseline 现象（本轮起点）**：AIV `vec` 占墙钟 **80–89%**——A5 旋转把 AIC 四管道重叠做到 1.5×
（和 2.95 vs 墙 1.886），代价是 AIV 的 epilogue 变成新墙（与检视判断一致）。AIC 单独跑（§2.3 探针）
的地板是 **1.599 µs/迭代 ⇒ 84.4 ms**，即目标 85 ms 恰在 AIC 地板上、必须让 AIV ≤1.60 才能达成。

smoke/skip 排除：`mla-dense-bwd-smoke-{bfloat16,float16}`（index 0 带 `pytest.mark.smoke`）只做精度回归；
无 skipped 案例。

## 2. Iteration Log

### 2.1 Round 1 — 候选与结果（同一 base 派生，每支只改一个主要点）

| 分支 | 优化点 | 结论 |
|---|---|---|
| **c1** | `st_u/dp_u` 按 half body 各一套（消除 fixpipe(k+1) 对 EP(k) 的 WAR，+24,576 B UB → 60.9%） | **有效 −4.35%**（94 679.8），parity PASS |
| c2 | `p16_u/dsT_u` 按 half body 各一套（+12,288 B） | 无效（98 804，−0.18%） |
| c3 | c1 + c2 | 无效（95 192，−3.83%），未合入 |
| c4 | epilogue 的 UB→L1 改为 4×16 列窄条（命中 `insert_nd2nz.cc` 的 `cols*elem_bytes==32` 快路径，跳过 scatter） | **数值错误 → 作废**（§3.1）；性能假象 88 705（−10.4%） |
| c5 | 在 c4 上去掉 qT/doT 转置填充（自然 (K,N) 操作数） | 作废（承载于 c4）；AIC mte2 0.688→0.476 但墙钟不变 ⇒ 该路径已非瓶颈 |
| c6 | c4 的 8 条窄条按 dsT-first 排序 | 作废（承载于 c4） |
| c7 | c5 + c6 | 作废（承载于 c4） |
| **c11** | **c1 + dV 的 A 操作数独立 L0A 缓冲 `a_p`**（L0A 36,864+12,288+12,288 = 61,440 = 93.75%） | **有效 −4.86%**（94 175.1 / final 重新实测 94 169.7），parity PASS |
| c9 | `dv_c` 直接以 bf16 从 L0C 落到 `out_v`（省掉 tail 的一半 VF） | **编译失败**：`dual_copy: src dtype (float32) != dst dtype (bfloat16). Type conversion during cc->ub dual-destination copy is NOT supported on dav-950` |
| c10 | 每个 serial 迭代只取一次 lse/delta（512 B，供两个 half body） | 未完成（构造脚本锚点失配；预期 ≤0.05 µs/迭代，且末块有越界读风险，主动放弃） |

**候选 vs current best（baseline v4）对比表**（`ds-v3-b2-8k-bfloat16`，平台测量）：

| candidate | T_op (µs) | Δ | us/it | AIC sum（µs·迭代⁻¹） | AIV vec（占比） | L0 / 精度 |
|---|---:|---:|---:|---:|---|---|
| **baseline v4（current best）** | **98 986.0** | — | 1.8859 | 2.948 | 1.674（88.7%） | Stage 5 门禁 |
| c1 st/dp 双版本 | 94 679.8 | −4.35% | 1.8012 | 2.870 | 1.622（90.1%） | parity PASS（dk/dv 逐位） |
| c2 p16/dsT 双版本 | 98 804.4 | −0.18% | 1.8825 | 2.946 | 1.671（88.7%） | 配置项（纯版本化） |
| c3 c1+c2 | 95 191.9 | −3.83% | 1.8112 | 2.882 | 1.632（90.1%） | 配置项 |
| c4 16 列窄条 | 88 705.0 | −10.4% | 1.6834 | 2.829 | 1.268（75.3%） | **PARITY_FAIL → 作废** |
| c5 = c4 + nt | 89 099.6 | −9.99% | 1.6912 | 2.581 | 1.278（75.6%） | 作废 |
| c6 = c4 dsT-first | 88 488.8 | −10.6% | 1.6791 | 2.832 | 1.251（74.5%） | 作废 |
| c7 = c5+c6 | 89 089.8 | −10.0% | 1.6852 | 2.585 | 1.27 | 作废 |
| c8 = c6 + a_p | 87 527.7 | −11.6% | 1.6600 | 2.834 | 1.261（75.9%） | 作废（承载于 c4） |
| **c11 = c1 + a_p** | **94 175.1** | **−4.86%** | 1.7912 | 2.879 | 1.626（90.8%） | **parity PASS** |

**winner / rollback**：c4 家族（c4–c8）给出 −10~−12% 的假象，但 `dk/dv` 与 v4 **不逐位、且数值错误**
（max_abs_diff 4.8）⇒ 整族回滚；真实有效者是 **c1（−4.35%）+ a_p（−0.5%）** ⇒ c11。
c2/c3（输出缓冲双版本）无效：epilogue 的产出缓冲不是瓶颈（WAR 在 `mte3`，代价被 vec 掩盖）。

**mechanism（c1 为什么有效）**：v4 的 `st_u/dp_u` 单版本时，GA(k+1) 的 fixpipe 必须等 EP(k) 的向量读
结束（WAR），这段暴露在 AIC 上（上一轮 Stage 4 已归因 ≈0.42 µs/迭代）。把 pair 按 half body 静态分成
`st_u0/dp_u0`（half A）与 `st_u1/dp_u1`（half B）后该 WAR 消失：AIC fixpipe 0.664→0.588 µs/迭代、
AIV vec 1.674→1.622，墙钟 1.886→1.801。a_p 再把 dV 的 A 操作数从 a_k 里分出来，使它的 L1→L0A copy
不再与 dK/dQ 的 mad 串行：再 −0.5%（mad 0.778 不变，mte1 的串行段缩短）。

### 2.2 Round 2（终止判定前的参照结构 diff，T-1）

| 检查 | 结果 |
|---|---|
| 参数面 | `st_u/dp_u` 与 `p16_u/dsT_u` 的 4 种版本组合、`num_stages`（A5 已不用）、nt 操作数、L0A 复用（a_p）——全部测过；仅 c1 + a_p 有效 |
| AIC 地板 | 探针「epilogue 输出全删 + VF 退化」= **84 444 µs（1.599 µs/迭代）** ⇒ 本 AIC 代码的下限；目标 85 ms 就压在这条地板上 |
| AIV 地板 | 探针「VF 只算 1/6 元素」= 86 239（vec 0.338）；「删 exp/mul/sub」= 86 151（vec 0.992）⇒ AIV 的 vec 主要是**按元素的 loads/stores**（斜率 ≈1.10 µs/满 tile，截距 ≈0.16），exp 只占 24% |
| 同族参照 | `_delta_kernel.py`（同包）用 SIMD 形态做逐元素；但本 epilogue 的 `exp` 是 v4 精度契约的一部分（`dk/dv` 逐位），换 SIMD/`exp2` 折叠会改末位 ⇒ 不可用 |
| 剩余结构候选 | ① 用 16 列窄条绕开 scatter（正确路径需 4 个 (96,16) L1 tile + 梯度 GEMM 拆 K/M，属设计层且 §3.1 的快路径本身不可用）；② L0/L0A 双版本（容量：L0A 2×49,152 > 65,536、L0B 2×61,440 > 65,536）；③ BM↑（换分块 ⇒ dk/dv 不再逐位，违反本轮约束）|

### 2.3 关键诊断（平台测量 + 生成代码）

| 证据 | 数值 | 来源 |
|---|---|---|
| AIC 四管道和 vs 墙钟 | 2.948 vs 1.886 ⇒ **1.56× 重叠**（A5 旋转的收益来源） | baseline `PipeUtilization*.csv` |
| AIV vec 占墙钟 | 88.7%（b2-8k）/ 86.4%（4k）/ 89.0%（kimi） | 同上 |
| AIC-only 地板 | 84 444 µs = 1.599 µs/迭代 | `profiles/round1/ds-v3-b2-8k-bfloat16_probe_no_aiv` |
| VF 元素代价 | 满 tile 1.10 µs，1/6 元素 0.34 ⇒ 斜率 ≈1.10/tile、截距 0.16 | `probe_vf_1of6`、`probe_nomath_c6` |
| 16 列窄条数值错误 | `dk` max_abs_diff 4.805、`cells_over_tol` 512 412；`dq` 2.598 | `logs/precision/parity_c4_smoke.log` |
| L0C→UB 不支持 dtype 转换 | `dual_copy` 直接抛错（dav-950） | `logs/round1/ds-v3-b2-8k-bfloat16_c9.prewarm.log` |

### 2.4 Autotune Log

未使用 autotune（TileOPs 侧 `tune` 策略已移除，tiling 为编译期常量）。参数面以手动分支完成并全部记录在 §2.1。

### 2.5 精度回归（最终版本）

| 检查 | 命令 | 结果 |
|---|---|---|
| 与 v4 逐位（4 例：bf16/fp16 smoke、B=2/S=512↔608 尾块+shift、ds-v3-4k） | `perf_opt/verify_parity.py --cand perf_opt/MLADenseBwdOp.py` | **PARITY_PASS 4/4**：`dk`/`dv` **逐位相同**；`dq` 仅 store-atomic 到达序抖动，0 单元越界 |
| L0（33 例，含 request 隔离 5×2、atomic 契约、arm 等价、rotation drain n_it=1/2/3、B=2 4-D） | `python perf_opt/MLADenseBwdOp.py --level L0` | **33/33 PASS**，max_diff 0.078373（269 s） |
| L1（8 例，含 rotation-vs-v3 两组） | `--level L1` | **8/8 PASS**，max_diff 0.112534（74 s） |
| `big`（`ds-v3-b2-8k` 真形，逐 request×head fp64 切片，8 对） | `--level big` | **PASS**：dk/dv 两次 launch 逐位相同；对同取整参考 **0 元素越界**；request-1 key 行无覆写；kernel-only 2.936 µs/迭代 |

**合并非回退检查（全部 tune workload，final 文件 = c11）**：

| workload | baseline (µs) | final (µs) | Δ | MFU(cube) b→f |
|---|---:|---:|---:|---|
| ds-v3-b2-8k-bf16 | 98 986.0 | 94 169.7 | **−4.87%** | 38.32% → **40.28%** |
| ds-v3-4k-bf16 | 13 610.3 | 13 178.4 | **−3.17%** | 34.84% → 35.98% |
| ds-v3-4k-fp16 | 13 616.3 | 13 173.5 | **−3.25%** | 34.83% → 36.00% |
| kimi-k2-8k-bf16 | 24 848.0 | 23 774.6 | **−4.32%** | 38.16% → 39.89% |
| ds-v2-lite-3k-bf16 | 3 729.9 | 3 705.5 | **−0.66%** | 35.76% → 36.00% |

### 2.6 终止判定

`blocked`：参数/版本面已穷举（10 个分支，仅 c1+a_p 有效 = −4.86%）；剩余候选全部被硬约束挡死——
① 想绕开 NZ scatter 的唯一正确路径需要把梯度 GEMM 的 K/M 拆成 16 列块（改设计，且 §3.1 显示工具链
的快路径本身对列偏移条带不成立）；② L0A/L0B 双版本超容量；③ `BM` 改动会破坏 dk/dv 逐位契约。
目标（≤85 ms / cube ≥45%）**未达**：实测 94.17 ms / 40.28%，差距 1.11×（不是 2× 级）。

## 3. 三条工具链/口径发现（供后续任务复用）

### 3.1 `insert_nd2nz.cc` 的 32 字节快路径对「列偏移条带」不成立（本轮最大教训）

`insert_nd2nz.cc` 在 `cols * elem_bytes == 32`（bf16 ⇒ cols=16）时跳过 UB→UB scatter，直接发
`nd2nz post-copy`（ND→NZ 在 DMA 内完成）。按此把 epilogue 的两条 UB→L1 copy 拆成 4×16 列窄条后：
`nd2nz_scatter` 从生成代码里消失、AIV vec 1.622→1.268、T_op −10.4%——**但结果全错**
（`dk` max_abs_diff 4.805）。根因：post-copy 的 NZ 指针修正只拿到 `(rows, cols, full_rows)`，
**不带列块偏移**，所以 js=16/32/48 的条带被写进第 0 个 NZ 列块。⇒ **该快路径只对「整宽 = 16 列」
的 L1 tile 正确**，条带用法必须避免。（`logs/precision/parity_c4_smoke.log` 为 repro。）

### 3.2 L0C→UB 不支持 dtype 转换（dav-950）

`T.dual_copy(dv_c(float32), out_v(bfloat16))` 被直接拒绝：`Type conversion during cc->ub
dual-destination copy is NOT supported on dav-950`。因此 v4 tail 里「fp32 累加器 → bf16 输出」的
那一趟 VF cast 无法省（tail ≈ 每块 15 360 元素 ≈ 墙钟 2–5%）。

### 3.3 profiler 窗口：冷编译会让一次调用跨成两次

若被 profile 的进程在窗口内做首次 JIT 编译，`--launch-count=6` 实测抓到 **12 个样本**（两次调用，
`window_ok=false`，T_op 翻倍）。`collect.py` 已加 `--prewarm`（窗口外先编译 + 跑一次）：
本轮所有记录 `window_ok=true`。

### 3.4 pipe 占比有 ±12% 的跨 run 归一化漂移

同一文件两次采集（c11 round1 vs final）：`aic_time`/`aiv_time` 完全一致（112.31/114.9 µs），但
`aic_cube_time` 48.79 vs 42.70（−12%）、`aiv_vec_time` 104.29 vs 91.27（−12%）。⇒ **pipe 绝对量
只能同 run 内比较**；winner 判定与目标对照一律用 `T_op`/MFU（本轮所有决策均如此）。

## 4. Final Summary

最终版本：`perf_opt/MLADenseBwdOp.py`（= Round 1 winner c11：`st_u/dp_u` 按 half body 双版本 +
dV 的 A 操作数独立 L0A 缓冲 `a_p`），md5 `dbc75010b909da4468ce272c16e3f827`，
sha256 `75119a74c6263491b01b845b5a113ecb550e4e0e9998fcb449bee41276e12305`，
candidate_id `final-r1-c11`（5 个 tune workload 的 final 记录全部来自该文件）。

### Final Performance Test Data

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---:|---:|---|
| tileops/kernels/attention/mla_dense_bwd/mla_dense_bwd_kernel::_bwd_kernel+_delta_kernel+_dq_land_kernel | ds-v3-b2-8k-bfloat16 | 98986.0 | 94169.7 | final-r1-c11 |
| tileops/kernels/attention/mla_dense_bwd/mla_dense_bwd_kernel::_bwd_kernel+_delta_kernel+_dq_land_kernel | ds-v3-4k-bfloat16 | 13610.3 | 13178.4 | final-r1-c11 |
| tileops/kernels/attention/mla_dense_bwd/mla_dense_bwd_kernel::_bwd_kernel+_delta_kernel+_dq_land_kernel | ds-v3-4k-float16 | 13616.3 | 13173.5 | final-r1-c11 |
| tileops/kernels/attention/mla_dense_bwd/mla_dense_bwd_kernel::_bwd_kernel+_delta_kernel+_dq_land_kernel | kimi-k2-8k-bfloat16 | 24848.0 | 23774.6 | final-r1-c11 |
| tileops/kernels/attention/mla_dense_bwd/mla_dense_bwd_kernel::_bwd_kernel+_delta_kernel+_dq_land_kernel | ds-v2-lite-3k-bfloat16 | 3729.9 | 3705.5 | final-r1-c11 |

| workload | us/it b→f | cube%(同 run) | AIV vec% | 目标 | 判定 |
|---|---|---|---|---|
| ds-v3-b2-8k-bf16 | 1.8859 → **1.7911** | 41.3 → 38.0（跨 run 漂移见 §3.4；round1 同 run 为 43.2→43.4） | 88.7 → 79.4 | ≤85 ms、cube ≥45% | ❌ 94.17 ms / 40.28% MFU |
| ds-v3-4k-bf16 | 1.9849 → 1.9189 | 40.0 → 41.4 | 86.4 → 88.2 | 同比例不回退 | ✅ −3.17% |
| ds-v3-4k-fp16 | 1.9857 → 1.9179 | 40.0 → 41.4 | 86.5 → 88.1 | 同比例不回退 | ✅ −3.25% |
| kimi-k2-8k-bf16 | 1.8945 → 1.8100 | 41.0 → 42.9 | 89.0 → 90.9 | 同比例不回退 | ✅ −4.32% |
| ds-v2-lite-3k-bf16 | 1.8810 → 1.8693 | 43.0 → 43.3 | 80.3 → 83.1 | 同比例不回退 | ✅ −0.66% |

**结论**：五个 tune workload **全部不回退、全部提速 0.7–4.9%**，b2-8k MFU 38.32% → **40.28%**；
但**目标（≤85 ms / cube ≥45%）未达**——94.17 ms，距目标 1.11×。原因是双约束：AIC（L0/L0C 单版本）
地板实测 **84.4 ms**，AIV epilogue 又因 NZ scatter 与逐元素 loads/stores 停留在 1.63 µs/迭代（§2.3），
两者叠加把 85 ms 变成不可能三角；唯一的正确解法是绕开 scatter，而工具链提供的快路径对条带不成立
（§3.1）。差距 1.11× < 2×，按 `perf-feedback.md` §1 的触发门槛**不产出** `perf_feedback.md`
（参数级不足留在迭代内）。

### 中止原因

`blocked`（参数面穷举 + L0 容量/逐位契约/工具链快路径缺陷三重硬约束）。

### 记录规范性说明（`perf_records.jsonl`，append-only）

- 12 行：5 baseline + 2 candidate（c1、c11）+ 5 final。**只记录通过精度门的分支**——
  c2/c3 是纯缓冲版本化配置项、c4–c8 因 §3.1 的数值错误作废、c9 编译失败，均不写性能记录
  （避免把无效候选伪装成 valid；它们的耗时与结论只在 §2.1 表与 `logs/` 中留档）。
- **唯一一次字段规范化（如实披露）**：5 条 `phase=final` 的 `artifact_path` 原为
  `perf_opt/MLADenseBwdOp.py`，gate 4 按 `signal-registry.md §5` 以 `perf_opt/` 为基准解析该字段，
  故规范化为 `MLADenseBwdOp.py`（`write_records.py` 同步修正为直接写相对 `perf_opt/` 的路径）。
  **只改路径前缀**：`duration_us`、`artifact_sha256`、`pipe`、`per_op_us`、`timestamp` 未变；
  改前全文备份 `logs/probes/perf_records.before_path_norm.jsonl` 可逐行 diff。

## 5. Skill Retrospective

| area | issue | evidence | 建议 |
|---|---|---|---|
| 精度前置 | 本轮先测性能、后补 parity，导致 c4–c8 五个分支的「−10~−12%」全部作废（结构性快路径的数值错误只在 parity 才暴露） | §3.1；`logs/precision/parity_c4_smoke.log` | `references/autotune.md` 的候选批处理模板应把 **parity/L0 放在采集之前**作为硬门（本轮顺序反了，白跑 5 次采集） |
| 口径 | pipe 占比跨 run 有 ±12% 归一化漂移（core time 不变、pipe time 整体缩放），用它判 winner 会误判 | §3.4（同文件两次采集） | `profile-collection.md` 补「pipe 绝对量只作同 run 对比；跨 run 用 T_op/MFU」 |
| 迁移陷阱 | `insert_nd2nz.cc` 的 32 B 快路径只在整宽 16 列时成立（列偏移条带会被写进第 0 列块）——**静默错数**，且能带来 10% 的假象提速 | §3.1 | pattern-library 增 TRAP 条目（含 repro 与「整宽 = C0」的边界条件），并在 `tilelang-ascend` 的 API 参考里标注 |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp |
|---|---|---|---|---|
| **A5 旋转后 AIV 的 `st_u/dp_u` 单版本 WAR 是 AIC 的暴露项**：按 half body 静态分成两套 pair ⇒ fixpipe 0.664→0.588 µs/迭代、墙钟 −4.35%（dk/dv 逐位） | D | `perf_records.jsonl` r1-c1 行；`profiles/round1/*_c1.json` | `python perf_opt/collect.py --impl perf_opt/MLADenseBwdOp_round1_c1.py --case ds-v3-b2-8k-bfloat16 --metrics pipe` | tilelang 0.1.15 / CANN 9.2.1 |
| **`insert_nd2nz` 32 B 快路径对列偏移条带静默出错**：4×16 列窄条跳过 scatter 后 vec −0.35 µs/迭代、T_op −10.4%，但 dk max_abs_diff 4.8 ⇒ 快路径只对「整宽 = C0（bf16 16 列）」的 L1 tile 正确 | D（陷阱） | `logs/precision/parity_c4_smoke.log`；`profiles/round1/*_c4.json` | `python perf_opt/verify_parity.py --case mla-dense-bwd-smoke-bfloat16 --cand perf_opt/MLADenseBwdOp_round1_c4.py` | 同上 |
| **dav-950 的 L0C→UB dual copy 不做 dtype 转换**：`dual_copy(f32, bf16)` 直接抛 ValueError ⇒ tail 的 cast VF 不可省 | D（能力缺口） | `logs/round1/ds-v3-b2-8k-bfloat16_c9.prewarm.log` | 同左（编译即复现） | 同上 |
| **AIC-only 地板探针法**：把 epilogue 输出全删 + VF 退化，可分离出「AIC 地板」并与 AIV 占比对照，直接判定目标是否可达（本例 1.599 µs/迭代 ⇒ 84.4 ms） | P | §2.3 探针；`profiles/round1/*_probe_no_aiv.json` | `python perf_opt/collect.py --impl logs/probes/probe_no_aiv.py --case ds-v3-b2-8k-bfloat16` | 同上 |
| **pipe 占比的跨 run 漂移 ±12%**：同文件两次采集 core time 一致而所有 pipe time 等比缩放 ⇒ pipe% 只能同 run 对比 | D | §3.4（两组 CSV 逐列均值） | 对比 `profiles/round1/*_c11/` 与 `profiles/final/*_final/` 的同名 CSV | 同上 |

## 6. 交付与采纳

- 最终版本 `perf_opt/MLADenseBwdOp.py` 与基线 v4 的差异只有三处（可用 `diff` 复核）：
  ① `st_u/dp_u` 各拆成 `_0/_1` 两套并按 half body 选择；② 新增 `a_p` L0A 缓冲并让 dV 用它；
  ③ `_gradients` 宏签名多一个 `a_p` 参数。数学、GEMM 顺序、累加顺序、grid、4-D 视图、契约全部未变。
- 采纳方式（conductor 在回归通过后翻转 wrapper 的双 import 切换块）：本文件可直接作为
  `mla_dense_bwd_kernel/perf_opt/_bwd_kernel.py` 落位（字节相同）；`_delta_kernel`/`_dq_land_kernel` 未改动。
- 保留：`MLADenseBwdOp_baseline_v4.py`（只读基线）、`MLADenseBwdOp_round1_c{1,2,3,4,5,6,7,8,9,11}.py`
  （分支，c4–c8 为已作废的错误分支，保留作 §3.1 的证据）、`logs/probes/`（诊断探针）、
  `collect.py`/`mla_dense_case.py`/`verify_parity.py`/`write_records.py`/`make_candidate.py`/`capture*.sh`。

## 7. 生产采纳与验收（production adoption，2026-11-26）

### 7.1 覆盖 / 备份清单

| 文件 | 变更 | 证据 |
|---|---|---|
| `…/mla_dense_bwd_kernel/perf_opt/_bwd_kernel.py` | **新增**（= 本目录 `MLADenseBwdOp.py`，md5 `dbc75010` / sha256 `75119a74…`，字节相同） | `md5sum` 对比 |
| `…/mla_dense_bwd_kernel/perf_opt/_delta_kernel.py`、`_dq_land_kernel.py` | **新增**（转出 Stage 3/5 实现，未改；md5 `9b178594` / `bd1b54b6`） | 同上 |
| `…/mla_dense_bwd_kernel/perf_opt/__init__.py` | **新增**（drop-in 包导出 + 训练要点说明） | — |
| `…/mla_dense_bwd.py` | **只改切换块 3 条 import**（baseline 注释掉 → perf_opt 启用）；op/manifest/test/bench 未动 | `diff history_version/mla_dense_bwd_pre_perfopt_stage4.py` 仅 87–93 行 |
| `…/mla_dense_bwd_kernel/history_version/mla_dense_bwd_pre_perfopt_stage4.py` | **新增**（切换前 wrapper 备份，md5 `367eeb05`） | — |
| `…/mla_dense_bwd_kernel/_bwd_kernel.py`（v4 基线）、`_delta_kernel.py`、`_dq_land_kernel.py` | **未改动**（perf_opt 是并列目录，基线保留作回退） | `ls -la` 时间戳 11-26 04:00 |

采纳前 `…/mla_dense_bwd_kernel/perf_opt/` **不存在**（无旧 perf_opt 需备份）；wrapper 切换块原指向 baseline。
导入核对：`mla_dense_bwd._bwd_kernel/_delta_kernel/_dq_land_kernel.__module__` 三者均解析到
`…mla_dense_bwd_kernel.perf_opt.*`；工厂签名（含 `n_versions=2`）、grid、4-D 视图、契约全部未变。

### 7.2 生产回归（cwd `examples/TileOPs`）

| 检查 | 命令 | 结果 |
|---|---|---|
| 单测 | `pytest tests/ops/test_mla_dense_bwd.py -q` | **10 passed**（199.5 s） |
| B=2/S=8192 逐位 + 非静默 | `perf_opt/check_prod.py`（生产真形，同一进程内 v4 vs adopted） | **PROD_CHECK_PASS**：`dk`/`dv` **bitwise=yes**（max_abs_diff 0.0）；`dq` 漂移 9.766e-04，与 v4 自身 run-to-run 漂移 9.766e-04 **完全相同** ⇒ 纯 store-atomic 到达序；request 0/1 的 key 0..63 **零元素为 0**；128 个 head 无空行集；全部有限 |
| L0 / L1 / big（本任务 §2.5） | `python perf_opt/MLADenseBwdOp.py --level L0|L1|big` | 33/33、8/8、PASS |

日志：`perf_opt/logs/prod_pytest.log`、`perf_opt/logs/precision/check_prod_b2_8k.log`。

### 7.3 单算子报告（`tileops.reporting.cli run --op MLADenseBwdOp --prof-mode msprof`）

- run-id：**`20261126_062159_108360_MLADenseBwdOp`**（`examples/TileOPs/reports/tileops/<run-id>/`）
- `status = passed`；`correctness_passed = true`（10/10，0 failed）；`benchmark_passed = true`（5/5）；
  `invalid_acceptance_count = 0`、`invalid_profile_count = 0`、`profiler_fallback_count = 0`、
  `kernel_filter_fallback_count = 0`、`pass_rate = 1.0`

| case | 基线生产 `t_op_us` | 采纳后 `t_op_us` | 变化 | MFU(cube 378) 基线 → 采纳后 |
|---|---:|---:|---:|---|
| ds-v3-b2-8k-bfloat16 | 99 091.7 | **94 130.1** | **−5.01%** | 38.32% → **40.30%** |
| ds-v3-4k-bfloat16 | 13 613.7 | **13 158.9** | −3.34% | 34.84% → 36.04% |
| ds-v3-4k-float16 | 13 593.8 | **13 169.0** | −3.12% | 34.83% → 36.01% |
| kimi-k2-8k-bfloat16 | 24 860.5 | **23 776.8** | −4.36% | 38.16% → 39.88% |
| ds-v2-lite-3k-bfloat16 | 3 747.2 | **3 703.0** | −1.18% | 35.76% → 36.02% |

breakdown（b2-8k）：`_bwd_main 90 796.6 / ZerosLike 964.1 / _dq_land 1 647.6 / _delta 717.1 / Range 4.7`。
与 Stage 4 采集（94 169.7）差 0.04%，口径一致；**无回退**（5 个 case 全部提速，MFU +1.2~+2.0 pt）。
`ZerosLike`（`dq_acc` fp32 清零，964 µs）与 `Range` 属算子调用的一部分，仍按原口径计入 `T_op`。

### 7.4 记录与清理

- `perf_records.jsonl` 追加 5 行 `phase=merged`（`round=3`、`candidate_id=prod-perf_opt-c11`、
  `parent_id=final-r1-c11`，含 run-id、report 路径、`invalid_*` 计数、`mfu_cube378`）；现有 17 行
  （5 baseline + 2 candidate + 5 final + 5 merged），append-only 未改动既有行。
- `/tmp/tileops_msprof_*` 已清理；`/workspace/TileOPs-ascend950` 未触碰。
