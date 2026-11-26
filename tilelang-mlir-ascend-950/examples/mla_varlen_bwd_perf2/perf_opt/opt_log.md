# MLAVarlenBwdOp — Stage 4 补轮（round 2：request 长度分层残差）

- 任务目录：`examples/mla_varlen_bwd_perf2`；基线（只读）：`examples/mla_varlen_bwd_perf/perf_opt/MLAVarlenBwdOp.py`
  （round-1 winner `_STAGES=3`，本目录副本 md5 `6e2e6935…` / sha256 `4e8fcb7d…`，字节相同）
- 设计：`examples/mla_varlen_bwd/_bwd_kernel/DESIGN.md`（v4 Cube）；平台 `ascend-950pr-9579` / `ascend`，单卡
- 目标：mixed **563.7 → ≤500 ms**（−11.3%），8x4k/kimi 不回退
- 测量：`msopprof` kernel `Task Duration(us)`，`T_op` = 一次调用全部 launch 求和；采集器 `perf_opt/collect.py`
  （本轮修正了 launch 窗口口径，见 §1.2）。原始采集 `perf_opt/profiles/`，stdout `perf_opt/logs/`。

## 0. 结论（先行）

**未达成**：mixed 维持在 **563.8 ms**（目标 500 ms，差 11.3%），8x4k 168.5 ms、kimi 159.1 ms（均未回退）。
本轮 6 个候选分支（2 个 query 填充粒度重构 + 3 个 head-slab 档 + 2 个诊断探针）**全部无效或更差**，
并逐项**证伪**了长度分层的候选机制（§2.3）。交付版本 = round-1 winner 原样（无回退），
残差按 `[DESIGN_LIMIT]` 上报（`perf_opt/perf_feedback.md`），附「必须改设计」的具体结构建议。

## 1. Performance Test Data

### 1.1 baseline（= 本轮起点 = 交付版本）

| workload | T_op (µs) | main (µs) | iterations | us/it | cube%（mad/mte1/mte2/fix/scal） | vec%（vec/mte2/mte3） | raw |
|---|---:|---:|---:|---:|---|---|---|
| ds-v3-8x4k-bf16 | 168 539.3 | 161 889.2 | 1 441 792 | 3.144 | 25.6/21.6/54.2/17.9/4.2 | 45.2/15.9/4.3 | `profiles/round0/ds-v3-8x4k-bfloat16_baseline` |
| ds-v3-8x4k-fp16 | 168 445.0 | 161 801.3 | 1 441 792 | 3.142 | 25.6/21.5/53.6/17.9/4.2 | 45.2/15.8/4.3 | `profiles/round0/ds-v3-8x4k-float16_baseline` |
| ds-v3-mixed | 563 742.0 | 557 202.2 | 3 770 496 | 4.138 | 26.3/22.4/56.1/18.7/3.8 | 44.8/17.9/4.1 | `profiles/round0/ds-v3-mixed-bfloat16_baseline` |
| kimi-k2-4x8k | 158 249.0 | 154 891.7 | 1 420 032 | 3.054 | 26.1/22.5/64.3/17.9/3.8 | 44.8/21.9/4.3 | `profiles/round0/kimi-k2-4x8k-bfloat16_baseline` |

smoke（`mla-varlen-bwd-smoke-{bf16,fp16}`）仅做精度回归，不采性能（benchmark index 0 带 `pytest.mark.smoke`）。
无 skipped 案例。与上一轮 final（168 461 / 168 520 / 563 778 / 159 077）逐项差 <0.6%（同一 sha256 产物）。

### 1.2 采集口径修正（本轮发现，已修）

上一轮的采集器用**基线**的 `plan_launch` 推算「一次调用含几次 launch」，用它算 warm-up/launch-count 窗口。
本轮把 `head_slab` 改成 8 后，一次调用是 19 次 launch，而窗口仍按 7 次算 → 抓到了**跨两次调用的片段**
（mixed 被读成 176 ms，比真值小 3.2×）。已改为：

1. 先跑一次**不加 profiler** 的同 case 调用预热编译缓存；
2. profiled 进程用 `--no-setup` 只发**恰好一次调用**，profiler `--warm-up=0 --launch-count=K`；
3. `K` 由**被测 impl 自己的** `plan_launch` 推出，`window_ok = (captured == K)` 逐次校验。

修正后 baseline 复现 563 971 µs（原 563 742，差 0.04%），s8 真值为 559 485 µs（此前误读 176 064）。
**教训**：多 launch 算子的采集窗口必须取自被测产物的 launch 计划，不能用固定常量。

## 2. Iteration Log

### 2.1 Round 1 — 现象复盘与候选点

上一轮已定位（同构对照，stages=3）：len≤5376 为 3.11–3.14 µs/迭代，len=16384 为 4.26；mixed（16384 主导）
4.14。本轮先做了一次**绝对量核算**（把 pipe 忙时除以迭代数，而不是看 share），得到关键事实：

| case | 每迭代 wall | AIC 忙时（aic_time） | AIC 五管道忙时之和 | AIV 忙时（每个 AIV） | AIV 四管道之和（每 AIV） |
|---|---:|---:|---:|---:|---:|
| 8x4k | **3.144** | 2.956 | 3.643 | 3.04 | 2.05 |
| 6×5376 | **3.108** | 3.006 | 3.797 | 3.00 | 2.10 |
| mixed | **4.138** | 2.905 | 3.661 | 2.90 | 2.00 |
| 1×16384 | **4.264** | 2.842 | 3.562 | 2.86 | 1.95 |

⇒ 四个 case 的**每迭代管道工作量完全一致**（AIC 3.56–3.80、AIV 1.95–2.10），
差别是 slow case 的每迭代 wall 比 AIC/AIV 忙时多出 **~1.3 µs** 的「双引擎同时空闲」时间。
即残差**不是多做的工作，而是块内的空转**（`Σ aic_time/28 / wall` = 0.94 / 0.97 / 0.70 / 0.67）。

### 2.2 Round 1 实验：query 填充粒度（用户建议 #1 的实现层版本）

假设：长请求每个 key block 都要重读 q/do 后缀，每次填充付一次 DRAM 往返（~1.1 µs）；
把填充粒度从 1 个 64 行查询块提到 2 个（一次 descriptor 覆盖 128 行），往返次数/查询行减半。

| branch | 结构 | 8x4k T_op | mixed T_op | 结论 |
|---|---|---:|---:|---|
| baseline | 1×64 行/填充, stages=3 | 168 539.3 | 563 742.0 | — |
| q2s2 | 2×64 行/填充, stages=2 | 249 776.2 | 845 089.3 | ❌ −48% |
| q2s3nt | 2×64 行/填充 + 自然 (K,N) 操作数, stages=3 | 217 160.5 | 726 963.2 | ❌ −29% |

机制：填充变大后**一个 chunk 内的两个子步共享单版本 L0/UB**（`a_k/b_lo/b_hi/pT_s/dsT_s/st_c/dpt_c`），
两步只能串行，叠加后 mad share 26.3%→17.5%、mte2 share 56%→20%，流水彻底塌掉。
⇒ **本结构无法用「更大填充」换更少往返**（L0C 84%/L0B 94% 已无第二份版本的空间）。

### 2.3 Round 2 实验：in-flight head footprint（用户建议 #2 的 launch 参数版本）+ 机制证伪

假设（上一轮遗留）：L2 工作集 = 在飞 head 数 × request 长度 × 640 B；len>6250 即超过 128 MB L2，
导致填充落到 DRAM。若成立，缩小 `HEAD_SLAB_DEFAULT` 应显著有效。

| branch | HEAD_SLAB | mixed T_op | us/it | 8x4k T_op | 结论 |
|---|---|---:|---:|---:|---|
| baseline | 32（在飞约 28 head × len → mixed 336 MB） | 563 742.0 | 4.138 | 168 539.3 | — |
| s8 | 8（同 head 的 3.5 个 tile 并行 → 约 84 MB） | 559 485.5 | 4.106 | — | ❌ −0.8%（噪声） |
| s4 | 4（约 42 MB） | 558 620.8 | 4.100 | 168 473.5 | ❌ −0.9%（噪声） |

⇒ **L2 容量假设被证伪**：工作集从 336 MB 降到 42 MB（4–8×）没有任何收益；
同时 `aic_mte2_share` 从 56.1% 降到 28.1%（DRAM 压力确实下来了），但 wall 不动 ⇒ 瓶颈不在访存。

### 2.4 Round 2 诊断探针（不进 perf_records；语义故意不成立，只测机制）

| probe | 改法 | mixed T_op | 说明 |
|---|---|---:|---|
| `probe_noatomic` | 去掉 `dq` 的 store-atomic，改普通 store（结果必然错） | **616 825.9** | 比原子版更慢 ⇒ **store-atomic 不是瓶颈**（L2 原子写不是热点） |
| `probe_masked_only` | `a2_fast=False`，循环内只剩一套（带掩码）epilogue，无分支 | 704 342.4 | 慢 25% ⇒ 循环内 `if fast:` 分支**不是**流水障碍；掩码本身更贵 |
| 计数器自洽性 | `aic_total_cycles / aic_time` | 恒 1.650e3（含 752 µs 长块） | 排除「profiler 对长块计数截断」的伪影 |

### 2.5 候选 vs current best（本轮汇总，平台时延）

| candidate | 改动 | mixed (µs) | Δ | 8x4k (µs) | Δ | dk/dv 逐位 | dq_acc | 判定 |
|---|---|---:|---:|---:|---:|---|---|---|
| **baseline（= final）** | round-1 winner, `_STAGES=3` | **563 742.0** | — | **168 539.3** | — | — | — | current best |
| r2-q2s2 | 填充粒度 2×64 行, stages=2 | 845 089.3 | +49.9% | 249 776.2 | +48.2% | 逐位 | 1e-3 内 | 回退，丢弃 |
| r2-q2s3nt | 同上 + 自然 (K,N) 操作数, stages=3 | 726 963.2 | +29.0% | 217 160.5 | +28.8% | 逐位 | 1e-3 内 | 回退，丢弃 |
| r2-s8 | `HEAD_SLAB_DEFAULT` 32→8 | 559 485.5 | −0.8% | — | — | 逐位 | 1e-3 内 | 噪声内，合入无意义 |
| r2-s4 | `HEAD_SLAB_DEFAULT` 32→4 | 558 620.8 | −0.9% | 168 473.5 | −0.04% | 逐位 | 1e-3 内 | 噪声内（且 launch 数 4→32），丢弃 |
| （诊断）noatomic | 去原子（结果错） | 616 825.9 | +9.4% | — | — | — | 错 | 仅机制定位 |
| （诊断）masked_only | 单一掩码 epilogue | 704 342.4 | +24.9% | 204 634.2 | +21.4% | 逐位 | 1e-3 内 | 仅机制定位 |

精度：所有候选 `dk/dv` 与 Stage 3 基线**逐位相同**、`dq_acc` 在 §8.2 容差内（`perf_opt/logs/precision/parity_*.log`）；
因此「更慢」是结构性的，不是精度/语义修正的代价。

### 2.6 本轮排除/确认的机制清单（长度分层残差）

| 假设 | 实验 | 结果 |
|---|---|---|
| 派发/长尾不均 | 逐核 `Σ aic_time` 汇总（上一轮） | 否（max/mean ≤1.007） |
| L2 工作集容量 | slab 4/8/16/32 剂量-反应 | **否**（336→42 MB 无变化） |
| 每次填充的 DRAM 往返次数 | 填充粒度 ×2（q2s2/q2s3nt） | 否（更差，因 L0 单版本串行） |
| store-atomic 写 | noatomic 探针 | 否（更快的是原子版） |
| 循环内 `if fast:` 分支阻断流水 | masked_only 探针 | 否（更差） |
| profiler 长块计数截断 | cycles/time = 1.65 GHz 恒定 | 否 |
| 预取深度 | stages 1/2/3/4/5（上一轮 + 本轮） | 3 为拐点，4/5 更差 |
| 填充字节/描述符数 | 自然 (K,N) 操作数（上一轮 c1/c5） | 否（mte2 忙时 −1%） |
| **块内 AIC↔AIV 串行（单版本 UB epilogue 缓冲）** | 每迭代 wall − max(AIC,AIV) 忙时 = 1.3 µs，两引擎同时空闲 | **唯一与全部数据一致的剩余解释**（未直接实现验证：需双版本 UB epilogue，见 §3） |

## 3. 为什么本轮不继续改（结构边界）

要让这 1.3 µs/迭代消失，必须让**下一迭代的 GEMM 与上一迭代的 AIV epilogue 重叠**。
当前 epilogue 的 UB 缓冲（`st_u/dp_u/p16_u/dsT_u`）与 L1 缓冲（`pT_s/dsT_s`）都是**单版本**，
所以循环只有一条硬串行链：`GEMMs(k) → dual_copy L0C→UB(k) → AIV epilogue(k) → UB→L1(k) → GEMMs(k+1)`。
可选做法与代价（均为设计层）：

1. **UB epilogue 双版本**：再加一套 `st_u/dp_u/p16_u/dsT_u`（+36 864 B，UB 129→166 KiB，可行）
   与 `pT_s/dsT_s`（+24 576 B，L1 205→229 KiB，可行），并在 kernel 里手工交替（`kq % 2` 两份 body）
   —— 代码体翻倍（源 ~30 KB → ~60 KB），且依赖编译器对 `dual_copy`/`SimtVF` 的版本识别，
   属 DESIGN §3.3/§6.3 的循环与多缓冲决策，需设计确认后用一轮实现 + A/B 验证。
   DESIGN 自己给出的完全重叠下界是 **1.47 µs/迭代（mixed 198 ms）**，即该方向的结构收益上界 **2.85×**。
2. **两遍 query 切分**（按 8192 拆 request）：需要 `dk/dv` 跨块累加（当前是赋值语义）⇒ 契约变更。
3. **更大 tile**（减迭代数）：L0C 已 84%（96×64 是上限），block_n/block_m 由 plan 断言冻结。

## 4. Final Summary

最终版本：`perf_opt/MLAVarlenBwdOp.py`（= round-1 winner，`_STAGES=3`，sha256 `4e8fcb7d…`，
与上一轮生产采纳文件字节相同；本轮**没有产生可合入的改进**，故未改动生产集成包与 wrapper）。

### Final Performance Test Data

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---:|---:|---|
| tileops/kernels/attention/mla_varlen_bwd/mla_varlen_bwd_kernel::_bwd_kernel+_delta_kernel+_dq_land_kernel | ds-v3-8x4k-bfloat16 | 168539.300 | 168461.500 | final-r1-winner |
| tileops/kernels/attention/mla_varlen_bwd/mla_varlen_bwd_kernel::_bwd_kernel+_delta_kernel+_dq_land_kernel | ds-v3-8x4k-float16 | 168445.000 | 168520.200 | final-r1-winner |
| tileops/kernels/attention/mla_varlen_bwd/mla_varlen_bwd_kernel::_bwd_kernel+_delta_kernel+_dq_land_kernel | ds-v3-mixed-bfloat16 | 563742.000 | 563778.200 | final-r1-winner |
| tileops/kernels/attention/mla_varlen_bwd/mla_varlen_bwd_kernel::_bwd_kernel+_delta_kernel+_dq_land_kernel | kimi-k2-4x8k-bfloat16 | 158249.000 | 159077.200 | final-r1-winner |

回退判定：四 workload 与 baseline 差 <0.6%（mixed +0.006%），**无回退**；mixed 未达 500 ms 目标（563.8，差 11.3%）。

### 长度分层的改善量

**0**：len≤5376 档 3.14 µs/迭代、len=16384 档 4.14–4.26 µs/迭代，与本轮起点相同（本轮所有分支都更差或打平）。

### 精度（本轮在交付文件上复跑）

| 检查 | 命令 | 结果 |
|---|---|---|
| L0（28 例） | `python perf_opt/MLAVarlenBwdOp.py --level L0` | **28/28 PASS**（max_diff 0.127985，`logs/precision/L0_final.log`） |
| L1（8 例） | `--level L1` | **8/8 PASS**（0.113342，`L1_final.log`） |
| 逐位（vs Stage 3，4 例：h32 / mixed-h16 / ragged + 本轮 slab 变体） | `verify_parity.py` | **dk/dv 逐位相同**，dq_acc 仅原子抖动（`parity_*.log`） |
| 8x4k 逐 request 非静默（H=128 真形） | `check_nonsilent.py` | **NONSILENT_PASS**（request 0..7 全覆盖，尾 slab head 96/112/126/127 非零，plan idle=0） |
| mixed 逐 request | L0-19 + G2（上一轮，同一 sha256）+ parity mixed-h16 | PASS |

## 5. Skill Retrospective

| area | issue | evidence | 建议 |
|---|---|---|---|
| 测量口径 | 多 launch 算子的 profiler 窗口必须由**被测产物**的 launch 计划推出；用固定常量/基线计划会在候选改 slab 时静默抓错窗口（本次把 559 ms 读成 176 ms，若不是数值异常几乎会误判为巨大收益） | §1.2；`perf_opt/collect.py` 修正前后 | profile-collection.md 增一条「窗口 K 取自候选自身 `plan_launch`，并要求 `captured == K`；候选改 slab/stage 时尤其要核对」 |
| 诊断方法 | 优化前先把 pipe 忙时**除以迭代数**做绝对量核算（而不是看 share/ratio 均值），能一眼把「多做工作」与「空转」分开：本轮据此把残差从「访存」改判为「块内双引擎同时空闲」 | §2.1 表；上一轮因只看 share 而误判为 mte2/填充问题 | iteration-diagnosis.md 增「每迭代绝对忙时核算」步骤 |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp |
|---|---|---|---|---|
| **Cube+Vector 混合 kernel 的「每迭代绝对忙时核算」**：把 `Σ pipe_time / iterations` 与每迭代 wall 并列，四个 case 的 AIC(3.56–3.80)/AIV(1.95–2.10) 工作量完全一致，而 wall 3.14 vs 4.26 ⇒ 残差是块内空转；单看 share 会误判（share 几乎相同） | D | `perf_records.jsonl`（round0/final 行含 pipe 逐管道时间与 iterations）；本日志 §2.1 | `python perf_opt/collect.py --impl perf_opt/MLAVarlenBwdOp.py --case ds-v3-mixed-bfloat16` 后读 JSON 的 `pipe`+`iterations` | tilelang 0.1.15 / Ascend950PR_9579 / CANN 9.2.1 |
| **packed 长请求的残差不是访存**：L2 工作集 336→42 MB（slab 32→4）无变化（563.7/558.6），填充粒度 ×2 更差（845），去 store-atomic 更慢（617）——四个方向共同证伪「访存/原子/填充」类解释 | D | `perf_records.jsonl`（r2-s4/s8、r2-q2s2/q2s3nt 行）；本日志 §2.3/§2.4 | 同上 + `--impl perf_opt/MLAVarlenBwdOp_r2_s4.py` | 同上 |
| **多 launch 算子 profiler 窗口口径**：`--warm-up/--launch-count` 数的是 kernel launch；窗口必须 = 被测产物一次调用的 launch 数，且用「预热进程 + 单调用进程（`--no-setup`）」才稳定 | P | §1.2（误读 176 vs 真值 559）；`perf_opt/collect.py` | `python perf_opt/collect.py --impl perf_opt/MLAVarlenBwdOp_r2_s8.py --case ds-v3-mixed-bfloat16`（K=19，window_ok=true） | 同上 |

## 6. 结论

本轮 6 个分支 + 3 个诊断探针把「长度分层」残差从「访存嫌疑」收敛为「块内 AIC↔AIV 串行」，
但没有在冻结结构内找到可行改法（唯一可行方向是 UB epilogue 双版本 / 两遍 query 切分，均属设计层，
见 §3）。交付版本与 round-1 生产采纳版本字节相同，四 workload 无回退，精度全绿；
mixed 500 ms 目标未达成，已按 `[DESIGN_LIMIT]` 上报并给出结构建议与收益上界（2.85×，来自 DESIGN 自述的
1.47 µs/迭代完全重叠下界）。
