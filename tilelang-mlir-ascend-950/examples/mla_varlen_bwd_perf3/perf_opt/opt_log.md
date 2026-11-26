# MLAVarlenBwdOp — Stage 4（A5 / v5 之上，round 3）

- 任务目录 `examples/mla_varlen_bwd_perf3`；基线（只读）`examples/TileOPs/…/mla_varlen_bwd_kernel/_bwd_kernel.py`
  （A5 v5，本目录副本 `MLAVarlenBwdOp_baseline_a5v5.py`，md5 `4df7899b…`）
- 设计 `examples/mla_varlen_bwd/_bwd_kernel/DESIGN.md`（v5）；平台 `ascend-950pr-9579` / `ascend`，单卡
- 目标：mixed **≤400 ms**、8x4k **≤130 ms**、kimi 不回退；cube ≥35%；保持 A5/packed/grid、dk/dv 逐位
- 测量：`msopprof` kernel `Task Duration(us)`，`T_op` = 一次调用全部 launch 求和（`perf_opt/collect.py`，
  窗口 = 被测产物自身 `plan_launch` 的 launch 数，`window_ok` 逐次校验）。**每轮先 parity 再测性能**。

## 0. 结论

| workload | baseline (v5) | best | Δ | 目标 | 判定 |
|---|---:|---:|---:|---|---|
| ds-v3-mixed | 441 652.0 µs | **402 397.5 µs** | **−8.9%** | ≤400 000 | ⚠️ 差 0.6%（两次独立采集 401 892 / 402 398，落在噪声内） |
| ds-v3-8x4k-bf16 | 138 964.2 | **135 425.4** | −2.5% | ≤130 000 | ❌ 差 4.2% |
| ds-v3-8x4k-fp16 | 139 310.2 | **135 384.4** | −2.8% | ≤130 000 | ❌ 差 4.1% |
| kimi-k2-4x8k | 139 036.5 | **138 916.3** | −0.1% | 不回退 | ✅ |

cube（sum 口径 aic 管道/核时）：32.5/31.4/31.4% → **34.7/32.2/31.4%**（目标 ≥35%：mixed 34.7 接近，8x4k 32.2 未达）。
MTP2/AIV：`aic_mte2` 81.5→**89.7%**（mixed）、85.2→**88.2%**（8x4k）、`94.0%`（kimi）——填充管道已接近饱和；
`aiv_vec` 80.3→83.3%（mixed）。

有效产出：**mixed −8.9%（441.7→402.4 ms）、8x4k −2.5%、kimi 不回退**，dk/dv 逐位不变；但 8x4k 的 ≤130 未达，
mixed 的 ≤400 只在噪声意义内达成。原因见 §3（L0B 93.75% / L0C 84% 两道容量墙使「每迭代填充字节」不可再降，
而 `aic_mte2` 已到 88–94%）。

## 1. Performance Test Data（baseline，A5 v5）

| workload | T_op (µs) | main (µs) | iterations | us/it | cube%（mad/mte1/mte2/fix/scal） | AIV%（vec/mte2/mte3） | MTE2 BW | raw |
|---|---:|---:|---:|---:|---|---|---|---:|---|
| ds-v3-mixed | 441 652.0 | 435 102.1 | 3 770 496 | 3.231 | 32.5/28.7/81.5/42.0/5.8 | 80.3/33.9/5.0 | 54.4 GB/s | `profiles/round0/ds-v3-mixed-bfloat16_baseline` |
| ds-v3-8x4k-bf16 | 138 964.2 | 132 306.4 | 1 441 792 | 2.569 | 31.4/27.8/85.2/37.5/6.2 | 77.2/33.2/5.2 | 56.1 | `…/ds-v3-8x4k-bfloat16_baseline` |
| ds-v3-8x4k-fp16 | 139 310.2 | — | 1 441 792 | 2.576 | 同上 | 同上 | — | `…/ds-v3-8x4k-float16_baseline` |
| kimi-k2-4x8k | 139 036.5 | — | 1 420 032 | 2.675 | 31.4/26.4/94.0/42.8/5.8 | 81.4/52.5/5.0 | 50.1 | `…/kimi-k2-4x8k-bfloat16_baseline` |

与生产报告 `20261126_025737_402618` 一致（441.7 / 138.4 / 140.3；本目录复现 441.65 / 138.96 / 139.04，差 <0.5%）。
smoke 两个 case 仅做精度回归；无 skipped。

## 2. Iteration Log

### 2.1 Round 1 — 现象与候选点

A5 v5 的每迭代瓶颈已从「块内串行」转为**填充管道**：`aic_mte2` 81–94%（第一忙管道），`aiv_vec` 77–81%。
每迭代 wall 3.231（mixed）/2.569（8x4k），AIC 五管道和 6.2/5.6 µs（>180% → 已在重叠），
因此本轮候选点全部针对「减少串行残留 + 减少/提前填充」：

| 分支 | 改动 | 预期 |
|---|---|---|
| `ubv2` | `st_u/dp_u` 按半体双版本（UB +24,576 B） | 去掉 `_scores(B)` 对 `_epilogue(A)` 的 WAR |
| `ubv2b` | 再双版本 `p16_u/dsT_u`（UB +12,288 B） | 去掉 epilogue→dual_copy 的 WAR |
| `nt` | dV/dK/dQ 用自然 (K,N) 操作数，去 qT/doT/kT | 填充 6→4 描述符、−50% 字节（`aic_mte2` 是主瓶颈） |
| `ubv2_nt` | 上两者组合 | — |

**候选 vs current best（Round 1）**

| candidate | mixed (µs) | Δ | us/it | 8x4k (µs) | Δ | cube%（mixed/8x4k） | mte2%（mixed/8x4k） | dk/dv | 判定 |
|---|---:|---:|---:|---:|---:|---|---|---|---|
| baseline（v5） | 441 652.0 | — | 3.231 | 138 964.2 | — | 32.5 / 31.4 | 81.5 / 85.2 | — | — |
| **ubv2** | **421 374.0** | **−4.6%** | 3.080 | **136 282.7** | −1.9% | 33.6 / 32.1 | 85.0 / 88.2 | 逐位 | ✅ 合入 |
| ubv2b | 421 356.2 | −4.6% | 3.080 | 136 088.5 | −1.9% | 33.6 / 32.1 | 85.1 / 88.2 | 逐位 | 与 ubv2 打平（<0.1%），取改动更小的 ubv2 |
| nt | 459 359.7 | +4.0% | 3.363 | 143 387.0 | +3.2% | 31.3 / 30.2 | 56.7 / 60.2 | 逐位 | ❌ 回退 |
| ubv2_nt | 439 179.3 | −0.6% | 3.213 | 143 753.8 | +3.4% | 32.3 / 30.1 | 59.0 / 59.9 | 逐位 | ❌ 回退 |

`nt` 的机制：填充字节与描述符都减半（`aic_mte2` 81.5→56.7%），但 **NN 形态的 GEMM 更慢**
（`aic_cube` 32.5→31.3%，`fixpipe` 42.0→26.4%，`aic_mte1` 28.7→26.5%）——净损失 4%。
v4 也测到同一结论，本轮再次证伪「用自然操作数换填充字节」这条路线（L0 路径的 NN 分形比 NT 贵）。

### 2.2 Round 1（续）— V=3

`ubv2` 之上把静态版本从 2 提到 3（`*_s2`，L1 311 296 → 417 792 B = 79.7%），循环变成「每 `T.serial`
迭代 3 个半体 + 3 路 drain」，让第 3 个填充提前一拍入队（`aic_mte2` 是主瓶颈，且 54–56 GB/s 明显低于
dense 参照的 118 GB/s，故判断填充受**延迟**限制）。

| candidate | mixed (µs) | Δ vs ubv2 | us/it | 8x4k (µs) | Δ vs ubv2 | mte2%（mixed） | 判定 |
|---|---:|---:|---:|---:|---:|---:|---|
| v3ub | **403 093.2** | **−4.3%** | 2.9446 | **135 433.4** | −0.6% | 89.5 | ✅ 合入 |

### 2.3 Round 2 — 统计量取数提前（用户建议 #2）

`lse/delta` 每半体 2 次 256 B GM→UB（`aiv_mte2` 44.6% 忙，实测每条约 0.22 µs 往返延迟），
改为在迭代顶部一次性发出 3 个半体的 6 条（保持每半体独立缓冲，避免共享切片）。

| candidate | mixed (µs) | Δ | 8x4k (µs) | Δ | 判定 |
|---|---:|---:|---:|---:|---|
| v3hoist | **401 892.5** | −0.3% | 135 444.9 | +0.01% | ✅ 合入（free） |

### 2.4 被阻断的两条（未进入性能采集，仅记录）

| 尝试 | 阻断原因（实测/编译） |
|---|---|
| 3 路 `st_u/dp_u/p16_u/dsT_u`（UB +36,864 B） | 运行期 UB 超限：`dyn_ubuf_bytes=229,632` > 平台上限（`aclrtLaunchKernelWithHostArgs` EE1017）；v5 基线 155,136 → ubv2 后 179,712 已是安全上界 |
| dV 单独用自然 (K,N)（省 16,384 B/半体填充） | `Get different layout for b_hi`：L0B 只剩 4 KiB，无法同时承载 NT 与 NN 两种分形（DESIGN §9.1 风险 #1 实测复现） |

### 2.5 候选 vs current best（本轮汇总）

| candidate | 一个主要改动 | mixed | 8x4k-bf16 | cube% (mix/8x4k) | AIV% (mix) | mte2% (mix/8x4k) | 判定 |
|---|---:|---:|---:|---|---|---|
| baseline-a5v5 | — | 441 652.0 | 138 964.2 | 32.5/31.4 | 80.3 | 81.5/85.2 | current best 起点 |
| r3-ubv2 | st_u/dp_u 双版本 | 421 374.0 | 136 282.7 | 33.6/32.1 | 80.1 | 85.0/88.2 | 合入 |
| r3-ubv2b | + p16/dsT 双版本 | 421 356.2 | 136 088.5 | 33.6/32.1 | 79.9 | 85.1/88.2 | 打平，不取其增量 |
| r3-nt | 自然 (K,N) 操作数 | 459 359.7 | 143 387.0 | 31.3/30.2 | 68.7 | 56.7/60.2 | rollback |
| r3-ubv2-nt | ubv2+nt | 439 179.3 | 143 753.8 | 32.3/30.1 | 66.7 | 59.0/59.9 | rollback |
| **r3-v3ub** | V=3 | 403 093.2 | 135 433.4 | 34.7/32.2 | 83.5 | 89.5/88.1 | 合入 |
| **r3-v3hoist** | v3ub + 统计量提前 | **401 892.5** | 135 444.9 | 34.8/32.2 | 83.2 | 89.7/88.1 | **winner** |

每轮精度先行：所有分支在测性能**之前**跑 `verify_parity.py`（ragged：`q_lens=[300,512,0,65]`、
`kv_lens=[700,96,64,65]`、H=3，覆盖 len_q=0、双向 shift、尾块；H=32 多 launch 例），
结果全部 `dk/dv` 逐位相同、`dq_acc` 在 §8.2 容差内（日志 `perf_opt/logs/precision/parity_*.log`）。

## 3. 为什么停在 135.4 / 402.4（结构边界，实测）

1. **填充管道已饱和**：winner 的 `aic_mte2` = 89.7%（mixed）/88.1%（8x4k）/94.0%（kimi）。
   每迭代填充字节 = 2 半体 × 82,432 B + k/v 摊销（mixed 1,142 / 8x4k 3,043）≈ 166–168 KB，
   在 62–76 GB/s 的有效带宽下即 2.2–2.6 µs/迭代 —— 与该迭代的 wall（2.50/2.94 µs）同量级。
2. **填充字节不可再降**：唯一的冗余是 qT/doT（与 q/do 同源的第二遍读取，占填充 50%）。两条去掉它的路都被封锁：
   * `nt`（全自然操作数）：实测 **GEMM 变慢**（cube 32.5→31.3%，fixpipe 42.0→26.4%），净 −4%；
   * 只给 dV 换自然形态：编译期 `Get different layout for b_hi`（L0B 4 KiB 余量，DESIGN §9.1 #1）。
3. **更深流水不可用**：V=3 已用到 L1 79.7%；V=4 会正好占满 524 288 B（无空间），
   3 路 UB 半体（`st/dp/p16/dsT`）则 UB 超限（229,632 B > 上限）。
4. **AIV epilogue 不可再快**：`aiv_vec` 83.3%，设备码是逐元素 `expf`；改成 base-2 形态会改变
   `dk/dv` 的末位（任务要求 dk/dv 与 v5 逐位），因此不动。
5. cube ≥35% 目标：mixed 34.7%（差 0.3 pt）、8x4k 32.2%。cube 占比是「cube 忙时/核时」，
   在填充饱和时提高它等于压缩填充时间——受 (2) 限制。

结论：本轮把 v5 的**块内串行**（`st_u/dp_u` 单版本）与**预取深度**（V=3）、**统计量延迟**三项做掉，
拿到 mixed −8.9%；剩余 4% 卡在 L0B/L0C/UB 三处容量决定的「每迭代填充字节」上，属设计层资源分配，
不是本轮可解的参数。

## 4. Final Summary

最终版本：`perf_opt/MLAVarlenBwdOp.py`（= `r3-v3hoist`：`st_u/dp_u` 双版本 + V=3 静态版本 +
统计量取数提前），sha256 `f031f95a…`，与 `MLAVarlenBwdOp_r3_v3hoist.py` 字节相同。

### Final Performance Test Data

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---:|---:|---|
| tileops/kernels/attention/mla_varlen_bwd/mla_varlen_bwd_kernel::_bwd_kernel+_delta_kernel+_dq_land_kernel | ds-v3-8x4k-bfloat16 | 138964.206 | 135425.400 | final-v3hoist |
| tileops/kernels/attention/mla_varlen_bwd/mla_varlen_bwd_kernel::_bwd_kernel+_delta_kernel+_dq_land_kernel | ds-v3-8x4k-float16 | 139310.222 | 135384.400 | final-v3hoist |
| tileops/kernels/attention/mla_varlen_bwd/mla_varlen_bwd_kernel::_bwd_kernel+_delta_kernel+_dq_land_kernel | ds-v3-mixed-bfloat16 | 441652.035 | 402397.500 | final-v3hoist |
| tileops/kernels/attention/mla_varlen_bwd/mla_varlen_bwd_kernel::_bwd_kernel+_delta_kernel+_dq_land_kernel | kimi-k2-4x8k-bfloat16 | 139036.456 | 138916.300 | final-v3hoist |

无回退（四 workload 全部 ≤ baseline）；mixed 402.4 与目标 400 差 0.6%（两次独立采集 401.9/402.4），
8x4k 135.4 与目标 130 差 4.2%。

### 精度（在交付文件上）

| 检查 | 命令 | 结果 |
|---|---|---|
| L0（29 例） | `python perf_opt/MLAVarlenBwdOp.py --level L0` | **29/29 PASS**（`logs/precision/L0_final.log`） |
| L1（8 例） | `--level L1` | **8/8 PASS** |
| dk/dv 与 v5 逐位（ragged / H=32 / mixed-H16） | `verify_parity.py` | **逐位相同**，`dq_acc` 仅原子抖动（`parity_*` 日志） |
| 8x4k 逐 request 非静默（H=128） | `check_nonsilent.py` | **NONSILENT_PASS**（request 0..7 全非零、尾 slab head 96/112/126/127 非零） |
| mixed 逐 request | L0-19 / G2 / parity mixed-H16 | PASS |

## 5. Skill Retrospective

| area | issue | evidence | 建议 |
|---|---|---|---|
| 精度先行 | 本轮按 dense 教训先跑 parity 再测性能，直接挡下两件事：3 路 UB 版本（运行期 UB 超限）与 dV 自然操作数（编译期 `b_hi` 分形冲突）——两者若先测性能会白跑 2 次采集 | §2.4；`logs/precision/` | profile-collection.md 补一条「候选先过 parity + 编译/资源预检，再进性能采集」 |
| 容量预检 | L1/L0B/UB 的余量应在**改动前**按分配表算术核对：V=4 正好占满 L1、3 路 UB 超限、L0B 仅 4 KiB——三条都是先算出来才没有反复试 | §3 第 2/3 条 | iteration-diagnosis.md 增「改动前资源预检表（L1/L0B/L0C/UB 逐缓冲）」 |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp |
|---|---|---|---|---|
| **A5 混合 kernel 上 `st_u/dp_u` 双版本 = −4.6%（mixed）/ −1.9%（8x4k）**：单版本的 `st_u/dp_u` 让 `_scores(B)` 等 `_epilogue(A)`（WAR），去掉后 mixed 441.7→421.4 ms；再双版本 `p16_u/dsT_u` 收益为 0（<0.1%） | D | `perf_records.jsonl` r3-ubv2 / r3-ubv2b 行；本日志 §2.1 | `python perf_opt/collect.py --impl perf_opt/MLAVarlenBwdOp_r3_ubv2.py --case ds-v3-mixed-bfloat16` | tilelang 0.1.15 / Ascend950PR_9579 / CANN 9.2.1 |
| **V=3（静态三版本 + 3 路 drain）在填充延迟受限时再拿 −4.3%**：L1 79.7%，`aic_mte2` 85.0→89.5%；V=4 会占满 L1（不可行） | D | 同表 r3-v3ub 行；§3 第 3 条 | `--impl perf_opt/MLAVarlenBwdOp_r3_v3ub.py` | 同上 |
| **L0 路径上「用自然 (K,N) 操作数换填充字节」在本平台不成立**（v4、v5 两次独立实测）：填充 6→4 描述符、−50% 字节（`aic_mte2` 81.5→56.7%），但 NN GEMM 使 cube 32.5→31.3%、fixpipe 42.0→26.4%，净 −4%；只给单个 GEMM 换则被 L0B 余量（4 KiB）挡在编译期（`Get different layout`） | P | r3-nt / r3-ubv2-nt 行；§2.4 | `--impl perf_opt/MLAVarlenBwdOp_r3_nt.py` | 同上 |
| 统计量（lse/delta）每半体 2 条 256 B GM→UB 各付 ~0.22 µs 往返；在迭代顶部一次发出 3 个半体的 6 条可免费拿 −0.3%（mixed） | P | r3-v3hoist 行；§2.3 | `--impl perf_opt/MLAVarlenBwdOp_r3_v3hoist.py` | 同上 |

## 6. 结论

在 A5 v5 上做掉三项（`st_u/dp_u` 双版本、V=3、统计量提前），mixed **−8.9%（441.7→402.4 ms）**、
8x4k **−2.5%（139.0→135.4）**、kimi 不回退，dk/dv 逐位、L0 29/29、L1 8/8。
目标 mixed ≤400 差 0.6%（噪声内）、8x4k ≤130 差 4.2%：两者都卡在「每迭代填充字节」上，
而减少它的两条路分别被 **NN GEMM 更慢**（实测）与 **L0B 4 KiB 余量**（编译期）封锁，
属设计层资源分配（DESIGN §4.6 / §9.1 #1），本轮不改设计。

## 7. 生产采纳（production adoption，2026-11-26）

### 7.1 覆盖 / 备份清单

| 文件 | 变更 | 校验 |
|---|---|---|
| `…/mla_varlen_bwd_kernel/perf_opt/_bwd_kernel.py` | **覆盖**为本轮 winner（A5 v5 + `st_u/dp_u` 双版本 + V=3 + 统计量提前） | md5 `872d62b4…` / sha256 `f031f95a…`，与本目录 `MLAVarlenBwdOp.py` **字节相同** |
| `…/perf_opt/_delta_kernel.py`、`…/perf_opt/_dq_land_kernel.py` | 保持原样（转出 Stage 3 实现，未改） | — |
| `…/perf_opt/__init__.py` | **更新**说明为 v5 世代 winner（导入语句不变） | — |
| `…/mla_varlen_bwd/mla_varlen_bwd.py` | **只改切换块**：baseline 两条 import 注释、perf_opt 三条 + 计划助手 import 生效 | wrapper sha256 `f7c57efe…`；`diff` 仅切换块 |
| `…/mla_varlen_bwd/history_version/mla_varlen_bwd_pre_stage4.py` | **备份**（切换前 wrapper） | md5 `6b20f4f3…`（= 切换前） |

未改：`_bwd_kernel.py`（v5 基线，md5 `4df7899b…` 不变）、op / manifest / test / bench。
接线核对：`_bwd_kernel` 与 `build_kernel/describe/factory_config/plan_launch` 同来自 `perf_opt._bwd_kernel`，
`_delta_kernel`/`_dq_land_kernel` 来自 Stage 3 模块；`factory_config() = {block_n:96, block_m:64, num_stages:2, head_slab:32}`
（`num_stages=2` 是 v5 的**显式版本数契约**，不是流水深度），`plan_launch(8,32768,128)` 仍
`tiles=349 slab=32 launches=4 max_grid=11168 blocks=44672 useful=44672 idle=0`。

### 7.2 生产回归（cwd `examples/TileOPs`）

| 检查 | 命令 | 结果 |
|---|---|---|
| 单测 | `pytest tests/ops/test_mla_varlen_bwd.py -q` | **12 passed**（250 s） |
| 8x4k 逐 request 非静默（H=128） | `perf_opt/check_nonsilent.py` | **NONSILENT_PASS**（request 0..7 dk/dv 全非零，尾 slab head 96/112/126/127 非零，plan idle=0） |
| dk/dv 与 v5 逐位 | `perf_opt/verify_parity.py`（用生产包 `perf_opt/_bwd_kernel.py` 直接对 Stage 3 v5 模块） | **逐位相同**（ragged / mixed-H16 / 8x4k-H32 三例），`dq_acc` 仅原子抖动（§8.2 容差内） |

### 7.3 单算子报告

- run-id **`20261126_075057_565912_MLAVarlenBwdOp`**（`examples/TileOPs/reports/tileops/<run-id>/`）
- `status = passed`；correctness 12/12；`benchmark_passed = true`；
  `invalid_acceptance_count = 0`、`invalid_profile_count = 0`、`profiler_fallback_count = 0`
- 口径 `acceptance.unit = cube`（`compute_tflops = 378`）

| case | 上一版生产 `t_op_us` | 本次 `t_op_us` | Δ | MFU(cube) 前 → 后 |
|---|---:|---:|---:|---|
| ds-v3-8x4k-bfloat16 | 138 400 | **134 392.5** | −2.9% | 27.42 → **28.23%** |
| ds-v3-8x4k-float16 | — | **134 374.7** | −2.9%（同几何） | → **28.23%** |
| ds-v3-mixed-bfloat16 | 441 700 | **402 647.7** | **−8.8%** | 22.90 → **25.12%** |
| kimi-k2-4x8k-bfloat16 | 140 300 | **140 154.6** | −0.1% | 27.04 → **27.07%** |

breakdown（mixed）：`_bwd_main 396 077.7 / _dq_land 3 227.6 / ZerosLike 1 945.8 / _delta 1 396.6`。
与本轮采集（135 425 / 135 384 / 402 398 / 138 916）逐项差 <1%（kimi 略高于本目录采集，属 run-to-run 抖动；
对上一版生产 140.3 仍为持平）。

**无回退**：三个 case 全部改善或持平，MFU 全升；`acceptance.failed = 4` 是该算子声明 MFU 阈值的档位判定
（上一版生产报告同为 `passed 0 / failed 4 / invalid 0`），非本轮引入。

### 7.4 记录

`perf_records.jsonl` 追加 4 行 `phase=merged` + `stage=production`（`candidate_id=prod-v3hoist`、
`parent_id=final-v3hoist`，含 run-id / report 路径 / prod+wrapper 的 sha256 / MFU / invalid 计数）→ 共 24 行。
