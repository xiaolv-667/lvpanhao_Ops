# SharedExpertMLPFwdOp — Stage 4 补轮（名义 MFU 89.24% → ≥90%）

- 算子：`SharedExpertMLPFwdOp`（`shared_expert_mlp`，family `moe`），语义/契约与本任务上一轮一致
  （bf16/fp16 输入、fp32 累加、输出同 dtype；`out = down(silu(hidden @ gate.T) * (hidden @ up.T))`）
- 平台：`ascend-950pr-9579`（28 Cube / 56 Vector），CANN 9.2.1，TileLang ascend 0.1.15，单卡串行
- 只读基线：`examples/shared_expert_mlp_perf/perf_opt/SharedExpertMLPFwdOp.py`
  （Stage 4 最优：`256×256×128 / ns3`、`split_k=1`、gated 激活融进 gate/up 的 Cube+Vector epilogue、
  精确 fp32 `exp`/`div`），sha256 `2c084887da85…`
- 产物：`perf_opt/SharedExpertMLPFwdOp.py`（本轮最优，sha256 `21566869dbd9…`）、本文件、
  `perf_opt/perf_records.jsonl`、`perf_opt/profiles/`、`perf_opt/logs/`
- 口径：平台 `msopprof` `Task Duration(us)`，**T_op = 一次算子调用内全部 launch 求和**，
  一次采集只 profile 一次调用，3 个独立进程取逐位置中位数（`harness/capture.py`）

**目标**：`512x7168x18432` T_op ≤ 1193 µs（名义 MFU ≥ 90%）；其余 5 case 不回退。

---

## 1. Workload inventory

沿用上一轮的 benchmark 展开（`benchmarks/ops/bench_fused_moe.py::test_shared_expert_mlp_bench`，
6 个案例均无 smoke 标记 → 全部 `kind=tune`），见 `workload_inventory.json`。没有新增/裁剪案例。

---

## 2. Performance Test Data（baseline = 上一轮最优，`phase=baseline` round 0）

基线采集就是上一轮在**同一个文件**上做的 msopprof 采集（`phase=final`，3 个独立进程），
本任务把该批原始证据原样搬迁到 `profiles/round0-baseline/`（`summary.json` 数值未改，
`msprof_raw_path` 指向搬迁后的 raw 目录），并在 opt_log 记明来源——避免对同一 artifact
重复采一次而把机器时间花在重复测量上。

| workload | launch 明细（Task Duration us） | T_op (µs) | 3 次进程 |
|---|---|---:|---|
| kimi-k2-t512 | gate_up_fused 836.24 + gemm_nt(down) 368.06 | **1204.301** | 1203.9 / 1203.5 / 1204.7 |
| ds-v3-t4096 | gate_up_fused 763.62 + gemm_nt(down) 371.42 | 1135.039 | 1133.6 / 1133.7 / 1135.2 |
| kimi-k2-t1 | gemm_nt(gate/up) 354.24 + silu_mul 3.85 + gemm_nt(down) 175.07 | 533.160 | 533.4 / 534.3 / 532.6 |
| ds-v3-t32-bf16 | gate_up_fused 44.30 + gemm_nt(down) 20.84 | 65.145 | 65.5 / 65.0 / 64.7 |
| ds-v3-t32-fp16 | gate_up_fused 44.14 + gemm_nt(down) 21.55 | 65.681 | 65.3 / 66.5 / 65.1 |
| glm45-t32 | gate_up_fused 26.98 + gemm_nt(down) 12.95 | 39.934 | 39.4 / 40.0 / 40.0 |

名义 MFU(cube 378 TF/s)：t512 **89.24%**，t4096 84.28%。与生产报告
`20261124_141934_327213`（1203.313 / 1132.632）一致（差 0.1%）。

---

## 3. 现象分析（差距在哪里）

主导 case 的 roofline = 1073.9 µs，基线 1204.3 → 富余 130.4 µs（10.8%）：

| 组成 | 实测 | 下界 | 差距 | 判据 |
|---|---:|---:|---:|---|
| gate/up（融合） | 836.2 | 716 | +120 | `aic_cube_ratio 0.970` → cube 忙 811 µs（mad 效率 88%）；`aic_mte2_ratio 0.941`（填充 2.11 GB / 836 µs = 2.5 TB/s，未到 2.9 TB/s 上限）；epilogue AIV 串在块尾 |
| down | 368.1 | 357 | +11 | `aic_cube_ratio 0.978` → cube 忙 359.7（99.4%），填充 1.06 GB / 368 µs = 2.9 TB/s = L2 上限 |

→ down 已到地板；gate/up 的 120 µs 里，一部分是块尾的 gated epilogue（AIV 串行），
一部分是 256×128 tile 的 mad 效率与 288 块 / 28 核 = 10.29 波的尾波。**本轮先打 epilogue**。

---

## 4. Iteration Log

### Round 1 — epilogue 的 exp/div 走向量 SFU（candidate `candC-sfu-epilogue`）

**单点改动**：`T.simd.vexp` / `T.simd.vdiv` 显式传 `precision="ftz_true"`（裸硬件 SFU）。
依据：`simd.py` 明确 `precision=None` 时跟随 `tl.enable_fast_math`，而该 pass config 默认关闭
→ 当前用的是**精确 0-ulp 序列**（`vdiv_0ulp_ftz_true` / 精确 `vexp`），在向量单元上是最慢的路径。
3 个 kernel（融合 epilogue、split-K reduce 的 gated 分支、独立 silu_mul）统一改为 SFU。

**分支（同一 base 派生，每个只改一点）**

| 分支 | 单点改动 | 结果（gate/up kernel，wall-clock 筛选） |
|---|---|---|
| V0 | —（精确 exp/div，现状） | 842.7 µs |
| **V1** | `precision="ftz_true"` | **815.6 µs（−3.22%）** ✅ 采纳 |
| V2 | V1 + 2 路独立 chunk 展开（ILP） | 816.4 µs（−0.05% vs V1）→ 无收益，淘汰 |
| V3 | V1 + `num_stages=4`（L1 恰好 512 KiB） | 815.7 µs（±0）→ 淘汰 |

**数值影响**（先验 + 实测）：SFU `exp`/`div` 相对精确序列差 ≤1 ulp(fp32)——
`gate/up` 乘积与 V0 对比：9.4M 元素中最大 1 个 bf16 ulp（4.883e-04，m=512 kimi），
fp16 档 3.052e-05；gated 乘积的 NSE 与 V0 相同（2.751e-06）。端到端 op 门禁
`moe_close`（atol=rtol=1e-2、NSE ≤ 1e-4）有 3 个数量级余量。

**本轮对比表（候选 vs current best，平台时延 / 利用率 / L0）**

| workload | current best T_op | candC T_op | 提升 | gate/up cube→mte2 | down cube→mte2 | L0 |
|---|---:|---:|---:|---|---|---|
| **kimi-k2-t512** | 1204.301 | **1175.811** | **−2.37%** | 0.970/0.941 → 0.970/0.942 | 0.978/0.936 → 0.979/0.927 | pass |
| ds-v3-t4096 | 1135.039 | **1110.132** | −2.19% | 0.966/0.886 → 0.966/0.889 | 0.887/0.679 → 0.883/0.699 | pass |
| kimi-k2-t1 | 533.160 | 533.502 | +0.06%（噪声，±0.2%） | 走非融合回退 | 同 | pass |
| ds-v3-t32-bf16 | 65.145 | **64.659** | −0.75% | 0.209/0.970 → 0.208/0.970 | 0.126/0.947 | pass |
| ds-v3-t32-fp16 | 65.681 | **64.912** | −1.17% | 0.211 → 0.209 | 0.125 | pass |
| glm45-t32 | 39.934 | 39.937 | ±0（噪声） | 0.244 → 0.244 | 0.157 | pass |

winner = V1（全 6 例不劣，主导两例 −2.2~−2.4%）。`profiles/round1-sfu/`、`logs/round1_sfu.log`。

**同一轮内的诊断实验（不改变代码，只用于判断是否还能再挖）**

| 实验 | 结果 | 结论 |
|---|---|---|
| S 扫描（同一 fused kernel，m=512/k=7168，S=14336/16128/18432） | 224 块（8.00 波）5.35 µs/tile；252 块（9.00 波）5.33；288 块（10.29 波）5.66 | 尾波不整使每 tile 慢 **6.2%**（约 46 µs）；要吃掉它需要「尾波 split-K」或 persistent+动态均衡，改动量与风险远超本轮 0.9% 的目标，故不动 |
| tile 空间 | 融合形态要求 `2·bm·bn·4 ≤ 256 KiB` → bm=256 时 bn ≤ 128；bn=256 需 bm=128，填充 +25%（2.64 GB） | 无更优 tile；非融合 256×256（805 µs）+ 独立 silu（53 µs）仍劣于 809 µs 的融合形态 |

### Round 2 — final 复测

`phase=final`（`profiles/final/`、`logs/final_capture.log`）在冻结文件上独立重测，与
`phase=candidate` 逐 workload 偏差 ≤0.2%（t512 1176.8 vs 1175.8；t4096 1109.4 vs 1110.1），
无翻转；6 个 final 记录同一 `candidate_id` + 同一文件 sha256。

**中止条件**：`success`。t512 1176.8 µs → 名义 MFU **91.30%**（≥90%，目标 T_op ≤1193 已满足）。

### 精度回归（每次合并与最终）

| 回归 | 范围 | 结果 |
|---|---|---|
| `perf_opt/SharedExpertMLPFwdOp.py --level all` | L0 10 例 + L1 7 例 + L2 3 例 + boundary 6 例 | 26/26 pass，`[PRECISION_PASS]`（NSE 与上一轮逐例相同：t512 1e-7 量级、m1/tail 1.6–2.8e-5，限 1e-4） |
| `harness/op_regression.py`（镜像 `tests/ops/test_fused_moe_shared_expert.py`，`moe_close` 策略） | 17 例 | 17/17 pass |
| `pytest tests/ops/test_fused_moe_shared_expert.py`（`harness/perf_opt_plugin.py` 把 op 的 kernel_map 指向本轮文件） | 仓库原有 25 例 | **25 passed**（容差未改） |

---

## 5. Final Summary

### Final Performance Test Data

（`phase=final`，round 2，同一最终候选 `candC-sfu-epilogue`；数值取自 `perf_records.jsonl`）

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---:|---:|---|
| shared_expert_mlp | kimi-k2-t1 | 533.160 | 534.483 | candC-sfu-epilogue |
| shared_expert_mlp | kimi-k2-t512 | 1204.301 | 1176.842 | candC-sfu-epilogue |
| shared_expert_mlp | ds-v3-t32-bf16 | 65.145 | 65.285 | candC-sfu-epilogue |
| shared_expert_mlp | ds-v3-t32-fp16 | 65.681 | 65.520 | candC-sfu-epilogue |
| shared_expert_mlp | ds-v3-t4096 | 1135.039 | 1109.437 | candC-sfu-epilogue |
| shared_expert_mlp | glm45-t32 | 39.934 | 40.125 | candC-sfu-epilogue |

### Final 派生指标（MFU / roofline）

| workload_id | 提升 | 名义 MFU（378 TF/s） | roofline_us | final 相对 roofline |
|---|---:|---|---:|---:|
| **kimi-k2-t512** | **−2.28%**（本补轮）/ 相对 Stage 3 基线 −45.2% | **89.24% → 91.30%** | 1073.9 | 91.3% |
| ds-v3-t4096 | −2.25% | 84.28% → 86.06% | 954.6 | 86.1% |
| kimi-k2-t1 | +0.25%（噪声） | 0.39% → 0.39%（带宽型，权重 792 MB / 534.5 µs = 1.48 TB/s） | 2.1 | — |
| ds-v3-t32-bf16 | +0.21%（噪声） | 11.43% → 11.40% | 7.5 | — |
| ds-v3-t32-fp16 | −0.25%（噪声） | 11.57% → 11.63% | 7.5 | — |
| glm45-t32 | +0.48%（噪声） | 9.94% → 9.90% | 4.0 | — |

> 4 个非主导 case 的 ±0.5% 波动是该平台的重复性量级（同一文件 3 次进程内 ≤0.2%，跨会话 ≤0.5%）；
> 它们都不经过改动的 epilogue 或只走与内存带宽绑定的路径，没有实际回退。若严格按「不回退」
> 判定，`kimi-k2-t1` +1.3 µs、`glm45-t32` +0.2 µs、`ds-v3-t32-bf16` +0.14 µs 属于噪声内。

### 结论

- 目标达成：`512x7168x18432` T_op **1176.8 µs ≤ 1193 µs**，名义 MFU **91.30%**（≥90%），
  相对 Stage 3 原始生产基线（2147.0 µs / 50.02%）为 **−45.2% / +41.3 个百分点**。
- 剩余 103 µs（相对 roofline）的构成已量化：gate/up 的 256×128 mad 效率（约 68 µs，
  受 L0C 256 KiB 对双 fp32 累加器的硬约束）与 288 块 / 28 核的尾波（约 46 µs，需尾波
  split-K 或 persistent 均衡才能吃掉），down 已到 99.4% 的 cube 效率。**未触发 `[DESIGN_LIMIT]`**
  （本次是参数/精度路径级改动，且目标已达成）。
- 生产集成包本轮**未改动**（任务只要求新 task 的 `perf_opt/`）：包内 `perf_opt/SharedExpertMLPFwdOp.py`
  仍是上一轮版本；若要把本轮 2.3% 的收益并入生产，把本文件复制过去覆盖即可（wrapper 无需改）。

---

## 6. Autotune Log

- 本轮只做了一次单点结构/精度路径改动（SFU）与两个同族变体（ILP 展开、ns=4），无框架 autotune。
- 筛选空间（`harness/probe_epilogue.py`，wall clock，5 次中位数）：`variant ∈ {v0 precise, v1 SFU,
  v2 SFU+2路展开, v3 SFU+ns4}`；winner = v1（`profiles` 记录见 `logs/`；筛选数字只用于排序，
  所有采纳数字都来自 msopprof 采集）。
- 另做了 tile/S 扫描用于判断尾波代价（见 §4 诊断表），未采纳任何结构性重排。

---

## 7. Skill Retrospective

### 7.1 流程问题

1. **采样脚本必须自定位**：本轮的 `harness/capture_all.sh` 是从上一轮拷来的，脚本里硬编码了
   上一轮的 `examples/shared_expert_mlp_perf/perf_opt/harness` 目录与相对的 `IMPL=../…`。
   结果一次 `TAG=round1-sfu` 的采集实际测的是**上一轮的文件**（835.5 µs = 精确 exp/div 的旧值），
   一度被误判为「同一文件双态 +3.4%」（BP_run_state_bimodality 的典型假象）。定位方式：
   对比 3 次进程内稳定（835.5/835.8/835.2）却与直连 `capture.py` 的 808 不符，再查 profile 目录
   落到了别的任务目录、`IMPL` 相对路径解析到旧文件。已把脚本改成 `HERE=$(dirname $0)` 自定位，
   并清理了误写进上一轮任务目录的 `profiles/round1-sfu`。**教训：跨任务复用 harness 时，
   路径必须由脚本位置推导，`--impl` 必须用绝对路径或打印解析后的绝对路径。**
2. **精度路径的默认值容易漏**：`vdiv/vexp` 的 `precision=None` 语义是「跟随 `tl.enable_fast_math`」，
   而该开关默认关闭 → 默认走精确 0-ulp 序列。Stage 3/4 两轮都是在精确路径上验证的 0-ulp 等价性，
   从未核对「是否必须精确」。这属于「默认最安全但最贵」的隐藏代价，建议在 pattern-library 里
   针对 `simd` 超越函数登记「精度-代价」对照。

### 7.2 value point proposals

| id | vp_type | 结论 | evidence（证据三件套） |
|---|---|---|---|
| VP-950-SIMD-SFU-PRECISION | **D**（实测数据） | `T.simd.vexp`/`vdiv` 的 `precision=None` = 精确 0-ulp 序列（`tl.enable_fast_math` 默认关）；显式 `precision="ftz_true"`（裸 SFU）在 9.4M 元素的 gated epilogue 上把 gate/up kernel 从 842.7 → 815.6 µs（−3.2%），对 bf16 输出的影响 ≤1 ulp，端到端 NSE 不变（2.751e-06） | evidence: `examples/shared_expert_mlp_perf2/perf_opt/logs/{round1_sfu,final_capture}.log`、`profiles/{round1-sfu,final}/kimi-k2-t512/summary.json`（gate_up_fused 836.24 → 809.24）；数值对比见 opt_log §4。repro: `.agents/platforms/toolchains/ascend/knowledge/repro/PL-950-silu-sfu-epilogue.py`。toolchain_stamp: `7a52f2f1274df07d8f4c9c01353ce9d290f88c16e85047de0c425b6a009de4aa` |
| PL-950-silu-sfu-epilogue | **P**（模式方法） | 融合 epilogue 里的 gated 激活改用 SFU `exp`/`div` 后，epilogue 从「块尾 27 µs」降到「~2 µs」量级，mixed Cube+Vector kernel 因此从 epilogue-visible 变为纯 cube-bound（`aic_cube_ratio` 0.970 不变而总时间降 3.2%）；同一改动对纯 Vector 的独立 silu kernel **无**收益（该 kernel 是访存受限），说明它只值得用在「AIV 串在 AIC 之后」的混合结构里 | evidence: 同 VP-950-SIMD-SFU-PRECISION + 独立 silu 3.85 → 3.70 µs（噪声内）。repro 同上 |
| VP-950-GEMM-TAIL-WAVE | **D**（实测数据 / 代价常数） | 28 核平台上，块数不是 28 的整数倍时的尾波代价可量化：同一 fused kernel 在 S=18432（288 块 = 10.29 波）每 tile 5.66 µs，而 224/252 块（8/9 波整）为 5.35/5.33 µs → **+6.2%**。tile 只能取 16 的倍数，故 288 % 28 = 8 无法用改 tile 消除；要吃掉它必须尾波 split-K 或 persistent 动态均衡 | evidence: opt_log §4 诊断表（`m=512,k=7168,S∈{14336,16128,18432}`）。repro: 同 PL-950-silu-sfu-epilogue.py 的第 2 段（S 扫描，自包含）。toolchain_stamp 同上 |
| R-950-HARNESS-SELF-LOCATING | **R**（流程规则） | Stage 4 harness 脚本必须自定位（`dirname $0`）并把被测文件解析成绝对路径打印出来；跨任务拷贝脚本时硬编码的 `cd <旧任务>/harness` + 相对 `--impl` 会静默测到旧文件，表现为「同一文件跨会话 +3.4%」的假双态 | evidence: opt_log §7.1-1（835.5 vs 808 的定位过程、误写目录的清理）。repro: 无（流程项） |
| C-950-shared-expert-mlp-mfu90 | **C**（案例索引） | 该算子的第二轮调优：89.24% → 91.30% 名义 MFU，唯一有效手段是 epilogue 的 SFU 化（−2.3%）；无效手段 = ILP 展开、ns=4、更窄 N tile；剩余地板 = L0C 双累加器约束（mad 效率 88%）+ 尾波 6.2% | evidence: `examples/shared_expert_mlp_perf2/perf_opt/{opt_log.md,perf_records.jsonl}` |

---

## 8. 生产采纳（第二版：SFU epilogue 进集成包）

### 8.1 文件清单

| 动作 | 文件 | sha256 |
|---|---|---|
| 覆盖（采纳件） | `examples/TileOPs/tileops/kernels/moe/shared_expert_mlp/shared_expert_mlp_kernel/perf_opt/SharedExpertMLPFwdOp.py` | `21566869dbd9…`（与本任务 `perf_opt/` 交付件逐字节相同） |
| 备份（覆盖前） | `…/shared_expert_mlp_kernel/history_version/SharedExpertMLPFwdOp_pre_sfu.py` | `2c084887da85…`（上一轮精确 exp/div 版本） |
| 未改 | wrapper `…/shared_expert_mlp/shared_expert_mlp.py`（切换块仍 `from .shared_expert_mlp_kernel.perf_opt.SharedExpertMLPFwdOp import SharedExpertMLPKernel as _PerfOptSharedExpertMLPKernel` + 末尾 rebind）、`perf_opt/__init__.py`、op/manifest/test/bench | — |

复核（`from tileops.kernels.moe.shared_expert_mlp import SharedExpertMLPKernel`）：
`__module__` 仍指向 `…shared_expert_mlp_kernel.perf_opt.SharedExpertMLPFwdOp`，与
`shared_expert_mlp.SharedExpertMLPKernel` 是**同一对象**；`perf_opt/__init__.py` 的 `__all__`
仍是 `[SharedExpertMLPKernel, TUNED_BLOCK_M, TUNED_BLOCK_N, TUNED_BLOCK_K, TUNED_NUM_STAGES, TUNED_SPLIT_K]`；
op 的 `kernel_map` 解析到 `perf_opt.SharedExpertMLPFwdOp.SharedExpertMLPKernel`，
`default_config = {block_m:256, block_n:256, block_k:128, num_stages:3, split_k:1, fuse_silu:True}`
（`SFU="ftz_true"` 是模块常量，不进 config）。

### 8.2 生产回归

```
cd examples/TileOPs && python -m pytest tests/ops/test_fused_moe_shared_expert.py -q
→ 25 passed, 14 warnings in 39.28s
```

容差未改（`moe_close`：atol=rtol=1e-2 + NSE ≤ 1e-4）；无兼容性适配、无 op 改动。

### 8.3 单算子报告（集成包、生产路径）

```
TILEOPS_TASK_DIR=/workspace/tilelang-mlir-ascend-950/examples/shared_expert_mlp TMPDIR=/workspace \
python -m tileops.reporting.cli run --op SharedExpertMLPFwdOp --prof-mode msprof
→ status=passed, run_id=20261125_103948_515964_SharedExpertMLPFwdOp
```

`report.md`：correctness `100.0% (25/25)`，Avg Max Abs Error `1.97e-04`，`MFU 91.3% (1P/5F)`；
`run.json`：`summary.invalid_acceptance_count = 0`、`summary.invalid_profile_count = 0`、
`acceptance_summary.SharedExpertMLPFwdOp.invalid = 0`，6 个 case `invalid_reason` 全 `null`。
`benchmark.json` 的 `tileops` 记录 `captured_op_names` 仍是 `['gate_up_fused_kernel','gemm_nt_kernel']`
（t1 为 `['gemm_nt_kernel','silu_mul_kernel']`），证明生产路径用的是本轮内核。

| case | 报告 T_op(us) | 上一版生产(us) | 变化 | 报告 MFU（名义/实测峰值） | Acceptance |
|---|---:|---:|---:|---|---|
| **512x7168x18432** | **1175.983** | 1203.313 | **−2.27%** | **91.3% / 93.2%** | **PASS** |
| 4096x7168x2048 | 1107.166 | 1132.632 | −2.25% | 86.2% / 88.0% | FAIL（90% 目标） |
| 1x7168x18432 | 532.831 | 535.080 | −0.4% | 0.394%（带宽上限 0.423%） | FAIL（带宽型） |
| 32x7168x2048 bf16 | 64.657 | 65.233 | −0.9% | 11.5%（上限 13.4%） | FAIL（带宽型） |
| 32x7168x2048 fp16 | 65.292 | 64.433 | +1.3% | 11.4%（上限 13.4%） | FAIL（带宽型） |
| 32x5120x1536 | 40.150 | 40.200 | −0.1% | 9.95%（上限 13.4%） | FAIL（带宽型） |

主导 case 现在是报告里**唯一** `Acceptance PASS` 的 case（1P/5F）；其余 5 个是权重带宽受限的
decode/small-M shape（MFU 上限 0.4%–13.4%），MFU 目标口径对它们不适用。
`32x7168x2048 fp16` 的 +0.9 µs（+1.3%）与本任务 harness 采集（65.681 → 65.520）方向相反，
属该 case 的重复性量级（多次采集 64.4–66.5 µs 区间），不是回退。

### 8.4 生产记录

`perf_records.jsonl` 追加 6 行 `phase=merged, round=3`（`report_run_id = 20261125_103948_515964…`，
`msprof_raw_path` 指向报告 artifacts，`artifact_path` 为集成包内的 perf_opt 文件）。
stage 门禁复核：`statectl gate 4 --dir examples/shared_expert_mlp_perf2` → `pass: true`（failures 0 / warnings 0）。
