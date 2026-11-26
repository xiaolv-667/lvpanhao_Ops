# `RoPEFwdOp` — Stage 4 调优日志（round 2：MBU 70% → 86%）

- 任务：`examples/rope_perf2`（scenario=optimize, mode=full；平台 `ascend-950pr-9579` / `ascend`，单卡）
- 基线（只读）：`examples/rope_perf/perf_opt/rope.py`（round-1 产物，sha256 `96f60401…`）
- 设计：`examples/rope/_rope_positions_kernel/DESIGN.md`；生产集成包：`examples/TileOPs/tileops/kernels/rope/`
- 产物：`perf_opt/RoPEFwdOp.py`（sha256 `162d7ed3…`）、`perf_opt/perf_records.jsonl`（31 行）、
  `perf_opt/workload_inventory.json`、`perf_opt/profiles/`、`perf_opt/logs/`、本文件
- **口径**：`msopprof --launch-count=1 --warm-up=0 --kernel-name=main` 包住**一次**算子调用，读
  `OpBasicInfo*.csv` 的 `Task Duration(us)`；本算子每次调用 1 个 launch（驱动断言 rows==1）。
  驱动 `perf_opt/experiments/bench_one.py`（warm-up 独立进程）。MBU = roofline bytes / T_op / 1.6 TB/s，
  MFU = 3·rotated / T_op / 该 dtype vector 峰值（fp32 23 TF/s、16-bit 47 TF/s）。
- **执行上下文**：按 `statectl platform` 的 `execution`（cwd = 任务目录）运行；`PYTHONPATH` 补
  `examples/TileOPs`；`TMPDIR=/workspace`、`TILELANG_CACHE_DIR=/tmp/opencode/tl-cache-opt-rope2`。

---

## 0. Phase 0 — 上下文与首轮必查项

- 工具链/设备与 round 1 相同（tilelang 0.1.15 / CANN 9.2.1、Ascend950PR_9579、28 Cube / 56 Vector）。
- 算子类型：纯 Vector / 访存型。
- **首轮必查 1（重新定位瓶颈）**：round-1 产物在主导 case 上的逐块画像（`profiles/baseline/base_c7_r1`）：
  `aiv_time 11.93 µs/块`、`aiv_vec_ratio 0.78`（向量 pipe 忙 9.33 µs）、`aiv_mte2_ratio 0.59`、
  `aiv_mte3 0.14`、`aiv_scalar 0.04`。即 **向量遍仍是第一大项**，DMA 已接近其速率 → 本轮主攻向量遍的执行方式。
- **首轮必查 2（向量化轴/布局）**：布局（每 token 行 `H·D` 连续、DMA 描述符搬运）保持；**变化的是核内执行方式**：
  `T.SimtVF` 的 per-lane 访问在本后端每次 64-lane 机器操作约 10 cycle（≈25.6 B/cycle），
  而 `T.SimdVF` 的 256 B 向量寄存器（`T.simd.vld/vsts/vmul/vsub`）可以一次覆盖整个 head 半区
  （`half·itemsize = 128/256 B`）。
- **首轮必查 3（陷阱/知识回验）**：`TRAP-950-vcvt-narrow-store-pack-b32`（PK_B32 + 32-bit lane 掩码是 16-bit
  窄化落盘的唯一稠密形态）与 `CONST-950-ub-value-as-gm-row-index` 在本轮直接采用；`TRAP-950-simd-ub-store-load`
  的 V→V 判据复核：本轮 SIMD 段内每个地址都先 `vld` 后 `vsts`（写后不再回读），不属于该陷阱形态，
  真机 13+7+3+11 例全过、L1/L2 逐位（max_diff=0）。

## 1. Phase 1 — Inventory 与 Baseline

workload 与 round 1 完全一致（benchmark 的 7 个参数化案例，无 smoke/skip → 全部 `kind=tune`），
见 `perf_opt/workload_inventory.json`。`kernel_id` 唯一。

| kernel_id | workload_id | baseline T_op(µs) | MBU | MFU |
|---|---|---:|---:|---:|
| rope | llama3-8b-b2-2k-float32 | **148.289** | 15.8→(round1)70.0% | 1.80% |
| rope | llama3-8b-prefill-4k-bfloat16 | 103.129 | 52.0% | 1.31% |
| rope | llama3-8b-prefill-4k-float16 | 103.199 | 51.1% | 1.29% |
| rope | gptj-6b-prefill-float16 | 85.661 | 49.5% | 0.32% |
| rope | phi-2-query-only-bfloat16 | 14.952 | 43.9% | 0.45% |
| rope | llama3-8b-decode-b64-bfloat16 | 9.628 | 8.4% | 0.21% |
| rope | gptj-6b-decode-b32-float16 | 5.997 | 10.9% | 0.07% |

（本表是本轮同 harness 复测；与生产报告 `20261124_181621` 的 150.1 / 99.9 / 99.7 / 83.6 / 13.8 / 8.4 / 5.1 µs
在 ±4% 内一致，decode 段因 launch 下限波动略高。）
排序：`llama3-8b-b2-2k`（148 µs，占 7 案例合计约 50%）→ prefill-4k ×2 → gptj-prefill → phi-2 → decode ×2。

**设计估算 vs 实测（D-2）**：DESIGN 的带宽下界 106.2 µs 按 1.6 TB/s 名义值给出；本机 HBM **实测**有效带宽
1.42–1.49 TB/s（`mfu_baseline_v1.json` 的 1.424、以及本轮 `p_copy` 探针 134.2 MB/89.8 µs = 1.49 TB/s），
故「完美实现」的下界是 169.9/1.49 = **114 µs（93% MBU）**，而 90% MBU = 118 µs 需要 1.44 TB/s
（≈ 实测峰值的 97–101%）。本轮据此定位目标区间。

## 2. Iteration Log

### Round 1 — 参数级：pipeline stage 数与 block 行数（无实质收益）

| 候选 vs current best | Task Duration(µs) | vs best | 说明 |
|---|---:|---:|---|
| current best（round-1 产物） | 148.289 | — | ns=2, bt=4, step=32 |
| `r1-ns3`（`num_stages` 2→3） | **145.291** | −2.0% | 小幅有效 |
| `r1-ns4` | 145.897 | −1.6% | 与 ns3 同级 |
| `r1-bt8`（子块 4→8 行） | 147.174 | −0.8% | 无收益 |
| `r1-bt16`（子块 16 行） | 145.790 | −1.7% | 与 ns3 同级 |

结论：block 几何已饱和（round 1 已扫到 32/64 行）；唯一可留的是 `num_stages=3`。真正的瓶颈是向量遍本身。

### Round 2 — 结构级：用 `T.simd` 向量寄存器重写旋转遍（主收益）

**候选优化点**：把原地旋转遍从 SIMT per-lane 执行改为 SIMD 向量寄存器执行（`T.SimdVF` + `T.simd.vld/vsts/vmul/vsub`）：
一个 head 的 lo/hi 半区（`half=64`）在 fp32 下正好是 1 个 256 B 寄存器、在 16-bit 下经 `UNPK_B16` 解开成
64 个 fp32 lane；cos/sin 行仍每块一次载入 UB，并在 `(i,ch)` 层加载一次、供该行 8 个 head 复用。
数值路径不变（fp32 运算、落盘一次舍入）：fp32 直接 `vsts`，16-bit 走 `vcvt(part=0, sat=False)` + `PK_B32`。
非 NeoX（GPT-J 相邻对）或 `half % 64 != 0` 的几何自动回落到原 SIMT 遍（构建期选择，不是运行期分派）。

| 候选 vs current best | workload | Task Duration(µs) | vs best | MBU | MFU | L0/L1 | 结论 |
|---|---|---:|---:|---:|---:|---|---|
| current best | llama3-8b-b2-2k | 148.289 | — | 70.0% | 1.80% | PASS | base |
| **`r2-simd-rotate`** | llama3-8b-b2-2k | **121.354** | **−18.2%** | **87.5%** | 2.21% | PASS | **采纳** |
| `r2-simd-ns2` | llama3-8b-b2-2k | 129.386 | −12.7% | 82.1% | 2.07% | PASS | 回退 |
| `r2-simd-bt8` | llama3-8b-b2-2k | 123.708 | −16.6% | 85.9% | 2.17% | PASS | 略差 |
| `r2-simd-step64` | llama3-8b-b2-2k | 123.612 | −16.6% | 86.0% | 2.17% | PASS | 略差 |
| `r2-simd-unroll`（trace 期展开 i/ch/h） | llama3-8b-b2-2k | 122.990 | −17.1% | 86.4% | 2.18% | PASS | 无收益（非循环开销受限） |
| `r2-simd-step16` | llama3-8b-b2-2k | 123.882 | −16.5% | 85.9% | 2.17% | PASS | 略差 |
| **`r2-simd-rotate`** | llama3-8b-prefill-4k bf16 | **79.079** | −23.3% | 67.1% | 1.69% | PASS | 采纳 |
| **`r2-simd-rotate`** | llama3-8b-prefill-4k fp16 | **79.206** | −23.3% | 67.0% | 1.69% | PASS | 采纳 |
| **`r2-simd-rotate`** | gptj-6b-prefill fp16（SIMT 回落） | 79.880 | −6.7% | 52.7% | 0.33% | PASS | 采纳 |
| **`r2-simd-rotate`** | phi-2（SIMT 回落） | 14.191 | −5.1% | 45.4% | 0.46% | PASS | 采纳 |
| **`r2-simd-rotate`** | llama3-8b-decode-b64 | **7.014** | −27.1% | 11.8% | 0.29% | PASS | 采纳 |
| **`r2-simd-rotate`** | gptj-6b-decode-b32 | **5.913** | −1.4% | 11.1% | 0.07% | PASS | 采纳 |

逐块画像对比（主导 case，`PipeUtilization` 均值）：
`aiv_time` 11.93 → **9.77 µs**，`aiv_vec_ratio` 0.78 → **0.53**（向量 pipe 9.33 → 5.22 µs），
`aiv_mte2_ratio` 0.59 → 0.68（DMA 占比升高，说明向量不再独占）。生成码见
`profiles/src/r3_simd_*.cce`：`__simd_vf__` + `simd_inst::vlds_norm/vmul/vsub/vadd/vsts_norm`，
每 head 8 条向量指令（2 vld + 4 ALU + 2 vsts），cs/sn 每 `(row,ch)` 只载一次。
被拒分支 `r5_vmula`/`r6_fma`（用 `vmula` 把 ALU 从 3 降到 2 条/head）在构建期就失败：
`vmula` 的 `dst` 必须是 `T.simd.alloc_local` 分配的缓冲区访问，且形如 `acc[0] = vmul(...)` 的
`float32x64` 赋值被 IR builder 拒绝（`TypeError: Incompatible types in BufferStore`）→ 未进入实测（`family_no_gain`）。

### 中止条件

`success`：主导 case MBU 70.0% → **86.2%**（本 harness）/ **88.4%**（生产 report）；T_op
123.2 µs（harness）/ **120.1 µs**（生产）< 目标 125 µs，
prefill-4k 52.0% → **67.1%**（目标 70%，差 3 pt），decode 全改善，7/7 workload 无回退，
L0/L1/L2/boundary 与 op 层 10/10 全过。90% MBU（118 µs）需要 1.44 TB/s ≈ 本机 HBM 实测峰值
（1.42–1.49）的 97–101%，即**已到达硬件边界**：本轮 123.2 µs 对「同载荷纯 DMA」实测 1.49 TB/s
的 114 µs 下界为 1.08x。剩余空间属工具链/硬件能力（SIMT↔SIMD 之外的 MTE2 与 V 完全重叠、16-bit 的
双半区单指令装载），非设计层假设矛盾 → 不产出 `perf_feedback.md`、不带 `[DESIGN_LIMIT]`。

## 3. Final Performance Test Data

最终工件 `perf_opt/RoPEFwdOp.py` 上重采（`profiles/final/*`，2–3 次独立采集取均值/中位数）。
本算子 7 个 benchmark 案例均无 smoke 标记，故无 smoke 行；精度见 §4。

### Final Performance Test Data

| kernel_id | workload_id | baseline_us | final_us | candidate_id |
|---|---|---:|---:|---|
| tileops/kernels/rope/rope/rope_kernel/_rope_positions_kernel.py::_rope_positions_kernel::main | llama3-8b-b2-2k-float32 | 148.289 | 123.220 | final-simd |
| tileops/kernels/rope/rope/rope_kernel/_rope_positions_kernel.py::_rope_positions_kernel::main | llama3-8b-prefill-4k-bfloat16 | 103.129 | 79.320 | final-simd |
| tileops/kernels/rope/rope/rope_kernel/_rope_positions_kernel.py::_rope_positions_kernel::main | llama3-8b-prefill-4k-float16 | 103.199 | 78.467 | final-simd |
| tileops/kernels/rope/rope/rope_kernel/_rope_positions_kernel.py::_rope_positions_kernel::main | gptj-6b-prefill-float16 | 85.661 | 80.639 | final-simd |
| tileops/kernels/rope/rope/rope_kernel/_rope_positions_kernel.py::_rope_positions_kernel::main | phi-2-query-only-bfloat16 | 14.952 | 14.140 | final-simd |
| tileops/kernels/rope/rope/rope_kernel/_rope_positions_kernel.py::_rope_positions_kernel::main | llama3-8b-decode-b64-bfloat16 | 9.628 | 7.189 | final-simd |
| tileops/kernels/rope/rope/rope_kernel/_rope_positions_kernel.py::_rope_positions_kernel::main | gptj-6b-decode-b32-float16 | 5.997 | 5.753 | final-simd |

### 3.1 提升、MBU/MFU 与证据

| workload | round-1 best(µs) | **final(µs)** | 本轮提升 | vs 原始基线(671 系列) | MBU | MFU | 采集样本(µs) | 证据目录 |
|---|---:|---:|---:|---:|---:|---:|---|---|
| llama3-8b-b2-2k-float32 | 148.289 | **123.220** | **−16.9%** | 671.119 → **5.45x（−81.6%）** | 15.8% → **86.2%** | 2.21% | 123.283 / 123.156 | `profiles/final/final_llama3-8b-b2-2k-float32_r0,r1` |
| llama3-8b-prefill-4k-bfloat16 | 103.129 | **79.320** | −23.1% | 430.361 → 5.43x | 52.0% → **67.1%** | 1.69% | 78.322 / 80.318 | 同上 |
| llama3-8b-prefill-4k-float16 | 103.199 | **78.467** | −24.0% | 429.756 → 5.48x | 51.1% → **67.7%** | 1.69% | 78.784 / 78.150 | 同上 |
| gptj-6b-prefill-float16 | 85.661 | **80.639** | −5.9% | 330.012 → 4.09x | 49.5% → 52.2% | 0.33% | 80.860 / 80.418 | 同上 |
| phi-2-query-only-bfloat16 | 14.952 | **14.140** | −5.4% | 141.328 → 9.99x | 43.9% → 45.6% | 0.46% | 14.140 / 13.990 / 14.200 | 同上 |
| llama3-8b-decode-b64-bfloat16 | 9.628 | **7.189** | −25.3% | 10.925 → 1.52x | 8.4% → 11.5% | 0.29% | 7.349 / 7.189 / 7.002 | 同上 |
| gptj-6b-decode-b32-float16 | 5.997 | **5.753** | −4.1% | 10.302 → 1.79x | 10.9% → 11.4% | 0.07% | 6.095 / 5.733 / 5.753 | 同上 |

- 主导 case 123.2 µs = **1.38 TB/s**，是本机实测 HBM 峰值（1.42–1.49 TB/s）的 93–97%；
  对「同载荷纯 `T.copy` DMA」实测下界 114 µs 为 1.08x。
- prefill-4k 的 79.3 µs = 1.07 TB/s（67%）：16-bit 下 tile 字节减半而元素数不变，SIMD 遍每 head 多两次
  `vcvt`（UNPK/PK），故其向量占比高于 fp32；进一步收益需要「16-bit 双半区单指令装载」（`vld2` + 交织），
  本轮未做，列为后续候选。
- decode 段仍是 launch/并行度下限（MBU 11%），但相对 round 1 全部改善。

## 4. 精度回归

| 层 | 结果（`perf_opt/RoPEFwdOp.py`） | 日志 |
|---|---|---|
| L0 | **13/13 PASS**，max_diff 4.77e-07 | `logs/precision/final_L0.log` |
| L1 | **7/7 PASS**，max_diff **0**（逐位） | `logs/precision/final_L1.log` |
| L2 | **3/3 PASS**，max_diff **0** | `logs/precision/final_L2.log` |
| boundary | **11/11 PASS**，max_diff 2.38e-07 | `logs/precision/final_boundary.log` |
| op 层 drop-in（`experiments/wrapper_check.py`） | **10/10 PASS** | 7 benchmark 形状 + 3 fixture 形状 |

非 NeoX / `half % 64 != 0` 的几何走 SIMT 回落分支，boundary（D=65、R=24、P=2、±inf/NaN、offsets 越表）
与 34 例生产测试面同源，全部通过；容差未改（fp16 1e-3、bf16 1.6e-2、fp32 1e-5）。

## 5. 生产采纳（交接给 Stage 5 / conductor）

### 5.1 生产路径实测（已用真实 report 验证，随后还原）

把 `perf_opt/RoPEFwdOp.py` 临时覆盖到
`examples/TileOPs/tileops/kernels/rope/rope/rope_kernel/perf_opt/_rope_positions_kernel.py`
（wrapper 的切换块已在 round 1 翻转为 perf_opt，本轮**未改 wrapper**），跑单算子报告后还原 round-1 文件
（安装前 sha256 `96f60401…`，还原后一致）：

- run-id **`20261125_123729_120109_RoPEFwdOp`**，`status=passed`，correctness **34/34**，
  benchmark 7/7，`invalid_acceptance_count=0`、`invalid_profile_count=0`、
  `profiler_fallback_count=0`、`kernel_filter_fallback_count=0`
- `pytest tests/ops/test_rope.py -q` → **34 passed**（日志 `logs/production/pytest_test_rope.log`）

| case | 生产 T_op(µs) | MBU | round-1 生产 | 本轮提升 |
|---|---:|---:|---:|---:|
| **llama3-8b-b2-2k（fp32）** | **120.106** | **88.4%** | 150.136 / 69.9% | −20.0% |
| llama3-8b-prefill-4k（bf16） | **76.060** | **69.8%** | 99.947 / 52.5% | −23.9% |
| llama3-8b-prefill-4k（fp16） | **75.321** | **70.5%** | 99.650 / 52.6% | −24.4% |
| gptj-6b-prefill（fp16） | 78.350 | 53.8% | 83.554 / 50.2% | −6.2% |
| phi-2-query-only（bf16） | 12.546 | 51.4% | 13.799 / 46.4% | −9.1% |
| llama3-8b-decode-b64（bf16） | 6.132 | 13.5% | 8.438 / 9.7% | −27.3% |
| gptj-6b-decode-b32（fp16） | 4.967 | 13.3% | 5.094 / 12.9% | −2.5% |

生产口径比本阶段自建 harness 好 2–4%（benchmark harness 的输入分布/分配方式不同，同为 msprof Task Duration）；
**主导 case 88.4% MBU、prefill-4k 69.8/70.5%（≥70% 达标）**。

### 5.2 采纳步骤（Stage 5）

- 覆盖 `…/rope_kernel/perf_opt/_rope_positions_kernel.py` 为 `perf_opt/RoPEFwdOp.py` 即可（同一 drop-in
  factory 契约）；`perf_records.jsonl` 的 7 行 `phase=merged` 已记录上述生产实测。
- **本轮未修改**任何基线 kernel / op / manifest / test / bench，也未改动集成包（只读复用了
  `experiments/baseline_c7.py` 作为本轮基线，其字节与 `examples/rope_perf/perf_opt/rope.py` 相同，sha256 `96f60401…`）。
- 数值/契约：fp32 运算、单次舍入、输出 dtype/shape 不变；`units_per_block=2048`、`staged` 双值、
  四个 out_idx 变体（query-only / key / offsets）不变 → op 层无需适配。

## 6. Skill Retrospective

### 6.1 流程发现

1. 本轮的目标（≥80–90% MBU）只有换**执行方式**才能达到：round 1 的 DMA/tiling/表 gather 已到位，
   参数级扫描（stage 数、block 行数、step）合计只值 2%。建议 skill 在 Phase 2 的候选清单里
   把「同算子换成 SIMD 向量寄存器执行」列为*纯访存/逐元素*算子的标准候选（当前 bottlenecks.md 没有该方向）。
2. `_SIMD_ROT` 的适用条件必须由**构建期几何**决定（约定 + `half % 64`），不能运行期分派；
   与 skill 的「条件路径只放在 `@T.prim_func` 内、不做 Python factory 分派」一致，但几何回退
   必须在同一个 factory 内以 Python 级 `if` 选择**代码形态**（非运行期分派），此处已按此实现。
3. `vmula` 类累加指令要求 `alloc_local` 目的地，且 `acc[0] = vmul(vector)` 的 `float32x64` 赋值被 IR builder
   拒绝；本次因此放弃「ALU 3→2 条/head」的小优化。工具链层面属**能力缺口候选**（未证实可行形态），
   已在复盘记录，未写入知识库（避免未验证断言）。

### 6.2 价值点提案

| id | vp_type | 内容 | 证据三件套 |
|---|---|---|---|
| VP-7 | D | 950PR 逐元素 kernel：`T.SimtVF` per-lane 遍 ≈10 cycle/64-lane 机器操作（≈25.6 B/cycle），换 `T.SimdVF` + 256 B 寄存器后同一遍 9.33 → 5.22 µs/块；整 kernel 148.3 → 121.4 µs（主导 case，MBU 70→87.5%） | evidence：`perf_opt/profiles/baseline/base_c7_r1`（vec 0.78）vs `profiles/experiments/r3_simd_r1`（vec 0.53）、`profiles/src/r3_simd_*.cce`（`__simd_vf__`）；repro：`perf_opt/experiments/r3_simd.py` + `bench_one.py`；toolchain_stamp 7a52f2f1… |
| VP-8 | P | 访存型算子的 SIMD 化步骤：①先确认一次搬运的连续段 ≥ 256 B（`half·itemsize`）；②`vld` 全寄存器装载 + 逐行复用表寄存器；③16-bit 用 `UNPK_B16`/`vcvt(part=0,sat=False)`/`PK_B32`+32-bit 掩码；④几何不满足时构建期回落 SIMT | evidence：§2 表与 §4 精度表（L1/L2 逐位 0）；repro：同上 |
| VP-9 | D | 16-bit 下 SIMD 收益被 `vcvt`（UNPK/PK）摊薄：prefill-4k 67% vs fp32 86%；「双半区单指令装载 + `vintlv`」是下一步（未实测） | evidence：§3.1 表；repro：`perf_opt/experiments/r3_simd.py`（bf16 workload） |

### 6.3 回写 pattern-library

已写入（`kb_lint --files` → **0 failures**）：

- `knowledge/patterns-950pr-verified-forms.md` 新增 **`CONST-950-simd-vs-simt-elementwise`**：
  同一 256 B 数据段下 SIMT per-lane 遍 9.33 µs/块 vs SIMD 向量寄存器 5.22 µs/块（整 kernel 145.3 → 121.4 µs），
  含适用前提（连续段 ≥ 256 B、16-bit 走 `UNPK_B16`/`PK_B32`、几何不满足时构建期回落）与边界；
- repro **`knowledge/repro/CONST-950-simd-vs-simt-elementwise.py`**：同形状 rotate 探针，编译码断言
  （`__simd_vf__`+`vlds_norm` vs `__simt_vf__`）与 `--time` 时延断言（实测 simt 130.8 µs → simd 113.9 µs，
  方向一致），已实跑通过；
- `knowledge/INDEX.md` 已登记该 id。

## 7. 原始证据索引

- 基线：`profiles/baseline/base_c7_r0,r1`（含 `PipeUtilization`）、`profiles/baseline/base_<workload>_r0`
- Round 1：`profiles/r1/{r1_ns3,r1_ns4,r1_bt8,r1_bt16}_r0`
- Round 2：`profiles/experiments/{r3_simd,r3_simd_ns2,r3_simd_bt8,r3_simd_step64,r4_unroll,r3s16_*}_r*`、
  `profiles/r1/r3_<workload>_r0`（7 个 workload 的 SIMD 版首测）
- 最终：`profiles/final/final_<workload>_r*`
- 生成码：`profiles/src/{baseline_c7,r3_simd,r3_simd_*,r4_simd_unroll,...}_<workload>.cce`
- 精度日志：`logs/precision/final_{L0,L1,L2,boundary}.log`、`logs/precision/r3_simd_*.log`
- 采集/回归命令与 stdout 归属：`perf_opt/logs/README.md`
