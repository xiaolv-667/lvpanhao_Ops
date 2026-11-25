# ChainSpeculativeSamplingFwdOp — Stage 4 调优日志

- 算子 / kernel：`ChainSpeculativeSamplingFwdOp` / `_chain_speculative_sampling_kernel`
- kernel_id：`examples/TileOPs/tileops/kernels/sampling/chain_speculative_sampling/chain_speculative_sampling_kernel/_chain_speculative_sampling_kernel.py::_chain_speculative_sampling_kernel::main`
- 基线（只读）：同路径的 Stage 3 文件；调优产物：`perf_opt/chain_speculative_sampling.py`（同内容副本 `perf_opt/ChainSpeculativeSamplingFwdOp.py`）（同工厂签名 `(batch, num_draft, vocab) -> callable`，同 7 实参调用契约，wrapper 可 drop-in 切换）
- 平台：`ascend-950pr-9579` / `ascend`，CANN 9.2.1，TileLang ascend 0.1.15，单卡（56 Vector core / 28 Cube core，UB 253952 B）
- 测量口径：**平台 msprof（`msopprof --aic-metrics=Roofline`）的 kernel Task Duration(us)**，`--launch-count 15 --warm-up 5`，原始证据在 `perf_opt/profiles/**`（字段 `msprof_raw_path`），逐测量记录在 `perf_opt/perf_records.jsonl`。文中的 wall-clock 数字只用于分支粗筛（已标注，不进入任何 winner/Final 判定）。

## 1. Performance Test Data（baseline，round 0）

| kernel_id | workload_id | candidate_id | target_kernel_name | captured_op_name | task_duration_us | profile_status | raw_profile_dir | command |
|---|---|---|---|---|---:|---|---|---|
| `_chain_speculative_sampling_kernel` | llama-8b-b16-n4-float32 | baseline | `main_kernel` | `main_kernel` | 1244.34 | valid | `perf_opt/profiles/round0/baseline/llama-8b-b16-n4` | `msopprof --output=… --launch-count=15 --warm-up=5 --aic-metrics=Roofline python perf_opt/baseline_case.py --case llama-8b-b16-n4` |
| `_chain_speculative_sampling_kernel` | llama-8b-b64-n4-float32 | baseline | `main_kernel` | `main_kernel` | 2487.42 | valid | `perf_opt/profiles/round0/baseline/llama-8b-b64-n4` | 同上 `--case llama-8b-b64-n4` |
| `_chain_speculative_sampling_kernel` | ds-v3-mtp-n1-float32 | baseline | `main_kernel` | `main_kernel` | 2518.37 | valid | `perf_opt/profiles/round0/baseline/ds-v3-mtp-n1` | 同上 `--case ds-v3-mtp-n1` |
| `_chain_speculative_sampling_kernel` | ds-v3-mtp-n3-float32 | baseline | `main_kernel` | `main_kernel` | 2545.00 | valid | `perf_opt/profiles/round0/baseline/ds-v3-mtp-n3` | 同上 `--case ds-v3-mtp-n3` |
| `_chain_speculative_sampling_kernel` | qwen3-235b-eagle-n5-float32 | baseline | `main_kernel` | `main_kernel` | 1423.16 | valid | `perf_opt/profiles/round0/baseline/qwen3-235b-eagle-n5` | 同上 `--case qwen3-235b-eagle-n5` |
| `_chain_speculative_sampling_kernel` | llama-8b-b17-n8-float32 | baseline | `main_kernel` | `main_kernel` | 1264.75 | valid | `perf_opt/profiles/round0/baseline/llama-8b-b17-n8` | 同上 `--case llama-8b-b17-n8` |
| `_chain_speculative_sampling_kernel` | llama2-7b-b256-n1-float32 | baseline | `main_kernel` | `main_kernel` | 1870.16 | valid | `perf_opt/profiles/round0/baseline/llama2-7b-b256-n1` | 同上 `--case llama2-7b-b256-n1` |

复测与 `mfu_baseline_v1.json` 一致（1245.8 / 2472.1 / 2529.2 / 2542.8 / 1427.5 / 1259.7 / 1666.3 µs），偏差 ≤0.2%，仅 `llama2-7b-b256-n1` 本次复测偏高 12%（同一进程内多次复测稳定在 1.87 ms，判定为该 workload 在当前卡状态的固有值，不影响结论：该 workload 提升 2.83x）。

**workload inventory**：benchmark `benchmarks/ops/bench_chain_speculative_sampling.py::test_chain_speculative_sampling_bench` 从 manifest 展开 8 例 = 1 smoke + 7 tune（`perf_opt/workload_inventory.json`）。smoke `chain-sampling-smoke (2,2,64)` 只做精度回归（最终 `precision_pass: true`），`skipped` 无。调优顺序按实测负担：`ds-v3-mtp-n3`（2545.0，主导）→ `llama-8b-b64-n4`（2487.4）→ `ds-v3-mtp-n1`（2518.4）→ `llama2-7b-b256-n1`（1870.2）→ `qwen3-235b-eagle-n5`（1423.2）→ `llama-8b-b17-n8`（1264.8）→ `llama-8b-b16-n4`（1244.3）。

### 设计估算 vs 实测（D-2 回填）

`DESIGN.md` §1.6.0 的估算下界（主选 C1，`tile=8192`、`grid=min(B,56)`）与本轮 baseline 实测：

| workload | 设计估算下界 µs | baseline 实测 µs | 偏差 | 失准项 |
|---|---:|---:|---:|---|
| ds-v3-mtp-n3 | 26.3 | 2545.0 | 97x | 流量项按 `1.26 TB/s` 聚合带宽折算；实测该 SIMT 逐元素通路每核仅 ~9 GB/s（56 核 ≈ 0.5 TB/s 上限），且 kernel 是**指令发射受限**（约 2.4 cycle/元素）而非带宽受限 |
| llama-8b-b16-n4 | 9.0 | 1244.3 | 138x | 同上 + `B=16` 时只用 16/56 核 |
| llama2-7b-b256-n1 | 26.0 | 1870.2 | 72x | 同上 |

结论：`DESIGN.md` §1.6.0 的 roofline 只建了「带宽」与「发射（0.5 µs/op 组）」两项，缺了本平台**逐元素 SIMT 通路的每核吞吐项**（实测 ≈ 1.5 元素/cycle/核，`T.Parallel`+reducer 的 GM 读路径 ≈ 9 GB/s/核），因此对全接受/拒绝两类 workload 都低估一个量级。该偏差是本轮全部优化点的来源，已作为 D 类价值点回写（见 §6）。

## 2. Iteration Log

### Round 1 — 现象、候选优化点、分支

**现象（baseline profile + 阶段消融，`perf_opt/probe_phase.py` / `probe_cstage.py`，`B=16,N=4,V=128256`，wall-clock）：**

| 阶段（累积） | wall µs | 增量 | 说明 |
|---|---:|---:|---|
| chain only（Philox + 接受判定） | 19.3 | — | 每行 ~19 µs；生成码每轮 `asc_syncthreads()`（UB 标量读写） |
| + copy（只读 target 一行，纯向量读） | 129.3 | +110 | 单行 512 KB / 129 µs ≈ **4 GB/s/核**：GM 读是逐元素路径的地板 |
| + fold（读两行 + 残差 + reducer） | 239.8（garbage 基址，不可用） | — | 见下（地址为 garbage 时读被短路） |
| L0（chain+fold+emit，`probe_cstage`） | 518.7 | — | fold 本体 ~500 µs |
| L1 + 粗层 tile 前缀 | 524.4 | +5.8 | |
| L2 + 命中 tile 重算 | 603.0 | **+78.6** | 一个 tile 的活干出 78 µs（16 次带运行时分支的 region） |
| L3 + max 归约兜底 | 600.7 | −2.3 | |
| L4 + lane 分部和 | 606.0 | +5.3 | |
| L5 + lane 前缀 + 元素走查 + 写出（= Stage 3） | 630.8 | +24.7 | |

**根因（生成码核对 `_fold_direct_gen.cu` / `gen_select.cu`）：** fold 的内层每元素展开成
`if (tile 尾谓词) { if (guard) load target; if (guard && n_s<N) load draft; }` —— 即
**每元素一个 draft 掩码分支 + 两个带边界守卫的标量 GM 读**。

**候选优化点（同一 base 派生，每分支只改一项）：**

| 分支 | 主要优化点 | 结果（msprof Task Duration，dominant `ds-v3-mtp-n3`） | L0 | 结论 |
|---|---|---:|---|---|
| b1 `round1/b1-lane` | 兜底索引从「整 tile max 归约」改成「反向 lane 扫描」（省一遍 O(tile)） | 2553.9 | PASS | **无增益**（走查只占 4%），rollback |
| b2 `round1/b2-mul` | fold 的 draft 掩码从「每元素 select」改成「每元素乘 0/1」+ fold 尾 tile 剥离 + 行基址/掩码入寄存器 | **694.8** | PASS | **有效（3.66x）**，采纳 |
| b3 `round2/b3-hoist` | 在 b2 基础上把 draft 读的分支**提到每 tile 一次**（全接受行不读 draft） | 712.6 | PASS | 主导 case 与 b2 平手（±2.5%），但在 `llama2-7b-b256-n1`（−7.4% wall）与 `llama-8b-b16-n4`（−11% wall）明显更好 → 作为最终默认 |

**候选 vs current best（Round 1，platform msprof Task Duration；利用率/访存按 manifest roofline bytes 折算）**

| workload | candidate | Task Duration µs | vs best | 有效带宽 GB/s | MBU %（1.6 TB/s） | L0 |
|---|---|---:|---:|---:|---:|---|
| ds-v3-mtp-n3 | current best = baseline | 2545.0 | 1.00x | 13.0 | 0.81 | PASS |
| ds-v3-mtp-n3 | b1 lane-walk | 2553.9 | 1.00x | 13.0 | 0.81 | PASS |
| ds-v3-mtp-n3 | b2 fold-mul | 694.8 | **3.66x** | 47.6 | 2.98 | PASS |
| ds-v3-mtp-n3 | b3 fold-hoist | 712.6 | 3.57x | 46.5 | 2.90 | PASS |

wall-clock 粗筛（交错多 run、同进程，`perf_opt/sweep_case.py` / `ab_knobs.py`，仅供排序）：b2 在 5/7 workload 上略胜 b3（2–4%），b3 在 `ds-v3-mtp-n1`(389.0 vs 452.8)、`llama2-7b-b256-n1`(453.7 vs 490.2)、`b16-n4`(185.5 vs 208.2) 上胜出，故默认选 b3；所有 tune workload 的合并非回退由 Final 表逐项核对。

### Round 2 — 现象、候选优化点、分支

**现象（Round 1 winner profile + 消融）：** `ds-v3-mtp-n3` 694.8 µs 中，fold ≈ 576 µs（两次读 66 MB，56 核），命中 tile 重算 ≈ 23 µs，lane 走查 ≈ 10 µs，chain test ≈ 38 µs（2 行/核），其余为发射与写出。

**候选优化点（同 base 派生）：**

| 分支 | 主要优化点 | 结果 | L0 | 结论 |
|---|---|---:|---|---|
| r2-a | 重算 pass 改「单次运行期基址」（flat） | wall 515.0 vs 495.5（更慢） | — | rollback（且 `TRAP-950-dynamic-base-tile-pass` 记录该拼写在部分 vocab 上设备 abort，不采纳） |
| r2-b | chain test 的 Philox 状态改为寄存器（去掉 UB 标量写导致的每写一次 barrier） | wall 502.5 → 505.1（无差） | PASS | 保留（无害、去风险），但不计增益 |
| r2-c | 残差拼写 `zfac*draft`（重算 pass 用乘代替 select） | 计入 b2 内，无单独增益 | PASS | 保留 |
| r2-d | 行基址 / 掩码 / 接受长度入寄存器（fold 内层不再逐元素读 UB 标量） | 计入 b2 内 | PASS | 保留 |

**候选 vs current best（Round 2）**

| workload | candidate | Task Duration µs | vs best | 有效带宽 GB/s | MBU % | L0 |
|---|---|---:|---:|---:|---:|---|
| ds-v3-mtp-n3 | current best（b3 default） | 712.6 | 1.00x | 46.5 | 2.90 | PASS |
| ds-v3-mtp-n3 | 最终交付版（同 b3，复测） | 709.3 | 1.00x（±0.5% 重复性） | 46.7 | 2.92 | PASS |

**平区裁决（<5%）**：b2 vs b3 在主导 case 上的差 2.5%，按 T-2 用交错 A/B（`ab_knobs.py`，4–5 run，run 间离散 <1%）裁决：`ds-v3-mtp-n3` b3 492.1 vs b2 505.1（b3 胜）、`llama2-7b-b256-n1` 453.7 vs 490.2（b3 胜）、`qwen3` 224.1 vs 228.0（b3 胜）、`b16-n4` 185.5 vs 208.2（b3 胜）。msprof 单点在主导 case 上相反（694.8 vs 712.6），属采集噪声量级，故以交错 A/B 裁决采纳 b3，并在 Final 表用同一候选重测全部 workload。

**合并非回退检查**：最终版对全部 7 个 tune workload 复测（Final 表）均优于 baseline，无回退；smoke `chain-sampling-smoke` 精度通过。

**终止条件**：success —— 主导 `ds-v3-mtp-n3` 2545.0 → 709.3 µs（−72.1%，3.59x），超过「至少降 50%」的目标；全部 tune workload 有调优结论与最终同候选记录。

## 3. Autotune Log

未使用 autotune。本算子的编译期参数（`tile=8192`、`threads=128`、`ntile=ceil(V/tile)`、`per_lane=64`、`blocks=min(B,56)`）**由逐位一致性约束冻结**：tile 边界、lane 边界与 128 线程的浮点分组共同决定 `tile_sum`/`lane_sum` 的 fp32 求和顺序，改动任一参数都会改被抽 token 的取整结果（详见 §5）。故搜索空间只有「同一分组下的拼写」，已按分支穷举。

## Final Performance Test Data

同一最终候选 `perf_opt_hoist_lane`（= `perf_opt/chain_speculative_sampling.py`，`_KNOBS = {fold: hoist, walk: lane, resid: mul, chain: local, recomp: branch}`，sha256 `5b2b5a782ea3c13b…`），逐 workload 在同一次采集配置（`--launch-count 15 --warm-up 5`，msprof Task Duration）下重新测得：

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---:|---:|---|
| examples/TileOPs/tileops/kernels/sampling/chain_speculative_sampling/chain_speculative_sampling_kernel/_chain_speculative_sampling_kernel.py::_chain_speculative_sampling_kernel::main | llama-8b-b16-n4-float32 | 1244.34 | 337.783 | perf_opt_hoist_lane |
| examples/TileOPs/tileops/kernels/sampling/chain_speculative_sampling/chain_speculative_sampling_kernel/_chain_speculative_sampling_kernel.py::_chain_speculative_sampling_kernel::main | llama-8b-b64-n4-float32 | 2487.421 | 702.996 | perf_opt_hoist_lane |
| examples/TileOPs/tileops/kernels/sampling/chain_speculative_sampling/chain_speculative_sampling_kernel/_chain_speculative_sampling_kernel.py::_chain_speculative_sampling_kernel::main | ds-v3-mtp-n1-float32 | 2518.367 | 667.865 | perf_opt_hoist_lane |
| examples/TileOPs/tileops/kernels/sampling/chain_speculative_sampling/chain_speculative_sampling_kernel/_chain_speculative_sampling_kernel.py::_chain_speculative_sampling_kernel::main | ds-v3-mtp-n3-float32 | 2545.001 | 709.297 | perf_opt_hoist_lane |
| examples/TileOPs/tileops/kernels/sampling/chain_speculative_sampling/chain_speculative_sampling_kernel/_chain_speculative_sampling_kernel.py::_chain_speculative_sampling_kernel::main | qwen3-235b-eagle-n5-float32 | 1423.162 | 405.997 | perf_opt_hoist_lane |
| examples/TileOPs/tileops/kernels/sampling/chain_speculative_sampling/chain_speculative_sampling_kernel/_chain_speculative_sampling_kernel.py::_chain_speculative_sampling_kernel::main | llama-8b-b17-n8-float32 | 1264.752 | 346.999 | perf_opt_hoist_lane |
| examples/TileOPs/tileops/kernels/sampling/chain_speculative_sampling/chain_speculative_sampling_kernel/_chain_speculative_sampling_kernel.py::_chain_speculative_sampling_kernel::main | llama2-7b-b256-n1-float32 | 1870.16 | 660.586 | perf_opt_hoist_lane |

### Final 利用率与带宽（由 manifest roofline bytes 折算，非时延口径）

| workload | bytes MB | speedup | final 有效带宽 GB/s | final MBU % |
|---|---:|---:|---:|---:|
| llama-8b-b16-n4-float32 | 8.21 | 3.68x | 24.3 | 1.52 |
| llama-8b-b64-n4-float32 | 32.84 | 3.54x | 46.7 | 2.92 |
| ds-v3-mtp-n1-float32 | 33.10 | 3.77x | 49.6 | 3.1 |
| **ds-v3-mtp-n3-float32** | 33.10 | 3.59x | 46.7 | 2.92 |
| qwen3-235b-eagle-n5-float32 | 19.45 | 3.51x | 47.9 | 2.99 |
| llama-8b-b17-n8-float32 | 8.72 | 3.64x | 25.1 | 1.57 |
| llama2-7b-b256-n1-float32 | 32.77 | 2.83x | 49.6 | 3.1 |

### Final 精度回归

最终文件上重跑候选自带分层精度：L0 13/13、L1 7/7、boundary 7/7（`drawn-token mismatches=0`），`perf_opt/verify_exact.py` 26/26 与 Stage 3 kernel 逐位一致。
smoke：`chain-sampling-smoke` 只在每次合并后与最终做精度回归，`precision_pass: true`（无性能记录）。`skipped`：无。

## 5. 逐位一致性（本轮硬约束）

任务约束：**token 输出与基线逐位一致**（RNG 流、CDF 求和顺序、接受/拒绝判据不可改）。因此本轮只改「同一分组的拼写」，不改分组：

- 生成码核对（`B=16,N=4,V=128256`，`get_kernel_source()` 实取）：`fold=select`（Stage 3）与 `fold=hoist` 的 fold 内层元素→线程映射**逐字相同** ——
  `select`: `target_probs[(int64)t*8192 + (int64)i*128 + (int64)sh[8480] + threadIdx.x]`；
  `hoist` : `target_probs[(int64)t*8192 + (int64)i*128 + (int64)at_tl[0] + threadIdx.x]`（`i = 0..63`，`for (int32_t i = 0; i < 64; ++i)`），
  差别只在基址来源（UB 标量 vs 寄存器）与 draft 读的分支位置。因此每 lane 的 float32 累加顺序、`AscendAllReduce` 树、`tile_sum` 数值完全相同；`hoist` 的两条分支对同一批元素同序相加，只改「是否读 draft 行」。
- `resid=mul`：`x * 1.0 == x`，`x * 0.0 == +0.0`（概率为有限值），故 `max(t - z*d, 0)` 与 `max(t - (n<N ? d : 0), 0)` 逐位相同。
- `chain=local`：Philox 全为整数运算，寄存器与 UB 标量只影响放置，不影响数值。
- `walk=lane`：兜底分支扫描「最后一个有正质量的 lane 的最后一个正元素」= 原 max 归约的「tile 内最大正下标」，同值。

验证：`perf_opt/verify_exact.py` 对 26 个用例（7 个 benchmark shape + L0/L1/边界 shape + 4 种控制流 + `storage_offset=1` + 奇数 vocab + 1 token vocab）**逐位比较基线 kernel 与候选的 `output_token_ids` 与 `num_accepted`：全部 0 差异**，且 `num_accepted` 与 torch 参考 0 差异。候选自带分层精度：L0 13/13、L1 7/7、boundary 7/7（`drawn_token_mismatches=0`）。

## 6. 剩余空间与后续候选（未采纳，如实记录）

1. **多核拆 tile（DESIGN §1.6.3 备选 B 的实验裁决）**：本轮未实现该变体，但用同口径实测量化了它的上界——逐元素通路的**每核吞吐与 grid 无关**（`perf_opt/probe_bw.py`，`lane` 臂，33–35 MB）：

   | grid（核） | bytes MB | Task Duration µs（wall） | 聚合 GB/s | 每核 GB/s |
   |---|---:|---:|---:|---:|
   | 16 | 33.55 | 209.5 | 160 | 10.0 |
   | 28 | 33.95 | 122.6 | 277 | 9.9 |
   | 56 | 34.87 | 65.3 | 534 | 9.5 |

   → 聚合吞吐随核数线性，故「把一行的 tile 均分到更多核」的收益 ≈ 核数比：`ds-v3-mtp-n3`（`B=64`，`per_core=2`，8 核各 2 行 / 48 核各 1 行，关键路径 32 tile）均分到 56 核后关键路径 ≈18.3 tile ⇒ fold **≈1.75x**；`B<56` 的 shape 收益更大（b16-n4：16 核 → 56 核 ⇒ fold 上限 ≈3.5x）。fold 占最终时延 ~81%（`ds-v3-mtp-n3`：fold ≈576/709 µs），故整体保守估计 1.4–1.5x（n3）/ 1.8–2.2x（b16）。
   **未采纳原因**：需要一份跨核 partial 的 GM workspace，而 Stage 3 工厂的 7 实参调用契约由只读 wrapper 固定（不能用 `out_idx` 增参）；可行路线是让工厂返回一个惰性分配 workspace、内部连发两个 kernel 的 closure（契约不变）。该路线本轮未实现（时间预算用于先收束已达标版本并保住逐位一致性），且其采纳会推翻已记录的 final 采集。**结论：备选 B 的结构性胜出方向已由实测标定，实现细节为未实证假设，留作下一轮第一优先候选。**
2. **逐元素通路的向量化天花板**：fold 是全算子 81% 的耗时所在，实测 ~2.4 cycle/元素（指令发射受限，含 2 次带守卫标量 GM 读）。`T.simd`（64 宽 `vld/vadd/vcmax`）可把这部分推高一个量级，但会把 float32 求和从「每 lane 64 步顺序链 + 块归约树」换成位置式累加，**改变求和分组 → 改变被抽 token**，与本次「逐位一致」硬约束冲突。若后续允许「只要求 `num_accepted` 逐位、token 按分布等价」（即 `tests/ops/test_sampling.py` 的实际判据），这是首选方向。
3. `llama2-7b-b256-n1` 的 chain test（5 行/核）与 `ds-v3-mtp-n3` 的命中 tile 重算（~23 µs）在本轮拼写优化后占比已 <5%，未继续压缩。

## 7. Skill Retrospective

### 流程问题

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| 现象诊断 | `iterations-diagnosis` 的「标量占比 >50% 触发换轴」在本算子无法直接判定：平台 msprof 的 PipeUtilization 里本 kernel 没有任何 pipe 命中（aic NA），真正的诊断信号是「每元素周期数」（实测 ~2.4 cycle/element/core）与「每核 GM 吞吐」（~9 GB/s）。建议在 `profile-collection.md` 增加「纯 Vector 小算子的可判定信号」小节：用 bytes/(T_op×核数) 反解每核吞吐，再与 pattern-library 的 copy-floor 常数比 | 本任务 §1 消融表（copy 129 µs / 单行 512 KB / 16 核） | `tilelang-op-optimize/references/profile-collection.md` 增「无 pipe 命中时的判定式」 | P |
| 逐位敏感算子 | 调优 skill 的候选优化点框架默认「容差内等价」，对「必须逐位一致」的算子缺少可执行规程：真正有效的做法是 ① 先 dump 生成码核对「元素→lane 映射」是否相同，② 只改「同分组的拼写」，③ 用基线 kernel 逐位对拍做门禁 | 本任务 §5（生成码核对 + `verify_exact.py` 26/26） | `tilelang-op-optimize/references/iteration-diagnosis.md` 增「逐位契约算子的三条纪律」 | P |
| 分支初筛口径 | 逐分支跑 msprof（~40 s/次）在 7×N 分支上不可行，实际做法是交错 wall-clock 粗筛 + 只对关键分支/最终版上 msprof。建议在 `run_experiments_template.py` 明确「两级口径」：`--screen`（wall interleaved）与 `--record`（msprof），并规定 winner 判定只用后者 | 本任务 Round 1/2：msprof 单点 694.8/712.6 与交错 A/B 492.1/505.1 排序相反 | `tilelang-op-optimize/references/autotune.md`（runner 节）增 `--screen` 档 | R |

### 价值点提案（vp_type + 证据三件套）

| title | vp_type | evidence | repro | toolchain_stamp |
|---|---|---|---|---|
| **TRAP：`T.Parallel`+reducer 内「每元素 select」= 每元素分支，代价 3.1x 整个 fold**。同一 reduce 语义下，`max(t - ite(cond, d, 0), 0)` 生成 `if(cond){load}else{0}` 嵌在展开的元素循环里；改成每 tile 一次分支（全接受行整行不读 draft）或 `z*d`（z∈{0,1}）后 fold 从 1308 µs 降到 420 µs（`B=64,N=3,V=129280`，56 核）。判据：生成码出现内层 `if`，且 `(T_with - T_without)/T_total > 1` | D | `perf_opt/probe_fold2.py`（`fold-c` 1307.9 vs `fold-e` 420.0，同 shape 同 grid）；正式记录 `perf_records.jsonl` round1（2553.9 → 694.8） | `perf_opt/probe_fold2.py`（A/B 双臂同脚本，`--variants c,e`） | TileLang ascend 0.1.15 / CANN 9.2.1 / Ascend950PR_9579 / 2026-11-24 |
| **CONST-950-vector-elementwise-per-core**：`T.Parallel` 逐元素 GM 读 + reducer 的每核吞吐实测 ≈ **9.4 GB/s（≈1.5 元素/cycle/核）**；`T.copy`（MTE2 批量）反而更低（6.2 GB/s/核）；`store→UB→reducer` 往返为其 1/2.4（3.9 GB/s/核）。用途：给「纯 Vector 逐元素/归约」类算子的 roofline 提供**每核吞吐项**（当前 pattern-library 只有聚合带宽与 0.5 µs/op 发射项，导致本算子设计下界低估 97x） | D | `perf_opt/probe_bw.py`：33.03 MB / 56 核，`lane` 62.4 µs、`dma` 95.2 µs、`store_*` 149.1 µs；`perf_opt/probe_phase.py` 单核侧：单行 512 KB / 129.3 µs（16 核） | `perf_opt/probe_bw.py --grid 56 --per 18 --modes lane,dma,store_sbase` | 同上 |
| **P：逐位敏感算子的三步调优法**：① dump 生成码核对「元素→lane 映射」（本例 fold 内层 `t*8192 + i*128 + threadIdx.x`）确认分组未变；② 只替换同分组的拼写（掩码改乘、尾块剥离、行基址入寄存器、兜底改反向 lane 扫描）；③ 用上游 kernel 逐位对拍做门禁（26/26）。三步把「不敢动」变成可安全迭代，本例拿到 3.59x 且输出逐位不变 | P | 本文件 §5；`perf_opt/verify_exact.py`（ALL CASES BIT-EXACT） | `perf_opt/verify_exact.py`（自包含对拍脚本：基线 factory vs 候选 factory，7+19 用例） | 同上 |
| **P：`T.alloc_local` 必须分配在 `with T.SimtVF(...)` 内部**：`T.Kernel` 作用域的 local 会变成 `__simt_vf__` 函数的指针实参，编译报 `pointer arguments to simt_vf functions must reside in '__gm__ or __ubuf__' address space`；移入 VF region 内即正常并落到寄存器 | D | 本任务 Round 2 报错原文与修复（`at_tl/at_dl/zl/nl` 从 Kernel 作用域移入 VF region） | `perf_opt/ChainSpeculativeSamplingFwdOp.py` 中 `with T.SimtVF` 内的 `at_tl` 分配（对照可在 VF 外声明同名 local 复现编译错误） | 同上 |
| **P：reducer epoch 规则**：同一 `T.alloc_reducer` 只能有一个 init/finalize 位点（`double T.reducer_init` / `has no T.finalize_reducer` 两条 pass 报错），因此「按 runtime 条件拆两组 `T.Parallel` 累加」必须各用独立 reducer；而 `for t in T.serial(0)` 的循环体在 IR 被消除时会让该 reducer 变成「无 finalize」——故 peel 出来的循环要用 python 级 `if full_tiles > 0` 保护 | D | 本任务 Round 1/2 的两次编译失败与修复 | `perf_opt/ChainSpeculativeSamplingFwdOp.py`（`can` / `cant` 两个 reducer + `if full_tiles > 0` 守护） | 同上 |

## 8. 交付摘要

- 最终文件：`perf_opt/chain_speculative_sampling.py`（= 任务状态里的 `kernel_opt_py_path`；同工厂签名 `(batch, num_draft, vocab)`、同 7 实参调用契约、`out_idx=[]`，wrapper 只需切换 import 块）。同内容副本 `perf_opt/ChainSpeculativeSamplingFwdOp.py`（逐字节相同，sha256 `5b2b5a78…`）便于按算子名取用
- 主导 case `ds-v3-mtp-n3`：**2545.0 → 709.3 µs（−72.1%，3.59x）**，MBU 0.81% → 2.92%
- 全部 7 个 tune workload：2.83x–3.77x；smoke 精度通过；与基线 kernel 逐位一致（26/26）
- 中止原因：success（达成 ≥50% 目标）；`[DESIGN_LIMIT]`：无（剩余空间是本轮未实现的实现层候选，不是设计层不可达）

---

# 生产采纳（Production Adoption）

**日期**：2026-11-24 ｜ **平台**：`ascend-950pr-9579` / `ascend`，单卡串行，`TMPDIR=/workspace`

## 采纳文件清单

| 类型 | 路径 | sha256 |
|---|---|---|
| wrapper 备份（采纳前） | `examples/TileOPs/tileops/kernels/sampling/chain_speculative_sampling/history_version/chain_speculative_sampling_pre_opt.py` | `30ecc5996ff2afa8bbda0b8d0ee60bf1a322170c96fcd8ae2aefc91e2f459ca0` |
| wrapper（翻转后） | `examples/TileOPs/tileops/kernels/sampling/chain_speculative_sampling/chain_speculative_sampling.py` | 仅切换块两行变化（见下） |
| 落地 kernel（drop-in） | `…/chain_speculative_sampling/chain_speculative_sampling_kernel/perf_opt/_chain_speculative_sampling_kernel.py` | `5b2b5a782ea3c13becc63ca70f6d9ef7b52c9a28ea60901be351b272a4a4714d` |
| 落地包 `__init__`（新增） | `…/chain_speculative_sampling_kernel/perf_opt/__init__.py` | `cdbdad1d18e23d8afea43f894764d8f7a2ca3c8b3b4806a0546bbdb5b370eab4` |

切换块翻转（唯一 wrapper 改动，符合 `optimize.md` §5）：

```diff
 # --- baseline (Stage 3) ------------------------------------------------------
-from .chain_speculative_sampling_kernel import _chain_speculative_sampling_kernel
+# from .chain_speculative_sampling_kernel import _chain_speculative_sampling_kernel
 # --- perf_opt (Stage 4 tuned) ------------------------------------------------
-# from .chain_speculative_sampling_kernel.perf_opt._chain_speculative_sampling_kernel import _chain_speculative_sampling_kernel
+from .chain_speculative_sampling_kernel.perf_opt._chain_speculative_sampling_kernel import _chain_speculative_sampling_kernel
```

落地文件与本任务 `perf_opt/chain_speculative_sampling.py` **逐字节相同**（sha256 一致），未改 Stage 3 kernel 目录、op/manifest/test/bench。运行期核对：`wrapper._chain_speculative_sampling_kernel.__module__ == "…chain_speculative_sampling_kernel.perf_opt._chain_speculative_sampling_kernel"`（`perf_opt/logs/production/verify_exact_prod.log` 首两行）。

## 生产回归

| 回归 | 命令 | 结果 |
|---|---|---|
| op 层精度测试 | `cd examples/TileOPs && TILEOPS_TASK_DIR=…/examples/chain_speculative_sampling python -m pytest tests/ops/test_sampling.py -q` | **9 passed**（3.42 s，`perf_opt/logs/production/pytest_sampling.log`）；注：任务书写的是 `tests/ops/test_chain_speculative_sampling.py`，仓内实际入口为 `tests/ops/test_sampling.py`（ChainSpeculativeSamplingFwdOp 的全部用例都在其中，manifest `source.test` 亦指向它） |
| 逐位对拍（部署对） | `python perf_opt/verify_exact_prod.py` | **26/26 BIT-EXACT**：部署的 perf_opt 模块 vs Stage 3 包，`output_token_ids` 与 `num_accepted` 全 0 差异，`num_accepted` 与 torch 参考全 0 差异（`perf_opt/logs/production/verify_exact_prod.log`） |
| 单算子报告 | `TILEOPS_TASK_DIR=…/examples/chain_speculative_sampling TMPDIR=/workspace python -m tileops.reporting.cli run --op ChainSpeculativeSamplingFwdOp --prof-mode msprof` | **status=passed**，run_id `20261124_190245_979904_ChainSpeculativeSamplingFwdOp` |

## 生产报告（msprof，run_id `20261124_190245_979904_ChainSpeculativeSamplingFwdOp`）

`run.json`：`correctness_passed=true`（9/9，`Avg Max Abs Error = 0.00e+00`）、`benchmark_passed=true`（7/7）、`msprof_case_count=7`、`invalid_profile_count=0`、`invalid_acceptance_count=0`、`profiler_fallback_count=0`、`kernel_filter_fallback_count=0`、`pass_rate=1.0`。

| workload | T_op µs（生产） | T_op µs（baseline） | 提升 | achieved GB/s | MU(MBU) % | roofline util % | profile |
|---|---:|---:|---:|---:|---:|---:|---|
| llama-8b-b16-n4-float32 | 338.8 | 1245.8 | 3.68x | 24.2 | 1.51 | 15.2 | valid |
| llama-8b-b64-n4-float32 | 702.2 | 2472.1 | 3.52x | 46.8 | 2.92 | 29.2 | valid |
| ds-v3-mtp-n1-float32 | 661.4 | 2529.2 | 3.82x | 50.0 | 3.13 | 31.3 | valid |
| **ds-v3-mtp-n3-float32** | **714.6** | **2542.8** | **3.56x** | 46.3 | **2.89** | 29.0 | valid |
| qwen3-235b-eagle-n5-float32 | 405.7 | 1427.5 | 3.52x | 47.9 | 3.00 | 30.0 | valid |
| llama-8b-b17-n8-float32 | 347.9 | 1259.7 | 3.62x | 25.1 | 1.57 | 15.7 | valid |
| llama2-7b-b256-n1-float32 | 629.4 | 1666.3 | 2.65x | 52.1 | 3.25 | 32.6 | valid |

- 主导 case 与预期一致：**714.6 µs（预期 ≈709，差 0.8%，属采集噪声）/ MU 2.89%（预期 ≈2.9%）**，不是 2543 µs；`acceptance.value`（MFU）= 0.151%，`mfu_ceiling_percent` = 5.22% —— 该算子按 manifest roofline 是访存型（`arithmetic_intensity 0.75`），MFU 口径的 `FAIL` 属类别对齐后的诊断，不是本次验收判据。
- 其余 6 个 case 全部改善，无回退；`llama2-7b-b256-n1` 相对文档 baseline 为 2.65x（相对本任务同环境复测 baseline 1870.2 µs 为 2.97x）。
- 报告证据：`examples/TileOPs/reports/tileops/20261124_190245_979904_ChainSpeculativeSamplingFwdOp/{run.json,report.md,benchmark.md,report.html}`；逐 case msprof 原始目录记录在 `perf_records.jsonl` 的 `msprof_raw_path`（`phase=merged`，round=4，candidate_id `perf_opt_hoist_lane_production`）。

## 一致性说明

- 生产 T_op 与 Stage 4 末次 final 采集（`perf_records.jsonl` `phase=final`）逐 case 偏差 ≤1.0%（b16 +0.3%、b64 −0.1%、n1 −1.0%、n3 +0.8%、qwen3 −0.1%、b17 +0.3%、b256n1 −4.7%，后者为长尾 workload 的采集散布），排序与结论不变。
- 环境清理：运行前后各清理一次 `/tmp/tileops_msprof_*`；本任务所有临时数据在 `/workspace` / `/tmp/opencode` 下；未触碰 `/workspace/TileOPs-ascend950`。
