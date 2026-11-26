# MLAVarlenBwdOp — Stage 4 调优日志（Cube 化后定向优化）

- 算子：`MLAVarlenBwdOp`（packed / THD MLA prefill backward，三 launch：`_delta` + N×`_bwd_main` + `_dq_land`）
- 基线：`examples/TileOPs/tileops/kernels/attention/mla_varlen_bwd/mla_varlen_bwd_kernel/_bwd_kernel.py`
  （本目录副本 `perf_opt/MLAVarlenBwdOp_baseline_stage3.py`，md5 `4262fe4b72cbf7c4106dbc40c1e91ee3`）
- 设计：`examples/mla_varlen_bwd/_bwd_kernel/DESIGN.md`（v4 Cube）
- 平台：`ascend-950pr-9579` / `ascend`，单卡（cube 28 / vector 56，L2 128 MB）；工具链 tilelang 0.1.15 / CANN 9.2.1
- 测量口径：**msopprof Task Duration(us)**；`T_op` = 一次调用内全部 launch 求和
  （`_delta_main` + N×`_bwd_main` + `_dq_land_main` + `ZerosLike`），与生产报告 `t_op_us` 同口径。
  采集流程见 `perf_opt/collect.py`（`msprof`/`msopprof op`，一次 capture 窗口正好覆盖一次调用，
  `window_ok=true` 逐次校验）；原始采集在 `perf_opt/profiles/`，stdout 在 `perf_opt/logs/`。

## 0. 环境与口径自校验

`perf_opt/collect.py` 对本任务基线的实测与生产报告 `20261125_061941_018389` 逐项一致，确认口径可对账：

| workload | 本日志 baseline `T_op` | 生产报告 `t_op_us` | 偏差 |
|---|---:|---:|---:|
| ds-v3-8x4k-bfloat16 | 203 983.0 µs | 203 548.7 µs | +0.2% |
| ds-v3-mixed-bfloat16 | 689 184.2 µs | 689 090.0 µs | +0.01% |
| kimi-k2-4x8k-bfloat16 | 194 026.6 µs | 194 260.4 µs | −0.1% |

## 1. Performance Test Data

target kernel：`_bwd_main_kernel`（主导 launch，占 `T_op` 94–99%）；`T_op` 含另外三个 launch。

| kernel_id | workload_id | candidate_id | target_kernel_name | captured_op_name | task_duration_us | profile_status | raw_profile_dir |
|---|---|---|---|---|---:|---|---|
| `_bwd_kernel`（+delta+land） | ds-v3-8x4k-bfloat16 | baseline-stage3 | `_bwd_main_kernel` | `_bwd_main_kernel` | 203 983.0 | valid | `perf_opt/profiles/round0/ds-v3-8x4k-bfloat16_baseline` |
| `_bwd_kernel`（+delta+land） | ds-v3-8x4k-float16 | baseline-stage3 | `_bwd_main_kernel` | `_bwd_main_kernel` | 203 948.1 | valid | `perf_opt/profiles/round0/ds-v3-8x4k-float16_baseline` |
| `_bwd_kernel`（+delta+land） | ds-v3-mixed-bfloat16 | baseline-stage3 | `_bwd_main_kernel` | `_bwd_main_kernel` | 689 184.2 | valid | `perf_opt/profiles/round0/ds-v3-mixed-bfloat16_baseline` |
| `_bwd_kernel`（+delta+land） | kimi-k2-4x8k-bfloat16 | baseline-stage3 | `_bwd_main_kernel` | `_bwd_main_kernel` | 194 026.6 | valid | `perf_opt/profiles/round0/kimi-k2-4x8k-bfloat16_baseline` |

smoke / skip 排除：

- `mla-varlen-bwd-smoke-bfloat16` / `-float16`：benchmark 参数化 index 0 带 `pytest.mark.smoke`，只做精度回归
  （不采性能、不选 winner）。精度结果见 §2.5（L0-1 / L0-2 + parity `PARITY_PASS`）。
- 无 `skipped` 案例（benchmark 未 skip）。

### 1.1 baseline 的逐 workload 分解（平台测量 + 设备 pipe）

`us/it` = `main_us × 28 / iterations`（iterations 由 DESIGN §1.6.0 R3 公式复算）；
`cube%` 是 **sum 口径**（`Σ pipe_time / Σ aic_time`，比逐块 ratio 均值对异质块更准，见 §3.1）；
`vec%` 同法取 AIV。

| workload | T_op (µs) | main (µs) | iterations | us/it | cube%: mad/mte1/mte2/fix/scal | vec%: vec/mte2/mte3 | mte2 active BW |
|---|---:|---:|---:|---:|---|---|---:|
| ds-v3-8x4k-bf16 | 203 983.0 | 197 323.8 | 1 441 792 | **3.832** | 20.8 / 19.0 / 30.7 / 16.2 / 3.0 | 36.9 / 10.0 / 3.2 | 79.3 GB/s |
| ds-v3-8x4k-fp16 | 203 948.1 | 197 305.0 | 1 441 792 | 3.832 | 20.8 / 19.0 / 30.7 / 16.2 / 3.0 | 36.9 / 10.0 / 3.2 | 79.5 GB/s |
| ds-v3-mixed | 689 184.2 | 682 644.0 | 3 770 496 | **5.069** | 21.4 / 19.6 / 30.9 / 16.9 / 2.7 | 36.4 / 10.7 / 3.1 | 77.4 GB/s |
| kimi-k2-4x8k | 194 026.6 | 190 670.1 | 1 420 032 | 3.760 | 20.8 / 18.7 / 32.4 / 15.9 / 2.7 | 35.7 / 12.1 / 3.0 | 73.5 GB/s |
| smoke-bf16（仅参考） | 86.7 | 73.3 | 192 | 10.690 | 15.8 / 14.5 / 21.7 / 13.0 / 10.4 | 46.6 / 3.7 / 3.5 | 113.5 GB/s |

### 1.2 设计期估算 vs 实测偏差（D-2 回填，DESIGN.md §1.6.0「设计期 roofline」投影）

| workload | 设计估算（2.97–3.19 µs/it 投影） | baseline 实测 | 偏差（实测/估算） |
|---|---:|---:|---:|
| ds-v3-mixed | 400.0–429.5 ms | 689.2 ms | **1.60–1.72×** |
| ds-v3-8x4k | 152.9–164.3 ms | 204.0 ms | 1.24–1.33× |
| kimi-k2-4x8k | 150.6–161.8 ms | 194.0 ms | 1.20–1.29× |

失准项：设计把**单价当作与 request 长度无关的常数**（2.97–3.19 µs/迭代）。实测单价随 request 长度分层
（§3.1：len≤5376 为 3.11–3.14、len=16384 为 4.26 µs/it），故 mixed（由 16384 请求主导）偏差最大。
该偏差在 Stage 4 的 winner（stages=3）下收窄到 1.31×（mixed）/1.03–1.10×（其余），见 §4。

### 1.3 采集的有效性判据

- 每次 capture 的 `captured_samples == per_call_launches`（7 / 7 / 7 / 5），日志 `window_ok=true`；
- `captured_op_name` 四元组固定为 `_bwd_main_kernel` / `_delta_main_kernel` / `_dq_land_main_kernel` /
  `ZerosLike_4_BYTES_high_performance_1`，即该调用的全部 launch，无框架小算子混入；
- 同一 case 的两次独立 capture（round0 与 round2/final）差 <0.3%（如 8x4k 168 511 → 168 556 µs）。

## 2. Iteration Log

### 2.0 调优排序与理由

按实测负担与结构收益排序：`ds-v3-mixed`（689 ms，目标缺口最大）→ `ds-v3-8x4k-bf16`（204 ms，cube 主路径）
→ `kimi-k2-4x8k`（194 ms，H=64/2 launch 的第二个 regime）→ `ds-v3-8x4k-fp16`（同几何的另一 dtype，最后）。
smoke 两个 case 不采性能。

### 2.1 Round 1 — 现象（基于 baseline 最新 profile）

1. **逐迭代 AIC 忙时与 workload 无关**（把 §1.1 的 share 乘以 `us/it`）：mad 0.76–0.87、mte1 0.69–0.73、
   mte2 1.09–1.22、fixpipe 0.60–0.62 µs —— 5 GEMM/2 段向量/1 次原子 store 的**工作量每迭代相同**，
   mixed 的 5.069 与 8x4k 的 3.832 之差不出现在 AIC 管道工作量上。
2. **派发是平衡的**（推翻「长尾/异质派发」猜测）：按 `block_id` 汇总每块 `aic_time`，`max/mean = 1.004–1.007`
   （8x4k / mixed / kimi 三个 case 全部），即硬件动态派发把每个核的累计工作量抹平到 0.7% 以内。
   设计 §5.5「逻辑块超发 + 动态派发压长尾」在本平台**成立**。
3. **差异跟着 request 长度走，不跟着异质走**（两个同构对照，baseline 实现）：
   - `probe-6x5376`（6 个等长 5376，token 数与 mixed 相同）：**3.830 µs/it**（与 8x4k 同）
   - `probe-1x16384`（单个 16384 请求）：**5.230 µs/it**（与 mixed 的 5.069 同）
   - `probe-1x16384-H16`（同请求、H=16，并发 head footprint 降 8×）：5.105 µs/it ⇒ **不是 L2 footprint**
     （三例 L2 读命中率 54.9–64.4%，没有量级差）；`mixed`/`1x16384` 的块时长比管道忙时多 30%。
4. **AIC 的 GM→L1 填充是延迟受限，不是字节/描述符受限**：把每迭代填充字节与 copy 条数同时减半
   （c1：去掉 qT/doT/kT 三条转置装载，82 432 B/8 条 → 40 960 B/2 条）后，mte2 忙时 **1.176 → 1.164 µs/迭代
   （−1%）**，active BW 反而从 79.3 掉到 45.2 GB/s —— 说明该 pipe 的时间被“单次填充的往返延迟”支配。
5. 基线实现里 `num_stages=2` 事实上没有跨迭代预取（设计 §6.3 已注明只有 `qT_s` 拿到双版本）。

→ 候选优化点（同一 base 派生，每支只改一个主要点）：

| 分支 | 优化点 | 预期 |
|---|---|---|
| c1 | 去掉 qT/doT/kT 三条转置 GM 装载，dV/dK/dQ 改用自然 (K,N) 操作数（`transpose_B=False`） | 每迭代 GM 填充 −50%、块级 k/v 装载 −37% |
| c3 | `num_stages` 2 → 3（查询块循环的真预取） | 用一层额外 slack 覆盖填充延迟 |
| c4 | `num_stages` 2 → 4 | 同上更深 |
| c5 / c6 / c7 | c1 + stages 3 / 4 / 5 | 组合（单点各自证明后再组合） |

**候选 vs current best（baseline）对比表**：

| candidate | 8x4k-bf16 T_op | Δ | 8x4k us/it | mixed T_op | Δ | mixed us/it | cube% (8x4k/mixed) | mte2 BW | dk/dv 逐位 | dq_acc | L0/parity |
|---|---:|---:|---:|---:|---:|---:|---|---|---:|---|---|
| baseline | 203 983.0 | — | 3.832 | 689 184.2 | — | 5.069 | 20.8 / 21.4 | 79.3 | — | — | Stage 3 门禁 |
| c1 nt-operands | 191 626.6 | −6.1% | 3.592 | 644 264.2 | −6.5% | 4.736 | 22.3 / 22.8 | 45.2 | **逐位相同** | 逐位相同（smoke） | PASS |
| **c3 stages=3** | **168 511.3** | **−17.4%** | **3.144** | **563 772.7** | **−18.2%** | **4.138** | **25.6 / 26.3** | 62.9 | **逐位相同** | 1e-3 内 | L0 28/28, L1 8/8, B 12/12, G 8/8 |
| c4 stages=4 | 172 179.4 | −15.6% | 3.215 | 574 108.5 | −16.7% | 4.215 | 25.0 / 25.8 | 68.8 | 逐位相同 | 1e-3 内 | parity PASS |
| c5 nt+stages3 | 169 036.3 | −17.1% | 3.153 | 565 949.4 | −17.9% | 4.154 | 25.5 / 26.0 | 45.1 | 逐位相同 | 1e-3 内 | parity PASS |
| c6 nt+stages4 | 184 676.0 | −9.5% | 3.457 | 617 964.1 | −10.3% | 4.540 | 23.1 / 23.7 | 50.9 | 逐位相同 | 1e-3 内 | parity PASS |
| c7 nt+stages5 | 176 003.8 | −13.7% | 3.289 | 593 154.5 | −13.9% | 4.356 | 24.4 / 25.1 | 46.2 | 逐位相同 | 1e-3 内 | parity PASS |

**winner / rollback**：c3（`num_stages=3`）在两个 workload、两个 dtype、kimi 上全部最优且回退最小；
c5 与 c3 打平（0.3–0.4% 内，属噪声），按「代码复杂度」决胜取 c3（保留设计原操作数布局，改动仅一个常量）。
c4/c6/c7 回退：更深流水使 L1 版本数上升、`aic_mte2_share` 掉回 29.6%（c6）而总时长变差 ⇒ 本结构上
`num_stages=3` 是拐点。c1 单独有效（−6%）但正对 mte2 延迟项无效（§2.1 第 4 条），组合不叠加 ⇒ 未合入。

**mechanism（为什么 stages=3 有效）**：base 的 `aic_mte2_share` 30.7% 且每迭代 mte2 1.18 µs 是串在
（mte2 → mte1 → mad → fixpipe → 向量 → UB→L1 → mte1）链上的；stages=3 让下一个查询块的 q/do 落在
前一迭代的 GEMM 之下，AIC 五管道之和 `mad 0.80 + mte1 0.68 + mte2 1.70 + fix 0.56 + scal 0.13 = 3.87 µs`
首次超过迭代墙钟 3.14 µs（即**出现真实重叠**；baseline 是 3.26 vs 3.83，无重叠）。

**合并非回退检查（本 kernel 全部 tune workload）**：c3 在 8x4k-bf16 / 8x4k-fp16 / mixed / kimi 四个
workload 上分别为 −17.4% / −17.4% / −18.2% / −17.6%，无一回退；smoke 两个 case 精度见 §2.5。
c3 即已合并 current best → `perf_opt/MLAVarlenBwdOp.py`（与 `_round1_c3.py` 字节相同，md5
`6e2e693565b238c1cec71c268cab3170`）。

### 2.2 Round 2 — 是否还有结构收益（终止判定前的参照结构 diff）

| 检查 | 结果 |
|---|---|
| `num_stages` 1/2/3/4/5 全扫 | 3 是拐点（表见上），更高反而回退 |
| 操作数布局（转置副本 vs 自然布局） | c1 有效但与 stages 不叠加；mte2 时间不受字节影响 ⇒ 该项已到结构下限 |
| 同族参照（`examples/mla_dense_bwd/_bwd_kernel`，同一 5-GEMM/L0 结构、静态 extent） | dense 生产实测 2.97 µs/it（S=2048）；本 kernel 在 len≤5376 已达 3.11–3.14 ⇒ **与参照同量级**，剩余差在 16384 长度区间，参照本身从未覆盖该区间 |
| 派发/尾波 | 逐核累计工作量 max/mean ≤1.007 ⇒ 无长尾可收 |
| 逐迭代 AIC 工作量 | 与 workload 无关（§2.1）⇒ 已无“重复工作”可删 |
| 剩余候选（改 block_n/block_m、query 切分、静态 extent 分桶、块序重排） | 均属**任务映射/设计层**：block_n/block_m 由契约冻结（grid slab + 断言不变）；块序重排对 `1x16384`（同构）同样慢 ⇒ 与异质性无关，无收益；静态 extent 分桶需 host 侧按 len_q 分组多次 launch，改变三 launch 契约 |

结论：冻结结构内的收益已收束；mixed 的剩余缺口（563.7 → 目标 500，需 −13%）需要改填充粒度/工作集驻留的
结构（设计层），按 §5 走 `[DESIGN_LIMIT]` 反馈，不在本轮擅自改设计。

### 2.3 关键诊断证据（platform）

| 证据 | 数值 | 来源 |
|---|---|---|
| 逐核 `Σ aic_time` 均衡度 | max/mean 1.004（8x4k）/ 1.007（mixed）/ 1.006（kimi） | `profiles/round0/*_baseline/**/PipeUtilization*.csv` 逐块 `aic_time` |
| mixed 的 30% 非管道时间 | 块墙钟 436.6 µs vs `aic_time` 307.5 µs（AIV 侧 310.2 µs，同幅） | 同上 |
| 长度分层（同构对照，stages=2） | 6×5376 = 3.830；1×16384 = 5.230 µs/it | `profiles/probe-6x5376`, `probe-1x16384` |
| 长度分层（stages=3） | 6×5376 = 3.108；1×16384 = 4.264 µs/it | `profiles/round2/probe-*_c3` |
| L2 读命中（不构成量级差） | 8x4k 64.4% / mixed 59.4% / 6x5376 61.7% / 1x16384 54.9% | `**/L2Cache*.csv`（cube 行均值） |
| 填充延迟受限 | c1 把填充字节与条数各减半，mte2 忙时 −1%，active BW 79.3→45.2 GB/s | `profiles/round1/*_c1.json` |

### 2.4 Autotune Log

未使用 autotune（TileOPs 侧 `tune` 策略在本 op 已移除，tiling 由工厂冻结为编译期常量；
`factory_config()` 的 `block_n/head_slab` 属 launch-plan 契约，本轮不动）。
参数扫描以手动分支形式完成并全部记录在 §2.1 表内（`num_stages` 1–5 × 操作数布局 2 种）。

### 2.5 精度回归（每个候选分支 + 最终版本）

| 检查 | 命令 | 结果 |
|---|---|---|
| 与 Stage 3 基线逐位对齐 | `perf_opt/verify_parity.py`（baseline 与 candidate 同进程、同输入） | smoke / 8x4k-H32 / mixed-H16 / ragged(含 len_q=0、shift 双向、尾块) 四例：**dk/dv 逐位相同**；dq_acc 仅 store-atomic 到达序抖动（max_abs 4.9e-4 @ \|ref\|≈3，且 baseline 自身两次运行同样抖动） |
| L0（28 例，含 packed 尾块/shift 双向/len_q=0/atomic 契约/arm 等价/launch plan） | `python perf_opt/MLAVarlenBwdOp.py --level L0` | **28/28 PASS**，max_diff 0.128（与 Stage 3 同一判据与量级） |
| L1（8 例，含 smoke 单 launch、H=64 多 launch） | `--level L1` | **8/8 PASS**，max_diff 0.113 |
| boundary（12 例） | `--level boundary` | **12/12 PASS** |
| G（H=128 真形：8x4k、mixed、H=126/127 slab 映射，逐 request） | `--level G` | **8/8 PASS**，max_diff 0（逐 request 全对） |
| smoke（benchmark 两个 smoke case） | L0-1 / L0-2 + parity | **PASS**（`precision_pass=true`） |

日志：`perf_opt/logs/precision/{L0,L1,boundary,G}_c3.log`（分支 c3，与最终文件字节相同）；最终文件本身的 L0 复跑在 `perf_opt/logs/precision/L0_final.log`（28/28 PASS，max_diff 0.127985，372.6 s）。

## 3. 关于测量口径的两条说明（供后续任务复用）

1. **`--launch-count` 数的是“launch”不是“call”**：`msopprof --warm-up/--launch-count` 以进程内的
   kernel launch 计数（实测 `--launch-count=10` 恰好抓到 10 次 launch）。一个 call 含 4–7 次 launch
   （随 head slab 数变化），故本任务的采集器按 `plan.launches+3` 计算每次调用的 launch 数，
   `--warm-up=(1+W)*K --launch-count=K` 抓**恰好一次调用**，并用 `window_ok` 逐次校验。
   直接用生产默认 `--launch-count=10 --warm-up=5` 会横跨两次调用（本次实测 main 出现 n=5 个样本）。
2. **`T_op` 是**逐 launch 求和**，不是逐 op 取中位数**：生产报告的 `t_op_breakdown['_bwd_main_kernel']`
   = 196 908 µs 是 4 次 slab launch 之和（单次 49.2 ms）。本日志同口径。
   另：`aic_cube_ratio` 的“逐块 ratio 均值”在异质块下会被偏差（mixed 的 3.5 vs 256 迭代块），
   `perf_records.jsonl` 同时给出 sum 口径的 `aic_*_share`，二者的差别只在 mixed 上超过 1%。

## 4. Final Summary

最终版本：`perf_opt/MLAVarlenBwdOp.py`（= Round 1 winner c3，`_STAGES = 3`），
sha256 `4e8fcb7d102b9f64be8d12046c0223afa8eec028c0a417bddc9890cbe32aadef`，
candidate_id `final-c3-stages3`（4 个 tune workload 的 final 记录全部来自该文件，同一 sha256）。

### Final Performance Test Data

| kernel_id | workload_id | baseline_us | final_us | final_candidate_id |
|---|---|---:|---:|---|
| tileops/kernels/attention/mla_varlen_bwd/mla_varlen_bwd_kernel::_bwd_kernel+_delta_kernel+_dq_land_kernel | ds-v3-8x4k-bfloat16 | 203983.046 | 168556.109 | final-c3-stages3 |
| tileops/kernels/attention/mla_varlen_bwd/mla_varlen_bwd_kernel::_bwd_kernel+_delta_kernel+_dq_land_kernel | ds-v3-8x4k-float16 | 203948.105 | 168551.550 | final-c3-stages3 |
| tileops/kernels/attention/mla_varlen_bwd/mla_varlen_bwd_kernel::_bwd_kernel+_delta_kernel+_dq_land_kernel | ds-v3-mixed-bfloat16 | 689184.203 | 563712.107 | final-c3-stages3 |
| tileops/kernels/attention/mla_varlen_bwd/mla_varlen_bwd_kernel::_bwd_kernel+_delta_kernel+_dq_land_kernel | kimi-k2-4x8k-bfloat16 | 194026.632 | 159757.058 | final-c3-stages3 |

提升：8x4k-bf16 **−17.4%（1.21×）**、8x4k-fp16 **−17.4%（1.21×）**、mixed **−18.2%（1.22×）**、
kimi **−17.7%（1.21×）**（四行均来自同一最终文件的同一 candidate，`artifact_sha256`
`4e8fcb7d102b9f64be8d12046c0223afa8eec028c0a417bddc9890cbe32aadef`）。

final 的逐 workload 分解：

| workload | T_op (µs) | main (µs) | us/it | cube%: mad/mte1/mte2/fix/scal | vec%: vec/mte2/mte3 | 目标 | 判定 |
|---|---:|---:|---:|---|---|---|---|
| ds-v3-8x4k-bf16 | 168 556.1 | 161 908.1 | 3.144 | 25.6 / 21.5 / 54.1 / 17.9 / 4.2 | 45.2 / 16.0 / 4.3 | ≤200 ms, cube ≥22%, vec ≤55% | ✅ 全部达成 |
| ds-v3-8x4k-fp16 | 168 551.5 | 161 907.9 | 3.144 | 25.6 / 21.5 / 54.1 / 17.9 / 4.2 | 45.2 / 15.9 / 4.3 | 同上 | ✅ |
| ds-v3-mixed | 563 712.1 | 557 167.3 | 4.138 | 26.3 / 22.3 / 56.1 / 18.7 / 3.8 | 44.8 / 17.9 / 4.1 | ≤500 ms, cube ≥22%, vec ≤55% | ⚠️ **520 ms 级未达（563.7）**，cube/vec 达成 |
| kimi-k2-4x8k | 159 757.1 | 156 409.3 | 3.084 | 26.1 / 22.5 / 63.3 / 17.9 / 3.8 | 44.8 / 21.4 / 4.3 | 不回退 | ✅ −17.7% |

对照 DESIGN.md §8.4「必达」门（相对 Cube 化前基线 1728.7 / 490.6 / 470.5 ms）：
mixed **3.07×**（≥2.8 ✅）、8x4k **2.91×**（≥2.5 ✅）、kimi **2.94×** ✅。
「预期」档（mixed 4.0–4.3×、8x4k 3.0–3.2×）未达，原因见 §2.1/§2.2 与 `perf_feedback.md`。

### smoke 最终精度

`mla-varlen-bwd-smoke-bfloat16` / `-float16`：L0-1 / L0-2 PASS，parity `PARITY_PASS`（dk/dv 逐位，
dq_acc 1e-3 内）→ `precision_pass = true`。

### 中止原因

`plateau`：连续 3 轮无超噪声提升的路径已覆盖（Round 1 六分支 × 两 workload，Round 2 参照结构 diff 与
剩余候选全部归为设计层）。混合 `mixed` 未达 500 ms 目标，其余三项达成，无回退。

## 5. Skill Retrospective

### 回写说明

两条 D 类实测常数已按 ED-C 追加到平台知识库 `knowledge/constants.md`
（`CONST-950-cube-queryloop-stages3-knee`、`CONST-950-packed-unitprice-vs-request-length`，
含 origin_task / toolchain_stamp / provenance 与边界声明）。最小 repro **尚未沉淀**（标 `repro-missing`）：
本算子的可复现最小形态需要从主 pass 裁出「5 GEMM + 双段向量 epilogue + 原子 store」的单 tile
（stages 扫描）与「两个定长 request 同 capture」（单价分层）两个骨架，属独立开发工作量，待补。
`knowledge/bottlenecks.md` 现存「尚缺的真机材料」清单中的前两项（按 shape/搬运路径的有效带宽与时延、
把采集指标对应到慢→快代码变化的案例）由本轮回填的条目直接覆盖。

### 流程问题

| area | issue | evidence | 建议 |
|---|---|---|---|
| 测量口径 | `msopprof --launch-count/--warm-up` 的真实语义（按 kernel launch 计数、跨行调用窗口）在现有 profile-collection 参考里没有写，`bench_kernel_msprof` 生成脚本的 `warm_up+launch_count` 次调用会产出**跨多次调用**的样本；多 launch 算子的 `T_op` 若按“逐 op 中位数”聚合会低估 3–4×（本例 `_bwd_main` 有 4 个 slab），而生产报告实际用的是逐 launch 求和。 | 本例：`--launch-count=10 --warm-up=5` 得到 main n=5（跨两次调用）；`perf_opt/collect.py` 按 `K=launches+3` 抓一次调用后 `window_ok=true`，且与生产报告 `t_op_us` 差 <0.3% | 在 `references/profile-collection.md` 增加「多 launch 算子：按每次调用的 launch 数整除 warm-up/launch-count，并以 `captured_samples == launches_per_call` 校验」与「`T_op` = 逐 launch 求和」两条 |
| 诊断口径 | 逐块 `*_ratio` 的均值在异质块 workload 上会误判（mixed 的 ratio 与 8x4k 几乎相同，但单价差 32%）：ratio 的分母是各自块的时长，长短块权重不同 | §3 第 2 条；sum 口径 `aic_*_share` 才与总时间可对账 | `references/iteration-diagnosis.md` 补「异质块 workload 用 `Σ pipe_time / Σ core_time`，不用 ratio 均值」 |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp |
|---|---|---|---|---|
| **`num_stages=3` 是本 Cube 结构（5 GEMM/单版本 L0/单版本 UB 半区）的流水拐点**：AIC 五管道之和首次超过迭代墙钟（3.87 > 3.14 µs，即出现真实重叠），8x4k −17.4%、mixed −18.2%、kimi −17.6%；`num_stages=4/5` 因 L1 版本压力回退（172.2 / 176.0 ms） | D | **已回写 `knowledge/constants.md#CONST-950-cube-queryloop-stages3-knee`**（repro-missing 待补）；`perf_opt/perf_records.jsonl`（r1-c3 / r1-c4 / r1-c7 行）；`profiles/round1,round2/*_c3,c4,c7.json` 的 `aic_mte2_share` 54%→47.9%→38.7% | `python perf_opt/collect.py --impl perf_opt/MLAVarlenBwdOp_round1_c4.py --case ds-v3-8x4k-bfloat16`（对照 c3） | tilelang 0.1.15 / Ascend950PR_9579 / CANN 9.2.1 |
| **packed 变长的单价由 request 长度分层**：同构 6×5376 = 3.11 µs/it vs 单个 16384 = 4.26 µs/it（stages=3 下），逐迭代 AIC 管道工作量两者相同；与“异质派发/长尾”无关（逐核累计工作量 max/mean ≤1.007）、与 L2 footprint 也无关（H 降 8× 不变快，L2 命中率 55–64% 无级差） | D | **已回写 `knowledge/constants.md#CONST-950-packed-unitprice-vs-request-length`**（repro-missing 待补）；`perf_records.jsonl` + `profiles/probe-*`；`profiles/*/L2Cache*.csv`；逐块 `aic_time` 汇总 | `python perf_opt/collect.py --impl baseline --case probe-1x16384 --custom '{"q_lens":[16384],"heads":128}'` | 同上 |
| **Cube 算子为 `transpose_B` 保留的转置 L1 副本可以整项删掉**：dV/dK/dQ 用自然 (K,N) 操作数（`transpose_B=False`）后 dk/dv **逐位不变**，每迭代 GM 填充 82 432 → 40 960 B、块级 k/v 装载 −37%、L1 占用 204 800 → 126 976 B；但 mte2 忙时只降 1%，说明 Cube 的 GM→L1 填充在本平台是**往返延迟受限**而非带宽/描述符受限 | P | `perf_opt/verify_parity.py`（四例 dk/dv 逐位）；c1 T_op 191 626.6 vs baseline 203 983.0；`aic_mte2_active_bw` 79.3 → 45.2 GB/s | `python perf_opt/verify_parity.py --case mla-varlen-bwd-smoke-bfloat16 --cand perf_opt/MLAVarlenBwdOp_round1_c1.py` | 同上 |
| 多 launch 算子的 Stage 4 采集配方：`collect.py` 从 `plan_launch` 推 `K=launches+3`，抓一次调用，`T_op=Σ`，并同时落 `mte2/mte1/cube share` 与 `us_per_iteration`，可直接与 DESIGN 的单价投影对账（本例复现生产报告 <0.3%） | P | `perf_opt/collect.py`、`perf_records.jsonl`（22 行，含 baseline/candidate/final 三 phase） | 同 §0 表 | 同上 |

## 6. 结论

冻结结构内已收束：`num_stages=3` 一个常量换来 17–18% 的整体提速，四 workload 无一回退，
精度逐位/容差不变（dk/dv 逐位、dq_acc 仅原子抖动）。mixed 的 500 ms 目标需要设计层改动
（填充粒度/工作集驻留/query 切分），已按 `[DESIGN_LIMIT]` 写入 `perf_opt/perf_feedback.md` 交 conductor 路由。

## 7. 交付布局与 wrapper 切换说明

- 最终版本：`perf_opt/MLAVarlenBwdOp.py`（自包含：主 pass + launch plan 助手 + golden + L0/L1/boundary/G 测试入口，
  `python perf_opt/MLAVarlenBwdOp.py --level L0|L1|boundary|G|all` 可直接跑精度门禁）。
- `delta` / `dq_land` 两个 launch 未改动，最终版本通过 baseline 包复用它们（本目录副本仅作对照）。
- 为让 wrapper 的 baseline/perf_opt 双 import 切换块可直接翻转，已按 skill 约定在算子目录内放置同构 drop-in 包：
  `examples/TileOPs/tileops/kernels/attention/mla_varlen_bwd/mla_varlen_bwd_kernel/perf_opt/`
  （`__init__.py` + `_bwd_kernel.py` 与本文件 **字节相同** md5 `6e2e693565b238c1cec71c268cab3170`，
  `_delta_kernel.py` / `_dq_land_kernel.py` 转出 baseline 实现）。切换时**两条 import 必须一起换**
  （三个工厂 + `build_kernel/describe/factory_config/plan_launch`），否则 plan 与产物会来自两份源码；
  已验证切换后 `factory_config() = {block_n:96, block_m:64, num_stages:3, head_slab:32}` 且
  `plan_launch(8, 32768, 128)` 仍是 `tiles=349 slab=32 launches=4 max_grid=11168 idle=0`（grid/slab/断言契约不变）。
- 本目录同时保留：`MLAVarlenBwdOp_baseline_stage3.py`（只读基线副本）、`MLAVarlenBwdOp_round1_c{1,3,4,5,6,7}.py`
  （实验分支）、`collect.py` / `mla_bwd_case.py` / `summarize.py` / `write_records.py` / `verify_parity.py` /
  `make_candidate.py` / `capture_baseline.sh`（采集与校验工具）、`profiles/`、`logs/`。

## 8. 生产采纳（production adoption，2026-11-25）

### 8.1 采纳清单

| 文件 | 变更 | 证据 |
|---|---|---|
| `examples/TileOPs/tileops/kernels/attention/mla_varlen_bwd/mla_varlen_bwd_kernel/perf_opt/_bwd_kernel.py` | 新增（= 本目录 `MLAVarlenBwdOp.py`，md5 `6e2e6935…` / sha256 `4e8fcb7d…`，字节相同） | `md5sum` 对比 |
| `…/perf_opt/_delta_kernel.py`、`…/perf_opt/_dq_land_kernel.py` | 新增（转出 Stage 3 实现，未改） | 同上 |
| `…/perf_opt/__init__.py` | 新增（drop-in 包导出 + 切换说明） | — |
| `…/mla_varlen_bwd.py` | **只改切换块两条 import**（三工厂 + `build_kernel/describe/factory_config/plan_launch` 同源）；head slab / `h_off` 多 launch / 三 launch 契约未动 | `diff history_version/mla_varlen_bwd_pre_opt_stages3.py` 仅 108–135 行 |
| `…/history_version/mla_varlen_bwd_pre_opt_stages3.py` | 切换前备份（md5 `3caa13a6…`） | — |

Stage 3 kernel 目录（`_bwd_kernel.py` md5 `4262fe4b…`）、op/manifest/test/bench 均未改动。
核对：`factory_config() = {block_n:96, block_m:64, num_stages:3, head_slab:32}`（perf_opt 源），
`plan_launch(8, 32768, 128)` 仍 `tiles=349 slab=32 launches=4 max_grid=11168 blocks=44672 useful=44672 idle=0`。

### 8.2 生产回归（cwd `examples/TileOPs`）

| 检查 | 命令 | 结果 |
|---|---|---|
| 单测 | `pytest tests/ops/test_mla_varlen_bwd.py -q` | **12 passed**（146 s） |
| 逐位/容差（对 Stage 3 基线，4 例：smoke / 8x4k-H32 / mixed-H16 / ragged 含 len_q=0+双向 shift+尾块） | `perf_opt/verify_parity.py --cand …/perf_opt/_bwd_kernel.py` | **dk/dv 逐位相同**（0 差异）；`dq_acc` 仅 store-atomic 到达序抖动（≤2.4e-4，基线自身两次运行同样抖动），**0 个单元超出 DESIGN 8.2 容差** |
| 8x4k 非静默（H=128 真形） | `perf_opt/check_nonsilent.py` | **NONSILENT_PASS**：request 0..7 的 dk/dv 每个 (row,head) 单元非零（各 524 288/524 288）；`dq` 仅允许的「单 key 行 dS=0」丢 70–80 个单元；尾 slab head 96/112/126/127 全非零；plan `blocks == useful`、`idle = 0`、`launches*max_grid = 44 672 ≤ 65 535` |

### 8.3 单算子报告（`tileops.reporting.cli run --op MLAVarlenBwdOp --prof-mode msprof`）

- run-id：**`20261125_083436_168731_MLAVarlenBwdOp`**（`examples/TileOPs/reports/tileops/<run-id>/`）
- `status = passed`；`correctness_passed = true`（12/12）；`benchmark_passed = true`；
  `invalid_acceptance_count = 0`、`invalid_profile_count = 0`、`profiler_fallback_count = 0`、`benchmark_failed = 0`
- 口径：`acceptance.unit = cube`（`peak_used.compute_tflops = 378`），MFU = roofline FLOPs / 378 TF/s

| case | baseline `t_op_us` | 生产 `t_op_us` | 变化 | MFU 基线 → 生产 |
|---|---:|---:|---:|---|
| ds-v3-8x4k-bfloat16 | 203 548.7 | **168 417.1** | −17.3% | 18.64% → **22.53%** |
| ds-v3-8x4k-float16 | 203 564.0 | **168 518.0** | −17.2% | 18.64% → **22.51%** |
| ds-v3-mixed-bfloat16 | 689 090.0 | **563 659.8** | −18.2% | 14.68% → **17.94%** |
| kimi-k2-4x8k-bfloat16 | 194 260.4 | **159 028.5** | −18.2% | 19.53% → **23.85%** |

breakdown（mixed）：`_bwd_main 557 111.1 / _dq_land 3 223.0 / ZerosLike 1 928.4 / _delta 1 397.3`。
与 Stage 4 采集（168 556.1 / 168 551.6 / 563 712.1 / 159 757.1）逐项差 <0.5%，口径一致。
无回退（四个 case 全部提速，cube MFU +3.3~+4.3pt；`acceptance.failed = 4` 与该算子声明 MFU 阈值有关，
采纳前基线报告同为 `passed 0 / failed 4 / invalid 0`，非本轮引入）。

### 8.4 记录

`perf_records.jsonl` 追加 4 行 `phase=merged` + `stage=production`（`candidate_id=prod-wired-perf_opt-c3`，
`parent_id=final-c3-stages3`，含 run-id、report 路径、wrapper/prod 的 sha256、MFU、invalid 计数）。
