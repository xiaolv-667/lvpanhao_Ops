# Ascend950PR_9579 实测代价常数（Stage 4 回填）

条目按 [knowledge-entry-schema.md](../../../../skills/_shared/standards/knowledge-entry-schema.md) 登记；
本文件只记**真机实测**常数，不引用型号 INI 的频率字段换算值。

---
id: CONST-950-gemm-mte2-burst
kind: constant
status: verified
origin_task: shared_expert_mlp-SharedExpertMLPFwdOp-20261124T132107Z (Stage 4)
toolchain: TileLang ascend 0.1.15 + CANN 9.2.1, 2026-11-24
toolchain_stamp: 7a52f2f1274df07d8f4c9c01353ce9d290f88c16e85047de0c425b6a009de4aa
toolchain_id: ascend
hardware_id: ascend-950pr-9579
repro: repro/PL-950-gemm-l1-fill.py
---

# bf16 L1 操作数 GEMM 的 MTE2 有效带宽取决于每行 DMA 段长度

被测算子：`C = A[m,k] @ B[n,k]^T`，A/B 均按 `(rows, k)` 存放，`transpose_B=True` 的
L1 操作数路径（`ascend_gemm_l1` 模板 + `T.Pipelined` 多版本 L1），fp32 L0C 累加，
`m=512, n=36864, k=7168, bf16`，`block_m=256, block_n=256, num_stages=3`。

| block_k | 每行 DMA 段 | `aic_mte2_active_bw` | gate/up 时延 |
|---:|---:|---:|---:|
| 64 | 128 B | 70 GB/s/core | 1201.6 µs |
| 128 | 256 B | 101–105 GB/s/core | 811.8 µs |
| 256 | 512 B | — | 818.9 µs（`num_stages=2`，L1 恰好 512 KiB） |

- 聚合口径：28 Cube 核 × 101–105 GB/s ≈ **2.7–2.9 TB/s** 的 L2→L1 填充带宽（MTE2 满载时）。
- 对照常数：HBM 有效带宽 **1.42–1.48 TB/s**（`kimi-k2-t1`：权重 792 MB / 535.9 µs = 1.48 TB/s，
  权重只读一遍、无复用）。
- 对照常数：bf16 Cube 有效峰值 **356–359 TF/s**（256×256 tile、k=18432 的 down GEMM，
  `aic_cube_ratio=0.978` × 367.6 µs = 358 µs，而 378 TF/s 的名义下界是 357 µs）。

边界：结论限于 bf16/fp16 L1 操作数 NT GEMM + 本工具链的 `ascend_gemm_l1` 模板；`block_k=256`
只测到「不比 128 快」，未验证更大 K 块（L1 预算不允许 `block_k ≥ 192` 配 `num_stages ≥ 3`）。
`aic_mte2_active_bw` 是 PipeUtilization CSV 的逐 block 字段，本表取其块间均值。

---

id: CONST-950-kernel-launch-floor-48blk
kind: constant
status: verified
origin_task: engram_gate_conv-engram_gate_conv-20261124T171256Z (Stage 4)
toolchain: TileLang ascend 0.1.15 + CANN 9.2.1, 2026-11-24
toolchain_stamp: 7a52f2f1274df07d8f4c9c01353ce9d290f88c16e85047de0c425b6a009de4aa
toolchain_id: ascend
hardware_id: ascend-950pr-9579
repro: repro/TRAP-950-rowload-dma-chain.py
provenance: 溯源见 origin_task；`engram_gate_conv_perf/perf_opt/opt_log.md` §1.3/§2/§4，
  `perf_opt/experiments/floor_probe.py` kind=0，原始采集 `perf_opt/profiles/floor/*`
apis: [T.Kernel, T.copy]
dtype: [bfloat16]
---

# 小算子 launch 地板：48 块一维 grid ≈ 2.7–2.8 µs（Task Duration 口径）

单 launch、`T.Kernel(48)`、d=768、bf16，每块只做 1 条行装载 + 2 条行落盘：msopprof
`Task Duration(us)` = **2.72 / 2.75 / 2.83 µs**（三次独立采集，另一次 3.44 µs 为同分布上沿）。
这既是该几何的**地板**，也是「多一次 launch 的代价」：两段 launch 的结构天然多付 ≈2.7 µs。

用途：判「两段 launch 备选是否可能胜出」。本算子的设计层不确定常数①（DESIGN 假设 2–5 µs）由此锚定为
2.7 µs，两段结构因此在 `train-s96` 上实测 9.99 µs vs 单 launch 8.46 µs（+18%），备选被否。

边界：地板随 grid 与 dtype 变化未系统扫描；`grid=1` 的同一探针给出 5.7–6.0 µs（与 grid=48 反向），
说明 1-D grid 的调度形态在极小 grid 下另有开销，**不要**用 grid=1 的 kernel 测地板。
口径是同任务平台的 kernel-only Task Duration，不是端到端时延，也不是 NPU event。

---

id: CONST-950-launch-floor-grid-sweep
kind: constant
status: verified
origin_task: engram_decode-engram_decode-20261124T171256Z (Stage 4)
toolchain: TileLang ascend 0.1.15 + CANN 9.2.1, 2026-11-24
toolchain_stamp: 7a52f2f1274df07d8f4c9c01353ce9d290f88c16e85047de0c425b6a009de4aa
toolchain_id: ascend
hardware_id: ascend-950pr-9579
repro: repro/CONST-950-launch-floor-grid-sweep.py
provenance: 溯源见 origin_task；`engram_decode_perf/perf_opt/opt_log.md` §2.2/§2.3/§5.3，
  `perf_opt/experiments/probe_floor.py`，原始采集 `perf_opt/profiles/floor/*`、
  `perf_opt/profiles/probe/*`
apis: [T.Kernel, T.copy]
dtype: [float16, bfloat16]
---

# launch 地板随 grid 的补扫（4/8/16 块）+ 两段 launch 算子地板 + mix 提升代价

补齐 `CONST-950-kernel-launch-floor-48blk`（grid=48）在**小 grid** 上的空白：同形态最小 kernel
（每块 1 条行装载 + 1 条行落盘，行宽 512 fp16，无其他工作），msopprof `Task Duration(us)`：

| grid | 2026-11-24 三采中位 | 2026-11-25 复跑（repro） |
|---:|---:|---:|
| 4 | **2.690** | 2.711 |
| 8 | **2.961** | 2.766 |
| 16 | **2.894** | 2.770 |

- **grid 4–48 的地板都在 2.7–3.0 µs 区间**（grid=1 例外，见上一条边界），因此
  **两次 launch 的算子结构性地板 = 5.4–5.9 µs**。解码类小算子的 T_op 若已进入 8–9 µs 区间，
  地板项通常占 65–70%，此时「再多一个 pass / 再省一条 DMA」的量级只有 0.1–0.5 µs。
- **mix(AIC+AIV) 提升代价 ≈ 0.49 µs**（同几何、同 grid）：同一 project kernel 去掉全部 AIV 代码后
  serving-b4 由 4.878 → 4.199 µs；把 AIV 侧缩成 1 条 1 行 copy（字节量降到 1/16）仍要 4.693 µs。
  即「只要块内存在 AIV 工作」就付 ≈0.49 µs 固定代价，其后的 AIV 字节按 ≈1.2 µs/MB（聚合）计。
  该值仅在本条目的矩阵形状（16 块 × `block_n=64` × `block_k=256`）下实测。

边界：fp16/bf16、1 维 grid、单波；不同 occupancy 与多波形态未扫。口径是 kernel-only
Task Duration，不是端到端时延、也不是 NPU event。

---
id: CONST-950-atomic-tier-cost
kind: constant
status: verified
origin_task: moe_permute_align-moe_permute_align-20261124T171256Z (Stage 4)
toolchain: TileLang ascend 0.1.15 + CANN 9.2.1, 2026-11-25
toolchain_stamp: 7a52f2f1274df07d8f4c9c01353ce9d290f88c16e85047de0c425b6a009de4aa
toolchain_id: ascend
hardware_id: ascend-950pr-9579
repro: repro/CONST-950-atomic-tier-cost.py
provenance: 溯源见 origin_task；`examples/moe_permute_align_perf/perf_opt/opt_log.md` §2（机制探针）、
  `perf_opt/logs/probe/mechanisms2.log`、原始采集 `perf_opt/profiles/r0_base/msprof/*` 与
  `perf_opt/probes/probe_mechanisms.py`
apis: [T.atomic_add, T.SimtVF, T.Parallel]
dtype: [int32]
---

# 逐 lane 原子/散写的层级代价（GM 原子 ≈1465 cyc vs UB 原子 ≈5 cyc/路由）

同一几何（numel=32768 条路由、56 核 × 128 SIMT lane、每 lane 5 条路由，int32），
`msprof` kernel `Task Duration(us)`：

| 探针（单机制单 launch） | `Task Duration` | 每路由代价（增量） |
|---|---:|---:|
| `contiguous`：每路由写自己下标 | 2.21 | 基线 |
| `scattered`：每路由写在 32768 长数组中打散的地址 | **6.70** | **+14 cyc** |
| `ub_atomic`：每路由对 **UB** 游标 `atomic_add(+1, prev)` + 连续写 | **4.05** | **+5 cyc** |
| `gm_atomic`：每路由对 **GM** 游标 `atomic_add(+1, prev)` + 打散写（= 计数器 bump 分配器） | **586.03** | **≈+1465 cyc** |
| `hist_compare`：直方图「每 token × E 个 lane 比较累加」 | 29.08 | 每 element-iter ≈82 cyc |
| `hist_atomic`：直方图「每路由一次 UB 原子自增」 | **3.79** | O(numel) |

结论与判据：

1. **同一条算法，原子所在层级决定 150×**：`gm_atomic` 586.03 vs `ub_atomic` 4.05。GM 原子在
   SIMT 逐 lane 形态下被串行化（每 lane 一次往返、无流水），UB 原子近乎免费。
   ⇒ 任何「每元素一次 GM 原子」的分配器/计数器（bump allocator、直方图、去重标记）在 950PR 上
   都是每元素 ~0.9 µs 量级，必须先换算再选结构。
2. **逐 lane 数据相关 GM *写* 很便宜**（+14 cyc/路由）：散射写本身不是瓶颈，别为了「避免散写」去
   做 UB 暂存（实测 UB 暂存 + 整块 DMA 出 = 15.54 µs，比直写 GM 的 6.70 µs 更慢）。
3. **直方图优先用「每路由一次 UB 原子」**，而不是「expert 维 lane 比较累加」：29.08 → 3.79 µs。
   8 条独立累加链的展开只能到 11.57 µs（2.5×），仍劣于原子形态。

边界：int32、1 维 grid、`T.SimtVF(threads=128)`、单波 56 块；未测 fp32 原子、Cube 侧原子、
`T.Parallel` 宽度 32/64 的形态，也未测跨核争用（本组测量每核写自己的缓冲，无跨核热点）。
口径是 kernel-only `Task Duration`，不是端到端时延、也不是 NPU event。
