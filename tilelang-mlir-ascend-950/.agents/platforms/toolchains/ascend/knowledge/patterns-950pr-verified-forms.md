# 950PR 已实测可用形态（batch2）

本文件承载 ascend 工具链在 950PR 上实测**可用且语义已确认**的构造：AIV 子块号读取、UB 标量索引、
行标量广播、动态界流水多版本、K 尾掩蔽、多版本清零与多缓冲启用条件。条目按实测值记录适用条件，
复现脚本在 `repro/`（自包含，头部记录条目 id 与首验版本戳）。

---
id: CONST-950-aiv-sid-capture
kind: pattern
status: verified
origin_task: batch2-20261123 (mla_dense_bwd / _bwd_kernel Stage 3)
toolchain: TileLang ascend 0.1.15 + CANN 9.2.1, 2026-11-23
toolchain_stamp: a010a95909e4e752d0d0b602885eca4b6fab233edca2d3e540248ed71c2135dc
toolchain_id: ascend
hardware_id: ascend-950pr-9579
repro: repro/CONST-950-aiv-sid-capture.py
provenance: 溯源见 origin_task；rows[0,32) 得 sid=0、rows[32,64) 得 sid=1000 与 L0-12 记录在 mla_dense_bwd/_bwd_kernel/RETROSPECTIVE.md Stage 3
apis: [T.call_extern, T.SimtVF, T.dual_copy]
dtype: [float16, bfloat16, float32]
---

# 混合 kernel 取 AIV 副块号的可用形态：kernel body 内 `asc_get_sub_block_id`

`T.call_extern("int64", "asc_get_sub_block_id")` 写在 kernel body（自动调度会把它落进
`if ASC_IS_AIV` 并作为 VF 实参传入）时，取到的值等于 `dual_copy` 的分片 id：单块 `[64]/[64]`
的 rows[0,32) 得 sid=0、rows[32,64) 得 sid=1000（行号编码进操作数），两个半区各自与 golden 一致。

不可用/受限的相邻形态：

- `with T.Vector(vector=2) as sid` 编译通过但运行值错且非确定（见 TRAP-950-aiv-sid-vector-frame）；
- `asc_get_sub_block_id()` 不能写在 `T.SimtVF` 内（`__simt_vf__` 无匹配重载）。

边界：结论限于「自动调度 `T.Kernel` + 显式 Vector 帧 + `dual_copy` 2:1」的混合结构；与
UB→L1 软件 dual_copy 混用会触发 SSA `dup sid`，不在本条覆盖范围。

---
id: CONST-950-ub-value-as-gm-row-index
kind: pattern
status: verified
origin_task: batch2-20261123 (rope / _rope_positions_kernel Stage 3)
toolchain: TileLang ascend 0.1.15 + CANN 9.2.1, 2026-11-23
toolchain_stamp: a010a95909e4e752d0d0b602885eca4b6fab233edca2d3e540248ed71c2135dc
toolchain_id: ascend
hardware_id: ascend-950pr-9579
repro: repro/CONST-950-ub-value-as-gm-row-index.py
provenance: 溯源见 origin_task；`cs_ub[i,k] = COS[row_ub[i], k]`、块级 if/else 双分支与 out_idx 变体在一个 kernel 内断言，记录在 rope/_rope_positions_kernel/RETROSPECTIVE.md Stage 3
apis: [T.Parallel, T.macro, T.SimtVF, T.dynamic]
dtype: [float16, bfloat16, float32]
---

# UB 标量缓冲的值可直接作 GM 行下标（表驱动逐元素）

由更早的 `T.Parallel` pass 写入的 UB 标量缓冲可直接作 GM 行下标：
`cs[i,k] = TBL[row_ub[i], k]`，不需要退回 `T.alloc_local` + `T.serial` 展开。同一 kernel 内同时
验证可用：

- `@T.macro` 同时接收缓冲参数与 Python int 参数；
- 块级运行期 `if bx < nq:` / `else` 两支各开 `T.SimtVF`；
- `out_idx` + `T.dynamic` 的动态 extent 输出由 callee 正确分配（query-only `[4]`、key `[5,6]`、
  offsets `[6,7]` 三变体共存）。

真机 `atol=rtol=0` 逐位通过；22 个 16-bit 用例（含 `D=65`、`R=24`、`R<D`、±inf/NaN/65504）与
fp32 torch golden 的 `max_diff` 全为 0，仅 fp32 输出出现 ≤4.77e-07（1 ulp 级）。

边界：结论限于「小表按行载入 UB、块内所有 head 复用」的表驱动逐元素形态；不覆盖任意 stride 视图
或跨核表广播。

---
id: PL-950-simt-scalar-broadcast
kind: pattern
status: verified
origin_task: batch2-20261123 (engram_decode / _engram_step_kernel Stage 3)
toolchain: TileLang ascend 0.1.15 + CANN 9.2.1, 2026-11-23
toolchain_stamp: a010a95909e4e752d0d0b602885eca4b6fab233edca2d3e540248ed71c2135dc
toolchain_id: ascend
hardware_id: ascend-950pr-9579
repro: repro/PL-950-simt-scalar-broadcast.py
provenance: 溯源见 origin_task；`RESULT: scalar broadcast via T.Parallel(1) is per-lane and needs no barrier` 与 27 例 3 连跑逐位一致记录在 engram_decode/_engram_step_kernel/RETROSPECTIVE.md Stage 3
apis: [T.Parallel, T.SimtVF, T.alloc_reducer, T.copy]
dtype: [float16, bfloat16, float32]
---

# 行标量经 `T.Parallel(1)` 写入后可直接广播（无需 UB 暂存与 mem_bar）

行标量（`alpha`、`rv`、`rrms`）放在 `(1,)` fragment、由 `T.Parallel(1)` 写入时，在 `SimtVF` 内被
发射为每 lane 无守卫执行，因而对下一次逐元素 pass 天然广播：与内联 PrimExpr 形态逐位相同
（4×256 RMSNorm，2.38e-07 vs fp32 参考）。

同一实测族：4 个并发 reducer epoch 在同一 `T.Parallel` 内在 256 与 512 lane 上都数值可靠
（生成码 4 条独立 `AscendAllReduce<SumOp, T, 1, 0>`，scratch 偏移 2816/3328/3840/4352 float，
UB 峰值 24 576 B，与设计 §4.5 一致）；27 例 `max_y_diff = 7.6e-06`、
`max_state_diff = 1.95e-03`（= 2.0 处 fp16 1 ulp），每例 3 次发射逐位一致。

SIMT 侧不需要 `T.simd.mem_bar("VST_VLD")`：该规则只针对 SIMD `vsts → vld` 的同缓冲回读
（见 TRAP-950-simd-ub-store-load）。判据用「同一编译产物连跑 3 次位一致」，不凭直觉加屏障。

边界：结论限于 SIMT reducer / `T.Parallel` 写 UB 后经 `T.copy` 出栈的路径，覆盖 256/512 lane、
950PR 真机；不在 950DT 仿真与 64 lane 组合上作断言。

---
id: PL-950-dynamic-pipeline-2versions
kind: pattern
status: verified
origin_task: batch2-20261123 (mla_varlen_bwd / _delta_kernel Stage 3)
toolchain: TileLang ascend 0.1.15 + CANN 9.2.1, 2026-11-23
toolchain_stamp: a010a95909e4e752d0d0b602885eca4b6fab233edca2d3e540248ed71c2135dc
toolchain_id: ascend
hardware_id: ascend-950pr-9579
repro: repro/PL-950-dynamic-pipeline-2versions.py
provenance: 溯源见 origin_task；生成码 `(it − it0) & 1` 偏移与 T_q=65/200 数值断言记录在 mla_varlen_bwd/_delta_kernel/RETROSPECTIVE.md Stage 3
apis: [T.Pipelined, T.annotate_buffer_versions, T.copy]
dtype: [float16, bfloat16]
---

# 运行期界的 `T.Pipelined` 仍保留两版本缓冲

`T.Pipelined(it0, it1, num_stages=2)` 的界是运行期表达式时，`annotate_buffer_versions` 仍被兑现：
生成码里 `o_ub`/`do_ub` 的读写地址带 `(it − it0) & 1` 版本偏移（步长 8192 half = 一个 `(64,128)`
tile，第二操作数再 +16384 half），同步 event 也是 `(it − it0) & 1` 的成对
`asc_sync_notify/wait`；`D_v = 64/128/192` 三档几何均无 `did not enable multi-buffering` 告警。
同一循环体内的三条描述符可带运行期 count `min(64, total_q − t0)`。真机 `T_q = 65`（n=1）与
`T_q = 200`（n=8）残块逐位达容差。

对照边界：静态尾段若在 `T.Pipelined` **之外**写同一 UB 缓冲，AutoSchedule 会拒绝多版本
（见 CONST-950-tail-no-multibuffer）。两条现象一并说明「动态界」本身不损失第二版本，
「循环外写」才是失效源。

边界：结论为编译码 + 真机数值双证，限于上述几何与 dtype；不含 `num_stages > 2` 或 UB 侧显式
ping-pong 形态。

---
id: CONST-950-mmad-ktail-nonaligned
kind: pattern
status: verified
origin_task: batch2-20261123 (engram_decode / _engram_project_kernel Stage 3)
toolchain: TileLang ascend 0.1.15 + CANN 9.2.1, 2026-11-23
toolchain_stamp: a010a95909e4e752d0d0b602885eca4b6fab233edca2d3e540248ed71c2135dc
toolchain_id: ascend
hardware_id: ascend-950pr-9579
repro: repro/CONST-950-mmad-ktail-nonaligned.py
provenance: 溯源见 origin_task；L0-ktail-519（max_diff 4.578e-05）与 Boundary-dmem17/1 记录在 engram_decode/_engram_project_kernel/RETROSPECTIVE.md Stage 3
apis: [T.gemm, T.copy]
dtype: [float16, bfloat16]
---

# `asc_mmad` 按有效 K 掩蔽尾 lane：K ∈ {1,7,16,17,263,519} 真机正确

收缩轴 `d_mem` 允许非 16 倍数、甚至小于 16：K ∈ {1, 7, 16, 17, 263, 519}（含「短于一个 K 块」
与「256 块 + 7 尾」）在真机上与 fp32 参考一致（`max_diff ≤ 3.1e-5`，atol/rtol 1e-3）；生成码
把长度写进 mmad 实参：`asc_mmad(..., M=(17-max(...)), K=(519-max(kt*256,263)), N=64, ...,
(kt==0))`。因此 `d_mem = 519` 不需要 host 补零到 16 的整数倍（补零会引入新张量并改变 kernel
参数）。

同族边界：`T.clear` 对 L1 区域要求 C0（16 元素）对齐，「K 尾块就地补零」不可表达，需改用
「有效 K 区间」进区域或 host 侧 pad；动态尾长的 `T.clear` 会直接编译失败。

边界：结论限于 950PR 的 Cube L1-操作数 GEMM 与其有效 K 掩蔽；不含 N/M 尾的非对齐结论。

---
id: CONST-950-clear-pad-all-versions
kind: constant
status: verified
origin_task: batch2-20261123 (mla_dense_bwd / _delta_kernel Stage 3)
toolchain: TileLang ascend 0.1.15 + CANN 9.2.1, 2026-11-23
toolchain_stamp: a010a95909e4e752d0d0b602885eca4b6fab233edca2d3e540248ed71c2135dc
toolchain_id: ascend
hardware_id: ascend-950pr-9579
repro: repro/CONST-950-clear-pad-all-versions.py
provenance: 溯源见 origin_task；两段 4096 次标量存 + 两条 PIPE_S notify 的生成码与「有 clear vs 无 clear」对照记录在 mla_dense_bwd/_delta_kernel/RETROSPECTIVE.md Stage 3
apis: [T.clear, T.annotate_buffer_versions]
dtype: [bfloat16]
---

# 多版本 UB 缓冲上的 `T.clear` 覆盖全部版本与 pad lane（代价：标量循环）

`T.clear(buf)` 作用于 `annotate_buffer_versions` 给出两版本的 `alloc_shared` 缓冲时，lowering 为
PIPE_S 标量存循环（每次 8 B、`#pragma unroll`），清零**整个分配**，即两个版本都清、因此每个 pad
lane 都被写；没有 clear 时这些 lane 从不被写。实测生成码：两个 operand 各一段 4096 次 store 的
零填充循环（每段覆盖 2 版本 × 16384 B）+ 两条 PIPE_S notify；非 clear 变体无这些结构。带 padding
的几何上该展开还使编译时间约 15×。

含义：`D_v < DW`（如 `D_v = 64`、`DW = 128`）的 pad 分支可以依赖 clear 覆盖两版本，但这条依赖
属于编译器行为而非契约，工具链变更时应重跑该 repro。

边界：结论为编译码结构 + 真机数值（父任务 32 例，`max_diff ≤ 3.8e-6`），限于 UB 双版本场景；
不覆盖 `l0c`／`l1` 域的 clear（后者要求 C0 对齐）。

---
id: CONST-950-tail-no-multibuffer
kind: constant
status: verified
origin_task: batch2-20261123 (mla_dense_bwd / _delta_kernel Stage 3)
toolchain: TileLang ascend 0.1.15 + CANN 9.2.1, 2026-11-23
toolchain_stamp: a010a95909e4e752d0d0b602885eca4b6fab233edca2d3e540248ed71c2135dc
toolchain_id: ascend
hardware_id: ascend-950pr-9579
repro: repro/CONST-950-tail-no-multibuffer.py
provenance: 溯源见 origin_task；`did not enable multi-buffering for buffer "o_ub"` 告警与 S=128/S=131 生成码对照记录在 mla_dense_bwd/_delta_kernel/RETROSPECTIVE.md Stage 3
apis: [T.Pipelined, T.annotate_buffer_versions, T.copy]
dtype: [bfloat16]
---

# 流水循环之外写同一 UB 缓冲会丢失多缓冲（性能项）

当 UB 缓冲既在 `T.Pipelined` 内被写、又在循环**之外**被写（静态 `nlast` 尾段）时，AutoSchedule
拒绝请求的 2 版本，流水装载保持单一 UB 基址并输出告警
`did not enable multi-buffering for buffer "o_ub"`。实测对照：

| 几何 | 流水内 GM→UB 目的地址 | 告警 |
|---|---|---|
| 无尾段 S=128（2 个流水迭代） | 随 tile 索引变化（双版本） | 无 |
| 有尾段 S=131 = 2×64+3 | 固定（单版本） | 有 |

该差异只影响性能：两种几何在真机上都与 golden 一致（`max_diff = 3.8e-06`）。行数动态、无静态
尾段的形态不受影响（见 PL-950-dynamic-pipeline-2versions）。

边界：结论来自编译码 + 父任务真机数值；不给出「尾段移出流水」的收益量化（Stage 4 事项）。
