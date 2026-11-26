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

---
id: CONST-950-dma-vs-lane-gm
kind: constant
status: verified
origin_task: rope-RoPEFwdOp-20261124T171256Z (Stage 4)
toolchain: TileLang ascend 0.1.15 + CANN 9.2.1, 2026-11-24
toolchain_stamp: 7a52f2f1274df07d8f4c9c01353ce9d290f88c16e85047de0c425b6a009de4aa
toolchain_id: ascend
hardware_id: ascend-950pr-9579
repro: repro/CONST-950-dma-vs-lane-gm.py
provenance: 溯源见 origin_task；逐变量口径与 169.9 MB RoPE 载荷的同比值观测记录在 perf_opt/opt_log.md §2（地板探针）
apis: [T.copy, T.SimtVF, T.Parallel, T.vectorized]
dtype: [float32, float16, bfloat16]
---

# 纯搬运算子：GM↔UB 走 `T.copy` 描述符比 per-lane 访问快 2–3 倍，per-lane 宽度 16 B 即饱和

被测算子：`Y = X`（1 读 1 写），`X` 为 4096×4096 fp32（134.2 MB），`T.Kernel(ceildiv(T, 4))`，
每块 4 行 × 4096 列，`T.SimtVF(threads=256)`。真机 `Task Duration(us)`（msopprof 单 launch）：

| 形态 | 时延 | 有效带宽 | 相对 DMA |
|---|---:|---:|---:|
| `T.copy` GM→UB→GM（MTE2/MTE3 描述符） | **89.8 µs** | 1.49 TB/s | 1.00 |
| per-lane `T.vectorized(4)`（16 B/lane，GM 直读直写） | 182.2 µs | 0.74 TB/s | 2.03 |
| per-lane 自动向量化（8 B/lane） | 286.8 µs | 0.47 TB/s | 3.19 |
| per-lane `T.vectorized(8)`（32 B/lane） | 291.4 µs | 0.46 TB/s | 3.24 |

同一比值在另一载荷（RoPE 的 query[4096,4096]+key[4096,1024]，167.8 MB）上复现：
124.5 / 245.3 / 322.2 / 352.5 µs，比值 1.00 / 1.97 / 2.59 / 2.83。

结论与适用条件：

- 纯访存 kernel 的 GM 侧优先写成 `T.copy`（block 级 2D region，行距与长度均为 32 B 倍数）；
  向量 pipe 只负责 UB 内的计算遍。
- per-lane GM 访问的有效带宽随宽度到 16 B 饱和，32 B 反而略差（后端拆成多次 8 B 访问）。
- 边界：结论是「1 读 1 写、无复用」的纯搬运；带 L2 复用或需逐元素索引（gather/scatter）的路径不在此列，
  后者必须 per-lane。时延含 block 级固定开销，换 tile 大小后比值应重测。

---
id: CONST-950-simt-lane-vec-8b
kind: constant
status: verified
origin_task: rope-RoPEFwdOp-20261124T171256Z (Stage 4)
toolchain: TileLang ascend 0.1.15 + CANN 9.2.1, 2026-11-24
toolchain_stamp: 7a52f2f1274df07d8f4c9c01353ce9d290f88c16e85047de0c425b6a009de4aa
toolchain_id: ascend
hardware_id: ascend-950pr-9579
repro: repro/CONST-950-dma-vs-lane-gm.py
provenance: 溯源见 origin_task；被拒分支的生成码与实测记录在 perf_opt/opt_log.md §3 Round 2（H4）与 profiles/src/rope_round2_c2a_vec4_*.cce、profiles/experiments/c4_rot4_r0
apis: [T.SimtVF, T.Parallel, T.vectorized]
dtype: [float32, float16]
---

# SIMT VF 的 per-lane 向量访问上限是 8 B：`T.vectorized(4)` 仍降为两次 `float2`

在 `T.SimtVF` 内把逐元素遍写成显式 `for u in T.vectorized(4)`，生成码仍是
`*(__ubuf__ float2*)dst = *(__ubuf__ float2*)src`，并在内层再套一层 `for (u = 0; u < 2; ++u)`；
`float4`/`float8` 不出现。repro 断言该形态（`--time` 另测：16 B/lane 的显式形态 182.2 µs 仍优于自动 8 B 的
286.8 µs，差距来自每次迭代搬更多元素、循环开销更少，而不是更宽的向量指令）。

同一上限的实测后果（RoPE 原地旋转遍，`profiles/experiments/c4_rot4_r0`）：把旋转遍写成 `T.vectorized(4)`
后整体时延 150.4 → 214.7 µs **回退**；在被拒分支的生成码里该形态还在循环体内插入 `asc_syncthreads()`
（`profiles/src/rope_round2_c2a_vec4_*.cce`，in-place UB 读写触发保守依赖处理）。因此：
地址连续的逐元素遍交给自动向量化（8 B/lane），要更宽的搬运就走 `T.copy` 或 SIMD 路径。

边界：结论限于 `target="ascend"` + SIMT VF + fp32/fp16 的 UB/GM 逐 lane 访问；不改写 SIMD（`T.simd.vld/vsts`，
其寄存器宽度是 256 B）与 `T.copy` 描述符路径的结论。工具链升级后按 repro 重验。

---

id: PL-950-window-bulk-copy
kind: pattern
status: verified
origin_task: engram_gate_conv-engram_gate_conv-20261124T171256Z (Stage 4)
toolchain: TileLang ascend 0.1.15 + CANN 9.2.1, 2026-11-24
toolchain_stamp: 7a52f2f1274df07d8f4c9c01353ce9d290f88c16e85047de0c425b6a009de4aa
toolchain_id: ascend
hardware_id: ascend-950pr-9579
repro: repro/TRAP-950-rowload-dma-chain.py
provenance: 溯源见 origin_task；`engram_gate_conv_perf/perf_opt/opt_log.md` §3 Round 2（r2_c5）
apis: [T.copy, T.annotate_buffer_versions, T.simd.vsts]
dtype: [bfloat16, float16]
---

# 整窗 2-D `T.copy` 取代逐行装载 + 行循环零 DMA（-19.9%）

**慢 → 快的关键改动**（`_engram_gate_conv_fwd_kernel`，行循环带 3 行因果 halo）：

```python
# before: 每行 3 条单行 copy，用完即等
for i in T.serial(rc + 3):
    r = t0 - 3 + i
    if r >= 0:
        T.copy(H[b, r, :], h_ub[:d]); T.copy(K[b, r, :], k_ub[:d]); T.copy(V[b, r, :], v_ub[:d])
    with T.SimdVF(): ...            # 计算里读 h_ub/k_ub/v_ub

# after: 每块 3 条 owned 整窗 + 3 条 halo 整窗，行循环内不再有 DMA
h_ub = T.alloc_shared((rc + 3, dw), dtype)          # 窗口按行索引
T.copy(H[b, t0 : t0 + rc, :], h_ub[3 : 3 + rc, :d])  # owned 行，静态度、动态度原点
if t0 >= 3:
    T.copy(H[b, t0 - 3 : t0, :], h_ub[0:3, :d])      # halo，块级标量守卫
for i in T.serial(rc + 3):
    with T.SimdVF(): ...            # 只读 h_ub[i, ch*VL]
```

**优化见解**：瓶颈不是字节也不是带宽（`aiv_mte2_ratio` 只有 0.31），而是每行一条描述符带来的
`asc_sync_wait(PIPE_MTE2, PIPE_V)`；把它从 27 次/块降到 1 次/块，同时 `aiv_vec_wait_ratio`
0.88 → 0.12。因果边界靠**块级标量守卫 + 头部零填充**表达，零填充行经同一确定性表达式得到
`g = fl(0.5·0)·r_v = 0`，因此行循环连 `if r >= 0` 都可以去掉。
配套：输出侧用 `(rc+3, d)` 暂存 + 1 条 2-D 落盘替代逐行落盘（见 TRAP-950-rowstore-mte3-drain）。

实测（`train-s96`，msopprof Task Duration，3 次独立采集中位）：11.084 → 8.875 µs；三个 tune
workload 全面下降 16–25%。**反向边界**：不要叠加行级预取（+16.5%，见 TRAP-950-rowload-dma-chain）。

边界：结论限于 1-D grid、单波、行内向量化轴 = 行内连续维、行宽 ≥ 512 B 的 16-bit 行；
窗口 3 行 halo 的因果守卫形态可直接复用，跨 `m` 的窗口需要额外的边界处理。

---

id: PL-950-serial-copy-chain-hoist
kind: pattern
status: verified
origin_task: engram_decode-engram_decode-20261124T171256Z (Stage 4)
toolchain: TileLang ascend 0.1.15 + CANN 9.2.1, 2026-11-24
toolchain_stamp: 7a52f2f1274df07d8f4c9c01353ce9d290f88c16e85047de0c425b6a009de4aa
toolchain_id: ascend
hardware_id: ascend-950pr-9579
repro: repro/CONST-950-launch-floor-grid-sweep.py
provenance: 溯源见 origin_task；`engram_decode_perf/perf_opt/opt_log.md` §2.3/§3 Round 1；
  慢→快改动 = `perf_opt/experiments/mkvariant.py` 的 `SHIFT_BODY_OLD` → `SHIFT_BODY_NEW_C4`
apis: [T.copy, T.Kernel, T.alloc_shared]
dtype: [bfloat16, float16]
---

# 按 batch 串行化的 GM↔UB 搬运链 → 每个 batch 行一个块的整行窗口 copy

**慢 → 快的关键改动**（`_engram_project_kernel` 的 cache 左移，mix kernel 内）：

```python
# before: n_tiles 个块里，每块对全部 batch 串行（每次 = 2 条依赖 DMA + 1 次 sync）
if which == 0:
    if m0 == 0:
        shift_ub = T.alloc_shared((mcl - 1, block_n), dtype_str)     # 128 B 窄行
        for b in T.serial(batch):
            T.copy(conv_state[b, 1:mcl, n0:n0 + an], shift_ub)       # gm2ub
            T.copy(shift_ub, new_conv_state[b, 0:mcl - 1, n0:n0 + an])# ub2gm

# after: 只有在「每块串行 >1 条」时才切换（batch > n_tiles）；每个 batch 行一个块，
#        一条整行（连续）窗口 copy 对，与 Cube 投影并行
shift_wide_form = nshift > 0 and batch > n_tiles
if shift_wide_form:
    if bx < nb_shift:                                  # nb_shift = min(batch, grid)
        shift_ub = T.alloc_shared((nshift, d_pad), dtype_str)
        for r in T.serial(shift_reps):                 # batch <= grid 时 reps = 1
            b = bx + r * nb_shift
            if b < batch:
                T.copy(conv_state[b, 1:mcl, :], shift_ub)              # 整行窗口
                T.copy(shift_ub, new_conv_state[b, 0:nshift, :])
# else: 保留基线的窄切片形态（bash <= n_tiles 时它已天然并行）
```

**优化见解**：成本按**每块串行往返条数**计，不按字节计。serving-b8（batch=8、n_tiles=4）上
基线是 4 个块 × 8 条依赖往返 = **3.67 µs**；改写后每块 1 条 = **0.42 µs**，project launch
6.672 → 3.31 µs（−50%），T_op 9.466 → 6.385 µs（**−32.5%**）。反向边界（同任务实测）：
① `batch ≤ n_tiles` 时改写无收益（serving-b4：8 个块 × 4 条，0.68 µs，改写 0.71 µs）——
ND 生成的 device 代码在这些 shape 上与该改写**逐字节相同**，可用编译产物 sha 判定；
② 再把同一份搬运**按列切细到 16 个块**也无收益（serving-b4 4.835 vs 4.848 µs），
进一步证明是「往返条数 + 依赖 sync」而不是带宽；
③ 不要与行级预取叠加（`TRAP-950-rowload-dma-chain` 的反向结论）。

边界：结论限于「整块内单条串行搬运链 + 该搬运是纯 dtype copy（连续窗口）」；
若窗口跨 stride（非连续）则只能拿到部分收益。已在 `engram_decode` 的 `batch=8/17` 复现。

---
id: CONST-950-simd-vs-simt-elementwise
kind: constant
status: verified
origin_task: rope-RoPEFwdOp-20261125T120600Z (Stage 4 round 2)
toolchain: TileLang ascend 0.1.15 + CANN 9.2.1, 2026-11-25
toolchain_stamp: 7a52f2f1274df07d8f4c9c01353ce9d290f88c16e85047de0c425b6a009de4aa
toolchain_id: ascend
hardware_id: ascend-950pr-9579
repro: repro/CONST-950-simd-vs-simt-elementwise.py
provenance: 溯源见 origin_task；逐块 PipeUtilization 与生成码（`__simd_vf__` vs `__simt_vf__`）记录在 perf_opt/opt_log.md
  §2/§3（round 2）；同形状 rotate 探针的时延对照见 repro
apis: [T.SimdVF, T.SimtVF, T.simd.vld, T.simd.vsts, T.simd.vmul, T.simd.vsub, T.simd.pset, T.copy]
dtype: [float32, float16, bfloat16]
---

# 逐元素旋转遍：`T.SimdVF` 向量寄存器比 `T.SimtVF` per-lane 快约 15–20%（同一 256 B 数据段）

被测算子：RoPE 旋转（`y_lo = x_lo·cs − x_hi·sn`、`y_hi = x_hi·cs + x_lo·sn`），tile 4 行 × 1024 列 fp32、
8 个 head × 128 列（`half = 64`），cos/sin 每行一次载入 UB；GM 两侧都是 `T.copy` 描述符搬运，
差别只在 UB 内的这一遍执行方式：

| 执行方式 | 每块向量忙时 | `aiv_vec_ratio` | 整 kernel（169.9 MB 载荷） |
|---|---:|---:|---:|
| `T.SimtVF` per-lane（8 B/lane，自动向量化） | 9.33 µs | 0.78 | 145.3 µs |
| `T.SimdVF` + 256 B 寄存器（`vld/vmul/vsub/vsts`） | **5.22 µs** | **0.53** | **121.4 µs** |
| 同形状自包含探针（repro，`--time`） | — | — | simt 130.8 µs → simd 113.9 µs |

说明与边界：

- 一个 SIMT per-lane 机器步只搬 8 B/lane，且实测约 10 cycle/64-lane 步（≈25.6 B/cycle）；SIMD 一条
  `vlds_norm/vsts_norm` 覆盖整个 256 B 向量寄存器，因此同样的访存次数下指令代价低一个量级。
  两者**访存次数相同**（每 pair 4 读 2 写），收益来自执行模型，不来自减少遍数。
- 适用前提：一次搬运的连续段 ≥ 向量寄存器宽度（fp32 64 元素 / 16-bit 128 元素，即 256 B）；
  `half·itemsize` 为 128 B 时 16-bit 数据必须经 `UNPK_B16` 解到 fp32 lane、落盘经 `PK_B32` + 32-bit
  掩码（见 CONST-950-vcvt-narrow-store-pack-b32），此时每 head 多两次 `vcvt`，收益被摊薄
  （本任务 16-bit prefill 段 67% MBU vs fp32 段 86%）。
- 数值：fp32 域运算、落盘一次舍入，与 per-lane 形态逐位等价（本任务 L1/L2 全部 `max_diff = 0`）。
- 几何不满足时（如 GPT-J 相邻对、`half % 64 != 0`）在**构建期**回落 per-lane 形态，不做运行期分派。
- 边界：结论限于纯访存/逐元素 UB 遍 + 已用 `T.copy` 搬 GM 的形态；含归约或 Cube 的混合 kernel、
  以及需要 `T.simd` 掩码序列的尾块形态不在本条范围。
