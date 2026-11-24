# _ssd_chunk_coupling_kernel 算子设计文档

> 任务：TileOPs 迁移（harness 子模式），源算子 `tileops/_kernels/mamba/ssd_chunk_coupling.py`（GPU TileLang，`tilelang.language` / 无 target 声明的 CUDA 后端）。
> 目标平台：`ascend-950pr-9579` / 工具链 `ascend`（TileLang ascend 0.1.15，CANN 9.2.1），编程模式 `cube`。
> 本文件的算法调研与优化结论（§1.6）是本设计的唯一算法输入；§3.1 公式拆解取自 §1.6.1 的优化后公式。

## 0. 源算子解读与迁移分析

> 按「三问框架 + 耦合性判定 + 重设计」组织（方法论：`.agents/skills/tilelang-op-design/references/migration-analysis.md`）。§0.1–§0.4 只描述源算子事实，§0.5–§0.6 是迁移决策，§0.7 记录 golden 来源。

### 0.1 源算子语义（做什么）

数学语义：对每个 `(batch b, chunk c, group g)`，取该 chunk 的 `chunk_len × d_state` 行块
`Cb = C[b, cQ : cQ+Q, g, :]`、`Bb = B[b, cQ : cQ+Q, g, :]`（`Q = chunk_len`, `N = d_state`），计算

```
cb[b,c,g,l,s] = Σ_{n<N} Cb[l,n] · Bb[s,n]      if s <= l     （含主对角线）
             = 0                               if s > l      （严格上三角）
```

即 batched `C @ Bᵀ` + 因果下三角保留（`tril`，`s <= l`）。**掩码方向是保留 `s <= l`、清零 `s > l`**（源码 `ssd_chunk_coupling.py:120-123`：`if s_abs <= l_abs` 取累加值，否则显式写 0；上三角 tile 亦显式写 0，`ssd_chunk_coupling.py:126-130`）。

| 语义要素 | 结论 | 依据 |
|---|---|---|
| 规约轴与累加顺序 | 唯一规约轴 `n ∈ [0, N)`（矩阵乘的内积维）；语义上为结合律求和，源码按 `block_n` 分 K 块顺序累加（`ssd_chunk_coupling.py:85` 的 `T.serial(ceildiv(N, block_n))`），块内由 mma 累加 | 源码 `ssd_chunk_coupling.py:82-112` |
| 中间 dtype | 累加器 `accum_dtype = "float"`（fp32，`ssd_chunk_coupling.py:30`）；输入 fp16/bf16；**输出在写回时一次性 cast 到输入 dtype** | `ssd_chunk_coupling.py:82-83, 121` |
| 边界语义（掩码） | `s > l` 的元素输出**恒为精确 0，与输入取值无关**（源码显式写 `T.cast(0.0, dtype)`，不是乘 0） | `ssd_chunk_coupling.py:123, 130` |
| 边界语义（nan/inf） | 保留区域（`s <= l`）不做任何 nan/inf 特殊处理，按 IEEE 传播；上三角区域不传播 | 源码无特殊分支 |
| 空维度 / 非整除 | 契约：`B,S,G,N > 0`；`S % chunk_len == 0` ⇒ `NC = S // Q` 无余数（manifest `shape_rules`）；`chunk_len` 不要求被 `block_l/block_s` 整除（源码用 `l_abs < Q` 谓词处理，`ssd_chunk_coupling.py:91,119`） | manifest `examples/TileOPs/tileops/manifest/mamba.yaml` |
| 输出分配契约 | 输出由 `out_idx=[-1]` 新建（未初始化），**kernel 必须写满整个 `cb`**（含严格上三角的 0），不得假定预置零 | `ssd_chunk_coupling.py:39` |
| host 侧语义 | `forward` 只做 `C_mat.contiguous()` / `B_mat.contiguous()`，其余交给 kernel | `ssd_chunk_coupling.py:207-214` |

**语义保持基线**：§8.1 的 golden 以本节为唯一依据（移植源仓参考实现 `ssd_chunk_coupling_fwd_ref`），不复刻 §0.6 的 NPU 算法。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|---|---|---|---|
| `C_mat` | `[B, S, G, N]` | float16 / bfloat16（`same_as`） | 序列融合、group 内嵌布局：`stride = (S·G·N, G·N, N, 1)`；`G > 1` 时**行内 N 连续、行间跨 `G·N`** |
| `B_mat` | `[B, S, G, N]` | 同上 | 与 `C_mat` 同布局（不转置存放） |
| `cb`（输出） | `[B, NC, G, Q, Q]` | 与输入同 dtype | 输出 shape 见 `source_output_shape`；`l` 为行（对应 `C` 的行）、`s` 为列（对应 `B` 的行），**不是转置布局**；末两维连续 |

约束：`S = NC * Q`；所有 workload 的 `Q ∈ {64, 256}`、`N ∈ {64, 128}`、`G ∈ {1, 8}`。非连续输入（`examples/TileOPs/tests/ops/test_mamba.py` 的 `test_ssd_chunk_coupling_fwd_noncontiguous`）由 wrapper 的 `.contiguous()` 归一，属 host 契约而非 kernel 语义。

### 0.3 实现算法解读（怎么算）

**入口签名与 host 侧**：`_ssd_chunk_coupling_kernel(batch, num_chunks, n_groups, chunk_len, d_state, dtype)` 返回 `kernel_func(block_l, block_s, block_n, threads)`，默认 `block_l=block_s=block_n=64, threads=128`（`ssd_chunk_coupling.py:180-187`），autotune 在 `{32,64,128}³ × {128,256}` 上枚举（`ssd_chunk_coupling.py:189-205`），`@target_cache(maxsize=32)` 按 shape 缓存编译产物，`@tilelang.jit(out_idx=[-1])` 自动分配输出。

**计算步骤分解**（源码每个计算语句均归入步骤）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式的部分 |
|---|---|---|---|---|
| 1 | 三维 grid 解码：`bb = bcg//(C·G)`, `bc = (bcg%(C·G))//G`, `bg = bcg%G`；`l0 = bl·block_l`, `s0 = bs·block_s` | grid 索引 `(bl, bs, bcg)` | 绝对坐标 | 任务映射（非数学） |
| 2 | 三角门控：`if s0 < l0 + block_l` 才计算，否则整块写 0 | `l0, s0` | 分支 | 语义中的 `s > l ⇒ 0`（整块层面） |
| 3 | 搬入 `C_tile[ll,nn] = C_mat[bb, bc·Q+l0+ll, bg, n0+nn]`（越界补 0） | `C_mat` GM | `C_tile` shared `(block_l, block_n)` | 乘数左因子 |
| 4 | 搬入 `B_tile[ss,nn] = B_mat[bb, bc·Q+s0+ss, bg, n0+nn]`（越界补 0） | `B_mat` GM | `B_tile` shared `(block_s, block_n)` | 乘数右因子（按行取，等价于转置） |
| 5 | `T.gemm(C_tile, B_tile, acc, transpose_B=True)`：`acc += C_tile · B_tileᵀ` | `C_tile, B_tile` | `acc` fragment `(block_l, block_s)` fp32 | `Σ_n Cb[l,n]·Bb[s,n]` 的 K 块部分和 |
| 6 | K 循环：步骤 3–5 重复 `ceildiv(N, block_n)` 次（`acc` 不清零，跨块累加） | — | `acc` | 完整 N 维求和 |
| 7 | 掩码 + cast 写回：`s_abs <= l_abs ? cast(acc[ll,ss]) : 0` | `acc` | `cb` GM | 语义中的 `tril` 与 dtype 舍入 |
| 8 | 上三角分支只写 0（不计算） | — | `cb` GM | 语义中的 `s > l ⇒ 0` |

**数据流与内存访问模式**（源硬件视角）：

```
GM[C_mat,B_mat] --T.Parallel 逐元素(谓词)--> shared C_tile/B_tile(swizzled) --mma--> fragment acc(fp32)
                                                                                       |
                                                       T.Parallel 掩码+cast <---------+
                                                                |
                                                                v
                                                        GM[cb]（上三角分支单独写 0）
```

- 每 tile 的 GM 读量 = `(block_l + block_s)·block_n·2B`/K 块；写量 = `block_l·block_s·2B`。
- shared 复用：`C_tile`/`B_tile` 在 K 循环内被覆盖，`acc` 常驻 fragment（寄存器）跨 K 块累加。
- 无跨 block 通信、无 workspace、无 host 中间张量。

**循环与并行结构**：grid = `(ceildiv(Q,block_l), ceildiv(Q,block_s), B·C·G)` 三维，`threads=128`；每 block 一个 `(l-tile, s-tile, batch-chunk-group)` 输出 tile；K 循环 `T.serial`（默认配置无显式流水）；元素级 `T.Parallel`（载入与 epilogue）。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|---|---|---|---|---|
| 1 | SMEM tiling（`alloc_shared` 双操作数） | 降低 GM 重复访问 | 每个 tile 的 `C`/`B` 行块只搬一次，K 循环内复用 | GPU shared memory | 硬件强相关（布局/容量绑定） |
| 2 | `T.gemm`（tensor core mma）+ fp32 fragment 累加 | 矩阵算力 + 精度 | fragment 驻留寄存器，K 块累加 | tensor core / 大寄存器堆 | 硬件强相关 |
| 3 | K 分块（`block_n` 循环，累加不清零） | 控制片上占用，允许 K > 分块 | 部分和跨块累加 | — | 可移植（算法层分块） |
| 4 | `make_swizzled_layout`（两操作数） | 消 shared bank conflict | 地址重排 | GPU SMEM bank 组织 | 硬件强相关（源布局专属） |
| 5 | 上三角 tile 整块跳过计算（`if s0 < l0 + block_l`） | 省掉约 1/4–3/8 的无效乘加 | 块级分支，只写 0 | — | **可移植（纯算法层剪枝）** |
| 6 | 掩码与写回融合（同一 `T.Parallel` 内 select+cast） | 避免"先写后清零"的第二遍写 | 写回时按谓词选择 0 | — | 可移植 |
| 7 | 严格上三角显式写 0（不是跳过不写） | 满足"输出必须写满"契约 | 单 store 路径 | — | 保留（语义要求） |
| 8 | `threads=128` + `GemmWarpPolicy.FullRow` | warp 级工作划分 | warp 映射到行方向 | GPU warp（32 线程） | 硬件强相关 |
| 9 | 载入谓词（`l_abs < Q`、`n_abs < N` 补 0） | 非整除尾块保护 | 逐元素谓词 + 0 填充 | — | 需按目标平台重写（静态 tile / region 尾块） |
| 10 | 三维 grid + `target_cache`/autotune（host 侧） | 并行度枚举与编译缓存 | 每 shape 一份 kernel | — | 可移植（host 层） |
| 11 | 无跨 block 同步、无 workspace | 启动即独立任务 | 每输出 tile 由唯一 block 独占写 | — | 保留 |

### 0.5 硬件耦合性分析与 NPU 适配决策

**判定问题**：实现算法与上述优化手段是否硬件强相关？能否用在 NPU（`ascend-950pr-9579`，Cube/Vector 分离、L1/L0A/L0B/L0C/UB 分层）上？

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|---|---|---|---|---|---|
| 数学语义、I/O 契约、fp32 累加、上三角精确 0 | 语义 | 无 | — | **保留** | 语义层无条件保留（§0.1） |
| #1 SMEM tiling | 优化 | GPU shared memory | 有：L1（524 288 B）承载 Tile 操作数 | **等价替换** | `T.alloc_l1` + GM→L1 `T.copy`（nd2nz）；容量按平台预算实查（`programming.md` 分配表；`gemm-tiles.md` 已认证路径） |
| #2 `T.gemm` + fragment 累加 | 优化 | tensor core + 寄存器 fragment | 有：Cube（28 核）+ L0A/L0B/L0C | **等价替换** | `T.alloc_l0a/l0b/l0c` + `T.gemm(..., transpose_B=True, clear_accum=kk==0)`；probe P5/P7 生成 `asc_mmad(acc,x0,y0,128,64,128,0,1,0,kk==0)` |
| #3 K 分块累加 | 算法 | 无 | 有 | **保留** | `T.Pipelined(ceildiv(N, block_n), num_stages=2)`，`clear_accum` 仅首块 |
| #4 `make_swizzled_layout` | 优化 | GPU SMEM bank | 分形/NZ 布局由后端推导（`AscendLayoutInference` + `NormalizeAscendFractalStorage`） | **舍弃** | NPU 侧不手写 swizzle：L1/L0 布局由后端按 `shared.l1/l0a/l0b` scope 与 `ascend/pipeline.py` 的布局推导 pass 决定；手写重排会与 fractal 推导冲突。依据：`programming.md`「L0A/B 的布局留给消费它们的 GEMM 推导」 |
| #5 上三角 tile 跳过计算 | 算法 | 无 | 有（块级剪枝） | **保留**（机制重设计） | 保留"不算被掩码丢弃的乘加"的意图，但**不照搬 `if` 包住 gemm**：改为 K 循环 trip 数置 0（`trips = s0 <= l0+BL-1 ? n_kt : 0`），K 循环后仍走统一掩码 epilogue 写 0。依据：probe P5 生成码显示流水 drain 的 `asc_sync_wait` 在循环**外**（`probe_out/probe_src_P5-autosched-mask-fp16.cce`），而 P8/P9 动态 trip 版本的前置 notify 与尾部 wait 数量在 0/2 迭代下均配对（同一文件核对） |
| #6 掩码与写回融合 | 算法 | 无 | 有（AIV 逐元素） | **保留** | 单次 `select + cast` 写入 UB，再 UB→GM 一次 store（§1.6.1 O1） |
| #7 上三角显式写 0 | 语义 | 无 | 有 | **保留** | 统一掩码 epilogue 对整块输出 0（同一条 store 路径） |
| #8 `threads=128` + `FullRow` warp policy | 优化 | GPU warp | Cube 侧无 warp 概念；逐元素侧有 AIV SIMT lane | **重新设计** | Cube 的 M/N/K 分形由 `T.gemm` 后端的 `asc_mmad` 指令参数决定；逐元素侧改为 `T.SimtVF(threads=128)` + `T.Vector(vector=2)` 双 AIV 分片（§0.6 R2/R3） |
| #9 载入谓词 + 0 填充 | 优化 | 无 | 部分：`T.copy` region + `T.min` 尾块 | **重新设计** | `l/s` 方向 tile 静态化（`Q % BL == 0`，见 §5.4），K 方向用 `nk = T.min(N-n0, BK)` 动态 region；不再需要逐元素谓词（probe P1/P5/P7 均生成带正确行跨度的 `asc_copy_gm2l1_nd2nz`） |
| #10 三维 grid | 优化 | GPU 三维 blockIdx | 无：`T.Kernel` 只接受一维网格（多维在实现中拒绝） | **重新设计** | 一维网格 + 标量解码（§0.6 R1）；依据 `ascend/language/kernel.py:161-168`（`T.Kernel(N) takes exactly one grid extent`） |
| #10 host autotune / `target_cache` | 优化 | 无 | 有 | **保留** | 沿用按 shape 特化的工厂 + Stage 4 A/B 调参（block/stages 为参数级旋钮） |
| #11 无跨 block 同步 | 算法 | 无 | 有 | **保留** | 每输出 tile 仍由唯一 block 独占写，无跨核归约 |
| epilogue 在 GPU 用 `T.Parallel` 直接写 GM | 优化 | GPU 每线程可做逐元素算术 | NPU Cube 无逐元素单元，故 epilogue 必须落到 AIV | **重新设计** | L0C →(dual copy)→ UB(fp32) → AIV `select+cast` → UB→GM（§0.6 R3；probe P5/P7 `asc_copy_l0c2ub` + `asc_copy_ub2gm_align`） |

> 每条处置的依据均为本平台 `examples/` / knowledge 条目 / probe 实测编译产物，未使用其他工具链（npuir）的 API 结论。

### 0.6 NPU 算法重设计

**重设计项 R1：并行结构与任务映射（三维 grid → 一维网格 + 标量解码）**

- **源方案**：`T.Kernel(ceildiv(Q,block_l), ceildiv(Q,block_s), B·C·G, threads=128)` 三维 grid，block 内 128 线程做载入/epilogue。
- **NPU 新算法**：一维网格 `grid = B·NC·G · nlt · nst`，`bid` 用整数除/模解码出 `(bb, bc, bg, lt, st)`；block 内不再有"线程"概念，Cube 段由 `T.gemm` 的后端分形调度占据，逐元素段由 `T.Vector(vector=2) as sid` 的两个 AIV 承担（`sid ∈ {0,1}`）。解码顺序设计为 `bcg → lt → st`，使相邻 block 共用同一 `(b,c,g)` 的 `C` 行块，利于 L2 复用。
- **语义保持论证**：数学映射不变——每个输出元素 `cb[b,c,g,l,s]` 仍由唯一 block 写（`(b,c,g,lt,st)` 与 tile 一一对应），无跨 block 归约；K 归约顺序仍为 `kk = 0 → n_kt-1`，块内分形顺序变化只改变 fp32 加法分组（§0.1 允许结合律；fp32 分组差异 ≤ 1e-7 相对，落盘前一次性舍入，见 §8.2）。边界：`Q % BL == 0` 保证无越界 tile；K 尾块由 `nk = T.min(N-n0, BK)` region 处理，与源 `n_abs < N` 补 0 语义一致（补 0 参与乘加，结果不变）。

**重设计项 R2：矩阵计算原语（fragment mma → L1/L0 显式操作数 + Cube）**

- **源方案**：`alloc_shared` 承载两操作数 + `T.gemm(..., transpose_B=True)` 累加进 fragment，`swizzled_layout` 消 bank conflict，`GemmWarpPolicy.FullRow` 做 warp 划分。
- **NPU 新算法**：`GM →(nd2nz)→ L1 →(MTE1)→ L0A/L0B →(MAD)→ L0C(fp32)`；每 K 块 `T.copy` 两次（`C`/`B` 行块）+ 两次 L1→L0 + 一次 `T.gemm(..., transpose_B=True, clear_accum=(kk==0))`；K 块双版本（`T.annotate_buffer_versions({cl1:2, bl1:2})` + `num_stages=2`）以让第 2 块搬入与第 1 块 MAD 重叠。算子天然是 `(M,K)·(N,K)ᵀ`，故 `transpose_B=True` 直接消费行式存放的 `B`，无需任何核内/核外转置。布局与分形由后端从 scope 推导（不手写 swizzle）。
- **语义保持论证**：计算仍是 `Σ_n Cb[l,n]·Bb[s,n]`，只把"寄存器累加"换成"L0C 累加"（同为 fp32）；`clear_accum` 仅首 K 块为真 ⇒ 与源"`T.clear(acc)` 后跨块累加"一致，无部分和丢失（`gemm-tiles.md` 认证的同一模式）。边界：M/N/K 用静态分配、region 用动态尾块，等价于源谓词补 0。

**重设计项 R3：epilogue 归属（Cube 无逐元素单元 → L0C→UB→AIV）**

- **源方案**：GEMM 结果留在 fragment，用一个 `T.Parallel` 就地做掩码 + cast 并写 GM（GPU 上每个线程都能做标量算术）。
- **NPU 新算法**：`acc(L0C fp32) →(hardware dual-destination copy, M 切分)→ 每个 AIV 的 UB `cv`(64×128 fp32) →T.SimtVF→ co(64×128, 输入 dtype) →(UB→GM, 按 sid 分片)→ cb`。掩码判据用 AIV 侧 `sid` 还原绝对行号：`l_abs = l0 + sid·(BL/2) + i`。
- **语义保持论证**：`co = s_abs <= l_abs ? cast(cv) : 0` 与源 `if s_abs <= l_abs: cast(acc) else 0` 逐元素同值；上三角元素在两个实现里都是**精确 0**（不是 `v*0`，避免 `nan*0=nan` 破坏契约，见 §1.6.1 与 `verify_equiv.py` E3 反例）。dtype 路径同为"fp32 计算 → 一次舍入到输入 dtype"。边界：AIV 持有的 `cv` 为 fp32（`dual_copy` 禁止 L0C→UB 跨 dtype，`copy_op.py:146` 明确拒绝），cast 在 UB 内完成。

**重设计项 R4：上三角 tile 的处理（`if` 包住 GEMM → K trip 数置 0 + 统一掩码）**

- **源方案**：`if s0 < l0 + block_l:` 包住"载入 + gemm + 掩码写"，`else:` 只写 0。意图：不计算注定被清零的乘加。
- **NPU 新算法**：保留同一意图，但把"分支包住 pipelined 循环"换成**循环 trip 数**：`trips = T.if_then_else(s0 <= l0 + BL - 1, n_kt, 0)`；`trips = 0` 时 K 循环体不执行（无 GM 读、无 MAD），`acc` 为未初始化值，随后**统一**掩码 epilogue 把整块写成 0。
- **语义保持论证**：整块被掩码 ⇒ 输出恒为 0，与"先算再清零"完全同值（`verify_equiv.py` E4 用 nan/inf/±0/subnormal 垃圾累加器机器验证：两种实现输出逐位相等）。边界：**不采用** `if` 直接包住 pipelined GEMM——probe（P5 生成码）显示流水 drain 的 `asc_sync_wait` 位于循环体外，条件化循环体存在 set/wait 配对风险；而 trip=0 版本（P8/P9）的前置 notify 与尾部 wait 在 0/2 迭代下都平衡（probe 产物已核对）。

### 0.7 标杆实现

- 源算子实现（GPU TileLang）：`/workspace/TileOPs/src/tileops/_kernels/mamba/ssd_chunk_coupling.py`（抽取副本：`examples/TileOPs/tileops/kernels/mamba/ssd_chunk_coupling/_ssd_chunk_coupling_kernels.py`）。
- 源仓参考实现（golden 依据）：`ssd_chunk_coupling_fwd_ref`，见 `examples/TileOPs/tileops/testing/mamba2_reference.py` —— `einsum("bcqgn,bcsgn->bcgqs")` + `tril` 掩码 + cast，与 §0.1 语义逐条一致；上游为 `mamba_ssm.ops.triton.ssd_bmm._bmm_chunk_fwd`（manifest `ref_api`）。
- golden 只移植该参考实现（Torch 广播/einsum 形态），不复刻 §0.6 的 tile 结构，保证验证独立性。
## 1. 概述

### 1.1 算子名称

`_ssd_chunk_coupling_kernel`（TileOPs `SSDChunkCouplingFwdOp` 的 kernel；Mamba-2 chunk 耦合矩阵 `C @ Bᵀ` + 因果掩码）。

### 1.2 功能描述

对每个 `(batch, chunk, group)` 计算 `chunk_len × d_state` 与 `d_state × chunk_len` 的乘积，并把严格上三角部分精确置 0，输出 `[B, NC, G, Q, Q]`，dtype 与输入一致。

### 1.3 数学公式

$$
\mathrm{cb}[b,c,g,l,s] =
\begin{cases}
\displaystyle\sum_{n=0}^{N-1} C[b,\,cQ+l,\,g,\,n]\cdot B[b,\,cQ+s,\,g,\,n], & s \le l\\[4pt]
0, & s > l
\end{cases}
\qquad Q=\text{chunk\_len},\; N=\text{d\_state},\; NC=S/Q
$$

### 1.4 算法描述

NPU 侧算法（= §1.6.0 调研选定、经 §0.6 重设计落地的算法）分三段：

1. **任务映射**：一维网格 `grid = B·NC·G · nlt · nst`（`nlt = nst = Q/BL`）。`bid` 解码出 `(bb, bc, bg, lt, st)`，得到输出 tile 左上角 `(l0, s0) = (lt·BL, st·BL)`。每 tile 由唯一 block 独占写。
2. **Cube 段（GEMM）**：`trips = (s0 ≤ l0+BL-1) ? ⌈N/BK⌉ : 0`。对 `kk ∈ [0, trips)`：`C` 行块与 `B` 行块 `GM→(nd2nz)→L1`，再 `L1→L0A/L0B`，`T.gemm(x0, y0, acc, transpose_B=True, clear_accum=(kk==0))` 累加进 `acc`(L0C fp32)。
3. **AIV 段（掩码 epilogue）**：`acc →(dual copy, M 切半)→` 每个 AIV 的 `cv`(64×128 fp32) → `T.SimtVF(threads=128)` 内 `co[i,j] = cast((s0+j) ≤ (l0+sid·64+i) ? cv[i,j] : 0)` → `UB→GM` 写回 tile（含整块置 0 的情形）。

**与源算法的差异**（逐条注明来源）：

| 差异 | 源 | 本设计 | 来源 |
|---|---|---|---|
| 网格 | 三维 `(l-tile, s-tile, B·C·G)` | 一维 + 标量解码 | §0.6 R1（`T.Kernel` 只收一维网格） |
| 上三角 tile | `if` 包住 GEMM | K 循环 trip 置 0 + 统一掩码写 0 | §0.6 R4 |
| 掩码所在单元 | fragment 内逐线程 | L0C→UB 后在 AIV | §0.6 R3 |
| 布局技巧 | `make_swizzled_layout` + `FullRow` | 后端推导分形布局 | §0.5（舍弃 #4/#8） |
| 尾块 | 逐元素谓词补 0 | 静态 l/s tile + K region 尾块 | §0.5（#9） |

### 1.5 数据流图

```
C_mat[B,S,G,N] ─┐                                    ┌─> cb[B,NC,G,Q,Q]
                ├─ GM→L1(nd2nz) ─> L1 ─> L0A/L0B ─> L0C(fp32) ─dual copy─> UB(fp32)
B_mat[B,S,G,N] ─┘      (每 K 块)      (Cube/MAD)      (每 tile)              │
                                                                  AIV select+cast
                                                                             │
                                                              UB(dtype) ─> GM (tile 写回)
```

### 1.6 算法调研与优化分析 ⭐

> 顺序：先调研（1.6.0，四问）→ 再数学等价优化（1.6.1）→ 再向量化替代（1.6.2）→ 再定向量化轴与布局（1.6.3）→ 备选与实验裁决（1.6.4）。

#### 1.6.0 算法调研（Algorithm Research）⭐

调研对象：同一数学语义（batched `tril(C·Bᵀ)`）的算法族。调研深度：**完整调研**（矩阵类；R1 候选表 + R3 四口径 + R4 逐候选）。

**R1 等价化简公式候选**（基线在表内；等价性初判在此，正式论证在 §1.6.1）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 纳入 R3 |
|---|---|---|---|---|---|
| 0 | **基线：源 GPU 算法** | `BL=BS=64` 方形 tile，`if s0 < l0+BL` 才做 GEMM，否则整块写 0；fragment 内掩码+cast | — | — | ✅（基线） |
| 1 | **trip-skip 静态 tile（主选）** | `BL=BS=128`；严格上三角 tile 令 K 循环 `trips=0` 只走 epilogue 写 0；对角/下三角 tile 正常 GEMM | 数学恒等：被掩码的 tile 输出恒 0（E4） | 少算：去掉 `1/4`（128 tile）无效乘加；少读：该 tile 不读 `C/B` | ✅ |
| 2 | 统一掩码不跳过（备选 B） | 同上但 `trips ≡ ⌈N/BK⌉`（上三角 tile 也算再丢弃） | 数学恒等 | 结构化简；计算量 +33% vs #1 | ✅ |
| 3 | 三角枚举 + 镜像置零 | 只对 `st ≤ lt` 的 tile 发起 GEMM；严格下三角 tile 顺带把镜像 tile 写 0 | 数学恒等 | 计算量降到 62.5%（BL=64）；代价：双 store 路径、块间负载不均、三角索引映射 | ✅ |
| 4 | 掩码并入 GEMM（把 `tril` 当矩阵乘进去） | `tril(C·Bᵀ) = (L⊙C)·Bᵀ` 不可分解；或把 mask 作为第三个操作数做 `Q×Q` 逐元素乘 | ❌ 不等价化简：mask 无法吸收进 `C`/`B`（`L` 依赖 `(l,s)` 双下标），只能追加一次 `Q²·N` 级运算 | 无收益（增加算量） | ❌ 排除 |
| 5 | 分块递推（行块间复用部分和） | `cb` 的 `(l,s)` 元素相互独立，行块之间**无递推关系**（`Cb[l,:]·Bb[s,:]` 不共享部分和） | ❌ 无递推化简空间（结构判据：bilinear、无跨 `(l,s)` 状态） | 无 | ❌ 排除 |
| 6 | 两段式 materialize | GEMM 写 fp32 中间张量到 GM，第二个 kernel 做掩码 + cast + 置零 | 等价（同值） | ❌ 访存翻倍（中间张量往返）+ 两个 launch | ✅（作为反例入 R3） |
| 7 | 对称性复用 | `cb` 一般不对称（`C ≠ B`） | ❌ 仅在 `C == B` 的特殊输入下对称，语义不允许假设 | 无 | ❌ 排除 |

**R2 在线算法**：**有**，且即本设计所用形态。结构判据：算子是 bilinear 求和 `Σ_n`，可分解为 running 部分和（L0C 常驻、每 K 块累加），因此单遍扫描即可，中间状态 `O(tile)`（64 KB L0C + 64 KB L1），无 `O(N)` 或 `O(Q²)` 的 GM 中间缓冲。不需要 online 重缩放类变体（无 softmax/归一化，`Σ` 对分组顺序不敏感，仅有 fp32 结合律分组差，容差内等价）。反向证据：源算子本身就是单遍累加（`ssd_chunk_coupling.py:85`），不存在两遍变体；候选 #6（两段式）是唯一的多次扫描形态，被 R3 淘汰。

**R3 复杂度对比**（以最大 workload `mamba2-grouped-prefill`（B=3, S=3072, G=8, N=128, Q=256, NC=12，288 个 `(b,c,g)` 单元）为基准；另列 wl1 便于分核；口径：FLOPs = MAC×2 只计乘加，不含索引/掩码/搬运指令；访存量 = GM 读写字节，含 tile 重读、不含 L2 命中削减；扫描遍数 = 同一输入数据的 GM 读次数（写按 1 遍计）；中间缓冲峰值 = 片上驻留峰值）：

| 候选 | FLOPs（wl3；wl1） | 访存量 Bytes（wl3，无 L2 削减 / 含 L2） | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---|---|---|---|---|---|
| 因果下界（参考，不可直接实现） | 2.42 GF；0.134 GF | 71.3 MB（33.6 读 + 37.7 写） | 1R + 1W | — | — |
| #0 基线（BL=64 三角） | 3.02 GF；0.168 GF | 132.2 MB / ≈108 MB | 2.5R + 1W | L1 32 KB、UB 24 KB、L0C 16 KB | 256 块（wl1）/4608（wl3）；无跨核 |
| **#1 trip-skip（主选）** | **3.62 GF；0.201 GF** | **94.4 MB / ≈71.3 MB** | **1.5R + 1W** | L1 64 KB、UB 48 KB、L0C 64 KB | 64/1152 块；无跨核 |
| #2 统一掩码（备选 B） | 4.83 GF；0.268 GF | 113.2 MB / ≈71.3 MB | 2R + 1W | 同 #1 | 64/1152 块；无跨核 |
| #3 三角枚举 + 镜像置零 | 3.02 GF；0.168 GF | 132.2 MB / ≈108 MB | 2.5R + 1W | L1 32 KB、UB 24+16 KB、L0C 16 KB | 160/2880 块；块间负载不均（下三角块双 store） |
| #6 两段式 materialize | 4.83 GF；0.268 GF | 222.1 MB / ≈150 MB | 3R + 2W | 需 GM 中间张量 75.5 MB | 2 个 launch + 全局同步边界 |
| #5 l-strip（M=Q=256） | 4.83 GF；0.268 GF | 113.2 MB / ≈71.3 MB | 2R + 1W | L0C 128 KB、UB 96 KB | 32/576 块（波次尾块更差） |

**R4 硬件亲和性评估**（对照清单：计算单元匹配 / 片上容量 / 对齐整除 / 静态边界 / 流水可融合性 / 跨核结构；容量取自任务平台 `capacity_bytes`：UB 253 952 B、L1 524 288 B、L0C 262 144 B、L0A/L0B 65 536 B）：

| 候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---|---|---|---|---|---|---|
| #0 基线（直译为 NPU） | MAD 密集段落 Cube ✅ | L1/UB/L0C 均 ≤ 16–32 KB ✅ | BL=64、BK=64 均 16 的倍数 ✅ | ✅ | ✅ | 可直接跑，但 64-tile 重读最多（2.5R）、无 K 双缓冲叠加 |
| **#1 trip-skip** | MAD 落 Cube、掩码落 AIV ✅（分工与 PL-950-mixed-epilogue 同型） | L1 64 KB / 512 KB、L0C 64 KB / 256 KB、UB 48 KB / 248 KB ✅ | BL=128、BK=64、cv 行 128 lane ✅ | K trip 为块级条件表达式（非逐元素）✅ | K 双版本 + AIV/Cube 异步 ✅ | **存活（最优）** |
| #2 统一掩码 | 同上 | 同上 | 同上 | 全静态 ✅ | ✅ | 存活（计算量 +33%） |
| #3 三角枚举 + 镜像置零 | 同 #1 | 需额外 UB 零缓冲 ✅ | ✅ | 三角索引映射为块级标量 ✅ | 镜像 store 引入块间不均 ⚠ | 存活但劣于 #1（负载不均 + 双 store 路径） |
| #6 两段式 | 第二段纯 Vector，Cube 空闲 ❌ | 需 75.5 MB GM 中间张量（超片上）❌ | ✅ | ✅ | 两 launch，无法融合 ❌ | **淘汰**（访存 2–3×，且中间张量落 GM） |
| #5 l-strip | ✅ | L0C 128 KB（50%）、UB 96 KB ✅ | ✅ | ✅ | ✅ | 淘汰（波次尾块 43% 空转、无计算量优势） |

**调研结论**：选定 **#1 trip-skip 静态 tile（BL=BS=128、BK=64、K trip 条件置 0、L0C→AIV 统一掩码）**。关键依据：R3 中它在存活的 C1 类候选中访存最低（94.4 MB → L2 削减后 ≈ 下界 71.3 MB，与 #2 打平）、计算量比 #2 少 25%、扫描遍数 1.5R 优于基线的 2.5R；R4 中片上占用（L1 64/512 KB、L0C 64/256 KB、UB 48/248 KB）留出双版本与 AIV 双分片余量，且全部 tile 边界静态。与基线的结构差异一句话：**把"64-tile 分支跳过 + fragment 掩码"换成"128-tile 的 K-trip 置 0 + L0C→UB→AIV 掩码"**。基线优势（#3 的 62.5% 计算量）被其结构性劣势（重读 2.5R、块数 4×、镜像双写路径、块间负载不均）抵消，且 #3 的剩余收益（计算量）在 R4 的"流水/负载均衡"维度上被 #1 反超；#4/#5/#7 因不成立被排除。

> 源算法优化手段的意图承接：§0.4 #5（上三角跳过计算）→ 由 trip=0 承接（意图保留，机制改为循环 trip，§0.6 R4）；#1/#2/#3（tiling/GEMM/K 分块）→ L1/L0 + `T.gemm`（§0.6 R2）；#6/#7（掩码融合、显式置零）→ 统一掩码 epilogue（§1.6.1 O1）；#4/#8（swizzle、warp policy）→ 舍弃，理由见 §0.5；#9/#10（谓词、三维 grid）→ 静态 tile + 一维网格（§0.6 R1）；#11（无跨块同步）→ 保留。

**设计期估算（D-2）**：本平台 `cost_measurements` 为空，`knowledge` 下无 `CONST-*` 实测常数条目（`bottlenecks.md` 明确「当前没有 950 真机验证的瓶颈模式」），故容量项按平台 `capacity_bytes` 核算（全部满足，无淘汰），带宽与算力两项为**未实测假设**：

| workload | 流量项（主选，含 L2 削减） | 发射项 | 容量项 | 估算下界 |
|---|---|---|---|---|
| wl1 mamba2-780m | 4.19 MB ÷ 1.6 TB/s ≈ 2.6 µs | 64 块 / 28 核 = 2.29 波，块启动代价未实测 | 无淘汰 | ≈ 2.6 µs + 启动 |
| wl3 grouped-prefill | 71.3 MB ÷ 1.6 TB/s ≈ 44.6 µs | 1152 块 / 28 核 = 41 波 | 无淘汰 | ≈ 44.6 µs + 启动 |

`估算下界: 待核验（缺少 ascend-950pr-9579 的 HBM 有效带宽、Cube fp16 峰值算力与单块启动代价实测常数；流量项按 1.6 TB/s 未实测假设：wl1 ≈ 2.6 us、wl2 ≈ 5.2 us、wl3 ≈ 44.6 us；算力项在 224 TFLOP/s 假设下 wl3 ≈ 16 us，非主导）`

#### 1.6.1 数学等价优化（公式级）

分析对象 = §1.6.0 选定的 #1（trip-skip 静态 tile 掩码）。

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益量化 |
|---|---|---|---|---|---|
| O1 | 掩码 + dtype cast 融合为单次 select 写（消掉"先写后清零"） | 先把 `cast(acc)` 全量写入 GM，再第二遍扫描把 `s>l` 写成 0 | `out[i,j] = (s_abs ≤ l_abs) ? cast(acc[i,j]) : 0`（同一次 UB 写） | 数学恒等：对 `s>l` 的元素两式均输出精确 0；对 `s≤l` 元素两式同值同舍入路径（fp64 参照下 **0 违反**，`verify_equiv.py` E1：random/corner/garbage 三组、fp16+bf16 共 1536 元素，max_ulp = 0） | 省掉上三角区域的第二次 GM 写：每 `(b,c,g)` 省 `Q²/2·2 B` = 65 536 B（wl3 合计 18.9 MB，占总量 20%），并省一次全网格第二遍扫描 |
| O2 | 掩码判据索引强度缩减 | `(s0 + j) ≤ (l0 + sid·(BL/2) + i)`（每元素 3 次加法 + 1 次比较） | `(j - i) ≤ (l0 + sid·(BL/2) - s0)`（右侧为块级标量，每元素 1 次减法 + 1 次比较） | 整数加法/减法恒等（无溢出、无舍入）：`s0+j ≤ l0+si+i ⇔ j-i ≤ l0+si-s0`；`verify_equiv.py` E2 在 `BL ∈ {16,32,64,128} × l0,s0 网格 × sid ∈ {0,1}` 共 262 144 个判据上 **0 违反** | 每元素少 2 次整数加/比较（向量化后 epilogue 的整数运算从 4 op 降到 2 op，矢量段 IPC 受限时约省 1 拍/元素） |
| O3 | 省略被掩码丢弃的 K 累加（trip 置 0） | 上三角 tile 也执行 `⌈N/BK⌉` 次"搬入 + MAD"，再整块丢弃 | `trips = (s0 ≤ l0+BL-1) ? ⌈N/BK⌉ : 0`，该 tile 的 `acc` 内容任意，统一掩码写 0 | 等价：被掩码 tile 的输出与 `acc` 取值无关（恒 0）。`verify_equiv.py` E4 用 `nan/inf/-0/subnormal` 垃圾累加器验证两种实现输出**逐位相等**（fp16+bf16，0 违反） | 少算 `1/4` tile 的乘加（wl3：4.83→3.62 GFLOP，−25%）与该 tile 的 `C/B` 读（wl3：−18.9 MB） |

**未采纳的等价式（附证据）**：

| 式 | 结论 | 证据 |
|---|---|---|
| 掩码用乘法 `v · 1[s≤l]` 代替 select | **不采纳**（破坏边界语义） | `nan·0 = nan`：`verify_equiv.py` E3 在 64 个元素中 28 个违反"上三角精确 0"；而 select 版本 E1 全通过 |
| `if (s0 ≤ l0+BL-1)` 包住 pipelined GEMM | **不采纳**（同步配对风险） | probe P5 生成码中流水 drain 的 `asc_sync_wait` 在循环体外（`probe_out/probe_src_P5-autosched-mask-fp16.cce`），条件化循环体可能使 set/wait 数量失配；trip=0 版本（P8/P9）前置 notify 与尾部 wait 在 0/2 迭代下平衡 |
| 两段式（GEMM 落 GM + 掩码 kernel） | **不采纳** | R3：访存 222 MB vs 94 MB，多一次 launch |
| 逐元素谓词补 0（源 #9） | **不采纳** | tile 边界静态化（`Q % BL == 0`）后无越界元素；K 尾块用 region `T.min`，比逐元素谓词少一次全 tile 扫描 |

**等价性机器验证结果**（`examples/ssd_chunk_coupling/_ssd_chunk_coupling_kernel/verify_equiv.py`，torch CPU，fp64 参照 + 目标 dtype 舍入；实际执行输出）：

| 采纳项 | 用例 | 元素数 | 最大 ulp 差 | 违反率 | 结论 |
|---|---|---|---|---|---|
| O1（E1）| random / corner(±inf,nan,±0,subnormal,max-finite) / garbage(nan·inf 累加器) × fp16 | 768 | 0 | 0.000 | **EQUIV_PASS** |
| O1（E1）| 同上 × bf16 | 768 | 0 | 0.000 | **EQUIV_PASS** |
| O2（E2）| `BL∈{16,32,64,128}` × `l0,s0∈[0,256)` × `sid∈{0,1}` 整数判据 | 262 144 | 0 | 0.000 | **EQUIV_PASS** |
| O3（E4）| 整块被掩码 tile（`s0 > l0+BL-1`）× fp16/bf16，累加器取 nan/inf/-0/subnormal | 512 | 0 | 0.000 | **EQUIV_PASS** |

**优化结论**：采纳 3 项（O1 融合 select+cast、O2 掩码索引强度缩减、O3 被掩码 tile 的 K-trip 置 0），全部经 `verify_equiv.py` 机器验证为 **EQUIV_PASS**（最大 ulp 差 0、违反率 0）。优化后公式（供 §3.1 使用）：

```
trips = (s0 ≤ l0+BL-1) ? ⌈N/BK⌉ : 0
acc[l,s] = Σ_{kk < trips} Σ_{n ∈ K-chunk(kk)} Cb[l,n] · Bb[s,n]     # L0C fp32，首块 clear_accum
out[i,j] = ( (j - i) ≤ (l0 + sid·(BL/2) - s0) ) ? cast(acc[i,j]) : 0    # i∈[0,BL/2), j∈[0,BL)
```

#### 1.6.2 向量化替代分析（循环 / 标量消除）

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由 |
|---|---|---|---|---|---|
| 1 | 掩码 select + dtype cast（每 AIV 64×128 元素） | 源用 `T.Parallel` 标量逐元素（GPU 每线程一元素） | `T.Vector(vector=2) as sid` + `T.SimtVF(threads=128)` + `T.Parallel(BL//2, BL)` 内 `T.if_then_else` + `T.cast`（每行 128 个连续 fp16 = 256 B，满向量宽度） | ✅ | — |
| 2 | 掩码谓词的整数索引运算（`(j-i) ≤ l0+sid·64-s0`） | 逐元素标量加/比较 | 并入 #1 的 `T.Parallel` 表达式，逐 lane 并行（§1.6.1 O2 已强度缩减） | ✅ | — |
| 3 | 严格上三角 tile 的置零写 | 源用独立 `T.Parallel` 循环写 0 | 由 #1 的同一次 select 完成（`dst = 0`），无独立循环 | ✅ | — |
| 4 | 输出 tile 写回（BL×BL） | 逐元素 store | `T.dual_copy(co, cb[tile region])` → `asc_copy_ub2gm_align`（DMA 整块写，2 个 AIV 各写 64 行） | ✅ | — |
| 5 | 输入 tile 搬入（C/B 行块） | 源 `T.Parallel` 逐元素 gather + 谓词补 0 | `T.copy(C_mat[bb, bcQ+l0 : +BL, bg, n0 : n0+nk], cl1)` → `asc_copy_gm2l1_nd2nz`（DMA 二维 strided 整块搬，行跨度 `G·N·2 B`） | ✅ | — |
| 6 | 矩阵乘累加（BL×BL×BK） | 逐元素乘加（若不用张量单元会是标量循环） | `T.gemm(x0, y0, acc, transpose_B=True, clear_accum=…)` → `asc_mmad`（Cube 单元） | ✅ | — |
| 7 | tile 解码 `(bb, bc, bg, lt, st)` | 标量整除/取模 | 无（保持标量） | ❌ | **block 级任务映射**：每 block 常数条整数运算，非逐元素热点；无逐元素等价 API（掩码/向量 API 不适用于任务索引） |
| 8 | `trips` 条件表达式 | 标量 `if_then_else` | 无（保持标量） | ❌ | **block 级控制**：决定整个 tile 的循环次数，是 tile 级顺序依赖（循环 trip），不构成逐元素热点 |
| 9 | K 循环 `T.Pipelined(⌈N/BK⌉, num_stages=2)` | 循环 | 无（保持循环） | ❌ | **tile 级调度循环**（K 分块累加），非逐元素标量循环；其内部每块由 #5/#6 的 DMA + MAD 覆盖 |
| 10 | 网格/核数调度 | `T.Kernel(grid)` | 无 | ❌ | 启动维度，由平台 1-D 网格契约决定（`T.Kernel` 拒绝多维，`ascend/language/kernel.py:161-168`） |

**向量化结论**：逐元素计算（掩码 select、cast、置零、tile 搬入/搬出）已全部由向量/DMA/Cube 原语覆盖，替代方案的 API 均在 probe P5/P6/P7 生成码中实证（`asc_copy_gm2l1_nd2nz`、`asc_copy_l0c2ub`、SIMT VF 主体 `if/condval` 逐 lane、`asc_copy_ub2gm_align`、`asc_mmad`）。保留 4 处标量/循环：block 级任务映射（#7）、block 级控制（#8）、tile 级 K 调度循环（#9）、启动维度（#10）——理由类别分别为 block 索引 / tile 级顺序依赖 / tile 级调度 / 启动契约，均非逐元素标量热点。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

I/O layout 是契约（`C/B` 为 `[B,S,G,N]`，`cb` 为 `[B,NC,G,Q,Q]`，均不改），核内布局与 lane 映射是设计变量。本算子属 **Cube/MixCV 类**，按该类清单枚举"矩阵布局 / 装载重排 vs 显式转置 / epilogue 归属侧"。

**轴质量评分维度**：

| 评分项 | 本算子取值 |
|---|---|
| 整除性 | 尾轴 `s`（`Q=256`）对 fp16 向量宽度（128 lane × 8 元素/拍 计）满整除；`BL=128`、`BK=64`、`cv` 行 128 元素均为 16/32 的倍数（Cube 分形与 DMA 32 B 对齐同时满足） |
| 尾 lane 浪费率 | epilogue 行宽 128 元素、fp16 = 256 B = 8×32 B ⇒ 0%；K 尾块仅当 `N % 64 ≠ 0` 出现（本 workload `N ∈ {64,128}` ⇒ 0%） |
| 累加链形态 | Cube 累加沿 K（`BK=64`）串行、M/N 为并行维；epilogue 的 `i`（行）为 lane 并行、`j` 为连续向量维（stride 1） |
| repack 代价 | 0：`nd2nz` 在 GM→L1 搬运中顺带完成 N→NZ，无需独立转置；`B` 以行式存放直接配 `transpose_B=True` |
| UB 容量影响 | `cv` 64×128 fp32 = 32 768 B + `co` 64×128 dtype = 16 384 B = 48 KB / 248 KB |

**候选矩阵**（逐候选量化；弃选给出量化理由）：

| # | 布局方案 | 向量化轴 / 路径 | repack 路径 | 预估收益 / 代价 | 采纳 |
|---|---|---|---|---|---|
| 1 | GM→L1 直接搬 `B` 的 `(行, K)` region（行跨度 `G·N`）+ `gemm(transpose_B=True)` | Cube 原生 M/N/K 分形；epilogue 向量轴 = `s`（连续 128 元素） | 无（`nd2nz` 顺带转 NZ） | 基线；probe P1/P5/P7 生成 `asc_copy_gm2l1_nd2nz(..., 行跨度 2·G·N B)`，probe P1（G=8）实测跨度 2048 B = `G·N·2`，与语义一致 | ✅ |
| 2 | GM→L1 用 `transpose=True`（dn2nz）把 `B` 装成 `(K, 行)` | 同 #1 | 核内转置装载 | 无收益：`gemm` 已支持 `transpose_B`，dn2nz 只会增加一次转置路径；probe P1/P5/P7 均 `dn2nz = 0`，说明主选路径不需要 | ❌ |
| 3 | host 侧 permute 输入为 `(G, B·S, N)` 或转置布局 | — | host 全量重排 | 代价 = 额外一次全张量读写（wl3 67 MB 级），远超算子本体（71 MB） | ❌ |
| 4 | epilogue 归 Cube：`T.copy(acc → cb region)` fixpipe 直存（带 fp32→dtype 量化） | Cube fixpipe | 无 | 可行（probe P1：`asc_copy_l0c2gm`，`quant_mode=1`），但 **Cube 无法施加因果掩码** ⇒ 必须第二次遍历清零（= §1.6.1 未采纳项，+20% 写）+ 上三角 block 无意义计算 | ❌（仅作为无掩码场景的对照） |
| 5 | **epilogue 归 AIV：`dual_copy(acc→cv)`(L0C→UB, M 切分) + `SimtVF` 掩码 + `dual_copy(co→GM)`** | AIV SIMT，向量轴 = `s`；两 AIV 各持 64 行 | 无 | probe P5/P6/P7 实证：`asc_copy_l0c2ub(dual_dst_mode=1)` + VF 逐 lane `if/condval` + `asc_copy_ub2gm_align(64 行, 行跨度 Q·2 B, sid 偏置 64·Q)` | ✅ |
| 6 | N 切分的 `dual_copy`（`dual_dst_ctl=2`） | AIV，两 AIV 各持 64 **列** | 无 | UB 占用相同，但每行只写 64 元素 = 128 B 连续段（#5 为 256 B），store burst 更短；且 `copy_op.py` 要求 N 切分时全 N 为 32 的倍数（满足但无收益） | ❌（量化：写 burst 256 B → 128 B） |
| 7 | 单 AIV epilogue（`sids=1`，整 128×128 入 UB） | AIV 单核 | 无 | 向量吞吐减半、UB 需 64 KB（128×128 fp32）；且需"L0C 整块→单 AIV UB"的路径（`dual_copy` 只认证 2:1 分片，`copy_op.py:53-93`），无执行证据 | ❌ |

**布局决策结论**：选定方案 #1（Cube 侧 L1 行式直接装载 + `transpose_B=True`，零 repack）+ 方案 #5（epilogue 归 AIV，`L0C →(M 切分)→ UB(fp32) → SimtVF 掩码+cast → UB(dtype) → GM`，向量轴 = `s`，行宽 128 元素 = 256 B）。该结论同步落入 §3.3 伪代码（`cv`/`co` 形状为 `(BL/2, BL)`、`i` 映射 lane、`j` 为连续向量维）与 §6 循环结构（epilogue 内层向量维是 `j`）；弃选 #2/#3/#4/#6/#7 的量化理由见上表，判定裕度依赖的未实测常数（store burst 效率、L0C→UB 单核路径可用性）已在 #6/#7 的弃选理由中显式标注。

#### 1.6.4 主选 / 备选方案与实验裁决计划

主选与备选的判定裕度落在**未实测常数**上（`ascend-950pr-9579` 的 HBM 有效带宽、Cube fp16 峰值算力、单块启动代价均无实测条目），故按判定三件套执行：

**主选（= §1.6.0 #1，进 Stage 3）**：`BL=BS=128`、`BK=64`、`num_stages=2`、K trip 条件置 0、统一掩码 epilogue。缓冲：`cl1/bl1` L1 `(128,64)` ×2 版本 = 64 KB；`x0/y0` L0A/L0B `(128,64)` = 16 KB 各一版；`acc` L0C `(128,128)` fp32 = 64 KB；`cv` UB `(64,128)` fp32 = 32 KB；`co` UB `(64,128)` = 16 KB。逻辑核数：wl1 64 / wl2 128 / wl3 1152 / smoke 2（对 28 物理核见 §5.5）。

**备选 B（≤1 个，结构完整）**：与主选**唯一差异**是 K 循环 trip 固定为 `⌈N/BK⌉`（不跳过严格上三角 tile）——即"上三角 tile 也做 GEMM，再由统一掩码丢弃"。缓冲形状、布局（#1+#5）、UB/L1/L0C 预算、分核参数、dtype 路径与主选完全相同；代码差异一行人可读的 `trips = n_kt`（probe P5/P7 即为该变体的编译实证）。代价：wl3 计算量 4.83 GFLOP（+33%）、GM 读 113.2 MB（+20%）；收益：无数据依赖的循环边界（最保守的调度形态）。

**实验裁决计划（Stage 4 执行）**：

| 项 | 内容 |
|---|---|
| 代表 shape | `mamba2-780m`（B=1,S=4096,G=1,N=128,Q=256,fp16）、`mamba2-grouped-prefill`（B=3,S=3072,G=8,N=128,Q=256,fp16）、`mamba2-2p7b`（B=4,S=2048,G=1,N=128,Q=256,bf16） |
| 测量指标 | kernel-only `Task Duration(us)`（平台 `measurement` = `msprof_task_duration`，原始证据 `msprof_raw_path`）；A/B 交错、≥5 次取中位；精度先验（`COUPLING_ATOL/RTOL = 1e-3`） |
| 反解未知常数 | 由两变体时延差反解"上三角 tile 的 GEMM+搬入代价"，与流量项对账，回填 §1.6.0「设计期估算」行 |
| 判定阈值 | 备选 B 的 `Task Duration` < 主选 × 0.95（即主选退化为纯流量受限、trip 条件带来开销）→ 翻转采纳 B；否则保留主选 |
| shape 分派 | 工厂按 shape 特化（`target_cache` 同型），允许 `Q/BL` 与小 `Q` 场景（smoke Q=64 时 `nlt=1`，两变体代码等价）按 shape 分派，不做全局二选一 |
| 回写路径 | 实测数据经 `perf_feedback`/附录补记回写本节与 §1.6.0 估算行；若实测显示掩码/UB 路径为瓶颈（而非 tile 结构），按设计修订路径 A 回 Stage 1 |
## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: **Cube**（platform `programming_modes.choices.cube`）。计算主体（`BL×BL×BK` 矩阵乘）落在 Cube 单元；epilogue 的掩码 + dtype cast 由同一 kernel 内的 AIV（`T.Vector(vector=2)` + `T.SimtVF`）承担——这是本平台"Cube 主体 + 混合尾部"的标准形态（与 `PL-950-mixed-epilogue` 同型，probe 生成 `__mix__(1, 2)` 全局函数）。

### 2.2 选型理由

| 判据 | 本算子 | 结论 |
|---|---|---|
| 计算类型 | 每 tile 为 `(128,128,·)` 矩阵乘 + 逐元素掩码 | 矩阵段必须 Cube（Vector 单元做 128³ 乘加慢 1–2 个数量级） |
| 含 matmul | 是（`C·Bᵀ`） | 平台适配器亦要求设计含 `T.gemm`（`has_matrix_path`） |
| 含归约 | 仅矩阵乘内积（Cube 内完成） | 无需 Vector 归约原语 |
| 含逐元素后处理 | 掩码 select + fp32→dtype cast | 必须 AIV：Cube 无逐元素单元，`T.gemm` 输出只能整块搬出 |
| 是否需流水 | K 方向双版本 + Cube/AIV 异步 | 自动调度（`allow_autoschedule` 默认开启）即可 |

> 偏离平台 `migration_default = simd` 的理由：本算子的算力主体是矩阵乘而非逐元素；Simd/SIMT 全程实现会把 `Q²·N` 级乘加压到向量单元，与源算子"用张量单元"的核心意图（§0.4 #2）相悖，且 Cube 路径已有本平台认证先例（`gemm-tiles.md`）。调度 prompt 亦指定 Cube。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|---|---|
| 内存分配 | L1 `T.alloc_l1`、L0A/B `T.alloc_l0a/l0b`、L0C `T.alloc_l0c`、UB `T.alloc_shared`（`scope=shared.dyn`）；无 GM workspace |
| 计算方式 | Cube：`T.gemm(transpose_B=True, clear_accum=…)`；AIV：`T.Parallel` + `T.if_then_else` + `T.cast` |
| 搬运 | `T.copy`（GM→L1 nd2nz、L1→L0A/B）；`T.dual_copy`（L0C→UB 硬件双目的、UB→GM 按 sid 分片） |
| 同步 | 自动（`InsertSync` + AIC/AIV intra sync），不手写 flag |
| 核数映射 | 1-D 网格 → 28 个 Cube 核 + 每块 2 个 AIV 子核（共 56 个 Vector 核） |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入为 §1.6.1 的**优化后公式**（trip 条件 + 强度缩减谓词），与 §1.6.2 的向量化结论一致。

| 步骤 | 数学表达 | 说明 |
|---|---|---|
| 1 | `(bb, bc, bg, l0, s0) ← bid` | 一维网格解码（整数除/模，block 级标量） |
| 2 | `trips = 1[s0 ≤ l0+BL-1] · ⌈N/BK⌉` | 被掩码 tile 的 K 循环次数置 0（§1.6.1 O3） |
| 3 | `x[l,n] = Cb[l, kk·BK+n]`，`y[s,n] = Bb[s, kk·BK+n]`，`n < nk` | K 块载入（`nk = min(N-kk·BK, BK)`） |
| 4 | `acc[l,s] += Σ_{n<nk} x[l,n]·y[s,n]` | Cube MAD，`clear_accum` 仅 `kk=0` |
| 5 | `acc` 按 M 切成两半 → `cv` | L0C→UB 硬件双目的拷贝 |
| 6 | `co[i,j] = ((j-i) ≤ (l0 + sid·BL/2 - s0)) ? cast(cv[i,j]) : 0` | 掩码 select + cast（§1.6.1 O1/O2） |
| 7 | `cb[b,c,g,l0+i', s0+j] = co[i', j]` | UB→GM 整块写（含整块 0） |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API（`tilelang.ascend.language`） | 参数 | 模式 |
|---|---|---|---|---|
| 1 | 任务解码 | `T.Kernel(grid) as bid` | 1-D 网格（多维被拒绝，`ascend/language/kernel.py:161-168`） | Cube/AIV 共用 |
| 2 | trip 条件 | `T.if_then_else(cond, k_trips, 0)` → `T.Pipelined(trips, num_stages=2)` | `num_stages=2` | Cube |
| 3a | `GM→L1`（C 行块） | `T.copy(C_mat[bb, bc*Q+l0 : bc*Q+l0+BL, bg, n0 : n0+nk], cl1[:, :nk])` | 4-D region，固定 `bb/bg`，行跨度 `G·N`；probe 生成 `asc_copy_gm2l1_nd2nz(..., 行跨度 = 2·G·N B)` | Cube（MTE2） |
| 3b | `GM→L1`（B 行块） | 同上，`s0` 起始 | 同上 | Cube（MTE2） |
| 3c | `L1→L0A/L0B` | `T.copy(cl1[:, :nk], x0[:, :nk])` / `T.copy(bl1[:, :nk], y0[:, :nk])` | region 尾块 | Cube（MTE1） |
| 4 | `acc += x·yᵀ` | `T.gemm(x0[:, :nk], y0[:, :nk], acc, transpose_B=True, clear_accum=(kk == 0))` | `node_M/N/K` = 128/128/64（静态分配）；probe 生成 `asc_mmad(...,128,64,128,0,1,0,kk==0)` | Cube（MAD） |
| 5 | `L0C→UB` M 切分 | `T.dual_copy(acc, cv)` | `acc(128,128)fp32 → cv(64,128)fp32`；`copy_op.py:146` 禁止跨 dtype ⇒ 必须 fp32 UB；probe 生成 `asc_copy_l0c2ub(..., dual_dst_mode=1)` | Cube→AIV |
| 6 | 掩码 + cast | `with T.Vector(vector=2) as sid: with T.SimtVF(threads=128): for i,j in T.Parallel(BL//2, BL): co[i,j] = T.cast(T.if_then_else((j-i) <= (l0 + sid*(BL//2) - s0), cv[i,j], T.float32(0.0)), dtype)` | 行 128 lane；probe P10 生成逐 lane `if ((threadIdx.x - i) <= (...))` | AIV（SIMT） |
| 7 | `UB→GM` 分片写 | `T.dual_copy(co, cb[bb, bc, bg, l0:l0+BL, s0:s0+BL])` | `co(64,128) → region(128,128)`，M 切分；probe 生成 `asc_copy_ub2gm_align(64 行, 行跨度 Q·2 B, sid 偏置 64·Q)` | AIV（MTE3） |

### 3.3 计算伪代码

> 与 probe `P10/P11/P12`（`history_version/design_probe_compile.py`）**逐行一致**，编译已通过（生成 `__mix__(1,2)`，含 `asc_copy_gm2l1_nd2nz` / `asc_mmad` / `asc_copy_l0c2ub(dual_dst_mode=1)` / SIMT VF 掩码 / `asc_copy_ub2gm_align`）。

```python
Q, N, G, S = chunk_len, d_state, n_groups, num_chunks * chunk_len
BL = 128 if Q % 128 == 0 else (64 if Q % 64 == 0 else 32)   # 断言 Q % BL == 0，否则工厂报错
BK = 64 if N % 64 == 0 else 16
nlt = nst = Q // BL
k_trips = ceildiv(N, BK)
grid = batch * num_chunks * n_groups * nlt * nst

@T.prim_func
def main(C_mat: T.Tensor((batch, S, G, N), dtype),
         B_mat: T.Tensor((batch, S, G, N), dtype),
         cb:    T.Tensor((batch, num_chunks, G, Q, Q), dtype)):
    with T.Kernel(grid) as bid:
        # --- 1) 任务解码（block 级标量） ---
        bcg = bid // (nlt * nst)
        rem = bid % (nlt * nst)
        bb  = bcg // (num_chunks * G)
        cg  = bcg % (num_chunks * G)
        bc  = cg // G
        bg  = cg % G
        l0  = (rem // nst) * BL
        s0  = (rem % nst) * BL

        # --- 片上缓冲 ---
        cl1 = T.alloc_l1((BL, BK), dtype)          # L1  C 行块
        bl1 = T.alloc_l1((BL, BK), dtype)          # L1  B 行块
        x0  = T.alloc_l0a((BL, BK), dtype)         # L0A
        y0  = T.alloc_l0b((BL, BK), dtype)         # L0B
        acc = T.alloc_l0c((BL, BL), "float32")     # L0C 累加
        cv  = T.alloc_shared((BL // 2, BL), "float32")   # UB, 每 AIV 半块
        co  = T.alloc_shared((BL // 2, BL), dtype)       # UB, 每 AIV 半块
        T.annotate_buffer_versions({cl1: 2, bl1: 2})

        # --- 2) K 循环（严格上三角 tile 的 trips = 0） ---
        trips = T.if_then_else(s0 <= l0 + BL - 1, k_trips, 0)
        for kk in T.Pipelined(trips, num_stages=2):
            n0 = kk * BK
            nk = T.min(N - n0, BK)
            T.copy(C_mat[bb, bc * Q + l0 : bc * Q + l0 + BL, bg, n0 : n0 + nk], cl1[:, :nk])
            T.copy(B_mat[bb, bc * Q + s0 : bc * Q + s0 + BL, bg, n0 : n0 + nk], bl1[:, :nk])
            T.copy(cl1[:, :nk], x0[:, :nk])
            T.copy(bl1[:, :nk], y0[:, :nk])
            T.gemm(x0[:, :nk], y0[:, :nk], acc, transpose_B=True, clear_accum=(kk == 0))

        # --- 3) 掩码 epilogue（AIV 双分片） ---
        T.dual_copy(acc, cv)
        with T.Vector(vector=2) as sid:
            with T.SimtVF(threads=128):
                for i, j in T.Parallel(BL // 2, BL):
                    co[i, j] = T.cast(
                        T.if_then_else((j - i) <= (l0 + sid * (BL // 2) - s0),
                                       cv[i, j], T.float32(0.0)),
                        dtype)
            T.dual_copy(co, cb[bb, bc, bg, l0 : l0 + BL, s0 : s0 + BL])
```

### 3.4 API 可行性确认

| API | 来源确认 | 执行/编译证据 | 状态 |
|---|---|---|---|
| `import tilelang.ascend.language as T`、`tilelang.compile(..., target="ascend", out_idx=-1)` | `references/programming.md`「API 与执行区域」 | probe P1–P12 全部经 `tilelang.compile(target="ascend")` | 已验证（编译） |
| `T.Kernel(grid)`（仅 1-D） | `ascend/language/kernel.py:150-170`（源码拒绝多维） | probe 全部用例 | 已验证（编译） |
| `T.alloc_l1 / alloc_l0a / alloc_l0b / alloc_l0c / alloc_shared` | `ascend/language/allocate.py`；`references/programming.md` 分配表 | probe 生成 `buf_dyn_l1 / x0(__ca__) / y0(__cb__) / acc(__cc__) / buf_dyn_shmem` | 已验证（编译） |
| `T.gemm(transpose_B=True, clear_accum=…)`（L0 操作数路径） | `ascend/language/gemm_op.py:31-100`（静态 M/N/K、region 动态 K）；`knowledge/gemm-tiles.md` 已认证 | probe 生成 `asc_mmad(acc,x0,y0,128,64,128,0,1,0,kk==0)` | 已验证（编译） |
| `T.Pipelined(trips, num_stages=2)` + `annotate_buffer_versions` | `ascend/language/frame.py`、`annotations.py`；`knowledge/gemm-tiles.md` | probe 生成 `for(kk…<condval)` + `(kk&1)` 双版本环 + notify/wait | 已验证（编译） |
| `T.copy` 4-D strided region（GM→L1） | `ascend/language/copy_op.py:230-330`；probe 地址/跨度推导 | probe P1/P7：`asc_copy_gm2l1_nd2nz(..., 2048, …, 128, 64, …)`，2048 B = `G·N·2`；P2/P6（G=1,N=64）为 128 B | 已验证（编译 + 地址推导核对） |
| `T.dual_copy`（L0C→UB，M 切分；UB→GM） | `ascend/language/copy_op.py:53-160`（2:1 形状推导、L0C 跨 dtype 拒绝） | probe 生成 `asc_copy_l0c2ub(..., dual_dst_mode=1)`、`asc_copy_ub2gm_align(64, 行跨度, sid 偏置)` | 已验证（编译） |
| `T.Vector(vector=2) as sid` | `ascend/language/frame.py:36-47`（返回 `asc_get_sub_block_id()`） | probe 生成 `int32_t sid_1 = asc_get_sub_block_id();`，VF 内用 `sid_1` 计算行偏置 | 已验证（编译） |
| `T.SimtVF(threads=128)` + `T.Parallel` + `T.if_then_else` + `T.cast` | `platforms/.../kernel_template.py`（`if_then_else` 用例）；`PL-950-row-primitives` | probe P10 VF 主体逐 lane `if ((threadIdx.x - i) <= …)` | 已验证（编译） |
| `T.MixedKernel(sids=2)` + 显式 `T.Cube()/T.Vector()` | `ascend/language/kernel.py:174-230`（docstring 示例） | probe P3/P4 **编译失败**：`Cannot assign scalar task because it accesses incompatible core-local memories on both Vector (UB) and Cube (L1/L0) cores` | **不可用（本形态）**，故不采用（见 §9.1） |
| `T.copy(acc → cb region)`（L0C→GM fixpipe + 量化） | — | probe P1：`asc_copy_l0c2gm(..., quant_mode=1)` | 可用但无掩码能力，仅作对照（§1.6.3 #4） |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束（平台 ruleset / 源码） | 本算子是否涉及 | 处理方案 |
|---|---|---|
| `T.Kernel` 只接受 1-D 网格（多维在实现中拒绝） | 是 | 一维 + 标量解码（§3.3 步骤 1） |
| `T.gemm` 的赋值 tile M/N/K 必须静态（`gemm_op.py:71-74`） | 是 | 分配形状 `(BL,BK)/(BL,BL)` 全静态，region 用 `:nk` 尾块 |
| L0C 只作 fp32 累加器，容量 262 144 B | 是 | `acc(128,128)fp32` = 65 536 B（25%） |
| `dual_copy` L0C→UB 跨 dtype 被前端拒绝（`copy_op.py:139-152`） | 是 | UB 中转 fp32（`cv`）+ 独立 dtype 输出缓冲（`co`），cast 在 VF 内 |
| `dual_copy` 需恰好一个尾维 2:1（`copy_op.py:120-135`），全 M 需偶数 | 是 | `(128,128)→(64,128)`、`(64,128)→(128,128)`，BL ≥ 32（smoke BL=64） |
| 手工跨核 flag / 跨 block barrier 未认证（`programming.md` 同步节） | 否（不使用） | 全部走自动调度同步 |
| 数组尾轴 32 B 对齐 / 16 元素分形 | 是 | `BK=64`、`BL=128`、行宽 128 元素×2 B = 256 B（8×32 B）；仅当 `N%64≠0` 出现 K 尾块 |
| host 侧只允许元数据视图（不得改输入内容） | 否（kernel 内不依赖） | wrapper 的 `.contiguous()` 沿用源算子语义（非本 kernel 职责） |
| UB 253 952 B / L1 524 288 B / L0A,B 65 536 B | 是 | §4.5 预算：UB 48 KB、L1 64 KB、L0A/B 16 KB each |

#### 3.5.2 参考实现差异说明

| 差异项 | 参考实现（GPU TileLang） | 本项目（Ascend） | 转换方案 |
|---|---|---|---|
| Kernel 结构 | `T.Kernel(bl, bs, bcg, threads=128)` 三维 + 线程 | `T.Kernel(grid)` 一维 + `T.Vector(vector=2) as sid` | §0.6 R1；probe 全部用例 |
| 矩阵 API | `T.gemm(shared, shared, fragment, transpose_B=True, policy=FullRow)` | `T.gemm(l0a, l0b, l0c, transpose_B=True, clear_accum=…)` | §0.6 R2；`gemm-tiles.md` |
| 内存分配 | `alloc_shared` + `alloc_fragment` | `alloc_l1/alloc_l0a/alloc_l0b/alloc_l0c/alloc_shared` | `programming.md` 分配表 |
| 布局技巧 | `make_swizzled_layout` | 后端分形推导（`AscendLayoutInference`） | §0.5（舍弃 #4） |
| epilogue | fragment 内 `T.Parallel` 直接写 GM | L0C→UB(dual_copy)→AIV 掩码+cast→UB→GM | §0.6 R3；§1.6.3 #5 |
| 三角剪枝 | `if s0 < l0+BL` 包住 GEMM | K trip 条件置 0 | §0.6 R4；§1.6.1 O3 |
| 尾块 | 逐元素谓词补 0 | 静态 tile + K region `T.min` | §0.5（#9） |
| 输出写回 | 逐元素 store | `dual_copy` → `asc_copy_ub2gm_align` | probe P5/P7 |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|---|---|---|
| `.agents/platforms/toolchains/ascend/knowledge/repro/PL-950-gemm-tiles.py` | 高（矩阵主体） | L1→L0A/L0B→`T.gemm(transpose_B=True, clear_accum=step==0)`；K 分块 + `T.min` 尾块；动态 region 与静态分配的组合 |
| `.agents/platforms/toolchains/ascend/knowledge/repro/PL-950-cube-tile.py` | 高（最小 Cube 骨架） | `alloc_l1/l0a/l0b/l0c` + `T.gemm` + `T.copy` 的显式 L0 路径 |
| `.agents/platforms/toolchains/ascend/knowledge/repro/PL-950-mixed-epilogue.py` | 高（epilogue 形态） | `T.dual_copy(L0C→UB)` + `T.SimtVF` 逐元素 + `T.dual_copy(UB→GM)` 的整条混合尾部链 |
| `.agents/platforms/toolchains/ascend/kernel_template.py` | 中（kernel 骨架与分层测试） | `T.Kernel` + `T.SimtVF` + `T.if_then_else` 的写法；`--level L0/all` 测试骨架 |
| `examples/shared_expert_mlp/_gemm_basic_kernel/history_version/design_probe_compile.py` | 中（工程方法） | 设计期 compile-only probe 的组织方式（本任务同法） |
| `examples/gemm/example_gemm.py`、`examples/gemm/matmul.py` | 低（**不可作为 API 证据**） | 二者用 `target="npuir"` / `tilelang.language`，属另一工具链；仅作结构参考，不采用其 API |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|---|---|---|---|
| `C_mat` | `[B, S, G, N]` | float16 / bfloat16 | 序列融合 group 内嵌布局，stride `(S·G·N, G·N, N, 1)`；要求 contiguous（wrapper 归一） |
| `B_mat` | `[B, S, G, N]` | 同上 | 同上 |

动态轴：仅 `B`（batch）在运行时可变；`S/G/N/Q` 由 shape 特化（工厂参数，编译期常量）。

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|---|---|---|---|
| `cb` | `[B, NC, G, Q, Q]` | 与输入同 | `NC = S/Q`；末两维连续；`out_idx=[-1]` 新建，kernel 写满（含严格上三角 0） |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|---|---|---|---|---|
| `cl1` | `(BL, BK)` = (128, 64) | dtype | L1（`shared.l1`）×2 版本 | `C` 行块 K 分块 |
| `bl1` | `(128, 64)` | dtype | L1 ×2 版本 | `B` 行块 K 分块 |
| `x0` | `(128, 64)` | dtype | L0A（`shared.l0a`） | MAD 左操作数 |
| `y0` | `(128, 64)` | dtype | L0B（`shared.l0b`） | MAD 右操作数（`(N,K)` 布局，配 `transpose_B`） |
| `acc` | `(128, 128)` | float32 | L0C（`shared.l0c`） | 累加器（跨 K 块） |
| `cv` | `(64, 128)` | float32 | UB（`shared.dyn`） | 每 AIV 的 L0C 半块（M 切分）；必须 fp32（跨 dtype 被拒绝） |
| `co` | `(64, 128)` | dtype | UB | 掩码 + cast 后的输出半块 |

### 4.4 内存搬运路径

```
GM[C_mat] --T.copy(nd2nz, strided 4-D region)--> L1[cl1] --T.copy--> L0A[x0] ┐
GM[B_mat] --T.copy(nd2nz, strided 4-D region)--> L1[bl1] --T.copy--> L0B[y0] ┴─ T.gemm(transpose_B)
                                                                              │
                                                                    L0C[acc] fp32
                                                                              │
                                          T.dual_copy（硬件双目的，M 切分）      │
                                                    ┌─────────────────────────┘
                                                    v
                                   AIV0: UB[cv 0:64]      AIV1: UB[cv 64:128]
                                        │  T.SimtVF 掩码 + cast   │
                                        v                          v
                                   UB[co] (AIV0)             UB[co] (AIV1)
                                        │  T.dual_copy（按 sid 分片）│
                                        v                          v
                             GM[cb tile 行 0:64]        GM[cb tile 行 64:128]
```

无 GM 中间张量、无 workspace、无跨 block 通信。

### 4.5 片上内存预算

**UB（每 AIV，`ascend-950pr-9579` = 253 952 B）**

| Buffer | Shape | dtype | 大小 (B) |
|---|---|---|---|
| `cv` | (64,128) | float32 | 32 768 |
| `co` | (64,128) | fp16/bf16 | 16 384 |
| **合计（含 32 B 对齐余量）** | | | **49 152 / 253 952（19.4%）** |

smoke 配置（BL=64、BK=64）：`cv` (32,64)fp32 = 8 192 + `co` 4 096 = 12 288 / 253 952（4.8%）。

**L1（524 288 B）**

| Buffer | Shape | dtype | 版本 | 大小 (B) |
|---|---|---|---|---|
| `cl1` | (128,64) | 2 B | 2 | 32 768 |
| `bl1` | (128,64) | 2 B | 2 | 32 768 |
| **合计** | | | | **65 536 / 524 288（12.5%）** |

**L0A / L0B（各 65 536 B）**：`x0`/`y0` (128,64)×2 B = 16 384 B each（25%）；后端在流水下给 L0 双版本（probe P10 生成 `(kk&1)*8192` 环）→ 32 768 B（50%），仍满足。

**L0C（262 144 B）**：`acc` (128,128)fp32 = 65 536 B（25%）。

### 4.6 动态轴定义

| 动态轴 | 声明方式 | 运行时范围 |
|---|---|---|
| `B`（batch） | 工厂按 `batch` 特化编译（`target_cache` 同型），wrapper 按实际 batch 取编译产物 | 1 ~ 数百 |

其余维度（`S/G/N/Q`）在编译期常量；`S % Q == 0` 由 manifest `shape_rules` 保证。非连续输入由 wrapper `.contiguous()` 归一（源算子 `forward` 同款语义）。

### 4.7 JIT 配置

```python
@functools.lru_cache(maxsize=32)          # 或沿用源算子的 target_cache
def _ssd_chunk_coupling_kernel(batch, num_chunks, n_groups, chunk_len, d_state, dtype):
    ...
    return tilelang.compile(main, target="ascend", out_idx=-1)
```

`pass_configs`：默认（`allow_autoschedule=True`），无自定义 flag。

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**：Cube（矩阵主体）+ AIV epilogue 的混合形态（平台模式键 `cube`，§2）。

**判定依据**：每 tile 为 `(128,128,64)` 的乘加累加（`Q²·N` 级），逐元素部分只有掩码 select + cast（`Q²` 级），算力主体在矩阵单元。

### 5.2 Block 划分

```python
BL = 128 if Q % 128 == 0 else (64 if Q % 64 == 0 else 32)   # l 与 s 同取
BK = 64 if N % 64 == 0 else 16                             # K 分块
nlt = nst = Q // BL
grid = batch * num_chunks * n_groups * nlt * nst
```

| 参数 | 取值 | 理由 |
|---|---|---|
| `BL` | 128（`Q=256`）/ 64（`Q=64` smoke） | `acc`(128,128)fp32=64 KB 仅占 L0C 25%，留出 AIV/Cube 异步与双版本余量；`Q % BL == 0` 使 tile 边界全静态 |
| `BK` | 64（`N=128` → 2 次 K 迭代） | L1 双版本仅 64 KB（12.5%），使第 2 块搬入与第 1 块 MAD 重叠；同时避免 `L0A/L0B` 单版本占满 64 KB |
| `num_stages` | 2 | K 迭代数 2，双版本即可覆盖 |
| `threads`（AIV） | 128 | 掩码行宽 128 元素 = 128 lane，一行一拍 |

### 5.3 约束分析

- **对齐约束**：`BK=64`、`BL=128`、AIV 行宽 128 元素（fp16 = 256 B，8×32 B 对齐）；`dual_copy` 要求 M 偶数（128/64 满足）。K 尾块仅 `N % 64 ≠ 0` 时出现（本 workload `N ∈ {64,128}` 无尾块）。
- **UB 容量**：48 KB / 248 KB（§4.5）。
- **L0C 容量**：64 KB / 256 KB；无多版本 L0C（单 tile 累加器，Block 内串行复用）。
- **L1 容量**：64 KB / 512 KB。
- **L0A/L0B**：16 KB（单版本）/ 32 KB（双版本），上限 64 KB。

### 5.4 注意事项（非整除处理）

| 维度 | 是否可能非整除 | 处理 |
|---|---|---|
| `Q / BL`（l、s 方向） | 工厂保证 `Q % BL == 0`（`BL` 取 128/64/32 中能整除 `Q` 的最大者）；`Q % 32 != 0` 或 `Q < 32` 时工厂直接报错 | 无尾块；不做 host padding（padding 会改变 `cb` 语义边界） |
| `N / BK`（K 方向） | 可能（如 `N=96`） | `nk = T.min(N - n0, BK)` 动态 region；源算子同样用补 0（`n_abs < N`），二者等价（补 0 参与乘加不改变结果） |
| `S % Q` | manifest 保证为 0 | 工厂断言，否则报错 |
| GEMM 类非整除 | M/N 方向已静态化；K 方向 region 尾块 | 见上 |

### 5.5 分核策略（物理核数适配）⭐

> 权威标准：`.agents/skills/_shared/standards/core-split-strategy.md` §1（三要素）。分核口径：Cube 核承载 block（AIV 由每 block 的 2 个子核承担 epilogue，随 Cube 块数同比放大）。

- **物理核数依据**（②）：目标型号 `ascend-950pr-9579`（`dav-3510`）。平台记录（2026-11-23 运行时探针）：
  - `Ascend950PR_9579.ini [SoCInfo]`：`ai_core_cnt=28`、`cube_core_cnt=28`、`vector_core_cnt=56`；
  - `torch.npu.get_device_properties(torch.npu.current_device())` 运行时返回值：`cube_core_cnt = 28`、`vector_core_cnt = 56`（`rtGetAiCoreCount() = 28`）。
  - 复核代码（Stage 3 上卡时重跑并记录实际返回）：
    ```python
    import torch, torch_npu
    p = torch.npu.get_device_properties(torch.npu.current_device())
    print(p.name, p.cube_core_cnt, p.vector_core_cnt)   # 期望 28 / 56
    ```
  - 说明：本阶段卡被其他任务占用，未重复实查；上值为平台在同一型号上的实查记录（`hardware.observed_cores`），Stage 3 首次上卡必须复核。

- **逻辑核数计算**（①）：`num_logical_kernels = B · NC · G · nlt · nst = grid`

  | workload | `B·NC·G` | `nlt·nst` | 逻辑核数 (grid) | ÷ 28 核 | 波效率（末波利用） |
  |---|---|---|---|---|---|
  | smoke（Q=64） | 2 | 1 | **2** | 0.07 | 单波（正确性用例，不计性能） |
  | mamba2-780m（wl1） | 16 | 4 | **64** | 2.29 | 3 波，末波 8/28 → 76% |
  | mamba2-2p7b（wl2） | 32 | 4 | **128** | 4.57 | 5 波，末波 16/28 → 91% |
  | mamba2-grouped-prefill（wl3） | 288 | 4 | **1152** | 41.1 | 42 波 → 98% |

- **规模判定**（③）与分核方案：
  - **smoke（2 ≤ 28）：无需适配** —— 逻辑核数小于物理核数，单波完成，每核至多 1 个 block。
  - **wl1/wl2：中等规模** —— 逻辑核数与 28 的关系为 2.29×/4.57×。对齐方案评估：把块数凑成 28 的整数倍（56/84/112）要求 `B·NC·G·nlt²` 被 7 整除，而全部 workload 的该量是 2 的幂（16/32/288 → 288 = 2⁵·3²，`288·4^k` 也不含因子 7），**在方块 tile 下不可达**；若为凑数把 `BL` 调到 256（wl1 → 16 块，0.57 波，43% 空转）或 64（wl1 → 256 块，9.1 波，91% 但块工作量为 1/4）都只是把不均衡挪位。故保留 `BL=128`：wl1 末波空转 20/28（占整kernel 时间 ≈ 1/3 波），wl2 末波空转 12/28，均小于把 tile 改到 256 的损失；`BL ∈ {64,128}` 与 `BK ∈ {64,128}` 作为 Stage 4 参数级 A/B（§1.6.4）。
  - **wl3：极大规模** —— 1152 块 = 41 波，波效率 98%，**不做"固定启动 28 个核 + 核内串行"（persistent）**：本工具链未认证常驻式核内循环/跨 block barrier（`programming.md` 同步节、`bottlenecks.md`），而 1-D 网格的多波调度是硬件块调度器的既有行为，核内串行化不摊薄启动开销反而会引入手工同步风险。该判定与 §0.5「无跨 block 同步」的保留项一致。
  - 负载不均的剩余手段：形状特化工厂（按 `B·NC·G` 与 `Q` 分派 `BL`），留待 Stage 4 依实测时延决定（§1.6.4 裁决计划含 shape 分派条款）。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|---|---|---|---|
| `(batch, l-tile, s-tile)` | block 级并行 | `T.Kernel(grid)` | 每 block 一个输出 tile，无跨块依赖 |
| 上三角 tile 门控 | block 级条件表达式（非分支包循环） | `T.if_then_else(...)` 作为 `T.Pipelined` 的 trip | 被掩码 tile 不计算（§1.6.1 O3）；避免条件化流水导致 sync 失配（§0.6 R4） |
| K 维（`⌈N/BK⌉`） | tile 级迭代 + 双版本流水 | `T.Pipelined(trips, num_stages=2)` + `T.annotate_buffer_versions({cl1:2, bl1:2})` | K 分块累加，第 2 块搬入与第 1 块 MAD 重叠 |
| epilogue 元素级（`64×128`/AIV） | 向量化 | `T.Vector(vector=2) as sid` + `T.SimtVF(threads=128)` + `T.Parallel(BL//2, BL)` | 掩码 select + cast（§1.6.2 #1/#2） |
| 输入/输出 tile 搬运 | DMA 整块 | `T.copy`（GM→L1、L1→L0）/ `T.dual_copy`（L0C→UB、UB→GM） | 二维 strided 整块搬，无逐元素 gather |

> 无逐元素标量循环：§1.6.2 表中 #7–#10 保留项均为 block 级或 tile 级，不在元素维。

### 6.2 循环伪代码

见 §3.3（与 probe 逐行一致）。调度要点：`bcg → lt → st` 的解码顺序让同一 `(b,c,g)` 的 3 个 GEMM tile 相邻发射，`C` 行块与 `B` 行块在 L2 复用；`st` 为最快轴，相邻块写输出相邻 tile（GM 写地址连续段）。

### 6.3 流水线优化

- `num_stages=2`：K 迭代数为 2（`N=128/BK=64`）或 1（`N=64`），双版本 L1 足以覆盖；`MaterializeMultiBuffer` 在后端给 `cl1/bl1`（及流水下的 L0A/L0B）建立 `(kk&1)` 版本环（probe P10 生成码可见）。
- L0C 累加器单版本：K 迭代串行复用，`clear_accum=(kk==0)` 只在首块清零（与 `gemm-tiles.md` 的"跨 K 块保留累加器"一致）。
- Cube/AIV 异构重叠：L0C→UB 之后 AIV 做掩码/Cast/写回时，下一 block 的 Cube 段可在另一核（或同核的不同 block）推进，由 AutoSchedule 的 core 分配完成；本设计不手写 flag。
- 不做 split-K：`N ≤ 128`，单核可一次累完，跨核归约代价大于收益（§1.6.0 R3「跨核代价」列）。

### 6.4 尾块处理

- l/s 方向：无尾块（`Q % BL == 0` 由工厂保证，§5.4）。
- K 方向：`nk = T.min(N - n0, BK)`，配合 `x0[:, :nk]`/`y0[:, :nk]` 的 region 形式传给 `T.gemm`（`gemm_op.py` 允许 region 动态、分配静态），与源算子"越界补 0"等价。
- 严格上三角 tile：`trips = 0`，K 循环不执行，`acc` 内容任意，由掩码 epilogue 写 0（§1.6.1 O3，机器验证 E4）。

---

## 7. 同步策略

### 7.1 同步模式

**模式**：**自动同步**（`ascend/pipeline.py` 的 `allow_autoschedule` 默认开启 → `AutoSchedule` + `InsertSync` + `MaterializeMultiBuffer`）。不手写 `ascend_set_flag/wait_flag`、不使用跨核 flag（平台未认证手工跨核配对）。

### 7.2 同步点说明

| 位置 | 同步方式 | 理由 |
|---|---|---|
| GM→L1 搬入 vs L1→L0 读取 | 自动（MTE2→MTE1 事件，probe 生成 `asc_sync_notify/wait(PIPE_MTE2, PIPE_MTE1, kk&1)`） | 双版本环的生产者/消费者配对 |
| L1→L0 vs MAD | 自动（MTE1→M 事件） | MAD 需等操作数就位 |
| MAD vs L0C→UB | 自动（M→FIX 事件，probe 生成 `asc_sync_notify(PIPE_M, PIPE_FIX, 0)` + wait） | 累加完成后才搬出 |
| L0C→UB vs AIV 掩码 | 自动（FIX→V intra，probe 生成 `asc_sync_intra_arrive(PIPE_FIX, …)` / `asc_sync_intra_wait(PIPE_V, …)`） | AIC 写 UB 完成后 AIV 才能读 |
| AIV 掩码 vs UB→GM | 自动（VF 内按线程隐式有序 + MTE3 事件） | 同缓冲读写 |
| 流水 drain | 自动（循环外尾部 `asc_sync_wait`，probe P5/P8 已核对 notify/wait 数量在 `trips ∈ {0,2}` 下均平衡） | trip 置 0 时不会悬挂（§0.6 R4 的证据） |

### 7.3 pass_configs 配置

```python
# 使用默认 pass context：TL_ENABLE_AUTO_SCHEDULE = True（Ascend 默认）
# 不自定义 pass_configs；不设 tl.enable_auto_schedule=False（无手工同步需求）
```

---

## 8. 验证方案

### 8.1 Golden 函数

> 以 **§0.1 源算子语义**为唯一依据（移植源仓参考实现 `ssd_chunk_coupling_fwd_ref`），**不复刻** §0.6 的 tile 结构 / 掩码实现，保证验证独立性。

```python
def golden_ssd_chunk_coupling(C_mat, B_mat, chunk_len):
    """cb[b,c,g,l,s] = sum_n C[b,cQ+l,g,n] * B[b,cQ+s,g,n] for s <= l else 0."""
    b, s, g, n = C_mat.shape
    nc = s // chunk_len
    # 与源仓参考实现一致：fp32 累加，最后一次性舍入到输入 dtype
    c = C_mat.reshape(b, nc, chunk_len, g, n).float()
    bb = B_mat.reshape(b, nc, chunk_len, g, n).float()
    cb = torch.einsum("bcqgn,bcsgn->bcgqs", c, bb)          # [b, nc, g, q, q]
    mask = torch.tril(torch.ones(chunk_len, chunk_len,
                                 device=C_mat.device, dtype=torch.bool))
    cb = torch.where(mask, cb, torch.zeros((), dtype=cb.dtype))  # 严格上三角精确 0
    return cb.to(C_mat.dtype)
```

与 `examples/TileOPs/tileops/testing/mamba2_reference.py` 的 `ssd_chunk_coupling_fwd_ref` 的唯一差别：把 `(cb * mask)` 换成 `torch.where(mask, cb, 0)`，使非有限输入下的上三角也是精确 0（与 §0.1 边界语义一致）；对有限输入两者逐位相同。

### 8.2 精度标准

| dtype | atol | rtol | 依据 |
|---|---|---|---|
| float16 | 1e-3 | 1e-3 | 仓库 coupling 用例 `COUPLING_ATOL/COUPLING_RTOL`（`examples/TileOPs/tests/ops/test_mamba.py`） |
| bfloat16 | 1e-3 | 1e-3 | 同上（两侧均为 fp32 累加后一次舍入，差异应为 0–1 ulp） |

补充：`N ≤ 128` 的内积在 fp32 中累加，累加分组变化带来的相对偏差 ≤ 1e-7，远小于一次 bf16 舍入（2⁻⁸）；若出现个别元素跨 ulp，按仓库 bf16 先例（同文件 scan 用例 `atol=2e-3`）放宽 atol，并在 Stage 3 记录实际最大误差。

### 8.3 L0 门槛测试计划

| 用例 | Shape | dtype | 覆盖点 |
|---|---|---|---|
| L0-1 smoke | `B=1, S=128, G=1, N=64, Q=64`（NC=2） | float16 | 唯一 tile 即对角 tile（掩码方向 `s<=l`、对角线保留）；`BL=64`（非 128）配置；K 迭代 1 次；`G=1` 的行跨度 |
| L0-2 smoke | 同上 | bfloat16 | dtype 路径（cast 舍入、UB 双缓冲 dtype） |
| L0-3 掩码方向断言 | L0-1 的基础上，额外断言：`cb[0,0,0,0,0] == Σ C[0,0,0,:]·B[0,0,0,:]`（对角元素**非零且等于内积**）与 `cb[0,0,0,0,1] == 0`（严格上三角为 0） | float16 | 防掩码方向写反（经典坑）、防 `<=` 写成 `<` |

L1/L2/boundary（三个 manifest workload、非连续输入、二次调用分配契约）由 `tilelang-op-develop` 按分层套件补齐，本设计不枚举。L0 全通过 + 与 golden 逐元素比对（`atol=rtol=1e-3`）是进入 L1 的门槛。

---

## 9. 风险点与注意事项

### 9.1 已知约束

1. **`T.MixedKernel` + 显式 `T.Cube()/T.Vector()` 的等价写法不可用**：probe P3/P4 编译失败（`Cannot assign scalar task because it accesses incompatible core-local memories on both Vector (UB) and Cube (L1/L0) cores`，见 `history_version/design_probe_compile.py` 与 `probe_out/probe_fail_P3-mixed-mask-fp16.log`）。本设计因此采用"`T.Kernel` 自动调度 + 显式 `T.Vector` 取 sid"的形态（probe P5–P12 通过）。
2. **L0C→UB 不能跨 dtype**（`copy_op.py:139-152`）⇒ 必须 `cv`(fp32) 中转 + `co`(dtype) 输出，UB 预算按两块计（48 KB）。
3. **`dual_copy` 形状约束**：恰好一个尾维 2:1、全 M 偶数（`copy_op.py:120-135`）⇒ `BL ≥ 32`；`BL=16` 的极小配置不可用（工厂最低取 32，且 `Q % 32 != 0` 直接报错）。
4. **`T.Kernel` 单维网格**：所有并行度靠一维解码，`bid` 的除法/取模为 block 级标量（已量化，§1.6.2 #7）。
5. **动态 trip 的生成码形态**：probe P8/P10 把 `nk = min(N-64kk, 64)` 简化为 `128 - max(kk,1)*64`；对实际 `trips ∈ {0,2}` 该式恒等于 64（正确），但属后端简化产物——Stage 3 必须用 L0 数值覆盖 `trips=0`（上三角写零）与 `trips=2`（正常 GEMM）两条路径。
6. **`Q % 32 != 0`、`Q < 32`、`S % Q != 0` 的形状不支持**（工厂断言报错）：manifest 与本任务 workload 均不涉及；若 TileOPs 侧出现此类 shape，需回 Stage 1 增加 padding 方案。
7. **非连续输入**依赖 wrapper 的 `.contiguous()`（源算子 `forward` 已有同款）；kernel 本身按 contiguous 设计。
8. **probe 结论绑定工具链版本**（TileLang ascend 0.1.15 / CANN 9.2.1 / `ascend-950pr-9579`）；工具链升级后须重跑 `history_version/design_probe_compile.py`。
9. **AIV sid 语义**（`asc_get_sub_block_id()`）与 M 切分方向的对应关系（sid=0 → tile 上半 64 行）由本设计假设；probe 只能证明编译正确，**数值上必须由 L0-3 的对角线断言确认**（若方向反了，对角断言会落在错的半块）。

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|---|---|---|---|
| 掩码方向写反（保留上三角） | 谓词写成 `s_abs >= l_abs` | 输出与 golden 完全不同 | L0-3 对角线断言；`verify_equiv.py` E2 固定 `s<=l` 判据 |
| 用乘法做掩码（`v*1[s≤l]`） | 以为"乘 0 更省" | `nan/inf` 输入时上三角出现 nan（`verify_equiv.py` E3：28/64 违反） | 必须用 `T.if_then_else` select |
| L0C→UB 直接写 fp16 缓冲 | 少分配一块 UB | 前端直接报错（跨 dtype 拒绝） | fp32 `cv` + dtype `co` 两段 |
| `if` 包住 pipelined GEMM | 照搬源算子分支 | 流水 drain 的 set/wait 可能失配（挂死） | trip 置 0（§0.6 R4） |
| 用 `tilelang.language` / `target="npuir"` 的 API | 参考 `examples/gemm/` 的写法 | 编译到另一后端，无法在本平台执行 | 只用 `tilelang.ascend.language`（§3.5.3 明确） |
| K 循环每块都 `clear_accum=True` | 误以为要清零 | 只剩最后一块的部分和 | `clear_accum=(kk == 0)` |
| UB 溢出 | `BL` 提到 256 且未减 `cv/co` | 编译失败 | `cv/co` 按 `BL/2` 分配；UB 预算表 §4.5 复算 |
| 以为上三角 tile 不必写 | 跳过 store | 输出含未初始化值 | 统一掩码 epilogue 必写整块（`out_idx` 新建、无预置零） |

### 9.3 特殊场景处理

- **极小 shape**：`Q < 32`（如 `Q=16`）不在支持范围（`BL ≥ 32` 是 `dual_copy` 的 2:1 分片下限，且 AIV 行宽会低于一个向量），工厂报错；`Q=64/N=64`（smoke）已覆盖实际最小用例。
- **非整除 K**：`N=96` 等 → `nk` region 尾块，`T.gemm` 用 `x0[:, :nk]`（分配仍 `(BL,BK)`）。
- **大批量 group**：`G=8` 时行跨度 `G·N·2 = 2048 B`，probe P11 已核对代码生成的跨度参数。
- **`trips = 0` 的 tile**：`acc` 未初始化 → 掩码写 0；不得依赖 `acc` 内容（O3/E4）。

---

## 10. 交付清单

```
examples/ssd_chunk_coupling/_ssd_chunk_coupling_kernel/
├── DESIGN.md                              # 本文件
├── RETROSPECTIVE.md                       # Stage 1 复盘（Phase 8）
├── verify_equiv.py                        # D-1 等价性机器验证（已执行，全部 EQUIV_PASS）
└── history_version/
    └── design_probe_compile.py            # D-3 设计期编译探针（12 用例，10 通过）
```

下游产物（本阶段不产出）：`_ssd_chunk_coupling_kernel.py`（kernel + golden + 分层测试，Stage 3）、`perf_opt/`（Stage 4）。

---

## 11. 移交与待确认项

| 项 | 内容 | 责任阶段 |
|---|---|---|
| 单卡实查复核 | `torch.npu.get_device_properties` 的 `cube_core_cnt/vector_core_cnt` 实查值（本阶段用平台记录 28/56） | Stage 3 首次上卡 |
| AIV sid 方向 | 由 L0-3 对角线断言确认 sid=0 对应 tile 上半 64 行 | Stage 3 |
| 动态 trip 数值路径 | `trips=0` 与 `trips=2` 两条路径的 L0 数值 | Stage 3 |
| 主选 vs 备选 B | 按 §1.6.4 裁决计划（阈值：备选 < 主选 × 0.95 则翻转） | Stage 4 |
| `BL/BK` 参数 A/B | `BL ∈ {64,128}`、`BK ∈ {64,128}`，含 wl1 波效率（64/28）的实测复核 | Stage 4 |
| 设计期估算对账 | 实测 baseline 时延 vs §1.6.0 估算行，偏差 > 2× 时归因并回写 | Stage 4 |
