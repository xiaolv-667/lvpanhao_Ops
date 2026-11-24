# `_engram_project_kernel` 算子设计文档

> **迁移任务（harness 子模式），首次设计 v1**。源算子：
> `/workspace/TileOPs/src/tileops/_kernels/sequence_modeling/engram/decode.py` 的
> `_engram_project_kernel` 工厂（第 25–111 行）与它的 host 侧调用方
> `EngramDecodeKernel.forward`（第 375–410 行）。
> 同仓抽取件：`examples/TileOPs/tileops/kernels/sequence_modeling/engram_decode/engram_decode.py`
> （Part A 的 `_engram_decode_fwd_kernels.py` 为工厂逐字副本）与
> `examples/TileOPs/tileops/kernels/sequence_modeling/engram_decode/_engram_decode_fwd_kernels.py`。
>
> 目标平台：`ascend-950pr-9579`（dav-3510，28 Cube / 56 Vector，UB 248 KiB / L1 512 KiB /
> L0C 256 KiB / L0A·L0B 各 64 KiB，L2 128 MiB，cube_freq 1650 MHz），工具链 `ascend`
> （TileLang ascend 0.1.15 + CANN 9.2.0 编译 SDK / 9.2.1 本站 runtime，`target="ascend"`，
> `import tilelang.ascend.language as T`）。平台选择由 `statectl platform --task-dir` 落库
> （2026-11-23）。
>
> **设计范围**：仅 `_engram_project_kernel`——两个投影（Cube 段）+ cache 平移。同一
> `EngramDecodeFwdOp` 的第二个 launch `_engram_step_kernel`（RMSNorm / gate / 膨胀卷积 /
> SiLU / history 末行）是**另一个独立设计任务**，本文只在 §0.1、§1.5、§4.2 说明**跨函数
> 中间量契约**（fp32 `kv`、`new_conv_state` 的前 `max_conv_len-1` 行）与边界。
>
> **只读参考（不照抄）**：`/workspace/TileOPs-ascend950/src/tileops/kernels/ascend/sequence_modeling/decode.py`
> 是 950DT 上已验证的 Ascend 实现（两个 launch：Cube 投影段 + Vector 逐行段）。本文在
> §0.4/§0.5 标注借用了它的哪些优化手段、以及 950PR（28/56 核）与 950DT（32/64 核）的适配差异；
> 它的**边界划分与本任务不同**（把 cache shift 放进 step kernel），差异见 §0.5 第 11 行与 §0.6 R4。
>
> 本文档由 `tilelang-ascend` → `tilelang-op-design` 流程产出：Phase M0（三问解读，§0.1–§0.4）
> → Phase R（算法调研，§1.6.0）→ Phase M1（耦合性判定与重设计，§0.5–§0.6）→ Phase 2
> （算法级优化，§1.6.1–§1.6.3）。设计期做了 3 个编译探针（§11.1，全部 compile-only、
> `ASCEND_RT_VISIBLE_DEVICES=""`、未占用设备）。首次设计，无「相对上一版的关键调整」。

---

## 0. 源算子解读与迁移分析

### 0.1 源算子语义（做什么）

`_engram_project_kernel(batch, d_mem, d_padded, max_conv_len, dtype)` 是一个**工厂函数**：
返回内层 JIT 工厂 `_func(block_m, block_n, block_k, threads, num_stages)`，内层再专门化出一个
`@T.prim_func`。一次 decode step 的「Cube 段 + cache 平移」在数学上是两件**互不依赖**的事：

```
K_out = e_t · W_K                      # (batch, d_mem) × (d_mem, d_padded) -> (batch, d_padded)
V_out = e_t · W_V                      # 同一 e_t，另一组权重
new_conv_state[b, s, :] = conv_state[b, s+1, :]     # s = 0 .. max_conv_len-2（逐元素 dtype 拷贝）
new_conv_state[:, max_conv_len-1, :]                # 本 launch 不写（由 step kernel 写）
```

- **I/O 契约**（源码 `@T.prim_func main` 签名，第 84–92 行）：
  `e_t (batch, d_mem)`、`W_K/W_V (d_mem, d_padded)`、`conv_state (batch, max_conv_len, d_padded)`
  为输入 dtype（fp16/bf16）；`kv (2, batch, d_padded)` 为 **fp32**；`new_conv_state
  (batch, max_conv_len, d_padded)` 为输入 dtype。所有输出**由调用方预分配**（`out_idx=[]`），
  `forward` 里 `kv = torch.empty(..., dtype=torch.float32)`（第 401 行）——这就是与 step kernel
  之间的跨 launch 中间量契约。
- **规约语义**：收缩轴 = `d_mem`，沿 `d_mem` 分 `block_k` 段**串行累加**，累加器是 L0C 上的
  fp32（`accum_dtype = "float"`，第 40 行），进循环前 `T.clear(acc)`（第 53 行）一次、跨 K 段保留。
  `block_k` 之外的第二个累加维度不存在。
- **dtype 语义**：输入 fp16/bf16 → 矩阵乘 fp32 累加 → **原样写 fp32 到 `kv`（不舍入到输入 dtype）**；
  cache 平移是**纯 dtype 拷贝**（逐位等价，不做任何算术）。注意与 step kernel 的分工：
  step 段自己把 `kv` 舍入回输入 dtype 再 RMSNorm（GPU 源第 167–169 行直接读 fp32、step 里
  `k_local` 是 fp32 片段），所以本 launch 的 fp32 输出是**契约要求**，不能在 project 段提前舍入。
- **边界语义**：
  - **m 尾块**：`row0 + i >= batch` 的行不写（第 81 行守卫）；装载时用
    `T.min(row0 + i, batch - 1)`（第 57 行）夹取，越界行读到的是最后一行的副本，只落在被丢弃的
    死行上。
  - **k 尾块**：`d_mem % block_k != 0` 时走第 63–79 行的谓词分支：`k_idx >= d_mem` 的
    `e_shared` / `w_shared` 元素**填 0**，`T.gemm` 仍按 `block_k` 整数宽累加（零不贡献）。
  - **n 尾块**：`col0 + j` 无守卫——源码假设 `d_padded % block_n == 0`
    （`d_padded` 在设计期由 host 保证，`block_n = min(64, d_padded)`）。
  - **`max_conv_len == 1`**：`max_conv_len > 1` 守卫（第 101 行）使平移整段消失，`new_conv_state`
    完全由 step kernel 写。
  - **`L < max_conv_len`（partial cache）**：kernel 侧无感知；host `forward` 先左填充到
    `max_conv_len`（第 388–390 行），因此 kernel 永远看到满长 history。
  - **非有限值**：无特殊处理，按 IEEE 传播（fp32 累加器 + 逐元素拷贝）。
- **host 侧语义**：`forward`（第 375–410 行）做两件**纯元数据/填充**动作——`conv_state` 左填充到
  `max_conv_len`、把 `h_t / conv_state / W_K / W_V / rms_w* / conv_w` 的最后一维 0 填充到
  `d_padded`（第 391–399 行），返回时再裁回 `d`（第 408–409 行）。这些是**实现手段**而非语义，
  NPU 侧沿用同一约定（§4.6）。

**语义保持基线**：§8.1 的 golden 以本节语义为唯一依据，并直接移植源仓 torch 参考
`engram_decode_step_torch`（`examples/TileOPs/tileops/workloads/sequence_modeling.py`）的投影与
平移部分。

### 0.2 源算子输入输出

本 kernel 的签名（6 个张量）与 5 条 manifest workload 的取值
（`examples/TileOPs/tileops/manifest/sequence_modeling.yaml` 的 `EngramDecodeFwdOp`）：

| 参数 | 方向 | Shape | dtype | 说明 |
|------|------|-------|-------|------|
| `e_t` | in | `(batch, d_mem)` | fp16 / bf16 | 本步 memory embedding（未 pad） |
| `W_K` | in | `(d_mem, d_padded)` | 同 `e_t` | K 投影权重；`d_padded = align_up(d, 256)`（wrapper 侧 pad） |
| `W_V` | in | `(d_mem, d_padded)` | 同 `e_t` | V 投影权重 |
| `conv_state` | in | `(batch, max_conv_len, d_padded)` | 同 `e_t` | 已由 host 左填充到 `max_conv_len` 的 history |
| `kv` | **out** | `(2, batch, d_padded)` | **float32** | `kv[0] = K`、`kv[1] = V`；跨 launch 契约 |
| `new_conv_state` | **out** | `(batch, max_conv_len, d_padded)` | 同 `e_t` | 只写第 `0 .. max_conv_len-2` 行 |

源算子输出 shape（迁移必需字段）：`kv` = `(2, batch, d_padded)` fp32、
`new_conv_state` = `(batch, max_conv_len, d_padded)` 输入 dtype；crop 后对外的 `d` 维为 `d`。

| workload | batch | d_mem | d | d_padded | max_conv_len | conv_kernel_size | dilation | dtype |
|---|---|---|---|---|---|---|---|---|
| decode-s1（smoke） | 1 | 256 | 256 | 256 | 9 | 4 | 3 | float16 |
| serving-b1 | 1 | 512 | 256 | 256 | 12 | 4 | 3 | float16 |
| serving-b4 | 4 | 1024 | 512 | 512 | 20 | 4 | 5 | float16 |
| serving-b8 | 8 | 512 | 256 | 256 | 18 | 4 | 3 | bfloat16 |
| decode-partial-cache | 2 | 512 | 256 | 256 | 12 | 4 | 3 | float16 |
| （测试 fixture，非 manifest）ktail-519 | 17 | 519 | 256 | 256 | 12 | 4 | 3 | float16 |

`ktail-519` 只出现在 `examples/TileOPs/tests/ops/test_engram.py` 的 `EngramDecodeFixture`
（第 338 行），它是**唯一**会让 `d_mem % block_k != 0` 的形状（§9.3）。

### 0.3 实现算法解读（怎么算）

**计算步骤分解**（源码 `decode.py:46-109`，每个计算语句都归入下列 6 步之一）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式的部分 |
|------|------|------|------|-------------------|
| 1 | `row0=row_block*block_m`、`col0=col_block*block_n`、`which∈{0,1}` 选 K/V | grid 坐标 | 任务索引 | 决定哪个输出 tile、哪组权重 |
| 2 | e_t tile 装载（`T.Parallel(block_m, block_k)`，行夹取 `min(row0+i, batch-1)`） | `e_t` GM | `e_shared` SMEM | `K = e_t·W_K` 的左操作数 tile |
| 3 | W tile 装载（`T.copy(weight[k0:k1, col0:col0+block_n], w_shared)`） | `W_K`/`W_V` GM | `w_shared` SMEM | 右操作数 tile |
| 4 | 分块矩阵乘累加（`T.gemm(e_shared, w_shared, acc)`，K 段循环 `T.Pipelined`） | `e_shared`,`w_shared` | `acc` fragment(fp32) | `W_K^T`/`W_V^T` 收缩 |
| 5 | 输出写回（`T.Parallel(block_m, block_n)` + `row0+i<batch` 守卫） | `acc` | `kv[which]` GM | `K`/`V` 落 fp32 |
| 6 | cache 平移（仅 `which==0 and row_block==0`）：`T.Parallel(batch*(max_conv_len-1), block_n)` 逐元素拷 | `conv_state` | `new_conv_state` | `new_conv_state[0:mcl-1] = conv_state[1:mcl]` |

K 方向两条互斥分支：`d_mem % block_k == 0`（精确，第 54–62 行）与否则的**谓词填零**分支
（第 63–79 行）。`d_mem` 的其他语句（`ceildiv`、`T.min`）只影响循环次数与有效区间。

**数据流与内存访问模式**（源硬件视角，CUDA）：

```
GM[e_t]      --T.Parallel(夹取行)--> SMEM[e_shared] --(tensor core 每 K 段消费)--> acc(fragment)
GM[W_K/W_V]  --T.copy(连续 2-D)----> SMEM[w_shared] --/
acc          --T.Parallel(守卫行)--> GM[kv[which]]            (fp32)
GM[conv_state] --T.Parallel(逐元素)--> GM[new_conv_state]      (dtype 拷贝，不经 SMEM)
```

**循环与并行结构**：grid = `(ceildiv(d_padded, block_n), 2, ceildiv(batch, block_m))` 三维
（`T.Kernel`，`threads=128`）；每 block 一条 `T.Pipelined(ceildiv(d_mem, block_k), num_stages=4)`
的 K 段循环；block 内三处元素级并行（两个装载 + 一个写回 + 一处平移）。默认
`block_m=16, block_n=min(64,d_padded), block_k=_projection_block_k(d_mem)（64/32/16）, num_stages=4`
（第 319–328 行）。

**host 侧逻辑**：`forward` 的 `conv_state` 左填充（对齐 `max_conv_len`）、`d_padded` 右填充、
以及返回时的 crop（§0.1 末条）。`kv`/`new_conv_state`/`y_t` 由 host 预分配（`out_idx=[]`）。
`_projection_block_k(d_mem)` 只是「挑一个受支持的 MMA K tile」（16/32/64），不是语义。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| O1 | SMEM tiling（`alloc_shared` 的 `e_shared`/`w_shared`） | 权重 tile 只从 GM 读一次，供整段 K 复用 | 分块驻留 shared memory | shared memory（CUDA 48–228 KiB/block） | 硬件强相关 |
| O2 | Tensor Core `T.gemm` + fp32 累加 | 矩阵乘算力 | mma fragment，K 段累加 | tensor core | 硬件强相关 |
| O3 | fragment 内累加（`acc` 常驻、跨 K 段不清零） | 免去中间结果落 GM | 寄存器/fragment 累加 | 大寄存器堆 | 硬件强相关 |
| O4 | `T.Pipelined(num_stages=4)` 异步流水 | 装载与 mma 重叠 | cp.async / 多缓冲 SMEM | 异步拷贝引擎 | 硬件强相关 |
| O5 | `block_m=16` 行 tile + 行夹取 `min(row0+i, batch-1)` | batch < tile 时仍只发一个 tile，且不越界 | 夹取使死行读到合法地址 | 无（纯索引） | 可移植（夹取本身应舍弃，见 O10） |
| O6 | K 尾块谓词填零（`if_then_else(k_idx < d_mem, ..., 0)`） | 支持 `d_mem % block_k != 0` | 逐元素选择 + 零元不贡献 | 无（纯数学） | 可移植 |
| O7 | **把 cache 平移融进同一 launch**，且只让 `which==0 and row_block==0` 的 block 做 | 省一次 kernel 启动、省一次 `conv_state` 的重复扫描 | 复用已有 grid 的一个切片 | 无（调度层） | 可移植（但落点要重设计，§0.6 R4） |
| O8 | 输出写回的 `row0+i<batch` 守卫 | 丢弃死行 | 逐元素条件写 | 无 | 可移植 |
| O9 | 三维 grid 把两次投影拆成 grid 的第二轴（`which`） | 两块权重不共享同一 tile 缓冲，block 调度简单 | grid 轴分解 | CUDA grid 维度 | 硬件强相关（NPU 只有 1-D grid） |
| O10 | 平移用 `T.Parallel(batch*(max_conv_len-1), block_n)` 逐元素并行 | 平移无归约、天然并行 | 线程并行映射到 `(b, s, j)` | CUDA thread 域 | 硬件强相关（NPU 无 thread 域，应改 DMA） |
| O11 | （来自 950DT 只读参考，非源算子）L1/L0 显式分层 + `transpose=True` 的 GM→L1 `dn2nz` 转置 + `transpose_B=True` 的 NT GEMM | 权重在搬运途中完成转置，L0B 直接可消费 | GM→L1 转置搬入 | Ascend MTE2 的 nd2nz/dn2nz 通道 | 硬件强相关（来源：950DT 参考 `_build_project`，见 §0.5 备注） |

> 识别不出来 ≠ 不存在。O1–O11 是本文后续判定的全集；O11 属于「目标平台已验证实现」的
> 手段（不是源算子自带的），列在此处是为了在 §0.5 明确它与 O1/O4 的**替代关系**而不是
> 额外叠加，避免重复搬运。

### 0.5 硬件耦合性分析与 NPU 适配决策

**判定问题**：实现算法和优化手段是硬件强相关吗？能用在 NPU 上吗？

| # | 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|---|------|------|-----------|-----------------|------|---------------------|
| 1 | 数学语义（两次投影 + cache 平移），含 fp32 累加与 fp32 `kv` | 语义 | 无 | — | **保留** | 语义层无条件保留（§0.1） |
| 2 | O1 SMEM tiling | 优化 | shared memory | 有（L1 512 KiB / UB 248 KiB） | **等价替换** | 改用 `T.alloc_l1` 驻留 `e/w` tile（80 KiB/版本预算，§4.5）；证据 `examples/shared_expert_mlp/_gemm_basic_kernel/DESIGN.md`、`.agents/platforms/toolchains/ascend/knowledge/gemm-tiles.md` |
| 3 | O2 tensor core `T.gemm` | 优化 | mma / tensor core | 有（Cube AIC + `asc_mmad`） | **等价替换** | `T.gemm(A_l0a, B_l0b, C_l0c, transpose_B=True, clear_accum=(kt==0))`；CUDA pattern/`threads` 语义消失 |
| 4 | O3 fragment 累加 | 优化 | 寄存器堆 | 有（L0C 256 KiB，每 block 仅 4 KiB） | **保留** | 累加器 = `T.alloc_l0c((block_m, block_n), "float32")`，跨 K 段保留 |
| 5 | O4 `T.Pipelined(num_stages=4)` | 优化 | cp.async + 多缓冲 SMEM | 有（`T.Pipelined` + `annotate_buffer_versions` + AutoSchedule） | **等价替换（降级）** | 保留 `T.Pipelined(k_tiles, num_stages=2)` 与 4 个 L1/L0 缓冲的版本注解；但编译期警告显示调度器**未**启用多缓冲（§11.1 探针 H1），故按「顺序 K 循环 + 自动同步」设计，流水收益留给 Stage 4 |
| 6 | O5 行 tile（`block_m=16`） | 算法 | 无 | 有 | **保留** | `block_m=16`（Cube 的 M 向 fractal 粒度）；`am = min(batch - m0, block_m)` 直接把真实 M 交给 `asc_mmad`（探针 H2：batch=8 生成 `asc_mmad(..., 8, 256, 64, ...)`），因此 M 方向**没有**乘加浪费 |
| 7 | O5 的行夹取 `min(row0+i, batch-1)` | 优化 | 无 | 无必要 | **舍弃** | NPU 侧区域拷贝只覆盖有效行 `[m0, m0+am)`，不存在越界行，夹取会额外制造一行重复读。等价性与收益由 §1.6.1 A2（机器验证 `A4`）覆盖 |
| 8 | O6 K 尾块谓词填零 | 算法 | 无 | 部分（`T.clear` 只接受 C0 对齐区域，探针 H4 失败） | **重新设计** | 改为把**有效 K 区间**作为区域交给 `copy`/`gemm`（`ak = min(d_mem-k0, block_k)`），由 `asc_mmad` 的 K 参数精确表达收缩长度 → §0.6 R2 |
| 9 | O8 输出守卫 + O10 平移逐元素并行 | 优化 | CUDA thread 域 | 无 thread 域；元素级写要落到 Vector 单元 | **重新设计** | 写回改 `T.copy(res[0:am,0:an], kv[which, m0:m0+am, n0:n0+an])`（L0C→GM 区域拷贝）；平移改 GM→UB→GM 的 2-D 区域拷贝 → §0.6 R4 |
| 10 | O9 三维 grid | 优化 | CUDA 三维 grid | 无（`T.Kernel` 只接受 1-D，多维直接拒绝） | **重新设计** | 1-D grid + 整数分解 `m0/which/n0 = f(bx)` → §0.6 R1 |
| 11 | O7 平移的落点（同 launch、单切片） | 优化 | 无 | 有（`T.Kernel` 内 UB 代码会被自动提升为 `__mix__(1,2)`） | **保留意图 + 重新设计落点** | 意图（省一次 launch）保留：平移留在本 launch；落点改为 Vector 区域（§0.6 R4）。**与 950DT 参考的差异**：参考把平移挪进 step kernel（它的 `_build_step` 逐 batch 行 chunked copy），本任务按 harness 的函数边界（`project` 必须产出 `new_conv_state[:, :-1, :]`）不能照搬；代价与收益见 §1.6.0 R4 |
| 12 | `threads=128` / `step_threads` 启动参数 | 优化 | CUDA block 线程数 | 无（`T.Kernel` 签名没有 `threads`） | **舍弃** | 由 grid + 编译器自动 AIC/AIV 切分取代（§2.3） |
| 13 | 950DT 参考的 `_pick_block_k`（L0B 64 KiB 反解 `block_k`） | 优化 | Ascend L0B 64 KiB | 有 | **保留（承接 O1/O4 的容量意图）** | `block_k = min(64 KiB / (block_n × itemsize × num_stages), 256)` = 256（§5.3）；950DT→950PR 差异：容量相同（64 KiB）、核数 32/64 → 28/56，故只影响 grid 波次判定，不影响该反解 |

> 每项处置的依据均指向 `.agents/platforms/toolchains/ascend/references/programming.md`
> 的接口条款、`examples/shared_expert_mlp/_gemm_basic_kernel/DESIGN.md` 的同平台先例、
> 或 §11.1 的设计期探针结果。判定为「重新设计」的 4 项（#8/#9/#10/#11）在 §0.6 逐项给出
> NPU 新算法与语义保持论证。

### 0.6 NPU 算法重设计

**重设计项 R1：三维 grid → 1-D grid + 整数分解（承接 O9 / §0.5 #10）**

- **源方案**：`T.Kernel(ceildiv(d_padded, block_n), 2, ceildiv(batch, block_m))`，用 grid 的第二轴
  区分 W_K/W_V、第三轴区分 batch tile。性能意图：两块权重 / 多个 batch tile 各自独立调度，
  互不共享 tile 缓冲。
- **NPU 新算法**：`T.Kernel(grid)`（1-D，平台硬限制），
  `grid = ceil(batch/block_m) * 2 * ceil(d_padded/block_n)`，块内
  `m0 = (bx // (2*n_tiles)) * block_m`、`which = (bx // n_tiles) % 2`、`n0 = (bx % n_tiles) * block_n`。
  性能意图保持：每个 block 仍独占一个 `(batch tile, which, d tile)` 输出 tile，权重 tile 只流经
  本 block 的 L1 一次；`which` 仍是最内层的「权重选择」轴，使相邻 block 读相邻的 W 列片（L2 友好）。
- **语义保持论证**：块到 tile 的映射是双射，每块写互不重叠的 `kv[which, m0:m0+am, n0:n0+an]`；
  平移只在 `which==0 and m0==0` 的 `n_tiles` 个 block 上发生，各块列片互不重叠。块的执行顺序
  不影响任何输出元素的值 → **精确等价**（累加顺序也未变：每块内部仍按 K 段串行累加）。

**重设计项 R2：K 尾块 —— 谓词填零 → 有效区间（承接 O6 / §0.5 #8）**

- **源方案**：`d_mem % block_k != 0` 时逐元素 `if_then_else(k_idx < d_mem, x, 0)` 把尾块补零，
  再按满 `block_k` 做 mma。性能意图：不越界读、零元不贡献。
- **NPU 新算法**：把有效长度显式表达为区域——
  `ak = T.min(d_mem - k0, block_k)`；`T.copy(E[..., k0:k0+ak], e_l1[0:am, 0:ak])`、
  `T.copy(W[k0:k0+ak, n0:n0+an], w_l1[0:an, 0:ak], transpose=True)`、
  `T.copy(e_l1[0:am,0:ak], e_l0[0:am,0:ak])`、`T.gemm(e_l0[0:am,0:ak], w_l0[0:an,0:ak],
  res[0:am,0:an], transpose_B=True, clear_accum=(kt==0))`。`asc_mmad` 的 K 参数因此直接等于
  `ak`（探针 H2 生成 `asc_mmad(..., (519 - max(kt*256, 263)), ...)` = 7），越界列**根本不进入
  乘积**。这正是 950DT 只读参考 `_build_project` 的做法（它的 fixture 也含 `d_mem=519`）。
- **语义保持论证**：数学上 `Σ_{k<d_mem} e_k w_k` 与「补零到 block_k 边界后求和」完全相同（零元
  不贡献）；`ak` 精确截断使乘积项的集合与源一致。fp32 累加分组不变（仍按 K 段分组）→ **精确等价**。
  边界：`d_mem % block_k == 0` 时 `ak ≡ block_k`，与精确分支完全一致。
  **残余风险**：`asc_mmad` 收到非 16 整数倍的 K（如 7）时是否由硬件自行掩蔽尾 lane，本设计
  **未能在无卡环境证实**（§9.2 E1 给出判据与回退：host 把 `d_mem` 补到 16 的整数倍）。

**重设计项 R3：死行消除（放弃行夹取，承接 O5/O8 的效率意图）**

- **源方案**：装载时把越界行夹取到 `batch-1`，靠写回守卫丢弃。
- **NPU 新算法**：`am = min(batch - m0, block_m)` 只装载/写回有效行；`asc_mmad` 的 M 参数 = `am`
  （探针 H2：batch=8 时 M=8、batch=17 的第二块 M=1），死行既不参与计算也不落 GM。
- **语义保持论证**：死行从不参与任何有效输出（源实现的夹取只是为了让越界地址合法且随后被丢弃）；
  本方案在 M 方向没有越界地址，被丢弃的死行值与输出无关 → **精确等价**。机器验证 `A4`
  （§1.6.1）把死行填 `NaN/±inf/-0` 后对照，逐位一致。

**重设计项 R4：cache 平移的落点（承接 O7/O10 / §0.5 #9、#11）**

- **源方案**：与本 launch 同一 kernel，`which==0 and row_block==0` 的 block 用
  `T.Parallel(batch*(max_conv_len-1), block_n)` 逐元素读 `conv_state[b, s+1, col0+j]` 写
  `new_conv_state[b, s, col0+j]`。
- **NPU 新算法**：同一 `T.Kernel` 内（不带 `T.MixedKernel`，见下），按 batch 行发 **2-D 区域拷贝**，
  经 UB 中转（DMA 搬运，不做逐元素计算）：
  `T.copy(conv_state[b, 1:max_conv_len, n0:n0+an], shift_ub[0:nshift, 0:an])` →
  `T.copy(shift_ub[0:nshift, 0:an], new_conv_state[b, 0:nshift, n0:n0+an])`，`b` 走
  `T.serial(batch)`。含 UB 代码的 `T.Kernel` 被编译器**自动提升为 `__mix__(1, 2)`**
  （1 AIC + 2 AIV）：AIC 段跑 L1/L0 GEMM，AIV 段跑平移（探针 H1/H2 生成的源码为证）。
  - **为什么不用 `T.MixedKernel`**：显式混合 frame 在本工具链编译失败
    `Need input to be in SSA form dup sid`（探针 X1/X2，`sid` 是否被用户引用都一样），
    故显式混合不是可用路径；自动提升路径 7/7 形状（含 M/K/N 尾块、bf16、`max_conv_len==1`）编译通过。
  - **为什么不经 L1**：`T.alloc_l1` + GM→L1→GM 同型拷贝编译失败
    `ScalarDcacheBypass: vector-valued buffer store not implemented`（探针 H5）→ UB 是唯一可用中转。
  - **为什么不用单个 3-D 区域拷贝**：`PlanMTECopy: >2D strided copy not yet supported`（探针 H6），
    因此保留「逐 batch 行 2-D 拷贝」的 `T.serial(batch)` 描述符循环（§1.6.2 第 3 项）。
- **语义保持论证**：平移是**纯 dtype 拷贝**（无算术、无舍入），因此与源逐元素写**逐位等价**；
  `nshift = max_conv_len - 1` 行、`b` 覆盖全部 batch、列片 `[n0, n0+an)` 覆盖全部 `d_padded`。
  边界：`max_conv_len == 1` 时整段编译期消失（`nshift == 0`，不分配 UB）。
  **重复写说明**：自动提升后两个 AIV 都会执行同一段平移代码（生成源码里 AIV 段没有 `sid` 守卫），
  即同一 GM 区域被两个 AIV 写入**相同的字节**；本 launch 内没有任何代码读 `new_conv_state`，
  且它 `torch.empty` 新分配、与 `conv_state` 不别名，故任何写序都收敛到同一内容
  → 语义无害（代价约为平移流量的 2 倍，见 §1.6.1 A4 的收益量化）。
  已评估的替代（用 `with T.Vector(vector=2) as sid:` + `if sid == 0/1` 把行表拆成两半）虽能编译，
  但生成的 AIV 段**丢掉了 UB 暂存的 MTE2→MTE3 同步**（对比见 §11.1 探针 H7），
  属正确性风险，故不采纳。

**重设计项 R5：输出写回 —— 逐元素 fragment 存 → L0C→GM 区域写（承接 O8）**

- **源方案**：`T.Parallel(block_m, block_n)` 逐元素把 fragment 写到 `kv[which, row0+i, col0+j]`，
  用 `if row0+i < batch` 丢弃死行。
- **NPU 新算法**：`T.copy(res[0:am, 0:an], kv[which, m0:m0+am, n0:n0+an])`，一条
  `asc_copy_l0c2gm` 完成整块 fp32 写（探针 H1/H2 生成
  `asc_copy_l0c2gm((&kv[((block_idx&3)*64 + 2048)]), res, 64, 8, 256, ...)`）。
- **语义保持论证**：写入的元素集合与源相同（有效 `am × an` 行块），值都是 L0C 的 fp32 累加结果，
  无额外舍入、无越界写 → **精确等价**；边界由区域天然表达（`am`/`an` 截断）。
  平台依据：`L0C → GM` 已在 `.agents/platforms/toolchains/ascend/knowledge/PL-950-cube-tile.md`
  的验证路径中（`GM → L1 → L0A/L0B → T.gemm → L0C → GM`）。

### 0.7 标杆实现

- **源算子路径**（golden 的语义出处）：`/workspace/TileOPs/src/tileops/_kernels/sequence_modeling/engram/decode.py`
  第 25–111 行（`_engram_project_kernel`）。
- **源仓参考实现（golden 移植对象）**：`examples/TileOPs/tileops/workloads/sequence_modeling.py`
  的 `engram_decode_step_torch`（第 420–487 行）与 `_rmsnorm_decode`；本 kernel 只移植其中的
  `k = e_t.float() @ W_K.float()`、`v = e_t.float() @ W_V.float()` 与新 history 构造三行。
  该参考的 `new_conv_state` 在「满 cache」分支是 `cat([conv_state[:, 1:, :], vhat_n])`
  ——与本 kernel 写的第 `0 .. max_conv_len-2` 行逐位对应（§8.1）。
- **只读实现参考（不作为 golden、不照抄）**：
  `/workspace/TileOPs-ascend950/src/tileops/kernels/ascend/sequence_modeling/decode.py` 的
  `_build_project`（Cube NT GEMM + 显式 L1/L0 分层 + `annotate_buffer_versions`）与
  `_pick_block_k`（按 L0B 64 KiB 反解 `block_k`）。借用点已在 §0.4 O11、§0.5 #13 标注；
  与它的边界差异见 §0.5 #11、§0.6 R4。
---

## 1. 概述

### 1.1 算子名称

`_engram_project_kernel`（Engram 单步 decode 的 Cube 投影段 + cache 平移；`EngramDecodeFwdOp`
的第 1 个 launch）。

### 1.2 功能描述

一步 decode 的两组投影 `K = e_t·W_K`、`V = e_t·W_V`（fp32 结果写 `kv`），并在同一 launch 内把
卷积缓存左移一格（`new_conv_state[:, 0:max_conv_len-1, :] = conv_state[:, 1:, :]`，逐位等价拷贝）。

### 1.3 数学公式

$$
\text{kv}_0[b,:] = \sum_{k=0}^{d_{mem}-1} e_t[b,k]\,W_K[k,:] ,\qquad
\text{kv}_1[b,:] = \sum_{k=0}^{d_{mem}-1} e_t[b,k]\,W_V[k,:]
$$

$$
\text{new\_conv\_state}[b,s,:] = \text{conv\_state}[b,s+1,:],\qquad s=0,\dots,L_{max}-2
$$

其中 `b < batch`、输出列 `0 ≤ j < d_padded`，`kv` 为 fp32、`new_conv_state` 为输入 dtype；
第 `L_max-1` 行由 `_engram_step_kernel` 写（§1.5 契约）。

### 1.4 算法描述

**（迁移决策后的 NPU 侧算法，与 §1.6.0 选定的 C1 一致）** 三条并行轴一次性铺开成 1-D grid，
每个 block 负责一个 `(batch tile × which × d tile)` 输出 tile：

1. **任务分解**（R1）：`grid = mt · 2 · nt`，`mt = ceil(batch/block_m)`、`nt = ceil(d_padded/block_n)`；
   `m0 = (bx // (2·nt)) · block_m`、`which = (bx // nt) % 2`、`n0 = (bx % nt) · block_n`。
2. **K 段流式累加**（R2 + §0.5 #4）：`for kt in T.Pipelined(ceil(d_mem/block_k), num_stages=2)`：
   把 `e_t[m0:m0+am, k0:k0+ak]` 搬进 `e_l1`、把 `W[which][k0:k0+ak, n0:n0+an]` 带 `transpose=True`
   搬进 `w_l1`（GM→L1 的 `dn2nz` 通道顺便完成 `[K,N] → [N,K]`），再分两跳进 `e_l0`/`w_l0`，
   `T.gemm(..., transpose_B=True, clear_accum=(kt == 0))` 累加到 L0C 的 fp32 `res`。
3. **写回**（R5）：`T.copy(res[0:am, 0:an], kv[which, m0:m0+am, n0:n0+an])`。
4. **cache 平移**（R4）：只在 `which == 0 and m0 == 0` 的 `nt` 个 block 上，按 batch 行做
   GM→UB→GM 的 2-D 区域拷贝（共 `batch · (max_conv_len-1)` 行 × `an` 列），`max_conv_len == 1` 时整段消失。

**与源算法的差异（逐条）**：三维 grid → 1-D（R1）；K 尾块由逐元素谓词 → 有效区域（R2）；
死行夹取 → 区域截断（R3）；输出逐元素写 → L0C→GM 区域拷贝（R5）；平移由逐元素并行写 →
UB 暂存的 DMA 拷贝（R4）；`threads` 参数消失（§0.5 #12）。**语义差异：无**（全部为 §0.6 论证的
精确等价变换）。**与 950DT 只读参考的差异**：参考把平移放在 step kernel（它没有「project 必须
写 history」的边界约束），本设计按 harness 函数边界留在本 launch（§0.5 #11）。

### 1.5 数据流图

```
GM[e_t] --(MTE2 nd2nz, am×ak)--> L1[e_l1] --(MTE1)--> L0A[e_l0] --\
GM[W_K|W_V] --(MTE2 dn2nz+transpose, an×ak)--> L1[w_l1] --(MTE1)--> L0B[w_l0] --(Cube)→ L0C[res] fp32
                                                                                      |
                                                                    (MTE3, am×an) ----+--> GM[kv[which]]
GM[conv_state] --(MTE2, nshift×an per row)--> UB[shift_ub] --(MTE3)--> GM[new_conv_state][:, 0:nshift, :]
```

跨 launch 契约（两个 launch 之间只有 GM）：

| 中间量 | 生产方 | 消费方 | shape / dtype | 边界 |
|--------|--------|--------|---------------|------|
| `kv` | 本 kernel（唯一写者） | `_engram_step_kernel`（只读） | `(2, batch, d_padded)` fp32 | 两片都必须在 step 启动前写完；step 读 `kv[0]`=K、`kv[1]`=V 并自行舍入到输入 dtype |
| `new_conv_state[:, 0:mcl-1, :]` | 本 kernel | step（只读；以及对外输出） | `(batch, mcl-1, d_padded)` 输入 dtype | 逐位等于 `conv_state[:, 1:, :]` |
| `new_conv_state[:, mcl-1, :]` | **step kernel** | 对外输出 | `(batch, d_padded)` | 本 kernel **不写**（`torch.empty` 的该行在 step 前是脏值，step 必须整行覆盖） |
| `conv_state` | host | 本 kernel + step | `(batch, mcl, d_padded)` | 只读，本 kernel 不就地修改 |

### 1.6 算法调研与优化分析 ⭐

#### 1.6.0 算法调研（Algorithm Research）⭐

调研对象 = 「同一数学语义（两组 `e_t·W` fp32 投影 + 缓存左移）」的算法族。**基线候选 = 源算子
（GPU TileLang 三维 grid + 逐元素尾块谓词 + 逐元素平移）**，它只是候选之一。

**R1 等价化简公式候选**（正式等价论证与收益量化在 §1.6.1，此处只做初判）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 是否纳入 R3 对比 |
|---|------|------------|-----------|---------|----------------|
| 0 | **基线：源算子** | 三维 grid；`block_m=16`；K 段 `T.Pipelined`；尾块逐元素填零；平移逐元素并行 | — | — | ✅（基线） |
| 1 | 权重拼接单 GEMM | `W_cat = [W_K \| W_V] (d_mem, 2·d_padded)`，一次 `e_t·W_cat` 出 `kv` 两片 | 数学恒等（分块对角） | 少读一遍 `e_t`（省 `batch·d_mem·2 B`）；块数减半 | ✅ |
| 2 | K/V 同块双累加器 | 同一 block 内 `n0` 相同、`which` 两片共用一份 `e_l1`，两个 L0C | 数学恒等 | 省一半 `e_t` 流量（同上），块数减半 | ✅ |
| 3 | 投影融进 step kernel | 不落 `kv`，在 step 的逐行归约里就地算 K/V | 数学恒等（fp32 累加仍在） | 省 `kv` 往返 `2·batch·d_padded·4 B` + 一次 launch | ✅（受契约约束，见 §0.5 #11） |
| 4 | 纯向量 rank-1 累积 | `out[b,:] += e_t[b,k]·W[k,:]` 沿 k 串行，全在 AIV | 容差内等价（累加顺序变为逐 k 串行） | 无（访存不变、算力降 1–2 个数量级） | ✅ |
| 5 | 平移走 3-D 单区域拷贝 | 一次 `T.copy` 覆盖 `(batch, nshift, an)` | 数学恒等 | 少发 `batch` 条 DMA 描述符 | ✅（编译不可用，探针 H6） |
| 6 | 平移走 L1 中转 | GM→L1→GM | 数学恒等 | 少一次 UB 往返 | ✅（编译不可用，探针 H5） |
| 7 | `clear_accum=(kt==0)` 代替显式清零 | 首段直接用乘法结果初始化累加器 | 数学恒等（加性单位元） | 省一次 4 KiB 的 L0C 清零 | ✅（§1.6.1 A1） |
| 8 | 有效区间代替尾块谓词 | `ak = min(d_mem-k0, block_k)` 进区域 | 数学恒等（零元不贡献） | 省 `block_m·block_k` 次逐元素选择/段 | ✅（§1.6.1 A3） |
| 9 | 死行消除 | 只装载/写回 `am` 行 | 数学恒等 | 省 `(block_m-batch)/block_m` 的 e_t 读与 L0C 写 | ✅（§1.6.1 A2） |

**R2 在线算法**：**有，且它就是基线结构**。投影沿收缩轴 `d_mem` 做**单遍流式累加**（running
accumulator 在 L0C，中间缓冲 O(tile) 而非 O(d_mem)），不存在需要「先物化完整中间矩阵再归约」的
两遍变体——GEMM 的两遍版本只会重算，不会更省。缓存平移逐元素无跨元素依赖，天然单遍。
结构判据（`references/algorithm-research.md` R2 第 ③ 条）：任何可分解为 running 统计量（和/平方和/
max/计数）的算子原则上存在在线变体；本算子的统计量只有「按 `k` 的 running sum」，因此在线形式即
K 段累加。收益口径：扫描遍数 = 权重 1 遍、`e_t` 2 遍（每 `which` 一遍）、`conv_state` 1 遍；
中间缓冲峰值 = 一个 `(block_m, block_n)` fp32 L0C（4 KiB）。故 R2 **不产生新候选**，只确认
基线的单遍结构不应被改成两遍。

**R3 复杂度对比**（四口径；代表 workload = serving-b4：batch=4、d_mem=1024、d_padded=512、mcl=20。
口径说明：FLOPs 只计乘加（2 FLOP/MAC），不含索引；Bytes 只计本 launch 的 GM 流量；遍数指同一
数据的 GM 重复读取次数；中间缓冲峰值只计片上驻留）：

| 算法候选 | FLOPs | 访存量 (Bytes) | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|-------|---------------|---------|-------------|---------------------|
| 0 基线（源算子直译，数学基准） | 8.39 M | 2.29 M（`e_t` 2×16 K + W 2.10 M + kv 16 K + 平移 156 K） | W 1、`e_t` 2、state 1 | L1 80 K + L0B 64 K + L0C 4 K | 16 逻辑块；三维 grid 在 NPU 不可用 |
| **C1 选定方案**（= 行 0 的数学 + §0.6 的 R1–R5 落地 + §1.6.1 的 A1–A3） | 8.39 M | **2.29 M**（与行 0 同；A1–A3 只减无效工作量，不减有效流量） | W 1、`e_t` 2、state 1 | L1 80 K + L0B 64 K + L0C 4 K + UB 2.4 K | **16 逻辑块 ≤ 28 Cube 核（单波）；1 launch** |
| 1 权重拼接单 GEMM | 8.39 M | 2.29 M + **host 拼接 4.19 M**（读写各 2.10 M） | W 1、`e_t` 1、state 1 | 同上 + 拼接缓冲 2.10 M（落 GM） | 8 块（更少）；host 侧串行开销约 2.6 µs（按 1.6 TB/s 假设） |
| 2 K/V 同块双累加器 | 8.39 M | 2.29 M − 16 K（省一遍 `e_t`） | W 1、`e_t` 1、state 1 | L1 +8 K、L0C ×2 = 8 K | 8 块（28 核用 29 %）；单块工作量翻倍 |
| 3 投影融进 step | 8.39 M | 2.29 M − 32 K（省 `kv` 往返） | 同上 | 需跨核归约 workspace 或逐行重读 W | step 是「每核一行」，投影会退化成每行读全量 W（2.10 M × batch） |
| 4 纯向量 rank-1 | 8.39 M | 2.29 M（相同） | W 1、`e_t` 1、state 1 | UB 累加器 `(batch, block_n)` fp32 ≈ 2 K | 56 AIV；无跨核归并 |
| 5/6 平移换路（3-D / L1） | — | 平移 156 K（不变） | 1 | 同上 | 只在延迟/描述符数上变化；均编译失败 |
| 7/8/9（§1.6.1 A1–A3） | 8.39 M | 2.29 M − 0（只省清零/选择/死行） | 不变 | 不变 | 主收益是「少算/少搬无用数据」，见 §1.6.1 |

**R4 硬件亲和性评估**（逐候选对照检查清单；负向淘汰附证据或标注未实测假设）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---------|-------------|---------|------------|---------|------------|------|
| 0 基线直译 | Cube ✅（`T.gemm`→`asc_mmad`） | ✅ | ✅ | 三维 grid 在 NPU **不可用** | 需重写 | ❌ 结构不可直译（R1/R2/R4/R5 重设计后取其实质） |
| 1 权重拼接 | Cube ✅ | ❌ 需 2.10 M 的 GM 拼接缓冲 + host 拷贝 | ✅ | ✅ | 破坏「1 算子 1 次搬运」 | ❌ 增益（省 16 K）远小于代价（多搬 4.19 M），见 R3 |
| 2 K/V 同块 | Cube ✅ | ✅（L0C 8 K） | ✅ | ✅ | 块内两段 GEMM 串行 | ❌ 块数 8 < 28 核（29 % 占用），总流量几乎不变（只省 16 K），判定裕度无正向 |
| 3 投影融进 step | ❌ 每核一行的结构会让 W 被读 `batch` 遍（`batch×` 流量） | ❌ | ✅ | ✅ | 跨 launch 融合需跨核协作 | ❌ 契约与流量双输（且越界改动 step 设计） |
| 4 纯向量 rank-1 | ❌ 算力：Cube 峰值 ≈ 378 TFLOP/s（16³ MMA × 28 核 × 1.65 GHz，**未实测假设**）对 AIV fp32 ≈ 23.6 TFLOP/s（128 lane × 2 FLOP × 56 核 × 1.65 GHz，**未实测假设**）→ 8.39 MFLOP 分别 0.022 µs / 0.36 µs；b4 的流量项是 1.4–2.3 µs（按 1.0–1.6 TB/s 假设），故 AIV 上计算与搬运**同量级**、失去余量 | ✅ | ✅ | ✅ | ❌ 无本地已验证的 rank-1 融合模式 | ❌ 不选为**主选**（见下方「调研结论」的保留条件） |
| 5 平移 3-D 拷贝 | — | ✅ | — | — | — | ❌ 编译报错 `>2D strided copy not yet supported`（探针 H6） |
| 6 平移 L1 中转 | — | ✅ | — | — | — | ❌ 编译报错 `ScalarDcacheBypass: vector-valued buffer store not implemented`（探针 H5） |
| 7/8/9（A1–A3） | Cube ✅ | ✅ | ✅ | ✅ | ✅ | ✅ 采纳（§1.6.1） |
| **C1（选定）**：1-D grid + L1/L0 Cube NT GEMM + AIV 平移 | Cube 主导计算 ✅ | ✅（L1 80 K/512 K、L0B 64 K/64 K、L0C 4 K/256 K、UB 2.4 K/248 K） | ✅（`d_padded % 16 == 0`；尾块由区域表达） | ✅（`batch/d_mem/d_padded/mcl` 全为编译期常量） | ✅（`__mix__(1,2)` 单 launch，GEMM 与平移无依赖，可重叠） | ✅ **选定** |

**设计期估算（D-2 roofline 下界行）**

估算口径：`估算下界 = max(流量项, 发射项, 容量项)`（`.agents/skills/_shared/standards/hardware-cost-model.md` §2）。
本 launch 只有 1 次 kernel 启动、无中间 GM 往返、无跨核归并，因此**发射项主导**：

| workload | 流量项（Bytes / 假设带宽） | 发射项（1 launch） | 容量项 | 估算下界 |
|---|---|---|---|---|
| decode-s1 | 273 KB → 0.17–0.27 µs | ~5 µs（未实测假设） | L0B 64/64 KiB 满足 | ≈ 5 µs |
| serving-b1 | 540 KB → 0.34–0.54 µs | ~5 µs | 同上 | ≈ 5 µs |
| serving-b4 | 2.29 MB → 1.4–2.3 µs | ~5 µs | 同上 | ≈ 5 µs |
| serving-b8 | 696 KB → 0.44–0.70 µs | ~5 µs | 同上 | ≈ 5 µs |
| decode-partial-cache | 559 KB → 0.35–0.56 µs | ~5 µs | 同上 | ≈ 5 µs |

`估算下界: 5 µs（发射项 1 launch ≈ 5 µs + 流量项 0.17–2.3 µs + 容量项 L0B 64 KiB/64 KiB 满足）`，
其中**带宽（1.0–1.6 TB/s）、单次 launch 开销（≈5 µs）、Cube/AIV 峰值（378 / 23.6 TFLOP/s）全部
是「未实测假设」**——平台 `cost_measurements` 为空、`knowledge/` 下无 `constants.md`，任务要求
「纯 latency（batch 1–8），不要用 MFU」，故本行只用于**结构决策**（见下）而非性能承诺；
Stage 4 首跑后按 `.agents/skills/_shared/standards/hardware-cost-model.md` §3 回填「设计估算 vs 实测」偏差行。
**该行驱动的两个设计结论**：(a) 本 launch 不能再拆（拆出独立平移 launch 会再加 ≈5 µs，故 R4 选
「同 launch 的 AIV 平移」而不是「两次 launch」）；(b) 任何把 Cube 换成 AIV 的候选（C4）在
serving-b4 上会把计算项从 0.02 µs 抬到 0.36 µs（≈16×），与流量项同量级，属于「把余量吃掉」，
不构成正向收益。

**调研结论**：选定 **C1 —— 1 个 launch、1-D grid、L1/L0 显式分层的 Cube NT GEMM tile（`block_m=16`、
`block_n=64`、`block_k=256`）+ 同 launch 内的 AIV UB 暂存平移**，并对公式做 §1.6.1 的 4 项等价优化。
关键依据：R3 中 C1 的访存（2.29 MB）已是所有候选的下界（C1/C4 并列；C2/C1 的差别只有 16 KB 而
C2 把并发块数从 16 降到 8），R4 中只有 C1 同时满足「Cube 主导计算 + 全片上驻留 + 全静态边界 +
单 launch」；C4（纯向量 rank-1）虽在流量上与 C1 并列，但计算项放大 ≈16× 且无本地已验证模式，
故**不选为主选**，仅作为「若 Stage 3/4 实测发现 Cube 侧 `d_mem < block_k` 的小 shape 反被启动开销
拖累」时的备选方向（本阶段不为其产出结构设计：R3/R4 已给出量化负向依据）。
**「无更优替代」部分的范围声明**：已逐条评估 `.agents/skills/tilelang-op-design/references/algorithm-research.md` 的替代空间
（权重拼接 / 多累加器 / 跨 launch 融合 / 计算单元替换 = R1 的 #1–#4），并检索本仓同类实现
（`examples/shared_expert_mlp/_gemm_basic_kernel/DESIGN.md` 的 GEMM 分区与尾块结论、
`.agents/platforms/toolchains/ascend/knowledge/gemm-tiles.md` 的 NT/K 分块/尾块条目、
`examples/engram_gate_conv/_engram_gate_conv_fwd_kernel/` 的同类算子目录）与平台
`knowledge/algorithm-candidates.md`（无 GEMM 之外的投影候选）；因此「无更优的等价化简公式」
限定在上述已评估集合内，不宣称全局最优。**源算法优化意图承接**：O1→L1 分层（§0.5 #2）、
O2/O3→L0A/L0B/L0C（#3/#4）、O4→`T.Pipelined`+版本注解（#5，退化说明见 §9.1）、
O7→同 launch 的 AIV 平移（#11）、O9→1-D grid 分解（#10）；O5 的夹取与 O10 的逐元素映射被
R3/R4 取代（意图分别是「不为 batch 浪费 tile」「平移并行化」，均由区域拷贝承接），无静默丢弃。

#### 1.6.1 数学等价优化（公式级）

分析对象 = §1.6.0 选定的 C1。本段没有除法、没有超越函数、没有可提取的公共子表达式
（`e_t` 只被两组权重各用一次，`conv_state` 只被平移一次），因此可优化空间集中在
**累加器初始化、尾块与死行的无效工作量**三处。

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| A1 | 累加器零初始化消除 | `acc ← 0`（整块 L0C 清零）；`for kt: acc += P_kt`，即 `acc = ((0 + P_0) + P_1) + …` | 首段用乘法结果直接初始化：`acc = P_0`；`for kt>0: acc += P_kt`（`clear_accum=(kt==0)`，其后恒 `False`） | 数学恒等：`0 + x ≡ x`（fp32 加法单位元，对 `±0`、`NaN/±inf`、subnormal 均按 IEEE 逐位相同——`0 + NaN = NaN`、`0 + (-0) = +0` 与「直接取 P_0」在 fp32 位型上一致，见下表 corner 用例） | 每 block 省 1 次 4 KiB 的 L0C 清零（`block_m·block_n·4 B`）；serving-b4 16 块 × 4 KiB = 64 KiB 的片上写流量 |
| A2 | 死行消除 | 装载 `block_m` 行（越界行夹取），写回时用 `if row0+i < batch` 丢弃 | 只装载/只写回 `am = min(batch-m0, block_m)` 行，`asc_mmad` 的 M = `am` | 死行不参与任何有效输出（源的夹取只是为避免越界地址），本方案 M 向无越界地址；被丢弃的死行值与输出无关 → 精确等价（死行填 `NaN/±inf/-0` 的对照见下表 `A4` 用例） | batch=1 时省 `15/16` 的 `e_t` 读（`15·d_mem·2 B`）与 L0C 写（`15·block_n·4 B`）；batch=8 时省 `8/16` |
| A3 | K 尾块谓词消除 | 逐元素 `if_then_else(k_idx < d_mem, x, 0)` 补零后按满 `block_k` 收缩 | 有效区间 `ak = min(d_mem-k0, block_k)` 直接进 `copy`/`gemm` 区域，`asc_mmad` 的 K = `ak` | 数学恒等：`Σ_{k<d_mem} e_k w_k` ≡ `Σ_{k<ceil·block_k} Ẽ_k W̃_k`（补零项恒为 0，fp32 下 `x + 0 = x` 逐位成立，`NaN·0` 在本式不会出现——补零只发生在**输入**侧，且 `d_mem % block_k == 0` 时该分支不生成） | 只在 `d_mem % block_k != 0` 时生效（ktail-519）：省 `3 段 × block_m × block_k = 12 288` 次逐元素选择与对应的填充写 |
| A4 | 平移的 launch 合并 | 独立 launch（或源实现的逐元素并行写） | 同 launch、`which==0 and m0==0` 的 block 用 GM→UB→GM 的 DMA 拷贝 | 平移是纯 dtype 拷贝（无算术、无舍入）：读 `conv_state[b, s+1, :]` 写 `new_conv_state[b, s, :]`，与源逐元素写**逐位等价**；本 launch 内无任何代码读 `new_conv_state`、它不与 `conv_state` 别名（§0.6 R4 的重复写亦然） | 省一次 kernel 启动（≈5 µs，未实测假设，占总下界的 ~100 %）；代价是重复写使平移流量 ×2（156 KB → 312 KB，serving-b4） |

**等价性机器验证（D-1）**：`examples/engram_decode/_engram_project_kernel/verify_equiv.py`
（torch CPU，fp64 参照；随机张量 + 角点值 ±inf/NaN/subnormal/±0；5 条 manifest workload +
ktail-519，6 个 dtype/shape 组合）。判据 = 候选式与基线式在**目标 dtype 舍入路径下的 fp32 位型
逐位相同**（`float ==` 会把 `-0.0` 与 `+0.0` 视作相等，故脚本直接比整数位型）。

| 采纳项（脚本内的用例标签） | 覆盖用例 | 最大 ulp 差 | >1 ulp 违反率 | NaN/±inf 位型 | 对 fp64 的最大相对误差（候选 / 基线） | 结论 |
|---|---|---|---|---|---|---|
| A1 累加器零初始化（`A3`） | 5 workload × {K,V}，bk=256/128 | **0** | 0.0e+00 | 一致 | 9.06e-06 / 9.06e-06 … 8.00e-04 / 8.00e-04（最大者 ktail-519/V） | **EQUIV_PASS** |
| A1 角点值（`A3 corner`） | fp16、bf16 各一组（±inf/NaN/subnormal/±0） | **0** | 0.0e+00 | 一致 | 该组输入含 NaN，相对误差无意义（两侧同为 NaN 位型） | **EQUIV_PASS** |
| A2 死行消除（`A4`） | 5 workload + ktail-519 的两个 m tile，死行填 NaN/±inf/-0 | **0** | 0.0e+00 | 一致 | 4.11e-05 / 4.11e-05 … 5.43e-02 / 5.43e-02（最大者 ktail-519 的 am=16/16，两侧相同） | **EQUIV_PASS** |
| A3 K 尾块（`T1`） | 5 workload（无尾块，零差异）+ ktail-519（bk=128，补到 640） | **0** | 0.0e+00 | 一致 | 1.50e-02 / 1.50e-02（ktail-519） | **EQUIV_PASS** |
| A4 平移逐位（`build_detail`） | fp16、bf16 各一组（含 NaN/±inf 传播） | 整数位型全等（`torch.equal` on int16 view）= True | 0.0e+00 | 一致 | — | **EQUIV_PASS** |

> 注：`对 fp64` 列是「离精确值的距离」的**观察值**，不是判据；候选与基线在所有行上数值相同，
> 该列相等即说明优化没有改变舍入路径。脚本执行结果末尾另打印 `RESULT: ALL EQUIV_PASS`。

**优化结论**：采纳 **4 项**（A1 累加器零初始化消除、A2 死行消除、A3 K 尾块谓词消除、A4 平移的
launch 合并），全部机器验证为逐位等价。优化后的公式（供 §3.1 拆解使用）：

```
am = min(batch - m0, block_m);  an = min(d_padded - n0, block_n)
res = 0-初始化省略：kt = 0 时 res ← e_t[m0:m0+am, 0:ak0] · W[which][0:ak0, n0:n0+an]
for kt = 1 .. ceil(d_mem/block_k)-1:  res += e_t[m0:m0+am, k0:k0+ak] · W[which][k0:k0+ak, n0:n0+an]
kv[which, m0:m0+am, n0:n0+an] = res                      # fp32，无额外舍入
new_conv_state[b, 0:nshift, n0:n0+an] = conv_state[b, 1:max_conv_len, n0:n0+an]   # 逐位拷贝
```
**无进一步优化空间的项（显式结论 + 依据）**：(a) 本段**没有除法/开方/指数**（`1/·`、`rsqrt`、
`exp`、减 max 稳定化一类变形全部不适用）——这是投影段与 step 段的分工边界；(b) **无公共子表达式**
（`e_t` 的两个消费者是两组独立权重，`W_K`/`W_V` 的 tile 不相交；`conv_state` 只被平移读一次）；
(c) **无算子降代换**（收缩轴上没有可换的低代价函数）；(d) **访存量已到候选下界**（`e_t` 2 遍
是 `which` 轴的必然结果，权重 1 遍是最小值，§1.6.0 R3）。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

盘点 §3.3 伪代码中的**全部**循环与标量计算点（共 7 处）：

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由（不可替代时必填） |
|---|--------|-----------|-------------|---------|------------------------------|
| 1 | 块索引分解 `m0 / which / n0` | 3 条标量整除/取模 | 无（`T.Kernel` 的 1-D 索引必须手工分解） | ❌ | **block 级索引 / 任务映射计算**：每 block 一次，不是逐元素热点；平台 `T.Kernel` 只给 1 个 `bx`，多维 grid 被拒绝（`references/programming.md`：`T.Kernel(N)` 一维网格） |
| 2 | 有效区间 `am / an / ak` | 每个 block（`ak` 每 K 段）一次 `min` | 无（区域边界必须是表达式） | ❌ | **block 级索引**：`am/an` 每 block 一次、`ak` 每 K 段一次（serving-b4 共 16 块 × 4 段 = 64 次），同时是 §0.6 R2/R3 的边界表达手段 |
| 3 | K 段累加循环（`T.Pipelined(k_tiles)`） | 串行循环，携带 L0C 累加器 | 归约/并行归约 | ❌ | **tile 级顺序依赖**：`res += P_kt` 是循环携带依赖（累加链），改并行需 split-K + 跨核归并（引入 workspace 与额外 launch，§1.6.0 R3 已量化负向） |
| 4 | e_t tile 装载（`am × ak`） | `T.copy` 区域 | — | ✅（已是） | — （`T.copy` → `asc_copy_gm2l1_nd2nz`，DMA 完成，无逐元素标量） |
| 5 | W tile 装载 + 转置（`an × ak`） | `T.copy(..., transpose=True)` | — | ✅（已是） | — （生成 `asc_copy_gm2l1_dn2nz`，转置在搬运通道内完成） |
| 6 | 矩阵乘（`block_m × block_k × block_n`） | `T.gemm` | — | ✅（已是） | — （Cube `asc_mmad`，非逐元素循环） |
| 7 | 平移的行循环（`for b in T.serial(batch)`） | `batch` 条 2-D DMA | ① 单个 3-D 区域拷贝；② 逐元素 `T.Parallel(batch·nshift, an)` | ❌（①②均否） | ① **平台 API 缺失**：`PlanMTECopy: >2D strided copy not yet supported`（探针 H6，`src has 3 modes`）——本工具链的区域拷贝只支持 ≤2 维跨步；② 逐元素 `T.Parallel` 会把 1 条 DMA/行替换成 `batch·nshift·an` 次标量读写（serving-b4 为 4·19·64 = 4864 次），且 UB→UB 的错位拷贝在本平台有已验证的反例（`examples/moe_permute_align/_make_align_kernel/repro/TRAP-950-shared-shift-copy.py`：UB→UB 非零偏移区域拷贝会送错 lane），属净负 |

**向量化结论**：逐元素计算 **0 处标量循环**（装载/写回/平移全部是 DMA 区域操作，矩阵乘在 Cube
单元），保留 **4 处**非向量化点，理由类别分别是：block 级索引（#1/#2）、tile 级顺序依赖（#3）、
平台 API 缺失 + 逐元素退化（#7）；全部为「每 block 一次/每 K 段一次/每行一条 DMA」的量级，
不是逐元素热点。§6 的循环表与本节一一对应（无遗漏、无额外逐元素循环）。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

本算子的两个计算面各有一次「轴」决策：**(i) 平移的向量化轴**（真正逐元素的面）、
**(ii) GEMM 的核内布局与 N tile 宽度**（决定 grid 与 L1/L0B 预算）。

**(i) 平移的向量化轴候选**

| # | 布局方案 | 向量化轴 | repack 路径 | 预估收益/代价 | 是否采纳 |
|---|---------|---------|------------|--------------|---------|
| 1 | I/O 原生布局（`(batch, mcl, d_padded)`，`d` 最内连续） | `d`（每行 `an = 64` 个 fp16 = 128 B 连续段） | 无 | 基线；每行一条 MTE2 + 一条 MTE3，`nshift·batch` 行 | ✅ **采纳** |
| 2 | 核内 `d`-major（`(d_padded, batch, mcl)`） | `mcl` 或 `batch` | 需要一次核内/核外转置 | 转置链要额外 `2·batch·mcl·d_padded·2 B` 的读写（serving-b4 = 312 KB，超过平移本体 156 KB），且平台无已验证的 3-D 转置区域拷贝 | ❌ 代价 > 本体 |
| 3 | host 侧 permute 进出 | 任意 | 两次 host permute | 两次 host 拷贝 ≈ 百 µs 级（按 1.0–1.6 TB/s 假设仍 ≫ 5 µs 下界），且改变跨 launch 契约的 layout | ❌ 净亏 |
| 4 | 逐元素 `T.Parallel`（源方案） | 手工映射到 `(b, s, j)` | 无 | 把 DMA 换成标量读写；§1.6.2 #7 已给出 API/机理依据 | ❌ 退化 |

轴质量评分（采纳项 #1）：整除性 —— `an ∈ {16,32,48,64}`，`64` 个 fp16 = 128 B = 4 个 32 B
UB block，无尾 lane 浪费（`d_padded` 由 host 保证 16 对齐，实测 5 条 workload 的 `an` 恒为 64）；
累加链形态 —— 平移无累加；repack 代价 —— 0；UB 容量影响 —— `shift_ub = nshift × block_n × 2 B
= 19×64×2 = 2432 B`（serving-b4 最大），占 UB 248 KiB 的 1 %。

**(ii) N tile 宽度与 GEMM 布局候选**

| # | 方案 | 整除/尾块 | L0B 预算（×2 版本） | grid（b4 / b1） | 结论 |
|---|------|----------|--------------------|----------------|------|
| 1 | `block_n = 64`，`block_k = 256`，NT（GM→L1 `dn2nz` + `transpose_B=True`） | `d_padded/64 ∈ {4,8}`，无 N 尾块；`d_mem=519` 有 K 尾（区域表达） | `64·256·2·2 = 64 KiB` = L0B 上限（满足） | 16 / 8 块（≤28 单波） | ✅ **采纳**（harness 默认 `min(64, d_padded)`，与 950DT 参考的 `_pick_block_k` 反解一致） |
| 2 | `block_n = 32`，`block_k = 256` | 同上，`d_padded/32 ∈ {8,16}` | `32·256·2·2 = 32 KiB` | 16 / 16 块 | ⚠️ **备选**（见下方裁决计划） |
| 3 | `block_n = 128`，`block_k = 128` | `d_padded/128 ∈ {2,4}` | `128·128·2·2 = 64 KiB` | 8 / 4 块 | ❌ 块数过少（4 块 = 14 % 占用），K 段数翻倍 |
| 4 | NN 布局（`B[K,N]` 原样进 L1，L1→L0B 转置） | — | 同上 | 同上 | ❌ `gemm.h` 明确「fp32 NN 必须走 NT」，b16 走 `asc_copy_l12l0b_transpose` 分支，本工具链的已验证条目只有 NT（`knowledge/gemm-tiles.md` 第 1 条） |

**布局决策结论**：选定 **(i)#1 + (ii)#1** —— 平移沿 `d`（最内连续轴）向量化、offset 0 的
GM↔UB 区域拷贝；GEMM 走 `block_m=16 / block_n=64 / block_k=256` 的 **NT** 布局（权重在 GM→L1 的
`dn2nz` 通道内转置，L0B 保持 `[N,K]`）。该结论已同步到 §3.3 伪代码（`e_l1/w_l1` 形状与
`transpose=True`）、§4.3（`shift_ub = (nshift, block_n)`）与 §6（K 段循环的内层是整块区域拷贝，
不存在逐元素内层向量维）。**弃选理由**均为量化或证据（转置链代价 > 本体、API 不支持、占用率下降、
无已验证 NN 条目），无「未考虑」项。

**实验裁决计划（主选 #1 vs 备选 (ii)#2）**：判定裕度落在两个**未实证常数**上——(a) 单核 MTE2
有效带宽与 fabric 总带宽的比值（决定「16 个小块」是否优于「8 个大块」），(b) 单次 launch 内
固定开销随块数的斜率。两者 `cost_measurements` 未覆盖，故按实验裁决模式执行：
① **主选** = `block_n=64`（上表 #1，进 Stage 3）；② **备选** = `block_n=32`（结构同主选，仅 tile
常量不同：`e_l1(16,256)` 8 KiB、`w_l1(32,256)` 16 KiB、`w_l0b(32,256)` 16 KiB×2、`res(16,32)` 2 KiB、
`shift_ub(nshift,32)` ≤ 1.2 KiB、`grid = mt·2·ceil(d_padded/32)`，b1 为 16 块 / b4 为 32 块——注意
b4 的 32 块 > 28 核，需要 `block_n=32` 时把 `block_k` 降到 128 以维持单波，属 Stage 4 的 A/B 变体，
不在本设计冻结范围）；③ **裁决指标与阈值**：在 serving-b1 与 serving-b8（`d_padded=256`）上量
`project` launch 的 `Task Duration`，若备选比主选快 **> 5 %** 则在对应 shape 上切到备选；工厂按
`(batch, d_mem, d_padded, max_conv_len, dtype)` 做 `lru_cache` 特化，**可以按 shape 分派**，
因此允许「b1/b8 用备选、b4 用主选」的分片结论；实测数据回写本节并触发 Stage 4 的 A/B 记录。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: **混合（Cube + Vector，单 launch 内 AIC/AIV 分区）**——平台 `programming_modes`
的 `hybrid` 键；实现上写作普通 `T.Kernel`，由编译器自动提升为 `__mix__(1, 2)`（探针 H1/H2 的
生成源码 `extern "C" __global__ __mix__(1, 2) void main_kernel(...)` 为证）。

### 2.2 选型理由

- **主导计算是矩阵乘**：`e_t(batch×d_mem) · W(d_mem×d_padded)`，两个投影合计
  0.26–8.39 MFLOP。平台矩阵单元（Cube/AIC，`T.gemm`→`asc_mmad`）是唯一能把这段放进「流量项
  之下」（0.02 µs 对 1.4–2.3 µs）的单元；纯向量路径会把计算项抬到与流量同量级
  （§1.6.0 R4 的 C4 行：0.36 µs，≈16×）。
- **本 launch 的逐元素部分只有 cache 平移**：纯数据搬移，落 AIV（Vector）的 DMA 通道最省。
  实测两单元在同一 `T.Kernel` 内无依赖（投影读 `e_t/W/conv_state`，平移读 `conv_state` 写
  `new_conv_state`，写集不相交），可以真并行。
- **相对 `simd` 默认值的偏离说明**：`op_requirements.programming_mode = simd` 是平台的
  `programming_modes.migration_default`（迁移任务的默认占位），并非用户显式约束。本算子的
  主计算是矩阵乘，纯 SIMD 路径在 §1.6.0 R3/R4 被量化否决（计算项放大 ≈16×、无本地已验证
  模式），故按证据选 `混合`；**平移部分仍然是 SIMD/向量语义**（DMA + 后续 step 段的
  `T.SimtVF` 逐元素），与默认值的意图一致。该偏离已回报 conductor（见交付摘要 `programming_mode`）。
- **相对 `cube` 单模式**：单用 Cube 无法完成平移——AIC 没有可用的 UB 中转路径
  （L1 中转编译失败，探针 H5），而平移必须经 UB；因此必须让 AIV 参与，即混合。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | Cube 段：`T.alloc_l1`（`e_l1`/`w_l1`）、`T.alloc_l0a`/`alloc_l0b`、`T.alloc_l0c`；Vector 段：`T.alloc_shared`（`shift_ub`）。两层不共享缓冲 |
| 计算方式 | Cube 段：`T.gemm`（`asc_mmad`，fp32 累加）；Vector 段：`T.copy` 区域 DMA（无逐元素算术） |
| 线程 / 并行 | 无 `threads` 参数（`T.Kernel` 签名没有）；并行度由 `grid` 与编译器的 AIC/AIV 切分给出（每 block 1 AIC + 2 AIV） |
| 同步 | 自动调度（`ascend/pipeline.py` 的 `allow_autoschedule` 默认开启）：AIC 段生成 L1/L0 的 `asc_sync_notify/wait` 配对，AIV 段生成 UB 暂存的 `PIPE_MTE2 → PIPE_MTE3` 配对（§7） |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式为 §1.6.1 的**优化后公式**（A1–A4 已并入）。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `m0, which, n0 = f(bx)`；`mt, nt, kt_max = ceil(batch/16), ceil(d_padded/64), ceil(d_mem/256)` | 1-D grid 分解（R1） |
| 2 | `am = min(batch-m0, 16)`；`an = min(d_padded-n0, 64)`；`ak = min(d_mem-256·i, 256)` | 有效区间（R2/R3） |
| 3 | `P_0 = e_t[m0:m0+am, 0:ak_0] · W[which][0:ak_0, n0:n0+an]` | 首段即初始化累加器（A1） |
| 4 | `P_i = e_t[m0:m0+am, k0:k0+ak] · W[which][k0:k0+ak, n0:n0+an]`，`res += P_i` | K 段流式累加（R2 在线形式） |
| 5 | `kv[which, m0:m0+am, n0:n0+an] = res` | fp32 直写，无额外舍入（R5） |
| 6 | `new_conv_state[b, 0:nshift, n0:n0+an] = conv_state[b, 1:mcl, n0:n0+an]`（`b < batch`） | 逐位拷贝（A4/R4）；`nshift = mcl-1` |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 1 | 任务分解 | `T.Kernel(grid)` | `grid = mt*2*nt`，`as bx`（1-D，无 `threads`） | 混合 |
| 2 | 有效区间 | `T.min` / `T.ceildiv` | 例：`ak = T.min(d_mem - k0, block_k)` | 混合 |
| 3 | e_t tile 装载 | `T.copy(src_region, e_l1[0:am, 0:ak])` | GM→L1 `nd2nz`；`am`/`ak` 为区域上界 | 混合 |
| 4 | W tile 装载 + 转置 | `T.copy(W[k0:k0+ak, n0:n0+an], w_l1[0:an, 0:ak], transpose=True)` | GM→L1 `dn2nz`，只对 GM→L1 有效（`copy` 文档） | 混合 |
| 5 | L1→L0A / L0B | `T.copy(e_l1[0:am,0:ak], e_l0[0:am,0:ak])`、`T.copy(w_l1[0:an,0:ak], w_l0[0:an,0:ak])` | L0 区域必须以 0 起始且紧凑（`gemm` 文档） | 混合 |
| 6 | 矩阵乘累加 | `T.gemm(e_l0[0:am,0:ak], w_l0[0:an,0:ak], res[0:am,0:an], transpose_B=True, clear_accum=(kt == 0))` | `clear_accum` 默认 `False`；A1 只让首段为 `True` | 混合 |
| 7 | 写回 | `T.copy(res[0:am,0:an], kv[which, m0:m0+am, n0:n0+an])` | L0C→GM，fp32 | 混合 |
| 8 | 平移装载 | `T.copy(conv_state[b, 1:mcl, n0:n0+an], shift_ub[0:nshift, 0:an])` | GM→UB（2-D 行片） | Vector |
| 9 | 平移写回 | `T.copy(shift_ub[0:nshift, 0:an], new_conv_state[b, 0:nshift, n0:n0+an])` | UB→GM，目的偏移 0（避开 UB→UB 错位陷阱） | Vector |
| 10 | 缓冲分配 | `T.alloc_l1` / `T.alloc_l0a` / `T.alloc_l0b` / `T.alloc_l0c` / `T.alloc_shared` | 见 §4.3 | 混合 |
| 11 | 多版本注解 | `T.annotate_buffer_versions({e_l1: ns, w_l1: ns, e_l0: ns, w_l0: ns})` | `ns = num_stages = 2`（正整数） | 混合 |
| 12 | K 段循环 | `T.Pipelined(k_tiles, num_stages=2)` | 自动调度；实测未启用多缓冲（§9.1） | 混合 |
| 13 | 平移行循环 | `T.serial(batch)` | 编译期常量边界 | Vector |

### 3.3 计算伪代码

```python
import functools
import tilelang
import tilelang.ascend.language as T


@functools.lru_cache(maxsize=None)
def _build_project(batch, d_mem, d_pad, max_conv_len, dtype_str,
                   block_m=16, block_n=64, block_k=256, num_stages=2):
    mt = -(-batch // block_m)          # batch tile 数
    nt = -(-d_pad // block_n)          # d tile 数
    kt_max = -(-d_mem // block_k)      # K 段数
    nshift = max_conv_len - 1          # 平移行数；0 表示无平移
    grid = mt * 2 * nt

    @T.macro
    def _projection(E, W, kv, e_l1, w_l1, e_l0, w_l0, res, out_idx, m0, n0, am, an):
        for kt in T.Pipelined(kt_max, num_stages=num_stages):
            k0 = kt * block_k
            ak = T.min(d_mem - k0, block_k)                 # R2：有效 K 区间
            T.copy(E[m0 : m0 + am, k0 : k0 + ak], e_l1[0:am, 0:ak])
            T.copy(W[k0 : k0 + ak, n0 : n0 + an], w_l1[0:an, 0:ak], transpose=True)
            T.copy(e_l1[0:am, 0:ak], e_l0[0:am, 0:ak])
            T.copy(w_l1[0:an, 0:ak], w_l0[0:an, 0:ak])
            T.gemm(e_l0[0:am, 0:ak], w_l0[0:an, 0:ak], res[0:am, 0:an],
                   transpose_B=True, clear_accum=(kt == 0))  # A1：首段即初始化
        T.copy(res[0:am, 0:an], kv[out_idx, m0 : m0 + am, n0 : n0 + an])   # R5

    @T.macro
    def _shift_cache(conv_state, new_conv_state, shift_ub, n0, an):
        for b in T.serial(batch):                            # 每行一条 2-D DMA
            T.copy(conv_state[b, 1:max_conv_len, n0 : n0 + an], shift_ub[0:nshift, 0:an])
            T.copy(shift_ub[0:nshift, 0:an], new_conv_state[b, 0:nshift, n0 : n0 + an])

    @T.prim_func
    def engram_project(
        e_t: T.Tensor((batch, d_mem), dtype_str),
        W_K: T.Tensor((d_mem, d_pad), dtype_str),
        W_V: T.Tensor((d_mem, d_pad), dtype_str),
        conv_state: T.Tensor((batch, max_conv_len, d_pad), dtype_str),
        kv: T.Tensor((2, batch, d_pad), "float32"),
        new_conv_state: T.Tensor((batch, max_conv_len, d_pad), dtype_str),
    ):
        with T.Kernel(grid) as bx:                            # 1-D grid（平台限制）
            e_l1 = T.alloc_l1((block_m, block_k), dtype_str)
            w_l1 = T.alloc_l1((block_n, block_k), dtype_str)
            e_l0 = T.alloc_l0a((block_m, block_k), dtype_str)
            w_l0 = T.alloc_l0b((block_n, block_k), dtype_str)
            res = T.alloc_l0c((block_m, block_n), "float32")
            T.annotate_buffer_versions(
                {e_l1: num_stages, w_l1: num_stages, e_l0: num_stages, w_l0: num_stages}
            )
            m0 = (bx // (2 * nt)) * block_m                   # R1：块索引分解
            which = (bx // nt) % 2
            n0 = (bx % nt) * block_n
            am = T.min(batch - m0, block_m)                   # R3：有效 M
            an = T.min(d_pad - n0, block_n)                   # 有效 N
            if which == 0:
                _projection(e_t, W_K, kv, e_l1, w_l1, e_l0, w_l0, res, 0, m0, n0, am, an)
                if nshift > 0 and m0 == 0:                    # A4：只在 K 片的首个 batch tile
                    shift_ub = T.alloc_shared((nshift, block_n), dtype_str)
                    _shift_cache(conv_state, new_conv_state, shift_ub, n0, an)
            else:
                _projection(e_t, W_V, kv, e_l1, w_l1, e_l0, w_l0, res, 1, m0, n0, am, an)

    return tilelang.compile(engram_project, target="ascend", out_idx=[])
```

要点：`shift_ub` 放在 `if nshift > 0 and m0 == 0` 分支内分配（`max_conv_len == 1` 时不分配、
不生成 AIV 代码）；`kv` / `new_conv_state` 由调用方预分配（`out_idx=[]`），与 harness wrapper 的
调用顺序 `project(e_t, W_K, W_V, conv_state, kv, new_conv_state)` 一致。

### 3.4 API 可行性确认

| API | 来源 / 证据 | 是否验证 |
|-----|------------|---------|
| `T.Kernel(N)`（1-D，无 `threads`） | `.agents/platforms/toolchains/ascend/references/programming.md`「API 与执行区域」第 1 行；`ascend/language/kernel.py:Kernel` | ✅ 文档 + 探针全量编译 |
| `T.alloc_l1 / alloc_l0a / alloc_l0b / alloc_l0c / alloc_shared` | 同上「分配、布局和搬运」表；`allocate.py` | ✅ 探针 H1/H2 |
| `T.copy(..., transpose=True)`（仅 GM→L1） | 同上，`copy_op.py:copy` 文档：`transpose` 的作用域为 GM→L1 的 DN→NZ | ✅ 探针 H1/H2（生成 `asc_copy_gm2l1_dn2nz`） |
| `T.gemm(..., transpose_B=True, clear_accum=...)` | `gemm_op.py:gemm`（`clear_accum` 默认 `False`）；`knowledge/gemm-tiles.md` | ✅ 探针 H1/H2（生成 `asc_mmad`） |
| `T.Pipelined(n, num_stages=2)` + `T.annotate_buffer_versions` | `references/programming.md`「同步与缓冲版本」；`annotations.py` | ✅ 编译通过（但多缓冲未被启用，§9.1） |
| `T.serial` / `T.min` | `references/programming.md`｜探针 | ✅ |
| GM→UB / UB→GM 区域拷贝 | `references/programming.md`「分配、布局和搬运」；`knowledge/PL-950-vector-execution.md` | ✅ 探针 H1/H2/H7 生成 `asc_copy_gm2ub_align` / `asc_copy_ub2gm_align` |
| 自动 `__mix__(1, 2)` 提升 | 探针 H1/H2 生成源码；`ascend/language/kernel.py:MixedKernel` 文档（混合组总是 `__mix__(1,2)`） | ✅ 编译 + 生成代码核对 |
| `T.MixedKernel(grid, sids=2)` | 同上（源码确认） | ❌ 本结构编译失败（探针 X1/X2），**未采用** |
| 3-D 区域拷贝 | `copy_op.py` | ❌ 编译失败（探针 H6），**未采用** |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| `T.Kernel` 只接受 1-D grid（多维直接拒绝） | **是** | 1-D grid + 整数分解（R1） |
| `T.Kernel` 没有 `threads` 参数 | **是** | 删除 `threads`/`step_threads`（§0.5 #12） |
| 显式 `T.MixedKernel` 在本结构下编译失败（SSA `dup sid`） | **是** | 用普通 `T.Kernel` 的自动 AIC/AIV 提升（探针 H1/H2） |
| L1 fill（`T.clear`）只接受 C0 对齐区域 → K 尾块无法就地补零 | **是** | 用有效区间代替补零（R2 / A3），探针 H4 记录失败形态 |
| `>2D strided copy` 不支持 | **是** | 平移按 batch 行做 2-D 拷贝（§1.6.2 #7） |
| GM→L1 `transpose` 只对 GM→L1 通道有效 | **是** | 权重转置只在装载阶段做；L1→L0 与 L0 布局交给 `T.gemm` 推导 |
| L0B 64 KiB（每核） | **是** | `block_n·block_k·itemsize·num_stages = 64 KiB` = 上限（§5.3）；不得再加版本数 |
| L0C 256 KiB、L1 512 KiB、UB 248 KiB | 是（用量远低于上限） | §4.5 预算表 |
| 跨 block barrier（`ascend.batch_schedule`）未认证（`knowledge/capability-gaps.md` CG-950-batch-barrier） | **否** | 本算子无跨 block 依赖：每块写互不重叠区域，平移按列片切分 |
| 高层 SIMD 二维 shared→shared 归约输出差异（CG-950-simd-shared-reduce） | **否** | 本算子不做归约，也不用 SIMD VF |
| `d_padded % 16 == 0`（尾轴向量对齐） | **是** | host 保证 `d_padded = align_up(d, ALIGNMENT=256)`；5 条 workload 的 `d_padded ∈ {256,512}` |
| `mcl >= dilation*(conv_kernel_size-1)` | 是（上游 `__init__` 已校验，第 293–298 行） | 本 kernel 只依赖 `max_conv_len > 1` 判平移有无 |

#### 3.5.2 参考实现差异说明

| 差异项 | 参考实现（GPU 源算子） | 本项目（Ascend 950PR） | 转换方案 |
|--------|----------------------|----------------------|---------|
| Kernel 结构 | 三维 `T.Kernel(nt, 2, mt)` + `threads=128` | 1-D `T.Kernel(mt*2*nt)`，编译器切 AIC/AIV | R1（§0.6） |
| 矩阵 API | `T.gemm(e_shared, w_shared, acc)`（SMEM→fragment） | `T.gemm(e_l0a_region, w_l0b_region, l0c_region, transpose_B=True)` | L1/L0 显式分层；NT 布局（`transpose=True` 装载） |
| 内存层级 | SMEM（48–228 KiB/block）+ fragment | L1 512 KiB → L0A/L0B 64 KiB → L0C 256 KiB；平移经 UB | §4.4 路径图 |
| 尾块 | 逐元素 `if_then_else` 补零 + 写回守卫 + 行夹取 | 有效区间进区域（`ak/am/an`），mmad 的 M/K 参数精确 | R2/R3/R5 |
| 平移 | 同 kernel 的 `T.Parallel` 逐元素写 | 同 kernel 的 AIV DMA（GM→UB→GM），两个 AIV 重复写同一字节值 | R4 / A4 |
| 启动参数 | `block_m/block_n/block_k/threads/num_stages` | `block_m/block_n/block_k/num_stages`（删 `threads`） | §0.5 #12 |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/shared_expert_mlp/_gemm_basic_kernel/DESIGN.md` | **高**（同为 ascend 平台的 GEMM 迁移设计） | 1-D grid 分核三要素写法、NT 布局、K 段累加、尾块与 L0B 预算的论证方式 |
| `.agents/platforms/toolchains/ascend/knowledge/gemm-tiles.md` | **高**（同平台已验证的 GEMM tile 模式） | `B=[K,N]` 走 `transpose=True` 装载 + `transpose_B=True`；`clear_accum` 只在首段；M/N/K 尾块的区域写法 |
| `.agents/platforms/toolchains/ascend/knowledge/PL-950-cube-tile.md` | 中（单 tile GEMM 的最小验证路径） | `GM → L1 → L0A/L0B → T.gemm → L0C → GM` 的合法性 |
| `.agents/platforms/toolchains/ascend/knowledge/PL-950-vector-execution.md` | 中（向量执行与尾块） | GM↔UB 的有效元素数搬入/写回 |
| `examples/moe_permute_align/_make_align_kernel/repro/TRAP-950-shared-shift-copy.py` | 中（**反例**） | UB→UB 的非零偏移区域拷贝会送错 lane → 本设计所有 UB 拷贝的目的偏移恒为 0 |
| `/workspace/TileOPs-ascend950/src/tileops/kernels/ascend/sequence_modeling/decode.py` | **高**（同算子族的 950DT 已验证实现，只读） | `_build_project` 的 L1/L0 分层与 `annotate_buffer_versions`；`_pick_block_k` 的 L0B 反解 |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `e_t` | `(batch, d_mem)` | float16 / bfloat16 | 本步 embedding，未 pad（`d_mem` 不参与任何 16 对齐要求） |
| `W_K` | `(d_mem, d_padded)` | 同上 | K 投影权重；末维由 host 0 填充到 `d_padded` |
| `W_V` | `(d_mem, d_padded)` | 同上 | V 投影权重 |
| `conv_state` | `(batch, max_conv_len, d_padded)` | 同上 | 已左填充到 `max_conv_len` 的 history（只读） |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `kv` | `(2, batch, d_padded)` | **float32** | `kv[0]`=K、`kv[1]`=V；跨 launch 契约，本 kernel 是唯一写者 |
| `new_conv_state` | `(batch, max_conv_len, d_padded)` | 同 `e_t` | **只写第 `0 .. max_conv_len-2` 行**；末行归 `_engram_step_kernel`，本 kernel 既不读也不写 |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| `e_l1` | `(16, 256)` | 同 `e_t` | L1 | e_t tile（`am × ak` 有效） |
| `w_l1` | `(64, 256)` | 同 `e_t` | L1 | 权重 tile，装载时转置为 `[N,K]` |
| `e_l0` | `(16, 256)` | 同 `e_t` | L0A | GEMM 左操作数 |
| `w_l0` | `(64, 256)` | 同 `e_t` | L0B | GEMM 右操作数（`[N,K]`，`transpose_B=True`） |
| `res` | `(16, 64)` | float32 | L0C | fp32 累加器 |
| `shift_ub` | `(max_conv_len-1, 64)` | 同 `e_t` | UB | 平移暂存（`max_conv_len==1` 时不分配） |

### 4.4 内存搬运路径

```
                   ┌────────────── Cube (AIC) ──────────────┐
GM[e_t]   ──MTE2 nd2nz──▶ L1[e_l1] ──MTE1──▶ L0A[e_l0] ─┐
GM[W_K|W_V] ─MTE2 dn2nz(转置)─▶ L1[w_l1] ─MTE1─▶ L0B[w_l0] ─┴─asc_mmad▶ L0C[res] fp32
                                                          │
                                          MTE3 (L0C→GM) ──┴─▶ GM[kv[which]][m0:m0+am, n0:n0+an]
                   └────────────────────────────────────────┘
                   ┌────────────── Vector (AIV) ────────────┐
GM[conv_state][b, 1:mcl, n0:n0+an] ──MTE2──▶ UB[shift_ub] ──MTE3──▶ GM[new_conv_state][b, 0:mcl-1, n0:n0+an]
                   └────────────────────────────────────────┘
```

两条路径**无数据依赖**（写集不相交、无共享缓冲），由编译器自动插入各自的 pipe 同步（§7）。

### 4.5 UB 内存预算

> 数据来源：平台 `design_budget.capacity_bytes`（CANN 9.1.0 `Ascend950PR_9579.ini` `[AICoreSpec]`，
> md5 `2f0eca1f7ca70f62ee7d6d33990c3be1`，2026-11-23 落库）；本机 9.2.1 INI 复核值一致
> （`ub_size=253952, l1_size=524288, l0_c_size=262144, l0_a_size=65536, l0_b_size=65536`）。

**UB**（Vector 段；`serving-b4` 为最大 `max_conv_len=20`）

| Buffer | Shape | dtype | 版本数 | 占用 |
|--------|-------|-------|--------|------|
| `shift_ub` | (19, 64) | fp16 | 1 | 2.4 KB |
| total | | | | 2.4 KB |

UB 合计 2.4 KB（另加编译器为混合 kernel 追加的少量标量/同步临时量，按编译分配核对）；UB 容量
248 KiB（`253952 B`），占比 ≈ 1 %。

**L1**（Cube 段）

| Buffer | Shape | dtype | 版本数 | 占用 |
|--------|-------|-------|--------|------|
| `e_l1` | (16, 256) | fp16 | 2 | 16 KB |
| `w_l1` | (64, 256) | fp16 | 2 | 64 KB |
| total | | | | 80 KB |

L1 合计 80 KB；L1 容量 512 KiB（`524288 B`），占比 15.6 %。

**L0A**

| Buffer | Shape | dtype | 版本数 | 占用 |
|--------|-------|-------|--------|------|
| `e_l0` | (16, 256) | fp16 | 2 | 16 KB |
| total | | | | 16 KB |

L0A 合计 16 KB；容量 64 KiB，占比 25 %。

**L0B**

| Buffer | Shape | dtype | 版本数 | 占用 |
|--------|-------|-------|--------|------|
| `w_l0` | (64, 256) | fp16 | 2 | 64 KB |
| total | | | | 64 KB |

L0B 合计 64 KB = 容量 64 KiB（`65536 B`），**100 %，是本设计唯一的紧约束**：`block_n`、`block_k`
或 `num_stages` 任一上调都会越界（`block_n=128` 必须同步把 `block_k` 降到 128，§1.6.3 (ii)#3）。

**L0C**

| Buffer | Shape | dtype | 版本数 | 占用 |
|--------|-------|-------|--------|------|
| `res` | (16, 64) | float32 | 1 | 4 KB |
| total | | | | 4 KB |

L0C 合计 4 KB；容量 256 KiB，占比 1.6 %。

### 4.6 动态轴定义

**无运行时动态轴**：`batch`、`d_mem`、`d_padded`、`max_conv_len`、`conv_kernel_size`、`dilation`、
`dtype` 全部是**编译期常量**（工厂键，`functools.lru_cache` + `tilelang.compile` 特化）；op 层
（`EngramDecodeFwdOp._get_kernel`）按 `(batch, d_mem, d, dtype, device_index)` 缓存已编译 kernel，
`conv_state.shape[1]` 故意不进键（短 history 由 host 左填充到 `max_conv_len`）。因此：
`batch` 的「动态」只体现在**跨调用换 shape 时重新选/编译 kernel**，单次执行内的循环边界全为常量
（满足 §3.5.1 的「静态边界」约束）。

### 4.7 JIT 配置

```python
@functools.lru_cache(maxsize=None)
def _build_project(batch, d_mem, d_pad, max_conv_len, dtype_str,
                   block_m, block_n, block_k, num_stages):
    ...
    return tilelang.compile(engram_project, target="ascend", out_idx=[])
```

- `out_idx=[]`：全部输出由调用方预分配（与源算子一致，`kv` 必须在 launch 前存在）。
- 不使用 `@tilelang.jit` 装饰器：编译期常量较多、且需要 `lru_cache` 复用（与 950DT 只读参考的
  `_build_project`、`examples/shared_expert_mlp/_gemm_basic_kernel/` 的写法一致）。
- 编译标志沿用平台默认（`target="ascend"`）；不传 `threads`（§0.5 #12）。
- **配置来源**：`block_m=16`、`block_n=min(64, d_padded)`、`block_k=256`、`num_stages=2`。注意
  harness 的 `default_config` 目前仍是 GPU 派生的
  `{block_m:16, block_n:min(64,d_padded), block_k:_projection_block_k(d_mem)（64/32/16）, num_stages:4}`；
  Stage 3 必须按本节改写（`block_k=256` 由 L0B 反解得到，见 §5.3）。

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**: **混合**（Cube 主导的矩阵乘 + Vector 的纯搬移）。

**判定依据**：两个投影是 `(batch, d_mem) × (d_mem, d_padded)` 的矩阵乘（`d_mem` 是收缩轴，
512–1024），必须落 Cube（§2.2）；cache 平移是纯数据搬移，落 Vector 的 DMA。二者在同一 launch
内由编译器切成 AIC/AIV 两个区域，写集不相交（§4.4）。

### 5.2 Block 划分

```python
block_m = 16          # Cube M 向 fractal 粒度；batch ∈ {1,2,4,8,17} 时 mt = 1/1/1/1/2
block_n = min(64, d_padded)   # 权重 tile 列宽：L0B 预算与 grid 占用的折中（§1.6.3 (ii)）
block_k = 256         # 由 L0B 64 KiB 反解：min(65536/(64*2*2), 256) = 256
block_num = mt * 2 * nt,  mt = ceildiv(batch, block_m), nt = ceildiv(d_padded, block_n)
```

每个 block 负责一个 `(batch tile, which, d tile)` 输出 tile：`kv[which, m0:m0+am, n0:n0+an]`（写）
与（当 `which==0 and m0==0`）该列片的 `nshift` 行平移。

| workload | mt | nt | K 段数 | grid（逻辑核数） | 平移列片 |
|---|---|---|---|---|---|
| decode-s1 | 1 | 4 | 1 | 8 | 4 块 |
| serving-b1 | 1 | 4 | 2 | 8 | 4 块 |
| serving-b4 | 1 | 8 | 4 | 16 | 8 块 |
| serving-b8 | 1 | 4 | 2 | 8 | 4 块 |
| decode-partial-cache | 1 | 4 | 2 | 8 | 4 块 |
| ktail-519（fixture） | 2 | 4 | 3 | 16 | 4 块（仅 `m0==0`） |

### 5.3 约束分析

- **对齐约束**：`d_padded` 由 host 保证 `align_up(d, ALIGNMENT=256)`（`ALIGNMENT` 来自
  `examples/TileOPs/tileops/kernels/sequence_modeling/engram_decode/_engram_decode_fwd_kernels.py`
  第 16 行，值 256），因此 `d_padded % 16 == 0` 恒成立（尾轴 16 元素 = 32 B）；`block_n=64` 时
  `d_padded/64` 整除（5 条 workload 的 `d_padded ∈ {256,512}`），N 尾块分支只在 `d_padded=272`
  这种非 workload 形状出现（探针 H2 覆盖）。`d_mem` **不做任何对齐**：`d_mem=519` 靠 R2 的
  有效区间表达（`ak = 7`）。
- **UB 容量**：`shift_ub` 2.4 KB（最大 `mcl=20`）≤ UB 248 KiB/核。
- **L1 容量**：80 KB（2 版本）≤ 512 KiB。
- **L0A/L0B 容量**：16 KB ≤ 64 KiB；**64 KB = 64 KiB（100 %）**——`block_n·block_k·itemsize·
  num_stages = 64·256·2·2 = 65536 B` 正好等于 L0B 上限。这是本设计唯一的紧约束：
  - 若 `block_n=128`：必须 `block_k=128`（`128·128·2·2 = 64 KiB`），K 段数翻倍（§1.6.3 (ii)#3）。
  - 若 `num_stages=3`：`block_k` 必须降到 128 或 `block_n` 降到 32。
  - 反解公式（承接只读参考 `_pick_block_k`）：`block_k = min(65536 // (block_n · itemsize ·
    num_stages), 256)`，本配置 = 256。
- **L0C 容量**：`16·64·4 = 4 KB` ≤ 256 KiB（fp32 累加器）。
- **K/ M / N 尾块**：见 §6.4。

### 5.4 注意事项

- **`max_conv_len == 1`**：`nshift = 0` → 平移代码整段编译期消失，`shift_ub` 不分配；该 launch
  只写 `kv`（`new_conv_state` 的唯一一行由 step kernel 写）。
- **`L < max_conv_len`（partial cache）**：kernel 侧不感知；host 先把 `conv_state` 左填充到
  `max_conv_len`（源 `forward` 第 388–390 行），因此平移始终按满长 `conv_state[1:mcl]` 读。
- **`batch > block_m`（ktail-519 的 17 行）**：`mt = 2`，第二块 `am = 1`；平移只在 `m0 == 0`
  的块上做（否则同一列片会被 `mt` 个块重复写）。
- **`d_padded % block_n != 0`**：最后一列片 `an < block_n`，区域拷贝与 `asc_mmad` 的 N 参数
  都取 `an`（探针 H2 的 `d_pad=272` 变体）。

### 5.5 分核策略（物理核数适配）⭐

> 三要素标准：`.agents/skills/_shared/standards/core-split-strategy.md` §1。

- **① 逻辑核数计算**：`num_logical_kernels = mt × 2 × nt`（每个逻辑块 = 一次 block 发射，
  负责一个 `(block_m, block_n)` 输出 tile 与一组权重；`2` 是 `which` 轴——K/V 各一份）。
  按 §5.2 的配置：

  | workload | `mt` | `nt` | 逻辑核数 | 波次（/28 Cube 核） |
  |---|---|---|---|---|
  | decode-s1 | 1 | 4 | **8** | 0.29 |
  | serving-b1 | 1 | 4 | **8** | 0.29 |
  | serving-b4 | 1 | 8 | **16** | 0.57 |
  | serving-b8 | 1 | 4 | **8** | 0.29 |
  | decode-partial-cache | 1 | 4 | **8** | 0.29 |
  | ktail-519（fixture） | 2 | 4 | **16** | 0.57 |

- **② 物理核数依据（任务平台记录，非假设值）**：目标型号 **ascend-950pr-9579**（架构 dav-3510）。
  平台 `statectl platform --task-dir examples/engram_decode/_engram_project_kernel` 落库的实查记录
  （`hardware.core_source`，读取日期 2026-11-23）：

  ```
  Ascend950PR_9579.ini [SoCInfo] ai_core_cnt=28, cube_core_cnt=28, vector_core_cnt=56;
  runtime probe rtGetAiCoreCount()=28 on Ascend950PR_9579
  observed_cores = {cube: 28, vector: 56}
  ```

  平台 ruleset 规定的查询口径是 `torch.npu.get_device_properties`
  （adapter `core_evidence()` = `get_device_properties`，实现见
  `.agents/tools/toolchains/ascend.py` 的 `probe_device`）：

  ```python
  import torch, torch_npu
  p = torch.npu.get_device_properties(torch.npu.current_device())
  p.name, p.cube_core_num, p.vector_core_num   # -> "Ascend950PR_9579", 28, 56
  ```

  本次设计期该卡被 `engram_gate_conv` 的 Stage 3 占用（`ASCEND_RT_VISIBLE_DEVICES=""`，只做
  无卡编译/静态检查），因此**引用平台已落库的探测记录**；另用 CPU 侧文件读取复核型号配置
  （2026-11-23，本机 `/usr/local/Ascend/cann-9.2.1/x86_64-linux/data/platform_config/Ascend950PR_9579.ini`，
  `grep -E "ai_core_cnt|cube_core_cnt|vector_core_cnt"` 输出 `28 / 28 / 56`，
  `cube_freq=1650`）。Stage 3 首跑前须用上面同一段代码复核型号与核数（列入 §9.1 待核项）。
  本算子的 Cube 段用 28 个 Cube 核（AIV 只承载 2×grid 个 sub-block，共 ≤ 32 个 AIV 任务，
  远低于 56）。逻辑块到物理核的映射：1 逻辑块 = 1 个 `__mix__(1,2)` block，由硬件块调度器
  按波次派发；L1/L0 为每核私有，block 之间无共享存储，故无 occupancy 约束。
- **③ 规模判定与分核方案**：**逻辑核数 ≤ 物理核数（16 ≤ 28，最大 workload）→ 无需适配**。
  依据：① 的最坏情形是 serving-b4 / ktail-519 的 16 个逻辑块，单波即可全部驻留 28 个 Cube 核，
  不存在第二波调度；不使用 `T.Persistent` 核内串行，也不做 block 数对齐（16 → 28 的整数倍需要
  把块数抬到 28 或 56，而当前每个块的工作量已经小于一次启动开销的量级，抬高块数只会增加
  描述符与调度开销）。
  **为何不再增大并发**：`nt` 由 `d_padded/block_n` 决定，缩小 `block_n` 只能把 8 块抬到 16 块
  （`d_padded=256`），而 L0B 预算与 L1 复用会同时变化——这正是 §1.6.3 末段实验裁决计划要回答
  的问题（备选 `block_n=32`）；在裁决前不改变主选的分核方案。

### 5.6 配置表（默认，进 Stage 3）

| 参数 | 值 | 依据 |
|------|-----|------|
| `block_m` | 16 | Cube M fractal；`asc_mmad` 的 M 取真实 `am`，无乘加浪费 |
| `block_n` | `min(64, d_padded)` = 64 | §1.6.3 (ii)#1；harness 默认一致 |
| `block_k` | 256 | L0B 反解（§5.3） |
| `num_stages` | 2 | 与 L0B 预算绑定；实测调度器未启用多缓冲（§9.1） |
| `grid` | `mt*2*nt` | §5.2 |
| 平移行循环 | `T.serial(batch)` | §1.6.2 #7 |

---

## 6. 循环与调度结构

### 6.1 循环结构总结

> 逐元素计算已按 §1.6.2 全部消除（0 处标量逐元素循环）——本表只列 block 级 / tile 级调度循环。

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| 输出 tile（`mt×2×nt`） | block 级并行 | `T.Kernel(grid)` | 每 block 一个 `(batch tile, which, d tile)`，互不重叠 |
| K 收缩（`ceil(d_mem/block_k)`） | 顺序迭代（流水注解） | `T.Pipelined(kt_max, num_stages=2)` | 循环携带 L0C 累加器（§1.6.2 #3）；`annotate_buffer_versions` 声明 L1/L0 四缓冲的版本数 |
| cache 行（`batch`） | 顺序迭代（DMA 描述符） | `T.serial(batch)` | 3-D 区域拷贝不被支持（探针 H6），逐行 2-D DMA 是唯一可用形态（§1.6.2 #7） |
| 元素级 | **无** | — | 装载/写回/平移都是 `T.copy` 区域 DMA；矩阵乘在 Cube 单元 |

### 6.2 循环伪代码

```python
with T.Kernel(mt * 2 * nt) as bx:                  # block 级并行
    m0 = (bx // (2 * nt)) * block_m
    which = (bx // nt) % 2
    n0 = (bx % nt) * block_n
    am = T.min(batch - m0, block_m)
    an = T.min(d_pad - n0, block_n)
    if which == 0:
        for kt in T.Pipelined(kt_max, num_stages=2):     # tile 级顺序 + 流水注解
            ak = T.min(d_mem - kt * block_k, block_k)
            T.copy(...)                                  # GM→L1→L0（DMA）
            T.gemm(..., clear_accum=(kt == 0))            # Cube
        T.copy(res[0:am, 0:an], kv[0, m0:m0 + am, n0:n0 + an])
        if nshift > 0 and m0 == 0:
            for b in T.serial(batch):                    # DMA 描述符循环
                T.copy(conv_state[b, 1:max_conv_len, n0:n0 + an], shift_ub[0:nshift, 0:an])
                T.copy(shift_ub[0:nshift, 0:an], new_conv_state[b, 0:nshift, n0:n0 + an])
    else:
        ... 同 which==0 的投影段，权重换 W_V，写 kv[1] ...
```

### 6.3 流水线优化

- 使用 `T.Pipelined(kt_max, num_stages=2)`，并对 `e_l1/w_l1/e_l0/w_l0` 注解 `num_stages` 个版本
  （正整数，`annotations.py` 要求）。
- **实测注意事项**：本结构下 AutoSchedule 打印
  `did not enable multi-buffering for buffer "e_l1"/"w_l1"/"e_l0"/"w_l0" although
  T.annotate_buffer_versions requested 2 versions. The buffer must be eligible in a scheduled
  serial loop and written before it is read.`（探针 H1/H2 每次编译都出现）。即：**当前得到的
  是「顺序 K 循环 + 编译器插入的 L1/L0 同步」**，跨段重叠未启用。这不影响正确性（累加器语义
  与同步由编译器保证），只影响 K 段数 ≥ 2 的 workload（serving-b1/b4/b8/partial/ktail）的
  潜在重叠收益；Stage 4 可在不改语义的前提下尝试 `num_stages=1`（单缓冲、少一次版本开销）或
  手工 `order`/`stage` 调度（`T.Pipelined` 支持，属调优项，不改变 §3.1 公式）。
- `num_stages` 与 L0B 预算绑定：`num_stages=2` 已是 64 KiB 上限（§5.3），不能上调。

### 6.4 尾块处理

| 尾块 | 触发条件 | 处理 |
|------|----------|------|
| M 尾块 | `batch % block_m != 0`（batch ∈ {1,2,4,8} 时 `am<16`；17 时第二块 `am=1`） | `am = min(batch-m0, block_m)` 进装载/`gemm`/写回区域；`asc_mmad` 的 M 参数 = `am`；死行不装载、不写回（R3） |
| K 尾块 | `d_mem % block_k != 0`（仅 `d_mem=519`） | `ak = min(d_mem-k0, block_k)` 进区域；`asc_mmad` 的 K 参数 = `ak`（R2/A3）；残余风险见 §9.2 E1 |
| N 尾块 | `d_padded % block_n != 0`（workload 不触发；`d_padded=272` 探针覆盖） | `an = min(d_padded-n0, block_n)` 进装载/写回区域 |
| 平移行数 | `max_conv_len == 1` | 编译期 `nshift == 0` → 整段消失 |
| 平移列片 | 最后一个 `d` tile | `an` 截断（同上） |

---

## 7. 同步策略

### 7.1 同步模式

**模式**: **自动同步**（`ascend/pipeline.py` 的 `allow_autoschedule` 默认开启：AutoSchedule →
InsertSync → MaterializeMultiBuffer）。本设计**不写**任何 `ascend_set_flag` / `ascend_cross_core_*`
/ `ascend_get_buf` 手工同步——平台 ruleset 明确「手工跨核 flag 配对未验证」，且本算子没有任何
跨 block 或跨 AIC/AIV 的数据依赖（§4.4）。

### 7.2 同步点说明

生成代码中由编译器插入的同步（探针 H1/H2/H7 的 `get_kernel_source()` 实读）：

| 位置 | 同步 API（生成代码） | 理由 |
|------|---------------------|------|
| AIC：`e_l1` 装载 → `l12l0a` | `asc_sync_notify(PIPE_MTE2, PIPE_MTE1, ev)` / `asc_sync_wait(PIPE_MTE2, PIPE_MTE1, ev)` | 等 GM→L1 完成再搬 L0A |
| AIC：`w_l1` 装载 → `l12l0b` | 同上（独立 event id） | 等 GM→L1 完成再搬 L0B |
| AIC：`l12l0a/l12l0b` → `asc_mmad` | `asc_sync_notify(PIPE_MTE1, PIPE_M, 0)` / `asc_sync_wait(PIPE_MTE1, PIPE_M, 0)` | 等 L0A/L0B 就绪再算 |
| AIC：`asc_mmad` → 下一段的 L1 覆盖 | `asc_sync_notify(PIPE_M, PIPE_MTE1, 0)` / `asc_sync_wait(PIPE_M, PIPE_MTE1, 0)`（每段循环末尾） | 累加器与 L0 复用保护 |
| AIC：最后一段 → `l0c2gm` | `asc_sync_notify(PIPE_M, PIPE_FIX, 0)` / `asc_sync_wait(PIPE_M, PIPE_FIX, 0)` | 等累加完成再写 GM |
| AIV：`gm2ub` → `ub2gm`（平移暂存） | `asc_sync_notify(PIPE_MTE2, PIPE_MTE3, 0)` / `asc_sync_wait(PIPE_MTE2, PIPE_MTE3, 0)` | 等 UB 装载完成再写回（**自动提升路径**下存在；显式 `sid` 拆分路径会丢这对同步，故不采纳，§0.6 R4） |
| AIV：`ub2gm` → 下一行 `gm2ub`（复用 `shift_ub`） | `asc_sync_notify(PIPE_MTE3, PIPE_MTE2, 0)` / `asc_sync_wait(PIPE_MTE3, PIPE_MTE2, 0)`（每行循环末尾） | 缓冲复用保护 |

AIC 与 AIV 之间**没有**同步点：两者写集不相交、无共享缓冲，由 `__mix__(1,2)` 的 block 级
生命周期保证同一 block 的两个区域都结束才回收。

### 7.3 pass_configs 配置

```python
# 平台默认流水线：allow_autoschedule 开启（不显式关闭）
pass_configs = {}   # 不覆盖 AutoSchedule / InsertSync / MaterializeMultiBuffer
```

不传任何自定义 pass 配置：本设计依赖默认自动调度；关闭自动调度需要手工配对全部 flag
（`.agents/platforms/toolchains/ascend/knowledge/` 的 PL-950-manual-sync 只覆盖单 tile 单核），
超出当前验证范围。

---

## 8. 验证方案

### 8.1 Golden 函数

> 迁移任务：golden 以 **§0.1 源算子语义**为唯一依据，直接移植源仓参考实现（**不复刻 §0.6 的
> NPU 算法**：不引入区域截断、不做 `clear_accum`、不写 `shift_ub` 等实现细节）。

```python
import torch
import torch.nn.functional as F


def golden_engram_project(e_t, W_K, W_V, conv_state, max_conv_len):
    """`_engram_project_kernel` 的逐位语义参考（移植自源仓 torch 参考）。

    返回 (kv, new_conv_state)：
      kv[0] = e_t @ W_K（fp32）、kv[1] = e_t @ W_V（fp32）；
      new_conv_state[:, :max_conv_len-1, :] = conv_state[:, 1:, :]，
      末行置 NaN 哨兵（本 kernel 不写该行，用于断言「未被触碰」）。
    """
    batch, d_mem = e_t.shape
    d_pad = W_K.shape[1]
    mcl = max_conv_len
    L = conv_state.shape[1]

    # 源参考：k = e_t.float() @ W_K.float(); v = e_t.float() @ W_V.float()
    k = e_t.float() @ W_K.float()
    v = e_t.float() @ W_V.float()
    kv = torch.stack([k, v], dim=0)                        # (2, batch, d_pad) fp32

    # 源参考：host 先左填充到 max_conv_len，再 new = cat([state[:, 1:, :], vhat_n])
    if mcl > L:
        state = F.pad(conv_state, (0, 0, mcl - L, 0))      # 左填充，与 forward 第 388-390 行一致
    else:
        state = conv_state
    new_state = torch.empty(batch, mcl, d_pad, dtype=state.dtype, device=state.device)
    new_state[:, : mcl - 1, :] = state[:, 1:, :]           # 本 kernel 负责的行（逐位拷贝）
    new_state[:, mcl - 1, :] = float("nan")               # step kernel 负责的行：哨兵
    return kv, new_state


def golden_engram_decode_full(e_t, h_t, conv_state, W_K, W_V, rms_w_h, rms_w_v, conv_w,
                              max_conv_len, dilation, eps=1e-6):
    """整算子 golden：直接调用源仓 engram_decode_step_torch（y_t + 完整 new_conv_state）。"""
    from tileops.workloads.sequence_modeling import engram_decode_step_torch
    state = conv_state
    if max_conv_len > conv_state.shape[1]:
        state = F.pad(conv_state, (0, 0, max_conv_len - conv_state.shape[1], 0))
    return engram_decode_step_torch(e_t, h_t, state, W_K, W_V, rms_w_h, rms_w_v, conv_w,
                                    max_conv_len, dilation, eps)
```

**分段断言（L0 层）**：
1. `kv` 与 `golden` 的 fp32 结果按 `atol=1e-3, rtol=1e-3` 比对（差异只可能来自 K 段累加顺序，
   但本设计保持「按 K 段串行」的同一分组，实测应远小于该阈值）；
2. `new_conv_state[:, :mcl-1, :]` 与 `conv_state[:, 1:, :]` **逐位相等**
   （`torch.equal` on `.view(torch.int16)`，覆盖 NaN/±inf）；对应源仓测试
   `test_engram_decode_new_conv_state_history`（第 406 行）的断言口径；
3. `new_conv_state[:, mcl-1, :]` **保持预填的 NaN 哨兵**（证明本 kernel 没有越界写）。

### 8.2 精度标准

| 对比项 | dtype | atol | rtol | 依据 |
|--------|-------|------|------|------|
| `kv` vs fp32 golden（分段 L0） | float32 | 1e-3 | 1e-3 | fp32 累加器，仅分组顺序可能不同 |
| `new_conv_state[:, :mcl-1, :]` vs `conv_state[:, 1:, :]` | fp16 / bf16 | 0（逐位） | 0 | 纯 dtype 拷贝 |
| 整算子 `y_t` / `new_conv_state`（op 级） | float16 | **0.05** | **0.05** | `examples/TileOPs/tests/ops/test_engram.py` 的 `_tol_decode`（第 67–70 行） |
| 整算子 `y_t` / `new_conv_state`（op 级） | bfloat16 | **0.1** | **0.05** | 同上 |
| `basis` 角点（±inf/NaN/subnormal） | 全部 | 位型一致 | — | 语义要求「按 IEEE 传播」 |

### 8.3 L0 门槛测试计划

按平台 `kernel_template.py` 的分层约定（`--level L0/L1/L2/boundary/all`），本 kernel 的
L0 门槛（Stage 3 必须先全绿才进 L1）：

| 层 | 用例（shape 来源） | 断言 |
|----|-------------------|------|
| **L0** | `decode-s1`（batch=1, d_mem=256, d_padded=256, mcl=9, fp16） | `kv` 数值 + 平移逐位 + 末行哨兵不变 |
| **L0** | `serving-b1`（batch=1, d_mem=512, d_padded=256, mcl=12, fp16） | 同上；K 段数 = 2（覆盖累加链） |
| **L0** | `serving-b8`（batch=8, d_mem=512, d_padded=256, mcl=18, bf16） | 同上；覆盖 bf16 与 `am=8` |
| **L1** | `serving-b4`（batch=4, d_mem=1024, d_padded=512, mcl=20, fp16） | 16 逻辑块 / 4 K 段 / `nt=8` |
| **L1** | `decode-partial-cache`（batch=2, d_mem=512, d_padded=256, L=5, mcl=12） | host 左填充路径；`new_conv_state` 12 行全检 |
| **L1** | `ktail-519`（batch=17, d_mem=519, mcl=12, fp16） | **K 尾块（`ak=7`）+ M 尾块（`am=1`）**；§9.2 E1 的判定用例 |
| **boundary** | `max_conv_len=1`（batch=1, d_padded=256） | 平移整段消失、`new_conv_state` 全 NaN 哨兵 |
| **boundary** | `d_padded=272`（N 尾块）、`batch=1` 与 `batch=17` | `an∈{16,32,48,64}`、`am∈{1,16}` |
| **boundary** | `d_mem < block_k`（`decode-s1` 的 256 = 1 段） | 单段 `clear_accum` 路径（A1） |
| all | 上述全部 + op 级 `test_engram_decode*` | 8.2 的 op 级容差 |

分层套件（L1/L2/Boundary 的完整枚举）由 `tilelang-op-develop` 承接；本节只冻结 L0 门槛。

---

## 9. 风险点与注意事项

### 9.1 已知约束

- **多缓冲未启用**：`annotate_buffer_versions` 请求 2 版本，但 AutoSchedule 在本结构下拒绝
  （§6.3 的警告原文）。后果：K 段之间没有软件流水的重叠，且版本计数实际为 1（L1/L0 占用可能
  低于 §4.5 的 2 版本预算）。**不得**据此把 `block_k` 抬到 1024（L0A/L0B 的单版本上限是
  `block_k ≤ 512` 且 `block_n·block_k·2 ≤ 64 KiB`，抬 `block_k` 会越界）。
- **L0B 100 %**：任何 tile 上调都需重算（§5.3）。
- **显式 `T.MixedKernel` 不可用**、**L1 中转不可用**、**3-D 区域拷贝不可用**（探针 X1/X2、H5、H6）。
- **待 Stage 3 复核项**：设备型号与核数（用 §5.5 的 `get_device_properties` 片段）；
  `asc_mmad` 对非 16 整数倍 K 的掩蔽行为（§9.2 E1）。
- **partial-cache 的参考实现形状不一致**（§9.2 E7）。

### 9.2 常见错误

| # | 错误 | 触发场景 | 影响 | 解决方案 |
|---|------|----------|------|----------|
| E1 | `asc_mmad` 收到 `K=7`（`d_mem=519` 的最后一段） | `d_mem % block_k != 0` 且尾段不是 16 的整数倍 | 若硬件按 16 lane 计算而不掩蔽，L0A/L0B 尾 lane 的陈旧数据会污染累加结果（**未在无卡环境证实**） | 判定：L0 层的 `ktail-519` 用例与 fp32 golden 比对（8.2 的 1e-3）。回退：host 把 `e_t` 与 `W_K/W_V` 的收缩轴 0 填充到 16 的整数倍（519 → 528，`F.pad` 产生新张量，不改输入内容），`d_mem` 作为 kernel 参数同步改成 528；代价仅限该 fixture（5 条 manifest workload 的 `d_mem ∈ {256,512,1024}` 全是 256 的倍数） |
| E2 | L0B 超限 | `block_n=128` 且 `block_k=256`（`128·256·2·2 = 128 KiB`） | 编译/分配失败 | 按 §5.3 的反解同步降 `block_k` 或 `num_stages` |
| E3 | 用 3-D 区域拷贝做平移 | 想「一条 DMA 搬整个 cache」 | 编译失败 `>2D strided copy not yet supported`（探针 H6） | 保持 `T.serial(batch)` 的逐行 2-D 拷贝 |
| E4 | 改用 `T.MixedKernel` 显式分区 | 想给 AIV 精确的 `sid` | 编译失败 `Need input to be in SSA form dup sid`（探针 X1/X2） | 用普通 `T.Kernel` 的自动提升（§0.6 R4） |
| E5 | 在普通 `T.Kernel` 里包 `T.Vector(...) as sid` 来拆平移行 | 想消除两个 AIV 的重复写 | 能编译，但 AIV 段的 UB 暂存**丢失 MTE2→MTE3 同步**（探针 H7 对比） | 不采纳该写法；接受重复写（同字节值、无读者、无别名） |
| E6 | 把平移放到「每个 batch tile」块上 | `mt > 1`（batch=17） | 同一列片被 `mt` 个块重复写（幂等但翻倍流量，且与 `m0==0` 守卫的意图冲突） | 保留 `m0 == 0` 守卫（源实现的 `row_block == 0` 同理） |
| E7 | partial-cache 的 golden 用未填充的 `conv_state` 调源参考 | `L < max_conv_len` | `engram_decode_step_torch` 在该分支返回 **`L+1` 行**（第 465–470 行），与 op 契约（恒 `max_conv_len` 行）不一致；`examples/TileOPs/tests/ops/test_engram.py` 第 464 行的 `ref_state.shape == (batch, max_conv_len, d)` 断言会与当前参考冲突 | golden 先左填充到 `max_conv_len` 再调参考（§8.1 的 `state = F.pad(...)`），使参考走「满 cache」分支返回 `max_conv_len` 行；该不一致须回报 harness 维护者 |
| E8 | 忘记 `an` 截断 | `d_padded % block_n != 0`（如 272） | 越界写 `kv` / 越界读 `W` | 全部区域与 `asc_mmad` 的 N 用 `an`（§5.4） |
| E9 | 沿用 GPU 的 `block_k=64, num_stages=4` | 直接照抄 harness `default_config` | 能运行但 K 段数 ×4、L0B 只用 1/4，且 `num_stages=4` 与 L0B 预算的耦合被打破 | 按 §4.7 / §5.6 改写配置 |

### 9.3 特殊场景处理

- **`d_mem < block_k`**（`decode-s1` 的 256）：`kt_max = 1`，K 循环只跑一段，`clear_accum=True`
  即唯一一段（A1 的极端形态：完全省掉清零）。
- **`batch` 极小（1）**：`am = 1`，`asc_mmad` 的 M = 1（探针 H2 实测 M=8/17 的同类路径）；
  M 方向无乘加浪费，只是 L0A/L0C 的分配按 16 行（容量上可忽略：L0C 4 KB）。
- **bf16**：`dtype_str = "bfloat16"`；`compile_flags` 里源实现带 `-DENABLE_BF16`，NPU 侧不需要
  （转由 `dtype_str` 与 `T.gemm` 推导），但 Stage 3 若遇到 bf16 编译问题，可按平台模板核对
  是否需要等价开关。
- **混合精度边界**：`kv` 恒 fp32（契约），`new_conv_state` 恒输入 dtype；两者之间**不做**任何
  额外的舍入或 cast。
- **极端值**：NaN/±inf 在 fp32 累加里按 IEEE 传播（与源一致）；平移逐位拷贝（不经过算术），
  因此 `-0.0` 的位型也被保留（`verify_equiv.py` 的 `build_detail` 与 A1 的 corner 用例覆盖）。

---

## 10. 交付清单

### 10.1 目录结构

```
examples/engram_decode/_engram_project_kernel/
├── DESIGN.md                                  # 本设计文档
├── verify_equiv.py                            # D-1 等价性机器验证（已执行，ALL EQUIV_PASS）
├── RETROSPECTIVE.md                           # Stage 1 复盘
└── history_version/
    ├── design_probe_compile.py                # 探针 #1（上一轮已存在，K 尾/N 尾/混合 kernel 形态）
    ├── design_probe_shift_placement.py        # 探针 #2（平移落点：X1/X2 + A_* 菜单）
    ├── design_probe_ktail_clear.py            # 探针 #2 附加：K 尾就地补零（失败证据）
    ├── design_probe_shift_3d.py               # 探针 #2 附加：3-D 区域拷贝（失败证据）
    └── __pycache__/                           # 探针运行产物
```

Stage 3 产出的 `_engram_project_kernel` 实现进入 TileOPs 抽取目录
`examples/TileOPs/tileops/kernels/sequence_modeling/engram_decode/`（与 `_engram_step_kernel`
同一文件），本阶段不产出 `{op}.py`。

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | ✅ 已完成 | 本设计文档 |
| `verify_equiv.py` | ✅ 已完成并执行 | D-1 机器验证（§1.6.1 结果表） |
| `RETROSPECTIVE.md` | ✅ 已完成 | Stage 1 复盘（追加式） |
| `history_version/design_probe_*.py` | ✅ 已完成并执行 | 设计期探针（compile-only） |
| `{op}.py`（kernel 实现） | ⬜ 待 Stage 3 | 按 §3.3 伪代码 + §5.6 配置 |

### 10.3 命名规范

- 算子目录：`examples/engram_decode/_engram_project_kernel/`（项目 `engram_decode`，算子
  `_engram_project_kernel`，与 harness 的 `.migration_meta.json` 逐函数切分一致）。
- 编译工厂：`_build_project(batch, d_mem, d_pad, max_conv_len, dtype_str, block_m, block_n,
  block_k, num_stages)`（`functools.lru_cache`）；`prim_func` 名 `engram_project`（与 step kernel
  的 `engram_decode_step` 区分，便于 profiler 按名字分行）。

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md）
2. ✅ 等价性机器验证（verify_equiv.py，ALL EQUIV_PASS）
3. ⬜ kernel 实现（`_engram_project_kernel`）→ L0 门槛（§8.3）→ L1/boundary
4. ⬜ 与 `_engram_step_kernel` 的跨 launch 契约联调（§1.5 表）+ op 级 `test_engram_decode*`

---

## 11. 设计期探针与未决项

### 11.1 设计期探针记录（compile-only，`ASCEND_RT_VISIBLE_DEVICES=""`，未占用设备）

环境：`source .agents/platforms/toolchains/ascend/env-950pr.sh`；
`/workspace/venv-ascend950/bin/python history_version/<脚本>`（cwd = 任务目录）。
工具链：TileLang ascend 0.1.15 + CANN 9.2.0 编译 SDK（`target="ascend"`，dav-3510）。

| 探针 | 问题 | 结果 | 结论 |
|------|------|------|------|
| **H1** | 普通 `T.Kernel` 同时持有 L1/L0 GEMM 与 UB 平移，能否编译？ | ✅ 编译通过；生成 `extern "C" __global__ __mix__(1, 2) void main_kernel(...)`，内含 `if ASC_IS_AIC`（`asc_copy_gm2l1_dn2nz`/`asc_copy_l12l0a`/`asc_mmad`/`asc_copy_l0c2gm`）与 `if ASC_IS_AIV`（`asc_copy_gm2ub_align`/`asc_copy_ub2gm_align`，带 MTE2→MTE3 同步） | **采纳**：混合模式由编译器自动生成，无需显式 frame |
| **H2** | 上述结构在 7 种形状（b8 fp16/bf16、b1 bn=32、b4 d512/bn64、b17 d_mem519、mcl=1、d_pad 272）下是否都能编译？ | ✅ 7/7 通过；`d_mem=519` 生成 `asc_mmad(..., M=(17-max(...)), K=(519-max(kt*256,263)), N=64, ...)`（尾段 K=7）、`d_pad=272` 生成 `N=(272-max(...))`、`mcl=1` 不生成 AIV 代码 | **采纳**：M/K/N 尾块与 bf16 全覆盖 |
| **H3/H4** | K 尾块能否就地补零（`T.clear(e_l1[0:am, ak:block_k])`）？ | ❌ `Ascend L1 fill requires a C0-aligned region on buffer e_l1, got Range(0, ?) with C0=16` | **放弃该路线**，改用有效区间（R2/A3） |
| **H5** | 平移走 L1 中转（`alloc_l1` + GM→L1→GM）？ | ❌ `ScalarDcacheBypass: vector-valued buffer store not implemented` | **放弃**，UB 是唯一可用中转 |
| **H6** | 平移用单个 3-D 区域拷贝？ | ❌ `PlanMTECopy: >2D strided copy not yet supported. src has 3 modes, dst has 1 modes after coalesce` | **放弃**，保留逐 batch 行 2-D 拷贝 |
| **H7** | 用 `with T.Vector(vector=2) as sid:` + `if sid==0/1` 把平移行拆给两个 AIV？ | ✅ 编译通过、行区间确实互斥（sid 0 → 行 0..3，sid 1 → 行 4..7）；但生成代码的 AIV 段**没有** `PIPE_MTE2→PIPE_MTE3` 的 notify/wait 对（H1 的自动提升路径有） | **不采纳**：同步缺失是正确性风险；接受重复写（同字节值） |
| **X1/X2** | 显式 `T.MixedKernel(grid, sids=2)` + `T.Cube()`/`T.Vector()` 分区？ | ❌ `Check failed: (!var_idmap_.count(v)) is false: Need input to be in SSA form dup sid`（用户是否引用 `sid` 都一样） | **不可用**，改用 H1 的自动提升 |

### 11.2 待 Stage 3/4 核对的假设与未决项

| # | 假设 / 未决项 | 核对方式 | 若不成立的影响 |
|---|--------------|----------|----------------|
| 1 | 设备型号 = Ascend950PR_9579、28 Cube / 56 Vector | §5.5 的 `torch.npu.get_device_properties` 片段（首跑前执行一次） | 分核方案（§5.5 ③「无需适配」）需按实际核数重算 |
| 2 | `asc_mmad` 对 `ak=7` 掩蔽尾 lane | L0 层 `ktail-519` 用例（§8.2 的 1e-3） | 按 §9.2 E1 的回退：host 把 `d_mem` 补到 16 的整数倍 |
| 3 | 两个 AIV 的重复写在真机上无副作用（无别名、无读者、字节相同） | L0 层「平移逐位 + 末行哨兵」断言（§8.1） | 若出现写竞争迹象，改用 H7 的 sid 拆分并**补手工同步**（需先取得手工 flag 配对的验证依据） |
| 4 | 带宽（1.0–1.6 TB/s）、单次 launch（≈5 µs）、Cube/AIV 峰值（378/23.6 TFLOP/s）三个**未实测假设** | Stage 4 baseline 采集后回填 `perf_opt/opt_log.md` 的「设计估算 vs 实测」偏差行（`.agents/skills/_shared/standards/hardware-cost-model.md` §3） | 只影响 §1.6.0 的「估算下界」行与 §1.6.3 的实验裁决阈值，不影响结构决策（发射项主导的结论对 1 µs 量级的常数不敏感） |
| 5 | `block_n=64` 优于 `block_n=32` | §1.6.3 末段的实验裁决计划（serving-b1/b8，阈值 5 %） | 按 shape 分派到备选（工厂 `lru_cache` 支持） |
| 6 | 多缓冲未启用（§6.3）是否可在不越 L0B 预算的前提下打开 | Stage 4 的 `num_stages` / 手工 `order`/`stage` 试验（A/B） | 只是性能收益项，不影响正确性 |
