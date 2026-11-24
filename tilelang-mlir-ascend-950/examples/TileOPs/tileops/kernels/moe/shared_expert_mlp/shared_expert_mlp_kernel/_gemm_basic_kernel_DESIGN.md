# `_gemm_basic_kernel` 算子设计文档

> **迁移任务（harness 子模式），首次设计 v1**。源算子：
> `/workspace/TileOPs/src/tileops/_kernels/gemm/dense.py` 的 `_gemm_basic_kernel`
> （GPU TileLang，CUDA target，约第 4669 行起）；同仓抽取件
> `examples/TileOPs/tileops/kernels/moe/shared_expert_mlp/_shared_expert_mlp_fwd_kernels.py`
> 为其逐字副本（含 `_add_bias` / `_main` 两个 helper）。
>
> 目标平台：`ascend-950pr-9579`（dav-3510，28 Cube / 56 Vector，UB 248 KiB / L1 512 KiB /
> L0C 256 KiB / L0A·L0B 各 64 KiB，L2 128 MiB），工具链 `ascend`
> （TileLang ascend 0.1.15 + CANN 9.2.0 编译 SDK / 9.1.0 runtime，`target="ascend"`，
> `import tilelang.ascend.language as T`）。平台选择由
> `statectl platform --task-dir` 落库（2026-11-23）。
>
> **设计范围**：仅 `_gemm_basic_kernel`（dense GEMM 主循环 + epilogue）。同批次的
> `splitk_reduce_kernel`（split-K 归约 + silu_and_mul epilogue）与 `_silu_mul_fused_kernel`
> 是另两个独立设计任务；本文只在**输出契约**上说明与本 kernel 的接口（§0.1、§4.2）。
> GPU 的 SM90 变体（`GEMMTMAKernel`/`GEMMTemplate`/grouped GEMM/warp-specialized）在 Stage 0
> 脚手架已按「NPU 无对应物」丢弃，本文不重复论证（§0.5 表末一行留档）。
>
> 本文档由 `tilelang-ascend` → `tilelang-op-design` 流程产出：Phase M0（三问解读）→
> Phase R（算法调研，§1.6.0）→ Phase M1（耦合性判定与重设计，§0.5–§0.6）→ Phase 2
> （算法级优化，§1.6.1–§1.6.3）。首次设计，无「相对上一版的关键调整」（§11.4）。

---

## 0. 源算子解读与迁移分析

### 0.1 源算子语义（做什么）

`_gemm_basic_kernel(m, n, k, trans_a, trans_b, dtype, has_bias, out_dtype)` 是一个
**工厂函数**：它返回一个内层 JIT 工厂 `_gemm_basic_func(block_m, block_n, block_k,
num_stages, threads, split_k)`，内层再专门化出一个 `T.prim_func`。因此「算子语义」有两层：

**外层（数学语义）**：对 `A`、`B` 两个输入张量按 `(trans_a, trans_b)` 指定的布局做收缩，
可选加列 bias，结果写入 `C`：

```
C[m, n] = op(A) · op(B) + (has_bias ? 1_m ⊗ bias : 0)
op(A) = A^T (trans_a) 或 A        # 语义形状恒为 (m, k)
op(B) = B^T (trans_b) 或 B        # 语义形状恒为 (k, n)
```

四种布局与存储形状（源码 `a_shape` / `b_shape`）：

| trans_a | trans_b | A 存储 | B 存储 | 语义 |
|---|---|---|---|---|
| False | True | `(m, k)` | `(n, k)` | NT（本 op 的两个 GEMM 都用这一种） |
| False | False | `(m, k)` | `(k, n)` | NN |
| True | True | `(k, m)` | `(n, k)` | TN |
| True | False | `(k, m)` | `(k, n)` | TT |

**I/O 契约**：

- `C` 的 shape 随 `split_k` 分段：`split_k == 1` → `(m, n)`，dtype = `out_dtype or dtype`；
  `split_k > 1` → `(split_k, m, n)`，dtype **固定 `float32`**（分片部分和，由伴随的
  `splitk_reduce_kernel` 求和并量化）。这是本 kernel 对下游的硬契约（§4.2）。
- `has_bias` 只在 `split_k == 1` 时生效（本 kernel 的签名里出现 `bias[n]` 参数）；
  `split_k > 1` 时 bias 留给归约 kernel（源 docstring 原文）。
- 输出由 `@tilelang.jit(out_idx=[-1])` 自动分配（调用方不预分配）。

**规约语义**：

- 收缩轴 = `k`，沿 `k` 分块累加；**累加器 dtype = fp32（`accum_dtype = "float"`）**，
  跨 K tile 保留（`T.clear(c_local)` 只在进入循环前一次，`T.gemm` 不再清零）。
- `split_k > 1` 时收缩被切成 `split_k` 个连续 K 段，每段独立累加到自己的 fp32 部分和；
  归约 kernel 再按切片序 `s = 0, 1, …` 顺序相加。**累加分组因此改变**（§1.6.1 A2 用机器验证
  量化该影响：目标 dtype 输出 0 ulp 差、violation rate 0）。
- `split_k > 1` 有前置约束（源码 `raise ValueError`）：`split_k` 必须整除 K tile 数，
  且 `m % block_m == 0 and n % block_n == 0 and k % block_k == 0`（精确 tile）。

**dtype 语义**：输入 fp16/bf16；中间累加 fp32；输出在最后一次存储时舍入到 `c_dtype`
（`split_k == 1`）或保持 fp32（`split_k > 1`）。bias 在 fp32 累加器上加（`_add_bias` 把
`bias` cast 到 `acc.dtype` 后相加），**不是**在输出 dtype 上加。

**边界语义**：

- `m`/`n` 尾块：M/N 方向允许非整除。尾块的越界行/列**不写**（`if m_start + i < m and
  n_start + j < n` 的 epilogue 守卫）；装载进 SMEM 的越界元素落在被守卫丢弃的死行/死列上
  （源码注释：「M/N tail reads land in c_local rows/cols that the epilogue guard skips」）。
- `k` 尾块：源码注释明确「a K tail would corrupt live outputs, so block_k must divide k」——
  **K 方向不允许尾块**（`k % block_k == 0` 是隐含前置；配置链/autotune 过滤器保证）。
- `k == 0`：`_GEMMCpAsyncKernel.refusal` 明确放行（主循环零步，只写 bias 或零）。
- 非有限值：无特殊处理，按 IEEE 传播（fp32 累加器；N 尾块的 NN 掩码路径用
  `T.if_then_else` 填 0）。

**host 侧语义**：`_main` 只做函数签名装配（有/无 bias 两种 `global_symbol`）与
`out_idx` 输出分配；shape 与 `split_k` 校验在 Python 层（编译期常量判定，不是运行时逻辑）。
无可变数据的预处理/后处理（无 im2col、无 permute、无 cast 前置）。

### 0.2 源算子输入输出

| 参数 | 方向 | Shape | dtype | 说明 |
|---|---|---|---|---|
| `A` | in | `(k, m)`（trans_a）或 `(m, k)` | float16 / bfloat16 | 存储布局由 `trans_a` 决定 |
| `B` | in | `(n, k)`（trans_b）或 `(k, n)` | float16 / bfloat16 | 存储布局由 `trans_b` 决定 |
| `bias` | in | `(n,)` | 同输入 dtype | 仅 `has_bias and split_k == 1` 时在签名中出现 |
| `C` | out | `(m, n)` / `(split_k, m, n)` | `out_dtype or dtype` / float32 | 源算子输出 shape：`C [m, n]`（`split_k == 1`）；split-K 为 `[split_k, m, n]` fp32 |

本任务 5 条 workload 下本 kernel 被调用的 10 次（每次 Op 调用 2 个 GEMM；manifest
`examples/TileOPs/tileops/manifest/moe.yaml` 的 `SharedExpertMLPFwdOp`）：

| workload | 调用 | m | n | k | dtype | split_k（调用方 `_split_k`） |
|---|---|---|---|---|---|---|
| kimi-k2-t1 | gate/up | 1 | 36864 | 7168 | bf16 | 4（**不可用，见 §5.2 / §9.2 E6**） |
| kimi-k2-t1 | down | 1 | 7168 | 18432 | bf16 | 4（同上） |
| kimi-k2-t512 | gate/up | 512 | 36864 | 7168 | bf16 | 1 |
| kimi-k2-t512 | down | 512 | 7168 | 18432 | bf16 | 1 |
| ds-v3-t32 | gate/up | 32 | 4096 | 7168 | bf16 / fp16 | 4（`block_m` 需为 32） |
| ds-v3-t32 | down | 32 | 7168 | 2048 | bf16 / fp16 | 4（同上） |
| ds-v3-t4096 | gate/up | 4096 | 4096 | 7168 | bf16 | 1 |
| ds-v3-t4096 | down | 4096 | 7168 | 2048 | bf16 | 1 |
| glm45-t32 | gate/up | 32 | 3072 | 5120 | bf16 | 4（`block_m` 需为 32） |
| glm45-t32 | down | 32 | 5120 | 1536 | bf16 | 4（同上） |

（`h_down` 的 n 是 `H`、k 是 `S`：`down = activated[T,S] @ w_down[H,S]^T`。）

### 0.3 实现算法解读（怎么算）

**源码结构化阅读**（`dense.py:4669-4831` 的 `_gemm_basic_kernel` + `dense.py:87-108` 的
`_add_bias` + `dense.py:175-239` 的 `_main`）：

1. **入口签名**：外层 8 个编译期参数（shape/布局/dtype/bias/out_dtype），内层 6 个可调
   参数（`block_m, block_n, block_k, num_stages, threads, split_k`）；`a_shape`/`b_shape`/
   `a_tile`/`b_tile` 由布局推导；`n_exact = n % block_n == 0`；`k_tiles = ceildiv(k, block_k)`；
   `k_slice = k_tiles // split_k`；`output_shape`/`output_dtype` 按 `split_k` 分支。
2. **Kernel 结构**：`T.Kernel(ceildiv(n, block_n), ceildiv(m, block_m), split_k, threads)` →
   三维 grid（`bx` = n 方向最快，`by` = m，`bz` = split-K）。
3. **计算原语**：`T.gemm(a_smem, b_smem, c_local, transpose_A=trans_a, transpose_B=trans_b,
   policy=GemmWarpPolicy.FullRow)` —— 单个矩阵乘累加；`_add_bias` 用
   `for i, j in T.Parallel(rows, cols): acc[i, j] += cast(bias[min(start + (i if by_row else j),
   n - 1)], acc.dtype)`；epilogue 用 `T.Parallel` 逐元素带守卫写。
4. **内存分配**：`a_smem`/`b_smem` = `alloc_shared`（SMEM，shape 随存储布局，即
   `(block_k, block_m)` 当 trans_a，`(block_n, block_k)` 当 trans_b）；`c_local` =
   `alloc_fragment((block_m, block_n), "float")`；`c_smem` = `alloc_shared((block_m, block_n),
   c_dtype)` 只在 `split_k == 1` 分配。
5. **循环与调度**：`for ki in T.Pipelined(k_slice, num_stages=num_stages)` —— K 方向软件流水；
   `T.use_swizzle(10, enable=True)` L2 光栅化；`make_swizzled_layout` 注解三个 SMEM buffer。
6. **host 侧逻辑**：`_main` 装配签名（`bias` 参数插在 operands 与 C 之间），
   `global_symbol = "{name}_main"` 或 `"{name}_bias_main"`；`@tilelang.jit(out_idx=[-1],
   pass_configs={"tl.disable_warp_specialized": True}, compile_flags=["-O3", "-DENABLE_BF16"])`。
7. **编译配置**：关闭 warp specialization（本 kernel 是非 warp-specialized 的通用路径）、
   `-O3`、`-DENABLE_BF16`。

**计算步骤分解表**（源码每条计算语句都归入某步骤）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式的部分 |
|---|---|---|---|---|
| 1 | 编译期校验与 shape 推导（`split_k` 整除 K tile 数；split-K 需精确 tile；`n_exact`） | 工厂参数 | `a_shape/b_shape/tile shape/output_shape` | 契约（非算术） |
| 2 | 逻辑块索引：`m_start = by*block_m`、`n_start = bx*block_n` | `by, bx` | 块内起点 | 输出 tile 定位 |
| 3 | K 段起点：`k_start = (bz*k_slice + ki)*block_k` | `bz, ki` | K 段起点 | 收缩轴分片 |
| 4 | A tile 装载（`cp.async`，按 `trans_a` 两种 region） | `A` | `a_smem` | `op(A)` 的元素搬入 |
| 5 | B tile 装载（`cp.async`，按 `trans_b` 两种 region；NN 且 N 尾块时掩码填 0） | `B` | `b_smem` | `op(B)` 的元素搬入 |
| 6 | 矩阵乘累加：`T.gemm(a_smem, b_smem, c_local, trans_a, trans_b)` | `a_smem, b_smem, c_local` | `c_local`（fp32） | `Σ_k op(A)[i,k]·op(B)[k,j]` |
| 7 | K 循环回到步骤 3（`T.Pipelined`，多缓冲） | — | — | 完整求和 |
| 8 | bias 加（仅 `has_bias and split_k == 1`）：`acc[i,j] += bias[n_start+j]` | `c_local, bias` | `c_local` | `+ bias[j]` |
| 9 | 累加器 → SMEM（`T.copy(c_local, c_smem)`，仅 `split_k == 1`） | `c_local` | `c_smem` | 输出中转（实现手段） |
| 10 | 带守卫的量化写回：`if in-range: C[i,j] = c_smem[i,j]` | `c_smem` | `C` | 输出（含舍入到 `c_dtype`） |
| 11 | split-K 分支：`T.copy(c_local, C[bz, m0:m0+bm, n0:n0+bn])` | `c_local` | `C`（fp32 部分和） | 分片部分和落盘 |

**数据流与内存访问模式**（源硬件视角）：

```
GM[A] --cp.async--> SMEM[a_smem:(bk,bm)|(bm,bk)] --ldmatrix/mma--> fragment[c_local (bm,bn) fp32]
GM[B] --cp.async--> SMEM[b_smem:(bn,bk)|(bk,bn)] --/                    （跨 K tile 保留累加）
                                                                   |
                            split_k==1:  c_local --T.copy--> SMEM[c_smem] --T.Parallel(带守卫)--> GM[C]
                            split_k>1 :  c_local --T.copy-------------------------------> GM[C[bz]]
```

`c_local` 是唯一的跨迭代状态；`a_smem`/`b_smem` 每 K tile 覆写（`num_stages` 版本轮转）；
`c_smem` 只在 `split_k == 1` 的 epilogue 存在一次。

**循环与并行结构**：grid = `ceil(n/bn) × ceil(m/bm) × split_k` 个 CUDA block；每 block
`threads` 个线程（默认 128，autotune 固定 128）；`T.gemm` 的 warp 划分 `FullRow`；
K 方向 `T.Pipelined(k_slice, num_stages)`；epilogue 一次 `T.Parallel(block_m, block_n)`
（split_k==1）或一次 `T.copy`（split_k>1）。

**host 侧逻辑**：仅编译期参数装配（见步骤 1）；无张量级预处理/后处理。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|---|---|---|---|---|
| 1 | SMEM tiling（`a_smem`/`b_smem`，每 K tile 一换） | 削减 GM 重复读，把 K 收缩变成片上循环 | 分块驻留 SMEM + 逐块搬入 | shared memory + 异步拷贝 | 硬件强相关 |
| 2 | tensor-core MAC（`T.gemm` → MMAMMA） | 把 2·m·n·k FLOPs 交给矩阵单元 | mma 指令 + 分形布局 | tensor core | 硬件强相关 |
| 3 | fp32 fragment 累加器跨 K tile 保留 | 免去每块清零与部分和落盘 | 寄存器 fragment，`clear_accum`/`T.clear` 只在首块 | 大寄存器堆 | 硬件强相关 |
| 4 | 软件流水 `T.Pipelined(num_stages)` | 访存与 MAC 重叠 | 多缓冲版本轮转 + 异步搬运 | cp.async / 异步 DMA 引擎 | 硬件强相关（多缓冲思想可移植） |
| 5 | 隐式多缓冲（`num_stages` 决定 SMEM 版本数） | 流水不 stall | 版本索引 `ki % num_stages` | — | 模式可移植 |
| 6 | swizzled SMEM 布局（`make_swizzled_layout`，三个 buffer 都注解） | 消 SMEM bank conflict，喂给 mma 的分形 | 索引重排 | SMEM bank 结构 | 硬件强相关 |
| 7 | L2 光栅化 `T.use_swizzle(10)` | 2D block 遍历按 panel 走，提升 L2 命中 | block 索引重排（panel 遍历） | GPU 2D grid + L2 | 硬件强相关 |
| 8 | split-K 分片（`bz` 轴 + fp32 partial + 归约 kernel） | 小 M（decode 段）时提高并行度 | 收缩轴切段，各自累加 | 无（纯算法层 + 额外 GM 往返） | 可移植 |
| 9 | M/N 尾块 epilogue 守卫 | 非整除 shape 正确性 | 逐元素条件写 | — | 可移植 |
| 10 | bias 在 fp32 累加器上融合 | 省一趟独立 bias kernel | fragment 逐元素加 | fragment 布局可逐元素访问 | 半可移植（依赖 fragment 布局） |
| 11 | 经 `c_smem` 中转输出（`c_local → SMEM → GM`） | coalesced 写、按 epilogue 守卫裁剪 | SMEM 中转 + 逐元素写 | SMEM 带宽 | 硬件强相关 |
| 12 | split-K 的精确 tile 前置校验 | 部分和写回不越界 | host 侧编译期 `raise` | — | 可移植（语义约束） |

> 识别不出来 ≠ 不存在。另外两项「未采用但源码存在」的手段：`_add_bias` 的
> `helper_layout` 分支调用 `tileops::dense_add_col_bias_64xN` 之类的外部 C++ helper
> （为 WGMMA 64 行 fragment 布局定制，见 `dense.py:87-108`），只被已丢弃的 SM90 变体使用，
> 本 kernel 的 `helper_layout=False` 默认路径不走它；`_load_a`/`native_acc_layout`
> 同样只服务 fp8 与 WGMMA 变体。这两项在本 kernel 中不构成优化手段，故不进入 §0.5 判定。

### 0.5 硬件耦合性分析与 NPU 适配决策

**判定问题**：实现算法与优化手段是硬件强相关吗？能用在 NPU 上吗？

判定依据来源：迁移映射 `references/migration-diagnostics.md`（原语映射）、平台规则
`.agents/platforms/toolchains/ascend/references/programming.md`、已核验条目
`.agents/platforms/toolchains/ascend/knowledge/gemm-tiles.md` 与 `PL-950-cube-tile`、
安装包内后端源码（`tilelang/ascend/op/gemm/gemm_mad.py`、`tilelang/ascend/op/copy.cc`、
`tilelang/src/tl_templates/ascend/gemm.h`）与本任务的设计期编译探针（§11.1）。

| # | 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|---|---|---|---|---|---|---|
| S1 | 计算语义（`C = op(A)·op(B) + bias`，fp32 累加，尾块排除，split-K 契约） | 语义 | 无 | — | **保留** | 语义层无条件保留；§0.1 为 golden 的唯一依据 |
| S2 | M/N 尾块 epilogue 守卫 | 算法 | 无 | — | **保留** | 改为「装载区段截断 + 存储区段截断」（§5.4），等价：越界行/列既不写也不影响有效行 |
| A1 | split-K 分片累加 | 算法 | 无（纯分块累加） | — | **保留** | 承接 §1.6.0 C1；NPU 侧落点见 R6（fp32 partial 落 GM + 伴随归约 kernel） |
| O1 | SMEM tiling | 优化 | shared memory | 有：L1（512 KiB，`alloc_l1`） | **等价替换** | `a1`/`b1` 为 L1 常驻操作数 tile；`GM→L1` 用 `T.copy`（MTE2，落 nd2nz 格式） |
| O2 | tensor-core mma | 优化 | tensor core（mma.sync） | 有：Cube MAD（`T.gemm` → `ascend_gemm_l1` / `ascend.mad`） | **等价替换** | `T.gemm(a1, b1, c0, transpose_B=True, clear_accum=…)`；后端 `Gemm::SelectInst` 对 ascend 恒选 `ascend.mad` |
| O3 | fp32 fragment 累加器跨 K 保留 | 优化 | 寄存器堆 | 有：L0C（256 KiB fp32，`alloc_l0c`） | **等价替换** | `c0 = T.alloc_l0c((bm,bn), "float32")` 跨 K tile 保留；`clear_accum=(step == 0)` 只首块清零 |
| O4 | `T.Pipelined(num_stages)` 软件流水 + 多缓冲 | 优化 | cp.async / 异步 DMA | 有：`T.Pipelined` + `T.annotate_buffer_versions`（MTE2 异步搬运 + 版本轮转） | **等价替换** | R2；探针实测 3/4 版本环形缓冲（`step % 3` / `step & 3`）与配对 set/wait 由 InsertSync 生成 |
| O5 | swizzled SMEM 布局（消 bank conflict、喂 mma 分形） | 优化 | SMEM bank 结构 | 有：编译器按 GEMM transpose flag 推导 NZ 分形布局 | **等价替换（机制不同）** | `GemmMAD.infer_layout` 对 L1 操作数注入 `make_ascend_major_k_layout`；GM→L1 的 nd2nz 直接产出该分形，故**不再需要** `make_swizzled_layout` 注解 |
| O6 | L2 光栅化 `T.use_swizzle(10)` | 优化 | GPU 2D grid + `rasterization2DRow` | **无** | **重新设计** | R4：ascend codegen 不实现 `threadblock_swizzle_pattern`（只有 `src/cuda/codegen/codegen_cuda.cc` 处理该 attr），且 `T.Kernel` 是一维网格；改为在 1-D `bid` 上自己定义 m-fastest 遍历序并给量化依据 |
| O7 | `c_local → c_smem → GM` 中转输出 | 优化 | SMEM 带宽 | 有，但**不必要** | **舍弃** | fixpipe 可直接 `L0C → GM`（`asc_copy_l0c2gm`，探针 D1 生成 `asc_quant_mode=16`）；再绕一趟 UB/SMEM 只是多一次片上往返。理由：NPU 的 L0C→GM 天然按 nz→nd 落盘，不存在源实现要修的 uncoalesced 写 |
| O8 | bias 在 fp32 fragment 上逐元素加 | 优化 | fragment 可逐元素访问（CUDA thread↔元素映射） | 部分：L0C **不可**逐元素访问，Vector 单元可 | **重新设计** | R5：Cube+Vector epilogue（`dual_copy` L0C→UB fp32 → `SimtVF` 内加 bias 并 cast → `dual_copy` UB→GM）；仅本 op 未使用的 `has_bias` 分支 |
| O9 | trans_a/trans_b 四布局由 `T.gemm` 的 transpose flag 协调 | 优化 | mma 指令的 A/B 布局位 | **部分**：ascend L1-GEMM 只支持 NT | **重新设计** | R1：四种存储布局统一在**装载**时用 nd2nz/dn2nz 归一成 `(M,K)`/`(N,K)`，GEMM 恒为 `transpose_A=False, transpose_B=True`。依据（硬约束）：`gemm_mad.py` `assert not self.trans_A and self.trans_B, "Ascend L1 GEMM currently only supports trans_A=False, trans_B=True (NT)."`；`gemm.h` 模板只有 `TRANS_B` 一个 flag |
| O10 | `threads` + `GemmWarpPolicy.FullRow` | 优化 | warp（32 线程）抽象 | **无** | **舍弃** | 平台 `T.Kernel(N)` 无 `threads` 参数（规则明载），AIC 内的 MAD 调度由 `ascend_gemm_l1` 模板内部完成；`policy` 在 ascend 侧被忽略（`ComputeWarpPartition` 恒返回 1/1） |
| O11 | `pass_configs={"tl.disable_warp_specialized": True}` + `-DENABLE_BF16` | 优化 | CUDA warp specialization / 编译宏 | 无 | **舍弃** | ascend 后端无 warp specialization 概念；bf16 由 dtype 路径直接支持（探针 D1 的 `bfloat16_t` 模板参数）。**保留** `compile_flags=["-O3"]` |
| O12 | `out_idx=[-1]` JIT 输出分配 + `_main` 签名装配 | 优化 | 无 | 有 | **保留** | `tilelang.compile(..., out_idx=-1)`；有/无 bias 两种签名在 `@T.prim_func` 层用 Python 分支生成 |
| O13 | split-K 精确 tile 前置校验 | 优化 | 无 | — | **保留** | 保留 `raise`（语义约束），并追加 `block_m` 夹取与 `m % block_m` 复核（§5.2、§9.2 E6） |
| D1 | SM90 变体（TMA / WGMMA / warp-specialized / grouped GEMM / 融合 epilogue 模板） | 实现 | SM90 专有 | 无 | **舍弃（Stage 0 已丢弃）** | 见 `_shared_expert_mlp_fwd_kernels.py` 模块 docstring 的 curation 说明；本 op 在 NPU 侧只走 non-SM90 的 `_gemm_basic_kernel` 路径（由脚手架决定），本文不重新设计它们 |

### 0.6 NPU 算法重设计

对 §0.5 判为「重新设计」的条目逐项给出 NPU 新算法。**调研结论（§1.6.0）是本节的输入**：
调研选定算法族 = 「NT 归一的分块流式 GEMM + 可选 split-K 分片」（= 源算法的同族，未替换为
Strassen 类或两遍物化类），因此 §0.4 的优化意图**逐项承接**（每项注明由什么结构承接），
无静默丢弃。

**重设计项 R1：四布局的 NT 装载归一（承接 O9）**

- **源方案**：SMEM tile 按存储布局分配（`a_tile = (bk,bm) if trans_a` 等），由 `T.gemm` 的
  `transpose_A/transpose_B` flag 在 mma 层与逻辑 `(M,K)×(K,N)` 收缩对齐。性能意图 =
  **不为布局差异复制/转置数据**，装载即用。
- **NPU 新算法**：GEMM 恒用后端唯一的 NT 形式（`a1:(bm,bk)`、`b1:(bn,bk)`、
  `transpose_B=True`）；布局差异在 **GM→L1 搬运**上消化：
  - `trans_a=False` → `T.copy(A[m0:m0+am, k0:k0+ak], a1[:am,:ak])`（nd2nz）；
  - `trans_a=True` → `T.copy(A[k0:k0+ak, m0:m0+am], a1[:am,:ak], transpose=True)`（dn2nz）；
  - `trans_b=True` → 直接 region 拷贝（nd2nz）；`trans_b=False` → 同样 `transpose=True`。
  这正是源仓自己在 NN 情形给出的建议（`gemm.h`：`for NN matmul use dn2nz to transpose B
  during GM→L1`）。
- **语义保持论证**：装载归一**不改变任何算术**——被乘元素的集合不变、K 分块顺序不变、
  累加器顺序不变，因此是精确等价（不是「容差内」）。机器验证见 §1.6.1 A3
  （对同一 K tile 分解逐块比对 fp32 位模式：`A3_bitwise_equal = pass`）。
  边界语义：转置装载的越界尾部由 `AscendInsertOOBPadding` 裁剪（转置副本用完整 extent
  装载是本平台已记录的写法，见 `examples/mla_dense_bwd/_bwd_kernel/DESIGN.md` §5.4）；
  越界行只落在被丢弃的死行上，与源实现一致。探针 D3/D4/D5 分别验证 TN / NN / TT 三种
  装载路径（`dn2nz` 计数 1/1/2）均编译通过。

**重设计项 R2：L1 常驻 + 多版本环形缓冲（承接 O1/O4/O5）**

- **源方案**：SMEM 双/多缓冲 + `cp.async` 预取，由 `num_stages` 决定 SMEM 版本数，
  手动 swizzle 布局喂 mma。
- **NPU 新算法**：`a1`/`b1` 为 L1 buffer（`alloc_l1`），用
  `T.annotate_buffer_versions({a1: num_stages, b1: num_stages})` 显式声明 `num_stages` 个版本，
  `T.Pipelined(k_slice, num_stages=num_stages)` 驱动轮转；L1→L0A/L0B 的分块与 L0 双缓冲
  **由后端 `ascend_gemm_l1` 模板内部完成**（模板用 M_MTE1 事件 0/1 做 L0 双缓冲，
  `TILE_K_SUB` 由 `_compute_tile_k_sub` 按 L0 容量 64 KiB / 2 阶段自动选，本设计为 64）。
  NZ 分形布局由编译器 `infer_layout` 注入，无需人工 swizzle。
- **语义保持论证**：缓冲版本只影响「同一时刻驻留几份输入」，不改变任何一次 MAD 的
  操作数或累加顺序——精确等价。已核验先例：`PL-950-gemm-tiles` 用同一
  `annotate_buffer_versions + T.Pipelined` 组合在 (16,16,64)/(17,19,35) 上通过精度检查；
  本任务探针 D1/D2/D11 生成 `step % 3` / `step & 3` 环形与 4 组配对事件。

**重设计项 R3：L0C fp32 累加 + fixpipe 硬件量化落盘（承接 O3/O7）**

- **源方案**：累加在 fragment（寄存器），epilogue 经 `c_smem` 中转后逐元素带守卫写 GM，
  在写的时候 cast 到 `c_dtype`。
- **NPU 新算法**：`c0 = alloc_l0c((bm,bn), "float32")` 跨 K tile 累加；`split_k == 1` 时
  一条 `T.copy(c0[:am,:an], C[m0:m0+am, n0:n0+an])` 直接落 GM，**fp32→c_dtype 的舍入由
  fixpipe 的 quant_pre 完成**：后端 `GetCCQuantPre` 把 `fp32→bfloat16` 映射为
  `asc_quant_mode=16`（F322BF16）、`fp32→float16` 映射为 `1`（F322F16）、同 dtype 为 `0`
  （NoQuant）；探针 D1/D2/D12 分别生成 16 / 1 / 0。
- **语义保持论证**：语义要求「fp32 累加，最后一次舍入到 `c_dtype`」。源路径是
  `fp32 → GM fp32 → cast`，本设计是 `fp32 → cast`；两者送给同一个舍入函数的都是同一批
  fp32 值（fp32 的 GM 往返是恒等映射），故**位级等价**——§1.6.1 A1 机器验证
  （`A1_store_bitwise_equal = pass`，含非有限值模式比对）。
  边界语义：M/N 尾块用 `c0[:am,:an]` 区段截断（`CheckCompactL0Region` 要求尾部维零起点，
  区段起点为 0，满足）；`k == 0` 时循环零步，`c0` 未初始化 → 由 `clear_accum` 语义
  决定（源实现在 `k == 0` 时只写 bias 或零；本设计在 `k == 0` 时把 `c0` 显式 `T.clear`
  后在 epilogue 写零/bias，见 §3.3 的 `if k_tiles == 0` 分支）。

**重设计项 R4：块遍历顺序（承接 O6）**

- **源方案**：`T.use_swizzle(10)` 让 CUDA block 索引按 10 行 panel 遍历，改善 L2 命中；
  grid 本身是 `(n, m, split_k)` 三维、n 最快。
- **NPU 新算法**：grid 是 1-D，`use_swizzle` 在本后端不生效（O6），因此**遍历序显式写进
  `bid` 分解**：`bx = bid // (mt·split_k)`（n tile，慢轴）、`by = bid % (mt·split_k) // split_k`
  （m tile，**快轴**）、`bz = bid % split_k`（split 切片，最快）。即同一波内的 block 共享
  同一个 B tile（权重 tile），A 面板在 L2 中跨 n panel 复用。
- **量化依据**（这是相对源实现的**有依据偏离**，不是照抄）：L2 = 128 MiB（型号档案
  `l2_size = 134217728`，配置原值）。逐 workload 的 A 面板与 B tile 足迹：

  | workload | A 全量（m·k·2B） | B tile（bn·k·2B） | m-fastest 下每 n panel 触达 | n-fastest 下每 m panel 触达 |
  |---|---|---|---|---|
  | ds-v3-t4096 gate/up | 58.7 MB | 3.67 MB | 58.7 + 3.67 = 62.4 MB < 128 MB → A 常驻 | B 3.67 MB × 144 tile × 32 m-panel |
  | kimi-k2-t512 gate/up | 7.3 MB | 3.67 MB | 11.0 MB → 双常驻 | B × 144 × 4 |
  | ds-v3-t32 gate/up | 0.46 MB | 3.67 MB | 4.1 MB | B × 16 × 1 |

  m-fastest 让「A 面板 ≤ L2」成立（最大 58.7 MB），B 的每个 tile 在其 mt 个 m-block 内被
  同波/相邻波消费；n-fastest 则让 B 每个 m panel 整体重读一次（t4096 时约 32 倍权重流量）。
  结论：**m-fastest**。该偏离的代价/收益量级（DRAM 流量 4–32 倍）远大于任何未实测常数
  的不确定性，不触发实验裁决三件套（§1.6.3 末段说明）。
- **语义保持论证**：块的执行顺序不影响任何输出元素的值（每块写互不重叠的 C tile；
  split-K 的 partial 写各自独立切片），精确等价。

**重设计项 R5：bias 的 Cube+Vector epilogue（承接 O8）**

- **源方案**：在 fp32 fragment 上 `T.Parallel` 逐元素加 `bias[n_start+j]`（越界 lane 读
  `bias[n-1]`，随后被守卫丢弃）。
- **NPU 新算法**（仅 `has_bias and split_k == 1`）：`dual_copy(c0 → cv_ub:(bm/2,bn) fp32)`
  （M 分片到两个 AIV）→ `with T.SimtVF(threads=128): for i,j in T.Parallel(bm/2,bn):
  co[i,j] = cast(cv[i,j] + cast(bv[j], "float32"), dtype)` → `dual_copy(co → GM tile)`；
  `bv` 是 `(bn,)` 的 bias 切片（`T.copy(bias[n0:n0+bn], bv)`）。
- **语义保持论证**：`C[i,j] = cast_f16(acc_fp32[i,j] + cast_f32(bias[j]))`，与源的
  `acc_fp32 += cast_f32(bias)` 后 `cast_f16` 是同一个 fp32 加法 + 同一个最终舍入 → 位级等价
  （§1.6.1 A1 的舍入论证同样覆盖；A2 覆盖 split-K 组合；bias 本身不引入新的舍入点）。
  边界语义：源的 `min(start+j, n-1)` 夹取只为让越界 lane 有一个可加的标量、随后被守卫丢弃；
  本设计只装载 `[n0, n0+bn)` 的有效区间（N 尾块用 `an` 截断），不存在越界 lane，故不需要
  该夹取（等价：夹取与否都不影响被写的元素）。
  平台证据：探针 D10 生成 `__mix__(1,2)` kernel、`ascopy_l0c2ub(..., asc_dual_dst_mode(1))`
  与 `__float22bfloat162_rn` 的 SIMT 向量 cast。

**重设计项 R6：split-K 的 NPU 落点与可用性判定（承接 A1）**

- **源方案**：`bz` 轴分片，每片写 `C[bz]` fp32 部分和；精确 tile 前置；由调用方决定
  `split_k`（与伴随归约 kernel 配合）。
- **NPU 新算法**：同一契约。1-D grid 携带 `split_k`（`bz` 为最快轴），片内 K 段
  `k0 = (bz·k_slice + step)·bk`，`c0` 落 `C[bz, m0:m0+am, n0:n0+an]`（fp32，无量化）。
- **语义保持论证**：`C = Σ_s partial_s`，数学等价；fp32 舍入分组变化的影响由 §1.6.1 A2
  机器验证（K=7168、split=4、宽动态范围：目标 dtype 输出 0 ulp 差、violation rate 0；
  fp32 层重结合足迹 ≤ 4864 fp32 ulp ≈ 6e-4 相对，远小于 §8.2 的 `rtol = 1e-2`）。
- **可用性修正（设计期发现）**：源前置条件 `m % block_m == 0` 与后端
  `block_m % 16 == 0`（`gemm.h` 的 `static_assert(M % FRAC_N == 0)`）合起来意味着
  **`m < 16` 时 split-K 不可用**。而调用方 `_split_k(T ≤ 32)` 对 `T = 1`（kimi-k2-t1）会返回 4，
  于是源实现的前置校验会抛 `ValueError`。本设计给出判定规则（§5.2）：`split_k > 1` 时
  `block_m = min(block_m, m)` 并要求 `m % block_m == 0`；**`m < 16` 一律回落 `split_k = 1`**
  （此时 mt=1，逻辑核数 = nt = 144/28 已 ≥ 物理核数，见 §5.5，split 无收益且只增部分和流量）。

**无「重新设计」的条目**：S1/S2/A1/O1–O5/O9（部分）/O12/O13 已在上表逐项处置；
O7/O10/O11/D1 为舍弃（理由见 §0.5）。

### 0.7 标杆实现

- 源算子：`/workspace/TileOPs/src/tileops/_kernels/gemm/dense.py::_gemm_basic_kernel`
  （GPU TileLang，CUDA target）；同仓逐字抽取件
  `examples/TileOPs/tileops/kernels/moe/shared_expert_mlp/_shared_expert_mlp_fwd_kernels.py`。
- 参考实现 / golden 依据：本文 §0.1 的语义（`C = op(A)·op(B) + bias`，fp32 累加、末次舍入、
  尾块排除、split-K 契约）。源仓没有为 `_gemm_basic_kernel` 单独提供 torch 参考实现，因此
  golden 直接按 §0.1 语义用 `torch.matmul` 写（§8.1），**不复刻** §0.6 的分块算法
  （不预先切块、不写部分和、不做硬件量化）。
- 上游 Op 级参考：`SharedExpertMLPWorkload.ref_program`（本任务的上层 op 参考，用来说明
  `A/B` 的实际布局：两个 GEMM 都 `trans_b=True`、`trans_a=False`，bias 不用）。

---
## 1. 概述

### 1.1 算子名称

`_gemm_basic_kernel`（dense GEMM，NT 归一 + L1 多版本流水 + L0C fp32 累加 + 量化 fixpipe 落盘；
可选 split-K 分片与 Cube+Vector bias epilogue）。

### 1.2 功能描述

计算 `C = op(A)·op(B) [+ bias]`，fp16/bf16 输入、fp32 累加；`(trans_a, trans_b)` 四种存储布局
由装载路径归一；`split_k > 1` 时输出 `[split_k, m, n]` fp32 部分和供伴随归约 kernel 消费。

### 1.3 数学公式

$$
C[i,j] \;=\; \sum_{t=0}^{K-1} \hat A[i,t]\cdot \hat B[j,t] \;+\; \big[\,\text{has\_bias}\,\big]\,bias[j],
\qquad
\hat A = \begin{cases} A & \text{trans\_a}=\text{False}\\ A^{\mathsf T} & \text{trans\_a}=\text{True}\end{cases},
\quad
\hat B = \begin{cases} B & \text{trans\_b}=\text{False}\\ B^{\mathsf T} & \text{trans\_b}=\text{True}\end{cases}
$$

其中累加在 fp32 中进行；`split_k = 1` 时 `C` 在存储时舍入到 `out_dtype or dtype`，
`split_k = s > 1` 时 `C` 的第 0 维是 K 切片索引、每个切片保存该段的 fp32 部分和：

$$
C[s,m,n] \;=\; \sum_{t \in \text{slice}_s} \hat A[m,t]\cdot \hat B[n,t], \qquad
C[m,n] \;=\; \text{cast}_{c\_dtype}\Big(\sum_{s} C[s,m,n]\Big) \quad \text{(reduce kernel)}
$$

第二式由伴随的 `splitk_reduce_kernel` 完成（另一个设计任务；它同时承担可选 bias 与
`silu_and_mul` epilogue）。

### 1.4 算法描述

本节描述**迁移决策后（§0.5/§0.6）的 NPU 侧算法**，与源算法的结构差异逐条注明来源：

1. **布局归一（R1，相对源实现的差异）**：源靠 `T.gemm` 的 `transpose_A/transpose_B` 接受四种
   布局；NPU 后端只支持 NT，因此 A、B 在 **GM→L1 装载**时分别被归一成 `(block_m, block_k)` 与
   `(block_n, block_k)` 的 NZ 分形 tile（`nd2nz` / `dn2nz`），GEMM 恒为
   `transpose_A=False, transpose_B=True`。
2. **分块流式收缩（保留源算法族，§1.6.0 C0 + R2）**：每个逻辑块负责一个 `(block_m, block_n)`
   输出 tile，沿 K 方向按 `block_k` 迭代；`a1`/`b1` 取 `num_stages` 个 L1 版本做环形缓冲，
   `T.Pipelined` 驱动 MTE2 预取与 Cube MAD 重叠；L1→L0A/L0B 的分块与 L0 双缓冲由
   `ascend_gemm_l1` 模板内部完成（`TILE_K_SUB = 64`）。
3. **fp32 累加与量化落盘（R3）**：`c0` 在 L0C 中跨 K tile 累加（`clear_accum=(step == 0)`），
   尾块用区段截断，`split_k == 1` 时一条 fixpipe 存储同时完成 `fp32 → c_dtype` 的硬件舍入
   （quant_pre 16 = F322BF16 / 1 = F322F16）。
4. **split-K 分片（§1.6.0 C1 / R6）**：`split_k > 1` 时把 K tile 序列切成 `split_k` 段，每段独立
   累加并写 `C[bz]` fp32 部分和；仅在 `block_m | m` 且 `m ≥ 16` 时可用（否则回落 `split_k = 1`）。
5. **bias 的 Cube+Vector epilogue（R5，仅 `has_bias and split_k == 1`）**：`dual_copy` 把
   `c0` 按 M 分片进两个 AIV 的 UB（fp32），`SimtVF` 内加 bias 并 cast，再 `dual_copy` 回 GM。
6. **块遍历序（R4）**：1-D `bid` 显式分解为 n tile（慢）/m tile（快）/split 切片（最快），
   让 A 面板驻留 L2（依据见 §0.6 R4 与 §1.6.3 D4）。

与源算法结构不同的点仅 1、3、6（装载归一、落盘路径、遍历序）与 4 的可用性约束；
收缩分解、fp32 累加、split-K 契约、尾块语义均与源算法同构。

### 1.5 数据流图

```
A(GM) ─MTE2 nd2nz/dn2nz─▶ a1 (L1, num_stages 版本) ─MTE1 分块─▶ L0A ┐
B(GM) ─MTE2 nd2nz/dn2nz─▶ b1 (L1, num_stages 版本) ─MTE1 分块─▶ L0B ┴─Cube MAD─▶ c0 (L0C, fp32, 跨 K 迭代)
                                                                          │
   split_k == 1, has_bias == False ──fixpipe(quant)──────────────────────▶ C (GM, c_dtype)
   split_k == 1, has_bias == True  ──dual_copy──▶ cv/co (UB, 2×AIV) ──SimtVF(+bias, cast)──▶ C (GM)
   split_k > 1                     ──fixpipe(no quant)───────────────────▶ C[bz] (GM, fp32)
```

### 1.6 算法调研与优化分析

> 调研对象 = 同一数学语义（dense 双线性收缩 + 可选列 bias）的算法族；源算法只是基线候选之一。

#### 1.6.0 算法调研（Algorithm Research）

**R1 等价化简公式候选**（基线在表内；正式等价论证与收益量化在 §1.6.1）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 是否纳入 R3 对比 |
|---|---|---|---|---|---|
| C0 | **基线：源 `_gemm_basic_kernel`** | 分块流式 `Σ_k`，fp32 累加，`T.Pipelined` + `T.gemm`，可选 split-K | — | — | ✅ 基线 |
| C1 | split-K 分片累加（源的 `bz` 轴） | `C = Σ_s (Σ_{t∈slice_s} Â·B̂)`，部分和 fp32 落 GM 后顺序求和 | 数学等价；**fp32 重结合**（舍入分组变化） | 并行度 ×split（decode 段填核）；代价 = 2·split·m·n·4 B 部分和往返 | ✅ |
| C2 | persistent 单波次（grid = 物理核数，核内遍历 tile） | 同上，块内循环处理多个输出 tile | 精确等价（只改任务映射） | 摊薄 block 派发开销；代价 = 负载不均风险 | ✅ |
| C3 | Strassen / Winograd 类亚三次算法 | 7 次半尺寸乘法代替 8 次（Strassen） | **容差内**不稳：中间加减在 fp16 输入域放大抵消误差，且需物化 M1–M7 | FLOPs −12.5%（Strassen 一阶）；访存与中间缓冲显著变差 | ❌ 见 R4 弃选依据 |
| C4 | 两遍物化：GEMM 写 fp32 GM → 独立 cast kernel | `C_tmp(fp32) = Â·B̂; C = cast(C_tmp)` | 精确等价（fp32 往返恒等） | **负收益**：多 m·n·4 写 + m·n·4 读 | ✅（作为 A1 的对照基线） |
| C5 | bias 并入收缩（扩 K 一维：追加一列全 1 与 bias 列） | `C = [Â, 1]·[B̂, bias]^T` | 数学等价 | 负收益：需构造 K+1 的填充操作数、破坏「`block_k` 整除 `k`」，且多一次 operand 物化 | ❌ 见 R4 |
| C6 | 收缩轴换序（`Σ_k` 变 `Σ_j` 等） | — | 不适用：收缩轴由语义固定 | — | ❌ 非候选 |

**R2 在线算法**：**有**，且基线本身已是在线形态——K 方向的 `T.Pipelined` 循环就是
「分块 running sum」：中间状态只有 `block_m·block_n` 的 fp32 累加器与 `num_stages` 份输入 tile，
**与 K 无关**（O(tile) 而非 O(K)），不存在需要 rescale 的第二统计量（无 softmax/max 类全局量）。
结构判据核对：可分解为 running 统计量（和）的算子存在在线变体 → 本算子已是单遍在线累加，
无「两遍 → 单遍」的进一步空间。**跨 launch 的 split-K + 归约不属本口径**（R2 定义要求同一
kernel launch 内），它是 R1 的 C1 候选。结论：无需在线化改造，收益维度（扫描遍数/中间缓冲）
已达到下界 1 遍 / O(tile)。

**R3 复杂度对比**（四口径；口径说明：FLOPs 只计 MAC（乘加=2），不含索引与搬运；
访存按「不可消除 I/O」= 读 A 一次 + 读 B 一次 + 写 C 一次，另列块级重复读放大；
扫描遍数按每元素 GM 往返次数；中间缓冲峰值指片上驻留或需额外 GM 物化的量）：

| 算法候选 | FLOPs | 访存量 (Bytes) | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---|---|---|---|---|---|
| C0 基线（split_k=1） | `2mnk`：decode `1.88 G` / prefill `240.5 G` | ideal `(mk+nk)·2 + mn·2` = decode **59.4 MB** / prefill **151.0 MB**；块级放大 `mt·nk + nt·mk`（无 L2 复用时 t4096 gate/up 达 1.9 GB） | A/B 各 1 遍（块内）；C 1 遍 | `bm·bn·4`（L0C）+ `2·(bm+bn)·bk·2`（L1 双版本起） = 128 KB + 96 KB | `mt·nt` 块（decode 16–28，prefill 512–896）；无跨核同步 |
| C1 split-K（split=4） | 同上（不变） | `+ 2·split·m·n·4` = decode gate/up **+4.2 MB**（占 ideal 7.1%）；kimi-k2-t1 gate/up +1.2 MB（占 0.2%） | 同上 + partial 1 写 1 读 | 同上 + GM 部分和张量 `split·m·n·4`（decode 4.2 MB） | `mt·nt·split`（decode 64–112，↑4×）；跨 kernel 归约（无 barrier） |
| C2 persistent（grid=28） | 同上 | 同 C0（遍历序可保持） | 同上 | 同 C0 | `28` 块 × 核内 `ceil(mt·nt/28)` 次 tile 循环；无跨核代价，但块间负载需静态均分 |
| C3 Strassen 一阶 | `≈0.875·2mnk` | 更差：7 个中间矩阵物化 + 输入切半重排 | ≥2（中间物化后再组合） | `≈7·(m/2)(n/2)·4` = 1.75·m·n·4（远超 L0C/UB） | 可并行但需额外同步点；fp16 数值不稳 |
| C4 两遍物化 | 同上 | `+ 2·m·n·4`（写+读 fp32 C_tmp）：decode +1.0 MB（+1.7%） | C 2 遍 | `+ m·n·4` GM 张量 | 同 C0 + 一次额外 launch |

**R4 硬件亲和性评估**（逐候选对照检查清单；负向淘汰附依据）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---|---|---|---|---|---|---|
| C0 基线（NPU 化后） | ✅ MAC 全落 Cube MAD（`ascend_gemm_l1` → `ascend.mad`）；无标量热点 | ✅ L0C 128 KB / 256 KB、L1 144 KB / 512 KB、L0A 32 KB / L0B 64 KB（§4.5） | ✅ `bm/bn/bk` 均为 16 的倍数（`gemm.h` `static_assert(M%16==0, N%16==0)`）；`k % bk == 0` 前置（`_compute_tile_k_sub` 断言） | ✅ m/n/k 编译期常量，循环界静态 | ✅ `T.Pipelined` + 显式 buffer versions 已核验（`PL-950-gemm-tiles`）；fixpipe 融合量化 | ✅ **采纳** |
| C1 split-K | ✅ 同一 Cube 路径 | ✅ 部分和落 GM（不进片上） | ✅ 前置：`block_m` 整除 `m`、`block_n` 整除 `n`、`block_k` 整除 `k`、`split_k` 整除 `k_tiles` | ✅（split 为编译期常量） | ⚠️ 跨 kernel 归约需第二个 launch（无 barrier/CV 融合需求） | ✅ **采纳（仅 decode 段，判定规则见 §5.2）** |
| C2 persistent | ✅ 同一 Cube 路径 | ✅ | ✅ | ✅ `T.Persistent` 需 tile 数与 28 的静态关系 | ⚠️ 平台知识库无 `T.Persistent` 的正确性条目（仅安装包 README 附带的 persistent GEMM 示例，本仓无该文件、知识库无对应条目），未核验 | ⏸ **备选**（Stage 4 A/B；主选不启用，理由见 §5.5） |
| C3 Strassen | ❌ 额外加减需 Vector 单元介入，且中间矩阵要物化/搬运，主导算力不再全在 Cube | ❌ 中间 `≈7·(m/2)(n/2)` 远超 L0C，只能落 GM | ❌ 需偶数维切分，`block_k=64` 的 K 分块被打断 | ✅ | ❌ 多阶段 + 物化 | ❌ **淘汰**（依据：容量 1.75·m·n·4 B 溢出 L0C 256 KiB 一个数量级；fp16 抵消误差无本平台实测，标「未文档化假设：Strassen 在 fp16 输入上误差放大，无本平台证据」） |
| C4 两遍物化 | ✅（cast 落 Vector） | ⚠️ +m·n·4 GM | ✅ | ✅ | ❌ 多一遍全 C 流量 | ❌ **淘汰**（依据：C 流量翻倍，decode 段 +1.7%、prefill 段 `m·n·4` 与 ideal 同量级；且 R3 已证明一次舍入即可位级等价，见 §1.6.1 A1） |
| C5 bias 并入 K | ⚠️ 运算退化（bias 列只做 1·bias 的无效 MAC） | ⚠️ 需 `(bk+1)` 的填充操作数 | ❌ 破坏「`block_k` 整除 `k`」与「`split_k` 整除 `k_tiles`」 | ❌ 需运行时构造填充张量 | ❌ | ❌ **淘汰**（依据：语义前置「`block_k` 整除 `k`」被破坏，源实现同样不允许 K 尾块） |

**调研结论**：选定算法族 = **C0（分块流式 NT GEMM，fp32 累加，fixpipe 量化落盘）+ 条件下的
C1（split-K 分片）**，并对 §0.6 的重设计项 R1–R6 落地。关键依据：R3 中 C0 的访存已达
「读 A/B 各一遍 + 写 C 一遍」下界、中间缓冲 O(tile)，C1 是唯一能在 decode 段提高核占用
（16 → 64 逻辑块）且代价 < 7.1% 流量的候选（R3/R4 表）；C3/C4/C5 在 R4 上被容量、流量或
整除前置淘汰。与基线（源算法）的**结构差异**一句话：收缩分解与 split-K 契约沿用源算法，
但布局归一、多版本 L1 缓冲、L0C/量化落盘与遍历序按 NPU 能力重设计（R1–R4）。
**未选「无更优替代」路径**：调研范围 = 平台 `knowledge/algorithm-candidates.md`（GEMM 行：
「显式 L1/L0 操作数、输入布局转换和 K 分块累加」）、`PL-950-gemm-tiles`、`examples/gemm/`
四份实现、安装包 README 的 persistent GEMM 示例、源仓 `dense.py` 的 11 个 GEMM 变体
（这是「该算子族存在什么算法」的直接证据）、以及 C1–C5 五类结构候选的自查。

**设计期估算（D-2 roofline 下界行）**

以 decode 代表 `ds-v3-t32 gate/up`（m=32, n=4096, k=7168, bf16, split=4）与 prefill 代表
`ds-v3-t4096 gate/up`（m=4096, n=4096, k=7168, bf16, split=1）各算一次：

| 项 | decode（bm=32, bn=256, bk=64, split=4） | prefill（bm=128, bn=256, bk=64, split=1） |
|---|---|---|
| 流量项 | 59.4 MB（A 0.46 + B 58.7 + C 0.26）+ partial 4.2 MB | 151.0 MB（A 58.7 + B 58.7 + C 33.6） |
| 每块每 K 迭代的复用（FLOP/B） | `2·32·256·64 / ((32+256)·64·2) = 28.4` | `2·128·256·64 / ((128+256)·64·2) = 85.3` |
| 全算子算术强度 AI | `1.88 GFLOP / 63.6 MB ≈ 29.6` | `240.5 GFLOP / 151.0 MB ≈ 1593` |
| 发射项 | 64 逻辑块（mt=1 × nt=16 × split=4） | 512 逻辑块（mt=32 × nt=16） |
| 容量项 | L0C 32 KB/256 KB、L1 110.6 KB/512 KB → 不构成下界 | L0C 128 KB/256 KB、L1 147.5 KB/512 KB → 不构成下界 |

`估算下界: 待核验（缺少 ascend-950pr-9579 的 HBM 有效带宽与 Cube 峰值吞吐实测常数——
平台 cost_measurements 为空、knowledge 下无 constants.md 常数条目，型号档案明确拒绝把
INI 频率字段换算成带宽/算力）。流量项已算出：decode 63.6 MB、prefill 151.0 MB，下界形如
max(63.6 MB / B_eff, FLOPs / P_cube, 发射项)。`

**由估算得到的、与常数无关的结论**（这决定了本设计的重点）：
AI = 29.6（decode，含 split 流量）≪ 典型机器平衡点（数十~数百 FLOP/B）→ **decode 段是
权重带宽受限**（`3SH` 权重流 792.8 MB 主导 kimi-k2-t1 的 792.9 MB 总量），因此 decode 段的
设计重点是「权重只读一遍 + tile 足够大以摊薄搬运」；prefill 段 AI = 1593 → **算力受限**，
重点是多版本流水与 Cube 占用。这正是 §5.6 配置表按 T 分档的依据，也是不把大 shape 的
MFU 结论套到 T=1/32 的原因。

#### 1.6.1 数学等价优化（公式级）

分析对象为 §1.6.0 选定的 C0 + C1。逐项评估：

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算 |
|---|---|---|---|---|---|
| A1 | 舍入点融合（量化落盘） | `C_tmp(fp32) = Σ Â·B̂; C = cast(C_tmp)`（两遍：GEMM 落 fp32 GM，再 cast） | `C = cast_fixpipe(Σ Â·B̂)`（一次落盘，硬件 quant_pre 完成舍入） | **位级等价**：中间的 fp32 GM 往返是恒等映射（存读同 dtype），两条路径送入同一舍入函数的 fp32 值完全相同；两条路径都只含 1 个舍入点（fp32→c_dtype）。fp16/bf16 舍入影响：无（同一舍入、同一操作数） | 省 `m·n·4` B 写 + `m·n·4` B 读（decode gate/up 共省 1.0 MB，prefill gate/up 共省 134.2 MB）与一次 launch |
| A2 | 分片累加（split-K） | `acc = Σ_{t=0}^{kt-1} p_t`（单累加器顺序加） | `part_s = Σ_{t∈slice_s} p_t; acc = Σ_{s=0}^{split-1} part_s` | 数学等价（加法结合律）；**fp32 重结合**：舍入分组改变。机器验证（下表）在 K=7168/split=4/宽动态范围上测得目标 dtype 输出 0 ulp 差、violation rate 0；fp32 层足迹 ≤ 4864 fp32 ulp（≈6e-4 相对）≪ `rtol = 1e-2` | 并行度 16 → 64 逻辑块（decode gate/up，1.0 → 2.3 波）；代价 2·split·m·n·4 = 4.2 MB（7.1% 流量） |
| A3 | NT 装载归一 | `C = op(A)·op(B)`，由 mma 的 transpose 位在消费时对齐 | 装载时把操作数物化为 `(M,K)`/`(N,K)`，GEMM 恒 NT | **精确等价**（零舍入）：被乘元素集合、K 分块顺序、累加顺序全部不变，只有内存里的排布换序（dn2nz 读的是同一批元素）。机器验证：逐 K tile 比对 fp32 位模式 = 全等 | 无 FLOPs 变化；收益是「唯一可用的 Cube 形态」（后端硬约束），并免掉为 4 种布局各写一套 kernel |

**等价性机器验证（D-1）**：脚本 `examples/shared_expert_mlp/_gemm_basic_kernel/verify_equiv.py`
（torch CPU，fp64 参照 + 按硬件实际精度回放舍入路径：fp32 累加器、目标 dtype 存储）。
运行方式 `python verify_equiv.py`，判定标准：A1 位级相等 + 非有限值模式相等；A3 位级相等；
A2 目标 dtype 最大 ulp 差 ≤ 1 且 `>1ulp` 违反率 ≤ 1e-3 且相对 fp64 参照不劣于基线。

| 用例 | shape | dtype | split_k | A1 位级 | A2 fp32 ulp 足迹 | A2 改动元素比例 | A2 目标 dtype ulp 差 | A2 >1ulp 违反率 | A2 vs fp64（split/基线） | A3 位级 | 结论 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| decode-splitk4-bf16 | [32,64,512] | bf16 | 4 | pass | 128.0 | 0.42 | 0.00 | 0.00e+00 | 0.00 / 0.00 | pass | **EQUIV_PASS** |
| decode-splitk4-fp16 | [32,64,512] | fp16 | 4 | pass | 2048.0 | 0.46 | 0.00 | 0.00e+00 | 1.00 / 1.00 | pass | **EQUIV_PASS** |
| prefill-splitk1-bf16 | [64,64,1024] | bf16 | 1 | pass | 0.0 | 0.00 | 0.00 | 0.00e+00 | 1.00 / 1.00 | pass | **EQUIV_PASS** |
| prefill-splitk1-fp16 | [64,64,1024] | fp16 | 1 | pass | 0.0 | 0.00 | 0.00 | 0.00e+00 | 1.00 / 1.00 | pass | **EQUIV_PASS** |
| longk-splitk2-bf16 | [16,32,2048] | bf16 | 2 | pass | 144.0 | 0.65 | 0.00 | 0.00e+00 | 0.00 / 0.00 | pass | **EQUIV_PASS** |
| widerange-splitk4-bf16 | [32,64,512] | bf16 | 4 | pass | 768.0 | 0.40 | 0.00 | 0.00e+00 | 0.00 / 0.00 | pass | **EQUIV_PASS** |
| widerange-splitk4-fp16 | [32,64,512] | fp16 | 4 | pass | 576.0 | 0.44 | 0.00 | 0.00e+00 | 1.00 / 1.00 | pass | **EQUIV_PASS** |
| k7168-splitk4-bf16 | [32,64,7168] | bf16 | 4 | pass | 4608.0 | 0.83 | 0.00 | 0.00e+00 | 0.00 / 0.00 | pass | **EQUIV_PASS** |
| k7168-splitk4-fp16 | [32,64,7168] | fp16 | 4 | pass | 4864.0 | 0.84 | 0.00 | 0.00e+00 | 1.00 / 1.00 | pass | **EQUIV_PASS** |
| corner（±inf/NaN/subnormal/max/±0） | [32,32,512] | bf16 | 4 | pass | 0.0 | 0.00 | 0.00 | 0.00e+00 | 0.00 / 0.00 | pass | **EQUIV_PASS** |

（`A2 fp32 ulp 足迹` / `改动元素比例` 是**观测项**，不是判据——分片累加本来就会改变 fp32 位模式，
这是 split-K 的固有性质；判据是「目标 dtype 输出差 ≤ 1 ulp 且违反率 ≤ 1e-3」。宽动态范围用例
把 A 放大 1e3、B 缩小 1e-3，刻意制造抵消，足迹升到 4.9e3 fp32 ulp ≈ 6e-4 相对，仍远小于
`rtol = 1e-2`，且**没有任何元素跨过 bf16/fp16 的舍入边界**。）

**优化结论**：采纳 A1、A2、A3 三项（全部 `EQUIV_PASS`，脚本与结果表随本文留存）。
优化后公式（供 §3.1 拆解使用）：

```
C[i,j] = Σ_{s} Σ_{t ∈ slice_s} Â[i,t]·B̂[j,t]  (+ bias[j])
其中 Â/B̂ 为装载期归一后的 (M,K)/(N,K) tile；最终仅在 fixpipe 落盘时舍入一次到 c_dtype。
```

无采纳项之外的进一步公式化简：dense 收缩的 FLOPs 下界就是 `2mnk`（C3 的亚三次路径在 R4 被
容量与数值淘汰，见 §1.6.0），访存下界就是「A 一遍 + B 一遍 + C 一遍」（A1 已经消掉了
C 的第二遍）。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

盘点实现方案中的**全部**循环与标量计算点：

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由（不可替代时必填） |
|---|---|---|---|---|---|
| L1 | K 方向主循环（`k_slice` 次） | tile 级 `for` 循环 | 无（`T.Pipelined` 是流水调度，不是向量化对象） | ❌ | **tile 级顺序依赖**：`c0` 是循环携带的累加器（第 t 块的 MAD 读第 t-1 块的累加结果），已用多版本 L1 + L0 双缓冲消除*等待*，但累加本身的顺序依赖无法用归约/向量指令消除——这正是 `T.gemm` 的语义 |
| L2 | 输出 tile 的逐元素写（源 `T.Parallel(block_m, block_n)` + 守卫） | 逐元素标量循环（CUDA 写法） | **已消除**：A1 让 fixpipe 一次 DMA 完成 `L0C→GM`（含量化），不再有逐元素循环 | ✅（消除） | —（探针 D1/D2/D12：无 `T.Parallel` 出现，`asc_copy_l0c2gm` 直接带 quant_mode） |
| L3 | M/N 尾块的越界判定 | 逐元素 `if m_start+i < m and n_start+j < n` | **已消除**：区段截断（`am = T.min(m-m0, block_m)`，`T.copy` 只覆盖有效区段） | ✅（消除） | —（探针 D9 的 m 尾块 + n 尾块同时成立时仍无逐元素守卫） |
| L4 | bias 加（`has_bias` 分支） | `for i,j in T.Parallel(rows, cols): acc[i,j] += cast(bias[min(start+j, n-1)])` | `with T.SimtVF(threads=128): for i,j in T.Parallel(bm/2, bn): co[i,j] = cast(cv[i,j] + cast(bv[j]))`——lane 映射到 j（连续轴），每线程 2 元素（生成 `float2` 载入 + `__float22bfloat162_rn`） | ✅ | —（探针 D10 生成的 SIMT VF 体为 `float2` 向量载入，非逐元素标量） |
| L5 | bias 的越界夹取 `min(start+j, n-1)` | 逐元素标量 `T.min` | **已消除**：只装载 `[n0, n0+an)` 的有效区间 | ✅（消除） | — |
| L6 | 块索引 / 逻辑块分解（`bz/by/bx`、`m0/n0`） | 每块若干标量乘加 | 无 | ❌ | **block 级索引 / 任务映射计算**：每块一次（≤ 6 次整数运算 / 数千次 MAC），非逐元素热点，也没有逐元素等价的向量 API |
| L7 | split-K 的片内 K 段起点 `(bz·k_slice+step)·bk` | 每迭代一次标量乘加 | 无 | ❌ | 同 L6（tile 级循环变量），且 `k_slice` 是编译期常量，会被折叠成常量偏移 |
| L8 | `am/an/ak` 的 `T.min` 截断 | 每块/每迭代一次标量比较 | 无 | ❌ | **依赖编译期常量的边界处理**：`m,n,k` 均为编译期常量，整除时该分支在 Python 层被整段消掉（`exact_m/exact_n/exact_k`，探针 D1 与 D9 的代码差异即此）；非整除时必须逐块计算，属 tile 级元数据，不是逐元素热点 |
| L9 | GM→L1 的装载 | 源实现是 `cp.async` + 逐线程地址计算 | `T.copy` → MTE2 DMA（`asc_copy_gm2l1_nd2nz`） | ✅（消除） | — |
| L10 | L1→L0 的分块搬运与 L0 双缓冲 | 源是 ldmatrix/逐线程 | `ascend_gemm_l1` 模板内部（含 M_MTE1 事件 0/1 配对） | ✅（消除） | — |
| L11 | host 侧 shape/split_k 校验 | Python 编译期 `raise` | 无（不在 kernel 内） | ❌ | **host 侧元数据计算**：不进 kernel |

**向量化结论**：逐元素计算点 **L2/L3/L5 被 A1 与区段截断整体消除**，L4（仅 bias 分支）由
`T.SimtVF + T.Parallel` 向量化（lane = n 连续轴，2 元素/线程），L9/L10 由 DMA 与 Cube 模板
承担——实现方案中**不存在逐元素标量循环**。保留 4 处标量/循环：L1（tile 级累加顺序依赖）、
L6/L7（block 级索引）、L8（编译期常量驱动的边界元数据）、L11（host 元数据），理由类别分别为
「顺序依赖」「block 索引」「动态边界（此处为编译期常量，整除时消去）」「host 元数据」。
这与 §6 的循环结构一致：§6 表中只有 block 级与 tile 级循环，无逐元素标量循环。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

按算子类别（**Cube / MixCV 类**）枚举候选：矩阵布局、装载重排 vs 显式转置、epilogue 归属侧；
另对 epilogue 的向量化轴（lane 映射）单独决策。

**候选矩阵**：

| # | 决策点 | 候选 | 方案 | 依据 / 预估代价 | 是否采纳 |
|---|---|---|---|---|---|
| D1 | GEMM 形态 | ① 四种 `(trans_a, trans_b)` 直接映射到 GEMM flag | `T.gemm(..., transpose_A=…, transpose_B=…)` | ❌ 后端 `gemm_mad.py` 断言 L1-GEMM 只支持 NT；`gemm.h` 模板无 `TRANS_A` | ❌ |
| D1 | 同上 | ② **装载归一 + 恒 NT** | A/B 在 `GM→L1` 用 nd2nz/dn2nz 归一成 `(M,K)/(N,K)`，GEMM 恒 `transpose_B=True` | ✅ 唯一可行形态；精确等价（§1.6.1 A3）；探针 D3/D4/D5 覆盖 TN/NN/TT 装载（dn2nz 计数 1/1/2） | ✅ **采纳** |
| D1 | 同上 | ③ 显式 L0A/L0B 分块（`alloc_l0a/l0b` + `L1→L0 copy` + `T.gemm` on L0） | 手工双缓冲 L0 | ⏸ 可行（`PL-950-gemm-tiles` 已核验）但需手工管 L0 版本；`bn=256` 时 L0B 单版本 32 KB，`num_stages ≥ 3` 会超 64 KB；等价性同 ②  | ❌ 主选不采纳（L0 容量把流水深度锁死在 2，且交给模板可自管）；列为 §9 备选 |
| D2 | epilogue 归属 | ① **fixpipe 直落 GM**（Cube 侧） | `T.copy(c0[:am,:an], C[...])`，硬件 quant_pre 舍入 | ✅ `GetCCQuantPre` 映射 fp32→bf16=16、fp32→fp16=1、同 dtype=0；探针 D1/D2/D12 | ✅ **采纳（主选）** |
| D2 | 同上 | ② L0C→UB→SimtVF cast→UB→GM | `dual_copy` + `T.SimtVF` | ⏸ 可行（探针 D10 编译通过、`PL-950-mixed-epilogue` 形态）但多一次 UB 往返（`2·bm·bn·(4+2)` B 片上流量） | ❌ 主选不采纳；作为「若 Stage 3 实测 quant_pre 舍入不符」的回退（§8.3、§9.3 第 1 条） |
| D2 | 同上 | ③ L0C→GM 后再起一个 cast kernel（两遍物化） | 源 GPU 的 C4 形态 | ❌ R3/R4 已淘汰（C 流量翻倍） | ❌ |
| D3 | bias epilogue 的向量轴 | ① lane = m（行轴），bias 退化为逐行标量 | `T.Parallel(bm/2, bn)` 但内层串行走 m | ❌ bias 在 n 上广播，lane 落 m 会让每个 lane 读不同标量 → 广播代价反向 | ❌ |
| D3 | 同上 | ② **lane = n（连续轴）**，bias 为向量 lane 对齐 | `T.Parallel(bm/2, bn)`，内层维 = n；每线程 2 元素（`float2`） | ✅ 连续轴 + 广播轴对齐；探针 D10 生成的 VF 体为 `*(__ubuf__ float2*)(...+ threadIdx.x*2)` | ✅ **采纳（仅 has_bias 分支）** |
| D3 | 同上 | ③ 分片轴 N-split（每 AIV 半列） | `dual_copy` N-split | ⏸ 可行（`copy_op.py` 支持 `dual_dst_ctl=0b10`，要求完整 N % 32 = 0），但 GM 写变成半行（bn=256 bf16 → 每行 512 B 连续段），且 `PL-950-mixed-epilogue` 只核验了 M-split | ❌ 主选 M-split；N-split 仅作奇数行尾块的后备（本 op 不用） |
| D4 | 块遍历序（L2 亲和） | ① n-fastest（源的顺序：`bx` = n 最快 + `T.use_swizzle(10)`） | `bid` 分解为 n 变化最快 | ❌ 每 m panel 重读整份 B：t4096 gate/up 约 1.9 GB DRAM 流量（vs ideal 151 MB） | ❌ |
| D4 | 同上 | ② **m-fastest**（`bx` 慢、`by` 快） | `bx = bid // (mt·split)`、`by = bid % (mt·split) // split` | ✅ A 面板 ≤ L2（最大 58.7 MB < 128 MiB），B tile 在 mt 个 m-block 内复用（§0.6 R4 表） | ✅ **采纳** |
| D4 | 同上 | ③ `T.use_swizzle` panel 遍历 | `T.use_swizzle(10)` | ❌ ascend codegen 不实现 `threadblock_swizzle_pattern`（只有 CUDA codegen 处理），且 grid 是 1-D | ❌（探针不涉及；依据为后端源码检索） |
| D5 | 输入 tile 的重排位置 | ① GM→L1 装载期重排（nd2nz/dn2nz） | `T.copy(..., transpose=True)` | ✅ 平台已核验路径（`PL-950-gemm-tiles` 的 NN 分支；`copy_op.py` 的 `transpose` 只作用于 GM→L1） | ✅ **采纳** |
| D5 | 同上 | ② host 侧 permute 输入 | `A.t().contiguous()` 等 | ❌ 额外 `mk·2` 的 GM 往返（t4096 时 58.7 MB × 2），且改变调用方契约 | ❌ |
| D5 | 同上 | ③ UB 内重排 | `UB→UB` + `make_ascend_compact_nz_layout` | ❌ 只适用于 UB 内 ND→NZ scatter，用不上（GEMM 直接吃 L1 分形），且多一趟 UB 流量 | ❌ |

**候选评分（关键量）**：

| 评分项 | D1② NT 归一 | D2① fixpipe 直落 | D3② lane=n | D4② m-fastest |
|---|---|---|---|---|
| 整除性 | `bm/bn/bk` 均 16 倍数（后端 `static_assert`）；本 op 的 n ∈ 4096/3072/7168/5120 全被 256 整除 | 尾块用区段截断，`CheckCompactL0Region` 要求尾部零起点（满足） | 内层维 = bn = 256 → fp16/bf16 向量宽 8 的整数倍，尾 lane 浪费 0 | mt·split 为整数，`bid` 分解无余数 |
| 尾 lane 浪费率 | 0（GEMM 不吃 lane） | 0（DMA 按 burst） | `(ceil(256/2)*2 - 256)/256 = 0`（每线程 2 元素整除） | 不适用 |
| 累加链形态 | K 维串行（stride≠1 的是 L1 内分形，硬件处理），不影响向量化 | — | lane 沿 n（stride=1），bias 广播沿 lane 对齐 | — |
| repack 代价 | 装载期一次（计入 DMA，不额外加流量）；host permute 被排除 | 无（无 UB 中转） | 无 | 无（仅索引重排） |
| UB / 片上容量影响 | L1 147.5 KB（3 版本）/ 512 KB | **UB 0 B**（主选路径不分配 UB） | bias 分支：每 AIV `cv+co+bv = 64·256·4 + 64·256·2 + 512 = 98816 B`（38.9%，仅 has_bias，见 §4.5） | 不改变片上占用 |

**布局决策结论**：采纳 **D1②（装载归一 + 恒 NT GEMM）、D2①（L0C 经 fixpipe 直落 GM，硬件
quant_pre 完成唯一一次舍入）、D3②（bias 分支 lane = n 连续轴、M-split 双 AIV）、D4②（m-fastest
块遍历序）、D5①（GM→L1 装载期重排）**。主选路径的核内布局是「L1 K-major NZ 分形 + L0C fp32
累加器 + 无 UB 缓冲」，向量化轴仅出现在 `has_bias` 的 Vector epilogue 上（lane = n）。
该结论同步落入 §3.3 伪代码（`a1/b1` 形状、`T.gemm` 的 `transpose_B=True`、`bid` 分解）与
§6 循环结构（累加循环的内层维 = `bk`，epilogue 无逐元素循环）。

**未触发实验裁决三件套的说明**：本节的取舍分歧都不是「落在一个未实证常数的判定裕度内」：
D1② 是后端硬约束（assert）；D2① 的舍入相同性有位级验证（§1.6.1 A1）；D4② 的收益量级是
4–32 倍 DRAM 流量（由 `l2_size = 128 MiB` 这个型号档案字段与可精确计算的足迹决定，不依赖
未实测带宽）；D3 只作用于本 op 未使用的 `has_bias` 分支。D2② 作为回退方案的结构深度已写到
§3.3/§4.5 可直接实现的程度（`cv/co/bv` 缓冲、`dual_copy`、`SimtVF` 体），并在 §8.3 与 §9.3 第 1 条给出
触发条件与切换判据（`Stage 3 L0 的目标 dtype 最大 ulp 差 > 1` 时切换），Stage 4 的 A/B 会把
它作为候选之一。主选与备选的结构差异不影响 §5 的分核参数。

---
## 2. 编程模式选型

### 2.1 模式结论

**选定模式：Cube**（平台 `programming_modes.choices` 的 `cube` 键）。
`has_bias and split_k == 1` 的变体在同一 kernel 内启用 Vector epilogue，其发射形态是
`__mix__(1, 2)`（Cube + 2 AIV 混合）；本 op 的 10 次调用全部 `has_bias = False`，因此
主选路径的模式记为 **Cube**，混合只出现在该可选分支（§9.3 单列）。

### 2.2 选型理由

平台 `migration_default` 是 `simd`。本算子**有意偏离**，理由如下（三条独立证据）：

1. **算术构成**：全部有效算术是 `2·m·n·k` 次 MAC（decode 段 1.88 GFLOP、prefill 段 240.5 GFLOP），
   没有逐元素/归约热点（§1.6.2：逐元素点已被 A1 整体消除，L4 仅 bias 分支）。用 `simd` 只能
   逐元素展开 GEMM，丧失了平台矩阵单元——这正是 R4「计算单元匹配」的第一条（MAC 密集段必须
   落平台矩阵单元）。
2. **API 事实**：`tilelang.ascend.language` 提供 `T.gemm`，其后端 `Gemm::SelectInst` 对 ascend
   恒返回 `ascend.mad`，落到 `ascend_gemm_l1<…>` → `ascend.mad` 指令；探针 D1/D2/D3/D4/D5/D6/
   D7/D8/D9/D11/D12 生成的 kernel 签名全部是 `extern "C" __global__ __cube__`（**纯 Cube 发射**，
   没有 AIV 参与）。
3. **同 op 先例的差异**：同批次 `mla_dense_bwd/_bwd_kernel` 选 `hybrid`，因为它的每个迭代
   都要在 Cube（5 个 GEMM）与 Vector（`exp/mask/cast/转置`）之间交替；本算子的主选路径没有
   这种交替（epilogue 由 fixpipe 硬件完成），所以不继承 `hybrid`。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_l1`（A/B 操作数，`num_stages` 版本）+ `T.alloc_l0c`（fp32 累加器）。主选路径**不分配 UB**；bias 分支分配 3 个 UB buffer（`cv`/`co`/`bv`） |
| 计算方式 | `T.gemm(a1, b1, c0, transpose_B=True, clear_accum=(step == 0))`；L1→L0 分块与 L0 双缓冲由后端模板内部完成（`TILE_K_SUB = 64`） |
| 搬运方式 | `T.copy`：GM→L1（MTE2，nd2nz / dn2nz）、L0C→GM（fixpipe，带 quant_pre）；bias 分支另有 `dual_copy`（L0C→UB / UB→GM）与 GM→UB（bias 切片） |
| 线程组织 | 无 `threads` 参数（平台 `T.Kernel(N)` 签名不含）；bias 分支的 `T.SimtVF(threads=128)` 自带线程数 |
| 同步 | 全自动（§7）：`AutoSchedule → InsertSync` 生成 MTE2/MTE1/M/FIX 事件配对，无手工 flag |

---

## 3. API 映射设计

### 3.1 公式拆解

输入为 §1.6.1 的**优化后公式**（A1/A2/A3 已采纳）：

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `m0 = by·bm, n0 = bx·bn` | 输出 tile 起点（§1.6.3 D4 的遍历序） |
| 2 | `k0 = (bz·k_slice + step)·bk` | 收缩轴分片（split-K，§1.6.1 A2） |
| 3 | `Â_tile = gather(Â, [m0,m0+am) × [k0,k0+ak))` | A 的 `(M,K)` 规整（§1.6.1 A3，nd2nz/dn2nz） |
| 4 | `B̂_tile = gather(B̂, [n0,n0+an) × [k0,k0+ak))` | B 的 `(N,K)` 规整（同上） |
| 5 | `c0 += Â_tile · B̂_tileᵀ` | Cube MAD；`clear_accum` 仅在 `step == 0` |
| 6 | `C[m0+·, n0+·] = cast_c_dtype(c0)` | 一次舍入（§1.6.1 A1），尾块区段截断 |
| 6′ | `C[bz, m0+·, n0+·] = c0` | split-K 分支：fp32 部分和，无舍入 |
| 6″ | `C[m0+·, n0+·] = cast(c0 + cast_f32(bias[n0+·]))` | has_bias 分支：Vector epilogue（R5） |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API（`import tilelang.ascend.language as T`） | 参数 | 模式 |
|------|----------|-------------|------|------|
| 0 | 一维核网格 | `T.Kernel(nt * mt * split_k)` | 仅一维；返回 `bid` | Cube |
| 3′ | 操作数多版本（分配期） | `T.annotate_buffer_versions({a1: num_stages, b1: num_stages})` | 正整数版本数 | Cube |
| 2 | K 流水 | `T.Pipelined(k_slice, num_stages=num_stages)` | 静态 trip count（编译期常量） | Cube |
| 3 | A tile 规整装载 | `T.copy(A[m0:m0+am, k0:k0+ak], a1[:am, :ak])` / trans_a 时 `T.copy(A[k0:k0+ak, m0:m0+am], a1[:am, :ak], transpose=True)` | `transpose=True` 只对 GM→L1 生效（dn2nz） | Cube（MTE2） |
| 4 | B tile 规整装载 | `T.copy(B[n0:n0+an, k0:k0+ak], b1[:an, :ak])` / trans_b=False 时同上加 `transpose=True` | 可选 `l2_cache_ctrl`（Stage 4 旋钮） | Cube（MTE2） |
| 5 | 矩阵乘累加 | `T.gemm(a1, b1, c0, transpose_B=True, clear_accum=(step == 0))` | `transpose_A` 必须 `False`（后端断言）；`policy` 在 ascend 侧被忽略 | Cube（`ascend.mad`） |
| 6 | 量化落盘 | `T.copy(c0[:am, :an], C[m0:m0+am, n0:n0+an])` | quant_pre 由 src/dst dtype 自动选（16/1/0） | Cube（fixpipe） |
| 6′ | 部分和落盘 | `T.copy(c0[:am, :an], C[bz, m0:m0+am, n0:n0+an])` | fp32→fp32，quant_mode 0 | Cube（fixpipe） |
| 6″ | bias 切片装载 | `T.copy(bias[n0:n0+block_n], bv)` | UB 目的地 `(block_n,)` | Vector（MTE2） |
| 6″ | L0C 分片出数 | `T.dual_copy(c0, cv)` | `cv:(bm/2, bn)` fp32，M-split（`dual_dst_ctl=1`） | Cube→Vector（fixpipe） |
| 6″ | bias 加 + cast | `with T.SimtVF(threads=128): for i, j in T.Parallel(bm/2, bn): co[i,j] = T.cast(cv[i,j] + T.cast(bv[j], "float32"), dtype)` | lane = n（§1.6.3 D3） | Vector |
| 6″ | 回写 | `T.dual_copy(co, C[m0:m0+bm, n0:n0+bn])` | UB→GM（per-AIV copy） | Vector（MTE3） |
| — | 输出分配 | `tilelang.compile(main, target="ascend", out_idx=-1)` | 自动分配 `C` | — |

### 3.3 计算伪代码

```python
def _gemm_basic_kernel(m, n, k, trans_a, trans_b, dtype="float16", has_bias=False, out_dtype=None):
    c_dtype = out_dtype or dtype

    def _gemm_basic_func(block_m=64, block_n=64, block_k=64, num_stages=2, split_k=1):
        # ---- 编译期契约（§5.2；全部是编译期常量判定，进不了运行期） ----
        if split_k > 1:                       # decode 段：调用方的 128 必须缩到 m
            block_m = min(block_m, m)
            assert m % block_m == 0 and n % block_n == 0 and k % block_k == 0
            assert (k // block_k) % split_k == 0
        assert block_m % 16 == 0 and block_n % 16 == 0 and block_k % 16 == 0   # 后端 static_assert(M%16)
        assert k % block_k == 0               # K 尾块会污染有效累加（源注释）
        mt, nt, k_tiles = ceil(m / block_m), ceil(n / block_n), k // block_k
        k_slice = k_tiles // split_k
        exact_m, exact_n = (m % block_m == 0), (n % block_n == 0)
        out_shape = (m, n) if split_k == 1 else (split_k, m, n)
        out_dtype_ = c_dtype if split_k == 1 else "float32"
        a_shape = (k, m) if trans_a else (m, k)
        b_shape = (n, k) if trans_b else (k, n)

        @T.prim_func
        def main(A: T.Tensor(a_shape, dtype), B: T.Tensor(b_shape, dtype),
                 bias: T.Tensor((n,), dtype), C: T.Tensor(out_shape, out_dtype_)):
            with T.Kernel(nt * mt * split_k) as bid:
                # §1.6.3 D4：n 慢 / m 快 / split 最快
                bx = bid // (mt * split_k)
                by = bid % (mt * split_k) // split_k
                bz = bid % split_k
                m0, n0 = by * block_m, bx * block_n
                am = block_m if exact_m else T.min(m - m0, block_m)
                an = block_n if exact_n else T.min(n - n0, block_n)

                a1 = T.alloc_l1((block_m, block_k), dtype)      # (M,K) 规整（R1）
                b1 = T.alloc_l1((block_n, block_k), dtype)      # (N,K) 规整（R1）
                c0 = T.alloc_l0c((block_m, block_n), "float32")  # fp32 累加（R3）
                T.annotate_buffer_versions({a1: num_stages, b1: num_stages})  # R2

                if k_tiles == 0:                                  # k == 0：源语义（写 bias 或零）
                    T.clear(c0)
                for step in T.Pipelined(k_slice, num_stages=num_stages):
                    k0 = (bz * k_slice + step) * block_k
                    if trans_a:
                        T.copy(A[k0:k0 + block_k, m0:m0 + am], a1[:am, :], transpose=True)
                    else:
                        T.copy(A[m0:m0 + am, k0:k0 + block_k], a1[:am, :])
                    if trans_b:
                        T.copy(B[n0:n0 + an, k0:k0 + block_k], b1[:an, :])
                    else:
                        T.copy(B[k0:k0 + block_k, n0:n0 + an], b1[:an, :], transpose=True)
                    T.gemm(a1, b1, c0, transpose_B=True, clear_accum=(step == 0))

                # ---- epilogue ----
                if has_bias:                                     # 仅 split_k == 1（源契约）
                    cv = T.alloc_shared((block_m // 2, block_n), "float32")   # UB, 2 AIV 各半
                    co = T.alloc_shared((block_m // 2, block_n), dtype)
                    bv = T.alloc_shared((block_n,), dtype)
                    T.copy(bias[n0:n0 + block_n], bv)
                    T.dual_copy(c0, cv)                          # fixpipe, M-split
                    with T.SimtVF(threads=128):                  # lane = n（D3）
                        for i, j in T.Parallel(block_m // 2, block_n):
                            co[i, j] = T.cast(cv[i, j] + T.cast(bv[j], "float32"), dtype)
                    T.dual_copy(co, C[m0:m0 + block_m, n0:n0 + block_n])
                elif split_k == 1:
                    if exact_m and exact_n:                      # 全静态 extent
                        T.copy(c0, C[m0:m0 + block_m, n0:n0 + block_n])
                    else:                                        # 区段截断（S2）
                        T.copy(c0[:am, :an], C[m0:m0 + am, n0:n0 + an])
                else:
                    T.copy(c0[:am, :an], C[bz, m0:m0 + am, n0:n0 + an])   # fp32 部分和

        return main

    return _gemm_basic_func     # 内层签名去掉 threads（K9 / §0.5 O10）
```

> 伪代码里 `bias` 一律出现在签名中；实现时按源 `_main` 的做法在 Python 层生成两套 `@T.prim_func`
> （有 bias：`(a, b, bias, c)`；无 bias：`(a, b, c)`），`global_symbol` 分别为
> `_gemm_basic_bias_main` / `_gemm_basic_main`。`ceil()` 指 Python 的 `-(-x // y)`（编译期整数）。

### 3.4 API 可行性确认

| API | 来源确认 | 设计期探针 | 状态 |
|---|---|---|---|
| `T.Kernel(N)`（一维，无 `threads`） | `.agents/platforms/toolchains/ascend/references/programming.md`「接口 / 已核对规则」；`ascend/language/kernel.py` | 全部 12 个探针用例 | ✅ 已验证（编译） |
| `T.alloc_l1` / `T.alloc_l0c` | 同上（`allocate.py`，scope `shared.l1` / `shared.l0c`） | D1–D12 | ✅ |
| `T.copy`（GM→L1，region 形式） | `copy_op.py`；`PL-950-gemm-tiles`（已核验条目） | D1–D9 | ✅ |
| `T.copy(..., transpose=True)`（GM→L1 dn2nz） | `copy_op.py`（`transpose` 文档作用域 = GM→L1） | D3/D4/D5（`dn2nz` 计数 1/1/2） | ✅ |
| `T.copy`（L0C→GM，含 dtype 转换） | `copy.cc::GetCCQuantPre`（fp32→bf16=16/ fp32→fp16=1/ 同 dtype=0）；`vf_checker.py` 对 `shared.l0c` 源显式豁免 dtype 检查 | D1（16）/D2（1）/D12（0） | ✅ 编译已验证；**运行期舍入**由 Stage 3 的 L0-1/L0-2 核验（§8.3） |
| `T.copy(c0[:am,:an], …)`（零起点部分区段） | `copy.cc::CheckCompactL0Region` 要求尾部维零起点 | D8（am=1）/D9（m 与 n 双尾块） | ✅ |
| `T.gemm(A_l1, B_l1, C_l0c, transpose_B=True, clear_accum=…)` | `gemm_mad.py::_lower_l1`（`ascend_gemm_l1`）、`gemm.h` 模板 | D1–D12（`ascend_gemm_l1<128,64,256,64,true,bfloat16_t>` 等） | ✅ |
| `T.annotate_buffer_versions` | `annotations.py`（正整数要求）；`PL-950-gemm-tiles` | D1（`step % 3` 环形）、D11（`step & 3`，4 版本 + 4 组事件） | ✅ |
| `T.Pipelined(n, num_stages=ns)` | 平台 ruleset + `PL-950-gemm-tiles`/`PL-950-vector-execution` | 全部用例 | ✅ |
| `T.dual_copy`（L0C→UB M-split、UB→GM） | `copy_op.py::dual_copy`（`dual_dst_ctl=0b01` = M-split）；`PL-950-mixed-epilogue`（已核验条目） | D10（`asc_dual_dst_mode(1)`） | ✅ 编译；运行期见 §9.3 第 1 条 |
| `with T.SimtVF(threads=128): T.Parallel(...)` + `T.cast` | `frame.py:SimtVF`；`PL-950-row-primitives`（已核验条目） | D10（生成 `float2` 载入 + `__float22bfloat162_rn`） | ✅ |
| `tilelang.compile(main, target="ascend", out_idx=-1)` | 平台 ruleset「使用 `tilelang.compile(..., target="ascend", out_idx=...)`」 | 全部用例 | ✅ |
| `T.use_swizzle` | 存在于 `tilelang.language`（经 `common` 转发），但 ascend codegen 不实现 `threadblock_swizzle_pattern` | — | ❌ 不使用（§0.5 O6 / §1.6.3 D4③） |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案与依据 |
|------|---------------|----------|
| `T.Kernel` 只支持一维网格、无 `threads` 参数 | **是** | grid 展平为 `nt·mt·split_k` 的一维 `bid`，在 kernel 内分解（§3.3）；不传 `threads` |
| `T.gemm` 的 L1 形式只支持 NT（`trans_A=False, trans_B=True`） | **是** | R1：四种布局在装载期归一（§0.6）；探针 D3/D4/D5 |
| `T.gemm` 要求 tile M/N 为 16 的倍数（`gemm.h` 的 `static_assert(M % FRAC_N == 0)`、`static_assert(N % FRAC_N == 0)`，`FRAC_N = 16`） | **是** | `block_m/block_n/block_k` 均取 16 的倍数并加编译期 `assert`（§5.2） |
| `T.gemm` 的 `tile_k_sub` 必须整除 `block_k`，且由 L0 容量反推（`K % TILE_K_SUB == 0`） | **是** | `block_k = 64`（bf16/fp16 的 C0 = 16，`64 % 16 == 0`）；`_compute_tile_k_sub` 选 64（L0B 2 阶段正好 64 KiB） |
| K 方向不允许尾块（K 尾块污染有效累加） | **是** | 前置 `k % block_k == 0`；本 op 的 k ∈ 7168/18432/2048/1536/5120 全被 64 整除 |
| L0C 容量 256 KiB | **是** | `block_m·block_n·4 ≤ 262144`；主选 (128,256) = 128 KiB（50%） |
| L1 容量 512 KiB | **是** | `2·(bm+bn)·bk·num_stages`；主选 147456 B（28.1%），`num_stages = 4` 时 196608 B（37.5%） |
| L0A/L0B 各 64 KiB（模板按 2 阶段均分） | **是** | 由 `_compute_tile_k_sub` 自动满足：L0A 2×16 KiB、L0B 2×32 KiB |
| 自动调度默认开启、手工跨核 flag 未认证 | **是** | 全部同步交自动调度（§7），不写 `ascend_set_flag/wait_flag/barrier` |
| `annotate_unlimit_memory` 不增加物理容量 | 是（不使用） | 不调用 |
| 平台 `T.Persistent` 无正确性条目 | 是（不采用） | 主选逻辑块超发 + 硬件波次派发；persistent 列为 §9.3 备选（§5.5 依据） |
| 张量/网格对齐（32 B burst） | 是 | 尾块走区段截断，`n` 侧 `bn=256` 使每行 ≥ 512 B（bf16）；`m=1` 时 `asc_copy_l0c2gm` 生成 burst=1 行（探针 D8 已生成该形态） |

#### 3.5.2 参考实现差异说明

（完整分析见 §0.5/§0.6，此处只列影响 API 选型的关键差异）

| 差异项 | 参考实现（GPU） | 本项目（Ascend） | 转换方案 |
|--------|----------------|-----------------|---------|
| Kernel 结构 | `T.Kernel(nt, mt, split_k, threads=128)` 三维 grid | `T.Kernel(nt*mt*split_k)` 一维 grid | `bid` 整数分解（§3.3），并按 L2 足迹选 m-fastest（§1.6.3 D4） |
| 矩阵 API | `T.gemm(a_smem, b_smem, c_frag, transpose_A, transpose_B, policy=FullRow)`，SMEM 输入 | `T.gemm(a1_l1, b1_l1, c0_l0c, transpose_B=True)`，L1 输入 + L0C 累加；仅 NT | R1 装载归一；`policy` 忽略 |
| 累加器 | `alloc_fragment`（寄存器，按线程分布） | `alloc_l0c`（fp32，不可逐元素访问） | R3：落盘走 fixpipe；bias 走 Vector（R5） |
| 内存分配 | `alloc_shared`（SMEM 操作数 + `c_smem` 中转） | `alloc_l1` + `alloc_l0c`，无 UB（主选） | 去掉了 `c_smem` 中转（O7 舍弃） |
| 布局注解 | `make_swizzled_layout(a_smem/b_smem/c_smem)` | 编译器按 GEMM flag 推导 NZ 分形（`infer_layout`） | 不写 `annotate_layout`（O5 等价替换） |
| L2 优化 | `T.use_swizzle(10)` | 无对应 attr | R4 自己定遍历序 |
| 流水 | `T.Pipelined(k_slice, num_stages)` + 隐式 SMEM 版本 | `T.Pipelined` + **显式** `annotate_buffer_versions` | 显式声明让 L1 预算可静态核算（§4.5）；隐式路径的版本数由调度器逐 buffer 决定（探针观测 A 3 版本 / B 2 版本），预算不可静态判定 |
| 输出 | `c_local → c_smem → T.Parallel(守卫) → GM` | `L0C → fixpipe → GM`（硬件 quant_pre） | §1.6.1 A1 |
| 编译配置 | `pass_configs={"tl.disable_warp_specialized": True}`、`-O3 -DENABLE_BF16` | 无 pass_configs（自动调度默认）、`-O3` | O11 舍弃 CUDA 专属项 |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `.agents/platforms/toolchains/ascend/knowledge/repro/PL-950-gemm-tiles.py` | **高度相似**（唯一的已核验多 tile GEMM 条目） | `alloc_l1/alloc_l0a/alloc_l0b/alloc_l0c` + `annotate_buffer_versions` + `T.Pipelined` + `T.gemm(transpose_B=True, clear_accum=step==0)` + `T.copy(..., transpose=True)` 的 NN 装载 + M/N/K 尾块写法 |
| `.agents/platforms/toolchains/ascend/kernel_template.py` | 中（结构模板） | 平台工程的统一骨架：`@T.prim_func` + `T.Kernel` + `tilelang.compile(target="ascend", out_idx=-1)` + `--level L0/L1/L2/boundary` 的分层测试形态 |
| `.agents/platforms/toolchains/ascend/knowledge/repro/PL-950-mixed-epilogue.py` | 中（仅 bias 分支） | `dual_copy(L0C→UB, M-split)` + `T.SimtVF(threads=64)` + `dual_copy(UB→GM)` 的混合 epilogue |
| `examples/gemm/example_gemm.py` | 中（低层写法对照） | `alloc_shared + T.Pipelined + T.gemm + T.copy` 的分块 GEMM 结构；**注意**其 `target="npuir"` 与 `initC=`/`size=`/`b_transpose=` 参数属于另一后端（NPUIR），本平台不可照搬（`target="ascend"` 用 `clear_accum=`/`transpose_B=`） |
| `examples/gemm/matmul.py` | 低（仅结构） | 同上：`T.load_nd2nz`/`T.store_fixpipe`/`is_npu=True` 是 NPUIR 后端 API，**不在** `tilelang.ascend.language` 的导出里（本文 §3.4 已按 ascend 命名空间逐个核对） |
| `examples/mla_dense_bwd/_bwd_kernel/DESIGN.md` | 中（平台设计写法与容量核算口径） | L1/L0C/UB 三层预算表格式、`T.min` 尾块与 OOB clamp 的讨论、自动同步表 |
| `examples/TileOPs/tileops/kernels/moe/shared_expert_mlp/_shared_expert_mlp_fwd_kernels.py` | **迁移源** | 逐字源实现（§0 的全部解读对象） |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `A` | `(m, k)`（`trans_a=False`）或 `(k, m)`（`trans_a=True`） | `float16` 或 `bfloat16` | 本 op 的两个 GEMM 均 `trans_a=False` |
| `B` | `(n, k)`（`trans_b=True`）或 `(k, n)`（`trans_b=False`） | 同上 | 本 op 均 `trans_b=True`（权重存成 `[n, k]`） |
| `bias` | `(n,)` | 同上 | 仅 `has_bias and split_k == 1` 出现 |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `C`（`split_k == 1`） | `(m, n)` | `out_dtype or dtype` | 一次舍入的最终结果；由 `out_idx=-1` 自动分配 |
| `C`（`split_k > 1`） | `(split_k, m, n)` | **float32** | 部分和；下游 `splitk_reduce_kernel` 按 slice 序求和、加 bias、可选 silu_and_mul、再 cast。**本 kernel 不得自行量化该张量**（否则归约前就丢精度，且破坏下游的 fp32 求和契约） |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| `a1` | `(block_m, block_k)` × `num_stages` 版本 | 输入 dtype | L1 | A 的 `(M,K)` 规整 tile（NZ 分形由编译器注入） |
| `b1` | `(block_n, block_k)` × `num_stages` 版本 | 输入 dtype | L1 | B 的 `(N,K)` 规整 tile |
| `a0` / `b0` | `(block_m, TILE_K_SUB)` / `(block_n, TILE_K_SUB)`，各 2 阶段 | 输入 dtype | L0A / L0B | 由 `ascend_gemm_l1` 模板内部申请（本设计不显式分配） |
| `c0` | `(block_m, block_n)` | float32 | L0C | 跨 K tile 的累加器（`clear_accum` 仅首块） |
| `cv` | `(block_m/2, block_n)` | float32 | UB（每 AIV） | **仅 has_bias**：L0C 出数半块，作 Vector 输入 |
| `co` | `(block_m/2, block_n)` | 输入 dtype | UB（每 AIV） | **仅 has_bias**：cast 后的回写块 |
| `bv` | `(block_n,)` | 输入 dtype | UB（每 AIV） | **仅 has_bias**：bias 切片（沿 lane 广播） |

### 4.4 内存搬运路径

```
GM[A] --MTE2 nd2nz (trans_a=False)｜dn2nz (trans_a=True)--> L1[a1, num_stages 版本]
GM[B] --MTE2 nd2nz (trans_b=True) ｜dn2nz (trans_b=False)--> L1[b1, num_stages 版本]
L1[a1] --MTE1 分块（TILE_K_SUB=64, 模板内部双缓冲）--> L0A[a0]
L1[b1] --MTE1 分块（同上）-------------------------> L0B[b0]
L0A,L0B --Cube MAD（ascend.mad）-------------------> L0C[c0, fp32, 跨 K 迭代累加]
L0C[c0] --fixpipe asc_copy_l0c2gm (quant_pre)------> GM[C]              # split_k == 1, has_bias == False
L0C[c0] --fixpipe asc_copy_l0c2gm (quant 0)--------> GM[C[bz]]          # split_k > 1
L0C[c0] --fixpipe asc_copy_l0c2ub (dual, M-split)--> UB[cv]             # has_bias
UB[cv] --> SimtVF(+ bias 广播, cast) --> UB[co]                          # has_bias
UB[co] --MTE3 asc_copy_ub2gm_align (+ per-AIV)-----> GM[C]              # has_bias
GM[bias] --MTE2--> UB[bv]                                                # has_bias
```

主选路径（本 op 的全部 10 次调用）只有 `GM→L1→L0A/L0B→L0C→GM` 五级，**不经 UB**；
无 GM 中转的中间块（A/B tile 全程在 L1/L0）。

### 4.5 片上预算

**UB**（主选路径）

| Buffer | Shape | dtype | 大小 (Bytes) |
|--------|-------|-------|-------------|
| 主选路径不分配（fixpipe 直落 GM） | — | — | 0 B |
| **合计** | | | 0 B |

bias 变体（仅 `has_bias`，本 op 不使用）在同一层级另需 `cv 64·256·4 = 65536` +
`co 64·256·2 = 32768` + `bv 256·2 = 512` = 98816 B（每 AIV，38.9% / 253952 B）。

**L1**（`num_stages = 3`，`block_m = 128, block_n = 256, block_k = 64`）

| Buffer | Shape | dtype | 大小 (Bytes) |
|--------|-------|-------|-------------|
| a1（3 版本） | (128, 64) × 3 | bf16 | 49152 B |
| b1（3 版本） | (256, 64) × 3 | bf16 | 98304 B |
| **合计** | | | 147456 B |

`num_stages = 4` 时 196608 B；decode 配置（`bm=32`）时 `2·(32+256)·64·3 = 110592` B。

**L0C**

| Buffer | Shape | dtype | 大小 (Bytes) |
|--------|-------|-------|-------------|
| c0 | (128, 256) | fp32 | 131072 B |
| **合计** | | | 131072 B |

decode 配置（`bm=32, bn=256`）时 32768 B；(128, 256) 是本设计允许的最大 tile。

**L0A**（模板内部，2 阶段）

| Buffer | Shape | dtype | 大小 (Bytes) |
|--------|-------|-------|-------------|
| a0 | (128, 64) × 2 | bf16 | 32768 B |
| **合计** | | | 32768 B |

**L0B**（模板内部，2 阶段）

| Buffer | Shape | dtype | 大小 (Bytes) |
|--------|-------|-------|-------------|
| b0 | (256, 64) × 2 | bf16 | 65536 B |
| **合计** | | | 65536 B |

`TILE_K_SUB` 由 `_compute_tile_k_sub` 取 `min(max_k_a, max_k_b, block_k)`（L0 容量 64 KiB / 2 阶段
→ 每阶段 32 KiB），本配置下 = 64：`b0` 每阶段 32 KiB × 2 = 64 KiB **恰好用满 L0B**，这是模板
的分配规则而非超限（`total > cap` 才越界）。

### 4.6 动态轴

**无**。`m, n, k, block_m, block_n, block_k, num_stages, split_k` 全部是工厂/内层工厂的编译期
常量（`@tilelang.jit` 按参数组合特化并缓存），`T.Kernel` 的 grid 与所有循环界都是编译期常量
（探针代码里 `for step in T.Pipelined(ks, …)` 的 trip count 为常量，生成 `for (int32_t step = 0;
step < 112; ++step)`）。`T_q/T_kv` 之类的 packed 视图不存在于本 kernel。

### 4.7 JIT 配置

```python
@tilelang.jit(out_idx=[-1], target="ascend", compile_flags=["-O3"])
def _gemm_basic_func(block_m: int = 64, block_n: int = 64, block_k: int = 64,
                     num_stages: int = 2, split_k: int = 1) -> Callable:
    ...
# 与源实现的差异：去掉 pass_configs={"tl.disable_warp_specialized": True}（CUDA 概念）
# 与 -DENABLE_BF16（ascend 后端按 dtype 直接支持 bf16）；保留 -O3。
```

`out_idx=-1`：`C` 由 JIT 分配（`split_k > 1` 时按 `(split_k, m, n)` fp32 分配）。

---

## 5. Tiling 策略

### 5.1 计算类型

**类型：纯 Cube**（主选路径，10/10 次调用）；`has_bias` 变体为 **Cube + Vector 混合**（2 个
AIV 的 epilogue，本 op 不触发）。判定依据：全部算术是 MAC（§1.6.2 逐元素点已消除），
Cube MAD 是唯一合适的计算单元（§2.2）。

机械核算（代表 workload ds-v3-t4096 gate/up：M = 4096, N = 4096, block_m = 128,
block_n = 256）：逻辑核数 = 512 = ceil(4096/128) × ceil(4096/256)；物理核数 28，故不属
「逻辑核数 ≤ 物理核数」的单波情形，按多波次派发（§5.5）。

### 5.2 Block 划分

主选（prefill / 任意 `m ≥ 128`）与 decode（`m ≤ 32`，走 split-K）两档：

```python
block_m = 128        # 主选：L0C 128 KB（50%），M 方向一个 mad 行块族
block_n = 256        # n = 4096/3072/7168/5120 全被它整除（尾块 0）
block_k = 64         # C0 = 16 的 4 倍；TILE_K_SUB 恰好 64 → L0B 2 阶段用满 64 KiB
num_stages = 3       # L1 147456 B / 512 KiB
```
```python
block_m = m          # decode：32（或 <32 时取 16 的倍数，且必须整除 m，见下）
block_n = 256
block_k = 64
num_stages = 3       # L1 110592 B；num_stages = 4 时 147456 B 仍在预算内
```

**配置契约（设计期确定的硬规则，写进 kernel 的编译期断言）**：

1. `block_m, block_n, block_k` 必须是 **16 的倍数**（后端 `gemm.h` 的
   `static_assert(M % FRAC_N == 0, N % FRAC_N == 0)`，`FRAC_N = 16`；
   `_compute_tile_k_sub` 要求 `TILE_K_SUB % C0 == 0`，bf16/fp16 的 `C0 = 16`）。
2. `k % block_k == 0`（K 方向无尾块；源实现同款约束，K 尾块会污染有效累加）。
3. `split_k > 1` 时追加：`block_m := min(block_m, m)`、`m % block_m == 0`、`n % block_n == 0`、
   `k % block_k == 0`、`split_k | (k // block_k)`（源实现的精确 tile 前置）。
4. **`m < 16` 时 `split_k` 必须为 1**（由 1+3 推得：`block_m ≥ 16` 且 `block_m | m` 无解）。
   此时走 `m` 尾块路径（`am = m`，探针 D8 已验证 `am=1`、`bm=16` 可编译）。
5. `block_m·block_n·4 ≤ 262144`（L0C），`2·(block_m+block_n)·block_k·2·num_stages ≤ 524288`（L1）。

> **对调用方的要求（跨文件，Stage 5 接线时处理）**：`SharedExpertMLPKernel.default_config` 的
> `block_m = 128` 与 `_split_k(T ≤ 32) = 4` 组合在 `m ∈ {1, 32}` 上违反规则 3/4，会命中源实现的
> `ValueError`（`split-K basic GEMM requires exact M/N/K tiles`）。本设计在 kernel 内做规则 3 的
> 夹取（`min(block_m, m)`，探针 D7 已验证 `block_m` 由 128 夹到 32 后编译通过），但 `m = 1` 必须
> 由调用方把 `split_k` 置 1（kernel 内也可按规则 4 强制 `split_k = 1`，两条路径等价）。列入
> §9.2 E6 与 §10。

### 5.3 约束分析

- **对齐**：`block_m/block_n/block_k` 均 16 倍数；`n` 的 4 个实际取值（4096/3072/7168/5120）都被
  `block_n = 256` 整除（尾块 0）；`k` 的 5 个取值（7168/18432/2048/1536/5120）都被 64 整除；
  `m ∈ {512, 4096}` 被 128 整除（尾块 0），`m ∈ {1, 32}` 走 M 尾块 / `block_m = m`。
- **L0C**：`block_m·block_n·4 = 131072 ≤ 262144`（50%）；`bm=32` 时 32768（12.5%）。
- **L1**：`num_stages=3` → 147456（28.1%）；`num_stages=4` → 196608（37.5%）。
- **L0A/L0B**：由模板的 `TILE_K_SUB` 规则保证（L0A 32 KiB、L0B 64 KiB，各 ≤ 64 KiB）。
- **同步事件**：`num_stages = 3` 用 MTE1→MTE2 事件 0..2、`4` 用 0..3（探针实测），另加
  M→FIX 事件 0；`ascend_gemm_l1` 模板内部另占 M_MTE1 的 0/1 做 L0 双缓冲（模板文档约定，
  由编译器生成配对）。**总事件数未文档化**，故 `num_stages ≤ 4` 为设计上限（§9.1）。

### 5.4 注意事项（非整除与边界）

1. **M/N 尾块**：`am = T.min(m - m0, block_m)`、`an = T.min(n - n0, block_n)` 作为装载与存储的
   区段界；**GEMM 仍用完整 tile**（`a1`/`b1`/`c0` 的分配 extent），因为 L1-GEMM 要求 M/N 静态
   （`_l0_gemm_call` 只对 L0 输入允许符号化 region；L1 输入走 `_gemm_impl` 的静态检查）。
   越界行/列在装载后被丢弃（死行），其累加结果落在 `c0` 的死行/死列上，被存储区段截断排除。
   这正是源实现的做法（「M/N tail reads land in c_local rows/cols that the epilogue guard skips」），
   只是源用逐元素守卫、NPU 用区段截断。
   ⚠️ 死行里的旧数据/未初始化数据可能是非有限值——它只影响 `c0` 的死行，不进 `C`。
2. **K 尾块**：不允许（§5.2 规则 2）。若某 workload 的 `k % block_k ≠ 0`，必须由 host 换
   `block_k`（源实现的配置链同样如此：`block_k = 64 if k % 64 == 0 else 32 if k % 32 == 0 else 16`）。
   若连 16 都不能整除，则本 kernel 不适用（上游需 padding），列入 §9.2 E4。
3. **`m = 1`**：`block_m = 16`、`am = 1`；`asc_copy_l0c2gm` 生成 burst=1 行的存储（探针 D8
   生成 `..., 256, 1, 7168, 16, ..., quant 16, ...`）。`m = 1` 时 mt=1，权重只读一遍
   （AI ≈ 1 FLOP/B，纯带宽受限，§1.6.0）。
4. **`n` 尾块**：本 op 不出现；一般用例由 `an` 截断（探针 D9 同时含 m 与 n 尾块）。
5. **`k == 0`**：`k_tiles == 0` 时主循环零步，`c0` 由 `T.clear(c0)` 置零后按 `has_bias` 写 bias 或零
   （源语义：`k == 0` 时写 bias 或 0）。

### 5.5 分核策略（物理核数适配）

> 三要素标准：`.agents/skills/_shared/standards/core-split-strategy.md` §1。

- **① 逻辑核数计算**：`num_logical_kernels = ceil(m/block_m) × ceil(n/block_n) × split_k`
  （每个逻辑块 = 一次 block 发射，负责一个 `(block_m, block_n)` 输出 tile 的一段 K）。
  配置取自 §5.6（prefill 128×256、decode 32×256 或 16×256）：

  | workload | 调用 | `mt` | `nt` | `split_k` | 逻辑核数 | 波次数（/28） |
  |---|---|---|---|---|---|---|
  | kimi-k2-t1 | gate/up | 1 | 144 | 1 | **144** | 5.14 |
  | kimi-k2-t1 | down | 1 | 28 | 1 | **28** | 1.00 |
  | kimi-k2-t512 | gate/up | 4 | 144 | 1 | **576** | 20.57 |
  | kimi-k2-t512 | down | 4 | 28 | 1 | **112** | 4.00 |
  | ds-v3-t32 | gate/up | 1 | 16 | 4 | **64** | 2.29 |
  | ds-v3-t32 | down | 1 | 28 | 1 | **28** | 1.00 |
  | ds-v3-t4096 | gate/up | 32 | 16 | 1 | **512** | 18.29 |
  | ds-v3-t4096 | down | 32 | 28 | 1 | **896** | 32.00 |
  | glm45-t32 | gate/up | 1 | 12 | 4 | **48** | 1.71 |
  | glm45-t32 | down | 1 | 20 | 4 | **80** | 2.86 |

- **② 物理核数依据（任务平台记录，非假设值）**：目标型号 **ascend-950pr-9579**
  （架构 dav-3510）。平台 `statectl platform --task-dir <任务目录>` 落库的实查记录
  （`hardware.core_source`，读取日期 2026-11-23）：
  `Ascend950PR_9579.ini [SoCInfo] ai_core_cnt=28, cube_core_cnt=28, vector_core_cnt=56;
  runtime probe rtGetAiCoreCount()=28 on Ascend950PR_9579`，`observed_cores = {cube: 28,
  vector: 56}`。平台 ruleset 规定的查询口径是 `torch.npu.get_device_properties`
  （adapter `core_evidence()`）：

  ```python
  import torch, torch_npu
  p = torch.npu.get_device_properties(torch.npu.current_device())
  p.name, p.cube_core_num, p.vector_core_num   # -> "Ascend950PR_9579", 28, 56
  ```

  本次设计期该卡被其他算子占用（`ASCEND_RT_VISIBLE_DEVICES=""`，只做无卡编译/静态检查），
  因此**引用平台已落库的探测记录**，并在 Stage 3 首跑前用上面同一段代码复核型号与核数
  （列入 §9.1）。本算子只用 Cube，故物理核数取 **28**（Vector 56 只与 bias 分支有关）。
  逻辑核数到物理核的映射：`1 逻辑块 = 1 block 发射`，由硬件块调度器按波次派发到 28 个
  Cube 核（无 shared-memory/occupancy 约束，L1 是每核私有 512 KiB）。

- **③ 规模判定与分核方案**：逻辑核数 **28–896**，即 **1.0–32 波**，属「**中等规模（多波次）**」：
  既不是「逻辑核数 ≤ 物理核数」的单波情形（只有 kimi-k2-t1 down 与 ds-v3-t32 down 恰好 28 = 1 波，
  这两例判「自然对齐，无需适配」），也远没到必须核内串行的量级。**分核方案 = 逻辑块超发、
  硬件波次派发**，并给出两条对齐说明：
  - **对齐物理核整数倍**：`n = 7168` 时 `nt = 7168/256 = 28` **恰好等于物理核数**（ds-v3 的
    down GEMM、kimi-k2-t1 的 down），一波正好铺满 28 核、无负载不均；`n = 4096/3072/5120`
    的 `nt = 16/12/20`（split-K 提到 64/48/80）——后者为 28 的非整数倍，尾波最多空 8 核
    （glm45 gate/up：48 块 / 2 波 = 24 核/波，利用率 86%）。
  - **不做核内串行（persistent）的量化依据**：(a) 单逻辑块的工作量 = `2·bm·bn·k`：
    ds-v3-t4096 down 每块 `2·128·256·2048 = 134 MFLOP`，即使按未实测假设的单核算力 10 TFLOP/s
    也是 13 µs 量级，远大于 block 派发开销（µs 以下量级），摊薄收益可忽略；(b) 平台的
    `T.Persistent` 在知识库无正确性条目（只有安装包 README 的示例，本仓 `examples/` 无对照），
    引入它要额外承担未核验的 tile 调度语义；(c) 波次派发天然对余数负载做尾部均衡，静态 28 核
    切分反而会在 `mt·nt` 非 28 倍数时产生长尾。persistent 作为 Stage 4 的 A/B 备选（§9.3），
    若实测派发开销异常再按 `[DESIGN_LIMIT]` 流程回写。

### 5.6 每 workload 的推荐配置

（`block_k = 64`、`dtype` 按 workload；`num_stages` 以 L1 预算为约束）

| workload | 调用 | 推荐 (block_m, block_n, split_k) | 逻辑核数 | 说明 |
|---|---|---|---|---|
| kimi-k2-t1 | gate/up | (16, 256, 1) | 144 | `m=1` → `split_k` 必须 1（规则 4）；权重只读一遍，带宽受限 |
| kimi-k2-t1 | down | (16, 256, 1) | 28 | 恰好 1 波 |
| kimi-k2-t512 | gate/up | (128, 256, 1) | 576 | 算力受限；m-fastest 让 A（7.3 MB）驻留 L2 |
| kimi-k2-t512 | down | (128, 256, 1) | 112 | mt=4 → B 每 tile 被 4 个 m-block 复用 |
| ds-v3-t32 | gate/up | (32, 256, 4) | 64 | 16 → 64 逻辑块；`m % 32 == 0` 满足规则 3 |
| ds-v3-t32 | down | (32, 256, 1) | 28 | `nt=28` 已满波，split 只增部分和流量 |
| ds-v3-t4096 | gate/up | (128, 256, 1) | 512 | 算力受限；A 58.7 MB < L2 |
| ds-v3-t4096 | down | (128, 256, 1) | 896 | 32 波，`nt=28` 与物理核数整除 |
| glm45-t32 | gate/up | (32, 256, 4) | 48 | 12 → 48 逻辑块（1.71 波） |
| glm45-t32 | down | (32, 256, 4) | 80 | 20 → 80 逻辑块（2.86 波） |

`num_stages`：prefill 用 3；decode（`bm=32`）可用 3 或 4（L1 110.6 / 147.5 KB，均在预算内），
属 Stage 4 的参数级搜索空间（不改变结构）。

---
## 6. 循环与调度结构

### 6.1 循环结构总结

> 逐元素计算已按 §1.6.2 全部消除/向量化——本表只有 block 级与 tile 级循环，**无逐元素标量循环**
> （与 §1.6.2 的结论一致）。

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| 输出 tile（nt·mt·split_k 个逻辑块） | block 级并行 | `T.Kernel(nt * mt * split_k)`（一维） | 每块一个 `(block_m, block_n)` 输出 tile；平台只支持一维网格，故 `bid` 在 kernel 内分解（§1.6.3 D4 的 m-fastest） |
| K 方向（`k_slice` 次） | tile 级流水 | `T.Pipelined(k_slice, num_stages=num_stages)` | 收缩轴的循环携带累加（`c0`），用多版本 L1 与 MTE2 预取重叠；trip count 为编译期常量 |
| L1→L0A/L0B 的子 K 分块（`SUB_K = block_k / TILE_K_SUB = 1`） | tile 级（模板内部） | `ascend_gemm_l1` 模板内 `sk` 循环 + L0 双缓冲 | 由后端模板承担，本设计不显式写 |
| bias epilogue（仅 has_bias） | 元素级并行 | `with T.SimtVF(threads=128): for i, j in T.Parallel(bm/2, bn)` | lane = n 连续轴（§1.6.3 D3），每线程 2 元素 |
| （无）逐元素主循环 | — | — | A1 让 epilogue 变成一次 fixpipe DMA |

### 6.2 循环伪代码

见 §3.3（完整伪代码）。调度骨架：

```python
with T.Kernel(nt * mt * split_k) as bid:          # 一维；波次派发（§5.5）
    <bid 分解为 bz/by/bx>                          # 纯整数运算
    <alloc a1/b1 (L1) + c0 (L0C); annotate_buffer_versions>
    for step in T.Pipelined(k_slice, num_stages=ns):   # trip count 编译期常量
        <GM→L1 双操作数（nd2nz/dn2nz）>
        T.gemm(a1, b1, c0, transpose_B=True, clear_accum=(step == 0))
    <epilogue：fixpipe 直落 GM ｜ split-K 部分和 ｜ bias(双 AIV)>
```

### 6.3 流水线优化

- **版本数**：`num_stages = 3`（主选）。L1 需要 `2·(bm+bn)·bk·ns`：
  (128,256,64) → 147456 B（28.1%）；`ns = 4` → 196608 B（37.5%），仍在 512 KiB 内。
- **必须显式声明版本**：`T.annotate_buffer_versions({a1: ns, b1: ns})`。探针对照（同一
  (512,4096,7168) 配置）显示：
  - 不声明时调度器自选版本 —— `b1` 走 `step & 1`（2 版本）、`a1` 走 `step % 3`（3 版本），
    预算需要读生成代码才能确定；
  - 声明 `ns = 3` 后两者都是 `step % 3` 的 3 版本环，`ns = 4` 时是 `step & 3` 的 4 版本环
    + 4 组 MTE1↔MTE2 事件。
  显式声明的收益是**L1 预算可静态核算**（§4.5 的表因此成立），并且与已核验条目
  （`PL-950-gemm-tiles`）的写法一致；代价是失去调度器的逐 buffer 最优版本选择（其收益未实测）。
- **L0 双缓冲**：由 `ascend_gemm_l1` 模板内部用 M_MTE1 事件 0/1 实现（模板文档约定
  "Pre-set these flags exactly once … Drain them exactly once"，实际配对由 InsertSync 生成）；
  本设计不介入。
- **Stage 4 旋钮**：`num_stages ∈ {2,3,4}`、`block_n ∈ {128,256}`、`block_m ∈ {64,128}` 属参数级
  搜索空间（不改变 §3.3 结构）；`num_stages > 4` 不列入（事件数上限未文档化，§5.3）。
- 不使用 `T.annotate_unlimit_memory`（不解除容量约束，避免把溢出推到运行期）。

### 6.4 尾块处理

见 §5.4：M/N 尾块 = 装载/存储区段截断（`T.min`），GEMM 用完整静态 tile（死行/死列被截断
排除）；K 尾块**不允许**（前置 `k % block_k == 0`）；`k == 0` 走 `T.clear` + 零/bias 写。
整除时（本 op 的 8/10 次调用 m 与 n 都整除）这些 `T.min` 与守卫在 Python 层被整段消掉
（`exact_m/exact_n`），生成代码里不出现多余的边界比较（探针 D1 与 D9 的代码差异即此）。

---

## 7. 同步策略

### 7.1 同步模式

**模式：自动同步。** 依据：平台 `ascend/pipeline.py` 的 `allow_autoschedule` 默认开启，
流水线依次执行 `AutoSchedule → AssignCore → PrepareMultiBuffer → ResolveCore → InsertSync →
MaterializeMultiBuffer`，由编译器在 MTE2/MTE1/M/FIX/V/MTE3 队列之间插入成对事件；
平台规则明确「手工跨核 flag 配对、循环复用下的复杂同步仍未认证」，因此本设计
**不写任何 `ascend_set_flag` / `ascend_wait_flag` / `ascend_pipe_barrier` / `ascend_get_buf`**。

### 7.2 同步点说明

| 位置 | 同步方式 | 理由（生成代码证据） |
|------|----------|---------------------|
| GM→L1 装载 ×2 → GEMM 读 L1 | 自动（MTE2→MTE1 事件，按版本号） | 探针 D1：`asc_sync_notify(PIPE_MTE2, PIPE_MTE1, event(step % 3))` + 对应 `wait` |
| GEMM 读毕 L1 → 下一轮覆写同名版本 | 自动（MTE1→MTE2 事件） | 同上（`asc_sync_notify(PIPE_MTE1, PIPE_MTE2, event(step % 3))`） |
| 循环首轮前的事件预置 / 末轮后的排空 | 自动 | 探针 D1：循环前 3 个 `notify(PIPE_MTE1, PIPE_MTE2, 0/1/2)`，循环后 3 个 `wait` |
| MAD 完成 → fixpipe 出数 | 自动（M→FIX 事件 0） | 探针 D1：`if (step == 111) notify(PIPE_M, PIPE_FIX, 0)` + 循环后 `wait(PIPE_M, PIPE_FIX, 0)` |
| L0C→UB（bias 分支）→ Vector 读 UB | 自动（FIX→V 事件，`asc_sync_intra_arrive/wait`） | 探针 D10 |
| Vector 写完 UB → MTE3 出 GM | 自动（V→MTE3 事件） | 探针 D10：`notify(PIPE_V, PIPE_MTE3, 0)` + `wait` |
| L0A/L0B 双缓冲（模板内部） | 模板自带 M_MTE1 事件 0/1（预置/排空由 InsertSync 生成） | `gemm.h` 的 Event ID contract 注释 |
| 跨 block（split-K 写不同 `C[bz]` 切片） | **无需同步**（写不重叠的切片） | 语义上无依赖 |

### 7.3 pass_configs 配置

```python
# 保持默认：不关闭自动调度、不改共享内存复用策略（平台规则：allow_autoschedule 默认开启）
pass_configs = {}                       # 源实现的 {"tl.disable_warp_specialized": True} 不适用（§0.5 O11）
compile_flags = ["-O3"]                 # 源实现的 -DENABLE_BF16 由 ascend 的 dtype 路径取代
```

若 Stage 3 出现同步相关失败，回退顺序：① `num_stages` 降到 1~2 并把版本注解同步调小；
② 去掉 `annotate_buffer_versions` 让调度器自选版本；③ 才考虑显式 `T.Cube()`/`T.Vector()` +
手工 sync（该形态只有单 tile 核验，见 `PL-950-manual-sync`）。

---

## 8. 验证方案

### 8.1 Golden 函数

> 依据 **§0.1 的语义**独立实现（`C = op(A)·op(B) + bias`，fp32 累加、末次舍入、split-K 契约），
> **不复刻** §0.6 的分块算法：不预切块、不写部分和、不做硬件量化、不按 tile 遍历。

```python
import torch

def golden_gemm_basic(a, b, trans_a=False, trans_b=False, bias=None, out_dtype=None):
    """C = op(A) @ op(B) [+ bias];  fp32 accumulation, one final rounding."""
    ah = a.float().t() if trans_a else a.float()          # (m, k)
    bh = b.float().t() if trans_b else b.float()          # (k, n)
    acc = ah @ bh                                          # fp32 matmul
    if bias is not None:
        acc = acc + bias.float().unsqueeze(0)              # 列广播，fp32 加
    return acc.to(out_dtype or a.dtype)


def golden_gemm_basic_splitk(a, b, split_k, trans_a=False, trans_b=False):
    """split_k > 1：返回 [split_k, m, n] fp32 部分和（与 C 的输出契约一致）。"""
    ah = a.float().t() if trans_a else a.float()
    bh = b.float().t() if trans_b else b.float()
    k = ah.shape[1]
    assert k % split_k == 0
    ks = k // split_k
    return torch.stack([ah[:, s * ks:(s + 1) * ks] @ bh[s * ks:(s + 1) * ks, :]
                        for s in range(split_k)])
```

端到端（`split_k > 1`）对照时，golden 侧按**下游归约语义**求和（`parts.sum(0)`，可选加 bias 与
`silu_and_mul`），归约 kernel 本身是另一个设计任务。golden 与 §8.1 的语义一致、与 §0.6 的
NPU 结构无关，满足验证独立性。

### 8.2 精度标准

| dtype | atol | rtol | 附加判据 | 来源 |
|-------|------|------|----------|------|
| float16 / bfloat16 | 1e-2 | 1e-2 | 归一化平方误差 ≤ 1e-4 | Op 级用例的策略（`examples/TileOPs/tests/ops/test_fused_moe_shared_expert.py` 用 `tileops.workloads.moe.moe_close`：elementwise `atol = rtol = 1e-2` + NSE ≤ 1e-4） |
| float32（`split_k > 1` 的部分和 / `out_dtype=float32`） | 1e-4 | 1e-4 | 参考量取「量化输入 → fp64 累加」，判据为最大 ulp 差 ≤ 1（fp32 输出的 1 ulp） | 平台惯例（`PL-950-gemm-tiles`/`PL-950-cube-tile` 的 atol=rtol=1e-4） |

参考量：fp16/bf16 输入先转 fp32 再做 matmul（与已核验条目的参考口径一致），避免把输入量化
误差记到 kernel 头上。

### 8.3 L0 门槛测试计划

L0 = 小 shape、覆盖每个结构分支（Stage 3 首跑必须全过；完整 L1/L2/boundary 套件由
`tilelang-op-develop` 依 §5.6 与 manifest 展开）：

| 用例 | (m, n, k) | dtype | 配置 | 覆盖点 | 判定 |
|------|-----------|-------|------|--------|------|
| L0-1 | (32, 64, 128) | bf16 | bm=32, bn=64, bk=64, ns=2 | 最小可用 tile；`quant_pre = 16`；3 版本以外的 ns=2 路径 | vs golden，atol=rtol=1e-2 |
| L0-2 | (32, 64, 128) | fp16 | 同上 | `quant_pre = 1`（F322F16） | 同上 |
| L0-3 | (1, 128, 256) | bf16 | bm=16, bn=128 | `am = 1` 的 M 尾块 + fixpipe 单行 burst（探针 D8 形态） | 同上 |
| L0-4 | (32, 64, 256) | bf16 | bm=32, bn=64, split_k=4 | split-K 部分和 `[4,32,64]` fp32 与 `golden_gemm_basic_splitk` 逐片比对 | fp32 ulp ≤ 1 vs fp64 参照 |
| L0-5 | (32, 64, 128) | bf16 ×4 | bm=32, bn=64 | `(trans_a, trans_b)` 四种组合的装载归一（dn2nz 计数 1/1/2） | vs golden，1e-2 |
| L0-6 | (32, 64, 128) | bf16 | bm=32, bn=64, has_bias=True | Cube+Vector bias epilogue（`dual_copy` + `SimtVF`） | vs golden（bias 在 fp32 上加），1e-2 |
| L0-7 | (100, 1000, 128) | bf16 | bm=32, bn=128 | M 与 N 同时有尾块（`am=4`/`an=104`），区段截断写 | vs golden，1e-2 |
| L0-8 | (16, 64, 0) | bf16 | bm=16, bn=64 | `k == 0`：写零 / 写 bias（源语义） | 精确比对 |
| L0-9 | (32, 7168, 2048) | bf16 | bm=32, bn=256, split_k=4 | **满 K（32 个 tile）**的分片累加 + 目标 shape 覆盖（decode down 的真实几何） | vs golden 端到端 `parts.sum(0)`，1e-2 |

门槛：L0-1…L0-9 全过 → `[PRECISION_PASS]` 的最低要求；L0-9 另外记录 kernel 级时延（Stage 4 基线）。
若 L0-1/L0-2 出现「目标 dtype 差 > 1 ulp」，触发 §9.3 第 1 条的回退（改用 D2② 的 UB epilogue）。

---

## 9. 风险点与注意事项

### 9.1 已知约束

1. **后端硬约束**（不可绕过，设计已适配）：L1-GEMM 仅 NT（`gemm_mad.py` assert）；`block_m/
   block_n` 必须 16 倍数（`gemm.h` static_assert）；`TILE_K_SUB | block_k` 且由 L0 容量反推
   （`_compute_tile_k_sub`）；一维网格、无 `threads`；L0 区段必须零起点（`CheckCompactL0Region`）。
2. **未实测常数 / 未核验行为（待确认，交 Stage 3/4）**：
   - `quant_pre` 的**运行期舍入模式**：编译期只证明 `quant_mode = 16/1` 被生成
     （`GetCCQuantPre` 的映射），实际舍入（RN/截断）与本平台数值行为未实测 —— 若为截断，
     误差 ≤ 1 个目标 dtype ulp（≤ 1e-3 相对），仍在 `rtol = 1e-2` 内；L0-1/L0-2 会直接暴露。
   - **事件（flag）总数上限**未文档化，故 `num_stages ≤ 4`。
   - **UB 可分配额度**：型号档案标注「UB 在运行时扣除系统预留后可分配的实际额度仍需实测
     确认」；主选路径不用 UB，只有 bias 分支受影响（98.8 KB / 248 KiB，裕度充足）。
   - `T.Persistent` 无平台正确性条目（只列为备选）。
3. **环境**：本次设计期卡被其他算子占用（`ASCEND_RT_VISIBLE_DEVICES=""`），只做无卡编译与
   静态核对；Stage 3 首跑前须用 `torch.npu.get_device_properties` 复核型号与核数（§5.5②）。
4. **性能收益未实测**：本文的全部容量/流量数字是静态估算；`bottlenecks.md` 明确「当前没有
   950 真机验证的瓶颈模式或收益阈值」，故 §1.6.0 的下界行按「待核验」记录（缺带宽/算力常数）。

### 9.2 常见错误

| # | 错误 | 触发场景 | 影响 | 解决方案 |
|---|------|----------|------|----------|
| E1 | `T.gemm requires static tile dimensions` / L1-GEMM NT 断言 | 把 `c0[:am,:an]` 这类符号化 region 传给 **L1 输入**的 `T.gemm`；或传 `transpose_A=True` | 编译失败 | GEMM 恒用完整静态 tile（`a1`/`b1`/`c0`），只对**装载/存储**用区段截断（§5.4 第 1 条）；NT 由装载归一保证 |
| E2 | `assert tile_k_sub > 0 / K % TILE_K_SUB == 0` | `k % block_k ≠ 0`，或 `block_k` 不是 16 的倍数 | 编译失败 | 前置 `k % block_k == 0` 且 `block_k % 16 == 0`；host 按 `k` 的因子选 `block_k`（64/32/16） |
| E3 | `static_assert(M % FRAC_N == 0)` | `block_m`/`block_n` 取了非 16 倍数（如 8、24） | 编译失败 | 编译期 assert（§5.2 规则 1） |
| E4 | L0C 溢出 | `block_m·block_n > 65536`（如 256×512） | 编译失败或运行异常 | 预算断言（§5.2 规则 5）；本 op 最大 (128,256) |
| E5 | L1 溢出 | `num_stages` 或 `block_n` 过大（如 (128,256,64) 取 `ns = 8` → `2·384·64·2·8 = 768 KB`） | 编译失败 | `2·(bm+bn)·bk·ns ≤ 524288` 断言；主选 147456 B |
| E6 | `split-K basic GEMM requires exact M/N/K tiles` | `split_k > 1` 但 `m % block_m ≠ 0`（**本 op 的 `m = 1/32` + 调用方默认 `block_m=128` 就会命中**） | 构造期 `ValueError` | kernel 内 `block_m := min(block_m, m)`；`m < 16` 强制 `split_k = 1`；调用方（`SharedExpertMLPKernel`）的 `_split_k` 需按 §5.6 表分档（§10 交付说明） |
| E7 | 死行非有限值扩散的误判 | 尾块的死行累加了未初始化 L1（可能是 inf/NaN） | 只影响 `c0` 的死行，不进 `C`；但若有人为了省事把死行写回 GM（去掉区段截断）就会污染输出 | 严禁去掉 `[:am,:an]` 存储截断；L0-3/L0-7 专门覆盖 |
| E8 | 同步失败（事件错配） | 手工写 `ascend_set_flag/wait_flag`，或 `num_stages` 超过事件预算 | 结果错 / 死锁 | 只用自动同步（§7）；回退阶梯见 §7.3 |
| E9 | 把 NPUIR 的 API 用到本后端 | 照抄 `examples/gemm/matmul.py` 的 `T.load_nd2nz`/`T.store_fixpipe`/`initC=`/`size=` | `AttributeError` / 参数错 | 本平台用 `tilelang.ascend.language`；`T.gemm` 用 `clear_accum=`/`transpose_B=`（§3.4 已逐条核对） |

### 9.3 特殊场景处理

1. **`has_bias` 分支（本 op 不使用，但工厂契约要求）**：走 D2② 的混合 epilogue。风险点是
   `T.copy(bias[n0:n0+bn], bv)` 这个 GM→UB 拷贝在 `__mix__` kernel 里由哪个 AIV 执行——
   若编译器只给一个 AIV 执行，另一 AIV 的 `bv` 会是垃圾。**触发条件**：L0-6 精度失败。
   **回退**（按代价排序）：① 把 bias 切片也做 `dual_copy`（若 shape 允许 2:1）；② 用
   `T.copy(..., sub_blockid=sid)` 在显式 `T.Vector(vector=2)` 作用域内按 AIV 各装一份
   （`copy_op.py` 文档的 `sub_blockid` 参数，需 auto-schedule 关闭，属未核验形态）；
   ③ 把 bias 留给下游 kernel（等价性不变，契约上 bias 只在 `split_k == 1` 出现，需上游同意）。
2. **persistent 备选**：若 Stage 4 实测小 shape（glm45 gate/up 48 块 / 1.71 波）出现派发开销
   异常，按 `T.Persistent([nt*mt*split_k], 28, bid)` 改写（安装包 README 形态），此时 §5.5 的
   核内串行边界须重算（每核处理 `ceil(逻辑块/28)` 个 tile，循环界静态）。
3. **`num_stages = 1`**：退化为无多缓冲（仍正确），用于排查同步问题（§7.3）。
4. **`block_n = 128`**：`nt` 翻倍（decode gate/up 由 16→32），逻辑核数上升、L1 下降
   （`2·(32+128)·64·3 = 61440` B）；属参数级搜索，不改变结构。
5. **`k == 0`（L0-8）**：本设计用 `T.clear(c0)` 把 L0C 累加器置零。`T.clear` 在 ascend 的 L0C
   上属**未核验**形态（本 op 的 k ≥ 1536，不会触发）。若 Stage 3 发现它不被接受，回退为
   「跳过 GEMM 与 fixpipe，用 bias 分支的 UB 路径直接写零/bias」（语义等价，代价只在该边界
   用例上）。

---

## 10. 交付清单

### 10.1 目录结构

```
examples/shared_expert_mlp/_gemm_basic_kernel/
├── DESIGN.md                              # 本文
├── verify_equiv.py                        # §1.6.1 的等价性机器验证（已执行，10/10 EQUIV_PASS）
├── history_version/
│   └── design_probe_compile.py            # §11.1 的编译探针（12/12 编译通过，无卡）
├── RETROSPECTIVE.md                       # Stage 1 复盘（自进化钩子）
└── _gemm_basic_kernel.py                  # 待 Stage 3 实现
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | ✅ 已完成 | 本设计文档 |
| `verify_equiv.py` | ✅ 已执行 | 三项采纳优化的位级/ulp 对照，结果表内嵌 §1.6.1 |
| `history_version/design_probe_compile.py` | ✅ 已执行 | 12 个结构变体的编译探针（无卡），结论见 §11.1 |
| `RETROSPECTIVE.md` | ✅ 已追加 | Stage 1 复盘 |
| `_gemm_basic_kernel.py` | ⬜ 待实现 | Stage 3：kernel + golden + 分层测试（§8.3） |
| `tests`/`bench` | ⬜ 待下游 | 复用 `examples/TileOPs/tests/ops/test_fused_moe_shared_expert.py` 与 `benchmarks/ops/bench_fused_moe.py` |

### 10.3 命名规范

- 实现文件 `_gemm_basic_kernel.py`（与源函数同名，保持 TileOPs 七文件脚手架的 import 契约：
  `from ._shared_expert_mlp_fwd_kernels import _gemm_basic_kernel` 的替代实现必须保留
  `_gemm_basic_kernel(m, n, k, trans_a, trans_b, dtype, has_bias, out_dtype)` 的外层签名与
  内层 `(block_m, block_n, block_k, num_stages, split_k)` 的**去 threads** 签名（K9））。
- 测试文件 `test__gemm_basic_kernel.py`（可选，放入该算子目录；Op 级用例已存在）。

### 10.4 交付给下游的额外说明（Stage 3 / Stage 5）

1. **签名**：外层 8 参 + 内层 5 参（去掉 `threads`）；`out_idx` 语义不变（`split_k == 1` →
   `[m,n]`、`split_k > 1` → `[split_k,m,n]` fp32）。
2. **调用方配置**：`SharedExpertMLPKernel.default_config` 的 `block_m = 128` 与
   `_split_k(T ≤ 32) ∈ {4}` 在 `m ∈ {1,32}` 上不满足精确 tile 前置（§5.2 规则 3/4）。
   本 kernel 已内建 `min(block_m, m)` 夹取与 `m < 16 → split_k = 1` 强制，但**建议** Stage 5
   接线时把 decode 段的 `block_m` 显式降到 32/16（§5.6 表），以免依赖 kernel 侧的兜底。
3. **`num_stages`/`block_n`** 的最终取值由 Stage 4 参数级搜索决定，结构不变。
4. 不要把 §8.3 的 L0 用例与 manifest 的大 shape 混在一层跑（大 shape 属 L1/L2）。

---

## 11. 设计期自检与探针记录

### 11.1 编译探针（D-3，无卡）

脚本 `history_version/design_probe_compile.py`，执行方式
`ASCEND_RT_VISIBLE_DEVICES="" python history_version/design_probe_compile.py`
（平台 `execution` 上下文为 device/precision 模式；本探针只做**编译**，不创建 NPU 张量、
不运行 kernel，故不占用设备）。**12/12 用例编译通过**：

| 用例 | 覆盖的设计决策 | 关键代码生成事实 |
|---|---|---|
| D1 prefill t4096 NT direct bf16 | 主选路径全链路 | `__cube__`；`ascend_gemm_l1<128,64,256,64,true,bfloat16_t>`；`asc_copy_l0c2gm(..., asc_quant_mode(16))`；L1 `step % 3` 环 |
| D2 prefill fp16 | dtype 分支 | `half` 模板；`quant_mode(1)` |
| D3 TN（trans_a+trans_b） | R1 装载归一（A 走 dn2nz） | `nd2nz ×1 + dn2nz ×1` |
| D4 NN（trans_b=False） | R1（B 走 dn2nz） | `nd2nz ×1 + dn2nz ×1` |
| D5 TT（两者都转置） | R1（两者 dn2nz） | `dn2nz ×2` |
| D6 split-K m=32 | R6 部分和落盘 | `quant_mode(0)`（fp32→fp32）；`ascend_gemm_l1<32,...>` |
| D7 split-K + `block_m` 夹取 128→32 | §5.2 规则 3 | 编译通过，`ascend_gemm_l1<32,...>` |
| D8 m=1 M 尾块 bm=16 | §5.4 第 3 条 | `asc_copy_l0c2gm(..., 256, 1, 7168, 16, ..., quant 16, ...)`（1 行 burst） |
| D9 m 与 n 双尾块 (100,1000,128) | §5.4 第 1/4 条 | 编译通过，无逐元素守卫 |
| D10 has_bias | R5 混合 epilogue | `__mix__(1,2)`；`asc_copy_l0c2ub(..., asc_dual_dst_mode(1))`；SIMT VF 用 `float2` + `__float22bfloat162_rn` |
| D11 ns=4 decode bm=32 | §6.3 版本环 | `step & 3` + 4 组 MTE1↔MTE2 事件 |
| D12 out_dtype=fp32 | `out_dtype` 分支 | `quant_mode(0)` |

探针同时否定了三条候选：`T.gemm` 的 L1 输入**不接受**符号化 region（`C is symbolic: am`）、
`dual_copy` 的 N-split 在 region extent 符号化时无法推断（`[rows,128] vs [rows, an]`）、
以及 `T.Kernel` 只有一维（多维 grid 需自行展平）。

### 11.2 等价性机器验证（D-1）

`python verify_equiv.py` → 10/10 `EQUIV_PASS`（结果表见 §1.6.1）。要点：A1（融合量化落盘）
与 A3（NT 装载归一）在全部用例上位级相等（含 ±inf/NaN/subnormal/±0 的角点用例）；A2（split-K）
在 K = 7168、split = 4、宽动态范围下目标 dtype 输出 **0 ulp 差、违反率 0**，fp32 层重结合足迹
≤ 4864 fp32 ulp（≈6e-4 相对）≪ `rtol = 1e-2`。

### 11.3 门禁自检

`python3 .agents/tools/statectl.py gate 1 --dir examples/shared_expert_mlp/_gemm_basic_kernel`
（只读）输出：

```json
{"action": "gate", "stage": 1, "ok": true, "pass": true, "failures": [], "warnings": []}
```

机械项（章节齐全、调研四问 + 基线 + 结论行、`EQUIV_PASS` 结果表与脚本存在、分核三要素、
Golden/L0 计划、引用路径存在、无占位符）逐项通过；`design_calc_check.py` 的
L1/L0C/L0A/L0B 预算与分核复算、R3 四口径也全部 `pass`。

### 11.4 回退说明

**首次设计（v1），无上一版**，故无「相对上一版的关键调整」。后续 revision 时本节改为
「相对 v1 的关键调整 + 为何不会再犯同一错误」。

---

## 附：本文引用的平台条目与证据索引

| 证据 | 位置 | 用途 |
|---|---|---|
| 平台容量/核数/预算 | `statectl platform --task-dir`（2026-11-23）、`.agents/platforms/hardware/ascend-950pr-9579-notes.md` | §4.5、§5.5、§9.1 |
| API 规则 | `.agents/platforms/toolchains/ascend/references/programming.md` | §3.4、§7 |
| 已核验 GEMM 模式 | `.agents/platforms/toolchains/ascend/knowledge/gemm-tiles.md`、`.agents/platforms/toolchains/ascend/knowledge/PL-950-cube-tile.md` | §0.5、§3.5.3、§6.3 |
| 混合 epilogue 模式 | `.agents/platforms/toolchains/ascend/knowledge/execution-patterns.md` | §0.6 R5、§3.5.3 |
| 瓶颈资料（无实测阈值） | `.agents/platforms/toolchains/ascend/knowledge/bottlenecks.md` | §1.6.0 下界行、§9.1 |
| 分核三要素标准 | `.agents/skills/_shared/standards/core-split-strategy.md` | §5.5 |
| roofline 估算协议 | `.agents/skills/_shared/standards/hardware-cost-model.md` | §1.6.0 |
| 后端实现（仅安装包，非本仓路径） | `tilelang/ascend/op/gemm/gemm_mad.py`、`tilelang/ascend/op/copy.cc`、`tilelang/src/tl_templates/ascend/gemm.h`、`tilelang/ascend/language/copy_op.py` | §0.5、§0.6、§3.4、§9.1 |
