# `splitk_reduce_kernel` 算子设计文档

> **迁移任务（harness 子模式），首次设计 v1**。源算子：
> `/workspace/TileOPs/src/tileops/_kernels/gemm/dense.py` 的 `splitk_reduce_kernel`
> （GPU TileLang，CUDA target，约第 1987 行起；docstring 称其为「split-K epilogue」）；同仓抽取件
> `examples/TileOPs/tileops/kernels/moe/shared_expert_mlp/_shared_expert_mlp_fwd_kernels.py`
> （第 308–374 行）为其逐字副本。调用方（GPU 侧 Kernel 类）见
> `examples/TileOPs/tileops/kernels/moe/shared_expert_mlp/shared_expert_mlp.py` 的 `_gemm_program`。
>
> 目标平台：`ascend-950pr-9579`（dav-3510，28 Cube / 56 Vector，UB 248 KiB / L1 512 KiB /
> L0C 256 KiB，L2 128 MiB），工具链 `ascend`（TileLang ascend 0.1.15 + CANN 9.2.0 编译 SDK /
> 9.1.0 runtime，`target="ascend"`，`import tilelang.ascend.language as T`）。平台选择由
> `statectl platform --task-dir` 落库（2026-11-23）。
>
> **设计范围**：仅 `splitk_reduce_kernel`（split-K 归约 + 可选 bias + 可选 `silu_and_mul`
> epilogue）。同批次的 `_gemm_basic_kernel`（dense GEMM 主循环 + 部分和落盘）与
> `_silu_mul_fused_kernel`（独立激活 kernel）是另两个设计任务；本文只在**输入契约**上依赖前者
> （§0.1、§4.1）；与后者的关系是「本 kernel 在 decode 段取代它」——wrapper 的 `_decompose` 在
> `gate_up_split == 1` 时起独立激活 kernel，`split_k > 1` 时把 `silu_and_mul` 融进本 kernel
> 并把 `activated` 置真。GPU 的 SM90/grouped
> 变体与 `GEMMTMAKernel` 系在 Stage 0 脚手架已按「NPU 无对应物」丢弃，本文不重复论证。
>
> 本文档由 `tilelang-ascend` → `tilelang-op-design` 流程产出：Phase M0（三问解读）→
> Phase R（算法调研，§1.6.0）→ Phase M1（耦合性判定与重设计，§0.5–§0.6）→ Phase 2
> （算法级优化，§1.6.1–§1.6.3）。首次设计，无「相对上一版的关键调整」（§11.4）。

---

## 0. 源算子解读与迁移分析

### 0.1 源算子语义（做什么）

`splitk_reduce_kernel(split_k, m, n, dtype="float16", activation="none", has_bias=False,
out_dtype=None)` 是**工厂函数**：外层接收编译期参数，返回内层 JIT 工厂
`_splitk_reduce_func(elems_per_cta=1024)`，内层再专门化出一个 `T.prim_func`。

**数学语义**（`dense.py:1998-2050`；`accum_dtype = "float"`）：

```
非激活分支（activation == "none"）：
    c[i, j] = cast_{c_dtype}( Σ_{s=0}^{split_k-1} w[s, i, j]  +  [has_bias] cast_f32(bias[j]) )
门控分支（activation == "silu_and_mul"，要求 n 为偶数，out_n = n/2）：
    gate[i, j] = Σ_s w[s, i, j]            + [has_bias] cast_f32(bias[j])
    up[i, j]   = Σ_s w[s, i, out_n + j]    + [has_bias] cast_f32(bias[out_n + j])
    c[i, j]    = cast_{c_dtype}( gate[i,j] * sigmoid(gate[i,j]) * up[i,j] )
其中 c_dtype = out_dtype or dtype；i ∈ [0,m), j ∈ [0,out_n)
```

**I/O 契约（对上游 `_gemm_basic_kernel` 的硬依赖）**：

- `w` = `[split_k, m, n]` **fp32 部分和**，由 `_gemm_basic_kernel` 的 `split_k > 1` 分支产出
  （该 kernel 的 `output_shape = (split_k, m, n)`、`output_dtype = accum_dtype = "float32"`，
  见 `examples/shared_expert_mlp/_gemm_basic_kernel/DESIGN.md` §4.2）。**本 kernel 不得假设
  任何其它布局**（无零填充、无 slide 轴、无 per-slice 偏移）。
- `bias` = `[n]`，dtype = **输入 dtype**（fp16/bf16，不是 fp32），仅在 `has_bias` 时出现在签名中；
  它在 fp32 域里被加（`T.cast(bias[gj], accum_dtype)`）。
- `C` = `[m, out_n]`，dtype = `out_dtype or dtype`。**`C` 是参数而不是 `out_idx` 自动分配的
  输出**（源码 docstring：「the caller allocates it before the mainloop, so no allocation sits
  between the two launches」）。
- `n` 是**完整宽度**（门控时 `2·ffn`），`out_n = n // 2` 只在门控分支生效。

**规约语义（累加顺序）**：

- 求和沿**切片轴 `s` 升序**：`w[0] + w[1] + … + w[split_k-1]`（`_slice_sum` 用 Python 循环从
  `w[0]` 开始逐项 `expr = expr + w[s]`，左结合），**bias 在切片和之后加**。
- 全部在 fp32 中累加，**只有一处舍入点**：写入 `C` 时 `T.cast(..., c_dtype)`。
- 门控分支的 `gate`/`up` 各自是一条独立切片和链，**两者都含 bias**，然后才做
  `gate·sigmoid(gate)·up`。

**dtype 语义**：`w` 恒 fp32；bias 取输入 dtype 并在加法前 `cast` 到 fp32；中间量全 fp32；
输出一次舍入。`out_dtype` 可与 `dtype` 不同（例如输入 bf16、输出 fp32）。

**边界语义**：

- 平坦化范围 `total = m * out_n`，逐元素守卫 `idx < total`（`idx = base + t`，
  `base = bx * elems_per_cta`）；最后一块的越界元素**不写**。
- `activation ∉ {none, silu_and_mul}` → Python 层 `ValueError`；门控且 `n` 为奇数 →
  `ValueError("silu_and_mul requires an even output width")`。
- `split_k == 1` 是**合法输入**（退化为「加 bias + 可选 silu」的纯逐元素 kernel，甚至无 bias 时
  就是 fp32→c_dtype 的纯转换）。
- 非有限值：无特殊处理，按 IEEE 传播（`gate = -inf` 时 `sigmoid(-inf) = 0`，
  `-inf · 0 = NaN`，与 torch 的 `x*sigmoid(x)` 同式；§1.6.1 的角点用例实测两式非有限模式一致）。

**host 侧逻辑**：`_main`（`dense.py:175-239`）只做签名装配（有/无 bias 两种 `global_symbol`）
与 `out_idx` 分配；shape/`split_k` 校验全在 Python 层（编译期常量判定）。无可变数据的预处理
/后处理，无 im2col、无 permute。

### 0.2 源算子输入输出

| 参数 | 方向 | Shape | dtype | 说明 |
|---|---|---|---|---|
| `w` | in | `(split_k, m, n)` | float32 | 上游 split-K GEMM 的部分和（`_gemm_basic_kernel` §4.2 的契约） |
| `bias` | in | `(n,)` | float16 / bfloat16 | 仅 `has_bias` 时出现；门控时前半段给 gate、后半段给 up |
| `C` | out | `(m, out_n)` | `out_dtype or dtype` | `out_n = n//2`（门控）或 `n`；源算子输出 shape：`C [m, n]`（非门控） |

**本 op（`SharedExpertMLPFwdOp`）中 `splitk_reduce_kernel` 的真实调用**：每次 forward 由
`_gemm_program` 在 `split_k > 1` 时构造一次 reduce（`shared_expert_mlp.py` 的
`_gemm_program`：`splitk_reduce_kernel(split_k, m, n, dtype_str, activation)(1024)`，
**`has_bias` 与 `out_dtype` 都取默认值**，即 `has_bias=False`、`out_dtype=None`）。
按 manifest `examples/TileOPs/tileops/manifest/moe.yaml` 的 `SharedExpertMLPFwdOp` 与调用方
`_split_k`（`num_tokens ≤ 32` 且 `k_tiles ≥ 2` 时才拆 K）：

| workload | 调用 | m | n | dtype | split_k | activation | has_bias | 本 kernel 是否执行 |
|---|---|---|---|---|---|---|---|---|
| kimi-k2-t1 | gate/up | 1 | 36864 | bf16 | **1**（Stage 5 修复后 `m<16 → 1`） | — | — | 否（退化路径仍需正确） |
| kimi-k2-t1 | down | 1 | 7168 | bf16 | 1 | — | — | 否（同上） |
| kimi-k2-t512 | gate/up / down | 512 | 36864 / 7168 | bf16 | 1 | — | — | 否 |
| **ds-v3-t32** | **gate/up** | 32 | 4096 | bf16 / fp16 | 4 | `silu_and_mul` | False | **是** |
| **ds-v3-t32** | **down** | 32 | 7168 | bf16 / fp16 | 4 | `none` | False | **是** |
| ds-v3-t4096 | gate/up / down | 4096 | 4096 / 7168 | bf16 | 1 | — | — | 否 |
| **glm45-t32** | **gate/up** | 32 | 3072 | bf16 | 4 | `silu_and_mul` | False | **是** |
| **glm45-t32** | **down** | 32 | 5120 | bf16 | 4 | `none` | False | **是** |

（`m` = token 数 `T`；gate/up 的 `n = 2S`，down 的 `n = H`。`has_bias=True` 与
`out_dtype="float32"` 在本 op 中不出现，但属 kernel 的公开签名，必须正确 —— 两者都进
§8.3 的 L0 用例。）

**已知调用方缺陷（Stage 5 修复，不在本设计范围）**：现行 wrapper 的
`_split_k(T≤32)=4` 在 `m < 16` 时会让 `_gemm_basic_kernel` 违反「`block_m | m` 且 `m ≥ 16`」
的精确 tile 前置（见 `examples/shared_expert_mlp/_gemm_basic_kernel/DESIGN.md` §9.2 E6）。
Stage 5 将改为 `num_tokens < 16 → 1`。**本 kernel 的义务**：`split_k == 1` 时必须退化为
「纯拷贝 / 加 bias / 可选 silu」的正确语义（§0.6 R6、§8.3 L0-4/L0-5）。

### 0.3 实现算法解读（怎么算）

**源码结构化阅读**（`dense.py:1987-2050`）：

1. **入口与构建期常量**：`accum_dtype="float"`、`c_dtype = out_dtype or dtype`、
   `gated = activation == "silu_and_mul"`、`out_n = n//2 if gated else n`；两个 Python 层拒绝
   （未知 activation、门控奇数宽）。
2. **内层工厂**：`_splitk_reduce_func(elems_per_cta=1024)` 决定「每块处理 1024 个输出元素」。
   装饰器 `@tilelang.jit(pass_configs={TL_ENABLE_FAST_MATH: gated}, compile_flags=["-O3",
   "-DENABLE_BF16"])` —— fast-math **只在门控分支**开启。
3. **Kernel 结构**：`with T.Kernel(T.ceildiv(total, elems_per_cta), threads=256) as bx` ——
   一维 grid，每块 1024 个**扁平**输出元素、256 线程。
4. **元素映射**：`idx = bx·1024 + t`；`gi = idx // out_n`；`gj = idx % out_n`；
   `if idx < total:` —— **逐元素整除/取模 + 逐元素守卫**。
5. **累加**：`_slice_sum(w, bias, gi, gj)` 在 Python 层展开成
   `w[0,gi,gj] + w[1,gi,gj] + … [+ cast_f32(bias[gj])]` 的表达式树。
6. **门控 epilogue**：`gate = _slice_sum(…, gj)`；`up = _slice_sum(…, out_n+gj)`；
   `c[gi,gj] = T.cast(gate * T.sigmoid(gate) * up, c_dtype)`。
7. **写回**：`c[gi, gj] = …`，逐元素单标量存。

**计算步骤分解**（覆盖源码全部计算语句）：

| # | 步骤 | 源码语句 | 备注 |
|---|---|---|---|
| 1 | 块号 → 元素区间 | `base = bx * elems_per_cta` | 块级标量 |
| 2 | 元素 → (行, 列) | `gi = idx // out_n; gj = idx % out_n` | **逐元素**标量 |
| 3 | 范围守卫 | `if idx < total` | **逐元素**谓词 |
| 4 | 切片和（每元素） | `expr = w[0,gi,gj] + … + w[split_k-1,gi,gj]` | `split_k-1` 次加法 |
| 5 | bias 加（可选） | `expr + T.cast(bias[gj], "float")` | fp32 域 |
| 6 | 门控第二流 | `up = _slice_sum(…, out_n+gj)` | 仅 gated |
| 7 | 激活 | `gate * T.sigmoid(gate) * up` | 仅 gated |
| 8 | 量化写回 | `c[gi,gj] = T.cast(expr_or_gated, c_dtype)` | 唯一舍入点 |

**数据流图（源硬件视角）**：

```
w(GM, fp32, [split_k,m,n]) ──逐元素标量 load × split_k──▶ 寄存器表达式树 ──┐
bias(GM, dtype, [n]) ──────逐元素标量 load（可选）──────▶ cast→fp32 ──────┼─▶ sigmoid/mul（gated）─▶ cast ─▶ c(GM, c_dtype, [m,out_n])
                                                                        ──┘
```

即：**没有片上复用的中间态**（每个输出元素独立、只读 GM），也没有跨线程归约——归约轴
（`split_k`）在每个线程内部串行展开。

**循环与并行结构**：`grid = ceildiv(m·out_n, 1024)` 个 CTA × 256 线程；每线程处理
`1024/256 = 4` 个输出元素（`T.Parallel(elems_per_cta)` 由编译器映射）；每元素的
`split_k(+1 if gated)` 条 load 与加法在寄存器里串行。

**host 侧逻辑**：`_main` 装配内层 `prim_func` 的签名（bias 有无 → 两种 `global_symbol`），
`out_idx` 无（`C` 由调用方分配并传入）；shape 校验在 Python 层完成。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 |
|---|---|---|---|---|
| O1 | **扁平 1-D 元素映射**（`idx = bx·1024 + t`） | 让任意 `(m, out_n)` 都能用一个 grid 覆盖，尾块只需一个谓词 | 线程 id ↔ 元素线性映射，访存由 warp 合并 | CUDA 的 `threadIdx` 与全局合并访存 |
| O2 | `elems_per_cta = 1024`、`threads = 256` | 每线程 4 个元素，摊薄索引开销 | 线程块粒度 | CUDA thread block 调度 |
| O3 | **单次 launch、`C` 由调用方预分配** | 归约紧跟上一次 GEMM launch，中间不插分配 | `C` 作为参数（非 `out_idx`） | CUDA 的 `cudaMalloc` 成本（源码 docstring 明示） |
| O4 | **fp32 单累加器链 + 单次量化** | 精度：分片和不引入 16-bit 中间舍入；性能：只有 1 个 cast | `expr` 在 fp32 域累加，末尾 `T.cast` | 寄存器 fp32 累加 |
| O5 | **切片升序顺序求和** | 与上游 GEMM 的分片序一致，便于对齐 golden | 左结合链 | — |
| O6 | **门控在同一个 kernel 内完成**（gate 与 up 同元素映射） | 省掉独立的激活 kernel 及其 `[m,n]` 中间往返 | 同一 `gj` 索引 `gj` 与 `out_n+gj` 两条流 | — |
| O7 | **仅门控分支开 fast-math** | 让 `sigmoid` 走快速近似 | `TL_ENABLE_FAST_MATH` | CUDA 的 `__expf`/快速除法 |
| O8 | `T.sigmoid` 高层内建 | 少写表达式 | TileLang 内建 | — |
| O9 | **L2 驻留假设** | 假设部分和张量留在 L2，带宽不是瓶颈（docstring：「the workspace of the shapes worth splitting fits in L2」） | 归约紧跟 GEMM，热数据在 L2 | 大 L2 |
| O10 | `-O3 -DENABLE_BF16` | 编译器优化 + bf16 支持 | 编译开关 | nvcc |

### 0.5 硬件耦合性分析与 NPU 适配决策

判定口径：`references/migration-analysis.md` §5 的三层模型（语义层 / 算法层 / 优化层）+
四态处置（保留 / 等价替换 / 重新设计 / 舍弃）。判定输入含 Phase R 的调研结论（§1.6.0）。

| # | 源手段/结构 | 语义层？ | 硬件强相关？ | NPU 上可用？ | 处置 | 依据 |
|---|---|---|---|---|---|---|
| O1 | 扁平 1-D 元素映射 + 逐元素 `//`、`%` | 否（只是任务映射） | 是（线程 ↔ 元素） | **否**：NPU 的向量 lane 必须映射到**连续列**，跨行的扁平映射会让 lane 落在不连续/跨界地址上，且每元素两次整数除模全是标量 | **重新设计** | §0.6 R1；R4 表（计算单元匹配列）；探针 P1 生成的 1 行 × BC 列 DMA（`asc_copy_gm2ub_align(..., 1, 4096, ...)`） |
| O2 | 每 CTA 1024 元素 / 256 线程 | 否 | 是（线程数） | **否**（`T.Kernel(N)` 无 `threads`） | **等价替换** | 改为「1 行 × BC 列」的块（BC 取 256–1024），元素粒度同量级；§5.2 |
| O3 | 单次 launch、`C` 预分配 | 否 | 否 | **是** | **保留** | 平台 `T.Kernel` + `C` 作为 `prim_func` 参数（不放 `out_idx`）；NPU 每次 launch 固定开销更高（§1.6.0 D-2），该意图更应保留 |
| O4 | fp32 单累加器 + 单次量化 | **是**（精度语义） | 否 | **是** | **保留** | §1.6.1（唯一舍入点即下界）；探针 P1/P2 生成 `vcvt(..., ROUND_R, RS_DISABLE)` + `vsts_pack_b32` 各一次 |
| O5 | 切片升序求和 | **是**（累加分组） | 否 | **是** | **保留** | `T.serial(split_k-1)` 链式 `vadd`（探针 P1/P2 的 `for s` → 3 条 `vadd`）；§1.6.1 O1 机器验证：与 `partial.sum(0)` 逐位一致 |
| O6 | 门控同 kernel、`gj` / `out_n+gj` 双流 | **是**（融合语义） | 否 | **是** | **保留** | 同一 block 内两条列流（编译期常量偏移 `out_n`）；探针 P2/P6：两次 `asc_copy_gm2ub_align` 的源地址差恰为 `2S` |
| O7 | `TL_ENABLE_FAST_MATH`（门控） | 否（但影响数值） | 是（CUDA 的 `__expf`） | **否/不采用** | **舍弃** | 平台 `vexp` 已是单条 SFU 指令、`vdiv` 默认走 `vdiv_0ulp_ftz_true`（探针 P2 实测）；开 fast-math 会改变 `vmul+vadd` 收缩与除法精度路径，使 §1.6.1 的机器验证前提失效。平台 `references/programming.md` 无 fast-math 语义条款 |
| O8 | `T.sigmoid` 内建 | 否 | 否 | **未取证** | **等价替换** | 改为显式 `vneg → vexp → vadd(1) → vdiv`（全部有本平台证据：`simd.py` 导出 + `PL-950-vector-execution`、engram 已验用例） |
| O9 | L2 驻留假设 | 否 | 是（L2 容量/带宽） | **是（更强）** | **保留（作为假设并在 D-2 双口径使用）** | 950PR 的 L2 = 128 MiB（`ascend-950pr-9579-notes.md`），部分和 ≤ 3.7 MB；但 950PR 的 L2/MTE2 有效带宽**未实测**（§1.6.0 D-2） |
| O10 | `-O3 -DENABLE_BF16` | 否 | 是（nvcc） | **部分** | **等价替换** | 保留 `-O3`（`tilelang.compile` 的编译选项）；`ENABLE_BF16` 是 CUDA 宏，ascend 后端按 dtype 直接支持 bf16（与 `_gemm_basic_kernel` §4.7 同一处置） |
| — | SM90 template / grouped GEMM / `GEMMTMAKernel` 相关 epilogue 分支 | — | 是 | **否** | **舍弃**（Stage 0 已定） | Stage 0 脚手架按「NPU 无对应物」丢弃；本 kernel 的源实现与它们无关（源 `splitk_reduce_kernel` 是独立工厂） |

**统计**：**保留 4 项**（O3 单 launch 与预分配、O4 单累加器与单次量化、O5 切片升序、
O6 门控同 kernel）；**等价替换 3 项**（O2 块粒度、O8 `T.sigmoid` 展开、O10 编译开关）；
**重新设计 1 项**（O1 扁平映射 → 块级行列映射 + 向量链式累加）；**舍弃 2 项**（O7 fast-math、
SM90 族）。O9（L2 驻留）是设计**前提**而非手段，保留但标为未在 950PR 实测（§1.6.0 D-2）。

### 0.6 NPU 算法重设计

重设计对象 = O1（扁平映射）以及「如何在 Vector 单元上表达切片归约」。算法族本身沿用源结构
（单遍流式切片和），因为 Phase R 的结论是它已在下界（§1.6.0）。逐项给出语义保持论证：

**R1 并行分解：一维 `grid = m × ceil(out_n/BC)`，块内 = 1 行 × BC 列**

- 源 → 新：`idx // out_n` / `idx % out_n` 的逐元素映射 → `row = bx // nc`、`c0 = (bx % nc) · BC`
  的**块级**映射；块内 lane 直接映射到列 `[c0, c0+BC)`。
- 语义保持：输出元素集合不变（每个 `(i,j)` 恰属一个块）；求和顺序不变（同一元素的切片链
  不因映射变化而改变）；写回位置不变。**没有**任何跨块归约，因此不需要跨核同步。
- 边界：`bc = min(BC, out_n - c0)` 是块级值（可整除时被编译器折叠成常量，探针 P1 生成
  `block_idx % 7` 与常量 4096；P3 生成运行期 `update_mask`），尾 lane 由谓词屏蔽。

**R2 数据通路：GM→UB 的多流 DMA**

- 每块 `split_k` 条 `T.copy(w[s, row, c0:c0+bc], g_ub[s, :bc])`（门控再加 `split_k` 条
  `out_n + c0` 偏移的 `u_ub` 流），bias 1–2 条，输出 1 条 `T.copy(o_ub[:bc], C[row, c0:c0+bc])`。
- 语义保持：搬运的是同一批 fp32 元素（无 dtype 转换、无重排）；`T.copy` 的区间端点即块边界。
- 证据：探针 P1 生成 `asc_copy_gm2ub_align(dst, &W[s·m·n + block·BC], 1, 4096, …, 4096, 4096)`
  ——**单条 4096 B 连续描述符**（不是 1024 次标量读）。

**R3 累加结构：一个 64-lane fp32 寄存器累加器 + 链式 `vadd`**

- `acc = T.simd.alloc_local((1,), "float32")`；`acc[0] = vld(g_ub[0, ch·64])`，随后
  `for s in T.serial(split_k-1): acc[0] = vadd(acc[0], vld(g_ub[s+1, ch·64]), mask)`；
  bias 在链尾 `vadd`。
- 语义保持：加法的**分组与顺序与源 `_slice_sum` 完全一致**（左结合、切片升序、bias 最后），
  因此**逐位相同**（§1.6.1 O1 机器验证：与 `partial.sum(0)` 在 fp16/bf16/fp32 网格上 0 ulp 差、
  0 违反、0 非有限不一致）。
- 跨 `T.serial` 迭代携带向量值必须用 `alloc_local`（平台语义检查：不可用 Python 变量重绑定），
  与 engram 的 V11 结论同形。

**R4 门控 epilogue：同块双列流 + A1 的 silu 改写**

- `u_ub` 第二条流（列偏移 `out_n`，编译期常量）；两条流各自链式累加；silu 用
  `den = 1 + vexp(vneg(g))`、`y = vdiv(g, den) · up`。
- 语义保持：`gate·sigmoid(gate)·up` 与 `gate/(1+e^{-gate})·up` 在实数域恒等；fp32 层差 ≤ 3 ulp
  （§1.6.1 A1 机器验证），输出 dtype 层 0 ulp。

**R5 一次量化落盘**

- `vcvt(y, c_dtype, round="ROUND_R", sat=False)` + `vsts(dist="PK_B32")`；`c_dtype == "float32"`
  时**跳过 vcvt**（探针 P5 实测前端拒绝 `vcvt float32→float32`，改 `NORM_B32` 直存）。
- 语义保持：唯一舍入点仍是写回；`ROUND_R` = 就近舍入（与 torch `.to(dtype)` 的 RNE 一致），
  `sat=False` 保持 ±inf 不被夹到有限最大值。

**R6 退化路径的构建期专用化（`split_k == 1` / 无 bias / 无激活）**

- 三个条件都是构建期常量，Python 层直接不生成对应的 DMA、`vadd` 与 silu 序列。探针 P4
  生成的 kernel 体只剩 `vld → vcvt → vsts_pack_b32`（+1 条 DMA in/out）。
- 语义保持：**逐位相同**（删掉的是空操作：0 条 `vadd`、0 条 bias DMA）；§1.6.1 A2 机器验证。

**重设计后的算法一句话**：把「扁平逐元素 + 逐元素整数除模 + 标量 load」换成「块级行/列映射 +
64-lane 内列 DMA + 寄存器链式 `vadd`」，数学内容、累加顺序、舍入点、I/O 契约全部不变。

**验证方式（`migration-analysis.md` §6.3 第 3 条）**：R1–R6 的语义保持由两层验证兜底 ——
① §8.1 的 golden 按 §0.1 语义独立实现（不复刻上述任何结构），逐元素对比；
② §8.3 的 L0-1…L0-9 分别覆盖 R1（块映射/尾块）、R2（多流 DMA）、R3（切片链与 `split_k=1`）、
R4（门控双流 + A1）、R5（fp16/bf16/fp32 三种落盘）、R6（退化路径），容差按 §8.2 的 dtype 表；
③ 累加分组未变（R3 保持源顺序），故 §1.6.1 O1 的 0 ulp 实测即为最强证据。

### 0.7 标杆实现

| 参照 | 关系 | 可借鉴 |
|---|---|---|
| 源 `splitk_reduce_kernel`（GPU） | **迁移源** | 数学语义、切片契约、门控融合方式、单次量化 |
| `examples/engram_gate_conv/_engram_gate_conv_fwd_kernel/_engram_gate_conv_fwd_kernel.py` | **同平台 SIMD 标杆**（已通过真机/仿真验证的 AIV kernel） | `T.SimdVF()` + `T.simd.*` 的完整写法、`UNPK_B16`/`PK_B32` 链、`update_mask`、`alloc_local` 累加器、`annotate_buffer_versions({…: 1})` |
| `examples/mla_dense_bwd/_delta_kernel/DESIGN.md` | 同平台设计范式 | §1.6.0/§1.6.3/§5.5 的写法；`immutable variable` 与二维 grid 的陷阱 |
| `.agents/platforms/toolchains/ascend/knowledge/PL-950-vector-execution.md` | 已核验条目 | 向量执行 + 尾块（SIMD 路径 fp32 N=256/257） |
| `.agents/platforms/toolchains/ascend/kernel_template.py` | 平台骨架 | `T.Kernel(grid)` + `T.SimtVF` + `--level L0/L1/L2/boundary` 的测试形态 |
| `/workspace/TileOPs-ascend950/src/tileops/kernels/ascend/elementwise/abs.py` | 同平台逐元素标杆（`PL-950-vector-execution` 的证据源，只读） | SIMD 逐元素路径与尾块填充 |

---

## 1. 概述

### 1.1 算子名称

`splitk_reduce_kernel`（split-K fp32 部分和归约 + 可选 bias + 可选 `silu_and_mul` 融合 epilogue；
SIMD/AIV 向量 kernel，1 行 × BC 列的块映射 + 64-lane 链式 `vadd` + 单次量化落盘）。

### 1.2 功能描述

把上游 split-K GEMM 落在 `[split_k, m, n]` fp32 的部分和沿切片轴求和，写入 `[m, out_n]`：
可选在 fp32 域加列 bias（输入 dtype），可选把 `silu_and_mul`（`gate·sigmoid(gate)·up`，
`out_n = n/2`）融进同一次落盘。全 fp32 累加、**唯一一次舍入**在写回时；`split_k == 1` 时
退化为「纯拷贝 / 加 bias / 可选 silu」。

### 1.3 数学公式

$$
c[i,j] \;=\; \mathrm{cast}_{c\_dtype}\!\left(\sum_{s=0}^{S-1} w[s,i,j] \;+\; [\,\text{has\_bias}\,]\cdot
\mathrm{f32}\big(b[j]\big)\right),
\qquad S = \text{split\_k},\; c\_dtype = \text{out\_dtype} \lor \text{dtype}
$$

门控分支（`activation = "silu_and_mul"`，`N = n/2`，`j \in [0,N)`）：

$$
g[i,j] = \sum_{s} w[s,i,j] + [\,\text{bias}\,]\mathrm{f32}(b[j]), \qquad
u[i,j] = \sum_{s} w[s,i,N+j] + [\,\text{bias}\,]\mathrm{f32}(b[N+j])
$$
$$
c[i,j] = \mathrm{cast}_{c\_dtype}\big(\,\mathrm{silu}(g[i,j])\cdot u[i,j]\,\big),
\qquad \mathrm{silu}(x) = \frac{x}{1+e^{-x}}
$$

求和是**切片升序左结合**（`((w_0+w_1)+w_2)+…`），bias 在切片和之后加；仅写回时舍入一次。

### 1.4 算法描述

本节描述**迁移决策后（§0.5/§0.6）的 NPU 侧算法**：

1. **块级行列映射（R1，相对源实现的差异）**：一维 `grid = m × ceil(N/BC)`；`row = bx // nc`、
   `c0 = (bx % nc)·BC` 都是块级标量，块内 = 1 行 × `BC` 列。源的逐元素 `idx // N`、`idx % N`
   与逐元素 `idx < total` 守卫整体消失（§1.6.2 V3/V4）。
2. **多流 DMA 装载（R2）**：每块 `split_k` 条 `T.copy` 把 `w[s, row, c0:c0+bc]` 搬进
   `g_ub[s, :bc]`（fp32、连续 `4·bc` 字节）；门控再加 `split_k` 条 `out_n+c0` 偏移的 `u_ub` 流；
   bias（若有）1–2 条。
3. **64-lane 链式累加（R3）**：`acc = alloc_local((1,), fp32)`，`vld` 第 0 片后
   `for s in 1..S-1: acc = vadd(acc, vld(slice_s))`，bias 接在链尾；累加顺序与源一致（逐位相同）。
4. **门控 epilogue（R4 + A1）**：同块第二条列流算 `u`；`den = 1 + vexp(vneg(g))`，
   `y = vdiv(g, den)·u`（等价于 `g·σ(g)·u`，少一条 `vmul`，§1.6.1 A1）。
5. **一次量化落盘（R5）**：`vcvt(y, c_dtype, ROUND_R, sat=False)` + `vsts(PK_B32)`；
   `c_dtype == fp32` 时跳过 vcvt、直接 `vsts(NORM_B32)`。
6. **退化路径专用化（R6）**：`split_k == 1` / 无 bias / 无激活在构建期消去对应指令。

与源算法结构不同的点只有 1、2、4 的**表达方式**（映射粒度、DMA 形态、silu 写法）与 6 的专用化；
数学内容、累加顺序、舍入点、I/O 契约与源算法同构。

### 1.5 数据流图

```
w[s,row,c0:c0+bc] (GM, fp32) ──MTE2 copy × split_k──▶ g_ub[s] (UB, fp32) ─┐
w[s,row,N+c0:…]  (GM, fp32) ──MTE2 copy × split_k──▶ u_ub[s] (UB, fp32) ─┤（仅 gated）
bias[c0:c0+bc] / bias[N+c0:…] (GM, dtype) ─MTE2──▶ b_ub[0..1] (UB, dtype) ─┤
                                                                          ▼
                     T.SimdVF: vld → (vadd × split_k-1) → [+bias] → [silu·up] → vcvt
                                                                          │
                                                       vsts(PK_B32 / NORM_B32) ─▶ o_ub (UB)
                                                                          │
                                              T.copy(o_ub[:bc], C[row, c0:c0+bc]) ─▶ C (GM)
```

无 L1/L0C 参与（无 Cube）；中间态只有 UB 里的 `split_k`（门控 `2·split_k`）份列块与一个
输出列块，**不落任何 GM 中间量**。

### 1.6 算法调研与优化分析

> 调研对象 = **同一数学语义（split-K 部分和的 fp32 归约 + 可选列 bias + 可选门控激活）的算法族**。
> 源算法（扁平逐元素、逐元素标量 load）只是基线候选之一。本算子属「规约 + 多步（bias/激活）」
> 类，按 `algorithm-research.md` §2 需**完整调研**（R1 候选表 + R3 四口径 + R4 逐候选）。
>
> 本地检索记录：`knowledge/algorithm-candidates.md`（命中 elementwise/broadcast、reduction、
> softmax、RMSNorm/LayerNorm、GEMM、attention 六族，**无**「split-K 归约 epilogue」专族）；
> `knowledge/INDEX.md` 全部条目；pattern-library `constants.md`；
> `python3 .agents/tools/kb_search.py "split-k reduction epilogue silu_and_mul vector reduce fp32 partial"
> --task-dir …` → **命中 1 条**（`CG-950-simd-shared-reduce`：高层二维 shared→shared 归约的输出
> 差异，属 open 能力缺口 —— 本设计**不使用** `T.reduce_sum` 的任何二维形式，全部归约在
> 64 lane 内以逐 lane `vadd` 表达，故该缺口不适用，仅作规避记录）；
> `examples/` 内全部 ascend-target 向量 kernel（engram gate-conv、`mla_dense_bwd/_delta_kernel`、
> `moe_permute_align`、`paged_cache_write`、`rope`、`shared_expert_mlp`）。
> 未做互联网检索的必要性：候选空间在 R1 表内封闭（4 类结构替代 + 3 类恒等变形），每项都能由
> 结构判据或本平台证据判定，无需外部算法补充（纪律见 `algorithm-research.md` §6）。

#### 1.6.0 算法调研（Algorithm Research）

**R1 等价化简公式候选**（基线在表内；正式等价论证与收益量化在 §1.6.1）

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 是否纳入 R3 |
|---|---|---|---|---|---|
| C0 | **基线：源 `splitk_reduce_kernel`** | 扁平 1-D 元素映射（每元素 `//`、`%`）、逐元素标量 load × `split_k` 链式加、单次 cast | — | — | ✅ 基线 |
| C1 | **块级行列映射 + 64-lane 向量链**（主选） | 同式；`1 行 × BC 列` 的块，`split_k` 条流 DMA + `vadd` 链，一次 `vcvt`+`vsts` | 数学恒等；**同分组同顺序 → 逐位相同**（§1.6.1 O1 实测 0 ulp） | 同流量；把「每元素 2 次整数除模 + `split_k` 次标量 load」换成块级标量 + DMA + 向量指令 | ✅ |
| C2 | **原子累加归约**：GEMM 直写 fp32 预激活缓冲（`set_atomic("add","float32")`），再一遍 epilogue | `out32 += w_s`（4 次 RMW），然后 `c = cast(silu/gate·up)` | 容差内等价；**非确定累加顺序**（原子无序） | 负收益：流量 2.3–2.4×（见 R3），且要改上游契约 + 2 次 launch | ✅ |
| C3 | **GEMM epilogue 融合**（把归约融进最后一个 K 切片） | 最后一个 Cube 块读其它分片的部分和并求和 | 数学恒等 | 流量降到 GEMM 自身的写出量（省 2.10 MB） | ✅ |
| C4 | **两遍物化**（fp32 预激活 → 独立 cast） | `t32 = Σ_s w_s`（写 GM），`c = cast(t32)` | 精确等价 | 负收益：多 `m·N·4` 写 + 读 + 一次 launch | ✅ |
| C5 | **树形/两两归约**（`(w0+w1)+(w2+w3)`） | 深度 2 而非 3 的加法树 | 数学等价；**fp32 重结合**（舍入分组改变） | FLOPs 与流量都不变，只缩短寄存器依赖链 | ✅ |
| C6 | **bias 折进第 0 个切片的 GEMM 累加** | GEMM 在 `s=0` 时加 bias | 容差内等价 | 省 1 次 bias 读（`2·BC·2 B`/块） | ✅ |
| C7 | **切片轴放到 lane 上**（每 lane 一个 `s`，`vcadd` 跨 lane 归约） | `lane = s`，`vcadd` 出和 | 数学等价 | 负收益：`split_k ≤ 4` 只占 4/64 lane（94% 浪费）+ 每条 chunk 一次跨 lane 归约 | ✅ |
| C8 | 恒等变形：`silu(g)·u` → `(g/(1+e^{-g}))·u` | 除法代替「取倒数再乘」 | 数学恒等；fp32 舍入位置少一处 | 每 chunk 少 1 条 `vmul`（§1.6.1 A1） | ✅（采纳，A1） |
| C9 | 结构性不变量：`split_k == 1` / 无 bias / 无激活的构建期专用化（**源在 Python 层已如此，直译成无条件发射才会退化**） | 删除空操作 | 逐位相同 | 退化路径只剩 `vld→vcvt→vsts`（§1.6.1 A2） | ✅（保留为不变量，A2） |
| C10 | 变换域 / 亚三次类算法 | — | 不适用：本算子是逐元素归约，无乘法规约结构 | — | ❌ 非候选（结构判据） |

**R2 在线算法**：**有，且基线本身已是在线形态**。① 每个输出元素只依赖同一位置的 `split_k` 个
输入，`split_k` 条流一遍扫完即出结果，中间状态是 **O(tile)**（`2·split_k·BC` 个 fp32）而非
O(`split_k·m·n`)；② 无可重缩放的全局统计量（无 softmax max/sum、无 Welford 方差），因此不存在
「两遍 → 单遍」的进一步空间；③ 源算子自身已是单遍（§0.4 O4/O6），是「该算子族存在单遍形态」的
直接证据。收益口径核对：扫描遍数已到 **1 遍/字节**下界，中间缓冲已到 **O(tile)** 下界。
**跨 launch 的「GEMM 分片 + 归约」两段结构不属本口径**（R2 限定同一 launch 内），它是 C1/C2/C3
的结构前提；本 kernel 内部已是单遍在线累加。结论：**无在线化改造空间**。

**R3 复杂度对比**（四口径；代表 workload = `ds-v3-t32 gate/up`（门控，`m=32, N=2048`，`split_k=4`，
bf16，`has_bias=False`）与 `ds-v3-t32 down`（非门控，`m=32, N=7168`，`split_k=4`）。
口径说明：**lane-op** = 一条 64-lane 向量指令的工作量，按「每元素 op 数 × 元素数」计
（乘加=1、超越函数=1、cast=1）；访存按核实际 DMA 字节数（读部分和 + 读 bias + 写 out），
不含 L2 命中后的 HBM 折减；扫描遍数按「每个输入字节被读的 GM 次数」；中间缓冲峰值指片上或需额外
GM 物化的量）

| 算法候选 | lane-ops（up / down） | 访存量 Bytes（up / down） | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---|---|---|---|---|---|
| **C0 基线（源结构）** | 0.92 M / 2.06 M | **2.10 MB / 4.13 MB**（部分和 2.10 / 3.67 + out 0.131 / 0.459） | 1（每片读一次） | 0（寄存器表达式树）+ 0 UB | `ceil(m·N/1024)` = 64 / 224 块 × 256 线程；无跨核 |
| **C1 主选（块级 + 向量链）** | 同上（不变） | 同上（**不变**） | 同上 | `2·split_k·BC·4` = **8 KiB / 16 KiB**（UB，up BC=256 / down BC=1024） | 256 / 224 逻辑块（4.6 / 4.0 波）；无跨核 |
| C2 原子归约 | 0.92 M | **5.36 MB / 9.60 MB**（零初始化 0.52/0.92 + 4×(RMW 读 0.52+写 0.52 / 0.92+0.92) + epilogue 读 0.52/0.92 + 写 0.13/0.46） | **2**（RMW + epilogue） | 预激活 fp32 `m·n·4` = 0.52 MB / 0.92 MB 落 GM | 需 `set_atomic` + 2 launch；**累加顺序非确定** |
| C3 GEMM epilogue 融合 | 0.92 M | **0.13 MB / 0.46 MB**（只剩 out） | 1（在 GEMM 内） | GEMM 的 L0C | 4 个 K 切片在不同 Cube 块 → **需跨 block 同步/原子**（平台未认证，§R4） |
| C4 两遍物化 | 0.92 M | 2.10 + 0.26(写 fp32) + 0.26(读) + 0.13 = **2.76 MB** | 2 | fp32 预激活 `m·N·4` = 0.26 MB 落 GM | 2 launch |
| C5 树形归约 | 0.92 M（同） | 2.10/4.13 MB（同） | 1 | 同 C1（多 1 个活向量） | 同 C1；**舍入分组改变** |
| C6 bias 折进 GEMM | 0.92 M | 2.10/4.13 MB（同，省 bias 读 0 / 0） | 1 | 0 | 要改上游契约；**bias 先加 → 舍入顺序改变** |
| C7 lane 上放切片 | 0.92 M | 2.10/4.13 MB（同） | 1 | 同 C1 | lane 只用 4/64（浪费 94%）；每 chunk 1 次 `vcadd` |

**R4 硬件亲和性评估**（检查清单：计算单元匹配 / 片上容量 / 对齐整除 / 静态边界 / 流水可融合 /
跨核结构；负向淘汰附依据）

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---|---|---|---|---|---|---|
| **C1 主选** | ✅ 全部算术是逐元素加/乘/`vexp`/`vdiv`/`vcvt` → 落 **Vector（AIV）SIMD**；无 MAC 密集段 → 不需要 Cube | ✅ UB 8–20 KiB / 248 KiB = 3–8% | ✅ `BC` 为 64 的倍数（= 1 个 fp32 寄存器宽度，`PL-950-vector-execution` 的 N=64/256 已验）；`N` 非 64 倍数时 `update_mask` 尾 lane（探针 P3） | ✅ `split_k`、`BC/64`、`nc` 全为构建期常量 | ✅ MTE2→V→MTE3 由编译器插 2 对 `asc_sync_*`（探针 P1/P2）；无 VST→VLD 回读 → 不需要 `mem_bar` | ✅ **采纳** |
| C0 基线 | ⚠️ 映射形态与 NPU lane 语义冲突（lane 必须落连续列）；逐元素整数除模是标量热点 | ✅ | ❌ 无向量轴概念 | ✅ | ❌ 逐元素标量 load 无法生成描述符 DMA | ❌ **淘汰**（依据：平台 SIMD 的向量化轴只有「连续元素」一种，`simd.py` 的 `vld/vsts` 以 `access_ptr` + 64 元素步进表达；逐元素 `//`、`%` 在 `T.Parallel` 内会退化为标量） |
| C2 原子归约 | ⚠️ `set_atomic` 存在（`PL-950-atomic-splitk`），但该条目只在 **950DT 仿真器的 2 核**用例上认证 | ✅ | ✅ | ✅ | ❌ 需 2 次 launch + 输出预清零契约 | ❌ **淘汰**（依据：R3 流量 2.3–2.4× + 累加顺序非确定 + 上游契约改动；`PL-950-atomic-splitk` 明确「不保证同一 kernel 内消费者读到最终值」） |
| C3 GEMM 融合 | ✅ | ✅ | ✅ | ✅ | ❌ 需跨 block 同步：平台 `references/programming.md` 记「手工跨核 flag...跨 block batch barrier 仍未认证」 | ❌ **淘汰**（等价的原子形态即 C2，已淘汰；无已认证 barrier） |
| C4 两遍物化 | ✅ | ⚠️ +0.26 MB GM | ✅ | ✅ | ❌ 多一遍全量流量 + 1 launch | ❌ **淘汰**（R3：+24% 流量；且 C1 一次舍入已位级等价，见 §1.6.1） |
| C5 树形归约 | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️ **不采纳**：流量/FLOPs 不变、仅缩短依赖链（`split_k=4` 时 3→2 条 `vadd`），但改变舍入分组；在本 kernel 带宽受限的区间内收益 < 1%（§1.6.0 D-2 发射项 0.07–0.32 µs ≪ 流量项），**代价（新增加法树正确性论证与 ULP 足迹）不成比例** |
| C6 bias 折进 GEMM | ✅ | ✅ | ✅ | ✅ | ❌ 改上游契约 + bias 先加改变舍入顺序 | ❌ **淘汰**（本 op 的 4 次调用 `has_bias=False`，收益为 0；通用路径保留本 kernel 的 bias） |
| C7 lane 上放切片 | ⚠️ 需每 chunk 一次 `vcadd`（跨 lane 归约） | ✅ | ❌ 4/64 lane 有效 | ✅ | ⚠️ | ❌ **淘汰**（94% lane 浪费 + 每 chunk 多一条归约指令，见 R3） |
| C8/A1 silu 改写 | ✅ 少一条 `vmul` | ✅ | ✅ | ✅ | ✅ | ✅ **采纳**（§1.6.1，唯一的公式级优化项） |
| C9/A2 构建期专用化 | ✅ 删空操作 | ✅ | ✅ | ✅ | ✅ | ✅ **保留为不变量**（§1.6.1；源已如此，非优化项） |

**调研结论**：选定算法族 = **C1（块级行列映射 + 64-lane 流 DMA + 寄存器链式 `vadd` + 单次量化）
+ A1 一项公式级等价优化 + A2 一项结构性不变量**，并在其上落地 §0.6 的重设计项 R1–R6。
关键依据：R3 中 C1 的访存已达「每个部分和字节读一次 + 输出写一次」下界（2.10/4.13 MB）、
中间缓冲 O(tile)（8–20 KiB UB），与基线 **完全同流量**但把标量热点整体消除；
C2/C4 分别把流量抬到 2.3× / +24%，C3 依赖未认证的跨 block 同步，C7 浪费 94% lane，
C5/C6 收益低于测量分辨率（或改上游契约）——均被 R4 的容量/对齐/跨核/契约条款淘汰。
**与基线（源算法）的结构差异一句话**：数学与累加顺序、舍入点、I/O 契约全部沿用源算法，
但把「扁平逐元素映射 + 逐元素标量 load」换成「块级行列映射 + 描述符 DMA + 向量链式累加」。
**「无更优替代」的调研范围**（针对「还能不能更省流量」这个问题）：已查
`knowledge/algorithm-candidates.md` 全表（6 族）、`knowledge/INDEX.md` 全部条目、
pattern-library `constants.md`、`examples/` 内全部 ascend-target 向量 kernel、kb_search（命中 1 条
能力缺口，已记录规避）、以及源仓 `dense.py` 中全部 5 处 `splitk_reduce_kernel` 的调用点
（`_fp8_ws_splitk_pair`、`_gemm_coop2_splitk_kernel`、`_gemm_splitk_kernel`、本 op 的 wrapper、
SM90 template）——它们用的都是同一份归约结构。**结论**：不存在能同时减少「读部分和 + 写 out」
字节数的算法族（几何下界就是每个字节一遍），因此优化空间只在**映射与指令形态**（C1 已取）与
**公式微优化**（A1/A2）。

**源算法优化意图的承接**（§0.4 逐项）：O1（扁平映射）→ R1 块级映射（意图「一个 grid 覆盖任意
shape」由 `nc = ceil(N/BC)` 达成）；O2（块粒度）→ 等价替换为 `1×BC`；O3（单 launch +
预分配）→ 保留；O4（单累加器 + 单次量化）→ 保留且证明为下界（§1.6.1）；O5（切片升序）→ 保留且
机器验证逐位相同；O6（门控同 kernel）→ 保留（双列流）；O7（fast-math）→ 舍弃并有本平台替代
（`vexp` 单指令 + 精确 `vdiv`）；O8（`T.sigmoid`）→ 展开为显式序列；O9（L2 驻留）→ 保留为假设，
在 D-2 中作为第二种口径；O10（`-O3`）→ 保留 `-O3`，`ENABLE_BF16` 舍弃。

**设计期估算（D-2 roofline 下界行）**

代表 workload：`ds-v3-t32 gate/up`（门控，BC=256）与 `ds-v3-t32 down`（非门控，BC=1024）；
全部 4 次真实调用（§0.2）的范围也给出。下界 = max(流量项, 发射项, 容量项)：

| 项 | ds gate/up | ds down | 全部 4 次调用范围 |
|---|---|---|---|
| 流量项（部分和 + bias + out） | 2.23 MB | 4.13 MB | 1.67–4.13 MB |
| 流量项 ÷ 1.03 TB/s（HBM 口径外推） | 2.2 µs | 4.0 µs | 1.6–4.0 µs |
| 流量项 ÷ 2.41 TB/s（L2 驻留口径外推） | 0.93 µs | 1.71 µs | 0.69–1.71 µs |
| 发射项（向量指令数 ÷ 56 核 ÷ 1 op·cycle⁻¹@1.8 GHz） | 384 op/核 → 0.21 µs | 576 op/核 → 0.32 µs | 288–576 op/核 → 0.16–0.32 µs |
| 容量项 | 9.5 KiB / 248 KiB = 3.8% | 18.0 KiB = 7.3% | 3.8–7.7% → 不构成下界 |
| 每 invocation 的 launch 次数 | 1 | 1 | 1 |

`估算下界: 待核验（缺 ascend-950pr-9579 的 HBM/L2 有效带宽与向量发射速率实测常数——平台
cost_measurements 为空；pattern-library constants.md 的每一条都绑定 Ascend910B2C 与
tilelang 0.1.2+15ad002b3d，只能作量级外推）。已算出的部分：流量项 1.67–4.13 MB、发射项
0.16–0.32 µs、容量项 ≤7.7%，故下界形如 max(流量/1.03–2.41 TB/s, 0.16–0.32 µs) + 1 次 launch 固定开销。`

**由估算得到的、与常数无关的结论**（决定本设计的重点）：

1. **AI ≈ 0.41 lane-op/字节**（ds gate/up：0.92 M op / 2.23 MB；down：2.06 M / 4.13 MB ≈ 0.50）——
   远低于任何机器平衡点 → **纯带宽受限**，因此设计目标是「每个字节只读一次 + 只写一次」，
   **不追求 MFU**（这也是本文不使用算力口径的原因）。
2. **发射项比流量项小一个量级**（0.16–0.32 µs vs 0.69–4.0 µs）→ 指令数优化（如 C5 的加法树）不是
   本 kernel 的收益来源，只有流量与 launch 次数是。
3. **单次 launch 的固定开销（未实测，910B2C 量级 2–5 µs）与流量项同量级甚至更大** → 本 kernel
   在 op 内的时延下界被「1 次 launch + 2–4 MB 流量」共同主导；因此 §9.1 明确记「不再拆分 kernel、
   不引入中间 GM 物化」，也解释了为什么 C3（省掉本 kernel）在结构上是唯一能实质降时延的方向
   （但被未认证的跨核同步阻塞）。

#### 1.6.1 数学等价优化（公式级）

分析对象为 §1.6.0 选定的 C1。逐项评估：

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算 |
|---|---|---|---|---|---|
| **A1** | SiLU 的除法改写 | `y = g · σ(g) · u`，其中 `σ(g) = 1/(1+e^{-g})`（源码 `gate * T.sigmoid(gate) * up`）：先取倒数、再与 `g` 相乘（两处 fp32 舍入） | `den = 1 + e^{-g}`；`y = (g / den) · u`（一处 fp32 舍入） | **数学恒等**：`g·(1/den)·u = (g/den)·u`（实数域除法定义）。fp32 层差别仅是「取倒数再乘」vs「直接除」的舍入位置：机器验证 36 用例/ dtype 的 fp32 足迹 **≤ 3 fp32 ulp**（改变元素比例 ≤ 0.23），输出 dtype 层 **0 ulp**、违反率 0；对 fp64 参照的误差与基线**逐位相同**（表内 `max abs vs fp64 (cand) == (base)`）。fp16/bf16 舍入影响：无（差 ≤3 fp32 ulp ≪ 半个 16-bit ulp） | 每 chunk 少 1 条 64-lane `vmul`：门控/workload 每 chunk 21→20 条向量指令（**−4.8%**）；全部 4 次调用合计少 **1 792** 条 `vmul`（门控 chunk 数 = 1 024 + 768） |
| **A2** | 退化路径的构建期专用化（**结构性不变量**，不是相对源的公式优化 —— 源在 Python 层本来就专用化，这里要求迁移后保持同一性质） | 若把源「直译」成无条件发射：`for s in T.serial(split_k)` 装载、`split_k-1` 条 `vadd`、bias DMA、silu 序列**全部生成**（`split_k == 1` / `has_bias == False` / `activation == "none"` 时它们是空操作） | `split_k == 1` → 1 条流 DMA、0 条 `vadd`；`has_bias == False` → 无 bias DMA / 无 bias `vcvt`；`activation == "none"` → 无 silu 序列 | **逐位相同**：删去的都是恒等空操作（0 次迭代的循环、0 条 DMA、不执行的表达式），不触碰任何有值的运算。机器验证：两种构建在 fp16/bf16 上 **bitwise equal**（`torch.equal` 于 `int16` 视图）——本项因此是**回归守卫**（防止 Stage 3 写成无条件发射）而非等价性风险项 | 相对「直译构建」：`split_k=1` + 无 bias + 无激活的路径从 12 条向量指令/chunk 降到 **3 条**（探针 P4 实测：只剩 `vld → vcvt → vsts_pack_b32`），DMA 从 5–9 条降到 2 条；相对源实现：无变化（意图保持） |

**等价性机器验证（D-1）**：脚本 `examples/shared_expert_mlp/splitk_reduce_kernel/verify_equiv.py`
（torch CPU，fp64 参照；候选式与基线式各自在 **fp32 存储路径**上忠实回放：fp32 `exp`、
fp32 倒数/除法、fp32 乘、最后一次舍入到输出 dtype；`sat=False` 语义由「不过滤 ±inf」体现）。
运行方式 `python verify_equiv.py`（无需 NPU）。判定标准：输出 dtype 最大 ulp 差 ≤ 1、
`>1ulp` 违反率 ≤ 1e-3、非有限模式完全一致、候选对 fp64 参照不劣于基线；A2 另需位级相等。
ulp 以**输出 dtype 的网格**度量且尺度下限取该 dtype 的*最小正规数*（否则 bf16 次正规区
1e-35 量级会把指标放大成无意义的大数——初版曾因此误判 34 例，已修正）。

| 用例 | dtype | 最大 ulp | >1ulp 违反率 | 非有限不一致 | fp32 足迹（观测） | 对 fp64 误差 cand / base | 结论 |
|---|---|---|---|---|---|---|---|
| A1：36 例/dtype = 6 组合（2 几何 4×128、8×256 × split_k 1/2/4）× 6 输入模式（normal/wide/saturating/cancel/zeros/corner），逐项明细见脚本输出 | float16 | **0** | 0.0 | 0 | ≤ 2 fp32 ulp（改动元素 ≤ 23%） | 0.000e+00 / 0.000e+00 | **EQUIV_PASS** |
| 同上 | bfloat16 | **0** | 0.0 | 0 | ≤ 3 fp32 ulp（改动元素 ≤ 23%） | 7.222e-35 / 7.222e-35 | **EQUIV_PASS** |
| A2：`split_k=1`+bias、无 bias 纯拷贝 | float16 / bfloat16 | **0** | 0.0 | 0 | 0（位级相等） | 0.000e+00 / 0.000e+00 | **EQUIV_PASS** |
| O1（观测项，非优化）：切片升序 fp32 和 vs `partial.sum(0)`，每 dtype 12 例（2 几何 × split_k 2/4 × 6 模式），共 36 例 | float16 | **0** | 0.0 | 0 | 0 | 3.906e-03 / 3.906e-03 | **EQUIV_PASS** |
| 同上 | bfloat16 | **0** | 0.0 | 0 | 0 | 0.000e+00 / 0.000e+00 | **EQUIV_PASS** |
| 同上 | float32 | **0** | 0.0 | 0 | 0 | 1.221e-04 / 1.221e-04 | **EQUIV_PASS** |

（脚本实跑输出：`TOTAL cases: 110, failures: 0` → `[EQUIV_ALL_PASS]`。`corner` 模式含
`±inf / NaN / ±0 / 次正规数 / dtype 最大值`，覆盖 `gate=-inf → NaN`（`-inf/(1+inf)` 与
`-inf·0` 同为 NaN）与 `gate=+inf → ±inf` 的传播一致性；`saturating` 模式覆盖 `e^{-g}` 在 fp32
上溢（`g < -88.7`）后两式同归 0 的情形。fp32 足迹列是**观测项**而非判据：改写的固有性质是
fp32 位模式可能变化，判据是输出 dtype 的 ulp 与违反率。）

**优化结论**：**公式级优化采纳 1 项 —— A1**（`EQUIV_PASS`）；A2 是**结构性不变量**
（源已专用化，非相对源的优化），以位级相等的回归守卫形式记录在同一张表里；O1 是观测项，
为 §8.2 的精度预算提供实测依据（三者合计 110 例、0 失败，脚本与结果表随本文留存）。
优化后公式（供 §3.1 拆解使用）：

```
# 非门控
acc = w[0,i,j];  for s in 1..S-1: acc += w[s,i,j];  acc += [bias] f32(b[j])
c[i,j] = cast_c(acc)
# 门控（N = n/2）
g = Σ_s w[s,i,j] (+f32(b[j]));  u = Σ_s w[s,i,N+j] (+f32(b[N+j]))
c[i,j] = cast_c( (g / (1 + e^{-g})) · u )          # A1
# 退化（S == 1 / 无 bias / 无激活）由构建期消去对应指令   # A2
其中 Σ_s 为切片升序左结合，仅在 cast_c 处舍入一次；cast_c 用 ROUND_R + sat=False。
```

**无进一步优化空间的结论与依据**：① 加法次数已到下界 —— 每个部分和必须被读一次并加一次
（`S-1` 次加法，树形 C5 不改变次数，只改顺序）；② 流量已到下界 —— 每个部分和字节读一次、
输出写一次（C2 的原子 RMW、C4 的物化都只会更多）；③ 舍入点已到下界 —— 全程 fp32、仅写回时
舍入一次（在 16-bit 域做部分和会破坏上游契约与精度）；④ bias 必须在激活前加（语义）；
⑤ silu 只剩 1 次 `vexp` + 1 次 `vdiv`（本平台无 `vexp2`，A1 已把乘法并掉，`simd.py` 导出表核对）；
⑥ 门控的 `gate` 与 `up` 两条链无公共子表达式（不同列、不同数据）。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

盘点实现方案中的**全部**循环与标量计算点：

| # | 计算点 | 原实现形态（源） | 向量替代方案 | 是否替代 | 不可替代理由（不可替代时必填） |
|---|---|---|---|---|---|
| V1 | 切片和（每元素 `S-1` 次加） | 逐元素标量加法表达式树 | `T.simd.vadd` 64-lane 链（`alloc_local` 累加器 + `T.serial(S-1)`） | ✅ | — |
| V2 | 切片维循环展开 | Python 循环生成表达式树 | `T.serial(split_k-1)` → 编译成 `S-1` 条 `vadd`（探针 P1：`for (s=0;s<3;++s)` 单条 `vadd`） | ✅ | — |
| V3 | 元素 → (行, 列) 映射（`idx // out_n`、`idx % out_n`） | **逐元素**整数除 + 模 | **已消除**：`row`、`c0` 是块级标量，lane 直接是列 | ✅（消除） | — |
| V4 | 范围守卫 `if idx < total` | 逐元素谓词 | **已消除**：`grid = m·ceil(N/BC)` 不含越界块；只有尾 chunk 的 lane 谓词（V9） | ✅（消除） | — |
| V5 | bias 列广播加 | 逐元素 `cast(bias[gj])` + 加 | `vlds_unpack(B16) → vcvt → vadd`（探针 P1：`vlds_unpack<half>` + `vcvt<float>` + `vadd`） | ✅ | — |
| V6 | 门控第二流（`out_n + gj`） | 逐元素索引 | 同 V1 的第二条流，列偏移是**编译期常量**（探针 P2：源地址差恰 `2S`） | ✅ | — |
| V7 | `sigmoid` | `T.sigmoid` 标量内建 | `vneg → vexp → vadd(1) → vdiv`（4 条向量指令；`PL-950-vector-execution`/engram 已验形态） | ✅ | — |
| V8 | `gate · sigmoid · up` | 两次标量乘 | 1 次 `vdiv` + 1 次 `vmul`（A1） | ✅ | — |
| V9 | fp32 → 输出 dtype 量化 | 逐元素 `T.cast` | `vcvt(..., ROUND_R, sat=False)` + `vsts(dist="PK_B32")`（探针 P1/P2：`vcvt<half>(..., ROUND_R, RS_DISABLE)` + `vsts_pack_b32`） | ✅ | — |
| V10 | 尾 lane 谓词（`N % BC ≠ 0`） | 逐元素守卫 | `T.simd.update_mask(T.min(64, bc - ch·64), 32)`（探针 P3 生成 `update_mask`；整除时整段不生成） | ✅ | — |
| V11 | GM 读 / 写 | 逐元素标量 load / store | `T.copy` → `asc_copy_gm2ub_align` / `asc_copy_ub2gm_align`（探针 P1：`1, 4096` 单描述符） | ✅ | — |
| V12 | 切片装载循环 `for s in T.serial(split_k)` | — | 保留为 IR 循环，体内是 1 条 DMA | ❌ | **不属于逐元素标量点**：该循环每迭代只发射 1 条 DMA 指令（块级搬运序列）；把它展开/流水化是 Stage 4 的旋钮（§6.3），不是标量计算。 |
| V13 | 列 chunk 循环 `for ch in T.serial(BC//64)` | — | 保留为 IR 循环，体内全是 64-lane 向量指令 | ❌ | **它就是向量化本身**：向量寄存器宽度固定 64 个 fp32 lane，`BC/64` 次迭代是「一个块的宽度 = 若干寄存器」的表达；`T.unroll` 只改指令排布（Stage 4 旋钮），不改变是否向量化。 |
| V14 | 块索引（`row = bx // nc`、`c0 = (bx % nc)·BC`、`bc = T.min(BC, N-c0)`） | — | 无 | ❌ | **block 级索引 / 任务映射**：每块 3 次整数运算（探针 P1 折成 `block_idx >> 2`、`& 3`、`% 7`），非逐元素热点。 |
| V15 | host 侧校验（`activation` 取值、门控 `n` 为偶数） | Python 层 `raise` | 无（不在 kernel 内） | ❌ | **host 侧元数据计算**：构建期常量判定，不进 kernel。 |

**向量化结论**：逐元素与归约计算点 **V1–V11 全部向量化或整体消除**（V3/V4 被块级映射消除，
V5–V10 落 `T.simd.*`，V11 落 DMA），所用 API 全部有本平台证据（`simd.py` 导出表 +
`PL-950-vector-execution` + engram 已验用例 + 本任务探针 P1–P6 的生成代码，见 §3.4）。
保留 **4 处**循环/标量：V12（DMA 发射循环）、V13（向量宽度迭代）、V14（block 索引）、
V15（host 元数据）——其中 V12/V13 都**不是标量计算点**（体内分别只有 1 条 DMA 与纯向量指令），
V14/V15 属「block 级索引」「host 元数据」两类可接受理由。**实现方案中不存在逐元素标量循环**：
探针的 `float2` 标量循环计数器在全部 7 个用例上为 **0**。与 §6 的循环表一致（§6 表中只有块级
并行、DMA 序列循环、向量 chunk 循环与静态展开的切片链）。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

按算子类别（**逐元素 / 广播类**，含一个「切片」规约轴）枚举候选：I/O 原生布局 + 连续轴选择 ×
广播轴与向量轴的对齐 × 累加形态。

**轴质量评分（候选逐项）**

| 评分项 | 说明 | 候选 1（主选：列轴） | 候选 2（行轴） | 候选 3（扁平轴） |
|---|---|---|---|---|
| 向量化轴 | lane 映射到什么轴 | `j`（`N` 方向，行内连续），64 lane | `i`（行方向），lane = 相邻行 | `m·N` 扁平 |
| 整除性 | 轴长 vs 向量宽度 | `N ∈ {2048, 7168, 1536, 5120}` 全部 `N % 64 == 0` → 恰好整数个 chunk（BC 规则优先取 `N` 的约数，§5.2） | `m = 32` → 每向量寄存器只用 32/64 lane | `m·N` 是 64 的倍数，但每块会跨行 |
| 尾 lane 浪费率 | `(ceil(len/64)·64 − len)/len` | **0%**（4 条真实调用；非约数 `N` 走 `update_mask`，浪费 < 1/64） | 50% | 0%（但跨界） |
| 累加链形态 | 串行维是否阻碍向量化 | 切片轴（`split_k ≤ 4`）是**寄存器内链式加法**，不占 lane → 无阻碍 ✅ | 同 ✅ | 同 ✅ |
| repack 代价 | 核内重排成本 | **无**（部分和是 `[S,m,n]` 行主序，`w[s,row,c0:c0+bc]` 本身就是连续 `4·bc` 字节） | 需 `m` 个跨 `N` 步长的 gather（`stride = 4·N`）→ 描述符无法表达，退化为逐步长搬运 | 无 |
| 广播轴对齐 | bias 沿 `j` 广播 | ✅ 与向量轴一致（一次 `vlds_unpack` + 广播加） | ❌ 每 lane 一个不同 `j` 的 bias → 需 `stride` gather | ✅（同候选 1） |
| UB 容量影响 | 驻留预算 | 8–20 KiB（3–8%） | 同量级 | 同量级 |

**候选矩阵**（全部枚举；弃选给量化理由）

| # | 布局方案 | 向量化轴 | repack 路径 | 预估收益 / 代价 | 是否采纳 |
|---|---|---|---|---|---|
| D1 | **I/O 原生 `[S,m,n]` + 行内列 chunk 流** | `j`（chunk = 64 fp32 = 1 个向量寄存器） | 无 | 尾 lane 0%；每 chunk 9–21 条向量指令；bias 广播与向量轴同轴 | ✅ **主选** |
| D2 | 行轴（lane = 行） | `i` | 需按 `N` 步长 gather | lane 只用 50%；bias 变成跨 lane 的非连续读 → 反而退化；`m=32` 时无法填满 56 核 | ❌ 弃选（对齐 + 并行度双输） |
| D3 | 扁平 `m·N`（源形态） | 扁平索引 | 无（但块跨行） | 恢复逐元素 `//`、`%` 与跨行块边界（一个块的 lane 落在不同行，写回变成两段不连续 descriptor）；探针已用 D1 生成单段 `1, 4096` 描述符；推断 D3 会拆成 `1, r1` + `1, r2` 两段（未实测） | ❌ 弃选（标量热点回归 + 写回拆段） |
| D4 | 两块分工（gate 一个块、up 另一个块） | `j` | 无 | 逻辑块数 ×2、每块 UB 减半；但每个 `(row, chunk)` 的 bias/索引开销 ×2，且两个块写同一输出行的不同列 → 无冲突但无收益 | ❌ 弃选（无收益，块数翻倍） |
| D5 | 切片轴放 lane（C7） | `s` | 无 | lane 用 4/64；每 chunk 一次 `vcadd` | ❌ 弃选（R3/R4） |

**辅决策（同属阻塞级）**：

| # | 决策点 | 采纳 | 弃选与理由 |
|---|---|---|---|
| S1 | 切片累加形态 | **全切片驻留 UB（`split_k` 份列块）+ 单帧链式 `vadd`**（探针 P1 形态） | 弃「逐切片 UB 累加往返（`vld`→`vadd`→`vsts` 回写累加缓冲）」：每片多 1 条 `vsts` + 1 条 `vld`（+2 指令/chunk），且引入 VST→VLD 回读依赖，需 `mem_bar("VST_VLD")`（`TRAP-950-simd-ub-store-load` 记录该依赖未隐式可见） |
| S2 | `T.Pipelined` 流水 | **不启用**（主选）；列 Stage 4 旋钮 | 弃选理由：整块只 4 条 DMA（4–16 KiB）与 ~84–144 条向量指令，装载与计算同量级；流水会引入 buffer 多版本（UB ×2）与 `num_stages` 组合，收益需实测（§6.3）。判定裕度依赖未实测常数 → 进 §9.1 的 Stage 4 清单而非主选 |
| S3 | 输出路径 | **`vcvt` → `vsts(PK_B32)` 到 UB → 单条 `T.copy` 到 GM** | 弃「向量指令直接写 GM」：`vsts` 的地址必须是 UB（`access_ptr` + 64 元素步进），SIMD_VF 块内禁止 GM 访问（§3.5.1） |
| S4 | 绑定 `BC` 的取值 | **按 §5.2 的构建期规则**（在 `{256, 512, 1024}` 中最小化 `ceil(grid/56)·BC`，并列取大） | 弃「固定单一 BC」：4 条 workload 的最优 BC 不同（256/512/1024），固定值在其中至少一条上多耗 20% 波次×块宽（§5.2 表） |

**布局决策结论**：选定 **D1**——核内布局 = `[split_k, BC]` 的 fp32 列块（+ 门控时的第二条同形流
+ `[BC]` 的 bias/输出块），**向量化轴 = `j`（列，chunk = 64 个 fp32 = 1 个向量寄存器）**，
**repack 路径 = 无**（部分和行主序天然连续，`T.copy` 单描述符 DMA 直接落 UB）；
辅助决策 S1 全切片驻留、S3 UB 中转写回、S4 按规则绑定 `BC`；S2 流水进 Stage 4。
该结论已同步落入 §3.3 伪代码（`g_ub[s, ch*64]`、`o_ub[ch*64]`）、§4.3 的缓冲形状与 §6 的循环结构
（内层向量维 = 列）。

**实验裁决计划（三件套；判定裕度依赖未实测常数）**：`BC` 的选择在「波次量化（`ceil(grid/56)`）」
与「每块固定开销 `f`」之间取舍，而 `f`（每块 DMA 建立 + VF 帧进入 + 块派发）在 950PR 上**未实测**
（平台 cost_measurements 为空）；主选的判定式把 `f` 当 0（纯流量模型），若真实 `f` 不可忽略，
更大 `BC` 会胜出。故：

- **主选（进 Stage 3）**：`BC = argmin_{BC ∈ {256,512,1024}} ceil(m·ceil(N/BC)/56)·BC`（并列取大），
  逐 workload 取值见 §5.2 / §5.6；结构 = §3.3 伪代码（探针 P1–P6 全部 COMPILE OK）。
- **备选（≤1 个，结构完整）**：**统一 `BC = 512`** 的同一 kernel（工厂参数差异，非新结构）。
  其分核三要素重算：逻辑块数 `grid = m·ceil(N/512)` = 128（ds gate/up）/ 448（ds down）/
  96（glm gate/up）/ 320（glm down）→ 3 / 8 / 2 / 6 波；UB = `2·4·512·4 + 2·512·2 + 512·2`
  = 19 456 B = 19.0 KiB（门控带 bias 的最坏情形）；尾块：4 条 workload 的 `N` 全被 512 整除 → 无尾块代码。
  选择它的原因：`BC` 固定后所有 shape 的每块 DMA 段（2048 B）与向量指令数（8 chunk × 9–21）
  相同，`f` 被摊薄得最均匀，且完全避免 `BC=256` 的 1 KiB 小段 DMA（`CONST-mte2-degradation`
  记录的「小块搬运是段数/指令受限」区间）。
- **裁决计划**：代表 shape = `ds-v3-t32 gate/up`（主选 BC=256，与备选差异最大）、
  `ds-v3-t32 down`（两者同为 4 波）、`glm45-t32 gate/up`（BC=512，与备选相同）、
  `glm45-t32 down`；指标 = `msopprof` 的 Task Duration(us)（Stage 4 口径，A/B 交错 ≥5 次）；
  **判定阈值**：备选 latency < 主选 × 0.9 才翻转；**按 shape 分片裁决**（本 kernel 是按
  `(split_k, m, n, activation, has_bias, out_dtype, BC)` 特化的工厂，允许按 shape 分派主选/备选，
  不必全局二选一）；裁决结果回写本节并进入 §9.1。
- **设计期探针（已完成，不是裁决替代）**：`history_version/design_probe_compile.py` 的 P1–P6
  只标定「结构可编译 + 生成什么指令」，不产生时延数字，故不替代 Stage 4 的实验裁决。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式：SIMD**（平台 `programming_modes.choices` 的 `simd` 键；与
`programming_modes.migration_default` 一致）。

### 2.2 选型理由

1. **算术构成**：本算子的全部有效算术是「逐元素加 / 乘 / `exp` / 除 / 类型转换」——
   `split_k-1` 次 add、bias 1 次 add、门控 5 条 op、1 次 cast，**没有乘法规约（MAC）**，
   所以不需要 Cube，也不存在 Cube+Vector 交替（不是 `hybrid`）。
2. **数据形态**：`[S, m, n]` fp32 行主序，块内 lane 是连续列 → SIMD 的 64-lane 寄存器加、
   `UNPK_B16`/`PK_B32` 打包、`update_mask` 尾谓词正好覆盖所需；`T.SimtVF`（线程域）在这里没有
   优势（每元素 op 数少、无跨 lane 归约需求），且 SIMT 路径的逐元素表达式会退化为标量。
3. **API 事实**：`tilelang.ascend.language` 导出 `T.SimdVF` / `T.simd.*`（`simd.py` 的
   `vld/vsts/vadd/vmul/vdiv/vexp/vneg/vcvt/pset/update_mask/alloc_local`），探针 P1–P6 生成的
   kernel 签名是 `extern "C" __global__ __vector__`（**纯 AIV 发射**，`__ubuf__` 缓冲、
   `__simd_vf__ inline void main_kernel_simd_vf_0`）。
4. **同 op 先例**：同批次的 `_gemm_basic_kernel` 选 `cube`（MAC 密集），
   `_silu_mul_fused_kernel`（另一设计任务）与 engram gate-conv 同样是纯 Vector；本算子的
   decode 段（`m = 32`）连一个 Cube tile 都填不满，Cube 路径（C3/C2）已在 §1.6.0 R4 被淘汰。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|---|---|
| 内存分配 | 只分配 **UB**（`T.alloc_shared`，scope `shared.dyn`）：`g_ub[split_k, BC]` fp32、（门控）`u_ub[split_k, BC]` fp32、`b_ub[1或2, BC]` dtype、`o_ub[BC]` c_dtype。无 L1/L0A/L0B/L0C |
| 计算方式 | `with T.SimdVF():` 一个帧内完成一个块的全部列 chunk：`vld` → 链式 `vadd` → `vadd`(bias) → `vdiv`/`vmul`(silu) → `vcvt` → `vsts` |
| 搬运方式 | `T.copy`：GM→UB（MTE2，`asc_copy_gm2ub_align`，块级描述符）、UB→GM（MTE3，`asc_copy_ub2gm_align`） |
| 线程组织 | 无 `threads` 参数（平台 `T.Kernel(N)` 签名不含线程数）；元素并行由 **64 lane** 承担 |
| 同步 | 全自动（§7）：探针生成 `asc_sync_notify/wait(PIPE_MTE2→PIPE_V)` 与 `(PIPE_V→PIPE_MTE3)` 各一对；无手工 flag、无 `mem_bar` |

---

## 3. API 映射设计

### 3.1 公式拆解

输入为 §1.6.1 的**优化后公式**（A1 + A2 已采纳）：

| 步骤 | 数学表达 | 说明 |
|---|---|---|
| 1 | `row = bx // nc`；`c0 = (bx % nc)·BC`；`bc = min(BC, N − c0)` | tile 任务映射（块级标量，§1.6.3 D1） |
| 2 | `g_s ← w[s, row, c0:c0+bc]`（`s = 0..S−1`） | `S` 条 fp32 列流 DMA（R2） |
| 3 | `u_s ← w[s, row, N+c0:N+c0+bc]`（仅门控） | 第二条列流，偏移是编译期常量 |
| 4 | `b_g ← bias[c0:c0+bc]`、`b_u ← bias[N+c0:…]`（仅 `has_bias`） | bias 以输入 dtype 搬入 |
| 5 | `g = Σ_{s=0}^{S−1} g_s`（切片升序左结合） | 64-lane 链式 `vadd`（R3，§1.6.1 O1 验证逐位同源） |
| 6 | `g += f32(b_g)`（仅 `has_bias`） | `vlds_unpack(B16) → vcvt → vadd` |
| 7 | `u = Σ_s u_s (+f32(b_u))`（仅门控） | 同 5/6 |
| 8 | `y = (g / (1 + e^{−g})) · u`（仅门控） | A1：`vneg → vexp → vadd → vdiv → vmul` |
| 9 | `y = g`（非门控） | — |
| 10 | `c[row, c0:c0+bc] = cast_c(y)` | `vcvt(ROUND_R, sat=False)` + `vsts(PK_B32)`；`c_dtype == fp32` 时跳过 vcvt（R5） |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API（`import tilelang.ascend.language as T`） | 参数 | 模式 |
|---|---|---|---|---|
| 1 | 一维核网格 + 块索引分解 | `T.Kernel(m*nc)`；`row = bx // nc`、`c0 = (bx % nc)*BC`、`bc = T.min(BC, N - c0)` | 仅一维；块级标量 | SIMD（核级） |
| 2/3 | 部分和列流装载 | `T.copy(W[s, row, c0:c0+bc], g_ub[s, :bc])`；门控 `T.copy(W[s, row, N+c0:N+c0+bc], u_ub[s, :bc])` | 源为 GM 三维 region（标量 + 区间），目的为 UB 二维 buffer 的行切片 | MTE2 |
| 2/3 | 切片循环 | `for s in T.serial(split_k):` | 构建期常量 trip count → 生成 `split_k` 条 DMA | MTE2 |
| 4 | bias 装载 | `T.copy(BIAS[c0:c0+bc], b_ub[0, :bc])`（门控再 `b_ub[1, :bc]`） | dtype 与输入一致 | MTE2 |
| 5 | 向量装载 | `T.simd.vld(g_ub[0, ch*64])` | 目标为 fp32 UB → `vlds_norm<float>` | SIMD |
| 5 | 链式累加 | `acc = T.simd.alloc_local((1,), "float32")`；`acc[0] = T.simd.vadd(acc[0], T.simd.vld(g_ub[s+1, ch*64]), mask)` | `alloc_local` 是跨 `T.serial` 迭代携带向量值的唯一合法载体 | SIMD |
| 6 | bias 提升 + 加 | `T.simd.vcvt(T.simd.vld(b_ub[0, ch*64], dist="UNPK_B16"), "float32", part=0)` → `T.simd.vadd(...)` | `dist="UNPK_B16"` 做 16→32 位提升；`part=0` 必须是构建期常量 | SIMD |
| 8 | silu | `T.simd.vneg` → `T.simd.vexp` → `T.simd.vadd(one, ·)` → `T.simd.vdiv` → `T.simd.vmul(·, u)` | `one = T.simd.vdup(T.float32(1), "float32", full)` | SIMD |
| 9/10 | 量化 + 写 UB | `T.simd.vcvt(y, c_dtype, round="ROUND_R", sat=False, part=0)`；`T.simd.vsts(o_ub[ch*64], ys, mask, dist="PK_B32")` | 16-bit 输出用 `PK_B32`；fp32 输出直接用 `vsts(dist="NORM_B32")`、不 vcvt | SIMD |
| 10 | 写回 GM | `T.copy(o_ub[:bc], C[row, c0:c0+bc])` | 一条 `asc_copy_ub2gm_align` | MTE3 |
| 10b | 尾 lane 谓词 | `mask = T.simd.update_mask(T.min(64, bc - ch*64), 32)`（`N % BC ≠ 0` 时；整除时段内不生成） | 谓词宽度 32（b32） | SIMD |
| — | 缓冲版本 | `T.annotate_buffer_versions({g_ub: 1, u_ub: 1, b_ub: 1, o_ub: 1})` | 每块单次使用 → 显式单版本，UB 预算可静态核算 | 调度 |
| — | 编译 | `tilelang.compile(main, target="ascend", out_idx=[])` | `C` 是参数（源 O3 的意图），不自动分配 | — |

### 3.3 计算伪代码

> **本节形态即探针 `history_version/design_probe_compile.py` 的代码**（P1–P6 全部 COMPILE OK，
> 见 §11.1）。四条硬约束（均由探针或平台文档确认）：① `T.SimdVF` 块内**不得**出现 GM
> `BufferLoad`（参数与数据必须先 `T.copy` 到 UB）；② 跨 `T.serial` 迭代的向量累加必须用
> `T.simd.alloc_local`；③ `vcvt` 的 `part` 必须是构建期常量；④ `vcvt fp32→fp32` 被前端拒绝，
> fp32 输出必须跳过 vcvt。

```python
def splitk_reduce_kernel(split_k, m, n, dtype="float16", activation="none",
                         has_bias=False, out_dtype=None):
    gated = activation == "silu_and_mul"
    out_n = n // 2 if gated else n            # 构建期；门控要求 n 为偶数（host 校验）
    c_dtype = out_dtype or dtype
    VL = 64                                    # fp32 向量寄存器宽度
    BC = _pick_block_cols(m, out_n)            # §5.2 规则，∈ {256, 512, 1024}
    nc = ceildiv(out_n, BC)                    # 每行的列块数
    grid = m * nc
    tail = (out_n % BC) != 0
    nch = BC // VL
    pk = "PK_B32" if itemsize(c_dtype) == 2 else "NORM_B32"

    @T.prim_func
    def main(W: T.Tensor((split_k, m, n), "float32"),
             BIAS: T.Tensor((n,), dtype),          # 仅 has_bias 时出现在签名
             C: T.Tensor((m, out_n), c_dtype)):
        with T.Kernel(grid) as bx:
            row = bx // nc; c0 = (bx % nc) * BC
            bc = T.min(BC, out_n - c0)             # 块级（整除时被折成常量）
            g_ub = T.alloc_shared((split_k, BC), "float32")
            u_ub = T.alloc_shared((split_k, BC), "float32") if gated else None
            b_ub = T.alloc_shared((2 if gated else 1, BC), dtype) if has_bias else None
            o_ub = T.alloc_shared((BC,), c_dtype)
            T.annotate_buffer_versions({...: 1})   # 单版本（见 §4.5）

            for s in T.serial(split_k):                       # R2：S 条列流 DMA
                T.copy(W[s, row, c0:c0 + bc], g_ub[s, :bc])
                if gated:
                    T.copy(W[s, row, out_n + c0:out_n + c0 + bc], u_ub[s, :bc])
            if has_bias:
                T.copy(BIAS[c0:c0 + bc], b_ub[0, :bc])
                if gated:
                    T.copy(BIAS[out_n + c0:out_n + c0 + bc], b_ub[1, :bc])

            with T.SimdVF():
                full = T.simd.pset(32)
                one = T.simd.vdup(T.float32(1), "float32", full)
                for ch in T.serial(nch):                      # 向量化：每 chunk 64 lane
                    mask = T.simd.update_mask(T.min(VL, bc - ch * VL), 32) if tail else full
                    g = T.simd.alloc_local((1,), "float32")
                    g[0] = T.simd.vld(g_ub[0, ch * VL])       # R3：切片升序链
                    for s in T.serial(split_k - 1):
                        g[0] = T.simd.vadd(g[0], T.simd.vld(g_ub[s + 1, ch * VL]), mask)
                    if has_bias:                              # bias 在链尾加（同源顺序）
                        g[0] = T.simd.vadd(g[0], _upconvert(b_ub[0, ch * VL]), mask)
                    if gated:
                        u = T.simd.alloc_local((1,), "float32")
                        u[0] = T.simd.vld(u_ub[0, ch * VL])
                        for s in T.serial(split_k - 1):
                            u[0] = T.simd.vadd(u[0], T.simd.vld(u_ub[s + 1, ch * VL]), mask)
                        if has_bias:
                            u[0] = T.simd.vadd(u[0], _upconvert(b_ub[1, ch * VL]), mask)
                        den = T.simd.vadd(one, T.simd.vexp(T.simd.vneg(g[0], mask), mask), mask)
                        y = T.simd.vmul(T.simd.vdiv(g[0], den, mask), u[0], mask)   # A1
                    else:
                        y = g[0]
                    ys = y if c_dtype == "float32" else T.simd.vcvt(
                        y, c_dtype, round="ROUND_R", sat=False, part=0)
                    T.simd.vsts(o_ub[ch * VL], ys, mask, dist=pk)                   # R5
            T.copy(o_ub[:bc], C[row, c0:c0 + bc])             # MTE3
    return tilelang.compile(main, target="ascend", out_idx=[])


def _upconvert(x):   # 16-bit UB → fp32 向量（UNPK_B16 提升）
    return T.simd.vcvt(T.simd.vld(x, dist="UNPK_B16"), "float32", part=0)


def _pick_block_cols(m, out_n):                   # §5.2 规则
    cands = [256, 512, 1024]
    return min(cands, key=lambda bc: (ceildiv(m * ceildiv(out_n, bc), 56) * bc, -bc))
```

### 3.4 API 可行性确认

| API | 来源（本平台证据） | 设计期探针 | 状态 |
|---|---|---|---|
| `T.Kernel(grid)`（一维，无 `threads`） | `references/programming.md`「一维核网格；签名没有 `threads`」；`ascend/language/kernel.py:Kernel` | P1–P6 全部 | ✅ 已编译（`extern "C" __global__ __vector__ main_kernel(...)`） |
| `T.SimdVF()` + `T.simd.*` | `references/programming.md`；`PL-950-vector-execution`；engram gate-conv（已验） | P1–P6 | ✅ 已编译（生成 `__simd_vf__ inline void main_kernel_simd_vf_0`） |
| `T.alloc_shared`（UB） | `references/programming.md`（默认 scope `shared.dyn`） | P1–P6（`__ubuf__ uint8_t* buf_dyn_shmem`） | ✅ |
| `T.copy`（GM→UB，三维源标量索引 + 区间） | `copy_op.py:copy`（局部拷贝用显式 region）；engram 的 `T.copy(H[b,r,:], h_ub[:d])` | P1：`asc_copy_gm2ub_align(..., 1, 4096, …, 4096, 4096)` | ✅ |
| `T.copy`（UB→GM，dtype 收窄已在 UB 完成） | 同上 | P1/P2/P5：`asc_copy_ub2gm_align(..., 1, 2048/1024, …, …, …)` | ✅ |
| 运行期 region 端点（`bc = T.min(BC, N-c0)`） | 同仓 `_gemm_basic_kernel` 的 `am/an` 区段截断（其 §5.4 已验） | P3（`update_mask` 运行期形态） | ✅ |
| `T.simd.vld` / `vsts`（`dist="NORM"`/`"UNPK_B16"`/`"PK_B32"`） | `simd.py:vld/vsts`；engram 的 `UNPK_B16`/`PK_B32` 链；`PL-950-vector-execution` | P1/P2/P4/P6 | ✅（生成 `vlds_norm<float>`、`vlds_unpack<bfloat16_t>`、`vsts_pack_b32`） |
| `T.simd.vadd/vmul/vdiv/vexp/vneg/vcvt` | `simd.py` 导出；engram（`vexp/vdiv/vsqrt/vneg` 实跑过）；`PL-950-vector-execution` | P1/P2 | ✅（`vdiv` → `vdiv_0ulp_ftz_true`；`vcvt` → `vcvt<half>(..., ROUND_R, RS_DISABLE)`） |
| `T.simd.vcvt(fp32→fp32)` | `simd.py:vcvt` 的目标 dtype 列表**不含 float32→float32** | **P5 实测报错**：`ValueError: Unsupported vcvt conversion float32->float32` | ❌ 禁止（已按 §3.3 的 `ys = y` 分支改写，P5 复测通过） |
| `T.simd.alloc_local((1,), "float32")` 跨 `T.serial` 携带 | `simd.py:alloc_local` docstring；engram（`hs/ks/dot/vs/acc` 累加器） | P1/P2（`vector_f32 g[1]; vector_f32 u[1];`） | ✅ |
| `T.simd.pset(32)` / `update_mask(v, 32)` | `simd.py`；`PL-950-row-primitives`/`PL-950-simd-reduce` | P1（`asc_create_mask_b32(PAT_ALL)`）、P3（`update_mask`） | ✅ |
| `T.simd.vdup`（标量广播常量） | `simd.py`；engram（`vdup(T.float32(...))`） | P1/P2（`auto one = vdup<float>(float(1.0f), mask, …)`） | ✅ |
| `T.annotate_buffer_versions({buf: 1})` | `annotations.py`（正整数；`{buf: 1}` 取消自动多缓冲） | P1–P6 均未触发多版本（UB 偏移与手算一致） | ✅ |
| `tilelang.compile(main, target="ascend", out_idx=[])` | 平台 ruleset「使用 `tilelang.compile(..., target="ascend", out_idx=...)`」 | P1–P6 | ✅（`C` 由调用方传入，与源 O3 一致） |
| `T.annotate_compile_flags(["-O3"])` | `language/eager/builder.py:1220`（本平台导出） | 未使用（默认编译选项已足） | ⏸ 可选 |
| `T.sigmoid`（高层内建） | 命名空间存在，但**无 SIMD 语义证据** | — | ❌ 不使用（§0.5 O8 展开为显式序列） |
| `T.reduce_sum`（二维 shared→shared） | `CG-950-simd-shared-reduce`（open 能力缺口：输出布局差异） | — | ❌ 不使用（本 kernel 无跨 lane 归约需求） |
| `T.clear`（大 fp32 缓冲） | 会降级为标量 `float2` 循环（engram 探针记录） | — | ❌ 不使用（本 kernel 无清零需求） |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束（来源 `references/programming.md` / `references/ascend-constraints.md`） | 涉及 | 处理方案 |
|---|---|---|
| `T.Kernel` 只支持一维网格、无 `threads` | ✅ | `T.Kernel(m*nc)`，`row/c0` 在 kernel 内分解（§3.3） |
| `T.SimdVF` 块内禁止访问 GM | ✅ | 所有 GM 访存都在 `T.copy`（帧外）；探针 P1–P6 无违规报错 |
| 向量值不得跨 IR `for` 迭代用 Python 变量重绑定 | ✅ | `acc/up` 用 `T.simd.alloc_local((1,), "float32")`（§3.3 约束②） |
| `vcvt` 的 `part` 必须是构建期常量 | ✅ | 只传 `part=0`；chunk 循环不参数化 `part` |
| `vcvt` 不支持 fp32→fp32 | ✅ | fp32 输出跳过 vcvt（P5 实测后修正） |
| 块级循环上界需静态 | ✅ | `split_k`、`nch = BC//64`、`nc`、`grid` 全为构建期常量 |
| `T.copy` 的整 buffer→整 buffer 要求元素总数相等 | ✅ | 全部用**显式 region**（两侧都给区间） |
| 尾块：`N % BC ≠ 0` | ✅ | `bc = T.min(BC, N-c0)` + `update_mask`（P3）；`BC` 规则优先取 `N` 的约数（4 条真实调用全部整除，§5.2） |
| 尾 lane：`N % 64 ≠ 0` | ✅ | 同上（`update_mask(min(64, bc-ch*64), 32)`）；L0-3 用 `N=1000` 覆盖 |
| UB 容量 253952 B + 自动多版本膨胀 | ✅ | 最坏 20.0 KiB（8.1%），×1.7 上界 34.8 KiB（13.7%）（§4.5） |
| L1 / L0A / L0B / L0C 容量 | ❌ 不涉及 | 无 Cube、无 L1 中转 |
| 跨核同步 / 跨 block barrier | ❌ 不涉及 | 每块独立、核间无数据交换（这也是 C3 被淘汰的原因） |
| `annotate_unlimit_memory` 不增加物理容量 | ❌ 不使用 | 不调用 |
| 手工 flag 配对 / `mem_bar` | ❌ 不需要 | 无「向量写 → 向量读」回读（`TRAP-950-simd-ub-store-load` 断言 A：`PK_B32` 存 + DMA 拷出是精确的，无需 barrier） |
| fast-math / `compile_flags` | ✅ 不启用 | 不加 `TL_ENABLE_FAST_MATH`：保持 `vdiv` 的精确路径与 `vmul+vadd` 不收缩，使 §1.6.1 A1 的机器验证前提成立 |

#### 3.5.2 参考实现差异说明

（完整分析见 §0.5/§0.6，此处只列影响 API 选型的差异）

| 差异项 | 参考实现（GPU 源算子） | 本项目（Ascend 950PR） | 转换方案 |
|---|---|---|---|
| Kernel 结构 | `T.Kernel(ceildiv(total, 1024), threads=256)` 一维扁平 grid + 256 线程 | `T.Kernel(m*nc)` 一维 grid + `T.SimdVF` 向量域 | §0.6 R1 |
| 元素映射 | `gi = idx // out_n`、`gj = idx % out_n`（逐元素） | `row`/`c0` 块级标量，lane = 列 | §0.6 R1 / §1.6.2 V3 |
| 范围守卫 | `if idx < total`（逐元素） | 无（grid 由 `N` 导出）+ 尾 lane 谓词 | §1.6.2 V4/V10 |
| 累加 | 标量表达式树（每线程 `split_k` 次 load） | 64-lane `vadd` 链（`alloc_local`） | §0.6 R3 |
| 中间存储 | 无（寄存器） | UB 列块 `[split_k, BC]` fp32 + 输出列块 | §4.3 |
| 量化 | `T.cast(..., c_dtype)` 逐元素 | `vcvt(ROUND_R, sat=False)` + `vsts(PK_B32)` | §0.6 R5 |
| 门控 | `gate * T.sigmoid(gate) * up` | `(g / (1+e^{-g})) · u`（A1） | §1.6.1 A1 |
| bias | `T.cast(bias[gj], 'float')` | `UNPK_B16 → vcvt → vadd`（bias 先搬入 UB） | §3.2 |
| 输出分配 | `C` 由调用方预分配（非 `out_idx`） | 同（`out_idx=[]`） | 保留（O3） |
| 编译配置 | `TL_ENABLE_FAST_MATH: gated`、`-O3 -DENABLE_BF16` | 无 fast-math；`ENABLE_BF16` 不需要 | §0.5 O7/O10 |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|---|---|---|
| `examples/engram_gate_conv/_engram_gate_conv_fwd_kernel/_engram_gate_conv_fwd_kernel.py` | **最高**（同平台、纯 AIV SIMD、已通过的 kernel） | `T.SimdVF()` + `T.simd.*` 全套写法；`UNPK_B16`/`PK_B32` 链；`update_mask`；`alloc_local` 累加器；`vsts` 后**直接** `T.copy` 到 GM（无 barrier）；`annotate_buffer_versions({…: 1})` |
| `examples/engram_gate_conv/_engram_gate_conv_fwd_kernel/repro/TRAP-950-simd-ub-store-load.py` | **高**（直接的写入路径证据） | 断言 A：`vsts(dist="PK_B32")` + `T.copy` 拷出**逐字节正确、无需 `mem_bar`** —— 这正是本 kernel 的写回路径 |
| `examples/engram_gate_conv/_engram_gate_conv_fwd_kernel/DESIGN.md` | **高**（同平台 §1.6.0/§1.6.3/§5.5 的范式与本任务一致） | 调研四问 + roofline 的写法；SIMD 分核三要素；`T.SimdVF` 的四条硬约束 |
| `examples/mla_dense_bwd/_delta_kernel/DESIGN.md` | 中 | 同平台 SIMD kernel 的设计写法；`immutable variable` 与二维 grid 的陷阱 |
| `.agents/platforms/toolchains/ascend/knowledge/PL-950-vector-execution.md` | 中（已核验条目） | SIMD 路径 + 尾块（fp32 N=256/257）的仿真验证 |
| `.agents/platforms/toolchains/ascend/kernel_template.py` | 中（骨架） | `T.Kernel(grid)` + `--level L0/L1/L2/boundary` 的测试骨架 |
| `examples/shared_expert_mlp/_gemm_basic_kernel/DESIGN.md` | 中（**上游契约**） | `[split_k, m, n]` fp32 部分和契约（§4.2）、`split_k` 的可用性约束（§5.2）、wrapper 缺陷（§9.2 E6） |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|---|---|---|---|
| `W`（部分和） | `(split_k, m, n)` | **float32** | 上游 `_gemm_basic_kernel` 的 `split_k > 1` 输出（其 §4.2 契约）；本 kernel 只读 |
| `BIAS` | `(n,)` | float16 / bfloat16 | 仅 `has_bias=True` 时出现在签名；本 op 的 4 次调用都不传 |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|---|---|---|---|
| `C` | `(m, out_n)`，`out_n = n//2`（门控）或 `n` | `out_dtype or dtype` | 源算子输出 shape `C [m,n]`（非门控）；**由调用方预分配并传入**（`out_idx=[]`），不在 kernel 内分配 |

### 4.3 中间缓冲区

| 缓冲 | Shape | dtype | 用途 | 版本数 |
|---|---|---|---|---|
| `g_ub` | `(split_k, BC)` | float32 | gate 半的 `split_k` 份部分和列块 | 1 |
| `u_ub` | `(split_k, BC)` | float32 | up 半（仅门控） | 1 |
| `b_ub` | `(1 或 2, BC)` | 输入 dtype | bias 列块（gate 半 / up 半） | 1 |
| `o_ub` | `(BC,)` | c_dtype | 输出列块（量化后、DMA 写回前） | 1 |

无 L1 / L0A / L0B / L0C 缓冲，无 GM 中间量。

### 4.4 内存搬运路径

```
GM(W, fp32) ──MTE2 × split_k（每片 4·bc 字节连续）──▶ g_ub（UB）
GM(W, fp32) ──MTE2 × split_k（偏移 N+c0，仅门控）────▶ u_ub（UB）
GM(BIAS, dtype) ──MTE2 × 1~2（2·bc 字节）──────────▶ b_ub（UB）
UB(g_ub/u_ub/b_ub) ──V（SimdVF：vld → vadd 链 → silu → vcvt）──▶ o_ub（UB）
UB(o_ub) ──MTE3 × 1（2·bc 或 4·bc 字节）───────────▶ GM(C)
```

单块 5–9 条 DMA 指令（门控 + bias 最坏：4+4+2 入 + 1 出 = 11 条），全部是**块级描述符**
（探针 P1：`1, 4096` 即 1 行 × 4096 B 单描述符）。

### 4.5 UB 内存预算

`BC` 按 §5.2 的规则逐 shape 绑定；下表给出全部实际配置与最坏情形
（`ub_size = 253952 B`，来源 `Ascend950PR_9579.ini [AICoreSpec]` 第 58 行，
`ascend-950pr-9579-notes.md`）：

| 配置 | `g_ub` | `u_ub` | `b_ub` | `o_ub` | 合计 | / 248 KiB | ×1.7 膨胀上界 |
|---|---|---|---|---|---|---|---|
| ds gate/up（门控，无 bias，BC=256） | 4 096 | 4 096 | 0 | 512 | **8 704 B**（8.5 KiB） | 3.4% | 5.8% |
| ds gate/up（门控 + bias，BC=256） | 4 096 | 4 096 | 1 024 | 512 | 9 728 B（9.5 KiB） | 3.8% | 6.5% |
| ds down（非门控，无 bias，BC=1024） | 16 384 | 0 | 0 | 2 048 | **18 432 B**（18.0 KiB） | 7.3% | 12.4% |
| ds down（非门控 + bias，BC=1024） | 16 384 | 0 | 2 048 | 2 048 | 20 480 B（20.0 KiB） | **8.1%**（最坏） | 13.7% |
| glm gate/up（门控，无 bias，BC=512） | 8 192 | 8 192 | 0 | 1 024 | 17 408 B（17.0 KiB） | 6.9% | 11.7% |
| glm gate/up（门控 + bias，BC=512） | 8 192 | 8 192 | 2 048 | 1 024 | 19 456 B（19.0 KiB） | 7.7% | 13.0% |
| glm down（非门控，无 bias，BC=1024） | 16 384 | 0 | 0 | 2 048 | 18 432 B（18.0 KiB） | 7.3% | 12.4% |
| 退化（`split_k=1`，门控 + bias，BC=512） | 2 048 | 2 048 | 2 048 | 1 024 | 7 168 B（7.0 KiB） | 2.8% | 4.8% |

交叉核对：探针 P2（门控 + bias，BC=512）生成的代码里 `o_ub` 起始于 `bf16` 偏移 9216
（= 18 432 B），总占用 9 216·2 + 512·2 = 19 456 B —— 与上表逐字节一致（说明编译器未插入额外的
隐藏缓冲，`annotate_buffer_versions({…: 1})` 把自动多版本关掉后 UB 预算可静态核算）。

### 4.6 动态轴

**无动态轴**：`split_k`、`m`、`n`、`dtype`、`activation`、`has_bias`、`out_dtype`、`BC` 全部是
JIT 构建期常量（与源算子相同 —— 源也是 `@target_cache` 的形状特化工厂）。运行期唯一的变量是
块号 `bx`。

### 4.7 JIT 配置

```python
# 与源实现的差异：
#  - 去掉 pass_configs={TL_ENABLE_FAST_MATH: gated}（§0.5 O7：保持精确 vdiv 与不收缩
#    的 vmul+vadd，使 §1.6.1 A1 的机器验证前提成立）
#  - 去掉 compile_flags=["-O3", "-DENABLE_BF16"]（ENABLE_BF16 是 CUDA 宏；ascend 后端按
#    dtype 直接支持 bf16。若 Stage 3 需要显式优化级别，用 T.annotate_compile_flags(["-O3"])，
#    该接口在本平台导出：language/eager/builder.py:1220）
#  - out_idx=[]：C 由调用方预分配（源 docstring 的意图，O3 保留）
tilelang.compile(main, target="ascend", out_idx=[])
```

自动调度保持默认开启（平台 `allow_autoschedule` 默认值），由编译器插入
`MTE2→V`、`V→MTE3` 的事件配对（§7）。

---

## 5. Tiling 策略

### 5.1 计算类型

**类型：纯 Vector（SIMD）逐元素 + 寄存器内规约**。判定依据：算子只含「逐元素加/乘/超越函数/
类型转换」与一个长度为 `split_k ≤ 4` 的**寄存器内**加法链，无 MAC、无跨 lane 归约、无跨核数据
交换；Cube 单元不参与（§1.6.0 R4 的 C2/C3 已淘汰）。带宽 + 归约型（AI ≈ 0.4–0.5 lane-op/字节），
**不用 MFU 口径**。

### 5.2 Block 划分

**Tiling 的自由变量是「每块多少列」（`BC`）**，行方向每块固定 **1 行**（`m` 只有 32，且行间无复用）。

```python
# 构建期规则（实现于工厂，非运行期）
VL   = 64                                   # fp32 向量寄存器宽度
CAND = [256, 512, 1024]                     # 候选块宽（均为 VL 的整数倍）
BC   = min(CAND, key=lambda bc: (ceil(m * ceil(N / bc) / 56) * bc, -bc))
                                            # 最小化「波次 × 块宽」；并列取更大块宽
nc   = ceil(N / BC);  grid = m * nc          # 逻辑块数
nch  = BC // VL                              # 每块的向量 chunk 数
waves = ceil(grid / 56)                      # 波次（56 = 物理 Vector 核数，§5.5）
```

| workload | `m` | `N` | `BC` | `nc` | **`grid`** | `waves` | 每块 chunk | 三候选的 `waves×BC`（256 / 512 / 1024） |
|---|---|---|---|---|---|---|---|---|
| **ds-v3-t32 gate/up**（门控） | 32 | 2048 | **256** | 8 | **256** | 5 | 4 | **1280** / 1536 / 2048 |
| **ds-v3-t32 down** | 32 | 7168 | **1024** | 7 | **224** | 4 | 16 | 4096 / 4096 / **4096**（并列取大） |
| **glm45-t32 gate/up**（门控） | 32 | 1536 | **512** | 3 | **96** | 2 | 8 | 1024 / **1024** / 2048（并列取大） |
| **glm45-t32 down** | 32 | 5120 | **1024** | 5 | **160** | 3 | 16 | 3072 / 3072 / **3072**（并列取大） |
| （若 `m = 1` 且 `split_k > 1` 被调用） | 1 | 18432 / 7168 | 512 / 1024 | 36 / 7 | 36 / 7 | 1 / 1 | 8 / 16 | 见 §9.1 R7（Stage 5 修复后不出现） |

**Tiling 语义**：`grid` 条任务 → 每条 1 行 × `BC` 列 → 每块 `nch` 个 64 元素向量段。
所有层都是构建期常量，运行期只有 `bx`。

### 5.3 约束分析

- **对齐**：`BC ∈ {256,512,1024}` 全部是 64（= 1 个 fp32 寄存器）的整数倍；4 条真实 workload 的
  `N ∈ {2048, 7168, 1536, 5120}` 都至少被其中一个候选整除，且规则选中的 `BC` **都整除 `N`**
  → 真实 workload **无尾块、无尾 lane**（尾路径只由 L0-3/L0-6 覆盖）。
- **DMA 段宽**：每片 `4·BC` 字节 = 1024 / 2048 / 4096 B 单描述符——远大于 128 B 的段粒度，
  避开 `CONST-mte2-degradation` 记录的「小块搬运段数受限」区间（该常数绑定 910B2C，仅作量级参考）。
- **UB**：最坏 20.0 KiB / 248 KiB = 8.1%（§4.5）。
- **L1/L0**：不使用。
- **块内并行度**：`nch = 4/8/16` 个独立 chunk → 每条 `vadd` 链之间无依赖，编译器可自由交错
  （`split_k ≤ 4` 的链深只有 3，不构成时延瓶颈）。

### 5.4 注意事项（非整除与边界）

| 场景 | 处理 |
|---|---|
| `N % BC == 0`（4 条真实 workload） | 无尾块：`bc` 被折成常量，`mask` 恒为 `pset(32)`，段内不生成 `update_mask`（探针 P1/P2/P6） |
| `N % BC != 0` | `bc = T.min(BC, N - c0)` 的运行期 region + `update_mask(min(64, bc-ch·64), 32)`（探针 P3） |
| `N % 64 != 0` | 同上（尾 lane 谓词覆盖） |
| 门控 `N` 为奇数 | **构建期拒绝**（与源一致：`ValueError("silu_and_mul requires an even output width")`）；`n` 为偶数 ⇔ `N = n/2` 为整数 |
| `split_k == 1` | 退化路径（R6）：只有 1 条流 DMA、0 条 `vadd`（探针 P4） |
| `has_bias=False` | bias 参数不出现在签名中（`out_idx=[]`，bias 为可选形参） |
| `c_dtype == "float32"` | 跳过 vcvt，`vsts(dist="NORM_B32")`（探针 P5） |
| `m` 或 `out_n` 为 1 | `grid = nc`；块内 `bc = min(BC, 1)` ⇒ 每个 chunk 只有 1 个有效 lane，其余被谓词屏蔽（正确但低效，L0-6 覆盖其正确性） |

### 5.5 分核策略（物理核数适配）

> 三要素判定标准与实查要求的权威版本：`.agents/skills/_shared/standards/core-split-strategy.md`
> （本任务按 §1 三要素与 §2.1 执行）。

**① 逻辑核数计算**

- `num_logical_kernels = grid = m × ceil(out_n / BC)`：
  `ds gate/up` → **256**；`ds down` → **224**；`glm gate/up` → **96**；`glm down` → **160**。
- 启动核数 = `grid`（一条任务一个核，**无核内多任务串行**）。
- 每核工作量：`nch × (9–21)` 条向量指令 + `split_k(+…)` 条 DMA。

**② 物理核数依据（实查记录）**

| 查询 | 命令 / 来源 | 实际返回值 |
|---|---|---|
| 平台任务上下文（本机实跑，2026-11-23） | `python3 .agents/tools/statectl.py platform --task-dir examples/shared_expert_mlp/splitk_reduce_kernel` | `hardware.hardware_id = "ascend-950pr-9579"`；`observed_cores = {"cube": 28, "vector": 56}`；`core_source = "Ascend950PR_9579.ini [SoCInfo] ai_core_cnt=28, cube_core_cnt=28, vector_core_cnt=56; runtime probe rtGetAiCoreCount()=28 on Ascend950PR_9579, 2026-11-23"` |
| 型号配置文件（只读复核） | `grep -n "core_cnt" /usr/local/Ascend/cann-9.2.1/x86_64-linux/data/platform_config/Ascend950PR_9579.ini`（同文件另查 `ub_size` / `l1_size` / `l0_c_size`） | 第 20–22 行：`ai_core_cnt=28`、`cube_core_cnt=28`、`vector_core_cnt=56`；第 50/51/58 行：`l0_c_size=262144`、`l1_size=524288`、`ub_size=253952` |
| `torch.npu.get_device_properties` | **本次未执行**：本机唯一的卡被同批次另一任务（engram decode Stage 3）占用，设计期约定只做**无卡编译探针**（`ASCEND_RT_VISIBLE_DEVICES=""`），初始化设备上下文会与并行的设备任务争用 | 以平台已记录的 runtime probe（上一行 `core_source`）为准，不猜测数值；**Stage 3 首次真机运行前须补 `torch.npu.get_device_properties` 并记录 `cube_core_num` / `vector_core_num`** |

**目标型号：`ascend-950pr-9579`**（dav-3510，28 AI Core / 56 Vector Core，UB 248 KiB）。
本算子为纯 Vector kernel，分核目标取 **56 个 Vector 核**（与同仓
`examples/engram_gate_conv/_engram_gate_conv_fwd_kernel/DESIGN.md` §5.5、
`examples/moe_permute_align/_make_scatter_kernel/DESIGN.md` 的取值一致）。

**③ 规模判定与分核方案**

- **规模判定：中等规模**（`grid = 96–256` > 56，但每块负载均匀、可由一次派发内的 2–5 波覆盖）。
- **分核方案：通过调整 `BC` 把「波次 × 块宽」压到最小**（§5.2 规则），结果
  `grid ∈ {96, 160, 224, 256}` → 波次 2 / 3 / 4 / 5，波内利用率
  96/(56·2)=85.7%、160/(56·3)=95.2%、224/(56·4)=100%、256/(56·5)=91.4%。
  每核 1 块、无核内串行、无跨核通信。
- **为什么不需要「极大规模 → 核内串行」**：块负载完全均匀（每块 `nch × 常数` 条指令），
  所以「逻辑核超发 + 硬件波派发」的时延等于 `ceil(grid/56) × 单块时延`，与「固定 56 核、
  核内串行 `ceil(grid/56)` 块」等价；后者只多省一点派发开销，却引入核内循环与 `T.Persistent`
  （平台无 `T.Persistent` 的正确性条目，见 `_gemm_basic_kernel` §5.5 的同结论）。
  上限情形（`N = 18432, m = 32`，BC=1024 → 576 块 / 11 波）同样是均匀块，结论不变。
- **核内循环边界**：`split_k`、`nch`、`nc` 全为构建期常量，满足平台「块级循环上界静态」要求。

### 5.6 每 workload 的推荐配置

| workload | 调用 | `BC` | `grid` | 每块 DMA（入/出） | 每块向量指令 | 全 kernel 向量指令 | 每核 |
|---|---|---|---|---|---|---|---|
| ds-v3-t32 gate/up | gate/up | 256 | 256 | 8 / 1 | 4×21 = 84 | 21 504 | 384 |
| ds-v3-t32 down | down | 1024 | 224 | 4 / 1 | 16×9 = 144 | 32 256 | 576 |
| glm45-t32 gate/up | gate/up | 512 | 96 | 8 / 1 | 8×21 = 168 | 16 128 | 288 |
| glm45-t32 down | down | 1024 | 160 | 4 / 1 | 16×9 = 144 | 23 040 | 412 |

（每 chunk 指令数：非门控 `split_k=4` = `vld`×1 + (`vld`+`vadd`)×3 + `vcvt` + `vsts` = **9**；
门控 `split_k=4` 无 bias = `7 + 7 + (vneg,vexp,vadd,vdiv,vmul)=5 + (vcvt,vsts)=2` = **21**。
带 bias 再加 3 条/半。计数取自探针 P1/P2 的生成代码，§11.1。）

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|---|---|---|---|
| 核网格 | block 级并行（一维） | `T.Kernel(grid)`，`grid = m·nc` | 每块 1 行 × BC 列（§5.2） |
| 块索引分解 | 块级标量 | `row = bx // nc`、`c0 = (bx % nc)·BC`、`bc = T.min(BC, N-c0)` | 每块 3 次整数运算（§1.6.2 V14） |
| 切片装载 | 核内串行（DMA 发射） | `for s in T.serial(split_k)` | 每迭代 1 条 `asc_copy_gm2ub_align`；`split_k ≤ 4` |
| 列 chunk | 向量化循环 | `for ch in T.serial(nch)` + `T.simd.*` | 每段 64 个 fp32 lane = 1 个向量寄存器（§1.6.2 V13） |
| 切片累加链 | 静态展开 | `for s in T.serial(split_k-1)` 的 `vadd` | 生成 `split_k-1` 条 `vadd`（探针：`for (s=0;s<3;++s)`） |
| silu | 无循环（5 条向量指令） | `vneg/vexp/vadd/vdiv/vmul` | A1 改写后 |
| 元素级 | **不使用 `T.Parallel`** | — | `SimdVF` 无线程域；元素并行由 64 lane 承担 |
| 流水 | **不使用 `T.Pipelined`** | — | 见 §6.3 |
| persistent | **不使用 `T.Persistent`** | — | 见 §5.5 ③ |

### 6.2 循环伪代码

见 §3.3（同一份伪代码即调度结构）：`T.Kernel(grid)` → 块索引分解 → `T.serial(split_k)` 装载
（+ bias）→ `T.SimdVF` 内 `T.serial(nch)` chunk 循环（内层 `T.serial(split_k-1)` 累加链）→
1 条 `T.copy` 写回。

### 6.3 流水线优化

- **主选不使用 `T.Pipelined`**：整块只有 4 条 1–4 KiB 的 DMA（合计 16 KiB）与 ~84–144 条向量
  指令，装载与计算同量级；`T.Pipelined` 会引入 buffer 多版本（UB ×2，最坏 40 KiB）与
  `num_stages` 组合，收益需实测（判定裕度落在未实测常数上 → 按 §1.6.3 记入 Stage 4 清单，
  不纸面拍板）。
- **已有的重叠**：同一块内 `for s` 的 `split_k` 条 DMA 是连续发射的（探针 P1：4 条
  `asc_copy_gm2ub_align` 后一个 `asc_sync_wait`），即 MTE2 内部已重叠；
  `MTE2→V` 与 `V→MTE3` 由自动调度插入事件（§7）。
- **Stage 4 可选旋钮（不改结构）**：① `T.Pipelined(split_k, num_stages=2)` + `g_ub/u_ub` 置 2 版本
  （UB +8–32 KiB），把「第 1 片到达」到「第 2 片到达」的等待藏起来；② `T.unroll` 替代 chunk 循环
  的 `T.serial` 以减少循环开销；③ 每核多块（核内串行）以摊薄派发。三者都登记为「未实测收益」。

### 6.4 尾块处理

| 场景 | 处理 |
|---|---|
| `N % BC == 0`（4 条真实 workload） | 无尾块、无尾 lane（`bc` 折成常量，`mask = pset(32)`） |
| `N % BC != 0` | 运行期 `bc = T.min(BC, N-c0)` + `update_mask`（探针 P3） |
| `out_n < BC` | `nc = 1`、`bc = out_n`；每 chunk 有效 lane 为 `min(64, bc-ch·64)` |
| `m` 小于 32（含 `m=1`） | `grid = m·nc`，无特殊处理（行方向无 tile 切分） |
| `split_k` 与形状的整除关系 | 无此约束：本 kernel 沿切片轴只做顺序读取，不要求它整除任何量（区别于上游 GEMM 的「`split_k` 整除 K tile 数」前置） |

---

## 7. 同步策略

### 7.1 同步模式

**模式：自动同步**（编译器 AutoSchedule + InsertSync 在 MTE2 / Vector / MTE3 之间插入事件）。
本设计**不写**任何 `ascend_set_flag` / `ascend_wait_flag` / `ascend_pipe_barrier`，也**不使用**
`T.simd.mem_bar`。

### 7.2 同步点说明

| 依赖 | 机制 | 证据 |
|---|---|---|
| MTE2（装载 `g_ub/u_ub/b_ub`）→ V（`vld` 读） | 编译器插入 `asc_sync_notify(PIPE_MTE2, PIPE_V, 0)` / `asc_sync_wait(...)` | 探针 P1/P2 生成代码（`if (s == 3) asc_sync_notify(PIPE_MTE2, PIPE_V, 0);` 后一个 `asc_sync_wait`） |
| V（`vsts` 写 `o_ub`）→ MTE3（`T.copy` 读出） | 编译器插入 `asc_sync_notify(PIPE_V, PIPE_MTE3, 0)` / `asc_sync_wait(...)` | 探针 P1/P2 生成代码 |
| V 内部（`vsts` → 后续 `vld` 同一 UB） | **不存在该依赖**：本 kernel 的每个 UB 缓冲要么只被 DMA 写/向量读（`g_ub/u_ub/b_ub`），要么只被向量写/DMA 读（`o_ub`），**没有任何「向量写 → 向量读」回读** | `examples/engram_gate_conv/_engram_gate_conv_fwd_kernel/repro/TRAP-950-simd-ub-store-load.py` 的断言 A：`vsts(PK_B32)` + DMA 拷出**逐字节正确，无需 `mem_bar`** |
| 块间 | 无（每块独立，核间无数据交换） | §5.5 ③ |

### 7.3 pass_configs 配置

```python
# 保持默认：不关闭自动调度、不改共享内存复用策略（平台规则：allow_autoschedule 默认开启）
# 不设置 TL_ENABLE_FAST_MATH（§0.5 O7 / §3.5.1）
# 若 Stage 4 发现需要显式优化级别：T.annotate_compile_flags(["-O3"])
```

---

## 8. 验证方案

### 8.1 Golden 函数

> 依据 **§0.1 的语义**独立实现（切片升序 fp32 和 + 可选 bias + 可选 `silu_and_mul`，一次舍入），
> **不复刻** §0.6 的 NPU 算法：不切块、不用向量操作、不做 UB 中转、不按 `BC` 遍历。
> `has_bias` 部分按源语义（bias 在 fp32 域、加在切片和之后、门控时两半各加）。

```python
import torch


def golden_splitk_reduce(w, dtype, bias=None, activation="none", out_dtype=None):
    """split-K 部分和归约的参考实现（与 NPU 结构无关）。

    Args:
        w: (split_k, m, n) float32 部分和（上游 GEMM 的输出契约）。
        dtype: 输入 dtype（bias 的 dtype / 默认输出 dtype）。
        bias: (n,) 可选列 bias，dtype = dtype。
        activation: "none" | "silu_and_mul"（后者要求 n 为偶数）。
        out_dtype: 输出 dtype，默认 = dtype。
    Returns:
        (m, out_n) ，out_n = n//2（门控）或 n。
    """
    s_dim, m, n = w.shape
    acc = w[0].to(torch.float32).clone()
    for s in range(1, s_dim):                    # 切片升序左结合（与源 _slice_sum 同序）
        acc = acc + w[s].to(torch.float32)
    if bias is not None:
        acc = acc + bias.to(torch.float32).unsqueeze(0)   # 列广播，fp32 域，链尾
    if activation == "silu_and_mul":
        assert n % 2 == 0, "silu_and_mul requires an even output width"
        gate, up = acc[:, : n // 2], acc[:, n // 2 :]
        acc = torch.nn.functional.silu(gate) * up
    elif activation != "none":
        raise ValueError(f"unsupported activation: {activation}")
    return acc.to(out_dtype or dtype)
```

端到端（op 级）对照时，golden 侧等价于 `w.sum(0)`（§1.6.1 O1 已验证两者在 fp16/bf16/fp32 网格上
0 ulp 差），因此 op 级参考 `tileops.workloads.moe.ref_shared_expert` 的口径与本 kernel 的 golden
一致（都是「fp32 累加 + 末次舍入」）。**独立性**：上式只用 torch 的逐元素加/乘/silu，不含任何
split-K tile、UB 缓冲或向量指令的痕迹。

### 8.2 精度标准

| dtype | atol | rtol | 附加判据 | 来源 |
|---|---|---|---|---|
| float16 | 1e-3 | 1e-3 | 期望值：与 golden 的差 **≤ 1 ulp**（本 kernel 与 golden 的累加顺序一致，实测 0 ulp；见 §1.6.1 O1） | Op 级容差表 `/workspace/TileOPs/workloads/numerics.py:reference_tolerance`（fp16 1e-3 / bf16 1.6e-2 / fp32 1e-5，`rtol = atol`） |
| bfloat16 | 1.6e-2 | 1.6e-2 | 同上 | 同上 |
| float32（`out_dtype="float32"`） | 1e-5 | 1e-5 | 同上 | 同上 |
| 非有限值 | — | — | `equal_nan=True`；位置模式必须与 golden 完全一致（§8.3 L0-8） | `torch.testing.assert_close(..., equal_nan=True)` |

**参考量口径**：partial 是 fp32（不是量化输入），因此不存在「输入量化误差」项；golden 无需先转
fp32 再算。容差表的实际裕度：fp16 1e-3 ≈ 2 个 ulp（1 ulp ≈ 4.9e-4），bf16 1.6e-2 ≈ 4 个 ulp
（1 ulp ≈ 3.9e-3）——即「期望 ≤1 ulp」之上留 2–4 倍，供回归检测。

### 8.3 L0 门槛测试计划

L0 = 小/真实 shape、覆盖**每个结构分支**（Stage 3 首跑必须全过；完整 L1/L2/boundary 套件由
`tilelang-op-develop` 展开）：

| 用例 | `(split_k, m, n)` | dtype / out_dtype | activation / has_bias | `BC` | 覆盖点 | 判定 |
|---|---|---|---|---|---|---|
| L0-1 | (4, 32, 4096) | bf16 / — | silu_and_mul / False | 256 | **真实几何**（ds-v3-t32 gate/up）：门控 + 双列流 + `out_n` 偏移 + 4 chunk/块 | vs golden，bf16 1.6e-2；另记 ≤1 ulp 的观测值 |
| L0-2 | (4, 32, 7168) | bf16 / — | none / False | 1024 | **真实几何**（ds-v3-t32 down）：非门控 + 切片链 + 16 chunk/块 + 224 块 | 同上 |
| L0-3 | (4, 32, 3072) | bf16 / — | silu_and_mul / False | 512 | **真实几何**（glm45-t32 gate/up）：`N=1536` 被 512 整除（无尾块） | 同上 |
| L0-4 | (4, 32, 4096) | float16 / — | silu_and_mul / False | 256 | fp16 输出路径（`vcvt<half>(..., ROUND_R, RS_DISABLE)` + `PK_B32`） | vs golden，fp16 1e-3 |
| L0-5 | (1, 4, 4096) | bf16 / — | silu_and_mul / False | 256 | **退化路径**（Stage 5 修复后 `m<16 → split_k=1`）：只有 1 条流 DMA、0 条 `vadd`；输出 = `silu(w0)·w0_up` | vs golden，1.6e-2 |
| L0-6 | (1, 2, 1024) | float16 / — | none / **True** | 256 | `split_k=1` + bias：纯拷贝 + bias（bias 的 `UNPK_B16 → vcvt` 路径） | vs golden，1e-3 |
| L0-7 | (3, 3, 2000) | bf16 / — | none / False | 256 | **尾块 + 尾 lane**：`out_n=2000`（2000 % 256 = 208，非 64 倍数）→ 运行期 `bc` + `update_mask` | vs golden，1.6e-2 |
| L0-8 | (2, 8, 512) | fp16 / **float32** | silu_and_mul / True | 256 | `out_dtype="float32"`（跳过 vcvt、`NORM_B32`）+ 门控带 bias | vs golden，1e-5 |
| L0-9 | (2, 4, 256) | bf16 / — | none / False | 256 | **边界值**：输入含 ±inf / NaN / 次正规数 / ±0（`partial` 直接构造） | `assert_close(..., equal_nan=True)` + 非有限位置模式一致 |

**门槛**：L0-1…L0-9 全过 = `[PRECISION_PASS]` 的最低要求；L0-1/L0-2 另记 kernel 级时延作为
Stage 4 基线。若出现「输出 dtype 差 > 1 ulp」，先查 §9.1 R2（`vdiv` 的 FTZ）与 R3（`sat`）。

---

## 9. 风险点与注意事项

### 9.1 已知约束与风险

| # | 风险 | 影响 | 缓解 / 判据 |
|---|---|---|---|
| R1 | **launch 固定开销与流量项同量级**（流量 1.67–4.13 MB ≈ 0.7–4.0 µs；launch 未实测，910B2C 量级 2–5 µs） | 本 kernel 的时延下界主要由「1 次 launch + 2–4 MB 流量」决定，指令级优化收益有限 | 设计目标定为「每字节只读/写一次 + 不拆 kernel + 不引入 GM 中间量」；不追求 MFU；Stage 4 只调 `BC` 与流水旋钮（§1.6.3 裁决计划） |
| R2 | `vdiv` 降为 `vdiv_0ulp_ftz_true`（探针 P2 实测）→ fp32 **次正规数 FTZ** | 只影响门控输入绝对值极小的情形（`g` → 0）；其 silu 结果本身 < 1e-37，经 16-bit 舍入后为 0 | L0-9 覆盖次正规输入；若出现分歧，A1 可退回「取倒数再乘」（只多 1 条 `vmul`，§1.6.1 已证明两式在输出 dtype 上同值） |
| R3 | `vcvt(..., sat=False)` 是保持 ±inf 的前提；若 Stage 3 误用默认 `sat=True`，±inf 会被夹到有限最大值 | 非有限语义与 golden 不一致 | L0-9 硬性覆盖；§3.2 的 API 表已固定 `sat=False` |
| R4 | bias 的 dtype（输入 dtype）与 `out_dtype` 可以不同（例如 fp16 bias + fp32 out） | 提升链必须与输出 dtype 解耦 | 提升固定为 `UNPK_B16 → vcvt(float32)`；L0-8 覆盖 fp16 bias + fp32 out |
| R5 | 门控 `n` 为奇数 | 语义未定义 | 构建期 `ValueError`（与源一致）；`n` 偶数 ⇔ `N` 整数 |
| R6 | **上游契约耦合**：`W` 必须是 `[split_k, m, n]` fp32 连续部分和 | 若 `_gemm_basic_kernel` 改成别的部分和布局，本 kernel 静默错 | 两文档互为契约（`examples/shared_expert_mlp/_gemm_basic_kernel/DESIGN.md` §4.2 ↔ 本文 §0.1/§4.1）；Stage 5 集成时以 wrapper 的 `mainloop`/`reduce_program` 配对为唯一入口 |
| R7 | **调用方缺陷（Stage 5 修复）**：`_split_k(T≤32)=4` 在 `m<16` 时违反上游精确 tile 前置 | 上游 GEMM 报错 | 本 kernel 的义务是 `split_k=1` 正确（L0-5/L0-6）；修复后 kimi-k2-t1 不再进入本 kernel |
| R8 | 每个 forward 有 **2 次**本 kernel 调用（T≤32 时），且与 GEMM 形成 `GEMM1 → reduce1 → GEMM2 → reduce2` 的严格链 | 无法把两次归约融合（后者依赖 GEMM2） | 记录事实，不在此层优化（wrapper 的职责；`ref` 一致）。Stage 4 若发现 launch 开销主导，只能由 op 级流水（跨 token 批）解决，属另一层设计 |
| R9 | UB 自动多版本膨胀（`TRAP-UB-multibuffer-inflation` 记录 ×1.7 上界） | 最坏 20.0 KiB → 34.8 KiB（13.7%） | 仍 ≪ 100%；`annotate_buffer_versions({...: 1})` 已把膨胀源钉死（§4.5 与探针 P2 逐字节核对一致） |
| R10 | `N % BC != 0` 的尾路径（真实 workload 不触发） | 若规则选到非约数，需要运行期 `bc` 与谓词 | L0-7 覆盖；`update_mask` 的宽度参数固定 32（b32） |
| R11 | `gate = -inf` 时两式同为 NaN（`-inf/(1+inf)` 与 `-inf·0`） | 与 torch 的 `x*sigmoid(x)` 同式；op 级断言用 `equal_nan=True` | 机器验证的 corner 用例已确认非有限模式一致（§1.6.1 表）；L0-9 再核一次 |
| R12 | 温度/测量缺口：950PR 无任何实测带宽/发射常数 | D-2 的下界只有量级意义 | §1.6.0 D-2 显式标「待核验」；Stage 4 的 `msopprof` Task Duration 是唯一验收口径 |

### 9.2 常见错误（本平台的实现陷阱，来自探针与同仓条目）

| # | 陷阱 | 症状 | 规避 |
|---|---|---|---|
| E1 | 在 `T.SimdVF` 块内直接读 GM（如 `W[s, row, ...]`） | `[TileLang Semantic Check] BufferLoad from global buffer ... inside a SIMD_VF block is not allowed`（engram 探针首轮遇到） | 所有 GM 访存走帧外 `T.copy`（§3.3 约束①） |
| E2 | 用 Python 变量跨 `T.serial` 迭代携带向量值 | `Immutable variable ... used outside its defining region` | `T.simd.alloc_local((1,), "float32")`（§3.3 约束②） |
| E3 | `vcvt` 的 `part` 传循环变量 | 前端报错（`part` 必须是构建期常量） | 只传 `part=0`（§3.3 约束③） |
| E4 | 对 fp32 输出调用 `vcvt(y, "float32")` | `ValueError: Unsupported vcvt conversion float32->float32`（**探针 P5 本轮实测到**） | `ys = y`（跳过 vcvt，`NORM_B32`） |
| E5 | `vsts` 的 `dist` 与 dtype 不匹配（16-bit 用 `NORM_B32`） | 只写一半字节 / 结果错乱 | `PK_B32` 配 16-bit、`NORM_B32` 配 32-bit（§3.2） |
| E6 | 在 `T.Kernel` 里写二维 grid | 平台实现拒绝（`_gemm_basic_kernel` §9.2 同结论） | 一维 grid + 块内分解（§5.2） |
| E7 | 用 `T.reduce_sum` 做二维 shared→shared 归约 | `CG-950-simd-shared-reduce`（open 能力缺口，输出布局差异） | 不用；本 kernel 无跨 lane 归约需求 |
| E8 | 忘记 `has_bias` 时 bias 的签名分支 | 编译期/运行期参数个数不匹配 | Python 层按 `has_bias` 决定 `prim_func` 的形参（§3.3） |

### 9.3 特殊场景处理

| 场景 | 处理 |
|---|---|
| `split_k == 1` 且无 bias 且 `activation="none"` | 纯 fp32→c_dtype 拷贝（探针 P4：`vld → vcvt → vsts_pack_b32`）；`c_dtype == fp32` 时为 `vld → vsts(NORM_B32)` |
| 门控 + `has_bias`（本 op 不出现，通用路径） | 两半各加 own bias（§3.3），探针 P2/P6 已编译 |
| `out_n` 极小（1） | `grid = m`；每 chunk 只有 1 个有效 lane（正确性由 L0-6 系列覆盖，性能不关心） |
| 同一 wrapper 内多次调用 | 每次调用是独立 launch；本 kernel 无跨 launch 状态（无 workspace、无原子） |
| 上游 `split_k` 变化 | 本 kernel 按 `split_k` 特化（DMA 条数与 `vadd` 链长随之变化），无需额外约束 |

---

## 10. 交付清单

### 10.1 目录结构

```
examples/shared_expert_mlp/splitk_reduce_kernel/
├── DESIGN.md                          # 本文
├── verify_equiv.py                    # D-1：A1/A2 的等价性机器验证（torch CPU）
├── RETROSPECTIVE.md                   # Stage 1 复盘（Phase 8 追加）
└── history_version/
    └── design_probe_compile.py        # D-3：无卡编译探针 P1–P6
```

### 10.2 文件清单

| 文件 | 作用 | 状态 |
|---|---|---|
| `DESIGN.md` | 设计文档（§0 迁移分析 + §1–§11） | 本次产出 |
| `verify_equiv.py` | `python verify_equiv.py` → 110 用例、0 失败、`[EQUIV_ALL_PASS]` | 已执行（§11.2） |
| `history_version/design_probe_compile.py` | `ASCEND_RT_VISIBLE_DEVICES="" python history_version/design_probe_compile.py all` → P1–P6 全部 COMPILE OK | 已执行（§11.1） |

### 10.3 命名规范

- 工厂签名与源一致：`splitk_reduce_kernel(split_k, m, n, dtype="float16", activation="none",
  has_bias=False, out_dtype=None)`（TileOPs 侧 `_shared_expert_mlp_fwd_kernels.py` 的 import 名不变）。
- 内层可调参数：`BC`（块宽）。**内层工厂仍保留 `elems_per_cta=1024` 这个位置参数**（wrapper 的
  `splitk_reduce_kernel(...)(_SPLIT_K_ELEMS_PER_CTA)` 调用形态不变），但它在 NPU 侧是 **no-op**：
  NPU 的块粒度由 §5.2 的规则从 `(m, N)` 决定，而不是「每块多少元素」。Stage 4 可把它改造成
  `BC` 覆盖旋钮（`elems_per_cta ∈ {256, 512, 1024}` 时直接指定 `BC`），默认路径仍走规则。

### 10.4 交付给下游的额外说明（Stage 3 / Stage 5）

1. **`C` 必须由调用方预分配并传入**（`out_idx=[]`），与 wrapper 现有形态
   （`out = torch.empty((m, out_n), dtype=..., device=...); reduce_program(partial, out)`）一致。
2. `BC` 用 §5.2 的规则（不要硬编码单一值）；备选统一 `BC=512` 留给 Stage 4 A/B。
3. `n` 是**完整宽度**（门控时 `2S`），不是 `out_n`；wrapper 已按此传参。
4. `has_bias` / `out_dtype` 在 op 中取默认值，但 L0-6/L0-8 必须覆盖它们。
5. Stage 5 修复 wrapper 的 `_split_k`（`num_tokens < 16 → 1`）后，kimi-k2-t1 不再进入本 kernel，
   但 L0-5/L0-6 的退化路径必须保持通过。

---

## 11. 设计期自检与探针记录

### 11.1 编译探针（D-3，无卡）

脚本 `history_version/design_probe_compile.py`；执行方式
`ASCEND_RT_VISIBLE_DEVICES="" python history_version/design_probe_compile.py all`
（平台 `execution` 是 device 模式，但本探针只做**编译**、不创建 NPU 张量、不运行 kernel，
故不占用设备）。**6 个探针 / 7 个用例全部 COMPILE OK**：

| 用例 | 覆盖的设计决策 | 关键代码生成事实 |
|---|---|---|
| P1 非门控 + bias，`(4,32,7168)`，BC=1024，fp16 | 主选路径全链路 | `asc_copy_gm2ub_align(..., 1, 4096, …, 4096, 4096)` ×4（每片单描述符 4 KiB）；`vlds_norm<float>` + `for (s<3) vadd`；`vlds_unpack<half>` + `vcvt<float>` + `vadd`（bias）；`vcvt<half>(y, ROUND_R, RS_DISABLE)` + `vsts_pack_b32`；`asc_copy_ub2gm_align(..., 1, 2048, …)`；同步：`MTE2→V`、`V→MTE3` 各一对 |
| P2 门控 + bias，`(4,32,4096)`，BC=512，bf16 | 双列流 + silu + bf16 | 两条流的源地址差恰 `2S`（`W[... + 2048]` vs `W[...]`）；`vneg/vexp/vadd/vdiv_0ulp_ftz_true/vmul`；`vcvt<bfloat16_t>` + `vsts_pack_b32`；`vector_f32 g[1]; vector_f32 u[1]`（`alloc_local`） |
| P3 门控 + bias 尾块，`N=1536`，BC=1024 | 运行期 `bc` + 尾 lane 谓词 | 生成 `update_mask`（计数 1）；其余同 P2 |
| P4a/P4b `split_k=1`、无 bias、无激活，fp16 / bf16 | **退化路径**（R6） | 只剩 `vld` + `vcvt` + `vsts_pack_b32` + 1 条 `gm2ub` / 1 条 `ub2gm`；**0 条 `vadd`** |
| P5 `out_dtype="float32"`，fp16 输入 + bias | fp32 输出路径 | **首轮 FAIL**：`ValueError: Unsupported vcvt conversion float32->float32` → 改为跳过 vcvt 后 **OK**；无 `vsts_pack_b32`（走 `NORM_B32`） |
| P6 glm45-t32 gate/up，`(4,32,3072)`，BC=512，bf16 | 除数路径（`N % BC == 0`） | 同 P2，但无 `update_mask` |

**探针的负向结论**（对设计的直接约束）：① `T.SimdVF` 内禁止 GM 访问；② 跨迭代向量累加必须用
`alloc_local`；③ `vcvt fp32→fp32` 不存在（设计新增 R5 的 fp32 分支）；④ 每个用例的
`float2` 标量循环计数为 **0**（无逐元素标量循环，§1.6.2 的结论得到生成代码支持）。

### 11.2 等价性机器验证（D-1）

`python verify_equiv.py` → **110 用例、0 失败、`[EQUIV_ALL_PASS]`**（结果表见 §1.6.1）。要点：
A1（silu 除法改写）在 fp16/bf16 上输出 dtype **0 ulp 差、违反率 0、非有限模式完全一致**，
fp32 足迹 ≤ 3 ulp（改动元素 ≤ 23%），对 fp64 参照的误差与基线**逐位相同**；
A2（退化路径专用化）在两个 dtype 上 **bitwise equal**；O1（切片升序和 vs `partial.sum(0)`）在
fp16/bf16/fp32 网格上 **0 ulp**。

### 11.3 门禁自检

`python3 .agents/tools/statectl.py gate 1 --dir examples/shared_expert_mlp/splitk_reduce_kernel`
（只读）输出：

```json
{"action": "gate", "stage": 1, "ok": true, "pass": true, "failures": [], "warnings": []}
```

### 11.4 回退说明

**首次设计（v1），无上一版**，故无「相对上一版的关键调整」。后续 revision 时本节改为
「相对 v1 的关键调整 + 为何不会再犯同一错误」。

---

## 附：本文引用的平台条目与证据索引

| 证据 | 位置 | 用途 |
|---|---|---|
| 平台容量 / 核数 / 预算 | `statectl platform --task-dir`（2026-11-23）、`.agents/platforms/hardware/ascend-950pr-9579-notes.md` | §4.5、§5.5、§9.1 |
| 物理核数实查 | `Ascend950PR_9579.ini` 第 20–22 行（`grep` 记录于 §5.5） | §5.5 ② |
| SIMD 执行区域与向量 API | `.agents/platforms/toolchains/ascend/references/programming.md`；`.agents/platforms/toolchains/ascend/knowledge/PL-950-vector-execution.md`；`.agents/platforms/toolchains/ascend/knowledge/PL-950-row-primitives.md` | §2、§3.2、§3.4 |
| VST → DMA 无需 barrier | `examples/engram_gate_conv/_engram_gate_conv_fwd_kernel/repro/TRAP-950-simd-ub-store-load.py`（断言 A） | §7.2、§9.2 |
| 二维 SIMD 归约的能力缺口 | `.agents/platforms/toolchains/ascend/knowledge/capability-gaps.md`（`CG-950-simd-shared-reduce`） | §1.6.0、§9.2 E7 |
| 原子 split-K 的适用边界 | `.agents/platforms/toolchains/ascend/knowledge/execution-patterns.md`（`PL-950-atomic-splitk`） | §1.6.0 C2/R4 |
| 硬件常数（只作量级外推） | `.agents/skills/tilelang-op-optimize/references/pattern-library/constants.md`（`CONST-mte2-degradation`、`CONST-vector-launch-overhead`，均绑定 910B2C） | §1.6.0 D-2、§5.3 |
| 上游部分和契约 | `examples/shared_expert_mlp/_gemm_basic_kernel/DESIGN.md` §4.2 / §5.2 / §9.2 E6 | §0.1、§0.2、§9.1 R6/R7 |
| 迁移方法论 / 调研方法论 / 分核标准 | `.agents/skills/tilelang-op-design/references/migration-analysis.md`、`.agents/skills/tilelang-op-design/references/algorithm-research.md`、`.agents/skills/_shared/standards/core-split-strategy.md` | §0.5、§1.6.0、§5.5 |
| 源算子（GPU） | `/workspace/TileOPs/src/tileops/_kernels/gemm/dense.py:1987-2050`；抽取件 `examples/TileOPs/tileops/kernels/moe/shared_expert_mlp/_shared_expert_mlp_fwd_kernels.py`（第 308–374 行） | §0 全章 |
| 调用方 | `examples/TileOPs/tileops/kernels/moe/shared_expert_mlp/shared_expert_mlp.py`（`_split_k` / `_gemm_program`） | §0.2、§9.1 R7/R8 |
| 本任务探针 / 等价验证 | `examples/shared_expert_mlp/splitk_reduce_kernel/history_version/design_probe_compile.py`、`examples/shared_expert_mlp/splitk_reduce_kernel/verify_equiv.py` | §11.1、§11.2 |
