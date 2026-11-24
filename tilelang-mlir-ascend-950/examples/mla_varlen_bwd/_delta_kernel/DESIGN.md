# _delta_kernel 算子设计文档（packed 分支 / `MLAVarlenBwdOp` 第 1 个 launch）

> **迁移任务（harness 子模式），首次设计 v0**。源算子：`/workspace/TileOPs/src/tileops/_kernels/attention/mla/bwd.py`
> 第 38–77 行的 `_delta_kernel`（GPU TileLang 实现；`MLADenseBwdKernel` 与 `MLAVarlenBwdKernel` **共用同一工厂**）。
> 目标平台：`ascend-950pr-9579`（dav-3510，CANN 9.2.1，TileLang ascend 0.1.15），编程模式按平台
> `programming_modes.migration_default = simd` 选定 **SIMD**（§2），`import tilelang.ascend.language as T`。
>
> **本函数的 GPU 工厂与 `MLADenseBwdOp` 的 `_delta_kernel` 逐字相同**：`_delta_kernel(batch, heads, dim_v,
> dense_seqlen, dtype)`，源码里只有三处 `dense_seqlen` 相关分支——输出 shape
> `(batch, heads, dense_seqlen) if dense_seqlen else (heads, total)`（第 49 行）与写回下标
> `delta[row // S, h, row % S]` / `delta[h, row]`（第 69–72 行）；计算语句（两条 `T.copy` → `T.Parallel` 乘 →
> `T.reduce_sum` → 谓词写回）**逐字一致**。dense 版设计（v2）已通过检视、实现已在真机 23/23 通过，因此本
> 设计按任务要求**明确分区**：验证过的核心（一维核 + `R` 行 UB tile 双缓冲 + 逐行 SIMD `vcadd` + 压缩写出 +
> 构建期 `nvec` 展开 + 目的端切片 + 两侧 `T.clear` + A1/A2/A3）**引用 dense 证据**（§0.8、§11），**只重新推导
> packed 差异**（动态 `total` 导致的动态 tile 流 / 动态 extent / 无静态尾段；head-major 输出的取址），并各自给
> 编译探针与等价性机器验证证据（§3.5.4、§1.6.1）。
>
> 三个 launch 的契约：**`_delta_kernel`（本函数：产出 fp32 `delta`，head-major `[H, T_q]`）** →
> `_bwd_kernel`（写 `dk/dv`，向 fp32 `dq_acc` 原地累加）→ `_dq_land_kernel`（`dq = dq_acc * scale` 并 cast）。
> 本函数的输出契约由 `_bwd_kernel` 的消费端固定：`T.Tensor((heads, total), "float32")`，按
> `delta[h, qoff : qoff + 64]`（一次 256 B 连续段）取数（`examples/mla_varlen_bwd/_bwd_kernel/DESIGN.md` §1.6.3 已按
> 该 layout 设计，真机段序装载证据为 `asc_copy_gm2ub_align(..., 1, 256, ...)`）。
>
> **设计期证据（本机编译，无卡：`ASCEND_RT_VISIBLE_DEVICES=""`，从不 launch）**：
> - 平台配置实查：`python3 .agents/tools/statectl.py platform --task-dir examples/mla_varlen_bwd/_delta_kernel`
>   → `hardware_id = ascend-950pr-9579`，`observed_cores = {cube: 28, vector: 56}`，UB 253952 B / L1 524288 B /
>   L0C 262144 B，`programming_modes.migration_default = simd`，`cost_measurements` 为空（§5.5 ② 有完整返回字段）。
> - 编译探针（`history_version/design_probe_packed_v2.py`，D-3，无卡）：**P1/P2/P3/P4/P5/P6/P7 全部 COMPILE OK**；
>   负向对照 **N1 COMPILE FAIL**（目的端不切片 → `PlanMTECopy ... got src=T.min(64, total_q - it // 4 * 64) * 64
>   and dst=8192`）、**N2 编译通过但 SIMD 计算体被静默丢弃**（5 个 SIMD 记号在设备码里全部缺失）、
>   **N3 静态 `R` extent 写回编译通过且被编译器自行 clamp 成 `min(64, max(total_q - t0, 0))`**（依赖未文档化
>   行为，故本设计仍显式 clamp——§3.4 注 5）。日志：`history_version/probe_v2_tails.log`（完整描述符行）、
>   `history_version/probe_v2_p5_p6_p7.log`、`history_version/probe_v2_neg.log`。
> - 等价性机器验证（`verify_equiv.py`，torch CPU + fp64 参照）：**A1/A2/A2b/A3/A4 全 `EQUIV_PASS`**
>   （A4 为本函数新增的 head-major 写回项），结果表见 §1.6.1。

## 修订说明

本版为该函数的**首次设计**（`first_design`，非 revision），故无「相对上一版的关键调整」条目；上一轮运行因
环境瞬时 `database or disk is full` 中断，未产出 `DESIGN.md`（`history_version/` 中留有当时的探针脚本
`design_probe_packed.py` 与日志，本版在此基础上补全并新增 P6/P7 与 N1–N3 三组对照，见 §3.5.4）。

---

## 0. 源算子解读与迁移分析

### 0.1 源算子语义（做什么）

`_delta_kernel(batch, heads, dim_v, dense_seqlen, dtype)` 是 MLA prefill 反向的**第一段**（docstring：
“`delta[.., h, row] = sum_d o[row, h, d] * do[row, h, d]` in float32”），为 `_bwd_kernel` 的
`dP - delta` 项提供每个 query 行的标量。

**数学语义（packed）**：输入是 packed THD 的 `o` / `do`（`(T_q, H, D_v)`，`T_q = Σ_b q_len[b]` 为**动态**行数），
每个全局 token（行）`t` 与每个头 `h` 独立地做 value 维点积：

```
delta[h, t] = Σ_{d < D_v} o[t, h, d] · do[t, h, d]        t ∈ [0, T_q), h ∈ [0, H)
```

**与 dense 的唯一语义差别是输出的下标空间**：dense 是 `delta[b, h, s]`（batch-major，`row = b·S_q + s` 由
host 侧 view 保证三维可解），packed 是 `delta[h, t]`（head-major，`t` 直接就是 packed token 行号，**不需要**
`cu_seqlens`——源码 packed 分支的写回只有 `delta[h, row]`，第 72 行）。`lse` 不进入本式。

**语义要素（逐项由源码核对）**：

- **规约语义**：规约轴是 `D_v`（最内层连续轴），输出少一维；累加顺序为源码
  `T.copy`（16-bit → float32 fragment）→ `T.Parallel(block, dim_v)` 逐元素乘 → `T.reduce_sum(o_f, rows, 1)`
  的 fragment 归约（GPU 侧展开为 warp 内树形 + 跨 warp 合并，即分组归约而非严格串行）。→ 迁移侧只要求
  **同 dtype 语义下的容差内等价**（分组可换，§1.6.1 A2 机器验证）。
- **dtype 语义**：输入 `float16 | bfloat16`；`T.alloc_fragment(..., "float32")` + `T.copy` 使 16-bit 输入
  **精确提升**为 float32（无损），乘积在 fp32 中四舍五入，累加全程 fp32，输出 fp32，中间不回落 16-bit。
  **这是本算子最重要的一条语义**（“fp32 乘 + fp32 累加”，不是“fp16 乘 + fp32 累加”），与 packed/dense 无关。
- **I/O 契约**：`o`/`do` 为 `(total, H, D_v)`，`total = T.dynamic("total_q")`；输出 `delta` 为 `out_idx=[2]`
  自动分配：`dense_seqlen` 非零 → `(batch, heads, dense_seqlen)`，**`dense_seqlen == 0`（packed）→
  `(heads, total)`**（第 49 行）。本函数固定走后者。
- **边界语义**：① 尾块行 `row ≥ total` 由 `if row < total` 谓词挡住，不写（packed 下 `total` 运行时才知道，
  这是**唯一**的尾处理手段——dense 侧可换成构建期静态尾段，packed 不行，见 §0.4 O6'）；② `nan`/`inf` 无特殊
  处理，自然通过“乘 + 加”传播（`inf·0 → nan`）；③ `D_v` 空维不出现（`MLABwdKernelBase.refusal` 要求
  `dim_qk % 16 == 0 and dim_v % 16 == 0`）；④ 行数与 `block=32` 非整除是正常路径；⑤ `batch` 在 packed 分支
  **不参与取址**（grid 为 `(heads, ceil(total/32))`，第 57 行），只作为工厂缓存键的一部分。
- **host 侧语义**：packed 调用**不做 view**（`MLAVarlenBwdKernel.forward` 直接把 `(T_q, H, D)` 交给 `_run`；
  dense 才有 `(B,S,H,D) → (B·S,H,D)` 的 view）；`delta` 由 `out_idx` 分配；
  `@target_cache(maxsize=32)` 按 `(batch, heads, dim_v, dense_seqlen, dtype)` 特化编译，`dense_seqlen=0` 是
  packed 的工厂键。这些属实现手段，只有 “`delta` 的 shape/dtype/layout” 属契约。

**语义保持基线**：§8.1 的 golden 只依据本节实现（`(o.float() * do.float()).sum(-1)` 的 head-major 转置），
**不参考 §0.6 的 NPU 算法**。

### 0.2 源算子输入输出

| 参数 | 方向 | Shape | dtype | 说明 |
|---|---|---|---|---|
| `o` | in | `(total, heads, dim_v)` | `float16 \| bfloat16` | packed THD；`total = T_q` **动态**，行主序 `(t, h, d)` |
| `do` | in | `(total, heads, dim_v)` | 同 `o` | 输出梯度，dtype 与 `o` 相同（`ATTENTION_DTYPES`） |
| `delta` | out | `(heads, total)` | `float32` | **head-major**，`dense_seqlen == 0` 分支（源码第 49 行） |

源算子输出 shape（任务给定）：`delta [H, T_q]` float32。与 `_bwd_kernel` 的入参声明
`T.Tensor(stat_shape, accum)`（`stat_shape = (heads, total_q)`，`bwd.py` 第 137 行赋值、第 148 行声明）以及消费处
`delta_s[i] = T.if_then_else(m0+i < len_q, delta[h, q_start + m0 + i], 0)`（第 222–224 行，packed 分支）三者一致
——本设计的输出契约不得偏离；`[T_q, H]` 的 token-major 变体会让消费端每次取 64 行 × 4 B（stride `H·4`），
本设计不引入该代价。

### 0.3 实现算法解读（怎么算）

**源码结构化阅读**（`_delta_kernel` 全文 40 行 + host 侧 `_compile`/`_run`）：

- **入口签名**：`_delta_kernel(batch, heads, dim_v, dense_seqlen, dtype)` → 内部 `_delta_func()` 无参
  （几何全部是闭包常量）；`total = T.dynamic("total_q")` 是唯一的运行期维度。
- **Kernel 结构**：`T.Kernel(heads, T.ceildiv(total, block), threads=128)` → **二维网格**（h, 行块），
  `block = 32` 行/tile，128 线程；packed 下 grid 的两个维度天然可解（不需要 `cu_seqlens`）。
- **计算原语**：`T.copy`（GM→fragment，含 16→32 提升）、`T.Parallel(32, dim_v)` 逐元素乘、
  `T.reduce_sum(o_f, rows, 1)`（dim=1 归约）、`T.Parallel(32)` 谓词写回。
- **内存分配**：`o_f`/`do_f` = `alloc_fragment([32, dim_v], "float32")`（纯寄存器/fragment，无 SMEM），
  `rows = alloc_fragment([32], "float32")`；无中间结果落 GM。
- **循环与调度**：无 K 维循环、无 `T.Pipelined`；一个 block 一个 tile，块内 128 线程摊平 `32×dim_v`。
- **host 侧逻辑**：`MLAVarlenBwdKernel.forward` **不 view**（packed 张量直接进 `_run`）；`delta` 作为
  `out_idx` 分配；`target_cache` 按 `(batch, heads, dim_v, 0, dtype)` 特化。
- **编译配置**：`pass_configs={TL_ENABLE_FAST_MATH: True}`、`compile_flags=["-O3", "-DENABLE_BF16"]`。

**计算步骤分解表**（源码每个计算语句均归入某步骤，无遗漏、无臆造；`dense_seqlen == 0` 分支）：

| # | 步骤 | 计算 | 输入 | 输出 | 对应语义公式 |
|---|---|---|---|---|---|
| 1 | 任务映射 | `(h, bx) ← T.Kernel(heads, ceildiv(total, 32))` | 网格索引 | — | 选择 `(h, 行块)` |
| 2 | 提升搬入 | `T.copy(o[bx·32 : bx·32+32, h, :], o_f)`（16→32；越界行不填） | GM `o` | `o_f` fp32 [32, D_v] | 取 `o[t,h,:]` 并精确提升 |
| 3 | 提升搬入 | 同步骤 2，`do` | GM `do` | `do_f` fp32 [32, D_v] | 取 `do[t,h,:]` |
| 4 | 逐元素乘 | `o_f[i,d] = o_f[i,d] * do_f[i,d]`（`T.Parallel(32, dim_v)`） | `o_f`, `do_f` | `o_f`（原地） | 每个 `d` 的乘积项 |
| 5 | 行归约 | `T.reduce_sum(o_f, rows, 1)` | `o_f` | `rows` fp32 [32] | `Σ_d` 部分 |
| 6 | 谓词写回 | `if row < total: delta[h, row] = rows[i]`（**packed 分支**，第 68/72 行） | `rows` | GM `delta` | 结果落 `(h, t)` |

**数据流图（源硬件视角）**：

```
GM o[total,H,D] --T.copy（16→32 提升，32 条 256B 跨步段）--> fragment o_f[32,D] fp32
GM do[total,H,D] --T.copy（同上）-------------------------> fragment do_f[32,D] fp32
o_f ⊙ do_f（T.Parallel，寄存器原地）--> fragment o_f
o_f --T.reduce_sum(dim=1)--> fragment rows[32] --谓词写回 delta[h, row]--> GM delta[H,total]
```

无 SMEM、无中间 GM 往返；一个元素从 GM 进来只被读一次、只参与一次乘法和一次累加。

**循环与并行结构**：grid = `heads × ceil(total/32)`（二维，共 `H·ceil(total/32)` 个 block，**动态**）；
block = 128 线程；每 block 一次 tile（32 行 × D_v），无循环携带依赖。

**host 侧逻辑**：packed 无 view；`cu_seqlens` 不参与本 launch；`delta` 由 `out_idx` 分配；`target_cache` 特化。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 源码特征 | 目的 | 依赖的源硬件特性 | 初判 |
|---|---|---|---|---|---|
| O1 | **单遍融合乘+归约** | 乘积写回 `o_f` 后立即 `reduce_sum`，不落 GM | 消掉 O(N) 中间张量与一次 GM 往返 | 无（算法层） | 可移植 |
| O2 | **float32 提升后计算** | fragment 声明 `"float32"`，`T.copy` 提升 | 数值稳定、与 golden 的 fp32 域一致 | 无（dtype 语义层） | 保留 |
| O3 | **fp32 product/累加全驻留寄存器** | `alloc_fragment`、无 `alloc_shared` | 每元素只付一次 GM 读 | 大寄存器堆（fragment 直接映射寄存器） | 硬件强相关 |
| O4 | **`block=32` 行级 tile** | `block = 32`，一次搬 32 行 | 摊薄 128 线程的启动/调度成本，凑够 DMA 段 | coalesce 的 256B 段（D_v·2 B） | 硬件强相关（可等价替换） |
| O5 | **head 固定 → 段内连续** | `o[rows, h, :]` 固定 `h` | 每行 256B 完全连续，段内无空洞 | GM 段式访问 | 硬件强相关（可保留） |
| O6' | **运行期谓词写回尾块**（packed 形态） | `if row < total`（`total` 动态） | 允许 `total % 32 ≠ 0` | 线程级发散容忍 | 可移植（**但 packed 下必须保留“运行期”语义**，见 §0.6 R4） |
| O7 | **warp 级树形归约** | `T.reduce_sum(...,1)`（warp shuffle + 跨 warp） | 行内 128 元素 O(log) 合并 | warp（32 lane）与 `__shfl` | 硬件强相关 |
| O8 | **二维网格 + `threads=128`** | `T.Kernel(heads, ceildiv(total,32), threads=128)` | 任务映射到 block/thread 两级 | CUDA block/thread 模型 | 硬件强相关 |
| O9 | **shape 特化编译** | `@target_cache` + 闭包常量 | 消运行期分支与元数据 | host | 可移植 |
| O10 | **head-major 统计量布局** | packed 下 `delta[h, row]` | 让消费端（`_bwd_kernel`）按 `[H,T_q]` 一次 256B 连续段取数 | 无（layout 层） | 保留（契约） |

**算法层 vs 实现层**：O1、O2、O6'、O9、O10 属算法/语义/布局层（单遍、dtype、边界、特化、输出布局），迁移优先
保留其意图；O3、O4、O5、O7、O8 属实现层（寄存器驻留、tile 形状、并行结构、warp 归约），逐项判定见 §0.5。

> **与 dense 的对照**：O1–O5、O7–O9 与 dense 完全相同；**O6 在 dense 侧被“等价替换”为构建期静态尾段
> （`S % R` 分支 + 静态 `nlast`），本函数不能复用那条结论**——`total` 是运行期量，`nlast` 不存在（§0.6 R4）。
> O10 是 packed 独有的布局项。

### 0.5 硬件耦合性分析与 NPU 适配决策

判定输入含 §1.6.0 调研结论（调研先于判定）。目标平台事实（`tilelang.ascend.language` + 平台 ruleset
`.agents/platforms/toolchains/ascend/references/programming.md`）：

- `T.Kernel(N)` 是**一维核网格，签名没有 `threads=`**（`ascend/language/kernel.py`；多维网格被拒绝）；
  线程域必须由 `T.SimtVF(threads=...)` 或 `T.SimdVF()` 在核内显式声明。
- 片上层级：UB 253952 B（`T.alloc_shared`，per-core）、L1 512 KB、L0C 256 KB、L0A/B 各 64 KB
  （`Ascend950PR_9579.ini` 第 50/51/58 行）；本算子不使用 Cube。
- 无 warp/`__shfl` 语义；`T.tvm_warp_shuffle_*` 属 CUDA 公共面，迁移不可用。
- 高层 `T.reduce_sum` 的二维 shared→shared 路径在 `T.SimdVF` 内有已确认的输出布局差异
  （`.agents/platforms/toolchains/ascend/knowledge/numerical-boundaries.md` 的 `TRAP-950-simd-shared-reduce`）；
  SIMT 路径 `alloc_reducer → reducer_init → reducer_update → finalize_reducer` 已验证，但每行一次跨线程
  AllReduce（`T.finalize_reducer` 走 `tvm_thread_allreduce` + barrier）。
- 显式 SIMD 寄存器归约已验证：`T.simd.vld / vcadd / vsts`
  （`.agents/platforms/toolchains/ascend/knowledge/repro/PL-950-simd-reduce.py`，FP32 (2,64) 两行，含尾 lane mask）。

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|---|---|---|---|---|---|
| 数学语义（行点积、fp32 乘、fp32 累加） | 语义 | 无 | — | **保留** | 语义层无条件保留 |
| I/O 契约（`(T,H,D)` 入、`[H,T_q]` fp32 出） | 语义/布局 | 无 | — | **保留** | 由 `_bwd_kernel` 消费端固定（O10）；packed 的 head-major 取址在本侧重推（§0.6 R5） |
| O1 单遍融合乘+归约 | 算法 | 无 | 有 | **保留** | 目标侧仍是“乘完立刻归约、不落 GM”（§1.6.1 A1） |
| O2 fp32 提升计算 | 语义/dtype | 无 | 有 | **保留** | `vcvt` 到 fp32 后再 `vmul`（禁止 fp16 乘，§1.6.0 R1 弃选 C4） |
| O9 shape 特化 | 优化 | 无 | 有 | **保留** | 工厂按 `(batch, seqlen, heads, dim_v, dtype)` 特化；**packed 下 `seqlen = 0`，`total` 仍动态**——本函数把“动态”留在 `total` 上（与源码一致），其余几何静态 |
| O10 head-major 布局 | 布局 | 无 | 有 | **保留** | `delta` 声明为 `(heads, total)`；写回目的端仍是连续 4 B 元素（§0.6 R5 给出描述符证据） |
| O6' 运行期谓词尾处理 | 优化/边界 | 线程发散 | 有（动态 extent） | **等价替换** | 不是“构建期静态尾段”（packed 构造性不可用），改为 **运行期 clamp 的动态 extent**：`n = min(R, total - t0)` 同时约束两条 `T.copy`、行循环上界与写回（§0.6 R4、探针 P1/P2） |
| O5 head 固定读 | 优化 | GM 段式访问 | 有 | **保留** | 目标侧同为固定 `h` 的 256 B 连续段（探针 P1：源间距 `H·D_v·2 B`，H=8 时 2048 B、H=128 时 32768 B） |
| O4 `block=32` tile | 优化 | 256B 段 + 行数凑够 | 有 | **等价替换** | tile 改为 `R = 64` 行（构建期常量），UB 双缓冲（§5.2/§6） |
| O3 fragment = 寄存器 | 优化 | 大寄存器堆 + fragment 语义 | 部分 | **等价替换** | SIMD 向量寄存器承载 fp32 中间量，**连 fp32 product tile 都不物化**（§1.6.1 A1） |
| O8 二维网格 + `threads=128` | 优化 | CUDA block/thread | **无** | **重新设计** | → §0.6 **R1**：一维 `T.Kernel(G)` 静态核网格 + 核内**动态 tile 流** |
| O7 warp 树形归约（`T.reduce_sum(dim=1)`） | 优化 | warp shuffle / fragment 归约 | **无直接等价** | **重新设计** | → §0.6 **R2**（承接 dense 已验证形态）：逐行显式 SIMD `vcadd` |

> 依据来源：`T.Kernel`/`SimtVF`/`SimdVF` 签名见 `ascend/language/kernel.py`、`frame.py`（平台 ruleset
> `programming.md` 第 13–15 行转录）；高层归约差异见 `TRAP-950-simd-shared-reduce`；SIMD 归约可行性见
> `PL-950-simd-reduce`；容量见 `Ascend950PR_9579.ini`。舍弃项：无（O1–O10 全部有承接）。

### 0.6 NPU 算法重设计

> **重设计项的划分原则**：R2 完全承接 dense 已验证形态（本函数不重推）；R1/R3/R4/R5 是 **packed 差异**，
> 逐项重新推导并给编译/数值证据。所有重设计项的语义保持论证共用同一条骨架：① 数学等价（每行仍是
> “`D_v` 个 fp32 乘积之和”，行与行**没有累加关系**，并行/流式重排不改变任何行的内容，只改变归约分组——
> 由 A2 机器验证覆盖）；② dtype 语义（16→32 无损提升，乘与加都在 fp32）；③ 边界（不越界读、不越界写，
> 且不污染相邻 head）。

**重设计项 R1：并行结构（二维网格 + 128 线程 → 一维静态核网格 + 核内动态 tile 流）**

- **源方案**：`T.Kernel(heads, ceildiv(total, 32), threads=128)`——一个 block 处理 `(h, 32 行)`，线程域 128
  隐式承担 `32×D_v` 的元素并行与归约；grid 的第二维**随 `total` 变化**（源实现允许动态 grid）。
- **NPU 新算法**：`T.Kernel(G)`，`G = 56`（=`vector_core_cnt`，实查见 §5.5 ②）为**构建期常量**；
  `ntiles = ceildiv(total, R) · heads`（运行期），第 `bx` 个核处理
  `it ∈ [ntiles·bx / G, ntiles·(bx+1) / G)`（均衡划分，两界都是运行期表达式，且恒有 `it0 ≤ it1`）；
  每个 `it` 解出 `h = it % heads`（`heads` 构建期常量 → 常量模）、`t0 = (it // heads) · R`；
  块内 `T.Pipelined(it0, it1, num_stages=2)` + `T.annotate_buffer_versions({o_ub: 2, do_ub: 2})`。
  核内无线程域：元素并行由向量宽度（128 lane/寄存器）承担，归约走 R2。
- **语义保持论证**：① 数学等价——每个 `(h, t0)` tile 的计算与“哪一行由哪个核的哪次迭代算”无关（行间无依赖）；
  ② 边界——`t0` 只取 `ceildiv(total,R)` 个块之一，恒有 `t0 < total`，故 `n = min(R, total−t0) ≥ 1`；
  `ntiles` 由运行期 `total` 算出，**不会漏算也不会多算**（探针 P1/P7 的循环界即
  `((total_q+63)>>6)·bx/7` 形态）；③ 无跨核归并、无原子、无共享状态。
- **源优化意图承接**：O8 的意图是“把 `H·ceil(S/32)` 个 tile 铺给硬件执行单元”——本结构用“静态 56 核 +
  核内串行 1170+ tile（代表 workload）”承接，摊薄了核启动开销（分核标准 §1 要素③的持久化形态）。

**重设计项 R2：归约（fragment `T.reduce_sum(dim=1)` → 逐行显式 SIMD `vcadd`）——承接 dense，不重推**

- **源方案**：`T.reduce_sum(o_f, rows, 1)`——GPU 上对 `[32, D_v]` fp32 fragment 沿 `d` 做块内归约
  （warp shuffle 树 + 跨 warp 合并）。
- **NPU 新算法**：每行在 `T.SimdVF` 内
  `vld(16-bit 行) → 2×vcvt(PART_EVEN/PART_ODD) → 2×vmul → vadd(两半) → vcadd(64 lane 二叉树) → vsts(PAT_VL1)`；
  `D_v > 128` 时按 `nvec = DW/128` 个寄存器分段，各段部分和**在构建期展开成一个表达式**
  （`vadd(vadd(p0,p1),p2)…`，段数不进运行期循环）；`D_v` 不足 128 的倍数时两侧清零后零填充。
- **为何本函数可直接承接**：该形态只依赖“一整行 `D_v` 个 16-bit 元素驻留在 UB 的连续 128 lane 上”，
  与“行号如何解出、输出写到哪里”无关——packed 的 `(h, t0)` 解址不改变行内布局（探针 P1/P2/P3 的
  `vlds_norm<half>(... + i*128)` 与 dense 逐字同形）。
- **语义保持论证**：同 dense（同一批 fp32 乘积的重新分组，A2 以 fp64 参照机器验证：12/12 用例违反率 0，
  树形比顺序更接近真值）。
- **源优化意图承接**：O7 的“行内低延迟合并、结果驻留寄存器”由 `vcadd` 单指令承接。

**重设计项 R3：数据搬运（fragment 搬入 → UB 描述符；packed 动态 extent）**

- **源方案**：`T.copy` 直接落 fragment（寄存器），无 SMEM；32 行 × 256 B 的跨步读由 GPU 的 L1/LSU 处理。
- **NPU 新算法**：每 tile 两条跨步 DMA：
  `T.copy(o[t0 : t0+n, h, :], o_ub[0:n, 0:D_v])`（`do` 同），目的端**必须**写成显式 region（`DW > D_v` 时
  元素数才相等）；`n = min(R, total − t0)` 是运行期标量，**源与目的两端都用它**。
  探针实测（P1，D_v=128，H=8）：`asc_copy_gm2ub_align(dst, src, min(64, total_q − t0), 256, 0, 0, 0, mode,
  2048, 256)`——单条描述符、64 行 × 256 B、源间距 `H·D_v·2 B`、目的行距 256 B；
  H=128 时源间距参数为 32768 B（P5）。行数 `n` 是描述符的**运行期 count 参数**，不是构建期常量。
- **语义保持论证**：纯数据搬运，不改变任何数值；`n` 取 `min(R, total−t0)` 保证只搬有效行（不越界读）。
- **源优化意图承接**：O3/O5 的“每元素只付一次 GM 读 + 段内连续”由单条跨步描述符承接。

**重设计项 R4：尾块与动态边界（dense 的静态 `nlast` 尾段 → 运行期 clamp；含静态化备选）**

- **源方案**：`if row < total` 逐元素谓词（warp 内发散，但只影响最后一个 tile）。
- **dense 的处置为何不能复用**：dense 把尾块改成“构建期 `S % R` 分支 + 静态 `nlast` 的三处 extent”，
  前提是 `S` 是构建期常量。packed 的 `total = T_q = Σ q_len` 是**运行期**量，`nfull = total // R` 与
  `nlast = total − nfull·R` 都不是常量 → **不存在**静态尾段。
- **NPU 新算法（主选）**：不做尾段特化，**每个 tile 都用运行期 clamp** `n = min(R, total − t0)` 约束
  （i）两条装载的源/目的 extent、（ii）写回的 extent；行循环上界保持静态 `R`（多算的行落在 `[n, R)`，
  结果丢弃，不写出）。三条描述符的 count 因此都是运行期表达式（P1/P2/P3/P5 实测生成）。
- **静态化备选（P6，已编译）**：把 tile 流按“满 tile / 残块”拆成两段——满 tile 段
  `nfull = (total // R)·heads` 个 tile，**两条装载与写回的 extent 全为静态 `R`**（记 `it` 的循环界仍是
  运行期）；残块段每 head 一个 tile，`n = total − (total//R)·R` 仍用运行期 clamp，由 `if bx < heads` 分给
  前 `heads` 个核。P6 生成码：满段写回 `asc_copy_ub2gm_align(..., 64, 4, ...)`（静态 64 个 fp32），
  残块段 `..., (total_q & 63), 4, ...`（运行期 count）。
  **代价**：残块段在流水循环**之外**写同一组 buffer，编译器的 AutoSchedule 因此关闭 `o_ub/do_ub` 的多版本
  （P6 实测告警 `did not enable multi-buffering for buffer "o_ub"`），即**用双缓冲换静态 extent**。
- **裁决**：主选 = 统一运行期 clamp（保住双缓冲，单代码路径）；备选 = P6 静态化（若 Stage 4 实测显示运行期
  count 拖慢描述符，则按 shape 分派）。判定阈值与裁决计划见 §1.6.3。
- **语义保持论证**：两种形态都只读写 `[t0, t0+n)` 的行，`n` 由同一件事算出（`min(R, total−t0)`）；
  多算的行不写出，与源 `if row < total` 同语义。残块段 `n == 0`（`total % R == 0`）时用 `if n > 0` 守卫。

**重设计项 R5：输出写回（head-major 取址 + 运行期 count）**

- **源方案**：`delta[h, row] = rows[i]` 逐行标量写（packed 分支）。
- **NPU 新算法**：每 tile 一次压缩写回：`T.copy(d_ub[0:n, 0:1], delta[h, t0 : t0+n])`，`d_ub` 为
  `(R, 8)` fp32（32 B 行间距，供 `vsts + PAT_VL1` 落单 lane）。
- **取址/步长推导（head-major）**：`delta` 是 `(heads, total)` 的 C 连续张量，元素 `(h, t)` 的线性下标是
  `h · total + t`（`total` 为运行期量）→ 描述符的目的基址 = `delta_base + 4·(h·total + t0)`，
  **每个 tile 一次运行期乘法**（block 级标量，非逐元素）。
  **目的端步长语义不变**：最内轴是 `t`（stride 4 B），故写回仍是一条“count = n、元素 4 B、目的连续、
  源间距 32 B”的压缩描述符——**与 dense 的 `(B,H,S)` 形态逐参数同构**（dense：`..., 64, 4, mode, 4, 32`；
  本函数 P1/P4：`..., min(64, total_q − t0), 4, mode, 4, 32`），差别只有 ① 基址多一次 `h·total` 乘法、
  ② count 由静态 `nlast` 变成运行期 clamp。
  实测证据（P5，H=128）：`asc_copy_ub2gm_align(&delta[(it>>7)*64 + (it&127)*total_q], d_ub, min(64,
  total_q − (it>>7)*64), 4, store_mode(4), 4, 32)`——`(it>>7)` 是 t0 块号、`(it&127)` 是 h，
  基址正是 `h·total + t0`。
- **为何必须用运行期 clamp 而不是静态 `R`**：N3 实测——把 extent 写成静态 `R` **也能编译**，且编译器会
  自行插入最后轴 clamp 使 count 变成 `min(64, max(total_q − t0, 0))`，对 `(heads, total)` 这个二维基址
  “恰好”正确。但这是**未文档化**的编译器行为：若把 `delta` 声明成一维扁平张量 `(heads·total,)`，同一
  clamp 只会停在张量末尾，静态 extent 就会把最后一个 tile 的 `R−n` 个值写进**下一个 head 的行**
  （`total % R ≠ 0` 时），且没有任何诊断。故本设计**显式**写 `n`（与编译器 clamp 同值，但边界不依赖它）。
- **语义保持论证**：索引映射 `d_ub[i,0] → delta[h, t0+i]`（`i < n`）由 A4 机器验证：8 个几何
  （含 `T=1/T=65/T=200/T=63` 的残块与 `T=0` 空输入）**逐位相同 + 覆盖恰好**（EQUIV_PASS，§1.6.1）。

**§0.5 处置汇总**：表内 12 个条目——**保留 7 项**（语义层 2：数学语义、I/O 契约；优化层 5：O1、O2、O5、O9、
O10）、**等价替换 3 项**（O3、O4、O6'）、**重新设计 2 项**（O8→R1、O7→R2）、**舍弃 0 项**；packed 差异中的
搬运 extent、尾块边界与写回取址由 §0.6 的 R3/R4/R5 承接（这三项不是“源优化手段被重设计”，而是“dense 的
处置结论在 packed 下失效后重新推导”，故单列）。
四项重设计均满足 §1.6.0 调研结论（单遍流式、无跨核归并、Vector 单元承担逐元素与归约），且已按 Phase 2
继续做公式级优化（§1.6.1 A1–A4）与向量化替代分析（§1.6.2），未止步于源公式直译。

### 0.7 标杆实现

- 源算子路径：`/workspace/TileOPs/src/tileops/_kernels/attention/mla/bwd.py`（第 38–77 行 `_delta_kernel`；
  第 137/148 行是 `delta` 的消费端 shape 与声明，第 222–224 行是 packed 分支的消费处）。
- 源仓参考基准：`examples/TileOPs/tests/ops/test_mla_varlen_bwd.py`（容差 `_tolerance(dtype)`；smoke 几何
  `q_lens [256, 384]`、`H=8`、`D_v=128`；`test_mla_varlen_bwd_general_kernel` 为 `q_lens [256,128]`、`H=4`、
  `D_qk=128`、**`D_v=64`**——即本算子的 **pad 路径** smoke）。
- manifest：`examples/TileOPs/tileops/manifest/attention.yaml` 的 `MLAVarlenBwdOp` 条目（roofline
  表达式含 `H·Σ seen_rows·2·D_v` 的 delta 项与 `H·Σ q_len·4` 的写出项，§1.6.0 已按同口径核对）。
- 本函数的 golden **不移植**反向全过程，而按 §0.1 语义落到单式
  `(o.float() * do.float()).sum(-1)` 的 head-major 转置——与源 docstring 逐字一致。

### 0.8 复用与差异分区（本函数 vs dense）

**A. 复用部分（引用 dense 证据，并说明为何对本函数成立）**

| # | 复用项 | dense 侧证据 | 为何对本函数成立（结构性理由） |
|---|---|---|---|
| 1 | **数学内容**：逐 `(h,t)` 的 `D_v` 维 fp32 点积 | `examples/mla_dense_bwd/_delta_kernel/DESIGN.md` §0.1（dtype 语义：16→32 无损、乘与加皆 fp32）、真机 23/23 | 同一源码工厂、同一计算语句（§0.3 步骤 2–5 逐字相同）；packed 与 dense 只差输出下标空间 |
| 2 | **编程模式 `simd`**（SIMD/`T.SimdVF`） | dense §2（Cube 分形 N≥16 与“每行 1 输出”不匹配；SIMT 每行一次 AllReduce） | 与 layout 无关：两者都是“乘 + 行内归约”，无矩阵乘 |
| 3 | **一维核 + `R` 行 UB tile + `T.Pipelined(num_stages=2)` 双缓冲** | dense §0.6 R1、§3.3；探针 `design_probe_compile_v2.py` F1 | 平台 `T.Kernel` 只接受一维网格（`programming.md`），与 packed/dense 无关 |
| 4 | **逐行 SIMD `vcadd` 链**（`vld→2×vcvt→2×vmul→vadd→vcadd→vsts`） | dense §0.6 R2、§3.4；`PL-950-simd-reduce.py`；探针 F1/F2/F4 | 只依赖“整行驻留在一个向量寄存器”，与行号解址无关；本函数 P1/P2/P3 生成码逐字同形 |
| 5 | **压缩批量写出**（`d_ub[R,8]` 32 B 行间距 + 单条 UB→GM 描述符） | dense §1.6.1 A3、§3.2 步骤 8；探针 P1 生成 `...,64,4,...,4,32` | 目的端最内轴仍是连续 4 B（`[H,T_q]` 的 `t` 轴），描述符参数同构（本函数 P1/P4/P5 实测） |
| 6 | **构建期 `nvec` 展开**（`row_dot` helper，禁运行期循环携带向量累加器） | dense §3.3 注 2；探针 F5/F6 COMPILE OK、F7 `Immutable variable 'acc' ...` | `nvec = DW/128` 只由 `dim_v` 决定；本函数 P3（`D_v=192`）生成 `vadd(vadd(simd_inst::vmul(...` 单表达式 |
| 7 | **目的端切片 + 两侧 `T.clear`**（`D_v < DW`） | dense §3.3 注 3、§9.2；探针 F2/F3（F3 复现 `PlanMTECopy ... got src=4096 and dst=8192`） | 切片是平台“整 buffer ↔ 整 buffer 需元素数相等”的要求；本函数 P2/P3 目的行距 256/512 B 而每行只写 128/384 B，N1 复现同一拒绝 |
| 8 | **A1/A2/A3 三项等价优化的结论** | dense §1.6.1（12/12 `EQUIV_PASS`），本函数 `verify_equiv.py` 重跑同一结果 | A1/A2/A3 是**逐行局部**性质（寄存器融合 / 归约分组 / 暂存-压缩映射），与行号空间、输出布局无关 |
| 9 | **UB 预算口径与 `R ≤ 64` 的取值** | dense §4.5（66 KB / 248 KB = 0.27） | 缓冲形状 `(R, DW)` 逐字节相同（`DW=128`），故容量结论直接承袭；packed 另测 `D_v=192` 的 pitch（§4.5） |
| 10 | **不启用 fast-math** | dense §7.3（FMA 收缩会改变乘积舍入，使 A1 的“逐位等价”失效） | 代码生成路径相同；源算子开 `TL_ENABLE_FAST_MATH` 属**不迁移**的编译配置 |

**B. packed 差异部分（本文重新推导，逐项给证据）**

| # | 差异 | 源 packed 分支 | 本设计 | 章节 | 证据 |
|---|------|---------------|--------|------|------|
| ① | **动态行轴 → 动态 tile 流** | grid 第二维 `ceildiv(total, 32)` 随运行期变化 | **静态 56 核 persistent + 核内动态 tile 流**（`ntiles = ceildiv(total,R)·heads`，均衡划分）；核内循环界运行期、tile 内几何全静态 | §0.6 R1、§5.2、§5.5 | P1/P7 COMPILE OK；设备码循环界 `((total_q+63)>>6)·bx/7`、`min(bx·cpt, ntiles)` |
| ② | **无静态尾段 → 运行期 clamp** | `if row < total` 谓词 | `n = min(R, total−t0)` 同时约束两条装载与写回；静态 `R` 的等价性由编译器自行 clamp“碰巧”成立但不作契约（N3） | §0.6 R4、§6.4 | P1/P2/P3/P5 生成 `min(64, total_q − t0)`；P6 为静态化备选；N3 为反例 |
| ③ | **head-major 输出取址** | `delta[h, row]` | `delta[h, t0:t0+n]`，基址 `4·(h·total + t0)`；目的步长语义不变（连续 4 B） | §0.6 R5、§3.2、§4.2 | P5 设备码 `delta[(it>>7)*64 + (it&127)*total_q] + count min(64, ...)` |
| ④ | **`total` 运行期 → 输出 shape 动态** | `(heads, T.dynamic("total_q"))` | 保持 `(heads, total)`；`out_idx=[2]` 分配随 `total_q` 变化 | §3.4、§4.2、§4.6 | P4 COMPILE OK（`out_idx=[2]`，`delta` 第二维动态） |
| ⑤ | **`D_v=64` pad 路径 × 动态 extent** | 无（GPU 无 pad 概念） | 切片 `[0:n, 0:D_v]` + 两侧 `T.clear`；`T.clear` 实测覆盖**全部版本**（P2：4096×4 half ×2 = 16384 half = 两个版本） | §3.3 注 3、§4.5 | P2 设备码：clear 覆盖 offset 0 与 16384 half 各 16384 half；装载 `..., min(64,...), 128, ..., 512, 256` |

**C. 为何“复用”在本函数成立（一句话）**：本函数与 dense 版来自**同一源码工厂、同一行级计算**——差异只在
「行轴是否静态」与「输出是 `[H,T_q]` 还是 `[B,H,S_q]`」，而复用项（SIMD 归约链、UB tile 形状与预算、
切片/清零、A1–A3）都由**行内维度与平台约束**决定，不依赖行轴空间；唯一的交互点是**写回的 extent 与基址**
（dense 用构建期静态 `nlast`，packed 必须运行期 clamp，即 §0.6 R4/R5），本文已单独重推并给 P5/P6/N3 证据。

---

## 1. 概述

### 1.1 算子名称

`_delta_kernel`（`MLAVarlenBwdOp` 第 1 个 launch：注意力反向的行标量 `delta`，packed / head-major 版）

### 1.2 功能描述

对每个 `(head, packed token)` 对，在 value 维上计算前向输出 `o` 与输出梯度 `do` 的**行点积**，得到后续
`dP − delta` 需要的 float32 行标量，并以 head-major `[H, T_q]` 布局写出（供 `_bwd_kernel` 按 `[H,T_q]`
一次连续段装载）。只依赖 `o`、`do`，与 `lse`、`cu_seqlens` 均无关；`T_q` 为**运行期**行数。

### 1.3 数学公式

$$
\delta[h,t] \;=\; \sum_{d=0}^{D_v-1} o[t,h,d]\cdot do[t,h,d],
\qquad
o,do \in \mathbb{R}^{T_q\times H\times D_v}\ (\text{fp16/bf16}),
\quad \delta \in \mathbb{R}^{H\times T_q}\ (\text{fp32})
$$

乘积与累加均在 float32 中进行（§0.1 dtype 语义）；`D_v = dim_v`（manifest workload 全为 128；op 级 smoke
另有 `D_v = 64`（pad 路径）与 `D_v = 192`（`nvec=2`），设计按参数化处理，§4.5/§8.3）。`T_q` 动态。

### 1.4 算法描述

**（迁移决策后的 NPU 侧算法；与源算法的差异逐条注明来源）**

1. **任务划分**（差异 ← §0.6 R1）：`T.Kernel(G)`，`G = 56`（构建期常量，§5.5 ②）。逻辑任务数
   `ntiles = ceildiv(T_q, R) · H`（运行期）；第 `bx` 个核负责
   `it ∈ [ntiles·bx / G, ntiles·(bx+1) / G)`。每个 `it` → `h = it % H`、`t0 = (it // H) · R`，
   `n = min(R, T_q − t0)`。块内 `T.Pipelined(it0, it1, num_stages=2)`。
2. **搬入**（差异 ← §0.6 R3）：每 tile 把 `o[t0 : t0+n, h, :]` 与 `do[t0 : t0+n, h, :]` 各用**一条跨步 DMA**
   搬进 UB（16-bit 原样，不提升），目的端写成显式 region `o_ub[0:n, 0:D_v]`（`DW > D_v` 时元素数才相等），
   双版本缓冲；`n` 是运行期标量。
3. **逐元素乘 + 提升**（差异 ← §1.6.1 A1）：在 `T.SimdVF` 内按行 `vld` 一个行向量（128 个 16-bit 元素 =
   一个向量寄存器），`vcvt` 两次（PART_EVEN/PART_ODD）得两半 float32x64，两两 `vmul`。**fp32 乘积只存在于
   向量寄存器**，不物化 fp32 tile。
4. **行归约**（差异 ← §0.6 R2）：`vadd` 合并两半 64 lane 部分和 → `vcadd` 做 64 lane 二叉树归约 →
   `vsts` 以 `PAT_VL1`/`extent=1` 写进 32 B 行间距的暂存缓冲 `d_ub[R, 8]`。`D_v > 128` 时 `nvec` 段的部分和由
   **构建期展开的单一表达式**合并（不进入运行期循环）。
5. **写出**（差异 ← §0.6 R5 / §1.6.1 A3/A4）：每 tile 结束后用**一条 UB→GM 跨步 DMA** 把 `d_ub[:, 0]`
   压缩写进 `delta[h, t0 : t0+n]`（float32、head-major、目的连续 4 B、源间距 32 B、count = `n` 运行期）。
6. **边界**（差异 ← §0.6 R4）：`T_q % R ≠ 0` 时**没有**静态尾段——所有 tile 统一用运行期 clamp；行循环上界
   仍是静态 `R`，多算的行不写出（`vsts` 已落 `d_ub`，但写回 count 为 `n`）。`D_v < DW` 时循环前两侧
   `T.clear`，装载只写 `[0, D_v]` 列。

与源算法的结构差异一句话：源是“二维网格（第二维随 `total` 动态）+ 32 行 fragment + warp 树归约 + 逐行标量
写”，本设计是“**静态 56 核 + 核内动态 tile 流 + `R` 行 UB tile + 逐行 SIMD `vcadd` + 每 tile 一次压缩写回**”；
**数学内容与 dtype 语义不变**（同一批 fp32 乘积、同一 fp32 和），只有分组、数据驻留位置与写回粒度变化。

### 1.5 数据流图

```
GM o[T_q,H,D] ──(跨步 DMA, n×256B；D_v=64 时 n×128B，目的 region [0:n,0:D_v]，n = min(R, T_q−t0))──► UB o_ub[R,DW] 16-bit ─┐
GM do[T_q,H,D] ─(同上)──────────────────────────────────────────────────────────────────────────────► UB do_ub[R,DW] 16-bit ─┤
                                                                        ├─► SimdVF 每行:
                                                                        │     vld → 2×vcvt → 2×vmul → vadd → vcadd
                                                                        ▼
                                                     UB d_ub[R,8] fp32 (32B 行间距) ──(压缩 DMA, n×4B, 目的基址 4·(h·total+t0))──► GM delta[H,T_q] fp32
```

`n` 是运行期标量（三条描述符的 count）；`DW = ceil(D_v/128)·128`（`D_v=64 → 128`，`D_v=192 → 192`，
编译器实测 UB 行 pitch 为 256 half = 512 B，§4.5）。无 L1/L0C/中间 GM 往返。
### 1.6 算法调研与优化分析

#### 1.6.0 算法调研（Algorithm Research）

> 调研对象：**同一数学语义（逐 `(h,t)` 的 value 维点积）的算法族**。源算法（单遍融合 + 并行归约 + 谓词写回）
> 只是基线候选之一。本算子属“多步/规约类”（乘 + 跨 `D_v` 元素归约 + 输出），按
> `.agents/skills/tilelang-op-design/references/algorithm-research.md` §2 需**完整调研**（R1 候选表 + R3 四口径
> 对比 + R4 逐候选评估）。
> **本地检索记录**：`python3 .agents/tools/kb_search.py "row dot product reduce delta o*do" --task-dir
> examples/mla_varlen_bwd/_delta_kernel` → `no entries matched`；`kb_search "reduction"` / `"vcadd"` 同样无命中；
> 平台 `knowledge.algorithm_candidates`（`.agents/platforms/toolchains/ascend/knowledge/algorithm-candidates.md`）
> 的 reduction 族只有一行“单行 SIMT sum/max reducer”（`PL-950-row-primitives.md`），已作为候选 C7 纳入评估；
> `pattern-library` 的 `constants.md`（硬件常数）与 `cases.md`（案例索引）无本算子族条目。故候选由
> **参考表命中行（C7）+ 恒等变形自查 + 结构判据 + 模型知识**枚举；算子族为单点积、候选空间在 R1 表内封闭
> 且每项都可由结构或常数判定，未做互联网检索（该层为可选补充层，未命中不影响结论）。

**R1 等价化简公式候选**（基线必须在表内；正式等价论证与收益量化在 §1.6.1 完成，此处只做初判）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 是否纳入 R3 对比 |
|---|------|------------|-----------|---------|----------------|
| C0 | **基线：源算法**（packed 分支） | `Σ_d fl32(o_d·do_d)`，单遍、fp32 乘 + fp32 树归约 + 谓词逐行写回 | — | — | ✅（基线） |
| C1 | 寄存器融合（不物化 fp32 乘积 tile） | 同 C0，乘积只在向量寄存器里存活 | 数学恒等（同一批 fp32 乘积与同一个和） | 省 UB/寄存器占用与一次片上往返 | ✅（采纳 A1） |
| C2 | 归约分组改为 64 lane 二叉树（两半 `vadd` 后 1 次 `vcadd`） | `Σ_d p_d`，分组不同 | 容差内等价（fp32 分组差异，无系统偏置） | 单指令 `vcadd` 代替多级跨线程合并 | ✅（采纳 A2） |
| C3 | 输出批量写（暂存 + 压缩 DMA） | 数值不变，只改写出粒度 | 数学恒等（纯数据搬运） | `R` 次标量写 → 1 条描述符 | ✅（采纳 A3） |
| C3b | **head-major 批量写回 + 运行期 clamp**（packed 新增） | 基址 `4·(h·total+t0)`、count `min(R, total−t0)` | 数学恒等（索引映射 + 数据搬运；A4 机器验证逐位相同） | 与 C3 同（1 条描述符/tile），并满足消费端 `[H,T_q]` 连续段契约 | ✅（采纳 A4） |
| C4 | fp16 乘 + fp32 累加（省 2 条 `vcvt`） | `fl16(o_d·do_d)` 再提升 | **不等价**：乘积先舍入到 fp16（相对误差 ≤ 2^-11 ≈ 4.9e-4） | 每行少 2 条向量指令 | ❌（精度：`delta` 绝对误差 ≈ 4.9e-4·‖p‖₂ ≈ 5e-3，与消费端 atol 5e-3 同阶） |
| C5 | 两遍：先物化 `p = o⊙do` 到 GM 再归约 | `p` 落 GM 后第二次读 | 数学恒等 | **负收益**：GM 流量 ×2.98（R3 已量化） | ❌（R3 淘汰） |
| C6 | 对角 GEMM：`diag(O·Doᵀ)` / ones-GEMM | 用矩阵单元算 `[R,R]` 取对角 | 数学恒等（精确算术下） | **负收益**：FLOPs ×64（R3 已量化）+ 跨引擎同步 | ❌（R3/R4 淘汰） |
| C7 | 单行 SIMT reducer（参考表唯一命中行） | `alloc_reducer → reducer_update → finalize_reducer`，每行一次跨线程 AllReduce | 容差内等价（求和分组不同） | 复用平台已验证原语 | ❌（R3/R4：每行一次 barrier，N=4.19M 行时发射开销不可接受） |
| C8 | D 轴跨核切分 + 跨核合并 | 两核各算 `D_v/2` 再相加 | 容差内等价 | **负收益**：每行 4 B 输出却要跨核同步 | ❌（R3/R4 淘汰） |
| C9 | 代数化简（对称性 / 缩放 / 复用相邻行） | — | 无：`o` 与 `do` 逐行独立，点积不存在把 `D` 次乘加降到更低的代数变形（信息量下界即 `D` 乘 + `D` 加） | — | ❌（结构判定） |
| C10 | 分请求（按 `q_len` 切）批处理 | 用 `cu_seqlens` 把 tile 流按请求分段 | 数学恒等 | **负收益**：本算子输出与请求边界无关（`delta[h,t]` 只依赖行 `t`），切分会引入请求解码开销而不减少任何工作 | ❌（结构判定：packed 行轴无需按请求分段；`cu_seqlens` 不进入本 launch） |

**R2 在线算法**：**有**（与 dense 同一条结构判据）。逐行点积天然是单遍流式：每行一个 `O(1)` 寄存器状态，
行与行**无依赖**（`delta[h,t]` 只依赖本行 `D_v` 个输入），因此 ① 单遍即可（每元素只读一次，扫描 1 遍）；
② `D_v > 128` 时同一行内按段合并，段部分和在**构建期**展开成一个 `vadd` 表达式，运行期状态仍是 `O(1)`
个寄存器；③ tile 循环本身是流式（每 tile 独立，无跨 tile 状态）；④ **packed 差异不改变这条结论**：
`total` 动态只让“流有多长”变成运行期量，流式结构不变（`ntiles` 运行期、每 tile 独立）。
不需要跨核 partial + merge（那是 C8，已按“每行仅 4 B 输出”淘汰）。中间缓冲峰值 `O(tile)` = 66 KB，不是 `O(N)`。

**R3 复杂度对比**（四口径；代表 workload = `ds-v3-8x4k`（packed manifest）：
`q_lens = 8×4096`，`H = 128`，`D_v = 128`，bf16，`T_q = 32768`，行数 `N = T_q·H = 4,194,304`；
口径含“每元素一次读 + 每行一次写”，不含 kernel 启动与 16-bit 提升）：

| 算法候选 | FLOPs | 访存量 (Bytes) | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|-------|---------------|---------|-------------|---------------------|
| **C0/C1/C2/C3/C3b 基线（源算法）** | 2·N·D = **1.074 G** | 读 2·N·D·2 = **2.147 GB** + 写 N·4 = 16.8 MB = **2.164 GB** | 1（`o`/`do` 各 1 次） | 32 KB（fp32 乘积 fragment [32,128]×2） | `N` 行可任意切核；**无跨核代价** |
| C4 fp16 乘 | 1.074 G（少 `vcvt`） | 2.164 GB | 1 | 同上 | 同基线，但**精度不达标**（R1 已量化） |
| C5 两遍物化乘积 | 1.074 G（+物化） | 读 2.147 + 写 2.147 + 读 2.147 + 写 0.0168 = **6.459 GB** | 3 | `O(N·D)` 落 GM | 同上，但流量 ×2.98 |
| C6 对角 GEMM | 2·R·N·D = **68.7 G**（64×） | 2.147 GB + L1/L0C 往返 + 对角抽取 | 1 | L0C `[R,R]` fp32 = 16 KB/tile + 跨引擎中转 | 需 Vector↔Cube 协作（跨引擎同步） |
| C7 单行 SIMT reducer | 1.074 G | 2.164 GB | 1 | 同上 | 每行一次 `tvm_thread_allreduce` + barrier；N = 4.19M 行 |
| C8 D 轴跨核切分 | 1.074 G | 2.164 GB + 每行 2 次 4 B 原子/部分和 = **+33.6 MB**（2·4·4.194304e6 B） | 1（+merge） | 同上 | 每行一次跨核合并/原子；原子次数 = N |
| **收敛候选（C0 的 NPU 落地）** | **1.074 G** | **2.164 GB** | **1** | **66 KB（67584 B）**：`o_ub`/`do_ub` 2×16 KB×2 版本 + `d_ub` 2 KB | 无跨核协作、无原子 |

**R4 硬件亲和性评估**（逐候选对照检查清单；负向淘汰依据见“结论”列）：

| 候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|------|-------------|---------|------------|---------|------------|------|
| **C0–C3b 收敛候选** | 逐元素乘 + 归约 → **Vector（SIMD）单元**，正是本平台能力（§0.5 证据） | 66 KB / 248 KB = **0.27**（67584/253952）✅ | 行宽 `D_v·2 B = 256 B` = 2 个完整 128 B 段 ✅；UB 行 pitch 与向量寄存器对齐 | **运行期分块**：`n = min(R, total−t0)`（`total` 动态，见 §0.6 R4）；核内循环界运行期、tile 内几何静态 | 双版本 UB + `T.Pipelined`（P1 生成双缓冲版本偏移） | ✅ 采纳 |
| C4 fp16 乘 | Vector ✅（还少 2 条 `vcvt`） | 同 | 同 | 同 | 同 | ❌ 精度不达标（R1 C4 已量化；dtype 语义层不得改） |
| C5 两遍 | 同基线，但 MTE2/MTE3 流量 ×2.98 | 需 `O(N·D)` 中间张量落 GM | 同 | 同 | 两 kernel 或核内两遍，无收益 | ❌ 流量 6.459 GB ≫ 2.164 GB（常数见下：1.03 TB/s 下 6.27 ms vs 1.72–2.10 ms） |
| C6 对角 GEMM | Cube 需 N ≥ 16 fractal，而每行只有 1 个输出 → 64 个 N lane 中 63 个浪费；乘积需先由 Vector 算出再送 Cube | L0C 16 KB/tile + L1 中转 | `R ≥ 16`（R=64 可行） | 同 | 需 Vector→Cube 跨引擎同步 | ❌ FLOPs 64× + 跨引擎同步，对“每 512 B 输入只产生 4 B 输出”的算子无收益 |
| C7 单行 SIMT reducer | Vector ✅（`PL-950-row-primitives.py` 已验证单行用法） | 同 | 同 | 每次归约需 `finalize_reducer` 的 barrier | 同 | ❌ 每行一次跨线程 AllReduce：N = 4.19M 行，barrier/发射开销与“每行 1 条 `vcadd`”相差一个量级（§1.6.1 A2 收益栏） |
| C8 D 轴跨核切分 | Vector ✅ | 同 | 64 lane/半 ✅ | 同 | 需 GM 原子（`PL-950-atomic-splitk` 路径为 fp32 GM add）或第二遍 merge | ❌ 每行仅 4 B 结果却要一次跨核合并，原子次数 = N |
| C10 按请求分段 | Vector ✅ | 同 | 同 | 引入 `cu_seqlens` 运行期解码 | 同 | ❌ 不减少任何工作、不改变输出布局，只增加解码（本 launch 与请求边界无关） |

**调研结论**：选定**候选 C0 的算法结构 + C1/C2/C3/C3b 的实现级等价优化**——即“单遍、fp32 乘积、逐行 64 lane
二叉树归约、结果按 head-major 批量压缩写回”。与基线（源算法）的结构差异只有“乘积驻留位置 + 归约分组 +
写出粒度/取址”，**无算法族替换**；packed 与 dense 共用同一结论（行轴是否静态不改变任何复杂度口径）。
依据：R3 中收敛候选的访存 **2.164 GB 是理论下界**（每元素必读一次、每行必写 4 B），C5/C6/C8 分别把流量
×2.98、FLOPs ×64、跨核原子引入，均在本平台 Vector 单元的带宽/发射比下为负收益；C7 是参考表唯一命中行但
每行一次 barrier，在 4.19M 行规模上不可接受；R4 中收敛候选在计算单元、容量、对齐、流水四个维度全绿，
唯一“边界不静态”项由运行期 clamp 处置（§0.6 R4）并有编译证据。
**“无更优替代”的调研范围**：已查 `algorithm-candidates.md` 全表（reduction 族仅 C7 一行）、
`.agents/platforms/toolchains/ascend/knowledge/INDEX.md` 索引条目、`ALG-950-formulas.md` /
`operator-patterns.md` / `PL-950-row-primitives.md`、`pattern-library` 的 `constants.md` / `cases.md`、
以及本仓 `examples/` 内全部 ascend-target 向量算子（`paged_cache_write`、`moe_permute_align` 等）；
kb_search 三组关键词无命中。本结论即 §1.4、§1.6.1–§1.6.3 的分析对象。

**设计期估算（D-2）**：代表 workload `ds-v3-8x4k`（`T_q=32768, H=128, D_v=128, bf16`），
估算下界 = max(流量项, 发射项, 容量项)：

- **流量项**：2.164 GB ÷ 带宽 → **1717–2101 µs**：
  - 上界 2101 µs：按 `CONST-mte2-degradation` 的**每核**地板 18.4 GB/s × 56 Vector 核 = **1.03 TB/s**
    → 2.1643 GB ÷ 1.03 TB/s = 2101 µs（本设计采用该保守值）；
  - 下界 1717 µs：直接复用该条目记录的 910B2C **48 核聚合**混合读写地板 **1.26 TB/s** → 1717 µs；
  - 两个常数均绑定 **910B2C**、toolchain `tilelang 0.1.2+15ad002b3d`（`CONST-mte2-degradation` 条目），
    **不是** ascend-950pr-9579 的实测值 → 标注为**未实测外推**。
- **发射项**：每行 11 条向量指令（2 `vld` + 4 `vcvt` + 2 `vmul` + 1 `vadd` + 1 `vcadd` + 1 `vsts`）×
  4,194,304 行 = 46.1 M 条 ÷ 56 核 ÷ 1.8 GHz ≈ **458 µs**（若 `vcvt`/`vcadd` 为多周期则上界约 0.8–1.0 ms；
  `CONST-vector-launch-overhead` 的“0.5 µs/op 发射”口径绑定 910B2C，同属未实测外推）。
  另加描述符开销：每 tile 3 条（2 装载 + 1 写回）× 65536 tile = 196.6 K 条 ÷ 56 核 = 3511 条/核；按
  `CONST-mte2-degradation` 的指令维度口径 ≈ 300 ns/指令（910B2C 标定，未在本型号复核）→ 约 1.05 ms
  **若完全串行化**；双缓冲下应与传输重叠，故只作风险项（§9.1 R2），不计入下界。
- **容量项**：66 KB / 248 KB = 0.27 → 不构成约束（UB 容量取自 `Ascend950PR_9579.ini` `ub_size=253952`）。
- **估算下界: 1717–2101 us**（= max(流量项 1717–2101 µs, 发射项 458 µs + 描述符开销)；容量项 0.27 不构成约束。
  描述符开销 ≤1.05 ms 是“完全串行化”上界，双缓冲下应与传输重叠，故不计入下界而列入 §9.1 R2。
  常数引用 `CONST-mte2-degradation`、`CONST-vector-launch-overhead`，均为 910B2C 未实测外推）。
- **与 manifest roofline 的同口径交叉核对**（`examples/TileOPs/tileops/manifest/attention.yaml` 的
  `roofline.flops` / `roofline.bytes`）：delta 的贡献是 `H·Σ q_len·2·D_v` = 128·32768·2·128 =
  **1.074 GFLOP**、写出 `H·Σ q_len·4` = **16.8 MB**，与本节 R3 的 FLOPs 与写出口径逐项吻合；
  读入项对应 `2·qv_elems·elem_bytes` = 2.147 GB。故 1.074 GFLOPs / 2.164 GB 与该 manifest 表达式同口径。
- **注意**：`CONST-mte2-degradation` 的 L2 驻留口径提示“同输入张量逐 launch 复用时有效读带宽远高于 HBM
  口径”，故 bench 循环下实测可能优于本估算——这是 §9.1 的风险项之一，不改变“流量主导”的结论。

#### 1.6.1 数学等价优化（公式级）

> 分析对象为 §1.6.0 选定的收敛候选。四项采纳项均不改数学内容，改的是“乘积驻留位置 / 归约分组 / 写出粒度 /
> 写回取址”。机器验证脚本：`verify_equiv.py`（本目录），torch CPU，fp64 精确参照；判决口径 =
> `|cand−ref| ≤ 1e-5·Σ|p_d| + 1e-6`（归约误差正比于 `Σ|p_d|` 而非 `|Σ p_d|`，见脚本内 `RTOL_SCALED` 注释），
> 并附标准容差（atol=1e-4, rtol=1e-4）下的违反率。A1/A2/A2b/A3 与 dense 同一批用例（同一数学）；**A4 是本函数
> 新增的 packed 写回项**，用自己的几何集（`T ∈ {0,1,63,65,128,200,256,4096}` × `H ∈ {1,4,8,128}`）。

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| A1 | 寄存器融合乘（不物化 fp32 乘积 tile） | `p[i,d] = fl32(o)·fl32(do)`（写入 fp32 tile）→ `Σ_d p[i,d]` | `Σ_d vmul(vcvt(o), vcvt(do))`，乘积只活在向量寄存器 | 数学恒等：两种形态对每个 `d` 都执行**同一个** fp32 乘法并四舍五入到 fp32，再对同一批项求和；只移动了值的存放位置（机器验证：cand 与 base **0 ulp**，12/12 用例） | 源基线物化 `[32,128]×2` fp32 = **32 KB**；若在 NPU 上同样物化 fp32 乘积 tile 则为 `2·R·D·4 B` = **64 KB**（R=64, D=128）；A1 把这 64 KB 降到 0，并省掉每 tile 两次 `R·D·4 B` 的片上往返——这正是双缓冲（UB 66 KB）能成立的原因 |
| A2 | 归约分组：`vadd` 两半 + 1 次 64 lane 二叉树（**实现分组**） | `acc = ((…((0+p0)+p1)+…)+p_{D-1})`（顺序/多级合并） | `vcadd( vmul(vcvt(o,PART_EVEN),vcvt(do,PART_EVEN)) + vmul(vcvt(o,PART_ODD),vcvt(do,PART_ODD)) )`——lane k 得 `p[2k]+p[2k+1]`，1 次 `vcadd` 做 6 级 pairwise | 容差内等价：同一批 fp32 项的重新分组，误差项仍是 fp32 舍入（无系统偏置）。机器验证（`impl_grouping`，即本行公式）：12/12 用例违反率 0；且树形比顺序更接近 fp64 真值（random/fp16/D128：322 ulp vs 基线 2238 ulp）。`nvec == 1` 下与 A2b（2×`vcadd` + 标量加）**逐位恒等**（12/12 数字完全相同）。fp16/bf16 舍入影响：输入提升到 fp32 后乘、加都在 fp32，16-bit 只在入口（无损提升），分组差异不会放大到 16-bit 量级 | 归约从 `T.reduce_sum` 的多级跨线程合并（barrier + 共享内存中转）变为每行 1 条 `vcadd`：若沿用 SIMT reducer 路径（C7），每行一次 `tvm_thread_allreduce` + barrier（未实测假设：>100 cycle/次），N = 4.19M 行时约 4–8 ms 量级，超过 1717–2101 µs 的流量下界 |
| A2b | （对照行）dense v1 引用的分组：2×`vcadd` + 标量加 | 同 A2 基线 | `vcadd(p[0:64]) + vcadd(p[64:128])` | 容差内等价；在 128 = 2×64 处平衡二叉树有子树边界，故 `nvec == 1` 时与 A2 逐位恒等（机器验证 12/12 与 A2 数字一致） | 与 A2 同量级（2 条 `vcadd` + 1 次标量加，略差）；仅作对照，**不采纳** |
| A3 | 输出批量写（暂存 + 压缩 DMA） | 每行一次 4 B 标量写回 GM（`delta[h,t] = rows[i]`） | 写入 32 B 行间距的 `d_ub[R,8]`，每 tile 一条压缩描述符写 `delta[h, t0:t0+R]` | 数学恒等：纯数据搬运；索引映射 `d_ub[i,0] → out[i]` 机器验证 **0 ulp**（12/12） | 写出指令数从 `R`/tile（N 次全局）降到 1/tile（N/R 次）；按 910B2C 的“段/指令”口径估算，N 次 4 B 写约 0.2–0.4 ms 的指令开销被摊掉（未实测外推） |
| A4 | **packed head-major 写回**：基址 `4·(h·total+t0)` + 运行期 count | 逐行 `delta[h, t] = s[t]`（`t ∈ [t0, t0+n)`，`n = min(R, total−t0)`） | 暂存 `d_ub[i,0]` → 一条描述符写 `delta[h, t0:t0+n]`（目的连续 4 B、源间距 32 B、count = `n`） | 数学恒等：① 索引映射 `(h, t0+i) ↔ d_ub[i,0]` 逐位相同（8 个几何含 `T=0/1/63/65/200` 的残块全部 `bit_equal=True`）；② tile 流**覆盖恰好**（`covered=True`：每个 `(h,t)` 恰被一个 tile 覆盖）；③ 域外未被写（`clean_outside=True`）；④ 每个 tile 都满足 `n ≥ 1`（`bad_tile=0`）；⑤ 非有限值（`inf·0→nan`）经暂存后不变 | 与 A3 同（1 条描述符/tile）；此外**避免了 `[H,T_q]` 转置成 `[T_q,H]` 的代价**（后者会让消费端退化为 `T_q·4 B` 间距的跨步装载）；量化诊断：若把 extent 写成静态 `R`，`T=65/H=4` 会多写 **252** 个越界元素（`4×(64−1)`）、`T=200/H=8` 多写 **448**（`8×(64−8)`）、`T=63/H=128` 多写 **128**（每个 head 64−63=1 个），故 A4 的显式 clamp 不是可选优化而是正确性条件 |

**等价性机器验证结果**（`verify_equiv.py` 实跑输出，torch CPU + fp64 参照；本轮执行时间 2026-11-23，
Python `/workspace/venv-ascend950/bin/python`）：

| 采纳项 | 用例数 | 最大 ulp(cand, ref) | 最大违反率（尺度化口径） | 标准容差违反率 | 非有限值不一致 | 结论 |
|--------|--------|---------------------|--------------------------|----------------|----------------|------|
| A1 寄存器融合 | 12 | 3310 | 0.0000 | 0.0156（仅 large/fp16 的 1/64 行，见下注） | 0 | **EQUIV_PASS** |
| A2 实现分组（`vadd` 两半 + 1×`vcadd`） | 12 | 2798 | 0.0000 | 0.0156（同上） | 0 | **EQUIV_PASS** |
| A2b 对照分组（2×`vcadd` + 标量加） | 12 | 2798 | 0.0000 | 0.0156（同上） | 0 | **EQUIV_PASS** |
| A3 批量写出（暂存 → 压缩） | 12 | 2798 | 0.0000 | 0.0156（同上） | 0 | **EQUIV_PASS** |
| **A4 packed head-major 写回（含运行期 clamp）** | **8** | **0（逐位相同）** | **0.0000** | **0.0000** | **0** | **EQUIV_PASS** |

> A4 的 8 个几何与诊断（`verify_equiv.py` 输出）：
> `T128-H4`（全满 tile）、`T256-H1`（全满）、`T65-H4`（残 1 行，静态 extent 会越界 252）、
> `T200-H8`（残 8 行，越界 448）、`T1-H1`（单行，越界 63）、`T63-H128`（残 63 行、128 头，越界 128）、
> `T4096-H128`（多 tile，无残块）、`T0-H4`（空输入，`tiles=0`，两形态都为空）。
> 注：`large/fp16` 的 1/64 行是**抵消行**（`Σ|p_d| ≈ 1e5` 而 `|Σ p_d| ≈ 0`），在 atol+rtol·|ref| 口径下任何
> fp32 累加（含基线自身）都会“违反”；尺度化口径（误差 ≤ 1e-5·Σ|p_d|）下 0 违反。这不是等价性问题，而是
> fp32 累加对抵消行的固有性质，源算子同样具备（同一 dtype 语义），列入 §9.1 风险项。

**优化结论**：采纳 **4 项**（A1 寄存器融合、A2 实现分组、A3 批量写出、A4 packed head-major 写回 + 运行期
clamp），全部机器验证通过、无失败项；A2b 为对照行（同样通过，证明两分组在 `nvec == 1` 下逐位恒等），不采纳。
优化后公式（供 §3.1 拆解使用）：

```
for each tile (h, t0) of the dynamic stream, with n = min(R, T_q - t0):
    for i in 0 .. R-1:                       # static row loop; rows >= n are not written
        v_o  = vcvt_pair(o_ub[i, :])         # 16 -> fp32, two halves of 64 lanes
        v_d  = vcvt_pair(do_ub[i, :])
        acc  = vadd(vmul(v_o.lo, v_d.lo), vmul(v_o.hi, v_d.hi))   # lane k = p[2k]+p[2k+1]
        d_ub[i, 0] = broadcast_sum(vcadd(acc))                    # fp32, 64-lane tree
    delta[h, t0 : t0 + n] = d_ub[0:n, 0]     # head-major, base 4*(h*T_q + t0), count n
# D_v > 128 (nvec > 1): segments merged at BUILD time into one vadd expression, then one vcadd
```

其余空间（乘、加、提升、索引）已是逐元素最小操作集：无公共子表达式可消（每行只算一次乘积）、无除法/超越
函数可降代换、无重复读取的输入 tile（每元素只读一次）、无可复用的相邻行（行间独立）——这是“无进一步优化
空间”的结论与依据。**Packed 特有的再优化空间也已封闭**：`cu_seqlens` 不参与（C10 淘汰）、行轴无跨请求语义、
写回已是单描述符 + 精确 count（A4）。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由（不可替代时必填） |
|---|--------|-----------|-------------|---------|------------------------------|
| V1 | 行内逐元素乘 | `o_f[i,d] * do_f[i,d]` 的逐元素循环 | `T.simd.vmul`（fp32x64） | ✅ | — |
| V2 | 16-bit → fp32 提升 | `T.copy` 隐式 cast 循环 | `T.simd.vcvt(…, "float32", part=PART_EVEN/PART_ODD)` | ✅ | — |
| V3 | 两半 64 lane 部分和合并 | 逐元素累加 | `T.simd.vadd` | ✅ | — |
| V4 | 行内 128 元素求和 | `T.reduce_sum` / 标量累加 | `T.simd.vcadd`（64 lane 二叉树） | ✅ | — |
| V5 | 每行结果写暂存 | `d_ub[i,0] = rows[i]`（标量写） | `T.simd.vsts(…, PAT_VL1, extent=1)` | ✅ | — |
| V6 | GM → UB tile 搬入 | 逐元素/逐行 load 循环 | `T.copy`（单条跨步 DMA，P1/P2/P3/P5 实测） | ✅ | — |
| V7 | UB → GM 结果写回 | 逐行标量 store 循环 | `T.copy(d_ub[0:n, 0:1], delta[h, t0:t0+n])`（单条压缩 DMA） | ✅ | — |
| V8 | pad 列清零（`D_v` 非 128 倍数时） | 逐元素清零 | `T.clear(o_ub)` **且** `T.clear(do_ub)`（构建期 `DW != D_v` 分支，循环前各一次；P2 实测覆盖**两个版本**） | ✅ | —（只清一侧不够：另一侧 pad lane 是未初始化 UB，bit 模式若为非有限值则 `0·Inf = NaN`，§9.2） |
| V9 | 行循环 `for i in T.serial(R)` | 外层行迭代 | 无（每行一次 `vld/vcvt/vmul/vadd/vcadd/vsts`） | ❌ | **API 缺失**：`vcadd` 一次只归约**一个**寄存器；本平台无“跨多行批量归约”的已验证指令，把 64 行塞进同一寄存器需要 UB 内 64×128 转置链（`vdintlv/vintlv`），代价远大于省下的 `vcadd`（每行 1 条占 11 条的 9%）。已核 `simd.py` 导出全表（`vcadd/vcgadd/vcpadd` 均为单寄存器内归约）。见 §1.6.3 候选 7。**注**：行内 `nvec`（`D_v>128`）的分段合并**不是**运行期循环（构建期展开，§3.3 `row_dot`） |
| V10 | tile/核索引计算（`h, t0, n, bx, it0, it1`） | 标量整数运算 | 无 | ❌ | **block 级索引**：每 tile 一次、每核一次，不是逐元素热点；`h = it % heads`（常量模）、`t0 = (it // heads) · R`、`n = min(R, total−t0)`、`it0 = ntiles·bx // G` 合计 6–8 条标量整数指令/tile（对比 64 行 × 11 条向量指令），且无逐元素等价 API |
| V11 | 尾块/变长的运行期 extent（`n = min(R, total−t0)`） | 源 `if row < total` 谓词 | 无（改用运行期 clamp 的动态 extent） | ❌ | **依赖动态 shape 的边界处理**：`total` 是运行期量，`n` 无法静态化（§0.6 R4）；这里不是“保留标量循环”，而是把标量判断变成三条描述符的 count 参数（0 条逐元素指令）。静态化备选 P6（满 tile 静态 + 残块动态）已编译但会关闭双缓冲，故不作为主选；两者都不引入逐元素掩码 |
| V12 | 写回基址 `h·total + t0`（head-major） | — | 无（标量乘加） | ❌ | **block 级索引**：每 tile 一次 64-bit 乘加（P5 实测 `delta[(it>>7)*64 + (it&127)*total_q]`），非逐元素；若改成逐元素会退化 |
| V13 | `T_q % R` 是否为 0 的分支 | 构建期分支（dense） | — | ❌ | **不适用**：`total` 运行期，该判断在本函数根本不存在（构建期无法判断），故尾处理只能走 V11 |

**向量化结论**：逐元素与归约计算点（V1–V8）已全部向量化，所用 API（`vld/vcvt/vmul/vadd/vcadd/vsts`、
`T.copy`、`T.clear`）均有本平台证据（`simd.py` 导出 + `PL-950-simd-reduce` 已验用例 + 本任务
`history_version/design_probe_packed_v2.py` 的 P1/P2/P3 生成代码）。保留 4 处标量/循环，理由类别分别为
**API 缺失**（V9，附量化与查证过程）、**block 级索引**（V10/V12）、**依赖动态 shape 的边界处理**（V11）、
**不适用**（V13，构建期判断不存在）——无一处以“实现简单/照源码写”为由保留。与 dense 的差异：dense 的 V11
是“静态边界”（构建期 `nlast`），本函数因 `total` 动态而升级为“动态 shape 边界”，已按该类别给出理由与
两条已编译形态（主选运行期 clamp / 备选 P6）。

#### 1.6.3 向量化轴与数据布局决策

**轴质量评分（候选逐项）**：

| 评分项 | 说明 | 候选 1（主选） | 候选 2（备选） |
|--------|------|---------------|---------------|
| 整除性 | 向量轴长 vs 向量宽度（16-bit = 128 lane） | `D_v = 128` = 恰好 1 个向量寄存器 ✅ | 同左（每 head 段仍是一个寄存器） |
| 尾 lane 浪费率 | `(ceil(D_v/128)·128 − D_v)/D_v` | `D_v=128` → 0%；`D_v=64` → 100%（一半 lane 是 pad，但每行仍只占 1 寄存器/11 条指令，pad 只浪费 lane 不增加指令数） | 同左 |
| 累加链形态 | 向量维 stride=1、串行维是否阻碍向量化 | 向量维 = `d`（stride 1）✅；行维是独立归约实例，无累加链 | 同左 |
| repack 代价 | 是否需要核内/ host 重排 | **无**（I/O 布局直接可用，只靠 DMA 描述符的段间距；输出 head-major 的基址是运行期乘法，不是重排） | **无**（输入完全连续） |
| UB 容量影响 | 驻留缓冲预算 | `2·R·DW·2 B` = 32 KB（双版本 64 KB）+ 2 KB 暂存 = **66 KB** ✅ | 较差：全 `H` 头布局每行 32 KB（H=128）→ 双缓冲只能 `R = 1` |

**候选矩阵**（全部枚举；弃选给量化理由）：

| # | 布局方案 | 向量化轴 | repack 路径 | 预估收益 / 代价 | 是否采纳 |
|---|---------|---------|------------|-----------------|---------|
| 1 | **I/O 原生布局 + head 固定 tile `[R, D_v]`** | `d`（行内连续，1 寄存器/行） | 无（跨步 DMA：`n` 行 × 256 B，源间距 `H·D_v·2 B`；P1/P5 实测单描述符） | 输入 256 B/段 = 2 个完整 128 B 段；输出 head-major 连续段（`n·4 B`）；UB 66 KB | ✅ **主选** |
| 2 | 全头 tile `[R, H, D_v]`，输入完全连续 | `d`（每 `(r,h)` 段 1 寄存器） | 无 | 输入 1 条连续描述符（无段间距）；但 `H·D_v·2 B = 32 KB/行`（H=128）→ 双缓冲下 `R = 1`，且 head-major 输出变成 `H` 个 `R·4 B` 的散段（R=1 时 4 B/段，写放大 8–32 倍） | ❌（备选，见裁决计划） |
| 3 | D 轴跨核切分（2 核 × `D_v/2`） | `d` | 需跨核中间结果 | 每行 4 B 却要一次跨核合并/原子：原子次数 = N = 4.19M，原子流量 +33.6 MB，且写出前必须跨核同步 | ❌（R3/R4 已量化淘汰） |
| 4 | lane = 行、`D` 串行（垂直扫描） | 无（行内退化为标量） | 无 | 每行 128 次标量乘加：向量化率 1/128，向量单元失效（估算 ≥ 8× 慢于候选 1） | ❌ 弃选 |
| 5 | Cube：`diag(O·Doᵀ)` / ones-GEMM | Cube fractal | 需 Vector→Cube 中转 | FLOPs 64×、每行 1 输出 ≪ N=16 fractal、跨引擎同步 | ❌（R3/R4 已量化淘汰） |
| 6 | host 侧 permute 成 `(H, T_q, D)` 让 `t` 连续 | `d` | 两次全张量 host permute | 额外一次全量读 + 写（2.1 GB 级），远超算子本体；且 host 只允许 stride/shape 视图，禁止改动输入内容 | ❌ 弃选 |
| 7 | 核内转置（lane = 行）后批量归约 | 转置后的 `d` | UB 内 64×128 转置链（`vdintlv/vintlv`） | 省 1 条 `vcadd`/行（11 条中的 9%），却要 ≥2 条/行的转置 + 额外 UB 往返 | ❌ 弃选（API 存在但代价倒挂） |
| 8 | **静态 extent 分段写回**（满 tile 静态 + 残块动态，P6） | `d`（同主选） | 无 | 满 tile 的描述符 count 为常量；但残块段在流水外写同一 buffer → 编译器关闭双缓冲（P6 实测告警） | ❌ 主选不采纳；**登记为结构完整的备选**（见下） |

**布局决策结论**：选定候选 1——**核内布局 = `[R 行, D_v]` 的 UB tile（head 固定），向量化轴 = `d`（行内连续、
`D_v=128` 恰好一个向量寄存器），repack 路径 = 无（跨步 DMA 描述符表达源间距 `H·D_v·2 B`、目的连续/
行 pitch `DW·2 B`）**；输出布局沿用契约的 head-major `[H,T_q]`（最内轴 `t` 连续 4 B，使写回仍是单条压缩
描述符）。该结论已同步落入 §3.3 伪代码（`o_ub[i, 0]` 起 `vld` 整行）与 §6 循环结构（内层向量维 = `d`，
行循环是归约实例维）；`D_v > 128` 时向量化轴仍是 `d`，只是按 `nvec = DW/128` 个寄存器分段，各段部分和在
**构建期展开成一个 `vadd` 表达式**（§3.3 `row_dot`；P3 已编译）。

**实验裁决计划（三件套；判定裕度依赖未实测常数）**：本函数的分歧点落在两个**本型号未实测**的量上——
① 运行期 count 的描述符相对静态 count 是否有额外代价（`CONST-mte2-degradation` 只有 910B2C 的“段/指令”
口径，指令维度 300 ns 未在本型号复核）；② 双缓冲与静态 extent 的取舍（P6 实测会关闭双缓冲）。

- **主选（进 Stage 3）**：候选 1 + 统一运行期 clamp（`n = min(R, total−t0)`）+ `T.Pipelined(it0, it1,
  num_stages=2)` + `annotate_buffer_versions`，`G = 56`（§5.5）。
- **备选（≤1 个，结构完整）**：**P6 静态化形态**——满 tile 段 extent 全静态 `R`（`nfull = (total//R)·heads`
  个 tile，`it` 界仍运行期），残块段每 head 一个 tile（`if bx < heads`、`if n > 0`）、extent 仍动态；buffer 形状
  与 UB 预算（`o_ub`/`do_ub` 各 `(R,DW)`、`d_ub (R,8)`，66 KB）、循环结构（满段
  `T.Pipelined(it0, it1, num_stages=2)` + 残块顺序段）、repack 路径（无，同候选 1）、dtype 路径
  （16-bit 原样搬入 + `vcvt` 提升，不变）、
  分核三要素（`G = 56`，每核 `nfull/G` 个满 tile + 至多 1 个残块，逻辑核数仍为 `ntiles`）全部不变，
  唯一结构差异是**失去 `o_ub/do_ub` 多版本**（P6 告警）与多一条代码路径。
- **裁决计划**：代表 shape = `ds-v3-8x4k`（`T_q=32768, H=128`）、`ds-v3-mixed`（`T_q=32256, H=128`）、
  `kimi-k2-4x8k`（`T_q=32768, H=64`），另加 smoke（`T_q=640, H=8`）作健全性检查；指标 = `msprof` 的
  Task Duration（us，同 Stage 4 口径，A/B 交错各 ≥5 次）；**判定阈值**：备选 latency < 主选 × 0.9 才翻转；
  裁决结果按 shape 分片回写本节（shape 特化工厂允许按 workload 分派主选/备选，不必全局二选一）。
- **参数级候选（同属 Stage 4，不改结构）**：`R ∈ {32, 64, 128}`——`R=128` 把每核 tile 数减半、描述符数减半
  （UB 132 KB，仍在预算内但余量变紧，§4.5）；`R=32` 反之。默认 `R = 64`（承袭 dense 已验证预算）。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式: SIMD**（任务平台 `programming_modes.choices.simd`；`migration_default = simd`，用户未另行指定）。

### 2.2 选型理由

- 算子**不含任何矩阵乘**：每行只有“乘 + `D_v` 元素归约”，Cube 的 fractal 语义（N ≥ 16）与“每行 1 个输出”
  不匹配（§1.6.0 R4 候选 C6 已量化淘汰：FLOPs ×64 + 跨引擎同步），故排除 `Cube` 与 `混合`。
- 归约是本算子的唯一结构性计算：`SIMD` 用 `vcadd` 一条指令完成 64 lane 二叉树归约，结果直接留在寄存器；
  `SIMT`（`T.SimtVF` + `T.reduce_sum` / `alloc_reducer`，即参考表命中的 C7）每行要做一次跨线程 AllReduce
  （barrier + 共享内存），在 N = **4.19M 行**（packed 代表 workload，是 dense 的 8 倍）规模上把发射开销推到
  4–8 ms 量级（§1.6.1 A2 收益栏），是模式选择上的决定性差异；行数越多，该差异越大。
- 逐元素乘与 16→32 提升是纯向量操作（`vmul`/`vcvt`），SIMD 的 128 lane（16-bit）/64 lane（fp32）宽度
  与 `D_v = 128` 恰好对齐（§1.6.3）；packed 的动态行轴不影响这条判断（向量化轴是行内的 `d`）。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_shared`（UB 253952 B/core）：`o_ub[R,DW]`、`do_ub[R,DW]` 16-bit（各双版本）、`d_ub[R,8]` fp32 暂存；不使用 L1/L0 |
| 计算方式 | `with T.SimdVF():` 内逐行 `T.simd.vld → vcvt → vmul → vadd → vcadd → vsts`；无 `T.Parallel`（SimdVF 无线程域） |
| 并行来源 | 核间：一维 `T.Kernel(56)`（静态）；核内：向量宽度（128 lane）+ 行循环；**tile 流是运行期量**（`ntiles`） |
| 同步 | 编译器自动插入 `asc_sync_notify/asc_sync_wait`（§7） |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式为 §1.6.1 的**优化后公式**（A1+A2+A3+A4）。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `it ∈ [ntiles·bx/G, ntiles·(bx+1)/G)`；`h = it % H`，`t0 = (it // H)·R`，`n = min(R, T_q − t0)` | tile 任务映射（block 级索引；`ntiles` 运行期） |
| 2 | `o_ub[r, d] ← o[t0+r, h, d]`，`do_ub[r, d] ← do[t0+r, h, d]`，`r < n`，`d < D_v` | GM→UB 跨步搬入（16-bit 原样），目的端显式 region `o_ub[0:n, 0:D_v]` |
| 3 | `vo = vld(o_ub[i, 0])`，`vd = vld(do_ub[i, 0])` | 取第 `i` 行的整行向量（`D_v = 128` = 1 寄存器） |
| 4 | `o_lo = vcvt(vo, PART_EVEN)`，`o_hi = vcvt(vo, PART_ODD)`；`d_lo/d_hi` 同理 | 16-bit → fp32（精确提升，两半各 64 lane） |
| 5 | `acc = vmul(o_lo, d_lo) + vmul(o_hi, d_hi)` | fp32 乘积（每条 `vmul` 四舍五入到 fp32）+ 两半合并 |
| 6 | `s_i = vcadd(acc)` | fp32 树形归约（64 lane → 广播标量） |
| 7 | `d_ub[i, 0] = s_i` | `vsts` + `PAT_VL1` + `extent=1`（32 B 行间距） |
| 8 | `delta[h, t0 : t0+n] = d_ub[0:n, 0]` | 每 tile 一次压缩 UB→GM 写出（head-major 基址 `4·(h·T_q + t0)`，count = `n`） |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API（`tilelang.ascend.language`） | 参数 | 模式 |
|------|----------|-------------------------------------------|------|------|
| 1 | 核/tile 索引 | `T.Kernel(G)` / `T.Pipelined(it0, it1, num_stages=2)` / `T.serial(R)` | `G`、`R` 构建期常量；`it0/it1` 运行期 | SIMD（核级） |
| 1b | 运行期取值 | `T.ceildiv(total, R)`、`T.min(R, total − t0)`、整数 `//`、`%` | `total = T.dynamic("total_q")` | 运行期标量 |
| 2 | GM→UB 搬入 | `T.copy(o[t0:t0+n, h, :], o_ub[0:n, 0:dim_v])`（`do` 同） | 源为 3-D 区域（固定 `h`），目的**显式 region** `[0:n, 0:D_v]`（`DW > D_v` 时必须）；两端 extent 同为运行期 `n` | SIMD（MTE2） |
| 2b | 双版本缓冲 | `T.annotate_buffer_versions({o_ub: 2, do_ub: 2})` | `num_stages=2` 配套 | 流水 |
| 2c | pad 清零（条件） | `T.clear(o_ub)` **且** `T.clear(do_ub)` | 仅 `DW != D_v` 的构建期分支，循环前各一次；P2 实测覆盖两个版本 | SIMD |
| 3 | 整行装载 | `T.simd.vld(o_ub[i, 0])` | 每行 128×16-bit = 1 个 2048-bit 寄存器 | SIMD |
| 4 | 类型提升 | `T.simd.vcvt(vo, "float32", part=0 / part=1)` | `PART_EVEN` / `PART_ODD` 取两半 64 lane | SIMD |
| 5 | 逐元素乘 + 合并 | `T.simd.vmul(a, b)`、`T.simd.vadd(a, b)` | fp32x64；默认 `PAT_ALL` mask | SIMD |
| 6 | 行归约 | `T.simd.vcadd(acc)` | 64 lane 二叉树，结果广播 | SIMD |
| 6b | `nvec > 1` 的段合并 | 构建期 python helper（§3.3 `row_dot`）返回 `vadd(vadd(p0,p1),…)` **单一表达式** | `nvec = DW/128` 是构建期常量；**不得**写成 IR 循环携带向量累加器 | 构建期展开（P3） |
| 7 | 结果暂存 | `T.simd.vsts(d_ub[i, 0], s, one, extent=1)`，`one = T.simd.pset(32, "PAT_VL1")` | 只写 1 个 lane；行间距 32 B（`d_ub[R,8]`） | SIMD |
| 8 | UB→GM 写出 | `T.copy(d_ub[0:n, 0:1], delta[h, t0:t0+n])` | 单条压缩 DMA：count = `n`（运行期）、元素 4 B、目的连续、源间距 32 B；基址 `h·T_q + t0` | SIMD（MTE3） |

### 3.3 计算伪代码

> **本节的形态即探针 `history_version/design_probe_packed_v2.py` 的 P1/P2/P3/P4/P7 代码**（全部 COMPILE OK）；
> 负向对照 N1（不切目的端）/N2（语句 helper）/N3（静态 write-back extent）见 §3.4。
> 三条硬约束：① 向量值**不得**跨 IR `for` frame 传递（`nvec` 段在构建期展开成一个表达式）；
> ② 目的端**必须**显式切片 `[0:n, 0:D_v]`；③ 写回的 **count 必须是运行期 `n`**（不得用静态 `R`）。

```python
# ---- 构建期常量（由 (heads, dim_v, dtype) 决定；total 是唯一的运行期维度）----
VL    = 128                       # 16-bit 向量宽度（2048 bit / 16）
DW    = ceil(D_v / VL) * VL       # UB 行宽；D_v=128 → 128；D_v=64 → 128（右半 64 列是 pad）；D_v=192 → 192
nvec  = DW // VL                  # 段数（构建期常量，D_v<=128 时恒为 1）
R     = 64                        # 行/tile（UB 预算，§4.5）
G     = 56                        # vector_core_cnt（实查，§5.5 ②）
dtype, heads, dim_v               # 构建期常量；batch 只进工厂缓存键，不参与取址


def row_dot(o_ub, do_ub, i, nvec):
    """构建期展开：nvec 段的部分和合成**一个** vadd 表达式，再唯一一次 vcadd。

    nvec 是 python int → 这里的 `for` 在 trace 期执行，不生成 IR ForFrame；
    返回的是单个 PrimExpr，不存在“跨迭代重绑定的向量变量”。
    """
    part = None
    for v in range(nvec):
        vo = T.simd.vld(o_ub[i, v * VL])
        vd = T.simd.vld(do_ub[i, v * VL])
        cur = T.simd.vadd(
            T.simd.vmul(T.simd.vcvt(vo, "float32", part=0), T.simd.vcvt(vd, "float32", part=0)),
            T.simd.vmul(T.simd.vcvt(vo, "float32", part=1), T.simd.vcvt(vd, "float32", part=1)),
        )
        part = cur if part is None else T.simd.vadd(part, cur)   # 表达式拼接
    return T.simd.vcadd(part)


@tilelang.jit(out_idx=[2], target="ascend")   # delta 是第 3 个参数（out_idx=2）
def kernel():
    total = T.dynamic("total_q")              # packed：唯一的运行期维度

    @T.prim_func
    def main(o: T.Tensor((total, heads, dim_v), dtype),
             do: T.Tensor((total, heads, dim_v), dtype),
             delta: T.Tensor((heads, total), "float32")):     # head-major packed 契约
        ntiles = T.ceildiv(total, R) * heads                  # 运行期逻辑任务数
        with T.Kernel(G) as bx:                               # 静态一维核网格
            o_ub  = T.alloc_shared((R, DW), dtype)
            do_ub = T.alloc_shared((R, DW), dtype)
            d_ub  = T.alloc_shared((R, 8), "float32")         # 32 B 行间距的暂存
            T.annotate_buffer_versions({o_ub: 2, do_ub: 2})
            if DW != D_v:                                     # 构建期：pad 列必须**两侧**都清零
                T.clear(o_ub)                                 #   （否则 0·(Inf/NaN bit 模式) = NaN）
                T.clear(do_ub)
            it0 = (ntiles * bx) // G                          # 均衡划分，恒有 it0 <= it1
            it1 = (ntiles * (bx + 1)) // G
            for it in T.Pipelined(it0, it1, num_stages=2):    # 运行期界 + 双版本
                h  = it % heads                               # 常量模
                t0 = (it // heads) * R
                n  = T.min(R, total - t0)                     # 运行期 clamp（恒 >= 1）
                T.copy(o[t0 : t0 + n, h, :], o_ub[0:n, 0:dim_v])    # 目的端显式 region
                T.copy(do[t0 : t0 + n, h, :], do_ub[0:n, 0:dim_v])
                with T.SimdVF():
                    one = T.simd.pset(32, "PAT_VL1")
                    for i in T.serial(R):                     # 静态行循环（V9：API 缺失，§1.6.2）
                        T.simd.vsts(d_ub[i, 0], row_dot(o_ub, do_ub, i, nvec), one, extent=1)
                T.copy(d_ub[0:n, 0:1], delta[h, t0 : t0 + n])  # 压缩写出，count = n（运行期）
    return main
```

> **注 1（构建期 vs 运行期）**：`DW/nvec/R/G/heads/dim_v/dtype` 是 python 常量，故 `if DW != D_v` 在 trace 期
> 折叠为单一路径；`total/ntiles/it0/it1/h/t0/n` 是运行期表达式。kernel 内**没有**逐元素 shape 谓词。
>
> **注 2（`nvec` 特化，禁运行期循环携带向量值）**：`row_dot` 是工厂层 python 函数，其 `for v in range(nvec)`
> 在 trace 期执行（`nvec` 是 int），只拼接表达式。若把它写进 prim_func 体内，会被提升为 IR `ForFrame` 并对
> 跨迭代重绑定的向量值报 `Immutable variable 'acc' is used outside its defining region!`（dense 探针 F7 复现，
> 本函数 P3 用 helper 形态验证 `nvec=2` 生成 `vadd(vadd(simd_inst::vmul(...` 单表达式）。
>
> **注 3（目的端切片）**：`o_ub` 完整形状是 `(R, DW)`、源区域是 `(n, D_v)`；`D_v < DW` 时整 buffer ↔ 整 buffer
> 元素数不等，平台直接拒绝。本函数 N1 实测原文：
> `Check failed: (analyzer->CanProveEqual(src_total, dst_total)) is false: PlanMTECopy: source and destination
> regions must contain the same number of elements, got src=T.min(64, total_q - it // 4 * 64) * 64 and
> dst=8192`（**注意运行期 extent 出现在报错表达式里**，说明切片对动态 extent 同样必需）。写成
> `o_ub[0:n, 0:D_v]` 后 COMPILE OK（P2：`min(64, total_q−t0), 128, …, 512, 256`——只写 `n` 行 × 128 B，
> 目的行 pitch 仍是 256 B，pad 列保持 `T.clear` 的 0）。
>
> **注 4（动态 extent取代静态尾段）**：`total` 是运行期量 → 不存在 `nlast`。三条描述符的 count 全部是
> `T.min(R, total − t0)`；行循环上界仍是静态 `R`（多算的行落在 `[n, R)`，不写出；`d_ub` 的行间距 32 B 保证
> `vsts` 不越行）。探针 P1/P2/P3/P5 生成的 count 表达式逐字相同（`min(64, (total_q − ((it >> k) * 64)))`）。
>
> **注 5（为何不用静态 extent，N3）**：把写回 extent 写成静态 `R` **也能编译**，且编译器会自行加最后轴
> clamp 使 count 变成 `min(64, max(total_q − t0, 0))`——对 `(heads, total)` 这个二维基址“恰好”正确。这是
> **未文档化**行为：若 `delta` 声明成一维扁平张量，同一 clamp 只停在张量末尾，静态 extent 就会把
> `R − n` 个值写进下一个 head 的行（`total % R ≠ 0` 时），且无任何诊断。故本设计**显式**写 `n`（与编译器
> clamp 同值，但边界不依赖它），并把“静态 extent 会越界多少”量化为 A4 的诊断列（§1.6.1）。
>
> **注 6（分核划分形态）**：主选 `it0 = ntiles·bx // G`（均衡，P1）；备选 `cpt = ceildiv(ntiles, G)`，
> `it0 = T.min(bx·cpt, ntiles)`、`it1 = T.min(it0 + cpt, ntiles)`（P7 已编译；**必须先 `min` 再入循环界**，
> 否则 `ntiles < (G−1)·cpt` 时会出现负 trip count）。两种划分覆盖相同 tile 集、无重叠、无遗漏。
> **整型宽度**：划分表达式是 `ntiles·bx`，上界 = `(T_q/R)·H·(G−1) ≈ 0.86·T_q·H`；
> int32 要求 `ntiles ≤ 2^31/55 ≈ 39.0e6`，即 **`T_q·H ≤ 2.5e9`**（`R=64` 时）。本函数最大 workload
> `T_q·H = 32768·128 = 4.19e6`（`ntiles·bx ≈ 3.6e6`），余量 >500×；`R=128` 时该上界再翻倍。
>
> **注 7（`batch` 与工厂键）**：packed 分支的 grid 与取址都不含 `batch`（源码第 57 行同样如此），但工厂键
> 仍保留 `(batch, heads, dim_v, dense_seqlen=0, dtype)`（与源 `@target_cache` 一致，便于 host 侧沿用调用点）。

### 3.4 API 可行性确认

| API | 证据来源 | 状态 |
|-----|---------|------|
| `T.Kernel(N)`（一维、无 `threads`） | `.agents/platforms/toolchains/ascend/references/programming.md` 第 13 行；`ascend/language/kernel.py` | ✅ P1–P7 全部编译通过 |
| `T.SimdVF()` + `T.simd.*` | 同上第 15 行；`.agents/platforms/toolchains/ascend/knowledge/repro/PL-950-simd-reduce.py` | ✅ 生成 `simd_inst::vlds_norm / vcvt / vmul / vadd / vcadd / vsts_norm` |
| `T.simd.vcvt(..., part=0/1)` | `ascend/language/simd.py`（`PART_EVEN/PART_ODD` widening） | ✅ P1/P2 双 dtype 生成 `PART_EVEN/PART_ODD`（`MODE_ZEROING`） |
| `T.copy`（3-D 区域固定中间索引 → UB，**目的端切片**，**动态 extent**） | P1（`D_v=128,H=8`）`asc_copy_gm2ub_align(dst, src, min(64, total_q−t0), 256, 0,0,0, mode, 2048, 256)`；P2（`D_v=64`）`…, 128, …, 512, 256`；P3（`D_v=192`）`…, 384, …, 1536, 512`；P5（`H=128`）源间距 32768 | ✅ 单条跨步描述符；**目的端必须显式切片**（N1 → `PlanMTECopy … got src=T.min(64, total_q - it // 4 * 64) * 64 and dst=8192`） |
| `T.copy`（`d_ub[0:n, 0:1]` → GM head-major） | P1/P4：`asc_copy_ub2gm_align(&delta[(it>>3)*64 + (it&7)*total_q], d_ub, min(64, total_q−t0), 4, store_mode(4), 4, 32)`；P5：`&delta[(it>>7)*64 + (it&127)*total_q]` | ✅ 单条压缩描述符；目的连续 4 B、源间距 32 B、count 运行期 |
| 运行期 `T.Pipelined(it0, it1)` + `annotate_buffer_versions` | P1/P2/P3/P5/P7：生成 `(it − it0) & 1` 版本偏移（`*8192` half）与成对 `asc_sync_notify/wait` | ✅ 编译通过、双缓冲生效（P6 的静态化形态反而被关闭，见 §6.3） |
| `T.clear`（区域清零） | `tilelang.ascend.language.clear`；P2 生成两段 `for i < 4096 { …4 half… }`，分别覆盖 `o_ub`（offset 0）与 `do_ub`（offset 16384 half），各 **16384 half = 两个版本** | ✅ 实测覆盖全部版本（pad lane 不会在第二个版本里留下未初始化值） |
| `T.simd.pset(32, "PAT_VL1")` / `vsts(..., extent=1)` | `PL-950-simd-reduce.py` 的已验证用法（含 `extent=1`） | ✅ 生成 `vsts_norm(..., 0, one)` |
| 构建期特化 `nvec` 段合并（`row_dot` helper） | P3（`nvec=2`）：生成 `vadd(vadd(simd_inst::vmul(...` 单一表达式 | ✅ COMPILE OK；反例见 dense F7 |
| 运行期 extent 的零长度的处理 | P6 残块段 `(total_q & 63)` 可为 0（`total % R == 0` 时） | ⚠️ 仅作备选：主选下 `n ≥ 1` 恒成立（`t0 < total`），不出现零长度描述符 |
| `T.reduce_sum`（高层二维归约） | **不采纳**：`TRAP-950-simd-shared-reduce` 记录 `T.SimdVF` 内二维 shared→shared 归约的输出布局差异 | ⛔ 弃用（§0.5） |
| `alloc_reducer/reducer_init/update/finalize`（SIMT 路径） | `PL-950-row-primitives.py` 已验证单行 64 元素用法 | ⛔ 弃用（每行一次 AllReduce，§1.6.1 A2） |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| `T.Kernel` 只接受一维网格、核级无 `threads` | **是**（源为二维 + 128 线程） | 一维静态 `T.Kernel(56)` + 核内动态 tile 流；线程域由 `T.SimdVF()` 取代（§0.6 R1） |
| UB 253952 B/core，auto multi-buffer 可能膨胀 | 是 | 66 KB/块（`D_v=64` 68 KB、`D_v=192` 132 KB）；1.7× 膨胀后 112/116/230 KB（§4.5） |
| 无 warp / `__shfl` / SMEM 归约语义 | 是 | 归约改为 `vcadd`（§0.6 R2） |
| 高层二维 `T.reduce_sum` 在 SimdVF 内的布局差异 | 是（若误用） | 明确不用，改显式 SIMD（§3.4） |
| 整 Buffer → 整 Buffer 的拷贝要求元素总数相等（`references/programming.md:37`） | **是**（`D_v < DW`，如 `D_v=64`；且 extent 为运行期） | 目的端写显式 region `o_ub[0:n, 0:D_v]`；N1 复现拒绝、P2 编译通过；`D_v=64` 由 op 级 smoke（`test_mla_varlen_bwd_general_kernel`）覆盖，L0 专门验证 |
| 尾轴对齐：16-bit 向量宽度 128、fp32 64 | 是 | `D_v = 128` 恰好 1 寄存器；一般 `D_v` 按 `DW = ceil(D_v/128)·128` 两侧零填充 |
| 动态 shape | **是（`T_q` 运行期）** | 几何（`heads/dim_v/R/G/DW/nvec`）构建期特化，**行轴动态**：`ntiles/nt0/n` 运行期、tile 内静态（§4.6） |
| 核内循环上界静态化 | 部分（tile 流运行期） | 平台允许运行期界的 `T.Pipelined`（P1/P7 实测）；tile **内部**的 `T.serial(R)` 与描述符 shape 全静态 |
| fast-math / FMA 收缩会改变乘积舍入 | 是 | 不开 `TL_ENABLE_FAST_MATH`（§7.3） |

#### 3.5.2 参考实现差异说明

| 差异项 | 参考实现（GPU，packed 分支） | 本项目（Ascend，packed） | 转换方案 |
|--------|----------------|-----------------|---------|
| Kernel 结构 | `T.Kernel(heads, ceildiv(total,32), threads=128)`（第二维动态） | `T.Kernel(56)` 静态 + 核内动态 tile 流 | §0.6 R1；P1/P7 编译通过 |
| 归约 API | `T.reduce_sum(fragment[D], rows, 1)`（warp 树） | `T.simd.vcadd`（64 lane 树） | §0.6 R2；`PL-950-simd-reduce` + P1 |
| 内存分配 | `alloc_fragment`（寄存器） | `alloc_shared`（UB）+ SIMD 向量寄存器 | §0.6 R3；66 KB 预算 |
| 类型提升 | `T.copy` 隐式 fp16→fp32 fragment | `vcvt` PART_EVEN/PART_ODD | §3.2 步骤 4 |
| 尾块处理 | 运行期 `if row < total` | 运行期 `n = min(R, total−t0)` 约束三处 extent（**无静态尾段**） | §0.6 R4、§6.4、P1 |
| 搬入区域 | 整 tile fragment，无 region 概念 | 目的端显式 region `o_ub[0:n, 0:D_v]`（`DW > D_v` 必须） | §3.3 注 3、P2/N1 |
| 输出写回 | 逐行谓词标量写 `delta[h, row]` | 每 tile 一条压缩 DMA，head-major 基址 `h·T_q + t0` | §0.6 R5、A3/A4、P5 |
| 输出 shape | `(heads, T.dynamic("total_q"))` | 同（`out_idx=[2]` 分配） | P4 |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `.agents/platforms/toolchains/ascend/knowledge/repro/PL-950-simd-reduce.py` | **极高** | 多行、每行 `vld → vcadd/vcmax → vsts(extent=1)`、32 B 行间距暂存、尾 lane mask —— R2/A3 的直接原型 |
| `.agents/platforms/toolchains/ascend/knowledge/repro/PL-950-vector-execution.py` | 高 | `T.Pipelined` + 嵌套 `SimdVF` + `annotate_buffer_versions` + 动态 grid-stride + 尾块静态分支 |
| `examples/mla_dense_bwd/_delta_kernel/DESIGN.md` | **极高** | 同一源码工厂的 dense 分支设计（v2，已检视 + 真机 23/23）；本函数的复用基线（§0.8、§11） |
| `examples/mla_varlen_bwd/_bwd_kernel/DESIGN.md` | 高 | 同一 packed 算子的下游 launch：head-major `lse/delta` 的消费形态（`[1, 256]` 连续段装载）、packed 动态维的处理口径 |
| `examples/paged_cache_write/_paged_cache_write_kernel/_paged_cache_write_kernel.py` | 高 | ascend 向量算子的工程形态：一维 `T.Kernel`、`T.dynamic` 尾块、golden/L0–L2 分层测试组织 |
| `examples/moe_permute_align/_make_scatter_kernel/_make_scatter_kernel.py` | 中高 | 以 56 Vector 核为分核目标（`_NBLK_TARGET = 56`）、构建期静态几何 |
| `.agents/platforms/toolchains/ascend/kernel_template.py` | 中 | 平台 kernel 骨架（`build/golden/check/run_L0..boundary/main` 结构，§8.3 沿用其分层命名） |

#### 3.5.4 编译探针证据（无设备，仅前端 + 编译）

脚本：`examples/mla_varlen_bwd/_delta_kernel/history_version/design_probe_packed_v2.py`（另有
`history_version/probe_dump_tails_v2.py` 打印完整描述符行）。执行（**必须离开仓根**，否则仓内 dev root 的
`tilelang/` 会遮蔽 site-packages 的 0.1.15）：

```bash
source /workspace/tilelang-mlir-ascend-950/.agents/platforms/toolchains/ascend/env-950pr.sh
cd /tmp && ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python \
  /workspace/tilelang-mlir-ascend-950/examples/mla_varlen_bwd/_delta_kernel/history_version/design_probe_packed_v2.py all
```

实测输出（2026-11-23，`ascend-950pr-9579` / TileLang ascend 0.1.15）：

| 探针 | 结构 | 实测结果 |
|------|------|---------|
| **P1 主形态** | `D_v=128, H=8`：静态 56 核 + 动态 tile 流 + 均衡划分 + 动态 extent 双端 + `T.Pipelined` 双版本 + 逐行 `vcadd` + head-major 压缩写回 | **COMPILE OK**，63 行源；`asc_copy_gm2ub_align(…, min(64, total_q−t0), 256, …, 2048, 256)`；`asc_copy_ub2gm_align(&delta[(it>>3)*64 + (it&7)*total_q], …, min(64, total_q−t0), 4, mode(4), 4, 32)` |
| **P2 pad 路径** | `D_v=64 < DW=128`：目的端切片 + 两侧 `T.clear` + 动态 extent | **COMPILE OK**；装载 `…, 128, …, 512, 256`（每行只写 128 B）；`T.clear` 生成两段 `for i<4096`，覆盖 offset 0 / 16384 half，各 16384 half = **两个版本** |
| **P3 `nvec=2`** | `D_v=192`：构建期 `row_dot` 展开 | **COMPILE OK**，61 行源；`vsts_norm(vcadd(vadd(vadd(vmul(vcvt<float>(vlds_norm<half>(…`）单一表达式；装载 `…, 384, …, 1536, 512` |
| **P4 `out_idx=[2]` 分配** | `delta (heads, total)` fp32（第二维动态） | **COMPILE OK**；签名 `main_kernel(__gm__ float* delta, __gm__ half* do_1, __gm__ half* o, int32_t total_q)` |
| **P5 `H=128`** | 完整描述符行，验证 head-major 基址与静态 4 B 步长 | **COMPILE OK**；装载 `&o[(it>>7)*1048576 + (it&127)*128]`、源间距 **32768 B**；写回 `&delta[(it>>7)*64 + (it&127)*total_q]`、count `min(64, total_q−(it>>7)*64)`、目的步长 4 B、源间距 32 B |
| **P6 静态化备选** | 满 tile 段静态 `R` extent + 每 head 一个残块段（动态） | **COMPILE OK**，81 行源；满段写回 `…, 64, 4, …`（静态）；残块段 `…, (total_q & 63), 4, …`；**告警** `did not enable multi-buffering for buffer "o_ub"/"do_ub"`（残块段在流水外写同一 buffer） |
| **P7 备选划分** | `cpt = ceildiv(ntiles, G)` + `it0 = min(bx·cpt, ntiles)` | **COMPILE OK**；循环界 `min(block_idx·ceildiv(ntiles,56), ntiles)` |
| **N1 反例：目的端不切片** | `D_v=64`，`T.copy(src_region, o_ub)` | **COMPILE FAIL**：`Check failed: (analyzer->CanProveEqual(src_total, dst_total)) is false: PlanMTECopy: source and destination regions must contain the same number of elements, got src=T.min(64, total_q - it // 4 * 64) * 64 and dst=8192` |
| **N2 反例：语句 helper** | 把 `vsts` 语句放进模块级 python 函数 | **COMPILE OK 但计算体被丢弃**：源仅 24 行，`vlds_norm/vcvt/vmul/vcadd/vsts_norm` 五个记号**全部缺失**，并告警 `d_ub is read before anything writes it`（`T.Pipelined`/`T.serial` 与语句必须在 prim_func 体内或被 `@T.macro` 承载） |
| **N3 反例：静态 write-back extent** | `m = R`（静态）而 `total` 运行期 | **COMPILE OK** 且 count 被编译器 clamp 成 `min(64, max(total_q−t0, 0))`——对 `(heads,total)` 二维基址“恰好”正确，但属未文档化行为（§3.3 注 5）；本设计不依赖它，改用显式 `n` |

**反例（不接受的设计形态，本机实测）**：

| 形态 | 实测错误（原文） |
|------|-----------------|
| `D_v < DW` 时目的端不切片（含运行期 extent） | `PlanMTECopy: source and destination regions must contain the same number of elements, got src=T.min(64, total_q - it // 4 * 64) * 64 and dst=8192`（N1） |
| 把 `vsts`/循环放进普通 python 函数 | 编译通过但 SIMD 计算体被静默丢弃（N2：5 个 SIMD 记号全缺 + `d_ub` read-before-write 告警） |
| 把 `for v in range(nvec)` 写进 prim_func 体内 | `RuntimeError: Immutable variable 'acc' is used outside its defining region!`（dense F7 复现；本函数用构建期 helper，P3 通过） |
| 写回用静态 `R` extent | 编译通过、无诊断；对一维扁平 `delta` 会跨 head 污染（N3 + A4 诊断列量化：`T=65/H=4` 越界 252 个元素） |
| `T.clear` 只清一侧 | pad lane 未初始化 → `0·Inf = NaN` 污染整行（dense §9.2；本函数 P2 证明两侧清零且覆盖两个版本） |

说明：以上为**编译级**证据，不含设备执行；运行期精度与性能仍待 Stage 3（前置探针清单见 §8.3）。

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `o` | `(T_q, H, D_v)` | `float16 \| bfloat16` | packed THD，第一维 `T_q` **动态**；host 直接传入，无 view |
| `do` | `(T_q, H, D_v)` | 同 `o` | 同上 |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `delta` | `(H, T_q)` | `float32` | **head-major**（`dense_seqlen == 0` 分支）；`out_idx=[2]` 分配，第二维随运行期 `total_q`；与 `_bwd_kernel` 的入参声明逐元素一致，消费端按 `delta[h, qoff:qoff+64]` 一次 256 B 连续段装载 |

> 为什么必须是 `(H, T_q)` 而不是 `(T_q, H)`：消费端 `_bwd_kernel` 以 `[H, T_q]` 声明并按 64 元素连续段取数；
> 若本函数写成 `(T_q, H)`，消费端每次取 64 行 × 4 B（stride `H·4 = 512 B`）→ 段数 ×64、写放大，属布局回退
> （§1.6.3 候选 6 的同族劣化）。

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| `o_ub` | `(R, DW)`，R=64, DW=128 | 输入 dtype（16-bit） | UB（2 版本） | 本 tile 的 `o` 行块；有效列仅 `[0, D_v)`（`D_v=64` 时右半是 pad，由 `T.clear` 保证为 0；`D_v=192` 时编译器把行 pitch 抬到 256 half = 512 B，见 §4.5） |
| `do_ub` | `(R, DW)` | 16-bit | UB（2 版本） | 本 tile 的 `do` 行块（同样两侧清零） |
| `d_ub` | `(R, 8)` | `float32` | UB（1–2 版本，编译器决定） | 行结果暂存（32 B 行间距，供 `vsts` + 压缩 DMA），2 KB（双版本 4 KB） |
| （寄存器）`vo/vd/o_lo/o_hi/d_lo/d_hi/acc/total` | 各 1 向量寄存器 | 16-bit×128 / fp32×64 | SIMD 向量寄存器 | 每行中间量，**不物化到 UB**（A1） |

### 4.4 内存搬运路径

```
GM o ──[T.copy: 跨步 DMA，n 段 × 256 B（D_v=64 时 128 B），源间距 H·D_v·2 B（H=8 时 2048/1024 B，H=128 时 32768/16384 B），
        目的 region [0:n, 0:D_v]，目的行距 DW·2 B]──► UB o_ub（ping-pong 2 版本）
GM do ─[同上]──────────────────────────────────────────────────────────────────► UB do_ub（ping-pong 2 版本）
UB o_ub/do_ub ──[vld，整行 1 寄存器]──► SIMD 向量寄存器 ──[vcvt/vmul/vadd/vcadd]──► d_ub[i,0]（vsts，1 lane）
UB d_ub[0:n,0] ──[T.copy: 压缩 DMA，n 元素 × 4 B，源间距 32 B，目的连续 4 B，基址 4·(h·T_q+t0)]──► GM delta
（n = min(R, T_q−t0) 为运行期量，三条描述符的 count 都是它；无 L1 / L0C / 中间 GM 往返）
```

### 4.5 UB 内存预算

三档几何（`R = 64`）的**实测/核算**占用；`D_v=64` 与 `D_v=192` 的数字取自探针生成码的缓冲偏移
（`history_version/probe_v2_tails.log`）：

| Buffer | Shape | dtype | `D_v=128` | `D_v=64`（pad） | `D_v=192`（`nvec=2`） |
|--------|-------|-------|-----------|------------------|------------------------|
| `o_ub` × 2 版本 | (64, DW) | fp16/bf16 | 16384×2 = 32768 | 32768 | 32768×2 = 65536（行 pitch 256 half） |
| `do_ub` × 2 版本 | (64, DW) | fp16/bf16 | 32768 | 32768 | 65536 |
| `d_ub` | (64, 8) | fp32 | 2048（1 版本） | 4096（2 版本） | 4096（2 版本） |
| **总计** | | | **67584 B = 66 KB** | **69632 B = 68 KB** | **135168 B = 132 KB** |
| 占用率（÷253952） | | | **0.27** | 0.27 | 0.53 |
| ×1.7 膨胀上界（`CONST-capacity-910B2C` 悲观口径） | | | 114893 ✅ | 118374 ✅ | 229786 ✅（余量 9.5%） |
| ×1.10–1.12（该条目实测结构口径） | | | 74342–75706 ✅ | 76595–77988 ✅ | 148685–151388 ✅ |

| 其余几何 | R | `o_ub`+`do_ub`（2 版本） | `d_ub` | 合计 |
|---|---|---|---|---|
| `D_v = 128`（全部 manifest workload） | 64 | 65536 | 2048 | **67584 B（66 KB）** |
| `D_v = 64`（op 级 smoke，`test_mla_varlen_bwd_general_kernel`） | 64 | 65536 | 4096 | **69632 B（68 KB）** |
| Stage 4 参数候选 `R = 128`（`D_v=128`） | 128 | 131072 | 4096 | **135168 B（132 KB）** |

> 膨胀系数来自 `CONST-capacity-910B2C`（910B2C / `tilelang 0.1.2+3a214cde`，**非本型号实测**），仅作上界参考。
> **`D_v=192` 是紧档**：1.7× 悲观口径下余量仅 9.5%，故该几何列入 §9.1 风险，且 manifest / smoke 均不使用
> `D_v=192`（`D_qk=192` 是 query/key 维，value 维恒 128 或 64）。`R ≤ 64` 时预算单调下降。

### 4.6 动态轴定义

| 动态轴 | 声明方式 | 运行时范围 | 影响的构造 |
|---|---|---|---|
| `T_q`（packed token 行数 `Σ_b q_len[b]`） | `total = T.dynamic("total_q")`；`o`/`do` 第一维与 `delta` 第二维都是它 | 任意 ≥ 0 的整数（`T_q = 0` 时 `ntiles = 0`，所有核空循环） | `ntiles`、`it0/it1`、`t0`、`n`（三条描述符 count） |

**构建期常量**：`heads`、`dim_v`、`dtype`、`R`、`G`、`DW`、`nvec`（与源 `@target_cache` 同策略按
`(batch, heads, dim_v, dense_seqlen=0, dtype)` 特化）。**与 dense 的关键差异**：dense 把 `T_q = B·S_q`
静态化为构建期常量（§4.6 的说明），因此能得到静态尾段；本函数**保留 `total` 为动态**（与源码一致：
`MLAVarlenBwdKernel.forward` 不 view，行数由 `cu_seqlens` 累计），代价是尾块只能用运行期 clamp（§0.6 R4）。
若上层希望“一份 kernel 覆盖任意 `T_q`”，本设计已满足（只有 `heads/dim_v/dtype` 变化才需重编译）。

### 4.7 JIT 配置

```python
@functools.lru_cache(maxsize=64)              # 与源 target_cache 同策略：按 (batch, heads, dim_v, 0, dtype)
def _delta_kernel(batch, heads, dim_v, dense_seqlen, dtype):
    ...
    return tilelang.compile(main, target="ascend", out_idx=[2])   # delta 是第 3 参数
```

不设 `pass_configs`（尤其**不设** `TL_ENABLE_FAST_MATH`，理由见 §7.3），不设额外 `compile_flags`。
工厂入口校验：`dtype ∈ {float16, bfloat16}`、`heads/dim_v > 0`、`dim_v % 16 == 0`（平台尾轴约束）；
`dense_seqlen` 非零时**转交 dense 分支的设计**（本文件只覆盖 `dense_seqlen == 0`，两者是不同的编译实例，
由工厂按 `dense_seqlen` 分派）。

---

## 5. Tiling 策略

### 5.1 计算类型

**类型：纯 Vector**（SIMD）。判定依据：算子只含“逐元素乘 + 行内归约 + 数据搬移”，无矩阵乘、无跨行依赖；
Cube 单元不参与（§1.6.0 R4 候选 C6 已淘汰）。

### 5.2 Block 划分

```python
VL    = 128                      # 16-bit 向量宽度
DW    = ceil(D_v/VL) * VL        # UB 行宽：D_v=128 → 128；D_v=64 → 128（pad）；D_v=192 → 192
nvec  = DW // VL                 # 段数（构建期常量）
R     = 64                       # 行/tile（构建期常量；Stage 4 可试 32/128，属参数级）
G     = 56                       # vector_core_cnt（实查，§5.5 ②）
ntiles = ceil(T_q/R) * H         # 逻辑任务数（运行期）
it0 = ntiles * bx // G           # 本核 tile 区间（均衡划分）
it1 = ntiles * (bx+1) // G
# 每 tile: h = it % H, t0 = (it // H) * R, n = min(R, T_q - t0)
```

`block_M = R`（行方向），`block_N = D_v`（value 方向，整行一个向量寄存器，**不切分**）。
**与 dense 的关键差异**：dense 的 `R` 取“`S` 在 `[16,64]` 内的最大因子”，`nsp/tpb` 由 `nfull` 静态整除；
本函数 `T_q` 运行期 → `R` 固定 64（无整除约束），`ntiles` 与划分全部运行期，**tile 内几何全静态**。

| workload（packed manifest / 测试） | T_q | H | D_v | ntiles | 每核 tile 数 | 残块（`T_q % R`） | UB/块 |
|---|---|---|---|---|---|---|---|
| `ds-v3-8x4k`（8×4096） | 32768 | 128 | 128 | 512×128 = **65536** | 1170.3 | 0 | 66 KB |
| `ds-v3-mixed`（512+1024+2048+4096+8192+16384） | 32256 | 128 | 128 | 504×128 = **64512** | 1152.0 | 0 | 66 KB |
| `kimi-k2-4x8k`（4×8192） | 32768 | 64 | 128 | 512×64 = **32768** | 585.1 | 0 | 66 KB |
| manifest smoke（`q_lens [256,384]`） | 640 | 8 | 128 | 10×8 = **80** | 1.43 | 0 | 66 KB |
| **op smoke `test_mla_varlen_bwd_general_kernel`**（`q_lens [256,128]`，`D_v=64`） | 384 | 4 | **64** | 6×4 = **24** | 0.43（24 个核有活） | 0 | 68 KB（pad 路径） |
| 小几何（`q_lens [64,32]`，`D_v=128`） | 96 | 4 | 128 | 2×4 = **8** | 0.14（8 个核有活） | **32**（残块） | 66 KB |

（`ntiles` 与每核 tile 数都是运行期量，表中按上式的静态代入给出；残块只在 `T_q % 64 ≠ 0` 时出现，
如 `T_q=96`、`65`、`200`。）

### 5.3 约束分析

- **对齐约束**：行宽 `D_v·2 B = 256 B` = 2 个完整 128 B DMA 段 ✅；`vld` 整行 = 1 个 2048-bit 向量寄存器
  （16-bit ×128）✅；fp32 中间量 = 64 lane ✅；`d_ub` 行间距 32 B（8 个 fp32）与 `PL-950-simd-reduce`
  的已验证布局一致 ✅；写回目的连续 4 B（`[H,T_q]` 最内轴）✅。
- **UB 容量**：66 KB / 248 KB（§4.5；`D_v=192` 为 132 KB 的紧档）。
- **L0 容量**：不使用 Cube → 不适用。
- **段数/指令开销**：每 tile 2 条装载 + 1 条写出描述符；每行 11 条向量指令（§1.6.0 估算）。
  65536 tile × 3 = 196.6 K 条描述符（`ds-v3-8x4k`），÷56 核 = 3511 条/核（§1.6.0 估算项已量化）。

### 5.4 注意事项（非整除、变长与动态边界）

- **非整除（`T_q % R ≠ 0`）**：**没有**静态尾段（`total` 运行期）——所有 tile 统一用 `n = min(R, T_q−t0)`；
  行循环上界仍是静态 `R`，多算的行不写出（§0.6 R4、§6.4）。`ntiles = ceil(T_q/R)·H` 保证最后一个 tile 的
  `t0 < T_q` ⇒ `n ≥ 1`，不会出现零长度描述符（P1 生成码的 count 恒为正）。
- **变长（多个请求拼接）**：本 launch **与请求边界无关**（`delta[h,t]` 只依赖行 `t`），tile 可以跨请求边界
  ——这不是 bug 而是语义允许（R1 候选 C10 已论证按请求分段是负收益）。`cu_seqlens` 不进入本 kernel。
- **`DW != D_v`（`D_v=64`，op smoke 几何）**：`DW = 128`；构建期分支里 `T.clear(o_ub)` **且**
  `T.clear(do_ub)`，P2 实测覆盖**两个版本**；两条 `T.copy` 的目的端显式切片 `[0:n, 0:D_v]`（元素数相等，
  N1 证明不切片会被拒绝）→ 零项严格不改变和。
- **`D_v > 128`**：`nvec = DW/128` 段，各段部分和在**构建期展开成一个 `vadd` 表达式**（§3.3 `row_dot`，
  P3 COMPILE OK），再做唯一一次 `vcadd`；**不使用**运行期循环携带的向量累加器。
- **`T_q = 0`**：`ntiles = 0` ⇒ 所有核的 `it0 = it1 = 0`（空循环），`delta` 为空张量；A4 的空用例覆盖该路径。

### 5.5 分核策略（物理核数适配）

三要素判定标准与实查要求的权威版本：`.agents/skills/_shared/standards/core-split-strategy.md`（本任务按 §1
三要素与 §2.1 执行）。

**① 逻辑核数计算**

- 逻辑任务数（tile 数）`T_total = H · ceil(T_q / R)`：`ds-v3-8x4k` → 128·512 = **65536**；
  `ds-v3-mixed` → 128·504 = **64512**；`kimi-k2-4x8k` → 64·512 = **32768**；smoke → 8·10 = **80**；
  op smoke（`D_v=64`）→ 4·6 = **24**；小几何 `q_lens [64,32]` → 4·2 = **8**。
- 分核后逻辑核数（= 启动的核数）`G = 56`（构建期常量），每核处理 `ntiles / 56` 个 tile：
  `65536/56 = 1170.3`、`64512/56 = 1152.0`、`32768/56 = 585.1`、`80/56 = 1.43`、`24/56 = 0.43`、`8/56 = 0.14`
  （均衡划分 `ntiles·bx//G` 使各核相差 ≤ 1 个 tile）。

**② 物理核数依据（实查记录）**

| 查询 | 命令 / 来源 | 实际返回值 |
|---|---|---|
| 平台任务上下文（本机实跑） | `python3 .agents/tools/statectl.py platform --task-dir examples/mla_varlen_bwd/_delta_kernel` | `hardware.hardware_id = "ascend-950pr-9579"`；`observed_cores = {"cube": 28, "vector": 56}`；`core_source = "Ascend950PR_9579.ini [SoCInfo] ai_core_cnt=28, cube_core_cnt=28, vector_core_cnt=56; runtime probe rtGetAiCoreCount()=28 on Ascend950PR_9579, 2026-11-23"`；`capacity_kb = {UB: 248, L1: 512, L0C: 256}` |
| 型号配置文件（只读复核，本机实跑） | `grep -n "core_cnt\|ub_size\|l1_size\|l0_c_size" /usr/local/Ascend/cann-9.2.1/x86_64-linux/data/platform_config/Ascend950PR_9579.ini` | 第 20–22 行 `ai_core_cnt=28`、`cube_core_cnt=28`、`vector_core_cnt=56`；第 50/51/58 行 `l0_c_size=262144`、`l1_size=524288`、`ub_size=253952` |
| `torch.npu.get_device_properties` | **本次未执行**：设计期约定只做无卡编译（`ASCEND_RT_VISIBLE_DEVICES=""`），不初始化设备上下文 | 以平台已记录的 runtime probe（第一行 `core_source`）为准，不猜测数值 |

**目标型号：`ascend-950pr-9579`**（dav-3510，28 AI Core / 56 Vector Core，UB 248 KB）。
本算子为纯 Vector kernel，故分核目标取 **56 个 Vector 核**（与同仓
`examples/moe_permute_align/_make_scatter_kernel/_make_scatter_kernel.py` 的 `_NBLK_TARGET = 56` 一致，
也与 dense 版的取值一致）。若硬件实际以 28 个 AI Core 为 block 调度粒度（每核 2 个 AIV 子块），则同一 `G`
仍成立、每 AIC 上跑 2 个 block，结论不变（属未文档化假设，列入 §9.1 R7）。

**③ 规模判定与分核方案**

- **规模判定：极大规模**。逻辑任务数 65536 / 64512 / 32768（代表 workload），是 56 个 Vector 核的
  585–1170 倍；若“一 tile 一 block”直接映射，每 block 只有 64 行 × 11 条向量指令的工作量，块调度开销会
  逐 tile 重复 6.5 万次。按标准 §1 要素③走“**固定启动内核数 = 物理核数、核内分批处理多个逻辑任务**”。
  **量化依据**：按 `CONST-vector-launch-overhead` 的“每 op ~0.5 µs 发射”口径（910B2C 实测，**未在本型号
  复核**），逻辑核超发形态每核需承担 1170 次块启动 ≈ **0.58 ms** 的额外发射开销，占 1717–2101 µs 流量下界的
  28–34%；而 persistent 形态只付一次 ~0.5 µs 的核启动。两者功能等价（同一 tile 集），故取 persistent。
- **分核方案：核内串行（persistent 形态）**。`G = 56` 固定；每核用 `T.Pipelined(it0, it1, num_stages=2)`
  消费自己的 `ntiles/56` 个 tile，配合 `T.annotate_buffer_versions` 双缓冲。`it0/it1` 是运行期表达式
  （`ntiles·bx//G`），tile 内几何全静态；平台允许运行期循环界（P1/P7 实测），核内 `T.serial(R)` 与
  三条描述符 shape 全静态。
- **对齐说明**：`G = 56` 恰为 `vector_core_cnt` 的 1 倍 ⇒ **无波次损失**（dense 的 `grid = 256` 是 4.57 波、
  利用率 91%；本函数因静态核数而天然对齐）。小几何（`ntiles < 56`）时部分核空循环（`it0 == it1`），
  由均衡划分保证不出现负 trip count（§3.3 注 6）。
- **与备选的关系**：备选（§1.6.3 裁决计划）是 **P6 静态 extent 形态**，核数与划分不变（`G = 56`，
  每核 `nfull/56` 个满 tile + 至多 1 个残块），由 Stage 4 A/B 实测裁决，阈值：备选 latency < 主选 × 0.9 才翻转。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| 核网格 | block 级并行（一维、静态 56） | `T.Kernel(G)`，`G = 56` | 每核独占 `ntiles/56` 个 `(h, t0)` tile（persistent 形态） |
| tile 流（逻辑任务） | 核内串行 + 流水 | `T.Pipelined(it0, it1, num_stages=2)`，`it0/it1` **运行期** | 流式消费 tile；`ntiles` 随 `T_q` 变化，界无法静态化 |
| 行（归约实例）方向 | 核内串行（静态界） | `T.serial(R)`（`T.SimdVF` 内） | 每行一次整行向量归约（V9：无跨行批量归约 API，§1.6.2）；上界静态 `R`，多算的行不写出 |
| 行内 `d` 方向 | **向量化** | `T.simd.vld/vcvt/vmul/vadd/vcadd` | `D_v=128` = 1 个向量寄存器；`nvec>1` 时按寄存器分段并在**构建期**展开（§3.3 `row_dot`） |
| 尾块/变长 | 无独立代码段（构建期折叠） | 运行期 `n = min(R, T_q−t0)` 进入三条描述符 | `total` 运行期 ⇒ 无静态尾段；也不引入逐元素掩码（§0.6 R4） |
| 元素级 | 不使用 `T.Parallel` | — | `SimdVF` 无线程域，元素并行由向量 lane 承担（§2.3） |

### 6.2 循环伪代码

见 §3.3（同一份伪代码即调度结构）：`T.Kernel(56)` → 运行期 `it0/it1` → `T.Pipelined(it0, it1, num_stages=2)`
→ `T.SimdVF` 内 `T.serial(R)`。

### 6.3 流水线优化

- `num_stages = 2`，`T.annotate_buffer_versions({o_ub: 2, do_ub: 2})`；`d_ub` 的版本由编译器决定
  （P1 单版本、P2 因 `D_v=64` 走了双版本——两种都正确，写后即读且同一迭代内完成）。
- **实测生成代码（P1/P2）**：`o_ub`/`do_ub` 的 UB 地址出现 `(it − it0) & 1` 版本偏移（P1：`* 8192` half；
  P2：`* 8192` half、`d_ub` 另有 `* 512` float 偏移），并插入成对的
  `asc_sync_notify/asc_sync_wait(PIPE_V ↔ PIPE_MTE2 / PIPE_MTE3, event = step)`（§7.2）。
  与 dense 相同：装载未被提前到上一 tile 计算之前（装载与计算同迭代、以 wait 分隔），故“双缓冲是否真带来
  重叠”属未实测量，已列入 §1.6.3 裁决计划；主选仍保留 `T.Pipelined`（代价为零，且 Stage 4 可在同结构上试
  `num_stages = 3`，UB 66 → 98 KB）。
- **与 dense 的差异（重要）**：dense 的非整除几何会因“尾段在流水外写同一 buffer”而被编译器关闭多版本；
  本函数的主选**没有尾段**（所有 tile 同一代码路径、同一 extent 形态），故**任意 `T_q` 都保持双缓冲**
  （P1/P2/P3/P5/P7 均生成两个版本）。这正是“统一运行期 clamp”相对 P6 静态化的核心优势：
  P6 的残块段在流水外写同一 buffer，实测告警 `did not enable multi-buffering for buffer "o_ub"`（§1.6.3 备选）。

### 6.4 尾块处理（非整除与变长）

| 场景 | 处理 |
|---|---|
| `T_q % R == 0`（manifest 全部 workload） | `ntiles = (T_q/R)·H`，每个 tile `n = R`（`min` 取到 `R`）；末 tile 的 `n` 仍是运行期量，但值恒为 `R` |
| `T_q % R ≠ 0`（如 `T_q = 96 / 200 / 65 / 63 / 1`） | 最后一个 tile 的 `n = T_q − t0 < R`：两条装载只搬 `n` 行（不越界读）、写回只写 `n` 个 fp32（不越界写、不污染相邻 head）；行循环仍跑满 `R` 次，多算的行不写出。**无运行期谓词、无静态尾段、无双路径**（P1/P2/P3/P5 生成 `min(64, total_q − t0)`，A4 以 8 个几何机器验证索引映射与覆盖） |
| `T_q < R`（如 `T_q = 1`、`63`） | `ntiles = 1·H`，每个 tile 的 `n = T_q`；无满 tile 概念（不需要 `if tpb > 0` 之类的构建期分支——本函数没有静态 tile 数） |
| `T_q = 0` | `ntiles = 0` ⇒ `it0 = it1 = 0`，所有核空循环，`delta` 为空（A4 空用例） |
| `D_v < DW`（`D_v=64`） | 循环前两侧 `T.clear`（覆盖两个版本，P2）；装载/写回走 `[0:n, 0:D_v]` 切片 |
| 静态 extent 备选（P6） | 满 tile 段 extent 全静态 `R` + 残块段（每 head 一个、`if bx < heads`、`if n > 0`）动态 extent；**代价**：关闭双缓冲。仅作 §1.6.3 裁决计划的备选 |

---

## 7. 同步策略

### 7.1 同步模式

**模式：自动同步**（编译器插入跨 pipe 的 flag）。本设计**不写**任何手动 `T.set_flag/wait_flag`。

### 7.2 同步点说明（P1/P2 的实测生成代码）

| 位置 | 编译器生成 | 理由 |
|------|-----------|------|
| `T.clear` 之后 | `asc_sync_notify(PIPE_S, PIPE_MTE2, 0)`、`asc_sync_notify(PIPE_S, PIPE_V, 0)` + 两个 `asc_sync_wait` | 清零走标量/搬运 pipe，装载（MTE2）与 `vld`（PIPE_V）必须等它落地——**这是 pad 路径的正确性依赖**（P2 实测） |
| block 起始 | `asc_sync_notify(PIPE_V, PIPE_MTE2, event 0/1)`、`asc_sync_notify(PIPE_MTE3, PIPE_V, 0/1)` | 初始化双版本缓冲的 flag，避免首个 tile 死等 |
| tile 迭代开始 | `asc_sync_wait(PIPE_V, PIPE_MTE2, event = (it − it0) & 1)` | 等该版本 UB 的上一轮读者（Vector）读完，才允许 MTE2 覆盖写 |
| 装载完成后 | `asc_sync_notify(PIPE_MTE2, PIPE_V, step)`；Vector 侧 `asc_sync_wait(PIPE_MTE2, PIPE_V, step)` | 等 DMA 落地后才允许 `vld` |
| 暂存写出前/后 | `asc_sync_wait(PIPE_MTE3, PIPE_V, step)` / `asc_sync_notify(PIPE_V, PIPE_MTE3, step)` | `d_ub`（可能单版本）下一 tile 的 `vsts` 必须等上一次压缩 DMA 读完 |
| 计算完成后 | `asc_sync_notify(PIPE_V, PIPE_MTE2, step)` | 释放该版本 UB 给下一轮装载 |
| block 结束 | 尾部 4 个 `asc_sync_wait`（event 0/1 × 两个 pipe 方向） | 收敛所有未完成的跨 pipe 事件 |

结论：`T.copy`（MTE2/MTE3）与 `T.simd.*`（PIPE_V）之间的依赖由编译器按 buffer 版本自动加锁，设计期无需手工
同步；`d_ub` 的单/双版本差异由编译器决定并已用对应 flag 覆盖（P1 单版本、P2 双版本均生成成对 flag）。

### 7.3 pass_configs 配置

```python
pass_configs = {}        # 不开 TL_ENABLE_FAST_MATH
compile_flags = []       # 不额外传 -O3/-DENABLE_BF16（ascend target 的默认后端已处理 dtype）
```

**为什么关掉 fast-math**：源算子开了 `TL_ENABLE_FAST_MATH: True`，但本算子没有任何超越函数（无 `exp/log/sqrt`），
fast-math 唯一可能的副作用是把 `vmul + vadd` 收缩成 FMA 或做其它重排——那会改变乘积的舍入位置（乘积不再先
舍入到 fp32），使 A1 的“逐位等价”结论失效。因此显式不开启，保持“乘积四舍五入到 fp32 → 再累加”的语义
（`verify_equiv.py` 的 A1 用例正是按此语义构造）。

---

## 8. 验证方案

### 8.1 Golden 函数

> 依据 **§0.1 语义**（不是 §0.6 的 NPU 算法）：与源 docstring
> “`delta[.., h, row] = sum_d o[row, h, d] * do[row, h, d]` in float32” 逐字对应，朴素实现、无分块、无向量化。

```python
def golden_delta(o, do, chunk_rows=256):
    """delta[h, t] = sum_d o[t, h, d] * do[t, h, d], float32, packed head-major.

    o, do: (T_q, H, D_v) 的 fp16/bf16 packed 张量（就是 kernel 的入参，无 view）。
    返回 (H, T_q) 的 float32：转 fp32 后逐元素乘，沿最后一维求和，再把 (T_q, H)
    转置成 (H, T_q)——不复制任何分块/归约顺序/向量化细节。
    chunk_rows 只限制 fp32 临时量（大 T_q 的内存），不改变数值路径。
    """
    total, heads, dim_v = o.shape
    out = torch.empty((heads, total), dtype=torch.float32)
    for t0 in range(0, total, chunk_rows):
        t1 = min(t0 + chunk_rows, total)
        prod = o[t0:t1].float() * do[t0:t1].float()            # (t, H, D_v) fp32 乘
        out[:, t0:t1] = prod.sum(dim=-1).transpose(0, 1)        # (H, t) fp32
    return out.contiguous()
```

（设计期 `verify_equiv.py` 另用 fp64 精确参照来评估分组误差与写回映射；那属设计侧证据，不替代本节 golden。
golden 与 §0.6 的 NPU 算法无共享假设：不做 UB 分块、不模拟 `vcadd` 分组、不模拟描述符/暂存缓冲。）

### 8.2 精度标准

| dtype（输入） | 输出 dtype | atol | rtol | 依据 |
|---|---|---|---|---|
| float16 | float32 | 1e-4 | 1e-4 | `delta` 本身是 fp32 输出，两侧都在 fp32 域累加（模板 fp32 标准） |
| bfloat16 | float32 | 1e-4 | 1e-4 | 同上；bf16 提升到 fp32 无损，误差只来自累加分组 |

补充：本算子作为 `MLAVarlenBwdOp` 的一环，最终 dq/dk/dv 的容差沿用
`examples/TileOPs/tests/ops/test_mla_varlen_bwd.py` 的 `_tolerance(dtype)`。`delta` 若在 fp32 下与 golden 的
偏差 ≤ 1e-4，经 `ds = p·(dp − δ)` 传递后不会成为该容差的瓶颈（A2 实测：树形归约比顺序累加更接近真值）。

### 8.3 L0 门槛测试计划

| 层级 | 形状（`q_lens` → `T_q`） | dtype | 覆盖的设计点 | 目的 |
|---|---|---|---|---|
| **L0** | `q_lens [256, 384]` → `T_q=640`，`H=8`，`D_v=128`（`ntiles=80`） | fp16 + bf16 | 主路径：动态 tile 流、双缓冲、head-major 压缩写回（§3.3） | 最小可运行：验证 `vld/vcvt/vmul/vadd/vcadd/vsts` 链与 `[H,T_q]` 写回；两 dtype 必须都过 |
| **L0** | **`q_lens [256, 128]` → `T_q=384`，`H=4`，`D_v=64`**（op 自身 `test_mla_varlen_bwd_general_kernel` 同几何；`DW=128`、`ntiles=24`） | **bf16**（op smoke 实际 dtype）+ fp16 | **pad 路径**：目的端切片 `[0:n,0:64]`、两侧 `T.clear`（覆盖两版本）、pad 列零项不污染和 | 保证 Stage 5 的单算子报告不会在这条路径上首次暴露问题 |
| **L0** | `q_lens [1]` → `T_q=1`，`H=1`，`D_v=128`（`ntiles=1`、`n=1`） | fp16 | 极小几何：`n < R` 的单行 tile、无满 tile | 最小边界：验证无越界读、无空循环问题 |
| **L0** | 契约用例（无设备）：`dense_seqlen=0` 接受；`float32` 输入拒绝；`heads=0` 拒绝；`dim_v=48`（非 16 倍数）拒绝；同几何二次调用返回同一编译实例 | — | 工厂契约与缓存键 | 与源 `target_cache` 语义对齐 |
| L1 | `q_lens [512, 1024, 2048]` → `T_q=3584`，`H=8`，`D_v=128`（多请求拼接，`ntiles=448`） | fp16 + bf16 | 跨请求边界（tile 可跨请求）、`ntiles>56` 的核内多 tile | 代表性小规模 |
| L1 | `q_lens [4096]`，`H=8`，`D_v=128` | bf16 | 单请求长序列、`ntiles=512` | 代表性小规模 |
| L2 | `ds-v3-8x4k`：8×4096 → `T_q=32768`，`H=128`，`D_v=128` | bf16 + fp16 | manifest 主力 workload，`ntiles=65536`、每核 1170 tile、源间距 32768 B | 流量主导口径的实测对账（§1.6.0 估算） |
| L2 | **`ds-v3-mixed`**：512+1024+2048+4096+8192+16384 → `T_q=32256`，`H=128` | bf16 | 变长拼接（6 个不同 `q_len`）、`ntiles=64512` | 变长语义（本 launch 与请求边界无关）的压力点 |
| L2 | `kimi-k2-4x8k`：4×8192 → `T_q=32768`，`H=64` | bf16 | 少头数（H=64）、`ntiles=32768` | 少头数 / 长序列 |
| Boundary | `T_q=65`（`q_lens [64,1]`，`H=8`）、`T_q=200`（`H=8`，`n=8`）、`T_q=96`（`H=4`，`n=32`）、`T_q=63`（`H=128`，`n=63`） | bf16 | **运行期 clamp 的数值闸门**（A4 的对应用例）；尾 tile 只写 `n` 个 fp32 | 越界写会污染相邻 head 或越出张量，必须以此闸门拦住 |
| Boundary | 全零输入（`delta` 应全 0）、含 `inf`/`NaN` 的输入行（与 golden 同为非有限，`equal_nan=True`） | fp16 | 非有限传播与零行（含 `inf·0 → nan` 落在残块行上） | 与源语义一致 |
| Boundary | **越界写的间接探测器**：先跑一个 `T_q=65` 的残块用例，紧接着跑一个全零输入用例并断言输出全零（同一进程、torch 缓存分配器可能复用刚释放的块） | bf16 | 若描述符 count 越界，脏写可能落进后续用例的分配块 | 补充手段（`out_idx` 分配的 `(H,T_q)` 无法直接加哨兵），不能替代“显式 `n`”的设计约束 |

判定：`torch.testing.assert_close(kernel(o, do), golden_delta(o, do), atol=1e-4, rtol=1e-4, equal_nan=True)`；
L0 必须两 dtype 全过才允许进入 L1/L2；`D_v=64` 的 L0 用例是 pad 路径的**唯一数值证据**（编译期只能证明描述符
合法，不能证明 pad 列为 0）；残块用例（`T_q % 64 ≠ 0`）是运行期 clamp 的**唯一数值证据**（编译期只能证明
count 表达式存在）。

---

## 9. 风险点与注意事项

### 9.1 已知约束与风险

| # | 风险 | 影响 | 依据 / 缓解 |
|---|------|------|------------|
| R1 | 跨步描述符（`n` 段 × 256 B、源间距 `H·D_v·2 B`，H=128 时为 32768 B——该跨度落在常规 DMA stride 字段范围内属未文档化假设）的 MTE2 有效带宽低于连续描述符 | 流量项估算偏乐观 | 探针已确认是**单条**描述符（不是 `n` 条 1-D 拷贝）；实际带宽差为 910B2C 常数未覆盖的量 → §1.6.3 裁决计划；Stage 4 以实测对账 §1.6.0 的估算下界 |
| R2 | **运行期 count 的描述符**可能比静态 count 多一次标量取参/分支（未文档化） | 若每条描述符多 ~300 ns（910B2C 指令维度口径未在本型号复核），196.6 K 条 ÷56 核 ≈ 1.05 ms 的串行开销 | 双缓冲下应与传输重叠；P6 静态化形态是已编译的备选（§1.6.3 裁决计划，阈值 0.9×） |
| R3 | 向量发射项可能高于估算（`vcvt`/`vcadd` 多周期） | 若发射项 > 流量项则变成向量受限（估算上界 0.8–1.0 ms vs 1717–2101 µs 流量项，仍有余量） | §1.6.0 估算已给上下界；Stage 4 若发现向量受限：`num_stages` 加深、或改用解包装载（`simd.py` 的 `_VLD_POSTUPDATE_DISTS` 列有 `UNPK_B16`/`UNPK_B32`，但仅见于 `post_inc` 形态，语义未验证，不在本设计内） |
| R4 | **int32 溢出**：划分表达式 `ntiles·bx`（`≈ 0.86·T_q·H`）在 `T_q·H > 2.5e9` 时超出 int32 | 划分表达式回绕 → 错误的 tile 分配 | 本函数最大 workload `T_q·H = 4.19e6`（余量 >500×）；若上游出现 `T_q·H > 2.5e9`，需把划分表达式改为 int64，或改 `cpt` 形态并同样用 int64（P7 的 `bx·cpt` 量级相同，受同一约束） |
| R5 | 写回 extent 的**平台 clamp 语义未文档化**（N3：静态 `R` 也被 clamp 成 `min(64, max(total_q−t0,0))`） | 若依赖它，`delta` 改成一维扁平张量时静态 extent 会跨 head 污染 | 本设计**显式**写 `n`（与 clamp 同值，不依赖它）；A4 的 8 个几何 + 诊断列量化了静态 extent 的越界量（`T=65/H=4` → 252） |
| R6 | `d_ub` 的版本数由编译器决定（P1 单版本、P2 双版本） | UB 预算 ±2 KB；同步 flag 数不同 | 预算按双版本上界 4 KB 计（§4.5）；P2 生成码显示两种形态的 flag 都成对出现（§7.2） |
| R7 | 分核目标“56 Vector 核”假设 block 调度粒度为 AIV | 若为 28 AIC 粒度，则每 AIC 跑 2 个 block（波数 ×2），不影响正确性但性能结论需重估 | §5.5 ② 已标注为未文档化假设；同仓 `moe_permute_align` 与 dense 版采用同一取值 |
| R8 | **`D_v=192` 的 UB 紧档**：编译器把行 pitch 抬到 256 half（512 B），实测占用 132 KB；1.7× 悲观膨胀 = 230 KB（余量 9.5%） | 该几何有编译期容量风险 | manifest / smoke 均不使用 `D_v=192`（value 维恒 64/128；192 是 query/key 维）；若 Stage 3/5 出现该几何，可降 `R` 到 32（预算减半）——属参数级 |
| R9 | `T_q < 56` 的小几何使部分核空循环（`it0 == it1`） | 波次利用率低（性能，非正确性） | 均衡划分保证无负 trip count（§3.3 注 6）；`T_q=0` 时全空（A4 覆盖） |
| R10 | L2 驻留使 bench 读数优于 HBM 口径 | 估算与实测偏差 >2× 时误判“设计失准” | `CONST-mte2-degradation` 的 L2 口径已记录该现象；Stage 4 对账时先看 `msprof` 的读带宽口径再归因 |
| R11 | 抵消行（`Σabs(p)` 大而 `abs(Σp)` 小）的 fp32 累加绝对误差 ~1e-6·`Σabs(p)` | 该行的 `delta` 相对误差大 | 源算子同 dtype 语义、同样存在（§1.6.1 注）；不属于迁移引入的偏差 |
| R12 | 行循环跑满静态 `R`（末 tile 的 `[n, R)` 行算的是陈旧数据，可能含 `inf/NaN`） | 结果不写出（count = `n`），无数值影响；但会多跑 `R−n` 行向量指令 | 属“静态循环界”换“无分支”的取舍；`R−n < 64` 行/head，相对 `ntiles·R` 的总行数可忽略（残块只在 `T_q % R ≠ 0` 时出现）。若 Stage 4 认为必要，可把行循环改成运行期 `T.serial(n)`（P6 残块段已实测该形态可编译） |
| R13 | 依赖 `T.clear` 覆盖**全部缓冲版本**（P2 实测成立） | 若编译器改成只清一个版本，pad lane 的 `0·Inf → NaN` 会污染双缓冲的第二个版本 | P2 生成码（`offset 0` 与 `offset 16384 half` 各 16384 half = 两版本）为当前工具链证据；L0 的 `D_v=64` 用例是数值闸门；退路：改用 `T.simd.update_mask(D_v, 32)` 只算有效 lane |

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|---------|
| `vsts` 写到相邻行 | `d_ub` 行间距 < 32 B | 行结果互相覆盖 | 固定 `d_ub[R, 8]`（32 B 间距），不要改成 `d_ub[R]` |
| **只对 `o_ub` 做 `T.clear`** 而 `D_v < DW` | `D_v=64`（op smoke 几何） | 另一侧的 pad lane 是未初始化 UB，`0 · (Inf/NaN bit 模式) = NaN` → 整行 `delta` 变 NaN | **两侧都清零**：`T.clear(o_ub)` 且 `T.clear(do_ub)`（§3.2、P2）；或用 `T.simd.update_mask(D_v, 32)` 只算有效 lane |
| **目的端不切片**的整 tile 搬入（含运行期 extent） | `D_v < DW`（如 `D_v=64`）且写 `T.copy(src_region, o_ub)` | 前端直接拒绝：`PlanMTECopy: ... got src=T.min(64, total_q - it // 4 * 64) * 64 and dst=8192`（N1） | 目的端写显式 region `o_ub[0:n, 0:D_v]`（`references/programming.md:37`） |
| **把向量值当循环携带量** | prim_func 体内写 `for v in range(nvec)` 并在循环内 `acc = T.simd.vadd(acc, …)` | `RuntimeError: Immutable variable 'acc' is used outside its defining region!`（dense F7） | `nvec` 段在**构建期**用 python helper 拼成一个表达式（§3.3 `row_dot`，P3） |
| **把 `vsts`/循环放进普通 python 函数** | 想“复用”计算体 | **编译通过但计算体被静默丢弃**（N2：SIMD 记号全缺、`d_ub` read-before-write 告警），运行期输出全 0/垃圾 | 语句必须写在 prim_func 体内，或放到 `@T.macro` 承载（本设计只有一个表达式 helper） |
| **写回用静态 `R` extent** | `T_q % R ≠ 0` 时 `T.copy(d_ub[0:R,0:1], delta[h, t0:t0+R])` | 编译通过、无诊断；靠未文档化的最后轴 clamp “恰好”正确，改成一维扁平输出即跨 head 污染 | 显式 `n = min(R, T_q−t0)`（§3.3 注 5、N3、A4） |
| **负 trip count** | `cpt` 划分且不 clamp：`it0 = bx·cpt` 可 > `ntiles` | 空/负循环在流水 prologue 下行为未定义 | 主选用均衡划分（恒 `it0 ≤ it1`）；备选必须先 `it0 = min(bx·cpt, ntiles)`（P7） |
| 用 `T.reduce_sum` 做行归约 | 误按 GPU 习惯 | `TRAP-950-simd-shared-reduce` 的输出布局错位 | 保持 `vcadd` 显式归约 |
| 打开 fast-math | 复制源算子 `pass_configs` | FMA 收缩改变乘积舍入，A1 结论失效 | §7.3：不设 `pass_configs` |
| `T.Kernel` 传二维网格或 `threads=` | 照抄 GPU 源码 | 前端直接拒绝 | §0.5：一维静止网格 + `T.SimdVF` |
| 把 `delta` 写成 `(T_q, H)` | 误以为 packed 要 token-major | 消费端退化为 64 段 × 4 B（stride 512 B） | 保持 `(H, T_q)`（§4.2） |
| 在 kernel 内引入 `cu_seqlens` | 以为“变长就要按请求切” | 多一路运行期解码 + 请求边界分支，收益为 0 | 本 launch 与请求边界无关（§1.6.0 C10、§5.4） |

### 9.3 特殊场景处理

- 极小 `T_q`（`1`、`63`、`96`）：`ntiles = H·1` 或 `H·2`，tile 少而 `n` 小；无特殊分支（A4 覆盖 `T=1/63`）。
- `T_q = 0`：`ntiles = 0`，全部核空循环，输出空张量（A4 空用例）。
- 残块（`T_q % R ≠ 0`）：末 tile 的 `n < R`，三条描述符 count 同时收窄；行循环仍跑满 `R`（R12）。
- 极大 `T_q`（如 `10^6` 级）：`ntiles` 增长，每核 tile 数线性增长；受 R4 的 int32 约束（`T_q·H ≤ 2.5e9`）。
- 输入全零：`delta` 全零（乘法与加法均保零）——L0/boundary 有专门用例。
- 非有限输入：与 golden 一致地传播（`equal_nan=True` 比较），含残块行。
- `D_v=64`（pad）：两侧清零 + 切片；`D_v=192`（`nvec=2`）：构建期段合并 + 容量紧档（R8）。

---

## 10. 交付清单

### 10.1 目录结构

```
examples/mla_varlen_bwd/_delta_kernel/
├── DESIGN.md                          # 本文（v0，首次设计）
├── verify_equiv.py                    # D-1 等价性机器验证（A1/A2/A2b/A3/A4；结果见 §1.6.1）
├── RETROSPECTIVE.md                   # Stage 1 复盘（skill Phase 8）
├── history_version/
│   ├── design_probe_packed.py         # 上一轮（中断前）的探针 v1，留档
│   ├── design_probe_packed_v2.py      # D-3 设计期编译探针 v2（P1–P7 + N1–N3）
│   ├── probe_dump_tails_v2.py         # 完整描述符行打印
│   ├── probe_v2_tails.log             # P1/P2/P3 完整描述符实测输出
│   ├── probe_v2_p5_p6_p7.log          # P5/P6/P7 实测输出
│   └── probe_v2_neg.log               # N1/N2/N3 负向对照实测输出
└── _delta_kernel.py                   # 待 Stage 3 生成（kernel + golden + L0/L1/L2/boundary）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | ✅ 已完成（v0 / first_design） | 本设计文档（§0 迁移决策 + 复用/差异分区 + §1–§11） |
| `verify_equiv.py` | ✅ 已完成并执行 | A1/A2/A2b/A3 × 12 用例 + A4 × 8 几何，全 `EQUIV_PASS`（§1.6.1） |
| `history_version/design_probe_packed_v2.py` | ✅ 已完成并执行 | P1–P7 COMPILE OK；N1 失败、N2 静默丢弃、N3 未文档化 clamp（§3.5.4） |
| `history_version/probe_v2_*.log` | ✅ 已留存 | 生成码/描述符原文（供 Stage 2 复核） |
| `RETROSPECTIVE.md` | ✅ 已完成 | Stage 1 复盘（skill Phase 8） |
| `_delta_kernel.py` | ⬜ 待实现 | Stage 3：kernel（§3.3 伪代码）+ `golden_delta`（§8.1）+ 分层测试（§8.3） |

### 10.3 命名规范

- 项目目录：`examples/mla_varlen_bwd/`（与 `MLAVarlenBwdOp` 的三个 launch 子目录并列）。
- 算子目录：`_delta_kernel`（与源函数名一致，便于 Stage 5 集成时定位 `_compile` 里的 `prep`）。
- 实现文件：`_delta_kernel.py`。

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md v0）+ 算法调研/优化分析 + 等价性验证 + 编译探针（P1–P7、N1–N3）
2. ⬜ `golden_delta`（§8.1）
3. ⬜ kernel 实现（§3.3：构建期特化 `nvec`、目的端切片、运行期 clamp 三处 extent、head-major 写回；`R=64`、`G=56`）+ L0 两 dtype（含 `D_v=64` pad 用例与 `T_q=1`）
4. ⬜ L1/L2/Boundary 分层（§8.3），随后交 Stage 2 检视与 Stage 3 精度闸门

---

## 11. 与 dense 版本的对照（复用证据索引与差异落点）

### 11.1 复用项 → 证据索引

| 复用项 | 索引 | 本函数的落点 |
|--------|------|-------------|
| 数学内容与 dtype 语义（fp32 乘 + fp32 累加） | `examples/mla_dense_bwd/_delta_kernel/DESIGN.md` §0.1 | §0.1、§1.3、§8.1（golden 独立重写） |
| SIMD 归约链（`vld→vcvt→vmul→vadd→vcadd→vsts`） | dense §0.6 R2、§3.4；`PL-950-simd-reduce.py` | §0.6 R2、§3.2 步骤 3–7 |
| 一维核 + `R` 行 UB tile + 双缓冲 | dense §0.6 R1、§5.2、§6.3 | §0.6 R1、§5.2、§6.3 |
| 压缩批量写出 + `d_ub[R,8]` | dense §1.6.1 A3；探针 P1 `...,64,4,...,4,32` | §1.6.1 A3、§3.2 步骤 8 |
| 构建期 `nvec` 展开（`row_dot`） | dense §3.3 注 2；F5/F6 通过、F7 失败 | §3.3 `row_dot`、P3 |
| 目的端切片 + 两侧 `T.clear` | dense §3.3 注 3、§9.2；F2/F3 | §3.3 注 3、§9.2、P2/N1 |
| A1/A2/A2b/A3 的等价结论 | dense §1.6.1（12/12 PASS） | §1.6.1（同用例重跑，数字一致） |
| UB 预算口径（66 KB / 0.27） | dense §4.5 | §4.5（并新增 `D_v=64` 68 KB、`D_v=192` 132 KB 两档实测） |
| 编程模式 `simd` | dense §2 | §2（结论与理由相同，补 packed 行数 8× 的量化） |
| 不启用 fast-math | dense §7.3 | §7.3 |

### 11.2 差异项 → 本文落点

| 差异 | 主论证 | 证据 | 派生章节 |
|------|--------|------|---------|
| ① 动态行轴 → 动态 tile 流 | §0.6 R1 | P1/P7（均衡划分、`cpt` 备选） | §1.4 步骤 1、§5.2、§5.5 |
| ② 无静态尾段 → 运行期 clamp | §0.6 R4 | P1/P2/P3/P5（`min(64, total_q−t0)`）、P6（静态化备选）、N3（未文档化 clamp） | §3.3 注 4/注 5、§6.4、§8.3 |
| ③ head-major 写回取址 | §0.6 R5 | P5（`delta[(it>>7)*64 + (it&127)*total_q]`）、A4（8 几何逐位相同 + 越界量化） | §1.6.1 A4、§3.2 步骤 8、§4.2 |
| ④ 输出 shape 动态 | §0.2、§4.2 | P4（`out_idx=[2]`） | §4.6 |
| ⑤ `D_v=64` pad × 动态 extent | §3.3 注 3 | P2（切片 128 B / pitch 256 B、clear 覆盖两版本）、N1 | §4.5、§8.3 |

### 11.3 本版未复用（并说明为何）的 dense 结论

| dense 结论 | 本函数处置 | 理由 |
|-----------|-----------|------|
| 「尾块用构建期 `S % R` 分支 + 静态 `nlast`（三处 extent 全静态）」 | **不成立**，改运行期 clamp | `total` 运行期 ⇒ `nlast` 不存在（§0.6 R4） |
| 「尾块几何取 `nsp = 1`、`grid = B·H`」 | **不适用** | 本函数无 `(b,h)` 网格，核数固定 56（§5.5 ③） |
| 「`R` 取 `S` 在 `[16,64]` 内的最大因子」 | **不适用**（改固定 64） | `T_q` 运行期 ⇒ 无整除约束；`R` 只由 UB 预算决定 |
| 「`T_q`/`B` 静态化（按 shape 特化编译）」 | **不采用**（保留 `total` 动态） | 与源 packed 分支一致；`MLAVarlenBwdKernel.forward` 不 view（§4.6） |
| 「非整除几何会被编译器关闭多版本」 | **不适用** | 主选无尾段 ⇒ 任意 `T_q` 都保持双缓冲（§6.3），这正是主选相对 P6 的优势 |
