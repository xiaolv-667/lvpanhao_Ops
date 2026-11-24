# _delta_kernel 算子设计文档

> **迁移任务（harness 子模式）**。源算子：`/workspace/TileOPs/src/tileops/_kernels/attention/mla/bwd.py`
> 第 38–77 行的 `_delta_kernel`（GPU TileLang，CUDA target）。目标平台：ascend-950pr-9579 / `ascend`
> 工具链（TileLang 0.1.15 + CANN 9.2.1，`target="ascend"`，`import tilelang.ascend.language as T`）。
>
> **本文只设计 `MLADenseBwdOp` 三个 launch 的第一个**：`delta = rowsum(o * do)`。第二个 launch
> （`_bwd_kernel`）与第三个（`_dq_land_kernel`）是同批次另两个设计任务；本文只在 §0.1 说明接口分工，
> 并以 `delta` 的**消费端契约**（`_bwd_kernel` 第 148/217 行：`delta[b, h, s]` float32、dense 布局
> `[B, H, S_q]`）为不可改动的输出契约。
>
> **版本 v2（revision 1，2026-11-23）**。本版逐项闭环 `REVIEW.md`（v1 检视）的 **3 个阻塞项 + 7 条建议**
> （调度单点名「顺手」的建议 1–4 全部采纳，另 5–7 亦一并修正）；
> 算法主线（单遍 fp32 行点积、一维核 + `R` 行 UB tile + 双缓冲、逐行 SIMD `vcadd`、压缩批量写出）、
> A1/A2/A3 三项等价优化、编程模式 `simd`、§0 迁移决策、§1.6 算法调研/优化分析、§5 分核三要素与
> §8 golden 独立性**全部保持不变**。逐项闭环见下方「修订说明」。
>
> 设计期证据（本机编译，**无卡**：`ASCEND_RT_VISIBLE_DEVICES=""`，从不 launch）：
> - 平台配置：`python3 .agents/tools/statectl.py platform --task-dir examples/mla_dense_bwd/_delta_kernel`
>   → hardware_id `ascend-950pr-9579`，UB 253952 B / L1 524288 B / L0C 262144 B，`observed_cores`
>   cube 28 / vector 56，`programming_modes.migration_default = simd`；`cost_measurements` 为空。
> - 编译探针 v1（`history_version/design_probe_compile.py`）：P1/P2/P4 全部 COMPILE OK，生成代码见 §3.4/§7.2。
> - 编译探针 v2（本版新增，**三个阻塞项的闭环证据**，`history_version/design_probe_compile_v2.py`）：
>   F1（`nvec=1`，D_v=128，持久化 + 双版本）**COMPILE OK**；F2（D_v=64 < DW=128，目的端切片 + 两侧清零）
>   **COMPILE OK**；F4（尾块 S=65、`nlast=1` 全静态）**COMPILE OK**；F5（D_v=256、`nvec=2` 内联展开）
>   **COMPILE OK**；F6（D_v=192、`nvec=2` 经构建期 helper 展开）**COMPILE OK**；F8（`S=1`：`tpb=0` 跳过满 tile
>   流水段、只有尾段）**COMPILE OK**；
>   F3（D_v=64 目的端**不**切片）**COMPILE FAIL**、F7（v1 §3.3 的 `for v in range(nvec)`）**COMPILE FAIL**
>   —— 后两者是 v1 缺陷的复现性证据（负向对照）。F7 会使工具链中止进程，需单独跑 `… f7`；
>   复跑全部正向项：`cd examples/mla_dense_bwd/_delta_kernel && source ../../../.agents/platforms/toolchains/ascend/env-950pr.sh
>   && ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python history_version/design_probe_compile_v2.py all`。
> - 等价性机器验证（`verify_equiv.py`，torch CPU + fp64 参照）：4 行（A1/A2/A2b/A3）全 `EQUIV_PASS`，
>   A2 已改为**实现分组**并附 A2b（v1 引用的分组）作对照，结果表见 §1.6.1。

## 修订说明（v1 → v2，对应 REVIEW.md 的 3 阻塞 + 7 建议）

> 本节是 revision 模式要求的「相对上一版的关键调整」与「为何不会再犯同一错误」。
> 每一项都给出「v1 问题 → v2 修正 → 编译/数值证据 → 防复发落点」。

| # | v1 的问题（REVIEW.md） | v2 的修正 | 证据 | 为何不会再犯 |
|---|---|---|---|---|
| **阻塞 1** | §3.3 写 `for v in range(nvec)` + `acc` 重绑定：eager builder 把 `range()` 提升为 IR `ForFrame`，跨迭代重绑定的向量值报 `Immutable variable 'acc' is used outside its defining region`（`nvec=1` 也触发）。§3.3 注记把「P2/P4 编译通过」说成整段伪代码已验证（实为手写展开版）；§6.4 与 §1.6.2 V9 的「`nvec` 个寄存器 `vadd` 累加」无可用写法 | §3.3 改为**构建期按 `nvec` 特化 + 手写展开为单一表达式**：`nvec` 是构建期常量，段合并由**python 层** helper 生成 `vadd(vadd(p0,p1),p2)…` 的**一个**表达式，向量值**从不跨 IR `for` frame**；§3.3 注记改为「探针验证的是特化展开形态」，§6.4/§1.6.2 V9 同步改写 | F1/F2/F4（`nvec=1`）、F5（`nvec=2` 内联）、F6（`nvec=2` 经 helper）**COMPILE OK**；F7（v1 原文形态）**COMPILE FAIL**：`RuntimeError: Immutable variable 'acc' is used outside its defining region!`。F6 生成代码为 `vadd(vadd(vmul(...` 单表达式 | §3.3 伪代码即探针 F1/F2/F4/F6 的代码形态；§9.2 新增「禁止把向量值当循环携带量」错误项；§6.4 用「构建期特化」而非「运行期累加」表述 |
| **阻塞 2** | §3.3 的 `T.copy(o[r0:r0+R, h, :], o_ub)` 在 `o_ub` 形状 `(R, DW)` 而 `D_v < DW` 时违反平台条款「整个 Buffer 对整个 Buffer 的拷贝要求元素总数相等」（`references/programming.md:37`）：`PlanMTECopy: ... got src=4096 and dst=8192`；且 pad 列只清 `o_ub` 一侧时 `do_ub` 的 pad lane 未初始化，`0·Inf → NaN`。`D_v=64` 是 op 自身 smoke 几何（`test_mla_dense_bwd_general_kernel`） | 两条搬入改为**目的端显式切片** `o_ub[0:R, 0:D_v]` / `do_ub[0:R, 0:D_v]`；pad 安全化改为**两侧都清零**（`T.clear(o_ub)` **且** `T.clear(do_ub)`，构建期 `DW != D_v` 分支）；§4.5/§5.2/§8.3/§9.2 补 `D_v=64` 几何 | F2（切片 + 两侧清零，D_v=64）**COMPILE OK**，生成 `asc_copy_gm2ub_align(..., 64, 128, ..., 512, 256)`（64 行 × 128 B，目的行距 256 B，pad 列不被写）；F3（不切片）**COMPILE FAIL**：`Check failed: (analyzer->CanProveEqual(src_total, dst_total)) ... got src=4096 and dst=8192` | §3.2/§3.3 只出现切片形态；§9.2 记「整 buffer ↔ 整 buffer 的陷阱」与「两侧清零」；§3.5.1 把 `D_v=64` 从「本任务不触发」改为「op smoke 覆盖，L0 专门验证」 |
| **阻塞 3** | 尾块（`S=65` 等）两条 `T.copy` 仍取 `R` extent → 越界读 63 行 + 越界写 `delta[b,h,64:128]`；运行期谓词 `if s0+i < S` 与构建期分支重复/矛盾；编译器不拦截 | 尾段独立成**构建期代码段**：`nfull = S // R` 个满 tile 走 `T.Pipelined`，尾段两条 `T.copy`、行循环、写回**全部用静态 `nlast`**；**删除**运行期谓词；尾块几何取 `nsp = 1`（每 `(b,h)` 一个核），使 loop bound 与 extent 全为构建期常量 | F4：尾段生成 `asc_copy_gm2ub_align(..., 1, 256, ..., 256, 256)`（1 行 × 256 B）与 `asc_copy_ub2gm_align(..., 1, 4, ..., 4, 4)`（写 1 个 fp32），**COMPILE OK**；F8（`S=1`、`nfull=0`）**COMPILE OK** | §3.3/§6.4 的尾段只有一个静态形态；§9.2 新增「`R` extent 用于尾块」错误项；§8.3 保留 S=65 并补 S=1 组合 |
| 建议 1 | §1.6.1 A2 表内公式（2×`vcadd`+标量加）≠ §3.1/§3.3 实现（`vadd` 两半 + 1×`vcadd`），`verify_equiv.py` 只验证前者 | `verify_equiv.py` 的 A2 改为**实现分组**（`impl_grouping`：PART_EVEN/ODD 交错相加 → 1 次 64 lane 二叉树），并**并列** A2b 保留 v1 分组作对照；表内公式与收益口径改为「每行 1 条 `vcadd`」 | 重跑结果：A2 与 A2b **12/12 用例逐格相同**（`random/fp16/D128` 322 ulp、`large/fp16` 2798、`ulp(cand,base)` 2560）——原因是平衡二叉树在 128 = 2×64 处正好有子树边界，两种分组在 `nvec==1`（本任务全部 workload）下**逐位恒等** | §1.6.1 公式列即实现分组；`impl_grouping` 同时被 A2/A3 引用，不再有「证据指向另一分组」 |
| 建议 2 | R3 候选 6「每行 2 次 4 B 原子/部分和 = +2.1 MB」算错 2× | 改为 **+4.19 MB**（2 × 4 B × 524288） | 算术：2·4·524288 = 4194304 B | §1.6.0 R3 表 + §1.6.3 候选 3 引用同一数字 |
| 建议 3 | A1 收益栏「`2·R·D·4 B`（32 KB @R=64,D=128）」与公式不自洽（实为 64 KB；32 KB 是源 `block=32` 口径） | 拆成「源基线 `[32,128]×2` fp32 = 32 KB；若在 NPU 物化 fp32 乘积 tile 则 `2·R·D·4 B` = **64 KB**（R=64）」 | `2·64·128·4 = 65536 B` | §1.6.1 A1 行 |
| 建议 4 | `68 KB` 与 §4.5 的 67584 B（66 KB）两处口径不一；`d_ub` 写成 4 KB（实为 2 KB） | 全文统一为 **66 KB（67584 B）**、`d_ub` 2 KB、占用率 **0.27（66/248）** | `32768 + 32768 + 2048 = 67584` | §1.6.0 R3/R4、§4.5、§5.2/§5.3、§3.5.1 全同步 |
| 建议 5 | 「215 µs 是偏保守下界」不成立（1.26 TB/s 是 910B2C 48 核聚合地板；56 核按每核地板外推为 1.03 TB/s → 262 µs），且未与 manifest roofline 交叉引用 | 流量项改写为**区间 215–262 µs**（并标注两种外推口径）；补 manifest `attention.yaml` 的 `roofline.flops/bytes` 同口径核对 | 270.5 MB ÷ 1.26 TB/s = 214.7 µs；÷ 1.03 TB/s = 262.3 µs；manifest delta 项 `B*H*rows*2*D_v` = 134.2 MFLOPs、写出 `B*H*S_q*4` = 2.1 MB | §1.6.0「设计期估算（D-2）」段重写 |
| 建议 6 | §8.3 缺 `D_v=64`、manifest smoke（B=1,S=512,H=8）、长 `tpb`（`ds-v3-b2-8k`）用例 | §8.3 增三行并加「覆盖的设计点」列 | 几何取自 `manifest/attention.yaml` 与 `test_mla_dense_bwd.py:151` | §8.3 |
| 建议 7 | §3.4 把 `T.clear` 标为 ⚠️ | 改为 ✅ 并注明本任务探针已编译通过 | F2/F3/F4（`T.clear` 参与编译）；`T.clear` 在 v1 的 P 系列探针中亦通过 | §3.4 |


## 0. 源算子解读与迁移分析

### 0.1 源算子语义（做什么）

`_delta_kernel(batch, heads, dim_v, dense_seqlen, dtype)` 是 MLA prefill 反向的**第一段**（docstring：
“``delta[.., h, row] = sum_d o[row, h, d] * do[row, h, d]`` in float32”），为 `_bwd_kernel` 提供
`dP - delta` 里的行标量。

**数学语义**：对每个 `(b, h, s)` 三元组，在 value 维 `d = 0 … D_v-1` 上做**单次点积**：

```
delta[b, h, s] = Σ_{d < D_v} o[s, h, d] · do[s, h, d]
```

其中 `s` 是 query 行（dense 下 `row = b·S_q + s`，`o`/`do` 以 `(T_q, H, D_v)` 扁平化传入），`H` 为头数。
**与 `lse` 无关**（`lse` 是前向的 log-sum-exp，不进入本式；源码签名只有 `o, do` 两个输入即证据）。

**语义要素**（逐项由源码核对，非臆测）：

- **规约语义**：规约轴是 `D_v`（最内层、连续轴），无 keepdim 概念——输出比输入少一维；**累加顺序**为
  源码 `T.copy` 把 16-bit 输入提升到 float32 fragment → `T.Parallel` 逐元素乘 → `T.reduce_sum(o_f, rows, 1)`
  的 fragment 归约（GPU 侧展开为 warp 内树形归约 + 跨 warp 合并，即 binary tree 分组，不是严格串行累加）。
  → 迁移侧只要求**同 dtype 语义下的容差内等价**（分组可换，见 §1.6.1 A2 的机器验证）。
- **dtype 语义**：输入 `float16 | bfloat16`；`T.alloc_fragment([block, dim_v], "float32")` + `T.copy` 使
  **16-bit 输入在搬入时精确提升为 float32**（16→32 无损），乘积在 float32 中计算并**四舍五入到 float32**，
  累加全程 float32，输出 float32，**中间不回落 16-bit**。这是本算子最重要的一条 dtype 语义：
  “fp32 乘 + fp32 累加”，不是“fp16 乘 + fp32 累加”。
- **I/O 契约**：`o`/`do` 为 `(total, H, D_v)`（`total = T.dynamic("total_q")`，dense 调用下
  `total = batch·dense_seqlen`）；输出 `delta` 为 `out_idx=[2]` 自动分配的
  `(batch, heads, dense_seqlen)` **float32**（dense）或 `(heads, total)`（packed）。
- **边界语义**：① 尾块行 `row ≥ total` 由 `if row < total` 谓词挡住，不写；② `nan`/`inf` 无特殊处理，
  自然通过“乘 + 加”传播（`inf·0 → nan`）；③ `D_v` 空维不出现（`MLABwdKernelBase.refusal` 要求
  `dim_qk % 16 == 0 and dim_v % 16 == 0`）；④ 行数与 `block=32` 非整除是正常路径（谓词写回）；
  ⑤ `dense_seqlen == 0`（packed）走另一分支的写回索引，**本任务不涉及**（`MLADenseBwdOp`）。
- **host 侧语义**：`MLADenseBwdKernel.forward` 把 `(B, S, H, D)` 视图成 `(B·S, H, D)` 再调 `_run`；
  `delta` 由 `out_idx` 分配；`@target_cache(maxsize=32)` 按 `(batch, heads, dim_v, dense_seqlen, dtype)`
  特化编译。这些都属于**实现手段**，只有“`delta` 的 shape/dtype/布局”属于契约。

**语义保持基线**：§8.1 的 golden 只依据本节实现（`(o.float() * do.float()).sum(-1)`，与源 docstring
逐字对应），**不参考 §0.6 的 NPU 算法**。

### 0.2 源算子输入输出

| 参数 | 方向 | Shape | dtype | 说明 |
|---|---|---|---|---|
| `o` | in | `(total, heads, dim_v)` | `float16 \| bfloat16` | dense 下 `total = B·S_q`，行主序 `(s, h, d)` |
| `do` | in | `(total, heads, dim_v)` | 同 `o` | 输出梯度，dtype 与 `o` 相同（`ATTENTION_DTYPES`） |
| `delta` | out | `(batch, heads, dense_seqlen)` | `float32` | packed 调用为 `(heads, total)` |

源算子输出 shape（任务给定）：`delta [B, H, S_q]` float32（dense）。与 `_bwd_kernel` 的 `delta` 入参声明
`T.Tensor((batch, heads, dense_seqlen_q), "float32")`（源码第 134、148 行）以及消费处
`delta_s[i] = delta[b, h, m0 + i]`（第 217 行）**三者一致**——本设计的输出契约不得偏离。

### 0.3 实现算法解读（怎么算）

**源码结构化阅读**（`_delta_kernel` 全文 40 行 + host 侧 `_compile`/`_run`）：

- **入口签名**：`_delta_kernel(batch, heads, dim_v, dense_seqlen, dtype)` → 内部 `_delta_func()` 无参
  （几何全部是闭包常量），`total = T.dynamic("total_q")` 是唯一的运行期维度。
- **Kernel 结构**：`T.Kernel(heads, T.ceildiv(total, block), threads=128)` → **二维网格**（h, 行块），
  块内 128 线程；`block = 32` 行/tile。
- **计算原语**：`T.copy`（GM→fragment，含 16→32 提升）、`T.Parallel(32, dim_v)` 逐元素乘、
  `T.reduce_sum(o_f, rows, 1)`（dim=1 归约）、`T.Parallel(32)` 谓词写回。
- **内存分配**：`o_f`/`do_f` = `alloc_fragment([32, dim_v], "float32")`（**纯寄存器/fragment，无 SMEM**）、
  `rows = alloc_fragment([32], "float32")`。没有 `alloc_shared`，没有中间结果落 GM。
- **循环与调度**：无 K 维循环、无 `T.Pipelined`；只有一次 tile 处理（一个 block 一个 tile），
  块内由 128 线程并行摊平 `32×dim_v = 4096` 个元素（每线程 32 个）。
- **host 侧逻辑**：`MLADenseBwdKernel.forward` 的 `view`（`(B,S,H,D) → (B·S,H,D)`）与
  `_run` 里 `prep(o, do)` 的调用顺序；`_compile` 里三个 launch 的构建。
- **编译配置**：`pass_configs={TL_ENABLE_FAST_MATH: True}`、`compile_flags=["-O3", "-DENABLE_BF16"]`。

**计算步骤分解表**（源码每个计算语句均归入某步骤，无遗漏、无臆造）：

| # | 步骤 | 计算 | 输入 | 输出 | 对应语义公式 |
|---|---|---|---|---|---|
| 1 | 任务映射 | `(h, bx) ← T.Kernel(heads, ceildiv(total, 32))` | 网格索引 | — | 选择 `(h, 行块)` |
| 2 | 提升搬入 | `T.copy(o[bx·32 : bx·32+32, h, :], o_f)`（16→32，越界行不填） | GM `o` | `o_f` fp32 [32, D_v] | 取 `o[s,h,:]` 并精确提升 |
| 3 | 提升搬入 | 同步骤 2，`do` | GM `do` | `do_f` fp32 [32, D_v] | 取 `do[s,h,:]` |
| 4 | 逐元素乘 | `o_f[i,d] = o_f[i,d] * do_f[i,d]`（`T.Parallel(32, dim_v)`） | `o_f`, `do_f` | `o_f`（原地） | 每个 `d` 的乘积项 |
| 5 | 行归约 | `T.reduce_sum(o_f, rows, 1)` | `o_f` | `rows` fp32 [32] | `Σ_d` 部分 |
| 6 | 谓词写回 | `if row < total: delta[row // S, h, row % S] = rows[i]`（dense） | `rows` | GM `delta` | 结果落 `(b,h,s)` |

**数据流图（源硬件视角）**：

```
GM o[total,H,D] --T.copy（16→32 提升，32 条 256B 跨步段）--> fragment o_f[32,D] fp32
GM do[total,H,D] --T.copy（同上）-------------------------> fragment do_f[32,D] fp32
o_f ⊙ do_f（T.Parallel，寄存器原地）--> fragment o_f
o_f --T.reduce_sum(dim=1)--> fragment rows[32] --谓词写回--> GM delta[B,H,S]
```

无 SMEM、无中间 GM 往返；一个元素从 GM 进来只被读一次、只参与一次乘法和一次累加。

**循环与并行结构**：grid = `heads × ceil(total/32)`（二维，共 `H·ceil(total/32)` 个 block）；
block = 128 线程；每 block 一次 tile（32 行 × D_v），无循环携带依赖。

**host 侧逻辑**：`forward` 的两次 `view`；`cu_seqlens` 的构造（delta 不用）；`delta` 作为 `out_idx`
分配；`target_cache` 按 shape/dtype 特化。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 源码特征 | 目的 | 依赖的源硬件特性 | 初判 |
|---|---|---|---|---|---|
| O1 | **单遍融合乘+归约** | 乘积直接写回 `o_f` 后立即 `reduce_sum`，不落 GM | 消掉 O(N) 中间张量与一次 GM 往返 | 无（算法层） | 可移植 |
| O2 | **float32 提升后计算** | fragment 声明为 `"float32"`，`T.copy` 提升 | 数值稳定、与 golden 的 fp32 域一致 | 无（dtype 语义层） | 保留 |
| O3 | **fp32 product/累加全驻留寄存器** | `alloc_fragment`、无 `alloc_shared` | 每元素只付一次 GM 读 | 大寄存器堆（fragment 直接映射寄存器） | 硬件强相关 |
| O4 | **`block=32` 行级 tile** | `block = 32`，`T.copy` 一次搬 32 行 | 摊薄 128 线程的启动/调度成本，凑够 DMA 段 | coalesce 的 256B 段（D_v·2 B） | 硬件强相关（但可等价替换） |
| O5 | **head 固定 → 段内连续** | `o[rows, h, :]` 固定 `h` | 每行 256B 完全连续，段内无空洞 | GM 段式访问 | 硬件强相关（但可保留） |
| O6 | **谓词写回尾块** | `if row < total` | 允许 `total % 32 ≠ 0` | 线程级发散容忍 | 可移植 |
| O7 | **warp 级树形归约** | `T.reduce_sum(...,1)`（GPU 展开为 warp shuffle + 跨 warp） | 行内 128 元素 O(log) 合并 | warp（32 lane）与 `__shfl` | 硬件强相关 |
| O8 | **二维网格 + `threads=128`** | `T.Kernel(heads, ceildiv(total,32), threads=128)` | 任务映射到 block/thread 两级 | CUDA block/thread 模型 | 硬件强相关 |
| O9 | **shape 特化编译** | `@target_cache` + 闭包常量 | 消运行期分支与元数据 | host | 可移植 |

**算法层 vs 实现层**：O1、O2、O6、O9 属算法/语义层（单遍、dtype 语义、边界、特化），迁移优先保留其意图；
O3、O4、O5、O7、O8 属实现层（寄存器驻留、tile 形状、并行结构、warp 归约），需逐项判定（§0.5）。

### 0.5 硬件耦合性分析与 NPU 适配决策

判定输入含 §1.6.0 调研结论（调研先于判定）。目标平台事实（`tilelang.ascend.language` + 本平台 ruleset
`.agents/platforms/toolchains/ascend/references/programming.md`）：

- `T.Kernel(N)` 是**一维核网格，签名没有 `threads=`**（`ascend/language/kernel.py`；多维网格被拒绝）；
  线程域必须由 `T.SimtVF(threads=...)` 或 `T.SimdVF()` 在核内显式声明。
- 片上层级：UB 253952 B（`T.alloc_shared`，per-core）、L1 512KB、L0C 256KB、L0A/B 各 64KB
  （`Ascend950PR_9579.ini` 第 50/51/58 行）；本算子不使用 Cube。
- 无 warp/`__shfl` 语义；`T.tvm_warp_shuffle_*` 属 CUDA 公共面，迁移不可用。
- 高层 `T.reduce_sum` 的二维 shared→shared 路径在 `T.SimdVF` 内有**已确认的输出布局差异**
  （`.agents/platforms/toolchains/ascend/knowledge/numerical-boundaries.md` 的 `TRAP-950-simd-shared-reduce`）；
  SIMT 路径 `alloc_reducer → reducer_init → reducer_update → finalize_reducer` 已验证但每行一次
  跨线程 AllReduce（`T.finalize_reducer` 走 `tvm_thread_allreduce` + barrier）。
- 显式 SIMD 寄存器归约已验证：`T.simd.vld / vcadd / vsts`
  （`.agents/platforms/toolchains/ascend/knowledge/repro/PL-950-simd-reduce.py`，FP32 (2,64) 两行，含尾 lane mask）。

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|---|---|---|---|---|---|
| 数学语义（行点积、fp32 乘、fp32 累加） | 语义 | 无 | — | **保留** | 语义层无条件保留 |
| I/O 契约（`(T,H,D)` 入、`[B,H,S]` fp32 出） | 语义 | 无 | — | **保留** | 由 `_bwd_kernel` 的消费端固定 |
| O1 单遍融合乘+归约 | 算法 | 无 | 有 | **保留** | 目标侧仍是“乘完立刻归约、不落 GM”（§1.6.1 A1） |
| O2 fp32 提升计算 | 语义/dtype | 无 | 有 | **保留** | `vcvt` 到 fp32 后再 `vmul`（禁止 fp16 乘，见 §1.6.0 R1 弃选 C4） |
| O9 shape 特化 | 优化 | 无 | 有 | **保留** | 工厂按 `(batch, seqlen, heads, dim_v, dtype)` 特化 |
| O6 尾块谓词写回 | 优化/边界 | 线程发散 | 有（静态分支） | **等价替换** | 编译期 `S % R` 分支 + 静态尾段：满 tile 走流水，尾段的**两条 `T.copy`、行循环与写回全用静态 `nlast`**，无运行期谓词（§6.4、探针 F4） |
| O5 head 固定读 | 优化 | GM 段式访问 | 有 | **保留** | 目标侧同为固定 `h` 的 256B 连续段（探针已确认单条跨步描述符） |
| O4 `block=32` tile | 优化 | 256B 段 + 行数凑够 | 有 | **等价替换** | tile 改为 `R` 行（构建期选 `R | S`），UB 双缓冲（§5、§6） |
| O3 fragment = 寄存器 | 优化 | 大寄存器堆 + fragment 语义 | 部分 | **等价替换** | 用 SIMD 向量寄存器承载 fp32 中间量，**连 fp32 product tile 都不物化**（§1.6.1 A1） |
| O8 二维网格 + `threads=128` | 优化 | CUDA block/thread | **无** | **重新设计** | → §0.6 **R1**：一维 `T.Kernel` + block 内 `T.Pipelined` 行块循环 |
| O7 warp 树形归约（`T.reduce_sum(dim=1)`） | 优化 | warp shuffle / fragment 归约 | **无直接等价** | **重新设计** | → §0.6 **R2**：逐行显式 SIMD `vcadd`（fp32 二叉树归约），理由：二维高层归约有 TRAP 记录、SIMT reducer 每行一次 barrier |

> 依据来源：`T.Kernel`/`SimtVF`/`SimdVF` 签名见 `ascend/language/kernel.py`、`frame.py`（平台 ruleset
> `programming.md` 第 13–15 行转录）；高层归约差异见 `TRAP-950-simd-shared-reduce`；SIMD 归约可行性见
> `PL-950-simd-reduce`；容量见 `Ascend950PR_9579.ini`。舍弃项：无（O1–O9 全部有承接）。

### 0.6 NPU 算法重设计

**重设计项 R1：并行结构（二维网格 + 128 线程 → 一维核网格 + 行块流水）**

- **源方案**：`T.Kernel(heads, ceildiv(total,32), threads=128)`，一个 block 处理 `(h, 32 行)`，线程域 128
  隐式承担 `32×D_v` 的元素并行与归约；性能意图是“大量小 block 铺满 SM + 每 block 一次 32 行 DMA”。
- **NPU 新算法**：`T.Kernel(grid)` 一维网格，`grid = batch · heads · nsp`（`nsp` 为每个 `(b,h)` 的行轴切片数，
  构建期取 `nt = ceil(S/R)` 的因子；有尾块时固定为 1，见 §3.3 注 4）；一个 block **独占** `(b, h)` 的一段行区间，块内用
  `T.Pipelined(tpb, num_stages=2)` 迭代 `tpb = nfull / nsp` 个满 tile，每 tile `R` 行；`T.annotate_buffer_versions`
  给 `o_ub/do_ub` 双版本。核内没有线程域：归约走 SIMD（R2），元素并行由向量宽度（128 lane/寄存器）承担。
  逻辑核数按 §5.5 取 `grid = batch·heads·nsp`（大 workload 为 256 ≈ 4.6×56 Vector 核，小 workload 退化为 16–64），无跨核协作、无原子。
- **语义保持论证**：① 数学等价——每行仍是“D_v 个 fp32 乘积之和”，改的只是“哪一行由哪个核的哪次迭代算”；
  行与行之间**没有累加关系**（`delta` 的每个元素只依赖自己那一行），所以并行重排**不改变任何行的累加内容**，
  只改变归约分组（R2 覆盖，机器验证通过）。② 边界语义——dense 下所有 `b·S + s < total` 的行都有效；
  行区间按 `(b, s)` 静态切分，`S % R ≠ 0` 时尾段取 `nlast = S − nfull·R`（`nfull = S // R`，构建期常量），
  **尾段的两条 `T.copy`、`T.serial` 行循环与写回全部使用该静态 `nlast`**（§3.3、§6.4；探针 F4 的生成代码为
  `asc_copy_gm2ub_align(..., 1, 256, ...)` 与 `asc_copy_ub2gm_align(..., 1, 4, ...)`），越界行**既不读也不写**，
  与源 `if row < total` 同语义。③ 无跨核归并，故无浮点重结合风险。

**重设计项 R2：归约（fragment `T.reduce_sum(dim=1)` → 逐行显式 SIMD `vcadd`）**

- **源方案**：`T.reduce_sum(o_f, rows, 1)`——GPU 上对 `[32, D_v]` 的 fp32 fragment 沿 `d` 做块内归约
  （warp shuffle 树 + 跨 warp 合并），每行得到 1 个 fp32。性能意图：行内 O(log D) 合并、结果留在寄存器。
- **NPU 新算法**：每行在 `T.SimdVF` 内做
  `vld(16-bit 行) → 2×vcvt(PART_EVEN/PART_ODD) → 2×vmul → vadd(两半) → vcadd(64 lane 二叉树) → vsts(PAT_VL1)`
  （每行 11 条向量指令，探针已确认生成该序列）。`D_v > 128` 时按 `nvec = DW/128` 个寄存器分段，各段的部分和
  **在构建期展开成一个表达式** `vadd(vadd(p₀,p₁),p₂)…`（`nvec` 是构建期常量，段数不进入运行期循环；
  §3.3 的 `row_dot` helper，探针 F5/F6 已编译通过）；`D_v` 不足 128 的倍数时以零填充——pad 列必须**两个操作数
  都清零**（`T.clear(o_ub)` 且 `T.clear(do_ub)`；只清一侧时另一侧的未初始化 bit 模式满足 `0·Inf = NaN`，
  见 §3.3/§9.2），零项才真正不改变和。
- **语义保持论证**：① 数学等价——和式对象与项数相同（同一批 fp32 乘积），只把“顺序累加/多级 warp 合并”
  换成“64 lane 二叉树 + 两半相加”；误差属 fp32 分组差异，A2 以 fp64 参照机器验证：12 个用例
  （fp16/bf16 × random/large/tiny/corner/D=1/D=127）违反率全为 0，且**树形归约比顺序累加更接近真值**
  （random/fp16/D128：候选 322 ulp vs 基线 2238 ulp）。② dtype 语义保持——乘积仍在 fp32 中四舍五入
  （`vmul` 输出 fp32，无 FMA 收缩，见 §7.3）。③ 边界——`inf/NaN` 自然传播，corner 用例（±inf、`inf·0→nan`、
  全零、交替符号）与基线逐位一致（`nf_mis = 0`）；短行零填充等价于加 0（两侧清零时严格成立）。
- **源优化意图承接**：O7 的意图是“行内低延迟合并、结果驻留寄存器”——`vcadd` 单指令完成跨 lane 归约、
  结果直接落在向量寄存器，意图由新结构**完整承接**；不再需要 warp shuffle 是因为没有 warp。

**重设计项 R3：数据搬运（fragment 搬入 → UB 描述符 + 双缓冲）**

- **源方案**：`T.copy` 直接落 fragment（寄存器），无 SMEM；32 行 × 256B 的跨步读由 GPU 的 L1/LSU 处理。
- **NPU 新算法**：`T.copy(o[r0:r0+R, h, :], o_ub[0:R, 0:D_v])` 生成**一条**跨步 DMA 描述符（探针实测生成
  `asc_copy_gm2ub_align(dst, src, 64, 256, …, 2048, 256)`：64 行、每行 256 B、源间距 2048 B、目的行距 256 B；
  探针几何 H=8、D=128 → 源间距 8·128·2 B = 2048 B，H=128 的 workload 该参数为 32768 B）。
  目的端**必须**写成显式 region `o_ub[0:R, 0:D_v]`：`o_ub` 的完整形状是 `(R, DW)`，`D_v < DW`（如 `D_v=64`）时
  整 buffer ↔ 整 buffer 的元素数不等，平台直接拒绝（`references/programming.md:37`；探针 F3 复现
  `PlanMTECopy ... got src=4096 and dst=8192`，F2 切片后 COMPILE OK 且目的行距仍是 256 B）；
  归约结果先写进 32 字节行间距的暂存缓冲 `d_ub[R, 8]`（`vsts` + `PAT_VL1`），再由**一条** UB→GM 描述符
  写回（探针实测 `asc_copy_ub2gm_align(…, 64, 4, …, 4, 32)`：64 个 fp32、目的连续、源间距 32B）。
- **语义保持论证**：纯数据搬运，不改变任何数值；`d_ub` 的行间距只影响寻址，不影响写出的 64 个 fp32 值
  （A3 以“暂存 → 压缩”的索引映射机器验证，12 用例 0 ulp 差异）。

**§0.5 处置汇总**：表内 11 个条目——保留 6 项（语义层 2 + O1/O2/O5/O9）、等价替换 3 项
（O3/O4/O6）、重新设计 2 项（O8→R1、O7→R2）、舍弃 0 项。三项重设计均满足 §1.6.0 调研结论
（单遍流式、无跨核归并、Vector 单元承担逐元素与归约），且已按 skill Phase 2 继续做公式级优化
（§1.6.1）与向量化替代分析（§1.6.2），未止步于源公式直译。

### 0.7 标杆实现

- 源算子路径：`/workspace/TileOPs/src/tileops/_kernels/attention/mla/bwd.py`（第 38–77 行
  `_delta_kernel`；第 134/148/217 行是 `delta` 的消费端契约）。
- 源仓参考基准：`MLADenseBwdWorkload.ref_program`（`examples/TileOPs/tileops/workloads/attention.py`
  第 328 行起，float32 梯度参考）与 `examples/TileOPs/tests/ops/test_mla_dense_bwd.py` 的容差
  （`atol = max(5e-3, rtol)`，`rtol` 见 `_RTOL`）。
- 本函数的 golden **不移植**上述反向全过程，而是按 §0.1 语义落到单式 `(o.float() * do.float()).sum(-1)`
  ——与源 docstring 逐字一致，且与 `_bwd_kernel` 里 `delta` 的用法（`dP - delta`）解耦。

## 1. 概述

### 1.1 算子名称

`_delta_kernel`（`MLADenseBwdOp` 第 1 个 launch：注意力反向的行标量 `delta`）

### 1.2 功能描述

对每个 `(batch, head, query row)`，在 value 维上计算前向输出 `o` 与输出梯度 `do` 的**行点积**，
得到后续 `dP - delta` 需要的 float32 行标量；只依赖 `o`、`do`，与 `lse` 无关。

### 1.3 数学公式

$$
\delta[b,h,s] \;=\; \sum_{d=0}^{D_v-1} o[s,h,d]\cdot do[s,h,d],
\qquad
o,do \in \mathbb{R}^{B\times S_q\times H\times D_v}\ (\text{fp16/bf16}),
\quad \delta \in \mathbb{R}^{B\times H\times S_q}\ (\text{fp32})
$$

乘积与累加均在 float32 中进行（§0.1 dtype 语义）；`D_v = dim_v`（manifest workload 全为 128，op 级 smoke 另有
`D_v = 64`，见 §4.5/§8.3；设计按参数化处理）。

### 1.4 算法描述

**（迁移决策后的 NPU 侧算法；与源算法的差异逐条注明来源）**

1. **任务划分**（差异 ← §0.6 R1）：`grid = batch · heads · nsp` 个一维核；第 `bx` 个核解出
   `b = bh / heads`、`h = bh % heads`（`bh = bx / nsp`）与行轴切片号 `sp = bx % nsp`，
   负责 `(b, h)` 的行区间 `[sp·tpb·R, min((sp+1)·tpb·R, S))`，块内 `T.Pipelined` 迭代 `tpb = nfull/nsp` 个满 tile；
  `S % R ≠ 0` 时 `nsp = 1`，并在满 tile 段之后接一个 extent 全为 `nlast` 的尾段（§3.3 注 4）。
2. **搬入**（差异 ← §0.6 R3）：每 tile 把 `o[r0:r0+R, h, :]` 与 `do[r0:r0+R, h, :]` 各用**一条跨步 DMA**
  搬进 UB（16-bit 原样，不提升），目的端写成显式 region `o_ub[0:R, 0:D_v]`（`DW > D_v` 时元素数才相等，
  见 §3.2/§9.2），双版本缓冲；尾段三处 extent 全用静态 `nlast`（§6.4）。
3. **逐元素乘 + 提升**（差异 ← §1.6.1 A1）：在 `T.SimdVF` 内按行 `vld` 一个行向量（128 个 16-bit 元素 =
   正好一个向量寄存器），`vcvt` 两次（PART_EVEN/PART_ODD）得到两半 float32x64，两两 `vmul`。
   **fp32 乘积只存在于向量寄存器**，不物化 fp32 tile（源实现在 fragment 里物化了 fp32 乘积）。
4. **行归约**（差异 ← §0.6 R2）：`vadd` 合并两半 64 lane 部分和 → `vcadd` 做 64 lane 二叉树归约
   （结果广播在寄存器内）→ `vsts` 以 `PAT_VL1`/`extent=1` 写进 32 字节行间距的暂存缓冲 `d_ub[R, 8]`。
   `D_v > 128` 时 `nvec` 段的部分和由**构建期展开的单一表达式**合并（不进入运行期循环）。
5. **写出**（差异 ← §1.6.1 A3）：每 tile 结束后用**一条 UB→GM 跨步 DMA** 把 `d_ub[:, 0]` 压缩写进
   `delta[b, h, r0:r0+R]`（float32，dense 布局）；尾段写 `delta[b, h, r0:r0+nlast]`。

与源算法的结构差异一句话：源是“二维网格 + 32 行 fragment + warp 树归约”，本设计是“一维网格 + `R` 行 UB tile +
逐行 SIMD `vcadd`”；**数学内容与 dtype 语义不变**（同一批 fp32 乘积、同一 fp32 和），只有分组与数据驻留位置变化。

### 1.5 数据流图

```
GM o[B·S,H,D] ──(跨步 DMA, R×256B；D_v=64 时 R×128B，目的 region [0:R,0:D_v])──► UB o_ub[R,DW] 16-bit ─┐
GM do[B·S,H,D] ─(同上)────────────────────────────────────────────────────────► UB do_ub[R,DW] 16-bit ─┤
                                                            ├─► SimdVF 每行:
                                                            │     vld → 2×vcvt → 2×vmul → vadd → vcadd
                                                            ▼
                                    UB d_ub[R,8] fp32 (32B 行间距) ──(压缩 DMA, R×4B)──► GM delta[B,H,S] fp32
```

### 1.6 算法调研与优化分析

#### 1.6.0 算法调研（Algorithm Research）

> 调研对象：**同一数学语义（逐 (b,h,s) 的 value 维点积）的算法族**。源算法（单遍融合 + 并行归约）只是基线候选之一。
> 本算子属“多步/规约类”（乘 + 跨 128 元素归约 + 输出），按 `algorithm-research.md` §2 需**完整调研**（R1 候选表 +
> R3 四口径对比 + R4 逐候选评估）。本地检索：`algorithm-candidates.md`、pattern-library
> `.agents/skills/tilelang-op-optimize/references/pattern-library/constants.md`、
> `.agents/platforms/toolchains/ascend/knowledge/INDEX.md` 与 `examples/` 同类实现的 kb_search 均无本算子族条目
> （知识库预注入结果亦为无命中），故候选由结构判据 + 恒等变形自查 + 模型知识枚举；未做互联网检索的必要性
> 不成立（算子族为单点积，候选空间在 R1 表内封闭且每项都可由结构判定弃选）。

**R1 等价化简公式候选**（基线候选必须在表内；正式等价论证与收益量化在 §1.6.1 完成，此处只做初判）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 是否纳入 R3 对比 |
|---|------|------------|-----------|---------|----------------|
| 0 | **基线：源算法** | `Σ_d fl32(o_d·do_d)`，单遍、fp32 乘 + fp32 树归约 | — | — | ✅（基线） |
| 1 | 寄存器融合（不物化 fp32 乘积 tile） | 同上，乘积只在向量寄存器里存活 | 数学恒等（同一批 fp32 乘积与同一个和） | 省 UB/寄存器占用与一次片上往返 | ✅（采纳 A1） |
| 2 | 归约分组改为 64 lane 二叉树（两半 `vadd` 后 1 次 `vcadd`） | `Σ_d p_d`，分组不同 | 容差内等价（fp32 分组差异，无系统偏置） | 单指令 `vcadd` 代替多级跨线程合并 | ✅（采纳 A2） |
| 3 | 输出批量写（暂存 + 压缩 DMA） | 数值不变，只改写出粒度 | 数学恒等（纯数据搬运） | R 次标量写 → 1 条描述符 | ✅（采纳 A3） |
| 4 | 两遍：先物化 `p = o⊙do` 到 GM 再归约 | `p` 落 GM 后第二次读 | 数学恒等 | **负收益**：GM 流量 ×4 | ❌（R3 已量化淘汰） |
| 5 | 对角 GEMM：`diag(O·Doᵀ)` 或 ones-GEMM | 用矩阵单元算 `[R,R]` 取对角 | 数学恒等（在精确算术下） | **负收益**：FLOPs ×64 放大（= tile 行数 R，取 R=64） | ❌（R3 量化淘汰） |
| 6 | D 轴跨核切分 + 跨核合并 | 两核各算 64 个 d 再相加 | 容差内等价 | **负收益**：每行 4B 输出却要跨核同步 | ❌（R3/R4 淘汰） |
| 7 | fp16 乘 + fp32 累加（省 2 条 `vcvt`） | `fl16(o_d·do_d)` 再提升 | **不等价**：乘积先舍入到 fp16（相对误差 ≤ 2^-11 ≈ 4.9e-4） | 每行少 2 条向量指令 | ❌（精度：`delta` 绝对误差 ≈ 4.9e-4·‖p‖₂ ≈ 5e-3，与测试 atol 5e-3 同阶，经 `ds = p·(dp−δ)` 传到 dq/dk/dv 后超出 rtol 1e-3） |
| 8 | 代数化简（利用对称性 / 缩放 / 复用相邻行） | — | 无：`o` 与 `do` 逐行独立，点积无可约的恒等式（不存在把 D 次乘加降到更低的代数变形） | — | ❌（结构判定：信息量下界即 D 乘 + D 加） |

**R2 在线算法**：**有**。逐行点积天然是单遍流式（每行 running 累加器 1 个 fp32，状态 O(1)），
行与行之间**无依赖**（输出元素只依赖本行的 D 个输入），因此：
① 单遍即可（每元素只读一次，扫描 1 遍）；
② `D_v > 128` 时同一行内按段合并（段部分和在**构建期**展开成一个 `vadd` 表达式，运行期状态仍是 O(1) 个寄存器），
中间状态不随 `D_v` 增长；
③ tile 循环本身就是流式（每 tile 独立，无跨 tile 状态）。
不需要跨核 partial + merge（那是候选 6，已按“每行仅 4B 输出”淘汰）。中间缓冲峰值 O(tile)，不是 O(N)。

**R3 复杂度对比**（四口径；代表 workload = `ds-v3-4k`：B=1, S=4096, H=128, D_v=128, bf16，
行数 N = 524288；口径含“每元素一次读 + 每行一次写”，不含 kernel 启动与 16-bit 提升）：

| 算法候选 | FLOPs | 访存量 (Bytes) | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|-------|---------------|---------|-------------|---------------------|
| **0 基线（源算法）** | 2·N·D = 134.2 M | 读 2·N·D·2 = 268.4 MB + 写 N·4 = 2.1 MB = **270.5 MB** | 1（`o`/`do` 各 1 次） | 32 KB（fp32 乘积 fragment [32,128]×2） | N=524288 行可任意切核；**无跨核代价** |
| 4 两遍物化乘积 | 134.2 M（+物化） | 读 268.4 + 写 268.4 + 读 268.4 + 写 2.1 = **807.3 MB** | 3 | O(N·D) 落 GM | 同上，但流量 ×2.98 |
| 5 对角 GEMM | 2·R·R·D·(N/R) = 8.6 G（**64×**） | 读 268.4 MB + L1/L0C 往返 + 对角抽取 | 1 | L0C `[R,R]` fp32 = 16 KB/tile + 跨引擎中转 | 需 Vector↔Cube 协作（跨引擎同步） |
| 6 D 轴跨核切分 | 134.2 M | 读 268.4 MB + 每行 2 次 4 B 原子/部分和 = **+4.19 MB**（2·4·524288 B） | 1（+merge） | 同上 | 每行一次跨核合并；原子/第二遍 |
| **收敛候选（C0 的 NPU 落地）** | 134.2 M | **270.5 MB** | 1 | **66 KB（67584 B）**：`o_ub`/`do_ub` 2×16 KB×2 版本 + `d_ub` 2 KB | 无跨核协作、无原子 |

**R4 硬件亲和性评估**（逐候选对照检查清单；负向淘汰依据见“结论”列）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---------|-------------|---------|------------|---------|------------|------|
| **0/收敛候选** | 逐元素乘 + 归约 → **Vector（SIMD）单元**，正是本平台能力（§0.5 证据） | 66 KB / 248 KB = 0.27（66/248 = 0.266）✅ | 行宽 `D_v·2 B = 256 B` = 2 个完整 128B 段 ✅；UB 行对齐到一个向量寄存器 | `R \| S`（构建期），否则静态尾段（三处 extent 全用 `nlast`） | 双版本 UB + `T.Pipelined`（探针已生成双缓冲） | ✅ 采纳 |
| 4 两遍 | 同 0，但 MTE2/MTE3 流量 ×2.98 | 需 O(N·D) 中间张量落 GM | 同 0 | 同 0 | 两 kernel 或核内两遍，无收益 | ❌ 流量 807 MB ≫ 270 MB（1.26 TB/s 下 641 µs vs 215–262 µs，常数见下） |
| 5 对角 GEMM | Cube 需 N ≥ 16 fractal，而本例每行只有 1 个输出 → 64 个 N lane 中 63 个浪费；且乘积需先由 Vector 算出再送 Cube | L0C 16 KB/tile + L1 中转 | R 必须 ≥ 16（本例 R=64 可行） | 同 0 | 需 Vector→Cube 跨引擎同步（本任务已有 `PL-950-mixed-epilogue` 类结构，但对本算子纯属额外成本） | ❌ FLOPs 64× + 跨引擎同步，对“每 512B 输入只产生 4B 输出”的算子无收益 |
| 6 D 轴跨核切分 | Vector ✅ | 同 0 | 64 lane/半 ✅ | 同 0 | 需 GM 原子（`PL-950-atomic-splitk` 路径为 fp32 GM add）或第二遍 merge | ❌ 每行仅 4B 结果却要一次跨核合并，原子次数 = N，远超收益 |
| 7 fp16 乘 | Vector ✅（还少 2 条 `vcvt`） | 同 0 | 同 0 | 同 0 | 同 0 | ❌ 精度不达标（R1 第 7 行已量化；dtype 语义层不得改） |

**调研结论**：选定**候选 0 的算法结构 + 候选 1/2/3 的实现级等价优化**——即“单遍、fp32 乘积、逐行 64 lane
二叉树归约、结果批量写回”，与基线的结构差异只有“乘积驻留位置 + 归约分组 + 写出粒度”，无算法族替换。
依据：R3 中收敛候选的访存 270.5 MB 是理论下界（每元素必读一次、每行必写 4B），候选 4/5/6 分别把流量 ×2.98、
FLOPs ×64、跨核同步引入，均在本平台 Vector 单元的带宽/发射比下为负收益；R4 中收敛候选在计算单元、容量、
对齐、静态边界、流水五个维度全绿。**“无更优替代”的调研范围**：已查 `algorithm-candidates.md` 全表、
pattern-library 的 `constants.md` 与 `INDEX.md` 索引条目、`.agents/platforms/toolchains/ascend/knowledge/`
下 `ALG-950-formulas.md` / `operator-patterns.md` / `PL-950-row-primitives.md`、以及本仓 `examples/` 内
所有 ascend-target 向量算子（`paged_cache_write`、`moe_permute_align` 等）；无命中本算子族的替代算法。
本结论即 §1.4、§1.6.1–§1.6.3 的分析对象。

**设计期估算（D-2）**：代表 workload `ds-v3-4k`，估算下界 = max(流量项, 发射项, 容量项)：

- 流量项：270.5 MB ÷ 带宽 → **215–262 µs 区间**：
  - 上界 262 µs：按 `CONST-mte2-degradation` 的**每核**地板 18.4 GB/s × 56 Vector 核 = **1.03 TB/s**
    → 270.5 MB ÷ 1.03 TB/s = 262.3 µs（比 v1 的 215 µs **更保守**，是本设计采用的估算下界）；
  - 下界 215 µs：直接复用该条目记录的 **910B2C 48 核聚合**混合读写地板 **1.26 TB/s** → 214.7 µs；
  - 两个常数均绑定 **910B2C**、toolchain `tilelang 0.1.2+15ad002b3d`，**不是** ascend-950pr-9579 的实测值
    → 标注为**未实测外推**；v1 文中「56 核线性外推 1.47 TB/s → 184 µs」没有来源，本版删除。
- 发射项：每行 11 条向量指令（2 `vld` + 4 `vcvt` + 2 `vmul` + 1 `vadd` + 1 `vcadd` + 1 `vsts`）× 524288 行
  = 5.77 M 条 ÷ 56 核 ÷ 1.8 GHz ≈ **57 µs**（若 `vcvt`/`vcadd` 为多周期则上界约 100–130 µs；`CONST-vector-launch-overhead`
  的“0.5 µs/op 发射”口径绑定 910B2C，同样属未实测外推）。
- 容量项：66 KB / 248 KB = 0.27 → 不构成约束（UB 容量取自 `Ascend950PR_9579.ini` `ub_size=253952`）。
- **估算下界: 215–262 µs**（流量主导，发射项占流量项的 22%–60%，容量项不约束）。
- **与 manifest roofline 的同口径交叉核对**（`examples/TileOPs/tileops/manifest/attention.yaml`
  的 `roofline.flops` / `roofline.bytes`）：该式里 delta 的贡献是 `B * H * rows * 2 * D_v`
  = 1·128·4096·2·128 = **134.2 MFLOPs**、写出 `B * H * S_q * 4` = **2.1 MB**，与本节 R3 的 FLOPs 与写出口径
  逐项吻合；读入项对应 `2 * qv_elems * elem_bytes` = 268.4 MB。故本算子的 134.2 MFLOPs / 270.5 MB 与该
  manifest 表达式同口径，不冲突。
- 注意：`CONST-mte2-degradation` 的 L2 驻留口径提示“同输入张量逐 launch 复用时有效读带宽远高于 HBM 口径”，
  故 bench 循环下实测可能优于本估算——这是 §9.1 的风险项之一，不改变“流量主导”的结论。

#### 1.6.1 数学等价优化（公式级）

> 分析对象为 §1.6.0 选定的收敛候选。三项采纳项均不改数学内容，改的是“乘积驻留位置 / 归约分组 / 写出粒度”。
> 机器验证脚本：`verify_equiv.py`（本目录），torch CPU，fp64 精确参照；判决口径 = `|cand−ref| ≤ 1e-5·Σ|p_d| + 1e-6`
> （归约误差正比于 Σ|p_d| 而非 |Σ p_d|，见脚本内 `RTOL_SCALED` 注释），并附标准容差（atol=1e-4, rtol=1e-4）下的违反率。

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| A1 | 寄存器融合乘（不物化 fp32 乘积 tile） | `p[i,d] = fl32(o)·fl32(do)`（写入 fp32 tile）→ `Σ_d p[i,d]` | `Σ_d vmul(vcvt(o), vcvt(do))`，乘积只活在向量寄存器 | 数学恒等：两种形态对每个 `d` 都执行**同一个** fp32 乘法并四舍五入到 fp32，再对同一批项求和；只移动了值的存放位置（机器验证：cand 与 base **0 ulp** 差异，12/12 用例） | 源基线物化 `[32,128]×2` fp32 = **32 KB**；若在 NPU 上同样物化 fp32 乘积 tile 则为 `2·R·D·4 B` = **64 KB**（R=64, D=128）；A1 把这 64 KB 降到 0，并省掉每 tile 两次 `R·D·4 B` 的片上往返——这正是双缓冲（UB 总计 66 KB）能成立的原因 |
| A2 | 归约分组：`vadd` 两半 + 1 次 64 lane 二叉树（**实现分组**） | `acc = ((…((0+p₀)+p₁)+…)+p_{D-1})`（顺序/多级合并） | `vcadd( vmul(vcvt(o,PART_EVEN),vcvt(do,PART_EVEN)) + vmul(vcvt(o,PART_ODD),vcvt(do,PART_ODD)) )`——lane k 得 `p[2k]+p[2k+1]`，1 次 `vcadd` 做 6 级 pairwise | 容差内等价：同一批 fp32 项的重新分组，误差项仍是 fp32 舍入（无系统性偏置）。机器验证（`impl_grouping`，即本行公式）：12/12 用例违反率 0；且树形比顺序更接近 fp64 真值（random/fp16/D128：322 ulp vs 基线 2238 ulp）。**与 v1 引用的「2×`vcadd` + 标量加」分组的关系**：平衡二叉树在 128 = 2×64 处正好有子树边界，两种分组在 `nvec == 1`（本任务全部 workload）下**逐位恒等**，A2 与 A2b 两行 12/12 用例数字完全一致即为机器证据。fp16/bf16 舍入影响：输入提升到 fp32 后乘、加都在 fp32，16-bit 只在入口（无损提升），故分组差异不会放大到 16-bit 量级 | 归约从 `T.reduce_sum` 的多级跨线程合并（barrier + 共享内存中转）变为每行 1 条 `vcadd`：若沿用 SIMT reducer 路径，每行一次 `tvm_thread_allreduce` + barrier（未实测假设：>100 cycle/次），N=524288 行时约 0.5–1 ms 量级，超过 215–262 µs 的流量下界 |
| A3 | 输出批量写（暂存 + 压缩 DMA） | 每行一次 4 B 标量写回 GM（`delta[b,h,s] = rows[i]`） | 写入 32 B 行间距的 `d_ub[R,8]`，每 tile 一条压缩描述符写 `delta[b,h,r0:r0+R]` | 数学恒等：纯数据搬运；索引映射 `d_ub[i,0] → out[i]` 机器验证 **0 ulp**（12/12 用例） | 写出指令数从 `R`/tile（N 次全局）降到 1/tile（`N/R` 次）；按 910B2C 的“段/指令”口径估算，N 次 4 B 写约 25–47 µs 的指令开销被摊掉（未实测外推） |

**等价性机器验证结果**（`verify_equiv.py` 实跑输出，torch CPU + fp64 参照；用例 = fp16/bf16 ×
random(D=128) / random(D=127) / large(±512) / tiny(1e-5) / corner(±inf、NaN、inf·0、全零、交替符号) / D=1）：

| 采纳项 | 用例数 | 最大 ulp(cand, ref) | 最大违反率（尺度化口径） | 标准容差违反率 | 非有限值不一致 | 结论 |
|--------|--------|---------------------|--------------------------|----------------|----------------|------|
| A1 寄存器融合 | 12 | 3310 | 0.0000 | 0.0156（仅 large/fp16 的 1/64 行，见下注） | 0 | **EQUIV_PASS** |
| A2 实现分组（`vadd` 两半 + 1×`vcadd`） | 12 | 2798 | 0.0000 | 0.0156（同上） | 0 | **EQUIV_PASS** |
| A2b v1 引用分组（2×`vcadd` + 标量加） | 12 | 2798 | 0.0000 | 0.0156（同上） | 0 | **EQUIV_PASS** |
| A3 批量写出 | 12 | 2798 | 0.0000 | 0.0156（同上） | 0 | **EQUIV_PASS** |

> A2 与 A2b 的 12 行逐格相同（含 `ulp(cand,base)` 最大值 2560）——这不是巧合而是恒等：`vcadd` 的 6 级 pairwise
> 与 `vadd(PART_EVEN, PART_ODD)` 的第一级正好构成对 128 元素的平衡二叉树，而平衡二叉树在 64 处有子树边界，
> 故 `tree128 = tree64(前半) + tree64(后半)`。因此 v1 把「2×`vcadd` + 标量加」写成公式、把「`vadd` 两半 + 1×`vcadd`」
> 写成实现，二者在 `nvec == 1` 下同值；本版把表内公式改成实现分组（A2），并保留 A2b 作对照。
> 注：`large/fp16` 的 1/64 行是**抵消行**（Σ|p_d| ≈ 1e5 而 |Σ p_d| ≈ 0），在 atol+rtol·|ref| 口径下
> 任何 fp32 累加（含基线自身）都会“违反”；尺度化口径（误差 ≤ 1e-5·Σ|p_d|）下 0 违反。这不是等价性问题，
> 而是 fp32 累加对抵消行的固有性质，源算子同样具备（同一 dtype 语义），列入 §9.1 风险项。
> 逐用例明细见脚本输出（每条含 `ulp(cand,ref)`、`ulp(cand,base)`、两种违反率与 `nf_mis`）。

**优化结论**：采纳 3 项（A1 寄存器融合、A2 二叉树归约〔实现分组〕、A3 批量写出），全部机器验证通过、
无失败项；A2b 是 v1 引用分组的对照行，同样通过并证明两分组在 `nvec == 1` 下逐位恒等。
优化后公式（供 §3.1 拆解使用）：

```
for each row (b, h, s):
    v_o = vcvt_pair(o_ub[s, :])      # 16→fp32，两半各 64 lane
    v_d = vcvt_pair(do_ub[s, :])
    acc = vadd(vmul(v_o.lo, v_d.lo), vmul(v_o.hi, v_d.hi))    # fp32，lane k = p[2k]+p[2k+1]
    delta[b, h, s] = broadcast_sum(vcadd(acc))                # fp32，64 lane 二叉树
# D_v > 128（nvec > 1）：每段的 acc 在构建期用 vadd 串成一个表达式，再做唯一一次 vcadd
```

其余空间（乘、加、提升）已是逐元素最小操作集，无公共子表达式可消、无高代价算子可降代换、无除法/超越函数
可改写——这是“无进一步优化空间”的结论与依据。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由（不可替代时必填） |
|---|--------|-----------|-------------|---------|------------------------------|
| V1 | 行内逐元素乘 | `o_f[i,d] * do_f[i,d]` 的逐元素循环 | `T.simd.vmul`（fp32x64） | ✅ | — |
| V2 | 16-bit → fp32 提升 | `T.copy` 隐式 cast 循环 | `T.simd.vcvt(…, "float32", part=PART_EVEN/PART_ODD)` | ✅ | — |
| V3 | 两半 64 lane 部分和合并 | 逐元素累加 | `T.simd.vadd` | ✅ | — |
| V4 | 行内 128 元素求和 | `T.reduce_sum` / 标量累加 | `T.simd.vcadd`（64 lane 二叉树） | ✅ | — |
| V5 | 每行结果写暂存 | `d_ub[i,0] = rows[i]`（标量写） | `T.simd.vsts(…, PAT_VL1, extent=1)` | ✅ | — |
| V6 | GM → UB tile 搬入 | 逐元素/逐行 load 循环 | `T.copy`（单条跨步 DMA，探针确认） | ✅ | — |
| V7 | UB → GM 结果写回 | 逐行标量 store 循环 | `T.copy(d_ub[0:R, 0:1], delta[b,h,r0:r0+R])`（单条压缩 DMA） | ✅ | — |
| V8 | pad 列清零（`D_v` 非 128 倍数时） | 逐元素清零 | `T.clear(o_ub)` **且** `T.clear(do_ub)`（构建期 `DW != D_v` 分支，循环前各一次） | ✅ | — （只清一侧不够：另一侧的 pad lane 是未初始化 UB，其 bit 模式若为非有限值则 `0·Inf = NaN`，见 §9.2） |
| V9 | 行循环 `for i in T.serial(R)` | 外层行迭代 | 无（每行一次 `vld/vcvt/vmul/vadd/vcadd/vsts`） | ❌ | **API 缺失**：`vcadd` 一次只归约**一个**寄存器的 64 个 lane，本平台没有“跨多行批量归约”的已验证指令；把 64 行塞进同一寄存器需要 UB 内 64×128 转置链（`vdintlv/vintlv` 组合），转置代价远大于省下的 `vcadd`（估算：每行 1 条 `vcadd` 占每行 11 条指令的 9%，转置需 ≥2 条/行 + 额外 UB 往返）。已核 `simd.py` 导出全表（`vcadd/vcgadd/vcpadd` 均为单寄存器内归约）。见 §1.6.3 候选 7。**注**：行内 `nvec`（`D_v > 128`）的分段合并不是运行期循环——它在构建期被展开成一个 `vadd` 表达式（§3.3 的 `row_dot`），不构成“循环携带向量累加器” |
| V10 | tile/核索引计算（`b, h, sp, r0, bh`） | 标量整数运算 | 无 | ❌ | **block 级索引**：每 block 一次、每 tile 一次，不是逐元素热点；无逐元素等价 API（也不需要） |
| V11 | 尾段行数分支（`S % R ≠ 0`） | 运行期谓词 | 无（改为构建期分支 + 静态行数） | ❌ | **静态边界**：`nlast` 是构建期常量，尾段的两条 `T.copy`、`T.serial(nlast)` 行循环与写回都用它展开成静态 extent（探针 F4），非逐元素判断；改成运行期 mask 反而引入逐元素谓词并让 extent 退回 `R`（即 v1 的越界缺陷） |

**向量化结论**：逐元素与归约计算点（V1–V8）已全部向量化，所用 API（`vld/vcvt/vmul/vadd/vcadd/vsts`、`T.copy`、
`T.clear`）均有本平台证据（`simd.py` 导出 + `PL-950-simd-reduce` 已验用例 + 本任务 `history_version/design_probe_compile.py`
与 `design_probe_compile_v2.py`（F1/F2/F4/F5/F6/F8）的生成代码）。保留 3 处标量/循环，理由类别分别为 **API 缺失**（V9，附量化与查证过程）、
**block 级索引**（V10）、**静态边界**（V11）——无一处以“实现简单/照源码写”为由保留。

#### 1.6.3 向量化轴与数据布局决策

**轴质量评分（候选逐项）**：

| 评分项 | 说明 | 候选 1（主选） | 候选 2（备选） |
|--------|------|---------------|---------------|
| 整除性 | 向量轴长 vs 向量宽度（16-bit = 128 lane） | `D_v = 128` = 恰好 1 个向量寄存器 ✅ | 同左（每个 head 段仍是一个寄存器） |
| 尾 lane 浪费率 | `(ceil(D_v/128)·128 − D_v)/D_v` | `D_v=128` → 0%；`D_v=64` → 100%（一半 lane 是 pad；但每行仍只占 1 个寄存器/11 条指令，pad 只浪费 lane 不增加指令数） | 同左 |
| 累加链形态 | 向量维 stride=1、串行维是否阻碍向量化 | 向量维 = `d`（stride 1）✅；行维是独立归约实例，无累加链 | 同左 |
| repack 代价 | 是否需要核内/ host 重排 | **无**（I/O 布局直接可用，只靠 DMA 描述符的段间距） | **无**（输入完全连续） |
| UB 容量影响 | 驻留缓冲预算 | `2·R·DW·2 B` = 32 KB（双版本 64 KB）+ 2 KB 暂存 = **66 KB** ✅ | 较差：全 `H` 头布局每行 32 KB（H=128）→ 双缓冲只能 `R = 1`（见候选 2 的量化） |

**候选矩阵**（全部枚举；弃选给量化理由）：

| # | 布局方案 | 向量化轴 | repack 路径 | 预估收益 / 代价 | 是否采纳 |
|---|---------|---------|------------|-----------------|---------|
| 1 | **I/O 原生布局 + head 固定 tile `[R, D_v]`** | `d`（行内连续，1 寄存器/行） | 无（跨步 DMA：R 行 × 256 B，源间距 H·D_v·2 B，探针已确认单描述符） | 输入 256 B/段 = 2 个完整 128B 段；输出完全连续（`R·4 B` = 256 B @R=64）；UB 66 KB（67584 B） | ✅ **主选** |
| 2 | 全头 tile `[R, H, D_v]`，输入完全连续 | `d`（每 `(r,h)` 段 1 寄存器） | 无 | 输入 1 条连续描述符（无段间距）；但 `H·D_v·2 B = 32 KB/行`（H=128）→ 双缓冲下 `R = 1`（单缓冲 `R ≤ 3`），且输出变成 `H` 个 `R·4 B` 的散段（R=1 时 4 B/段，写放大 8–32 倍） | ❌（备选，见裁决计划） |
| 3 | D 轴跨核切分（2 核 × 64 d） | `d` | 需跨核中间结果 | 每行 4 B 却要一次跨核合并/原子：原子次数 = N = 524288，原子流量 +4.19 MB（2×4 B/行），且写出前必须跨核同步 | ❌（R3/R4 已量化淘汰） |
| 4 | lane = 行、`D` 串行（垂直扫描） | 无（行内退化为标量） | 无 | 每行 128 次标量乘加：向量化率 1/128，向量单元失效（估算 ≥ 8× 慢于候选 1） | ❌ 弃选 |
| 5 | Cube：`diag(O·Doᵀ)` / ones-GEMM | Cube fractal | 需 Vector→Cube 中转 | FLOPs 64×、N=1 < 16 fractal、跨引擎同步 | ❌（R3/R4 已量化淘汰） |
| 6 | host 侧 permute 成 `(H, S, D)` 让 `s` 连续 | `d` | 两次全张量 host permute | 额外一次全量读 + 写（268 MB 级），远超算子本体收益；且 host 只允许 stride/shape 视图，禁止改动输入内容 | ❌ 弃选 |
| 7 | 核内转置（lane = 行）后批量归约 | 转置后的 `d` | UB 内 64×128 转置链（`vdintlv/vintlv`） | 省 1 条 `vcadd`/行（11 条中的 9%），却要 ≥2 条/行的转置 + 额外 UB 往返 | ❌ 弃选（API 存在但代价倒挂） |

**布局决策结论**：选定候选 1——**核内布局 = `[R 行, D_v]` 的 UB tile（head 固定），向量化轴 = `d`（行内连续、
`D_v=128` 恰好一个向量寄存器），repack 路径 = 无（靠跨步 DMA 描述符表达源间距 2048 B / 目的连续）**。
该结论已同步落入 §3.3 伪代码（`o_ub[i, 0]` 起 `vld` 整行）与 §6 循环结构（内层向量维 = `d`，行循环是归约实例维）；
弃选候选的量化理由见候选矩阵。`D_v > 128` 时向量化轴仍是 `d`，只是按 `nvec = DW/128` 个寄存器分段，各段部分和
在**构建期展开成一个 `vadd` 表达式**（§3.3 的 `row_dot`；段数不进入运行期循环，探针 F5/F6 已验证）。

**实验裁决计划（三件套；判定裕度依赖未实测常数）**：主选与备选的分歧点落在**两个本型号未实测的常数**上——
① 跨步描述符（64 段 × 256 B，源间距 2048 B）相对单条连续描述符的 MTE2 有效带宽差；② 散段输出（`H` 个
`R·4 B` 段）的写放大代价。`constants.md` 只有 910B2C 的“段/指令”口径（`CONST-mte2-degradation`），
不足以纸面定案，故：

- **主选（进 Stage 3）**：候选 1 + 核内 `T.Pipelined(tpb, num_stages=2)` 双缓冲，`grid = batch·heads·nsp`（§5.5）。
- **备选（≤1 个，结构完整）**：候选 1 的**逻辑核超发**形态——`grid = batch·heads·nt`（一 tile 一 block），
  去掉 `T.Pipelined` 与 `annotate_buffer_versions`，其余（tile 形状 `R`、UB 预算 34 KB、行循环、SIMD 归约、
  暂存写出）不变，分核三要素按 `grid` 重算（256 → 8192 级，靠硬件波调度；UB 34 KB/块，无双缓冲）。
  选择它作为备选的原因：探针显示 `T.Pipelined(2)+num_stages=2` 的生成代码**并未把下一 tile 的装载提前到
  上一 tile 计算之前**（§7.2 的完整生成代码里 `asc_sync_wait(PIPE_V, PIPE_MTE2, step)` 与 compute 同迭代），
  因此“流水收益 vs 每 block 固定开销”是真实未知量，值得实测。
- **裁决计划**：代表 shape = `ds-v3-4k`、`ds-v3-b2-8k`、`kimi-k2-8k`（另加 smoke 作健全性检查）；
  指标 = `msprof` 的 Task Duration(us)（同 Stage 4 口径，A/B 交错各 ≥5 次）；**判定阈值**：备选 latency
  < 主选 × 0.9 才翻转；裁决结果按 shape 分片回写本节（shape 特化工厂允许按 workload 分派主选/备选，
  不必全局二选一）。Stage 4 的 A/B 清单直接取本节主选 + 备选。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式: SIMD**（任务平台 `programming_modes.choices.simd`；`migration_default = simd`，用户未另行指定）。

### 2.2 选型理由

- 算子**不含任何矩阵乘**：每行只有“乘 + 128 元素归约”，Cube 单元的 fractal 语义（N ≥ 16）与“每行 1 个输出”
  不匹配（§1.6.0 R4 候选 5 已量化淘汰），故排除 `Cube` 与 `混合`。
- 归约是本算子的唯一结构性计算：`SIMD` 用 `vcadd` 一条指令完成 64 lane 二叉树归约，结果直接留在寄存器；
  `SIMT`（`T.SimtVF` + `T.reduce_sum` / `alloc_reducer`）每行要做一次跨线程 AllReduce（barrier + 共享内存），
  在 N = 524288 行的规模上把发射开销推到 0.5–1 ms 量级（§1.6.1 A2 已量化），是模式选择上的决定性差异。
- 逐元素乘与 16→32 提升是纯向量操作（`vmul`/`vcvt`），SIMD 的 128 lane（16-bit）/64 lane（fp32）宽度
  与 `D_v = 128` 恰好对齐（§1.6.3）。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_shared`（UB 253952 B/core）：`o_ub[R,DW]`、`do_ub[R,DW]` 16-bit（各双版本）、`d_ub[R,8]` fp32 暂存；不使用 L1/L0 |
| 计算方式 | `with T.SimdVF():` 内逐行 `T.simd.vld → vcvt → vmul → vadd → vcadd → vsts`；无 `T.Parallel`（SimdVF 无线程域） |
| 并行来源 | 核间：一维 `T.Kernel(grid)`（grid = 256 级）；核内：向量宽度（128 lane）+ 行循环；无 warp/线程概念 |
| 同步 | 编译器自动插入 `asc_sync_notify/asc_sync_wait`（§7） |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式为 §1.6.1 的**优化后公式**（A1+A2+A3）。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `r0 = sp·tpb·R + it·R`，`h`、`b` 由 `bx` 解出 | tile 任务映射（block 级索引） |
| 2 | `o_ub[r, d] ← o[r0+r, h, d]`，`do_ub[r, d] ← do[r0+r, h, d]`，`r < R`，`d < D_v` | GM→UB 跨步搬入（16-bit 原样），目的端显式 region `o_ub[0:R, 0:D_v]`；尾段把 `R` 换成静态 `nlast` |
| 3 | `vo = vld(o_ub[i, 0])`，`vd = vld(do_ub[i, 0])` | 取第 `i` 行的整行向量（`D_v = 128` = 1 寄存器） |
| 4 | `o_lo = vcvt(vo, PART_EVEN)`，`o_hi = vcvt(vo, PART_ODD)`；`d_lo/d_hi` 同理 | 16-bit → fp32（精确提升，两半各 64 lane） |
| 5 | `acc = vmul(o_lo, d_lo) + vmul(o_hi, d_hi)` | fp32 乘积（每条 `vmul` 四舍五入到 fp32）+ 两半合并 |
| 6 | `s_i = vcadd(acc)` | fp32 树形归约（64 lane → 广播标量） |
| 7 | `d_ub[i, 0] = s_i` | `vsts` + `PAT_VL1` + `extent=1`（32 B 行间距） |
| 8 | `delta[b, h, r0:r0+R] = d_ub[:, 0]` | 每 tile 一次压缩 UB→GM 写出 |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API（`tilelang.ascend.language`） | 参数 | 模式 |
|------|----------|-------------------------------------------|------|------|
| 1 | 核/tile 索引 | `T.Kernel(grid)` / `T.Pipelined(tpb, num_stages=2)` / `T.serial(R)` | `grid`、`tpb`、`R` 全为构建期常量 | SIMD（核级） |
| 2 | GM→UB 搬入 | `T.copy(o[r0:r0+R, h, :], o_ub[0:R, 0:D_v])`（`do` 同） | 源为 3-D 区域（固定 `h`），目的**显式 region** `[0:R, 0:D_v]`（`DW > D_v` 时必须，否则元素总数不等）；生成单条跨步 DMA | SIMD（MTE2） |
| 2b | 双版本缓冲 | `T.annotate_buffer_versions({o_ub: 2, do_ub: 2})` | `num_stages=2` 配套 | 流水 |
| 2d | 尾段搬入 / 写回 | `T.copy(o[r0:r0+nlast, h, :], o_ub[0:nlast, 0:D_v])`、`T.copy(d_ub[0:nlast, 0:1], delta[b, h, r0:r0+nlast])` | `nlast` 为构建期常量（`S % R != 0` 分支）；满 tile 段不用 | SIMD（MTE2/MTE3） |
| 3 | 整行装载 | `T.simd.vld(o_ub[i, 0])` | 每行 128×16-bit = 1 个 2048-bit 寄存器 | SIMD |
| 4 | 类型提升 | `T.simd.vcvt(vo, "float32", part=0 / part=1)` | `PART_EVEN` / `PART_ODD` 取两半 64 lane | SIMD |
| 5 | 逐元素乘 + 合并 | `T.simd.vmul(a, b)`、`T.simd.vadd(a, b)` | fp32x64；默认 `PAT_ALL` mask | SIMD |
| 6 | 行归约 | `T.simd.vcadd(acc)` | 64 lane 二叉树，结果广播 | SIMD |
| 6b | `nvec > 1` 的段合并 | 构建期 python helper（§3.3 `row_dot`）返回 `vadd(vadd(p₀,p₁),p₂)…` **单一表达式** | `nvec = DW/128` 是构建期常量；**不得**写成 IR 循环携带向量累加器 | 构建期展开（探针 F5/F6） |
| 7 | 结果暂存 | `T.simd.vsts(d_ub[i, 0], s, one, extent=1)`，`one = T.simd.pset(32, "PAT_VL1")` | 只写 1 个 lane；行间距 32 B（`d_ub[R,8]`） | SIMD |
| 8 | UB→GM 写出 | `T.copy(d_ub[0:R, 0:1], delta[b, h, r0:r0+R])` | 生成单条压缩 DMA（64 元素、源间距 32 B、目的连续） | SIMD（MTE3） |
| 2c | pad 清零（条件） | `T.clear(o_ub)` **且** `T.clear(do_ub)` | 仅 `DW != D_v` 的构建期分支，循环前各一次（两侧都清，见 §9.2） | SIMD |

### 3.3 计算伪代码

> **本节的形态即探针 `history_version/design_probe_compile_v2.py` 的 F1/F2/F4/F5/F6/F8 代码**（全部 COMPILE OK）；
> v1 的两个不可构建形态（F3 不切目的端、F7 的 `for v in range(nvec)`）作为负向对照保留在注记里。
> 三条硬约束：① 向量值**不得**跨 IR `for` frame 传递（`nvec` 段在构建期展开成一个表达式）；
> ② 目的端**必须**显式切片 `[0:R, 0:D_v]`；③ 尾段的**三处 extent**（两条 `T.copy` + 行循环）全用静态 `nlast`。

```python
# ---- 构建期常量（全部由 (batch, S, heads, D_v, dtype) 决定，无运行期分支）----
VL    = 128                      # 16-bit 向量宽度（2048 bit / 16）
DW    = ceil(D_v / VL) * VL      # UB 行宽；D_v=128 → 128；D_v=64 → 128（右半 64 列是 pad）
nvec  = DW // VL                 # 段数（构建期常量，D_v<=128 时恒为 1）
R     = max{ d ∈ [16, 64] : d | S } 若存在，否则 64     # 构建期规则（§5.2）
nfull = S // R                   # 满 R 行 tile 数
nlast = S - nfull * R            # 尾段行数（0 = 无尾块）
tail  = nlast != 0
nt    = ceil(S / R)              # 逻辑 tile 数（含尾段）
nsp   = 1 if tail else (nt 的最小因子 d 使 batch*heads*d >= 224；无则 nt)
tpb   = nfull // nsp             # 每核的满 tile 数（无尾块时 = nt//nsp）
grid  = batch * heads * nsp


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
    @T.prim_func
    def main(o: T.Tensor((batch * S, heads, D_v), dtype),
             do: T.Tensor((batch * S, heads, D_v), dtype),
             delta: T.Tensor((batch, heads, S), "float32")):
        with T.Kernel(grid) as bx:
            bh, sp = bx // nsp, bx % nsp          # 任务映射（block 级标量）
            b, h = bh // heads, bh % heads
            o_ub  = T.alloc_shared((R, DW), dtype)
            do_ub = T.alloc_shared((R, DW), dtype)
            d_ub  = T.alloc_shared((R, 8), "float32")     # 32 B 行间距的暂存
            T.annotate_buffer_versions({o_ub: 2, do_ub: 2})
            if DW != D_v:                                  # 构建期：pad 列必须**两侧**都清零
                T.clear(o_ub)                              #   （否则 0·(Inf/NaN bit 模式) = NaN）
                T.clear(do_ub)
            # ---- 满 tile 段（流水双缓冲）----
            if tpb > 0:                                    # 构建期：S < R 时只有尾段
                for it in T.Pipelined(tpb, num_stages=2):
                    s0 = (sp * tpb + it) * R
                    r0 = b * S + s0                        # 3-D 扁平行号
                    T.copy(o[r0 : r0 + R, h, :], o_ub[0:R, 0:D_v])    # 目的端显式 region
                    T.copy(do[r0 : r0 + R, h, :], do_ub[0:R, 0:D_v])
                    with T.SimdVF():
                        one = T.simd.pset(32, "PAT_VL1")
                        for i in T.serial(R):              # 归约实例维（V9：API 缺失，见 §1.6.2）
                            T.simd.vsts(d_ub[i, 0], row_dot(o_ub, do_ub, i, nvec), one, extent=1)
                    T.copy(d_ub[0:R, 0:1], delta[b, h, s0 : s0 + R])  # 压缩写出
            # ---- 尾段（S % R != 0；三处 extent 全用静态 nlast，无运行期谓词）----
            if tail:
                s0 = nfull * R
                r0 = b * S + s0
                T.copy(o[r0 : r0 + nlast, h, :], o_ub[0:nlast, 0:D_v])
                T.copy(do[r0 : r0 + nlast, h, :], do_ub[0:nlast, 0:D_v])
                with T.SimdVF():
                    one = T.simd.pset(32, "PAT_VL1")
                    for i in T.serial(nlast):
                        T.simd.vsts(d_ub[i, 0], row_dot(o_ub, do_ub, i, nvec), one, extent=1)
                T.copy(d_ub[0:nlast, 0:1], delta[b, h, s0 : s0 + nlast])
    return main
```

> **注 1（构建期 vs 运行期）**：`DW/nvec/R/nfull/nlast/tail/nsp/tpb/grid` 全是 python 常量，所以 `if DW != D_v`、
> `if tpb > 0`、`if tail` 在 trace 期被折叠成单一代码路径；kernel 内**没有**运行期 shape/边界谓词
> （v1 的 `if s0 + i < S` 已删除——它在两份构建期路径下要么恒真、要么会与静态 extent 冲突）。
>
> **注 2（`nvec` 特化，阻塞 1）**：`row_dot` 是**模块/工厂层的 python 函数**，其 `for v in range(nvec)` 在 trace 期
> 用 python 执行（`nvec` 是 int），只拼接表达式；不得把它写成 prim_func 体内的 `for v in range(nvec)`——
> 那会被提升为 IR `ForFrame`，跨迭代重绑定向量值直接报
> `RuntimeError: Immutable variable 'acc' is used outside its defining region!`（探针 F7 复现）。
> 探针证据：F1/F2/F4（`nvec=1`）、F5（`nvec=2` 内联展开）、F6（`nvec=2` 经 `row_dot`）全部 COMPILE OK，
> F6 生成 `vadd(vadd(simd_inst::vmul(...` 的单一表达式。
>
> **注 3（目的端切片，阻塞 2）**：`o_ub` 的完整形状是 `(R, DW)`，源区域是 `(R, D_v)`；`D_v < DW` 时整 buffer ↔
> 整 buffer 的元素数不等，平台直接拒绝（`references/programming.md:37`；探针 F3：
> `Check failed: (analyzer->CanProveEqual(src_total, dst_total)) ... got src=4096 and dst=8192`）。
> 写成 `o_ub[0:R, 0:D_v]` 后 COMPILE OK（F2），生成 `asc_copy_gm2ub_align(..., 64, 128, ..., 512, 256)`
> —— 只写 64 行 × 128 B，目的行距仍是 256 B，pad 列保持 `T.clear` 的 0。
>
> **注 4（尾段，阻塞 3）**：`S % R ≠ 0` 时 `nfull` 个满 tile 走流水，尾段是**独立构建期代码段**，
> `nlast` 同时约束两条 `T.copy` 的源/目的 extent、`T.serial` 上界与写回 extent。探针 F4（S=65、R=64、`nlast=1`）
> 的生成代码：尾段 `asc_copy_gm2ub_align(..., 1, 256, ..., 256, 256)`、`asc_copy_ub2gm_align(..., 1, 4, ..., 4, 4)`
> —— 只读 1 行、只写 1 个 fp32。`nfull == 0`（`S < R`，如 `S=1`）时 `if tpb > 0` 在 trace 期被折叠掉，
> 只剩尾段（探针 F8：`S=1, H=1` COMPILE OK，生成 `asc_copy_gm2ub_align(..., 1, 256, ...)`）。
> **代价**：尾块几何取 `nsp = 1`（`grid = B·H`），且编译器会因该 buffer 在流水循环
> 之外也被写而**关闭多版本**（F4/F8 输出 "Ascend AutoSchedule did not enable multi-buffering for buffer `o_ub`"
> 警告）——尾块几何不追求流水收益，只保证正确性（只出现在非整除 `S` 的边界用例）。
>
> **注 5（`T.Pipelined` 的收益）**：探针生成代码显示装载并未提前到上一 tile 计算之前（§6.3），故流水收益属未实测量，
> 已列入 §1.6.3 裁决计划；本条与 v1 一致，未改变。


### 3.4 API 可行性确认

| API | 证据来源 | 状态 |
|-----|---------|------|
| `T.Kernel(N)`（一维、无 `threads`） | `.agents/platforms/toolchains/ascend/references/programming.md` 第 13 行；`ascend/language/kernel.py` | ✅ 探针 P1/P2/P4 编译通过 |
| `T.SimdVF()` + `T.simd.*` | 同上第 15 行；`.agents/platforms/toolchains/ascend/knowledge/repro/PL-950-simd-reduce.py` | ✅ 探针生成 `simd_inst::vlds_norm / vcvt / vmul / vadd / vcadd / vsts_norm` |
| `T.simd.vcvt(..., part=0/1)` | `ascend/language/simd.py`（`vcvt` 文档：`PART_EVEN/PART_ODD` 用于 widening） | ✅ 探针 fp16 与 bf16 双 dtype 均生成 `PART_EVEN/PART_ODD` |
| `T.copy`（3-D 区域固定中间索引 → UB，**目的端切片**） | 探针 P1（几何 H=8、D=128）与 F2（D_v=64、DW=128）：生成 `asc_copy_gm2ub_align(dst, src, 64, 256, …, 2048, 256)` / `(…, 64, 128, …, 512, 256)` = 64 行、每行 256 B / 128 B、源间距 2048 B / 512 B、目的行距 256 B；H=128 时源间距参数为 32768 B | ✅ 单条跨步描述符；**目的端必须显式切片**（F3 不切片 → `PlanMTECopy ... got src=4096 and dst=8192`，COMPILE FAIL） |
| `T.copy`（`d_ub[0:R, 0:1]` → GM） | 探针 P1：`asc_copy_ub2gm_align(…, 64, 4, …, 4, 32)` = 64 元素、源间距 32 B、目的连续 | ✅ 单条压缩描述符 |
| `T.copy`（尾段，extent = 静态 `nlast`） | 探针 F4（S=65、R=64、`nlast=1`）与 F8（S=1、`nfull=0`）：`asc_copy_gm2ub_align(…, 1, 256, …, 256, 256)`、`asc_copy_ub2gm_align(…, 1, 4, …, 4, 4)` | ✅ COMPILE OK；尾段只读 `nlast` 行、只写 `nlast` 个 fp32（无越界） |
| 构建期特化 `nvec` 段合并（`row_dot` helper） | 探针 F1/F2/F4（`nvec=1`）、F5（`nvec=2` 内联）、F6（`nvec=2` 经 helper）：生成 `vadd(vadd(simd_inst::vmul(...` 单一表达式 | ✅ COMPILE OK；反例 F7（`for v in range(nvec)`）→ `RuntimeError: Immutable variable 'acc' is used outside its defining region!` |
| `T.Pipelined` + `T.annotate_buffer_versions` + 嵌套 `T.SimdVF` | `.agents/platforms/toolchains/ascend/knowledge/repro/PL-950-vector-execution.py`（已验证形态）；本任务探针 P4/F1/F2 | ✅ 编译通过，生成双版本 UB（`(it & 1) * 8192` half 偏移） |
| `T.clear`（区域清零） | `tilelang.ascend.language.clear`（`hasattr` 确认存在）；本任务探针 F2/F3/F4 与 v1 的 P2–P4 系列均参与编译并通过 | ✅ 可用（本任务探针已编译通过，无设备）；模板 `.agents/platforms/toolchains/ascend/kernel_template.py` 未直接使用 → 属备注，不影响状态 |
| `T.simd.pset(32, "PAT_VL1")` / `vsts(..., extent=1)` | `PL-950-simd-reduce.py` 的已验证用法（含 `extent=1`） | ✅ 探针生成 `vsts_norm(total, …, 0, one)` |
| `T.reduce_sum`（高层二维归约） | **不采纳**：`.agents/platforms/toolchains/ascend/knowledge/numerical-boundaries.md` 的 `TRAP-950-simd-shared-reduce` 记录了 `T.SimdVF` 内二维 shared→shared 归约的输出布局差异 | ⛔ 弃用（§0.5） |
| `alloc_reducer/reducer_init/update/finalize`（SIMT 路径） | `PL-950-row-primitives.py` 已验证单行 64 元素用法 | ⛔ 弃用（每行一次 AllReduce，见 §1.6.1 A2） |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| `T.Kernel` 只接受一维网格、核级无 `threads` | **是**（源为二维 + 128 线程） | 一维 `grid = batch·heads·nsp`；线程域由 `T.SimdVF()` 取代（§0.6 R1） |
| UB 253952 B/core，auto multi-buffer 可能膨胀 | 是 | 66 KB/块（1.7× 膨胀后 112 KB）< 248 KB（§4.5） |
| 无 warp / `__shfl` / SMEM 归约语义 | 是 | 归约改为 `vcadd`（§0.6 R2） |
| 高层二维 `T.reduce_sum` 在 SimdVF 内的布局差异 | 是（若误用） | 明确不用，改显式 SIMD（§3.4） |
| 整 Buffer → 整 Buffer 的拷贝要求元素总数相等（`references/programming.md:37`） | **是**（`D_v < DW`，如 `D_v=64`） | 目的端写显式 region `o_ub[0:R, 0:D_v]`；`D_v=64` 由 op 级 smoke（`test_mla_dense_bwd_general_kernel`）覆盖，L0 专门验证 pad 路径（§8.3） |
| 尾轴对齐：16-bit 向量宽度 128、fp32 64 | 是 | `D_v = 128` 恰好 1 寄存器；一般 `D_v` 按 `DW = ceil(D_v/128)·128` 两侧零填充（`T.clear(o_ub)` 且 `T.clear(do_ub)`） |
| 动态 shape | 是（`T_q`、`B`） | 按 `(batch, seqlen, heads, dim_v, dtype)` 构建期特化（与源 `target_cache` 同策略），见 §4.6 |
| fast-math / FMA 收缩会改变乘积舍入 | 是 | 不开 `TL_ENABLE_FAST_MATH`（§7.3） |

#### 3.5.2 参考实现差异说明

| 差异项 | 参考实现（GPU） | 本项目（Ascend） | 转换方案 |
|--------|----------------|-----------------|---------|
| Kernel 结构 | `T.Kernel(heads, ceildiv(total,32), threads=128)` | `T.Kernel(grid)` 一维 + 块内 tile 流水 | §0.6 R1；探针 P4 编译通过 |
| 归约 API | `T.reduce_sum(fragment[D], rows, 1)`（warp 树） | `T.simd.vcadd`（64 lane 树） | §0.6 R2；`PL-950-simd-reduce` + 探针 P2 |
| 内存分配 | `alloc_fragment`（寄存器） | `alloc_shared`（UB）+ SIMD 向量寄存器 | §0.6 R3；66 KB 预算 |
| 类型提升 | `T.copy` 隐式 fp16→fp32 fragment | `vcvt` PART_EVEN/PART_ODD | §3.2 步骤 4 |
| 尾块处理 | 运行期 `if row < total` | 构建期 `S % R` 分支 + 静态 `nlast`（两条 copy / 行循环 / 写回三处 extent） | §6.4、探针 F4 |
| 搬入区域 | 整 tile fragment，无 region 概念 | 目的端显式 region `o_ub[0:R, 0:D_v]`（`DW > D_v` 必须） | §3.3 注 3、探针 F2/F3 |
| 输出写回 | 逐行谓词标量写 | 每 tile 一条压缩 DMA | §1.6.1 A3 |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `.agents/platforms/toolchains/ascend/knowledge/repro/PL-950-simd-reduce.py` | **极高** | 多行、每行 `vld → vcadd/vcmax → vsts(extent=1)`、32 B 行间距暂存、尾 lane mask —— 本设计 R2/A3 的直接原型 |
| `.agents/platforms/toolchains/ascend/knowledge/repro/PL-950-vector-execution.py` | 高 | `T.Pipelined` + 嵌套 `SimdVF` + `annotate_buffer_versions` + grid-stride + 尾块静态分支 |
| `examples/paged_cache_write/_paged_cache_write_kernel/_paged_cache_write_kernel.py` | 高 | ascend 向量算子的工程形态：一维 `T.Kernel`、`T.dynamic` 尾块、golden/L0–L2 分层测试组织 |
| `examples/moe_permute_align/_make_scatter_kernel/_make_scatter_kernel.py` | 中高 | 以 56 Vector 核为分核目标（`_NBLK_TARGET = 56`）、`T.Kernel + SimtVF` 合并上下文、构建期静态几何 |
| `.agents/platforms/toolchains/ascend/kernel_template.py` | 中 | 平台 kernel 骨架（`build/golden/check/run_L0..boundary/main` 结构，本设计的 §8.3 沿用其分层命名） |
| `examples/TileOPs/tileops/kernels/attention/mla_dense_bwd/_mla_dense_bwd_kernels.py` | 中 | 迁移 scaffold 中 `_delta_kernel` 的占位位置与 `_bwd_kernel` 的 `delta` 签名（接口对齐目标） |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `o` | `(B·S_q, H, D_v)` | `float16 \| bfloat16` | dense 扁平化视图；调用方传 `o.view(B·S_q, H, -1)` |
| `do` | `(B·S_q, H, D_v)` | 同 `o` | 同上 |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `delta` | `(B, H, S_q)` | `float32` | dense 布局，`out_idx=[2]`；与 `_bwd_kernel` 的 `delta` 入参逐元素一致 |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| `o_ub` | `(R, DW)`，R=64, DW=128 | 输入 dtype（16-bit） | UB（2 版本） | 本 tile 的 `o` 行块；有效列仅 `[0, D_v)`（`D_v=64` 时右半是 pad，由 `T.clear` 保证为 0） |
| `do_ub` | `(R, DW)` | 16-bit | UB（2 版本） | 本 tile 的 `do` 行块（同样两侧清零） |
| `d_ub` | `(R, 8)` | `float32` | UB（1 版本） | 行结果暂存（32 B 行间距，供 `vsts` + 压缩 DMA），2 KB |
| （寄存器）`vo/vd/o_lo/o_hi/d_lo/d_hi/acc/total` | 各 1 向量寄存器 | 16-bit×128 / fp32×64 | SIMD 向量寄存器 | 每行中间量，**不物化到 UB**（A1） |

> `D_v = 64`（op 级 smoke 几何）时 `DW` 仍是 128（pad 到 1 个完整向量寄存器），故 §4.5 的 UB 预算不变；
> 差別只是每条 `T.copy` 只搬/写 64 列（§3.3 注 3、探针 F2）。

### 4.4 内存搬运路径

```
GM o ──[T.copy: 跨步 DMA，R 段 × 256 B（D_v=64 时 128 B），源间距 H·D_v·2 B（H=8 时 2048/1024 B，H=128 时 32768/16384 B），
        目的 region [0:R, 0:D_v]，目的行距 DW·2 B]──► UB o_ub（ping-pong 2 版本）
GM do ─[同上]──────────────────────────────────────────────────────────────► UB do_ub（ping-pong 2 版本）
UB o_ub/do_ub ──[vld，整行 1 寄存器]──► SIMD 向量寄存器 ──[vcvt/vmul/vadd/vcadd]──► d_ub[i,0]（vsts，1 lane）
UB d_ub[:,0] ──[T.copy: 压缩 DMA，R 元素 × 4 B，源间距 32 B，目的连续]──► GM delta
（尾段：同样三条路径，但 extent 全为静态 nlast；无 L1 / L0C / 中间 GM 往返）
```

### 4.5 UB 内存预算

以 `R = 64`、`DW = 128`、16-bit 输入计（`D_v = 128` 与 `D_v = 64` 两种几何的 UB 占用**相同**，因为 `DW` 都取 128）：

| Buffer | Shape | dtype | 大小 (Bytes) |
|--------|-------|-------|-------------|
| `o_ub` × 2 版本 | (64, 128) | fp16/bf16 | 16384 × 2 = 32768 |
| `do_ub` × 2 版本 | (64, 128) | fp16/bf16 | 16384 × 2 = 32768 |
| `d_ub` | (64, 8) | fp32 | 2048 |
| **总计（手工核算）** | | | **67584 B = 66 KB** / 253952 B（248 KB）（UB，来源 `Ascend950PR_9579.ini` `ub_size=253952`） |
| 占用率 | | | 66/248 = **0.27**（67584/253952 = 0.266） |
| 按 1.7× 膨胀上界 | | | 114893 ≤ 253952 ✅ |
| 按 1.10–1.12× 膨胀（`CONST-capacity-910B2C` 的显式 alloc 结构口径） | | | 74342–75706 ✅ |

| 其余几何 | R | `o_ub`+`do_ub`（2 版本） | `d_ub` | 合计 |
|---|---|---|---|---|
| `D_v = 64`（op smoke）、`D_v = 128`（全部 manifest workload） | 64 | 65536 | 2048 | **67584 B（66 KB）** |
| `S=200`（`R=50`） | 50 | 51200 | 1600 | **52800 B（≈52 KB）** |
| `S=65`（`R=64`、尾段 `nlast=1`） | 64 | 65536 | 2048 | **67584 B（66 KB）**（尾段复用同一 buffer） |

> 膨胀系数来自 `CONST-capacity-910B2C`（910B2C / `tilelang 0.1.2+3a214cde`，**非本型号实测**），
> 仅作上界参考；即便按最悲观的 1.7× 仍有 55% 余量。`R ≤ 64` 时预算单调下降，故构建期只在该区间取 `R`。

### 4.6 动态轴定义

| 动态轴 | 声明方式 | 运行时范围 |
|---|---|---|
| `T_q`（= `B·S_q`） | 由构建期常量 `batch·seqlen` 决定；`o`/`do` 的第一维即该常量 | 编译实例内固定 |

**说明（重要）**：`op_requirements` 把 `T_q`、`B` 列为动态轴，但本算子的可执行语义是
“**按 `(batch, seqlen, heads, dim_v, dtype)` 特化编译**”——与源算子 `@target_cache(maxsize=32)` 的
`(batch, heads, dim_v, dense_seqlen, dtype)` 特化完全同策略：源算子把 `dense_seqlen`、`batch` 都当构建期常量，
只把 `total` 声明为 `T.dynamic` 是为了与 varlen 分支（`MLAVarlenBwdOp`，另一个 manifest 条目）共用工厂。
dense 调用下 `total ≡ batch·dense_seqlen` 恒成立（`MLADenseBwdKernel.forward` 的
`q.view(batch*seqlen_q, heads, -1)` 保证），因此本设计把三个维全静态化，换来：
① 无运行期边界判断（`S % R` 分支在构建期定死）；② DMA 描述符的段数/间距全为常量；③ 尾段可静态展开
（两条 `T.copy`、行循环、写回三处 extent 都是构建期常量）。
host 侧任何 shape 变化 → 重新编译（与源一致），故不存在“运行时 shape 未处理”的路径。

### 4.7 JIT 配置

```python
@tilelang.jit(out_idx=[2], target="ascend")   # o, do 入；delta 为第 3 参数，由 out_idx 分配
def kernel():                                  # 几何由闭包常量给出（构建期特化）
    ...
```

不设 `pass_configs`（尤其**不设** `TL_ENABLE_FAST_MATH`，理由见 §7.3），不设额外 `compile_flags`。

---

## 5. Tiling 策略

### 5.1 计算类型

**类型：纯 Vector**（SIMD）。判定依据：算子只含“逐元素乘 + 行内归约 + 数据搬移”，无矩阵乘、无跨行依赖；
Cube 单元不参与（§1.6.0 R4 候选 5 已淘汰）。

### 5.2 Block 划分

```python
VL   = 128                       # 16-bit 向量宽度（2048 bit / 16 bit）
DW   = ceil(D_v/VL) * VL         # UB 行宽；D_v=128 与 D_v=64 都取 128
R    = 构建期选：S 在 [16, 64] 内的最大因子；无则 R = 64
nfull = S // R                   # 满 R 行 tile 数
nlast = S - nfull * R            # 尾段行数（0 = 无尾块）
nt   = ceil(S / R)               # 每个 (b,h) 的逻辑 tile 数（含尾段）
nsp  = 1 if nlast else (nt 的最小因子 d 使 batch*heads*d >= 224；无则 nsp = nt)
tpb  = nfull // nsp              # 每 block 的满 tile 数（无尾块时 = nt//nsp）
grid = batch * heads * nsp
```

`block_M = R`（行方向），`block_N = D_v`（value 方向，整行一个向量寄存器，**不切分**）。
有尾块时取 `nsp = 1`：这样「满 tile 循环上界」与「尾段三处 extent」同时是构建期常量，且无运行期谓词
（§3.3 注 4）；代价是尾块几何的 `grid = B·H` 可能小于 56 核（只影响非整除 `S` 的边界用例）。

| workload（manifest `MLADenseBwdOp` / 测试用例） | B,S,H,D_v | R | nt | nsp | grid | tpb | UB/块 |
|---|---|---|---|---|---|---|---|
| `ds-v3-4k` | 1,4096,128,128 | 64 | 64 | 2 | 256 | 32 | 66 KB |
| `ds-v3-b2-8k` | 2,8192,128,128 | 64 | 128 | 1 | 256 | 128 | 66 KB |
| `kimi-k2-8k` | 1,8192,64,128 | 64 | 128 | 4 | 256 | 32 | 66 KB |
| `ds-v2-lite-3k` | 4,3072,16,128 | 64 | 48 | 4 | 256 | 12 | 66 KB |
| manifest smoke（`mla-dense-bwd-smoke`） | 1,512,8,128 | 64 | 8 | 8 | 64 | 1 | 66 KB |
| `test_mla_dense_bwd` 的 B=2,S=256,H=8 形态 | 2,256,8,128 | 64 | 4 | 4 | 64 | 1 | 66 KB |
| **op smoke `test_mla_dense_bwd_general_kernel`（D_v=64）** | 1,256,4,64 | 64 | 4 | 4 | 16 | 1 | 66 KB（`DW=128`，走 pad 路径） |
| 边界 `S=200`（不被 2 的幂整除） | 1,200,8,128 | **50** | 4 | 4 | 32 | 1 | 52 KB |
| 边界 `S=128` | 1,128,8,128 | 64 | 2 | 2 | 16 | 1 | 66 KB |
| 边界 `S=65`（尾块 `nlast=1`） | 1,65,8,128 | 64 | 2 | **1** | 8 | 1（+尾段） | 66 KB |
| 边界 `S=1`（只有尾段） | 1,1,1,128 | 64 | 1 | **1** | 1 | 0（+尾段 `nlast=1`） | 66 KB |

（测试用例的 `heads`/`dim_v` 取各用例实际几何；`R=50` 说明 `R` 不必是 2 的幂，
只要整除 `S` 且 `R ≤ 64`，UB 行宽仍是 `DW=128`。`D_v=64` 与 `D_v=128` 的 UB 占用相同。）

### 5.3 约束分析

- **对齐约束**：行宽 `D_v·2 B = 256 B` = 2 个完整 128 B DMA 段 ✅；`vld` 整行 = 1 个 2048-bit 向量寄存器
  （16-bit ×128）✅；fp32 中间量 = 64 lane ✅；`d_ub` 行间距 32 B（8 个 fp32）与 `PL-950-simd-reduce`
  的已验证布局一致 ✅。
- **UB 容量**：66 KB / 248 KB（§4.5，含 1.7× 膨胀上界仍余 55%）。
- **L0 容量**：不使用 Cube → 不适用。
- **段数/指令开销**：每 tile 2 条装载描述符 + 1 条写出描述符；每行 11 条向量指令（§1.6.0 估算）。

### 5.4 注意事项

- **非整除处理**：`S % R ≠ 0` 时满 tile 走流水、尾段用静态 `nlast = S − nfull·R`（两条 `T.copy`、行循环、
  写回三处 extent，§6.4、§3.3 注 4），**不做运行期 mask**；`DW != D_v`（如 `D_v=64`）时 pad 列由
  **两侧** `T.clear` 清零（只清一侧会让另一侧的未初始化 pad lane 以 `0·Inf = NaN` 污染该行）。
- **`B·H·nsp` 不能被 56 整除**时的波次损失：如 256 块 / 56 核 = 4.57 → 5 波（利用率 91%）。
  这是“每 block 等量 tile（nsp 取 `nt` 的因子）”与“波次整对齐”的取舍，本设计优先前者（等量负载、无 straggler）；
  若 Stage 4 实测显示波次损失显著，可改 `nsp` 使 `grid` 贴近 56 的整数倍（属于参数级调优，不改结构）。

### 5.5 分核策略（物理核数适配）

三要素判定标准与实查要求的权威版本：`.agents/skills/_shared/standards/core-split-strategy.md`（本任务按 §1 三要素与 §2.1 执行）。

**① 逻辑核数计算**

- 逻辑任务数（tile 数）`T_total = batch · heads · nt = batch · heads · ceil(S/R)`：
  `ds-v3-4k` → 1·128·64 = **8192**；`ds-v3-b2-8k` → 2·128·128 = **32768**；`kimi-k2-8k` → 1·64·128 = **8192**；
  `ds-v2-lite-3k` → 4·16·48 = **3072**；smoke → 2·8·4 = **64**。
- 分核后逻辑核数（= 启动的核数）`grid = batch · heads · nsp`：上表分别为 **256 / 256 / 256 / 256 / 64**，
  每核处理 `tpb = nfull / nsp` 个满 tile（32 / 128 / 32 / 12 / 1）。
- 有尾块的边界几何取 `nsp = 1`（`grid = B·H`，如 `S=65,H=8` → 8 核；`S=1` → 1 核）。这是「静态 extent 优先」的
  降级：尾块几何只出现在非整除 `S` 的边界用例，manifest 与 op smoke 几何全部 `S % R == 0`，不受影响。

**② 物理核数依据（实查记录）**

| 查询 | 命令 / 来源 | 实际返回值 |
|---|---|---|
| 平台任务上下文（本机实跑） | `python3 .agents/tools/statectl.py platform --task-dir examples/mla_dense_bwd/_delta_kernel` | `hardware.hardware_id = "ascend-950pr-9579"`；`observed_cores = {"cube": 28, "vector": 56}`；`core_source = "Ascend950PR_9579.ini [SoCInfo] ai_core_cnt=28, cube_core_cnt=28, vector_core_cnt=56; runtime probe rtGetAiCoreCount()=28 on Ascend950PR_9579, 2026-11-23"` |
| 型号配置文件（只读复核） | `grep -n "core_cnt" /usr/local/Ascend/cann-9.2.1/x86_64-linux/data/platform_config/Ascend950PR_9579.ini` | `ai_core_cnt=28`、`cube_core_cnt=28`、`vector_core_cnt=56`（第 20–22 行）；`ub_size=253952`、`l1_size=524288`、`l0_c_size=262144`（第 50/51/58 行） |
| `torch.npu.get_device_properties` | **本次未执行**：本机仅 1 张卡且被同批次另一算子的 Stage 3 占用，设计期约定只做无卡编译/静态探针，直接初始化设备上下文会与并行的设备任务争用 | 以平台已记录的 runtime probe 结果（上一行）为准，不猜测数值 |

**目标型号：`ascend-950pr-9579`**（dav-3510，28 AI Core / 56 Vector Core，UB 248 KB）。
本算子为纯 Vector kernel，故分核目标取**56 个 Vector 核**（与同仓 `examples/moe_permute_align/_make_scatter_kernel/_make_scatter_kernel.py`
的 `_NBLK_TARGET = 56` 取值一致）。若硬件实际以 28 个 AI Core 为 block 调度粒度（每核 2 个 AIV 子块），
则同一 `grid` 仍成立、波数减半，结论不变（属未文档化假设，列入 §9.1）。

**③ 规模判定与分核方案**

- **规模判定：极大规模**。逻辑任务数 3072–32768 个 tile，远超 56 个 Vector 核（`grid` 若取逻辑任务数则
  每核要串行 55–585 个 tile，且 block 调度开销逐块重复）。因此**不能用“一 tile 一 block”直接映射**，
  按标准 §1 要素③走“固定启动内核数、核内分批处理多个逻辑块任务”的分核方案。
- **分核方案：核内串行（persistent 形态）**。固定 `grid = batch·heads·nsp`（每核 `tpb` 个 tile），
  块内用 `T.Pipelined(tpb, num_stages=2)` 迭代并配合 `T.annotate_buffer_versions` 双缓冲；
  `nsp` 的取法在 `nt` 允许时使 `grid ≥ 224 ≈ 4×56`（4 波以上，保证负载均衡；小 workload 取 `nsp = nt`
  即 `grid = B·H·nt`），且 `nsp` 整除 `nt`（每核等量 tile，无 straggler）。核内循环边界全部是构建期常量，满足本平台“核内循环上界静态化”的限制。
- **对齐说明**：`grid = 256` 与 56 的比值 4.57 → 5 波（91% 利用率）；`grid = 64`（smoke）→ 2 波。
  二者都由 `B·H` 的整数因子约束（`grid` 必须能被 `B·H` 整除才能保持 `(b,h)` 独占），故不强求 56 的整数倍；
  如 Stage 4 需要，可改 `nsp` 使 `grid` 更贴近整数倍（参数级）。
- **本方案与备选的关系**：备选（§1.6.3 裁决计划）正是“一 tile 一 block”的**逻辑核超发**形态
  （`grid = batch·heads·nt`，8192–32768 块，靠硬件波调度）；两者由 Stage 4 A/B 实测裁决，
  阈值：备选 latency < 主选 × 0.9 才翻转。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| 核网格 | block 级并行（一维） | `T.Kernel(grid)`，`grid = batch·heads·nsp` | 每核独占一个 `(b, h)` 的行切片 |
| 满 tile（行块）方向 | 核内串行 + 流水 | `T.Pipelined(tpb, num_stages=2)`（`tpb > 0` 的构建期分支） | 流式消费行块，装载与计算可重叠（双版本 UB） |
| 行（归约实例）方向 | 核内串行 | `T.serial(R)`；尾段 `T.serial(nlast)`（`T.SimdVF` 内） | 每行一次整行向量归约（V9：无跨行批量归约 API，§1.6.2） |
| 行内 `d` 方向 | **向量化** | `T.simd.vld/vcvt/vmul/vadd/vcadd` | `D_v=128` = 1 个向量寄存器；`nvec>1` 时按寄存器分段并在**构建期**展开（§3.3 `row_dot`） |
| 尾段 | 核内串行（构建期独立代码段） | 两条 `T.copy` + `T.serial(nlast)` + 压缩写回，extent 全静态 | `S % R ≠ 0` 时只此一份代码路径，无运行期谓词 |
| 元素级 | 不使用 `T.Parallel` | — | `SimdVF` 无线程域，元素并行由向量 lane 承担（§2.3） |

### 6.2 循环伪代码

见 §3.3（同一份伪代码即调度结构）：`T.Kernel(grid)` → 构建期分支（`tpb > 0` 的满 tile 流水段 + `tail` 尾段）
→ `T.SimdVF` 内 `T.serial(R)` / `T.serial(nlast)`。

### 6.3 流水线优化

- `num_stages = 2`，`T.annotate_buffer_versions({o_ub: 2, do_ub: 2})`；`d_ub` **不**多版本（写后即读，同迭代内完成）。
- 探针 P4/F1/F2 的生成代码显示：编译器为 `PIPE_V/PIPE_MTE2/PIPE_MTE3` 插入了成对的
  `asc_sync_notify/asc_sync_wait`（event 0/1 对应两个 UB 版本，UB 地址出现 `(it & 1) * 8192` 的版本偏移），
  但**本 tile 的计算并未被提前到上一 tile 的装载之后**（装载与计算仍同迭代、以 wait 分隔）。因此“双缓冲是否
  真的带来重叠”属未实测量，已列入 §1.6.3 的裁决计划（主选仍保留 `T.Pipelined`，因为它的代价为零且不排除
  编译器在更大 `tpb` 下更积极地重排）。
- **尾段几何的例外**（探针 F4 的新发现）：因为 `o_ub`/`do_ub` 在流水循环**之外**的尾段也被写，编译器的
  AutoSchedule 会关闭这两个 buffer 的多版本（F4 输出 “did not enable multi-buffering for buffer `o_ub` …
  written before it is read” 警告）。这只影响 `S % R ≠ 0` 的边界几何——它们**不追求流水收益**，
  正确性由静态 extent 保证；`S % R == 0` 的全部 workload 不受影响。
- Stage 4 若需更深的流水，可在同一结构上试 `num_stages = 3`（UB 66 → 98 KB，仍在预算内）。

### 6.4 尾块处理

| 场景 | 处理 |
|---|---|
| `S % R == 0`（本任务全部 manifest workload 与 `S=128/256` 用例） | 无尾块，`nfull = nt = S/R`，`tpb = nt/nsp`，行循环上界恒为 `R` |
| `S % R ≠ 0`（如素数 `S`、`S=65`、`S=1`） | 构建期取 `nfull = S // R`、`nlast = S − nfull·R`、`nsp = 1`：`nfull` 个满 tile 走流水，尾段是**独立构建期代码段**，两条 `T.copy`、`T.serial` 上界与写回 extent **全部用静态 `nlast`**（探针 F4：`asc_copy_gm2ub_align(…, 1, 256, …)`、`asc_copy_ub2gm_align(…, 1, 4, …)`）；越界行既不读也不写。**没有**运行期谓词（v1 的 `if s0+i < S` 已删除） |
| `DW != D_v`（`D_v=64`，op smoke 几何） | `DW = ceil(D_v/128)·128 = 128`；构建期分支里 `T.clear(o_ub)` **且** `T.clear(do_ub)` 使 pad 列两侧都为 0；两条 `T.copy` 的目的端显式切片 `[0:R, 0:D_v]`（元素数相等，探针 F2）→ 零项严格不改变和（无需 mask） |
| `D_v > 128`（本任务不出现，`refusal` 允许 `dim_v % 16 == 0` 的任意值） | `nvec = DW/128` 段，各段部分和在**构建期展开成一个 `vadd` 表达式**（§3.3 `row_dot`，探针 F5/F6 COMPILE OK），再做唯一一次 `vcadd`；**不使用**运行期循环携带的向量累加器（v1 的写法在工具链下不可构建，探针 F7） |

---

## 7. 同步策略

### 7.1 同步模式

**模式：自动同步**（编译器插入跨 pipe 的 flag）。本设计**不写**任何手动 `T.set_flag/wait_flag`。

### 7.2 同步点说明（探针 P4 的实测生成代码）

| 位置 | 编译器生成 | 理由 |
|------|-----------|------|
| block 起始 | `asc_sync_notify(PIPE_V, PIPE_MTE2, event 0/1)`、`asc_sync_notify(PIPE_MTE3, PIPE_V, 0)` | 初始化双版本缓冲的 flag，避免首个 tile 的装载与“上一轮不存在的计算”死等 |
| tile 迭代开始 | `asc_sync_wait(PIPE_V, PIPE_MTE2, event = step)` | 等该版本 UB 的上一轮读者（Vector）读完，才允许 MTE2 覆盖写 |
| 装载完成后 | `asc_sync_notify(PIPE_MTE2, PIPE_V, event = step)`；Vector 侧 `asc_sync_wait(PIPE_MTE2, PIPE_V, event = step)` | 等 DMA 落地后才允许 `vld` |
| 暂存写出前 | `asc_sync_wait(PIPE_MTE3, PIPE_V, 0)` / `notify(PIPE_V, PIPE_MTE3, 0)` | `d_ub` 单版本：下一 tile 的 `vsts` 必须等上一次压缩 DMA 读完 |
| 计算完成后 | `asc_sync_notify(PIPE_V, PIPE_MTE2, event = step)` | 释放该版本 UB 给下一轮装载 |
| block 结束 | 尾部 3 个 `asc_sync_wait` | 收敛所有未完成的跨 pipe 事件 |

结论：`T.copy`（MTE2/MTE3）与 `T.simd.*`（PIPE_V）之间的依赖由编译器按 buffer 版本自动加锁，设计期无需手工同步；
`d_ub` 单版本带来的串行已在 §6.3 说明（若 Stage 4 要为 `d_ub` 加双版本，只需把 2 加进 `annotate_buffer_versions`）。

### 7.3 pass_configs 配置

```python
pass_configs = {}        # 不开 TL_ENABLE_FAST_MATH
compile_flags = []       # 不额外传 -O3/-DENABLE_BF16（ascend target 的默认后端已处理 dtype）
```

**为什么关掉 fast-math**：源算子开了 `TL_ENABLE_FAST_MATH: True`，但本算子没有任何超越函数（无 `exp/log/sqrt`），
fast-math 唯一可能的副作用是把 `vmul + vadd` 收缩成 FMA 或做其它重排——那会改变乘积的舍入位置（乘积不再
先舍入到 fp32），使 A1 的“逐位等价”结论失效。因此显式不开启，保持“乘积四舍五入到 fp32 → 再累加”的语义
（`verify_equiv.py` 的 A1 用例正是按此语义构造）。

---

## 8. 验证方案

### 8.1 Golden 函数

> 依据 **§0.1 语义**（不是 §0.6 的 NPU 算法）：与源 docstring
> “`delta[.., h, row] = sum_d o[row, h, d] * do[row, h, d]` in float32” 逐字对应，朴素实现、无分块、无向量化。

```python
def golden_delta(o, do):
    """delta[b, h, s] = sum_d o[b, s, h, d] * do[b, s, h, d], float32.

    o, do: (B, S, H, D_v) 的 fp16/bf16 原始张量（测试用例持有 4-D 形态；
    kernel 入参是 (B*S, H, D_v) 的视图，由 host 侧 reshape 对应，不是语义差异）。
    返回 (B, H, S) 的 float32：转成 fp32 后逐元素乘，再沿最后一维求和，最后把
    (B, S, H) 转置成 (B, H, S)——不复制任何分块/归约顺序的实现细节。
    """
    prod = o.float() * do.float()                      # (B, S, H, D_v) fp32 乘
    return prod.sum(dim=-1).transpose(1, 2).contiguous()   # (B, H, S) fp32
```

（设计期 `verify_equiv.py` 另用 fp64 精确参照来评估分组误差；那属于设计侧证据，不替代本节 golden。）

### 8.2 精度标准

| dtype（输入） | 输出 dtype | atol | rtol | 依据 |
|---|---|---|---|---|
| float16 | float32 | 1e-4 | 1e-4 | `delta` 本身是 fp32 输出，两侧都在 fp32 域累加（模板 fp32 标准） |
| bfloat16 | float32 | 1e-4 | 1e-4 | 同上；bf16 输入提升到 fp32 无损，误差只来自累加分组 |

补充：本算子作为 `MLADenseBwdOp` 的一环，最终 dq/dk/dv 的容差沿用
`examples/TileOPs/tests/ops/test_mla_dense_bwd.py` 的 `atol = max(5e-3, rtol)`（`_RTOL` = fp16 1e-3 / bf16 1.6e-2）。
`delta` 若在 fp32 下与 golden 的偏差 ≤ 1e-4，经 `ds = p·(dp − δ)` 传递后不会成为该容差的瓶颈
（A2 实测：树形归约比顺序累加更接近真值）。

### 8.3 L0 门槛测试计划

| 层级 | 形状 | dtype | 覆盖的设计点 | 目的 |
|---|---|---|---|---|
| **L0** | `B=1, S=128, H=4, D_v=128`（`R=64, nt=2, nsp=2, grid=8`，每核 1 个 tile） | fp16 + bf16 | 满 tile 流水段、双缓冲、压缩写出（`§3.3` 主路径） | 最小可运行：验证 `vld/vcvt/vmul/vadd/vcadd/vsts` 链；两 dtype 必须都过 |
| **L0** | **`B=1, S=256, H=4, D_v=64`**（与 op 自身的 `test_mla_dense_bwd_general_kernel` 同几何；`DW=128`、`nt=4, nsp=4, grid=16`） | **bf16**（op smoke 的实际 dtype）+ fp16 | **pad 路径**：目的端切片 `[0:R,0:64]`、两侧 `T.clear`、pad 列零项不污染和（阻塞 2 的数值闸门） | 保证 Stage 5 的单算子报告不会在这条路径上首次暴露问题 |
| **L0** | `B=1, S=1, H=1, D_v=128`（单行，`nfull=0`、只有尾段 `nlast=1`） | fp16 | `tpb = 0` 的构建期分支 + 尾段单行 | 最小边界：验证无越界读、无空循环问题 |
| L1 | manifest smoke：`B=1, S=512, H=8, D_v=128`（`mla-dense-bwd-smoke`） | fp16 + bf16 | `grid=64`、`tpb=1`（无核内循环） | 代表性小规模，与 manifest 计分口径一致 |
| L1 | `B=2, S=256, H=8, D_v=128`（与 `test_mla_dense_bwd` 的 smoke 几何同族） | fp16 + bf16 | 多 batch 的 `(b,h)` 映射 | 代表性小规模，`grid=64` |
| L2 | `ds-v3-4k`：`B=1, S=4096, H=128, D_v=128` | bf16 + fp16 | manifest 主力 workload，`grid=256, tpb=32`，源间距 32768 B | 流量主导口径的实测对账（§1.6.0 估算） |
| L2 | **`ds-v3-b2-8k`：`B=2, S=8192, H=128, D_v=128`** | bf16 | **长核内循环 `tpb=128`**、`grid=256`、`nsp=1` | 流水/尾段收敛与波次利用率的压力点（§6.3 的未实测量在此暴露） |
| L2 | `kimi-k2-8k`：`B=1, S=8192, H=64, D_v=128` | bf16 | `nsp=4, grid=256, tpb=32` | 少头数、长序列 |
| Boundary | `S=200`（`R=50`）、`S=320`（`nt=5, nsp=5`）、**`S=65`（`nfull=1, nlast=1` 尾段）**、`S=1`+`H=1` 组合 | bf16 | **尾段三处 extent 全静态**（阻塞 3 的数值闸门）；非整除、极小头数 | 越界写会污染相邻 head/batch，必须以此闸门拦住 |
| Boundary | 全零输入（`delta` 应全 0）、含 `inf`/`NaN` 的输入行（与 golden 同为非有限，`equal_nan=True` 比较） | fp16 | 非有限传播与零行 | 与源语义一致 |

判定：`torch.testing.assert_close(kernel(o, do), golden_delta(o, do), atol=1e-4, rtol=1e-4, equal_nan=True)`；
L0 必须两 dtype 全过才允许进入 L1/L2（与 `tilelang-op-develop` 的分层契约一致）；`D_v=64` 的 L0 用例是
阻塞 2 的**唯一数值证据**（编译期只能证明描述符合法，不能证明 pad 列为 0）。

---

## 9. 风险点与注意事项

### 9.1 已知约束与风险

| # | 风险 | 影响 | 依据 / 缓解 |
|---|------|------|------------|
| R1 | 跨步描述符（`R` 段 × 256 B、源间距 `H·D_v·2 B`，H=128 时为 32768 B——该跨度落在常规 DMA stride 字段范围内属未文档化假设）的 MTE2 有效带宽低于连续描述符 | 流量项估算偏乐观 | 探针已确认是**单条**描述符（不是 R 条 1-D 拷贝）；实际带宽差为 910B2C 常数未覆盖的量 → §1.6.3 裁决计划（备选形态）；Stage 4 以实测对账 §1.6.0 的估算下界 |
| R2 | 向量发射项可能高于估算（`vcvt`/`vcadd` 多周期） | 若发射项 > 流量项则变成向量受限（估算上界 100–130 µs vs 215–262 µs 流量项，仍有余量） | §1.6.0 估算已给出上下界；Stage 4 若发现向量受限，可选手段：`num_stages` 加深、或改用解包装载（`simd.py` 的 `_VLD_POSTUPDATE_DISTS` 列有 `UNPK_B16`/`UNPK_B32`，但仅见于 `post_inc` 形态，语义未验证，不在本设计内） |
| R3 | `T.copy(d_ub[0:R, 0:1], …)` 的压缩写出语义（源间距 32 B）只做了**编译期**验证，未在设备上验证数值 | 若描述符间距错，`delta` 会错位 | 探针生成的描述符参数（64 元素 / 源间距 32 B / 目的连续）与设计意图一致；L0 用例即为该路径的数值闸门；退路：逐行 `T.copy(d_ub[i, :1], delta[.., i:i+1])`（R 条 1-D 拷贝，慢但确定） |
| R4 | `d_ub` 行间距 32 B 的必要性来自 `PL-950-simd-reduce` 的实测经验（`(2,8)` 分配），不是文档化契约 | 若实际只写 4 B，则多占 1.5 KB（无害）；若实际写 32 B，间距必须 ≥32 B（已满足） | 采用已验证形态；UB 代价可忽略 |
| R5 | 抵消行（Σabs(p) 大而 abs(Σp) 小）的 fp32 累加绝对误差 ~1e-6·Σabs(p) | 该行的 `delta` 相对误差大 | 源算子同 dtype 语义、同样存在（§1.6.1 注）；不属于迁移引入的偏差 |
| R6 | L2 驻留使 bench 读数优于 HBM 口径 | 估算与实测偏差 >2× 时误判“设计失准” | `CONST-mte2-degradation` 的 L2 口径已记录该现象；Stage 4 对账时先看 `msprof` 的读带宽口径再归因 |
| R7 | `grid` 非 56 的整数倍（4.57 波） | 尾波 91% 利用率 | §5.5 已量化；Stage 4 可调 `nsp` |
| R8 | 分核目标“56 Vector 核”假设 block 调度粒度为 AIV | 若为 28 AIC 粒度，波数减半（不影响正确性，性能结论需重估） | §5.5 ② 已标注为未文档化假设；同仓 `moe_permute_align` 采用同一取值 |
| R9 | `T_q`/`B` 静态化（按 shape 特化编译） | host 侧 shape 变化需重编译 | 与源 `target_cache` 同策略（§4.6）；若上层要“一份 kernel 覆盖多 shape”，需回到动态 `total` 并把尾处理改为运行期 mask（结构变更，非本设计范围） |
| R10 | 尾块几何（`S % R ≠ 0`）因尾段在流水循环外也写同一 buffer，编译器**关闭** `o_ub/do_ub` 的多版本（探针 F4 警告），且 `nsp = 1`（`grid = B·H`，可能 < 56 核） | 尾块几何没有流水重叠、波次利用率低（性能，非正确性） | §6.3/§5.2 已记录；只影响非整除 `S` 的边界用例，manifest 与 op smoke 几何全部 `S % R == 0`。若 Stage 4 认为必要，可为尾段单独分配小 buffer（`(nlast, DW)`）以解除该限制——属结构变更，需回写设计 |
| R11 | `D_v > 128`（`nvec > 1`）只做了**编译期**证据（F5/F6），无设备数值验证；`nvec > 2` 未单独探针 | 该几何的数值正确性未经实测 | `row_dot` 的展开对任意 `nvec` 同构（python 层循环拼接表达式），F6 已验证 `nvec=2`；`refusal` 允许的 `dim_v` 均为 16 的倍数，实际 workload 全为 64/128，故列入待确认项；L0 的 `D_v=64` 用例覆盖 pad 路径，不含 `nvec>1` |
| R12 | `T.clear` 的零填充依赖“UB 分配后再清零、且两条 `T.copy` 只写有效列”的顺序 | pad 列若被后续写脏，零项假设失效 | §3.3 的代码顺序（clear → copy into `[0:R, 0:D_v]`）即探针 F2 的顺序；L0 `D_v=64` 用例是数值闸门 |

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|---------|
| `vsts` 写到相邻行 | `d_ub` 行间距 < 32 B | 行结果互相覆盖 | 固定 `d_ub[R, 8]`（32 B 间距），不要改成 `d_ub[R]` |
| **只对 `o_ub` 做 `T.clear`** 而 `D_v < DW` | `D_v=64`（op smoke 几何） | 另一侧的 pad lane 是未初始化 UB，`0 · (Inf/NaN bit 模式) = NaN` → 整行 `delta` 变 NaN | **两侧都清零**：`T.clear(o_ub)` 且 `T.clear(do_ub)`（§3.3、探针 F2）；或改用 `T.simd.update_mask(D_v, 32)` 只算有效 lane |
| **目的端不切片**的整 tile 搬入 | `D_v < DW`（如 `D_v=64`）且写 `T.copy(src_region, o_ub)` | 前端直接拒绝：`PlanMTECopy: ... got src=4096 and dst=8192`（探针 F3） | 目的端写显式 region `o_ub[0:R, 0:D_v]`（`references/programming.md:37`） |
| **把向量值当循环携带量** | prim_func 体内写 `for v in range(nvec)` 并在循环内 `acc = T.simd.vadd(acc, …)` | `RuntimeError: Immutable variable 'acc' is used outside its defining region!`（`nvec=1` 也触发，探针 F7） | `nvec` 段在**构建期**用 python helper 拼成一个表达式（§3.3 `row_dot`，探针 F5/F6） |
| **尾块沿用 `R` extent** | `S % R ≠ 0`（如 `S=65`）时两条 `T.copy` / 行循环用 `R` | 越界读 63 行 + 越界写 `delta[b,h,64:128]`，**编译器不报错** | 尾段三处 extent 全用静态 `nlast`（§3.3 注 4、探针 F4）；`nsp = 1` |
| 用 `T.reduce_sum` 做行归约 | 误按 GPU 习惯 | `TRAP-950-simd-shared-reduce` 的输出布局错位 | 保持 `vcadd` 显式归约 |
| 打开 fast-math | 复制源算子 `pass_configs` | FMA 收缩改变乘积舍入，A1 结论失效 | §7.3：不设 `pass_configs` |
| `R` 不整除 `S` 却按整除展开 | 构建期规则漏判 | 越界读 / 少算 | §5.2 的 `R` 选择规则 + §6.4 尾段 |
| `T.Kernel` 传二维网格或 `threads=` | 照抄 GPU 源码 | 前端直接拒绝 | §0.5：一维网格 + `T.SimdVF` |

### 9.3 特殊场景处理

- 极小 shape（`S=1`、`H=1`）：`nfull=0`、`nlast=1`、`tpb=0`、`grid=B·H`，只有尾段（行循环 1 次）；无特殊分支。
- 尾块（`S % R ≠ 0`）：`nsp = 1`，满 tile 段 + 静态尾段（§6.4），无运行期谓词；编译器关闭该几何的多版本（§6.3）。
- 极大 `S`（`S ≥ 8192`）：`nt ≥ 128`，`nsp` 增大以维持 `grid ≈ 256`；每核 tile 数 ≤ 128（循环上界静态）。
- 输入全零：`delta` 全零（乘法与加法均保零）——L0 有专门用例。
- 非有限输入：与 golden 一致地传播（`equal_nan=True` 比较）。

---

## 10. 交付清单

### 10.1 目录结构

```
examples/mla_dense_bwd/_delta_kernel/
├── DESIGN.md                          # 本文（v2）
├── REVIEW.md                          # Stage 2 检视报告（v1 不通过 + 本版逐项闭环依据）
├── verify_equiv.py                    # D-1 等价性机器验证（A2 已改为实现分组；结果见 §1.6.1）
├── history_version/
│   ├── design_probe_compile.py        # D-3 设计期编译探针 v1（无卡，P1/P2/P4 全部编译通过）
│   ├── design_probe_compile_v2.py     # D-3 设计期编译探针 v2（F1–F8：三个阻塞项的闭环证据）
│   └── design_v0.md                   # v1 设计备份（revision 前）
└── _delta_kernel.py                   # 待 Stage 3 生成（kernel + golden + L0/L1/L2/boundary）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | ✅ 已完成（v2 / revision 1） | 本设计文档（§0 迁移决策 + 修订说明 + §1–§10） |
| `verify_equiv.py` | ✅ 已完成并执行（v2） | A1/A2/A2b/A3 × 12 用例全通过（§1.6.1） |
| `history_version/design_probe_compile.py` | ✅ 已完成并执行 | v1 关键 API 编译证据（§3.4 的 P 系列） |
| `history_version/design_probe_compile_v2.py` | ✅ 已完成并执行 | F1/F2/F4/F5/F6/F8 COMPILE OK；F3/F7 为负向对照（§3.3/§3.4） |
| `_delta_kernel.py` | ⬜ 待实现 | Stage 3：kernel（§3.3 伪代码）+ `golden_delta`（§8.1）+ 分层测试（§8.3） |
| `RETROSPECTIVE.md` | ✅ 已完成（含 revision 1 章节） | 本任务的 Stage 1 复盘（skill Phase 8） |

### 10.3 命名规范

- 项目目录：`examples/mla_dense_bwd/`（与 `MLADenseBwdOp` 的三个 launch 子目录并列）。
- 算子目录：`_delta_kernel`（与源函数名一致，便于 Stage 5 集成时定位 `_compile` 里的 `prep`）。
- 实现文件：`_delta_kernel.py`。

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md v2）+ 算法调研/优化分析 + 等价性验证 + 两轮编译探针
2. ⬜ `golden_delta`（§8.1）
3. ⬜ kernel 实现（§3.3 伪代码：构建期特化 `nvec`、目的端切片、尾段静态 `nlast`；按 §5.2 的构建期几何规则取 `R/nsp/tpb`）+ L0 两 dtype（**含 `D_v=64` pad 用例**）
4. ⬜ L1/L2/Boundary 分层（§8.3），随后交 Stage 2 检视与 Stage 3 精度闸门
