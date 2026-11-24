# `_silu_mul_fused_kernel` 算子设计文档

> **迁移任务（harness 子模式），首次设计 v1**。源算子：
> `/workspace/TileOPs/src/tileops/_kernels/moe/shared_expert_mlp.py` 的
> `_silu_mul_fused_kernel`（GPU TileLang，CUDA target，第 24–64 行）；同仓抽取件
> `examples/TileOPs/tileops/kernels/moe/shared_expert_mlp/_shared_expert_mlp_fwd_kernels.py`
> 为其逐字副本。
>
> 目标平台：`ascend-950pr-9579`（dav-3510，28 AI Core / 56 Vector Core，UB 253952 B /
> L1 524288 B / L0C 262144 B），工具链 `ascend`（TileLang ascend 0.1.15 + CANN 9.2.x，
> `target="ascend"`，`import tilelang.ascend.language as T`）。平台选择由
> `statectl platform --task-dir` 落库（2026-11-23）。
>
> **设计范围**：仅 `_silu_mul_fused_kernel`（`gate_up[M, 2N] → silu(gate)·up[M, N]`，纯逐元素）。
> 同批次的 `_gemm_basic_kernel`（已冻结 v1）与 `splitk_reduce_kernel`（进行中）是另两个独立
> 设计任务；本文只在**输出契约**（§0.2、§4.2）与**调用关系**（§0.3）上说明接口。
> 本 kernel 不涉及 Cube / L0 / L1。
>
> 本文档由 `tilelang-ascend` → `tilelang-op-design` 流程产出：Phase M0（三问解读）→
> Phase R（算法调研，§1.6.0）→ Phase M1（耦合性判定与重设计，§0.5–§0.6）→ Phase 2
> （算法级优化，§1.6.1–§1.6.3）。首次设计，无「相对上一版的关键调整」。

---

## 0. 源算子解读与迁移分析

### 0.1 源算子语义（做什么）

**数学语义**（源 `_main` 的唯一计算语句，逐元素、无跨元素依赖）：

```
out[m, j] = silu(gate_up[m, j]) * gate_up[m, N + j]      ,  m ∈ [0, M), j ∈ [0, N)
silu(x)   = x · σ(x) = x / (1 + e^{-x})
```

即「gated activation」的 `silu_and_mul`：`gate_up` 的**前半列**是 gate，**后半列**是 up，
输出宽度减半。这与仓库参考实现 `tileops/workloads/moe.py:gated_activation`
（`gate, up = gate_up.chunk(2, dim=-1); silu(gate) * up`）一致，也与 manifest
`w_gate_up [2S, H]` 把 gate 行叠在 up 行之上（`gated_activation(hidden @ w_gate_up.T, ...)`）
一致。

**I/O 契约**：

- 输入 `gate_up`：`[M, 2N]`，连续（row-major，尾轴连续），dtype `float16 | bfloat16`。
- 输出 `out`：`[M, N]`，同 dtype，由 `@tilelang.jit(out_idx=[1])` 自动分配（调用方不预分配）。
- 无动态轴：源工厂 `_silu_mul_fused_kernel(M, N, dtype_str)` 由 `@target_cache(maxsize=16)`
  按 `(M, N, dtype)` 专门化；`M = T`（token 数）、`N = S`（expert 宽度）都是编译期常量。
- 无转置、无 stride 变体、非 in-place（输出是独立张量）。

**规约语义**：**无**。本算子不含任何规约（无累加轴、无 max/sum、keepdim 不适用），
因此不存在「累加顺序 → fp16/bf16 舍入分组」类迁移风险——这是它与同批次两个 GEMM 设计的
本质差别，也是 §1.6.1 的等价性只需讨论**单元素表达式重结合**的原因。

**dtype 语义**（决定性）：

```
g_f32 = cast(g, float32)                      # 精确加宽（16-bit → fp32 无损）
u_f32 = cast(u, float32)
sigmoid_g = sigmoid(g_f32)                    # fp32 内计算
result = g_f32 * sigmoid_g * u_f32            # fp32 乘，链式左结合
out = cast(result, dtype)                     # 只有这一次舍入
```

即：**16-bit 输入精确加宽 → 全程 fp32 → 末尾单次舍入**。这是必须保持的语义层要素
（迁移后若在 16-bit 域做算术，会引入 2–3 次额外舍入，见 §1.6.1 C4 的弃选论证）。

**边界语义**：

- 尾块：`if m_start + i < M and n_start + j < N` 逐元素守卫——越界行/列**不写**（谓词存储）。
  源实现用「守卫」而不是 padding；读侧越界行/列的装载结果落在被守卫丢弃的 lane 上。
- 非有限值：无任何特殊处理，按 IEEE 自然传播（`σ(-inf) = 0` → `(-inf)·0 = NaN`；
  `σ(+inf) = 1` → `+inf·1 = +inf`；NaN 传播）。迁移必须保持同一分类（§1.6.1 的机器验证
  逐元素核对了 inf/NaN 分类一致性）。
- 空维度：manifest `shape_rules` 保证 `H > 0, S > 0`，且 `T ≥ 1`，故 `M, N ≥ 1`；
  `M`/`N` 非 2 的幂、非 64 的幂都出现（`N ∈ {18432, 2048, 1536}`，`M ∈ {1, 32, 512, 4096}`）。
- 无 `k == 0` / 归约空轴类退化。

**host 侧语义**：`out_idx=[1]` 只做输出分配；`pass_configs={TL_ENABLE_FAST_MATH: True}` 与
`compile_flags=["-O3", "-DENABLE_BF16"]` 是 nvcc 域编译开关（§0.5 处置为「舍弃」），
不改变数学语义；没有数据预处理/后处理（无 pad、无 permute、无 cast）。

**语义保持基线**：§8.1 golden 函数以本节语义为唯一依据实现（fp32 加宽 → fp32 计算 →
单次舍入），**不复刻** §0.6 的 NPU 实现。

### 0.2 源算子输入输出

| 参数 | 方向 | Shape | dtype | 说明 |
|------|------|-------|-------|------|
| `gate_up` | in | `[M, 2N]` | float16 / bfloat16 | 前半列 `[:, 0:N]` = gate，后半列 `[:, N:2N]` = up；尾轴连续 |
| `out` | out | `[M, N]` | 同输入 | **输出 shape `[M, N]`**（迁移必需字段，已确认） |

**与 `_gemm_basic_kernel` 的输出契约**（跨文件，必须显式约定）：

```
gate_up[m, j]      = Σ_h hidden[m, h] · w_gate_up[j,   h]      # j ∈ [0, N)     ← gate 半区
gate_up[m, N + j]  = Σ_h hidden[m, h] · w_gate_up[N + j, h]    # j ∈ [0, N)     ← up 半区
```

- `w_gate_up` 是 `[2S, H]`，**gate 行在前**（行 `[0, S)`），**up 行在后**（行 `[S, 2S)`）；
  `_gemm_basic_kernel(m=T, n=2S, k=H, trans_b=True)` 的输出因此天然是「gate 列半区 + up 列半区」。
- 本 kernel 的输出 `out [M, N] = [T, S]` 就是 `_gemm_basic_kernel(m=T, n=H, k=S, trans_b=True)`
  （down GEMM）的 A 操作数。
- 若把 gate/up 半区对调，数学结果改变（源仓测试
  `test_shared_expert_mlp_keeps_the_gate_up_row_order` 正是这条负向对照）——本设计在 §8.1
  golden 与 L0 用例中都保留该顺序。

**本 kernel 的 5 条实际 shape**（manifest `moe.yaml:SharedExpertMLPFwdOp`，`M = T`、`N = S`；
其中 ds-v3-t32 有 bf16/fp16 两个 dtype 用例）：

| workload | dtype | `gate_up` | `out` | 读 + 写流量 |
|---|---|---|---|---|
| kimi-k2-t1 | bfloat16 | `[1, 36864]` | `[1, 18432]` | 110 592 B |
| kimi-k2-t512 | bfloat16 | `[512, 36864]` | `[512, 18432]` | 56 623 104 B |
| ds-v3-t32 | bfloat16 / float16 | `[32, 4096]` | `[32, 2048]` | 393 216 B |
| ds-v3-t4096 | bfloat16 | `[4096, 4096]` | `[4096, 2048]` | 50 331 648 B |
| glm45-t32 | bfloat16 | `[32, 3072]` | `[32, 1536]` | 294 912 B |

### 0.3 实现算法解读（怎么算）

**计算步骤分解**（源码每个计算语句都归入下表，无遗漏、无臆造）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式的部分 |
|---|---|---|---|---|
| 1 | `bx_m = bx // num_blocks_n; bx_n = bx % num_blocks_n` | `bx`（一维 block 索引） | `(m_start, n_start)` | 任务映射（无数学语义） |
| 2 | `g = gate_up[m_start+i, n_start+j]`（守卫内） | GM `gate_up` | 寄存器 | `gate_up[m, j]` |
| 3 | `u = gate_up[m_start+i, n_start+j+N]`（守卫内） | GM `gate_up` | 寄存器 | `gate_up[m, N+j]` |
| 4 | `g_f32 = cast(g, f32)`、`u_f32 = cast(u, f32)` | 16-bit 寄存器值 | fp32 寄存器 | dtype 语义（加宽） |
| 5 | `sigmoid_g = T.sigmoid(g_f32)` | `g_f32` | fp32 | `σ(g)` |
| 6 | `result = g_f32 * sigmoid_g * u_f32` | `g_f32, σ, u_f32` | fp32 | `g·σ(g)·u`（左结合两次乘法） |
| 7 | `out[...] = cast(result, dtype)`（守卫内） | fp32 | GM `out`（16-bit） | 单次舍入写回 |

**数据流与内存访问模式**（源硬件视角）：

```
GM[gate_up, 2N 行宽]──(每线程 coalesced load, 无 SMEM 中转)──> 寄存器 (16-bit)
      ──cvt──> fp32 寄存器 ──exp/div/mul──> fp32 结果 ──cvt──> GM[out]
```

- **无 shared memory**：源实现在 `T.Parallel(block_m, block_n)` 里直接读 GM、直接写 GM，
  没有 `alloc_shared` 中转，也没有 `T.Pipelined`。中间量只存在于寄存器和编译器插入的
  向量化 load/store 中。
- `gate_up` 的两个半区在同一行内相距 `N` 个元素（字节距离 `N·2`），因此每个 block 的两次
  读都是「行内 `block_n` 个连续元素 + 行跨 `2N·2` 字节」的 2-D 跨段访问。
- 一个元素被读 1 次（gate）、其配对元素读 1 次（up），输出写 1 次 → 理论最小流量
  `3·M·N·2 B`（不含任何中间往返）。

**循环与并行结构**：

- grid：一维 `num_blocks_m · num_blocks_n = ceil(M/block_m) · ceil(N/block_n)` 个 block。
- block 内：`threads=256` 个 CUDA 线程；`T.Parallel(block_m, block_n)`（默认 128×256 =
  32 768 元素 / 256 线程 = 128 元素/线程）由编译器做线程映射与向量化（每个线程的连续元素
  合并为 16 B 访问）。
- 逐元素守卫 `if m_start+i < M and n_start+j < N` 是谓词化的 load/store（不是分支）。
- 无流水、无双缓冲、无跨 block 通信。

**host 侧逻辑**：

- `_silu_mul(cfg)`（`SharedExpertMLPKernel` 内）用 `cfg["block_m"]`、`cfg["block_n"]`、
  `cfg["threads"]` 实例化内层工厂；默认 `(128, 256, 256)`。
- `forward` 的调用条件（源 GPU）：`fused is None and not activated`——即 gate/up GEMM
  **没有**把激活融进 epilogue 时，本 kernel 才被单独 launch。GPU 在 SM90 且
  `T ≥ template_min_m(512)` 时会把 `silu_and_mul` 融进模板 epilogue（`activated=True`），
  或对 `T=32 + split_k>1` 用带激活的 split-K epilogue（`fused`）。
- 迁移后（NPU wrapper `examples/TileOPs/tileops/kernels/moe/shared_expert_mlp/shared_expert_mlp.py`）**没有** SM90 模板分支：
  `_decompose` 只在 `gate_up_split > 1` 时让 split-K 归约融合 `silu_and_mul`（`fused`），
  否则 `gate_up_split == 1` → 本 kernel 被调用。按该 wrapper 的 `_split_k`（`T ≤ 32` 且
  `k_tiles ≥ 2` 时切 4 片），当前接线下调用的 workload 是 **kimi-k2-t512 与 ds-v3-t4096**
  （另 3 条走 splitk_reduce 的融合 epilogue）。本设计仍对全部 5 条 shape 给配置（§5.6），
  因为工厂按 `(M, N, dtype)` 专门化、L0/Boundary 与未来接线都会用到。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|---|---|---|---|---|
| O1 | **silu 与 mul 融合进一个 kernel** | 消掉一次中间物化（若拆两个 kernel：`+2·M·N·2 B`，流量 3MN→5MN） | 单 kernel 内 `g·σ(g)·u` | 无（算法层，纯数据流合并） | **可移植** |
| O2 | 读 `g`/`u` 各一次、写 `out` 一次 | 流量达到理论下界 `3MN·2 B` | 直接在寄存器里配对同一 `(m, j)` 的两个半区元素 | 无（算法层） | **可移植** |
| O3 | fp32 中间精度 | 数值正确（16-bit 链会累积 2–3 次舍入） | cast → fp32 计算 → cast 回 | 无（数值语义） | **可移植（语义层，必须保留）** |
| O4 | 尾块谓词守卫（而非 padding） | 非整除 shape 免 host padding/裁剪 | 逐元素 `if` 谓词 | 无 | **可移植** |
| O5 | `T.Parallel(block_m, block_n)` 线程级并行 + 编译器向量化 | 用满 SM 的 256 线程 × 16 B 访问 | CUDA 线程模型 + 每线程多元素合并访存 | warp/coalescing/线程块 | **硬件强相关** |
| O6 | 二维 `(block_m, block_n)` 输出 tile 网格 | 提高 L2 复用/占用率（tile 越大，跨块行首开销越少） | block 索引二维分解 | SM 占用率 + L2 | **硬件强相关**（tile 取值需按 56 AIV + UB 重定） |
| O7 | `TL_ENABLE_FAST_MATH=True`、`-O3`、`-DENABLE_BF16` | 更快的 `exp`/除法、bf16 代码生成 | nvcc 编译开关 | nvcc / `--use_fast_math` | **不可移植**（BishengIR 无对应开关；`ENABLE_BF16` 在本后端是原生 dtype） |
| O8 | 二维 tile 尺寸 128×256（`default_config`） | 适配 SM90 的 SM 数与共享内存 | 静态 tile 常数 | SM 数量 | **硬件强相关**（按 56 Vector Core + UB 248 KB 重定） |

> 识别不出来 ≠ 不存在。以上 8 项均已在 §0.5 逐项判定处置，无静默丢弃。

### 0.5 硬件耦合性分析与 NPU 适配决策

**判定问题**：实现算法和优化手段是硬件强相关吗？能用在 NPU 上吗？

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|---|---|---|---|---|---|
| 数学语义（§0.1 公式、fp32 中间、单次舍入、非有限值传播） | 语义 | 无 | — | **保留** | 语义层无条件保留（§1.3 / §3.1 与 §0.1 逐项对齐） |
| I/O 契约 `[M,2N] → [M,N]`、gate 前半列 / up 后半列、`out_idx` 分配 | 语义 | 无 | — | **保留** | §4.1/§4.2；工厂签名保持 `(M, N, dtype)` 外层 + `(block_m, block_n)` 内层 |
| O1 激活融合（1 kernel 而非 2） | 算法 | 无 | — | **保留** | 本 kernel 本身就是融合结果；§1.6.0 C7 给出「拆两 kernel」的量化反例（+67% 流量） |
| O2 最小流量（g/u 各读一次、out 写一次） | 算法 | 无 | — | **保留** | §1.6.0 R3 的流量下界 `3MN·2 B`；§4.4 搬运路径无中间 GM 往返 |
| O3 fp32 中间精度 | 语义 | 无 | 有（`vcvt` fp16/bf16 ↔ fp32 精确加宽；`vdiv` 0 ulp） | **保留** | §3.2 的 `vcvt(UNPK_B16)` + fp32 `vexp/vadds/vmul/vdiv` 链；§1.6.1 机器验证 0 ulp |
| O4 尾块谓词守卫 | 算法 | 无（谓词等价于区段截断） | 有 | **等价替换** | 用 `T.min(bm, M-m0)` / `T.min(bn, N-n0)` 的**区段 extent** 取代逐元素谓词：装载/存储只覆盖有效区段，死行/死 lane 不进 GM（与源守卫语义等价，且省掉每元素比较）。依据：`kernel_template.py` 的 `tail_size` 写法 + 探针 P3（`am=1`、`an=476` 编译通过） |
| O5 `T.Parallel` 线程级并行 + 每线程向量化 | 优化 | warp / coalescing / 256 线程块 | 部分（**无 thread 域**：`T.Kernel` 无 `threads` 参数） | **重新设计** | → §0.6 R1：一维 `T.Kernel(grid)` + `T.SimdVF` 的 **64 lane 显式向量链**（`vld/vcvt/vmul/vdiv/vexp/vsts`），lane 即源实现「每线程多元素」的等价物。依据：`programming.md`「`T.SimdVF()` SIMD 执行区域；`T.simd.*` 提供寄存器级操作」、`examples/engram_gate_conv/_engram_gate_conv_fwd_kernel/_engram_gate_conv_fwd_kernel.py`（已验证 silu 链） |
| O6 二维 block 网格（tile 覆盖） | 优化 | SM 占用率 / L2 | 有（一维 grid + 核内 tile 循环） | **等价替换** | 一维 grid = `ceil(M/bm) × nsp`，每核用 `T.Pipelined(tpb)` 走 `tpb` 个列 tile（§5.5/§6.2）。依据：`_delta_kernel.py` 的 `nsp/tpb` 几何 + 本设计探针 P1–P4 |
| O7 `FAST_MATH`/`-O3`/`-DENABLE_BF16` | 优化 | nvcc | 无对应开关 | **舍弃** | BishengIR/CANN 无 `--use_fast_math`；bf16 是原生 dtype（`vcvt` 直接支持）。数值后果：使用 CANN `vexp` 默认变体（`simd.py` 注明 default exact FP32 division / `ftz_false` 变体族），**比 fast-math 更精确**，方向与 golden 一致（不视为偏差；机器验证覆盖） |
| O8 tile 128×256 | 优化 | SM90 SM 数 / SMEM | 有，但常数不同 | **重新设计** | → §0.6 R4：按 UB 预算与 56 Vector Core 重定 `bm/bn`（§5.2、§5.6）；同时**夹取** wrapper 传入的 128×256（§5.2 规则 1/5，避免 UB 溢出） |

> 依据来源：`programming.md`（API/容量/同步规则）、`knowledge/PL-950-*`（已验证模式）、
> `examples/engram_gate_conv`、`examples/mla_dense_bwd/_delta_kernel`（同后端已验证 kernel）、
> 本设计的无卡编译探针（`history_version/design_probe_compile.py`，4 例 COMPILE OK）。
> 舍弃项（O7）已写明理由；无「无法判定」条目。

### 0.6 NPU 算法重设计

**重设计项 R1：并行结构与向量化（O5）**

- **源方案**：`threads=256` 的 CUDA 线程块 + `T.Parallel(block_m, block_n)`，每个线程承担
  128 个元素并由 nvcc 合并为向量化访存；逐元素标量表达式 `g*σ(g)*u`。
- **NPU 新算法**：一维 `T.Kernel(grid)`（无 thread 域），每核承担 `tpb` 个 `(bm, bn)` tile；
  tile 内用 `T.SimdVF()` 的 64-lane 向量寄存器链：每个 64 元素 chunk 走
  `vld(UNPK_B16) → vcvt(fp32) → vexp/vneg/vadds/vmul/vdiv → vcvt(narrow) → vsts(PK_B32)`。
  行循环 `T.serial(bm)`、chunk 循环 `T.serial(bn/64)` 都是**调度循环**，逐元素计算不存在
  标量循环（§1.6.2）。承接源意图：每 lane 一个元素的吞吐，且元素级并行度不变
  （`bm·bn` 个元素 / 64 lane 条向量指令）。
- **语义保持论证**：数学映射逐元素不变；fp32 中间精度与末尾单次舍入不变（`vcvt
  (UNPK_B16)` 是精确加宽，`vcvt(...,dtype,ROUND_R)` 是 round-to-nearest-even 单次舍入）；
  边界语义由区段 extent 实现（等价替换 O4）；非有限值传播经机器验证分类一致（§1.6.1）。
  累加顺序无关（无归约）。

**重设计项 R2：数据搬运层级（源为隐式寄存器直读 + L2）**

- **源方案**：线程直接从 GM 读 `gate_up` 两个半区、直接写 GM `out`；靠 L2/合并访问隐藏
  延迟，无显式片上中转、无流水。
- **NPU 新算法**：`GM --MTE2 T.copy--> UB(g_ub, u_ub) --vld--> 向量寄存器 --vsts--> UB(y_ub)
  --MTE2 T.copy--> GM`。片上驻留 `3·bm·bn·2 B`（§4.5），列 tile 用 2 个 buffer version 与
  `T.Pipelined(num_stages=2)` 把 MTE2 与向量计算重叠（§6.3）。承接源意图：一次读入后元素
  只被读一次，输出只写一次（流量下界不变）；把源实现隐式依赖的「L2 吸收重复访问」换成
  显式 UB 驻留 + 双缓冲。
- **语义保持论证**：搬运不改变任何元素值（同 dtype 的 `T.copy` 位级恒等）；数据流仍为
  「读两次、算一次、写一次」，无中间 GM 往返；死行/死 lane 不进入显式区段拷贝。
  依据：`PL-950-vector-execution`（UB 内的尾块填充 + 有效区段搬入/写回）、探针 P1–P4。

**重设计项 R3：tile 几何（O6/O8）**

- **源方案**：`block_m=128, block_n=256`（SM90 的 SM 占用率 × 共享内存取向），二维网格
  `ceil(M/128) × ceil(N/256)`。
- **NPU 新算法**：两档几何（§5.2/§5.6）：大 shape `(bm, bn) = (16, 512)`、小 shape
  `(1, 256)`；一维 grid = `mt × nsp`，每核 `tpb = nt/nsp` 个列 tile，`nsp` 由「grid 覆盖
  56 个 Vector Core 且 `tpb ≥ 2`」的构建期规则取（§5.5）。承接源意图：tile 足够大以摊薄
  搬运并让 grid 覆盖全部核；常数按 UB 248 KB 与 56 Vector Core 重算。
- **语义保持论证**：tile 只改变任务映射，元素值与其配对关系（`j` 与 `N+j`）由拷贝区段
  显式保证；`bm/bn` 不整除 `M/N` 时走动态 extent（§5.4）。

**无重设计项的条目**：O1–O4、O6 已按「保留 / 等价替换」处置，无其他重设计。
**§0.4 优化意图承接核对**：O1/O2 → 保留（本 kernel 即融合形态，R2 不增加流量）；
O3 → 保留（§1.6.1 机器验证 0 ulp）；O4 → O4 等价替换（区段 extent）；O5 → R1；
O6 → R3（+O6 等价替换）；O7 → 舍弃（理由已给）；O8 → R3。**无静默丢弃**。

### 0.7 标杆实现

- 源算子路径（golden 依据）：`/workspace/TileOPs/src/tileops/_kernels/moe/shared_expert_mlp.py`
  的 `_silu_mul_fused_kernel`（`_main` 的 7 个计算语句即 §0.3 的分解）。
- 源仓参考实现（语义等价基准）：`examples/TileOPs/tileops/workloads/moe.py`（`gated_activation` / `ref_shared_expert`）
  `gated_activation`（`chunk(2, -1)` + `F.silu(gate) * up`）与 `:173-182` `ref_shared_expert`；
  manifest `moe.yaml:SharedExpertMLPFwdOp` 的 `roofline` 把本步记作 `6·T·S` FLOPs。
- 源仓测试：`examples/TileOPs/tests/ops/test_fused_moe_shared_expert.py`
  （`moe_verification(1)`：逐元素 `atol = rtol = 1e-2` + 归一化平方误差 ≤ 1e-4；
  gate/up 顺序负向对照 `test_shared_expert_mlp_keeps_the_gate_up_row_order`）。
- golden 实现见 §8.1（以 §0.1 语义为依据，优先移植上述参考实现）。

---

## 1. 概述

### 1.1 算子名称

`_silu_mul_fused_kernel`（gated activation `silu_and_mul`，`[M, 2N] → [M, N]`）

### 1.2 功能描述

把 `SharedExpertMLPFwdOp` 的 gate/up GEMM 输出 `gate_up[M, 2N]` 压成
`silu(gate) · up [M, N]`：前半列取 gate、后半列取 up，逐元素做 SiLU 后相乘。
纯逐元素、零规约、纯带宽受限；在 950 上落在 Vector（AIV）单元的 SIMD 路径。

### 1.3 数学公式

$$
\text{out}[m, j] \;=\; \text{silu}\big(\text{gate\_up}[m, j]\big)\cdot \text{gate\_up}[m, N+j],
\qquad \text{silu}(x) = \frac{x}{1+e^{-x}},
\qquad m\in[0,M),\ j\in[0,N)
$$

dtype：`gate_up`、`out` 均为 `float16` 或 `bfloat16`；**中间计算全程 fp32，输出一次舍入**（§0.1）。

### 1.4 算法描述

**迁移决策后的 NPU 侧算法**（§0.6 R1–R3 落地，与源算法的差异逐条注明）：

1. **任务映射**（源 O6/O8 → §0.6 R3）：一维 `grid = mt × nsp`，`mt = ceil(M/bm)`；
   第 `bx` 个核负责行块 `im = bx / nsp` 与列 tile 段 `sp = bx % nsp`，核内沿列走
   `tpb = nt/nsp` 个 tile（`nt = ceil(N/bn)`）。
2. **装载**（源为线程直读 GM → §0.6 R2 重新设计）：每个 tile 两次 `T.copy` 把
   `gate_up[m0:m0+am, n0:n0+an]`（gate 半区）与 `gate_up[m0:m0+am, N+n0:N+n0+an]`
   （up 半区）搬进 UB 的 `g_ub`/`u_ub`；`am = min(bm, M-m0)`、`an = min(bn, N-n0)`
   为动态区段（源 O4 → 等价替换）。
3. **逐元素计算**（源 O5 线程级并行 → §0.6 R1 重新设计）：`T.SimdVF()` 内按
   行 `T.serial(bm)` × chunk `T.serial(bn/64)` 走 64-lane 向量链：
   `vld(UNPK_B16) → vcvt(fp32) → 数学链 → vcvt(narrow) → vsts(PK_B32)`。
   数学链形如 §1.6.1 的优化后公式。
4. **写回**：`T.copy(y_ub[0:am, 0:an], out[m0:m0+am, n0:n0+an])`，一次写 GM。
5. **流水**：列 tile 循环 `T.Pipelined(tpb, num_stages=2)` + 三个 buffer 各 2 个 version
   （源无流水 → §0.6 R2 的新增结构，承接「用片上驻留隐藏访存延迟」的意图）。

**与源算法的结构差异**：① 无 thread 域，用 64-lane 显式 SIMD 链替代线程映射 + 编译器
向量化；② 新增 UB 中转与双缓冲（源直接寄存器直读 GM）；③ 逐元素谓词守卫换成区段
extent；④ tile 常数 `(16, 512) / (1, 256)` 取代 `(128, 256)`；⑤ 无 `FAST_MATH` 开关。
数学映射、fp32 精度与单次舍入**未变**。

### 1.5 数据流图

```
GM gate_up[M, 2N]  (fp16/bf16, 行尾连续)
   ├─半区 gate: [m0:m0+am, n0:n0+an]  --T.copy(MTE2)--> UB g_ub[bm, bn]
   └─半区 up  : [m0:m0+am, N+n0:N+n0+an] --T.copy(MTE2)--> UB u_ub[bm, bn]
                              │
                    T.SimdVF 64-lane 向量链
        vld(UNPK_B16) → vcvt(fp32)      （gate、up 各一次）
        num = vmul(g, u) ; den = vadds(vexp(vneg(g)), 1)
        y   = vdiv(num, den)            （§1.6.1 A1 优化后公式）
        vcvt(y, dtype) → vsts(PK_B32)
                              │
                        UB y_ub[bm, bn]
                              │
                    T.copy(MTE2) --> GM out[M, N]  (fp16/bf16)
```

### 1.6 算法调研与优化分析

> 调研对象 = 同一数学语义（`silu(gate)·up`，逐元素、零规约）的算法族；源算法只是基线候选之一。
> 调研深度：**轻量调研**（单步逐元素，无规约/窗口/矩阵）——四问逐问回答并给依据，R3 仍按
> 四口径量化（本算子带宽受限，访存/遍数是主导项）。

#### 1.6.0 算法调研（Algorithm Research）

**R1 等价化简公式候选**（基线在表内；正式等价论证与收益量化在 §1.6.1）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 是否纳入 R3 对比 |
|---|---|---|---|---|---|
| C0 | **基线：源算法** | `p = g·σ(g); y = p·u`，σ 用一次倒数 `1/(1+e^{-g})` | — | — | ✅ 基线 |
| C1 | **分子前置** | `y = (g·u)/(1+e^{-g})` | 数学恒等（有理式结合/交换）；fp32 舍入点由 3 个减到 2 个 | 少 1 条向量算符/64 元素（6→5） | ✅ **采纳**（§1.6.1 A1） |
| C2 | 指数取正 | `y = (g·u)·e^{g}/(1+e^{g})` | 实数域恒等，但 **fp32 溢出域不等价**：`g>88.7` 时 `e^g=inf` → `inf/inf=NaN`（真值有限） | op 数与 C1 相同（5） | ✅（作为反例量化） |
| C3 | exp2 换底 | `σ(g) = 1/(1+e2^{-g·log2e})` | 容差内等价（CANN `vexp` 本身即 SFU 实现） | **负收益**：多 1 条 `vmuls`（6），无精度收益 | ❌ 见 R4 |
| C4 | 16-bit 算术链 | 全程 fp16/bf16 运算（lane 数 ×2 → 指令数减半） | **不容差内等价**：每步 8–11 bit 舍入，破坏 §0.1「fp32 中间 + 单次舍入」 | 指令数 ≈ −55% | ✅（作为 R3 对照） |
| C5 | 与 gate/up GEMM epilogue 融合 | 在 GEMM 累加器上直接算 `silu_and_mul` 再落盘（源的 `activated=True` 路径） | 数学恒等 | 消掉本 kernel 的全部流量（子链 5MN→MN） | ❌ 本函数契约外（见 R4） |
| C6 | 原地写回 gate 半区 | 结果写入 `gate_up[:, 0:N]` | 数学恒等 | **0 收益**（写字节数不变） | ❌ 见 R4 |
| C7 | 拆成两个 kernel（silu 落盘 → 再逐元素乘） | `t = silu(g)` 落 GM；`y = t·u` | 恒等 | **负收益**：+2·M·N·2 B（3MN→5MN） | ✅（作为反例） |

**R2 在线算法**：**有，且已是单遍**——逐元素映射无跨元素依赖，每元素只需其自身的
`(g, u)`，天然 online（无 running 统计量、无 rescale、无需第二遍）。结构判据：算子不含
max/sum/计数类可分解统计量，也不含输出依赖全局索引的结构，因此「两遍 → 单遍」的收益空间
不存在。**源算法本身已是融合单遍**（§0.4 O1/O2：读 2 份、写 1 份 = 流量下界）。唯一存在的
「两遍」是**算子级**分解（gate/up GEMM 把 `gate_up` 落 GM，本 kernel 再读回），它是跨 kernel
结构、属 C5 候选，不是本函数可在内部在线化的对象。收益口径核对：扫描遍数 = 1（下界），
中间缓冲 = `O(bm·bn)`（与 M、N 无关），UB 驻留 49 KB（§4.5）✅。

**R3 复杂度对比**（四口径；口径说明：FLOPs 只折算**向量算符条数**（每 64 元素一条 32-bit
lane 指令；超越函数 `vexp`/`vdiv` 按 1 条算，不含 load/store 与 `vcvt`）；访存 Bytes 按
不可消除 I/O = 读 `2·M·N·elem` + 写 `M·N·elem`；扫描遍数按每元素 GM 往返次数；中间缓冲峰值
指片上驻留或需额外 GM 物化的量。代表 workload：ds-v3-t4096（M=4096, N=2048, bf16）与
kimi-k2-t512（M=512, N=18432, bf16）；口径与 manifest `roofline` 的 `6·T·S` 一致但更细）：

| 算法候选 | 向量算符 / 64 元素（含 cvt） | 访存量 (Bytes) | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---|---|---|---|---|---|
| **C0 基线** | 2 vld + 2 vcvt + **6** + 1 vcvt + 1 vsts = **12** | 50.33 MB（ds-v3-t4096）/ 56.62 MB（t512） | 1 | 3·bm·bn·2 = 49 152 B（2 version，`bm=16,bn=256`） | mt·nsp ≤ 256 逻辑块（≤ 4.6 波）；无跨核同步 |
| **C1（采纳）** | 2 vld + 2 vcvt + **5** + 1 vcvt + 1 vsts = **11**（−8.3%） | 同 C0（**不变**） | 1 | 同 C0 | 同 C0 |
| C2 | 11 | 同 C0 | 1 | 同 C0 | **非有限值不等价**（`g>88.7` → NaN；§1.6.1 机器实测：exp-overflow 用例 bf16 **647 ulp**、角点 **5 个非有限值分类不符**） |
| C4 16-bit 链 | 2 vld + **3**（128-lane）+ 1 vsts ≈ **4.5/64 元素** | 同 C0 | 1 | 同 C0 | 指令数最低，但**破坏 dtype 语义**（3 次额外舍入）→ R4 淘汰 |
| C5 GEMM epilogue 融合 | 0（本函数不存在） | 本 kernel **0**；子链 5MN·2 → MN·2（ds-v3-t4096: 83.9 MB → 16.8 MB，**−80%**） | 1（GEMM 内） | 需 GEMM 侧 L0C/UB 驻留（128 KB L0C 量级） | 与 gate/up GEMM 同核（Cube+Vector 混合）；**本函数契约外** |
| C6 原地写回 | 11 | 50.33 MB（写量不变，仍 16.78 MB） | 1 | 同 C0 | 破坏 I/O 契约（输入被覆盖、调用方张量别名） |
| C7 两 kernel | 12 + 12 | **83.9 MB**（+67%：多写+多读 MN） | 3（silu 读写 + mul 读写） | +M·N·2 = 16.8 MB **GM** | 2 次 launch + 全量 GM 往返 |

**R4 硬件亲和性评估**（逐候选对照检查清单；负向淘汰附依据）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---|---|---|---|---|---|---|
| C0 基线 | ✅ 全部落 AIV SIMD（`vexp/vdiv/vmul` 有寄存器级 intrinsic，`simd.py` 导出已核）；无标量热点 | ✅ 49 152 B / 253 952 B = 19.4%（×1.12 膨胀 = 21.8%） | ✅ `N ∈ {18432,2048,1536}` 全被 64（chunk）与 512/256（bn）整除 | ✅ `M/N/dtype/bm/bn` 编译期常量（工厂按 shape 专门化） | ✅ `T.Pipelined` + `annotate_buffer_versions` 已核（`_delta_kernel`、探针 P1/P2/P4） | ✅ 可用（被 C1 支配） |
| **C1（采纳）** | ✅ 同上，且少 1 条算符 | ✅ 同上 | ✅ 同上 | ✅ 同上 | ✅ 同上 | ✅ **采纳** |
| C2 | ✅ 同一单元 | ✅ | ✅ | ✅ | ✅ | ❌ **淘汰**：`e^{g}` 在 `g>88.7` 溢出 → `inf/inf = NaN`，与真值（有限）**非容差内等价**；§1.6.1 机器验证实测 647 ulp / 5 处非有限值不符，**违反 §0.1 边界语义**（依据：本仓 `verify_equiv.py` 输出，非估计） |
| C3 | ✅ 单元一样 | ✅ | ✅ | ✅ | ✅ | ❌ **淘汰**：`exp2` 换底需额外 `vmuls`（6 > 5），且 CANN `vexp` 已按 SFU 实现（`simd.py` 默认精确变体），无速度/精度收益；与任务给定的「避免 exp2 近似超差」一致 |
| C4 16-bit 链 | ✅（128 lane 满宽度） | ✅ | ⚠️ `vinstr` 16-bit 需 `UNPK/PK` 变体 | ✅ | ✅ | ❌ **淘汰**：**改变 §0.1 的 dtype 语义**（fp32 中间 → 16-bit 中间，多 3 次舍入）；任务与源 docstring 均要求 fp32 中间。注意本算子**带宽受限**（§1.6.0 估算：流量项 ≫ 发射项），指令数减半不降低 GM 流量 → 收益上限 ≈ 发射项（14.3→~7 µs）而风险是精度（§9.1 风险 3） |
| C5 GEMM epilogue 融合 | ⚠️ 需 Cube+Vector 混合（`T.Cube` + dual_copy，`PL-950-mixed-epilogue` 只验证过 M 分片 ReLU） | ⚠️ 需在 GEMM 内核里多驻留一份 fp32 累加器切片 | ✅ | ✅ | ⚠️ 跨阶段融合，平台知识库只有单 tile 混合条目 | ❌ **本函数不采纳（契约外）**：本 kernel 的输入是 GM 上**已物化**的 `gate_up`；融合需要改 `_gemm_basic_kernel` 的 epilogue（另一设计任务），且 split-K 路径已由 `splitk_reduce_kernel` 的融合 epilogue 覆盖。**记录为算子级后续优化项**（§9.3），不静默丢弃 |
| C6 原地写回 | ✅ | ✅ | ✅ | ✅ | ⚠️ 需保证 `out` 与 `gate_up` 同一 base（别名） | ❌ **淘汰**：写字节数不变（0 收益），却把 I/O 契约改成别名存储（调用方张量被破坏，down GEMM 的输入与本 kernel 输出重叠） |
| C7 两 kernel | ✅ | ⚠️ +16.8 MB GM | ✅ | ✅ | ❌ 多 1 次 launch、多 2 遍全量 GM | ❌ **淘汰**：流量 +67%，且与源 O1 的融合意图相反 |

**调研结论**：选定算法族 = **C1（单遍、g/u 各读一次、`y=(g·u)/(1+e^{-g})` 的 fp32 向量链、
区段 extent 尾块、UB 双缓冲）**，即「在基线 C0 的算法结构上做一次数学等价重结合」。
关键依据：R3 中 C1 与 C0 流量完全相同（已达 `3MN·elem` 下界）而向量算符少 1 条（12→11），
且 §1.6.1 的机器验证显示两者在输出 dtype 上**逐位相同（0 ulp）**；C2/C4/C6/C7 分别在
边界语义、dtype 语义、契约与流量上被淘汰；C5 是**算子级**最优（子链流量 −80%），但改的是
`_gemm_basic_kernel` 的 epilogue，不在本函数契约内（§9.3 记录为后续优化项）。
**与基线的结构差异一句话**：算法仍是单遍逐元素，改变的是「分子先乘、分母一次除」的
结合顺序与 NPU 侧的执行结构（§0.6 R1–R3）。

**调研范围记录**（"无更优替代"类断言的举证口径）：平台 `knowledge/algorithm-candidates.md`
（elementwise / broadcast 行：输入转 FP32、行标量广播后逐元素计算；softmax 行不适用）、
`knowledge/PL-950-vector-execution`、`PL-950-row-primitives`、`PL-950-simd-reduce`、
`pattern-library/constants.md`（`CONST-vector-launch-overhead`、`CONST-mte2-degradation`）、
本仓 `examples/` 的 4 份 ascend 逐元素/激活实现（`engram_gate_conv`、
`examples/mla_dense_bwd/_delta_kernel`、`examples/TileOPs/tileops/kernels/elementwise/mish/mish_kernel/mish.py`、`.agents/platforms/toolchains/ascend/kernel_template.py`）、
源仓 `_silu_mul_fused_kernel` 与 `splitk_reduce_kernel` 的 `silu_and_mul` epilogue（同一数学
语义的第二份实现，见 §1.6.1）、以及 C1–C7 七类结构候选的恒等变形自查。
互联网检索未使用（轻量调研，本地源已覆盖该算子族全部候选形态）。

**设计期估算（D-2 roofline 下界行）**

代表 workload：大 shape `ds-v3-t4096`（M=4096, N=2048, bf16, bm=16, bn=256, grid 256）与
`kimi-k2-t512`（M=512, N=18432, bf16, bm=16, bn=256, grid 192）；小 shape 取
`kimi-k2-t1`（M=1, N=18432, bm=1, bn=256, grid 72）。

| 项 | ds-v3-t4096 | kimi-k2-t512 | kimi-k2-t1 |
|---|---|---|---|
| 流量项（读 2MN·2 + 写 MN·2） | 50.33 MB | 56.62 MB | 0.111 MB |
| 向量指令数（11 条/64 元素） | 1.44 M | 1.62 M | 3.2 k |
| 发射项（÷56 核 ÷ 1.8 GHz × 1 IPC，**未实测假设**） | 14.3 µs | 16.1 µs | ~0.03 µs |
| 容量项（UB 驻留 / 容量） | 49 152 B / 253 952 B = 19.4%（×1.12 = 21.8%）→ 不构成下界 | 同左 | 3 072 B → 不构成下界 |
| 流量项 @ HBM 地板 1.03 TB/s（`CONST-mte2-degradation`：18.4 GB/s/核 × 56） | 48.9 µs | 55.0 µs | 0.11 µs |
| 流量项 @ L2 驻留 2.41 TB/s（同条目 43.1 GB/s/核 × 56） | 20.9 µs | 23.5 µs | 0.05 µs |

`估算下界: 20.9–23.5 µs（L2 驻留口径）/ 48.9–55.0 µs（HBM 口径）（流量 50.3–56.6 MB ÷ 2.41 或
1.03 TB/s；发射 14.3–16.1 µs；容量项不约束 49 152 B/248 KiB = 19.4%，引用
CONST-mte2-degradation〔910B2C 实测，外推至本型号 56 Vector Core，未在 ascend-950pr-9579
实测〕与 CONST-vector-launch-overhead〔910B2C〕）`

**常数适用性声明**：平台 `cost_measurements` 为空，`pattern-library/constants.md` 的全部
条目绑定 Ascend910B2C（版本戳 `CONST-mte2-degradation` / `CONST-vector-launch-overhead`），
**对本型号属外推**，已按「未实测假设 + 估算依据」标注；「每 AIV 1 IPC @1.8 GHz」无本地文档
支撑，标注为**未文档化假设**。因此下界只用于判定「流量 vs 发射谁是地板」这一结构性结论，
不作为性能承诺。

**由估算得到的、与常数无关的结论**（决定设计重点）：

1. 大 shape 的流量项（20.9–48.9 µs）≫ 发射项（14.3 µs 级），即**带宽受限**：设计重点是
   「少搬」——1 遍扫描、无中间 GM 往返、按 64/512 对齐避免尾 lane 浪费；不是算术强度。
2. 小 shape（110–393 KB）流量仅 0.1–0.4 µs，**发射/启动主导**：这些 workload 的时延由
   launch + 每核指令链长度决定，与 M/N 的缩放无关（§5.6 对小 shape 单独取 `bm=1, bn=256`
   以提高逻辑核数）。
3. 容量项从不构成下界（≤ 21.8%），因此 `bm/bn` 的上界由**双缓冲观感与尾块**决定，不是 UB 容量。

#### 1.6.1 数学等价优化（公式级）

分析对象为 §1.6.0 选定的 C1（与基线 C0 同结构、仅改结合顺序）。逐项评估：

| # | 优化项 | 原式（基线 C0） | 优化后公式（C1） | 等价性论证 | 收益估算 |
|---|---|---|---|---|---|
| **A1** | 分子前置 + 单次除法 | `t=e^{-g}; d=1+t; s=1/d; p=g·s; y=p·u`（1 指数 + 1 加 + 1 倒数 + 2 乘 = **6** 条向量算符） | `t=e^{-g}; d=1+t; num=g·u; y=num/d`（1 指数 + 1 加 + 1 乘 + 1 除 = **5** 条） | **有理式恒等**：`g·(1/(1+e^{-g}))·u = (g·u)/(1+e^{-g})`（实数域精确；仅乘法结合律 + 倒数定义）。fp32 舍入：基线有 3 个舍入点（`s`、`p`、`y`），优化后 2 个（`num`、`y`），**误差不增**。机器验证（下表）：输出 dtype 上 **0 ulp、max_rel = 0**，非有限值分类一致（含 `±inf/NaN/subnormal/±0` 角点） | 向量算符 12→11 条/64 元素（**−8.3%**），算术链 −16.7%；流量不变（带宽受限算子的真实收益见 §1.6.0 结论 1：只降发射项） |

**采纳项 A1 的机器验证（D-1）**：脚本 `examples/shared_expert_mlp/_silu_mul_fused_kernel/verify_equiv.py`
（torch CPU；两式均按 kernel 的实际舍入路径回放：每步 fp32 舍入 + 末尾按目标 dtype 舍入；
另以 fp64 为参照）。判定标准：目标 dtype 最大 ulp 差 ≤ 1、`>1ulp` 比例 ≤ 1e-3、非有限值分类
不一致数 = 0、`atol=rtol=1e-2` 违反率 = 0。运行方式
`/workspace/venv-ascend950/bin/python verify_equiv.py`，实测输出（2026-11-23）：

| dtype | 用例 | max_ulp | frac_gt1_ulp | max_rel | 非有限值不符 | tol 违反率 | C2（弃选）max_ulp | C2 非有限值不符 | 结论 |
|---|---|---|---|---|---|---|---|---|---|
| bfloat16 | random（16 384 元素） | 0 | 0.00e+00 | 0.000e+00 | 0 | 0.00e+00 | 0 | 0 | **EQUIV_PASS** |
| bfloat16 | wide-range（g×12, u×1e3） | 0 | 0.00e+00 | 0.000e+00 | 0 | 0.00e+00 | 1 | 0 | **EQUIV_PASS** |
| bfloat16 | corner（±inf/NaN/subnormal/±0/max/tiny） | 0 | 0.00e+00 | 0.000e+00 | 0 | 0.00e+00 | 0 | **5** | **EQUIV_PASS** |
| bfloat16 | exp-overflow（g∈[-200,-80]） | 0 | 0.00e+00 | 0.000e+00 | 0 | 0.00e+00 | **647** | 0 | **EQUIV_PASS** |
| float16 | random | 0 | 0.00e+00 | 0.000e+00 | 0 | 0.00e+00 | 0 | 0 | **EQUIV_PASS** |
| float16 | wide-range | 0 | 0.00e+00 | 0.000e+00 | 0 | 0.00e+00 | 0 | 0 | **EQUIV_PASS** |
| float16 | corner | 0 | 0.00e+00 | 0.000e+00 | 0 | 0.00e+00 | 0 | **5** | **EQUIV_PASS** |
| float16 | exp-overflow | 0 | 0.00e+00 | 0.000e+00 | 0 | 0.00e+00 | 0 | 0 | **EQUIV_PASS** |

（同一脚本另外核对 **golden 形式**：`silu(g.float())*u.float()` 上取整路径 vs torch 原生
dtype 的 `F.silu(g)*u` —— bf16 最大差 1 ulp / max_rel 7.8e-3、fp16 1 ulp / 9.8e-4，
`atol=rtol=1e-2` 违反率均为 0，故 §8.1 选前者（与源 kernel 的 dtype 语义一致）不会与
源仓 `moe_close` 策略冲突。C2 列同时给出弃选候选的**实测**反例：

- 弃选项 **C2** 的实测证据：`g > 88.7` 时 `e^{g}=inf` → `inf/inf=NaN` 而真值有限（宽动态范围
  用例 bf16 最大 647 ulp），角点用例 5 个元素非有限值分类不符 → 等价性不成立，**禁止采纳**。
- 弃选项 **C4**（16-bit 链）不可机器验证为等价：其输出与 §0.1 的 fp32 中间语义本就不同
  （3 次额外舍入），属语义层变更，按 §0.5「语义层无条件保留」淘汰，不进本表。

**交叉一致性**（与同 op 的另一份 silu 实现）：`splitk_reduce_kernel` 的 `silu_and_mul`
epilogue 用的是基线形式 `gate * T.sigmoid(gate) * up`。由于 A1 在输出 dtype 上与基线
**逐位相同**（上表 0 ulp），本 kernel 采用 A1 **不会**在两条路径之间制造数值分歧。

**优化结论**：采纳 **A1** 一项（`EQUIV_PASS`，脚本与结果表随本文留存）。优化后公式
（供 §3.1 拆解使用）：

```
d   = 1 + exp(-g)          # fp32
y   = (g · u) / d          # fp32，单次除法
out = cast(y, dtype)       # 唯一一次舍入
```

**无其余优化空间的结论与依据**：① 无公共子表达式（`g`、`u` 各用一次，`d` 只算一次）；
② 无归约可合并（零规约）；③ 无高代价算符可降代换（`vexp`/`vdiv` 已是 SFU；`exp2` 换底见
R4 C3 的负收益）；④ 访存量已在下界（§1.6.0 R3：`3MN·2 B`，无中间往返）；⑤ 稳定化变形
（减 max）不适用（σ 无溢出风险，`e^{-g}` 在 `g→-∞` 时下溢到 0 是正确极限，见上表
exp-overflow 用例 0 ulp）；⑥ 唯一更大的收益（C5 融合）不在本函数契约内。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

盘点实现方案中的**全部**循环与标量计算点：

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由（不可替代时必填） |
|---|---|---|---|---|---|
| 1 | `d = 1 + exp(-g)`、`num = g·u`、`y = num/d` | 源逐元素标量表达式（每 lane 一次） | `T.simd.vexp / vneg / vadds / vmul / vdiv`（64 lane/条） | ✅ | — |
| 2 | 16-bit → fp32 加宽（2 处） | 源 `T.cast(g, "float32")` 逐元素 | `vld(dist="UNPK_B16")` + `vcvt(...,"float32",part=0)` | ✅ | — |
| 3 | fp32 → 16-bit 收敛（1 处） | 源 `T.cast(result, dtype)` 逐元素 | `vcvt(y, dtype, round="ROUND_R", part=0)` + `vsts(dist="PK_B32")` | ✅ | — |
| 4 | 元素级谓词守卫 `if m_start+i<M and n_start+j<N` | 每元素比较 + 谓词存储 | 区段 extent：`T.min(bm, M-m0)` / `T.min(bn, N-n0)` + 区段 `T.copy` | ✅ | — （由每元素谓词降为每 tile 一次标量 `T.min`，见 #6） |
| 5 | 列 tile 循环 `T.Pipelined(tpb)` | 源无（GPU 一 block 一 tile） | 无（保持为调度循环） | ❌ | **tile 级调度**：循环体是「两次 MTE2 + 一整段 SIMD 链 + 一次 MTE2」，不是逐元素计算；kernel 内不存在能把它整体化为一条向量指令的 API（多段 DMA + 多指令链）。它承担 §0.6 R2 的双缓冲重叠职责 |
| 6 | 行循环 `T.serial(bm)` + chunk 循环 `T.serial(bn/64)` | 新结构（每迭代 1 条 64-lane 指令） | 无（循环体**已经**是向量指令） | ❌ | **向量化轴本身**：每迭代 = 一条 64 lane 向量算符，等于「已完全向量化」；展开/展平只会引入 `div/mod` 标量索引（每迭代 `r = q // nchunk`），比 2-D 循环更贵。属 tile 级调度循环（§6.1） |
| 7 | `im = bx / nsp; sp = bx % nsp; m0 = im·bm; n0 = sp·tpb·bn` | 标量 | 无 | ❌ | **block 级索引/任务映射**：每核一次（grid ≤ 512，每核 1 次），不是逐元素热点 |
| 8 | `am = T.min(bm, M-m0)`、`an = T.min(bn, N-n0)` | 标量 | 无 | ❌ | **block 级元数据**：每 tile 2 次标量 min；向量 API 不表达「区段长度」，且它正是 #4 的替代物（净收益仍为正） |
| 9 | `full = T.simd.pset(32)`（lane 掩码） | 标量 | 无 | ❌ | 每条 SIMD 链一次、循环不变（编译期常量掩码），由 `pset` 直接生成，无逐元素代价 |
| 10 | 尾 lane（`an % 64 != 0` 时 chunk 内越界 lane） | — | 无显式处理：越界 lane 落在 `y_ub` 的填充区，**不进** `T.copy` 的写出区段 | ❌ | **动态边界**：真实 workload `N ∈ {18432,2048,1536}` 全为 64 的整数倍 → 该路径不触发；一般 N 的填充 lane 不参与写回，无需掩码（若强行掩码，`PK_B32` 的谓词是 **32-bit lane 模式**，用 16-bit 模式会静默漏写一半槽位 —— `repro/CONST-950-vcvt-narrow-store-pack-b32.py` 的实测陷阱） |

**向量化结论**：全部**逐元素**计算已向量化（#1–#4 替代，API 均有本工具链佐证：`simd.py`
导出 + `engram_gate_conv` / `_delta_kernel` 的已验证用法 + 本设计探针生成的
`vlds_unpack/vcvt/vmul/vadds/vexp/vneg/vdiv_0ulp_ftz_true/vsts_pack_b32` CCE 代码）。
保留 6 处非向量点（#5–#10），理由类别全部为 **tile/block 级调度与索引、动态边界**，
无一处是逐元素标量热点。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

I/O layout 是契约（`gate_up[M, 2N]` / `out[M, N]`，行尾连续，不可改）；**核内布局与
lane 映射是设计变量**。源实现的轴选择是「CUDA 线程铺满 `T.Parallel(block_m, block_n)`」
（§0.3），本表独立评估，不继承该结论。

**候选矩阵**（枚举全部可行布局；弃选给量化理由）：

| # | 布局方案 | 向量化轴 | repack 路径 | 预估收益 / 代价 | 是否采纳 |
|---|---|---|---|---|---|
| **1** | **I/O 原生行主序**，列 tile `(bm, bn)`，chunk = 64 元素（`UNPK_B16`/`PK_B32`） | **N（行内连续）** | 无（`vld`/`vsts` 直接寻址 UB 行内偏移） | 基线即最优：尾 lane 0%（`N % 64 == 0`）、无 repack、UB 49 KB（19.4%） | ✅ **主选** |
| 2 | I/O 原生行主序 + **全行 tile**（`bn = N`，每半区一次连续 2-D 拷贝，扁平 64-lane 循环） | N（同 #1，chunk 边界跨行） | 无 | 收益：MTE2 段数最少（每 buffer 1 段 vs 8–32 段/行块）；代价：`3·bm·N·2 ≤ UB` 限制 `bm`，且单 version（2 version 需 2×）→ N=18432 时 `bm=1` 且 `2×110 KB > 248 KB` **不可双缓冲**，`mt=1` 时 grid=1（饿死 55 核）；N=2048/1536 时可 `bm=8`（98 KB/version） | ⏸ **备选**（仅 N∈{2048,1536}，§1.6.4 实验裁决） |
| 3 | 列主序 / 向量轴 = **M** | M（跨行 stride = 2N 或 N） | 需核内转置 | `vld(addr, dist, post_inc)` **无 stride 参数**（`simd.py` 签名已核），跨步访问只能逐元素 `vld` 一条 lane，指令数 ×64 | ❌ **弃选（API 缺失）** |
| 4 | 扁平视图 `[2M, N]` + `pdintlv` 解交织（gate=偶行、up=奇行） | N | 核内 `pdintlv` 2 条/64 元素 | 收益：两半区一次连续拷贝（但 #1 的行段已 ≥ 1 KiB，2-D 跨段拷贝在段宽 ≥ 数百字节时无惩罚 —— `CONST-gm-to-ub-bw-dilution`，910B2C 外推）；代价：+2 算符/64 元素（11→13，**+18%**） | ❌ **弃选（净负）** |
| 5 | host/前序侧 repack（gate、up 拆成两个独立张量） | N | 两次全量 permute | 代价：多写 `2·M·N·2` + 多读 `2·M·N·2` → 流从 50.3 MB 变 **117.4 MB（+133%）** | ❌ **弃选（流量）** |
| 6 | 原地写回 gate 半区（`out` 与 `gate_up[:, 0:N]` 别名） | N | 无 | 收益 0 字节；代价：破坏 I/O 契约（down GEMM 的 A 与输入张量别名），且 `out_idx` 自动分配的前提消失 | ❌ **弃选（契约）** |

**轴质量评分**（主选 #1 vs 备选 #2）：

| 评分项 | #1 主选 | #2 备选 |
|---|---|---|
| 整除性（vs 64 lane / 512） | `N ∈ {18432,2048,1536}` 全被 512 与 64 整除 ✅ | 同 ✅ |
| 尾 lane 浪费率 | 0%（`bn % 64 == 0`，`an = bn`） | 0% |
| 累加链形态 | 无归约 → 不适用（逐元素） | 同 |
| repack 代价 | 无 | 无 |
| UB 容量影响 | 49 152 B（2 version）/ 19.4% | 98 304 B（**1 version**，`bm=8, N=2048`）/ 38.7%；N=18432 时 110 KB 单 version 且不可双缓冲 |
| MTE2 段数（每 tile） | `2·bm = 32` 段 × 512 B（`bm=16,bn=256`） | 2 段 × 16 KiB（`bm=8,N=2048`） |
| 可覆盖核数 | `mt·nsp ≥ 96`（最差 glm45-t32） | `mt`（N=2048 时 512/32；N=1536 时 32） |

**布局决策结论**：选定 **#1**：核内布局 = I/O 原生行主序、UB tile `(bm, bn)`、**向量化轴 = N**、
chunk = 64 元素、无 repack。该结论已同步落入 §3.3 伪代码（`g_ub[r, c*64]` 的 chunk 基址）
与 §6 循环结构（内层循环维 = N 方向的 chunk 循环），§4.5 的 UB 预算与 §5.2 的 `bm/bn`
取值与该布局一致。**#2 与 #1 的判定裕度落在两个未实测常数上**（MTE2 段开销 vs 双缓冲
带来的重叠收益；`CONST-mte2-degradation` 只给了段≈128 B/4 ns 的 L1 nd2nz 口径，GM→UB
无本型号数据）→ 按实验裁决模式处理（§1.6.4）。

#### 1.6.4 实验裁决计划（主选 #1 vs 备选 #2）

> 触发条件（skill Phase 2 第 3 项）：主选与备选的判定裕度依赖未实测常数（GM→UB 段数开销、
> 单 version 串行化代价）。三件套如下；§3.3/§4/§6 只承载主选 #1。

- **① 主选（进 Stage 3）**：#1，`bm=16, bn=256`（大 shape）/ `bm=1, bn=256`（小 shape），
  2 version + `T.Pipelined(2)`。结构、UB 预算、分核见 §4.5/§5.2/§5.6。
- **② 备选（≤1 个，结构完整可实现）**：#2 全行 tile —— 仅对 `N ∈ {2048, 1536}` 生成：
  `bm = 8`（`3·8·2048·2 = 98 304 B` 单 version；N=1536 时 `3·8·1536·2 = 73 728 B`）、
  `bn = N`（chunk 数 `N/64 = 32 / 24`）、`nsp = 1`（`tpb = 1`，**无 `T.Pipelined`**、
  `annotate_buffer_versions({g_ub: 1, u_ub: 1, y_ub: 1})`）、grid = `mt`（ds-v3-t4096 用 `bm=8` → mt=512；
  glm45-t32 用 `bm=8` → mt=4，只覆盖 4 个核（饿死 52 个）→ **备选不适用于它**，按 shape
  分派时 glm45-t32 固定用主选）。
  分核三要素按新粒度重算：逻辑核数 = `ceil(M/8) × 1`。
- **③ 裁决计划**：
  - 代表 shape：`ds-v3-t4096`（4096×2048 bf16）、`ds-v3-t32`（32×2048 fp16/bf16，
    主观测 dtype 路径）。
  - 指标：kernel-only `Task Duration (us)`（`msopprof` 口径，A/B 交错、3 次取中位）。
  - 判定阈值：`备选 latency < 主选 × 0.97` 才对该 shape 翻转到备选（3% 是 MTE2 段数收益
    的乐观上限，低于它不值得多一条代码路径）。
  - 按 shape 分派可行性：工厂本来就按 `(M, N, dtype)` 专门化（`@target_cache` / 本设计的
    `lru_cache`），因此**按 workload 分派主选/备选是同一工厂内的常数选择**，无需两套代码
    路径共存于同一 shape。
  - 回写路径：结果经 `perf_feedback.md` + 附录补记固化 §1.6.3 判定；若备选全局胜出，
    以 `[DESIGN_LIMIT]` 交 conductor 走设计修订路径。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**：`simd`（SIMD，`T.SimdVF()` + `T.simd.*` 寄存器级向量 intrinsic）。
依据 `statectl platform --task-dir` 返回的 `programming_modes.choices`
（`cube: Cube` / `hybrid: 混合` / `simd: SIMD` / `simt: SIMT`）与
`migration_default: simd`（本任务无用户显式覆盖）。

### 2.2 选型理由

1. **计算类型**：纯逐元素（§5.1），零 MAC、零规约、无矩阵收缩 → **Cube 与混合模式不适用**
   （`T.gemm` 无操作数可算；Cube 只做矩阵）。
2. **SIMD vs SIMT**：两者都能表达逐元素表达式，差别在「谁来铺 lane」：
   - `T.SimdVF` + `T.simd.*`：显式 64-lane 向量寄存器链（`vld → vexpr → vsts`），
     每 64 元素**恰好** 11 条指令（§1.6.0 R3），无逐 lane 地址计算、无谓词；
   - `T.SimtVF(threads=…)`：逐元素标量表达式，`threads` 个线程在 VF 内分工（平台文档只说
     「线程域位于 VF 内」，线程到向量 lane 的具体映射**未文档化**），表达式中的
     `T.cast`/`T.exp` 逐 lane 下沉；已验证路径见 `kernel_template.py`、
     `PL-950-row-primitives`。
   本算子的算术链短（4 个算符 + 3 次转换）、无归约、无 mask 逻辑，**显式 SIMD 链的指令数
   与可预测性最好**，且 `vexp/vdiv` 的精度变体（0 ulp 除法）在 `T.simd` 层可直接指定。
3. **数据布局匹配**：输入输出都是行尾连续，`vld/vsts` 的 64 元素 chunk 天然对齐
   （`N % 64 == 0`），无需 strided lane 映射（§1.6.3 #3 已证 strided 不可行）。
4. **SIMT 作为备选**：若 Stage 3/4 发现 SIMD 链在某 dtype 上受限，`T.SimtVF` 版本是
   结构等价的降级路径（§9.3），不改变本设计的任何布局或分核决策。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_shared(shape, dtype)` → UB（`shared.dyn`）；无 L1/L0A/L0B/L0C 分配 |
| 计算方式 | `T.SimdVF()` 内 `T.simd.vld/vcvt/vneg/vexp/vadds/vmul/vdiv/vsts`；无 `T.Parallel` 逐元素 |
| 线程/核模型 | `T.Kernel(grid)` 一维，**无 `threads` 参数**（平台实现拒绝该参数）；并行度 = `grid` 个 block × 每核 64 lane |
| 同步 | 默认自动调度（`allow_autoschedule`），无手写 flag（§7） |
| 流水 | `T.Pipelined(tpb, num_stages=2)` + `T.annotate_buffer_versions`（3 个 buffer × 2 version） |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式为 §1.6.1 的**优化后公式**（A1），与 §1.6.2 的向量化结论一致。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `G = gate_up[m0:m0+am, n0:n0+an]`；`U = gate_up[m0:m0+am, N+n0:N+n0+an]` | 两个半区的区段（源语义 `[:, :N]` / `[:, N:]`） |
| 2 | `g = fp32(G)`, `u = fp32(U)` | 精确加宽（16→32 bit），逐 64 lane |
| 3 | `t = exp(-g)` | fp32 SFU 指数 |
| 4 | `d = 1 + t` | 标量加（`vadds`） |
| 5 | `num = g · u` | fp32 逐 lane 乘 |
| 6 | `y = num / d` | fp32 单次除法（0 ulp 精确变体） |
| 7 | `out = cast_dtype(y)` | 唯一一次舍入（RNE）+ 打包写回 |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 / 动作 | TileLang API（本平台实查签名） | 参数 | 模式 |
|------|----------------|------------------------------|------|------|
| 0 | 核网格 | `T.Kernel(grid)` | 一维 `grid`（无 `threads`） | SIMD |
| 1 | GM→UB 区段搬入 | `T.copy(src_region, dst_region)` | `gate_up[m0:m0+am, n0:n0+an] → g_ub[0:am, 0:an]`；up 同理（`N+n0` 偏移） | MTE2 |
| 1b | UB 分配 | `T.alloc_shared((bm, bn), dtype)` | `bm ∈ {1,16}`、`bn = 256` | UB（`shared.dyn`） |
| 1c | 双缓冲版本 | `T.annotate_buffer_versions({g_ub: 2, u_ub: 2, y_ub: 2})` | 正整数版本数 | 流水 |
| 1d | 列 tile 循环 | `T.Pipelined(tpb, num_stages=2)` | `tpb = nt/nsp` | 流水 |
| 2 | 进入向量域 | `T.SimdVF(latency=0)` | 默认参数 | SIMD |
| 2a | 行 / chunk 调度 | `T.serial(bm)`、`T.serial(bn // 64)` | 静态上界（`bm, bn` 编译期常量） | 调度 |
| 3 | lane 掩码 | `T.simd.pset(32)` | `elem_width=32` → `PAT_ALL`（64 个 32-bit lane 全开） | SIMD |
| 4 | 16-bit 载入 | `T.simd.vld(g_ub[r, c*64], dist="UNPK_B16")` | `UNPK_B16`：64 个 16-bit 元素展开到 64 个 32-bit lane | SIMD |
| 5 | 加宽 | `T.simd.vcvt(v, "float32", part=0)` | `part=0`（单 value/word），`round`/`sat` 默认 | SIMD |
| 6 | `-g`、`exp`、`+1`、`g·u`、`/ d` | `T.simd.vneg(g, full)`；`T.simd.vexp(t, full)`；`T.simd.vadds(t, 1.0, full)`；`T.simd.vmul(g, u, full)`；`T.simd.vdiv(num, den, full)` | `mask=full`、`mode=MODE_ZEROING` 默认 | SIMD |
| 7 | 收窄 + 打包写回 | `T.simd.vcvt(y, dtype, round="ROUND_R", sat=False, part=0)` → `T.simd.vsts(y_ub[r, c*64], v, full, dist="PK_B32")` | `PK_B32`：把「每 32-bit 字低半」的 64 个结果打包成 64 个连续 16-bit | SIMD |
| 8 | 区段边界 | `T.min(bm, M - m0)`、`T.min(bn, N - n0)` | 动态 extent | 元数据 |
| 9 | UB→GM 写回 | `T.copy(y_ub[0:am, 0:an], out[m0:m0+am, n0:n0+an])` | 一次写 | MTE2 |
| 10 | 工厂专门化 | `functools.lru_cache` + `tilelang.compile(main, target="ascend", out_idx=[1])` | 按 `(M, N, dtype)` 专门化 | host |

### 3.3 计算伪代码

（与 `history_version/design_probe_compile.py` 的 P1–P4 同构，4 例均无卡编译通过；
P2 覆盖 `bn = 256` 的主选档，P1/P4 覆盖 `bn = 512` 的 Stage 4 档）

```python
VL = 64
CORES = 56          # Ascend950PR_9579 vector_core_cnt（§5.5 实查记录）

def _geometry(M, N, block_m, block_n):
    """构建期几何：tile、每核列 tile 数、grid（§5.2 规则 1–5）。"""
    bn = min(block_n, 512)
    bn -= bn % VL                       # 对齐到 64 lane
    bm = 16 if M >= 64 else 1           # 两档行 tile（§5.2 规则 2）
    bm = min(bm, M)
    mt, nt = -(-M // bm), -(-N // bn)
    # nsp：nt 的因子中取使 |mt*nsp - 4*CORES| 最小者（grid 目标 ≈ 4 波，§5.2 规则 3）
    nsp = grid_target_divisor(mt, nt, CORES)
    return bm, bn, mt, nt, nsp, nt // nsp, mt * nsp   # bm, bn, mt, nt, nsp, tpb, grid

@functools.lru_cache(maxsize=16)
def _silu_mul_fused_kernel(M, N, dtype_str):
    def _func(block_m, block_n):                      # 源签名去掉 threads（K9）
        bm, bn, mt, nt, nsp, tpb, grid = _geometry(M, N, block_m, block_n)
        nchunk = bn // VL
        assert 2 * 3 * bm * bn * 2 <= 253952          # UB 预算（含 2 version）

        @T.prim_func
        def _main(gate_up: T.Tensor((M, 2 * N), dtype_str),
                  out: T.Tensor((M, N), dtype_str)) -> None:
            with T.Kernel(grid) as bx:
                im = bx // nsp
                sp = bx % nsp
                g_ub = T.alloc_shared((bm, bn), dtype_str)
                u_ub = T.alloc_shared((bm, bn), dtype_str)
                y_ub = T.alloc_shared((bm, bn), dtype_str)
                T.annotate_buffer_versions({g_ub: 2, u_ub: 2, y_ub: 2})
                for it in T.Pipelined(tpb, num_stages=2):
                    tn = sp * tpb + it
                    m0, n0 = im * bm, tn * bn
                    am, an = T.min(bm, M - m0), T.min(bn, N - n0)
                    T.copy(gate_up[m0:m0 + am, n0:n0 + an], g_ub[0:am, 0:an])
                    T.copy(gate_up[m0:m0 + am, N + n0:N + n0 + an], u_ub[0:am, 0:an])
                    with T.SimdVF():
                        full = T.simd.pset(32)
                        for r in T.serial(bm):
                            for c in T.serial(nchunk):
                                g = T.simd.vcvt(
                                    T.simd.vld(g_ub[r, c * VL], dist="UNPK_B16"),
                                    "float32", part=0)
                                u = T.simd.vcvt(
                                    T.simd.vld(u_ub[r, c * VL], dist="UNPK_B16"),
                                    "float32", part=0)
                                num = T.simd.vmul(g, u, full)
                                den = T.simd.vadds(
                                    T.simd.vexp(T.simd.vneg(g, full), full), 1.0, full)
                                y = T.simd.vdiv(num, den, full)
                                T.simd.vsts(y_ub[r, c * VL],
                                    T.simd.vcvt(y, dtype_str, round="ROUND_R",
                                                sat=False, part=0),
                                    full, dist="PK_B32")
                    T.copy(y_ub[0:am, 0:an], out[m0:m0 + am, n0:n0 + an])

        return tilelang.compile(_main, target="ascend", out_idx=[1])
    return _func
```

### 3.4 API 可行性确认

| API | 来源确认 | 验证状态 |
|---|---|---|
| `T.Kernel(grid)`（无 threads，一维） | `ascend/language/kernel.py:Kernel`（`programming.md`「签名没有 threads，多维网格拒绝」） | ✅ 探针 P1–P4 编译 |
| `T.SimdVF()` / `T.SimtVF(threads)` | `ascend/language/frame.py`（`programming.md` 已核） | ✅ 探针 |
| `T.alloc_shared` / `T.annotate_buffer_versions` / `T.Pipelined` | `allocate.py` / `annotations.py` / `programming.md` | ✅ 探针 P1/P2/P4 |
| `T.copy(region, region)` 动态 extent | `copy_op.py` + `kernel_template.py` 的 `tail_size` 写法 | ✅ 探针 P3（`am=1`、`an=476`） |
| `T.simd.vld(addr, dist="UNPK_B16")` | `simd.py` 导出 + `engram_gate_conv` 已验证 16-bit 载入 | ✅ 探针；生成的 CCE 为 `vlds_unpack<half>` |
| `T.simd.vcvt(src, "float32", part=0)` | 同上 + `_delta_kernel.py`（`PART_EVEN`） | ✅ 生成 `vcvt<float>(..., PART_EVEN, MODE_ZEROING)` |
| `T.simd.vexp / vneg / vadds / vmul / vdiv` | `simd.py` 导出（`vdiv` 默认「0 ulp 精确」变体）；`engram_gate_conv` 的 silu 链同款 | ✅ 生成 `vexp` / `vneg` / `vadds` / `vmul` / `vdiv_0ulp_ftz_true` |
| `T.simd.vcvt(y, dtype, round="ROUND_R", sat=False, part=0)` + `vsts(dist="PK_B32")` | `repro/CONST-950-vcvt-narrow-store-pack-b32.py`（窄化必须 `PK_B32` + 32-bit mask） | ✅ 生成 `vsts_pack_b32(vcvt<half>(...))` |
| `T.simd.pset(32)` | `engram_gate_conv` / `_delta_kernel` | ✅ 探针 |
| `T.min`（区段） | `tilelang/language/tir/ir.py` + `kernel_template.py` | ✅ 探针 P3 |

**无卡编译探针结论（D-3，`history_version/design_probe_compile.py`，2026-11-23）**：
`ASCEND_RT_VISIBLE_DEVICES= /workspace/venv-ascend950/bin/python history_version/design_probe_compile.py`
→ `PROBE_RESULT: ALL_OK`（4/4 COMPILE OK）：

| 探针 | M, N, dtype | (bm, bn, nsp) | grid | tpb | UB | 覆盖的构造 |
|---|---|---|---|---|---|---|
| P1 | 4096, 2048, bf16 | (16, 512, 2) | 512 | 2 | 98 304 B | 大 tile + 流水 + 双版本 + bf16 `vexp/vdiv` |
| P2 | 32, 1536, fp16 | (1, 256, 3) | 96 | 2 | 3 072 B | **fp16 路径**（`vcvt` 收窄 RNE）+ 本设计主选的 `bn=256` |
| P3 | 1, 1500, bf16 | (1, 512, 3) | 3 | 1 | 6 144 B | **双尾块**（`am=1`、`an=476`）+ `T.Pipelined(1)` 退化 |
| P4 | 512, 18432, bf16 | (16, 512, 9) | 288 | 4 | 98 304 B | 长 `tpb` 流水 + 最大 N |

探针的生成代码（P2）已核对为本设计期望的指令序列：
`vlds_unpack<half>` → `vcvt<float>(PART_EVEN)` → `vmul` → `vadds(vexp(vneg(...)), 1.0)` →
`vdiv_0ulp_ftz_true` → `vcvt<half>(ROUND_R, RS_DISABLE)` → `vsts_pack_b32`。
（探针用 `bn=512` 覆盖更大的 tile；主选的 `bn=256` 由 P2 覆盖——两者同一代码路径。）

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| `T.Kernel` 一维、签名无 `threads`，多维网格被拒 | **Yes** | 一维 `grid`；`bm/bn` 的二维划分在核内用循环表达（§6） |
| `SIMD_VF` 区域内不得读 GM（必须先 stage 到 UB） | **Yes** | 所有 `T.copy` 在 `T.SimdVF()` 之外（与 `engram_gate_conv` 同构） |
| 整 buffer 拷贝要求元素总数相等；局部拷贝须显式 region | **Yes** | 全部用显式 region（`[m0:m0+am, n0:n0+an]`） |
| 窄化 `vcvt` 的结果落在偶数 lane（每 32-bit 字一个值），稠密存储必须 `PK_B32` + 32-bit lane 掩码 | **Yes** | `round="ROUND_R", sat=False, part=0` + `dist="PK_B32"` + `pset(32)`（§3.4 引用了该陷阱的实测条目） |
| `T.clear` 在多 KB fp32 UB 上会降级为标量循环 | No | 不使用 `T.clear`（死 lane 不写回，无需清零；`engram_gate_conv` 的实测教训已避开） |
| UB 容量 253 952 B（`Ascend950PR_9579.ini [AICoreSpec] ub_size`） | **Yes** | 峰值 49 152 B（19.4%）；按 1.10–1.12 膨胀 = 21.8% ✅（§4.5） |
| L1 512 KB / L0C 256 KB / L0A·L0B 64 KB + 矩阵分形/nd2nz 约束 | No | 本算子无 Cube 操作，不使用 L1/L0 |
| 跨 block barrier / 跨核 flag 未认证 | No | 单 kernel、单阶段、无跨核通信 |
| 动态 shape（`T.dyn`） | No | `M/N/dtype` 全为编译期常量；工厂按 `(M,N,dtype)` 专门化 |
| 自动调度默认开启（`allow_autoschedule`），`InsertSync`/`MaterializeMultiBuffer` 生效 | **Yes** | 依赖自动同步，不手写 flag（§7）；版本数用 `annotate_buffer_versions` 显式声明 |

#### 3.5.2 参考实现差异说明

| 差异项 | 参考实现（GPU） | 本项目（Ascend） | 转换方案 |
|--------|----------------|-----------------|---------|
| Kernel 结构 | 一维 `T.Kernel(num_blocks_m*num_blocks_n, threads=256)`，二维下标手算 | 一维 `T.Kernel(grid)`（无 threads），核内列 tile 循环 | §0.6 R1/R3；grid 从 `mt·nt` 改为 `mt·nsp`（每核 `tpb` 个 tile） |
| 并行原语 | `T.Parallel(block_m, block_n)` + 线程映射 + nvcc 向量化 | `T.SimdVF` + `T.simd.*` 64-lane 显式链 | §3.2 步骤 4–7 |
| 内存层级 | 无 SMEM，寄存器直读 GM（依赖 L2/合并访问） | GM→UB（`g_ub/u_ub`）→寄存器→UB（`y_ub`）→GM，2 version + 流水 | §0.6 R2；§4.4 |
| 中间精度 | `T.cast(g, "float32")` 逐元素 | `vcvt(UNPK_B16 → fp32)` 向量 | 语义相同（精确加宽），实现层替换 |
| 尾块 | 逐元素 `if` 守卫 | 动态区段 extent（`T.min`） | §0.5 O4 等价替换 |
| 编译开关 | `TL_ENABLE_FAST_MATH=True`、`-O3`、`-DENABLE_BF16` | 无（BishengIR 无对应开关；bf16 原生） | §0.5 O7 舍弃 |
| 内层工厂签名 | `_func(block_m, block_n, threads)` | `_func(block_m, block_n)` | K9：去掉 `threads`，与 NPU wrapper 的现有调用 `(block_m, block_n)` 一致 |
| 配置取值 | `block_m=128, block_n=256` | 夹取为 `bm ∈ {1,16}`、`bn = min(block_n, 512)` | §5.2 规则 1–2（避免 UB 溢出；wrapper 传 128×256） |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/engram_gate_conv/_engram_gate_conv_fwd_kernel/_engram_gate_conv_fwd_kernel.py` | **极高**（同一后端、同样含 silu 与 64-lane 向量链） | silu 的 SIMD 实现 `vdiv(acc, vadds(vexp(vneg(acc)),1))`；16-bit `vld(dist="UNPK_B16")` + `vcvt(fp32, part=0)`；窄化 `vcvt(dtype, round="ROUND_R", sat=False, part=0)` + `vsts(dist="PK_B32")`；`pset(32)` 全 lane 掩码；SIMD 帧外做 `T.copy` |
| `examples/mla_dense_bwd/_delta_kernel/_delta_kernel.py` | 高（同为 `target="ascend"` 的一维 grid + 核内 tile 流水） | `T.Kernel(grid)` + `nsp/tpb` 几何（每核 `tpb` 个 tile）、`T.Pipelined(tpb, num_stages=2)`、`annotate_buffer_versions({...:2})`、静态尾段与 `T.min` 区段、UB 预算自查 |
| `examples/mla_dense_bwd/_dq_land_kernel/repro/CONST-950-vcvt-narrow-store-pack-b32.py` | 方法学（陷阱证据） | 窄化 `vcvt` 必须 `PK_B32` + 32-bit 掩码，否则静默漏写一半槽位 |
| `.agents/platforms/toolchains/ascend/kernel_template.py` | 高（平台官方逐元素模板） | 一维 `T.Kernel`、`tile/tail` 两段式尾块处理、`T.alloc_shared` UB 缓冲、`T.copy` 区段 |
| `examples/TileOPs/tileops/kernels/elementwise/mish/mish_kernel/mish.py` | 中（同为「激活函数迁移」的工程先例） | golden 的 fp32 上取整 + 回舍入写法、`block_size` 折叠 `threads·npt`、L0/L1/Boundary 分层；**注意其 API 属 npuir 后端，本设计只借鉴数学与工程结构** |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `gate_up` | `[M, 2N]` | float16 / bfloat16 | 前半列 gate、后半列 up；尾轴连续；`M = T`、`N = S`（§0.2） |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `out` | `[M, N]` | 同输入 | `out_idx=[1]` 自动分配；是 down GEMM（`n=H, k=S, trans_b=True`）的 A |

**契约**：`out` 与 `gate_up` 无别名（§1.6.3 候选 #6 已弃选），调用方可假定输入不被修改。

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| `g_ub` | `(bm, bn)` = `(16, 256)` / `(1, 256)` | 输入 dtype（2 B） | UB（`shared.dyn`） | gate 半区 tile，**2 version** |
| `u_ub` | 同上 | 同上 | UB | up 半区 tile，**2 version** |
| `y_ub` | 同上 | 输出 dtype（同输入） | UB | 结果 tile，**2 version** |
| 向量寄存器 | 64 lane × fp32 × 4 个活跃值（`g/u/num/den`） | float32 | Vector 寄存器 | 无 UB 占用（不物化） |

无 GM 中间缓冲、无 fp32 中间张量（对比 §1.6.0 C7 的 `+16.8 MB`）。

### 4.4 内存搬运路径

```
GM gate_up[M, 2N]                                    GM out[M, N]
   │  T.copy (MTE2, 区段 [m0:m0+am, n0:n0+an])           ▲
   ▼                                                     │ T.copy (MTE2, 区段写回)
UB g_ub[bm,bn] (v0) / (v1)                               │
UB u_ub[bm,bn] (v0) / (v1)                               │
   │  T.simd.vld(UNPK_B16)                              │
   ▼                                                     │
Vector 寄存器 (64 lane fp32)：vcvt → vneg/vexp/vadds/vmul/vdiv
   │  T.simd.vcvt(narrow, ROUND_R) + vsts(PK_B32)        │
   ▼                                                     │
UB y_ub[bm,bn] (v0) / (v1) ──────────────────────────────┘
```

- 每元素 GM 往返：gate 读 1 次 + up 读 1 次 + out 写 1 次（= 下界，§1.6.0 R3）。
- 无 L1 / L0A / L0B / L0C 参与（无 Cube）。
- 无 layout 变换（`UNPK_B16`/`PK_B32` 只是寄存器打包方式，UB 内布局仍是行主序）。

### 4.5 UB 内存预算

| Buffer | Shape | dtype | 单 version (Bytes) | version 数 | 小计 (Bytes) |
|--------|-------|-------|--------------------|-----------|--------------|
| `g_ub` | (16, 256) | bf16/fp16 | 8 192 | 2 | 16 384 |
| `u_ub` | (16, 256) | 同上 | 8 192 | 2 | 16 384 |
| `y_ub` | (16, 256) | 同上 | 8 192 | 2 | 16 384 |
| **总计（大 shape）** | | | 24 576 | | **49 152** |
| 小 shape（`bm=1`）| (1, 256) | 同上 | 512 | 2 | **3 072** |

- 容量来源：`Ascend950PR_9579.ini [AICoreSpec] ub_size = 253952`（平台 `design_budget.capacity_bytes.UB`）。
- 占比：49 152 / 253 952 = **19.4%**；按显式 alloc 结构的实测膨胀系数 1.10–1.12
  （`CONST-capacity-910B2C`，910B2C 实测，本型号未实测）放大后 **21.8%** ✅。
- 该预算使「#2 全行 tile 备选」的 `bm` 也可以到 8（§1.6.4），即 UB 不是主选的约束边界。

### 4.6 动态轴定义

**无**。`M`、`N`、`dtype` 均为编译期常量（源工厂同样按 `(M, N, dtype)` 用
`@target_cache` 专门化）；动态量只有 tile 边界的 `am/an`（由 `T.min` 从静态常量推出），
不构成 `T.dyn`。

### 4.7 JIT 配置

```python
@functools.lru_cache(maxsize=16)          # 源用 @target_cache(maxsize=16)，等价专门化
def _silu_mul_fused_kernel(M, N, dtype_str):
    ...
    return tilelang.compile(_main, target="ascend", out_idx=[1])
```

与源的差异：去掉 `pass_configs={TL_ENABLE_FAST_MATH: True}`（nvcc 概念，本后端无对应开关）
与 `compile_flags=["-O3","-DENABLE_BF16"]`（BishengIR 无对应 flag，bf16 原生）；
`out_idx=[1]` 保留（输出自动分配）。

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**：**纯 Vector**（AIV SIMD）。判定依据：算子只有逐元素算术与 dtype 转换，无 MAC、
无归约、无窗口、无 gather/scatter（§0.3 的 7 个计算步骤全部是逐元素映射）。因此：
`bm/bn` 只决定「一个 block 搬多少、算多少」，不受 Cube 分形、L0 容量或 L1 端口约束。

### 5.2 Block 划分

```python
bn = min(block_n, 512); bn -= bn % 64      # 规则 1：≤512 且对齐 64 lane
bm = 16 if M >= 64 else 1                   # 规则 2：两档行 tile
bm = min(bm, M)
mt, nt = ceil(M / bm), ceil(N / bn)
nsp = argmin_{d | nt} |mt*d - 4*56|          # 规则 3：grid 目标 ≈ 4 波（224）
tpb = nt // nsp
grid = mt * nsp
assert 2 * 3 * bm * bn * 2 <= 253952         # 规则 4：UB（含 2 version）
```

**配置契约**（设计期确定，写进 kernel 的构建期断言/夹取，**防 UB 溢出**）：

1. **`bn` 上限 512、下界对齐 64**：`bn = min(block_n, 512)` 后向下取整到 64 的倍数。
   源 wrapper 传 `block_n = 256`（`SharedExpertMLPKernel.default_config`）→ 实际 `bn = 256`。
   依据：`vld/vsts` 的 chunk = 64 元素；`bn = 512` 时单 buffer 16 KiB/version，是 UB 与
   MTE2 段宽的平衡点（Stage 4 可把它当调优旋钮，见 §1.6.4 备注）。
2. **`bm` 两档**：`M ≥ 64 → bm = 16`（每 buffer 8 KiB，行段 512 B，`tpb` 可达 8 层流水）；
   `M ≤ 32（decode 段）→ bm = 1`（`bm = 16` 会把逻辑核压到 16–32 个、饿死 56 核的 40–70%，
   见 §5.5 的比较）。`bm = min(bm, M)` 保证不留 0 行块。
3. **`nsp / tpb`**：在 `nt` 的因子中取 `grid = mt·nsp` 最接近 `4 × 56 = 224` 者
   （同距离取较小 grid，即较少派发、较长流水）；`tpb = nt/nsp ≥ 1`。
4. **UB**：`2 version × 3 buffer × bm·bn·2 B ≤ 253 952 B`（构建期 assert；实际用量见 §4.5）。
5. **`block_m/block_n` 是输入提示而非最终值**：wrapper 传 `(128, 256)`，`128` 若不夹取会
   需要 `2·3·128·256·2 = 393 216 B > 248 KiB` → 编译期失败。夹取后与 §5.6 的表一致。

**block 数**：`mt · nt` 个 tile，由 `mt · nsp` 个 block 承担（每 block `tpb` 个 tile）。
非整除由 `am/an` 的动态 extent 处理（§5.4），**不做 host 侧 padding**。

### 5.3 约束分析

- **对齐约束**：`bn % 64 == 0`（chunk 整除，尾 lane 0 浪费）；`bm·bn·2` 是 2 的幂倍数
  （8 KiB / 512 B），UB 起始偏移天然 512 B 对齐；`gate_up` 行段起点 `n0·2` 与 `(N+n0)·2`
  在 `N % 64 == 0` 时均为 128 B 的整数倍（本 workload `N ∈ {18432,2048,1536}` ✅）。
- **UB 容量**：49 152 B / 253 952 B = 19.4%（×1.10–1.12 膨胀 = 21.8%）✅ 来源见 §4.5。
- **L0 容量 / L1**：不适用（无 Cube 操作）。
- **事件/flag 预算**：不手工分配 flag（自动调度），不适用 `CONST-flag-id-budget` 的 15/核限制。
- **向量寄存器压力**：每个 chunk 同时活跃 4 个 fp32 向量（`g, u, num, den`）+ 1 个掩码
  = 5 个 2048-bit 寄存器；`SimdVF` 的分配由编译器完成，无 `alloc_local` 需求（对比
  `engram_gate_conv` 的循环携带累加器场景）。

### 5.4 注意事项（非整除与边界）

1. **M 尾块**：`am = T.min(bm, M - m0)`。`M = 1` 且 `bm = 1` 时 `am = 1`（无尾块）；
   `M = 32/512/4096` 被 `bm = 1/16/16` 整除（无尾块）。理论上 `bm = 16` 与 `M % 16 ≠ 0`
   的组合走 `am < bm`：装载只覆盖 `[0, am)` 行，计算仍按静态 `bm` 行执行（死行读到的 UB 内容
   是未初始化值，可能非有限，但**不写回**——`T.copy` 的写出区段是 `[0:am)`），与源实现的
   「越界行被守卫丢弃」语义等价。探针 **P3** 已编译验证 `am = 1 < bm = 1`（该例 bm=1）与
   `an = 476 < bn = 512` 的动态 extent 路径。
2. **N 尾块**：`an = T.min(bn, N - n0)`。本 workload 的 `N` 全被 256 整除（无尾块）。
   一般 `N` 不被 `bn` 整除时：装载 `[0:an)`、计算按静态 `bn`（chunk 数 `bn/64`），最后一个
   chunk 的越界 lane 落在 `y_ub` 的填充区、**不进入**写出区段 → 结果正确。
   **不要**给 `PK_B32` 存储加 16-bit 掩码（会静默漏写一半槽位，见
   `repro/CONST-950-vcvt-narrow-store-pack-b32.py`）。
3. **`N < 64`**：`bn = max(64, ...)` 后 `N < bn` → `an = N`、chunk 内 `N` 个有效 lane，
   其余为死 lane；本 op 的 manifest 不含该情形（`S ≥ 1536`），列入 §9.2 E3 的防御性说明。
4. **`M = 1`**（kimi-k2-t1）：`bm = 1`、`grid = 72`（= `nt`，`nsp = 72`、`tpb = 1`）；
   单个 block 只搬 512 B × 3，属发射主导（§1.6.0 结论 2）。
5. **不需要 `T.clear`**：死 lane 不写回，且 `T.clear` 在多 KB fp32 UB 上会降级为标量循环
   （`engram_gate_conv` 的实测教训）。若 Stage 3 发现未初始化 UB 的非有限值引起数值陷阱
   （理论上不会：`vexp/NaN` 只影响死 lane），再按需清零填充区。

### 5.5 分核策略（物理核数适配）⭐

> 三要素标准：`.agents/skills/_shared/standards/core-split-strategy.md` §1。

- **① 逻辑核数计算**：`num_logical_kernels = grid = mt × nsp`，其中 `mt = ceil(M/bm)`、
  `nsp | nt`、每核处理 `tpb = nt/nsp` 个 `(bm, bn)` tile。逐 workload：

  | workload | M | N | bm | bn | mt | nt | nsp | tpb | **逻辑核数** |
  |---|---|---|---|---|---|---|---|---|---|
  | kimi-k2-t1 | 1 | 18432 | 1 | 256 | 1 | 72 | 72 | 1 | **72** |
  | kimi-k2-t512 | 512 | 18432 | 16 | 256 | 32 | 72 | 6 | 12 | **192** |
  | ds-v3-t32 | 32 | 2048 | 1 | 256 | 32 | 8 | 8 | 1 | **256** |
  | ds-v3-t4096 | 4096 | 2048 | 16 | 256 | 256 | 8 | 1 | 8 | **256** |
  | glm45-t32 | 32 | 1536 | 1 | 256 | 32 | 6 | 6 | 1 | **192** |

- **② 物理核数依据（任务平台实查，非假设值）**：目标型号 **ascend-950pr-9579**
  （dav-3510）。平台 `statectl platform --task-dir <任务目录>` 落库记录（读取日期 2026-11-23）：

  ```
  hardware.core_source = "Ascend950PR_9579.ini [SoCInfo] ai_core_cnt=28,
      cube_core_cnt=28, vector_core_cnt=56;
      runtime probe rtGetAiCoreCount()=28 on Ascend950PR_9579, 2026-11-23"
  hardware.observed_cores = {"cube": 28, "vector": 56}
  ```

  平台 ruleset 规定的查询口径是 `torch.npu.get_device_properties`（分别记录 Cube/Vector）：

  ```python
  import torch, torch_npu
  p = torch.npu.get_device_properties(torch.npu.current_device())
  p.name, p.cube_core_num, p.vector_core_num   # -> "Ascend950PR_9579", 28, 56
  ```

  本算子**纯 Vector**，故物理核数取 **56（AIV）**；28 个 AI Core 各含 2 个 Vector 单元，
  逻辑 block 由硬件块调度器派发到 AIV。**设计期该卡被其他算子（Stage 3）占用**
  （`ASCEND_RT_VISIBLE_DEVICES=""`，仅做无卡编译探针），因此引用平台已落库的探测记录；
  Stage 3 首跑前必须用上面同一段代码复核型号与核数（§9.1 风险 6）。
  （block→AIV 的映射细节无本地文档条目，标注为**未文档化假设**：grid 个 block 由硬件
  调度器按波次铺到 56 个 AIV，无 occupancy 上限约束——本算子每 block UB 仅 49 KB / 19.4%，
  若实际每 AIV 只驻留 1 个 block，则 §6.3 的核内双缓冲就是唯一的 MTE2/计算重叠手段。）

- **③ 规模判定与分核方案**：逻辑核数 **72–256** = **1.29–4.57 波**，属「**中等规模**」
  （逻辑核数 > 物理核数、但远未到必须核内串行的量级）。方案：
  - **对齐物理核整数倍**：`nsp` 按「grid 最接近 4 波（224）」选取，得到
    `192 / (4×56) = 86%`（kimi-k2-t512、glm45-t32）、`256 / (5×56) = 91%`
    （ds-v3-t32、ds-v3-t4096）。**为什么不是 100%**：`nsp` 必须是 `nt` 的因子
    （`nt ∈ {6, 8, 72}`），可选 grid 集合为离散值；在可选点里
    `192/224` 与 `256/280` 已是最大利用率，代价是尾波最多空 32 个核——
    对**带宽受限**的算子，尾波的空核由剩余核的带宽共享补偿（不是纯算力浪费），
    因此不为此改 `bm/bn`（改 `bm` 会同时改变 UB 与段宽，收益无法纸面判定，见 §1.6.4）。
  - **核内分批（`tpb > 1`）**：kimi-k2-t512（`tpb = 12`）与 ds-v3-t4096（`tpb = 8`）
    每核串行走多个列 tile——这里的核内循环是**为双缓冲流水服务的**（§6.3），不是为压缩
    核数；`am/an` 与 tile 遍历序都满足平台的「一维 grid + 静态循环上界」限制。
  - **`M = 1` 的特例**：逻辑核数上限 = `nt = 72`（`mt = 1`），利用率 64%
    （72/112）。该 shape 总流量 0.11 MB、核内仅 4 条 chunk 链（`bn/64 = 4`），
    **是发射主导而非并行度不足**（§1.6.0 结论 2），故不为其引入 `bm < 1` 之类的切分。

### 5.6 每 workload 的推荐配置

（`bn = 256`、`bm ∈ {16, 1}`、dtype 按 workload；UB 用量按 2 version 计）

| workload | dtype | (M, N) | bm | bn | nsp | tpb | grid | 波数（/56） | 利用率 | UB |
|---|---|---|---|---|---|---|---|---|---|---|
| kimi-k2-t1 | bf16 | (1, 18432) | 1 | 256 | 72 | 1 | 72 | 1.29 | 64% | 3 072 B |
| kimi-k2-t512 | bf16 | (512, 18432) | 16 | 256 | 6 | 12 | 192 | 3.43 | 86% | 49 152 B |
| ds-v3-t32 | bf16 / fp16 | (32, 2048) | 1 | 256 | 8 | 1 | 256 | 4.57 | 91% | 3 072 B |
| ds-v3-t4096 | bf16 | (4096, 2048) | 16 | 256 | 1 | 8 | 256 | 4.57 | 91% | 49 152 B |
| glm45-t32 | bf16 | (32, 1536) | 1 | 256 | 6 | 1 | 192 | 3.43 | 86% | 3 072 B |

> `block_n = 512` 是 Stage 4 的一级旋钮（`bn` 翻倍 → 段宽 1 KiB、chunk 数翻倍、UB 38.7%）；
> `bm = 32` 虽满足 UB（`2·3·32·256·2 = 98 304 B`），但不作主选：`mt` 减半会减少逻辑核数，
> 列入 Stage 4 的调优清单。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

> 逐元素计算已按 §1.6.2 全部向量化——下表只有 block 级 / tile 级调度循环与**已经向量化**
> 的 chunk 循环，无逐元素标量循环。

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| block（`grid`） | 并行派发 | `T.Kernel(grid)`（一维） | 每 block 一个（行块, 列段）任务；硬件按波次铺 56 个 AIV |
| 列 tile（`tpb`） | **软件流水** | `T.Pipelined(tpb, num_stages=2)` | 双缓冲把 MTE2 与向量链重叠（§6.3） |
| 行（`bm`） | 串行调度 | `T.serial(bm)` | 每迭代驱动 8 条 64-lane 指令；不展开（展开只增代码体积、不改指令数） |
| chunk（`bn/64`） | 串行调度（**向量化轴**） | `T.serial(bn // 64)` | 每迭代 = 1 条 64-lane 向量算符，已完全向量化（§1.6.2 #6） |
| 元素（64 lane） | 向量 | `T.simd.*`（`vld/vcvt/vexp/vdiv/vsts`） | 64 元素/指令，lane 内无标量路径 |

### 6.2 循环伪代码

见 §3.3（与探针 P1–P4 同构）。结构要点：`T.copy` × 2 在 `T.SimdVF()` 之外（SIMD 帧内不能
读 GM）；`T.SimdVF()` 内只有 `T.serial` 调度循环 + `T.simd.*` 指令；写回 `T.copy` 在帧外。

### 6.3 流水线优化

- **`T.Pipelined(tpb, num_stages=2)`** + `T.annotate_buffer_versions({g_ub: 2, u_ub: 2,
  y_ub: 2})`：把「第 `i+1` 个 tile 的 MTE2 搬入」与「第 `i` 个 tile 的向量链」重叠。
  `num_stages = 2` 的理由：单 tile 的 UB 需求 ×2 仍只占 19.4%（×1.12 → 21.8%），
  `num_stages = 3` 会到 29%（仍然可行，但收益只在 MTE2 延迟 ≫ 计算时出现，属 Stage 4 旋钮）。
- **`tpb = 1` 的 workload**（kimi-k2-t1 / ds-v3-t32 / glm45-t32）：循环退化为单次迭代，
  重叠由**多 block 并发**承担（逻辑核数 72–256 ≥ 56）。探针 P3 已用 `T.Pipelined(1)` 编译。
- **不使用 `T.Persistent` / `T.annotate_unlimit_memory`**：逻辑核数已 > 56 且每 block 工作量
  小（tile = 16×256 = 4096 元素、约 0.3–0.5 µs 级），persistent 化摊薄不到启动开销
  （与同批次 GEMM 设计的结论方向一致）；平台知识库亦无 `T.Persistent` 的正确性条目。

### 6.4 尾块处理

- **M/N 尾块**：由 `am = T.min(bm, M-m0)`、`an = T.min(bn, N-n0)` 的**动态区段 extent**
  表达（装载与写回都只覆盖有效区段），与源实现的逐元素守卫语义等价（§0.5 O4）。
- **死行/死 lane**：计算按静态 `bm`/`bn` 执行，死行/死 lane 的结果落在 UB 填充区、不进 GM。
- **无 K 方向**：本算子无收缩轴，不存在「K 尾块污染累加」类问题（对比同批次 GEMM 设计）。
- **本 workload 的实际尾块**：`M ∈ {1,32,512,4096}` 被 `bm ∈ {1,16}` 整除、
  `N ∈ {18432,2048,1536}` 被 `bn = 256` 整除 → **5 条 workload 全部无尾块**；
  尾块路径由 §5.4 的通用逻辑与探针 P3（`am=1`、`an=476`）保证可编译。

---

## 7. 同步策略

### 7.1 同步模式

**模式**：**自动同步**（`allow_autoschedule` 默认开启，`InsertSync` 在 MTE2 与 Vector 之间
插入 `asc_sync_*`；`MaterializeMultiBuffer` 落实版本）。不手写 `ascend_set_flag` /
`ascend_wait_flag` / `ascend_pipe_barrier`。

依据：`programming.md`「`ascend/pipeline.py` 的 `allow_autoschedule` 默认开启……当前用例使用
默认流水线；未验证手工跨核 flag 配对」。本算子是单核单阶段（无跨核、无 Cube↔Vector 协作），
自动同步足以覆盖全部依赖。

### 7.2 同步点说明

| 位置 | 同步方式 | 理由 |
|------|----------|------|
| `T.copy`（GM→UB）→ `T.simd.vld` | 自动（MTE2→V 依赖由 `InsertSync` 插入） | 向量指令必须等 UB 数据就绪 |
| `T.simd.vsts` → 下一个 tile 的 `T.copy` 覆盖同一 UB 版本 | 自动（V→MTE2 反依赖 + 版本轮换） | 双缓冲的写后读保护 |
| `T.simd.vsts` → `T.copy`（UB→GM） | 自动 | 写回前必须等向量结果落 UB |
| 跨 block | **无同步** | 各 block 完全独立（无跨核归约/无共享输出） |

### 7.3 pass_configs 配置

```python
# 保持默认：不关闭自动调度、不改共享内存复用策略
# （与源实现的差异：去掉 pass_configs={TL_ENABLE_FAST_MATH: True} 与 nvcc compile_flags）
```

---

## 8. 验证方案

### 8.1 Golden 函数

> 以 **§0.1 源算子语义**为唯一依据（fp32 加宽 → fp32 计算 → 单次舍入），**不复刻** §0.6 的
> NPU 算法（不出现 `vdiv/vneg/UNPK_B16/PK_B32` 之类的实现痕迹）。与源仓
> `tileops/workloads/moe.py:gated_activation` 逐项对应。

```python
import torch
import torch.nn.functional as F


def golden_silu_mul(gate_up: torch.Tensor, n: int) -> torch.Tensor:
    """`out = silu(gate_up[:, :n]) * gate_up[:, n:]`，与源 kernel 的 dtype 语义一致。

    源 `_silu_mul_fused_kernel._main`：g/u 精确加宽到 fp32 → fp32 内 sigmoid 与两次乘 →
    末尾一次舍入到输入 dtype。这里用 fp32 上取整实现同一语义（已核对：与 torch 原生 dtype
    路径的最大差异为 1 ulp，见 verify_equiv.py 的 golden-form 输出）。
    """
    gate = gate_up[:, :n].float()
    up = gate_up[:, n:].float()
    out = torch.nn.functional.silu(gate) * up          # fp32 内 silu（x·σ(x)）
    return out.to(gate_up.dtype)                        # 唯一一次舍入


# 与仓库参考实现的等价形式（源仓 examples/TileOPs/tileops/workloads/moe.py 的 gated_activation）：
#   gate, up = gate_up.chunk(2, dim=-1); return F.silu(gate) * up
```

**独立性**：golden 只引用 `gate_up` 的**列切分语义**与 `silu` 定义，不引用本设计的 tile 形状、
SIMD 指令或 `(g·u)/(1+e^{-g})` 的结合顺序——因此 A1 的等价性由 `verify_equiv.py`（机器验证）
与 golden 共同交叉约束，而不是自证。**gate/up 顺序**由 `[:, :n]` / `[:, n:]` 显式固定
（对应源仓 `test_shared_expert_mlp_keeps_the_gate_up_row_order` 的负向对照）。

### 8.2 精度标准

| dtype | atol | rtol | 依据 |
|-------|------|------|------|
| float16 | 1e-2 | 1e-2 | 源仓 `moe_verification(1)`（顶点级） |
| bfloat16 | 1e-2 | 1e-2 | 同上 |
| （附加）归一化平方误差 | — | ≤ 1e-4 | 源仓 `moe_close` → `_assert_normalized_error`（op 级测试口径） |

本 kernel 的中间精度是 fp32（§0.1），故 `atol/rtol = 1e-2` 有 ~10× 以上裕度
（§1.6.1 实测：输出 dtype 上与基线 0 ulp，与 fp64 参照 0 违反）。

### 8.3 L0 门槛测试计划

L0 = 最小集，覆盖「主路径 + M 尾块 + 两种 dtype」（Stage 3 实现，分层套件随后）：

| # | 用例 | 覆盖点 | 判据 |
|---|------|--------|------|
| L0-1 | `(M, N) = (32, 2048)`，bfloat16 | 主路径（`bm=1`、`grid=256`、`tpb=1`）、小 shape | `assert_close(golden, atol=rtol=1e-2)` + 归一化误差 ≤ 1e-4 |
| L0-2 | `(M, N) = (32, 2048)`，float16 | dtype 路径（`vcvt` 收窄 RNE）、ds-v3-t32 的第二 dtype 用例 | 同上 |
| L0-3 | `(M, N) = (1, 18432)`，bfloat16 | `M = 1` 的 `am` 边界（`mt=1`、`grid=72`、`tpb=1`）、最大 N | 同上 |
| L0-4 | `(M, N) = (64, 2048)`，bfloat16 | `bm=16` 分支 + `tpb=8` 流水（ds-v3-t4096 的几何缩影，128 个元素/块级验证） | 同上 |

> L1（大 shape：`(512,18432)` / `(4096,2048)`）、L2（最大 shape / 压力）、Boundary
> （`N % 64 != 0`、`N < 64`、非有限值、`M % bm != 0`）由 `tilelang-op-develop` 的分层套件
> 承担，不在此枚举。L0-4 的 `M = 64` 是**非 manifest 形状**，仅用于在 L0 阶段触达 `bm=16`
> 与 `tpb = 8` 分支（manifest 的对应 shape 太大，不适合放 L0）。

---

## 9. 风险点与注意事项

### 9.1 已知约束

| # | 约束 / 风险 | 影响 | 缓解 |
|---|---|---|---|
| 1 | `T.simd.vcvt` 窄化的「偶数 lane」语义：必须 `part=0` + `dist="PK_B32"` + 32-bit mask | 遗漏则**静默漏写一半输出槽位**（编译通过、结果错） | §3.2/§3.5.1 已固定写法；探针生成的 CCE 为 `vsts_pack_b32(vcvt<half>(...))`；L0 会在两 dtype 上暴露 |
| 2 | `vdiv` 生成的变体是 `vdiv_0ulp_ftz_true`（FTZ：次正规结果刷 0） | `num` 为次正规（|y| < 1.18e-38）时结果为 0；而 16-bit 输出的最小次正规是 6e-8（fp16）/ 9.2e-41（bf16） | 对 fp16 完全无影响；bf16 的次正规区间（<1.18e-38）在 `verify_equiv.py` 的 corner/exp-overflow 用例中与基线 0 ulp 一致；列入 Stage 3 Boundary |
| 3 | C4（16-bit 算术链）不得被 Stage 3/4「顺手优化」 | 会改变 §0.1 的 dtype 语义（3 次额外舍入） | §1.6.0 R4 已淘汰并写明理由；`verify_equiv.py` 可复跑复核 |
| 4 | UB 膨胀系数 1.10–1.12 是 910B2C 实测（`CONST-capacity-910B2C`） | 本型号可能更悲观/乐观 | 用量 19.4% → 即使按最悲观的 1.7× 也只有 33%，仍有 3× 余量 |
| 5 | 常量表全部绑定 910B2C；`cost_measurements` 为空 | §1.6.0 的时延数字只是**外推**，不是承诺 | 已标注「未实测假设 + 估算依据」；Stage 4 的 baseline 采集后按 `hardware-cost-model.md` §3 做偏差对账 |
| 6 | 设计期无法实查本型号核数（卡被 Stage 3 占用，`ASCEND_RT_VISIBLE_DEVICES=""`） | 若 `vector_core_cnt ≠ 56`，§5.5 的 `nsp/tpb` 利用率数字失效 | 引用平台落库记录；Stage 3 首跑前用 §5.5 的 `torch.npu.get_device_properties` 复核，若不同则重算 `CORES` 常数（只影响 `nsp` 选择，不影响正确性） |
| 7 | `T.Pipelined` + `SimdVF` 的重叠是否在生成代码中真正生效 | 若未重叠，时延 ≈ 流量项 + 计算链串行（最坏 +14 µs 级） | Stage 4 的 A/B（`num_stages=1` vs `2`）；不影响正确性 |
| 8 | 每 AIV 的 block 驻留数无文档（**未文档化假设**） | 若只能驻留 1 个 block，`tpb = 1` 的三个 workload 无任何 MTE2/计算重叠 | 这三个 workload 是发射/带宽混合主导且流量 ≤ 0.4 MB；§1.6.4 的备选不改变该结论 |
| 9 | wrapper 传入 `block_m = 128` 若不夹取会 UB 溢出（393 KB > 248 KB） | 编译失败 | §5.2 规则 1/2/5 的构建期夹取 + 断言；Stage 5 接线无需改 wrapper（夹取在 kernel 内） |

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|---|---|---|---|
| E1 UB 溢出 | 直接把 wrapper 的 `(128, 256)` 当 tile 用；或 `bm` 未随 `bn` 缩放 | 编译失败（`ub overflow`） | §5.2 规则 1/2/4 的夹取与 assert |
| E2 窄化存储漏写 | `vsts` 用默认 `dist="NORM_B32"` 或 16-bit 掩码 | 结果一半错、一半对（隐蔽） | §3.2 的 `PK_B32` + `pset(32)`；L0 双 dtype |
| E3 `N < 64` | `bn` 夹到 64 后 `an < 64`（本 op 不出现） | chunk 内大量死 lane（正确但低效） | 可接受；如需极致效率，Stage 4 按 shape 特化 `bn = N` |
| E4 SIMD 帧内读 GM | 把 `T.copy` 写进 `with T.SimdVF()` | 编译/校验失败（SIMD_VF 不得读 GM） | 所有 `T.copy` 在帧外 |
| E5 用 `T.clear` 清 4 KB 以上 fp32 UB | 想清填充区 | 降级为标量循环，拖慢整体 | 不清（死 lane 不写回） |
| E6 gate/up 半区取反 | 误写 `gate_up[:, N:]` 当 gate | 结果错但数值范围看似合理 | §8.1 golden 显式 `[:, :n]` / `[:, n:]`；L0 复用源仓的顺序负向对照思路 |
| E7 依赖 `FAST_MATH` | 以为 `T.exp` 会走快速近似 | 无影响（本后端无该开关） | §4.7 已去除 |

### 9.3 特殊场景处理

- **算子级融合（C5）**：把 `silu_and_mul` 融进 gate/up GEMM 的 epilogue 可再省 ~80% 的子链
  流量（§1.6.0 R3），但需要 `_gemm_basic_kernel` 的 epilogue 改造 + Cube/Vector 协作
  （`PL-950-mixed-epilogue` 只验证过单 tile ReLU）。**本设计不做**，作为 Stage 4/5 的
  算子级优化项记录；`splitk_reduce_kernel` 的 `silu_and_mul` epilogue 已在 decode 段覆盖了
  该形态的一部分。
- **SIMT 降级路径**：若 Stage 3 发现 SIMD 链在某 dtype/编译组合上不可用，可换
  `T.SimtVF(threads=…)` + 逐元素表达式（`kernel_template.py` 的已验证路径），几何与分核
  决策不变（每 lane 仍有 `T.exp`/`T.cast`）。
- **小型 shape**：`M ≤ 32` 的 3 条 workload 是「launch + 指令链」主导；若 Stage 4 测量显示
  启动开销占 > 50%，可考虑把这些 shape 的 `bn` 降到 128 以翻倍逻辑核数（不影响正确性）。

---

## 10. 交付清单

### 10.1 目录结构

```
examples/shared_expert_mlp/_silu_mul_fused_kernel/
├── DESIGN.md                                  # 本设计文档
├── verify_equiv.py                            # D-1 等价性机器验证（A1）
├── RETROSPECTIVE.md                           # Stage 1 复盘（自进化钩子）
├── history_version/
│   └── design_probe_compile.py                # D-3 无卡编译探针（P1–P4）
└── _silu_mul_fused_kernel.py                  # 待实现（Stage 3）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | ✅ 已完成 | 设计文档（§0–§10） |
| `verify_equiv.py` | ✅ 已执行 | `EQUIV_RESULT: ALL_PASS`（8 用例，bf16/fp16 × 4） |
| `history_version/design_probe_compile.py` | ✅ 已执行 | `PROBE_RESULT: ALL_OK`（4 例无卡编译） |
| `_silu_mul_fused_kernel.py` | ⬜ 待实现 | kernel + golden + L0/L1/L2/Boundary（Stage 3） |

### 10.3 命名规范

- 项目目录：`examples/shared_expert_mlp/`（同 op 的三个抽取函数各一子目录）
- 算子目录：`_silu_mul_fused_kernel`（与源函数同名，保持 harness 的 `kernel_map` 可追溯）
- 实现文件：`_silu_mul_fused_kernel.py`

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md）+ 算法调研 + 等价性验证 + 编译探针
2. ⬜ Golden 函数（§8.1，先写并自证：与 `F.silu(gate)*up` 原生路径 ≤ 1 ulp）
3. ⬜ 算子实现（`_silu_mul_fused_kernel.py`）+ L0 四例
4. ⬜ L1/L2/Boundary（Stage 3 分层套件）

### 10.5 交给后续阶段的硬约束

1. tile 取值必须走 §5.2 的夹取规则（**不得**直接用 wrapper 的 128×256，否则 UB 溢出）。
2. 窄化存储必须是 `vcvt(part=0) + vsts(dist="PK_B32") + pset(32)`（§3.2 / §9.2 E2）。
3. `T.copy` 不得进入 `T.SimdVF()` 帧。
4. Stage 4 的 A/B 清单：`num_stages ∈ {1,2,3}`、`bn ∈ {256,512}`、§1.6.4 的备选 #2（按 shape）。
5. Stage 5 接线：wrapper `_silu_mul(gate_up)` 的调用签名 `(M, N, dtype)(block_m, block_n)`
   已与本次设计一致（K9：去掉 `threads`），无需改动 wrapper。
