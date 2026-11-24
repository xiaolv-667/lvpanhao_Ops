# `_engram_gate_conv_fwd_kernel` 算子设计文档（Engram GateConv 前向，post-projection 融合）

| 项 | 值 |
|---|---|
| 任务目录 | `examples/engram_gate_conv/_engram_gate_conv_fwd_kernel` |
| 算子 | `_engram_gate_conv_fwd_kernel`（TileOPs `EngramGateConvFwdOp`，`sequence_modeling` family） |
| 迁移类型 | migration / harness（GPU TileLang 源算子 → ascend 950PR） |
| 平台 | `ascend-950pr-9579`（dav-3510，28 AI Core / 56 Vector Core，UB 253952 B、L1 524288 B、L0C 262144 B） |
| 工具链 | `ascend`：TileLang 0.1.15 + CANN 9.2.1 SDK/runtime，GCC 12.3.0（`statectl platform --task-dir` 判定） |
| 编程模式 | SIMD（平台 `programming_modes.migration_default = simd`） |
| 版本 | v1（first_design） |
| 设计期证据 | `verify_equiv.py`（等价性机器验证，实跑）、`history_version/design_probe_compile.py`（编译探针，6 个几何 ALL_OK） |

**本次设计的三个关键决定（详见 §1.6.0 / §1.6.3）**

1. **单次 launch**：把源算子「gate 段 + conv 段」两次 kernel 融合为一次 AIV launch——原始 vhat 不再经 GM 往返；conv 需要的 3 个前驱行由本核**重算 halo**（确定性同式，§0.6 R3），前驱行数据用 UB 环形窗口（4 个 slot）承接。备选（两段 launch，每行一核）随文档冻结，交 Stage 4 实测裁决（§1.6.3 三件套）。
2. **gate 点积与归一化融合**：`dot = rrms_h·rrms_k·Σ_d h_d k_d w_d²`，省掉 `h_norm/k_norm` 两个中间向量（每行省 3d 次乘 + 2d 个 fp32 驻留），机器验证 0.0018 以内容差预算占用（§1.6.1 A1）。
3. **`rms_w_v` 折叠进 depthwise conv 权重**（`cwv = conv_w·w_v`，每核一次），conv 段每 tap 每 chunk 省一次乘（§1.6.1 A2 + §1.6.3）。

---

## 0. 源算子解读与迁移分析

> 本章按「三问框架 + 耦合性判定 + 重设计」组织（方法论：`.agents/skills/tilelang-op-design/references/migration-analysis.md`）。
> **源码阅读范围**：GPU 源算子 `/workspace/TileOPs/src/tileops/_kernels/sequence_modeling/engram/gate_conv_fwd.py`（全文 176 行，含
> `@tilelang.jit` builder、`@T.macro`/`@T.prim_func` 主体与 host 侧 `forward`）、其调用契约
> `/workspace/TileOPs/src/tileops/_kernels/sequence_modeling/engram/call_spec.py`、以及本仓抽取件
> `examples/TileOPs/tileops/kernels/sequence_modeling/engram_gate_conv/_engram_gate_conv_fwd_kernels.py`（与源逐行一致）。
> **补充只读参考**（用于 §0.4/§0.5 的来源标注，不作为设计结论直接照抄）：950DT 上已验证的 Ascend 实现
> `/workspace/TileOPs-ascend950/src/tileops/kernels/ascend/sequence_modeling/gate_conv_fwd.py`（232 行，两次 AIV launch）。

### 0.1 源算子语义（做什么）

post-projection 的 Engram 门控 + 因果深度卷积融合，逐 `(m, t)` 行（行 = 位置，维 = 隐藏维 `d`）：

```
h_norm = RMSNorm(H[m,t,:], w_h)          rrms_h = (mean_d(H²) + eps)^(-1/2)
k_norm = RMSNorm(k[m,t,:], w_h)          rrms_k = (mean_d(k²) + eps)^(-1/2)      ← 用同一个 w_h
dot    = Σ_d h_norm·k_norm
alpha  = sigmoid(dot / sqrt(d))
v_hat  = alpha · v[m,t,:]                （fp32）
vhat   = fl16(v_hat)                     ← 输出张量之一，先舍入到输入 dtype
vhat_norm = RMSNorm(vhat, w_v)           rrms_v = (mean_d(vhat²) + eps)^(-1/2)    ← 统计量取**已舍入**的 vhat
conv_out[t] = Σ_{p=0..3} conv_w[p] ⊙ vhat_norm[t-3+p]   （depthwise，左侧补 3 个零位置，dilation=1）
Y      = fl16( SiLU(conv_out) + v_hat )  ← 残差用**未舍入**的 fp32 v_hat
```

**语义要素逐项结论**

| 要素 | 结论（源码依据） | 实现位置 |
|---|---|---|
| 规约轴 / 累加顺序 | 4 个行内归约：`Σ H²`、`Σ k²`、`Σ h_norm·k_norm`、`Σ vhat²`，均为**末轴 d**、`keepdim=False` 输出 `(M,T)`；源码用 fragment + `T.reduce_sum(dim=1)`（fp32），累加顺序为**分组树**，不与 torch 逐元素顺序一致 | 源 `_gate_fwd` 的 4 次 `T.reduce_sum` |
| 规约分母 | `mean_d` 用 **d**（不是 padding 宽度）；抽取件 `d_padded = align_up(d, 256)` 但分母仍是 `float(d)` | 源 `rrms_h_val = T.rsqrt(sumsq_h[0] / float(d) + eps)` |
| dtype 语义 | 输入/权重/conv 权重 16-bit；**全部中间量 fp32**（`accum_dtype = "float"`）；`vhat` 与 `Y` 输出 16-bit；`alpha/rrms_*` 输出 fp32 | 源 `accum_dtype`、`out_idx=[6..11]`、`call_spec.EngramGateConvFwdInterface.forward` docstring |
| 舍入点（逐点） | ①`vhat = fl16(alpha·v)`；②`rrms_v` 的统计量取①之后的值（`vhat_buf` 读回）；③`conv_out` 的输入 `vhat_norm` 由②的 vhat 在 fp32 相乘得到；④`conv_w`/`w_v` 保持 16-bit 存储、fp32 参与乘；⑤`Y` 的残差项取**未舍入** `alpha·v`（fp32）；⑥`Y` 最后一次 fl16；⑦`alpha/rrms_*` 以 fp32 写出 | torch 参考 `engram_gate_conv_fwd_torch`（②⑤ 的判据）、源 `_gate_fwd`/conv 段 |
| 边界语义 | 因果左补 3 位置（`t-3+p < 0` 的 tap 贡献为 0）；`d` 任意正整数（源码 host 侧 `F.pad` 到 256 倍数并裁回）；`M·seq_len` 任意；无非有限值特殊处理（inf/NaN 自然传播） | 源 `T.if_then_else(src_t >= 0, …, 0.0)`、host `F.pad` + `results[0][:, :, :d]` |
| host 侧语义 | `F.pad` 把 H/k/v/`w_h`/`w_v`/`conv_w` 六张量右侧补到 `d_padded`，输出 2 张量裁回 `d`；这是**实现手段**（padding 契约），非语义 | 源 `EngramGateConvFwdKernel.forward` |
| dilation | **本算子无 dilation**：`EngramGateConvCall` 无该字段，`CONV_KERNEL_SIZE = 4`，torch 参考为 `F.pad(k-1,0)` + `conv1d(groups=d)`（dilation 默认 1）。`dilation` 只出现在同 family 的 `EngramDecodeCall`（另一算子），**不适用于本算子** | `call_spec.py`；`examples/TileOPs/tileops/workloads/sequence_modeling.py` |

**语义保持基线**：§8.1 的 golden 以本节为唯一依据（直接移植 torch 参考），不引用 §0.6 的任何 NPU 结构。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|---|---|---|---|
| `H` | `(M, seq_len, d)` | `float16 \| bfloat16` | hidden states；`M ≤ 3` 为动态轴（manifest 用 1/2/3） |
| `k` | `(M, seq_len, d)` | 同 `H` | key projection |
| `v` | `(M, seq_len, d)` | 同 `H` | value projection |
| `rms_w_h` | `(d,)` | 同 `H` | **H 与 k 共用**的 RMSNorm 权重 |
| `rms_w_v` | `(d,)` | 同 `H` | vhat 的 RMSNorm 权重 |
| `conv_w` | `(4, d)` | 同 `H` | depthwise 因果 conv 权重（4 tap） |
| `Y` | `(M, seq_len, d)` | 同 `H` | 输出：`SiLU(conv_out) + v_hat`（out_idx 0） |
| `vhat` | `(M, seq_len, d)` | 同 `H` | 输出：舍入到输入 dtype 的 `alpha·v`（out_idx 1，backward 保存） |
| `alpha` | `(M, seq_len)` | **float32** | 门控标量 |
| `rrms_h` | `(M, seq_len)` | **float32** | `H` 的倒数 RMS |
| `rrms_k` | `(M, seq_len)` | **float32** | `k` 的倒数 RMS |
| `rrms_v` | `(M, seq_len)` | **float32** | `vhat` 的倒数 RMS |
| `eps` | 标量（params，默认 `1e-6`） | — | 三个 RMSNorm 共用 |

### 0.3 实现算法解读（怎么算）

**计算步骤分解**（源 GPU kernel 每个计算语句归入下列步骤，无遗漏）

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式 |
|---|---|---|---|---|
| S1 | 3 行搬入 + fp32 提升：`h_2d/k_2d/v_1d = cast(H/k/v[b,t,:])` | GM 三行（`d_padded`） | fragment（fp32，`(1,d_padded)`/`(d_padded,)`） | 输入读取 |
| S2 | `hsq = h²`，`reduce_sum(dim=1)` → `sumsq_h`；`ksq/ksq` 同 | fragment | 2 个 fp32 标量 | `mean_d(H²)`、`mean_d(k²)` |
| S3 | `rrms_h = rsqrt(sumsq_h/d + eps)`、`rrms_k` 同；写 `rrms_h_buf/rrms_k_buf` | 标量 | GM `(M,T)` fp32 ×2 | `rrms_h`、`rrms_k` |
| S4 | 归一化：`h_2d *= rrms_h·w_h`、`k_2d *= rrms_k·w_h` | fragment + `rms_w_h` | fragment（覆盖写） | `h_norm`、`k_norm` |
| S5 | `hk_prod = h_2d·k_2d`，`reduce_sum(dim=1)` → `dot_hk` | fragment | fp32 标量 | `Σ h_norm·k_norm` |
| S6 | `alpha = 1/(1+exp(-(dot_hk/sqrt(d))))`；写 `alpha_buf` | 标量 | GM `(M,T)` fp32 | `alpha` |
| S7 | `vhat = alpha·v`；写 `vhat_buf`（**dtype 舍入**）；同时 `vhsq = vhat²`、`reduce_sum` → `rrms_v`；写 `rrms_v_buf` | fragment | GM `vhat`（dtype）+ GM 标量 | `vhat`、`rrms_v` |
| S8 | kernel 2：读回 `vhat_buf[b,t,:]` → `vhat_cur`（fp32） | GM | fragment | vhat 复用 |
| S9 | 4 tap 循环（`T.serial(4)`）：`src_t = t-3+p`，`if_then_else(src_t>=0, vhat_buf[b,src_t,:], 0)`、`src_rrms = if_then_else(..., rrms_v_buf[b,src_t], 0)`，`normed = raw·src_rrms·w_v`，`conv_out += conv_w[p]·normed` | GM 4 行 vhat + 4 个标量 + `w_v` + `conv_w` | fragment `conv_out`（fp32） | `conv_out`（因果左补） |
| S10 | `sig = 1/(1+exp(-conv_out))`；`Y = conv_out·sig + vhat_cur` → 舍入写 GM | fragment | GM `Y`（dtype） | `SiLU` + 残差 |

**数据流与内存访问模式（源硬件视角）**

```
GM H,k,v ──T.copy──► fragment(fp32, 1×d_padded) ──平方/乘/reduce_sum──► fp32 标量
                                   │
                                   ├─► GM rrms_h/rrms_k/alpha  (每行 3 次 4B 标量写)
                                   └─► GM vhat_buf (dtype，行写)  ─┐
GM rrms_v ◄── reduce ────────────────────────────────────────────┤
                                                                  ▼
kernel 2: GM vhat_buf ×4 行 + GM rrms_v ×4 + GM v + GM w_v/conv_w ──► fragment ──► GM Y
```
无 L1/L0 参与（纯 Vector/SIMT 路径，GPU 侧为 register fragment + warp 归约）。中间量 `vhat_buf` **既是被保存的输出又是 kernel 间的中转**（`out_idx` 里同时出现）。

**循环与并行结构**：grid = `(seq_len, M)`（二维 block，每 `(m,t)` 一个 block）；block 内 `threads ∈ {128,256,512}`（默认 128）自动调优；行内 `T.Parallel(d_padded)` 由 128 个线程铺满；归约用 `T.reduce_sum(dim=1)`（block 内多级合并）；conv 段 4 个 tap 是 `T.serial`，tap 内 `T.Parallel(d_padded)`。

**host 侧逻辑**：`EngramGateConvFwdKernel.forward` 中 (a) `F.pad` 六张量到 `d_padded = align_up(d, 256)`；(b) `out_idx` 分配 6 个输出；(c) 把 `Y/vhat` 裁回 `[:, :, :d]`。语义上无预处理（pad 是空间契约）。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|---|---|---|---|---|
| O1 | 行内驻留融合（单 block 读一次行，4 个归约复用同一 fragment） | 消 GM 重复读 | 行数据驻留 register fragment，平方/乘积/归约共享 | 大寄存器堆 | 硬件强相关（实现层） |
| O2 | `T.reduce_sum(dim=1)` 二维 fragment 归约 | 低延迟跨线程归约 | block 内 warp shuffle + SMEM 多级合并 | warp（32 线程） | 硬件强相关 |
| O3 | 3 个统计量 + vhat 在同一 kernel 内完成（gate 段单遍） | 省中间张量 | 顺序扫描 3 个输入行一次 | — | 算法层，可移植 |
| O4 | 中间 `vhat` 落 GM 供第二 kernel 读（**跨 kernel 物化**） | 解跨 block 数据依赖（conv 需要邻居行） | 第二 kernel 从 GM 取 4 个邻居行 | 无（这是 GPU 每 `(m,t)` 一 block 的必然） | 实现层，可重设计 |
| O5 | 因果左补用 `T.if_then_else(src_t>=0, …)` 逐元素选择 | 免分支越界 | 逐元素谓词选择 | — | 实现层，可替换 |
| O6 | `d_padded = align_up(d, 256)` + host `F.pad`/裁回 | 让行宽整除、归约安全 | host 侧补零与裁剪 | — | 实现层，可替换 |
| O7 | `fp32` 全中间量（`accum_dtype="float"`），fragment 存 fp32 | 精度与稳定性 | 16-bit 载入即提升 | 大寄存器堆 | 语义层 + 实现层 |
| O8 | conv 段把 `w_v` 与 `rrms_v` 逐 tap 重新乘上 | 复用同一 tap 公式 | 每 tap 两次乘 | — | 实现层（有 CSE 空间） |
| O9 | 6 输出同一 prim_func（`out_idx=[6..11]`） | 单次 launch 出 6 结果 | jit 多输出分配 | — | 可移植 |
| O10 | `@target_cache(maxsize=32)` 按 `(M,seq_len,d,eps,dtype)` 特化 | 编译期常量折叠 | 每个 shape 一份 kernel | — | 可移植（本设计用 `lru_cache` 同策略） |
| O11 | `threads` 自动调优（128/256/512） | 匹配 block 内并行度 | 线程数扫描 | GPU 线程模型 | **不可移植**（本平台无 thread 域） |
| O12 | `compile_flags=["-O3","-DENABLE_BF16"]`、GPU 侧 `TL_ENABLE_FAST_MATH` | 编译优化 | 后端开关 | CUDA 后端 | 不可移植（ascend target 默认后端不同） |

**950DT 参考实现（只读补充，来源标注）**的优化手段，本设计是否承接见 §0.5/§0.6：

| # | 手段（来源：`/workspace/TileOPs-ascend950/.../gate_conv_fwd.py`） | 说明 |
|---|---|---|
| A-1 | 两次 AIV launch（gate 段、conv 段），一维 `T.Kernel(M*seq_len)`，一行一核 | 用 1D grid 替换 GPU 二维 grid |
| A-2 | `dp = align_up(d, 64)`：64 元素 chunk + `update_mask(min(64, d-chunk*64))` 尾 lane 屏蔽 | 无需 host padding |
| A-3 | `vld(dist="UNPK_B16")` + `vcvt(part=0)` 提升；`vcvt(..., round="ROUND_R")` + `vsts(dist="PK_B32")` 收窄 | 16-bit ↔ fp32 的每 chunk 链 |
| A-4 | vhat 的 dtype 舍入往返（`narrowed → vcvt→fp32`）后再做 `rrms_v` 平方和 | 与 torch 参考的 ② 舍入点一致 |
| A-5 | `stats(4,8)` fp32 暂存 + `vsts(dist="ONEPT_B32", extent=1)` + 压缩 DMA 批量写 4 个标量 | 标量输出批量写 |
| A-6 | conv 段每 tap 从 GM 读 `vhat[src]` + `rrms_v[src]`，逐 tap 乘 `w_v` | 与源 O4/O8 同构 |
| A-7 | `_MAX_HIDDEN_DIM = 4096` refusal（`d` 上界） | 与本平台 UB 容量匹配 |

### 0.5 硬件耦合性分析与 NPU 适配决策

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|---|---|---|---|---|---|
| 语义（S1–S10 的数学内容、7 个舍入点、因果左补、eps/分母口径） | 语义 | 无 | — | **保留** | 无条件保留；§0.6 每项的语义保持论证均以此为准 |
| `T.Kernel(seq_len, M, threads=…)` 二维 grid + thread 域（O11） | 优化 | GPU block/thread 两级并行 | 部分：`T.Kernel` 只支持**一维** grid 且无 `threads` | **重新设计** | R1：一维 `T.Kernel(grid)` + `T.SimdVF` 向量域（`references/programming.md`「`T.Kernel(N)` 一维核网格；多维网格在实现中拒绝」） |
| 行内 `T.reduce_sum(dim=1)`（O2） | 优化 | warp shuffle + SMEM 合并 | 有：SIMD `vcadd`（64 lane 成对树） | **等价替换** | `T.simd.vcadd(acc, full)`；高层 shared→shared 归约有已知输出布局差异（`TRAP-950-simd-shared-reduce`），故不用 |
| register fragment 驻留（O1） | 优化 | 大寄存器堆 | 有：向量寄存器（fp32 64 lane / 16-bit 128 lane）+ UB | **等价替换** | 行数据驻留 UB（`h_ub/k_ub/v_ub`）+ 每 chunk `vld` 进寄存器，乘积/累加不落 UB（`simd.alloc_local` 累加器） |
| `vhat` 跨 kernel 落 GM（O4） | 算法/优化 | 每 `(m,t)` 一 block 使邻居行不可见 | 无等价（同 launch 内无跨 block barrier：`ascend.batch_schedule` 跨 block barrier 未认证） | **重新设计** | R3：单 launch + 每核 RC 行连续所有权 + 3 行 halo 重算 + UB 环形窗口；跨 block 依赖被「同核内顺序」替代 |
| 逐元素 `if_then_else(src_t>=0)`（O5） | 优化 | — | 有：标量分支 / 掩码 | **等价替换** | R4：因果零边界改为 (a) 3 个 halo slot 预先向量清零 + (b) halo 计算用**块级**标量 `if r >= 0`（模板 `if bx < full:` 同形）；不再是逐元素谓词（探针 P1/P2 已验证生成 `if (1 <= …)` 与 `if (3 <= i_1)`） |
| `d_padded = align_up(d, 256)` + host `F.pad`/裁回（O6） | 优化 | — | 有：尾 lane 屏蔽 | **等价替换** | R2：`DW = align_up(d, 64)`，行内 chunk = 64 元素，`update_mask(min(64, d-chunk*64), 32)` 屏蔽尾 lane；**host 不再 padding、输出不再裁剪**（6 个张量免拷贝） |
| 4 个独立 `T.reduce_sum` + 中间 fragment（O3/§0.3 S2/S5/S7） | 算法 | — | 有 | **等价替换 + 增强** | R5：同一 chunk 循环内一次读取算 3 个统计量（`Σh²`、`Σk²`、`Σ h·k·w²`），再加第 4 次扫描算 `Σvhat²`；`h_norm/k_norm` 中间向量被 §1.6.1 A1 消掉 |
| 逐 tap 重乘 `w_v`/`rrms_v`（O8 / A-6） | 优化 | — | 有 | **等价替换** | R5：`w_v` 折叠进 `conv_w`（A2）；`rrms_v` 与 vhat 的乘积（即 conv 输入行）**每行只算一次**并驻留环形窗口（A3） |
| `vhat` dtype 舍入往返（A-4） | 语义 | — | 有：`vcvt` 双向 | **保留** | 与 golden 的舍入点 ② 一致（`vcvt(..., dtype, round="ROUND_R")` → `vcvt(..., "float32")`） |
| `stats(4,8)` + `ONEPT_B32` 标量批量写（A-5） | 优化 | — | 有 | **保留（扩展）** | 每核一个 `(4, RC+3, 8)` fp32 暂存，tile 末 4 条压缩 DMA 写 4 个 `(M,T)` 输出（探针生成的 `asc_copy_ub2gm_align(…, 6, 4, …, 4, 32)` 即此形态） |
| `threads` 自动调优 + `compile_flags`/fast-math（O11/O12） | 优化 | GPU 线程模型 / CUDA 后端 | 无 | **舍弃** | 本平台并行来自「grid × 向量 lane」，无线程域可调；fast-math 会在 `vmul+vadd` 上做收缩、改变乘积舍入位置，使 §1.6.1 A1 的等价性论证失效（同 `examples/mla_dense_bwd/_delta_kernel/DESIGN.md` §7.3 的处置） |
| `@target_cache` 形状特化（O10） | 优化 | — | 有：`functools.lru_cache` | **保留** | 按 `(M, seq_len, d, eps, dtype)` 编译期特化（探针即以该形态编译 6 个几何） |

> 依据来源：`references/programming.md`（Kernel/SimdVF/分配/拷贝/归约/同步条款）、`knowledge/INDEX.md` 各条目、
> 950DT 参考实现（仅作「该算子族在 Ascend 上存在什么做法」的直接证据，其结构结论在本设计中被重新评估）。

### 0.6 NPU 算法重设计

**重设计项 R1：并行结构与 Kernel 结构（二维 grid + thread → 一维 grid + 向量 lane）**

- **源方案**：`T.Kernel(seq_len, M, threads=128)`，每 `(m,t)` 一 block，`T.Parallel(d_padded)` 铺 128 线程；性能意图是「一行一核、行内并行」。
- **NPU 新算法**：一维 `T.Kernel(grid)`，`grid = M·ceil(seq_len/RC) ≤ 56`；每核拥有一段**连续行** `[t0, t0+RC)`；行内并行由 `T.SimdVF` 的 64 lane fp32 向量承担，chunk 循环 `T.serial(nchunks)`；核内行循环 `T.serial(RC+3)`。
- **语义保持论证**：任务映射变化不改变任何数学表达式或行内归约轴（仍是 `d`）。累加分组由「128 线程树」变为「64 lane `vcadd` 树 + chunk 顺序累加」，属 fp32 内重分组，`verify_equiv.py` 的 `rrms_h/rrms_k` 与 baseline **0 ulp**（同表达式同分组），`alpha` 最大 9 ulp（fp32）；与既有平台约定一致（`PL-950-normalization` 的 fp32 统计量门限 1e-5）。边界：因果零位置与 `M<block` 情形由 R4/R3 处理，`grid ≥ 1` 恒成立。

**重设计项 R2：非整除处理（host pad + crop → 尾 lane 屏蔽）**

- **源方案**：host 侧把 6 张量 `F.pad` 到 `d_padded = align_up(d, 256)`，kernel 内按 `d_padded` 计算，输出再裁回。性能意图：让行宽整除、归约安全。
- **NPU 新算法**：`DW = align_up(d, 64)`（一个 fp32 向量寄存器的宽度），每 chunk 用 `T.simd.update_mask(min(64, d - chunk*64), 32)` 屏蔽无效 lane；归约分母仍是 `d`；chunk 数 `nchunks = DW/64`。host 侧 6 个张量**零拷贝**（不再 pad、不再 crop）。
- **语义保持论证**：被屏蔽 lane 在 `vmul/vadd` 中输出 0（`MODE_ZEROING`），不影响 `Σ`；分母保持 `d`（与源 `float(d)` 一致）。对 `d ∈ {256,512,768}`（全部 manifest workload）`d % 64 == 0`，屏蔽退化为全 lane，结果与「无屏蔽」逐位相同；`d % 64 != 0` 的几何由 L0-3（`d=196` → `update_mask(64/64/64/4)`）与 Boundary（`d=200`）覆盖。边界：`d=1` 时 `nchunks=1`、1 lane 有效，`update_mask(1)` 兜住。

**重设计项 R3：跨阶段融合（两次 launch + vhat 物化 → 单 launch + 环形窗口 + halo 重算）**

- **源方案**：kernel 1 产出 `vhat/alpha/rrms_*` 落 GM；kernel 2 逐输出行从 GM 读 4 个邻居行（`vhat` + `rrms_v`）+ 本行 `v` 做 conv。性能意图：把「跨 block 的邻居数据」用 GM 中转解决。
- **NPU 新算法**：单 launch。每核按行序流式处理 `RC+3` 行（前 3 行为 halo），环形窗口 `slot[4][DW]`（fp32）保存每个源行的 conv 输入行 `g_r = fl16(alpha_r·v_r)·rrms_v_r`：
  - 对行 `r`（`i = r - t0 + 3`）：算 gate 统计量 → `alpha_r`、`rrms_v_r` → 写 `vhat`（输出，仅 owned 行）→ `g_r` 写入 `slot[i % 4]`（**每行只算一次**）；
  - owned 行（`i ≥ 3`）：`conv_out = Σ_p cwv[p] ⊙ slot[(i-3+p) % 4]` → `SiLU` → `+ alpha_r·v_r`（未舍入残差）→ `Y`。
  - halo 行只算 gate 与 `slot`，不写任何 GM（其余核负责写）。
- **语义保持论证**：
  1. **数学等价**：每行的 `g_r`、`conv_out`、`Y` 表达式与源逐字相同（仅把「每 tap 重算 `g`」改为「每行算一次、4 次复用」，见 §1.6.1 A2/A3，机器验证 **1 ulp 以内**）。
  2. **halo 重算的确定性**：某核重算行 `r` 时使用的输入（`H/k/v[r]`、`w_h`、`w_v`、`eps`、`d`）与拥有该行的核完全相同，且执行**同一条构建期展开的指令序列**（同一份代码、同一 group），故 `g_r` 逐位相同；因此 conv 的输入与「先写 GM 再读回」一致。
  3. **边界**：`c == 0` 时 `t0 = 0`，halo 行 `r < 0` 由 `if r >= 0` 跳过，`slot[0..2]` 被**预先向量清零**，等价于源的「零位置」；序列首尾不跨 `m` 边界（每核只在本 `m` 内取 `t0`/halo），与源 `(m,t)` 独立语义一致。
  4. **累加顺序**：conv 的 4 tap 按 `p = 0..3`（与源 `T.serial(4)` 同序）；`conv_w` 与 `w_v` 的折叠（A2）改变乘积的舍入位置（`fl(cw·w_v)` vs `fl(fl(x·rrms)·w_v)·cw`），fp32 下机器验证 1 ulp。
  5. **代价**：每核 3 行 halo 的额外 gate 计算（占 owned 行的 `3/RC`）与 3 行的额外 GM 读；换来少一次 launch 与 `vhat` 的 GM 往返（§1.6.0 R3 已量化）。

**重设计项 R4：因果零边界的实现（逐元素谓词 → 块级分支 + 预清零）**

- **源方案**：`T.if_then_else(src_t >= 0, vhat_buf[...], 0.0)` 与 `if_then_else(..., rrms_v_buf[...], 0.0)`：每元素一次谓词选择。
- **NPU 新算法**：零边界在**块级**表达——3 个 halo slot 由向量 `vsts` 预清零；halo 是否计算由标量 `if r >= 0`（`t0, i` 的整数比较）决定；输出行是否计算由标量 `if i >= 3` 决定。
- **语义保持论证**：零位置贡献 `0`（与 `if_then_else(...,0)` 完全同值；`0` 参与 `vmul/vadd` 仍是 `0`，不引入 NaN 源——slot 是**显式**清零而非未初始化）。边界：`t0 ∈ {0..}` 全部由同一比较式覆盖，`RC=1` 时也成立（探针 P1 生成的 `if (1 <= ((block_idx & 15) * 2 + i_1 / 3))` 即该式经编译器化简）。

**重设计项 R5：行内多统计量单遍 + 归一化融合 + 权重折叠**

- **源方案**：4 次独立归约，每次物化一个 fragment（`hsq/ksq/hk_prod/vhsq`），并物化 `h_norm/k_norm` 两个归一化向量；conv 每 tap 重乘 `w_v·rrms_v`。
- **NPU 新算法**：(a) 一个 chunk 循环同时累积 `Σh²`、`Σk²`、`Σ h·k·w_h²`（3 个 `alloc_local` 向量累加器 + 3 次 `vcadd`）；(b) `alpha = sigmoid(Σ(h·k·w_h²)·rrms_h·rrms_k/sqrt(d))`，`h_norm/k_norm` 不再物化；(c) `cwv[p] = conv_w[p]·w_v` 每核算一次，conv 段直接用 `cwv`；(d) `g_r` 每行算一次（R3）。
- **语义保持论证**：(a)(b) 见 §1.6.1 A1（数学恒等，机器验证 `alpha` 9 ulp、容差预算 0.1%）；(c) 见 A2（机器验证 1 ulp）；(d) 见 A3（同一确定性表达式，逐位相同）。`h_norm/k_norm` 不物化不改变 `dot` 的数学值（分配律），只改变 fp32 舍入分组。

**重设计项 R6：Kernel 结构与资源（refusal 边界、UB 预算）**

- **源方案**：`d_padded = align_up(d, 256)` + `d <= 4096` refusal（因为整行驻留 fp32 fragment）；950DT 参考 `_MAX_HIDDEN_DIM = 4096`。
- **NPU 新算法**：行数据以 **chunk 流**处理（每 chunk 只驻留一个向量寄存器对），UB 常驻量 = `O(DW)`（6 个 dtype 行缓冲 + 3 个 fp32 行级缓冲 + 1 个 4×DW 窗口）；`d` 上界由 `4·DW·4 B + 7·DW·elem ≤ UB` 决定，`d ≤ 4096` 时 4×4096×4 = 64 KB + 7×4096×2 = 56 KB + cwv 4×4096×4 = 64 KB ≈ 184 KB < 248 KB ✅（仍保留 `d ≤ 4096` refusal 与源一致）。
- **语义保持论证**：分块方式不改变任何行的计算结果（chunk 之间无数据依赖，归约在寄存器累加器上按 chunk 顺序合并——与 §1.6.1 A1 的分组论证同源）。

### 0.7 标杆实现

| 用途 | 路径 | 说明 |
|---|---|---|
| 源算子（GPU） | `/workspace/TileOPs/src/tileops/_kernels/sequence_modeling/engram/gate_conv_fwd.py` | 迁移对象；三问解读来源 |
| 本仓抽取件 | `examples/TileOPs/tileops/kernels/sequence_modeling/engram_gate_conv/_engram_gate_conv_fwd_kernels.py` | 与源逐行一致，供 §0.3/§0.4 引用 |
| 调用契约 | `/workspace/TileOPs/src/tileops/_kernels/sequence_modeling/engram/call_spec.py` | 6 入 6 出、`CONV_KERNEL_SIZE=4`、无 dilation |
| **Golden 来源** | `examples/TileOPs/tileops/workloads/sequence_modeling.py` | `engram_gate_conv_fwd_torch` / `_rmsnorm`，§8.1 直接移植 |
| 正确性用例 | `examples/TileOPs/tests/ops/test_engram.py` | 6 输出逐个比对，fp16 `atol=rtol=0.1`、bf16 `atol=0.2/rtol=0.1` |
| 性能用例 | `examples/TileOPs/benchmarks/ops/bench_engram.py` | manifest 驱动，roofline 公式与 §1.6.0 交叉核对 |
| workload 清单 | `examples/TileOPs/tileops/manifest/sequence_modeling.yaml` | 4 条 workload（train-s32/s64/s128/s96） |
| 只读参考实现（950DT） | `/workspace/TileOPs-ascend950/src/tileops/kernels/ascend/sequence_modeling/gate_conv_fwd.py` | 两次 launch 的 Ascend 做法；本设计在其基础上改为单 launch（§0.5/§0.6 R3） |
| 迁移适配说明 | `examples/TileOPs/docs/gpu_to_npu_adaptation.md` | manifest 引用的适配记录 |

---

## 1. 概述

### 1.1 算子名称

`_engram_gate_conv_fwd_kernel`（TileOPs `EngramGateConvFwdOp`，Engram 序列建模的 post-projection 前向融合）。

### 1.2 功能描述

对 `(M, seq_len, d)` 的 `H/k/v` 三张量：各自做 RMSNorm，用归一化后的 `H·k` 行点积算门控 `alpha`，`vhat = alpha·v`，再把 `vhat` 归一化后做 **因果宽度 4 的 depthwise 卷积**、SiLU、加回未舍入残差得到 `Y`；同时输出 `vhat` 与 4 个 fp32 行标量（`alpha/rrms_h/rrms_k/rrms_v`）供反向复用。**一次 AIV launch 完成**（源算子为两次）。

### 1.3 数学公式

逐 `(m, t)` 行（`j = 1..d`），`eps = 1e-6`，`N = 4`（tap 数）：

$$
\begin{aligned}
r_h &= \big(\tfrac{1}{d}\textstyle\sum_{j} H_{j}^{2} + \varepsilon\big)^{-1/2},\quad
r_k = \big(\tfrac{1}{d}\textstyle\sum_{j} k_{j}^{2} + \varepsilon\big)^{-1/2} \\
\dot{} &= \textstyle\sum_{j} (H_j r_h w_{h,j})(k_j r_k w_{h,j})
= r_h r_k \textstyle\sum_j H_j k_j w_{h,j}^2 \\
\alpha &= \sigma\big(\dot{} / \sqrt{d}\big),\qquad
\widehat{v}_j = \text{fl16}(\alpha\, v_j),\qquad
r_v = \big(\tfrac{1}{d}\textstyle\sum_j \widehat{v}_j^{2} + \varepsilon\big)^{-1/2} \\
g_{t,j} &= \widehat{v}_{t,j}\, r_v^{(t)}\, w_{v,j},\qquad
c_{t,j} = \textstyle\sum_{p=0}^{N-1} cw_{p,j}\, g_{t-3+p,\,j}\quad(\text{pad}=0) \\
Y_{t,j} &= \text{fl16}\big(\mathrm{SiLU}(c_{t,j}) + \alpha_t v_{t,j}\big)
\end{aligned}
$$

输出：`Y, vhat`（dtype），`alpha, rrms_h, rrms_k, rrms_v`（fp32）。第 2 行是 §1.6.1 A1 采纳后的**等价化简形式**（`r_h r_k` 提到求和外），`g` 是 A2/A3 采纳后的形式（`w_v` 折进 `cwv`、每行只算一次）。
（`c` 的第 3 行里 `t-3+p < 0` 的 tap 取 0，即因果左补 3 个零位置、dilation = 1。）

### 1.4 算法描述

**迁移后的 NPU 算法（单 launch、行级流式、UB 环形窗口）**，与源算法（两次 kernel、逐行物化 vhat）的结构差异标 *（来源 §0.6 R3）*：

1. **任务映射**：`grid = M · ceil(seq_len / RC) ≤ 56`，每核独占同一 `m` 内**连续 RC 行** `[t0, t0+RC)`；`RC` 由构建期规则取（§5.2），对 4 条 workload 分别为 1/4/4/6。
2. **参数预置（每核一次）**：`w_h² → w2h_ub`（fp32，供 A1 的 `Σ H·k·w_h²` 直接使用）、`conv_w·w_v → cwv_ub`（4×DW fp32，A2）；3 个 halo slot 向量清零（R4）。*（源无此步骤；950DT 参考逐 tap 重乘 `w_v`）*
3. **行流水**：`for i in T.serial(RC + 3)`，源行 `r = t0 - 3 + i`：
   - `if r ≥ 0`（halo 存在）→ 搬入 `H/k/v[r]` 三行（**一次**，源为 3 次独立 fragment 搬入）；
   - 一个 chunk 循环（`nchunks = DW/64`）同时累积 `ΣH²`、`Σk²`、`Σ H·k·w_h²`（3 个 `alloc_local` 向量累加器）→ 3 次 `vcadd` → `r_h, r_k` → `alpha`（A4：`sqrt(d)` 折成标量乘）；
   - 第二个 chunk 循环：`vhat = fl16(alpha·v)`（写 `vh_ub` 并做 dtype 舍入往返）→ 累积 `Σv̂²` → `vcadd` → `r_v` → `g = v̂·r_v` 写入 `slot[i % 4]`（A3：每行一次，4 tap 复用）；
   - `if i ≥ 3`（owned 行）→ conv：4 tap × `nchunks` 次 `vld(slot)·vld(cwv)` 累加 → `SiLU` → `+ alpha·v`（未舍入残差）→ 收窄写 `y_ub`；随后两条单行 DMA 写 `vhat/Y`。*（源在第二 kernel、从 GM 读 4 邻居行、逐 tap 重乘 `w_v`/`rrms_v`）*
4. **标量输出**：`(4, RC+3, 8)` fp32 暂存按行 `vsts(extent=1)`，tile 末 4 条压缩 DMA 写 `alpha/rrms_h/rrms_k/rrms_v` 的 `[t0, t0+RC)` 段（源为逐行 4 条 4 B 写）。
5. **无需 host 预处理**：`d` 不整除时由尾 lane 屏蔽处理（§0.6 R2），6 个输入/输出张量零拷贝（源 host 侧 `F.pad`+裁回 6+2 次拷贝被消除）。

### 1.5 数据流图

```
GM H,k,v ──[T.copy 单行 DMA]──► UB h_ub/k_ub/v_ub (dtype) ──[vld UNPK_B16 + vcvt]──► 向量寄存器
                                                                 │
        ┌────────────────────────────────────────────────────────┴───────────────┐
        ▼                                                                        ▼
  Σh²,Σk²,Σhk·w² (alloc_local 累加器) ──vcadd──► r_h,r_k,alpha ──┐        vhat=f16(α·v)
                                                                  │              │
                                       GM alpha/rrms_h/rrms_k ◄───┘      UB vh_ub (dtype) ──► GM vhat
                                                                                 │ (每行一次)
                                          Σv̂² ──vcadd──► r_v ──► g = v̂·r_v ──► UB slot[4][DW] (fp32 环形窗口)
                                                                                 │
                    UB cwv_ub (4×DW fp32) ──┐              (i≥3 时读 slot[(i-3+p)%4])
                                            ▼                        ▼
                                    c = Σ_p cwv[p]·slot[...] ──► SiLU(c) + α·v ──► UB y_ub ──► GM Y
```
无 L1 / L0C / 跨核 workspace；`slot` 是唯一的跨行中间态（`O(RC)`，不是 `O(M·T)`）。

### 1.6 算法调研与优化分析

#### 1.6.0 算法调研（Algorithm Research）

> 调研对象：**同一数学语义（逐行 RMSNorm 门控 + 因果 4-tap depthwise conv + SiLU + 残差）的算法族**。源算法（两 kernel、逐行物化）
> 只是基线候选之一。本算子属「多步 + 规约 + 窗口」类，按 `algorithm-research.md` §2 需**完整调研**（R1 候选表 + R3 四口径 +
> R4 逐候选）。本地检索：`knowledge/algorithm-candidates.md`（命中 elementwise/reduction/softmax/RMSNorm/GEMM/attention 六族，
> **无** gate-conv 融合族）、`knowledge/INDEX.md` 全部 16 条、pattern-library `constants.md`、
> `python3 .agents/tools/kb_search.py "engram gate conv rmsnorm depthwise conv fusion" --task-dir …`（**no entries matched**，
> 与 conductor 预注入的「无命中」一致）、`examples/` 内 ascend-target 向量算子。
> 未做互联网检索的必要性：候选空间在 R1 表内封闭（4 个方向的恒等变形 + 3 类结构替代），每项都能由结构判据或本平台证据判定弃选，
> 无需外部算法补充（纪律见 `algorithm-research.md` §6）。

**R1 等价化简公式候选**（基线在表内；正式等价论证与收益量化在 §1.6.1）

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 是否纳入 R3 |
|---|---|---|---|---|---|
| 0 | **基线：源算法** | 2 kernel：gate 段逐行物化 `h_norm/k_norm` + 4 次 `reduce_sum`，`vhat` 落 GM；conv 段逐 tap 从 GM 读邻居行、逐 tap 乘 `w_v·rrms_v` | — | — | ✅（基线） |
| 1 | 单 launch + 行所有权 + halo 重算 + UB 环形窗口 | 同数学，跨行依赖用核内顺序替代 GM 中转 | 数学恒等（同式、同输入、同指令序列 → 逐位） | 少 1 次 launch；少 `vhat` 的 GM 写+读（`2·M·T·d·elem`）；4 tap 共享同一 `g` 行 | ✅（采纳，A3） |
| 2 | 门控点积与两个归一化融合 | `dot = r_h r_k Σ_d H k w_h²`（把 `r_h,r_k` 提到求和外） | 数学恒等（分配律）；fp32 重分组 | 每行省 `3d` 次乘，且 `h_norm/k_norm` 两个 fp32 向量不再物化（省 `2d` 驻留 + 2 次全行写回） | ✅（采纳，A1） |
| 3 | `rms_w_v` 折叠进 conv 权重 | `cwv = conv_w·w_v`（每核一次），conv 输入改存 `g = v̂·r_v` | 数学恒等（乘结合律）；fp32 舍入位置移动 | conv 段每 tap 每 chunk 省 1 次乘（`4·d` 次/行） | ✅（采纳，A2） |
| 4 | conv 输入行每行算一次、4 tap 复用 | `g_r` 驻留，不再逐 tap 重算 `v̂·r_v·w_v` | 数学恒等（同一确定性表达式复用 → 逐位） | 每行省 `3×` 的归一化工作量（`3·2d` 次乘） | ✅（采纳，A3） |
| 5 | `√d` 除法折成标量乘 | `alpha = σ(dot·(1/√d))` | 数学恒等（`1/√d` 为编译期常数） | 1 次 64-lane `vdiv` → 1 次标量常量乘 | ✅（采纳，A4） |
| 6 | `r_v` 用未舍入统计量：`mean(v̂²) = α²·mean(v²)` | 省掉 1 个行归约（`Σv̂²`→`Σv²` 可与其他扫描合并） | **不等价**：torch 参考把 `r_v` 定义在**已舍入**的 `v̂` 上（舍入点 ②），改用未舍入统计量是 **dtype 语义改变**（相对偏差 ~5e-4，且 conv 输入同时用了舍入后的 `v̂`，两者不一致） | 省 `d` 次乘 + 1/4 归约 | ❌（语义层不得改；收益 < 收益面：`Σv̂²` 只占每行 ~8%） |
| 7 | Cube 求门控点积（`H_norm·K_normᵀ` 取对角 / ones-GEMM） | 用矩阵单元算 `Σ_d` | 精确算术下恒等 | **负收益**：只取 `RC×RC` 矩阵的对角（`RC-1` 倍浪费），且需 Vector→Cube 中转 + fractal `N≥16`；本算子每行只 1 个输出 | ❌（R3/R4 量化淘汰） |
| 8 | 变换域 conv（FFT / overlap-save） | 频域乘 | 精确算术下恒等 | **负收益**：`N=4 ≪ d`，FFT 代价 `O(d log d) > 4d`，且要 `d` 长变换 + 逆变换，跨行还要重叠保存 | ❌（结构判定：短核） |
| 9 | D 轴跨核切分 + 跨核合并 | 两核各算 `d/2` 的行统计量再合并 | 容差内等价（分组变化） | **负收益**：每行只有 4 个 fp32 标量输出却要跨核同步/原子；且 conv 也需按 `d` 切分（无冲突），但 gate 归约必须合并 | ❌（R3/R4 淘汰：跨核代价 ≫ 收益） |
| 10 | 两遍物化（先写 `vhat`+4 标量到 GM，第二 kernel 读 4 邻居） | = 基线结构显式化 | 数学恒等 | **负收益**：多 1 次 launch + `2·M·T·d·elem` 的中间往返 + conv 段 4 倍 `vhat` 读 | ❌（= 基线，作为 §1.6.3 备选） |
| 11 | host 侧 permute 到 `(M, d, T)` 让 conv 沿 `T` 向量化 | 换轴 | 数学恒等 | **负收益**：两次全张量 permute（`2·M·T·d·elem` 级搬运）> 算子本体；host 只允许 stride/shape 视图（`ascend-constraints` 精神同 `migration-analysis` §6 host 约束） | ❌（R4 淘汰） |
| 12 | `sigmoid`/`SiLU` 的 exp 降代换（`exp2·log2e`） | `e^{-x} = 2^{-x·log2e}` | 容差内等价 | 本平台 `T.simd.vexp` 已是单条 SFU 指令，`vexp2` 不存在（`simd.py` 导出表无 `vexp2`），降代换无收益 | ❌（API 缺失 + 无收益） |

**R2 在线算法**：**有**。理由（结构判据 + 源证据）：
① 门控部分是逐行独立映射（4 个 `Σ_d` 归约，每行 running 状态 O(1) 个 fp32），单遍即在线；
② `conv` 的跨行依赖是**有限窗口**（宽度 4），因此「分块 + 环形窗口 + 3 行 halo」是标准的流式形态：中间状态 `O(RC)` 而非 `O(M·T)`；
③ 源算子自身已用「单遍 gate + 逐行物化」（§0.4 O3）——即该算子族存在单遍形态的直接证据；
④ 更强形态（跨核流式、单核顺序扫全序列）被并行度否决（§5.5：`M·seq_len = 32..288` 行，单核扫全序列只用 1–3 核）。
收益口径：扫描遍数（`H/k/v` 各 1 遍 + halo 重读 `3/RC`）、中间缓冲 `O(RC·DW)`（~12 KB）而非 `O(M·T·d)`（~0.44 MB）落 GM。
⑤ 「在线」不能消除的部分：`r_v` 依赖完整的 `v̂` 行（同一行内），因此 `v̂` 必须先算出并收窄再平方——这是**行内**两遍，不是跨行两遍（`v̂` 留在寄存器/UB）。

**R3 复杂度对比**（四口径；代表 workload = `train-s96`：`M=3, seq_len=96, d=768`, bf16，行数 `N=288`。
FLOPs 只计乘加（乘加=2），超越函数按 1 次计；访存按**核实际 DMA 搬运量**（MTE2+MTE3，含 halo 与 per-core 权重重读）计，
不含 L2 命中后的 HBM 折减；扫描遍数按 `H/k/v` 的行读取次数计）

| 算法候选 | FLOPs | 访存量 (Bytes) | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---|---|---|---|---|---|
| **0 基线（源算法，2 launch，一行一核）** | 6.39 M（manifest 公式） | gate 读 1.34 MB + `vhat` 写 0.44 MB + conv 读 4×0.44=1.77 MB + `v` 0.44 MB + `Y/vhat` 写 0.66 MB ≈ **4.66 MB**（每核另读权重 6d·2 B） | 1（gate）+ 4（conv 邻居行） | `O(M·T·d)` = 0.44 MB（`vhat` 落 GM） | `N=288` 行一核 → 6 波（288/56）；无跨核代价；**2 次 launch** |
| **1 主选：单 launch + 窗口 + halo（RC=6, grid=48）** | 6.39 M + halo 144 行的 gate 份额 2.32 M ≈ **8.7 M** | `H/k/v` `(RC+3)/RC = 1.5×` = 2.01 MB + 权重 48×9.2 KB = 0.44 MB + `Y/vhat` 0.66 MB + 点积重算 4.6 KB ≈ **3.11 MB** | 1.5（`H/k/v`；halo 重读） | `O(RC·DW)` = **12 KB**（`slot`）+ 44.6 KB 常驻 UB | grid=48 → **1 波**；无跨核代价；**1 次 launch** |
| 2 单 launch + RC=1（最大并行，halo 4×） | gate 段 ×4：`4×4.65 + 1.74` ≈ **20 M** | `H/k/v` 4× = 5.36 MB + 权重 288×9.2 KB = 2.65 MB + 写 0.66 MB ≈ **8.7 MB** | 4 | 同 1（`slot` 只用 1 行有效，4 slot 保留以统一代码） | grid=288 → 6 波；1 次 launch |
| 3 两遍物化（= 基线显式化） | 6.39 M | 同基线 4.66 MB | 1+4 | 0.44 MB 落 GM | 同基线 |
| 4 Cube 求 dot + implicit-GEMM conv | 点积段 `2·d·N` = 0.44 M → `2·RC·d·N` = 2.65 M（×RC=6），整体 6.39 M → 8.6 M | 4.66 MB + L1/L0C 往返 + 跨引擎中转 | 同基线 | L0C `[RC,RC]` fp32 + `vhat` 中转 | 需 Vector↔Cube 跨引擎同步；fractal `N≥16` 与「每行 1 输出」矛盾 |
| 5 变换域 conv | `O(N·d log d)` ≈ 288×768×9.6×2 ≈ 4.2 M（仅 conv 段） | 4.66 MB + 变换中间 | 1 + 2（正/逆变换） | `O(d)` 复数缓冲 ×2 | 同基线 |
| 6 D 轴跨核切分 | 6.39 M | 4.66 MB + 每行 4 B 的部分和（`N` 次原子/合并） | 1（+merge） | 同基线 | 每行一次跨核合并：`N=288` 次；原子路径未在本平台认证 |
| **参考：950DT 只读实现（2 launch，一行一核）** | 6.39 M | ≈ 4.66 MB（与基线同构，无 halo） | 1+4 | `O(M·T·d)` | 6 波 ×2 launch |

**R4 硬件亲和性评估**（检查清单：计算单元匹配 / 片上容量 / 对齐整除 / 静态边界 / 流水可融合 / 跨核结构；负向淘汰附依据）

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---|---|---|---|---|---|---|
| **1 主选（单 launch + 窗口）** | 逐元素 + 4 个行归约 + 4-tap 窗 → 全部落在 Vector（SIMD）单元：`vcadd` 一次完成 64 lane 树归约，`vmul/vadd/vexp/vdiv` 逐 lane | 44.6 KB 常驻 + 12 KB 窗口 / 248 KB = **18%**（`Ascend950PR_9579.ini ub_size=253952`）✅ | chunk = 64 元素 = 1 个 fp32 寄存器；`d ∈ {256,512,768}` 全部 `d % 64 == 0` → 尾 lane 0 浪费 ✅ | `RC` 整除 `seq_len`（构建期保证，见 §5.2）→ 无尾块代码；行循环上界 `RC+3` 静态 | 单 launch 内 MTE2↔Vector 由编译器插入 `asc_sync_*`（探针生成 16 处）；`slot` 单版本 | ✅ **采纳** |
| 2 单 launch + RC=1 | 同 1 | 同 1 | 同 1 | 同 1 | 同 1 | ⚠️ 可用但 halo `4×`、权重读 `288×`（R3 已量化 2.8× 流量） |
| 0/3 基线（2 launch） | 同 1（向量单元够用，但多一次 kernel 边界） | `O(M·T·d)` 落 GM | 同 1 | 同 1 | 两次 launch 之间必须序列化（同一 stream）→ 无重叠 | ⚠️ 备选（§1.6.3 裁决） |
| 4 Cube 求 dot | ❌ 主导计算是「每行 1 个标量」，Cube 的 16×16 fractal 让 15/16 的 N lane 空转；且 `vhat` 要跨引擎中转 | L0C 需 `RC×RC` fp32（6×6×4=144 B，不成 fractal） | 需 pad 到 16×16（浪费 7×） | — | 需 Vector→Cube 同步（`ascend.batch_schedule` 跨 block barrier 未认证） | ❌ FLOPs ×6 + 跨引擎，无收益 |
| 5 变换域 | Vector 可做，但 `d log d` 逐行 FFT 无本平台已验证实现（`simd.py` 无 FFT 原语） | 复数缓冲 ×2 | — | — | — | ❌ API 缺失 + 复杂度更高 |
| 6 D 轴跨核切分 | Vector ✅ | 同 1 | 半个寄存器（32 lane）→ lane 浪费 50% | — | 需 GM 原子/第二次 merge（`atomic_add` 存在但 fp32 GM 原子代价未实测） | ❌ 每行 4 B 输出换一次跨核合并 |

**调研结论**：选定 **候选 1 的算法族（单 launch、行级流式、UB 环形窗口 + 3 行 halo 重算），并在其上采纳 R1 候选 2/3/4/5 的
公式级等价优化（A1–A4）**；与基线的结构差异一句话：**把源算子的「两次 kernel + `vhat` 经 GM 往返」替换为「一次 launch +
核内连续行所有权 + 4 行环形窗口」，并把三处归一化/权重乘法折叠进点积与 conv 权重**。
依据：R3 中候选 1 的访存 3.11 MB 与中间缓冲 12 KB 都优于基线的 4.66 MB / 0.44 MB（省一次 launch 的固定开销），
候选 2 把流量推到 8.7 MB、候选 4/5/6 分别把 FLOPs ×6 / 复杂度升高 / 引入跨核合并，均在本平台 Vector 单元的发射/流量比下为负收益；
R4 中候选 1 在计算单元、容量、对齐、静态边界、流水五项全绿。
**「无更优替代」的调研范围**：已查 `knowledge/algorithm-candidates.md` 全表（6 族）、`knowledge/INDEX.md` 全部条目
（含 `operator-patterns.md` 的 RMSNorm/LayerNorm、`numerical-boundaries.md` 的 SIMD 归约、`PL-950-vector-execution.md`）、
pattern-library `constants.md`、`examples/` 内全部 ascend-target 向量算子（`mla_dense_bwd/_delta_kernel`、`moe_permute_align`、
`paged_cache_write`、`rope`、`ssd_chunk_coupling`、`shared_expert_mlp`）以及 kb_search（无命中）；
**没有发现能同时减少扫描遍数与中间缓冲的其它算法族**（`Σ`/窗口两类原语的可选结构已在 R1 第 1–12 行枚举完）。
本结论即 §1.4、§1.6.1–§1.6.3 的分析对象。
**源算法优化意图的承接**：§0.4 O1（行内驻留融合）→ 候选 1 的 UB 行缓冲 + 寄存器累加器；O2（warp 归约）→ `vcadd`；
O3（gate 单遍）→ 保留并扩展为「一次 chunk 循环算 3 个统计量」；O4（跨 kernel 物化）→ 被候选 1 的窗口替代（意图：「让邻居行可见」
由「同核连续所有权」达成，代价从 GM 往返变为 `3/RC` 的 halo 重算）；O8（逐 tap 重乘）→ A2+A3 消除；
O11/O12（threads/fast-math）→ §0.5 舍弃（理由已记）。

**设计期估算（D-2）**：代表 workload `train-s96`（最大）与 `train-s32`（smoke）。估算下界 = max(流量项, 发射项, 容量项)：

- **流量项**：候选 1 的 3.11 MB。带宽常数在 950PR 上**未实测**（平台 `cost_measurements` 为空），只能用 910B2C 常数外推：
  - HBM 口径：`CONST-mte2-degradation` 的每核地板 18.4 GB/s × 56 核 = **1.03 TB/s** → 3.11 MB / 1.03 TB/s = **3.0 µs**；
  - L2 驻留口径（Stage 4 bench 循环下真实情形，同条目 43.1 GB/s/core）：43.1 × 56 = 2.41 TB/s → **1.29 µs**；
  - 两个常数都绑定 910B2C / `tilelang 0.1.2+15ad002b3d`，**不是** ascend-950pr-9579 的实测值 → 标注「未实测外推」。
- **发射项**：每 owned 行 ≈ 618 条向量指令、halo 行 ≈ 294 条（口径见下表，64 元素/chunk）：`288×618 + 48×3×294`
  = 220 k 条 ÷ 48 核 = **4.59 k 条/核**；按「1 op/cycle/核 @1.8 GHz」（**未实测假设**，取自 `CONST-vector-launch-overhead`
  的 910B2C 量级的等价折算）→ **2.5 µs**。若 `vcvt/vdiv/vexp` 为多周期，上界按 ×1.5 计 ≈ 3.8 µs。
- **容量项**：44.6 KB / 248 KB = 18%（UB 容量取 `Ascend950PR_9579.ini ub_size=253952`）→ 不构成约束。
- **估算下界: 2.5–3.8 µs**（流量 1.3–3.0 µs + 发射 2.5–3.8 µs 取 max，容量项不约束；`train-s32` 为 0.3 µs 级流量 + 0.5 µs 级发射，
  **两条 workload 都被「每 launch 固定开销」（未实测，量级 2–5 µs）主导**，故本设计把 launch 次数从 2 降到 1）。
- **与 manifest roofline 的同口径交叉核对**（`examples/TileOPs/tileops/manifest/sequence_modeling.yaml` 的 `roofline.flops/bytes`）：
  该式 `M*(seq_len*(21d+14) + 2d*(T+(T-1)+(T-2)+(T-3)))` = **6.39 MFLOPs**、`bytes` = **2.00 MB**（读 1.34 + 写 0.66），
  与本节 R3 的 FLOPs 及「无冗余」访存口径逐项吻合；本设计的 3.11 MB 是「含 halo 与 per-core 权重重读」的实现量，
  比 manifest 多 1.11 MB——差额全部来自 A3 的 halo（0.67 MB）与 per-core 权重（0.44 MB），已在 §1.6.3 的裁决计划中作为
  备选（两段 launch，无 halo）的比较项。

**逐行指令数估算口径**（64 元素/chunk，`d=768 → nchunks=12`；用于复核上面的发射项）

| 段 | 每 chunk 指令 | ×12 | 行级指令 | 合计 |
|---|---|---|---|---|
| gate（vld×2 + vcvt×2 + vmul×3 + vadd×3 + vld w2） | 11 | 132 | 3 `vcadd` + ~10 | **~145** |
| vhat（vld + vcvt + vmul + 收窄 vcvt + vsts_pack + 展宽 vcvt + vmul + vadd） | 8 | 96 | `vcadd` + ~5 | **~101** |
| slot（vld + vcvt + vmul + vsts） | 4 | 48 | — | **48** |
| conv（4 tap × (vld slot + vld cwv + vmul + vadd)） | 16 | 192 | — | **192** |
| SiLU+残差+收窄（vneg+vexp+vadd+vdiv；vld v + vcvt + vmul + vadd；vcvt + vsts） | 11 | 132 | — | **132** |
| **owned 行** | | | | **618** |
| **halo 行**（gate+vhat+slot） | | | | **294** |

#### 1.6.1 数学等价优化（公式级）

> 分析对象为 §1.6.0 选定的候选 1（单 launch + 窗口 + halo）。4 个采纳项都不改数学内容，只改「乘的分配位置 / 权重折叠 /
> 复用粒度 / 标量常量位置」。机器验证脚本：`verify_equiv.py`（本目录，torch CPU，fp64 参照；两式均以 fp32 执行后按目标 dtype
> 舍入对照；6 个几何 × 6 个输入模式（random/zeros/large/tiny/nonfinite/ones_w）× 2 dtype = 72 用例/项；
> 判决口径 = `|cand−base| ≤ atol + rtol·|base|`（op 自身容差：fp16 `0.1/0.1`、bf16 `0.2/0.1`；fp32 标量 `1e-4/1e-4`））。

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算 |
|---|---|---|---|---|---|
| **A1** | 门控点积与两个 RMSNorm 归一化融合 | `dot = Σ_d (h_d·r_h·w_d)·(k_d·r_k·w_d)`，需先物化 `h_norm`、`k_norm` | `dot = r_h·r_k·Σ_d h_d·k_d·w_d²`（`w_d²` 每核预算一次） | 数学恒等（乘法结合/分配律，`r_h,r_k` 与 `d` 无关可提出）。fp32 舍入：改的是「每个 `d` 项的相乘顺序」（5 次乘 → 2 次乘），逐项相对误差 ~1e-7，36 用例最大 **9 ulp**（fp16）/ 7 ulp（bf16），容差预算占用 **0.1%** | 每行省 `3d` 次乘（`d=768` → 2304 次乘 ≈ 36 条 `vmul`）；`h_norm/k_norm` 两个 fp32 行向量（各 `d·4 B = 3 KB`）不再物化，省 2 次全行写回 + 2d 个驻留值 |
| **A2** | `rms_w_v` 折叠进 depthwise conv 权重 | `normed = v̂·r_v·w_v`，`c += cw[p]·normed` | `g = v̂·r_v`（每行一次），`cwv[p] = cw[p]·w_v`（每核一次），`c += cwv[p]·g` | 数学恒等（`cw·(v̂·r_v·w_v) = (cw·w_v)·(v̂·r_v)`，标量-向量乘可交换）。fp32 舍入：`fl(cw·w_v)` 一次 + 每行 `fl(v̂·r_v)`，与「每 tap 两次乘」的舍入位置不同，36 用例最大 **1 ulp**（fp16）/ 3 ulp（bf16），容差预算 ≤ 0.18% | conv 段每 tap 每 chunk 省 1 次乘 → 每行省 `4d` 次乘（`d=768` → 3072 次 ≈ 48 条 `vmul`）；`w_v` 不再在行内参与广播 |
| **A3** | conv 输入行每行算一次、4 tap 复用（公共子表达式） | 每 `(输出行, tap)` 重算 `g = v̂·r_v·w_v` → 同一源行被算 4 次 | `g_r` 算一次写入 `slot[r % 4]`，4 个输出行各读一次 | 数值上**逐位相同**（同一确定性表达式、同一输入、同一指令序列；36 用例 **0 ulp**，容差预算 0.18% 的 1 ulp 来自 A2 合并生效） | 省 3× 的归一化工作量（`3·2d` 次乘/行）；同时**消除** `vhat` 的 GM 写+读（`2·M·T·d·elem` = 0.44+1.77 MB → 0，与 A3 一起把中间态压到 12 KB） |
| **A4** | `√d` 除法折成标量乘 | `alpha = σ(dot / √d)` | `alpha = σ(dot · (1/√d))`，`1/√d` 为编译期常数 | 数学恒等（`1/√d` 为精确常量以外，仍是同一实数除法，仅 fp32 舍入次序不同：先取倒数再乘 vs 直接除）。36 用例最大 **3 ulp**，容差预算 0.07% | 1 次 64-lane `vdiv` → 1 次标量立即数乘（`d=768` 时每行省 1 条 SFU 指令 + 1 次广播） |

**等价性机器验证结果**（`verify_equiv.py` 实跑输出，torch CPU + fp64 参照；用例 = 6 几何 × 6 输入模式 × 2 dtype）

| 采纳项 | dtype | 对照字段数 | 最大 ulp(cand, base) | 最大 abs(cand, base) | 尺度相对误差(vs fp64) | 最大容差预算占用 | 非有限不一致 | 结论 |
|---|---|---|---|---|---|---|---|---|
| A1 | float16 | 36 | 9 | 1.788e-07 | 1.271e-07 | 0.0010 | 0 | **EQUIV_PASS** |
| A2 | float16 | 36 | 1 | 2.441e-04 | 3.639e-04 | 0.0018 | 0 | **EQUIV_PASS** |
| A3 | float16 | 36 | 1 | 2.441e-04 | 3.639e-04 | 0.0018 | 0 | **EQUIV_PASS** |
| A4 | float16 | 36 | 3 | 1.192e-07 | 1.169e-07 | 0.0007 | 0 | **EQUIV_PASS** |
| ALL | float16 | 216 | 1 | 2.441e-04 | 3.647e-04 | 0.0018 | 0 | **EQUIV_PASS** |
| A1 | bfloat16 | 36 | 7 | 1.192e-07 | 1.181e-07 | 0.0008 | 0 | **EQUIV_PASS** |
| A2 | bfloat16 | 36 | 3 | 3.815e-06 | 2.857e-03 | 0.0000 | 0 | **EQUIV_PASS** |
| A3 | bfloat16 | 36 | 3 | 3.815e-06 | 2.857e-03 | 0.0000 | 0 | **EQUIV_PASS** |
| A4 | bfloat16 | 36 | 4 | 1.192e-07 | 1.181e-07 | 0.0007 | 0 | **EQUIV_PASS** |
| ALL | bfloat16 | 216 | 3 | 3.815e-06 | 2.857e-03 | 0.0000 | 0 | **EQUIV_PASS** |

> **逐输出明细（`train-s96`、bf16、random，A1–A4 全开）**：`Y` 2 ulp / 预算 0.12%；`vhat` 1 ulp / 0.23%；`alpha` 4 ulp / 0.08%；
> `rrms_h`、`rrms_k` **0 ulp**（同表达式同分组）；`rrms_v` 224 ulp / 预算 24.9%——该偏差与**基线自身对 fp64 的偏差**
> （223 ulp / 24.8%）同阶，根源是 `r_v` 定义在**已舍入到 16-bit** 的 `v̂` 上（语义舍入点 ②，不是本设计的近似），
> 属源算子固有性质，列入 §9.1 风险 R5 并据此把 fp32 标量的验收容差定为 `rtol=1e-3`（比实测 2.5e-5 宽 40 倍，仍严于 op 测试的 0.1）。
> **`nf_mis = 0`** 覆盖非有限模式（`±inf`、`NaN`、`inf·0`）：两式以同样方式传播非有限值，无额外 NaN。
> 早期版本曾把 ulp 相对 `ref` 的**逐元素**网格计（在零/抵消元素上得到 5 位数字），改为「按被评量 dtype 网格」后上表即稳定值；
> 尺度相对误差列同时给出，避免读者误读 ulp。

**优化结论**：采纳 **4 项**（A1 门控融合、A2 权重折叠、A3 conv 输入行复用、A4 标量常量折叠），全部机器验证
**EQUIV_PASS**、无失败项。优化后公式（供 §3.1 拆解使用）：

```
# 每核预置
w2h[j] = w_h[j]²            (fp32, 一次)
cwv[p][j] = conv_w[p][j]·w_v[j]   (fp32, 一次, p=0..3)

# 每行 r（i = r - t0 + 3）
sh = Σ_j h_j² ;  sk = Σ_j k_j² ;  shk = Σ_j h_j·k_j·w2h[j]        # 一次扫描, 3 个累加器
r_h = 1/√(sh/d + eps) ;  r_k = 1/√(sk/d + eps)
alpha = σ( shk·r_h·r_k·(1/√d) )                                   # A1 + A4
v̂ = fl16(alpha·v) ;  r_v = 1/√(Σ_j v̂²/d + eps)                   # 舍入往返后才做平方和
g[j] = v̂[j]·r_v                                                   # A2 (w_v 已折进 cwv)
slot[r % 4] = g
# owned 行
c[j] = Σ_{p=0..3} cwv[p][j]·slot[(i-3+p) % 4][j]                   # A2 + A3
Y = fl16( SiLU(c) + alpha·v )                                      # 残差用未舍入 alpha·v
```

无进一步优化空间的结论与依据：其余计算已是「每元素必要的最小操作集」——
`Σh²/Σk²/Σh·k·w²` 是 3 个不可再约的独立归约（`Σh²` 与 `Σk²` 无公共子式）；`SiLU`/`σ` 是 1 次 `vexp` + 1 次 `vdiv`（本平台
无更廉价的等价原语，`vexp2` 不存在）；`v̂` 的舍入往返是**语义要求**（§0.1 舍入点 ②），不能省；4 tap 的乘加已由 A2/A3 降到
每元素 1 乘 1 加；`1/√(·)` 需要 `r_h/r_k/r_v` 三个输出，无法再合并（`r_h·r_k` 已在 A1 内合并）。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由（不可替代时必填） |
|---|---|---|---|---|---|
| V1 | 行内平方/乘积/累加（`Σh²`、`Σk²`、`Σh·k·w²`、`Σv̂²`） | 逐元素标量循环 | `T.simd.vmul/vadd` + `alloc_local` 向量累加器（64 lane） | ✅ | — |
| V2 | 行内 4 个归约 | GPU `T.reduce_sum(dim=1)` / 标量累加 | `T.simd.vcadd(acc, full)`（64 lane 成对树，一次出标量） | ✅ | — |
| V3 | 16-bit ↔ fp32 | 逐元素 cast | `T.simd.vld(..., dist="UNPK_B16")` + `vcvt(part=0)`；收窄用 `vcvt(..., dtype, round="ROUND_R")` + `vsts(dist="PK_B32")` | ✅ | — |
| V4 | `r_h/r_k/r_v` 的倒数平方根 | `1/sqrt(x)` 标量 | `vdiv(one, vsqrt(x))`（一次性 64 lane，结果广播） | ✅ | — |
| V5 | `alpha`、`SiLU` 的 exp / 除法 | 标量 `exp` + 除法 | `T.simd.vexp` / `vneg` / `vadd` / `vdiv` | ✅ | — |
| V6 | conv 的 4 tap 累加 | 逐 tap 逐元素标量乘加 | 每 tap 一次 `vld(slot)·vld(cwv)` + `vmul/vadd`（64 lane），tap 用 `T.unroll(4)` | ✅ | — |
| V7 | 因果零边界（`src_t < 0` 的 tap） | 逐元素 `if_then_else` | 3 个 halo slot 预清零（向量 `vsts`）+ 块级标量 `if r >= 0` | ✅ | — |
| V8 | 行搬入 / 行写出 | 逐元素 load/store | `T.copy`（单行 DMA，`d` 任意宽度，探针生成 `asc_copy_gm2ub_align(..., 1, 1536, ...)`） | ✅ | — |
| V9 | 4 个 fp32 行标量的输出 | 逐行 4 次 4 B 标量写 | `vsts(..., ONEPT_B32, extent=1)` 暂存 + tile 末 4 条压缩 DMA（探针生成 `asc_copy_ub2gm_align(..., 6, 4, ..., 4, 32)`） | ✅ | — |
| V10 | 行循环 `for i in T.serial(RC + 3)` | 外层行迭代 | 无（每行是一次完整的向量行处理） | ❌ | **tile 级顺序依赖**：`slot` 是跨行携带的环形窗口（`i` 写、`i+1..i+3` 读），conv 的 4 个 tap 分别来自 4 个不同行；把 4 行压进一个向量需要 UB 内 4×DW 转置（`vintlv/vdintlv` 链），代价 ≥ 4 条/chunk 而省下的只是行循环开销（每行 ~5 条标量指令），且转置后 conv 的「同 `j` 跨 `t`」语义需额外的 tile 边界处理 |
| V11 | chunk 循环 `for ch in T.serial(nchunks)` | 行内分段 | 保留为 IR 循环（每段自带 load→compute→store，无跨迭代向量值） | ❌ | **API 约束**：向量累加器不得跨 IR `for` 迭代传递（"Immutable variable … used outside its defining region"，见 `examples/mla_dense_bwd/_delta_kernel/DESIGN.md` §9.2 的探针 F7）；跨迭代累加已由 `T.simd.alloc_local` 寄存数组承担（950DT 参考同形），故该循环本身保留 |
| V12 | 核/块索引（`b, c, t0, r, i % 4`） | 标量整数运算 | 无 | ❌ | **block 级索引**：每核一次、每行一次，不是逐元素热点；`i % 4` 由编译器折成 `i & 3`（探针生成的 `(i_1 & 3)`） |
| V13 | `d % 64 != 0` 的尾 lane | 逐元素谓词 | `T.simd.update_mask(min(64, d-ch*64), 32)` 整个 chunk 一次 | ✅ | — |
| V14 | halo 是否计算 / 是否 owned（`r >= 0`、`i >= 3`） | — | 块级标量分支 | ❌ | **静态边界 + block 级条件**：每核 3 次判断，非逐元素；若改成「运行期 mask」则需对整个行做谓词乘（每行 +`nchunks` 次 `vmul`），反而更贵 |

**向量化结论**：逐元素与归约计算点（V1–V9、V13）**已全部向量化**，所用 API（`vld/vsts/vcvt/vmul/vadd/vdiv/vexp/vsqrt/vneg/vcadd/update_mask/pset/alloc_local`、`T.copy`）
均有本平台证据（`simd.py` 导出表 + `PL-950-simd-reduce`、`PL-950-normalization` 已验用例 + 本任务探针 P1–P8 的生成代码，
见 §3.4）。保留 **5 处**循环/标量，理由类别分别为 **tile 级顺序依赖**（V10 行循环）、**API 约束**（V11 chunk 循环的累加器形态）、
**block 级索引**（V12）、**静态边界/块级条件**（V14）——无一处以「实现简单 / 照源码写」为由保留。

#### 1.6.3 向量化轴与数据布局决策

**轴质量评分（候选逐项）**

| 评分项 | 说明 | 候选 1（主选） | 候选 2 | 候选 3 |
|---|---|---|---|---|
| 向量化轴 | lane 映射到什么轴 | `d`（行内连续），64 lane | `t`（行方向），lane = 相邻行 | `d` + 行内转置 |
| 整除性 | 轴长 vs 向量宽度 | `d ∈ {256,512,768}` 全部 `d % 64 == 0` → 恰好整数个 chunk ✅ | `seq_len ∈ {32,64,96,128}`，`RC ≤ 8` → 每向量寄存器只用 8/64 lane ❌ | 同候选 1 |
| 尾 lane 浪费率 | `(ceil(len/64)·64 − len)/len` | **0%**（4 条 workload） | conv 段 `(64−RC)/RC` = 700%–100% ❌ | 0% |
| 累加链形态 | 串行维是否阻碍向量化 | 归约轴 = 向量维（`vcadd` 一次出标量），行/`t` 是独立归约实例（无累加链）✅ | 归约轴 `d` 变成**跨 chunk 串行**（lane = 行）→ 累积在 lane 上（可行），但 conv 的 `t` 维变成 lane 后每个 tap 是 lane 平移（需 `vsel`/移位，非连续 load） | 同候选 1 |
| repack 代价 | 核内重排成本 | **无**（I/O 原生 `(M,T,d)`，行内连续直接用） | 需把 `(RC, DW)` 转成 `(DW, RC)`：`RC·DW` 元素重排（`d=768, RC=6` → 4608 元素 ≈ 72 个向量寄存器对 + `vintlv` 链），估算 ≥ 300 条指令/核 | 同候选 2 |
| UB 容量影响 | 驻留预算 | 44.6 KB / 248 KB（18%）✅ | 转置缓冲另加 `RC·DW·4 B` = 18 KB | 同上 |

**候选矩阵**（全部枚举；弃选给量化理由）

| # | 布局方案 | 向量化轴 | repack 路径 | 预估收益 / 代价 | 是否采纳 |
|---|---|---|---|---|---|
| 1 | **I/O 原生 `(M,T,d)` + 行内 chunk 流 + UB 环形窗口** | `d`（chunk = 64 元素 = 1 个 fp32 寄存器） | 无（行搬入即 DMA 单行，`d` 任意宽度） | 尾 lane 0%；每行 618 条指令（§1.6.0 表）；UB 44.6 KB | ✅ **主选** |
| 2 | 核内转置成 `(d, t)`（lane = 行），归约变 lane 累积、conv 变 lane 平移 | `t` | UB 内 `RC×DW` 转置（`vintlv/vdintlv` 链） | 省掉每行 4 次 `vcadd`（约 4/618 = 0.6% 指令），但转置 ≥ 300 条/核 + 18 KB 缓冲 + conv 的 tap 需跨 lane 取数（`vsel` 或重复 load）→ **净亏** | ❌ 弃选 |
| 3 | host 侧 permute 成 `(M, d, T)` | `t` | 两次全张量 permute | 额外 `2·M·T·d·elem`（0.9 MB @train-s96）搬运 ≫ 算子本体 3.11 MB 的 29%，且 host 只允许 stride/shape 视图 | ❌ 弃选 |
| 4 | Cube 布局（`H_norm·K_normᵀ` 走 L1/L0 分形） | 矩阵 `(M·T, d)` | 需 nd2nz 装载 + L0C→UB 中转 | FLOPs ×(RC) 浪费 + 每核 2 次布局转换 | ❌ 弃选（§1.6.0 R4） |
| 5 | 按 `d` 分块到多核（D 轴切分） | `d`（一半） | 无 | lane 只用 32 → 浪费 50%；每行一次跨核合并 | ❌ 弃选（§1.6.0 R3/R4） |

**布局决策结论**：选定**候选 1**——核内布局 = 行内连续 `(DW,)` 缓冲 + `slot[4][DW]` fp32 环形窗口，
**向量化轴 = `d`**（chunk = 64 元素 = 一个完整 fp32 向量寄存器，`d % 64 == 0` 的 4 条 workload 尾 lane 浪费 0%），
**repack 路径 = 无**（`T.copy` 单行 DMA 直接落到行缓冲；输出同样单行写出）。
该结论已同步落入 §3.3 伪代码（`h_ub[ch * 64]`、`slot[i % 4, ch * 64]`）与 §6 循环结构（内层向量维 = `d`，
行循环是归约实例 + 窗口携带维）；弃选候选的量化理由见候选矩阵。

**实验裁决计划（三件套；判定裕度依赖未实测常数）**：主选（单 launch + halo）与备选（两段 launch、每行一核）的差异集中在
**两个本型号未实测的量**上：① 每 launch 的固定开销（本设计假设 2–5 µs，平台 `cost_measurements` 为空，无 950PR 实测）；
② `(RC+3)/RC` 的 halo 重读与 per-core 权重重读在 950PR 上的实际代价（只有 910B2C 常数可外推）。
按 §1.6.0 的估算，二者在发射项上相差 ~13%（主选 4590 op-times/核 × 1 波 vs 备选 3960 op-times 但 2 次 launch），
流量项主选 3.11 MB / 备选 4.66 MB → 判定裕度落在常数①的不确定区间内，故：

- **主选（进 Stage 3）**：单 launch + `RC` 规则（§5.2）+ 环形窗口 + halo（§3.3 伪代码，探针 P1–P8 已 COMPILE OK）。
- **备选（≤1 个，结构完整）**：**两段 launch**——launch A `grid = M·seq_len`（一行一核）算 gate + `vhat/alpha/rrms_*` 落 GM；
  launch B `grid = M·seq_len` 读 `vhat[4 行]/rrms_v[4]/v/w_v/conv_w` 做 conv+SiLU+残差。结构与 §0.5 的 950DT 参考同构，
  但**带 A1/A2/A4**（不含 A3 的窗口复用，conv 段逐 tap 乘 `cwv` 与 `g = v̂·r_v`）；分核三要素按 `grid = 288/128/128/32` 重算
  （288 → 6 波；32–128 → 1–3 波）；UB 预算按「单行 `DW` 缓冲 ×3 + `cwv` 4×DW」= 21.5 KB（`d=768`）重算；
  `slot` 缓冲消失，`vhat_ub` 单行；无 halo、无块级 `r >= 0` 分支。
  选择它作为备选的原因：它是**唯一能去掉 halo 冗余**的结构，且 950DT 参考已在其上跑过正确性；若实测 launch 固定开销 ≪ 估算，
  它会因 13% 的发射优势与更少的分支而胜出。
- **裁决计划**：代表 shape = `train-s96`（288 行，halo 收益最大）、`train-s32`（32 行，launch 开销占比最高）、
  `train-s128`、`train-s64`；指标 = `msoprof` 的 Task Duration(us)（Stage 4 口径，A/B 交错 ≥5 次）；
  **判定阈值**：备选 latency < 主选 × 0.9 才翻转；**按 shape 分片裁决**（本算子是 `lru_cache` 形状特化工厂，
  允许「小 shape 用主选、大 shape 用备选」的分派，不必全局二选一——分派条件写入构建期规则）；裁决结果回写本节并进入 §9 风险表。
- **设计期探针（已完成，不是裁决替代）**：`history_version/design_probe_compile.py` 已确认主选的全部结构可编译（P1–P8，
  6 个几何 ALL_OK），并暴露一处实现陷阱（`T.clear` 对 12 KB fp32 缓冲会降成 1536 次 `float2` 标量存 → 改为向量 `vsts` 清零，
  见 §9.2）。探针只标定「结构可编译」，不产生时延数字，故不替代 Stage 4 实验裁决。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式: SIMD**（任务平台 `programming_modes.choices.simd`；`migration_default = simd`，用户未另行指定）。

### 2.2 选型理由

- 算子**不含任何矩阵乘**：只有逐元素乘加、4 个行内归约（`Σ_d`）和一个 4-tap 窗口累加。Cube 的 fractal 语义（`N ≥ 16`）与
  「每行 1 个标量输出」不匹配（§1.6.0 R4 候选 4 已量化淘汰），故排除 `Cube` 与 `混合`。
- 归约与窗口是唯一的结构性计算：`SIMD` 用 `vcadd` 一条指令完成 64 lane 成对树归约、结果留在寄存器；`SIMT`
  （`T.SimtVF` + `T.Parallel` + `alloc_reducer`）每行要做一次跨线程 AllReduce（barrier + 共享内存中转），
  而本算子每行有 4 个归约、行数 32–288、`grid ≤ 56`（每核 4–9 行）——SIMT 的 barrier 次数 = 行数×归约数，
  在 latency 场景下把每行固定成本从「1 条 `vcadd`」抬到「1 次 block 同步」，与 §1.6.0 的发射项估算冲突。
- 逐元素链（`vcvt/vmul/vadd/vexp/vdiv/vsqrt/vneg`）在 64 lane fp32 上铺满一个向量寄存器，与 `d ∈ {256,512,768}`
  （= 4/8/12 个 chunk）恰好整除（§1.6.3）。
- SIMT 在同一 kernel 内也不能替代 SIMD 的寄存器级累加（`alloc_local` + `vcadd` 的形态在 `PL-950-simd-reduce` 与
  950DT 参考中均已验证），两者混用无收益。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|---|---|
| 内存分配 | `T.alloc_shared`（UB 253952 B/核）：`h_ub/k_ub/v_ub/vh_ub/y_ub/wh_ub/wv_ub`（dtype，`DW`）、`cw_ub`（4×DW dtype）、`w2h_ub`（DW fp32）、`cwv_ub`（4×DW fp32）、`slot`（4×DW fp32）、`st`（4×(RC+3)×8 fp32）；不使用 L1/L0A/L0B/L0C |
| 计算方式 | `with T.SimdVF():` 内 chunk 循环 `T.serial(nchunks)` + `vld → vcvt → vmul/vadd/vexp/vdiv/vsqrt → vsts`；无 `T.Parallel`（`SimdVF` 无线程域） |
| 并行来源 | 核间：一维 `T.Kernel(grid)`（`grid = 32..48`）；核内：向量宽度（64 fp32 lane）× chunk 循环 × 行循环 |
| 同步 | 编译器自动插入 `asc_sync_notify/asc_sync_wait`（探针生成的 16 处配对于 MTE2/V/MTE3/S 之间，§7） |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式为 §1.6.1 的**优化后公式**（A1+A2+A3+A4）。

| 步骤 | 数学表达 | 说明 |
|---|---|---|
| 1 | `b = bx / nch`、`c = bx % nch`、`t0 = c·RC`、`r = t0 − 3 + i` | tile 任务映射（block 级标量） |
| 2 | `h_ub[j] ← H[b, r, j]`，`k_ub`、`v_ub` 同（`j < d`） | GM→UB 单行 DMA（`if r ≥ 0` 保护） |
| 3 | `sh = Σ_{ch} Σ_lane (h·h)`、`sk = Σ (k·k)`、`shk = Σ (h·k)·w2h` | 一个 chunk 循环内 3 个 `alloc_local` 累加器（A1） |
| 4 | `r_h = (sh/d + eps)^{-1/2}`、`r_k` 同 | `vmul(1/d) → vadd(eps) → vsqrt → vdiv` |
| 5 | `alpha = σ(shk·r_h·r_k·(1/√d))` | `vdup` 广播 + `vmul` + `vneg/vexp/vadd/vdiv`（A1+A4） |
| 6 | `v̂ = fl_dtype(alpha·v)`，同时 `Σv̂²` | 收窄 `vcvt(..., dtype, ROUND_R)` → `vsts(PK_B32)` 写 `vh_ub`；展宽后平方累加 |
| 7 | `r_v = (Σv̂²/d + eps)^{-1/2}` | 同步骤 4（统计量取**已舍入** `v̂`） |
| 8 | `g = v̂·r_v` → `slot[i % 4]` | 每行一次（A3）；`w_v` 已在 `cwv` 内（A2） |
| 9 | `c = Σ_{p<4} cwv[p]·slot[(i−3+p) % 4]` | 4 tap 用 `T.unroll(4)`（A2+A3） |
| 10 | `Y = fl_dtype(SiLU(c) + alpha·v)` | `vneg/vexp/vadd/vdiv` + 残差（未舍入 `alpha·v`）→ `vsts(PK_B32)` 写 `y_ub` |
| 11 | `st[k, i, 0] ← {alpha, r_h, r_k, r_v}[k]`；tile 末 `T.copy(st[k, 3:3+RC, 0:1], Out[b, t0:t0+RC])` | 单 lane `vsts(extent=1)` + 压缩 DMA |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API（`tilelang.ascend.language`） | 参数 | 模式 |
|---|---|---|---|---|
| 1 | 核/tile/行索引 | `T.Kernel(grid)` / `T.serial(RC + 3)` / `if r >= 0` / `if i >= 3` | 全部构建期常量 + 2 个块级标量分支 | SIMD（核级） |
| 2 | GM→UB 单行搬入 | `T.copy(H[b, r, :], h_ub[:d])` | 单行、`d` 任意宽度（探针生成 `asc_copy_gm2ub_align(..., 1, 1536, ...)`） | MTE2 |
| 3 | 行内 16-bit 提升 | `T.simd.vld(h_ub[ch*64], dist="UNPK_B16")` + `T.simd.vcvt(v, "float32", part=0)` | 每 chunk 64 元素 → 一个 fp32 向量 | SIMD |
| 3b | 三个统计量累加 | `T.simd.alloc_local((1,), "float32")` + `T.simd.vmul/vadd(..., mask)` | 3 个寄存器累加器，跨 chunk 携带（950DT 参考同形） | SIMD |
| 4 | 行归约 | `T.simd.vcadd(acc[0], full)` + `T.simd.vdupv(scalar, full)` | 64 lane 成对树 → 广播标量 | SIMD |
| 4b | 尾 lane 屏蔽 | `T.simd.update_mask(T.min(64, d - ch*64), 32)` | `d % 64 != 0` 时生效；4 条 workload 退化为全 64 | SIMD |
| 5 | `r_h/r_k/r_v` | `T.simd.vadd/vmul/vsqrt/vdiv`，常量用 `T.simd.vdup(T.float32(c), "float32", full)` | `1/d`、`eps`、`1/√d` 均为构建期浮点常量 | SIMD |
| 5b | `alpha` | `T.simd.vneg/vexp/vadd/vdiv` | `σ(x) = 1/(1+e^{-x})` | SIMD |
| 6 | `v̂` 收窄 + 写回 | `T.simd.vcvt(gated, dtype, round="ROUND_R", sat=False, part=0)` + `T.simd.vsts(vh_ub[ch*64], narrowed, mask, dist="PK_B32")` | 与 golden 的舍入点 ② 一致 | SIMD |
| 6b | `Σv̂²` | `T.simd.vcvt(narrowed, "float32", part=0)` + `vmul/vadd` | 展宽后再平方（**不**用未舍入值） | SIMD |
| 8 | 窗口写 | `T.simd.vsts(slot[i % 4, ch*64], g, mask)` | fp32 全向量存；`slot` 单版本（§4.5） | SIMD |
| 9 | conv 4 tap | `for p in T.unroll(4): acc = vadd(acc, vmul(vld(slot[(i-3+p) % 4, ch*64]), vld(cwv_ub[p, ch*64]), mask))` | tap 循环静态展开；`cwv` fp32 免 `vcvt` | SIMD |
| 10 | SiLU + 残差 + 收窄 | `vneg/vexp/vadd/vdiv` + `vmul/vadd` + `vcvt(..., dtype)` + `vsts(dist="PK_B32")` | 残差取 `alpha·v`（fp32，未舍入） | SIMD |
| 11 | 标量暂存 / 批量写出 | `T.simd.vsts(st[k, i, 0], v, T.simd.pset(32, "PAT_VL1"), extent=1)`；`T.copy(st[k, 3:3+RC, 0:1], Out[b, t0:t0+RC])` | 行距 32 B（`st` 末维 8）× 4 个输出（探针生成 `asc_copy_ub2gm_align(..., 6, 4, ..., 4, 32)`） | SIMD + MTE3 |
| 12 | 参数预置 | `T.copy(WH, wh_ub[:d])` + `vcvt/vmul/vsts` → `w2h_ub`；`T.copy(CW, cw_ub[:, :d])` + `vmul` → `cwv_ub` | **必须在 `T.SimdVF` 之外搬入**（SIMD_VF 块禁止访问 GM，探针首轮即报该语义检查） | MTE2 + SIMD |
| 13 | 缓冲版本 | `T.annotate_buffer_versions({slot: 1, w2h_ub: 1, cwv_ub: 1, h_ub: 1, k_ub: 1, v_ub: 1, vh_ub: 1, y_ub: 1, st: 1})` | 环形窗口与行缓冲都按单版本（跨行依赖 + 单行使用） | 调度 |

### 3.3 计算伪代码

> **本节形态即探针 `history_version/design_probe_compile.py` 的代码**（6 个几何 ALL_OK：train-s96/s64/s128/s32、prime-95、pad-d196）。
> 四条硬约束（均由探针或平台文档确认）：
> ① `T.SimdVF` 块内**不得**出现 GM `BufferLoad`（首轮探针报 `[TileLang Semantic Check] BufferLoad from global buffer 'WH' inside a SIMD_VF block is not allowed`）；
> ② 向量累加器必须用 `T.simd.alloc_local`（可跨 IR `for` 迭代），不能用 python 变量重绑定；
> ③ chunk 内 16-bit ↔ fp32 用 `UNPK_B16`/`PK_B32` 的 64 元素映射，`part` 参数必须是**构建期常量**（`T.unroll` 的变量不能传给 `part`）；
> ④ 因果零边界只出现在两个块级标量分支（`r >= 0`、`i >= 3`）上，不得下沉到逐元素谓词。

```python
# 构建期常量：M, seq_len, d, dtype, eps, DW = align_up(d, 64), nchunks = DW // 64,
#             RC（整除 seq_len，§5.2）, nch = seq_len // RC, grid = M * nch
with T.Kernel(grid) as bx:
    b, c = bx // nch, bx % nch
    t0 = c * RC
    # --- UB 缓冲（§4.3）+ 单版本注解
    h_ub, k_ub, v_ub, vh_ub, y_ub, wh_ub, wv_ub = [T.alloc_shared((DW,), dtype) for _ in 7]
    cw_ub = T.alloc_shared((4, DW), dtype)
    w2h_ub = T.alloc_shared((DW,), "float32")
    cwv_ub = T.alloc_shared((4, DW), "float32")
    slot = T.alloc_shared((4, DW), "float32")
    st = T.alloc_shared((4, RC + 3, 8), "float32")
    T.annotate_buffer_versions({slot: 1, w2h_ub: 1, cwv_ub: 1, h_ub: 1, k_ub: 1, v_ub: 1,
                                vh_ub: 1, y_ub: 1, st: 1, wh_ub: 1, wv_ub: 1, cw_ub: 1})

    # --- 参数预置（GM 搬入在 SimdVF 之外）
    T.copy(WH, wh_ub[:d]); T.copy(WV, wv_ub[:d]); T.copy(CW, cw_ub[:, :d])
    with T.SimdVF():
        full = T.simd.pset(32); one_lane = T.simd.pset(32, "PAT_VL1")
        zero = T.simd.vdup(T.float32(0), "float32", full)
        for h in T.serial(3):                    # A3/R4: 3 个 halo slot 预清零（向量存）
            for ch in T.serial(nchunks):
                T.simd.vsts(slot[h, ch * 64], zero, mask=full)
        for ch in T.serial(nchunks):             # w2h = w_h²（A1）
            mask = T.simd.update_mask(T.min(64, d - ch * 64), 32)
            wv = T.simd.vcvt(T.simd.vld(wh_ub[ch * 64], dist="UNPK_B16"), "float32", part=0)
            T.simd.vsts(w2h_ub[ch * 64], T.simd.vmul(wv, wv, mask))
        for p in T.serial(4):                    # cwv = conv_w · w_v（A2）
            for ch in T.serial(nchunks):
                mask = T.simd.update_mask(T.min(64, d - ch * 64), 32)
                cf = T.simd.vcvt(T.simd.vld(cw_ub[p, ch * 64], dist="UNPK_B16"), "float32", part=0)
                vf = T.simd.vcvt(T.simd.vld(wv_ub[ch * 64], dist="UNPK_B16"), "float32", part=0)
                T.simd.vsts(cwv_ub[p, ch * 64], T.simd.vmul(cf, vf, mask))

    # --- 行流水（R3/R4）
    for i in T.serial(RC + 3):
        r = t0 - 3 + i
        if r >= 0:                               # 因果 halo 存在才搬入（块级分支）
            T.copy(H[b, r, :], h_ub[:d]); T.copy(K[b, r, :], k_ub[:d]); T.copy(V[b, r, :], v_ub[:d])
        with T.SimdVF():
            sh = T.simd.alloc_local((1,), "float32")   # 3 个跨 chunk 累加器
            sk = T.simd.alloc_local((1,), "float32")
            shk = T.simd.alloc_local((1,), "float32")
            vsq = T.simd.alloc_local((1,), "float32")
            alpha_v = T.simd.alloc_local((1,), "float32")
            rv_v = T.simd.alloc_local((1,), "float32")
            if r >= 0:
                sh[0] = zero; sk[0] = zero; shk[0] = zero
                for ch in T.serial(nchunks):                        # A1: 一次扫描 3 个统计量
                    mask = T.simd.update_mask(T.min(64, d - ch * 64), 32)
                    hf = T.simd.vcvt(T.simd.vld(h_ub[ch * 64], dist="UNPK_B16"), "float32", part=0)
                    kf = T.simd.vcvt(T.simd.vld(k_ub[ch * 64], dist="UNPK_B16"), "float32", part=0)
                    w2 = T.simd.vld(w2h_ub[ch * 64])
                    sh[0]  = T.simd.vadd(sh[0],  T.simd.vmul(hf, hf, mask), full)
                    sk[0]  = T.simd.vadd(sk[0],  T.simd.vmul(kf, kf, mask), full)
                    shk[0] = T.simd.vadd(shk[0], T.simd.vmul(T.simd.vmul(hf, kf, mask), w2, mask), full)
                rh_v = recip_sqrt( vcadd(sh), d, eps )               # 1/√(sh/d+eps)
                rk_v = recip_sqrt( vcadd(sk), d, eps )
                alpha_v[0] = sigmoid( vcadd(shk) · rh_v · rk_v · (1/√d) )   # A1 + A4
                T.simd.vsts(st[0, i, 0], alpha_v[0], one_lane, extent=1)
                T.simd.vsts(st[1, i, 0], rh_v, one_lane, extent=1)
                T.simd.vsts(st[2, i, 0], rk_v, one_lane, extent=1)
                vsq[0] = zero
                for ch in T.serial(nchunks):                        # v̂ = fl_dtype(α·v) + Σv̂²
                    mask = T.simd.update_mask(T.min(64, d - ch * 64), 32)
                    vf = T.simd.vcvt(T.simd.vld(v_ub[ch * 64], dist="UNPK_B16"), "float32", part=0)
                    narrowed = T.simd.vcvt(T.simd.vmul(vf, alpha_v[0], mask), dtype,
                                           round="ROUND_R", sat=False, part=0)
                    T.simd.vsts(vh_ub[ch * 64], narrowed, full, dist="PK_B32")
                    back = T.simd.vcvt(narrowed, "float32", part=0)
                    vsq[0] = T.simd.vadd(vsq[0], T.simd.vmul(back, back, mask), full)
                rv_v[0] = recip_sqrt( vcadd(vsq), d, eps )
                T.simd.vsts(st[3, i, 0], rv_v[0], one_lane, extent=1)
                for ch in T.serial(nchunks):                        # A3: g 每行一次 → 环形窗口
                    back = T.simd.vcvt(T.simd.vld(vh_ub[ch * 64], dist="UNPK_B16"), "float32", part=0)
                    T.simd.vsts(slot[i % 4, ch * 64], T.simd.vmul(back, rv_v[0]), full)
            if i >= 3:                                              # owned 行：conv + SiLU + 残差
                for ch in T.serial(nchunks):
                    mask = T.simd.update_mask(T.min(64, d - ch * 64), 32)
                    acc = T.simd.alloc_local((1,), "float32"); acc[0] = zero
                    for p in T.unroll(4):                           # A2 + A3
                        acc[0] = T.simd.vadd(acc[0], T.simd.vmul(
                            T.simd.vld(slot[(i - 3 + p) % 4, ch * 64]),
                            T.simd.vld(cwv_ub[p, ch * 64]), mask), full)
                    c_v = acc[0]
                    silu = T.simd.vdiv(c_v, T.simd.vadd(T.simd.vdup(T.float32(1), "float32", full),
                                                        T.simd.vexp(T.simd.vneg(c_v, full), full), full), full)
                    vf = T.simd.vcvt(T.simd.vld(v_ub[ch * 64], dist="UNPK_B16"), "float32", part=0)
                    y = T.simd.vadd(silu, T.simd.vmul(vf, alpha_v[0], mask), mask)
                    T.simd.vsts(y_ub[ch * 64], T.simd.vcvt(y, dtype, round="ROUND_R", sat=False, part=0),
                                full, dist="PK_B32")
        if i >= 3:                                                  # 单行 DMA 写回
            T.copy(vh_ub[:d], VHAT[b, r, :]); T.copy(y_ub[:d], Y[b, r, :])

    # --- 4 个 fp32 标量输出：tile 末压缩 DMA（源为逐行 4 条 4 B 写）
    T.copy(st[0, 3 : 3 + RC, 0:1], Alpha[b, t0 : t0 + RC])
    T.copy(st[1, 3 : 3 + RC, 0:1], RH[b, t0 : t0 + RC])
    T.copy(st[2, 3 : 3 + RC, 0:1], RK[b, t0 : t0 + RC])
    T.copy(st[3, 3 : 3 + RC, 0:1], RV[b, t0 : t0 + RC])
```

其中 `recip_sqrt(s, d, eps)` 展开为 `vdiv(vdup(1), vsqrt(vadd(vmul(vdupv(s), vdup(1/d)), vdup(eps))))`，
`sigmoid(x)` 为 `vdiv(vdup(1), vadd(vdup(1), vexp(vneg(x))))`（两者在探针中均以内联形式出现）。

### 3.4 API 可行性确认

| API | 来源（本平台证据） | 状态 |
|---|---|---|
| `T.Kernel(grid)`（一维，无 `threads`） | `references/programming.md`「一维核网格；多维网格在实现中拒绝」；探针 P1 | ✅ 已编译 |
| `T.SimdVF()` + `T.simd.*` | `references/programming.md`；`PL-950-vector-execution`；`PL-950-simd-reduce`（`vcadd/vcmax/vld/vsts`） | ✅ 已编译 |
| `T.simd.vld(..., dist="UNPK_B16")` / `vsts(..., dist="PK_B32")` | 950DT 参考 `gate_conv_fwd.py`（逐 chunk 16-bit 链）；探针 P4（生成 `vlds_unpack<bfloat16_t>` / `vsts_pack_b32`） | ✅ 已编译 |
| `T.simd.vsts(..., mask, extent=1)` + `T.simd.pset(32, "PAT_VL1")` | `PL-950-simd-reduce`；`PL-950-normalization`；探针 P5（生成 `vsts_norm(..., extent/one)`） | ✅ 已编译 |
| `T.simd.alloc_local((1,), "float32")` 跨 IR 循环累加 | 950DT 参考（`hs/ks/dot/vs` 累加器）；`alloc_local` docstring（`for i in T.Unroll(...): v[i] = vld(...)`） | ✅ 已编译 |
| `T.simd.vcadd` / `vdupv` / `vdup` / `update_mask` | `PL-950-simd-reduce`；`simd.py` 导出 | ✅ 已编译 |
| `T.simd.vexp/vdiv/vsqrt/vneg/vmul/vadd` | `simd.py` 导出 + 950DT 参考（`vexp/vdiv/vsqrt/vneg` 全部实跑过） | ✅ 已编译 |
| `T.copy`（单行 GM↔UB，`d` 非对齐宽度） | `references/programming.md`（局部拷贝用显式 region）；950DT 参考 `T.copy(H[batch, token, :], h[:d])`；探针生成 `asc_copy_gm2ub_align(..., 1, 1536)` | ✅ 已编译 |
| `T.copy`（跨步压缩写标量 4×(RC)×4 B） | `examples/mla_dense_bwd/_delta_kernel/DESIGN.md` §1.6.1 A3（同形态已实测）；探针生成 `asc_copy_ub2gm_align(..., 6, 4, ..., 4, 32)` | ✅ 已编译 |
| `T.annotate_buffer_versions({...: 1})` | `references/programming.md`（`{buf: 1}` 取消自动多版本） | ✅ 已编译 |
| `if`（块级标量条件）包住 `T.copy` / `T.SimdVF` | `kernel_template.py`（`if bx < full:`）；950DT 参考（`if token >= …: T.copy`）；探针 P1/P2（生成 `if (1 <= …)` / `if (3 <= i_1)`） | ✅ 已编译 |
| `T.clear`（大 fp32 缓冲） | 可用，但**降级为标量 `float2` 循环**（探针首轮：1536 次迭代）→ 本设计改用向量 `vsts` 清零（§9.2） | ⚠️ 规避 |
| GM 访问位于 `T.SimdVF` 内 | **不允许**（探针报 `[TileLang Semantic Check]`） | ❌ 禁止（已按此改写） |
| `T.reduce_sum`（高层 2-D shared→shared） | `TRAP-950-simd-shared-reduce` 记录输出布局差异 | ❌ 不使用 |
| `T.Pipelined`（行循环） | `references/programming.md`（AutoSchedule 默认开启；跨 block barrier 未认证） | ⏸ 本设计不用（行间有 `slot` 依赖），Stage 4 可试 |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束（来源 `references/programming.md` / `ascend-constraints.md`） | 涉及 | 处理方案 |
|---|---|---|
| Kernel 网格必须一维、无 `threads` 参数 | ✅ | 一维 `T.Kernel(grid)` + `T.SimdVF`（§0.6 R1） |
| SIMD_VF 块内禁止访问 GM | ✅ | 参数与行数据先 `T.copy` 到 UB，`vld` 只读 UB（§3.3 约束①，探针首轮实测到该检查） |
| 向量值不得跨 IR `for` 迭代重绑定 | ✅ | 跨 chunk 累加用 `T.simd.alloc_local((1,), …)`（§3.3 约束②） |
| `vcvt` 的 `part` 必须是构建期常量 | ✅ | 只传 `part=0`（64 元素映射），chunk 循环不把 `part` 参数化 |
| 块级循环上界需静态 | ✅ | `RC`、`nchunks`、`4` 全为构建期常量；行循环 `RC+3` 静态 |
| 尾块：`seq_len % RC != 0` | ⚠️ 已消除 | `RC` 构造上整除 `seq_len`（§5.2 规则），无尾块代码路径 |
| 尾 lane：`d % 64 != 0` | ✅ | `update_mask(min(64, d-ch*64), 32)`（§0.6 R2）；L0-3 用 `d=196`、Boundary 用 `d=200` 覆盖 |
| UB 容量 253952 B + 多版本膨胀 | ✅ | 44.6 KB（`d=768`）→ 18%；单版本注解已把膨胀源（`slot`/行缓冲）钉死（§4.5） |
| L1/L0C 容量 | ❌ 不涉及 | 无 Cube、无 L1 中转 |
| 跨核同步 / 跨 block barrier | ❌ 不涉及 | 单 launch 内每核独立，核间无数据交换 |
| `d ≤ 4096` refusal（与源/950DT 参考同界） | ✅ | `d > 4096` 在 build 期拒绝（UB 核算见 §0.6 R6） |
| fast-math / `compile_flags` | ✅ 不启用 | 不开 `TL_ENABLE_FAST_MATH`：避免 `vmul+vadd` 收缩改变乘积舍入位置，使 A1/A2 的机器验证前提改变 |

#### 3.5.2 参考实现差异说明

| 差异项 | 参考实现（GPU 源算子） | 本项目（Ascend 950PR） | 转换方案 |
|---|---|---|---|
| Kernel 结构 | `T.Kernel(seq_len, M, threads=128)` 二维 grid + thread 域 | 一维 `T.Kernel(grid)`（`grid ≤ 56`）+ `T.SimdVF` 向量域 | §0.6 R1；探针 6 几何 COMPILE OK |
| 阶段结构 | 2 个 `@T.macro`/kernel（gate 段、conv 段）由同一 `prim_func` 串联 | 1 次 launch（`grid = M·nch`），核内连续行 + 环形窗口 + halo | §0.6 R3；备选=两段 launch（§1.6.3） |
| 归约 | `T.reduce_sum(hsq, sumsq_h, dim=1)`（fragment 2-D） | `vcadd`（64 lane 成对树）+ `alloc_local` 累加器 | §0.5 等价替换 |
| 归一化 | 物化 `h_2d *= r_h·w_h`、`k_2d *= r_k·w_h` | 不物化，折进点积 `Σh·k·w_h²`（A1） | §1.6.1 A1（机器验证） |
| padding | `d_padded = align_up(d, 256)`，host `F.pad` 6 张量 + 裁回 2 输出 | `DW = align_up(d, 64)`，尾 lane 屏蔽，host 零拷贝 | §0.6 R2 |
| 因果零边界 | 逐元素 `if_then_else(src_t>=0, …)` ×2（raw 与 rrms） | 3 个 halo slot 预清零 + 块级 `if r >= 0` | §0.6 R4 |
| conv 权重 | 每 tap `conv_w[p]·normed`（`normed` 含 `w_v·rrms_v`） | `cwv = conv_w·w_v` 预算一次，`g = v̂·r_v` 每行一次 | §1.6.1 A2/A3 |
| 标量输出 | 逐行写 `alpha/rrms_*`（4×4 B） | `(4,RC+3,8)` 暂存 + tile 末 4 条压缩 DMA | §1.6.1（V9） |
| dtype/精度 | fp32 全中间量（`accum_dtype="float"`） | 同（fp32 全中间量，7 个舍入点照抄） | 语义层保留 |
| threads/fast-math/compile_flags | 自动调优 threads；`-O3 -DENABLE_BF16`；GPU 侧 fast-math | 全部舍弃 | §0.5 已记理由 |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|---|---|---|
| `examples/mla_dense_bwd/_delta_kernel/_delta_kernel.py` | **高**（同平台、SIMD、行归约 + 逐行标量输出） | 一维 `T.Kernel(grid)` + `T.simd.vcadd` 行归约的完整落地形态；`alloc_local` 标量寄存器；`vsts(PAT_VL1, extent=1)` 写标量；压缩 DMA 写 `(rows)` fp32；`T.copy` 目的端必须显式切片；尾块静态 extent 的写法 |
| `examples/mla_dense_bwd/_delta_kernel/DESIGN.md` | **高**（同平台的 §1.6.0/§1.6.3/§5.5 结构范式） | 调研四问与 roofline 的写法；分核三要素的实查记录范式；§9.2 的 `T.Kernel` 二维拒绝、`immutable variable` 陷阱 |
| `examples/moe_permute_align/_make_scatter_kernel/DESIGN.md` | 中（同平台小 shape / latency 场景） | `grid` 与 56 核的对齐取值、`无需适配` 判定的写法 |
| `.agents/platforms/toolchains/ascend/knowledge/repro/PL-950-simd-reduce.py` | **高**（归约形态的可执行样板） | `vld → vcadd → vsts(extent=1)` 的最小闭环；mask 与统计暂存的行距 32 B |
| `.agents/platforms/toolchains/ascend/knowledge/repro/PL-950-normalization.py` | **高**（RMSNorm 语义样板） | fp32 统计量 + 尾 lane 不计入分母；输出转回输入 dtype |
| `.agents/platforms/toolchains/ascend/kernel_template.py` | 中（Kernel 骨架 + 尾块分支） | `if bx < full:` / `if bx == full:` 的块级分支范式；`--level L0/L1/L2/boundary` 测试骨架 |
| `/workspace/TileOPs-ascend950/src/tileops/kernels/ascend/sequence_modeling/gate_conv_fwd.py` | **最高**（同算子的 Ascend 实现，只读） | 64 元素 chunk 的 `UNPK_B16`/`PK_B32` 链、`update_mask`、`stats(4,8)` 批量标量写、`vcvt` 舍入往返；其「两次 launch」被本设计改为单 launch（§0.6 R3） |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|---|---|---|---|
| `H` | `(M, seq_len, d)` | `float16 \| bfloat16` | 隐藏态；kernel 入参即**未 pad** 的原形状（源算子要求 `d_padded`，本设计不需要） |
| `k` | `(M, seq_len, d)` | 同 `H` | key 投影 |
| `v` | `(M, seq_len, d)` | 同 `H` | value 投影 |
| `rms_w_h` | `(d,)` | 同 `H` | H 与 k 共用 |
| `rms_w_v` | `(d,)` | 同 `H` | vhat 归一化权重 |
| `conv_w` | `(4, d)` | 同 `H` | 4 个 tap 的 depthwise 权重 |

`eps` 为 build 期常量（`1e-6`，来自 manifest `params`）。

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|---|---|---|---|
| `Y` | `(M, seq_len, d)` | 同 `H` | `out_idx` 0；`fl_dtype(SiLU(conv) + alpha·v)` |
| `vhat` | `(M, seq_len, d)` | 同 `H` | `out_idx` 1；`fl_dtype(alpha·v)`（反向保存） |
| `alpha` | `(M, seq_len)` | `float32` | `out_idx` 2 |
| `rrms_h` | `(M, seq_len)` | `float32` | `out_idx` 3 |
| `rrms_k` | `(M, seq_len)` | `float32` | `out_idx` 4 |
| `rrms_v` | `(M, seq_len)` | `float32` | `out_idx` 5 |

`out_idx=[6, 7, 8, 9, 10, 11]`（前 6 个是入参）。输出无 padding、无需裁剪（源算子 host 侧 `results[0][:, :, :d]` 被消除）。

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|---|---|---|---|---|
| `h_ub`, `k_ub`, `v_ub` | `(DW,)` | 输入 dtype | UB（单版本） | 当前行 `H/k/v` |
| `vh_ub`, `y_ub` | `(DW,)` | 输入 dtype | UB（单版本） | 当前行 `vhat` 输出 / `Y` 输出（供单行 DMA 写出） |
| `wh_ub`, `wv_ub` | `(DW,)` | 输入 dtype | UB（单版本） | `rms_w_h` / `rms_w_v` 的一次性暂存 |
| `cw_ub` | `(4, DW)` | 输入 dtype | UB（单版本） | `conv_w` 的一次性暂存 |
| `w2h_ub` | `(DW,)` | `float32` | UB（单版本） | `w_h²`（A1） |
| `cwv_ub` | `(4, DW)` | `float32` | UB（单版本） | `conv_w·w_v`（A2），fp32 免 `vcvt` |
| `slot` | `(4, DW)` | `float32` | UB（单版本，环形窗口） | `g_r = v̂·r_v` 的行滑窗（A3），跨行携带 |
| `st` | `(4, RC+3, 8)` | `float32` | UB（单版本） | 4 个行标量的暂存（行距 32 B，供 `vsts(ONEPT_B32)`） |
| （寄存器）`sh/sk/shk/vsq/alpha_v/rv_v/acc` | 各 1 个 fp32 向量 | `float32` | SIMD 寄存器 | 行内累加与标量，**不物化到 UB** |

### 4.4 内存搬运路径

```
GM H/k/v ──[T.copy 单行 DMA，1536 B @d=768,bf16；块级 if r>=0 保护]──► UB h_ub/k_ub/v_ub
GM WH/WV/CW ──[T.copy 一次/核：1536+1536+6144 B]──► UB wh_ub/wv_ub/cw_ub
UB wh_ub/wv_ub/cw_ub ──[vld UNPK_B16 + vcvt + vmul + vsts]──► UB w2h_ub / cwv_ub（fp32，一次/核）
UB h_ub/k_ub/v_ub ──[vld UNPK_B16 + vcvt]──► SIMD 寄存器 ──[vmul/vadd 累加器 + vcadd]──► 行标量（alpha/r_h/r_k/r_v）
SIMD 寄存器 ──[vmul + vcvt 收窄 + vsts PK_B32]──► UB vh_ub ──[T.copy 单行]──► GM vhat
SIMD 寄存器 ──[vmul(g·r_v) + vsts]──► UB slot[i%4]  （环形窗口，无 GM 往返）
UB slot[(i-3..i)%4] + UB cwv_ub ──[4×vld + vmul/vadd]──► conv 累加器 ──[SiLU + 残差 + 收窄]──► UB y_ub ──[T.copy 单行]──► GM Y
SIMD 寄存器 ──[vsts ONEPT_B32 extent=1]──► UB st[k,i,0] ──[T.copy 压缩 DMA：RC×4 B，源距 32 B]──► GM alpha/rrms_*
（无 L1 / L0A / L0B / L0C / 跨核 workspace；`slot` 与行缓冲均无多版本）
```

### 4.5 UB 内存预算

`d = 768`（`train-s96`，`RC=6`，`DW=768`；最坏几何）：

| Buffer | Shape | dtype | 大小 (Bytes) |
|---|---|---|---|
| `h_ub`/`k_ub`/`v_ub`/`vh_ub`/`y_ub`/`wh_ub`/`wv_ub` | 7 × (768,) | 16-bit | 7 × 1536 = 10 752 |
| `cw_ub` | (4, 768) | 16-bit | 6 144 |
| `w2h_ub` | (768,) | fp32 | 3 072 |
| `cwv_ub` | (4, 768) | fp32 | 12 288 |
| `slot` | (4, 768) | fp32 | 12 288 |
| `st` | (4, 9, 8) | fp32 | 1 152 |
| **总计（手工核算）** | | | **45 696 B = 44.6 KB** / 253 952 B（248 KB，`Ascend950PR_9579.ini ub_size=253952`） |
| 占用率 | | | 45 696 / 253 952 = **18.0%** |
| 按 1.7× 膨胀上界（`CONST-capacity-910B2C`，910B2C 口径，非本型号实测） | | | 77 683 ≤ 253 952 ✅ |
| 按 1.10–1.12× 显式 alloc 口径 | | | 50 266–51 180 ✅ |

| 其余几何 | `DW` | `RC` | `slot`+`cwv`+`w2h` | 7 行缓冲 + `cw` | `st` | 合计 |
|---|---|---|---|---|---|---|
| `train-s32`（M=1,S=32,d=256） | 256 | 1 | 4 096+4 096+1 024 = 9 216 | 3 584+2 048 = 5 632 | 512 | **15 360 B（15 KB）** |
| `train-s64`（M=2,S=64,d=512） | 512 | 4 | 8 192+8 192+2 048 = 18 432 | 7 168+4 096 = 11 264 | 896 | **30 592 B（29.9 KB）** |
| `train-s128`（M=1,S=128,d=256） | 256 | 4 | 9 216 | 5 632 | 896 | **15 744 B（15.4 KB）** |
| `d=4096`（refusal 边界） | 4096 | — | 65 536+65 536+16 384 = 147 456 | 57 344+32 768 = 90 112 | 1 152 | **238 720 B（233 KB，94%）** |

> **构建期预算断言**：factory 计算 `4·DW·4 + 4·DW·4 + DW·4 + 7·DW·elem + 4·DW·elem + 4·(RC+3)·32` 并断言
> `footprint × 1.2 ≤ 253 952`，否则 raise（拒绝编译）。这使 `d=4096` 的「94% 占用」在膨胀后可能被拒绝——属预期行为，
> 并在 §9.1 R6 记录「`d ≥ 3072` 余量不足，若上层需要须降 `slot` 版本数或改 dtype 存 `slot`」。

### 4.6 动态轴定义

| 动态轴 | 声明方式 | 运行时范围 |
|---|---|---|
| `M` | **构建期常量**（`lru_cache` 按 `(M, seq_len, d, eps, dtype)` 特化） | manifest 用 1 / 2 / 3；新增取值 → 重新编译 |

**说明**：`op_requirements` 把 `M` 列为动态轴，但本算子的可执行语义是「按 `(M, seq_len, d, eps, dtype)` 特化编译」——
与源算子 `@target_cache(maxsize=32)` 的 `(M, seq_len, d, eps, dtype)` 特化**完全同策略**（§0.4 O10）。特化带来：
① `grid`、`RC`、`nchunks` 全是常量，无运行期边界判断；② DMA 描述符的段数/长度全为常量；③ 行循环上界静态（满足平台「循环上界静态」约束）。
host 侧任何 shape 变化 → 重新编译（与源一致），不存在「运行时 shape 未处理」的路径。若上层要求「一份 kernel 覆盖多 `M`」，
则需把 `M` 放进 `T.dynamic` 并把 `grid`/行循环改为运行期值——属结构变更，不在本设计范围（§9.1 R8）。

### 4.7 JIT 配置

```python
@functools.lru_cache(maxsize=64)          # 形状/ dtype 特化（源用 target_cache，同策略）
def build(M, seq_len, d, dtype, eps):     # 构建期常量全部来自闭包
    ...
    return tilelang.compile(main, target="ascend", out_idx=[6, 7, 8, 9, 10, 11])
```

`pass_configs = {}`（**不**开 `TL_ENABLE_FAST_MATH`，理由见 §3.5.1 与 `examples/mla_dense_bwd/_delta_kernel/DESIGN.md` §7.3）；
不额外传 `compile_flags`（ascend target 的后端默认处理 dtype）。

---

## 5. Tiling 策略

### 5.1 计算类型

**类型：纯 Vector（SIMD）**。判定依据：算子只含「逐元素乘加 + 行内 4 个归约 + 4-tap 窗口累加 + 逐元素超越函数」，
无矩阵乘、无跨核数据交换；Cube 单元不参与（§1.6.0 R4 候选 4 已淘汰）。

### 5.2 Block 划分

**Tiling 的自由变量是「每核多少连续行」（`RC`）**，不是二维 tile：本算子每行是独立的 `d` 维向量计算，`d` 由 chunk 流覆盖（不分块）。

```python
# 构建期规则（实现于 factory，非运行期）
target = max(1, ceil(M * seq_len / 56))          # 56 = 物理 Vector 核数（§5.5）
cands  = [dd for dd in divisors(seq_len) if dd >= target]
RC     = min(cands)                              # 构造上 RC | seq_len ⇒ 无尾块
if RC > 1 and M * (seq_len // RC) < 28:          # 该除数会把核数压到 28 以下
    RC = 1                                       # 退回最大并行（RC=1 恒整除）
nch   = seq_len // RC                            # 每条的 chunk 数（块级）
grid  = M * nch                                  # 启动核数
DW    = align_up(d, 64); nchunks = DW // 64      # 行内 chunk 数（向量维）
```

| workload | `M` | `seq_len` | `d` | dtype | `target` | `RC` | `nch` | **grid** | 每核行数 `RC+3` | `nchunks` |
|---|---|---|---|---|---|---|---|---|---|---|
| train-s32 | 1 | 32 | 256 | fp16 | 1 | **1** | 32 | **32** | 4 | 4 |
| train-s64 | 2 | 64 | 512 | fp16 | 3 | **4** | 16 | **32** | 7 | 8 |
| train-s128 | 1 | 128 | 256 | bf16 | 3 | **4** | 32 | **32** | 7 | 4 |
| train-s96 | 3 | 96 | 768 | bf16 | 6 | **6** | 16 | **48** | 9 | 12 |

**Tiling 语义**：`grid` 条任务 → 每条 `nch` 个 row-chunk → 每 chunk `RC` 个 owned 行 + 3 个 halo 行 → 每行 `nchunks`
个 64 元素向量段。三层全为构建期常量，运行期只有 `bx`（核号）一个变量。

### 5.3 约束分析

- **对齐约束**：行搬入/写出的 destination/source region 都是 `[:d]`（元素数相等，`d` 任意）；向量段 64 元素 = 1 个 fp32
  寄存器；`d ∈ {256,512,768}` 全部 `d % 64 == 0` → 尾 lane 0 浪费；`st` 行距 32 B（8 fp32），与 `PL-950-simd-reduce`
  的已验证布局一致。
- **UB 容量**：45.7 KB / 248 KB（§4.5，含 1.7× 膨胀上界仍余 69%）。
- **L0 容量**：不使用 Cube → 不适用。
- **DMA 段数**：每行 3 条搬入（1536 B 单段 @d=768）+ 2 条写出（1536 B）+ 每核 3 条参数搬入 + tile 末 4 条压缩写出
  （`RC×4 B`，源距 32 B）。无窄行（「行」宽度 = `d·elem` ≥ 512 B，远超 128 B 段粒度）。

### 5.4 注意事项

- **非整除（`d`）**：`DW = align_up(d, 64)`，尾 lane 由 `update_mask(min(64, d-ch*64), 32)` 屏蔽；分母恒为 `d`。
  4 条 workload 都是 `d % 64 == 0`（无屏蔽路径），`d=196`/`d=200` 走屏蔽路径（L0-3 与 Boundary 用例）。
- **非整除（`seq_len`）**：由「`RC` 整除 `seq_len`」的构建期规则**消除**，不存在尾 chunk 代码路径。
  `seq_len` 为质数且行数 > 56 时退回 `RC=1`（grid = `M·seq_len`，正确但波数增加，§5.5 已量化）。
- **`grid` 小于 56**：行数本身只有 32–288，`grid=32` 时波内只用到 32 核（57%）。这不是「分核不足」，而是
  「workload 行数 < 核数」的必然；`RC=1` 可提到 32/128/128/288 但会引入 `4×` 的 halo 冗余（§1.6.0 R3 候选 2 已量化），
  由 §1.6.3 的裁决计划在 Stage 4 按 shape 决定。
- **halo 的 `if r >= 0`**：`c == 0` 时 3 个 halo 行被跳过，`slot[0..2]` 靠**预清零**保证为零（不得依赖未初始化 UB）。

### 5.5 分核策略（物理核数适配）

> 三要素判定标准与实查要求的权威版本：`.agents/skills/_shared/standards/core-split-strategy.md`（本任务按 §1 三要素与 §2.1 执行）。

**① 逻辑核数计算**

- 逻辑任务数（row-chunk 数）= `M · ceil(seq_len / RC)`：`train-s32` → **32**；`train-s64` → **32**；`train-s128` → **32**；
  `train-s96` → **48**（每核 owned 行 1 / 4 / 4 / 6，另有 3 行 halo）。
- 启动核数 = `grid` = 同上（「一条 row-chunk 一个核」，无核内多任务串行）。
- 一般规则：`grid = M · (seq_len // RC) ≤ 56` 由 §5.2 的 `target = ceil(M·seq_len/56)` + 除数约束保证；
  极端情形（`seq_len` 质数、`M·seq_len > 56`）退回 `RC=1`，此时 `grid = M·seq_len` 可能 > 56（如 `seq_len=95, M=1`
  → 95 核 = 2 波），属「逻辑核超发 + 硬件波调度」，探针的 `prime-95` 几何已验证该形态可编译。

**② 物理核数依据（实查记录）**

| 查询 | 命令 / 来源 | 实际返回值 |
|---|---|---|
| 平台任务上下文（本机实跑） | `python3 .agents/tools/statectl.py platform --task-dir examples/engram_gate_conv/_engram_gate_conv_fwd_kernel` | `hardware.hardware_id = "ascend-950pr-9579"`；`observed_cores = {"cube": 28, "vector": 56}`；`core_source = "Ascend950PR_9579.ini [SoCInfo] ai_core_cnt=28, cube_core_cnt=28, vector_core_cnt=56; runtime probe rtGetAiCoreCount()=28 on Ascend950PR_9579, 2026-11-23"` |
| 型号配置文件（只读复核） | `grep -n "core_cnt" /usr/local/Ascend/cann-9.2.1/x86_64-linux/data/platform_config/Ascend950PR_9579.ini` | 第 20–22 行：`ai_core_cnt=28`、`cube_core_cnt=28`、`vector_core_cnt=56`；第 50/51/58 行：`l0_c_size=262144`、`l1_size=524288`、`ub_size=253952` |
| `torch.npu.get_device_properties` | **本次未执行**：本机仅 1 张卡且被同批次另一算子的 Stage 3 占用；设计期约定只做无卡编译/静态探针（`ASCEND_RT_VISIBLE_DEVICES=""`），直接初始化设备上下文会与并行的设备任务争用 | 以平台已记录的 runtime probe（上一行）为准，不猜测数值；**Stage 3 首次真机运行前须补 `torch.npu.get_device_properties` 并记录 `cube_core_num` / `vector_core_num`** |

**目标型号：`ascend-950pr-9579`**（dav-3510，28 AI Core / 56 Vector Core，UB 248 KB）。本算子为纯 Vector kernel，
故分核目标取 **56 个 Vector 核**（与同仓 `examples/moe_permute_align/_make_scatter_kernel/DESIGN.md` 的取值一致）。

**③ 规模判定与分核方案**

- **规模判定：逻辑核数 ≤ 物理核数 → 无需适配**。4 条 workload 的逻辑核数 32 / 32 / 32 / 48 全部 ≤ 56：
  一次 wave 内全部发射完毕，每核 1 条任务、无核内多任务串行、无超发调度开销，也不需要为对齐 56 而凑批。
- **分核方案**：`grid = M·(seq_len // RC)`，每核独占同一 `m` 内的连续 `[t0, t0+RC)` 行；核内顺序处理 `RC+3` 行
  （前 3 行为 halo，只算不写）。负载均衡：每核 owned 行数相同（`RC`），halo 行数相同（3），故**无 straggler**；
  `train-s96` 的 `grid=48` 与 56 的比值 0.86（一次 wave 内 48 核工作、8 核空闲），`train-s32/s64/s128` 为 32/56 = 0.57。
- **波内利用率的取舍（已量化，非疏漏）**：把 `RC` 降到 1 可把 `train-s64/s128` 提到 128 核（3 波）并让 `train-s96` 到 288 核（6 波），
  但 halo 冗余从 `3/4` 升到 `3/1`（gate 计算 ×4、`H/k/v` 读 ×4，§1.6.0 R3 候选 2 已量化 8.7 MB vs 3.11 MB）。
  本设计优先「少 halo + 少波次」，并把 `RC` 的取值列入 §1.6.3 的裁决计划（按 shape 分派）。
- **核内循环边界**：`RC+3`、`nchunks`、`4` 全为构建期常量，满足平台「循环上界静态」约束（`references/programming.md`）。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|---|---|---|---|
| 核网格 | block 级并行（一维） | `T.Kernel(grid)`，`grid = M·nch` | 每核独占一段连续行（§5.5） |
| 行方向（含 halo） | 核内串行 | `T.serial(RC + 3)` | 环形窗口的跨行依赖（`slot`），无法并行（§1.6.2 V10） |
| 块级分支：halo 是否存在 | 标量 `if` | `if r >= 0:`（包住 3 条 `T.copy` 与 gate 块） | 因果零边界（R4），每核 3 次判断 |
| 块级分支：是否 owned | 标量 `if` | `if i >= 3:`（包住 conv 块与 2 条写出） | 每核 3 次判断，避免 halo 行白算 conv |
| 行内 `d` 方向 | 向量化 chunk 循环 | `T.serial(nchunks)` + `T.simd.*` | 每段 64 元素 = 1 个 fp32 寄存器；跨迭代无向量值（累加器在 `alloc_local`，§1.6.2 V11） |
| conv tap | 静态展开 | `T.unroll(4)` | 4 个 tap 全部展开，免循环开销（`references/programming.md` 的 `T.unroll` 形态） |
| 参数预置 | 串行 | `T.serial(nchunks)` / `T.serial(4)` | 每核一次 |
| 元素级 | 不使用 `T.Parallel` | — | `SimdVF` 无线程域，元素并行由向量 lane 承担（§2.3） |

### 6.2 循环伪代码

见 §3.3（同一份伪代码即调度结构）：`T.Kernel(grid)` → 参数预置（2 个 `T.serial`）→ `T.serial(RC+3)` 行循环
（2 个块级 `if` + 每行 2~3 个 `T.serial(nchunks)`）→ tile 末 4 条压缩 `T.copy`。

### 6.3 流水线优化

- **本设计不使用 `T.Pipelined`**：行循环携带 `slot` 的真实依赖（行 `i` 写、`i+1..i+3` 读），流水排布会与
  AutoSchedule 的多版本假设冲突；且核内只有 4–9 行，装载（1536 B×3）与计算（~600 条指令）本身已在同一行内
  由编译器按 pipe 自动重叠（探针确认生成了 MTE2↔V 的成对 `asc_sync_*`，§7）。
- **`slot` 与行缓冲都按单版本**（`annotate_buffer_versions` 全 1）：避免编译器为「写后即读」的缓冲插入无用的双版本
  （UB 预算与地址计算都更简单）；它们在流水循环外的复用也会让 AutoSchedule 关闭多版本（`examples/mla_dense_bwd/_delta_kernel/DESIGN.md` §6.3 的探针 F4 同现象），显式置 1 更可预期。
- **Stage 4 可选手段**（不改结构）：① 把行循环改成 `T.Pipelined(RC+3, num_stages=2)` 并给 `h_ub/k_ub/v_ub` 置 2 版本
  （UB +4.6 KB@d=768）以重叠下一行的 DMA；② 加大 `nchunks` 的展开粒度（`T.unroll` 替代 `T.serial`）以消循环开销。
  两者都在 §9.1 R4 登记为「未实测收益」。

### 6.4 尾块处理

| 场景 | 处理 |
|---|---|
| `seq_len % RC == 0`（本任务全部 workload，由 §5.2 规则保证） | 无尾块：每核恰好 `RC` 个 owned 行 + 3 个 halo 行 |
| `seq_len` 无法被 `RC` 整除 | 不发生：`RC` 的候选集是 `seq_len` 的约数；`target` 无可用约数时退回 `RC=1`（恒整除） |
| `d % 64 != 0` | `DW = align_up(d,64)` + `update_mask(min(64, d-ch*64), 32)` 屏蔽尾 lane；分母恒 `d` |
| `c == 0`（序列首块） | 3 个 halo 行 `r < 0` 被 `if r >= 0` 跳过，`slot[0..2]` 预清零 ⇒ 等价于「零位置」 |
| `seq_len < 3`（如 `seq_len=1/2`） | `RC=1`、每核只有自己一行；`slot[0..2]` 全零 ⇒ conv 退化为「只有 p=3 的 tap」+ 零贡献，与 golden 的 `F.pad(3)` 一致 |
| `M·seq_len < 56`（如 `train-s32`） | `RC=1`，`grid=32`，波内只 32 核（§5.5 已量化取舍） |

---

## 7. 同步策略

### 7.1 同步模式

**模式：自动同步**（编译器 AutoSchedule + InsertSync 在 MTE2 / Vector / MTE3 / Scalar 之间插入 flag）。本设计**不写**任何
手动 `T.set_flag/wait_flag`、`ascend_pipe_barrier`、`ascend_get_buf`。

### 7.2 同步点说明（探针生成代码的实证）

探针 `train-s96` 的生成代码在 `grid` 内共出现 **16 处 `asc_sync_notify/asc_sync_wait`**（event 0/1），覆盖下列依赖：

| 位置 | 编译器生成（摘录） | 理由 |
|---|---|---|
| block 起始 | `asc_sync_notify(PIPE_MTE3, PIPE_V, 0)` / `notify(PIPE_V, PIPE_MTE2, 0)` | 初始化首个 tile 的 buffer 状态，避免首轮死等 |
| 参数搬入后 | `asc_sync_wait(PIPE_MTE2, PIPE_V, 1)` + `notify(PIPE_MTE2, PIPE_V, 1)` | 等 `wh_ub/wv_ub/cw_ub` 落地后才 `vld` |
| 行搬入后（每行） | `asc_sync_wait(PIPE_MTE2, PIPE_V, 0)` | 等 `H/k/v` 单行 DMA 落地才做 gate |
| 行计算后 | `asc_sync_notify(PIPE_V, PIPE_MTE2, 0)` | 释放 `h_ub/k_ub/v_ub` 给下一行搬入（单版本 ⇒ 下一行必须等本行算完） |
| 写出前（每 owned 行） | `asc_sync_notify(PIPE_V, PIPE_MTE3, 0)` + `wait(PIPE_V, PIPE_MTE3, 0)` | `vh_ub/y_ub` 单版本：下一行的 `vsts` 必须等本行 DMA 读完 |
| 标量写回前 | `asc_sync_wait(PIPE_MTE3, PIPE_V, 0)` | `st` 单版本：tile 末压缩 DMA 前等所有 `vsts` 完成 |
| Scalar pipe | `asc_sync_notify(PIPE_S, PIPE_V, 0)` / `wait(PIPE_S, PIPE_V, 0)` | `vsts(extent=1)` 的标量侧与向量侧配对 |
| block 结束 | 尾部若干 `asc_sync_wait` | 收敛未完成的跨 pipe 事件 |

**结论**：`T.copy`（MTE2/MTE3）与 `T.simd.*`（PIPE_V）之间的依赖**全部由编译器按缓冲版本自动加锁**，设计期无需手工同步；
`slot` 的跨行依赖是**同 pipe（Vector）内的顺序依赖**，不需要 flag（同一 pipe 天然有序）。

### 7.3 pass_configs 配置

```python
pass_configs = {}      # 不开 TL_ENABLE_FAST_MATH（理由见 §3.5.1）
compile_flags = []     # ascend target 默认后端已处理 dtype，不额外传 -O3/-DENABLE_BF16
```

**为什么关掉 fast-math**：本算子有 `vmul + vadd` 的密集链（conv 的 4 tap 累加、gate 的 `Σ`）。fast-math 允许把
`vmul+vadd` 收缩成 FMA 或做其它重排，会改变乘积的舍入位置，使 §1.6.1 A1/A2 的机器验证前提（乘积先舍入到 fp32 再累加）
失效。显式不开启即可保持验证过的语义。

---

## 8. 验证方案

### 8.1 Golden 函数

> **独立性声明**：golden 以 **§0.1 的源算子语义**为唯一依据，直接移植 `examples/TileOPs/tileops/workloads/sequence_modeling.py`
> 的 torch 参考（`_rmsnorm` + `engram_gate_conv_fwd_torch`），**不复刻** §0.6 的任何 NPU 结构（无 chunk、无环形窗口、无 halo、
> 无 `cwv` 折叠、无 `Σh·k·w²` 化简、无 `(4,RC,8)` 暂存）。golden 与设计之间的差异只有「实现手段」，数学式逐字对应 §1.3 的
> 原始形式（不是 A1–A4 化简后的形式）。

```python
import torch, torch.nn.functional as F

CONV_KERNEL_SIZE = 4

def _rmsnorm(x, w, eps=1e-6):
    """fp32 统计量 + fp32 归一化；返回 (normed, rrms)。与源 torch 参考逐字一致。"""
    x_f = x.float()
    rrms = (x_f ** 2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    return x_f * rrms * w.float(), rrms.squeeze(-1)

def golden_engram_gate_conv_fwd(H, k, v, rms_w_h, rms_w_v, conv_w, eps=1e-6):
    """Engram post-projection GateConv 前向（源语义，独立于 NPU 实现）。

    返回 (Y, vhat, alpha, rrms_h, rrms_k, rrms_v)：
      Y, vhat           : (M, T, d)，H.dtype
      alpha, rrms_*     : (M, T)，float32
    """
    M, T, d = H.shape
    h_norm, rrms_h = _rmsnorm(H, rms_w_h, eps)          # H 与 k 共用 rms_w_h
    k_norm, rrms_k = _rmsnorm(k, rms_w_h, eps)

    dot = (h_norm * k_norm).sum(dim=-1, keepdim=True)   # Σ_d h_norm·k_norm
    alpha = torch.sigmoid(dot / (d ** 0.5))

    v_hat = alpha * v.float()                           # fp32，未舍入
    # 注意：r_v 的统计量取「已舍入到 H.dtype」的 vhat（源语义的舍入点 ②）
    v_hat_norm, rrms_v = _rmsnorm(v_hat.to(H.dtype), rms_w_v, eps)

    v_perm = v_hat_norm.float().permute(0, 2, 1)        # (M, d, T)
    v_padded = F.pad(v_perm, (CONV_KERNEL_SIZE - 1, 0)) # 因果左补 3
    conv_w_expanded = conv_w.float().T.unsqueeze(1)     # depthwise (d, 1, 4)
    conv_out = F.conv1d(v_padded, conv_w_expanded, groups=d).permute(0, 2, 1)

    Y = F.silu(conv_out) + v_hat.float()                # 残差用未舍入 v_hat（舍入点 ⑤）

    return (Y.to(H.dtype), v_hat.to(H.dtype), alpha.squeeze(-1).float(),
            rrms_h.float(), rrms_k.float(), rrms_v.float())
```

对照 `examples/TileOPs/tileops/workloads/sequence_modeling.py` 的源参考：`_rmsnorm` / `engram_gate_conv_fwd_torch` **逐行等价**
（同 `mean(dim=-1)`、同 `add(eps).rsqrt()`、同 `F.pad(K-1, 0)` + `F.conv1d(groups=d)`、同 `v_hat.to(H.dtype)` 的舍入点），
因此与 `examples/TileOPs/tests/ops/test_engram.py` 的 `ref_program` 同值。**无 dilation**（§0.1 已确认）。

### 8.2 精度标准

| dtype（输入） | 输出 | atol | rtol | 依据 |
|---|---|---|---|---|
| float16 | `Y`, `vhat` | **1e-2** | **1e-2** | 设计自定严阈值；实测 A1–A4 全开的 `max abs(cand,base)` = 2.4e-4、`ulp` ≤ 1（§1.6.1） |
| bfloat16 | `Y`, `vhat` | **2e-2** | **2e-2** | 同上（bf16 尾数 8 bit，实测 max abs 3.8e-6） |
| float16 / bfloat16 | `alpha`, `rrms_h`, `rrms_k`, `rrms_v`（fp32） | **1e-3** | **1e-3** | 实测最大容差预算占用：`alpha` 0.1%、`rrms_h/k` 0%、`rrms_v` 24.9%（对 `1e-4` 口径），故 `1e-3` 有 40× 余量；`rrms_v` 的固有偏差来自舍入点 ②（§9.1 R5） |

**与 op 自身容差的关系**：`examples/TileOPs/tests/ops/test_engram.py` 对 6 个输出用 `atol=0.1, rtol=0.1`（fp16）/ `atol=0.2, rtol=0.1`（bf16），
Stage 5 以它为准；上表的**设计闸门更严**（`Y/vhat` 5–10×，fp32 标量 100×），用于在 Stage 3 就能拦住结构性错误
（如 `r_v` 误用未舍入统计量、残差误用舍入值），而不是等到 0.1 的宽松容差掩盖问题。

### 8.3 L0 门槛测试计划

| 层级 | 形状 | dtype | 覆盖的设计点 | 目的 |
|---|---|---|---|---|
| **L0-1** | `M=1, seq_len=32, d=256`（manifest smoke；`RC=1, grid=32, nchunks=4`） | fp16 **与** bf16 | 单行全链：gate 三统计量 → `alpha` → `v̂` 舍入往返 → `r_v` → `slot` → conv（`RC=1` 时 4 个 slot 全由本核 halo 填充）→ SiLU/残差 → 6 个输出 | 最小可运行；两 dtype 必须都过 |
| **L0-2** | `M=2, seq_len=64, d=512`（manifest `train-s64`；`RC=4, grid=32`） | fp16 | 多 batch 的 `(m, t0)` 映射 + `RC=4` 的窗口滚动（跨 4 行复用 `slot`）+ halo 跨 chunk 边界 | 窗口正确性（**错位一行即整段错**，必须有此闸门） |
| **L0-3** | `M=1, seq_len=8, d=196`（`d % 64 != 0 → DW=256`，`nchunks=4`） | bf16 | 尾 lane 屏蔽（`update_mask` 依次 64/64/64/4）+ `T.copy` 的 `[:d]` 非对齐宽度（392 B） | pad 路径的**唯一数值证据**（编译期只能证明描述符合法） |
| **L0-4** | `M=1, seq_len=1, d=64`；`M=1, seq_len=2, d=64` | fp16 | `seq_len < 3`：`slot[0..2]` 全零、conv 退化为单 tap；`grid=1`、`i >= 3` 分支从不为真的边界 | 最小边界（无越界读、无空循环） |
| **L0-5** | `M=3, seq_len=96, d=768`（manifest 最大；`RC=6, grid=48`） | bf16 | 12 chunk、`h = 3` 个 halo、单核 9 行的完整路径 | 最大几何（IR 规模 + 编译时间 + 数值） |
| L1 | manifest 其余两条（`train-s128` bf16、`train-s96` fp16 交叉） | fp16 / bf16 | dtype × 几何交叉 | 代表性小规模 |
| L2 | 4 条 manifest workload 原 dtype | — | 与 manifest 计分口径一致 | 流量/发射估算的实测对账（§1.6.0） |
| Boundary | `d=200`（任意非 128 倍数）；`seq_len=95`（质数 → `RC=1`，`grid=95`）；全零输入；含 `inf`/`NaN` 的行（`equal_nan=True`） | fp16 / bf16 | 尾 lane、`RC=1` 超发、零行、非有限传播 | 与源语义一致的边界 |

判定：`torch.testing.assert_close(kernel(H,k,v,w_h,w_v,cw), golden_engram_gate_conv_fwd(...), atol/rtol=§8.2, equal_nan=True)`，
6 个输出**逐个**比对（`Y`/`vhat` 用 8.2 的 dtype 阈值，4 个 fp32 标量用 1e-3）。**L0-1..L0-5 全过才允许进入 L1/L2**
（与 `tilelang-op-develop` 的分层契约一致）。

---

## 9. 风险点与注意事项

### 9.1 已知约束

| # | 风险 | 影响 | 依据 / 缓解 |
|---|---|---|---|
| R1 | **每 launch 固定开销未实测**（平台 `cost_measurements` 为空；设计假设 2–5 µs） | 单 launch（主选）相对两段 launch（备选）的收益全部落在这个假设上；若实际固定开销 < 0.3 µs，备选会因 13% 的发射优势胜出 | §1.6.3 裁决计划（阈值：备选 < 主选 × 0.9 才翻转，按 shape 分片） |
| R2 | **halo 重算的 3 行冗余**（`3/RC`） | `train-s32`（`RC=1`）的 gate 计算与 `H/k/v` 读取各 ×4 | §1.6.0 R3 已量化（3.11 MB vs 基线 4.66 MB 仍占优）；`RC=1` 只在行数 < 56 时出现，此时绝对量极小（0.33 MB） |
| R3 | 950PR 的**带宽与向量吞吐常数全靠 910B2C 外推**（`CONST-mte2-degradation`、`CONST-vector-launch-overhead`，均绑定 `tilelang 0.1.2+15ad002b3d`/910B2C） | roofline 估算可能偏乐观/悲观 2× | 已在 §1.6.0 的估算行如实标注「未实测外推」；Stage 4 用 `msprof` 的 Task Duration 对账 |
| R4 | 行循环未流水（单版本缓冲 ⇒ 下一行搬入等本行算完） | 每行有「搬入→算」的串行段（1536 B DMA + ~600 条向量指令）；吞吐可能低于估算 | §6.3 给出 Stage 4 手段（`T.Pipelined` + 行缓冲双版本，UB +4.6 KB）；不改结构 |
| R5 | `rrms_v` 对 golden 的相对偏差 ~2.5e-5（实测 224 ulp / 预算 24.9% @1e-4） | 若 Stage 3 用 fp32 的 `atol=rtol=1e-4` 口径比对 `rrms_v` 会「假失败」 | 该偏差**来自源语义**（`r_v` 定义在已舍入 `v̂` 上），baseline 对 fp64 的偏差同为 24.8%；§8.2 已把 fp32 标量阈值定为 1e-3 |
| R6 | `d ≥ 3072` 时 UB 余量不足（`d=4096` 裸占 233 KB，94%），1.2× 预算断言会拒绝编译 | 支持不了 `d=4096` 的极端几何 | §4.5 的构建期断言即为拒绝点（fail-fast）；若上层需要，可把 `slot` 改存 16-bit 或降到 2 个 slot（`d=4096` 时省 32 KB）——属结构变更，需回写设计 |
| R7 | `T.simd.vsts` 默认 `dist="NORM_B32"` 的整向量 fp32 落 UB 语义只有编译期证据（探针生成 `vsts_norm(…, full)`），无真机数值验证 | 若分布参数语义不符，`slot` 内容错位 → `Y` 全错 | L0-1/L0-2 是数值闸门；退路：改用 16-bit 存 `slot`（`PK_B32`/`UNPK_B16` 链，950DT 参考已验证）并接受 ~5e-4 的额外舍入（仍在 §8.2 容差内） |
| R8 | `M` 为构建期常量（形状特化） | host 侧 `M` 变化需重编译 | 与源 `target_cache` 同策略（§4.6）；若上层要求动态 `M`，需把 `grid` 与行循环改为运行期值（结构变更） |
| R9 | `grid < 56`（32/32/32/48）时波内利用率 57%–86% | 核级并行度未跑满 | §5.5 ③ 已量化取舍（降 `RC` 换并行 ⇒ halo ×4）；裁决计划按 shape 决定 |
| R10 | halo 重算「逐位一致」依赖「同一条构建期展开的指令序列」 | 若 Stage 3 把 halo 与 owned 行写成两份不同的代码（如不同的 chunk 顺序、不同的 `mask` 用法），可能与拥有核的结果不一致 | §0.6 R3 已写明约束；实现上 halo 与 owned 行**共用同一个 python 代码块**（只有 `if` 包裹不同），L0-2 的窗口用例可暴露不一致 |
| R11 | 平台未在 950PR 上实测 `vsts(extent=1)` + `ONEPT_B32` 的行距 32 B 契约（来自 `PL-950-simd-reduce` 的 910DT 实证） | 若 `vsts` 实际写 32 B 而 `st` 行距不足 32 B，会覆盖相邻行 | `st` 末维固定 8 个 fp32（32 B），与已验证布局一致；`train-s96` 生成代码的 `asc_copy_ub2gm_align(…, 6, 4, …, 4, 32)` 显示源距 32 B 已生效 |

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|---|---|---|---|
| **在 `T.SimdVF` 块内读 GM** | 直接 `T.simd.vld(WH[ch*64])` | 前端语义检查直接拒绝：`BufferLoad from global buffer 'WH' inside a SIMD_VF block is not allowed`（探针首轮实测） | 先把参数 `T.copy` 到 UB（`wh_ub/wv_ub/cw_ub`），`vld` 只读 UB（§3.3 约束①） |
| **用 `T.clear` 清 `slot`** | 12 KB fp32 缓冲 | 降级为 1536 次 `float2` 标量存（探针首轮生成代码实测），单核 ~2 k 周期 | 用向量 `vsts` 清零（3 个 halo slot × `nchunks` 次）；探针改后生成 `vsts_norm(zero_p, ...)` |
| **`vcvt` 的 `part` 传循环变量** | `for part in T.unroll(2): vcvt(..., part=part)` | `part` 是编译期参数，运行期值不可接受（构建期报错或语义错） | 只映射 64 元素/chunk（`part=0`），或用 python 元组 `for part in (0, 1)` 展开 |
| **向量累加器用 python 变量跨 IR 循环** | `acc = vadd(acc, …)` 在 `T.serial` 内 | `RuntimeError: Immutable variable 'acc' is used outside its defining region`（`examples/mla_dense_bwd/_delta_kernel/DESIGN.md` §9.2 的探针 F7） | 用 `T.simd.alloc_local((1,), "float32")`（本设计与 950DT 参考同形） |
| **halo 行写成零而非跳过** | `c == 0` 时也去算 `r < 0` | 越界读 GM（`r = -1`） | `if r >= 0` 包住 3 条 `T.copy` 与 gate 块；`slot` 预清零 |
| **`slot` 索引写成 `i % 4` 但 tap 用 `(i-3+p)` 且漏 `% 4`** | 4 tap 累加 | 越界读 `slot` 或读错行 → `Y` 全错 | tap 索引恒为 `(i - 3 + p) % 4`（与写入 `i % 4` 同一模） |
| **conv 残差用舍入后的 `v̂`** | `y += vh_ub[...]` | 与 golden 的舍入点 ⑤ 不符（golden 用未舍入 `alpha·v`） | 残差用寄存器里的 `alpha_v·v_f32`（§3.1 步骤 10） |
| **`r_v` 用未舍入 `v̂` 统计** | 省掉收窄往返 | 语义漂移（§1.6.1 R1 候选 6 已拒） | 必须先 `vcvt` 收窄再 `vcvt` 展宽，然后平方累加 |
| **`T.Kernel` 传二维或 `threads=`** | 照抄 GPU 源码 | 前端直接拒绝 | §0.5：一维 grid + `T.SimdVF` |
| **开 fast-math** | 复制源算子 `pass_configs` | FMA 收缩改变乘积舍入，A1/A2 验证前提失效 | §7.3 不设 `pass_configs` |
| **`RC` 不整除 `seq_len`** | 构建期规则漏判 | 最后一条 chunk 越界读 `r > seq_len-1` 且越界写 `Y/vhat`（编译器不报错） | §5.2 的约数规则 + 构建期断言 `seq_len % RC == 0` |

### 9.3 特殊场景处理

- **极小 shape**（`seq_len ≤ 2`、`d ≤ 64`）：`RC=1`、`grid = M·seq_len`，`slot[0..2]` 恒零，conv 只剩 `p=3` 的有效 tap；
  无特殊分支（L0-4 覆盖）。
- **`seq_len` 质数且行数 > 56**：退回 `RC=1`（`grid = M·seq_len`，可能 > 56 → 硬件波次调度）；探针 `prime-95` 已验证可编译。
- **`d` 非 64 倍数**：`DW = align_up(d, 64)`，尾 lane 屏蔽；`T.copy` 的 `[:d]` region 与 `DW` 不等（元素数相等，§3.5.3 的
  `references/programming.md` 条款）。
- **全零输入**：`H=k=v=0` → `sh=sk=0` → `r_h=r_k=eps^{-1/2}`；`dot=0` → `alpha=0.5`；`v̂=0`、`r_v=eps^{-1/2}`、
  conv=0、`Y=0`（`SiLU(0)=0`，残差 0）。golden 同值，L0 有零行用例。
- **非有限输入**：`inf/NaN` 与 golden 同样传播（`equal_nan=True` 比较）；`0·inf` 不产生额外 NaN 的前提是 `slot` 被**显式**清零
  （而非未初始化），见 §9.2。
- **fp16 vs bf16**：同一份代码路径，只有 `dtype` 常量不同（构建期特化）；`vcvt` 的 `round="ROUND_R"`、`sat=False` 对两者一致。

---

## 10. 交付清单

### 10.1 目录结构

```
examples/engram_gate_conv/_engram_gate_conv_fwd_kernel/
├── DESIGN.md                              # 本设计文档
├── verify_equiv.py                        # §1.6.1 的等价性机器验证（已实跑，EQUIV_PASS）
├── RETROSPECTIVE.md                       # Stage 1 复盘（Phase 8）
├── history_version/
│   └── design_probe_compile.py            # §1.6.3 / §3.3 的编译探针（6 几何 ALL_OK）
└── _engram_gate_conv_fwd_kernel.py        # Stage 3 产出（kernel + golden + 分层测试）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|---|---|---|
| `DESIGN.md` | ✅ 已完成 | 本文件 |
| `verify_equiv.py` | ✅ 已完成并执行 | A1–A4 的 fp64 参照对照，全部 EQUIV_PASS |
| `history_version/design_probe_compile.py` | ✅ 已完成并执行 | P1–P8 全部 COMPILE OK（train-s96/s64/s128/s32、prime-95、pad-d196） |
| `_engram_gate_conv_fwd_kernel.py` | ⬜ 待 Stage 3 | kernel（单 launch、§3.3 的形态）+ `golden_engram_gate_conv_fwd`（§8.1）+ L0/L1/L2/boundary |
| `RETROSPECTIVE.md` | ✅ 已完成 | Stage 1 复盘（Phase 8） |

### 10.3 命名规范

- 算子目录：`examples/engram_gate_conv/_engram_gate_conv_fwd_kernel`（conductor 指定，保持 scaffold 布局）
- 实现文件：`_engram_gate_conv_fwd_kernel.py`（与目录 / manifest 的 `kernel_map` 一致）
- kernel 工厂：`_engram_gate_conv_fwd_kernel(M, seq_len, d, eps, dtype)`（与源抽取件的签名一致，`lru_cache` 特化）
- golden：`golden_engram_gate_conv_fwd(H, k, v, rms_w_h, rms_w_v, conv_w, eps=1e-6)`

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md）+ 等价性验证 + 编译探针
2. ⬜ Golden（§8.1，先跑通 torch 参考比对）
3. ⬜ Kernel（参数预置 → 行流水 → 标量批量写回）+ L0-1（两 dtype）
4. ⬜ L0-2..L0-5 → L1 → L2 → Boundary
5. ⬜ 与 `examples/TileOPs/tests/ops/test_engram.py` 的 6 输出对接（Stage 5 前置）

---

## 11. 平台规则与技术约束汇总

| 平台规则（`references/programming.md` / `workflow.md`） | 本设计的落点 |
|---|---|
| API 命名空间 `tilelang.ascend.language`；`tilelang.compile(..., target="ascend", out_idx=…)` | §3.2 / §4.7 |
| `T.Kernel(N)` 一维、无 `threads` | §0.5 / §0.6 R1 / §5.5 |
| 容量预算取任务平台配置（UB 253952 / L1 524288 / L0C 262144） | §4.5（只用 UB） |
| 物理核数实查并记录（Cube/Vector 分别） | §5.5 ② |
| 同步优先自动（AutoSchedule + InsertSync），跨核 flag 未认证 | §7（全自动，无跨核） |
| `annotate_buffer_versions` 的 `{buf: 1}` 语义 | §4.3 / §6.3 |
| 性能采集 `msopprof` + Task Duration，A/B 交错 | 交接给 Stage 4（§1.6.3 裁决计划已给代表 shape 与阈值） |
| 迁移 kernel 按本平台 API 实现，manifest/wrapper/test 保持原布局 | §0.7 / §10.3 |
