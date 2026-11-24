# _engram_gate_conv_bwd_kernel 算子设计文档

> 任务：TileOPs 迁移（harness 子模式），源算子 `/workspace/TileOPs/src/tileops/_kernels/sequence_modeling/engram/gate_conv_bwd.py`
> （GPU TileLang，`tilelang.language`，4 次 launch：清零 + 三段计算）。
> 目标平台：`ascend-950pr-9579`（dav-3510，CANN 9.2.1，TileLang ascend 0.1.15，28 Cube / 56 Vector，
> UB 248 KiB / L1 512 KiB / L0C 256 KiB），编程模式按平台 `programming_modes.migration_default = simd` 评估后
> 选 **SIMD**（§2）。本函数的 6 个输出对应 `EngramGateConvBwdOp` 的全部输出。
> 参考实现（只读）`/workspace/TileOPs-ascend950/src/tileops/kernels/ascend/sequence_modeling/gate_conv_bwd.py`
> 是同一算子在 950DT 上的 staged SIMD 实现；本设计**采纳其若干已验证构造**（§0.5 逐项标注来源与差异），
> 但结构与它不同（融合 vs staged，§1.6.0 R3），不照抄。
> 本文件的算法调研（§1.6.0）与优化结论（§1.6.1–§1.6.3）是 §3.1 公式拆解与 §6 循环结构的唯一算法输入。

## 0. 源算子解读与迁移分析

> 方法论：`references/migration-analysis.md`（三问解读 → 硬件耦合性四态判定 → NPU 重设计）。
> §0.1–§0.4 只记录源算子事实，§0.5–§0.6 是迁移决策，§0.7 记录 golden 来源。本章结论驱动 §1–§10。

### 0.1 源算子语义（做什么）

给 `dY`（输出梯度）与前向保存的 `vhat`（**前向已舍入的**低精度张量）、`alpha`、`rrms_h/rrms_k/rrms_v`
（fp32）以及前向输入 `H/k/v/rms_w_h/rms_w_v/conv_w`，反传整条 Engram GateConv 融合链：

前向（`examples/TileOPs/tileops/workloads/sequence_modeling.py` 的 `engram_gate_conv_fwd_torch`，语义基准）：

```
h_norm = H_f · rrms_h · w_h            rrms_h = (mean(H_f²) + eps)^(-1/2)
k_norm = k_f · rrms_k · w_h            rrms_k = (mean(k_f²) + eps)^(-1/2)
dot    = Σ_j h_norm·k_norm             alpha = sigmoid(dot / sqrt(d))
v_hat  = alpha · v_f                   vhat = round_dtype(v_hat)          （保存的是舍入后的 vhat）
z      = vhat_f · rrms_v · w_v         rrms_v = (mean(vhat_f²) + eps)^(-1/2)    （用舍入后的 vhat）
conv   = Σ_{p<4} conv_w[p] · z[t-3+p]   （因果宽度 4 深度可分离卷积，t-3+p < 0 处零填充）
Y      = silu(conv) + v_hat
```

反向按链式法则（源码逐行核对）：

```
d_silu[t,j]     = dY[t,j] · (s + conv·s·(1-s)),  s = sigmoid(conv[t,j])          ← 前向 conv 需重算
dconv_w[p,j]    = Σ_t d_silu[t,j] · z[t-3+p,j]
d_vnorm[t,j]    = Σ_p conv_w[p,j] · d_silu[t+3-p,j]                              ← 卷积转置
drms_w_v[j]     = Σ_t d_vnorm[t,j] · vhat[t,j] · rrms_v[t]
dot_v[t]        = Σ_j vhat[t,j]·w_v[j]·d_vnorm[t,j]
dvhat[t,j]      = rrms_v·w_v[j]·d_vnorm[t,j] − rrms_v³·vhat[t,j]·dot_v[t]/d + dY[t,j]
dalpha[t]       = Σ_j dvhat[t,j]·v[t,j]          dv[t,j] = alpha[t]·dvhat[t,j]
ddot[t]         = dalpha·alpha·(1-alpha)/sqrt(d)
dh_norm[t,j]    = ddot·k_norm[t,j]               dk_norm[t,j] = ddot·h_norm[t,j]
dot_h[t]        = Σ_j H[t,j]·w_h[j]·dh_norm[t,j]
dH[t,j]         = rrms_h·w_h[j]·dh_norm[t,j] − rrms_h³·H[t,j]·dot_h[t]/d
dot_k[t]        = Σ_j k[t,j]·w_h[j]·dk_norm[t,j]
dk[t,j]         = rrms_k·w_h[j]·dk_norm[t,j] − rrms_k³·k[t,j]·dot_k[t]/d
drms_w_h[j]     = Σ_t [ ddot·k_norm[t,j]·H[t,j]·rrms_h + ddot·h_norm[t,j]·k[t,j]·rrms_k ]
```

| 语义要素 | 结论 | 依据 |
|---|---|---|
| 规约轴 | 行内 `j ∈ [0,d)` 的行点积共 4 处（`dot_v / dalpha / dot_h / dot_k`），每 token 独立；跨 token 的 3 个权重梯度沿 token 求和 | 源码 `gate_conv_bwd.py` 各 `T.reduce_sum(dot_2d, dot_sum, dim=1)` 与 `T.atomic_add` |
| 累加顺序 | 行点积：块内 `T.reduce_sum` 的树形顺序（源硬件相关）；**权重梯度：`T.atomic_add`，跨 CTA 顺序不确定**（浮点加法不满足结合律 → 无逐位可复现性契约） | 源码 `atomic_add(dconv_w…)`、`atomic_add(drms_w_h…)`、`atomic_add(drms_w_v…)` |
| dtype 语义 | 输入/输出 `float16 \| bfloat16`；`alpha/rrms_*` 与三个权重梯度 fp32；**全部中间计算 fp32**（`accum_dtype = "float"`），`dH/dk/dv` 写回时一次 cast | 源码 `accum_dtype`、`dH[bid,tid,j] = T.cast(dh_local[j], dtype)` |
| 保存态的舍入语义 | `z` 用**前向保存的已舍入 vhat** 计算（不是重算的 `alpha·v`）；`conv` 因而按「舍入后 vhat 的 RMSNorm」重算 | 源码 `T.cast(vhat[bid, src_t, j], accum_dtype)`；workload 注释「the forward's *rounded* vhat, not a recomputed one」 |
| 边界语义 | 因果卷积：`src_t = t-3+p < 0` 时该 tap 贡献 0（batch 内独立，**不跨 batch**）；`seq_len < 4` 时只有部分 tap 有效；`d` 任意正数（源侧 host 补到 256 再裁回） | 源码 `T.if_then_else(src_t >= 0, …)`、`_run_len` 的 clamp、`d_padded = align_up(d, 256)` |
| 空/极值 | 无空张量语义（`M,S,d ≥ 1`）；`sigmoid` 溢出时 `1/(1+e^{-x})` 自然饱和到 0/1，源实现无 nan 特判；`conv_w` 全 0 时梯度为 0 | 源码无特殊分支；`verify_equiv.py` 的 zeros/spike/nonfinite 模式覆盖 |
| eps | 反向工厂**不接收 eps**（`_engram_gate_conv_bwd_kernel(M, seq_len, d, dtype, run_len)`），eps 已固化进保存的 `rrms_*` | 源码签名；任务 `op_requirements` 亦注明 |
| host 侧语义 | `forward()` 只做 `F.pad(d, → d_padded=align_up(d,256))` 与输出 `[:, :, :d]` 裁剪 + `.contiguous()` 归一 | 源码 `EngramGateConvBwdKernel.forward` |

**语义保持基线**：§8.1 的 golden 以本节为唯一依据（移植源仓参考实现 `ref_engram_gate_conv_bwd`，fp32 autograd），
不复刻 §0.6 的 NPU 算法。

**小结（第一问 What）**：语义 = 「用 `dY` 与前向保存态反传 gate(RMSNorm) → `v_hat` → RMSNorm(v) →
causal width-4 深度可分离卷积 → SiLU + 残差 这整条链」；
4 处行点积沿 `d`、3 个权重梯度沿 token 求和；**中间计算全 fp32、输出一次舍入到输入 dtype**；
权重梯度在源实现中顺序不定（原子加），本设计将其确定化；前向卷积需按「保存的已舍入 vhat」重算；
因果零填充按 batch 内 `t-3+p < 0` 判定，反向工厂不含 eps。无「待确认」项（逐条已对源码核对）。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|---|---|---|---|
| `dY` | `[M, seq_len, d]`（kernel 内 `d_padded`） | float16 / bfloat16 | 输出梯度；尾轴连续 |
| `H`, `k`, `v` | `[M, seq_len, d]` | same as `dY` | 前向输入（`k`、`v` 在 workload 里按 0.1 缩放） |
| `rms_w_h`, `rms_w_v` | `[d]` | same as `dY` | 两个 RMSNorm 权重（**`H` 与 `k` 共用 `rms_w_h`**） |
| `conv_w` | `[4, d]` | same as `dY` | 因果深度可分离卷积的 4 个 tap |
| `vhat` | `[M, seq_len, d]` | same as `dY` | 前向**已舍入**的 `alpha·v` |
| `alpha`, `rrms_h`, `rrms_k`, `rrms_v` | `[M, seq_len]` | float32 | 前向保存的 gate / 三个 rsqrt 标量 |
| `dH`, `dk`, `dv`（输出） | `[M, seq_len, d]` | same as `dY` | 三个输入梯度（`source_output_shape` 一致） |
| `drms_w_h`, `drms_w_v`（输出） | `[d]` | float32 | 两个 RMSNorm 权重梯度 |
| `dconv_w`（输出） | `[4, d]` | float32 | 卷积权重梯度 |

约束：`d > 0`、`d ≤ 4096`（manifest `shape_rules`，源侧因 host padding 与整行 fragment 驻留而声明）；
`M, seq_len ≥ 1`；kernel 内另有 fp32 中间张量 `dvhat_buf`、`dvhat_out`（`[M, seq_len, d_padded]`），
不属于算子契约。测试容差（`_tol_bwd`，见 `examples/TileOPs/tests/ops/test_engram.py`）：
fp16 `atol=rtol=0.2`，bf16 `atol=0.3 / rtol=0.2`，**对每个输出分别施加**。

### 0.3 实现算法解读（怎么算）

**入口与 host 侧**：`_engram_gate_conv_bwd_kernel(M, seq_len, d, dtype, run_len)` 返回 `_func(threads)`，
`@target_cache(maxsize=32)` 按 shape 缓存；`@tilelang.jit(out_idx=[12..19], compile_flags=["-O3","-DENABLE_BF16"])`；
默认 `threads=128`，autotune 枚举 128/256/512。`run_len = _run_len(tokens, sm_count)`：目标 runs `= ceil(tokens/8)`，上限 `4·sm_count/KS`
（`KS=4` → 等于 `sm_count`），`run_len = ceil(tokens/runs)`。

**计算步骤分解**（源码每个计算语句均归入步骤）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式 |
|---|---|---|---|---|
| 0 | launch#1：`drms_w_h/drms_w_v` 清零；`token_runs>1` 时 `dconv_w` 清零 | — | 3 个 fp32 输出 | 原子累加的初值 |
| 1 | launch#2，grid=(seq_len, M)，每 block 一个 token：搬入 `dY`；用 `vhat/rrms_v/conv_w/rms_w_v` **重算 `conv_out`**（4 tap 顺序累加，`src_t<0` 取 0） | `dY, vhat, rrms_v, conv_w, rms_w_v` | `conv_out` fragment | `conv[t,j]` |
| 2 | `sig = 1/(1+exp(-conv))`；`d_silu = dY·(sig + conv·sig·(1-sig))` | `conv_out, dY` | `dvhat_buf[t]`（GM） | `d_silu` |
| 3 | launch#3，grid=(4, token_runs)：每个 (tap, run) 一个 CTA，沿 run 内 token 串行累加 `d_silu[t]·(vhat[t-3+p]·rrms_v[t-3+p]·w_v)` | `dvhat_buf, vhat, rrms_v, rms_w_v` | `dconv_local` → `atomic_add(dconv_w[p])` 或直接写 | `dconv_w[p,j]` |
| 4 | launch#4，grid=(seq_len, M)：卷积转置 `d_vnorm = Σ_p conv_w[p]·d_silu[t+3-p]`（`dst_t` 越界取 0） | `dvhat_buf, conv_w` | `d_vnorm` | `d_vnorm` |
| 5 | `atomic_add(drms_w_v[j], d_vnorm·vhat·rrms_v)` | `d_vnorm, vhat, rrms_v` | `drms_w_v` | `drms_w_v` |
| 6 | `dot_v = Σ_j vhat·w_v·d_vnorm`（`T.reduce_sum`）；`dvhat = rrms_v·w_v·d_vnorm − rrms_v³·vhat·dot_v/d`；`+= dY`（残差旁路） | 同上 | `dvhat_out[t]`（GM） | `dvhat` |
| 7 | launch#5，grid=(seq_len, M)：`dalpha = Σ_j dvhat·v`；`dv = alpha·dvhat` | `dvhat_out, v, alpha` | `dv` GM | `dalpha`, `dv` |
| 8 | `ddot = dalpha·alpha·(1-alpha)/sqrt(d)`；`h_norm = H·rrms_h·w_h`，`k_norm = k·rrms_k·w_h` | 标量 + `H,k` | — | `ddot`, 两个 norm |
| 9 | `dot_h = Σ_j H·w_h·ddot·k_norm`；`dH = rrms_h·w_h·ddot·k_norm − rrms_h³·H·dot_h/d` | | `dH` GM | `dH` |
| 10 | `atomic_add(drms_w_h, ddot·k_norm·H·rrms_h)` | | `drms_w_h` | `drms_w_h` 第一项 |
| 11 | `dot_k = Σ_j k·w_h·ddot·h_norm`；`dk = rrms_k·w_h·ddot·h_norm − rrms_k³·k·dot_k/d` | | `dk` GM | `dk` |
| 12 | `atomic_add(drms_w_h, ddot·h_norm·k·rrms_k)` | | `drms_w_h` | `drms_w_h` 第二项 |
| 13 | 输出 cast：`dH/dk/dv = T.cast(·, dtype)` | | 3 个 dtype 输出 | dtype 语义 |

**数据流与内存访问模式**（源硬件视角）：

```
GM[dY,H,k,v,vhat,rms_w_*,conv_w,rms标量]
   │ T.Parallel 线程级逐元素搬入 fragment（整行 d_padded 驻留寄存器/共享）
   ▼
fragment: conv_out/d_silu ──T.copy──> GM[dvhat_buf]  (launch#2 尾)
   │
   ├─ launch#3: GM[dvhat_buf] + GM[vhat,rrms_v] ──fragment 累加──> atomic_add GM[dconv_w]
   └─ launch#4: GM[dvhat_buf] + GM[conv_w] ──fragment──> d_vnorm ──> GM[dvhat_out]
                                                          └─ atomic_add GM[drms_w_v]
launch#5: GM[dvhat_out,v,H,k,alpha,rrms_*] ──fragment 行归约──> GM[dH,dk,dv] + atomic_add GM[drms_w_h]
```

**循环与并行结构**：两级——`T.Kernel(seq_len, M)` / `T.Kernel(4, token_runs)` 的 grid，与 block 内
`threads=128` 的 `T.Parallel(d_padded)` 线程级逐元素并行；`for p in T.serial(KS)` 为 tap 顺序累加，
`for i in T.serial(run_len)` 为 dconv_w 归约的 token 顺序累加。无显式流水、无双缓冲。

**host 侧逻辑**：`F.pad` 把 5 个 `[M,S,d]` 输入、2 个 `[d]` 权重、`[4,d]` 卷积权重从 `d` 补到
`align_up(d,256)`；输出 6 个张量再 `[:, :, :d]` / `[:d]` / `[:, :d]` 裁回。

**小结（第二问 How）**：实现算法 = 「4 次 launch（清零 + stage-1 重算 conv/d_silu → stage-1a dconv 部分和 →
stage-2 卷积转置 + RMSNorm(vhat) 反向 → stage-3 gate + 两个 RMSNorm 反向），每 token 一个 CTA、
整行 `d_padded` 驻留 fragment，跨 CTA 用原子累加 3 个权重梯度」；中间量 `dvhat_buf`/`dvhat_out`
两度经 GM；13 个计算步骤与源码逐语句对应（无遗漏、无臆造步骤），数据流图覆盖全部 buffer。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|---|---|---|---|---|
| 1 | 三段式 staged 分解（launch#2 / #4 / #5）+ `dvhat_buf`、`dvhat_out` 落 GM | `d_silu` 被 4 个 tap 复用、`dvhat` 被两个后续步骤复用，避免重复计算 | 中间张量写回 GM 再读 | 大容量 GM + 低 kernel 启动开销 | 中（GM 往返的代价假设是 GPU 特有的） |
| 2 | 浮点原子累加（`T.atomic_add`）写 `dconv_w`/`drms_w_h`/`drms_w_v` | 跨 CTA 累加 3 个权重梯度，免跨核同步 | GM 原子加 | GPU 全局原子单元 | **强** |
| 3 | 清零 launch#0 + `set` 语义 | 为原子累加准备初值 | 独立 launch | kernel 启动便宜 | 中 |
| 4 | `d_padded = align_up(d, 256)` host padding + crop | 尾轴对齐、消除 kernel 内尾块分支 | host `F.pad` 全量拷贝 | 256B 对齐的向量化宽度 | 中（对齐宽度依平台） |
| 5 | `run_len` run 切分（目标 `ceil(tokens/8)`，上限 `4·sm_count/KS`） | 把 `dconv_w` 的原子加摊到多个 run，降低争用 | 按 SM 数决定 CTAs 数 | **`sm_count`（GPU SM 数量）** | **强** |
| 6 | 每 token 一个 CTA，整行驻留 fragment（`d_padded` 个 fp32） | 行点积在片上完成，不落 GM | `T.alloc_fragment` + `T.Parallel` | 大寄存器堆 / 128 线程协作 | **强** |
| 7 | `conv_out` 重算而非保存 | 前向未保存 `conv_out`，反向只有 `vhat/rrms_v` 可用 | 4 tap 顺序累加 | 无（算法层约束） | 可移植 |
| 8 | 行点积 `T.reduce_sum(dot_2d, dot_sum, dim=1)` | 行内归约 | 块内归约原语 | GPU 块内树形归约 | 强 |
| 9 | `T.if_then_else(src_t >= 0, …)` 零填充 | 因果卷积边界，免分支发散 | 逐元素选择 | 无（纯表达） | 可移植 |
| 10 | fp16/bf16 输入 + fp32 累加（`accum_dtype`） | 精度：行点积与权重梯度全 fp32 | 类型提升 | 无（数值设计） | 可移植 |
| 11 | 一个 `T.macro` 内顺序启动 4 段 `T.Kernel` | 一次 JIT 产出全部 pass，共享 `dvhat_buf` 生命周期 | 多 launch 宏 | 无 | 可移植 |
| 12 | `threads ∈ {128,256,512}` autotune | 线程级并行度调优 | 编译期参数 | 线程束宽度 | 强 |

> 识别不出来 ≠ 不存在：以上 12 项来自源码逐语句核对（`gate_conv_bwd.py` 全文 + `_run_len` 注释），
> 未列出的优化在迁移中会被静默丢弃。

**小结（第三问 Why fast）**：源实现的加速来自三处——(i) **GM 中间量复用**（`d_silu` 供 4 个 tap、
`dvhat` 供后续两段，1 次写 2 次读），(ii) **原子免同步**（3 个权重梯度直接原子累加，含专门的
`run_len` 切分压争用），(iii) **行驻留 + 线程级并行**（每 token 一个 CTA，整行在片上完成 4 个行点积，
host 侧把 `d` 补到 256 消除尾块）。其中 (ii) 的 `sm_count`/原子组合与 (iii) 的 CTA/线程模型是**硬性强相关**，
(i) 的代价假设（GM 往返便宜）在 56 核 NPU 上不成立；四项处置见 §0.5。

### 0.5 硬件耦合性分析与 NPU 适配决策

**判定问题**：实现算法与优化手段是硬件强相关吗？能用在 NPU 上吗？

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|---|---|---|---|---|---|
| 计算语义（§0.1 全部公式） | 语义 | 无 | — | **保留** | 语义层无条件保留；§8.1 golden 以此为准 |
| fp32 中间 + dtype 输出（#10） | 语义 | 无 | — | **保留** | `T.simd.vcvt(…, dtype, round="ROUND_R", sat=False)`，`.agents/platforms/toolchains/ascend/knowledge/numerical-boundaries.md` 的转换条目已核验特殊值/舍入 |
| 因果零填充（#9） | 语义 | 无 | — | **保留**（表达改为均匀标量分支 + 零槽） | §0.6 R5；标量 `if` 分支在 SIMD 帧内外均可用（探针已核验） |
| `conv_out` 重算（#7） | 算法 | 无 | — | **保留** | 契约只提供 `vhat/rrms_v`（§0.1），重算不可避免 |
| 行点积（#8 `T.reduce_sum` 二维 shared→shared） | 优化 | 块内树形归约 | 有（显式 SIMD 寄存器归约） | **等价替换** | `T.simd.vadd` 链 + `vcadd` + `vdupv`；依据 `knowledge/numerical-boundaries.md` 的 `PL-950-simd-reduce`（FP32 (2,64)/(2,63) 通过），**高层二维 `T.reduce_sum` 有已登记的差异（`TRAP-950-simd-shared-reduce` / `CG-950-simd-shared-reduce`），故不采用** |
| 每 token 一 CTA + 整行 fragment（#6） | 优化 | CUDA 线程块 + 大寄存器堆 | 有（一维核网格 + UB 整行驻留） | **等价替换** | `T.Kernel(grid)`（一维，`threads=` 被实现拒绝，见 `references/programming.md`）+ `T.alloc_shared` + `T.SimdVF`；一核处理一个 token-chunk |
| `T.Parallel(d_padded)` 线程级逐元素（#6 的计算形态） | 优化 | CUDA 线程束 | 有（SIMD 寄存器级 64 lane） | **等价替换** | `T.simd.*` 逐 64 lane 向量指令；fp32 向量宽度 64（`PL-950-vector-execution`） |
| staged 分解 + 中间张量落 GM（#1） | 优化 | GM 往返便宜 | 有（GM 可用）但**不必** | **重新设计** | §0.6 R2：融合为单计算 kernel，`d_silu`/`vhat_norm`/`dY` 用 4 槽 UB 滚动窗口，跨 token 只多算 3 行 halo |
| 浮点原子累加（#2） | 优化 | GPU 全局原子 | 部分（`set_atomic("add","float32")` 仅对 **Cube 写回** 路径核验，见 `PL-950-atomic-splitk`；UB→GM 的 `T.atomic_add` 在本平台未核验，且**顺序不确定**） | **重新设计** | §0.6 R1：每核 UB 内 fp32 累积 + 第二个 launch 的有序归约（确定性） |
| 清零 launch#0（#3） | 优化 | 原子需要初值 | 不需要 | **舍弃** | 归约 kernel 直接写最终值，无原子 → 无清零需求；依据 §0.6 R1 |
| host padding `align_up(d,256)`（#4） | 优化 | 256B 向量对齐 | 有（设备侧掩码） | **重新设计** | §0.6 R6：`dw = ceil(d/64)·64` 设备侧掩码 + 只写 `[:d]`，去掉 host 全量 `F.pad` 拷贝 |
| `run_len` / `sm_count` 切分（#5） | 优化 | **GPU SM 数量** | 无（不同核模型） | **重新设计** | §0.6 R4：按 token 数与 **56 个 Vector 核**重导 chunk 几何（§5.5），不读 `sm_count` |
| autotune `threads`（#12） | 优化 | 线程束 | 无（SIMD 帧无 threads） | **舍弃** | 本平台 kernel 级 `threads=` 被实现拒绝；调优维度改为 chunk 长度与窗口策略（Stage 4） |

**迁移来源与 950DT 参考的适配说明（任务要求显式记录）**：除源算子外，本设计**采纳**了
`/workspace/TileOPs-ascend950/src/tileops/kernels/ascend/sequence_modeling/gate_conv_bwd.py`（950DT 已验证 SIMD 实现）的
以下构造，并逐项给出 950PR 适配差异与等价性：

| 采纳的构造 | 950DT 做法 | 本设计用法与差异 | 等价性/依据 |
|---|---|---|---|
| SIMD 叶子形态（`vld` UNPK_B16 → `vcvt` fp32 → `vsts`，dtype 写回用 `PK_B32` + `ROUND_R`） | 每行 chunk 循环 64 lane | 同形态；chunk 数由 `dw/64` 决定 | 数值路径相同（fp32 计算 + 一次舍入写回），与源算子 `T.cast` 一致；编译已核验（§1.6.0 探针） |
| `_zero_simd`（uint32 视图 + `vsts` 清零，含短于一个 VReg 的缓冲） | 清零 halo slot / 表尾 | 用于清零 6 个 per-core partial 与表尾 lane | 纯清零，无语义差异 |
| 有序归约 kernel（6 组 × 列 tile，逐行 fp32 向量加） | 归约 `rows` 行（`row_block` 分块） | 归约 **G ≤ 56 行**（per-core partial），无需 `row_block` 外层循环 | 数学上同为「按固定行序求和」，仅分组粒度不同（float 加法结合律差异在 §1.6.1 容差论证内） |
| 「保存态用前向舍入后的 vhat」 | 同 | 同 | 与 §0.1 语义一致，是本设计刻意继承的正确性点 |
| `refusal` 的 d 上限（`_MAX_HIDDEN_DIM = 4096`，42 B/channel × 4096 ≈ 168 KiB） | staged 结构，单行驻留 | 本设计窗口更多（**134 B/channel** + `16·seq`），d 上限 **1024**；大 d 走 §1.6.4 备选（52–76 B/channel） | 说明：950PR 与 DT 容量相同（UB 248 KiB），差异来自结构而非容量；风险见 §9 |

### 0.6 NPU 算法重设计

对 §0.5 判为「重新设计」的条目逐项给出 NPU 新算法（每项含语义保持论证）。

**重设计项 R1：权重梯度归约确定化（原子 → per-core 部分和 + 有序归约）**

- **源方案**：每个 (tap, run) CTA 把部分和 `atomic_add` 到 `dconv_w[p]`；`drms_w_h/drms_w_v` 由每个
  token CTA 直接原子加。目标是免跨核同步、吃满原子带宽；代价是**顺序不确定**（末位随调度变化）。
- **NPU 新算法**：每个计算核在自己的 UB 里维护 6 个 fp32 累加器（`pc[4][dw]`、`ph[dw]`、`pv[dw]`），
  按**本核 token 升序**逐 token 累加（token 内固定顺序：先 `h` 项后 `k` 项）；核结束时各写一行到
  GM workspace `PART[6, G, dw]`（G = 逻辑核数 ≤ 56）；第二个 launch 的归约 kernel 对每个
  `(量, 列块)` 按 **核号升序** 求和并直接写最终输出（无原子、无需清零）。
- **语义保持论证**：`dconv_w[p,j] = Σ_t …`、`drms_w[j] = Σ_t …` 的数学语义是「对 token 集合求和」，
  加法结合律保证任意分组顺序给出同一实数；实现上 fp32 累加的相对误差上界为 `n·u`（`u = 2^-24`），
  本算子 `n ≤ 2406`（`drms_w_h` 每 token 两项）→ 最坏 `1.4e-4` 相对，典型 `sqrt(n)·u ≈ 3e-6`；
  测试容差 fp16 `0.2/0.2`、bf16 `0.3/0.2`，余量 ≥ 3 个数量级（§8.2）。为**确定性**付出的代价是
  多一次 launch 与 6·G·dw·4 B 的 workspace 流量（§1.6.0 R3 已计入）。边界：`t-3+p < 0` 的 tap 贡献
  为 0（§0.1），由 `nrm` 窗口的零值 slot 自动满足（R5）。
- **验证**：`verify_equiv.py` 覆盖 `dconv_w/drms_w_*` 三个 fp32 输出（7 几何 × 6 模式 × 2 dtype），
  且 §8.3 L0 计划含「同一输入跑两次逐位相同」的确定性用例。

**重设计项 R2：三段 staged → 单计算 kernel + 滚动窗口（DS 不落 GM）**

- **源方案**：`d_silu` 写 `dvhat_buf`(GM)、`dvhat` 写 `dvhat_out`(GM)，3 个计算 launch + 1 个清零 launch。
  目的是让 `d_silu` 被 4 个 tap 复用、`dvhat` 被 gate 段复用，避免重算；代价是 fp32 中间往返
  （§1.6.0 R3 计得 26 B/lane/token：DS 12 + 原子 14）以及 4 次 launch。
- **NPU 新算法**：一个计算 kernel：每核负责 batch 内一段连续 token（长度 `RC`，§5.2）。迭代
  `i = 0 … RC+5`：在第 `i` 步 (a) 载入 position `pn = t0-3+i` 的 `vhat`/`dY`，算出 `nrm = (vhat·rrms_v)·w_v`
  与 fp32 `dY` 存入 **4 槽滚动窗口**（槽号 `floormod(pn,4)`）；(b) `i ≥ 3` 时用窗口 `nrm`（4 个 tap）
  重算 `conv`、`d_silu` 存入 4 槽 `ds` 窗口；(c) `i ≥ 6` 时对 token `tp = t0+i-6` 用 `ds` 窗口
  `ds[tp..tp+3]` 做卷积转置与整条 value/gate 反向。因为串行推进，`d_silu` 每行**只算一次**，
  跨核只多算 3 行 halo（尾部 3 行），跨 token 不再经 GM。权重梯度按 R1 累积。
- **语义保持论证**：滚动窗口只是「同一元素的生命周期管理」——`nrm`/`ds`/`dY` 的每个元素仍由同一公式、
  同一 fp32 精度算出，并被同一组消费方读取；窗口深度 4 由依赖距离证明充分：`ds[t]` 只被
  `t' ∈ [t, t+3]` 消费（卷积转置的 4 个 tap），`nrm[p]` 只被 `p' ∈ [p, p+3]` 消费（前向卷积），
  而 `nrm[p]` 在迭代 `i = p-t0+3` 写入、最后一次被消费于 `i = p-t0+6`（DS 行 `p+3`），
  跨度恰为 4 个位置 → 槽位 `floormod(·,4)` 不冲突（同一槽的旧值恰在其最后一次消费之后被覆盖）。
  边界：`pn < 0` 或 `pn ≥ seq` 时**不载入、不计算**，只把窗口槽写 0（`if pn >= 0: if pn < seq: … / if pn >= seq: …`
  与独立的 `if pn < 0: …` 互补分支），等价于源算子的 `T.if_then_else(…, 0)`（§0.6 R5）；
  `i` 循环跨越 `seq` 尾部时 `ds` 越界槽同样写 0。三个权重梯度的求和顺序变化见 R1 论证。
- **验证**：§8.3 L0 覆盖 `seq < 4`（`(2,1,256)`）、单 chunk（`(1,8,256)`）、非整除/素数 `seq`
  （`(3,401,256)`）、`d` 非 64 倍数（`(1,16,100)`）。

**重设计项 R3：GPU 线程级并行 → 950PR SIMD 向量化**

- **源方案**：`threads=128` 的 `T.Parallel(d_padded)` 把逐元素工作分给线程，`T.reduce_sum` 做块内树形归约。
- **NPU 新算法**：一维核网格（一核一个 token-chunk），核内 `T.SimdVF()` 帧中用 64 lane fp32 向量指令
  逐 chunk 计算；行点积用 `vadd` 链（升序 chunk）+ `vcadd` + `vdupv` 广播；dtype 写回用
  `vcvt(…, "ROUND_R")` + `vsts(dist="PK_B32")`。**不使用**高层二维 `T.reduce_sum`（已登记差异）。
- **语义保持论证**：向量化只是同一逐元素表达式的数据并行重排；行点积的求和顺序由「chunk 升序 +
  `vcadd` 的固定 lane 树」确定（比源算子更确定），相对误差 `≈ sqrt(d)·u ≈ 7e-7`（d=768），
  远小于容差。掩码 `update_mask(min(64, d-64·ch))` 保证 `j ≥ d` 的 lane 不参与任何行点积（
  `MODE_ZEROING`），与源算子 host padding 后「补 0 不影响和」等价。
- **验证**：编译期已核验（§1.6.0 的 D-3 探针，7 几何全通过）；数值路径待 §8.3 L0。

**重设计项 R4：分核几何（`sm_count` → 56 Vector 核 + token 数重导）**

- **源方案**：`run_len` 由 `sm_count` 与固定 `target_run=8`、`ctas_per_sm=4` 决定，只用于 `dconv_w` 的
  run 切分；计算 kernel 的 grid 是 `(seq_len, M)`（每 token 一块），完全不看核数。
- **NPU 新算法**：计算 kernel 的 grid 由 `nch = min(seq_len, max(1, 56 // M))`、`RC = ceil(seq_len/nch)`、
  `nch = ceil(seq_len/RC)`、`grid = M·nch ≤ 56` 决定（§5.2/§5.5），**不出现 `sm_count`**：每核占一个
  AIV，逻辑核数按构造不超过物理核数，无需波次调度。归约 kernel grid = `6·tiles ≤ 54`。
- **语义保持论证**：分核只改变「哪些 token 由哪个核计算」，每个 token 的公式与求和顺序（核内 token
  升序）不变；权重梯度的跨核顺序由 R1 的归约 kernel 固定。边界：`RC` 不整除 `seq_len` 时最后一块用
  `pds < seq_len` 的均匀标量 guard（源算子 `token_live` 的同一思路）。
- **验证**：§5.5 给出 8 条 shape 的几何表；探针覆盖素数 `seq_len` 与 `seq_len=1`。

**重设计项 R5：边界零填充的表达（`T.if_then_else` → 均匀标量分支 + 零槽）**

- **源方案**：每个 tap 用 `T.if_then_else(src_t >= 0, 载入, 0)`（CUDA 上的逐元素选择）。
- **NPU 新算法**：跨度依赖的边界（`pn < 0`、`pn ≥ seq_len`、`pds ≥ seq_len`）由**整行均匀的标量分支**
  处理：in-batch 时搬入并计算窗口槽；out-of-batch 时**不搬入、不计算**，只用 SIMD 把该槽写 0
  （与源算子「越界行不读取」完全同构）。为避免 traced 前端不支持标量 `else`，实现用**互补的
  两个 `if`**（`if pn >= 0: if pn < seq: … / if pn >= seq: …`，以及独立的 `if pn < 0: …`），
  探针已核验该形态可编译（7 条几何）。
- **语义保持论证**：out-of-batch 时窗口槽恒为 0，消费方的 `conv` / 卷积转置对应 tap 贡献恒为 0，
  与 `T.if_then_else(…, 0)` 逐位一致；因为**越界行从不被载入**，`inf/nan` 输入不会经
  `0·inf` 污染（源算子在该处也不读该行）。行内 `j ≥ d` 的尾 lane 由 `update_mask` 排除（`MODE_ZEROING`），
  保证行点积只累加有效 `j`。
- **验证**：`verify_equiv.py` 的 `nonfinite` / `zeros` 模式；探针（`history_version/design_probe_compile.py`）
  已核验分支形态、`floormod` 槽号与掩码写零在所有目标几何上可编译。

**重设计项 R6：host padding → 设备侧掩码**

- **源方案**：host `F.pad` 把 8 个张量补到 `align_up(d,256)`，输出裁回；代价是 8 次全量拷贝。
- **NPU 新算法**：UB 缓冲宽度 `dw = ceil(d/64)·64`，`T.copy` 只搬 `[:d]` 区域，SIMD 用
  `update_mask(min(64, d-64·ch))` 掩掉尾 lane，输出只写 `[:d]`；表尾 lane 显式清零。
- **语义保持论证**：掩码 lane 不参与任何行点积（零乘零），`/d` 用的是真实 `d` 而非 `dw`（源算子
  同样用 `float(d)`），因此结果与 host padding 版本逐位一致（掩掉的部分对和贡献恰为 0）。
- **验证**：探针覆盖 `d=192`；L0 覆盖 `(1,16,100)`（`d` 非 64 倍数）。

### 0.7 标杆实现

- **golden 来源**：`examples/TileOPs/tileops/workloads/sequence_modeling.py` 的
  `ref_engram_gate_conv_bwd`（fp32 autograd，`torch.autograd.set_multithreading_enabled(False)` 便于
  benchmark）。它**不使用**保存的 `vhat/alpha/rrms_*`，而是对 fp32 前向重新求导——天然独立于
  任何 NPU 实现（§8.1）。
- 源仓测试基准：`examples/TileOPs/tests/ops/test_engram.py`（Bwd 18 例，含 `(1,9,256)`、
  `(1,8,256)`、`(2,1,256)`、`(3,401,256)`、非连续输入、`dY` dtype 非法等拒绝用例）。
- 性能基准：`examples/TileOPs/benchmarks/ops/bench_engram.py`。
- 抽取的源 kernel 副本（只读参考）：`examples/TileOPs/tileops/kernels/sequence_modeling/engram_gate_conv_bwd/_engram_gate_conv_bwd_kernels.py`。

---

## 1. 概述

### 1.1 算子名称

`_engram_gate_conv_bwd_kernel`（Engram GateConv 全反向融合，`EngramGateConvBwdOp` 的 kernel）。

### 1.2 功能描述

给定输出梯度 `dY` 与前向保存态（`vhat/alpha/rrms_h/rrms_k/rrms_v`）+ 前向输入（`H/k/v/rms_w_h/rms_w_v/conv_w`），
一次算出 6 个梯度：`dH/dk/dv`（输入 dtype）、`drms_w_h/drms_w_v`（fp32）、`dconv_w`（fp32）。

### 1.3 数学公式

见 §0.1 的完整链式法则。核心子式（供 §3.1 拆解）：

$$
\begin{aligned}
& s=\sigma(\mathrm{conv}),\quad \mathrm{conv}[t,j]=\sum_{p<4} \mathrm{conv\_w}[p,j]\cdot \hat z[t-3+p,j],\quad \hat z=\mathrm{vhat}\cdot \mathrm{rrms\_v}\cdot w_v\\
& d_{silu}[t,j]=dY[t,j]\cdot\big(s+\mathrm{conv}\cdot s\,(1-s)\big),\qquad
  d_{conv\_w}[p,j]=\sum_t d_{silu}[t,j]\,\hat z[t-3+p,j]\\
& g[t,j]=\sum_{p<4}\mathrm{conv\_w}[p,j]\,d_{silu}[t+3-p,j],\qquad
  \dot v[t]=\sum_j \mathrm{vhat}[t,j]\,w_v[j]\,g[t,j]\\
& \hat d[t,j]=\mathrm{rrms\_v}[t]\,w_v[j]\,g[t,j]-\mathrm{rrms\_v}[t]^3\,\mathrm{vhat}[t,j]\,\dot v[t]/d+dY[t,j]\\
& dv[t,j]=\alpha[t]\hat d[t,j],\quad \dot\alpha[t]=\sum_j \hat d[t,j]v[t,j],\quad
  \dot\delta[t]=\dot\alpha\,\alpha(1-\alpha)/\sqrt d\\
& u_h[t,j]=\dot\delta\,k_{n}[t,j],\quad u_k[t,j]=\dot\delta\,h_{n}[t,j],\quad
  h_n=H\,\mathrm{rrms\_h}\,w_h,\; k_n=k\,\mathrm{rrms\_k}\,w_h\\
& dH[t,j]=\mathrm{rrms\_h}\,w_h[j]\,u_h-\mathrm{rrms\_h}^3 H[t,j]\Big(\textstyle\sum_j H w_h u_h\Big)/d\\
& dk[t,j]=\mathrm{rrms\_k}\,w_h[j]\,u_k-\mathrm{rrms\_k}^3 k[t,j]\Big(\textstyle\sum_j k\,w_h\,u_k\Big)/d\\
& d_{rms\_w\_h}[j]=\sum_t\big(u_h H\,\mathrm{rrms\_h}+u_k\,k\,\mathrm{rrms\_k}\big),\qquad
  d_{rms\_w\_v}[j]=\sum_t g[t,j]\,\mathrm{vhat}[t,j]\,\mathrm{rrms\_v}[t]
\end{aligned}
$$

### 1.4 算法描述

**本节描述迁移决策后的 NPU 算法**（与源算子的 staged + 原子结构不同，差异来源逐条标注）。

一次算子调用 = **2 次 launch**（源算子为 4 次：清零 + 3 段计算）：

1. **计算 kernel（launch#1，grid = 逻辑核数 ≤ 56）**：一核负责 batch 内一段连续 token（长度 `RC`，§5.2）。
   沿 `i = 0 … RC+5` 推进，每步做三件事（对应 §0.6 R2）：
   - (a) 载入 position `pn = t0-3+i` 的 `vhat`、`dY`，算出 `nrm = (vhat·rrms_v)·w_v`（fp32）与 fp32 `dY`，
     存入 4 槽滚动窗口（槽号 `floormod(pn,4)`）——**这是源算子 stage-1a 里重复计算的量，这里一行只算一次**（§1.6.1 O5）；
   - (b) `i ≥ 3` 时用窗口里 4 个 tap 的 `nrm` 重算 `conv`、`silu'`，得 `d_{silu}` 存入 4 槽 `ds` 窗口，
     并（仅当该行属于本核）累加 `dconv_w` 的 per-core 部分和（§0.6 R1）；
   - (c) `i ≥ 6` 时对 token `tp = t0+i-6` 用 `ds` 窗口的 `ds[tp..tp+3]` 做卷积转置 → `d_vnorm`，
     再做 value 段（`dot_v`→`dvhat`→`dv`→`dalpha`→`ddot`）与 gate 段（`h_n/k_n`→`dot_h/dot_k`→`dH/dk`），
     顺带累加 `drms_w_v`/`drms_w_h` 的 per-core 部分和，输出 `dH/dk/dv`。
   结构差异（vs 源算子）：`d_silu` 与 `dvhat` **不落 GM**（源为 `dvhat_buf`/`dvhat_out` 两次 GM 往返，
   §0.6 R2）；无原子（源为 3 处 `T.atomic_add`，§0.6 R1）；无 host padding（源为 `align_up(d,256)`，§0.6 R6）；
   分核不读 `sm_count`（§0.6 R4）。
2. **归约 kernel（launch#2，grid = 6·tiles ≤ 54）**：把 `PART[6, G, dw]` 按 **核号升序** 求和，
   直接写出 `drms_w_h`、`drms_w_v`、`dconv_w[0..3]`（无原子、无需清零，§0.6 R1）。

`d_silu`/`nrm`/`dY` 的每个元素仍然只由同一公式、同一 fp32 精度产生（§0.6 R2 的语义保持论证）；
只有跨 token 权重梯度的**求和分组**从「原子乱序」变成「核内 token 升序 + 跨核核号升序」（§1.6.1 容差论证）。

### 1.5 数据流图

```
GM[dY, vhat]  ──row copy──> UB[单行暂存] ──vcvt──> UB[nrm 窗口 4×dw fp32]
GM[dY]        ──row copy──> UB[单行暂存] ──vcvt──> UB[dyw 窗口 4×dw fp32]
GM[conv_w,rms_w_v,rms_w_h] ──一次──> UB[fp32 表 cwf/wvf/whf]        (每核一次)
GM[alpha,rrms_*] ──每批一次──> UB[标量条带 4×seq fp32]              (每核一次)
        │
        ├─(b) conv 重算 + silu' → UB[ds 窗口 4×dw fp32] ──> pc[4×dw] (per-core dconv 部分和)
        │
        └─(c) 卷积转置 → gn → dot_v → dvhat(gz) → dv → dalpha → ddot
                     ├──> UB[gz] ──> GM[dv]（dtype）
                     ├──> pv[dw]（per-core drms_w_v 部分和）
                     └──> gate: h_n/k_n → dot_h/dot_k → GM[dH],GM[dk]（dtype）, ph[dw]（drms_w_h 部分和）
        │
        └── kernel 尾：pc/ph/pv ──> GM[PART 6×G×dw fp32]

launch#2: GM[PART] ──每 (量,列块) 按核号升序求和──> GM[drms_w_h, drms_w_v, dconv_w]
```

### 1.6 算法调研与优化分析

#### 1.6.0 算法调研（Algorithm Research）

> 调研对象：同一数学语义（§0.1 反向链）的**算法族**——包括源算子的 staged 结构在内的 4 类结构。
> 深度：本算子为「多步 + 融合 + 3 处跨 token 规约」类，按 `references/algorithm-research.md` §2 属**完整调研**。
> 本地信息源已覆盖（参考表 `knowledge/algorithm-candidates.md`、pattern-library 条目、
> `examples/` 同类实现、源算子本身），无需联网检索。

**R1 等价化简公式候选**（含基线；正式等价论证与收益量化在 §1.6.1）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 纳入 R3 |
|---|---|---|---|---|---|
| 0 | 基线：源算子 staged 三段 + 原子 | `d_silu`/`dvhat` 落 GM，权重梯度原子累加，行点积 `T.reduce_sum` | — | — | ✅（基线） |
| 1 | 权重梯度「per-core 部分和 + 有序归约」 | `Σ_t` 改为「核内 token 升序 + 跨核核号升序」两级求和 | 数学恒等（加法结合律）；fp32 分组变化见容差论证 | 去掉 3 处原子（12 B/lane 的 drms 原子加 + run 切分）与清零 launch；常驻中间量由 `O(tokens·dp)` 降为 `O(G·dw)`（G≤56） | ✅ |
| 2 | 融合 + 4 槽滚动窗口（`d_silu`/`nrm`/`dY` 不落 GM） | 单核沿 token 串行，窗口深度 4 覆盖依赖距离 | 数学恒等（同一元素同一公式，仅生命周期管理） | fp32 中间往返 26 B/lane/token → 仅 PART 的 8 B/lane；launch 4→2 | ✅ |
| 3 | `nrm`（vhat_norm）物化复用 | 每行算一次 `(vhat·rrms_v)·w_v`，供 4 个 tap 与 dconv 复用 | 数学恒等（源算子逐 tap 重算同一乘积，逐位相同） | 省 4×(1 cvt+2 mul) 与 4×(1 cvt+2 mul)（stage-1a）| ✅（O5） |
| 4 | 倒数乘替代除法 | `/d`、`/sqrt(d)` → 预计算常量乘 | 容差内等价（各差 ≤1 舍入） | 4 处向量除法/元素/位置 → 乘法 | ✅（O1） |
| 5 | RMSNorm 修正系数折叠 | `rrms³·z·dot/d` → `((rrms³·(1/d))·z)·dot` | 容差内等价（重结合 + 倒数） | 4 mul + 1 div → 2 mul（向量侧） | ✅（O2） |
| 6 | value/gate 链公共子式 | `z·g` 同时供行点积与 `drms_w_v`；`u_h=ddot·k_n` 同时供 `dot_h`/`dH`/`drms_w_h` | 容差内等价（结合律重排） | 每 lane 每 token 省 4–6 乘 | ✅（O3/O4） |
| 7 | 每 tap 重算 `d_silu`（放弃窗口） | 不存 `ds` 窗口，逐 tap 重算 `conv`+`silu'` | 数学恒等 | 省 UB（4·dw·4 B），但 stage-1 计算 ×4 | ✅（对照 C3） |
| 8 | `sigmoid` 改写为 `s·(1+conv·(1-s))` | 代数恒等重排 | 容差内等价 | 无 fma 时操作数相同（4 vs 4） | ❌ 无收益，不采纳（见 §1.6.1 末尾） |
| 9 | `exp` → `exp2` 降代换 | `e^{-x}` → `2^{-x·log2e}` | 容差内等价 | 需 `vexp2`：本平台 `simd` 导出只有 `vexp`/`vexpdif`（`tilelang.ascend.language.simd` 实测导出表），**API 不存在** | ❌ 不采纳（API 缺失） |

**R2 在线算法**：**有**，且是选定结构的基础。结构判据：全部 4 个行点积都是 running 统计量
（和），权重梯度是沿 token 的 running 和 → 可单遍流式。具体在线化 = 「按 token 顺序推进、只保留
依赖距离内的窗口」（候选 2/3）与「per-core running 部分和」（候选 1）。收益口径：数据扫描遍数由
源算子的 3 遍（写 DS、4 tap 读 DS、stage-1a 再读 DS）降为 1 遍；fp32 中间往返由 26 B/lane/token
降为 PART 的 8 B/lane（`O(G·dw)` 常驻）加 `O(dw)` 级 UB 窗口。**反向已记住前向 `alpha/rrms_*`，
不存在需要重扫的全局量**（这是 R2 成立的直接证据）。

**R3 复杂度对比**（四口径；代表 workload = manifest train-s96：M=3, seq=96, d=768, tokens=288, dw=768）：

口径说明：①「元素操作数/lane/token」= 每 lane 每 token 的向量元素操作数，`mul/add/sub/div/exp/cvt`
各计 1，行点积的 `vcadd`/`vdupv`（每 token 每 64 lane 一次）不计；②「GM 字节/lane/token」按
fp16 输入输出 2 B/lane、fp32 中间 4 B/lane；③「扫描遍数」= fp32 梯度平面的 GM 往返次数；
④「中间缓冲」= 需常驻 GM 的 fp32 中间量。平台 manifest 另有 `flops` 公式（`45d+10`/位置 + 卷积项），
其计数法比本表粗（不含 cvt/超越函数），故只用于源侧基线口径，跨候选比较以本表为准。

| 候选 | FLOPs 口径：元素操作数/lane/token | 总量（288×768 lane） | 访存 GM 字节/lane/token | 总流量 | 扫描遍数 | 中间缓冲（常驻 GM） | 可并行度 / 跨核代价 |
|---|---|---|---|---|---|---|---|
| C0 基线（源算子：4 launch、原子、DS 落 GM） | 117 | 25.9 M | 50 | 11.1 MB | 3（DS 写 + 4 tap 读 + stage-1a 读） | `dvhat_buf/dvhat_out` 常驻 8 B/lane | tokens 并行；**需 3 处原子**（顺序不定） |
| C1 staged 确定化（3 launch、DS 落 GM、per-core 部分和） | 97 | 21.5 M | 48 | 10.6 MB | 3 | DS 4 B/lane + PART 8 B/lane | tokens 并行；跨核归约 1 次（有序） |
| **C2 融合 + 滚动窗口 + per-core 部分和（主选）** | **84** | **18.6 M** | **29** | **6.4 MB** | **1** | 仅 PART 8 B/lane | tokens 并行（G≤56 单波）；跨核归约 1 次（有序） |
| C3 融合但逐 tap 重算 `d_silu`（无窗口） | 145 | 32.1 M | 29 | 6.4 MB | 1 | 同 C2 | 同 C2；UB 少 12 KiB（d=768），但 stage-1 计算 ×4 |

**R4 硬件亲和性评估**（逐候选；负向淘汰附证据）：

| 候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---|---|---|---|---|---|---|
| C0 | 逐元素段匹配 SIMD；但 3 处原子需 GPU 全局原子单元，本平台 UB→GM 原子未核验（`PL-950-atomic-splitk` 仅覆盖 Cube 写回 + `set_atomic`） | 无需 UB 窗口，最小 | 需 host pad 到 256 | 静态 | 4 launch | ❌ 原子路径无等价证据 + 顺序不定（语义/确定性风险） |
| C1 | 全 SIMD，匹配 | DS `tokens·dp·4` 在 GM；UB ≈ 76 B/channel（省 2 个窗口，仍留 `nrm` 窗口与 `d_silu` 单行） | 同 C2 | 静态 | 3 launch | ⚠️ 可行但流量 1.7 倍于 C2（48 vs 29 B/lane，见 R3）且多 1 次 launch；作为备选（§1.6.4） |
| **C2** | 全 SIMD：4 个行点积 = `vadd`+`vcadd`（`PL-950-simd-reduce` 已验证 FP32 行归约）；逐元素 = `vld/vcvt/vmul/vadd/vdiv/vexp/vsts` | UB 104,448 B / 253,952 B（41%），无 L1/L0C 需求 | `dw = ceil(d/64)·64` + 掩码；无需 host pad | 全部静态（`RC`、`nchunks`、`G` 编译期常量）；边界用标量均匀 guard | 单 kernel 内 3 段串行 + 窗口单版本；无跨核 flag（`CG-950-batch-barrier` 未认证 → 用第 2 个 launch 代替） | ✅ **选定** |
| C3 | 同 C2 | UB 少 12 KiB（d=768） | 同 C2 | 同 C2 | 同 C2 | ❌ 计算量 ×1.7 换 12 KiB UB，d≤1024 时无必要 |

**设计期 roofline 估算（D-2，代表 workload train-s96，主选 C2）**

- **流量项**：6.41 MB（读 `dY/H/k/v/vhat` 10 B/lane × (1+3/RC) = 15 B + 写 `dH/dk/dv` 6 B +
  PART 写读 8 B；× 288×768 lane = 29 B/lane）。
- **发射项**：2 次 launch（清零 launch 已被 §0.6 R1 消除）。
- **容量项**：UB 峰值 104,448 B / 253,952 B（41%，见 §4.5）；L1 与 L0C 不使用 → 不构成约束。
- **估算下界**：`待核验`（缺 950PR 的 GM 有效带宽、单 AIV 向量吞吐与单次 launch 开销实测常数——
  平台 `cost_measurements` 为空，`knowledge/bottlenecks.md` 明确「当前没有 950 真机验证的瓶颈模式」）。
  以**未实测假设** GM 带宽 1.2 TB/s 折算流量项 ≈ 5.3 µs；若单次 launch 3–5 µs，则发射项 6–10 µs
  与流量项同量级 → 量级参考 **10 µs 级**（标注：未实测假设，仅供 Stage 4 对账起点，不作判定依据）。

**调研结论**：选定 **C2（融合单计算 kernel + 4 槽滚动窗口 + per-core 部分和 + 有序归约，2 次 launch）**
作为主选算法族。关键依据：R3 中 C2 的 GM 流量（29 B/lane）是基线 C0（50 B/lane）的 58%、
C1（48 B/lane）的 60%，元素操作数同时最低（84 vs 117/97，即 −28%/−13%），扫描遍数 1 vs 3；
R4 中 C0 的原子路径在本平台无等价证据且顺序不定，
C3 用 1.7× 计算换 12 KiB UB 无必要。与基线的结构差异一句话：**把「三段 staged + 原子累加」换成
「单遍 token 串行 + 4 槽 UB 滚动窗口 + 两级有序归约」**。

调研范围（用于支撑「无更优候选」的边界）：`knowledge/algorithm-candidates.md`（reduction/elementwise/
normalization 三族）、`knowledge/operator-patterns.md`（PL-950-normalization：带权 RMSNorm）、
`knowledge/PL-950-row-primitives.md`（行归约/广播）、`knowledge/numerical-boundaries.md`
（`PL-950-simd-reduce`、`TRAP-950-simd-shared-reduce`）、`knowledge/execution-patterns.md`
（`PL-950-atomic-splitk`、`PL-950-mixed-epilogue`）、`examples/` 同类实现（§3.5.3）、
以及本算子自身的结构分析（依赖距离 3 → 窗口深度 4 为最小）。**源算子 12 项优化手段（§0.4）的意图承接**：
#1 staged → C2 用 UB 窗口承接（意图=避免重算，C2 每行只算一次）；#2 原子 → R1 有序归约承接（意图=跨核累加，
顺序变为确定）；#3 清零 launch → 不再需要；#4 host pad → R6 设备掩码承接；#5 `sm_count` 切分 → R4 按
56 Vector 核重导；#6 每 token 一 CTA 整行驻留 → C2 每核一 chunk 整行驻留；#7 `conv` 重算 → 保留；
#8 `T.reduce_sum` → 显式 `vcadd` 归约；#9 边界零填充 → R5；#10 fp32 中间 → 保留；#11 多 launch 宏 →
2 次 launch；#12 autotune threads → 舍弃（本平台 kernel 级 `threads=` 被拒绝）。

#### 1.6.1 数学等价优化（公式级）

> 分析对象 = §1.6.0 选定的 C2 算法。R1 表中初判采纳的 5 项（候选 4/5/6/3）在此完成四要素论证；
> 每项都用 `verify_equiv.py`（torch CPU，fp64 参照，7 几何 × 6 模式 × 2 dtype，共 252 个字段对照）机器验证。

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算（每 lane 每 token） |
|---|---|---|---|---|---|
| O1 | 倒数乘替代除法 | `dot/d`、`dot_h/d`、`dot_k/d`、`dalpha·α(1-α)/sqrt(d)` | `dot·(1/d)`、`dot_h·(1/d)`、`dot_k·(1/d)`、`dalpha·α(1-α)·(1/sqrt d)`（`1/d`、`1/sqrt d` 编译期常量） | 容差内等价：`x·fl(1/d)` vs `fl(x/d)` 差 ≤1 个 fp32 舍入；fp16/bf16 输出再舍入后差为 0（见验证表） | 4 处向量除法 → 乘法（`vdiv` 单位代价远高于 `vmul`，倍率未有实测常数，见 §9） |
| O2 | RMSNorm 修正系数折叠 | `dot_v = Σ vhat·w_v·g`；`corr = (rrms³·vhat·dot_v)/d`；`dH`/`dk` 修正项同形 | `coef = rrms³·(1/d)`（每 token 一次标量乘）；`corr = (coef·vhat)·dot_v` | 容差内等价：倒数+重结合，误差 ≤2 个 fp32 舍入 | 每处 4 mul + 1 div → 2 mul（3 处共省 6 mul + 3 div） |
| O3 | value 链公共子式 | `dot_v = Σ(vhat·w_v)·g`；`drms_w_v = Σ(g·vhat)·rrms_v`；`direct = (rrms_v·w_v)·g` | `zd = vhat·g`（一次）；`dot_v = Σ zd·w_v`；`drms_w_v = Σ zd·rrms_v`；`direct = rrms_v·(w_v·g)` | 容差内等价：乘法结合律重排（fp32 非结合，差 ≤1 舍入；机器验证见下） | 省 3 mul/lane/token |
| O4 | gate 链公共子式 | `dot_h = Σ(H·w_h)·ddot·k_n`；`dH_dir = (rrms_h·w_h)·ddot·k_n`；`drms_w_h += ddot·k_n·H·rrms_h`（源算子三次重复乘积链） | `u_h = ddot·k_n`；`wuh = w_h·u_h`；`dot_h = Σ H·wuh`；`dH_dir = rrms_h·wuh`；`drms_w_h += (u_h·H)·rrms_h` | 容差内等价：同上 | 9 mul → 6 mul（h/k 两路各省 3） |
| O5 | `vhat_norm` 物化复用 | 源算子在 stage-1 与 stage-1a 各按 tap 重算 `(vhat·rrms_v)·w_v`（8 次/lane/token） | 每行算一次存 4 槽窗口，4 个 tap 与 `dconv_w` 共用 | **数学恒等且逐位相同**（同一表达式同一操作数，机器验证差为 0） | 省 7×(1 cvt + 2 mul) ≈ 21 元素操作 |
| — | `sigmoid` 改写 `s·(1+conv·(1-s))`（R1 候选 8） | `s + conv·s·(1-s)` | `s·(1+conv·(1-s))` | 代数恒等 | **无收益**：无 fma 时两者都是 4 个向量操作；若 Stage 4 能用 `vmadd` 融合才值，故不采纳（不引入额外舍入差异） |
| — | `exp→exp2`（R1 候选 9） | `e^{-conv}` | `2^{-conv·log2e}` | 容差内等价 | **API 缺失**：`tilelang.ascend.language.simd` 只导出 `vexp`/`vexpdiff`（无 `vexp2`），不采纳 |

**等价性机器验证（D-1）**：脚本 `examples/engram_gate_conv_bwd/_engram_gate_conv_bwd_kernel/verify_equiv.py`
（torch CPU，fp64 autograd 参照；基线 = 源算子算术顺序的 fp32 逐元素复刻；候选 = 单项切换后的同一代码）。
几何 `(1,32,256) (2,64,512) (1,128,256) (3,96,768) (1,8,256) (2,1,256) (3,7,192)` × 模式
`random/zeros/large/tiny/spike/nonfinite` × dtype `fp16/bf16`；判据 = 容差预算 ≤1（算子自身容差）
且「字段尺度相对误差」≤ 对应输出格点（fp16 9.8e-4 / bf16 7.8e-3；fp32 权重梯度 1e-5）且无非有限不一致。
实跑输出 `EQUIV_SUMMARY ALL_PASS`：

| 项 | dtype | 对照字段数 | 尺度相对误差 | 最大 abs | 最大容差预算 | 非有限不一致 | 结论 |
|---|---|---|---|---|---|---|---|
| BASE（基线自比） | float16 | 252 | 0.000e+00 | 0.000e+00 | 0.0000 | 0 | EQUIV_PASS |
| O1 | float16 | 252 | 2.633e-04 | 4.883e-04 | 0.0013 | 0 | EQUIV_PASS |
| O2 | float16 | 252 | 1.663e-04 | 1.953e-03 | 0.0030 | 0 | EQUIV_PASS |
| O3 | float16 | 252 | 0.000e+00 | 0.000e+00 | 0.0000 | 0 | EQUIV_PASS |
| O4 | float16 | 252 | 4.496e-04 | 4.883e-04 | 0.0014 | 0 | EQUIV_PASS |
| O5 | float16 | 252 | 0.000e+00 | 0.000e+00 | 0.0000 | 0 | EQUIV_PASS |
| O1+O2+O3+O4+O5 | float16 | 252 | 4.496e-04 | 1.953e-03 | 0.0030 | 0 | EQUIV_PASS |
| BASE（基线自比） | bfloat16 | 252 | 0.000e+00 | 0.000e+00 | 0.0000 | 0 | EQUIV_PASS |
| O1 | bfloat16 | 252 | 2.083e-03 | 3.906e-03 | 0.0089 | 0 | EQUIV_PASS |
| O2 | bfloat16 | 252 | 2.083e-03 | 3.906e-03 | 0.0092 | 0 | EQUIV_PASS |
| O3 | bfloat16 | 252 | 0.000e+00 | 0.000e+00 | 0.0000 | 0 | EQUIV_PASS |
| O4 | bfloat16 | 252 | 7.911e-04 | 1.953e-03 | 0.0051 | 0 | EQUIV_PASS |
| O5 | bfloat16 | 252 | 0.000e+00 | 0.000e+00 | 0.0000 | 0 | EQUIV_PASS |
| O1+O2+O3+O4+O5 | bfloat16 | 252 | 2.083e-03 | 3.906e-03 | 0.0092 | 0 | EQUIV_PASS |

逐字段（全部项叠加，最坏几何/模式）：

| dtype | 字段 | 尺度相对误差 | 容差预算 | 说明 |
|---|---|---|---|---|
| float16 | dH / dk / dv | 4.50e-4 / 1.98e-4 / 2.90e-5 | 1e-4 / 1.0e-3 / 3.0e-3 | 输出走 fp16 格点，差 ≤ 0.46 个 fp16 格点 |
| float16 | drms_w_h / drms_w_v / dconv_w | 1.61e-6 / 0 / 0 | ≈0 / 0 / 0 | `drms_w_v`、`dconv_w` 逐位相同（O3/O5 不改变其乘积链） |
| bfloat16 | dH / dk / dv | 1.32e-3 / 2.08e-3 / 5.79e-5 | 8e-4 / 8.9e-3 / 9.2e-3 | 差 ≤ 0.27 个 bf16 格点 |
| bfloat16 | drms_w_h / drms_w_v / dconv_w | 2.44e-6 / 0 / 0 | ≈0 / 0 / 0 | 同上 |

尺度参照：**基线自身相对 fp64 真值的尺度误差**为 fp16 `8.88e-3`、bf16 `5.39e-2`
（主因是保存态用的是前向**舍入后**的 `vhat`，而 fp64 autograd 参照走未舍入路径，§0.1），
即 5 项优化的差异（≤2.1e-3）比基线自身与真值的差距小 4–25 倍；容差预算最大仅 0.0092
（算子容差 fp16 0.2/0.2、bf16 0.3/0.2）→ **余量 ≥ 100 倍**。

**优化结论**：采纳 5 项（O1 倒数乘、O2 修正系数折叠、O3 value 链 CSE、O4 gate 链 CSE、O5 `vhat_norm` 物化），
5 项单独与叠加均 EQUIV_PASS（脚本可重跑，Stage 2 检视复核）。优化后公式即 §1.3 的
`u_h/u_k/wuh` 形式、`coef = rrms³(1/d)` 形式与 `zd = vhat·g` 复用形式，直接作为 §3.1 公式拆解与 §3.3 伪代码的输入。
放弃 2 项（候选 8 无收益、候选 9 API 缺失），理由已列。**权重梯度求和顺序**的等价性不属公式改写，
其论证见 §0.6 R1 与 §8.2。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

> 盘点 §3.3 实现方案中的**全部**循环与标量计算点。API 均在 `tilelang.ascend.language`（`T.simd.*`）查证，
> 叶子形态由 `history_version/design_probe_compile.py` 编译核验（7 几何全通过）。

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由（不可替代时必填） |
|---|---|---|---|---|---|
| 1 | `conv` 重算（4 tap） | `for p in T.serial(4)` 内逐元素 | `T.unroll(4)` + `vmul/vadd`（64 lane） | ✅ | — |
| 2 | `sigmoid` 与 `silu'` | 逐元素标量 `1/(1+exp(-x))` | `vneg`+`vexp`+`vadds`+`vdiv`（4 条向量指令） | ✅ | — |
| 3 | 卷积转置（4 tap） | 逐元素累加 | `T.unroll(4)` + `vmul/vadd` | ✅ | — |
| 4 | 4 处行点积 | 标量累加 | `vadd` 链（chunk 升序）+ `vcadd` + `vdupv` 广播 | ✅ | 不用高层 `T.reduce_sum`：二维 shared→shared 路径有已登记差异（`TRAP-950-simd-shared-reduce`），改用 `PL-950-simd-reduce` 的显式寄存器归约 |
| 5 | 全部逐元素链（`nrm/z/dvhat/dv/dH/dk/partial`） | 逐元素标量 | `vmul/vadd/vsub/vmuls/vdiv/vcvt/vsts` | ✅ | — |
| 6 | dtype 写回 cast | 逐元素 `T.cast` | `vcvt(…, dtype, round="ROUND_R")` + `vsts(dist="PK_B32")` | ✅ | — |
| 7 | 表准备（`w_v/w_h/conv_w` → fp32） | host 或逐元素 | 每核一次 `vcvt` 块（`UNPK_B16` → fp32，掩码清零尾 lane） | ✅ | — |
| 8 | partial 初始化与窗口零槽 | `T.clear`（实测降为标量 float2 循环，见 fwd 探针注释） | `vdup(0)` + `vsts`（每 64 lane 一条） | ✅ | — |
| 9 | token 推进循环 `i = 0 … RC+5` | — | 无（`T.serial`，段内串行） | ❌ | **tile 级顺序依赖**：`ds[t]` 的消费者跨 4 个 token，lane 轴固定在 `d`；改为 token 并行需把 `d_silu` 落回 GM（即 C1 备选）。依赖距离 3 → 串行链是算法固有，已被 §1.6.0 R2 的在线化吸收到窗口，无进一步并行空间 |
| 10 | chunk 循环 `ch = 0 … dw/64-1` | — | 无（`T.serial`） | ❌ | **tile 级顺序依赖**：行点积必须先算完才能用于 `dvhat`/`dH`/`dk`（两遍行算法），chunk 循环是「向量指令粒度」而非标量循环，每次迭代即一条 64 lane 向量指令，已完全向量化 |
| 11 | 槽号 `floormod(position, 4)`、`pn/pds/tp` 索引、`live/owner` 判定 | 标量 | 无 | ❌ | **行级索引/任务映射计算**：每 token 各 O(1) 次标量整数运算，非逐元素热点；标量值随之经 `vdup` 广播为向量，不构成逐元素标量循环 |
| 12 | 标量元数据载入（`alpha/rrms_h/rrms_k/rrms_v`） | 每 token 逐元素 | `vld(…, "BRC_B32")` 单 lane 广播（每批一次性条带 `T.copy`） | ✅（部分） | 索引寻址仍是标量（#11）；4 个标量/token 的**载入**已向量化为 BRC 广播载入，且条带化后不再每 token 一次 DMA |
| 13 | 归约 kernel 的核维循环 `row = 0 … G-1` | 标量累加 | 每次迭代是一条 64 lane `vadd`（`run = vadd(run, vld(part[row,col]))`） | ❌（仅行序） | **保持归约顺序**：该循环就是「按核号升序求和」的确定性契约（§0.6 R1），向量化跨核（树形合并）会改变累加顺序、放弃确定性契约。循环上限 G ≤ 56，每步都是整向量操作 |
| 14 | 归约 kernel 的列 tile / 组选择（6 组 × tiles） | — | 6 路静态 `if` 链（每支索引为常量） | ❌ | **block 级任务映射**：每 block O(1) 次分支，`group` 只用于选常量索引，不参与逐元素计算 |

**向量化结论**：逐元素计算已 100% 向量化（#1–#8、#12、#13 的算子本体），所用 API 均在
`tilelang.ascend.language.simd` 实测存在且有 `examples/`/knowledge 佐证；保留 5 处标量/循环（#9、#10、#11、#13 的行序、#14），
理由类别分别为「tile 级顺序依赖（2 处，依赖距离 3 的卷积窗口与两遍行算法）」「行级索引/任务映射（1 处）」、
「归约顺序契约（2 处）」，逐项见上表，无「实现简单」类空泛理由。

#### 1.6.3 向量化轴与数据布局决策（阻塞级）

> I/O 契约固定为 `[M, seq_len, d]` + 尾轴连续（`contiguous`），故「哪个轴做 lane」是设计变量。

**候选矩阵**（全部枚举）：

| # | 布局方案 | 向量化轴 | repack 路径 | 预估收益/代价 | 是否采纳 |
|---|---|---|---|---|---|
| 1 | I/O 原生 `[M,S,d]`，lane = 行内 `d`（64 lane fp32） | `d` | 无（`T.copy` 行搬入 + `UNPK_B16` 就地升 fp32） | 尾 lane 浪费 `(64·ceil(d/64)-d)/d`：d=768 为 0%、d=192 为 0%、d=100 为 28%、d=7 为 800%（小 d 时占比高但绝对量小） | ✅ **主选** |
| 2 | 转置为 `[M,d,S]`，lane = token | `S`（token） | 需把 5 个输入 + 3 个输出全量转置：核内 `T.copy(transpose=True)` 只文档化 GM→L1 的 DN→NZ（`references/programming.md`），UB 内转置未核验；host `permute().contiguous()` = 全量 16 B/lane 往返 | repack 代价 ≈ 算子本体流量（29 B/lane）甚至更高；且 `d` 变串行维 → 4 处行点积退化为「每 lane 沿 `d` 串行 768 次标量加」，彻底违背向量化 | ❌ 弃选（repack 代价 + 行点积退化） |
| 3 | `d` 维分块跨核（列并行），行点积跨核合并 | `d` | 无 | 可把核数从 ≤56 提到任意；但 4 处行点积需**跨核归约**，本平台跨 block barrier 未认证（`CG-950-batch-barrier`），只能再落 GM 往返（每 token 2 次 fp32 往返） | ❌ 弃选（跨核同步无证据） |
| 4 | 窗口缓冲布局 `[4][dw]`（tap/position 为外层行） | `d`（行内连续） | 无 | `floormod(p,4)` 只改行偏移，`vld/vsts` 仍取连续 64 lane | ✅ 采纳（O5/窗口的实现形式） |
| 5 | 窗口缓冲布局 `[dw][4]`（`d` 为外层） | — | 需转置 | lane 沿 4 个位置 → 每 lane 只有 4 个有效元素，向量宽度浪费 16×，且 4 个 tap 变成跨 lane 移位 | ❌ 弃选 |
| 6 | host padding `d→align_up(d,256)` | `d` | host `F.pad`（8 个张量全量拷贝） | 多 8 次全量拷贝；对齐到 64 已足够（fp32 向量宽度 64） | ❌ 弃选（§0.6 R6） |
| 7 | 标量元数据条带 `[4][seq]` 常驻 vs 每 token 4 次标量 DMA | — | 每批一次 `T.copy` | 条带 16·seq B（seq=512 → 8 KiB）；换掉 4·tokens 次小 DMA | ✅ 采纳（`seq ≤ 4096` 时；更长的 `seq` 回退到每 token 标量拷贝，见 §9） |

**评分（主选轴 `d` vs 次选 `token`）**：

| 评分项 | `d` 轴（主选） | `token` 轴（弃选 #2） |
|---|---|---|
| 整除性 | `dw = ceil(d/64)·64`，d=256/512/768 全整除；d=192 也整除 | token 数 288/1203 等，需 64 的倍数才有用；实际几乎总有尾 lane |
| 尾 lane 浪费率 | 0%（全部 manifest/test 的 d 都是 64 的倍数；d=100 时 28%） | `(64·ceil(tokens/64)-tokens)/tokens`：tokens=288 → 10%；tokens=1203 → 6% |
| 累加链形态 | 4 处行点积 = `d` 轴归约 → `vadd`+`vcadd`，天然向量归约 | 行点积沿 `d` 串行 → 每 lane 768 次标量迭代，不可向量化 |
| repack 代价 | 0 | ≥16 B/lane 往返（≈ 算子本体流量） |
| UB 容量影响 | 窗口 3×16 + 表 24 + partial 24 + 暂存 22 + 整行暂存 16 + 条带 ≈ **134 B/channel**（d=768 → 102 KiB） | 需 `[dw][tile_tokens]` 布局，`d` 串行导致每 lane 一个 token，UB 反而需驻留整列 → `4·dw·4·tokentile` 爆炸 |

**布局决策结论**：选定方案 #1+#4+#7：核内布局 = 「`d` 为 64 lane 连续向量轴、position/tap 为外层行」的
`[4][dw]` 滚动窗口 + 逐行暂存的 fp16 输入行；向量化轴 = `d`；repack 路径 = 无（`T.copy` 行搬入 +
`UNPK_B16` 就地升 fp32 + `PK_B32` 就地降 dtype）；标量元数据用 `[4][seq]` 条带。该结论同步落入
§3.3 伪代码（窗口形状 `[KS, dw]`、`ch` 循环取 `vld(…, ch*VL)`）与 §6 循环结构（累加循环的内层向量维 = `d`）。

#### 1.6.4 备选方案与实验裁决计划

**为什么需要裁决**：主选（C2）与备选（C1）的取舍取决于两个**未实测常数**——(i) 单次 kernel launch 的
固定开销，(ii) UB→GM→UB 的 fp32 中间往返相对向量计算的价格。平台 `cost_measurements` 为空、
`knowledge/bottlenecks.md` 明确无 950 真机瓶颈数据，故按 skill Phase 2 第 3 项给三件套。

**主选（Stage 3 实现目标）**：C2 = 融合单计算 kernel（§1.4/§3.3）+ 有序归约 kernel，2 次 launch。
适用 `d ≤ 1024`（UB 预算见 §4.5），`seq ≤ 4096`（条带路径）。

**备选（B1，结构完整、可直接实现）**：C1 = staged 确定化，**3 次 launch**，`d_silu` 落 GM：

- kernel A（grid = 同一几何 `M·nch`）：只做 §1.4 的 (a)+(b)：算 `nrm`/`ds` 并写 `DS[tokens, dw]` fp32，
  同时按 per-core 累加 `pc[4][dw]`；UB = 仅需 `nrm` 窗口 16 B/channel + ds 单行 4 B + 表 24 B +
  partial 16 B + 单行暂存 12 B ≈ 76 B/channel（d=2048 → 152 KiB ✓）；若把 3 张 fp32 表回退成 fp16 逐次转换
  （−24 B/channel ≈ 52 B/channel）则 d=4096 → 208 KiB ✓（余量 18%）。**故备选覆盖 `1024 < d ≤ 4096`，
  但 4096 端依赖「表留 fp16」变体**。
- kernel B（grid 同上）：读 `DS` 的 4 行做 §1.4 的 (c)，写 `dH/dk/dv` + `ph/pv`；UB ≈ 76 B/channel
  （`DS` 4 行窗口 16 B + `gn/gz/hnf/knf` 16 B + 表 24 B + `ph/pv` 8 B + 暂存 12 B）。
- kernel C：与主选相同的归约 kernel（`PART[6, G, dw]`，6·tiles ≤ 54）。
- 循环结构、分核三要素、同步策略与主选同一套（仅 token 循环内的窗口改为「GM 读 DS 的 4 行」）；
  `T.copy(DS[t+3-p, :], dsrow[p, :d])` 用静态 `p` 行 + 动态源行（与源算子 `dvhat_buf[bid, dst_t, j]` 同形）。

**实验裁决计划**（Stage 4 执行，基准不动）：

| 项 | 内容 |
|---|---|
| 代表 shape | `(1,32,256,fp16)`、`(2,64,512,fp16)`、`(1,128,256,bf16)`、`(3,96,768,bf16)`、`(2,512,256,fp16)`、`(3,401,256,bf16)` |
| 指标 | 平台口径 `msprof_task_duration`（`perf_records.jsonl`），A/B 交错 ≥5 轮取中位 |
| 待反解常数 | (i) 单次 launch 开销 ≈ (C1 − C2) 在「同流量 shape」的差值基线；(ii) DS 往返单价 = 从 kernel A/B 与融合版的流量差反解 |
| 判定阈值 | 备选 latency < 主选 × 0.90 才翻转；差异 <10% 保留主选（2 launch + 单遍扫描的确定性优势） |
| 按 shape 分派 | 工厂按 `(M, seq_len, d, dtype)` 做 `lru_cache` 特化，**允许按 shape 分派主选/备选**（例如 `d > 1024` 一律走 B1；`seq_len < 8` 一律走主选）。裁决结果写回 §1.6.3/§1.6.4 并触发设计修订 |
| 回写路径 | 实测数据 → `perf_opt/opt_log.md` 的「设计估算 vs 实测」偏差行 → 若翻转则 `perf_feedback.md` + 设计修订 |

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: SIMD（`T.SimdVF` + `tilelang.ascend.language.simd` 寄存器级向量指令，一维核网格
`T.Kernel(grid)`，grid ≤ 56 = 目标型号 Vector 核数）。

### 2.2 选型理由

- 算子**没有任何矩阵乘**：4 个「行点积」是 `1×d` 与 `d×1` 的内积形态，且必须与大量逐元素链
  （`sigmoid`、`silu'`、RMSNorm 修正、dtype cast）紧耦合；Cube 路径要先把行变成 GEMM tile，
  再为每个 token 搬一次，反而引入 L0C/L1 往返与 `M=1` 的 tile 浪费。
- 逐元素 + 归约的组合正是 SIMD 的强项：64 lane fp32 向量 + `vcadd` 行归约
  （`knowledge/PL-950-row-primitives.md`、`knowledge/numerical-boundaries.md` 的 `PL-950-simd-reduce` 均已核验）。
- 平台 `programming_modes.migration_default = simd`；`simt` 亦可表达同一代码（`T.SimtVF` + `T.Parallel`），
  但本算子的行归约已在 SIMD 路径有验证条目，且 SIMT 的线程域在本平台需显式 `T.SimtVF(threads=…)`，
  对 64 lane 向量化无额外收益。
- **不用混合模式**：Cube 侧无计算可承接（无 GEMM），§1.6.0 R4 的「计算单元匹配」一条即淘汰 Cube 候选。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|---|---|
| 内存分配 | `T.alloc_shared(shape, dtype)`（UB，默认 `shared.dyn`）；不使用 `alloc_l1/l0a/l0b/l0c` |
| 计算方式 | `T.SimdVF()` 帧内 `vld/vcvt/vmul/vadd/vsub/vdiv/vexp/vcadd/vdupv/vsts`（64 lane/指令） |
| 数据搬运 | `T.copy`（GM↔UB 行/块）；`GM→UB` 只在每 token 一次行搬入 + 每批一次表搬入 |
| 并行结构 | 一维核网格，一核一段 token；核内 token 串行（依赖距离 3 的窗口），chunk 串行（两遍行算法） |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式 = §1.6.1 的**优化后公式**（O1–O5 已折叠入下式）；每步与 §1.6.2 的向量化结论一致。

| 步骤 | 数学表达（每 token `t`、每 lane `j < d`） | 说明 |
|---|---|---|
| S0 | `nrm[p] = (vhat[p] · rrms_v[p]) · w_v`，`p = pn` | 每行一次（O5），写 4 槽窗口 |
| S1 | `conv = Σ_{p<4} conv_w[p] · nrm[pds-3+p]` | 4 tap 展开（`T.unroll`） |
| S2 | `s = 1/(1+e^{-conv})`；`d_{silu} = dY · (s + conv·s·(1-s))` | `vneg/vexp/vadds/vdiv` |
| S3 | `pc[p] += d_{silu} · nrm[pds-3+p]` | per-core `dconv_w` 部分和（R1） |
| S4 | `g = Σ_{p<4} conv_w[p] · d_{silu}[t+3-p]` | 卷积转置（窗口读） |
| S5 | `zd = vhat · g`；`dot_v = Σ_j zd·w_v`（→ `vcadd`）；`pv += zd · rrms_v` | O3 公共子式 |
| S6 | `coef = rrms_v³·(1/d)`；`gz = rrms_v·(w_v·g) − (coef·vhat)·dot_v + dY` | O2 + O1 |
| S7 | `dv = alpha · gz`（→ dtype）；`dalpha = Σ_j gz·v`（→ `vcadd`）；`ddot = dalpha·alpha·(1-alpha)·(1/sqrt d)` | O1 |
| S8 | `h_n = (H·rrms_h)·w_h`，`k_n = (k·rrms_k)·w_h`；`u_h = ddot·k_n`，`u_k = ddot·h_n` | O4 |
| S9 | `ph += (u_h·H)·rrms_h + (u_k·k)·rrms_k`；`dot_h = Σ_j H·(w_h·u_h)`，`dot_k = Σ_j k·(w_h·u_k)` | O4 + `vcadd` |
| S10 | `rh3d = rrms_h³·(1/d)`；`dH = rrms_h·(w_h·u_h) − (rh3d·H)·dot_h`；`dk` 同形（`rrms_k`） | O1 + O2；→ dtype 写回 |
| S11 | 归约：`dconv_w[p,j] = Σ pc[p,j]`（按核号升序求和），`drms_w_v[j] = Σ pv`，`drms_w_h[j] = Σ ph` | §0.6 R1 |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|---|---|---|---|---|
| 任务映射 | `b = bx // nch`，`c = bx % nch`，`t0 = c·RC` | `T.Kernel(grid)` + 整数除模 | 一维 grid（`threads=` 被实现拒绝） | SIMD |
| 表准备 | `w_v,w_h,conv_w → fp32` | `T.copy` + `T.simd.vld(dist="UNPK_B16")` + `T.simd.vcvt(..., "float32", mask, part=0)` + `T.simd.vsts` | 每核一次 | SIMD |
| 标量条带 | `alpha/rrms_h/rrms_k/rrms_v` | `T.copy(ALPHA[b, :], sc[0, :])` | 每批一次，`[4, seq]` fp32 | SIMD |
| 行搬入 | `vhat[pn]`, `dY[pn]`, `H/K/V[t]` | `T.copy(VHAT[b, pn, :], vrow[:d])` | 静态目的行 + 动态源行 | SIMD |
| 行标量 | `rrms_v[pn]` 等 | `T.simd.vld(sc[3, pn], "BRC_B32")` | 单 lane 广播载入 | SIMD |
| S0 | `nrm = (vhat·rv)·w_v` | `vcvt` + `vmul` + `vmul`；`vsts(nrmw[slot, col], …)` | 槽 `floormod(pn,4)` | SIMD |
| S1/S4 | 4 tap 累加 | `T.unroll(KS)` + `vld` + `vmul` + `vadd` | `mask=update_mask(min(64, d-64·ch), 32)` | SIMD |
| S2 | `sigmoid` / `silu'` | `vneg` + `vexp` + `vadds` + `vdiv` + `vmul` + `vsub` | — | SIMD |
| S5/S7/S9 | 行点积 | `vadd` 链 + `vcadd(acc, full)` + `vdupv(scalar, full)` | chunk 升序（确定性） | SIMD |
| S6/S10 | 标量系数 | 标量寄存器 `T.simd.alloc_local((1,), "float32")` + `vmul/vmuls` + `vdup` | `coef/rh3d/rk3d` 每 token 一次 | SIMD |
| S7/S10 | dtype 写回 | `T.simd.vcvt(v, dtype, round="ROUND_R", sat=False, part=0)` + `vsts(dist="PK_B32")` | 掩码排除 `j ≥ d` | SIMD |
| 输出搬出 | `dH/dk/dv` | `T.copy(hrow[:d], DH[b, tp, :])` | 静态源 + 动态目的 | SIMD |
| partial 搬出 | `pc/ph/pv` | `T.copy(pc[p, :], PART[2+p, bx, :])` / `T.copy(ph[:], PART[0, bx, :])` | 动态第二维索引 | SIMD |
| 窗口保持单版本 | 滚动窗口不得自动多缓冲 | `T.annotate_buffer_versions({buf: 1})` | 全部 UB 缓冲 | — |
| 零槽/零 partial | 清零 | `T.simd.vdup(0, "float32", full)` + `vsts`（短缓冲用 `T.view` + `T.uint32`，同 950DT `_zero_simd`） | — | SIMD |
| 归约 kernel | 按核号升序求和 | `T.Kernel(6·tiles)` + 6 路静态 `if` + `vld(part[row, col])` 链 + `vcadd` 无（逐 lane 和） | 动态列 extent `cols = min(width, dw-off)` | SIMD |

### 3.3 计算伪代码

> 与 `history_version/design_probe_compile.py` 的实现一一对应（该探针已在 7 条几何上编译通过，
> 下面只写关键结构；`VL = 64`、`KS = 4`、`dw = ceil(d/VL)·VL`、`nchunks = dw/VL`）。

```python
# ---------------- launch #1: fused compute ----------------
with T.Kernel(grid) as bx:                    # grid = M * nch <= 56
    b, c = bx // nch, bx % nch
    t0 = c * RC
    # UB: 单行暂存(fp16) x7, 表(fp32) cwf/wvf/whf, 窗口 nrmw/dsw/dyw[4,dw] fp32,
    #     partial pc[4,dw]/ph[dw]/pv[dw] fp32, 暂存 gn/gz/hnf/knf[dw] fp32, 标量条带 sc[4,seq]
    T.annotate_buffer_versions({... : 1})     # 滚动窗口必须单版本
    zero_partials(); load_tables(); load_scalar_strips()
    for i in T.serial(RC + 2 * KS - 2):        # RC + 6 步
        pn = t0 - 3 + i                       # 本步负责的 nrm/dY 行
        slot = T.floormod(pn, KS)
        if pn >= 0:
            if pn < seq:                      # in-batch：搬入并计算窗口槽
                T.copy(VHAT[b, pn, :], vrow[:d]); T.copy(DY[b, pn, :], drow[:d])
                with T.SimdVF():
                    rv = T.simd.vld(sc[3, pn], "BRC_B32")
                    for ch in T.serial(nchunks):
                        mask = update_mask(min(VL, d - ch*VL))
                        nrm = vmul(vmul(cvt(vrow), rv, mask), wvf, mask)   # O5
                        vsts(nrmw[slot, ch*VL], nrm); vsts(dyw[slot, ch*VL], cvt(drow))
            if pn >= seq:                     # 批尾越界：只写零槽（不载入）
                zero_slot(nrmw[slot]); zero_slot(dyw[slot])
        if pn < 0:                            # 批首越界
            zero_slot(nrmw[slot]); zero_slot(dyw[slot])

        if i >= 3:                            # 本步产出的 d_silu 行
            pds = t0 + i - 3; dslot = T.floormod(pds, KS)
            if pds < seq:
                with T.SimdVF():
                    for ch in T.serial(nchunks):
                        conv = 0
                        for p in T.unroll(KS):    # S1
                            conv = vadd(conv, vmul(vld(nrmw[floormod(pds-3+p, KS)]), cwf[p], mask))
                        s = vdiv(1, vadds(vexp(vneg(conv)), 1));  deriv = vadd(s, vmul(vmul(conv,s), vsub(1,s)))  # S2
                        vsts(dsw[dslot], vmul(vld(dyw[dslot]), deriv, mask))
            if pds >= seq:
                zero_slot(dsw[dslot])
            if i < RC + 3:                    # 只有拥有该行的核累加 dconv_w（S3）
                if pds < seq:
                    with T.SimdVF():
                        for ch in T.serial(nchunks):
                            for p in T.unroll(KS):
                                vsts(pc[p], vadd(vld(pc[p]), vmul(vld(dsw[dslot]), vld(nrmw[...]), mask)))

        if i >= 6:                            # 滞后 3 的 token: value + gate 反向
            tp = t0 + i - 6
            T.copy(H[b, tp, :], hrow[:d]); T.copy(K[b, tp, :], krow[:d]); T.copy(V[b, tp, :], vvrow[:d])
            with T.SimdVF():
                rv, rh, rk, al = vld(sc[3,tp],"BRC_B32"), vld(sc[1,tp],...), vld(sc[2,tp],...), vld(sc[0,tp],...)
                # pass A: S4/S5 -> gn, dot_v, pv
                # pass B: S6/S7 -> gz, dv 输出, dalpha
                # pass C: S8/S9 -> hnf, knf, ph, dot_h, dot_k
                # pass D: S10  -> dH/dk 输出
            T.copy(hrow[:d], DH[b, tp, :]); T.copy(krow[:d], DK[b, tp, :]); T.copy(vvrow[:d], DV[b, tp, :])
    for p in T.serial(KS): T.copy(pc[p, :], PART[2+p, bx, :])
    T.copy(ph[:], PART[0, bx, :]); T.copy(pv[:], PART[1, bx, :])

# ---------------- launch #2: ordered reduction ----------------
width = 64 * ceil(dw / (64 * 9)); tiles = ceil(dw / width)          # 6*tiles <= 54
with T.Kernel(6 * tiles) as bx:
    g, off = bx // tiles, (bx % tiles) * width
    cols = min(width, dw - off)
    if g == 0: copy(PART[0, :, off:off+cols] -> part)   # ... 6 路静态分支
    with T.SimdVF():
        for ch in T.serial(width // 64):
            run = 0
            for row in T.serial(G): run = vadd(run, vld(part[row, ch*64]))   # 核号升序
            vsts(tot[ch*64], run)
    if g == 0: copy(tot[:cols] -> DWH[off:off+cols])     # g=1..5 -> DWV / DCW[0..3]
```

**四个 chunk pass 的必要性**（`d` 轴两遍算法的直接后果，§1.6.2 #10）：pass A 求 `g` 与 `dot_v`
（全行标量）→ pass B 才能算 `dvhat`/`dv`/`dalpha`（`ddot` 依赖全行 `dalpha`）→ pass C 求 `h_n/k_n` 与两个行点积 →
pass D 才能算 `dH/dk`。每 pass 都是一次 `for ch in T.serial(nchunks)` 的纯向量循环；`gn/hnf/knf/gz`
驻留 UB 以免重复计算（`4·dw·4 B`）。

### 3.4 API 可行性确认

| API | 来源确认 | 状态 |
|---|---|---|
| `T.Kernel(grid)`（无 `threads=`） | `references/programming.md`「`T.Kernel(N)` 一维核网格；签名没有 `threads`」；探针 7 几何 | 已核验（编译） |
| `T.simd.vld/vsts`（`NORM` / `UNPK_B16` / `BRC_B32` / `PK_B32` / `extent=`） | `tilelang.ascend.language.simd`（`/workspace/venv-ascend950/…/tilelang/ascend/language/simd.py` 导出表）+ 950DT 已验证实现 + 探针 | 已核验（编译） |
| `T.simd.vcvt(src, dtype, mask, round="ROUND_R", sat=False, part=0)` | `simd.py` 签名 `(src, target_dtype, mask=None, round='ROUND_R', sat=True, part=0, mode='MODE_ZEROING')`；`knowledge/numerical-boundaries.md` 的转换条目 | 已核验 |
| `T.simd.update_mask(w, 32)` / `pset(32)` | 950DT 实现 + `PL-950-simd-reduce` | 已核验 |
| `T.simd.vcadd(acc, full)` + `vdupv` | `PL-950-simd-reduce`（FP32 (2,64)/(2,63) 通过） | 已核验 |
| `T.simd.vexp` | `PL-950-row-primitives`（softmax 路径） | 已核验 |
| `T.alloc_shared` / `T.annotate_buffer_versions({b:1})` | `references/programming.md`（`{buf: 1}` 取消自动多缓冲资格） | 已核验 |
| `T.copy`（静态行 ↔ 动态行/区域，动态 extent） | 950DT 归约实现（动态 `group`/`columns`）+ 探针（动态目的行、动态源行） | 已核验（编译） |
| `T.floormod` | 探针（滚动窗口槽号） | 已核验（编译） |
| 未使用 | `T.atomic_add`（§0.6 R1 刻意避开）、`T.reduce_sum`（`TRAP-950-simd-shared-reduce`）、`T.set_atomic`、`T.Pipelined` | — |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束（来源） | 本算子是否涉及 | 处理方案 |
|---|---|---|
| `T.Kernel` 只支持一维网格，`threads=` 被拒绝（`references/programming.md`） | 是 | 一维 grid；线程域不声明（纯 SIMD 帧） |
| 高层二维 shared→shared 归约输出布局差异（`CG-950-simd-shared-reduce` / `TRAP-950-simd-shared-reduce`） | 是（4 处行点积） | 改用显式寄存器归约（`vadd` 链 + `vcadd` + `vdupv`，`PL-950-simd-reduce`） |
| 跨 block barrier 未认证（`CG-950-batch-barrier`） | 是（权重梯度需跨核合并） | 拆成 2 次 launch，经 GM workspace 合并（§0.6 R1） |
| UB 容量 253,952 B（CANN 9.1.0 `Ascend950PR_9579.ini`，`design_budget.capacity_bytes`） | 是 | 峰值 104,448 B（d=768）/ 142 KiB（d=1024）；d > 1024 走 §1.6.4 备选 |
| L0A/L0B/L0C 容量 | 否 | 无 Cube 计算，不使用 |
| 缓冲自动多缓冲会破坏滚动窗口语义 | 是 | `annotate_buffer_versions({…: 1})` 全部 UB 缓冲单版本 |
| `T.copy` 元素数必须相等（整 Buffer 对整 Buffer）；局部拷贝用显式 region | 是 | 所有搬入/搬出都带 `[:d]` / `[:, off:off+cols]` 显式 region |
| 手工跨核 flag / 循环复用复杂同步未认证 | 否 | 不写任何 `ascend_set_flag` / 跨核 flag |

#### 3.5.2 参考实现差异说明

| 差异项 | 参考实现（GPU 源算子） | 本项目（Ascend 950PR） | 转换方案 |
|---|---|---|---|
| Kernel 结构 | 4 个 `T.Kernel` 段（清零 + 3 段），`threads=128` | 2 个 launch：融合计算 + 有序归约，一维核网格 | §0.6 R2/R1 |
| 并行原语 | `T.Parallel(d_padded)` 线程级 + `T.reduce_sum` 块内归约 | `T.SimdVF` + `T.simd.*` 64 lane + `vcadd` | §0.6 R3 |
| 跨核累加 | `T.atomic_add`（3 处） | per-core UB 部分和 + 归约 kernel | §0.6 R1 |
| 内存分配 | `T.alloc_fragment`（寄存器/共享），中间量落 GM | `T.alloc_shared`（UB）+ 4 槽滚动窗口 | §0.6 R2 |
| host 侧 | `F.pad` 到 `align_up(d,256)` + 裁回 | 无 host padding，设备侧 `dw` 掩码 | §0.6 R6 |
| 分核 | grid = (seq_len, M)，`run_len` 由 `sm_count` 决定 | `nch/RC` 由 56 Vector 核与 token 数重导 | §0.6 R4 |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|---|---|---|
| `examples/engram_gate_conv/_engram_gate_conv_fwd_kernel/history_version/design_probe_compile.py` | **高度相似**（同族前向、融合单 launch + 滚动窗口） | 滚动窗口动态槽号 `slot[i%4]` 的 `vld/vsts` 用法、`T.serial(rc+3)` 行循环 + 因果 halo、标量写回条带、`annotate_buffer_versions` 单版本、几何规则 |
| `examples/TileOPs/tileops/kernels/sequence_modeling/engram_gate_conv_bwd/engram_gate_conv_bwd.py` | 源实现（同算子 GPU 版） | 公式与累加顺序的**唯一权威**（§0.3/§0.4 逐语句核对） |
| `examples/mla_dense_bwd/_bwd_kernel/DESIGN.md` | 中（迁移反向算子的设计范式） | 分核三要素写法、D-1 等价验证表、风险登记口径、迁移章节组织 |
| `examples/ssd_chunk_coupling/_ssd_chunk_coupling_kernel/DESIGN.md` | 中（迁移设计文档范式） | §0 三问解读 + 四态处置的写法、语义要素表 |
| `examples/paged_cache_write/_paged_cache_write_kernel/DESIGN.md` | 中（同平台 SIMD/SIMT 先例） | 章节组织、UB 预算表、同步说明 |
| `examples/TileOPs/tests/ops/test_engram.py` | 高（本算子的验收口径） | 容差 `_tol_bwd`、18 例边界（`(2,1,256)`、`(1,8,256)`、`(3,401,256)`、非连续、dtype 拒绝） |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|---|---|---|---|
| `dY` | `[M, seq_len, d]` | float16 / bfloat16 | 输出梯度，尾轴连续（wrapper 保证 `.contiguous()`） |
| `H`, `k`, `v` | `[M, seq_len, d]` | same as `dY` | 前向输入 |
| `rms_w_h`, `rms_w_v` | `[d]` | same as `dY` | RMSNorm 权重（`H`/`k` 共用 `rms_w_h`） |
| `conv_w` | `[4, d]` | same as `dY` | 4 个卷积 tap |
| `vhat` | `[M, seq_len, d]` | same as `dY` | 前向**已舍入**的 `alpha·v` |
| `alpha`, `rrms_h`, `rrms_k`, `rrms_v` | `[M, seq_len]` | float32 | 前向保存的 gate 与三个 rsqrt |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|---|---|---|---|
| `dH`, `dk`, `dv` | `[M, seq_len, d]` | same as `dY` | 输入梯度；`vcvt(ROUND_R)` 一次舍入 |
| `drms_w_h`, `drms_w_v` | `[d]` | float32 | 两个 RMSNorm 权重梯度 |
| `dconv_w` | `[4, d]` | float32 | 卷积权重梯度 |
| `PART`（内部 workspace） | `[6, G, dw]` | float32 | per-core 权重梯度部分和；`G = grid ≤ 56`；由 `out_idx` 自动分配 |

### 4.3 中间缓冲区（均在 UB，单版本）

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|---|---|---|---|---|
| `vrow/drow/hrow/krow/vvrow` | `[dw]` | dtype | UB | 单行暂存（`vhat/dY/H/k/v`；输出复写 `hrow/krow/vvrow`） |
| `whs/wvs` | `[dw]` | dtype | UB | `rms_w_h/rms_w_v` 暂存 |
| `cws` | `[4, dw]` | dtype | UB | `conv_w` 暂存 |
| `whf/wvf` | `[dw]` | fp32 | UB | fp32 表（每核一次转换） |
| `cwf` | `[4, dw]` | fp32 | UB | 同上 |
| `nrmw` | `[4, dw]` | fp32 | UB | `vhat_norm` 4 槽滚动窗口（O5） |
| `dsw` | `[4, dw]` | fp32 | UB | `d_silu` 4 槽滚动窗口 |
| `dyw` | `[4, dw]` | fp32 | UB | `dY` 4 槽窗口（DS 与残差两处消费，距离 3） |
| `pc` | `[4, dw]` | fp32 | UB | per-core `dconv_w` 部分和 |
| `ph`, `pv` | `[dw]` | fp32 | UB | per-core `drms_w_h`/`drms_w_v` 部分和 |
| `gn`, `gz`, `hnf`, `knf` | `[dw]` | fp32 | UB | 整行驻留暂存（`d_vnorm`/`dvhat`/`h_norm`/`k_norm`） |
| `sc` | `[4, seq]` | fp32 | UB | `alpha/rrms_h/rrms_k/rrms_v` 标量条带（每批一次 `T.copy`） |
| `part`, `tot`（归约 kernel） | `[G, width]`, `[width]` | fp32 | UB | 部分和暂存与运行和 |

### 4.4 内存搬运路径

```
GM ──T.copy(行/表)──> UB 单行暂存(fp16) ──vld(UNPK_B16)+vcvt──> UB fp32 表/窗口 ──vmul/vadd/vdiv/vexp──>
    UB fp32 结果 ──vsts──> UB fp32 暂存/窗口 ──vcvt(PK_B32)+vsts──> UB 单行暂存(fp16) ──T.copy──> GM
GM ──T.copy──> UB[sc 条带 fp32] ──vld(BRC_B32)──> 标量寄存器（每 token 4 个）
UB[pc/ph/pv] ──T.copy──> GM[PART] ──(launch#2)──> vld 链 ──> UB[tot] ──T.copy──> GM[输出]
```

- **无 L1/L0 路径**（无 Cube）；**无 GM→L1→UB 中转**（一维核网格下 `T.copy` 直接 GM↔UB）。
- 每核 GM 读取次数：`vhat/dY` 各 `RC+6` 行（含 3 行 halo）、`H/k/v` 各 `RC` 行、表 1 次、标量条带 1 次/批；
  每核写出：`dH/dk/dv` 各 `RC` 行、`PART` 6 行。
- 中间量 `nrm/ds/dY/gz/hnf/knf` **不落 GM**（源算子落 2 个 fp32 张量，§1.6.0 R3）。

### 4.5 UB 内存预算

代表 workload `train-s96`（M=3, seq=96, d=768 → `dw=768`, `nchunks=12`, `RC=6`, `G=48`）：

| Buffer | Shape | dtype | 大小 (Bytes) |
|---|---|---|---|
| `vrow/drow/hrow/krow/vvrow` | 5 × `[768]` | bf16 | 7,680 |
| `whs/wvs` | 2 × `[768]` | bf16 | 3,072 |
| `cws` | `[4, 768]` | bf16 | 6,144 |
| `nrmw/dsw/dyw` | 3 × `[4, 768]` | fp32 | 36,864 |
| `whf/wvf` | 2 × `[768]` | fp32 | 6,144 |
| `cwf` | `[4, 768]` | fp32 | 12,288 |
| `pc` | `[4, 768]` | fp32 | 12,288 |
| `ph`, `pv` | 2 × `[768]` | fp32 | 6,144 |
| `gn/gz/hnf/knf` | 4 × `[768]` | fp32 | 12,288 |
| `sc` | `[4, 96]` | fp32 | 1,536 |
| **总计** | | | **104,448 B = 102.0 KiB / 253,952 B（41.1%）** |

容量来源：`statectl platform` 的 `hardware.capacity_bytes.UB = 253952`，出自 CANN 9.1.0 `Ascend950PR_9579.ini`
`[AICoreSpec] ub_size`（file md5 `2f0eca1f7ca70f62ee7d6d33990c3be1`）。归约 kernel 另需
`G·width·4 + width·4 ≤ 56·128·4 + 512 = 29,184 B`，与计算 kernel 不同 launch，不叠加。

随 `d` 的缩放（每 channel ≈ **134 B** + `16·seq` B；`seq=512` 时）：

| d | dw | UB（seq=512） | UB（seq=2048） | 结论 |
|---|---|---|---|---|
| 256 | 256 | 41.5 KiB | 65.5 KiB | ✓ |
| 512 | 512 | 75.0 KiB | 99.0 KiB | ✓ |
| 768 | 768 | 108.5 KiB | 132.5 KiB | ✓（manifest/test 最大 d） |
| 1024 | 1024 | 142.0 KiB | 166.0 KiB | ✓ **主选上界**（余量 34%） |
| 1536 | 1536 | 209.0 KiB | 233.0 KiB | ⚠ 无余量，不承诺 |
| 2048 | 2048 | 276.0 KiB | — | ✗ 超容量 → 走 §1.6.4 备选 |
| 4096 | 4096 | — | — | ✗ 主选不支持；备选可覆盖（≈41 B/channel ≈ 164 KiB） |

### 4.6 动态轴定义

无 kernel 级动态轴：`M`、`seq_len`、`d`、`dtype` 全部在 JIT 期由工厂参数固定（与源算子的
`@target_cache`/`lru_cache` 同构，shape 特化编译）。算子层 `M` 属动态轴 → 每个 `M` 产生一个编译产物；
`dw/RC/nch/G/nchunks` 均由这三个量在 build 期算出，kernel 内循环边界全为编译期常量。

### 4.7 JIT 配置

```python
@functools.lru_cache(maxsize=32)                  # 与源算子 target_cache(32) 对齐
def build(M, seq_len, d, dtype): ...
    return tilelang.compile(main, target="ascend", out_idx=[12, 13, 14, 15])
# 12,13,14 = dH,dk,dv ; 15 = PART workspace
# 无 compile_flags（源算子的 "-O3 -DENABLE_BF16" 属 CUDA 后端参数，本平台不需要）
```

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**: 纯 Vector（SIMD）。

**判定依据**：无矩阵乘、无 tensor-core 形态计算；全部是逐元素链 + 4 处 `d` 轴行点积 + 3 处 token 轴求和，
按 `references/decision-tree.md` 属「逐元素 + 规约」型 → 纯 Vector，使用 56 个 AIV（Cube 核空闲）。

### 5.2 Block 划分

```python
dw      = ceil(d / 64) * 64                  # fp32 向量宽度 64；无 host padding
nchunks = dw // 64                           # 每行的向量 chunk 数
nch     = min(seq_len, max(1, 56 // M))      # 每 batch 的 chunk 数（56 = Vector 核数）
RC      = ceil(seq_len / nch)                # 每核 token 数（chunk 长度）
nch     = ceil(seq_len / RC)                 # 回填：RC 取整后可能减少 chunk
grid    = M * nch                            # 逻辑核数（<= 56）
```

`RC` 选择理由：核内工作量 ≈ `(RC+3)` 行 stage-1 + `RC` 行 stage-2，halo 占比 `3/RC`；
而核数 `M·nch` 决定并行度。取「`nch` 尽量大（占满 56 核）→ `RC` 由整除上取整」使
两方向同时最优：token 数少于 56 时 `RC=1`（并行度优先，halo 的绝对量极小），
token 数多时 `RC` 自动变长（`3/RC` 降到 13–16%）。

几何表（8 条代表 shape：4 条 manifest + 4 条测试边界）：

| 用例 | M | seq_len | d | dw | RC | nch | grid（逻辑核数） | 每核行数 | UB |
|---|---|---|---|---|---|---|---|---|---|
| train-s32（smoke） | 1 | 32 | 256 | 256 | 1 | 32 | 32 | 4 | 34.0 KiB |
| train-s64 | 2 | 64 | 512 | 512 | 3 | 22 | 44 | 6 | 68.0 KiB |
| train-s128 | 1 | 128 | 256 | 256 | 3 | 43 | 43 | 6 | 35.5 KiB |
| train-s96 | 3 | 96 | 768 | 768 | 6 | 16 | 48 | 9 | 102.0 KiB |
| `(2,512,256)` | 2 | 512 | 256 | 256 | 19 | 27 | 54 | 22 | 41.5 KiB |
| `(3,401,256)`（素数 seq） | 3 | 401 | 256 | 256 | 23 | 18 | 54 | 26 | 39.8 KiB |
| `(1,8,256)` | 1 | 8 | 256 | 256 | 1 | 8 | 8 | 4 | 33.6 KiB |
| `(2,1,256)` | 2 | 1 | 256 | 256 | 1 | 1 | 2 | 4 | 33.5 KiB |

归约 kernel：`width = 64·ceil(dw/576)`，`tiles = ceil(dw/width) ≤ 9`，`grid = 6·tiles ≤ 54`
（d=768 → width=128, tiles=6, grid=36；d=256 → width=64, tiles=4, grid=24）。

### 5.3 约束分析

- **对齐约束**：`d` 尾轴按 64 lane 取 `dw`；`T.copy` 只搬 `[:d]`；SIMD 用
  `update_mask(min(64, d-64·ch))` 排除尾 lane；表尾 lane 显式清零。GM 行步长 = `d` 个元素：
  fp16 下 `d % 16 == 0` 时每行 32 B 对齐，否则行首按元素寻址（`T.copy` 行搬入本身不要求每行
  32 B 对齐，但 DMA 效率下降 → 计入 Stage 4 调优）。探针覆盖 `d=192`（64 的倍数、非 128），
  `d` 非 64 倍数由 L0 的 `(1,16,100)` 复核（§8.3）。
- **UB 容量**：峰值 102.0 KiB（d=768）/ 142 KiB（d=1024）< 253,952 B，见 §4.5。
- **L0 容量**：无 Cube 计算，不适用。
- **GM workspace**：`PART[6, G, dw]` fp32 ≤ `6·56·1024·4 ≈ 1.38 MB`（d=1024 时）。

### 5.4 注意事项（非整除处理）

| 非整除情形 | 处理 |
|---|---|
| `RC ∤ seq_len`（如 `seq=401, RC=23` → 最后一块 10 个 token） | 末块用 `pds < seq_len` / `i` 范围的**均匀标量 guard**，越界行不载入、只写零槽（§0.6 R5）；块内 token 数自然变短，无 mask 张量 |
| `seq_len < 4`（`(2,1,256)`） | halo 全部越界 → 所有 tap 读零槽，`conv = 0`、`sigmoid(0)=0.5`、`d_silu = dY·0.5`、`d_vnorm = 0`（与语义一致：只有 `p=3` 的 tap 在 `t=0` 有效） |
| `d` 非 64 倍数（`d=192` 整除；`d=100` 不整除） | `dw` 上取整 + `update_mask` + 表尾清零；`/d` 用真实 `d` |
| 一个 batch 内 chunk 跨 batch 边界 | 不允许：`nch` 是**每 batch** 的 chunk 数，`b = bx // nch` 保证 chunk 不跨界 |
| `tokens < 56` | `RC=1`，grid = tokens（≤56），部分核空闲；halo 的绝对量 = 3 行/核 |

### 5.5 分核策略（物理核数适配）

> 三要素判定标准与实查要求：`.agents/skills/_shared/standards/core-split-strategy.md` §1（设计要求 §2.1）。

- **① 逻辑核数计算**：计算 kernel `num_logical_kernels = M · nch`，其中
  `nch = ceil(seq_len / RC)`、`RC = ceil(seq_len / min(seq_len, max(1, 56 // M)))`
  → 由构造保证 `M·nch ≤ 56`（§5.2 表：32/44/43/48/54/54/8/2）。归约 kernel
  `num_logical_kernels = 6 · tiles`，`tiles = ceil(dw / (64·ceil(dw/576))) ≤ 9` → ≤ 54。
- **② 物理核数依据（任务平台记录，非假设值）**：目标型号 `ascend-950pr-9579`（dav-3510）。
  平台上下文（`statectl platform --task-dir examples/engram_gate_conv_bwd/_engram_gate_conv_bwd_kernel`，
  2026-11-23 读取）记录 `hardware.observed_cores = {cube: 28, vector: 56}`，来源为
  `Ascend950PR_9579.ini [SoCInfo] ai_core_cnt=28, cube_core_cnt=28, vector_core_cnt=56`
  与运行时探针 `rtGetAiCoreCount()=28 on Ascend950PR_9579`。设备侧查询口径（平台 `core_evidence`
  要求的形式）为：

  ```python
  import torch, torch_npu
  p = torch.npu.get_device_properties(0)      # 记录 aicore/cube/vector 三项
  print(p.name, torch.npu.get_device_name(0))  # Ascend950PR_9579
  ```

  本次设计**未在该卡上重跑该查询**：卡被其他算子 Stage 3 占用（设计期只允许无卡编译/静态探针），
  故引用平台已登记的同型号实查记录（`design_budget`/`hardware.core_source`）。本算子为纯 Vector，
  取 **物理核数 = vector_core_num = 56**（Cube 的 28 不参与）。
- **对脚手架 K6 注记的复核（`sm_count` → `cube_core_num`(28) 是否有理）**：脚手架注记建议把 GPU 的
  `sm_count` 换成 `cube_core_num`（28）。**本设计判定该替代不合理**：本算子是纯 Vector 实现，全部计算
  与 3 处归约都在 AIV（Vector 核）上，Cube 核全程空闲；按 28 切分会砍掉一半并行度——以 train-s96 为例，
  `nch = min(96, 28//3 = 9)` → `RC = 11` → `grid = 27`（27/28 核），而按 56 得 `RC = 6`、`grid = 48`，
  每核行数 9 vs 15、halo 占比 50% vs 27%。故**物理核数取 `vector_core_num = 56`**，并进一步按 token 数
  重导 chunk 长度（`nch = min(seq_len, max(1, 56//M))`），而不是沿用「每 CTA 8 token」的固定 run 规则
  （该规则的目标是压原子争用，本设计无原子，目标是并行度与 halo 的平衡）。
- **③ 规模判定与分核方案**：**逻辑核数 ≤ 物理核数（56）→ 无需适配**。依据：几何规则
  `nch = min(seq_len, max(1, 56 // M))` 使 `M·nch` 上界恰为 56，任何 shape 都不会超发；
  每核独占一个 AIV、单波完成，不存在波次调度与负载不均（同一 batch 内各 chunk 仅最后一块可能短
  `≤ RC-1` 个 token）。归约 kernel 的 `6·tiles ≤ 54 ≤ 56` 同理。**不采用**「固定启动核数 = 物理核数 +
  核内串行多任务」也不需要「对齐物理核整数倍」的 block 调整——核数与核内 token 数已由上述两式
  同时确定，且 `grid ≤ 56` 是编译期常量。若未来出现 `M > 56` 的 shape（如 M=64），`max(1, 56//M)` 退化
  为 1、`nch=seq_len`，此时 `M·nch` 会超过 56 → 由硬件块调度器按波次派发（本设计未依赖跨块同步，
  波次安全），并列入 §9 风险。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|---|---|---|---|
| 核网格（token-chunk） | block 级并行 | `T.Kernel(grid)`，`grid = M·nch ≤ 56` | 每核一段连续 token，占满 56 个 AIV |
| token 推进（`i = 0 … RC+5`） | 核内串行 | `T.serial(RC + 2·KS - 2)` | 依赖距离 3 的卷积窗口（§1.6.2 #9）；已由在线化吸收到 4 槽窗口 |
| 行内 chunk（`ch = 0 … nchunks-1`） | 核内串行 | `T.serial(dw // 64)` | 两遍行算法（`dot` 先于使用）；每次迭代 = 一条 64 lane 向量指令（§1.6.2 #10） |
| conv tap（`p = 0 … 3`） | 展开 | `T.unroll(KS)` | 4 个 tap 是静态小环，展开后消除循环开销并固定累加顺序 |
| 4 个 chunk pass（A/B/C/D） | 顺序段 | `T.serial(nchunks)` × 4 | `dot_v → dvhat → dalpha → ddot` 的全行依赖链（§3.3） |
| partial 搬出 | 展开 | `T.serial(KS)` + `T.copy` | 6 行 workspace 写 |
| 归约核维（`row = 0 … G-1`） | 核内串行 | `T.serial(G)` | 归约顺序契约（§1.6.2 #13） |
| 归约列 tile | block 级并行 | `T.Kernel(6·tiles)` | 6 个量 × 列 tile |

### 6.2 循环伪代码

见 §3.3（与 `history_version/design_probe_compile.py` 一一对应）。要点复述：

```python
with T.Kernel(M * nch) as bx:                 # 单波，<= 56 个块
    for i in T.serial(RC + 6):                # 每步：1 个 nrm 行 + 1 个 ds 行 + 1 个 token 的 stage-2
        <load/fill nrm & dy slot (floormod)>  # 均匀标量 guard：in-batch 载入；否则写零槽
        if i >= 3: <conv 重算 + silu' -> ds slot ; if 拥有该行: pc 累加>
        if i >= 6: <4 个 chunk pass: gn -> gz/dv -> hn/kn+dot -> dH/dk>
```

### 6.3 流水线优化

**不使用 `T.Pipelined`**：滚动窗口是**单版本**缓冲（`annotate_buffer_versions({…: 1})`，语义要求同一槽
在 4 步内不被覆写），显式多缓冲会破坏窗口语义；跨 token 的数据依赖（`ds[t]` 被 `t..t+3` 消费）也使
「先搬后算」的双缓冲只能在单步内重叠（行搬入 vs 上一行的 SIMD 计算）。平台默认开启
`allow_autoschedule`（`AutoSchedule`/`InsertSync`/`MaterializeMultiBuffer`），行搬入与 SIMD 的重叠
由编译器自动调度负责；**是否真的重叠未实测**（无真机时段），列为 Stage 4 的调优项（§9）。

### 6.4 尾块处理

- `d` 尾 lane：`update_mask`（`MODE_ZEROING`）+ 表尾清零 + 输出只写 `[:d]`。
- `seq_len` 尾块：末 chunk 的 `i` 循环里越界行走「只写零槽」分支；`RC+6` 的静态上界保证不多算。
- `tokens < 56`：部分核空闲（grid < 56），无尾块问题。
- 归约 kernel 的列尾块：`cols = min(width, dw - off)` 动态 extent（950DT 已验证同形态）。

---

## 7. 同步策略

### 7.1 同步模式

**模式**: 自动同步（保留平台默认 `allow_autoschedule = True`，由 `InsertSync` 插入 GM↔UB 与
向量流水之间的同步）；**不写任何显式 `ascend_set_flag`/`ascend_wait_flag`/跨核 flag**。

### 7.2 同步点说明

| 位置 | 同步 API | 理由 |
|---|---|---|
| `T.copy`（GM→UB 行/表）后接 `T.SimdVF` 读同一缓冲 | 自动（`InsertSync` 插入 MTE2→V） | 编译器识别写后读；探针编译日志出现 `MTE2->V` 事件即为该同步 |
| `T.SimdVF` 写 UB 后接 `T.copy`（UB→GM） | 自动（V→MTE3） | 同上 |
| 4 槽窗口跨迭代复用（写后读达 4 步） | 自动 + `annotate_buffer_versions({…: 1})` | 单版本消除「多缓冲换版本」的隐式假设，保证窗口语义 |
| 两个 launch 之间（`PART` 的写→读） | kernel 边界（隐式全局可见） | 同一 `main` 内的两次 `tilelang.compile` 产物之间不存在跨核 flag 依赖 |
| 同一核内 4 个 chunk pass（共用 `gn/gz/hnf/knf`） | 自动（同 buffer 同核顺序读写） | pass 之间是同址顺序读写，不要求额外屏障 |

### 7.3 pass_configs 配置

```python
T.annotate_buffer_versions(
    {vrow: 1, drow: 1, hrow: 1, krow: 1, vvrow: 1, whs: 1, wvs: 1, cws: 1,
     nrmw: 1, dsw: 1, dyw: 1, whf: 1, wvf: 1, cwf: 1,
     pc: 1, ph: 1, pv: 1, gn: 1, gz: 1, hnf: 1, knf: 1, sc: 1}
)                                  # 全部单版本（滚动窗口 + 整行暂存的语义要求）
# 不开 annotate_unlimit_memory（容量充裕，§4.5）；不写 annotate_manual_multi_buffer；
# 不写 annotate_compile_flags（源算子的 "-O3 -DENABLE_BF16" 属 CUDA 后端）
```

---

## 8. 验证方案

### 8.1 Golden 函数

> 迁移任务：golden 以 **§0.1 源算子语义** 为唯一依据，**移植源仓参考实现**
> `ref_engram_gate_conv_bwd`（`examples/TileOPs/tileops/workloads/sequence_modeling.py`），
> **不复刻** §0.6 的 NPU 算法（无 staged、无窗口、无 per-core 部分和、无有序归约）。

```python
def golden_engram_gate_conv_bwd(
    dY, H, k, v, rms_w_h, rms_w_v, conv_w, vhat, alpha, rrms_h, rrms_k, rrms_v, eps=1e-6
):
    """fp32 autograd reference: differentiate the float32 forward over the six
    differentiable inputs.  ``vhat/alpha/rrms_*`` are the forward's saved values
    and are deliberately NOT used here (same as the source op's own oracle),
    which keeps this verification independent of any NPU implementation."""
    H_ag = H.float().detach().requires_grad_(True)
    k_ag = k.float().detach().requires_grad_(True)
    v_ag = v.float().detach().requires_grad_(True)
    w_h_ag = rms_w_h.float().detach().requires_grad_(True)
    w_v_ag = rms_w_v.float().detach().requires_grad_(True)
    cw_ag = conv_w.float().detach().requires_grad_(True)

    h_norm, _ = _rmsnorm(H_ag, w_h_ag, eps)          # x_f * rrms * w
    k_norm, _ = _rmsnorm(k_ag, w_h_ag, eps)
    dot = (h_norm * k_norm).sum(dim=-1, keepdim=True)
    alpha_ag = torch.sigmoid(dot / (H.shape[-1] ** 0.5))
    v_hat_ag = alpha_ag * v_ag
    v_hat_norm, _ = _rmsnorm(v_hat_ag, w_v_ag, eps)

    v_perm = v_hat_norm.permute(0, 2, 1)             # [M, d, T]
    v_padded = F.pad(v_perm, (conv_w.shape[0] - 1, 0))
    conv_out = F.conv1d(v_padded, cw_ag.T.unsqueeze(1), groups=H.shape[-1]).permute(0, 2, 1)
    y = F.silu(conv_out) + v_hat_ag

    with torch.autograd.set_multithreading_enabled(False):
        y.backward(dY.float())
    return (H_ag.grad.to(H.dtype), k_ag.grad.to(k.dtype), v_ag.grad.to(v.dtype),
            w_h_ag.grad, w_v_ag.grad, cw_ag.grad)


def _rmsnorm(x, w, eps=1e-6):
    x_f = x.float()
    rrms = (x_f ** 2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    return x_f * rrms * w.float(), rrms.squeeze(-1)
```

**独立性说明**：golden 不读 `vhat/alpha/rrms_*`（与源算子自身 oracle 一致），因此不会「用 NPU 的
中间结果证明 NPU 的结果」；它与 §0.6 的任何结构（窗口、per-core 部分和、有序归约）无关。
**已知的有意差异**：NPU kernel 用**前向保存的已舍入 vhat**（§0.1 契约），golden 走未舍入的 fp32
autograd 路径 → 二者的差已在 `verify_equiv.py` 中量化为基线自身相对 fp64 真值的
尺度误差 fp16 `8.9e-3` / bf16 `5.4e-2`（≪ 容差 0.2/0.3）；这一点必须写进测试注释，避免误判。

### 8.2 精度标准

| dtype | atol | rtol | 来源 |
|---|---|---|---|
| float16（`dH/dk/dv`） | 0.2 | 0.2 | ``_tol_bwd`（examples/TileOPs/tests/ops/test_engram.py 内）` |
| bfloat16（`dH/dk/dv`） | 0.3 | 0.2 | 同上 |
| float32（`drms_w_h/drms_w_v/dconv_w`） | 0.2 | 0.2 | 同上（manifest 对每个输出施加同一组界；3 个权重梯度的 fp32 实测相对差 ≤2.4e-6，余量 ≥4 个数量级） |

**权重梯度归约顺序的容差依据**（本算子的关键点）：3 个权重梯度跨 `tokens` 求和，
NPU 侧顺序 = 「核内 token 升序 + 跨核核号升序」（确定、可逐位复现）；源算子为 `T.atomic_add`（不确定）。
两种顺序的数学语义相同（加法结合律）；fp32 累加的相对误差最坏 `n·u`（`n ≤ 2406`, `u = 2^-24` → `1.4e-4`），
典型 `sqrt(n)·u ≈ 3e-6`，与 §7.2 的实测（`drms_w_h` 2.4e-6）一致，均在 0.2 的相对容差内（余量 >10^3）。
**确定性**：同一输入两次运行的 3 个权重梯度必须逐位相同（§8.3 的 L0 用例显式检查）。

### 8.3 L0 门槛测试计划

L0（`--level L0`，最小可判集；完整 L1/L2/Boundary 由 `tilelang-op-develop` 展开）：

| # | shape（M, seq_len, d, dtype） | 覆盖点 | 判据 |
|---|---|---|---|
| 1 | `(1, 32, 256, fp16)`（manifest smoke） | 主路径 + 6 个输出全比对 | `_tol_bwd` 逐输出 |
| 2 | `(3, 96, 768, bf16)`（最大 d） | 最大 UB / 最长 chunk（RC=6，halo 生效） | 同上 |
| 3 | `(2, 1, 256, fp16)` | `seq_len < 4`：halo 全越界、零槽路径 | 同上（`d_vnorm` 应为 0 → `dvhat = dY`，可另断言语义） |
| 4 | `(3, 401, 256, bf16)` | 素数 `seq_len` + 末块短尾 + 跨 batch 边界 | 同上 |
| 5 | `(1, 8, 256, fp16)` | 单 chunk（`nch=1`，grid=tokens） | 同上 |
| 6 | `(1, 16, 100, fp16)` | `d` 非 64 倍数（尾 lane 掩码、`/d` 用真实 d） | 同上 |
| 7 | 同 #2 连跑两次 | 3 个权重梯度的**逐位确定性** | `torch.equal`（或 bitwise 比较）为真 |

附加要求：`dconv_w/drms_w_h/drms_w_v` 额外与 `verify_equiv.py` 的 fp32 基线对照一次
（同输入、同保存态），确认归约分组变化的相对差 ≤ `1e-5`；`drms_w_v`/`dconv_w` 在 O3/O5 下应逐位相同（实测如此）。

---

## 9. 风险点与注意事项

### 9.1 技术约束检测结论

| 检测项 | 结论 | 处理 |
|---|---|---|
| 三维 Kernel（源算子在 Cube 后端常用） | **不涉及**：本平台 `T.Kernel` 只支持一维网格且拒绝 `threads=`（`references/programming.md`） | 一维 grid，任务索引由 `b = bx // nch`、`c = bx % nch` 自行解码 |
| L0C / L0A / L0B 容量 | **不涉及**：无 Cube 计算 | 不使用 `alloc_l0*` |
| GEMM 非整除（M/N/K tile） | **不涉及**：无 matmul | — |
| UB 容量（253,952 B） | 峰值 102.0 KiB（d=768）/ 142 KiB（d=1024） | d > 1024 走 §1.6.4 备选；refusal 需与 wrapper 对齐 |
| 跨 block barrier（未认证 `CG-950-batch-barrier`） | 权重梯度归约**不需要**跨块同步 | 2 次 launch + GM workspace |
| 高层二维 SIMD 归约（`TRAP-950-simd-shared-reduce`） | 已规避 | 显式寄存器归约 |
| `T.copy` 动态目的/源行、动态 extent | 探针已核验（编译） | 但**无真机数值验证** → L0 必须覆盖 |
| 缓冲多版本 | 全部单版本 | `annotate_buffer_versions` |
| 混合精度 cast | `ROUND_R` + `sal=False`（饱和关闭，与源 `T.cast` 一致） | 与 `PL-950-conversion-layout` 的转换结论一致 |

### 9.2 已知设计边界与风险

| # | 风险 | 影响 | 缓解 |
|---|---|---|---|
| R-1 | **主选 d 上界 1024** < manifest 声明的 `d ≤ 4096` | 若出现 `d ∈ (1024, 4096]` 的 workload，主选 UB 溢出（>253,952 B） | §1.6.4 备选 B1 覆盖 `1024 < d ≤ 4096`（52–76 B/channel，4096 端需 fp16 表变体）；**Stage 3/5 必须让 wrapper 的 `refusal` 与最终实现的上界一致**（不得声明支持却编译失败）。列为交付前必须确认项 |
| R-2 | `seq_len > 4096` 时标量条带 `[4, seq]` 超预算（16·seq B > 64 KiB） | UB 溢出 | build 期开关：`seq_len ≤ 4096` 用条带，否则回退「每 token 4 次标量 `T.copy`」（源算子形态，已验证）。当前 test/benchmark 的 `seq_len ≤ 512` |
| R-3 | 保存态用**已舍入 vhat**，golden 走未舍入 fp32 路径 | 基线自身相对 fp64 真值的尺度差 fp16 8.9e-3 / bf16 5.4e-2，占容差预算的可观份额（bf16 约 18%） | 已在 §8.1/§8.2 显式记录并在测试注释中说明；与源算子行为一致（源算子也用保存的 vhat），不改 |
| R-4 | 权重梯度求和顺序与源算子不同（源为原子乱序，本设计确定有序） | 末位差异；确定性更强 | 容差论证 §8.2（余量 >10^3）+ L0 #7 的确定性用例；**不追求与 GPU 逐位相同**（GPU 侧本身不可复现） |
| R-5 | 探针只做了**编译**验证（卡被占用，未跑真机） | 数值正确性未知 | L0 计划（§8.3）7 条用例；Stage 3 必须先跑 L0 |
| R-6 | 编译日志出现 `flag V->MTE3: reverse ordering too loose to bound flag reuse` 与 `MTE2->V` 同类告警（`(2,512,256)`、`(2,1,256)`、`(1,16,192)` 等几何） | 同步事件复用可能不足，理论上存在搬运与计算交叠的隐患（编译器告警，非错误） | Stage 3 首跑即查该告警对应的几何；必要时对该几何插入显式 `ascend_pipe_barrier` 或在 Stage 4 调优（不改变算法） |
| R-7 | `T.Pipelined` 未使用（滚动窗口单版本） | 行搬入与 SIMD 计算是否重叠未知（未实测） | Stage 4：在**不改变窗口语义**前提下评估 `T.serial` 内的搬运/计算重叠（例如把 `H/k/v` 三行搬入提前一步） |
| R-8 | Cube 核（28 个）完全空闲 | 理论上行点积可用 Cube 做 `1×d · d×T` 的 batched GEMM；本设计未用 | Stage 4 可选方向（混合模式需重新做 §1.6.3 布局决策与 §5 分核），不进本版 |
| R-9 | `M > 56` 时 `M·nch > 56` 会分波 | 波次调度，无跨块依赖故正确，但负载可能不均 | 几何规则已保证常见 shape 单波；若出现大 M，按 §5.5 的说明由调度器处理，必要时在 Stage 4 把 `nch` 压到 `56//M` 之外再对齐 |
| R-10 | `vexp` 在 `conv` 极大时的溢出行为 | `exp(-x)` → inf → `1/(1+inf) = 0`，`silu'` 极限正确（与源算子同式） | `verify_equiv.py` 的 `spike` 模式（`conv_w` × 40）已覆盖 |
| R-11 | 归约 kernel 的 `part[G, width]` 需要 `G` 行；`G` 为编译期常量 | shape 特化导致编译产物数量增加（每个 `(M,seq,d,dtype)` 一份，与源算子 `target_cache(32)` 同构） | 保持 `lru_cache(maxsize=32)` |

### 9.3 常见错误与处理

| 错误 | 触发场景 | 影响 | 解决方案 |
|---|---|---|---|
| UB 溢出 | `d > 1024` 或 `seq_len > 4096` | 编译失败 | 走备选 / 回退标量拷贝（R-1/R-2） |
| 窗口槽冲突 | 依赖距离算错（把 lead 写成 2 或 4） | 数值错误（读到过期/未写的槽） | §0.6 R2 的依赖距离论证（4 = 窗口深度）；L0 #3/#4 覆盖 |
| 重复计 `dconv_w` | halo 行（`pds ≥ t0+RC`）也累加 `pc` | `dconv_w` 偏大（约 `3/RC` 倍） | `if i < RC+3` 的 owner guard（§3.3） |
| 漏计 `dconv_w` | 末块短尾的 `pds ≥ seq_len` 也判为 owner | 无影响（该行 `ds=0` 且不写零槽以外的贡献） | 同 owner guard + `pds < seq_len` |
| `rm` 与 `rm` 共享 `rms_w_h` 写错 | 误用 `rms_w_v` 给 gate | 结果错误 | §0.1 语义表已显式标注「`H` 与 `k` 共用 `rms_w_h`」 |
| 尾 lane 参与行点积 | 忘记 `update_mask` 或表尾未清零 | 点积偏大 | §5.3 对齐约束 + L0 #6 |
| `T.copy` 维度不匹配 | 整 Buffer 对整 Buffer 拷贝要求元素数相等 | 编译报错 | 全部用显式 region（`[:d]`、`[:, off:off+cols]`） |
| 原子误用 | 想用 `T.atomic_add` 简化归约 | 未核验路径 + 顺序不定 | §0.6 R1 明确禁止 |

---

## 10. 交付清单

### 10.1 目录结构

```
examples/engram_gate_conv_bwd/_engram_gate_conv_bwd_kernel/
├── DESIGN.md                          # 本设计文档
├── verify_equiv.py                    # §1.6.1 等价性机器验证（D-1，可重跑：ALL_PASS）
├── history_version/
│   └── design_probe_compile.py        # D-3 无卡编译探针（7 几何 × 2 kernel）
├── RETROSPECTIVE.md                   # Stage 1 复盘（skill Phase 8）
└── _engram_gate_conv_bwd_kernel.py    # 待实现（Stage 3）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|---|---|---|
| `DESIGN.md` | 已完成 | 本设计文档（§0 迁移分析 + §1–§10） |
| `verify_equiv.py` | 已完成并执行 | `EQUIV_SUMMARY ALL_PASS`（5 项 × 2 dtype × 252 字段） |
| `history_version/design_probe_compile.py` | 已完成并执行 | `PROBE_RESULT: ALL_OK`（7 几何 × 2 kernel 编译通过） |
| `_engram_gate_conv_bwd_kernel.py` | 待实现 | 融合计算 kernel + 有序归约 kernel + golden + L0/L1/Boundary 分层测试 |
| `README.md` | 可选 | 使用说明 |

### 10.3 命名规范

- 项目目录：`examples/engram_gate_conv_bwd/`（由 conductor 指定）
- 算子目录：`_engram_gate_conv_bwd_kernel`（与 `manifest` 的 kernel 名一致）
- 实现文件：`_engram_gate_conv_bwd_kernel.py`

### 10.4 实现顺序

1. ✅ 设计文档 + 等价性机器验证 + 编译探针
2. ⬜ Golden 函数（§8.1，移植源仓参考实现）
3. ⬜ 融合计算 kernel（§3.3）+ 有序归约 kernel，接 golden 跑 L0
4. ⬜ L0 7 条用例（§8.3）+ 确定性用例
5. ⬜ （若需 d ∈ (1024, 4096]）备选 B1（§1.6.4）
