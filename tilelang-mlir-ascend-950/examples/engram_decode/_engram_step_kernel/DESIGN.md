# `_engram_step_kernel` 算子设计文档

> **迁移任务（harness 子模式），首次设计 v1**。源算子：
> `/workspace/TileOPs/src/tileops/_kernels/sequence_modeling/engram/decode.py` 的
> `_engram_step_kernel` 工厂（第 114–225 行）与它的 host 侧调用方
> `EngramDecodeKernel.forward`（第 375–410 行）。
>
> 同仓抽取件：`examples/TileOPs/tileops/kernels/sequence_modeling/engram_decode/` 下与
> `_engram_project_kernel` 同一文件的 step 段（Stage 3 产出）。
>
> 目标平台：`ascend-950pr-9579`（dav-3510，28 Cube / 56 Vector，UB 248 KiB / L1 512 KiB /
> L0C 256 KiB / L0A·L0B 各 64 KiB，cube_freq 1650 MHz），工具链 `ascend`
> （TileLang ascend 0.1.15 + CANN 9.2.1 本站，`target="ascend"`，
> `import tilelang.ascend.language as T`）。平台选择由 `statectl platform --task-dir` 落库
> （2026-11-23）。
>
> **设计范围**：仅 `_engram_step_kernel`——`EngramDecodeFwdOp` 的**第二个 launch**：
> 三个 RMSNorm / gate / 膨胀卷积 / SiLU / history 末行。第一个 launch
> `_engram_project_kernel`（两个投影 + cache 左移）是**已完成的另一个设计**
> （`examples/engram_decode/_engram_project_kernel/DESIGN.md`），本文只在 §0.1、§1.5、§4.2
> 说明**跨函数中间量契约**与边界（谁写 `kv`、谁做左移、末行归谁）。
>
> **只读参考（不照抄）**：`/workspace/TileOPs-ascend950/src/tileops/kernels/ascend/sequence_modeling/decode.py`
> 是 950DT 上已验证的 Ascend 实现（`_build_step`，第 126–298 行）。本文在 §0.4/§0.5 标注
> 借用了它的哪些手段、以及在 950PR 上的**适配差异与两处刻意偏离**（§0.5 #7 的 dtype 往返、
> §1.6.1 的 B1 项）；它的边界划分与本任务不同（把 cache shift 放进 step kernel），差异见
> §0.5 #11。
>
> 本文档由 `tilelang-ascend` → `tilelang-op-design` 流程产出：Phase M0（三问解读，§0.1–§0.4）
> → Phase R（算法调研，§1.6.0）→ Phase M1（耦合性判定与重设计，§0.5–§0.6）→ Phase 2
> （算法级优化，§1.6.1–§1.6.3）。设计期做了 1 组编译探针（§11.1，7 个形状变体，全部
> compile-only、`ASCEND_RT_VISIBLE_DEVICES=""`、未占用设备）与 1 个等价性机器验证脚本
> （§1.6.1）。首次设计，无「相对上一版的关键调整」。

---

## 0. 源算子解读与迁移分析

### 0.1 源算子语义（做什么）

`_engram_step_kernel(batch, d, d_padded, max_conv_len, conv_kernel_size, dilation, eps, dtype)`
是一个**工厂函数**：返回内层 JIT 工厂 `_func(threads)`，内层再专门化出一个 `@T.prim_func`。
它计算「一次 decode step 里所有沿 `d` 归约的部分」，每核一行 batch。数学上是 6 件事：

```
h_norm      = RMSNorm(h_t, rms_w_h)                       # 按行，分母 d
k_norm      = RMSNorm(kv[0], rms_w_h)                     # kv[0] 是 project 段的 fp32 输出
alpha       = sigmoid( dot(h_norm, k_norm) / sqrt(d) )    # 每行一个标量
v_hat       = alpha * kv[1]
vhat_norm   = RMSNorm(v_hat, rms_w_v)                     # 按行，分母 d
new_conv_state[:, mcl-1, :] = cast(vhat_norm, dtype)      # 只写末行（前 mcl-1 行归 project 段）
conv_out    = conv_w[w-1] * vhat_norm
            + Σ_{p<w-1} conv_w[p] * conv_state[:, mcl-(w-1-p)*δ, :]    # 膨胀因果卷积
y_t         = cast( silu(conv_out) + alpha * kv[1], dtype )
```

- **I/O 契约**（源码 `@T.prim_func main` 签名，第 140–149 行，8 个张量）：
  `h_t (batch, d_padded)` 输入 dtype、`kv (2, batch, d_padded)` **fp32**（跨 launch 中间量）、
  `conv_state (batch, max_conv_len, d_padded)` 输入 dtype、`rms_w_h/rms_w_v (d_padded,)`、
  `conv_w (w, d_padded)` 输入 dtype；输出 `y_t (batch, d_padded)` 与
  `new_conv_state (batch, max_conv_len, d_padded)`（只写第 `mcl-1` 行）。全部输出由调用方
  预分配（`out_idx=[]`，`forward` 第 401–407 行）。
- **规约语义**：**三个按行归约**（`Σ h²`、`Σ k²`、`Σ v̂²`）加**一个按行点积**
  （`Σ h_norm·k_norm`），归约轴 = 完整的 `d_padded` 维（每核一行，跨线程树形合并）。源实现
  用 `T.reduce_sum(frag2d, out, dim=1)`（第 173/180/187/195 行），累加器是 fp32。
  归约**按行独立**，batch 行之间无依赖，因此是「每行一个独立归约域」。
- **dtype 语义（关键）**：`k_local`/`v_local` 从 `kv` **直接以 fp32 读入**（第 167–168 行），
  中间量与标量全 fp32；**只在写回时** cast：`new_conv_state` 末行与 `y_t` 转输入 dtype
  （第 202、218–221 行）；`conv_state` 的 tap 读入即 cast 到 fp32（第 211–214 行）。
  `alpha = 1/(1+exp(-(dot/sqrt(d))))`（第 188 行）是显式写法，与 `torch.sigmoid` 同义。
  **源实现不做 `kv → dtype → fp32` 的往返舍入**——这是与 950DT 参考实现的实质差异
  （§0.5 #7）。
- **归一化分母是 `d` 而不是 `d_padded`**（第 174、181、196 行的 `/ float(d)`）：`d_padded > d`
  时填充列必须恰好为 0，它们的平方/乘积贡献 0，故「在 `d_padded` 域求和再除以 `d`」与
  「在 `d` 域求和再除以 `d`」等价。这一等价是 host 零填充契约的直接推论（§1.6.1 A0 机器验证）。
- **边界语义**：
  - **tap 索引**：`conv_state[bid, max_conv_len - (w-1-p)*δ, :]`，`p = 0..w-2`。由 shape rule
    `max_conv_len ≥ δ·(w-1)` 得索引 ∈ `[mcl-δ(w-1), mcl-1] ⊆ [0, mcl-1]`，**恒合法**（无越界，
    也不需要守卫——§1.6.2 #6 论证后删除）。`max_conv_len == δ(w-1)` 时 `p=0` 命中索引 0
    （`decode-s1` 的 `mcl=9, δ=3, w=4` 正好是该边界）。
  - **`w == 1`**（fixture 第 4 行 `max_conv_len=1, conv_kernel_size=1, dilation=1`）：tap 数为 0，
    `conv_out = conv_w[0]*vhat_norm`，`new_conv_state` 只有一行且就是本步写入的那行。
  - **partial cache（`L < max_conv_len`）**：kernel 侧无感知——host `forward` 先把
    `conv_state` 左填充到 `max_conv_len`（第 388–390 行），因此 kernel 永远看到满长 history；
    末行写入恒在第 `mcl-1` 行。
  - **非有限值**：无特殊处理，按 IEEE 传播（fp32 逐步计算 + 最后一次 cast）。
  - **`d_padded > d`**：最后一维由 host 0 填充（第 391–399 行），返回时裁回 `d`（第 408–409 行）。
- **host 侧语义**：`forward` 的两件纯元数据/填充动作（`conv_state` 左填充、6 个张量最后一维 0 填充
  到 `d_padded`）与返回 crop 都属于**实现手段**，NPU 侧沿用同一约定（§4.6）。

**语义保持基线**：§8.1 的 golden 以本节语义为唯一依据，直接调用源仓 torch 参考
`engram_decode_step_torch`（`examples/TileOPs/tileops/workloads/sequence_modeling.py`，
脚手架刚修正的 partial-cache 分支——恒返回 `max_conv_len` 行）。**golden 不复刻本设计的
NPU 算法**（不复刻因子提出、不复刻两遍结构、不复刻 reducer）。

### 0.2 源算子输入输出

| 参数 | 方向 | Shape | dtype | 说明 |
|------|------|-------|-------|------|
| `h_t` | in | `(batch, d_padded)` | fp16 / bf16 | 本步隐状态（host 已 pad 到 `d_padded`） |
| `kv` | in | `(2, batch, d_padded)` | **float32** | `kv[0]`=K、`kv[1]`=V；**project 段产出** |
| `conv_state` | in | `(batch, max_conv_len, d_padded)` | 同 `h_t` | 已左填充到 `max_conv_len` 的 history（只读） |
| `rms_w_h` | in | `(d_padded,)` | 同 `h_t` | h/k 共用的 RMSNorm 权重 |
| `rms_w_v` | in | `(d_padded,)` | 同 `h_t` | v̂ 的 RMSNorm 权重 |
| `conv_w` | in | `(conv_kernel_size, d_padded)` | 同 `h_t` | 深度可分离卷积权重（每 tap 一行） |
| `y_t` | **out** | `(batch, d_padded)` | 同 `h_t` | 本步输出（对外 crop 到 `d`） |
| `new_conv_state` | **out** | `(batch, max_conv_len, d_padded)` | 同 `h_t` | **只写第 `mcl-1` 行** |

源算子输出 shape（迁移必需字段）：`y_t = (batch, d_padded)`、`new_conv_state =
(batch, max_conv_len, d_padded)`，crop 后对外的 `d` 维为 `d`。

本 kernel 相关的 5 条 manifest workload（`EngramDecodeFwdOp`）+ 2 条 fixture 关注行
（`d_mem` 与本 kernel 无关，故不列）：

| workload | batch | d | d_padded | max_conv_len | conv_kernel_size | dilation | dtype | 归约域 `tile` |
|---|---|---|---|---|---|---|---|---|
| decode-s1（smoke） | 1 | 256 | 256 | 9 | 4 | 3 | float16 | 256 |
| serving-b1 | 1 | 256 | 256 | 12 | 4 | 3 | float16 | 256 |
| serving-b4 | 4 | 512 | 512 | 20 | 4 | 5 | float16 | 512 |
| serving-b8 | 8 | 256 | 256 | 18 | 4 | 3 | bfloat16 | 256 |
| decode-partial-cache | 2 | 256 | 256 | 12（L=5） | 4 | 3 | float16 | 256 |
| （fixture）batch-17 | 17 | 256 | 256 | 12 | 4 | 3 | float16 | 256 |
| （fixture）one-tap | 1 | 256 | 256 | 1 | **1** | 1 | float16 | 256 |

### 0.3 实现算法解读（怎么算）

**计算步骤分解**（源码 `decode.py:150-221`，每个计算语句归入下列 10 步之一）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式的部分 |
|------|------|------|------|-------------------|
| 1 | `k_local/v_local ← kv[0/1, bid, j]`（fp32 直读）、`h_local ← cast(h_t[bid,j])` | GM `kv`,`h_t` | fragment `k/v/h_local` | 三路原始行 |
| 2 | `hsq = h·h`；`T.reduce_sum(dim=1)`；`rrms_h = rsqrt(Σh²/d + eps)` | `h_local` | 标量 `rrms_h` | `RMSNorm(h_t)` 的统计量 |
| 3 | `h_local ← h_local · rrms_h · rms_w_h` | `h_local`,`rrms_h` | `h_norm`（就地） | `h_norm` |
| 4 | `ksq = k·k`；reduce；`rrms_k = rsqrt(Σk²/d+eps)`；`k_local ← k·rrms_k·rms_w_h` | `k_local` | 标量 + `k_norm` | `RMSNorm(kv[0])` |
| 5 | `hk = h_norm · k_norm`；reduce → `dot`；`alpha = 1/(1+exp(-(dot/√d)))` | `h_norm`,`k_norm` | 标量 `alpha` | gate |
| 6 | `vhat ← alpha·v` | `v_local`,`alpha` | `vhat` | `v_hat = α·V` |
| 7 | `vsq = vhat·vhat`；reduce；`rrms_v = rsqrt(Σv̂²/d+eps)`；`vhat_norm ← vhat·rrms_v·rms_w_v` | `vhat` | 标量 + `vhat_norm` | `RMSNorm(v_hat)` |
| 8 | `new_conv_state[bid, mcl-1, j] ← cast(vhat_norm)` | `vhat_norm` | GM（末行） | history 末行 |
| 9 | `conv_out ← conv_w[w-1]·vhat_norm`；`p = 0..w-2`：`conv_out += conv_w[p]·cast(conv_state[bid, idx_p, j])` | `vhat_norm`,`conv_w`,`conv_state` | `conv_out` | 膨胀因果卷积 |
| 10 | `y_t[bid,j] ← cast(conv_out·sigmoid(conv_out) + alpha·v)` | `conv_out`,`alpha`,`v` | GM `y_t` | SiLU + 残差 |

块内无 K/M/N 方向的分块循环（每核一行、行内满宽），所以「分块」只体现在 `T.Kernel(batch)`
的核间划分上。`threads` 只是启动参数（默认 `step_threads=128`，第 327 行），不改变算法。

**数据流与内存访问模式**（源硬件视角，GPU）：

```
GM[h_t] ──T.Parallel──▶ fragment[h_local]  ─┐
GM[kv[0]](fp32) ──────▶ fragment[k_local]   ├─reduce(dim=1)─▶ 4 个标量(rrms_h/rrms_k/dot/alpha/rrms_v)
GM[kv[1]](fp32) ──────▶ fragment[v_local]   ┘                        │
GM[rms_w_h], GM[rms_w_v] ──T.copy──▶ fragment[w_h/w_v]                │（标量广播回行内）
GM[conv_w] ──T.copy──▶ fragment[cw] ────────────────┐                 ▼
GM[conv_state] ──T.Parallel──▶ fragment[conv_state tap] ──▶ fragment[conv_out] ◀── vhat_norm
                                                              │
                                            ┌─────────────────┴──▶ GM[new_conv_state][:, mcl-1, :]
                                            └────────────────────▶ GM[y_t]
```

全部中间量驻留 fragment（不落 GM）；`d` 行只从 GM 读一次。

**循环与并行结构**：grid = `batch`（1 维 `T.Kernel(batch)`）；block 内 `threads` 个线程，
每个 `T.Parallel(d_padded)` 元素级映射；4 次 `T.reduce_sum(dim=1)` 跨线程归约；
`for p in T.serial(w-1)` 的 tap 累加循环。默认 `threads=128`。

**host 侧逻辑**：`forward` 的 `conv_state` 左填充（对齐 `max_conv_len`）、6 张量最后一维 0 填充
到 `d_padded`、返回 crop（§0.1 末条）。`kv`/`y_t`/`new_conv_state` 由 host 预分配。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| O1 | **整行驻留 fragment**（`k/v/h/vhat/conv_out` 全在 fragment，不落 GM） | `d` 只读一次、中间量不往返 | 寄存器/fragment 常驻 | 大寄存器堆 | 硬件强相关（容量与映射），意图可移植 |
| O2 | `T.reduce_sum(frag2d, out, dim=1)` 行归约 | 4 个统计量 | warp/block 树形归约 + fragment | warp shuffle + SMEM | 硬件强相关 |
| O3 | 标量广播回行内（`h_local[j] * rrms_h * w[j]`，第 176 行） | 避免二次扫描 | 标量→向量广播 | 标量/向量混合流水 | 模式可移植 |
| O4 | 每核一行（`T.Kernel(batch)`），batch 行间零通信 | 归约域不跨核 | 独立 block | CUDA block 级隔离 | 模式可移植（NPU 同样 1 核 1 行） |
| O5 | 把 4 个统计量与元素级链**分开成串行 pass**（源码逐 pass 写法） | 每 pass 语义单一、寄存器压力低 | 顺序 pass | 无（算法结构） | 可移植（**本设计重新评估，见 §1.6.0/§1.6.1**） |
| O6 | tap 用 `for p in serial(w-1)` 累加 | 卷积 = (w-1) 次 SAXPY | 循环 | 无 | 可移植（可 Python 展开为一次 pass） |
| O7 | `alpha·v` 在残差与 `vhat` 中各算一次（第 191、219 行） | 无（源码未做 CSE） | — | 无 | 可移植（本设计做 CSE，§1.6.1 A4） |
| O8 | `rsqrt(x/d + eps)` 代替 `1/sqrt(...)`（第 174/181/196 行） | 省除法 | SFU rsqrt | SFU | 可移植（保留） |
| O9 | `threads=128` 启动参数（`step_threads`） | 行宽 256/512 时每线程 2–4 元素 | CUDA block 线程数 | — | 可舍弃（NPU 用 `T.SimtVF(threads=)` 表达，语义不同） |
| O10 | （来自 950DT 只读参考，非源算子）`T.SimtVF(threads=min(512,d_pad))` + `alloc_reducer` 四段 epoch；tap 预取到 UB 后按 p 取用；跨步 UB 读 | 用 SIMT reducer 表达行归约与广播 | SIMT 线程域 + reducer | Ascend AIV 线程域 | 硬件强相关（来源：950DT `_build_step`，§0.5 备注） |
| O11 | （来自 950DT 只读参考）`kf[i] ← cast(cast(kf[i], dtype), fp32)`、`vhat` 同款往返（其第 220、247 行） | 让其测试对齐某个 GPU 版本 | 两次 cast | 无 | **可移植但不采纳**：与 §0.1 语义冲突（§0.5 #7 / §1.6.1 B1） |

> 识别不出来 ≠ 不存在。O1–O11 是本文后续判定的全集；O10/O11 属于「目标平台已验证实现」的手段
> （不是源算子自带的），列在此处是为了在 §0.5 明确它们与 O1–O9 的**替代关系**而不是额外叠加。

### 0.5 硬件耦合性分析与 NPU 适配决策

**判定问题**：实现算法和优化手段是硬件强相关吗？能用在 NPU 上吗？

| # | 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|---|------|------|-----------|-----------------|------|---------------------|
| 1 | 数学语义（3 个 RMSNorm + gate + 膨胀卷积 + SiLU + 末行写），含 fp32 中间量与归一化分母 `d` | 语义 | 无 | — | **保留** | 语义层无条件保留（§0.1） |
| 2 | O1 整行驻留 fragment | 优化 | 大寄存器堆 | 有（UB 248 KiB + `local.fragment` 每线程若干寄存器） | **等价替换** | 一行读入 UB，元素级中间量放 `T.alloc_fragment`（探针生成的 `float vf[1]`/`float vf[4]` 即寄存器） |
| 3 | O2 `T.reduce_sum(dim=1)` 行归约 | 优化 | warp shuffle + SMEM | 部分（**高层 SIMD 二维 shared→shared 归约有已知输出布局缺陷**） | **重新设计** | 改 `T.alloc_reducer(op="sum")` + `reducer_init/reducer_update/finalize_reducer`（§0.6 R2）；证据 `PL-950-row-primitives`、`TRAP-950-simd-shared-reduce` |
| 4 | O3 标量广播回行内 | 优化 | 无 | 有 | **保留** | reducer 的 `dst` fragment 在元素级表达式中直接读 `alpha[0]`（`PL-950-row-primitives` 已验证「每个输出元素读取同一已完成行统计量」） |
| 5 | O4 每核一行、行间零通信 | 算法 | CUDA block | 有（`T.Kernel(batch)` 1 维网格） | **保留** | 1 逻辑块 = 1 batch 行（§5.2）；`batch ≤ 17` |
| 6 | O5「每统计量一个 pass」的逐 pass 结构 | 算法 | 无 | 有 | **重新设计** | 融合为 2 个 pass（§0.6 R1、§1.6.1 A1–A4）：源结构 15 个元素级 pass，本设计 2 个 |
| 7 | O11 `kv/vhat → dtype → fp32` 往返舍入 | 优化 | 无 | 有 | **舍弃** | 与 §0.1 语义不符（参考实现直接对 fp32 `kv` 归一化）；机器验证（§1.6.1 B1）显示往返使写出的 dtype 值仅 0.39–76 % 逐位一致、最大绝对偏差 1.56e-02——是**语义偏差**而非优化 |
| 8 | O6 tap 串行累加 | 算法 | 无 | 有 | **保留（结构内联）** | `for p in range(w-1)` 编译期展开进同一次元素级 pass（`w` 是编译期常量） |
| 9 | O7 `α·v` 重复计算 / O8 rsqrt | 优化 | 无 | 有 | **等价替换** | CSE 成一次 `av = α·v`（A4）；rsqrt 保留（A6 只把 `x/d` 换成常量倒数） |
| 10 | O9 `threads=128` 启动参数 | 优化 | CUDA block 线程数 | 无（`T.Kernel` 无 `threads`）；但 `T.SimtVF(threads=)` 提供线程域 | **等价替换** | 线程域改由 `T.SimtVF(threads=min(512,tile))` 表达（§5.6） |
| 11 | 950DT 参考把 **cache 左移**放进 step kernel | 优化 | 无 | 有 | **舍弃（本任务边界不允许）** | harness 的函数边界把 `new_conv_state[:, 0:mcl-1, :]` 归 `_engram_project_kernel`（已冻结的设计 §1.5 契约）；本 kernel 只写末行。若照搬参考会在两个 launch 里写同一区域（幂等但翻倍流量） |
| 12 | 950DT 参考的 O10 四段 reducer epoch（每统计量一段） | 优化 | AIV 线程域 | 有 | **重新设计（收敛为 1 段并发 4 reducer）** | 探针证明 4 个 reducer epoch 可并发于同一 `T.Parallel` 循环，生成 4 条 `AscendAllReduce<SumOp, ...>`（§11.1 S1）→ 一次扫描出 4 个统计量 |
| 13 | 950DT 参考的 `_pick_chunk_rows` / `shift_ub` / `_UB_BUDGET_BYTES` 反解 | 优化 | 其 UB 预算 | 不需要 | **舍弃** | 那套算术只服务于它挪进 step 的平移；本 kernel 无平移（#11） |

> 每项处置的依据均指向 `.agents/platforms/toolchains/ascend/references/programming.md` 的接口条款、
> `.agents/platforms/toolchains/ascend/knowledge/` 的已验证条目、或 §11.1 的设计期探针结果。
> 判定为「重新设计」的 3 项（#3/#6/#12）在 §0.6 逐项给出 NPU 新算法与语义保持论证。

### 0.6 NPU 算法重设计

**重设计项 R1：逐 pass 结构 → 两遍融合结构（承接 O5 / §0.5 #6）**

- **源方案**：每个统计量与每条元素级链各占一个 pass（`hsq` → reduce → scale → `ksq` → reduce →
  scale → `hk` → reduce → `vhat` → `vsq` → reduce → scale → 写末行 → conv 初始化 →
  `w-1` 个 tap pass → `y`），`w=4` 时 **15 个元素级 pass + 4 次行归约**。
- **NPU 新算法**：**phase 1 一次扫描**同时产出 4 个统计量（`Σh²`、`Σk²`、`Σ h·k·w_h²`、`Σv²`），
  **phase 2 一次扫描**同时产出 `vhat_norm`（写末行）、`conv_out`、`y_t`；中间只夹一条**标量链**
  （`rsqrt`/`exp`/`mul`，每行约 10 个标量运算）。
- **语义保持论证**：每个统计量的加法链与源**逐项同序**（同一 `j` 序遍历同一批项，只是 4 条独立链
  交错执行——A1 机器验证 ulp 差 0）；phase 2 的元素级表达式逐项对应源的第 197–221 行
  （`vn = av·rrms_v·w_v`、`co = cw[w-1]·vn + Σ_p cw[p]·tap_p`、`y = co·σ(co) + av`）。**精确等价**
  到 fp32 舍入级别（§1.6.1 A2–A4 给出逐项 ulp 实测）。
- **收益量化**：元素级 pass 15 → 2（−13），按 `CONST-vector-launch-overhead`（910B2C 实测
  ~0.5 µs/op；950PR 未实测）估算省 ≈ 6.5 µs，与一次 kernel launch 同量级（§1.6.0 R3）。

**重设计项 R2：行归约 → SIMT reducer epoch（承接 O2 / §0.5 #3）**

- **源方案**：`T.reduce_sum(frag2d, out, dim=1)`（二维 fragment → 一维标量）。
- **NPU 新算法**：`acc = T.alloc_reducer((1,), "float32", op="sum")` → `reducer_init(acc)` →
  在 `T.Parallel(tile)` 内 `reducer_update(acc[0], expr)` → `finalize_reducer(acc, dst)`，
  `dst = T.alloc_fragment((1,), "float32")`。
- **语义保持论证**：reducer 的语义定义是「每个逻辑迭代恰好贡献一次，编译期选择物理副本与
  跨线程通信方案，逻辑贡献次数不因物理布局改变」（`tilelang/language/reduce_op.py` 的
  `alloc_reducer`/`reducer_update` 文档原文语义）。因此结果 = 全部 `tile` 个贡献之和，
  与源 `reduce_sum(dim=1)` 同义；差异只在树形合并顺序（fp32 下 ≤ 数个 ulp，§1.6.1 A3 实测 2 ulp）。
- **平台依据**：`PL-950-row-primitives`（sum/max reducer + 广播，24 项通过）、
  `PL-950-normalization`（带 affine 的 RMSNorm，fp16/bf16/fp32 六项通过）为同平台已验证形态；
  **不用**高层 SIMD 二维 shared→shared 归约（`TRAP-950-simd-shared-reduce` 的输出布局缺陷）。
- **收益量化**：4 次 reducer 各自生成 1 条 `AscendAllReduce<SumOp, L, 1, 0>`（`L` = 线程数；探针实读），
  每核 4 次归约、无跨核通信。

**重设计项 R3：4 段 reducer epoch → 1 段并发 4 reducer（承接 O10 / §0.5 #12）**

- **源方案（950DT 参考）**：`acc` → init → update → finalize → `hf` 缩放 → `acc_k` → … → `acc_v`，
  四段串行，每段一次独立扫描（4 个 pass）。
- **NPU 新算法**：4 个 `alloc_reducer` 的 `init` 连排，4 条 `reducer_update` 落在**同一个**
  `T.Parallel(tile)` 循环体内，4 个 `finalize_reducer` 连排。探针 S1 的生成源码为证：
  一行内 4 条 `acc_*[0] = acc_*[0] + …`，随后 4 条 `AscendAllReduce<tl::SumOp, 512, 1, 0>::run`
  （各自独立的 UB scratch 槽 2816/3328/3840/4352 float）。
- **语义保持论证**：4 个 reducer 相互独立，各自 epoch 内贡献顺序不变 → 每个统计量与
  「4 段串行」版本逐位相同（A1 验证 0 ulp）。**边界**：4 个 scratch 槽各 `tile·4 B`
  （`serving-b4` 为 8 KiB），已计入 §4.5 UB 预算。
- **收益量化**：4 次行扫描 → 1 次；UB scratch 4×2 KiB（`d_pad=512`）。

**重设计项 R4：`x / d`、`dot / sqrt(d)` → 常量倒数乘（承接 O8/O9 / §0.5 #9）**

- **源方案**：`sumsq / float(d)`、`dot / T.sqrt(float(d))`——每行 4 次除法（3 个 `rrms` + 1 个 alpha 入参）。
- **NPU 新算法**：`d` 是编译期常量，host 侧算 `inv_d = 1/d`、`inv_sqrt_d = 1/sqrt(d)` 交给
  kernel 作 fp32 字面量：`rsqrt(sumsq * inv_d + eps)`、`sigmoid(dot * inv_sqrt_d)`。
  探针生成的常量印证：`float(0x1p-9f)`（= 1/512，**精确**）与 `float(0x1.6a09e667f3bccp-5f)`
  （= 1/√512）。
- **语义保持论证**：`d = 2^k`（本算子 5 条 workload 的 `d ∈ {256, 512}` 全是 2 的幂）时
  `x·(1/d) ≡ x/d` **逐位相同**（乘以 2 的幂只改指数）；`1/√d` 不可精确表示，误差 ≤ 1 ulp
  （A6 实测最大 2 ulp，写出的 dtype 值 99.95 % 逐位一致、最大绝对偏差 9.54e-07 ≪ atol=0.05）。
- **收益量化**：每行省 4 次标量除法（标量级，收益小；此项按 §1.6.1 的「四要素」保留证据，
  不夸大收益）。

**重设计项 R5：`threads=128` → `T.SimtVF(threads=min(512, tile))`（承接 O9 / §0.5 #10）**

- **源方案**：`threads=128` 的 CUDA block，每线程处理 `d_padded/128` 个元素。
- **NPU 新算法**：`T.SimtVF(threads=L)`，`L = min(512, tile)`；`T.Parallel(tile, coalesced_width=1)`
  让每 lane 一个元素（`tile = 512` → 512 lane；`tile = 256` → 256 lane）。
- **语义保持论证**：线程数是实现参数，不改变任何输出的数学定义（只改变归约树的形状——见
  §1.6.1 A3 的 2 ulp 实测）。`threads=128` 变体同样编译通过（探针 S1 第五变体），每线程 4 元素、
  生成 `#pragma unroll for (i<4)` + 4 次串行累加——这会把树的形状从「512 lane 一次」变成
  「128×4 两段」，差异同样在 ulp 级。
- **收益量化**：`d_pad=512` 时 512 lane 摊满一行；`threads` 与 reducer 树高的折中列入 §1.6.3 的
  Stage 4 A/B 清单（{128, 256, 512}）。

### 0.7 标杆实现

- **源算子路径**（golden 的语义出处）：`/workspace/TileOPs/src/tileops/_kernels/sequence_modeling/engram/decode.py`
  第 114–225 行（`_engram_step_kernel`）与第 375–410 行（host `forward`）。
- **源仓参考实现（golden 移植对象）**：`examples/TileOPs/tileops/workloads/sequence_modeling.py`
  的 `engram_decode_step_torch` 与 `_rmsnorm_decode`；本 kernel 对应其第 3 段之后的所有计算
  （`k/v` 投影之后），并且**直接调用它**而不是复制公式（§8.1）。
- **只读实现参考（不作为 golden、不照抄）**：
  `/workspace/TileOPs-ascend950/src/tileops/kernels/ascend/sequence_modeling/decode.py` 的
  `_build_step`（SIMT + reducer + UB 预取）。借用点（O10/O12）与两处刻意偏离（O11 的 dtype
  往返、cache 平移的归属）已在 §0.4/§0.5 标注。
- **同族兄弟算子设计（同平台、同 op family，只读）**：
  `examples/engram_gate_conv/_engram_gate_conv_fwd_kernel/DESIGN.md`——它选了 SIMD
  （`T.SimdVF` + `vcadd`）。本文 §2.2 给出「为什么本算子仍以 SIMT 为主选」的量化对照，并把
  SIMD 形态作为**备选方案 + Stage 4 裁决计划**冻结进 §1.6.3。

---
## 1. 概述

### 1.1 算子名称

`_engram_step_kernel`（Engram 单步 decode 的 Vector 段：三个 RMSNorm + gate + 膨胀卷积 + SiLU +
history 末行；`EngramDecodeFwdOp` 的第 2 个 launch）。

### 1.2 功能描述

对每个 batch 行做「沿 `d` 归约」的全部计算：把 `h_t`、projection 段产出的 `kv[0]`/`kv[1]`
分别归一化，用归一化后的 h/k 点积得到 gate 标量 `alpha`，用它缩放 V 再归一化，写
`new_conv_state` 的末行，然后把 `w` 个 tap 的膨胀因果卷积与 SiLU + 残差算出 `y_t`。
一次 kernel 启动、每核一行、行内两遍扫描。

### 1.3 数学公式

$$
h_n = \frac{h_t}{\sqrt{\frac1d\sum_j h_t[j]^2+\epsilon}}\odot w_h,\qquad
k_n = \frac{k}{\sqrt{\frac1d\sum_j k[j]^2+\epsilon}}\odot w_h,\qquad
\alpha = \sigma\!\left(\frac{\sum_j h_n[j]\,k_n[j]}{\sqrt d}\right)
$$

$$
\hat v = \alpha\, v,\qquad
v_n = \frac{\hat v}{\sqrt{\frac1d\sum_j \hat v[j]^2+\epsilon}}\odot w_v
$$

$$
c[j] = w_{\text{conv}}[w-1,j]\,v_n[j] + \sum_{p=0}^{w-2} w_{\text{conv}}[p,j]\;\text{state}\!\left[b,\;mcl-(w-1-p)\delta,\;j\right]
$$

$$
y_t[b,j] = \text{cast}\!\left(\frac{c[j]}{1+e^{-c[j]}} + \alpha\,v[b,j]\right),\qquad
\text{new\_conv\_state}[b,mcl-1,j] = \text{cast}(v_n[j])
$$

其中 `b < batch`、`0 ≤ j < d_padded`、`v = kv[1, b, :]`、`k = kv[0, b, :]`（fp32），
`σ` 为 sigmoid，`δ = dilation`、`w = conv_kernel_size`、`mcl = max_conv_len`；三个归约的求和域是
完整的 `d_padded`（填充列为 0，除以 `d`）。`w = 1` 时求和项为空。

### 1.4 算法描述

**（迁移决策后的 NPU 侧算法，与 §1.6.0 选定的 C1 一致）** 一个逻辑块负责一个 batch 行，
行内只有两遍元素级扫描（§0.6 R1）：

1. **任务分解**（§0.5 #5）：`T.Kernel(batch)` → 块 `bx` 处理第 `bx` 行；`tile = next_pow2(d_padded)`
   是该块的行宽（fragment 的宽度）。
2. **预取**：`h_t[bx,:]`、`kv[0/1,bx,:]`、`rms_w_h`、`rms_w_v`、`conv_w`、以及 `w-1` 个 tap 行
   （`conv_state[bx, mcl-(w-1-p)δ, :]`）全部 GM→UB，**每个张量只读一次**；编译器把它们批量排到
   一个 MTE2 段（§11.1）。
3. **phase 1（一次扫描，4 路归约）**：`T.Parallel(tile)` 内读 UB，同时向 4 个 reducer 贡献
   `h²`、`k²`、`h·k·w_h²`、`v²`（§0.6 R3、§1.6.1 A1/A2）。
4. **标量链**（每行一次）：`rrms_h = rsqrt(Σh²·inv_d+eps)`、`rrms_k` 同、
   `dot = rrms_h·rrms_k·Σ h k w_h²`（A2）、`alpha = 1/(1+exp(-dot·inv_sqrt_d))`、
   `Σv̂² = alpha²·Σv²`（A3）、`rrms_v = rsqrt(Σv̂²·inv_d+eps)`。
5. **phase 2（一次扫描，3 路输出）**：`av = alpha·v`（A4）→ `vn = av·rrms_v·w_v` →
   写 `new_conv_state[bx, mcl-1, :]`；`c = cw[w-1]·vn + Σ_{p<w-1} cw[p]·tap_p`；
   `y = cast(c/(1+exp(-c)) + av)` → 写 `y_t`。
6. **写回**：`y_ub → y_t[bx,:]`、`vn_ub → new_conv_state[bx, mcl-1, :]` 两条 UB→GM DMA。

**与源算法的差异（逐条）**：15 个元素级 pass → 2 个（R1）；4 次 `reduce_sum(dim=1)` →
4 个并发 reducer epoch（R2/R3）；`x/d` → 常量倒数（R4）；`threads=128` → `T.SimtVF(threads=L)`（R5）；
`α·v` 的 CSE（A4）；tap 守卫删除（索引范围由 shape rule 保证，§1.6.2 #6）。
**语义差异：无**——全部为 §1.6.1 机器验证过的等价变换（含 B1 这类**被拒绝**的语义偏差项）。

### 1.5 数据流图

```
GM[h_t](dtype)        --MTE2--> UB[h_ub]   --\
GM[kv[0],kv[1]](fp32) --MTE2--> UB[k_ub,v_ub] --\
GM[rms_w_h,rms_w_v]   --MTE2--> UB[wh_ub,wv_ub] --+--> [phase 1: T.Parallel(tile) + 4 reducer] --> 4 标量
GM[conv_w]            --MTE2--> UB[cw_ub]   --/                     |
GM[conv_state] taps   --MTE2--> UB[tap_ub]  --/                     v
                                                          标量链(rsqrt/exp/mul, 每行 1 次)
                                                                    |
        ┌───────────────────────────────────────────────────────────┘
        v
[phase 2: T.Parallel(tile)] --▶ UB[vn_ub] --MTE3--> GM[new_conv_state][:, mcl-1, :]
                            --▶ UB[y_ub]  --MTE3--> GM[y_t]
```

跨 launch 契约（两个 launch 之间只有 GM；本 kernel 是 step 段，只读 `kv`、只写末行）：

| 中间量 | 生产方 | 消费方 | shape / dtype | 边界 |
|--------|--------|--------|---------------|------|
| `kv` | `_engram_project_kernel` | **本 kernel（只读）** | `(2, batch, d_padded)` **fp32** | 本 kernel 直接以 fp32 读入并归一化，**不做 dtype 往返**（§0.5 #7） |
| `new_conv_state[:, 0:mcl-1, :]` | `_engram_project_kernel` | 对外输出 | `(batch, mcl-1, d_padded)` 输入 dtype | **本 kernel 不读不写**；L0 用哨兵断言证明未被触碰（§8.3） |
| `new_conv_state[:, mcl-1, :]` | **本 kernel（唯一写者）** | 对外输出 | `(batch, d_padded)` 输入 dtype | 整行覆盖 `[0, d_padded)`；`mcl == 1` 时它就是第 0 行 |
| `conv_state` | host（已左填充到 `mcl`） | 本 kernel（读 `w-1` 个 tap 行） | `(batch, mcl, d_padded)` | 只读；`h_t`/权重同理 |

### 1.6 算法调研与优化分析

#### 1.6.0 算法调研（Algorithm Research）

调研对象 = 「同一数学语义（3 个按行 RMSNorm + gate + w-tap 膨胀因果卷积 + SiLU + history 末行）」
的算法族。**基线候选 = 源算子（GPU TileLang：逐统计量 pass、`reduce_sum(dim=1)`、
`threads=128`）**，它只是候选之一。

**R1 等价化简公式候选**（正式等价论证与收益量化在 §1.6.1，此处只做初判）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 是否纳入 R3 对比 |
|---|------|------------|-----------|---------|----------------|
| 0 | **基线：源算子** | 15 个元素级 pass + 4 次 `reduce_sum(dim=1)`；`x/d` 与 `dot/√d` 各一次除法 | — | — | ✅（基线） |
| 1 | 两遍融合 + 因子提出 | phase 1 一次扫描出 4 统计量；phase 2 一次扫描出末行 + 卷积 + y；`dot = rrms_h·rrms_k·Σhk w²`、`Σv̂² = α²Σv²` | 实数域恒等，fp32 下仅舍入路径变化 | 元素级 pass 15 → 2 | ✅（选定） |
| 2 | 保守逐统计量（950DT 参考形态） | 每统计量一段 reducer epoch，元素级链逐 pass | 与基线逐位同构 | 无（pass 数不变，只换归约原语） | ✅ |
| 3 | SIMD 寄存器路径 | `T.SimdVF` + `vld/vmul/vadd` + `vcadd`，chunk 循环 + 寄存器累加器 | 实数域恒等，累加分组变为「chunk 串行 + 64 lane 树」 | 归约指令更少；但需显式 mask/分布参数 | ✅ |
| 4 | 分 `d` 跨核 + 跨核归并 | 把行切成 `nc` 段分给 `nc` 个核，部分和再归并 | 恒等（累加分组变化） | 并行度 ×nc；代价 = 跨核 barrier 或 +1 launch | ✅ |
| 5 | 与 project 段融合成单 launch | 每核一行算完整的 `kv[b,:]`（GEMV）后直接接 step | 恒等 | 省 1 次 launch；但权重被读 `batch` 遍、Cube 退化为 M=1 | ✅ |
| 6 | 深度卷积走 Cube | 把 `Σ_p cw[p]·tap_p` 表达成矩阵乘 | **不成立**：深度卷积的权重矩阵是对角的（每列一个独立 `w`-元组），没有可收缩的公共轴 | 无 | ✅（负向，附结构依据） |
| 7 | 两遍 materialize | 先把 `vhat_norm` 与 4 个标量写 GM，第二 kernel 读 tap | 恒等 | 负收益 | ✅ |
| 8 | 数值稳定化变形（softmax 式减 max） | 对 `alpha` 的 sigmoid 入参或 `conv_out` 减 max | `c/(1+e^{-c})` 在 `|c| ≤ 65504` 时 `e^{-c}` 只在下溢侧丢精度（下溢到 0 → 结果 0，与真值差 < 1 ulp of fp16）；减 max 需多一次行归约 | 无（净增一次归约） | ✅（否决，见 R4） |
| 9 | 求和精度提升（Kahan / 分块补偿） | 对 3 个平方和做补偿求和 | 恒等且更准 | 每项 +2 次运算；本算子行宽仅 256/512、fp32 累加已远优于 fp16 输出容差 | ✅（否决） |

**R2 在线算法**：**有，且被采纳为 phase 1 的单遍形式**。三个 RMSNorm 的统计量与 gate 点积都是
可分解的 **running 统计量**（和/平方和/乘积和），因此存在单遍扫描变体：4 条累加链在一次
`j` 遍历里并行推进（§0.6 R3）；中间状态是 4 个 fp32 累加器（O(1)）而不是 4 个 O(d) 中间向量
（源结构里的 `h_norm`、`k_norm`、`vhat` 三个 O(d) 向量在 phase 1 里**根本不需要物化**）。
**收益口径**：行内扫描遍数 15 → 2；中间缓冲 `h_norm`/`k_norm`/`v_sq` 三个 O(d) fragment 消失。
**结构判据 + 边界**：`alpha` 依赖整行归约的完成，所以「归约 → 缩放」的两阶段边界**不可消除**
（任何在线变体也只能把它压到 2 个阶段；这不是实现选择而是算子的 inherent 依赖）。同理，
`vhat_norm` 的归一化依赖 `alpha`，因此 `Σv̂²` 不能在 `alpha` 之前出结果——本设计用因子提出
（`Σ(αv)² = α²Σv²`，A3）把这条依赖**从数据依赖降级成标量乘法**，从而让 phase 1 完全不含 `alpha`。
两遍结构因此是「2 遍 = 语义下界」，不是「未优化的三遍」。

**R3 复杂度对比**（四口径；代表 workload = serving-b4：batch=4、d=512、d_padded=512、mcl=20、
w=4、δ=5、fp16。口径说明：FLOPs 只计乘加与超越函数（exp 按 1 次向量运算计），不含索引与 cast；
Bytes 只计本 launch 的 GM 流量，权重按「HBM 一遍」计（L2 内的 batch 次重复不计）；
「扫描遍数」区分 GM 遍数与**行内元素级 pass 数**（片上重复扫描——在本算子上它才是主导项）；
中间缓冲峰值只计片上驻留）：

| 算法候选 | FLOPs | 访存量 (Bytes) | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|-------|---------------|---------|-------------|---------------------|
| 0 基线（源算子直译） | ≈ 41 k（20 FLOP/元素 × d × batch） | 47 KB（h 4 K + kv 16 K + tap 12 K + 权重 6 K + 写出 8 K） | GM 各 1 遍；**行内 pass 15** + 4 归约 | UB 16 K + 归约 scratch 8 K | 4 逻辑核；无跨核 |
| **C1 选定**（= 候选 1 + §0.6 R1–R5） | ≈ 41 k（同基线） | **47 KB**（与基线相同：优化只减片上 pass，不减 GM 流量） | GM 各 1 遍；**行内 pass 2** + 4 归约 | UB 24 K（含 4×2 K scratch） | **4 逻辑核 ≤ 56 Vector 核（单波）；1 launch** |
| 2 保守逐统计量（SIMT） | ≈ 41 k | 47 KB | 15 + 4 | 24 K | 4 核；1 launch |
| 3 SIMD 寄存器路径 | ≈ 41 k | 47 KB | 每阶段 `nchunks`(=8) 段 + 4 次 `vcadd` | 16 K（无 reducer scratch） | 4 核；`vcadd` 的 mask/分布参数需逐路径核对 |
| 4 分 `d` 跨核（nc=4/8） | ≈ 41 k | 47 KB + 归并 workspace `3·batch·nc·4 B` ≈ 0.4 K | 同 C1 | 同 C1 | 4·nc 个逻辑块；**代价 = 跨核 barrier（`ascend.batch_schedule` 未认证）或 +1 launch** |
| 5 与 project 融合 | ≈ 41 k + 8.4 M（Cube） | **权重 `batch` 遍：6 K → 24 K**；且需 tile 间归约 | 同 C1（step 段） | 需 Cube tile + 跨 tile 归约 | 跨设计边界（project 段已冻结）；M=1 的 Cube 浪费 |
| 6 Cube 做卷积 | — | — | — | — | **无矩阵结构**（对角权重），Cube 无有效工作 |
| 7 两遍 materialize | ≈ 41 k | 47 KB + 4 K（`vhat_norm` 往返）+ tap 4 倍重读 48 K | 3+ | +GM 中间 | **+1 launch** |
| 8/9（稳定化 / 补偿求和） | +1 行归约 / +2 运算每项 | 同 | 同 | 同 | 净负（见 R4） |

**R4 硬件亲和性评估**（逐候选对照检查清单；负向淘汰附证据或标注未实测假设）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---------|-------------|---------|------------|---------|------------|------|
| 0 基线直译 | Vector ✅（逐元素 + 行归约） | ✅ | ✅ | ✅ | ❌ 15 pass 串行、无重叠 | ⚠️ 结构可直译但**发射项吃亏**（15×0.5 µs，见下） |
| 1 / **C1** | Vector ✅（SIMT 线程域 + 4 并发 reducer） | ✅ 24 K/248 K | ✅ `d_pad % 16 == 0`，`tile` 覆盖尾列 | ✅ 全部编译期常量 | ✅ 单 launch、MTE2 批量预取与 VF 无依赖 | ✅ **选定** |
| 2 保守（SIMT） | Vector ✅ | ✅ | ✅ | ✅ | ⚠️ pass 数不变 | ✅ 可作 Stage 3 的**降落伞**（探针 S2 已编译，§11.1） |
| 3 SIMD | Vector ✅ | ✅ 16 K | ✅ 兄弟算子用 `update_mask(min(64,d-ch*64))` 处理尾 lane | ✅ | ✅ | ⚠️ **备选**（§1.6.3 裁决）——本地无「4 统计量 + 仿射广播」的 SIMD 已验证条目 |
| 4 分 `d` 跨核 | ✅ | ✅ | ✅ | ✅ | ❌ 跨核归并需要 barrier（`CG-950-batch-barrier` 未认证） | ❌ 淘汰：+1 launch ≈ +5 µs（未实测假设），或未认证 barrier；而本算子的每行计算只有 ≈ 0.02 µs 量级 |
| 5 与 project 融合 | ⚠️ Cube 退化为 M=1 GEMV | ⚠️ 权重 `batch` 遍 | ✅ | ✅ | ⚠️ 跨 tile 归约需 barrier | ❌ 淘汰：跨设计边界（project 段已冻结）+ 权重流量 ×batch |
| 6 Cube 卷积 | ❌ 深度卷积无公共收缩轴 | — | — | — | — | ❌ 淘汰（结构不成立） |
| 7 两遍 materialize | ✅ | ✅ | ✅ | ✅ | ❌ +1 launch + 中间往返 | ❌ 淘汰（≈ +5 µs） |
| 8 稳定化变形 | ✅ | ✅ | ✅ | ✅ | ⚠️ 多一次行归约 | ❌ 否决：`c/(1+e^{-c})` 在 fp16 输出域已足精度（`e^{-c}` 下溢即真值极小），减 max 反增一次归约 |
| 9 补偿求和 | ✅ | ✅ | ✅ | ✅ | ⚠️ 每项 +2 运算 | ❌ 否决：行宽 256/512 的 fp32 累加误差 ≈ 1e-6，输出容差 5e-2，无收益 |

**设计期估算（D-2 roofline 下界行）**

估算口径：`估算下界 = max(流量项, 发射项, 容量项)`（`.agents/skills/_shared/standards/hardware-cost-model.md` §2）。
流量按 §0.2 的 5 条 workload 逐项记账（`bytes = batch·d_pad·(2 + 8 + (w-1)·2 + 4) + (w+2)·d_pad·2`，
其中 2/8/(w-1)·2/4 分别是 h、kv fp32、tap、两行写出的每元素字节数）；发射项 = 1 次 launch
（≈ 5 µs，**未实测假设**，沿用同项目 `examples/engram_decode/_engram_project_kernel/DESIGN.md`
的 §1.6.0 口径）+ 2 个元素级 pass（按 `CONST-vector-launch-overhead` 的 ~0.5 µs/op，**910B2C 实测、
950PR 未实测**）+ 4 次 `AscendAllReduce`（**无本地常数，未实测假设**）；容量项 = UB 24 KiB / 248 KiB、
无 L1/L0 使用。

| workload | 流量项（Bytes / 假设带宽 1.0–1.6 TB/s） | 发射项（1 launch + 2 pass + 4 归约） | 容量项 | 估算下界 |
|---|---|---|---|---|
| decode-s1 | 8.2 KB → 0.005–0.008 µs | ≈ 6 µs | UB 24 K/248 K 满足 | ≈ 6 µs |
| serving-b1 | 8.2 KB → 0.005–0.008 µs | ≈ 6 µs | 同上 | ≈ 6 µs |
| serving-b4 | 47 KB → 0.03–0.05 µs | ≈ 6 µs | 同上 | ≈ 6 µs |
| serving-b8 | 44 KB → 0.03–0.04 µs | ≈ 6 µs | 同上 | ≈ 6 µs |
| decode-partial-cache | 13 KB → 0.008–0.013 µs | ≈ 6 µs | 同上 | ≈ 6 µs |

`估算下界: 6 µs（发射项 1 launch ≈5 µs + 2 pass × 0.5 µs + 4 次 AllReduce 未计常数；流量项 0.005–0.05 µs；容量项 UB 24 KiB/248 KiB 满足）`，
其中**带宽（1.0–1.6 TB/s）、单次 launch 开销（≈5 µs）、`AscendAllReduce` 单次代价、Vector 峰值全是
「未实测假设」**——平台 `cost_measurements` 为空、pattern-library 的常数全部绑定 Ascend910B2C
（`CONST-vector-launch-overhead` 的版本戳为 `tilelang 0.1.2+3a214cde + CANN 8.5.0 / 910B2C`），
故本行只用于**结构决策**（见下）而非性能承诺；Stage 4 首跑后按
`.agents/skills/_shared/standards/hardware-cost-model.md` §3 回填「设计估算 vs 实测」偏差行。

**该行驱动的三个设计结论**：
(a) **整算子 = 2 次 launch ≈ 11 µs**，其中 step 段 ≈ 6 µs 且以发射项为主 → 本 kernel 内部任何
「多一个 pass」都是在 5 µs 的 20 % 量级上做加法，因此 §1.6.1 的 pass 融合是主要收益来源；
(b) 把 `d` 分给多个核（候选 4）只能减少那 ≈ 0.02 µs 的计算项，却要付 5 µs 的第二次 launch 或
未认证的跨核 barrier → **不划算**；
(c) 与 project 段融合（候选 5）能省 5 µs，但要改已冻结的函数边界并把权重流量乘 `batch`
→ 属于**跨设计变更**，本阶段不做（列入 §9.1 的「非本设计可解决的余量」）。

**调研结论**：选定 **C1 —— 1 个 launch、`T.Kernel(batch)` 每核一行、`T.SimtVF` 线程域内
两遍扫描 + 4 并发 SIMT reducer 的行归约**，并对公式做 §1.6.1 的 6 项等价优化（A0–A4 与 A6）。
关键依据：R3 中 C1 把行内元素级 pass 从 15 压到 2（唯一能压到语义下界 2 的候选），访存与基线相同
（已是最小值：每张量 1 遍），R4 中只有 C1/C2/C3 满足「单 launch + 全静态边界 + 全片上驻留」；
候选 3（SIMD）与 C1 同属「2 阶段 + 寄存器/线程域归约」的算法族，差别只在执行单元与归约指令，
本地缺少「4 统计量 + 仿射广播」的 SIMD 已验证条目，故**作为 §1.6.3 的备选 + Stage 4 裁决**，
不列为纸面主选；候选 4/5/7 各有一条量化负向依据（+launch / 流量 ×batch / 中间往返），
候选 6 结构不成立，候选 8/9 净负。
**「无更优替代」部分的范围声明**：已逐条评估
`.agents/skills/tilelang-op-design/references/algorithm-research.md` 的替代空间
（在线/单遍、因子提出、跨核并行、跨 launch 融合、计算单元替换、稳定化、补偿求和 = R1 的 #1–#9），
并检索本仓同类实现（`examples/engram_gate_conv/_engram_gate_conv_fwd_kernel/DESIGN.md` 的
SIMD 形态与 chunk 划分、`examples/norm/fused_add_rms_norm/` 的归一化写法、
`.agents/platforms/toolchains/ascend/knowledge/operator-patterns.md` 的 PL-950-normalization 条目、
`.agents/platforms/toolchains/ascend/knowledge/algorithm-candidates.md` 的 reduction/normalization 候选行，
平台知识库中**没有**本算子族的其他候选）；因此「无更优的等价化简公式」限定在上述已评估集合内，
不宣称全局最优。**源算法优化意图承接**：O1 → 行驻留 UB/fragment（§0.6 R1）；
O2 → 并发 reducer epoch（R2/R3）；O3 → reducer 的 `dst` 标量在元素级表达式中直接广播（保留）；
O4 → `T.Kernel(batch)`（保留）；O5 → 重新设计为两遍（R1）；O6 → 编译期展开进 phase 2（R1）；
O7 → A4 的 CSE；O8 → 保留 rsqrt + A6 的常量倒数；O9 → `T.SimtVF(threads=)`（R5）。
**唯一被舍弃的优化是 O11（dtype 往返）**，理由是与语义冲突（§0.5 #7 / §1.6.1 B1），
不是静默丢弃。

#### 1.6.1 数学等价优化（公式级）

分析对象 = §1.6.0 选定的 C1。本段没有可降代的高代价算子（`exp`/`rsqrt` 已由 SFU 承担，
`exp→exp2·log2e` 在本平台无 `vexp2`，见 §1.6.2 #9），可优化空间集中在
**归约的扫描结构、标量因子的位置、以及访存/常量折叠**四处。

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益量化 |
|---|--------|------|-----------|-----------|---------|
| A0 | 零填充域归约（**前提**） | `Σ_{j<d} x_j²`（按 `d` 掩码归约） | `Σ_{j<tile} x_j²`，其中 `j ≥ d` 的列由 host 零填充恒为 0，`tile ≥ d_pad ≥ d` | 实数域恒等；fp32 下逐位恒等（加性零元：`s + 0.0 = s` 对任意有限值成立；`tile = d_pad` 时连 fragment 尾 lane 都不存在（host 零填充列仍为 0）） | 去掉 4 条归约里每 lane 的 `j<d` 谓词与比较（`d_pad` 列 × 4）；`serving-b4` 为 2048 次比较 |
| A1 | 4 路归约融合为单遍 | 4 次独立行扫描：`Σh²`、`Σk²`、`Σh_n k_n`、`Σv̂²` 各一遍 | 一次 `T.Parallel(tile)` 扫描内 4 条独立累加链（§0.6 R3） | 每条链的项集合与遍历顺序**都不变**，只是四条链交错执行——每个累加器的中间值序列与「独立跑一遍」逐位相同 | 行内元素级 pass 15 → 2；`serving-b4` 省 13 个 pass ≈ 6.5 µs（`CONST-vector-launch-overhead` 0.5 µs/op，910B2C 实测 / 950PR 未实测） |
| A2 | gate 点积的归一化因子提出 | `h_n = h·rh·w`、`k_n = k·rk·w`（两个 O(d) 向量物化），`dot = Σ h_n·k_n` | `dot = rh·rk·Σ_j h_j k_j w_j²`，只留一个 O(d) 中间量 `Σ h k w²` | 实数域恒等：`Σ (h rh w)(k rk w) = rh·rk·Σ h k w²`。fp32 下把两个标量因子从「逐项乘」提到「和之后乘」，舍入路径变化 ≤ 数个 ulp（实测最大 30 ulp ≈ 3.6e-6 相对，出现在 alpha-small 角点） | 省 2 个 O(d) 标量缩放 pass（`h_norm`/`k_norm` 不再物化）与 2 个 `tile` 宽 fragment（`serving-b4` 各 2 KiB） |
| A3 | v̂ 方差的因子提出 | `v̂ = α·v`；`Σv̂² = Σ(α v)²`（依赖 α 完成后才能算） | `Σv̂² = α²·Σv²`；`Σv²` 在 phase 1 就与其它统计量同遍算出 | 实数域恒等：`Σ(α v_j)² = α² Σ v_j²`。fp32 下把逐项因子提出，误差 ≤ 1 ulp（实测最大 2 ulp）。**边界**：由 Cauchy–Schwarz `|dot|/√d ≤ √d`（`w_h ≡ 1` 时），`alpha ≥ σ(-√d) ≥ 1.5e-10`（`d = 512`）→ `α² ≥ 2.3e-20 ≫ 1.2e-38`（fp32 最小正规数），**不会下溢**；即便退化，两式在 `α → 0` 时都给出 `rrms_v → rsqrt(eps)`，行为一致（alpha-small 角点覆盖） | 把 `alpha` 从 phase 1 的数据依赖里**移除**（phase 1 不再需要 `v̂`），从而省掉 1 个 pass + 1 个 O(d) fragment；这是把 R2 的两阶段压到语义下界的关键一步 |
| A4 | 残差公共子表达式 | `v̂ = α·v`（第 191 行）与 `y = SiLU(c) + α·v`（第 219 行）各算一次 `α·v` | 一次 `av = α·v`，同时供 `vn = av·rrms_v·w_v` 与残差 `y = … + av` | 完全相同的表达式复用同一 fp32 值——源实现两次计算的也是同一表达式（fp32 确定性），故**逐位恒等**（实测 0 ulp） | 省 1 个 `tile` 宽乘法与 1 次 `v` 的 UB 读（`serving-b4` 2048 乘 + 2 KiB 读） |
| A6 | 标量除法 → 常量倒数 | `x / float(d)`、`dot / T.sqrt(float(d))`（每行 4 次除法） | `x · inv_d`、`dot · inv_sqrt_d`（`inv_d`/`inv_sqrt_d` 是 host 计算的 fp32 常量） | `d = 2^k`（本算子 workload 的 `d ∈ {256,512}` 全是 2 的幂）时乘以 `1/d` **逐位等于**除以 `d`（只改指数）；`1/√d` 不可精确表示，误差 ≤ 1 ulp（实测最大 2 ulp） | 每行省 4 次标量除法（收益小，按四要素如实记录；探针生成的常量 `0x1p-9f` / `0x1.6a09e667f3bccp-5f` 印证已折叠） |

**等价性机器验证（D-1）**：`examples/engram_decode/_engram_step_kernel/verify_equiv.py`
（torch CPU，fp64 参照；8 组用例 = 5 条 manifest workload + 尾列填充 `d=600`/`d_pad=768`
（`tile=1024`，非 2 的幂的 `d_pad`）+ `alpha→0` 角点 + 特殊值角点（±inf/NaN/±0/subnormal/±65504）；每项都是「只开该项 vs 只关该项」
的**单项隔离**对照，基线 = 源算子的逐 pass 结构。判据：fp32 中间量 ≤ 32 ulp（≈4e-6 相对，
比 op 级容差 5e-2 低 4 个数量级）+ 写出的 dtype 值逐位一致率 ≥ 99.9 % + 非有限值类型不出现
分歧 + 最大绝对偏差 ≤ 1e-3）。实测（脚本末尾的 per-item 汇总表；脚本执行打印
`RESULT: ALL EQUIV_PASS`，48 个 cell、0 失败、最坏 30 ulp）：

| 采纳项（脚本标签） | 覆盖用例 | 最大 ulp 差 | >1 ulp 违反率 | 写出 dtype 逐位一致率 | 最大绝对偏差 | 结论 |
|---|---|---|---|---|---|---|
| A0 零填充域（`A0`） | 8 组（含 `d_pad=768` 的 168 列零尾） | **0** | 0.0 % | 100 % | 0 | **EQUIV_PASS** |
| A1 四路归约融合（`A1`） | 8 组（标量 fp32 串行累加，交错 vs 4 遍） | **0** | 0.0 % | 100 % | 0 | **EQUIV_PASS** |
| A2 gate 因子提出（`A2`） | 8 组 | **30**（alpha-small 角点；其余 1–3） | ≤ 100 %（统计量最低位差异） | **100 %** | 0 | **EQUIV_PASS** |
| A3 方差因子提出（`A3`） | 8 组 | **2** | ≤ 100 % | **100 %** | 0 | **EQUIV_PASS** |
| A4 残差 CSE（`A4`） | 8 组 | **0** | 0.0 % | 100 % | 0 | **EQUIV_PASS** |
| A6 常量倒数（`A6`） | 8 组 | **2** | ≤ 50 % | **99.9512 %** | 9.54e-07 | **EQUIV_PASS** |

> 注：`>1 ulp 违反率` 一列在 A2/A3 上达 100 %，指的是**每行那个标量统计量**的最后一个 ulp
> 有差异（8 行里 2–8 行如此），不是逐元素向量的差异；对应写出的 dtype 值是 100 % 逐位一致。

**测量但拒绝的候选（同一脚本的 informational 表，不参与判定）**

| 候选（脚本标签） | 原式 → 候选式 | 实测 | 判定 |
|---|---|---|---|
| B0 SiLU 单除法（`B0`） | `c·(1/(1+e^{-c}))` → `c/(1+e^{-c})` | fp32 最大 256 ulp（全部来自 `y≈0` 的抵消区）、写出 dtype 值 99.80–100 % 逐位一致、最大绝对偏差 9.54e-07 | **不采纳**：判据（≤32 ulp）不满足，且收益只有 1 次 `d` 宽乘法——不值得把「近零抵消区」的度量争议带进设计；保留源的 `c·σ(c)` 形态 |
| B1 `kv`/`v̂` 的 dtype 往返（`B1`，950DT 参考的做法） | `k` 直接 fp32 归一化 → `cast(cast(k,dtype),fp32)` 再归一化（`v̂` 同款） | **最大 ulp 差 1.8e+09、>1 ulp 违反率 100 %、写出 dtype 值仅 0.39–76.17 % 逐位一致、最大绝对偏差 1.56e-02** | **不采纳**：这是**语义偏差**而不是优化（§0.1 的参考实现直接对 fp32 `kv` 归一化）。950DT 参考的这处往返在 950PR 上照搬会引入 1.6e-2 量级的偏差 |

**优化结论**：采纳 **6 项**（A0 零填充域归约、A1 四路归约融合、A2 gate 因子提出、
A3 方差因子提出、A4 残差 CSE、A6 常量倒数），全部机器验证通过（最坏 30 ulp，
写出值逐位一致率 ≥ 99.95 %）；拒绝 B0（收益不足 + 度量争议）与 B1（语义偏差）。
优化后的公式（供 §3.1 拆解使用，`inv_d`/`inv_sqrt_d` 为 host 常量）：

```
tile = next_pow2(d_padded);  am = 1（每核一行）
# phase 1 —— 一次扫描
S_h = Σ_j h_j²;  S_k = Σ_j k_j²;  P = Σ_j h_j k_j w_h[j]²;  S_v = Σ_j v_j²
# 标量链（每行一次）
rh = rsqrt(S_h·inv_d + eps);  rk = rsqrt(S_k·inv_d + eps)
alpha = 1 / (1 + exp(-(rh·rk·P)·inv_sqrt_d))
rv = rsqrt(alpha²·S_v·inv_d + eps)
# phase 2 —— 一次扫描
av    = alpha·v_j
vn    = av·rv·w_v[j]                      → new_conv_state[b, mcl-1, j] = cast(vn)
c     = cw[w-1,j]·vn + Σ_{p<w-1} cw[p,j]·cast(state[b, mcl-(w-1-p)δ, j])
y_t[b,j] = cast( c·sig + av )
```

**无进一步优化空间的项（显式结论 + 依据）**：(a) **没有除法/开方的逐元素热点**——`rsqrt` 已是
源实现的形态，`x/d` 被 A6 折叠，逐元素只剩 `1/(1+e^{-c})` 的一次除法（B0 已测量并拒绝）；
(b) **公共子表达式已穷尽**——`α·v`（A4）、`w_h²` 与 `h·k` 的乘积（A2 合并为一项）、`v` 的
两次使用（A4/A3）；(c) **无可降代的高代价算子**——`exp`/`rsqrt` 由 SFU 直接执行，
`simd.py` 导出表无 `vexp2`（本工具链无 exp2 降代换路径，见 §1.6.2 #9）；
(d) **访存量已是下界**——每个输入张量 GM 1 遍（权重在 L2 内的 `batch` 次重复不计 HBM），
输出 2 行写出是语义要求；(e) **归约遍数已是下界**——2 遍来自 `alpha` 对全行归约的依赖（§1.6.0 R2），
任何单遍变体都只能重排这个依赖而不能消除。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

盘点 §3.3 伪代码中的**全部**循环与标量计算点（共 11 处）：

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由（不可替代时必填） |
|---|--------|-----------|-------------|---------|------------------------------|
| 1 | 块索引 `bx` → 行号 | `T.Kernel(batch)` 的 1-D 索引 | 无 | ❌ | **block 级索引 / 任务映射**：每块一次；平台 `T.Kernel(N)` 只给 1 维 `bx`（`references/programming.md`：多维网格被拒绝） |
| 2 | `tile = next_pow2(d_padded)`、`L = min(512, tile)` | host/编译期常量 | 无 | ❌（不需要） | **编译期常量**：不产生任何运行时指令（工厂函数参数） |
| 3 | tap 索引 `mcl-(w-1-p)·δ` | 源的 `for p in T.serial(w-1)` | Python 层 `for p in range(w-1)` 展开为常量索引 | ✅（已替代） | — 展开后 `p` 是编译期常量，`w-1 = 3` 条 `T.copy` 与 3 个乘积项，**运行时无 tap 循环**（探针生成的是 `for (p<3)` 的 C 循环，其边界为常量、无索引算术） |
| 4 | `T.Parallel(tile)` 内的 4 条 `h²`/`k²`/`h·k·w_h²`/`v²` | 源的 3 个独立逐元素 pass | 单个 `T.Parallel(tile, coalesced_width=1)` 内的 4 条表达式（lane 并行） | ✅（已替代） | — 每 lane 一个元素，S1 生成源码即 4 条 `acc_*[0] += …` |
| 5 | 4 次行归约 | 源 `T.reduce_sum(frag2d, out, dim=1)` | `T.alloc_reducer` + `reducer_update` ×4 + `finalize_reducer` ×4（跨 lane 归约由编译器生成的 `AscendAllReduce` 完成） | ✅（已替代） | — `PL-950-row-primitives` 已验证的 SIMT reducer 路径 |
| 6 | tap 索引守卫 `if 0 <= idx < mcl` | 源参考（torch）的 Python `if` | 无（编译期删除） | ✅（已替代） | — 由 shape rule `mcl ≥ δ(w-1)` 得 `idx ∈ [0, mcl-1]` 恒成立（§0.1 已证），生成代码不含该分支 |
| 7 | 每行标量链（3×`rsqrt` + 1×`exp` + 1÷ + 4×`mul`） | 逐标量表达式 | 向量化（把 batch 个标量拼成向量一起算） | ❌ | **block 级标量**：每核只有 1 行、每行算 1 次（O(1)/行），拼向量的前提是「一核多行」——而本设计是 1 核 1 行（§5.5 的并行度选择），拼起来只能得到 1 个有效 lane，属净负。它也不是逐元素热点（`batch ≤ 17` 个标量 vs `tile × 2` 个元素） |
| 8 | phase 2 的 `av`/`vn`/`c`/`y` 元素级链 | 源的 6 个独立逐元素 pass | 单个 `T.Parallel(tile)` 内的一条复合表达式（含 4 个 tap 项） | ✅（已替代） | — 生成源码为一次 lane 并行区域 |
| 9 | `exp` 的降代换（`e^{-x} = 2^{-x·log2e}`） | — | `T.simd.vexp2` 不存在（`simd.py` 导出表无 `vexp2`）；`T.exp` 在本平台已映射到 SFU `expf`（生成源码为证） | ❌ | **所需向量 API 在本项目不存在**：查证方式 = 在任务目录静态检查 `tilelang.ascend.language.simd` 的导出列表（94 个符号，无 `vexp2`）；且 `T.exp` 已是单条 SFU 调用，降代换无收益 |
| 10 | `i < d_pad` 的尾列谓词 | 源 `T.if_then_else` 逐元素选择 | 编译期分支：`tile == d_pad` 时**不生成**谓词；否则生成 lane 级条件（fragment 尾 lane 置 0） | ✅（已替代） | — 5 条 workload 的 `tile ≡ d_pad`（256/512 都是 2 的幂）→ 零谓词；`d=600`/`d_pad=768`（`tile=1024`）的探针变体走生成的条件路径 |
| 11 | GM↔UB 的 9 条装载 + 2 条写回 | 源 `T.Parallel` 逐元素 + `T.copy` 混用 | 全部 `T.copy` 区域 DMA（`asc_copy_gm2ub_align` / `asc_copy_ub2gm_align`） | ✅（已替代） | — 探针生成源码即 9 条 `gm2ub` + 2 条 `ub2gm`，无逐元素搬运指令 |

**向量化结论**：逐元素计算点 **全部**由 `T.Parallel(tile)` 的 lane 并行 +
`reducer`/`DMA` 承担（#3–#6、#8、#10、#11 共 8 处已替代）；保留 **3 处**非向量化点，
理由类别分别是：block 级索引（#1）、编译期常量（#2，不产生指令）、block 级标量链（#7，
每行 O(1) 且一核只算一行）、API 缺失（#9，`vexp2` 不存在）。
没有以「实现简单」为由保留的任何逐元素标量循环。§6 的循环表与本节一一对应。

#### 1.6.3 向量化轴与数据布局决策（阻塞级）

本算子有两个轴决策 + 一个执行单元决策：**(i) 归约类的 lane↔归约轴映射**（水平归约 vs 垂直扫描）、
**(ii) lane 宽度与每 lane 元素数**、**(iii) 执行单元：SIMT（主选）vs SIMD（备选）**。

**(i) 归约轴候选**

| # | 布局方案 | lane ↔ 归约轴 | 归约形态 | repack 路径 | 收益 / 代价 | 是否采纳 |
|---|---------|--------------|---------|------------|------------|---------|
| 1 | 行内连续 `(tile,)` + lane↔`j` | 水平（lane 参与归约） | 4 次 `AscendAllReduce`（每行 4 次） | 无（I/O 原生行主序） | 归约 4 次/行；元素级 pass 满宽；UB 24 KiB | ✅ **采纳** |
| 2 | 垂直扫描：lane↔batch 行，串行沿 `d` | 垂直（lane 各自独立行） | 0 次跨 lane 归约 | 无 | `batch ≤ 8` 只用 8/512 lane（1.6 %）；元素级吞吐掉 64×；且 `batch=1` 时只剩 1 lane | ❌ 量化否决 |
| 3 | 2-D lane（行 × `d`-chunk）+ chunk 内部分和 + 核内合并 | 混合 | 每行 `nchunk×4` 次归约 + 合并 | 无 | `batch=1` 时退化；部分和缓冲 + 二次归约；归约次数 ×nchunk | ❌ 量化否决 |
| 4 | `d`-major 转置布局（`(d, batch, …)`） | `d` 或 `batch` | 同 1 | 需要核内/核外转置链 | 转置链的读写 = `2·batch·d·2 B`（`serving-b4` 8 KiB ≈ 本体流量的 17 %），且契约就是行主序 | ❌ 代价 > 本体 |

轴质量评分（采纳项 #1）：**整除性**——`tile ∈ {256, 512}`、fp32 向量宽度 64 → `tile/64 ∈ {4, 8}`
整除，无尾 lane（`d_pad % 256 == 0` 由 host 的 `align_up(d, 256)` 保证）；
**累加链形态**——4 条独立链、`threads = tile` 时每 lane 恰好 1 项（无跨步系数阻碍），
树形合并由 `AscendAllReduce` 完成；**repack 代价**——0（行主序即契约）；**UB 容量影响**——
24 KiB / 248 KiB = 9.7 %。

**(ii) lane 宽度与每 lane 元素数**

| # | `T.SimtVF(threads=)` | 每 lane 元素数（`tile=512`） | 生成形态（探针实读） | 归约树 | 结论 |
|---|---|---|---|---|---|
| 1 | 512 | 1 | 直接 `buf[tid + k]` 一条表达式（`float vf[1]`） | 512 lane 单段 | ✅ **默认**（每 pass 满宽、树最矮） |
| 2 | 128 | 4 | `#pragma unroll for(i<4)` + 4 次串行累加（`float vf[4]`） | 4 项串行 → 128 lane | ⚠️ Stage 4 A/B（{128, 256, 512}）：合并同 lane 的 4 个 UB 读可能更省指令，但归约多一段 |
| 3 | 64 | 8 | 同 2 的放大版 | 8 项串行 → 64 lane | ❌ 归约串行段过长；`batch=1` 时并行度也够，但无收益证据 |

**默认取 #1**（与 `PL-950-row-primitives`「每行一个核、N 元素 N lane」的已验证形态一致，
探针 S1 即该形态）。`threads ∈ {128,256,512}` 作为 Stage 4 的 A/B 清单（改动只是启动参数，
不改变 §3.1 公式与 §6 循环结构）。

**(iii) 执行单元：主选 SIMT + 备选 SIMD + 实验裁决计划**

判定裕度落在**两个未实证常数**上：(a) 一次 `AscendAllReduce<T, 512, 1, 0>` 的实际代价
（含 barrier + UB 往返）；(b) `vcadd` 一条 64-lane 树归约的代价。平台 `cost_measurements` 为空，
pattern-library 的常数全部绑定 910B2C（`CONST-vector-launch-overhead` 只说「向量 op ~0.5 µs」，
未区分 reducer 与 `vcadd`）。故必须按三件套执行：

> **① 主选（默认，进 Stage 3）= SIMT**：`T.Kernel(batch)` + `T.SimtVF(threads=min(512,tile))` +
> 4 并发 `alloc_reducer`。理由（见 §2.2 的量化对照）：本算子每核只做 **4 次**归约
> （1 行/核），而同族兄弟算子 `examples/engram_gate_conv/_engram_gate_conv_fwd_kernel/DESIGN.md`
> 选 SIMD 的依据是「每核 4–9 行 × 4 次归约 = 16–36 次 barrier」——该理由在本算子上弱 4–9 倍；
> 且 950 平台的 **SIMT 归约/归一化有两条已验证条目**（`PL-950-row-primitives`、
> `PL-950-normalization`），SIMD 侧只有单行 sum/max（`PL-950-simd-reduce`）与一个已知缺陷
> （`TRAP-950-simd-shared-reduce`），而这正是 950DT 上本 kernel 的已验证形态（`_build_step`）。
>
> **② 备选（结构完整，交 Stage 4 A/B）= SIMD**（`T.SimdVF` + `T.simd.*`）：
> - **布局**：行内连续，`64` 个 fp32 lane = 一个向量寄存器；`nchunks = d_pad // 64 ∈ {4, 8}`，
>   `d_pad % 64 == 0` 严格整除（`d_pad` 是 256 的倍数）→ 无尾 lane、不需要 `update_mask`
>   （若将来出现非 64 倍数的 `d_pad`，用 `T.simd.update_mask(min(64, d_pad-ch*64), 32)`，
>   与兄弟算子的已验证写法一致）。
> - **UB 预算**：与主选相同的 9 个 UB 缓冲（`h/k/v/wh/wv/cw/tap/y/vn`）= 16 KiB，
>   **不需要** 4×2 KiB 的 reducer scratch（`vld` 直接读 UB、结果留在寄存器）→ 16 KiB/248 KiB。
> - **寄存器累加器**：`T.simd.alloc_local((4,), "float32")` 承担 4 个统计量，跨 chunk 循环携带
>   （向量累加器不能跨 IR `for` 迭代传递——兄弟设计 §1.6.2 V11 与
>   `examples/mla_dense_bwd/_delta_kernel/DESIGN.md` §9.2 的探针 F7 记录了这一约束）。
> - **结构**：chunk 循环（`T.serial(nchunks)`）内 `vld(..., dist="UNPK_B16")` + `vcvt(part=0)`
>   提升 fp16/bf16 → fp32，`vmul/vadd` 累加；循环后 `vcadd(acc, full)` ×4 + `vdupv` 广播标量；
>   第二个 chunk 循环做 `av/vn/c/y`，`vcvt(dtype, round="ROUND_R")` + `vsts(dist="PK_B32")` 写回。
> - **分核三要素按新粒度重算**：逻辑核数仍是 `batch`（1 行/核），物理 56 Vector 核，
>   `batch ≤ 17 ≤ 56` → 结论与主选相同（§5.5）。
> - **风险**：`T.simd` 的 load/store 分布参数与 `vcvt` 的舍入/饱和参数需逐路径核对
>   （`references/programming.md` 明确「需要按具体路径核对」），且本地**没有**「4 统计量 + 仿射
>   广播」的 SIMD 已验证条目 → 这是它不作主选的直接原因。
>
> **③ 裁决计划**：
> - **代表 shape**：`serving-b1`（batch=1、d=256、nchunks=4，最少行 + 最少 chunk）与
>   `serving-b4`（batch=4、d=512、nchunks=8，最多行 + 最多 chunk）。
> - **测量指标**：`msopprof` 的 kernel-only `Task Duration`(us)；A/B 交错 ≥3 轮；两次变体都
>   先过 L0/L1 精度（`--level all`）。
> - **判定阈值**：备选 < 主选 × **0.90**（10 % 裕度，覆盖测量噪声）才翻转；**允许按 shape 分派**
>   ——工厂已按 `(batch, d, d_padded, max_conv_len, conv_kernel_size, dilation, eps, dtype)` 做
>   `lru_cache` 特化，因此可以出现「b1 走备选、b4 走主选」的分片结论。
> - **回写路径**：实测数与结论回写本节，并在 Stage 4 的 `perf_records.jsonl` 里作为 A/B 分支记录；
>   若翻转则按 `perf-feedback.md` 走受控设计修订（不回退 Stage 1 门禁）。

**布局决策结论**：选定 **(i)#1 + (ii)#1 + (iii)①** —— 行内连续 `(tile,)` 布局、lane ↔ `d`
（水平归约、`AscendAllReduce` 4 次/行）、`T.SimtVF(threads=min(512,tile))` 每 lane 一个元素、
执行单元 = SIMT；SIMD 形态作为 §1.6.3 的备选与 Stage 4 的 A/B 分支随文档冻结。
该结论已同步到 §3.3 伪代码（`T.Parallel(tile, coalesced_width=1)`、4 个 reducer）、
§4.3（9 个 UB 缓冲 + 4 个 scratch 槽）与 §6（两遍扫描、无运行时 tap 循环）。
**弃选理由均为量化或证据**（lane 利用率 1.6 %、转置链 17 % 流量、归约次数 ×nchunk、
SIMD 缺本地已验证条目），无「未考虑」项。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式: SIMT**（任务平台 `programming_modes.choices` 的 `simt` 键；实现 = `T.Kernel(batch)` 内的
`T.SimtVF(threads=min(512, tile))` 线程域 + `T.Parallel(tile)` 逐元素表达式 +
`T.alloc_reducer(op="sum")` 行归约）。

### 2.2 选型理由

- **算子不含任何矩阵乘**：只有逐元素乘加、4 个行归约、w 个 tap 的窗口累加与一次
  `exp`/除法。平台矩阵单元（Cube）的 fractal 语义（N ≥ 16）与「每行一个标量输出」不匹配
  （§1.6.0 R1 候选 6 已给结构依据），故排除 **Cube** 与 **混合**。
- **为什么偏离 `migration_default = simd`**：默认值是平台的迁移占位，不是用户约束。本算子的
  **行归约 + 标量广播回元素级**在 SIMT 侧有两条同平台已验证条目（`PL-950-row-primitives`
  的 24 项、`PL-950-normalization` 的带 affine RMSNorm 6 项），而 SIMD 侧只有单行 FP32
  sum/max（`PL-950-simd-reduce`）+ 一个已知缺陷（`TRAP-950-simd-shared-reduce`），
  且 950DT 上**本 kernel 的实现就是 SIMT**（`_build_step`）。
- **与同族兄弟算子（`engram_gate_conv` 选 SIMD）的量化对照**：它选 SIMD 的核心理由是
  「SIMT 每行一次 AllReduce（barrier + UB 中转），而行数 32–288、每核 4–9 行 × 4 次归约
  = 16–36 次 barrier」。**本算子每核只有 1 行、4 次归约**（`batch ≤ 17` 且 1 核 1 行），
  barrier 次数是它的 1/4–1/9；同时本算子的元素级链更长（`av/vn/conv/y` 4 条），
  SIMT 的「一个 `T.Parallel` 里写复合表达式」比 SIMD 的「逐 chunk 分寄存器拼装」更直接。
  因此**主选 SIMT**，并把 SIMD 作为备选交 Stage 4 实测（§1.6.3 (iii) 的三件套）。
- **为什么不用 `T.SimdVF` 做主选**：`T.simd` 的 `vld/vsts` 分布参数与 `vcvt` 的舍入/饱和参数
  「需要按具体路径核对」（`references/programming.md`），而本地没有「4 统计量 + 仿射广播
  + 两阶段」的 SIMD 端到端条目——风险/收益比不如已被 950DT 实现验证过的 SIMT 形态。
- 该偏离已在交付摘要的 `programming_mode` 字段回报 conductor。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | 仅 UB：`T.alloc_shared`（9 个张量缓冲：`h_ub/k_ub/v_ub/wh_ub/wv_ub/cw_ub/tap_ub/y_ub/vn_ub`）+ `T.alloc_fragment`（`local.fragment`，每 lane 数个寄存器）；reducer 的物理 scratch 由编译器追加（4 × `tile` × 4 B）。**不使用 L1/L0A/L0B/L0C** |
| 计算方式 | `with T.SimtVF(threads=L):` 内两段 `T.Parallel(tile, coalesced_width=1)` + 4 个 reducer epoch + 一段 `T.Parallel(1)` 标量链；逐元素表达式由编译器映射到 lane |
| 线程 / 并行 | `T.Kernel(batch)` 给核间并行（1 行/核）；核内 `L = min(512, tile)` 个 lane 给元素级并行；归约的跨 lane 合并由 `AscendAllReduce` 自动生成 |
| 同步 | 编译器自动插入（默认 `allow_autoschedule`）：`asc_sync_notify(MTE2,V)`/`wait` 在 VF 之前、`asc_sync_notify(V,MTE3)`/`wait` 在 UB→GM 之前（探针实读，§7.2）；**不写任何手工 flag** |

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式为 §1.6.1 的**优化后公式**（A0–A4、A6 已并入）。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `tile = next_pow2(d_padded)`；`bx = batch 行号`；`L = min(512, tile)` | 编译期/块级索引（§1.6.2 #1/#2） |
| 2 | 预取 `h_t[bx,:]`、`kv[0,bx,:]`、`kv[1,bx,:]`、`rms_w_h`、`rms_w_v`、`conv_w`、`conv_state[bx, mcl-(w-1-p)δ, :]`（`p<w-1`） | 每张量 GM 1 遍（§1.6.0 R3） |
| 3 | `S_h = Σ_{j<tile} h_j²`、`S_k = Σ k_j²`、`P = Σ h_j k_j w_hj²`、`S_v = Σ v_j²` | phase 1 单遍 4 路归约（A1/A2/A0） |
| 4 | `rh = rsqrt(S_h·inv_d + eps)`；`rk = rsqrt(S_k·inv_d + eps)`；`alpha = 1/(1+exp(-(rh·rk·P)·inv_sqrt_d))` | 标量链（A2/A6） |
| 5 | `rv = rsqrt(alpha²·S_v·inv_d + eps)` | 方差因子提出（A3） |
| 6 | `av = alpha·v_j`；`vn = av·rv·w_vj` | phase 2 前段（A4） |
| 7 | `new_conv_state[bx, mcl-1, j] = cast(vn)` | history 末行（§1.5 契约） |
| 8 | `c = cw[w-1,j]·vn + Σ_{p<w-1} cw[p,j]·cast(state[bx, mcl-(w-1-p)δ, j])` | 膨胀因果卷积（tap 索引恒合法，§1.6.2 #6） |
| 9 | `y_t[bx, j] = cast(c·(1/(1+exp(-c))) + av)` | SiLU + 残差（保留源形态，B0 已拒绝） |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 1 | 行级任务分解 | `T.Kernel(batch)` | 1-D 网格（平台不接受多维），`as bx` | SIMT |
| 2a | 行预取 | `T.copy(h_t[bx, :], h_ub)`、`T.copy(kv[0, bx, :], k_ub)`、`T.copy(kv[1, bx, :], v_ub)` | GM→UB 1-D 区域（`asc_copy_gm2ub_align`） | SIMT |
| 2b | 权重/tap 预取 | `T.copy(rms_w_h, wh_ub)`、`T.copy(rms_w_v, wv_ub)`、`T.copy(conv_w, cw_ub)`、`T.copy(conv_state[bx, mcl-(w-1-p)*dilation, :], tap_ub[p, :])` | `p` 为 Python 展开的编译期常量 | SIMT |
| 3 | 4 路统计量 | `acc = T.alloc_reducer((1,), "float32", op="sum")` + `T.reducer_init(acc)` + `T.reducer_update(acc[0], expr)`（×4 个 reducer，同处一个 `T.Parallel`） | `expr` = `h²` / `k²` / `h·k·w_h²` / `v²` | SIMT |
| 3b | 元素级 lane 并行 | `for i in T.Parallel(tile, coalesced_width=T.int32(1))` | 每 lane 1 元素 | SIMT |
| 3c | 归约落盘 | `T.finalize_reducer(acc, sh)` → `sh = T.alloc_fragment((1,), "float32")` | 4 组 | SIMT |
| 4/5 | 标量链 | `T.Parallel(1)` 内的 `T.rsqrt`/`T.exp`/乘除 | `alpha`/`rv` 各一个 `(1,)` fragment | SIMT |
| 6–9 | phase 2 复合表达式 | 单个 `T.Parallel(tile, coalesced_width=T.int32(1))` 内的 `T.cast`/乘加/`T.exp` | 含 `for p in range(w-1)` 的 Python 展开 | SIMT |
| 10 | 写出 | `T.copy(y_ub, y_t[bx, :])`、`T.copy(vn_ub, new_conv_state[bx, mcl-1, :])` | UB→GM 区域（`asc_copy_ub2gm_align`） | SIMT |
| 11 | 线程域 | `with T.SimtVF(threads=L):` | `L = min(512, tile)`，正整数 | SIMT |
| 12 | 缓冲分配 | `T.alloc_shared(shape, dtype)` / `T.alloc_fragment(shape, "float32")` | 见 §4.3 | SIMT |
| 13 | 编译 | `tilelang.compile(prim_func, target="ascend", out_idx=[])` | 输出全部由调用方预分配 | — |

> **未使用的 API（明确记录）**：`T.gemm`/`T.alloc_l1`/`alloc_l0*`（无矩阵乘）、
> `T.Pipelined`（本 kernel 单次扫描、无跨 iteration 的搬运/计算重叠可做，见 §6.3）、
> `T.reduce_sum`（改用 reducer，`TRAP-950-simd-shared-reduce`）、`T.Vector`/`T.MixedKernel`
> （无 AIC 段）、`T.simd.*`（备选形态，§1.6.3 (iii)②）。

### 3.3 计算伪代码

```python
import functools

import tilelang
import tilelang.ascend.language as T


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


@functools.lru_cache(maxsize=None)
def _build_step(batch: int, d: int, d_pad: int, max_conv_len: int, conv_kernel_size: int,
                dilation: int, eps: float, dtype_str: str, threads: int):
    w = conv_kernel_size
    tile = _next_pow2(d_pad)                    # 行宽（fragment 的另一维）
    exact = tile == d_pad                       # 尾列谓词在编译期消失（§1.6.2 #10）
    inv_d = 1.0 / float(d)                      # A6：常量倒数
    inv_sqrt_d = 1.0 / (float(d) ** 0.5)

    @T.prim_func
    def engram_decode_step(
        h_t: T.Tensor((batch, d_pad), dtype_str),
        kv: T.Tensor((2, batch, d_pad), "float32"),
        conv_state: T.Tensor((batch, max_conv_len, d_pad), dtype_str),
        rms_w_h: T.Tensor((d_pad,), dtype_str),
        rms_w_v: T.Tensor((d_pad,), dtype_str),
        conv_w: T.Tensor((w, d_pad), dtype_str),
        y_t: T.Tensor((batch, d_pad), dtype_str),
        new_conv_state: T.Tensor((batch, max_conv_len, d_pad), dtype_str),
    ):
        with T.Kernel(batch) as bx:             # 1 核 = 1 batch 行
            h_ub = T.alloc_shared((d_pad,), dtype_str)
            k_ub = T.alloc_shared((d_pad,), "float32")     # kv 是 fp32，直接搬（§0.5 #7）
            v_ub = T.alloc_shared((d_pad,), "float32")
            wh_ub = T.alloc_shared((d_pad,), dtype_str)
            wv_ub = T.alloc_shared((d_pad,), dtype_str)
            cw_ub = T.alloc_shared((w, d_pad), dtype_str)
            tap_ub = T.alloc_shared((w - 1, d_pad), dtype_str)   # w == 1 时为 0 行（合法）
            y_ub = T.alloc_shared((d_pad,), dtype_str)
            vn_ub = T.alloc_shared((d_pad,), dtype_str)

            T.copy(h_t[bx, :], h_ub)
            T.copy(kv[0, bx, :], k_ub)
            T.copy(kv[1, bx, :], v_ub)
            T.copy(rms_w_h, wh_ub)
            T.copy(rms_w_v, wv_ub)
            T.copy(conv_w, cw_ub)
            for p in range(w - 1):              # 编译期展开：tap 索引恒合法
                T.copy(conv_state[bx, max_conv_len - (w - 1 - p) * dilation, :], tap_ub[p, :])

            with T.SimtVF(threads=threads):
                vf = T.alloc_fragment((tile,), "float32")
                vnf = T.alloc_fragment((tile,), "float32")
                cof = T.alloc_fragment((tile,), "float32")

                acc_h = T.alloc_reducer((1,), "float32", op="sum")
                acc_k = T.alloc_reducer((1,), "float32", op="sum")
                acc_p = T.alloc_reducer((1,), "float32", op="sum")
                acc_v = T.alloc_reducer((1,), "float32", op="sum")
                T.reducer_init(acc_h)
                T.reducer_init(acc_k)
                T.reducer_init(acc_p)
                T.reducer_init(acc_v)
                for i in T.Parallel(tile, coalesced_width=T.int32(1)):      # phase 1（A1）
                    vf[i] = v_ub[i]
                    T.reducer_update(acc_h[0], T.cast(h_ub[i], "float32") * T.cast(h_ub[i], "float32"))
                    T.reducer_update(acc_k[0], k_ub[i] * k_ub[i])
                    T.reducer_update(acc_p[0], T.cast(h_ub[i], "float32") * k_ub[i]
                                     * T.cast(wh_ub[i], "float32") * T.cast(wh_ub[i], "float32"))
                    T.reducer_update(acc_v[0], v_ub[i] * v_ub[i])

                sh = T.alloc_fragment((1,), "float32")
                sk = T.alloc_fragment((1,), "float32")
                sp = T.alloc_fragment((1,), "float32")
                sv = T.alloc_fragment((1,), "float32")
                T.finalize_reducer(acc_h, sh)
                T.finalize_reducer(acc_k, sk)
                T.finalize_reducer(acc_p, sp)
                T.finalize_reducer(acc_v, sv)

                alpha = T.alloc_fragment((1,), "float32")
                rv = T.alloc_fragment((1,), "float32")
                for i in T.Parallel(1):                                      # 标量链（A2/A3/A6）
                    alpha[i] = 1.0 / (1.0 + T.exp(
                        -(T.rsqrt(sh[0] * inv_d + eps) * T.rsqrt(sk[0] * inv_d + eps)
                          * sp[0] * inv_sqrt_d)))
                    rv[i] = T.rsqrt(alpha[0] * alpha[0] * sv[0] * inv_d + eps)

                for i in T.Parallel(tile, coalesced_width=T.int32(1)):       # phase 2（A4）
                    vnf[i] = alpha[0] * vf[i] * rv[0] * T.cast(wv_ub[i], "float32")
                    cof[i] = T.cast(cw_ub[w - 1, i], "float32") * vnf[i]
                    for p in range(w - 1):                                   # Python 展开
                        cof[i] = cof[i] + T.cast(cw_ub[p, i], "float32") * T.cast(tap_ub[p, i], "float32")
                    y_ub[i] = T.cast(cof[i] * (1.0 / (1.0 + T.exp(-cof[i]))) + alpha[0] * vf[i],
                                     dtype_str)
                    vn_ub[i] = T.cast(vnf[i], dtype_str)

            T.copy(y_ub, y_t[bx, :])
            T.copy(vn_ub, new_conv_state[bx, max_conv_len - 1, :])

    return tilelang.compile(engram_decode_step, target="ascend", out_idx=[])
```

要点：`tap_ub` 在 `w == 1` 时是**零行分配**（本工具链合法，探针 S1 第七变体编译通过），
`for p in range(0)` 不产生代码；`exact` 分支（尾列谓词）在本文件的探针里以
`if i < d_pad` 形态覆盖（`d=600`/`d_pad=768` 变体），生产代码按 `exact` 在 Python 层选择是否生成谓词；
`y_ub`/`vn_ub` 的写入是 UB 内的 lane 并行写（目的偏移 0，避开 UB→UB 错位陷阱）。

### 3.4 API 可行性确认

| API | 来源 / 证据 | 是否验证 |
|-----|------------|---------|
| `T.Kernel(N)`（1-D，无 `threads`） | `.agents/platforms/toolchains/ascend/references/programming.md`「API 与执行区域」第 1 行 | ✅ 文档 + 探针 S1/S2（`__vector__` 核函数） |
| `T.SimtVF(threads=L)` | 同上第 2 行（`frame.py:SimtVF`，正整数/≤3 维） | ✅ 探针 S1（`__launch_bounds__(512)` / `(256)` / `(128)`） |
| `T.alloc_shared` / `T.alloc_fragment` | 同上「分配、布局和搬运」表 | ✅ 探针 S1/S2 |
| `T.alloc_reducer(shape, dtype, op="sum")` + `reducer_init` / `reducer_update` / `finalize_reducer(acc, dst)` | `allocate.py:alloc_reducer`、`reduce_op.py:reducer_init/reducer_update/finalize_reducer`（v2 形态：`dst` 给出即独立 fragment 输出） | ✅ 探针 S1（4 个并发 epoch → 4 条 `AscendAllReduce<tl::SumOp, 512, 1, 0>`） |
| `T.Parallel(n, coalesced_width=T.int32(1))` | `PL-950-row-primitives`（每 lane 1 元素） | ✅ 探针 S1（生成 `buf[tid + k]` / `#pragma unroll`×4） |
| `T.copy` GM→UB / UB→GM 区域 | `references/programming.md`「分配、布局和搬运」；`PL-950-vector-execution` | ✅ 探针 S1（9 条 `asc_copy_gm2ub_align` + 2 条 `asc_copy_ub2gm_align`） |
| `T.cast` / `T.exp` / `T.rsqrt` / `T.if_then_else` / `T.int32` | `PL-950-row-primitives`（带 weight 的 RMSNorm、exp、sqrt 均有条目） | ✅ 探针 S1（生成 `expf`/`rsqrtf`/`(half)`） |
| 零行 `T.alloc_shared((0, d_pad), dtype)` | 本工具链实现允许零 extent（独立静态探针验证，见 §11.1 第六变体） | ✅ 探针 S1 第七变体（`w=1`） |
| `T.simd.*`（`vld/vsts/vcvt/vcadd/vdupv/vmul/vadd/vexp/vdiv/update_mask/alloc_local`） | 静态检查 `tilelang.ascend.language.simd` 的 94 个导出符号全部存在；用法先例 = `examples/engram_gate_conv/_engram_gate_conv_fwd_kernel/DESIGN.md` §3.2 | ⚠️ 仅备选形态（§1.6.3 (iii)②），**未编译**，参数需 Stage 4 逐路径核对 |
| `T.reduce_sum(..., dim=1)`（高层双层归约） | `reduce_op.py` + `TRAP-950-simd-shared-reduce` | ❌ **不采用**（已知输出布局缺陷） |
| `T.Pipelined` / `T.annotate_buffer_versions` | `references/programming.md`「同步与缓冲版本」 | ❌ 不使用（§6.3 说明理由） |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| `T.Kernel` 只接受 1-D grid | **是** | `T.Kernel(batch)`（本身就是 1-D） |
| `T.Kernel` 没有 `threads` 参数 | **是** | 线程域由 `T.SimtVF(threads=)` 表达（§0.6 R5） |
| 三维 Kernel（M/N/K 三分块）| **否** | 本算子无 M/N/K 分块：每核一行，行内满宽 |
| L0C 容量 / L0A·L0B 分形 / GEMM 非整除尾块 | **否** | 不使用任何 Cube / L0 资源（§3.4 的「未使用 API」） |
| 高层 SIMD 二维 shared→shared 归约缺陷（`CG-950-simd-shared-reduce`） | **是（已规避）** | 用 SIMT reducer（§0.6 R2），不用 `T.reduce_sum` |
| `ascend.batch_schedule` 跨 block barrier 未认证（`CG-950-batch-barrier`） | **否** | 每块独立写自己的行（`y_t[bx]`、`new_conv_state[bx, mcl-1]`），无跨块依赖 |
| UB 容量 248 KiB（`253952 B`） | 是（用量低） | 24 KiB / 9.7 %（§4.5） |
| `d_padded % 16 == 0`（尾轴向量对齐） | **是** | host 保证 `d_padded = align_up(d, 256)`；5 条 workload 的 `d_pad ∈ {256,512}` 且都是 2 的幂 → `tile == d_pad`，无尾 lane |
| `max_conv_len >= dilation*(conv_kernel_size-1)` | 是（上游 `__init__` 校验，第 293–298 行） | 本 kernel 依赖它来删除 tap 索引守卫（§1.6.2 #6） |
| `w == 1` / `max_conv_len == 1`（fixture 第 4 行） | **是** | `tap_ub` 零行、`for p in range(0)` 空、末行索引 `mcl-1 = 0` |
| 动态 shape | **否** | `batch/d/d_pad/mcl/w/δ/eps/dtype` 全是编译期常量（§4.6） |

#### 3.5.2 参考实现差异说明

| 差异项 | 参考实现（GPU 源算子） | 本项目（Ascend 950PR） | 转换方案 |
|--------|----------------------|----------------------|---------|
| Kernel 结构 | `T.Kernel(batch)` + `threads=128` | `T.Kernel(batch)` + `T.SimtVF(threads=L)` | R5（§0.6） |
| 行归约 | `T.reduce_sum(frag2d, out, dim=1)` | `T.alloc_reducer` + `reducer_update` + `finalize_reducer` | R2/R3 |
| 统计量组织 | 每统计量一个 pass + 单独 reduce | 4 路并发 reducer 同处一遍 | R3 + A1 |
| 元素级链 | 15 个 pass（逐 pass 语义单一） | 2 个 pass（相位 1/2） | R1 + A1–A4 |
| 标量归一化 | `x / float(d)`、`dot / T.sqrt(float(d))` | `x · inv_d`、`dot · inv_sqrt_d` | A6 |
| 内存层级 | fragment（寄存器）+ `T.copy` 权重 | UB（`alloc_shared`）+ `local.fragment` | §4.4 |
| cache 平移归属 | 源算子：**不在本 kernel**（在 project 段） | 同（项目段负责） | 与 950DT 参考不同：它把平移挪进 step kernel，本任务按 harness 边界不照搬（§0.5 #11） |
| `kv` 归一化精度 | 直接 fp32 | 直接 fp32 | **刻意不照搬** 950DT 参考的 dtype 往返（§0.5 #7 / B1） |
| 尾列 | 无（`d_padded` 直接满宽） | `tile = next_pow2(d_pad)`，`d_pad` 非 2 的幂时生成谓词 | §1.6.2 #10 |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/engram_gate_conv/_engram_gate_conv_fwd_kernel/DESIGN.md` | **高**（同 op family、同平台的 gate+RMSNorm+4-tap conv+SiLU） | SIMD 形态与 chunk 划分、`update_mask` 尾 lane 处理、`vcadd` 归约、分核三要素范式、§1.6.3 的实验裁决写法 |
| `/workspace/TileOPs-ascend950/src/tileops/kernels/ascend/sequence_modeling/decode.py` | **高**（同算子族的 950DT 已验证实现，只读） | `_build_step` 的 SIMT + reducer + UB 预取结构（借用点 O10/O12，偏离点 O11） |
| `examples/engram_decode/_engram_project_kernel/DESIGN.md` | **高**（同一 op 的第 1 个 launch） | 跨 launch 契约、roofline 口径、分核三要素的实查记录、`statectl` 平台记录引用 |
| `.agents/platforms/toolchains/ascend/knowledge/PL-950-row-primitives.md` | **高**（行归约/广播的最小正确性证据） | reducer 的 `init/update/finalize` 用法、无效 lane 置 0、归一化分母用有效长度 |
| `.agents/platforms/toolchains/ascend/knowledge/operator-patterns.md` | **高**（PL-950-normalization） | 带 affine 的 RMSNorm 端到端写法、fp16/bf16/fp32 的阈值 |
| `.agents/platforms/toolchains/ascend/knowledge/numerical-boundaries.md` | 中（**反例**） | `TRAP-950-simd-shared-reduce`（不用的路径）与 `PL-950-simd-reduce`（备选形态的归约写法） |
| `.agents/platforms/toolchains/ascend/knowledge/execution-patterns.md` | 低 | 混合 kernel 与手工同步的边界（本算子都不需要） |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `h_t` | `(batch, d_padded)` | float16 / bfloat16 | 本步隐状态（host 已把最后一维 0 填充到 `d_padded`） |
| `kv` | `(2, batch, d_padded)` | **float32** | `kv[0]`=K、`kv[1]`=V；**project 段产出**，本 kernel 只读 |
| `conv_state` | `(batch, max_conv_len, d_padded)` | 同 `h_t` | 已左填充到 `max_conv_len` 的 history；只读 `w-1` 个 tap 行 |
| `rms_w_h` | `(d_padded,)` | 同 `h_t` | h/k 共用（k 也用 `rms_w_h`，与源一致） |
| `rms_w_v` | `(d_padded,)` | 同 `h_t` | v̂ 用 |
| `conv_w` | `(conv_kernel_size, d_padded)` | 同 `h_t` | 深度可分离卷积权重 |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `y_t` | `(batch, d_padded)` | 同 `h_t` | 整行写入 `[0, d_padded)` |
| `new_conv_state` | `(batch, max_conv_len, d_padded)` | 同 `h_t` | **只写第 `max_conv_len-1` 行**；前 `mcl-1` 行由 project 段写、本 kernel 不读不写（§1.5 契约） |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| `h_ub` | `(d_pad,)` | 同 `h_t` | UB | `h_t[bx,:]` 行 |
| `k_ub` | `(d_pad,)` | **float32** | UB | `kv[0,bx,:]`（fp32 直搬） |
| `v_ub` | `(d_pad,)` | **float32** | UB | `kv[1,bx,:]` |
| `wh_ub` / `wv_ub` | `(d_pad,)` | 同 `h_t` | UB | 两个 RMSNorm 权重 |
| `cw_ub` | `(w, d_pad)` | 同 `h_t` | UB | 卷积权重（`w` 行） |
| `tap_ub` | `(w-1, d_pad)` | 同 `h_t` | UB | `w-1` 个 history tap 行（`w==1` 时 0 行） |
| `y_ub` / `vn_ub` | `(d_pad,)` | 同 `h_t` | UB | 两行输出（GC 前的 cast 结果） |
| `vf` / `vnf` / `cof` | `(tile,)` | float32 | fragment（寄存器，每 lane 1 个） | `v` 的复用副本、`vhat_norm`、`conv_out` |
| `sh`/`sk`/`sp`/`sv`/`alpha`/`rv` | `(1,)` | float32 | fragment（标量） | 4 个统计量 + 2 个缩放因子 |
| 4× reducer scratch | `(tile,)` ×4 | float32 | UB（编译器追加） | `AscendAllReduce` 的跨 lane 中转（探针实读偏移 2816/3328/3840/4352 float） |

### 4.4 内存搬运路径

```
                        ┌─────────────── Vector（AIV，单核）───────────────┐
GM[h_t]                --MTE2--> UB[h_ub]   --\
GM[kv[0],kv[1]] (fp32) --MTE2--> UB[k_ub,v_ub] --\
GM[rms_w_h,rms_w_v]    --MTE2--> UB[wh_ub,wv_ub] --+--> VF: phase 1（T.Parallel + 4 reducer, 寄存器累加）
GM[conv_w]             --MTE2--> UB[cw_ub]   --/             |
GM[conv_state] taps    --MTE2--> UB[tap_ub]  --/      4× AscendAllReduce（UB scratch）
                                                             |
                                                       标量链（每行一次）
                                                             |
                                                      VF: phase 2（T.Parallel）
                                                             |
                        UB[vn_ub] --MTE3--> GM[new_conv_state][:, mcl-1, :]
                        UB[y_ub]  --MTE3--> GM[y_t]
                        └──────────────────────────────────────────────────┘
```

两条 MTE3 写回是**互不重叠**的 GM 区域；整个 kernel 只有一个 VF 区段，无 AIC 参与，
不涉及 L1/L0A/L0B/L0C、不涉及跨核 flag。

### 4.5 UB 内存预算

> 数据来源：平台 `design_budget.capacity_bytes`（CANN 9.1.0 `Ascend950PR_9579.ini` `[AICoreSpec]`，
> md5 `2f0eca1f7ca70f62ee7d6d33990c3be1`，2026-11-23 落库）；本机 9.2.1 INI 复核值一致
> （`ub_size=253952, l1_size=524288, l0_c_size=262144, l0_a_size=65536, l0_b_size=65536`）。

**最大 workload = serving-b4**（batch=4、`d_pad=512`、`w=4`、fp16）

| Buffer | Shape | dtype | 版本数 | 占用 |
|--------|-------|-------|--------|------|
| `h_ub` | (512,) | fp16 | 1 | 1 024 B |
| `k_ub` | (512,) | fp32 | 1 | 2 048 B |
| `v_ub` | (512,) | fp32 | 1 | 2 048 B |
| `wh_ub` | (512,) | fp16 | 1 | 1 024 B |
| `wv_ub` | (512,) | fp16 | 1 | 1 024 B |
| `cw_ub` | (4, 512) | fp16 | 1 | 4 096 B |
| `tap_ub` | (3, 512) | fp16 | 1 | 3 072 B |
| `y_ub` | (512,) | fp16 | 1 | 1 024 B |
| `vn_ub` | (512,) | fp16 | 1 | 1 024 B |
| reducer scratch ×4 | (512,) ×4 | fp32 | 1 | 8 192 B |
| **合计（手工核算）** | | | | **23 552 B = 23 KiB** |
| **探针实测（生成源码的最大 UB 偏移）** | | | | **24 576 B = 24 KiB**（含对齐填充） |

UB 合计 **24 KiB**，容量 248 KiB（`253952 B`），占比 **9.7 %**。**无 L1 / L0A / L0B / L0C 占用**。
`serving-b8`（`d_pad=256`）为 12 KiB；`decode-s1` 为 12 KiB。缓冲版本数为 1（无可双缓冲的循环，
见 §6.3）；不申报 `annotate_buffer_versions`。

### 4.6 动态轴定义

**无运行时动态轴**：`batch`、`d`、`d_padded`、`max_conv_len`、`conv_kernel_size`、`dilation`、
`eps`、`dtype` 全部是**编译期常量**（工厂键 + `functools.lru_cache` + `tilelang.compile` 特化）。
op 层按 `(batch, d_mem, d, dtype, device_index)` 缓存已编译 kernel，`conv_state.shape[1]`
故意不进键（短 history 由 host 左填充到 `max_conv_len`，§0.1）。因此 `batch` 的「动态」只体现为
跨调用换 shape 时重新编译，单次执行内所有循环边界都是常量。
**host 零填充契约**（A0 的前提）：`h_t`/`conv_state`/`rms_w_*`/`conv_w` 的最后一维 0 填充到
`d_padded`，`kv` 的填充列因 `W` 的填充列为 0 而恰为 `+0.0`。

### 4.7 JIT 配置

```python
@functools.lru_cache(maxsize=None)
def _build_step(batch, d, d_pad, max_conv_len, conv_kernel_size, dilation, eps, dtype_str, threads):
    ...
    return tilelang.compile(engram_decode_step, target="ascend", out_idx=[])
```

- `out_idx=[]`：两个输出都由调用方预分配（与源算子一致）。
- 不使用 `@tilelang.jit` 装饰器：编译期常量较多、需要 `lru_cache` 复用（与 950DT 只读参考的
  `_build_step`、`examples/engram_decode/_engram_project_kernel/DESIGN.md` 的 §4.7 写法一致）。
- 编译标志沿用平台默认（`target="ascend"`）；`threads` 由工厂算好（`min(512, next_pow2(d_pad))`）。
- **配置来源**：`threads` 见 §5.6；harness 的 `default_config["step_threads"] = 128` 是 GPU 派生值，
  Stage 3 必须按 §5.6 改写（函数签名保持 `threads` 形参，便于 Stage 4 A/B）。

## 5. Tiling 策略

### 5.1 计算类型

**类型**: **归约 + 逐元素（纯 Vector）**。

**判定依据**：本 kernel 的全部计算 = 4 个行归约（沿 `d`）+ 逐元素乘加/超越函数 + 2 行写出，
**没有矩阵乘**（深度卷积的权重矩阵是对角的，无公共收缩轴，§1.6.0 R1 候选 6），
因此不用 Cube，也不构成混合模式。

### 5.2 Block 划分

```python
grid     = batch                    # 1 个逻辑块 = 1 个 batch 行
threads  = min(512, tile)           # 核内 lane 数（每 lane 1 元素）
tile     = next_pow2(d_padded)      # 行宽（fragment 第二维 / T.Parallel 范围）
reducers = 4                        # S_h, S_k, P, S_v（并发 epoch）
taps     = w - 1                    # 编译期展开的 tap 数
```

每个 block 只写自己的 `y_t[bx, :]` 与 `new_conv_state[bx, mcl-1, :]`（互不重叠）。
`tile` 与 `d_pad` 的关系决定是否生成尾列谓词（§1.6.2 #10）。

### 5.3 约束分析

- **对齐约束**：`d_padded` 由 host 保证 `align_up(d, ALIGNMENT=256)`
  （`ALIGNMENT` 来自 `examples/TileOPs/tileops/kernels/sequence_modeling/engram_decode/_engram_decode_fwd_kernels.py`），
  因此 `d_pad % 64 == 0` 恒成立 → fp32 向量宽度（64）整除、`T.Parallel` 无尾 lane；
  5 条 workload 的 `d_pad ∈ {256, 512}` 还是 2 的幂 → `tile == d_pad`、谓词整段消失。
- **UB 容量**：24 KiB ≤ 248 KiB（9.7 %，§4.5）。
- **无 L1/L0 约束**：不使用 Cube 资源，因此 L0B 的 64 KiB 与 L0C 的 256 KiB 与本算子无关。
- **规约轴容量**：4 个 reducer 的 scratch 各 `tile·4 B`（最大 2 KiB），合计 8 KiB，已计入。
- **线程域**：`T.SimtVF(threads=L)` 要求正整数（`frame.py:SimtVF` 拒绝 0 与 4 维），
  `L ∈ {256, 512}`（workload）与 `{128}`（A/B 变体）都合法。
- **静态边界**：`batch/tile/taps` 全是编译期常量；`tap` 索引 `mcl-(w-1-p)δ` 是常量表达式。
- **K/M/N 尾块**：不适用（无 GEMM）；行尾列由 `tile` vs `d_pad` 处理。

### 5.4 注意事项

- **`max_conv_len == δ·(w-1)`**（`decode-s1` 的 `mcl=9, δ=3, w=4`）：`p=0` 的 tap 索引恰为 0
  （最老一格），必须覆盖该边界（§8.3）。
- **`w == 1` / `mcl == 1`**（fixture 第 4 行）：`tap_ub` 零行、tap 循环为空、末行索引 0。
- **`batch == 1`**：`grid = 1`，只有 1 个 Vector 核在工作——这是**有意为之**（§5.5 ③），
  不是并行度缺陷：每行的计算量只有 ≈ 0.02 µs 量级，拆行要付 launch/barrier 的代价。
- **`batch > 8`**（fixture 的 17 行）：`grid = 17 ≤ 56`，仍单波；每核仍 1 行。
- **partial cache（`L < mcl`）**：kernel 侧不感知（host 左填充），末行恒写 `mcl-1`。
- **`d_padded` 非 2 的幂**（如 `d=600` → `d_pad=768`、`tile=1024`）：走谓词分支，
  fragment 尾 lane 必须置 0（否则归约被垃圾污染）——`d=600`/`d_pad=768` 的探针变体覆盖该形态。

### 5.5 分核策略（物理核数适配）

> 三要素标准：`.agents/skills/_shared/standards/core-split-strategy.md` §1。

- **① 逻辑核数计算**：`num_logical_kernels = batch`（每个逻辑块 = 一个 batch 行的完整两遍扫描；
  本算子没有 N/M 方向的 tile 划分——输出网格就是 `(batch,)`）。

  | workload | `batch`（逻辑核数） | `threads` | 波次（/56 Vector 核） |
  |---|---|---|---|
  | decode-s1 | **1** | 256 | 0.018 |
  | serving-b1 | **1** | 256 | 0.018 |
  | serving-b4 | **4** | 512 | 0.071 |
  | serving-b8 | **8** | 256 | 0.143 |
  | decode-partial-cache | **2** | 256 | 0.036 |
  | （fixture）batch-17 | **17** | 256 | 0.304 |
  | （fixture）one-tap | **1** | 256 | 0.018 |

- **② 物理核数依据（任务平台记录，非假设值）**：目标型号 **ascend-950pr-9579**（架构 dav-3510）。
  平台 `statectl platform --task-dir examples/engram_decode/_engram_step_kernel` 落库的实查记录
  （`hardware.core_source`，读取日期 2026-11-23）：

  ```
  Ascend950PR_9579.ini [SoCInfo] ai_core_cnt=28, cube_core_cnt=28, vector_core_cnt=56;
  runtime probe rtGetAiCoreCount()=28 on Ascend950PR_9579
  observed_cores = {cube: 28, vector: 56}
  ```

  平台 ruleset 规定的查询口径是 `torch.npu.get_device_properties`
  （adapter `core_evidence()` → `get_device_properties`，实现见
  `.agents/tools/toolchains/ascend.py` 的 `probe_device`）：

  ```python
  import torch, torch_npu
  p = torch.npu.get_device_properties(torch.npu.current_device())
  p.name, p.cube_core_num, p.vector_core_num   # -> "Ascend950PR_9579", 28, 56
  ```

  本次设计期该卡被 `engram` 的 bwd Stage 3 占用（只做 `ASCEND_RT_VISIBLE_DEVICES=""` 的无卡编译），
  因此**引用平台已落库的探测记录**；另用 CPU 侧文件读取复核型号配置（2026-11-23，本机
  `/usr/local/Ascend/cann-9.2.1/x86_64-linux/data/platform_config/Ascend950PR_9579.ini`，
  `grep -E "ai_core_cnt|cube_core_cnt|vector_core_cnt"` 输出 `28 / 28 / 56`，`cube_freq=1650`）。
  本算子只用 **Vector 核**（56 个），逻辑块到物理核的映射：1 逻辑块 = 1 个 AIV 任务，
  由硬件块调度器按波次派发；UB 为每核私有，block 之间无共享存储，故无 occupancy 约束。
- **③ 规模判定与分核方案**：**逻辑核数 ≤ 物理核数（最大 17 ≤ 56）→ 无需适配**。
  依据：① 的最坏情形是 fixture 的 17 个逻辑块，单波即可全部驻留 56 个 Vector 核，
  不存在第二波调度；不使用 `T.Persistent` 核内串行，也不做块数对齐
  （把 1–8 个块抬到 56 的整数倍只会增加空块与描述符开销）。
  **为什么不再往 `d` 方向拆核**（量化依据，§1.6.0 R3 候选 4）：
  - 每行的计算量 ≈ 20·d = 10 kFLOP；单核向量峰值若按 0.42 TFLOP/s（128 lane × 2 FLOP × 1.65 GHz，
    **未实测假设**）估，计算 ≈ 0.024 µs，远小于发射项（≈5 µs）。拆行只能减少这 0.024 µs。
  - 拆行的代价是硬性的：跨核归并**要么**用 `ascend.batch_schedule` 的跨 block barrier
    （`CG-950-batch-barrier` **未认证**），**要么**加第二次 launch（≈5 µs = 下界的 ~83 %）。
  - 交易不划算：省 ≤ 0.024 µs，付 ≥ 5 µs。
  因此本算子的分核结论是「1 行 1 核、批内不拆行」，并把它写进 §1.6.3 (iii) 的备选重算
  （SIMD 形态的分核结论相同）。
  **余量归属**：整算子（2 次 launch）的主要成本是两次固定开销；要再降只能把 step 段并进
  project 段的 launch（§1.6.0 R3 候选 5）——那是跨设计的函数边界变更，本阶段不做（§9.1）。

### 5.6 配置表（默认，进 Stage 3）

| 参数 | 值 | 依据 |
|------|-----|------|
| `grid` | `batch` | §5.2（1 行 1 核） |
| `tile` | `next_pow2(d_padded)`（workload 上 = `d_pad`） | §1.6.3 (i)#1 |
| `threads` | `min(512, tile)`（workload 上 512 / 256） | §1.6.3 (ii)#1 |
| reducers | 4（并发 epoch） | §0.6 R3 / A1 |
| tap 处理 | Python 展开 `w-1` 项 | §1.6.2 #3 |
| `eps` | `1e-6`（manifest 默认） | 工厂键 |
| 缓冲版本数 | 1（不申报 `annotate_buffer_versions`） | §6.3 |
| `pass_configs` | `{}`（默认流水线） | §7.3 |
| 冗余参数 | 无（`step_threads` 改名 `threads`） | §4.7 |

---

## 6. 循环与调度结构

### 6.1 循环结构总结

> 逐元素计算已按 §1.6.2 全部消除（0 处逐元素标量循环）——本表只列 block 级调度、归约原语与
> 编译期展开的结构。

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| batch 行（`batch`） | block 级并行 | `T.Kernel(batch)` | 每块一行，输出区域互不重叠（§5.5） |
| 行内元素（`tile`），phase 1 | 线程并行 + 归约 | `T.Parallel(tile, coalesced_width=T.int32(1))` + 4×`T.reducer_update` | 每 lane 1 元素；4 条累加链同遍（A1） |
| 4 个统计量 | 跨 lane 归约 | `T.finalize_reducer(acc, dst)` ×4 | 编译器生成 `AscendAllReduce<SumOp, L, 1, 0>` |
| 标量链 | 单元素线程并行（每行一次） | `T.Parallel(1)` | `T.rsqrt`/`T.exp`/`T.div`；O(1)/行（§1.6.2 #7） |
| tap（`w-1`） | 编译期展开（Python `for`） | `range(w-1)`（两侧都是 Python 层展开） | `w` 是编译期常量；运行时无 tap 循环（§1.6.2 #3） |
| 行内元素（`tile`），phase 2 | 线程并行 | 同 phase 1 的 `T.Parallel` | 一次扫描出 3 类结果（§1.6.2 #8） |
| 元素级 | **无** | — | 装载/写回全是 `T.copy` DMA；元素级表达式在 lane 并行区内 |

### 6.2 循环伪代码

```python
with T.Kernel(batch) as bx:                       # block 级并行（1 行/核）
    # 预取：9 条 DMA（tap 由 Python 展开为 w-1 条）
    T.copy(h_t[bx, :], h_ub); T.copy(kv[0, bx, :], k_ub); T.copy(kv[1, bx, :], v_ub)
    T.copy(rms_w_h, wh_ub); T.copy(rms_w_v, wv_ub); T.copy(conv_w, cw_ub)
    for p in range(w - 1):                        # 编译期展开
        T.copy(conv_state[bx, max_conv_len - (w - 1 - p) * dilation, :], tap_ub[p, :])

    with T.SimtVF(threads=threads):               # lane 并行域（每 lane 1 元素）
        # ---- phase 1：一次扫描 + 4 路归约 ----
        for i in T.Parallel(tile, coalesced_width=T.int32(1)):
            vf[i] = v_ub[i]
            T.reducer_update(acc_h[0], T.cast(h_ub[i], "float32") * T.cast(h_ub[i], "float32"))
            T.reducer_update(acc_k[0], k_ub[i] * k_ub[i])
            T.reducer_update(acc_p[0], T.cast(h_ub[i], "float32") * k_ub[i]
                             * T.cast(wh_ub[i], "float32") * T.cast(wh_ub[i], "float32"))
            T.reducer_update(acc_v[0], v_ub[i] * v_ub[i])
        T.finalize_reducer(acc_h, sh); T.finalize_reducer(acc_k, sk)
        T.finalize_reducer(acc_p, sp); T.finalize_reducer(acc_v, sv)
        # ---- 标量链（每行一次）----
        for i in T.Parallel(1):
            alpha[i] = 1/(1+exp(-(rsqrt(sh·inv_d+eps)·rsqrt(sk·inv_d+eps)·sp·inv_sqrt_d)))
            rv[i]    = rsqrt(alpha²·sv·inv_d + eps)
        # ---- phase 2：一次扫描 + 3 路结果 ----
        for i in T.Parallel(tile, coalesced_width=T.int32(1)):
            vnf[i] = alpha[0] * vf[i] * rv[0] * T.cast(wv_ub[i], "float32")   # A4
            cof[i] = T.cast(cw_ub[w - 1, i], "float32") * vnf[i]
            for p in range(w - 1):                                            # 编译期展开
                cof[i] += T.cast(cw_ub[p, i], "float32") * T.cast(tap_ub[p, i], "float32")
            y_ub[i]  = cast(cof[i] * (1/(1+exp(-cof[i]))) + alpha[0] * vf[i])
            vn_ub[i] = cast(vnf[i])
    # 写回（两条 DMA）
    T.copy(y_ub, y_t[bx, :]); T.copy(vn_ub, new_conv_state[bx, max_conv_len - 1, :])
```

### 6.3 流水线优化

**不使用 `T.Pipelined`，也不申报缓冲版本。** 理由：

1. 本 kernel 的 GM 访问**全部集中在开头的 9 条预取**（编译器已把它们批量排到同一个 MTE2 段、
   在 VF 之前一次 `asc_sync_wait` 之后才开始计算——§11.1 的生成源码），VF 内部不再有 GM 访问，
   因此没有「搬运与计算重叠」的对象。
2. 行内只有 2 遍扫描，且第二遍依赖第一遍的 4 个标量（`alpha`、`rv`）——这是**循环携带的数据依赖**，
   双缓冲无法消除（§1.6.0 R2 的 inherent 2-phase 边界）。
3. `tap` 展开后没有运行时循环可流水；`batch` 维度是 block 并行（硬件调度器负责波次），
   不需要软件流水。

若 Stage 4 要试「`batch` 行内串行处理多行」的 persistent 形态（§5.5 ③ 已给出量化否决依据：
1–17 块 ≪ 56 核），那才会出现可流水的循环——不属于本设计冻结范围。

### 6.4 尾块处理

| 尾块 | 触发条件 | 处理 |
|------|----------|------|
| 行尾列（`tile > d_pad`） | `d_padded` 不是 2 的幂（如 `d=600` → `d_pad=768`、`tile=1024`） | fragment 尾 lane 由 `T.if_then_else(i < d_pad, …, 0)` 置 0（读 UB 与归约贡献都为 0）；写回用 `if i < d_pad` 守卫。workload 上 `tile == d_pad` → **整段不生成** |
| tap 数 | `w == 1` | `tap_ub` 零行分配、`for p in range(0)` 空 |
| 末行索引 | `mcl == 1` | `new_conv_state[bx, 0, :]`（= `mcl-1`） |
| batch 尾块 | — | 不存在（`grid = batch`，一行一块） |
| 归约域 | `tile` 与 `d` 不等 | 归约跑满 `tile`（填充列恒 0），**分母仍是 `d`**（A0 的前提） |

## 7. 同步策略

### 7.1 同步模式

**模式**: **自动同步**（`ascend/pipeline.py` 的 `allow_autoschedule` 默认开启：AutoSchedule →
InsertSync → MaterializeMultiBuffer）。本设计**不写**任何 `ascend_set_flag` /
`ascend_cross_core_*` / `ascend_get_buf` 手工同步——平台 ruleset 明确「手工跨核 flag 配对未验证」，
且本算子没有跨 block、跨 AIC/AIV 的数据依赖（§1.5）。

### 7.2 同步点说明

生成代码中由编译器插入的同步（探针 S1 的 `get_kernel_source()` 实读，`serving-b4` 形态）：

| 位置 | 同步 API（生成代码） | 理由 |
|------|---------------------|------|
| 9 条 GM→UB 预取 → VF 开始 | `asc_sync_notify(PIPE_MTE2, PIPE_V, 0)` + `asc_sync_wait(...)` | 等全部行/权重/tap 到达 UB 再算 |
| 4 次 `AscendAllReduce` | 由 `AscendAllReduce<…>::run(acc, scratch)` 内部完成（UB 中转 + 线程同步），scratch 偏移 2816/3328/3840/4352 float | 跨 lane 归约的必经同步 |
| VF 内 phase 2 之前 | `asc_syncthreads()`（编译器插入，位于 `cof` 初始化与 tap 累加之间） | reducer scratch 的复用保护（保守，单次） |
| VF 结束 → UB→GM 写回 | `asc_sync_notify(PIPE_V, PIPE_MTE3, 0)` + `asc_sync_wait(...)` | 等 `y_ub`/`vn_ub` 写完再搬出 |
| 两条 UB→GM 之间 | 无（同一 MTE3 队列顺序执行） | 写集不重叠 |

`w == 1` 时（无 tap 预取循环）生成的是 1 条 `asc_sync_notify(MTE2,V)` + 1 条 `wait`
（不再嵌在 `p` 循环里），语义相同（探针第七变体实读）。

### 7.3 pass_configs 配置

```python
# 平台默认流水线：allow_autoschedule 开启（不显式关闭）
pass_configs = {}   # 不覆盖 AutoSchedule / InsertSync / MaterializeMultiBuffer
```

不传任何自定义 pass 配置：本算子的正确性依赖自动同步（尤其是 reducer 的跨 lane 合并），
关闭自动调度需要手工配对全部 flag（平台只验证过单 tile 单核管线），超出当前验证范围。

## 8. 验证方案

### 8.1 Golden 函数

> 迁移任务：golden 以 **§0.1 源算子语义**为唯一依据，**直接调用**源仓参考实现
> `engram_decode_step_torch`（脚手架修正后的版本，恒返回 `max_conv_len` 行）。
> **不复刻 §0.6 的 NPU 算法**：不复刻因子提出（A2/A3）、不复刻两遍结构、不复刻 reducer、
> 不做 `inv_d` 常量折叠。golden 在 host 侧用 fp32 torch 算子逐步实现源语义。

```python
import torch
import torch.nn.functional as F

from tileops.workloads.sequence_modeling import engram_decode_step_torch   # 源仓参考（唯一依据）


def golden_engram_decode(e_t, h_t, conv_state, W_K, W_V, rms_w_h, rms_w_v, conv_w,
                         max_conv_len, dilation, eps=1e-6):
    """整算子 golden：y_t 与**完整**的 new_conv_state（恒 max_conv_len 行）。

    局部 cache（L < max_conv_len）由参考实现自己左填充（修正后的分支），调用方无需预处理。
    """
    return engram_decode_step_torch(
        e_t, h_t, conv_state, W_K, W_V, rms_w_h, rms_w_v, conv_w,
        max_conv_len, dilation, eps,
    )


def host_kv(e_t, W_K, W_V, d_pad):
    """step kernel 的 kv 输入：与 project 段同语义的 fp32 投影（L0 分段测试用）。

    与 `_engram_project_kernel` 的输出语义一致：fp32、不做 dtype 往返（§0.5 #7），
    末维 0 填充到 d_pad。参考实现内部也用 `e_t.float() @ W_K.float()`，因此两边同源。
    """
    k = e_t.float() @ W_K.float()
    v = e_t.float() @ W_V.float()
    pad = d_pad - k.shape[1]
    if pad:
        k = F.pad(k, (0, pad))
        v = F.pad(v, (0, pad))
    return torch.stack([k, v], dim=0)          # (2, batch, d_pad) float32
```

**L0 分段断言**（step kernel 单独测试时）：

1. `y_t` 对 `golden_engram_decode(...)` 的 `y_ref`：走 §8.2 的容差。
2. `new_conv_state[:, mcl-1, :]` 对 `ref_state[:, mcl-1, :]`：同容差。
3. **末行以外必须未被触碰**：启动前把 `new_conv_state` 的 `[:, 0:mcl-1, :]` 预填一个可辨识的
   图案（如 `0.01·(slot+1)·(row+1)` 的广播），启动后 `torch.equal` 断言**逐位不变**——
   这证明 step 段既不写 history 也不做平移（history 由 project 段负责，见
   `examples/engram_decode/_engram_project_kernel/DESIGN.md` §8.1 的第 2 条断言）。
4. **partial cache**：`conv_state` 只有 `L` 行时，golden 自己左填充；`new_conv_state` 仍是
   `max_conv_len` 行（回归源仓测试 `test_engram_decode_partial_cache` 的口径）。

### 8.2 精度标准

| 对比项 | dtype | atol | rtol | 依据 |
|--------|-------|------|------|------|
| `y_t` vs golden（L0 分段） | float16 | 0.05 | 0.05 | `examples/TileOPs/tests/ops/test_engram.py` 的 `_tol_decode` |
| `y_t` vs golden（L0 分段） | bfloat16 | 0.1 | 0.05 | 同上 |
| `new_conv_state[:, mcl-1, :]` vs golden | float16 / bfloat16 | 同 `y_t` | 同 `y_t` | 同上 |
| `new_conv_state[:, :mcl-1, :]` vs 预填哨兵 | 全部 | **0（逐位）** | 0 | 本 kernel 不写这些行（§8.1 第 3 条） |
| 角点（±inf/NaN/subnormal/±0） | 全部 | 位型一致 | — | 语义要求「按 IEEE 传播」（§0.1） |

### 8.3 L0 门槛测试计划

按平台 `kernel_template.py` 的分层约定（`--level L0/L1/L2/boundary/all`），本 kernel 的
L0 门槛（Stage 3 必须先全绿才进 L1）：

| 层 | 用例（shape 来源） | 断言 |
|----|-------------------|------|
| **L0** | `decode-s1`（batch=1, d=256, mcl=9, w=4, δ=3, fp16） | 4 条断言；**tap 索引命中 0**（`mcl = δ(w-1)`）+ `grid=1` |
| **L0** | `serving-b1`（batch=1, d=256, mcl=12, δ=3, fp16） | 4 条断言 |
| **L0** | `serving-b8`（batch=8, d=256, mcl=18, δ=3, **bf16**） | 4 条断言；bf16 路径 + 8 个逻辑块 |
| **L0** | `decode-partial-cache`（batch=2, d=256, L=5, mcl=12, fp16） | 4 条断言；host 左填充路径，`new_conv_state` 12 行全检 |
| **L0** | `one-tap`（fixture：batch=1, d=256, **mcl=1, w=1, δ=1**） | 4 条断言；`tap_ub` 零行、末行索引 0 |
| **L1** | `serving-b4`（batch=4, d=512, mcl=20, δ=5, fp16） | 最大 `tile=512`、`threads=512` |
| **L1** | `batch-17`（fixture：17 行） | `grid=17` 单波、每核 1 行 |
| **L1** | 多步滚动（12 步，cache 从空长到满再旋转） | 每步 **y 和完整 history** 都比（回归 `test_engram_decode_multi_step`） |
| **boundary** | `d=600`（`d_pad=768`, `tile=1024`） | 尾列谓词路径（fragment 尾 lane 置 0、写回守卫） |
| **boundary** | `batch=1` + `d=256`；`batch=17` + `d=512` | `threads` 的两档（256/512） |
| **boundary** | 角点输入（±inf/NaN/subnormal/±0） | 位型一致（§8.2 末行） |
| all | 上述全部 + op 级 `test_engram_decode*`（16 项） | 8.2 的 op 级容差 |

分层套件（L1/L2/Boundary 的完整枚举）由 `tilelang-op-develop` 承接；本节只冻结 L0 门槛。

---

## 9. 风险点与注意事项

### 9.1 已知约束

- **发射项主导、且常数未实测**：§1.6.0 的下界（≈6 µs）全部落在「launch ≈5 µs」
  「向量 op ≈0.5 µs」「`AscendAllReduce` 代价」三个**未实测**常数上（平台的
  `cost_measurements` 为空、pattern-library 常数绑定 910B2C）。因此**本设计不承诺绝对时延**，
  只承诺结构判断（2 pass 优于 15 pass、不拆行、不并 launch）。Stage 4 首跑后必须回填
  「设计估算 vs 实测」偏差行（§11.3）。
- **整算子的余量不在本设计范围内**：整算子 = 2 次 launch ≈ 11 µs，其中 step 段 ≈ 6 µs
  且以固定开销为主。要再降只能把 step 段并进 project 段的 launch（§1.6.0 R3 候选 5，权重流量 ×batch）
  ——属跨设计的函数边界变更，本阶段不做。
- **SIMD 备选未编译**：§1.6.3 (iii)② 的 SIMD 形态只有 API 符号存在性 + 兄弟算子的用法先例，
  没有本地编译/精度证据；若 Stage 4 翻转，必须先补编译与精度再比性能。
- **`T.simd` 参数需逐路径核对**：`vld/vsts` 的分布参数、`vcvt` 的舍入/饱和参数
  （`references/programming.md` 明确这一点）。
- **`d_pad` 非 2 的幂的路径**：`tile = 1024 > d_pad = 768` 已由探针编译通过，但**未在真机做精度**
  （属 Stage 3 的 boundary 用例）。
- **待 Stage 3 复核项**：设备型号与核数（用 §5.5 的 `get_device_properties` 片段）；
  `AscendAllReduce` 在 `L = 256/512` 两档的数值一致性（本设计按树形合并 + 容差处理）。
- **精度余量的观察**：A2 在 alpha-small 角点达 30 ulp（≈3.6e-6 相对），与 op 级容差 5e-2 差
  4 个数量级；若 Stage 3 出现超容差，第一嫌疑是**本 kernel 之外的段**（project 段的 `kv`），
  而不是本段的两遍结构。

### 9.2 常见错误

| # | 错误 | 触发场景 | 影响 | 解决方案 |
|---|------|----------|------|----------|
| E1 | 用 `T.reduce_sum(..., dim=1)` 做行归约 | 想复用源算子的写法 | 已知输出布局缺陷（`TRAP-950-simd-shared-reduce`）：第二行统计量与参考不一致 | 用 `T.alloc_reducer` + `reducer_update` + `finalize_reducer`（§0.6 R2） |
| E2 | 照搬 950DT 参考的 `cast(cast(x, dtype), fp32)` 往返 | 想「对齐参考实现」 | **语义偏差**：写出 dtype 值仅 0.39–76 % 逐位一致、最大绝对偏差 1.56e-02（§1.6.1 B1） | 直接对 fp32 `kv` 归一化（§0.5 #7） |
| E3 | 把 `Σv̂²` 放在 `alpha` 之后算 | 没想到因子提出 | 引入第三次扫描（`v̂` 必须先物化） | 用 A3：`Σv̂² = α²Σv²`，`Σv²` 进 phase 1 |
| E4 | 在 step 段里也做 cache 左移 | 照搬 950DT 参考 | 与 project 段重复写 `new_conv_state` 的前 `mcl-1` 行（幂等但流量 ×2），并破坏 §1.5 的写者归属 | 本 kernel 只写末行（§0.5 #11） |
| E5 | 用 `if w > 1:` 在 `@T.prim_func` 内做 Python 守卫 | `w == 1` 想省掉 `tap_ub` | **eager builder 会把两支都 trace** → `UnboundLocalError: tap_ub`（探针实测，§11.1） | 用零行 `T.alloc_shared((w-1, d_pad), …)`（本工具链合法）+ `for p in range(0)` 空循环 |
| E6 | 写 `T.cast(x, "float32") ** 2` | 想简写平方 | eager builder 报 `unsupported operand type(s) for ** or pow(): 'Cast' and 'int'`（探针实测） | 写成 `x * x` |
| E7 | `T.Kernel(batch, threads=128)` | 想沿用 GPU 的 `threads` | 平台拒绝 `T.Kernel` 的 `threads` 参数 | 线程域用 `T.SimtVF(threads=)`（§0.6 R5） |
| E8 | `d_pad` 非 2 的幂时忘记 fragment 尾 lane 置 0 | `d=600` → `tile=1024` | 归约被 UB 垃圾污染（填充列非 0） | `T.if_then_else(i < d_pad, …, 0)`（§6.4）；写回加守卫 |
| E9 | 归一化用 `d_pad` 或 `tile` 做分母 | 误以为「归约多宽就除多宽」 | 结果整体偏小（`d_pad/d` 倍） | 分母恒为模型维 `d`（§0.1） |
| E10 | 在 VF 内改 `tap_ub`/`y_ub` 的写目的偏移 | 想省一次中间 buffer | UB→UB 非零偏移区域拷贝在本平台有已验证反例（`examples/moe_permute_align/_make_align_kernel/repro/TRAP-950-shared-shift-copy.py`） | 所有 UB 内写的目的偏移恒为 0 |
| E11 | 沿用 harness 的 `step_threads=128` | 直接照抄 `default_config` | 每 lane 4 元素（能跑、精度更宽松地合格），但不是默认配置 | 按 §5.6 取 `threads = min(512, tile)`；`{128,256,512}` 留作 Stage 4 A/B |

### 9.3 特殊场景处理

- **`w == 1`（fixture 第 4 行）**：`tap_ub` 零行、tap 循环空、`conv_out = conv_w[0]·vhat_norm`；
  探针第七变体编译通过（5 156 字节源码，无 tap 相关指令）。
- **`max_conv_len == 1`**：末行索引 0，`new_conv_state` 只有本步写入的那一行。
- **`max_conv_len == δ(w-1)`**（`decode-s1`）：最老 tap 的索引恰为 0 —— 边界覆盖点。
- **partial cache（`L < mcl`）**：host 左填充；kernel 无感知；golden 由参考实现自己 pad。
- **`batch == 1`**：`grid = 1`（1 个 Vector 核）。这是刻意选择（§5.5 ③），不是缺陷。
- **bf16**：`dtype_str = "bfloat16"`；`T.cast` 与 `expf`/`rsqrtf` 的接口不变；探针第三变体
  （b8 + bf16）编译通过。
- **混合精度边界**：`kv` 恒 fp32（契约），其余输入输出恒输入 dtype；中间量全 fp32。
  **不做**额外的舍入（E2）。
- **极端值**：NaN/±inf 在 fp32 计算里按 IEEE 传播（与源一致，§8.2 末行）；`±0` 的位型在
  「未修改的行」上与输入一致（本 kernel 不碰那些行）。
- **`d_padded > d`**：填充列恒 0，因此归约、卷积、SiLU 在填充列上给出的都是 `0`（无 NaN/无 inf），
  对外 crop 掉。

## 10. 交付清单

### 10.1 目录结构

```
examples/engram_decode/_engram_step_kernel/
├── DESIGN.md                                  # 本设计文档
├── verify_equiv.py                            # D-1 等价性机器验证（已执行，ALL EQUIV_PASS）
├── RETROSPECTIVE.md                           # Stage 1 复盘
└── history_version/
    └── design_probe_step.py                   # 探针：7 个形状/结构变体，全部 compile-only
```

Stage 3 产出的 `_engram_step_kernel` 实现进入 TileOPs 抽取目录
`examples/TileOPs/tileops/kernels/sequence_modeling/engram_decode/`
（与 `_engram_project_kernel` 同一文件），本阶段不产出 `{op}.py`。

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | ✅ 已完成 | 本设计文档 |
| `verify_equiv.py` | ✅ 已完成并执行 | D-1 机器验证（§1.6.1 结果表；`RESULT: ALL EQUIV_PASS`） |
| `RETROSPECTIVE.md` | ✅ 已完成 | Stage 1 复盘（追加式） |
| `history_version/design_probe_step.py` | ✅ 已完成并执行 | 设计期编译探针（7/7 编译通过，§11.1） |
| `{op}.py`（kernel 实现） | ⬜ 待 Stage 3 | 按 §3.3 伪代码 + §5.6 配置 |

### 10.3 命名规范

- kernel 函数名：`engram_decode_step`（与 project 段的 `engram_decode_project` 不重名——
  两个 launch 的 profile 记录按名字区分，参见 950DT 参考的同款注释）。
- 工厂：`_build_step(...)` + `functools.lru_cache`。
- 缓冲：`*_ub` = `alloc_shared`（UB）；`*f` = `alloc_fragment`；`acc_*` = reducer handle。

### 10.4 实现顺序

1. 骨架：`T.Kernel(batch)` + 9 条 `T.copy` 预取 + 两条写回（先跑通「拷贝 + 恒等」）。
2. 行归约：4 个 reducer epoch + phase 1 单遍（对 `torch.sum` 做 L0 分段校验）。
3. 标量链 + phase 2（对 golden 做 L0 全量校验，`decode-s1` 先绿）。
4. 边界：`w=1/mcl=1`、`d=600` 尾列谓词、`batch=17`、partial cache。
5. 交 Stage 4 做 `threads` A/B 与 §1.6.3 (iii) 的 SIMT/SIMD 裁决。

## 11. 设计期探针与估算复算

### 11.1 编译探针结果

脚本：`examples/engram_decode/_engram_step_kernel/history_version/design_probe_step.py`
（compile-only，`ASCEND_RT_VISIBLE_DEVICES=""`，未占用设备；每组记录生成的源码标记与长度）：

| 变体 | 形状 / 结构 | 结果 | 关键观察 |
|------|------------|------|---------|
| S1 serving-b4 | b=4, d=d_pad=512, mcl=20, w=4, δ=5, fp16, threads=512 | **COMPILE OK**（6 000 字节） | `__vector__` 单核函数；9 条 `asc_copy_gm2ub_align` 批量在 VF 前；4 条 `AscendAllReduce<tl::SumOp, 512, 1, 0>`（scratch 偏移 2816/3328/3840/4352 float）；UB 最大偏移 24 576 B |
| S1 decode-s1 | b=1, d_pad=256, mcl=9, threads=256 | **COMPILE OK** | `grid=1`；tap 索引命中 0（`mcl=δ(w-1)`） |
| S1 serving-b8 | b=8, d_pad=256, bf16, threads=256 | **COMPILE OK** | bf16 路径 |
| S1 masked tail | b=2, d=600, d_pad=768（`tile=1024`）, threads=512 | **COMPILE OK**（6 777 字节） | 尾列谓词路径；fragment 尾 lane 置 0 |
| S1 threads=128 | b=4, d_pad=512, threads=128 | **COMPILE OK**（6 384 字节） | 每 lane 4 元素：`#pragma unroll for(i<4)` + 4 次串行累加；`AscendAllReduce<…, 128, …>` |
| S1 w=1 / mcl=1 | b=1, d_pad=256, mcl=1, w=1, δ=1 | **COMPILE OK**（5 156 字节） | 零行 `tap_ub` 合法；1 条 `notify(MTE2,V)` + `wait`（不再嵌在 p 循环里） |
| S2 保守结构 | b=4, d_pad=512；逐统计量 pass（950DT 参考形态） | **COMPILE OK**（5 785 字节） | Stage 3 的降落伞（§1.6.0 R4 候选 2） |

**过程中的两个 API 事实（已写入 §9.2）**：(a) `@T.prim_func` 内的 Python `if w > 1:` 会把两支都
trace（`ctx_if`），导致 `UnboundLocalError`，因此结构守卫必须放在 prim_func 之外或用零行分配；
(b) `T.cast(x, "float32") ** 2` 不被 eager builder 支持（`pow()` 只接受 PrimExpr 底数）。

### 11.2 roofline 复算

`serving-b4`（最大流量 workload）逐项复算：

| 项 | 记账 | Bytes |
|----|------|-------|
| `h_t` 行 | `batch·d_pad·2` | 4 096 |
| `kv` 两片（fp32） | `2·batch·d_pad·4` | 16 384 |
| tap 行（`w-1=3`） | `batch·(w-1)·d_pad·2` | 12 288 |
| 权重（`rms_w_h/w_v` + `conv_w`，HBM 一遍） | `(2+w)·d_pad·2` | 6 144 |
| `y_t` + `new_conv_state` 末行 | `2·batch·d_pad·2` | 8 192 |
| **合计** | | **47 104 B ≈ 46 KiB** |
| 流量项（1.0–1.6 TB/s，未实测假设） | | **0.029–0.047 µs** |
| 发射项 | 1 launch（≈5 µs）+ 2 pass × 0.5 µs | **≈ 6 µs** |
| 容量项 | UB 24 KiB / 248 KiB | 满足 |
| **估算下界** | `max(流量, 发射, 容量)` | **≈ 6 µs** |

与 manifest 的步段口径 `batch·(16d + w·2d) = 4·(16·512 + 8·512) = 49 152 B` 一致（差额来自
权重按 HBM 一遍计 6 144 B 而 manifest 未计权重）。

### 11.3 Stage 4 回填计划

1. baseline 采集后，在 `perf_opt/opt_log.md` 的 Baseline 段追加「设计估算（≈6 µs）vs 实测」偏差行；
   偏差 > 2× 时按 §1.6.0 的三项分解（流量/发射/容量）归因，重点核对 launch 固定开销与
   `AscendAllReduce` 代价这两个**设计期未实测**的量。
2. `threads ∈ {128, 256, 512}` 的 A/B（§1.6.3 (ii)）与 SIMT/SIMD 的 A/B（§1.6.3 (iii)）按
   已冻结的阈值与代表 shape 执行；若 SIMD 翻转，先补齐它的编译与精度证据。
3. 若实测发现新的硬件常数（如单次 `AscendAllReduce` 的时延），按
   `.agents/skills/_shared/standards/hardware-cost-model.md` §3 的流程回写平台知识库的 constants。
