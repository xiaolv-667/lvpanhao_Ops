# MLA prefill 反向主 pass（_bwd_kernel）算子设计文档

> 迁移任务（harness 子模式）。源算子：`/workspace/TileOPs/src/tileops/_kernels/attention/mla/bwd.py` 中的
> `_bwd_kernel`（GPU TileLang 实现，`MLADenseBwdKernel` / `MLAVarlenBwdKernel` 共用工厂）。
> 目标平台：`ascend-950pr-9579`（dav-3510，CANN 9.2.1，TileLang ascend 0.1.15），编程模式按平台
> `programming_modes.migration_default = simd` 评估后选 **混合（Cube + Vector）**，偏离理由见 §2。
> 本函数是 `MLADenseBwdOp` 三次 launch 的第二个（`_delta_kernel` → **`_bwd_kernel`** → `_dq_land_kernel`）；
> 跨函数接口契约为：`delta` fp32 `[B,H,S_q]`（`_delta_kernel` 产出）、`dq_acc` fp32 `[T_q,H,D_qk]`（本函数
> 原地累加，`_dq_land_kernel` 统一乘 `scale` 并 cast）。
>
> **本版为 revision v1**（检视不通过后的重做）。相对 v0 的两处阻塞级修正——GEMM 全 extent 静态化、删除
> 显式 `T.Vector`/`sid`——的逐项闭环、编译探针证据与"为何不会再犯"见 §11。

---

## 0. 源算子解读与迁移分析

> 方法论：tilelang-op-design skill 的 references/migration-analysis.md（三问解读 → 耦合性四态 → 重设计）。
> 本章结论驱动 §1–§10。

### 0.1 源算子语义（做什么）

对每个 `(request b, head h, key 位置 n)` 三元组，给定前向的 `q, k, v, do` 与每 query 行的自然对数
log-sum-exp `lse`、`delta = rowsum(o * do)`，计算注意力的三个梯度：

$$
P[m,n] = \exp\left(\mathrm{scale}\cdot q_m\!\cdot\! k_n - \mathrm{lse}_m\right),\qquad
dP[m,n] = do_m\!\cdot\! v_n,\qquad
dS[m,n] = P[m,n]\cdot\left(dP[m,n] - \mathrm{delta}_m\right)
$$

$$
dV[n,:] = \sum_m P[m,n]\, do[m,:],\qquad
dK[n,:] = \mathrm{scale}\cdot\sum_m dS[m,n]\, q[m,:],\qquad
dQ[m,:] = \mathrm{scale}\cdot\sum_n dS[m,n]\, k[n,:]
$$

其中 `q_m` 是第 `m` 个 query 行、`k_n` 是 key 位置 `n` 的行，二者宽度均为 `D_qk = 192`；`v_n, do_m` 宽度
`D_v = 128`。**对齐方式为 FA3 的 bottom-right**：query `i` 落在 key 位置 `i + (S_kv - S_q)`（`shift`），
因果可见条件为 `n <= m + shift`（含对角）；`is_causal=False` 时无 mask。

**语义要素逐一确认**（源码逐行核对，无「待确认」项）：

| 要素 | 结论 | 来源 |
|------|------|------|
| 数学语义 | 上式；`P` 由 `lse` 重算（不复用前向概率），`delta` 由 `o*do` 行和给出 | `bwd.py:229-260`；`workloads/attention/mla.py:441-460`（golden） |
| I/O 契约 | `q,k [T,H,D_qk]`、`v,do [T,H,D_v]`、`lse,delta [B,H,S_q]` fp32、`cu_seqlens_* [B+1]` int32、`dq_acc [T,H,D_qk]` fp32 in-out、`dk [T_kv,H,D_qk]`、`dv [T_kv,H,D_v]` 与输入同 dtype | `bwd.py:141-154` |
| layout | 全部 packed BSHD（`(token, head, dim)`），尾轴连续；dense 调用在 host 侧 `view(B*S, H, D)` | `bwd.py:363-379` |
| 规约语义 | `dV/dK` 沿 query 块按 `m` 升序 fp32 累加（`dv_f`/`dk_f` fragment，`T.clear` 后逐块 `gemm` 累加）；`dQ` 沿 key 块累加，用 `T.atomic_add` 写 fp32 `dq_acc`，**跨核顺序不确定**（浮点加法不可交换，故 `dq_acc` 无逐位可复现性契约）；`P/dS` 的 K 维归约在 tensor core 内完成 | `bwd.py:200-254` |
| dtype 语义 | GEMM 输入为 fp16/bf16，**累加器全 fp32**（`st/dpt/dk_f/dv_f/dq_f` 皆 fp32）；`P/dS` 在做第二个 GEMM 前 cast 到输入 dtype；`dk` 写回时乘 `scale` 再 cast，`dv` 直接 cast；`dq` 只写 fp32（缩放交给 `_dq_land_kernel`） | `bwd.py:164-170, 244-260` |
| 边界语义 | (a) `len_q/len_kv` 之外的 query/key 装载按 0 填充；(b) 因果不可见位置 `P=0`（从而 `dS=0`），实现为 `T.if_then_else` 选择而非乘掩码；(c) `lse = -inf` 的「看不到任何 key」行：因 mask 全假而 `P=0`，梯度为 0，**不得产生 NaN**；(d) 空块：bottom-right 因果下每个 key 块至少可见对角 query 块 | `bwd.py:207-243`；golden 注释 `mla.py:400-421` |
| host 侧语义 | `theta`：`delta` 由独立 launch 产出、`dq` 的缩放 cast 由第三个 launch 完成——三者同属一次算子调用，故本函数只负责 `dq_acc` 的 fp32 累加，**不得**在 kernel 内乘 `scale` 后再写 `dq_acc` | `bwd.py:338-347` |

**小结（第一问 What）**：语义 = 「用给定 `lse`/`delta` 重算 `P`、`dP`、`dS`，产出 `dV`、`dK`（含 `scale`）与 `dQ` 的 fp32 累加」；累加顺序只在 `dK/dV` 上固定为 query 块升序，`dQ` 跨核顺序不确定；中间累加全 fp32，GEMM 输入为输入 dtype；`lse=-inf` 的空行为零梯度且不得产生 NaN；`scale` 只在 `dK` 写回与 `dQ` 的后继 launch 中生效。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `q` | `(T_q, H, 192)` | f16/bf16 | query；`T_q = B*S_q`（dense） |
| `k` | `(T_kv, H, 192)` | f16/bf16 | key（latent 已解压，每 head 私有 192 宽） |
| `v` | `(T_kv, H, 128)` | f16/bf16 | value |
| `do` | `(T_q, H, 128)` | f16/bf16 | 输出梯度 |
| `o` | `(T_q, H, 128)` | f16/bf16 | 前向输出；**仅由同一次算子调用的 `_delta_kernel` launch 消费**（`rowsum(o*do)`，`bwd.py:341`），本 kernel 不读；按 `MLADenseBwdOp.inputs` 与 `_run` 签名（`bwd.py:338`）保留在同一算子调用的输入契约内 |
| `lse` | `(B, H, S_q)` | fp32 | **自然对数** log-sum-exp |
| `delta` | `(B, H, S_q)` | fp32 | `rowsum(o*do)` |
| `cu_seqlens_q/kv` | `(B+1,)` | int32 | dense 下为 `arange(0,(B+1)*S,S)` |
| `dq_acc` | `(T_q, H, 192)` | fp32 | **in-out**，累加缓冲 |
| `dk` | `(T_kv, H, 192)` | f16/bf16 | 输出，与 `k` 同 shape |
| `dv` | `(T_kv, H, 128)` | f16/bf16 | 输出，与 `v` 同 shape |

输出 shape：`dk/dv` 与 `k/v` 完全一致；`dq_acc` 原地累加，shape 不变（后续 launch 缩放 cast 成 `dq`）。

### 0.3 实现算法解读（怎么算）

**计算步骤分解**（`bwd.py:176-260` 每个计算语句均归入下列步骤）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式 |
|------|------|------|------|-------------|
| S1 | 块解码：`request = tile // ceil(S_kv/block_n)`、`n0 = (tile % …)*block_n`（packed 走 `GroupTiling.decode`） | `tile`, `cu_seqlens_kv` | `b, n0, len_q, len_kv, shift` | — （任务映射） |
| S2 | 装载 key/value 块（越界补 0） | `k[kv_start+n0 …]`, `v[…]` | `k_s (block_n,192)`, `v_s (block_n,128)` SMEM | `k_n, v_n` |
| S3 | 清零 `dk_f, dv_f` | — | fp32 fragment | 累加器初值 |
| S4 | 计算 query 块循环区间 `loop_st=max(0,(n0-shift)//block_m)`、`loop_ed=ceil(len_q/block_m)` | `n0, shift` | 循环界 | 因果可见性剪枝 |
| S5 | 逐 query 块装载 `q_s, do_s`（越界补 0）与 `lse_s, delta_s` | `q, do, lse, delta` | SMEM | `q_m, do_m, lse_m, delta_m` |
| S6 | `st = gemm(k_s, q_s^T)` | `k_s, q_s` | `st (block_n,block_m)` fp32 | `s = scale*q·k`（未乘 scale） |
| S7 | `dpt = gemm(v_s, do_s^T)` | `v_s, do_s` | `dpt (block_n,block_m)` fp32 | `dP` |
| S8 | `st = if(mask) exp2(st*scale*LOG2E - lse*LOG2E) else 0` | `st, lse_s` | `st`（就地） | `P` + 因果 mask |
| S9 | `p16 = cast(st)` | `st` | `p16 (block_n,block_m)` f16 | `P` 的 GEMM 输入 |
| S10 | `dv_f += gemm(p16, do_s)` | `p16, do_s` | `dv_f (block_n,128)` fp32 | `dV = P^T dO` |
| S11 | `ds16 = st * (dpt - delta_s)` | `st, dpt, delta_s` | `ds16` f16 | `dS = P∘(dP-delta)` |
| S12 | `dk_f += gemm(ds16, q_s)` | `ds16, q_s` | `dk_f (block_n,192)` fp32 | `dK`（未乘 scale） |
| S13 | `ds_s = cast(ds16)`（SMEM 中转） | `ds16` | `ds_s (block_n,block_m)` | `dS` 的第二个 GEMM 输入 |
| S14 | `dq_f = gemm(ds_s^T, k_s)`（`transpose_A=True`） | `ds_s, k_s` | `dq_f (block_m,192)` fp32 | `dQ`（未乘 scale） |
| S15 | `atomic_add(dq_acc[q_start+m0+i, h, :], dq_f[i, :])`（仅有效行） | `dq_f` | `dq_acc` GM | `dQ` 跨 key 块累加 |
| S16 | 循环结束：`dk = cast(dk_f*scale)`、`dv = cast(dv_f)`（仅有效 key 行） | `dk_f, dv_f` | `dk, dv` | `dk` 的 scale 与输出 dtype |

**数据流与内存访问模式**（源 GPU 视角）：

```
GM[q,k,v,do,lse,delta] --T.copy--> SMEM[k_s,v_s,q_s,do_s] --ldmatrix--> fragment
   fragment[st]  --T.gemm(TensorCore)--> fragment[st] --exp2/mask--> fragment[p16]
   fragment[dv_f] <--T.gemm(p16,do_s)--
   fragment[ds16] = st*(dpt-delta) --cast--> SMEM[ds_s] --T.gemm(transpose_A)--> fragment[dq_f]
   fragment[dq_f] --T.atomic_add--> GM[dq_acc]     （每 query 块一次全局原子累加）
   循环后 fragment[dk_f/dv_f] --cast(*scale)--> GM[dk/dv]
```

**循环与并行结构**：grid = `num_tiles × heads`（GPU 为二维 block 网格；`num_tiles = B*ceil(S_kv/block_n)`，
packed 用 `GroupTiling.tile_upper_bound`）；每 block 内按 query 块 `T.Pipelined(loop_st, loop_ed,
num_stages=2)` 串行迭代（流水深度 2），块内 `threads=256` 由 `T.gemm` 的 warp 划分
（`T.GemmWarpPolicy.FullRow`）与 `T.Parallel` 逐元素循环共同消费；默认配置
`block_n=128, block_m=32, num_stages=2, threads=256`（`bwd.py:336`）。寄存器驻留：`dk_f`/`dv_f` 跨整个
query 循环常驻（key-stationary），`dq_f` 每次迭代清零后原子写回。

**host 侧逻辑**：`MLADenseBwdKernel.forward` 把 BSHD 张量 `view(B*S,H,D)`、构造 `cu_seqlens=arange`
（纯元数据，无数据搬运）；`_run` 依序 launch `delta → bwd → dq_land`；`dq_acc` 由 host `torch.zeros`
分配（fp32）。属于算子调用的 host 预处理/后处理仅此三项，均不在本 kernel 内。

**小结（第二问 How）**：算法 = 「一个 key 块常驻 → 枚举可见 query 块 → 每块 5 次矩阵乘 + 1 次 exp/掩码 + 1 次逐元素乘 + 1 次转置 GEMM」；`P/dS` 从不物化到 GM；`dK/dV` 由块内累加器承担，`dQ` 由全局原子加承担。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | key-stationary 分块（S2/S3/S16） | `k/v` 只搬一次、`dK/dV` 只在块尾写一次 | 一块 key 常驻 SMEM，`dk_f/dv_f` 跨 query 循环寄存器累加 | SMEM + 大寄存器堆 | 硬件强相关（结构可移植，容量依赖） |
| 2 | recompute 而非物化 `P/dS`（S6/S8/S11） | 消 O(S²) 中间缓冲与 GM 往返（flash 家族核心） | 每个 query 块重算 `s` 与 `dP`，`P/dS` 只活在寄存器/SMEM | 无（纯算法层） | 可移植（算法层，保留） |
| 3 | 因果块级剪枝（S4） | 少算约一半 (key 块, query 块) 对 | `loop_st` 从可见性下界起算 | 无（纯算法层） | 可移植（保留） |
| 4 | tensor core GEMM（S6/S7/S10/S12/S14） | 矩阵算力 | `T.gemm` → mma | tensor core | 硬件强相关（等价替换为 Cube） |
| 5 | 寄存器 fragment 作累加器（S3/S10/S12） | 免 SMEM 往返 | `alloc_fragment` + fp32 累加 | 大寄存器堆 | 硬件强相关（重设计为 L0C） |
| 6 | `transpose_A=True` 复用同一 `ds_s`（S14） | 省一份 `dS` 的 SMEM 副本 | mma 直接吃转置操作数 | mma 的 A 转置控制 | 硬件强相关（重设计，见 §0.6 R3） |
| 7 | `exp2` + `scale*LOG2E` 折入指数（S8） | 用便宜底 2 指数指令 | `exp(x)=exp2(x*log2e)`，`lse*LOG2E` 逐元素重算 | SFU 上 `exp2` 快于 `exp` | 硬件弱相关（等价替换，见 §1.6.1） |
| 8 | mask 用 `if_then_else` 选择而非乘掩码（S8） | mask 位置严格为 0，`lse=-inf` 行不产生 NaN | select | 无 | 可移植（保留，转成向量 select） |
| 9 | `T.Pipelined(num_stages=2)`（S5 装载 / 计算重叠） | 消 MTE 停顿 | 双 buffer 轮转 | 异步拷贝引擎 | 模式可移植（等价替换为版本标注） |
| 10 | `T.atomic_add` 写 `dq_acc`（S15） | 免 workspace / 免跨 block 归约 | 全局 fp32 原子加 | 全局原子 ALU + L2 | 硬件强相关（等价替换为 store-atomic） |
| 11 | `lse/delta` 行广播（S8/S11） | 每行一次标量代替逐元素重算 | SMEM 标量读 | — | 可移植（保留） |
| 12 | warp 划分 `FullRow` + `threads=256`（S6–S14） | 匹配 mma 的 warp 分形 | warp-level 分片 | warp 32 线程 + 寄存器分片 | 硬件强相关（重新设计） |
| 13 | **（NPU 适配构造，非源码手段）** `clear_accum` | 表达「首块清零 / 跨块累加」 | 由 mma 的零累加位承担 | — | — |
| 14 | packed/dense 双分支 `GroupTiling`（S1） | 一个工厂服务两种 layout | 编译期分支 | 无 | 可移植（本函数只做 dense，见 §0.5） |

**关于 #13 的口径更正**：源码**没有** `clear_accum` 参数，只用 `T.clear(st)`（`bwd.py:225`）、
`T.clear(dpt)`（`bwd.py:227`）、`T.clear(dq_f)`（`bwd.py:250`）以及 `T.clear(dk_f/dv_f)`（`bwd.py:200-201`）
显式清零；`T.gemm` 本身的累加器语义由 fragment 初值决定。`clear_accum` 是**本设计的 NPU 侧构造**
（§3.2 步骤 4），用于让 L0C 累加器表达同一语义，并新增了跨 query 循环的 `clear_accum=(kq == loop_st)`
（动态布尔，编译级已验证，见 §3.5.4 探针 B 与 §9.1 风险 #7）。

**小结（第三问 Why fast）**：共识别 14 项，其中 5 项是纯算法层（recompute、因果剪枝、掩码语义、清零语义、
行统计），8 项依赖源硬件（SMEM/寄存器/tensor core/warp/cp.async/原子 ALU/二维 grid），1 项（#13）是
迁移侧的语义表达构造；依赖硬件者是 §0.5 的逐项处置对象，未识别的优化会在迁移中被静默丢弃，故此处以
「源码每类显式手段至少一行」为准。

### 0.5 硬件耦合性分析与 NPU 适配决策

判定问题：实现算法与优化手段是否硬件强相关？能否用在 NPU 上？依据一律指向本平台已核验资料
（`.agents/platforms/toolchains/ascend/references/programming.md` 与 knowledge/repro 下的可执行条目）。

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 方案 / 依据 |
|------|------|-----------|-----------------|------|----------------|
| 数学语义 / I/O 契约 / 边界语义 | 语义 | 无 | — | **保留** | 语义层无条件保留；golden 依 §0.1 独立实现 |
| 二维 block 网格 `T.Kernel(tiles, heads)`（S1 的并行划分） | 优化 | CUDA blockIdx 二维 | 无（平台只支持一维核网格） | **重新设计** | → §0.6 R1：一维网格 + 块内 div/mod 解码 |
| key-stationary 分块 + `dK/dV` 跨块累加（#1） | 算法 | 无（结构） | 有 | **保留** | 同一 key-stationary 结构；`dk/dv` 累加器从寄存器迁到 L0C（§0.6 R2） |
| recompute 不物化 `P/dS`（#2） | 算法 | 无 | 有 | **保留** | flash 类单遍结构，`P/dS` 只活 UB（§1.6.0 R2 结论） |
| 因果块级剪枝（#3） | 算法 | 无 | 有 | **保留** | 同一 `loop_st/loop_ed` 语义，`T.serial`/`T.Pipelined` 动态界（编译级已验证，§3.5.4） |
| tensor core GEMM（#4） | 优化 | mma / fragment | 有（Cube MAD） | **等价替换** | `T.gemm` + L1 操作数 + L0C 累加器；**L1 路径本轮已编译级走通**（§3.5.4 探针 A：5 个 `ascend_gemm_l1`） |
| 寄存器 fragment 累加器（#5） | 优化 | 每线程 255 寄存器 + 寄存器分片 | 部分（Cube 侧只有 L0C 可累加） | **重新设计** | → §0.6 R2：L0C 累加器 + 整 tile `T.copy` 落地 UB 做向量 epilogue |
| `transpose_A=True` 复用一个 `ds` 副本（#6） | 优化 | mma 的 A 转置控制 | L0 路径有（MN-major A），**L1 路径显式禁止** | **重新设计** | → §0.6 R3：NT-only 重构 + UB 内转置写第二份 `ds`；备选为 L0 路径 `trans_A=True`（§1.6.3、§4.6） |
| `exp2` + `LOG2E` 折入（#7） | 优化 | SFU 指令选择 | 有（`T.exp` 已核验；`exp2` 未核验） | **等价替换** | 改用自然对数域 `T.exp(s*scale - lse)`（§1.6.1 A1，机器验证 1 ulp） |
| `if_then_else` 掩码（#8） | 优化 | 无 | 有 | **等价替换** | `T.SimtVF` 内 `T.if_then_else` + 运行时块级 mask 特化（§1.6.1 A2、§3.5.4 探针 C） |
| `T.Pipelined(num_stages=2)`（#9） | 优化 | cp.async / 双 SMEM 槽 | 有（`annotate_buffer_versions` + `num_stages`） | **等价替换** | 首版 `num_stages=1`（未认证复杂流水），版本标注留给 Stage 4；依据 programming.md「同步与缓冲版本」 |
| `T.atomic_add` 逐元素（#10） | 优化 | 全局原子 ALU | 有（store-mode 原子） | **等价替换** | `T.set_atomic("add","float32")` + L0C→GM `T.copy`；依据 knowledge/repro/PL-950-atomic-splitk.py（同 dtype fp32 GM 累加已核验），本轮编译级再次确认（§3.5.4 探针 A：`asc_copy_l0c2gm` 落在 `asc_set_atomic_add_float` / `asc_set_atomic_none` 之间） |
| `lse/delta` 行广播（#11） | 优化 | SMEM 标量读 | 有（UB 内 lane 索引读） | **等价替换** | 向量 lane 落在 query 轴，`lse_ub[j]` 为 lane 私有读（§1.6.3）；列广播另见 knowledge/repro/PL-950-conversion-layout.py |
| warp 划分 + `threads=256`（#12） | 优化 | warp 32 线程 / 线程级分片 | 无（NPU 无 warp） | **重新设计** | → §0.6 R4：`T.SimtVF(threads=128)` 整 tile lane 映射（**不写 `T.Vector`**；副块 id 由自动调度合成） |
| `FullRow` policy 参数 | 优化 | warp 划分启发式 | 无 | **舍弃** | Cube 侧由 mad 指令形状与 L1 分形决定；`policy` 在 L0/L1 路径不入 lowering（`gemm_op.py` 的 L0 装配不读 policy 语义，仅透传），无等价可调项；不影响正确性 |
| `T.clear` 显式清零（#13 的源码侧形态） | 优化 | mma 累加器初值 | 有 | **等价替换** | 以 `clear_accum` 表达同一语义（§3.2 步骤 4）；跨 query 循环用 `clear_accum=(kq == loop_st)`（编译级已验证，探针 B） |
| packed / `GroupTiling` 分支（#14） | 优化 | 无 | 有（但属另一个 op） | **舍弃（本函数）** | 本设计只覆盖 dense（`MLADenseBwdOp`）。差异见 §0.7；packed `MLAVarlenBwdOp` 的变长块解码与 `T_q` 动态性单独设计，不在本函数内引入分支 |
| host `view`/`arange` 元数据 | 语义 | 无 | 有 | **保留** | 仅 shape/stride 视图与常量构造，无数据搬运（符合 host 约束） |

统计（18 行判定）：保留 5 项（数学语义、key-stationary、recompute、因果剪枝、host 元数据）、
等价替换 7 项、重新设计 4 项（R1–R4）、舍弃 2 项（`FullRow` policy、packed 分支）。

### 0.6 NPU 算法重设计

**重设计项 R1：二维 block 网格 → 一维核网格 + 块内解码**

- **源方案**：`T.Kernel(num_tiles, heads)` 二维 grid，硬件直接给出 `(tile, h)`。
- **NPU 新算法**：`T.Kernel(grid)` 一维核网格（`kernel.py:Kernel` 明确拒绝多维网格），
  `grid = heads * B * ceil(S_kv/block_n)`，块内 `h = bid // num_tiles_per_head`、
  `tile = bid % num_tiles_per_head` 再解出 `b, n0`。head 取外层（同一 head 的相邻 key 块在相邻 bid）
  以让同 head 的 `q/do/dq_acc` 切片在 L2 内复用（§1.6.0 R4）。
- **语义保持论证**：`(tile, h)` 与 `bid` 是双射，任务集合完全一致；无算术改变。
  边界：`grid` 用静态整数（`S_kv/H/B` 是编译期常量），空块不存在（每个 key 块至少一次迭代）。

**重设计项 R2：寄存器 fragment 累加器 → L0C 累加器 + 整 tile UB 中转**

- **源方案**：`st/dpt/dk_f/dv_f/dq_f` 全部 `T.alloc_fragment`（寄存器）；`dk_f/dv_f` 跨 query 循环常驻；
  `T.gemm` 的 C 直接在寄存器文件里累加，向量化 epilogue 直接读写同一 fragment。
- **NPU 新算法**：Cube 的累加器只能是 L0C（`gemm_mad.py:GemmMAD._prepare_lower` 断言 C 必须带
  Ascend fractal layout；`infer_layout` 只为 L0/L1 的 A、B 建立分形布局，UB 缓冲拿不到该注解，
  故 C 必须是 `alloc_l0c` 分配）。故：
  - `dk_l0c (64,192) fp32 = 48KB`、`dv_l0c (64,128) fp32 = 32KB` 跨 query 循环累加（key-stationary 保持）；
  - `st_l0c (64,64) fp32 = 16KB`、`dpt_l0c (64,64) fp32 = 16KB`、`dq_l0c (64,192) fp32 = 48KB` 每次迭代清零；
  - 共 160KB / 256KB（62.5%），见 §4.6；
  - `P/dS` 的向量 epilogue 在 UB 内做：**整 tile** `T.copy(st_c, st_u)`（(64,64) fp32，同 dtype）把 L0C
    落到 UB，`T.SimtVF(threads=128)` 内 `T.Parallel(64,64)` 计算，再整 tile `T.copy` 送回 L1 的
    `pT_s/dsT_s/ds_s`。**不使用 `T.Vector`/`sid`**：副块（AIV）编号由自动调度在 AIC/AIV 分段时自行合成
    （生成的设备码里 `int32_t sid = asc_get_sub_block_id()`，用户不可见），两个 AIV 冗余执行同一整 tile
    向量段——正确性不受影响（逐元素、无跨 lane 归约），向量侧有效吞吐按 1 个 AIV 计（§1.6.3）。
- **语义保持论证**：累加顺序保持「`dV/dK` 沿 `m` 升序、`dQ` 沿 key 块跨核累加」；累加 dtype 仍为 fp32。
  唯一变化是 K 维归约在 L0C 内由 mad 完成（与源 mma 同为 fp32 累加、分组方式不同），与源算子一样只承诺
  fp32 累加精度而非逐位一致（源侧 warp/tile 分组与值有关）。边界：无效 key 行/列的贡献由 §1.6.1 A2 的
  掩码置零（覆写 0 填充引入的非零 `exp(-lse)`），块尾写回时尾部由平台的 OOB clamp 裁剪（§5.4）。

**重设计项 R3：`transpose_A` 的 dQ → NT 形式 + UB 内转置副本**

- **源方案**：`dq_f = gemm(ds_s, k_s, transpose_A=True)`——同一份 `ds_s` 同时服侍 `dk`（`ds^T` 作 A）与
  `dq`（`ds^T` 的转置作 A）。
- **NPU 新算法**：平台 GEMM 的 L1 路径**只支持 NT**（`gemm_mad.py:GemmMAD.lower`：
  `assert not self.trans_A and self.trans_B, "Ascend L1 GEMM currently only supports trans_A=False,
  trans_B=True (NT)."`），L0 路径虽允许 `trans_A=True`（走 MN-major 分形）但本工具链未核验。
  故把 5 个乘法全部改写成 `C = A(M,K) · B(N,K)^T`：
  | 乘积 | A（L1，K-major） | B（L1，K-major） | C（L0C） |
  |---|---|---|---|
  | `S^T` | `k_s (64,192)` | `q_s (64,192)` | `st_c (64,64)` |
  | `dP^T` | `v_s (64,128)` | `do_s (64,128)` | `dpt_c (64,64)` |
  | `dV` | `pT_s (64,64)` | `doT_s (128,64)` | `dv_c (64,128)` |
  | `dK` | `dsT_s (64,64)` | `qT_s (192,64)` | `dk_c (64,192)` |
  | `dQ` | `ds_s (64,64)` | `kT_s (192,64)` | `dq_c (64,192)` |
  其中 `qT/doT/kT` 用 GM→L1 的 `T.copy(..., transpose=True)`（dn2nz 转置映射，已核验形态见
  knowledge/repro/PL-950-gemm-tiles.py 的 `trans_b` 分支）；`ds (Mq,Nk)` 由向量 epilogue **额外写一份**
  转置到 UB 再 UB→L1（SIMT 内转置写，已核验形态见 knowledge/repro/PL-950-conversion-layout.py
  「转置与列广播在 SIMT 中表达」，本轮编译级再次走通，见 §3.5.4 探针 A）。
- **语义保持论证**：`(A B^T)^T = B A^T` 只是把同一个矩阵积写成不同操作数顺序，逐元素恒等；`dS` 的第二份
  副本是同一 fp32 值的同一 fp16 舍入（同一次 cast）。代价是每迭代多 8KB 的 UB（`ds_u`）与 8KB 的 L1 缓冲，
  以及 4096 次跨步写（整 tile，两个 AIV 冗余后为 2×4096 次等价工作）。
  边界：转置副本与主副本同形（全 extent），尾块同样由掩码补 0。

**重设计项 R4：warp/线程级并行 → SimtVF 整 tile lane 映射（无 `sid`）**

- **源方案**：`threads=256` 的线程块，`T.Parallel(block_n, block_m)` 的元素映射由编译器按 256 线程划分，
  `T.gemm` 内部按 warp 切分。
- **NPU 新算法**：Cube 侧由 mad 指令承担（AIC，28 核），向量侧在 `T.SimtVF(threads=128)` 内以
  `T.Parallel(64, 64)` 覆盖**整 tile**，最内轴为 query（连续 `lse_u[j]` 的 lane 私有读）。副块编号不入
  用户代码：`T.Vector(vector=2)` 形态在本工具链**不可编译**（`InternalError … dup sid`，见 §3.5.4 反例
  表），自动调度会在 AIC/AIV 分段时自行合成 cthread sid 并令两个 AIV 各自执行整 tile 向量段。
- **语义保持论证**：逐元素计算的并行划分不影响结果（无跨 lane 归约）；`P/dS` 的 fp32 值由全局 key/query
  索引直接决定（`gi = n0 + i`、`gj = m0 + j`，均为整 tile 内的普通表达式，**不再需要半区偏移**）。
  累加分组变化只出现在 GEMM 的 K 维（见 R2）。冗余执行只重复同值写入，不改变任何输出。
  回退形态（若运行期发现双 AIV 数据分发异常）：`dual_copy` 2:1 分片 + 两个**静态**半区 buffer
  （`pT_lo/pT_hi`，各 (32,64)）+ 两段 `T.SimtVF` 半区循环 + `T.copy` 到 `pT_s[0:32,:]`/`pT_s[32:64,:]`
  ——半区偏移是编译期常量，同样不需要 `sid`（该形态的切片 dst copy 待 Stage 3 验证，见 §9.1 风险 #3）。

**重设计项 R5：全局原子加 → store-mode 原子写回**

- **源方案**：`T.atomic_add(dq_acc[...], dq_f[i,d])` 在 `T.Parallel` 内逐元素原子加。
- **NPU 新算法**：`T.set_atomic("add", "float32")` 打开硬件 store 原子模式，随后把整个
  `dq_c` 用一次 `T.copy` 落到 `dq_acc[q_start+m0 : …, h, :]`，再 `T.set_atomic_none()`。
  完全省掉 UB 中转（L0C fp32 → GM fp32 同 dtype，`mode.py:set_atomic` 明确该路径；已核验用例
  knowledge/repro/PL-950-atomic-splitk.py；本轮编译级确认成 `asc_copy_l0c2gm` 落在
  `asc_set_atomic_add_float` / `asc_set_atomic_none` 之间）。
- **语义保持论证**：`dq_acc += tile` 与逐元素原子加是同一数学累加；fp32 累加、顺序仍不确定（与源一致）。
  边界：尾 query 块由平台 OOB clamp 把 store 区段裁到 `S_q` 以内（设备码 `min(64, S_q - m0)`），
  不会写到下一 request 的行。

**无重设计项之外的补充说明**：`delta` 的产生（`_delta_kernel`）与 `dq` 的缩放 cast（`_dq_land_kernel`）
不在本 kernel 内，本设计只依赖其接口契约（fp32 / fp32 in-out），未对它们做任何假设。

### 0.7 标杆实现与 packed 分支说明

- **源算子路径**：`/workspace/TileOPs/src/tileops/_kernels/attention/mla/bwd.py`（`_bwd_kernel`，`bwd.py:109-264`），
  抽取副本 `examples/TileOPs/tileops/kernels/attention/mla_dense_bwd/_mla_dense_bwd_kernels.py`（逐行同源）。
- **golden 基准**：`/workspace/TileOPs/workloads/attention/mla.py:441-460` 的 `mla_bwd_reference`（fp32 torch，
  slab 物化 `probs` 后 `einsum` 反向，bottom-right 对齐，`probs` 用 `nan_to_num` 处理 `lse=-inf` 行）与
  `MLADenseBwdWorkload.ref_program`（`mla.py:506-518`，把 fp32 梯度**只舍入一次**到输入 dtype）。
  §8.1 的 golden 移植该实现，**不复刻** §0.6 的分块算法。
- **精度/性能用例**：`/workspace/TileOPs/tests/ops/test_mla_bwd.py`、`/workspace/TileOPs/benchmarks`（manifest
  `MLADenseBwdOp` 5 条 workload）；容差来源 `workloads/numerics.py:273-283` 的 `reference_tolerance` 与
  `MLADenseBwdWorkload.verification`（atol 下限 5e-3）。
- **packed 分支差异（为何不在本函数覆盖）**：`dense_seqlen_q=0` 时源工厂改走 `GroupTiling`
  （`bwd.py:138-139, 181-184`）：每 task 的 `(request, n0)` 由「变长序列的块累计表」解码，`total_q/total_kv`
  是动态维，`lse/delta` 布局从 `(B,H,S_q)` 变为 head-major `(H,T)`。这与 dense 的差别是**任务解码与统计张量
  layout**，不是数学；本平台一次编译绑定静态 shape，变长解码需要另一套块表与动态 extent 策略。该分支属于
  另一个算子（`MLAVarlenBwdOp`），会单独设计，本函数只实现 dense，`cu_seqlens_*` 参数保留在签名中是
  为了与 `_run` 的三 launch 调用序列一致（dense 下其值恒为 `arange(0,(B+1)*S,S)`，kernel 内用静态
  `S_q/S_kv/B` 直接寻址）。

---

## 1. 概述

### 1.1 算子名称

`_bwd_kernel`（算子：`MLADenseBwdOp` 的主 pass，MLA prefill 反向）。

### 1.2 功能描述

给定前向的 `q, k, v, do, lse, delta`，按 key-stationary 单遍重算 `P` 与 `dP`，产出 `dk, dv`（输入 dtype）
并向 fp32 缓冲 `dq_acc` 原地累加 `dQ`（缩放 cast 由 `_dq_land_kernel` 完成）。

### 1.3 数学公式

$$
\begin{aligned}
P[m,n] &= \exp\!\big(\mathrm{scale}\,(q_m\!\cdot\! k_n) - \mathrm{lse}_m\big)\cdot \mathbb{1}\big[n \le m + \mathrm{shift},\ n < S_{kv},\ m < S_q\big] \\
dP[m,n] &= do_m \cdot v_n \\
dS[m,n] &= P[m,n]\cdot\big(dP[m,n] - \mathrm{delta}_m\big) \\
dV[n,:] &= \textstyle\sum_m P[m,n]\,do[m,:], \qquad
dK[n,:] = \mathrm{scale}\cdot\textstyle\sum_m dS[m,n]\,q[m,:], \qquad
dQ[m,:] = \mathrm{scale}\cdot\textstyle\sum_n dS[m,n]\,k[n,:]
\end{aligned}
$$

本 kernel 的输出为 `dk = cast(dK)`、`dv = cast(dV)`、`dq_acc += dS\cdot k`（**不含** `scale`，由
`_dq_land_kernel` 乘 `scale` 后 cast），`scale = D_{qk}^{-1/2} = 192^{-1/2}`，`shift = S_{kv} - S_q`。

### 1.4 算法描述

算法族选择见 §1.6.0：**key-stationary 单遍重算（flash 反向族）**，与源算子同一族，但按 NPU 的
「GEMM 只支持 NT + 累加器只能在 L0C + 向量只在 AIV」三项约束重构操作数方向与数据落地（§0.6 R1–R5）。
迁移后的算法步骤（与 §1.6.0 选定算法一致）：

1. **任务映射**：一维核网格，`grid = H * B * ceil(S_kv/64)`；块内解出 `(h, b, n0)`，`n0` 为该块的 key 起点。
2. **key 常驻（per launch）**：`k`、`v` 与其转置副本 `kT` 从 GM 搬入 L1（`k_s (64,192)`、`v_s (64,128)`、
   `kT_s (192,64)`），跨整个 query 循环常驻；三者一次 `T.copy` 全 extent 装载。`dk/dv` 的 L0C 累加器清零。
3. **query 块循环**（因果剪枝：`kq ∈ [max(0,(n0-shift)//64), ceil(S_q/64))`）：
   a. 搬入 `q_s (64,192)`、`qT_s (192,64)`、`do_s (64,128)`、`doT_s (128,64)`、`lse/delta (64,)` fp32；
   b. `st_c = k_s · q_s^T`、`dpt_c = v_s · do_s^T`（NT，全 extent，`clear_accum` 每迭代）；
   c. 整 tile `T.copy` 把两块 L0C 落到 UB（fp32，同 dtype），向量侧算
      `p = exp(st*scale - lse)`（运行时块级 mask 特化 + 因果/尾块掩码）→ `pT`(f16)、
      `dsT = p*(dpt-delta)`(f16) 与转置副本 `ds`(f16, `(64,64)` 转置写)；
   d. 三份 fp16 中间块经整 tile UB→L1 落 `pT_s/dsT_s/ds_s`；
   e. `dv_c += pT_s · doT_s^T`、`dk_c += dsT_s · qT_s^T`（跨迭代累加）、`dq_c = ds_s · kT_s^T`（每迭代清零）；
   f. `set_atomic("add","float32")` 后把 `dq_c` 整块落到 `dq_acc[q 行, h, :]` 并关闭原子模式。
4. **块尾**：`dk_c`、`dv_c` 整 tile `T.copy` 落到 UB，向量侧 cast（`dk` 另乘 `scale`）后整 tile 写回 GM；
   尾部行由平台 OOB clamp 裁剪（§5.4）。

与源算法的结构差异（逐条对应 §0.6）：一维网格（R1）；L0C 累加 + 整 tile UB 中转（R2）；5 个 NT 乘积，
`dS` 双方向 + `qT/doT/kT` 转置装载（R3）；整 tile SimtVF lane 映射（R4，无 `sid`）；store-atomic 写
`dq_acc`（R5）。公式层面与源一致，仅采纳 §1.6.1 的两项等价化简。

### 1.5 数据流图

```
GM: q,k,v,do(f16)  lse,delta(fp32)  dq_acc(fp32, in-out)
        │  GM→L1 (MTE2, nd2nz / dn2nz；越界尾部由 OOB pass clamp+fill)   │
        ▼                                                                 │
   L1: k_s,v_s,kT_s(常驻)  q_s,qT_s,do_s,doT_s(每迭代)                    │
        │  L1→L0 (MTE1, 由 GEMM lowering 内部按 tile_k_sub 完成)           │
        ▼                                                                 │
   L0A/L0B ──mad──▶ L0C: st,dpt(每迭代) dk,dv(跨迭代) dq                  │
        │                                                                 │
        │ T.copy 整 tile (L0C→UB, fp32 同 dtype, 双 AIV 同名地址)          │
        ▼                                                                 │
      UB(fp32): st_u, dp_u ──SimtVF(128) 整 tile──▶ UB(f16): p16_u, dsT_u, ds_u
        │  UB→L1 (整 tile；InsertNd2Nz 改写成 NZ scatter)                  │
        ▼                                                                 │
      L1: pT_s, dsT_s, ds_s ──▶ 作为 dV/dK/dQ 的 A 操作数                  │
                                                                          │
   L0C(dq) ──set_atomic(add) + T.copy──▶ GM: dq_acc ──────────────────────┘
   L0C(dk,dv) ──T.copy→UB──cast(+scale)──T.copy→ GM: dk,dv
```

### 1.6 算法调研与优化分析

#### 1.6.0 算法调研（Algorithm Research）

> 调研对象：**MLA 注意力反向（给定 lse/delta 的梯度）这一数学语义的算法族**。源算法只是基线候选。
> 检索范围：平台 knowledge/algorithm-candidates.md（`ALG-attention` 行必查）、knowledge 下全部 950 条目
> （PL-950-*）、本仓 `examples/` 同类实现（`examples/flash_attention/flash_attn_npuir.py`、
> `examples/TileOPs/tileops/kernels/attention/mla_dense_bwd/_mla_dense_bwd_kernels.py`）、源仓 golden 与
> manifest roofline、结构判据自查。网络检索未使用（本地源已覆盖该族的全部已知结构，见下）。

**R1 等价化简公式候选**（基线在表内；正式论证与收益在 §1.6.1）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 纳入 R3 |
|---|------|------------|-----------|---------|---------|
| 0 | **基线（源算法）** | `P=exp2(s·scale·log2e − lse·log2e)`；5 个 GEMM（`dQ` 用 `transpose_A`）；`dq` 逐元素原子加 | — | — | ✅（基线） |
| 1 | 自然对数域 `exp` | `P=exp(s·scale − lse)` | 数学恒等（`exp(x)=exp2(x·log2e)`）；差异仅两式的舍入路径 | 每 score 少 1 次乘（源式逐元素重算的 `lse·log2e`），乘数由 `scale·log2e` 变为 `scale`（不增减） | ✅ |
| 2 | 块级 mask 特化 | 全可见块省掉 `select`，只在对角/尾块生成掩码 | 谓词在该块内恒真（可判定），恒等 | 每 score 少 2 次比较 + 1 次选择（约占 (key,query) 对数的 49%，实测 9285/18761 块走快路） | ✅ |
| 3 | 行统计量 `lse·log2e` 逐行提升 | 先算 `lse2[m]=lse[m]·log2e` 再进 score 循环 | 恒等 | 被 #1 完全吸收（自然域下无此因子） | ❌（冗余） |
| 4 | `scale` 折进 `q`/`k` 的装载 | `(q·scale)` 参与 score GEMM，省每 score 一次乘、并省 `dk` 写回的乘 | 数学恒等，但 fp16 输入先乘常量 → 每元素多一次 fp16 舍入 | 逐迭代乘数：score 64×64=4096 次 vs 预乘 64×192=12288 次 → **不省**；`q` 还要供 `dK`、`k` 还要供 `dQ`，需两份副本 | ❌（§1.6.1 定量） |
| 5 | `delta` 项改写成 rank-1 修正 | `dS = P∘dP − P∘delta` 拆成两项分别进 GEMM | 数学恒等 | 需多一个 GEMM（`P^T·diag(delta)·q`）→ 计算量增加 | ❌ |
| 6 | `dV`、`dK` 合并为一个 GEMM | 拼 `A=[pT; dsT]`、`B=[doT; qT]` | **不等价**（会被求和成 `pT·doT^T + dsT·qT^T`） | — | ❌（不等价） |
| 7 | 物化 `P/dS`（非 flash） | 一遍算 `s/P/dP/dS` 落 GM，再扫描做 3 个 GEMM | 恒等 | 中间缓冲 O(B·H·S²) 与多遍扫描 | ✅（进 R3 作为反例） |
| 8 | query-stationary 变体 | `dQ` 常驻、`dK/dV` 原子累加 | 恒等 | 把原子放大项从 `dQ` 换成 `dK+dV`（更宽） | ✅ |
| 9 | 两 kernel 拆分（dQ kernel + dK/dV kernel） | 各自单遍、输出只写一次 | 恒等 | FLOPs ×2，且破坏本 op 的三 launch 契约 | ✅ |

**R2 在线算法**：**有，且基线即在线变体。** 结构依据：本算子的两个统计量 `lse`（前向 log-sum-exp）与
`delta`（`rowsum(o*do)`）都由外部提供，因此反向**不需要**前向 online softmax 的 running max / 重缩放机器；
每个 (key 块, query 块) 对只需重算局部 `s/dP` 并立刻消费（`P/dS` 只活在片上），得到单遍扫描、
O(tile) 中间态的流式结构（源算子即如此，见 §0.4 #2，是「该族存在在线结构」的直接证据）。对照的非在线变体是 R1#7（物化
`P/dS`，3–4 遍、O(B·H·S²) 缓冲），在 R3/R4 中被容量淘汰。**无更优在线结构**：本算子的输出里必有一个
（`dQ` 或 `dK/dV`）要沿另一轴跨块归并，单遍结构下无法再压缩（R1#6 的合并已证不等价），故在线族内部
不存在「更少遍数」的变体——这条负向断言的结构依据是上面的交换性分析，而非经验。

**R3 复杂度对比**（四口径；代表 workload = `ds-v3-b2-8k`：B=2, S_q=S_kv=8192, H=128,
D_qk=192, D_v=128, bf16）：

| 候选 | FLOPs | 访存量 GM (Bytes) | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|------|-------|------------------|---------|-------------|--------------------|
| C0 基线（GPU 布局） | 14.3 T | ≈ 192 GB（q 51.5 + do 34.4 + k/v 1.4 + `dq` 原子 103 + `dk/dv` 1.3） | 1 | 寄存器 fragment + SMEM | 32768 任务 × 28 核；`dq` 原子跨核 |
| **C1 本设计（NT-only）** | 14.3 T | ≈ 280.5 GB（q/do 各两份 173.1 + k/v/kT 2.15 + `dq` 原子 103.9 + `dk/dv` 1.34） | 1 | L0C 160KB + L1 168KB + UB 176.5KB | 同上 |
| C2 query-stationary | 14.3 T | ≈ 346 GB（k/v 两份重读 171 + `dK/dV` 原子 172 + 3.6） | 1 | `dQ` 常驻 L0C 48KB + L1 同量级 | 同量级；原子目标更宽（`D_qk+D_v`） |
| C3 两 kernel 拆分 | **28.6 T**（`P`、`dP` 各算两遍） | ≈ 175 GB（无原子） | 2 | 同量级 | 两倍 launch 与两倍整型工作量 |
| C4 物化 `P/dS`（R1#7） | 14.3 T | ≥ 270 GB 额外往返 | 3–4 | **O(B·H·S²)** ≈ 34 TB → 远超 128 MiB L2 / HBM | 不可行 |
| C5 纯 Vector（无 Cube） | 14.3 T | 同 C1 | 1 | 同量级 | 28 个 Vector 核做矩阵乘（每核 lane 级），算力低 1–2 个数量级 |

口径说明：FLOPs 按 manifest roofline 公式 `B·H·(scores·(2(3D_qk+2D_v)+5) + rows·2·D_v)`；GM 访存量含
输入读 + 输出写 + 跨块累加往返，**不含** L1/UB 内部搬运（那部分单列在本节「L1 端口分解」与 R4）；
扫描遍数指 `q/k` 被 GM 读取的遍数；中间缓冲峰值指片上驻留（C4 列的是 GM 物化量）。所有数字在本设计的
分块参数下由公式手算，非实测。C1 的 14.3 T 是解析值；按 `ceil` 分块实际计算 14.48 T（含 0.78% 的对角
块量化开销：2 113 536 次迭代 × 4096 score 位 vs 解析 8.59e9 个因果 score 位）。

**每迭代 L1 端口分解（block_n=block_m=64；口径：一次 (key 块, query 块) 迭代）**：

| 项 | 组成 | 字节 |
|----|------|------|
| GEMM 操作数读 | `k_s 24576 + q_s 24576 + v_s 16384 + do_s 16384 + pT_s 8192 + doT_s 16384 + dsT_s 8192 + qT_s 24576 + ds_s 8192 + kT_s 24576` | 172032 |
| UB→L1 写 | `pT_s + dsT_s + ds_s = 3 × 8192` | 24576 |
| GM→L1 装载（每迭代） | `q_s 24576 + qT_s 24576 + do_s 16384 + doT_s 16384` | 81920 |
| GM→L1 常驻摊销 | `k_s + v_s + kT_s = 65536` ÷ 平均 64.5 次迭代 | 1016 |
| **合计** | | **279544**（≈279.5 KB，KB=1000 B；272.9 KiB） |

对应 `6 852 608 FLOP/迭代`（`4096×1669 + 64×256`）→ **24.5 FLOP/字节**；全量 = 2 113 536 × 279 544 B =
**590.6 GB** L1 端口流量。该口径（上一版写作 248 KB / 27 FLOP/字节，无法复现）由本表统一：
248 KB 落在「不含 GM→L1」（196.6 KB）与「全含」（278.5 KB）之间，本版采用**全含**口径。

**R4 硬件亲和性评估**（逐候选；容量对照 §4.5，常数缺失项显式标注）：

| 候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|------|-------------|---------|------------|---------|------------|------|
| C0 基线 | 需 `transpose_A` GEMM 与寄存器累加器，NPU 无对应（`gemm_mad.py:lower` 断言 L1 路径只允许 NT） | — | — | — | — | ❌ 不能直译 |
| **C1 本设计** | 5 个 NT GEMM 全落 Cube；逐元素 `exp/mask/cast` 落 AIV；无标量热点 | L0C 160/256KB（62.5%）、L1 168/512KB（32.8%）、UB 176.5/248KB（71.2%） | D_qk=192、D_v=128、64 全为 16 的倍数（mad 分形 16×16） | `S_q/S_kv/B/H` 编译期常量；GEMM 的 M/N/K 全为**静态 allocation extent**；循环界为整数式 | GEMM 与向量 epilogue 同 kernel 交替（混合模式），首版 `num_stages=1` | ✅（混合流水为最大未核验项，见 §9.1） |
| C2 query-stationary | 同 C1 | 同量级 | 同 | 同 | 同 | ❌ `dK/dV` 的原子放大项宽 320（>192），GM 访存再 +25%，且 bf16 输出做原子累加会二次舍入（源算子正是为避开它才选 key-stationary） |
| C3 两 kernel 拆分 | 同 C1 | 同 | 同 | 同 | 需两次 launch | ❌ FLOPs ×2，且违反「三次 launch」的算子契约 |
| C4 物化 | 纯 Cube 友好 | O(B·H·S²) ≈ 34 TB ≫ 128 MiB L2 | — | — | — | ❌ 容量淘汰 |
| C5 纯 Vector | 矩阵乘落错单元（56 个 Vector 核 lane 级乘加 vs Cube 分形） | 同 | — | — | — | ❌ 计算单元不匹配（无需实测即可判定量级差） |

**设计期估算**（D-2；约定见 `.agents/skills/_shared/standards/hardware-cost-model.md`）：
- 流量项 = GM 视图 **280.5 GB**（其中 q/do 的重读与 `dq` 原子写按 L2 驻留处理：每 (b,h) 工作集
  `q 3.1 + qT 3.1 + do 2.1 + doT 2.1 + dq_acc 6.3 ≈ 17 MB` « L2 128 MiB，故 HBM 侧 ≈ 理想 5.4 GB + 一次写出）。
- L1 端口项 = **590.6 GB**（2 113 536 次迭代 × 279 544 B，逐项见上表）。
- 发射项 ≈ 2 113 536 次迭代 × 约 18 个 tile 级 op（4 GM→L1 + 2 GM→UB + 5 GEMM + 2 L0C→UB + 3 UB→L1 + 2 向量段 + 1 原子 store）。
- 容量项 = L0C 62.5%、L1 32.8%、UB 71.2%（均满足）。

**估算下界: 待核验**（缺少 950 的聚合 GM/L2 带宽与 Cube/Vector 吞吐实测常数——平台 `cost_measurements`
为空，`.agents/platforms/toolchains/ascend/knowledge/bottlenecks.md` 明确「不使用 910B 的数值补齐」；
pattern-library/constants.md 的 `CONST-L1-port-bw`（148–154 GB/s/核）、`CONST-fabric-bw`（1.73 TB/s）
条目 device 字段为 910B2C，不适用本型号，仅作量级参考）。结构结论（不依赖缺失常数）：本算子的
**L1 端口操作数流是地板项**（24.5 FLOP/字节），提高分块只能小幅度改善（备选 128/32 实测口径 −5.1%，
见 §1.6.3）。

**调研结论**：选定算法族 **key-stationary 单遍重算（flash 反向族，C1）**，关键依据：R3 中 C1 与基线
FLOPs 相同而 GM 访存最低可达（C2 +25%、C3 2×、C4 容量淘汰、C5 单元不匹配），R4 中只有 C1 的 5 个乘法
能全部映射到 NT GEMM 且片上容量留有余量。相对基线的结构差异：**同一算法族，但按 NPU 约束重构操作数
方向（NT-only，q/do/k 各多一份转置副本）、把累加器从寄存器迁到 L0C、把逐元素原子加换成 store-atomic**；
源算子 §0.4 的 14 项优化手段的意图承接见 §0.5 处置列（全部有承接或明确舍弃理由，无静默丢弃）。
下游约束：§1.4 与 §1.6.1–§1.6.3 的分析对象即 C1；§3.1 用 §1.6.1 的优化后公式。

#### 1.6.1 数学等价优化（公式级）

分析对象 = §1.6.0 选定的 C1。逐项四要素如下，机器验证见本节末。

| # | 优化项 | 原式（基线） | 优化后公式 | 等价性论证 | 收益量化 |
|---|--------|-------------|-----------|-----------|---------|
| A1 | **自然对数域概率** | `P = exp2(s·scale·log2e − lse·log2e)`（其中 `s·scale` 后逐元素乘 `log2e`，`lse` 亦逐元素重算 `lse·log2e`） | `P = exp(s·scale − lse)` | `exp2(x·log2e) ≡ exp(x)` 是恒等式；两式对同一实数输入的结果只差舍入路径。fp64 参照下两式**各自**距真值 ≤ 1 个目标 dtype ulp，候选式不劣于基线式（下表 A1 两行） | 每 score 少 1 次乘（源式在 (i,j) 内逐元素重算的 `lse_s[j]·LOG2E`；`s·scale·log2e` 的乘数变为 `s·scale` 不增减）→ 每迭代 4096 次向量乘；`exp` 与 `exp2` 在本平台同属 SFU 指令（`T.exp` 有核验条目、`exp2` 无，见 §0.5 处置 #7），故指令数不减但不引入未核验 API |
| A2 | **块级 mask 特化** | 每个 (key 块, query 块) 都生成 `select(mask, p, 0)` | 块级谓词 `n0+64−1 < S_kv ∧ m0+64−1 < S_q ∧ n0+64−1 ≤ m0+shift` 为真时直接输出 `p`，否则同一 `select` | 该分支条件是掩码谓词在整块上恒真的**可判定充分条件**（`n≤m+shift` 对块内最大 `n` 与最小 `m` 成立即全真），故两式逐元素相同；机器验证对 6 组 shape 的全部 18761 个块逐元素比对，违反率 0 | 快路块占 49.5%（9285/18761）；每 score 省 2 次整数比较 + 1 次选择 → 大 S 下约 4096×2 次向量操作/迭代 |

**A2 的实现机制（v1 更正）**：块级谓词依赖 `n0 = f(bid)` 与 `m0 = kq·BLOCK_M`，二者都是**运行时量**，
因此 A2 由 kernel 内的**运行时块级分支**实现（Python 脚本层的 `if` 生成 IR 的 IfThenElse，两支各一套
整 tile epilogue；编译后成为两个 SIMT VF 函数 + 运行时二选一，见 §3.5.4 探针 C）。编译期唯一可静态判定的
只有「是否存在尾块」（`S_q % 64`、`S_kv % 64`），它决定掩码里要不要保留 `m0+j < S_q` / `n0+i < S_kv` 的
比较结构，但**不**决定某个具体块是否全可见。两支的代价：一个额外的布尔判断/块 + 两份 VF 代码体积
（本探针下源文本 4.3 KB → 两臂各约 1.4 KB），不影响结论与数值。

**快路的安全性论证**（为什么全可见块可以完全省掉掩码而不产生 inf/NaN）：快路要求 `m0+BLOCK_M ≤ S_q`
（块内无填充 query 行）且 `n0+BLOCK_N−1 ≤ m0+shift`（块内每个 key 对块内每个 query 都可见）——
即该块**不可能**包含「看不到任何 key」的行（`lse=-inf` 只出现在这类行上），故 `exp(s·scale − lse)` 的
指数输入有限，不会产生 inf/NaN；分块外的 `lse=-inf` 行一律落在掩码臂并与掩码一起置 0。

已确认无需改动的项（CSE/降代换自查，写明结论以免被误判为遗漏）：
`v = p*(dpt − delta)` 已是最优形态（`p*dpt − p*delta` 会多一次乘，无 CSE 空间）；
`p` 的 fp32 结果被 `pT`(cast) 与 `dsT`(cast) 共用（每元素只算 1 次 `exp`）；
`lse/delta` 已是每 query 行一次装载、按 lane 广播读（无逐元素重算）；
`dk` 的 `scale` 乘只发生在块尾（每 launch 64×192 = 12288 次），已无提升空间。

弃选项的定量依据（对应 R1 表 #4/#5/#6）：
`scale` 折进 `q` 需 64×192 = 12288 次预乘、且因 `q` 同时供 `dK`、`k` 同时供 `dQ` 而需两份副本，净亏；
`delta` 的 rank-1 拆分需额外一次 GEMM（+192×64×64 MAC/迭代 ≈ +5.7% 计算量）；
`dV/dK` 合并已被证明会将两项求和，不等价，禁止采纳。

**等价性机器验证（D-1）**：脚本 `examples/mla_dense_bwd/_bwd_kernel/verify_equiv.py`（torch CPU，fp64 参照，
含 `s·scale−lse ∈ [-8e2, 8e2]` 的随机样本 + 角点值：精确抵消、上/下溢、次正规、`lse=-inf` 空行、
`700/−700` 极值；A2 覆盖 6 组 shape 含尾块与 `S_q<S_kv`/`S_kv<S_q`）。revision v1 于 2026-11-23 重跑
（`ASCEND_RT_VISIBLE_DEVICES=""`，toolchain `ascend-950pr-9579 / TileLang ascend 0.1.15`），
输出 `EQUIV_SUMMARY ALL_PASS`：

| 采纳项 | 样本数 | 候选 vs 基线 最大 ulp | 违反率 | 候选 vs fp64 最大 ulp | 基线 vs fp64 最大 ulp | 结论 |
|---|---|---|---|---|---|---|
| A1 exp 域 P（目标 dtype float16） | 1048592 | 1 | 0.000e+00 | 1 | 1 | EQUIV_PASS |
| A1 exp 域 P（目标 dtype bfloat16） | 1048592 | 1 | 0.000e+00 | 1 | 1 | EQUIV_PASS |
| A2 块级 mask 特化（block_n=64, block_m=64） | 76838912 | 0 | 0.000e+00 | — | — | EQUIV_PASS |

（A2 快路块 9285/18761；违反率定义：与基线相差超过目标 dtype 1 个 ulp 的元素占比。）

**优化结论**：采纳 2 项（A1 自然对数域 `exp`、A2 块级 mask 特化），两项均 EQUIV_PASS；v1 的公式与
v0 完全一致（本轮变更只在 A2 的**实现机制**描述，不涉及数学）。
**优化后公式**（供 §3.1 使用）：

$$
P[m,n]=\exp\!\big(\mathrm{scale}\cdot s[m,n]-\mathrm{lse}_m\big)\quad\text{all-visible block}\qquad
dS[m,n]=P[m,n]\cdot\big(dP[m,n]-\mathrm{delta}_m\big)
$$

对角块与尾块再乘因果掩码（`T.if_then_else`，见 §1.6.1 A2 的块级特化）。

其余乘积与写回形态不变（`dV = P^T dO`、`dK = scale·dS^T q`、`dq_acc += dS·k`）。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

盘点实现方案中**全部**循环与标量计算点（C1 + §0.6 重设计结构；v1 已删除 `sid` 相关条目，并把尾块边界
改为"由平台 pass 消除"）：

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由（附查证） |
|---|--------|-----------|-------------|---------|----------------------|
| 1 | `P = exp(s·scale − lse)` | GPU 上 `T.Parallel(bn,bm)` 逐元素 | `T.SimtVF(threads=128)` + `T.Parallel(64,64)` + `T.exp` | ✅ | —（`T.exp` 核验：knowledge/repro/PL-950-row-primitives.py、PL-950-softmax-boundaries.py；本轮编译级确认，§3.5.4 探针 A） |
| 2 | 因果 / 尾块掩码 | 逐元素布尔 + `select` | 同一 `T.Parallel` 内的 `T.if_then_else`（A2 运行时块级特化掉全可见块） | ✅ | —（`if_then_else` 核验：PL-950-attention.py:50 一带；A2 分支形态见探针 C） |
| 3 | `dS = P·(dP − delta)` + 列广播 | 逐元素 | 同一 `T.Parallel` 内读 `dlt_u[j]`（lane 私有读，无广播指令） | ✅ | —（列/行广播 + 转置在 SIMT 内表达已核验：knowledge/repro/PL-950-conversion-layout.py） |
| 4 | `pT`/`dsT`/`ds` 的 fp32→fp16 cast | 逐元素 | `T.cast` 写另一个整 tile UB buffer | ✅ | —（转换含 ±inf/次正规已核验：PL-950-conversion-layout.py「转换」段） |
| 5 | `ds (Mq,Nk)` 转置副本 | GPU 不需要（`transpose_A`） | 同一 `T.Parallel` 里 `ds_u[j,i] = v`：SIMT 内转置写 | ✅ | —（「转置与列广播在 SIMT 中表达」已核验，同上条目；探针 A 的设备码中该 copy 被改写成 `ascend_nd2nz_scatter`） |
| 6 | `dk` 写回 `cast(dk_c·scale)` | 逐元素 | 块尾一次 `T.Parallel(64,192)`（整 tile `T.copy` 落地后） | ✅ | — |
| 7 | `P/dS` 的 K 维归约 | GPU tensor core | Cube mad（`T.gemm`，L0C fp32 累加） | ✅（换单元，非标量循环） | — |
| 8 | `dq_acc += dq` | 逐元素 `atomic_add` | `T.set_atomic("add","float32")` + 一次 L0C→GM `T.copy` | ✅（硬件原子，无标量循环） | —（核验：knowledge/repro/PL-950-atomic-splitk.py；探针 A 设备码 `asc_copy_l0c2gm` 位于 set/none 之间） |
| 9 | 任务索引 `h/b/n0`、`loop_st/loop_ed`、`shift` | 标量整数运算 | 无 | ❌ | block 级任务映射计算（每 block 一次的常数项），逐元素等价 API 不存在；见 §3.3 伪代码 |
| 10 | 尾块边界（有效行/列数与 clamp） | 源码 `T.if_then_else` 逐元素守卫 + `T.min` 区段 | **无需替代**：GEMM 用全 extent，GM↔片上搬运的越界尾部由平台 `AscendInsertOOBPadding` 自动改写成 `min()` clamp + `asc_fill_l1` 补 0（§5.4、探针 B 设备码） | ✅（由平台 pass 消除） | —（`pipeline.py` 该 pass 的 docstring 明确 "Clamp DMA copy OOB tails … emit GM->L1 padding fills"；探针 B 生成码含 `min(64, 96 - n0)` 与 3 处 `asc_fill_l1`） |
| 11 | `lse/delta` 行统计装载 | 逐元素 | `T.copy` 搬 (64,) fp32 | ✅ | — |

**向量化结论**：11 个计算点中 **10 项**已向量化（#1–#8、#11）/换单元/由平台 pass 消除（#10），仅 **1 项（#9）**保留为标量，
理由类别 = **block 级任务映射**（每 block 一次，非逐元素热点）。上一版的 #12「两 AIV 的 key 轴偏移
（`sid`）」已随阻塞问题 2 一并删除——重设计后整 tile epilogue 不需要任何执行体标识。

#### 1.6.3 向量化轴与数据布局决策 ⭐

> 迁移任务：源算子的轴选择（GPU 线程/warp 映射）只是输入，下面是独立评估。

本算子的核内布局自由度集中在 **(key × query) 的 2-D 中间块**：`P`、`dP`、`dS` 各 64×64。I/O 契约固定为
BSHD（尾轴 `D` 连续），`D` 维是矩阵维（Cube 契约），因此「向量化轴」只能在 key/query 两轴中选。

| 评分项 | 候选 1：lane = query（本设计） | 候选 2：lane = key | 候选 3：UB 内双转置（(Mq,Nk) in UB, (Nk,Mq) in L1） |
|--------|------------------------------|-------------------|--------------------------------------------------|
| 整除性 / 尾 lane | Mq=64，fp32 向量宽度 4 → 16 个向量/行，无尾 lane | Nk=64，同 | 同 |
| 累加链形态 | 最内轴 stride=1（`(Nk,Mq)` 行内连续），`lse/delta` 为 lane 私有读（无需广播指令） | 最内轴是 key，`lse/delta` 需沿 key 轴**列广播**（每次多一条广播/索引重算） | 同候选 1 |
| repack 代价 | 只需 1 份转置副本（`ds`，4096 元素跨步写/迭代） | GEMM 的 A 操作数（`P^T/dS^T`）本就要 (Nk,Mq)，而 lane=key 会让 `P/dS` 的 UB 布局变成 (Mq,Nk) → 需给 `dV` **和** `dK` 各造一份转置副本（2 份） | 同候选 2（2 份转置） |
| UB 容量影响 | 每 AIV 每迭代 57856 B（整 tile） | 同量级 | +8192 B（多一份替身） |
| 弃选量化理由 | ✅ 采纳 | ❌ 转置副本 2 份 vs 1 份（多 4096 次跨步写 + 8KB UB/迭代），且每元素多一次列广播 | ❌ 同上 |
| 候选 4：纯 SIMT 直写 GM（`paged_cache_write` 先例） | AIC 不参与 | — | ❌ 5 个 64×64×192 级矩阵乘落 56 个 Vector 核，算力差 1–2 个数量级（R4 C5） |
| 候选 5：中间块经 GM 往返（`store_fixpipe` L0C→GM） | — | — | ❌ 每迭代多 3×8KB 的 GM 往返，全量约 +103 GB HBM 流量（`CONST-store-fixpipe-gm-only` 的 910 结论在 950 上已被 L0C→UB 双目标搬运取代，见 knowledge/repro/PL-950-mixed-epilogue.py） |

**多 AIV 的执行体映射（v1 更正）**：本平台把线程域限定在 `T.SimtVF` 内，副块（AIV）编号只由自动调度
在 AIC/AIV 分段时合成（设备码中 `int32_t sid = asc_get_sub_block_id()`，用户不可见）；显式
`T.Vector(vector=2) as sid` 在本工具链**不可编译**（§3.5.4 反例表）。因此本设计的向量段是**整 tile**
（`T.Parallel(64,64)`，无半区偏移），一个混合 kernel 的 `__mix__(1,2)` 下两个 AIV 各自执行同一段
（冗余计算、同值写入），**向量侧有效吞吐按 1 个 AIV 计**。这不改变任何数值结果：逐元素映射无跨 lane
归约，`P/dS` 只由全局 key/query 索引决定。代价评估：向量工作量按每迭代 4096 元素 ×（1 exp + 3 乘 +
3 cast + 1 select）≈ 每迭代 3.3 万 lane-op（4096 × 8），相对 6.85 MFLOP 的 Cube 工作量仍低一个量级，冗余执行不构成
新的地板项（L1 端口仍是主导项，见下）。

**布局决策结论**：选定候选 1——核内布局为 **(key, query)** 且**向量化轴 = query**（lane 映射到 `Mq=64`
的最内轴），中间块以 `(Nk,Mq)` 整 tile 存在 L0C/UB，仅 `dS` 额外产一份 `(Mq,Nk)` 转置副本供 `dQ` 的
NT 形式使用；向量段为整 tile、无 `sid`、由 `__mix__(1,2)` 的双 AIV 冗余执行（有效吞吐按 1 AIV 计）。
该结论已同步到 §3.3 伪代码（`T.Parallel(64, 64)` 的内层为 query 维）、§4.5（整 tile UB 预算）与 §6 循环结构。

**备选方案与实验裁决计划**（判定裕度依赖未实测常数：L1 端口带宽与 Cube 吞吐，见 §1.6.0 估算行）：

- **主选（进 Stage 3）**：`block_n=64, block_m=64`。L0C 163840 B/262144（62.5%），L1 172032 B
  （双版本 278528 B，53.1%），UB 180736 B（71.2%）；每迭代 L1 端口 279544 B、24.5 FLOP/字节，
  2 113 536 次迭代合计 590.6 GB；GM 视图 280.5 GB。
- **备选（≤1 个）**：`block_n=128, block_m=32`。同样只改分块参数：
  L0C = `dk 98304 + dv 65536 + st 16384 + dpt 16384 + dq 24576 = 221184 B`（216 KiB，84.4%，余量 40 KB）；
  L1 = `k_s 49152 + v_s 32768 + kT_s 49152 + q_s 12288 + qT_s 12288 + do_s 8192 + doT_s 8192 +
  pT_s 8192 + dsT_s 8192 + ds_s 8192 = 196608 B`（192 KiB，37.5%）；
  UB：每迭代 57600 B + 块尾 122880 B（**块尾必须分两段 (64,·) 写回**，否则整块尾缓冲 245760 B 会使
  UB 上界到 303360 B、超 248 KB）= 180480 B（71.1%）；
  每迭代 L1 端口 263164 B、26.0 FLOP/字节（−5.1%），2 129 920 次迭代合计 560.5 GB；GM 视图 143.0 GB
  （`dq` 原子 103.9 → 52.3 GB）；代价是 L0C 余量降到 15.6%、GEMM 的 N 维（query）降到 32。
  核数：`grid = H·B·ceil(S_kv/128)`，smoke 下 8·1·4 = 32 个任务（≥ 28 核）仍满足一核一次发射。
- **裁决计划**：代表 shape 取 smoke（512）与 `ds-v3-4k`；指标 = kernel Task Duration（msopprof）+ 反解
  L1 端口有效带宽（用 `mte2`/`mte1` busy 时间与上述流量数反解 GB/s）；判定阈值 = 备选 latency 优于主选
  ×0.9 才翻转（整体或按 `S_kv` 分片经 `lru_cache` 工厂分派）；实测数据回写本节与 §1.6.0 估算行。
  量化裕度说明：按本轮统一口径，备选的 L1 端口只降 5.1%（非上一版的 15%），而 L0C 余量从 37.5% 掉到
  15.6%，故**只有 GM 侧（HBM/L2）成为实测地板项时**才可能翻转——这正是裁决计划要测的量。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式：混合（Cube + Vector）。**（平台 `programming_modes.choices` 的 `hybrid`；`migration_default`
为 `simd`，偏离理由见 2.2，属有证据的偏离。）

### 2.2 选型理由

- 算子含 5 个 64×64×192 级矩阵乘（每迭代约 3.4M MAC），**必须落 Cube**（`T.gemm`）；若走 SIMD/SIMT
  纯向量路径，56 个 Vector 核的 lane 级乘加比 Cube 分形低 1–2 个数量级（§1.6.0 R4 C5）。
- 同时每迭代必须做 `exp`、掩码、cast、跨步转置等逐元素工作，**必须落 Vector（AIV）**；平台把线程域
  限定在 `T.SimtVF` 内（`.agents/platforms/toolchains/ascend/references/programming.md`「API 与执行区域」）。
- 两侧必须在**同一个 kernel** 内交替（中间块 `P/dS` 不落 GM），平台支持该形态：`T.Kernel` 单核网格 +
  自动调度把 op 分派到 AIC/AIV（`AssignCore`/`ResolveCore`/`InsertSync`，见平台 `pipeline.py`），
  已核验形态见 knowledge/repro/PL-950-mixed-epilogue.py（Cube GEMM + Vector ReLU + L0C→UB 同 kernel）；
  本设计的主选骨架已在**编译级**走通（§3.5.4 探针 A，设备码 `__mix__(1, 2)`）。
- 平台默认 `migration_default=simd` 面向逐元素/规约类算子；本算子的主导算力在矩阵乘，选 `simd` 会把
  矩阵乘降级到向量单元，因此按上述证据选 `hybrid`。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_shared`（UB，向量侧）、`T.alloc_l1`（GEMM 操作数）、`T.alloc_l0c`（fp32 累加器）；L0A/L0B 由 GEMM lowering 内部管理，不显式分配 |
| 计算方式 | Cube：`T.gemm(A_l1, B_l1, C_l0c, transpose_B=True, clear_accum=…)`，A/B/C 一律用 allocation 全 extent；Vector：`with T.SimtVF(threads=128): for i, j in T.Parallel(64, 64)` |
| 执行体 | **不写 `T.Vector`**。整 tile 向量段由 `T.SimtVF(threads=128)` 表达；副块 id 由自动调度在 AIC/AIV 分段时合成（设备码 `int32_t sid = asc_get_sub_block_id()`），两个 AIV 冗余执行同一整 tile 段（§1.6.3） |
| 同步 | 自动（默认 `allow_autoschedule=True`，AutoSchedule + InsertSync 插入）→ §7 |
| 跨单元数据 | L0C→UB 用**整 tile** `T.copy`（fp32 同 dtype，兼容回退为 `dual_copy` 2:1 分片）；UB→L1 用整 tile `T.copy`（`InsertNd2Nz` 改写成 `ascend_nd2nz_scatter`）；L0C→GM 用 `T.copy` + store-atomic |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入为 §1.6.1 优化后公式；每个步骤与 §1.6.2 的向量化结论一致。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `s^T = k_s · q_s^T` | NT GEMM，A=`k_s (64,192)`、B=`q_s (64,192)`，C=`st_c (64,64)` fp32 |
| 2 | `dP^T = v_s · do_s^T` | NT GEMM，A=`v_s (64,128)`、B=`do_s (64,128)`，C=`dpt_c (64,64)` |
| 3 | `p^T = exp(s^T·scale − lse[j]) ⊙ mask` | 向量（lane=query，整 tile），A2 运行时块级特化：全可见块省掩码 |
| 4 | `pT = f16(p^T)`；`dsT = f16(p^T ⊙ (dP^T − delta[j]))`；`ds = f16(p^T ⊙ (dP^T − delta[j]))`（转置写） | 向量 + cast；`p^T` 只算一次 `exp`（CSE） |
| 5 | `dV = pT · doT^T` | NT GEMM，A=`pT_s (64,64)`、B=`doT_s (128,64)`，C=`dv_c (64,128)` 累加 |
| 6 | `dK = scale · dsT · qT^T` | NT GEMM，A=`dsT_s (64,64)`、B=`qT_s (192,64)`，C=`dk_c (64,192)` 累加（`scale` 在块尾乘） |
| 7 | `dQ = ds · kT^T` | NT GEMM，A=`ds_s (64,64)`、B=`kT_s (192,64)`，C=`dq_c (64,192)` 每迭代清零 |
| 8 | `dq_acc += dQ` | store-atomic：`set_atomic("add","float32")` + `T.copy(dq_c → dq_acc)` |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API（`tilelang.ascend.language`） | 参数 | 模式 |
|------|----------|--------------------------------------------|------|------|
| 0 | 任务映射 | `T.Kernel(grid)` 一维核网格 | `grid = H*B*ceil(S_kv/64)`；`T.Kernel` 不接受 `threads` | — |
| 1 | `k/v/kT` 常驻 | `T.copy(GM region, l1_buf)`（`kT` 加 `transpose=True`）；dst 用**全 allocation extent** | `k[b*S_kv+n0 : +64, h, :] → k_s`；`kT_s` 走 dn2nz | MTE2 |
| 2 | `q/do` 双方向装载 | 同上（两份：ND 与 `transpose=True`） | `q_s (64,192)`、`qT_s (192,64)` | MTE2 |
| 3 | `lse/delta` | `T.copy(lse[b,h,m0:m0+64], lse_u)` | 一维 fp32 区段 | MTE2 |
| 4 | 5 个 NT GEMM | `T.gemm(A, B, C, transpose_B=True, clear_accum=cb)`；`A/B/C` 一律**全 extent**（无切片） | 见 §3.1；`A/B` 在 L1、`C` 在 L0C | Cube |
| 5 | L0C→UB | `T.copy(st_c, st_u)` / `T.copy(dpt_c, dp_u)` | 整 tile (64,64) fp32 同 dtype | AIC fixpipe |
| 6 | `exp/mask/dS/cast/转置写` | `with T.SimtVF(threads=128): for i, j in T.Parallel(64, 64)` | `T.exp`、`T.if_then_else`、`T.cast`；A2 为运行时块级 `if` 两支 | Vector |
| 7 | UB→L1 | `T.copy(p16_u, pT_s)`、`T.copy(dsT_u, dsT_s)`、`T.copy(ds_u, ds_s)` | 整 tile；由 `InsertNd2Nz` 改写为 NZ scatter | MTE |
| 8 | `dq_acc` 累加 | `T.set_atomic("add","float32")` → `T.copy(dq_c, dq_acc[rows,h,:])` → `T.set_atomic_none()` | L0C fp32 → GM fp32 | AIC fixpipe |
| 9 | `dk/dv` 写回 | `T.copy(dk_c, tail_u)` → 向量 `cast`（`dk` 乘 `scale`）→ `T.copy(out_k, GM region)` | 整 tile；尾部由 OOB pass 裁到 `S_kv` | AIC + Vector |
| 10 | 多缓冲 | `T.annotate_buffer_versions({buf: 1})` | 首版固定 1 版本（§6.3） | — |

### 3.3 计算伪代码

```python
BLOCK_N, BLOCK_M = 64, 64          # keys / queries
D_QK, D_V = 192, 128
NT_PER_B = ceildiv(S_KV, BLOCK_N)              # 每 request 的 key 块数（静态）
NUM_TILES = B * NT_PER_B                       # 静态
SHIFT = S_KV - S_Q

@T.prim_func
def _bwd_main(q, k, v, do, lse, delta, cu_seqlens_q, cu_seqlens_kv, dq_acc, dk, dv):
    with T.Kernel(H * NUM_TILES) as bid:
        h = bid // NUM_TILES
        tile = bid % NUM_TILES
        b = tile // NT_PER_B
        n0 = (tile % NT_PER_B) * BLOCK_N
        # ---- L1 常驻（每 launch） ----
        k_s  = T.alloc_l1((BLOCK_N, D_QK), dt);  v_s = T.alloc_l1((BLOCK_N, D_V), dt)
        kT_s = T.alloc_l1((D_QK, BLOCK_N), dt)
        # ---- L1 每迭代 ----
        q_s  = T.alloc_l1((BLOCK_M, D_QK), dt);  qT_s = T.alloc_l1((D_QK, BLOCK_M), dt)
        do_s = T.alloc_l1((BLOCK_M, D_V), dt);   doT_s = T.alloc_l1((D_V, BLOCK_M), dt)
        pT_s = T.alloc_l1((BLOCK_N, BLOCK_M), dt)
        dsT_s = T.alloc_l1((BLOCK_N, BLOCK_M), dt); ds_s = T.alloc_l1((BLOCK_M, BLOCK_N), dt)
        # ---- L0C 累加器 ----
        st_c  = T.alloc_l0c((BLOCK_N, BLOCK_M), "float32")
        dpt_c = T.alloc_l0c((BLOCK_N, BLOCK_M), "float32")
        dq_c  = T.alloc_l0c((BLOCK_M, D_QK), "float32")
        dk_c  = T.alloc_l0c((BLOCK_N, D_QK), "float32")   # 跨迭代
        dv_c  = T.alloc_l0c((BLOCK_N, D_V), "float32")    # 跨迭代
        # ---- UB（整 tile，无半区 / 无 sid） ----
        st_u = T.alloc_shared((BLOCK_N, BLOCK_M), "float32")
        dp_u = T.alloc_shared((BLOCK_N, BLOCK_M), "float32")
        p16_u = T.alloc_shared((BLOCK_N, BLOCK_M), dt)
        dsT_u = T.alloc_shared((BLOCK_N, BLOCK_M), dt)
        ds_u  = T.alloc_shared((BLOCK_M, BLOCK_N), dt)
        lse_u = T.alloc_shared((BLOCK_M,), "float32"); dlt_u = T.alloc_shared((BLOCK_M,), "float32")
        tail_u  = T.alloc_shared((BLOCK_N, D_QK), "float32")   # 块尾 dk 中转（fp32）
        tailv_u = T.alloc_shared((BLOCK_N, D_V), "float32")    # 块尾 dv 中转（fp32）
        out_k = T.alloc_shared((BLOCK_N, D_QK), dt)
        out_v = T.alloc_shared((BLOCK_N, D_V), dt)

        # 全 extent 装载；越界尾部由 AscendInsertOOBPadding 改写为 min() clamp + fill 0
        T.copy(k[b * S_KV + n0 : b * S_KV + n0 + BLOCK_N, h, :], k_s)
        T.copy(v[b * S_KV + n0 : b * S_KV + n0 + BLOCK_N, h, :], v_s)
        T.copy(k[b * S_KV + n0 : b * S_KV + n0 + BLOCK_N, h, :], kT_s, transpose=True)
        loop_st = T.max(0, T.floordiv(n0 - SHIFT, BLOCK_M))
        loop_ed = ceildiv(S_Q, BLOCK_M)
        for kq in T.Pipelined(loop_st, loop_ed, num_stages=1):
            m0 = kq * BLOCK_M
            T.copy(q[b * S_Q + m0 : b * S_Q + m0 + BLOCK_M, h, :], q_s)
            T.copy(do[b * S_Q + m0 : b * S_Q + m0 + BLOCK_M, h, :], do_s)
            T.copy(q[b * S_Q + m0 : b * S_Q + m0 + BLOCK_M, h, :], qT_s, transpose=True)
            T.copy(do[b * S_Q + m0 : b * S_Q + m0 + BLOCK_M, h, :], doT_s, transpose=True)
            T.copy(lse[b, h, m0 : m0 + BLOCK_M], lse_u)
            T.copy(delta[b, h, m0 : m0 + BLOCK_M], dlt_u)
            # ---- 2 个 NT GEMM：A/B/C 全 extent（M/N/K 编译期静态） ----
            T.gemm(k_s, q_s, st_c, transpose_B=True, clear_accum=True)
            T.gemm(v_s, do_s, dpt_c, transpose_B=True, clear_accum=True)
            # ---- 整 tile epilogue（无 T.Vector / 无 sid） ----
            T.copy(st_c, st_u); T.copy(dpt_c, dp_u)
            if (n0 + BLOCK_N <= S_KV) & (m0 + BLOCK_M <= S_Q) & (n0 + BLOCK_N - 1 <= m0 + SHIFT):
                # A2 快路：整块全可见且无填充行 → 不需要掩码
                with T.SimtVF(threads=128):
                    for i, j in T.Parallel(BLOCK_N, BLOCK_M):     # 内层 = query（lane 轴）
                        p = T.exp(st_u[i, j] * SCALE - lse_u[j])
                        p16_u[i, j] = T.cast(p, dt)
                        dsv = p * (dp_u[i, j] - dlt_u[j])
                        dsT_u[i, j] = T.cast(dsv, dt)
                        ds_u[j, i] = T.cast(dsv, dt)              # 转置副本（SIMT 内转置写）
            else:
                # 掩码臂：对角块与尾块
                with T.SimtVF(threads=128):
                    for i, j in T.Parallel(BLOCK_N, BLOCK_M):
                        gi = n0 + i
                        gj = m0 + j
                        vis = (gi < S_KV) & (gj < S_Q) & (gi <= gj + SHIFT)
                        p = T.exp(st_u[i, j] * SCALE - lse_u[j])
                        pv = T.if_then_else(vis, p, T.cast(0, "float32"))
                        p16_u[i, j] = T.cast(pv, dt)
                        dsv = pv * (dp_u[i, j] - dlt_u[j])
                        dsT_u[i, j] = T.cast(dsv, dt)
                        ds_u[j, i] = T.cast(dsv, dt)
            T.copy(p16_u, pT_s); T.copy(dsT_u, dsT_s); T.copy(ds_u, ds_s)
            # ---- 3 个 NT GEMM：dV/dK 跨迭代累加，dQ 每迭代清零 ----
            T.gemm(pT_s, doT_s, dv_c, transpose_B=True, clear_accum=(kq == loop_st))
            T.gemm(dsT_s, qT_s, dk_c, transpose_B=True, clear_accum=(kq == loop_st))
            T.gemm(ds_s, kT_s, dq_c, transpose_B=True, clear_accum=True)
            # dQ 原子累加（L0C fp32 → GM fp32）
            T.set_atomic("add", "float32")
            T.copy(dq_c, dq_acc[b * S_Q + m0 : b * S_Q + m0 + BLOCK_M, h, :])
            T.set_atomic_none()
        # 块尾：dk = cast(dk_c*SCALE)、dv = cast(dv_c)，越界 key 行由 OOB clamp 裁掉
        T.copy(dk_c, tail_u); T.copy(dv_c, tailv_u)
        with T.SimtVF(threads=128):
            for i, d in T.Parallel(BLOCK_N, D_QK):
                out_k[i, d] = T.cast(tail_u[i, d] * SCALE, dt)
            for i, d in T.Parallel(BLOCK_N, D_V):
                out_v[i, d] = T.cast(tailv_u[i, d], dt)
        T.copy(out_k, dk[b * S_KV + n0 : b * S_KV + n0 + BLOCK_N, h, :])
        T.copy(out_v, dv[b * S_KV + n0 : b * S_KV + n0 + BLOCK_N, h, :])
```

注（与 v0 的差异，逐条对应 §11 的阻塞项修正）：

1. **5 个 GEMM 的 A/B/C 全部是 allocation 全 extent**，没有任何 `[:an]/[:am]` 切片——L1-GEMM 的
   M/N/K 取自 region extent，符号量会直接 `ValueError`（§3.5.4 反例 1）。
2. **尾块由「全 extent 装载 + 掩码 + 平台 OOB clamp」三件套处理**：装载/写回的越界尾部由
   `AscendInsertOOBPadding` 改写成 `min()` clamp 并补 0（§3.5.4 探针 B 的设备码证据），被补 0 的行/列
   再由掩码臂的 `gi < S_KV` / `gj < S_Q` 置零（补 0 只让 `s=0`，但 `exp(0·scale − lse) ≠ 0`，所以掩码
   必须保留这两项——这是与「靠补 0 就够」的关键区别）。
3. **无 `T.Vector`/`sid`**：整 tile 向量段，不出现任何执行体标识（§3.5.4 反例 2）。
4. `dq_acc` 行偏移为 `b*S_Q + m0`（dense 下等价于 `q_start + m0`）。

### 3.4 API 可行性确认

| API | 来源 / 核验状态 |
|-----|----------------|
| `T.Kernel(grid)`（一维，无 `threads`） | `ascend/language/kernel.py:Kernel`（源码确认，多维拒绝已检查）；核验条目 PL-950-gemm-tiles.py 用一维网格 |
| `T.alloc_l1` / `T.alloc_l0c` / `T.alloc_shared` | `ascend/language/allocate.py`；PL-950-cube-tile.md、PL-950-gemm-tiles.py（已核验） |
| `T.gemm(A_l1, B_l1, C_l0c, transpose_B=True, clear_accum=…)` | `gemm_op.py:_dialect_gemm_call` → `GemmMAD._lower_l1`（源码确认：**只支持 NT**，含 `assert`）；**L1 路径本轮编译级走通**（§3.5.4 探针 A/B：设备码 5 个 `ascend_gemm_l1<...>`，`clear_accum` 可为运行期布尔） |
| `T.copy` GM→L1（含 `transpose=True` dn2nz） | `copy_op.py:copy` 文档 + PL-950-gemm-tiles.py 的 `trans_b` 分支（已核验）；探针 A/B 生成 `asc_copy_gm2l1_nd2nz` / `asc_copy_gm2l1_dn2nz` |
| `T.copy` L0C→UB（整 tile，fp32 同 dtype） | `copy.cc` 的 `kL0CToUB` 路径（同 dtype 断言在 `ICHECK(!destination_exceeds_source)`）；探针 A 生成 `asc_copy_l0c2ub(..., 64, 64, 64, 64, ...)` |
| `T.copy` UB→L1（整 tile） | 由 `InsertNd2Nz` 把 UB→L1 复制改写成 NZ scatter（`ascend/pipeline.py:88-91`「Rewrite UB→L1 copies into NZ buffers as scatter+post_copy」）；探针 A 生成 3 个 `ascend_nd2nz_scatter<64, 64, half, half>`（`pT_s/dsT_s/ds_s`）——**这是本轮可复现的证据**（v0 误引了 `dual_copy` 的 docstring） |
| `T.dual_copy` L0C→UB 2:1 分片 | PL-950-mixed-epilogue.py（已核验，同 dtype fp32、M-split 2 目标）；仅作 §9.1 风险 #3 的回退形态 |
| `T.set_atomic("add","float32")` + L0C→GM `T.copy` | `mode.py:set_atomic` + PL-950-atomic-splitk.py（已核验）；探针 A 生成 `asc_set_atomic_add_float` … `asc_copy_l0c2gm` … `asc_set_atomic_none` |
| `T.SimtVF(threads=128)` + `T.Parallel(64,64)` + `T.exp` + `T.if_then_else` + `T.cast` | PL-950-row-primitives.py、PL-950-softmax-boundaries.py、PL-950-attention.py（已核验）；探针 A/C 生成 `__simt_vf__ __launch_bounds__(128)` + `asc_vf_call<...>(cce::dim3(128), ...)` |
| SIMT 内转置写 | PL-950-conversion-layout.py（已核验，fp32/fp16/bf16）；探针 A 的 `ds_u[j,i]` 写法编译通过并并入 `ascend_nd2nz_scatter` |
| `T.annotate_buffer_versions` | `annotations.py` + programming.md「同步与缓冲版本」（源码确认）；未核验 |
| `T.Pipelined(start, stop, num_stages=1)` 动态界 | 探针 A/B 生成 `for (kq = <运行时下界>; kq < <上界>; ++kq)`（源码确认）；**动态界 + num_stages>1 未核验**（首版取 1） |
| `T.Vector(vector=2) as sid` | **本工具链不可用**：与 `dual_copy`/UB→L1/GEMM 混用触发 `InternalError … dup sid`（§3.5.4 反例 2）。本设计**不使用** |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束（来源） | 是否涉及 | 处理方案 |
|--------------|---------|----------|
| `T.Kernel` 只接受一维核网格、无 `threads`（programming.md） | 是 | 一维 `grid = H*B*ceil(S_kv/64)`，块内 div/mod 解码（§0.6 R1） |
| L1 GEMM 只支持 `trans_A=False, trans_B=True`（`gemm_mad.py:lower` 断言） | 是 | 5 个乘法全部改写成 NT，`q/do/k` 各多一份转置副本，`dS` 多一份转置副本（§0.6 R3） |
| **L1-GEMM 的 M/N/K 取自 region extent，必须编译期静态**（`gemm_op.py:89-104`，符号量抛 `ValueError`） | 是 | GEMM 的 A/B/C **一律用 allocation 全 extent**（§3.3）；尾块改由「全 extent 装载 + OOB clamp 补 0 + 掩码臂置 0」处理（§5.4）；动态 extent **只出现在 copy**（§3.5.4 探针 A/B 已验证） |
| L0C 是唯一可累加层，且必须带 Ascend fractal layout（`gemm_mad.py:_prepare_lower`） | 是 | `dk_c/dv_c/dq_c/st_c/dpt_c` 五个 L0C 累加器，共 163840 B（§4.6） |
| L0C→UB 不支持 dtype 转换（`copy_op.py` 的 L0C→UB 路径断言同 dtype） | 是 | L0C→UB 保持 fp32 整 tile，cast 在 UB 内用 `T.cast` 完成 |
| 线程域只在 `T.SimtVF` 内（programming.md） | 是 | 所有逐元素工作包在 `T.SimtVF(threads=128)` |
| 显式 `T.Vector(vector=2)` 与 `dual_copy`/UB→L1/GEMM 混用不可编译（`InternalError … dup sid`） | 是 | **不使用** `T.Vector`/`sid`；整 tile 向量段，副块编号由自动调度合成（§1.6.3） |
| `dual_copy` 要求恰有一维 2:1 且 dst 不能是带单例中轴的 GM region（`copy_op.py:107-134`） | 是 | 块尾写回**不用** `dual_copy`（§3.5.4 反例 3），改用普通整 tile `T.copy`；`dual_copy` 只作 L0C→UB 的回退形态 |
| 原子存储支持 dtype 集合 float32/float16/bfloat16/int8/16/32（`mode.py`） | 是 | `dq_acc` 为 fp32，正好在支持集内 |
| 未认证项：手工跨核 flag、循环内复杂同步、跨 block barrier（programming.md） | 是 | 不写手工 flag，全部依赖默认 AutoSchedule + InsertSync；不引入跨 block barrier（§7） |
| 动态 shape | 部分 | `S_q/S_kv/B/H/D` 编译期常量（dense 工厂按 shape 编译）；`cu_seqlens_*` 只保留签名兼容，kernel 内用静态寻址（OOB clamp 依赖这些常量，见探针 B 的 `min(64, 96 - n0)`） |

#### 3.5.2 参考实现差异说明

| 差异项 | 参考实现（GPU TileLang） | 本项目（TileLang ascend） | 转换方案 |
|--------|------------------------|--------------------------|---------|
| Kernel 结构 | `T.Kernel(num_tiles, heads, threads=256)` 二维网格 + 线程块 | 一维核网格 + AIC/AIV 自动分派 | §0.6 R1/R4 |
| 矩阵 API | `T.gemm(fragment)`，C 在寄存器，支持 `transpose_A` | `T.gemm(A_l1, B_l1, C_l0c)`，仅 NT，全 extent | §0.6 R2/R3 |
| 累加器 | `T.alloc_fragment` | `T.alloc_l0c` | §0.6 R2 |
| 片上层级 | SMEM + register file | L1（操作数）+ L0C（累加）+ UB（向量） | §4.4 |
| 原子累加 | `T.atomic_add` 逐元素 | `set_atomic` + 整块 store | §0.6 R5 |
| 指数 | `T.exp2`（log2 域） | `T.exp`（自然域） | §1.6.1 A1 |
| 掩码 | 逐元素 `if_then_else` | 同 + 运行时块级特化 | §1.6.1 A2 |
| 流水 | `T.Pipelined(num_stages=2)`（cp.async） | `annotate_buffer_versions` + `num_stages`（首版 1） | §6.3 |
| 尾块 | `T.if_then_else` 逐元素守卫 + 区段截断 | 全 extent + 平台 OOB clamp/补 0 + 掩码置 0 | §5.4 |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/TileOPs/tileops/kernels/attention/mla_dense_bwd/_mla_dense_bwd_kernels.py` | 高（同一函数的 GPU 原样抽取） | 语义与分块结构、`dk` 的 `scale`、掩码谓词、原子写回 |
| `.agents/platforms/toolchains/ascend/knowledge/repro/PL-950-mixed-epilogue.py` | 高（Cube GEMM + Vector + `dual_copy` 同 kernel） | 混合模式写法、L0C→UB 落地、同 dtype 约束 |
| `.agents/platforms/toolchains/ascend/knowledge/repro/PL-950-gemm-tiles.py` | 高（GM→L1→GEMM、转置装载、动态 region、多版本） | `transpose=True` 的 dn2nz、`clear_accum` 首块清零、`T.Pipelined` |
| `.agents/platforms/toolchains/ascend/knowledge/repro/PL-950-atomic-splitk.py` | 中（store-atomic 累加） | `set_atomic("add","float32")` + `T.copy(C)` 的配对与清零契约 |
| `.agents/platforms/toolchains/ascend/knowledge/repro/PL-950-conversion-layout.py` | 中（SIMT 转换 + 转置写 + 列广播） | `T.cast` 的 ±inf/次正规行为、转置写与广播的可行形态 |
| `examples/paged_cache_write/_paged_cache_write_kernel/DESIGN.md` | 中（同平台 SIMT 先例的设计文档范式） | 章节组织、分核三要素的写法、风险登记口径 |
| `.agents/platforms/toolchains/ascend/kernel_template.py` | 中（kernel 骨架） | 一维 `T.Kernel`、`T.SimtVF` 包逐元素、静态 `if tail` 尾块分支 |
| `examples/mla_dense_bwd/_bwd_kernel/history_version/design_probe_compile.py` | —（本设计的编译探针） | §3.5.4 三个正向探针 + 可复现命令 |

#### 3.5.4 编译探针证据（无设备，仅前端 + 编译）

脚本：`examples/mla_dense_bwd/_bwd_kernel/history_version/design_probe_compile.py`；执行（必须离开仓根，
否则仓内 dev root 的 `tilelang/` 会遮蔽 site-packages 的 0.1.15 私有 TVM）：

```bash
cd /tmp && ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python \
  /workspace/tilelang-mlir-ascend-950/examples/mla_dense_bwd/_bwd_kernel/history_version/design_probe_compile.py
```

实测输出（2026-11-23，toolchain `ascend-950pr-9579 / TileLang ascend 0.1.15`）：

| 探针 | 结构 | 实测结果 |
|------|------|---------|
| **A `A_primary`** | §3.3 的完整主选骨架：5 个全 extent NT GEMM + 整 tile L0C→UB + `SimtVF(128)` 整 tile epilogue（含 `ds_u[j,i]` 转置写）+ 3 个整 tile UB→L1 + store-atomic + 块尾 cast/写回；S=128 整除 | **COMPILE_OK**，源 13816 B；设备码含 `ascend_gemm_l1` ×5、`asc_copy_l0c2ub` ×4、`ascend_nd2nz_scatter` ×3、`asc_copy_l0c2gm`（在 `asc_set_atomic_add_float`/`asc_set_atomic_none` 之间）、`__mix__(1, 2)` 与 `int32_t sid = asc_get_sub_block_id()`（**自动合成**） |
| **B `B_tail_oob`** | S=96（非整除）：全 extent 装载/写回的越界尾部 + 动态 `clear_accum=(kq == loop_st)` | **COMPILE_OK**；设备码把 DMA 收窄成 `asc_copy_gm2l1_nd2nz(..., min(64, 96 - n0), ...)`、`asc_copy_ub2gm_align(..., min(64, 96 - n0), ...)`，并对尾部补 0：`asc_fill_l1(...)` ×3；GEMM 的 `clear_accum` 形参直接是运行期表达式 `(kq == (block_idx & 1))` |
| **C `C_a2_branch`** | A2 运行时块级分支（两支各一套整 tile epilogue） | **COMPILE_OK**；生成两个 SIMT VF（`simt_vf_0` 快路 / `simt_vf_1` 掩码臂）+ 运行时 `if (...)` 二选一，掩码在 VF 内按 thread 索引展开 |

**反例（不接受的设计形态，均在本机实测）**：

| 形态 | 实测错误（原文） |
|------|-----------------|
| `T.gemm(a1[:an, :], b1[:am, :], c0[:an, :am], transpose_B=True)`，`an/am` 符号 | `ValueError: T.gemm requires static tile dimensions, but M is symbolic: an` |
| `with T.Vector(vector=2) as sid:` 包 `dual_copy` + `SimtVF` + UB→L1 + 第二个 GEMM | `InternalError: Check failed: (!var_idmap_.count(v)) is false: Need input to be in SSA form dup sid` |
| 块尾 `T.dual_copy(out_k, dk[b, n0:n0+64, h, :])`（dst 带单例中轴） | `ValueError: Cannot infer dual_copy split direction: src shape [64, 192] vs dst shape [64, 1, 192]. Exactly one trailing dimension must be halved.` |

说明：以上为**编译级**证据，不含设备执行；运行期精度与性能仍待 Stage 3（前置探针清单见 §8.3）。

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `q` | `(T_q, H, 192)` | `float16 \| bfloat16` | `T_q = B*S_q`（dense packed 视图） |
| `k` | `(T_kv, H, 192)` | 同上 | `T_kv = B*S_kv` |
| `v` | `(T_kv, H, 128)` | 同上 | |
| `do` | `(T_q, H, 128)` | 同上 | |
| `lse` | `(B, H, S_q)` | `float32` | **自然对数** log-sum-exp；空行为 `-inf` |
| `delta` | `(B, H, S_q)` | `float32` | `rowsum(o*do)`，由 `_delta_kernel` 产出 |
| `cu_seqlens_q` | `(B+1,)` | `int32` | dense 下 `arange(0,(B+1)*S_q,S_q)`；本 kernel 只保留签名兼容 |
| `cu_seqlens_kv` | `(B+1,)` | `int32` | 同上（`S_kv`） |

（源算子签名另含 `o`：由同一次算子调用的 `_delta_kernel` launch 消费（`bwd.py:341`），**本 kernel 不读**，
故不进入本 kernel 的搬运路径，见 §0.2。）

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `dk` | `(T_kv, H, 192)` | `float16 \| bfloat16` | 与 `k` 同 shape；已乘 `scale` |
| `dv` | `(T_kv, H, 128)` | 同上 | 与 `v` 同 shape |
| `dq_acc` | `(T_q, H, 192)` | `float32` | **in-out**，原子累加；未乘 `scale`（由 `_dq_land_kernel` 处理） |

三个输出与输入同 layout（BSHD、尾轴连续），**无需任何转置或 permute**；`dk/dv` 的 shape 与 `k/v` 逐维相同，
`dq_acc` 与 `q` 同 shape（仅 dtype 为 fp32），故 host 侧只做 `view`/`zeros`。

### 4.3 中间缓冲区

| Buffer | Shape | dtype | 层级 | 用途 |
|--------|-------|-------|------|------|
| k_s | (64,192) | 输入 dtype | L1 | score GEMM 的 A，跨 query 循环常驻 |
| kT_s | (192,64) | 输入 dtype | L1 | dQ GEMM 的 B（key 转置副本，dn2nz 装载） |
| v_s | (64,128) | 输入 dtype | L1 | dP GEMM 的 A |
| q_s | (64,192) | 输入 dtype | L1 | score GEMM 的 B |
| qT_s | (192,64) | 输入 dtype | L1 | dK GEMM 的 B（query 转置副本） |
| do_s | (64,128) | 输入 dtype | L1 | dP GEMM 的 B |
| doT_s | (128,64) | 输入 dtype | L1 | dV GEMM 的 B（do 转置副本） |
| pT_s | (64,64) | 输入 dtype | L1 | dV GEMM 的 A（`P^T`，向量侧产出） |
| dsT_s | (64,64) | 输入 dtype | L1 | dK GEMM 的 A |
| ds_s | (64,64) | 输入 dtype | L1 | dQ GEMM 的 A（`dS` 转置副本） |
| st_c | (64,64) | float32 | L0C | `s^T` 累加器（每迭代清零） |
| dpt_c | (64,64) | float32 | L0C | `dP^T` 累加器（每迭代清零） |
| dq_c | (64,192) | float32 | L0C | `dQ` 累加器（每迭代清零） |
| dk_c | (64,192) | float32 | L0C | `dK` 累加器（跨 query 循环） |
| dv_c | (64,128) | float32 | L0C | `dV` 累加器（跨 query 循环） |
| st_u / dp_u | (64,64) | float32 | UB | L0C 整 tile 落地 + 向量输入 |
| p16_u / dsT_u | (64,64) | 输入 dtype | UB | 向量产出的 fp16 整 tile |
| ds_u | (64,64) | 输入 dtype | UB | `dS` 转置整 tile |
| lse_u / dlt_u | (64,) | float32 | UB | 行统计量 |
| tail_u | (64,192) | float32 | UB | 块尾 `dk_c` 中转 |
| tailv_u | (64,128) | float32 | UB | 块尾 `dv_c` 中转 |
| out_k / out_v | (64,192) / (64,128) | 输入 dtype | UB | 块尾 cast 结果（写 GM 前） |

### 4.4 内存搬运路径

```
GM ──(MTE2: nd2nz / dn2nz；越界尾部 clamp+fill 由 AscendInsertOOBPadding 生成)──▶ L1
     k_s, v_s, kT_s(常驻) | q_s, qT_s, do_s, doT_s(每迭代)
GM ──(MTE2)──▶ UB lse_u, dlt_u(每迭代，256 B 级)
L1 ──(MTE1: 由 L1-GEMM lowering 内部按 tile_k_sub 分块)──▶ L0A/L0B
L0A/L0B ──(Cube mad)──▶ L0C st_c, dpt_c, dq_c(每迭代) / dk_c, dv_c(跨迭代)
L0C ──(fixpipe, 整 tile T.copy, fp32 同 dtype)──▶ UB st_u/dp_u(64,64)
UB ──(SimtVF(128) 整 tile: exp / mask / mul / cast / 转置写)──▶ UB p16_u, dsT_u, ds_u
UB ──(InsertNd2Nz → ascend_nd2nz_scatter + post_copy)──▶ L1 pT_s, dsT_s, ds_s（整 tile）
L0C(dq_c) ──(fixpipe + store-atomic add, fp32)──▶ GM dq_acc
L0C(dk_c/dv_c) ──(整 tile T.copy)──▶ UB tail_u/tailv_u ──(cast)──▶ UB out_k/out_v ──(整 tile T.copy)──▶ GM dk/dv
```

无 GM 中转的中间块（`P/dP/dS` 全程留在 L0C/UB/L1），符合 §1.6.0 R2 的「不物化」结论；`CONST-store-fixpipe-gm-only`
（910B 上 `S/P` 类中间块必须 GM 往返）在 950 上被 L0C→UB 搬运化解（knowledge/repro/PL-950-mixed-epilogue.py 已核验）。

### 4.5 UB 内存预算（每 AIV，整 tile）

**UB** 容量 253952 B（248 KB）；块尾缓冲与每迭代缓冲不复用（保守上界）：

| Buffer | Shape | dtype | 大小 (Bytes) |
|--------|-------|-------|-------------|
| st_u | (64,64) | float32 | 16384 B |
| dp_u | (64,64) | float32 | 16384 B |
| p16_u | (64,64) | f16/bf16 | 8192 B |
| dsT_u | (64,64) | f16/bf16 | 8192 B |
| ds_u | (64,64) | f16/bf16 | 8192 B |
| lse_u + dlt_u | (64,) 各一 | float32 | 512 B |
| tail_u | (64,192) | float32 | 49152 B |
| tailv_u | (64,128) | float32 | 32768 B |
| out_k | (64,192) | f16/bf16 | 24576 B |
| out_v | (64,128) | f16/bf16 | 16384 B |
| 合计（不别名上界） | — | — | 180736 B |

其中每迭代活跃部分 = `st_u + dp_u + p16_u + dsT_u + ds_u + lse_u + dlt_u` = 57856 B，块尾活跃部分 =
`tail_u + tailv_u + out_k + out_v` = 122880 B。总占比 180736 / 253952 = **71.2%**，余量 73216 B。

- 上一版（半区 32×64）的 90624 B 已随阻塞问题 2 的整 tile 修订重算（检视复算给出「~176 KB / 71%」，
  与本表的 176.5 KiB / 71.2% 一致）。
- **Stage 4 流水约束**：若把每迭代组取 2 版本，上界 = 180736 + 57856 = 238592 B = **94.0%**，余量仅
  15360 B —— 判定为**不可行**，故 Stage 4 的多版本只应在 L1 侧展开（§4.6 的 278528 B / 53.1%），UB 保持
  单版本。该结论写入 §9.1 风险 #5。
- 编译侧旁证：探针 A 的设备码中 `buf_dyn_shmem` 的最大偏移用到 86208 个 half 元素（= 172416 B），
  低于 253952 B，说明本预算与实际分配同量级。

### 4.6 L1 与 L0C 预算

| 层 | Buffer 组合 | 合计 (Bytes) | 容量 (Bytes) | 占比 |
|----|------------|-------------|-------------|------|
| L1 | `k_s 24576 + v_s 16384 + kT_s 24576 + q_s 24576 + qT_s 24576 + do_s 16384 + doT_s 16384 + pT_s 8192 + dsT_s 8192 + ds_s 8192` | 172032（168 KiB） | 524288（512 KiB） | 32.8% |
| L1（Stage 4 双版本，7 个每迭代缓冲各 ×2） | +106496 | 278528（272 KiB） | 524288 | 53.1% |
| L0C | `dk_c 49152 + dv_c 32768 + st_c 16384 + dpt_c 16384 + dq_c 49152` | 163840（160 KiB） | 262144（256 KiB） | 62.5% |
| L0A/L0B | 不显式分配；L1-GEMM lowering 按 `2 阶段 × 32 KB/操作数` 自行分块（单块 `64×192×2 = 24576 ≤ 32768`） | — | 65536 各 | 由 lowering 保证 |

**L0 回退路径的 L0A/L0B 预算（检视复算口径，阻塞问题 1 的可选回退）**：若 L1-GEMM 在运行期失败、改走
显式 L0 路径（`_l0_gemm_call` 读 allocation 形状，region extent 可符号，`prove_equal` 可证），需自行
分配 L0A/L0B：`Nk=64, Mq=32` 下 5 个积的**唯一操作数形状并集**（保守，含 Mq=64 与 Mq=32 两种 M 形态）：

| 层 | 唯一形状 → 字节 | 合计 | 占 65536 B |
|----|----------------|------|-----------|
| L0A | (64,192) 24576、(64,128) 16384、(64,64) 8192、(32,64) 4096 | 53248（52 KiB） | 81.3% |
| L0B | (32,192) 12288、(32,128) 8192、(128,64) 16384、(192,64) 24576 | 61440（60 KiB） | **93.8%** |

结论与风险：L0B 余量仅 4096 B（6.2%），**只允许单版本**；若需要双缓冲则必须把 `Mq` 降到 16（此时
`(32,192)`/`(32,128)` 减半到 6144/4096，L0B 合计降到 53248 B）。该路径同时在 `Mq=32` 下 L0C 降到
122880 B（46.9%），但每迭代 L1 端口流量与 `dq` 原子流量上升 1.5–2×（§9.1 风险 #1）。若全部统一到
Mq=32 且不做 (64,64)/(128,64) 形态，L0B 可降到 53248 B（81.3%），但 `pT/dsT` 的 M 维必须同步降到 32。

### 4.7 动态轴

dense 路径下**无运行时动态轴**：`B/S_q/S_kv/H/D_qk/D_v` 全部是编译期常量（工厂按 shape 编译，
`@lru_cache` 级缓存，与源 `target_cache(maxsize=32)` 等价）。`T_q/T_kv` 的 packed 视图只是 host 侧
`view` 的元数据。变长是另一个算子（§0.7）。这一静态性同时是尾块 OOB clamp 的前提（设备码里出现的是
字面量 `96` 与 `64`，§3.5.4 探针 B）。

### 4.8 JIT 配置

```python
@tilelang.jit(out_idx=[], target="ascend")
def _bwd_func(block_n: int = 64, block_m: int = 64, num_stages: int = 1) -> Callable:
    ...
# 编译标志沿用源：-O3、-DENABLE_BF16；不开 fast-math 之外的非默认选项
# （快速数学：T.exp 走 SFU 近似已在精度验证范围内，见 §8.2）
```

`out_idx=[]`：`dk/dv` 与 `dq_acc` 都是调用方预分配的张量（`dq_acc` 必须由 host 先
`torch.zeros`——store-atomic 的累加契约，参照 PL-950-atomic-splitk 的调用方式）。

---

## 5. Tiling 策略

### 5.1 计算类型

**类型：混合（Cube + Vector）**。判定依据：5 个矩阵乘（每迭代 3.4M MAC）必须落 Cube；每迭代另有
4096 元素的 `exp/mask/cast/转置` 必须落 Vector；两者在同一 kernel 内交替（§2.2、§0.6 R2/R4）。

### 5.2 Block 划分

```python
block_n = 64   # key 方向（= dk/dv 累加器的 M，以及 dq 的 K）
block_m = 64   # query 方向（= dq 累加器的 M，以及 dv/dk 的 K）
# 逻辑任务数 = heads * B * ceil(S_kv / block_n)
```

选择理由：
- `block_n` 同时决定 L0C 里 `dk_c`（`192·block_n·4 B`）与 `dv_c`（`128·block_n·4 B`）的大小，
  二者跨整个 query 循环常驻，是 L0C 的主项：`block_n=64` 时 80 KiB，`block_n=128` 时 160 KiB
  （再加每迭代的 `st/dpt/dq` 就要逼近溢出，见 §5.3）。
- `block_m=64` 让 `dq_c = 64×192×4 = 48 KiB`；同时 `dv/dk` 的 K 维取 64，mad 的 K 分块不退化；
  整 tile 向量段（64×64）在 `T.SimtVF(threads=128)` 下每线程 32 个元素，向量宽度整除无尾 lane。
- 二者均为 16 的倍数（mad 分形 16×16），`D_qk=192 = 12×16`、`D_v=128 = 8×16` 整除。
- 每迭代 L1 端口字节 ≈ 279544（24.5 FLOP/字节，§1.6.0 分解表）；备选 `block_n=128, block_m=32` 提到
  26.0（−5.1%），但其 L0C 占用升到 84.4%，留作 Stage 4 裁决项（§1.6.3）。

### 5.3 约束分析

- **对齐**：`D_qk=192`、`D_v=128`、`block_n=block_m=64` 全为 16 的倍数；fp16/bf16 的 L1/L0 分形与
  mad 的 16×16 分块无尾块；`dq_acc` 尾轴 192 个 fp32 = 768 B（32 B 对齐 ✓）。
- **L0C**：单个 GEMM 累加器的容量校验 `M × N × sizeof(fp32) ≤ 262144`：最大者为
  `dk_c = 64 × 192 × 4 = 49152 B` ✓（其余 `dq_c 49152`、`dv_c 32768`、`st_c/dpt_c 16384`）；
  5 个累加器同时驻留共 **160 KiB / 256 KiB（62.5%）**。`block_n=128 + block_m=64` 会到 272 KiB → 溢出，
  故 `block_n=128` 只能配 `block_m=32`（216 KiB，84.4%）。
- **L1**：168 KiB 单版本（32.8%）；双版本 272 KiB（53.1%）。
- **UB**：176.5 KiB（71.2%）/ 248 KiB；双版本不可行（94.0%，§4.5）。
- **L0A/L0B**：L1 路径由 lowering 按 `tile_k_sub` 分块保证（每操作数每阶段 ≤ 32 KB）；L0 回退路径见
  §4.6（L0B 93.8%，余量 4 KiB）。

### 5.4 注意事项（非整除与边界）

> 统一原则：**GEMM 一律全 extent；动态 extent 只出现在 copy；copy 的越界尾部由平台 pass 改写为
> `min()` clamp + 补 0；补 0 的语义后果由掩码臂的 `gi < S_kv` / `gj < S_q` 收口。**

1. **尾 key 块**（`S_kv % 64 ≠ 0`）：`k_s/v_s/kT_s` 用全 extent 装载，越界行由 `AscendInsertOOBPadding`
   收窄 DMA 并补 0（设备码 `min(64, S_kv - n0)` + `asc_fill_l1`）。补 0 使 `s^T` 的这些行为 0，但
   `exp(0·scale − lse) = exp(−lse) ≠ 0`，故**必须**由掩码臂的 `n0+i < S_kv` 把 `P` 置 0（→ `dS=0` →
   对 `dV/dK/dQ` 贡献为 0）；`dk/dv` 写回时同一 clamp 保证不写出 `T_kv`（§3.5.4 探针 B）。
   数学上与源 `if_then_else` 补 0 等价，代价是尾块多算一个满 tile（占比 ≤ 1/ceil(S/64)）。
2. **尾 query 块**（`S_q % 64 ≠ 0`）：`q_s/do_s/qT_s/doT_s/lse_u/dlt_u` 同样全 extent 装载 + clamp/补 0；
   掩码臂保留 `m0+j < S_q`，使填充列的 `P=0`；`dq_c` 的原子 store 区段由同一 clamp 裁到 `S_q` 内
   （设备码 `min(64, S_q - m0)`），不会写到下一 request 的行。
3. **因果对角块**：只有对角及相邻块进入掩码臂，其余走 §1.6.1 A2 快路；`loop_st = max(0,(n0-shift)//64)`
   保证不遗漏可见 query 块（最坏多算一个全掩码块，`P=0` 无副作用）。
4. **`S_q < S_kv` 与 `S_kv < S_q`**：`shift` 可为正/负；后者会让头部 query 看不到任何 key（`lse=-inf`），
   此时掩码全假 → `P=0` → 梯度为 0，`exp(−inf)` 结果被 `if_then_else` 丢弃（不产生 NaN，已由 golden 的
   `nan_to_num` 语义对齐）。**快路的安全性**由 §1.6.1 的充要条件保证：满足快路的块不可能包含
   「看不到任何 key」的行（否则 `n0+63 ≤ m0+shift` 不成立）。
5. **`S` 极小**（`S < 64`）：全部走尾块路径；`grid = H*B*1`，所有 DMA 都被 clamp 到 `S` 行并补 0。
6. **`is_causal=False`**：`loop_st=0`，所有块都进入快路（仍受 clamped 尾块影响，见 1/2）。

### 5.5 分核策略（物理核数适配）

> 三要素标准：`.agents/skills/_shared/standards/core-split-strategy.md` §1。

- **① 逻辑核数**：`num_logical_kernels = heads × B × ceil(S_kv / block_n)`（每个逻辑块 = 一个 key 分块 ×
  一个 head，即一次 block 发射）：

  | workload | B | H | S_kv | `ceil(S_kv/64)` | 逻辑核数 |
  |----------|---|---|------|-----------------|---------|
  | mla-dense-bwd-smoke | 1 | 8 | 512 | 8 | **64** |
  | ds-v3-4k | 1 | 128 | 4096 | 64 | **8192** |
  | ds-v3-b2-8k | 2 | 128 | 8192 | 128 | **32768** |
  | kimi-k2-8k | 1 | 64 | 8192 | 128 | **8192** |
  | ds-v2-lite-3k | 4 | 16 | 3072 | 48 | **3072** |

- **② 物理核数依据（任务平台记录，非假设值）**：目标型号 `ascend-950pr-9579`（dav-3510）。
  平台 `statectl platform` 落库的实查记录（`core_source` 字段，2026-11-23）：
  `Ascend950PR_9579.ini [SoCInfo] ai_core_cnt=28, cube_core_cnt=28, vector_core_cnt=56;`
  `runtime probe rtGetAiCoreCount()=28`。平台 ruleset 要求的查询口径是
  `torch.npu.get_device_properties`（adapter `core_evidence()`），本机实测值同为此数
  （**28 Cube / 56 Vector**，见本任务上下文）。本次设计期该卡被其他算子占用（`ASCEND_RT_VISIBLE_DEVICES=""`，
  只做无卡静态校验），因此引用上述已落库探测记录，并在 Stage 3 首跑前用
  `torch.npu.get_device_properties` 复核型号与核数后再确认。
- **③ 规模判定与分核方案**：5 条 workload 的逻辑核数（64 … 32768）**全部 ≫ 物理核数 28**
  （最小 smoke 也是 64 ≈ 2.3 波），且无法通过调整分块把核数压到 28 的整数倍
  （`block_n` 上限受 L0C 约束，最大 128 → ds-v3-b2-8k 仍有 16384 个逻辑块）。判定为 **极大规模**。
  **分核方案：1 逻辑块 = 1 次 block 发射（逻辑块超发，由硬件块调度器按波次派发）**，不采用「固定启动
  内核数 = 物理核数 + 核内串行多任务」，依据：
  (a) 每逻辑块的计算量是 10²–10³ µs 级（ds-v3-b2-8k：14.3 TFLOP / 32768 块 ≈ 436 MFLOP/块），
  远大于 block 派发开销，摊薄启动开销的收益可忽略；
  (b) 因果注意力下块间工作量相差最多 64 倍（对角块 1 次迭代 vs 远端块 128 次），动态派发天然负载均衡，
  固定 28 核的静态切分会引入长尾；
  (c) 「核内串行」需要把 Cube+Vector 混合体包进任务循环，而平台明确未认证「循环内跨单元/跨核同步」
  （`.agents/platforms/toolchains/ascend/references/programming.md`「同步与缓冲版本」末段），
  风险高于收益。

  **规则偏离声明（检视建议 6 的闭环）**：本项**偏离** `core-split-strategy.md` §1 的「极大规模」标准分支
  （该分支要求「固定启动内核数 = 物理核数，核内分批处理多个逻辑块」），采用标准未列出的第四方案
  「逻辑块超发」。理由即 (a)(b)(c)。**复评触发条件**（满足任一条则按 `[DESIGN_LIMIT]` 流程回写并重估）：
  ① Stage 4 实测 block 派发/尾波占比 > 5% 的单 kernel 时长；② `num_stages` 提到 2 后出现派发瓶颈或
  核间负载方差 > 2×；③ 平台认证「循环内 Cube+Vector 交替 + 跨迭代同步」后，核内串行方案转为可选。
  另记：核内串行需用 `T.Pipelined` 包住 Cube+Vector 混合体，属 `programming.md` 未认证组合。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

> 逐元素计算已按 §1.6.2 全部向量化；下表只有 block 级 / tile 级调度循环，无逐元素标量循环。

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| 逻辑任务（head × request × key 块） | block 级并行 | `T.Kernel(grid)` 一维核网格 | 每个 block 处理一个 key 分块（key-stationary），§5.5 |
| query 块方向 | 迭代（动态界） | `T.Pipelined(loop_st, loop_ed, num_stages=1)` | 因果剪枝后界随 `n0` 变化；首版单阶段，Stage 4 提到 2 |
| 5 个矩阵乘 | tile 级 | `T.gemm(A_l1, B_l1, C_l0c, transpose_B=True, clear_accum=…)` | Cube；`clear_accum` 区分「每迭代清零」与「跨迭代累加」 |
| 逐元素 epilogue | 向量并行（整 tile） | `with T.SimtVF(threads=128): for i, j in T.Parallel(64, 64)` | lane = query 轴（§1.6.3）；A2 由运行时块级 `if` 生成两支 |
| L0C↔UB | tile 级搬运 | `T.copy(C_l0c, ub)`（整 tile） | 同 dtype、整 tile；`dual_copy` 2:1 仅作回退（§9.1 风险 #3） |
| 块尾 cast | 向量并行 | `T.Parallel(64, 192)` / `T.Parallel(64, 128)` | `dk` 乘 `scale` 后 cast |

### 6.2 循环伪代码

见 §3.3（同一份伪代码，此处只列骨架；**GEMM 全 extent、无 `T.Vector`/`sid`**）：

```python
with T.Kernel(H * B * ceildiv(S_KV, BLOCK_N)) as bid:      # 静态 grid，最多 32768 块（按波次派发）
    h, tile = bid // NUM_TILES, bid % NUM_TILES
    b, n0 = tile // NT_PER_B, (tile % NT_PER_B) * BLOCK_N
    <全 extent 装载 k_s/v_s/kT_s>                            # 每 launch 一次
    for kq in T.Pipelined(T.max(0, T.floordiv(n0 - SHIFT, BLOCK_M)), ceildiv(S_Q, BLOCK_M), num_stages=1):
        <全 extent 装载 q_s/qT_s/do_s/doT_s/lse_u/dlt_u>
        <2 个 NT GEMM：st_c, dpt_c>                            # A/B/C 全 extent
        <整 tile T.copy: st_c→st_u, dpt_c→dp_u>
        if <运行时块级谓词>:                                    # A2：两支各一套整 tile epilogue
            with T.SimtVF(threads=128): <快路：p=exp(...)，无 mask，写 p16_u/dsT_u/ds_u>
        else:
            with T.SimtVF(threads=128): <掩码臂：vis 三项，写 p16_u/dsT_u/ds_u>
        <整 tile T.copy: p16_u→pT_s, dsT_u→dsT_s, ds_u→ds_s>
        <3 个 NT GEMM：dv_c += , dk_c += , dq_c = >
        <set_atomic + T.copy(dq_c → dq_acc) + set_atomic_none>
    <块尾：整 tile T.copy dk_c/dv_c → UB, cast(+scale), 整 tile T.copy → GM dk/dv>
```

### 6.3 流水线优化

- 首版（Stage 3）**`num_stages=1`**，并对全部每迭代缓冲与 L0C 累加器声明单版本
  （`T.annotate_buffer_versions({q_s: 1, qT_s: 1, do_s: 1, doT_s: 1, pT_s: 1, dsT_s: 1, ds_s: 1,
  st_c: 1, dpt_c: 1, dq_c: 1})`）。理由：平台仅核验过「单 tile 混合」与「静态界两阶段 GEMM」，
  「多阶段混合循环 + 跨单元重叠」未认证（§3.4、§9.1 风险 #2）；先把正确性钉住。
- Stage 4 的流水路径：**只展开 L1 侧**（7 个每迭代缓冲取 2 版本 → 278528 B / 512 KiB = 53.1%）。
  UB **不展开**：整 tile 缓冲双版本会到 238592 B（94.0%，余量 15 KiB），判定不可行（§4.5）。
  实测若动态界 + 多阶段不被接受，则改成「固定 trip count = `ceildiv(S_q,64)` + 头部 `if kq >= loop_st`」，
  牺牲剪枝换流水。
- 不做 `T.annotate_unlimit_memory`（不解除容量约束，避免把溢出推到运行期）。

### 6.4 尾块处理

统一口径（与 §3.3 注 2、§5.4 一致，纠正 v0 的「编译期判定」表述）：

- **GEMM**：不参与尾块处理——A/B/C 一律 allocation 全 extent，M/N/K 编译期静态。代价是尾块多算一个
  满 tile（`≤ 1/ceil(S/64)` 的额外 FLOPs），换来「无符号 extent」这一硬约束的满足。
- **搬运**：GM→L1 / L1→GM / GM→UB 的越界尾部由平台 pass `AscendInsertOOBPadding` 在编译期改写成
  运行时 `min()` clamp，并对 GM→L1 的补 0 区段生成 `asc_fill_l1`（`pipeline.py` 该 pass 的职责描述 +
  §3.5.4 探针 B 的设备码）。这是**编译期生成 clamp 代码**，不是「编译期判定某个块是否有尾」。
- **掩码**：`gi < S_kv` / `gj < S_q` 两项**必须**保留在掩码臂中（补 0 不等于梯度 0，见 §5.4 第 1 条）。
  快路分支通过运行时谓词 `n0+63 < S_kv ∧ m0+63 < S_q ∧ n0+63 ≤ m0+shift` 保证两支等价（A2）。
- **分支归属**：A2 是**运行时块级分支**（`n0` 来自核索引、`m0` 来自循环变量），生成两支 SIMT VF；
  编译期可静态判定的只有「是否存在尾块」（`S_q % 64`、`S_kv % 64`），它只影响 clamp 表达式是否退化。

---

## 7. 同步策略

### 7.1 同步模式

**模式：自动同步。** 依据：平台 `ascend/pipeline.py` 的 `allow_autoschedule` 默认开启，流水线会依次跑
`AutoSchedule → AssignCore → PrepareMultiBuffer → ResolveCore → InsertSync → MaterializeMultiBuffer`，
由编译器在 AIC/AIV 与 MTE 队列之间插入事件；平台规则明确「手工跨核 flag 配对、循环复用下的复杂同步
仍未认证」，因此本设计**不写任何手工 `ascend_set_flag/wait_flag/ascend_pipe_barrier`**。

### 7.2 同步点说明

| 位置 | 同步方式 | 理由 |
|------|----------|------|
| GM→L1 / GM→UB 之后到消费 | 自动（InsertSync 插 MTE2→MTE1/M 事件） | 搬运与 GEMM/向量读的依赖 |
| `T.gemm` 之后到 `T.copy(L0C→UB)` | 自动（M→FIX 事件）；探针 A 设备码为 `asc_sync_wait(PIPE_M, PIPE_FIX, …)` | L0C 写后读 |
| `T.copy(L0C→UB)` 之后到 `SimtVF` | 自动（FIX→V 事件）；设备码为两次 `asc_sync_intra_arrive(PIPE_FIX, e)` 与 `asc_sync_intra_arrive(PIPE_FIX, e+16)`（两次 arrive 覆盖两个副块，`e` 由调度分配，探针 A 实测为 1/17），AIV 侧对应 `asc_sync_intra_wait(PIPE_V, …)` | UB 写后读，且需对两个 AIV 都可见 |
| `SimtVF` 之后到 `T.copy(UB→L1)` | 自动（V→MTE） | L1 写前读完成 |
| `UB→L1` 之后到下一个 `T.gemm`（A 操作数） | 自动（MTE→MTE1/M） | GEMM 读 L1 前必须写完 |
| `set_atomic` ↔ `T.copy(L0C→GM)` ↔ `set_atomic_none` | 硬件模式寄存器；探针 A 设备码里 `asc_set_atomic_add_float` / `asc_copy_l0c2gm` / `asc_set_atomic_none` 由调度器保序 | 原子模式必须在 store 前生效、之后关闭（PL-950-atomic-splitk 形态） |
| 跨 block（不同 key 块累加同一 `dq_acc`） | **无同步**（store-atomic 由硬件保证原子性） | 本设计不引入任何跨 block barrier |

### 7.3 pass_configs 配置

```python
# 保持默认：不关闭自动调度、不改共享内存复用策略
pass_configs = {}          # 即 TL_ENABLE_AUTO_SCHEDULE=True（默认）
compile_flags = ["-O3", "-DENABLE_BF16"]
```

若 Stage 3 出现同步相关失败，回退顺序是「先降 `num_stages` 到 1、再钉 `annotate_buffer_versions` 到 1」，
最后才考虑 `T.MixedKernel(N, sids=2)` + 显式 `T.Cube()/T.Vector()` + 手工 sync——**注意该形态在本工具链
与 `dual_copy`/UB→L1 混用时不可编译**（§3.5.4 反例 2），故它已不是本设计的可行回退，仅作记录。

---

## 8. 验证方案

### 8.1 Golden 函数

> 依据 §0.1 语义独立实现（移植源仓 `mla_bwd_reference` / `MLADenseBwdWorkload.ref_program`），
> **不复刻** §0.6 的分块算法：它物化概率 `P` 再用 `einsum` 反向，不重算任何块、不做原子累加，也不消费外部
> 传入的 `lse`/`delta`（两者都由它自己从 `q,k,v,do` 重算）。

```python
import torch

def golden_bwd_kernel(q, k, v, do, *, scale=None, is_causal=True, dtype=None, slab_heads=8):
    """_bwd_kernel 的 golden：fp32/fp64 物化参考，bottom-right 因果。

    q,k:   (B, S_q, H, D_qk)      v,do: (B, S_kv / S_q, H, D_v)
    返回 (dq, dk, dv) 与源 ref_program 同口径：中间 fp64，最后一次舍入到输入 dtype；
    另返回 dq_acc_true = dq_true / scale（供本 kernel 的 fp32 中间输出直接比对）。
    """
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    dtype = dtype or q.dtype
    B, S_q, H, D_qk = q.shape
    S_kv, D_v = k.shape[1], v.shape[-1]
    shift = S_kv - S_q
    qf, kf, vf, dof = (t.to(torch.float64) for t in (q, k, v, do))
    dq = torch.zeros(B, S_q, H, D_qk, dtype=torch.float64, device=q.device)
    dk = torch.zeros(B, S_kv, H, D_qk, dtype=torch.float64, device=q.device)
    dv = torch.zeros(B, S_kv, H, D_v, dtype=torch.float64, device=q.device)
    key_pos = torch.arange(S_kv, device=q.device)
    q_pos = torch.arange(S_q, device=q.device) + shift
    hidden = key_pos[None, :] > q_pos[:, None] if is_causal else None
    for b in range(B):
        for h0 in range(0, H, slab_heads):          # 按 head 分片，避免 O(H*S^2) 同时驻留
            hs = slice(h0, min(h0 + slab_heads, H))
            s = torch.einsum("shd,nhd->hsn", qf[b, :, hs], kf[b, :, hs]) * scale
            if hidden is not None:
                s = s.masked_fill(hidden[None], float("-inf"))
            lse = s.logsumexp(-1, keepdim=True)
            p = torch.exp(s - lse).nan_to_num(0.0)          # 全 -inf 行 → 概率 0
            dp = torch.einsum("shd,nhd->hsn", dof[b, :, hs], vf[b, :, hs])
            ds = p * (dp - (p * dp).sum(-1, keepdim=True))   # delta 由概率重算，独立于输入 delta
            dv[b, :, hs] += torch.einsum("hsn,shd->nhd", p, dof[b, :, hs])
            dk[b, :, hs] += torch.einsum("hsn,shd->nhd", ds, qf[b, :, hs])
            dq[b, :, hs] = torch.einsum("hsn,nhd->shd", ds, kf[b, :, hs])
    dq, dk = dq * scale, dk * scale
    dq_true, dk_true, dv_true = dq.to(dtype), dk.to(dtype), dv.to(dtype)
    dq_acc_true = (dq / scale).to(torch.float32)             # 本 kernel 的 dq_acc 期望值
    return dq_true, dk_true, dv_true, dq_acc_true
```

独立性说明：golden 只读 `q,k,v,do`（甚至不使用 `lse/delta`），概率与 `delta` 全部重算——与本设计的
分块/原子/转置结构无共享假设，`lse/delta` 的正确性由 `_delta_kernel` 的 L0 负责。

### 8.2 精度标准

| 输出 | dtype | atol | rtol | 依据 |
|------|-------|------|------|------|
| `dk`, `dv` | float16 | 5e-3 | 1e-3 | `reference_tolerance(float16)` + `MLADenseBwdWorkload.verification` 的 `max(5e-3, atol)` |
| `dk`, `dv` | bfloat16 | 5e-3 | 1.6e-2 | 同上（bf16 一行） |
| `dq_acc`（fp32 中间量，测试侧再乘 `scale` 比对 `dq`） | float32 | 5e-3 / `scale` ≈ 6.9e-2 | 1e-3 | `dq_acc = dq/scale`，故 atol 按 `scale = 192^-0.5` 反比放大 |
| `P/dS` 的公式等价性 | — | — | — | §1.6.1 D-1 已机器验证（1 ulp，违反率 0） |

补充口径：`dq_acc` 的累加顺序跨核不确定（store-atomic），因此**不做逐位比对**，只做 dtype 容差；
`zeroed_input(0)` 控制（源 `verification` 的 controls）在本 kernel 的 L0 用「把 `do` 置零 → 三个梯度全零」
覆盖。

### 8.3 L0 门槛测试计划

| 用例 | shape | is_causal | 目的 |
|------|-------|-----------|------|
| L0-1（门槛） | B=1, S_q=S_kv=512, H=8, D=192/128，bf16 与 fp16 各一次 | True | manifest smoke；5 个 GEMM 链 + 原子 + 转置副本全路径 |
| L0-2 | 同 L0-1 | False | 非因果（`loop_st=0`、A2 快路全开） |
| L0-3 | B=1, S_q=S_kv=96, H=2 | True | 尾块：`S % 64 = 32`（clamp + 补 0 + 掩码臂） |
| L0-4 | B=1, S_q=96, S_kv=192, H=2 | True | `shift>0`（query 落在后段）；`loop_st>0` |
| L0-5 | B=1, S_q=128, S_kv=64, H=2 | True | `shift<0`：头部 64 个 query 看不到任何 key（`lse=-inf`）→ `dq=0` 且无 NaN/inf |
| L0-6 | B=1, S_q=S_kv=64, H=1 | True | 单块单迭代（`clear_accum=(kq==loop_st)` 首块语义、单块快路） |
| L0-7 | B=1, S_q=S_kv=65, H=1 | True | 非 16 倍数边长（尾块 1 行；clamp 边界 `S-64=1`） |

L1/L2 与完整分层套件（`ds-v3-4k` / `ds-v3-b2-8k` / 多 dtype）交由 `tilelang-op-develop` 的 L1/L2/boundary；
此处只锁 L0 门槛。

**Stage 3 必须前置的最小探针清单**（按序，全部为无卡/单卡最小用例；编译级已由 §3.5.4 覆盖，
下列是**运行期**必须确认的量）：

| # | 探针 | 输入 | 判定 |
|---|------|------|------|
| P1 | **L0C→UB 整 tile 的双 AIV 可见性**：1 个 NT GEMM + `T.copy(L0C→UB)` + `SimtVF` 写 GM（探针 A 的 S=64 版本） | `S_q=S_kv=64` | 输出逐元素 == golden；若错位 → 切 §9.1 风险 #3 的回退形态（`dual_copy` 2:1 + 静态半区 buffer） |
| P2 | **OOB clamp 语义**：`S_kv=96` 的 `dk/dv` 写回不带脏行 | L0-3 | `dk` 的第 64–95 行正确、无越界写（用 `torch.full` 前置哨兵值检测） |
| P3 | **尾块掩码收口**：`n0+i ≥ S_kv` 的行贡献为 0（补 0 的 `exp(-lse)` 不得泄漏） | L0-3 | `dk/dv` 尾部行与 golden 在容差内 |
| P4 | **动态 `clear_accum`**：`kq == loop_st` 首块清零 | L0-4/L0-5 | 与「显式前置 `T.clear`」版本逐元素一致 |
| P5 | **store-atomic 契约**：`dq_acc` 由 host 预清零 | L0-1 | 与 `dq_acc` 未清零的对照组相比不翻倍 |
| P6 | **A2 两臂一致性**：强制 `is_causal=False`（全快路）与 `True`（混合）在同一块上的输出一致 | L0-1/L0-2 | 快路与掩码臂对同一 (key,query) 块逐元素相同 |

实施顺序：P1 → P2/P3 → P4 → P5 → P6 → 全量 L0-1…L0-7。

---

## 9. 风险点与注意事项

### 9.1 已知约束与风险（按优先级）

| # | 风险 | 证据/来源 | 影响 | 处置 / 回退 |
|---|------|----------|------|------------|
| 1 | **L1-GEMM 路径的运行期正确性**（A/B 在 L1、C 在 L0C）：编译级已走通，但 knowledge 的已核验 GEMM 用例走显式 L0 路径，运行期（尤其 5 个 GEMM + 两段向量交替）未实测 | §3.5.4 探针 A（编译级）；`.agents/platforms/toolchains/ascend/knowledge/PL-950-cube-tile.md`、repro/PL-950-gemm-tiles.py（L0 路径已核验） | 整个 kernel 的 5 个矩阵乘都依赖它 | Stage 3 首步跑 P1/P4；若运行期失败 → 显式 L0A/L0B 路径 + 缩小分块（`Nk=64, Mq=32`）：**L0A 53248 B（81.3%）、L0B 61440 B（93.8%，余量 4 KiB，只允许单版本）**，L0C 降到 122880 B（46.9%），代价是每迭代 L1 端口流量与 `dq` 原子流量上升 1.5–2×（逐 buffer 表见 §4.6）。该回退的容量风险：L0B 余量 < 4 KiB，任何额外缓冲即溢出 |
| 2 | **多阶段混合（Cube+Vector 交替 5 GEMM + 2 向量阶段，循环内）未核验**：已核验只有「单 tile 混合 epilogue」「静态界两阶段 GEMM」 | PL-950-mixed-epilogue、gemm-tiles 的适用边界段 | 编译失败或同步错乱 | 首版 `num_stages=1` + 版本钉死；Stage 4 只在 L1 侧提双版本（UB 双版本 94.0% 不可行，§4.5） |
| 3 | **整 tile L0C→UB 的双 AIV 数据可见性未实测**：设备码里 AIC 发一次 `asc_copy_l0c2ub(..., dual_dst_mode=0)`，随后两次 `asc_sync_intra_arrive(PIPE_FIX, e)` / `(…, e+16)` 通知两个副块；若该 fixpipe 只写一个 AIV 的 UB，则另一 AIV 会读到旧数据 | §3.5.4 探针 A 设备码（同步点覆盖两个副块）；`copy.cc` 的 `kL0CToUB` 路径含 `sid`/`dual_dst_ctl` 形参 | 向量段输出错（整块） | Stage 3 P1 验证；回退：`dual_copy` 2:1 分片（L0C→两个半区 UB，各 (32,64)）+ 两个**静态**半区 buffer（`pT_lo/pT_hi`）+ 两段 `T.SimtVF(32,64)` + `T.copy` 到 `pT_s[0:32,:]`/`pT_s[32:64,:]`——偏移是编译期常量，**不需要 `sid`**；GEMM 仍保持整 tile（v0 的「拆 `pT_lo/pT_hi` + AIC 侧半区 GEMM」不必再要，半区只出现在 UB 侧）。该回退的切片 dst copy 形态须在 Stage 3 一并验证（v0 的 sid 方案已证明不可编译，不再作为回退） |
| 4 | **跨步 GM region 搬运**：`q[b*S_q+m0 : …, h, :]` 是 (T,H,D) 张量的固定-head 切片（行 stride = H*D），nd2nz/dn2nz 与 fixpipe store 的跨步支持 | 编译级已证（v0 检视探针 `probe_strided`：跨步 region + `transpose=True` + `set_atomic` + L0C→跨步 GM 均 COMPILE_OK；本轮探针 A/B 的 `min()` clamp 也出现在跨步路径上） | 装载/写回错位 | P2 用哨兵值检测越界写；回退：GM→UB（逐行 copy）→UB→L1，代价是每迭代多一次 UB 往返（+80 KB/迭代的 UB 流量） |
| 5 | **UB 双版本不可行**（整 tile 化后 238592 B = 94.0%） | §4.5 复算 | Stage 4 若强行双版本 → 编译期容量报错 | Stage 4 的多版本只落在 L1（278528 B / 53.1%）；UB 保持单版本；必要时把 `st_u/dp_u` 与块尾 `tail_u` 显式复用（编译器已做部分别名，探针 A 实际用到 172416 B） |
| 6 | `dq` 原子流量 ≈ 103.9 GB（L2 驻留假设，§1.6.0）；若 L2 未命中则退化为 HBM 带宽瓶颈 | 结构计算 + L2 = 128 MiB（硬件档案） | 时延高于估算 | Stage 4 提 `block_n`（§1.6.3 备选，原子流量减半到 52.3 GB） |
| 7 | `clear_accum=(kq == loop_st)` 是动态布尔 | 编译级已验证（探针 B：`ascend_gemm_l1<...>(..., (kq == (block_idx & 1)), 0, 0)`）；**运行期语义未实测** | 首块未清零 → 结果叠加脏值 | Stage 3 P4：与「把首迭代提出循环 + 显式 `T.clear(dk_c)`」的版本逐元素对照 |
| 8 | `T.exp` 为 SFU 近似 + fast-math | PL-950-softmax-boundaries 在 fp32 上以 1e-6 级误差通过 | bf16/fp16 输出超差 | 已在 §8.2 容差内预期；若超差，关闭 fast-math 或改用 `T.exp` 的 fp32 高精路径后复测 |
| 9 | `dq_acc` 依赖 host 预清零（store-atomic 累加） | `mode.py:set_atomic` 语义 + PL-950-atomic-splitk 的调用契约 | 结果翻倍 | wrapper 必须 `torch.zeros`（现有三 launch 序列已如此）；在设计交付说明中显式标注（P5 检查） |
| 10 | `lse` 域误用（把 log2 域当自然对数传入）会静默错 | 源算子显式乘 `LOG2E`，本设计不再乘 | 全量精度错 | 接口层不做任何 log/ln 换算；golden 对照会在 L0-1 立即暴露 |
| 11 | A2 的两支 VF 代码体积与编译时间 | 探针 C 源文本 4.3 KB（两臂） | 大 shape 下编译时间增加 | 可接受；若编译时间成为问题，退化为单臂（带掩码，−49.5% 的快路收益） |

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|---------|
| UB/L0C 溢出 | 分块放大或双版本 | 编译失败 | 按 §4.5/§4.6 预算回退分块；UB 不双版本 |
| `T.Kernel` 传 `threads=` | 照搬 GPU 源码 | Python 级拒绝 | 线程域只写在 `T.SimtVF` 内 |
| `T.gemm` 用符号 region extent（`x[:an, :]`） | 想用区段截断处理尾块 | `ValueError: T.gemm requires static tile dimensions` | GEMM 全 extent + 掩码臂 + 平台 OOB clamp（§3.5.4 反例 1） |
| 显式 `T.Vector(vector=2) as sid` + `dual_copy`/UB→L1/GEMM | 想手工按副块分片 | `InternalError … Need input to be in SSA form dup sid` | 不写 `T.Vector`；整 tile 向量段（§3.5.4 反例 2） |
| 块尾用 `T.dual_copy` 写 GM | dst 是带单例中轴的 GM region | `ValueError: Cannot infer dual_copy split direction` | 块尾用普通整 tile `T.copy`（§3.5.4 反例 3） |
| 用 `transpose_A=True` 走 L1 GEMM | 照搬 GPU 的 dQ | 断言失败 | 全部 NT（§0.6 R3） |
| L0C→UB 直接 cast 到 fp16 | 想省一次 UB | 前端显式拒绝（dtype 不同） | UB 内 `T.cast` |
| 手工 `set_flag/wait_flag` 配错 | 关掉自动调度后自写同步 | 死锁 / 数据竞争 | 不写手工同步（§7） |
| `dq_acc` 未清零 | 复用 buffer | 梯度翻倍 | host `torch.zeros` |
| 认为「补 0 就等于梯度为 0」 | 尾块只靠 OOB fill | `exp(-lse) ≠ 0` → 尾部行出现伪梯度 | 掩码臂必须保留 `gi < S_kv` / `gj < S_q`（§5.4） |

### 9.3 特殊场景处理

非整除（§5.4）、极小 shape（§5.4 第 5 条）、`is_causal=False`（`loop_st=0`，掩码整体省为 A2 快路）、
`D_qk=192` 非 2 的幂（仅影响 L1 缓冲形状，不影响分形，`192 = 12×16`）、`S_q≠S_kv`（`shift` 正负两种情况
已列 L0-4/L0-5）。

---

## 10. 交付清单

### 10.1 目录结构

```
examples/mla_dense_bwd/_bwd_kernel/
├── DESIGN.md                       # 本设计文档
├── verify_equiv.py                 # §1.6.1 等价性机器验证（D-1，可重跑）
├── RETROSPECTIVE.md                # Stage 1 复盘（逐次追加）
├── history_version/
│   ├── design_v0.md                # 上一版（revision 输入）
│   └── design_probe_compile.py     # §3.5.4 编译探针（可重跑）
└── _bwd_kernel.py                  # 算子实现（Stage 3 产出，不在本阶段）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | 已完成（revision v1） | 设计文档（含 §0 迁移分析、§1.6 算法调研与优化、§11 回退说明） |
| `verify_equiv.py` | 已完成（2026-11-23 重跑，3/3 EQUIV_PASS） | 等价性机器验证脚本 |
| `history_version/design_probe_compile.py` | 已完成（3/3 COMPILE_OK） | §3.5.4 编译探针 |
| `_bwd_kernel.py` | 待实现（Stage 3） | kernel + golden（§8.1）+ L0 套件（§8.3） |

### 10.3 命名规范

- 项目目录：`examples/mla_dense_bwd/`（TileOPs 迁移项目，按算子名聚合）
- 算子目录：`_bwd_kernel`（对应源工厂函数名；同项目下还有 `_delta_kernel`、`_dq_land_kernel`）
- 实现文件：`_bwd_kernel.py`

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md）+ 等价性验证（verify_equiv.py）+ 编译探针（design_probe_compile.py）
2. ⬜ Stage 3 P1：运行期最小混合链（1 GEMM + 整 tile L0C→UB + SimtVF + UB→L1 + 第二个 GEMM，S=64）
3. ⬜ P2/P3：尾块 clamp 与掩码收口（S=96）
4. ⬜ golden（§8.1）+ L0-6 单块全链（先 `dk/dv`）
5. ⬜ P4/P5：动态 `clear_accum` 与 store-atomic 契约 → L0-1（bf16/fp16）+ L0-2
6. ⬜ P6 + 尾块与 shift 边界（L0-3/4/5/7）
7. ⬜ L1/L2 与性能采集（交 Stage 3 / Stage 4）

---

## 11. 回退说明与检视问题闭环（revision v1）

### 11.1 相对上一版的关键调整

| # | 维度 | v0 的设计选择 | v1 的调整 | 驱动依据 |
|---|------|--------------|----------|---------|
| 1 | GEMM 维度 | 5 个 `T.gemm` 用**动态 region extent**（`k_s[:an,:]`、`q_s[:am,:]`、`c0[:an,:am]`、块尾 `kr`）实现尾块 | **全部改为 allocation 全 extent**，M/N/K 编译期静态；尾块改由「全 extent 装载 + 平台 OOB clamp/补 0 + 掩码臂置 0」处理 | 检视阻塞 1：`ValueError: T.gemm requires static tile dimensions, but M is symbolic: an`（`gemm_op.py:89-104`，M/N/K 取自 region extent）；本轮复现并给出可编译替代（§3.5.4 探针 A/B） |
| 2 | 向量执行体 | `with T.Vector(vector=2) as sid` 包住 epilogue，两个 AIV 各持 key 半区（32,64），`sid` 参与掩码与写回区段 | **删除 `T.Vector`/`sid`**；整 tile（64,64）向量段 + `T.SimtVF(threads=128)`；副块 id 由自动调度合成（设备码 `int32_t sid = asc_get_sub_block_id()`），两 AIV 冗余执行同一整 tile | 检视阻塞 2：`InternalError … Need input to be in SSA form dup sid`；本轮确认该形态在 `T.Kernel` 与 `T.MixedKernel(1, sids=2)` 两条路径下均失败，而整 tile 形态 COMPILE_OK |
| 3 | 块尾写回 | `T.dual_copy(out_k[:kr,:], dk[b, n0+sid*32:, h, :])`（半区源 + sid 偏移 + 动态 `kr`） | 整 tile `T.copy(out_k, dk[b*S_kv+n0 : +64, h, :])`，越界行由 OOB clamp 裁剪 | 检视建议 3 + 本轮实测反例：`dual_copy` 的 dst 带单例中轴时报 `Cannot infer dual_copy split direction`；且 sid 偏移会与 2:1 推断叠加 |
| 4 | UB 预算 | 半区 32×64（每迭代 29184 B，总 90624 B） | 整 tile 64×64（每迭代 57856 B，总 180736 B = 71.2%） | 整 tile 化的必然结果；复算见 §4.5 |
| 5 | A2 机制 | 声称「Python 级 `if` 用静态量在编译期判空」 | 改为**运行时块级分支**（两支各一套 epilogue，编译成两个 SIMT VF） | 检视建议 4：`n0=f(bid)`、`m0=kq·64` 都是运行时量；探针 C 证实可编译 |
| 6 | L1 端口口径 | 每迭代 248 KB / 27 FLOP/字节 | 逐项分解表：**279544 B / 24.5 FLOP/字节**（全含口径），全量 590.6 GB | 检视建议 7：v0 的 248 KB 落在「不含 GM→L1」与「全含」之间不可复现 |
| 7 | 备选方案量化 | 备选 128/32「−15% L1 端口」 | 按统一口径重算：**263164 B / 26.0 FLOP/字节（−5.1%）**、GM 视图 143.0 GB、L0C 84.4%；并指出其块尾必须分两段写回（否则 UB 303360 B 超容量） | 检视建议 7 的连带项 |
| 8 | L0 回退预算 | 「L0B 恰好 64 KB、L0A 40 KB」 | 逐 buffer 表：**L0A 53248 B（81.3%）、L0B 61440 B（93.8%，余量 4 KiB）**，并登记容量风险 + `Mq=16` 的双版本条件 | 检视建议 5（复算不可复现） |
| 9 | §0.4 #13 | 「`T.clear` + `clear_accum` 两种累加初始化（源码手段）」 | 源码只有 `T.clear`；`clear_accum` 标注为 **NPU 适配构造（非源码手段）** | 检视建议 1（源码 `bwd.py:200-201,225,227,250` 无该参数） |
| 10 | §3.4 证据 | 引 `dual_copy` 的 docstring 当作 `T.copy` UB→L1 的证据 | 改引 `pipeline.py:88-91` 的 `InsertNd2Nz` + 探针 A 设备码的 3 个 `ascend_nd2nz_scatter` | 检视建议 2 |
| 11 | §5.5③ | 选「逻辑块超发」但未标注规则偏离 | 显式标注**偏离** `core-split-strategy.md` §1 的极大规模分支，并写 3 条复评触发条件与 `[DESIGN_LIMIT]` 流程 | 检视建议 8 |
| 12 | §0.2 I/O | 未列 `o` | 补 `o` 行并注明「仅被 `_delta_kernel` 消费，本 kernel 不读」 | 检视建议 10 |

### 11.2 为何不会再犯同一错误

- **「静态 tile 维」这条约束的判据被写进 §3.5.1 并附源码定位（`gemm_op.py:89-104`）**，同时给出设备的
  正/反例对照（§3.5.4）。v0 的错误根因是把 `_l0_gemm_call` 的「读 allocation 形状」推广到了 L1 路径；
  v1 在文档里区分了两条路径的取材来源，后续任何 GEMM 形态改动都必须同时通过「探针 A 形态」的编译。
- **凡涉及平台形态可行性，v1 一律以本机可复现探针为前置**：3 个正向探针 + 3 条反例落盘为
  `history_version/design_probe_compile.py`（一条命令可重跑），而不是引用 docstring 或类推。v0 的 sid
  方案与 UB→L1 证据都属于「未验证的类比」。
- **尾块策略改为单一原则**（GEMM 全 extent / 动态 extent 只在 copy / clamp 由平台 pass 生成 / 掩码收口），
  并在 §3.3 注 2、§5.4、§6.4 三处交叉引用同一口径，消除 v0 里「§5.4 说截断、§6.4 说编译期判定」的
  自相矛盾。
- **预算表统一到「整 tile 形态 + 全含口径」**，并让 §4.5 的 UB 表可被机械复核（`design_calc_check` 的
  `ub_budget` 现已解析本表并给出 `pass`：10 个 buffer 行 + 合计行 = 176.5 KB / 248 KB），避免下一次出现
  口径漂移。
- **回退阶梯里的每一步都必须本身可编译**：v1 把「`T.MixedKernel` + 手工 sync」从可行回退降级为记录项，
  把「`dual_copy` 2:1 + 静态半区 buffer」立为唯一回退，因为后者不含任何 `sid`。

### 11.3 检视问题逐项闭环

| 检视编号 | 严重度 | 处理 | 落点 |
|---|---|---|---|
| 问题 1（动态 GEMM extent） | **阻塞** | 已修：全 extent + OOB clamp + 掩码；探针 A/B COMPILE_OK，反例与原文并列 | §3.3、§3.5.1、§3.5.4、§5.4、§6.4、§9.2 |
| 问题 2（`T.Vector`/sid） | **阻塞** | 已修：删除 sid 机制，整 tile epilogue；设备码确认副块 id 由调度器合成；R4/§1.6.2/§1.6.3/§4.5/§9.1 同步重写 | §0.6 R4、§1.6.2、§1.6.3、§2.3、§3.3、§4.5、§9.1 风险 3 |
| 建议 1（`clear_accum` 归属） | 建议 | 已改：标为 NPU 适配构造并补源码定位 | §0.4 #13、§0.5、§3.2 |
| 建议 2（§3.4 证据误引） | 建议 | 已改：引 `InsertNd2Nz` + 探针设备码 | §3.4 |
| 建议 3（块尾 `dual_copy` 的 sid 偏移） | 建议 | 已删该形态（改为整 tile `T.copy`），并给出实测反例说明为何不能保留 | §3.3、§3.5.4 反例 3、§9.2 |
| 建议 4（A2 判空机制） | 建议 | 已改：运行时块级分支 + 快路安全性论证 | §1.6.1、§3.3、§3.5.4 探针 C、§6.4 |
| 建议 5（L0 回退预算） | 建议 | 已重写：L0A 52 KiB / L0B 60 KiB（93.8%）+ 余量与 `Mq=16` 条件 + 容量风险 | §4.6、§9.1 风险 1 |
| 建议 6（§5.5③ 偏离未标注） | 建议 | 已标注偏离 + 复评触发条件 | §5.5③ |
| 建议 7（L1 端口口径） | 建议 | 已补逐项分解表并重算主选/备选 | §1.6.0、§1.6.3、§5.2 |
| 建议 8（§0.2 缺 `o`） | 建议 | 已补 | §0.2、§4.1 |

### 11.4 保持不变的部分（本版未改动）

- 算法族：key-stationary 单遍重算（C1）；5 个矩阵乘、bottom-right 因果、`lse` 自然对数、`dq_acc` fp32 →
  `_dq_land_kernel` 缩放、`D_qk=192`、`D_v=128`。
- A1（自然域 exp）与 A2（块级 mask 特化）的等价结论与机器验证结果（3/3 EQUIV_PASS，本轮重跑一致）。
- 编程模式 `hybrid`（有证据偏离 `simd`）。
- §0 的三问解读与四态处置统计（仅 §0.4 #13 的归属与 §0.6 R2/R3/R4 的实现形态按修正更新）。
- §5.5 分核三要素（逻辑核数表、物理核数实查记录、规模判定与方案）。
- §8.1 golden 的独立性（只读 `q,k,v,do`，重算概率与 delta）与 §8.3 的 L0 计划（本轮增补前置探针清单）。
