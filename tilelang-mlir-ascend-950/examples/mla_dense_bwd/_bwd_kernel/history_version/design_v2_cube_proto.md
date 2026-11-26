# MLA prefill 反向主 pass（_bwd_kernel）算子设计文档

> 迁移任务（harness 子模式）。源算子：`/workspace/TileOPs/src/tileops/_kernels/attention/mla/bwd.py` 中的
> `_bwd_kernel`（GPU TileLang 实现，`MLADenseBwdKernel` / `MLAVarlenBwdKernel` 共用工厂）。
> 目标平台：`ascend-950pr-9579`（dav-3510，CANN 9.2.1，TileLang ascend 0.1.15），编程模式 **hybrid**
> （Cube 显式 L0 路径 + Vector AIV），偏离理由见 §2。
> 本函数是 `MLADenseBwdOp` 三次 launch 的第二个（`_delta_kernel` → **`_bwd_kernel`** → `_dq_land_kernel`）；
> 跨函数接口契约为：`delta` fp32 `[B,H,S_q]`、`dq_acc` fp32 `[T_q,H,D_qk]`（本函数原地累加，`_dq_land_kernel`
> 统一乘 `scale` 并 cast）。
>
> **本版为 revision v2（降 Vector 依赖 / 提 Cube 占比）**。v1 的设计与实现（`_bwd_kernel.py`，L0 13/13 通过）
> 实测为 **Vector 受限**：`MLADenseBwdOp` 主导 case `ds-v3-b2-8k` 的 **T_op 455.3 ms、cube 8.3% / vector 73.5%**。
> 本版用三个可复现探针/原型定位根因（§1.6.4 实测表），把结构改成「**5 个 GEMM 全走显式 L0 路径 + 向量 epilogue 全合并写
> + tile 64→96**」，并用端到端原型实测新结构：**3.46 µs/迭代（v1 6.03 µs）→ 外推 175 ms（2.6×）**，
> cube 占比 8.3% → 25%（估算）。相对 v1 的逐项调整与「为何不会再犯」见 §11。

---

## 0. 源算子解读与迁移分析

> 方法论：tilelang-op-design skill 的 references/migration-analysis.md（三问解读 → 耦合性四态 → 重设计）。
> 本章结论驱动 §1–§10。**v2 只改判定所依据的证据与结论（§0.5/§0.6），不改 §0.1–§0.3 的语义解读。**

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

**小结（第一问 What）**：语义 = 「用给定 `lse`/`delta` 重算 `P`、`dP`、`dS`，产出 `dV`、`dK`（含 `scale`）与
`dQ` 的 fp32 累加」；累加顺序只在 `dK/dV` 上固定为 query 块升序，`dQ` 跨核顺序不确定；中间累加全 fp32，GEMM 输入为输入 dtype；`lse=-inf` 的空行为零梯度且不得产生 NaN；`scale` 只在 `dK` 写回与 `dQ` 的后继 launch 中生效。

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
| S1 | 块解码：`request = tile // ceil(S_kv/block_n)`、`n0 = (tile % …)*block_n`（packed 走 `GroupTiling.decode`） | `tile`, `cu_seqlens_kv` | `b, n0, len_q, len_kv, shift` | —（任务映射） |
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
| S16 | 循环结束：`dk = cast(dk_f*scale)`、`dv = cast(dv_f)`（仅有效 key 行） | `dk_f, dv_f` | `dk`, `dv` | `dk` 的 scale 与输出 dtype |

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

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性判定（v2 复核） |
|---|----------|------|------|-----------------|---------------|
| 1 | key-stationary 分块（S2/S3/S16） | `k/v` 只搬一次、`dK/dV` 只在块尾写一次 | 一块 key 常驻 SMEM，`dk_f/dv_f` 跨 query 循环寄存器累加 | SMEM + 大寄存器堆 | 硬件强相关（结构可移植）→ **保留**，累加器迁 L0C（§0.6 R2） |
| 2 | recompute 而非物化 `P/dS`（S6/S8/S11） | 消 O(S²) 中间缓冲与 GM 往返 | 每个 query 块重算 `s` 与 `dP` | 无（纯算法层） | **保留**（§1.6.0 R2） |
| 3 | 因果块级剪枝（S4） | 少算约一半 (key 块, query 块) 对 | `loop_st` 从可见性下界起算 | 无（纯算法层） | **保留** |
| 4 | tensor core GEMM（S6/S7/S10/S12/S14） | 矩阵算力 | `T.gemm` → mma | tensor core | 有 Cube 等价 → **等价替换**（显式 L0 路径，§0.6 R6） |
| 5 | 寄存器 fragment 作累加器（S3/S10/S12） | 免 SMEM 往返 | fp32 累加 | 大寄存器堆 | Cube 侧只能 L0C → **重新设计**（§0.6 R2） |
| 6 | **`transpose_A=True` 复用同一份 `ds`**（S14） | 省一份 `dS` 的 SMEM 副本；`ds` 同时服侍 `dK`(作 A) 与 `dQ`(作转置 A) | mma 的 A 转置控制 | **v2 更正：可保留** —— 显式 L0 路径支持 `trans_A=True`（`gemm_mad.py:126-134` 的 MN-major 布局推断），本版探针 2 实测编译 + 数值 PASS；L0C 侧由 `asc_copy_l12l0a_transpose` 承担转置（MTE1，不进向量管线） | **保留（v1 误判为重新设计）** |
| 7 | `exp2` + `scale*LOG2E` 折入指数（S8） | 用便宜底 2 指数指令 | `exp(x)=exp2(x*log2e)` | SFU 上 `exp2` 快于 `exp` | **等价替换**（自然域 `exp`，§1.6.1 A1） |
| 8 | mask 用 `if_then_else` 选择而非乘掩码（S8） | mask 位置严格为 0，`lse=-inf` 行不产生 NaN | select | 无 | **等价替换**（向量 select + 运行时块级特化，§1.6.1 A2） |
| 9 | `T.Pipelined(num_stages=2)`（S5 装载 / 计算重叠） | 消 MTE 停顿 | 双 buffer 轮转 | 异步拷贝引擎 | **等价替换**（`annotate_buffer_versions` + `num_stages`）；**v2 实测：本平台多版本对本结构零收益**（§1.6.4） |
| 10 | `T.atomic_add` 写 `dq_acc`（S15） | 免 workspace / 免跨 block 归约 | 全局 fp32 原子加 | 全局原子 ALU + L2 | **等价替换**（store-atomic） |
| 11 | `lse/delta` 行广播（S8/S11） | 每行一次标量代替逐元素重算 | SMEM 标量读 | — | **保留**（lane 私有读） |
| 12 | warp 划分 `FullRow` + `threads=256`（S6–S14） | 匹配 mma 的 warp 分形 | warp-level 分片 | warp 32 线程 + 寄存器分片 | **重新设计**（§0.6 R4） |
| 13 | **（NPU 适配构造，非源码手段）** `clear_accum` | 表达「首块清零 / 跨块累加」 | mma 零累加位 | — | — |
| 14 | packed/dense 双分支 `GroupTiling`（S1） | 一个工厂服务两种 layout | 编译期分支 | 无 | **舍弃（本函数）**，见 §0.7 |

**关于 #13 的口径**：源码**没有** `clear_accum` 参数，只用 `T.clear`（`bwd.py:200-201, 225, 227, 250`）；
`clear_accum` 是**本设计的 NPU 侧构造**（§3.2 步骤 4），用于让 L0C 累加器表达同一语义。

**小结（第三问 Why fast）**：共 14 项，其中 5 项纯算法层（保留）、5 项有 NPU 等价形态（等价替换）、
2 项需重设计（寄存器累加器、warp 划分）、1 项本函数舍弃（packed）、1 项为迁移侧构造（`clear_accum`）。
**v2 对 #6 的判定做了更正**：v1 把它判为「重新设计（NT-only + UB 内转置副本）」并因此在向量段引入跨步
scatter 写，实测该 scatter 占内层迭代的 63%（§1.6.4）——这正是本版要修掉的根因。

### 0.5 硬件耦合性分析与 NPU 适配决策

判定问题：实现算法与优化手段是否硬件强相关？能否用在 NPU 上？依据一律指向本平台已核验资料与**本轮
可复现探针**（`history_version/design_probe_l0_gemm.py`、`design_probe_bwd_cost.py`、`design_probe_bwd_proto.py`）。

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 方案 / 依据 |
|------|------|-----------|-----------------|------|----------------|
| 数学语义 / I/O 契约 / 边界语义 | 语义 | 无 | — | **保留** | 语义层无条件保留；golden 依 §0.1 独立实现 |
| 二维 block 网格 `T.Kernel(tiles, heads)` | 优化 | CUDA blockIdx 二维 | 无（一维核网格） | **重新设计** | §0.6 R1：一维网格 + 块内 div/mod 解码 |
| key-stationary 分块 + `dK/dV` 跨块累加（#1） | 算法 | 无（结构） | 有 | **保留** | 同一结构；累加器迁 L0C（R2） |
| recompute 不物化 `P/dS`（#2） | 算法 | 无 | 有 | **保留** | 单遍结构，`P/dS` 只活 UB（§1.6.0 R2） |
| 因果块级剪枝（#3） | 算法 | 无 | 有 | **保留** | 同一 `loop_st/loop_ed`（动态界已验证） |
| tensor core GEMM（#4） | 优化 | mma / fragment | 有（Cube MAD） | **等价替换** | **显式 L0 路径**（`T.alloc_l0a/l0b` + 手工 L1→L0）：这是平台**已核验**的 GEMM 形态（PL-950-gemm-tiles 八项全过），且支持四种 `trans_A/trans_B` 组合（探针 2 实测） |
| 寄存器 fragment 累加器（#5） | 优化 | 每线程 255 寄存器 | 部分（只有 L0C 可累加） | **重新设计** | R2：L0C 累加器 + `dual_copy` 半区落地 UB 做向量 epilogue |
| **`transpose_A=True`（#6）** | 优化 | mma 的 A 转置控制 | **有（L0 路径 MN-major）** | **保留（v1 误判）** | 探针 2：`a(96,64)`(K-major L1) → L0A(MN-major) → `transpose_A=True` 的 `asc_mmad`，数值 PASS；编译器自动生成 `asc_copy_l12l0a_transpose`（转置落在 MTE1，不占向量管线）。**由此不再需要向量侧的 `ds (Mq,Nk)` 副本** |
| `exp2` + `LOG2E` 折入（#7） | 优化 | SFU 指令选择 | 有（`T.exp` 已核验；`exp2` 未核验） | **等价替换** | 自然对数域 `T.exp(s*scale − lse)`（§1.6.1 A1，机器验证 1 ulp） |
| `if_then_else` 掩码（#8） | 优化 | 无 | 有 | **等价替换** | `T.SimtVF` 内 `T.if_then_else` + 运行时块级 mask 特化（§1.6.1 A2） |
| `T.Pipelined(num_stages=2)`（#9） | 优化 | cp.async / 双 SMEM 槽 | 有（`annotate_buffer_versions`） | **等价替换** | L1 每迭代缓冲取 2 版本（§6.3）；**v2 实测本结构 `num_stages=2` 无实质收益（3.391 vs 3.469 µs），故只保留版本标注，重叠靠结构而非调度器**（§1.6.4） |
| `T.atomic_add` 逐元素（#10） | 优化 | 全局原子 ALU | 有（store-mode 原子） | **等价替换** | `T.set_atomic("add","float32")` + L0C→GM `T.copy`（已核验 PL-950-atomic-splitk；v1 实现 L0 全过） |
| `lse/delta` 行广播（#11） | 优化 | SMEM 标量读 | 有（UB lane 索引读） | **等价替换** | lane = query 轴，`lse_u[j]` 为 lane 私有读（§1.6.3） |
| warp 划分 + `threads=256`（#12） | 优化 | warp 32 线程 | 无（NPU 无 warp） | **重新设计** | R4：`T.SimtVF(threads=128)` 半区 lane 映射（副块 id 由调度合成） |
| `FullRow` policy 参数 | 优化 | warp 划分启发式 | 无 | **舍弃** | L0 路径按 mad 指令形状与 L0 布局决定分形，policy 无语义；不影响正确性 |
| `T.clear` 显式清零 | 优化 | mma 累加器初值 | 有 | **等价替换** | `clear_accum`（跨 query 循环用 `clear_accum=(kq == loop_st)`，编译级已验证） |
| packed / `GroupTiling` 分支（#14） | 优化 | 无 | 有（属另一 op） | **舍弃（本函数）** | 只覆盖 dense（§0.7） |
| host `view`/`arange` 元数据 | 语义 | 无 | 有 | **保留** | 仅 shape/stride 视图与常量构造 |

统计（18 行判定）：保留 7 项（语义、#1、#2、#3、**#6**、#11、host 元数据）、等价替换 6 项（#4、#7、#8、
#9、#10、`T.clear`）、重新设计 3 项（R1/R2/R4）、舍弃 2 项（`FullRow` policy、packed 分支）。

### 0.6 NPU 算法重设计

**重设计项 R1：二维 block 网格 → 一维核网格 + 块内解码**

- **源方案**：`T.Kernel(num_tiles, heads)` 二维 grid，硬件直接给出 `(tile, h)`。
- **NPU 新算法**：`T.Kernel(grid)` 一维核网格（`kernel.py:Kernel` 拒绝多维），
  `grid = heads * B * ceil(S_kv/96)`，块内 `h = bid // num_tiles_per_head`、`tile = bid % num_tiles_per_head`
  再解出 `b, n0`。head 取外层（同 head 的相邻 key 块在相邻 bid）以让同 head 的 `q/do/dq_acc` 切片在 L2 内复用。
- **语义保持论证**：`(tile, h)` 与 `bid` 是双射，任务集合完全一致；无算术改变。边界：`grid` 为静态整数
  （`S_kv/H/B` 编译期常量；`ds-v3-b2-8k` = 22016 < 65535 的 grid 上界，§9.1 风险 12）。

**重设计项 R2：寄存器 fragment 累加器 → L0C 累加器 + 半区 UB 中转**

- **源方案**：`st/dpt/dk_f/dv_f/dq_f` 全部 `T.alloc_fragment`；`dk_f/dv_f` 跨 query 循环常驻。
- **NPU 新算法**：Cube 的累加器只能是 L0C（`gemm_mad.py:_prepare_lower` 要求 C 带 Ascend fractal layout）。
  故 `dk_l0c (96,192) fp32 = 72 KiB`、`dv_l0c (96,128) = 48 KiB` 跨 query 循环累加（key-stationary 保持）；
  `st_l0c (96,64)`、`dpt_l0c (96,64)` 各 24 KiB 每迭代清零；`dq_l0c (64,192)` 48 KiB 每迭代清零；
  合计 216 KiB / 256 KiB（84.4%，§4.6）。
- **实现形态约束（v1 运行期实测，必须保留）**：`T.copy(L0C→UB)`（`dual_dst_mode=0`）只落**一个** AIV 的 UB，
  整 tile 向量段会读到脏数据（v1 探针 P1：99.2% 不符）。因此所有 L0C↔UB 交叉一律 `T.dual_copy` 2:1 分片
  （两个 AIV 各持 key 半区 48 行，各自 UB bank）。**v2 新增**：L1 侧仍是整 tile（`pT_s/dsT_s` 为 (96,64) 整块），
  由两个半区拼回。
- **语义保持论证**：累加顺序保持「`dV/dK` 沿 `m` 升序、`dQ` 沿 key 块跨核累加」；累加 dtype 仍 fp32。
  K 维归约在 L0C 内由 mad 完成（与源 mma 同为 fp32 累加、分组不同），与源一样只承诺 fp32 累加精度而非逐位一致。
  边界：无效 key/query 位置的贡献由 §1.6.1 A2 掩码置零（覆写补 0 引入的非零 `exp(-lse)`），块尾写回由平台 OOB clamp 裁剪（§5.4）。

**重设计项 R3（v2 重写）：`dS` 的双向使用 → 一份 UB tile + `transpose_A` 的 L0 转置**

- **源方案**：`dq_f = gemm(ds_s, k_s, transpose_A=True)`——同一份 `ds_s` 同时服侍 `dK`（`ds^T` 作 A）与
  `dQ`（`ds^T` 的转置作 A）。
- **v1 的错误判定**：v1 认为「L1 路径只支持 NT（`gemm_mad.py:162` 断言），故 `dQ` 必须另造一份
  `ds (Mq,Nk)` 的 L1 副本，并在向量段用跨步写 `ds_u[j,i]` 生成它」。实测该跨步写在 5.95 µs 的内层迭代里
  占 **3.75 µs（63%）**（§1.6.4），是把 kernel 拖成 Vector 受限的直接原因。
- **NPU 新算法（v2）**：把 `dQ` 改走**显式 L0 路径**并保留 `transpose_A=True`：
  | 乘积 | A（L0A） | B（L0B） | C（L0C） | trans |
  |---|---|---|---|---|
  | `S^T` | `k_s (96,192)` | `q_s (64,192)` | `st_c (96,64)` | (F,T) |
  | `dP^T` | `v_s (96,128)` | `do_s (64,128)` | `dpt_c (96,64)` | (F,T) |
  | `dV` | `pT_s (96,64)` | `doT_s (128,64)` | `dv_c (96,128)` | (F,T) |
  | `dK` | `dsT_s (96,64)` | `qT_s (192,64)` | `dk_c (96,192)` | (F,T) |
  | `dQ` | `dsT_s (96,64)` | `kT_s (192,96)` | `dq_c (64,192)` | **(T,T)** |
  即**同一份 `dsT_s`** 既作 `dK` 的 A（K-major）又作 `dQ` 的 A（`trans_A=True`），后者由
  `asc_copy_l12l0a_transpose` 在 L1→L0A 搬运中完成转置（MTE1，每迭代 12 KiB），**向量段不再产生第二份 tile**。
- **语义保持论证**：`(A B^T)` 与 `(A^T)^T B^T` 逐元素恒等；转置只改变操作数的搬运顺序，不改变任何算术。
  唯一代价是 MTE1 上的 12 KiB/迭代转置搬运与 L0A 需两块布局不同的 tile（K-major 的 `a_k` 与 MN-major 的 `a_t`，§4.6）。
  边界：转置 tile 与主 tile 同形（全 extent），尾块同样由掩码补 0。
- **证据**：探针 2（`design_probe_l0_gemm.py`）对 (T,T)/(F,F)/(F,T) 三种组合各跑一次小 shape 数值比对，
  全部 PASS（rel 误差 1.1e-7）；端到端原型（`design_probe_bwd_proto.py`）在 `S=384` 与 `S=2048` 上
  `dq/dk/dv` 全 PASS。

**重设计项 R4：warp/线程级并行 → SimtVF 半区 lane 映射**

- **源方案**：`threads=256` 的线程块，`T.Parallel(block_n, block_m)` 的元素映射由编译器按 256 线程划分。
- **NPU 新算法**：Cube 侧由 mad 承担（AIC，28 核），向量侧在 `T.SimtVF(threads=128)` 内以
  `T.Parallel(48, 64)` 覆盖本 AIV 的半区（`_HN=48` 行 × 64 列），最内轴为 query（连续 `lse_u[j]` 的 lane 私有读）。
  副块编号不入用户代码：`T.Vector(vector=2)` 形态在本工具链不可编译（`InternalError … dup sid`，§3.5.4 反例），
  自动调度在 AIC/AIV 分段时合成 cthread sid；半区的**全局 key 下标**由
  `sid = T.call_extern("int64", "asc_get_sub_block_id")` 在 kernel 体绑定（v1 运行期已验证 sid 取值 0/1 且确定）。
- **语义保持论证**：逐元素映射不影响结果（无跨 lane 归约）；`P/dS` 的 fp32 值由全局 key/query 下标决定
  （`gi = n0 + sid*48 + i`、`gj = m0 + j`）。累加分组变化只出现在 GEMM 的 K 维（见 R2）。

**重设计项 R5：全局原子加 → store-mode 原子写回**

- **源方案**：`T.atomic_add(dq_acc[...], dq_f[i,d])` 逐元素原子加。
- **NPU 新算法**：`T.set_atomic("add","float32")` 打开硬件 store 原子模式，随后把 `dq_c` 用一次 `T.copy`
  落到 `dq_acc[q 行, h, :]`，再 `T.set_atomic_none()`。省掉 UB 中转（L0C fp32 → GM fp32 同 dtype）。
- **语义保持论证**：`dq_acc += tile` 与逐元素原子加是同一数学累加；fp32 累加、顺序仍不确定（与源一致）。
  边界：尾 query 块由平台 OOB clamp 把 store 区段裁到 `S_q` 以内（设备码 `min(64, S_q - m0)`）。

**重设计项 R6（v2 新增）：5 个 GEMM 统一走显式 L0 路径，禁止与 L1 路径混用**

- **背景（本轮实测的新约束）**：`T.gemm(A_l1, B_l1, C_l0c)`（L1 操作数路径）由 lowering 内部管理
  L0A/L0B 的暂存；若同一 kernel 内另有用户显式分配的 `T.alloc_l0a/l0b`，两者的 L0 空间会互相覆盖。
  **实测**：把 `dQ` 改成 L0 路径、其余 4 个留在 L1 路径时，`dk/dv` 出现 4.8e-1 / 3.1e-2 的错误
  （原型 bisect，§1.6.4）；把 5 个 GEMM 全部改为显式 L0 后同一测试全 PASS。
- **NPU 新算法**：5 个 GEMM 全部 `T.gemm(a_l0a, b_l0b, c_l0c, …)`，A/B 由**每个 GEMM 前的一次
  `T.copy`（L1→L0）**填充；L0A 两块（K-major `a_k (96,192)`、MN-major `a_t (96,64)`），
  L0B 两块（`b_lo (64,192)` 装 `q/do`、`b_hi (192,96)` 装 `qT/doT/kT`）。
- **语义保持论证**：L0 路径与 L1 路径表达同一 `C = A·B^T` 语义（同一 mad 指令、同一 fp32 累加），
  差别只是操作数的暂存由用户显式管理。数值上原型与 fp64 golden 一致（1.6e-2 以内，bf16 容差）。
  容量：L0A 48 KiB / 64 KiB（75%）、L0B 60 KiB / 64 KiB（93.75%），见 §4.6；**L0B 余量仅 4 KiB，
  故 L0 侧一律单版本**（不做 L0 双缓冲，§9.1 风险 3）。

**无重设计项之外的补充说明**：`delta` 的产生（`_delta_kernel`）与 `dq` 的缩放 cast（`_dq_land_kernel`）
不在本 kernel 内，本设计只依赖其接口契约（fp32 / fp32 in-out）。

### 0.7 标杆实现与 packed 分支说明

- **源算子路径**：`/workspace/TileOPs/src/tileops/_kernels/attention/mla/bwd.py`（`_bwd_kernel`，`bwd.py:109-264`），
  抽取副本 `examples/TileOPs/tileops/kernels/attention/mla_dense_bwd/_mla_dense_bwd_kernels.py`（逐行同源）。
- **golden 基准**：`/workspace/TileOPs/workloads/attention/mla.py:441-460` 的 `mla_bwd_reference`（fp32 torch，
  slab 物化 `probs` 后 `einsum` 反向，bottom-right 对齐，`probs` 用 `nan_to_num` 处理 `lse=-inf` 行）与
  `MLADenseBwdWorkload.ref_program`（`mla.py:506-518`，fp32 梯度**只舍入一次**到输入 dtype）。
  §8.1 的 golden 移植该实现，**不复刻** §0.6 的分块算法。
- **精度/性能用例**：`/workspace/TileOPs/tests/ops/test_mla_bwd.py`、`/workspace/TileOPs/benchmarks`
  （manifest `MLADenseBwdOp` 5 条 workload）；容差来源 `workloads/numerics.py:273-283` 的 `reference_tolerance`。
- **packed 分支差异（为何不在本函数覆盖）**：`dense_seqlen_q=0` 时源工厂改走 `GroupTiling`（`bwd.py:138-139, 181-184`）：
  每 task 的 `(request, n0)` 由变长序列的块累计表解码，`total_q/total_kv` 是动态维，`lse/delta` 布局变为
  head-major `(H,T)`。差别是**任务解码与统计张量 layout**，不是数学；本平台一次编译绑定静态 shape。
  该分支属另一个算子（`MLAVarlenBwdOp`），单独设计；**本版的共享核心（5 个 L0 GEMM + 2-tile 合并写 epilogue +
  dQ 的 `transpose_A` 复用）可直接被 packed 版复用**，packed 只替换块解码与 `lse/delta` 寻址。
---

## 1. 概述

### 1.1 算子名称

`_bwd_kernel`（算子：`MLADenseBwdOp` 的主 pass，MLA prefill 反向）。

### 1.2 功能描述

给定前向的 `q, k, v, do, lse, delta`，按 key-stationary 单遍重算 `P` 与 `dP`，产出 `dk, dv`（输入 dtype）
并向 fp32 缓冲 `dq_acc` 原地累加 `dQ`（缩放 cast 由 `_dq_land_kernel` 完成）。
**v2 的执行结构**：5 个矩阵乘全部走 Cube 的显式 L0 路径，向量侧只产生两个**合并写**的 fp16 tile
（`P^T`、`dS^T`），`dQ` 用 `transpose_A=True` 直接复用 `dS^T`——源算子的「一份 `dS` 服侍两个 GEMM」优化被完整保留。

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

算法族选择见 §1.6.0：**key-stationary 单遍重算（flash 反向族）**，与源算子同一族、同一分块结构。
v2 相对 v1 的变化**全部在执行结构层**（不改数学、不改算法族）：

1. **任务映射**：一维核网格，`grid = H * B * ceil(S_kv/96)`；块内解出 `(h, b, n0)`。
2. **key 常驻（per launch）**：`k (96,192)`、`v (96,128)`、`kT (192,96)` 从 GM 搬入 L1 并跨整个 query
   循环常驻；`dk/dv` 的 L0C 累加器清零。
3. **query 块循环**（因果剪枝：`kq ∈ [max(0,(n0-shift)//64), ceil(S_q/64))`）：
   a. 搬入 `q_s (64,192)`、`qT_s (192,64)`、`do_s (64,128)`、`doT_s (128,64)`、`lse/delta (64,)` fp32；
   b. `st_c = k·q^T`、`dpt_c = v·do^T`（显式 L0，每迭代清零）；
   c. 半区 `dual_copy` 把两块 L0C 落到 UB（fp32，同 dtype），向量侧算 `p = exp(st*scale − lse)`
      （运行时块级 mask 特化 + 因果/尾块掩码）→ 只写两个 fp16 tile：`p16`、`dsT = p*(dpt−delta)`；
   d. 两块 fp16 经 `dual_copy` 落 L1（`pT_s`、`dsT_s`）；
   e. `dv_c += pT·doT^T`、`dk_c += dsT·qT^T`（跨迭代累加）、`dq_c = dsT·kT^T`（**`transpose_A=True`**，每迭代清零）；
   f. `set_atomic("add","float32")` 后把 `dq_c` 整块落到 `dq_acc[q 行, h, :]` 并关闭原子模式。
4. **块尾**：`dk_c`、`dv_c` 半区落地 UB，向量侧 cast（`dk` 另乘 `scale`）后写回 GM；尾部行由平台 OOB clamp 裁剪。

与 v1 的结构差异（逐条对应 §0.6）：`dQ` 由「向量侧跨步转置副本 + NT GEMM」改为「`dsT` 直接 + `trans_A`
L0 GEMM」（R3）；5 个 GEMM 全部显式 L0、不与 L1 路径混用（R6）；tile `64→96`（§5.2）；向量段由 3 个输出
tile 降为 2 个（§1.6.2）。

### 1.5 数据流图

```
GM: q,k,v,do(f16)  lse,delta(fp32)  dq_acc(fp32, in-out)
        │  GM→L1 (MTE2, nd2nz / dn2nz；越界尾部由 OOB pass clamp+fill)   │
        ▼                                                                 │
   L1: k_s,v_s,kT_s(常驻)  q_s,qT_s,do_s,doT_s(每迭代)                    │
        │  L1→L0 (MTE1，每次 GEMM 前一次 T.copy；dQ 的 A 走 transpose 搬运) │
        ▼                                                                 │
   L0A/L0B ──mad──▶ L0C: st,dpt(每迭代) dk,dv(跨迭代) dq(每迭代)           │
        │                                                                 │
        │ 半区 dual_copy (L0C→UB, fp32 同 dtype, AIV 各持 48 key 行)        │
        ▼                                                                 │
      UB(fp32): st_u, dp_u ──SimtVF(128) 半区 48×64──▶ UB(f16): p16_u, dsT_u
        │  UB→L1 (半区 dual_copy；InsertNd2Nz 改写为 NZ scatter)             │
        ▼                                                                 │
      L1: pT_s, dsT_s ──▶ dV 的 A / dK 的 A(直接) / dQ 的 A(transpose_A)    │
                                                                          │
   L0C(dq) ──set_atomic(add) + T.copy──▶ GM: dq_acc ──────────────────────┘
   L0C(dk,dv) ──dual_copy→UB──cast(+scale)──T.copy→ GM: dk,dv
```

**P/dS 全程留在 L0C/UB/L1，不落 GM**；`dS` 只有**一份** UB tile（v1 有两份：`dsT` 与转置的 `ds`）。

### 1.6 算法调研与优化分析

#### 1.6.0 算法调研（Algorithm Research）

> 调研对象：**MLA 注意力反向（给定 lse/delta 的梯度）这一数学语义的算法族**。源算法只是基线候选。
> 检索范围：平台 knowledge 的 `algorithm-candidates.md`（`ALG-attention`）、knowledge 下全部 950 条目
> （`PL-950-*`、`CONST-950-*`、`TRAP-950-*`）、本仓 `examples/` 同类实现、源仓 golden 与 manifest roofline、
> 结构判据自查。网络检索未使用（本地源已覆盖该族的全部已知结构）。

**R1 等价化简公式候选**（基线在表内；正式论证与收益在 §1.6.1）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 纳入 |
|---|------|------------|-----------|---------|------|
| 0 | **基线（源算法）** | `P=exp2(s·scale·log2e − lse·log2e)`；5 个 GEMM（`dQ` 用 `transpose_A`）；`dq` 逐元素原子加 | — | — | ✅（基线） |
| 1 | 自然对数域 `exp` | `P=exp(s·scale − lse)` | 数学恒等（`exp(x)=exp2(x·log2e)`） | 每 score 少 1 次乘（源式逐元素重算的 `lse·log2e`） | ✅（§1.6.1 A1） |
| 2 | 块级 mask 特化 | 全可见块省掉 `select`，只在对角/尾块生成掩码 | 谓词在该块内恒真，恒等 | 每 score 少 2 次比较 + 1 次选择；快路块占 49.5% | ✅（A2） |
| 3 | 行统计量 `lse·log2e` 逐行提升 | 先算 `lse2[m]=lse[m]·log2e` | 恒等 | 被 #1 完全吸收 | ❌（冗余） |
| 4 | `scale` 折进 `q`/`k` 装载 | `(q·scale)` 参与 score GEMM | 数学恒等，但 fp16 输入先乘常量 → 每元素多一次 fp16 舍入 | 逐迭代乘数：score 6144 次 vs 预乘 12288 次 → 不省；且 `q` 还要供 `dK`、`k` 还要供 `dQ`，需两份副本 | ❌（定量） |
| 5 | `delta` 项拆成 rank-1 修正 | `dS=P∘dP − P∘δ` 分两项进 GEMM | 数学恒等 | 需多一个 GEMM（+192·Mq·Nk MAC/迭代 ≈ +5.7% 计算量） | ❌ |
| 6 | `dV`、`dK` 合并为一个 GEMM | 拼 `A=[pT; dsT]`、`B=[doT; qT]` | **不等价**（会求和成 `pT·doT^T + dsT·qT^T`） | — | ❌（不等价） |
| 7 | 物化 `P/dS`（非 flash） | 一遍算 `s/P/dP/dS` 落 GM，再扫描做 3 个 GEMM | 恒等 | 中间缓冲 O(B·H·S²) 与多遍扫描 | ❌（R3 淘汰） |
| 8 | query-stationary 变体 | `dQ` 常驻、`dK/dV` 原子累加 | 恒等 | 把原子放大项从 `dQ` 换成 `dK+dV`（宽 320） | ❌（R4 淘汰） |
| 9 | 两 kernel 拆分（dQ kernel + dK/dV kernel） | 各自单遍 | 恒等 | FLOPs ×2，破坏三 launch 契约 | ❌ |
| 10 | **单份 `dS` 双向复用**（`trans_A`） | 只产 `dS^T (Nk,Mq)`；`dK` 用它作 A，`dQ` 用它作 `trans_A` 的 A | 恒等（`(AB^T)` 与 `(A^T)^T B^T`） | 消掉一份转置副本与向量侧 scatter（实测 −3.75 µs/迭代，§1.6.4） | ✅（§0.6 R3） |

**R2 在线算法**：**有，且基线即在线变体。** 结构依据：`lse`（前向 log-sum-exp）与 `delta`（`rowsum(o*do)`）
都由外部提供，因此反向**不需要**前向 online softmax 的 running max / 重缩放机器；每个 (key 块, query 块) 对
只需重算局部 `s/dP` 并立刻消费（`P/dS` 只活在片上），得到单遍扫描、O(tile) 中间态的流式结构。
对照的非在线变体是 R1#7（物化 `P/dS`，3–4 遍、O(B·H·S²) 缓冲），在 R3/R4 中被容量淘汰。
**无更优在线结构**：本算子的输出里必有一个（`dQ` 或 `dK/dV`）要沿另一轴跨块归并，单遍结构下无法再压缩
（R1#6 的合并已证不等价），故在线族内部不存在「更少遍数」的变体——这条负向断言的结构依据是上面的交换性分析。

**R3 复杂度对比**（四口径；代表 workload = `ds-v3-b2-8k`：B=2, S_q=S_kv=8192, H=128, D_qk=192, D_v=128, bf16；
因果 score 数 8.59e9）：

| 候选 | FLOPs | 访存 GM 视图 (Bytes) | 扫描遍数 | 中间缓冲峰值 / 片上预算 | 每 query 行 L1 填充 |
|------|-------|---------------------|---------|----------------------|-------------------|
| **C1 = v1（L1 路径 NT + 向量转置副本，Nk=64）** | 14.34 T | ≈ 280 GB（q/do 各两份 173 + k/v/kT 2.2 + `dq` 原子 104 + `dk/dv` 1.3） | 1 | L0C 160 KiB / L1 168 KiB / UB 176 KiB | 1280 B × (S_kv/64 = 128) 次 |
| **C2 = v2（全 L0 路径 + `trans_A` 复用，Nk=96）** | 14.48 T（+1% 尾块量化） | ≈ 281 GB（同上；`dq` 原子 69.4） | 1 | L0C 216 KiB / L1 304 KiB（双版本）/ UB 163 KiB / L0A 48 KiB / L0B 60 KiB | 1280 B × 86 次 |
| C3 = v1 + 两趟向量转置（`epi2`） | 14.34 T | ≈ 280 GB | 1 | 同 C1 | 1280 B × 128 |
| C4 = 两 kernel 拆分 | **28.7 T**（`P`、`dP` 各算两遍） | ≈ 175 GB（无原子） | 2 | 同量级 | 两倍 launch |
| C5 = 物化 `P/dS`（R1#7） | 14.34 T | ≥ 270 GB 额外往返 | 3–4 | **O(B·H·S²) ≈ 34 TB** → 远超 128 MiB L2 | 不可行 |
| C6 = 纯 Vector（无 Cube） | 14.34 T | 同 C2 | 1 | 同量级 | 28 Vector 核 lane 级乘加，算力低 1–2 个数量级 |

口径说明：FLOPs 按 manifest roofline 公式 `B·H·(scores·(2(3D_qk+2D_v)+5) + rows·2·D_v)`；GM 访存含输入读 +
输出写 + 跨块累加往返，**不含** L1/UB 内部搬运（单列在 R4 的端口分解）；扫描遍数指 `q/k` 被 GM 读取的遍数。
C2 的 FLOPs 是解析值；按 `ceil` 分块实际计算 14.48 T（含 1.0% 尾块量化开销：1 415 000 次迭代 × 6144 score 位
vs 解析 8.59e9 个因果 score 位）。

**每迭代端口分解**（`Nk=96, Mq=64`，一次 (key 块, query 块) 迭代 = 6144 个 score；口径 = 该迭代的
L1/UB 端口流量，含 GM→L1）：

| 项 | 组成 | 字节 |
|----|------|------|
| GEMM 操作数 L1→L0（显式） | `k_s 36864 + q_s 24576 + v_s 24576 + do_s 16384 + pT_s 12288 + doT_s 16384 + dsT_s 12288 + qT_s 24576 + dsT_s 12288 + kT_s 36864` | 221184 |
| UB→L1 写（2 个 tile 半区） | `p16_u 6144 + dsT_u 6144` ×2（两个 AIV） | 24576 |
| GM→L1 装载（每迭代） | `q_s 24576 + qT_s 24576 + do_s 16384 + doT_s 16384 + lse/delta 512` | 82432 |
| GM→L1 常驻摊销 | `k_s + v_s + kT_s = 98304` ÷ 平均 64.3 次迭代 | 1529 |
| **合计** | | **325625**（≈318 KiB） |

对应 `10 240 000 FLOP/迭代`（`6144×1664 + 64×256 = 10 223 616 + 16 384`）→ **31.4 FLOP/字节**（v1 为 24.5）。
全量 = 1 415 000 × 325 625 B = **460.8 GB** L1/UB 端口流量（v1 为 590.6 GB）。

**R4 硬件亲和性评估**（逐候选；容量对照 §4.5/§4.6；常数一律取本平台实测条目或本轮探针）：

| 候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|------|-------------|---------|------------|---------|------------|------|
| C1 = v1 | 5 个 NT GEMM（L1 路径）可跑，但向量侧被迫做跨步转置 | L0C 62.5% / L1 32.8% / UB 71.2% | 64 全为 16 倍数 | 全静态 | 单阶段 | ❌ **实测 Vector 受限**（cube 8.3% / vector 73.5%），scatter 占 63% |
| **C2 = 本版** | 5 个 GEMM 全显式 L0（**平台已核验形态**），四类 `trans` 组合齐全；逐元素落 AIV | L0C 84.4% / L1 59.4% / UB 65.7% / L0A 75% / L0B 93.75% | 96/64/192/128 全为 16 倍数 | `S_q/S_kv/B/H` 编译期常量 | GEMM 与向量 epilogue 同 kernel 交替（`__mix__(1,2)`），L0 单版本 | ✅ **原型实测 PASS，3.46 µs/迭代** |
| C3 = v1 + 两趟转置 | 同 C1，但第二趟跨步读 | 同 C1 | 同 | 同 | 同 | ❌ 实测第二趟代价 +1.64 µs/迭代（+80%）；C2 只多一次 12 KiB 的 MTE1 转置搬运（≈0.02 µs） |
| C4 = 两 kernel 拆分 | 同 C2 | 同 | 同 | 需两次 launch | — | ❌ FLOPs ×2，违反三 launch 契约 |
| C5 = 物化 | 纯 Cube 友好 | O(B·H·S²) ≈ 34 TB ≫ L2 | — | — | — | ❌ 容量淘汰 |
| C6 = 纯 Vector | 矩阵乘落错单元 | 同 | — | — | — | ❌ 无实测必要（量级差） |

**设计期定量 roofline（D-2；常数全部来自本平台实测条目 + 本轮探针，见 §1.6.4 表）**

| 项 | 公式 | 值（`ds-v3-b2-8k` / C2） | 依据 |
|----|------|------------------------|------|
| 计算项（Cube） | 14.48 TFLOP ÷ 358 TF/s | **40.5 ms** | `CONST-950-gemm-mte2-burst`：bf16 Cube 有效峰值 356–359 TF/s（256×256 tile） |
| 流量项（GM→L1 填充） | 1 415 000 × 82432 B = 116.6 GB ÷ 1.96 TB/s | **59.5 ms** | `CONST-950-gemm-mte2-burst`：单核 MTE2 70 GB/s（128 B 行段）→ 28 核 1.96 TB/s |
| 向量项（epilogue） | 8.70e9 score × 0.20 ns | **62 ms** | 本轮探针 `epi0`：0.825 µs / 4096 score（2 AIV） |
| fixpipe 项（L0C→UB） | 1 415 000 × 49152 B ÷ (92 GB/s × 2 AIV) | **12.6 ms** | 本轮探针 `l0c2ub` |
| UB→L1 项 | 1 415 000 × 24576 B ÷ (52 GB/s × 2 AIV) | **11.2 ms** | 本轮探针 `ub2l1` |
| `dq` 原子项 | 8.70e9 × 768/96 B = 69.6 GB ÷ ~1.5 TB/s | **≤ 46 ms**（v1 的 104 GB 在 455 ms 内可被掩盖 ⇒ 实测下界 ≥1.1 TB/s） | 结构性推算（未单独标定） |
| 发射/同步项 | 1 415 000 × 约 14 个 tile 级 op + 2 次 VF call | 未单独标定 | — |

**估算下界 = max(计算项, 流量项, 向量项) ≈ max(40.5, 59.5, 62) = 62 ms**；把 AIC 侧的串行项
（填充 59.5 + fixpipe 12.6 + 原子 ≤46）与 AIV 侧（62 + 11.2）按「无重叠」相加得
**~130–190 ms**，C2 原型实测外推 **175 ms** 正落在该区间（§1.6.4）。三项相对大小说明：
**本算子在这套片上预算下的地板由「每 query 行的 L1 填充」与「每 score 的向量 epilogue」共同决定，
Cube 计算项（40.5 ms）单独达不到**——这是本版与 v1「L1 端口是唯一地板」结论的关键修正，
也是 §1.6.4 里「80% Cube MFU 不可达」的定量依据。

**调研结论**：选定**算法族 = key-stationary 单遍重算（flash 反向族 C2）**，关键依据：
R3 中 C2 与基线 FLOPs 相同（+1% 尾块）而 GM 访存最低可达（C4 2×、C5 容量淘汰、C6 单元不匹配、C1/C3 多一次
全 tile 向量 pass），R4 中 C2 的 5 个乘法在平台**已核验的 L0 形态**下全部成立（含 `trans_A`），且
`trans_A` 复用把 v1 的向量 scatter（占 63%）整项消掉。相对基线的结构差异：
**同一算法族，按 NPU 约束把 GEMM 全部落到显式 L0、累加器迁 L0C、把「转置副本」从向量段移到 MTE1**；
源算子 §0.4 的 14 项优化手段的意图承接见 §0.5 处置列（全部有承接或明确舍弃理由，无静默丢弃）。
下游约束：§1.4 与 §1.6.1–§1.6.3 的分析对象即 C2；§3.1 用 §1.6.1 的优化后公式。

#### 1.6.1 数学等价优化（公式级）

分析对象 = §1.6.0 选定的 C2。逐项四要素如下，机器验证见本节末。

| # | 优化项 | 原式（基线） | 优化后公式 | 等价性论证 | 收益量化 |
|---|--------|-------------|-----------|-----------|---------|
| A1 | **自然对数域概率** | `P = exp2(s·scale·log2e − lse·log2e)` | `P = exp(s·scale − lse)` | `exp2(x·log2e) ≡ exp(x)` 是恒等式；两式对同一实数输入只差舍入路径。fp64 参照下两式各自距真值 ≤ 1 个目标 dtype ulp，候选式不劣于基线式 | 每 score 少 1 次乘（源式在 (i,j) 内逐元素重算的 `lse·log2e`）→ 每迭代 6144 次向量乘；`exp` 与 `exp2` 在本平台同属 SFU 指令（`T.exp` 有核验条目），不引入未核验 API |
| A2 | **块级 mask 特化** | 每个 (key 块, query 块) 都生成 `select(mask, p, 0)` | 块级谓词 `n0+95 < S_kv ∧ m0+63 < S_q ∧ n0+95 ≤ m0+shift` 为真时直接输出 `p`，否则同一 `select` | 该分支条件是掩码谓词在整块上恒真的**可判定充分条件**（`n≤m+shift` 对块内最大 `n` 与最小 `m` 成立即全真），故两式逐元素相同；机器验证对 6 组 shape 的全部 block 逐元素比对，违反率 0 | 快路块占 49.5%；每 score 省 2 次整数比较 + 1 次选择 → 约 6144×2 次向量操作/迭代 |

**A2 的实现机制**：块级谓词依赖 `n0 = f(bid)` 与 `m0 = kq·BLOCK_M`，二者都是**运行时量**，因此 A2 由
kernel 内的**运行时块级分支**实现（两支各一套半区 epilogue；编译后成为两个 SIMT VF 函数 + 运行时二选一）。
编译期唯一可静态判定的是「是否存在尾块」（`S_q % 64`、`S_kv % 96`），它决定掩码里要不要保留
`m0+j < S_q` / `n0+i < S_kv` 的比较结构，但不决定某个具体块是否全可见。

**快路的安全性论证**：快路要求 `m0+BLOCK_M ≤ S_q`（无填充 query 行）且 `n0+BLOCK_N−1 ≤ m0+shift`
（块内每个 key 对块内每个 query 可见），即该块**不可能**包含「看不到任何 key」的行（`lse=-inf` 只出现在
这类行上），故 `exp(s·scale − lse)` 的指数输入有限，不会产生 inf/NaN。

**已确认无需改动的项**（CSE/降代换自查）：`ds = p*(dpt − delta)` 已是最优形态（`p*dpt − p*delta` 会多一次乘）；
`p` 的 fp32 结果被 `p16`(cast) 与 `dsT`(cast) 共用（每元素只算 1 次 `exp`）；`lse/delta` 已是每 query 行一次
装载、按 lane 私有读；`dk` 的 `scale` 乘只发生在块尾（每 launch 96×192 = 18432 次）。

**弃选项的定量依据**（对应 R1 表 #4/#5/#6）：`scale` 折进 `q` 需 64×192 = 12288 次预乘且需两份副本，净亏；
`delta` 的 rank-1 拆分需额外一次 GEMM（+192·64·96 MAC/迭代 ≈ +5.7% 计算量）；`dV/dK` 合并会求和，不等价。

**等价性机器验证（D-1）**：脚本 `examples/mla_dense_bwd/_bwd_kernel/verify_equiv.py`（torch CPU，fp64 参照，
含 `s·scale−lse ∈ [-8e2, 8e2]` 的随机样本 + 角点值：精确抵消、上/下溢、次正规、`lse=-inf` 空行、`700/−700`
极值；A2 覆盖 6 组 shape 含尾块与 `S_q<S_kv`/`S_kv<S_q`）。**revision v2 于 2026-11-25 重跑**
（`ASCEND_RT_VISIBLE_DEVICES=""`，toolchain `ascend-950pr-9579 / TileLang ascend 0.1.15`），输出
`EQUIV_SUMMARY ALL_PASS`：

| 采纳项 | 样本数 | 候选 vs 基线 最大 ulp | 违反率 | 候选 vs fp64 最大 ulp | 基线 vs fp64 最大 ulp | 结论 |
|---|---|---|---|---|---|---|
| A1 exp 域 P（目标 dtype float16） | 1048592 | 1 | 0.000e+00 | 1 | 1 | EQUIV_PASS |
| A1 exp 域 P（目标 dtype bfloat16） | 1048592 | 1 | 0.000e+00 | 1 | 1 | EQUIV_PASS |
| A2 块级 mask 特化（block_n=64, block_m=64） | 76838912 | 0 | 0.000e+00 | — | — | EQUIV_PASS |

（A2 快路块 9285/18761。A2 的谓词形状与 block 大小无关（只用到块的最大/最小下标），故 block_n=96 的分块
下结论不变；v2 未改动 A1/A2 的数学。）

**优化结论**：采纳 2 项（A1、A2），均 EQUIV_PASS；本版**未新增公式级优化**——v2 的收益来自执行结构
（§0.6 R3/R6），不来自数学变形（源公式已无更多 CSE/降代换空间，见上「已确认无需改动的项」）。

**优化后公式**（供 §3.1 使用）：

$$
P[m,n]=\exp\!\big(\mathrm{scale}\cdot s[m,n]-\mathrm{lse}_m\big)\quad\text{all-visible block}\qquad
dS[m,n]=P[m,n]\cdot\big(dP[m,n]-\mathrm{delta}_m\big)
$$

对角块与尾块再乘因果掩码（`T.if_then_else`，见 A2 的块级特化）。其余乘积与写回形态不变
（`dV = P^T dO`、`dK = scale·dS^T q`、`dq_acc += dS·k`）。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

盘点实现方案（C2）中**全部**循环与标量计算点：

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由（附查证） |
|---|--------|-----------|-------------|---------|----------------------|
| 1 | `P = exp(s·scale − lse)` | GPU 上 `T.Parallel(bn,bm)` 逐元素 | `T.SimtVF(threads=128)` + `T.Parallel(48,64)` + `T.exp` | ✅ | —（`T.exp` 核验：knowledge/repro/PL-950-row-primitives.py；v1 实现 L0 全过） |
| 2 | 因果 / 尾块掩码 | 逐元素布尔 + `select` | 同一 `T.Parallel` 内的 `T.if_then_else`（A2 运行时块级特化掉全可见块） | ✅ | —（`if_then_else` 核验：PL-950-attention.py） |
| 3 | `dS = P·(dP − delta)` + 行广播 | 逐元素 | 同一 `T.Parallel` 内读 `dlt_u[j]`（lane 私有读，无广播指令） | ✅ | —（SIMT 内列/行广播已核验：knowledge/repro/PL-950-conversion-layout.py） |
| 4 | `p16`/`dsT` 的 fp32→f16 cast | 逐元素 | `T.cast` 写另一个 UB tile | ✅ | —（转换含 ±inf/次正规已核验：同上） |
| 5 | **`dS` 的 (Mq,Nk) 转置副本** | v1：向量段 `ds_u[j,i] = cast(dsv)` 跨步写（实测 3.75 µs/迭代） | **取消该计算点**：`dQ` 改用 `trans_A=True` 的 L0 GEMM，转置由 `asc_copy_l12l0a_transpose`（MTE1）完成 | ✅（换单元 + 换层） | —（探针 2 实测 (T,T) 数值 PASS；原型设备码含 `asc_copy_l12l0a_transpose`） |
| 6 | `dk` 写回 `cast(dk_c·scale)` | 逐元素 | 块尾一次 `T.Parallel(48,192)`（半区落地后） | ✅ | — |
| 7 | `P/dS` 的 K 维归约 | GPU tensor core | Cube mad（显式 L0，fp32 累加） | ✅（换单元，非标量循环） | — |
| 8 | `dq_acc += dq` | 逐元素 `atomic_add` | `T.set_atomic("add","float32")` + 一次 L0C→GM `T.copy` | ✅（硬件原子，无标量循环） | —（PL-950-atomic-splitk；v1 L0-11 实测加法语义成立） |
| 9 | 任务索引 `h/b/n0`、`loop_st/loop_ed`、`shift` | 标量整数运算 | 无 | ❌ | **block 级任务映射**：每 block 一次的常数项，非逐元素热点，逐元素等价 API 不存在 |
| 10 | 尾块边界（有效行/列数与 clamp） | 源码 `T.if_then_else` 逐元素守卫 | **无需替代**：GEMM 用全 extent，GM↔片上搬运的越界尾部由 `AscendInsertOOBPadding` 改写成 `min()` clamp + `asc_fill_l1` 补 0 | ✅（平台 pass 消除） | —（`pipeline.py` 该 pass 的 docstring 明确 "Clamp DMA copy OOB tails … emit GM->L1 padding fills"；v1 探针 B 设备码含 `min()` 与 `asc_fill_l1`） |
| 11 | `lse/delta` 行统计装载 | 逐元素 | `T.copy` 搬 (64,) fp32 | ✅ | — |
| 12 | L1→L0 操作数暂存（v2 新增） | — | 每次 GEMM 前一次 `T.copy(L1 region, L0 region)`，由 MTE1 整块搬运 | ✅（DMA，非标量循环） | —（PL-950-gemm-tiles 的已核验形态；本轮原型设备码 `asc_copy_l12l0a/l12l0b`） |

**向量化结论**：12 个计算点中 **11 项**已向量化 / 换单元 / 由平台 pass 消除，仅 **1 项（#9）**保留为标量，
理由类别 = **block 级任务映射**（每 block 一次，非逐元素热点）。相对 v1 的量化改进：
向量段每个 score 的输出 tile 由 3 个降为 2 个（`p16`/`dsT`），UB 写字节 14 B → 12 B/score，
且不再有跨步 scatter（v1 的 #5 计算点整项消除）。

#### 1.6.3 向量化轴与数据布局决策 ⭐

本算子的核内布局自由度集中在 **(key × query) 的 2-D 中间块**（`P`、`dP`、`dS` 各 96×64）。I/O 契约固定为
BSHD（尾轴 `D` 连续），`D` 维是矩阵维（Cube 契约），因此「向量化轴」只能在 key/query 两轴中选。

| 评分项 | 候选 1：lane = query（本设计） | 候选 2：lane = key | 候选 3：UB 内双转置 |
|--------|------------------------------|-------------------|------------------|
| 整除性 / 尾 lane | Mq=64，fp32 向量宽 4 → 16 向量/行，无尾 lane；实测编译成 `float2`（32 线程盖 64 列） | Nk=96 → 6 向量/行（fp32），尾 lane 无，但行数 96 非 2 幂 | 同候选 1 |
| 累加链形态 | 最内轴 stride=1（`(Nk,Mq)` 行内连续），`lse/delta` 为 lane 私有读 | 最内轴是 key，`lse/delta` 需沿 key 轴**列广播** | 同候选 2 |
| 输出 tile 数 | **2 个**（`p16`、`dsT`，全部合并写） | 2 个 + 1 个转置 | 3 个（多一份替身） |
| repack 代价 | 0 次跨步写；`dQ` 的转置由 MTE1 承担（12 KiB/迭代） | 需给 `dV`/`dK` 各造转置副本（2 份跨步写） | +6144 B/迭代 UB |
| 实测代价 | epilogue **0.201 ns/score**（合并写，0.825 µs/4096） | **1.050 ns/score**（v1 的 `epi1`，4.302 µs/4096） | **0.603 ns/score**（v1 的 `epi2`，2.469 µs/4096） |
| 弃选量化理由 | ✅ 采纳 | ❌ 跨步写 **5.2×** 于合并写（探针 `epi1` vs `epi0`） | ❌ 两趟 **3.0×** 于合并写（探针 `epi2`） |
| 候选 4：纯 SIMT 直写 GM | AIC 不参与 | — | ❌ 5 个 96×64×192 级矩阵乘落 56 个 Vector 核，算力差 1–2 个数量级（R4 C6） |
| 候选 5：中间块经 GM 往返 | — | — | ❌ 每迭代多 3×12 KiB 的 GM 往返，全量约 +103 GB HBM 流量 |

**多 AIV 的执行体映射**：本平台把线程域限定在 `T.SimtVF` 内，副块（AIV）编号只由自动调度在 AIC/AIV 分段时
合成；显式 `T.Vector(vector=2) as sid` 在本工具链**不可编译**。因此向量段是**半区**（`T.Parallel(48,64)`，
本 AIV 的 48 个 key 行），半区的全局 key 下标由 `sid = T.call_extern("int64","asc_get_sub_block_id")` 提供
（v1 运行期实测 0/1 确定）。

**epilogue 的代价模型与优化阶梯**（本版新增，依据探针实测）：

- 实测：合并写形态 0.825 µs / 4096 score / 2 AIV = **0.20 ns/score**（UB 端口约 12 B/score：读两个 fp32 tile
  8 B + 写两个 f16 tile 4 B）。
- 反解：128 线程 × `float2`/指令 ⇒ 2048 score/AIV 需 8 轮 × 约 12 条指令 ≈ 96 条指令，实测 1485 周期
  ⇒ **约 15 周期/指令**，说明当前形态受**指令级依赖/延迟**限制而非算力限制（向量 ALU 峰值下应有 3–5× 余量）。
- **优化阶梯（Stage 4 A/B，均带判定阈值）**：
  ① 加宽向量宽度：`mq` 64→128 会把 `float2` 提升到 `float4`（每 lane 4 元素）→ 目标 ≤0.14 ns/score；
     但 `Mq=128` 时 `dq_c` 需 96 KiB，与 `Nk=96` 组合的 L0C = 319488 B（**122%，超**），只能配 `Nk=48`
     （208896 B = 80%），而 `Nk=48` 把 GM→L1 填充总量从 116.6 GB 翻到 233 GB（59.5 → 119 ms）——
     **净亏，弃选**（除非 epilogue 收益 > 2× 填充损失，届时再评估）；
  ② 显式 SIMD 内建：`T.simd.vexpdif(hi, lo) = exp(hi − lo)`（本工具链导出，见
     `tilelang/ascend/language/simd.py`「Fused exp-sub」）把「乘 scale + 减 lse + exp」合成一条 SFU 指令，
     省 2 条向量指令/score → 目标 ≤0.16 ns/score；**该形态在混合 kernel 内的可编译性未核验**（`T.simd`
     与 `T.SimtVF` 属两套 frame），故列为 A/B 而非主选；
  ③ 每线程处理 2 个独立 score（手工展开）以打断依赖链 → 目标 ≤0.13 ns/score。
  判定阈值：A/B 候选的 epilogue 单价 ≤ 主选 × 0.85 才翻转；实测数据回写本节与 §1.6.4。

**布局决策结论**：选定候选 1——核内布局为 **(key, query)**、**向量化轴 = query**（lane 映射到 `Mq=64` 的最内轴），
中间块以 `(Nk,Mq)` 存在 L0C/UB；`dS` **只有一份** `(Nk,Mq)` 的 L1 tile，`dQ` 通过 `trans_A=True` 消费它；
向量段为半区（48×64）、由 `__mix__(1,2)` 的两个 AIV 各持一半。该结论已同步到 §3.3 伪代码、§4.5 预算与 §6 循环结构。

**主选 / 备选与实验裁决计划**：

- **主选（进 Stage 3）**：`block_n=96, block_m=64`。L0C 221184 B（84.4%）、L1 311296 B（59.4%，双版本）、
  UB 166912 B（65.7%）、L0A 49152 B（75%）、L0B 61440 B（93.75%）；每迭代端口 325625 B、31.4 FLOP/字节；
  原型实测 **3.47 µs/迭代** → 外推 **175 ms**。
- **备选（≤1 个）**：`block_n=112, block_m=64` + `dq_c` 拆两半（(64,96) 各 24 KiB）。L0C
  `dk 86016 + dv 57344 + st 28672 + dpt 28672 + dq 24576 = 225280 B`（86.0%）；
  每 query 行填充摊到 112 key 行 → MTE2 总量 98.8 GB（vs 主选 116.6 GB，−15%），迭代数 1 198 000（−15%）
  （每字节 FLOP 与主选持平 ≈31，收益全来自迭代数）；
  代价：`dQ` 拆成两个 K 半程 GEMM（B 需两块 zero-origin 的 `kT` L1 tile，`kT_lo/kT_hi` 各 (96,112)=43 KiB，
  L1 增至 397 KiB = 76%），且两次原子 store。
  **裁决阈值**：备选 latency ≤ 主选 × 0.90 才翻转；按 shape 分片经 `lru_cache` 工厂分派（`S_kv ≥ 7168` 用备选）。
- **实验裁决计划**：代表 shape = smoke(512) 与 `ds-v3-4k`、`ds-v3-b2-8k`；指标 = kernel `Task Duration(us)` +
  msprof 的 `aic_cube_ratio` / `aiv_vec_ratio` / `aic_mte2_active_bw`；实测数据回写本节与 §1.6.4；
  Stage 4 的 A/B 清单 = ①Nk/Mq 网格（96/64、112/64、128/48）②epilogue 优化阶梯 ③`trans_B=False` 消 qT/doT。

#### 1.6.4 实测证据：v1 的瓶颈分解与本版原型的代价（本轮新增）⭐

> 三个可复现脚本（均在本机单卡跑小 shape，命令见各脚本头部与 §3.5.4）：
> `design_probe_bwd_cost.py`（分段单价）、`design_probe_l0_gemm.py`（L0 路径四类 trans）、
> `design_probe_bwd_proto.py`（新结构端到端原型）。

**A. v1 内层迭代的分段单价**（`Nk=Mq=64`，grid=28，块内 R=400 次迭代，取 3 次最小值）：

| 分段 | 单价 (µs/迭代) | 占 v1 整链 |
|------|---------------|-----------|
| `mte2`：`q,qT,do,doT` + lse/delta 的 GM→L1 | 1.176 | 20% |
| `gemm5`：5 个 NT GEMM（操作数常驻 L1） | 0.587 | 10% |
| `l0c2ub`：2 个 L0C→UB `dual_copy` | 0.173 | 3% |
| `ub2l1`：3 个 UB→L1 `dual_copy` | 0.231 | 4% |
| `epi0`：**合并写** epilogue（`p16`+`dsT`） | 0.825 | 14% |
| `epi1`：epi0 + 跨步写 `ds_u[j,i]`（**v1 现形态**） | **4.302** | 72% |
| `epi2`：epi0 + 第二趟（跨步读、合并写） | 2.469 | 41% |
| `chain_notr`：整链但去掉跨步写 | 2.196 | — |
| **`chain`：v1 内层整链** | **5.950** | 100% |

**与生产实测对齐**：v1 在 `ds-v3-b2-8k` 上 T_op 455.3 ms / 2 113 536 次迭代 / 28 核 = **6.03 µs/迭代**，
探针复现 5.95 µs（差 1.3%）→ 该分解可信。

**B. 由 A 得到的根因（回答「为什么 5 个 GEMM 没让 Cube 成为瓶颈」）**

1. **向量侧的跨步转置 scatter 独占 63%**：`ds_u[j,i]` 按 lane=query 写一个 `(Mq,Nk)` 的 f16 tile，
   步长 32 个 f16，bank 冲突 + 循环整体退化（不能向量化），单这一句 = 3.75 µs/迭代。
   它是 v1 为「L1 路径 NT-only」付出的代价（§0.4 #6 的误判）。
2. **Cube 本身并不慢，它只是在等**：cube 忙时比例 8.3% × 455.3 ms = 37.8 ms ≈ 14.34 TFLOP ÷ 378 TF/s，
   说明 mad 在满速运行，空闲来自**串行依赖 + 无重叠**（`num_stages=1`，每迭代「装载 → GEMM → fixpipe →
   向量 → UB→L1 → GEMM → 原子写」全串行）。
3. **L1 填充已与 Cube 同量级**：`mte2` 1.176 µs vs `gemm5` 0.587 µs（2×）。单核 MTE2 实测 70 GB/s
   （`CONST-950-gemm-mte2-burst`：128 B 行段口径），每 query 行 1280 B × 128 个 key 块 = 164 KB/行的放大。
4. **tile 太小**：`Nk=64` 把 1280 B 的每行填充摊到 64 个 key 行上；L0C 此时只用 62.5%，本可放大。

**C. v1 的根因归属（为什么 v1 没做到）**：v1 的设计目标错位——它把「L1 端口流量」当作唯一地板项
（§1.6.0 旧结论「L1 端口操作数流是地板项」），因此用「缩小 tile + 单阶段 + 转置副本」换取 L1 占用；
而实测地板项是**向量 scatter + 每行填充 + 无重叠**三者叠加，其中最大的一项来自 v1 自己引入的向量段。

**D. 新结构（C2）的端到端原型实测**

| 测试 | shape | 结果 |
|------|-------|------|
| 数值（causal） | B=1, S=384, H=4 | **PASS**（dq/dk/dv 最大差 7.8e-3 / 7.8e-3 / 3.1e-2，bf16 口径） |
| 数值（非 causal） | B=1, S=384, H=4 | **PASS**（3.9e-3 / 2.0e-3 / 3.9e-3） |
| 单价（causal） | B=1, S=2048, H=8，2904 次迭代 | **3.469 µs/迭代**（`num_stages=1`，可复跑值） |
| 单价（causal, 2 版本） | 同上 | 3.391 µs/迭代（−2.2%，在块间波动内）⇒ **`num_stages=2` 无实质重叠收益** |

外推主导 case：1 415 000 次迭代 × 3.469 µs ÷ 28 核 = **175 ms**（相对 455.3 ms = **2.6×**）。
其中 cube 理论 0.88 µs/迭代（40.5 ms）→ 迭代内占比 25%（v1 为 8.3%）。

**E. 结构层的两条硬约束（本轮实测，写进 §0.5/§0.6/§9）**

1. **L0 路径与 L1 路径不能在同一 kernel 内混用**：把 `dQ` 单独改 L0、其余 4 个留 L1 时，`dk/dv` 出现
   4.8e-1 / 3.1e-2 的错误（`ascend_gemm_l1` 模板内部会用 L0A/L0B，与用户显式分配的 L0 缓冲互相覆盖）；
   5 个 GEMM 全部改显式 L0 后同一测试全 PASS。
2. **`T.copy(UB, L1, transpose=True)` 不可编译**（`Ascend device compilation failed`）：UB→L1 的转置
   不是可用形态，故「把转置挪到 UB→L1 搬运」这条路被排除；可用形态只有 `trans_A` 的 L1→L0A 转置
   （`asc_copy_l12l0a_transpose`）。

**F. 80% Cube MFU 的可行性边界（定量结论）**

| 量 | 值 | 来源 |
|----|----|------|
| 迭代内 Cube（受计算项约束） | 0.88 µs | 14.48 TFLOP/358 TF/s ÷ 1.415e6 迭代 |
| 迭代内 GM→L1 填充（MTE2） | 1.18 µs | 探针 A |
| 迭代内向量 epilogue | 1.24 µs（6214 score） | 探针 A（0.20 ns/score） |
| 三者之和（假设 fixpipe/UB→L1/原子/同步全部免费、且完全不重叠） | **3.30 µs** | — |
| 达到 80% Cube MFU 所需 | ≤1.00 µs/迭代（T_op ≤ 50.6 ms） | 14.48 T ÷ (0.8 × 358 TF/s) = 50.6 ms；× 28 ÷ 1.415e6 |

**结论：50.6 ms 远低于「Cube + 填充 + 向量」三项之和（3.30 µs/迭代），即使把搬运与同步全部优化掉也不可达**
（唯一出路是把每 query 行的 1280 B 填充减半并让 epilogue 再快 2×，前者需 `trans_B=False` 消掉 `qT/doT`，
后者需 §1.6.3 的优化阶梯；两者都落地后迭代约 2.0–2.1 µs → **105–115 ms，Cube 占比约 42%**）。
因此本版的设计目标是：**Stage 3 交付 ≤227 ms（实测原型外推 175 ms，1.3× 余量）**，
Stage 4 的优化目标是 **≤115 ms（≈4×）/ Cube 占比 ≥40%**，并把「80% Cube MFU 不可达」作为
**已量化的边界结论**登记在 §9.1。这与调度方给的目标（≥2× 必达、接近 Cube roofline 尽力）一致：
本版把 455.3 ms → 175 ms 的路径做成**已实测**，把 175 → 115 ms 的路径做成**已定位 + 有阈值**的 A/B 清单。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式：混合（Cube + Vector），且 Cube 侧走显式的 L0 路径（Expert 形态）。**
（平台 `programming_modes.choices` 的 `hybrid`；`migration_default = simd`，偏离理由见 2.2。）

### 2.2 选型理由

- 算子含 5 个矩阵乘（每迭代 `2·96·64·(3·192+2·128) = 11.9 MFLOP`），**必须落 Cube**；若走纯向量路径，
  56 个 Vector 核的 lane 级乘加比 Cube 分形低 1–2 个数量级（§1.6.0 R4 C6）。
- 同时每迭代必须做 `exp`、掩码、cast 等逐元素工作（96×64 = 6144 个 score），**必须落 Vector（AIV）**。
- **为什么 Cube 侧用显式 L0 而不是 L1 操作数路径**（v2 的关键依据）：
  ① 只有 L0 路径支持 `transpose_A=True`（`gemm_mad.py:162` 的 L1 路径断言只允许 NT），而 `trans_A` 是
     消掉 v1 的向量 scatter 的唯一低成本手段（§0.6 R3）；
  ② L0 路径是平台**已核验**的 GEMM 形态（`PL-950-gemm-tiles` 八项全过，含双版本与显式 L0 缓冲）；
  ③ 两条路径**不能混用**（§1.6.4 E1 实测），故必须整体二选一。
- 两侧在同一 kernel 内交替（中间块 `P/dS` 不落 GM），平台支持该形态：`T.Kernel` 单核网格 + 自动调度把 op
  分派到 AIC/AIV（`AssignCore`/`ResolveCore`/`InsertSync`），已核验形态见 `PL-950-mixed-epilogue.py`。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_shared`（UB，向量侧）、`T.alloc_l1`（GM 装载与 GEMM 操作数暂存）、**`T.alloc_l0a/l0b`（显式操作数）**、`T.alloc_l0c`（fp32 累加器） |
| 计算方式 | Cube：`T.gemm(a_l0a_region, b_l0b_region, c_l0c, transpose_A=…, transpose_B=…, clear_accum=…)`；Vector：`with T.SimtVF(threads=128): for i, j in T.Parallel(48, 64)` |
| 执行体 | **不写 `T.Vector`**。半区向量段由 `T.SimtVF(threads=128)` 表达；副块 id 由自动调度合成，用户侧只在 kernel 体取一次 `asc_get_sub_block_id()` 作 key 行偏移 |
| 同步 | 自动（`allow_autoschedule=True`，AutoSchedule + InsertSync）→ §7 |
| 跨单元数据 | L0C→UB 走**半区** `dual_copy`（v1 运行期实测：整 tile `T.copy` 只落一个 AIV，不可用）；UB→L1 走半区 `dual_copy`（`InsertNd2Nz` 改写为 NZ scatter）；L1→L0 走 `T.copy`（含 `trans_A` 时的自动转置搬运）；L0C→GM 走 `T.copy` + store-atomic |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入为 §1.6.1 优化后公式；每个步骤与 §1.6.2 的向量化结论一致。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `s^T = k_s · q_s^T` | L0 GEMM，A=`k_s (96,192)`、B=`q_s (64,192)`，C=`st_c (96,64)` fp32，(trans_A=F, trans_B=T) |
| 2 | `dP^T = v_s · do_s^T` | L0 GEMM，A=`v_s (96,128)`、B=`do_s (64,128)`，C=`dpt_c (96,64)` |
| 3 | `p^T = exp(s^T·scale − lse[j]) ⊙ mask` | 向量（lane=query，半区 48×64），A2 运行时块级特化：全可见块省掩码 |
| 4 | `pT = f16(p^T)`；`dsT = f16(p^T ⊙ (dP^T − delta[j]))` | 向量 + cast；`p^T` 只算一次 `exp`（CSE）；**只 2 个输出 tile** |
| 5 | `dV = pT · doT^T` | L0 GEMM，A=`pT_s (96,64)`、B=`doT_s (128,64)`，C=`dv_c (96,128)` 累加 |
| 6 | `dK = scale · dsT · qT^T` | L0 GEMM，A=`dsT_s (96,64)`、B=`qT_s (192,64)`，C=`dk_c (96,192)` 累加（`scale` 在块尾乘） |
| 7 | `dQ = dsT^T · k^T`（`transpose_A`） | L0 GEMM，A=`dsT_s (96,64)` **(trans_A=T)**、B=`kT_s (192,96)`，C=`dq_c (64,192)` 每迭代清零 |
| 8 | `dq_acc += dQ` | store-atomic：`set_atomic("add","float32")` + `T.copy(dq_c → dq_acc)` |

（步骤 7 与步骤 6 共用同一份 `dsT_s`，这就是源算子 `transpose_A=True` 的复用点。）

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API（`tilelang.ascend.language`） | 参数 | 模式 |
|------|----------|--------------------------------------------|------|------|
| 0 | 任务映射 | `T.Kernel(grid)` 一维核网格 | `grid = H*B*ceil(S_kv/96)`；`T.Kernel` 不接受 `threads` | — |
| 1 | `k/v/kT` 常驻 | `T.copy(GM region, l1_buf)`（`kT` 加 `transpose=True`）；dst 用全 allocation extent | `k_s (96,192)`、`v_s (96,128)`、`kT_s (192,96)` | MTE2 |
| 2 | `q/do` 双方向装载 | 同上（两份：ND 与 `transpose=True`） | `q_s (64,192)`、`qT_s (192,64)`、`do_s (64,128)`、`doT_s (128,64)` | MTE2 |
| 3 | `lse/delta` | `T.copy(lse[b,h,m0:m0+64], lse_u)` | 一维 fp32 区段 | MTE2 |
| 4 | L1→L0 暂存 | `T.copy(l1_buf_region, l0a_or_l0b_region)`（`trans_A=True` 的 A 由编译器生成转置搬运） | 每次 GEMM 前一次；region 原点必须为 0 | MTE1 |
| 5 | 5 个 GEMM | `T.gemm(a_l0a, b_l0b, c_l0c, transpose_A=…, transpose_B=…, clear_accum=…)` | 见 §3.1；A/B 在 L0A/L0B，C 在 L0C | Cube |
| 6 | L0C→UB | `T.dual_copy(st_c, st_u)` / `T.dual_copy(dpt_c, dp_u)` | 2:1 M 分片（48×64 每 AIV），fp32 同 dtype | AIC fixpipe |
| 7 | `exp/mask/dS/cast` | `with T.SimtVF(threads=128): for i, j in T.Parallel(48, 64)` | `T.exp`、`T.if_then_else`、`T.cast`；A2 为运行时块级 `if` 两支 | Vector |
| 8 | UB→L1 | `T.dual_copy(p16_u, pT_s)`、`T.dual_copy(dsT_u, dsT_s)` | 2:1 M 分片；由 `InsertNd2Nz` 改写为 NZ scatter | MTE |
| 9 | `dq_acc` 累加 | `T.set_atomic("add","float32")` → `T.copy(dq_c, dq_acc[rows,h,:])` → `T.set_atomic_none()` | L0C fp32 → GM fp32 | AIC fixpipe |
| 10 | `dk/dv` 写回 | `T.dual_copy(dk_c, tail_u)` → 向量 `cast`（`dk` 乘 `scale`）→ `T.copy(out_k, GM region)` | 半区；尾部由 OOB pass 裁到 `S_kv` | AIC + Vector |
| 11 | 多缓冲 | `T.annotate_buffer_versions({buf: 2})` | 仅 L1 每迭代缓冲（§6.3） | — |
### 3.3 计算伪代码

> 以下结构**已在设备上跑通并与 fp64 golden 数值一致**（`history_version/design_probe_bwd_proto.py`，
> `S=384` 与 `S=2048`，causal/非 causal 两组，见 §1.6.4 D）。

```python
BLOCK_N, BLOCK_M = 96, 64          # keys / queries
HN = BLOCK_N // 2                  # 48：每个 AIV 的 key 行数
D_QK, D_V = 192, 128
NT_PER_B = ceildiv(S_KV, BLOCK_N)              # 每 request 的 key 块数（静态）
SHIFT = S_KV - S_Q

@T.prim_func
def _bwd_main(q, k, v, do, lse, delta, cu_seqlens_q, cu_seqlens_kv, dq_acc, dk, dv):
    with T.Kernel(H * B * NT_PER_B) as bid:
        h = bid // (B * NT_PER_B)
        tile = bid % (B * NT_PER_B)
        b = tile // NT_PER_B
        n0 = (tile % NT_PER_B) * BLOCK_N
        sid = T.call_extern("int64", "asc_get_sub_block_id")   # 0/1，本 AIV 的 key 半区
        # ---- L1：key 侧常驻（每 launch） ----
        k_s  = T.alloc_l1((BLOCK_N, D_QK), dt)
        v_s  = T.alloc_l1((BLOCK_N, D_V), dt)
        kT_s = T.alloc_l1((D_QK, BLOCK_N), dt)
        # ---- L1：每迭代（2 版本，§6.3） ----
        q_s  = T.alloc_l1((BLOCK_M, D_QK), dt);  qT_s = T.alloc_l1((D_QK, BLOCK_M), dt)
        do_s = T.alloc_l1((BLOCK_M, D_V), dt);   doT_s = T.alloc_l1((D_V, BLOCK_M), dt)
        pT_s = T.alloc_l1((BLOCK_N, BLOCK_M), dt); dsT_s = T.alloc_l1((BLOCK_N, BLOCK_M), dt)
        # ---- L0：显式操作数（单版本，L0B 余量 4 KiB） ----
        a_k = T.alloc_l0a((BLOCK_N, D_QK), dt)   # K-major：k/v/pT/dsT
        a_t = T.alloc_l0a((BLOCK_N, BLOCK_M), dt)  # MN-major：dsT（dQ 的 trans_A）
        b_lo = T.alloc_l0b((BLOCK_M, D_QK), dt)  # q / do
        b_hi = T.alloc_l0b((D_QK, BLOCK_N), dt)  # qT / doT / kT
        # ---- L0C 累加器 ----
        st_c = T.alloc_l0c((BLOCK_N, BLOCK_M), "float32")
        dpt_c = T.alloc_l0c((BLOCK_N, BLOCK_M), "float32")
        dq_c = T.alloc_l0c((BLOCK_M, D_QK), "float32")
        dk_c = T.alloc_l0c((BLOCK_N, D_QK), "float32")   # 跨迭代
        dv_c = T.alloc_l0c((BLOCK_N, D_V), "float32")    # 跨迭代
        # ---- UB：每 AIV 半区（2 版本） + 块尾 ----
        st_u = T.alloc_shared((HN, BLOCK_M), "float32")
        dp_u = T.alloc_shared((HN, BLOCK_M), "float32")
        p16_u = T.alloc_shared((HN, BLOCK_M), dt)
        dsT_u = T.alloc_shared((HN, BLOCK_M), dt)
        lse_u = T.alloc_shared((BLOCK_M,), "float32"); dlt_u = T.alloc_shared((BLOCK_M,), "float32")
        tail_u = T.alloc_shared((HN, D_QK), "float32"); tailv_u = T.alloc_shared((HN, D_V), "float32")
        out_k = T.alloc_shared((HN, D_QK), dt); out_v = T.alloc_shared((HN, D_V), dt)

        koff = b * S_KV + n0
        # 全 extent 装载；越界尾部由 AscendInsertOOBPadding 改写为 min() clamp + fill 0
        T.copy(k[koff : koff + BLOCK_N, h, :], k_s)
        T.copy(v[koff : koff + BLOCK_N, h, :], v_s)
        T.copy(k[koff : koff + BLOCK_N, h, :], kT_s, transpose=True)

        loop_st = T.max(0, (n0 - SHIFT) // BLOCK_M)
        loop_ed = ceildiv(S_Q, BLOCK_M)
        for kq in T.Pipelined(loop_st, loop_ed, num_stages=1):     # 版本由 annotate 控制
            m0 = kq * BLOCK_M
            qoff = b * S_Q + m0
            T.copy(q[qoff : qoff + BLOCK_M, h, :], q_s)
            T.copy(do[qoff : qoff + BLOCK_M, h, :], do_s)
            T.copy(q[qoff : qoff + BLOCK_M, h, :], qT_s, transpose=True)
            T.copy(do[qoff : qoff + BLOCK_M, h, :], doT_s, transpose=True)
            T.copy(lse[b, h, m0 : m0 + BLOCK_M], lse_u)
            T.copy(delta[b, h, m0 : m0 + BLOCK_M], dlt_u)

            # ---- 2 个 L0 GEMM（每迭代清零）----
            T.copy(k_s, a_k)
            T.copy(q_s, b_lo)
            T.gemm(a_k, b_lo, st_c, transpose_B=True, clear_accum=True)          # s^T
            T.copy(v_s, a_k[0:BLOCK_N, 0:D_V])
            T.copy(do_s, b_lo[0:BLOCK_M, 0:D_V])
            T.gemm(a_k[0:BLOCK_N, 0:D_V], b_lo[0:BLOCK_M, 0:D_V], dpt_c,
                   transpose_B=True, clear_accum=True)                            # dP^T
            # ---- 半区落地 + 向量 epilogue（只写 2 个 tile）----
            T.dual_copy(st_c, st_u); T.dual_copy(dpt_c, dp_u)
            if (n0 + BLOCK_N <= S_KV) & (m0 + BLOCK_M <= S_Q) & (n0 + BLOCK_N - 1 <= m0 + SHIFT):
                with T.SimtVF(threads=128):            # A2 快路：整块可见
                    for i, j in T.Parallel(HN, BLOCK_M):
                        p = T.exp(st_u[i, j] * SCALE - lse_u[j])
                        p16_u[i, j] = T.cast(p, dt)
                        dsT_u[i, j] = T.cast(p * (dp_u[i, j] - dlt_u[j]), dt)
            else:
                with T.SimtVF(threads=128):            # 掩码臂（对角块 / 尾块）
                    for i, j in T.Parallel(HN, BLOCK_M):
                        gi = n0 + T.cast(sid, "int32") * HN + i
                        gj = m0 + j
                        vis = (gi < S_KV) & (gj < S_Q) & (gi <= gj + SHIFT)
                        p = T.exp(st_u[i, j] * SCALE - lse_u[j])
                        pv = T.if_then_else(vis, p, T.cast(0, "float32"))
                        p16_u[i, j] = T.cast(pv, dt)
                        dsT_u[i, j] = T.cast(pv * (dp_u[i, j] - dlt_u[j]), dt)
            T.dual_copy(p16_u, pT_s); T.dual_copy(dsT_u, dsT_s)
            # ---- 3 个 L0 GEMM：dV/dK 跨迭代累加，dQ 每迭代清零 ----
            T.copy(pT_s, a_k[0:BLOCK_N, 0:BLOCK_M])
            T.copy(doT_s, b_hi[0:D_V, 0:BLOCK_M])
            T.gemm(a_k[0:BLOCK_N, 0:BLOCK_M], b_hi[0:D_V, 0:BLOCK_M], dv_c,
                   transpose_B=True, clear_accum=(kq == loop_st))
            T.copy(dsT_s, a_k[0:BLOCK_N, 0:BLOCK_M])
            T.copy(qT_s, b_hi[0:D_QK, 0:BLOCK_M])
            T.gemm(a_k[0:BLOCK_N, 0:BLOCK_M], b_hi[0:D_QK, 0:BLOCK_M], dk_c,
                   transpose_B=True, clear_accum=(kq == loop_st))
            T.copy(dsT_s, a_t)                       # L1→L0A 的转置搬运（MTE1）
            T.copy(kT_s, b_hi)
            T.gemm(a_t, b_hi, dq_c, transpose_A=True, transpose_B=True, clear_accum=True)
            # ---- dQ 原子累加（L0C fp32 → GM fp32）----
            T.set_atomic("add", "float32")
            T.copy(dq_c, dq_acc[qoff : qoff + BLOCK_M, h, :])
            T.set_atomic_none()

        # ---- 块尾：dk = cast(dk_c*SCALE)、dv = cast(dv_c)，越界 key 行由 OOB clamp 裁掉 ----
        T.dual_copy(dk_c, tail_u); T.dual_copy(dv_c, tailv_u)
        with T.SimtVF(threads=128):
            for i, d in T.Parallel(HN, D_QK):
                out_k[i, d] = T.cast(tail_u[i, d] * SCALE, dt)
            for i, d in T.Parallel(HN, D_V):
                out_v[i, d] = T.cast(tailv_u[i, d], dt)
        kown = koff + T.cast(sid, "int32") * HN
        T.copy(out_k, dk[kown : kown + HN, h, :])
        T.copy(out_v, dv[kown : kown + HN, h, :])
```

注（与原型的差异与约束，逐条可复核）：

1. **GEMM 的 M/N/K 全静态**：L0 路径从 **allocation 形状**取 `node_M/node_N/node_K`（`gemm_op.py:_l0_gemm_call`），
   而 region extent 可以是静态切片（`a_k[0:96,0:128]` 等），因此 L0 缓冲可以按「最大形状分配 + 切片使用」复用。
   **region 的原点必须为 0**（`gemm_mad.py:_l0_operation_extents` 的 `require_zero_origin`），故
   `kT_s` 若需拆半（备选方案的 112/64）必须用两块独立 L1 缓冲。
2. **尾块由「全 extent 装载 + 掩码 + 平台 OOB clamp」三件套处理**：被补 0 的行/列再由掩码臂的
   `gi < S_KV` / `gj < S_Q` 置零（补 0 只让 `s=0`，而 `exp(0·scale − lse) ≠ 0`，故掩码必须保留这两项）。
3. **无 `T.Vector`/`sid` 帧**：只取一次执行体 id 做 key 行偏移（v1 运行期实测该取法确定且正确）。
4. `dq_acc` 行偏移为 `b*S_Q + m0`（dense 下等价于 `q_start + m0`）。
5. `b_hi` 被 `qT/doT/kT` 复用：**`kT` 必须在 `doT/qT` 之后重灌**（原型首版曾因顺序错误得到 dQ 全错，
   见 §9.2 的「常见错误」表）。

### 3.4 API 可行性确认

| API | 来源 / 核验状态 |
|-----|----------------|
| `T.Kernel(grid)`（一维，无 `threads`） | `ascend/language/kernel.py:Kernel`（源码确认）；PL-950-gemm-tiles 用一维网格 |
| `T.alloc_l1` / `T.alloc_l0a` / `T.alloc_l0b` / `T.alloc_l0c` / `T.alloc_shared` | `ascend/language/allocate.py`；PL-950-gemm-tiles（已核验的双版本 + 显式 L0 形态）；**本轮原型设备码含 5 个 `asc_mmad` + 4 个 `asc_copy_l12l0a/l12l0b`** |
| `T.gemm(a_l0a, b_l0b, c_l0c, transpose_A=…, transpose_B=…, clear_accum=…)` | `gemm_op.py:_dialect_gemm_call` → `GemmMAD._lower_l0`；四种 trans 组合的布局推断见 `gemm_mad.py:126-134`；**本轮探针 2 实测 (T,T)/(F,F)/(F,T) 三种组合编译 + 数值 PASS** |
| `T.copy` GM→L1（含 `transpose=True` dn2nz） | `copy_op.py` + PL-950-gemm-tiles 的 `trans_b` 分支；v1 设备码 `asc_copy_gm2l1_nd2nz` / `dn2nz` |
| `T.copy` L1→L0（含 `trans_A` 的转置搬运） | 原型设备码 `asc_copy_l12l0a` / `asc_copy_l12l0b` / **`asc_copy_l12l0a_transpose`**（`trans_A=True` 的 A），数值 PASS |
| `T.dual_copy` L0C→UB（2:1 M 分片） | `copy_op.py:dual_copy` 文档 + v1 运行期 L0 全过（13/13）；原型 `asc_copy_l0c2ub(..., dual_dst_mode=1)` |
| `T.dual_copy` UB→L1（2:1 M 分片） | 同上（`InsertNd2Nz` 改写成 NZ scatter）；原型设备码 3 个 `ascend_nd2nz_scatter`（v1 为 3 个，本版 2 个） |
| `T.set_atomic("add","float32")` + L0C→GM `T.copy` | `mode.py:set_atomic` + PL-950-atomic-splitk；v1 运行期实测加法语义成立（L0-11） |
| `T.SimtVF(threads=128)` + `T.Parallel(48,64)` + `T.exp` + `T.if_then_else` + `T.cast` | PL-950-row-primitives / PL-950-attention（已核验）；本轮探针 `epi0`/`epi1`/`epi2` 三种形态均编译并实测 |
| `sid = T.call_extern("int64","asc_get_sub_block_id")` | v1 运行期实测：两 AIV 各得 0/1、确定（`repro/CONST-950-aiv-sid-capture.py`） |
| `T.annotate_buffer_versions` | `annotations.py` + programming.md「同步与缓冲版本」；**本轮实测对本结构无实质收益（3.391 vs 3.469 µs），故仅作声明** |
| `T.Pipelined(start, stop, num_stages=1)` 动态界 | v1 设备码 `for (kq = <运行时下界>; kq < <上界>; ++kq)` |
| `T.copy(UB, L1, transpose=True)` | **不可用**：`Ascend device compilation failed`（§1.6.4 E2 实测） |
| `T.Vector(vector=2) as sid` | **不可用**：与 `dual_copy`/UB→L1/GEMM 混用触发 `InternalError … dup sid`（v1 反例） |
| L1 操作数与显式 L0 缓冲**混用** | **不可用**：`ascend_gemm_l1` 模板内部占用 L0A/L0B，与用户显式分配互相覆盖（§1.6.4 E1 实测，`dk/dv` 错 4.8e-1） |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束（来源） | 是否涉及 | 处理方案 |
|--------------|---------|----------|
| `T.Kernel` 只接受一维核网格、无 `threads`（programming.md） | 是 | 一维 `grid = H*B*ceil(S_kv/96)`，块内 div/mod 解码（§0.6 R1） |
| **L1-GEMM 路径只支持 NT**（`gemm_mad.py:162` 断言） | 是→**规避** | 全部 GEMM 改走 L0 路径（四种 trans 组合齐全），不再使用 L1 路径（§0.6 R6） |
| **L0 路径与 L1 路径不能混用**（本轮实测） | 是 | 5 个 GEMM 全部显式 L0；kernel 内不出现 `T.gemm(l1, l1, l0c)` |
| L1-GEMM 的 M/N/K 取自 region extent 且必须静态（v1 结论，仅适用 L1 路径） | 否 | 本版走 L0 路径：`node_M/N/K` 取 allocation 形状，region extent 可为静态切片（`require_zero_origin` 仍约束原点为 0） |
| L0C 是唯一可累加层且必须带 fractal layout | 是 | `dk_c/dv_c/dq_c/st_c/dpt_c` 五个累加器，共 216 KiB（§4.6） |
| L0C 容量 256 KiB（实测上界约 244 KiB：249856 B 时设备报 507015） | 是 | 主选 221184 B（84.4%），备选 225280 B（86.0%）；**任一配置超过 240 KiB 即视为不可行**（§9.1 风险 2） |
| L0A/L0B 各 64 KiB，无版本余量 | 是 | L0A 48 KiB（75%）、L0B 60 KiB（93.75%）；L0 一律单版本（§4.6） |
| L0C→UB 不支持 dtype 转换（同 dtype 断言） | 是 | L0C→UB 保持 fp32 半区，cast 在 UB 内用 `T.cast` 完成 |
| 整 tile `T.copy(L0C→UB)` 只落一个 AIV（v1 运行期实测） | 是 | 所有 L0C↔UB 交叉用半区 `dual_copy`（§0.6 R2） |
| 线程域只在 `T.SimtVF` 内；显式 `T.Vector` 不可编译 | 是 | 所有逐元素工作包在 `T.SimtVF(threads=128)`；只取一次 subblock id |
| 原子存储支持 dtype 集合含 float32（`mode.py`） | 是 | `dq_acc` 为 fp32 ✓ |
| 未认证项：手工跨核 flag、循环内复杂同步、跨 block barrier | 是 | 不写手工 flag，全部依赖 AutoSchedule + InsertSync（§7） |
| grid 上限 65535（`TRAP-950-grid-over-65535`） | 是 | 主选 grid 最大 22016（`ds-v3-b2-8k`）< 65535 ✓；factory 侧加断言（§9.1 风险 12） |
| 动态 shape | 部分 | `S_q/S_kv/B/H/D` 编译期常量；`cu_seqlens_*` 只保留签名兼容 |

#### 3.5.2 参考实现差异说明

| 差异项 | 参考实现（GPU TileLang） | 本项目（TileLang ascend） | 转换方案 |
|--------|------------------------|--------------------------|---------|
| Kernel 结构 | `T.Kernel(num_tiles, heads, threads=256)` 二维网格 | 一维核网格 + AIC/AIV 自动分派 | §0.6 R1/R4 |
| 矩阵 API | `T.gemm(fragment)`，C 在寄存器，支持 `transpose_A` | `T.gemm(a_l0a, b_l0b, c_l0c)`，显式 L0，四种 trans | §0.6 R3/R6 |
| 累加器 | `T.alloc_fragment` | `T.alloc_l0c` | §0.6 R2 |
| 片上层级 | SMEM + register file | L1（装载/暂存）+ L0A/L0B（操作数）+ L0C（累加）+ UB（向量） | §4.4 |
| 原子累加 | `T.atomic_add` 逐元素 | `set_atomic` + 整块 store | §0.6 R5 |
| 指数 | `T.exp2`（log2 域） | `T.exp`（自然域） | §1.6.1 A1 |
| 掩码 | 逐元素 `if_then_else` | 同 + 运行时块级特化 | §1.6.1 A2 |
| 流水 | `T.Pipelined(num_stages=2)`（cp.async） | `annotate_buffer_versions` + `num_stages`（实测零收益） | §6.3 |
| 尾块 | `T.if_then_else` 逐元素守卫 + 区段截断 | 全 extent + 平台 OOB clamp/补 0 + 掩码置 0 | §5.4 |
| `dS` 的转置使用 | `transpose_A=True`（mma 直接支持） | 同（L0 路径 `trans_A`），转置落在 L1→L0A 搬运 | §0.6 R3 |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `.agents/platforms/toolchains/ascend/knowledge/repro/PL-950-gemm-tiles.py` | **高（本版主形态的直接依据）** | 显式 L0 路径（`alloc_l0a/l0b` + `T.copy(L1→L0)` + `T.gemm`）、`T.annotate_buffer_versions`、K 分块与尾块、`trans_b` 的装载方向 |
| `.agents/platforms/toolchains/ascend/knowledge/repro/PL-950-mixed-epilogue.py` | 高（Cube GEMM + Vector 同 kernel） | 混合模式写法、L0C→UB 落地、同 dtype 约束 |
| `examples/TileOPs/tileops/kernels/attention/mla_dense_bwd/_mla_dense_bwd_kernels.py` | 高（同一函数的 GPU 原样抽取） | 语义与分块结构、`dk` 的 `scale`、掩码谓词、`transpose_A` 的复用意图 |
| `.agents/platforms/toolchains/ascend/knowledge/repro/PL-950-atomic-splitk.py` | 中（store-atomic 累加） | `set_atomic("add","float32")` + `T.copy(C)` 的配对与清零契约 |
| `.agents/platforms/toolchains/ascend/knowledge/repro/PL-950-row-primitives.py` | 中（向量 exp / 逐元素） | `T.exp` 的可行形态与精度 |
| `.agents/platforms/toolchains/ascend/knowledge/repro/PL-950-conversion-layout.py` | 中（SIMT 转换 + 列广播） | `T.cast` 的 ±inf/次正规行为、lane 私有读 |
| `examples/paged_cache_write/_paged_cache_write_kernel/DESIGN.md` | 中（同平台设计文档范式） | 章节组织、分核三要素的写法、风险登记口径 |
| `.agents/platforms/toolchains/ascend/kernel_template.py` | 中（kernel 骨架） | 一维 `T.Kernel`、`T.SimtVF` 包逐元素、静态 `if tail` 尾块分支 |
| `history_version/design_probe_bwd_proto.py` | —（本版原型） | 新结构的可跑通形态与实测单价（§1.6.4 D） |
| `history_version/design_probe_bwd_cost.py` / `design_probe_l0_gemm.py` | —（本版探针） | 分段单价、L0 路径四类 trans 的数值证据 |

#### 3.5.4 探针证据（可复现命令）

三个脚本都在**仓根之外**运行（仓根 `tilelang/` 是 dev root，会遮蔽 site-packages 的 0.1.15 私有 TVM）：

```bash
cd /tmp && source /workspace/tilelang-mlir-ascend-950/.agents/platforms/toolchains/ascend/env-950pr.sh
TMPDIR=/workspace /workspace/venv-ascend950/bin/python \
  /workspace/tilelang-mlir-ascend-950/examples/mla_dense_bwd/_bwd_kernel/history_version/design_probe_bwd_cost.py
TMPDIR=/workspace /workspace/venv-ascend950/bin/python \
  /workspace/tilelang-mlir-ascend-950/examples/mla_dense_bwd/_bwd_kernel/history_version/design_probe_l0_gemm.py
TMPDIR=/workspace /workspace/venv-ascend950/bin/python \
  /workspace/tilelang-mlir-ascend-950/examples/mla_dense_bwd/_bwd_kernel/history_version/design_probe_bwd_proto.py
```

实测输出（2026-11-25，toolchain `ascend-950pr-9579 / TileLang ascend 0.1.15 / CANN 9.2.1`）：

| 探针 | 内容 | 实测结果 |
|------|------|---------|
| `design_probe_bwd_cost.py` | v1 内层迭代的 9 个分段单价（`mte2/gemm5/l0c2ub/ub2l1/epi0/epi1/epi2/chain_notr/chain`） | 见 §1.6.4 A 表；`chain`=5.950 µs 复现生产 6.03 µs |
| `design_probe_l0_gemm.py` | L0 路径的 `(T,T)`/`(F,F)`/`(F,T)` 三种 trans 组合 + 数值 | 全部 **PASS**（rel 1.1e-7 / 1.1e-7 / 1.4e-7）；单次 launch 4.1–4.2 µs（含 2.7 µs launch 地板） |
| `design_probe_bwd_proto.py` | C2 结构端到端（`S=384` 与 `S=2048`，causal/非 causal） | 数值全 **PASS**；单价 **3.469 µs/迭代**（stages=1）/ 3.391（stages=2） |
| 原型 bisect（L0/L1 混用） | 4 个 L1 路径 GEMM + 显式 L0 缓冲 | **FAIL**：`dk/dv` 错 4.8e-1 / 3.1e-2 → 已列为硬约束（§3.5.1） |
| 原型 bisect（UB→L1 转置） | `T.copy(UB, L1, transpose=True)` | **COMPILE_FAIL**（`Ascend device compilation failed`） |
| L0C 容量 | `Nk=128, Mq=48`（249856 B = 95.3%）端到端 | **设备报 507015**（`npuSynchronizeDevice`）→ 视为 L0C 上界证据 |

说明：以上除「数值 PASS」外均为**单价/可行性**证据；生产 shape 的精度与性能仍由 Stage 3 的 L0/L1/L2 套件
与 msprof 采集（§8.3）。

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

（源算子签名另含 `o`：由同一次算子调用的 `_delta_kernel` launch 消费，**本 kernel 不读**，见 §0.2。）

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
| k_s | (96,192) | 输入 dtype | L1 | `s^T` 的 A，跨 query 循环常驻 |
| v_s | (96,128) | 输入 dtype | L1 | `dP^T` 的 A |
| kT_s | (192,96) | 输入 dtype | L1 | `dQ` 的 B（`trans_B=True`） |
| q_s | (64,192) | 输入 dtype | L1 | `s^T` 的 B |
| qT_s | (192,64) | 输入 dtype | L1 | `dK` 的 B |
| do_s | (64,128) | 输入 dtype | L1 | `dP^T` 的 B |
| doT_s | (128,64) | 输入 dtype | L1 | `dV` 的 B |
| pT_s | (96,64) | 输入 dtype | L1 | `dV` 的 A（向量侧产出） |
| dsT_s | (96,64) | 输入 dtype | L1 | **`dK` 的 A 与 `dQ` 的 A（`trans_A`）共用** |
| a_k | (96,192) | 输入 dtype | L0A | K-major 操作数：k / v / pT / dsT |
| a_t | (96,64) | 输入 dtype | L0A | MN-major 操作数：dsT（`trans_A=True`） |
| b_lo | (64,192) | 输入 dtype | L0B | q / do（按切片使用） |
| b_hi | (192,96) | 输入 dtype | L0B | qT / doT / kT（按切片使用，**kT 最后重灌**） |
| st_c | (96,64) | float32 | L0C | `s^T`（每迭代清零） |
| dpt_c | (96,64) | float32 | L0C | `dP^T`（每迭代清零） |
| dq_c | (64,192) | float32 | L0C | `dQ`（每迭代清零） |
| dk_c | (96,192) | float32 | L0C | `dK`（跨 query 循环） |
| dv_c | (96,128) | float32 | L0C | `dV`（跨 query 循环） |
| st_u / dp_u | (48,64) | float32 | UB | L0C 半区落地 + 向量输入 |
| p16_u / dsT_u | (48,64) | 输入 dtype | UB | 向量产出的 fp16 半区 |
| lse_u / dlt_u | (64,) | float32 | UB | 行统计量 |
| tail_u | (48,192) | float32 | UB | 块尾 `dk_c` 半区中转 |
| tailv_u | (48,128) | float32 | UB | 块尾 `dv_c` 半区中转 |
| out_k / out_v | (48,192) / (48,128) | 输入 dtype | UB | 块尾 cast 结果（写 GM 前） |

### 4.4 内存搬运路径

```
GM ──(MTE2: nd2nz / dn2nz；越界尾部 clamp+fill 由 AscendInsertOOBPadding 生成)──▶ L1
     k_s, v_s, kT_s(常驻) | q_s, qT_s, do_s, doT_s(每迭代)
GM ──(MTE2)──▶ UB lse_u, dlt_u(每迭代，256 B 级)
L1 ──(MTE1: 每个 GEMM 前一次 T.copy；trans_A 的 A 用 asc_copy_l12l0a_transpose)──▶ L0A/L0B
L0A/L0B ──(Cube mad)──▶ L0C st_c, dpt_c, dq_c(每迭代) / dk_c, dv_c(跨迭代)
L0C ──(fixpipe, 半区 dual_copy, fp32 同 dtype)──▶ UB st_u/dp_u(48,64)
UB ──(SimtVF(128) 半区: exp / mask / mul / cast)──▶ UB p16_u, dsT_u
UB ──(半区 dual_copy；InsertNd2Nz → ascend_nd2nz_scatter)──▶ L1 pT_s, dsT_s
L0C(dq_c) ──(fixpipe + store-atomic add, fp32)──▶ GM dq_acc
L0C(dk_c/dv_c) ──(半区 dual_copy)──▶ UB tail_u/tailv_u ──(cast)──▶ UB out_k/out_v ──(T.copy)──▶ GM dk/dv
```

无 GM 中转的中间块（`P/dP/dS` 全程留在 L0C/UB/L1），符合 §1.6.0 R2 的「不物化」结论。

### 4.5 UB 内存预算（每 AIV，半区）

**UB** 容量 253952 B（248 KiB）；每迭代缓冲取 2 版本，块尾缓冲单版本：

| Buffer | Shape | dtype | 大小 (Bytes) |
|--------|-------|-------|-------------|
| st_u | (48,64) | float32 | 12288 B |
| dp_u | (48,64) | float32 | 12288 B |
| p16_u | (48,64) | bf16 | 6144 B |
| dsT_u | (48,64) | bf16 | 6144 B |
| lse_u + dlt_u | (64,) 各一 | float32 | 512 B |
| 每迭代缓冲的**第二个版本**（上 5 行的 ×1） | — | — | 37376 B |
| tail_u | (48,192) | float32 | 36864 B |
| tailv_u | (48,128) | float32 | 24576 B |
| out_k | (48,192) | bf16 | 18432 B |
| out_v | (48,128) | bf16 | 12288 B |
| 合计 | — | — | 166912 B |

总占比 166912 / 253952 = **65.7%**，余量 87040 B。
- 与 v1 对比：v1 的 `ds_u`（第二份转置 tile）整项删除；半区由 32 行变 48 行（`Nk` 64→96），
  每迭代 UB 从 57856 B 降到 37376 B。
- **Stage 4 流水约束**：若块尾缓冲（92160 B）也要 2 版本，会上界到 259072 B → 超容量；
  故块尾保持单版本（它在 query 循环之外，不参与流水）。
- 编译侧旁证：原型设备码 `buf_dyn_shmem` 的最大偏移用到 61632 个 bf16 元素（= 123264 B），
  低于 253952 B，说明实际分配与预算同量级。

### 4.6 L1 / L0C / L0A / L0B 预算

| 层 | Buffer 组合 | 合计 (Bytes) | 容量 (Bytes) | 占比 |
|----|------------|-------------|-------------|------|
| L1（常驻） | `k_s 36864 + v_s 24576 + kT_s 36864` | 98304 | 524288 | 18.8% |
| L1（每迭代，2 版本） | `(q_s 24576 + qT_s 24576 + do_s 16384 + doT_s 16384 + pT_s 12288 + dsT_s 12288) × 2` | 212992 | 524288 | 40.6% |
| L1 合计 | — | 311296（304 KiB） | 524288 | **59.4%** |
| L0C | `dk_c 73728 + dv_c 49152 + st_c 24576 + dpt_c 24576 + dq_c 49152` | 221184（216 KiB） | 262144（256 KiB） | **84.4%** |
| L0A | `a_k (96,192) f16 36864 + a_t (96,64) f16 12288` | 49152（48 KiB） | 65536（64 KiB） | **75.0%** |
| L0B | `b_lo (64,192) f16 24576 + b_hi (192,96) f16 36864` | 61440（60 KiB） | 65536（64 KiB） | **93.75%** |

**L0C 上界证据**：`Nk=128, Mq=48`（`dk 98304 + dv 65536 + st 24576 + dpt 24576 + dq 36864 = 249856 B` = 95.3%）
在端到端原型上触发 `device error 507015`；`Nk=96, Mq=64`（221184 B = 84.4%）稳定运行。故**设计红线：L0C ≤ 240 KiB**，
主选 84.4%、备选 86.0% 均留 ≥14% 余量。

**L0B 余量说明（93.75%，仅 4096 B）**：`b_lo` 与 `b_hi` 都按「最大形状分配、切片使用」；
`b_hi` 承载 `qT (192,64)`、`doT (128,64)`、`kT (192,96)` 三种切片。若 Stage 3 发现需要第三块 L0B，
回退是把 `Nk` 降到 80（`b_hi (192,80)` = 30720 B，L0B 降到 55296 B = 84.4%），代价是 MTE2 总量 +20%。

### 4.7 动态轴

dense 路径下**无运行时动态轴**：`B/S_q/S_kv/H/D_qk/D_v` 全部是编译期常量（工厂按 shape 编译，
`@lru_cache` 级缓存，与源 `target_cache(maxsize=32)` 等价）。`T_q/T_kv` 的 packed 视图只是 host 侧
`view` 的元数据。变长是另一个算子（§0.7）。这一静态性同时是尾块 OOB clamp 与 L0 静态 extent 的前提。

### 4.8 JIT 配置

```python
@tilelang.jit(out_idx=[], target="ascend")
def _bwd_func(block_n: int = 96, block_m: int = 64, num_stages: int = 1) -> Callable:
    ...
# 编译标志沿用源：-O3、-DENABLE_BF16；不开 fast-math 之外的非默认选项
```

`out_idx=[]`：`dk/dv` 与 `dq_acc` 都是调用方预分配的张量（`dq_acc` 必须由 host 先 `torch.zeros`——
store-atomic 的累加契约，参照 PL-950-atomic-splitk 的调用方式）。

---

## 5. Tiling 策略

### 5.1 计算类型

**类型：混合（Cube + Vector）**。判定依据：5 个矩阵乘（每迭代 11.9 MFLOP）必须落 Cube 的显式 L0 路径；
每迭代另有 6144 个 score 的 `exp/mask/cast` 必须落 Vector；两者在同一 kernel 内交替（§2.2、§0.6 R2/R4）。

### 5.2 Block 划分

```python
block_n = 96   # key 方向（= dk/dv 累加器的 M，以及 dQ 的 K）
block_m = 64   # query 方向（= dq 累加器的 M，以及 dv/dk 的 K）
# 逻辑任务数 = heads * B * ceil(S_kv / block_n)
```

选择理由：
- **`block_n` 是 L1 填充的摊销杠杆**：每 query 行要读 `q,qT,do,doT = 1280 B`，这笔开销按 key 块重复
  `S_kv/block_n` 次。`block_n` 64→96 把总量从 171.8 GB 降到 116.6 GB（−32%），是本次放大的首要目的。
  再往上受 L0C 约束：`block_n=128` 需 272 KiB（超）；`block_n=112` 需配 `dq` 拆半（225280 B，备选）。
- **`block_m=64`**：`dq_c = 64×192×4 = 48 KiB`；`dq` 的 N=192、`dV/dK` 的 K=64 让 mad 的 K 分块不退化；
  半区向量段 `T.Parallel(48,64)` 每 lane 2 个元素（实测编译成 `float2`，无尾 lane）。
- 二者均为 16 的倍数（mad 分形 16×16），`D_qk=192 = 12×16`、`D_v=128 = 8×16` 整除。
- 每迭代端口 325625 B / 31.4 FLOP/字节（v1 为 24.5）；备选的每字节 FLOP 与主选持平（≈31），
  其收益来自**迭代数 −15%**（每迭代固定开销被摊薄）与 MTE2 总量 −15%，不是来自端口效率。

### 5.3 约束分析

- **对齐**：`D_qk=192`、`D_v=128`、`block_n=96`、`block_m=64` 全为 16 的倍数；L1/L0 的 NZ 分形与 mad 的
  16×16 无尾块；`dq_acc` 尾轴 192 个 fp32 = 768 B（32 B 对齐 ✓）。
- **L0C**：单个累加器最大 `dk_c = 96×192×4 = 73728 B ≤ 262144` ✓；5 个合计 **216 KiB / 256 KiB（84.4%）**。
  红线 ≤240 KiB（§4.6 的实测上界）。
- **L0A/L0B**：48 KiB / 60 KiB（75% / 93.75%）；L0B 无版本余量 ⇒ L0 单版本。
- **L1**：304 KiB（59.4%，含每迭代 2 版本）；常驻部分 96 KiB。
- **UB**：163 KiB（65.7%）/ 248 KiB；每迭代 2 版本、块尾单版本。

### 5.4 注意事项（非整除与边界）

> 统一原则：**GEMM 一律全 extent（L0 路径的 M/N/K 取 allocation 形状）；动态 extent 只出现在 copy；
> copy 的越界尾部由平台 pass 改写为 `min()` clamp + 补 0；补 0 的语义后果由掩码臂的
> `gi < S_kv` / `gj < S_q` 收口。**

1. **尾 key 块**（`S_kv % 96 ≠ 0`）：`k_s/v_s/kT_s` 用全 extent 装载，越界行由 `AscendInsertOOBPadding`
   收窄 DMA 并补 0（设备码 `min(96, S_kv - n0)` + `asc_fill_l1`）。补 0 使 `s^T` 的这些行为 0，但
   `exp(0·scale − lse) = exp(−lse) ≠ 0`，故**必须**由掩码臂的 `n0+i < S_kv` 把 `P` 置 0；
   `dk/dv` 写回时同一 clamp 保证不写出 `T_kv`。
2. **尾 query 块**（`S_q % 64 ≠ 0`）：`q_s/do_s/qT_s/doT_s/lse_u/dlt_u` 同样全 extent 装载 + clamp/补 0；
   掩码臂保留 `m0+j < S_q`；`dq_c` 的原子 store 区段由同一 clamp 裁到 `S_q` 内。
3. **因果对角块**：只有对角及相邻块进入掩码臂，其余走 §1.6.1 A2 快路；`loop_st = max(0,(n0-shift)//64)`
   保证不遗漏可见 query 块（最坏多算一个全掩码块，`P=0` 无副作用）。
4. **`S_q < S_kv` 与 `S_kv < S_q`**：`shift` 可为正/负；后者会让头部 query 看不到任何 key（`lse=-inf`），
   此时掩码全假 → `P=0` → 梯度为 0（不产生 NaN，已由 golden 的 `nan_to_num` 语义对齐）。
   **快路的安全性**由 A2 的充分条件保证：满足快路的块不可能包含「看不到任何 key」的行。
5. **`S` 极小**（`S < 96`）：全部走尾块路径；`grid = H*B*1`，所有 DMA 都被 clamp 到 `S` 行并补 0。
6. **`is_causal=False`**：`loop_st=0`，所有块都进入快路（仍受 clamped 尾块影响，见 1/2）。

### 5.5 分核策略（物理核数适配）

> 三要素标准：`.agents/skills/_shared/standards/core-split-strategy.md` §1。

- **① 逻辑核数**：`num_logical_kernels = heads × B × ceil(S_kv / block_n)`（每个逻辑块 = 一个 key 分块 ×
  一个 head，即一次 block 发射）：

  | workload | B | H | S_kv | `ceil(S_kv/96)` | 逻辑核数 |
  |----------|---|---|------|-----------------|---------|
  | mla-dense-bwd-smoke | 1 | 8 | 512 | 6 | **48** |
  | ds-v3-4k | 1 | 128 | 4096 | 43 | **5504** |
  | ds-v3-b2-8k | 2 | 128 | 8192 | 86 | **22016** |
  | kimi-k2-8k | 1 | 64 | 8192 | 86 | **5504** |
  | ds-v2-lite-3k | 4 | 16 | 3072 | 32 | **2048** |

  闭式口径：`逻辑核数 = ceil(S_kv / block_n) × heads × B`；代入 `ds-v3-b2-8k` 得
  `ceil(8192/96) × 128 × 2 = 86 × 256 = 22016`。注意：`design_calc_check.py` 的 `core_split` 按
  「二维 GEMM 网格 `ceil(M/bm) × ceil(N/bn)`」解析本节，本算子的任务网格是**三轴**（key 块 × head × batch），
  不在其约定内，故该项输出 `skip`（不是 fail）——**上表即权威口径**，核数已按 5 条 workload 逐条列出。

- **② 物理核数依据（任务平台记录，非假设值）**：目标型号 `ascend-950pr-9579`（dav-3510）。
  平台 `statectl platform` 落库的实查记录（`core_source` 字段，2026-11-23，
  `.agents/platforms/toolchains/ascend/` 与本次调用返回的 `hardware.observed_cores`）：
  `Ascend950PR_9579.ini [SoCInfo] ai_core_cnt=28, cube_core_cnt=28, vector_core_cnt=56;`
  `runtime probe rtGetAiCoreCount()=28`；`observed_cores = {cube: 28, vector: 56}`。
  平台 ruleset 要求的查询口径是 `torch.npu.get_device_properties`（adapter `core_evidence()`）；
  Stage 3 首跑前用该接口复核型号与核数后再确认（§9.1 风险 11）。
- **③ 规模判定与分核方案**：5 条 workload 的逻辑核数（48 … 22016）**全部 ≥ 物理核数 28**，且块内工作量
  103–1033 次迭代（`ds-v3-b2-8k`：1.415e6 次迭代 / 22016 块 ≈ 64 次/块 × 3.46 µs ≈ 222 µs/块），
  远大于 block 派发开销，判定为 **极大规模**。
  **分核方案：1 逻辑块 = 1 次 block 发射（逻辑块超发，由硬件块调度器按波次派发）**，不采用「固定启动
  内核数 = 物理核数 + 核内串行多任务」，依据：
  (a) 每块工作量是 10²µs 级（见上），摊薄启动开销的收益可忽略；
  (b) 因果注意力下块间工作量相差最多 86 倍（对角块 1 次迭代 vs 远端块 128 次），动态派发天然负载均衡，
  固定 28 核的静态切分会引入长尾；
  (c) 核内串行需要把 Cube+Vector 混合体包进任务循环，而平台未认证「循环内跨单元/跨核同步」。
- **规则偏离声明**：本项**偏离** `core-split-strategy.md` §1 的「极大规模」标准分支（该分支要求「固定启动
  内核数 = 物理核数，核内分批处理多个逻辑块」），采用标准未列出的第四方案「逻辑块超发」。理由即 (a)(b)(c)。
  **复评触发条件**：① Stage 4 实测 block 派发/尾波占比 > 5% 的单 kernel 时长；② 核间负载方差 > 2×；
  ③ 平台认证「循环内 Cube+Vector 交替 + 跨迭代同步」后，核内串行方案转为可选。
- **小规模特例**：smoke（48 块 = 1.71 波）存在约 40% 的尾波空转，属小 shape 固有；Stage 3 的 L0 用例
  不以其时延为性能判据。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

> 逐元素计算已按 §1.6.2 全部向量化；下表只有 block 级 / tile 级调度循环，无逐元素标量循环。

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| 逻辑任务（head × request × key 块） | block 级并行 | `T.Kernel(grid)` 一维核网格 | 每 block 处理一个 key 分块（key-stationary），§5.5 |
| query 块方向 | 迭代（动态界） | `T.Pipelined(loop_st, loop_ed, num_stages=1)` | 因果剪枝后界随 `n0` 变化；缓冲版本由 `annotate_buffer_versions` 声明为 2 |
| 5 个矩阵乘 | tile 级 | `T.gemm(a_l0a, b_l0b, c_l0c, transpose_A=…, transpose_B=…, clear_accum=…)` | Cube 显式 L0；`clear_accum` 区分「每迭代清零」与「跨迭代累加」 |
| L1→L0 暂存 | tile 级搬运 | 每个 GEMM 前一次 `T.copy(l1_region, l0_region)` | L0 路径的显式契约；`trans_A` 的 A 由编译器生成转置搬运 |
| 逐元素 epilogue | 向量并行（半区） | `with T.SimtVF(threads=128): for i, j in T.Parallel(48, 64)` | lane = query 轴（§1.6.3）；A2 由运行时块级 `if` 生成两支 |
| L0C↔UB / UB→L1 | tile 级搬运 | `T.dual_copy`（2:1 半区） | 整 tile `T.copy`(L0C→UB) 只落一个 AIV（v1 实测），必须半区 |
| 块尾 cast | 向量并行 | `T.Parallel(48,192)` / `T.Parallel(48,128)` | `dk` 乘 `scale` 后 cast |
| 块内 serial 子循环 | **不使用 `T.serial`** | — | 除 query 循环外无其它顺序子循环；L1→L0 暂存是 tile 级 `T.copy`（DMA），不需要 serial 循环 |
| 核内多任务（persistent） | **不使用 `T.Persistent`** | — | persistent 需把 Cube+Vector 混合体包进任务循环并做跨迭代同步，平台未认证；本设计按 §5.5 的「逻辑块超发」分核，1 块 = 1 任务 |

### 6.2 循环伪代码

见 §3.3（同一份结构，**已在设备上跑通**；L0 操作数按「最大形状分配 + 切片使用」，`kT` 最后重灌）。

### 6.3 流水线优化

- **实测结论（本轮）**：本结构在 `num_stages=1` 与 `2` 下单价 3.469 vs 3.391 µs（差 2.2%，在块间波动内），
  即**平台的多版本缓冲对「GM→L1 填充 ∥ Cube」不产生实质重叠**。因此本版**不把重叠作为承诺**：
  - 每迭代 L1 缓冲仍声明 2 版本（`T.annotate_buffer_versions({q_s:2, qT_s:2, do_s:2, doT_s:2, pT_s:2, dsT_s:2})`），
    UB 缓冲 2 版本（st_u/dp_u/p16_u/dsT_u），L0C/L0/L0 侧单版本；
  - **重叠靠结构**：Cube（AIC）与向量（AIV）本就是不同核，其并发由 `__mix__(1,2)` 提供；同一 AIC 内
    「填充 → GEMM」的串行由 auto-schedule 决定，本版不承诺消除。
- **Stage 4 的显式流水路径（A/B）**：把 query 循环手工展开为「预取 i+1 的 q/do/qT/doT 到另一组 L1 缓冲，
  再对 i 做 GEMM + epilogue」，即显式双缓冲（不依赖调度器）。预期收益 = min(填充 1.18, GEMM 0.88)
  ≈ 0.8 µs/迭代（→ ~135 ms）；判定阈值：实测 < 主选 × 0.9 才采纳。
- 不做 `T.annotate_unlimit_memory`（不解除容量约束，避免把溢出推到运行期）。

### 6.4 尾块处理

统一口径（与 §3.3 注 2、§5.4 一致）：

- **GEMM**：不参与尾块处理——L0 路径的 M/N/K 取 allocation 形状（静态），region extent 用全 extent。
- **搬运**：GM→L1 / L1→GM / GM→UB 的越界尾部由平台 pass `AscendInsertOOBPadding` 在编译期改写成
  运行时 `min()` clamp，并对 GM→L1 的补 0 区段生成 `asc_fill_l1`。
- **掩码**：`gi < S_kv` / `gj < S_q` 两项**必须**保留在掩码臂中（补 0 不等于梯度 0，见 §5.4 第 1 条）。
  快路分支通过运行时谓词 `n0+95 < S_kv ∧ m0+63 < S_q ∧ n0+95 ≤ m0+shift` 保证两支等价（A2）。
- **分支归属**：A2 是**运行时块级分支**（`n0` 来自核索引、`m0` 来自循环变量），生成两支 SIMT VF。

---

## 7. 同步策略

### 7.1 同步模式

**模式：自动同步。** 依据：平台 `ascend/pipeline.py` 的 `allow_autoschedule` 默认开启，流水线依次跑
`AutoSchedule → AssignCore → PrepareMultiBuffer → ResolveCore → InsertSync → MaterializeMultiBuffer`，
由编译器在 AIC/AIV 与 MTE 队列之间插入事件；平台规则明确「手工跨核 flag 配对、循环复用下的复杂同步
仍未认证」，因此本设计**不写任何手工 `ascend_set_flag/wait_flag/ascend_pipe_barrier`**。

### 7.2 同步点说明

| 位置 | 同步方式 | 理由 |
|------|----------|------|
| GM→L1 / GM→UB 之后到消费 | 自动（InsertSync 插 MTE2→MTE1/M 事件） | 搬运与 GEMM/向量读的依赖 |
| L1→L0 之后到 `T.gemm` | 自动（MTE1→M 事件）；原型设备码 `asc_sync_notify(PIPE_MTE1, PIPE_M, …)` | L0 写后读 |
| `T.gemm` 之后到 `dual_copy(L0C→UB)` | 自动（M→FIX）；原型设备码 `asc_sync_wait(PIPE_M, PIPE_FIX, …)` | L0C 写后读 |
| `dual_copy(L0C→UB)` 之后到 `SimtVF` | 自动（FIX→V，两次 `asc_sync_intra_arrive(PIPE_FIX, e)` 覆盖两个副块） | UB 写后读，且需对两个 AIV 都可见 |
| `SimtVF` 之后到 `dual_copy(UB→L1)` | 自动（V→MTE） | L1 写前读完成 |
| `UB→L1` 之后到 `T.copy(L1→L0)` | 自动（MTE→MTE1） | L0 写前读完成 |
| `set_atomic` ↔ `T.copy(L0C→GM)` ↔ `set_atomic_none` | 硬件模式寄存器；原型设备码 `asc_set_atomic_add_float` / `asc_copy_l0c2gm` / `asc_set_atomic_none` 由调度器保序 | 原子模式必须在 store 前生效、之后关闭 |
| 跨 block（不同 key 块累加同一 `dq_acc`） | **无同步**（store-atomic 由硬件保证原子性） | 不引入任何跨 block barrier |

### 7.3 pass_configs 配置

```python
pass_configs = {}          # 即 TL_ENABLE_AUTO_SCHEDULE=True（默认）
compile_flags = ["-O3", "-DENABLE_BF16"]
```

若 Stage 3 出现同步相关失败，回退顺序是「先钉单版本 → 再拆 `dual_copy` 为两段半区 `T.copy`」；
**不使用** `T.MixedKernel` + 手工 sync（该形态与本工具的 `dual_copy`/UB→L1 混用不可编译，v1 反例 2）。

---

## 8. 验证方案

### 8.1 Golden 函数

> 依据 §0.1 语义独立实现（移植源仓 `mla_bwd_reference` / `MLADenseBwdWorkload.ref_program`），
> **不复刻** §0.6 的分块算法：它物化概率 `P` 再用 `einsum` 反向，不重算任何块、不做原子累加，也不消费外部
> 传入的 `lse`/`delta`（两者都由它自己从 `q,k,v,do` 重算）。

```python
import torch

def golden_bwd_kernel(q, k, v, do, *, scale=None, is_causal=True, dtype=None, slab_heads=8):
    """_bwd_kernel 的 golden：fp64 物化参考，bottom-right 因果。

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
分块/L0C 累加/半区分片/`trans_A` 结构无共享假设，`lse/delta` 的正确性由 `_delta_kernel` 的 L0 负责。

### 8.2 精度标准

| 输出 | dtype | atol | rtol | 依据 |
|------|-------|------|------|------|
| `dk`, `dv` | float16 | 5e-3 | 1e-3 | `reference_tolerance(float16)` + `MLADenseBwdWorkload.verification` 的 `max(5e-3, atol)` |
| `dk`, `dv` | bfloat16 | 5e-3 | 1.6e-2 | 同上（bf16 一行） |
| `dq_acc`（fp32 中间量，测试侧再乘 `scale` 比对 `dq`） | float32 | 5e-3 / `scale` ≈ 6.9e-2 | 1e-3 | `dq_acc = dq/scale`，故 atol 按 `scale = 192^-0.5` 反比放大 |
| `P/dS` 的公式等价性 | — | — | — | §1.6.1 D-1 已机器验证（1 ulp，违反率 0） |

补充口径：`dq_acc` 的累加顺序跨核不确定（store-atomic），因此**不做逐位比对**，只做 dtype 容差；
bf16 的强抵消元素（`dS` cast 舍入 × 17× 放大）历史上有单点 7.8e-3 的实测，属固有量（v1 Stage 3 已二分确认），
判据沿用「把参考实现的 `dS` 单独 cast 到输入 dtype 后重算」的对比手法。

### 8.3 L0 门槛测试计划与性能验收

| 用例 | shape | is_causal | 目的 |
|------|-------|-----------|------|
| L0-1（门槛） | B=1, S_q=S_kv=512, H=8, D=192/128，bf16 与 fp16 各一次 | True | manifest smoke；5 个 L0 GEMM 链 + 原子 + `trans_A` 全路径 |
| L0-2 | 同 L0-1 | False | 非因果（`loop_st=0`、A2 快路全开） |
| L0-3 | B=1, S_q=S_kv=96, H=2 | True | 尾块：`S % 96 = 0` 但 `S % 64 ≠ 0`；再补 `S=100` 覆盖两侧尾块 |
| L0-4 | B=1, S_q=96, S_kv=192, H=2 | True | `shift>0`；`loop_st>0` |
| L0-5 | B=1, S_q=128, S_kv=64, H=2 | True | `shift<0`：头部 query 看不到任何 key（`lse=-inf`）→ 梯度 0 且无 NaN |
| L0-6 | B=1, S_q=S_kv=96, H=1 | True | 单块单迭代（`clear_accum=(kq==loop_st)` 首块语义、单块快路） |
| L0-7 | B=1, S_q=S_kv=97, H=1 | True | 非 16 倍数边长（尾块 1 行；clamp 边界 `S-96=1`） |
| L0-8 | B=1, S_q=S_kv=384, H=4 | True/False | **本版原型的回归基线**（已实测 PASS，见 §1.6.4 D） |

L1/L2 与完整分层套件（`ds-v3-4k` / `ds-v3-b2-8k` / 多 dtype）交由 `tilelang-op-develop` 的 L1/L2/boundary。

**Stage 3 必须前置的最小探针（运行期，按序）**：

| # | 探针 | 输入 | 判定 |
|---|------|------|------|
| P1 | **L0 路径 `trans_A=True` 的 dQ**：单 GEMM | (96,64)·(192,96)ᵗ | 与 torch fp32 一致（rel < 5e-2，bf16 输入）——已在设计期由探针 2 与原型 PASS |
| P2 | **OOB clamp 语义**：`S_kv=100` 的 `dk/dv` 写回不带脏行 | L0-3 | 第 96–99 行正确、无越界写（哨兵值检测） |
| P3 | **尾块掩码收口**：`n0+i ≥ S_kv` 的行贡献为 0 | L0-3 | `dk/dv` 尾部行与 golden 在容差内 |
| P4 | **动态 `clear_accum`**：`kq == loop_st` 首块清零 | L0-4/L0-5 | 与「显式前置 `T.clear`」版本逐元素一致 |
| P5 | **store-atomic 契约**：`dq_acc` 由 host 预清零 | L0-1 | 与未清零对照组相比不翻倍 |
| P6 | **A2 两臂一致性** | L0-1/L0-2 | 快路与掩码臂对同一 (key,query) 块逐元素相同（`dk/dv` 可逐位、`dq` 用 1e-5） |
| P7 | **`b_hi` 复用顺序**：`kT` 必须在 `qT/doT` 之后重灌 | L0-8 | 去掉重灌即 dQ 全错（本版原型首版的失败即此，见 §9.2） |

**性能验收（设计期估算 + Stage 3 用 msprof 复核）**：

| 指标 | v1 实测 | 本版设计期值 | Stage 3 复核方式 / 通过线 |
|------|---------|-------------|--------------------------|
| 内层迭代单价 | 6.03 µs（生产）/ 5.95（探针复现） | **3.47 µs**（原型实测，§1.6.4 D） | 同几何微基准或 msprof 反解；通过线 ≤ 4.0 µs |
| `ds-v3-b2-8k` 的 T_op | 455.3 ms | **175 ms**（外推） | msopprof Task Duration；通过线 **≤ 227 ms** |
| `aic_cube_ratio`（ArithmeticUtilization） | 8.3% | **≈25%**（0.88/3.47） | msprof `PipeUtilization`；通过线 ≥ 22% |
| `aiv_vec_ratio` | 73.5% | ≈42%（(1.24+0.23)/3.47） | 同上；通过线 ≤ 55% |
| `aic_mte2_active_bw` | 未采集 | 70 GB/s/核（常数） | 同上；用于反解填充项是否与设计一致 |
| Stage 4 目标 | — | **≤115 ms / cube ≥40%** | 需 §1.6.3 的 epilogue 阶梯 + §6.3 的显式双缓冲 |
| 80% Cube MFU（T_op ≈ 50.6 ms） | — | **不可达**（§1.6.4 F 的定量边界） | 作为已量化边界登记，不作为验收项 |

---

## 9. 风险点与注意事项

### 9.1 已知约束与风险（按优先级）

| # | 风险 | 证据/来源 | 影响 | 处置 / 回退 |
|---|------|----------|------|------------|
| 1 | **L0 路径的 L1→L0A 转置搬运（`trans_A=True`）在生产 shape 的稳定性未实测**：设计期只在 (96,64)/(64,64) 与端到端原型上验证 | §1.6.4 D；原型设备码 `asc_copy_l12l0a_transpose` | `dQ` 全错或编译失败 | P1/P7 前置；回退①=把 `dQ` 改回「向量侧两趟转置（`epi2`）+ NT L1 GEMM」——实测代价 +1.64 µs/迭代（→ 约 250 ms，仍在 227 ms 边缘）；回退②=把 (Mq,Nk) 的 `dS` 用一次 identity GEMM 在 L0C 生成（+24 KiB L0C，与主选 84.4% 冲突，需先把 `block_n` 降到 80） |
| 2 | **L0C 上界比标称小**：249856 B（95.3%）设备报 507015，221184 B（84.4%）稳定 | §1.6.4；原型 bisect | 分块放大失败 | 设计红线 L0C ≤ 240 KiB；备选 112/64 为 86.0%；任何新配置先用最小原型跑一次 |
| 3 | **L0B 余量仅 4 KiB（93.75%）**，且 L0 一律单版本 | §4.6 | 无法加第三块 L0B / 无法双缓冲 | 回退：`block_n` 80（L0B 84.4%，填充 +20%）；或把 `kT` 拆 `kT_lo/kT_hi` 放 L1 后按需重灌（每迭代 +36 KiB MTE1） |
| 4 | **`dQ` 的 48 KiB 原子写**：总量 69.6 GB（主选） | 结构性推算（§1.6.0 R4） | 若 L2 未驻留则退化为 HBM 瓶颈（MBU 0.74% 说明当前 L2 命中良好） | Stage 3 采集 `asc_copy_l0c2gm` 的耗时；备选 112/64 把原子量降到 59 GB |
| 5 | **`num_stages=2` 无实质重叠收益**（实测 3.391 vs 3.469 µs） | §1.6.4 D | 「靠流水消填充停顿」不成立 | 不承诺；Stage 4 用 §6.3 的**显式**双缓冲，判定阈值 0.9× |
| 6 | **向量 epilogue 0.20 ns/score 高于 UB 端口下界（约 0.06–0.1）**，受指令依赖限制 | §1.6.3 代价模型 | 迭代内 36% 时间 | §1.6.3 的三级优化阶梯（float4 / `simd.vexpdif` / 手工展开），阈值 0.85× |
| 7 | `dq_acc` 依赖 host 预清零（store-atomic 累加） | `mode.py:set_atomic` + PL-950-atomic-splitk | 结果翻倍 | wrapper 必须 `torch.zeros`（现有三 launch 序列已如此）；P5 检查 |
| 8 | `T.exp` 为 SFU 近似 + fast-math | PL-950-softmax-boundaries 在 fp32 上以 1e-6 级误差通过 | bf16/fp16 输出超差 | 已在 §8.2 容差内预期；若超差，关闭 fast-math 或改用 fp32 高精路径后复测 |
| 9 | `lse` 域误用（把 log2 域当自然对数传入）会静默错 | 源算子显式乘 `LOG2E`，本设计不再乘 | 全量精度错 | 接口层不做 log/ln 换算；L0-1 会立即暴露 |
| 10 | A2 的两支 VF 代码体积与编译时间 | v1 探针 C（两臂源文本 4.3 KB） | 大 shape 编译时间增加 | 可接受；若编译时间成为问题，退化为单臂（带掩码，−49.5% 快路径收益） |
| 11 | 物理核数/型号须在 Stage 3 前复核 | 平台 `core_source`（2026-11-23）+ `observed_cores` | 分核判据失效 | Stage 3 首跑前用 `torch.npu.get_device_properties` 复核 28/56 |
| 12 | grid 上限 65535（`TRAP-950-grid-over-65535`） | 主选最大 grid 22016 | 若上游给出更大 B/H 会静默部分执行 | factory 侧断言 `grid <= 65535`，越界 raise（dense 下 `B*H*ceil(S_kv/96)` 全为编译期常量，可在 host 断言） |
| 13 | **80% Cube MFU 不可达** | §1.6.4 F 的定量边界（Cube+填充+向量 = 3.30 µs/迭代 vs 目标 1.00） | 与调度方的期望不一致 | 已把设计目标改写为「Stage 3 ≤227 ms / Stage 4 ≤115 ms + cube ≥40%」，并把边界结论登记为交付物的一部分 |

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|---------|
| **L0 路径与 L1 路径混用** | 只想把 `dQ` 改 L0，其余留 L1 | `dk/dv` 静默错（实测 4.8e-1） | 5 个 GEMM 全部走显式 L0（§0.6 R6） |
| **`b_hi` 内的 `kT` 被 `qT/doT` 覆盖** | 为省 L0B 而复用同一缓冲 | `dQ` 全错 | `kT` 在 `qT/doT` 之后重灌；P7 检查 |
| L0 region 原点非 0 | 想用 `kT_s[96:192,:]` 拆半 | `Ascend compact L0 region … requires zero origin` | 拆半必须用两块独立 L1 缓冲 |
| L0C 超 240 KiB | 分块放大到 128/48 | 设备 507015（`npuSynchronizeDevice`） | 按 §4.6 预算；新配置先跑最小原型 |
| 整 tile `T.copy`(L0C→UB) | 沿用 GPU 习惯 | 只落一个 AIV 的 UB，向量段读脏数据 | 半区 `dual_copy` |
| `T.copy(UB, L1, transpose=True)` | 想把转置挪到搬运 | `Ascend device compilation failed` | 用 `trans_A` 的 L1→L0A 转置 |
| `T.Kernel` 传 `threads=` | 照搬 GPU 源码 | Python 级拒绝 | 线程域只写在 `T.SimtVF` 内 |
| `T.gemm` 用符号 region extent（`x[:an, :]`） | 想用区段截断处理尾块 | `ValueError: T.gemm requires static tile dimensions` | 全 extent + 掩码 + 平台 OOB clamp |
| 显式 `T.Vector(vector=2) as sid` | 想手工按副块分片 | `InternalError … dup sid` | 只取一次 `asc_get_sub_block_id()` |
| L0C→UB 直接 cast 到 fp16 | 想省一次 UB | 前端显式拒绝（dtype 不同） | UB 内 `T.cast` |
| 手工 `set_flag/wait_flag` 配错 | 关掉自动调度后自写同步 | 死锁 / 数据竞争 | 不写手工同步（§7） |
| `dq_acc` 未清零 | 复用 buffer | 梯度翻倍 | host `torch.zeros` |
| 认为「补 0 就等于梯度为 0」 | 尾块只靠 OOB fill | `exp(-lse) ≠ 0` → 尾部行出现伪梯度 | 掩码臂必须保留 `gi < S_kv` / `gj < S_q` |

### 9.3 特殊场景处理

非整除（§5.4）、极小 shape（§5.4 第 5 条）、`is_causal=False`（`loop_st=0`，掩码整体省为 A2 快路）、
`D_qk=192` 非 2 的幂（仅影响缓冲形状，不影响分形，`192 = 12×16`）、`S_q≠S_kv`（`shift` 正负两种情况
已列 L0-4/L0-5）、`S_kv < 96`（单块 + 全 clamp）。

---

## 10. 交付清单

### 10.1 目录结构

```
examples/mla_dense_bwd/_bwd_kernel/
├── DESIGN.md                       # 本设计文档（v2）
├── verify_equiv.py                 # §1.6.1 等价性机器验证（D-1，可重跑）
├── RETROSPECTIVE.md                # Stage 1 复盘（逐次追加）
├── _bwd_kernel.py                  # 算子实现（Stage 3 产出，不在本阶段；v1 版本待按本设计更新）
├── repro/                          # v1 运行期探针（L0C→UB 单 AIV、sid 捕获）
└── history_version/
    ├── design_v0.md                # 首版
    ├── design_v1_pre_cube.md       # 上一版（revision 输入）
    ├── design_probe_compile.py     # v1 编译探针
    ├── design_probe_bwd_cost.py    # v2 探针 1：分段单价（§1.6.4 A）
    ├── design_probe_l0_gemm.py     # v2 探针 2：L0 路径四类 trans（§1.6.4 D）
    └── design_probe_bwd_proto.py   # v2 原型：C2 结构端到端（§3.3/§1.6.4 D）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | 已完成（revision v2） | 含 §0 迁移分析、§1.6.0–§1.6.4 算法调研与实测分解、§11 回退说明 |
| `verify_equiv.py` | 已完成（2026-11-25 重跑，3/3 EQUIV_PASS） | 等价性机器验证脚本 |
| `history_version/design_probe_bwd_cost.py` | 已完成（2026-11-25，9 个分段单价） | v1 瓶颈分解 |
| `history_version/design_probe_l0_gemm.py` | 已完成（2026-11-25，3/3 PASS） | L0 路径 trans 组合 |
| `history_version/design_probe_bwd_proto.py` | 已完成（2026-11-25，数值 PASS + 3.46 µs） | C2 结构原型 |
| `_bwd_kernel.py` | **待按本设计更新（Stage 3）** | 需把 5 个 GEMM 改 L0 路径、删 `ds_u`、`Nk` 96、`b_hi` 复用顺序 |

### 10.3 命名规范

- 项目目录：`examples/mla_dense_bwd/`（TileOPs 迁移项目，按算子名聚合）
- 算子目录：`_bwd_kernel`（对应源工厂函数名；同项目下还有 `_delta_kernel`、`_dq_land_kernel`）
- 实现文件：`_bwd_kernel.py`

### 10.4 实现顺序（Stage 3 建议）

1. ✅ 设计文档 + 等价性验证 + 三个探针/原型
2. ⬜ 把 §3.3 的 L0 结构落到 `_bwd_kernel.py`（最小改动：5 个 GEMM 改 L0、删 `ds_u`/`dsM_s`、`Nk=96`）
3. ⬜ P1/P7（`trans_A` 与 `b_hi` 复用顺序）→ P2/P3（尾块 clamp 与掩码收口）
4. ⬜ P4/P5（动态 `clear_accum`、store-atomic 契约）→ L0-1…L0-8
5. ⬜ L1/L2 与 msprof 采集（`aic_cube_ratio` / `aiv_vec_ratio` / `mte2_bw`），回写 §8.3 表
6. ⬜ Stage 4：§1.6.3 的 epilogue 阶梯 + §6.3 的显式双缓冲 + 112/64 备选裁决

---

## 11. 回退说明（revision v2）

### 11.1 相对上一版的关键调整

| # | 维度 | v1（`design_v1_pre_cube.md` / 当前实现） | v2 | 驱动依据 |
|---|------|----------------------------------------|----|---------|
| 1 | **`dS` 的转置** | 向量段跨步写 `ds_u[j,i]` 生成 `(Mq,Nk)` 副本，`dQ` 走 NT L1 GEMM | **删除该 tile**；`dQ` 用 `transpose_A=True` 的 L0 GEMM 直接消费 `dsT` | 探针 1：该 scatter 占内层迭代 63%（5.950→2.196 µs）；§0.4 #6 的「L1 路径 NT-only」判定被证伪（L0 路径支持 `trans_A`） |
| 2 | **GEMM 路径** | 5 个 `T.gemm(A_l1,B_l1,C_l0c)`（L1 操作数） | 5 个全部显式 L0（`alloc_l0a/l0b` + 手工 `T.copy`） | 要 `trans_A` 就必须用 L0；且两条路径**混用会互相覆盖 L0**（原型 bisect 实测 `dk/dv` 错 4.8e-1） |
| 3 | **tile** | `block_n=64, block_m=64`（L0C 62.5%） | `block_n=96, block_m=64`（L0C 84.4%） | 每 query 行的 1280 B 填充按 key 块重复：64→96 把填充总量 171.8 GB → 116.6 GB（−32%） |
| 4 | **向量输出 tile 数** | 3（`p16`/`dsT`/`ds`），14 B/score | 2（`p16`/`dsT`），12 B/score | 由 #1 直接得到；UB 每迭代 57856 → 37376 B |
| 5 | **UB 半区** | 32 行（`Nk/2`） | 48 行 | 跟 `Nk=96` 走；`T.Parallel(48,64)` 实测编译成 `float2` |
| 6 | **流水** | 声称「Stage 4 提 `num_stages=2`、L1 双版本」 | 声明双版本但**不承诺收益**（实测 3.439 vs 3.464 µs）；重叠改由「AIC/AIV 天然并发 + Stage 4 显式双缓冲」承担 | §1.6.4 D |
| 7 | **L1 端口口径** | 每迭代 279544 B / 24.5 FLOP/B；全量 590.6 GB | 325625 B / 31.4 FLOP/B；全量 460.8 GB（逐项表见 §1.6.0 R3） | 结构变化后的重算 |
| 8 | **地板项判定** | 「L1 端口操作数流是地板项」，估算下界「待核验」 | **三项并列地板**：Cube 40.5 ms / GM→L1 填充 59.5 ms / 向量 epilogue 62 ms；估算下界 = max = 62 ms，串行合成 130–190 ms，原型实测外推 175 ms | 本轮探针给出可引用常数（70 GB/s/核、0.20 ns/score、92/52 GB/s/AIV） |
| 9 | **性能目标** | 未给定量目标 | Stage 3 **≤227 ms**（原型外推 175 ms）；Stage 4 ≤115 ms + cube ≥40%；**80% Cube MFU 登记为不可达**（§1.6.4 F） | 调度方的 ≥2× 硬目标 + 本轮的定量边界 |
| 10 | **验证方案** | L0-1…L0-7 + P1–P6 | 增 L0-8（原型回归基线）与 P7（`b_hi` 复用顺序）；增「性能验收表」（cube/vec ratio 通过线） | 本轮原型暴露的实际坑 |
| 11 | **风险登记** | 风险 1「L1-GEMM 运行期正确性」为首要 | 首要风险改为「L0 `trans_A` 生产稳定性」「L0C 240 KiB 红线」「L0B 4 KiB 余量」 | v1 的 L1 路径在本版已不使用 |

### 11.2 为何不会再犯同一错误

- **「哪种 GEMM 形态可用」以本机探针而非源码断言为准**：v1 依据 `gemm_mad.py:162` 的 L1 路径断言推出
  「NPU 上没有 `trans_A`」，据此引入了一个占 63% 的向量 scatter。v2 直接把四种 trans 组合各跑一次
  （`design_probe_l0_gemm.py`，编译 + 数值），并把结论写进 §0.5/#6 与 §3.4。凡是「某形态不可用」的负向
  判据，今后必须附一次编译/数值探针或明确标注「未核验假设」。
- **任何结构级改动都先做端到端原型再写设计**：v1 的设计在 Stage 3 首跑才发现「整 tile `T.copy`(L0C→UB)
  只落一个 AIV」，代价是一轮返工。v2 的 §3.3 伪代码就是原型代码（`design_probe_bwd_proto.py`）的
  逐行同构版本，数值已 PASS，单价已实测——设计文档不再含「未跑过的结构」。
- **容量红线用实测上界而不是标称值**：`249856 B`（95.3%）触发 507015 而 `221184 B`（84.4%）稳定，
  故红线定在 240 KiB 并写进 §4.6/§9.1，任何新分块先跑最小原型。
- **路径不能混用**：§3.5.1 增「L0 路径与 L1 路径不能混用」硬约束，并给出实测错误值；§9.2 列为第一条常见错误。
- **「地板项」必须有可引用的实测常数**：v1 的地板结论是纸面推算（且常数缺失），v2 用 `CONST-950-gemm-mte2-burst`
  与三个自测单价把流量/计算/向量三项都落到同一个口径，再据此给出「80% MFU 不可达」的边界——避免下一轮
  再出现「目标定得比物理地板还低」的情况。

### 11.3 保持不变的部分

- 算法族（key-stationary 单遍重算）与全部数学（A1/A2 的等价结论与机器验证结果 3/3 EQUIV_PASS，本轮重跑一致）。
- I/O 契约、dtype 路径、`scale` 归属、`lse` 自然对数域、bottom-right 因果、`dq_acc` 的 fp32 原子累加语义。
- 编程模式 `hybrid`（有证据偏离 `simd`）。
- §0.1–§0.3 的三问解读；§0.7 的 packed 分支说明。
- §5.5 的分核三要素（逻辑核数表按新 `block_n` 重算，物理核数实查记录与规模判定结论不变）。
- §8.1 golden 的独立性（只读 `q,k,v,do`，重算概率与 delta）。
