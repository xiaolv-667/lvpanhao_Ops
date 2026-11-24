# MLA prefill 反向主 pass（_bwd_kernel，packed / MLAVarlenBwdOp）算子设计文档

> 迁移任务（harness 子模式），首次设计 v0。源算子：`/workspace/TileOPs/src/tileops/_kernels/attention/mla/bwd.py`
> 中的 `_bwd_kernel`（GPU TileLang 实现，`MLADenseBwdKernel` / `MLAVarlenBwdKernel` **共用同一工厂**）。
> 目标平台：`ascend-950pr-9579`（dav-3510，CANN 9.2.1，TileLang ascend 0.1.15），编程模式按平台
> `programming_modes.migration_default = simd` 评估后选 **混合（Cube + Vector）**，偏离理由见 §2。
>
> **本函数与 `examples/mla_dense_bwd/_bwd_kernel` 是同一个 GPU 工厂的两个分支**（源码只差 `dense` 为真/假），
> dense 版的设计（v1，已通过检视）与实现（已真机通过：L0 13/13、L1 6/6）在本任务中作为**已核验基线**复用。
> 本文按任务要求**明确分区**：满足「同一源码工厂 + 同一设备码形态」的部分直接引用 dense 证据（§0.8、§11），
> **仅重新推导 3 处 packed 差异**（变长块枚举 / head-major 统计量 / 逐 request 边界读写），并给出各自的
> 编译探针与机器验证证据（§3.5.4、§1.6.1）。
>
> 三个 launch 的契约：`_delta_kernel`（产出 fp32 `delta`，head-major）→ **`_bwd_kernel`**（本函数：写 `dk/dv`，
> 向 fp32 `dq_acc` 原地累加）→ `_dq_land_kernel`（`dq = dq_acc * scale` 并 cast）。

---

## 0. 源算子解读与迁移分析

> 方法论：tilelang-op-design skill 的 `references/migration-analysis.md`（三问解读 → 耦合性四态 → 重设计）。
> 本章结论驱动 §1–§11。三问的解读对象是**源工厂的 packed 分支**（`dense_seqlen_q == 0`）。

### 0.1 源算子语义（做什么）

对每个 `(request b, head h, key 位置 n)` 三元组，给定前向的 `q, k, v, do` 与每 query 行的自然对数
log-sum-exp `lse`、`delta = rowsum(o*do)`，计算注意力三个梯度。**packed 语义与 dense 完全相同，差别只在
「每个 request 只看自己的 key」与两个统计量的 layout**：

$$
P[m,n] = \exp\left(\mathrm{scale}\cdot q_m\!\cdot\! k_n - \mathrm{lse}_m\right),\quad
dP[m,n] = do_m\!\cdot\! v_n,\quad
dS[m,n] = P[m,n]\cdot\left(dP[m,n] - \mathrm{delta}_m\right)
$$

$$
dV[n,:] = \sum_m P[m,n]\, do[m,:],\qquad
dK[n,:] = \mathrm{scale}\cdot\sum_m dS[m,n]\, q[m,:],\qquad
dQ[m,:] = \mathrm{scale}\cdot\sum_n dS[m,n]\, k[n,:]
$$

下标是**请求内**下标：request `b` 占 `q/do/dq_acc` 的行区间 `q_start..q_start+len_q`
（`q_start = cu_seqlens_q[b]`）、占 `k/v/dk/dv` 的行区间 `kv_start..kv_start+len_kv`；`q_m` / `k_n` 宽度
`D_qk = 192`，`v_n, do_m` 宽度 `D_v = 128`。**对齐方式与 dense 相同（FA3 bottom-right）**：请求内 query `i`
落在 key 位置 `i + shift`（`shift = len_kv - len_q`），因果可见条件 `n <= m + shift`（含对角）；
`is_causal=False` 时无 mask。

**语义要素逐一确认**（源码逐行核对，无「待确认」项）：

| 要素 | 结论 | 来源 |
|------|------|------|
| 数学语义 | 上式；`P` 由 `lse` 重算（不复用前向概率），`delta` 由 `o*do` 行和给出 | `bwd.py:229-260`；`workloads/attention/mla.py` 的 `mla_bwd_reference` |
| I/O 契约 | `q,k [T,H,D_qk]`、`v,do [T,H,D_v]`（packed THD，`T_q/T_kv` **动态**）、`lse,delta [H,T_q]` fp32（**head-major**）、`cu_seqlens_q/kv [B+1]` int32、`dq_acc [T_q,H,D_qk]` fp32 in-out、`dk [T_kv,H,D_qk]`、`dv [T_kv,H,D_v]` | `bwd.py:141-154`；manifest `MLAVarlenBwdOp` |
| layout | 全部 packed THD（`(token, head, dim)`），尾轴连续；`lse/delta` 为 `(head, token)` | `bwd.py:137` |
| 规约语义 | `dV/dK` 沿**请求内** query 块按 `m` 升序 fp32 累加（`dv_f`/`dk_f`，`T.clear` 后逐块累加）；`dQ` 沿 key 块累加，用 `T.atomic_add` 写 fp32 `dq_acc`，**跨核顺序不确定** | `bwd.py:200-254` |
| dtype 语义 | GEMM 输入 fp16/bf16，**累加器全 fp32**；`P/dS` 在做第二个 GEMM 前 cast 到输入 dtype；`dk` 写回乘 `scale` 再 cast，`dv` 直接 cast；`dq` 只写 fp32（缩放交给 `_dq_land_kernel`） | `bwd.py:164-170, 244-260` |
| 边界语义 | (a) `len_q/len_kv` 之外的 query/key 装载按 0 填充；(b) 因果不可见位置 `P=0`（从而 `dS=0`），实现为 `T.if_then_else` 选择而非乘掩码；(c) `lse=-inf` 的「看不到任何 key」行：mask 全假 → 梯度 0，**不得产生 NaN**；(d) 变长：`len_q`/`len_kv` 逐 request 不同，末尾 request 的最后一个 tile 不满 | `bwd.py:181-243`（`live` 与 `n0[0]+i < len_kv` 守卫） |
| host 侧语义 | packed 调用**不做 view**：`MLAVarlenBwdKernel.forward` 直接把 `(T_q,H,D)` 张量交给 `_run`；`delta` 由独立 launch 产出，`dq` 的缩放 cast 由第三个 launch 完成，故本函数**不得**在 kernel 内乘 `scale` 后写 `dq_acc` | `bwd.py:338-347, 385-387` |

**小结（第一问 What）**：语义 = 「在每个 request 自己的 key 集合上，用给定 `lse`/`delta` 重算 `P/dP/dS`，
产出该 request 的 `dV`、`dK`（含 `scale`）与 `dQ` 的 fp32 累加」。与 dense 相比，语义没有变化，变化的是
**索引空间**（一切行号都经 `cu_seqlens_*` 平移）与**统计量 layout**（head-major `[H,T_q]`）。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `q` | `(T_q, H, 192)` | f16/bf16 | query；`T_q` 为动态轴（`T.dynamic("total_q")`） |
| `k` | `(T_kv, H, 192)` | f16/bf16 | key（latent 已解压，每 head 私有 192 宽） |
| `v` | `(T_kv, H, 128)` | f16/bf16 | value |
| `do` | `(T_q, H, 128)` | f16/bf16 | 输出梯度 |
| `o` | `(T_q, H, 128)` | f16/bf16 | 前向输出；**仅由同一次算子调用的 `_delta_kernel` launch 消费**（`bwd.py:341`），本 kernel 不读；保留在算子输入契约内 |
| `lse` | `(H, T_q)` | fp32 | **自然对数** log-sum-exp，**head-major** |
| `delta` | `(H, T_q)` | fp32 | `rowsum(o*do)`，head-major |
| `cu_seqlens_q/kv` | `(B+1,)` | int32 | packed 前缀偏移；`B` 是编译期常量（工厂参数 `batch`） |
| `dq_acc` | `(T_q, H, 192)` | fp32 | **in-out**，累加缓冲 |
| `dk` | `(T_kv, H, 192)` | f16/bf16 | 输出，与 `k` 同 shape |
| `dv` | `(T_kv, H, 128)` | f16/bf16 | 输出，与 `v` 同 shape |

输出 shape：`dk/dv` 与 `k/v` 完全一致；`dq_acc` 原地累加，shape 不变（后续 launch 缩放 cast 成 `dq`）。

### 0.3 实现算法解读（怎么算）

**计算步骤分解**（`bwd.py:176-260` 的每个计算语句均归入下列步骤；标 ★ 的是 packed 分支独有）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式 |
|------|------|------|------|-------------|
| S1★ | **块表前缀和**：`tiling.cumsum_offsets(cu_seqlens_kv, tile_cum)` 把「每 request 的 key 块数」累加成块表 | `cu_seqlens_kv` | `tile_cum[B+1]`（SMEM int32） | —（任务映射） |
| S2★ | **块解码 + 存活性**：`live = tile < tile_cum[B]`，`tiling.decode(tile, tile_cum, …)` 解出 `(request, n0)` | `tile`, `tile_cum` | `b, n0` | —（任务映射） |
| S3 | 请求内边界：`q_start/kv_start/len_q/len_kv/shift` | `cu_seqlens_*`, `b` | 5 个标量 | —（索引空间） |
| S4 | 装载 key/value 块（越界补 0） | `k[kv_start+n0 …]`, `v[…]` | `k_s (bn,192)`, `v_s (bn,128)` | `k_n, v_n` |
| S5 | 清零 `dk_f, dv_f` | — | fp32 累加器 | 累加器初值 |
| S6 | 因果剪枝区间 `loop_st=max(0,(n0-shift)//bm)`、`loop_ed=ceil(len_q/bm)` | `n0, shift, len_q` | 循环界 | 因果可见性剪枝 |
| S7 | 逐 query 块装载 `q_s, do_s`（越界补 0）与 `lse_s, delta_s`（**head-major 取址**） | `q, do, lse[h, …], delta[h, …]` | SMEM | `q_m, do_m, lse_m, delta_m` |
| S8 | `st = gemm(k_s, q_s^T)` | `k_s, q_s` | `st (bn,bm)` fp32 | `s = scale*q·k`（未乘 scale） |
| S9 | `dpt = gemm(v_s, do_s^T)` | `v_s, do_s` | `dpt` fp32 | `dP` |
| S10 | `st = if(mask) exp2(st*scale*LOG2E - lse*LOG2E) else 0` | `st, lse_s` | `st`（就地） | `P` + 因果/尾块 mask |
| S11 | `p16 = cast(st)` | `st` | `p16` f16 | `P` 的 GEMM 输入 |
| S12 | `dv_f += gemm(p16, do_s)` | `p16, do_s` | `dv_f (bn,128)` | `dV = P^T dO` |
| S13 | `ds16 = st * (dpt - delta_s)` | `st, dpt, delta_s` | `ds16` f16 | `dS` |
| S14 | `dk_f += gemm(ds16, q_s)` | `ds16, q_s` | `dk_f (bn,192)` | `dK`（未乘 scale） |
| S15 | `ds_s = cast(ds16)`（SMEM 中转） | `ds16` | `ds_s` | `dS` 的第二个 GEMM 输入 |
| S16 | `dq_f = gemm(ds_s^T, k_s)` | `ds_s, k_s` | `dq_f (bm,192)` | `dQ`（未乘 scale） |
| S17 | `atomic_add(dq_acc[q_start+m0+i, h, :], dq_f[i,:])`（仅有效行） | `dq_f` | `dq_acc` GM | `dQ` 跨 key 块累加 |
| S18 | 循环结束：`dk = cast(dk_f*scale)`、`dv = cast(dv_f)`（仅有效 key 行，**按 request 偏移写回**） | `dk_f, dv_f` | `dk, dv` | `dk` 的 scale 与输出 dtype |

**数据流与内存访问模式**（源 GPU 视角）：

```
GM[cu_seqlens_kv] --(整块每 CTA 重算)--> SMEM[tile_cum] --(二分)--> (request, n0) --(cu_seqlens_q/kv)--> 请求内偏移
GM[q,k,v,do,lse,delta] --T.copy--> SMEM[k_s,v_s,q_s,do_s] --ldmatrix--> fragment
   fragment[st] --T.gemm(TensorCore)--> fragment[st] --exp2/mask--> fragment[p16]
   fragment[dv_f] <--T.gemm(p16,do_s)--
   fragment[ds16] = st*(dpt-delta) --cast--> SMEM[ds_s] --T.gemm(transpose_A)--> fragment[dq_f]
   fragment[dq_f] --T.atomic_add--> GM[dq_acc]      （每 query 块一次全局原子累加）
   循环后 fragment[dk_f/dv_f] --cast(*scale)--> GM[dk/dv]（request 内偏移）
```

**循环与并行结构**：grid = `num_tiles × heads`，其中 packed 下
`num_tiles = GroupTiling.tile_upper_bound(total_kv) = total_kv // block_n + batch`
（**运行期动态**，多算 ≤ `batch` 个 tile 由 `live` 丢弃）；每 block 内按 query 块
`T.Pipelined(loop_st, loop_ed, num_stages=2)` 串行迭代；块内 `threads=256` 由 `T.gemm` 的 warp 划分与
`T.Parallel` 逐元素循环共同消费；默认配置 `block_n=128, block_m=32, num_stages=2, threads=256`
（`bwd.py:336`）。寄存器驻留：`dk_f/dv_f` 跨整个 query 循环常驻（key-stationary），`dq_f` 每迭代清零。

**host 侧逻辑**：`MLAVarlenBwdKernel.forward` 不做任何 view/transpose（packed 张量原样传入），只把
`cu_seqlens_q/kv` 转交；`_run` 依序 launch `delta → bwd → dq_land`，`dq_acc` 由 host `torch.zeros` 分配（fp32）。
属于算子调用的 host 预处理/后处理仅此三项，均不在本 kernel 内。

**小结（第二问 How）**：算法 = 「一个 key 块常驻 → 枚举该 request 内可见的 query 块 → 每块 5 次矩阵乘 +
1 次 exp/掩码 + 1 次逐元素乘 + 1 次转置 GEMM」；`P/dS` 从不物化到 GM；`dK/dV` 由块内累加器承担，`dQ` 由
全局原子加承担；**packed 额外承担「块表前缀和 + 解码 + 逐 request 边界」三件事，且它们都发生在每个
block 的标量头部**。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | key-stationary 分块（S4/S5/S18） | `k/v` 只搬一次、`dK/dV` 只在块尾写一次 | 一块 key 常驻 SMEM，`dk_f/dv_f` 跨 query 循环寄存器累加 | SMEM + 大寄存器堆 | 硬件强相关（结构可移植，容量依赖） |
| 2 | recompute 而非物化 `P/dS`（S8/S10/S13） | 消 O(Σ l_q·l_kv) 中间缓冲与 GM 往返 | 每个 query 块重算 `s` 与 `dP` | 无（纯算法层） | 可移植（保留） |
| 3 | 因果块级剪枝（S6） | 少算约一半 (key 块, query 块) 对 | `loop_st` 从可见性下界起算 | 无 | 可移植（保留） |
| 4 | tensor core GEMM（S8/S9/S12/S14/S16） | 矩阵算力 | `T.gemm` → mma | tensor core | 硬件强相关（等价替换为 Cube） |
| 5 | 寄存器 fragment 累加器（S5/S12/S14） | 免 SMEM 往返 | `alloc_fragment` + fp32 累加 | 大寄存器堆 | 硬件强相关（重设计为 L0C） |
| 6 | `transpose_A=True` 复用同一 `ds_s`（S16） | 省一份 `dS` 的 SMEM 副本 | mma 直接吃转置操作数 | mma 的 A 转置控制 | 硬件强相关（重设计，见 §0.6 R3） |
| 7 | `exp2` + `scale*LOG2E` 折入指数（S10） | 用便宜底 2 指数指令 | `exp(x)=exp2(x*log2e)` | SFU 上 `exp2` 快于 `exp` | 硬件弱相关（等价替换，§1.6.1 A1） |
| 8 | mask 用 `if_then_else` 选择而非乘掩码（S10） | mask 位置严格为 0，`lse=-inf` 行不产生 NaN | select | 无 | 可移植（保留，转成向量 select） |
| 9 | `T.Pipelined(num_stages=2)`（S7/计算重叠） | 消 MTE 停顿 | 双 buffer 轮转 | 异步拷贝引擎 | 模式可移植（首版 `num_stages=1`） |
| 10 | `T.atomic_add` 写 `dq_acc`（S17） | 免 workspace / 免跨 block 归约 | 全局 fp32 原子加 | 全局原子 ALU + L2 | 硬件强相关（等价替换为 store-atomic） |
| 11 | `lse/delta` 行广播（S10/S13） | 每行一次读代替逐元素重算 | SMEM 标量读 | — | 可移植（保留） |
| 12 | warp 划分 `FullRow` + `threads=256` | 匹配 mma 的 warp 分形 | warp-level 分片 | warp 32 线程 | 硬件强相关（重新设计） |
| 13 | （NPU 适配构造，非源码手段）`clear_accum` | 表达「首块清零 / 跨块累加」 | mma 零累加位 | — | — |
| 14 | **packed/dense 双分支 + `GroupTiling`（S1/S2）** | 一个工厂服务两种 layout | 编译期 `dense` 分支 + 块表前缀扫描 + 二分解码 + `live` | 无（算法层） | 可移植（**本函数的主体重设计对象**，见 R6） |
| 15 | **块表每 CTA 重算**（S1，源码注释明示） | 免一次 host 侧块表构造与额外 GM 张量 | 每 CTA 独立累加 `s_cum` | SMEM + warp scan 原语 | 硬件强相关（重新设计，见 R6） |

**关于 #13 的口径**：源码**没有** `clear_accum` 参数，只用 `T.clear(st/dpt/dq_f/dk_f/dv_f)`
（`bwd.py:200-201,225,227,250`）显式清零；`clear_accum` 是本设计的 NPU 侧构造（§3.2 步骤 4）。

**小结（第三问 Why fast）**：共识别 15 项，其中 5 项纯算法层（recompute、因果剪枝、掩码语义、行统计、
变长分支的结构），9 项依赖源硬件（SMEM/寄存器/tensor core/warp/cp.async/原子 ALU/二维 grid/warp scan），
1 项是迁移侧语义表达构造。**与 dense 相比新增的只有 #14/#15 的 packed 形态**（dense 版把 #14 判为「本函数
舍弃」，本函数把它作为主体）。

### 0.5 硬件耦合性分析与 NPU 适配决策

判定问题：实现算法与优化手段是否硬件强相关？能否用在 NPU 上？依据一律指向本平台已核验资料
（`.agents/platforms/toolchains/ascend/references/programming.md`、knowledge/repro 下的可执行条目、
以及本轮 §3.5.4 的编译探针）。

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 方案 / 依据 |
|------|------|-----------|-----------------|------|----------------|
| 数学语义 / I/O 契约 / 边界语义 | 语义 | 无 | — | **保留** | 语义层无条件保留；golden 依 §0.1 独立实现（§8.1） |
| 二维 block 网格 `T.Kernel(tiles, heads)` | 优化 | CUDA blockIdx 二维 | 无（平台只支持一维核网格） | **重新设计** | R1：一维网格 + 块内 div/mod 解码；packed 下 grid 为动态表达式 |
| key-stationary 分块 + `dK/dV` 跨块累加（#1） | 算法 | 无（结构） | 有 | **保留** | 同一结构；`dk/dv` 累加器从寄存器迁到 L0C（R2） |
| recompute 不物化 `P/dS`（#2） | 算法 | 无 | 有 | **保留** | 单遍结构，`P/dS` 只活 UB（§1.6.0 R2） |
| 因果块级剪枝（#3） | 算法 | 无 | 有 | **保留** | 同一 `loop_st/loop_ed` 语义；packed 下界为运行期量（探针 P2 已编译） |
| tensor core GEMM（#4） | 优化 | mma / fragment | 有（Cube MAD） | **等价替换** | `T.gemm` + L1 操作数 + L0C 累加器；dense 已真机通过，设备码 5 个 `ascend_gemm_l1`（本轮 P2 复现） |
| 寄存器 fragment 累加器（#5） | 优化 | 每线程 255 寄存器 | 部分（Cube 侧只有 L0C 可累加） | **重新设计** | R2：L0C 累加器 + 半区 UB 中转 |
| `transpose_A=True`（#6） | 优化 | mma 的 A 转置控制 | L0 路径有，**L1 路径显式禁止** | **重新设计** | R3：NT-only 重构 + UB 内转置写第二份 `ds` |
| `exp2` + `LOG2E` 折入（#7） | 优化 | SFU 指令选择 | 有（`T.exp` 已核验） | **等价替换** | 自然对数域 `T.exp(s*scale - lse)`（A1，机器验证 1 ulp） |
| `if_then_else` 掩码（#8） | 优化 | 无 | 有 | **等价替换** | `T.SimtVF` 内 `T.if_then_else` + 运行时块级 mask 特化（A2，packed 界重新验证） |
| `T.Pipelined(num_stages=2)`（#9） | 优化 | cp.async 双槽 | 有（`num_stages` + 版本标注） | **等价替换** | 首版 `num_stages=1`（与 dense 一致），多阶段留 Stage 4 |
| `T.atomic_add` 逐元素（#10） | 优化 | 全局原子 ALU | 有（store-mode 原子） | **等价替换** | `set_atomic("add","float32")` + L0C→GM `T.copy`；packed 新增「溢出零行」论证（R5、A3） |
| `lse/delta` 行广播（#11） | 优化 | SMEM 标量读 | 有（UB 内 lane 索引读） | **等价替换** | lane 落在 query 轴，`lse_u[j]` 为 lane 私有读（§1.6.3） |
| warp 划分 + `threads=256`（#12） | 优化 | warp 32 线程 | 无（NPU 无 warp） | **重新设计** | R4：`T.SimtVF(threads=128)` + 半区（`dual_copy` 2:1）+ `sid` 捕获（dense 真机已验证） |
| `FullRow` policy 参数 | 优化 | warp 划分启发式 | 无 | **舍弃** | Cube 侧由 mad 指令形状与 L1 分形决定，`policy` 不入 lowering；不影响正确性 |
| **`GroupTiling.cumsum_offsets`（#14/#15）** | 优化 | **warp shuffle 前缀扫描**（`T.shfl_up`/`WARP_SHUFFLE_STAGES`，且要求 kernel 级 thread extent） | **无**（该形态要求 kernel 作用域存在线程域，ascend 的 `T.Kernel` 显式不建立；`T.shfl_*` 只能落在 `T.SimtVF` 内，而结果需对 AIC 可见） | **重新设计** | R6：**融合式 running-prefix 解码**（O(B) 串行、纯 local 标量、无共享块表） |
| **`GroupTiling.decode` 二分查找** | 优化 | 共享块表 + `T.serial` | 有（纯标量） | **等价替换** | 合并进 R6 的同一次扫描（first-match 赋值），语义与二分等价 |
| **`tile_upper_bound` 的 `live` 谓词**（S2） | 优化 | 无 | 有 | **保留** | 同一 `live` 语义；grid 取同一上界（R1/R6） |
| **逐 request 边界守卫**（S4/S7/S17/S18 的 `n0+i < len_kv`、`m0+j < len_q`） | 优化 | 无 | 有（向量 select） | **等价替换** | 掩码臂保留两项区间比较；`dk/dv` 写回改双分支（R7） |
| **`dk/dv` 的按 request 偏移写回** | 优化 | 无 | 部分（DMA 只按张量 extent 裁剪） | **重新设计** | R7：内部半块整 tile 快路 + 尾半块逐 lane 守卫（探针 P2 设备码双分支） |
| packed 的 `T_q/T_kv` 动态 extent + 动态 grid | 优化 | grid 由 host 按 shape 计算 | 有（`T.dynamic` + 动态 1-D grid） | **保留** | 与 `examples/paged_cache_write/_paged_cache_write_kernel/DESIGN.md` 同机制；本轮 P1/P2/P5 编译级确认 |
| host `torch.zeros(dq_acc)` | 语义 | 无 | 有 | **保留** | store-atomic 的累加契约（不变） |

统计（21 行判定）：**保留 6 项**（数学语义、key-stationary、recompute、因果剪枝、`live`、动态 extent）、
**等价替换 8 项**、**重新设计 6 项**（R1–R4、R6、R7）、**舍弃 1 项**（`FullRow` policy）。
相对 dense 的三处升级：#14 从「舍弃」改为「主体」（R6）、#15 新增（R6）、`dk/dv` 写回从「平台 clamp 即可」
改为「必须双分支」（R7，见 §5.4 的静态边界风险）。

### 0.6 NPU 算法重设计

**重设计项 R1：二维 block 网格 → 一维核网格 + 块内解码（packed 版）**

- **源方案**：`T.Kernel(num_tiles, heads)` 二维 grid（`bwd.py:155`），硬件直接给出 `(tile, h)`。
- **NPU 新算法**：`T.Kernel(grid)` 一维核网格，`grid = (total_kv // block_n + batch) * heads`（**动态表达式**，
  与 `tile_upper_bound` 同式）；块内 `tile = bid // heads`（**静态除数**）、`h = bid % heads`。
  把 head 放在**内层**的两个理由：(a) 动态除数 `bid // ntiles_dyn` 的除法是运行期量除法，未取证（dense 的
  `bid // (batch*ntiles)` 是静态除数）；(b) 同一 (request, key 块) 的相邻 bid 落在相邻 head，而
  `(token, head, dim)` 行主序下同几行的不同 head 切片是相邻内存，L2 行复用更好。
- **语义保持论证**：`(tile, h) ↔ bid` 是双射，任务集合完全一致；无算术改变。边界：非 live 的
  ≤ `batch` 个 tile 由 `live` 丢弃（R6），不产生任何读写。

**重设计项 R2：寄存器 fragment 累加器 → L0C 累加器 + 半区 UB 中转（复用 dense 真机结论）**

- **源方案**：`st/dpt/dk_f/dv_f/dq_f` 全 `T.alloc_fragment`；`dk_f/dv_f` 跨 query 循环常驻。
- **NPU 新算法**：Cube 的累加器只能是 L0C（`gemm_mad.py:_prepare_lower` 要求 C 带 Ascend fractal layout）：
  `dk_c (64,192) fp32 = 48 KB`、`dv_c (64,128) = 32 KB` 跨循环累加；`st_c/dpt_c (64,64) = 16 KB`、
  `dq_c (64,192) = 48 KB` 每迭代清零；共 160 KB / 256 KB（62.5%，§4.6）。
  向量 epilogue 在 UB 内做，**并采用 dense Stage 3 实测确定的半区形态**：`T.copy(L0C→UB)`（`dual_dst_mode=0`）
  只落到一个 AIV 的缓冲（dense 探针 P1 实测 99.2% 元素错位），因此每个 L0C↔UB 交叉一律用 `T.dual_copy`
  2:1 切分（`_HN = 32` 个 key 行/每 AIV），而 L1 侧的三份 fp16 tiles（`pT_s/dsT_s/ds_s`）仍是整块
  `(64,64)` 的 GEMM 操作数（`dual_copy` 的 UB→L1 路径按 AIV 分片写入同一整块）。
- **语义保持论证**：累加顺序保持「`dV/dK` 沿请求内 `m` 升序、`dQ` 沿 key 块跨核累加」，累加 dtype 仍 fp32；
  K 维归约由 mad 完成（与源 mma 同为 fp32 累加、分组不同），与源算子一样只承诺 fp32 累加精度而非逐位一致。
  半区切分只改变「哪个 AIV 计算哪 32 行」，逐元素无跨 lane 归约，两个 AIV 的并集恰好覆盖整块。
- **证据**：dense `_bwd_kernel.py` 已真机 L0 13/13、L1 6/6；本轮 P1/P2 设备码再次出现 `asc_copy_l0c2ub`
  及两次 `asc_sync_intra_arrive`（覆盖两个副块）。

**重设计项 R3：`transpose_A` 的 dQ → NT 形式 + UB 内转置副本（复用）**

- **源方案**：`dq_f = gemm(ds_s, k_s, transpose_A=True)`——同一份 `ds_s` 服侍 `dk` 与 `dq`。
- **NPU 新算法**：平台 L1 GEMM **只支持 NT**（`gemm_mad.py:GemmMAD.lower` 的断言），故 5 个乘法全部写成
  `C = A(M,K)·B(N,K)^T`：`S^T = k_s·q_s^T`、`dP^T = v_s·do_s^T`、`dV = pT_s·doT_s^T`、`dK = dsT_s·qT_s^T`、
  `dQ = ds_s·kT_s^T`；`qT/doT/kT` 用 GM→L1 的 `T.copy(..., transpose=True)`（dn2nz），`ds` 由向量
  epilogue **额外写一份**转置到 UB 再 UB→L1（SIMT 内转置写）。
- **语义保持论证**：`(A B^T)^T = B A^T` 逐元素恒等；`dS` 的第二份副本是同一 fp32 值的同一次 fp16 舍入。
  代价是每迭代多一份 4 KB 的半区 `ds_u` 与 8 KB 的 L1 缓冲。

**重设计项 R4：warp/线程级并行 → SimtVF 半区 lane 映射 + `sid` 捕获（复用 dense 真机结论）**

- **源方案**：`threads=256` 线程块，`T.Parallel(bn,bm)` 的元素映射由编译器按 256 线程划分。
- **NPU 新算法**：Cube 侧由 mad 承担；向量侧在 `T.SimtVF(threads=128)` 内以 `T.Parallel(32, 64)` 覆盖
  **半区**（最内轴为 query，`lse_u[j]` 为 lane 私有读）。副块编号用
  `sid = T.call_extern("int64","asc_get_sub_block_id")` 在 kernel body 中取一次、由 `T.SimtVF` 捕获为 VF 参数
  （`asc_get_sub_block_id()` 不可在 `__simt_vf__` 体内调用）；`T.Vector(vector=2) as sid` 在本工具链
  **不可编译**（`InternalError … dup sid`，dense §3.5.4 反例 2），不得使用。
- **语义保持论证**：`sid` 只进入两个位置——掩码的 key 索引 `gi = n0 + sid*HN + i`（半区偏移是编译期常量
  `HN=32`）与写回起点 `own = koff + sid*HN`；二者与 `dual_copy` 的分片方向一致，故每个 AIV 只处理自己那
  32 行，并集即整块。逐元素映射无跨 lane 归约，结果与划分无关。

**重设计项 R5：全局原子加 → store-mode 原子写回 + 溢出零行（packed 新增论证）**

- **源方案**：`T.atomic_add(dq_acc[q_start+m0+i, h, :], dq_f[i,:])` 在 `T.Parallel` 内逐元素原子加，
  且**逐元素守卫** `if m0+i < len_q`（`bwd.py:252-254`）——只有有效行才写。
- **NPU 新算法**：`T.set_atomic("add","float32")` 打开硬件 store 原子模式，把整个 `dq_c` 用一次 `T.copy`
  落到 `dq_acc[qoff : qoff+64, h, :]`，再 `set_atomic_none()`。**packed 下无法逐元素守卫**（L0C→GM 是整块
  DMA），故必须论证「越出 `len_q` 的行贡献恰为 0」：
  - 掩码臂的 `vis = (gi<len_kv) & (gj<len_q) & (gi<=gj+shift)` 使 `gj >= len_q` 的列满足 `P=0 → dS=0`；
  - `dq_c` 的第 `j` 行 = `Σ_i ds16[j,i]·k[i,:]`，`ds=0` 时是精确 0（fp32 加零）；
  - 因此溢出到**下一个 request** dq_acc 行的写是「原子加精确 0」，数值不变（唯一差别是邻居若恰为 `-0.0`
    会被加成 `+0.0`，数值等价）；
  - 张量末尾的溢出由平台 clamp 裁到 `total_q`（设备码 `min(64, max((total_q-q_start)-kq*64, 0))`）。
- **语义保持论证**：`dq_acc += tile` 与逐元素原子加是同一数学累加；fp32、顺序仍不确定（与源一致）。
  「零行」结论由 §1.6.1 A3 机器验证（30528 个溢出差分元素与 3816 个溢出 dq 行全为精确 0）。

**重设计项 R6（packed 专属）：`GroupTiling` 的 warp 前缀扫描 → 融合式 running-prefix 解码**

- **源方案**：三件套（`bwd.py:138-139, 181-184`）
  1. `num_tiles = GroupTiling(batch, block_n).tile_upper_bound(total_kv) = total_kv // block_n + batch`；
  2. `tiling.cumsum_offsets(cu_seqlens_kv, tile_cum)`：`_tile_prefix_sum` 在 `num_groups > 8` 时走
     **warp shuffle 扫描**（`T.shfl_up` × `WARP_SHUFFLE_STAGES`、`T.get_thread_binding`、
     `KernelLaunchFrame.Current().get_thread_extents()`、`T.sync_threads`），把每组的
     `ceil((offset[g+1]-offset[g]) / block_n)` 累加进共享块表 `s_cum[B+1]`；
  3. `tiling.decode(tile, tile_cum, …)`：`_search_steps = max(1, (B-1).bit_length())` 步二分，求
     `(request, n0 = (tile - s_cum[request]) * block_n)`。
- **NPU 新算法**：一次**融合的 O(B) 串行扫描**，全部在 core-local 标量上完成，**不物化块表**：

  ```
  req = 0; n0 = 0; cnt = 0
  for g in T.serial(batch):                 # batch 是编译期常量，循环可完全展开
      nt = (cu_kv[g+1] - cu_kv[g] + block_n-1) // block_n
      if (tile >= cnt) & (tile < cnt + nt):  # 首个匹配即 decode 的结果
          req = g; n0 = (tile - cnt) * block_n
      cnt = cnt + nt                        # 前缀和（running）
  live = tile < cnt                         # == tile < s_cum[batch]
  ```

- **语义保持论证**：`(i)` `nt` 与源 `tiles(offsets, g)` 逐字同式；`(ii)` `cnt` 在 g 的末尾等于
  `s_cum[g+1]`，因此 `first-match` 赋值给出的 `g` 与 `decode` 的二分结果相同（`decode` 求的正是
  `s_cum[g] <= tile < s_cum[g+1]` 的 g），`n0` 同式；`(iii)` 扫描结束的 `cnt == s_cum[batch]`，`live` 同义；
  `(iv)` 源对 `num_groups <= 8` 本来就退化为同形状的串行 prefix 循环（`_tile_prefix_sum` 的
  `serial_groups = 8` 分支），本设计把它推广到任意 B 并删除共享块表。
  不做向量化的理由见 §1.6.2 #9：该量决定 **block 级控制流与全部全局地址**，且必须对 AIC 与两个 AIV 一致；
  若放到某个 AIV 的 UB 里再读，需要未认证的跨核可见性。成本量化：每 block `≤ B` 次 int32 标量 GM 读
  （`cu_seqlens` 常驻 L2）+ `≤ 3B` 次整数运算，对每 block 数千次 GEMM 迭代而言是 10⁻³ 量级。
- **证据**：探针 P2 设备码
  `for (g=0; g<3; ++g) { ntg = ((cu_kv[g+1]+63) - cu_kv[g]) >> 6; if (cnt <= tile && tile < cnt+ntg) {…} cnt += ntg; }`
  与 `if (tile < cnt) live = 1`；`grid` 由 `total_kv` 动态给出。

**重设计项 R7（packed 专属）：`dk/dv` 的按 request 写回 → 内部快路 + 尾半块逐 lane 守卫**

- **源方案**：`if n0+i < len_kv: dk[kv_start+n0+i, h, :] = cast(dk_f[i,:]*scale)`（`bwd.py:255-260`）——
  **逐元素守卫**，天然不会写出 request 边界。
- **风险起点**：平台的 OOB 裁剪只针对**张量 extent**。设备码实测（P2）装载裁剪是
  `min(64, max((total_kv - kv_start) - n0, 0))`——它保护到**整个张量的末尾**，**不**保护 request 边界。
  于是尾块整块写回会把「精确 0」覆盖到**下一个 request 的 dk/dv 行**（`dk/dv` 是赋值而非累加，写 0 即破坏）。
- **NPU 新算法**：块尾双分支（谓词全为运行期量）
  ```
  own = kv_start + n0 + sid*HN            # 本 AIV 的 32 行起点
  if own + HN <= kv_end:                  # 内部半块：整块 DMA 快路
      T.copy(out_k, dk[own : own + HN, h, :]);  T.copy(out_v, dv[...])
  else:                                   # 每个 request 的尾半块：逐 lane 守卫
      with T.SimtVF(threads=128):
          for i, d in T.Parallel(HN, DQK):
              if own + i < kv_end: dk[own + i, h, d] = out_k[i, d]
          for i, d in T.Parallel(HN, DV):
              if own + i < kv_end: dv[own + i, h, d] = out_v[i, d]
  ```
- **语义保持论证**：快路条件 `own+32 <= kv_end` 保证 32 行全部落在本 request 内（整块写回与逐元素写
  逐元素相同）；慢路是源的逐元素守卫的直接对应（同一比较、同一 dtype cast 值），且对 `own >= kv_end`
  的 AIV（`len_kv - n0 < 32` 时的第二个 AIV）自然写 0 个元素。
  代价：慢路每个 request 最多触发 1 次（每 AIV 半块），影响 ≤ `B·H·2` 个块；慢路的 GM 写是全量
  `T_kv·H·(D_qk+D_v)` 的 ≤ 1/`ceil(len_kv/64)` 部分。
- **证据**：P2 设备码同时出现两条路径——`if ((sid*32 + kv_start + n0) + 32 <= (kv_start + lkv))` 下的
  `asc_copy_ub2gm_align(..., min(32, max((total_kv-kv_start-n0) - sid*32, 0)), ...)`，以及 `else` 下的
  `asc_vf_call<main_kernel_simt_vf_3>`（其内部是 `if (idx < kv_start+lkv)` 的逐 lane 存储）。
  **备选（P3 已编译）**：符号长度区段 `nr = min(HN, kv_end-own)`，`T.copy(out_k[:nr,:], dk[own:own+nr,h,:])`
  与 `T.copy(dq_c[:nq,:], dq_acc[...])` 同样可编译，登记为回退形态（§9.1 风险 2）。

**无重设计项之外的补充说明**：`delta` 的产生（`_delta_kernel`）与 `dq` 的缩放 cast（`_dq_land_kernel`）
不在本 kernel 内，本设计只依赖其接口契约（fp32 / fp32 in-out，head-major），未对它们做任何假设。

### 0.7 标杆实现与上下文

- **源算子路径**：`/workspace/TileOPs/src/tileops/_kernels/attention/mla/bwd.py`（`_bwd_kernel`，
  `bwd.py:109-264`；packed 分支 `bwd.py:136-139, 176-184, 218-224`），仓库内抽取副本
  `examples/TileOPs/tileops/kernels/attention/mla_varlen_bwd/_mla_varlen_bwd_kernels.py`（逐行同源）。
- **golden 基准**：`/workspace/TileOPs/workloads/attention/mla.py` 的 `mla_bwd_reference`（fp32 torch，按
  request 物化 `probs` 后 `einsum` 反向，bottom-right 对齐，`nan_to_num` 处理 `lse=-inf` 行）与
  `MLAVarlenBwdWorkload.ref_program`（把 fp32 梯度**只舍入一次**到输入 dtype）。§8.1 的 golden 移植该实现，
  **不复刻** §0.6 的分块算法。
- **精度用例**：`/workspace/TileOPs/tests/ops/test_mla_bwd.py`；容差来源 `workloads/numerics.py` 的
  `reference_tolerance` 与 `MLAVarlenBwdWorkload.verification`（复用 dense 的 `verification`，atol 下限 5e-3）。
- **manifest workloads**（`src/tileops/manifest/spec/attention.yaml` 的 `MLAVarlenBwdOp`）：
  `ds-v3-8x4k`（T_q=T_kv=32768，B=8×4096，H=128）、`ds-v3-mixed`（T=32256，B=6，lens 512…16384，H=128）、
  `kimi-k2-4x8k`（T=32768，B=4×8192，H=64）。全部 `is_causal=true`，D_qk=192、D_v=128。
- **dense 版基线**：`examples/mla_dense_bwd/_bwd_kernel/DESIGN.md`（设计 v1）与
  `examples/mla_dense_bwd/_bwd_kernel/_bwd_kernel.py`（实现，真机 L0 13/13、L1 6/6，含 6 个前置设备探针）。

### 0.8 复用与差异分区（本函数 vs dense）

**A. 复用部分（引用 dense 证据，并说明为何对本函数成立）**

| # | 复用项 | dense 侧证据 | 为何对本函数成立（结构性理由） |
|---|--------|-------------|------------------------------|
| 1 | **算法族**：key-stationary 单遍重算（flash 反向族） | dense DESIGN §1.6.0 调研结论 + 真机精度 | **同一源码工厂**：`_bwd_kernel` 的 packed/dense 只差 `stat_shape`、`num_tiles`、`request/n0` 解码、`lse/delta` 取址（`bwd.py:133-139, 176-184, 214-224`）；5 个 GEMM 的语义、操作数方向、边界语义逐字相同 |
| 2 | **5 个 NT 乘积**及操作数形状（`k_s·q_s^T`、`v_s·do_s^T`、`pT_s·doT_s^T`、`dsT_s·qT_s^T`、`ds_s·kT_s^T`），`block_n=block_m=64`、`D_qk=192`、`D_v=128` | dense §0.6 R3 + §4.6 容量表 | 维度参数与本函数完全相同（manifest `D_qk=192, D_v=128`），故 L1/L0C/L0A-B 预算逐字节相同 |
| 3 | **全 extent GEMM + 平台 OOB clamp + 掩码臂收口** | dense §3.5.4 探针 A/B 设备码（`min(64, 96-n0)` + `asc_fill_l1`） | L1-GEMM 的 M/N/K 取自 region extent、必须编译期静态（`gemm_op.py` 断言）是平台约束，与 layout 无关；本函数同样只能用全 extent |
| 4 | **半区 epilogue**（`T.dual_copy` 2:1、`_HN=32`）与 `sid` 捕获 | dense Stage 3 真机：整块 L0C→UB 只落一个 AIV（99.2% 错位），改用 `dual_copy` 后 L0 13/13 | 设备行为（`dual_dst_mode=0` 的 fixpipe 目标）与 layout 无关；本函数 P1/P2 设备码复现同一形态 |
| 5 | **store-atomic 写 `dq_acc`** + `_dq_land` 缩放 + `dk` 乘 `scale`、`dv` 不乘 | dense §0.6 R5 + `PL-950-atomic-splitk.py` | 三 launch 契约与本函数完全相同（`bwd.py:338-347`）；`dq_acc` 同为 fp32 |
| 6 | **A1（自然域 exp）与 A2（块级 mask 特化）的等价结论** | dense §1.6.1（3/3 EQUIV_PASS，2026-11-23 重跑） | A1 是逐元素数学恒等式，与索引空间无关；A2 的等价性是**逐块局部**性质（谓词在块内恒真 ⇒ 可省掩码），只需把 `S_q/S_kv` 换成运行期 `len_q/len_kv`（本函数重新机器验证，§1.6.1） |
| 7 | `num_stages=1` + 版本钉死 | dense §6.3 | 平台「多阶段混合循环 + 跨单元重叠」未认证，与本函数的变量无关 |
| 8 | UB/L1/L0C 预算表（半区 90624 B / 172032 B / 163840 B） | dense Stage 3 实现的 buffer 清单 | 缓冲形状逐字相同（`_HN=32` 半区），故容量结论直接承袭 |
| 9 | 编程模式 `hybrid`（Cube+Vector 同 kernel） | dense §2 与 `PL-950-mixed-epilogue.py` | 同一 mix kernel 结构（5 GEMM + 2 段向量） |

**B. packed 差异部分（本文重新推导，逐项给证据）**

| # | 差异 | 源 packed 分支 | 本设计 | 章节 | 证据 |
|---|------|---------------|--------|------|------|
| ① | **变长块枚举** | `GroupTiling(batch, block_n)`：warp shuffle 前缀和 + 二分解码 + `tile_upper_bound` | **融合式 running-prefix 解码**（O(B) 串行、local 标量、无块表）+ 同一 `tile_upper_bound` 作动态 grid 上界 + `live` 丢弃越界 tile；静态化备用方案已编译 | §0.6 R6、§5.4、§5.5 | 探针 P1/P2/P5 COMPILE_OK；P2 设备码含 `for(g…)+cnt` 扫描与 `if (tile<cnt)` |
| ② | **统计量 layout** | `lse/delta` 为 head-major `[H,T_q]`（`stat_shape=(heads,total)`） | **保持 head-major，不转置成 dense 的 `[B,H,S_q]`**：`T.copy(lse[h, qoff:qoff+64], lse_u)` 是一次 256 B 连续搬运；lane=query 的 lane 私有读不变 | §1.6.3、§3.2、§4.1 | P2 设备码 `asc_copy_gm2ub_align(... lse[kq*64 + (bid&1)*total_q + q_start] ..., 1, min(64,…)*4, …)` |
| ③ | **逐 request 边界读写** | 装载/写回全部经 `cu_seqlens_*` 偏移，`dq` 逐元素原子加（含行守卫） | 装载用运行期偏移 + 平台张量尾裁剪 + 掩码收口；`dq` 整块原子（**溢出零行**论证）；`dk/dv` **双分支**（内部整块快路 / 尾半块逐 lane 守卫）；`dk` 乘 `scale`、`dv` 不乘（不变） | §0.6 R5/R7、§5.4、§8.3 | P2 设备码双路径；A3 机器验证（30528 + 3816 个零元素，0 违反）；P3 符号长度备选 |

**C. 为何「复用」在本函数成立（一句话）**：本函数与 dense 版来自**同一源码工厂、同一设备码形态**——差异只
在「任务索引空间」与「两个一维统计量的 layout」，而所有复用项（分块结构、5 个 NT 乘积、L0C 累加器、
半区 epilogue、store-atomic、UB/L1/L0C 预算）都由**维度与平台约束**决定，不依赖索引空间；唯一的交互点是
`dk/dv` 写回的边界（dense 可交给平台 clamp，packed 必须双分支，即 R7），本文已单独重推。

---

## 1. 概述

### 1.1 算子名称

`_bwd_kernel`（算子：`MLAVarlenBwdOp` 的主 pass，MLA prefill 反向，packed / THD）。

### 1.2 功能描述

给定前向的 `q, k, v, do, lse, delta`（packed）与 `cu_seqlens_q/kv`，按 key-stationary 单遍重算 `P` 与 `dP`，
产出 `dk, dv`（输入 dtype）并向 fp32 缓冲 `dq_acc` 原地累加 `dQ`（缩放 cast 由 `_dq_land_kernel` 完成）。
每个 request 只看自己的 key，query `i` 对齐到 key 位置 `i + len_kv - len_q`。

### 1.3 数学公式

$$
\begin{aligned}
P[m,n] &= \exp\!\big(\mathrm{scale}\,(q_m\!\cdot\! k_n) - \mathrm{lse}_m\big)
  \cdot \mathbb{1}\big[n \le m + \mathrm{shift},\ n < len_{kv},\ m < len_{q}\big] \\
dP[m,n] &= do_m \cdot v_n, \qquad
dS[m,n] = P[m,n]\cdot\big(dP[m,n] - \mathrm{delta}_m\big) \\
dV[n,:] &= \textstyle\sum_m P[m,n]\,do[m,:], \qquad
dK[n,:] = \mathrm{scale}\cdot\textstyle\sum_m dS[m,n]\,q[m,:], \qquad
dQ[m,:] = \mathrm{scale}\cdot\textstyle\sum_n dS[m,n]\,k[n,:]
\end{aligned}
$$

其中 `m, n` 为**请求内**下标；`shift = len_kv - len_q`，`scale = D_{qk}^{-1/2} = 192^{-1/2}`。
本 kernel 输出 `dk = cast(dK)`、`dv = cast(dV)`、`dq_acc += dS·k`（**不含** `scale`，由 `_dq_land_kernel`
乘 `scale` 后 cast）。

### 1.4 算法描述

算法族选择见 §1.6.0：**key-stationary 单遍重算（flash 反向族）**，与源算子同一族；操作数方向与数据落地
按 NPU 的三项约束重构（§0.6 R1–R5），块枚举与逐 request 边界按 packed 语义重推（R6/R7）。迁移后的算法步骤
（与 §1.6.0 选定算法一致）：

1. **任务映射**：一维核网格，`grid = (total_kv // 64 + B) * H`（动态上界）；块内 `tile = bid // H`、
   `h = bid % H`，再用**融合式 running-prefix 扫描**解出 `(request, n0)` 与 `live`（`live==0` 的块立即退出）。
2. **请求内几何**：`q_start/kv_start`、`len_q/len_kv`、`shift` 由 `cu_seqlens_*` 现算。
3. **key 常驻**：`k`、`v` 与其转置副本 `kT` 从 GM 搬入 L1（`k_s (64,192)`、`v_s (64,128)`、`kT_s (192,64)`），
   跨整个 query 循环常驻；`dk/dv` 的 L0C 累加器由首迭代的 `clear_accum` 清零。
4. **query 块循环**（因果剪枝：`kq ∈ [max(0,(n0-shift)//64), max(ceil(len_q/64), loop_st+1))`）：
   a. 搬入 `q_s/qT_s/do_s/doT_s` 与 head-major `lse/delta` 的 64 元区段；
   b. `st_c = k_s·q_s^T`、`dpt_c = v_s·do_s^T`（NT，全 extent，`clear_accum` 每迭代）；
   c. `T.dual_copy` 把两块 L0C 按 key 行 2:1 落到两个 AIV 的半区 UB；向量侧算
      `p = exp(st*scale - lse)`（A2 运行时块级特化）→ `pT`(f16)、`dsT`(f16) 与转置副本 `ds`(f16)；
   d. 三份半区 fp16 经 `T.dual_copy` 落 `pT_s/dsT_s/ds_s`（L1 侧仍是整块 64×64 操作数）；
   e. `dv_c += pT_s·doT_s^T`、`dk_c += dsT_s·qT_s^T`（跨迭代累加）、`dq_c = ds_s·kT_s^T`（每迭代清零）；
   f. `set_atomic("add","float32")` 后把 `dq_c` 整块落到 `dq_acc[q 行, h, :]` 并关闭原子模式。
5. **块尾**：`dk_c`、`dv_c` 按半区 `T.dual_copy` 落到 UB，向量侧 cast（`dk` 另乘 `scale`）后写回 GM——
   **内部半块整块写、尾半块逐 lane 守卫**（R7）。

### 1.5 数据流图

```
GM: cu_seqlens_q/kv(int32) ──标量读──▶ (request, n0, live, q_start, kv_start, len_q, len_kv, shift)
GM: q,k,v,do(f16)  lse,delta(fp32, head-major)  dq_acc(fp32, in-out)
        │  GM→L1 (MTE2, nd2nz / dn2nz；越界尾部由 OOB pass clamp+fill，按张量 extent)   │
        ▼                                                                            │
   L1: k_s,v_s,kT_s(常驻)  q_s,qT_s,do_s,doT_s(每迭代)                                │
        │  L1→L0 (MTE1, 由 GEMM lowering 内部按 tile_k_sub 完成)                       │
        ▼                                                                            │
   L0A/L0B ──mad──▶ L0C: st,dpt(每迭代) dk,dv(跨迭代) dq(每迭代)                       │
        │  T.dual_copy 2:1 (L0C→UB, fp32 同 dtype, 每 AIV 32 个 key 行)                 │
        ▼                                                                            │
      UB(fp32 半区): st_u, dp_u ──SimtVF(128)──▶ UB(f16 半区): p16_u, dsT_u, ds_u      │
        │  T.dual_copy (UB→L1，InsertNd2Nz → NZ scatter)                                │
        ▼                                                                            │
      L1: pT_s, dsT_s, ds_s ──▶ 作为 dV/dK/dQ 的 A 操作数（整块 64×64）                │
                                                                                     │
   L0C(dq) ──set_atomic(add) + T.copy 整块──▶ GM: dq_acc ─────────────────────────────┘
   L0C(dk,dv) ──dual_copy→UB(半区)──cast(+scale)──┬─ 内部半块: T.copy → GM dk/dv
                                                  └─ 尾半块:  逐 lane 守卫写 GM dk/dv
```

`P/dP/dS` 全程留在 L0C/UB/L1，不物化到 GM（§1.6.0 R2 结论）。

### 1.6 算法调研与优化分析

#### 1.6.0 算法调研（Algorithm Research）

> 调研对象：**「变长（packed）批量注意力反向，给定 lse/delta 的梯度」这一数学语义的算法族**。
> 源算法（key-stationary + `GroupTiling`）只是基线候选之一。
> 检索范围：平台 `knowledge/algorithm-candidates.md` 的 attention 族、`knowledge/` 下全部 950 条目
> （`PL-950-*`、`ALG-950-formulas.md`、`bottlenecks.md`）、本仓 `examples/` 同类实现
> （`examples/mla_dense_bwd/_bwd_kernel/DESIGN.md`、`examples/flash_attention/`、`examples/mixcv/`）、
> 源仓 golden 与 manifest roofline（`mla_varlen_bwd_roofline`）、以及 §0 的结构判据。网络检索未使用
> （本地源已覆盖该族全部已知结构，见 R2 的结构依据）。
>
> 代表 workload = `ds-v3-mixed`（B=6，lens 512/1024/2048/4096/8192/16384 对应 q 与 kv，
> T_q=T_kv=32256，H=128，D_qk=192，D_v=128，bf16）。

**R1 等价化简公式候选**（基线在表内；正式论证与收益在 §1.6.1）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 纳入 R3 |
|---|------|------------|-----------|---------|---------|
| 0 | **基线（源 packed 算法）** | `P=exp2(s·scale·log2e − lse·log2e)`；5 个 GEMM（`dQ` 用 `transpose_A`）；`dq` 逐元素原子加；`GroupTiling` 块枚举 | — | — | ✅（基线） |
| 1 | 自然对数域 `exp` | `P=exp(s·scale − lse)` | 数学恒等（`exp(x)=exp2(x·log2e)`），差异只两式舍入路径 | 每 score 少 1 次乘；指令数不减（§1.6.1 A1） | ✅ |
| 2 | 块级 mask 特化（运行期谓词） | 全可见块省掉 `select`；packed 下谓词用 `len_kv/len_q/shift` | 谓词是「块内掩码恒真」的可判定充分条件 | 每 score 少 2 次比较 + 1 次选择；packed 实测快路占 48.6% 块 | ✅ |
| 3 | 行统计量 `lse·log2e` 逐行提升 | 先算 `lse2[m]=lse[m]·log2e` | 恒等 | 被 #1 完全吸收 | ❌（冗余） |
| 4 | `scale` 折进 `q`/`k` 装载 | `(q·scale)` 进 score GEMM | 数学恒等，但 fp16 先乘常量多一次舍入 | 逐迭代乘数 4096 vs 预乘 12288，**不省**；且 `q`/`k` 各需两份副本 | ❌（§1.6.1 定量） |
| 5 | `delta` 项改 rank-1 修正 | `dS = P∘dP − P∘delta` 拆两项进 GEMM | 恒等 | 多一个 GEMM（+≈5.7% 计算量） | ❌ |
| 6 | `dV`、`dK` 合并为一个 GEMM | 拼 `A=[pT;dsT]`、`B=[doT;qT]` | **不等价**（会求和） | — | ❌（不等价） |
| 7 | 物化 `P/dS` | 一遍落 GM 再扫描 3 个 GEMM | 恒等 | 中间缓冲 ≥45.8 GB/张量、3–4 遍扫描 | ✅（进 R3 作反例） |
| 8 | query-stationary 变体 | `dQ` 常驻、`dK/dV` 原子累加 | 恒等 | 原子放大项从 `D_qk` 换成 `D_qk+D_v`（更宽），且 fp16 输出二次舍入 | ✅（进 R3） |
| 9 | 两 kernel 拆分（dQ / dK+dV） | 各自单遍 | 恒等 | FLOPs ×2，破坏三 launch 契约 | ✅（进 R3） |
| 10 | **flat token 轴 tile（忽略 request 边界）** | 把 `T_q` 当连续 token 轴切 64 行块，块内 decode 出「首个」request | **不等价**：跨边界的 tile 只解出一个 group，边界 token 的 `dk/dv` 会漏写/错写（源 `GroupTiling` docstring 明示该失败模式） | — | ❌（不等价） |
| 11 | **按 request 补齐到 64 的均匀分块（pad）** | 每 request pad 到 64 的倍数再枚举 | 恒等（补 0 后掩码收口） | 需 host 侧重建/拷贝 `q/k/v/do` 与 `dq_acc`（T 级额外往返），违反「packed 原样传入」的契约 | ❌ |
| 12 | **一个 request 一个 block、块内串行 key 块** | 并行度 = `B·H` | 恒等 | 并行度 768（ds-v3-mixed），最长 request 的 256 个 key 块串行 → 长尾 32× | ✅（进 R3/R4） |
| 13 | **块表改由 host 预计算（一次 launch 前算 `s_cum`）** | host 侧 `cumsum(ceil(len/64))` 成 int32 张量 | 恒等 | 多一个 GM 张量 + 一次 host 同步（`cu_seqlens` 在 device 上，需 D2H）；且违反「kernel 自包含」的迁移约定 | ❌（成本/契约） |

**R2 在线算法**：**有，且基线即在线变体。** 结构依据：两个统计量 `lse`（前向 log-sum-exp）与
`delta = rowsum(o*do)` 都由外部提供，因此反向**不需要**前向 online softmax 的 running max / 重缩放机器；
每个 (key 块, query 块) 对只需重算局部 `s/dP` 并立刻消费（`P/dS` 只活片上），得到单遍扫描、O(tile) 中间态
的流式结构（源算子即如此，§0.4 #2）。非在线对照是 R1#7（物化 `P/dS`，3–4 遍、≥45.8 GB 缓冲）。
**无更优在线结构**：本算子的输出里必有一组（`dQ` 或 `dK/dV`）要沿另一轴跨块归并，单遍结构下无法再压缩
（R1#6 的合并已证不等价，R1#10 的扁平枚举已证不等价），故在线族内部不存在「更少遍数」的变体——该负向
断言的依据是上面的交换性/边界分析，而非经验。**packed 特有的在线结论**：变长只改变「每个 request 的扫描
区间」，不改变单遍性；`live`/`decode` 是任务映射（O(B) 标量），不引入第二遍数据扫描。

**R3 复杂度对比**（四口径；代表 workload = `ds-v3-mixed`，H=128，T_q=T_kv=32256，B=6，
可见 score 对 = 22.9e9，分块迭代 = 5,623,296）：

| 候选 | FLOPs | 访存量 GM (Bytes) | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|------|-------|------------------|---------|-------------|--------------------|
| C0 基线（GPU 布局 + `GroupTiling`） | 38.2 T | ≈ 744 GB（q/do 双读 460.7 + `dq` 原子 276.4 + k/v 4.2 + `dk/dv` 2.6） | 1 | 寄存器 fragment + SMEM + 块表 `s_cum` | 64,512 任务 × 28 核；`dq` 原子跨核 |
| **C1 本设计（NT-only + L0C + 半区 UB + 融合 decode + 溢出零行）** | 38.2 T | ≈ 744 GB（同上，无额外往返） | 1 | L0C 160 KB + L1 168 KB + UB 88.5 KB | 同上；块内无共享块表（省 SMEM 与 warp scan） |
| C2 query-stationary | 38.2 T | ≈ 926 GB（k/v 两份重读 + `dK/dV` 原子 458 GB） | 1 | `dQ` 常驻 L0C 48 KB | 原子目标宽 320（>192）→ +67% 原子流量；fp16 输出二次舍入 |
| C3 两 kernel 拆分 | **76.4 T** | ≈ 470 GB（无原子） | 2 | 同量级 | 两倍 launch 与两倍整型工作量 |
| C4 物化 `P/dS`（R1#7） | 38.2 T | ≥ 92 GB 额外往返 | 3–4 | **≥ 45.8 GB/张量**（远超 128 MiB L2） | 不可行（容量淘汰） |
| C5 纯 Vector（无 Cube） | 38.2 T | 同 C1 | 1 | 同量级 | 56 个 Vector 核做矩阵乘（lane 级），算力低 1–2 个数量级 |
| C6 flat token tile（R1#10） | 38.2 T | 同 C1 | 1 | 同 | **不等价**（跨 request 边界漏写 `dk/dv`）→ 淘汰 |
| C7 每 request 一块（R1#12） | 38.2 T | 同 C1 | 1 | `dQ` 常驻 | 并行度 768（≈27 波）；最长块 256 次串行迭代 → 长尾 ≈32× |

口径说明：FLOPs 用 manifest roofline 公式 `H·(scores·(2(3D_qk+2D_v)+5) + rows·2·D_v)`
（`scores = Σ_b l_q·l_kv − l_q(l_q−1)/2`，`rows = Σ_b l_q`）；GM 访存量含输入读 + 输出写 + 跨块累加往返，
**不含** L1/UB 内部搬运（单列于下表的「每迭代 L1 端口分解」）。分块迭代数按 `Σ_b M(M+1)/2 × H`（bottom-right
因果的精确剪枝计数，`M = ceil(l_q/64)`）：`(36+136+528+2080+8256+32896)×128 = 5,623,296`；按 `ceil` 分块
实际 FLOPs = 5,623,296 × 6,852,608 = 38.5 T（含 0.9% 的尾部量化开销）。所有数字由公式手算，非实测。

**packed 的负载不均（512→16384）量化**：每 request 的 key 块数 `ceil(len_kv/64)` = 8/16/32/64/128/256
（合计 504 块/head）；每块的 query 迭代数 = `M − kn`（`kn` 为 key 块号），最长块 256 次迭代。
总迭代 5.62M → 单块最长占总量 256 / 5.62M = **0.0046%**，因此在「1 逻辑块 = 1 次发射 + 硬件动态派发」下，
尾波最多浪费这么多工作（对比静态 28 等分：块时长差 32×，必然长尾）。该结论驱动 §5.5 ③。

**每迭代 L1 端口分解（block_n=block_m=64；口径与 dense 一致，一次 (key 块, query 块) 迭代）**：

| 项 | 组成 | 字节 |
|----|------|------|
| GEMM 操作数读 | `k_s 24576 + q_s 24576 + v_s 16384 + do_s 16384 + pT_s 8192 + doT_s 16384 + dsT_s 8192 + qT_s 24576 + ds_s 8192 + kT_s 24576` | 172032 |
| UB→L1 写 | `pT_s + dsT_s + ds_s = 3 × 8192` | 24576 |
| GM→L1 装载（每迭代） | `q_s 24576 + qT_s 24576 + do_s 16384 + doT_s 16384` | 81920 |
| GM→L1 常驻摊销 | `k_s + v_s + kT_s = 65536` ÷ 平均 ≈87 次迭代/块（5,623,296/64,512） | 752 |
| **合计** | | **279280**（≈279.3 KB，KB=1000 B） |

对应每迭代 `6,852,608 FLOP` → **24.5 FLOP/字节**；全量 = 5,623,296 × 279,264 B = **1.571 TB** L1 端口流量。
（半区 `dual_copy` 不改变 L1 侧字节：L1 的三个中间块仍是整块 64×64；UB 侧流量按两 AIV 各半计。）

**R4 硬件亲和性评估**（逐候选；容量对照 §4.5/§4.6，常数缺失项显式标注）：

| 候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|------|-------------|---------|------------|---------|------------|------|
| C0 基线 | 需 `transpose_A` GEMM 与寄存器累加器，NPU 无对应（L1 路径断言只允许 NT） | — | — | 块表用 warp scan，NPU 无 warp | — | ❌ 不能直译（R3/R6 重设计） |
| **C1 本设计** | 5 个 NT GEMM 全落 Cube；`exp/mask/cast/转置` 落 AIV；标量热点只有 O(B) decode | L0C 160/256 KB（62.5%）、L1 168/512 KB（32.8%）、UB 88.5/248 KB（35.7%） | `D_qk=192 = 12×16`、`D_v=128 = 8×16`、64 全为 16 的倍数；`_HN=32` 对 `dual_copy` 2:1 成立 | **需静态化设计**：GEMM 的 M/N/K 全为 allocation extent；grid 用动态上界 + `live`；静态备用方案已编译（P5） | GEMM 与向量 epilogue 同 kernel 交替（混合模式），首版 `num_stages=1` | ✅（动态 grid 的运行期行为是最大未核验项，§9.1 风险 1） |
| C2 query-stationary | 同 C1 | 同量级 | 同 | 同 | 同 | ❌ 原子放大项宽 320（>192），GM 访存 +24%，bf16 输出原子累加二次舍入 |
| C3 两 kernel 拆分 | 同 C1 | 同 | 同 | 同 | 需两次 launch | ❌ FLOPs ×2，违反三 launch 契约 |
| C4 物化 | 纯 Cube 友好 | ≥45.8 GB ≫ 128 MiB L2 | — | — | — | ❌ 容量淘汰 |
| C5 纯 Vector | 矩阵乘落错单元 | 同 | — | — | — | ❌ 计算单元不匹配（量级差，无需实测） |
| C6 flat token tile | 同 C1 | 同 | — | 跨 request 边界 | — | ❌ 不等价（R1#10） |
| C7 每 request 一块 | 同 C1 | 同 | 同 | 同 | 长尾 32× | ❌ 并行度不足（768 块 / 28 核 ≈ 27 波，最长块 256 串行迭代） |

**设计期估算**（D-2；协议见 `.agents/skills/_shared/standards/hardware-cost-model.md`）：
- **流量项** = GM 视图 **744 GB**；其中 q/do 的双读（460.7 GB）与 `dq` 原子写（276.4 GB）按 L2 驻留处理：
  每 (b,h) 工作集 = `q 6.3 + qT 6.3 + do 4.2 + doT 4.2 + dq_acc 12.6 ≈ 33.6 MB` « L2 128 MiB，故 HBM 侧
  ≈ 理想 17.2 GB（q 2.1 + do 1.4 + k/v 4.2 + dq 6.6 + dk/dv 2.6 + 冗余）。
- **L1 端口项** = **1.571 TB**（5,623,296 次迭代 × 279,264 B，逐项见上表）。
- **发射项** ≈ 5,623,296 次迭代 × 约 20 个 tile 级 op（4 GM→L1 + 2 GM→UB + 5 GEMM + 2 L0C→UB 半区对
  + 3 UB→L1 半区对 + 2 向量段 + 1 原子 store），外加每块 1 次 O(B) decode 扫描（≤B 次标量读）。
- **容量项** = L0C 62.5%、L1 32.8%、UB 35.7%（均满足）。
- **估算下界: 待核验**（缺少 950 的聚合 GM/L2 带宽与 Cube/Vector 吞吐实测常数——平台 `cost_measurements`
  为空，`knowledge/bottlenecks.md` 明确「不使用 910B 的数值补齐」；`constants.md` 的
  `CONST-L1-port-bw`（148–154 GB/s/核）与 `CONST-fabric-bw`（1.73 TB/s）条目 device 字段为 910B2C，
  不适用本型号，仅作量级参考：1.571 TB / (28 × 150 GB/s ≈ 4.2 TB/s) ≈ 375 µs，**仅量级指示，不作判据**）。
  结构结论（不依赖缺失常数）：本算子的 **L1 端口操作数流是地板项**（24.5 FLOP/字节），提高分块只能小幅度
  改善（备选 128/32 的 L1 端口 −5.9%，见 §1.6.3）。

**调研结论**：选定算法族 **key-stationary 单遍重算（flash 反向族，C1）**，并以 **融合式 running-prefix
块解码**替代 `GroupTiling` 的 warp 扫描（R6）。关键依据：R3 中 C1 与基线 FLOPs 相同而 GM 访存最低可达
（C2 +24%、C3 2×、C4 容量淘汰、C5 单元不匹配、C6 不等价、C7 并行度不足）；R4 中只有 C1 的 5 个乘法能全部
映射到 NT GEMM、块枚举能在无 warp 的核上以 O(B) 标量完成、且片上容量留有余量。相对基线的结构差异：
**同一算法族，按 NPU 约束重构操作数方向（NT-only，`q/do/k` 各多一份转置副本）、累加器从寄存器迁到 L0C、
逐元素原子加换 store-atomic 并以「溢出零行」补上 packed 的边界论证、块表 warp 扫描换融合式标量扫描、
`dk/dv` 写回分内部快路与尾块守卫**。源算子 §0.4 的 15 项优化手段的意图承接见 §0.5 处置列（全部有承接或
明确舍弃理由，无静默丢弃）。下游约束：§1.4 与 §1.6.1–§1.6.3 的分析对象即 C1；§3.1 用 §1.6.1 的优化后公式。

#### 1.6.1 数学等价优化（公式级）

分析对象 = §1.6.0 选定的 C1。A1/A2 为 dense 已验证项（本文重新以 packed 界验证），A3 为 packed 新增的
**不变式**（不是优化项，而是整块原子写回所依赖的等价前提，同样给出机器验证）。

| # | 项 | 原式（基线） | 优化后公式 | 等价性论证 | 收益量化 |
|---|-----|-------------|-----------|-----------|---------|
| A1 | **自然对数域概率**（复用） | `P = exp2(s·scale·log2e − lse·log2e)`（`s·scale` 后逐元素乘 `log2e`，`lse` 亦逐元素重算 `lse·log2e`） | `P = exp(s·scale − lse)` | `exp2(x·log2e) ≡ exp(x)` 恒等式；两式对同一实数输入只差舍入路径，fp64 参照下**各自**距真值 ≤1 个目标 dtype ulp，候选式不劣于基线式（下表两行 A1） | 每 score 少 1 次乘（`lse·log2e` 的逐元素重算）→ 每迭代 4096 次向量乘；`exp` 与 `exp2` 同属 SFU 指令（`T.exp` 有核验条目、`exp2` 无），指令数不减但不引入未核验 API |
| A2 | **块级 mask 特化（packed 运行时界）**（复用 + 重验） | 每个 (key 块, query 块) 都生成 `select(vis, p, 0)`，`vis = (gi<len_kv)&(gj<len_q)&(gi<=gj+shift)` | 运行期谓词 `(n0+64<=len_kv) ∧ (m0+64<=len_q) ∧ (n0+63<=m0+shift)` 为真时直接输出 `p`（不生成掩码），否则同一 `select` | 谓词是「掩码在整块上恒真」的**可判定充分条件**：`n≤m+shift` 对块内最大 `n`（`n0+63`）与最小 `m`（`m0`）成立即全真；`n0+64<=len_kv`/`m0+64<=len_q` 排除填充行/列。故两式逐元素相同。机器验证对 28,261 个块、115,696,896 个元素逐元素比对，违反率 0，且快路块的 `vis` 断言全真 | 快路块占 48.6%（13,734/28,261）；每 score 省 2 次整数比较 + 1 次选择 → 约 4096×2 次向量操作/迭代 |
| A3 | **packed 溢出零行不变式**（新增，整块原子写回的前提） | `dk/dv/dq` 只写有效行（源用逐元素守卫 `if m0+i < len_q`） | `dq` 改**整块**原子写回，越出 `len_q` 的行贡献必须恰为 0；`dk/dv` 改用双分支（R7）而非依赖零值 | 掩码臂使 `gj ≥ len_q` 的列 `P=0 → dS=0`；`dq` 的第 `j` 行 = `Σ_i ds16[j,i]·k[i,:]`，`ds=0` 时 fp32 累加结果是精确 0；因此对下一 request 的 `dq_acc` 行是「原子加精确 0」，数值不变（仅邻居恰为 `-0.0` 时会变 `+0.0`，数值等价）。机器验证：30,528 个溢出差分元素与 3,816 个溢出 dq 行**全部为精确 0**；4,102 个累加器样本在 `+0.0` 下数值不变（1 个 `-0.0 → +0.0` 的零号翻转，数值等价） | 省掉 `dq` 的逐元素原子（源为每元素一次全局原子）与 L0C→UB→GM 的 fp32 往返：每迭代 12,288 个原子操作 → 1 次 64×192 的 store-atomic DMA；`dk/dv` 的慢路只在每 request 的尾半块触发（≤`B·H·2` 个块） |

**A2 的实现机制**：谓词依赖 `n0 = f(tile)`、`m0 = kq·64`、`len_q/len_kv/shift = f(cu_seqlens)`，全是**运行时量**，
故 A2 由 kernel 内的**运行时块级分支**实现（Python 层 `if` 生成 IR 的 IfThenElse，两支各一套半区 epilogue，
编译成两个 SIMT VF + 运行时二选一，见 §3.5.4 探针 P1/P2 的设备码 `simt_vf_1` / `simt_vf_0`）。
编译期唯一可静态判定的是「是否存在尾块」（由 `cu_seqlens` 决定，运行期才知道 ⇒ 连这条在 packed 下也不是
编译期信息），因此两支都必须生成。

**快路的安全性论证**（为什么全可见块可以省掉掩码而不产生 inf/NaN）：快路要求 `m0+64 ≤ len_q`
（块内无填充 query 行）与 `n0+63 ≤ m0+shift`（块内每个 key 对块内每个 query 都可见）——即该块**不可能**
包含「看不到任何 key」的行（`lse=-inf` 只出现在这类行上），故 `exp(s·scale − lse)` 的指数输入有限；
分块外的 `lse=-inf` 行一律落在掩码臂并与掩码一起置 0（`T.if_then_else` 选择，不是乘法，故 `0·inf` 不出现）。

**已确认无需改动的项**（CSE / 降代换自查，写明结论以免被误判为遗漏）：
`v = p*(dpt − delta)` 已是最优形态（`p*dpt − p*delta` 会多一次乘）；
`p` 的 fp32 结果被 `pT`(cast) 与 `dsT`/`ds`(cast) 共用（每元素只算 1 次 `exp`）；
`lse/delta` 已是每 query 行一次装载、按 lane 私有读（无逐元素重算）；
`dk` 的 `scale` 乘只发生在块尾（每 launch `T_kv×192` 次）；
**packed 新增自查**：`q_start/kv_start/len_q/len_kv/shift` 每个 block 只算一次（不是每迭代）；
`cnt/req/n0/live` 每个 block 只算一次；`own`/`kv_end` 每个 block 只算一次。

**弃选项的定量依据**（对应 R1 表 #4/#5/#6/#10）：
`scale` 折进 `q` 需 64×192 = 12,288 次预乘、且 `q` 同时供 `dK`、`k` 同时供 `dQ` 而需两份副本，净亏；
`delta` 的 rank-1 拆分需额外一次 GEMM（+192×64×64 MAC/迭代 ≈ +5.7% 计算量）；
`dV/dK` 合并会把两项求和，不等价，禁止采纳；
flat token tile 会让跨 request 边界的 tile 只解出一个 group（源 docstring 明示），不等价，禁止采纳。

**等价性机器验证（D-1）**：脚本 `examples/mla_varlen_bwd/_bwd_kernel/verify_equiv.py`（torch CPU，fp64 参照，
含 `s·scale−lse ∈ [-8e2, 8e2]` 的随机样本 + 角点值：精确抵消、上/下溢、次正规、`lse=-inf` 空行、`700/−700`
极值；A2 覆盖 manifest 的 5 组 packed 长度集合，含尾块与 `len_q>len_kv` / `len_q<len_kv` / 单 token 请求；
A3 覆盖 6 组请求几何的整块枚举）。本次（2026-11-23，`ASCEND_RT_VISIBLE_DEVICES=""`，toolchain
`ascend-950pr-9579 / TileLang ascend 0.1.15`）实跑输出 `EQUIV_SUMMARY ALL_PASS`：

| 采纳项 | 样本数 | 候选 vs 基线 最大 ulp | 违反率 | 候选 vs fp64 最大 ulp | 基线 vs fp64 最大 ulp | 结论 |
|---|---|---|---|---|---|---|
| A1 exp 域 P（目标 dtype float16） | 1048592 | 1 | 0.000e+00 | 1 | 1 | EQUIV_PASS |
| A1 exp 域 P（目标 dtype bfloat16） | 1048592 | 1 | 0.000e+00 | 1 | 1 | EQUIV_PASS |
| A2 块级 mask 特化 · packed 运行时界（block_n=64, block_m=64） | 115696896 | 0 | 0.000e+00 | — | — | EQUIV_PASS |
| A3 packed 溢出零行不变式（dS/dq 越界行恒 0，`+0.0` 原子累加无副作用） | 38446 | 0 | 0.000e+00 | — | — | EQUIV_PASS |

（A2 快路块 13,734/28,261；分组明细 `ds-v3-8x4k/4req 3968/8192 | ds-v3-mixed 6511/13376 |
kimi-k2-4x8k/2req 3250/6656 | tails+shift 5/31 | tiny 0/6`；A3 明细
`spill_entries=30528 dq_rows=3816 acc=4102 zero_sign_flips=1`。违反率定义：与基线相差超过目标 dtype
1 个 ulp 的元素占比。）

**优化结论**：采纳 2 项公式优化（A1 自然对数域 `exp`、A2 块级 mask 特化）并全部 EQUIV_PASS；A3 是 packed
整块原子写回所依赖的不变式，同样 PASS（30528+3816 个溢出元素/行全为精确 0）。A1 与 dense 结论一致
（同式、同验证口径），A2 在 packed 的**运行期界**下重新验证（这是本函数唯一需要重验的部分），A3 为本函数新验。

**优化后公式**（供 §3.1 使用）：

$$
P[m,n]=\exp\!\big(\mathrm{scale}\cdot s[m,n]-\mathrm{lse}_m\big)\quad\text{all-visible block}\qquad
dS[m,n]=P[m,n]\cdot\big(dP[m,n]-\mathrm{delta}_m\big),\quad
\mathrm{dqAcc\_row}(m\ge len_q)\equiv 0
$$

对角块与尾块再乘因果/区间掩码（`T.if_then_else`，见 A2 的块级特化）。其余乘积与写回形态不变
（`dV = P^T dO`、`dK = scale·dS^T q`、`dq_acc += dS·k`）。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

盘点实现方案中**全部**循环与标量计算点（C1 + §0.6 重设计结构，含 packed 专属的 R6/R7）：

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由（附查证） |
|---|--------|-----------|-------------|---------|----------------------|
| 1 | `P = exp(s·scale − lse)` | GPU `T.Parallel(bn,bm)` 逐元素 | `T.SimtVF(threads=128)` + `T.Parallel(32,64)` + `T.exp` | ✅ | —（`T.exp` 核验：`PL-950-row-primitives.py`、`PL-950-softmax-boundaries.py`；P1/P2 设备码 `__simt_vf__`） |
| 2 | 因果 / 尾块掩码 | 逐元素布尔 + `select` | 同一 `T.Parallel` 内的 `T.if_then_else`（A2 运行期块级特化掉全可见块） | ✅ | —（`if_then_else` 核验：`PL-950-attention.py`；两支 VF 见 P2） |
| 3 | `dS = P·(dP − delta)` + 列广播 | 逐元素 | 同一 `T.Parallel` 内读 `dlt_u[j]`（lane 私有读，无广播指令） | ✅ | —（列/行广播与转置在 SIMT 内表达：`PL-950-conversion-layout.py`） |
| 4 | `pT`/`dsT`/`ds` 的 fp32→fp16 cast | 逐元素 | `T.cast` 写另一个半区 UB buffer | ✅ | —（转换含 ±inf/次正规：`PL-950-conversion-layout.py`） |
| 5 | `ds (Mq,Nk)` 转置副本 | GPU 不需要（`transpose_A`） | 同一 `T.Parallel` 里 `ds_u[j,i] = v`：SIMT 内转置写 | ✅ | —（「转置与列广播在 SIMT 中表达」；P2 设备码 `ascend_nd2nz_scatter`） |
| 6 | `dk` 写回 `cast(dk_c·scale)`、`dv` cast | 逐元素 | 块尾一次 `T.Parallel(32,192)` / `T.Parallel(32,128)` | ✅ | — |
| 7 | `P/dS` 的 K 维归约 | GPU tensor core | Cube mad（`T.gemm`，L0C fp32 累加） | ✅（换单元，非标量循环） | — |
| 8 | `dq_acc += dq` | 逐元素 `atomic_add` | `T.set_atomic("add","float32")` + 一次 L0C→GM `T.copy` | ✅（硬件原子，无标量循环） | —（`PL-950-atomic-splitk.py`；P2 设备码 `asc_set_atomic_add_float` … `asc_copy_l0c2gm` … `asc_set_atomic_none`） |
| 9 | **变长块枚举**（前缀和 + decode） | 源用 warp shuffle 前缀扫描 + 二分 | 无（保留标量 O(B) 扫描） | ❌ | **block 级控制流与寻址**：该量决定「本块是否存活、所有 GM 地址、循环界」，逐元素 API 不适用；且结果必须对 AIC 与两个 AIV 一致，向量化到某 AIV 的 UB 需要未认证的跨核可见性。另：源的 `_tile_prefix_sum` 走 `KernelLaunchFrame.Current().get_thread_extents()` + `T.shfl_up` 的 warp 扫描，而 ascend 的 `T.Kernel` 在 kernel 作用域**显式不建立线程域**（`ascend/language/kernel.py` 模块文档：「no SIMT thread domain at kernel scope」），故该形态不能按源样使用；把 `T.shfl_*` 搬进 `T.SimtVF` 则是另一种机制，其结果会落在某个 AIV 的 UB，跨核可见性未取证。量化：每块 ≤B 次 int32 读 + ≤3B 次整数运算（B ≤ 16 时 ≈ 50 个标量 op），对比每块数千次 GEMM 迭代 = 10⁻³ 量级 |
| 10 | **逐 request 边界标量**（`q_start/kv_start/len_q/len_kv/shift`、`loop_st/loop_ed`、`own/kv_end`） | 源逐元素守卫里的同一批标量 | 无（保留标量） | ❌ | **block 索引**：每块算一次，参与 GM 地址与循环界，逐元素替代会退化成每元素重算（更多工作）。边界：`loop_ed` 取 `max(ceil(len_q/64), loop_st+1)` 以覆盖 `len_q=0` 的退化请求（§5.4 第 6 条） |
| 11 | **`dk/dv` 写回的双分支谓词**（`own+32 <= kv_end`） | 源为逐元素守卫 `if n0+i < len_kv` | 谓词本身保留为块级标量；**两条支路内部都已向量化**（快路 DMA、慢路逐 lane 守卫的并行 store） | ✅（谓词为块级，工作已向量化） | —（P2 设备码同时出现 `asc_copy_ub2gm_align` 与 `asc_copy_ub2gm_align` 的守卫分支 `simt_vf_3`） |
| 12 | **尾半块的逐 lane 守卫 store** | 源逐元素守卫 | `T.SimtVF` 内 `T.Parallel(32,192)` + `if own+i < kv_end` 的 lane 谓词 | ✅（向量化执行，非标量循环） | —（lane 谓词化 GM 写：`examples/paged_cache_write/_paged_cache_write_kernel/DESIGN.md` 的同类形态与外部仓 `test_dynamic_outputs.py` 设备用例；P2 设备码 `simt_vf_3`） |
| 13 | `lse/delta` 行统计装载（head-major） | 逐元素 | `T.copy(lse[h, qoff:qoff+64], lse_u)`（1×64 fp32 = 256 B） | ✅ | —（P2 设备码 `asc_copy_gm2ub_align(..., 1, 256, …)`） |
| 14 | 尾块边界（有效行/列数与 clamp） | 源 `T.if_then_else` 逐元素守卫 + `T.min` 区段 | **无需替代**：装载的越界尾部由平台 `AscendInsertOOBPadding` 改写成 `min()` clamp + `asc_fill_l1` 补 0；写回由 R7 双分支收口 | ✅（由平台 pass + 双分支消除） | —（`ascend/pipeline.py` 该 pass 的职责描述；P2 设备码 `min(64, max((total_kv-kv_start)-n0, 0))` 与 `asc_fill_l1`） |

**向量化结论**：14 个计算点中 **11 项**已向量化/换单元/由平台 pass 消除（#1–#8、#11–#14），保留 **3 项**
（#9 变长块枚举、#10 逐 request 边界标量、#11 的双分支谓词本身）为标量，理由类别 = **block 级控制流与寻址**
（每块一次，非逐元素热点；#9 另有「kernel 级无线程域 / 跨核一致性」两条硬约束）。相对 dense 版新增的向量化点是 #11/#12
（`dk/dv` 双分支与逐 lane 守卫写），新增的保留点是 #9（packed 块枚举）。

#### 1.6.3 向量化轴与数据布局决策 ⭐

> 迁移任务：源算子的轴选择（GPU 线程/warp 映射）只是输入，下面是独立评估。

本算子的核内布局自由度集中在 **(key × query) 的 2-D 中间块**（`P`、`dP`、`dS` 各 64×64）与
**block ↔ (tile, head) 的展平方式**。I/O 契约固定为 THD（尾轴 `D` 连续），`D` 维是矩阵维（Cube 契约），
因此「向量化轴」只能在 key/query 两轴中选。

| 评分项 | 候选 1：lane = query，半区 = key（本设计） | 候选 2：lane = key | 候选 3：UB 内双转置（(Mq,Nk) in UB, (Nk,Mq) in L1） |
|--------|--------------------------------------|-------------------|--------------------------------------------------|
| 整除性 / 尾 lane | Mq=64，fp32 向量宽度 4 → 16 向量/行，无尾 lane | Nk=64，同 | 同 |
| 累加链形态 | 最内轴 stride=1（`(Nk,Mq)` 行内连续），`lse/delta` 为 lane 私有读 | 最内轴是 key，`lse/delta` 需沿 key 轴**列广播** | 同候选 1 |
| repack 代价 | 只需 1 份转置副本（`ds`，半区 32×64 跨步写/迭代） | GEMM 的 A 操作数本就要 `(Nk,Mq)`，lane=key 会让 `P/dS` 的 UB 布局变 `(Mq,Nk)` → 需给 `dV` **和** `dK` 各造一份转置副本（2 份） | 同候选 2（2 份转置） |
| UB 容量影响 | 每 AIV 每迭代 = 2×8192(fp32) + 3×4096(f16) + 512 ≈ 29 KB | 同量级 | +4 KB（多一份半区替身） |
| 弃选量化理由 | ✅ 采纳 | ❌ 转置副本 2 份（+4096 次跨步写 + 4 KB UB/迭代），且每元素多一次列广播 | ❌ 同上 |
| 候选 4：纯 SIMT 直写 GM | AIC 不参与 | — | ❌ 5 个 64×64×192 级矩阵乘落 56 个 Vector 核，算力差 1–2 个量级（R4 C5） |
| 候选 5：中间块经 GM 往返（`store_fixpipe`） | — | — | ❌ 每迭代多 3×8 KB 的 GM 往返（全量约 +276 GB HBM 流量；`CONST-store-fixpipe-gm-only` 的 910 结论在 950 上已被 L0C→UB 双目标搬运取代，见 `PL-950-mixed-epilogue.py`） |
| 候选 6：block 展平 `h` 外层（dense 的写法） | `h = bid // ntiles_dyn` | — | ❌ **动态除数**（`ntiles` 含 `total_kv`）：运行期整数除法未取证；且相邻 bid 跨 head 导致同行不同 head 的切片不连号，L2 行复用变差 |
| 候选 7：block 展平 `h` 内层（本设计） | `tile = bid // H`（静态除数）、`h = bid % H` | — | ✅ 采纳（静态除数 + 同行相邻 head 的 L2 复用） |

**多 AIV 的执行体映射**：本平台把线程域限定在 `T.SimtVF` 内；显式 `T.Vector(vector=2) as sid` 在本工具链
**不可编译**（`InternalError … dup sid`，dense §3.5.4 反例 2）。本设计沿用 dense 真机验证的形态：半区向量段
（`T.Parallel(32,64)`）+ `sid = T.call_extern("int64","asc_get_sub_block_id")` 在 kernel body 取一次并被
`T.SimtVF` 捕获，L0C↔UB 与 UB→L1 全部 `T.dual_copy` 2:1，两个 AIV 各持 32 个 key 行。

**布局决策结论**：选定候选 1 + 候选 7——核内布局为 **(key, query)** 且**向量化轴 = query**（lane 映射到
`Mq=64` 的最内轴），中间块以 `(Nk,Mq)` 整块存在 L0C（按 key 行 2:1 分给两个 AIV 的 UB）、仅 `dS` 额外产一份
`(Mq,Nk)` 转置副本供 `dQ` 的 NT 形式使用；block 展平取 `tile = bid // H, h = bid % H`（静态除数 + L2 复用）；
统计量保持 head-major `[H,T_q]`（`lse/delta` 的一次 256 B 连续搬运）。该结论已同步到 §3.3 伪代码
（`T.Parallel(32, 64)` 的内层为 query 维）、§4.5（半区 UB 预算）与 §6 循环结构。

**备选方案与实验裁决计划**（判定裕度依赖未实测常数：L1 端口带宽与 Cube 吞吐，见 §1.6.0 估算行）：

- **主选（进 Stage 3）**：`block_n=64, block_m=64`。L0C 163840 B/262144（62.5%），L1 172032 B（32.8%），
  UB 90624 B（35.7%）；每迭代 L1 端口 279280 B、24.5 FLOP/字节，5,623,296 次迭代合计 1.571 TB；
  GM 视图 744 GB。
- **备选（≤1 个）**：`block_n=128, block_m=32`。L0C = `dk 98304 + dv 65536 + st 16384 + dpt 16384 +
  dq 24576 = 221184 B`（216 KiB，84.4%）；L1 = `k_s 49152 + v_s 32768 + kT_s 49152 + q_s 12288 + qT_s 12288
  + do_s 8192 + doT_s 8192 + pT_s 8192 + dsT_s 8192 + ds_s 8192 = 196608 B`（37.5%）；
  UB：每迭代半区（`_HN=64`）`st_u/dp_u 2×16384 + p16_u/dsT_u 2×8192 + ds_u 4096 + 512 ≈ 55 KB`
  + 块尾半区 `tail_u 49152 + tailv_u 32768 + out_k 24576 + out_v 16384 = 122880`（须按半区分段写回）
  → 约 178 KB（70%）；每迭代 L1 端口 ≈ 263.2 KB、26.0 FLOP/字节（−5.8%），迭代数 ≈5.62M（ceil 结构
  略变）→ 合计 ≈1.480 TB（−5.8%）；代价是 L0C 余量降到 15.6%、GEMM 的 N 维（query）降到 32、块尾缓冲
  逼近 UB 上界。
- **裁决计划**：代表 shape 取 `ds-v3-8x4k`（B=8×4096）与 `ds-v3-mixed`；指标 = kernel Task Duration
  （`msopprof`，`msprof_task_duration`）+ 反解 L1 端口有效带宽；判定阈值 = 备选 latency 优于主选 ×0.9 才翻转
  （整体或按 `len_kv` 分片经工厂缓存分派）；实测数据回写本节与 §1.6.0 估算行。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式：混合（Cube + Vector）。**（平台 `programming_modes.choices` 的 `hybrid`；`migration_default`
为 `simd`，偏离理由见 2.2，属有证据的偏离。）

### 2.2 选型理由

- 算子含 5 个 64×64×192 级矩阵乘（每迭代约 3.4M MAC），**必须落 Cube**（`T.gemm`）；若走 SIMD/SIMT
  纯向量路径，56 个 Vector 核的 lane 级乘加减比 Cube 分形低 1–2 个量级（§1.6.0 R4 C5）。
- 同时每迭代必须做 `exp`、掩码、cast、跨步转置等逐元素工作，**必须落 Vector（AIV）**；平台把线程域
  限定在 `T.SimtVF` 内（`.agents/platforms/toolchains/ascend/references/programming.md`「API 与执行区域」）。
- 两侧必须在**同一个 kernel** 内交替（中间块 `P/dS` 不落 GM），平台支持该形态：`T.Kernel` 单核网格 +
  自动调度把 op 分派到 AIC/AIV（`AssignCore`/`ResolveCore`/`InsertSync`）；本设计的 packed 骨架已在本轮
  **编译级**走通（§3.5.4 探针 P1/P2，设备码 `__mix__(1, 2)`），且 dense 版的同一骨架已**真机通过**
  （L0 13/13、L1 6/6）。
- 平台默认 `migration_default=simd` 面向逐元素/规约类算子；本算子的主导算力在矩阵乘，选 `simd` 会把
  矩阵乘降级到向量单元，因此按上述证据选 `hybrid`。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_shared`（UB，向量侧，半区）、`T.alloc_l1`（GEMM 操作数）、`T.alloc_l0c`（fp32 累加器）；L0A/L0B 由 GEMM lowering 内部管理，不显式分配 |
| 计算方式 | Cube：`T.gemm(A_l1, B_l1, C_l0c, transpose_B=True, clear_accum=…)`，A/B/C 一律用 allocation 全 extent；Vector：`with T.SimtVF(threads=128): for i, j in T.Parallel(32, 64)` |
| 执行体 | **不写 `T.Vector`**。半区向量段由 `T.SimtVF(threads=128)` 表达；副块 id 由 `asc_get_sub_block_id()` 在 body 取一次并被 VF 捕获（dense 真机验证） |
| 同步 | 自动（默认 `allow_autoschedule=True`，AutoSchedule + InsertSync 插入）→ §7 |
| 跨单元数据 | L0C→UB 用 `T.dual_copy` 2:1（半区）；UB→L1 用 `T.dual_copy`（`InsertNd2Nz` 改写成 NZ scatter）；L0C→GM 用 `T.copy` + store-atomic；`dk/dv` 写回双分支（R7） |
| 边界 | 装载靠平台 OOB clamp（按张量 extent）+ 掩码收口；写回靠双分支（按 request 边界） |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入为 §1.6.1 优化后公式；每个步骤与 §1.6.2 的向量化结论一致。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 0 | `(req, n0, live, q_start, kv_start, len_q, len_kv, shift)` | 融合式 O(B) 扫描（R6），`live==0` 早退 |
| 1 | `s^T = k_s · q_s^T` | NT GEMM，A=`k_s (64,192)`、B=`q_s (64,192)`，C=`st_c (64,64)` fp32 |
| 2 | `dP^T = v_s · do_s^T` | NT GEMM，A=`v_s (64,128)`、B=`do_s (64,128)`，C=`dpt_c (64,64)` |
| 3 | `p^T = exp(s^T·scale − lse[j]) ⊙ mask` | 向量（lane=query，半区 32×64），A2 运行时块级特化 |
| 4 | `pT = f16(p^T)`；`dsT = f16(p^T ⊙ (dP^T − delta[j]))`；`ds = f16(同一值)`（转置写） | 向量 + cast；`p^T` 只算一次 `exp`（CSE） |
| 5 | `dV = pT · doT^T` | NT GEMM，A=`pT_s (64,64)`、B=`doT_s (128,64)`，C=`dv_c (64,128)` 累加 |
| 6 | `dK = scale · dsT · qT^T` | NT GEMM，A=`dsT_s (64,64)`、B=`qT_s (192,64)`，C=`dk_c (64,192)` 累加（`scale` 在块尾乘） |
| 7 | `dQ = ds · kT^T` | NT GEMM，A=`ds_s (64,64)`、B=`kT_s (192,64)`，C=`dq_c (64,192)` 每迭代清零 |
| 8 | `dq_acc[q 行] += dQ` | store-atomic + 整块 `T.copy`；溢出行为精确 0（A3） |
| 9 | `dk[kv 行] = cast(dk_c·scale)`、`dv[kv 行] = cast(dv_c)` | 半区 cast + 双分支写回（内部快路 / 尾块守卫） |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API（`tilelang.ascend.language`） | 参数 | 模式 |
|------|----------|--------------------------------------------|------|------|
| 0 | 任务映射 | `T.Kernel(grid)` 一维核网格 | `grid = (total_kv // 64 + B) * H`（`total_kv` 为 `T.dynamic`）；`T.Kernel` 不接受 `threads` | — |
| 0' | 块枚举 | 纯标量：`T.serial(B)` + `T.alloc_local([1],"int32")` | `_decode` 宏（§3.3）；`live` 谓词 | 标量 |
| 1 | `k/v/kT` 常驻 | `T.copy(GM region, l1_buf)`（`kT` 加 `transpose=True`）；dst 用**全 allocation extent** | `k[koff:koff+64, h, :] → k_s`；`kT_s` 走 dn2nz | MTE2 |
| 2 | `q/do` 双方向装载 | 同上（两份：ND 与 `transpose=True`） | `q_s (64,192)`、`qT_s (192,64)`，偏移含 `q_start` | MTE2 |
| 3 | `lse/delta`（head-major） | `T.copy(lse[h, qoff:qoff+64], lse_u)` | 一维 fp32 区段（256 B） | MTE2 |
| 4 | 5 个 NT GEMM | `T.gemm(A, B, C, transpose_B=True, clear_accum=cb)`；A/B/C 一律**全 extent** | 见 §3.1；A/B 在 L1、C 在 L0C；`cb` 为 `True` 或 `kq == loop_st` | Cube |
| 5 | L0C→UB（半区） | `T.dual_copy(st_c, st_u)` / `T.dual_copy(dpt_c, dp_u)` | 2:1 M-split，每 AIV 32 个 key 行，fp32 同 dtype | AIC fixpipe |
| 6 | `exp/mask/dS/cast/转置写` | `with T.SimtVF(threads=128): for i, j in T.Parallel(32, 64)` | `T.exp`、`T.if_then_else`、`T.cast`；A2 为运行时块级 `if` 两支；`sid` 捕获 | Vector |
| 7 | UB→L1（半区） | `T.dual_copy(p16_u, pT_s)`、`T.dual_copy(dsT_u, dsT_s)`、`T.dual_copy(ds_u, ds_s)` | 由 `InsertNd2Nz` 改写为 NZ scatter；L1 侧仍是整块 64×64 | MTE |
| 8 | `dq_acc` 累加 | `T.set_atomic("add","float32")` → `T.copy(dq_c, dq_acc[qoff:qoff+64, h, :])` → `T.set_atomic_none()` | L0C fp32 → GM fp32 | AIC fixpipe |
| 9 | `dk/dv` 写回 | `T.dual_copy(dk_c, tail_u)` → 向量 `cast`（`dk` 乘 `scale`）→ 双分支：`T.copy(out_k, dk[own:own+32, h, :])` / `T.Parallel` + `if own+i < kv_end` | 半区；尾块按 request 边界收口 | AIC + Vector |
| 10 | 多缓冲 | 首版不加 `annotate_buffer_versions`（`num_stages=1` 已钉死单版本，dense 实测如此） | §6.3 | — |

### 3.3 计算伪代码

```python
BLOCK_N, BLOCK_M = 64, 64       # keys / queries
HN = 32                         # 每 AIV 的 key 行（dual_copy 2:1）
D_QK, D_V = 192, 128
NTILE_MAX = "<运行期> total_kv // BLOCK_N + batch"   # 动态上界（tile_upper_bound）
GRID = NTILE_MAX * heads

@T.prim_func
def _bwd_main(q, k, v, do, lse, delta, cu_q, cu_kv, dq_acc, dk, dv):
    with T.Kernel(GRID) as bid:
        tile = bid // heads                     # 静态除数
        h    = bid %  heads                     # 内层 head → L2 行复用
        sid  = T.call_extern("int64", "asc_get_sub_block_id")
        <L1 / L0C / UB 半区 buffer 分配>          # 预算见 4.5 / 4.6

        # ---- R6：融合式 (前缀和 + decode)，O(batch) 串行，local 标量 ----
        req = 0; n0 = 0; cnt = 0
        for g in T.serial(batch):
            nt = (cu_kv[g + 1] - cu_kv[g] + BLOCK_N - 1) // BLOCK_N
            if (tile >= cnt) & (tile < cnt + nt):
                req = g
                n0  = (tile - cnt) * BLOCK_N
            cnt = cnt + nt
        live = tile < cnt
        if live:
            q_start  = cu_q[req]
            kv_start = cu_kv[req]
            len_q    = cu_q[req + 1]  - q_start
            len_kv   = cu_kv[req + 1] - kv_start
            shift    = len_kv - len_q
            koff     = kv_start + n0

            # ---- 全 extent 装载（尾部由 OOB pass 按张量 extent clamp + 补 0）----
            T.copy(k[koff : koff + BLOCK_N, h, :], k_s)
            T.copy(v[koff : koff + BLOCK_N, h, :], v_s)
            T.copy(k[koff : koff + BLOCK_N, h, :], kT_s, transpose=True)

            loop_st = T.max(0, (n0 - shift) // BLOCK_M) if is_causal else 0
            loop_ed = T.max(ceildiv(len_q, BLOCK_M), loop_st + 1)   # 覆盖 len_q == 0
            for kq in T.Pipelined(loop_st, loop_ed, num_stages=1):
                m0   = kq * BLOCK_M
                qoff = q_start + m0
                T.copy(q[qoff : qoff + BLOCK_M, h, :], q_s)
                T.copy(do[qoff : qoff + BLOCK_M, h, :], do_s)
                T.copy(q[qoff : qoff + BLOCK_M, h, :], qT_s, transpose=True)
                T.copy(do[qoff : qoff + BLOCK_M, h, :], doT_s, transpose=True)
                T.copy(lse[h, qoff : qoff + BLOCK_M], lse_u)      # head-major
                T.copy(delta[h, qoff : qoff + BLOCK_M], dlt_u)
                T.gemm(k_s, q_s, st_c, transpose_B=True, clear_accum=True)
                T.gemm(v_s, do_s, dpt_c, transpose_B=True, clear_accum=True)
                T.dual_copy(st_c, st_u); T.dual_copy(dpt_c, dp_u)  # 每 AIV 32 行
                fast = (n0 + BLOCK_N <= len_kv) & (m0 + BLOCK_M <= len_q) \
                       & (n0 + BLOCK_N - 1 <= m0 + shift)           # 运行期谓词
                if fast:
                    with T.SimtVF(threads=128):
                        for i, j in T.Parallel(HN, BLOCK_M):        # 内层 = query（lane 轴）
                            p = T.exp(st_u[i, j] * scale - lse_u[j])
                            p16_u[i, j] = T.cast(p, dt)
                            dsv = p * (dp_u[i, j] - dlt_u[j])
                            dsT_u[i, j] = T.cast(dsv, dt)
                            ds_u[j, i]  = T.cast(dsv, dt)           # 转置副本
                else:
                    with T.SimtVF(threads=128):
                        for i, j in T.Parallel(HN, BLOCK_M):
                            gi = n0 + sid * HN + i
                            gj = m0 + j
                            vis = (gi < len_kv) & (gj < len_q)
                            if is_causal:
                                vis = vis & (gi <= gj + shift)
                            p  = T.exp(st_u[i, j] * scale - lse_u[j])
                            pv = T.if_then_else(vis, p, 0.0)
                            p16_u[i, j] = T.cast(pv, dt)
                            dsv = pv * (dp_u[i, j] - dlt_u[j])
                            dsT_u[i, j] = T.cast(dsv, dt)
                            ds_u[j, i]  = T.cast(dsv, dt)
                T.dual_copy(p16_u, pT_s); T.dual_copy(dsT_u, dsT_s); T.dual_copy(ds_u, ds_s)
                T.gemm(pT_s,  doT_s, dv_c, transpose_B=True, clear_accum=(kq == loop_st))
                T.gemm(dsT_s, qT_s,  dk_c, transpose_B=True, clear_accum=(kq == loop_st))
                T.gemm(ds_s,  kT_s,  dq_c, transpose_B=True, clear_accum=True)
                T.set_atomic("add", "float32")
                T.copy(dq_c, dq_acc[qoff : qoff + BLOCK_M, h, :])    # 溢出行为精确 0（A3）
                T.set_atomic_none()

            # ---- 块尾：dk = cast(dk_c * scale)、dv = cast(dv_c) ----
            T.dual_copy(dk_c, tail_u); T.dual_copy(dv_c, tailv_u)
            with T.SimtVF(threads=128):
                for i, d in T.Parallel(HN, D_QK):
                    out_k[i, d] = T.cast(tail_u[i, d] * scale, dt)
                for i, d in T.Parallel(HN, D_V):
                    out_v[i, d] = T.cast(tailv_u[i, d], dt)
            own    = koff + sid * HN          # 本 AIV 的 32 行起点
            kv_end = kv_start + len_kv
            if own + HN <= kv_end:            # R7 快路：整块 DMA
                T.copy(out_k, dk[own : own + HN, h, :])
                T.copy(out_v, dv[own : own + HN, h, :])
            else:                             # R7 慢路：逐 lane 守卫（每 request 最多 1 次）
                with T.SimtVF(threads=128):
                    for i, d in T.Parallel(HN, D_QK):
                        if own + i < kv_end:
                            dk[own + i, h, d] = out_k[i, d]
                    for i, d in T.Parallel(HN, D_V):
                        if own + i < kv_end:
                            dv[own + i, h, d] = out_v[i, d]
```

注（逐条对应 packed 的三处差异）：
1. **块枚举**：R6 的单次扫描（差异 ①），`grid` 用 `tile_upper_bound` 同式作动态上界，`live` 丢弃越界 tile；
   静态化备用方案见 §5.4（探针 P5）。
2. **统计量**：`lse[h, qoff:qoff+64]`（差异 ②），未转置成 `[B,H,S_q]`。
3. **边界**：装载靠平台 clamp（张量 extent）+ 掩码；`dq` 整块原子（溢出零行 A3）；`dk/dv` 双分支（差异 ③）。
4. **GEMM 全 extent**：没有任何 `[:an]/[:am]` 切片——L1-GEMM 的 M/N/K 取自 region extent，符号量会直接
   `ValueError`（dense §3.5.4 反例 1，本轮 P1/P2 沿用同一写法）。
5. **无 `T.Vector`**：只有 `sid = asc_get_sub_block_id()`（dense 真机验证形态）。

### 3.4 API 可行性确认

| API | 来源 / 核验状态 |
|-----|----------------|
| `T.Kernel(grid)`（一维、无 `threads`、**grid 可为动态表达式**） | `ascend/language/kernel.py`（源码：一维核网格，`threads=` 被 Python 拒绝）；动态 grid 见 `examples/paged_cache_write/_paged_cache_write_kernel/DESIGN.md`（`T.Kernel(ceildiv(tokens, R)*nchunks)`，设备验证）；本轮 P1/P2/P5 编译级确认（P2 设备码的 kernel 实参含 `total_kv/total_q`） |
| `T.dynamic("total_q"/"total_kv")` | `tilelang/language/symbolics.py`；同仓 `examples/paged_cache_write` 与 `examples/rope/_rope_positions_kernel` 的既有用法；本轮 P1/P2 编译级确认（设备码 `min(64, max((total_kv - kv_start) - n0, 0))`） |
| `T.alloc_l1` / `T.alloc_l0c` / `T.alloc_shared` | `ascend/language/allocate.py`；`PL-950-cube-tile.md`、`PL-950-gemm-tiles.py`（已核验） |
| `T.gemm(A_l1, B_l1, C_l0c, transpose_B=True, clear_accum=…)` | `gemm_op.py` → `GemmMAD._lower_l1`（**只支持 NT**，含断言）；dense 真机通过；本轮 P2 设备码 5 个 `ascend_gemm_l1`，`clear_accum` 形参为运行期布尔 |
| `T.dual_copy`（L0C→UB 2:1、UB→L1 2:1） | `copy_op.py:dual_copy` 文档（L0C→UB 为硬件双目标；UB→L1 按 `cthread` sid 分片 + `InsertNd2Nz`）；`PL-950-mixed-epilogue.py`（已核验）；dense 真机通过（整块 L0C→UB 只落一个 AIV → 必须 2:1） |
| `T.copy` GM→L1（含 `transpose=True` dn2nz） | `copy_op.py` 文档 + `PL-950-gemm-tiles.py`；P2 设备码 `asc_copy_gm2l1_nd2nz` / `asc_copy_gm2l1_dn2nz` |
| `T.copy` GM→UB（head-major 1-D 区段） | P2 设备码 `asc_copy_gm2ub_align(..., 1, 256, ...)` |
| `T.copy` UB→GM（整块快路） | P2 设备码 `asc_copy_ub2gm_align(..., min(32, max(...)), 384, ..., 768, 384)` |
| GM 直接逐元素写（尾块守卫路） | `T.SimtVF` 内 `T.Parallel` + `if` 的 lane 谓词存储；`examples/paged_cache_write/_paged_cache_write_kernel/DESIGN.md` 的设备用例（`test_dynamic_outputs.py`、`test_ascend_fractal_storage_integration.py`）；P2 设备码 `simt_vf_3` |
| `T.set_atomic("add","float32")` + L0C→GM `T.copy` | `mode.py:set_atomic` + `PL-950-atomic-splitk.py`；P2 设备码 `asc_set_atomic_add_float` → `asc_copy_l0c2gm` → `asc_set_atomic_none` |
| `T.SimtVF(threads=128)` + `T.Parallel` + `T.exp` + `T.if_then_else` + `T.cast` | `PL-950-row-primitives.py`、`PL-950-softmax-boundaries.py`、`PL-950-attention.py`；dense 真机通过；P2 设备码 `__simt_vf__ __launch_bounds__(128)` + `asc_vf_call` |
| SIMT 内转置写 | `PL-950-conversion-layout.py`；P2 设备码 `ascend_nd2nz_scatter` |
| `T.call_extern("int64","asc_get_sub_block_id")` | dense 真机通过（`sid` 在 body 取、VF 捕获）；P2 设备码 `__mix__(1, 2)` + `sid` |
| `T.serial(B)` + `T.alloc_local([1], "int32")`（block 级标量） | `examples/moe_permute_align/_make_align_kernel` 的既有用法；P2 设备码 `for (g=0; g<3; ++g)` |
| `T.clear` 作用于 **L0C** | **不可用**：探针 P4 实测 `InternalError: Unsupported scope shared.l0c`（§3.5.4 反例 1）→ 累加器初值仍由 `clear_accum=(kq == loop_st)` 表达 |
| `T.Vector(vector=2) as sid` | **不可用**：`InternalError … dup sid`（dense §3.5.4 反例 2）。本设计不使用 |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束（来源） | 是否涉及 | 处理方案 |
|--------------|---------|----------|
| `T.Kernel` 只接受一维核网格、无 `threads`（programming.md） | 是 | 一维 `grid = (total_kv//64 + B) * H`，块内解码（R1/R6） |
| L1 GEMM 只支持 `trans_A=False, trans_B=True`（`gemm_mad.py:lower` 断言） | 是 | 5 个乘法全部 NT，`q/do/k` 各多一份转置副本，`dS` 多一份转置副本（R3） |
| **L1-GEMM 的 M/N/K 取自 region extent，必须编译期静态**（`gemm_op.py`，符号量抛 `ValueError`） | 是 | GEMM 的 A/B/C **一律 allocation 全 extent**；尾块由「全 extent 装载 + 掩码臂 + 平台 clamp」处理（§5.4）；动态 extent **只出现在 copy 与 grid**（P1/P2 已验证） |
| L0C 是唯一可累加层，且必须带 Ascend fractal layout（`gemm_mad.py:_prepare_lower`） | 是 | 五个 L0C 累加器，共 163840 B（§4.6） |
| **`T.clear` 不支持 L0C**（P4 实测） | 是 | 不写 `T.clear(dk_c)`；首迭代清零由 `clear_accum=(kq == loop_st)` 表达，并用 `loop_ed = max(ceil(len_q/64), loop_st+1)` 保证循环非空（§5.4 第 6 条） |
| L0C→UB 不支持 dtype 转换（同 dtype 断言） | 是 | L0C→UB 保持 fp32，cast 在 UB 内用 `T.cast` |
| **整块 L0C→UB 只落到一个 AIV**（dense 真机 P1：99.2% 错位） | 是 | 所有 L0C↔UB 交叉用 `T.dual_copy` 2:1（`_HN=32`），与 `sid` 的索引一致 |
| 线程域只在 `T.SimtVF` 内（programming.md） | 是 | 所有逐元素工作包在 `T.SimtVF(threads=128)` |
| 显式 `T.Vector(vector=2)` 不可编译（`dup sid`） | 是 | 不使用；`sid` 用 `T.call_extern` 取 |
| `dual_copy` 要求恰有一维 2:1 且 dst 不能是带单例中轴的 GM region（`copy_op.py`） | 是 | `dual_copy` 只用于 L0C↔UB 与 UB→L1；GM 写回用普通 `T.copy`（快路）或逐 lane 存储（慢路） |
| 原子存储支持 dtype 集合 float32/float16/bfloat16/int8/16/32（`mode.py`） | 是 | `dq_acc` 为 fp32，在支持集内 |
| 未认证项：手工跨核 flag、循环内复杂同步、跨 block barrier（programming.md） | 是 | 不写手工 flag，全部依赖 AutoSchedule + InsertSync；不引入跨 block barrier（§7） |
| **动态 grid + 混合 kernel**（本平台未组合验证过） | 是 | 编译级已通（P1/P2/P5）；运行期列入 Stage 3 首跑验证，静态 grid 备用方案已备好（§5.4） |
| 动态 shape | 是 | `T_q/T_kv` 为 `T.dynamic`；`B/H/D_qk/D_v/block_n/block_m` 编译期常量；`len_q/len_kv/shift` 全为运行期量并只进标量/谓词/copy 偏移 |

#### 3.5.2 参考实现差异说明

| 差异项 | 参考实现（GPU TileLang） | 本项目（TileLang ascend） | 转换方案 |
|--------|------------------------|--------------------------|---------|
| Kernel 结构 | `T.Kernel(num_tiles, heads, threads=256)` 二维网格（`num_tiles` 含动态量） | 一维核网格 + AIC/AIV 自动分派 + 动态 grid | §0.6 R1 |
| 任务枚举 | `GroupTiling`：warp shuffle 前缀和 + 二分 + `live` | 融合式 O(B) 标量扫描 + `live` | §0.6 R6 |
| 矩阵 API | `T.gemm(fragment)`，C 在寄存器，支持 `transpose_A` | `T.gemm(A_l1, B_l1, C_l0c)`，仅 NT，全 extent | §0.6 R2/R3 |
| 累加器 | `T.alloc_fragment` | `T.alloc_l0c` + 半区 UB 中转 | §0.6 R2 |
| 统计量 layout | packed：`(heads, total)`；dense：`(B,H,S)` | 保持 packed 的 `(heads, total_q)` | 差异 ② |
| 原子累加 | `T.atomic_add` 逐元素（含行守卫） | `set_atomic` + 整块 store（溢出零行） | §0.6 R5 + A3 |
| `dk/dv` 写回 | 逐元素守卫 `if n0+i < len_kv` | 双分支（内部整块 / 尾块逐 lane） | §0.6 R7 |
| 指数 | `T.exp2`（log2 域） | `T.exp`（自然域） | §1.6.1 A1 |
| 掩码 | 逐元素 `if_then_else` | 同 + 运行期块级特化（界为 `len_q/len_kv`） | §1.6.1 A2 |
| 流水 | `T.Pipelined(num_stages=2)`（cp.async） | `num_stages=1`（首版） | §6.3 |
| 尾块 | `T.if_then_else` 逐元素守卫 + 区段截断 | 装载：平台 clamp + 掩码；写回：双分支 | §5.4 |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/mla_dense_bwd/_bwd_kernel/_bwd_kernel.py` | **极高**（同一源码工厂的 dense 分支，真机通过） | 半区 `dual_copy`、`sid` 捕获、5 个 NT GEMM、store-atomic、块尾 cast、golden 与 L0 套件写法 |
| `examples/mla_dense_bwd/_bwd_kernel/DESIGN.md` | **极高**（同一算子的 dense 设计 v1） | 章节组织、四态处置、编译探针范式、预算表口径、L0 计划 |
| `examples/TileOPs/tileops/kernels/attention/mla_varlen_bwd/_mla_varlen_bwd_kernels.py` | 高（本函数的 GPU 原样抽取） | packed 分支语义（`GroupTiling`、head-major 统计量、逐元素守卫） |
| `examples/paged_cache_write/_paged_cache_write_kernel/DESIGN.md` | 中高（同平台**动态 extent + 动态 grid** 先例） | `T.dynamic` + 动态 1-D grid、尾块 lane 谓词、GM 直写形态 |
| `.agents/platforms/toolchains/ascend/knowledge/repro/PL-950-mixed-epilogue.py` | 高（Cube GEMM + Vector + `dual_copy` 同 kernel） | 混合模式写法、L0C→UB 落地、同 dtype 约束 |
| `.agents/platforms/toolchains/ascend/knowledge/repro/PL-950-gemm-tiles.py` | 高（GM→L1→GEMM、转置装载、动态 region） | `transpose=True` 的 dn2nz、`clear_accum` 首块清零、`T.Pipelined` |
| `.agents/platforms/toolchains/ascend/knowledge/repro/PL-950-atomic-splitk.py` | 中（store-atomic 累加） | `set_atomic` + `T.copy(C)` 的配对与 host 预清零契约 |
| `.agents/platforms/toolchains/ascend/knowledge/repro/PL-950-conversion-layout.py` | 中（SIMT 转换 + 转置写 + 列广播） | `T.cast` 的 ±inf/次正规行为、转置写形态 |
| `.agents/platforms/toolchains/ascend/kernel_template.py` | 中（kernel 骨架） | 一维 `T.Kernel`、`T.SimtVF` 包逐元素、静态尾块分支 |
| `examples/mla_varlen_bwd/_bwd_kernel/history_version/design_probe_compile.py` | —（本设计的编译探针） | §3.5.4 五个探针 + 可复现命令 |

#### 3.5.4 编译探针证据（无设备，仅前端 + 编译）

脚本：`examples/mla_varlen_bwd/_bwd_kernel/history_version/design_probe_compile.py`；执行（必须离开仓根，
否则仓内 dev root 的 `tilelang/` 会遮蔽 site-packages 的 0.1.15 私有 TVM）：

```bash
cd /tmp && ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python \
  /workspace/tilelang-mlir-ascend-950/examples/mla_varlen_bwd/_bwd_kernel/history_version/design_probe_compile.py
```

实测输出（2026-11-23，toolchain `ascend-950pr-9579 / TileLang ascend 0.1.15`）：

| 探针 | 结构 | 实测结果 |
|------|------|---------|
| **P1 `packed_skeleton`** | 动态 `total_q/total_kv` + **动态 grid** `H*(total_kv//64+B)` + 运行期偏移装载 + 5 个全 extent NT GEMM + `dual_copy` 2:1 + `sid` + store-atomic + 块尾整块写回 | **COMPILE_OK**，源 40956 B；设备码含 `__mix__(1, 2)`、`ascend_gemm_l1`×5、`asc_copy_l0c2ub`、`ascend_nd2nz_scatter`、`asc_copy_l0c2gm`（位于 `asc_set_atomic_add_float`/`asc_set_atomic_none` 之间）、`asc_get_sub_block_id`、`asc_fill_l1` |
| **P2 `packed_full`** | P1 + **融合式 (前缀和 + decode)** + `live` 整块包裹 + head-major `lse/delta` + 运行期 mask + 块尾**双分支** | **COMPILE_OK**，源 27084 B；设备码：`for (g=0; g<3; ++g) { ntg = ((cu_kv[g+1]+63) - cu_kv[g]) >> 6; if (cnt <= tile && tile < cnt+ntg) {...} cnt += ntg; }`、`if (tile < cnt)` → `live`；装载 clamp `min(64, max((total_kv - kv_start) - n0, 0))` + `asc_fill_l1`；`lse` 为 `asc_copy_gm2ub_align(..., 1, 256, ...)` 的连续 1-D 段；`dq` 为 `asc_copy_l0c2gm(..., min(64, max((total_q-q_start)-kq*64, 0)), ...)` 夹在 `asc_set_atomic_add_float`/`asc_set_atomic_none` 之间；块尾双分支 = `asc_copy_ub2gm_align`（`if own+32 <= kv_end`）与 `asc_vf_call<main_kernel_simt_vf_3>`（逐 lane 守卫）；共 4 个 SIMT VF（快路 / 掩码臂 / 块尾 cast / 尾块守卫存储） |
| **P3 `store_symbolic_rows`** | 块尾与原子写回改用**符号长度**区段（`nr = min(HN, kv_end-own)`、`nq = min(64, q_end-qoff)`） | **COMPILE_OK**，源 25066 B → 登记为 R7 的回退形态 |
| **P4 `l0c_clear`** | `T.clear` 作用于 L0C 累加器（空循环的防御性初始化） | **COMPILE_FAIL**：`InternalError: Unsupported scope shared.l0c` → 不采用；改用 `loop_ed = max(ceil(len_q/64), loop_st+1)` 保证循环非空 |
| **P5 `static_grid_live`** | **静态化方案**：grid 取编译期上界 `H*(TK_MAX//64+B)`（`TK_MAX=4096`），size 仍为 `T.dynamic`，同一 `live` 谓词丢弃越界 tile | **COMPILE_OK**，源 27084 B（与 P2 同规模）→ 静态 grid 是可直接替换的备用形态 |
| **P2b（对照）** | P2 + `T.clear(dk_c)` | **COMPILE_FAIL**（同 P4）→ 确认该限制与 kernel 形态无关 |

**反例（不接受的设计形态，本机实测）**：

| 形态 | 实测错误（原文） |
|------|-----------------|
| `T.clear` 作用于 `T.alloc_l0c` 缓冲 | `InternalError: Unsupported scope shared.l0c`（P4、P2b） |
| `T.gemm` 用符号 region extent（`x[:an, :]`） | `ValueError: T.gemm requires static tile dimensions, but M is symbolic: an`（dense §3.5.4 复现，本设计中所有 GEMM 仍用全 extent） |
| `with T.Vector(vector=2) as sid:` 与 `dual_copy`/UB→L1/GEMM 混用 | `InternalError … Need input to be in SSA form dup sid`（dense §3.5.4；本设计不使用） |
| 在 `@T.prim_func` 体内调用普通 Python 函数（未标 `@T.macro`）承载循环 | `TypeError: 'ForFrame' object is not iterable`（本轮首次探针即因此失败 → 计算体改用 `@T.macro`，与源 `GroupTiling` 的写法一致） |

说明：以上为**编译级**证据，不含设备执行；运行期精度与性能仍待 Stage 3（前置探针清单见 §8.3）。

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `q` | `(T_q, H, 192)` | `float16 \| bfloat16` | `T_q = T.dynamic("total_q")`（动态轴） |
| `k` | `(T_kv, H, 192)` | 同上 | `T_kv = T.dynamic("total_kv")` |
| `v` | `(T_kv, H, 128)` | 同上 | |
| `do` | `(T_q, H, 128)` | 同上 | |
| `lse` | `(H, T_q)` | `float32` | **自然对数** log-sum-exp，**head-major**；空行为 `-inf` |
| `delta` | `(H, T_q)` | `float32` | `rowsum(o*do)`，head-major，由 `_delta_kernel` 产出 |
| `cu_seqlens_q` | `(B+1,)` | `int32` | `B` 为编译期常量；值域 `0..T_q` 单调 |
| `cu_seqlens_kv` | `(B+1,)` | `int32` | 同上（值域 `0..T_kv`） |

（源算子签名另含 `o`：由同一次算子调用的 `_delta_kernel` launch 消费（`bwd.py:341`），**本 kernel 不读**。）

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `dk` | `(T_kv, H, 192)` | `float16 \| bfloat16` | 与 `k` 同 shape；已乘 `scale` |
| `dv` | `(T_kv, H, 128)` | 同上 | 与 `v` 同 shape；**不乘** `scale` |
| `dq_acc` | `(T_q, H, 192)` | `float32` | **in-out**，原子累加；未乘 `scale`（由 `_dq_land_kernel` 处理） |

三个输出与输入同 layout（THD、尾轴连续），**无需任何转置或 permute**；`dk/dv` 与 `k/v` 逐维相同，
`dq_acc` 与 `q` 同 shape（仅 dtype 为 fp32），故 host 侧只做 `zeros`（`dq_acc`）。

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
| pT_s | (64,64) | 输入 dtype | L1 | dV GEMM 的 A（`P^T`，两 AIV 各半写出） |
| dsT_s | (64,64) | 输入 dtype | L1 | dK GEMM 的 A |
| ds_s | (64,64) | 输入 dtype | L1 | dQ GEMM 的 A（`dS` 转置副本） |
| st_c | (64,64) | float32 | L0C | `s^T` 累加器（每迭代清零） |
| dpt_c | (64,64) | float32 | L0C | `dP^T` 累加器（每迭代清零） |
| dq_c | (64,192) | float32 | L0C | `dQ` 累加器（每迭代清零） |
| dk_c | (64,192) | float32 | L0C | `dK` 累加器（跨 query 循环） |
| dv_c | (64,128) | float32 | L0C | `dV` 累加器（跨 query 循环） |
| st_u / dp_u | (32,64) | float32 | UB（半区） | L0C 落地 + 向量输入 |
| p16_u / dsT_u | (32,64) | 输入 dtype | UB（半区） | 向量产出的 fp16 |
| ds_u | (64,32) | 输入 dtype | UB（半区） | `dS` 转置半区 |
| lse_u / dlt_u | (64,) | float32 | UB | 行统计量（head-major 装载） |
| tail_u | (32,192) | float32 | UB（半区） | 块尾 `dk_c` 中转 |
| tailv_u | (32,128) | float32 | UB（半区） | 块尾 `dv_c` 中转 |
| out_k / out_v | (32,192) / (32,128) | 输入 dtype | UB（半区） | 块尾 cast 结果（写 GM 前） |

### 4.4 内存搬运路径

```
GM[cu_seqlens_q/kv] ──(标量读, L2 常驻)──▶ 标量: (req, n0, live, q_start, kv_start, len_q, len_kv, shift)
GM ──(MTE2: nd2nz / dn2nz；越界尾部按**张量 extent** clamp+fill，由 AscendInsertOOBPadding 生成)──▶ L1
     k_s, v_s, kT_s(常驻) | q_s, qT_s, do_s, doT_s(每迭代)
GM ──(MTE2)──▶ UB lse_u, dlt_u(每迭代，head-major 1-D 256 B)
L1 ──(MTE1: 由 L1-GEMM lowering 内部按 tile_k_sub 分块)──▶ L0A/L0B
L0A/L0B ──(Cube mad)──▶ L0C st_c, dpt_c, dq_c(每迭代) / dk_c, dv_c(跨迭代)
L0C ──(fixpipe, dual_copy 2:1, fp32 同 dtype)──▶ UB 半区 st_u/dp_u(32,64)
UB ──(SimtVF(128) 半区: exp / mask / mul / cast / 转置写)──▶ UB 半区 p16_u, dsT_u, ds_u
UB ──(dual_copy 2:1 → InsertNd2Nz → ascend_nd2nz_scatter)──▶ L1 pT_s, dsT_s, ds_s（整块 64×64）
L0C(dq_c) ──(fixpipe + store-atomic add, fp32, 整块)──▶ GM dq_acc（溢出行为精确 0）
L0C(dk_c/dv_c) ──(dual_copy 2:1)──▶ UB 半区 tail_u/tailv_u ──(cast)──▶ out_k/out_v
     ├─ 内部半块：T.copy 整块 ──▶ GM dk/dv
     └─ 尾半块：  SimtVF 逐 lane（`if own+i < kv_end`）──▶ GM dk/dv
```

无 GM 中转的中间块（`P/dP/dS` 全程留在 L0C/UB/L1），符合 §1.6.0 R2 的「不物化」结论。

### 4.5 UB 内存预算（每 AIV，半区）

**UB** 容量 253952 B（248 KB）；块尾缓冲与每迭代缓冲不复用（保守上界），与 dense 真机通过的实现逐 buffer 相同：

| Buffer | Shape | dtype | 大小 (Bytes) |
|--------|-------|-------|-------------|
| st_u | (32,64) | float32 | 8192 B |
| dp_u | (32,64) | float32 | 8192 B |
| p16_u | (32,64) | f16/bf16 | 4096 B |
| dsT_u | (32,64) | f16/bf16 | 4096 B |
| ds_u | (64,32) | f16/bf16 | 4096 B |
| lse_u | (64,) | float32 | 256 B |
| dlt_u | (64,) | float32 | 256 B |
| tail_u | (32,192) | float32 | 24576 B |
| tailv_u | (32,128) | float32 | 16384 B |
| out_k | (32,192) | f16/bf16 | 12288 B |
| out_v | (32,128) | f16/bf16 | 8192 B |
| **合计（不别名上界）** | — | — | **90624 B** |

其中每迭代活跃部分 = `st_u + dp_u + p16_u + dsT_u + ds_u + lse_u + dlt_u` = 29184 B，块尾活跃部分 =
`tail_u + tailv_u + out_k + out_v` = 61440 B。总占比 90624 / 253952 = **35.7%**，余量 163328 B。

- **Stage 4 流水约束**：若把每迭代组取 2 版本，上界 = 90624 + 29184 = 119808 B = **47.2%**（可行），
  但 `dual_copy` 的双版本语义未认证，故首版固定单版本（与 dense 一致）。
- 与 dense 的差异：dense 设计 v1 的整块 UB 表（180736 B / 71.2%）是「整块 epilogue」形态的预算；真机验证后
  改为半区形态，本函数直接继承**已验证的半区预算**（dense v0 的 90624 B）。

### 4.6 L1 与 L0C 预算

| 层 | Buffer 组合 | 合计 (Bytes) | 容量 (Bytes) | 占比 |
|----|------------|-------------|-------------|------|
| L1 | `k_s 24576 + v_s 16384 + kT_s 24576 + q_s 24576 + qT_s 24576 + do_s 16384 + doT_s 16384 + pT_s 8192 + dsT_s 8192 + ds_s 8192` | 172032（168 KiB） | 524288（512 KiB） | 32.8% |
| L1（Stage 4 双版本，7 个每迭代缓冲各 ×2） | +106496 | 278528（272 KiB） | 524288 | 53.1% |
| L0C | `dk_c 49152 + dv_c 32768 + st_c 16384 + dpt_c 16384 + dq_c 49152` | 163840（160 KiB） | 262144（256 KiB） | 62.5% |
| L0A/L0B | 不显式分配；L1-GEMM lowering 按 `2 阶段 × 32 KB/操作数` 自行分块（单块 `64×192×2 = 24576 ≤ 32768`） | — | 65536 各 | 由 lowering 保证 |

**L0 回退路径的 L0A/L0B 预算（若 L1-GEMM 运行期失败）**：`Nk=64, Mq=32` 下 5 个积的唯一操作数形状并集
（含 Mq=64 与 Mq=32 两种 M 形态）：L0A `(64,192) 24576 + (64,128) 16384 + (64,64) 8192 + (32,64) 4096 =
53248 B`（81.3%）；L0B `(32,192) 12288 + (32,128) 8192 + (128,64) 16384 + (192,64) 24576 = 61440 B`（93.8%，
余量 4 KiB，**只允许单版本**）。该路径下 L0C 降到 122880 B（46.9%），代价是每迭代 L1 端口流量与 `dq` 原子
流量上升 1.5–2×。

### 4.7 动态轴

| 动态轴 | 声明方式 | 运行时范围 |
|--------|----------|-----------|
| `T_q` | `total_q = T.dynamic("total_q")` | 契约上任意正整数；workload 实测 32256 ~ 32768 |
| `T_kv` | `total_kv = T.dynamic("total_kv")` | 同上 |

`B`（组数）、`H`、`D_qk=192`、`D_v=128`、`block_n/block_m=64`、`_HN=32` 全部是**编译期常量**（进工厂特化
参数，`@lru_cache` 级缓存，与源 `target_cache(maxsize=32)` 等价）。`len_q/len_kv/shift/q_start/kv_start/n0/
live/loop_st/loop_ed/own/kv_end` 是**运行期标量**，只出现在标量运算、谓词与 copy 的偏移/长度中，**不出现
在 GEMM 的 M/N/K**（§3.5.1）。动态 grid 依赖 `total_kv`（与 `examples/paged_cache_write` 同机制）。

### 4.8 JIT 配置

```python
@functools.lru_cache(maxsize=64)
def _bwd_kernel(batch, heads, dim_qk, dim_v, dense_seqlen_q, dense_seqlen_kv,
                is_causal, scale, dtype, a2_fast=True):
    total_q = T.dynamic("total_q")
    total_kv = T.dynamic("total_kv")
    ...
    return tilelang.compile(_bwd_main, target="ascend")
# 编译标志沿用源：-O3、-DENABLE_BF16；不开 fast-math 之外的非默认选项
```

`out_idx`：`dk/dv` 与 `dq_acc` 都是调用方预分配的张量（`dq_acc` 必须由 host 先 `torch.zeros`——store-atomic
的累加契约）；`dense_seqlen_q == 0` 是本函数唯一接受的形态（非 0 属于 dense 算子，直接 `ValueError`）。

---

## 5. Tiling 策略

### 5.1 计算类型

**类型：混合（Cube + Vector）**。判定依据：5 个矩阵乘（每迭代 3.4M MAC）必须落 Cube；每迭代另有 4096 元素的
`exp/mask/cast/转置` 必须落 Vector；两者在同一 kernel 内交替（§2.2、§0.6 R2/R4）。

### 5.2 Block 划分

```python
block_n = 64        # key 方向（= dk/dv 累加器的 M，以及 dq 的 K）
block_m = 64        # query 方向（= dq 累加器的 M，以及 dv/dk 的 K）
_HN     = 32        # 每 AIV 的 key 行（dual_copy 2:1）
# 逻辑任务数 = heads * (Σ_b ceil(len_kv[b]/block_n) + 多算的 ≤B 个 tile)
```

选择理由：
- `block_n` 同时决定 L0C 里 `dk_c`（`192·block_n·4 B`）与 `dv_c`（`128·block_n·4 B`）的大小，二者跨整个
  query 循环常驻：`block_n=64` 时 80 KiB，`block_n=128` 时 160 KiB（再加每迭代的 `st/dpt/dq` 就要逼近溢出）。
- `block_m=64` 让 `dq_c = 64×192×4 = 48 KiB`；同时 `dv/dk` 的 K 维取 64，mad 的 K 分块不退化；半区向量段
  （32×64）在 `T.SimtVF(threads=128)` 下每线程 16 个元素，向量宽度整除无尾 lane。
- 二者均为 16 的倍数（mad 分形 16×16），`D_qk=192 = 12×16`、`D_v=128 = 8×16` 整除。
- 每迭代 L1 端口字节 ≈ 279280（24.5 FLOP/字节，§1.6.0 分解表）；备选 `block_n=128, block_m=32` 提到
  26.1（−5.9%），但其 L0C 占用升到 84.4%，留作 Stage 4 裁决项（§1.6.3）。

### 5.3 约束分析

- **对齐**：`D_qk=192`、`D_v=128`、`block_n=block_m=64`、`_HN=32` 全为 16 的倍数；fp16/bf16 的 L1/L0 分形与
  mad 的 16×16 分块无尾块；`dq_acc` 尾轴 192 个 fp32 = 768 B（32 B 对齐 ✓）。
- **L0C**：单累加器校验 `M × N × 4 ≤ 262144`：最大者 `dk_c = 64×192×4 = 49152 B` ✓；5 个同时驻留共
  **160 KiB / 256 KiB（62.5%）**。`block_n=128 + block_m=64` 会到 272 KiB → 溢出，故 `block_n=128` 只能配
  `block_m=32`（216 KiB，84.4%）。
- **L1**：168 KiB 单版本（32.8%）；双版本 272 KiB（53.1%）。
- **UB**：88.5 KiB（35.7%）/ 248 KiB；双版本 117 KiB（47.2%，容量可行但语义未认证）。
- **L0A/L0B**：L1 路径由 lowering 按 `tile_k_sub` 分块保证（每操作数每阶段 ≤ 32 KB）；L0 回退路径见 §4.6。
- **动态边界**：GEMM 的 M/N/K 静态（allocation extent）；grid 为动态上界 + `live`；copy 的偏移/长度为
  运行期表达式（平台 clamp 只认张量 extent，request 边界由掩码与 R7 双分支收口）。

### 5.4 注意事项（非整除、变长与静态边界）

> 统一原则：**GEMM 一律全 extent；动态 extent 只出现在 copy 与 grid；装载的越界尾部由平台 pass 按
> 张量 extent clamp+补 0，request 边界由掩码收口；写回的 request 边界由 R7 双分支收口。**

1. **变长块枚举的静态边界风险（packed 核心风险）**：`total_kv` 是运行期量，`tile_upper_bound` 给出的是
   **上界**（真实 tile 数 `tile_cum[B] ≤ total_kv//64 + B`）。
   - **主方案（已编译，P1/P2）**：grid 用该上界（动态表达式），块内用 `live = tile < cnt` 早退；
     多算的 tile 数 = `total_kv//64 + B − Σ_b ceil(len_kv_b/64) ≤ B`（每 head），且非 live 块**不发起任何
     DMA**（整个 body 在 `if live` 内）。量化：`ds-v3-mixed` 下 64,512 个 live 块 + 6×128 = 768 个空块
     （1.2%），空块成本 = 一次 O(B) 扫描（≤ 3B 个整数 op）。
   - **静态化备用方案（已编译，P5）**：若运行期发现动态 grid 有问题，改为编译期上界
     `grid = (TK_MAX//64 + B) * H`（host 契约 `total_kv ≤ TK_MAX`），其余不变（`live` 谓词原样保留）。
     代价：`TK_MAX` 取 32768 时，`ds-v3-mixed`（T_kv=32256）每 head 多算 `512 + 6 − 504 = 14` 个空块
     → 共 1792 个（占 66,304 个发射的 2.7%），每个空块仍只花一次 O(B) 扫描。
   - **必须显式记录的运行期前提**：两者都要求 `cu_seqlens_kv[B] == total_kv`（源算子亦然），且
     `cu_seqlens_*` 单调不减（manifest 的 `requires: prefix_offsets(T_kv)`）。
2. **尾 key 块（`len_kv % 64 ≠ 0`）**：`k_s/v_s/kT_s` 用全 extent 装载，越界行由 `AscendInsertOOBPadding`
   收窄 DMA 并补 0（设备码 `min(64, max((total_kv - kv_start) - n0, 0))` + `asc_fill_l1`）。补 0 使这些行的
   `s^T` 为 0，但 `exp(0·scale − lse) ≠ 0`，故**必须**由掩码臂的 `gi < len_kv` 把 `P` 置 0（→`dS=0`）。
   `dk/dv` 写回时**不能**依赖同一 clamp（它只到 `total_kv`）→ R7 双分支。
3. **尾 query 块（`len_q % 64 ≠ 0`）**：`q_s/do_s/qT_s/doT_s/lse_u/dlt_u` 同样全 extent 装载 + clamp/补 0；
   掩码臂保留 `gj < len_q`，使填充列的 `P=0`；`dq_c` 的整块原子 store 溢出到下一 request 的行是**精确 0**
   （A3），张量末尾由平台 clamp 裁到 `total_q`（设备码 `min(64, max((total_q-q_start)-kq*64, 0))`）。
4. **因果对角块**：只有对角及相邻块进入掩码臂，其余走 A2 快路；`loop_st = max(0,(n0-shift)//64)` 保证不遗漏
   可见 query 块（最坏多算一个全掩码块，`P=0` 无副作用）。
5. **`len_q < len_kv` 与 `len_kv < len_q`**：`shift` 可为正/负；后者会让头部 query 看不到任何 key（`lse=-inf`），
   此时掩码全假 → `P=0` → 梯度 0，`exp(-inf)` 被 `if_then_else` 丢弃（不产生 NaN，与 golden 的
   `nan_to_num` 语义对齐）。**快路安全性**由 §1.6.1 的充分条件保证（满足快路的块不可能包含空行）。
6. **退化请求（`len_kv ≥ 1` 但 `len_q = 0`）**：tile 由 `len_kv` 枚举，因此这样的请求仍有 tile；
   若循环为空，`dk_c/dv_c` 从未被 `clear_accum` 初始化（**`T.clear` 不支持 L0C**，P4 实测）。处置：
   `loop_ed = max(ceil(len_q/64), loop_st + 1)` 保证至少一次迭代；该次迭代的全部 `P=0`（`gj < len_q` 恒假），
   于是 `dk_c/dv_c = 0`（正确：无 query 即无梯度），`dq` 整块写回的是精确 0（A3）。代价：只有这类退化请求
   多一次迭代。**边界说明**：manifest 的 workload 与 golden 都不产生 `len_q = 0` 的请求，此处置是防御性的。
7. **`is_causal=False`**：`loop_st=0`，所有块都进入快路（仍受 clamped 尾块影响，见 2/3）。
8. **极小请求（`len_kv < 64`）**：全部走尾块路径；`ceil(len_kv/64)=1`，DMA 被 clamp 到 `len_kv` 行并补 0，
   R7 慢路生效（`own + 32 <= kv_end` 可能为假）。

### 5.5 分核策略（物理核数适配）

> 三要素标准：`.agents/skills/_shared/standards/core-split-strategy.md` §1。

- **① 逻辑核数**：`num_logical_kernels = heads × (Σ_b ceil(len_kv[b] / block_n) + (grid 上界多算的 ≤B))`
  ——每个逻辑块 = 一个 (request, key 分块, head)，即一次 block 发射：

  | workload | B | lens（q=kv） | H | Σ_b ceil(len/64) | 上界多算 | **逻辑核数** |
  |----------|---|---|---|---|---|---|
  | ds-v3-8x4k | 8 | 4096 ×8 | 128 | 64×8 = 512 | 8 | **66,560** |
  | ds-v3-mixed | 6 | 512,1024,2048,4096,8192,16384 | 128 | 504 | 6 | **65,280** |
  | kimi-k2-4x8k | 4 | 8192 ×4 | 64 | 128×4 = 512 | 4 | **33,024** |
  | smoke（Stage 3 门槛） | 2 | 512, 96 | 4 | 8+2 = 10 | 1 | **44** |

- **② 物理核数依据（任务平台记录，非假设值）**：目标型号 `ascend-950pr-9579`（dav-3510）。
  平台 `statectl platform` 落库的实查记录（`core_source` 字段，2026-11-23）：
  `Ascend950PR_9579.ini [SoCInfo] ai_core_cnt=28, cube_core_cnt=28, vector_core_cnt=56;`
  `runtime probe rtGetAiCoreCount()=28`。本次设计期该卡被其他算子占用（`ASCEND_RT_VISIBLE_DEVICES=""`，
  只做无卡编译探针），因此引用上述已落库探测记录，并在 Stage 3 首跑前用 `torch.npu.get_device_properties`
  复核型号与核数后再确认。
- **③ 规模判定与分核方案**：三条 workload 的逻辑核数（33,024 … 66,560）**全部 ≫ 物理核数 28**
  （最小 smoke 也是 44 ≈ 1.6 波），且无法通过调整分块把核数压到 28 的整数倍（`block_n` 上限受 L0C 约束，
  最大 128 → ds-v3-mixed 仍有 ≈32,640 个逻辑块）。判定为 **极大规模**。
  **分核方案：1 逻辑块 = 1 次 block 发射（逻辑块超发，由硬件块调度器按波次派发）**，不采用「固定启动
  内核数 = 物理核数 + 核内串行多任务」，依据：
  (a) 每逻辑块的计算量是 10¹–10³ µs 级（ds-v3-mixed：38.5 TFLOP / 64,512 块 ≈ 597 MFLOP/块），
  远大于 block 派发开销；
  (b) **packed 的负载不均（512→16384，块内迭代数 8→256）使动态派发成为必需**：静态 28 等分会把 256 次
  迭代的长块钉在某个核上，最长块占总工作量 0.0046%（§1.6.0），而块时长差 32× → 静态切分的长尾可达
  单核总时长的 1/28 × 32 量级；动态派发把长尾压到「一块」= 0.0046%；
  (c) 「核内串行」需要把 Cube+Vector 混合体包进任务循环，而平台明确未认证「循环内跨单元/跨核同步」
  （`programming.md`「同步与缓冲版本」末段），风险高于收益。
  **规则偏离声明**：本项**偏离** `core-split-strategy.md` §1 的「极大规模」标准分支（该分支要求「固定启动
  内核数 = 物理核数，核内分批处理多个逻辑块」），采用标准未列出的第四方案「逻辑块超发」。理由即 (a)(b)(c)。
  **复评触发条件**（满足任一条则按 `[DESIGN_LIMIT]` 流程回写并重估）：① Stage 4 实测 block 派发/尾波占比
  > 5% 的单 kernel 时长；② `num_stages` 提到 2 后出现派发瓶颈或核间负载方差 > 2×；③ 平台认证「循环内
  Cube+Vector 交替 + 跨迭代同步」后，核内串行方案转为可选。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

> 逐元素计算已按 §1.6.2 全部向量化；下表只有 block 级 / tile 级调度循环与一处 block 级标量扫描，
> 无逐元素标量循环。

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| 逻辑任务（head × tile） | block 级并行 | `T.Kernel(grid)` 一维核网格（grid 含 `total_kv`） | 每 block 处理一个 key 分块（key-stationary），§5.5 |
| **变长块枚举（O(B)）** | block 级标量串行 | `T.serial(batch)` + `T.alloc_local` | 前缀和 + decode 融合，batch 编译期常量；§1.6.2 #9 |
| query 块方向 | 迭代（动态界） | `T.Pipelined(loop_st, loop_ed, num_stages=1)` | 因果剪枝后界随 `n0/len_q/shift` 变化；首版单阶段 |
| 5 个矩阵乘 | tile 级 | `T.gemm(A_l1, B_l1, C_l0c, transpose_B=True, clear_accum=…)` | Cube；`clear_accum` 区分「每迭代清零」与「跨迭代累加」 |
| 逐元素 epilogue | 向量并行（半区） | `with T.SimtVF(threads=128): for i, j in T.Parallel(32, 64)` | lane = query 轴（§1.6.3）；A2 由运行时块级 `if` 生成两支 |
| L0C↔UB / UB→L1 | tile 级搬运 | `T.dual_copy(...)`（2:1） | 半区，`asc_get_sub_block_id` 一致 |
| 块尾 cast | 向量并行 | `T.Parallel(32, 192)` / `T.Parallel(32, 128)` | `dk` 乘 `scale` 后 cast |
| 块尾写回 | 运行时分支 | `if own + 32 <= kv_end:`（整块 `T.copy`）`else`（`T.Parallel` + lane 谓词） | request 边界收口（R7） |

### 6.2 循环伪代码

见 §3.3（同一份伪代码，此处只列骨架）：

```python
with T.Kernel((total_kv // 64 + batch) * heads) as bid:   # 动态 grid，最多 ≈66k 块（按波次派发）
    tile, h = bid // heads, bid % heads                    # 静态除数
    <O(batch) 扫描：req / n0 / live>                        # T.serial + local 标量
    if live:
        <请求内几何：q_start / kv_start / len_q / len_kv / shift>
        <全 extent 装载 k_s/v_s/kT_s>
        for kq in T.Pipelined(loop_st, max(ceildiv(len_q,64), loop_st+1), num_stages=1):
            <全 extent 装载 q_s/qT_s/do_s/doT_s 与 head-major lse_u/dlt_u>
            <2 个 NT GEMM：st_c, dpt_c>
            <dual_copy 半区：st_c→st_u, dpt_c→dp_u>
            if <运行期快路谓词>:   <SimtVF 半区：p=exp(...)，无 mask，写 p16_u/dsT_u/ds_u>
            else:                  <SimtVF 半区：掩码臂（gi<len_kv, gj<len_q, 因果），同上>
            <dual_copy 半区：p16_u→pT_s, dsT_u→dsT_s, ds_u→ds_s>
            <3 个 NT GEMM：dv_c += , dk_c += , dq_c = >
            <set_atomic + 整块 copy(dq_c → dq_acc) + set_atomic_none>
        <块尾：dual_copy dk_c/dv_c → 半区 UB, cast(+scale)>
        if own + 32 <= kv_end: <整块 T.copy → GM dk/dv>
        else:                  <SimtVF 逐 lane 守卫 → GM dk/dv>
```

### 6.3 流水线优化

- 首版（Stage 3）**`num_stages=1`**：与 dense 真机通过的配置一致。理由：平台仅核验过「单 tile 混合」与
  「静态界两阶段 GEMM」，「多阶段混合循环 + 跨单元重叠」未认证，且 `dual_copy` 的双版本语义未取证。
- Stage 4 的候选路径：L1 侧 7 个每迭代缓冲取 2 版本（278528 B / 53.1%）；UB 侧半区取 2 版本（119808 B /
  47.2%，容量可行但需先认证 `dual_copy` 的多版本语义）。
- 若动态界 + 多阶段不被接受，则改成「固定 trip count = `ceil(TK_MAX? 不: ceil(len_q/64)` 上界不可静态）+ 头部
  `if kq >= loop_st`」——但 `len_q` 是运行期量，`ceil(len_q/64)` 无法静态化，故 packed 下该退化方案不可用，
  只能保留动态界单阶段（记录在 §9.1 风险 4）。
- 不做 `T.annotate_unlimit_memory`（不解除容量约束，避免把溢出推到运行期）。

### 6.4 尾块处理

统一口径（与 §3.3 注、§5.4 一致）：

- **GEMM**：不参与尾块处理——A/B/C 一律 allocation 全 extent，M/N/K 编译期静态。代价是每个 request 的
  尾块多算一个满 tile（≤ `1/ceil(len_kv/64)` 的额外 FLOPs，ds-v3-mixed 全量 +0.9%）。
- **装载**：GM→L1 / GM→UB 的越界尾部由平台 pass `AscendInsertOOBPadding` 在编译期改写成运行时 `min()`
  clamp，并对 GM→L1 的补 0 区段生成 `asc_fill_l1`（**按张量 extent**，不认 request 边界）。
- **掩码**：`gi < len_kv` / `gj < len_q` 两项**必须**保留在掩码臂中（补 0 不等于梯度 0）。
  快路分支通过运行时谓词保证两支等价（A2）。
- **写回**：`dq` 由「溢出零行」+ 张量尾 clamp 收口；`dk/dv` 由 R7 双分支收口（**这是 packed 与 dense 的
  关键差异：dense 可以只靠平台 clamp，packed 不行**）。

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
| `cu_seqlens` 标量读 → 所有 DMA/循环界 | 同核标量依赖（无跨核）；`T.serial` 的读在 body 前完成 | decode 结果只在本核使用 |
| GM→L1 / GM→UB 之后到消费 | 自动（MTE2→MTE1/M 事件） | 搬运与 GEMM/向量读的依赖 |
| `T.gemm` 之后到 `dual_copy(L0C→UB)` | 自动（M→FIX 事件） | L0C 写后读 |
| `dual_copy(L0C→UB)` 之后到 `SimtVF` | 自动（FIX→V 事件，两次 arrive 覆盖两个副块） | UB 写后读，且需对两个 AIV 都可见（dense 真机验证） |
| `SimtVF` 之后到 `dual_copy(UB→L1)` | 自动（V→MTE） | L1 写前读完成 |
| `UB→L1` 之后到下一个 `T.gemm`（A 操作数） | 自动（MTE→MTE1/M） | GEMM 读 L1 前必须写完 |
| `set_atomic` ↔ `T.copy(L0C→GM)` ↔ `set_atomic_none` | 硬件模式寄存器；调度器保序 | 原子模式必须在 store 前生效、之后关闭 |
| 块尾 `dual_copy` 与两条写回支路之间 | 自动（V→MTE3） | 写 GM 前 UB 必须就绪（P2 设备码 `asc_sync_wait(PIPE_V, PIPE_MTE3, …)`） |
| 跨 block（不同 key 块累加同一 `dq_acc` 行） | **无同步**（store-atomic 由硬件保证原子性） | 本设计不引入任何跨 block barrier |

### 7.3 pass_configs 配置

```python
# 保持默认：不关闭自动调度、不改共享内存复用策略
pass_configs = {}          # 即 TL_ENABLE_AUTO_SCHEDULE=True（默认）
compile_flags = ["-O3", "-DENABLE_BF16"]
```

若 Stage 3 出现同步相关失败，回退顺序是「先降 `num_stages` 到 1（已是 1）→ 再查 `dual_copy` 的副块配对 →
最后才考虑 `T.MixedKernel(N, sids=2)` + 显式 `T.Cube()/T.Vector()` + 手工 sync」——**注意该形态与 `dual_copy`/
UB→L1 混用时不可编译**（dense §3.5.4 反例 2），故它已不是本设计的可行回退。

---

## 8. 验证方案

### 8.1 Golden 函数

> 依据 §0.1 语义独立实现（移植源仓 `mla_bwd_reference` 与 `MLAVarlenBwdWorkload.ref_program`），
> **不复刻** §0.6 的分块算法：它按 request 物化概率 `P` 再用 `einsum` 反向，不重算任何块、不做原子累加，
> 也不消费外部传入的 `lse`/`delta`（两者都由它自己从 `q,k,v,do` 重算）。**packed 版按 `cu_seqlens` 逐请求
> 切片**，与 kernel 的块枚举、半区切分、`live`、双分支写回无任何共享假设。

```python
import torch

def golden_bwd_kernel(q, k, v, dout, cu_seqlens_q, cu_seqlens_kv, *, scale=None,
                      is_causal=True, dtype=None, slab_heads=4):
    """_bwd_kernel（packed）的 golden：fp32/fp64 物化参考，bottom-right 因果。

    q,do: (T_q, H, D_qk)/(T_q, H, D_v)   k: (T_kv, H, D_qk)   v: (T_kv, H, D_v)
    返回 (dq, dk, dv)（各舍入一次到 dtype）与 dq_acc_true = dq_true / scale
    （本 kernel 写进 dq_acc 的 fp32 中间量）。
    """
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    dtype = dtype or q.dtype
    H, D_qk = q.shape[1], q.shape[2]
    D_v = v.shape[2]
    qf, kf, vf, dof = (t.detach().to(torch.float64) for t in (q, k, v, dout))
    dq = torch.zeros_like(qf)
    dk = torch.zeros_like(kf)
    dv = torch.zeros_like(vf)
    bq, bkv = cu_seqlens_q.tolist(), cu_seqlens_kv.tolist()
    for b in range(len(bq) - 1):
        q0, q1, k0, k1 = bq[b], bq[b + 1], bkv[b], bkv[b + 1]
        len_q, len_kv = q1 - q0, k1 - k0
        shift = len_kv - len_q
        key_pos = torch.arange(len_kv, device=q.device)
        hid = key_pos[None, :] > (torch.arange(len_q, device=q.device) + shift)[:, None] \
            if is_causal else None
        for h0 in range(0, H, slab_heads):      # 按 head 分片，避免 O(H*len_q*len_kv) 同时驻留
            hs = slice(h0, min(h0 + slab_heads, H))
            s = torch.einsum("shd,nhd->hsn", qf[q0:q1, hs], kf[k0:k1, hs]) * scale
            if hid is not None:
                s = s.masked_fill(hid[None], float("-inf"))
            lse = s.logsumexp(-1, keepdim=True)
            p = torch.exp(s - lse).nan_to_num(0.0)          # 全 -inf 行 → 概率 0
            dp = torch.einsum("shd,nhd->hsn", dof[q0:q1, hs], vf[k0:k1, hs])
            ds = p * (dp - (p * dp).sum(-1, keepdim=True))   # delta 由概率重算，独立于输入 delta
            dv[k0:k1, hs] += torch.einsum("hsn,shd->nhd", p, dof[q0:q1, hs])
            dk[k0:k1, hs] += torch.einsum("hsn,shd->nhd", ds, qf[q0:q1, hs])
            dq[q0:q1, hs] = torch.einsum("hsn,nhd->shd", ds, kf[k0:k1, hs])
    dq, dk = dq * scale, dk * scale
    dq_acc_true = (dq / scale).to(torch.float32)
    return dq.to(dtype), dk.to(dtype), dv.to(dtype), dq_acc_true
```

独立性说明：golden 只读 `q,k,v,do,cu_seqlens_*`（不使用 `lse/delta`），概率与 `delta` 全部重算；
它按 request 物化 `P` 并用 `einsum`，与本设计的分块/半区/原子/双分支结构无共享假设。`lse/delta` 的正确性
由 `_delta_kernel` 的 L0 负责（同一算子调用的另一个 launch）。

### 8.2 精度标准

| 输出 | dtype | atol | rtol | 依据 |
|------|-------|------|------|------|
| `dk`, `dv` | float16 | 5e-3 | 1e-3 | `reference_tolerance(float16)` + `MLAVarlenBwdWorkload.verification` 的 `max(5e-3, atol)` |
| `dk`, `dv` | bfloat16 | 5e-3 | 1.6e-2 | 同上（bf16 一行） |
| `dq_acc`（fp32 中间量，测试侧再乘 `scale` 比对 `dq`） | float32 | 5e-3 / `scale` ≈ 6.9e-2 | 1e-3 | `dq_acc = dq/scale`，故 atol 按 `scale = 192^-0.5` 反比放大 |
| `P/dS` 的公式等价性 | — | — | — | §1.6.1 D-1 已机器验证（A1 1 ulp；A2/A3 违反率 0） |

补充口径：`dq_acc` 的累加顺序跨核不确定（store-atomic），因此**不做逐位比对**，只做 dtype 容差；
`zeroed_input(0)` 控制（`verification` 的 controls）用「把 `do` 置零 → 三个梯度全零」覆盖。

### 8.3 L0 门槛测试计划

| 用例 | 长度配置（q_lens / kv_lens） | H | is_causal | 目的 |
|------|------------------------------|---|-----------|------|
| L0-1（门槛） | `[512, 96]` / `[512, 96]` | 4 | True | smoke：5 GEMM 链 + 原子 + 半区 + 变长 decode + **一个尾块请求** |
| L0-2 | 同 L0-1，fp16 | 4 | True | dtype 覆盖 |
| L0-3 | `[512, 96]` | 4 | False | 非因果（`loop_st=0`、A2 快路全开） |
| L0-4 | `[96]` / `[192]` | 2 | True | `shift>0`；`loop_st>0` |
| L0-5 | `[128]` / `[64]` | 2 | True | `shift<0`：头部 64 个 query 看不到任何 key（`lse=-inf`）→ `dq=0` 且无 NaN/inf |
| L0-6 | `[65, 1, 200]` / `[65, 1, 200]` | 2 | True | 极小/非对齐：单 token 请求、`len=1`、`len=65`（尾块 1 行） |
| L0-7 | `[64]` / `[64]` | 1 | True | 单块单迭代（`clear_accum=(kq==loop_st)` 首块语义） |
| L0-8 | `[1]` / `[1]` | 1 | True | 最小退化（全 1 元素，`_HN` 半区一半为空 → R7 慢路） |
| L0-9 | `[512]` / `[512]` 与 golden 的逐元素对照 + **request 间污染检测** | 2 | True | **R7 关键回归**：request 0 的尾块不得写 request 1 的 `dk/dv`（B=2 且第 2 个请求的行单独比对；另在 `dk/dv` 末尾追加哨兵行检测越过 `T_kv` 的写） |
| L0-10 | `[300]` / `[700]` | 2 | True | 不等长 + 尾块（`300 % 64 = 44`、`700 % 64 = 60`） |
| L0-11 | store-atomic 契约：`dq_acc` 预填 2.0 与清零各一次（同几何） | 2 | True | 原子加而非覆盖（差值恒为 2.0） |
| L0-12 | A2 两臂一致性：强制掩码臂（`a2_fast=False`）与特化版逐元素对照 | 2 | True | `dk/dv` 逐位相同、`dq` 差 ≤ 1e-5（原子序） |

L1/L2 与完整分层套件（`ds-v3-8x4k` / `ds-v3-mixed` / 多 dtype / 更多 B 与长度组合）交由
`tilelang-op-develop` 的 L1/L2/boundary；此处只锁 L0 门槛。

**Stage 3 必须前置的最小探针清单**（按序；编译级已由 §3.5.4 覆盖，下列是**运行期**必须确认的量）：

| # | 探针 | 输入 | 判定 |
|---|------|------|------|
| R1 | **动态 grid + live 的运行期正确性**：非整除长度 + B≥2 | `[96, 128]` / `[192, 64]` | 三个梯度在容差内；无越界写（哨兵）；非 live 块不产生写 |
| R2 | **动态 grid 的 host 侧 grid 计算**：`total_kv` 变化（两个不同 shape 各编译一次） | `[512]`/`[512]` 与 `[96]`/`[96]` | 两次 launch 的 grid 不同且都正确（验证 `T.dynamic` 进入 grid 的实参传递） |
| R3 | **R7 慢路（尾半块守卫写）** | `[1]`/`[1]`、`[300]`/`[700]` | `dk/dv` 尾行正确、相邻 request 不受污染 |
| R4 | **溢出零行（A3）的设备确认** | `[96]`/`[96]` 且 B=2 | request 1 的 `dq_acc` 与「只有 request 1 时」的结果逐位相同 |
| R5 | **head-major 统计量** | `[512]`/`[512]` | 与 `[B,H,S]` 版对照（同一数据重排 lse/delta）结果一致 |
| R6 | **半区 `dual_copy` + `sid` 一致性**（dense 已验证，packed 复核） | L0-7 | 逐元素 == golden；否则切 §9.1 风险 3 的回退 |
| R7 | **退化请求防御**（`len_q=0`） | `[0, 64]` / `[64, 64]` | 不崩、request 1 正确（request 0 无输出） |

实施顺序：R1/R2 → R6 → R3/R4 → R5 → R7 → 全量 L0-1…L0-12。

---

## 9. 风险点与注意事项

### 9.1 已知约束与风险（按优先级）

| # | 风险 | 证据/来源 | 影响 | 处置 / 回退 |
|---|------|----------|------|------------|
| 1 | **动态 grid + 混合 kernel 的运行期行为未实测**（编译级已通；平台先例是纯 SIMT 的动态 grid） | §3.5.4 P1/P2/P5；`examples/paged_cache_write/_paged_cache_write_kernel/DESIGN.md`（动态 grid 的设备验证是纯 Vector 形态） | 整个 kernel 的并行度；若 host 侧 grid 计算或 `__mix__` 与动态 grid 不兼容 → 启动失败/只跑部分块 | Stage 3 R1/R2 首验；回退 = P5 的**静态 grid 上界**（已编译），只需 host 契约 `total_kv ≤ TK_MAX`，其余代码不变 |
| 2 | **R7 尾半块逐 lane 写 GM 的运行期正确性/性能**（编译级已通，设备未测） | §3.5.4 P2 设备码 `main_kernel_simt_vf_3` | `dk/dv` 尾行错误或相邻 request 污染 | Stage 3 R3 + L0-9；回退 = P3 的**符号长度区段 copy**（已编译：`T.copy(out_k[:nr,:], dk[own:own+nr,h,:])`） |
| 3 | **半区 `dual_copy` 在 packed 下的可见性**（dense 真机已验证，本函数复核） | dense Stage 3 P1（整块 L0C→UB 只落一个 AIV，99.2% 错位）→ 必须 2:1 | 向量段读到旧数据（整块错） | Stage 3 R6；回退阶梯：`dual_copy` 2:1 + 静态半区（已采用）→ 若仍错则 `dual_copy` + 两个显式半区 buffer 的切片写 |
| 4 | **动态循环界 + `num_stages=1` 无法提流水**：packed 的 `loop_ed = ceil(len_q/64)` 是运行期量，无法像 dense 那样退化成「静态 trip count + 头部谓词」 | §6.3；`loop_start/stop` 均为运行期表达式（P2 设备码 `for (kq = …; …)`） | Stage 4 的流水收益受限 | Stage 4 只在 L1 侧考虑双版本；若必须静态 trip count，需把 `len_q` 上界变成 host 契约（不推荐） |
| 5 | **`dq` 整块原子写入下一 request 的 `dq_acc` 行**（依赖 A3 的溢出零行） | §1.6.1 A3（30528 + 3816 个零元素，0 违反）+ P2 设备码（store 长度只按 `total_q` 裁剪） | 若 `dS` 的越界列不是**精确** 0（例如未来改成乘掩码），邻居 `dq_acc` 会被加脏值 | 掩码必须保持 `T.if_then_else` 的**选择**语义（不得改成 `p * mask`）；Stage 3 R4 复验；备选 = P3 的符号长度原子 copy |
| 6 | **`dq` 原子流量 ≈ 276 GB**（L2 驻留假设）；若 L2 未命中则退化为 HBM 带宽瓶颈 | §1.6.0 流量分解；L2 = 128 MiB（硬件档案） | 时延高于估算 | Stage 4 提 `block_n`（§1.6.3 备选，原子流量 −7.7% 的 L1 端口与迭代数） |
| 7 | **`clear_accum=(kq == loop_st)` 是动态布尔** | 编译级已验证（dense 探针 B + 本设计 P2 设备码的实参为运行期表达式）；dense 真机 P4 已验证语义 | 首块未清零 → 结果叠加脏值 | dense 已真机验证同一形态；本函数额外用 `loop_ed ≥ loop_st+1` 覆盖空循环；Stage 3 L0-7 复核 |
| 8 | **`len_q = 0` 的退化请求**：`T.clear` 不支持 L0C（P4） | §3.5.4 P4；§5.4 第 6 条 | 若不加处置，`dk_c/dv_c` 未初始化 → 脏写 | 已用 `loop_ed = max(ceil(len_q/64), loop_st+1)` 保证至少一次全掩码迭代（产出精确 0）；Stage 3 R7 验证 |
| 9 | **`T.exp` 为 SFU 近似 + 可能的 fast-math** | `PL-950-softmax-boundaries` 在 fp32 上以 1e-6 级误差通过 | bf16/fp16 输出超差 | 已在 §8.2 容差内预期；若超差，关闭 fast-math 后复测 |
| 10 | **`dq_acc` 依赖 host 预清零** | `mode.py:set_atomic` 语义 + `PL-950-atomic-splitk.py` 的调用契约 | 结果翻倍 | wrapper 必须 `torch.zeros`（现有三 launch 序列已如此）；L0-11 检查 |
| 11 | **`lse` 域误用**（把 log2 域当自然对数传入）或 **layout 误用**（当 `[B,H,S]` 传） | 源算子显式乘 `LOG2E`、packed 用 head-major | 全量精度错/读错内存 | 接口层不做任何 log/ln 换算，也不转置；L0-5/R5 会立即暴露 |
| 12 | **A2 的两支 VF 代码体积与编译时间** | P2 设备码 4 个 SIMT VF，源 27084 B | 大 shape 下编译时间增加 | 可接受；若编译时间成为问题，退化为单臂（带掩码，−48.6% 的快路收益） |
| 13 | **`cu_seqlens` 契约**：`cu_seqlens_kv[B] == T_kv`、单调不减 | manifest `requires: prefix_offsets(T_kv)` | grid 上界/`live` 失效 → 漏算或多算 | 由算子接口保证；L0-1 覆盖，Stage 3 加一条显式断言用例 |

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|---------|
| UB/L0C 溢出 | 分块放大或双版本 | 编译失败 | 按 §4.5/§4.6 预算回退分块 |
| `T.Kernel` 传 `threads=` | 照搬 GPU 源码 | Python 级拒绝 | 线程域只写在 `T.SimtVF` 内 |
| `T.gemm` 用符号 region extent | 想用区段截断处理尾块 | `ValueError: T.gemm requires static tile dimensions` | GEMM 全 extent + 掩码臂 + 平台 clamp |
| `T.clear` 作用于 L0C | 想给累加器显式清零 | `InternalError: Unsupported scope shared.l0c`（P4） | 用 `clear_accum=(kq == loop_st)` + 非空循环保证 |
| 显式 `T.Vector(vector=2) as sid` | 想手工按副块分片 | `InternalError … dup sid` | 用 `T.call_extern("asc_get_sub_block_id")` 并在 VF 中捕获 |
| 在 `@T.prim_func` 体内用普通函数承载循环 | 想抽取计算体 | `TypeError: 'ForFrame' object is not iterable` | 计算体标 `@T.macro` |
| 把 `dk/dv` 写回当成 dense 那样只靠平台 clamp | 忽略 request 边界 | **静默污染下一个 request 的 `dk/dv`** | R7 双分支（内部整块 / 尾块逐 lane） |
| 把 `dq` 的掩码改成乘法（`p * mask`） | 想省一个 select | `lse=-inf` 行出现 `0·inf = NaN`；且溢出列不再是精确 0 → 污染邻居 `dq_acc` | 必须用 `T.if_then_else` 选择 |
| 用 `h = bid // ntiles_dyn`（dense 写法） | 照搬 dense 的展平 | 运行期整数除法未取证 | 改 `tile = bid // H, h = bid % H` |
| `dq_acc` 未清零 | 复用 buffer | 梯度翻倍 | host `torch.zeros` |
| 认为「补 0 就等于梯度为 0」 | 尾块只靠 OOB fill | `exp(-lse) ≠ 0` → 尾部行出现伪梯度 | 掩码臂必须保留 `gi < len_kv` / `gj < len_q` |
| 认为「`live` 只影响性能」 | 忽略非 live 块 | 若 body 未包在 `if live` 内，会读越界 `cu_seqlens[req]` 与错地址 | body 整体置于 `if live` 内（P2 设备码已确认） |

### 9.3 特殊场景处理

非整除（§5.4）、极小请求（§5.4 第 8 条）、`is_causal=False`（`loop_st=0`）、`D_qk=192` 非 2 的幂
（仅影响 L1 缓冲形状，不影响分形，`192 = 12×16`）、`len_q≠len_kv`（`shift` 正负两种情况已列 L0-4/L0-5）、
`B=1`（decode 扫描退化为单次判断）、`len_kv < 64`（R7 慢路必走）、`len_q = 0`（§5.4 第 6 条）。

---

## 10. 交付清单

### 10.1 目录结构

```
examples/mla_varlen_bwd/_bwd_kernel/
├── DESIGN.md                       # 本设计文档
├── verify_equiv.py                 # §1.6.1 等价性机器验证（D-1，可重跑）
├── RETROSPECTIVE.md                # Stage 1 复盘
├── history_version/
│   └── design_probe_compile.py     # §3.5.4 编译探针（5 个探针 + 1 对照，可重跑）
└── _bwd_kernel.py                  # 算子实现（Stage 3 产出，不在本阶段）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | 已完成（首次设计 v0） | 设计文档（§0 迁移分析 + §0.8 复用/差异分区 + §1.6 算法调研与优化） |
| `verify_equiv.py` | 已完成（2026-11-23 实跑，4/4 EQUIV_PASS） | 等价性与不变式验证脚本（A1/A2/A3） |
| `history_version/design_probe_compile.py` | 已完成（4/6 COMPILE_OK；2 个失败项是**反例**，用来钉住 L0C `T.clear` 不可用） | 编译探针 P1–P5 + P2b |
| `_bwd_kernel.py` | 待实现（Stage 3） | kernel + golden（§8.1）+ L0 套件（§8.3） |

### 10.3 命名规范

- 项目目录：`examples/mla_varlen_bwd/`（TileOPs 迁移项目，按算子名聚合）
- 算子目录：`_bwd_kernel`（对应源工厂函数名；同项目下还有 `_delta_kernel`、`_dq_land_kernel`）
- 实现文件：`_bwd_kernel.py`

### 10.4 实现顺序

1. ✅ 设计文档（DESIGN.md）+ 等价性验证（verify_equiv.py）+ 编译探针（design_probe_compile.py）
2. ⬜ Stage 3 R1/R2：动态 grid + `live` 的运行期正确性、grid 实参传递
3. ⬜ R6：半区 `dual_copy` + `sid` 的 packed 复核（L0-7）
4. ⬜ golden（§8.1）+ L0-7 单块全链（先 `dk/dv`）
5. ⬜ R3/R4：R7 双分支与溢出零行（L0-9/L0-6/L0-10）
6. ⬜ R5/R7：head-major 统计量、退化请求
7. ⬜ L0-1…L0-12 全量 + L1/L2 与性能采集（交 Stage 3 / Stage 4）

---

## 11. 与 dense 版本的对照（复用证据索引与差异落点）

### 11.1 复用项 → 证据索引

| 复用项 | 索引 | 本函数的落点 |
|--------|------|-------------|
| key-stationary 结构、5 个 NT 乘积 | `examples/mla_dense_bwd/_bwd_kernel/DESIGN.md` §1.4、§3.1 | §1.4、§3.1（逐项同形） |
| 半区 `dual_copy` + `sid` | `examples/mla_dense_bwd/_bwd_kernel/_bwd_kernel.py`（`_HN=32`、`T.call_extern`）；dense Stage 3 记录 | §0.6 R2/R4、§3.3 |
| 全 extent GEMM + 掩码 + 平台 clamp | dense §3.5.4 探针 A/B | §3.3 注 4、§5.4、§6.4 |
| store-atomic + 三 launch 契约 | dense §0.6 R5；`PL-950-atomic-splitk.py` | §0.6 R5、§3.2 步骤 8 |
| A1/A2 等价结论 | dense §1.6.1（3/3 PASS） | §1.6.1（A1 同式同验证；A2 以 packed 界重验） |
| UB/L1/L0C 预算 | dense 真机实现的 buffer 清单 | §4.5（90624 B）、§4.6（172032/163840 B） |
| 编程模式 `hybrid` | dense §2 | §2（结论相同，偏离理由相同） |

### 11.2 差异项 → 本文落点

| 差异 | 主论证 | 证据 | 派生章节 |
|------|--------|------|---------|
| ① 变长块枚举 | §0.6 R6 | P1/P2/P5（动态 grid、融合扫描、静态化方案） | §1.6.2 #9、§5.4 第 1 条、§5.5 |
| ② head-major `lse/delta` | §0.6（R7 旁注）、§1.6.3 | P2 设备码（256 B 连续段） | §3.2 步骤 3、§4.1、§4.4 |
| ③ 逐 request 边界读写 | §0.6 R5/R7 | P2 设备码（双分支 + 原子 store）、P3（符号长度备选）、A3（溢出零行） | §1.6.1 A3、§3.3 注 3、§5.4、§8.3 |

### 11.3 本版未复用（并说明为何）的 dense 结论

| dense 结论 | 本函数处置 | 理由 |
|-----------|-----------|------|
| 「packed 分支在本函数内**舍弃**，单独设计」 | 本函数**就是**那个单独设计 | 设计对象变化 |
| 「平台 clamp 足以处理尾块写回」 | **不成立**，改 R7 双分支 | 平台 clamp 只按张量 extent（P2 设备码实测），不认 request 边界；`dk/dv` 的整块写零会污染邻居 |
| 「整块 L0C→UB（无 `sid`）可行」 | 不复用（dense 已实测不可行） | dense Stage 3 P1：整块 L0C→UB 只落一个 AIV |
| 「`T.clear` 显式清零」 | 不复用 | P4：`T.clear` 不支持 L0C；改用 `clear_accum` + 非空循环保证 |
