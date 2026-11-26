# MLA prefill 反向主 pass（_bwd_kernel，packed / MLAVarlenBwdOp）算子设计文档

> **版本：v4（2026-11-25，revision_index=1，Cube 化重设计）**。
> 历史版本：v3 = grid-slab 修复版（`history_version/design_v3_pre_cube.md`，**原样保留**）、
> v2/v1/v0 见其文件头。
>
> **本版只改执行结构，不改数学、不改算法族、不改输入输出契约**：把主计算体从 v3 的
> 「L1 操作数路径的 `T.gemm` + 向量侧转置副本」换成 **dense `MLADenseBwd` 已验证的 C2 Cube 核**
> ——**5 个矩阵乘全走显式 L0 路径**（`alloc_l0a/l0b/l0c` + 每次 GEMM 前一次 L1→L0 `T.copy`）、
> **`trans_A=True` 复用单份 `dsT`**（消掉向量侧跨步转置）、**`Nk=96` 的 key 块**、**2-tile 合并写 epilogue**、
> 半区 `dual_copy` 2:1、store-atomic `dq_acc`。packed 的三处差异（变长块枚举、head-major 统计量、
> 逐 request 边界读写）与 head slab / grid 上界断言**全部保留**。
>
> **根因（生产实测，不是推断）**：v3 的每迭代 5.9 µs 里有 **3.75 µs（63%）是向量侧的跨步转置
> `ds_u[j,i]`**（dense 同结构探针 `epi1` 4.302 µs vs 合并写 `epi0` 0.825 µs），另有 MTE2 填充与 GEMM
> 串行不重叠；msprof 的 `aic_cube_ratio = 8.3%` 与「GEMM 段只占整链 10%」互为印证——**Cube 不是在算，
> 是在等**。修复不能让 GEMM 变快，只能把**向量段的跨步 scatter 整项消掉、把 tile 放大摊薄每行填充**，
> 让迭代时间从"向量受限"变成"AIC 四管道之和"。
>
> **本版相对 v3 的关键调整**（逐条对照与「为何不再犯同一错误」见 **§11**；三条必须保留的 v3 成果：
> grid 上界断言、head slab 发射、packed 逐 request 边界收口）。
>
> 迁移任务（harness 子模式），源算子：`/workspace/TileOPs/src/tileops/_kernels/attention/mla/bwd.py`
> 的 `_bwd_kernel`（GPU TileLang，`MLADenseBwdKernel` / `MLAVarlenBwdKernel` **共用同一工厂**）。
> 目标平台 `ascend-950pr-9579`（dav-3510，CANN 9.2.1，TileLang ascend 0.1.15），编程模式 **hybrid**
> （Cube + Vector 同 kernel）。
>
> 三个 launch 的契约不变：`_delta_kernel`（fp32 `delta`，head-major）→ **`_bwd_kernel`**（本函数：写 `dk/dv`，
> 向 fp32 `dq_acc` 原地累加）→ `_dq_land_kernel`（`dq = dq_acc * scale` 并 cast）。`_bwd_kernel` 内部按 head
> slab 分多次 launch（同一编译产物、不同 `h_off` 实参），对外仍是「一个 launch 序列」。

---

## 0. 源算子解读与迁移分析

> 方法论：tilelang-ascend skill 的 `references/migration-analysis.md`（三问解读 → 耦合性四态 → 重设计）。
> 本章结论驱动 §1–§11。解读对象是**源工厂的 packed 分支**（`dense_seqlen_q == 0`）。
> **v4 只改 §0.5/§0.6 的处置与重设计形态（L1 路径 → 显式 L0 路径 + `trans_A`），
> §0.1–§0.4 的语义/算法/优化手段解读逐字沿用 v3（与源码一致，本版重新逐行核对）。**

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

**语义要素逐一确认**（源码逐行核对，全部要素均已在源码中定位并核验）：

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
产出该 request 的 `dV`、`dK`（含 `scale`）与 `dQ` 的 fp32 累加」。与 dense 相比语义不变，变化的是
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

**NPU 侧签名口径（packed，与 dense 的关键差异）**：dense 版把 GM 张量改成 4-D batch 视图 `(B,S,H,D)`，
让平台 `AscendInsertOOBPadding` 的 clamp 维度等于「本 request 的 `S`」，从而根治 `dk/dv` 尾块跨 request 覆盖
（dense v3 的 B1 阻塞项）。**packed 不能用这个办法**：`(T,H,D)` 是单张连续张量，没有 `S` 维可切，
任何 view 的行上界都只能是 `total_kv`。因此 packed 的等价手段是**内核内的逐 request 行守卫**
（§0.6 R7 的双分支），与源算子 `bwd.py:255-260` 的 `if n0[0]+i < len_kv` 逐行同义。
本 kernel 的签名另有一个**内部 launch 参数** `h_off: int32[1]`（head slab 原点），不属于算子 I/O 契约（§4.1）。

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

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | v4 耦合性判定 |
|---|----------|------|------|-----------------|---------------|
| 1 | key-stationary 分块（S4/S5/S18） | `k/v` 只搬一次、`dK/dV` 只在块尾写一次 | 一块 key 常驻 SMEM，`dk_f/dv_f` 跨 query 循环寄存器累加 | SMEM + 大寄存器堆 | **保留**（累加器迁 L0C，§0.6 R2） |
| 2 | recompute 而非物化 `P/dS`（S8/S10/S13） | 消 O(Σ l_q·l_kv) 中间缓冲与 GM 往返 | 每个 query 块重算 `s` 与 `dP` | 无（纯算法层） | **保留**（§1.6.0 R2） |
| 3 | 因果块级剪枝（S6） | 少算约一半 (key 块, query 块) 对 | `loop_st` 从可见性下界起算 | 无 | **保留**（界为运行期量，已编译） |
| 4 | tensor core GEMM（S8/S9/S12/S14/S16） | 矩阵算力 | `T.gemm` → mma | tensor core | **等价替换**：显式 L0 路径（5 个全 L0，不混用；§0.6 R6） |
| 5 | 寄存器 fragment 累加器（S5/S12/S14） | 免 SMEM 往返 | `alloc_fragment` + fp32 累加 | 大寄存器堆 | **重新设计**：L0C 累加器 + 半区 UB 中转（R2） |
| 6 | **`transpose_A=True` 复用同一 `ds`**（S16） | 省一份 `dS` 的 SMEM 副本 | mma 直接吃转置操作数 | mma 的 A 转置控制 | **保留**（v3 曾误判为「重新设计」）：显式 L0 路径支持 `trans_A`，转置由 `asc_copy_l12l0a_transpose` 在 MTE1 完成——本轮 P7 设备码实测该指令存在 |
| 7 | `exp2` + `scale*LOG2E` 折入指数（S10） | 用便宜底 2 指数指令 | `exp(x)=exp2(x*log2e)` | SFU 上 `exp2` 快于 `exp` | **等价替换**：自然域 `T.exp`（A1，机器验证 1 ulp） |
| 8 | mask 用 `if_then_else` 选择而非乘掩码（S10） | mask 位置严格为 0，`lse=-inf` 行不产生 NaN | select | 无 | **等价替换**：向量 `select` + 运行时块级特化（A2，packed 界重验） |
| 9 | `T.Pipelined(num_stages=2)`（S7/计算重叠） | 消 MTE 停顿 | 双 buffer 轮转 | 异步拷贝引擎 | **等价替换**：`num_stages=2`（dense 实测仅 `qT_s` 真正拿到 2 版本，Z3 回退，§6.3） |
| 10 | `T.atomic_add` 写 `dq_acc`（S17） | 免 workspace / 免跨 block 归约 | 全局 fp32 原子加 | 全局原子 ALU + L2 | **等价替换**：store-atomic（R5 + A3 溢出零行） |
| 11 | `lse/delta` 行广播（S10/S13） | 每行一次读代替逐元素重算 | SMEM 标量读 | — | **等价替换**：lane = query，`lse_u[j]` 为 lane 私有读（§1.6.3） |
| 12 | warp 划分 `FullRow` + `threads=256` | 匹配 mma 的 warp 分形 | warp-level 分片 | warp 32 线程 | **重新设计**：`T.SimtVF(threads=128)` + 半区（`_HN=48`）+ `sid` 捕获（R4） |
| 13 | （NPU 适配构造，非源码手段）`clear_accum` | 表达「首块清零 / 跨块累加」 | mma 零累加位 | — | —（构造项） |
| 14 | **packed/dense 双分支 + `GroupTiling`（S1/S2）** | 一个工厂服务两种 layout | 编译期 `dense` 分支 + 块表前缀扫描 + 二分解码 + `live` | 无（算法层） | **重新设计**：融合式 O(B) running-prefix 解码（R6） |
| 15 | **块表每 CTA 重算**（S1，源码注释明示） | 免一次 host 侧块表构造与额外 GM 张量 | 每 CTA 独立累加 `s_cum` | SMEM + warp scan 原语 | **重新设计**：同一 O(B) 扫描，无块表（R6） |

**关于 #13 的口径**：源码**没有** `clear_accum` 参数，只用 `T.clear(st/dpt/dq_f/dk_f/dv_f)`
（`bwd.py:200-201,225,227,250`）显式清零；`clear_accum` 是本设计的 NPU 侧构造（§3.3）。
**注意**：`T.clear` 作用于 L0C **不可用**（v3 探针 P4 实测 `InternalError: Unsupported scope shared.l0c`），
故 L0C 的清零只能由 `clear_accum=(kq == loop_st)` 表达。

**小结（第三问 Why fast）**：共 15 项，其中 3 项纯算法层（#1/#2/#3 保留）、6 项有 NPU 等价形态
（#4/#7/#8/#9/#10/#11 等价替换，另 #6 现判保留）、3 项需重设计（#5/#12 与 packed 的 #14/#15）、
1 项为迁移侧构造（#13）。**与 v3 相比只有一处判定变化：#6 从「重新设计（NT-only + 向量侧转置副本）」
改判「保留」**——这正是本版性能修复的根因（§11-A1）。

### 0.5 硬件耦合性分析与 NPU 适配决策

判定问题：实现算法与优化手段是否硬件强相关？能否用在 NPU 上？依据一律指向本平台已核验资料
（`.agents/platforms/toolchains/ascend/references/programming.md`、`knowledge/` 下的 repro 条目、
dense 的**真机**实现/检视，以及本轮 P7–P9 编译探针与设备数值原型）。

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 方案 / 依据 |
|------|------|-----------|-----------------|------|----------------|
| 数学语义 / I/O 契约 / 边界语义 | 语义 | 无 | — | **保留** | 语义层无条件保留；golden 依 §0.1 独立实现（§8.1） |
| 二维 block 网格 `T.Kernel(tiles, heads)` | 优化 | CUDA blockIdx 二维 | 无（一维核网格） | **重新设计** | R1：一维网格 + 块内 `bid // slab` 解码；packed 下 grid 是**动态表达式 × 编译期常量** |
| key-stationary 分块 + `dK/dV` 跨块累加（#1） | 算法 | 无（结构） | 有 | **保留** | 同一结构；累加器迁 L0C（R2） |
| recompute 不物化 `P/dS`（#2） | 算法 | 无 | 有 | **保留** | 单遍结构，`P/dS` 只活 L0C/UB（§1.6.0 R2） |
| 因果块级剪枝（#3） | 算法 | 无 | 有 | **保留** | 同一 `loop_st/loop_ed` 语义；packed 下界为运行期量（P7 设备码已展开为 `>> 6`） |
| tensor core GEMM（#4） | 优化 | mma / fragment | 有（Cube MAD） | **等价替换** | **显式 L0 路径**（`alloc_l0a/l0b/l0c` + 手工 L1→L0 `T.copy`）：平台已核验形态（`PL-950-gemm-tiles`），dense 真机 L0 13/13、生产 150.7 ms；本轮 P7 设备码 5 个 `asc_mmad` |
| 寄存器 fragment 累加器（#5） | 优化 | 每线程 255 寄存器 | 部分（Cube 侧只有 L0C 可累加） | **重新设计** | R2：5 个 L0C 累加器（221,184 B，84.4%）+ 半区 UB 中转 |
| **`transpose_A=True`（#6）** | 优化 | mma 的 A 转置控制 | **有（L0 路径 MN-major）** | **保留**（v3 误判） | dense 探针 2 的 (T,T) 组合编译 + 数值 PASS；本轮 P7 设备码 `asc_copy_l12l0a_transpose` ×1。**由此向量段不再产第二份 `dS` 副本** |
| `exp2` + `LOG2E`（#7） | 优化 | SFU 指令选择 | 有（`T.exp` 已核验） | **等价替换** | 自然对数域 `T.exp(s*scale − lse)`（A1，1 ulp） |
| `if_then_else` 掩码（#8） | 优化 | 无 | 有 | **等价替换** | `T.SimtVF` 内 `T.if_then_else` + 运行期块级特化（A2，packed 界重验） |
| `T.Pipelined(num_stages=2)`（#9） | 优化 | cp.async 双槽 | 有（`num_stages`） | **等价替换** | `num_stages=2`（§6.3）；**实测只有 `qT_s` 拿到 2 版本**（Z3 unsat 回退），故不承诺重叠收益 |
| `T.atomic_add` 逐元素（#10） | 优化 | 全局原子 ALU | 有（store-mode 原子） | **等价替换** | `set_atomic("add","float32")` + L0C→GM `T.copy`；packed 的新增论证是「溢出零行」（A3） |
| `lse/delta` 行广播（#11） | 优化 | SMEM 标量读 | 有（UB lane 索引读） | **等价替换** | lane 落在 query 轴，`lse_u[j]` 为 lane 私有读（§1.6.3） |
| warp 划分 + `threads=256`（#12） | 优化 | warp 32 线程 | 无（NPU 无 warp） | **重新设计** | R4：`T.SimtVF(threads=128)` + 半区 + `sid` 捕获（dense 真机 + 本轮 P7 设备码） |
| `FullRow` policy 参数 | 优化 | warp 划分启发式 | 无 | **舍弃** | L0 路径按 mad 形状与 L0 布局决定分形，`policy` 不入 lowering；不影响正确性 |
| **`GroupTiling.cumsum_offsets`（#14/#15）** | 优化 | **warp shuffle 前缀扫描** | 无（ascend 的 `T.Kernel` 在 kernel 作用域不建立线程域，`T.shfl_*` 只能落在 `SimtVF` 内，结果需对 AIC 可见） | **重新设计** | R6：融合式 running-prefix 解码（O(B) 串行、纯 local 标量、无共享块表） |
| **`GroupTiling.decode` 二分查找** | 优化 | 共享块表 + `T.serial` | 有（纯标量） | **等价替换** | 合并进 R6 的同一次扫描（first-match 赋值），语义与二分等价 |
| **`tile_upper_bound` 的 `live` 谓词**（S2） | 优化 | 无 | 有 | **保留** | 同一 `live` 语义；grid 取同一上界（R1/R6） |
| **逐 request 边界守卫**（S4/S7/S17/S18 的 `n0+i < len_kv`、`m0+j < len_q`） | 优化 | 无 | 有（向量 select / 双分支） | **等价替换** | 装载与 `dq` 用掩码臂收口（配合平台 clamp），`dk/dv` 写回用 R7 双分支 |
| **`dk/dv` 的按 request 偏移写回** | 优化 | 无 | 部分（OOB 只按张量 extent 裁剪；packed 无 `S` 维） | **重新设计** | R7：内部半块整 tile 快路 + 尾半块逐 lane 守卫（P7 设备码双路径） |
| packed 的 `T_q/T_kv` 动态 extent + 动态 grid | 优化 | grid 由 host 按 shape 计算 | 有（`T.dynamic` + 动态 1-D grid） | **保留** | 与 `examples/paged_cache_write` 同机制；P7/P8/P9 编译级确认，设备数值原型 6/6 PASS |
| host `torch.zeros(dq_acc)` | 语义 | 无 | 有 | **保留** | store-atomic 的累加契约（不变） |
| **单次 launch 的 `T.Kernel(grid)` 块数上界 65,535** | 优化（任务映射） | CUDA grid 无 16 位块索引 | **无**（`grid>65535`：整倍数 abort，余数非 0 **静默只执行** `grid % 65536` 块） | **重新设计** | R8：head slab 多次 launch（单次 `grid = tiles × slab`）+ host 启动前断言；A4 逐 bid 穷举验证 |

统计（23 行判定）：**保留 8 项**（数学语义、#1、#2、#3、**#6**、`live`、动态 extent、host `zeros`）、
**等价替换 8 项**（#4、#7、#8、#9、#10、#11、二分、逐 request 守卫）、**重新设计 6 项**（R1、R2、R4、R6、R7、R8）、
**舍弃 1 项**（`FullRow` policy）。

### 0.6 NPU 算法重设计

**重设计项 R1：二维 block 网格 → 一维核网格 + 块内解码（packed 版）**

- **源方案**：`T.Kernel(num_tiles, heads)` 二维 grid（`bwd.py:155`），硬件直接给出 `(tile, h)`。
- **NPU 新算法**：`T.Kernel(grid)` 一维核网格，单次 launch 的
  `grid = (total_kv // 96 + batch) * slab`（动态表达式 × 编译期常量 `slab`）；块内
  `tile = bid // slab`（**静态除数**）、`h = h_off[0] + bid % slab`（运行期 head 原点 + 静态槽位），
  `h >= heads` 的块（尾 slab）立即返回。共 `ceil(heads/slab)` 次 launch，单次 `grid ≤ 65,535`（§5.6）。
  head 放在**内层**的两个理由：(a) 静态除数（运行期除数未取证，§1.6.3 候选 6/8）；(b) 同一 (request, key 块)
  的相邻 bid 落在相邻 head，`(token, head, dim)` 行主序下同几行的不同 head 切片相邻，L2 行复用更好。
- **语义保持论证**：`(tile, h) ↔ (launch, bid)` 是到 `[0, tiles) × [0, heads)` 的双射（A4 对全部 manifest
  几何逐 bid 穷举：缺失 0、重复 0、越界 0）；任务集合完全一致，无算术改变。非 live 的 ≤ `batch` 个 tile 由
  `live` 丢弃，尾 slab 的 `h ≥ heads` 块由守卫立即返回。

**重设计项 R2：寄存器 fragment 累加器 → L0C 累加器 + 半区 UB 中转（复用 dense 真机结论）**

- **源方案**：`st/dpt/dk_f/dv_f/dq_f` 全 `T.alloc_fragment`；`dk_f/dv_f` 跨 query 循环常驻。
- **NPU 新算法**：Cube 的累加器只能是 L0C（`gemm_mad.py:_prepare_lower` 要求 C 带 Ascend fractal layout）：
  `dk_c (96,192) fp32 = 72 KiB`、`dv_c (96,128) = 48 KiB` 跨循环累加；`st_c/dpt_c (96,64) = 24 KiB` 各、
  `dq_c (64,192) = 48 KiB` 每迭代清零；合计 **221,184 B / 262,144 B（84.4%）**，仍在 240 KiB 红线内（§4.6）。
- **实现形态约束（dense 真机实测，必须保留）**：`T.copy(L0C→UB)`（`dual_dst_mode=0`）只落**一个** AIV 的 UB
  （dense v1 探针：99.2% 元素错位）。因此每个 L0C↔UB 交叉一律 `T.dual_copy` 2:1（每 AIV 持 key 半区
  `_HN = 96/2 = 48` 行）；L1 侧的三份 fp16 tile（`pT_s`、`dsT_s`）仍是整块 `(96,64)` 的 GEMM 操作数。
- **语义保持论证**：累加顺序保持「`dV/dK` 沿请求内 `m` 升序、`dQ` 沿 key 块跨核累加」，dtype 仍 fp32；
  K 维归约由 mad 完成（与源 mma 同为 fp32 累加、分组不同），与源一样只承诺 fp32 累加精度而非逐位一致。
  半区切分只改变「哪个 AIV 计算哪 48 行」，逐元素无跨 lane 归约，两个 AIV 的并集恰好覆盖整块。
- **证据**：dense `_bwd_kernel.py` 真机 L0 13/13、L1 6/6、生产 150.7 ms / cube 25.2%；本轮 P7 设备码
  `asc_copy_l0c2ub` ×4（两套 epilogue 臂各两次）+ `__mix__(1, 2)`；设备数值原型 6/6 PASS（§1.6.4）。

**重设计项 R3（v4 改判：保留而非重设计）：`trans_A` 复用单份 `dsT`**

- **源方案**：`dq_f = gemm(ds_s, k_s, transpose_A=True)`——同一份 `dS` 服侍 `dK`（作 A）与 `dQ`（作转置 A）。
- **v3 的错误判定**：v3 认为 L1 路径只支持 NT（`gemm_mad.py` 的 L1 断言），故让 `dQ` 另造一份
  `ds (Mq,Nk)` 的 L1 副本，并在向量段用跨步写 `ds_u[j,i]` 生成它。实测该 scatter 占内层迭代的
  **3.75 µs / 5.95 µs = 63%**（dense §1.6.4 A），是把 kernel 拖成 Vector 受限的直接原因，也是本版要修的根因。
- **v4 新算法**：5 个乘法全部写成显式 L0 的 `C = A(M,K)·B(N,K)^T`，`dQ` 用 `transpose_A=True`：

  | 乘积 | A（L0A） | B（L0B） | C（L0C） | (trans_A, trans_B) |
  |---|---|---|---|---|
  | `S^T` | `k_s (96,192)` | `q_s (64,192)` | `st_c (96,64)` | (F,T) |
  | `dP^T` | `v_s (96,128)` | `do_s (64,128)` | `dpt_c (96,64)` | (F,T) |
  | `dV` | `pT_s (96,64)` | `doT_s (128,64)` | `dv_c (96,128)` | (F,T) |
  | `dK` | `dsT_s (96,64)` | `qT_s (192,64)` | `dk_c (96,192)` | (F,T) |
  | `dQ` | `dsT_s (96,64)` | `kT_s (192,96)` | `dq_c (64,192)` | **(T,T)** |

  同一份 `dsT_s` 既作 `dK` 的 A（K-major，L0A 的 `a_k` 切片）又作 `dQ` 的 A（MN-major，L0A 的 `a_t`），
  后者由 `asc_copy_l12l0a_transpose` 在 L1→L0A 搬运中完成转置（MTE1，每迭代 12,288 B）。
- **语义保持论证**：`(A B^T)` 与 `(A^T)^T B^T` 逐元素恒等；转置只改变操作数搬运顺序，不改变任何算术，
  也不改变 fp16 舍入点（`dsT` 只 cast 一次）。代价是 L0A 需两块布局不同的 tile（`a_k` 36,864 B + `a_t` 12,288 B）。
- **证据**：dense 探针 2 的 (T,T)/(F,F)/(F,T) 三组合数值 PASS；本轮 P7 设备码含
  `asc_copy_l12l0a_transpose` ×1、`asc_mmad` ×5；设备数值原型 6/6 PASS。

**重设计项 R4：warp/线程级并行 → SimtVF 半区 lane 映射 + `sid` 捕获**

- **源方案**：`threads=256` 线程块，`T.Parallel(bn,bm)` 的元素映射由编译器按 256 线程划分。
- **NPU 新算法**：Cube 侧由 mad 承担；向量侧在 `T.SimtVF(threads=128)` 内以 `T.Parallel(48, 64)` 覆盖
  **半区**（最内轴为 query，`lse_u[j]` 为 lane 私有读）。副块编号用
  `sid = T.call_extern("int64","asc_get_sub_block_id")` 在 kernel body 取一次、由 `T.SimtVF` 捕获
  （`asc_get_sub_block_id()` 不可在 `__simt_vf__` 体内调用）；`T.Vector(vector=2) as sid` 在本工具链
  **不可编译/值不确定**（`InternalError … dup sid`，dense §3.5.4 反例 2 + `TRAP-950-aiv-sid-vector-frame`），
  不得使用。
- **语义保持论证**：`sid` 只进入两个位置——掩码的 key 索引 `gi = n0 + sid*48 + i` 与写回起点
  `own = koff + sid*48`；二者与 `dual_copy` 的分片方向一致，故每个 AIV 只处理自己那 48 行，并集即整块。
  逐元素映射无跨 lane 归约，结果与划分无关。
- **证据**：P7 设备码 `asc_get_sub_block_id` ×2（AIC/AIV）、4 个 `__simt_vf__`（快路 / 掩码臂 / 块尾 cast /
  尾块守卫存储）；设备数值原型含 `H=3, slab=2` 的尾 slab 与 `slab=1` 三 launch，均 PASS。

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
  - 张量末尾的溢出由平台 clamp 裁到 `total_q`（P7 设备码 `min(64, max((total_q - q_start) - kq*64, 0))`）。
- **语义保持论证**：`dq_acc += tile` 与逐元素原子加是同一数学累加；fp32、顺序仍不确定（与源一致）。
  「零行」结论由 §1.6.1 A3 机器验证（31,296 个溢出差分元素与 2,608 个溢出 dq 行全为精确 0，0 违反）。

**重设计项 R6（packed 专属）：`GroupTiling` 的 warp 前缀扫描 → 融合式 running-prefix 解码**

- **源方案**（`bwd.py:138-139,181-184`）：`tile_upper_bound` 定上界；`cumsum_offsets` 在 `num_groups > 8`
  时走 warp shuffle 扫描（`T.shfl_up` × `WARP_SHUFFLE_STAGES`、`get_thread_extents()`、`T.sync_threads`）
  建共享块表 `s_cum[B+1]`；`decode` 二分求 `(request, n0)`。
- **NPU 新算法**：一次**融合的 O(B) 串行扫描**，全部在 core-local 标量上完成，**不物化块表**：

  ```
  req = 0; n0 = 0; cnt = 0
  for g in T.serial(batch):                 # batch 是编译期常量，循环可完全展开
      nt = (cu_kv[g+1] - cu_kv[g] + 95) // 96
      if (tile >= cnt) & (tile < cnt + nt): # 首个匹配即 decode 的结果
          req = g; n0 = (tile - cnt) * 96
      cnt = cnt + nt                        # 前缀和（running）
  live = tile < cnt                         # == tile < s_cum[batch]
  ```

- **语义保持论证**：`(i)` `nt` 与源 `tiles(offsets, g)` 逐字同式；`(ii)` `cnt` 在 g 的末尾等于
  `s_cum[g+1]`，故 first-match 赋值给出的 `g` 与 `decode` 的二分结果相同（二分求的正是
  `s_cum[g] <= tile < s_cum[g+1]` 的 g），`n0` 同式；`(iii)` 扫描结束 `cnt == s_cum[batch]`，`live` 同义；
  `(iv)` 源对 `num_groups <= 8` 本就退化为同形状的串行 prefix 循环，本设计把该形态推广到任意 B 并删除块表。
  不做向量化的理由见 §1.6.2 #9（决定 block 级控制流与全部全局地址，且必须对 AIC 与两个 AIV 一致）。
- **与 L0C 累加器的交互（本版必须回答的点）**：解码**完全发生在 query 循环之前**，只写 core-local 标量
  `req/n0/live`；5 个 L0C 累加器的首次触碰仍由 `clear_accum=(kq == loop_st)` 在循环体内表达。
  两者之间唯一的联系是 `loop_st = max(0, (n0 - shift) // 64)`（运行期值）——P7 设备码把它展开为
  `max(0, ((n0_dyn + lq) - lkv) >> 6)`，`loop_ed = max(((lq + 63) >> 6), loop_st + 1)`。
  **不存在「解码与 L0C 初始化顺序」的耦合**：解码是纯标量前缀，不读不写任何 L0C/UB/L1 缓冲。
- **证据**：P7/P8/P9 设备码 `for (int32_t g = 0; g < 3; ++g)` + `if (cnt <= tile && tile < cnt+ntg)`；
  设备数值原型 6/6 PASS（含 `len_q > len_kv` 的 `lse=-inf` 行、尾块、head slab）。

**重设计项 R7（packed 专属）：`dk/dv` 的按 request 写回 → 内部快路 + 尾半块逐 lane 守卫**

- **源方案**：`if n0+i < len_kv: dk[kv_start+n0+i, h, :] = cast(dk_f[i,:]*scale)`（`bwd.py:255-260`）——
  **逐元素守卫**，天然不会写出 request 边界。
- **风险起点（packed 与 dense 的关键差异）**：平台的 OOB 裁剪只针对**张量 extent**。P7 设备码的装载裁剪是
  `min(96, max((total_kv - kv_start) - n0, 0))`——它保护到**整个张量的末尾**，**不**保护 request 边界
  （dense 用 4-D batch 视图让 clamp 维度等于本 request 的 `S_kv`；packed 没有这一维，故必须在核内补）。
  于是尾块整块写回会把「精确 0」覆盖到**下一个 request 的 dk/dv 行**（`dk/dv` 是赋值而非累加，写 0 即破坏）。
- **NPU 新算法**：块尾双分支（谓词全为运行期量）

  ```
  own    = kv_start + n0 + sid*48         # 本 AIV 的 48 行起点
  kv_end = kv_start + len_kv
  if own + 48 <= kv_end:                  # 内部半块：整块 DMA 快路
      T.copy(out_k, dk[own : own + 48, h, :]);  T.copy(out_v, dv[own : own + 48, h, :])
  else:                                   # 每个 request 的尾半块：逐 lane 守卫
      with T.SimtVF(threads=128):
          for i, d in T.Parallel(48, 192):
              if own + i < kv_end: dk[own + i, h, d] = out_k[i, d]
          for i, d in T.Parallel(48, 128):
              if own + i < kv_end: dv[own + i, h, d] = out_v[i, d]
  ```

- **语义保持论证**：快路条件 `own + 48 <= kv_end` 保证 48 行全部落在本 request 内（整块写回与逐元素写
  逐元素相同）；慢路是源逐元素守卫的直接对应（同一比较、同一 dtype cast 值），且对 `own >= kv_end`
  的 AIV（`len_kv - n0 < 48` 时的第二个 AIV）自然写 0 个元素。
  代价：慢路每个 request 最多触发 1 次（每 AIV 半块），影响 ≤ `B·H·2` 个块，全量 ≤ 1/`ceil(len_kv/96)` 的写回量。
- **证据**：P7 设备码同时出现两条路径——`if (sid*48 + kv_start + n0) + 48 <= kv_start + lkv` 下的
  `asc_copy_ub2gm_align(...)`，以及 `else` 下的 `asc_vf_call<main_kernel_simt_vf_3>`（内部逐 lane 谓词
  `... + kv_start + n0 + i < kv_start + lkv`）。设备数值原型在 `lens 128/320`、`320/128`、`96/96` 上
  **逐 request 比对**均 PASS（§1.6.4），即未出现跨 request 泄漏。

**重设计项 R8：单次 launch 的块数上界 65,535 → head slab 多次 launch + 启动前断言（v3 成果，保留）**

- **触发（实测，非推断）**：`T.Kernel(grid)` 的块索引按 16 位携带。`grid ≤ 65,535` 正常；`grid` 为 65,536
  整数倍时 abort（`retCode 0x26` / `error code 507035`）；**余数非 0 时不报错，只执行 `grid % 65536` 个块**
  （最危险形态）。v3 的 `ds-v3-8x4k`（Nk=64）`grid = 66,560`，实测只有 1,024 个 CTA 执行，request 1..7 的
  `dq/dk/dv` 全为 0。证据：`/tmp/opencode/MLA_VARLEN_BWD_8x4K_FINDINGS.md`（逐 CTA 映射、逐 request 对照、
  H=126/127/128 的 A/B 计时）。
- **NPU 新算法**：`slab = min(32, heads, 65535 // tiles)`（编译期常量），共 `ceil(heads/slab)` 次 launch，
  第 i 次携带 `h_off = i*slab`；`plan_launch` 在**发射前**断言 `grid ≤ 65,535`（越界抛 `ValueError`，绝不静默）。
- **本版的重要更新（诚实记录）**：改用 `Nk=96` 后 `tiles = total_kv // 96 + batch` 只有 Nk=64 时的约 2/3，
  manifest 的全部几何**都不再逼近 65,535**（8x4k：44,672 而非 66,560；mixed：43,776 而非 65,280）。
  但 **slab 与断言必须保留**，因为 `tiles` 是**运行期量**、算子契约允许任意 `T_kv`：A4 的 `auto-shrink`
  几何（2,100 tiles）仍在 `slab=31` 处触发收缩、越界几何仍抛 `ValueError`（§1.6.1 验证输出）。
- **语义保持论证**：(i) slab 区间是 `[0, heads)` 的划分（Σ active_heads = heads，两两不交）；(ii) bid ↔ (tile,h)
  双射（A4 逐 bid 穷举，违反 0）；(iii) 三个输出都是 `(token, head, dim)`，head 占行内连续 `D` 个元素，
  故不同 launch 的写地址集合不交；(iv) head 维不参与任何归约 ⇒ launch 之间无需同步；
  (v) `h` 只进入 GM region 中间轴与掩码索引，5 个 GEMM/掩码/原子/双分支逐字不变。
- **证据**：P7 设备码 `int32_t h = (h_off[0] + (((int32_t)block_idx) & 1))`（AIC 与 AIV 各一份）、
  `tile = block_idx >> 1`、`if (h < 3)` 包住整个 body；A4 机器验证 475,113 个任务/几何点，违反 0。

**重设计项 R9（本版新增）：dense C2 核在 packed 上的适配（4 处必须改的地方）**

dense C2 的计算体可以直接复用（同一源码工厂、同一维度参数、同一 tile），但**有 4 处必须按 packed 重写**，
逐项给出「dense 写法 → packed 写法 → 依据」：

| # | dense C2 写法 | packed 写法 | 依据 / 证据 |
|---|--------------|------------|------------|
| ① | `bid` 直接解出 `(h, b, n0)`（全静态：`h = bid // (B*NT)`, `n0 = (tile % NT)*96`） | `tile = bid // slab` + **O(B) 融合解码** + `live`（R6）；grid 含运行期量 | `total_kv/T_q` 是 `T.dynamic`，块数只能由 `cu_seqlens` 现算 |
| ② | `lse/delta` 为 `(B,H,S)`，`T.copy(lse[b,h,m0:m0+64], lse_u)` | **head-major `(H,T_q)`**：`T.copy(lse[h, qoff:qoff+64], lse_u)`（一次 256 B 连续搬运） | 源 packed 分支 `stat_shape=(heads,total)`；P7 设备码 `asc_copy_gm2ub_align(..., 1, 256, ...)` |
| ③ | `dk/dv` 写回靠 **4-D batch 视图**让 OOB clamp 收到本 request 的 `S_kv` | **R7 双分支**（整半块快路 / 尾半块逐 lane 守卫），因为 packed 没有 `S` 维可切 | 见 R7；P7 设备码双路径 |
| ④ | `dq` 整块原子写回，尾部由 OOB clamp 收到 `S_q` | 同一整块原子写回 + **A3 溢出零行**不变式（跨 request 的行是精确 0） | A3 在 Nk=96 下重验：31,296 溢出元素 + 2,608 dq 行全 0 |

**这 4 处之外的其余部分逐字复用**：5 个 L0 GEMM 的操作数/累加器形状、`clear_accum` 语义、半区 `dual_copy`
（`_HN=48`）、`sid` 捕获、`kT` 最后重灌 `b_hi` 的顺序、`num_stages=2`、块尾 cast、store-atomic。
**head slab 与 L0 的交互结论（本版必须回答的点）：无耦合**——`h_off` 只决定 GM region 的中间轴与掩码的
逐元素 key 下标，不进入任何 GEMM 的 M/N/K（三者都取自 allocation 形状）、不改变 L0A/L0B 的复用顺序
（`a_k` 的 k/v/pT/dsT 四段、`b_hi` 的 doT/qT/kT 三段由 GEMM 序列决定，与 head 无关）；P7 设备码里
`h` 只在 GM 地址与 `if (h < 3)` 里出现，5 个 `asc_mmad` 的操作数与 slab 无关，即结构性证明。

**无重设计项之外的补充说明**：`delta` 的产生（`_delta_kernel`）与 `dq` 的缩放 cast（`_dq_land_kernel`）
不在本 kernel 内，本设计只依赖其接口契约（fp32 / fp32 in-out，head-major），未对其做任何假设。

### 0.7 标杆实现与上下文

- **源算子路径**：`/workspace/TileOPs/src/tileops/_kernels/attention/mla/bwd.py`（`_bwd_kernel`，
  `bwd.py:109-264`；packed 分支 `bwd.py:136-139, 176-184, 218-224`），仓库内抽取副本
  `examples/TileOPs/tileops/kernels/attention/mla_varlen_bwd/_mla_varlen_bwd_kernels.py`（逐行同源）。
- **golden 基准**：`/workspace/TileOPs/workloads/attention/mla.py` 的 `mla_bwd_reference`（fp32 torch，按
  request 物化 `probs` 后 `einsum` 反向，bottom-right 对齐，`nan_to_num` 处理 `lse=-inf` 行）与
  `MLAVarlenBwdWorkload.ref_program`（把 fp32 梯度**只舍入一次**到输入 dtype）。§8.1 的 golden 移植该实现，
  **不复刻** §0.6 的分块算法（独立性见 §8.1 末段）。
- **精度用例**：`/workspace/TileOPs/tests/ops/test_mla_bwd.py`；容差来源 `workloads/numerics.py` 的
  `reference_tolerance` 与 `MLAVarlenBwdWorkload.verification`（atol 下限 5e-3）。
- **manifest workloads**（`src/tileops/manifest/spec/attention.yaml` 的 `MLAVarlenBwdOp`）：
  `ds-v3-8x4k`（B=8×4096，T=32,768，H=128）、`ds-v3-mixed`（T=32,256，B=6，lens 512…16384，H=128）、
  `kimi-k2-4x8k`（B=4×8192，T=32,768，H=64）。全部 `is_causal=true`，`D_qk=192`、`D_v=128`。
- **dense 侧已冻结的参照（本版的复用来源）**：
  `examples/mla_dense_bwd/_bwd_kernel/DESIGN.md`（v3）、`_bwd_kernel.py`（真机 L0 13/13、L1 6/6、
  生产 `ds-v3-b2-8k` **150.7 ms / 2.97 µs 每迭代 / cube 25.2% / vec 45.7%**）、`REVIEW.md`（通过）、
  `RETROSPECTIVE.md`（生产常数与两条硬约束：`block_n/2 % 16 == 0`、L1→L0 实测 299.7 GB/s/核）。
- **v3 的实测证据（保留引用）**：`/tmp/opencode/MLA_VARLEN_BWD_8x4K_FINDINGS.md`（grid 上界的三组决定性实验）、
  `/tmp/opencode` 下的 `probe_*` 逐 CTA/逐 request 对照。
- **知识条目可用性（`kb_stale_check --task-dir …` 实跑，2026-11-25）**：`stale_count = 43`，其中
  `PL-950-gemm-l1-fill`、`CONST-950-gemm-mte2-burst`、`TRAP-950-grid-over-65535` 均报 **stale**
  （toolchain stamp 不一致）。本文只把它们当**线索**，量化结论以 dense 生产实测（`evidence/msprof_pipe_capture1.csv`、
  `repro/CONST-950-mla-bwd-cube-split.py`）与本轮 P7–P9/设备原型为准。
  **另需记录**：调度简报提到的 `TRAP-950-nd2nz-rows-multiple16` 与 `CONST-950-mla-bwd-cube-split` **不是
  knowledge 条目**（全库检索无此 id），它们是 `examples/mla_dense_bwd/_bwd_kernel/repro/` 下的**可执行复现脚本**；
  本文按后者引用（§9.1 风险 4）。

### 0.8 复用与差异分区（本函数 vs dense C2）

**A. 可直接复用（dense 已真机/生产验证，且对本函数成立的结构性理由）**

| # | 复用项 | dense 侧证据 | 为何对对 packed 成立 |
|---|--------|-------------|----------------------|
| 1 | **算法族与分块**：key-stationary 单遍重算、`Nk=96 / Mq=64` | dense 生产 150.7 ms、cube 25.2% | 同一源码工厂，只差索引空间；维度参数完全相同（`D_qk=192, D_v=128`） |
| 2 | **5 个 NT 乘积 + 显式 L0 路径**（含 `dQ` 的 `trans_A`） | dense L0 13/13 + 探针 2 的四类 trans | 维度与 `m_split`/分形无关；**P7 设备码复现同一形态**（5 `asc_mmad` + 1 `l12l0a_transpose`） |
| 3 | **L0C 累加器布局**（`dk/dv` 跨迭代、`st/dpt/dq` 每迭代） | 同 tile 的容量表与真机 | 缓冲形状逐字节相同 ⇒ L0C 221,184 B（84.4%）结论直接承袭 |
| 4 | **半区 `dual_copy` 2:1 + `sid` 捕获** | dense 真机（整块 L0C→UB 只落一个 AIV，99.2% 错位） | 设备行为与 layout 无关；P7 设备码 `asc_copy_l0c2ub`×4 + 4 个 `__simt_vf__` |
| 5 | **2-tile 合并写 epilogue**（只写 `p16`、`dsT`） | dense `epi0` 0.825 µs vs `epi1` 4.302 µs（scatter 占 63%） | 逐元素性质，与索引空间无关；packed 版**不再有第三份 `ds` 副本** |
| 6 | **store-atomic `dq_acc`** + `dk` 乘 `scale` / `dv` 不乘 | dense 真机 L0-11 原子契约 | 三 launch 契约与本函数完全相同，`dq_acc` 同为 fp32 |
| 7 | **A1（自然域 exp）/ A2（块级 mask 特化）的等价结论** | dense 3/3 EQUIV_PASS | A1 是逐元素恒等式；A2 只需把 `S_q/S_kv` 换成运行期 `len_q/len_kv`（本函数重验，§1.6.1） |
| 8 | **UB/L1/L0A/L0B 预算表**（半区 129,536 B / 204,800 B / 49,152 B / 61,440 B） | dense 实现清单 | `_HN=48`、tile 形状相同 ⇒ 逐字节相同 |
| 9 | 编程模式 `hybrid`（Cube+Vector 同 kernel） | dense §2 | 同一 `__mix__(1,2)` 结构（5 GEMM + 2 段向量） |
| 10 | **`kT` 必须最后重灌 `b_hi`** 的顺序约束 | dense 原型首版曾因此得到 dQ 全错 | L0B 复用的机械约束，与 layout 无关（§9.2 常见错误表） |

**B. 因 packed 必须改（R9 的 4 项 + 分核/发射）**

| # | 差异 | dense C2 | 本设计 | 章节 | 证据 |
|---|------|---------|--------|------|------|
| ① | 任务枚举 | 静态 div/mod | 融合 O(B) 解码 + `live` + 动态 grid | §0.6 R6 | P7 设备码 `for(g...)`；原型 PASS |
| ② | 统计量 layout | `(B,H,S)` | **保持 head-major `(H,T_q)`** | §1.6.3 | P7 设备码 256 B 段搬运 |
| ③ | `dk/dv` 写回隔离 | 4-D 视图靠 OOB clamp | **R7 双分支**（packed 无 `S` 维） | §0.6 R7 | P7 设备码双路径；原型逐 request PASS |
| ④ | `dq` 越界 | OOB clamp 到 `S_q` | 整块原子 + **溢出零行**（A3） | §1.6.1 A3 | 31,296 + 2,608 个零元素 |
| ⑤ | 发射 | 单次 launch（22,016 块） | **head slab 多次 launch + 断言** | §0.6 R8、§5.6 | A4 475,113 个任务点 |

**C. 一句话**：本函数与 dense 是**同一源码工厂 + 同一设备码形态**，差异只在「任务索引空间 + 一维统计量
layout + 写回隔离手段」；所有复用项都由**维度与平台约束**决定，不依赖索引空间。

---

## 1. 概述

### 1.1 算子名称

`_bwd_kernel`（算子：`MLAVarlenBwdOp` 的主 pass，MLA prefill 反向，packed / THD）。

### 1.2 功能描述

给定前向的 `q, k, v, do, lse, delta`（packed）与 `cu_seqlens_q/kv`，按 key-stationary 单遍重算 `P` 与 `dP`，
产出 `dk, dv`（输入 dtype）并向 fp32 缓冲 `dq_acc` 原地累加 `dQ`（缩放 cast 由 `_dq_land_kernel` 完成）。
每个 request 只看自己的 key，query `i` 对齐到 key 位置 `i + len_kv - len_q`。

**v4 的执行结构**：5 个矩阵乘全部走 Cube 的**显式 L0 路径**，向量侧只产生两个**合并写**的 fp16 tile
（`P^T`、`dS^T`），`dQ` 用 `transpose_A=True` 直接复用 `dS^T`——源算子「一份 `dS` 服侍两个 GEMM」的优化被
完整保留（v3 曾为它付 63% 的向量代价）。

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

算法族选择见 §1.6.0：**key-stationary 单遍重算（flash 反向族 C2）**，与源算子同一族；操作数方向与数据落地
按 NPU 约束重构（§0.6 R1–R9）。迁移后的算法步骤：

1. **任务映射**：一维核网格，**每次 launch** 的 `grid = (total_kv // 96 + B) * slab`，
   `slab = min(32, H, 65535 // (total_kv//96 + B))`（编译期常量，host 侧 `plan_launch` 定档并断言
   `grid ≤ 65,535`）；共 `ceil(H/slab)` 次 launch，第 i 次携带 `h_off = i*slab`。块内 `tile = bid // slab`、
   `h = h_off + bid % slab`，`h ≥ H` 的块（尾 slab）立即返回；再用**融合式 running-prefix 扫描**解出
   `(request, n0)` 与 `live`（`live==0` 的块立即退出）。
2. **请求内几何**：`q_start/kv_start`、`len_q/len_kv`、`shift` 由 `cu_seqlens_*` 现算。
3. **key 常驻**：`k (96,192)`、`v (96,128)` 与其转置副本 `kT (192,96)` 从 GM 搬入 L1，跨整个 query 循环
   常驻；`dk_c/dv_c` 的 L0C 累加器由首迭代的 `clear_accum` 清零。
4. **query 块循环**（因果剪枝：`kq ∈ [max(0,(n0-shift)//64), max(ceil(len_q/64), loop_st+1))`）：
   a. 搬入 `q_s (64,192)`、`qT_s (192,64)`、`do_s (64,128)`、`doT_s (128,64)` 与 head-major `lse/delta` 的 64 元区段；
   b. `st_c = k_s·q_s^T`、`dpt_c = v_s·do_s^T`（显式 L0，每迭代 `clear_accum`）；
   c. `T.dual_copy` 把两块 L0C 按 key 行 2:1 落到两个 AIV 的半区 UB；向量侧算
      `p = exp(st*scale − lse)`（A2 运行时块级特化）→ 只写两个 fp16 tile `pT`、`dsT`；
   d. 两块 fp16 经 `T.dual_copy` 落 L1（`pT_s`、`dsT_s`，L1 侧仍是整块 96×64）；
   e. `dv_c += pT_s·doT_s^T`、`dk_c += dsT_s·qT_s^T`（跨迭代累加）、`dq_c = dsT_s·kT_s^T`（`transpose_A=True`，
      由 L1→L0A 转置搬运承担，每迭代清零）；
   f. `set_atomic("add","float32")` 后把 `dq_c` 整块落到 `dq_acc[q 行, h, :]` 并关闭原子模式。
5. **块尾**：`dk_c/dv_c` 按半区 `T.dual_copy` 落地 UB，向量侧 cast（`dk` 另乘 `scale`）后写回 GM——
   **内部半块整块写、尾半块逐 lane 守卫**（R7）。

### 1.5 数据流图

```
GM: cu_seqlens_q/kv(int32) ──标量读──▶ (request, n0, live, q_start, kv_start, len_q, len_kv, shift)
GM: h_off(int32[1]) ──标量读──▶ h = h_off[0] + bid % slab（每 block 一次，AIC/AIV 同式）
GM: q,k,v,do(f16)  lse,delta(fp32, head-major)  dq_acc(fp32, in-out)
        │  GM→L1 (MTE2, nd2nz / dn2nz；越界尾部由 OOB pass clamp+fill，按张量 extent)   │
        ▼                                                                            │
   L1: k_s,v_s,kT_s(常驻)  q_s,qT_s,do_s,doT_s(每迭代)                                │
        │  L1→L0 (MTE1，每次 GEMM 前一次 T.copy；dQ 的 A 走 transposing copy)          │
        ▼                                                                            │
   L0A/L0B ──mad──▶ L0C: st,dpt(每迭代) dk,dv(跨迭代) dq(每迭代)                       │
        │  T.dual_copy 2:1 (L0C→UB, fp32 同 dtype, 每 AIV 48 个 key 行)                 │
        ▼                                                                            │
      UB(fp32 半区): st_u, dp_u ──SimtVF(128) 半区 48×64──▶ UB(f16): p16_u, dsT_u     │
        │  T.dual_copy (UB→L1，InsertNd2Nz → NZ scatter)                                 │
        ▼                                                                            │
      L1: pT_s, dsT_s ──▶ dV 的 A / dK 的 A(直接) / dQ 的 A(transpose_A 复用)          │
                                                                                     │
   L0C(dq) ──set_atomic(add) + T.copy 整块──▶ GM: dq_acc ─────────────────────────────┘
   L0C(dk,dv) ──dual_copy→UB(半区)──cast(+scale)──┬─ 内部半块: T.copy → GM dk/dv
                                                  └─ 尾半块:  逐 lane 守卫写 GM dk/dv
```

`P/dP/dS` 全程留在 L0C/UB/L1，不物化到 GM；**`dS` 只有一份 L1 tile**（`dsT_s`），向量段只有 2 个输出 tile。

### 1.6 算法调研与优化分析

#### 1.6.0 算法调研（Algorithm Research）

> 调研对象：**「变长（packed）批量注意力反向，给定 lse/delta 的梯度」这一数学语义的算法族**。
> 源算法（key-stationary + `GroupTiling`）只是基线候选之一。
> 检索范围：平台 `knowledge/algorithm-candidates.md` 的 attention 族、`knowledge/` 下全部 950 条目
> （`PL-950-*`、`ALG-950-formulas.md`、`bottlenecks.md`、`constants.md`）、本仓 `examples/` 同类实现
> （`examples/mla_dense_bwd/_bwd_kernel/`（设计 + 真机实现 + 检视 + 复盘）、`examples/flash_attention/`、
> `examples/mixcv/`、`examples/paged_cache_write/`）、源仓 golden 与 manifest roofline、结构判据自查。
> 网络检索未使用（本地源已覆盖该族全部已知结构，见 R2 的结构依据）。
>
> 代表 workload = `ds-v3-mixed`（B=6，lens 512/1024/2048/4096/8192/16384，H=128，D_qk=192，D_v=128，bf16）；
> 对照 `ds-v3-8x4k`（B=8×4096）与 `kimi-k2-4x8k`（B=4×8192，H=64）。全部数字由
> `history_version/design_probe_cube_cost.py` 逐式产出（可复跑）。

**R1 等价化简公式候选**（基线在表内；正式论证与收益在 §1.6.1）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 纳入 |
|---|------|------------|-----------|---------|------|
| 0 | **基线（源 packed 算法）** | `P=exp2(s·scale·log2e − lse·log2e)`；5 个 GEMM（`dQ` 用 `transpose_A`）；`dq` 逐元素原子加；`GroupTiling` 块枚举 | — | — | ✅（基线） |
| 1 | 自然对数域 `exp` | `P=exp(s·scale − lse)` | 数学恒等（`exp(x)=exp2(x·log2e)`），差异只在舍入路径 | 每 score 少 1 次乘 | ✅（A1） |
| 2 | 块级 mask 特化（运行期谓词） | 全可见块省掉 `select`；packed 下谓词用 `len_kv/len_q/shift` | 「块内掩码恒真」的可判定充分条件 | 每 score 少 2 次比较 + 1 次选择；快路块 9,051/18,990（47.7%） | ✅（A2） |
| 3 | 行统计量 `lse·log2e` 逐行提升 | 先算 `lse2[m]=lse[m]·log2e` | 恒等 | 被 #1 完全吸收 | ❌（冗余） |
| 4 | `scale` 折进 `q`/`k` 装载 | `(q·scale)` 与 `(k·scale)` 参与 GEMM | 数学恒等 | 乘次翻倍：每迭代 `2×64×192 = 24,576` 次 **fp16** 乘（`q`、`k` 各一份）换掉 `6,144` 次 **fp32** 乘（每 score 一次 `s*scale`）+ 块尾 `96×192`；且多一次 fp16 舍入 | ❌（定量，非「需两份副本」） |
| 5 | `delta` 项拆成 rank-1 修正 | `dS = P∘dP − P∘δ` 分两项进 GEMM | 数学恒等 | 多一个 GEMM：`192·64·96 = 1,179,648` MAC/迭代，对每迭代 GEMM 总 MAC `5,120,000` 为 **+23.0%** | ❌ |
| 6 | `dV`、`dK` 合并为一个 GEMM | 拼 `A=[pT;dsT]`、`B=[doT;qT]` | **不等价**（会求和） | — | ❌（不等价） |
| 7 | 物化 `P/dS`（非 flash） | 一遍算 `s/P/dP/dS` 落 GM，再扫描做 3 个 GEMM | 恒等 | 中间缓冲 `23.0e9 scores × 2 B = 46.1 GB`/张量、3–4 遍扫描 | ❌（R3 容量淘汰） |
| 8 | query-stationary 变体 | `dQ` 常驻、`dK/dV` 原子累加 | 恒等 | 原子目标宽 `D_qk+D_v = 320` vs `192`：每迭代原子字节 `122,880` vs `49,152`（+150%），GM 总量 +278 GB | ❌（R3/R4 淘汰） |
| 9 | 两 kernel 拆分（dQ / dK+dV） | 各自单遍 | 恒等 | FLOPs **×2 = 77.2 T**，破坏三 launch 契约 | ❌ |
| 10 | flat token 轴 tile（忽略 request 边界） | 把 `T_q` 当连续 token 轴切 64 行块 | **不等价**：跨边界的 tile 只解出一个 group，边界 token 的 `dk/dv` 会漏写/错写 | — | ❌（不等价） |
| 11 | 按 request 补齐到 96 的均匀分块 | 每 request pad 到 96 的倍数再枚举 | 恒等 | 需 host 侧重建/拷贝 q/k/v/do 与 `dq_acc`（T 级额外往返），违反「packed 原样传入」 | ❌（契约） |
| 12 | 一个 request 一个 block、块内串行 key 块 | 并行度 = `B·H` | 恒等 | 并行度 768（mixed），最长 request 的 171 个 key 块串行 → 长尾 | ❌（并行度） |
| 13 | 块表改由 host 预计算 | host 侧 `cumsum(ceil(len/96))` 成 int32 张量 | 恒等 | 多一个 GM 张量 + 一次 host 同步（`cu_seqlens` 在 device 上，需 D2H），违反「kernel 自包含」 | ❌（成本/契约） |
| 14 | **单份 `dS` 双向复用**（`trans_A`） | 只产 `dsT (Nk,Mq)`；`dK` 用它作 A，`dQ` 用它作 `trans_A` 的 A | 恒等（`(AB^T)` 与 `(A^T)^T B^T`） | 消掉一份转置副本与向量侧 scatter（dense 实测 −3.48 µs/迭代） | ✅（§0.6 R3） |

**R2 在线算法**：**有，且基线即在线变体。** 结构依据：两个统计量 `lse`（前向 log-sum-exp）与
`delta = rowsum(o*do)` 都由外部提供，因此反向**不需要**前向 online softmax 的 running max / 重缩放机器；
每个 (key 块, query 块) 对只需重算局部 `s/dP` 并立刻消费（`P/dS` 只活片上），得到单遍扫描、O(tile) 中间态
的流式结构（源算子即如此，§0.4 #2）。非在线对照是 R1#7（物化，3–4 遍、46 GB/张量）。
**无更优在线结构**：本算子的输出里必有一组（`dQ` 或 `dK/dV`）要沿另一轴跨块归并，单遍结构下无法再压缩
（R1#6 的合并已证不等价，R1#10 的扁平枚举已证不等价，R1#9 的拆分需第二遍）。该负向断言的依据是
**数据依赖上的交换性/边界分析**，不是经验。
**packed 特有的在线结论**：变长只改变「每个 request 的扫描区间」，不改变单遍性；`live`/`decode` 是任务映射
（O(B) 标量），不引入第二遍数据扫描。

**R3 复杂度对比**（四口径；代表 workload `ds-v3-mixed`：可见 score 对 23.0e9，H=128，T=32,256，B=6）：

| 候选 | FLOPs | GM 访存 (Bytes) | 扫描遍数 | 中间缓冲峰值 / 片上预算 | 每 query 行 L1 填充 |
|------|-------|-----------------|---------|----------------------|-------------------|
| **C1 = v3（L1 路径 NT + 向量转置副本，Nk=64）** | 38.42 T | **746.8 GB**（q/do 填充 463.5 + k/v/kT 常驻 4.2 + `dq` 原子 276.4 + `dk/dv` 2.6） | 1 | L0C 163,840 B（62.5%）/ L1 172,032 B（32.8%）/ UB 90,624 B（35.7%） | 1280 B × **87.2**（8x4k：32.5） |
| **C2 = 本版（全 L0 + `trans_A` 复用，Nk=96）** | 38.61 T（+0.5% 尾块量化） | **503.1 GB**（q/do 填充 310.8 + 常驻 4.3 + `dq` 原子 185.3 + `dk/dv` 2.7） | 1 | L0C 221,184 B（84.4%）/ L1 204,800 B（39.1%）/ UB 129,536 B（51.0%）/ L0A 49,152 B（75%）/ L0B 61,440 B（93.75%） | 1280 B × **58.4**（8x4k：22.0） |
| C2′ = C2 但 `dQ` 另造转置副本（v3 的形态） | 38.61 T | 503.1 GB | 1 | 同 C2 + 4,096 B/迭代 UB | 同 C2，但向量段 **+3.48 µs/迭代**（dense 实测 `epi1`−`epi0`） |
| C3 = query-stationary | 38.61 T | ≈781 GB（原子项 `3.77e6 × 122,880 = 463 GB` 取代 185 GB；填充项同量级） | 1 | `dQ` 常驻 L0C 48 KiB | 同 C2 量级 |
| C4 = 两 kernel 拆分 | **77.2 T**（`P`、`dP` 各算两遍） | ≈503 GB（无原子） | 2 | 同量级 | 两倍 launch |
| C5 = 物化 `P/dS`（R1#7） | 38.61 T | ≥ 138 GB 额外往返（P/dP/dS 各 46.1 GB） | 3–4 | 46.1 GB/张量 ≫ 片上总量（UB+L1+L0C ≈ 1.0 MB）的 4×10⁴ 倍 | 不可行 |
| C6 = 纯 Vector（无 Cube） | 38.61 T | 同 C2 | 1 | 同量级 | 56 个 Vector 核 lane 级乘加，算力低 1–2 个数量级 |
| C7 = flat token tile（R1#10） | 38.61 T | 同 C2 | 1 | 同 | **不等价**（跨 request 边界漏写 `dk/dv`） |
| C8 = 每 request 一块（R1#12） | 38.61 T | 同 C2 | 1 | `dQ` 常驻 | 并行度 768 ≈ 27 波，最长块 171 次串行迭代 |

**口径说明（每个数字的公式与出处）**：

- **迭代数**：`Σ_b Σ_{kn} max(0, max(ceil(l_q/64), loop_st+1) − loop_st)`，`loop_st = max(0,(n0−shift)//64)`，
  `n0 = 96·kn`。产出：mixed **3,770,496**、8x4k **1,441,792**、kimi **1,420,032**（C1 同式为 5,623,296 /
  2,129,920 / 2,113,536）。C2 相对 C1 的迭代数 **−32.9%**，来源是 `Nk` 64→96 把每 query 行的填充次数
  从 `S_kv/(2·64)+1 = 21.1` 降到 `S_kv/(2·96)+1 = 14.1`。
- **FLOPs**：`iters × (6144 × 1664 + 64 × 256)`，`1664 = 2(3·D_qk + 2·D_v)` 是 manifest roofline 的 GEMM 项。
  mixed `3,770,496 × 10,240,000 = 38.61 T`；8x4k `14.76 T`；kimi `14.54 T`。相对 C1（38.42 T）只 +0.5%，
  差额来自 `⌈⌉` 分块量化的 score 槽位（8.694e9 → 23.166e9 槽位 vs 真实 23.03e9）。
- **GM 访存**：四项之和，逐项公式见下表。
- **每 query 行 L1 填充次数**：query 行 `m` 被 key 块 `n0 ≤ m+shift` 读，单请求（`l_q = l_kv = S`
  且因果）时平均 `S/(2·Nk)+1`；`lens` 不等的工作负载按 `Σ_b l_b² / (2·Nk·Σ_b l_b) + 1` 加权。
  每行字节固定 `q+qT+do+doT = 1280 B`（`2×(192+128)×2 B`）。代入：mixed 加权
  `358,917,376/(192·32,256)+1 = 58.9`（实测口径 58.4，来自 `iters·64/rows`），C1 同式为 `87.9`（实测 87.2）；
  8x4k（`S=4096`）为 22.3（实测 22.0）/ C1 为 33.0（实测 32.5）。这四条与 R3 表的填充总量逐项自洽
  （`iters × 82,432 B`）。

**GM 访存逐项（mixed，C2）**：

| 项 | 公式 | 值 |
|----|------|----|
| 每迭代填充 | `q 24,576 + qT 24,576 + do 16,384 + doT 16,384 + lse/delta 512` | 82,432 B |
| 填充总量 | `3,770,496 × 82,432` | **310.8 GB** |
| 常驻摊销 | `43,392 块 × 98,304 B`（`k_s+v_s+kT_s`） | 4.3 GB |
| `dq` 原子 | `3,770,496 × 64×192×4` | **185.3 GB** |
| `dk/dv` 写 | `43,392 块 × (96×192 + 96×128) × 2 B` | 2.7 GB |
| **合计** | | **503.1 GB**（C1 = 746.8 GB，**−32.6%**） |

**每迭代 L1/UB 端口分解**（`Nk=96, Mq=64`，一次 (key 块, query 块) 迭代；口径 = 该迭代的 L1/UB 端口流量，
含 GM→L1）：

| 项 | 组成（字节） | 合计 |
|----|------|------|
| GEMM 操作数 L1→L0（10 条显式 `T.copy`） | `k_s 36864 + q_s 24576 + v_s 24576 + do_s 16384 + pT_s 12288 + doT_s 16384 + dsT_s(dK) 12288 + qT_s 24576 + dsT_s(dQ) 12288 + kT_s 36864` | **217,088** |
| UB→L1 写（2 个 tile 半区） | `p16_u 6144 + dsT_u 6144` ×2（两个 AIV） | 24,576 |
| GM→L1 装载（每迭代） | 82,432 | 82,432 |
| GM→L1 常驻摊销 | `98,304 ÷ (iters/blocks)` = `98,304 ÷ 86.9` | 1,131 |
| **合计** | | **325,227**（≈318 KiB） |

对应 `10,240,000 FLOP/迭代` → **31.5 FLOP/字节**（C1 = `6,852,608 / 271,600 = 25.2`，**+25%**）。
全量 = `3,770,496 × 325,227 = 1.226 TB` L1/UB 端口流量（C1 = 1.527 TB，−19.7%）。

**R4 硬件亲和性评估**（逐候选；容量对照 §4.5/§4.6；常数取 dense 生产实测或本轮探针）：

| 候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|------|-------------|---------|------------|---------|------------|------|
| C0 基线 | 需寄存器累加器与 `transpose_A` 的 mma 分形；块枚举要 warp shuffle | — | — | 块表用 warp scan，NPU 无 warp | — | ❌ 不能直译（R1/R2/R6） |
| C1 = v3 | 5 个 NT GEMM 可跑，但**向量侧被迫做跨步转置**（占 63%） | L0C 62.5% / L1 32.8% / UB 35.7% | 64 全为 16 倍数 | 全静态 + 动态 grid | 单阶段、无重叠 | ❌ **实测 Vector 受限**（cube 8.3% / vec 73.5%） |
| **C2 = 本版** | 5 个 GEMM **全显式 L0**（平台已核验，含 `trans_A`）；逐元素落 AIV | L0C 84.4% / L1 39.1% / UB 51.0% / L0A 75% / L0B 93.75% | 96/64/192/128 全为 16 倍数；**`_HN=48` 是 16 的倍数**（nd2nz 硬约束） | 见 §5.6（动态 grid + slab 断言） | GEMM 与向量 epilogue 同 kernel 交替；L0 单版本 | ✅ **dense 真机 150.7 ms / cube 25.2%；本轮 packed 原型 6/6 PASS** |
| C2′ = C2 + 转置副本 | 同 C2 | 同 C2 + 4 KiB | 同 | 同 | 同 | ❌ 实测 +3.48 µs/迭代（+117%），见 §1.6.4 |
| C3 query-stationary | 同 C2 | `dQ` 常驻 | 同 | 同 | 同 | ❌ 原子放大项宽 320（>192），GM +55%，fp16 输出二次舍入 |
| C4 两 kernel 拆分 | 同 C2 | 同 | 同 | 两次 launch | — | ❌ FLOPs ×2，违反三 launch 契约 |
| C5 物化 | 纯 Cube 友好 | 46.1 GB/张量 ≫ L2 | — | — | — | ❌ 容量淘汰 |
| C6 纯 Vector | 矩阵乘落错单元 | 同 | — | — | — | ❌ 单元不匹配（量级差） |
| C7 flat token tile | 同 C2 | 同 | — | 跨 request 边界 | — | ❌ 不等价（R1#10） |
| C8 每 request 一块 | 同 C2 | 同 | 同 | 同 | 长尾 | ❌ 并行度 768 ≈ 27 波 |

**设计期 roofline（D-2；常数取 dense 生产实测 + 本轮探针，逐项给出处）**

> 口径：per-iteration 单价（单核）与全量（28 核聚合）互为校验 `全量 = 单价 × iters / 28`。
> **dense 生产实测的同一 tile（Nk=96/Mq=64）**：`2.97 µs/迭代`、`cube 25.2%`、`vec 45.7%`、
> `MTE1 0.668 µs`（299.7 GB/s/核）、`MTE2 118.5 GB/s/核`、**AIC 四管道忙时之和 2.74 µs ≈ 迭代 2.97 µs
> （单 AIC 内无重叠）**。出处：`examples/mla_dense_bwd/_bwd_kernel/evidence/msprof_pipe_capture1.csv`
> （22,016 块均值，两次独立采集同值）+ `repro/CONST-950-mla-bwd-cube-split.py`（T_op 150.637 ms）。

| 管道 | per-iteration | 出处 / 公式 |
|------|--------------|------------|
| AIC mad（5 GEMM） | **0.75** | dense 实测 cube busy = `25.2% × 2.97` |
| MTE1（L1→L0，10 条 copy） | **0.668** | dense 实测 299.7 GB/s/核 × `217,088 B` |
| MTE2（GM→L1 填充） | **0.696** | dense 实测 118.5 GB/s/核 × `82,432 B` |
| fixpipe + `dq` 原子 + 块尾 | **0.63** | 由四管道和 2.74 µs 反解（余项） |
| **AIC 串行合计** | **2.74** | dense 实测 |
| **迭代** | **2.97** | dense 生产实测（22,016 块） |
| AIV epilogue（与之并行） | ~1.23 | `6,144 score × 0.201 ns`（dense 探针 `epi0`） |

**投影（packed，按 C2 的迭代数）**：

| workload | iters | 单价 2.97 µs | 单价 3.19 µs（**本轮 packed 实测，保守上界**） | 相对基线 |
|---|---|---|---|---|
| ds-v3-mixed | 3,770,496 | **400.0 ms** | 429.5 ms | 1728.7 → **4.32× / 4.02×** |
| ds-v3-8x4k | 1,441,792 | **152.9 ms** | 164.3 ms | 490.6 → **3.21× / 2.99×** |
| kimi-k2-4x8k | 1,420,032 | **150.6 ms** | 161.8 ms | 470.5 → **3.12× / 2.91×** |

单价 3.19 µs 来自本轮 `design_probe_packed_cube_num.py --time`（`q=kv=[4096,4096], H=16`，45,056 次迭代，
3 次取最小 = 5.140 ms ⇒ 3.194 µs/迭代），**小 grid（<2 波）会含尾波与 L2 冷启动，故它是保守上界**——
dense 的同口径外推曾高估 15%（173–176 预测 vs 150.7 实测）。
**Cube 口径 MFU**：mixed `38.61 T ÷ (0.400 s × 358 TF/s) = 27.0%`（`CONST-950-gemm-mte2-burst` 报 bf16 Cube
356–359 TF/s，**条目 stale**，故与 dense 的 msprof `aic_cube_ratio = 25.2%` 交叉参照，取 **23–27%** 区间表述）。
基线是 cube 8.3%（msprof）。**本设计不承诺 90% MFU**：完全重叠下界是
`max(mad 0.80 + MTE1 0.67, MTE2 0.70, AIV 1.23) = 1.47 µs`（mixed 198 ms，cube 54%），
它需要跨迭代软件流水与更宽的向量段，属 Stage 4 的探索档，不作为验收承诺。

**调研结论**：选定**算法族 = key-stationary 单遍重算（flash 反向族 C2）**，并以**融合式 running-prefix
块解码**替代 `GroupTiling` 的 warp 扫描（R6）。关键依据：R3 中 C2 与基线 FLOPs 相同（+0.5% 量化）而
**GM 访存最低可达（503.1 vs 746.8 GB，−32.6%）**、每字节 FLOP 最高（31.5 vs 25.2），其余候选被淘汰
（C3 +55% GM、C4 2× FLOPs、C5 容量、C6 单元不匹配、C7 不等价、C8 并行度）；R4 中只有 C2 的 5 个乘法
能全部映射到平台**已核验的显式 L0 形态**（含 `trans_A`），且 `trans_A` 复用把 v3 的向量 scatter（占 63%）
整项消掉。相对基线的结构差异：**同一算法族，按 NPU 约束把 GEMM 全部落到显式 L0、累加器迁 L0C、
把「转置副本」从向量段移到 MTE1、块表 warp 扫描换 O(B) 标量扫描、`dk/dv` 写回分快路与尾块守卫**。
源算子 §0.4 的 15 项优化手段的意图承接见 §0.5 处置列（全部有承接或明确舍弃理由，无静默丢弃）。
下游约束：§1.4 与 §1.6.1–§1.6.3 的分析对象即 C2；§3.1 用 §1.6.1 的优化后公式。

#### 1.6.1 数学等价优化（公式级）

分析对象 = §1.6.0 选定的 C2。A1/A2 为 dense 已验证、本文以 **packed 运行期界**重验的公式优化；
A3 为 packed 整块原子写回所依赖的**不变式**（不是优化项，但等价性必须成立）；A4 为 v3 的
**任务映射等价重构**（grid 上界修复，§0.6 R8），其「等价性」是任务集双射 + 写集不交。

| # | 项 | 原式（基线） | 优化后公式 | 等价性论证 | 收益量化 |
|---|-----|-------------|-----------|-----------|---------|
| A1 | **自然对数域概率** | `P = exp2(s·scale·log2e − lse·log2e)` | `P = exp(s·scale − lse)` | `exp2(x·log2e) ≡ exp(x)` 恒等式；两式对同一实数输入只差舍入路径，fp64 参照下各自距真值 ≤1 个目标 dtype ulp，候选式不劣于基线式 | 每 score 少 1 次乘（`lse·log2e` 的逐元素重算）→ 每迭代 6,144 次向量乘；`exp` 与 `exp2` 同属 SFU 指令（`T.exp` 有核验条目），指令数不减但不引入未核验 API |
| A2 | **块级 mask 特化（packed 运行期界）** | 每个 (key 块, query 块) 都生成 `select(vis, p, 0)`，`vis=(gi<len_kv)&(gj<len_q)&(gi<=gj+shift)` | 运行期谓词 `(n0+96<=len_kv) ∧ (m0+64<=len_q) ∧ (n0+95<=m0+shift)` 为真时直接输出 `p`，否则同一 `select` | 谓词是「掩码在整块上恒真」的**可判定充分条件**：`n≤m+shift` 对块内最大 `n`（`n0+95`）与最小 `m`（`m0`）成立即全真；`n0+96<=len_kv`/`m0+64<=len_q` 排除填充行/列。故两式逐元素相同 | 快路块 9,051/18,990（47.7% 全方阵口径）；每 score 省 2 次整数比较 + 1 次选择 → 约 6,144×2 次向量操作/迭代 |
| A3 | **packed 溢出零行不变式** | `dq` 只写有效行（源用逐元素守卫 `if m0+i < len_q`） | `dq` 改**整块**原子写回，越出 `len_q` 的行贡献必须恰为 0 | 掩码臂使 `gj ≥ len_q` 的列 `P=0 → dS=0`；`dq` 第 `j` 行 = `Σ_i ds16[j,i]·k[i,:]`，`ds=0` 时 fp32 累加是精确 0；故对下一 request 的 `dq_acc` 行是「原子加精确 0」，数值不变（仅 `-0.0 → +0.0` 的零号翻转，数值等价） | 省掉逐元素原子（源为每元素一次全局原子）：每迭代 12,288 个原子操作 → 1 次 64×192 的 store-atomic DMA |
| A4 | **单次 launch 的 grid 分解**（任务映射级） | 一次 launch：`grid = tiles × heads`（实测 `grid > 65535` 时静默只跑 `grid % 65536` 块） | `ceil(heads/slab)` 次 launch，`grid = tiles × slab`、`h = h_off[i] + bid % slab`；尾 slab 的 `h ≥ heads` 块立即返回 | (i) slab 区间是 `[0,heads)` 的划分；(ii) `bid ↔ (tile,h)` 双射（A4 逐 bid 穷举）；(iii) `(token,head,dim)` 下 head 占行内连续 `D` 个元素 ⇒ 写集不交；(iv) head 维不参与归约 ⇒ 无 launch 间同步；(v) `h` 只进入 GM 中间轴与掩码；(vi) `tiles > 65535`（即使 slab=1）时 `plan_launch` 抛 `ValueError`，无静默路径 | 单次 launch 块数 44,672（8x4k）/ 43,776（mixed），远低于 65,535；截断导致的「静默少算」不再可能 |

**A2 的实现机制**：谓词依赖 `n0 = f(tile)`、`m0 = kq·64`、`len_q/len_kv/shift = f(cu_seqlens)`，全是**运行时量**，
故 A2 由 kernel 内的**运行时块级分支**实现（Python 层 `if` 生成 IfThenElse，两支各一套半区 epilogue，
编译成两个 SIMT VF + 运行时二选一；P7 设备码 4 个 `__simt_vf__`）。编译期唯一可静态判定的是「是否存在
尾块」，而 packed 下它由 `cu_seqlens` 决定 ⇒ 连这条也不是编译期信息，两支都必须生成。

**快路的安全性论证**（为什么全可见块可以省掉掩码而不产生 inf/NaN）：快路要求 `m0+64 ≤ len_q`
（块内无填充 query 行）与 `n0+95 ≤ m0+shift`（块内每个 key 对块内每个 query 都可见）——即该块**不可能**
包含「看不到任何 key」的行（`lse=-inf` 只出现在这类行上），故 `exp(s·scale − lse)` 的指数输入有限；
块外的 `lse=-inf` 行一律落在掩码臂并与掩码一起置 0（`T.if_then_else` 选择，不是乘法，故 `0·inf` 不出现）。

**已确认无需改动的项**（CSE / 降代换自查，写明结论以免被误判为遗漏）：
`ds = p*(dp − delta)` 已是最优形态（`p*dp − p*delta` 会多一次乘）；`p` 的 fp32 结果被 `p16`(cast) 与
`dsT`(cast) 共用（每元素只算 1 次 `exp`）；`lse/delta` 已是每 query 行一次装载、按 lane 私有读；
`dk` 的 `scale` 乘只发生在块尾（每块 `96×192` 次）；**packed 自查**：`q_start/kv_start/len_q/len_kv/shift`
每 block 只算一次，`cnt/req/n0/live` 每 block 只算一次，`own/kv_end` 每 block 只算一次。

**弃选项的定量依据**（对应 R1 #4/#5/#6/#10）：`scale` 折进 `q`/`k` 需每迭代 24,576 次 fp16 乘换 6,144 次
fp32 乘 + 块尾 18,432 次，乘次翻倍且多一次 fp16 舍入；`delta` 的 rank-1 拆分需额外一次同规模 GEMM
（+23.0% 计算量）；`dV/dK` 合并会求和，不等价；flat token tile 会让跨 request 边界的 tile 只解出一个 group
（源 docstring 明示该失败模式），不等价。

**等价性机器验证（D-1）**：脚本 `examples/mla_varlen_bwd/_bwd_kernel/verify_equiv.py`（torch CPU，fp64 参照；
A1 含 `s·scale−lse ∈ [-8e2, 8e2]` 的随机样本 + 角点值：精确抵消、上/下溢、次正规、`lse=-inf` 空行、`±700`
极值；A2 覆盖 5 组 packed 长度集合，含尾块与 `len_q>len_kv` / `len_q<len_kv` / 单 token 请求；
A3 覆盖 6 组请求几何的整块枚举；A4 覆盖 8 组几何的**逐 bid 任务映射穷举** + 上界/越界边界检查）。
**本版按 C2 的 tile 重跑**（`BLOCK_N=96, BLOCK_M=64`，2026-11-25，`ASCEND_RT_VISIBLE_DEVICES=""`，
toolchain `ascend-950pr-9579 / TileLang ascend 0.1.15`，cwd = 算子目录，
`/workspace/venv-ascend950/bin/python verify_equiv.py`），输出 `EQUIV_SUMMARY ALL_PASS`：

| 采纳项 | 样本数 | 候选 vs 基线 最大 ulp | 违反率 | 候选 vs fp64 最大 ulp | 基线 vs fp64 最大 ulp | 结论 |
|---|---|---|---|---|---|---|
| A1 exp 域 P（目标 dtype float16） | 1048592 | 1 | 0.000e+00 | 1 | 1 | EQUIV_PASS |
| A1 exp 域 P（目标 dtype bfloat16） | 1048592 | 1 | 0.000e+00 | 1 | 1 | EQUIV_PASS |
| A2 块级 mask 特化 · packed 运行时界（block_n=96, block_m=64） | 115696896 | 0 | 0.000e+00 | — | — | EQUIV_PASS |
| A3 packed 溢出零行不变式（dS/dq 越界行恒 0，`+0.0` 原子累加无副作用） | 38006 | 0 | 0.000e+00 | — | — | EQUIV_PASS |
| A4 单次 launch grid ≤ 65535 的 head-slab 分解（任务集一一对应、写集按 head 不交） | 475117 | 0 | 0.000e+00 | — | — | EQUIV_PASS |

（明细：A2 快路块 **9,051/18,990**；分组 `ds-v3-8x4k/4req 2604/5504 | ds-v3-mixed 4295/8986 |
kimi-k2-4x8k/2req 2150/4472 | tails+shift 2/23 | tiny 0/5`。A3 `spill_entries=31296 dq_rows=2608 acc=4102
zero_sign_flips=1`。A4 `tasks=475113 over_cap=0 edge_checks=2/2 raised_on_overflow=2/2`，逐几何：
`ds-v3-8x4k slab=32/4 launches/max_grid=11168/blocks=44672/idle=0 | ds-v3-mixed 32/4/10944/43776/0 |
kimi-k2-4x8k 32/2/11040/22080/0 | smoke 8/1/64/64/0 | H=126 32/4/11168/44672/useful=43974/idle=698 |
H=127 32/4/11168/44672/useful=44323/idle=349 | H=128 one-tile 32/4/1600/6400/0 |
auto-shrink(2100 tiles) 31/5/65348/326740/useful=269824/idle=56916`。）
违反率定义：与基线相差超过目标 dtype 1 个 ulp 的元素占比；A4 的「违反」= 缺失 + 重复 + 越界任务数 +
上界失败 + 越界未抛错。

**优化结论**：采纳 2 项公式优化（A1、A2）+ 1 项任务映射重构（A4），全部 EQUIV_PASS；A3 是 packed 整块
原子写回所依赖的不变式，同样 PASS（31,296 + 2,608 个溢出元素/行全为精确 0）。**v4 未新增公式级优化**：
本版的收益来自执行结构（§0.6 R2/R3/R6）与写回隔离（R7），不来自数学变形——源公式已无更多
CSE/降代换空间（见上「已确认无需改动的项」）。

**优化后公式**（供 §3.1 使用）：

$$
P[m,n]=\exp\!\big(\mathrm{scale}\cdot s[m,n]-\mathrm{lse}_m\big)\quad\text{all-visible block}\qquad
dS[m,n]=P[m,n]\cdot\big(dP[m,n]-\mathrm{delta}_m\big),\quad
\mathrm{dqAcc\_row}(m\ge len_q)\equiv 0
$$

$$
\text{launch } i:\quad \mathrm{grid}_i=tiles\cdot slab,\quad
tile=\Big\lfloor \frac{bid}{slab}\Big\rfloor,\quad
h=i\cdot slab+(bid \bmod slab),\quad slab=\min\big(32,\ H,\ \lfloor 65535/tiles\big\rfloor\big)
$$

对角块与尾块再乘因果/区间掩码（`T.if_then_else`，见 A2 的块级特化）。其余乘积与写回形态不变
（`dV = P^T dO`、`dK = scale·dS^T q`、`dq_acc += dS·k`）。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

盘点实现方案（C2 + packed 的 R6/R7/R8）中**全部**循环与标量计算点：

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由（附查证） |
|---|--------|-----------|-------------|---------|----------------------|
| 1 | `P = exp(s·scale − lse)` | GPU `T.Parallel(bn,bm)` 逐元素 | `T.SimtVF(threads=128)` + `T.Parallel(48,64)` + `T.exp` | ✅ | —（`T.exp` 核验：`PL-950-row-primitives`；P7 设备码 `__simt_vf__ __launch_bounds__(128)`） |
| 2 | 因果 / 尾块掩码 | 逐元素布尔 + `select` | 同一 `T.Parallel` 内的 `T.if_then_else`（A2 运行时块级特化掉全可见块） | ✅ | —（P7 设备码两个 epilogue VF 二选一） |
| 3 | `dS = P·(dP − delta)` + 行广播 | 逐元素 | 同一 `T.Parallel` 内读 `dlt_u[j]`（lane 私有读，无广播指令） | ✅ | —（SIMT 内列/行广播已核验：`PL-950-conversion-layout`） |
| 4 | `p16`/`dsT` 的 fp32→fp16 cast | 逐元素 | `T.cast` 写另一个半区 UB buffer | ✅ | —（转换含 ±inf/次正规已核验：同上） |
| 5 | **`dS` 的转置副本** | v3：向量段 `ds_u[j,i] = cast(dsv)` 跨步写（实测 3.75 µs/迭代 = 63%） | **取消该计算点**：`dQ` 改用 `trans_A=True` 的 L0 GEMM，转置由 `asc_copy_l12l0a_transpose`（MTE1）完成 | ✅（换单元 + 换层） | —（P7 设备码该指令 ×1；dense 探针 2 的 (T,T) 数值 PASS） |
| 6 | `dk` 写回 `cast(dk_c·scale)`、`dv` cast | 逐元素 | 块尾 `T.Parallel(48,192)` / `T.Parallel(48,128)` | ✅ | —（P7 设备码 `simt_vf_2`） |
| 7 | `P/dS` 的 K 维归约 | GPU tensor core | Cube mad（显式 L0，fp32 累加） | ✅（换单元，非标量循环） | —（P7 设备码 5 个 `asc_mmad`） |
| 8 | `dq_acc += dq` | 逐元素 `atomic_add` | `T.set_atomic("add","float32")` + 一次 L0C→GM `T.copy` | ✅（硬件原子，无标量循环） | —（P7 设备码 `asc_set_atomic_add_float` + `asc_copy_l0c2gm` + `asc_set_atomic_none`） |
| 9 | **变长块枚举**（前缀和 + decode） | 源用 warp shuffle 前缀扫描 + 二分 | 无（保留标量 O(B) 扫描） | ❌ | **block 级控制流与寻址**：该量决定「本块是否存活、所有 GM 地址、循环界」，逐元素 API 不适用；且结果必须对 AIC 与两个 AIV 一致，向量化到某 AIV 的 UB 需要未认证的跨核可见性。另：源的 `_tile_prefix_sum` 依赖 `KernelLaunchFrame…get_thread_extents()` + `T.shfl_up` 的 warp 扫描，而 ascend 的 `T.Kernel` 在 kernel 作用域**不建立线程域**，该形态不能按源样使用。量化：每块 ≤B 次 int32 读 + ≤3B 次整数运算（manifest 的 B ≤ 8 时约 25 个标量 op），且只在该块的**标量头部执行一次**（不在迭代内）；最坏情形（`len_kv ≤ 96` 的单迭代块）也只有数十个标量 op，对照该块至少 20 个 tile 级 op。实测单价（§1.6.4 C）与 dense 同量级，未出现解码开销 |
| 10 | **逐 request 边界标量**（`q_start/kv_start/len_q/len_kv/shift`、`loop_st/loop_ed`、`own/kv_end`） | 源逐元素守卫里的同一批标量 | 无（保留标量） | ❌ | **block 索引**：每块算一次，参与 GM 地址与循环界，逐元素替代会退化成每元素重算（更多工作）。边界：`loop_ed` 取 `max(ceil(len_q/64), loop_st+1)` 以覆盖 `len_q=0` 的退化请求（§5.4 第 6 条） |
| 11 | **`dk/dv` 写回的双分支谓词**（`own+48 <= kv_end`） | 源为逐元素守卫 `if n0+i < len_kv` | 谓词本身保留为块级标量；**两条支路内部都已向量化**（快路 DMA、慢路逐 lane 守卫的并行 store） | ✅（谓词为块级，工作已向量化） | —（P7 设备码 `asc_copy_ub2gm_align` 与 `simt_vf_3` 两条路径） |
| 12 | **尾半块的逐 lane 守卫 store** | 源逐元素守卫 | `T.SimtVF` 内 `T.Parallel(48,192)` + `if own+i < kv_end` 的 lane 谓词 | ✅（向量化执行，非标量循环） | —（P7 设备码 `simt_vf_3` 的逐 lane 谓词；同形态先例 `examples/paged_cache_write`） |
| 13 | `lse/delta` 行统计装载（head-major） | 逐元素 | `T.copy(lse[h, qoff:qoff+64], lse_u)`（1×64 fp32 = 256 B） | ✅ | —（P7 设备码 `asc_copy_gm2ub_align(..., 1, 256, ...)`） |
| 14 | 尾块边界（有效行/列数与 clamp） | 源 `T.if_then_else` 逐元素守卫 + `T.min` 区段 | **装载侧无需替代**：越界尾部由平台 `AscendInsertOOBPadding` 改写成 `min()` clamp + `asc_fill_l1` 补 0；写回侧由 R7 双分支收口 | ✅（平台 pass + 双分支消除） | —（P7 设备码 `min(96, max((total_kv - kv_start) - n0, 0))` + `asc_fill_l1` ×7） |
| 15 | **host 侧 head-slab launch 循环** | 源为一次二维 grid 发射 | 无（保留为 host 侧编排） | ❌ | **不是设备计算**：只做 launch 编排与 `h_off` 取址，无逐元素工作；把它「向量化」等价于「一次发射全部 heads」，而那正是越界的根因。每算子调用 ≤ `ceil(H/32)` = 4 次（manifest），每次 host 侧 O(1) |
| 16 | **`h = h_off[0] + bid % slab` 的 head 解码** | 源 `h = %ctaid.y`（硬件给出） | 无（保留为块级标量） | ❌ | **block 索引**：每 block 算一次（2 个标量运算 + 1 次 4 B `h_off` GM 读），决定全部 GM region 的中间轴与掩码的 key 索引，且必须对 AIC 与两个 AIV 一致（P7 设备码在 AIC/AIV 各生成一次同一表达式，无需跨核通信） |
| 17 | **尾 slab 守卫 `if h < heads`** | 源无（二维 grid 天然不越界） | 无（保留为块级控制流） | ❌ | **block 级控制流**：谓词为块常量，块内全体线程同向；两条支路的工作量分别是「0」与「整个 body」，逐元素谓词化会把空块变成 `48×192 = 9,216` 次逐元素比较（P7 设备码 `if (h < 3) { … }` 包住整个 body） |

**向量化结论**：17 个计算点中 **12 项**已向量化 / 换单元 / 由平台 pass 消除（#1–#8、#11–#14），保留 **4 项**
设备侧标量点（#9 变长块枚举、#10 逐 request 边界标量、#11 的双分支块级谓词、#16 head 解码、#17 尾 slab 守卫
中的块级谓词部分）+ **1 项 host 侧编排**（#15），理由类别 = **block 级控制流与寻址 / host 元数据与 launch
编排**（每 block 一次或每次 launch 一次，非逐元素热点；#9 另有「kernel 级无线程域 / 跨核一致性」硬约束）。
相对 v3 的关键改进：**#5 整项取消**（v3 是唯一被实测标定为 63% 的向量热点），向量段每 score 的输出 tile
由 3 个降为 2 个（UB 写 14 B → 12 B/score），且不再有任何跨步 scatter。

#### 1.6.3 向量化轴与数据布局决策 ⭐

> 迁移任务：源算子的轴选择（GPU 线程/warp 映射）只是输入，下面是独立评估。

本算子的核内布局自由度集中在 **(key × query) 的 2-D 中间块**（`P`、`dP`、`dS` 各 96×64）与
**block ↔ (tile, head) 的展平方式**。I/O 契约固定为 THD（尾轴 `D` 连续），`D` 维是矩阵维（Cube 契约），
因此「向量化轴」只能在 key/query 两轴中选。

| 评分项 | 候选 1：lane = query，半区 = key（本设计） | 候选 2：lane = key | 候选 3：UB 内双转置 |
|--------|--------------------------------------|-------------------|------------------|
| 整除性 / 尾 lane | `Mq=64`，fp32 向量宽 4 → 16 向量/行，无尾 lane | `Nk=96` → 24 向量/行（fp32），行数非 2 幂 | 同候选 1 |
| 累加链形态 | 最内轴 stride=1（`(Nk,Mq)` 行内连续），`lse/delta` 为 lane 私有读 | 最内轴是 key，`lse/delta` 需沿 key 轴**列广播** | 同候选 1 |
| 输出 tile 数 | **2 个**（`p16`、`dsT`，全部合并写） | 2 个 + 1 个转置 | 3 个（多一份替身） |
| repack 代价 | **0 次跨步写**；`dQ` 的转置由 MTE1 承担（12,288 B/迭代） | 需给 `dV`/`dK` 各造转置副本（2 份跨步写） | +6,144 B/迭代 UB |
| 实测代价（dense，同 tile） | epilogue **0.201 ns/score**（合并写，0.825 µs/4096） | **1.050 ns/score**（`epi1`，4.302 µs） | **0.603 ns/score**（`epi2`，2.469 µs） |
| 弃选量化理由 | ✅ 采纳 | ❌ 跨步写 **5.2×** 于合并写 | ❌ 两趟 **3.0×** 于合并写 |
| 候选 4：纯 SIMT 直写 GM | AIC 不参与 | — | ❌ 5 个 96×64×192 级矩阵乘落 56 个 Vector 核，算力差 1–2 个量级（R4 C6） |
| 候选 5：中间块经 GM 往返 | — | — | ❌ 每迭代多 3×12 KiB 的 GM 往返，全量约 +103 GB HBM 流量 |
| 候选 6：block 展平 `h` 外层（dense 的写法） | `h = bid // ntiles_dyn` | — | ❌ **动态除数**（`ntiles` 含 `total_kv`）：运行期整数除法未取证；且相邻 bid 跨 head 导致同行不同 head 的切片不连号，L2 行复用变差 |
| 候选 7：block 展平 `h` 内层 + **head slab**（本设计） | `tile = bid // slab`（静态除数）、`h = h_off[i] + bid % slab` | — | ✅ 采纳（静态除数 + L2 复用 + 单次 grid ≤ 65,535） |
| 候选 8：每次 launch 用**运行期 head 数**做除数 | `h = h_off[0] + bid % active_heads` | — | ❌ 动态除数（同上）+ 尾 slab 的 grid 需随 `active_heads` 变化，等于把上界重新交给运行期；收益仅省 ≤1.6% 空块 |

**多 AIV 的执行体映射**：本平台把线程域限定在 `T.SimtVF` 内；显式 `T.Vector(vector=2) as sid` 在本工具链
**不可编译且运行值不确定**（`InternalError … dup sid`，`TRAP-950-aiv-sid-vector-frame`）。本设计沿用 dense
真机验证的形态：半区向量段（`T.Parallel(48,64)`）+ `sid = T.call_extern("int64","asc_get_sub_block_id")`
在 kernel body 取一次并被 `T.SimtVF` 捕获，L0C↔UB 与 UB→L1 全部 `T.dual_copy` 2:1，两个 AIV 各持 48 个
key 行（`_HN=48` 同时满足「2:1 可整除」与「每 AIV 行数 % 16 == 0」两条硬约束）。

**布局决策结论**：选定候选 1 + 候选 7——核内布局为 **(key, query)** 且**向量化轴 = query**（lane 映射到
`Mq=64` 的最内轴），中间块以 `(Nk,Mq)` 整块存在 L0C（按 key 行 2:1 分给两个 AIV 的 UB）、`dS` **只有一份**
`(Nk,Mq)` 的 L1 tile 由 `dQ` 通过 `trans_A` 消费；block 展平取 `tile = bid // slab, h = h_off[i] + bid % slab`；
**统计量保持 head-major `[H,T_q]`**（`lse/delta` 的一次 256 B 连续搬运，不转置成 dense 的 `[B,H,S_q]`）。
该结论已同步到 §3.3 伪代码（`T.Parallel(48, 64)` 的内层为 query 维）、§4.5（半区 UB 预算）、§5.6（launch 计划）
与 §6（循环结构）。

**主选 / 备选与实验裁决计划**：

- **主选（进 Stage 3）**：`block_n=96, block_m=64`。L0C 221,184 B（84.4%）、L1 204,800 B（39.1%，
  实测口径 229,376 B = 43.8%）、UB 129,536 B（51.0%）、L0A 49,152 B（75%）、L0B 61,440 B（93.75%）；
  每迭代端口 325,227 B、31.5 FLOP/字节。
- **备选（唯一一个）**：`block_n=128, block_m=32`。L0C = `dk 98304 + dv 65536 + st 16384 + dpt 16384 +
  dq 24576 = 221,184 B`（84.4%，与主选相同）；L1 = `k_s 49152 + v_s 32768 + kT_s 49152 + q_s 12288 + qT_s 12288
  + do_s 8192 + doT_s 8192 + pT_s 8192 + dsT_s 8192 = 204,800 B`；UB 半区 `_HN=64` ⇒ `st_u/dp_u 2×16384 +
  p16_u/dsT_u 2×8192 + 512 = 49,664` + 块尾 `tail_u 49152 + tailv_u 32768 + out_k 24576 + out_v 16384 =
  122,880` ⇒ 约 172,544 B（67.9%）；每迭代端口 ≈ 236 KiB、36.5 FLOP/字节（更省 MTE2），但 GEMM 的 N 维
  （query）降到 32、`dq_c` 的 M 也降到 32，mad 效率下降，且块尾缓冲逼近 UB 上界。
  **裁决阈值**：备选 latency ≤ 主选 × 0.90 才翻转；按 shape 分片经 `lru_cache` 工厂分派。
- **`block_n` 的可行集（硬约束，非偏好）**：UB→L1 的 `dual_copy` 经 `InsertNd2Nz` 改写为 `ascend_nd2nz_scatter`，
  要求**每 AIV 行数 % 16 == 0** ⇒ `block_n/2 % 16 == 0` ⇒ `block_n ∈ {32, 64, 96, 128, …}`。
  dense 的复盘实测 `block_n=80`（半区 40 行）**不可编译**（`nd2nz_copy.h:59 static_assert ROWS % 16 == 0`），
  故 v3 文档里的 `Nk=80` 回退与 dense 的 `Nk=112`（半区 56 行）**都不可用**，本版把可行阶梯收敛为 64/96/128。
- **实验裁决计划**：代表 shape = smoke（640 tokens）与 `ds-v3-8x4k`、`ds-v3-mixed`；指标 = kernel
  `Task Duration(us)` + msprof 的 `aic_cube_ratio` / `aiv_vec_ratio` / `aic_mte1_active_bw` /
  `aic_mte2_active_bw`；实测数据回填本节与 §1.6.4；Stage 4 的 A/B 清单 = ①`Nk/Mq`（96/64、128/32）
  ②epilogue 优化阶梯（`float4` 加宽 / `simd.vexpdif` / 手工展开）③`trans_B=False` 消 `qT/doT`
  ④跨迭代软件流水。

#### 1.6.4 本版实测证据（packed + Cube）⭐

> 三个可复现脚本（本轮，全部在算子目录内运行，`TMPDIR=/workspace`，
> `/workspace/venv-ascend950/bin/python`，toolchain `ascend-950pr-9579 / TileLang ascend 0.1.15`）：
> `history_version/design_probe_packed_cube.py`（编译探针，无设备）、
> `history_version/design_probe_packed_cube_num.py`（**设备数值原型** + `--time` 单价）、
> `history_version/design_probe_cube_cost.py`（纯算术代价模型）。

**A. 编译探针（P7–P9）**

| 探针 | 结构 | 实测结果 |
|------|------|---------|
| **P7 `packed_cube_stages2`** | packed 全套：动态 `total_q/total_kv` + **动态 grid `(total_kv//96 + B) * SLAB`** + `tile=bid//SLAB`/`h=h_off[0]+bid%SLAB`/`if h<H` + 融合 decode + `live` + head-major `lse/delta` + **5 个显式 L0 GEMM（含 `trans_A`）** + 半区 `dual_copy` + store-atomic + R7 双分支，`num_stages=2` | **COMPILE_OK**，源 30,162 B。marks：`asc_mmad`×5、`asc_copy_l12l0a`×5、`asc_copy_l12l0b`×5、**`asc_copy_l12l0a_transpose`×1**、`asc_copy_l0c2ub`×4、`ascend_nd2nz_scatter`×2、`asc_copy_l0c2gm`×1、`asc_set_atomic_add_float`×2、`__mix__`×1、`asc_get_sub_block_id`×2、`asc_fill_l1`×7、`asc_copy_gm2l1_nd2nz`×4、`asc_copy_gm2l1_dn2nz`×3、`asc_copy_ub2gm`×2 |
| **P8 `packed_cube_stages1`** | P7 但 `num_stages=1` | COMPILE_OK，30,430 B，marks 同 P7 ⇒ 阶段数与核内形态无关（回归用） |
| **P9 `packed_cube_no_slab`** | P7 但 `h = bid % H`（无 slab，单次 launch） | COMPILE_OK，28,830 B，marks 同 P7 ⇒ slab 化只改 head 的取值来源（与 v3 P6 的结论一致） |

P7 设备码关键行（`/tmp/opencode/probe_P7_device.asc`）：

- `int32_t h = (h_off[0] + (((int32_t)block_idx) & 1));`（AIC 与 AIV 各一份）、`if (h < 3) { … }` 包住整个 body；
- `for (int32_t g = 0; g < 3; ++g) { ntg = ((cu_kv[g+1]+95) - cu_kv[g]) / 96; if (cnt <= tile && tile < cnt+ntg) {…} cnt += ntg; }`；
- `for (int32_t kq = max(0, (((n0_dyn + lq) - lkv) >> 6)); kq < max(((lq + 63) >> 6), (max(0, …) + 1)); ++kq)`；
- 装载 clamp `min(96, max(((total_kv - kv_start) - n0_dyn), 0))`、`dq` clamp `min(64, max(((total_q - q_start) - (kq * 64)), 0))`；
- 4 个 SIMT VF：`simt_vf_0/1`（两套 epilogue 臂）、`simt_vf_2`（块尾 cast）、`simt_vf_3`（尾半块逐 lane 守卫，
  内部条件为 `sid*48 + kv_start + n0_dyn + i < kv_start + lkv`）。

**B. 设备数值原型（6/6 PASS）**

脚本 `history_version/design_probe_packed_cube_num.py`：与 P7 同构的 kernel（实际执行）+ 按 §0.1 语义独立写的
fp32 逐 request 参考（bottom-right 因果、`lse` 由 `logsumexp` 现算、`delta = Σ P·dP`），逐 request 切片比对
（不只比全张量，以免掩盖跨 request 泄漏）：

| 用例 | lens_q / lens_kv | H | dtype | dk | dv | dq | 逐 request 最差 | 结论 |
|---|---|---|---|---|---|---|---|---|
| exact 2 key blocks | [192,192] / [192,192] | 2 | bf16 | 3.50e-05 | 4.34e-04 | 7.97e-06 | 4.34e-04 | **PASS** |
| ragged + tail blocks | [128,320] / [128,320] | 2 | bf16 | 2.67e-04 | 3.63e-04 | 1.03e-05 | 4.96e-04 | **PASS** |
| **len_q > len_kv（shift<0，`lse=-inf` 空行）** | [320,128] / [128,320] | 2 | bf16 | 8.40e-05 | 0.00e+00 | 6.22e-06 | 1.31e-03 | **PASS** |
| **len_q < len_kv（shift>0）** | [128,192] / [320,384] | 2 | bf16 | 2.49e-03 | 2.18e-03 | 2.03e-04 | 2.65e-03 | **PASS** |
| single key block | [96,96] / [96,96] | 2 | fp16 | 2.27e-04 | 2.42e-04 | 2.39e-05 | 2.61e-04 | **PASS** |
| **head slab 3/2（尾 slab 空块）** | [192,320] / [192,320] | 3 | bf16 | 4.11e-04 | 1.89e-03 | 2.14e-05 | 1.89e-03 | **PASS** |
| slab=1（3 次 launch） | [192,320] / [192,320] | 3 | bf16 | 4.11e-04 | 1.89e-03 | 2.14e-05 | 1.89e-03 | **PASS** |

（误差 = `max|diff| / max|ref|`；判据 3e-2，实际最差 2.65e-03，**低 1 个量级**。）
四点结论：(i) **packed 语义在 Cube 核上数值成立**（含尾块、`shift` 双向、`lse=-inf` 空行）；
(ii) **R7 的写回隔离有效**——逐 request 切片误差与全张量同量级，说明尾块没有把 0 写进下一个 request；
(iii) **head slab 与 `h_off` 正确**（`H=3, slab=2` 的尾 slab 空块不影响结果，`slab=1` 与 `slab=H` 数值同值）；
(iv) **`trans_A` 复用 `dsT` 正确**（`dq` 误差 2.4e-05，是全表最小项——若 `b_hi` 的重灌顺序错，`dq` 会全错）。

**C. 单价（`--time`，`q=kv=[4096,4096], H=16`，45,056 次迭代，3 次取最小）**

| shape | iters | T | 单价（单核） |
|---|---|---|---|
| `[4096,4096]`, H=16 | 45,056 | **5.140 ms** | **3.194 µs/迭代** |
| `[512,4096]`, H=16 | 22,960 | 3.483 ms | 4.247 µs/迭代 |

**口径**：该 shape 只有 1,376 个块（≈49 波），比 dense 的 `S=2048/H=8`（176 块）更接近生产，但仍含尾波与
L2 冷启动 ⇒ 3.19 µs 是**保守上界**（dense 的同口径外推高估了 15%）。第二个 shape 的 4.25 µs 说明
**短 request 的块（迭代数少）单价更高**（每块固定开销摊薄不足），这解释了 mixed 的投影为什么要带上界。

**D. 与生产实测的交叉校验（dense，同 tile 同结构）**

`ds-v3-b2-8k` 实测 **150.7 ms / 2.97 µs 每迭代 / cube 25.2% / vec 45.7% / MTE1 0.668 µs（299.7 GB/s/核）/
MTE2 118.5 GB/s/核**，且「AIC 四管道忙时之和 2.74 µs ≈ 迭代 2.97 µs」⇒ **单 AIC 内无重叠**。
本 kernel 的每迭代工作（5 GEMM + 2 epilogue + 2 段 UB→L1 + 1 次原子 store）与 dense 逐项相同，
差异只在块级标量（decode）与块尾写回的慢路：前者每块 ≤ 3B 次整数运算，后者每 request 最多 2 次
（≤ `B·H·2` 个块）⇒ 对单价的影响落在 10⁻² 量级，**不改变 2.97–3.19 µs 的估计**。

**E. 结构层的两条硬约束（沿用 dense 实测，本版再次满足）**

1. **L0 路径与 L1 路径不能在同一 kernel 内混用**：dense 实测「`dQ` 单独改 L0、其余 4 个留 L1」时
   `dk/dv` 出现 4.8e-1 / 3.1e-2 的错误（`ascend_gemm_l1` 模板内部会用 L0A/L0B，与用户显式分配的 L0 缓冲
   互相覆盖）；5 个全改显式 L0 后同一测试全 PASS。本设计 5 个全 L0（P7 设备码 5 个 `asc_mmad`）。
2. **`T.copy(UB, L1, transpose=True)` 不可编译** ⇒ 「把转置挪到 UB→L1」这条路被排除，可用形态只有
   `trans_A` 的 L1→L0A 转置（`asc_copy_l12l0a_transpose`）。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式：混合（Cube + Vector），且 Cube 侧走显式的 L0 路径（Expert 形态）。**
（平台 `programming_modes.choices` 的 `hybrid`；`migration_default = simd`，偏离理由见 2.2。）

### 2.2 选型理由

- 算子含 5 个矩阵乘（每迭代 `2·96·64·(3·192+2·128) = 11.9 MFLOP`），**必须落 Cube**；若走纯向量路径，
  56 个 Vector 核的 lane 级乘加比 Cube 分形低 1–2 个数量级（§1.6.0 R4 C6）。
- 同时每迭代必须做 `exp`、掩码、cast 等逐元素工作（96×64 = 6,144 个 score），**必须落 Vector（AIV）**；
  平台把线程域限定在 `T.SimtVF` 内（`references/programming.md`「API 与执行区域」）。
- **为什么 Cube 侧用显式 L0 而不是 L1 操作数路径**（本版的关键依据）：
  ① 只有 L0 路径支持 `transpose_A=True`（L1 路径断言只允许 NT），而 `trans_A` 是消掉 v3 的向量 scatter
     （实测 63%）的唯一低成本手段（§0.6 R3）；
  ② L0 路径是平台**已核验**的 GEMM 形态（`PL-950-gemm-tiles`），dense 真机 L0 13/13 + 生产 150.7 ms；
     本轮 P7 在 packed 形态下复现（5 `asc_mmad` + 1 `l12l0a_transpose`）；
  ③ 两条路径**不能混用**（§1.6.4 E1 实测），故必须整体二选一。
- 两侧必须在**同一个 kernel** 内交替（中间块 `P/dS` 不落 GM），平台支持该形态：`T.Kernel` 单核网格 +
  自动调度把 op 分派到 AIC/AIV（`AssignCore`/`ResolveCore`/`InsertSync`）；P7 设备码 `__mix__(1, 2)`。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | `T.alloc_shared`（UB，向量侧半区）、`T.alloc_l1`（GM 装载与 GEMM 操作数暂存）、**`T.alloc_l0a/l0b`（显式操作数）**、`T.alloc_l0c`（fp32 累加器） |
| 计算方式 | Cube：`T.gemm(a_l0a_region, b_l0b_region, c_l0c, transpose_A=…, transpose_B=…, clear_accum=…)`，A/B/C 用 allocation 全 extent 或 0 原点的静态切片；Vector：`with T.SimtVF(threads=128): for i, j in T.Parallel(48, 64)` |
| 执行体 | **不写 `T.Vector`**。半区向量段由 `T.SimtVF(threads=128)` 表达；副块 id 由 `sid = T.call_extern("int64","asc_get_sub_block_id")` 在 body 取一次并被 VF 捕获 |
| 同步 | 自动（`allow_autoschedule=True`，AutoSchedule + InsertSync）→ §7；**跨 head slab 的 launch 之间不需要同步**（写集不交，§5.6 d） |
| 跨单元数据 | L0C→UB 用 `T.dual_copy` 2:1（半区）；UB→L1 用 `T.dual_copy`（`InsertNd2Nz` 改写成 NZ scatter）；**L1→L0 用 `T.copy`（`trans_A` 时自动换成转置搬运）**；L0C→GM 用 `T.copy` + store-atomic |
| 边界 | 装载靠平台 OOB clamp（按张量 extent）+ 掩码收口；写回靠 **R7 双分支**（按 request 边界）；**grid 靠 head slab + 启动前断言**（R8，§5.6） |
| launch 参数 | prim_func 多一个 `h_off: T.Tensor((1,), "int32")`（本次 launch 的 head 原点）。它是**内部 launch 参数**，不属于算子 I/O 契约（§4.1）；一个编译产物服务全部 slab |

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入为 §1.6.1 优化后公式；每个步骤与 §1.6.2 的向量化结论一致。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 0 | （launch 计划）`slab, h_off[i], grid_i = tiles·slab` | host 侧 `plan_launch`（§5.6）：定 slab、断言 `grid ≤ 65,535`、生成 `ceil(H/slab)` 次 launch |
| 0′ | `tile = bid // slab`、`h = h_off[i] + bid % slab`（`h ≥ H` 立即返回） | R1/R8 的块解码；`tile_upper_bound` 作动态上界，`live` 丢弃越界 tile |
| 0″ | `(req, n0, live, q_start, kv_start, len_q, len_kv, shift)` | 融合式 O(B) 扫描（R6），`live==0` 早退 |
| 1 | `s^T = k_s · q_s^T` | L0 GEMM，A=`k_s (96,192)`、B=`q_s (64,192)`，C=`st_c (96,64)` fp32，(F,T) |
| 2 | `dP^T = v_s · do_s^T` | L0 GEMM，A=`v_s (96,128)`、B=`do_s (64,128)`，C=`dpt_c (96,64)` |
| 3 | `p^T = exp(s^T·scale − lse[j]) ⊙ mask` | 向量（lane=query，半区 48×64），A2 运行时块级特化 |
| 4 | `pT = f16(p^T)`；`dsT = f16(p^T ⊙ (dP^T − delta[j]))` | 向量 + cast；`p^T` 只算一次 `exp`（CSE）；**只 2 个输出 tile** |
| 5 | `dV = pT · doT^T` | L0 GEMM，A=`pT_s (96,64)`、B=`doT_s (128,64)`，C=`dv_c (96,128)` 累加 |
| 6 | `dK = scale · dsT · qT^T` | L0 GEMM，A=`dsT_s (96,64)`、B=`qT_s (192,64)`，C=`dk_c (96,192)` 累加（`scale` 在块尾乘） |
| 7 | `dQ = dsT^T · kT^T`（`transpose_A`） | L0 GEMM，A=`dsT_s (96,64)` **(T)**、B=`kT_s (192,96)`，C=`dq_c (64,192)` 每迭代清零 |
| 8 | `dq_acc[q 行] += dQ` | store-atomic + 整块 `T.copy`；越界行为精确 0（A3） |
| 9 | `dk[kv 行] = cast(dk_c·scale)`、`dv[kv 行] = cast(dv_c)` | 半区 cast + **R7 双分支写回** |

（步骤 6 与 7 共用同一份 `dsT_s`——这就是源算子 `transpose_A=True` 的复用点。）

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API（`tilelang.ascend.language`） | 参数 | 模式 |
|------|----------|--------------------------------------------|------|------|
| 0 | 任务映射 | `T.Kernel(grid)` 一维核网格 | `grid = (total_kv // 96 + batch) * slab`（`total_kv` 为 `T.dynamic`，`slab` 编译期常量，≤65,535 由 host 断言）；`T.Kernel` 不接受 `threads` | — |
| 0a | head 原点 | prim_func 形参 `h_off: T.Tensor((1,), "int32")`；块内 `h = h_off[0] + bid % slab` | 一元素 int32 张量，逐 launch 传 `i*slab`；GM 标量读（P7 设备码 `h_off[0]`） | 标量 |
| 0b | 尾 slab 守卫 | `if h < heads:` 包住整个 body | `heads` 编译期常量；空块只做 1 次 4 B `h_off` 读 + 2 个标量运算即返回（P7 设备码 `if (h < 3)`） | 块级控制流 |
| 0′ | 块枚举 | `T.serial(batch)` + `T.alloc_local([1],"int32")` | `_decode` 宏（§3.3）；`live` 谓词 | 标量 |
| 1 | `k/v/kT` 常驻 | `T.copy(GM region, l1_buf)`（`kT` 加 `transpose=True`）；dst 用全 allocation extent | `k[koff:koff+96, h, :] → k_s`；`kT_s` 走 dn2nz | MTE2 |
| 2 | `q/do` 双方向装载 | 同上（两份：ND 与 `transpose=True`） | `q_s (64,192)`、`qT_s (192,64)`，偏移含 `q_start` | MTE2 |
| 3 | `lse/delta`（head-major） | `T.copy(lse[h, qoff:qoff+64], lse_u)` | 一维 fp32 区段（256 B） | MTE2 |
| 4 | L1→L0 暂存 | `T.copy(l1_buf_region, l0a_or_l0b_region)`（`trans_A=True` 的 A 生成转置搬运） | 每次 GEMM 前一次；region 原点必须为 0 | MTE1 |
| 5 | 5 个 GEMM | `T.gemm(a_l0a, b_l0b, c_l0c, transpose_A=…, transpose_B=…, clear_accum=…)` | 见 §3.1；A/B 在 L0A/L0B，C 在 L0C | Cube |
| 6 | L0C→UB（半区） | `T.dual_copy(st_c, st_u)` / `T.dual_copy(dpt_c, dp_u)` | 2:1 M 分片（48×64 每 AIV），fp32 同 dtype | AIC fixpipe |
| 7 | `exp/mask/dS/cast` | `with T.SimtVF(threads=128): for i, j in T.Parallel(48, 64)` | `T.exp`、`T.if_then_else`、`T.cast`；A2 为运行时块级 `if` 两支；`sid` 捕获 | Vector |
| 8 | UB→L1（半区） | `T.dual_copy(p16_u, pT_s)`、`T.dual_copy(dsT_u, dsT_s)` | 2:1 M 分片；由 `InsertNd2Nz` 改写为 NZ scatter | MTE |
| 9 | `dq_acc` 累加 | `T.set_atomic("add","float32")` → `T.copy(dq_c, dq_acc[qoff:qoff+64, h, :])` → `T.set_atomic_none()` | L0C fp32 → GM fp32；尾部由 OOB clamp 裁到 `total_q`，跨 request 的行是精确 0（A3） | AIC fixpipe |
| 10 | `dk/dv` 写回 | `T.dual_copy(dk_c, tail_u)` → 向量 `cast`（`dk` 乘 `scale`）→ **R7 双分支**：`T.copy(out_k, dk[own:own+48, h, :])` / `T.Parallel` + `if own+i < kv_end` | 半区；按 request 边界收口 | AIC + Vector |
| 11 | 多缓冲 | `T.Pipelined(…, num_stages=2)` | 与 dense 一致；**实测只有 `qT_s` 真正拿到 2 版本**（Z3 回退，§6.3），不承诺重叠收益 | — |

### 3.3 计算伪代码

> 以下结构**已在设备上跑通并与 fp32 参考一致**：`history_version/design_probe_packed_cube_num.py`
> （6/6 PASS，含尾块、`shift` 双向、`lse=-inf` 空行、head slab）。编译形态见
> `history_version/design_probe_packed_cube.py`（P7–P9）。

```python
BLOCK_N, BLOCK_M = 96, 64        # keys / queries
HN = BLOCK_N // 2                # 48：每个 AIV 的 key 行（且 48 % 16 == 0）
D_QK, D_V = 192, 128
HEAD_SLAB = 32                   # 编译期 head slab（v3 成果，保留）
NTILE = "total_kv // BLOCK_N + batch"    # 运行期上界（tile_upper_bound）
GRID = NTILE * HEAD_SLAB         # 单次 launch；host 保证 <= 65535 并断言
NUM_STAGES = 2

@T.prim_func
def _bwd_main(q, k, v, dout, lse, delta, cu_q, cu_kv, h_off, dq_acc, dk, dv):
    #                                                 ^^^^^ int32[1]：本次 launch 的 head 原点
    with T.Kernel(GRID) as bid:
        tile = bid // HEAD_SLAB                # 静态除数
        h    = h_off[0] + bid % HEAD_SLAB      # 运行期原点 + 静态槽位
        sid  = T.call_extern("int64", "asc_get_sub_block_id")
        <L1 / L0A / L0B / L0C / UB 半区 buffer 分配>       # 预算见 §4.5 / §4.6

        if h < heads:                          # 尾 slab 守卫
          # ---- R6：融合式 (前缀和 + decode)，O(batch) 串行，local 标量 ----
          req = 0; n0 = 0; cnt = 0
          for g in T.serial(batch):
              nt = (cu_kv[g + 1] - cu_kv[g] + BLOCK_N - 1) // BLOCK_N
              if (tile >= cnt) & (tile < cnt + nt):
                  req = g; n0 = (tile - cnt) * BLOCK_N
              cnt = cnt + nt
          live = tile < cnt
          if live:
            q_start = cu_q[req];  kv_start = cu_kv[req]
            len_q = cu_q[req+1] - q_start;  len_kv = cu_kv[req+1] - kv_start
            shift = len_kv - len_q;  koff = kv_start + n0

            # ---- key 常驻（尾部由 OOB pass 按张量 extent clamp + 补 0）----
            T.copy(k[koff : koff + BLOCK_N, h, :], k_s)
            T.copy(v[koff : koff + BLOCK_N, h, :], v_s)
            T.copy(k[koff : koff + BLOCK_N, h, :], kT_s, transpose=True)

            loop_st = T.max(0, (n0 - shift) // BLOCK_M) if is_causal else 0
            loop_ed = T.max(ceildiv(len_q, BLOCK_M), loop_st + 1)   # 覆盖 len_q == 0
            for kq in T.Pipelined(loop_st, loop_ed, num_stages=NUM_STAGES):
                m0 = kq * BLOCK_M;  qoff = q_start + m0
                T.copy(q[qoff : qoff + BLOCK_M, h, :], q_s)
                T.copy(dout[qoff : qoff + BLOCK_M, h, :], do_s)
                T.copy(q[qoff : qoff + BLOCK_M, h, :], qT_s, transpose=True)
                T.copy(dout[qoff : qoff + BLOCK_M, h, :], doT_s, transpose=True)
                T.copy(lse[h, qoff : qoff + BLOCK_M], lse_u)      # head-major
                T.copy(delta[h, qoff : qoff + BLOCK_M], dlt_u)

                # ---- 2 个 L0 GEMM（每迭代清零）----
                T.copy(k_s, a_k); T.copy(q_s, b_lo)
                T.gemm(a_k, b_lo, st_c, transpose_B=True, clear_accum=True)
                T.copy(v_s, a_k[0:BLOCK_N, 0:D_V]); T.copy(do_s, b_lo[0:BLOCK_M, 0:D_V])
                T.gemm(a_k[0:BLOCK_N, 0:D_V], b_lo[0:BLOCK_M, 0:D_V], dpt_c,
                       transpose_B=True, clear_accum=True)
                T.dual_copy(st_c, st_u); T.dual_copy(dpt_c, dp_u)   # 每 AIV 48 行

                fast = (n0 + BLOCK_N <= len_kv) & (m0 + BLOCK_M <= len_q) \
                       & (n0 + BLOCK_N - 1 <= m0 + shift)            # 运行期谓词
                if fast:                                             # A2 快路
                    with T.SimtVF(threads=128):
                        for i, j in T.Parallel(HN, BLOCK_M):         # 内层 = query（lane）
                            p = T.exp(st_u[i, j] * scale - lse_u[j])
                            p16_u[i, j] = T.cast(p, dt)
                            dsT_u[i, j] = T.cast(p * (dp_u[i, j] - dlt_u[j]), dt)
                else:                                                # 掩码臂
                    with T.SimtVF(threads=128):
                        for i, j in T.Parallel(HN, BLOCK_M):
                            gi = n0 + sid * HN + i;  gj = m0 + j
                            vis = (gi < len_kv) & (gj < len_q)
                            if is_causal:
                                vis = vis & (gi <= gj + shift)
                            p = T.exp(st_u[i, j] * scale - lse_u[j])
                            pv = T.if_then_else(vis, p, 0.0)
                            p16_u[i, j] = T.cast(pv, dt)
                            dsT_u[i, j] = T.cast(pv * (dp_u[i, j] - dlt_u[j]), dt)
                T.dual_copy(p16_u, pT_s); T.dual_copy(dsT_u, dsT_s)

                # ---- 3 个 L0 GEMM：dV/dK 累加，dQ 用 trans_A 复用 dsT ----
                T.copy(pT_s, a_k[0:BLOCK_N, 0:BLOCK_M])
                T.copy(doT_s, b_hi[0:D_V, 0:BLOCK_M])
                T.gemm(a_k[0:BLOCK_N, 0:BLOCK_M], b_hi[0:D_V, 0:BLOCK_M], dv_c,
                       transpose_B=True, clear_accum=(kq == loop_st))
                T.copy(dsT_s, a_k[0:BLOCK_N, 0:BLOCK_M])
                T.copy(qT_s, b_hi[0:D_QK, 0:BLOCK_M])
                T.gemm(a_k[0:BLOCK_N, 0:BLOCK_M], b_hi[0:D_QK, 0:BLOCK_M], dk_c,
                       transpose_B=True, clear_accum=(kq == loop_st))
                T.copy(dsT_s, a_t)            # L1 -> L0A 转置搬运（MTE1）
                T.copy(kT_s, b_hi)            # ★ kT 必须在 doT/qT 之后最后重灌
                T.gemm(a_t, b_hi, dq_c, transpose_A=True, transpose_B=True, clear_accum=True)

                # ---- dQ 整块原子累加（溢出行为精确 0，A3）----
                T.set_atomic("add", "float32")
                T.copy(dq_c, dq_acc[qoff : qoff + BLOCK_M, h, :])
                T.set_atomic_none()

            # ---- 块尾：dk = cast(dk_c*scale)、dv = cast(dv_c) ----
            T.dual_copy(dk_c, tail_u); T.dual_copy(dv_c, tailv_u)
            with T.SimtVF(threads=128):
                for i, d in T.Parallel(HN, D_QK):
                    out_k[i, d] = T.cast(tail_u[i, d] * scale, dt)
                for i, d in T.Parallel(HN, D_V):
                    out_v[i, d] = T.cast(tailv_u[i, d], dt)
            own = koff + sid * HN;  kv_end = kv_start + len_kv
            if own + HN <= kv_end:                # R7 快路：整块 DMA
                T.copy(out_k, dk[own : own + HN, h, :])
                T.copy(out_v, dv[own : own + HN, h, :])
            else:                                 # R7 慢路：逐 lane 守卫（每 request ≤ 2 次）
                with T.SimtVF(threads=128):
                    for i, d in T.Parallel(HN, D_QK):
                        if own + i < kv_end: dk[own + i, h, d] = out_k[i, d]
                    for i, d in T.Parallel(HN, D_V):
                        if own + i < kv_end: dv[own + i, h, d] = out_v[i, d]
```

注（逐条可复核）：

1. **GEMM 的 M/N/K 全静态**：L0 路径从 **allocation 形状**取 `node_M/node_N/node_K`，region 可以是
   0 原点的静态切片（`a_k[0:96,0:128]`），故 L0 缓冲按「最大形状分配 + 切片使用」复用；region 原点必须为 0，
   因此 `kT_s` 若需拆半必须用两块独立 L1 缓冲（本设计不拆）。符号 extent（`x[:an, :]`）会直接 `ValueError`。
2. **尾块由「全 extent 装载 + 掩码 + 平台 OOB clamp」三件套处理**：被补 0 的行/列再由掩码臂的
   `gi < len_kv` / `gj < len_q` 置零（补 0 只让 `s=0`，而 `exp(0·scale − lse) ≠ 0`，故掩码必须保留这两项）。
   **注意 packed 的装载 clamp 只到 `total_kv`**（不是本 request 的末尾），越界读到的下一个 request 的真实行
   由掩码丢弃，不产生数值影响。
3. **`is_causal` 的两处编译期守卫**（`loop_st` 与掩码臂的因果项）：`is_causal=False` 时 `loop_st` 必须为 0、
   掩码不得含 `gi <= gj + shift`。源码同构写法见 `bwd.py:202, 230-243`。
4. **`T.clear` 不支持 L0C**（v3 探针 P4 实测）→ 累加器初值只能由 `clear_accum=(kq == loop_st)` 表达，
   并用 `loop_ed = max(ceil(len_q/64), loop_st+1)` 保证循环非空（否则 `len_q=0` 的退化请求会留下未初始化的 L0C）。
5. **无 `T.Vector`/`sid` 帧**：只取一次执行体 id 做 key 行偏移。
6. **`b_hi` 被 `doT/qT/kT` 复用**：**`kT` 必须在 `doT/qT` 之后重灌**（dense 原型首版曾因顺序错误得到 dQ
   全错；本轮数值原型的 `dq` 误差 2.4e-05 说明顺序正确）。
7. **`num_stages=2` 与版本数**：dense 实测 `annotate_buffer_versions` 在本结构上不生效（Z3 `unsat` 回退到
   topological sort），只有 `qT_s` 真正拿到 2 版本（L1 实占 229,376 B）。故 §4.6 同时给出「单版本 204,800 B」
   与「实测 229,376 B」两个口径，**不承诺多版本的重叠收益**。
8. **head slab 与核内计算体的隔离**：`h_off`/`slab` 只出现在 `tile`/`h` 的取值处；5 个 GEMM、掩码、原子写回、
   块尾双分支与 v3/P7 逐字相同（P9 的 `no_slab` 变体 marks 与 P7 完全一致，是结构性证明）。

### 3.4 API 可行性确认

| API | 来源 / 核验状态 |
|-----|----------------|
| `T.Kernel(grid)`（一维、无 `threads`、**grid 可为动态表达式 × 编译期常量**） | `ascend/language/kernel.py`；动态 grid 见 `examples/paged_cache_write`（设备验证）；本轮 P7/P8/P9 编译级确认（设备码 kernel 实参含 `total_kv/total_q`，`tile = block_idx >> 1`） |
| 一元素 int32 张量作运行期 launch 标量（`h_off`） | 与 `cu_seqlens` 的 GM 标量读同性质；P7 设备码 `int32_t h = (h_off[0] + ((int32_t)block_idx & 1))`，AIC 与 AIV 各一份 |
| `T.dynamic("total_q"/"total_kv")` | `tilelang/language/symbolics.py`；`examples/paged_cache_write`、`examples/rope` 既有用法；P7 编译级确认 |
| `T.alloc_l1 / alloc_l0a / alloc_l0b / alloc_l0c / alloc_shared` | `ascend/language/allocate.py`；`PL-950-gemm-tiles`（已核验的双版本 + 显式 L0 形态）；P7 设备码 5 个 `asc_mmad` + 每迭代 10 条 `asc_copy_l12l0a/l12l0b` |
| `T.gemm(a_l0a, b_l0b, c_l0c, transpose_A=…, transpose_B=…, clear_accum=…)` | `gemm_op.py:_dialect_gemm_call` → `GemmMAD._lower_l0`；四类 trans 的布局推断见 `gemm_mad.py:126-134`；dense 探针 2 实测 (T,T)/(F,F)/(F,T) 三组合编译 + 数值 PASS；本轮 P7 含 (T,T) |
| `T.copy` L1→L0（含 `trans_A` 的转置搬运） | P7 设备码 `asc_copy_l12l0a`×5 / `asc_copy_l12l0b`×5 / **`asc_copy_l12l0a_transpose`×1**；设备数值原型 `dq` 正确 |
| `T.copy` GM→L1（含 `transpose=True` dn2nz） | `copy_op.py` + `PL-950-gemm-tiles`；P7 设备码 `asc_copy_gm2l1_nd2nz`×4 / `dn2nz`×3、`asc_fill_l1`×7 |
| `T.copy` GM→UB（head-major 1-D 区段） | P7 设备码 `asc_copy_gm2ub_align(..., 1, 256, ...)` |
| `T.dual_copy` L0C→UB（2:1 M 分片） | `copy_op.py:dual_copy`（硬件双目标）+ dense 真机（整块只落一个 AIV）；P7 设备码 `asc_copy_l0c2ub`×4 |
| `T.dual_copy` UB→L1（2:1 M 分片） | 同上（`InsertNd2Nz` → `ascend_nd2nz_scatter`）；P7 设备码 ×2；**要求每 AIV 行数 % 16 == 0**（`_HN=48` ✓） |
| GM 直接逐元素写（尾块守卫路） | `T.SimtVF` 内 `T.Parallel` + `if` 的 lane 谓词存储；`examples/paged_cache_write` 的同类形态；P7 设备码 `simt_vf_3` |
| `T.set_atomic("add","float32")` + L0C→GM `T.copy` | `mode.py:set_atomic` + `PL-950-atomic-splitk`；P7 设备码 `asc_set_atomic_add_float` → `asc_copy_l0c2gm` → `asc_set_atomic_none` |
| `T.SimtVF(threads=128)` + `T.Parallel` + `T.exp` + `T.if_then_else` + `T.cast` | `PL-950-row-primitives`、`PL-950-attention`、`PL-950-conversion-layout`；dense 真机；P7 设备码 `__simt_vf__ __launch_bounds__(128)` + `asc_vf_call` |
| `T.call_extern("int64","asc_get_sub_block_id")` | dense 真机 + P7 设备码（`asc_get_sub_block_id`×2 + `__mix__(1,2)`） |
| `T.serial(batch)` + `T.alloc_local([1],"int32")`（block 级标量） | `examples/moe_permute_align` 既有用法；P7 设备码 `for (int32_t g = 0; g < 3; ++g)` |
| `T.clear` 作用于 **L0C** | **不可用**：v3 探针 P4 实测 `InternalError: Unsupported scope shared.l0c` → 用 `clear_accum=(kq == loop_st)` |
| `T.Vector(vector=2) as sid` | **不可用**：`InternalError … dup sid` + 运行值不确定（`TRAP-950-aiv-sid-vector-frame`）。本设计不使用 |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束（来源） | 是否涉及 | 处理方案 |
|--------------|---------|----------|
| `T.Kernel` 只接受一维核网格、无 `threads` | 是 | 一维 `grid = (total_kv//96 + B) * slab`，块内解码（R1/R6） |
| L1 GEMM 路径只支持 NT，且**不能与 L0 路径混用** | 是 | 5 个乘法**全部**显式 L0（含 `dQ` 的 `trans_A`），不混用（§1.6.4 E1） |
| L0-GEMM 的 M/N/K 取自 allocation 形状，region 原点必须为 0 | 是 | A/B/C 一律「最大形状分配 + 0 原点静态切片」；符号 extent 与偏移原点都不使用 |
| L0C 是唯一可累加层，且必须带 Ascend fractal layout | 是 | 5 个 L0C 累加器共 221,184 B（84.4%，< 240 KiB 红线） |
| **`T.clear` 不支持 L0C**（P4 实测） | 是 | 首迭代清零由 `clear_accum=(kq == loop_st)` 表达 + `loop_ed` 保证循环非空 |
| L0C→UB 不支持 dtype 转换（同 dtype 断言） | 是 | L0C→UB 保持 fp32，cast 在 UB 内用 `T.cast` |
| **整块 L0C→UB 只落到一个 AIV**（dense 真机 P1：99.2% 错位） | 是 | 所有 L0C↔UB 交叉用 `T.dual_copy` 2:1（`_HN=48`），与 `sid` 索引一致 |
| **UB→L1 `dual_copy` 要求每 AIV 行数 % 16 == 0**（dense 复盘：`block_n=80` 不可编译） | 是 | `block_n/2 = 48`（✓）；可行集 `block_n ∈ {32,64,96,128,…}`（§1.6.3） |
| 线程域只在 `T.SimtVF` 内 | 是 | 所有逐元素工作包在 `T.SimtVF(threads=128)` |
| 显式 `T.Vector(vector=2)` 不可用 | 是 | 不使用；`sid` 用 `T.call_extern` 取 |
| `dual_copy` 要求恰有一维 2:1，且 dst 不能是带单例中轴的 GM region | 是 | `dual_copy` 只用于 L0C↔UB 与 UB→L1；GM 写回用 `T.copy`（快路）或逐 lane 存储（慢路） |
| 原子存储支持 dtype 集合 float32/float16/bfloat16/int8/16/32 | 是 | `dq_acc` 为 fp32，在支持集内 |
| 未认证项：手工跨核 flag、循环内复杂同步、跨 block barrier | 是 | 不写手工 flag，全部依赖 AutoSchedule + InsertSync；不引入跨 block barrier（§7） |
| **单次 launch 的 `T.Kernel(grid)` 块数上限 65,535**（`grid % 65536 != 0` 时**静默只执行**余数个块） | 是（`tiles` 是运行期量） | R8：head slab 多次 launch + host 侧 `plan_launch` 启动前断言（越界抛 `ValueError`）；A4 机器验证 |
| 动态 shape | 是 | `T_q/T_kv` 为 `T.dynamic`；`B/H/D_qk/D_v/block_n/block_m/head_slab` 编译期常量；`len_q/len_kv/shift` 全为运行期量并只进标量/谓词/copy 偏移 |

#### 3.5.2 参考实现差异说明

| 差异项 | 参考实现（GPU TileLang） | 本项目（TileLang ascend） | 转换方案 |
|--------|------------------------|--------------------------|---------|
| Kernel 结构 | `T.Kernel(num_tiles, heads, threads=256)` 二维网格 | 一维核网格 + AIC/AIV 自动分派 + 动态 grid + **head slab 多次 launch** | §0.6 R1/R8 |
| 任务枚举 | `GroupTiling`：warp shuffle 前缀和 + 二分 + `live` | 融合式 O(B) 标量扫描 + `live` | §0.6 R6 |
| 矩阵 API | `T.gemm(fragment)`，C 在寄存器，支持 `transpose_A` | `T.gemm(A_l0a, B_l0b, C_l0c)` 显式 L0，四类 trans 齐全 | §0.6 R2/R3/R6 |
| 累加器 | `T.alloc_fragment` | `T.alloc_l0c` + 半区 UB 中转 | §0.6 R2 |
| 统计量 layout | packed：`(heads, total)`；dense：`(B,H,S)` | 保持 packed 的 `(heads, total_q)` | §0.8 B② |
| 原子累加 | `T.atomic_add` 逐元素（含行守卫） | `set_atomic` + 整块 store（溢出零行 A3） | §0.6 R5 |
| `dk/dv` 写回 | 逐元素守卫 `if n0+i < len_kv` | 双分支（内部整块 / 尾半块逐 lane） | §0.6 R7 |
| 指数 | `T.exp2`（log2 域） | `T.exp`（自然域） | §1.6.1 A1 |
| 掩码 | 逐元素 `if_then_else` | 同 + 运行期块级特化（界为 `len_q/len_kv`） | §1.6.1 A2 |
| 流水 | `T.Pipelined(num_stages=2)`（cp.async） | `num_stages=2`（多版本实际只对 `qT_s` 生效） | §6.3 |
| 尾块 | `T.if_then_else` 逐元素守卫 + 区段截断 | 装载：平台 clamp + 掩码；写回：双分支 | §5.4 |

#### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/mla_dense_bwd/_bwd_kernel/_bwd_kernel.py` | **极高**（同一源码工厂的 dense 分支，**真机通过 + 生产 150.7 ms**） | 全文可复用：5 个显式 L0 GEMM、`trans_A` 复用、半区 `dual_copy`、`sid` 捕获、store-atomic、块尾 cast、`kT` 重灌顺序、golden 与 L0 套件写法 |
| `examples/mla_dense_bwd/_bwd_kernel/DESIGN.md` | **极高**（v3，已通过检视） | 章节组织、四态处置、编译探针范式、预算表口径、管道级 roofline 口径 |
| `examples/mla_dense_bwd/_bwd_kernel/RETROSPECTIVE.md` | 高 | 生产常数（2.97 µs/迭代、cube 25.2%、MTE1 299.7 GB/s/核）、两条硬约束（半区 %16、L0/L1 不混用）、外推高估 15% 的口径提醒 |
| `examples/TileOPs/tileops/kernels/attention/mla_varlen_bwd/_mla_varlen_bwd_kernels.py` | 高（本函数的 GPU 原样抽取） | packed 分支语义（`GroupTiling`、head-major 统计量、逐元素守卫） |
| `examples/paged_cache_write/_paged_cache_write_kernel/DESIGN.md` | 中高（同平台**动态 extent + 动态 grid** 先例） | `T.dynamic` + 动态 1-D grid、尾块 lane 谓词、GM 直写形态 |
| `.agents/platforms/toolchains/ascend/knowledge/repro/PL-950-gemm-tiles.py` | 高（GM→L1→L0→GEMM、转置装载、动态 region） | `transpose=True` 的 dn2nz、显式 L0 缓冲、`clear_accum` |
| `knowledge/repro/PL-950-mixed-epilogue.py` | 高（Cube GEMM + Vector + `dual_copy` 同 kernel） | 混合模式写法、L0C→UB 落地、同 dtype 约束 |
| `knowledge/repro/PL-950-atomic-splitk.py` | 中（store-atomic 累加） | `set_atomic` + `T.copy(C)` 的配对与 host 预清零契约 |
| `knowledge/repro/PL-950-conversion-layout.py` | 中（SIMT 转换 + 列广播） | `T.cast` 的 ±inf/次正规行为 |
| `knowledge/kernel_template.py` | 中（kernel 骨架） | 一维 `T.Kernel`、`T.SimtVF` 包逐元素、静态尾块分支 |
| `history_version/design_probe_packed_cube.py` / `_num.py` / `design_probe_cube_cost.py` | —（本设计的探针与原型） | §3.5.4 / §1.6.4 的全部证据，可复跑 |

#### 3.5.4 编译探针与设备原型证据

**运行方式（按平台 `execution` 上下文，工作目录 = 任务目录；`prototype_command` 的解释器本机无
torch/tilelang，故沿用已登记的工具链 venv，与 `runtime.toolchain_stamp` 一致；另**不能在仓根运行**
——仓内 dev-root `tilelang/` 会遮蔽 site-packages 的 0.1.15 TVM）**：

```bash
cd examples/mla_varlen_bwd/_bwd_kernel
source ../../../.agents/platforms/toolchains/ascend/env-950pr.sh
# 编译探针（无设备）
TMPDIR=/workspace ASCEND_RT_VISIBLE_DEVICES="" \
  /workspace/venv-ascend950/bin/python history_version/design_probe_packed_cube.py
# 设备数值原型 + 单价
TMPDIR=/workspace /workspace/venv-ascend950/bin/python \
  history_version/design_probe_packed_cube_num.py          # 6 个用例
TMPDIR=/workspace /workspace/venv-ascend950/bin/python \
  history_version/design_probe_packed_cube_num.py --time   # 单价
# 等价性机器验证
TMPDIR=/workspace ASCEND_RT_VISIBLE_DEVICES="" \
  /workspace/venv-ascend950/bin/python verify_equiv.py      # ALL_PASS
# 代价模型（纯算术）
/workspace/venv-ascend950/bin/python history_version/design_probe_cube_cost.py
```

实测输出（2026-11-25，`ascend-950pr-9579` / TileLang ascend 0.1.15）：编译探针 **3/3 COMPILE_OK**
（P7/P8/P9，marks 见 §1.6.4 A）；设备原型 **6/6 PASS**（§1.6.4 B）；`verify_equiv` **ALL_PASS**（§1.6.1）。

**反例（不接受的设计形态，均本机实测）**：

| 形态 | 实测错误（原文） |
|------|-----------------|
| `T.clear` 作用于 `T.alloc_l0c` 缓冲 | `InternalError: Unsupported scope shared.l0c`（v3 P4 / P2b） |
| `T.gemm` 用符号 region extent（`x[:an, :]`） | `ValueError: T.gemm requires static tile dimensions, but M is symbolic: an` |
| `with T.Vector(vector=2) as sid:` 与 `dual_copy`/GEMM 混用 | `InternalError … Need input to be in SSA form dup sid` |
| `block_n=80`（半区 40 行）的 UB→L1 `dual_copy` | `nd2nz_copy.h:59 static_assert ROWS % 16 == 0`（dense `repro/TRAP-950-nd2nz-rows-multiple16.py`） |
| 在 `@T.prim_func` 体内调用普通 Python 函数承载循环 | `TypeError: 'ForFrame' object is not iterable`（计算体必须用 `@T.macro`） |
| `T.copy(UB, L1, transpose=True)` | `Ascend device compilation failed`（dense §1.6.4 E2） |

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
| `h_off`（**内部 launch 参数**） | `(1,)` | `int32` | 本次 launch 的 head 原点 `i*slab`（§5.6）；由 wrapper 构造，**不属于算子 I/O 契约**（`MLAVarlenBwdOp` 的输入仍是上表 8 项）；一个编译产物服务全部 slab |

（源算子签名另含 `o`：由同一次算子调用的 `_delta_kernel` launch 消费（`bwd.py:341`），**本 kernel 不读**。）

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `dk` | `(T_kv, H, 192)` | 输入 dtype | 与 `k` 同 shape；已乘 `scale` |
| `dv` | `(T_kv, H, 128)` | 同上 | 与 `v` 同 shape；**不乘** `scale` |
| `dq_acc` | `(T_q, H, 192)` | `float32` | **in-out**，原子累加；未乘 `scale`（由 `_dq_land_kernel` 处理） |

三个输出与输入同 layout（THD、尾轴连续），**无需任何转置或 permute**；`dk/dv` 与 `k/v` 逐维相同，
`dq_acc` 与 `q` 同 shape（仅 dtype 为 fp32），故 host 侧只做 `zeros`（`dq_acc`）。

### 4.3 中间缓冲区

| Buffer | Shape | dtype | 层级 | 用途 |
|--------|-------|-------|------|------|
| k_s | (96,192) | 输入 dtype | L1 | score GEMM 的 A，跨 query 循环常驻 |
| kT_s | (192,96) | 输入 dtype | L1 | dQ GEMM 的 B（dn2nz 装载，常驻） |
| v_s | (96,128) | 输入 dtype | L1 | dP GEMM 的 A |
| q_s | (64,192) | 输入 dtype | L1 | score GEMM 的 B |
| qT_s | (192,64) | 输入 dtype | L1 | dK GEMM 的 B |
| do_s | (64,128) | 输入 dtype | L1 | dP GEMM 的 B |
| doT_s | (128,64) | 输入 dtype | L1 | dV GEMM 的 B |
| pT_s | (96,64) | 输入 dtype | L1 | dV GEMM 的 A（`P^T`，两 AIV 各半写出） |
| dsT_s | (96,64) | 输入 dtype | L1 | dK 的 A（K-major）**兼** dQ 的 A（经 `trans_A` 由 MTE1 转置） |
| a_k | (96,192) | 输入 dtype | **L0A** | K-major 操作数：k / v（切片 96×128）/ pT（切片 96×64）/ dsT（切片 96×64） |
| a_t | (96,64) | 输入 dtype | **L0A** | MN-major 操作数：dsT（dQ 的 `trans_A`） |
| b_lo | (64,192) | 输入 dtype | **L0B** | q / do（切片 64×128） |
| b_hi | (192,96) | 输入 dtype | **L0B** | qT / doT（切片 128×64）/ **kT（最后重灌）** |
| st_c / dpt_c | (96,64) | float32 | L0C | `s^T` / `dP^T` 累加器（每迭代清零） |
| dq_c | (64,192) | float32 | L0C | `dQ` 累加器（每迭代清零） |
| dk_c | (96,192) | float32 | L0C | `dK` 累加器（跨 query 循环） |
| dv_c | (96,128) | float32 | L0C | `dV` 累加器（跨 query 循环） |
| st_u / dp_u | (48,64) | float32 | UB（半区） | L0C 落地 + 向量输入 |
| p16_u / dsT_u | (48,64) | 输入 dtype | UB（半区） | 向量产出的两个 fp16 tile |
| lse_u / dlt_u | (64,) | float32 | UB | 行统计量（head-major 装载） |
| tail_u | (48,192) | float32 | UB（半区） | 块尾 `dk_c` 中转 |
| tailv_u | (48,128) | float32 | UB（半区） | 块尾 `dv_c` 中转 |
| out_k / out_v | (48,192) / (48,128) | 输入 dtype | UB（半区） | 块尾 cast 结果（写 GM 前） |

### 4.4 内存搬运路径

```
GM[cu_seqlens_q/kv] ──(标量读, L2 常驻)──▶ 标量: (req, n0, live, q_start, kv_start, len_q, len_kv, shift)
GM[h_off] ──(4 B 标量读)──▶ h（每 block 一次，AIC/AIV 同式）
GM ──(MTE2: nd2nz / dn2nz；越界尾部按**张量 extent** clamp+fill，由 AscendInsertOOBPadding 生成)──▶ L1
     k_s, v_s, kT_s(常驻) | q_s, qT_s, do_s, doT_s(每迭代)
GM ──(MTE2)──▶ UB lse_u, dlt_u(每迭代，head-major 1-D 256 B)
L1 ──(MTE1: 每次 GEMM 前一次 T.copy；dQ 的 A 走 asc_copy_l12l0a_transpose)──▶ L0A / L0B
L0A/L0B ──(Cube mad)──▶ L0C st_c, dpt_c, dq_c(每迭代) / dk_c, dv_c(跨迭代)
L0C ──(fixpipe, dual_copy 2:1, fp32 同 dtype)──▶ UB 半区 st_u/dp_u(48,64)
UB ──(SimtVF(128) 半区: exp / mask / mul / cast)──▶ UB 半区 p16_u, dsT_u
UB ──(dual_copy 2:1 → InsertNd2Nz → ascend_nd2nz_scatter)──▶ L1 pT_s, dsT_s（整块 96×64）
L0C(dq_c) ──(fixpipe + store-atomic add, fp32, 整块)──▶ GM dq_acc（跨 request 的行是精确 0）
L0C(dk_c/dv_c) ──(dual_copy 2:1)──▶ UB 半区 tail_u/tailv_u ──(cast)──▶ out_k/out_v
     ├─ 内部半块（own+48 <= kv_end）：T.copy 整块 ──▶ GM dk/dv
     └─ 尾半块（每 request ≤2 次）：SimtVF 逐 lane（`if own+i < kv_end`）──▶ GM dk/dv
```

无 GM 中转的中间块（`P/dP/dS` 全程留在 L0C/UB/L1），符合 §1.6.0 R2 的「不物化」结论。

### 4.5 UB 内存预算（每 AIV，半区）

**UB** 容量 253952 B（248 KiB）；块尾缓冲与每迭代缓冲不复用（保守上界，不做别名假设）：

| Buffer | Shape | dtype | 大小 (Bytes) |
|--------|-------|-------|-------------|
| st_u | (48,64) | float32 | 12288 B |
| dp_u | (48,64) | float32 | 12288 B |
| p16_u | (48,64) | bf16 | 6144 B |
| dsT_u | (48,64) | bf16 | 6144 B |
| lse_u + dlt_u | (64,) 各一 | float32 | 512 B |
| 每迭代缓冲的**第二个版本**（上 5 行的 ×1，容量余量口径） | — | — | 37376 B |
| tail_u | (48,192) | float32 | 36864 B |
| tailv_u | (48,128) | float32 | 24576 B |
| out_k | (48,192) | bf16 | 18432 B |
| out_v | (48,128) | bf16 | 12288 B |
| 合计 | — | — | 166912 B |

- **单版本（首版实现口径）**：把「第二个版本」行去掉 ⇒ `129536 B`（**50.99%**），余量 124416 B。
  其中每迭代活跃部分 = `st_u+dp_u+p16_u+dsT_u+lse_u/dlt_u = 37376 B`，块尾活跃部分 =
  `tail_u+tailv_u+out_k+out_v = 92160 B`。
- **双版本（容量余量口径）**：`166912 B`（**65.7%**），余量 87040 B ⇒ 即使每迭代缓冲取 2 版本也不超容量。
- **实现选择**：首版按单版本落地。理由不是容量，而是 `dual_copy` 的双版本语义未认证，且 dense 的设备码反查
  显示 `annotate_buffer_versions` 在本结构上只对 `qT_s` 生效（Z3 `unsat` 回退），见 §6.3。
- 编译侧旁证：dense 同 tile 原型的 `buf_dyn_shmem` 实占 116992 B（< 本表的单版本上界 129536 B，编译器做了
  别名复用），说明本表是**保守上界**。

### 4.6 L1 / L0A / L0B / L0C 预算

| 层 | Buffer 组合 | 合计 (Bytes) | 容量 (Bytes) | 占比 |
|----|------------|-------------|-------------|------|
| L1 | `k_s 36864 + v_s 24576 + kT_s 36864 + q_s 24576 + qT_s 24576 + do_s 16384 + doT_s 16384 + pT_s 12288 + dsT_s 12288` | **204,800**（200 KiB） | 524,288（512 KiB） | **39.1%** |
| L1（dense 设备码实测口径：仅 `qT_s` 拿到 2 版本） | +24,576 | **229,376** | 524,288 | **43.8%** |
| L1（假想的全每迭代双版本，不实现） | +106,496 | 311,296 | 524,288 | 59.4% |
| L0A | `a_k 36864 + a_t 12288` | **49,152**（48 KiB） | 65,536（64 KiB） | **75.0%** |
| L0B | `b_lo 24576 + b_hi 36864` | **61,440**（60 KiB） | 65,536（64 KiB） | **93.75%**（余量 4 KiB） |
| L0C | `dk_c 73728 + dv_c 49152 + st_c 24576 + dpt_c 24576 + dq_c 49152` | **221,184**（216 KiB） | 262,144（256 KiB） | **84.4%**（红线 240 KiB，余 18.6 KiB） |

- **L0B 余量只有 4 KiB ⇒ L0 侧一律单版本**（不做 L0 双缓冲）。这一结论是容量硬结论，不是偏好：
  `2×(24,576 + 384·Nk) ≤ 65,536` 只在 `Nk ≤ 21` 时成立，任何有用 tile 下 L0 都是单版本。
- **L0 单版本的代价**：`copy→mad→copy→mad` 在 AIC 时间线上串行，故
  `T_AIC = MTE1 + mad = 0.668 + 0.75 ≈ 1.42 µs`（dense 实测口径），这是本设计每迭代 2.97 µs 的主体。
- **L0C 红线**：221,184 B < 245,760 B（240 KiB），余 24,576 B。**head slab 不改变这个数字**（tile 形状不变）。

### 4.7 动态轴

| 动态轴 | 声明方式 | 运行时范围 |
|--------|----------|-----------|
| `T_q` | `total_q = T.dynamic("total_q")` | 契约上任意正整数；manifest 实测 32,256 ~ 32,768 |
| `T_kv` | `total_kv = T.dynamic("total_kv")` | 同上 |

`B`（组数）、`H`、`D_qk=192`、`D_v=128`、`block_n=96`、`block_m=64`、`_HN=48`、**`head_slab`** 全部是
**编译期常量**（进工厂特化参数，`@lru_cache` 级缓存，与源 `target_cache(maxsize=32)` 等价）。
`len_q/len_kv/shift/q_start/kv_start/n0/live/loop_st/loop_ed/own/kv_end/h_off` 是**运行期标量**，只出现在
标量运算、谓词与 copy 的偏移/长度中，**不出现**在 GEMM 的 M/N/K。

### 4.8 JIT 配置

```python
@functools.lru_cache(maxsize=64)
def _bwd_kernel(batch, heads, dim_qk, dim_v, dense_seqlen_q, dense_seqlen_kv,
                is_causal, scale, dtype, a2_fast=True, head_slab=None):
    if dense_seqlen_q != 0:
        raise ValueError("packed (_bwd_kernel) requires dense_seqlen_q == 0; dense lives in MLADenseBwd")
    head_slab = head_slab or min(32, heads)
    if not 1 <= head_slab <= heads:
        raise ValueError(f"grid: head_slab ({head_slab}) must be in [1, heads={heads}]")
    total_q  = T.dynamic("total_q")
    total_kv = T.dynamic("total_kv")
    grid = (total_kv // 96 + batch) * head_slab     # 单次 launch；<= 65535 由 wrapper 断言
    ...
    return tilelang.compile(_bwd_main, target="ascend")
# 编译标志沿用源：-O3、-DENABLE_BF16
```

`head_slab` 进 `lru_cache` 的键（同一 geometry 若因 `total_kv` 很大而缩小 slab，会再编译一份；
manifest 的四个 workload 都取 32，**只编译一次**）。v1 的 launch 侧形态（本版只把 `block_n` 从 64 改成 96）：

```python
cfg_block_n = 96                                        # 与工厂 cfg["block_n"] 同一常量（§5.6 g）
plan = plan_launch(batch, total_kv, heads, block_n=cfg_block_n)   # 越界抛 ValueError
fn, meta = _bwd_kernel(batch, heads, dim_qk, dim_v, 0, 0, is_causal, scale, dtype,
                       a2_fast, head_slab=plan.slab)
assert (total_kv // cfg_block_n + batch) * plan.slab == plan.max_grid   # 计划 == 产物
h_offs = torch.tensor([[l.head_off] for l in plan.launches], dtype=torch.int32, device=dev)
dq_acc.zero_()                                          # 仅一次，slab 之间不得再清
for launch in plan.launches:
    fn(q, k, v, dout, lse, delta, cu_q, cu_kv, h_offs[launch.index], dq_acc, dk, dv)
```

`out_idx`：`dk/dv/dq_acc` 都是调用方预分配的张量（`dq_acc` 必须由 host 先 `torch.zeros`——store-atomic 的
累加契约，**只在第一次 slab launch 之前清零一次**）。

---

## 5. Tiling 策略

### 5.1 计算类型

**类型：混合（Cube + Vector），Cube 侧走显式 L0 路径**。判定依据：5 个矩阵乘（每迭代 11.9 MFLOP）必须落
Cube；每迭代另有 6,144 个 score 的 `exp/mask/cast` 必须落 Vector；两者在同一 kernel 内交替（§2.2、§0.6 R2/R4）。

### 5.2 Block 划分

```python
block_n = 96        # key 方向（= dk/dv 累加器的 M，以及 dq 的 K）
block_m = 64        # query 方向（= dq 累加器的 M，以及 dv/dk 的 K）
_HN     = 48        # 每 AIV 的 key 行（dual_copy 2:1；且 48 % 16 == 0，见 nd2nz 硬约束）
head_slab = 32      # 每次 launch 的 head 数（不是分块参数，见下）
# 逻辑任务数 = heads * Σ_b ceil(len_kv[b]/block_n)  （+ 上界多算的 ≤ batch 个 tile/head）
# 单次 launch 的块数 = (total_kv // block_n + batch) * head_slab  <= 65,535（host 断言）
```

选择理由：
- `block_n` 决定 L0C 里 `dk_c`（`192·Nk·4 B`）与 `dv_c`（`128·Nk·4 B`），二者跨整个 query 循环常驻：
  `Nk=96` 时 120 KiB（+ 每迭代 `st/dpt/dq` 共 96 KiB = 216 KiB，84.4%）；`Nk=128` 会到 240 KiB 上下
  且必须把 `block_m` 降到 32（§1.6.3 备选）。
- `block_m=64` 让 `dq_c = 64×192×4 = 48 KiB`，同时 `dv/dk` 的 K 维取 64（mad 的 K 分块不退化）；
  半区向量段（48×64）在 `T.SimtVF(threads=128)` 下每线程 24 个元素，fp32 向量宽 4 整除、无尾 lane。
- **`Nk` 从 64 提到 96 是本版除 `trans_A` 之外的第二大收益源**：每 query 行的 L1 填充次数从
  `S_kv/128+1` 降到 `S_kv/192+1`（8x4k：32.5 → 22.0；mixed 加权：87.2 → 58.4），迭代数与 GM 填充总量
  同步 **−32.9%**（§1.6.0 R3）。
  `Nk=64` 是 v3 的选择，当时被 L0C 的 `dq_c` 压缩所迫；改用显式 L0 后 L0C 有余量放大到 96。
- **可行集**：`block_n/2 % 16 == 0` ⇒ `block_n ∈ {32, 64, 96, 128, …}`（§1.6.3）。**`block_n=80` 与 112 不可用**
  （半区 40/56 行，nd2nz `static_assert`）。
- **`head_slab` 不是分块参数**：它只改变「一次 launch 覆盖多少个 head」，不改变 block 形状、片上容量、
  FLOPs 或访存量；**不采用**「把 `block_n` 调大来把 grid 红线推远」的做法（`block_n=96` 已经把 grid 降到
  44,672，但 `tiles` 仍是运行期量，红线必须靠 slab + 断言守）。

### 5.3 约束分析

- **对齐**：`D_qk=192 = 12×16`、`D_v=128 = 8×16`、`block_n=96 = 6×16`、`block_m=64 = 4×16`、`_HN=48 = 3×16`
  全为 16 的倍数；mad 的 16×16 分形无尾块。`dq_acc` 尾轴 192 个 fp32 = 768 B（32 B 对齐 ✓）。
- **L0C**：单累加器校验 `M×N×4 ≤ 262,144`：最大者 `dk_c = 96×192×4 = 73,728 B` ✓；5 个同时驻留 221,184 B
  （84.4%，< 240 KiB 红线）。
- **L0A/L0B**：49,152 B（75.0%）/ 61,440 B（93.75%，余 4 KiB）⇒ **单版本**。
- **L1**：204,800 B（39.1%）；实测口径 229,376 B（43.8%）。
- **UB**：129,536 B（51.0%）；每迭代双版本上界 166,912 B（65.7%，可行但不实现）。
- **nd2nz 行数**：UB→L1 的 `dual_copy` 每 AIV 48 行（% 16 == 0 ✓）。
- **动态边界**：GEMM 的 M/N/K 静态；grid 为动态上界 + `live`；copy 偏移/长度为运行期表达式
  （平台 clamp 只认张量 extent，request 边界由掩码与 R7 双分支收口）。

### 5.4 注意事项（非整除、变长与静态边界）

> 统一原则：**GEMM 一律全 extent（或 0 原点静态切片）；动态 extent 只出现在 copy 与 grid；
> 装载的越界尾部由平台 pass 按张量 extent clamp+补 0，request 边界由掩码收口；
> 写回的 request 边界由 R7 双分支收口。**

1. **变长块枚举的静态边界**：`total_kv` 是运行期量，`tile_upper_bound = total_kv//96 + batch` 是**上界**
   （真实 tile 数 `Σ_b ceil(len_kv_b/96) ≤ 该上界`）。多算的 tile 数 ≤ `batch`（每 head），且非 live 块
   **不发起任何 DMA**（整个 body 在 `if live` 内）。量化：mixed 下 43,392 个 live 块 + 3×128 = 384 个空块
   （0.88%），空块成本 = 一次 O(B) 扫描。**运行期前提**（与源相同）：`cu_seqlens_kv[B] == total_kv` 且
   `cu_seqlens_*` 单调不减。
   静态化备用方案（v3 探针 P5 已编译）：grid 取编译期上界 `(TK_MAX//96 + B)*slab`，`live` 谓词原样保留，
   代价是 `TK_MAX` 取 32,768 时每 head 多算 14 个空块（mixed 共 1,792 个，2.7%）。
2. **尾 key 块（`len_kv % 96 ≠ 0`）**：`k_s/v_s/kT_s` 用全 extent 装载，越界行由 `AscendInsertOOBPadding`
   收窄 DMA 并补 0（设备码 `min(96, max((total_kv - kv_start) - n0, 0))` + `asc_fill_l1`）。补 0 使这些行的
   `s^T` 为 0，但 `exp(0·scale − lse) ≠ 0`，故**必须**由掩码臂的 `gi < len_kv` 把 `P` 置 0（→`dS=0`）。
   `dk/dv` 写回**不能**依赖同一 clamp（它只到 `total_kv`）→ R7 双分支。
3. **尾 query 块（`len_q % 64 ≠ 0`）**：`q_s/do_s/qT_s/doT_s/lse_u/dlt_u` 同样全 extent 装载 + clamp/补 0；
   掩码臂保留 `gj < len_q` 使填充列的 `P=0`；`dq_c` 的整块原子 store 溢出到下一 request 的行是**精确 0**（A3），
   张量末尾由平台 clamp 裁到 `total_q`（设备码 `min(64, max((total_q - q_start) - kq*64, 0))`）。
4. **因果对角块**：只有对角及相邻块进入掩码臂，其余走 A2 快路；`loop_st = max(0,(n0-shift)//64)` 保证不遗漏
   可见 query 块（最坏多算一个全掩码块，`P=0` 无副作用）。P7 设备码把它展开为
   `max(0, ((n0_dyn + lq) - lkv) >> 6)`。
5. **`len_q < len_kv` 与 `len_kv < len_q`**：`shift` 可为正/负；后者让头部 query 看不到任何 key（`lse=-inf`），
   此时掩码全假 → `P=0` → 梯度 0，`exp(-inf)` 被 `if_then_else` 丢弃（不产生 NaN，与 golden 的 `nan_to_num`
   对齐）。**快路安全性**由 §1.6.1 的充分条件保证。设备原型覆盖了这两种 shift 方向（§1.6.4 B）。
6. **退化请求（`len_kv ≥ 1` 但 `len_q = 0`）**：tile 由 `len_kv` 枚举，故这样的请求仍有 tile；若循环为空，
   `dk_c/dv_c` 从未被 `clear_accum` 初始化（**`T.clear` 不支持 L0C**）。处置：
   `loop_ed = max(ceil(len_q/64), loop_st + 1)` 保证至少一次迭代；该次迭代的全部 `P=0`（`gj < len_q` 恒假），
   于是 `dk_c/dv_c = 0`（正确：无 query 即无梯度），`dq` 整块写回精确 0（A3）。代价：只有这类退化请求多一次迭代。
   manifest 的 workload 与 golden 都不产生 `len_q = 0` 的请求，此处置是防御性的。
7. **`is_causal=False`**：`loop_st=0`，所有块进入快路（仍受 clamp 的尾块影响，见 2/3）。
8. **极小请求（`len_kv < 96`）**：全部走尾块路径；`ceil(len_kv/96)=1`，DMA 被 clamp 到 `len_kv` 行并补 0；
   R7 慢路生效（`own + 48 <= kv_end` 可能对两个 AIV 都为假）。设备原型 `[96,96]` 与 `[128,320]` 覆盖了这类几何。

### 5.5 分核策略（物理核数适配）

> 三要素标准：`.agents/skills/_shared/standards/core-split-strategy.md` §1。

- **① 逻辑核数**：`num_logical_kernels = heads × (Σ_b ceil(len_kv_b / 96) + 上界多算的 ≤B)`
  —— 每个逻辑块 = 一个 (request, key 分块, head)，即**全局**一次 block 发射；按 head slab 切成
  `ceil(H/slab)` 次 launch，**每次 launch 的块数**必须 ≤ 65,535：

  | workload | B | lens（q=kv） | H | Σ_b ceil(len/96) | 上界 tiles | **逻辑核数（live）** | slab | **单次 launch 块数** | launch 数 |
  |----------|---|---|---|---|---|---|---|---|---|
  | ds-v3-8x4k | 8 | 4096 ×8 | 128 | 43×8 = 344 | 349 | **44,032** | 32 | **11,168**（发出 44,672） | 4 |
  | ds-v3-mixed | 6 | 512,1024,2048,4096,8192,16384 | 128 | 339 | 342 | **43,392** | 32 | **10,944**（发出 43,776） | 4 |
  | kimi-k2-4x8k | 4 | 8192 ×4 | 64 | 86×4 = 344 | 345 | **22,016** | 32 | **11,040** | 2 |
  | smoke（Stage 3 门槛，manifest-smoke） | 2 | 256, 384（T=640） | 8 | 3+4 = 7 | 12 | **56** | 8 | **96** | 1 |
  | H=126 回归 | 8 | 4096 ×8 | 126 | 344 | 349 | **43,344** | 32 | **11,168**（尾 slab 空 698） | 4 |
  | H=127 回归 | 8 | 4096 ×8 | 127 | 344 | 349 | **43,688** | 32 | **11,168**（尾 slab 空 349） | 4 |

  （空块数 = `(slab − heads % slab) × tiles`，由 A4 逐 launch 校验。）

- **② 物理核数依据（任务平台实查记录，非假设值）**：目标型号 `ascend-950pr-9579`（dav-3510）。
  平台 `statectl platform --task-dir examples/mla_varlen_bwd/_bwd_kernel` 本次返回
  `hardware.observed_cores = {"cube": 28, "vector": 56}`，`core_source` 为
  `Ascend950PR_9579.ini [SoCInfo] ai_core_cnt=28, cube_core_cnt=28, vector_core_cnt=56;
  runtime probe rtGetAiCoreCount()=28 on Ascend950PR_9579, 2026-11-23`。
  本轮另在本机以 `torch.npu.device_count() = 1` / `get_device_name(0) = Ascend950PR_9579` 复核了型号
  （单卡可用，设备原型即跑在该卡上）。
- **③ 规模判定与分核方案**：六条几何的逻辑核数（22,016 … 44,032；最小一条 smoke 56，≈2 波）
  **全部 ≥ 物理核数 28**，且无法通过调整分块压到 28 的整数倍（`block_n` 上限受 L0C 与 nd2nz 约束，
  最大 128 ⇒ mixed 仍有 33,024 量级的逻辑块）。判定为 **极大规模**。
  **分核方案：1 逻辑块 = 1 次 block 发射（逻辑块超发，由硬件块调度器按波次派发）**，不采用「固定启动
  内核数 = 物理核数 + 核内串行多任务」，依据：
  (a) 每逻辑块的计算量是 10¹–10³ µs 级（mixed：38.6 TFLOP / 43,392 块 ≈ 890 MFLOP/块），远大于派发开销；
  (b) **packed 的负载不均（lens 512→16384，块内迭代数 1→256）使动态派发成为必需**：静态 28 等分会把
  256 次迭代的长块钉在某个核上，而块时长差 200×+；动态派发把长尾压到「一块」（最长块 = 256×2.97 µs
  = 0.76 ms，相对 mixed 的 400 ms 总量 = 0.19%）；
  (c) 「核内串行」需要把 Cube+Vector 混合体包进任务循环，而平台明确未认证「循环内跨单元/跨核同步」，
  风险高于收益。
  **规则偏离声明**：本项**偏离** `core-split-strategy.md` §1 的「极大规模」标准分支（该分支要求「固定启动
  内核数 = 物理核数，核内分批处理多个逻辑块」），采用标准未列出的第四方案「逻辑块超发」。理由即 (a)(b)(c)。
  **head slab 与超发方案的关系**：slab 只是**发射批次**的切分，每个逻辑块仍是一次独立发射、仍由硬件动态
  派发，故 (a)(b) 结论不变；slab 只把「一批 44,672 次发射」拆成「4 批 11,168 次」。
  **复评触发条件**（满足任一条则按 `[DESIGN_LIMIT]` 流程回写并重估）：① Stage 4 实测 block 派发/尾波占比
  > 5% 的单 kernel 时长；② 出现核间负载方差 > 2×；③ 平台认证「循环内 Cube+Vector 交替 + 跨迭代同步」后，
  核内串行方案转为可选；④ 实测 launch 边界（≤4 次）的冷启动开销超过单 kernel 时长的 1%。

### 5.6 启动计划：head slab 分解与越界断言

> 本节是 v3 的核心成果，**本版原样保留**（算法与分块变了，任务映射不变）。

**a) 平台约束与实测形态**

`T.Kernel(grid)` 的块索引按 16 位携带（知识条目 `TRAP-950-grid-over-65535`；`kb_stale_check` 报 **stale**，
内容由 v3 的本机实测补全）：

| 形态 | `grid` | 实测行为 |
|------|--------|----------|
| ≤ 65,535 | 11,168 / 43,776 / 65,535 | 正常，全部 block 执行 |
| `grid % 65536 == 0` | 65,536 / 131,072 | `aicore exception`（`device error type 3`，`error code 507035`） |
| **`grid % 65536 != 0`（最危险）** | 66,040 / 66,560 | **不报错**，只执行 `grid % 65536` 个 block；其余静默丢弃 |

`tiles` 含运行期量 `total_kv`，故编译期查不出、运行期不报错——v3 的 `ds-v3-8x4k`（Nk=64，66,560）只跑了
1,024 块。**因此这条约束必须由 host 侧在发射前判定并断言**，不能交给编译器或设备。

**b) grid 分解公式与 slab 选择依据**

```
tiles  = total_kv // block_n + batch                       # = GroupTiling.tile_upper_bound（运行期）
slab   = min(HEAD_SLAB_DEFAULT=32, heads, 65535 // tiles)  # 编译期常量
n      = ceil(heads / slab)                                # launch 次数
grid_i = tiles * slab                                      # 每次 launch 相同，≤ 65,535（断言）
tile   = bid // slab                                       # 静态除数
h      = h_off[i] + bid % slab                             # h_off[i] = i*slab（int32[1]，运行期）
生效条件：h < heads（尾 slab 守卫）；否则该 block 只做 1 次 4 B h_off GM 读 + 若干块级比较后立即返回
```

slab 取三个上界的 min：`heads`（H ≤ 32 时自动退化为单次 launch）、`65535 // tiles`（硬上界；
`tiles > 65535` 时该值 < 1 ⇒ `plan_launch` 直接抛错）、`HEAD_SLAB_DEFAULT = 32`（manifest 的 128 heads ⇒
4 次 launch，余量 ≥ 2×）。`slab` 是**编译期常量**（进工厂特化参数与 `lru_cache` 键），因此
`bid // slab`、`bid % slab` 仍是静态除数——这是 §1.6.3 候选 6/8 被否掉的同一条纪律（运行期除数未取证）。

**c) 逐 geometry 的 launch 计划（`plan_launch` 实跑输出，机器验证见 A4）**

| workload | tiles | slab | launch 数 | 单次 grid | 总块数 | 有效块 | 空块（尾 slab） |
|---|---|---|---|---|---|---|---|
| ds-v3-8x4k（B=8, H=128, L=4096） | 349 | 32 | 4 | **11,168** | 44,672 | 44,032 | 0 |
| ds-v3-mixed（B=6, H=128, T=32,256） | 342 | 32 | 4 | **10,944** | 43,776 | 43,392 | 0 |
| kimi-k2-4x8k（B=4, H=64, T=32,768） | 345 | 32 | 2 | 11,040 | 22,080 | 22,016 | 0 |
| smoke（B=2, H=8, T=640） | 12 | 8 | 1 | 96 | 96 | 56 | 0 |
| **H=126 回归** | 349 | 32 | 4 | 11,168 | 44,672 | 43,344 | **698（1.56%）** |
| **H=127 回归** | 349 | 32 | 4 | 11,168 | 44,672 | 43,688 | **349（0.78%）** |
| 自动缩小（2,100 tiles ≈ 201,600 tokens, H=128） | 2,100 | **31** | 5 | 65,348 | 326,740 | 269,824 | 56,916 |
| 上界本身（T_kv 使 tiles=65,535, H=128） | 65,535 | **1** | 128 | 65,535 | 8,388,480 | 8,388,480 | 0 |
| 越界（tiles=65,536, H=128） | 65,536 | — | — | — | — | — | **`ValueError`**（slab=1 也放不下） |

尾 slab 处理：`heads % slab != 0` 时最后一次 launch 的 `grid` 仍是 `tiles*slab`，其中
`(slab − heads % slab) × tiles` 个块的 `h ≥ heads`，由 `if h < heads:` 立即返回——设备码里这些块只做
**1 次 4 B 的 `h_off` GM 标量读**、若干块级比较与编译期固定的 pipe 同步事件，**不发起任何 DMA、不读
`cu_seqlens`**（P7 设备码 `if (h < 3) { … }` 把整个 body 包在里面）。备选「尾 slab 用更窄的 slab 再编译
一份」被否：只为省 ≤1.6% 的空块而多一份编译产物与一条代码路径，收益不成比例。

**d) 多 launch 的写集不重叠论证与同步需求**

三个输出都是 `(token, head, dim)` 行主序，head `h` 占每行内**连续**的 `D` 个元素：

| launch | 写 `dq_acc` 的地址集 | 写 `dk` 的地址集 | 写 `dv` 的地址集 |
|--------|--------------------|-----------------|-----------------|
| i（heads `[o, o+n_i)`） | `{(t, h, d) : t < T_q, o ≤ h < o+n_i, d < 192}` | 同（`t < T_kv`） | `{(t, h, d) : t < T_kv, o ≤ h < o+n_i, d < 128}` |
| j ≠ i | head 区间不交 ⇒ 地址集不交 | 同 | 同 |

1. **区间不交**：slab 区间是 `[0, heads)` 的划分（A4 逐 launch 校验 `Σ active_heads = heads`、两两交集 0）；
2. **无跨 head 归约**：算法沿 query/key 轴归约（`dV/dK` 沿 `m`，`dQ` 沿 key 块），**head 维从不参与任何
   归约或共享中间量**（`lse/delta` 也是 per-(head, token) 的）⇒ 每个 head 的输出只由该 head 的 block 决定；
3. **无需任何 launch 间同步**：写集不交 ⇒ launch 之间可交换、可任意顺序；实现上仍按同一 stream 顺序发射，
   仅需要「`_dq_land_kernel` 在所有 slab launch 之后」这一既有顺序保证；
4. **`dq_acc` 的清零只在第一次 launch 之前一次**：`dq` 是 store-atomic 累加，slab 之间再清零会丢掉前面
   slab 的贡献（§8.3 V7 覆盖）；
5. **`dk/dv` 是赋值**（不是累加）：每个 `(token, h)` 只被唯一一个 live tile 的块尾写一次 ⇒ slab 划分不引入
   覆盖顺序问题。

**e) 代价（只记账）**

- **launch 开销**：每次 launch 的 host 提交 + 设备端启动约 µs 级，manifest 最多 4 次；另有 `h_offs[i]` 的
  int32 单元素 H2D（4 B，可预构造一个 `(n,1)` 张量后按行切片）。
- **尾 slab 空块**：H=126 时 698 个（1.56%），每块只做 1 次 4 B `h_off` 标量读 + 2 个标量运算 + 一次分支；
  H 为 32 的倍数时为 0（manifest 的三个大 case 都是 0）。
- **launch 边界的 L2 冷启动**：每个边界让 q/do 的复用窗口重新预热，增量上界 = `(n−1) × 一次 q/do 全量读`
  （8x4k：`q_s+qT_s 3.22 GB + do_s+doT_s 2.15 GB ≈ 5.37 GB`）≈ 16 GB ≈ 3%（按 503 GB 的 3.2%）——
  这一项在 `Nk=96` 下比 v3（744 GB 口径的 2%）略高，但绝对量不变；Stage 4 用 `msopprof` 的 Task Duration
  实测后回填。

**f) 备选方案逐一裁决**

| 备选 | 语义 | 定量理由 | 裁决 |
|------|------|---------|------|
| **F1 head slab 多次 launch（采纳）** | 等价 | 单次 grid ≤ 11,168；空块 0~1.56%；launch ≤ 4 次 | ✅ |
| F2 调大 `block_n`（推远红线） | 等价 | `block_n=128` 只能配 `block_m=32`：L0C 221,184 B（84.4%）、UB 172,544 B（67.9%）；且 `tiles` 仍是运行期量，token 再长就复发 | ❌ |
| F3 把 head 折进核内串行 | 等价 | 丢失 packed 的动态派发收益（§5.5 b） | ❌ |
| F4 静态 grid 上界（`TK_MAX`） | 等价 | 已编译（v3 P5），作运行期回退；代价是 +1,792 个空块（mixed，2.7%） | 备用 |

---

## 6. 循环与调度结构

### 6.1 循环层级

| 层级 | 结构 | 说明 |
|------|------|------|
| L0（kernel） | `T.Kernel(grid)` 一维 | `grid = (total_kv//96 + B) * slab`；无 `threads` 参数 |
| L1（block 内控制流） | `if h < heads:` → 融合 decode → `if live:` | 三个块级谓词，全是标量；空块/越界 tile 立即返回 |
| L2（query 块循环） | `for kq in T.Pipelined(loop_st, loop_ed, num_stages=2)` | 运行期上下界；块内 10 条 L1→L0 copy + 5 GEMM + 2 段向量 + 1 次原子 store |
| L3（GEMM 内部） | `T.gemm` → `GemmMAD._lower_l0` | 由 lowering 按 mad 分形切 M/N/K（本设计的 M/N/K 全静态） |
| L4（逐元素） | `with T.SimtVF(threads=128): for i, j in T.Parallel(48, 64)` | 半区；两套（快路 / 掩码臂）+ 块尾 cast + 尾块守卫 |

**为什么不用 `T.Persistent`**：`T.Persistent` 面向「固定核数 + 核内串行多任务」的静态分核，而本算子的
packed 负载极不均（块内迭代 1→256），必须靠硬件动态派发压长尾（§5.5 ③ (b)）；且平台未认证「循环内
Cube+Vector 交替 + 跨迭代同步」。故采用 `T.Kernel` + 逻辑块超发。

### 6.2 循环边界与剪枝

- `loop_st = T.max(0, (n0 - shift) // 64)`（`is_causal=False` 时恒 0），`loop_ed = max(ceil(len_q/64), loop_st+1)`。
- 剪枝收益（bottom-right 因果，非因果上界 = `Σ_b K_b·M_b·H`）：8x4k 实际 1,441,792 vs
  `344×64×128 = 2,818,048` ⇒ **省 48.8%**；mixed 实际 3,770,496 vs `58,464×128 = 7,483,392` ⇒ **省 49.6%**
  （`58,464 = 6·8 + 11·16 + 22·32 + 43·64 + 86·128 + 171·256`）。
- 尾部：最后一个 query 块可能不满 64 行，由平台 clamp + 掩码收口（§5.4 3）。

### 6.3 多缓冲（`num_stages`）

`T.Pipelined(..., num_stages=2)`，与 dense 的配置一致。**实测事实（dense 设备码反查 + 本轮 P8 对照）**：
`annotate_buffer_versions` 面向 `T.Persistent + Z3` 自动调度器，本 kernel 用 `T.Pipelined` 时 Z3 报
`solver returned unsat, falling back to topological sort`，最终只有 `qT_s` 拿到 2 版本（L1 实占 229,376 B），
L0A/L0B/L0C 全单版本（容量所限本就不可能多版本）。因此：
- **不承诺**多版本带来的重叠收益；`num_stages=1` 与 `2` 的 marks 与预算在 P7/P8 中一致；
- 真正的重叠靠**执行单元分工**（AIC 做 5 个 GEMM 与 MTE1/MTE2，AIV 做 epilogue），而不是缓冲版本数；
- Stage 4 的 A/B 清单里保留「显式双缓冲 + 预取 i+1」一项（§1.6.3）。

### 6.4 调度形态

自动调度（`allow_autoschedule=True`）：`AssignCore` 把 `T.gemm`/L1↔L0 copy/L0C→UB/L0C→GM 分到 AIC，
把 `T.SimtVF` 与 UB→L1 分到 AIV（`__mix__(1, 2)`），`InsertSync` 插入跨单元事件。用户代码**不写**
手工 flag/barrier（平台未认证）。

---

## 7. 同步策略

| 位置 | 同步方式 | 依据 |
|------|---------|------|
| L1→L0 copy 与 mad | `InsertSync` 自动插入（同属 AIC，事件按 pipe 依赖） | 自动调度；P7 设备码含编译期固定的 pipe 事件 |
| mad（L0C 写）与 fixpipe（L0C→UB） | 自动（AIC 内） | 同上 |
| L0C→UB 与 `T.SimtVF` | 自动跨单元事件（AIC → AIV）；P7 设备码 `asc_sync_wait(PIPE_V, PIPE_MTE2, …)` | 同上 |
| `T.SimtVF`（UB 写）与 UB→L1（MTE） | 自动（AIV 内） | 同上 |
| UB→L1 与随后的 L1→L0 | 自动跨单元事件（AIV → AIC） | 同上 |
| `dq_c` 与 store-atomic | `set_atomic` / `set_atomic_none` 配对，夹住一次 `T.copy` | P7 设备码 `asc_set_atomic_add_float` → `asc_copy_l0c2gm` → `asc_set_atomic_none` |
| 跨 head slab 的 launch 之间 | **不需要同步**（写集不交，§5.6 d） | 同一 stream 顺序发射即可 |
| 块内两 AIV 之间 | **不需要**（`dual_copy` 2:1 把行切开，各 AIV 只处理自己的 48 行，无跨 lane 归约） | R2/R4 的语义论证 + 设备原型 6/6 PASS |

**不使用**：手工跨核 flag、`T.sync_threads`（kernel 作用域无线程域）、跨 block barrier（平台未认证）。

---

## 8. 验证方案

### 8.1 golden 函数（PyTorch 参考实现草案）

> **独立性要求**：golden 以 §0.1 的**数学语义**为依据（移植 `workloads/attention/mla.py::mla_bwd_reference`
> 的逐 request fp32 实现），**不复刻** §0.6 的分块算法（不使用 96/64 tile、不做 A2 快路特化、不模拟
> store-atomic 的溢出零行）。

```python
def golden_varlen_bwd(q, k, v, dout, lens_q, lens_kv, heads, scale, is_causal,
                      dtype, round_inputs=False):
    """fp32 per-request reference; gradients rounded to `dtype` exactly once."""
    DQK = q.shape[-1]
    Tq, Tkv = int(sum(lens_q)), int(sum(lens_kv))
    dk = torch.zeros(Tkv, heads, DQK, dtype=torch.float32)
    dv = torch.zeros(Tkv, heads, v.shape[-1], dtype=torch.float32)
    dq = torch.zeros(Tq, heads, DQK, dtype=torch.float32)
    oq = okv = 0
    for lq, lkv in zip(lens_q, lens_kv):
        sq, sk = slice(oq, oq + lq), slice(okv, okv + lkv)
        shift = lkv - lq
        m = torch.arange(lq)[:, None]; n = torch.arange(lkv)[None, :]
        vis = (n <= m + shift) if is_causal else torch.ones_like(m + n, dtype=torch.bool)
        for h in range(heads):
            qb, kb = q[sq, h].float(), k[sk, h].float()
            vb, dob = v[sk, h].float(), dout[sq, h].float()
            s = (qb @ kb.T) * scale
            sm = torch.where(vis, s, torch.full_like(s, float("-inf")))
            lse = torch.logsumexp(sm, dim=1)                       # 看不到任何 key 的行 = -inf
            p = torch.where(vis, torch.exp(s - lse[:, None]), torch.zeros(()))
            dp = dob @ vb.T
            delta = (p * dp).sum(1)                                # == rowsum(o*do)
            ds = p * (dp - delta[:, None])
            if round_inputs:                                       # 二级判据（见 8.2）
                p, ds = p.to(dtype).float(), ds.to(dtype).float()
            dv[sk, h] = p.T @ dob
            dk[sk, h] = scale * (ds.T @ qb)
            dq[sq, h] = ds @ kb                                    # 不含 scale
        oq += lq; okv += lkv
    return dk.to(dtype), dv.to(dtype), dq          # dq 留 fp32，供 _dq_land 对照
```

Stage 3 另需 `_dq_land` 与 `_delta` 的 host 侧参考（或直接对 `dq` 与 `dq_acc*scale` 比对）。

### 8.2 数值判据

| 输出 | 一级判据（vs fp32 golden，只在末尾舍入一次） | 说明 |
|------|--------------------------------------------|------|
| `dk` | `atol = max(5e-3, 6.9e-2 × max|ref|)`（沿用 manifest `reference_tolerance` 的 bf16 口径） | 含 `scale` |
| `dv` | 同上 | 不含 `scale` |
| `dq`（`dq_acc * scale`） | 同上；**任何绝对阈值必须 ≥ 1 个 fp32 ulp 且不得比 dtype 容差更紧** | v2 的教训：`|dq_acc|≈1.05e3` 时 1 ulp ≈ 1.22e-4，写 `1e-5` 必然假失败 |

**二级判据（dense 复盘的验收教训，必须内置）**：bf16 的强抵消元素可能出现个别越界（dense 实测 524,288 个
元素里 1 个）。判据流程 = 一级按上述容差比对；若失败，用 `round_inputs=True` 的 golden（把 `P/dS` 先 cast
到输入 dtype 再重算）复核，并把越界元素数写进结果行。判别值：dense 的 kernel-vs-same-rounding 2.3e-3、
kernel-vs-fp64 7.8e-2 ⇒ 属固有舍入而非缺陷。**本版设备原型的误差 2.65e-03（相对值）远低于 3e-2**，
说明该二级判据只有在生产大 shape 上才可能触发。

### 8.3 分层验证用例

**本版（Stage 1）已完成**（§1.6.4）：编译探针 P7–P9（3/3）、设备数值原型 6/6、`verify_equiv` 5/5、
单价测量 2 组、代价模型。

**Stage 3 必须新增/复跑的用例**（编号供实施与检视对齐）：

| # | 用例 | 期望 |
|---|------|------|
| V1 | **`ds-v3-8x4k` 逐 request 隔离**：整跑一次，再按 request 切片与 golden 逐 request 比对（8 个 request 各自） | 每 request 的 dk/dv/dq 都在 8.2 容差内；**不得出现「request k 的误差 = 全张量误差」**（那是跨 request 泄漏的特征） |
| V2 | **`ds-v3-mixed` 全 6 个 request**（512…16384）含尾块 | 同上；特别检查 512 请求（尾块最多）与 16384 请求（块最长） |
| V3 | **H=126 / H=127**（slab=32 的尾 slab） | 数值与 H=128 同 shape 的对应 head 切片一致；空块 698/349 不影响结果 |
| V4 | **grid 断言**：`plan_launch` 对越界几何抛 `ValueError`；`tiles*slab == max_grid` 一致性断言；H ≤ 32 时单 launch | 无静默路径；断言与产物一致（§4.8 的两条 assert） |
| V5 | **H=128 大 shape 逐切片复核**（`dq_acc` 每 head 切片） | 与 golden 逐 head 比对通过；`dq_acc` 的 fp32 累加与 golden 的 fp32 差异在容差内 |
| V6 | **cube/vector 利用率复核**（msprof） | `aic_cube_ratio ≥ 22%`（目标 25%±3）、`aiv_vec_ratio ≤ 55%`；与迭代单价 2.97–3.19 µs 互校 |
| V7 | **原子契约**：`dq_acc` 只在第一次 slab launch 前清零一次；再跑一次不清零会翻倍 | 加法语义成立；slab 之间不清零 |
| V8 | **`b_hi` 重灌顺序回归**：故意把 `kT` 提前灌 → `dq` 全错 | 证明当前顺序是必要条件（生产代码不引入该改动，仅作回归用例） |
| V9 | **A2 双臂等价**：`a2_fast=False` 强制全掩码臂，与默认结果比对 | 逐元素一致（A2 的等价性在实现层再确认一次） |
| V10 | **尾块慢路**：`len_kv ∈ {1, 47, 48, 49, 95}` | 慢路写回正确、无跨 request 覆盖（R7） |
| V11 | **`len_q = 0` 的退化请求**（防御性） | 不产生 NaN，`dk/dv` 为 0 |
| V12 | **smoke（560 tokens, H=8）** 作为 L0 门槛 | 5 个 launch 契约与三 kernel 串联全部通过 |

### 8.4 验收判据（本版）

1. **精度**：V1/V2/V3/V5/V9/V10 全过 8.2 容差（一级；必要时二级复核并记录越界元素数）。
2. **语义**：V4 的断言全部生效；V7 的原子契约成立。
3. **性能**（相对基线 1728.7 / 490.6 / 470.5 ms）：
   - **必达**：mixed ≥ 2.8×（≤ 616 ms）、8x4k ≥ 2.5×（≤ 196 ms）；
   - **预期**（本设计的投影）：mixed **400–430 ms（4.0–4.3×）**、8x4k **153–165 ms（3.0–3.2×）**、
     kimi 151–162 ms（2.9–3.1×）；
   - **利用率**：`aic_cube_ratio ≥ 22%`、`aiv_vec_ratio ≤ 55%`（基线 8.3% / 73.5%）；
   - **不承诺** 90% MFU（完全重叠下界 198 ms / cube 54%，属 Stage 4 探索档）。
4. **工程**：一次编译产物服务全部 slab；`dq_acc` 只清一次；无静默截断路径。

---

## 9. 风险点

### 9.1 技术风险与处置

| # | 风险 | 影响 | 处置 / 证据 | 状态 |
|---|------|------|------------|------|
| 1 | **`Nk=96` 下 L0B 只剩 4 KiB 余量**，L0 无法双版本 ⇒ `T_AIC = MTE1 + mad` 串行 | 每迭代多 0.67 µs（占 2.97 的 22%） | 容量硬结论（`2×(24576+384·Nk) ≤ 65536` 仅 `Nk ≤ 21`）；已计入投影；Stage 4 若要压 MTE1，唯一方向是**减少每迭代的 L1→L0 字节**（`trans_B=False` 让 `qT/doT` 不再单独装载），不是加缓冲 | 已量化 |
| 2 | **head slab 与 5 GEMM 的交互** | 若 `h_off` 进入 GEMM 形状或改变 L0 复用顺序，会静默错 | 结构性论证（§0.6 R9 末段）：`h` 只进 GM 中间轴与掩码；P9 的 `no_slab` 变体 marks 与 P7 完全一致；设备原型 `H=3/slab=2` 与 `slab=1` 数值同值 | 已闭合 |
| 3 | **`cu_seqlens` 解码与 L0C 累加器的交互** | 若 decode 依赖未初始化的 L0C 或与 `clear_accum` 顺序冲突 | decode 在循环前、纯标量、不触 L0C/UB/L1（§0.6 R6）；`clear_accum=(kq == loop_st)` 不受 decode 影响；P7 设备码两者分离 | 已闭合 |
| 4 | **UB→L1 `dual_copy` 的行数约束** | `block_n/2 % 16 != 0` 直接编译失败 | `_HN=48` ✓；可行集 `{32,64,96,128}`；`Nk=80/112` 已从备选清单剔除（dense 实测 `nd2nz_copy.h:59`） | 已闭合 |
| 5 | **`b_hi` 的重灌顺序** | `kT` 若不在 `doT/qT` 之后重灌，`dQ` 全错 | 代码注释 + 结构断言；设备原型 `dq` 误差 2.4e-05 是最小项 ⇒ 顺序正确；V8 作回归用例 | 已闭合 |
| 6 | **packed 的动态 grid + 混合 kernel** | 未组合验证过的形态 | P7/P8/P9 编译级确认；设备原型在动态 grid 上数值 PASS（`total_q/total_kv` 由张量推断） | 已闭合 |
| 7 | **`dq` 整块原子写回的跨 request 溢出** | 若溢出行的 `dS` 不是精确 0，会污染下一 request | A3 不变式（31,296 溢出元素 + 2,608 dq 行全 0）；掩码臂 `gj < len_q` 是前提，故 **A2 的掩码臂不可被"优化"掉** | 已闭合（有前提） |
| 8 | **`dk/dv` 尾块写回跨 request** | 写 0 覆盖下一 request 的真实梯度 | R7 双分支；设备原型逐 request 比对（`lens 128/320` 等）PASS | 已闭合 |
| 9 | **外推口径高估**（小 shape 单价含尾波/L2 冷启动） | 投影偏保守（dense 实测高估 15%） | 投影给两条单价（2.97 生产实测 / 3.19 本轮小 shape），并把 3.19 标为保守上界 | 已标注 |
| 10 | **packed 的负载不均**（块内迭代 1→256） | 尾波浪费 | 逻辑块超发 + 动态派发；最长块 0.76 ms 相对 mixed 400 ms = 0.19% | 已量化 |
| 11 | **L0C 红线**（240 KiB） | `Nk` 再放大就溢出 | 221,184 B（84.4%），余 24,576 B；备选 `128/32` 同为 221,184 B | 已量化 |
| 12 | **知识条目 stale** | 引用过期常数会误导 | `kb_stale_check` 报 `PL-950-gemm-l1-fill` / `CONST-950-gemm-mte2-burst` / `TRAP-950-grid-over-65535` stale；本文改用 dense 生产实测与本轮探针，stale 条目只作线索 | 已披露 |
| 13 | **`Nk=96` 让 grid 上界"看起来"不再是问题** | 可能被误判为「可以删掉 slab 与断言」 | §5.6 b/c/A4 明确：`tiles` 是运行期量，`auto-shrink` 几何与 `ValueError` 用例证明断言仍必要 | 已固化 |
| 14 | **msprof 的 cube/vector 口径** | cube 高不代表 AIC/AIV 真并发 | 验收同时看 `aic_cube_ratio` 与 `aiv_vec_ratio`（§8.4）；dense 的 25.2%/45.7% 是参照 | 已定义 |

### 9.2 常见错误（实施期必须避免）

| 错误 | 症状 | 正确做法 |
|------|------|---------|
| `T.clear(dk_c)` 清 L0C | `InternalError: Unsupported scope shared.l0c` | `clear_accum=(kq == loop_st)` + `loop_ed` 保证循环非空 |
| `kT` 提前灌进 `b_hi` | `dq` 全错（dk/dv 正确） | `kT` 在 `doT/qT` 之后**最后**重灌 |
| L0 路径与 L1 路径混用 | `dk/dv` 静默错（4.8e-1 / 3.1e-2） | 5 个 GEMM 全显式 L0 |
| 整块 `T.copy(L0C→UB)` | 只有一个 AIV 拿到数据（99.2% 错位） | 一律 `T.dual_copy` 2:1 |
| `T.vector(2) as sid` | 编译失败或运行值不确定 | `T.call_extern("int64","asc_get_sub_block_id")` |
| GEMM 用符号 extent | `ValueError: T.gemm requires static tile dimensions` | 全 extent 或 0 原点静态切片 |
| 用 `Nk=80/112` 回退 | `static_assert ROWS % 16 == 0` | `block_n ∈ {32,64,96,128}` |
| 每个 slab launch 都 `dq_acc.zero_()` | 前面的贡献被清掉 | 只在第一次 launch 前清一次 |
| 用「绝对值 ≤1e-5」判 `dq` | 必然假失败（1 ulp = 1.22e-4） | 用 dtype 容差，且阈值 ≥ 1 ulp |

---

## 10. 参考实现与资料

| 类型 | 路径 / 条目 | 用途 |
|------|------------|------|
| 源算子（迁移源） | `/workspace/TileOPs/src/tileops/_kernels/attention/mla/bwd.py`（`_bwd_kernel`，109–264） | 语义 / 算法 / 优化手段解读 |
| 源算子抽取副本 | `examples/TileOPs/tileops/kernels/attention/mla_varlen_bwd/_mla_varlen_bwd_kernels.py` | 逐行同源的 packed 分支 |
| golden 基准 | `/workspace/TileOPs/workloads/attention/mla.py::mla_bwd_reference` / `MLAVarlenBwdWorkload.ref_program` | §8.1 的移植来源 |
| 容差 | `/workspace/TileOPs/workloads/numerics.py::reference_tolerance` | §8.2 |
| **dense 冻结参照** | `examples/mla_dense_bwd/_bwd_kernel/{DESIGN.md, _bwd_kernel.py, REVIEW.md, RETROSPECTIVE.md, evidence/}` | C2 核的全部复用来源与生产常数 |
| 本版探针 | `history_version/design_probe_packed_cube.py`（P7–P9 编译） | §3.5.4 / §1.6.4 A |
| 本版设备原型 | `history_version/design_probe_packed_cube_num.py`（6 用例 + `--time`） | §1.6.4 B/C |
| 本版代价模型 | `history_version/design_probe_cube_cost.py` | §1.6.0 R3 与端口分解的全部数字 |
| 等价性验证 | `verify_equiv.py`（A1/A2/A3/A4，tile 96/64） | §1.6.1 的结果表 |
| 启动计划 | `launch_plan.py`（`plan_launch` / `NPU_GRID_MAX` / `HEAD_SLAB_DEFAULT`） | §5.6；Stage 3 需把 `BLOCK_N` 常量改为 96 |
| v3 证据 | `/tmp/opencode/MLA_VARLEN_BWD_8x4K_FINDINGS.md`、`history_version/design_probe_grid_slab.py` | grid 上界（R8）的三组决定性实验 |
| 平台资料 | `.agents/platforms/toolchains/ascend/references/programming.md`、`knowledge/repro/PL-950-gemm-tiles.py`、`PL-950-mixed-epilogue.py`、`PL-950-atomic-splitk.py`、`PL-950-conversion-layout.py`、`PL-950-row-primitives.py`、`knowledge/traps-950pr-execution.md`（`TRAP-950-aiv-sid-vector-frame`、`TRAP-950-l0c-region-store-stride`） | API 可行性、约束、陷阱 |
| 平台知识（**stale**，仅作线索） | `PL-950-gemm-l1-fill`、`CONST-950-gemm-mte2-burst`、`TRAP-950-grid-over-65535` | 已被 dense 生产实测与本轮探针替代 |

---

## 11. 相对上一版（v3）的关键调整与「为何不会再犯」

### A. 改动清单

| # | v3 的形态 | v4 的形态 | 依据 |
|---|----------|----------|------|
| **A1** | 5 个 GEMM 走 **L1 操作数路径**，`dQ` 被迫在向量段造第二份 `dS`（`ds_u[j,i]` 跨步写） | 5 个 GEMM 走**显式 L0 路径**，`dQ` 用 `trans_A=True` 复用 `dsT_s`；向量段只写 2 个 tile | §0.6 R3/R6、§1.6.4 A/C；dense 实测 scatter = 3.75 µs/迭代（63%） |
| **A2** | `block_n = 64` | **`block_n = 96`**（`_HN=48`） | 每 query 行填充次数 21.1→14.1，迭代数 −32.9%，GM 填充 −33%；L0C 有 84.4% 余量（v3 被 `dq_c` 压到 160 KiB） |
| **A3** | 向量段每 score 输出 3 个 tile（`p16`/`dsT`/`ds`），UB 写 14 B/score | **2 个 tile**，12 B/score，无跨步 scatter | A1 的直接结果 |
| **A4** | 累加器 `dk/dv/dq` 已在 L0C，但 GEMM 是 L1 路径 ⇒ L0A/L0B 由 lowering 隐式管理 | **显式 `alloc_l0a/l0b`**，5 个 GEMM 全 L0，不混用 | §1.6.4 E1（混用会静默错） |
| **A5** | 文档的 `Nk=80` 回退、`block_n=112` 备选 | **删除**（不可编译：半区 40/56 行 % 16 ≠ 0）；备选收敛为 `128/32` | dense 复盘 `nd2nz_copy.h:59 static_assert ROWS % 16 == 0` |
| **A6** | 每迭代价格式 roofline（L1 端口带宽为地板项，估算「待核验」） | **用 dense 的实测生产常数**（2.97 µs/迭代、cube 25.2%、MTE1 299.7 GB/s/核）给出管道级 roofline 与投影 | §1.6.0 roofline / §1.6.4 D |
| **A7** | grid 上界论证基于 Nk=64 的 66,560 块 | 更新为 Nk=96 的 44,672；**保留 slab + 断言**并新增「为何仍必要」的论证 | §5.6 b/c、A4 |
| **A8** | 风险表把「L0 单版本 MTE1」列为未量化 | 量化为 **0.668 µs/迭代（22%）**，并给出「压 MTE1 只能靠减字节」的方向 | §9.1 风险 1 |

**原样保留（v3 的成果，不得回退）**：head slab 发射与 `h_off` 传参、`plan_launch` 的启动前断言与
`ValueError`、融合式 O(B) 解码 + `live`、head-major `lse/delta`、`dq` 整块原子 + 溢出零行（A3）、
`dk/dv` 双分支（R7）、三 launch 契约、dtype 路径（`dk` 乘 `scale`、`dv` 不乘、`dq` 只写 fp32）。

### B. 为何不会再犯同一错误

| 错误 | 本次的防线 |
|------|-----------|
| **把 `transpose_A` 判为不可用**（v3 的根因误判）→ 引入 63% 的向量 scatter | 判定改为「以**平台已核验的 L0 路径**为参照」：`asc_copy_l12l0a_transpose` 在 dense 真机与 P7 设备码中各出现一次；§0.4 #6 与 §0.5 的处置列已同步纠正，并在 §9.2 列为常见错误 |
| 用**小 shape 单价**直接外推生产（dense 曾高估 15%） | §1.6.0 的投影给两条单价（2.97 生产实测 / 3.19 本轮小 shape 保守上界），并显式标注外推口径的偏差方向 |
| 把**分段的「和」当物理下界**（dense v2 的错误） | §1.6.0 roofline 按管道分类给单价，并给出「完全重叠下界 1.47 µs（mixed 198 ms）」作为 Stage 4 参照，**不作验收承诺** |
| 备选方案**不可编译**（Nk=80 / 112） | 把 `block_n/2 % 16 == 0` 写成硬约束并收敛可行集；§1.6.3 的备选只保留 `128/32` |
| 打包语义被 dense 的 4-D 技巧"顺手复用" | §0.2 / §0.8 B③ 明确 packed 无 `S` 维、必须用 R7；出片前的 6 个设备用例含尾块与跨 request 的逐切片比对 |
| 用「实测通过」否证边界缺陷 | §8.3 V1/V10 要求**逐 request 切片**比对（全张量 max 会掩盖泄漏），并要求 `len_kv ∈ {1,47,48,49,95}` 的慢路用例 |

---

## 12. 结论速览（对调度方六个问题的直接回答）

1. **根因**：v3 的内层迭代 5.95 µs 里 63%（3.75 µs）是向量侧的跨步转置 `ds_u[j,i]`——它是"L1 路径只支持 NT"
   的误判带来的代价；Cube 本身只占 10%（`gemm5` 0.587 µs），msprof 的 cube 8.3% / vec 73.5% 与此一致。
2. **与 dense C2 的复用/差异**：复用 10 项（5 个显式 L0 GEMM 含 `trans_A`、L0C 累加器、半区 `dual_copy`、
   2-tile 合并写、store-atomic、`kT` 重灌顺序、`hybrid` 模式、预算表、A1/A2 的等价结论、`Nk=96`）；
   **必须改 5 项**（融合解码取代静态 div/mod、head-major 统计量、R7 双分支取代 4-D 视图、A3 溢出零行、
   head slab 发射）。**head slab 与 5 GEMM 的 L0 复用无耦合**（`h` 不进 M/N/K，P9 的 marks 与 P7 全同）。
3. **roofline / MFU（Cube 口径）**：mixed **400–430 ms**、8x4k **153–165 ms**、kimi 151–162 ms；
   相对基线 **4.0–4.3× / 3.0–3.2× / 2.9–3.1×**；`aic_cube_ratio` **23–27%**（基线 8.3%）、`aiv_vec_ratio` 目标 ≤55%。
   **不承诺 90% MFU**：完全重叠下界为 mixed 198 ms / cube 54%（需跨迭代软件流水，属 Stage 4 探索档）。
4. **片上预算**：L0C 221,184 B（**84.4%**，红线 240 KiB 内）；L1 204,800 B（39.1%，实测口径 229,376 B = 43.8%）；
   UB 129,536 B（**51.0%**）；L0A 49,152 B（75.0%）；L0B 61,440 B（**93.75%，余 4 KiB ⇒ L0 单版本**）。
   **dense 的 L0B 93.75% 与 L0C 240 KiB 红线在 packed 下逐字节相同**（tile 形状与 `_HN` 相同）。
5. **关键风险**：L0B 仅 4 KiB 余量 ⇒ `T_AIC = MTE1(0.668) + mad(0.75)` 串行（占 2.97 µs 的 48%）；
   `block_n/2 % 16 == 0` 的硬约束；`b_hi` 的重灌顺序（`kT` 最后）；`dq` 溢出零行依赖「掩码臂不可省」；
   packed 的逐 request 写回必须靠 R7（无 4-D 视图可用）。全部有证据或已量化的处置（§9.1）。
6. **验证清单**：V1 逐 request 隔离（8x4k 8 个）、V2 mixed 全 6 request + 尾块、V3 H=126/127 尾 slab、
   V4 grid 断言与一致性、V5 H=128 逐 head 切片、V6 cube/vector 利用率、V7 原子契约、V8 `b_hi` 顺序回归、
   V9 A2 双臂等价、V10 慢路 `len_kv ∈ {1,47,48,49,95}`、V11 退化请求、V12 smoke 门槛（§8.3）。
