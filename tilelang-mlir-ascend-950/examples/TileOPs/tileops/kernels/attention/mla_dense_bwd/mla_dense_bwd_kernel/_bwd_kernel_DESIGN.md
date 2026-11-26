# MLA prefill 反向主 pass（_bwd_kernel）算子设计文档

> 迁移任务（harness 子模式）。源算子：`/workspace/TileOPs/src/tileops/_kernels/attention/mla/bwd.py` 中的
> `_bwd_kernel`（GPU TileLang 实现，`MLADenseBwdKernel` / `MLAVarlenBwdKernel` 共用工厂）。
> 目标平台：`ascend-950pr-9579`（dav-3510，CANN 9.2.1，TileLang ascend 0.1.15），编程模式 **hybrid**
> （Cube 显式 L0 路径 + Vector AIV），偏离理由见 §2。
> 本函数是 `MLADenseBwdOp` 三次 launch 的第二个（`_delta_kernel` → **`_bwd_kernel`** → `_dq_land_kernel`）；
> 跨函数接口契约为：`delta` fp32 `[B,H,S_q]`、`dq_acc` fp32 `[T_q,H,D_qk]`（本函数原地累加，`_dq_land_kernel`
> 统一乘 `scale` 并 cast）。
>
> **本版为 revision v4**（在 v3 的冻结结构上落地 **A5「延迟梯度 GEMM」旋转调度**，逐条对照见 **§13**，
> 六个调度问题的直接回答见 **§14**）。v3 的结构（5 个 GEMM 全显式 L0 + 向量 epilogue 合并写 +
> `trans_A` 复用 `dS` + tile 96/64 + 4-D batch 视图）与全部数学、I/O、dtype、grid **一律不动**；v4 只改
> **语句时序**：把 `GB(k)={dV,dK,dQ}` 从 `EP(k)` 之后移到 `EP(k+1)` 之旁（`T.serial` + 2-tile 静态展开 +
> 显式两份缓冲 + drain），使 AIC 不必等 AIV 的 epilogue。
>
> **v4 的输入不是检视意见，而是 Stage 4 的 `[DESIGN_LIMIT]` 归因**（`examples/mla_dense_bwd_perf/perf_opt/
> perf_feedback.md`）：`ds-v3-b2-8k` 实测 154.4 ms / 2.944 µs 每迭代，AIC 五管道忙时之和 2.822 µs
> 「近乎全串行」，`mad` 忙时已到峰值 98.7%（13.32 TF/s/核），残差是迭代内 `mad→fixpipe→AIV epilogue→
> scatter→mte1→mad` 的**跨核依赖链**，不是计算量、不是 tiling、也不是 `num_stages`（13 个分支全在 ±3%
> 噪声内，最好 −0.95%）。同族 varlen 的 v5 已用同一手段实测 **−26…−34%**（3.167→2.092 µs），本版把它
> 适配到 dense 的 **4-D batch 视图 + 无 head slab** 形态，并**用本算子的探针重新取证**（§1.6.4 I）。
>
> v3 相对 v2 的改动（3 个阻塞项 + 5 条建议）保留备查：
> ① **`dk/dv` 块尾写回的 request 隔离**（GM 张量改 4-D batch 视图 `(B,S,H,D)`，根治 B≥2 且 `S_kv % 96 ≠ 0`
> 时尾块跨 request 覆盖，§5.4 / §12-B1，探针实测 3-D 变体非因果 5/5 失败、4-D 5/5 通过）；
> ② **「80% Cube MFU 不可达」改为「当前串行迭代结构下不可达」**（管道级 `max` 下界 1.24 µs ⇒ 62.7 ms，
> 改进后 `0.80 + MTE1` ⇒ 40.5–50.6 ms，关键硬条件 `MTE1 ≤0.20 µs`，§1.6.4 F/G / §12-B2）；
> ③ **R3 与全部 roofline 数字逐项复算**（C2 的 GM 访存 281 → **190 GB**、端口分解 221,184 → **217,088 B**、
> fixpipe/UB→L1 补 `×28`、`×86` → 因果均值 43.7、C1 的 UB 71.2% → **47.2%**，§1.6.0 / §12-B3）。
> v2 的引入理由与逐项调整见 §11。**执行上下文偏离**（平台 `prototype_command` 解释器无 torch）已在 §3.5.4 登记。

---

## 0. 源算子解读与迁移分析

> 方法论：tilelang-op-design skill 的 references/migration-analysis.md（三问解读 → 耦合性四态 → 重设计）。
> 本章结论驱动 §1–§10。**v2 只改判定所依据的证据与结论（§0.5/§0.6），不改 §0.1–§0.3 的语义解读；
> v3 只改 §0.2 的 NPU 侧签名口径（3-D packed → 4-D batch 视图）与 §0.4/§0.5 的 #11 判定口径；
> v4 不改本章任何结论**——A5 旋转是 §0.6 R2「Cube+Vector 同 kernel 交替」的实现层调度选择（§0.4 #4
> 「异步流水」的等价替换形态），不触碰语义（§0.1）、I/O（§0.2）、算法（§0.3）与耦合性判定（§0.5）。

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

**NPU 侧签名口径（v3 修正，B1）**：源算子按 packed `(T,H,D)` 传参、并在块尾用**逐行守卫**
`if n0[0]+i < len_kv` 保证只写本 request 的行（`bwd.py:255-260`）。本设计把 GM 张量改成 **4-D batch 视图**
`q/do/dq_acc (B,S_q,H,192|128)`、`k/v/dk/dv (B,S_kv,H,192|128)`——host 侧 `.view(B,S,H,D)` 与源算子的
`view(B*S,H,D)`（`bwd.py:363-378`）同为**零拷贝元数据操作**，语义等价；而平台上 `AscendInsertOOBPadding`
的 clamp 维度是**buffer 形状的行数**，4-D 视图让它自动变成「按 request 裁剪」，与源的逐行守卫同义
（3-D 视图下 clamp 只能裁到整张 `B*S_kv`，见 §5.4 第 1 条与 §12-B1 的设备码证据）。
`cu_seqlens_*` 在 dense 下仍只保留签名兼容。

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
| 11 | `lse/delta` 行广播（S8/S11） | 每行一次标量代替逐元素重算 | SMEM 标量读 | — | **等价替换**（lane 私有读，§1.6.3；v3 统一口径：§0.4 与 §0.5 同判，修 v2 的 #11 判据矛盾） |
| 12 | warp 划分 `FullRow` + `threads=256`（S6–S14） | 匹配 mma 的 warp 分形 | warp-level 分片 | warp 32 线程 + 寄存器分片 | **重新设计**（§0.6 R4） |
| 13 | **（NPU 适配构造，非源码手段）** `clear_accum` | 表达「首块清零 / 跨块累加」 | mma 零累加位 | — | — |
| 14 | packed/dense 双分支 `GroupTiling`（S1） | 一个工厂服务两种 layout | 编译期分支 | 无 | **舍弃（本函数）**，见 §0.7 |

**关于 #13 的口径**：源码**没有** `clear_accum` 参数，只用 `T.clear`（`bwd.py:200-201, 225, 227, 250`）；
`clear_accum` 是**本设计的 NPU 侧构造**（§3.2 步骤 4），用于让 L0C 累加器表达同一语义。

**小结（第三问 Why fast）**：共 14 项，其中 4 项纯算法层（保留：#1/#2/#3/#6）、6 项有 NPU 等价形态
（等价替换：#4/#7/#8/#9/#10/**#11**）、2 项需重设计（寄存器累加器 #5、warp 划分 #12）、1 项本函数舍弃
（packed #14）、1 项为迁移侧构造（`clear_accum` #13）。**v3 修口径**：v2 的 §0.4 把 #11 判「保留」而 §0.5
判「等价替换」，本版统一为**等价替换**（与 §0.5 的 18 行统计一致）。
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
| `T.Pipelined(num_stages=2)`（#9） | 优化 | cp.async / 双 SMEM 槽 | 有（`annotate_buffer_versions`） | **等价替换** | L1 每迭代缓冲取 2 版本（§6.3）；**实测本结构 `num_stages=2` 无可判定收益（3.391–3.421 vs 3.446–3.486 µs，−1.6%，±2% 内），故只保留版本标注，重叠靠结构而非调度器**（§1.6.4） |
| `T.atomic_add` 逐元素（#10） | 优化 | 全局原子 ALU | 有（store-mode 原子） | **等价替换** | `T.set_atomic("add","float32")` + L0C→GM `T.copy`（已核验 PL-950-atomic-splitk；v1 实现 L0 全过） |
| `lse/delta` 行广播（#11） | 优化 | SMEM 标量读 | 有（UB lane 索引读） | **等价替换** | lane = query 轴，`lse_u[j]` 为 lane 私有读（§1.6.3） |
| warp 划分 + `threads=256`（#12） | 优化 | warp 32 线程 | 无（NPU 无 warp） | **重新设计** | R4：`T.SimtVF(threads=128)` 半区 lane 映射（副块 id 由调度合成） |
| `FullRow` policy 参数 | 优化 | warp 划分启发式 | 无 | **舍弃** | L0 路径按 mad 指令形状与 L0 布局决定分形，policy 无语义；不影响正确性 |
| `T.clear` 显式清零 | 优化 | mma 累加器初值 | 有 | **等价替换** | `clear_accum`（跨 query 循环用 `clear_accum=(kq == loop_st)`，编译级已验证） |
| packed / `GroupTiling` 分支（#14） | 优化 | 无 | 有（属另一 op） | **舍弃（本函数）** | 只覆盖 dense（§0.7） |
| host `view`/`arange` 元数据 | 语义 | 无 | 有 | **保留** | 仅 shape/stride 视图与常量构造 |

统计（18 行判定）：保留 6 项（语义、#1、#2、#3、**#6**、host 元数据）、等价替换 7 项（#4、#7、#8、
#9、#10、**#11**、`T.clear`）、重新设计 3 项（R1/R2/R4）、舍弃 2 项（`FullRow` policy、packed 分支）。
（v3 修口径：v2 此处为「保留 7 / 等价替换 6」，把 #11 计入保留；按 §0.4 #11 已统一为等价替换，故本行改为 6/7。）

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
  边界：无效 key/query 位置的贡献由 §1.6.1 A2 掩码置零（覆写补 0 引入的非零 `exp(-lse)`）；块尾写回在
  **4-D batch 视图**下由平台 OOB clamp **按本 request** 裁剪（§5.4 第 1 条）。**v3 修正**：v2 用 3-D packed
  视图，clamp 只能裁到整张 `B·S_kv`，B≥2 且 `S_kv % 96 ≠ 0` 时尾块会写进下一个 request 的行（§12-B1）。

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
  边界：尾 query 块由平台 OOB clamp 把 store 区段裁到**本 request 的** `S_q` 以内（4-D batch 视图，设备码
  `min(64, S_q − m0)`；3-D 视图下为 `min(64, B·S_q − …)`，越界写是「store-atomic 加 0」的良性情形，
  但为与 §5.4 的 request 隔离口径统一，v3 一并改为 4-D）。

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
3. **query 块循环**（因果剪枝：`kq ∈ [max(0,(n0-shift)//64), ceil(S_q/64))`）。v4 把每个迭代的三段工作
   **显式命名**，并把梯度 GEMM 段**滞后一迭代**（A5，§1.6.1/§6.3）：
   a. **`F(k)` 装载**：搬入 `q_s (64,192)`、`qT_s (192,64)`、`do_s (64,128)`、`doT_s (128,64)`、
      `lse/delta (64,)` fp32 —— 写入**版本 `v(k)`**（`_s0`/`_s1`，静态名字）；
   b. **`GA(k)` 打分**：`st_c = k·q^T`、`dpt_c = v·do^T`（显式 L0，每迭代清零）→ 半区 `dual_copy` 落到 UB；
   c. **`EP(k)` epilogue**：向量侧算 `p = exp(st*scale − lse)`（运行时块级 mask 特化 + 因果/尾块掩码）→
      只写两个 fp16 tile：`p16`、`dsT = p*(dpt−delta)`，经 `dual_copy` 落 L1 的 `pT_s[v(k)]`、`dsT_s[v(k)]`；
   d. **`GB(k−1)` 梯度 GEMM**（**滞后一迭代**，读 `v(k−1)=1−v(k)` 的 `pT/dsT/qT/doT`）：
      `dv_c += pT·doT^T`、`dk_c += dsT·qT^T`（跨迭代累加，`clear_accum=(k−1 == loop_st)`）、
      `dq_c = dsT·kT^T`（**`transpose_A=True`**，每迭代清零）；随后 `set_atomic("add","float32")` 把
      `dq_c` 整块落到 `dq_acc[(k−1)·64 行, h, :]` 并关闭原子模式。
   **首迭代**没有可滞后的 `GB`；**末迭代**的 `GB(loop_ed−1)` 由循环后的 drain 补（版本按 `n_it` 奇偶选）。
4. **块尾**：`dk_c`、`dv_c` 半区落地 UB，向量侧 cast（`dk` 另乘 `scale`）后写回 GM；写回区段在
   **4-D batch 视图**下由平台 OOB clamp 按本 request 裁剪（`min(HN, S_kv − (n0+sid·HN))`，与源算子的
   逐行守卫同义，§5.4）。

**v4 只重排 c/d 的先后与归属迭代，不改变任何算术**：每次 `GB` 消费的 5 个操作数与数据流图逐字相同、
`dV/dK` 的 fp32 累加顺序仍沿请求内 query 块升序、`dq` 的写回行区间不变。因此 `dk/dv` 与 v3 **逐位相同**
（探针实测，§1.6.4 I-3）。**为什么必须 2-tile 静态展开**：L0 操作数 region 的原点必须为 0 且 M/N/K 取自
allocation 形状，故版本只能由**静态缓冲名**给出，不能用运行期奇偶索引（§1.6.3 决策 D5）。

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

**v4 的版本增补（A5）**：上图中 `q_s,qT_s,do_s,doT_s,lse,delta`（`F` 产物）与 `pT_s,dsT_s`（`EP` 产物）
各存在**两份物理缓冲**（版本 A/B），`GB(k−1)` 从**上一迭代**的那一份读；`st_c/dpt_c/st_u/dp_u/p16_u/dsT_u/
dk_c/dv_c/dq_c` 的生产与消费都在**同一迭代内**，故**仍是单版本**，数据流与 v3 逐字节相同（§4.6）。

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
| 2 | 块级 mask 特化 | 全可见块省掉 `select`，只在对角/尾块生成掩码 | 谓词在该块内恒真，恒等 | 每 score 少 2 次比较 + 1 次选择；快路块占**全方阵 49.5%**（`verify_equiv` 口径）/ **实际执行块 ≥98%**（`loop_st` 剪枝后，见下） | ✅（A2） |
| 3 | 行统计量 `lse·log2e` 逐行提升 | 先算 `lse2[m]=lse[m]·log2e` | 恒等 | 被 #1 完全吸收 | ❌（冗余） |
| 4 | `scale` 折进 `q`/`k` 装载 | `(q·scale)` 与 `(k·scale)` 参与两个 GEMM | 数学恒等（`dK=ds·q'`、`dQ=dsᵀ·k'` 都自动带 scale，块尾 `dk*scale` 也可省） | **代价是乘次与舍入**：每迭代 2×(64×192)=12,288 次 **fp16** 乘（`q`、`k` 各一份），换掉 6,144 次 **fp32** 乘（score 的 `s*scale`，每次 1 条 FMA）+ 块尾 18,432 次 fp32 乘；且输入先被舍入一次（fp16×常量后落 fp16） | ❌（定量：乘次 2× 且精度更差；v3 重写理由，v2 的「需两份副本」论据不成立——折进后两侧都自动带 scale） |
| 5 | `delta` 项拆成 rank-1 修正 | `dS=P∘dP − P∘δ` 分两项进 GEMM | 数学恒等 | 需多一个 GEMM（+192·Mq·Nk MAC/迭代 ≈ +5.7% 计算量） | ❌ |
| 6 | `dV`、`dK` 合并为一个 GEMM | 拼 `A=[pT; dsT]`、`B=[doT; qT]` | **不等价**（会求和成 `pT·doT^T + dsT·qT^T`） | — | ❌（不等价） |
| 7 | 物化 `P/dS`（非 flash） | 一遍算 `s/P/dP/dS` 落 GM，再扫描做 3 个 GEMM | 恒等 | 中间缓冲 O(B·H·S²) 与多遍扫描 | ❌（R3 淘汰） |
| 8 | query-stationary 变体 | `dQ` 常驻、`dK/dV` 原子累加 | 恒等 | 把原子放大项从 `dQ` 换成 `dK+dV`（宽 320） | ❌（R4 淘汰） |
| 9 | 两 kernel 拆分（dQ kernel + dK/dV kernel） | 各自单遍 | 恒等 | FLOPs ×2，破坏三 launch 契约 | ❌ |
| 10 | **单份 `dS` 双向复用**（`trans_A`） | 只产 `dS^T (Nk,Mq)`；`dK` 用它作 A，`dQ` 用它作 `trans_A` 的 A | 恒等（`(AB^T)` 与 `(A^T)^T B^T`） | 消掉一份转置副本与向量侧 scatter（实测 −3.75 µs/迭代，§1.6.4） | ✅（§0.6 R3） |
| **11** | **迭代内语句重排：`GB` 滞后一迭代**（v4 新增，调度级） | 体 = `F(k)→GA(k)→EP(k)→GB(k)`（单版本、`GB` 紧跟 `EP`）  **→ 优化后：**体 = `F(k)→GA(k)→EP(k)→GB(k−1)`；`F/EP` 产物取 `v(k)`，`GB(k−1)` 取 `v(k−1)=1−v(k)`；首迭代无 `GB`，末迭代由 drain 补 | **调度等价**：每个 (k, query 块) 的 5 个 GEMM 的操作数集合、`clear_accum` 语义、fp32 累加顺序（`dV/dK` 沿 query 块升序）与 `dq` 写回行区间逐字不变；只改「与谁并发」。本算子探针：`dk/dv` **逐位相同**、`dq` 差 1.8e-8 相对（fp32 原子到达序，§1.6.4 I-3） | 每迭代消掉 `fixpipe + AIV 1.30 + ub2l1` 的跨核串行：同构族 varlen 实测 **3.167→2.092 / 4.267→3.134 µs（−33.9% / −26.5%）**；**dense 本算子探针 2.965→1.870 µs（−36.9%）**（§1.6.4 I-4） | ✅（**A5，本版核心**） |
| **12** | 两级旋转（`GA` 也提前一迭代，体 = `GB(k−1)+GA(k+1)+EP(k)`） | 体 = `GB(k−1)+GA(k+1)+EP(k)`（`GA` 提前）；**新增代价**：`st_c/dpt_c` 需两份 | 恒等 | **收益 ≈0**：A5 落地后墙钟已由 `mte1+mad+fixpipe`（dense 0.680+0.769+0.570 = 2.019 µs）界定，`GA` 再与 `EP` 重叠不再缩短它；代价确定：`2×(st_c+dpt_c)+… = 270,336 B` = **L0C 103.1% 越界**，必须把 `dP` 并入 `st_c` 并在 `GA` 内部引入 fixpipe→mad 串行（§6.3 E） | ❌（R3/R4 行 C2‴、§1.6.4 I-5） |

**R2 在线算法**：**有，且基线即在线变体。** 结构依据：`lse`（前向 log-sum-exp）与 `delta`（`rowsum(o*do)`）
都由外部提供，因此反向**不需要**前向 online softmax 的 running max / 重缩放机器；每个 (key 块, query 块) 对
只需重算局部 `s/dP` 并立刻消费（`P/dS` 只活在片上），得到单遍扫描、O(tile) 中间态的流式结构。
对照的非在线变体是 R1#7（物化 `P/dS`，3–4 遍、O(B·H·S²) 缓冲），在 R3/R4 中被容量淘汰。
**无更优在线结构**：本算子的输出里必有一个（`dQ` 或 `dK/dV`）要沿另一轴跨块归并，单遍结构下无法再压缩
（R1#6 的合并已证不等价），故在线族内部不存在「更少遍数」的变体——这条负向断言的结构依据是上面的交换性分析。
**v4 补充**：A5 旋转**不改变扫描遍数**（仍是 1 遍），也不增加中间缓冲——它只把同一迭代内的三段工作在
**时间轴**上错开一迭代，故 R2 的「单遍 + O(tile) 中间态」结论与负向断言均不受影响。

**R3 复杂度对比**（四口径；代表 workload = `ds-v3-b2-8k`：B=2, S_q=S_kv=8192, H=128, D_qk=192, D_v=128, bf16；
因果 score 数 8.59e9）。**v3 对本表逐项复算**，每个数字的公式与出处见本节末「R3 口径与出处」：

| 候选 | FLOPs | 访存 GM 视图 (Bytes) | 扫描遍数 | 中间缓冲峰值 / 片上预算 | 每 query 行 L1 填充 |
|------|-------|---------------------|---------|----------------------|-------------------|
| **C1 = v1（L1 路径 NT + 向量转置副本，Nk=64）** | 14.44 T | **≈282 GB**（q/do 填充 174.2 + k/v/kT 2.1 + `dq` 原子 103.9 + `dk/dv` 1.3） | 1 | L0C 160 KiB / L1 168 KiB / **UB 119.8 KiB（47.2%）** | 1280 B × **因果均值 64.5** 次（非因果上界 128） |
| **C2 = v2/v3（全 L0 路径 + `trans_A` 复用，Nk=96）** | 14.49 T（+0.3% 尾块量化） | **≈190 GB**（q/do 填充 116.6 + k/v/kT 2.2 + `dq` 原子 69.6 + `dk/dv` 1.4） | 1 | L0C 216 KiB / L1 304 KiB（双版本）/ UB 163 KiB / L0A 48 KiB / L0B 60 KiB | 1280 B × **因果均值 43.7** 次（非因果上界 86） |
| **C2″ = C2 + A5 旋转（本版 v4）** | 14.49 T（**逐位不变**） | **189.8 GB（逐字节不变）** | 1 | L0C 221,184（**84.4%，不变**）/ **L1 311,296（59.4%，与 C2 同）** / **UB 130,048（51.2%，比 C2 的 65.7% 更低）** / L0A 49,152（75%）/ L0B 61,440（93.75%） | 同 C2；墙钟 **2.965→1.870 µs**（本算子探针 B=2/S=8192/H=8，§1.6.4 I-4） |
| C3 = v1 + 两趟向量转置（`epi2`） | 14.44 T | ≈282 GB | 1 | 同 C1 | 1280 B × 64.5 |
| C4 = 两 kernel 拆分 | **28.9 T**（`P`、`dP` 各算两遍） | ≈190 GB（无原子） | 2 | 同量级 | 两倍 launch |
| C5 = 物化 `P/dS`（R1#7） | 14.44 T | ≥270 GB 额外往返 | 3–4 | **O(B·H·S²) ≈ 34 TB** → 远超 128 MiB L2 | 不可行 |
| C6 = 纯 Vector（无 Cube） | 14.44 T | 同 C2 | 1 | 同量级 | 28 Vector 核 lane 级乘加，算力低 1–2 个数量级 |

**口径说明（v3 重写，修 B3）**：
- **迭代数**：C2 口径值 **N_it = 1,415,000**（解析精确值 `Σ_{t=0}^{85}(128−⌊1.5t⌋) × H × B = 5,547 × 256 =
  1,420,032`，差 −0.35%；本版所有**聚合量**统一乘 1.415e6，避免二次误差累积）。C1 的
  `N_it = Σ_{t=0}^{127}(128−t) × 256 = 8,256 × 256 = 2,113,536`。
- **FLOPs**：`N_it × (scores/迭代 × 1664 + Mq × 2 × D_v)`，其中 `1664 = 2(3·192 + 2·128)` 是 manifest roofline
  公式的 GEMM 项（`§1.6.0` 上文引自 `ALG-attention` 的 FLOPs 口径）。C2 = 1.415e6 × (6144×1664 + 64×256)
  = 1.415e6 × 10,240,000 = **14.49 T**；C1 = 2,113,536 × (4096×1664 + 64×256) = 2,113,536 × 6,832,128 =
  **14.44 T**。解析下界（只数真实因果 score 位 8.59e9）= 14.34 T；C2 相对**解析**下界 +1.0%、相对 **C1**
  只 +0.35%，两者都来自 `⌈⌉` 分块量化的 score 槽位（C2 8.694e9 槽位 / C1 8.657e9 槽位 vs 真实 8.59e9）。
- **GM 访存（v3 修正 C2 的 281 GB → 190 GB）**：四项之和，逐项公式见下。
- **每 query 行 L1 填充次数**：query 行 `m` 被 key 块 `n0 ≤ m+shift` 读，故因果下平均
  `S_kv/(2·Nk) + 1`：C2 = 8192/192 + 1 = **43.7**、C1 = 8192/128 + 1 = **64.5**；
  **86 / 128 是非因果上界**（v2 把上界当均值，总量是对的、口径错）。每行字节数固定
  `q+qT+do+doT = 1280 B`（`2×(192+128)×2 B`）。

**R3 口径与出处（每个数字一行，v3 新增）**：

| 数字 | 公式 | 值 |
|------|------|----|
| C2 每迭代填充 | `q 24,576 + qT 24,576 + do 16,384 + doT 16,384 + lse/delta 512` | 82,432 B |
| C2 填充总量 | `1.415e6 × 82,432` | 116.6 GB |
| C2 常驻摊销 | `22,016 块 × 98,304 B`（`k_s+v_s+kT_s`） | 2.2 GB |
| C2 `dq` 原子 | `1.415e6 × 64×192×4`（每迭代整块 store） | 69.6 GB |
| C2 `dk/dv` 写 | `22,016 块 × (96×192 + 96×128) × 2 B` | 1.4 GB |
| **C2 GM 合计** | `116.6 + 2.2 + 69.6 + 1.4` | **189.8 ≈ 190 GB** |
| C1 填充总量 | `2,113,536 × 82,432` | 174.2 GB |
| C1 常驻摊销 | `32,768 块 × 65,536 B` | 2.1 GB |
| C1 `dq` 原子 | `2,113,536 × 64×192×4` | 103.9 GB |
| C1 `dk/dv` 写 | `32,768 块 × (64×192 + 64×128) × 2 B` | 1.3 GB |
| **C1 GM 合计** | `174.2 + 2.1 + 103.9 + 1.3` | **281.5 ≈ 282 GB** |
| C2 相对 C1 | `189.8 / 281.5 = 0.674` | **−33%**（主因 `N_it` −33%） |

**每迭代端口分解**（`Nk=96, Mq=64`，一次 (key 块, query 块) 迭代 = 6144 个 score；口径 = 该迭代的
L1/UB 端口流量，含 GM→L1）。**v3 修正**：v2 的 L1→L0 合计写成 221,184 B（那其实是 L0C 五累加器的容量，
串到了本行），而该行**逐项**相加只有 **217,088 B**——与本表的「合计」325,625 B 也是自洽的（v2 的合计用的
就是 217,088）。本版按逐项和改正：

| 项 | 组成 | 字节 |
|----|------|------|
| GEMM 操作数 L1→L0（10 条显式 `T.copy`） | `k_s 36864 + q_s 24576 + v_s 24576 + do_s 16384 + pT_s 12288 + doT_s 16384 + dsT_s(dK) 12288 + qT_s 24576 + dsT_s(dQ) 12288 + kT_s 36864` | **217088** |
| UB→L1 写（2 个 tile 半区） | `p16_u 6144 + dsT_u 6144` ×2（两个 AIV） | 24576 |
| GM→L1 装载（每迭代） | `q_s 24576 + qT_s 24576 + do_s 16384 + doT_s 16384 + lse/delta 512` | 82432 |
| GM→L1 常驻摊销 | `k_s + v_s + kT_s = 98304` ÷ 64.27 次迭代（`N_it / 22,016 块`） | 1529 |
| **合计** | `217088 + 24576 + 82432 + 1529` | **325625**（≈318 KiB） |

对应 `10 240 000 FLOP/迭代`（`6144×1664 + 64×256 = 10 223 616 + 16 384`；manifest 口径再含 `6144×5` 的
逐元素项，共 10,270,720）→ **31.4 FLOP/字节**（v1 为 `6,832,128 / 280,056 = 24.4`）。
全量 = `1.415e6 × 325,625 = 460.8 GB` L1/UB 端口流量（v1 = `2,113,536 × 280,056 = 591.9 GB`，
v2 曾写 590.6 GB，差 0.2% 属取整）。

> **口径防混淆**：C1 的「每迭代端口 280,056 **B**」与 R3 表的「GM 访存 281.5 **GB**」是两个独立口径
> （前者 = 每迭代 L1/UB 端口字节数，后者 = 全量 GM 读写总和），数值接近纯属巧合，不可互换。

**R4 硬件亲和性评估**（逐候选；容量对照 §4.5/§4.6；常数一律取本平台实测条目或本轮探针）：

| 候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|------|-------------|---------|------------|---------|------------|------|
| C1 = v1 | 5 个 NT GEMM（L1 路径）可跑，但向量侧被迫做跨步转置 | L0C 62.5% / L1 32.8% / **UB 47.2%**（`2×29,184 + 61,440 = 119,808 B`，v3 复算；v2 的 71.2% 无法由 §4.5 口径复现，已改） | 64 全为 16 倍数 | 全静态 | 单阶段 | ❌ **实测 Vector 受限**（cube 8.3% / vector 73.5%），scatter 占 63% |
| **C2 = 本版** | 5 个 GEMM 全显式 L0（**平台已核验形态**），四类 `trans` 组合齐全；逐元素落 AIV | L0C 84.4% / L1 59.4% / UB 65.7% / L0A 75% / L0B 93.75% | 96/64/192/128 全为 16 倍数 | `S_q/S_kv/B/H` 编译期常量 | GEMM 与向量 epilogue 同 kernel 交替（`__mix__(1,2)`），L0 单版本 | ✅ **原型实测 PASS，3.46 µs/迭代** |
| C3 = v1 + 两趟转置 | 同 C1，但第二趟跨步读 | 同 C1 | 同 | 同 | 同 | ❌ 实测第二趟代价 +1.64 µs/迭代（+80%）；C2 只多一次 12 KiB 的 MTE1 转置搬运（≈0.02 µs） |
| **C2″ = C2 + A5 旋转** | 同 C2 | **L0C/L0A/L0B 逐字节不变；L1 与 C2 同（311,296 B）；UB −36,864 B**（块尾缓冲之外不再给 `st_u/dp_u/p16_u/dsT_u` 第二版） | 同 C2 | 同 C2 |同 C2（`__mix__(1,2)` 不变；跨迭代的 RAW 边被版本切断）| **✅ 本算子探针实测 −36.9%（2.965→1.870 µs），`dk/dv` 逐位相同；代价 = 生成源 20,341→50,879 B（×2.50；绝对值仍小于 varlen v5 的 61,913 B，§9.1 风险 14）** |
| C4 = 两 kernel 拆分 | 同 C2 | 同 | 同 | 需两次 launch | — | ❌ FLOPs ×2，违反三 launch 契约 |
| C5 = 物化 | 纯 Cube 友好 | O(B·H·S²) ≈ 34 TB ≫ L2 | — | — | — | ❌ 容量淘汰 |
| C6 = 纯 Vector | 矩阵乘落错单元 | 同 | — | — | — | ❌ 无实测必要（量级差） |

**设计期定量 roofline（D-2；常数全部来自本平台实测条目 + 本轮探针，见 §1.6.4 表）**

> **v3 重算口径**：每项给「每迭代单价（单核）」与「全量（28 核聚合）」，两者互为校验
> `全量 = 单价 × N_it / 28`。聚合式必须含 `÷28` 或 `×28`——v2 的 fixpipe / UB→L1 两式漏了核数因子。

| 项（执行单元） | 每迭代单价公式 | 单价 (µs) | 全量公式 | 全量 (ms) | 依据 |
|----|------|------|------|------|------|
| 计算项（AIC mad） | `14.49 T ÷ 358 TF/s ÷ 1.415e6 × 28` | **0.801** | `14.49 TF ÷ 358 TF/s` | **40.5** | `CONST-950-gemm-mte2-burst`：bf16 Cube 有效峰值 356–359 TF/s（256×256 tile） |
| 实测口径 mad | `0.587 µs @(64,64) × 1.5`（96/64 的 score 比） | **0.880** | `0.88 × 1.415e6 ÷ 28` | **44.5** | 本轮探针 `gemm5`；隐含 ≈326 TF/s（比解析峰值低 9%） |
| 流量项（MTE2 GM→L1） | `82,432 B ÷ 70 GB/s` | **1.178** | `1.415e6 × 82,432 ÷ (70e9 × 28)` | **59.5** | `CONST-950-gemm-mte2-burst`：单核 70 GB/s（128 B 行段）→ 28 核 1.96 TB/s |
| 向量项（AIV epilogue） | `6144 × 0.201 ns` | **1.235** | `8.694e9 × 0.201e-9 ÷ 28` | **62.4** | 本轮探针 `epi0`：0.825 µs / 4096 score（2 AIV） |
| fixpipe 项（L0C→UB） | `49,152 B ÷ (92 GB/s × 2 AIV)` | **0.267** | `1.415e6 × 49,152 ÷ (92e9 × 2 × 28)` | **13.5**（v2 写 12.6，漏 `×28`） | 本轮探针 `l0c2ub` |
| UB→L1 项（MTE） | `24,576 B ÷ (52 GB/s × 2 AIV)` | **0.236** | `1.415e6 × 24,576 ÷ (52e9 × 2 × 28)` | **11.9**（v2 写 11.2，漏 `×28`） | 本轮探针 `ub2l1` |
| MTE1 项（L1→L0，10 条 copy） | `217,088 B`；L0 单版本 ⇒ 与 mad 在 **AIC 时间线相加** | ≤0.74（严格上界）/ 0.1–0.3（点估计） | `1.415e6 × 0.20 ÷ 28`（按点估计上限） | ≤10.1 | §1.6.4 G：由 v1 `gemm5` 折算的上界 + 未核验带宽假设 |
| `dq` 原子项（L0C→GM） | `49,152 B ÷ 53 GB/s/核`（**假设** 1.5 TB/s 聚合） | ≤0.91（未标定） | `69.6 GB ÷ 1.5 TB/s` | ≤46 | 结构性推算，**标为未标定假设**；Stage 3 采集 `asc_copy_l0c2gm` 耗时 |

**两个下界（口径必须分开写，v3 修正 B2）**：

- **管道级 `max` = 完全重叠下界**（各分项落在**不同执行单元**：AIC / MTE2 / MTE1 / FIX / AIV，
  流水正确时临界路径是它们中的最大值，**不是和**；**但 AIC 内部的 `MTE1 + mad` 是同一条时间线上的串行链**）：
  `max(0.88+MTE1[0.1–0.3] = 0.98–1.18, 1.178 填充, 1.235 向量, 0.267 fixpipe, 0.236 UB→L1) = 1.235 µs/迭代`
  ⇒ `1.415e6 × 1.235 ÷ 28 =` **62.4 ms**（Cube 占比 ≈ `0.88/1.235 = 71%`）。
- **串行合成**（把各段当不可重叠相加）= `1.18+1.235+0.267+0.236+(0.88+0.24) = 4.04 µs` ⇒ 204 ms；
  实测整链只有 **3.42–3.49 µs**（§1.6.4 D）⇒ 当前结构**已有约 14% 的段间重叠**，故「和」既不是
  下界也不是实测值（v2 把「三项之和」当物理下界，见 §12-B2）。

**结论**：本算子的地板由「每 query 行的 L1 填充（1.18）」与「每 score 的向量 epilogue（1.24）」中较大者
决定，Cube 计算项（0.80–0.88）单独达不到——这是与 v1「L1 端口是唯一地板」结论的关键修正。
但**这两项都落在与 AIC 不同的执行单元上**，因此它们不构成 Cube 在时间上的互补下界；「80% Cube MFU 可达性」
的正确表述见 §1.6.4 F（当前串行结构下不可达，跨迭代流水后原理上可及）。

**v4 的设计期 roofline 重算（D-2，用 Stage 4 的实测管道口径替换 v3 的探针外推）**：

| 项（执行单元） | v3 的探针/外推 | **Stage 4 实测（`ds-v3-b2-8k`，2.944 µs/迭代）** | v4 的结论 |
|---|---|---|---|
| 计算项 mad（AIC） | 0.88 | **0.769 µs（13.32 TF/s/核 = 峰值 13.5 的 98.7%）** | **不可压**（算法层已无计算量可省，§1.6.1） |
| MTE1（AIC） | 0.1–0.3（点估计）/ ≤0.74（上界） | **0.680 µs（217,088 B @ 300–340 GB/s）** | v3 要求的 `≤0.20` **实测不成立**（3.4×）；`k/v` 常驻化需 L0A 98,304 B > 65,536 ⇒ 不可行 |
| MTE2（AIC，GM→L1） | 1.178 | **0.720 µs（118.5 GB/s）** | 静态 extent 比 packed 快 47% ⇒ dense 不会重演 varlen 的「mte2 忙而慢」 |
| fixpipe（AIC） | 0.267 | **0.570 µs** | 与 mad 部分交错后仍有约 0.3–0.4 µs 暴露（v4 残差的主要来源，风险 17） |
| AIV 向量 | 1.235 | **1.30 µs（44.3%）** | 旋转后 **1.30 / 1.870 = 70%**，低于 80% |
| 五管道忙时之和 | — | **2.822 µs ≈ 墙钟 2.944** | **v3 结构下 AIC 内部零重叠**（这就是 A5 要修的东西） |
| **A5 后的地板（本版）** | — | — | **`max(AIV 1.30, mte1 + mad = 1.449)` = 1.449 µs 的 AIC 主链 + 未覆盖的 fixpipe 尾部**；实测 **1.870**（残差 0.42 µs，§6.3 F） |
| **A5 后的投影** | — | — | `1.870 × 1.420e6 ÷ 28 + 3.3 ms = **98.1 ms 的 `T_op`（区间 94.5–105）**，cube 38–40%（§14.3） |

口径纪律（沿用 v3）：每个聚合量必须给「每迭代单价 × `N_it` ÷ 28」或逐项和；**不同执行单元的分项取 `max`，
同一单元内的串行链才相加**（`mte1 + mad` 同属 AIC 且 L0 单版本）。**不再把「完全重叠的 max 下界」写成可达目标**
（v3 的 62.7 ms 就是这一错误，其前提 `MTE1 ≤0.20 µs` 已被实测否证）。

**调研结论**：选定**算法族 = key-stationary 单遍重算（flash 反向族 C2）**，关键依据：
R3 中 C2 与基线 FLOPs 相同（+0.3% 尾块量化）而 **GM 访存最低可达（189.8 vs 281.5 GB，−33%）**，
其余候选被淘汰（C4 2× FLOPs、C5 容量淘汰、C6 单元不匹配、C1/C3 多一次全 tile 向量 pass）；
R4 中 C2 的 5 个乘法在平台**已核验的 L0 形态**下全部成立（含 `trans_A`），且
`trans_A` 复用把 v1 的向量 scatter（占 63%）整项消掉。相对基线的结构差异：
**同一算法族，按 NPU 约束把 GEMM 全部落到显式 L0、累加器迁 L0C、把「转置副本」从向量段移到 MTE1**；
源算子 §0.4 的 14 项优化手段的意图承接见 §0.5 处置列（全部有承接或明确舍弃理由，无静默丢弃）。
**调度族的选择（v4 新增，属 R1 的结构类候选）**：在 C2 的算法族内，**迭代内语句顺序**是一个独立的设计
维度（R1 候选 #11/#12）。选定 **S1 = `GB` 滞后一迭代**（A5）：它把「AIC 等 AIV」的 RAW 边换成「读另一个缓冲
对象」，代价只有 L1 的 108 KB（两份装载缓冲），实测 **−36.9%**；**否决 S2（两级旋转）**：L0C 越界（103.1%）
且收益 ≈0（§6.3 E）。**该选择不改变算法族、不改变扫描遍数、不改变任何公式**，故 R2/R3 的结论不变。

下游约束：§1.4 与 §1.6.1–§1.6.3 的分析对象即 C2 + S1（C2″）；§3.1 用 §1.6.1 的优化后公式。

#### 1.6.1 数学等价优化（公式级）

分析对象 = §1.6.0 选定的 C2。逐项四要素如下，机器验证见本节末。

| # | 优化项 | 原式（基线） | 优化后公式 | 等价性论证 | 收益量化 |
|---|--------|-------------|-----------|-----------|---------|
| A1 | **自然对数域概率** | `P = exp2(s·scale·log2e − lse·log2e)` | `P = exp(s·scale − lse)` | `exp2(x·log2e) ≡ exp(x)` 是恒等式；两式对同一实数输入只差舍入路径。fp64 参照下两式各自距真值 ≤ 1 个目标 dtype ulp，候选式不劣于基线式 | 每 score 少 1 次乘（源式在 (i,j) 内逐元素重算的 `lse·log2e`）→ 每迭代 6144 次向量乘；`exp` 与 `exp2` 在本平台同属 SFU 指令（`T.exp` 有核验条目），不引入未核验 API |
| A2 | **块级 mask 特化** | 每个 (key 块, query 块) 都生成 `select(mask, p, 0)` | 块级谓词 `n0+95 < S_kv ∧ m0+63 < S_q ∧ n0+95 ≤ m0+shift` 为真时直接输出 `p`，否则同一 `select` | 该分支条件是掩码谓词在整块上恒真的**可判定充分条件**（`n≤m+shift` 对块内最大 `n` 与最小 `m` 成立即全真），故两式逐元素相同；机器验证对 6 组 shape 的全部 block 逐元素比对，违反率 0 | 快路块占**全方阵 49.5%**（`verify_equiv` 口径）/ **实际执行块 ≥98%**（`loop_st` 剪枝后：剪掉的对角块全部进掩码臂、保留块除对角一列外全走快路）；每 score 省 2 次整数比较 + 1 次选择 → 约 6144×2 次向量操作/迭代 |
| **A5** | **迭代内语句重排：`GB` 滞后一迭代**（v4 新增，**调度级**） | 体 = `F(k) → GA(k) → EP(k) → GB(k)`；`GB` 紧跟 `EP`，单版本缓冲，AIC 在 `GB` 前等 AIV 的 `PIPE_V` | 体 = `F(k) → GA(k) → EP(k) → GB(k−1)`；`F/EP` 产物（`q,qT,do,doT,lse,delta,pT,dsT`）取 `v(k)`，`GB(k−1)` 取 `v(k−1)=1−v(k)`；首迭代无 `GB`（`if kqA > loop_st`），末迭代由 drain 补（版本按 `n_it` 奇偶） | **(i) 逐元素恒等**：`GB(k−1)` 的 5 个操作数（`pT/dsT/doT/qT/kT`）与 §1.5 数据流图逐字相同，只是取址晚一迭代；`clear_accum` 由 `(kq == loop_st)` 改为 `(kq−1 == loop_st)`（首个 `GB` 仍恰一次清零；dense 下 `loop_ed = ceil(S_q/64) ≥ loop_st+1` 保证非空）；`dq` 的写回行区间 `[(kq−1)·64, +64)` 在同一 request 内不变。<br>**(ii)** `dV/dK` 的 fp32 累加顺序仍沿 request 内 query 块**升序**（`GB` 的次序按 k 递增，与 v3 相同）。<br>**(iii)** 唯一新增的同步是跨迭代的 `EP(k)→GB(k)`（AIV→AIC，1 迭代距离），由 2 版本缓冲把 `GB(k−1)` 与 `EP(k)` **解耦成不同的缓冲对象**，依赖边不存在。<br>**(iv)** `st_u/dp_u/p16_u/dsT_u/st_c/dpt_c` 的生存期不变（同迭代内生产-消费），**故不需要版本化**；`dk_c/dv_c/dq_c` 的单版本性不变。<br>**(v)** 4-D batch 视图、尾块掩码、`dq` 原子写回、块尾 `dk/dv` 写回**逐字不变**（dense 专属前提，§0.2/§5.4）。 | 每迭代消掉 `fixpipe→AIV epilogue→nd2nz/ub2l1` 的跨核串行：<br>• **本算子探针：B=2/S=8192/H=8 → 2.965 → 1.870 µs/迭代（−36.9%，生产同几何 v3 体 2.944 ⇒ 探针标定 +0.7%）**；B=2/S=4096/H=16 → **3.069 → 1.968（−35.9%）**；<br>• 同构族 varlen：3.167→2.092（−33.9%）/ 4.267→3.134（−26.5%）；<br>• `dk/dv` 与 v3 **逐位相同**（5/5 PARITY_PASS，§1.6.4 I-3）；<br>• 代价：生成源 20,341→50,879 B（×2.50）、`__simt_vf__` 3→5、`asc_mmad` 5→16（§9.1 风险 14）  **✅ 采纳（A5）**|

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

**弃选项的定量依据**（对应 R1 表 #4/#5/#6）：
- **#4 `scale` 折进 `q`/`k`**：代价是**每迭代 12,288 次 fp16 乘**（`2 × 64 × 192`，`q`、`k` 各一份）
  取代 **6,144 次 fp32 乘**（每 score 一次 `s*scale`）+ 块尾 18,432 次 fp32 乘（`96×192`），
  即乘次翻倍且多一次 fp16 舍入。**v3 重写**：v2 的「需两份副本」论据不成立——折进后 `dK = ds·q'` 与
  `dQ = dsᵀ·k'` 都自动带 scale，不需要额外副本；结论（不采纳）不变，理由改为上述定量口径。
- **#5 `delta` 拆 rank-1**：需额外一次同规模 GEMM（`192·64·96 = 1,179,648` MAC/迭代，对每迭代 GEMM
  总 MAC `5,120,000` 为 **+23.0%**；v2 写的 +5.7% 偏低，本版按公式改正）。
- **#6 `dV/dK` 合并**：会求和成 `pT·doT^T + dsT·qT^T`，**不等价**，直接排除。

**等价性机器验证（D-1）**：脚本 `examples/mla_dense_bwd/_bwd_kernel/verify_equiv.py`（torch CPU，fp64 参照，
含 `s·scale−lse ∈ [-8e2, 8e2]` 的随机样本 + 角点值：精确抵消、上/下溢、次正规、`lse=-inf` 空行、`700/−700`
极值；A2 覆盖 6 组 shape 含尾块与 `S_q<S_kv`/`S_kv<S_q`）。**revision v3 于 2026-11-25 重跑**
（`ASCEND_RT_VISIBLE_DEVICES=""`，cwd 在仓根之外，toolchain `ascend-950pr-9579 / TileLang ascend 0.1.15`），
输出 `EQUIV_SUMMARY ALL_PASS`（与 v2 逐字段一致）：

| 采纳项 | 样本数 | 候选 vs 基线 最大 ulp | 违反率 | 候选 vs fp64 最大 ulp | 基线 vs fp64 最大 ulp | 结论 |
|---|---|---|---|---|---|---|
| A1 exp 域 P（目标 dtype float16） | 1048592 | 1 | 0.000e+00 | 1 | 1 | EQUIV_PASS |
| A1 exp 域 P（目标 dtype bfloat16） | 1048592 | 1 | 0.000e+00 | 1 | 1 | EQUIV_PASS |
| A2 块级 mask 特化（block_n=64, block_m=64） | 76838912 | 0 | 0.000e+00 | — | — | EQUIV_PASS |
| **A5 旋转调度代数（版本分配 / drain 奇偶 / `clear_accum`；block_n=96, block_m=64）** | **71384** | — | **0.000e+00** | — | — | **EQUIV_PASS** |

（A2 快路块 9285/18761。A2 的谓词形状与 block 大小无关（只用到块的最大/最小下标），故 block_n=96 的分块
下结论不变；v2 未改动 A1/A2 的数学。）

**A5 的机器验证（两级：调度代数 + 设备逐位）**

A5 没有「候选公式」，只有**语句时序**，故 fp64 参照不能判它；v4 用两级机器验证（与 varlen v5 同一方法，
但**在本算子的 4-D / 静态 S / 无 slab 结构上重新取证**）：

**（i）调度代数（CPU，`verify_equiv.py` 的 `check_a5()`，可无设备复跑）**：把展开后的语句流
（`kqA = loop_st + 2i`、`kqB = kqA+1` + drain）逐事件模拟，覆盖 **478 个调度点 / 17,846 个 `GB` 事件**：
① dense 的**全部任务网格**（`(S_q,S_kv) ∈ {(512,512),(8192,8192),(4096,4096),(3072,3072),(96,192),(128,64),
(64,64),(96,96),(128,128),(100,100)}` × causal/非 causal × 每个 key 块 `n0`，`loop_st = max(0,(n0−shift)//64)`）；
② 合成扫描 `loop_st ∈ {0,1,2,5,7}` × `n_it ∈ 1..64`。检查五条不变式：每个 `k` 恰一次 `GB` 且递增 /
`GB(k)` 读的版本 == `EP(k)` 写的版本且未被更晚的 `EP` 覆盖 / 填充版本未被 `F(k+2)` 提前覆盖 /
`clear_accum` 恰一次且在 `GB(0)` / `F→GA→EP→GB` 的顺序。结果 **violations = 0，EQUIV_PASS**。
该检查**经三个反例自检**（守护其判别力）：drain 奇偶取反、删掉 drain、半体 A 读当前版本 ⇒ 三者全被抓到，
`selftest_caught=3/3`（与 varlen 首版探针踩中的那个坑同型）。

**（ii）设备逐位（与 v3 冻结体同进程对照，比 fp64 对照更强）**：
`history_version/design_probe_dense_rotation.py --parity`（同进程编译 `mode=base`（v3 体，
`T.Pipelined(num_stages=2)`）与 `mode=rot`（v4 体），同一输入、各自从零缓冲起跑，`torch.equal` 逐张量比对）：

| 用例 | 几何 | `dk` 逐位 | `dv` 逐位 | `dq` 最大绝对差 | 结论 |
|---|---|---|---|---|---|
| 单 key 块 / 单迭代 | B=1, S=96, H=1 | **相同** | **相同** | 0.0 | PARITY_PASS |
| 2 key 块（`n_it=2`，drain 走偶数分支） | B=1, S=128, H=1 | **相同** | **相同** | 0.0 | PARITY_PASS |
| 尾块（320 = 3×96+32） | B=1, S=320, H=2 | **相同** | **相同** | 0.0 | PARITY_PASS |
| **4-D B=2 + 尾块（512 = 5×96+32）** | **B=2, S=512, H=4, causal** | **相同** | **相同** | 2.98e-08 … 5.96e-08（≤1.8e-7 相对，**两次独立运行取值不同**） | PARITY_PASS |
| **4-D B=2 非因果** | B=2, S=512, H=4 | **相同** | **相同** | 2.98e-08 … 5.96e-08（≤1.8e-7 相对） | PARITY_PASS |

> **两次独立复跑**（2026-11-26，同一二进制）都是 `PARITY_SUMMARY ALL_PASS`、`dk/dv` 逐位相同；
> 只有 `dq` 的末位在 1.8e-7 相对量级内抖动——这正是 fp32 store-atomic 的到达序，而非版本差异。
> 完整输出：`history_version/dense_rot_probe_run.log`。

`dq` 的差异量级 = fp32 在 `|dq_acc|≈1e0–1e3` 下的 1 个 ulp，与**同一实现自身两次运行的抖动同量级**
（fp32 store-atomic 的到达序，v3 §8.2 已登记），不是结构差异。**`dk/dv` 逐位相同是 A5 等价性的决定性证据**：
若滞后一迭代把任何一个操作数取错版本，`dk/dv` 必然成片出错（varlen 首版探针正是这样暴露了 drain 的偶数
分支缺陷）。

**A5 的正确性前提（实现期必须守住，缺一即错）**：
1. **drain 必须无条件执行**：`GB(loop_ed−1)` 没有「下一个迭代」，版本由 `n_it` 奇偶决定（奇 → A，偶 → B）。
2. **`clear_accum` 跟着 `GB` 走**：`(kq−1 == loop_st)`；退化请求（`loop_ed == loop_st+1`）仍恰有一次 `GB`，
   `dk_c/dv_c` 仍被正确初始化。
3. **`kT` 仍是 `b_hi` 的最后一次重灌**（`doT → qT → kT`），与 v3 一致（§9.2）。
4. **dense 专属**：`GB(k−1)` 的 `dq` 写回行区间是 `[(kq−1)·64, +64)` 在同一 `b` 内——不得把 `b/h/n0` 的取值
   挪到 `GB` 内部重算（三者由 `bid` 一次性解出，与 v3 相同）。

**优化结论**：采纳 2 项公式优化（A1、A2）+ **1 项调度级重排（A5，v4 核心）**，全部 EQUIV_PASS；A5 的
`dk/dv` 与 v3 **逐位相同**。v4 **没有新增任何公式级优化**——`P`、`dS`、5 个乘积、`scale` 的落点、cast 次数
逐字不变；**v4 的全部收益来自 A5（语句顺序）**。这正说明「迭代内跨核串行」不能用数学变形解决：它不在
公式里，在语句顺序里——Stage 4 的 13 个 `num_stages`/版本标注分支全部落在 ±3% 噪声内（最好 −0.95%）
也是同一结论的反面证据。

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
| 10 | 尾块边界（有效行/列数与 clamp） | 源码 `T.if_then_else` 逐元素守卫 | **无需替代**：GEMM 用全 extent，GM↔片上搬运的越界尾部由 `AscendInsertOOBPadding` 改写成 `min()` clamp + `asc_fill_l1` 补 0；**前提是 GM 张量的 rank 让 clamp 维度等于本 request 的 `S`（即 4-D，§5.4 第 1 条）** | ✅（平台 pass 消除，但**依赖张量 rank**） | —（`pipeline.py` 该 pass 的 docstring 明确 "Clamp DMA copy OOB tails … emit GM->L1 padding fills"；v1 探针 B 设备码含 `min()` 与 `asc_fill_l1`；**v3 补充**：3-D 视图下 clamp 只能到 `B·S_kv`，故 rank 本身是正确性条件，见 §1.6.4 H） |
| 11 | `lse/delta` 行统计装载 | 逐元素 | `T.copy` 搬 (64,) fp32 | ✅ | — |
| 13 | **`n_half = (n_it+1)//2`、`kqA = loop_st+2i`、`kqB = kqA+1`**（v4 新增） | 旋转后的半体展开索引 | 无（保留为 `T.serial` 的循环变量与两处整数加） | ❌ | **tile 级循环控制**：每半体 2 次整数加、每块 ≤ ⌈n_it/2⌉ 次，不接触任何逐元素数据；且它是 `T.serial` 的循环界，逐元素 API 在语义上不适用（全 kernel 无「按元素算循环下标」的形态） |
| 14 | **drain 的奇偶判定 `(n_it % 2)`**（v4 新增） | 块级谓词，选 `pT_s0` 或 `pT_s1` | 无（保留一次 `%`/比较 + 一个 `if`） | ❌ | **block 级控制流**：每块**一次**，决定「哪一份静态缓冲被读」；向量化无意义（不是逐元素量），且版本必须由**静态缓冲名**给出（§1.6.3 D5） |
| 15 | **`clear_accum=(kq−1 == loop_st)`**（v4 新增） | tile 级 GEMM 属性谓词 | 无（保留为 `T.gemm` 的 `clear_accum` 参数） | ❌ | **tile 级顺序依赖**：它是累加器初始化语义的载体（`T.clear` 不支持 L0C，v3 探针 P4），不是可向量化的计算；谓词每块只求值一次/每个 `GB` 一次 |
| 16 | **版本分配（`_s0`/`_s1`）**（v4 新增） | 编译期静态缓冲名 | 无 | ❌ | **不是运行期计算**：两份缓冲在 trace 期就是不同对象，生成代码里没有任何运行期版本索引（这正是 A5 能成立的原因，§6.3 A2） |
| 12 | L1→L0 操作数暂存（v2 新增） | — | 每次 GEMM 前一次 `T.copy(L1 region, L0 region)`，由 MTE1 整块搬运 | ✅（DMA，非标量循环） | —（PL-950-gemm-tiles 的已核验形态；本轮原型设备码 `asc_copy_l12l0a/l12l0b`） |

**向量化结论**：16 个计算点中 **11 项**已向量化 / 换单元 / 由平台 pass 消除；**5 项（#9、#13–#16）**保留为
标量，理由类别分别是 **block 级任务映射**（#9）、**tile 级循环控制**（#13）、**block 级控制流**（#14）、
**tile 级顺序依赖**（#15）与**非运行期量**（#16）——**没有一项是「逐元素热点」**，且全部集中在 A5 新增的
块级头部/尾部（每块 O(1) 次求值）。v4 **没有引入任何逐元素标量循环**。相对 v1 的量化改进：
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

**决策 D5：版本分配与 2-tile 静态展开（v4 新增，阻塞级）**——「哪条缓冲需要第二份、由谁的名字给出」：

| 候选 | 结构 | 版本来源 | 评估 | 结论 |
|---|---|---|---|---|
| **D5-1（选定）** | `T.serial(n_half)` × 2-tile 静态展开：半体 A 用 `_s0`、半体 B 用 `_s1`，`GB(k−1)` 交叉读另一组 | **静态缓冲名**（trace 期两个不同对象） | 生成代码**无任何运行期版本索引**；L0 操作数 region 原点保持 0；代价 = 代码体积 +97% | ✅ |
| D5-2 | `T.Pipelined` + `annotate_buffer_versions({buf:2})`（v3 的做法） | 编译器按**当前迭代下标**重写索引 | **不可行**：生成代码把生产者与消费者**两侧**都写成 `(kq−loop_st)%2`，无法表达「滞后一迭代的消费者」；且本 kernel 上 Z3 `unsat` 后静默退化为拓扑序（§6.3 A） | ❌ |
| D5-3 | 单个 `T.serial(n_it)` + 运行期奇偶索引 `buf[kq % 2]` | 运行期下标 | **不可行**：L0 GEMM 的 region 原点必须为 0（`gemm_mad.py` 的 `require_zero_origin`），运行期下标的 L1 缓冲无法作为 L0 拷贝源；且会把 epilogue 复制成 2 份而非 4 份以外的不可控形态 | ❌ |
| D5-4 | 把 `st_u/dp_u/p16_u/dsT_u` 也给两版本（保守法） | 静态名 | **不必要**：它们的生产与消费都在同一迭代内（§4.3 版本判定表），实测 `dk/dv` 逐位一致已证明单版本正确；若实现期出现同步告警，这是**第一个回退旋钮**（UB 130,048 → 154,624 B = 60.9%，仍在容量内） | ⏸ 备用 |

**量化取舍**：D5-1 只增加 **L1 的 6 个装载缓冲各一份 + `pT_s/dsT_s` 各一份 = 106,496 B**（L1 204,800 → 311,296 B，
39.1% → 59.4%），**L0C/L0A/L0B 与 UB 的每迭代缓冲零增长**；换来的墙钟收益由本算子探针实测 **−35.9%**
（§1.6.4 I-4）。icache 是唯一确定代价（生成源 31,435 → 61,913 B，×1.97），列入 §9.1 风险 14 并带处置。

**主选 / 备选与实验裁决计划**：

- **主选（进 Stage 3，v4 = 旋转体）**：`block_n=96, block_m=64`。L0C 221184 B（84.4%）、
  **L1 311296 B（59.4%，6 个装载缓冲 + `pT/dsT` 各两份）**、**UB 130048 B（51.2%）**、L0A 49152 B（75%）、
  L0B 61440 B（93.75%）；每迭代端口 325625 B、31.4 FLOP/字节（**逐字节不变**）。
  **单价：v4 体 1.870 µs/迭代（B=2/S=8192/H=8 探针）vs v3 体 2.965**（§1.6.4 I-4；生产同几何
  v3 体 2.944 ⇒ 探针标定 +0.7%）→ 生产投影见 §14.3。
- **备选（≤1 个）**：`block_n=112, block_m=64` + `dq_c` 拆两半（(64,96) 各 24 KiB）。L0C
  `dk 86016 + dv 57344 + st 28672 + dpt 28672 + dq 24576 = 225280 B`（86.0%）；
  每 query 行填充摊到 112 key 行 → MTE2 总量 98.8 GB（vs 主选 116.6 GB，−15%），迭代数 1 198 000（−15%）
  （每字节 FLOP 与主选持平 ≈31，收益全来自迭代数）；
  代价：`dQ` 拆成两个 K 半程 GEMM（B 需两块 zero-origin 的 `kT` L1 tile，`kT_lo/kT_hi` 各 (96,112)=43 KiB，
  L1 增至 397 KiB = 76%），且两次原子 store。
  **裁决阈值**：备选 latency ≤ 主选 × 0.90 才翻转；按 shape 分片经 `lru_cache` 工厂分派（`S_kv ≥ 7168` 用备选）。
- **预置回退 `block_n=80`（v3 新增，回应「L0 单版本 MTE1 未量化」）**：
  `b_hi (192,80) = 30,720 B` ⇒ L0B = `24,576 + 30,720 = 55,296 B`（84.4%）、L0A = `30,720 + 10,240 =
  40,960 B`（62.5%）、L0C = `61,440 + 40,960 + 20,480 + 20,480 + 49,152 = 192,512 B`（73.4%）；
  每迭代 MTE1 从 **217,088 → 194,560 B（−10.4%）**，L0C/L0B 压力同步下降。
  **口径修正（v3）**：`Nk=80` **并不能**换来 L0 双版本——L0B 双版本需 `2 × 55,296 = 110,592 B > 64 KiB`；
  更一般地 `2×(24,576 + 384·Nk) ≤ 65,536` 只在 `Nk ≤ 21` 时成立，故**任何有用 tile 下 L0 都是单版本**
  （v2/检视建议里「留出 2 版本余量」的说法按容量算术不成立，本版改为如实登记）。
  回退的真实代价与收益：MTE2 总量 `116.6 → 139.9 GB`（`×96/80 = +20%`，`+0.236 µs/迭代`）对
  MTE1 `−22,528 B`（按 §1.6.4 G 的点估计 0.1–0.3 µs 折算 ≈ **−0.010…−0.031 µs**）⇒ **净亏**。
  因此**启用阈值**：仅当 Stage 3 由 `aic_mte1_active_bw` 反解出 **MTE1 暴露单价 > 0.5 µs/迭代**
  （当前上界 ≤0.74、点估计 0.1–0.3，§1.6.4 G）时才启用该回退；否则不动主选。
- **实验裁决计划**：代表 shape = smoke(512) 与 `ds-v3-4k`、`ds-v3-b2-8k`；指标 = kernel `Task Duration(us)` +
  msprof 的 `aic_cube_ratio` / `aiv_vec_ratio` / `aic_mte2_active_bw` / **`aic_mte1_active_bw`**；
  实测数据回写本节与 §1.6.4；
  Stage 4 的 A/B 清单 = ①**V=3（装载缓冲三份，L1 393,216 B = 75.0%）**——v5 族在长序列上收益小于短序列，
  可能是深流水的填充延迟未被覆盖，**未实测，不得写成收益** ②**`st_u/dp_u` 第二版**（UB 154,624 B = 60.9%，
  仅当出现 AIV→AIC 的 WAR 停顿证据时启用）③Nk/Mq 网格（96/64、112/64、128/48）④epilogue 优化阶梯
  ⑤`trans_B=False` 消 qT/doT ⑥**Nk=80**（阈值见上，主要用于压低 L0C 以换取更大 Mq）。
  **「跨迭代软件流水 / epilogue 旋转」已由 A5 落地，不再作为 A/B 项**。

#### 1.6.4 实测证据：v1 的瓶颈分解、v3 原型与 **v4 的旋转探针**（A–H 为历次证据，**I 为本轮**）⭐

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
| `chain_notr`：整链但去掉跨步写 | 2.196（复跑 2.190） | — |
| **`chain`：v1 内层整链** | **5.944–5.950**（设计期/复跑） | 100% |

**与生产实测对齐**：v1 在 `ds-v3-b2-8k` 上 T_op 455.3 ms / 2,113,536 次迭代 / 28 核 = **6.03 µs/迭代**，
探针复现 5.95 µs（差 1.3%）→ 该分解可信。（A 表的分段和 `1.176+0.587+0.173+0.231+0.825 = 2.992 µs`
**低于**整链去 scatter 的 `chain_notr = 2.196 µs`，说明段间已存在重叠 —— 这正是 §1.6.4 F 不再把
「分项和」当下界的实测依据；v3 复跑 `chain_notr` = 2.190 µs。）

**B. 由 A 得到的根因（回答「为什么 5 个 GEMM 没让 Cube 成为瓶颈」）**

1. **向量侧的跨步转置 scatter 独占 63%**：`ds_u[j,i]` 按 lane=query 写一个 `(Mq,Nk)` 的 f16 tile，
   步长 32 个 f16，bank 冲突 + 循环整体退化（不能向量化），单这一句 = 3.75 µs/迭代。
   它是 v1 为「L1 路径 NT-only」付出的代价（§0.4 #6 的误判）。
2. **Cube 本身并不慢，它只是在等**：探针的直接证据是 `gemm5 = 0.587 µs`，只占整链 `5.95 µs` 的 10%
   （v1 的 `aic_cube_ratio = 8.3%` 与它同量级）。**v3 口径修正**：v2 另写的
   「`8.3% × 455.3 ms = 37.8 ms ≈ 14.34 T ÷ 378 TF/s`」隐含 379 TF/s，**高于**平台实测峰值
   356–359 TF/s（`CONST-950-gemm-mte2-burst`），该换算超出证据约 6%，已删除；结论仍由 `gemm5` 支撑。
   空闲来自**串行依赖 + 无重叠**（`num_stages=1`，每迭代「装载 → GEMM → fixpipe → 向量 → UB→L1 →
   GEMM → 原子写」全串行）。
3. **L1 填充已与 Cube 同量级**：`mte2` 1.176 µs vs `gemm5` 0.587 µs（2×）。单核 MTE2 实测 70 GB/s
   （`CONST-950-gemm-mte2-burst`：128 B 行段口径），每 query 行 `1280 B × 因果均值 64.5 个 key 块`
   = 82.6 KB/行的放大（v3 修正：v2 用非因果上界 128 次 ⇒ 164 KB，总量对、口径错）。
   C2 的对应值是 `1280 B × 43.7 = 55.9 KB/行`。
4. **tile 太小**：`Nk=64` 把 1280 B 的每行填充摊到 64 个 key 行上；L0C 此时只用 62.5%，本可放大。

**C. v1 的根因归属（为什么 v1 没做到）**：v1 的设计目标错位——它把「L1 端口流量」当作唯一地板项
（§1.6.0 旧结论「L1 端口操作数流是地板项」），因此用「缩小 tile + 单阶段 + 转置副本」换取 L1 占用；
而实测地板项是**向量 scatter + 每行填充 + 无重叠**三者叠加，其中最大的一项来自 v1 自己引入的向量段。

**D. 新结构（C2）的端到端原型实测**

| 测试 | shape | 结果 |
|------|-------|------|
| 数值（causal） | B=1, S=384, H=4 | **PASS**（dq/dk/dv 最大差 7.8e-3 / 7.8e-3 / 3.1e-2，bf16 口径） |
| 数值（非 causal） | B=1, S=384, H=4 | **PASS**（3.9e-3 / 2.0e-3 / 3.9e-3） |
| 单价（causal，3-D 视图） | B=1, S=2048, H=8，2904 次迭代 | **3.446 / 3.477 / 3.486 µs/迭代**（`num_stages=1`，三次独立跑） |
| 单价（causal，**4-D 视图**，v3 主选签名） | 同上 | **3.461 / 3.476 / 3.509 µs/迭代**（`num_stages=1`）→ 与 3-D **同值**（差 ≤0.6%，签名代价为零） |
| 单价（causal，4-D + `num_stages=2`） | 同上 | **3.418 / 3.421 µs/迭代**（−1.6%，与 v2 的 3.391 同带） |
| 数值（4-D + `num_stages=2`，causal/非 causal 各 2 次） | B=2, S=512, H=4 | 全部 **PASS**（dk ≤1.03e-2 / dv ≤1.77e-3） |

外推主导 case：`1.415e6 × 3.469 ÷ 28 = 175.3 ms`（用实测 3.42–3.49 的带 ⇒ **173–176 ms**；相对 455.3 ms =
**2.6×**）。其中 cube 理论 0.88 µs/迭代（44.5 ms 实测口径 / 40.5 ms 解析口径）→ 迭代内占比 **25%**
（v1 为 8.3%）。**跑间波动 ±2%**：同一配置的 3.446–3.486（stages=1）/ 3.391–3.421（stages=2）都是同分布
取样，不据此下「stages=2 有收益」的结论（§6.3）。

**E. 结构层的两条硬约束（本轮实测，写进 §0.5/§0.6/§9）**

1. **L0 路径与 L1 路径不能在同一 kernel 内混用**：把 `dQ` 单独改 L0、其余 4 个留 L1 时，`dk/dv` 出现
   4.8e-1 / 3.1e-2 的错误（`ascend_gemm_l1` 模板内部会用 L0A/L0B，与用户显式分配的 L0 缓冲互相覆盖）；
   5 个 GEMM 全部改显式 L0 后同一测试全 PASS。
2. **`T.copy(UB, L1, transpose=True)` 不可编译**（`Ascend device compilation failed`）：UB→L1 的转置
   不是可用形态，故「把转置挪到 UB→L1 搬运」这条路被排除；可用形态只有 `trans_A` 的 L1→L0A 转置
   （`asc_copy_l12l0a_transpose`）。

**F. 「80% Cube MFU」的可行性边界（v3 重写，修 B2）**

> ⚠️ **本节是 v3 的推导，其结论已被 Stage 4 实测推翻，保留为归因证据（不要当成本版口径）**：
> ① 表中的 `mad 0.88` 实测为 **0.769**；② 表中 `MTE2 1.18` 实测为 **0.720**（119 GB/s）；
> ③ 本节把 `MTE1 ≤0.20 µs` 列为「80% MFU 的硬条件」，Stage 4 实测 **0.60–0.68 µs**（已是可达带宽），
> **该条件不成立**，故「80% MFU」在任何流水下都不可达（仅 `mte1+mad = 1.449 µs` 就给出 73.2 ms 下限）。
> **v4 的口径以 §1.6.0 的「v4 roofline 重算」表与 §1.6.4 I 为准。**

> **v2 的推导错在哪**：v2 把「Cube 0.88 + 填充 1.18 + 向量 1.24 = 3.30 µs」当作**物理下界**，并写
> 「即使把搬运与同步全部优化掉也不可达」。这三个分项落在**不同执行单元**（AIC / MTE2 / AIV），
> 若流水正确则临界路径是它们的 **max**（1.24 µs），而不是**和**；只有「串行不可重叠」时和才是下界，
> 而这与 §2.2/§6.3 声称的「AIC/AIV 并发」自相矛盾。设计自己的实测也否定可加性（§1.6.4 A：
> `chain_notr 2.190 < Σ分段 2.992`）。本版按管道级口径重写：

| 管道 | 当前结构单价 (µs/迭代) | 出处 |
|------|----------------------|------|
| AIC mad（5 GEMM） | **0.88**（实测口径 `0.587×1.5`）/ 0.80（解析 `14.49 T ÷ 358 TF/s`） | §1.6.4 A + `CONST-950-gemm-mte2-burst` |
| MTE2（GM→L1 填充） | **1.18** | `82,432 B ÷ 70 GB/s` |
| AIV（epilogue） | **1.24** | `6144 × 0.201 ns` |
| FIX（L0C→UB） | 0.27 | `49,152 ÷ (92×2) GB/s` |
| MTE（UB→L1） | 0.24 | `24,576 ÷ (52×2) GB/s` |
| MTE1（L1→L0，10 条 copy） | **≤0.74**（严格上界，§1.6.4 G）/ 点估计 **0.1–0.3** | `217,088 B`；L0 单版本 ⇒ 与 mad 在 AIC 时间线上**相加** |
| `dq` 原子（L0C→GM） | ≤0.91（**假设** 1.5 TB/s 聚合，未标定） | 结构性推算 |

| 口径 | 值 | 折算 T_op |
|------|----|----------|
| **实测（当前串行迭代结构）** | 3.42–3.49 µs/迭代 | **173–176 ms** |
| 串行合成（各段不可重叠相加，非下界） | `0.88+1.18+1.24+0.27+0.24+MTE1(0.24) = 4.05` µs | 205 ms（比实测高 16% ⇒ 已有重叠） |
| **完全重叠下界（当前算子集）** | `max(mad 0.88 + MTE1, 1.18, 1.24, 0.27, 0.24) = 1.24` µs（当 MTE1 ≤0.36） | **62.7 ms**（Cube 占比 71%） |
| **两项改进 + 跨迭代流水** | `max(0.80 + MTE1, 0.59 填充, 0.62 epi, 0.27, 0.24) = 0.80 + MTE1` µs | **40.5 ms**（MTE1→0）～50.6 ms（MTE1=0.20） |
| 80% Cube MFU 所需 | ≤1.00 µs/迭代 | 50.6 ms（`14.49 T ÷ (0.8×358 TF/s)`） |

**结论（v3 定稿；v4 已撤销其「跨迭代流水后原理上可及」的后半句，见上方警示）**：「80% Cube MFU」在**当前串行迭代结构下不可达**（实测 3.42–3.49 µs/迭代 ⇒ 173–176 ms，
Cube 占比 25%）；但它在**原理上可及**——完全重叠下界 1.24 µs（62.7 ms），两项改进后 `0.80 + MTE1` µs
（MTE1 ≤0.20 时 ≤1.00 µs ⇒ ≤50.6 ms）。v2 的「即使把搬运与同步全部优化掉也不可达」一句**删除**。

**达成 50.6 ms（80% MFU）的充分条件清单**（缺一不可，Stage 4 逐项 A/B）：

1. **填充减半**：`trans_B=False` 让 `qT/doT` 不再单独装载（B 侧由 L0B 的转置消费或改用 `(K,Mq)` 布局），
   MTE2 `82,432 → 41,472 B` ⇒ `1.18 → 0.59 µs`；
2. **epilogue ≤0.62 µs**（0.10 ns/score，§1.6.3 的三级阶梯至少落地两级：`float4` 加宽 / `simd.vexpdif` / 手工展开）；
3. **跨迭代软件流水**：`q/do/qT/doT` 与 `st_u/dp_u/p16_u/dsT_u` 显式双缓冲 + 预取 i+1，使 AIC 的 mad 不与
   同一迭代的填充/epilogue 串行（这是当前 3.45 µs 与 1.24 µs 的差距来源）；
4. **MTE1 暴露 ≤0.20 µs/迭代**（**新增的硬条件**，§1.6.4 G）：L0 单版本使
   `AIC 时间线 = mad + MTE1`；`0.80 + MTE1 ≤ 1.00` 才可能 80% MFU，即 MTE1 ≤0.20 µs
   （等价于每核 L1→L0 有效带宽 ≥ `217,088/0.20e-6 = 1.09 TB/s`）。L0 双版本在本平台 L0B 容量下不可行
   （`2×(24,576+384·Nk) ≤ 65,536` 仅 `Nk ≤ 21`），故这条不能靠双缓冲绕过；
5. **fixpipe/UB→L1/`dq` 原子三项各自 ≤0.80 µs**：0.27 / 0.24 已满足；`dq` 原子的 0.91 µs 是**唯一未标定项**，
   若 Stage 3 实测超过 0.80，则 80% MFU 需先把 `dq` 换成 half-store + 后置归约或降低原子量（备选 112/64 可降到 59 GB）；
6. **AIC/AIV/MTE2/MTE1/FIX 五条管道真并发**：由 `__mix__(1,2)` + InsertSync 提供，Stage 4 须用
   msprof 的 `aic_cube_ratio ≥ 60%` 与 `aiv_vec_ratio` 同时确认（只有 cube 高不代表并发成功）。

**因此本版的设计目标（v3 两档并列）**：

- **Stage 3 交付**：**≤227 ms**（实测原型外推 173–176 ms，1.3× 余量）。
- **Stage 4 目标档 A（无跨迭代流水，按 §1.6.3 阶梯 + §6.3 显式双缓冲落地）**：**≤115 ms / cube ≥40%**。
- **Stage 4 目标档 B（实现跨迭代软件流水后的理论区间，Stage 4 A/B 裁决）**：**≤62 ms / cube ≥60%**；
  若档 B 的 A/B 全部通过，则 80% MFU（≤50.6 ms）进入可达区间，但**不作为验收承诺**。

这与调度方给的目标（≥2× 必达、接近 Cube roofline 尽力）一致：455.3 → 173–176 ms 的路径**已实测**；
176 → 115 ms 的路径**已定位 + 有阈值**；115 → 62 ms 的路径**有管道级下界与充分条件清单**。

**G. L0 单版本下 MTE1 的量化（v3 新增，回应「221 KB MTE1 全串行未量化」）**

- **流量**：每迭代 10 条显式 `T.copy(L1→L0)` 共 **217,088 B**（逐项见 §1.6.0 端口分解）。L0 单版本
  ⇒ 结构上是 `copy→mad→copy→mad`，故在 **AIC 时间线上 MTE1 与 mad 相加**：
  `T_AIC = Σcopy + Σmad = MTE1 + 0.88 µs`。这一点在 v2 中未量化，直接导致 §6.3 的双缓冲收益被高估
  （双缓冲只能省 `min(填充, mad)`，省不掉 `MTE1 + mad` 这条串行链）。
- **严格上界**：v1 的 `gemm5 = 0.587 µs` 是「5 个 NT GEMM（操作数常驻 L1）」的整段耗时，L1 路径由
  `ascend_gemm_l1` 模板在内部完成 L1→L0 搬运 ⇒ 该值**已包含** v1 几何（172,032 B）的全部 MTE1。
  按流量比折算到 C2（217,088 B，×1.26）：**MTE1 ≤ 0.587 × 1.26 = 0.74 µs/迭代**（严格的、可复核的上界）。
- **点估计 0.1–0.3 µs**：同几何的 mad 解析值为 `6,832,128 FLOP ÷ (358 TF/s ÷ 28) = 0.534 µs`，
  与 `gemm5 = 0.587` 只差 10% ⇒ 该模板内 MTE1 与 mad **高度重叠**（模板自身对 L0 做了流水），
  故 v1 的 MTE1 实际 ≈0.05 µs 量级；C2 的流量多 26%、但 L0 缓冲由用户显式管理（无模板内流水），
  按「无重叠」重算的 217,088 B 若达 1.0–2.0 TB/s 的 L1→L0 带宽即落在 **0.11–0.22 µs**。
  （平台 `knowledge/constants.md` 与 `bottlenecks.md` 中**没有 MTE1 带宽条目**，本估计标为「未核验假设」。）
- **敏感性（决定 80% MFU 的关键量）**：

  | MTE1 (µs/iter) | AIC 时间线下界 `MTE1+0.80` | 完全重叠下界（含 AIC 链） | 80% MFU（≤1.00 µs） |
  |---|---|---|---|
  | 0.10 | 0.90 | `max(0.90, 1.24)=1.24` | 可及（≤1.00） |
  | **0.20** | **1.00** | 1.24 | **临界**（恰好 1.00） |
  | 0.40 | 1.20 | 1.24 | 不可及 |
  | 0.74（上界） | 1.54 | 1.54 | 不可及 |

  ⇒ **80% MFU 的必要条件是 MTE1 ≤0.20 µs/迭代**（L1→L0 有效带宽 ≥1.09 TB/s）；这已写进 §1.6.4 F 清单第 4 条。
- **Stage 3 标定方法**：msprof 采集 `aic_mte1_ratio` / `aic_mte1_active_bw`，用
  `217,088 B ÷ aic_mte1_active_bw` 反解单价并回写本节；**回退阈值**：实测 >0.5 µs/迭代时启用 §1.6.3 的
  `Nk=80` 回退（MTE1 流量 −10.4%），并重新评估 `trans_B=False` 与 `kT` 拆块的优先级。

**H. `dk/dv` 块尾写回的 request 隔离（v3 新增，修 B1）**

探针 `history_version/design_probe_b1_tailrace.py`：同一份 C2 结构，仅 GM 张量签名/寻址不同
（3-D packed vs 4-D batch），B=2, S=512, H=4（`512 = 5×96 + 32` ⇒ request 0 的尾块 n0=480 越界写
64 行进 request 1），每个变体各跑 5 次（`seed=0..4`）：

| 变体 | causal | dk 最大差 | dv 最大差 | **request 1 key 0…63**（受害区） | 结论 |
|---|---|---|---|---|---|
| 3-D（v2 形态） | True | 1.08e-2 | 1.60e-2 | 1.08e-2 / 1.60e-2 | 5/5 数值 PASS（**侥幸**：尾块只需 1 次迭代、总在 victim 之前完成，见下） |
| 3-D（v2 形态） | **False** | **3.87e-1** | **3.99e-1** | **3.87e-1 / 3.99e-1** | **5/5 FAIL**（非因果下两块迭代数相同 ⇒ 竞态暴露） |
| **4-D（v3 形态）** | True | 1.08e-2 | 1.60e-2 | 1.08e-2 / 1.60e-2 | **5/5 PASS**（与 3-D 同值） |
| **4-D（v3 形态）** | **False** | **2.09e-3** | 1.77e-3 | **1.38e-3** / 1.47e-3 | **5/5 PASS** |
| 4-D + `num_stages=2` | True/False | ≤1.03e-2 | ≤1.77e-3 | ≤1.38e-3 | 2/2 PASS（§4.8 选定配置的数值回归） |

**三点结论（写进 §5.4 与 §8.3）**：

1. **3-D 视图确实越界**：非因果下 5/5 失败，误差全部落在 request 1 的 key 0…63（受害区误差 = 全张量误差），
   证明写回行数按整张 `B·S_kv` 计算。**因果下 5/5 侥幸通过**——这恰好印证「单次运行不可判定」：
   bottom-right 因果下尾块的 query 迭代数最少（`loop_st = (n0−shift)/64` 最靠后），总是先于 victim 完成，
   0 值被 victim 的真实值覆盖；任一派发顺序/抢占/异步 store 延迟变化都会翻转这个结论。
   **所以不能用「实测通过」来否证该缺陷**（v1 的 `_OOB_PAD` 哨兵只查整张张量尾部，同样测不出）。
2. **4-D 视图是根治性的**：读、写两侧的 clamp 都变成按 request（设备码见下），因果/非因果都 5/5 通过，
   且数值与 3-D 逐位同值（causal 两组完全相同）⇒ 语义未变。
3. **4-D 签名零代价**：单价 3-D `3.446 / 3.477 / 3.486` vs 4-D `3.461 / 3.476 / 3.509` µs/迭代
   （三次独立取样，同机同 shape）⇒ 差 ≤0.6%，在跑间 ±2% 波动内。

**设备码证据**（编译产物 `device_kernel.asc`，B=2/S=512/H=4，保存在
`history_version/b1_device_code_bounds.txt`）：

| 侧 | 3-D 视图 | 4-D 视图 |
|---|---|---|
| `dk` 写回行数 | `min(48, max((**1024** − sid·48) − (bid%6)·96 − ((bid%12)/6)·512, 0))` ← 上界是 `B·S_kv = 1024` | `min(48, max((**512** − sid·48) − (bid%6)·96, 0))` ← 上界是 `S_kv = 512` |
| `k` 装载行数 | `min(96, (1024 − (bid%6)·96 − ((bid%12)/6)·512))` | `min(96, (512 − (bid%6)·96))` |

⇒ 3-D 下尾块（`n0=480, sid=1`）写满 48 行（全局 528…575，其中 512…575 属 request 1）；
4-D 下同一处算出 `min(48, 512−48−480) = 0` 行、`sid=0` 只写 `512−480 = 32` 行
——**与源算子 `bwd.py:255-260` 的逐行守卫逐行同义**。

---

**I. v4（A5 旋转）的本轮实测证据 ⭐**

> 脚本：`history_version/design_probe_dense_rotation.py`（自包含：`mode=base` 复刻 v3 冻结体
> `T.Pipelined(num_stages=2)`、`mode=rot` 复刻 v4 体（`T.serial` + 2-tile + drain）、按 §8.1 语义独立写的
> fp32 逐 request 参考、`--compile/--numeric/--parity/--time` 四种入口）。
> 运行方式与 §3.5.4 相同（cwd = 仓根之外、`TMPDIR=/workspace`、`/workspace/venv-ascend950/bin/python`、
> 单卡，toolchain `ascend-950pr-9579 / TileLang ascend 0.1.15 / CANN 9.2.1`）。

**I-1 编译（`--compile`）**——两个体都能编译；`rot` 仍会打印 Z3 `unsat` 回退，但**v4 不依赖 Z3**
（版本化由静态缓冲名给出）：

| 体 | 结果 | 生成源 (B) | `asc_mmad` | `__simt_vf__` / `asc_vf_call` | sync wait/notify | `asc_copy_l12l0a` | `nd2nz_scatter` |
|---|---|---:|---:|---:|---:|---:|---:|
| `base`（v3 体，`T.Pipelined(2)`） | COMPILE_OK | **20,341** | 5 | 3 / 3 | 29 / 29 | 5 | 2 |
| **`rot`（v4 体）** | COMPILE_OK | **50,879（×2.50）** | 16（半体 5×2 + drain 3×2 奇偶） | **5 / 5** | 76 / 76 | 16 | 4 |

`asc_mmad` 5→16 与 `__simt_vf__` 3→5 是「两份体 + 三个 drain GEMM 的奇偶分支」的预期结果
（`5+5+3+3 = 16`、`2 臂 × 2 半体 + 块尾 cast = 5`）。**代码体积是 A5 的真实代价**：绝对量 50,879 B
仍**小于**已在生产运行的 varlen v5 的 61,913 B，故 icache 暴露被一个已交付案例上界住（§9.1 风险 14）。

**I-2 数值（`--numeric`，`mode=rot` vs fp32 逐 request 参考，判据 3e-2）**

| 用例 | 几何 | `dk` | `dv` | `dq` | 逐 batch 最差 | 结论 |
|---|---|---|---|---|---|---|
| 单 key 块 / 单迭代 | B=1, S=96, H=1 | 4.36e-09 | 0.00e+00 | 5.96e-08 | 5.96e-08 | **PASS** |
| 2 key 块（`n_it=2`） | B=1, S=128, H=1 | 0.00e+00 | 0.00e+00 | 6.70e-08 | 6.70e-08 | **PASS** |
| 尾块 | B=1, S=320, H=2 | 1.77e-04 | 0.00e+00 | 1.67e-05 | 1.77e-04 | **PASS** |
| **4-D B=2 + 尾块** | B=2, S=512, H=4, causal | 4.43e-04 | 1.78e-04 | 2.32e-04 | 4.90e-04 | **PASS** |
| **4-D B=2 非因果** | B=2, S=512, H=4 | 2.34e-03 | 2.54e-03 | 3.38e-04 | 2.72e-03 | **PASS** |

（误差 = `max|diff|/max|ref|`；最差 2.72e-03，比判据低一个量级。**逐 batch 切片**比对，故 4-D 视图下
没有跨 request 泄漏。）

**I-3 逐位对照（`--parity`，base vs rot 同进程、同一输入、各自从零缓冲起跑）**——见 §1.6.1 的表：
**5/5 PARITY_PASS（两次独立复跑同结论），`dk`/`dv` 逐位相同**（`torch.equal`），`dq` 的差为 fp32 原子到达序
（≤1.8e-7 相对，两次运行末位不同）。完整输出：`history_version/dense_rot_probe_run.log`。

**I-4 单价（`--time`，每形状 3 次取最小，`per-iter = T×1e3/iters×28`）**

| 几何 | iters | v3 体 | **v4 体** | 变化 | 与生产对账 |
|---|---:|---:|---:|---:|---|
| `B=2, S=8192, H=8`（与 `ds-v3-b2-8k` 同 S/B，**同 per-block 迭代数 64.5**） | 88,752 | 2.965 µs | **1.870 µs** | **−36.9%** | **v3 体 = 生产 2.944（+0.7%）⇒ 该形状可直接外推** |
| `B=2, S=4096, H=16` | 45,056 | 3.069 µs | **1.968 µs** | **−35.9%** | v4 体在同族 varlen 上的对应值为 2.092（本算子更低：AIV 更便宜） |

**I-5 为什么两级旋转（把 `GA` 也提前）不再有收益**：v4 体在 `S=8192` 上 1.870 µs，已**低于**
`mte1 + mad + fixpipe = 0.680 + 0.769 + 0.570 = 2.019 µs` 的「三段相加」口径——说明旋转后这三段之间
**也已发生部分重叠**（fixpipe 与相邻 mad 交错），墙钟由 AIC 的 `mte1+mad` 主链（1.449 µs）加上未覆盖的
fixpipe 尾部构成。再提前 `GA` 只能去压已被压过的部分，代价却是确定的：`st_c/dpt_c` 双版本使 L0C 到
**270,336 B（103.1%，越界）**，必须把 `dP` 并入 `st_c` 并在 `GA` 内部插入 fixpipe→mad 串行（§6.3 E）。

**I-6 这一版没有解决的问题（如实记录）**：
- 1.870 µs 仍比 `mte1 + mad = 1.449` 高 29%：残差是 fixpipe（0.570）没有被完全藏掉 + 块级固定开销
  （每块 `k/v/kT` 常驻装载 98,304 B 与块尾 cast/写回）。**它的分解未实测**（需要 msprof 的 AIC 管道
  时间线），列入 §9.1 风险 17。
- `n_it ∈ {1,2}` 的短块（`S_q` 小或对角块）只有 1–2 次迭代，首/末迭代的 `EP` 不与 `GB` 重叠。
  量化：每块 2 次不重叠迭代，`ds-v3-b2-8k` 上 22,016 块 × 2 次 × ~1.0 µs（可重叠部分）÷ 28 核
  ≈ **1.6 ms ≈ 1.7% 的投影时间**（已计入 §14.3 的区间，不是额外风险）。
- 探针的 grid 是 1,376 块（49 波），生产是 22,016 块（786 波）；尾波效应在生产上更小，故 1.870 应视为
  **生产单价的上界**。

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
**v4 的时序注（A5）**：上表是**数学分解**，与 v3 逐字相同。执行时步骤 1–4 作用于迭代 `k` 的数据
（`st_c/dpt_c` → `EP(k)` → `pT_s[v(k)]/dsT_s[v(k)]`），而**步骤 5–8 作用于迭代 `k−1` 的数据**
（读 `pT_s[v(k−1)]/dsT_s[v(k−1)]/qT_s[v(k−1)]/doT_s[v(k−1)]`），即在 `EP(k)` 之后与它**并列**发起。
`clear_accum` 因此从 `(kq == loop_st)` 变为 `(kq−1 == loop_st)`。**这只改语句顺序，不改任何一步的算式**
（§1.6.1 A5）。

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API（`tilelang.ascend.language`） | 参数 | 模式 |
|------|----------|--------------------------------------------|------|------|
| 0 | 任务映射 | `T.Kernel(grid)` 一维核网格 | `grid = H*B*ceil(S_kv/96)`；`T.Kernel` 不接受 `threads` | — |
| 1 | `k/v/kT` 常驻 | `T.copy(k[b, n0:n0+96, h, :], l1_buf)`（`kT` 加 `transpose=True`）；dst 用全 allocation extent | `k_s (96,192)`、`v_s (96,128)`、`kT_s (192,96)`；**GM 侧 4-D batch 视图** | MTE2 |
| 2 | `q/do` 双方向装载 | 同上（两份：ND 与 `transpose=True`） | `q_s (64,192)`、`qT_s (192,64)`、`do_s (64,128)`、`doT_s (128,64)` | MTE2 |
| 3 | `lse/delta` | `T.copy(lse[b,h,m0:m0+64], lse_u)` | 一维 fp32 区段 | MTE2 |
| 4 | L1→L0 暂存 | `T.copy(l1_buf_region, l0a_or_l0b_region)`（`trans_A=True` 的 A 由编译器生成转置搬运） | 每次 GEMM 前一次；region 原点必须为 0 | MTE1 |
| 5 | 5 个 GEMM | `T.gemm(a_l0a, b_l0b, c_l0c, transpose_A=…, transpose_B=…, clear_accum=…)` | 见 §3.1；A/B 在 L0A/L0B，C 在 L0C | Cube |
| 6 | L0C→UB | `T.dual_copy(st_c, st_u)` / `T.dual_copy(dpt_c, dp_u)` | 2:1 M 分片（48×64 每 AIV），fp32 同 dtype | AIC fixpipe |
| 7 | `exp/mask/dS/cast` | `with T.SimtVF(threads=128): for i, j in T.Parallel(48, 64)` | `T.exp`、`T.if_then_else`、`T.cast`；A2 为运行时块级 `if` 两支 | Vector |
| 8 | UB→L1 | `T.dual_copy(p16_u, pT_s)`、`T.dual_copy(dsT_u, dsT_s)` | 2:1 M 分片；由 `InsertNd2Nz` 改写为 NZ scatter | MTE |
| 9 | `dq_acc` 累加 | `T.set_atomic("add","float32")` → `T.copy(dq_c, dq_acc[b, m0:m0+64, h, :])` → `T.set_atomic_none()` | L0C fp32 → GM fp32；4-D 视图裁剪到本 request 的 `S_q` | AIC fixpipe |
| 10 | `dk/dv` 写回 | `T.dual_copy(dk_c, tail_u)` → 向量 `cast`（`dk` 乘 `scale`）→ `T.copy(out_k, dk[b, n0+sid·48 : …, h, :])` | 半区；**4-D 视图使 OOB pass 按本 request 的 `S_kv` 裁剪**（§5.4 第 1 条） | AIC + Vector |
| 11 | **多缓冲（v4 改）** | **`for i in T.serial(n_half)` + 2-tile 静态展开**；版本由**静态缓冲名** `_s0/_s1` 给出 | `BLOCK_M` 个 query 块的半体循环；`n_half = (n_it+1)//2`（运行期标量界） | — |
| 12 | **延迟梯度 GEMM（A5）** | `GB(k−1)` 放在 `EP(k)` 之后、读**另一组**版本；`if kqA > loop_st` 守首迭代 | 5 个 GEMM 的操作数与 v3 逐字相同，只是取址晚一迭代 | Cube |
| 13 | **drain** | 循环后 `if (n_it % 2) == 1: GB(…_s0 …) else: GB(…_s1 …)` | 补 `GB(loop_ed−1)`；版本按奇偶 | Cube |
| 14 | **不再使用** | ~~`T.Pipelined(num_stages=…)` + `T.annotate_buffer_versions`~~ | **删除**：它按*当前*迭代下标重写版本索引，无法表达滞后消费者；`num_stages` 在本 kernel 上被 Z3 `unsat` 静默退化（§6.3 A） | — |

### 3.3 计算伪代码

> 以下结构**已在设备上跑通并与 fp64 golden 数值一致**：`history_version/design_probe_bwd_proto.py`
> （v2，3-D 视图，`S=384`/`S=2048`，causal/非 causal）与 `history_version/design_probe_b1_tailrace.py`
> （v3，**4-D 视图**，B=2/S=512/H=4 的 causal 与非 causal 各 5 次 + `num_stages=2` 回归），见 §1.6.4 D/H。

```python
BLOCK_N, BLOCK_M = 96, 64          # keys / queries
HN = BLOCK_N // 2                  # 48：每个 AIV 的 key 行数
D_QK, D_V = 192, 128
NT_PER_B = ceildiv(S_KV, BLOCK_N)              # 每 request 的 key 块数（静态）
SHIFT = S_KV - S_Q
IS_CAUSAL = is_causal                          # 编译期常量：生成两套 kernel（§6.1）
NV = 2                                         # 显式版本数（§6.3；3 是 Stage 4 旋钮，未实测）

# ===== 三段工作 + 一个滞后段（v4 把 v3 的一个循环体拆成四个可命名的宏）=====
@T.macro
def F(q, do, lse, delta, q_s, do_s, qT_s, doT_s, lse_u, dlt_u, b, h, m0):
    T.copy(q[b, m0:m0 + BLOCK_M, h, :], q_s)          # 写版本 v 的四个装载
    T.copy(do[b, m0:m0 + BLOCK_M, h, :], do_s)
    T.copy(q[b, m0:m0 + BLOCK_M, h, :], qT_s, transpose=True)
    T.copy(do[b, m0:m0 + BLOCK_M, h, :], doT_s, transpose=True)
    T.copy(lse[b, h, m0:m0 + BLOCK_M], lse_u)          # (64,) fp32，版本 v
    T.copy(delta[b, h, m0:m0 + BLOCK_M], dlt_u)

@T.macro
def GA(k_s, v_s, q_s, do_s, a_k, b_lo, st_c, dpt_c, st_u, dp_u):
    T.copy(k_s, a_k);                     T.copy(q_s, b_lo)
    T.gemm(a_k, b_lo, st_c, transpose_B=True, clear_accum=True)                 # s^T
    T.copy(v_s, a_k[0:BLOCK_N, 0:D_V]);   T.copy(do_s, b_lo[0:BLOCK_M, 0:D_V])
    T.gemm(a_k[0:BLOCK_N, 0:D_V], b_lo[0:BLOCK_M, 0:D_V], dpt_c,
           transpose_B=True, clear_accum=True)                                  # dP^T
    T.dual_copy(st_c, st_u);              T.dual_copy(dpt_c, dp_u)

@T.macro
def EP(sid, st_u, dp_u, p16_u, dsT_u, lse_u, dlt_u, pT_dst, dsT_dst, n0, m0):
    fast = (n0 + BLOCK_N <= S_KV) & (m0 + BLOCK_M <= S_Q)
    if IS_CAUSAL:
        fast = fast & (n0 + BLOCK_N - 1 <= m0 + SHIFT)
    if fast:
        with T.SimtVF(threads=128):                     # A2 快路：整块可见
            for i, j in T.Parallel(HN, BLOCK_M):
                p = T.exp(st_u[i, j] * SCALE - lse_u[j])
                p16_u[i, j] = T.cast(p, dt)
                dsT_u[i, j] = T.cast(p * (dp_u[i, j] - dlt_u[j]), dt)
    else:
        with T.SimtVF(threads=128):                     # 掩码臂（对角块 / 尾块）
            for i, j in T.Parallel(HN, BLOCK_M):
                gi = n0 + T.cast(sid, "int32") * HN + i
                gj = m0 + j
                vis = (gi < S_KV) & (gj < S_Q)
                if IS_CAUSAL:
                    vis = vis & (gi <= gj + SHIFT)
                p = T.exp(st_u[i, j] * SCALE - lse_u[j])
                pv = T.if_then_else(vis, p, T.cast(0, "float32"))
                p16_u[i, j] = T.cast(pv, dt)
                dsT_u[i, j] = T.cast(pv * (dp_u[i, j] - dlt_u[j]), dt)
    T.dual_copy(p16_u, pT_dst);  T.dual_copy(dsT_u, dsT_dst)   # dst = 版本 v(k) 的 L1

@T.macro
def GB(pT_p, dsT_p, doT_p, qT_p, kT_s, a_k, a_t, b_hi,
       dq_c, dk_c, dv_c, dq_acc, b, m0, h, clr):
    T.copy(pT_p,  a_k[0:BLOCK_N, 0:BLOCK_M]); T.copy(doT_p, b_hi[0:D_V, 0:BLOCK_M])
    T.gemm(a_k[0:BLOCK_N, 0:BLOCK_M], b_hi[0:D_V, 0:BLOCK_M], dv_c,
           transpose_B=True, clear_accum=clr)                                  # dV += pT·doT^T
    T.copy(dsT_p, a_k[0:BLOCK_N, 0:BLOCK_M]); T.copy(qT_p,  b_hi[0:D_QK, 0:BLOCK_M])
    T.gemm(a_k[0:BLOCK_N, 0:BLOCK_M], b_hi[0:D_QK, 0:BLOCK_M], dk_c,
           transpose_B=True, clear_accum=clr)                                  # dK += dsT·qT^T
    T.copy(dsT_p, a_t);                        T.copy(kT_s, b_hi)  # ★ kT 最后重灌
    T.gemm(a_t, b_hi, dq_c, transpose_A=True, transpose_B=True, clear_accum=True)  # dQ
    T.set_atomic("add", "float32")
    T.copy(dq_c, dq_acc[b, m0:m0 + BLOCK_M, h, :])                        # 原子写回（4-D 视图）
    T.set_atomic_none()

@T.prim_func
def _bwd_main(
    q:  T.Tensor((B, S_Q,  H, D_QK), dt),      # 4-D batch 视图（零拷贝 .view）
    k:  T.Tensor((B, S_KV, H, D_QK), dt),
    v:  T.Tensor((B, S_KV, H, D_V),  dt),
    do: T.Tensor((B, S_Q,  H, D_V),  dt),
    lse: T.Tensor((B, H, S_Q), "float32"),
    delta: T.Tensor((B, H, S_Q), "float32"),
    dq_acc: T.Tensor((B, S_Q, H, D_QK), "float32"),
    dk: T.Tensor((B, S_KV, H, D_QK), dt),
    dv: T.Tensor((B, S_KV, H, D_V),  dt),
):
    with T.Kernel(H * B * NT_PER_B) as bid:
        h    = bid // (B * NT_PER_B)
        tile = bid % (B * NT_PER_B)
        b    = tile // NT_PER_B
        n0   = (tile % NT_PER_B) * BLOCK_N
        sid  = T.call_extern("int64", "asc_get_sub_block_id")   # 0/1，本 AIV 的 key 半区
        # ---- L1：key 侧常驻（每 launch，单版本）----
        k_s  = T.alloc_l1((BLOCK_N, D_QK), dt)
        v_s  = T.alloc_l1((BLOCK_N, D_V), dt)
        kT_s = T.alloc_l1((D_QK, BLOCK_N), dt)
        # ---- L1/UB：每迭代缓冲的**两个版本**（静态名字，无运行期版本索引）----
        q_s0  = T.alloc_l1((BLOCK_M, D_QK), dt);  qT_s0 = T.alloc_l1((D_QK, BLOCK_M), dt)
        do_s0 = T.alloc_l1((BLOCK_M, D_V), dt);   doT_s0 = T.alloc_l1((D_V, BLOCK_M), dt)
        pT_s0 = T.alloc_l1((BLOCK_N, BLOCK_M), dt); dsT_s0 = T.alloc_l1((BLOCK_N, BLOCK_M), dt)
        lse_u0 = T.alloc_shared((BLOCK_M,), "float32"); dlt_u0 = T.alloc_shared((BLOCK_M,), "float32")
        q_s1  = T.alloc_l1((BLOCK_M, D_QK), dt);  qT_s1 = T.alloc_l1((D_QK, BLOCK_M), dt)
        do_s1 = T.alloc_l1((BLOCK_M, D_V), dt);   doT_s1 = T.alloc_l1((D_V, BLOCK_M), dt)
        pT_s1 = T.alloc_l1((BLOCK_N, BLOCK_M), dt); dsT_s1 = T.alloc_l1((BLOCK_N, BLOCK_M), dt)
        lse_u1 = T.alloc_shared((BLOCK_M,), "float32"); dlt_u1 = T.alloc_shared((BLOCK_M,), "float32")
        # ---- L0：显式操作数（单版本，L0B 余量 4 KiB）----
        a_k = T.alloc_l0a((BLOCK_N, D_QK), dt)     # K-major：k/v/pT/dsT
        a_t = T.alloc_l0a((BLOCK_N, BLOCK_M), dt)  # MN-major：dsT（dQ 的 trans_A）
        b_lo = T.alloc_l0b((BLOCK_M, D_QK), dt)    # q / do
        b_hi = T.alloc_l0b((D_QK, BLOCK_N), dt)    # qT / doT / kT（kT 最后重灌）
        # ---- L0C 累加器（v4 与 v3 逐字节相同，**不需要第二版**）----
        st_c  = T.alloc_l0c((BLOCK_N, BLOCK_M), "float32")
        dpt_c = T.alloc_l0c((BLOCK_N, BLOCK_M), "float32")
        dq_c  = T.alloc_l0c((BLOCK_M, D_QK), "float32")
        dk_c  = T.alloc_l0c((BLOCK_N, D_QK), "float32")   # 跨迭代
        dv_c  = T.alloc_l0c((BLOCK_N, D_V), "float32")    # 跨迭代
        # ---- UB：每 AIV 半区（单版本，同迭代内生产-消费）+ 块尾 ----
        st_u  = T.alloc_shared((HN, BLOCK_M), "float32")
        dp_u  = T.alloc_shared((HN, BLOCK_M), "float32")
        p16_u = T.alloc_shared((HN, BLOCK_M), dt)
        dsT_u = T.alloc_shared((HN, BLOCK_M), dt)
        tail_u = T.alloc_shared((HN, D_QK), "float32"); tailv_u = T.alloc_shared((HN, D_V), "float32")
        out_k  = T.alloc_shared((HN, D_QK), dt);        out_v  = T.alloc_shared((HN, D_V), dt)

        # 全 extent 装载；越界尾部由 AscendInsertOOBPadding 改写为 min() clamp + fill 0
        # ★ 4-D 视图：clamp 的行上界 = 本 request 的 S_KV（设备码 min(96, S_KV - n0)）
        T.copy(k[b, n0:n0 + BLOCK_N, h, :], k_s)
        T.copy(v[b, n0:n0 + BLOCK_N, h, :], v_s)
        T.copy(k[b, n0:n0 + BLOCK_N, h, :], kT_s, transpose=True)

        loop_st = T.max(0, (n0 - SHIFT) // BLOCK_M) if IS_CAUSAL else 0
        loop_ed = ceildiv(S_Q, BLOCK_M)
        n_it    = loop_ed - loop_st                 # 块级标量（§1.6.2 #13）
        n_half  = (n_it + 1) // 2

        for i in T.serial(n_half):                  # ← v4：T.serial + 2-tile 展开（§6.3 B）
            # ---------------- 半体 A：版本 A ----------------
            kqA = loop_st + i * 2
            F(q, do, lse, delta, q_s0, do_s0, qT_s0, doT_s0, lse_u0, dlt_u0, b, h, kqA * BLOCK_M)
            GA(k_s, v_s, q_s0, do_s0, a_k, b_lo, st_c, dpt_c, st_u, dp_u)
            EP(sid, st_u, dp_u, p16_u, dsT_u, lse_u0, dlt_u0, pT_s0, dsT_s0, n0, kqA * BLOCK_M)
            if kqA > loop_st:                       # 首迭代没有可滞后的 GB
                GB(pT_s1, dsT_s1, doT_s1, qT_s1, kT_s, a_k, a_t, b_hi, dq_c, dk_c, dv_c,
                   dq_acc, b, (kqA - 1) * BLOCK_M, h, kqA - 1 == loop_st)   # 读版本 B
            # ---------------- 半体 B：版本 B ----------------
            kqB = kqA + 1
            if kqB < loop_ed:                       # 迭代数为奇数时最后一次不展开半体 B
                F(q, do, lse, delta, q_s1, do_s1, qT_s1, doT_s1, lse_u1, dlt_u1, b, h, kqB * BLOCK_M)
                GA(k_s, v_s, q_s1, do_s1, a_k, b_lo, st_c, dpt_c, st_u, dp_u)
                EP(sid, st_u, dp_u, p16_u, dsT_u, lse_u1, dlt_u1, pT_s1, dsT_s1, n0, kqB * BLOCK_M)
                GB(pT_s0, dsT_s0, doT_s0, qT_s0, kT_s, a_k, a_t, b_hi, dq_c, dk_c, dv_c,
                   dq_acc, b, (kqB - 1) * BLOCK_M, h, kqB - 1 == loop_st)   # 读版本 A

        # ---- drain：GB(last) 没有"下一个迭代"，必须无条件补（§9.2 反例）----
        last = loop_ed - 1
        if (n_it % 2) == 1:  GB(pT_s0, dsT_s0, doT_s0, qT_s0, kT_s, a_k, a_t, b_hi, dq_c, dk_c,
                                dv_c, dq_acc, b, last * BLOCK_M, h, last == loop_st)   # 版本 A
        else:                GB(pT_s1, dsT_s1, doT_s1, qT_s1, kT_s, a_k, a_t, b_hi, dq_c, dk_c,
                                dv_c, dq_acc, b, last * BLOCK_M, h, last == loop_st)   # 版本 B

        # ---- 块尾：dk = cast(dk_c*SCALE)、dv = cast(dv_c)（与 v3 逐字相同）----
        T.dual_copy(dk_c, tail_u); T.dual_copy(dv_c, tailv_u)
        with T.SimtVF(threads=128):
            for i, d in T.Parallel(HN, D_QK):
                out_k[i, d] = T.cast(tail_u[i, d] * SCALE, dt)
            for i, d in T.Parallel(HN, D_V):
                out_v[i, d] = T.cast(tailv_u[i, d], dt)
        # ★ 写回区段按 **本 request** 的行号，OOB clamp 因此裁到 S_KV
        kown = n0 + T.cast(sid, "int32") * HN
        T.copy(out_k, dk[b, kown:kown + HN, h, :])
        T.copy(out_v, dv[b, kown:kown + HN, h, :])
```

注（与原型的差异与约束，逐条可复核）：

1. **GEMM 的 M/N/K 全静态**：L0 路径从 **allocation 形状**取 `node_M/node_N/node_K`（`gemm_op.py:_l0_gemm_call`），
   而 region extent 可以是静态切片（`a_k[0:96,0:128]` 等），因此 L0 缓冲可以按「最大形状分配 + 切片使用」复用。
   **region 的原点必须为 0**（`gemm_mad.py:_l0_operation_extents` 的 `require_zero_origin`），故
   `kT_s` 若需拆半（备选方案的 112/64）必须用两块独立 L1 缓冲。
2. **尾块由「全 extent 装载 + 掩码 + 平台 OOB clamp」三件套处理**：被补 0 的行/列再由掩码臂的
   `gi < S_KV` / `gj < S_Q` 置零（补 0 只让 `s=0`，而 `exp(0·SCALE − lse) ≠ 0`，故掩码必须保留这两项）。
3. **★ 块尾写回的 request 隔离（v3 修 B1）**：GM 张量必须是 **4-D batch 视图**。3-D packed 视图下
   `AscendInsertOOBPadding` 的 clamp 维度是整张 `B*S_KV`（本仓实测设备码
   `min(48, max(1024 − sid·48 − (bid%6)·96 − ((bid%12)/6)·512, 0))`，B=2/S=512），于是 request 0 的尾块会
   把 0 写进 request 1 的 key 0…63；4-D 视图的同一处设备码是 `min(48, max(512 − sid·48 − (bid%6)·96, 0))`
   ⇒ 只剩 `S_KV − (n0+sid·48)` 个有效行，与源算子 `bwd.py:255-260` 的逐行守卫同义。实测见 §1.6.4 H。
4. **`IS_CAUSAL` 的两处编译期守卫**（`loop_st` 与掩码臂的因果项）：`is_causal=False` 时 `loop_st` 必须为 0、
   掩码不得含 `gi <= gj + SHIFT`，否则边界块被错误剪枝/错误掩蔽（v3 补，v2 伪代码漏）。
   源码同构写法见 `bwd.py:202, 230-243`。
5. **无 `T.Vector`/`sid` 帧**：只取一次执行体 id 做 key 行偏移（v1 运行期实测该取法确定且正确）。
6. `dq_acc` 行偏移为 `[b, m0, h, :]`（dense 下等价于 `q_start + m0`）。
7. `b_hi` 被 `qT/doT/kT` 复用：**`kT` 必须在 `doT/qT` 之后重灌**（原型首版曾因顺序错误得到 dQ 全错，
   见 §9.2 的「常见错误」表）。
8. **版本由静态缓冲名给出，不再用 `T.Pipelined`/`annotate_buffer_versions`（v4，A5）**：上表的两份
   `_s0/_s1` 就是 §4.6 的「每迭代 2 版本」；`T.serial(n_half)` + 2-tile 展开使「谁读哪一份」在**编译期**
   确定，而 `num_stages`/`annotate_buffer_versions` 只会把索引写成 `(kq − loop_st) % V`（生产与消费者
   两侧同一表达式），无法表达滞后一迭代的读者（§6.3 A）。`factory` 的 `n_versions` 键只保留为 Stage 4 的
   A/B 接口，不再传给 `T.Pipelined`。
9. **`GB(k−1)` 只读上一迭代版本的四个操作数**：`pT_(1−v)`、`dsT_(1−v)`（`EP` 产物）与 `qT_(1−v)`、
   `doT_(1−v)`（`F` 产物）。`kT_s` 是常驻缓冲；`doT/qT/kT` 在 `b_hi` 内的复用顺序不变，故 **k 的 GEMM 与
   k−1 的 GEMM 不会互相踩缓冲**（探针 `dk/dv` 逐位一致，§1.6.4 I-3）。
10. **drain 是必需的、且要按奇偶选版本**：`GB(loop_ed−1)` 没有「下一个迭代」。`n_it` 为奇数时最后一个半体是
   A（`_s0`），偶数时是 B（`_s1`）。漏掉偶数分支会让 `n_it=2` 的块少一次 `dk/dv` 累加（varlen 首版探针
   在 `q=kv=[192,192]` 上误差 5.7e-2）；本算子 L0-12 专测该分支。
11. **`st_u/dp_u/p16_u/dsT_u` 仍由**两个半体**共用（单版本）**：它们的生产（fixpipe）与消费（`EP`）都在
   同一半体内；跨半体的唯一 hazard 是 `fixpipe(k+1)` 对 `EP(k)` 的 WAR，由 InsertSync 自动处理
   （§9.1 风险 16 是它的量化）。

### 3.4 API 可行性确认

| API | 来源 / 核验状态 |
|-----|----------------|
| `T.Kernel(grid)`（一维，无 `threads`） | `ascend/language/kernel.py:Kernel`（源码确认）；PL-950-gemm-tiles 用一维网格 |
| `T.alloc_l1` / `T.alloc_l0a` / `T.alloc_l0b` / `T.alloc_l0c` / `T.alloc_shared` | `ascend/language/allocate.py`；PL-950-gemm-tiles（已核验的双版本 + 显式 L0 形态）；**本轮原型设备码含 5 个 `asc_mmad` + 每迭代 10 条 `asc_copy_l12l0a/l12l0b`**（v3 更正：v2 写「4 个」，实为每迭代 10 条 L1→L0 拷贝） |
| `T.gemm(a_l0a, b_l0b, c_l0c, transpose_A=…, transpose_B=…, clear_accum=…)` | `gemm_op.py:_dialect_gemm_call` → `GemmMAD._lower_l0`；四种 trans 组合的布局推断见 `gemm_mad.py:126-134`；**本轮探针 2 实测 (T,T)/(F,F)/(F,T) 三种组合编译 + 数值 PASS** |
| `T.copy` GM→L1（含 `transpose=True` dn2nz） | `copy_op.py` + PL-950-gemm-tiles 的 `trans_b` 分支；v1 设备码 `asc_copy_gm2l1_nd2nz` / `dn2nz` |
| `T.copy` L1→L0（含 `trans_A` 的转置搬运） | 原型设备码 `asc_copy_l12l0a` / `asc_copy_l12l0b` / **`asc_copy_l12l0a_transpose`**（`trans_A=True` 的 A），数值 PASS |
| `T.dual_copy` L0C→UB（2:1 M 分片） | `copy_op.py:dual_copy` 文档 + v1 运行期 L0 全过（13/13）；原型 `asc_copy_l0c2ub(..., dual_dst_mode=1)` |
| `T.dual_copy` UB→L1（2:1 M 分片） | 同上（`InsertNd2Nz` 改写成 NZ scatter）；原型设备码 **2 个** `ascend_nd2nz_scatter`（v1 为 3 个——v3 更正：v2 同一句里既写「3 个」又写「本版 2 个」，自相矛盾） |
| `T.set_atomic("add","float32")` + L0C→GM `T.copy` | `mode.py:set_atomic` + PL-950-atomic-splitk；v1 运行期实测加法语义成立（L0-11） |
| `T.SimtVF(threads=128)` + `T.Parallel(48,64)` + `T.exp` + `T.if_then_else` + `T.cast` | PL-950-row-primitives / PL-950-attention（已核验）；本轮探针 `epi0`/`epi1`/`epi2` 三种形态均编译并实测 |
| `sid = T.call_extern("int64","asc_get_sub_block_id")` | v1 运行期实测：两 AIV 各得 0/1、确定（`repro/CONST-950-aiv-sid-capture.py`） |
| `T.annotate_buffer_versions` | **v4 不再使用**：v3 曾与 `num_stages=2` 配对声明 2 版本，实测 3.418 vs 3.446–3.486 µs（±2% 内，不承诺收益），且它按*当前*迭代下标重写索引、无法表达滞后消费者（§6.3 A）。**登记为「存在但本结构不适用」** |
| `T.Pipelined(start, stop, num_stages=1)` 动态界 | v1 设备码 `for (kq = <运行时下界>; kq < <上界>; ++kq)` |
| **`T.serial` 支持运行期上界与块内多段语句**（v4） | 本轮探针实测：`for i in T.serial(n_half)`（`n_half` 为 `(loop_ed-loop_st+1)//2` 的运行期表达式）编译并数值 PASS；生产 varlen v5 同形态已上线（§1.6.4 I-1/I-2） |
| **静态命名的两份 L1 缓冲 + 交叉读取**（v4） | 本轮探针：`q_s0/q_s1` 等 6 组 + `pT/dsT`，`GB` 交叉读另一组；设备码中 `GB` 前**无**以 `PIPE_V` 为前驱的 `asc_sync_wait`（§1.6.4 I-1 的 `l12l0a=16` 与 `sync 76` 计数） |
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
| L0A/L0B 各 64 KiB，无版本余量 | 是 | L0A 48 KiB（75%）、L0B 60 KiB（93.75%）；**任何有用 tile 下 L0 只能单版本**（`2×(24,576+384·Nk) ≤ 65,536` 仅 `Nk ≤ 21` 成立，§1.6.3）；后果量化见 §1.6.4 G |
| **`AscendInsertOOBPadding` 的 clamp 维度是 buffer 形状**（本轮实编译设备码） | 是 | GM 张量一律用 **4-D batch 视图**，使 clamp 变成「按 request」；3-D packed 视图会让尾块写到下一个 request（§5.4 第 1 条、§12-B1） |
| L0C→UB 不支持 dtype 转换（同 dtype 断言） | 是 | L0C→UB 保持 fp32 半区，cast 在 UB 内用 `T.cast` 完成 |
| 整 tile `T.copy(L0C→UB)` 只落一个 AIV（v1 运行期实测） | 是 | 所有 L0C↔UB 交叉用半区 `dual_copy`（§0.6 R2） |
| 线程域只在 `T.SimtVF` 内；显式 `T.Vector` 不可编译 | 是 | 所有逐元素工作包在 `T.SimtVF(threads=128)`；只取一次 subblock id |
| 原子存储支持 dtype 集合含 float32（`mode.py`） | 是 | `dq_acc` 为 fp32 ✓ |
| 未认证项：手工跨核 flag、循环内复杂同步、跨 block barrier | 是 | 不写手工 flag，全部依赖 AutoSchedule + InsertSync（§7） |
| grid 上限 65535（`TRAP-950-grid-over-65535`） | 是 | 主选 grid 最大 22016（`ds-v3-b2-8k`）< 65535 ✓；factory 侧加断言（§9.1 风险 12） |
| 动态 shape | 部分 | `S_q/S_kv/B/H/D` 编译期常量；`cu_seqlens_*` 只保留签名兼容 |
| **L1 容量 512 KiB**（v4：6 个装载缓冲 + `pT/dsT` 各两份） | 是 | **311,296 B（59.4%）**；余 212,992 B。`V=3` 旋钮会到 393,216 B（75.0%）——未实测，列 Stage 4 A/B（§1.6.3） |
| **生成代码体积（icache）**（v4 实测） | 是 | 20,341 → **50,879 B（×2.50）**，绝对量仍小于已上线的 varlen v5（61,913 B）；处置见 §9.1 风险 14 |

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
| 流水 | `T.Pipelined(num_stages=2)`（cp.async，自动双缓冲 + 语句重排） | **`T.serial` + 2-tile 静态展开 + 显式两份缓冲 + `GB` 滞后一迭代**（不依赖调度器，§6.3） | §6.3、§1.6.1 A5 |
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
| **`examples/mla_varlen_bwd/_bwd_kernel/DESIGN.md`（§6.3 / §1.6.4 F）** | **高（v4 的调度模板，同族已上线）** | 2-tile 静态展开 + 显式两版本 + drain 奇偶 + `clear_accum` 后移的完整写法与设备码判据 |
| **`examples/mla_varlen_bwd/_bwd_kernel/history_version/design_probe_epilogue_rotation.py`** | **高（探测方法）** | v4 探针的直接模板（base/rot 双体同进程 + `--parity` 逐位 + `--time` 单价） |
| **`examples/mla_varlen_bwd/_bwd_kernel/RETROSPECTIVE.md`（v5 行）** | 中（**负面**经验） | 旋转后 MTE2 变第一忙管道的现场证据（80.5→55.5 GB/s）⇒ 本版风险 19 的来源 |
| **`examples/mla_dense_bwd_perf/perf_opt/perf_feedback.md`** | **高（v4 的问题定义）** | 迭代内依赖链的逐段实测（2.94 µs）、13 个 `num_stages` 分支的裁决、Z3 `unsat` 日志 |
| `history_version/design_probe_bwd_proto.py` | —（v3 原型） | 新结构的可跑通形态与实测单价（§1.6.4 D） |
| **`history_version/design_probe_dense_rotation.py`** | **—（v4 探针）** | **base vs rot 的编译/数值/逐位/单价（§1.6.4 I）** |
| `history_version/design_probe_bwd_cost.py` / `design_probe_l0_gemm.py` | —（本版探针） | 分段单价、L0 路径四类 trans 的数值证据 |

#### 3.5.4 探针证据（可复现命令）

**执行上下文偏离登记（v3 新增，回应建议 5）**：平台 `statectl platform --task-dir …` 返回的 `execution` 是
`mode=device`、`prototype_command=[/usr/local/python3.12.16/bin/python3, <script>]`、`working_directory=<task_dir>`。
本任务的探针**不能**按该默认命令执行，偏离项与理由逐条登记如下（Stage 3/4 必须沿用同一调用方式）：

| 项 | 平台默认 | 实际采用 | 原因（实测） |
|---|---|---|---|
| 解释器 | `/usr/local/python3.12.16/bin/python3` | `/workspace/venv-ascend950/bin/python` | 平台默认解释器**没有 torch / tilelang**，`import tilelang` 直接 ImportError |
| 工作目录 | `<task_dir>`（在仓根内） | **仓根之外**（`cd /tmp`） | cwd 在仓根内时 dev-root 的 `tilelang/` 遮蔽 site-packages 的 0.1.15，报 `TileLang's private TVM Python package is missing` |
| 环境脚本 | 无 | 先 `source .agents/platforms/toolchains/ascend/env-950pr.sh` | 需要 CANN 9.2.1 的 `LD_LIBRARY_PATH` / `ASCEND_HOME` |
| TMPDIR | 系统默认 | `TMPDIR=/workspace` | `/tmp` 容量不足，编译中间件写入失败 |
| 设备 | 单卡 | 单卡（`ASCEND_RT_VISIBLE_DEVICES=""` 仅用于纯 CPU 的 `verify_equiv.py`） | — |

四个探针/原型脚本都在**仓根之外**运行：

```bash
cd /tmp && source /workspace/tilelang-mlir-ascend-950/.agents/platforms/toolchains/ascend/env-950pr.sh
PROBE_DIR=/workspace/tilelang-mlir-ascend-950/examples/mla_dense_bwd/_bwd_kernel/history_version
for s in design_probe_bwd_cost design_probe_l0_gemm design_probe_bwd_proto design_probe_b1_tailrace; do
  TMPDIR=/workspace /workspace/venv-ascend950/bin/python "$PROBE_DIR/$s.py"
done
# v4 的旋转探针（四种入口，逐项）
for m in --compile "" --parity --time; do
  TMPDIR=/workspace /workspace/venv-ascend950/bin/python \
    "$PROBE_DIR/design_probe_dense_rotation.py" $m
done
# 纯 CPU 的等价性脚本（无需设备）
cd /tmp && ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python \
  /workspace/tilelang-mlir-ascend-950/examples/mla_dense_bwd/_bwd_kernel/verify_equiv.py
```

实测输出（2026-11-25，toolchain `ascend-950pr-9579 / TileLang ascend 0.1.15 / CANN 9.2.1`）：

| 探针 | 内容 | 实测结果 |
|------|------|---------|
| `design_probe_bwd_cost.py` | v1 内层迭代的 9 个分段单价（`mte2/gemm5/l0c2ub/ub2l1/epi0/epi1/epi2/chain_notr/chain`） | 见 §1.6.4 A 表；`chain`=5.944 µs 复现生产 6.03 µs |
| `design_probe_l0_gemm.py` | L0 路径的 `(T,T)`/`(F,F)`/`(F,T)` 三种 trans 组合 + 数值 | 全部 **PASS**（rel 1.1e-7 / 1.1e-7 / 1.4e-7）；单次 launch 4.1–4.2 µs（含 2.7 µs launch 地板） |
| `design_probe_bwd_proto.py` | C2 结构端到端（`S=384` 与 `S=2048`，causal/非 causal，3-D 视图） | 数值全 **PASS**；单价 3.469（stages=1，v3 复跑带 3.446–3.486）/ 3.391（stages=2，带 3.391–3.421） |
| **`design_probe_dense_rotation.py`（v4 新增）** | base（v3 体）vs rot（v4 体）：编译 + 数值 5/5 + **逐位 5/5** + 单价 | **COMPILE_OK ×2（源 20,341→50,879 B，`asc_mmad` 5→16，`__simt_vf__` 3→5）；NUMERIC 5/5 PASS（最差 2.72e-03）；PARITY 5/5（`dk/dv` **逐位相同**，`dq` ≤1.8e-8 相对）；TIME 2.965→**1.870 µs**（S=8192）/ 3.069→1.968（S=4096）**。详见 §1.6.4 I |
| **`design_probe_b1_tailrace.py`（v3 新增）** | **3-D vs 4-D 视图**的 `dk/dv` 块尾写回；B=2/S=512/H=4，causal 与非 causal 各 5 次 + 4-D/stages=2 回归 + 单价对照 | **3-D 非 causal：5/5 FAIL**（`dk` 0.33–0.39、`dv` 0.33–0.40，误差全部落在 request 1 的 key 0…63）；**4-D：causal/非 causal 均 5/5 PASS**（≤2.1e-3）；4-D+`stages=2` 数值 PASS；单价 3-D 3.446–3.486 vs 4-D 3.461–3.509（同值）vs 4-D/stages=2 3.418–3.421。详见 §1.6.4 H |
| 原型 bisect（L0/L1 混用） | 4 个 L1 路径 GEMM + 显式 L0 缓冲 | **FAIL**：`dk/dv` 错 4.8e-1 / 3.1e-2 → 已列为硬约束（§3.5.1） |
| 原型 bisect（UB→L1 转置） | `T.copy(UB, L1, transpose=True)` | **COMPILE_FAIL**（`Ascend device compilation failed`） |
| L0C 容量 | `Nk=128, Mq=48`（249856 B = 95.3%）端到端 | **设备报 507015**（`npuSynchronizeDevice`）→ 视为 L0C 上界证据 |

说明：以上除「数值 PASS」外均为**单价/可行性**证据；生产 shape 的精度与性能仍由 Stage 3 的 L0/L1/L2 套件
与 msprof 采集（§8.3）。`design_probe_b1_tailrace.py` 的完整输出保存在
`history_version/b1_probe_run.log`，设备码 clamp 证据保存在 `history_version/b1_device_code_bounds.txt`。

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `q` | `(B, S_q, H, 192)` | `float16 \| bfloat16` | **4-D batch 视图**（host `.view`，零拷贝）；源算子传入的是 `(T_q,H,192)` packed 视图，二者内存布局相同 |
| `k` | `(B, S_kv, H, 192)` | 同上 | `S_kv` 每 request 相同（dense） |
| `v` | `(B, S_kv, H, 128)` | 同上 | |
| `do` | `(B, S_q, H, 128)` | 同上 | |
| `lse` | `(B, H, S_q)` | `float32` | **自然对数** log-sum-exp；空行为 `-inf` |
| `delta` | `(B, H, S_q)` | `float32` | `rowsum(o*do)`，由 `_delta_kernel` 产出 |
| `cu_seqlens_q` | `(B+1,)` | `int32` | dense 下 `arange(0,(B+1)*S_q,S_q)`；本 kernel 只保留签名兼容 |
| `cu_seqlens_kv` | `(B+1,)` | `int32` | 同上（`S_kv`） |

（源算子签名另含 `o`：由同一次算子调用的 `_delta_kernel` launch 消费，**本 kernel 不读**，见 §0.2。
**v3 变更**：4 个 token-major 张量由 3-D 改 4-D batch 视图，理由是让平台 OOB clamp 按 request 裁剪，
见 §5.4 第 1 条与 §12-B1。）

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `dk` | `(B, S_kv, H, 192)` | `float16 \| bfloat16` | 与 `k` 同 shape；已乘 `scale` |
| `dv` | `(B, S_kv, H, 128)` | 同上 | 与 `v` 同 shape |
| `dq_acc` | `(B, S_q, H, 192)` | `float32` | **in-out**，原子累加；未乘 `scale`（由 `_dq_land_kernel` 处理） |

三个输出与输入同 layout（BSHD、尾轴连续），**无需任何转置或 permute**；`dk/dv` 的 shape 与 `k/v` 逐维相同，
`dq_acc` 与 `q` 同 shape（仅 dtype 为 fp32），故 host 侧只做 `view`/`zeros`。
`dk/dv` 的块尾写回**按本 request 的行号**寻址（`dk[b, n0+sid·48 : …]`），由平台 OOB clamp 裁到 `S_kv`。

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

**版本判定表（v4 新增，A5 的核心容量结论）**——「哪块缓冲要两份」由**生存期**决定，不由旋钮决定：

| Buffer | 生产 → 消费 | 跨迭代？ | 版本数 | 依据 |
|---|---|---|:---:|---|
| `q_s/qT_s/do_s/doT_s/lse_u/dlt_u`（`F` 产物） | `F(k)` → `GA(k)`（同迭代）**与** `GB(k)`（**下一迭代**） | **是（距离 1）** | **2** | `GB(k)` 的 `qT/doT` 必须是**上一迭代**的那一份（§1.6.1 A5(iii)） |
| `pT_s/dsT_s`（`EP` 产物） | `EP(k)` → `GB(k)`（**下一迭代**） | **是（距离 1）** | **2** | 同上；这也是 `GB(k−1)` 与 `EP(k)` 之间**没有** `PIPE_V` 边的原因 |
| `st_u/dp_u/p16_u/dsT_u` | fixpipe(k) → `EP(k)` | 否（同迭代） | **1** | 设备级 `dk/dv` 逐位一致已验证（§1.6.4 I-3）；回退旋钮见 §9.1 风险 16 |
| `st_c/dpt_c/dq_c` | mad(k) → fixpipe(k) → `EP(k)`/原子写 | 否 | **1** | **旋转不需要 L0C 双版本**——这一条推翻了 `perf_feedback.md` 里「必须把 dP 并入 st_c 才能双版本」的路径（它对应的是**两级旋转**，§6.3 E） |
| `dk_c/dv_c` | 跨迭代累加 | 是（累加器，不是版本） | **1** | `clear_accum=(kq−1==loop_st)` 初始化 |
| `k_s/v_s/kT_s` | 每 launch 一次 | 否 | **1** | 常驻 |

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
**v4 的搬运增补（版本 A/B，A5）**：`q/qT/do/doT/lse/delta` 的 `GM→L1/UB` 每次写入**本迭代的版本**，
`GB(k−1)` 的 `L1→L0` 从**上一迭代的版本**读出；`P^T/dS^T` 的 `UB→L1` 写本版本、`GB(k−1)` 读上一版本。
**L0C/UB 侧（`st_c/dpt_c/st_u/dp_u/p16_u/dsT_u`）的路径与 v3 逐字相同，没有新增任何搬运**；
GM 侧字节数与 v3 **逐字节相同**（189.8 GB，§1.6.0 R3）。

### 4.5 UB 内存预算（每 AIV，半区）

**UB** 容量 253,952 B（248 KiB）。**v4 口径（A5）**：只有 `lse/delta` 需要第二版，其余每迭代缓冲
（`st_u/dp_u/p16_u/dsT_u`）**保持单版本**——它们的生产与消费都在同一迭代内（§4.3 版本判定表）：

| Buffer | Shape | dtype | 版本 | 大小 (Bytes) |
|--------|-------|-------|:---:|---:|
| st_u | (48,64) | float32 | 1 | 12,288 |
| dp_u | (48,64) | float32 | 1 | 12,288 |
| p16_u | (48,64) | bf16 | 1 | 6,144 |
| dsT_u | (48,64) | bf16 | 1 | 6,144 |
| lse_u | (64,) | float32 | **2** | 512 |
| dlt_u | (64,) | float32 | **2** | 512 |
| **每迭代小计** | — | — | — | **37,888** |
| tail_u | (48,192) | float32 | 1 | 36,864 |
| tailv_u | (48,128) | float32 | 1 | 24,576 |
| out_k | (48,192) | bf16 | 1 | 18,432 |
| out_v | (48,128) | bf16 | 1 | 12,288 |
| **合计（v4 口径）** | — | — | — | **130,048（51.2%）**，余 123,904 B |

- **相对 v3 的 166,912 B（65.7%）反而少 36,864 B**：v3 给 `st_u/dp_u/p16_u/dsT_u` 也声明了第二版
  （`num_stages=2` 时代的保守预算），而 A5 **不需要**它们（生存期同迭代）。这是 A5 唯一的容量正收益。
- **保守上界（若实现期发现同步告警）**：把「每迭代小计」整体再乘 2 ⇒ `75,776 + 92,160 = 167,936 B`
  （**66.1%**），仍不超容量。**这是 §9.1 风险 16 的第一回退旋钮**，不是实现目标。
- **块尾缓冲不加版本**：`tail_u/tailv_u/out_k/out_v` 在 query 循环之外，`92,160 × 2` 会上界到
  `130,048 + 24,576 = 154,624 B`（60.9%，仍可，但与 A5 无关）。
- 编译侧旁证：v3 同 tile 原型的 `buf_dyn_shmem` 实占 123,264 B（编译器做了别名复用），本表是**保守上界**。

### 4.6 L1 / L0C / L0A / L0B 预算

| 层 | Buffer 组合 | 合计 (Bytes) | 容量 (Bytes) | 占比 |
|----|------------|-------------|-------------|------|
| L1（常驻，单版本） | `k_s 36864 + v_s 24576 + kT_s 36864` | 98304 | 524288 | 18.8% |
| L1（每迭代，**显式 2 版本**，A5） | `(q_s 24576 + qT_s 24576 + do_s 16384 + doT_s 16384 + pT_s 12288 + dsT_s 12288) × 2` | 212992 | 524288 | 40.6% |
| **L1 合计（v4 实现口径）** | — | **311296（304 KiB）** | 524288 | **59.4%**（余 212,992 B） |
| L1（`V=3` 旋钮：装载缓冲三份、`pT/dsT` 仍两份） | `98304 + 3×98304/…`（逐项：`3×(24576+24576+16384+16384) + 2×24576`） | **393216（384 KiB）** | 524288 | **75.0%**（未实测，Stage 4 A/B 第一项） |
| L0C | `dk_c 73728 + dv_c 49152 + st_c 24576 + dpt_c 24576 + dq_c 49152` | 221184（216 KiB） | 262144（256 KiB） | **84.4%** |
| L0A | `a_k (96,192) f16 36864 + a_t (96,64) f16 12288` | 49152（48 KiB） | 65536（64 KiB） | **75.0%** |
| L0B | `b_lo (64,192) f16 24576 + b_hi (192,96) f16 36864` | 61440（60 KiB） | 65536（64 KiB） | **93.75%** |

**L0C 上界证据**：`Nk=128, Mq=48`（`dk 98304 + dv 65536 + st 24576 + dpt 24576 + dq 36864 = 249856 B` = 95.3%）
在端到端原型上触发 `device error 507015`；`Nk=96, Mq=64`（221184 B = 84.4%）稳定运行。故**设计红线：L0C ≤ 240 KiB**，
主选 84.4%、备选 86.0% 均留 ≥14% 余量。

**L0B 余量说明（93.75%，仅 4096 B）**：`b_lo` 与 `b_hi` 都按「最大形状分配、切片使用」；
`b_hi` 承载 `qT (192,64)`、`doT (128,64)`、`kT (192,96)` 三种切片。若 Stage 3 发现需要第三块 L0B，
回退是把 `Nk` 降到 80（`b_hi (192,80)` = 30720 B，L0B 降到 55296 B = 84.4%），代价是 MTE2 总量 +20%
（**v3 修正**：该回退**不能**换来 L0 双版本——`2×55,296 = 110,592 B > 64 KiB`；L0 单版本是容量上的硬结论，
见 §1.6.3 与 §1.6.4 G）。

**L0 单版本的代价（v3 定性，v4 用实测标定）**：每迭代 `217,088 B` 的 L1→L0 与 5 个 mad 在 AIC 时间线上
**相加**（`T_AIC ≥ MTE1 + mad = 0.680 + 0.769 = 1.449 µs`）。v3 曾把「`MTE1 ≤0.20 µs`」列为 80% MFU 的
硬条件，**Stage 4 实测为 0.60–0.68 µs（217,088 B @ 300–340 GB/s，已是该路径的可达带宽），该条件不成立**
（`perf_feedback.md` 实测第 2 条）。v4 因此**不再以 MTE1 压降为前提**：A5 的收益来自「AIC 不必等 AIV」，
落地后墙钟 1.870 µs 已接近 `mte1+mad` 主链（1.449）+ 未覆盖的 fixpipe 尾部（§1.6.4 I-5/I-6）。

**A5 不改变 L0C/L0A/L0B（v4 的核心容量结论）**：三者的字节数与 v3 **逐字节相同**（221,184 / 49,152 /
61,440 B）。「让 `GB(k−1)` 与 `EP(k)` 重叠」**不需要** L0C 双版本——这正是 `perf_feedback.md` 提出的
「把 dP 并入 `st_c` + 双版本」路线**没有被采纳**的原因：那条路线是为**两级旋转**（`GA` 也提前）准备的，
而两级旋转在 v4 的墙钟下收益 ≈0（§6.3 E）。L0B 仍只有 4 KiB 余量 ⇒ L0 侧一律单版本
（`2×(24,576+384·Nk) ≤ 65,536` 只在 `Nk ≤ 21` 时成立），但 A5 **不需要** L0 双版本：`GA(k)` 与 `GB(k−1)`
在 AIC 上顺序执行，各自在自己的时段内独占 `a_k/b_lo/b_hi`。

### 4.7 动态轴

dense 路径下**无运行时动态轴**：`B/S_q/S_kv/H/D_qk/D_v` 全部是编译期常量（工厂按 shape 编译，
`@lru_cache` 级缓存，与源 `target_cache(maxsize=32)` 等价）。`T_q/T_kv` 的 packed 视图只是 host 侧
`view` 的元数据。变长是另一个算子（§0.7）。这一静态性同时是尾块 OOB clamp 与 L0 静态 extent 的前提。

### 4.8 JIT 配置

```python
@tilelang.jit(out_idx=[], target="ascend")
def _bwd_func(block_n: int = 96, block_m: int = 64, n_versions: int = 2) -> Callable:
    ...
# 编译标志沿用源：-O3、-DENABLE_BF16；不开 fast-math 之外的非默认选项
```

`out_idx=[]`：`dk/dv` 与 `dq_acc` 都是调用方预分配的张量（`dq_acc` 必须由 host 先 `torch.zeros`——
store-atomic 的累加契约，参照 PL-950-atomic-splitk 的调用方式）。
**v4 把 `num_stages` 键改为 `n_versions`**：含义从「流水站数」变为「**显式版本数**」（默认 2），因为 v4
不再使用 `T.Pipelined`/`annotate_buffer_versions`（§6.3 A）。3 是第一个调优旋钮（L1 393,216 B = 75.0%），
**未实测，不得写成收益**。factory 侧保留该键只为兼容 Stage 4 的 A/B 接口。

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
- 每迭代端口 325625 B / 31.4 FLOP/字节（v1 为 24.4）；备选的每字节 FLOP 与主选持平（≈31），
  其收益来自**迭代数 −15%**（每迭代固定开销被摊薄）与 MTE2 总量 −15%，不是来自端口效率。

### 5.3 约束分析

- **对齐**：`D_qk=192`、`D_v=128`、`block_n=96`、`block_m=64` 全为 16 的倍数；L1/L0 的 NZ 分形与 mad 的
  16×16 无尾块；`dq_acc` 尾轴 192 个 fp32 = 768 B（32 B 对齐 ✓）。
- **L0C**：单个累加器最大 `dk_c = 96×192×4 = 73728 B ≤ 262144` ✓；5 个合计 **216 KiB / 256 KiB（84.4%）**。
  红线 ≤240 KiB（§4.6 的实测上界）。
- **L0A/L0B**：48 KiB / 60 KiB（75% / 93.75%）；L0B 无版本余量 ⇒ L0 单版本。
- **L1**：304 KiB（59.4%，含每迭代 2 版本）；常驻部分 96 KiB。**v4 与 v3 同值**（A5 不改变 L1 总量）。
- **UB**：**130 KiB（51.2%）** / 248 KiB（v4：只有 `lse/delta` 两版本，`st_u/dp_u/p16_u/dsT_u` 单版本，
  §4.5）；块尾单版本。

### 5.4 注意事项（非整除与边界）

> 统一原则：**GEMM 一律全 extent（L0 路径的 M/N/K 取 allocation 形状）；动态 extent 只出现在 copy；
> copy 的越界尾部由平台 pass 改写为 `min()` clamp + 补 0（**裁剪维度 = buffer 形状的行数**，故 GM 张量
> 一律用 4-D batch 视图，使裁剪按 request）；补 0 的语义后果由掩码臂的 `gi < S_kv` / `gj < S_q` 收口。**

1. **尾 key 块**（`S_kv % 96 ≠ 0`）——**v3 重写（B1）**：
   - **装载侧**：`k_s/v_s/kT_s` 全 extent 装载，越界行由 `AscendInsertOOBPadding` 收窄 DMA 并补 0。
     设备码（4-D，B=2/S=512）：`min(96, 512 − n0)`；3-D 会是 `min(96, 1024 − n0 − b·512)`（把邻居的行读进来，
     但被掩码丢弃，属良性）。
   - **计算侧**：补 0 使 `s^T` 的这些行为 0，但 `exp(0·scale − lse) = exp(−lse) ≠ 0`，故**必须**由掩码臂的
     `gi < S_kv` 把 `P/dS` 置 0。
   - **写回侧（本次修复的核心）**：`dk/dv` 的块尾 store 必须**只写本 request 的行**。源算子的做法是
     逐行守卫 `if n0[0]+i < len_kv`（`bwd.py:255-260`）；本设计的做法是 4-D batch 视图
     `T.copy(out_k, dk[b, n0+sid·48 : …])`，于是 OOB clamp 算出的行数是
     `min(48, max(S_kv − sid·48 − n0, 0))`——与逐行守卫逐行同义（设备码实测）。
     **反例（v2 的 3-D 视图）**：同一处设备码是 `min(48, max(B·S_kv − sid·48 − n0, 0))`，B=2 时上界是 1024，
     故 request 0 的尾块把 48 行写满、其中 64 行落进 request 1 的 key 0…63。这些行的 `out_k/out_v` 是 0
     （掩码臂已置零），而写回是**普通 store（非原子）**，与 request 1 自己的真实值写入**竞态**
     ⇒ 非确定性的零覆盖。实测：非因果 5/5 失败（`dk` 0.33–0.39）、4-D 5/5 通过（≤2.1e-3），见 §1.6.4 H。
   - **为何 v1/v2 没暴露**：v1 的 `block_n=64` 对 5 条 manifest workload 全部整除（4096/8192/8192/3072/512），
     v2 的 `block_n=96` 才引入部分尾块，其中只有 `ds-v3-b2-8k`（B=2, `8192 = 85×96+32`）真正跨 request。
2. **尾 query 块**（`S_q % 64 ≠ 0`）：`q_s/do_s/qT_s/doT_s/lse_u/dlt_u` 同样全 extent 装载 + clamp/补 0；
   掩码臂保留 `m0+j < S_q`；`dq_c` 的原子 store 区段由同一 clamp 裁到**本 request 的** `S_q`
   （4-D 视图；3-D 下越界写是「原子加 0」的良性情形，但为口径统一也改 4-D）。
3. **因果对角块**：只有对角及相邻块进入掩码臂，其余走 §1.6.1 A2 快路；`loop_st = max(0,(n0-shift)//64)`
   保证不遗漏可见 query 块（最坏多算一个全掩码块，`P=0` 无副作用）。
4. **`S_q < S_kv` 与 `S_kv < S_q`**：`shift` 可为正/负；后者会让头部 query 看不到任何 key（`lse=-inf`），
   此时掩码全假 → `P=0` → 梯度为 0（不产生 NaN，已由 golden 的 `nan_to_num` 语义对齐）。
   **快路的安全性**由 A2 的充分条件保证：满足快路的块不可能包含「看不到任何 key」的行。
5. **`S` 极小**（`S < 96`）：全部走尾块路径；`grid = H*B*1`，所有 DMA 都被 clamp 到 `S` 行并补 0。
6. **`is_causal=False`**：`loop_st=0`，所有块都进入快路（仍受 clamped 尾块影响，见 1/2）；
   **掩码臂此时不得含因果项**（编译期守卫，§3.3 注 4）。注意：非因果下尾块与 victim 块的迭代数相同，
   3-D 视图的越界写会**立刻**暴露（§1.6.4 H），因此 L0-10（非因果 + B=2 + 尾块）是必跑用例。
7. **B≥2 的一般性**：4-D 视图下每个 request 的行区间 `[b·S_kv, (b+1)·S_kv)` 由张量维度天然隔离，
   与 `B` 的取值无关；`dk/dv` 的写回行数只依赖 `S_kv`。**Stage 3 的 L0 用例必须含 `B≥2`**（§8.3 L0-9/L0-10）。

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
| query 块方向 | 迭代（动态界） | **`for i in T.serial(n_half)`，体内 2-tile 静态展开**（半体 A/B） | 因果剪枝后界随 `n0` 变化；**v4**：2 版本由显式静态缓冲名给出，不再用 `T.Pipelined`（§6.3 A/B）。每半体 = `F → GA → EP → GB(k−1)`，循环后 drain |
| 5 个矩阵乘 | tile 级 | `T.gemm(a_l0a, b_l0b, c_l0c, transpose_A=…, transpose_B=…, clear_accum=…)` | Cube 显式 L0；`clear_accum` 区分「每迭代清零」与「跨迭代累加」 |
| L1→L0 暂存 | tile 级搬运 | 每个 GEMM 前一次 `T.copy(l1_region, l0_region)` | L0 路径的显式契约；`trans_A` 的 A 由编译器生成转置搬运 |
| 逐元素 epilogue | 向量并行（半区） | `with T.SimtVF(threads=128): for i, j in T.Parallel(48, 64)` | lane = query 轴（§1.6.3）；A2 由运行时块级 `if` 生成两支 |
| L0C↔UB / UB→L1 | tile 级搬运 | `T.dual_copy`（2:1 半区） | 整 tile `T.copy`(L0C→UB) 只落一个 AIV（v1 实测），必须半区 |
| 块尾 cast | 向量并行 | `T.Parallel(48,192)` / `T.Parallel(48,128)` | `dk` 乘 `scale` 后 cast |
| 块内 serial 子循环 | **使用 `T.serial`**（v4） | `T.serial(n_half)` | query 方向的主循环改为 `T.serial`（运行期上界）；L1→L0 暂存仍是 tile 级 `T.copy`（DMA），不需要 serial 循环 |
| 核内多任务（persistent） | **不使用 `T.Persistent`** | — | persistent 需把 Cube+Vector 混合体包进任务循环并做跨迭代同步，平台未认证；本设计按 §5.5 的「逻辑块超发」分核，1 块 = 1 任务 |

### 6.2 循环伪代码

见 §3.3（同一份结构，**已在设备上跑通**：v3 的 `design_probe_bwd_proto.py` 与 v4 的
`design_probe_dense_rotation.py`；L0 操作数按「最大形状分配 + 切片使用」，`kT` 最后重灌）。
**v4 的循环体是四个宏 + 一个 drain**：`F`（装载）/ `GA`（打分 2 GEMM）/ `EP`（epilogue）/ `GB`（梯度 3 GEMM），
半体按静态版本名展开，drain 按 `n_it` 奇偶选版本。

### 6.3 多缓冲与迭代环转（v4 核心，A5）

#### A. 为什么不能靠 `annotate_buffer_versions` / `num_stages`（四条独立证据，全部来自本任务族）

1. **调度器在本 kernel 上 `unsat`，并静默退化为拓扑序**。三份原始日志
   （`examples/mla_dense_bwd_perf/perf_opt/logs/probes/`）：`z3_dense.log` / `z3_varlen.log` /
   `z3_scheduler_probe.log` 各含 `Z3 scheduling failed: solver returned unsat, falling back to
   topological sort`；第三份还打印 `compiled ok: manual order/stage list [0]/[0] was accepted and had no
   effect` ——**手工 `order=`/`stage=` 被接受但被忽略**。本轮 v4 探针 `--compile` 同样复现该警告
   （§1.6.4 I-1），**两个体都复现**，说明它不是 v4 引入的。
2. **版本化只作用于「装载目的地」，且下标绑定当前迭代**。`bwd_c1_stages3.generated.cpp` 里
   `q/do/qT/doT/lse/delta` 的副本与消费者**两侧**都写 `(kq − loop_st) % 3`
   （`asc_copy_gm2ub_align(..., buf_dyn_shmem + ((…%3)*64) + 19968)` 与 VF 里的读是同一表达式）。
   也就是说 `num_stages` 只**加宽 WAR 距离**，从不改变「谁读哪个迭代的版本」——**它无法表达滞后一迭代的消费者**。
3. **机制层论证**：`EP(k) → GB(k)` 是一条 **RAW** 边。加版本只能放宽 WAR（写者不必等旧读者），
   对 RAW 一点作用都没有：`GB(k)` 无论等几个版本，都必须等 `EP(k)` 完成。**要消掉它只能改语句顺序**
   （把 `GB(k)` 挪到下一迭代），而语句顺序正是 `unsat` 之后被放弃的那一半能力。
4. **实测反证**：dense 的 `c4`（显式给填充缓冲两版本）墙钟 **−0.00%**；13 个参数/配置分支
   （`num_stages` 1…6 × 版本标注 4 组 × 操作数布局 2 种）里**没有任何分支超过 3% 噪声阈值**，
   最好者 `num_stages=4 + pT/dsT 两版本` = **−0.95%**（交错 A/B/A/B 3 对裁决，符号 6/6）。
   **版本数是有拐点的 WAR 手段，不是可无限叠加的重叠手段。**

#### B. v4 的做法：显式两版本 + 2-tile 静态展开 + 滞后一迭代

```
for i in T.serial(n_half):                     # n_half = ceil(n_it/2)，运行期
    kqA = loop_st + 2i                         # 版本 A
    F(kqA)→A ;  GA(kqA) ;  EP(kqA) 写 pT/dsT→A ;  if kqA > loop_st: GB(kqA-1) 读 B（另一版本）
    kqB = kqA + 1                              # 版本 B
    if kqB < loop_ed:
        F(kqB)→B ; GA(kqB) ; EP(kqB) 写 →B ; GB(kqB-1) 读 A
drain: GB(loop_ed-1) 读 (n_it 奇 ? A : B)
```
- 版本由**静态名字**（`_s0`/`_s1`）给出，**没有任何运行期缓冲索引**——这是必须 2-tile 展开而不能用
  「奇偶分支 + 运行期索引」的原因（L0 的 GEMM 操作数 region 原点必须静态，§1.6.3 D5-3）。
- 每个半体内部的顺序是 **`F → GA → EP → GB(k−1)`**；`GB(k−1)` 出现在 `EP(k)` 之后*只是因为代码顺序*，
  它读的缓冲与 `EP(k)` 写的缓冲**不是同一个对象**，故 InsertSync 不会在它前面插 `PIPE_V` 等待。
  **这就是全部机制**：不靠调度器，靠「给依赖边换一个缓冲身份」。

#### C. 依赖矩阵（每个 buffer 的 RAW / WAR 与版本需求）

| 边 | 类型 | 跨单元 | 距离 | 由什么保证 | 需要的版本 |
|---|---|---|---|---|---|
| `GM → q_s/do_s`（F）→ `GA(k)` | RAW | 否（AIC 内） | 0 | MTE2→MTE1 事件 | 1（功能）/**2（避免 WAR 全暴露）** |
| `GM → qT/doT`（F）→ `GB(k)` | RAW | 否 | **1 迭代** | MTE2 事件 + 版本 | **2** |
| `st_c/dpt_c → st_u/dp_u`（GA 内）→ `EP(k)` | RAW | AIC→AIV | 0 | fixpipe 事件 | 1 |
| `EP(k) → pT_s/dsT_s` → `GB(k)` | RAW | AIV→AIC | **1 迭代** | AIV→AIC 事件 + 版本 | **2** |
| `EP(k)` 读 `st_u` ↔ `GA(k)` 写 `st_u` | **WAR** | 跨单元 | 0 | AIV→AIC 事件（同迭代内） | 1 |
| **`GB(k−1)` 读 `pT_s` ↔ `EP(k)` 写 `pT_s`** | **WAR** | 跨单元 | **0（不同版本）** | 版本 ⇒ **无需等待** | **2** |
| `F(k)` 写 `q_s` ↔ `GA(k−1)` 读 `q_s` | **WAR** | AIC 内 | 1 | 版本 ⇒ 无需等待 | **2** |
| `GB(k−1)` 写 `dq_c` → `GB(k)` 写 `dq_c` | WAW | AIC 内 | 1 | mad 的 `clear_accum` + 原子写排队 | 1 |
| `dk_c/dv_c` 累加 | — | — | — | L0C 累加器 + `clear_accum=(kq−1==loop_st)` | 1 |

**关键一行**：`GB(k−1)` 读 `pT_s` 与 `EP(k)` 写 `pT_s` 之间是 **WAR** 且**恰好被两版本消掉**
（一个读 A、一个写 B）。这正是「AIC 不必等 AIV」的机制层原因——不是「编译器聪明」，是依赖边不存在了。

#### D. 版本数旋钮与实测取值

- **本轮实测 V=2**（`_s0/_s1`）：`B=2/S=8192/H=8` 1.870 µs、`B=2/S=4096/H=16` 1.968 µs（§1.6.4 I-4）。
- **V=3 是第一个调优旋钮**（装载缓冲 3 份、`pT/dsT` 仍 2 份，L1 393,216 B = 75.0%）：动机是长序列
  （8192）的残差（1.870 vs 4096 的 1.968，反而更小，说明长序列**没有**额外的填充延迟问题）——
  与 varlen 相反，dense 的长序列收益**更大**，故 V=3 的动机更弱。**未实测，不得写成收益**；仍是 Stage 4
  A/B 第一项，判定阈值：≤ 主选 × 0.95 才采纳。
- **不采用**「`T.Pipelined` + 显式两版本混用」：`T.Pipelined` 会把**它管理的**缓冲按当前迭代下标重写，
  与手写的 lag-1 消费者冲突（证据 A2）。两者只能二选一，本设计选「显式」。

#### E. 为什么不做两级旋转（把 `GA` 也提前一迭代）

v4 体在 `S=8192` 上 1.870 µs，已低于「`mte1 + mad + fixpipe` 三段相加」的 2.019 µs，说明这些段之间
**已经发生部分重叠**；墙钟由 `mte1 + mad = 1.449 µs` 的主链加上未覆盖的 fixpipe 尾部构成。再提前 `GA`
只能压已被压过的部分，而代价是确定的：`st_c/dpt_c` 双版本 ⇒ L0C **270,336 B（103.1%，越界）**，必须把
`dP` 并入 `st_c`（`2×24,576 + 49,152 + 73,728 + 49,152 = 221,184`，恰 84.4%）**并在 `GA` 内部引入一次
fixpipe→mad 串行**。**结论：S2 不做**（这是本版对调度族的量化淘汰，也与 `perf_feedback.md` 的
「dP 并入 st_c」建议的适用条件一致——那条建议对应 S2，不对应 S1）。

#### F. 这一版没有解决的问题（如实记录）

- 1.870 µs 仍比 `mte1+mad = 1.449` 高 **29%**：残差 = 未被藏掉的 fixpipe（0.570 的一部分）+ 块级固定开销
  （每块 `k/v/kT` 常驻装载 98,304 B ≈ 0.5 µs + 块尾 cast/写回）。**其分解未实测**（需 msprof 时间线），
  列入 §9.1 风险 17；方向是「fixpipe 与 mad 的交错度」而不是再旋转。
- `n_it ∈ {1,2}` 的短块（对角块、`S_q` 很小的 workload）只有 1–2 次迭代，首/末迭代的 `EP` 不与 `GB` 重叠。
  量化：`ds-v3-b2-8k` 上 22,016 块 × 2 次 × ~1.0 µs（可重叠部分）÷ 28 核 ≈ **1.6 ms ≈ 1.7%**，
  **已计入 §14.3 的投影区间**，不是额外风险。

### 6.4 尾块处理

统一口径（与 §3.3 注 2/注 3、§5.4 一致）：

- **GEMM**：不参与尾块处理——L0 路径的 M/N/K 取 allocation 形状（静态），region extent 用全 extent。
- **搬运**：GM→L1 / L1→GM / GM→UB 的越界尾部由平台 pass `AscendInsertOOBPadding` 在编译期改写成
  运行时 `min()` clamp，并对 GM→L1 的补 0 区段生成 `asc_fill_l1`。
  **clamp 的裁剪维度 = buffer 形状的行数**，故 GM 张量一律 4-D：`dk/dv` 的写回行数为
  `min(48, max(S_kv − sid·48 − n0, 0))`（按 request），而不是 `min(48, max(B·S_kv − …))`。
- **掩码**：`gi < S_kv` / `gj < S_q` 两项**必须**保留在掩码臂中（补 0 不等于梯度 0，见 §5.4 第 1 条）。
  快路分支通过运行时谓词 `n0+95 < S_kv ∧ m0+63 < S_q ∧ n0+95 ≤ m0+shift`（`shift` 项仅 `IS_CAUSAL`）保证两支等价（A2）。
- **写回隔离**：`dk/dv` 的块尾 store 是**普通 store（非原子）**，因此必须靠 4-D 视图把写回区间限制在本
  request 内；`dq_acc` 是 store-atomic（越界写等价于加 0），但同样用 4-D 以统一口径。
- **分支归属**：A2 是**运行时块级分支**（`n0` 来自核索引、`m0` 来自循环变量），生成两支 SIMT VF；
  `IS_CAUSAL` 是**编译期**分支，生成两套 kernel。

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
| **`EP(k)` 写 `pT_s[v]` ↔ `GB(k−1)` 读 `pT_s[1−v]`（v4，A5）** | **不需要同步**：两个不同的缓冲对象，依赖边不存在（这正是收益来源） | §6.3 C 的「关键一行」；本轮探针 5/5 逐位一致（§1.6.4 I-3） |
| **`EP(k+1)` 写 `pT_s[v]` ↔ `GB(k−1)` 读 `pT_s[v]`（v4，WAR，距离 1 迭代）** | **需要**：AIV 在覆盖前必须等 AIC 完成 `GB(k−1)` 的 L1→L0 读；由 InsertSync 按**缓冲身份**自动插入（AIC→AIV 事件） | 探针复跑 `dk/dv` 逐位相同 ⇒ 该边已被正确插入；若缺失会表现为**随机**坏块 |
| **`F(k+2)` 写 `qT_s[v]` ↔ `GB(k)` 读 `qT_s[v]`（v4，WAR，距离 1 迭代）** | 同上，AIC 内 MTE2→MTE1 事件 | 同上 |
| **`fixpipe(k+1)` 写 `st_u/dp_u` ↔ `EP(k)` 读 `st_u/dp_u`（v4，WAR，单版本）** | 自动（AIV→AIC）；**这是 A5 唯一的单版本风险点**——若 AIC 提前到达该点，会出现结构性停顿 | §9.1 风险 16 + Stage 4 A/B ②（`st_u/dp_u` 加第二版，UB 154,624 B = 60.9%） |
| **首/末迭代与 drain（v4）** | 无额外同步：`if kqA > loop_st` 与 drain 的奇偶分支都是**块级**谓词，两支在同一单元内 | 设备码同类形态（varlen v5 已上线） |
| 跨 block 写 `dk/dv`（同一 request 内的不同 key 块） | **无同步**（写区间互不相交） | 4-D 视图下每个 block 只写 `[n0+sid·48, +48)`，天然不相交；**3-D 视图会让相邻 request 的区间重叠且是普通 store ⇒ 竞态，靠同步无法修复**（§12-B1） |

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
| **v4 相对 v3 的 `dk`/`dv`** | — | **0（逐位相同）** | — | §1.6.1 A5 / §1.6.4 I-3 已机器验证（5/5 PARITY_PASS）。**Stage 3 必须把它作为回归判据**：同一输入下 v4 的 `dk/dv` 与 v3 冻结实现 `torch.equal`；不逐位相同即说明版本/drain 取错（§9.2） |

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
| **L0-9（v3 新增，必跑）** | **B=2, S_q=S_kv=512, H=4, bf16**（`512 = 5×96+32`，与探针同形） | **True** | **尾块跨 request 写回**：request 0 的尾块 n0=480 越界到 request 1 的 key 0…63；**重复 5 次逐次比对**，并断言「request 1 的 key 0…63 与 golden 一致」（竞态必须重复才可判定，§1.6.4 H）；`H=2` 亦应覆盖 |
| **L0-10（v3 新增，必跑）** | **B=2, S_q=S_kv=512, H=4**（同 L0-9，仅换 mask） | **False** | **非因果**：尾块与 victim 块迭代数相同 ⇒ 3-D 视图在此形态下 5/5 失败；4-D 必须 5/5 通过（本用例是 B1 的**最敏感探针**）；重复 5 次 |
| L0-11（v3 新增） | B=2, S_q=512, S_kv=608, H=2 | True | `shift>0` + B≥2 + 尾块（608 = 6×96+32）三重叠加 |
| **L0-12（v4 新增，必跑）** | **B=1, S_q=S_kv=128, H=1** | True | **`n_it = 2`（偶数）⇒ drain 走 `_s1` 分支**；同时覆盖 `kqB < loop_ed` 的真/假两支 |
| **L0-13（v4 新增，必跑）** | **B=1, S_q=S_kv=96, H=1** | True | **`n_it = 1`（奇数）⇒ drain 走 `_s0` 分支**；且 `if kqA > loop_st` 全程为假（首迭代无 `GB`） |
| **L0-14（v4 新增，必跑）** | **B=2, S_q=S_kv=8192, H=16**（可用 H=8 降内存） | True | **旋转的完整几何**：`n_it` 跨 1…128、drain 奇偶都会出现；逐位对照 v3 冻结实现（`dk/dv`），重复 **3 次** |
| **L0-15（v4 新增）** | L0-14 同 shape | False | 非因果：`loop_st=0`、全部块进快路 + 旋转；重复 3 次 |

L1/L2 与完整分层套件（`ds-v3-4k` / `ds-v3-b2-8k` / 多 dtype）交由 `tilelang-op-develop` 的 L1/L2/boundary。
**门槛口径（v3 强化）**：L0-9/L0-10 的判据是「**逐 request、逐次**」——单次运行通过不算通过
（3-D 视图在因果下 5/5 侥幸通过，见 §1.6.4 H）；`_OOB_PAD` 式的整张量尾部哨兵**不足以**发现该缺陷，
必须按 `dk[b, 0:64]` 分片比对。

**Stage 3 必须前置的最小探针（运行期，按序）**：

| # | 探针 | 输入 | 判定 |
|---|------|------|------|
| P1 | **L0 路径 `trans_A=True` 的 dQ**：单 GEMM | (96,64)·(192,96)ᵗ | 与 torch fp32 一致（rel < 5e-2，bf16 输入）——已在设计期由探针 2 与原型 PASS |
| P2 | **OOB clamp 语义**：`S_kv=100` 的 `dk/dv` 写回不带脏行 | L0-3 | 第 96–99 行正确、无越界写（哨兵值检测） |
| **P2b（v3 新增）** | **写回区段按 request 裁剪** | **L0-9/L0-10** | 编译产物里 `dk` 的 `asc_copy_ub2gm_align` 行数上界必须是 `min(48, max(S_kv − sid·48 − n0, 0))`（4-D），**不得**出现 `B·S_kv`；同时 request 1 的 key 0…63 5/5 与 golden 一致 |
| P3 | **尾块掩码收口**：`n0+i ≥ S_kv` 的行贡献为 0 | L0-3 | `dk/dv` 尾部行与 golden 在容差内 |
| P4 | **动态 `clear_accum`**：`kq == loop_st` 首块清零 | L0-4/L0-5 | 与「显式前置 `T.clear`」版本逐元素一致 |
| P5 | **store-atomic 契约**：`dq_acc` 由 host 预清零 | L0-1 | 与未清零对照组相比不翻倍 |
| P6 | **A2 两臂一致性** | L0-1/L0-2 | 快路与掩码臂对同一 (key,query) 块逐元素相同（`dk/dv` 可逐位、`dq` 用 1e-5） |
| P7 | **`b_hi` 复用顺序**：`kT` 必须在 `qT/doT` 之后重灌 | L0-8 | 去掉重灌即 dQ 全错（本版原型首版的失败即此，见 §9.2） |
| **P8（v3 新增）** | **MTE1 标定**：`aic_mte1_active_bw` | L0-1 / `ds-v3-b2-8k` | `217,088 B ÷ bw` ≤0.5 µs；若 >0.5 触发 §1.6.3 的 `Nk=80` 回退并回写 §1.6.4 G |
| **P9（v4 新增，必跑）** | **drain 奇偶**：`n_it` 为奇/偶各一次 | L0-12（偶）/ L0-13（奇） | 两者 `dk/dv` 与 golden 一致；**删掉 drain 后必须立即失败**（反例自检） |
| **P10（v4 新增，必跑）** | **版本取错自检**：把半体 A 的 `GB` 改读 `_s0`（当前版本） | L0-12 | **必须失败**（若仍通过，说明版本根本没起作用 ⇒ 检查是否退化为单个缓冲） |
| **P11（v4 新增，必跑，结构性判据）** | **设备码里 `GB` 之前不得有以 `PIPE_V` 为前驱的 `asc_sync_wait`** | L0-14 的编译产物 | 若出现 ⇒ 同步退化，A5 的收益归零，**必须回滚到 v3 体**（§9.1 风险 18）；AIC 内部的 `wait(PIPE_M, PIPE_MTE1)` / `wait(PIPE_M, PIPE_FIX)` 属正常 |
| **P12（v4 新增）** | **`st_u/dp_u` 的 WAR 停顿**：msprof 的 AIC 空闲窗口 | `ds-v3-b2-8k` | AIC 空闲 < 5% 视为无停顿；若 > 10% ⇒ 启用 `st_u/dp_u` 第二版（UB 154,624 B） |

**性能验收（设计期估算 + Stage 3 用 msprof 复核）**：

| 指标 | v3 生产实测 | 本版（v4）设计期值 | Stage 3 复核方式 / 通过线 |
|------|---------|-------------|--------------------------|
| 内层迭代单价 | **2.944 µs**（`ds-v3-b2-8k`，Stage 4 实测） | **1.870 µs**（探针，B=2/S=8192/H=8；同几何 v3 体探针 2.965 = 生产 +0.7%） | 同几何微基准或 msprof 反解；通过线 **≤ 2.30 µs**（= v3 × 0.78） |
| `ds-v3-b2-8k` 的 T_op | **154.4 ms** | **93–96 ms**（§14.3 的投影区间 93–101） | msopprof Task Duration；通过线 **≤ 115 ms**（该形状 6 个 launch 之和） |
| `aic_cube_ratio`（ArithmeticUtilization） | **25.1%**（MFU 24.6%） | **39–41%**（38.0 ms 忙时 ÷ 94.5–97 ms；cube 忙时与 T_op 无关，只由 FLOPs 决定） | msprof `PipeUtilization`；**通过线 ≥ 40%**（= 调度方目标，对应 T_op ≤ 95 ms；若实测落在 38–39% 属投影的保守端，不视为设计错误） |
| `aiv_vec_ratio` | 44.3% | **≈70%**（1.30 µs ÷ 1.87） | 同上；**警戒线 80%**（超过即说明 AIV 成为新墙，§9.1 风险 15） |
| `aic_mte2_active_bw` | 118.5 GB/s/核 | 若保持 118.5 ⇒ 0.72 µs/迭代（share 38%） | 同上；**< 85 GB/s 触发**「忙而慢」排查（varlen v5 的前例，§9.1 风险 14） |
| `aic_mte1_active_bw` | 0.680 µs（300–340 GB/s） | 同量级（工作集不变） | 同上；若 > 0.75 µs 说明 L1 争用，回看 V 旋钮 |
| **Stage 4 目标（v4 重定义）** | — | **≤115 ms 档已由 A5 单独达成（投影 93–96）；cube ≥40% 落在投影的乐观端**<br>档 B（≤62 ms / cube ≥60%）**在 1.449 µs 的 `mte1+mad` 主链下不可达**（62 ms 要求 1.22 µs），除非能压 MTE1 | 档 A 的判据是 `T_op ≤115 ms` 且 L0-14 逐位一致；档 B 的 6 条充分条件见 §1.6.4 F（其中 `MTE1 ≤0.20 µs` 已被 Stage 4 实测否证） |
| 80% Cube MFU（T_op ≈ 48 ms） | — | **不可达**：仅 `mte1 + mad` 主链（1.449 µs）就给出 `1.449 × 1.415e6 ÷ 28 = 73.2 ms` 的下限 ⇒ cube ≤ 52% | 不作为验收承诺；v3 的 6 条充分条件里 `MTE1 ≤0.20 µs` 已被 Stage 4 否证 |

---

## 9. 风险点与注意事项

### 9.1 已知约束与风险（按优先级）

| # | 风险 | 证据/来源 | 影响 | 处置 / 回退 |
|---|------|----------|------|------------|
| 1 | **L0 路径的 L1→L0A 转置搬运（`trans_A=True`）在生产 shape 的稳定性未实测**：设计期只在 (96,64)/(64,64) 与端到端原型上验证 | §1.6.4 D；原型设备码 `asc_copy_l12l0a_transpose` | `dQ` 全错或编译失败 | P1/P7 前置；回退①=把 `dQ` 改回「向量侧两趟转置（`epi2`）+ NT L1 GEMM」——实测代价 +1.64 µs/迭代（→ 约 250 ms，仍在 227 ms 边缘）；回退②=把 (Mq,Nk) 的 `dS` 用一次 identity GEMM 在 L0C 生成（+24 KiB L0C，与主选 84.4% 冲突，需先把 `block_n` 降到 80） |
| 2 | **L0C 上界比标称小**：249856 B（95.3%）设备报 507015，221184 B（84.4%）稳定 | §1.6.4；原型 bisect | 分块放大失败 | 设计红线 L0C ≤ 240 KiB；备选 112/64 为 86.0%；任何新配置先用最小原型跑一次 |
| 3 | **L0B 余量仅 4 KiB（93.75%）**，且 L0 单版本使每迭代 217,088 B 的 MTE1 与 mad 在 AIC 时间线上相加 | §4.6、§1.6.4 G | 无法加第三块 L0B / 无法双缓冲（`2×(24,576+384·Nk) ≤ 65,536` 仅 `Nk ≤21`）；`80% MFU` 要求 MTE1 ≤0.20 µs | P8 标定 `aic_mte1_active_bw`；回退：`block_n=80`（MTE1 −10.4%，填充 +20%，阈值 >0.5 µs，§1.6.3）；或把 `kT` 拆 `kT_lo/kT_hi` 放 L1 后按需重灌（每迭代 +36 KiB MTE1） |
| 4 | **`dQ` 的 48 KiB 原子写**：总量 69.6 GB（主选） | 结构性推算（§1.6.0 R4） | 若 L2 未驻留则退化为 HBM 瓶颈（MBU 0.74% 说明当前 L2 命中良好） | Stage 3 采集 `asc_copy_l0c2gm` 的耗时；备选 112/64 把原子量降到 59 GB |
| 5 | **`num_stages`/`annotate_buffer_versions` 无可判定的重叠收益**（13 个分支全在 ±3%，最好 −0.95%） | `perf_feedback.md` 实测第 3 条 / §6.3 A | 「靠流水消迭代内串行」不成立 | **v4 已改为显式语句顺序（A5）**；`num_stages` 从设计中删除。此项保留为「为什么不靠调度器」的证据 |
| 6 | **向量 epilogue 0.20 ns/score 高于 UB 端口下界（约 0.06–0.1）**，受指令依赖限制 | §1.6.3 代价模型 | 迭代内 36% 时间 | §1.6.3 的三级优化阶梯（float4 / `simd.vexpdif` / 手工展开），阈值 0.85× |
| 7 | `dq_acc` 依赖 host 预清零（store-atomic 累加） | `mode.py:set_atomic` + PL-950-atomic-splitk | 结果翻倍 | wrapper 必须 `torch.zeros`（现有三 launch 序列已如此）；P5 检查 |
| 8 | `T.exp` 为 SFU 近似 + fast-math | PL-950-softmax-boundaries 在 fp32 上以 1e-6 级误差通过 | bf16/fp16 输出超差 | 已在 §8.2 容差内预期；若超差，关闭 fast-math 或改用 fp32 高精路径后复测 |
| 9 | `lse` 域误用（把 log2 域当自然对数传入）会静默错 | 源算子显式乘 `LOG2E`，本设计不再乘 | 全量精度错 | 接口层不做 log/ln 换算；L0-1 会立即暴露 |
| 10 | A2 的两支 VF 代码体积与编译时间 | v1 探针 C（两臂源文本 4.3 KB） | 大 shape 编译时间增加 | 可接受；若编译时间成为问题，退化为单臂（带掩码，−49.5% 快路径收益） |
| 11 | 物理核数/型号须在 Stage 3 前复核 | 平台 `core_source`（2026-11-23）+ `observed_cores` | 分核判据失效 | Stage 3 首跑前用 `torch.npu.get_device_properties` 复核 28/56 |
| 12 | grid 上限 65535（`TRAP-950-grid-over-65535`） | 主选最大 grid 22016 | 若上游给出更大 B/H 会静默部分执行 | factory 侧断言 `grid <= 65535`，越界 raise（dense 下 `B*H*ceil(S_kv/96)` 全为编译期常量，可在 host 断言） |
| 13 | **80% Cube MFU 在「当前串行迭代结构」下不可达**（实测 3.42–3.49 µs/迭代 vs 目标 1.00），但**原理上可及**（完全重叠下界 1.24 µs，改进后 `0.80+MTE1`） | §1.6.4 F/G（管道级 `max` 口径；v3 删除 v2 的「即使优化掉搬运也不可达」） | 与调度方的期望不一致 | 设计目标改为三档：Stage 3 ≤227 ms；Stage 4 档 A ≤115 ms/cube ≥40%、档 B ≤62 ms/cube ≥60%；80% MFU 登记为「有条件的可达区间」，条件清单 6 条（含 `MTE1 ≤0.20 µs`） |
| **14** | **代码体积 ×2.50 ⇒ icache 压力**（v4 实测 20,341 → 50,879 B；`asc_mmad` 5→16） | §1.6.4 I-1 | 指令取指抖动，可能吃掉部分收益（未实测） | **绝对值仍小于已上线的 varlen v5（61,913 B）**，故暴露有上界；处置顺序：① 若 Stage 3 实测单价显著差于 1.870，先测「同 kernel 跑两遍」的第二次时间（icache 暖机的直接判据）② 减少 drain 的重复体（把 `if (n_it%2)` 改为运行期索引 `_s{parity}` 不可行，§1.6.3 D5-3）③ 把 `EP` 的两支（快路/掩码）合并为单支（−1 个 VF，代价 +6,144×2 次向量操作/迭代） |
| **15** | **AIV 向量占比升到 ≈70%**（1.30 µs ÷ 1.87），距 80% 警戒线 10 pt | §8.3 性能表；varlen v5 已到 0.768/0.800 | 若 Stage 4 压 AIC 更狠，AIV 会成为新墙（届时再旋转无收益） | **警戒线 80%**；处置：§1.6.3 的 epilogue 优化阶梯（`float4` / `simd.vexpdif` / 手工展开），判定阈值 0.85× |
| **16** | **`st_u/dp_u` 单版本的 WAR 停顿**（`fixpipe(k+1)` 必须等 `EP(k)` 读完） | §6.3 C、§7.2 | 若 AIC 提前到达，出现结构性停顿（探针未出现：`S=8192` 1.870 µs） | **P12 判据**：AIC 空闲 >10% ⇒ 给 `st_u/dp_u` 加第二版（UB 130,048 → 154,624 B = 60.9%，成本很低） |
| **17** | **1.870 µs 比 `mte1+mad`（1.449）高 29%**，残差分解未实测 | §6.3 F | 投影的保守度（若残差全来自 fixpipe，则 Stage 4 有 0.4 µs 的空间；若来自块级开销，则没有） | 列入 Stage 4 第一优先的 msprof 分解（AIC 五管道时间线 + 每块固定开销）；**方向是 fixpipe 与 mad 的交错，不是再旋转** |
| **18** | **同步退化：`GB` 前出现以 `PIPE_V` 为前驱的 `asc_sync_wait`** ⇒ A5 收益归零 | varlen v5 的判据（§1.6.4 I-1 的设备码形态） | 退回 3.0 µs 量级 | **P11 结构性判据**；一旦出现**立即回滚 v3 体**（不改其他任何东西），并登记为平台能力缺口 |
| **19** | **MTE2 变「忙而慢」**（varlen v5 的前例：80.5 → 55.5 GB/s，share 0.86） | varlen `RETROSPECTIVE.md`；本算子 dense mte2 = 118.5 GB/s（静态 extent） | dense 的填充是**静态长度**（比 packed 快 47%），且 L1 常驻集在 v3 就已是 311 KB（`c4` 实测 −0.00% 说明两版本已materialize）⇒ **A5 不再抬高 L1**，机制上弱于 varlen | **判据**：`aic_mte2_active_bw < 85 GB/s` 或 share > 60% ⇒ 先查 `lse/delta` 的 64 行小拷贝（合并为一次 512 B）与 q/do 的 dn2nz 交错；**不改变算法** |

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
| **`dk/dv` 写回只看张量全局 clamp** | 3-D packed 视图 + `B≥2` + `S_kv % 96 ≠ 0` | 尾块把 0 写进下一个 request 的 key 行，与真实值竞态（非因果下实测 5/5 失败，误差 0.33–0.40） | GM 张量改 **4-D batch 视图**（`dk[b, n0+sid·48:…]`），clamp 自动按 request（§5.4 第 1 条、§1.6.4 H） |
| `dq_acc` 未清零 | 复用 buffer | 梯度翻倍 | host `torch.zeros` |
| 认为「补 0 就等于梯度为 0」 | 尾块只靠 OOB fill | `exp(-lse) ≠ 0` → 尾部行出现伪梯度 | 掩码臂必须保留 `gi < S_kv` / `gj < S_q` |
| 认为「单次运行通过就没错」 | 尾块竞态 | 因果下 5/5 侥幸通过，换 mask/换派发即错 | 竞态用例必须重复 ≥5 次并逐 request 比对（§8.3 L0-9/L0-10） |
| **忘记 drain**（v4） | 只写循环体，以为最后一个 `GB` 已在半体里发出 | **最后一个 key/query 块的 `dk/dv` 少一次累加**（`S=192` 时误差 5.7e-2，varlen 首版探针实测） | 循环后无条件补 `GB(loop_ed−1)`；版本按 `n_it` 奇偶（P9） |
| **drain 只写奇数分支**（v4） | 以为 `n_half` 展开总会以半体 B 结束 | 偶数 `n_it` 时漏一次 `GB` | 两个分支都必须写（L0-12 覆盖偶数 / L0-13 覆盖奇数） |
| **`clear_accum` 忘了跟着 `GB` 走**（v4） | 保留 `(kq == loop_st)` | `dk_c/dv_c` 永不清零（读到脏 L0C）或清零两次 | 改成 `(kq−1 == loop_st)`；P4 复跑 |
| **用运行期下标选版本**（v4） | 想省代码体积写 `buf[kq % 2]` | L0 region 原点非 0 ⇒ 编译失败或静默错 | 版本只能由静态缓冲名给出（§1.6.3 D5-3）；代码体积的代价接受（风险 14） |
| **把 `b`/`h`/`n0` 挪进 `GB` 重算**（v4，dense 专属） | 为「解耦」在宏内重新解 `bid` | 与 `dq` 的写回行区间/`dk` 的写回区段不一致风险 | 三者由 `bid` 一次性解出，与 v3 相同（§1.6.1 A5 前提 4） |

### 9.3 特殊场景处理

非整除（§5.4）、极小 shape（§5.4 第 5 条）、`is_causal=False`（`loop_st=0`，掩码整体省为 A2 快路）、
`D_qk=192` 非 2 的幂（仅影响缓冲形状，不影响分形，`192 = 12×16`）、`S_q≠S_kv`（`shift` 正负两种情况
已列 L0-4/L0-5）、`S_kv < 96`（单块 + 全 clamp）。

---

## 10. 交付清单

### 10.1 目录结构

```
examples/mla_dense_bwd/_bwd_kernel/
├── DESIGN.md                       # 本设计文档（v4 = A5 旋转）
├── verify_equiv.py                 # §1.6.1 等价性机器验证（D-1，可重跑）
├── RETROSPECTIVE.md                # Stage 1 复盘（逐次追加）
├── _bwd_kernel.py                  # 算子实现（Stage 3 产出，不在本阶段；v1 版本待按本设计更新）
├── repro/                          # v1 运行期探针（L0C→UB 单 AIV、sid 捕获）
└── history_version/
    ├── design_v0.md                # 首版
    ├── design_v1_pre_cube.md       # v1（revision 1 输入）
    ├── design_v2_cube_proto.md     # v2（revision 2 输入）
    ├── design_probe_compile.py     # v1 编译探针
    ├── design_probe_bwd_cost.py    # v2 探针 1：分段单价（§1.6.4 A）
    ├── design_probe_l0_gemm.py     # v2 探针 2：L0 路径四类 trans（§1.6.4 D）
    ├── design_probe_bwd_proto.py   # v2 原型：C2 结构端到端（§3.3/§1.6.4 D）
    ├── design_probe_b1_tailrace.py # v3 探针：3-D vs 4-D 的 dk/dv 尾块隔离（§1.6.4 H）
    ├── design_probe_dense_rotation.py # v4 探针：base vs rot（编译/数值/逐位/单价，§1.6.4 I）
    ├── design_v3_pre_multibuf.md   # v4 的输入基线（冻结的 v3 设计）
    ├── b1_probe_run.log            # v3 探针完整输出（5 次 × 4 组合 + 单价）
    └── b1_device_code_bounds.txt   # v3 设备码 clamp 证据（1024 vs 512）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | 已完成（revision v3） | 含 §0 迁移分析、§1.6.0–§1.6.4 算法调研与实测分解、§11 v2 回退说明、**§12 v3 修订对照** |
| `verify_equiv.py` | 已完成（**2026-11-26 v4 重跑，4/4 EQUIV_PASS**：A1×2 + A2 + **A5 调度代数 478 点/17,846 事件，selftest 3/3**） | 等价性机器验证脚本（D-1） |
| **`history_version/design_probe_dense_rotation.py`** | **已完成（2026-11-26）：COMPILE_OK ×2 / NUMERIC 5/5 / PARITY 5/5（`dk/dv` 逐位，两次复跑）/ TIME 3.069→1.968 与 2.965→1.870** | **v4 的 A5 证据（§1.6.4 I）；完整输出 `history_version/dense_rot_probe_run.log`** |
| `history_version/design_probe_b1_tailrace.py` | 已完成（2026-11-25，3-D 非因果 5/5 FAIL / 4-D 5/5 PASS） | B1 的语义修复证据 |
| `history_version/design_probe_bwd_cost.py` | 已完成（2026-11-25，9 个分段单价） | v1 瓶颈分解 |
| `history_version/design_probe_l0_gemm.py` | 已完成（2026-11-25，3/3 PASS） | L0 路径 trans 组合 |
| `history_version/design_probe_bwd_proto.py` | 已完成（2026-11-25，数值 PASS + 3.46 µs） | C2 结构原型（3-D 视图） |
| `_bwd_kernel.py` | **待按本设计更新（Stage 3）** | 在 v3 实现的 4-D 视图基础上做 **1 处结构改动**：把 query 循环体拆成 `F/GA/EP/GB` 四个宏 + **`T.serial` 2-tile 展开 + drain**，并把 6 个装载缓冲与 `pT_s/dsT_s` 显式两份；**数学、grid、L0C/L0A/L0B、4-D 视图、原子写回、块尾写回一律不动** |

### 10.3 命名规范

- 项目目录：`examples/mla_dense_bwd/`（TileOPs 迁移项目，按算子名聚合）
- 算子目录：`_bwd_kernel`（对应源工厂函数名；同项目下还有 `_delta_kernel`、`_dq_land_kernel`）
- 实现文件：`_bwd_kernel.py`

### 10.4 实现顺序（Stage 3 建议）

0. ✅（**首跑方式固定**）`cd /tmp && source …/env-950pr.sh` + `TMPDIR=/workspace
   /workspace/venv-ascend950/bin/python <kernel>.py`（平台默认解释器无 torch/tilelang、cwd 在仓根内会遮蔽
   site-packages，见 §3.5.4 的偏离登记表；Stage 3 不得改用平台 `prototype_command` 的默认值）
1. ✅ 设计文档 + 等价性验证 + 四个探针/原型（含 v3 的 `design_probe_b1_tailrace.py`）
2. ⬜ 把 §3.3 的**旋转结构**落到 `_bwd_kernel.py`（v4 最小改动：拆 `F/GA/EP/GB` 四个宏 + `T.serial` 2-tile +
   drain + 6 个装载缓冲与 `pT/dsT` 显式两份 + `clear_accum` 后移；**不改**数学/4-D 视图/grid/L0C/L0A/L0B，
   diff 见 §13.3）
3. ⬜ P1/P7（`trans_A` 与 `b_hi` 复用顺序）→ **P2b（写回 clamp 按 request，设备码 + 逐 request 比对）**
   → P2/P3（尾块 clamp 与掩码收口）
4. ⬜ P4/P5（动态 `clear_accum`、store-atomic 契约）→ L0-1…L0-8 → **L0-9/L0-10/L0-11（B=2 + 尾块，各 5 次）**
5. ⬜ L1/L2 与 msprof 采集（`aic_cube_ratio` / `aiv_vec_ratio` / `mte2_bw` / **`mte1_bw`（P8）**），回写 §8.3 表
6. ⬜ **v4 结构落地**：§3.3 的 `F/GA/EP/GB` + `T.serial(n_half)` + drain（最小改动的 diff 见 §13.3）
7. ⬜ **v4 的必跑回归**：P9（drain 奇偶）/ P10（版本取错自检）/ **P11（`GB` 前不得有 `PIPE_V` 等待——结构性判据）**
   / P12（AIC 空闲窗口）→ L0-12…L0-15（各重复 3 次，`dk/dv` 与 v3 **逐位**一致）
8. ⬜ Stage 4：§1.6.3 的 epilogue 阶梯 + `V=3`（L1 75%）+ `st_u/dp_u` 第二版（若 P12 触发）+ 112/64 备选裁决

---

## 11. 回退说明（revision v2，**历史，备查**）

### 11.1 相对上一版的关键调整

| # | 维度 | v1（`design_v1_pre_cube.md` / 当前实现） | v2 | 驱动依据 |
|---|------|----------------------------------------|----|---------|
| 1 | **`dS` 的转置** | 向量段跨步写 `ds_u[j,i]` 生成 `(Mq,Nk)` 副本，`dQ` 走 NT L1 GEMM | **删除该 tile**；`dQ` 用 `transpose_A=True` 的 L0 GEMM 直接消费 `dsT` | 探针 1：该 scatter 占内层迭代 63%（5.950→2.196 µs）；§0.4 #6 的「L1 路径 NT-only」判定被证伪（L0 路径支持 `trans_A`） |
| 2 | **GEMM 路径** | 5 个 `T.gemm(A_l1,B_l1,C_l0c)`（L1 操作数） | 5 个全部显式 L0（`alloc_l0a/l0b` + 手工 `T.copy`） | 要 `trans_A` 就必须用 L0；且两条路径**混用会互相覆盖 L0**（原型 bisect 实测 `dk/dv` 错 4.8e-1） |
| 3 | **tile** | `block_n=64, block_m=64`（L0C 62.5%） | `block_n=96, block_m=64`（L0C 84.4%） | 每 query 行的 1280 B 填充按 key 块重复：64→96 把填充总量 171.8 GB → 116.6 GB（−32%） |
| 4 | **向量输出 tile 数** | 3（`p16`/`dsT`/`ds`），14 B/score | 2（`p16`/`dsT`），12 B/score | 由 #1 直接得到；UB 每迭代单版本 28,928 → 37,376 B（半区 32→48 行；同口径比较见 §4.5） |
| 5 | **UB 半区** | 32 行（`Nk/2`） | 48 行 | 跟 `Nk=96` 走；`T.Parallel(48,64)` 实测编译成 `float2` |
| 6 | **流水** | 声称「Stage 4 提 `num_stages=2`、L1 双版本」 | 声明双版本但**不承诺收益**（v3 统一为 `num_stages=2`；实测 3.391–3.421 vs 3.446–3.486 µs，−1.6%，在 ±2% 波动内）；重叠改由「AIC/AIV 天然并发 + Stage 4 显式双缓冲」承担 | §1.6.4 D / §6.3 |
| 7 | **L1 端口口径** | 每迭代 280,056 B（复算）/ 24.4 FLOP/B；全量 591.9 GB | 325,625 B / 31.4 FLOP/B；全量 460.8 GB（逐项表见 §1.6.0 R3；v3 把 L1→L0 行从误写的 221,184 改为 217,088） | 结构变化后的重算 |
| 8 | **地板项判定** | 「L1 端口操作数流是地板项」，估算下界「待核验」 | **管道级 max = 完全重叠下界**：Cube 40.5–44.5 ms / 填充 59.5 ms / 向量 62.4 ms / fixpipe 13.5 ms / UB→L1 11.9 ms；实测外推 173–176 ms（v3 修正：串行合成 204 ms 是上界参考，不是下界） | 本轮探针给出可引用常数（70 GB/s/核、0.201 ns/score、92/52 GB/s/AIV） |
| 9 | **性能目标** | 未给定量目标 | Stage 3 **≤227 ms**（外推 173–176 ms）；Stage 4 两档：**A ≤115 ms/cube ≥40%**、**B ≤62 ms/cube ≥60%**；80% MFU 改成「当前串行结构下不可达、跨迭代流水后原理上可及」（v3 修 B2） | 调度方的 ≥2× 硬目标 + 本轮管道级下界 |
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

---

## 12. 回退说明（revision v3，**历史，备查；v4 的输入基线**）

> 输入：Stage 2 的 `REVIEW.md`（结论「不通过」，3 个阻塞项 + 5 条建议）。
> 本轮**不推翻 C2 的结构**（5 个显式 L0 GEMM / `trans_A` 复用 `dsT` / tile 96-64 / 2-tile 合并写 /
> 半区 `dual_copy`）——检视方复跑 3.489 µs/迭代、176 ms 成立，且本轮 3-D/4-D 两形态复跑同值
> （3.446–3.486 / 3.461–3.509）。改动集中在**语义修复（B1）**、**下界推导（B2）**、**算术口径（B3）**
> 与 5 条建议。

### 12.1 三个阻塞项

#### B1 `dk/dv` 块尾写回跨 request 覆盖 → 改 4-D batch 视图（语义修复）

| 项 | 内容 |
|---|---|
| **问题** | 源算子逐行守卫 `if n0[0]+i < len_kv`（`bwd.py:255-260`）；v2 只靠平台 OOB clamp，而 clamp 的裁剪维度是 **buffer 形状**（整张 `B·S_kv`）。`block_n=96` 让 `ds-v3-b2-8k`（B=2, `8192 = 85×96+32`）的尾块把 0 写进下一个 request 的 key 0…63，与真实值写入竞态；v1 的 `BN=64` 对 5 条 workload 全整除 + L0 用例全 B=1 + 哨兵只查整张量尾部 ⇒ 从未暴露 |
| **改了什么** | ① kernel 签名的 4 个 token-major GM 张量改 **4-D batch 视图** `(B,S,H,D)`，寻址 `k[b, n0:n0+96, h, :]` / `dk[b, n0+sid·48 : …, h, :]`（host `.view` 零拷贝，与源算子 `view(B*S,H,D)` 同性质）；② 该修复**同时根治读+写两侧**（装载 clamp 也变按 request）；③ 非「只改 OOB clamp」——clamp 的语义取决于张量 rank，故必须在签名层面改视图 |
| **在哪节** | §0.2（签名口径）、§0.6 R2/R5（边界论证）、§1.4 步骤 4、§3.2 行 1/2/9/10、**§3.3（伪代码 + 注 3）**、§3.5.1（新增一行）、§4.1/§4.2、**§5.4 第 1 条（重写）**、§6.4、**§8.3（L0-9/L0-10/L0-11 + P2b）**、§9.2（两条常见错误）、**§1.6.4 H（新证据）** |
| **证据（新增探针）** | `history_version/design_probe_b1_tailrace.py`（同一 C2 结构，仅签名/寻址不同，B=2/S=512/H=4，各跑 5 次）：<br>• **3-D 非因果：5/5 FAIL**，`dk` 0.33–0.39、`dv` 0.33–0.40，误差**全部**落在 request 1 的 key 0…63；<br>• **3-D 因果：5/5 侥幸通过**（尾块迭代数最少、总先完成）——正是「单次运行不可判定」的实证；<br>• **4-D 因果/非因果：5/5 PASS**（≤2.1e-3），且因果下与 3-D **逐位同值**；<br>• **4-D + `num_stages=2`：causal/非因果 2/2 PASS**；<br>• 单价 3-D 3.446–3.486 vs 4-D 3.461–3.509 µs ⇒ **签名零代价** |
| **设备码证据** | `min(48, max(1024 − sid·48 − n0 − b·512, 0))`（3-D，上界 = `B·S_kv`）→ 尾块写满 48 行（全局 528…575，越界 64 行）<br>`min(48, max(512 − sid·48 − n0, 0))`（4-D，上界 = `S_kv`）→ `sid=1` 算 0 行、`sid=0` 只写 32 行，**与逐行守卫同义**。存于 `history_version/b1_device_code_bounds.txt` |
| **用例** | §8.3 增 **L0-9（B=2/S=512/H=4, causal, 重复 5 次，与探针同形）**、**L0-10（同 shape, 非因果, 5 次）**、L0-11（B=2, `S_kv=608`）；门槛口径改为「逐 request、逐次比对」，并说明 `_OOB_PAD` 式整张量哨兵不足 |

#### B2 「80% Cube MFU 不可达」的推导错误 → 改为「当前串行结构下不可达」+ 达成条件清单

| 项 | 内容 |
|---|---|
| **问题** | v2 把 Cube 0.88 + 填充 1.18 + 向量 1.24 **求和**当物理下界；三个分项落在**不同执行单元**，正确下界是完全重叠的 `max = 1.24 µs`（62.7 ms）。v2 自己也实测出 `chain_notr 2.190 < Σ分段 2.992`，直接否定可加性 |
| **改了什么** | ① §1.6.4 F 重写为**管道级表**（AIC/mad、MTE2、AIV、FIX、MTE、MTE1、原子），并列三个口径：实测 3.42–3.49（173–176 ms）/ 串行合成 4.04（204 ms，非下界）/ **完全重叠下界 1.24（62.7 ms）**；② 两项改进后 `0.80 + MTE1`（40.5–50.6 ms）；③ **删除**「即使把搬运与同步全部优化掉也不可达」；④ 结论改为「**当前串行迭代结构下不可达**」+ 6 条达成条件清单（填充减半 / epilogue ≤0.62 / 跨迭代流水 / **MTE1 ≤0.20** / 三项 ≤0.80 / 五管道真并发）；⑤ Stage 4 目标并列两档 `≤115 ms / cube ≥40%` 与 `≤62 ms / cube ≥60%`；⑥ 删掉「`8.3% × 455.3 ms = 37.8 ms ≈ 379 TF/s`」的超证据换算（超出实测峰值 6%），只保留 `gemm5 = 0.587 µs` 的证据 |
| **在哪节** | **§1.6.4 F（重写）**、§1.6.4 B2（换算删除）、§1.6.0 roofline（两个下界分列）、§1.6.3（主选单价带）、§8.3 性能验收（两档 + 80% MFU 改「有条件可达」）、§9.1 风险 13（重写）、§11.1 #8/#9（重写） |
| **新增的关键结论** | MTE1 与 mad 在 AIC 时间线上**相加**（L0 单版本），故 `80% MFU` 的必要条件是 **MTE1 ≤0.20 µs**（L1→L0 ≥1.09 TB/s）；L0 双版本在本平台 L0B 容量下**不可行**（`2×(24,576+384·Nk) ≤ 65,536` 仅 `Nk ≤ 21`），故不能靠双缓冲绕过 |

#### B3 R3 访存与多处算术不自洽 → 逐项复算并统一口径

| 项 | 原文 | 复算 | 出处/公式 | 落在哪节 |
|---|---|---|---|---|
| C2 GM 访存 | ≈281 GB | **189.8 GB** | `116.6 + 2.2 + 69.6 + 1.4`（逐项公式见 §1.6.0「R3 口径与出处」表） | §1.6.0 R3 |
| C1 GM 访存 | ≈280 GB | **281.5 GB** | `174.2 + 2.1 + 103.9 + 1.3` | §1.6.0 R3 |
| C2 每迭代 L1→L0 | 221,184 B | **217,088 B** | 10 条 copy 逐项相加（原值其实是 L0C 五累加器容量，串行；与同表「合计 325,625」自相矛盾） | §1.6.0 端口分解 |
| fixpipe 全量 | 12.6 ms（式得 378 ms） | **13.5 ms** | `1.415e6 × 49,152 ÷ (92e9 × 2 × 28)`（补 `×28`） | §1.6.0 roofline |
| UB→L1 全量 | 11.2 ms（式得 334 ms） | **11.9 ms** | `1.415e6 × 24,576 ÷ (52e9 × 2 × 28)`（补 `×28`） | §1.6.0 roofline |
| 每 query 行填充次数 | ×86 / ×128（非因果上界） | **因果均值 43.7 / 64.5** | `S_kv/(2·Nk) + 1` | §1.6.0 R3、§1.6.4 B3 |
| C1 UB 占比 | 71.2%（176 KiB，无法复现） | **47.2%（119,808 B）** | `2×29,184 + 61,440` | §1.6.0 R4、§4.5 |
| 单价数字 | §11.1 #6「3.439 vs 3.464」vs §1.6.4「3.469/3.391」 | **3.446–3.486（stages=1）/ 3.391–3.421（stages=2）** | 三次独立取样，±2% 跑间波动 | §1.6.4 D、§11.1 #6、§6.3 |
| FLOPs | C1 14.34 T / C2 14.48 T | **14.44 T / 14.49 T**（解析下界 14.34 T） | `N_it × (scores×1664 + 64×256)` | §1.6.0 R3 |
| 迭代数 | 1,415,000（未说明） | 口径值 1.415e6；解析 **1,420,032** | `Σ_{t=0}^{85}(128−⌊1.5t⌋)×256`，差 −0.35%，聚合量统一用口径值 | §1.6.0 R3 |
| 全量端口 | v1 590.6 / C2 460.8 GB | **591.9 / 460.8 GB** | `N_it × 每迭代端口` | §1.6.0 端口分解 |

口径纪律（写进 §1.6.0）：**每个聚合量必须给「每迭代单价 × N_it ÷ 28」或逐项和**，两者互为校验；
任何「三项之和 / 两项之和」都必须标注是**上界参考**而非下界。

### 12.2 五条建议

| # | 建议 | 处理 | 落在哪节 |
|---|---|---|---|
| 1 | §3.3 漏 `is_causal` 守卫 | `loop_st = … if IS_CAUSAL else 0`；掩码臂的因果项包在 `if IS_CAUSAL:` 内；§6.1/§6.4 注明「编译期常量、生成两套 kernel」；§5.4 第 6 条点明非因果下不得含因果项 | **§3.3（代码 + 注 4）**、§5.4、§6.1、§6.4 |
| 2 | L0 单版本使 221 KB MTE1 全串行 → 量化 + 预置 `Nk=80` 回退 | 新增 **§1.6.4 G**：流量 217,088 B；**严格上界 0.74 µs**（由 v1 `gemm5 = 0.587` 按流量比 ×1.26 折算，该值已含 L1 路径的 L1→L0）、**点估计 0.1–0.3 µs**（`gemm5` 与同几何 mad 解析值 0.534 只差 10% ⇒ 模板内高度重叠）；**敏感性表**给出 `MTE1 ≤0.20` 是 80% MFU 的必要条件；§1.6.3 预置 `Nk=80`（MTE1 −10.4%、MTE2 +20%，**净亏**，阈值 >0.5 µs 才启用），§8.3 增 P8（`aic_mte1_active_bw` 标定）。**同时更正检视建议里「Nk=80 可留出 L0 2 版本余量」的算术**（`2×55,296 > 64 KiB`，任何有用 tile 都不行） | **§1.6.4 G（新增）**、§1.6.3、§4.6、§6.3、§9.1 风险 3、§8.3 P8 |
| 3 | `num_stages=1` 与声明 2 版本矛盾 | 二选一取 **`num_stages=2` + 2 版本**：§3.3/§3.2 行 11/§4.8/§6.1 全部改 `num_stages=2`，并说明 v1 注释的适用范围；实测 3.391–3.421 vs 3.446–3.486（−1.6%，±2% 内），不承诺重叠收益 | **§3.3 注 8**、§3.2、§4.8、§6.1、§6.3 |
| 4 | 6 处文档数字/口径不一致 | ① §3.4 scatter 计数改「本版 2 个（v1 为 3 个）」、L1→L0 拷贝改「每迭代 10 条」；② §4.5 的 `57,856→37,376` 改为同口径（v1 单 28,928/双 57,856；C2 单 37,376/双 74,752）；③ §0.4 #11 由「保留」改「等价替换」，§0.5 统计改 6 保留/7 等价替换；④ §11.1 #6 与 §1.6.4 D 统一为区间；⑤ A2 快路收益标注「全方阵 49.5% / 实际执行块 ≥98%」；⑥ §1.6.1 #4 弃选理由改为「12,288 次 fp16 乘 > 6,144 次 fp32 乘 + 多一次舍入」（原「需两份副本」不成立）；另把 `delta` 拆分代价从 +5.7% 更正为 **+23.0%** | §3.4、§4.5、§0.4/§0.5、§11.1、§1.6.0 R1、§1.6.1 |
| 5 | 探针复现与平台 `execution` 偏离未登记 | §3.5.4 增「执行上下文偏离登记」表：解释器 `/workspace/venv-ascend950/bin/python`（平台默认解释器无 torch/tilelang）、cwd 必须在仓根之外（dev-root 遮蔽 0.1.15）、`source env-950pr.sh`、`TMPDIR=/workspace`、单卡；`§10.4` 把该命令固化为 Stage 3 首跑方式 | **§3.5.4**、§10.4 |

### 12.3 为何不会再犯同一错误（v3）

- **「clamp 能处理尾块」这类平台行为必须有 rank 级的设备码证据**：v2 把 `AscendInsertOOBPadding` 当成
  「按语义维度裁剪」，实际它按 buffer 形状裁剪。今后任何依赖该 pass 的边界论证，都要贴出**该张量 rank 下**
  的 `min(...)` 实参（本轮贴了 4 条：3-D/4-D × 读/写）。
- **竞态类缺陷不能用「实测通过」否证**：因果下 5/5 侥幸通过、非因果下 5/5 失败——同一份代码。今后
  涉及跨 request/跨 block 并发写的改动，门槛用例必须**重复 ≥5 次 + 逐段比对**（L0-9/L0-10）。
- **下界一律按「执行单元」列管道，取 `max`；同一单元内的串行链才相加**：v2 的错误是把不同单元的
  分项求和。§1.6.0 roofline 与 §1.6.4 F 现在都用管道表，且明确标注哪两项是同一单元（`MTE1 + mad`）。
- **聚合量的每个数字必须能由公式复现，且与同表合计自洽**：本轮发现 v2 的端口行（221,184）与同表合计
  （325,625）相差 4,096，正是「抄错一个数」的典型。§1.6.0 新增「R3 口径与出处」表逐行给公式。
- **建议本身也要做容量/带宽的二次核算**：检视建议「Nk=80 留出 L0 双版本余量」按 `2×(24,576+384·Nk)`
  不成立，本轮据实更正而不是照抄。

### 12.4 是否仍建议 C2

**是，仍建议 C2（经 B1 修复后的 4-D 形态）进 Stage 3**，理由：

1. 结构与数值已被两方独立复跑：检视方 3.489 µs/176 ms、本轮 3.446–3.509 µs（3-D/4-D 同值），
   且 4-D 修复后 causal/非因果 **5/5 + 2/2 全 PASS**，语义与源算子逐行守卫一致；
2. 相对 v1 的收益已被多重证据锁定：内层单价 6.03 → 3.42–3.49 µs、GM 访存 281.5 → 189.8 GB（−33%）、
   迭代数 −33%（2,113,536 → 1.415e6）、cube 占比 8.3% → ≈25%；Stage 3 的 ≤227 ms 有 1.3× 余量；
3. 剩余风险都已量化并带阈值：L0C 84.4%（红线 240 KiB）、L0B 93.75%（回退 Nk=80，阈值 >0.5 µs）、
   L0 `trans_A` 生产稳定性（P1/P7）、`dq` 原子 ≤0.91 µs（未标定，P5/msprof）、MTE1 ≤0.74（P8）；
4. 改动面小且可控：Stage 3 只需在 v1 现有 `_bwd_kernel.py` 上做「5 个 GEMM 改 L0、删 `ds_u`、
   `Nk=96`、`b_hi` 重灌顺序、**GM 张量改 4-D 视图**、`num_stages=2`」六处，全部有原型对照；
5. 唯一需要用户/调度方认知调整的是性能目标：**Stage 4 分两档**（≤115 ms / ≤62 ms），
   80% MFU 从「不可达」改为「有 6 个前置条件的可达区间」。

---

## 13. 回退说明（revision v4：A5 旋转调度）

> **输入**：Stage 4 的 `[DESIGN_LIMIT]` 归因（`examples/mla_dense_bwd_perf/perf_opt/perf_feedback.md`）
> 与同族 varlen v5 的已上线结构（`examples/mla_varlen_bwd/_bwd_kernel/DESIGN.md`），不是检视意见。
> **本轮不推翻 v3 的任何结构**（5 个显式 L0 GEMM / `trans_A` 复用 `dsT` / tile 96-64 / 2-tile 合并写 /
> 半区 `dual_copy` / 4-D batch 视图 / grid / 原子 `dq`），**只改语句时序**：把 `GB(k)={dV,dK,dQ}`
> 从 `EP(k)` 之后移到 `EP(k+1)` 之旁。

### 13.1 相对上一版（v3）的关键调整

| # | 维度 | v3（`design_v3_pre_multibuf.md` / 当前 `_bwd_kernel.py`） | v4 | 驱动依据 |
|---|------|------|----|---------|
| 1 | **迭代内语句顺序** | `F(k) → GA(k) → EP(k) → GB(k)`（`GB` 紧跟 `EP`） | **`F(k) → GA(k) → EP(k) → GB(k−1)`**（滞后一迭代） | `perf_feedback.md` 实测第 1 条：链长 2.94 µs ≈ 墙钟，`mad→fixpipe→AIV 1.35→scatter→mte1→mad` 全串行 |
| 2 | **循环 API** | `T.Pipelined(loop_st, loop_ed, num_stages=2)` | **`T.serial(n_half)` + 2-tile 静态展开**（`kqA = loop_st+2i`、`kqB = kqA+1`） | §6.3 A 的四条证据（Z3 `unsat`、版本索引绑定当前迭代、RAW 边与版本无关、13 个分支 ±3%） |
| 3 | **缓冲版本来源** | `annotate_buffer_versions({buf:2})`（编译器按 `kq%2` 重写） | **静态缓冲名 `_s0/_s1`**（trace 期两个对象）；6 个装载缓冲 + `pT_s/dsT_s` | §1.6.3 D5-1/D5-2 |
| 4 | **`clear_accum`** | `(kq == loop_st)` | **`(kq−1 == loop_st)`** | 首迭代无 `GB`；`GB(loop_st)` 仍是第一次清零（§1.6.1 A5） |
| 5 | **首/末迭代** | 由 `T.Pipelined` 的循环界处理 | **首迭代 `if kqA > loop_st` 跳过 `GB`；末迭代由 drain 无条件补**（版本按 `n_it` 奇偶） | §6.3 B、§9.2 |
| 6 | **UB 预算** | 166,912 B（65.7%，含 `st_u/dp_u/p16_u/dsT_u` 的第二版） | **130,048 B（51.2%）**：只有 `lse/delta` 两版本 | §4.3 版本判定表（生存期同迭代 ⇒ 不需要版本） |
| 7 | **L1 预算** | 311,296 B（59.4%，编译器版本化） | 311,296 B（59.4%，**显式**两份）——**总量不变，来源变** | §4.6 |
| 8 | **L0C/L0A/L0B** | 221,184 / 49,152 / 61,440 B | **逐字节相同** | §4.6；A5 不需要 L0C 双版本 |
| 9 | **内层单价** | 2.944 µs（生产，Stage 4 实测） | **1.870 µs**（探针；同几何 v3 体探针 2.965 = 生产 +0.7%） | §1.6.4 I-4 |
| 10 | **`T_op` 目标** | 154.4 ms（实测） | **投影 94.5–105 ms**（central 98） | §14.3 |
| 11 | **roofline 口径** | 「`MTE1 ≤0.20 µs` 是 80% MFU 的硬条件」 | **撤销该条件**（实测 0.60–0.68 µs）；新地板 = `mte1 + mad = 1.449 µs`；80% MFU 不可达 | `perf_feedback.md` 实测第 2 条；§1.6.4 F |
| 12 | **对 Stage 4 建议的处置** | — | **采纳其归因、否决其处方**：`perf_feedback.md` 建议「把 `dP` 并入 `st_c` 以腾出 L0C 双版本」——那是**两级旋转（S2）**的必要条件，而 S2 在 v4 的墙钟下收益 ≈0（§6.3 E）。S1（本版）只需 L1 两份 | §6.3 E、§1.6.0 R1 #12 |
| 13 | **验证** | L0-1…L0-11、P1–P8 | **+ L0-12…L0-15、P9–P12**，并把「`dk/dv` 与 v3 逐位一致」写进 §8.2 的判据 | §8.3 |
| 14 | **代码体积** | 20,341 B | **50,879 B（×2.50）** | 已量化的代价（§9.1 风险 14） |

### 13.2 为何不会再犯同一错误

- **不再把「编译器能自动流水」当作设计前提**：v3 的 §6.3 把「Stage 4 的显式双缓冲」当作可选 A/B，而
  实际该kernel 上 Z3 `unsat` 后语句顺序根本不被旋转（四条独立证据，§6.3 A）。v4 把重叠**显式写进语句
  顺序**，不依赖任何调度器能力；`num_stages` 从设计里删除（只留一个 `n_versions` 键做 A/B 接口）。
- **「跨迭代重叠」的判据是依赖边的身份，不是版本数**：`EP(k)→GB(k)` 是 RAW，加多少版本都没用；
  唯一有效手段是让 `GB(k−1)` 读**另一个对象**。这条已写进 §6.3 C 的依赖矩阵并作为设计的第一原理。
- **不把「完全重叠的 max 下界」当承诺**：v3 的 62.7 ms 下界（`max` 口径）本身没错，但它被当成可达目标，
  而 `MTE1 ≤0.20 µs` 的前提实测不成立（0.60–0.68）。v4 的投影只用**实测单价 × 迭代数**外推，并给出
  探针与生产的对账（+0.7%）与区间下限，不再给「理论下界」。
- **对上游建议做适用条件核对**：`perf_feedback.md` 的「dP 并入 `st_c` + 双版本」在 S2 下是必需的，
  在 S1 下是**多余且有害**的（会引入 `GA` 内部的 fixpipe→mad 串行）。本轮据实否决并写明理由，
  而不是照抄（§6.3 E）。
- **「调度级等价」必须两级机器验证，且验证器本身要有反例自检**：代数层（`check_a5`，478 调度点 /
  17,846 事件 / selftest 3/3）+ 设备层（`dk/dv` 逐位，5/5）。varlen 首版探针漏掉 drain 的偶数分支
  正是靠「反例会失败」这一类检查抓住的。
- **族内结论的方向不能外推**：varlen 是**长序列收益更小**（−26.5% vs −33.9%），dense 实测是
  **长序列收益更大**（−36.9% vs −35.9%）。因此本版对 dense 的每个关键数字都重新取证，没有把 varlen 的
  2.092 / 3.134 直接搬过来（§1.6.4 I-4）。

### 13.3 Stage 3 的最小改动 diff（v3 实现 → v4）

1. 把 query 循环体拆成 4 个 `@T.macro`：`F` / `GA` / `EP` / `GB`（内容与现有语句**逐行相同**）；
2. 缓冲声明：`q_s/qT_s/do_s/doT_s/pT_s/dsT_s/lse_u/dlt_u` 各复制一份为 `_s0`/`_s1`；
   **删掉 `T.annotate_buffer_versions`，`num_stages` 参数改名 `n_versions`（默认 2，不再传入 `T.Pipelined`）**；
3. `for kq in T.Pipelined(...)` → `for i in T.serial(n_half)` + 半体 A/B + 循环后 drain（§3.3）；
4. `clear_accum=(kq == loop_st)` → `(kq−1 == loop_st)`；
5. `dq` 原子写回的 `m0` → `(kq−1)*BLOCK_M`；
6. **不改**：GM 签名/4-D 视图、`loop_st/loop_ed` 计算、掩码两支、块尾 `dk/dv` 写回、L0C/L0A/L0B 声明、grid。

---

## 14. 结论速览：对调度方六个问题的直接回答

### 14.1 六个问题

| # | 问题 | 回答 | 依据 |
|---|------|------|------|
| 1 | **A5 在 dense 的映射** | 见 §14.2：体 = `F(k)→GA(k)→EP(k)→GB(k−1)`，`GB(k−1)` 与 `EP(k)` **并列**；`q/qT/do/doT/lse/delta/pT/dsT` 各两份（静态 `_s0/_s1`）；首迭代 `if kqA > loop_st` 无 `GB`，末迭代 drain 按 `n_it` 奇偶；`clear_accum=(kq−1==loop_st)` | §1.6.1 A5、§3.3、§6.3 B |
| 2 | **与 varlen 的差异** | 见 §14.2 表：**可复用**的是「依赖边换缓冲身份」的机制、2-tile 静态展开、drain 奇偶、`clear_accum` 后移、UB 只需 `lse/delta` 两版本；**必须重算**的是 `loop_st/loop_ed`（静态 S、`SHIFT = S_kv−S_q`、无 `cu_seqlens` decode）、块尾写回（4-D clamp、无慢路）、`dq` 行偏移（`b` 内 `m0`）、**收益量级**（dense AIV 1.30 vs varlen 1.95–2.10 ⇒ dense **不受 AIV 限制**）、**MTE2 风险**（dense 静态 extent 118.5 vs varlen packed 80.5 GB/s，且 dense 的 L1 在 v3 已是 311 KB） | §14.2、§1.6.4 I |
| 3 | **容量与代码** | **L1 311,296 B（59.4%，余 212,992）**、**UB 130,048 B（51.2%，余 123,904）**、**L0C 221,184 B（84.4%，与 v3 逐字节相同）**、L0B 61,440 B（93.75%）、L0A 49,152 B（75%）。**代码 20,341 → 50,879 B（×2.50）**，icache 风险见风险 14（绝对值仍小于已上线的 varlen v5 的 61,913 B） | §4.5/§4.6、§1.6.4 I-1 |
| 4 | **收益投影** | **1.870 µs/迭代（探针）× 1.420e6 ÷ 28 = 94.8 ms 的 `_bwd_main`**；加 3.3 ms 其余 launch ⇒ **`T_op` 94.5–105 ms（central ≈98）**；**cube 38–40%（central 39%）**。**不承诺 ≥40%**（40% 需要 `T_op ≤95 ms`，在投影下沿）；**不承诺 90% 重叠**（实测已从 2.944 压到 1.870，即 63.5%，残差 0.42 µs 的分解未实测，风险 17） | §14.3、§1.6.4 I-4/I-6 |
| 5 | **验证** | L0/L1 组合 + **L0-14（B=2/S=8192/H=16，重复 3 次）** + `dk/dv` 与 v3 **逐位** + 容差（§8.2）+ 代码体积/P11 结构性判据/msprof 管道 | §8.2、§8.3、§14.4 |
| 6 | **风险** | MTE2 变第一忙管道（风险 19，dense 机制上弱于 varlen）、AIV 70%（风险 15）、icache ×2.50（风险 14）、`T.serial` 的填充延迟与 WAR（风险 16/17）、同步退化（风险 18，判据 P11） | §9.1 |

### 14.2 A5 在 dense 的逐迭代映射（问题 1、2）

```
每块（= 一个 (h, b, n0)）：
  loop_st = max(0, (n0 − SHIFT) // 64)      # SHIFT = S_kv − S_q（编译期常量）；非 causal 时 0
  loop_ed = ceil(S_q / 64)                  # 编译期常量
  n_it    = loop_ed − loop_st ;  n_half = (n_it + 1) // 2
  for i in T.serial(n_half):
      半体 A（版本 _s0）:  F(kqA) → GA(kqA) → EP(kqA ⇒ pT_s0/dsT_s0)
                           if kqA > loop_st:  GB(kqA−1, 读 _s1 组, clr=(kqA−1==loop_st))
      半体 B（版本 _s1）:  if kqB < loop_ed:  F(kqB) → GA(kqB) → EP(kqB ⇒ pT_s1/dsT_s1)
                                              GB(kqB−1, 读 _s0 组, clr=(kqB−1==loop_st))
  drain: GB(loop_ed−1, 读 _s0 组 if n_it 奇 else _s1 组, clr=(loop_ed−1==loop_st))
  块尾（不变）：dk/dv 半区落地 → cast(+scale) → 4-D 视图写回
```

| 维度 | varlen v5（已上线） | **dense v4（本版）** | 可否直接复用 |
|---|---|---|---|
| 依赖边切断机制 | `GB` 读另一版本的 `pT/dsT/qT/doT` | **同** | ✅ 直接复用（机制与语言无关） |
| 2-tile 静态展开 | `kqA = loop_st + 2i` | **同**（`loop_st` 的算法不同） | ✅ 结构复用；`loop_st` 重算（静态 S / `SHIFT`） |
| 首/末迭代 | `if kqA > loop_st` + drain 奇偶 | **同** | ✅ 直接复用 |
| `clear_accum` | `(kq−1 == loop_st)` | **同** | ✅ 直接复用 |
| `dq` 写回偏移 | `q_start + m0`（packed 全局行） | **`dq_acc[b, m0:m0+64, h, :]`**（4-D，`b` 内偏移） | ❌ 重写（dense 无 `q_start`） |
| 块尾 `dk/dv` 写回 | 快路 + 逐 lane 慢路（`kv_end` 守卫） | **只有 4-D clamp 快路**（clamp 到本 request 的 `S_kv`） | ❌ 简化（dense 无 `kv_end` 概念） |
| block 解码 | `cu_seqlens` 前缀和 + `req/n0/live` | **`h = bid//(B·NT)`、`tile = bid%(B·NT)`、`b = tile//NT`、`n0 = (tile%NT)·96`** | ❌ 重写（dense 是静态网格） |
| head slab / `h_off` | 有（3 次 launch 的 head 切分） | **无**（单次 launch） | ❌ 不适用（`grid` 无 slab） |
| AIV 单价 | 1.95–2.10 µs（**旋转后墙钟由 AIV 界定**） | **1.30 µs**（44.3% × 2.944） | ❌ **不同**：dense 旋转后由 **AIC** 界定（1.870 > 1.30） |
| MTE2 | packed 动态 extent，80.5 GB/s，旋转后 **share 0.86 成为第一忙管道** | **静态 extent，118.5 GB/s，share ≈38%** | ❌ **不同**：dense 的 MTE2 不会成为第一忙管道（风险 19 已按此降级） |
| 收益 | −33.9%（4k）/ −26.5%（16k） | **−35.9%（4k）/ −36.9%（8k）** | ❌ 量级需重测（已重测） |

**结论**：v5 的**调度机制**可 100% 复用；**任务映射、写回、单价归因**必须按 dense 重算——本版已用探针
在本算子的 4-D/B=2/静态 S 结构上重新取证（§1.6.4 I）。

### 14.3 收益投影（问题 4）

**单价**（探针 `--time`，3 次取最小；生产对账见 §1.6.4 I-4）：

| 量 | v3（生产） | v4（探针） | 依据 |
|---|---:|---:|---|
| 内层迭代 | **2.944 µs** | **1.870 µs**（B=2/S=8192/H=8，同 per-block 迭代数 64.5） | §1.6.4 I-4 |
| 探针标定误差 | — | **+0.7%**（v3 体探针 2.965 vs 生产 2.944） | 同上 |

**`_bwd_main` 与 `T_op`**：

| 量 | 公式 | v3 | **v4 投影** |
|---|---|---:|---|
| 迭代数 | `Σ_{t=0}^{85}(128−⌊1.5t⌋) × B·H` | 1,420,032 | 不变 |
| 单核迭代数 | `N_it / 28` | 50,715 | 不变 |
| `_bwd_main` | `50,715 × T_iter ÷ 1000` | 149.2 ms（实测 149.3 ✓） | **94.8 ms**（1.870）|
| 其余 5 个 launch | 实测 | 3.3 ms | 3.3 ms（与 `_bwd_main` 无关） |
| **`T_op`** | — | **154.4 ms** | **98.1 ms（区间 94.5–105）** |
| **cube pipe** | `38.0 ms ÷ T_op` | 24.6–25.1% | **38.8%（区间 36–40%）** |
| **AIV vec** | `1.30 µs × 50,715` | 44.3% | **≈70%** |
| 相对 v3 | — | — | **−36.5%（T_op）** |

**区间来源**：`T_iter ∈ [1.80, 2.00]`（探针 1.870 ± 3%，覆盖 icache 冷/热机、L2 驻留、尾波差异）。
**对照目标**：
- **`T_op ≤115 ms`（硬目标）**：投影 98.1，**余量 15%**，达成概率高。
- **`T_op ≤105 ms`（期望）**：投影上界 105 恰在线上，**central 达标**。
- **`cube ≥40%`**：需要 `T_op ≤95 ms` ⇒ `T_iter ≤1.80 µs`，落在投影**下沿**。**不承诺**；Stage 4 的
  三个 A/B（`V=3` / `st_u/dp_u` 第二版 / epilogue 阶梯）都有机会把 0.42 µs 的残差（风险 17）压掉一部分。
- **`90% 重叠`**：**不承诺**（见问题 4 的回答）。

**其余 workload（同比例，`×0.6307`，并以「短序列收益略小」保守化）**：

| workload | v3 实测 | v4 投影 | 说明 |
|---|---:|---:|---|
| `ds-v3-b2-8k` | 154.4 ms | **94.5–105 ms（central 98）** | 主目标；探针直接覆盖 |
| `kimi-k2-8k`（H=64） | 38.4 ms | **24–26 ms** | 同 S/B，仅 H 不同 ⇒ 同单价 |
| `ds-v3-4k` | 20.6 ms | **13.5–14.8 ms** | 同 S/B；块数 5,504 |
| `ds-v2-lite-3k` | 6.3 ms | **4.2–5.0 ms** | 块数 2,048（73 波），尾波与块级固定开销占比更高 ⇒ 收益略小 |
| smoke（512） | — | — | 不作为性能判据（§5.5） |

`ds-v3-b2-8k` 的其余 dtype/shape 变体按同单价线性缩放；**5 条 workload 的 per-iteration 在 v3 是 2.92–3.15，
与 shape 弱相关**（`perf_feedback.md`），故旋转后的单价也应在 1.85–2.00 的窄带内。

### 14.4 验证清单（问题 5）

| 层 | 用例 | 判据 |
|---|---|---|
| **L0（必跑）** | L0-1…L0-11（v3 的全部） | 不变（§8.2 容差） |
| **L0（v4 新增，必跑）** | **L0-12**（S=128，`n_it=2`，drain 偶）/ **L0-13**（S=96，`n_it=1`，drain 奇）/ **L0-14**（B=2/S=8192/H=8…16，causal，重复 **3 次**）/ **L0-15**（同 shape，非因果，重复 3 次） | 与 golden 在 §8.2 容差内；**且与 v3 冻结实现的 `dk/dv` 逐位相同**（`torch.equal`），`dq` 相对差 < 1e-3 |
| **L1** | `ds-v3-4k`、`ds-v3-b2-8k`、`kimi-k2-8k`、`ds-v2-lite-3k`，bf16 + fp16 | §8.2 容差；每 shape 重复 2–3 次（fp32 原子的 run-to-run spread ≤ 3e-8） |
| **结构判据（v4 专属）** | **P11**：生成代码里 `GB` 的三组 `L1→L0` 与三个 `asc_mmad` 之前**不得**有以 `PIPE_V` 为前驱的 `asc_sync_wait`（AIC 内部的 `wait(PIPE_M, PIPE_MTE1/FIX)` 是正常的） | 出现即**回滚 v3 体** |
| **自检反例** | **P9**（删 drain 必须失败）/ **P10**（`GB` 读当前版本必须失败） | 两个反例都必须失败，否则说明版本没生效 |
| **代码体积** | `asc_mmad` 应为 **16**、`__simt_vf__` 应为 **5**；生成源 ≈50.9 KB | 偏离 ⇒ 半体/分支数量与设计不符 |
| **管道（msprof）** | `aic_cube_ratio` / `aiv_vec_ratio` / `aic_mte2_active_bw` / `aic_mte1_active_bw` / **AIC 空闲窗口（P12）** | cube ≥40%（目标）/ AIV <80% / mte2 >85 GB/s / mte1 <0.75 µs / AIC 空闲 <10% |
| **性能** | `ds-v3-b2-8k` 的 `T_op`（6 launch 之和） | **≤115 ms**（硬）；≤105 ms（期望）；回退判据：>135 ms（= v3 × 0.875）视为 A5 未生效，按风险 18 排查 |

### 14.5 风险一句话汇总（问题 6）

7 项已量化、全部带判据与回退（§9.1 风险 14–19 + 风险 1–13 不变）：
**icache ×2.50**（绝对值 < 已上线的 varlen v5）/ **AIV 70%**（< 80% 警戒）/ **`st_u/dp_u` WAR**（P12，
回退只花 24,576 B UB）/ **残差 0.42 µs 未分解**（Stage 4 第一优先 msprof）/ **同步退化**（P11，回滚 v3）/
**MTE2 忙而慢**（dense 静态 extent 118.5 GB/s，机制上弱于 varlen 的 80.5；阈值 85 GB/s）/
**首末迭代不重叠**（已计入投影，≈1.7%）。
