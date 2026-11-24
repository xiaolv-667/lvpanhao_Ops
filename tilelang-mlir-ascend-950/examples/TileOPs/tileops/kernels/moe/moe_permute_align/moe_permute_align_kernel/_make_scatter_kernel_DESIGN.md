# _make_scatter_kernel 算子设计文档（MoE 对齐第二段：路由散射）

> **任务平台**：`ascend-950pr-9579`（Ascend950PR_9579，dav-3510）；后端 `ascend`（`import tilelang.ascend.language as T`、`tilelang.compile(..., target="ascend", out_idx=[])`）；工具链 tilelang 0.1.15（`/workspace/venv-ascend950/lib/python3.12/site-packages/tilelang`）、CANN 9.2.1（编译 SDK + runtime，`/usr/local/Ascend/cann-9.2.1`），容量预算取自 `statectl platform --task-dir examples/moe_permute_align/_make_scatter_kernel` 的 `design_budget`（UB 253952 B / L1 524288 B / L0C 262144 B）。
>
> **编程模式**：**SIMT**（`T.SimtVF(threads=128)` + `T.Parallel`）。平台 `programming_modes.migration_default` 为 `simd`，偏离理由与证据见 §2.2。
>
> **迁移任务**：源算子 `_make_scatter_kernel`（GPU TileLang / CUDA target），源文件 `/workspace/TileOPs/src/tileops/_kernels/moe/permute_align.py`（472 行，已全文阅读）。§0 为源算子三问解读 → 算法调研（§1.6.0）→ 硬件耦合性判定 → NPU 重设计；§1–§11 全部基于 §0.6 重设计后的 NPU 算法。
>
> **设计期未做设备验证**：设计期只执行**无卡编译/静态核对**（`ASCEND_RT_VISIBLE_DEVICES=""`，不 launch 任何 kernel），证据全部来自：① 本机选中工具链的 language 模块源码与降级后设备代码（`history_version/design_probe_primitives.py`、`history_version/design_probe_skeleton.py` 两个探针，均 ≤10 min）；② CANN 9.2.1 交付的 SIMT 设备 API 头文件与其自带算子的调用点；③ 平台 `examples/` 同类实现。凡未经设备执行的条目显式标注「待 Stage 3 真机确认」。
>
> **roofline 口径**：manifest 声明 `flops: "0"`（纯整型索引重排，无算术），故全文按**访存 + 原子/发射**论证，不使用 MFU。

---

## 0. 源算子解读与迁移分析（迁移任务必填）

> 组织方式：先彻底读懂源算子（0.1–0.4），再做算法调研（结论落 §1.6.0，是 M1 判定的输入），再判定硬件耦合性（0.5），最后给出 NPU 算法重设计（0.6）。本章结论驱动 §1–§11。

### 0.1 源算子语义（做什么）

**数学语义（稳定的计数排序散射，`moe_align_block_size` 第二段）**：设 `numel = T·K` 为路由总数，`flat = topk_ids.flatten()`（`int32[numel]`，元素 ∈ `[0, num_experts)`），`cumsum[e]`（`int32[num_experts+1]`）为**每个 expert 已按 `block_size` 补齐的区段起点**（由 align 段产出；`cumsum[e+1]-cumsum[e] = ⌈count_e/block_size⌉·block_size`）。记 `r_i = #{ j < i : flat[j] == flat[i] }`（路由 `i` 在**全部路由的索引升序**中的 expert 内序号），则

$$\text{sorted\_token\_ids}\big[\,\text{cumsum}[\text{flat}[i]] + r_i\,\big] = i,\qquad i \in [0,\,numel)$$

未被任何路由占用的槽位保持 align 段写入的哨兵值 `numel`。`sorted_token_ids` 长度 `P = numel + (num_experts+1)·(block_size-1)`。

- **规约语义**：无算术归约、无浮点累加 ⇒ 不存在累加顺序精度问题。**唯一的顺序语义**是「expert 内部路由的排列顺序」：本段源码用 `atomic_add` 的到达顺序写入，因此**expert 内部的顺序在源码层面即未定义**；GPU 侧参考实现（`tileops/workloads/moe.py::ref_permute_align` 的 `slot[eid] += 1` 顺序推进）给出的是**索引升序**这一种合法排布，而 op 契约明确只要求「expert 内集合相同」——见 `examples/TileOPs/tests/ops/test_moe_permute_align.py` 的 `_assert_route_equivalent`（对每个 expert 区段内按路由索引归并后再比对）。**该契约是本设计可以采用原子到达顺序的依据（§1.6.0 R2 / §9 R3）。**
- **I/O 契约**：`flat` 只读；`sorted_token_ids` 读改写（只覆写每个 expert 区段的前 `count_e` 个槽，其余槽保持哨兵）；`cumsum` **原地读写**（作为跨 kernel 的 bump 游标，调用方每次调用新建）。
- **dtype 语义**：全 int32 索引数组，无中间精度、无 cast。`atomic_add` 在 int32 上做，`P ≤ 102131`（全部 workload，见 §1.6.1 O3）远小于 `2^31`，无溢出。
- **边界语义**：`numel = 0` 时 factory 直接 `raise`（host 侧，调用方不发起本段）；`id ∉ [0, num_experts)` 属调用方违约（manifest `requires: in_range(0, num_experts)`），源码只 guard `eid >= 0`；`numel` 不被 block 宽度整除时用**夹取索引 + `-1` 哨兵**处理尾 lane（`flat[T.min(idx, numel-1)]` + `if_then_else(idx < numel, …, -1)`）。
- **与 align 段的衔接（本段的前置契约）**：align 段（`_make_align_kernel`）完成 ① 直方图、② 补齐前缀和写入 `cumsum`、③ `num_tokens_post_pad`、`expert_ids`、④ 用 `numel` 填满 `sorted_token_ids`；本段只消费 `flat` + `cumsum`，并把每个 expert 的游标从 `cumsum[e]` 推进到 `cumsum[e]+count_e`。**两段必须分开**：`cumsum` 依赖**全部**路由的全局直方图与前缀和，任何 slot 写入都必须等这个全局结果落定；GPU 上用两次 kernel launch 的**kernel 边界**充当全局屏障，NPU 上 `ascend.batch_schedule` 的跨 block barrier 未认证（`.agents/platforms/toolchains/ascend/references/programming.md` §同步与缓冲版本）、且无认证的跨核 flag 模板 ⇒ 同一条 kernel 边界继续承担该屏障，两段结构**保留**（§0.5 #1）。
- **host 侧语义**：`flatten().contiguous()` 视图、`max_padded`/`max_num_blocks` 的 host 计算、`cumsum = torch.empty(E+1)` 的分配、align→scatter 的**同流顺序调用**（`moe_permute_align.py::forward`），均属实现手段而非语义；NPU side 沿用冻结 wrapper，本设计不改。

**语义保持基线**：§8.1 golden 以本节为唯一依据（移植 `ref_permute_align` 的稳定顺序语义），**不得**参考 §0.6 的 NPU 算法——保证验证独立性。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `flat` | `(numel,)`，`numel = T·K` 为**编译期常量** | `int32` | 路由的 expert id，∈ `[0, num_experts)`；动态轴 `T` 只影响 `numel`，进 kernel 特化参数 |
| `sorted_token_ids` | `(P,)`，`P = numel + (num_experts+1)·(block_size-1)` | `int32` | **读改写**：align 段已填 `numel` 哨兵，本段只覆写每个 expert 区段的前 `count_e` 槽 |
| `cumsum` | `(num_experts+1,)` | `int32` | **原地读写**：`[0,num_experts)` 为各 expert 的 bump 游标（进入时 = 补齐起点，退出时 = 起点 + `count_e`）；末元 `cumsum[num_experts] = num_tokens_post_pad`（本段不得改写） |
| 返回值 | — | — | 无（`out_idx=[]`，副作用全在 `sorted_token_ids` / `cumsum` 上） |

补充：`num_experts`、`block_size`、`numel` 均为编译期常量（factory 参数），**kernel 内无动态轴**；本次 12 条 workload 的 `(numel, num_experts, block_size, P)` 见 §4.1。

### 0.3 实现算法解读（怎么算）

**计算步骤分解**（源码每一个计算语句都归入某一步，无遗漏）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式的部分 |
|------|------|------|------|-------------------|
| 1 | host：`if numel < 1: raise`；`threads = 512`；`scatter_blocks = ⌈numel/threads⌉`；`expert_rounds = ⌈E/threads⌉` | `numel/E/bs` | 网格与循环次数 | 任务划分（步骤 3 的 `bid/tx`） |
| 2 | 块内清零：`for i in serial(expert_rounds): if expert< E: s_count[expert] = 0` | — | 块内直方图缓冲清零 | 前置状态初始化 |
| 3 | 路由 → 线程映射与尾 lane：`idx = bid·threads + tx`；`eid = if_then_else(idx < numel, flat[min(idx,numel-1)], -1)` | `flat` | 每 lane 一条路由 id | 路由集合 `[0,numel)` 全覆盖 |
| 4 | 块内局部序号：`if eid >= 0: rank = atomic_add(s_count[eid], 1, return_prev=True)` | `s_count`（SMEM） | 该路由在**本块内**同 expert 的序号 | `r_i` 的块内分量 |
| 5 | 逐 expert 全局预留：`for ex: if s_count[ex] > 0: s_base[ex] = atomic_add(cumsum[ex], s_count[ex], return_prev=True)` | `s_count`, `cumsum` | 本块在该 expert 区段的起始槽 | `cumsum[e] + (# 更早块的同 expert 路由数)` |
| 6 | 散射写：`if eid >= 0: sorted_token_ids[s_base[eid] + rank] = idx` | 步骤 4/5 结果 | `sorted_token_ids` | 公式 `cumsum[e_i] + r_i ← i` |
| 7 | host：`@target_cache(maxsize=32)` 缓存 factory、`@tilelang.jit(out_idx=[], compile_flags=["-O3"])` | — | 编译产物 | 构建/缓存策略 |

**数据流图（源 GPU 视角）**

```
GM[flat]        --lane 读（idx 定位，clamp+哨兵）--> register(eid)
GM[flat]  ─┐
           ├─(每 lane 一条路由)
SMEM[s_count]  --atomic_add(+1, prev)--> register(rank)          # 块内直方图/序号
SMEM[s_base]   <--atomic_add(cumsum[ex], s_count[ex], prev)-- GM[cumsum]   # 每块每 expert 一次批量预留
                                   │
register(rank, s_base[eid]) ───────┴─► GM[sorted_token_ids[s_base[eid]+rank]] = idx
```

- 源实现**没有** tile 搬入/搬出、没有 `T.copy`、没有 matrix 单元、没有 warp shuffle 归约（源码不含 `tvm_warp_shuffle`；该 API 只出现在同文件的 `_make_align_kernel` 中）。
- 写地址**数据相关**（扫描写）：lane 之间的目标槽互不相同（由 bump 游标保证），但地址在 GM 上不连续。

**循环与并行结构**：1-D block 网格 `scatter_blocks = ⌈numel/512⌉`，块内 512 线程 = 1 路由/线程（`T.Parallel` 无，直接 `tx`），另有 `expert_rounds = ⌈E/512⌉`（E ≤ 512 时为 1 轮）的 expert 清零/预留循环。**每线程只持有一条路由**，故 `eid/rank` 常驻寄存器。

**host 侧逻辑**：`numel < 1` 抛 `ValueError`；`max_padded`/`scatter_blocks` 计算；`cumsum` 由 `forward` 分配（`torch.empty(E+1)`）并在 align 后传入；align → scatter 同流顺序调用；`target_cache(maxsize=32)` 按 `(numel, E, bs)` 缓存 factory。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | 两级计数（SMEM 直方图 `s_count` + 每 expert 批量预留） | 把「每路由一次全局原子」降为「每块每 expert 一次全局原子」，避开 GM 原子高延迟与热点争用 | 块内先 `atomic_add` 到 SMEM 计数，再用 `s_count[ex]` 一次性 `atomic_add` 到 `cumsum[ex]` | SMEM 低延迟原子（~20 cyc）远低于 GM 原子（~200+ cyc）；块内线程共享 SMEM | **硬件强相关**（SMEM 原子代价模型） |
| 2 | 一 lane 一路由（`idx = bid·threads+tx`），`eid/rank` 常驻寄存器 | 零额外缓冲、零二次访存；散射只需 1 次 load + 1 次 store | 线程与路由一一绑定，`rank` 由原子返回值直接得到 | CUDA threadIdx / 大寄存器堆 | **硬件相关**（线程模型） |
| 3 | 尾 lane 夹取 + `-1` 哨兵（`flat[min(idx,numel-1)]` + `idx<numel`） | 让所有 lane 走同一条无分支载入路径，避免越界读 | 一次 clamp 索引 + 一次表达式级 select，`eid<0` 作为跳过谓词 | 无（表达式级） | **可移植** |
| 4 | `s_count` / `s_base` 分离缓冲 | 规避 TileLang 把 sync 插进 expert guard 内导致的死锁（源码注释明确记录） | 预留结果写独立缓冲 `s_base`，不改写 `s_count` | 编译器 sync 插入行为（框架相关） | **框架相关**（本平台需重新核对） |
| 5 | `expert_rounds = ⌈E/threads⌉` 多轮 expert 循环 | 支持 `E > threads`（最大 1100 > 512） | 每轮处理 `threads` 个 expert | CUDA 线程数上界 | 硬件相关（线程数） |
| 6 | 1-D block 网格 `⌈numel/threads⌉` | 网格维数与数据量线性、无二维折叠开销 | `bid` 直接给路由块号 | 1-D grid 调度 | **可移植** |
| 7 | `-O3` 编译 + factory 缓存 | 降低构建/启动开销 | `compile_flags`、`target_cache` | 框架相关 | **可移植** |
| 8 | 与 align 段的两段式（kernel 边界作全局屏障） | 让「全局直方图+前缀」与「散射」各自独立并行 | 两次 launch 同流 | kernel 边界即全局同步 | **可移植**（两侧都有 kernel 边界） |

> 识别不出 ≠ 不存在。本文件 `_make_scatter_kernel` 共 60 行、无 `T.copy`/无 `T.gemm`/无 warp shuffle/swizzle/无流水/multi-buffer/无 online 统计量——上表覆盖其全部显式优化意图。**特别注意 #1（两级计数）是唯一的「算法层性能优化」，迁移时必须显式承接或给出量化舍弃理由（§0.5 #4、§1.6.1 O1）。**

### 0.5 硬件耦合性分析与 NPU 适配决策

| # | 条目 | 层级（语义/算法/优化） | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|---|------|----------------------|-----------|-----------------|------|---------------------|
| 1 | 两段式（align → scatter，kernel 边界作全局屏障） | 语义/算法 | 无（kernel 边界两侧都有） | 有 | **保留** | 冻结 wrapper 已按此接线（`moe_permute_align.py::forward`）；跨 block barrier 在本工具链未认证 ⇒ 不用单 kernel 跨核同步 |
| 2 | 数学映射 `sorted[cumsum[e_i]+r_i] = i`、哨兵保持、expert 内顺序未定义 | 语义 | 无 | — | **保留（无条件）** | §0.1 为唯一基线 |
| 3 | 一 lane 一路由（`bid·threads+tx`）+ 尾部夹取哨兵 | 算法/优化 | CUDA threadIdx | 有：`T.SimtVF(threads=128)` + `T.Parallel(EPT, threads)` | **等价替换** | 探针 K6 降级代码即 `block_idx·tile + i·128 + threadIdx.x`；线程数 512→128 属调度参数（§0.6 R1） |
| 4 | 两级计数（SMEM 直方图 + 每 expert 批量预留） | 优化 | SMEM 原子代价 << GM 原子，且块内线程共享 SMEM | **部分**：UB 原子存在（探针 K3 编译通过）但「UB 原子 vs GM 原子」代价在本平台无实测数据；且本平台 block 粒度下两级结构**不减少全局原子数** | **等价替换**（意图承接见 §0.6 R2 的量化） | 单级 bump：`slot = atomic_add(cumsum[eid], 1, ret_prev)`，每路由 1 次 GM 原子；两级形式保留为 §1.6.3 备选 B（若 Stage 4 实测争用为主因） |
| 5 | `s_count`/`s_base` 分离（规避编译器把 sync 插进 guard 死锁） | 优化（框架 workaround） | TileLang GPU 后端 sync 插入行为 | 不适用（本设计无 expert guard 循环、无块内 sync） | **舍弃**（不再需要） | 单级 bump 无块内直方图/预留两趟，不含 guard 内 sync 结构（§0.6 R3） |
| 6 | `expert_rounds = ⌈E/threads⌉` expert 多轮循环 | 优化 | CUDA 线程数（E 最大 1100 > 512） | 不需要 | **舍弃** | 单级 bump 无按 expert 的循环 ⇒ `E = 1100` 也不增加任何代价（§1.6.0 R3 对比） |
| 7 | 逐 lane 数据相关 GM 散射写 `sorted[slot] = idx` | 算法 | 无（普通 ND 张量） | 有：`T.Parallel` 内逐 lane 索引写 | **保留（承载方式换 SimtVF）** | 探针 K1/K6：编译通过，降级为 `out[pos[...]] = src[...]` 并由编译器自动插入值域 guard（§3.4） |
| 8 | 每路由一次 GM `atomic_add`（`cumsum` 作游标，取旧值） | 算法 | CUDA `atomicAdd` 返回旧值 | **有**：CANN SIMT `asc_atomic_add(__gm__ int32_t*, int32_t) -> int32_t`（arch 3510 专用） | **保留** | `T.atomic_add(buf[i], 1, return_prev=True)`；探针 K2/K6 降级为 `condval = asc_atomic_add(&cumsum[eid], 1)`；头文件 `/usr/local/Ascend/cann-9.2.1/asc/include/simt_api/device_atomic_functions.h:31`（`__NPU_ARCH__ == 3510`） |
| 9 | 1-D block 网格 + `threads` 参数 | 调度 | CUDA grid/block 维度 | 部分：`T.Kernel(N)` 只有 1-D 核网格、**无 `threads`**；线程域靠 `T.SimtVF` | **等价替换** | 探针 K6：`T.Kernel(nblocks) as bid, T.SimtVF(threads=128)`（§0.6 R1） |
| 10 | `-O3` / factory 缓存 / `out_idx=[]` | 构建 | 无 | 有 | **保留** | `tilelang.compile(..., target="ascend", out_idx=[])`；factory 级 `functools.lru_cache` 由 wrapper 承担 |
| 11 | `numel < 1` 的 host 抛错与 dispatch | host | 无 | 有 | **保留** | 冻结 wrapper 已固化（`self._scatter_fn = ... if numel else None`） |
| 12 | 浮点/矩阵/SMEM tiling/流水/warp shuffle | — | — | 源算子中不存在 | **不适用** | 源码全文核对：无 `T.gemm`、无 `T.copy`、无 `T.alloc_shared` 之外的片上缓冲、无 shuffle |

> 判定依据来源：语义项引 §0.1；API 项引本任务两个探针的降级设备代码（§11.1）与 CANN 9.2.1 `simt_api` 头文件；框架项引 `programming.md`（`T.Kernel` 无 `threads`、`allow_autoschedule` 默认开启）。**未发现**需要「重新设计算法」的条目（存疑项按 §0.5 #4 处理：以「等价替换 + 备选 + 实测裁决」承载，而非纸面重设计）。

### 0.6 NPU 算法重设计

对 §0.5 判定为「等价替换」的条目给出 NPU 落地形态与语义保持论证（无「重新设计」项——三条处置均为等价替换或舍弃，且均有探针/头文件佐证）。

**R1（← 条目 3/9）：线程与网格映射 —— CUDA block/thread → 1-D 核网格 + SimtVF 线程域 + 每线程 EPT 条路由**

- **源方案**：`T.Kernel(⌈numel/512⌉, threads=512)`，一 lane 一路由。
- **NPU 新算法**：`T.Kernel(nblocks) as bid, T.SimtVF(threads=128)`；块内 `for i in T.serial(EPT)` 展开为「每线程 EPT 条路由」：
  `idx = (bid·EPT + i)·128 + tx`。`EPT = max(1, ⌈⌈numel/56⌉/128⌉)`、`nblocks = ⌈numel/(128·EPT)⌉`（§5.2/§5.5；全部为编译期常量）。
- **语义保持论证**：映射 `(bid, i, tx) ↦ idx` 是 `[0, nblocks·128·EPT)` 到路由空间的**行主序双射**，`nblocks·128·EPT ≥ numel` ⇒ 每条路由恰被一个 lane 处理一次（多出的尾 lane 由 §0.4 #3 的哨兵跳过）。每条路由的算式仍是「取 eid → 原子取槽 → 写 `idx`」，与线程数、EPT、块数**无关**（逐路由映射的线程无关性）。`cumsum` 游标的单调推进与写入槽的互异性只依赖「每路由一次原子」，与分组方式无关。
- **边界语义**：`numel` 不被 `128·EPT` 整除时，尾 lane 的 `idx ≥ numel`，按 #3 取 `eid = -1` 跳过（不读 `flat` 越界、不原子、不写）。

**R2（← 条目 4）：两级计数 → 单级 bump 游标（源唯一的算法层优化，显式承接）**

- **源方案**：块内 SMEM 直方图 `s_count` → 每 expert 批量预留 `s_base[e] = atomic_add(cumsum[e], s_count[e])` → 二趟写 `sorted[s_base[eid]+rank]`。
- **NPU 新算法**：单级——`slot = atomic_add(cumsum[eid], 1, return_prev=True)`；`sorted[slot] = idx`。即把「块内序号 + 批量预留」合并成「每路由一次游标自增」。
- **等价性论证**（数学 + 机器验证）：对任一 expert `e`，两级形式给该块分配 `[base, base+s_count[e])` 内**互异**的连续槽；单级形式给每条路由分配 `prev` 值，因游标单调自增 1，同一 expert 的 `count_e` 次取槽**恰好覆盖** `[cumsum[e], cumsum[e]+count_e)` 且两两互异、无越界。两种形式产出的**每 expert 槽集合完全相同**，仅 expert 内的**排列顺序**不同；而该顺序按 §0.1 的契约本就未定义（`_assert_route_equivalent` 对每个 expert 区段归一化后再比对）⇒ **契约级等价**。`verify_equiv.py` 的 O1 项在 8 个 shape × 3 种合法到达顺序下逐项核对「每 expert 多重集相等 / 槽互异 / 槽区间 / 哨兵保持 / 游标终值 = 起点+count_e」，结果 `EQUIV_PASS`（§1.6.1）。
- **承接源优化意图（量化）**：两级形式的目的（#1「把每路由一次全局原子降为每块每 expert 一次」）在本平台的块粒度下**并不能减少全局原子数**：单块路由数 `tile = 128·EPT ≤ 640`，而 `num_experts ∈ {128,256,384}`（workload）⇒ 每块非空 expert 数上界为 `min(tile, E) = tile`，故全局原子数上界 `nblocks·min(tile,E) ≈ numel`，与单级相同；单级反而少 2 次 `syncthreads`、少 2 个 UB 缓冲（`2·E·4 B` = 1–3 KB）、少一趟 rank 缓冲读写。若未来出现 `tile ≫ E` 的配置（例如把 `threads` 提到 1024），两级形式的全局原子数按 `min(tile,E)/tile` 缩小 ⇒ 保留为 **§1.6.3 备选 B**，由 Stage 4 实测裁决（阈值见 §1.6.3 T2）。
- **边界语义**：`count_e = 0` 的 expert 区段长度为 0，无游标移动；`cumsum[e+1]` 在本设计中**不被读**（单级形式只需要 `cumsum[e]`），因此 `count_e` 未知也不影响正确性——这正是 bump 形式相对「需先求 count 再定 base」的两趟形式的鲁棒点。

**R3（← 条目 5/6）：删除 `s_count`/`s_base` 分离与 expert 多轮循环**

- **源方案**：`s_count`/`s_base` 两个 SMEM 缓冲 + `expert_rounds = ⌈E/512⌉` 的 expert 清零/预留循环（源码注释记录了「就地更新 `s_count` 会让 TileLang 把 sync 插进 expert guard 内、超过一轮即死锁」）。
- **NPU 新算法**：不分配任何 UB 缓冲、不含任何 expert 循环、不含块内 `sync_threads`。`E = 1100` 与 `E = 128` 的 kernel 结构完全相同。
- **语义保持论证**：该结构只服务于两级计数的中间状态（`s_count`/`s_base`），单级形式不需要这些状态 ⇒ 删除不改变任何路由的落槽集合（同上 R2 的论证）。「规避编译器 sync 插入死锁」的需求随之消失：新算法块内无共享缓冲的读写交替，不存在可被插入 sync 的 guard 内依赖。
- **验证方式**：§8.1 golden（按 §0.1 语义的稳定顺序参考）+ 契约级归一化比对；O1 机器验证已覆盖「任意到达顺序下槽集合正确、无重复、无越界」。

> 结论：§0.5 的 12 条处置（保留 6 / 等价替换 4 / 舍弃 2 / 重新设计 0）全部落地；源算子 8 项优化意图（§0.4）逐项有承接或量化说明（#1→R2、#2→R1、#3 保留、#4→R3 删除并说明、#5→R3 删除并说明、#6 保留、#7 保留、#8 保留）。**无静默丢弃项。**

### 0.7 标杆实现

| 用途 | 路径 | 说明 |
|------|------|------|
| 源算子（GPU TileLang） | `/workspace/TileOPs/src/tileops/_kernels/moe/permute_align.py`（`_make_scatter_kernel`，行 168–229） | §0.1–§0.4 的解读对象；factory 签名 `(numel, num_experts, block_size)`，内部 `threads=512` |
| 源文件抽取副本 | `examples/TileOPs/tileops/kernels/moe/moe_permute_align/_moe_permute_align_fwd_kernels.py` | Stage 0 抽取的 GPU 工厂原文（迁移前状态；本设计将其中的 scatter 工厂改写为 NPU 实现） |
| golden 基准（独立参考） | `examples/TileOPs/tileops/workloads/moe.py` 的 `ref_permute_align`（行 31–80）与 `MoEPermuteAlignWorkload.ref_program`（行 128+） | 稳定顺序（索引升序）参考：`slot[eid] += 1` 逐条推进；§8.1 直接移植 |
| 契约级比对规则 | `examples/TileOPs/tests/ops/test_moe_permute_align.py` 的 `_assert_route_equivalent`（行 71–96） | expert 内顺序归一化后比对，是全设计「原子到达顺序合法」的依据 |
| 规格来源 | `examples/TileOPs/tileops/manifest/moe.yaml`（`MoEPermuteAlignFwdOp`，12 条 workload，`roofline.flops = "0"`） | 输入规格与 workload 的机器可读来源 |
| 冻结接口（Stage 0） | `examples/TileOPs/tileops/kernels/moe/moe_permute_align/moe_permute_align.py`（`forward`，行 190–215） | `align(flat, sorted, expert_ids, n_post_pad, cumsum)` 后 `scatter(flat, sorted, cumsum)`；factory 无 launch 参数（K9） |
| 同平台同类实现（唯一 ascend 用例） | `examples/paged_cache_write/_paged_cache_write_kernel/DESIGN.md` | 同为「数据相关地址的 GM 直写」类，`T.Kernel` 1-D + `T.SimtVF` + 逐 lane GM 写的证据与风险组织方式 |

> golden 以 §0.1 语义为唯一依据（移植源仓 `ref_permute_align`），**不复刻** §0.6 的 NPU 算法（不使用原子、不使用 1-D 核网格、不使用 `SimtVF`），保证验证独立性。

---

## 1. 概述

### 1.1 算子名称

`_make_scatter_kernel`（`MoEPermuteAlignFwdOp` 的**第二段** kernel factory；NPU 实现文件为本算子目录下的 kernel 文件（`_make_scatter_kernel.py`，Stage 3 产出））。

### 1.2 功能描述

把每条路由（token, k）写入它所属 expert 区段的「下一个空槽」：`slot = cumsum[e]++`，`sorted_token_ids[slot] = 路由索引`。本段只覆写每个 expert 区段的前 `count_e` 个槽，其余槽保持 align 段写入的哨兵 `numel`；`cumsum` 被原地推进为「起点 + count_e」。expert **内部**的排列顺序按 op 契约未定义（§0.1）。

### 1.3 数学公式

记 `flat[i]` 为路由 `i` 的 expert id，`S_e = cumsum[e]`（该 expert 补齐后的区段起点），`r_i = #{ j < i : flat[j] = flat[i] }`（索引升序序号，参考实现给出的合法排布），`1[·]` 为指示函数：

$$
\text{sorted\_token\_ids}\big[\,S_{\text{flat}[i]} + r_i\,\big] = i,\quad i \in [0, numel);\qquad
\text{sorted\_token\_ids}[p] = numel\ \ \text{for}\ \ p \notin \bigcup_e [S_e,\, S_e + \text{count}_e)
$$

**本设计实际计算的形式**（§1.6.1 O1，契约级等价，顺序自由）：

$$
\text{slot}_i = \mathrm{prev}\!\left(\text{cumsum}[\text{flat}[i]] \mathrel{+}= 1\right),\qquad
\text{sorted\_token\_ids}[\text{slot}_i] = i
$$

即 `slot_i` 是「对 `cumsum[e]` 做 `+1` 原子自增所返回的旧值」；同一 expert 的 `count_e` 次取槽恰好覆盖 `[S_e, S_e+count_e)` 且两两互异（§1.6.1 机器验证）。

### 1.4 算法描述

本节描述**迁移决策后的 NPU 算法**（§0.5/§0.6 落地形态），与源算法的差异逐条标注来源：

1. **任务映射（差异 ← §0.6 R1）**：1-D 核网格 `nblocks = ⌈numel / (128·EPT)⌉`，块内 `T.SimtVF(threads=128)`，每线程 `EPT` 条路由，`idx = (bid·EPT + i)·128 + tx`。
2. **单级 bump 游标（差异 ← §0.6 R2）**：`slot = T.atomic_add(cumsum[eid], 1, return_prev=True)`；不再有块内直方图、批量预留与二趟写。
3. **逐 lane 数据相关散射写（保留 ← §0.5 #7）**：`sorted_token_ids[slot] = idx` 直接写 GM，不经 UB 中转。
4. **尾 lane 夹取 + 哨兵（保留 + 表达式化 ← §0.4 #3 / §1.6.1 O2）**：`eid = if_then_else(idx < numel, flat[min(idx, numel-1)], -1)`；`eid >= 0` 才原子与写。
5. **expert 上界 guard（新增安全强化 ← §1.6.1 O4）**：`0 <= eid < num_experts` 才原子与写，避免契约外 id 改写 `cumsum[num_experts]`（`num_tokens_post_pad`）。
6. **零片上缓冲、零块内同步（差异 ← §0.6 R3）**：UB 占用 0 B；无 `sync_threads`、无 UB 原子；`E` 与 `block_size` 不进入任何循环（`E = 1100` 与 `E = 128` 结构相同）。

**与源算法的一致性核对**：落槽语义 `sorted[cumsum[e_i] + r_i] = i` 在契约级（每 expert 槽集合）完全一致；被替换的只有「块内序号 + 批量预留」的表达方式（单级自增）、线程/网格映射（1-D + SimtVF + EPT）与尾块/上界 guard 的写法。无算术、无归约 ⇒ 不存在累加顺序或精度差异。

### 1.5 数据流图

```
GM[cumsum(E+1)]  ──┐（每路由一次原子 RMW：读旧值→+1→写回；工作集 4·(E+1) B，L2 常驻）
GM[flat(numel)]  ──┼─►  lane: idx=(bid·EPT+i)·128+tx → eid=select(clamp) → 原子取 slot
GM[sorted(P)]    ◄─┘                                   └► 逐 lane 散射写 sorted[slot] = idx
```

- 全链路 **GM → 标量寄存器 → GM**：无 UB / L1 / L0 分配，无 `T.copy`、无 DMA、无 layout 变换。
- 无块内同步点；块间唯一交互是 `cumsum` 上的原子自增（硬件原子保证互异，无需 barrier）。
- 跨 kernel 依赖：`cumsum` 的初值由 align 段写入，靠**同流顺序 launch 的 kernel 边界**保证可见性（§0.1 / §7）。

### 1.6 算法调研与优化分析 ⭐

> **设计第一优先级**：先调研（1.6.0 四问：等价化简公式 / 在线算法 / 复杂度 / 硬件亲和；输入公式与源算法只是基线候选之一），再在数学等价前提下优化公式（1.6.1），再把保留的计算交给向量单元（1.6.2），最后回答「在哪个轴上向量化」（1.6.3）。本节结论是 §3.1 公式拆解与 §6 循环结构的输入；1.6.1–1.6.3 的分析对象 = §1.6.0 选定、经 §0.6 重设计落地的 NPU 算法。

#### 1.6.0 算法调研（Algorithm Research）⭐

> **调研对象**：同一数学语义（「按 expert 分桶的索引重排散射」= 稳定计数排序的散射段）的算法族。任务类别判定（§0.3/§0.6）：**多步 + 索引/散射类**（非逐元素、非规约）⇒ 按 `algorithm-research.md` §2 执行**完整调研**（R1 候选表 + R3 四口径对比表 + R4 逐候选评估）。
> **信息源核对记录**：① 平台 `knowledge.algorithm_candidates`（`algorithm-candidates.md`）= 仅覆盖 elementwise/reduction/softmax/norm/GEMM/attention，**本算子族无命中条目**；② `kb_search.py "scatter histogram atomic counting sort moe align scatter"` → `no entries matched`（与 conductor 预注入的「kb_search 无命中」一致）；③ pattern-library 目录在本仓不存在（`.agents/skills/_shared/` 仅有 `standards/`），`knowledge/bottlenecks.md` 明示「当前没有 950 真机验证的瓶颈模式或收益阈值」⇒ **代价常数一律标「未实测假设」**；④ 本仓 `examples/` 无同类索引散射算子（唯一 ascend 先例 `paged_cache_write` 是「定长行散射直写」，可借鉴承载方式但无计数排序语义）；⑤ 结构判据（见各问）；⑥ 互联网检索：本族属成熟经典算法（计数排序/桶排序/radix sort/prefix-sum 扫描），模型知识已足以枚举候选，未额外联网（网络工具非必需，按 `algorithm-research.md` §6 第 6 层降级规则记录）。

**R1 等价化简公式候选**（基线在表内；正式等价论证与收益量化在 §1.6.1 完成，此处只做初判）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 纳入 R3 |
|---|------|------------|-----------|---------|---------|
| 0 | **基线（源算法直译）**：两级计数 bump | `rank=atomic_add(s_count[e])`；`base=atomic_add(cumsum[e], s_count[e])`；`sorted[base+rank]=i` | — | — | ✅ |
| 1 | **单级 bump 游标（选定）** | `slot = prev(atomic_add(cumsum[eid],1))`；`sorted[slot]=i` | 契约级等价（每 expert 槽集合相同；expert 内顺序未定义，两式都只保证互异与界内） | 少 2×E UB、少 2 次块内 sync、少一趟 rank 读写；全局原子数不变 | ✅ |
| 2 | **expert 所有权 + 谓词前缀压缩（确定性）** | 每块独占一组 expert，扫全部路由，`mask = (flat==e)` → 前缀和 → 每命中 lane 写 `S_e + carry + rank - 1` | 数学等价且**顺序为索引升序**（与参考实现逐位相同） | 无原子、位级可复现；代价 = `nblocks × numel` 读 + `E` 遍比较 | ✅ |
| 3 | **UB 内计数/基数排序**（`chistv2` 直方图 + 分桶散射） | 先直方图（`dhistv2`，uint8/128 bin）→ 前缀 → 逐位（radix）重排到 UB | 数学等价 | 单遍 UB 内完成；**但**逐 lane 定 rank 需 match/ballot 语义 | ✅（判定见 R4） |
| 4 | **单块串行扫描** | 一个 core 顺序扫全部路由，per-expert 运行计数 | 数学等价（顺序 = 索引升序） | 零原子、零协调；串行步数 = numel | ✅ |
| 5 | **host 侧预计算**（torch `argsort`/`scatter` 在 host） | 把重排整体搬到 host | 数学等价 | 完全避免 kernel 内散射 | ❌（见 R4：违反「kernel 内完成」的算子契约 + 一次 host 同步 ≫ kernel 时延） |

**R2 有没有在线算法？** **有，且选定它。** 结构依据：本算子的全部跨元素状态是「每 expert 一个运行计数（游标）」，是可增量更新的 running 统计量 ⇒ 天然存在**单遍流式**变体（`slot` 由游标自增直接给出，无需先知道 `count_e`、无需中间缓冲，中间状态 O(E) 而非 O(numel)）。对照的两遍形式 = 基线 #0（先块内直方图/预留，再二趟写）与 #2（先计数再放置）。**可使单遍合法的充要条件是 op 契约允许 expert 内顺序任意**（§0.1，`_assert_route_equivalent` 归一化）；若契约要求「索引升序」的稳定序，单遍就必须带 intra-tile rank（即 #2 的谓词前缀），在线性随之丧失——该约束不适用本 op，故 #1 可行。

**R3 算法复杂度（候选间四口径量化对比）**

代表 workload：`numel = 32768`、`num_experts = 384`、`block_size = 64`、`P = 57023`（kimi-k2-t4096，最大规模）。口径：FLOPs 只计算术（本族算术为 0，故列「整型操作数 / 原子操作数」）；Bytes 只计 GM 流量（不含 L2 命中重读）；扫描遍数 = 对 `flat` 的 GM 读取轮次；中间缓冲峰值 = 单核片上驻留。

| # | 候选 | 算术 / 原语量 | 访存 Bytes（GM，代表） | 扫描遍数 | 中间缓冲峰值 | 可并行度 | 跨核归并代价 |
|---|------|--------------|----------------------|---------|------------|---------|------------|
| 0 | 两级 bump（基线） | 2·numel 原子（numel UB + ≈numel GM）+ 2·ceil(E/tile) sync | ≈264 KB（读 131 KB + 写 131 KB + 游标 1.5 KB） | 1 | 2·E·4 = 3.0 KB（UB） | 52 块 × 128 lane | 无（原子即协调） |
| 1 | **单级 bump（选定）** | numel GM 原子 + numel store + numel load | ≈264 KB（同上） | 1 | **0 B** | 52 块 × 128 lane | 无 |
| 2 | expert 所有权 + 前缀 | `E·numel/64` ≈ 196 K 比较组 + `E·numel/L·log2L` 前缀步 + numel store | **≈7.3 MB**（`nblocks × numel × 4B`，nblocks=56；L2 命中可降低但仍是 56× 放大）+ 131 KB 写 | **56** | flat tile + mask + prefix ≈ 12 KB | 56 块（每块扫全量，负载固定） | 无（按 expert 切分，写域天然不交） |
| 3 | UB 计数/基数排序 | 2–3 遍 histogram+scatter；**逐 lane rank 无法表达**（见 R4） | ≈264 KB（若可行） | 1（UB 内多遍） | keys 131 KB + out 131 KB = **262 KB > UB 248 KB** | 有限 | 无 |
| 4 | 单块串行 | numel 串行步 × O(1) | ≈264 KB | 1 | 4·E = 1.5 KB | **1 块** | 无 |
| 5 | host 预计算 | host 侧排序/散射 | + 一次 host↔device 同步 | 0（kernel 内） | — | — | — |

**R4 硬件亲和性（逐候选评估）**

检查清单：① 计算单元匹配 ② 片上容量 ③ 对齐/整除 ④ 静态边界 ⑤ 流水可融合性 ⑥ 跨核结构。

| # | ① 计算单元 | ② 容量 | ③ 对齐/整除 | ④ 静态边界 | ⑤ 流水 | ⑥ 跨核 | 结论 |
|---|-----------|-------|------------|-----------|-------|-------|------|
| 0 | 整型原子 + 标量 store 落 Vector 核（✓）；但需 UB 原子 + 2 次块内 sync | 3.0 KB ✓ | int32 标量访问无向量宽度要求 ✓ | `numel/E/bs` 全编译期 ✓ | 无明显 stage 可叠 | 原子即协调 ✓ | **淘汰**：块粒度下全局原子数不减（§0.6 R2 量化），却多 2 次 sync 与 3 KB UB ⇒ 严格劣于 #1 |
| 1 | ✓ 同上（1 原子 + 1 load + 1 store / 路由） | **0 B** ✓ | ✓ | ✓ | 单遍无 stage | ✓ | **选定** |
| 2 | 大量逐元素比较 + 手写前缀落 Vector 核（可行但主导计算是「每 expert 扫全量」） | 12 KB ✓ | ✓ | ✓ | 每 expert 一遍，无复用 | ✓ | **淘汰**：GM 流量 28× ↑、比较量 `E·numel/64` ≈ 196 K 组 ⇒ 发射项比 #1 高 1–2 个数量级（结构性，不依赖未实测常数） |
| 3 | 需 per-lane rank（match/ballot 语义）与 `chistv2`（**仅 uint8 输入、128 bin**） | **超容量**（262 KB > 248 KB） | E ≤ 1100 ⇒ 需 ≥2 位 8-bit 分桶 + 多次重排 | ✓ | 多遍 UB | ✓ | **淘汰**：能力缺失（per-lane rank 无 API）+ 容量超限（UB 253952 B，`design_budget`） |
| 4 | 单核串行 ⇒ 主导计算落成「串行标量累加」（重度失分） | 1.5 KB ✓ | ✓ | ✓ | 无 | 无 | **淘汰**：串行步数 = 32768 ⇒ ≥ 32768 cycle ≈ 20 µs（单核，且无流水掩盖） |
| 5 | 不落设备 | — | — | — | — | 需 host 同步 | **淘汰**：违反算子「设备内完成」契约（wrapper 只传 device 张量）；一次 host-visible 同步 ≥ 数十 µs |

> 负向淘汰的证据口径：`chistv2`/`dhistv2` 的「uint8 输入、128 bin」取自选中工具链 `tilelang/ascend/language/simd.py` 的函数 docstring（本机 site-packages，2026-11-23 读取）；UB 容量 253952 B 取自平台 `design_budget`（CANN 9.1.0 `Ascend950PR_9579.ini [AICoreSpec] ub_size`）；「per-lane rank 无 API」为选中模块 `simd.py` 导出清单核对（`ppack`/`vpack`/`vcadd` 无 match/ballot 语义）+ `T.ballot*` 属 common 层（CUDA 语义，未在 ascend 侧取证）⇒ 标「未文档化假设：ascend 无 per-lane rank 原语」。发射/流量类的定量论断均为结构性算术（不依赖未实测常数）。

**调研结论（选定算法族 + 依据）**

- **选定：单级 bump 游标（在线单遍，候选 #1）**——R3 中与基线 #0 同为最小流量（≈264 KB）与单遍扫描，中间缓冲 0 B；R4 中唯一「全部检查项通过且无冗余原语」的候选。它是基线 #0 的**在线化简**（把「块内直方图 + 批量预留 + 二趟写」折叠为「每路由一次游标自增」），源算法 #0 已成为其特例。
- **与基线的结构差异一句话**：基线是「两趟 + 块内中间状态（O(E) per block）」，选定算法是「单趟 + 仅全局游标状态（O(E) 全局）」，且不需要知道 `count_e`。
- **调研范围声明（针对「无更优替代」类断言）**：已核对 ① 平台算法候选库（无本族条目）② `kb_search`（无命中）③ `examples/` 全部 ascend 用例（仅 `paged_cache_write`，不同语义）④ 平台 knowledge（无瓶颈常数）⑤ 结构判据枚举出的 6 个候选（R1 表）；**在 O(E) 状态、单遍、无中间缓冲三个维度上，#1 已达下界**（每路由至少 1 次读 id、1 次定位、1 次写；bump 不动点），因此不存在「流量更低」的替代；「顺序更确定」的替代（#2/#4）以 56× 流量或 numel 串行为代价，本 op 契约下不采纳（作为确定性备选登记于 §9 R3）。**未检索到「无需 O(E) 游标状态」的算法族**（重排必须知道已落槽数），此为信息论级下界而非检索缺失。
- **源算子优化意图承接**：§0.4 #1（两级计数）→ §0.6 R2 的量化承接（全局原子数不减 ⇒ 单级更省）；#2（一 lane 一路由）→ 保留；#3（尾哨兵）→ O2；#4/#5（`s_count` 分离、expert 多轮）→ R3 删除并说明；#6–#8 → 保留。无静默丢弃。

**设计期估算（D-2）**

| workload（numel / E） | 流量项 | 发射/原子项 | 容量项 | 估算下界 |
|---|---|---|---|---|
| 8 / 384 | 0.2 KB ⇒ <0.01 µs | 1 块 × 8 lane ⇒ <0.01 µs | 0 B | 由 **launch/调度开销**支配（未实测假设 3–6 µs，本平台无 `cost_measurements` 条目） |
| 256 / 384 | 2 KB ⇒ <0.01 µs | 2 块 ⇒ <0.01 µs | 0 B | 同上（launch 支配） |
| 4096 / 128 | 33 KB ⇒ 0.03 µs | 32 块 × 128 路由 = 128 原子/核 × ~4 cyc ⇒ ≈0.5 K cyc ≈ 0.3 µs | 0 B | ≈0.3 µs（小于 launch 抖动） |
| 32768 / 384 | 264 KB ⇒ 0.22 µs | 52 块 × 630 路由 ⇒ 4 cyc/原子 ≈ 2.5 K cyc + 热点 expert 争用（count_e ≈ 85 × ~30 cyc ≈ 2.6 K cyc）≈ 5.1 K cyc ≈ 3.2 µs | 0 B | ≈3.2 µs |

**估算下界**：`≈3.2 us`（流量 0.22 + 发射/原子 ≈3.2 + 容量 0；代表 workload `numel=32768, E=384, P=57023`）。常数来源：**无匹配实测条目**（平台 `cost_measurements` 为空；`knowledge/bottlenecks.md` 明示无 950 真机瓶颈数据）⇒ HBM 有效带宽 1.2 TB/s、SIMT 原子吞吐 4 cyc/op、热点争用延迟 30 cyc、AIV 1.6 GHz **全部为「未实测假设 + 估算依据」**，正式值待 Stage 4 用 `msprof` Task Duration 回填（`hardware-cost-model.md` §3）。

#### 1.6.1 数学等价优化（公式级）

> 对象 = §1.6.0 选定算法（单级 bump）。本算子 `flops = 0`、无浮点/超越函数/公共子表达式，故不存在「除法转倒数 / rsqrt / 减 max」类代数化简空间（**这一点本身即结论**：全部计算是 3 个整型原语 `select/atomic_add/store`，见 §1.6.2 盘点）。以下 4 项是该算法上可采纳的全部等价改写，逐项给四要素。

**O1（核心）两级计数 → 单级 bump 游标**

- **原式**（基线 #0 / 源算子）：`rank = atomic_add(s_count[eid], 1, prev)`；`base_e = atomic_add(cumsum[e], s_count[e], prev)`（每块每 expert 一次）；`sorted[base_e + rank] = idx`。
- **优化后**：`slot = atomic_add(cumsum[eid], 1, prev)`；`sorted[slot] = idx`。
- **等价性论证**（契约级，非逐位）：对固定 expert `e`，两级形式把块 `b` 的路由放入 `[base_b, base_b + s_count_b[e])`，这些区间按块序首尾相接覆盖 `[S_e, S_e+count_e)`；单级形式给同一批路由分配的旧值序列在忽略到达顺序后恰为同一区间的一个排列。两式的差异**仅在 expert 内的排列次序**，而该次序按 §0.1 契约未定义（`_assert_route_equivalent` 每 expert 区段归一化）。整数语义上无舍入/无溢出（O3）。经 `verify_equiv.py` O1 项机器验证：8 个 shape × 3 种合法到达顺序（顺序 / 逆序 / 打乱）= 24 次重放，逐项核对「每 expert 多重集 = 基线 / 槽两两互异 / 槽在 `[S_e, S_e+count_e)` / 区段外哨兵保持 / 游标终值 = 起点 + count_e」→ **0 违反，EQUIV_PASS**。
- **收益量化**：单块 `tile = 128·EPT ≤ 640` 路由，`E ∈ {128,256,384}`（workload）⇒ 每块非空 expert 数上界 `min(tile,E) = tile`，故全局原子数上界 `nblocks·min(tile,E) ≈ numel`——**与单级相同，两级不省全局原子**；同时省掉 `2·E·4 B`（1.0–3.0 KB）UB、`2` 次 `sync_threads`/块、一趟 rank 缓冲读写。若 `tile ≫ E`（如 `threads=1024`）则两级可把全局原子数按 `E/tile` 缩小 ⇒ 保留为 §1.6.3 备选 B，交 Stage 4 实测裁决。

**O2 尾 lane：夹取索引 + 「-1 哨兵」取代带边界的分支读**

- **原式**：`if idx < numel: eid = flat[idx]`（分支化载入）。
- **优化后**：`eid = if_then_else(idx < numel, flat[min(idx, numel-1)], -1)`，后接 `if eid >= 0`（与源码同型）。
- **等价性论证**：对 `idx < numel`，两式的载入与结果逐元素相同；对 `idx ≥ numel`，两式都不产生任何原子或写（新式 `eid = -1` 被谓词拦掉，旧式整段被分支跳过），且 `min(idx, numel-1)` 保证不越界读 `flat`（`numel ≥ 1` 由 factory 的 `raise` 保证）。机器验证 O2：6 个含尾块的 shape（`numel ∈ {1,8,255,1032,4096,32768}` × `tile ∈ {128,640}`），`eid` 流逐元素相等、索引恒不越界 → **0 违反，EQUIV_PASS**。
- **收益量化**：把尾块从控制流降为表达式级 select ⇒ 全 lane 走同一执行路径（SIMT 谓词化友好），省 1 个分支块/路由组；不改变访存量。

**O3 int32 域容量核算（不改变算式，证明无需 int64）**

- **原式 / 优化后**：均为 int32（源码即 int32，但无容量论证）。
- **等价性论证**：全部 workload 的容量上界 `P = numel + (E+1)(bs−1) ≤ 102131 < 2^31`（最大者 `numel=32768, E=1100, bs=64`），槽值 `≤ P−1`、游标 `≤ S_e + count_e ≤ P` ⇒ int32 表示精确，与 int64 逐值相同。机器验证 O3：4 组极值组合的上界核算 → **0 违反，EQUIV_PASS**。
- **收益量化**：原子与 store 的元素宽度 4 B 而非 8 B ⇒ 游标原子与写流量减半（若改 int64，GM 流量 264 KB → 396 KB，`+50%`）。

**O4 expert 上界 guard（语义强化，契约内等价）**

- **原式**（源）：仅 `eid >= 0`。**优化后**：`0 <= eid < num_experts`。
- **等价性论证**：契约内（manifest `requires: in_range(0, num_experts)`）两式行为逐字相同（上界比较恒真）；契约外（`eid == num_experts`）源式会用 `cumsum[num_experts]` 做原子自增、**静默破坏 `num_tokens_post_pad`**，新式不改写任何状态。故为**契约域外的语义强化**，契约域内等价。机器验证 O4：5 个 `E ∈ {1,8,128,384,1100}` 的合法抽样，上界 guard 命中数恒为 0 → **0 违反，EQUIV_PASS**。
- **收益量化**：消除一个静默破坏上一阶段输出的失效模式；代价 1 次整型比较/路由（与 `eid >= 0` 合并为一次区间比较）。

**机器验证结果表**（`examples/moe_permute_align/_make_scatter_kernel/verify_equiv.py`，本机执行 `/workspace/venv-ascend950/bin/python verify_equiv.py`，torch CPU / int64 精确参照；本 kernel 无浮点运算，int64 精确比较即 fp64 参照的对应物）

| 项 | 对照（候选 vs 基线） | 规模 | 最大差 | 违反率 | 结论 |
|---|---------------------|------|-------|-------|------|
| O1 | 单级 bump vs 两级计数（8 shape × 3 到达顺序 = 24 次重放） | numel 1…32768；E 1…1100；bs 1…64 | 0（契约级逐项） | 0/24 | **EQUIV_PASS** |
| O2 | 夹取哨兵载入 vs 分支载入 | 6 shape（含尾块） | 0（逐元素） | 0/6 | **EQUIV_PASS** |
| O3 | int32 域容量（vs int64 语义） | 4 极值组合，最大 P = 102131 | 0 | 0/4 | **EQUIV_PASS** |
| O4 | 上界 guard vs 仅下界 guard | 5 个 E | 0（契约域内） | 0/5 | **EQUIV_PASS** |

**优化结论**：采纳 O1–O4 四项，机器验证 4/4 通过（失败项：无，见上表）。本算子 `flops = 0`、无浮点/超越函数/公共子表达式，因此不存在「除法转倒数 / rsqrt / 减 max / 算子降代换」类代数化简空间；O1（单级 bump）与 O2/O4（尾哨兵 + 上界 guard）即为该算法上全部可采纳的等价改写，O3 为 int32 域容量论证。全部采纳项的结果表均为 `EQUIV_PASS`。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

> 盘点范围：§3.3 伪代码中的**全部**循环与标量计算点。先说明一个对本平台至关重要的语义：在 `T.SimtVF(threads=128)` 内，`T.Parallel` 展开的逐 lane 表达式由向量单元以 **lane 并行**方式执行（降级代码的 `threadIdx.x` 即 lane 号），因此「每个 lane 一段标量 C 代码」**不是**串行标量计算，而是本平台的 SIMT 向量形态；真正的「串行标量」只有每线程 `EPT` 次迭代控制与块索引算术（O(1)）。

| # | 循环 / 标量点 | 性质 | 向量替代方案 | 判定 | 依据 |
|---|--------------|------|-------------|------|------|
| L1 | `bid` → 路由块基址 `(bid·EPT + i)·128` | block 级任务映射 | 无（每个核 1 个 `bid`） | **保留（合法理由：block 级索引/任务映射）** | 每核一次 O(1) 算术，非逐元素热点；`T.Kernel` 只给 1-D 核索引 |
| L2 | `for i in T.serial(EPT)`（EPT ≤ 5） | 每线程遍历自己的路由 | 无：各迭代的目标地址与原子地址由**运行期数据**决定 | **保留（合法理由：所需向量 API 缺失 + 数据相关地址）** | 见 S2/S3/S4；迭代间无规则 stride 可映射到同一向量寄存器 |
| S1 | `idx` 表达式 | 逐 lane 索引算术 | 编译器已折叠为 `block_idx·tile + i·128 + threadIdx.x`（探针 K6 降级码） | **已消除**（无需人工改写） | 探针 `design_probe_skeleton.py` K6 输出 |
| S2 | `eid = if_then_else(idx<numel, flat[min(...)], -1)` | 逐 lane 数据相关 **gather** | `T.simd.vgather2/vgatherb` **只在 UB 上取数**（`base` 为 UB 指针），不能对 GM 做 per-lane gather | **保留（合法理由：API 缺失）** | `simd.py::vgather2/vgatherb` docstring（"Gather elements from base using per-lane offsets"，SIMD 上下文 = UB） |
| S3 | `T.atomic_add(cumsum[eid], 1, return_prev=True)` | 逐 lane 标量 RMW（需**返回旧值**） | 平台唯一「tile 级原子」是 store-mode 批量加（`T.set_atomic("add","int32")` + `T.copy`），只能实现 `GM[tile] += UB[tile]`，无法表达「每 lane 对不同地址 +1 并取旧值」；且探针 K5 在本工具链下因 `T.Task` 单管线限制无法与向量语句同区 | **保留（合法理由：API 能力不匹配）** | `ascend/language/mode.py::set_atomic` docstring（"reduces into GM (D op= tile) … no vector loop"）；CANN `asc_atomic_add` 为**标量** SIMT 设备函数；探针 K2/K6 降级码 `condval = asc_atomic_add(&cumsum[eid], 1)` |
| S4 | `sorted_token_ids[slot] = idx` | 逐 lane 数据相关 **scatter store** 到 GM | `T.simd.vscatter(src, base, index)` 的 `base` 是 **UB** 缓冲 ⇒ 只能 UB 内散射；若改「UB 窗口 + DMA」需 ≥ P 的窗口（worse case 57023×4 = 228 KB，逼近 UB 253952 B 且与其它缓冲冲突） | **保留（合法理由：API 缺失）** | `simd.py::vscatter` docstring（"Scatter-store: base[index[lane]] = src[lane]"）；探针 K1/K6 编译通过并降级为 `out[pos[i]] = src[i]` |
| S5 | `if eid >= 0`（含 O4 上界） | 逐 lane 谓词 | 已是谓词化（lane 并行执行），无串行标量 | **已向量化（谓词形态）** | 探针 K6 降级码：`if ((eid < 385) && (0 <= eid)) { condval = … }` |
| S6 | 尾块判断 `idx < numel` | 逐 lane 谓词 | O2 已把控制流降为 select 表达式 | **已消除（表达式化）** | §1.6.1 O2 |

**向量化结论**：全部**逐元素计算点**（S1/S5/S6）已由编译器折叠/谓词化为 lane 并行形态；保留的 4 处（L1/L2/S2/S3/S4）理由分别为 **block 索引算术**（L1）、**数据相关地址 + 向量 API 缺失**（L2、S2、S3、S4，均附 `simd.py` 签名/探针降级码佐证）。**不存在**「因为实现简单而保留的标量循环」：唯一的循环 L2 的每次迭代都是一条**不同路由**的处理，其地址由运行期数据决定，任何向量 API 都无法表达（若改用 #2 的谓词前缀路径，代价见 §1.6.0 R3：56× 流量）。

#### 1.6.3 向量化轴与数据布局决策（阻塞级）

> 本算子的 I/O layout 是契约（`flat`/`sorted`/`cumsum` 均为 1-D int32 GM 数组）；自由变量是**核内任务映射与 lane 语义**。按 `gather / scatter / 索引类` 清单（skill Phase 2 第 3 项）枚举候选。

| # | 候选（核内布局 / lane 语义） | 轴整除与尾 lane 浪费 | 累加链形态 | repack 代价 | UB 容量 | 判定 |
|---|---------------------------|--------------------|-----------|------------|---------|------|
| 1 | **route 轴 → lane**（1 路由 / lane·迭代），GM 直写，无 UB | 尾 lane = `nblocks·128·EPT − numel`（最坏 512/33280 = 1.5%） | 无（每路由独立，无跨 lane 链） | **0**（无重排） | **0 B** | **选定** |
| 2 | expert 轴 → lane（lane = expert，路由串行） | 无尾 lane | 每 lane 一条 numel 长串行链 | 0 | ≈4·E = 1.5 KB | 弃选：串行步数 = numel（32768）⇒ ≥20 µs 单核级，且 lane 数（≤64 int32/指令）经 `T.Parallel` 映射后仍需 E 维展开 |
| 3 | SIMD 寄存器路径（`vld/vsts/vscatter`）：槽先聚到 UB 窗口再 DMA | 无 | 无 | 需构造 per-lane 槽索引向量（+4 B/路由） | 窗口需 ≥ `P·4 B`（最大 228 KB）⇒ 逼近 248 KB 上限 | 弃选：`vscatter` 只写 UB（API 约束）+ 容量不可行 |
| 4 | UB 中转 tile（`T.copy(flat_tile→UB)` 后逐 lane 写） | 同 #1 | 无 | 多一次 UB 往返（+2×131 KB UB 流量） | tile 4–16 KB | 弃选：每路由仅访问一次、无复用 ⇒ 纯增开销（且不解决 S4 的 scatter 地址问题） |
| 5 | UB 内计数/基数排序后连续 DMA 出 | 无 | radix 遍间有依赖 | 需 keys+out 双缓冲 | 262 KB ⇒ **超 248 KB** | 弃选：容量超限 + per-lane rank 无 API（§1.6.0 R4 #3） |

**评分与选定**：候选 #1 在「repack 代价 = 0」「UB 占用 = 0」「无串行链」「尾 lane 浪费 ≤1.5%」四项同时最优；#2/#3/#4/#5 分别因**串行化**、**API/容量**、**额外往返**、**容量+API** 淘汰（理由均为结构性或已核对的 API/容量事实，非「未考虑」）。**判定裕度不落在未实测常数上**（#1 与其余候选的差距是 1–2 个数量级或「不可行」），故按 skill 规则不需要为「轴选择」开实验裁决；但 O1 的**调度变体**（单级 vs 两级）确实依赖一个未实测常数（GM 原子 vs UB 原子的相对代价），故按规则给出裁决三件套：

**布局决策结论**：选定候选 #1（route 轴 → lane、1 路由/lane·迭代、GM 直写、UB 0 B），理由是 repack 代价 0 / 无串行链 / 无 UB 压力 / 尾 lane 浪费 ≤1.5% 四项同时最优；#2–#5 的弃选理由分别为串行化、API 约束、额外往返、容量+API（均为结构性或已核对事实，无「未考虑」项）。因主选与备选的判定裕度不落在未实测常数上，轴选择不开实验裁决；仅 O1 的**调度变体**（单级 vs 两级）给出下面的裁决三件套。

**主选方案（进 Stage 3）**：候选 #1 + §1.6.1 O1 的单级 bump；`threads=128`、`EPT = max(1, ⌈⌈numel/56⌉/128⌉)`、`nblocks = ⌈numel/(128·EPT)⌉`。

**备选方案 B（两级计数，结构完整，可直落 Stage 4 A/B）**：与主选**同构的 1-D 网格与 lane 映射**，块内加 3 个 UB 缓冲与 2 次 `syncthreads`：

```python
# 备选 B：块内直方图 + 每 expert 批量预留 + 二趟写（源算子的两级结构）
with T.Kernel(nblocks) as bid, T.SimtVF(threads=128):
    s_cnt = T.alloc_shared((num_experts,), "int32")   # 4·E B（≤4.4 KB）
    s_base = T.alloc_shared((num_experts,), "int32")  # 4·E B
    s_eid = T.alloc_shared((tile,), "int32")          # 4·tile B（≤2.6 KB）
    s_rnk = T.alloc_shared((tile,), "int32")          # 4·tile B
    for i in T.Parallel(tile):                        # 第 1 趟：块内计数
        idx = bid * tile + i
        eid = T.if_then_else(idx < numel, flat[T.min(idx, numel - 1)], -1)
        s_eid[i] = eid
        if 0 <= eid and eid < num_experts:
            s_rnk[i] = T.atomic_add(s_cnt[eid], T.int32(1), return_prev=True)
    T.sync_threads()
    for i in T.Parallel(tile):                        # 批量预留（每块每 expert 一次）
        if i < num_experts:                           # 若 E > tile 则外层再加 expert 轮
            if s_cnt[i] > 0:
                s_base[i] = T.atomic_add(cumsum[i], s_cnt[i], return_prev=True)
    T.sync_threads()
    for i in T.Parallel(tile):                        # 第 2 趟：写
        idx = bid * tile + i
        if 0 <= s_eid[i] and s_eid[i] < num_experts:
            sorted_token_ids[s_base[s_eid[i]] + s_rnk[i]] = idx
```

  该备选的 UB 预算：`3·E·4 + 2·tile·4 ≤ 3·1100·4 + 2·640·4 = 13200 + 5120 = 18320 B` ≪ 253952 B；分核三要素与主选相同（`nblocks`、56 AIV、`nblocks ≤ 56`）；**限制**：`E > tile` 时预留趟需外层 `T.serial` expert 轮（`⌈E/tile⌉`），且 `s_cnt`/`s_base` 尺寸随 `E` 变化（编译期常量，可接受）。

**实验裁决计划**：① 代表 shape：`numel/E ∈ {(32768,384), (32768,128), (4096,128), (256,384)}`；② 指标：`msprof` Task Duration（us）+ 以主选/备选差值反解「GM 原子 vs UB 原子的相对代价」；③ **判定阈值**：备选 B 在任一代表 shape 上 `< 主选 × 0.9`（即 ≥10% 收益）才翻转，且须在最大规模 shape（32768/384）上不劣于主选；④ 回写路径：实测数字经附录补记固化 §1.6.3（主选胜）或经 `perf_feedback.md` + `[DESIGN_LIMIT]` 受控路由回写设计（备选胜）；⑤ 按 shape 分派评估：`numel/E` 为编译期常量，若只有大规模 shape 上两级占优，可在 factory 内按 `tile ≫ E` 条件分派（无需全局二选一）。

---

## 2. 编程模式选型

### 2.1 模式结论

**SIMT**（`programming_modes.choices` 的键 `simt`）：`T.Kernel(nblocks)` 开 1-D 核网格，`T.SimtVF(threads=128)` 开 128-lane 线程域，块内用 `T.Parallel` / `T.serial` 展开「每线程 EPT 条路由」的 lane 并行计算。

### 2.2 选型理由

| 判据 | 结论 | 依据 |
|------|------|------|
| 计算类型 | 无矩阵乘（`flops = 0`）、无归约、无逐元素算术；全部工作 = 3 个**逐路由**原语：GM 数据相关 gather、GM 标量原子 RMW（取旧值）、GM 数据相关 scatter store | §0.3 / §1.6.2 |
| 三个原语在 SIMD 模式是否可表达 | **否**：① `T.simd.*` 为 UB↔向量寄存器操作，`vscatter`/`vgather*` 的 `base` 均为 UB 指针；② `simd.py` 导出清单中**没有任何原子**（`dir(T.simd)` 核对，本机 2026-11-23）；③ SIMD 路径无法对 GM 做 per-lane 索引写 | `tilelang/ascend/language/simd.py`；§1.6.2 S2/S3/S4 |
| 三个原语在 SIMT 模式是否可表达 | **是**（编译已验证）：`T.Parallel` 内 `out[pos[i]] = src[i]`、`T.atomic_add(GM[i], 1, return_prev=True)`、`asc_atomic_add(__gm__ int32_t*, int32_t)` | 探针 K1/K2/K3/K6（§11.1）；CANN `simt_api/device_atomic_functions.h` |
| 并行形态 | 128 lane 并行（每 lane 一条路由），lane 内 O(1) 标量代码由向量单元 lane-parallel 执行；无串行链 | 探针 K6 降级码 `#pragma unroll for (i…) { … threadIdx.x … }` |
| Cube / 混合模式 | 不适用：无 `T.gemm`、无 L0 需求、无 Cube↔Vector 协作 | §5.1 |

### 2.3 模式影响

- 核网格：`T.Kernel(nblocks)`（1-D，无 `threads` 参数——ascend 方言拒绝该参数）。
- 线程域：`T.SimtVF(threads=128)`；`threads` 是编译期常量（`T.Kernel` 同级框架参数），不是 wrapper 的 launch 参数（K9 已冻结接口只传 `(flat, sorted, cumsum)`）。
- 同步：全程无显式同步（`allow_autoschedule` 默认开启，见 §7）。
- 片上资源：UB/L1/L0 **全 0 B**（§4.5）。

### 2.4 与平台迁移默认的偏离说明

平台 `programming_modes.migration_default = simd`。**偏离为必需而非偏好**：SIMD 模式只能表达「UB 驻留的向量寄存器运算」，而本算子的三个原语（GM 数据相关 gather、GM 标量原子、GM 数据相关 scatter）在 SIMD 模式下无 API（§2.2 逐条佐证；同平台先例 `paged_cache_write` 亦因「数据相关地址散射写」从 simd 偏离到 SIMT，见其 `DESIGN.md` §2.4 与 `REVIEW.md` 第 76 行）。**Stage 4 的可调空间因此不在模式切换**，而在 `threads ∈ {64,128,256}`、`EPT`/`nblocks` 取值、以及单级 vs 两级计数（§1.6.3 裁决计划）。

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式为 §1.6.1 的**优化后公式**（O1 单级 bump + O2 尾哨兵 + O4 上界 guard）；每步使用选定平台可用的 API 或线程并行表达式，与 §1.6.2 的向量化结论一致。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `EPT = max(1, ⌈⌈numel/56⌉/THREADS⌉)`，`tile = THREADS·EPT`，`nblocks = ⌈numel/tile⌉` | 编译期常量（§5.2/§5.5）；`nblocks ≤ 56` 由构造保证 |
| 2 | `base = (bid·EPT + i)·THREADS + tx`；`tx ∈ [0,THREADS)`，`i ∈ [0,EPT)` | 每 lane 一条路由的索引（行主序双射，§0.6 R1） |
| 3 | `eid = if_then_else(base < numel, flat[min(base, numel−1)], −1)` | 夹取 + 哨兵（O2），无越界读 |
| 4 | `valid = (0 ≤ eid) ∧ (eid < num_experts)` | 谓词（含 O4 上界 guard） |
| 5 | `slot = atomic_add(cumsum[eid], 1)ₚᵣₑᵥ` | 单级 bump（O1）：取旧值即本路由的目标槽 |
| 6 | `sorted_token_ids[slot] = base` | 逐 lane 散射写（只覆写 expert 区段前 `count_e` 槽） |
| 7 | — | 无归约、无 epilogue、无中间结果写回 |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 | API 依据 |
|------|----------|-------------|------|------|---------|
| 1 | 1-D 核网格 | `T.Kernel(nblocks) as bid` | `nblocks`（Python int，编译期） | SIMT | `ascend/language/kernel.py::Kernel`（1-D、无 `threads`；探针 K6 降级出 `block_idx`） |
| 1' | 线程域 | `T.SimtVF(threads=128)` | `threads=128`（`latency` 默认 0） | SIMT | `frame.py::SimtVF`（签名 `threads: int \| list[int] \| tuple = 128`） |
| 2 | 每线程 EPT 条路由 | `for i in T.serial(EPT):` + `T.Parallel` 展开的 `tx` | `EPT` 编译期；`T.get_thread_binding()` 或 `T.Parallel` 隐式 lane | SIMT | 探针 K6：`for i in T.serial(EPT)` + `T.Parallel(tile)` 均降级为 `#pragma unroll` 的每线程循环 |
| 2' | lane 号 | `T.Parallel(tile)` 的隐式线程映射（推荐，统一由 `T.Parallel` 表达向量宽度） | `tile = 128·EPT` | SIMT | 探针 K7/K6 均用 `T.Parallel`（降级出 `threadIdx.x`） |
| 3 | 夹取 + 哨兵 | `T.if_then_else(cond, a, b)`、`T.min(a, b)` | 纯表达式（无分支） | SIMT | 探针 K6 同型；`T.if_then_else` 为 common 层表达式 |
| 4 | 谓词 | `if 0 <= eid and eid < num_experts:` | lane 级谓词 | SIMT | 探针 K1/K6（编译器另自动插入值域 guard） |
| 5 | 游标原子自增取旧值 | `T.atomic_add(cumsum[eid], T.int32(1), return_prev=True)` | `return_prev=True`；dtype int32 | SIMT | 签名 `atomic_add(dst, value, memory_order=None, return_prev=False, annotations=None)`；探针 K2/K6 降级 `condval = asc_atomic_add(&cumsum[eid], 1)` |
| 6 | 逐 lane 散射写 | `sorted_token_ids[slot] = idx` | 逐 lane 索引写 GM | SIMT | 探针 K1 降级 `out[pos[...]] = src[...]` + 编译器值域 guard；同平台先例 `paged_cache_write` §3.4 同类证据 |
| 7 | 构建 | `@tilelang.jit(out_idx=[], target="ascend")`（lazy 模式：函数返回 prim_func） | `out_idx=[]`；`target` 固定 ascend（GPU 源省略 target，默认取 cuda） | — | 签名 `jit(func=None, *, out_idx, target, target_host, …)`；**探针 K9** 用此装饰器形式端到端编译出 `JITKernel` |
| 8 | factory 缓存 | `functools.lru_cache(maxsize=32)`（替代源 GPU 的 `@target_cache`） | 键 `(numel, num_experts, block_size)` | — | 冻结 wrapper 的 `_compiled_align()` 缓存已承担；kernel 文件内保留 `lru_cache` 便于 Stage 3 直调 |

### 3.3 计算伪代码

**主选路径（进 Stage 3，与 §1.6.3 主选一致）**：

```python
import functools
import math

import tilelang
import tilelang.ascend.language as T

THREADS = 128          # lane 宽度（编译期常量，不是 wrapper 的 launch 参数）


@functools.lru_cache(maxsize=32)
def _make_scatter_kernel(numel: int, num_experts: int, block_size: int):
    """单级 bump 游标散射：slot = prev(atomic_add(cumsum[eid], 1))；sorted[slot] = idx。"""
    if numel < 1:
        raise ValueError(f"the scatter needs at least one route, got numel={numel}")
    max_padded = numel + (num_experts + 1) * (block_size - 1)   # = P（与 align 段一致）
    # 分核：把路由数按 56 个 AIV 摊平，再向上取整到 THREADS 的整数倍（§5.2 / §5.5）
    ept = max(1, math.ceil(math.ceil(numel / 56) / THREADS))    # 每线程路由数
    tile = THREADS * ept
    nblocks = math.ceil(numel / tile)                           # ≤ 56，编译期常量

    @tilelang.jit(out_idx=[], target="ascend")   # lazy 模式：返回 prim_func（K9 已验证）
    def _scatter():
        @T.prim_func
        def _scatter_main(
            flat: T.Tensor([numel], "int32"),
            sorted_token_ids: T.Tensor([max_padded], "int32"),
            cumsum: T.Tensor([num_experts + 1], "int32"),
        ):
            with T.Kernel(nblocks) as bid, T.SimtVF(threads=THREADS):
                for i in T.serial(ept):                          # 每线程 ept 条路由（展开）
                    for j in T.Parallel(THREADS):                # 128 lane 并行
                        idx = (bid * ept + i) * THREADS + j
                        # O2：夹取索引 + -1 哨兵，尾 lane 不越界读、不原子、不写
                        eid = T.if_then_else(
                            idx < numel, flat[T.min(idx, numel - 1)], -1
                        )
                        if 0 <= eid and eid < num_experts:       # O4：上界 guard
                            slot = T.atomic_add(                     # O1：单级 bump
                                cumsum[eid], T.int32(1), return_prev=True
                            )
                            sorted_token_ids[slot] = idx             # 逐 lane 散射写
        return _scatter_main

    return _scatter
```

**关键性质（供 Stage 3 核对）**：

1. `slot` 的**每一个取值都落在 `[cumsum[e], cumsum[e]+count_e)` 内**且按 expert 两两互异（`verify_equiv.py` O1）。代码**不读** `cumsum[e+1]`、也不读 `num_tokens_post_pad`，因此不依赖 align 段是否已把 count 写出来。
2. 越界安全网：探针 K6 显示编译器会自动为 `sorted_token_ids[slot]` 插入 `0 <= slot < P` 的值域 guard（契约内恒真，仅作安全网）。
3. `cumsum` 是**原地读写**参数：进入时是各 expert 的补齐起点，退出时是「起点 + count_e」。**不得**在 wrapper 里复用同一个 `cumsum` 张量做第二次调用（冻结 wrapper 每次 `forward` 新建，符合契约）。
4. `flat`、`sorted_token_ids`、`cumsum` 的 shape 全部编译期常量 ⇒ 无动态轴、无运行期边界计算。

### 3.4 API 可行性确认

> 方法：按 `workflow.md`（"仓库根的同名源码包会遮蔽安装包，静态查询不要在仓库根运行"），全部静态核对与**无卡编译**在**任务目录**（`examples/moe_permute_align/_make_scatter_kernel/`）内以 `/workspace/venv-ascend950/bin/python` 执行，先确认 `tilelang.__file__` = site-packages（非仓库根）。探针跑在 `ASCEND_RT_VISIBLE_DEVICES=""` 下，只做 lower + `bishengir` 编译，不 launch。

| API | 证据来源 | 验证状态 |
|-----|---------|---------|
| `T.Kernel(nblocks)`（1-D、无 `threads`） | `ascend/language/kernel.py::Kernel` docstring（"a 1-D grid of AI cores … has no `threads` parameter"）；探针 K6 编译通过，降级出 `block_idx` | ✅ 源码级 + **编译验证**（待 Stage 3 真机数值确认） |
| `T.SimtVF(threads=128)` | `frame.py::SimtVF` 签名；探针 K1/K2/K3/K6/K7 全部编译通过，降级为 `asc_vf_call<kernel>(cce::dim3(128), …)` | ✅ 编译验证 |
| `T.serial(EPT)` + `T.Parallel(THREADS)` 嵌套 | 探针 K7（`for t in T.serial` + 内层 `T.Parallel`）与 K6（`T.Parallel(tile)`）编译通过 | ✅ 编译验证 |
| `T.atomic_add(GM[i], 1, return_prev=True)` | 签名核对；探针 K2/K6 降级为 `condval = asc_atomic_add(&cumsum[eid], 1)`；CANN `simt_api/device_atomic_functions.h:31` 提供 `int32_t asc_atomic_add(__gm__ int32_t*, int32_t)`（`__NPU_ARCH__ == 3510` 分支，本机 CANN 9.2.1）；CANN 自带 `grid_sampler2_d_grad` 的 arch35 SIMT 设备代码以其做梯度累加（`/usr/local/Ascend/cann-9.2.1/opp/.../arch35/grid_sampler2_d_grad_simt.h:75`） | ✅ 编译验证 + 厂商 API/用法佐证；**运行时行为待 Stage 3** |
| 逐 lane 数据相关 GM store `GM[pos] = v` | 探针 K1 降级 `out[pos[...]] = src[...]`（并自动带 `0 <= pos < extent` guard）；同平台先例 `paged_cache_write` §3.4 的 `test_dynamic_outputs.py` 类证据 | ✅ 编译验证；**运行时待 Stage 3** |
| **§3.3 的完整 factory 形式**（`@tilelang.jit(out_idx=[], target="ascend")` lazy + `T.serial(ept)`×`T.Parallel(THREADS)` + 原子 + 散射写） | **探针 K9**：以 `(numel=64, E=8, bs=4)` 端到端编译出 `JITKernel`，降级码见 §11.1 | ✅ 编译验证（整形式） |
| `T.atomic_add(UB[i], 1, return_prev=True)`（仅备选 B 用） | 探针 K3 降级 `out[...] = asc_atomic_add(&s_count[key[...]], 1)` | ✅ 编译验证 |
| `T.cumsum(src, dst, dim=0)` | **不可用**：`tl.cumsum requires a target-specific implementation, but no cumsum implementation is registered for {"kind":"ascend",…}`（探针 K4/K8 实测报错原文） | ❌ 平台缺失 ⇒ 设计完全避开（备选 B 也不使用；确定性路径改手写 Hillis-Steele 前缀，探针 K7 编译通过） |
| `T.set_atomic("add","int32")` + `T.copy(UB→GM)` | **不可与向量语句同区**：`T.Task supports exactly one Ascend hardware pipe … mask=80`（探针 K5 实测报错原文）；仅在纯搬运 `T.Task` 内可用 | ⚠️ 本设计不使用 |
| `T.clamp` / `T.min` / `T.if_then_else` | common 层表达式；探针 K6/K7 使用 `T.min`、`T.if_then_else` 编译通过 | ✅ 编译验证 |

**未设备验证项汇总**（→ §9）：`asc_atomic_add` 的**运行时**正确性/争用性能、逐 lane 数据相关 GM store 的**运行时**正确性/效率、原子到达顺序造成的 expert 内顺序不确定。

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束（来自平台 ruleset / 探针） | 本算子是否涉及 | 处理方案 |
|---|---|---|
| `T.Kernel` 只有 1-D core grid、无 `threads` 参数 | **是** | §0.6 R1：1-D 网格 + `T.SimtVF(threads=128)` 线程域 |
| 逐元素表达式必须位于 `T.SimtVF`（SIMT 模式）内 | **是** | §3.3 伪代码的 `T.Parallel` 全在 `T.SimtVF` frame 内 |
| `tl.cumsum` 无 ascend 实现（探针实测） | **是（规避）** | 设计不含任何前缀和；确定性备选路径改手写 Hillis-Steele（探针 K7 已编译通过） |
| store-mode 原子（`set_atomic`）不能与向量语句同 `T.Task`（探针 K5 实测） | 否 | 不使用 |
| UB 容量 253952 B | **是（0 B）** | 主选路径 0 B；备选 B 18320 B ≪ 253952 B（§1.6.3） |
| L0A/L0B/L0C 仅 Cube 使用 | 否 | 无 `T.gemm`，L0 全部不适用（§5.1） |
| 跨 block barrier / 跨核 flag 未认证 | **是（规避）** | 无块间 barrier 需求：正确性只依赖 GM 原子，不依赖块序（§7） |
| 无动态轴（`numel/E/bs` 编译期常量） | — | 全部边界静态，`T.dynamic` 不使用 |
| 原子只支持标量、tile 级原子仅 store-mode 批量加 | **是** | §1.6.2 S3 已论证无法向量化替代 |
| 空分支/谓词内原子 | **是** | 探针 K2/K6 显示原子被包在 `if` 内、else 分支给 `0`；设计显式用 `valid` 谓词包住原子**与**写（避免 else 的 `0` 被当作合法 slot） |

#### 3.5.2 参考实现差异说明

| 差异项 | 参考（GPU / `target` 为 CUDA） | 本项目（ascend） | 转换方案 |
|--------|------------------------------|-----------------|---------|
| Kernel 结构 | `T.Kernel(scatter_blocks, threads=512)` 二维?（实为一维 + `threads`） | `T.Kernel(nblocks)` 一维，无 `threads` | §0.6 R1 |
| 语言模块 | `import tilelang.language as T` | `import tilelang.ascend.language as T` | 换命名空间；`alloc_shared` 语义 = UB（本设计不用） |
| 线程模型 | 512 线程、1 路由/线程 | `T.SimtVF(threads=128)`、EPT 路由/线程 | §0.6 R1（EPT 由 `numel` 与 56 核推出） |
| 计数结构 | 两级（SMEM `s_count` + 批量预留） | 单级 bump 游标 | §0.6 R2 / §1.6.1 O1（含量化承接） |
| E 维度处理 | `expert_rounds = ⌈E/512⌉` 多轮 | 无 E 循环 | §0.6 R3（E = 1100 与 E = 128 同构） |
| 尾块 | 夹取 + `-1` 哨兵 | 同（并补上界 guard） | §1.6.1 O2/O4 |
| 输出构建 | `@tilelang.jit(out_idx=[], compile_flags=["-O3"])` | `@tilelang.jit(out_idx=[], target="ascend")`（`compile_flags` 由平台构建控制；如需保留可在 Stage 3 加 `T.annotate_compile_flags`） | 构建层差异，无语义影响；`target` 必须显式固定（GPU 源默认 cuda） |
| factory 缓存 | `@target_cache(maxsize=32)` | `functools.lru_cache(maxsize=32)`（且 wrapper 自身已按 numel 实例缓存） | 框架层等价 |

#### 3.5.3 本项目同类实现参考

**结论**：本项目 `examples/` 下 ascend 后端的算子仅 `paged_cache_write` 一个（全库检索 `import tilelang.ascend.language` 只命中它与本任务链路），**无同语义（分桶计数散射）的兼容示例**；据此按「无兼容示例」显式记录，并以平台 kernel 模板 + 两个设计期探针构造最小实现。参考清单：

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/paged_cache_write/_paged_cache_write_kernel/DESIGN.md` | 高（承载方式） | 同为「数据相关地址的 GM 直写」：1-D 网格 + `T.SimtVF` + 逐 lane GM 写；风险登记与备选路径的组织方式 |
| `.agents/platforms/toolchains/ascend/kernel_template.py` | 高（骨架） | `@T.prim_func` + `T.Kernel` + `T.SimtVF(threads=256)` 的 1-D 结构、尾块写法、`--level` 分层测试骨架 |
| `examples/TileOPs/tileops/kernels/moe/moe_permute_align/_moe_permute_align_fwd_kernels.py` | 高（源语义/L0 基准） | 三个 GPU factory 原文与 `WARP_LANES` 等常量（迁移前状态） |
| `examples/TileOPs/tests/ops/test_moe_permute_align.py` | 高（比对规则） | `_assert_route_equivalent`：expert 内顺序归一化 + 区段外必须非合法 id |
| `examples/TileOPs/tileops/workloads/moe.py` | 高（golden） | `ref_permute_align` / `ref_program`：稳定顺序参考与哨兵填充 |
| `examples/elementwise/vec_add_2d_dynamic_shape.py` | 低（结构参考，**NPUIR 后端**） | 只借鉴动态边界组织；API 不作本平台依据 |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `flat` | `(numel,)`，`numel = T·K` | `int32` | 路由的 expert id；`numel` 为**编译期常量**（wrapper 按 numel 建实例） |
| `cumsum` | `(num_experts+1,)` | `int32` | align 段产出的补齐起点（**原地推进**）；末元 `num_tokens_post_pad` 本段不读写 |

12 条 workload 的实例化参数（`THREADS=128`，§5.2 的分核规则）：

| workload | T | K | numel | E | bs | P（输出长度） | EPT | tile | nblocks | UB |
|---|---|---|---|---|---|---|---|---|---|---|
| kimi-k2-t1 | 1 | 8 | 8 | 384 | 64 | 24263 | 1 | 128 | 1 | 0 B |
| kimi-k2-t32 | 32 | 8 | 256 | 384 | 64 | 24511 | 1 | 128 | 2 | 0 B |
| kimi-k2-t512 | 512 | 8 | 4096 | 384 | 64 | 28351 | 1 | 128 | 32 | 0 B |
| kimi-k2-t4096 | 4096 | 8 | 32768 | 384 | 64 | 57023 | 5 | 640 | 52 | 0 B |
| ds-v3-t1 | 1 | 8 | 8 | 256 | 64 | 16199 | 1 | 128 | 1 | 0 B |
| ds-v3-t32 | 32 | 8 | 256 | 256 | 64 | 16447 | 1 | 128 | 2 | 0 B |
| ds-v3-t512 | 512 | 8 | 4096 | 256 | 64 | 20287 | 1 | 128 | 32 | 0 B |
| ds-v3-t4096 | 4096 | 8 | 32768 | 256 | 64 | 48959 | 5 | 640 | 52 | 0 B |
| qwen3-235b-t1 | 1 | 8 | 8 | 128 | 64 | 8135 | 1 | 128 | 1 | 0 B |
| qwen3-235b-t32 | 32 | 8 | 256 | 128 | 64 | 8383 | 1 | 128 | 2 | 0 B |
| qwen3-235b-t512 | 512 | 8 | 4096 | 128 | 64 | 12223 | 1 | 128 | 32 | 0 B |
| qwen3-235b-t4096 | 4096 | 8 | 32768 | 128 | 64 | 40895 | 5 | 640 | 52 | 0 B |

测试集另有 `E ∈ {1,4,8,32,33,64,512,513,1024,1100}`、`bs ∈ {1,4,16,48,64,128}`、`numel ∈ {1,8,256,512,600,1032,2048,8192,16384}` 等组合（`examples/TileOPs/tests/ops/test_moe_permute_align.py`），分核规则对它们一律给出 `nblocks ≤ 56`（`numel=1` ⇒ EPT=1, nblocks=1）。

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `sorted_token_ids` | `(P,)`，`P = numel + (num_experts+1)(bs−1)` | `int32` | **读改写**：只覆写每个 expert 区段的前 `count_e` 个槽（∈ `[cumsum[e], cumsum[e]+count_e)`），区段外与补齐区保持 align 段写入的哨兵 `numel` |
| `cumsum` | `(num_experts+1,)` | `int32` | 副作用输出：`cumsum[e] ← cumsum[e] + count_e`（`e < num_experts` 全部严格前移；`cumsum[num_experts]` 不变） |

### 4.3 中间缓冲区

**无。** 主选路径不分配 `alloc_shared` / `alloc_l1` / `alloc_l0*` / `alloc_fragment`；标量 `eid / slot / idx` 落在寄存器（每 lane 3 个 int32 标量）。备选 B（§1.6.3）才需要 `3·E + 2·tile` 个 int32 的 UB 缓冲。

### 4.4 内存搬运路径

```
GM[flat]        ──(逐 lane 标量 load，clamp 索引)──► 寄存器 eid
GM[cumsum]      ──(逐 lane 标量原子 RMW：读-加-写，L2 常驻)──► 寄存器 slot（旧值）
寄存器 idx      ──(逐 lane 标量 store，数据相关地址)──► GM[sorted_token_ids]
```

- **无 UB / L1 / L0 参与**，无 `T.copy`、无 DMA、无 layout 变换（GMP→register→GM 直通）。
- GM 流量（代表规模 32768/384）：读 `flat` 131072 B + 写 `sorted` 131072 B + 游标原子 4·(E+1) = 1540 B（热区在 L2）≈ **264 KB**。
- 写粒度：单元素 int32 散射写，**不保证 128 B 事务合并**（地址数据相关）——这是本算子契约带来的固有代价，非分块引入（§5.3）。

### 4.5 UB 内存预算

| 项 | 元素数 | dtype | 字节 | 版本数 | 合计 |
|---|---|---|---|---|---|
| 主选路径全部 UB 缓冲 | — | — | **0** | — | **0 B / 253952 B（0.0%）** |
| 备选 B（若启用） | `3E + 2·tile` | int32 | `≤ 18320` | 1 | 18320 B / 253952 B（7.2%） |

容量来源：`statectl platform` 的 `design_budget.capacity_bytes.UB = 253952`（CANN 9.1.0 `Ascend950PR_9579.ini [AICoreSpec] ub_size`，md5 `2f0eca1f…`，读取日期 2026-11-23）。多版本膨胀：主选无缓冲 ⇒ `MaterializeMultiBuffer` 无作用对象。

### 4.6 动态轴定义

**无动态轴。** `numel`、`num_experts`、`block_size` 全部为 factory 入参（编译期常量，wrapper 按 `(numel, E, bs)` 建实例并缓存），`nblocks/tile/EPT/P` 全部由它们算出 ⇒ 所有循环边界与 guard 在编译期可证（§1.6.0 R4 ④ 通过）。GPU 源同样如此（`scatter_blocks`/`expert_rounds` 在 Python 侧算出）。故本算子不需要 `T.dynamic`。

### 4.7 JIT 配置

```python
@tilelang.jit(out_idx=[], target="ascend")   # 无返回值：结果体现在 sorted_token_ids / cumsum 上
```
factory 用 `@tilelang.jit(out_idx=[], target="ascend")` 的 lazy 形式（与源算子同构，只多固定 `target`；探针 K9 端到端编译通过），wrapper（Stage 0 冻结）调用 `self._scatter_fn()` 得到 `JITKernel` 并按实例缓存；本设计不引入 `pass_configs`（§7.3）。

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**：纯 Vector（SIMT lane 并行）的**索引散射**算子，`flops = 0`。

**判定依据**：全部工作 = 每路由 `1 load + 1 atomic RMW + 1 store`（§1.6.2 盘点），无矩阵乘（无 `T.gemm`）、无归约（无 `reduce_*`）、无窗口。因此 **Cube 路径与 L0A/L0B/L0C 全部不适用**，L1 也不需要，UB 占用 0 B（§4.5）。分核单位是 **AIV（Vector 核）**，不是 AIC。

### 5.2 Block 划分

```python
THREADS = 128                                              # lane 宽度（编译期）
ept     = max(1, ceil(ceil(numel / 56) / THREADS))         # 每线程路由数（编译期）
tile    = THREADS * ept                                    # 每个逻辑块的路由数
nblocks = ceil(numel / tile)                               # 逻辑块数 ≤ 56
```

- 代表规模（`numel = 32768`，kimi/ds-v3/qwen3 的 t4096 三档）：`ept = 5`、`tile = 640`、`nblocks = 52`。
- 中规模（`numel = 4096`）：`ept = 1`、`tile = 128`、`nblocks = 32`。
- 小规模（`numel ∈ {8, 256}`）：`ept = 1`、`nblocks ∈ {1, 2}`。
- **为什么按 56 摊平**：`nblocks ≤ 56 = 物理 AIV 数` ⇒ 每块独占一个核，硬件不做分波、无尾部空核（§5.5）。
- **为什么 `tile` 取 `THREADS` 的整数倍**：使 `idx = (bid·ept + i)·THREADS + j` 的 lane 映射与 128-lane 向量单元对齐，且 idx 计算退化为常量移位/乘加（探针 K6 降级码即 `block_idx·640 + i·128 + threadIdx.x`）。
- **为什么不做更深的分块**：每路由只被访问一次、写地址数据相关，块内无任何可复用数据（§1.6.3 候选 #4 弃选）。

### 5.3 约束分析

- **对齐约束**：`flat`/`sorted`/`cumsum` 均为 int32 标量访问；`flat` 的读取按 lane 连续（合并友好）；`sorted` 的写地址数据相关 ⇒ **无对齐/事务保证**（单元素写）。不引入 int32 向量宽度约束，故无「尾 lane 不满向量」的度量问题（浪费仅体现为尾 lane 空闲，见 §1.6.3 候选 #1 的 1.5%）。
- **UB 容量**：0 B（§4.5）✓。
- **L0 容量**：不适用（无 `T.gemm`）。
- **线程域宽度**：`threads = 128` 落在平台模板与先例（`paged_cache_write` 用 128；平台模板用 256）已验证区间；`T.SimtVF` 支持 64/128/256（Stage 4 可调）。
- **原子域**：`cumsum[e]` 的 32 位 RMW；每 expert 的原子次数 = `count_e`（随机路由下 ≈ numel/E：E=384 ⇒ ≈85，E=128 ⇒ ≈256）。这是**本算子唯一的争用热点**，§9 R8 登记为 Stage 4 杠杆。
- **静态边界**：全部循环边界（`nblocks/tile/ept`）与 guard（`idx<numel`、`eid<E`）编译期可证 ✓。
- **写域安全**：`slot ∈ [cumsum[e], cumsum[e]+count_e) ⊆ [0, total_padded) ⊆ [0, P)`；编译器另有值域 guard 作安全网。

### 5.4 注意事项

- **`cumsum` 的原地语义**：本段把游标推进到「起点 + count_e」。若 Stage 4/5 的 wrapper 在同一 `forward` 内复用 `cumsum`（例如重跑 scatter），第二次会写到错误槽位。冻结 wrapper 每次新建 `cumsum`，符合契约；Stage 3 的 L0 用例也必须每次重新构造 `cumsum`（§8.3）。
- **尾 lane**：`nblocks·tile − numel` 个 lane 必须走 `eid = -1` 分支（不读越界、不原子、不写）。最坏情况（`numel=32768`）为 512/33280 = 1.5% 空闲。
- **`E > THREADS`（最大 1100）**：单级 bump 无 expert 循环 ⇒ 无影响；`E = 1` 时所有路由争用 1 个游标（热点极端，正确性不受影响，性能见 §9 R8）。
- **`block_size` 只影响 `P`**（输出长度），不进入任何计算 ⇒ `bs ∈ {1,48,64,128}` 结构相同。
- **契约外 id**（`≥ E`）：O4 上界 guard 拦截，不破坏 `num_tokens_post_pad`（源实现会破坏）。

### 5.5 分核策略（物理核数适配）⭐

> 三要素（① 核数计算 / ② 物理核数实查依据 / ③ 规模判定与分核方案）按 `.agents/skills/_shared/standards/core-split-strategy.md` §1 组织。

**① 逻辑核数计算**：本算子无 `block_M/block_N` 输出网格（非 GEMM 类），逻辑核数 = 路由块数：

```
num_logical_kernels = nblocks = ceil(numel / (THREADS * ept)),
                      ept = max(1, ceil(ceil(numel / 56) / THREADS))
```
对 12 条 workload：`numel=8 ⇒ 1`；`256 ⇒ 2`；`4096 ⇒ 32`；`32768 ⇒ 52`（§4.1 表逐行给出）。

**② 物理核数依据（实查记录）**：本算子全部计算落在 Vector 核 ⇒ 按 **56 个 AIV** 分核。实查命令与返回值：

```
$ python3 .agents/tools/statectl.py platform --task-dir examples/moe_permute_align/_make_scatter_kernel
  hardware.hardware_id      = "ascend-950pr-9579"
  hardware.architecture     = "dav-3510"
  hardware.observed_cores   = {"cube": 28, "vector": 56}
  hardware.core_source      = "Ascend950PR_9579.ini [SoCInfo] ai_core_cnt=28, cube_core_cnt=28,
                               vector_core_cnt=56; runtime probe rtGetAiCoreCount()=28 on
                               Ascend950PR_9579, 2026-11-23"
```
（读取日期 2026-11-23；本机仅 1 张卡且被其他算子 Stage 3 占用，故未再跑 `torch.npu.get_device_properties`——平台记录即任务平台要求的实查结果，未使用任何假设值或「固定乘二」换算。）28 个 AIC 不参与（无 Cube 计算）。

**③ 规模判定与分核方案**：**三档统一为「无需适配」——设计上把逻辑核数**固化在 `≤ 56`：

| 规模档 | 判据 | workload | 结论与依据 |
|---|---|---|---|
| 小规模 | `nblocks ≤ 56` 且 < 物理核数的 1/4 | `numel ∈ {8, 256}` ⇒ 1–2 块 | **无需适配**：1–2 个 AIV 工作，其余空闲；绝对时延由 launch 支配（§1.6.0 估算行） |
| 中等规模 | `32–52` 块，接近但不超物理核数 | `numel = 4096` ⇒ 32 块（= 56×0.57）；`numel = 32768` ⇒ 52 块（= 56×0.93） | **无需适配**：`nblocks ≤ 56` ⇒ 每块独占一核、硬件不分波、无尾部波次浪费（52/56 ⇒ 4 个核空闲，浪费 = 4/56 = 7.1% 的**核**，但因为每核工作量被 56 摊平后取整，实际尾部为 0 波） |
| 极大规模（`numel` 远超 `56·128`） | `ept` 随 `numel` 线性增长 | 本任务无（最大 numel = 32768 ⇒ ept = 5）；契约上 `numel` 任意 | **核内串行**由构造实现：`nblocks` 恒 ≤ 56，超出的路由量全部进入 `ept`（每线程 `T.serial(ept)` 展开）；即「固定启动 56 核 + 核内分批」，摊薄核启动开销。若 Stage 4 发现大 `ept` 需更好均衡，可改为 `nblocks = 56` 固定 + 核内 `T.serial` 分块（等价语义，仅调度不同） |

- **分核方案自洽性核验**：`numel = 32768`：`ceil(32768/56) = 586` → `ept = ceil(586/128) = 5` → `tile = 640` → `nblocks = ceil(32768/640) = 52`，`52 ≤ 56` ✓；`numel = 4096`：`ceil(4096/56) = 74` → `ept = 1` → `tile = 128` → `nblocks = 32` ✓；`numel = 1`：`ceil(1/56) = 1` → `ept = 1` → `nblocks = 1` ✓（`≤ 56` 对任意 `numel` 恒成立，因为 `tile ≥ ceil(numel/56)`）。
- **不做跨核协同**：正确性只依赖 GM 原子（每 expert 游标的互异自增），**不依赖块执行顺序**，因此无需 `ascend_cross_core_*` 或 block barrier（后者在本工具链未认证，`programming.md` §同步与缓冲版本）。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| 路由块（`nblocks ≤ 56`） | block 级并行 | `T.Kernel(nblocks) as bid` | 1-D 核网格；每块独占一个 AIV（§5.5） |
| 线程 / lane（128） | 线程域声明 | `T.SimtVF(threads=128)` | 打开真实线程域（kernel 级不接受 `threads`） |
| 每线程 EPT 条路由 | 顺序 + 展开 | `for i in T.serial(ept)`（编译期常量上界，降级为 `#pragma unroll`） | 每迭代处理一条**不同**路由（数据相关地址），无跨迭代依赖，可安全展开以提升 ILP |
| lane 维（128） | lane 并行（向量形态） | `for j in T.Parallel(THREADS)` | 128 lane 同时处理 128 条路由（§1.6.2 结论：这是本平台的向量化形态） |
| 流水（`T.Pipelined`） | **不使用** | — | 单遍、无 tile 复用、无 UB 缓冲可多版本化（§6.3） |
| 常驻（`T.Persistent`） | **不使用** | — | 逻辑块数 ≤ 物理核数，硬件无需分波；超大规模由 `ept`（核内串行）承担（§5.5 ③） |

### 6.2 循环伪代码

与 §3.3 的主选伪代码一致，此处只标注调度属性：

```python
with T.Kernel(nblocks) as bid, T.SimtVF(threads=THREADS):   # 1 波、每核 1 块
    for i in T.serial(ept):                # 编译期常量（1…5），降级为 #pragma unroll
        for j in T.Parallel(THREADS):      # 128 lane 并行（向量单元 lane 并行执行）
            idx  = (bid * ept + i) * THREADS + j
            eid  = T.if_then_else(idx < numel, flat[T.min(idx, numel - 1)], -1)
            if 0 <= eid and eid < num_experts:
                slot = T.atomic_add(cumsum[eid], T.int32(1), return_prev=True)
                sorted_token_ids[slot] = idx
```

### 6.3 流水线优化

**不使用 `T.Pipelined`/多缓冲**，三条理由：

1. **无 stage 可叠**：本 kernel 只有一次数据相变（GM 读 → 原子 → GM 写），没有「搬入 tile → 计算 → 搬出 tile」的三段结构，也无 UB 缓冲可供多版本化。
2. **无跨迭代复用**：每条路由只被访问一次，`ept` 迭代之间没有共享数据。
3. **自动调度已开启**：`ascend/pipeline.py` 的 `allow_autoschedule` 默认开启，会跑 AutoSchedule / InsertSync / MaterializeMultiBuffer；本 kernel 无缓冲可多版本化，这些 pass 不产生额外结构（§7.3）。

**Stage 4 显式性能杠杆**（本设计只登记，需真机测量）：

| 杠杆 | 改动 | 预期方向 | 依据 / 风险 |
|------|------|---------|------------|
| `threads` 128 → 256 / 64 | 改 `THREADS` 常量；`tile = THREADS·ept` 相应变化 | 128→256 时 `nblocks` 减半（52→26，低于 56 核）、每 lane 工作量不变；64 则块数翻倍（更细粒度负载均衡） | 需重测 lane 宽度与向量单元宽度的匹配（§9 R6） |
| 单级 → 两级计数（§1.6.3 备选 B） | 块内 UB 直方图 + 批量预留 | 若 GM 原子争用主导 ⇒ 全局原子数降为 `min(tile,E)` 次/块 | 需 `syncthreads` 与 UB 原子；裁决阈值见 §1.6.3 |
| 按 expert 分流热点 | 若 `E` 小（如 1/8）导致单游标极端热点，可对热点 expert 用「每核子游标 + 后归一」 | 降争用 | **会改变槽分配方案**（需保持互异），属结构性改动，须回写设计；本阶段不采纳 |
| `ept` 调大 / 调小 | 改 56 的摊平目标 | 块数 vs 每核工作量的权衡 | §5.5 ③ |

### 6.4 尾块处理

| 尾块来源 | 是否可能 | 处理方式 | 代价 |
|---------|---------|---------|------|
| 路由维（`numel % tile ≠ 0`） | **是**（唯一真实尾块：`numel=1032`、`numel=8` 等） | `nblocks = ⌈numel/tile⌉` 覆盖最后一个不满块；块内 `idx ≥ numel` 的 lane 由 `eid = -1` 跳过（夹取索引保证不越界读） | 尾块空闲 lane ≤ `tile−1`（最坏 639，占 1.5%/`numel=32768`）；绝对代价可忽略 |
| expert 维（`count_e = 0` / 未满 `block_size`） | 是（常规） | 无需处理：`count_e = 0` 的 expert 区段不被任何路由命中；补齐区保持哨兵 | 0（哨兵由 align 段写） |
| `numel = 0` | 是（测试集含 `no-routes`） | factory 内 `raise ValueError`；wrapper 也按 `if numel else None` 不建本段 | 0（不 launch） |
| 契约外 id（`≥ num_experts` 或 `< 0`） | 契约内不会出现 | `0 <= eid < num_experts` 谓词拦截（O4） | 1 次区间比较/路由 |
| 槽位越界（`slot ≥ P`） | 契约内不会出现（`slot < total_padded ≤ P`） | 编译器自动值域 guard（探针 K6 观测到）作安全网 | 0 |

---

## 7. 同步策略

### 7.1 同步模式

**自动同步**（`allow_autoschedule` 默认开启，由 `InsertSync` 在检测到跨 pipe 依赖时插入事件），且本 kernel **没有任何跨 lane / 跨核的显式同步需求**：

- 块内：每条 lane 只读写自己的标量寄存器；`cumsum` 的互异槽分配由**硬件原子**保证（不需要 barrier）。
- 块间：块之间**无顺序依赖**（任一到达顺序都合法，§1.6.1 O1 已机器验证）。
- 段间：align → scatter 的可见性由**同流顺序 launch 的 kernel 边界**保证（host 侧，冻结 wrapper）。

### 7.2 同步点说明

| 位置 | 同步 API | 理由 |
|------|----------|------|
| kernel 内（块内） | **无**（不调用 `sync_threads` / `ascend_pipe_barrier` / `ascend_set_flag` 等） | 无 UB 缓冲、无跨 lane 数据交换；`T.Parallel` 内 atomic/store 是逐 lane 独立操作 |
| kernel 内（块间） | **无**（不使用 `ascend_cross_core_set_flag/wait_flag`） | 正确性只依赖 GM 原子；跨核 flag/barrier 在本工具链未认证（`programming.md` §同步与缓冲版本） |
| align → scatter（段间） | host 侧同流顺序 launch（kernel 边界） | `cumsum` 必须先写完；NPU 无认证的跨 block barrier，故沿用两次 launch（§0.1/§0.5 #1） |
| scatter 内部对 `cumsum` 的读改写 | 硬件原子（`asc_atomic_add` 的 RMW 语义） | 这是**唯一**的并发访问点；没有任何软件同步可替代（原子本身即同步原语） |
| 备选 B（若启用） | `T.sync_threads()` × 2（直方图后、预留后） | UB 缓冲的写→读、预留→写两处跨 lane 依赖 |

### 7.3 pass_configs 配置

```python
pass_configs = {}   # 不额外配置：默认流水线已含 AutoSchedule / InsertSync / MaterializeMultiBuffer
```

说明：① 构建走 `tilelang.compile(main, target="ascend", out_idx=[])`（与平台模板一致）；② 本 kernel 无 alloca/UB 缓冲，多缓冲相关 pass 无作用对象；③ 不使用 `annotate_unlimit_memory`（无容量压力）与 `annotate_manual_multi_buffer`。

---

## 8. 验证方案

> **设计期未做设备验证**：本节只给验证计划与 golden 草案，实际执行在 Stage 3。

### 8.1 Golden 函数

golden 以 **§0.1 的源算子语义**为唯一依据，直接移植源仓参考实现（`examples/TileOPs/tileops/workloads/moe.py` 的 `ref_permute_align`，稳定顺序 + 哨兵填充），**不复刻** §0.6/§3.3 的 NPU 算法（不含原子、不含 1-D 核网格、不含 `SimtVF`、不含单级 bump），保证验证独立性。

本段是「第二段」，因此 kernel 级用例必须先构造 align 段的产出（`cumsum` 补齐起点 + 哨兵填充的 `sorted_token_ids`），再调用本段：

```python
import math
import torch


def golden_scatter_setup(topk_ids: torch.Tensor, num_experts: int, block_size: int):
    """§0.1 语义：align 段产出（cumsum 补齐起点 + 哨兵填充）+ 稳定顺序散射参考。

    返回 (sorted_ref, cumsum_in, cumsum_out)：cumsum_in 是喂给 kernel 的输入，
    cumsum_out 是 kernel 跑完后 cumsum 应处的值（起点 + count）。
    """
    flat = topk_ids.flatten().to(torch.int64)
    numel = flat.numel()
    capacity = numel + (num_experts + 1) * (block_size - 1)

    counts = torch.bincount(flat, minlength=num_experts).to(torch.int64)
    padded = (counts + block_size - 1) // block_size * block_size
    cumsum_in = torch.zeros(num_experts + 1, dtype=torch.int64)
    cumsum_in[1:] = torch.cumsum(padded, 0)                    # cumsum_in[e] = 起点
    cumsum_in[num_experts] = int(padded.sum())

    sorted_ref = torch.full((capacity,), numel, dtype=torch.int64)   # 哨兵 = numel
    slot = cumsum_in[:num_experts].clone()
    for i, e in enumerate(flat.tolist()):                      # 稳定顺序（索引升序）
        sorted_ref[int(slot[e])] = i
        slot[e] += 1
    cumsum_out = cumsum_in.clone()
    cumsum_out[:num_experts] = slot
    return sorted_ref, cumsum_in, cumsum_out


def assert_route_equivalent(got, cumsum_in, cumsum_out, topk_ids, num_experts):
    """契约级比对（同 tests/ops/test_moe_permute_align.py::_assert_route_equivalent 的口径）：
    每 expert 区段内的多重集相等、槽互异、区段外哨兵保持、cumsum 精确推进。"""
    numel = topk_ids.numel()
    counts = torch.bincount(topk_ids.flatten().to(torch.int64), minlength=num_experts)
    for e in range(num_experts):
        lo, hi = int(cumsum_in[e]), int(cumsum_in[e]) + int(counts[e])
        seg = got[lo:hi]
        assert seg.numel() == int(counts[e])
        assert len(set(seg.tolist())) == seg.numel(), "slots must be distinct"
        assert torch.equal(torch.sort(seg).values,
                           (topk_ids.flatten() == e).nonzero().squeeze(1)), "expert multiset differs"
        assert torch.equal(cumsum_out[e], torch.tensor(int(cumsum_in[e]) + int(counts[e])))
    used = torch.zeros(got.numel(), dtype=torch.bool)
    for e in range(num_experts):
        used[int(cumsum_in[e]):int(cumsum_in[e]) + int(counts[e])] = True
    assert bool(torch.all(got[~used] == numel)), "sentinel overwritten outside routed slots"
```

> 与 `assert_route_equivalent` 的等价性说明：GPU 侧 `_assert_route_equivalent` 用「按 (expert, 路由索引) 归并后再比」的方式实现同一契约；上面的实现直接对每个 expert 区段比多重集并按索引升序比对，两者对合法输出给出同一判定（前者是后者的等价归一化形式）。

### 8.2 精度标准

- **判定类型**：**精确（Exact）**——全 int32 索引数组，无浮点、无舍入。比对口径 = §8.1 的契约级（expert 内多重集 + 槽互异 + 哨兵保持 + `cumsum` 精确推进），不是逐位相等（expert 内顺序按契约未定义）。
- **强于契约的自检**（可选）：由于本设计的 `verify_equiv` 表明「同一到达顺序下槽集合与稳定参考一致」，Stage 3 也可断言「每 expert 区段内按索引升序归一化后与 `sorted_ref` 逐元素相等」——该断言对任意原子顺序都成立（多重集 + 排序后比较），不依赖运行期到达顺序。
- **必须显式否定的错误**：① 任何写入落在 `[cumsum_in[e], cumsum_in[e]+count_e)` 之外；② 同一 expert 内重复槽；③ 哨兵被覆写；④ `cumsum[num_experts]` 被改写。

### 8.3 L0 / L1 / boundary 测试计划

> 完整分层套件（L1/L2/Boundary 命名与组织）交由 `tilelang-op-develop` 落地；本节给出 Stage 1 必须覆盖的门槛用例。**每个用例都必须重建 `cumsum` 与哨兵数组**（§5.4）。

| 层级 | 用例（numel / E / bs） | 覆盖点 | 断言 |
|------|----------------------|--------|------|
| **L0** | 8 / 384 / 64（kimi-k2-t1，manifest smoke） | 最小规模、`E ≫ THREADS`、`nblocks=1` | §8.1 全部契约项 + `cumsum` 推进 |
| **L0** | 4096 / 128 / 64（qwen3-t512） | `nblocks=32`、EPT=1、常规规模 | 同上 |
| **L0** | 32768 / 384 / 64（kimi-k2-t4096） | 最大规模、EPT=5、52 块并发原子 | 同上（**并发原子正确性的主用例**） |
| **L1** | 256 / 384 / 64；32768 / 128 / 1 | 小规模；`bs = 1`（`P = numel + E + 1`，无补齐余量） | 同上 |
| **boundary** | 1 / 1100 / 48 | 单路由 + `E > THREADS` | 单槽写入 + 其余全哨兵 |
| **boundary** | 1032 / 128 / 64 | `numel % tile ≠ 0`（尾 lane 512 个） | 尾 lane 不越界、不写、不原子 |
| **boundary** | 2048 / 1 / 64 | 单 expert 极端热点（`count = numel`） | 槽互异 + 全覆盖 |
| **boundary** | 8192 / 8 / 16 | 少数 expert、`numel/E ≫ 1` | 同上 |
| **boundary** | 1023 / 8 / 16（skew：全 id 相同） | 最坏偏斜 + 小规模 | 同上 |
| **boundary** | 0 / 8 / 16 | `numel = 0`（wrapper 不建本段） | factory `raise ValueError` |

---

## 9. 风险点与注意事项

### 9.1 已知约束与风险

| # | 风险 / 约束 | 触发条件 | 影响 | 依据 | 缓解 / 兜底 |
|---|---|---|---|---|---|
| R1 | **`asc_atomic_add` 运行时行为未设备验证** | 真机执行 | 若 GM 原子在 SIMT VF 下不可用/语义不同（例如不返回旧值），槽分配错误 | 探针 K2/K6 仅证明**编译**通过；CANN 头文件与厂商算子用法证明 API 存在 | Stage 3 **第一步**就跑 L0（8/384/64）确认；失败则切 §1.6.3 备选 B（仍用 GM 原子，但只做批量预留）→ 仍失败则切 §11.3 无原子确定性路径 |
| R2 | **逐 lane 数据相关 GM store 运行时未验证** | 真机执行 | 写丢失/写错槽 | 探针 K1 编译通过（编译器另插入值域 guard）；同平台先例 `paged_cache_write` 的同类证据但为「行级索引」 | L0 断言「槽互异 + 多重集相等」可立即暴露；兜底同 R1（§11.3 路径用同一 store，故若该 store 不可用则整族不可行 → 需上报能力缺口 `knowledge.capability_gaps`） |
| R3 | **expert 内顺序不确定**（原子到达顺序） | 每次运行 | 输出非逐位可复现 | op 契约允许（§0.1，`_assert_route_equivalent`）；源 GPU 实现同样不确定 | 测试按契约归一化；若下游需要位级可复现，切 §11.3 确定性路径（索引升序），代价见 §11.3 |
| R4 | **`cumsum` 原地推进** | 同一 `cumsum` 被二次使用 | 第二次写错槽位 | 源实现同样原地推进；冻结 wrapper 每次新建 | 文档化（§5.4）；wrapper 不改；L0 每次重建 |
| R5 | 契约外 id（`≥ E`） | 违约输入 | 源实现会破坏 `num_tokens_post_pad` | §1.6.1 O4 | 显式上界 guard（本次已采纳），不改变契约内结果 |
| R6 | SIMT lane 宽度 / 原子吞吐未实测 | 大 `numel` | 时延偏差（§1.6.0 估算基于未实测假设） | 平台 `cost_measurements` 为空 | Stage 4 用 `msprof` Task Duration 回填（§1.6.3 裁决计划） |
| R7 | 尾 lane 空闲 | `numel % tile ≠ 0` | ≤1.5% lane 浪费 | §1.6.3 候选 #1 | 无需处理（绝对代价 < launch 抖动） |
| R8 | 原子争用热点：每 expert 的原子次数 = `count_e` | `E` 小（1/8）或 `numel/E` 大（E=128 时 ≈256） | 该 expert 的原子串行化 | 结构性（同地址 RMW 必串行） | Stage 4 杠杆：两级计数（§1.6.3 备选 B）；`E=1` 场景的绝对时延仍由 numel/56 决定，可接受 |
| R9 | `nblocks ≤ 56` 依赖平台 56 AIV 记录 | 换型号/换 runtime | 分核参数不再最优（**不影响正确性**） | `statectl platform` 实查记录 | 分核参数由 `numel` 与常量 56 推出，换型号时按 §5.5 重算；正确性与块数无关 |
| R10 | 无 `T.cumsum`（平台缺失） | 若后续阶段想用前缀和实现确定性路径 | 需手写进位扫描 | 探针 K4/K8 报错原文 | §11.3 给出已编译通过的手写 Hillis-Steele 方案（探针 K7） |

### 9.2 常见错误（供 Stage 3 避坑）

1. **在 `if` 外使用 `slot`**：探针 K6 显示 `else` 分支给 `condval = 0`，若把 `sorted[slot] = idx` 写在 `if` 之外，非法 lane 会写坏槽 0。**必须**把原子与写都放在同一谓词内。
2. **忘记 `cumsum` 是原地输入输出**：不要传只读视图或常量张量。
3. **把 `num_experts` 上界 guard 省掉**：源实现省掉后，`eid == num_experts` 会原子自增 `cumsum[num_experts]`（即 `num_tokens_post_pad`），静默破坏 align 段结果。
4. **用 `T.Kernel(nblocks, threads=…)`**：ascend 方言拒绝 `threads` 参数（`kernel.py::Kernel`），线程域必须用 `T.SimtVF`。
5. **误以为 `T.cumsum` 可用**：本工具链无 ascend 实现（§3.4）。
6. **尾 lane 直接 `flat[idx]`**：必须 `T.min(idx, numel-1)` 夹取，否则越界读（探针 K7 的写法即夹取）。

### 9.3 特殊场景处理

| 场景 | 处理 |
|------|------|
| `E = 1`（全部路由同一 expert） | 单游标串行化 → 正确性由原子保证；时延由 `numel/56` 摊平（§9 R8） |
| `E > THREADS`（最大 1100） | 无影响：单级 bump 无 expert 循环（§0.6 R3） |
| `block_size = 1` | `P = numel + E + 1`；无补齐余量 ⇒ 每个 expert 区段恰好 `count_e` 槽，写域更紧，无需特殊处理 |
| `numel < tile`（如 8 路由、`tile=128`） | 单块、120 个 lane 空转；launch 主导（§1.6.0） |
| 所有路由都在少数 expert（skew） | 热点集中 → 争用上升但正确性不变；L0 用 skew 用例覆盖 |

---

## 10. 交付清单

### 10.1 目录结构

```
examples/moe_permute_align/_make_scatter_kernel/
├── DESIGN.md                                  # 本文档（Stage 1 交付）
├── RETROSPECTIVE.md                           # Stage 1 复盘（Phase 8）
├── verify_equiv.py                            # D-1 等价性机器验证（已执行，全 EQUIV_PASS）
└── history_version/
    ├── design_probe_primitives.py             # 设计期探针 1（原语可用性，编译级）
    └── design_probe_skeleton.py               # 设计期探针 2（主选/备选骨架 + 降级代码）
# Stage 3 产出（本阶段不创建）：
#   _make_scatter_kernel.py                    # kernel factory + golden + 分层测试（--level L0/all）
```

### 10.2 文件清单与职责

| 文件 | 状态 | 内容 |
|------|------|------|
| `DESIGN.md` | ✅ 本阶段 | §0 迁移分析 + §1–§11 设计 |
| `verify_equiv.py` | ✅ 本阶段（已执行） | O1–O4 四项机器验证，`EQUIV_PASS` |
| `history_version/design_probe_primitives.py` | ✅ 本阶段（已执行） | K1–K5 原语编译级验证 |
| `history_version/design_probe_skeleton.py` | ✅ 本阶段（已执行） | K1b/K2b/K6/K7/K8 骨架与降级代码 |
| `_make_scatter_kernel.py` | ⏳ Stage 3 | 按 §3.3 伪代码实现 factory + §8.1 golden + §8.3 分层测试 |

### 10.3 命名与接口约束（冻结，Stage 3 不得改）

- factory：`_make_scatter_kernel(numel: int, num_experts: int, block_size: int)`，返回零参可调用（`_scatter()`），编译产物以 `(flat, sorted_token_ids, cumsum)` 调用。
- 张量名与顺序：`flat`、`sorted_token_ids`、`cumsum`（与 `moe_permute_align.py::forward` 一致）。
- `out_idx=[]`（无返回值），`target="ascend"`。
- 不得改动 `numel < 1` 的 `raise` 行为（wrapper 与本文件的契约）。

### 10.4 实现顺序建议（Stage 3）

1. 先落 §3.3 主选路径 + §8.1 golden，跑 `--level L0` 的 8/384/64 用例 → 验证 R1/R2（原子与散射写的运行时可用性）。
2. 再跑 `32768/384/64`（并发原子）与 boundary 组（尾 lane、`E=1100`、`E=1`、`bs=1`）。
3. 全部通过后再交给 Stage 4；若 R1/R2 失败，按 §9.1 的兜底顺序切换（备选 B → §11.3 确定性路径），并按 AGENTS.md 「Compiler capability gaps」登记能力缺口。

---

## 11. 附录

### 11.1 设计期探针原始证据（编译级，`ASCEND_RT_VISIBLE_DEVICES=""`，未 launch）

**执行方式**（两条探针均在任务目录内、用站点 python 执行；先 `source .agents/platforms/toolchains/ascend/env-950pr.sh`）：

```
$ cd examples/moe_permute_align/_make_scatter_kernel
$ ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python history_version/design_probe_primitives.py
$ ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python history_version/design_probe_skeleton.py
$ /workspace/venv-ascend950/bin/python verify_equiv.py
# K9（追加于同一次探针文件内，逐一调用）:
$ ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python -c "import sys; sys.path.insert(0,'history_version'); \
    import design_probe_primitives as P; P.k9_factory_form()"
```

| 探针 | 内容 | 结果 |
|---|---|---|
| K1 | `T.Parallel` 内逐 lane 数据相关 GM store | **通过**；降级 `out[pos[(i*64)+threadIdx.x]] = src[...]` + 自动值域 guard |
| K2 | `T.atomic_add(GM[i], 1, return_prev=True)` | **通过**；降级 `condval = asc_atomic_add((&(cursor[key[...]])), 1)`，并带 `key<8 && 0<=key` guard |
| K3 | `T.atomic_add(UB[i], 1, return_prev=True)` | **通过**；降级 `asc_atomic_add((&(s_count[key[...]])), 1)` |
| K4 | `T.cumsum(UB_int32, dim=0)` | **失败**（原文）：`tl.cumsum requires a target-specific implementation, but no cumsum implementation is registered for {"kind":"ascend",…}` |
| K5 | `T.set_atomic("add","int32")` + `T.copy(UB→GM)` 于 SimtVF 内 | **失败**（原文）：`T.Task supports exactly one Ascend hardware pipe, but its body uses multiple pipes (mask=80)` |
| K6 | **主选骨架**（route-slice + 单级 bump + 散射写） | **通过**；降级代码见下 |
| K7 | **确定性备选骨架**（expert 所有权 + 手写 Hillis-Steele 前缀） | **通过**；降级出 `asc_syncthreads()` 与 9 步前缀循环 |
| K8 | `T.cumsum` 负向对照 | **失败**（同 K4，作为记录） |
| K9 | **§3.3 的完整 factory 形式**（jit lazy + `target="ascend"` + serial×parallel + 原子 + 散射写，`numel=64/E=8/bs=4`） | **通过**；产出 `JITKernel`，降级码见下（尾部 select、O4 guard、原子取旧值、散射写全部为预期形态） |

**K6 主选骨架的降级设备代码（原文，`numel=4096, E=384, tile=1024, threads=128`）**：

```c
__simt_vf__ __launch_bounds__(128) inline void main_kernel_simt_vf_0(
    __gm__ int32_t* flat, __gm__ int32_t* cumsum, __gm__ int32_t* sorted_token_ids) {
  #pragma unroll
  for (int32_t i = 0; i < 8; ++i) {
    int32_t eid = flat[(((((int32_t)block_idx) * 1024) + (i * 128)) + ((int32_t)threadIdx.x))];
    int32_t condval;
    if (((eid < 385) && (0 <= eid))) {
      condval = asc_atomic_add((&(cumsum[((int64_t)eid)])), 1);
    } else {
      condval = 0;
    }
    int32_t slot = condval;
    if ((slot < 28351) && (0 <= slot)) {
      sorted_token_ids[slot] = (((((int32_t)block_idx) * 1024) + (i * 128)) + ((int32_t)threadIdx.x));
    }
  }
}
extern "C" __global__ __vector__ void main_kernel(__gm__ int32_t* cumsum, __gm__ int32_t* flat,
                                                  __gm__ int32_t* sorted_token_ids) {
  asc_init();
  asc_vf_call<main_kernel_simt_vf_0>(cce::dim3(128), flat, cumsum, sorted_token_ids);
}
```

**K9（§3.3 完整 factory 形式）的降级设备代码（原文，`numel=64, E=8, bs=4`；`ept=1` 使外层 `T.serial` 被折叠）**：

```c
__simt_vf__ __launch_bounds__(128) inline void _scatter_main_kernel_simt_vf_0(
    __gm__ int32_t* flat, __gm__ int32_t* cumsum, __gm__ int32_t* sorted_token_ids) {
  int32_t condval;
  if ((((int32_t)threadIdx.x) < 64)) { condval = flat[((int32_t)threadIdx.x)]; } else { condval = -1; }
  int32_t eid = condval;
  if ((0 <= eid) && (eid < 8)) {
    int32_t slot = asc_atomic_add((&(cumsum[eid])), 1);
    if ((slot < 91) && (0 <= slot)) {
      sorted_token_ids[slot] = ((int32_t)threadIdx.x);
    }
  }
}
extern "C" __global__ __vector__ void _scatter_main_kernel(__gm__ int32_t* cumsum, __gm__ int32_t* flat,
                                                          __gm__ int32_t* sorted_token_ids) {
  asc_init();
  asc_vf_call<_scatter_main_kernel_simt_vf_0>(cce::dim3(128), flat, cumsum, sorted_token_ids);
}
```

即：① `idx < numel` 尾块谓词被折叠成 `threadIdx.x < 64` 的 lane guard（O2 生效）；② O4 的 `0 <= eid && eid < 8` 出现在原子前；③ `asc_atomic_add` 的返回旧值赋给 `slot`（O1 生效）；④ 散射写带编译器插入的 `slot < 91`（= P）值域 guard。**四条设计要点在同一份降级码中全部可见。**

**从 K6/K9 降级码得出的三条设计确认**：① 原子与 store 均为逐 lane 标量（不存在可向量化的 GM 散射/原子指令）；② 编译器**自动**为 `cumsum` 索引插入 `< 385`（= E+1，缓冲区 extent）与为 `slot` 插入 `< 28351`（= P）的值域 guard ⇒ 安全网存在，但 `eid == E` 仍能穿过（故 §1.6.1 O4 显式加上界 guard）；③ `else` 分支给 `slot = 0` ⇒ **写必须与原子同处一个谓词内**（§9.2 第 1 条）。

### 11.2 被否决候选的定量依据（补充 §1.6.0 R4 / §1.6.3）

| 候选 | 否决依据（可复核） |
|------|------------------|
| UB 计数/基数排序（#3） | ① `T.simd.chistv2/dhistv2` 只接受 **uint8** 向量、**128 bin**（`simd.py` docstring 原文："Accumulate a cumulative histogram of a uint8 vector into a uint16 vector. bin=0 returns cumulative counts through values [0, 127]…"）⇒ `E ≤ 1100` 需 ≥2 位分桶多遍；② 逐 lane 定 rank 需 match/ballot 语义，本模块只导出 `ppack/vpack/vcadd`（无 match）⇒ 「未文档化假设：ascend 无 per-lane rank 原语」；③ keys 131 KB + out 131 KB = 262 KB > UB 253952 B |
| `T.simd.vscatter` 直写散射（#3 变体） | `vscatter` 的 `base` 是 **UB** 缓冲（docstring："Scatter-store: base[index[lane]] = src[lane]"，SIMD 上下文 = UB 指针）；要散射到 GM 必须先在 UB 建 ≥ `P·4 B`（最大 228 KB）的窗口 ⇒ 容量不可行 |
| expert 所有权 + 前缀压缩（#2） | `nblocks × numel × 4 B` = 56 × 131 KB = **7.3 MB**（28× 放大）+ `E·numel/64` ≈ 196 K 比较组；`T.cumsum` 不可用 ⇒ 需手写前缀（探针 K7 已证明可编译，但代价成立）；作为**确定性备选**保留（§11.3），非主选 |
| 单块串行（#4） | 串行步数 = `numel` = 32768 ⇒ ≥32768 cycle ≈ 20 µs（单核、无并行度） |
| host 侧预计算（#5） | wrapper 只传 device 张量（冻结接口），host 计算需 D2H→排序→H2D，一次 host 同步 ≥ 数十 µs，且违反「设备内完成」契约 |

### 11.3 确定性备选路径（若 R1/R2 失败或需位级可复现）

**结构**（探针 K7 已编译通过，可作为 Stage 3/4 的直落实现）：**expert 所有权**（每块独占 `G = ⌈E/nblk⌉` 个 expert，`nblk = min(56, E)`），块内对每个 expert 顺序扫过路由 tile，用**手写 Hillis-Steele 前缀**（`log2(tile)` 步、每步 `T.Parallel` + `sync_threads`）得到谓词前缀，再逐 lane 写 `sorted[cumsum[e] + carry + rank − 1] = idx`；`carry` 为块内跨 tile 的运行计数（`T.alloc_var`）。

- **语义**：`rank` 取索引升序 ⇒ 输出与稳定参考**逐位相同**（比契约更强），无原子、无到达顺序依赖。
- **代价（代表规模 32768/384）**：GM 读 = `nblk × numel × 4 B` = 7.3 MB（L2 可吸收部分）+ `E × numel/64` ≈ 196 K 比较组 + 前缀 `E × numel/64 × log2(tile)` 步；UB = flat tile + mask + prefix ≈ 12 KB。按 §1.6.0 的未实测常数口径估算，发射项比主选高 1–2 个数量级。
- **适用**：仅在 ① 原子/散射写在真机不可用，或 ② 下游要求位级可复现时启用；切换点与阈值由 §9.1 R1/R3 触发，实测数据回写 §1.6.3。

### 11.4 调研来源清单（可追溯）

| 来源 | 用途 | 访问日期 |
|------|------|---------|
| `.agents/platforms/toolchains/ascend/knowledge/{INDEX,algorithm-candidates,bottlenecks,execution-patterns,PL-950-vector-execution}.md` | R1/R2 参考表命中检查（本族无命中）、代价常数缺失声明 | 2026-11-23 |
| `python3 .agents/tools/kb_search.py "scatter histogram atomic counting sort moe align scatter" --task-dir …` | 知识库检索 → `no entries matched` | 2026-11-23 |
| `tilelang/ascend/language/{kernel,frame,mode,allocate,simd}.py`（site-packages 0.1.15） | API 签名/语义/导出清单 | 2026-11-23 |
| `/usr/local/Ascend/cann-9.2.1/asc/include/simt_api/device_atomic_functions.h` | `asc_atomic_add(__gm__ int32_t*, int32_t)`，`__NPU_ARCH__ == 3510` | 2026-11-23 |
| `/usr/local/Ascend/cann-9.2.1/opp/.../grid_sampler2_d_grad/arch35/grid_sampler2_d_grad_simt.h` | 厂商算子用 `asc_atomic_add(__gm__ ptr, val)` 做散射梯度累加 | 2026-11-23 |
| `examples/TileOPs/tileops/{manifest/moe.yaml, workloads/moe.py, tests/ops/test_moe_permute_align.py, kernels/moe/moe_permute_align/*}` | 规格、golden、契约级比对、冻结接口 | 2026-11-23 |
| `/workspace/TileOPs/src/tileops/_kernels/moe/permute_align.py` | §0.1–§0.4 的源算子解读对象（全文 472 行） | 2026-11-23 |
| `examples/paged_cache_write/_paged_cache_write_kernel/{DESIGN.md,REVIEW.md}` | 同平台「数据相关地址 GM 直写」先例与风险组织方式 | 2026-11-23 |
| 本任务两个探针 + `verify_equiv.py` | 编译级 API 证据、契约级等价性机器验证 | 2026-11-23 |
| 互联网检索 | **未使用**（本族为经典计数排序/散射族，平台本地源与结构判据已覆盖候选枚举；按 `algorithm-research.md` §6 第 6 层降级规则记录） | — |
