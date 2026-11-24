# _make_align_kernel 算子设计文档

> 迁移任务（harness 子模式）。源算子：`/workspace/TileOPs/src/tileops/_kernels/moe/permute_align.py`
> 中的 `_make_align_kernel`（GPU TileLang，CUDA target）。目标：Ascend950PR_9579 / `ascend` 工具链
> （TileLang 0.1.15 + CANN 9.2.1，`target="ascend"`，`import tilelang.ascend.language as T`）。
> 设计范围：仅 `_make_align_kernel`（align 段）；`_make_scatter_kernel`、`_make_small_batch_kernel` 为同批次
> 另两个设计任务，本文只在 §0.1 说明分工。

## 0. 源算子解读与迁移分析

### 0.1 源算子语义（做什么）

`_make_align_kernel(numel, num_experts, block_size)` 构建 MoE「token → expert 对齐」流水线的**第一段**。
它以编译期常量 `numel`（= T×K 条路由）、`num_experts`、`block_size` 特化，一次调用写四个输出：

| 输出 | 语义 |
|---|---|
| `sorted_token_ids[max_padded]` | **只写哨兵**：全部 `max_padded` 个槽位写 `numel`（= `flat.numel()`）。真正的路由索引由随后的 scatter kernel 写入每个 expert 区段的前部 |
| `cumsum[num_experts+1]` | 各 expert **按 block_size 向上取整后**的计数之**独占前缀和**：`cumsum[0]=0`，`cumsum[e+1]=cumsum[e]+ceil(cnt_e/B)*B`；`cumsum[E]=` 补齐后总 token 数。既是输出（传给 scatter 当游标），也是跨 kernel 交接缓冲 |
| `expert_ids[max_num_blocks]` | 每个 block（`block_size` 行）的 owner expert：`b` 落在 `[cumsum[e]/B, cumsum[e+1]/B)` 时写 `e`；越过补齐总长的 block 写 `num_experts` |
| `num_tokens_post_pad[1]` | `cumsum[E]`（补齐后总 token 数） |

其中 `max_padded = numel + (num_experts+1)*(block_size-1)`，`max_num_blocks = ceil(max_padded/block_size)`。

**语义要素**（源码逐项核对，非臆测）：

- **数学语义**：`cnt_e = #{i : flat[i]==e}`；`padded_e = ceil(cnt_e/B)·B`；`cumsum = exclusive_prefix(padded)`；
  `owner(b) = e ⟺ cumsum[e] ≤ b·B < cumsum[e+1]`（源码以「expert 主序、变长 scatter」实现，见 §0.3 步骤 5）。
- **I/O 契约**：`flat` 为 `topk_ids.flatten()` 的视图（host 侧 `flatten().contiguous()`），一维 int32，
  取值域 `[0, num_experts)`；所有输出均为 int32 索引数组（无自由 dtype，manifest 单一 int32 用例）。
  **输出 shape 固定由 `numel/num_experts/block_size` 决定**，与运行时数据无关。
- **规约语义**：整数计数与前缀和——**无浮点累加，无舍入**；累加顺序不影响结果（整数加法结合律成立），
  这一点使 NPU 侧的并行重排完全无精度风险（§0.6、§1.6.1）。
- **dtype 语义**：全程 int32；中间量上界 `max_padded ≤ numel + (E+1)(B-1)`，本任务最大 57,023，远小于 int32 上限；
  `ceil` 用整数 `T.ceildiv`，无隐式提升。
- **边界语义**：① `numel = 0`（wrapper 仍会调用 align，只跳过 scatter）→ 全部计数 0、`cumsum` 全 0、
  `expert_ids` 全写 `num_experts`；② 某 expert 计数为 0 → 该 expert 的 block 区段为空，不写；
  ③ 连续多个零计数 expert → `cumsum` 出现相等相邻值，owner 判定必须取「最后一个满足 `cumsum[e] ≤ pos` 的 e」
  （源码 scatter 循环天然满足，§1.6.1 A2 的等价性论证覆盖此情形，已机器验证）；
  ④ `num_experts > threads`、`max_padded > threads` 时源实现靠外层 chunk 循环 + `s_carry` 处理。
- **host 侧语义**：`max_padded`/`max_num_blocks` 由 host 计算；`flat` 的 `flatten().contiguous()` 是**实现手段**
  （输入已要求连续，见 wrapper），不属于算子语义。
- **与另两个 kernel 的分工**：align 只做「计数 + 前缀 + block→expert 映射 + 哨兵填充」，
  **不写任何路由索引**；scatter（`_make_scatter_kernel`）在同一 stream 之后运行，读 `cumsum` 当游标、
  用 `atomic_add(cumsum[e], cnt_e)` 预留区段基点后把路由索引写进各区段前部；
  小批量融合路径（`_make_small_batch_kernel`，`numel < 1024 且 num_experts ≤ 32`）在一个 block 内完成
  计数、前缀、block 映射与 scatter。三者对外输出契约完全相同。本任务的 12 条 workload（E=384/256/128）
  全部走 align + scatter 路径（E > 32）。

### 0.2 源算子输入输出

| 张量 | 方向 | shape | dtype | 说明 |
|---|---|---|---|---|
| `flat` | in | `[numel]` | int32 | 路由 id（`topk_ids` 展平），`0 ≤ v < num_experts` |
| `sorted_token_ids` | out | `[max_padded]` | int32 | align 段只写哨兵 `numel` |
| `expert_ids` | out | `[max_num_blocks]` | int32 | owner expert / `num_experts` 哨兵 |
| `num_tokens_post_pad` | out | `[1]` | int32 | 补齐后总 token 数 |
| `cumsum` | out（内部交接） | `[num_experts+1]` | int32 | 独占前缀和，供 scatter 消费 |

源算子输出 shape（任务给定）：`sorted_token_ids [P] int32; expert_ids [ceil_div(P, block_size)] int32;
num_tokens_post_pad [1] int32`，其中 `P = T*K + (num_experts+1)*(block_size-1)`，与上表一致。

本任务 12 条 workload 的规模（manifest `MoEPermuteAlignFwdOp`，K=8，block_size=64）：
`numel ∈ {8, 256, 4096, 32768}` × `num_experts ∈ {384, 256, 128}`。

### 0.3 实现算法解读（怎么算）

**源码结构化阅读**（`_make_align_kernel` 与 host 侧 `forward`）：入口 `_align(threads)` → `T.Kernel(2, threads=threads)`；
2 个 block 分别承担「计数 + 前缀 + block 映射」（`bx==0`）与「哨兵填充」（`bx==1`）；
`threads ∈ [32, 1024]` 且为 32 的倍数（`forward` 中校验，默认 `MAX_BLOCK_THREADS`）。

**计算步骤分解表**（源码每条计算语句均归入某步骤）：

| # | 步骤 | 计算 | 输入 | 输出 | 对应语义 |
|---|---|---|---|---|---|
| 1 | 哨兵填充（block 1） | `for i in serial(ceildiv(max_padded, threads)): idx=i*threads+tx; if idx<max_padded: sorted_token_ids[idx]=numel` | — | GM `sorted_token_ids` | 全槽位哨兵 |
| 2 | 清零（block 0） | `s_counts[idx]=0`（shared，`idx<num_experts`） | — | SMEM | 直方图初始化 |
| 3 | 直方图（block 0） | `for i in serial(ceildiv(numel, threads)): if idx<numel: T.atomic_add(s_counts[flat[idx]], 1)` | GM `flat` | SMEM `s_counts[E]` | `cnt_e` |
| 4 | 分块 warp-scan（block 0，`for c in serial(num_chunks)`） | ① `s_vals[tx]=ceildiv(s_counts[expert],B)*B`（越界补 0）；② `for d in serial(WARP_SHUFFLE_STAGES)`：`shuffle_up(s_vals[tx], 1<<d)` 且 `lane≥stride` 时累加 → warp 内包含式扫描；③ `lane==31` 把 warp 和写 `s_warp_sum`；④ `tx==0` 串行累加 `num_warps` 个 warp 和，得 `s_warp_excl[w]` 与 `s_total`；⑤ `excl = s_vals[tx]-own_padded+s_warp_excl[warp_id]+s_carry[0]` 写 `s_cumsum[expert]` 与 GM `cumsum[expert]`；⑥ `s_carry += s_total` | SMEM `s_counts` | SMEM `s_cumsum`、GM `cumsum` | `cumsum[e]` 独占前缀（按 `threads` 分块，块间用 `s_carry` 传递） |
| 5 | 收尾（block 0） | `s_cumsum[E]=s_carry[0]`；`cumsum[E]=s_carry[0]`；`num_tokens_post_pad[0]=s_carry[0]` | SMEM | GM | `cumsum[E]`、`num_tokens_post_pad` |
| 6 | 越界 block 哨兵（block 0） | `for i in serial(ceildiv(max_num_blocks, threads)): if blk >= s_carry[0]//B and blk<max_num_blocks: expert_ids[blk]=num_experts` | SMEM | GM `expert_ids` | block 区段之外无名 |
| 7 | block→expert 映射（block 0，`for c in serial(num_chunks)`） | `e_start=cumsum[e]//B; e_blocks=cumsum[e+1]//B-e_start; for b in serial(e_blocks): expert_ids[e_start+b]=e` | SMEM `s_cumsum` | GM `expert_ids` | `owner(b)` |

**数据流图（源硬件视角）**：

```
GM flat[numel] --(per-thread 标量读, 线程 idx 网格)--> SMEM s_counts[E]   [步骤 3, shared atomic]
SMEM s_counts --(ceil * B)--> SMEM s_vals[threads] --(warp shuffle 树 + warp 和串行)-->
    SMEM s_cumsum[E+1] / s_carry --(GM store)--> GM cumsum[E+1]           [步骤 4-5]
SMEM s_cumsum --(÷B, 变长内层循环, 逐元素 GM store)--> GM expert_ids       [步骤 6-7]
（并行 block 1）GM sorted_token_ids <--(逐线程条件写常量)----              [步骤 1]
```

- **循环与并行结构**：grid = 2 block；block 内 `threads` 个线程（默认 `MAX_BLOCK_THREADS`）；
  `tx % 32` 为 lane、`tx // 32` 为 warp；`num_chunks = ceildiv(num_experts, threads)` 为专家方向的分块数。
- **保留的中间量**：`s_warp_excl/s_total/s_carry` 是**跨 chunk 的前缀传递状态**（`s_carry`），
  仅在 `num_experts > threads` 时非零。
- **host 侧逻辑**：`max_padded`/`max_num_blocks` 计算；`forward` 里分配 4 个输出 + `cumsum`；
  `target_cache` 按 `(numel, num_experts, block_size)` 特化构建；`threads` 合法性校验（32 的倍数）。

**完成标准核对**：源码中每条计算语句都出现在表中（步骤 1–7 覆盖 `_align_main` 全部语句，
含 `alloc_shared`/`alloc_var` 之外的 `if` 守卫与循环边界）；数据流图覆盖 `s_counts/s_vals/s_warp_sum/
s_warp_excl/s_total/s_carry/s_cumsum/sorted_token_ids/expert_ids/num_tokens_post_pad/cumsum` 全部 buffer；
无臆造步骤。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 源码特征 | 目的 | 依赖的源硬件特性 | 初判 |
|---|---|---|---|---|---|
| O1 | SMEM 暂存全部中间量 | `s_counts/s_vals/s_warp_*/s_cumsum` 均为 `alloc_shared` | 计数与扫描全程片上，GM 只写最终结果 | shared memory（~228KB/CTA 级） | 硬件强相关（映射为 UB） |
| O2 | **shared 原子计数直方图** | `T.atomic_add(s_counts[flat[idx]], 1)` | 一趟完成 scatter-add 式直方图，O(numel) | SMEM 原子加（字级硬件原子） | 硬件强相关（NPU 无 UB scatter-add） |
| O3 | **warp shuffle 树形扫描** | `T.tvm_warp_shuffle_up(..., 1<<d)`，`WARP_SHUFFLE_STAGES=5` | 段内前缀和 O(log 32) 而非 O(32) | warp（32 lane 同步寄存器交换） | 硬件强相关（NPU 无 warp） |
| O4 | 两个 block 分工 + 无依赖并发 | `bx==1` 只做哨兵填充，`bx==0` 做计数/扫描 | 填充（纯带宽）与计数（原子/扫描）重叠 | 多 CTA 并发、GM 写冲突由后续 launch 保证 | 模式可移植（但映射需重设计） |
| O5 | 跨 chunk 前缀 carry | `s_carry` + `num_chunks` 外层循环 | 让 `num_experts > threads` 仍只需一份 SMEM 扫描 | — | 模式可移植（E ≤ 384 在 NPU 上整块驻留后**不再需要**） |
| O6 | 变长 scatter 写 `expert_ids` | `for b in serial(e_blocks)`，迭代数数据相关 | 写数最少（= 有效 block 数） | 线程级发散容忍 | 实现层（数据相关 trip count，反向量化） |
| O7 | 构建期 shape 特化 | `target_cache` 按 `(numel, E, B)` 缓存；循环上界全静态 | 消运行期分支/元数据 | host | 可移植（NPU 同样按 shape 特化） |
| O8 | 两 kernel 流水（align 只填哨兵） | 路由索引留给 scatter，align 不读 `flat` 第二遍 | align 无需 compaction/gather；哨兵一次成块写 | — | **算法层**，可移植 |
| O9 | 原子计数替代排序 | 不做 sort，直方图 + 前缀两步 | 复杂度 O(numel + E)（不含原子争用） | 同 O2 | 算法层意图，需等价替换承载 |

**算法层 vs 实现层**：O8、O9 属算法层（两段流水 + 直方图替代排序），迁移优先保留其意图；
O1/O2/O3/O4/O5/O6 属实现层（SMEM、原子、warp、block 分工、chunk carry、变长发散写），需逐项判定（§0.5）。

### 0.5 硬件耦合性分析与 NPU 适配决策

判定输入含 §1.6.0 调研结论（调研先于判定）。目标平台事实（`tilelang.ascend.language` + 本平台 ruleset）：

- `T.Kernel(*blocks)` 是**一维 AI core 网格，签名没有 `threads=`**（`ascend/language/kernel.py`），
  线程域必须由 `T.SimtVF(threads=...)` 或 `T.SimdVF()` 在核内显式声明。
- 片上层级：UB 248KB（`T.alloc_shared`）/ L1 512KB / L0C 256KB / L0A-B 64KB；本算子不使用 Cube 单元。
- 无 warp 概念；`T.shfl_*`/`T.tvm_warp_shuffle_*` 属 CUDA 公共面，迁移不可用。
- 原子：已验证路径是 **GM 侧** `T.set_atomic("add", dtype)` + `T.copy` 写回（PL-950-atomic-splitk，
  FP32 用例）；**UB/共享内存的 scatter-add 没有可用证据**（`atomic_add` 的向量写回语义见
  `examples/elementwise/atomic_add.py`，但该文件是 `target="npuir"` 的另一工具链，不作为本工具链证据）。
- 跨核同步：`ascend_cross_core_set_flag/wait_flag` 只有接口声明，平台明示「跨核 flag 与跨 block batch
  barrier 仍未认证」；`T.sync_grid()` 仅存在于公共面，ascend 后端无降低实现。

| # | 源条目 | 层 | 处置 | 依据 |
|---|---|---|---|---|
| D1 | O8 两 kernel 流水（align 只填哨兵） | 算法 | **保留** | 纯算法结构，与硬件无关；NPU 上同样只需哨兵 + 前缀（§1.6.0 R1 基线） |
| D2 | O9 直方图 + 前缀替代排序 | 算法 | **保留（意图）** | 排序在 NPU 上无可用 API，直方图路线仍是 O(numel+E) 最小路线（§1.6.0 R1/R2） |
| D3 | O1 片上暂存 | 实现 | **等价替换** | SMEM → UB（`T.alloc_shared`）；E+1 ≤ 385 个 int32 全部驻留 UB（1.6KB） |
| D4 | O2 shared 原子直方图 | 实现 | **重新设计** | NPU 无 UB scatter-add；改为「expert lane 比较累加」直方图（§0.6 R2），GM 原子方案留作备选 M2 |
| D5 | O3 warp shuffle 扫描 | 实现 | **重新设计** | 无 warp；改为 UB 内 Hillis–Steele 向量扫描（§0.6 R3），意图（O(log E) 深度）由 log 步数承接 |
| D6 | O4 两 block 分工并发 | 实现 | **保留意图 + 重新设计映射** | 两段无数据依赖，NPU 上按 core grid 重新映射（§0.6 R1）；填充与直方图仍并发 |
| D7 | O5 chunk carry | 实现 | **舍弃** | E ≤ 384 与 UB 容量（248KB）相比可忽略，无需分块扫描，`s_carry` 无对应物；理由：NPU 全量驻留 → 不需要跨 chunk 状态 |
| D8 | O6 变长 scatter 写 expert_ids | 实现 | **重新设计** | 数据相关 trip count 阻碍向量化/分核；改为 rank-count 恒等式（§0.6 R4 / §1.6.1 A2） |
| D9 | O7 构建期 shape 特化 | 实现 | **保留** | 两边都按 `(numel, E, B)` 特化；NPU 侧循环上界全静态（`@lru_cache`/每实例缓存） |
| D10 | `threads` 启动参数 + 32 倍数校验 | 实现 | **舍弃** | NPU launch 签名无 `threads`（K9），线程域由 `T.SimtVF` 在核内决定 |
| D11 | `T.sync_threads()` 显式同步 | 实现 | **等价替换** | 单核内 UB 写后读由 ascend 默认流水（AutoSchedule/InsertSync）保证；本设计不引入跨核依赖，故无需显式同步点 |

**不确定项**：D4 的备选（GM 原子累加，`set_atomic("add","int32")`）在 int32 上**未文档化**（PL-950-atomic-splitk
只验 FP32）；因此不作为主选，只在 §1.6.3 备选 M2 与 Stage 4 裁决计划中出现。

### 0.6 NPU 算法重设计

重设计对象：D4/D5/D8 + D4/D6 的并行结构；承接 §1.6.0 选定的算法族（见 §1.6.0 调研结论）。
源优化意图（O1–O9）逐项承接说明见表末。

**R1 并行结构重设计（源：`T.Kernel(2, threads)` 线程级并行 → NPU：一维 core grid + 核内 lane）**

- 源方案：2 个 CTA，`bx==0` 承担计数/扫描，`bx==1` 承担哨兵填充；块内 `threads` 线程按 `tx` 切分数据。
- NPU 新算法：`T.Kernel(G)` 一维核网格，每核内部用 `T.SimtVF(threads=...)` 的 lane 并行表达式处理
  一段连续 token / 一段连续输出 block。由于**跨核无可用 barrier**，凡「读全量计数」的步骤必须在
  **同一核内**完成，因此按规模分两条路径（§5 分核、§6 循环）：
  - 路径 S（`numel` 小）：1 次 launch，grid=2 —— 核 0 做「直方图 + 补齐 + 前缀 + block 映射 + 上半填充」，
    核 1 做「下半填充」；与源结构一一对应。
  - 路径 L（`numel` 大）：2 次 launch —— launch-1 `grid=G_h` 各核算**局部直方图**并写入独立 GM 槽位
    `partial[G_h, E_pad]`（无原子、无初始化契约）；launch-2 `grid=G_a` 各核**各自**读全量 `partial`
    并冗余求和，得到同一份 `cumsum`（整数确定性，逐位相同），核 0 负责写 GM `cumsum`/`num_tokens_post_pad`，
    各核再分担 `expert_ids` 与哨兵填充的分片。**跨核依赖被「冗余读同一份只读数据」消除**。
- 语义保持论证：整数计数/前缀和的并行分组不改变结果（加法结合律、无舍入）；写域互不相交
  （`cumsum`/`num_tokens_post_pad` 由核 0 独占，`expert_ids` 与填充按 block / 下标分片），无写冲突；
  边界 `numel=0` 时 `G_h=1`，该核写出全零行，launch-2 结果与源一致（全 `num_experts`）。
- 验证：§8 golden（移植源仓 `ref_permute_align`）+ §1.6.1 verify_equiv 已覆盖多核等价（同值冗余）。

**R2 直方图重设计（源：shared 原子 scatter-add → NPU：expert-lane 比较累加）**

- 源方案：`atomic_add(s_counts[flat[idx]], 1)`，靠 SMEM 原子把 O(numel) 次随机字增量摊到硬件。
- NPU 新算法：**lane ↔ expert id**，每个 lane 持有一个私有计数器；对 token 块逐个 token 广播比较、
  累加 `[x_i == e]`。每核工作量 `tokens_per_core × ceil(E/V)` 次向量比较（V = 向量 lane 数，int32 时 64）。
  等价性：`cnt_e = Σ_i [x_i==e]` 是计数的定义式，与原子加法的结果**逐位相同**（整数）。
- 备选（M2，§1.6.3）：GM 原子 `T.set_atomic("add","int32")` 合并局部直方图，省掉 launch-2 的冗余读；
  int32 原子未在本工具链取证 → 仅备选。
- 代价：由 O(numel) 的随机字增量变为 O(numel·E/V) 的规整向量比较——**这是本迁移的主要代价项**
  （§1.6.0 R4 与 §9 风险项），靠分核（56 个 vector core）把墙钟压到 ~2.5µs（最大 workload 估算）。

**R3 前缀和重设计（源：warp shuffle 树 → NPU：UB 内 Hillis–Steele 向量扫描）**

- 源方案：5 级 `shuffle_up` 完成 warp 内包含式扫描，再由 `tx==0` 串行合并 `num_warps` 个 warp 和。
- NPU 新算法：`s_pad[E_pad]`（补齐值）在 UB 内做 `ceil(log2 E) = 9` 步包含式扫描
  `s[e] += (e ≥ 2^d) ? s[e-2^d] : 0`（左 halo 补零，消除条件选择），再左移一位得独占前缀；
  `cumsum[E]` 即补齐总数。E ≤ 384 全部驻留一份 UB 缓冲，**无 chunk carry**。
- 语义保持论证：整数加法结合律 ⇒ 任意结合顺序结果相同；halo 区恒为 0 ⇒ 与带边界条件的扫描逐位一致。
  机器验证：verify_equiv.py `A1+A3 cumsum (shift pad + scan)` 3,240 例全部 EQUIV_PASS（§1.6.1）。

**R4 `expert_ids` 重设计（源：expert 主序变长 scatter → NPU：rank-count 恒等式）**

- 源方案：`for e: e_start=cumsum[e]//B; for b in serial(cumsum[e+1]//B - e_start): expert_ids[e_start+b]=e`
  ——写数最少（= 有效 block 数），但 trip count 数据相关、专家间串行。
- NPU 新算法：对输出 block 下标 `b` **并行**求
  `owner(b) = #{e ∈ [0,E) : cumsum[e] ≤ b·B} − 1`，`b·B ≥ cumsum[E]` 时写 `num_experts`。
  等价性证明（含零计数 expert）：因 `cumsum` 非减且区间 `[cumsum[e], cumsum[e+1])` 互不相交地铺满
  `[0, cumsum[E])`，故对 `pos = b·B < cumsum[E]` 恰有一个 e 满足 `cumsum[e] ≤ pos < cumsum[e+1]`；
  又 `cumsum[0]=0 ≤ pos`，所以 `#{e: cumsum[e] ≤ pos} − 1` 就是该 e 的**最大**解（零长度区间对应的 e
  会被后续相等值覆盖，与源 scatter 的「空区间不写」行为一致）。`pos ≥ cumsum[E]` 时源写 `num_experts`，
  新式同判。机器验证：verify_equiv.py `A2 expert_ids (rank-count)` 3,240 例（含 E=1、B=1、
  连续零计数 expert、单 expert 独占）全部 EQUIV_PASS。
- 语义保持论证（数据相关 trip count 消失）：新式对每个 b 独立、无分支（除末尾 mask），可向量化/分核；
  代价由「= 有效 block 数」变为 `E · max_num_blocks` 次比较（最大 342K lane-ops，摊到 56 核 ≈ 95 条向量指令），
  墙钟更短（§1.6.0 R3）。

**R5 同步重设计（源：`T.sync_threads()` → NPU：无显式块内同步 + 无跨核依赖）**

- 源方案：`sync_threads()` 分隔「清零 / 原子计数 / 扫描 / 写回」各阶段。
- NPU 新算法：单核内 UB 写后读依赖交由 ascend 默认流水（AutoSchedule + InsertSync）处理；
  本设计**不构造任何跨核数据依赖**（R1 用冗余读替代跨核等待），因此不需要跨核 flag/barrier。
- 语义保持论证：依赖图与源一致（计数 → 补齐/扫描 → 写回），只是把「跨线程同步」换成「核内流水同步」+
  「跨核解耦」。风险：若 Stage 3 发现某处需显式 `T.ascend_pipe_barrier("PIPE_ALL")`，按 §7.3 说明补点位。

**源优化手段意图承接核对**（不得静默丢弃）：O1→UB 驻留（D3）；O2→R2 比较累加（O(numel·E/V) 取代原子，
原子意图「一趟完成」保留为单遍扫描）；O3→R3 log 深度扫描（意图保留）；O4→R1 分核并发（意图保留）；
O5→舍弃（UB 全量驻留，理由见 D7）；O6→R4 rank-count（意图「少写」让位于「可并行/可向量化」，
并给出量化对比）；O7→保留；O8/O9→D1/D2 保留为算法骨架。

### 0.7 标杆实现

- 源算子：`/workspace/TileOPs/src/tileops/_kernels/moe/permute_align.py` 中的 `_make_align_kernel`（GPU TileLang）。
- 上游算法来源：sglang/sgl-kernel `csrc/moe/moe_align_kernel.cu`（源码 docstring 标注），
  vLLM `moe_align_block_size` 为同语义公开实现（GPU 侧 `ref_api` 原本指它，NPU 侧不可用）。
- NPU 侧参考实现（golden 依据，独立于本设计）：`examples/TileOPs/tileops/workloads/moe.py`
  的 `ref_permute_align`（`bincount` + `cumsum` + 二分查找），其纯 Python 循环版本是 §8.1 golden 的移植源。
## 1. 概述

### 1.1 算子名称

`_make_align_kernel`（项目 `moe_permute_align`；产品语义：MoE token→expert 对齐段 / `moe_align_block_size` 前半）。

### 1.2 功能描述

把 `T×K` 条路由（`topk_ids`）整理成 MoE grouped GEMM 需要的三段索引信息，本段（align）负责：

1. 直方图：统计每个 expert 的路由数 `cnt_e`；
2. 补齐：每个 expert 的计数向上取整到 `block_size` 的整数倍，并做**独占前缀和** `cumsum[0..E]`；
3. 输出补齐后总长 `num_tokens_post_pad = cumsum[E]`；
4. 输出 block→expert 映射 `expert_ids`（越过补齐总长的 block 写 `num_experts`）；
5. 把 `sorted_token_ids` 全部槽位写成哨兵 `numel`（路由索引由随后的 scatter 段填入各区段前部）。

本段**不写任何路由索引**，因此不读第二遍 `flat`、不做 compaction（§0.4 O8）。
适用场景：MoE 全流程（decode/prefill 的 token 分发、grouped GEMM 的索引准备）。
性能口径：`flops = 0`（纯整型索引重排），按访存/时延评估，不用 MFU。

### 1.3 数学公式

给定 `numel`、`E = num_experts`、`B = block_size`，`flat ∈ [0,E)^numel`：

基线（源算子，逐字对应 §0.3）：

```
cnt_e    = Σ_{i<numel} [ flat_i == e ]                              (e = 0..E-1)
p_e      = ceil(cnt_e / B) · B
cumsum_0 = 0 ,  cumsum_{e+1} = cumsum_e + p_e                       (独占前缀和)
total    = cumsum_E                                                 → num_tokens_post_pad[0]
out_s[i] = numel            for i = 0..P-1                          (哨兵)
owner(b) = e  s.t. cumsum_e ≤ b·B < cumsum_{e+1}   (b·B < total)     → expert_ids[b]
owner(b) = E                (b·B ≥ total)
P = numel + (E+1)(B-1) ,  nblk = ceil(P / B)
```

本设计采用的等价改写（论证与机器验证见 §1.6.1；A1–A4 为采纳项）：

```
A1  p_e  = ((cnt_e + (B-1)) >> p) << p          ，B = 2^p 时（否则用 div/mul）
A2  owner(b) = (#{e ∈ [0,E) : cumsum_e ≤ b·B}) − 1  (b·B < total) ；否则 E
A3  cumsum_e = (inclusive_scan(p))_e − p_e      （Hillis–Steele，log2 E 步）
A4  cnt_e    = Σ_{i} [ flat_i == e ]            （expert-lane 比较累加，逐位同 A 定义式）
```

### 1.4 算法描述

**选定算法族**（§1.6.0 调研结论）：**直方图 + 补齐前缀和 + rank-count 映射**的单遍索引路线，
分两条规模路径（§5）：

- **路径 S（`numel` 小，估计核内指令 ≤ 8192）**：1 次 launch，`grid = 2`。
  核 0：搬入全部 `flat` → expert-lane 比较累加得局部直方图 → 补齐（A1）→ HS 扫描（A3）→
  写 `cumsum`/`num_tokens_post_pad` → 用 A2 计算本核分片的 `expert_ids` → 填充本核分片的哨兵；
  核 1：填充另一半哨兵。与源算子 `bx==0 / bx==1` 的分工同构。
- **路径 L（`numel` 大）**：2 次 launch。
  launch-1（`grid = G_h`）：每核取 `tokens_per_core` 个 token，expert-lane 累加得局部直方图，
  写 GM `partial[g, 0..E_pad)`（**纯 store、无原子、无需初始化**；空核写全零行）。
  launch-2（`grid = G_a`）：每核读全量 `partial` 并**冗余**求和（整数确定性 ⇒ 各核得到逐位相同的 `cnt`），
  补齐 + HS 扫描得本核的 `cumsum` 副本；核 0 写 GM `cumsum` 与 `num_tokens_post_pad`；
  各核按 block 下标 / 填充下标分片写 `expert_ids` 与哨兵。
  **跨核数据依赖被「冗余读只读数据」消除**（不依赖任何未认证的跨核 barrier，§0.5/§0.6 R1/R5）。

与源算法的结构差异（一句话）：源的「单 kernel 共享内存原子直方图 + warp shuffle 扫描 + 变长 scatter」
被换成「多核比较累加直方图 + 局部和 GM 归并 + 向量扫描 + rank-count 映射」，
原因见 §0.5/§0.6，量化依据见 §1.6.0。

### 1.5 数据流图

```
路径 S（1 launch, grid=2）
  GM flat[numel] ──T.copy(GM→UB)──> UB s_x[CHUNK] ──(expert-lane 比较累加)──> UB s_cnt[E_pad]
     │                                                                          │
     │                                                        A1 补齐 ──> UB s_pad[E_pad]
     │                                                                          │
     │                                                        A3 HS 扫描 ─> UB s_cs[E_pad]
     │                                                                          │
     │                          ┌── 核0 store ──> GM cumsum[E+1] / num_tokens_post_pad[1]
     │                          ├── 核0 A2 rank-count ──> UB s_ids ──T.copy──> GM expert_ids 分片
     └─────────────────────────┴── 核0/核1 常量 UB s_fill ──T.copy──> GM sorted_token_ids 分片（哨兵）

路径 L（2 launches）
  launch-1  grid=G_h : GM flat 分片 ─> UB s_x ─> UB s_hist[E_pad] ─T.copy──> GM partial[G_h, E_pad]
  launch-2  grid=G_a : GM partial 全量 ─> UB 逐行累加 ─> s_cnt ─> A1 ─> A3 ─> 本核 cumsum 副本
                       核0 ──> GM cumsum / num_tokens_post_pad
                       各核 ──> GM expert_ids 分片（A2）、GM sorted_token_ids 分片（哨兵常量）
```

不使用 L1 / L0A / L0B / L0C（无矩阵运算）。GM↔UB 搬运全部经 `T.copy`（MTE2/MTE3）。

### 1.6 算法调研与优化分析 ⭐

#### 1.6.0 算法调研（Algorithm Research）⭐

调研对象：**同一数学语义（§1.3 基线公式）的算法族**——所有能算出同样 `cumsum`/`expert_ids`/哨兵/总长的
算法结构。源算子（GPU 实现）只作为**基线候选**参与比较。调研深度：**完整调研**
（本算子属「多步 + 规约 + 索引分发」类）。参考表命中情况：平台 `knowledge.algorithm_candidates` 与
`kb_search "MoE permute align histogram prefix-sum scatter int32"` 均无命中条目（任务上下文已预注入：
结果为空），故候选集合由源算子证据（§0.3/§0.4）、本仓 `examples/` 检索结果与结构判据构成，并在下方
逐条给出本地证据。

**R1 等价化简公式（候选表，含基线）**

| # | 候选（公式 / 结构） | 等价性初判 | 收益方向 | 本地证据 |
|---|---|---|---|---|
| C1 | **基线**：shared 原子直方图 + warp shuffle 扫描 + expert 主序变长 scatter（源算子） | —（基线） | — | 源算子 §0.3；`examples/TileOPs/tileops/kernels/moe/moe_permute_align/_moe_permute_align_fwd_kernels.py` |
| C2 | 直方图改 **expert-lane 比较累加**（A4），前缀改 **Hillis–Steele 向量扫描**（A3），`expert_ids` 改 **rank-count 恒等式**（A2） | 数学恒等（整数，逐位相同；§1.6.1 已机器验证 3,240 例） | 消掉「UB 原子 / warp 交换 / 数据相关 trip count」三项 NPU 不可用能力；全部步骤可向量化、可跨核并行 | 平台 API 事实见 §0.5；`numerical-boundaries.md` PL-950-simd-reduce（向量归约）、`PL-950-row-primitives.md`（SIMT lane 并行） |
| C3 | 直方图用 **GM 原子**合并（局部直方图 `T.set_atomic("add","int32")`） | 恒等（整数原子加） | 省掉多核冗余读（§1.6.3 M2） | `execution-patterns.md` PL-950-atomic-splitk（仅 FP32 已验证，int32 未取证） |
| C4 | **排序 + 游程计数**（按 expert 排序 `flat` 后数 run）替代直方图 | 恒等（排序不变量） | 复杂度 O(N log N)，不比 C2 的 `N·E/V` 低（N=32768,E=384,V=64：491K vs 196K 指令） | 本工具链无已验证 sort API（`tilelang.ascend.language` 无 sort 导出；`reduce_*`/`cumsum` 无排序能力）→ 淘汰 |
| C5 | `expert_ids` 用 **二分查找**（log E 次比较/block）替代 rank-count | 恒等（同 `owner` 定义） | 比较数 `nblk·log E` < `nblk·E`；但每步是数据相关的 gather（下标依赖上一轮结果），不可 lane 并行 | 源仓参考实现即二分（`examples/TileOPs/tileops/workloads/moe.py` 中的 `ref_permute_align`）；本设计**保留给 golden**，不用作 kernel（理由：gather 串行 9 步 × nblk，抹掉向量化收益） |
| C6 | 前缀和改用串行扫描（单 lane 累加 E 次） | 恒等 | 无（E 次标量 UB 读写） | 仅作为 A3 的退化对照；被 A3 淘汰（§1.6.2 第 4 项） |
| C7 | 把 align 与 scatter **融合**成单 kernel（源 `_make_small_batch_kernel` 路线） | 恒等（输出契约相同） | 少一次 launch | 仅 `numel<1024 且 E≤32` 才成立（源实现），本任务 E≥128 → 不适用；且属另一 kernel 工厂的设计范围 |

**R2 在线算法（单遍 / O(tile) 中间态）**

- 结论：**有**，且本设计已采纳「单遍」形态。
- 结构依据：`cnt_e` 是可分解为 running 统计量（计数和）的归约 → 存在在线变体；源算子本身就是单遍
  （`flat` 只读一次，§0.3 步骤 3 即"在线累加"的直接证据）。`cumsum` 是 E 长度的一维前缀，
  天然是「先局部后合并」的两段式（本设计 = 局部直方图 + 冗余合并，中间态 O(E) 而非 O(numel)）。
- 收益口径：`flat` 扫描遍数 = 1（与基线同）；中间缓冲 O(E) = 1.5KB（基线 SMEM ≈ 7KB，因 `s_vals[threads]`）。
- 不存在「必须两遍」的结构理由：`expert_ids` 与 `cumsum` 的消费方（grouped GEMM / scatter）在 kernel 之外。

**R3 复杂度对比（四口径 + 附加维度）**

代表 workload：最大档 `numel=32768, E=384, B=64`（`max_padded=57023`，`nblk=891`）。
口径说明：FLOPs 记整数比较/加法条数（lane-ops，除以向量宽度 V 才是指令数）；访存只算 GM
（UB 内不计）；「扫描遍数」指 `flat` 的 GM 读次数；「中间缓冲峰值」指片上常驻量（不含 GM scratch，
GM scratch 单列）。

| 候选 | FLOPs（lane-ops） | GM 访存（B） | flat 扫描遍数 | 片上中间缓冲 | 可并行度 | 跨核归并代价 |
|---|---|---|---|---|---|---|
| C1 基线（源，单 CTA 计数） | `numel`（原子增量）+ `E·log32`≈2K + `nblk` | 128K + 228K + 5K ≈ **361K** | 1 | ≈7KB SMEM | **1 CTA**（计数/扫描段） | 无（单 CTA 内完成） |
| C2 本设计（主选） | `numel·ceil(E/V)` = 196K（V=64） | 361K + 写 `G_h·E·4`=86K + 冗余读 `G_a·G_h·E·4`≈2.4M（L2 命中） | 1 | ≈30KB UB | **56 vector core** | 无同步；冗余读同一份只读数据 |
| C3 主选 + GM 原子合并 | 同 C2 | 361K + 原子写 86K（无冗余读） | 1 | ≈30KB UB | 56 core | 无同步，但需 `counts` 每次调用前清零（额外契约） |
| C4 排序路线 | `N log N` ≈ 491K + `E` | 361K + 排序中间态（>1M） | 1（+排序多趟） | 排序缓冲 ≫ UB | 取决于 sort | 高 | 
| C5 二分 `expert_ids` | `nblk·log2 E` ≈ 8K（串行 gather） | 361K | 1 | 同上 | 仅 block 维 | 无 |

**R4 硬件亲和（逐候选评估 + 设计期 roofline 定量）**

| 候选 | 计算单元匹配 | 片上容量 | 对齐/整除 | 静态边界 | 流水可融合 | 跨核结构 | 结论 |
|---|---|---|---|---|---|---|---|
| C1 | ✗ 依赖 SMEM 原子加与 warp shuffle：NPU 无 UB scatter-add、无 warp（§0.5 证据） | ✓ | ✓ | ✓ | ✓ | 单 CTA 计数段无法利用 56 核 | **淘汰（能力缺失）** |
| C2 | ✓ 全部为 lane 并行比较/加/扫，落在 vector core；无矩阵段 | ✓ 30KB ≪ 248KB | ✓ `E_pad`、`CHUNK`、`FILL_TILE` 均 64 对齐 | ✓ 全部上界为特化常量（仅 `numel=0` 需守卫） | ✓ `T.Pipelined` 覆盖 GM→UB | ✓ 冗余读替代同步 | **选定** |
| C3 | ✓ 同 C2，另需 int32 GM 原子写回 | ✓ | ✓ | ✓ | ✓ | ✓ | **备选（int32 原子未取证）** |
| C4 | ✗ 无 sort API | ✗ 排序缓冲超 UB | — | — | — | — | **淘汰（API 缺失）** |
| C5 | ~ 仅 block 维并行，内核是 9 步数据相关 gather | ✓ | ✓ | ✓ | ✓ | ✓ | **弃选（收益低于 C2，用于 golden）** |

**设计期 roofline（D-2，代表 workload = 最大档 `numel=32768, E=384, B=64`）**

- 流量项：`362,732 B`（读 `flat` 128KB + 写哨兵 228KB + `cumsum`/`expert_ids` 5KB）÷
  `CONST-fabric-bw`（≥1.73 TB/s 聚合，**910B2C 实测值，跨型号外推至 950PR，未实测**）= **0.21 µs**；
  路径 L 另加 partial 写 86KB + 冗余读 ≈2.4MB（L2 驻留，按 `CONST-mte2-degradation` 的 L2 口径
  43.1 GB/s/core × 56 核，910B2C 实测）≈ **1.0 µs**。
- 发射项：每核 `tokens_per_core·ceil(E/V)·2`（V=64）≈ 3,500 条 + launch-2 ≈1,300 条 ≈ **4,800 条向量指令**；
  按「1 条向量指令/cycle、1.4 GHz」（**未实测假设，估算依据**：CCE 向量单元每 cycle 单发射，
  950PR 主频平台记录未给出）⇒ **3.4 µs**。
- 容量项：UB 峰值 ≈30KB / 248KB（平台配置 `ascend-950pr-9579` UB=253952B）⇒ **0.12**，非瓶颈。
- 容量/带宽淘汰依据：C4 的排序缓冲超 UB（>248KB，见 `ascend-950pr-9579` 容量预算）；
  C1 的 SMEM 原子/warp 能力在本工具链不存在（§0.5 证据链）。
- **估算下界: 3.4 µs（流量 0.21 + 冗余读 1.0；发射 3.4；容量 0.12；取 max = 3.4 µs，引用 CONST-fabric-bw /
  CONST-mte2-degradation〔910B2C 外推〕+ 向量发射未实测假设）**；加 2 次 kernel launch（常数未实测，
  按每 launch ≈2 µs 估）后 ≈7 µs。小档（`numel=8`，路径 S，1 次 launch）估算 ≈2.5–4 µs，
  此时**发射项与 launch 开销主导，流量项仅 0.06 µs**——与 §9.1「小档 latency-bound」判断一致。

**调研结论**

选定 **C2**（直方图 + 补齐前缀 + rank-count 的单遍索引路线，多核比较累加 + 局部和归并 + 向量扫描）：
R1 中它与基线数学恒等且消掉全部 NPU 缺失能力；R2 单遍形态成立（`flat` 只读一次、中间态 O(E)）；
R3 其 lane-ops（196K）与基线同量级，而基线受「单 CTA + 原子争用」限制无法利用 56 核，
C2 的墙钟下界（发射项 3.4 µs）显著优于基线在 NPU 上的可行性（基线有两项能力根本缺失）；
R4 逐项检查唯一存活并被选。**「无更优替代」的调研范围**：已查平台 algorithm_candidates（无命中）/
`kb_search`（无命中）/ 本仓 `examples/` 同类实现（`examples/TileOPs/tileops/kernels/moe/` 仅本算子族、
无 index/histogram 类参照）/ 结构判据（排序类 C4 因 API 缺失与复杂度不占优淘汰、二分 C5 因 gather 串行
淘汰、GM 原子 C3 因 int32 未取证降为备选）。因此「不存在比 C2 更优且在本工具链可实现的算法族」
这一判断的边界是上述四项检索 + 两项结构判据，不含未检索的算法族。
源算子已识别优化手段的承接：见 §0.6 末尾核对表（O1–O9 逐项）。

#### 1.6.1 数学等价优化（公式级）

分析对象 = §1.6.0 选定的 C2（经 §0.6 重设计落地）。逐项四要素如下；等价性**全部机器验证**
（`verify_equiv.py`，torch CPU，int64/fp64 精确参照，3,240 例 = 648 组 `(numel,E,B)` × 5 种分布，
覆盖 `numel=0`、`E=1`、`B=1`、连续零计数 expert、单 expert 独占、均匀/交替/偏斜分布）。

**A1 补齐由「除法」改「移位」（`B = 2^p` 时）**

- 原式：`p_e = ceil(cnt_e / B)·B`（整数除法 + 乘）
- 优化后：`p_e = ((cnt_e + (B-1)) >> p) << p`，`p = log2 B`（`B` 非 2 的幂时回退 div/mul，构建期判定）
- 等价性：`cnt ≥ 0` 时 `ceil(cnt/B) = (cnt + B - 1) >> p`，二进制右移即下取整除法；随后左移乘回。
  整数恒等，无舍入（非负域，无符号语义歧义）。
- 收益：每 expert 由「1 次整数除法 + 1 次乘」降为「1 加 + 2 移位」。整除在向量单元上无硬件指令、
  须多指令模拟（**未实测假设**，估算依据：CCE 无整数除法指令；Stage 4 实测校核）。E=384 时节省
  384 次除法/次调用（核内一次性，绝对量小，但去掉长延迟指令链有助发射项）。

**A2 `expert_ids` 由「expert 主序变长 scatter」改「rank-count 恒等式」**

- 原式：`for e: for b in [0, cumsum_{e+1}/B − cumsum_e/B): expert_ids[cumsum_e/B + b] = e`
- 优化后：`expert_ids[b] = (#{e ∈ [0,E): cumsum_e ≤ b·B}) − 1`（`b·B < cumsum_E`），否则 `num_experts`
- 等价性：`cumsum` 非减、区间 `[cumsum_e, cumsum_{e+1})` 互不相交且铺满 `[0, cumsum_E)`；`cumsum_0 = 0 ≤ pos`
  ⇒ 满足 `cumsum_e ≤ pos` 的 e 集合是前缀 `{0..k}`，`k+1 = 计数`；`pos < cumsum_E` 时该 k 恰是包含 pos 的
  区间右端（零长度区间不改变「最大满足者」），与源 scatter 的「空区间不写」一致；`pos ≥ cumsum_E` 时源写
  `num_experts`，新式同判。**零计数 expert 情形已被 verify_equiv 显式覆盖**（`same`/`single_first` 分布）。
- 收益：消掉数据相关 trip count（源内层迭代数 = 有效 block 数），每个 b 独立、无分支 → 可 lane 并行、
  可跨核分片；比较数上限 `E·nblk`（342K lane-ops ≈ 95 条向量指令/核，56 核）——**墙钟**优于源串行
  变长写（nblk=891 次串行迭代）。**代价换墙钟**，量化见 R3 表。

**A3 独占前缀由「串行累加」改「Hillis–Steele 包含式扫描 + 左移一位」**

- 原式：`cumsum_{e+1} = cumsum_e + p_e`（串行链，E 步，每次 UB 读改写）
- 优化后：`s ← p`；对 `d = 0..ceil(log2 E)−1`：`s_e ← s_e + (e ≥ 2^d ? s_{e−2^d} : 0)`；然后 `cumsum_{e+1} = s_e`，
  `cumsum_0 = 0`（左 halo 区预置 0，消掉条件选择）
- 等价性：整数加法结合律/交换律 ⇒ 任意结合顺序结果相同（无浮点舍入），包含式扫描末值 = 总和，
  左移一位 = 独占前缀。
- 收益：1 次调用由 E=384 步串行（每步 UB 读+加+写）降为 `ceil(log2 384)=9` 步向量操作（每步 6 条 int32
  向量指令）≈ 54 条指令，**7 倍以上指令削减且延迟链由 O(E) 降到 O(log E)**。

**A4 计数由「原子 scatter-add」改「expert-lane 比较累加」**

- 原式：`cnt_e = Σ_i [flat_i == e]`（源用 `atomic_add` 实现该定义式）
- 优化后：同样按定义式求和，但形式为「lane ↔ e，逐 token 广播比较后累加」（§0.6 R2）
- 等价性：同一个定义式的两种求值顺序，整数计数逐位相同（与原子加法结果一致）；无饱和/溢出
  （`cnt ≤ numel ≤ 32768 < 2^31`）。
- 收益：消掉 NPU 缺失的 UB 原子能力；把随机字增量换成规整向量比较（`numel·ceil(E/V)` lane-ops），
  代价是一维访存放大（`flat` 被读 E/V 次**驻留 UB 内**，不产生 GM 流量）。

**等价性机器验证结果（`verify_equiv.py` 执行输出，2026-11-23，torch CPU）**

| 采纳项 / 检查 | 用例数 | 最大绝对差 | 最大 ulp 差 | 违反率 | 结论 |
|---|---|---|---|---|---|
| A1+A3 `cumsum`（移位补齐 + HS 扫描）vs 基线串行式 | 3,240 | 0 | 0 | 0.000000 | **EQUIV_PASS** |
| A1+A3 `cumsum` vs 第三方参考（`cumsum` 原语） | 3,240 | 0 | 0 | 0.000000 | **EQUIV_PASS** |
| A2 `expert_ids`（rank-count）vs 基线变长 scatter | 3,240 | 0 | 0 | 0.000000 | **EQUIV_PASS** |
| A2 `expert_ids` vs 第三方参考（二分/searchsorted） | 3,240 | 0 | 0 | 0.000000 | **EQUIV_PASS** |
| A2 + 哨兵填充 `sorted_token_ids` vs 基线 | 3,240 | 0 | 0 | 0.000000 | **EQUIV_PASS** |
| `num_tokens_post_pad` 总长一致 | 3,240 | 0 | 0 | 0.000000 | **EQUIV_PASS** |
| 基线 vs 第三方参考（基线自检） | 3,240 | 0 | 0 | 0.000000 | **EQUIV_PASS** |

脚本：`examples/moe_permute_align/_make_align_kernel/verify_equiv.py`（随设计留存，Stage 2 可重跑；
退出码 0 = 全 PASS）。所有量在 int64/fp64 中精确可表示（< 2^20），故「0 ulp 差」是精确结论而非容差结论。

**优化结论**：采纳 A1、A2、A3、A4 四项（全部 EQUIV_PASS，整数逐位等价，无精度风险）。
未采纳项：C3 GM 原子合并（int32 原子未取证，降为 §1.6.3 备选 M2）；
`num_tokens_post_pad` 与 `cumsum[E]` 的双写（源两处写同一值）保留为一次写 + 一次 GM store，无可化简空间。
**无进一步公式化简空间**：算子本体是「计数 + 前缀 + 一次映射」，A1–A4 已把除法、数据相关迭代、
串行链、原子四处高代价结构替换完毕；剩余工作量由 §1.6.2（向量化）与 §5（分核）优化。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

盘点范围：路径 S 与路径 L 的实现方案中**全部**循环与标量计算点。

| # | 计算点 | 位置 | 处理 | 替代 API / 依据 | 判定 |
|---|---|---|---|---|---|
| V1 | 哨兵填充（源 `for i in serial(ceildiv(max_padded,threads))` + `if idx<max_padded`） | 两路径，核内分片 | **向量化替代**：UB 常量 tile `T.fill(s_fill, numel)` 一次，再 `T.copy(s_fill, GM 分片)` 批量 store；尾片用显式 region `T.copy(s_fill[0:tail], GM[...])` | `T.fill`、`T.copy`（`ascend/language/copy_op.py`；尾块 region 用法见本平台 `kernel_template.py` 的 `T.copy(X[...:n], x[:tail])`） | 已替代（逐元素条件写 → 定长块 store） |
| V2 | 直方图累加（源 `for i in serial: atomic_add`） | 两路径 | **向量化替代**：`for e in T.Parallel(E_pad)` 映射 lane，内层 `for i in serial(len)` 为**每 lane 的私有归约轴**；每迭代 1 条广播比较 + 1 条累加（向量宽度 = 64 int32 lane） | `T.SimtVF` + `T.Parallel`（`PL-950-row-primitives.md` 已验证 lane 并行 + 每 lane 私有统计量）；`T.if_then_else` 生成 mask-add | 已替代（scatter-add → lane 并行归约） |
| V3 | 内层 `for i in serial(len)`（V2 的归约轴） | 两路径 | **保留为串行轴**，理由：(a) 该轴是**每 lane 的归约（scan）轴**，与 lane 轴正交，向量化发生在 lane 方向（一次迭代已覆盖 E 个 expert）；(b) 若改为 token 并行则需要对 `s_cnt[eid]` 做 scatter-add，**本工具链无 UB scatter-add / 向量 scatter 累加 API**（`atomic_add` 只在 GM 侧有已验证路径 `set_atomic`+`copy`，PL-950-atomic-splitk；`T.simd` 导出的 `vld/vsts/vadd/vcvt` 等无 scatter 累加）；(c) 结构判据：`cnt_e` 的求和轴与输出索引轴必须是同一张量的不同轴，二者只能一个占用 lane | 依据同上 + §0.5 原子证据链 | 保留（API 缺失 + 归约轴正交） |
| V4 | 补齐 `p_e = ceil(cnt_e/B)·B`（源逐 expert 标量） | 两路径 | **向量化替代**：`for e in T.Parallel(E_pad)` 一次向量化（A1 移位形式） | `T.Parallel` + 移位算子 | 已替代 |
| V5 | 前缀和（源 warp shuffle 树 + `tx==0` 串行合并） | 两路径 | **向量化替代**：A3 的 9 步 `for e in T.Parallel(E_pad)` 扫描（左右 buffer 轮换 + 左 halo 补零） | `T.Parallel`；`numerical-boundaries.md` PL-950-simd-reduce 已验证显式向量归约可用（本设计的扫描不做跨 lane 归约，只做相邻 lane 位移加） | 已替代 |
| V6 | `expert_ids` 映射（源 `for c in serial(num_chunks): for b in serial(e_blocks)`） | 两路径 | **向量化替代**：A2 rank-count —— `for b in T.Parallel(BLK)`（lane = 输出 block），内层 `for e in serial(E_pad)` 为每 lane 的归约轴（同 V3 理由，且不涉 scatter） | `T.Parallel` + `T.if_then_else` 掩码 | 已替代（数据相关 trip count 消失） |
| V7 | 哨兵越界 block（源 `if blk >= total//B and blk < max_num_blocks: expert_ids[blk]=E`） | 两路径 | **并入 V6**：A2 的末尾 mask 直接产出 `num_experts`，不再单独一遍扫描 | 同 V6 | 已替代（消掉一次独立扫描） |
| V8 | `num_tokens_post_pad[0]`、`cumsum[0]`、`cumsum[E]` 写入 | 两路径 | **保留为标量写**，理由：单元素写，属 **block 级索引 / 元数据计算**（每核 ≤3 次写，非逐元素热点）；且 `cumsum[E]` 必须与扫描末元素一致，由同一 lane 写可省一次归约 | `T.copy` 单元素 region 或 lane 0 直写 | 保留（block 级，非热点） |
| V9 | 核/分片下标计算（`base = g·tokens_per_core`、`blk0 = a·BLK`、`tail` 等） | 两路径 | **保留为标量**，理由：host/kernel 入口的**任务映射计算**，每核常数次，不随数据量增长；上界全部为构建期常量 | — | 保留（任务映射） |
| V10 | host 侧 `max_padded`/`max_num_blocks`/路径判定 `G_h`/`G_a` | host | **保留**，理由：**host 侧 shape/元数据计算**，不在 kernel 内，且是构建期常量折叠的输入 | — | 保留（host 元数据） |
| V11 | 路径 S/L 的路径判定 | 构建期 | **保留为 Python 分支**，理由：构建期常量判定（非运行期分支），进 kernel 会引入无用代码与寄存器压力 | — | 保留（构建期） |
| V12 | `numel = 0` 守卫（`if numel > 0`） | 两路径 | **保留**，理由：**依赖动态数量的边界处理**；`numel` 虽为构建期常量，但 0 与非 0 决定循环体是否存在，必须保留条件（构建期折叠，运行时无分支） | — | 保留（边界） |

**向量化结论**：实现方案中 12 个循环/标量计算点中，V1/V2/V4/V5/V6/V7 共 6 处由逐元素条件写、
scatter-add、串行扫描、变长 scatter 替换为 lane 并行向量操作；V3 是 lane 归约轴的串行扫描轴，
保留理由为「本工具链无 UB scatter-add（API 缺失）+ 归约轴与 lane 轴正交」；
V8–V12 为 block 级元数据 / host 元数据 / 构建期与边界条件，共 5 处，理由逐项给出。
**不残留任何逐元素标量循环**：所有随数据量增长的循环都在 lane 方向向量化。

#### 1.6.3 向量化轴与数据布局决策 ⭐

**候选枚举**（按算子类别：索引 / 直方图 / scatter 类；同时含「broadcast 填充类」的等价选择）：

| 候选 | 结构 | 向量轴 | lane 语义 | 判据 |
|---|---|---|---|---|
| **L1（主选）** | 直方图：`Parallel(E_pad)` × `serial(tokens)`（垂直扫描）；`expert_ids`：`Parallel(BLK)` × `serial(E_pad)` | expert 轴 / block 轴 | lane ↔ expert；lane ↔ 输出 block | 每 lane 私有累加器，**无需跨 lane 归约、无需原子、无需 GM scatter**；E ≤ 384 ⇒ `E_pad` 取 384/512，尾 lane 浪费率 `(E_pad−E)/E_pad`：E=384→0%，E=256→0%（E_pad=256），E=128→0%（E_pad=128） |
| L2（备选 B1） | 直方图：token tile × expert tile 的 2D 比较 + 沿 token 轴归约（`T.reduce_sum(dim=0)` 或 SIMT reducer） | token 轴 | lane ↔ token | 需要二维归约；本工具链高层 2D shared→shared 归约存在已验证的**输出布局差异**（`numerical-boundaries.md` TRAP-950-simd-shared-reduce），须改用 `alloc_reducer/reducer_update` 或显式 `vcadd`，实现风险与指令数（每 expert tile 一次归约）均升高 |
| L3 | token 并行 + 对 `s_cnt[eid]` scatter | token 轴 | lane ↔ token | 需要 UB scatter-add：**本工具链无此 API**（§1.6.2 V3 证据链）→ 淘汰 |
| L4（合并策略 M1，主选） | GM `partial[G_h, E_pad]` + launch-2 各核冗余求和 | E 轴 | lane ↔ expert | 无原子、无初始化契约、无跨核 barrier；代价 = 冗余读（L2 驻留） |
| L5（合并策略 M2，备选） | GM 原子 `set_atomic("add","int32")` 累加到 `counts[E]` | E 轴 | lane ↔ expert | 省冗余读，但 int32 原子未取证 + 每次调用前须清零（`counts` 初始化契约） |
| L6 | 哨兵填充：UB 常量 tile → GM 批量 `T.copy` | 连续下标轴（64 int32/lane） | lane ↔ 连续输出下标 | 尾块用 region 裁剪；无条件写 |

**逐候选评分（主选 vs 备选）**

- 指令数（最大档）：L1 ≈ 3,500 条/核（直方图）+ 1,300 条（launch-2）；L2 ≈ `numel·ceil(E/V)/核` 同量级
  但额外 9 次二维归约的跨 lane 代价与 layout 风险；L3 不可实现。
- 尾 lane 浪费率：L1 在三条 workload（E=384/256/128）上均为 0%；`E_pad = align_up(E, 64)`。
- UB 预算：≤30KB（§4.5），`CHUNK=1024` 时 2 版本 `s_x` 共 8KB；膨胀系数即使按 1.7x 也不过 ~51KB。
- 弃选量化理由：L2 未比 L1 少指令（同 `N·E/V` 量级）却引入已验证的 2D 归约缺陷风险（TRAP-950-simd-shared-reduce）；
  L3 API 缺失；L5 移除 ~1.0 µs 冗余读但要付出未取证能力 + 清零契约（**判定裕度依赖未实证常数**：
  int32 GM 原子可行性 + 冗余读真实代价）→ 走**实验裁决模式**（下）。

**实验裁决模式（主选 + 备选 + 裁决计划）**

- **主选**：L1 + L4 + L6（进 Stage 3）：一维 core grid + `T.SimtVF(threads=E_pad)` lane=expert 垂直扫描 +
  partial GM 归并 + 常量 tile 填充。
- **备选（≤1）**：L1 + **L5**（GM 原子合并）+ **比较 lane 用 uint16**（`E ≤ 65535`，lane 数 128 ⇒ 直方图
  指令数减半）。结构完整度：buffer 形状（`counts[E_pad]` 取代 `partial[G_h,E_pad]`）、循环结构
  （launch-2 改为读 `counts` 一次）、`s_x` 需多一次 `int32→uint16` 转换 pass（`CHUNK/128` 条指令）、
  分核三要素按 `counts` 单缓冲重算（`G_a` 不变，`G_h` 只影响原子争用）、dtype 路径（uint16 比较 + int32 累加）。
  该备选可直接交 Stage 4 实现（`perf_opt/` 下产出，基准不动）。
- **裁决计划**：代表 shape = `(numel=32768,E=384,B=64)`、`(numel=4096,E=128,B=64)`、`(numel=8,E=384,B=64)`；
  指标 = 平台 `msopprof` Task Duration（us）+ 反解未知常数（冗余读有效带宽 = `G_a·G_h·E·4 / Δt`；
  uint16 lane 发射吞吐 >> int32 的比值）；**判定阈值**：备选 latency < 主选 × 0.85 才翻转（全局）；
  若仅在大档胜出，则**按 shape 分派**（`numel ≥ 4096` 走备选、否则主选——本算子是 `lru_cache`/每实例
  构建的 shape 特化工厂，分派可行且已在 §5 路径 S/L 中体现）；裁决结果回写 §1.6.3 并触发设计修订。

**布局决策结论**：主选 **L1（lane ↔ expert 的垂直扫描）+ L4（partial GM 冗余归并）+ L6（常量 tile 填充）**，
向量化轴 = **expert 轴（直方图/补齐/扫描）与输出 block 轴（`expert_ids`）**，核内布局为
`UB s_x[CHUNK] int32`、`UB s_cnt/s_pad/s_cs[E_pad] int32`、`UB s_part[E_pad] int32`、
`UB s_ids[BLK] int32`、`UB s_fill[FILL_TILE] int32`，`CHUNK/BLK/FILL_TILE/E_pad` 均 64 对齐；
备选 L5 + uint16 lane 交 Stage 4 按上述裁决计划 A/B，不进入 §3.3 / §4 / §6 的结构描述。
## 2. 编程模式选型

### 2.1 模式结论

**编程模式：SIMD**（平台 `programming_modes.choices` 的 `simd` 键，即 `migration_default`；任务未指定其他模式）。
实现形态：一维 `T.Kernel(G)` core grid + 核内 `T.SimtVF(threads=E_pad)` 的 lane 并行表达式
（lane 即向量 lane，由 AIV 向量单元执行）；**不使用 Cube 单元、不使用 L1/L0 层级、不使用混合模式**。

### 2.2 选型理由

1. **计算单元匹配**：本算子零矩阵运算（`flops = 0`），全部是整型比较/加法/前缀扫描与索引写回，
   天然属于向量单元（AIV）。落到 Cube 需要把直方图改写成 one-hot 矩阵乘（`N·E` 元素 one-hot 构造
   仍是向量工作，matmul 只承担 O(N·E) 归约），徒增 L1/L0 搬运与 32B 对齐分形约束，无收益 → 不用 Cube。
2. **混合模式无收益**：混合（`T.Cube()` + `T.Vector()` + `dual_copy`）用于「矩阵 + 逐元素 epilogue」
   （见 `execution-patterns.md` PL-950-mixed-epilogue），本算子无矩阵段可融。
3. **SIMT 与 SIMD 的关系**：平台把「线程域表达式」放在 `T.SimtVF(threads=...)` 内，编译到向量单元执行；
   本设计用 `T.SimtVF` + `T.Parallel` 表达「lane ↔ expert」的并行（`PL-950-row-primitives.md` 已验证该路径的
   单行转换/归约/广播），而不是逐条写 `T.simd.vld/vadd` 寄存器内建。理由：lane 语义（一 lane 一 expert
   的私有计数器）用并行表达式可直接表达，指令选择交给后端；若 Stage 3/4 发现自动向量化未达预期，
   **备选**是用 `T.simd.*`（`vld/vsts/vadd/vcvt`、`pset` 掩码）显式写寄存器级实现（`simd.py` 已导出、
   `numerical-boundaries.md` PL-950-simd-reduce 已验证显式 vld/vcadd 归约）——该备选与 §1.6.3 的
   uint16 lane 备选共用裁决计划。
4. **无跨核协作需求**：本设计刻意避免任何跨核数据依赖（§0.6 R1/R5），因此不需要混合模式/跨核 flag。

### 2.3 模式影响

- 只需 UB 一种片上缓冲（`T.alloc_shared`），内存规划简化为 GM↔UB 双向 `T.copy`（§4.4）。
- 同步策略退化为「核内流水自动同步、核间无依赖」（§7），无需 `ascend_cross_core_*`。
- `T.Kernel` 无 `threads=` 参数（K9）：线程域由 `T.SimtVF` 决定，故 `MoEPermuteAlignKernel.default_config`
  返回空 dict、kernel 工厂不接受 launch 参数（与脚手架 wrapper 现状一致）。

## 3. API 映射设计

### 3.1 公式拆解

以 §1.3 的 A1–A4 改写式为准（§1.6.1 已机器验证），按数据依赖拆成 6 个核内阶段：

| 阶段 | 公式 | 数据规模 | 依赖 |
|---|---|---|---|
| S1 分片装载 | `s_x ← flat[base : base+len]` | `len ≤ CHUNK` | — |
| S2 局部直方图 | `cnt_e = Σ_i [s_x[i] == e]`（A4） | `E_pad` lane × `len` 步 | S1 |
| S3 归并（仅路径 L） | `cnt_e = Σ_g partial[g][e]` | `G_h × E_pad` | 上一 launch |
| S4 补齐 | `p_e = ((cnt_e + B-1) >> p) << p`（A1） | `E_pad` | S2/S3 |
| S5 独占前缀 | A3：9 步 HS 包含式扫描 + 左移一位；`total = cumsum_E` | `E_pad` | S4 |
| S6 写回 | ① `cumsum[0..E]`、`num_tokens_post_pad[0]`；② `expert_ids[b]`（A2，含末尾 mask）；③ 哨兵 `T.fill`+`T.copy` | `E+1` / `BLK` / `max_padded` 分片 | S5（②③ 与 S5 无关者并行） |

### 3.2 TileLang API 映射

| 公式 / 阶段 | TileLang API（`import tilelang.ascend.language as T`） | 关键参数 / 说明 |
|---|---|---|
| 核网格（S/L） | `T.Kernel(G)` | 一维 core grid；**无 `threads=`**（`ascend/language/kernel.py`）；`G` 为构建期常量 |
| lane 并行域 | `T.SimtVF(threads=E_pad)`、`T.SimtVF(threads=BLK)`、`T.SimtVF(threads=FILL_LANES)` | `threads` 取 64 的整数倍（64/128/256/384/512） |
| lane 循环 | `T.Parallel(E_pad)` / `T.Parallel(BLK)` | 一 lane 对应一个 expert / 一个输出 block |
| 归约轴串行 | `T.serial(len)` / `T.serial(G_h)` / `T.serial(E_pad)` | 每 lane 私有累加的扫描轴（§1.6.2 V3） |
| GM→UB / UB→GM | `T.copy(src_region, dst_region)` | 两向均用；支持运行期 region 长度（`kernel_template.py` 的尾块写法） |
| 片上缓冲 | `T.alloc_shared([n], "int32")` | UB（`scope='shared.dyn'`）；无 L1/L0 分配 |
| 常量填充 | `T.fill(s_fill, numel)` | 哨兵 tile 一次填充，再多次 `T.copy` 到 GM 分片 |
| 条件选择 / 掩码 | `T.if_then_else(cond, a, b)` | S2 的 mask-add、S6② 的 `b·B ≥ total → num_experts` |
| 标量变量 | `T.alloc_var("int32")` | `base`、`tail`、`total` 等 block 级标量 |
| 整除 / 上取整 | `T.ceildiv(a, b)`（构建期用 `math.ceil`） | 仅用于下标与分片计算 |
| 流水 | `T.Pipelined(n_chunks, num_stages=2)` | S1↔S2 的 GM→UB 与向量计算重叠（自动多缓冲） |
| 精度/边界 | `T.min`、`T.max` | 尾片长度截断 |

**反查记录（不使用的 API 与原因）**：`T.atomic_add`（UB scatter-add 无本工具链证据，§0.5）、
`T.set_atomic`（int32 未取证，仅 §1.6.3 备选 M2）、`T.shfl_*` / `T.tvm_warp_shuffle_*`（无 warp，D5）、
`T.cumsum`（公共面 API：在 `tilelang/ascend/language/` 下无该原语的降低/分派证据（grep 无命中；
C++ pass 侧未核验，故按不可用处理），因此自行实现 A3 扫描）、`T.reduce_sum(dim=0)` 二维 shared→shared（`TRAP-950-simd-shared-reduce` 已证实
输出布局差异，本设计不需要二维归约）。

### 3.3 计算伪代码

```python
# ============ 构建期（host，每 (numel, E, B) 一次） ============
numel, E, B = ...
P          = numel + (E + 1) * (B - 1)
NB         = ceildiv(P, B)
E_pad      = align_up(E, 64)                     # 384 / 256 / 128 三条 workload 均 64 对齐
p_shift    = log2(B) if B is power of 2 else None
# 路径判定（§5）：估计核内向量指令数
hist_instr = numel * ceildiv(E, 64) * 2
ids_instr  = NB * ceildiv(E, 64)
path_S     = (hist_instr + ids_instr + ceildiv(P, 64)) <= 8192
G_h        = clamp(ceildiv(numel, 512), 1, 56)   # 路径 L
G_a        = clamp(ceildiv(NB, 16), 1, 56)       # 路径 L

# ============ 路径 S：_align_single(numel, E, B) —— 1 launch, grid=2 ============
def kernel(flat, sorted_ids, expert_ids, npost, cumsum):
    with T.Kernel(2) as bx:
        s_x    = T.alloc_shared([CHUNK], "int32")        # CHUNK = align_up(numel,64)（≤ 512）
        s_cnt  = T.alloc_shared([E_pad],  "int32")
        s_pad  = T.alloc_shared([E_pad],  "int32")
        s_scan = T.alloc_shared([2 * E_pad], "int32")    # 右半为数据，左半 halo 恒 0
        s_cs   = T.alloc_shared([E_pad + 64], "int32")   # E+1 ≤ E_pad+64（跨一个 lane tile）
        s_ids  = T.alloc_shared([BLK],    "int32")       # BLK = align_up(NB, 64)
        s_fill = T.alloc_shared([FILL_TILE], "int32")    # FILL_TILE = 2048

        # ---- 与源 bx==1 对应：哨兵填充分片（两核各半，互不重叠）----
        T.fill(s_fill, numel)
        for t in T.serial(ceildiv(ceildiv(P, 2), FILL_TILE)):      # 本核负责的 tile 序号
            off = bx * ceildiv(P, 2) + t * FILL_TILE
            n   = T.min(FILL_TILE, P - off)
            if n > 0:
                T.copy(s_fill[0:n], sorted_ids[off:off + n])

        if bx == 0 and numel > 0:                                   # 源 bx==0 的计数/扫描/映射
            # ---- S1+S2 直方图（A4）：Pipelined 覆盖 GM→UB 与向量累加 ----
            with T.SimtVF(threads=E_pad):
                for e in T.Parallel(E_pad):
                    s_cnt[e] = 0
            for c in T.Pipelined(ceildiv(numel, CHUNK), num_stages=2):
                base = c * CHUNK
                n    = T.min(CHUNK, numel - base)
                T.copy(flat[base:base + n], s_x[0:n])
                with T.SimtVF(threads=E_pad):
                    for e in T.Parallel(E_pad):
                        acc = T.alloc_var("int32", init=0)          # 每 lane 私有寄存器累加
                        for i in T.serial(n):
                            acc[0] = acc[0] + T.if_then_else(s_x[i] == e, 1, 0)
                        s_cnt[e] = s_cnt[e] + acc[0]
            # ---- S4 补齐（A1） ----
            with T.SimtVF(threads=E_pad):
                for e in T.Parallel(E_pad):
                    pv = T.if_then_else(e < E, s_cnt[e], 0)
                    s_pad[e] = (((pv + (B - 1)) >> p_shift) << p_shift) if p_shift is not None \
                               else T.ceildiv(pv, B) * B
            # ---- S5 HS 扫描（A3）：左 halo 清零，9 步 in-place 轮换 ----
            with T.SimtVF(threads=E_pad):
                for e in T.Parallel(E_pad):
                    s_scan[E_pad + e] = s_pad[e]                # 数据区
                for e in T.Parallel(E_pad):
                    s_scan[e] = 0                               # 左 halo 区（恒 0）
            k = 1
            while k < E:
                with T.SimtVF(threads=E_pad):
                    for e in T.Parallel(E_pad):
                        s_cs[e] = s_scan[E_pad + e] + s_scan[E_pad + e - k]   # 越界已被 halo 吸收
                T.copy(s_cs, s_scan[E_pad:2 * E_pad])
                k *= 2
            # s_scan[E_pad + e] = 包含式前缀；独占前缀 = 左移一位
            with T.SimtVF(threads=E_pad):
                for e in T.Parallel(E_pad + 64):
                    if e <= E:
                        s_cs[e] = T.if_then_else(e == 0, 0, s_scan[E_pad + e - 1])
            # ---- S6① 写回 cumsum / total（单元素 region 写，block 级） ----
            total = s_cs[E]
            T.copy(s_cs[0:E + 1], cumsum[0:E + 1])
            npost[0] = total
            # ---- S6② rank-count（A2） ----
            with T.SimtVF(threads=BLK):
                for b in T.Parallel(BLK):
                    pos = b * B
                    cnt = T.alloc_var("int32", init=0)
                    for e in T.serial(E_pad):
                        cnt[0] = cnt[0] + T.if_then_else(s_cs[e] <= pos, 1, 0)
                    s_ids[b] = T.if_then_else(pos < total, cnt[0] - 1, E)
            T.copy(s_ids[0:NB], expert_ids[0:NB])

# ============ 路径 L：_hist_partial + _align_partial —— 2 launches ============
def hist_kernel(flat, partial):                       # launch-1, grid = G_h
    with T.Kernel(G_h) as g:
        s_x   = T.alloc_shared([CHUNK], "int32")      # CHUNK = align_up(ceildiv(numel,G_h),64) ≤ 1024
        s_hist = T.alloc_shared([E_pad], "int32")
        base = g * TPC                                # TPC = ceildiv(numel, G_h)
        n_g  = T.max(0, T.min(TPC, numel - base))
        with T.SimtVF(threads=E_pad):
            for e in T.Parallel(E_pad):
                s_hist[e] = 0
        for c in T.Pipelined(ceildiv(n_g, CHUNK), num_stages=2):
            b2 = base + c * CHUNK
            n  = T.min(CHUNK, numel - b2)
            if n > 0:
                T.copy(flat[b2:b2 + n], s_x[0:n])
                with T.SimtVF(threads=E_pad):
                    for e in T.Parallel(E_pad):
                        acc = T.alloc_var("int32", init=0)
                        for i in T.serial(n):
                            acc[0] = acc[0] + T.if_then_else(s_x[i] == e, 1, 0)
                        s_hist[e] = s_hist[e] + acc[0]
        T.copy(s_hist, partial[g * E_pad:(g + 1) * E_pad])       # 纯 store，无原子、无需清零

def align_kernel(partial, sorted_ids, expert_ids, npost, cumsum, flat):   # launch-2, grid = G_a
    with T.Kernel(G_a) as a:
        s_part = T.alloc_shared([E_pad], "int32")
        s_cnt  = T.alloc_shared([E_pad], "int32")
        s_cs   = T.alloc_shared([E_pad + 64], "int32")
        s_ids  = T.alloc_shared([BLK], "int32")
        s_fill = T.alloc_shared([FILL_TILE], "int32")
        # ---- S3 冗余归并（每核读同一份 partial，整数确定性 ⇒ 各核结果逐位相同）----
        with T.SimtVF(threads=E_pad):
            for e in T.Parallel(E_pad):
                s_cnt[e] = 0
        for g in T.serial(G_h):
            T.copy(partial[g * E_pad:(g + 1) * E_pad], s_part)
            with T.SimtVF(threads=E_pad):
                for e in T.Parallel(E_pad):
                    s_cnt[e] = s_cnt[e] + s_part[e]
        # ---- S4 + S5 同路径 S（补齐 + HS 扫描）----
        ...
        total = s_cs[E]
        if a == 0:
            T.copy(s_cs[0:E + 1], cumsum[0:E + 1]); npost[0] = total
        # ---- S6② expert_ids 本核分片 ----
        blk0 = a * BLKPC                                       # BLKPC = ceildiv(NB, G_a)
        with T.SimtVF(threads=BLK):
            for j in T.Parallel(BLK):
                b = blk0 + j
                if b < NB:
                    cnt = ... (同路径 S 的 rank-count)
                    s_ids[j] = T.if_then_else(b * B < total, cnt - 1, E)
        valid = T.max(0, T.min(BLKPC, NB - blk0))
        if valid > 0:
            T.copy(s_ids[0:valid], expert_ids[blk0:blk0 + valid])
        # ---- S6③ 哨兵填充本核分片（互相不重叠：本核区间 [a*SHARE, min((a+1)*SHARE, P)) ）----
        SHARE = ceildiv(P, G_a)                                # 构建期常量
        T.fill(s_fill, numel)
        for t in T.serial(ceildiv(SHARE, FILL_TILE)):
            off = a * SHARE + t * FILL_TILE
            n   = T.min(FILL_TILE, T.min((a + 1) * SHARE, P) - off)   # 不越过本核区间
            if n > 0:
                T.copy(s_fill[0:n], sorted_ids[off:off + n])
```

> 说明：`T.alloc_var` 放在 `T.Parallel` 体内表达「每 lane 一个私有累加器」是语义写法，
> Stage 3 若该写法不被接受，等价改为 `T.alloc_local([1],"int32")`（`local` scope）或直接
> 在 `s_cnt[e]` 上读改写（代价：每 token 多一次 UB 读写，属性能差异不影响语义）。
> 路径 L 的 `align_kernel` 保留 `flat` 形参只为与 wrapper 的调用签名对齐（未使用）。

### 3.4 API 可行性确认

| API | 证据（本工具链） | 状态 |
|---|---|---|
| `T.Kernel(G)`（一维、无 `threads=`） | `ascend/language/kernel.py`（签名 `Kernel(*blocks, prelude=None)`）；`references/programming.md` 表「`T.Kernel(N)` 一维核网格；签名没有 `threads`」 | 已确认 |
| `T.SimtVF(threads=...)` / `T.Parallel` | `ascend/language/frame.py:SimtVF`；`PL-950-row-primitives.md`（64 线程、行转换/归约/广播 24 项通过） | 已确认（fp32；int32 同类元素算子属未文档化假设，见 §9） |
| `T.alloc_shared` | `ascend/language/allocate.py`（UB，默认 `shared.dyn`） | 已确认 |
| `T.copy`（含运行期 region 长度、双向） | `ascend/language/copy_op.py`；`kernel_template.py` 尾块 `T.copy(X[full*tile:n], x[:tail])`；`PL-950-vector-execution.md` | 已确认 |
| `T.fill(buf, value)` | `tilelang.language` 公共面 `fill(buffer, value, annotations)`；本设计只用于 UB 常量填充 | 签名已确认；**ascend 降低未单独验证**（§9 风险，备选：`T.Parallel` 写常量） |
| `T.Pipelined(n, num_stages=2)` + 自动多缓冲 | `ascend/pipeline.py` `allow_autoschedule` 默认开启（AutoSchedule/InsertSync/MaterializeMultiBuffer）；`references/programming.md` 同步与缓冲章节 | 已确认（默认流水） |
| `T.alloc_var` / `T.alloc_local` | `T.alloc_var(dtype, scope='local.var', init=...)`；`SimtVF` 内每 lane 私有量 | 已确认（`PL-950-row-primitives.md` 使用行标量） |
| `T.if_then_else` / `T.min` / `T.max` / `T.ceildiv` | TIR 公共算子 | 已确认 |
| int32 收窄为 uint16 lane（备选） | `simd.py` `_ELEM_BITS` 含 int16/uint16，`_vreg_lanes`=2048/16=128；文档「validated 8/16/32-bit operations」 | 仅源码确认，**未验证**（备选，裁决计划内） |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 限制（本平台） | 本设计处置 | 结论 |
|---|---|---|
| Kernel 只支持 **一维** core grid，且**无 `threads=` 参数** | `T.Kernel(G)` 一维；线程域在核内用 `T.SimtVF` | ✓ 合规（源算子的 `T.Kernel(2, threads=threads)` 已重设计，D10） |
| 不允许依赖**未认证的跨 block barrier / 跨核 flag** | 全程无跨核数据依赖（冗余读替代），不用 `ascend_cross_core_*`/`sync_grid` | ✓ 合规 |
| 无 warp / 无 warp shuffle | 前缀和改 UB 向量扫描（A3，D5） | ✓ 合规 |
| 无 UB scatter-add（原子只在 GM 侧有 FP32 证据） | 直方图改比较累加（A4，D4） | ✓ 合规 |
| UB 248KB/核（`ascend-950pr-9579`，平台配置 `ub_size`，md5 2f0eca1f…） | 峰值 ≈30KB（§4.5），含 1.7x 膨胀余量仍 ≪ 上限 | ✓ 合规 |
| 不使用 L1/L0C/L0A/L0B 时不得申请 | 全设计只 `T.alloc_shared` | ✓ 合规 |
| 尾块必须显式限定有效 region（`pad_value`/`data_select` 未验证执行行为） | 全部 `T.copy` 用显式 region + `T.min` 截断，不依赖 `pad_value` | ✓ 合规 |
| 非整除 shape（`numel` 不整除 `CHUNK`、`P` 不整除 `FILL_TILE`、`NB` 不整除 `G_a`） | 三条尾块路径均显式 region + `n > 0` 守卫；`numel=0` 另有折叠守卫 | ✓ 合规 |
| `T.cumsum` 等公共面原语不得假定可用 | 自查：`ascend/language/` 无 cumsum 降低实现 → 自行实现 A3 | ✓ 已规避 |
| NPUIR 专属规则（三维 Kernel、reduction 宏、H200 列上限、8KiB 预留）不适用本工具链 | 未使用任何 NPUIR 规则或 `examples/elementwise/atomic_add.py`（`target="npuir"`）的写法作证据 | ✓ 合规 |

#### 3.5.2 参考实现差异说明

| 维度 | GPU 源算子 | 本设计（NPU） | 差异原因 |
|---|---|---|---|
| 并行结构 | 2 CTA × up to 1024 线程 | 1-D core grid（路径 S 2 核 / 路径 L `G_h`+`G_a` 核） | 平台 launch 无 `threads=`（K9 + kernel.py） |
| 直方图 | SMEM `atomic_add` | expert-lane 比较累加 + partial GM 归并 | NPU 无 UB scatter-add（§0.5 D4） |
| 前缀和 | warp shuffle 树 + warp 和串行 | UB 内 9 步 HS 向量扫描 | 无 warp（D5） |
| `expert_ids` | expert 主序变长 scatter | rank-count 恒等式 | 数据相关 trip count 无法向量化/分核（D8） |
| 同步 | 显式 `T.sync_threads()` 4 处 | 无显式同步（核内自动、核间无依赖） | 平台默认流水 + 刻意解耦（D11） |
| 哨兵填充 | 逐线程条件写 | 常量 UB tile + 批量 `T.copy` | 向量化（V1） |
| launch 次数 | 1（align） | 路径 S 1 次 / 路径 L 2 次 | 跨核无 barrier ⇒ 大档需两次 launch 完成「局部直方图 → 归并」 |

#### 3.5.3 本项目同类实现参考

- lane 并行 / 行归约 / 广播：`.agents/platforms/toolchains/ascend/knowledge/PL-950-row-primitives.md`
  （及其 repro `.agents/platforms/toolchains/ascend/knowledge/repro/PL-950-row-primitives.py`）。
- 向量执行与尾块读写：`.agents/platforms/toolchains/ascend/knowledge/PL-950-vector-execution.md`。
- 平台 kernel 模板（`T.Kernel` 一维 + `T.SimtVF` + 尾块 region）：
  `.agents/platforms/toolchains/ascend/kernel_template.py`。
- 显式向量归约（备选实现的 API 出处）：`.agents/platforms/toolchains/ascend/knowledge/repro/PL-950-simd-reduce.py`。
- **无同类 index/histogram 算子的已认证实现**：本仓 `examples/` 下无 histogram / scatter / permutation
  类算子（已检索 `examples/TileOPs/tileops/kernels/` 下全部 7 个算子族目录：attention / elementwise /
  mamba / moe / norm / reduction / rope，其中 `moe` 目录只含本算子族自身的脚手架，均无索引分发类参照）。
  因此本设计的「直方图 + 前缀 + 映射」三段结构为**本项目首例**，
  相关 API 组合的风险在 §9 显式登记（`T.fill` 降低、int32 lane 并行、`alloc_var` 在 `Parallel` 内的语义）。
## 4. 数据规格与内存规划

### 4.1 输入张量

| 名称 | shape | dtype | 动态轴 | 说明 |
|---|---|---|---|---|
| `flat` | `[numel]`，`numel = T*K` | int32 | 构建期特化（T 为运行时 token 数，但每个 `(numel,E,B)` 单独构建） | `topk_ids.flatten().contiguous()` 的视图；取值 `[0, num_experts)` |

### 4.2 输出张量

| 名称 | shape | dtype | 本段写入内容 |
|---|---|---|---|
| `sorted_token_ids` | `[P]`，`P = numel + (E+1)(B-1)` | int32 | 全槽位哨兵 `numel`（路径 L：`G_a` 核分片；路径 S：2 核分片） |
| `expert_ids` | `[NB]`，`NB = ceil(P/B)` | int32 | `[0,total/B)` 为 owner expert；其余写 `num_experts` |
| `num_tokens_post_pad` | `[1]` | int32 | `total = cumsum[E]` |
| `cumsum` | `[E+1]` | int32 | 独占前缀和（内部交接给 scatter，非公开返回值） |

三条 workload 的 `max_padded`/`max_num_blocks`（B=64）：

| E \ numel | 8 | 256 | 4096 | 32768 |
|---|---|---|---|---|
| 384 | P=24,263 / NB=380 | 24,511 / 383 | 28,351 / 443 | 57,023 / 891 |
| 256 | 16,199 / 254 | 16,447 / 257 | 20,287 / 317 | 48,959 / 765 |
| 128 | 8,135 / 128 | 8,383 / 131 | 12,223 / 191 | 40,895 / 639 |

### 4.3 中间缓冲区

| 名称 | 位置 | shape | dtype | 用途 | 生命周期 |
|---|---|---|---|---|---|
| `s_x` | UB | `[CHUNK]`（≤1024） | int32 | token 分片 | 每 chunk 覆盖写（2 版本流水） |
| `s_cnt` | UB | `[E_pad]`（E_pad = align_up(E,64) ≤ 384/512） | int32 | 局部直方图 / 归并结果 | 每核 |
| `s_pad` | UB | `[E_pad]` | int32 | 补齐后计数 | 每核 |
| `s_scan` | UB | `[2*E_pad]`（左半 halo 恒 0） | int32 | HS 扫描工作区（含左 halo） | 每核 |
| `s_cs` | UB | `[E_pad + 64]` | int32 | 独占前缀（`cumsum` 副本，`E+1` 个有效元素） | 每核 |
| `s_ids` | UB | `[BLK]`（= align_up(NB,64) 或 BLKPC 对齐） | int32 | `expert_ids` 分片暂存 | 每核 |
| `s_fill` | UB | `[FILL_TILE]`（=2048） | int32 | 哨兵常量 tile | 每核复用 |
| `s_part` | UB | `[E_pad]` | int32 | 路径 L：读 `partial` 的一行 | 每行覆盖写 |
| `partial` | **GM scratch** | `[G_h, E_pad]` | int32 | 路径 L 的局部直方图（≈86KB @ G_h=56,E_pad=384） | 由组合可调用对象按实例持久分配（`torch.empty`，无需初始化） |

`partial` 的生命周期约定：同一编译实例内的连续调用共用同一块 scratch；两次 launch 之间**只写不累加**
（每次 launch-1 全量覆盖 `[g, 0..E_pad)` 行，空核写全零行），因此无需清零、无跨调用残留；多流并发使用
同一实例不安全（§9 风险项，wrapper 现状为单 stream 串行调用）。

### 4.4 内存搬运路径

```
路径 S：GM flat ──MTE2(T.copy)──> UB s_x ──AIV 向量──> UB s_cnt ─> s_pad ─> s_scan/s_cs ──(s_ids)──>
        GM expert_ids / cumsum / num_tokens_post_pad ；UB s_fill ──MTE3(T.copy)──> GM sorted_token_ids
路径 L：launch-1  GM flat ──MTE2──> UB s_x ──AIV──> UB s_hist ──MTE3(T.copy)──> GM partial
        launch-2  GM partial ──MTE2(T.copy, G_h 次)──> UB s_part ──AIV 累加──> s_cnt ─> ...（同路径 S 写回）
```

- 不经过 L1：无矩阵操作，`T.copy` 直接 GM↔UB（平台规则允许：UB 为默认 `shared.dyn` scope）。
- 无 L0A/L0B/L0C 分配（不调用 `T.gemm`）。
- 所有写回均为**互不重叠 region**（分片写），无需跨核同步。

### 4.5 UB 内存预算

| 路径 | buffer | 字节 |
|---|---|---|
| S（E=384） | `s_x` 512×4 = 2,048；`s_cnt/s_pad` 2×384×4 = 3,072；`s_cs` 448×4 = 1,792；`s_scan` 2×384×4 = 3,072；`s_ids` 384×4 = 1,536；`s_fill` 2048×4 = 8,192 | **19,712 B（19KB）** |
| L launch-1（E=384） | `s_x` 2 版本 ×1024×4 = 8,192；`s_hist` 384×4 = 1,536 | **9,728 B（9.5KB）** |
| L launch-2（E=384） | `s_part` 2 版本 ×384×4 = 3,072；`s_cnt/s_pad` 3,072；`s_cs` 448×4 = 1,792；`s_scan` 3,072；`s_ids` 1,536；`s_fill` 8,192 | **20,736 B（20KB）** |

上限校验：UB = 253,952 B/核（平台配置 `ascend-950pr-9579`）。最坏路径峰值 20KB，即使按
`CONST-capacity-910B2C` 记录的 auto-multi-buffer 膨胀上界 1.7x（**910B2C 实测，跨型号外推**）也仅 ~35KB，
占 14%；本设计显式 alloc 的 buffer 少且小，容量**非瓶颈**（§1.6.0 roofline 容量项 0.12）。
L1/L0 预算：不使用。

### 4.6 动态轴定义

- 运行时唯一可变的是**数据内容**（`flat` 的取值分布）；shape 全部由 `(numel, E, B)` 在构建期固定。
- 因此 kernel 内**所有循环上界均为构建期常量**（除 `n = min(CHUNK, numel - base)` 这类纯常量表达式），
  无动态 shape 分支；`numel=0` 由构建期折叠的守卫覆盖。
- `T`（token 数）在 wrapper 层是多实例的：每个 `numel` 一个编译实例与一处 scratch。

### 4.7 JIT 配置

| 项 | 值 | 说明 |
|---|---|---|
| `target` | `"ascend"` | 平台 ruleset 指定；`out_idx=[]`（全部缓冲由调用方传入） |
| 编译缓存 | 每 `(numel, num_experts, block_size)` 一次（`functools.lru_cache` / 实例字段） | 对应 GPU 的 `target_cache`（D9 保留） |
| launch 参数 | 无（无 `threads`） | K9；`default_config` 返回空 dict |
| 流水 | 自动调度开（默认）；`T.Pipelined(num_stages=2)` 用于 chunk 循环 | `ascend/pipeline.py` 默认 `allow_autoschedule=True` |
| `pass_configs` | 不设置（默认） | 如需禁用自动多缓冲：`annotate_buffer_versions({s_x: 1})`（仅在调试时） |
| 返回值 | 路径 S 返回 1 个编译程序；路径 L 返回**组合可调用对象**（内部依次启动 `hist` 与 `align`，并持有 `partial`） | 保持 wrapper 的 `align(flat, sorted_ids, expert_ids, npost, cumsum)` 调用签名不变 |
| 执行命令（本机实测） | `source .agents/platforms/toolchains/ascend/env-950pr.sh` 后用 `/workspace/venv-ascend950/bin/python`；cwd = 任务目录 | 平台 `execution` 模板中的 `/usr/local/python3.12.16/bin/python3` 在本机不存在（任务上下文实测），Stage 3 须按本行执行 |

## 5. Tiling 策略

### 5.1 计算类型

索引分发类（直方图 + 一维前缀和 + block→expert 映射 + 常量填充）：**纯整型、无矩阵、无逐元素数值计算**，
`flops = 0`；性能由「GM 流量（哨兵填充为主）+ 向量指令发射 + launch 开销」决定。

### 5.2 Block 划分

| 维度 | 划分 | 取值（构建期） |
|---|---|---|
| token 轴 | 每核 `tokens_per_core = ceildiv(numel, G_h)`，核内再按 `CHUNK` 切块 | `CHUNK = min(1024, align_up(tokens_per_core, 64))`；路径 S 单核全量 |
| expert 轴 | 不切分：`E_pad ≤ 384/512` 全部驻留一个 UB 缓冲（一条 lane-向量链） | `E_pad = align_up(E, 64)` ∈ {128, 256, 384} |
| 输出 block 轴 | 每核 `BLKPC = ceildiv(NB, G_a)` 个 block；核内按 `BLK` 切块 | `BLK = align_up(BLKPC, 64)` |
| 哨兵填充轴 | 每核 `ceildiv(P, G_a)`（路径 L）/ `ceildiv(P, 2)`（路径 S）个元素，按 `FILL_TILE` 切块 | `FILL_TILE = 2048` |

**非整除处理**（GEMM 类以外同样适用）：三条尾块路径（token 尾、block 尾、填充尾）均用
`n = T.min(TILE, total_remaining)` 计算有效长度并以显式 region `T.copy(src[0:n], dst[off:off+n])` 搬运；
`n ≤ 0` 时跳过（`if n > 0`）。不依赖 `pad_value`/`data_select`（平台标注其执行行为未验证）。
`numel=0` 时 `G_h = 1`、`tokens_per_core = 0`，直方图输出全零行，结果仍与参考一致。

### 5.3 约束分析

| 约束 | 检查 |
|---|---|
| 单核 UB 峰值 ≤ 248KB | 20KB（§4.5）✓ |
| 核内循环上界静态 | 全部构建期常量 ✓ |
| 写域互不重叠 | `cumsum`/`num_tokens_post_pad` 由核 0 独占；`expert_ids` 按 block 区间分片；填充按下标分片 ✓ |
| 无跨核依赖 | 冗余读 `partial`（只读）+ 分片写 ✓ |
| 单次 launch 逻辑核数 ≤ 物理核数 | 见 §5.5 ✓ |

### 5.4 注意事项

- 路径 S 只用 2 个核：前缀和与 `expert_ids` 必须由单一核完成（跨核无 barrier），
  因此瓶颈是**核内工作量 + 1 次 launch 延迟**，而不是核数；小档 (`numel ≤ 341`) 已证明核内工作量 ≤ 8K 指令。
- 路径 L 的「每核冗余读 `partial`」成本随 `G_h·E` 增长；`G_h` 的选择在「直方图工作量 / 冗余读」间取舍
  （§9 风险项 1，Stage 4 按 §1.6.3 裁决计划实测）。
- `partial` 布局用 `[G_h, E_pad]` 行主序：launch-2 每次读一整行（`E_pad×4 = 1.5KB` 连续），
  符合平台「128B 段 / 减少指令数」的搬运偏好（`CONST-mte2-degradation` 的指令维度口径）。

### 5.5 分核策略（物理核数适配）⭐

**① 逻辑核数计算**

本算子的「逻辑核数」由三段可并行工作各自的自然分块数决定（不是 `M/block_M × N/block_N` 形式，
因为无矩阵网格）：

```
直方图段   : G_h_logical = ceildiv(numel, TOK_PER_CORE)        TOK_PER_CORE = 512
映射/写回段: G_a_logical = ceildiv(NB, BLK_PER_CORE)           BLK_PER_CORE = 16
填充段     : G_f_logical = ceildiv(P, FILL_TILE)               FILL_TILE = 2048
路径 S（1 launch）: G = 2（核 0 = 计数/扫描/映射 + 半填充，核 1 = 半填充）
路径 L（2 launch）: launch-1 G_h = G_h_logical，launch-2 G_a = max(G_a_logical, ceildiv(G_f_logical, 8))
```

三条 workload 的实际取值：

| E | numel | 路径 | G_h | G_a | 每核 token 数 | 每核 block 数 |
|---|---|---|---|---|---|---|
| 384 | 8 | S | — | 2 | 8（核 0） | 380（核 0） |
| 384 | 256 | S | — | 2 | 256（核 0） | 383（核 0） |
| 384 | 4096 | L | 8 | 28 | 512 | 16 |
| 384 | 32768 | L | 56 | 56 | 586 | 16 |
| 256 | 4096 | L | 8 | 28 | 512 | 12 |
| 256 | 32768 | L | 56 | 56 | 586 | 14 |
| 128 | 4096 | L | 8 | 14 | 512 | 14 |
| 128 | 32768 | L | 56 | 56 | 586 | 12 |

**② 物理核数依据（按任务平台规则实查）**

- 查询代码与实际返回值（本机执行，2026-11-23）：

  ```python
  import torch, torch_npu
  p = torch.npu.get_device_properties(0)
  p.name               -> 'Ascend950PR_9579'
  p.cube_core_num      -> 28
  p.vector_core_num    -> 56
  ```

- 平台记录交叉核对：`statectl platform --task-dir examples/moe_permute_align/_make_align_kernel` 返回
  `hardware.hardware_id = ascend-950pr-9579`，`hardware.observed_cores = {cube: 28, vector: 56}`，
  `core_source = "Ascend950PR_9579.ini [SoCInfo] ai_core_cnt=28, cube_core_cnt=28, vector_core_cnt=56;
  runtime probe rtGetAiCoreCount()=28 on Ascend950PR_9579, 2026-11-23"`。
- 本算子只用 AIV（向量核），故物理核数基准 = **56**（cube 28 不使用）。

**③ 规模判定与分核方案**

- **结论：中等规模 → 对齐物理核整数倍**。取值集合限定为 56 的因数 `{1, 2, 4, 7, 8, 14, 28, 56}`，
  使每核任务数均衡（`⌈n/G⌉` 与 `⌊n/G⌋` 之差 ≤ 1），且 56 = 28（cube）×2 的整倍数关系清晰：
  - 路径 S：`G = 2` ⟂ 逻辑核数上限 2（单核串行依赖所致），**2 ≤ 56 无需适配物理核**（依据：前缀与映射
    必须单核完成，增加核数无法分摊；核内工作量 ≤ 8K 指令，延迟由 launch + 单核工作量决定）。
  - 路径 L：`G_h`、`G_a` 各自 `≤ 56`；`G_h = 56 = 56×1`（最大档满核），中间档 8/14/28 均为 56 的因数
    （512 token/核 ≈ 3K 向量指令/核，避免核启动开销摊不薄）；`G_a` 由 `ceildiv(NB,16)` 向上取到因数集，
    使 `expert_ids` 与填充分片均衡。
  - **极大规模**（`numel` 远超 32768，例如 prefill 的百万级路由）：逻辑核数仍固定为 56，
    靠**核内串行**处理多块任务（`tokens_per_core = ceildiv(numel, 56)` 增大 + `T.Pipelined` 多 chunk 覆盖），
    即「固定启动核数 = 物理核数，核内分批」，不增加 launch 规模。
- 核内循环边界限制：本平台 `T.Kernel` 一维网格、无 block 维上限文档值；`T.SimtVF(threads=...)` 取
  ≤ 512（E_pad ≤ 384 / BLK ≤ 512），循环使用 `T.Pipelined`/`T.serial`/`T.Parallel`（平台 ruleset 支持）。
- 逻辑核数超发检查：路径 L 单次 launch ≤ 56 = 物理核数，无超发；路径 S 2 ≤ 56，无超发。

## 6. 循环与调度结构

### 6.1 循环结构总结

| # | 循环 | 结构 | 展开/调度 | 说明 |
|---|---|---|---|---|
| L1 | chunk 循环（路径 S/L 的 token 分片） | `T.Pipelined(n_chunks, num_stages=2)` | 自动多缓冲 | GM→UB 与向量累加重叠 |
| L2 | lane 并行（expert / block 轴） | `T.Parallel(E_pad)` / `T.Parallel(BLK)` | 向量化（64 int32 lane/向量） | lane ↔ expert，一 lane 一私有计数器 |
| L3 | per-lane 归约轴（token / expert 轴） | `T.serial(n)` | 不展开（或小步展开） | §1.6.2 V3 保留项（归约轴，无 scatter 替代） |
| L4 | 路径 L 的 partial 归并 | `T.serial(G_h)` + 内层 `T.Parallel(E_pad)` | 每轮一次 GM→UB（1.5KB，可流水） | 只读、无依赖 |
| L5 | HS 扫描 9 步 | 构建期 `while k < E` 展开 | 每步 1 次 `T.Parallel(E_pad)` 向量加 | 9 步顺序依赖（UB 内），无并行空间（log 深度已最优） |
| L6 | 哨兵填充 tile 循环 | `T.serial(ceildiv(share, FILL_TILE))` | 每次 `T.copy` 批量 store | 无计算 |
| L7 | `expert_ids` block tile 循环 | `T.serial(ceildiv(BLKPC, BLK))` | 每 tile 一次 rank-count | 可核内串行 |

不使用 `T.Persistent`（任务数 ≤ 56×2，核启动开销已被 launch 次数主导，persistent 无收益）；
不使用 `T.serial` 处理 lane 轴（会退化为标量）。

### 6.2 循环伪代码

见 §3.3（含路径 S 与路径 L 两段完整伪代码）。关键点：

- 归约轴（`T.serial(n)`）在 lane 轴内层，形成「一 lane 一私有累加器」的垂直扫描；
- HS 扫描的 9 步用构建期 `while` 展开（不是运行期循环），每步一个 `T.Parallel` 向量区；
- 所有 `T.Parallel` 之外的计算（`base`、`tail`、`total`）都是 block 级标量。

### 6.3 流水线优化

- chunk 循环 `num_stages=2`：`s_x` 双版本，GM→UB（MTE2）与向量累加（V）重叠；平台默认 AutoSchedule
  会自动插入同步与多缓冲（`ascend/pipeline.py`）。若 `numel` 小（`n_chunks = 1`）流水退化为单拍，无害。
- 路径 L 的 partial 归并（L4）可视为「只读流水」：`s_part` 双版本可与累加重叠（由自动多缓冲决定）。
- 不手工设置 `annotate_buffer_versions`（默认由 MaterializeMultiBuffer 处理）；仅在调试关闭流水时使用。

### 6.4 尾块处理

| 尾块 | 处理 |
|---|---|
| token 尾（`numel % CHUNK != 0`） | `n = T.min(CHUNK, numel - base)`；`T.copy(flat[base:base+n], s_x[0:n])`；`T.serial(n)` |
| block 尾（`NB % BLK != 0` / `NB % G_a != 0`） | `b < NB` 守卫；写回用 `expert_ids[blk0 : blk0 + valid]` 显式 region |
| 填充尾（`P % FILL_TILE != 0`） | `T.copy(s_fill[0:n], sorted_ids[off:off+n])`，`n = T.min(FILL_TILE, P - off)` |
| `E` 非 64 倍数（如 E=33） | `E_pad = align_up(E,64)`；`e < E` 的 lane 补 0（`s_cnt` 补位恒 0 ⇒ `p_e=0`，不进入前缀）；`cumsum` 只写 `[0,E]` |
| `numel = 0` | 构建期折叠的 `if numel > 0` 守卫；`G_h=1` 时 launch-1 写全零行 |

## 7. 同步策略

### 7.1 同步模式

**自动同步（默认流水）+ 零跨核同步**。平台 `ascend/pipeline.py` 的 `allow_autoschedule` 默认开启，
AutoSchedule/InsertSync 负责核内 UB 的写后读、读后写依赖与 pipeline 屏障；本设计**不使用**任何
`ascend_cross_core_set_flag/wait_flag`、`ascend_pipe_barrier`、`sync_grid`、`T.sync_threads()`。

### 7.2 同步点说明

| 依赖 | 位置 | 处理 |
|---|---|---|
| `s_x` 写 → 读（同 chunk） | L1↔L2 | 自动同步（pipeline 屏障） |
| `s_cnt` 清零 → 累加 | S2 前后 | 同一 `T.Parallel` 区顺序执行 + 自动同步 |
| `s_pad` → `s_scan` → `s_cs` | S4→S5 | 同核顺序区，自动同步；HS 步内 `s_cs` 轮换写回 `s_scan` 需写后读，自动插入 |
| `s_cs` → `expert_ids` | S5→S6② | 同核顺序区 |
| partial 写（launch-1）→ 读（launch-2） | **跨 launch** | 同一 stream 上的两次 launch 顺序保证（无需 device 内同步） |
| 分片 GM 写（多核） | S6②③ | 下标区间互不重叠，无数据竞争（**无需同步**） |
| `cumsum` / `npost` 写 | 核 0 独占 | 无竞争 |

### 7.3 pass_configs 配置

- 默认流水线（不设 `pass_configs`）；不关闭 `allow_autoschedule`。
- 若 Stage 3 出现 UB 竞争（例如 HS 扫描的 `s_cs`↔`s_scan` 轮换被误判为并行），处置顺序：
  ① 用两个独立 buffer 交替（`s_scan_a`/`s_scan_b`）消除同一个 buffer 的读写混用；
  ② 仍报错则加 `T.ascend_pipe_barrier("PIPE_ALL")`（平台 `sync.py` 提供，单核管线配对已有执行证据见
  `execution-patterns.md` PL-950-manual-sync）；
  ③ 关闭自动调度仅在调试时使用（`tl.enable_auto_schedule=False`），不作为交付形态。
- 不手工多缓冲（`annotate_manual_multi_buffer`）：buffer 少且小，交给自动调度更稳。
## 8. 验证方案

### 8.1 Golden 函数

**独立性声明**：golden 依据 §0.1 的**语义**编写，移植源仓参考实现
（`examples/TileOPs/tileops/workloads/moe.py` 中的 `ref_permute_align`，其纯 Python 循环版为语义基准；
该实现用 `bincount` + `cumsum` + **二分查找**求 owner）。它**不复刻**本设计的 NPU 算法：
golden 用 `torch.bincount`/`torch.cumsum`/`torch.searchsorted`，而 kernel 用「比较累加直方图 + HS 扫描 +
rank-count」——两条独立路径的数值一致由 §1.6.1 的 verify_equiv.py 三方对照保证（基线 / 本设计 / 参考实现）。

```python
import math
import torch

def golden_align(topk_ids: torch.Tensor, num_experts: int, block_size: int):
    """The align stage's outputs, from the operator's semantics (§0.1) only.

    Returns (sorted_token_ids, expert_ids, num_tokens_post_pad, cumsum).
    sorted_token_ids is the sentinel-filled buffer: the align stage writes the
    route indices *not at all* (the scatter stage does), so every slot holds
    ``numel``.
    """
    B = block_size
    E = num_experts
    flat = topk_ids.flatten()
    numel = int(flat.numel())
    dev = flat.device

    counts = torch.bincount(flat.long(), minlength=E)[:E]                  # cnt_e
    padded = (counts + B - 1) // B * B                                    # ceil to B
    cumsum = torch.zeros(E + 1, dtype=torch.int64, device=dev)
    cumsum[1:] = torch.cumsum(padded, 0)                                  # exclusive prefix
    total = int(cumsum[E].item())

    P = numel + (E + 1) * (B - 1)
    NB = math.ceil(P / B)
    sorted_token_ids = torch.full((P,), numel, dtype=torch.int32, device=dev)
    expert_ids = torch.full((NB,), E, dtype=torch.int32, device=dev)

    nb = total // B                                                       # blocks owned by an expert
    if nb:
        pos = torch.arange(nb, dtype=torch.int64, device=dev) * B
        # the reference's binary-search semantics: the e with cumsum[e] <= pos < cumsum[e+1]
        expert_ids[:nb] = torch.searchsorted(cumsum[1:], pos, right=True).to(torch.int32)
    return sorted_token_ids, expert_ids, torch.tensor([total], dtype=torch.int32, device=dev), \
        cumsum.to(torch.int32)
```

在整算子层（`sorted_token_ids` 的路由槽位由 scatter 填），L0 只核对 align 段可独立判定的四项：

1. `sorted_token_ids == numel`（全部槽位；align 之后、scatter 之前）；
2. `num_tokens_post_pad` 精确相等；
3. `cumsum[0..E]` 精确相等（独占前缀和）；
4. `expert_ids[0 : total//B]` 精确相等，且 `expert_ids[total//B : ]` 全部 `≥ num_experts`
   （与 `examples/TileOPs/tests/ops/test_moe_permute_align.py` 中的 `_assert_route_equivalent` 的尾段判据一致：
   该测试允许尾段写任意「无名」值，本设计写 `num_experts`，是源算子的选择）。

### 8.2 精度标准

- **精确相等**（int32 索引数组，`torch.equal`）：`atol = rtol = 0`。理由：全流程整数运算，
  无浮点舍入，A1–A4 已机器验证逐位等价（§1.6.1，3,240 例）。
- 不做容差比较；任何差异都视为语义错误（不是精度问题）。
- 值域断言：`0 ≤ flat < E`（输入契约，wrapper 注释亦声明）；越界输入不在契约内（源算子同样不检查）。

### 8.3 L0 门槛测试计划

执行方式（本机实测环境）：`cd examples/moe_permute_align/_make_align_kernel` →
`source .agents/platforms/toolchains/ascend/env-950pr.sh` → `/workspace/venv-ascend950/bin/python {op}.py --level L0`
（平台 `execution.precision_command` 的 Python 路径在本机不存在，按任务上下文实测替换）。
L0 = 「最小可用 + 每条代码路径 + 每个边界」的一次性门槛，≤ 8 个用例，单次运行 < 1 min：

| # | 用例 `(numel, E, B)` | 覆盖点 | 期望路径 |
|---|---|---|---|
| L0-1 | `(8, 384, 64)`（manifest smoke，T=1,K=8） | 最小规模；`E ≫ numel`；填充量远大于有效量 | 路径 S |
| L0-2 | `(256, 384, 64)`（T=32） | 路径 S 上界附近；多 chunk 直方图 | 路径 S |
| L0-3 | `(4096, 384, 64)`（T=512） | 路径 L 全链路（partial 归并 + 分片写） | 路径 L |
| L0-4 | `(32768, 384, 64)`（T=4096，最大档） | 满核 56 + 大直方图 | 路径 L |
| L0-5 | `(1024, 8, 16)` | 小 E（`E_pad=64` 尾 lane 浪费）+ `B=16` 非 64 的整除关系 | 路径 S |
| L0-6 | `(600, 33, 64)` | `E` 非 64 倍数（33）；奇数 expert 数 | 路径 S |
| L0-7 | `(0, 8, 16)` | `numel=0` 边界（wrapper 只调 align、跳过 scatter）：`total=0`、`expert_ids` 全 `E` | 路径 S 守卫 |
| L0-8 | `(4096, 128, 64)` | E=128 的 32 位对齐 + 路径 L 中间档 | 路径 L |

判定：8 条全部精确相等 → `[PRECISION_PASS]`；任一不等 → 记录首个差异下标与 `(path, E, B, numel)` → 回 Stage 3。
L1/L2/Boundary 分层套件由 `tilelang-op-develop` 生成（含 12 条 manifest workload + 随机分布扫描），本文不枚举。

## 9. 风险点与注意事项

### 9.1 已知约束

1. **性能低于 roofline（主要风险）**：直方图由源的 `O(numel)` 原子增量变为 `O(numel·ceil(E/V))` 向量比较，
   最大档估算 3.4 µs（发射项）vs 流量下界 0.21 µs（§1.6.0 roofline），差 ~16x。
   这是「NPU 无 UB scatter-add」的必然代价（§1.6.0 R1 C2 的代价项），Stage 4 可按 §1.6.3 裁决计划用
   uint16 lane（指令数减半）与 GM 原子合并（省冗余读）压缩；若两项都不成立，则本算子的性能天花板
   即「每核 6 条指令/token」，应在 `[DESIGN_LIMIT]` 中如实记录。
2. **自动向量化不确定**：`T.SimtVF` + `T.Parallel(E_pad)` 内层 `T.serial(n)` + 每 lane `alloc_var` 的写法，
   平台已认证案例覆盖 fp32 逐元素/归约/广播（`PL-950-row-primitives.md`），**未覆盖 int32 每 lane 私有累加器**
   （未文档化假设）。若后退化为标量循环，指令数会放大约 64 倍 → 必须在 L0 之后由 Stage 3 的耗时或
   生成的 CCE 代码确认；退路：显式 `T.simd.vld/vadd/vsts` 寄存器级实现（`repro/PL-950-simd-reduce.py` 已验证）。
3. **`T.fill` 的 ascend 降低未单独验证**（只有签名证据）。退路：用 `T.Parallel` 写常量 tile（同样是向量写），
   语义不变。
4. **路径 L 的两次 launch 由「组合可调用对象」实现**（§4.7）：wrapper 的 `align(...)` 签名不变，但
   `_make_align_kernel` 的返回值在路径 L 下是 Python 组合对象（内含 2 个编译程序 + 1 块持久 scratch）。
   Stage 3 必须按 §4.7 实现；若 wrapper 侧要求「单编译程序」，则退化为路径 S 结构并接受
   大档单核性能（≈ 100 µs 级）——**不允许**为省 launch 而引入未认证的跨核 barrier。
5. **`partial` scratch 的并发安全**：同一实例的串行调用安全（每次 launch-1 全量覆盖）；多 stream 并发
   调用同一实例不安全（wrapper 现状为单 stream）。若未来并发，改为每次调用新分配（`torch.empty`，
   无清零成本）即可。
6. **路径阈值**（`hist_instr + ids_instr + fill_instr ≤ 8192`）依赖「1 向量指令/cycle」的未实测假设；
   对本任务 12 条 workload 的判定裕度很大（路径 S 最大 5.4K，路径 L 最小 16K），阈值误判风险低；
   若 Stage 4 实测发现路径 S 在 `numel=256` 已超过 2 次 launch 的开销，只需下调阈值（构建期常量）。
7. **HS 扫描的跨 lane 位移读**（`s_scan[E_pad + e - k]`）：语义上每 lane 读相邻 lane 的值。若后端把它
   降级为 gather，正确性不受影响、性能略降（9 步 × 6 条指令的规模很小）；若编译器拒绝该下标形式，
   退路是「两步交替 buffer + `T.copy` 带偏移 region」。
8. **`numel=0` 的建图**：`T.Kernel(G)` 的 `G` 由 `clamp(..., 1, 56)` 保证 ≥ 1；`T.copy` 的零长度 region
   必须被 `if n > 0` 完全挡住（平台未文档化零长度 region 行为，本设计不依赖它）。
9. **`T.SimtVF(threads=E_pad)` 的最大线程数未文档化**：`frame.py:SimtVF` 只校验「正整数或 ≤3 维序列、
   非零」，平台已认证案例用过 64 与 256，`384/512` **未验证**。退路（语义不变）：固定
   `threads = 128`，每 lane 负责 `EX = ceildiv(E_pad, 128)` 个 expert（外层 `T.serial(EX)` 覆盖该 lane 的
   expert 子集），`s_cnt` 索引变为 `lane + 128*j`；代价是每 lane 的累加器 ×EX、指令数不变。
   L0 需在路径 S/L 各加一条该变量形态的用例（或 Stage 3 先做一次静态编译探针）。

### 9.2 常见错误

- **把 `cumsum` 当作公开输出**：它是 align→scatter 的内部交接缓冲；只在 L0 内部核对，不进 `forward` 返回值。
- **误以为 align 写路由索引**：align 只写哨兵；整算子层必须用 scatter 后的归一化比较
  （`test_moe_permute_align.py` 中的 `_assert_route_equivalent`），不能逐元素比 `sorted_token_ids`。
- **尾段 `expert_ids` 写 0 而不是 `num_experts`**：源算子与参考实现都写 `num_experts`（≥ 无名哨兵），
  写 0 会被下游 grouped GEMM 当作 expert 0。
- **路径 L 忘记「空核写全零行」**：漏写会让 launch-2 读到 `partial` 的残留值（scratch 未初始化）。
- **`E_pad` 与 `E` 混用**：`cumsum` 只能写 `[0, E]`（长度 `E+1`），`E_pad` 只用于 UB 内 lane 缓冲；
  多写会越界写 GM。
- **用 `T.Persistent` 或三维 `T.Kernel`**：本平台不支持（kernel.py 一维网格）。

### 9.3 特殊场景处理

| 场景 | 行为 |
|---|---|
| 某 expert 计数为 0 | `p_e = 0`，其 block 区段为空；rank-count 与源 scatter 一致（verify_equiv 覆盖） |
| 连续多个零计数 expert | 同上；rank-count 取「最后一个满足者」，已证明等价（§1.6.1 A2） |
| 单 expert 吸收全部 token | `padded` 只有一个非零段，`total = ceil(numel/B)*B` |
| `E ≫ numel`（如 8 路由 / 384 expert） | 绝大多数 lane 的计数为 0；填充量（P≈24K）主导 GM 流量，kernel 时间由填充与发射决定 |
| `B = 1` | `p_e = cnt_e`，`total = numel`，`NB = P = numel + 0`；`p_shift = 0`（移位退化为恒等），已机器验证 |
| `E = 1` | `E_pad = 64`，单个有效 lane；`expert_ids` 全 0（或全 `num_experts` 当 `total=0`） |

## 10. 交付清单

### 10.1 目录结构

```
examples/moe_permute_align/_make_align_kernel/
├── DESIGN.md            # 本文件
├── RETROSPECTIVE.md     # Stage 1 复盘（Phase 8）
├── verify_equiv.py      # §1.6.1 等价性机器验证（可重跑，退出码 0 = 全 PASS）
└── {op}.py              # Stage 3 产出（kernel + golden + 分层测试），本阶段不创建
```

### 10.2 文件清单

| 文件 | 责任 | 状态 |
|---|---|---|
| `DESIGN.md` | 本设计（迁移分析 §0 + 调研/优化 §1.6 + API/内存/Tiling/循环/同步/验证/风险） | 已产出 |
| `verify_equiv.py` | 基线 vs 优化后公式 vs 第三方参考的三方数值对照（torch CPU，int64/fp64） | 已产出并执行（全 EQUIV_PASS） |
| `RETROSPECTIVE.md` | 复盘章节（Skill Flow Issues / Value Point Proposals / Transferable Lessons） | 已产出 |
| `{op}.py` | kernel（路径 S/L 两个工厂 + 组合调用）+ golden + L0/L1/L2/Boundary | Stage 3 |

### 10.3 命名规范

- kernel 工厂：`_make_align_kernel(numel, num_experts, block_size)`（与脚手架 import 名一致，不得改名）。
- 编译程序内部命名：`_align_single`（路径 S）、`_hist_partial` / `_align_partial`（路径 L）。
- 缓冲名：`flat`、`sorted_token_ids`、`expert_ids`、`num_tokens_post_pad`、`cumsum`（GM，与 wrapper 位置参数
  顺序一致）；UB 统一 `s_` 前缀；GM scratch 为 `partial`。

### 10.4 实现顺序

1. `_align_single`（路径 S）：先把直方图 → 补齐 → 扫描 → 写回跑通（L0-1/L0-5/L0-6/L0-7）；
2. 路径 S 的 rank-count `expert_ids` 与哨兵填充（L0-2）；
3. `_hist_partial` + `_align_partial`（路径 L）+ 组合可调用对象（L0-3/L0-4/L0-8）；
4. 路径判定阈值与 `G_h`/`G_a` 分核参数（§5.5）；
5. 交给 Stage 4：按 §1.6.3 裁决计划做「uint16 lane / GM 原子合并」A/B。
