# _make_small_batch_kernel 算子设计文档（MoE 对齐小批融合路径）

> 迁移任务（harness 子模式）。源算子：`/workspace/TileOPs/src/tileops/_kernels/moe/permute_align.py`
> 中的 `_make_small_batch_kernel`（第 222–340 行，GPU TileLang / CUDA target）。目标：Ascend950PR_9579 /
> `ascend` 工具链（TileLang 0.1.15 + CANN 9.2.1，`target="ascend"`，`import tilelang.ascend.language as T`）。
> 设计范围：仅小批融合路径；`_make_align_kernel` 与 `_make_scatter_kernel` 是同批次另两个设计任务，
> 本文只在 §0.1 说明分工与输出契约一致性。
>
> **编程模式**：**SIMT**（`T.SimtVF(threads=32)` + `T.Parallel`）。平台 `programming_modes.migration_default`
> 为 `simd`；偏离理由与逐项证据见 §2.2（本算子的核心操作是数据相关地址的标量散射写与 UB 原子自增，
> SIMD 寄存器面 `T.simd.vscatter` 只能做 UB 内散射，无 GM 散射、无原子）。
>
> **算法族**：**计数 → 补齐前缀 → 放置**的单遍索引路线（§1.6.0 选定 C2），与源算子同族；被重设计的是
> 「并行组织」与「放置手段」，不是数学路线。
>
> **roofline 口径**：manifest 声明 `flops: "0"`（纯整型索引重排，无算术），全文按**延时 + 访存 + 发射**论证，
> 不使用 MFU。950 真机性能常数当前**不存在**（`bottlenecks.md` 明示「没有 950 真机验证的瓶颈模式」），
> 定量估算引用 910B2C 实测常数并显式标注外推（§1.6.0 R4）。
>
> **设计期未做设备验证**：设计期只执行**无卡编译/静态探针**（`ASCEND_RT_VISIBLE_DEVICES=""`，不 launch
> 任何 kernel），证据来自 ① 本机选中工具链的 language 模块源码；② 两个设计期探针的**降级设备代码**
> （`history_version/design_probe_forms.py`、`history_version/design_probe_refined.py`）；③ CANN 9.2.1
> 交付的 SIMT 设备 API 头文件；④ 同 op 另两条路径的已冻结设计（作 API 取证的一致来源）。
> 凡未经设备执行的结论显式标注「待 Stage 3 真机确认」。

## 0. 源算子解读与迁移分析

> 组织方式：先彻底读懂源算子（0.1–0.4），再做算法调研（结论落 §1.6.0，是 M1 判定与重设计的输入），
> 再判定硬件耦合性（0.5），最后给出 NPU 算法重设计（0.6）。本章结论驱动 §1–§11。

### 0.1 源算子语义（做什么）

`_make_small_batch_kernel(numel, num_experts, block_size)` 是 `MoEPermuteAlignKernel` 的**第三条构建路径**：
当 `numel < 1024` 且 `num_experts <= 32`（`WARP_LANES`）时，`forward` 只启动**一条** fused kernel，
一次写满三个输出——即把 align 段（计数 / 补齐前缀 / `expert_ids` / 哨兵）与 scatter 段（路由落槽）
在**同一个 block 内**完成，省掉一次 launch 与 `cumsum` 的 GM 往返。

**数学语义**（与 align + scatter 两段路径的输出契约完全相同）：

设 `flat = topk_ids.flatten()`（`int32[numel]`，取值 ∈ `[0, num_experts)`）、`E = num_experts`、
`B = block_size`、`P = numel + (E+1)(B-1)`、`NB = ⌈P/B⌉`：

```
cnt_e    = #{i < numel : flat[i] == e}                       e ∈ [0, E)
p_e      = ⌈cnt_e / B⌉ · B
cs_0 = 0 , cs_{e+1} = cs_e + p_e                             （独占前缀）
total    = cs_E                                              → num_tokens_post_pad[0]
owner(b) = e  ⟺  cs_e ≤ b·B < cs_{e+1}   (b·B < total)       → expert_ids[b]
owner(b) = E                             (b·B ≥ total)
sorted_token_ids[ slot_e(r) ] = i_e(r)                       （每个 expert 的路由索引）
sorted_token_ids[j] = numel                 （未被任何路由占用的槽位 = 哨兵）
```

其中 `i_e(r)` 是 expert `e` 的第 `r` 个路由的路由索引，`slot_e(r) ∈ [cs_e, cs_e + cnt_e)` 两两互异。
**expert 内部的排列顺序按 op 契约未定义**（源实现用 lane 到达顺序；参考实现用索引升序；测试
`_assert_route_equivalent` 先把每个 expert 区段归一化再比对）——这是本设计可以采用「原子到达顺序」的依据。

**语义要素**（源码逐项核对）：

- **规约语义**：整数计数与整数前缀和，**无浮点累加、无舍入**；累加/结合顺序不影响结果 ⇒ 并行重排无精度风险。
- **dtype 语义**：全程 `int32`；`P ≤ numel + 33(B-1)`（本路径 `E ≤ 32`）；`numel < 1024` ⇒ 所有中间量 < 2^31。
- **边界语义**：① `num_experts > 32` 时 factory `raise ValueError`（host 侧，wrapper 已拦截）；② `numel = 0`
  **确实会进入本路径**（wrapper 判据是 `numel < 1024 and num_experts <= 32`，不看 `numel > 0`；层测用例
  `no-routes-few-experts` 即 `numel=0, E=8, B=16`）⇒ 三个输出必须写成契约值（`total=0`、`expert_ids` 全 `E`、
  哨兵全 `0`）；③ 某 expert 计数为 0 ⇒ 其区段为空，不写；④ `numel` 不被 32 整除由尾 lane 守卫处理。
- **I/O 契约**：`flat` 只读；三个输出全部为 `int32` 索引数组（无自由 dtype）。**没有 `cumsum` 参数**——
  前缀和是本 kernel 的 UB 内部量（与 align 段不同）。输出 shape 只由 `(numel, E, B)` 决定，与数据无关。
- **host 侧语义**：`max_padded` / `max_num_blocks` 计算、三个 `torch.empty` 分配、按 `(numel, E, B)`
  特化构建（`target_cache`）——均为实现手段，NPU 侧由冻结 wrapper 承担（§4.7）。

### 0.2 源算子输入输出

| 张量 | 方向 | shape | dtype | 说明 |
|---|---|---|---|---|
| `flat` | in | `[numel]`（`numel = T·K`，编译期常量） | int32 | 路由 id，`0 ≤ v < num_experts` |
| `sorted_token_ids` | out | `[P]` | int32 | 每个 expert 区段的前 `cnt_e` 槽放路由索引，其余放哨兵 `numel` |
| `expert_ids` | out | `[NB]` | int32 | 每 `B` 行的 owner；越过补齐总长的 block 写 `num_experts` |
| `num_tokens_post_pad` | out | `[1]` | int32 | `total = cs_E` |

源算子输出 shape（任务给定，与另两条路径一致）：`sorted_token_ids [P]`、`expert_ids [⌈P/B⌉]`、
`num_tokens_post_pad [1]`，`P = T·K + (num_experts+1)·(block_size-1)`。

本路径的规模上界：`numel ≤ 1023`、`E ≤ 32`（`B` 无上界，由调用方给定）。取 `B = 64` 时
`numel=1023, E=32` ⇒ `P = 3102`、`NB = 49`（本路径的最大 workload）。

### 0.3 实现算法解读（怎么算）

**源码结构化阅读**（`_make_small_batch_kernel` + host 侧 `forward`）：入口 `_small()` →
`T.Kernel(1, threads=fill_threads + worker_threads)`，即**单 block、288 线程**（256 个 fill 线程 +
32 个 worker 线程，每个 worker 对应一个 expert）；`threads` 不是 launch 参数（构建期常量）。

**计算步骤分解表**（源码每条计算语句均归入某步骤）：

| # | 步骤 | 计算 | 输入 | 输出 | 对应语义 |
|---|---|---|---|---|---|
| 1 | 哨兵填充（fill 线程，`tx < 256`） | `for i in serial(⌈P/256⌉): idx = i*256+tx; if idx<P: sorted[idx] = numel` | — | GM `sorted` | 全槽位哨兵 |
| 2 | 私有行清零（worker `wid = tx-256`） | `for i in serial(E): tokens_cnts[(wid+1)*E + i] = 0` | — | SMEM | 直方图初值 |
| 3 | 私有行计数（worker） | `for i in serial(⌈numel/32⌉): idx = wid + i*32; if idx<numel: tokens_cnts[(wid+1)*E + flat[idx]] += 1` | GM `flat` | SMEM 行 `wid+1` | `cnt_e` 按 lane 分片 |
| 4 | 行前缀（worker，`wid < E`） | `tokens_cnts[wid] = 0`；`for k in serial(32): tokens_cnts[(k+1)*E + wid] += tokens_cnts[k*E + wid]` | SMEM | SMEM | 行 `k` = lane `k` 的**首 rank**，行 32 = `cnt_e` |
| 5 | 补齐前缀（`wid == 0`） | `for e in serial(E): cumsum_s[e+1] = cumsum_s[e] + ⌈tokens_cnts[32E+e]/B⌉*B`；`npost[0] = cumsum_s[E]` | SMEM | UB / GM | `cs_e`、`total` |
| 6 | 越界 block 哨兵（fill 线程） | `for i: blk = i*256+tx; if blk ≥ cumsum_s[E]//B and blk < NB: expert_ids[blk] = E` | UB | GM | `owner(b) = E` |
| 7 | block→expert（worker，`wid < E`） | `e_start = cumsum_s[wid]//B; e_blocks = cumsum_s[wid+1]//B - e_start; for b in serial(e_blocks): expert_ids[e_start+b] = wid` | UB | GM | `owner(b)` |
| 8 | 散射（worker） | `for i in serial(⌈numel/32⌉): idx = wid + i*32; if idx<numel: rank = tokens_cnts[wid*E + flat[idx]]; sorted[rank + cumsum_s[flat[idx]]] = idx; tokens_cnts[wid*E + flat[idx]] += 1` | SMEM + UB | GM `sorted` | `sorted[cs_e + r] = i` |
| 9 | host | `max_padded`/`max_num_blocks` 计算、三个 `torch.empty`、`target_cache` 特化缓存、`num_experts > 32` 抛错 | — | — | 构建/缓存 |

**数据流图（源 GPU 视角）**

```
GM[flat] --(每 lane 一条路由, idx = wid + i·32)--> SMEM tokens_cnts[33][E]
       (行 wid+1 = lane wid 的每 expert 计数；行 0 清零)
SMEM[33][E] --(沿行轴 32 步前缀, lane↔expert)--> SMEM[k][e] = lane k 首 rank, 行 32 = cnt_e
行 32 --(÷B 取整 + 串行前缀, 仅 lane 0)--> UB cumsum_s[E+1] / GM npost
UB cumsum_s --(÷B + 变长循环)--> GM expert_ids
SMEM 行 wid --(读 rank, 自增)--> 寄存器 slot --(散射写)--> GM sorted_token_ids[slot]
(并行) GM sorted_token_ids <--(逐线程条件写常量 numel)--  [步骤 1]
```

- **循环与并行结构**：grid = **1 block**；block 内 288 线程分两组（256 fill / 32 worker，`tx == 256+wid`）；
  worker 侧 `⌈numel/32⌉` 轮 token 循环 + `E` 长度 expert 循环；**无 warp shuffle、无 `T.copy`、无流水、
  无 matrix 单元**（与 align 段不同：本 kernel 不含 shuffle 扫描）。
- **保留的中间量**：`tokens_cnts[33][E]`（≥ 1056 int32）、`cumsum_s[E+1]`。
- **host 侧逻辑**：`max_padded`/`max_num_blocks`；`num_experts > WARP_LANES` 抛错；`target_cache(maxsize=32)`
  按 `(numel, E, B)` 缓存；`fill_threads=256`、`worker_threads=32` 为构建期常量。

**完成标准核对**：`_small_main` 的全部语句均落入步骤 1–8（含两条 `if` 分流与三个 `T.sync_threads`）；
数据流图覆盖 `tokens_cnts`、`cumsum_s`、`sorted_token_ids`、`expert_ids`、`num_tokens_post_pad` 与
寄存器 `rank/slot/e_start/e_blocks` 全部 buffer；无臆造步骤。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 源码特征 | 目的 | 依赖的源硬件特性 | 初判 |
|---|---|---|---|---|---|
| O1 | **一条 launch 完成两段** | 单 `T.Kernel(1, threads=288)`，不落 GM 中间量 | 省一次 launch + 省 `cumsum` 的 GM 往返（align+scatter 路径要两次 launch） | 单 CTA 内即可完成全局归约 | **算法层**（可移植，是路径存在的理由） |
| O2 | **每 lane 私有计数行**（lane ↔ token 分片） | `tokens_cnts[(wid+1)*E + eid] += 1`，行地址只由 `wid` 决定 | 计数**不需要原子**：每行只有一个写者 | SMEM 随机字写 + 32 线程私有分区 | 硬件相关（线程私有行） |
| O3 | **行轴前缀 + lane↔expert 转置**（步骤 4） | `tokens_cnts[k+1][e] += tokens_cnts[k][e]`，`wid` 即 expert | 只用 32 步得到「每 lane 的每 expert 首 rank」与总数，避免 32×32 的二维扫描 | 32 lane 并行、lane 内串行 | 硬件相关（lane 数 = warp 宽） |
| O4 | **worker / fill 线程分工并发** | `tx < 256` 填充、`tx ≥ 256` 计数/散射 | 纯带宽的哨兵填充与串行的计数/扫描重叠 | 单 block 多线程 + `__syncthreads` | 模式可移植（映射需重设计：NPU 无 block 内 thread team） |
| O5 | **rank 即游标**（步骤 8） | `rank = tokens_cnts[wid*E+eid]++`，`slot = rank + cumsum_s[eid]` | 一次读改写得到落槽，无需第二趟前缀 | 线程私有行 ⇒ 无原子 | 算法层意图（可移植：任何「每 expert 游标」都可承载） |
| O6 | **每 lane 处理 `⌈numel/32⌉` 条路由**（stride 32） | `idx = wid + i*32` | 32 线程覆盖全部路由，行尾无需 padding | 32 线程 | 模式可移植（映射到 lane） |
| O7 | 构建期特化 | `target_cache` 按 `(numel,E,B)` 缓存，循环上界全静态 | 消运行期分支与元数据 | host | 可移植（D9） |
| O8 | 变长 `expert_ids` 写（步骤 7） | `for b in serial(e_blocks)`，迭代数数据相关 | 写数最少（= 有效 block 数） | 线程级发散容忍 | 实现层（数据相关 trip count，反向量化） |

**算法层 vs 实现层**：O1/O5 属算法层（单 launch 融合 + 游标放置），迁移优先保留其意图；
O2/O3/O4/O6/O8 属实现层（私有行、lane↔expert 转置、线程分组、stride 分片、变长写），需逐项判定（§0.5）。
**注意**：本 kernel **不含** warp shuffle（shuffle 只在 align 段出现）、不含 `T.copy`/SMEM tile 搬运、
不含流水——故上述 8 项即其全部显式优化意图，识别清单（migration-analysis §4.1）逐项核对后无遗漏。

### 0.5 硬件耦合性分析与 NPU 适配决策

判定输入含 §1.6.0 调研结论（调研先于判定）。目标平台事实（`tilelang.ascend.language` + 本平台 ruleset
`programming.md` + 本任务两个探针的降级设备代码）：

- `T.Kernel(N)` 是**一维 AI core 网格，签名没有 `threads=`**；线程域由 `T.SimtVF(threads=...)` 在核内声明
  （`ascend/language/kernel.py`）。
- 片上层级：UB 248KB（`T.alloc_shared`）/ L1 512KB / L0C 256KB；本算子不用 Cube、不用 L1/L0。
- 无 warp；`T.shfl_*` 属公共面，本设计不需要。
- **UB 原子存在**：`T.atomic_add(shared_buf[i], 1, return_prev=True)` 编译通过并降级为
  `asc_atomic_add((&(s[key[...]])), 1)`（探针 K6）；CANN 头文件
  `/usr/local/Ascend/cann-9.2.1/asc/include/simt_api/device_atomic_functions.h` 第 25 行声明
  `asc_atomic_add(__ubuf__ int32_t*, int32_t) -> int32_t`（`__NPU_ARCH__ == 3510`）。真机语义待 Stage 3 确认。
- **数据相关 UB 下标访问会插入核内 barrier**：探针 K5/M1 的降级代码在 `s_list[e*cap + run] = i` /
  `atomic_add(&s_cs[eid], 1)` 与同循环的 UB 读之间插入 `asc_syncthreads()`，并打印
  `[ThreadSync] Hoisting sync out of an if ...`。**这是硬约束**（见 D3/D4 的处置）。
- **跨核无 barrier / flag 未认证**：`ascend_cross_core_set_flag/wait_flag` 只有接口声明，
  `ascend.batch_schedule` 的跨 block barrier 记为能力缺口 `CG-950-batch-barrier`。

| # | 源条目 | 层 | 处置 | 依据 |
|---|---|---|---|---|
| D1 | O1 单 launch 融合（align+scatter 合体） | 算法 | **保留** | 纯算法结构；NPU 上同样只需一次 launch（§0.6 R1） |
| D2 | 计数 + 补齐前缀 + `owner` 映射 + 游标放置的数学路线 | 算法 | **保留** | §0.1 是唯一基线；§1.6.0 选定 C2 与之一致 |
| D3 | O2 每 lane 私有计数行（`tokens_cnts[(wid+1)*E+eid] += 1`） | 实现 | **舍弃**（意图由 D4/D5 承接） | 数据相关 **UB 读改写** 使编译器在每个**循环迭代**插一个核内 `asc_syncthreads()`：lane↔token 形式 = `⌈numel/32⌉`（≤32）个 barrier（探针 M1/M2 实测 19 个），lane↔expert 逐 token 形式 = `⌈numel/UNROLL⌉`（≤256，探针 K4 实测 600 个）；并在发散守卫上打印 `[ThreadSync] ... the race remains` 告警。另需 `(W+1)×E` 的 UB 中间态（≤4.2 KB） |
| D4 | O3 行轴前缀 + lane↔expert 转置 | 实现 | **等价替换** | 「lane↔expert」本身在 NPU 上成立（`T.SimtVF(threads=32)`），但源式需要先有 32 行计数矩阵；改为**直接** lane↔expert 比较累加（§0.6 R2），省掉行矩阵与转置 |
| D5 | O5 游标放置（rank 即游标） | 算法 | **保留 + 等价替换** | 意图保留；rank 不再来自私有行读改写，而是 `atomic_add(cursor[e], 1, prev)`（UB 原子，int32 已取证） |
| D6 | O4 worker / fill 线程分工并发 | 实现 | **重新设计** | NPU 核内没有 256+32 的 thread team 语义，且**核间不能同步**；而 fills 写满 `[0,P)`（含 scatter 的目标槽）⇒ 若把填充放到另一个核，填充与散射发生**跨核写-写竞争**（无 barrier 可排序）⇒ 填充必须与散射**同核**，作为前序 bulk store（§0.6 R1） |
| D7 | O6 stride 32 的 lane 分片 | 实现 | **等价替换** | `T.Parallel` 的隐式 lane 映射即 `threadIdx.x`（探针降级码），`idx = r*32 + j` 等价于 `wid + i*32` 的重排 |
| D8 | O8 变长 `expert_ids` 写 | 实现 | **重新设计** | 数据相关 trip count 阻碍向量化/并行；改为 rank-count 恒等式（§0.6 R4，与 align 段同一处置，机器验证见 §1.6.1 A2） |
| D9 | O7 构建期特化 | 实现 | **保留** | 两侧都按 `(numel, E, B)` 特化；NPU 侧循环上界全静态 |
| D10 | `threads`（288）与 `T.sync_threads()` 三处 | 实现 | **舍弃** | `T.Kernel` 无 `threads`；核内 UB 依赖由默认流水（AutoSchedule/InsertSync）排序（探针 M9 的 sync 链即证据），本设计不写显式同步 |

**不确定项**：D5 的 UB 原子在**真机**上的吞吐与「返回旧值」语义只有编译期与头文件证据，无执行证据 ⇒
不作为「一定更快」的论断，主选/备选与裁决计划见 §1.6.3。

### 0.6 NPU 算法重设计

重设计对象：D3（私有行）、D4（转置）、D6（线程分工）、D8（变长写）+ 并行结构；承接 §1.6.0 选定算法族
（C2：计数 → 补齐前缀 → 放置）。源优化意图 O1–O8 逐项承接见表末。

**R1 并行结构重设计（源：1 block × 288 线程，fill/worker 分组 → NPU：1 core，填充为同核前序）**

- 源方案：单 block；256 个 fill 线程写哨兵，32 个 worker 线程计数/扫描/散射；block 内 `__syncthreads`
  作为分组间的屏障（哨兵在散射之前完成）。
- NPU 新算法：`T.Kernel(1)`，**一核**。核内顺序为：① 哨兵填充（`T.Parallel` 常量写 tile +
  `T.copy` bulk store，MTE3）→ ② 载入路由 tile（`T.copy` GM→UB）→ ③ 直方图 → ④ 补齐 → ⑤ 前缀 →
  ⑥ `expert_ids` → ⑦ 散射。
- **为什么不能把填充放到别的核**（关键判定）：填充写满 `[0, P)`，而散射也要写其中 `[0, total)` 的前段槽位；
  两者的先后必须由屏障保证。NPU 跨核 barrier 未认证（`CG-950-batch-barrier`），因此**不能**像源算子那样
  把填充交给并行线程组 ⇒ 填充降级为同核前序，由默认流水插入 `asc_sync_notify/wait(PIPE_MTE3, PIPE_V)`
  排序（探针 M9 降级码：填充的 3 次 `asc_copy_ub2gm_align` 之后，`asc_sync_wait(PIPE_MTE3, PIPE_V)` 正好
  落在散射的 VF 调用之前）。
- 语义保持论证：填充只写常量 `numel`，散射只写 `[0, total)` 内互异的槽位；把两者串到同一核不改变
  「哨兵 = numel（除被路由占据的槽）」这一结果，也不改变 `expert_ids`/`npost`。`numel = 0` 时哨兵常量
  折叠为 0，与参考实现一致（§1.6.1 A7 机器验证）。
- 代价：单核承担全部工作（≈3.4K 向量指令，§1.6.0 R4 roofline），但本路径的存在理由就是**省 launch**
  （O1），不是利用多核；`numel < 1024` 时单核工作量 ≤ 3.4K 指令 ≈ 2.4 µs（估算），低于两次 launch 的固定开销。

**R2 直方图重设计（源：32 私有行 scatter-add → NPU：lane↔expert 比较累加）**

- 源方案：`tokens_cnts[(wid+1)*E + eid] += 1`——每 lane 一行，靠「行私有」免原子。
- NPU 新算法：**lane ↔ expert**（`T.SimtVF(threads=32)`），每个 lane 持有一个私有累加器，对 token tile
  逐 token 广播比较后累加 `[x_i == e]`；再把计数按 `UNROLL=4` 拆成 4 个独立累加器（A4）缩短依赖链：
  每个 lane 的循环上界 `CHUNK/4`，`CHUNK = align_up(numel, 64) ≤ 1024`。
- 等价性：`cnt_e = Σ_i [flat_i == e]` 是计数的定义式，与私有行 + 行求和的**结果逐位相同**（整数）；
  `UNROLL` 拆分只是整数加法的重新结合（§1.6.1 A4 机器验证 280 例全 EQUIV_PASS）。
- 语义边界：`CHUNK > numel` 的尾部用 `-1` 填充（不匹配任何 `e ≥ 0`），从而**热循环无边界守卫**。
- 代价：`CHUNK/4` 步 × 4（比较+选择+加）≈ 3.1K 指令（E=32、numel=1023），是单核链的主要部分。

**R3 前缀和重设计（源：32 步行前缀 + lane 0 串行补齐前缀 → NPU：UB 内 Hillis–Steele 向量扫描）**

- 源方案：先用 32 步把行轴累加成「每 lane 首 rank + 总数」，再由 lane 0 串行算 `E` 步补齐前缀。
- NPU 新算法：只保留**补齐之后**的一维前缀——`s_pad[e] = ⌈cnt_e/B⌉·B`（A1 移位形式）在 UB 内做
  `log2(32) = 5` 步包含式扫描（左 halo 补零），再左移一位得独占前缀 `s_cs[0..32]`；`total = s_cs[32]`。
  双缓冲轮换 `s_scan[64]`（右半数据、左半 halo 恒 0），每步一次 `T.copy` 回写。
- 语义保持论证：整数加法结合律 ⇒ 任意结合顺序结果相同；左 halo 恒 0 ⇒ 与带边界条件的扫描逐位一致；
  `e ≥ E` 的补位 lane 计数恒 0（比较永不成立）⇒ `p_e = 0` ⇒ `s_cs[e] = total`（定义值，不是残留）。
- 机器验证：A1+A3+A4 `cumsum` 与基线串行式 70 例全 EQUIV_PASS（§1.6.1）。

**R4 `expert_ids` 重设计（源：expert 主序变长 scatter → NPU：rank-count 恒等式）**

- 源方案：`for b in serial(e_blocks)`，`e_blocks` 数据相关；lane 之间不均衡。
- NPU 新算法：对输出 block 下标 `b` **并行**求
  `owner(b) = #{e ∈ [0,E) : s_cs[e] ≤ b·B} − 1`；`b·B ≥ total` 时写 `num_experts`。
  `BLK = align_up(NB, 64)` 个 lane，内层 `serial(E)` 为每 lane 的归约轴（上界取构建期常量 `E`）。
- 等价性（含零计数 expert）：`s_cs` 非减且区间 `[cs_e, cs_{e+1})` 互不相交地铺满 `[0, total)`，故
  `pos < total` 时恰有一个 `e` 满足 `cs_e ≤ pos < cs_{e+1}`，而 `#{e: cs_e ≤ pos} − 1` 就是该 `e` 的最大解
  （零长度区间不改变「最大满足者」），与源式「空区间不写」一致；`pos ≥ total` 时两式同判 `E`。
- 机器验证：A2 70 例（含 `E=1`、`B=1`、连续零计数、单 expert 独占）与参考实现的二分结果逐元素相同。

**R5 散射重设计（源：私有行 rank 读改写 → NPU：per-expert UB 游标原子自增）**

- 源方案：`rank = tokens_cnts[wid*E + eid]++`，`sorted[rank + cumsum_s[eid]] = idx`。
- NPU 新算法：**token 并行**（lane ↔ token，`idx = r*32 + j`，⌈numel/32⌉ 轮），
  `slot = atomic_add(s_cs[eid], 1, return_prev=True)`、`sorted[slot] = idx`；游标直接复用前缀缓冲
  `s_cs`（原子只触及 `[0, E) ⊆ [0, 32)`，**不会改写** `s_cs[32] = total`，而 `total` 在此之前已写入
  `num_tokens_post_pad`）。路由 id 从 **GM** `flat[idx]` 读取（而非 UB `s_x`）：这样原子循环内不存在
  UB 读-写别名，探针 M5/M6 的降级码因此**没有**核内 barrier（对比 M1/M2 读 UB 时插入了 `asc_syncthreads()`）。
- 语义保持论证：对任一 expert，游标从 `cs_e` 单调 +1 共 `cnt_e` 次 ⇒ 恰好覆盖 `[cs_e, cs_e+cnt_e)`、
  两两互异、界内；未被路由占用的槽保持哨兵。与源式的区别只是 expert **内部顺序**（源 = lane 到达序，
  新 = 轮次到达序），而该顺序按 §0.1 契约未定义 ⇒ **契约级等价**。机器验证（A5）：70 例的
  「每 expert 多重集相等 / 槽互异 / 槽区间 / 哨兵保持 / 游标终值 = 起点+count」全部 EQUIV_PASS，
  且与参考实现的归一化比较一致。
- 边界：`numel = 0` ⇒ 轮数 0（构建期折叠），不动任何游标。

**R6 同步重设计（源：3 处 `T.sync_threads()` → NPU：默认流水 + 单核顺序）**

- 源方案：`sync_threads()` 分隔「填充 / 计数 / 扫描 / 散射」四段。
- NPU 新算法：全部阶段在同一核内按程序顺序执行，依赖由 AutoSchedule/InsertSync 自动插事件排序
  （探针 M9 的 `asc_sync_notify/wait` 链：`V(s_fill)→MTE3(填充)→V(散射)`、`MTE2(载入)→V(直方图)`、
  `V→MTE3(expert_ids 写回)`）；**不写任何显式同步**，不使用跨核 flag。
- 语义保持论证：依赖图与源一致，只是「线程同步」换成「核内流水同步」。风险与退路见 §7.3。

**源优化手段意图承接核对**（不得静默丢弃）：O1→R1（单 launch 保留，跨核填充改为同核前序）；
O2→R2（私有行舍弃，意图「免原子计数」由「lane↔expert 私有累加器」承接，同样零原子）；
O3→R3（32 步行前缀保留为 5 步 HS 扫描，意图「log/线性少步数」保留）；O4→R1（并发意图让位于
「跨核无屏障」的硬约束，并给出量化：填充在单核上 ≤ 3 次 4KB bulk store）；O5→R5（游标放置保留，
承载手段换成 UB 原子）；O6→R5（lane 分片保留为 `r*32 + j`）；O7→D9 保留；O8→R4（「少写」让位于
「可并行/可向量化」，量化对比见 §1.6.0 R3）。

### 0.7 标杆实现

- 源算子：`/workspace/TileOPs/src/tileops/_kernels/moe/permute_align.py` 中的 `_make_small_batch_kernel`。
- 上游来源：sglang/sgl-kernel `moe_align_block_size_small_batch_expert_kernel`（源码 docstring 标注）；
  vLLM `moe_align_block_size` 为同语义公开实现。
- NPU 侧参考实现（golden 依据，**独立于本设计**）：`examples/TileOPs/tileops/workloads/moe.py` 的
  `ref_permute_align`（`bincount` + `cumsum` + 二分查找），纯 Python 循环版是 §8.1 golden 的移植源。
- 同 op 另两条路径的已冻结设计（仅作 API/容量取证与契约核对，不照抄）：
  `examples/moe_permute_align/_make_align_kernel/DESIGN.md`（simd，HS 扫描 + rank-count）、
  `examples/moe_permute_align/_make_scatter_kernel/DESIGN.md`（simt，单级 bump）。
## 1. 概述

### 1.1 算子名称

`_make_small_batch_kernel`（项目 `moe_permute_align`；产品语义：MoE token→expert 对齐的**小批融合路径**，
等价于 `moe_align_block_size` 的「计数 + 补齐 + 散射」在单次 launch 内完成）。

### 1.2 功能描述

对 `numel < 1024` 且 `num_experts ≤ 32` 的路由调用，用**一条 kernel**写出与 align+scatter 两段路径
完全相同的三个输出：

1. 直方图：`cnt_e = #{i : flat_i == e}`；
2. 补齐 + 独占前缀：`p_e = ⌈cnt_e/B⌉·B`、`cs_{e+1} = cs_e + p_e`、`total = cs_E`；
3. `num_tokens_post_pad[0] = total`；
4. `expert_ids[b] = owner(b)`（越过 `total` 的 block 写 `num_experts`）；
5. `sorted_token_ids`：先把全部 `P` 个槽写成哨兵 `numel`，再把每条路由索引写进其 expert 区段的
   `[cs_e, cs_e + cnt_e)` 内（**互异槽位，expert 内部顺序未定义**）。

适用场景：MoE decode / 小 prefill 的 token 分发（每步路由数 < 1024 且专家数 ≤ 32）。性能口径：
`flops = 0`（纯整型索引重排），按延时 / 访存 / 发射评估。

本 kernel **不产生 `cumsum` 输出**（与 align 段不同：前缀和是 UB 内部量），三个输出与另两条路径逐项同契约。

### 1.3 数学公式

基线（源算子语义，与 §0.1 一致；`flat ∈ [0,E)^numel`）：

```
cnt_e    = Σ_{i<numel} [ flat_i == e ]                        e = 0..E-1
p_e      = ceil(cnt_e / B) · B
cs_0 = 0 ,  cs_{e+1} = cs_e + p_e                             (独占前缀)
total    = cs_E                                               → num_tokens_post_pad[0]
out_s[j] = numel                                              (哨兵, j = 0..P-1)
owner(b) = e  s.t. cs_e ≤ b·B < cs_{e+1}   (b·B < total)       → expert_ids[b]
owner(b) = E                               (b·B ≥ total)
放置:     sorted[ cs_{flat_i} + rank_i ] = i                  (rank_i 两两互异、∈ [0, cnt_{flat_i}))
P = numel + (E+1)(B-1) ,  NB = ceil(P / B)
```

本设计采用的等价改写（论证与机器验证见 §1.6.1；A1–A5 为采纳项）：

```
A1  p_e      = ((cnt_e + (B-1)) >> p) << p         ，B = 2^p（否则回退 div/mul）
A2  owner(b) = (#{e ∈ [0,E) : cs_e ≤ b·B}) − 1     (b·B < total) ；否则 E
A3  cs       = (Hillis-Steele 包含式扫描(p) 左移一位)，log2(W) = 5 步
A4  cnt_e    = Σ_j acc_j，acc_j 为 4 条独立累加链（整数结合律）
A5  sorted[ atomic_add(cs_{flat_i}, 1)ₚᵣₑᵥ ] = i   （游标自增取旧值，槽位互异、界内）
```

### 1.4 算法描述

**选定算法族**（§1.6.0 调研结论 C2）：**计数 → 补齐前缀 → 游标放置**的单遍索引路线，
**单核、单次 launch**，分为 7 个核内阶段（§3.1）：

- **P0 哨兵填充**：UB 常量 tile（`T.Parallel(1024)` 写 `numel`）→ 按 1024 元素分块 `T.copy` 到
  `sorted_token_ids[0:P)`（MTE3 bulk store）。
- **P1 载入路由**：`T.copy` 把 `flat[0:numel)` 搬进 UB `s_x[CHUNK]`，尾部 `[numel, CHUNK)` 用 `-1` 填满
  （不匹配任何 expert），使热循环无边界守卫。
- **P2 直方图**：`lane ↔ expert`（`T.SimtVF(threads=32)`），每 lane 一个私有累加器，4 路独立累加
  （A4），逐 token 广播比较累加 → `s_cnt[32]`。
- **P3 补齐**：`s_cnt[e] = ((s_cnt[e] + B-1) >> p) << p`（A1）。
- **P4 前缀**：UB 内 5 步 Hillis–Steele 包含式扫描 + 左移一位 → `s_cs[0..32]`（A3），`total = s_cs[32]`。
- **P5 `expert_ids`**：rank-count 恒等式（A2），`BLK = align_up(NB, 64)` 个 lane，内层 `serial(E)`。
- **P6 散射**：`lane ↔ token`（`idx = r·32 + j`，⌈numel/32⌉ 轮），路由 id 取自 **GM** `flat[idx]`，
  `slot = atomic_add(s_cs[eid], 1, prev)`，`sorted[slot] = idx`（A5）。

与源算法的结构差异（一句话）：源的「32 私有计数行 + 行轴转置 + 私有行 rank 读改写 + fill/worker 线程分组」
被换成「lane↔expert 比较累加 + 5 步 HS 扫描 + per-expert UB 游标原子自增 + 同核前序填充」，
原因见 §0.5 / §0.6，量化依据见 §1.6.0。

### 1.5 数据流图

```
GM flat[numel] ──MTE2(T.copy)──> UB s_x[CHUNK]（尾 -1）
                                    │
                     lane↔expert 比较累加 ×(CHUNK/4) 步（A4）
                                    ▼
                              UB s_cnt[32] ──A1 移位──> UB s_cnt[32]
                                    │
                              5 步 HS 扫描（A3, s_scan[64] 双缓冲）
                                    ▼
                     UB s_cs[0..32] ──┬──> GM num_tokens_post_pad[0]
                                    │  └──> A2 rank-count × NB ──> GM expert_ids[0:NB]
                                    ▼
             GM flat[idx] ──(P6, lane↔token 读)──> UB 原子 s_cs[eid]++ ──slot──> GM sorted[slot] = idx
UB s_fill[1024] ──T.Parallel 常量──> ──MTE3(T.copy, 分块)──> GM sorted_token_ids[0:P]（哨兵, 先于 P6）
```

不使用 L1 / L0A / L0B / L0C（无矩阵运算）；GM↔UB 全部经 `T.copy`（MTE2/MTE3），
P6 的散射写是逐 lane 标量 store。

### 1.6 算法调研与优化分析

#### 1.6.0 算法调研（Algorithm Research）

调研对象：**同一数学语义（§1.3 基线公式）的算法族**——所有能算出同样 `sorted_token_ids` /
`expert_ids` / `num_tokens_post_pad` 的算法结构（expert 内部顺序按契约不计）。源算子（GPU 实现）
只作为**基线候选**参与比较。调研深度：**完整调研**（本算子属「多步 + 规约 + 索引分发」类）。

参考表命中情况：平台 `knowledge.algorithm_candidates` 的 7 个算子族（elementwise / reduction /
softmax / RMSNorm-LayerNorm / GEMM / attention / GEMM+ReLU）**无索引分发类条目**；`kb_search`
预注入结论为无命中（任务上下文给出）。故候选集合由源算子证据（§0.3/§0.4）、本仓 `examples/` 检索、
同 op 另两条路径的已冻结设计、以及结构判据构成，逐条附本地证据。

**R1 等价化简公式（候选表，含基线）**

| # | 候选（公式 / 结构） | 等价性初判 | 收益方向 | 本地证据 |
|---|---|---|---|---|
| C1 | **基线**：32 私有计数行（lane↔token）+ 行轴转置前缀 + 私有行 rank 读改写 + fill/worker 线程分组 | —（基线） | — | 源算子 §0.3 步骤 1–8；`examples/TileOPs/tileops/kernels/moe/moe_permute_align/_moe_permute_align_fwd_kernels.py` |
| C2 | **本设计**：lane↔expert 比较累加（A4）+ 5 步 HS 扫描（A3）+ rank-count `expert_ids`（A2）+ per-expert 游标原子自增放置（A5） | 数学恒等（整数；A5 为契约级等价，§1.6.1 机器验证 70 例 × 5 分布全 PASS） | 消掉「数据相关 UB 下标访问」与「核内 barrier」；UB 4.2KB → 8.8KB 但无矩阵中间态；无原子计数 | §0.5 能力事实 + `PL-950-row-primitives.md`（SIMT lane 并行已验证）+ 探针 K6/M5/M9 |
| C3 | **单遍 staging**：一趟把每条路由写进 UB 内 per-expert 桶 `s_list[e*cap + r] = i`，前缀后按 expert 做变长 `T.copy` 输出 | 恒等（同一多重集，顺序同为主序） | 少一趟 token 扫描；GM 写变为 ≤E 次连续拷贝 | 探针 K4/M4 降级码：per-lane 数据相关 UB 写使 `asc_syncthreads()` **落在 token 循环内**（600 次 + `[ThreadSync] Hoisting sync ...` 警告）；UB 峰值 `32×numel×4 = 128 KB` |
| C4 | **原子直方图 + 原子放置**：两趟都用 `atomic_add`（lane↔token） | 恒等（整数原子加，顺序无关） | 计数与放置都变成 ⌈numel/32⌉ 轮 | 探针 K6 降级为 `asc_atomic_add(&s[eid],1)`；每趟 ⌈numel/32⌉ 个 barrier（C1 同源） |
| C5 | **无原子放置**（备选）：P2 不变，P6 改 lane↔expert 逐 token 计数器（索引升序） | 契约级等价（顺序 = 索引升序，落在契约允许的集合内） | 不依赖 UB 原子；代价 = numel 步串行链 | 探针 K1/M4（同形代码）降级码无 barrier、无警告 |
| C6 | **排序 + 游程**：按 expert 排序后连续段直接输出 | 恒等（排序不变量） | 复杂度 O(n log n) | 本工具链无已验证 sort API（`ascend/language/` 无 sort 导出）→ 淘汰 |
| C7 | **Cube one-hot 直方图**：构造 `[numel, E]` one-hot 再乘全 1 向量 | 恒等（`onehot·1` = 计数） | 计数搬上 28 个 Cube 核 | one-hot **构造**才是主体：广播比较 = `numel·E` = 32K element-ops（与 C2 的 1024 条指令等价，因 32 lane 并行），中间态 32 KB（int8）进 UB 后还要 32B 分形搬 L1/L0 → 净增搬运，无收益；`gemm-tiles.md` 的 L1/L0 路径约束 |
| C8 | **复用两段路径**（align + scatter，即另两个 kernel 工厂） | 恒等（同契约，本路径的对照） | 无需融合 kernel | 多 1 次 launch（≈2 µs 级固定开销，`CONST-vector-launch-overhead`〔910B2C〕外推）+ `cumsum[E+1]` 的 GM 写读；这正是源算子设小批路径的理由（O1） |

**R2 在线算法（单遍 / O(tile) 中间态）**

- 结论：**有**——单遍 staging 变体（C3）在结构上成立（一趟扫描同时计数与暂存 per-expert 桶），
  但它需要 `O(numel·E)` 的 UB 桶（128 KB）+ 每 token 的数据相关 UB 写，实测降级码显示**每次迭代插一个
  核内 barrier**（探针 K4/M4）⇒ 已评估并淘汰（理由见 C3 行，不是「未考虑」）。
- 结构依据：`cnt_e` 可分解为 running 计数（源算子本身就是单遍累加的直接证据，§0.3 步骤 3）；
  `cs` 是长度 E 的一维前缀 ⇒ 天生「先局部后合并」两段式，中间态 O(E)。
- 收益口径（代表 workload `numel=1023, E=32, B=64`）：`flat` 的 **GM 扫描遍数** —— C1/C2 = 2
  （P1 一次 `T.copy` 4,092 B + P6 逐 lane 读 GM 4,092 B，后者 L2 驻留）；C3/C5 = 1（P6 读 UB `s_x`，4,092 B）；
  **中间缓冲峰值** —— C2/C5 = `s_x 4 KB + s_cs/s_cnt/s_scan/s_ids ≈ 0.8 KB`，C3 = `s_x 4 KB + 桶 128 KB`。
  C3 的 2 KB 级流量优势不足以抵消 128 KB 占用与 1023 个 barrier。

**R3 复杂度对比（四口径 + 附加维度）**

口径说明（代表 workload `numel=1023, E=32, B=64`，`P=3102`，`NB=49`，`CHUNK=1024`，`W=32`）：

- **FLOPs** 记「标量元素操作数 element-ops」（一次整数比较或加法 = 1）；**指令数 = element-ops ÷ 并行 lane 数**
  （32 lane 同时执行一条 SIMT 指令）。P2 每 token 每 lane 是「1 比较 + 1 选择 + 1 加」= 3，
  `CHUNK/4 = 256` 轮 × 4 = 1,024 次迭代 ⇒ **3,072 条**。
- **GM 访存（B）** 只算 GM：读 `flat` 4,092×2 + 写哨兵 12,408 + 写路由（1,023 次 4B store，落在已写页面内）
  4,092 + `expert_ids` 196 + `npost` 4 = **24,884 B ≈ 24.3 KB**（C1/C2/C5/C8 的对外流量同量级）。
- 「扫描遍数」= `flat` 的 GM 读次数；「中间缓冲峰值」= 片上常驻（GM scratch 单列）。

| 候选 | FLOPs（element-ops；≈指令数） | GM 访存（B） | flat 扫描遍数 | 片上中间缓冲 | 可并行度 | 跨核归并代价 |
|---|---|---|---|---|---|---|
| C1 基线（源结构直译） | ≈1,024（私有行 RMW 2×1,023 + 行前缀 32×32）≈ **200 条指令 + 64 个核内 barrier** | 24,884 | 2 | 4.2 KB（`33×E` 矩阵） | 1 核（32 lane） | 无（单核） |
| C2 本设计（主选） | P2 3,072 + P6 128 + 扫描 110 + `ids` 64 + 填充 11 ≈ **3,385 条指令，0 barrier** | 24,884 | 2 | 8.8 KB | 1 核（32 lane） | 无（单核） |
| C3 单遍 staging | 一趟 1,023 次 UB 索引写 ≈ 1,023 条 + **1,023 个 barrier** + 32 次变长拷贝 | 20,792 | 1 | **128 KB** | 1 核 | 无 |
| C4 原子直方图 | 2×1,023 原子 + 2×32 轮 ≈ 2,110 次 UB 原子 ≈ 66 条指令 + 64 barrier | 24,884 | 2 | 8.0 KB | 1 核 | 无 |
| C5 无原子放置（备选） | P2 3,072 + P6 3,072（numel 步串行）+ 其余 ≈ **6,300 条指令，0 barrier** | 20,792 | 1 | 8.9 KB | 1 核 | 无 |
| C6 排序路线 | `n·log₂n` ≈ 10K + 游程 | 24,884 + 中间态 | ≥2 | 排序缓冲 ≫ UB | 取决于 API | — |
| C7 Cube one-hot | 构造 32K element-ops（≈1,024 条）+ MAC 32K | 24,884 + one-hot 32 KB 写读 | 1（+中间态） | 32 KB + L1/L0 分形 | 28 Cube（构造段仍占 vector） | 需 L1/L0 搬运 |
| C8 两段路径 | align ≈ 300 条 + scatter ≈ 100 条 | 24,884 + `cumsum` 132 | 2 | ≈10 KB（两 kernel 各一份） | 单核起 | **2 次 launch** |

**R4 硬件亲和（逐候选评估 + 设计期 roofline 定量）**

| 候选 | 计算单元匹配 | 片上容量 | 对齐/整除 | 静态边界 | 流水可融合 | 跨核结构 | 结论 |
|---|---|---|---|---|---|---|---|
| C1 | ~ 全部落 vector 核，但含 2×1,023 次**数据相关 UB 读改写** | ✓ 4.2 KB | ✓ `E≤32` 整除 32 lane | ✓ 全静态 | ✗ barrier 打断流水（64 个） | ✓ 单核 | **淘汰（结构风险）**：ThreadSync 在发散守卫上把 barrier 提到 if 外并显式告警「the race remains」（探针 M1/M2 输出）⇒ 正确性依赖未认证行为（§9 风险 3） |
| C2 | ✓ 比较 / 加法 / UB 原子全落 vector 核 | ✓ 8.8 KB ≪ 253,952 B | ✓ `W=32`、`FILL_TILE=1024`、`BLK` 均 64 对齐 | ✓ 全静态（`numel=0` 构建期折叠） | ✓ 填充 MTE3 与 P2/P3/P4 的 V 重叠（探针 M9 的 sync 链） | ✓ 单核 | **选定** |
| C3 | ~ 索引写落 vector 核，但每步 barrier | ✗ 128 KB（占 UB 50%） | ✓ | ✓ | ✗ barrier 打断 | ✓ | 淘汰（容量 + barrier） |
| C4 | ~ UB 原子吞吐未实测（`asc_atomic_add` 只到编译证据） | ✓ | ✓ | ✓ | ✗ barrier（每趟 ⌈numel/32⌉ 个） | ✓ | 弃选（barrier 同 C1；原子总数 2×numel 并不比 C2 的 3.1K 比较指令更省，且引入未实测常数） |
| C5 | ✓ 同 C2（多一趟串行链） | ✓ 8.9 KB | ✓ | ✓ | ✓ | ✓ | **备选**（不依赖 UB 原子；指令 1.9×） |
| C6 | ✗ 无 sort API | ✗ | — | — | — | — | 淘汰（API 缺失） |
| C7 | ✓ Cube 有矩阵能力 | ✗ 32 KB 中间态 + 分形 | 需 32B 分形 | — | ✗ 构造段与矩阵段无法同缓冲融合 | 需跨核 | 淘汰（构造无更省路径） |
| C8 | ✓ | ✓ | ✓ | ✓ | — | ✓ | 淘汰（+1 launch，正是本路径要省的） |

**设计期 roofline（D-2，代表 workload = 本路径最大档 `numel=1023, E=32, B=64`）**

- **流量项**：24,884 B ÷ `CONST-mte2-degradation` 的 L2 驻留口径 43.1 GB/s/core（**910B2C 实测，跨型号外推
  至 950PR，未实测**）= **0.58 µs**；按同条目 HBM 口径 18.4 GB/s/core 保守估 = **1.35 µs**。
- **发射项**：3,385 条 SIMT 指令（明细：P2 3,072 + P6 128 + P5 64 + P4 110 + P0 11）。按「1 条向量指令 / cycle、
  1.4 GHz」⇒ 3,385/1.4e9 = **2.42 µs**（**未实测假设**，估算依据：CCE 向量单元单发射，950PR 主频平台
  记录未给出；`CONST-vector-launch-overhead` 只测到「tile 级 op ~0.5 µs 固定开销」，不适用于逐指令口径）；
  深度受限口径：P2 的 4 路独立链每轮 3 个依赖步 ⇒ 256×3 = 768 cycle = **0.55 µs**（作为乐观下界）。
- **容量项**：UB 峰值 `s_x 4,096 + s_fill 4,096 + s_cnt/s_scan/s_cs/s_ids 772` = **8,964 B** /
  253,952 B（平台 `design_budget.capacity_bytes.UB`，`Ascend950PR_9579.ini`，md5 2f0eca1f…）= **0.035**。
- 淘汰依据（引用常数/文档）：C3 的 128 KB 桶对照 `CONST-capacity-910B2C` 的 1.7× 膨胀上界（910B2C 实测，
  外推）⇒ 217 KB 逼近 248 KB；C6 无 sort API（`ascend/language/` 无导出）；C1/C4 的 barrier 与
  数据相关 UB 访问证据见探针 K5/M1 的降级码与 `[ThreadSync]` 告警。
- **估算下界: ≈2.4 µs（= max(流量 0.58〔L2 口径〕～1.35〔HBM 口径〕, 发射 2.42, 容量 0.035)，引用
  CONST-mte2-degradation / CONST-capacity-910B2C〔均 910B2C 实测外推〕）**；加 1 次 kernel launch
  （常数未实测，按 ≈2 µs 估）后 ≈ **4.4 µs**。
  小档（`numel=8, E=4, B=4`，`P=23`）估算：流量 0.004 µs、发射 ≈320 条/1.4 GHz = 0.23 µs ⇒
  **launch 开销主导（≈2.2 µs）**——这正是「单 launch 融合」在此档位的全部价值（C8 需 2 次）。

**调研结论**

选定 **C2**（计数 → 补齐前缀 → 游标放置，单核单 launch；lane↔expert 比较累加 + HS 扫描 + rank-count +
per-expert 游标原子自增）：R1 中它与基线**契约级等价**（A5 顺序自由度已由 op 契约覆盖，§1.6.1 机器验证）；
R2 的单遍变体（C3）存在但被容量与 barrier 证据淘汰（不是「无在线变体」的断言）；R3 上它的指令数高于
C1 的 ~200 条，但 C1 的 64 个 barrier + 数据相关 UB 访问带 ThreadSync 告警级风险，且 C2 的墙钟下界
（2.4 µs，发射项主导）与 C1 同量级；R4 逐候选检查中 C2 与 C5 存活，C2 以「一趟短链 + 无原子计数」胜出，
C5 作为唯一备选进入 §1.6.3 的裁决计划。

**「无更优替代」的调研范围**（供 Stage 2 复核）：已查 ① 平台 `knowledge.algorithm_candidates` 全部 7 个
算子族条目（无索引分发型）；② `kb_search "MoE permute align histogram prefix-sum scatter int32"`（预注入无命中）；
③ 本仓 `examples/TileOPs/tileops/kernels/` 的 7 个算子族目录（`moe` 只有本算子族脚手架，无索引分发参照）；
④ `knowledge/` 的 PL-950 条目（row-primitives / vector-execution / atomic-splitk / simd-reduce /
TRAP-950-simd-shared-reduce / execution-patterns / numerically-boundaries）；⑤ 结构判据（排序类 C6 因 API 缺失 +
复杂度淘汰；one-hot C7 因构造无更省路径淘汰；单遍 C3 因容量 + barrier 淘汰；两段 C8 因 launch 淘汰）。
因此「不存在比 C2 更优且在本工具链可实现的算法族」这一判断的边界是上述五项检索 + 三项结构判据。
源算子已识别优化手段 O1–O8 的承接逐项见 §0.6 末尾核对表。

#### 1.6.1 数学等价优化（公式级）

分析对象 = §1.6.0 选定的 C2（经 §0.6 重设计落地）。逐项四要素如下；等价性**全部机器验证**
（`verify_equiv.py`，纯 Python 整数精确算术〔等价于 int64/fp64 精确参照，无 torch 依赖〕，14 组 `(numel,E,B)` × 5 种分布 = 70 例/检查，
覆盖 `numel=0`、`E∈{1,3,4,8,32}`、`B∈{1,4,16,32,48,64}`、非 2 的幂 `B=48`、单 expert 独占、连续零计数）。

**A1 补齐由「除法」改「移位」（`B = 2^p` 时）**

- 原式：`p_e = ceil(cnt_e / B) · B`（整数除法 + 乘）
- 优化后：`p_e = ((cnt_e + (B-1)) >> p) << p`；`B` 非 2 的幂时回退 `ceildiv·B`（构建期判定，`B=48` 用例覆盖）
- 等价性：`cnt ≥ 0` 时 `ceil(cnt/B) = (cnt + B - 1) >> p`（右移即下取整除），随后左移乘回；非负域内整数恒等，无舍入。
  机器验证：7 个 `B` × `cnt ∈ [0,1024)` 共 7,168 例，0 差。
- 收益：每 expert 由「1 除法 + 1 乘」降为「1 加 + 2 移位」。整除在向量单元无硬件指令（**未实测假设**，
  估算依据：CCE 无整数除法指令）；E ≤ 32 时绝对量小（32 lane 一次完成），价值在于去掉长延迟除法链。

**A2 `expert_ids` 由「expert 主序变长 scatter」改「rank-count 恒等式」**

- 原式：`for e: for b in [0, cs_{e+1}/B − cs_e/B): expert_ids[cs_e/B + b] = e`（trip count 数据相关）
- 优化后：`expert_ids[b] = (#{e ∈ [0,E): cs_e ≤ b·B}) − 1`（`b·B < total`），否则 `num_experts`
- 等价性：`cs` 非减、区间 `[cs_e, cs_{e+1})` 互不相交且铺满 `[0, total)`，`cs_0 = 0 ≤ pos` ⇒ 满足
  `cs_e ≤ pos` 的 e 构成前缀 `{0..k}`，`k+1 = 计数`，而 `pos < total` 时该 k 正是包含 pos 的区间右端
  （零长度区间不改变「最大满足者」），与源式「空区间不写」一致；`pos ≥ total` 两式同判 `E`。
  机器验证：70 例与基线变长循环逐元素相同，且与参考实现（二分）在 `[0, total/B)` 上逐元素相同。
- 收益：消掉数据相关 trip count ⇒ 每个 b 独立、可并行（32 lane 覆盖 32 个 block）；比较数 `E·NB`
  = 32×49 = 1,568 element-ops ≈ 64 条指令/核，远低于源式的「49 次串行写 + 每 expert 一次判断」。

**A3 独占前缀由「32 步行前缀 + lane 0 串行补齐」改「5 步 Hillis–Steele + 左移一位」**

- 原式：先 32 步行累加（得到每 lane 首 rank 与总数），再由 lane 0 串行 `E` 步算 `cs`；总深度 32+E = 64 步
- 优化后：`s_pad[e] = p_e` 直接做 `log2(W) = 5` 步包含式扫描 `s[e] += (e ≥ 2^d ? s[e−2^d] : 0)`（左 halo 恒 0），
  再左移一位得 `s_cs[0..32]`；深度 5 步
- 等价性：整数加法结合律/交换律 ⇒ 任意结合顺序结果相同；halo 恒 0 ⇒ 与带边界条件的扫描逐位一致；
  补位 lane（`e ≥ E`）的 `p_e = 0` ⇒ `s_cs[e] = total`（定义值）。机器验证：70 例 `cumsum` 与基线逐元素相同。
- 收益：深度 64 → 5 步；指令 `5×32/32 = 5` 条扫描 + 5 次 UB 回写（每次 32 个 int32 → 编译器折叠为 16 个 `int2`
  拷贝，探针降级码可见）。**阶梯级削减且延迟链由 O(E) 降到 O(log W)**。

**A4 直方图由「单累加器」改「4 路独立累加器」（归约结构优化）**

- 原式：`acc += [x_i == e]`（单一依赖链，深度 = numel/1）
- 优化后：`acc_j += [x_{4i+j} == e]`，`j = 0..3`，末尾 `cnt_e = Σ_j acc_j`
- 等价性：整数加法结合律（重新结合求和顺序不改变整数和）；无溢出（`cnt_e ≤ 1023 < 2^31`）。
  机器验证：`UNROLL ∈ {1,2,4,8}` 共 280 例与 `bincount` 逐元素相同。
- 收益：依赖链深度 1,024 → 256（**4×**），指令总数不变（3,072）；探针 M1/M5 降级码保留 `acc[4]`
  + 160 轮外层（`CHUNK=640` 时），确认编译器未把 4 路重新串行成一条链。

**A5 放置由「私有行 rank 读改写」改「per-expert 游标原子自增」（契约级等价）**

- 原式：`rank = tokens_cnts[wid*E + eid]++`，`sorted[rank + cs_eid] = idx`（rank 来自 32 行矩阵的转置结果）
- 优化后：`slot = atomic_add(s_cs[eid], 1, return_prev=True)`，`sorted[slot] = idx`（游标即前缀本身）
- 等价性（契约级）：对任一 expert，游标从 `cs_e` 单调 +1 共 `cnt_e` 次 ⇒ 分配到的槽**恰好覆盖**
  `[cs_e, cs_e + cnt_e)`、两两互异、界内；未被路由占用的槽保持哨兵。与源式的差别只是 expert **内部顺序**
  （源 = lane 到达序，新 = 轮次到达序），而该顺序按 §0.1 契约**未定义**（测试 `_assert_route_equivalent`
  先把每个 expert 区段归一化再比对）。机器验证：70 例的「每 expert 多重集相等 / 槽互异 / 槽区间 /
  哨兵保持 / 游标终值 = 起点 + count」全部相等；归一化后与参考实现一致。
- 收益：消掉 33×E 计数矩阵与行转置（UB 4.2 KB → 0）与 2×numel 次数据相关 UB 访问；放置变成
  ⌈numel/32⌉ 轮 lane 并行（32 轮 vs 1,023 步串行）。**代价**：`numel` 次 UB 原子（吞吐未实测，§9 风险 1）。

**A6 哨兵填充由「逐线程条件写」改「常量 tile + 分块 bulk copy」**（向量化项，见 §1.6.2 V1；此处只记等价性）

- 原式：`for i: idx = i*256+tx; if idx < P: sorted[idx] = numel`
- 优化后：`T.Parallel(FILL_TILE)` 写常量 tile → `T.copy(s_fill[0:n], sorted[off:off+n])` 分块
- 等价性：写入集合恒为 `[0, P)` 且值恒为 `numel`；分块边界由 `n = min(FILL_TILE, P-off)` 精确裁剪。
  机器验证：A5 的「哨兵保持」检查（70 例）与 A7 的 `numel=0` 检查共同覆盖。

**等价性机器验证结果（`verify_equiv.py` 执行输出，2026-11-23，本机 CPU）**

| 采纳项 / 检查 | 用例数 | 最大绝对差 | 最大 ulp 差 | 违反率 | 结论 |
|---|---|---|---|---|---|
| A1+A3+A4 `cumsum`（移位补齐 + HS 扫描 + 4 路累加）vs 基线串行式 | 70 | 0 | 0 | 0.000000 | **EQUIV_PASS** |
| `num_tokens_post_pad`（= `s_cs[32]`）vs 基线 | 70 | 0 | 0 | 0.000000 | **EQUIV_PASS** |
| `cumsum` 与基线自检 | 70 | 0 | 0 | 0.000000 | **EQUIV_PASS** |
| A2 `expert_ids`（rank-count）vs 基线 expert 主序变长循环 | 70 | 0 | 0 | 0.000000 | **EQUIV_PASS** |
| A2 `expert_ids` vs 参考实现（二分） | 70 | 0 | 0 | 0.000000 | **EQUIV_PASS** |
| A5 散射：每 expert 多重集 + 槽互异/界内 | 70 | 0 | 0 | 0.000000 | **EQUIV_PASS** |
| A5 哨兵在 expert 区段之外保持不变 | 70 | 0 | 0 | 0.000000 | **EQUIV_PASS** |
| A5 游标终值 = 起点 + count | 70 | 0 | 0 | 0.000000 | **EQUIV_PASS** |
| A5 归一化 `sorted_token_ids` vs 参考实现（+ 尾段哨兵） | 70 | 0 | 0 | 0.000000 | **EQUIV_PASS** |
| A1 移位补齐 vs `ceil(c/B)*B`（`B=2^p`，`c<1024`） | 7,168 | 0 | 0 | 0.000000 | **EQUIV_PASS** |
| A4 直方图（UNROLL 1/2/4/8）vs `bincount`（补位 lane = 0） | 280 | 0 | 0 | 0.000000 | **EQUIV_PASS** |
| A3 补位 lane 纪律（`E<W`）：归约上界 `E` 与 `W` 两种写法 | 110 | 0 | 0 | 0.000000 | **EQUIV_PASS** |
| A3 判别性对照（补位 lane = UB 垃圾，50 例）：界 E 错 0/50、界 W 错 50/50 | 50 | - | - | - | **EQUIV_PASS** |
| A7 `numel=0` 边界（4 组 `(E,B)`：三个输出均为契约值） | 4 | 0 | 0 | 0.000000 | **EQUIV_PASS** |

L0 地标（脚本内显式断言）：

```
(0,8,16)     uniform: total=0    cumsum MATCH True  expert_ids MATCH True  (哨兵全 0, ids 全 E)
(100,8,48)   uniform: total=384  cumsum MATCH True  expert_ids MATCH True  (B 非 2 的幂 -> div/mul)
(600,32,64)  uniform: total=2048 cumsum MATCH True  expert_ids MATCH True  (本路径最大 E)
(1000,3,1)   uniform: total=1000 cumsum MATCH True  expert_ids MATCH True  (B=1, E 非 2 的幂)
(1023,8,16)  uniform: total=1072 cumsum MATCH True  expert_ids MATCH True  (最后一档小批路由数)
A3 判别性对照: 补位 lane 填 UB 垃圾时, 归约上界取 E 仍全对(0/50), 取 W 全错(50/50) => 该不变量是承重的
RESULT: ALL EQUIV_PASS   (退出码 0, 约 0.6 s)
```

脚本：`examples/moe_permute_align/_make_small_batch_kernel/verify_equiv.py`（随设计留存，Stage 2 可重跑）。
全部量在整数域精确可表示（< 2^21），故「0 ulp 差」是**精确结论**而非容差结论；`num_tokens_post_pad` 等
三个输出与两段路径逐位一致。

**优化结论**：采纳 **A1、A2、A3、A4、A5** 五项（全部 EQUIV_PASS，整数逐位/契约级等价，无精度风险），
外加 A6 的向量化形态（§1.6.2 V1）。未采纳项：C3 单遍 staging（128 KB 桶 + 1,023 个 barrier，§1.6.0 R2/R4）、
C4 原子直方图（不减少原子总数且引入 barrier，§1.6.0 R4）、C7 Cube one-hot（构造无更省路径）。
**无进一步公式化简空间**：算子本体是「计数 + 前缀 + 一次放置」，A1–A5 已把除法、串行前缀链、数据相关
trip count、私有行矩阵、单累加链五处结构处理完毕；剩余工作量由 §1.6.2（向量化）与 §5（分核）承载。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

盘点范围：§3.3 实现方案中**全部**循环与标量计算点（P0–P6 各阶段）。

| # | 计算点 | 位置 | 处理 | 替代 API / 依据 | 判定 |
|---|---|---|---|---|---|
| V1 | 哨兵填充（源 `for i in serial(⌈P/256⌉)` + `if idx<P`） | P0 | **向量化替代**：`T.Parallel(FILL_TILE)` 一次写常量 tile（探针 M7 降级为 `make_int2` 向量 store，`__launch_bounds__(256)`），再按 `FILL_TILE` 分块 `T.copy` 到 GM（尾块 `n = min(FILL_TILE, P-off)`） | `T.Parallel` + `T.copy` region（`ascend/language/copy_op.py`；尾块写法见 `kernel_template.py`） | 已替代（逐元素条件写 → 常量 tile + bulk store） |
| V2 | 直方图累加（源 `for i: tokens_cnts[(wid+1)*E+eid] += 1`） | P2 | **向量化替代**：`T.Parallel(W)` 映射 lane↔expert，每 lane 私有累加器，内层 `serial(CHUNK/4)` 为其归约轴（A4 拆 4 链） | `T.SimtVF` + `T.Parallel`（`PL-950-row-primitives.md` 已验证 lane 并行 + 每 lane 私有统计量）；探针 M1/M5 降级码 | 已替代（数据相关 UB scatter-add → lane 并行比较累加） |
| V3 | P2 的内层 token 轴 `T.serial(CHUNK/4)` | P2 | **保留为串行轴**，理由：(a) 它是**每 lane 的归约轴**，与 lane 轴正交（一次迭代已覆盖全部 E 个 expert）；(b) 改为 token 并行需要对 `cnt[eid]` 做 scatter-add = **数据相关 UB 读改写**，实测插入逐轮 barrier 并触发 `[ThreadSync]` 告警（探针 K4/K5/M1）；(c) 结构判据：求和的索引轴与被求和的轴只能一个占 lane | 探针 K5/M1 降级码 + §0.5 能力事实 | 保留（API/结构 + 正确性风险） |
| V4 | 补齐 `p_e = ceil(cnt_e/B)·B`（源逐 expert 标量） | P3 | **向量化替代**：`T.Parallel(W)` 一次（A1 移位形式） | `T.Parallel` + 移位 | 已替代 |
| V5 | 前缀（源 32 步行前缀 + lane 0 串行补齐） | P4 | **向量化替代**：A3 的 5 步 `T.Parallel(W)` 扫描（左右 buffer 轮换 + 左 halo 补零） | `T.Parallel`；扫描只做相邻 lane 位移加，不做跨 lane 归约（规避 `TRAP-950-simd-shared-reduce`） | 已替代 |
| V6 | `expert_ids` 映射（源 `for b in serial(e_blocks)`） | P5 | **向量化替代**：A2 rank-count —— `T.Parallel(BLK)`（lane = 输出 block），内层 `serial(E)` 为每 lane 归约轴 | `T.Parallel` + `T.if_then_else` 掩码 | 已替代（数据相关 trip count 消失） |
| V7 | 越界 block 哨兵（源 `if blk ≥ total//B: expert_ids[blk] = E`） | P5 | **并入 V6**：A2 的末尾 mask 直接产出 `num_experts`，不再单独一遍扫描 | 同 V6 | 已替代（消掉一次独立扫描） |
| V8 | 散射写（源 `sorted[rank + cs] = idx`） | P6 | **向量化替代（lane 并行 + 谓词化 store）**：`T.Parallel(W)` + `if idx < numel` 守卫 + `atomic_add` 取槽；降级码为 `if (slot<... && 0<=slot) sorted[slot] = r*32+tx`（编译器自动加值域守卫） | 探针 M5/M6 降级码；`simd.py::vscatter` 只支持 **UB 内**散射（`base[index[lane]] = src[lane]`），不能做 GM 散射 ⇒ SIMT lane store 是唯一形态 | 已替代（1,023 次串行 → 32 轮 lane 并行） |
| V9 | `num_tokens_post_pad[0]` 写入 | P4 末 | **保留为标量写**，理由：单元素写，属 **block 级元数据**（每核 1 次），且必须与 `s_cs[32]` 同值，由同一处写出可省一次读 | `tl::write_gm_bypass_dcache`（探针降级码） | 保留（block 级，非热点） |
| V10 | 核/分片下标计算（`off`、`idx = r*W+j`、`b*B`） | 全阶段 | **保留为标量/表达式**，理由：**任务映射算术**，每核常数次或已折叠进 lane 索引（探针降级码 `r*32 + threadIdx.x`） | — | 保留（任务映射） |
| V11 | host 侧 `P`/`NB`/`CHUNK`/`BLK`/`W`/`UNROLL` | host | **保留**：host 元数据计算，是构建期常量折叠的输入 | — | 保留（host 元数据） |
| V12 | `numel=0` 的构建期分支（`if numel > 0:` 只包 `flat` 的 GM 读） | P1/P6 | **保留为 Python 分支**：`numel` 是特化常量，决定「是否存在 GM 读 / 是否有轮次」，属构建期折叠（运行期无分支）；**计数/扫描/写回链不在分支内** ⇒ 零输入时三个输出照常写出（§1.6.1 A7） | 探针 M3/M11（`numel=0` COMPILE_OK） | 保留（构建期边界折叠） |
| V13 | `s_x` 尾部的 `-1` 填充（`T.fill(s_x[numel:CHUNK], -1)`） | P1 | **保留为小循环**，理由：≤ 63 个元素的常量写（探针 M8 降级为 20 轮 × 2 个标量 store = 40 条），其作用是**消掉 P2 热循环的边界守卫**（换来 1,024 步无分支），比逐 token 判断更省 | 探针 M8 | 保留（以常量填充换取热循环无守卫） |

**向量化结论**：13 个循环/标量计算点中，V1/V2/V4/V5/V6/V7/V8 共 7 处由逐元素条件写、数据相关
scatter-add、串行前缀/变长 scatter 替换为 lane 并行向量操作；V3（每 lane 归约轴）保留理由为
「数据相关 UB 读改写会插入逐轮 barrier + 归约轴与 lane 轴正交」；V9–V13 为 block 级元数据、host 元数据、
构建期折叠与一个 ≤63 元素的常量尾填，理由逐项给出。**不残留任何随数据量增长的逐元素标量循环**：
所有随 `numel`/`E`/`NB` 增长的循环都在 lane 方向并行。

#### 1.6.3 向量化轴与数据布局决策

**候选枚举**（按算子类别：gather / scatter / 索引分发类；本算子同时含「常量填充」与「直方图归约」子结构）：

| 候选 | 结构 | 向量轴 / lane 语义 | 判据 |
|---|---|---|---|
| **L1（主选，P2/P3/P5）** | 直方图/补齐/`expert_ids`：`Parallel(W)` × `serial(tokens)` 垂直扫描 | lane ↔ expert；lane ↔ 输出 block | 每 lane 私有累加器，**无跨 lane 归约、无 UB scatter-add、无原子**；`W=32 = E_max` ⇒ 尾 lane 浪费率 `(32−E)/32`：E=32 → 0%，E=8 → 75%（但一次指令本来就覆盖 32 lane，见下） |
| **L1'（主选，P6）** | 散射：`Parallel(W)` × `serial(⌈numel/32⌉)` 轮，lane ↔ **token** | lane ↔ token | 一次 `atomic_add` 取槽 + 一次标量 store；轮数 32（vs 逐 token 的 1,023） |
| L2（备选，P6） | 散射：lane ↔ expert 逐 token 计数（同 L1 的轴，但 counter 保序） | lane ↔ expert | 零原子，但 1,023 步串行链（探针 K1/M4 降级码无 barrier）；作为 §1.6.3 的备选 |
| L3 | 散射：lane ↔ 输出槽，反查「第 r 个出现」 | lane ↔ slot | 需要按 rank 反查 token（gather 链 + 计数扫描），深度与 L2 同但多一次反查 → 淘汰 |
| L4 | 直方图：token tile × expert tile 的 2D 比较 + `T.reduce_sum(dim=0)` | lane ↔ token | 二维 shared→shared 归约有已验证的输出布局差异（`TRAP-950-simd-shared-reduce`）→ 淘汰 |
| L5 | 直方图/散射：token 并行 + per-lane 私有行（源结构直译） | lane ↔ token，行私有 | 数据相关 UB 读写在逐轮循环里插 barrier + ThreadSync 告警（探针 K5/M1）→ 淘汰（§1.6.0 R4 C1） |
| L6 | 哨兵：常量 tile（`Parallel(1024)` + 分块 `T.copy`） | lane ↔ 连续下标 | 尾块用 region 裁剪；无条件写 |

**逐候选评分（代表 workload）**

- 指令数：L1 的 P2 = 3,072 条（`CHUNK/4 × 4 × 3`）——注意 **lane↔expert 的 E=8 与 E=32 指令数相同**
  （一条指令覆盖 32 lane，空闲 lane 不增加指令），故「尾 lane 浪费率」在本算子的指令口径下不构成代价，
  只影响能量/带宽口径；L1' 的 P6 = 128 条；L2 的 P6 = 3,072 条且深度 1,023。
- 尾 lane 浪费：L1 在 E=32 为 0%；E=8 时空闲 lane 75%（不增指令）。
- UB 预算：8,964 B（§4.5），1.7× 膨胀上界下 ≈ 15 KB（6%）。
- 弃选量化理由：L3 多一次按 rank 的反查（深度 +1,023 步）；L4 命中已证实缺陷（`TRAP-950-simd-shared-reduce`）；
  L5 有 64 个 barrier + ThreadSync 告警（正确性风险）。
- **判定裕度**：L1' 与 L2 的差距依赖**两个未实测常数**——UB `int32` 原子的真实吞吐（只有编译证据与头文件声明）
  与「1,023 步标量链」的每步代价 ⇒ 按规则走**实验裁决模式**（下）。

**实验裁决模式（主选 + 备选 + 裁决计划）**

- **主选（进 Stage 3）**：L1 + L1' + L6 —— `T.Kernel(1)`；P2 用 lane↔expert 的 4 路累加；P6 用
  token 并行 + `atomic_add(s_cs[eid], 1, prev)` + GM `flat[idx]` 读。
- **备选（≤1，结构完整可实现）**：**L2（P6 改为 lane↔expert 逐 token 计数器，零原子）**。结构完整度：
  缓冲形状不变（`s_cnt` 复用为末计数、`s_run` 可省——计数器用 `T.alloc_local([1])`）；循环结构
  P6 改为 `Parallel(W)` × `serial(numel)`（探针 K1/M4 同形，COMPILE_OK 且无 barrier/无告警）；
  分核三要素不变（仍单核）；dtype 路径不变（int32）；代价 = 1,023 步串行链（指令 ≈6,300 条）。
  该备选可直接交 Stage 4 在 `perf_opt/` 下实现（基准不动）。
- **裁决计划**：代表 shape = `(numel=1023, E=32, B=64)`、`(numel=600, E=8, B=64)`、`(numel=200, E=8, B=16)`；
  指标 = 平台 `msopprof` 的 Task Duration（us），并反解两个未知常数（UB int32 原子吞吐 = `numel/Δt_atomic`；
  串行链每步代价 = `Δt_chain/numel`）；**判定阈值**：备选 latency < 主选 × 0.85 才翻转（全局）；
  若仅在小 `numel` 档胜出，则**按 shape 分派**（factory 输入 `numel` 是编译期常量，`numel < 512` 走备选、
  否则主选——本算子是逐 `(numel,E,B)` 特化的工厂，分派天然可行）；裁决结果回写本节并触发设计修订。
- 设计期**未**做探针标定这两个常数（无卡可用，且探针需真机执行原子与长链计时）⇒ 标注为未实测假设，
  交 Stage 4 裁决（设计期探针 D-3 只被用于形式的**可构建性**，见 §3.4）。

**布局决策结论**：主选 **L1（lane↔expert 垂直扫描）+ L1'（lane↔token 原子自增放置）+ L6（常量 tile 填充）**；
核内布局 = `UB s_x[CHUNK] int32`（`CHUNK = align_up(numel,64) ≤ 1024`）、`UB s_cnt[32]`、`UB s_scan[64]`
（左半 halo 恒 0）、`UB s_cs[33]`（有效 `[0,32]`）、`UB s_ids[BLK]`、`UB s_fill[1024]`；备选 L2 交 Stage 4
按上述裁决计划 A/B，不进入 §3.3 / §4 / §6 的结构描述。

## 2. 编程模式选型

### 2.1 模式结论

**编程模式：SIMT**（平台 `programming_modes.choices` 的 `simt` 键）。实现形态：一维 `T.Kernel(1)` 核网格 +
核内 `T.SimtVF(threads=32)` 的 lane 并行表达式（`T.Parallel` 映射到 `threadIdx.x`）；**不使用 Cube、不用
L1/L0、不用 `T.simd.*` 寄存器面**。

### 2.2 选型理由

1. **计算单元匹配**：本算子零矩阵运算（`flops = 0`），全部是整型比较/加法/前缀与**数据相关地址**的
   索引读写。数据相关地址（`sorted[slot]`、`s_cs[eid]` 的原子自增）只能由 SIMT lane 表达：
   本工具链 `simd.py` 的 `vscatter(src, base, index)` 的 `base` 必须是 **UB** 缓冲（docstring:
   `base[index[lane]] = src[lane]`），不能做 GM 散射；也没有寄存器面的原子。
2. **偏离 `migration_default = simd` 的依据**：`simd` 适合逐元素/行归约的寄存器向量链；本算子的
   核心两项（GM 散射写、UB 原子自增）在 SIMD 面**无对应 API**（同上）。同 op 的 `_make_scatter_kernel`
   设计亦以 SIMT 落地并有编译证据（`examples/moe_permute_align/_make_scatter_kernel/DESIGN.md` §2.2）。
   本设计的 `T.SimtVF` 形态已由两个探针编译验证（`__simt_vf__` + `threadIdx.x` 的降级码，§3.4）。
3. **混合模式无收益**：本 kernel 无矩阵段可融（无 `T.gemm`），混合（`T.Cube()` + `T.Vector()`）无对象。
4. **无跨核协作**：单核执行（§5.5），不需要跨核 flag / 双 AIV 协作。

### 2.3 模式影响

- 只需 UB 一种片上缓冲（`T.alloc_shared`）；内存规划退化为 GM↔UB 双向 `T.copy` + lane 标量 store（§4.4）。
- 同步退化为「核内流水自动同步」（§7），不出现 `ascend_cross_core_*`。
- `T.Kernel` 无 `threads=` 参数 ⇒ kernel 工厂不接受 launch 参数（wrapper 现状即如此，§4.7）。

## 3. API 映射设计

### 3.1 公式拆解

以 §1.3 的 A1–A5 改写式为准（§1.6.1 已机器验证），按数据依赖拆成 7 个核内阶段：

| 阶段 | 公式 | 数据规模 | 依赖 |
|---|---|---|---|
| P0 哨兵填充 | `sorted[0:P) ← numel`（常量 tile + 分块 copy） | `P`（≤ 3,102 @ B=64） | — |
| P1 载入路由 | `s_x ← flat[0:numel)`；`s_x[numel:CHUNK) ← -1` | `CHUNK ≤ 1024` | — |
| P2 直方图 | `cnt_e = Σ_j acc_j`（A4，`CHUNK/4` 轮 × 4） | 32 lane × 256 轮 | P1 |
| P3 补齐 | `p_e = ((cnt_e + B-1) >> p) << p`（A1） | 32 | P2 |
| P4 前缀 | A3：5 步 HS 包含式扫描 + 左移一位；`total = s_cs[32]` | 32 lane | P3 |
| P5 `expert_ids` | A2：`ids[b] = (#{e<E: s_cs[e] ≤ b·B}) − 1`（`b·B < total`），否则 `E` | `BLK` lane × `E` | P4 |
| P6 散射 | `slot = atomic_add(s_cs[flat_i], 1, prev)`；`sorted[slot] = i`（A5） | ⌈numel/32⌉ 轮 | P4（+P0 的写完成） |

### 3.2 TileLang API 映射

| 公式 / 阶段 | TileLang API（`import tilelang.ascend.language as T`） | 关键参数 / 说明 |
|---|---|---|
| 核网格 | `T.Kernel(1)` | 一维 core grid；**无 `threads=`**（`ascend/language/kernel.py`） |
| lane 并行域 | `T.SimtVF(threads=32)` / `T.SimtVF(threads=256)`（P0）/ `T.SimtVF(threads=64)`（P4 左移步） | `threads` 为构建期常量；探针已覆盖 32/64/256/288 形态（`dim3(32)` 等降级码） |
| lane 循环 | `T.Parallel(W)` / `T.Parallel(FILL_TILE)` / `T.Parallel(BLK)` | 一 lane ↔ 一个 expert / 一个输出下标 / 一个 block |
| 归约轴串行 | `T.serial(CHUNK//4)`（P2）、`T.serial(E)`（P5，上界 = 构建期常量 `E`）、`T.serial(⌈numel/32⌉)`（P6） | 每 lane 私有的扫描轴（§1.6.2 V3） |
| GM→UB / UB→GM | `T.copy(src_region, dst_region)` | P0 分块写、P1 载入、P5 写回 `expert_ids` |
| 片上缓冲 | `T.alloc_shared([n], "int32")` | UB（`shared.dyn`）；无 L1/L0 分配 |
| 常量填充（tile 初始化） | `T.Parallel(n)` 常量写 + `T.fill(s_x[numel:CHUNK], -1)` | 探针 M7/M8 降级码（向量 store / 小 unroll） |
| 每 lane 私有累加器 | `T.alloc_local([4], "int32")`（P2）/ `T.alloc_local([1], "int32")`（P5 的 cnt、备选 P6 的 run） | `T.alloc_var(...)` 返回标量、不可下标（同 op align 设计的探针 K4 已证否）→ 不用 |
| UB 原子自增取旧值 | `T.atomic_add(s_cs[eid], T.int32(1), return_prev=True)` | 探针 K6 降级为 `asc_atomic_add(&s_cs[eid], 1)`；CANN 头文件 `device_atomic_functions.h:25` 声明 `__ubuf__ int32_t*` 重载（arch 3510） |
| 逐 lane GM 散射写 | `sorted_token_ids[slot] = idx`（`T.Parallel` 体内） | 探针 M5/M6 降级为带值域守卫的 `sorted[slot] = r*32+tx` |
| 条件选择 / 掩码 | `T.if_then_else(cond, a, b)` | P2 的 mask-add、P5 的 `b·B ≥ total → E` |
| 整除 / 上取整 | `T.ceildiv(a, b)`（构建期用 `math.ceil`）；移位 `>>`/`<<` | 仅用于下标与 A1 |
| 边界 | `T.min`、`T.max` | P0 尾块裁剪 |
| 流水 | 不显式使用 `T.Pipelined` | 本 kernel 每阶段只有 1 块（`CHUNK ≤ 1024`），无多 chunk 循环 |

**反查记录（不使用的 API 与原因）**：`T.cumsum` —— 探针 K7 报
`tl.cumsum requires a target-specific implementation, but no cumsum implementation is registered for
{"kind":"ascend",...}`（实证不可用，故自实现 A3）；`T.simd.*`（`vld/vsts/vscatter` 为 UB 内寄存器面，
无 GM 散射与原子，§2.2）；`T.atomic_add` 的 GM 形式（本 kernel 无需 GM 原子：游标在 UB）；
`T.sync_threads()`（核内由默认流水排序，§7）；`T.Pipelined`（无 chunk 循环可叠）；
`T.reduce_sum(dim=0)`（二维 shared→shared 归约命中 `TRAP-950-simd-shared-reduce`，本设计不需要）。

### 3.3 计算伪代码

```python
# ============ 构建期（host，每 (numel, E, B) 一次） ============
numel, E, B = ...                       # numel < 1024, E <= 32
P          = numel + (E + 1) * (B - 1)
NB         = ceildiv(P, B)
W          = 32                         # lane 数 = E 的上界（一次指令覆盖 32 lane）
UNROLL     = 4                          # P2 的独立累加链数
CHUNK      = max(64, align_up(numel, 64))          # <= 1024
BLK        = align_up(NB, 64)                       # P5 的 lane 数
FILL_TILE  = 1024
FILL_LANES = 256
N_FILL     = ceildiv(P, FILL_TILE)
ROUNDS     = ceildiv(numel, W)                      # P6 轮数
p_shift    = log2(B) if B is power of 2 else None

# ============ 编译程序：_small() —— 1 launch, grid = 1 ============
def kernel(flat, sorted_token_ids, expert_ids, num_tokens_post_pad):
    with T.Kernel(1) as bx:
        s_x    = T.alloc_shared([CHUNK], "int32")      # 路由 tile（尾 -1）
        s_cnt  = T.alloc_shared([W], "int32")          # 计数 / 补齐后计数
        s_scan = T.alloc_shared([2 * W], "int32")      # HS 扫描：右半数据、左半 halo 恒 0
        s_cs   = T.alloc_shared([W + 1], "int32")      # 独占前缀，P6 中复用为游标
        s_ids  = T.alloc_shared([BLK], "int32")        # expert_ids 分片
        s_fill = T.alloc_shared([FILL_TILE], "int32")  # 哨兵常量 tile

        # ---- P0 哨兵填充（同核前序；见 §0.6 R1 为什么不能放别的核） ----
        with T.SimtVF(threads=FILL_LANES):
            for i in T.Parallel(FILL_TILE):
                s_fill[i] = numel
        for t in T.serial(N_FILL):
            off = t * FILL_TILE
            n   = T.min(FILL_TILE, P - off)
            if n > 0:
                T.copy(s_fill[0:n], sorted_token_ids[off:off + n])

        # ---- P1 载入路由（尾填 -1：不匹配任何 expert） ----
        T.fill(s_x[numel:CHUNK], T.int32(-1))
        if numel > 0:                                   # 构建期折叠：只包 GM 读
            T.copy(flat[0:numel], s_x[0:numel])

        # ---- P2 直方图：lane<->expert，4 路独立累加（A4） ----
        with T.SimtVF(threads=W):
            for e in T.Parallel(W):
                acc = T.alloc_local([UNROLL], "int32")
                for j in range(UNROLL):
                    acc[j] = 0
                for i in T.serial(CHUNK // UNROLL):
                    for j in range(UNROLL):             # python 级展开
                        acc[j] = acc[j] + T.if_then_else(s_x[i * UNROLL + j] == e, 1, 0)
                s_cnt[e] = acc[0] + acc[1] + acc[2] + acc[3]

        # ---- P3 补齐（A1；B 非 2 的幂回退 div/mul，构建期判定） ----
        with T.SimtVF(threads=W):
            for e in T.Parallel(W):
                if p_shift is not None:
                    s_cnt[e] = ((s_cnt[e] + (B - 1)) >> p_shift) << p_shift
                else:
                    s_cnt[e] = T.ceildiv(s_cnt[e], B) * B

        # ---- P4 前缀：5 步 HS 包含式扫描 + 左移一位（A3） ----
        with T.SimtVF(threads=W):
            for e in T.Parallel(W):
                s_scan[W + e] = s_cnt[e]
            for e in T.Parallel(W):
                s_scan[e] = 0                               # 左 halo
        for d in range(int(log2(W))):                       # python 级展开，禁 while
            k = 1 << d
            with T.SimtVF(threads=W):
                for e in T.Parallel(W):
                    s_cs[e] = s_scan[W + e] + s_scan[W + e - k]
            T.copy(s_cs[0:W], s_scan[W:2 * W])
        with T.SimtVF(threads=W + 32):
            for e in T.Parallel(W + 32):
                if e <= W:                                  # 写满 [0, W]：补位 lane = total
                    s_cs[e] = T.if_then_else(e == 0, 0, s_scan[W + e - 1])
        num_tokens_post_pad[0] = s_cs[W]                    # total

        # ---- P5 expert_ids：rank-count（A2，归约上界 = E） ----
        with T.SimtVF(threads=BLK):
            for b in T.Parallel(BLK):
                cnt = T.alloc_local([1], "int32")
                cnt[0] = 0
                for e in T.serial(E):
                    cnt[0] = cnt[0] + T.if_then_else(s_cs[e] <= b * B, 1, 0)
                s_ids[b] = T.if_then_else(b * B < s_cs[W], cnt[0] - 1, E)
        T.copy(s_ids[0:NB], expert_ids[0:NB])

        # ---- P6 散射：token 并行 + per-expert 游标原子自增（A5） ----
        #   路由 id 取自 GM flat[idx]（不读 s_x）：因此本循环内没有 UB 读-写别名，
        #   编译器不插核内 barrier（探针 M5/M6 降级码实测）。原子只触及 [0,E)，
        #   s_cs[W] = total 已在 P4 末写入 npost，不会被改写。
        with T.SimtVF(threads=W):
            for j in T.Parallel(W):
                for r in T.serial(ROUNDS):
                    idx = r * W + j
                    if idx < numel:
                        eid = flat[idx]
                        slot = T.atomic_add(s_cs[eid], T.int32(1), return_prev=True)
                        sorted_token_ids[slot] = idx
```

> **写法依据（全部经编译探针，见 §3.4）**：`T.alloc_local([4])` 的 4 路累加器、python 级
> `for d in range(5)` 的 HS 展开、`if e <= W` 的左移写满、`T.fill` 的 region 形式、
> `T.SimtVF(threads=32/64/256)` 的混合、`numel=0` 的构建期折叠、以及 P6 的原子+散射写
> 均 **COMPILE_OK**（脚本 `history_version/design_probe_refined.py` 的 M9/M10/M11）。
> `@T.prim_func` 内**禁止 `while`**（同 op align 设计的探针 K2：`Infinite while loop detected`），
> 故 HS 步数用 python `range` 展开。

### 3.4 API 可行性确认

| API | 证据（本工具链） | 状态 |
|---|---|---|
| `T.Kernel(N)`（一维、无 `threads=`） | `ascend/language/kernel.py:Kernel`；`references/programming.md` API 表；探针 M9 降级出 `block_idx` | 已确认 |
| `T.SimtVF(threads=...)` + `T.Parallel` | `ascend/language/frame.py:SimtVF`；`PL-950-row-primitives.md`（64 线程、行转换/归约/广播 24 项通过）；探针 M9 覆盖 32/64/256 并降级为 `asc_vf_call<...>(cce::dim3(32))` | 已确认（int32 lane 形态经编译验证；**真机 lane 吞吐待 Stage 3 确认**） |
| `T.alloc_shared` | `ascend/language/allocate.py`（UB，默认 `shared.dyn`） | 已确认 |
| `T.copy`（双向、运行期 region 长度） | `ascend/language/copy_op.py`；`kernel_template.py` 尾块写法；探针 M9 降级为 `asc_copy_ub2gm_align(..., min(1024, P-t*1024)*4, ...)` | 已确认 |
| `T.fill(buf_region, value)` | 探针 M8 降级为 20 个 `int2` store（`-1` 尾填）；region 形式可用 | 已确认 |
| `T.atomic_add(shared[i], 1, return_prev=True)`（int32） | 探针 K6 降级为 `asc_atomic_add((&(s[key[...]])), 1)`；CANN `device_atomic_functions.h:25`（`__ubuf__ int32_t*`，`__NPU_ARCH__ == 3510`） | 编译/头文件已确认；**真机语义与吞吐待 Stage 3** |
| `T.Parallel` 体内逐 lane GM 散射写 | 探针 M5/M6 降级为 `if ((slot<3102) && (0<=slot)) sorted[slot] = r*32+tx` | 已确认 |
| `T.alloc_local([k], "int32")`（`T.Parallel` 体内） | 探针 M9 降级为 `int32_t acc[4]` / `int32_t cnt[1]` | 已确认 |
| python 级 `for d in range(N)` 展开 HS | 探针 M9 降级为 `for (d=0; d<5; ++d) asc_vf_call<...>(..., d)` | 已确认 |
| `T.alloc_var(...)[0]` | 同 op align 设计探针 K4 `COMPILE_FAIL`（`'BufferLoad' object is not subscriptable`） | **不可用**（本设计不出现） |
| `@T.prim_func` 内 `while` | 同 op align 设计探针 K2 `COMPILE_FAIL`（`Infinite while loop detected`） | **不可用**（本设计用 python `for`） |
| `T.cumsum` | 探针 K7 `COMPILE_FAIL`：`tl.cumsum requires a target-specific implementation, but no cumsum implementation is registered for {"kind":"ascend",...}` | **不可用**（自实现 A3） |
| 数据相关 UB 下标读写在循环内 | 探针 K5/M1：插入 `asc_syncthreads()` 并触发 `[ThreadSync] Hoisting sync out of an if ...` 告警 | **规避**（本设计热循环无 UB 下标访问，§1.6.0 R4 C1 淘汰依据） |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 限制（本平台） | 本设计处置 | 结论 |
|---|---|---|
| Kernel 只支持**一维** core grid，且无 `threads=` | `T.Kernel(1)` 一维；线程域在核内用 `T.SimtVF` | ✓ 合规（源 `T.Kernel(1, threads=288)` 已重设计，D10） |
| 不得依赖未认证的**跨 block barrier / 跨核 flag** | 单核；不使用 `ascend_cross_core_*` / `sync_grid`；**填充与散射同核**（D6） | ✓ 合规 |
| 无 warp；shuffle 不适用 | 本 kernel 本就不含 shuffle（shuffle 只在 align 段）；前缀用 HS 扫描 | ✓ 合规 |
| 数据相关 UB 下标访问触发核内 barrier（+ 告警） | 热循环（P2/P6）零 UB 下标访问：P2 用 lane↔expert 比较累加，P6 读 GM `flat` + UB 原子 | ✓ 已规避（探针 M5/M6 无 barrier） |
| UB 容量 253,952 B/核（`ascend-950pr-9579` 设计预算） | 峰值 8,964 B（3.5%），1.7× 膨胀下 ≈15 KB | ✓ 合规 |
| 不使用 L1/L0C/L0A/L0B 时不得申请 | 全设计只 `T.alloc_shared` | ✓ 合规 |
| 尾块必须显式限定有效 region（`pad_value`/`data_select` 执行行为未验证） | P0 用 `n = T.min(FILL_TILE, P-off)` + `n > 0` 守卫；P5 用 `s_ids[0:NB]` / `expert_ids[0:NB]` | ✓ 合规 |
| 非整除 shape（`numel` 不整除 32/64、`P` 不整除 `FILL_TILE`、`NB` 不整除 64） | P0 尾块裁剪、P1 尾 `-1` 填充、P5 `BLK` 对齐 + `b*B < total` 掩码、P6 `idx < numel` 守卫 | ✓ 合规 |
| `T.cumsum` / 公共面原语不得假定可用 | 探针 K7 实证不可用 → 自实现 HS 扫描 | ✓ 已规避 |
| NPUIR 专属规则（三维 Kernel、reduction 宏、8KiB 预留、H200 列上限）不适用 | 未使用任何 NPUIR 规则，也未以 `examples/elementwise/atomic_add.py`（另一工具链）为证据 | ✓ 合规 |

#### 3.5.2 参考实现差异说明

| 维度 | GPU 源算子 | 本设计（NPU） | 差异原因 |
|---|---|---|---|
| 并行结构 | 1 block × 288 线程（256 fill + 32 worker） | 1 core × `T.SimtVF(32/64/256)`（分阶段） | 平台 launch 无 `threads`；核内无 thread-team 语义 |
| 计数 | 32 私有行 scatter-add（免原子） | lane↔expert 比较累加 + 4 路链（A4） | 数据相关 UB 读改写会插 barrier（探针 K5/M1，D3） |
| 前缀 | 32 步行前缀 + lane 0 串行 `E` 步 | 5 步 HS 向量扫描（A3） | 无 warp；行矩阵不再存在（D4） |
| `expert_ids` | expert 主序变长 scatter | rank-count 恒等式（A2） | 数据相关 trip count 无法并行（D8） |
| 放置 | 私有行 rank 读改写 + `cumsum_s` 偏移 | UB 游标 `atomic_add` 取旧值（A5） | 私有行被舍弃；UB int32 原子有编译/头文件证据（D5） |
| 哨兵填充 | 256 fill 线程与 worker 并发（block 内屏障排序） | 同核前序 bulk store（MTE3） | 跨核无屏障且写域重叠（D6） |
| 同步 | 3 处 `T.sync_threads()` | 无显式同步（核内自动流水） | 默认 AutoSchedule/InsertSync（D10） |
| launch 次数 | 1 | 1 | 保留 O1（本路径的存在理由） |

#### 3.5.3 本项目同类实现参考

- lane 并行 / 每 lane 私有统计量 / 行归约：`.agents/platforms/toolchains/ascend/knowledge/PL-950-row-primitives.md`
  （repro：`.agents/platforms/toolchains/ascend/knowledge/repro/PL-950-row-primitives.py`）。
- 向量执行与尾块读写：`.agents/platforms/toolchains/ascend/knowledge/PL-950-vector-execution.md`。
- GM 原子写回（对照本设计的 UB 原子）：`.agents/platforms/toolchains/ascend/knowledge/execution-patterns.md`
  的 `PL-950-atomic-splitk`（repro：`.agents/platforms/toolchains/ascend/knowledge/repro/PL-950-atomic-splitk.py`）。
- 平台 kernel 模板（一维 `T.Kernel` + `T.SimtVF` + 尾块 region）：
  `.agents/platforms/toolchains/ascend/kernel_template.py`。
- 同 op 两条路径的已冻结设计（API/容量取证与契约一致性）：
  `examples/moe_permute_align/_make_align_kernel/DESIGN.md`、`examples/moe_permute_align/_make_scatter_kernel/DESIGN.md`。
- **本仓无「单 launch 融合计数 + 散射」的同类实现**：已检索 `examples/TileOPs/tileops/kernels/` 下全部
  算子族目录（attention / elementwise / mamba / moe / norm / reduction / rope），`moe` 只有本算子族脚手架。
  故「同核前序填充 + UB 游标原子放置」这一组合为本项目首例，相关风险在 §9 显式登记。
## 4. 数据规格与内存规划

### 4.1 输入张量

| 名称 | shape | dtype | 动态轴 | 说明 |
|---|---|---|---|---|
| `flat` | `[numel]`，`numel = T·K < 1024` | int32 | 构建期特化（每个 `numel` 单独构建） | `topk_ids.flatten().contiguous()` 的视图；取值 `[0, num_experts)`；`numel = 0` 合法 |

### 4.2 输出张量

| 名称 | shape | dtype | 本 kernel 写入内容 |
|---|---|---|---|
| `sorted_token_ids` | `[P]`，`P = numel + (E+1)(B-1)` | int32 | 先全写哨兵 `numel`，再把每条路由索引写进其 expert 区段的前 `cnt_e` 槽 |
| `expert_ids` | `[NB]`，`NB = ⌈P/B⌉` | int32 | `[0, total/B)` 为 owner expert；其余写 `num_experts` |
| `num_tokens_post_pad` | `[1]` | int32 | `total = cs_E` |

本路径各用例的规模（`W=32`，`CHUNK = align_up(numel,64)`，`BLK = align_up(NB,64)`）：

| `(numel, E, B)` | 用例来源 | `P` | `NB` | `CHUNK` | `BLK` | 填充块数 | P6 轮数 |
|---|---|---|---|---|---|---|---|
| `(0, 8, 16)` | 层测 `no-routes-few-experts` | 135 | 9 | 64 | 64 | 1 | 0 |
| `(2, 4, 4)` | 层测 `single-token` | 17 | 5 | 64 | 64 | 1 | 1 |
| `(8, 4, 4)` | 层测 `tiny-bs4`（smoke） | 23 | 6 | 64 | 64 | 1 | 1 |
| `(32, 8, 16)` | 层测 `small-bs16` | 167 | 11 | 64 | 64 | 1 | 1 |
| `(64, 8, 1)` | `B=1` 边界 | 64 | 64 | 64 | 64 | 1 | 2 |
| `(100, 8, 48)` | `B` 非 2 的幂 | 523 | 11 | 128 | 64 | 1 | 4 |
| `(200, 8, 16)` | 层测 `sb-numel200` | 335 | 21 | 256 | 64 | 1 | 7 |
| `(600, 32, 64)` | 层测 `sb-numel600-maxexp` | 2,679 | 42 | 640 | 64 | 3 | 19 |
| `(1023, 8, 16)` | 层测 `routes-1023`（dispatch 边界） | 1,158 | 73 | 1,024 | 128 | 2 | 32 |
| `(1023, 32, 64)` | 本路径最大档（代表 workload） | 3,102 | 49 | 1,024 | 64 | 4 | 32 |

### 4.3 中间缓冲区

| 名称 | 位置 | shape | dtype | 用途 | 生命周期 |
|---|---|---|---|---|---|
| `s_x` | UB | `[CHUNK]`（≤1024） | int32 | 路由 tile；`[numel, CHUNK)` 填 `-1` | 每核一次载入，P2 只读 |
| `s_cnt` | UB | `[W]` = `[32]` | int32 | 每 expert 计数 → 就地改为补齐后计数 | P2/P3 |
| `s_scan` | UB | `[2W]` = `[64]` | int32 | HS 扫描工作区：右半数据、左半 halo 恒 0 | P4 |
| `s_cs` | UB | `[W+1]` = `[33]` | int32 | 独占前缀（`[0,32]` 全部为定义值）→ P6 复用为 per-expert 游标 | P4/P5/P6 |
| `s_ids` | UB | `[BLK]`（≤128） | int32 | `expert_ids` 分片暂存 | P5 |
| `s_fill` | UB | `[FILL_TILE]` = `[1024]` | int32 | 哨兵常量 tile | P0 |

**无 GM scratch**：本 kernel 不需要跨 kernel 交接缓冲（`cumsum` 只存在于 UB），因此**不引入任何额外
GM 分配**——这也是相对两段路径（align 需 `cumsum[E+1]`）的一项收益。

### 4.4 内存搬运路径

```
GM flat[numel] ──MTE2(T.copy, 4·numel B)──> UB s_x[CHUNK]
UB s_cnt ──(AIV 比较累加 / 移位 / HS 扫描)──> UB s_cs[0..32]
UB s_cs ──(单元素 store)──> GM num_tokens_post_pad[0]
UB s_ids ──MTE3(T.copy, 4·NB B)──> GM expert_ids[0:NB]
UB s_fill ──MTE3(T.copy, 分块 4·1024 B)──> GM sorted_token_ids[0:P]        （P0，先序）
GM flat[idx] ──(P6 逐 lane 标量读)──> UB 原子 s_cs[eid]++ ──slot──> GM sorted_token_ids[slot] = idx
```

- 不经过 L1：无矩阵操作，`T.copy` 直接 GM↔UB。
- 无 L0A/L0B/L0C 分配（不调用 `T.gemm`）。
- 所有写域在单核内按程序顺序 + 自动流水排序（§7），无跨核写。

### 4.5 UB 内存预算

| buffer | 字节（代表 workload `(1023,32,64)`） |
|---|---|
| `s_x` | 1,024 × 4 = 4,096 |
| `s_fill` | 1,024 × 4 = 4,096 |
| `s_cnt` | 32 × 4 = 128 |
| `s_scan` | 64 × 4 = 256 |
| `s_cs` | 33 × 4 = 132 |
| `s_ids` | 64 × 4 = 256 |
| **合计** | **8,964 B（8.8 KB）** |

上限校验：UB = 253,952 B/核（平台 `design_budget.capacity_bytes.UB`，来源 `Ascend950PR_9579.ini`
`[AICoreSpec] ub_size`，md5 2f0eca1f…）。全部用例的全局峰值 = `(1023,8,16)` 的 9,220 B
（`BLK=128` 使 `s_ids` 变 512 B），占 3.6%；按 `CONST-capacity-910B2C` 记录的 auto-multi-buffer 膨胀
上界 1.7×（**910B2C 实测，跨型号外推**）也仅 ≈15.7 KB（6.2%）。**容量非瓶颈**（§1.6.0 roofline 容量项 0.035）。
L1 / L0 预算：不使用。

### 4.6 动态轴定义

- 运行时唯一可变的是**数据内容**（`flat` 的取值分布）；shape 全部由 `(numel, E, B)` 在构建期固定。
- kernel 内所有循环上界均为构建期常量（`CHUNK//4`、`E`、`BLK`、`ROUNDS`、`N_FILL`），运行期只有
  `n = min(FILL_TILE, P-off)` 与 `idx < numel` 两类常量表达式守卫。
- `T`（token 数）在 wrapper 层是多实例的：每个 `numel` 一个编译实例（wrapper 缓存，§4.7）。

### 4.7 JIT 配置

| 项 | 值 | 说明 |
|---|---|---|
| `target` | `"ascend"` | 平台 ruleset 指定；`out_idx=[]`（全部缓冲由调用方传入） |
| 工厂签名 | `_make_small_batch_kernel(numel, num_experts, block_size) -> Callable[[], Program]` | 与冻结 wrapper 的 `self._small_batch_fn()` 调用一致（无 launch 参数） |
| 编译程序签名 | `(flat, sorted_token_ids, expert_ids, num_tokens_post_pad)` | 与 wrapper `forward` 的位置参数一致；声明 shape 用 `numel`（`numel=0` 时设备端会 DCE 该参数，但 host 侧 `k.params` 仍保留 4 个参数，实测已验证） |
| 编译缓存 | 每 `(numel, num_experts, block_size)` 一次（wrapper 的 `_small_program` 字段） | 对应 GPU 的 `target_cache`（D9 保留） |
| 流水 | 默认自动调度（`ascend/pipeline.py` 的 `allow_autoschedule` 默认开启） | 不显式 `T.Pipelined`；填充 MTE3 与 P2/P3/P4 的 V 链天然重叠（探针 M9 sync 链） |
| `pass_configs` | 不设置 | 如需排查同步：`annotate_buffer_versions` / `ascend_pipe_barrier` 仅在调试时用（§7.3） |
| guard | `num_experts > 32` 时 `raise ValueError` | 与源 factory 一致（wrapper 已拦截；直接调用者仍受保护） |
| 执行命令（本机实测） | `source .agents/platforms/toolchains/ascend/env-950pr.sh` 后用 `/workspace/venv-ascend950/bin/python`；cwd = 任务目录 | 平台 `execution` 模板中的 `/usr/local/python3.12.16/bin/python3` 在本机实测不可用于本工具链，Stage 3 按本行执行（同 op 另两条路径已采用） |

## 5. Tiling 策略

### 5.1 计算类型

索引分发类（直方图 + 一维前缀 + block→expert 映射 + 常量填充 + 散射）：**纯整型、无矩阵、无数值计算**，
`flops = 0`；性能由「发射（3.4K 条 SIMT 指令）+ GM 流量（哨兵填充为主）+ 1 次 launch」决定。

### 5.2 Block 划分

| 维度 | 划分 | 取值（构建期） |
|---|---|---|
| 核网格 | 不划分：**1 核**（§5.5） | `T.Kernel(1)` |
| expert 轴 | 不切分：`W = 32 ≥ E` 全部驻留一条 lane 链 | `W = 32` |
| token 轴（P2） | 每 lane 全量扫描，`UNROLL=4` 拆 4 条独立链 | 迭代 `CHUNK/4 ≤ 256` |
| token 轴（P6） | 每轮 32 条路由，`ROUNDS = ⌈numel/32⌉ ≤ 32` 轮 | `idx = r*32 + j` |
| 输出 block 轴（P5） | `BLK = align_up(NB, 64)` 个 lane + 内层 `E` | 迭代 `E ≤ 32` |
| 填充轴（P0） | `FILL_TILE = 1024` 元素/块，`N_FILL = ⌈P/1024⌉` 块 | 每块一次 `T.copy` |

**非整除处理**（GEMM 类以外同样适用）：① P0 尾块 `n = T.min(FILL_TILE, P-off)` + `n > 0` 守卫，
用显式 region `T.copy(s_fill[0:n], sorted[off:off+n])`；② P1 尾 `[numel, CHUNK)` 填 `-1`（不匹配任何 expert）
使 P2 无守卫；③ P5 的 `BLK ≥ NB`，用 `b*B < total` 掩码给出尾部 `E`，写回用 `expert_ids[0:NB]`；
④ P6 尾 lane `idx < numel` 守卫；⑤ `B` 非 2 的幂走 div/mul 分支（构建期判定）；⑥ `numel = 0` 由构建期
折叠（只折叠 GM 读与轮数，写回链无条件执行，§1.6.1 A7）。不依赖 `pad_value`/`data_select`。

### 5.3 约束分析

| 约束 | 检查 |
|---|---|
| 单核 UB 峰值 ≤ 248 KB | 9,220 B（全局峰值）✓ |
| 核内循环上界静态 | 全部构建期常量 ✓ |
| 写域互不重叠 | 单核顺序执行：P0 写 `[0,P)` 全段，P6 只写 `[0,total)` 内互异槽（由游标保证），二者由自动流水排序 ✓ |
| 无跨核依赖 | 单核；无 flag/barrier ✓ |
| 单次 launch 逻辑核数 ≤ 物理核数 | 1 ≤ 56 ✓ |

### 5.4 注意事项

- **单核是本路径的刻意选择**：它的价值是「1 次 launch」（O1）而不是并行度；`numel < 1024` 时单核
  指令量 ≤ 3.4K（≈2.4 µs 估算），小于第 2 次 launch 的固定开销（≈2 µs 级）。
- **填充不能外移**：P0 与 P6 的写域重叠，跨核无屏障 ⇒ 两者必须同核（D6，Stage 3/4 不得"优化"回去）。
- P6 的路由 id 从 GM 读（不读 `s_x`）是**刻意的**：读 UB 会让 UB 原子与 UB 读落在同一循环，
  编译器将插入逐轮 `asc_syncthreads()`（探针 M1/M2）。
- P5 的归约上界取构建期常量 `E`（不是 `W`）：虽然补位 lane 的值恒为 `total`（A3 左移写满 `[0,W]`），
  取 `E` 使「定义域」与 A2 的公式完全一致，且省 `W−E` 次比较（判别性对照见 §1.6.1）。

### 5.5 分核策略（物理核数适配）⭐

**① 逻辑核数计算**

本算子的并行度不是 `M/block_M × N/block_N` 形式（无矩阵网格），而是由三段工作各自的自然分块数决定：

```
直方图/前缀/映射段 : G_alg  = 1     （整条链是单一顺序依赖：计数 → 前缀 → 放置，不可分片）
散射段             : G_sc   = 1     （依赖 G_alg 的 cs，且写域与填充重叠，不可外移）
填充段             : G_fill = 1     （见 §0.6 R1：跨核无屏障 + 写域重叠 ⇒ 与散射同核）
逻辑核数           : num_logical = G_alg + G_sc + G_fill = 1
```

**② 物理核数依据（按任务平台规则实查）**

- 实查代码与返回值（本机执行，2026-11-23，`ASCEND_RT_VISIBLE_DEVICES` 未屏蔽，只做属性查询、不 launch）：

  ```python
  import torch, torch_npu
  p = torch.npu.get_device_properties(0)
  p.name            -> 'Ascend950PR_9579'
  p.cube_core_num   -> 28
  p.vector_core_num -> 56
  ```

- 平台记录交叉核对：`statectl platform --task-dir examples/moe_permute_align/_make_small_batch_kernel`
  返回 `hardware_id = ascend-950pr-9579`、`observed_cores = {cube: 28, vector: 56}`，
  `core_source = "Ascend950PR_9579.ini [SoCInfo] ai_core_cnt=28, cube_core_cnt=28, vector_core_cnt=56;
  runtime probe rtGetAiCoreCount()=28 on Ascend950PR_9579, 2026-11-23"`。
- 本算子只用 AIV（向量核）⇒ 物理核数基准 = **56**（cube 28 不使用）。

**③ 规模判定与分核方案**

- **结论：逻辑核数 1 ≤ 物理核数 56 ⇒ 无需适配**（不引入核内串行分批、不做 block 尺寸对齐）。
  依据三条：
  1. **结构**：本 kernel 的 7 个阶段是一条依赖链（计数 → 补齐 → 前缀 → 映射 → 放置），任何跨核分片都会
     在「局部计数 → 全局前缀」处引入**跨核归约**，而跨核 barrier/flag 在本工具链未认证
     （`CG-950-batch-barrier`）⇒ 只能单核。
  2. **写域**：填充（`[0,P)`）与放置（`[0,total)`）重叠，无法按槽位切给不同核（§0.6 R1）。
  3. **量级**：`numel < 1024`、`E ≤ 32` ⇒ 单核 ≤ 3.4K 条指令（≈2.4 µs 估算）与 ≤ 24.3 KB GM 流量
     （≈0.58 µs）；即使用 56 核把指令摊薄，也只能省下 < 2 µs，而多核调度与分片判定的收益不足以覆盖
     跨核归约的代价（后者在本工具链根本不可用）。
- **核内循环边界限制**：`T.SimtVF(threads=...)` 取值 32/64/256（`threads` 为构建期常量、非 launch 参数）；
  循环使用 `T.serial` / `T.Parallel`（平台 ruleset 支持）；不使用 `T.Persistent`（无跨 block 任务循环，
  单核无调度收益）。
- **逻辑核数超发检查**：1 ≤ 56，无超发。
- **若未来放开跨核能力**（记录供后续判定，不在本设计实施）：可沿 token 轴分片求局部直方图（GM
  `partial[G, E]`），再由第二次 launch 冗余合并（即 align 段路径 L 的做法）——对本路径的小规模而言
  必然更慢（多一次 launch）。

## 6. 循环与调度结构

### 6.1 循环结构总结

| # | 循环 | 结构 | 展开/调度 | 说明 |
|---|---|---|---|---|
| L1 | 填充块循环 | `T.serial(N_FILL)` | 每块一次 `T.copy`（MTE3 bulk store） | 无计算；尾块 `n > 0` 守卫 |
| L2 | 直方图 lane 轴 | `T.Parallel(W)` | lane 并行（`T.SimtVF(threads=32)`） | lane ↔ expert，每 lane 私有 `acc[4]` |
| L3 | 直方图归约轴 | `T.serial(CHUNK//UNROLL)` + python 级 `for j in range(4)` | 不展开外层；内层 4 路展开成独立累加链（A4） | 每 lane 的扫描轴（§1.6.2 V3） |
| L4 | 补齐 / 扫描 lane 轴 | `T.Parallel(W)` | 向量化 | P3/P4 |
| L5 | HS 扫描步 | **python 级 `for d in range(5)` 展开** | 每步 1 次 `T.Parallel(W)` + 1 次 `T.copy` 回写 | 5 步顺序依赖（UB 内），`log2(W)` 已最优；**禁 `while`** |
| L6 | `expert_ids` lane 轴 | `T.Parallel(BLK)` | lane 并行（`threads=BLK`） | lane ↔ 输出 block |
| L7 | `expert_ids` 归约轴 | `T.serial(E)` | 上界 = 构建期常量 `E` | 每 lane 的扫描轴 |
| L8 | 散射轮循环 | `T.serial(ROUNDS)` | 每轮 32 条路由，lane 并行（`threads=32`） | 轮数 `⌈numel/32⌉ ≤ 32`，尾轮 `idx < numel` |
| L9 | 尾部 `-1` 填充 | `T.fill` region（≤63 元素） | 常量写 | 消掉 L3/L8 的边界守卫 |

不使用 `T.Pipelined`（无多 chunk 循环可叠）、不使用 `T.Persistent`（单核无任务循环）、
不使用 `T.serial` 处理 lane 轴（会退化为标量）。

### 6.2 循环伪代码

见 §3.3（完整 kernel 伪代码）。关键点：

- **lane 轴的每一处都是 `T.Parallel`**（L2/L4/L6 + L8 的 32 个 lane），标量只出现在 lane 内层归约轴；
- **HS 扫描的 5 步用 python `for` 展开**（构建期常量表，不是运行期循环）——`@T.prim_func` 内 `while`
  会建出 TIR `While` 并报 `Infinite while loop detected`（同 op align 设计探针 K2）；
- `acc[4]` / `cnt[1]` 都是 `T.alloc_local`（`T.Parallel` 体内，每 lane 私有）；
- 所有 `T.Parallel` 之外的计算（`off`、`n`、`idx`、`total` 的读取）都是 block 级标量。

### 6.3 流水线优化

- 默认自动调度已把 P0 的 MTE3 bulk store 与 P2/P3/P4 的向量链重叠：探针 M9 的降级码里，填充的
  3 次 `asc_copy_ub2gm_align` 之后才出现 `asc_sync_notify(PIPE_MTE3, PIPE_V)`，而对应的
  `asc_sync_wait` 落在 **P6 的 VF 调用之前**——即填充与直方图/前缀并行，只在散射前需完成。
- 本 kernel 每个阶段只有一份数据（`CHUNK ≤ 1024`），没有「多 chunk 双缓冲」的空间；
  不手工设置 `annotate_buffer_versions`。
- 若 Stage 4 需要进一步压缩延时，可先测「P0 是否真的被隐藏」（`msopprof` 的 MTE3 busy 占比），
  再决定是否加大 `FILL_TILE`（减少 copy 次数）——但这不改变本设计的结构。

### 6.4 尾块处理

| 尾块 | 处理 |
|---|---|
| 填充尾（`P % FILL_TILE != 0`） | `n = T.min(FILL_TILE, P - off)`，`if n > 0: T.copy(s_fill[0:n], sorted[off:off+n])` |
| 路由尾（`numel % 64 != 0`） | `s_x[numel:CHUNK]` 填 `-1` ⇒ P2 迭代上界固定 `CHUNK/4`，无需守卫 |
| 散射尾（`numel % 32 != 0`） | `if idx < numel` 守卫（P6 尾轮） |
| `expert_ids` 尾（`NB % 64 != 0`） | `BLK = align_up(NB, 64)`；`b*B ≥ total` 掩码写 `E`；写回用 `expert_ids[0:NB]` |
| `B` 非 2 的幂（如 48） | 构建期回退 `T.ceildiv(cnt, B) * B`（探针未单独编译该分支，属**等价替换的构建期变体**，§9 风险 7） |
| `B = 1` | `p_shift = 0` ⇒ `((c + 0) >> 0) << 0 = c`；`P = numel`，`NB = numel` |
| `E < W`（如 E=8） | 补位 lane 的比较永不成立 ⇒ `cnt = 0` ⇒ `p = 0` ⇒ `s_cs[e] = total`（定义值）；P5 归约上界取 `E` |
| `numel = 0` | 构建期折叠 P1 的 GM 读与 P6 的轮次；P0/P3/P4/P5 与写回**无条件执行** ⇒ `total=0`、`expert_ids` 全 `E`、哨兵全 `0`（§1.6.1 A7 机器验证） |

## 7. 同步策略

### 7.1 同步模式

**自动同步（默认流水），零显式同步点、零跨核同步**。平台 `ascend/pipeline.py` 的 `allow_autoschedule`
默认开启，AutoSchedule/InsertSync 负责核内 UB 写后读、读后写依赖以及跨 pipe（V / MTE2 / MTE3 / S）的事件
排序。本设计**不使用** `T.sync_threads()`、`ascend_pipe_barrier`、`ascend_cross_core_*`、`sync_grid`。

### 7.2 同步点说明

| 依赖 | 位置 | 处理（探针 M9 降级码实证） |
|---|---|---|
| `s_fill` 写 → 读（填充 copy） | P0 内 | 自动：`asc_sync_notify(PIPE_V, PIPE_MTE3, 1)` + `asc_sync_wait` |
| 填充 copy → P6 散射（**写-写**顺序） | P0 → P6 | 自动：`asc_sync_notify(PIPE_MTE3, PIPE_V, 0)`（最后一次 copy 后）+ `asc_sync_wait` 落在 P6 VF 调用前 |
| `flat` 载入 → P2 读 `s_x` | P1 → P2 | 自动：`notify(PIPE_MTE2, PIPE_V)` + `wait` |
| `-1` 尾填 → P2 读（写后读） | P1 内 | 自动：`notify(PIPE_S, PIPE_V, 1)` + `wait` |
| P2/P3/P4/P5 之间的 UB 写后读 | 顺序区 | 自动（同 pipe 顺序；HS 步内 `s_cs`→`s_scan` 回写由 InsertSync 插事件） |
| `s_cs` 读（P5）→ 原子写（P6） | P5 → P6 | 自动（程序顺序；P5 的 `s_ids` 已 copy 到 GM） |
| `total` 写 GM | P4 末 | `tl::write_gm_bypass_dcache`（单元素 store） |
| 跨核 | 无 | 单核，无任何跨核依赖 |

### 7.3 pass_configs 配置

- 默认流水线（不设 `pass_configs`），不关闭 `allow_autoschedule`。
- 若 Stage 3 遇到 UB 竞争告警（例如有人把 P6 的路由读取改回 UB，导致 ThreadSync 在发散守卫上插 barrier）：
  处置顺序 ① **首选恢复本设计形态**（P6 读 GM `flat`，或在 P4 之前把路由 tile 拷进 `T.alloc_local`）；
  ② 如需显式屏障，用 `T.ascend_pipe_barrier("PIPE_ALL")`（`sync.py` 提供，单核管线配对已有执行证据见
  `PL-950-manual-sync`，`execution-patterns.md`）；③ 关闭自动调度（`tl.enable_auto_schedule=False`）
  仅在调试时使用，不作为交付形态。

## 8. 验证方案

### 8.1 Golden 函数

**独立性声明**：golden 依据 §0.1 的**语义**编写，移植源仓参考实现
`examples/TileOPs/tileops/workloads/moe.py` 的 `ref_permute_align`（`bincount` 式计数 + `cumsum` 式前缀 +
**二分查找**求 owner，路由按索引升序放置）。它**不复刻**本设计的任何技巧：无 lane↔expert 比较累加、
无 4 路累加器、无 HS 扫描、无 rank-count、无原子游标、无分块填充。两条独立路径的一致性由
§1.6.1 的 `verify_equiv.py` 三方对照（基线 / 本设计 / 参考实现）保证。

```python
import math
import torch

def golden_permute_align(topk_ids: torch.Tensor, num_experts: int, block_size: int):
    """The op's three outputs, from the operator semantics (§0.1) only.

    expert-internal order is the ascending route index (the reference's order);
    the op contract leaves it undefined, so the caller must normalise before
    comparing with a kernel that picks another order.
    """
    E, B = num_experts, block_size
    flat = topk_ids.flatten().tolist()
    numel = len(flat)

    counts = [0] * E
    for e in flat:
        counts[e] += 1
    cs = [0] * (E + 1)
    for i in range(E):
        cs[i + 1] = cs[i] + math.ceil(counts[i] / B) * B
    total = cs[E]

    P = numel + (E + 1) * (B - 1)
    nblocks = math.ceil(P / B)
    dev = topk_ids.device

    sorted_ids = torch.full((P,), numel, dtype=torch.int32, device=dev)
    slot = list(cs[:E])
    for i, e in enumerate(flat):
        sorted_ids[slot[e]] = i
        slot[e] += 1

    expert_ids = torch.full((nblocks,), E, dtype=torch.int32, device=dev)
    nb = total // B
    if nb:
        pos = torch.arange(nb, device=dev, dtype=torch.int64) * B
        # the reference's binary-search semantics: the e with cs[e] <= pos < cs[e+1]
        expert_ids[:nb] = torch.searchsorted(
            torch.tensor(cs[1:], device=dev), pos, right=True
        ).to(torch.int32)

    return sorted_ids, expert_ids, torch.tensor([total], dtype=torch.int32, device=dev)
```

op 层判定沿用 `examples/TileOPs/tests/ops/test_moe_permute_align.py` 的 `_assert_route_equivalent`：
`num_tokens_post_pad` 与 `expert_ids[0:total//B]` 精确相等；`expert_ids[total//B:]` 只要求
`< 0` 或 `>= num_experts`（本 kernel 写 `E`）；`sorted_token_ids` 按 (expert, 路由索引) 归一化后逐元素相等。

### 8.2 精度标准

- **精确相等**（int32 索引数组，`torch.equal`）：`atol = rtol = 0`。全流程整数运算、无舍入，
  A1–A5 已机器验证逐位/契约级等价（§1.6.1，14 组 shape × 5 分布）。
- 不做容差比较；任何差异都视为语义错误（不是精度问题）。
- 值域断言：`0 ≤ flat < E` 是输入契约（wrapper 与 manifest 均声明），越界输入不在契约内
  （源算子同样不检查；`E ≤ 32` 由 factory 保证，因此 `s_cs[eid]` 的原子不会越出 `[0, E) ⊆ [0, 32)`）。

### 8.3 L0 门槛测试计划

执行方式（本机实测）：`cd examples/moe_permute_align/_make_small_batch_kernel` →
`source .agents/platforms/toolchains/ascend/env-950pr.sh` → `/workspace/venv-ascend950/bin/python {op}.py --level L0`。
L0 = 「本路径的每条代码路径 + 每个边界」的一次性门槛，共 9 例，单次运行 < 1 min；每例都调用 §8.1 golden
并用 `_assert_route_equivalent` 判定，且**额外断言** `sorted_token_ids` 中非路由槽位恰为 `numel`：

| # | 用例 `(numel, E, B)` | 覆盖点 |
|---|---|---|
| L0-1 | `(8, 4, 4)` | smoke（`tiny-bs4`）：最小可用；`P=23` 单填充块；1 轮散射 |
| L0-2 | `(2, 4, 4)` | `single-token`：路由数 < lane 数（P6 尾轮守卫） |
| L0-3 | `(32, 8, 16)` | `small-bs16`：`numel` 恰为 lane 数（整除边界）；`B=16` |
| L0-4 | `(200, 8, 16)` | `sb-numel200`：`CHUNK=256`、7 轮散射、`E=8` 的补位 lane |
| L0-5 | `(600, 32, 64)` | `sb-numel600-maxexp`：`E=32=W`（无补位 lane）；3 个填充块 |
| L0-6 | `(1023, 8, 16)` | `routes-1023`：dispatch 边界（最后一档小批）；`BLK=128`；32 轮 |
| L0-7 | `(0, 8, 16)` | `numel=0`（wrapper 走本路径）：`total=0`、`expert_ids` 全 `8`、哨兵全 `0` |
| L0-8 | `(100, 8, 48)` | `B` 非 2 的幂（div/mul 分支）；`B=48` 的补齐 |
| L0-9 | `(64, 8, 1)` | `B=1`（`p_shift=0`，移位退化为恒等）；`NB=numel` |

判定：9 条全部通过 → `[PRECISION_PASS]`；任一不等 → 记录首个差异下标与 `(numel,E,B)` → 回 Stage 3。
L1/L2/Boundary 分层套件由 `tilelang-op-develop` 生成（覆盖 manifest 12 条 workload 中 E 小的档位 +
`tests/ops/test_moe_permute_align.py` 的全部本路径用例 + 随机分布扫描），本文不枚举。

## 9. 风险点与注意事项

### 9.1 已知约束

1. **UB int32 原子的真机行为（主要风险）**：`asc_atomic_add(__ubuf__ int32_t*, int32_t)` 只有头文件声明
   （`device_atomic_functions.h:25`，arch 3510）与编译通过的降级码（探针 K6），**无执行证据**；
   `numel` 次原子在同一核内对 ≤ 32 个地址自增，吞吐与争用代价未实测。主选依赖它；备选（§1.6.3 L2）
   完全不用原子。判定阈值与回写路径见 §1.6.3 的裁决计划。若原子语义（返回旧值）在真机不符，
   Stage 3 会直接出现槽位冲突（重复槽/漏写）→ 走备选。
2. **「1 条向量指令/cycle、1.4 GHz」是未实测假设**：950 真机性能常数当前不存在
   （`bottlenecks.md` 明示无 950 实测瓶颈模式）。§1.6.0 的发射项（2.42 µs）与 §1.6.3 的判定阈值都基于它；
   Stage 4 的 baseline 采集必须按 `hardware-cost-model.md` §3 回填「设计估算 vs 实测」偏差行。
3. **ThreadSync 的别名保守性**（承重约束）：只要原子/索引访问与 UB 读在同一循环内，编译器就插逐轮
   `asc_syncthreads()`，并在发散守卫上告警「the race remains」（探针 M1/M2/M4 输出）。本设计因此规定
   **P2/P6 热循环不得出现数据相关 UB 下标访问**。Stage 3/4 若改回该形态，必须重新评估正确性
   （不是性能问题）。判别性证据：探针 M5/M6（读 GM）无 barrier、M1/M2（读 UB）有 barrier 与告警。
4. **填充与散射必须同核**（承重约束）：二者写域重叠且跨核无屏障（`CG-950-batch-barrier`）。
   Stage 3/4 不得把 P0 拆到别的核（那会产生「哨兵覆盖路由」的间歇性错误）。
5. **单核的代价**：本路径的延时下界 ≈ max(发射 2.4 µs, 流量 0.6–1.4 µs) + 1 次 launch ≈ 4.4 µs（估算）。
   这是「NPU 无跨核 barrier」的必然结果，不是分核参数可调项；应在 `[DESIGN_LIMIT]` 中如实记录
   （若 Stage 4 实测显著高于 4.4 µs，归因顺序：原子吞吐 → 发射率假设 → launch 开销）。
6. **`T.Parallel(1024)` 的填充代码体积**：探针 M7 降级为每 lane 2 个 `int2` store（1024/256 = 4 元素/线程，
   折叠成 2 次 int2），代码量可接受；若把 `FILL_TILE` 提到 4096，展开会线性增长（`#pragma unroll`），
   需重新核对代码体积与 I-cache 影响。
7. **`B` 非 2 的幂的 div/mul 分支未单独编译探针**：该分支只在构建期选择（`p_shift is None`），
   语法与 `T.ceildiv` 同族（align 段设计已用同一写法并有探针覆盖）；L0-8（`B=48`）是它的运行期回归。
   若 Stage 3 编译该分支失败，退路是用 `(cnt + B - 1) // B * B`（同样的 `T.ceildiv` 语义）。
8. **`numel = 0` 的编译程序签名**：设备端会 DCE 未被读的 `flat` 参数，但 host 侧 `k.params` 仍保留
   4 个参数（本机实测 `KernelParam(shape=[0])` 保留）⇒ wrapper 传 4 个张量可正常调用。Stage 3 必须
   **保持 4 参数声明**（不要为了"消参数"手改签名）。
9. **`expert_ids` 尾段写 `E`（不是 0）**：源算子与参考实现都写 `num_experts`；写 0 会被下游 grouped GEMM
   当作 expert 0。层测容忍任意 `≥ E` 的值，但契约上应与两段路径一致 ⇒ 写 `E`。
10. **factory 的 `raise` 必须保留**：`num_experts > 32` 时 `_make_small_batch_kernel` 直接抛
    `ValueError`（与源 factory 一致）。wrapper 已按 `num_experts <= 32` 分流（层测 `experts-33` 走
    align+scatter），但直接调用者仍受该守卫保护。

### 9.2 常见错误

- **把 P0 的填充挪到另一个核**：见 §9.1 风险 4（写域重叠 + 无屏障）。这是本 kernel 最危险的"优化"。
- **P6 改读 UB `s_x`**：会引入逐轮 `asc_syncthreads()`（性能）与 ThreadSync 告警（正确性风险），
  见 §9.1 风险 3。读 GM `flat` 只多 4 KB 流量。
- **用 `W` 代替 `E` 做 P5 的归约上界**：当前等价（补位 lane = `total`，§1.6.1 判别性对照），但一旦
  未来把「左移写满 `[0,W]`」改成只写 `[0,E]`，补位 lane 就是 UB 残留 ⇒ `expert_ids` 静默错位。
  两处必须同改同上（§1.6.1 A3 的两条不变式）。
- **把 `numel = 0` 的写回放进运行期守卫**：契约要求零输入下三个输出全部写出（§1.6.1 A7）；
  只能折叠「GM 读」与「轮数」，不能折叠写回链。
- **在 `@T.prim_func` 内写 `while` 或 `T.alloc_var(...)[0]`**：两者实测不可构建（§3.4），
  分别改用 python `for d in range(5)` 与 `T.alloc_local([1])`。
- **假设 `T.cumsum` 可用**：探针 K7 明确不可用（`no cumsum implementation is registered`），
  必须自实现 HS 扫描。
- **`s_cs[32]` 被原子改写**：原子只触及 `eid < E ≤ 32`，故 `s_cs[32] = total` 安全；但若把 `W` 调成
  `E` 且 `E` 恰好等于 `W`，`eid` 仍 `< W` ⇒ 边界恒定成立。**不要**把 `s_cs` 的长度改成 `W`（需要 `W+1`
  存放 `total`）。
- **`-1` 尾填与 `T.min` 裁剪混用**：`s_x` 的尾部只用于 P2 的比较（`-1` 不匹配），不得用于 P6（P6 有
  `idx < numel` 守卫，绝不读尾部）。

### 9.3 特殊场景处理

| 场景 | 行为 |
|---|---|
| 某 expert 计数为 0 | `p_e = 0`，其区段为空；P5 的 rank-count 取「最大满足者」，与源式一致（§1.6.1） |
| 连续多个零计数 expert | 同上；`s_cs` 出现相等相邻值，`owner` 判定已证明等价 |
| 单 expert 吸收全部路由（`E=1`） | `p_0 = ⌈numel/B⌉·B`，`total = p_0`；`expert_ids` 全 0；lane 1–31 恒 0 |
| `E ≫ numel`（如 8 路由 / 32 expert） | 大部分 lane 计数 0；时间由填充（`P`）与发射决定 |
| `E < W`（E=8） | 补位 lane 恒 0（比较不成立）⇒ `s_cs[e] = total`；P5 上界取 `E` |
| `numel = 0` | `total = 0`、`expert_ids` 全 `E`、哨兵全 `0`；P6 零轮 |
| `B = 1` | `p_e = cnt_e`，`P = NB = numel`；`p_shift = 0`（移位退化为恒等） |
| `B` 非 2 的幂 | 构建期回退 `ceildiv·B`；L0-8 覆盖 |
| `num_experts > 32` | factory 抛 `ValueError`（wrapper 已分流到两段路径） |

## 10. 交付清单

### 10.1 目录结构

```
examples/moe_permute_align/_make_small_batch_kernel/
├── DESIGN.md                       # 本文件
├── RETROSPECTIVE.md                # Stage 1 复盘（Phase 8）
├── verify_equiv.py                 # §1.6.1 等价性机器验证（可重跑，退出码 0 = 全 PASS，约 0.6 s）
├── history_version/
│   ├── design_probe_forms.py       # 探针 1：初级形态 + T.cumsum 负对照（7 项）
│   └── design_probe_refined.py     # 探针 2：主选 / 备选 / 最终单核骨架 + 填充形态（11 项）
└── {op}.py                         # Stage 3 产出（kernel + golden + 分层测试），本阶段不创建
```

### 10.2 文件清单

| 文件 | 责任 | 状态 |
|---|---|---|
| `DESIGN.md` | 本设计（迁移分析 §0 + 调研/优化 §1.6 + API/内存/Tiling/循环/同步/验证/风险 + §11 自检） | 已产出 |
| `verify_equiv.py` | 基线 vs 本设计 vs 参考实现三方数值对照（纯 Python 整数精确）+ 补位 lane 判别性对照 + `numel=0` 边界 | 已产出并执行（14 行全 EQUIV_PASS，退出码 0，约 0.6 s） |
| `history_version/design_probe_forms.py` | 形态可构建性探针 1（K1–K7） | 已产出并执行（6/7 COMPILE_OK，K7 为 `T.cumsum` 负对照） |
| `history_version/design_probe_refined.py` | 形态可构建性探针 2（M1–M11，含最终单核骨架与填充形态） | 已产出并执行（11/11 COMPILE_OK） |
| `RETROSPECTIVE.md` | 复盘章节（Skill Flow Issues / Value Point Proposals / Transferable Lessons） | 已产出 |
| `{op}.py` | kernel（`_make_small_batch_kernel` 工厂）+ golden + L0/L1/L2/Boundary | Stage 3 |

### 10.3 命名规范

- kernel 工厂：`_make_small_batch_kernel(numel, num_experts, block_size)`（与脚手架 import 名一致，不得改名）。
- 编译程序内部命名：`_small`（与源 factory 内层同名）。
- 缓冲名：`flat`、`sorted_token_ids`、`expert_ids`、`num_tokens_post_pad`（GM，与 wrapper 位置参数顺序一致）；
  UB 统一 `s_` 前缀。

### 10.4 实现顺序

0. **先落实两条承重约束**（Stage 3 的第一验收点）：
   ① P0 填充与 P6 散射**同核**（`T.Kernel(1)`）；② P2/P6 热循环**无数据相关 UB 下标访问**
   （P6 读 GM `flat`）。二者分别是「正确性」与「正确性 + 性能」的守卫（§9.1 风险 3/4）。
1. P1 + P2 + P3 + P4（载入 → 直方图 → 补齐 → 前缀）：先跑通 `cumsum`/`total`（L0-1/L0-2/L0-7）；
2. P0 填充（哨兵）+ P5 `expert_ids`：L0-3/L0-4；
3. P6 散射（原子游标）：L0-5/L0-6；
4. 边界：L0-8（`B=48`）、L0-9（`B=1`）；
5. 交 Stage 4：按 §1.6.3 裁决计划做「L1' 原子放置 vs L2 无原子放置」A/B，并回填
   `hardware-cost-model.md` §3 的估算/实测偏差行。

## 11. 设计自检与输出契约核对

### 11.1 与两段路径的输出契约核对（承重）

本 kernel 是同一 op 的三条构建路径之一，**输出必须与 align+scatter 逐项一致**：

| 项 | align + scatter 路径 | 本 kernel | 一致性依据 |
|---|---|---|---|
| `sorted_token_ids` shape/dtype | `[P]` int32 | `[P]` int32 | 同一 `P` 公式（§4.2） |
| 路由槽位 | 每 expert 区段前 `cnt_e` 槽 | 同 | §1.6.1 A5（多重集 + 槽区间 + 互异） |
| 哨兵值 | `numel`（align 段写） | `numel`（P0 写） | §1.6.1 A5「哨兵保持」 |
| expert 内顺序 | 未定义（原子到达序） | 未定义（轮次到达序） | §0.1 契约；测试归一化比对 |
| `expert_ids` | `[NB]`，owner / `E` | 同 | §1.6.1 A2（与参考二分逐元素相同） |
| `num_tokens_post_pad` | `[1]` = `cs_E` | 同 | §1.6.1 A1+A3 |
| 额外输出 | `cumsum[E+1]`（内部交接） | **无**（UB 内部量） | wrapper 只消费三个输出 |

`expert_ids` 的尾段（`b·B ≥ total`）三条路径都写 `num_experts`——层测允许任意 `≥ E`，但三者一致是契约要求。

### 11.2 门禁自检（`statectl gate 1`）

交付前执行 `python3 .agents/tools/statectl.py gate 1 --dir examples/moe_permute_align/_make_small_batch_kernel`；
本机实测结果：`{"ok": true, "pass": true, "stage": 1, "failures": [], "warnings": []}`（0 failure / 0 warning）。
机械校验项与本文件的对应关系：

| 校验项 | 落点 |
|---|---|
| 必需章节（§0.1/0.3/0.4/0.5/0.6、§1.x、§2、§3、§3.5、§4.x、§5、§6、§7、§8） | 全文 |
| 调研四问（R1 等价化简 / R2 在线 / R3 复杂度 / R4 亲和）+ 基线 + 调研结论 | §1.6.0 |
| `§1.6.1` 优化结论 + `EQUIV_PASS` 结果表 + 脚本存在 | §1.6.1 与 `verify_equiv.py` |
| `§1.6.2` 向量化结论 / `§1.6.3` 布局决策结论 | §1.6.2 / §1.6.3 |
| 编程模式（SIMD/SIMT） | §2.1（SIMT） |
| 分核三要素（逻辑核数 / `ascend-950pr-9579` 实查 / 规模判定） | §5.5 |
| Golden + L0 计划 | §8.1 / §8.3 |
| 无占位符 | 全文 |

### 11.3 交付状态

- 设计期未做真机验证（仅无卡编译探针 18 项 + 等价性脚本 14 项检查），所有执行期结论标注「待 Stage 3」。
- 待 Stage 4 裁决的项：UB int32 原子放置（L1'）vs 无原子放置（L2），阈值与回写路径见 §1.6.3。
