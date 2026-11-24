# rope（`_rope_positions_kernel`）算子设计文档

> **任务平台**：`ascend-950pr-9579`（Ascend950PR_9579，`dav-3510`，28 Cube / 56 Vector，UB 253952 B / L1 524288 B / L0C 262144 B）；后端 `ascend`（`import tilelang.ascend.language as T`、`tilelang.compile(..., target="ascend")`）；工具链 tilelang 0.1.15 + CANN 9.2.1（GCC 12.3.0），环境由 `.agents/platforms/toolchains/ascend/env-950pr.sh` 提供。容量与核数读取记录见 §5.5 与 §4.5。
>
> **任务类型**：迁移（TileOPs GPU `RoPEFwdOp` 家族 → Ascend）；harness 子模式，设计对象是 GPU 侧四个 kernel 类共用的 `@tilelang.jit` 工厂 `_rope_positions_kernel`（抽取文件 `examples/TileOPs/tileops/kernels/rope/rope/_ro_p_e_fwd_kernels.py`，402 行）。§0 是三问解读 → 调研 → 耦合性判定 → NPU 重设计，§1–§11 全部基于 §0.6 重设计后的 NPU 算法。
>
> **设计期不做设备验证**：本机只有 1 张卡且被别的算子占用，本次**未编译、未运行任何 kernel、未跑 msprof**，也未设置设备可见性环境变量。所有 API 结论来自**本机安装包静态核对**（`/workspace/venv-ascend950/lib/python3.12/site-packages/tilelang`，`tilelang.__version__ == 0.1.15`）与平台知识条目；凡未经设备实证的条目在下文逐条标注「待 Stage 3 用真实编译确认」。
>
> **编程模式**：SIMT（`T.SimtVF(threads=256)` + `T.Parallel` + fp32 表达式链）。平台 `programming_modes.migration_default` 为 `simd`，本次**有据偏离**，理由与 SIMD 备选路径见 §2.4。
>
> **修订记录**：revision 0（首次设计，无回退输入）。

---

## 0. 源算子解读与迁移分析（迁移任务必填）

> 组织方式：先读懂源算子（0.1–0.4），再判定硬件耦合性（0.5），最后给出 NPU 算法重设计（0.6）。本章结论驱动 §1–§11。

### 0.1 源算子语义（做什么）

**数学语义**：给定每 token 的位置 `positions`、cos/sin 表 `cos, sin`（形状 `(P, R/2)`，`P = max_position`，`R = rotary_dim`）与展平了头的 `query/key`（`[*S, D·H]`，`D = head_size`），每个 token 的每个 head 的前 `R` 列被旋转，其余列原样拷贝：

- **NeoX（`neox=True`）**：`h = R/2`，对 `k ∈ [0, h)`：
  `y[..., k] = x[..., k]·c_k - x[..., k+h]·s_k`，`y[..., k+h] = x[..., k+h]·c_k + x[..., k]·s_k`，其中 `c_k = cos[row, k]`、`s_k = sin[row, k]`。
- **GPT-J / interleaved（`neox=False`）**：对 `k ∈ [0, h)`：
  `y[..., 2k] = x[..., 2k]·c_k - x[..., 2k+1]·s_k`，`y[..., 2k+1] = x[..., 2k+1]·c_k + x[..., 2k]·s_k`。
- **行号**：`row = clamp(int64(positions[t]) + int64(offsets[t]), 0, P-1)`（无 offsets 时 `+0`）；越界位置读最近行（源注释与测试 `test_rope_unchecked_position_outside_the_cache_reads_its_nearest_row` 均为该语义）。
- **尾列**：`x[..., R:D]` 每 head 原样拷贝到输出。

**规约语义**：**无归约**。每个输出元素只由同一 token 的 2 个输入元素、2 个表值与 1 个行号决定，不存在累加顺序问题，也不存在跨元素依赖。

**dtype 语义**：`x`、`key`、`cos`、`sin` 同 dtype（`float16 | bfloat16 | float32`）；**算术在 float32 域完成、每个输出值只舍入一次**回输出 dtype（源 `_rotate_pair`：`x_lo = T.Cast("float32", lo[i])`、`lo[i] = T.Cast(lo.dtype, x_lo*c - x_hi*s)`；golden 侧 `_rotate_pairs(heads[..., :rotary_dim].float(), ...)` 后 `.to(x.dtype)` 同口径）。表值本身是**已经按 dtype 舍入过的输入**，kernel 不得重算三角函数。

**边界语义**：

| 边界 | 源行为（§0.1 基线） |
|------|---------------------|
| `positions.numel() == 0` | host 侧 `torch.empty_like(query)` 提前返回，**不发起 kernel** |
| `position ∉ [0, P)` | clamp 到表边界（`T.max(0, T.min(seen, P-1))`），不越界 |
| `positions` dtype | `int32` 或 `int64`（`RoPECall.position_dtype` 校验） |
| `key` 为 None 或 `key_heads == 0` | query-only 程序（`out_idx=[4]`）；`key` 存在但 0 列时 host 返回 `empty_like(key)` |
| `offsets` | 可选 `int32/int64`；源 GPU 契约「有 offsets 必有 key」 |
| `rotary_dim < head_size` | 尾列拷贝（`tail_units > 0` 分支） |
| `head_size` 为奇数（如 65） | 允许：`rot_vec = gcd(lanes, D, R/2)` 退化为 1，源按标量访问并保证正确 |
| `Hq ≥ 1`、`rotary_dim` 偶且 `≤ head_size` | 构造期 `ValueError`（host，不进 kernel） |
| 多维 `positions`（如 `[B,S]`） | host `positions.view(-1)`、`query.view(tokens, cols)` 后统一为 2-D（`[B,S,H,D]` 的 BSND 布局展平即 `[B·S, H·D]`） |

**输出**：新张量（非原地），shape 与 dtype 同输入；`key_out` 在 `key is None` 时为 None。**§8.1 的 golden 以本节为唯一依据**（移植 `examples/TileOPs/tileops/workloads/rope.py` 的 `rope_tables` + `rope_reference`），不得参考 §0.6 的 NPU 算法。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `positions` | `[*S]`（kernel 内 `(num_tokens,)`） | `int32` / `int64` | 每 token 的表行号；`num_tokens` 为动态轴 |
| `query` | `[*S, D·Hq]`（kernel 内 `(num_tokens, Hq·D)`） | `fp16/bf16/fp32` | 每 token 的头已展平；列维 `D·Hq` 编译期常量 |
| `key` | `[*S, D·Hk]`，nullable | 同 query | `Hk == 0` 时允许存在且为 0 列 |
| `offsets` | `[*S]`，nullable | `int32` / `int64` | 加到 positions 上；有 offsets 必有 key |
| `cos` / `sin` | `(P, R/2)` | 同 query | 调用方预计算并按 dtype 舍入 |
| `query_out` / `key_out` | 同 query / key | 同 query | 新张量（`out_idx`：offset 变体 `[6,7]`、key 变体 `[5,6]`、query-only `[4]`） |
| 编译期常量 | — | — | `Hq, Hk, D, R, P, dtype, position_dtype, offset_dtype, neox` |

**唯一动态轴**：`num_tokens`（源 `T.dynamic("num_tokens")`）。`max_position`、`head_size`、`rotary_dim`、头数都是编译期常量，进 kernel 特化参数。

### 0.3 实现算法解读（怎么算）

**计算步骤分解**（源文件 `_rope_positions_kernel` 与两个 kernel 类逐语句归入下表，无遗漏）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义 |
|------|------|------|------|----------|
| 1 | host：`positions.numel() == 0` 早退 | positions | 空输出 | 空输入不 launch |
| 2 | host：`positions.view(-1)`、`query.view(tokens, Hq·D)`、`key.view(...)` | 输入张量 | 2-D 行视图 | `[*S] → [tokens]`、BSND 展平 |
| 3 | host：`rot_vec = gcd(lanes, D, R/2)`、`tail_vec = gcd(lanes, D, R)`、`row_vec = gcd(lanes, D)`（`lanes = 16 // itemsize`） | dtype、D、R | 三种向量宽度 | 源选择的 16 字节对齐访问宽度 |
| 4 | `_head_units`：`width = rot_vec`（NeoX）/ `max(rot_vec,2)`（GPT-J）、`rot_units = (R/2 或 R)//width`、`tail_units = (D-R)//tail_vec` | 同上 | 每 head 的旋转/拷贝单元数 | 每 head 的 R 列与尾列如何切单元 |
| 5 | grid：`T.Kernel(num_tokens * parts, threads=threads)`；`token = bx // parts`、`part = bx % parts` | 动态 tokens、`parts` | 块索引 | 块 ↔ (token, part) 双射 |
| 6 | 行号：`row = clamp(positions[token] + offsets[token], 0, P-1)`，写入 `T.alloc_local([1])` | positions/offsets | 标量 row | 步骤「行号」 |
| 7 | 单元映射：`within = (part·units + r)·threads + threadIdx`；`head = within // head_units`、`slot = within % head_units` | 块/线程索引 | (head, slot) | 单元 ↔ 列偏移 |
| 8 | 旋转单元载入：低半（NeoX 另有高半 `+half`）按 `width` 向量 `T.vectorized` 读入寄存器 | query/key | 寄存器 `lo/hi` | 读输入 |
| 9 | 表载入：`table_slots` 按 `table_width` 读 `cos/sin[row, first+:]` 到寄存器 `cs/sn` | cos/sin | 寄存器 `cs/sn` | 读表 |
| 10 | 旋转：`rotate_unit` 对每对执行 `x_lo·c - x_hi·s` 与 `x_hi·c + x_lo·s`（fp32，舍入一次） | 寄存器 | 寄存器 | 步骤「旋转」 |
| 11 | 写回：旋转单元写 `y[token, low + j]`（NeoX 另写高半），尾单元 `T.vectorized(tail_vec)` 直拷 | 寄存器 | query_out/key_out | 写输出/尾列拷贝 |
| 12 | staged 变体：`T.alloc_shared(heads·D)` 行缓冲 + 线程协作搬入 → `T.sync_threads` → 每线程把自己负责的单元在 SMEM 内旋转 → `T.sync_threads` → `T.copy(buf, y[token, :])` 整行搬出 | query/key | SMEM 行 + 输出行 | 与 8–11 同语义，仅中转不同 |
| 13 | host：`_shape(tokens)` 选择 wide/config 程序；`RoPEStagedRowsKernel._shape` 选 staged 或 positions 程序；`@target_cache(maxsize=32)` 缓存编译产物 | tokens、设备参数 | 选中的 program | 程序形态选择（性能，不改语义） |

**数据流图**（源硬件视角）：

```
positions-program（默认）：
GM[query/key] --lane 向量直读--> 寄存器(lo/hi, dtype) --cast--> fp32 --rotate--> 寄存器 --cast--> GM[query_out/key_out]
GM[cos/sin]   --lane 向量直读(每 block 一次 row)--> 寄存器(cs/sn)        （无 SMEM）
GM[positions] --每 block 一次标量读--> 标量 row --clamp--> 表行地址

staged-rows-program：
GM[query/key 行] --线程协作 T.copy(GM→SMEM)--> SMEM[heads·D] --单元读--> 寄存器 --rotate--> SMEM --T.copy(SMEM→GM)--> 输出行
```

- 两个程序都**没有跨 block 依赖**、没有归约、没有原子操作。
- 默认程序**不使用 SMEM**；staged 程序用一块 `heads·D` 的行缓冲，靠 `T.sync_threads()` 在「搬入」与「旋转后搬出」之间两次同步。

**循环与并行结构**：grid = `num_tokens × parts`（`parts = ceil(token_units / (threads·units))` 或 staged 的 `sides ∈ {1,2}`）；块内 `threads=128` 线程、每线程 `units` 个单元（16-bit 4、fp32 1）；单元内 `T.vectorized(width)` 或 `T.serial(width)`（GPT-J 且 `rot_vec < 2` 时）。GPT-J 的「单元」含整数个相邻对，`width = max(rot_vec, 2)`。

**host 侧逻辑**：① 空 token 早退；② 三次 `view` 展平；③ 按 `(query_heads, key_heads, dtype, position_dtype, offset_dtype, 约定)` 选 kernel 类并缓存；④ 选 wide/config 与 staged/positions 程序；⑤ 无 offsets 时 Op 层不传 offsets。**没有任何 host 预处理改动张量内容**（无 im2col/permute/cast）。

**编译配置**：`@tilelang.jit(out_idx=...)` 仅声明输出位置；无 `pass_configs`；`@target_cache(maxsize=32)` 缓存工厂。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 耦合性初判 |
|---|----------|------|------|-----------------|-----------|
| 1 | 每线程多单元（`units` = 4 / fp32 用 1） | 提高 in-flight 载入数，掩盖访存延迟 | 每线程连续处理 4 个单元，寄存器级并行 | CUDA 大寄存器堆 + 多 warp 并发 | 硬件强相关（线程/寄存器模型） |
| 2 | 块 = 一个 token 的若干单元（`parts` 折叠，`bx // parts`、`bx % parts`） | 连续块走连续内存 | 块索引折叠使相邻块地址相邻 | grid 调度顺序 | 硬件相关（grid 语义） |
| 3 | 16 字节对齐向量访问（`rot_vec`/`tail_vec`/`row_vec`） | 减少访存指令数 | `T.vectorized(width)`，宽度取「整除列偏移的最大 2 的幂（≤16 B）」 | 向量化 load/store 指令 | 模式可移植（宽度需重算） |
| 4 | 表行每 block 读一次（`table_row` 先 clamp 一次，`cs/sn` 寄存器复用） | 表流量从 O(H·R) 降到 O(R) per token | 每 block 一次行读 + 单元复用 | 无（纯访存削减） | 可移植（算法/访存层） |
| 5 | load-before-store 重排（`tail_units>0` 时先全部 load 再 store；`units==1` 时 copying lane 先 load） | 避免「拷贝列的 load 等 position/表的 load」串行 | 语句重排 + ptxas 的 load 提前 | ptxas 调度、内存依赖分析 | 弱相关（意图可移植，机制不可） |
| 6 | staged-rows：SMEM 行中转 + 整行 bulk store（`T.copy`） | 整行一次连续大段写，消除单元级 load→store 往返 | `alloc_shared` + 两次 `sync_threads` + 一次 `T.copy` 搬出 | shared memory 容量与带宽、block 级同步 | 硬件强相关（SMEM + barrier） |
| 7 | fp32 单次舍入算术（`_rotate_pair`） | 数值与 golden 一致（fp32 中转、一次舍入） | cast→fp32 运算→cast 回 | 无（数值语义） | **语义层，必须保留** |
| 8 | 动态 token 数 + `@target_cache` 编译缓存 | 一个编译产物服务任意 token 数 | `T.dynamic("num_tokens")` + 工厂缓存 | 动态 grid | 可移植 |
| 9 | query/key 单 launch（query 单元在前、key 单元在后；staged 用 `part ∈ {0,1}` 交替） | 省一次 kernel 启动 | 单元编号跨两张量连续 | 无 | 可移植（任务映射） |
| 10 | 尾列直通单元（`tail_units`，`tail_vec` 向量宽度） | 尾列不参与算术、零额外计算 | 独立单元类型 | 无 | 可移植 |
| 11 | 两套程序形状 + 两级启发式（wide/config；staged/positions） | 小规模降延迟、大规模用 bulk store | host 侧按 `tokens`、`sm_count·threads_per_sm`、SMEM 预算选程序 | SM 数量、SMEM 容量、常驻线程数 | **硬件强相关**（设备参数驱动的选择） |
| 12 | warp 内单元/列分布（`threads=128`，单元列由 `threadIdx` 决定） | warp 内 coalesced | CUDA thread/warp 语义 | warp + memory coalescing | 硬件强相关 |

> 识别不出来 ≠ 不存在。上表覆盖源文件 402 行的全部显式优化意图：无 matrix/mma、无 warp shuffle 归约、无 swizzle、无 online 统计量、无 `T.Pipelined`、无 atomics。

### 0.5 硬件耦合性分析与 NPU 适配决策

判定问题：实现算法与优化手段是否硬件强相关、能否用在 NPU 上？依据列给出 `examples/`、平台知识条目或安装包源码；无等价能力且影响正确性/性能者判「重新设计」（§0.6）。

| # | 条目 | 层级 | 源硬件依赖 | NPU 等价能力 | 处置 | 依据 |
|---|------|------|-----------|-------------|------|------|
| 1 | 旋转数学语义（两种配对 + clamp 行号 + 尾列拷贝 + fp32 单次舍入） | 语义 | 无 | — | **保留** | §0.1 为唯一基线（语义层无条件保留） |
| 2 | 动态 `num_tokens` | 语义/接口 | 无 | 有：`T.dynamic("num_tokens")` + 1-D 动态 grid | **保留** | 安装包 `T.dynamic(name, dtype="int32")` 签名；`examples/paged_cache_write/_paged_cache_write_kernel/DESIGN.md` §3.4 同用法 |
| 3 | host 侧 view / 早退 / 编译缓存 | 接口 | 无 | 有（torch view + Python） | **保留** | 不进 kernel；wrapper 已固化 |
| 4 | 每 token 一次表行读（#4） | 优化 | 无 | 有（UB 暂存 + 复用） | **等价替换** | §0.6 R2 的 `cs_ub/sn_ub`；见 §1.6.1 A1（机器验证 0 失配） |
| 5 | 16 B 对齐向量访问（#3） | 优化 | CUDA 向量 load/store 宽度 = 16 B | 有：`T.Parallel`/`T.vectorized` 在 128-bit 向量上自动向量化 | **等价替换** | 平台 `PL-950-vector-execution`（SIMD/SIMT 两条路径的尾块填充）；`PL-950-conversion-layout`（SIMT 逐元素 + 广播） |
| 6 | 每线程多单元（#1，`units`） | 优化 | CUDA 寄存器堆/多 warp | 无等价「线程级 out-of-order」；SIMT 线程域由 `T.SimtVF(threads=256)` 提供固定 lane 数 | **重新设计 R4** | §0.6 R4：把「每线程多单元」换成「每块更大 tile（4096 元素 / 256 lane = 16 元素/lane）」 |
| 7 | 块 ↔ token 折叠（#2） | 优化 | 2-D/3-D grid 语义 | ascend `T.Kernel` 只有 1-D core grid（无 `threads` 参数） | **重新设计 R1** | 安装包 `Kernel(*blocks, prelude=None)` 签名；`examples/paged_cache_write/_paged_cache_write_kernel/DESIGN.md` §3.4 |
| 8 | load-before-store 重排（#5） | 优化 | ptxas 的 load 提前与依赖分析 | 无对应机制；改用**双缓冲分离**（`x_ub` 只读、`y_ub` 只写）从结构上消除依赖 | **重新设计 R3** | §0.6 R3：读写缓冲分离后，「先读后写」顺序不再是正确性要求 |
| 9 | staged-rows SMEM 行中转（#6） | 优化 | SMEM 容量/带宽 + block barrier | 有：UB（`T.alloc_shared`，248 KB）+ 自动同步 | **等价替换（升格为主选）** | §0.6 R3：整块 UB 双缓冲暂存 + 整块连续写回，对应源的「bulk store」意图；块 DMA 形态为备选 B（§3.3） |
| 10 | 两套程序 + 两级启发式（#11） | 调度 | `sm_count`、`threads_per_sm`、SMEM 预算 | **无等价设备事实**（NPU 无 SM/常驻线程概念） | **重新设计 R6** | §0.6 R6：单一程序 + `units_per_block` 预算；`_STAGED`/`wide` 两值编译同一程序 |
| 11 | warp 内单元/列分布（#12） | 调度 | CUDA warp（32 lane）+ coalescing | 有：`T.SimtVF(threads=256)` 的 lane 域 | **等价替换** | 平台 `kernel_template.py` 用 `T.SimtVF(threads=256)`；`PL-950-row-primitives`（每行 64 线程 SIMT） |
| 12 | GPT-J 单元 `width = max(rot_vec,2)` 与 `T.serial(width)` 标量回退 | 优化 | 无（源对非 2 对齐情况的降级） | 有（整块连续覆盖后不再需要按 head 对齐） | **舍弃** | §0.6 R5：NPU 按「整块连续列」访问，`head_size=65` 等奇数列不再触发标量回退（对应测试 `test_rope_rotates_the_leading_columns_in_either_convention[odd-head]`） |
| 13 | `@target_cache(maxsize=32)` | 框架 | 无 | 有（wrapper 的 `_programs` dict 缓存，已固化） | **保留** | `examples/TileOPs/tileops/kernels/rope/rope/rope.py` `_program` |
| 14 | autotune / `tune` | 框架 | GPU 计时搜索 | 无（Stage 0 已移除，K8） | **舍弃** | wrapper `default_config` 只有 `units_per_block`；搜索空间 1 维 |
| 15 | 无 matrix/mma、无 warp shuffle、无 swizzle、无 `T.Pipelined`、无 atomics | — | — | — | **不适用（源未使用）** | §0.4 注：源文件无这些构造，无优化意图需承接 |

### 0.6 NPU 算法重设计

对 §0.5 判「重新设计」的 4 项（R1/R3/R4/R6）与 3 项结构性重写（R2/R5/R7）逐项给出 NPU 算法与语义保持论证。**调研结论（§1.6.0）是本节输入**：调研选定「整块 tile 暂存 + 单遍直通」算法族，与本节的 R2/R3 一致。

**重设计项 R1：3 级 grid/thread 折叠 → 1-D core grid + `T.SimtVF` 线程域**

- **源方案**：`T.Kernel(num_tokens * parts, threads=threads)`，块内 `threadIdx` 逐单元，`bx // parts`、`bx % parts` 给出 (token, part)。
- **NPU 新算法**：`T.Kernel(nblocks) as bx`（1-D core grid，无 `threads` 参数）+ 块内 `T.SimtVF(threads=256)` 打开 lane 域；块索引折叠为三元组 `(side, rb, cb)`：
  `nrb = ceildiv(tokens, bt)`（动态）；`nq = nrb · nchunks_q`；`bx < nq` ⇒ `side=query, rb = bx // nchunks_q, cb = bx % nchunks_q`；否则 `t = bx - nq, side=key, rb = t // nchunks_k, cb = t % nchunks_k`。
- **语义保持论证**：`[0, nblocks)` 到 `[0,nrb) × ([0,nchunks_q) ⊎ [0,nchunks_k))` 是双射（行主序展开）⇒ 每 (token 行块, 列块, 张量侧) 恰被一个块处理，无遗漏、无重叠；块内元素集合与源实现相同（同一 token 行、同一列区间）。线程数只影响元素在 lane 上的摊分，不改任何算式 ⇒ 逐元素映射与线程无关，结果恒等。

**重设计项 R2：按 head 的「单元」→ 按整块 (行, 列) 的 tile**

- **源方案**：单元 = 一个 head 内 `rot_vec` 列（NeoX 低半 + 对应高半；GPT-J 相邻若干整对）或 `tail_vec` 列尾列；单元宽度受 16 B 对齐与 head 边界约束，非对齐时降级为 `T.serial` 标量。
- **NPU 新算法**：块负责 `bt × cols` 个连续元素（`cols = hpc · D`，`hpc` 为 `H_side` 的、不超过 `max(1, 1024//D)` 的最大因子）；旋转按「对」展开为 2-D 并行 `T.Parallel(bt, hpc·half)`，`h = p // half`、`k = p % half`（编译期常量除法），列偏移 `lo = h·D + k`（NeoX）/ `h·D + 2k`（GPT-J），配对列 `hi = lo + half`（NeoX）/ `lo + 1`（GPT-J）；尾列不进入算术，由 R3 的整块拷贝覆盖。
- **语义保持论证**：对每一对 `(lo, hi)` 施加的算式与 §0.1 逐字相同（三角恒等式未改）；`(h, k)` 遍历 `[0,hpc) × [0,half)` 恰好一次 ⇒ 每对恰被处理一次；`head_size` 为奇数或 `R < D` 时不再需要「列偏移 16 B 对齐」这一 CUDA 约束（整块列连续），语义不变而路径更短。**`half` 需为向量宽度的整数倍**（本任务全部 workload 的 `half ∈ {16,32}`，fp16/bf16 向量宽度 8、fp32 为 4）——不满足时按 `h` 显式展开为 `hpc` 个 `T.Parallel(bt, half)` 循环（同语义，仅循环形态变化）。

**重设计项 R3：load-before-store 重排 + SMEM 行中转 → 整块 UB 双缓冲暂存与整块搬出**

- **源方案**：默认程序靠语句重排让「拷贝列的 load」不等 position/表（依赖 ptxas）；staged 程序用 SMEM 整行中转 + 一次 bulk store。
- **NPU 新算法（主选 P）**：每块
  ① `x_ub`（GM→UB 逐元素、按行谓词）→ ② 整块复制 `y_ub = x_ub`（一条并行 pass；备选 B 用 `T.copy`；尾列与填充行一并复制）→ ③ 旋转两遍写 `y_ub`（低半/高半各一遍，每遍一次 UB 写）→ ④ `y_ub`（UB→GM 逐元素、按行谓词）。
  读写缓冲物理分离 ⇒ **没有任何 load 需要等 store**，「先读后写」不再是正确性条件（源 #5 的意图由结构保证，而不是依赖编译器调度）。
- **语义保持论证**：`y_ub` 初值等于 `x_ub` 的逐位拷贝（P 用逐元素赋值、B 用 `T.copy`；两侧同 dtype ⇒ 无转换、无精度损失；`traps-compiler.md` C12 记录 `T.copy` 的隐式 dtype 转换只在跨 dtype 时发生，本设计所有拷贝两侧同 dtype）；旋转区在 ③ 被覆写，尾列保持 ①② 的字节不变 ⇒ 输出 = 「旋转区按 §0.1 计算 + 尾列逐位原样」；A3 项机器验证（9 组用例，含 `D=65`、`D=80`）逐位 0 失配。填充行（`row0+i ≥ tokens`）只写 UB、不写 GM ⇒ 输出张量只被合法行触达。

**重设计项 R4：每线程多单元（`units`）→ 每块固定元素预算（16 元素/lane）**

- **源方案**：`units=4`（16-bit）/`1`（fp32）每线程，靠寄存器级并行掩盖延迟。
- **NPU 新算法**：块 tile = `units_per_block` 对（= `tile_elems` 元素），默认 `units_per_block = 2048` 对，并被 4096 元素的 tile 上限夹住 ⇒ **4096 元素**，`threads=256` ⇒ **16 元素/lane**（与平台已验证的 `n = threads × 16` 点一致）；`bt = max(1, min(units_per_block // (hpc·half), max(1, 4096 // cols)))`（两 dtype 同一取值：fp32 的 tile 也是 4096 元素 / 16 KB）。
- **语义保持论证**：`units_per_block` 只决定 tile 形状与块数，不进入任何算式；lane 摊分方式对逐元素映射无影响 ⇒ 结果与 tile 选择无关（tile 只影响性能）。机器验证无需覆盖（纯调度参数）。

**重设计项 R5：按 head 的向量宽度自适应 → 128-bit 向量 + 整块连续列**

- **源方案**：`rot_vec`/`tail_vec`/`row_vec` 取「整除所有列偏移的最大 2 的幂（≤ lanes）」，非整除时按 `T.serial` 标量访问。
- **NPU 新算法**：保留三个参数（wrapper 调用点不变），但 NPU 侧访问宽度由「列是否连续」决定：tile 内列连续 ⇒ 一律 128-bit 向量；旋转对的跨 lane 连续性由 `half` 与向量宽度的整除性保证（见 R2）。三个参数在 NPU 路径**不再驱动访存形状**，仅作接口兼容（它们的取值只改变 CUDA 访问宽度，属纯性能信息，忽略不改变语义）。
- **语义保持论证**：向量宽度是访存形状而非语义；同一元素集合、同一算式 ⇒ 逐位相同。`head_size=65` / `D=80` 等奇数列在 NPU 上不再走标量路径（源对应用例 `test_rope_heads_copying_columns[scalar-gptj]`、`test_rope_rotates_a_query_alone_with_copied_columns` 仍由 §8.3 覆盖）。

**重设计项 R6：两套程序 + 两级启发式 → 单一程序 + 预算参数**

- **源方案**：`_shape(tokens)` 在 wide/config 间选；`_RoPEStagedRowsKernel._shape` 按 SMEM 预算与常驻线程数在 staged/positions 间选。
- **NPU 新算法**：只有一套程序（P）；`_STAGED` 与 `wide` 两个入参保留在调用签名中但**编译同一 prim_func**（NPU 无 SM/常驻线程/SMEM-预算事实，启发式无输入）。备选程序 B（无 `y_ub`、逐元素直写 GM，见 §3.3）由模块常量 `_NPU_PROGRAM` 选定，不参与运行时选择。
- **语义保持论证**：程序选择只影响性能与访存路径，全部候选实现同一算式 ⇒ 结果恒等（B 路径的算式与 P 相同，只是写目标为 GM 而非 UB）。

**重设计项 R7：源 int64 行号 + 每单元重复读 position → clamp 后窄化 int32（每行一次）**

- **源方案**：`row` 存为 position dtype（int64 或 int32），每 block clamp 一次；每单元用它算表地址（int64 地址运算）。
- **NPU 新算法**：`seen = int64(POS[r]) + int64(OFF[r])`（无 offsets 时 `+0`）→ `clamp(seen, 0, P-1)` → `T.cast(..., "int32")` 存入 `row_ub[i]`；每行一次，块内所有 lane 复用。
- **语义保持论证**：clamp 值域是 `[0, P-1]`，`P ≤ 2^31` 恒成立（表在内存中）⇒ 窄化不改变值（A2 项对 `-70..P+70` 全枚举 + int32/int64 极端值验证 0 失配；A2 反例项给出「先窄化后 clamp」会回绕 5/8341 的证据）。int64 求和避免 `int32` 位置 + `int32` offsets 溢出，与源 `T.Cast("int64", ...)` 逐字一致。

**R8（结构性，随 R3）：配对算式拆成两个单写循环**

- **源方案**：一个单元一次写低半与高半（两处 store 在同一段代码里）。
- **NPU 新算法**：低半、高半各一个 `T.Parallel` 循环，每个循环体内**只有一次存储**。
- **语义保持论证**：两遍读同一对输入、写不同列，结果与单遍两写逐位相同（读数不变、算式不变）；拆分的理由见 §1.6.2 #5（平台实测：一个并行体内两次 store 会使向量化退化；`examples/paged_cache_write/_paged_cache_write_kernel/_paged_cache_write_kernel.py` L101–107 的 fp8 双 store 记录）。

> **§0.4 优化意图承接核对**（编号按 §0.4 的 12 行）：保留 4 项（#7 fp32 单次舍入＝语义、#8 动态 token + 编译缓存、#9 单 launch 服务两侧、#10 尾列直通）；等价替换 4 项（#3 16 B 向量宽度→128-bit 列连续、#4 表行复用→A1、#6 SMEM 中转→主选 P 的 UB 双缓冲、#12 warp 分布→`T.SimtVF` lane 域）；重新设计 4 项（#1 每线程多单元→R4 元素预算、#2 块↔token 折叠→R1 三元折叠、#5 load-before-store→R3 双缓冲、#11 两套程序+启发式→R6 单一程序）；舍弃 2 项（§0.5 #12 的 `T.serial` 标量回退分支——它是 §0.4 #3 在非整除时的降级，由 R5 的「整块连续列」消去；§0.5 #14 autotune——Stage 0 已移除、搜索空间仅 1 维）。合计 4 + 4 + 4 + 2 = 14 项有处置，第 15 行（源未使用的 matrix/shuffle/swizzle/pipeline/atomics）不适用。无静默丢弃项。

### 0.7 标杆实现

| 用途 | 路径 | 说明 |
|------|------|------|
| 源算子（GPU TileLang） | `/workspace/TileOPs/src/tileops/_kernels/rope.py` | §0.1–§0.4 的解读对象；四个 kernel 类共用 `_rope_positions_kernel(query_heads, key_heads, head_size, rotary_dim, max_position, dtype, position_dtype, offset_dtype, neox, rot_vec, tail_vec, row_vec)` |
| 抽取文件（Stage 0 现态） | `examples/TileOPs/tileops/kernels/rope/rope/_ro_p_e_fwd_kernels.py` | 待 Stage 3 按 §3.3 重写为 NPU 工厂（内层 `kernel(units_per_block, staged)`） |
| 冻结接口（Stage 0） | `examples/TileOPs/tileops/kernels/rope/rope/rope.py` | `default_config = {"units_per_block": ...}`、`forward` 的三分支（query-only / key / offsets）与 `out_idx` 约定；本文设计须与其一致 |
| golden 基准（独立） | `examples/TileOPs/tileops/workloads/rope.py` 的 `rope_tables` + `rope_reference` | 以 §0.1 为唯一依据，`fp32` 计算后一次舍入；§8.1 直接移植 |
| L0 基准用例 | `examples/TileOPs/tests/ops/test_rope.py`（33 例） | 语义边界来源：clamp、0 头 key、奇数 head、batched positions、int32 位置 |
| 规格与 workload | `examples/TileOPs/tileops/manifest/rope.yaml`（`RoPEFwdOp`，6 条 workload） | roofline `vars/flops/bytes` 机器可读来源 |
| 性能用例 | `examples/TileOPs/benchmarks/ops/bench_rope.py` | manifest 驱动的 6 workload × dtype |
| 同平台结构参考 | `examples/paged_cache_write/_paged_cache_write_kernel/DESIGN.md` | 1-D grid 折叠 + `T.SimtVF` + 动态 token + lane 谓词的已验证组织方式（API 证据链见其 §3.4） |
| 同族数学参考（另一后端） | `examples/posembedding/rope_llama31/kernels/table_low_precision.py` | 同平台仓库内的 RoPE 实现，**使用 `tilelang.language` / `T.Kernel(..., is_npu=True)` / `T.alloc_ub`（另一工具链方言）**，只借鉴其「表驻留 + 双半旋转」的数学组织，API 一律按 §3.2 的 ascend 方言重写 |

> golden 独立性：§8.1 只依 §0.1，**不复刻** §0.6 的 NPU 算法（不用 1-D grid、不用 `T.SimtVF`、不做 int32 窄化、不依赖行谓词）。

---

## 1. 概述

### 1.1 算子名称

`_rope_positions_kernel`（`RoPEFwdOp` 的 NPU kernel 工厂；四个 GPU kernel 类 `RoPENeoxPositionIdsKernel` / `RoPENonNeoxKernel` / `RoPENeoxStagedRowsKernel` / `RoPENonNeoxStagedRowsKernel` 在本平台编译为同一程序，见 §0.6 R6）。

### 1.2 功能描述

按每个 token 的位置索引 cos/sin 表的一行，对 `query` / `key` 每个 head 的前 `rotary_dim` 列做二维旋转（NeoX 前后半配对，或 GPT-J 相邻配对），其余列原样拷贝；位置越界时钳到表边界；一次 launch 同时服务 query 与 key（可无 key、可带 offsets）。

### 1.3 数学公式

记 `D = head_size`、`R = rotary_dim`、`h = R/2`、`P = max_position`、`Hq/Hk` 为 query/key 头数、`x_t = x[t, :]` 为第 `t` 个 token 的行（`D·H` 列，`H` = 该侧的 `Hq` 或 `Hk`）：

$$
r_t = \operatorname{clamp}\big(\operatorname{int64}(p_t) + \operatorname{int64}(o_t),\;0,\;P-1\big),\qquad
c^{(t)}_k = \cos[r_t, k],\quad s^{(t)}_k = \sin[r_t, k],\quad 0 \le k < h
$$

**NeoX**（`neox=True`），对 `0 ≤ j < H`、`0 ≤ k < h`、`lo = jD + k`、`hi = jD + k + h`：

$$
y[t, lo] = \mathrm{cast}\!\left(\mathrm{f32}(x[t,lo])\cdot c^{(t)}_k - \mathrm{f32}(x[t,hi])\cdot s^{(t)}_k\right),\quad
y[t, hi] = \mathrm{cast}\!\left(\mathrm{f32}(x[t,hi])\cdot c^{(t)}_k + \mathrm{f32}(x[t,lo])\cdot s^{(t)}_k\right)
$$

**GPT-J / interleaved**（`neox=False`），对 `lo = jD + 2k`、`hi = jD + 2k + 1`：同一对式，仅配对列不同。

**尾列**：`y[t, jD + R + q] = x[t, jD + R + q]`（`0 ≤ q < D-R`，逐位拷贝）。

`f32(·)` 为升到 float32，`cast(·)` 为按输出 dtype 舍入一次；表值 `c/s` 是调用方按 dtype 预先舍入的输入，**不得**在 kernel 内重算（X3 项证据）。

### 1.4 算法描述

本节描述**迁移决策后的 NPU 算法**（§0.6 落地形态），与源算法的差异逐条标注来源：

1. **任务映射（差异 ← §0.6 R1）**：1-D core grid，块索引折叠为 `(side, rb, cb)`；`nblocks = ceildiv(tokens, bt)·(nchunks_q + nchunks_k)`。
2. **tile（差异 ← §0.6 R2/R4）**：块负责 `bt` 行 × `cols = hpc·D` 列的连续 tile（默认 4096 元素），`T.SimtVF(threads=256)` 打开 lane 域，`T.Parallel` 把 tile 摊到 256 lane（16 元素/lane）。
3. **整块 UB 双缓冲（差异 ← §0.6 R3）**：`x_ub`（GM→UB）、一条整块复制 pass（`y_ub = x_ub`）、两遍旋转写 `y_ub`、`y_ub`（UB→GM）；尾列靠这次整块复制直通。
4. **表行复用（保留源 #4 ← §1.6.1 A1）**：每行一次 `row_ub[i]`（clamp 后 int32），`cs_ub/sn_ub` 每行 `half` 个值，块内所有 head 复用。
5. **fp32 单次舍入（保留语义 ← §0.4 #7）**：`T.cast(..., "float32")` 进、算完 `T.cast(..., dtype)` 出；fp32 输入时 cast 为恒等。
6. **单 launch 服务两侧（保留源 #9 ← §0.4 #9）**：query 块在前、key 块在后（offset 变体另有 offsets 张量）。
7. **无 head 对齐约束（差异 ← §0.6 R5）**：列连续 ⇒ `head_size` 奇数、`R<D`、`hpc` 不整除头数等情形都不再降级为标量路径（`hpc` 仍取 `H_side` 的因子以保持「无列尾块」）。
8. **程序唯一（差异 ← §0.6 R6）**：`_STAGED`/`wide` 两个入参在 NPU 路径编译同一 prim_func；备选程序 B（直写 GM）由模块常量选择（§3.3）。

**与源算法的一致性核对**：元素级算式（第 5 条）与源 `_rotate_pair` 逐字相同；配对列（第 4 条）、行号 clamp（§0.6 R7）、尾列直通（第 3 条）与源语义一致。被替换的只有任务映射、tile 形状、程序选择与暂存方式。无归约 ⇒ 不存在累加顺序差异。

### 1.5 数据流图

```
GM[POS(num_tokens,)]  --每行一次--> 标量 pos(+OFF) --clamp--> int32 row_ub(bt)
GM[COS(P,R/2)]        --每行 half 个--> cs_ub(bt,half)   （块内所有 head 复用：A1）
GM[SIN(P,R/2)]        --每行 half 个--> sn_ub(bt,half)

GM[query/key 行块] --按行谓词, 列连续--> UB[x_ub(bt,cols)] --整块复制 pass--> UB[y_ub(bt,cols)]
                                        y_ub 旋转区: 低半循环 / 高半循环 各写一次（fp32 算术、一次舍入）
UB[y_ub] --按行谓词, 列连续--> GM[query_out/key_out 行块]

路径口径：GM → UB → GM，每元素读一次、写一次；表按「每行一次、块内复用」计；无 L1/L0 参与（纯 Vector）。
```

### 1.6 算法调研与优化分析 ⭐

> 顺序：先调研（1.6.0 四问：等价化简 / 在线算法 / 复杂度 / 硬件亲和，输入公式与源算法只是基线候选之一）→ 数学等价优化（1.6.1，含机器验证）→ 向量化替代（1.6.2）→ 向量化轴与布局（1.6.3）。1.6.1–1.6.3 的分析对象 = 1.6.0 选定并经 §0.6 落地的 NPU 算法。

#### 1.6.0 算法调研（Algorithm Research）⭐

> **调研对象**：同一数学语义（「按行索引预计算表 + 成对二维旋转 + 尾列直通」）的算法族。本算子是**单遍逐元素**类（无归约、无窗口、无矩阵），按 `algorithm-research.md` §2 的**轻量调研**档执行：四问各给明确结论 + 依据，四问不跳过；因算子是带宽型，R3 仍给出四口径对照表与设计期 roofline 下界。源 GPU 直译 = 基线候选 #0。

**R1 等价化简公式候选**（基线在表内；正式等价论证与收益量化在 §1.6.1）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 纳入 R3 |
|---|------|------------|-----------|---------|--------|
| 0 | 基线：源直译（每单元读 position/表、逐单元向量） | `y_lo = x_lo·c - x_hi·s`；表按单元读 | — | — | ✅ 基线 |
| 1 | 表行复用 + 整块连续 tile（P，选定） | 同一算式，表按行读一次；tile = `bt × hpc·D` | 逐位等价（同一批表值，只是读取次数减少；A1 0 失配） | 表流量 `O(H·R)`→`O(R+每块一次)`；源基线约 5.9%→2.4% 额外流量 | ✅ |
| 2 | 三角恒等式重写：`y = x·cos + rotate(x)·sin`（矩阵形式） | 与 #0 同一式子换写法 | 恒等（矩阵乘即两式） | 无（算术量相同） | ✅（并入 #1） |
| 3 | 极坐标重写：`y_lo = r·cos(θ+φ)`、`y_hi = r·sin(θ+φ)`（`r = hypot(x_lo,x_hi)`、`φ = atan2`） | 用幅角相加代替两条乘加 | **容差内不等价**：引入 `sqrt/atan2/cos/sin` 四个超越函数，且中间舍入路径完全不同 | 算术更贵（超越函数 vs 4 mul + 2 add），带宽型算子上无收益 | ❌ |
| 4 | 表即时重算（kernel 内由 position 生成 cos/sin，省表读） | `angle = p·base^(-2k/R)` → cos/sin | **不等价**：契约的表是「fp32 角度 → dtype 舍入」的输入；X3 项 96%+ 元素失配 | 省 ≤6% 流量，代价是每元素超越函数 + 语义破坏 | ❌ |
| 5 | FMA 单次舍入（`fma(-x_hi, s, x_lo·c)`） | 合并乘加 | **fp32 输出不等价**（X2：fp32 3132/6144 元素位差）；16-bit 输出舍入后位同 | 算术 -1 op/值 | ❌（带宽型，无收益） |
| 6 | 原位旋转（in-place，省一次 UB 拷贝） | 直接在 `x_ub` 上写 | **不正确**：跨 lane 读同一 tile 的另一列会与本 lane 的写冲突（RAW） | 省一次 UB→UB 拷贝 | ❌（正确性） |

**R2 在线算法**：**无在线变体**（结论「无」）。结构依据：输出元素 `(t, col)` 由同一 token 的 2 个输入元素、2 个表值与 1 个行号唯一决定，**不存在跨元素/跨 tile 的中间统计量**（无 max/sum/平方和/计数），因此「分块 running 统计量 + 重缩放」这类在线算法在数学上没有可压缩对象：每个元素天然只读一次、只写一次，**单遍即最优**。对照 softmax/RMSNorm 之所以有在线算法，是其输出依赖整行归约；本算子的列之间彼此独立，token 之间也彼此独立。故 R3 全部候选的扫描遍数下界均为 1，在线化无收益空间。（迁移任务的源侧证据：§0.4 的 12 项手段中没有任何 running-statistic 结构，与结构判据一致。）

**R3 复杂度对比**（四口径；代表 workload = `llama3-8b-prefill-4k`：`tokens=4096`、`Hq=32/Hk=8`、`D=128`、`R=128`、bf16，共 5120 块）：

| 算法候选 | FLOPs | 访存 Bytes（含什么／不含什么） | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|-------|------------------------------|---------|-------------|---------------------|
| #0 基线（逐单元读表/位置） | `3·tokens·R·(Hq+Hk)` = 62.9 M（含 4 mul + 2 add 与尾列 0 算术的契约口径：每旋转值 3 FLOP） | 数据 83.9 MB + 表按单元读 `R` 值/head ≈ 33.6 MB + positions 每 token 一次 ≈ 0.03 MB ≈ **117.5 MB**（不含 L2 命中） | 1 | 0（无片上暂存；寄存器级） | 4096×parts 块 / 56 AIV，块内 128 线程 |
| #1 表行复用 + 整块 tile（P，选定） | 同上 62.9 M | 数据 83.9 MB + 表每行 256 B × 5 个列块 ≈ 5.2 MB + positions 每行一次 ≈ 0.16 MB ≈ **89.3 MB**（不含 L2） | 1 | `x_ub + y_ub + cs_ub + sn_ub + row_ub` = 17.0 KB/块（16-bit，§4.5） | 5120 块 / 56 AIV = 91.4 波 |
| #3 极坐标重写 | 4 超越函数/对 ≈ 2.4 G 等效 op | 同 #1（表仍要读） | 1 | 需 `r/φ` 中间量（+8 KB/块，dtype fp32） | 同 #1 |
| #4 表即时重算 | 2 超越函数/元素 ≈ 5.3 G 等效 op | 数据 83.9 MB（省 5.2 MB 表）≈ **84.1 MB** | 1 | 无表缓冲（省 1 KB） | 同 #1 |

fp32 代表 workload（`llama3-8b-b2-2k`：`tokens=4096`、fp32、5120 块）同口径：#0 ≈ 245 MB、#1 ≈ 178.4 MB（数据 167.8 MB + 表 10.5 MB）；gptj-6b-prefill（`tokens=2048`、`D=256`、`R=64`、fp16）：#1 ≈ 69.3 MB。全表见 §1.6.0 末「设计期估算」。

**R4 硬件亲和性评估**（逐候选对照清单 + 设计期 roofline；常数与算法见本节末）：

| 候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|------|-------------|---------|------------|---------|------------|------|
| #0 基线 | ✅ 纯 Vector（乘法/加法 + 两路 load/store，无 Cube 需求） | ✅ 无片上缓冲 | ⚠️ 每单元重复读表（同一表行被 `H_side` 个 head 各读一次）：`R` 值/head ⇒ 33.6 MB | ✅ 同 #1 | ❌ 无 stage 可叠 | ❌ **淘汰**：流量 117.5 MB vs #1 89.3 MB（1.32×），估算下界 79.9 µs vs 60.7 µs（引用 `CONST-mte2-degradation`） |
| #1 表行复用 + 整块 tile（P） | ✅ 纯 Vector：`T.Parallel` 内 `vcast/vmul/vsub` + `T.copy`（GM↔UB 与 UB↔UB），无 Cube、无归约 | ✅ 17.0 KB（16-bit）/34.0 KB（fp32）≪ 248 KB；按平台膨胀口径（20/26 B/elem）最坏 81.9/106.5 KB 仍 ≤ 248 KB | ✅ 列连续 ⇒ 128-bit 向量；`half ∈ {16,32}` 为 8/4 的整数倍；列块整除头数（`hpc` 取因子） | ✅ 唯一动态量 `tokens`，由 grid 上界 + 行谓词覆盖 | ⚠️ 单 tile/块，无跨 tile 复用 ⇒ 无 `T.Pipelined` 空间（§6.3） | **✅ 选定**：流量最低（89.3 MB），下界 60.7 µs |
| #3 极坐标重写 | ⚠️ `hypot/atan2/cos/sin` 为超越函数，向量单元吞吐远低于乘加（未实测假设，估算按 SFU 1/4 折算） | ✅ | ✅ | ✅ | ❌ | ❌ 淘汰：等效算力需求 ~38×（2.4 G vs 62.9 M），带宽型算子上把瓶颈从访存推到 ALU |
| #4 表即时重算 | ⚠️ 同上 + 破坏语义 | ✅ | ✅ | ✅ | ❌ | ❌ 淘汰：**数值不等价**（X3 96%+ 元素失配），且只省 6% 流量 |
| #5 FMA | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ 淘汰：**fp32 输出不等价**（X2 3132/6144 位差），带宽型无收益 |
| #6 原位旋转 | ✅ | ✅（省一个 buffer） | ✅ | ✅ | ✅ | ❌ 淘汰：RAW 冲突（同一 tile 内跨 lane 读写重叠），正确性问题 |
| 表驻留（把整表 `P×R/2` 常驻 UB 跨块复用） | ✅ | ❌ `P·R·itemsize` = 8192×128×2 = 2.0 MB ≫ 248 KB（`llama3-8b`），只有 `P ≤ 950` 左右才可行 | — | — | — | ❌ 淘汰（容量，引用 `CONST-capacity-910B2C` 的 UB 上限口径；本平台 UB 253952 B 见 §4.5） |

**设计期估算（D-2）**：代表 workload 全部 6 条（块数按 §5.2 的 `bt/hpc` 规则算）。流量项 = `流量 ÷ 1.47 TB/s`；发射项（DMA 段粒度）= `流量/128 B × 4 ns ÷ 56 AIV`（`CONST-mte2-degradation` 的指令/段维度口径）；ALU 项 = `元素数 × 6 ÷ (128 lane × 1.5 GHz × 56)`（**未实测假设**，仅作量级校核）。

| workload | 块数 | 流量 (Bytes) | 流量下界 (µs) | 段发射项 (µs) | ALU 项 (µs) | 估算下界 (µs) |
|----------|------|-------------|--------------|--------------|------------|--------------|
| llama3-8b-decode-b64（64 tok, bf16） | 80 | 1,396,096 | 0.9 | 0.8 | 0.2 | **0.9** |
| **llama3-8b-prefill-4k（4096 tok, bf16）** | **5120** | **89,292,800** | **60.7** | **49.8** | **11.7** | **60.7** |
| llama3-8b-b2-2k（4096 tok, fp32） | 5120 | 178,421,760 | 121.4 | 99.6 | 11.7 | **121.4** |
| gptj-6b-prefill（2048 tok, fp16） | 4096 | 69,340,160 | 47.2 | 38.7 | 9.4 | **47.2** |
| gptj-6b-decode-b32（32 tok, fp16） | 64 | 1,082,112 | 0.7 | 0.6 | 0.1 | **0.7**（启动地板） |
| phi-2-query-only（1000 tok, bf16, int32） | 668 | 10,511,360 | 7.2 | 5.9 | 1.4 | **7.2** |

**估算下界: 60.7 us（流量 60.7 + 发射 49.8 + 容量 0.0，引用 CONST-mte2-degradation / CONST-vector-launch-overhead）**

估算口径与常数纪律：

- **流量项常数**：`CONST-mte2-degradation`（Ascend910B2C，tilelang 0.1.2+15ad002b3d / CANN 8.5.0 / 2026-09-21，48 vector core）给出「3:1 读写混合全流量 ≈ 1.26 TB/s」地板；本平台 56 AIV，按核数线性外推 `1.26 × 56/48 = 1.47 TB/s`——**跨型号外推，未在 950PR 实测**（`knowledge/bottlenecks.md` 明确「当前没有 950 真机验证的瓶颈模式」），只用于候选间 1.32× 量级淘汰与下界量级，不作绝对时延承诺（§9 R9）。
- **段发射项**：同条目 2026-09-17 的指令维度口径（≈4 ns / 128 B 段，910B2C），折算 `56 × 32 GB/s = 1.79 TB/s` 段速率；与流量项互证（1.79 ≥ 1.47 ⇒ 段发射不成为地板）。
- **`CONST-vector-launch-overhead`**（0.5 µs/op，910B2C，0.1.2+3a214cde / 2026-09-09）：本设计**不用**它作逐 block 项（1.6.0 的 tile 元素数是 4096，而该常数的 op 口径与 tile 大小绑定的方式未文档化，直接套用会把估算推到毫秒级而显然失真）；改为在上面用段粒度口径覆盖搬运发射，并在 §9 R9 登记该取舍。仅当 Stage 4 实测出块级固定开销显著时，才按该常数重估。
- **容量项**：0（无跨块驻留，单块 UB 用量见 §4.5，远低于容量）。
- **ALU 项**：`128 lane × 1.5 GHz` 为**未实测假设**（按 Ascend 向量单元 256 B/周期、fp16 128 lane 折算），只用来确认 ALU 不是地板（11.7 ≪ 60.7）。

**调研结论**：选定算法族 **#1「表行复用 + 整块连续 tile + UB 双缓冲单遍直通」**，关键依据三条：(a) R3 显示它把表读从「按单元（每 head 一次）」压到「按行一次」（117.5 MB → 89.3 MB，-24%），且已是「每元素一读一写」的流量下界（数据项 83.9 MB = 契约 roofline 的 `2·(q+k)`）；(b) R4 显示计算全落在 `vcast/vmul/vsub` + `T.copy` 这些在本平台有仿真验证的算子上（`PL-950-conversion-layout`、`PL-950-row-primitives`、`PL-950-vector-execution`），无 Cube、无归约、无流水需求；(c) 与源算法结构同构（同一算式、同一配对、同一 clamp），语义漂移最小。**调研范围**：`algorithm-research.md` §5 参考表（经 `kb_search "rope rotary position embedding"` 检索：本平台 `knowledge/algorithm-candidates.md` 与 pattern-library 均**无 rope 命中条目**，故候选由源算法 + 三角恒等自查 + `examples/posembedding/rope_llama31` 同族实现补足）；pattern-library 的 `constants.md`（4 条常数）、`elementwise.md`（PL-1.6/1.10/1.17/1.19）、`traps-compiler.md`（C10/C12/TRAP-parallel2d-bm-ge2-scalarize/TRAP-UB-multibuffer-inflation）；平台知识 `PL-950-vector-execution`、`PL-950-row-primitives`、`PL-950-conversion-layout`、`PL-950-simd-reduce`、`PL-950-bottlenecks`；结构分析（无归约 ⇒ 无在线空间、单遍即最优）。**源算法优化意图承接**：见 §0.6 末的核对表（12 项全部有处置，无静默丢弃）。

#### 1.6.1 数学等价优化（公式级）

> 分析对象 = §1.6.0 选定的 #1。三项采纳项经 `verify_equiv.py`（torch CPU + fp64 参照，本目录）机器验证，全部 `EQUIV_PASS`；三项被否决项的失配数字见 §11.1。

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| A1 | 表行复用（访存削减） | 每 (head, 对) 读 `cos[row,k]`、`sin[row,k]`（源按单元） | 每行一次 `row_ub[i]`，`cs_ub[i,:] = COS[row_ub[i], :]`、`sn_ub[i,:] = SIN[...]`，块内所有 head 复用 | 取到的表值集合与逐单元读完全相同（同一 `row`、同一 `k`），只是读取次数减少 ⇒ 逐位相同。A1 项用两套独立寻址实现对照（5 组配对/头宽配置 × 3 dtype = 15 组），0 失配 | 表流量：llama-prefill 33.6 MB → 5.2 MB（-85%）；总计 117.5 MB → 89.3 MB |
| A2 | clamp 后窄化 int64→int32 | `row` 全程 int64（地址运算 64 位） | `seen = int64(pos)+int64(off); row = int32(clamp(seen,0,P-1))` | clamp 值域 `[0,P-1]` 且 `P < 2^31` ⇒ 窄化为恒等；int64 求和避免 int32 位置+offset 溢出（与源 `T.Cast("int64", ...)` 同）。A2 项对 `-70..P+70` 全枚举 + int32/int64 极端值 0 失配；反例项（先窄化后 clamp）在 5/8341 上回绕 | 表地址运算由 64 位降 32 位；位置读每行一次（int64 8 B → 每块 `bt` 次） |
| A3 | 尾列直通（整块拷贝） | 尾列按 `tail_vec` 单独循环读写（源 `tail_units` 分支） | 整块复制 `y_ub = x_ub`（P：一条并行拷贝 pass；B：`T.copy`），旋转区随后覆写 | 同 dtype 整块拷贝逐位不变（`T.copy` 的隐式转换只在跨 dtype 时发生，此处两侧同 dtype）；未覆写列即输出值 ⇔ 源「尾列直通」。A3 项 9 组（含 `D=65`、`D=80`、`R<D`）0 失配 | 尾列不产生任何额外循环/指令分支；`D>R` 用例（gptj、phi-2）省 1 条并行循环 |

**等价性机器验证（D-1，`verify_equiv.py` 执行结果）**：命令 `python examples/rope/_rope_positions_kernel/verify_equiv.py`（torch CPU，无设备参与），摘要如下（完整 40 行见脚本输出；对照组为 §0.1 基线的同 dtype 位级输出）。

| 项（配置） | 类型 | 对照元素数 | 逐位违反 | 违反率 | 容差违反 | 最大 ulp | 结论 |
|-----------|------|-----------|---------|--------|---------|---------|------|
| A1 表行复用 `D=64 R=64 neox` fp16 | 采纳 | 6144 | 0 | 0.000e+00 | 0 | 0.000 | **EQUIV_PASS** |
| A1 表行复用 `D=64 R=64 gptj` fp16 | 采纳 | 6144 | 0 | 0.000e+00 | 0 | 0.000 | **EQUIV_PASS** |
| A1 表行复用 `D=80 R=32 neox` bf16 | 采纳 | 7680 | 0 | 0.000e+00 | 0 | 0.000 | **EQUIV_PASS** |
| A1 表行复用 `D=65 R=32 gptj` fp32 | 采纳 | 6240 | 0 | 0.000e+00 | 0 | 0.000 | **EQUIV_PASS** |
| A1 表行复用 `D=256 R=64 gptj` 三种 dtype 共 3 行 | 采纳 | 12288/行 | 0 | 0.000e+00 | 0 | 0.000 | **EQUIV_PASS** |
| A2 clamp 后窄化（`-70..P+70` 全枚举 + 极端值） | 采纳 | 8341 | 0 | 0.000e+00 | 0 | 0.000 | **EQUIV_PASS** |
| A3 尾列直通 `D=80 R=32 neox` fp16 / bf16 / fp32 | 采纳 | 7680/行 | 0 | 0.000e+00 | 0 | 0.000 | **EQUIV_PASS** |
| A3 尾列直通 `D=65 R=32 gptj` fp16 / bf16 / fp32 | 采纳 | 6240/行 | 0 | 0.000e+00 | 0 | 0.000 | **EQUIV_PASS** |
| A3 尾列直通 `D=256 R=64 gptj` fp16 / bf16 / fp32 | 采纳 | 12288/行 | 0 | 0.000e+00 | 0 | 0.000 | **EQUIV_PASS** |
| G1 golden(f32 单次舍入) vs fp64 参照，fp16/bf16 5 组配置 ×2 dtype | 检查项 | 6144–12288/行 | 0 | 0.000e+00 | 0 | 0.000 | **EQUIV_PASS** |
| G1 golden vs fp64 参照，fp32 5 组配置 | 检查项 | 6144–12288/行 | 880–1935 | ≤3.15e-01 | **0** | ≤2142 | **EQUIV_PASS**（容差口径；fp32 位差来自 fp32 两步乘加的 1 ulp 级差异 + 近零元素的对消放大，`atol=rtol=1e-5` 下 0 违反） |
| X1 原生 dtype 逐 op 舍入（否决，见 §11.1） | 否决 | 6144–12288/行 | 722–1760 | ≤2.87e-01 | 0 | ≤15041 | **EQUIV_FAIL**（位级不等价） |
| X2 FMA 单次舍入（否决） | 否决 | 6144–12288/行 | fp16/bf16 0；fp32 878–1924 | ≤3.13e-01 | 0 | ≤4117 | **EQUIV_FAIL**（fp32 位级不等价） |
| X3 表即时重算（否决） | 否决 | 16384 | 15785–15886 | ≥9.63e-01 | 15202–15871 | ≤3.4e4 | **EQUIV_FAIL** |

脚本收尾行：`adopted items: 40, failed: 0` / `ADOPTED_VERDICT: ALL_PASS`。验证域：`neox ∈ {True, False}` × `(D,R) ∈ {(64,64),(80,32),(65,32),(256,64)}` × 3 dtype，输入含随机正态与角点集（`±0, ±1, ±inf, NaN, 65504, ±6.1e-5, ±3.4e38`，覆盖上溢/下溢/次正规），位置含负值、越界值、`±2^31±1`、`±2^32±1`、`±2^40`。

**优化结论**：采纳 3 项（A1/A2/A3），全部逐位 `EQUIV_PASS`。**优化后公式**（§3.1 的唯一输入）：

```
每块：
  row_i   = int32( clamp( int64(POS[r_i]) + int64(OFF[r_i] or 0), 0, P-1 ) )        # A2
  cs[i,:] = COS[row_i, :] , sn[i,:] = SIN[row_i, :]                                # A1（每行一次）
  x_ub    = X[rows, c0:c0+cols]                                                    # 行谓词
  y_ub    = copy(x_ub)                                                             # A3（尾列直通）
  对每个 (i, h, k)，lo/hi 按 neox 取：
     y_ub[i,lo] = dtype( f32(x_ub[i,lo])·f32(cs[i,k]) - f32(x_ub[i,hi])·f32(sn[i,k]) )
  对每个 (i, h, k)：
     y_ub[i,hi] = dtype( f32(x_ub[i,hi])·f32(cs[i,k]) + f32(x_ub[i,lo])·f32(sn[i,k]) )
  Y[rows, c0:c0+cols] = y_ub                                                       # 行谓词
```

无优化空间的方向已核实并写明结论：**旋转核心无代数化简**（二维旋转是 4 mul + 2 add 的下界；R1 的 #3/#5/#6 分别因超越函数代价、fp32 位级不等价、RAW 冲突被否决）；**无公共子表达式可跨 lane 复用**（每个 lane 的 (x_lo, x_hi, c, s) 互不相同，唯一的公共项是表行本身，已由 A1 承接）；**无高代价算子可降代**（无 `exp/log/sqrt/rsqrt`）；**访存已到「每元素一读一写」下界**（数据项 = 契约 roofline 的 `2·(q+k)`，其余项由 A1/A2 消尽）；**无归约可合并**（无归约）。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

> 盘点 §3.3 伪代码的全部循环与标量计算点，逐项给出向量替代或不可替代的具体理由。API 佐证见 §3.4。

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由（不可替代时必填） |
|---|--------|-----------|-------------|---------|------------------------------|
| 1 | tile 载入 `x_ub[i,c] = X[row,c0+c]`（`bt×cols` 元素） | 标量二重循环 | `for i, c in T.Parallel(bt, cols)`（内层 `c` 连续 ⇒ 128-bit 向量；`T.SimtVF(threads=256)` 摊 16 元素/lane） | ✅ | — |
| 2 | 表行载入 `cs_ub/sn_ub`（`bt×half`） | 标量循环 | 同上（两条独立 `T.Parallel`，每条一次 store；avoid 单体内双 store，平台实测见 §0.6 R8） | ✅ | — |
| 3 | 低半旋转写 `y_ub[i,lo]` | 标量 `for` | `for i, p in T.Parallel(bt, hpc·half)`，`h=p//half`、`k=p%half`（编译期常量除法），表达式内 `vcast/vmul/vsub` + 一次 `vcast` 回写 | ✅ | — |
| 4 | 高半旋转写 `y_ub[i,hi]` | 标量 `for` | 同 #3 的第二条循环（单写/体） | ✅ | — |
| 5 | 尾列直通 | 标量逐列拷贝（源 `tail_units` 循环） | `T.copy(x_ub, y_ub)` 整块拷贝（同 dtype、元素数相等） | ✅ | — |
| 6 | tile 写回 `Y[row,c0+c] = y_ub[i,c]` | 标量二重循环 | `for i, c in T.Parallel(bt, cols)`（同 #1） | ✅ | — |
| 7 | 行号计算 `row_i = clamp(int64(pos)+int64(off))` | 每行标量 | 无（保留为 `T.Parallel(bt)` 的 8–16 lane 标量链） | ❌ | **地址/索引计算**：结果作为后续 gather 的行基址（每 lane 都需要同一个值），是 O(`bt`) 的映射计算而**不是** O(`bt·cols`) 的逐元素热点；改写成向量归约无对象（无跨元素运算） |
| 8 | 行谓词 `row0+i < tokens` | lane 级标量比较 | 无（保留为 `T.Parallel` 内的条件语句） | ❌ | **依赖动态 shape 的边界处理**：`tokens` 运行期才知（`T.dynamic`），grid 上界 `ceildiv(tokens,bt)` 必然含不满的尾块；无法静态化，也不宜 padding（会引入越界读或多余写） |
| 9 | 块索引折叠 `bx < nq`、`rb = bx//nchunks`、`cb = bx%nchunks` | 每块标量 | 无 | ❌ | **block 级索引/任务映射**：每块 2–4 次整数运算（`nchunks < 2^5`，常量除数），非逐元素热点 |
| 10 | `h = p // half`、`k = p % half` | 每 lane 标量 | 除数 `half` 为编译期常量且是 2 的幂（16/32）⇒ 退化为移位/掩码；仍保留在原式里 | ✅（被编译器强度削减，非热点） | —（若 Stage 3 发现 3-D `T.Parallel(bt,hpc,half)` 可用，可完全消去，作为等价改写备选） |
| 11 | host：`view` 展平、空 token 早退、程序缓存 | Python 元数据 | 无 | ❌ | **host 侧元数据**：不在 kernel 内 |
| 12 | GPT-J 相邻对的两列索引 `2k`、`2k+1` | 每 lane 标量 | 保留（同一 `T.Parallel` 内的地址运算） | ⚠️ 见 §1.6.3 #5 | 跨 lane 步长 2 的访问是否仍向量化依赖未实测的编译器行为 → 进 §1.6.3 的实验裁决计划（不是「无理由保留」） |

**向量化结论**：逐元素计算已**全部向量化**（#1–#6 走 `T.Parallel` + `vcast/vmul/vsub` + `T.copy`，替代方案在平台有仿真验证：`PL-950-conversion-layout`（SIMT 逐元素 + 广播 + 特殊值转换）、`PL-950-vector-execution`（SIMT/SIMD 尾块路径）、`PL-950-row-primitives`（SIMT 行内逐元素））；保留 4 处标量/循环点，理由类别分别为：**地址/索引计算**（#7 行号）、**动态边界**（#8 尾块谓词）、**block 级任务映射**（#9 折叠）、**host 元数据**（#11）；#10 由编译期常量强度削减承担，#12 的向量化行为进 §1.6.3 的实验裁决（阈值 T2）。**没有以「实现简单」为由保留的标量循环**。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

> I/O layout 是契约（`[*S, H·D]` 行主序 + `(P, R/2)` 表），**核内向量化轴与暂存布局是设计变量**。本算子有 2 个可并行维（token 行、行内列）与 1 个配对维，需枚举。

**轴质量评分**（对选定布局的候选轴）：

| 评分项 | 说明 | 判定 |
|--------|------|------|
| 整除性 | 轴长是否为向量宽度整数倍 | 列轴 `cols = hpc·D`：`D ∈ {64,65,80,128,256}`，列块起点 `cb·cols` 元素对齐（**不保证 32 B 对齐**，`D=65` 时步长 130 B）→ 元素级对齐成立、段级可能非对齐；配对轴 `half ∈ {16,32}` 是 fp16/bf16 宽度 8、fp32 宽度 4 的整数倍 ✓ |
| 尾 lane 浪费率 | `(ceil(len/vw)·vw − len)/len` | 列轴：0%（`hpc` 取 `H_side` 的因子 ⇒ `nchunks·cols = H_side·D` 精确）；行轴：尾块最多浪费 `bt−1` 行（`gptj-6b-decode-b32` 32 行 / bt=4 → 0%；`tokens` 非 `bt` 倍数时最多 3 行） |
| 累加链形态 | 该轴是否为串行累加维 | 无归约 ⇒ 无累加链；列轴 stride=1（连续），行轴 stride=`H_side·D`（跨行跳变） |
| repack 代价 | 改布局的重排代价 | **0**（选定方案不改 I/O 布局，无核内重排） |
| UB 容量影响 | 布局对驻留缓冲的影响 | 17.0 KB（16-bit）/34.0 KB（fp32），见 §4.5 |

**候选矩阵**（全部枚举，弃选给量化理由）：

| # | 布局方案 | 向量化轴 | repack 路径 | 预估收益/代价 | 是否采纳 |
|---|---------|---------|------------|--------------|---------|
| 1 | I/O 原生行主序；tile = `bt × (hpc·D)`；lane 沿列 + 配对轴 | 行内列（stride 1），配对轴 `k` 与列同向 | 无 | 基线：89.3 MB / 60.7 µs（prefill），17.0 KB UB | **✅ 采纳（主选 P）** |
| 2 | lane 沿 token 轴（每 lane 一个 token，列串行） | token 轴（stride = `H·D·itemsize`，128–1024 B） | 无 | 每 lane 的 8–32 次列访问各成一串行步，向量维度指向不同行 ⇒ 一次「向量」写拆成 `bt` 个不连续段；段数 = 1/`bt` 效率，实测同类证据 `PL-1.14-mte3-strided-ws`（band 化写使 MTE3 时间 3×） | ❌ 明确更差 |
| 3 | 表值扩宽到全 rotary 宽度（`cos_e[i, j] = cos[row, j//2]` 等），lane 沿列直接读扩宽表 | 行内列（stride 1） | 需一次广播/扩宽 pass（`half → R`），UB +`bt·R·itemsize` | 只为让 GPT-J 的「列-per-lane」成立；本设计 GPT-J 用「对-per-lane」，扩宽表会让表流量翻倍（5.2 → 10.4 MB）而不减少任何指令；NeoX 无需求 | ❌ 无收益 |
| 4 | 物化 fp32 tile（`x_f32_ub = cast(x_ub)`，两次 pass） | 行内列 | 无（同布局，多一个 fp32 buffer） | UB 16-bit 路径 +16 KB/块（双缓冲后 +32 KB），多 2 次 pass；而表达式级 `T.cast` 已在寄存器内完成同样转换（`PL-1.17-subfp32-fp32-transit` 的链结构就是「UB 同 dtype → cast → fp32 链」） | ❌ 明确更差（UB 与指令都增加，无收益） |
| 5 | GPT-J 去交织双缓冲（UB 内 `ev/od` 两个半区，配对计算后在 UB 内交织回） | 行内列（每半区连续） | UB 内 2 次重排 pass（去交织 + 交织） | 仅当「跨 lane 步长 2」的访问被标量化时才可能赢；代价是 2 次额外 UB pass + 2 个 buffer（`bt·R·itemsize` each）。当前证据：`traps-compiler.md` C10（跨步 gather 向量指令缺失，截至 2026-08-28 build）**提示但未证明**本形态会被标量化 | ❌ 非主选（**备选 B_gptj**，进下方实验裁决计划） |
| 6 | SIMD 去交织路径（`T.simd.vld2(dist="DINTLV_B16")` 双目标载入） | 行内列（寄存器级 DINTLV） | 无（载入即去交织） | 安装包 `simd.py` 有 `vld2` 与 `vintlv/vdintlv` 的签名与 docstring 示例，但本仓**无任何 `T.simd.*` 的算子实现或仿真验证**（PL-950-simd-reduce 只验证了 vld/vcadd/vcmax/vsts 的归约形态，且不覆盖 16-bit 算术） | ❌ 非主选（**备选 B_simd**，需先补取证；与主选混用两种执行区未获验证） |
| 7 | host 侧 permute 进出（把 x 改成 pair-major 布局） | 任意 | 两次全量 permute | 全量拷贝 = 2× 数据流量（83.9 MB → +167.8 MB）≈ +114 µs，远超本体 60.7 µs；且输出契约是原布局，需再转回 | ❌ repack 代价 > 本体 2.8× |

**布局决策结论**：选定 **#1**：核内布局 = **I/O 原生行主序**（tile = `bt × (hpc·D)`，`x_ub/y_ub` 与 GM 同布局），**向量化轴 = 行内列（stride 1）+ 配对轴 `k`（与列同向）**，token 行作为 `T.Parallel` 的第一维摊到 lane 组；**repack 路径 = 无**。该结论同步落入 §3.3 伪代码（`T.Parallel(bt, cols)` 与 `T.Parallel(bt, hpc·half)`，内层列/配对轴连续）与 §6 循环结构（无内层串行维）。弃选 #2–#7 的量化理由见上表。

##### 实验裁决计划（Stage 4 执行；针对 GPT-J 配对访问是否被标量化）

主选 P 对 GPT-J 的配对访问是「lane `p` 读 `x_ub[i, h·D+2k]` 与 `+1`」，**跨 lane 步长 2**；它是否仍以向量指令执行依赖未实测的编译行为（`traps-compiler.md` C10 只给了「跨步 gather 指令缺失」的旧结论，未覆盖 UB 内 2 元素步长）。

- **(a) 代表 shape（必测）**：`gptj-6b-prefill`（tokens 2048、`D=256`、`R=64`、fp16、4096 块）与 `gptj-6b-decode-b32`（tokens 32、同参、64 块）；另测 `llama3-8b-prefill-4k`（NeoX，纯连续访问）作为对照锚点，**不参与 GPT-J 判定**。
- **(b) 指标与反解**：同一 shape 下跑 P 与备选 B_gptj（§3.3 备选路径，UB 内去交织）各一次 kernel 级 msprof（`scope=kernel`、unit `us`），取 `L_P`、`L_B`；同时读 msprof 的 `aiv_scalar_ratio` / `vec_ratio`（或 `mte3_ratio`）判定标量化。原始数据落 `perf_opt/` 并在 `perf_feedback.md` 记 `raw_evidence_path`。
- **(c) 可机械判定的阈值**：**T1** `L_P ≥ 1.15 × L_B` ⇒ 该 GPT-J shape 切 B_gptj（系数 `1.15 = 1 + 2·CV + 5%`，`CV` 为同 shape 重复测量的变异系数，**设计假设 5% 未实测**；实测 `CV > 5%` 时按 `1 + 2·CV + 0.05` 上修后重判）。**T2** 主选路径的 `scalar_ratio ≥ 0.5`（GPT-J shape）⇒ 直接切 B_gptj（标量化已确认，不必等时延）。**T3（语义门，优先于 T1/T2）** Stage 3 精度失败 ⇒ 不采信性能数据，先按 §9 R3 修 P。三者皆未命中 ⇒ 维持 P。
- **(d) 按 workload 分派**：工厂按 `(neox, dtype, head_size, rotary_dim)` 特化，判定结果可只对 `neox=False` 且 `half % 8 != 0` 之类受影响组合生效——本任务 6 条 workload 中受影响者仅 gptj 两条，**不新增编译产物**（B_gptj 复用同一 prim_func 的备选分支，由模块常量切换）。
- **(e) 回写路径**：Stage 4 结果以 `[DESIGN_LIMIT]` 附录补记写入 `perf_opt/perf_feedback.md`（schema 见 `.agents/skills/_shared/standards/perf-feedback.md`），由 conductor 受控路由；**未回写前一律维持主选 P**，本节与 §11.3 的口径以该补记更新。

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: SIMT（`T.SimtVF(threads=256)` 线程域 + `T.Parallel` 逐元素并行表达式 + fp32 表达式链）

### 2.2 选型理由

本算子是**纯逐元素旋转 + 表行 gather**：无 matmul（排除 Cube）、无归约（排除 `reduce_*`）、无跨 block 依赖（排除跨核旗标/persistent）、每块一个 tile 且无跨 tile 复用（排除 `T.Pipelined`）。四条候选路径的取证情况：

| 路径 | 承载 API | 本算子所需算子的取证情况 | 判定 |
|------|---------|------------------------|------|
| **SIMT** | `T.SimtVF(threads=…)` + `T.Parallel` + 逐元素表达式 + `T.copy` | 平台本任务组合有**成体系的仿真验证**：`PL-950-conversion-layout`（SIMT 内 `T.cast` 的 fp32↔fp16/bf16 转换，含 ±0/±Inf/NaN/次正规/ties，`atol=rtol=0` 对 torch）、`PL-950-row-primitives`（SIMT 行内逐元素 + 广播 + 尾块，FP32/FP16/BF16）、`PL-950-vector-execution`（SIMT 与 SIMD 两条 abs 路径 + UB 尾块填充）；`examples/paged_cache_write/_paged_cache_write_kernel/_paged_cache_write_kernel.py` 在本平台用「`T.Kernel` + `T.SimtVF(threads=128)` + `T.Parallel(16,128)` + 行谓词 + 动态 grid」通过分层精度验证 | **✅ 选定** |
| SIMD | `T.SimdVF()` + `T.simd.vld/vsts/vmul/vsub/vadd/vcvt` | 安装包 `simd.py` 有完整签名与 docstring（含 `vld2(dist="DINTLV_B16")` 去交织双目标载入），`PL-950-simd-reduce` 在仿真器上验证了 `vld/vcadd/vcmax/vsts/update_mask` 的**归约**形态（FP32）；但**本仓无任何算子用 `T.simd.*` 实现逐元素算术**，16-bit 的 v-算术与 vsts 组合、UB 地址算术（`access_ptr`/`make_ubuf_ptr`）均无可用佐证 | 备选（§2.4 与 §1.6.3 #6，仅用于 GPT-J 去交织，需先补取证） |
| Cube | `T.gemm` / L1+L0A/B/C | 本算子无矩阵乘；`R ≈ 64` 的旋转写成 2×2 块对角矩阵乘会引入 Cube 的 L1/L0 搬运与分形约束，属「用小矩阵单元做逐元素工作」的错配 | ❌ 不适用 |
| 混合 | Cube + Vector | 同上，无 Cube 部分 | ❌ 不适用 |

补充依据：平台规则 `.agents/platforms/toolchains/ascend/workflow.md` 明确「向量操作参考上游 elementwise 的 `T.SimdVF`、`T.simd.*` 或 `T.SimtVF`」「根据实际执行路径选择 SIMD、SIMT、Cube 或混合模式」——模式按实际执行路径选，不由默认值决定。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | UB 五块：`x_ub`、`y_ub`（`bt×cols`）、`cs_ub`、`sn_ub`（`bt×half`）、`row_ub`（`bt`，int32）；**不分配** L1/L0A/L0B/L0C（无 Cube） |
| 计算方式 | `T.Parallel` 内的逐元素表达式（`T.cast` / `*` / `-` / `+`），由管线向量化到 `vcast/vmul/vsub`；无显式 `T.simd.*` 手写指令 |
| 线程域 | `T.SimtVF(threads=256)` 在 kernel body 内打开 lane 域（`T.Kernel` 无 `threads` 参数，传了会被 Python 拒绝——见安装包 `Kernel` docstring） |
| 同步 | 全部走自动同步（§7）：单 SimtVF 帧内的 pass 顺序由数据依赖决定，无跨核通信 |
| 向量宽度 | 由列连续性决定（128-bit ⇒ 16-bit 8 元素、fp32 4 元素），不再由 `rot_vec/tail_vec/row_vec` 驱动（§0.6 R5） |

### 2.4 与平台迁移默认的偏离说明

平台 `programming_modes.migration_default = simd`，本设计选 **SIMT**，偏离理由与边界：

1. **默认值的适用面**：`simd` 面向「寄存器级手写向量指令」路径（`T.SimdVF` + `T.simd.*` + `access_ptr` 地址算术）。本算子的访存是「整块连续 tile 的多 pass 读写 + 表行 gather」，需要用 `T.Parallel` 的多维迭代与行谓词表达，而不是逐向量寄存器手写。
2. **取证面（决定性）**：本算子需要的三件事——fp32↔fp16/bf16 的**舍入语义**、逐元素乘加、行谓词下的动态边界——在 SIMT 路径都有本平台仿真验证（§2.2 表）；`T.simd.*` 侧只有归约形态的 FP32 验证，**16-bit v-算术与 vsts 组合未取证**（`PL-950-simd-reduce` 明确「不覆盖其他 dtype」）。
3. **偏离的代价**：SIMT 路径的向量宽度由编译器决定（`T.Parallel` 自动向量化），设计上少一层手写指令控制；性能杠杆相应改为 `units_per_block`（tile 大小）与 `threads` 常量（§6.3）。
4. **保留的 SIMD 备选路径**（仅 GPT-J 的去交织对，Stage 4 若 §1.6.3 T1/T2 命中时启用）：`T.SimdVF()` 内 `vld2(dist="DINTLV_B16")` 载入去交织 → `vmul/vsub/vadd` → `vintlv` → `vsts`；**启用前必须补齐 16-bit v-算术 + vintlv/vsts 的编译与数值验证**（安装包 `simd.py` 只有签名与 docstring 示例，无算子级佐证）。该备选不改数学语义（同算式、同 dtype 转换），只改指令表达。

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式为 §1.6.1 的**优化后公式**；每步使用选定平台可用的向量 API 或线程并行表达式，与 §1.6.2 的向量化结论一致。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 0 | `nrb = ceildiv(tokens, bt)`，`nq = nrb·nchunks_q`，`nblocks = nq + nrb·nchunks_k` | 1-D 块任务映射（§0.6 R1）；`bt`、`nchunks_*` 为编译期常量 |
| 1 | `bx < nq ⇒ (side=query, rb=bx//nchunks_q, cb=bx%nchunks_q)`；否则 `(side=key, rb=(bx-nq)//nchunks_k, cb=(bx-nq)%nchunks_k)` | 三元组折叠（运行时分支，块内一致） |
| 2 | `row0 = rb·bt`；`r_i = min(row0+i, tokens-1)` | 行基址 + 表读取的行钳位（越界行只读，不写） |
| 3 | `row_ub[i] = int32(clamp(int64(POS[r_i]) + int64(OFF[r_i] or 0), 0, P-1))` | A2：clamp 后窄化；每行一次 |
| 4 | `cs_ub[i,k] = COS[row_ub[i], k]`，`sn_ub[i,k] = SIN[row_ub[i], k]`，`0 ≤ k < half` | A1：表行每行一次，块内复用 |
| 5 | `x_ub[i,c] = X[row0+i, cb·cols + c]`（`0 ≤ c < cols`） | tile 载入（行谓词 `row0+i < tokens`） |
| 6 | `y_ub[i,c] = x_ub[i,c]` | A3：整块复制 ⇒ 尾列直通 |
| 7 | `y_ub[i,lo] = dtype(f32(x_ub[i,lo])·f32(cs_ub[i,k]) - f32(x_ub[i,hi])·f32(sn_ub[i,k]))`，`lo = h·D + k`（NeoX）/ `h·D + 2k`（GPT-J），`hi = lo + half` / `lo + 1` | 低半/偶数位（fp32 算术、一次舍入） |
| 8 | `y_ub[i,hi] = dtype(f32(x_ub[i,hi])·f32(cs_ub[i,k]) + f32(x_ub[i,lo])·f32(sn_ub[i,k]))` | 高半/奇数位 |
| 9 | `Y[row0+i, cb·cols + c] = y_ub[i,c]` | tile 写回（行谓词；每行 `cols` 列连续） |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 0 | 1-D 块网格 | `T.Kernel(nblocks)`，`as bx` | `nblocks` 含动态 `tokens`；**不传 `threads`** | SIMT |
| 0' | 线程域 | `T.SimtVF(threads=256)` | `threads=256`、`latency=0`（默认） | SIMT |
| 0'' | 动态轴 | `tokens = T.dynamic("num_tokens")` | 用于 `T.Tensor((tokens, H·D), dt)` extents 与 `nblocks` | — |
| 1 | 折叠索引 | `nq = T.ceildiv(tokens, bt) * nchunks_q`；`with T.If(bx < nq), T.Then(): ...` / `T.Else()` | 运行时块级分支（用 `T.If/T.Then/T.Else` 或 Python `if` 的运行时形态；模板 `kernel_template.py` 用 `if bx < full:`） | SIMT |
| 2–3 | 行号 | `row_ub[i] = T.cast(T.max(0, T.min(T.cast(POS[r], "int64") + ..., P-1)), "int32")`；`T.Parallel(bt)` | `T.cast(value, dtype)`、`T.min/T.max`（PrimExpr） | SIMT |
| 4 | 表行 | `cs_ub[i,k] = COS[row_ub[i], k]`（`T.Parallel(bt, half)`） | 行下标来自 UB 变量（动态 gather 行基址） | SIMT |
| 5 | tile 载入 | `for i, c in T.Parallel(bt, cols): if row0+i < tokens: x_ub[i,c] = X[row0+i, c0+c]` | 2-D `T.Parallel` + lane 谓词 | SIMT |
| 6 | 整块复制 | `for i, c in T.Parallel(bt, cols): y_ub[i,c] = x_ub[i,c]` | （备选 B 用 `T.copy(x_ub, y_ub)`，见 §3.3） | SIMT |
| 7–8 | 旋转 | `for i, p in T.Parallel(bt, hpc*half):` + `h=p//half`、`k=p%half` + 表达式 | `T.cast(x, "float32")`、`*`、`-`、`+`、`T.cast(..., dt)` | SIMT |
| 9 | tile 写回 | `for i, c in T.Parallel(bt, cols): if row0+i < tokens: Y[row0+i, c0+c] = y_ub[i,c]` | 同 #5 | SIMT |
| B1 | 备选：UB 分配 | `T.alloc_shared((bt, cols_max), dt)` | 签名 `alloc_shared(shape, dtype, scope="shared.dyn")`（= UB） | SIMT |
| B2 | 备选：块搬运 | `T.copy(X[b0:b0+bt, c0:c0+cols], x_ub)` / `T.copy(y_ub, Y[b0:b0+bt, c0:c0+cols])` | 区域 copy（两侧元素数相等）；B 路径用 `b0 = min(row0, max(0, tokens-bt))` 的钳位基址避开谓词 | SIMT |
| B3 | 备选：GPT-J 去交织 | `T.SimdVF()` + `T.simd.vld2(..., dist="DINTLV_B16")` / `vintlv` / `vsts` | 见 §2.4 第 4 条与 §1.6.3 #6（需先补取证） | SIMD（仅 B 路径） |

### 3.3 计算伪代码

**主选路径 P（单 SimtVF 帧、八条并行 pass：1 条行号 + 2 条表 + 1 条载入 + 1 条整块复制 + 2 条旋转 + 1 条写回）**：

```python
import tilelang
import tilelang.ascend.language as T

def _rope_positions_kernel(query_heads, key_heads, head_size, rotary_dim, max_position,
                           dtype, position_dtype, offset_dtype, neox,
                           rot_vec, tail_vec, row_vec):
    """NPU factory: same parameter list as the GPU source so the frozen wrapper binds.

    ``rot_vec``/``tail_vec``/``row_vec`` are accepted and unused on NPU (they only chose
    CUDA access widths, §0.6 R5); ``staged`` is accepted and both values compile this
    one program (§0.6 R6); ``units_per_block`` is the per-block pair budget (R4).
    """
    D, R = head_size, rotary_dim
    half = R // 2
    itemsize = 2 if dtype in ("float16", "bfloat16") else 4
    hpc_cap = max(1, 1024 // D)                      # heads per column chunk

    def _hpc(heads: int) -> int:
        return max(d for d in range(1, min(hpc_cap, heads) + 1) if heads % d == 0)

    hpc_q = _hpc(query_heads)
    hpc_k = _hpc(key_heads) if key_heads else 0
    cols_q, cols_k = hpc_q * D, hpc_k * D
    nch_q = query_heads // hpc_q
    nch_k = (key_heads // hpc_k) if key_heads else 0
    cols_max = max(cols_q, cols_k)

    @tilelang.jit(out_idx=[6, 7] if offset_dtype else [5, 6] if key_heads else [4])
    def kernel(units_per_block, staged):
        tokens = T.dynamic("num_tokens")
        # rows per block: the pair budget, capped by the 4096-element tile envelope
        bt = max(1, min(units_per_block // (hpc_q * half), max(1, 4096 // cols_q)))
        nrb = T.ceildiv(tokens, bt)
        nq = nrb * nch_q                                        # query blocks
        nblocks = nq + nrb * nch_k

        def body(X, Y, POS, OFF, H, hpc, cols, nch, rb, cb):
            """One side's tile work; traced once per side (static cols/hpc/nch)."""
            c0 = cb * cols
            row0 = rb * bt
            for i in T.Parallel(bt):                            # step 2-3 (per row)
                r = T.min(row0 + i, tokens - 1)                 # clamp row for the table read
                seen = T.cast(POS[r], "int64")
                if offset_dtype:
                    seen = seen + T.cast(OFF[r], "int64")
                row_ub[i] = T.cast(T.max(0, T.min(seen, max_position - 1)), "int32")
            for i, k in T.Parallel(bt, half):                   # step 4 (A1)
                cs_ub[i, k] = COS[row_ub[i], k]
            for i, k in T.Parallel(bt, half):
                sn_ub[i, k] = SIN[row_ub[i], k]
            for i, c in T.Parallel(bt, cols):                   # step 5
                if row0 + i < tokens:
                    x_ub[i, c] = X[row0 + i, c0 + c]
            for i, c in T.Parallel(bt, cols):                   # step 6 (A3)
                y_ub[i, c] = x_ub[i, c]
            for i, p in T.Parallel(bt, hpc * half):             # step 7: low / even output
                if row0 + i < tokens:
                    h = p // half
                    k = p - h * half
                    lo = h * D + (k if neox else 2 * k)
                    hi = lo + (half if neox else 1)
                    c = T.cast(cs_ub[i, k], "float32")
                    s = T.cast(sn_ub[i, k], "float32")
                    x_lo = T.cast(x_ub[i, lo], "float32")
                    x_hi = T.cast(x_ub[i, hi], "float32")
                    y_ub[i, lo] = T.cast(x_lo * c - x_hi * s, dtype)
            for i, p in T.Parallel(bt, hpc * half):             # step 8: high / odd output
                if row0 + i < tokens:
                    h = p // half
                    k = p - h * half
                    lo = h * D + (k if neox else 2 * k)
                    hi = lo + (half if neox else 1)
                    c = T.cast(cs_ub[i, k], "float32")
                    s = T.cast(sn_ub[i, k], "float32")
                    x_lo = T.cast(x_ub[i, lo], "float32")
                    x_hi = T.cast(x_ub[i, hi], "float32")
                    y_ub[i, hi] = T.cast(x_hi * c + x_lo * s, dtype)
            for i, c in T.Parallel(bt, cols):                   # step 9
                if row0 + i < tokens:
                    Y[row0 + i, c0 + c] = y_ub[i, c]

        if offset_dtype:
            @T.prim_func
            def main(
                POS: T.Tensor((tokens,), position_dtype),
                Q: T.Tensor((tokens, query_heads * D), dtype),
                K: T.Tensor((tokens, key_heads * D), dtype),
                OFF: T.Tensor((tokens,), offset_dtype),
                COS: T.Tensor((max_position, half), dtype),
                SIN: T.Tensor((max_position, half), dtype),
                QO: T.Tensor((tokens, query_heads * D), dtype),
                KO: T.Tensor((tokens, key_heads * D), dtype),
            ) -> None:
                with T.Kernel(nblocks) as bx:
                    x_ub = T.alloc_shared((bt, cols_max), dtype)
                    y_ub = T.alloc_shared((bt, cols_max), dtype)
                    cs_ub = T.alloc_shared((bt, half), dtype)
                    sn_ub = T.alloc_shared((bt, half), dtype)
                    row_ub = T.alloc_shared((bt,), "int32")
                    with T.If(bx < nq), T.Then():
                        with T.SimtVF(threads=256):
                            body(Q, QO, POS, OFF, query_heads, hpc_q, cols_q, nch_q,
                                 bx // nch_q, bx % nch_q)
                    with T.Else():
                        with T.SimtVF(threads=256):
                            t = bx - nq
                            body(K, KO, POS, OFF, key_heads, hpc_k, cols_k, nch_k,
                                 t // nch_k, t % nch_k)
            return main
        # ... the key-only variant (no OFF) and the query-only variant are the same
        # prim_func with the OFF / K / KO parameters dropped (out_idx=[5,6] / [4]).
```

覆盖性论证：`bx ∈ [0, nq + nrb·nch_k)`；`[0,nq)` 经 `(bx//nch_q, bx%nch_q)` 双射到 `[0,nrb) × [0,nch_q)`，`[nq, nblocks)` 经 `(t//nch_k, t%nch_k)` 双射到 `[0,nrb) × [0,nch_k)`；列区间并集 `⋃_cb [cb·cols, (cb+1)·cols) = [0, nch·cols) = [0, H·D)`（因 `nch·cols = H·D`）⇒ 每列恰被一块覆盖，**列方向无尾块**；行方向唯一尾块由 `if row0+i < tokens` 覆盖。

**备选路径 B（块搬运 DMA 形态；主选 P 若在 Stage 3/4 显出搬运指令开销时启用）**：

```python
        # 与 P 同构的网格与折叠；差异只在搬运与暂存：
        #   b0 = T.max(0, T.min(row0, tokens - bt))     # 钳位基址：始终读满 bt 行
        #   T.copy(X[b0:b0+bt, c0:c0+cols], x_ub)       # GM -> UB（一次块搬入）
        #   T.copy(x_ub, y_ub)                          # UB -> UB（尾列直通）
        #   <两条旋转 T.Parallel 循环，写 y_ub>
        #   T.copy(y_ub, Y[b0:b0+bt, c0:c0+cols])       # UB -> GM（一次块搬出）
        # 钳位基址使最后一个行块与前任块重叠 bt − rem 行，重叠行写入的是**同一算式
        # 的同一结果**（同 POS → 同表行 → 同输入），逐字节相同 ⇒ 幂等，语义不变；
        # tokens < bt 时另走 P（P 无此约束）。
```

**每块 pass 数与 `T.copy` 条数**：P 每块 8 条 `T.Parallel` pass（1 条行号、2 条表、1 条载入、1 条复制、2 条旋转、1 条写回），0 条 `T.copy`；B 每块 3 条 `T.copy`（搬入 / UB→UB / 搬出）+ 2 条旋转 pass。

### 3.4 API 可行性确认

> 静态核对位置：`/workspace/venv-ascend950/lib/python3.12/site-packages/tilelang`（`tilelang.__version__ == 0.1.15`，`tilelang.__file__` 已核对；**未**在仓库根运行查询，避免仓库同名包遮蔽——平台规则 `.agents/platforms/toolchains/ascend/workflow.md`「API 证据与执行环境」）。

| API | 证据来源 | 验证状态 |
|-----|---------|---------|
| `T.Kernel(*blocks, prelude=None)`（1-D、无 `threads`） | 安装包 `ascend/language/kernel.py` docstring：「a 1-D grid of AI cores … no SIMT thread domain at this scope … has no `threads` parameter」；`examples/paged_cache_write/_paged_cache_write_kernel/DESIGN.md` §3.4（同平台同工具链） | ✅ 源码级；本算子用法待 Stage 3 真实编译确认 |
| `T.SimtVF(threads=128, latency=0)` | 安装包 `frame.py`：`SimtVF(threads: int \| list[int] \| tuple = 128, latency: int = 0)`；`PL-950-row-primitives`（每行 64 线程 SIMT）、`.agents/platforms/toolchains/ascend/kernel_template.py`（`T.SimtVF(threads=256)`） | ✅ 签名 + 同平台用例；`threads=256` + 4096 元素 tile（16 元素/lane）为已验证 `threads×16` 点的同构使用 |
| `T.Parallel(*extents, ...)`（2-D、4096 元素） | 安装包 `frame.py`：`Parallel(*extents, coalesced_width=None, loop_layout=None, annotations=None)`；`PL-950-row-primitives`（行内逐元素 + 广播）、`PL-950-conversion-layout`（2-D `(3,17)` + 广播） | ✅ 2-D 签名与仿真用例；**4096 元素的 2-D 规模为 4× 外推**（既有用例 48–1024 元素），风险见 §9 R4 |
| `T.dynamic("num_tokens")` | 安装包 `symbolics.py`：`dynamic(name, dtype="int32")` | ✅ |
| `T.cast(value, dtype, round="", sat=True)` | 安装包 `common` 层签名；`PL-950-conversion-layout` 以 `atol=rtol=0` 对照 torch 验证 fp32→fp16/bf16（含 65504/65520 溢出、1.00390625 tie、±0 signbit） | ✅ 存在性 + 舍入语义（同平台仿真） |
| `T.min/T.max/T.ceildiv`（PrimExpr） | 安装包签名：`min(a, b) -> PrimExpr`、`max(a, b)`、`ceildiv(lhs, rhs)` | ✅ |
| `T.alloc_shared((bt, cols), dtype)` = UB | 安装包 `allocate.py`：`alloc_shared(shape, dtype, scope="shared.dyn")`；`PL-950-row-primitives`（64 元素 UB 分配） | ✅ 签名；**17.0–34.0 KB 的多缓冲规模为外推**（既有用例 ≤ 64 元素/行），风险见 §9 R5 |
| `T.Parallel` 内的 lane 谓词（`if row0+i < tokens`） | `examples/paged_cache_write/_paged_cache_write_kernel/_paged_cache_write_kernel.py` L129–172（同平台、`T.SimtVF(threads=128)` 内 `if token < tokens` + `if slot >= 0`，通过分层精度验证） | ✅ 同平台算子级证据 |
| UB 变量作 GM 行下标（`COS[row_ub[i], k]`） | 无直接用例；`T.copy` 的源/目标接受 runtime 索引切片（`TRAP-expert-v-operands` 的 `ws_s[kernel_id, slot, ...]` 先例），但那是 v-op 之外的 copy | ⚠️ 存在性推断；待 Stage 3 真实编译确认（§9 R2），兜底：把行号写回 `T.alloc_local` 标量并在 `T.Parallel(bt)` 内展开为逐行循环 |
| `T.copy(x_ub, y_ub)`（UB→UB 整块） | 平台 `references/programming.md`：`copy_op.py:copy`「整个 Buffer 对整个 Buffer 的拷贝要求元素总数相等，形状不要求逐维相同」 | ✅ 源码级（仅备选 B 使用） |
| `T.copy(GM_region, UB)` / `T.copy(UB, GM_region)` | 同上（局部拷贝用显式 region 表达）；平台 `kernel_template.py` 用 `T.copy(X[start:start+tile], x)` 与 `T.copy(y, Y[start:start+tile])` | ✅（仅备选 B 使用；动态 extent 为未取证项，§9 R8） |
| `T.simd.vld2/vintlv/vsts` | 安装包 `simd.py`：`vld2(addr, dist="DINTLV_B16")` docstring 明确 ``a, b = vld2(x_ub[i, col], dist="DINTLV_B8")``；`PL-950-simd-reduce` 验证 `vld/vcadd/vcmax/vsts` 归约形态（FP32） | ⚠️ 签名 + 归约用例；**逐元素 16-bit 算术未取证**（§2.4 备选，不用于主选） |
| `T.if_then_else` / `T.If/T.Then/T.Else` | 安装包签名；`kernel_template.py` 用 Python 层 `if bx < full:` / `if bx == full:` 表达块级运行时分支 | ✅（块级分支） |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束（来源） | 本算子是否涉及 | 处理方案 |
|-------------|---------------|----------|
| `T.Kernel` 只有 1-D core grid、无 `threads` 参数（`references/programming.md`） | **是** | §0.6 R1 折叠 + `T.SimtVF(threads=256)` |
| 逐元素表达式须在 `T.SimtVF`（SIMT 模式）内 | **是** | §3.3 的全部 `T.Parallel` 均在 SimtVF 帧内 |
| UB 容量 253952 B（248 KB，`design_budget.capacity_bytes.UB`） | **是** | §4.5：最坏 34.0 KB（fp32）/17.0 KB（16-bit）手工预算；按平台膨胀口径（20/26 B/elem）最坏 81.9/106.5 KB，均 ≤ 248 KB |
| L0A/L0B/L0C 仅 Cube 使用 | 否 | 无 `T.gemm` ⇒ L0 全部不适用（§5.1） |
| `ascend.batch_schedule` 跨 block barrier 未认证（`knowledge/capability-gaps.md` CG-950-batch-barrier） | **是（规避）** | 无跨块同步、无 persistent 归约（§5.5、§7） |
| 动态 grid（`T.dynamic`）只支持 1-D extent | **是** | 唯一动态轴 `tokens` 只进 1-D grid 与张量 extents |
| 条件 `T.alloc` 非法、`if` 只有纯名常量折叠（`TRAP-tvm-parser-rules`） | **是** | 五块 UB **在块级分支之前**统一分配（按 `cols_max` 定尺寸）；块级分支只用运行时 `if bx < nq` 包住计算，不包分配 |
| 一个并行体内两次 store 可能使向量化退化（`examples/paged_cache_write/_paged_cache_write_kernel/_paged_cache_write_kernel.py` L101–107） | **是** | §0.6 R8：低半/高半各一条 pass，每体一次 store |
| 二维 `T.Parallel(bm,N)` + 融合 `T.if_then_else` 在 bm≥2 标量化（`traps-compiler.md` TRAP-parallel2d-bm-ge2-scalarize，910B2C/0.1.2） | **是（需规避形态）** | 本设计的条件是**语句级** `if`（`if row0+i < tokens:`），不是 `T.if_then_else` 表达式融合；同平台 `paged_cache_write` 的语句级行谓词已验证。仍登记为 §9 R3，Stage 3 首个配置即核对 `scalar_ratio` |
| 跨步 gather 向量指令缺失（`traps-compiler.md` C10，截至 2026-08-28 build） | **是（GPT-J 配对）** | §1.6.3 #5 的候选矩阵 + 实验裁决计划（T1/T2）；主选按「对-per-lane」写，备选 UB 去交织 / SIMD `vld2` |
| `T.copy` 跨 dtype 静默转换（`traps-compiler.md` C12） | **是（仅备选 B）** | 全部 `T.copy` 两侧同 dtype；不做 cast 融合 |
| 精度门槛（`tests/ops/test_rope.py`：fp32 1e-5、fp16 1e-3、bf16 1.6e-2，atol=rtol） | **是** | fp32 单次舍入（§1.6.1 优化后公式）；G1 项对 fp64 参照 0 容差违反（§1.6.1） |

#### 3.5.2 参考实现差异说明

| 差异项 | 参考实现（GPU `target` = CUDA） | 本项目（Ascend `target="ascend"`） | 转换方案 |
|--------|------------------|-----------------|----------|
| Kernel 结构 | `T.Kernel(num_tokens*parts, threads=128)` | `T.Kernel(nblocks)` 一维、无 `threads` | §0.6 R1 + `T.SimtVF(threads=256)` |
| 语言模块 | `import tilelang.language as T` | `import tilelang.ascend.language as T` | 换命名空间；`alloc_shared` 语义变 UB |
| 并行轴 | CUDA thread/warp + 单元映射 | lane 域 + `T.Parallel` 多维迭代 | §0.6 R2/R4 |
| 暂存 | 默认程序无 SMEM；staged 程序用 SMEM 行缓冲 + `T.sync_threads` + `T.copy` 搬出 | UB 双缓冲 + 八条并行 pass（P，其中 5 条作用于 tile）；DMA 形态为备选 B | §0.6 R3 |
| 向量宽度 | `rot_vec/tail_vec/row_vec`（16 B 对齐，按 head 计算） | 列连续性决定 128-bit；三参数保留但不驱动访存 | §0.6 R5 |
| 程序选择 | wide/config + staged/positions 启发式（SM 数、常驻线程、SMEM 预算） | 单一程序；`staged`/`wide` 编译同一 prim_func | §0.6 R6 |
| 行号 | position dtype 的 int64/int32 | clamp 后 int32（每行一次） | §0.6 R7（A2） |
| 表读取 | 每 block 一次（源已做） | 每行一次 + 块内复用 | §1.6.1 A1（等价替换 + 机器验证） |
| 尾列 | `tail_units` 独立循环 | 整块复制 pass 直通 | §1.6.1 A3 |
| 编译缓存 | `@target_cache(maxsize=32)` | wrapper `_programs` dict | 保留（冻结 wrapper） |

#### 3.5.3 本项目同类实现参考

**结论：本平台 `examples/` 下没有以 `target="ascend"`（`tilelang.ascend.language`）实现的 RoPE 兼容示例**；`examples/posembedding/rope_llama31` 是 RoPE 但使用 `tilelang.language` + `T.Kernel(..., is_npu=True)` + `T.alloc_ub`（另一工具链方言，平台规则明确不得作为本工具链 API 依据）。结构最接近的参考如下，只借鉴任务映射与循环组织，API 一律按 §3.2 重写：

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/paged_cache_write/_paged_cache_write_kernel/DESIGN.md` | 高（同平台同工具链） | 1-D 折叠网格、`T.SimtVF` + `T.Parallel` 行谓词、动态 `tokens`、UB 预算与分核口径、API 证据链 |
| `examples/paged_cache_write/_paged_cache_write_kernel/_paged_cache_write_kernel.py` | 高（已通过精度验证的实现） | `T.Kernel(nblocks) as bx, T.SimtVF(threads=128)`、`T.Parallel` 内语句级行谓词、单 store/体的向量化经验 |
| `.agents/platforms/toolchains/ascend/kernel_template.py` | 高（平台模板） | `@T.prim_func` + `T.Kernel(grid)` + `T.SimtVF(threads=256)` + `T.copy` + 编译期尾块分支 |
| `examples/posembedding/rope_llama31/kernels/table_fp32.py` | 中（数学组织，非 API） | 「表驻留 + 双半旋转 + 分块 token」的组织方式；fp32 无 cast 路径 |
| `examples/elementwise/vec_add_2d_dynamic_shape.py` | 中（结构，非 API） | 动态 shape + 尾块保护的组织方式（NPUIR 方言） |
| `.agents/platforms/toolchains/ascend/knowledge/PL-950-row-primitives.md` | 高（SIMT 行内原语） | SIMT 行内逐元素 + 广播 + 尾块填充的仿真验证口径 |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `POS`（positions） | `(tokens,)`（Op 层 `[*S]`，host `view(-1)`） | `int32` / `int64` | `tokens = T.dynamic("num_tokens")`；越界值由 kernel clamp |
| `Q`（query） | `(tokens, Hq·D)`（Op 层 `[*S, Hq·D]`） | `float16` / `bfloat16` / `float32` | 头已展平；列维编译期常量 |
| `K`（key） | `(tokens, Hk·D)`，可缺省 | 同 `Q` | `Hk == 0` 时由 query-only 程序处理（`K/KO` 参数不出现） |
| `OFF`（offsets） | `(tokens,)`，可缺省 | `int32` / `int64` | 仅 offset 变体出现；与 `POS` 同形 |
| `COS` / `SIN` | `(P, half)` | 同 `Q` | 调用方预计算并按 dtype 舍入 |

补注：Op 层的 `[B,S,H,D]`（BSND）与 `[S,H,D]` 在 host 侧 `view(tokens, H·D)` 后对 kernel 完全等价（`tokens = B·S`），故 kernel 只认 2-D；`Q/K` 与输出行宽相同，元素坐标一致。

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `QO` | 同 `Q` | 同 `Q` | 新张量（`out_idx` 之一）；每元素恰被一个块写一次 |
| `KO` | 同 `K` | 同 `K` | `key is None` 时不出现（`out_idx=[4]`） |
| 返回值 | — | — | `(query_out, key_out)` 由 `out_idx` 交由运行时分配 |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| `x_ub` | `(bt, cols_max)` | 同 `Q` | UB（`T.alloc_shared`，`scope="shared.dyn"`） | tile 载入后只读（旋转两遍都读它） |
| `y_ub` | `(bt, cols_max)` | 同 `Q` | UB | 输出 tile：初值为 `x_ub` 的副本（尾列直通），旋转区被覆写 |
| `cs_ub` | `(bt, half)` | 同 `Q` | UB | cos 行（每行一次，块内复用） |
| `sn_ub` | `(bt, half)` | 同 `Q` | UB | sin 行 |
| `row_ub` | `(bt,)` | `int32` | UB | 每行的表行号（clamp 后窄化） |

无 fp32 中转 buffer（`T.cast` 在寄存器内完成，§1.6.3 #4 弃选理由）；无 L1/L0 buffer。

### 4.4 内存搬运路径

**主选路径 P（单 SimtVF 帧，八条并行 pass，其中 5 条作用于 tile）**：

```
GM[POS]  --每行一次标量读--> row_ub(bt, int32)                     （谓词：无，行基址钳位）
GM[COS]  --每行 half 个--> cs_ub(bt, half)   （块内 H_side 个 head 复用：A1）
GM[SIN]  --每行 half 个--> sn_ub(bt, half)

GM[X 行块 (bt, cols)] --pass 3 (行谓词, 列连续 128-bit)--> UB[x_ub]
UB[x_ub]             --pass 4 (整块复制)----------------> UB[y_ub]        （A3：尾列直通）
UB[cs/sn, x_ub]      --pass 5 (低半/偶数)--------------> UB[y_ub]        （fp32 算术 + 一次舍入）
UB[cs/sn, x_ub]      --pass 6 (高半/奇数)--------------> UB[y_ub]
UB[y_ub]             --pass 7 (行谓词, 列连续)----------> GM[Y 行块 (bt, cols)]
```

- 全链路 **GM → UB → GM**，每元素 GM 读一次、写一次（流量下界）；表与位置为「每行一次」。
- 无 L1/L0 参与；无跨块数据流；块内 pass 之间是同一 UB 的 RAW/WAR 依赖（由自动同步处理，§7）。

**备选路径 B（块搬运 DMA 形态）**：

```
GM[X[b0:b0+bt, c0:c0+cols]] --T.copy--> UB[x_ub] --T.copy--> UB[y_ub]
UB[y_ub] --T.copy--> GM[Y[b0:b0+bt, c0:c0+cols]]        （b0 = clamp 基址，重叠行幂等）
```

两路径的 GM 流量相同（每元素一读一写）；差异只在搬运指令形态（逐 lane 向量 vs 块 DMA）与谓词处理（行谓词 vs 钳位基址）。

### 4.5 UB 内存预算

> 层级 = **UB**（`T.alloc_shared`，`scope="shared.dyn"`）；取**最坏 dtype 组合 = float32**（同 tile 元素数下字节数最大：`llama3-8b-b2-2k`，`bt=4`、`cols=1024`、`half=64`）。L1 / L0A / L0B / L0C 不使用（§5.1）。

**UB**（主选路径 P 的片上缓冲清单，float32 最坏情形）

| Buffer | Shape | dtype | 大小 (Bytes) |
|--------|-------|-------|-------------|
| x_ub | (4, 1024) | float32 | 16384 B |
| y_ub | (4, 1024) | float32 | 16384 B |
| cs_ub | (4, 64) | float32 | 1024 B |
| sn_ub | (4, 64) | float32 | 1024 B |
| row_ub | (4,) | int32 | 16 B |
| 合计 | — | — | 34832 B |

合计 34.0 KB（float32）/ 17.0 KB（16-bit，`bt=4`、`cols=1024`、`half=64` 时 `x_ub+y_ub = 16384 B`、`cs/sn = 512 B` 各、`row_ub = 16 B`）；gptj-6b（16-bit，`half=32`）为 16.5 KB；phi-2（16-bit，`bt=6`、`cols=640`、`half=16`）为 15.4 KB。UB 容量 253952 B（248 KB）⇒ 最坏占用率 13.7%。容量佐证：CANN 9.1.0 `data/platform_config/Ascend950PR_9579.ini` `[AICoreSpec] ub_size`（md5 `2f0eca1f7ca70f62ee7d6d33990c3be1`，读于 2026-11-23），经 `statectl platform --task-dir examples/rope/_rope_positions_kernel` 的 `design_budget.capacity_bytes.UB` 读取；`ub_inflation` 为 `null`（本平台未配置膨胀系数）。

**多缓冲风险核算**：`traps-compiler.md` TRAP-UB-multibuffer-inflation 的 910B2C 平台值给出「16-bit 中转链 20 B/elem、fp32 直连 26 B/elem」的膨胀后 footprint。按本设计的 tile 4096 元素折算：16-bit 81.9 KB、fp32 106.5 KB，均 ≤ 248 KB（余量 ≥ 2.3×）。若 Stage 3 实测膨胀更大，按 `annotate_buffer_versions({x_ub: 1, y_ub: 1})` 固定单版本（`references/programming.md` §同步与缓冲版本），或把 `units_per_block` 减半（tile 2048 元素）。

### 4.6 动态轴定义

| 动态轴 | 声明方式 | 运行时范围 |
|--------|----------|-----------|
| `tokens` | `tokens = T.dynamic("num_tokens")`；进入 `Q/K/QO/KO` 的 extents、`nrb = T.ceildiv(tokens, bt)`、`nq`、`nblocks` | 契约上任意非负整数；`tokens == 0` 由 wrapper 提前返回、不 launch（§0.1）。workload 覆盖 32 / 64 / 1000 / 2048 / 4096（`2×2048` 展平后 4096） |

`P`、`Hq`、`Hk`、`D`、`R`、`half`、`bt`、`cols`、`nchunks`、`threads`、`dtype` 全部编译期常量；无运行期分支依赖 shape 之外的量。

### 4.7 JIT 配置

```python
@tilelang.jit(
    out_idx=[6, 7] if offset_dtype else [5, 6] if key_heads else [4],
    # 无 pass_configs：默认流水线含 AutoSchedule / InsertSync / MaterializeMultiBuffer
)
```

wrapper 侧以 `tilelang.compile(main, target="ascend", out_idx=...)` 构建（与 `kernel_template.py` 一致），并按 `(units_per_block, staged, wide)` 缓存编译产物。**不引入** `pass_configs`、不调用 `annotate_unlimit_memory`（会解除自动调度的容量约束，本算子不需要）。

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**: 纯 Vector（SIMT lane 域内的逐元素乘加 + tile 搬运）。

**判定依据**: 算子只含同 dtype 拷贝、fp32 提升、两次乘、两次加（每次旋转对）与一次舍入回写——无矩阵乘（无 `T.gemm`）、无归约、无跨 block 依赖。因此 **Cube 路径与 L1/L0A/L0B/L0C 全部不适用**，唯一涉及的片上层级是 UB。

### 5.2 Block 划分

下列数值为编译期常量（进 kernel 特化参数），代表规模取 `llama3-8b-prefill-4k`：

```python
M = 4096          # 代表规模：tokens（动态轴的上界实例）
N = 4096          # 代表规模：query 的列数 = Hq·D（32 × 128）
bm = 4            # = bt：每块的行数（token 数）
bn = 1024         # = cols：每块的列数（hpc·D = 8 × 128）
nchunks_q = N // bn          # 4
nchunks_k = 1024 // bn       # 1（key 的列数 Hk·D = 1024）
```

- **块数（query 侧）** `= ceil(M/bm) × nchunks_q = 1024 × 4 = 4096`；**key 侧** `= 1024 × 1 = 1024`；**合计启动块数 5120**（§5.5 的逻辑核数口径见该节）。
- **tile 元素数** `= bm × bn = 4096`，配 `threads=256` ⇒ **16 元素/lane**（与平台已验证的 `n = threads × 16` 点同构）。
- `cols = hpc·D`，`hpc = max{d : d | H_side, d ≤ max(1, 1024//D)}`：`D=128` ⇒ `hpc=8`；`D=256` ⇒ `hpc=4`；`D=80` ⇒ `hpc=8`（`cols=640`）；`D=65` ⇒ `hpc` 为 `Hq` 的、≤15 的最大因子（如 `Hq=8` ⇒ 8，`cols=520`）。**`hpc` 取 `H_side` 的因子保证列方向无尾块**。
- `bt = max(1, min(units_per_block // (hpc·half), max(1, 4096 // cols)))`：**两 dtype 共用 `units_per_block = 2048` 对**，被 4096 元素上限夹住 ⇒ tile 4096 元素（fp32 为 16 KB）。
- 全部 6 条 workload 的实际取值（由 `units_per_block` 与上式算出）：

| workload | `bt` | `hpc_q`/`hpc_k` | `cols_q`/`cols_k` | `nchunks_q`/`nchunks_k` | tile 元素 | 启动块数 |
|----------|------|-----------------|-------------------|--------------------------|-----------|----------|
| llama3-8b-decode-b64 | 4 | 8 / 8 | 1024 / 1024 | 4 / 1 | 4096 | 80 |
| llama3-8b-prefill-4k | 4 | 8 / 8 | 1024 / 1024 | 4 / 1 | 4096 | 5120 |
| llama3-8b-b2-2k（fp32） | 4 | 8 / 8 | 1024 / 1024 | 4 / 1 | 4096 | 5120 |
| gptj-6b-prefill | 4 | 4 / 4 | 1024 / 1024 | 4 / 4 | 4096 | 4096 |
| gptj-6b-decode-b32 | 4 | 4 / 4 | 1024 / 1024 | 4 / 4 | 4096 | 64 |
| phi-2-query-only | 6 | 8 / — | 640 / — | 4 / 0 | 3840 | 668 |

### 5.3 约束分析

- **对齐**：列块起点 `cb·cols` 的元素偏移连续但不保证 32 B 段对齐（`D=65`/`D=80` 时 `cols·itemsize` 非 32 B 倍数）⇒ 元素级对齐成立、段级可能跨段；`half ∈ {16,32}` 是 16-bit 向量宽度 8、fp32 宽度 4 的整数倍 ⇒ 配对轴向量化成立。
- **UB 容量**：最坏 34.0 KB ≪ 248 KB（§4.5，含多缓冲风险核算 ≤106.5 KB）。
- **L0 容量**：不适用（无 `T.gemm`、不分配 L0）。
- **动态 grid**：只有 `tokens` 一个动态轴，且只进 1-D grid 与张量 extents。
- **单块 lane 负载**：`bm·bn / threads = 4096/256 = 16` 元素/lane，落在已验证点；`phi-2` 为 `3840/256 = 15`。
- **块级分支**：五块 UB 在分支前分配（TRAP-tvm-parser-rules「条件 alloc 非法」），分支内只有计算。

### 5.4 注意事项

**列方向（无尾块，编译期可证）**：`hpc` 取 `H_side` 的因子 ⇒ `nchunks·cols = H_side·D` 精确成立，`c = cb·cols + c' < H_side·D` 恒成立，**不需要运行期列判断**。代价：`D` 不是 32 B 倍数时行起点可能跨段（例如 `D=65`、`hpc=8` ⇒ `cols=520`，`cols·2 = 1040 B`，每行起点偏移 1040 B 非 32 B 对齐）——这是形状本身的性质，不是分块引入的浪费。

**行方向（唯一运行期尾块）**：`nrb = ceildiv(tokens, bt)` 覆盖最后不满块；块内 `if row0 + i < tokens` 保护读写（§1.6.2 #8）。表行读取用 `r = min(row0+i, tokens-1)` 钳位 ⇒ 越界行**只读不写**（不会读到 GM 之外）。

**`tokens < bt`**（如 `tokens = 1`、`bt = 4`）：`nrb = 1`，仅 1 个行块、3 行被谓词挡住；表读取的钳位使 `POS[0]` 被读 `bt` 次（无害）。**不采用**备选 B 的钳位基址（`tokens - bt` 为负）——B 路径在 `tokens < bt` 时回退到 P。

**`Hk` 小于 `hpc_cap`**（如 `Hk = 8` 与 `hpc_cap = 8`）：`hpc_k = 8`、`nchunks_k = 1`；若 `Hk < hpc_cap`（如 `Hk = 2`）则 `hpc_k = 2`、`nchunks_k = 1`，仍无列尾块。

**`half` 非向量宽度整数倍**（如未来的 `R = 24` ⇒ `half = 12`）：`T.Parallel(bt, hpc·half)` 仍正确，但向量化可能退化（跨 head 边界的配对不连续）；兜底为按 `h` 展开成 `hpc` 条 `T.Parallel(bt, half)`（§0.6 R2 末）。本任务 6 条 workload 的 `half ∈ {16,32}` 均满足。

### 5.5 分核策略（物理核数适配）⭐

> 三要素按 `.agents/skills/_shared/standards/core-split-strategy.md` §1 组织；核数按任务平台记录实查。

- **② 物理核数依据（实查记录）**：目标型号 **`ascend-950pr-9579`**（`Ascend950PR_9579`，`dav-3510`）。查询命令与实际返回：
  ```
  python3 .agents/tools/statectl.py platform --task-dir examples/rope/_rope_positions_kernel
    → "hardware": {"hardware_id": "ascend-950pr-9579", "architecture": "dav-3510",
                   "observed_cores": {"cube": 28, "vector": 56}}
    → "core_source": "Ascend950PR_9579.ini [SoCInfo] ai_core_cnt=28, cube_core_cnt=28,
                      vector_core_cnt=56; runtime probe rtGetAiCoreCount()=28 on
                      Ascend950PR_9579, 2026-11-23"
  ```
  本算子**无 Cube 计算、全部落在 Vector** ⇒ 分核基数 = **56 个 AIV**（28 个 AIC 不参与）。设备侧 `torch.npu.get_device_properties` 的复核因本机唯一卡被其它任务占用而**未执行**，本轮以上述平台记录（含运行期探针来源）为依据；Stage 3 首次真机运行时须补记该调用返回的 `cube_core_num`/`vector_core_num` 并与本值对账（§9 R10）。
- **① 逻辑核数计算**：按算子实际输出网格（每块 = `bt` 行 × `cols` 列的 tile，覆盖 query 与 key 两侧）：
  - 代表规模（llama3-8b-prefill-4k）：**逻辑核数 = 4096**（query 侧 `= ceil(M/bm) × ceil(N/bn) = ceil(4096/4) × ceil(4096/1024) = 1024 × 4`）；key 侧另有 `1024 × 1 = 1024`；**合计启动块数 = 5120**。
  - 其余 workload：`decode-b64` 80；`b2-2k(fp32)` 5120；`gptj-prefill` 4096；`gptj-decode-b32` 64；`phi-2` 668（见 §5.2 表）。
- **③ 规模判定与分核方案**：
  - **小规模（≤ 物理核数）**：`gptj-6b-decode-b32`（64 块 ≈ 56 × 1.14）、`llama3-8b-decode-b64`（80 块 ≈ 56 × 1.43）⇒ **无需适配**：每块独占一个 AIV，硬件不做分波；绝对时延落在启动地板（§1.6.0 估算 0.7–0.9 µs），调整 block 无意义。
  - **中等规模（物理核数的个位数倍）**：`phi-2-query-only`（668 块 = 56 × 11.9）⇒ 末波利用率 `(668 − 56×11)/56 = 0.2`（11 波满载 + 1 波 12 个 AIV）；可接受，不调 block。
  - **极大规模（≫ 物理核数）**：`llama3-8b-prefill-4k`（5120 = 56 × 91 + 24）、`b2-2k`（同）、`gptj-6b-prefill`（4096 = 56 × 73 + 8）⇒ 走**硬件分波调度**（逻辑块超发，按 56 AIV 逐波排空），**不做 `T.Persistent` 核内串行**。依据：单块工作量仅 4096 元素（§1.6.0 的 ALU 项 11.7 µs ≪ 流量项），核内串行会把 91 个块绑到一个核上并引入每迭代的索引重算，且失去硬件对不规则耗时的自然均衡；本算子无 UB 驻留复用压力（每块独立装载，无跨块复用对象）。
  - **负载均衡量化**：`5120 = 56 × 91 + 24` ⇒ 尾部浪费 `(56 − 24)/5120 = 0.63%`（≈ 0.4 µs @ 60.7 µs）；把 `bt` 提到 8（`units_per_block = 4096` 对）会使块数减半（2560 = 56 × 45 + 40，尾部 `0.63%`）但 tile 翻倍（8192 元素，2× 于已验证规模）——故**维持 `bt = 4`**，把 `bt` 作为 Stage 4 的显式扫参杠杆（§6.3）。
  - **不做跨核协同**：`ascend.batch_schedule` 的跨 block barrier 未认证（`knowledge/capability-gaps.md` CG-950-batch-barrier），本算子也无需跨块同步（每块独立、读写目标不重叠）。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| 逻辑块（`side` × 行块 × 列块） | block 级并行 | `T.Kernel(nblocks) as bx` | 1-D core grid；`bx` 折叠为 `(side, rb, cb)`（§0.6 R1） |
| lane | 线程域声明 | `T.SimtVF(threads=256)` | 打开 256 lane 的向量域 |
| 行号（`bt` 行） | 并行（小规模） | `for i in T.Parallel(bt)` | 每行一次行号计算；`bt ≤ 8`，非热点 |
| 表行（`bt × half`） | 并行（向量化） | `for i, k in T.Parallel(bt, half)` | 内层 `k` 连续 ⇒ 128-bit 向量 |
| tile 载入/复制/写回（`bt × cols`） | 并行（向量化） | `for i, c in T.Parallel(bt, cols)` | 内层 `c` 连续（§1.6.3 选定轴） |
| 旋转低半/高半（`bt × hpc·half`） | 并行（向量化） | `for i, p in T.Parallel(bt, hpc*half)` | 配对轴与列同向 |
| `T.serial` | **不使用** | — | 唯一的顺序语义是 pass 之间的 UB 依赖，由 pass 顺序表达，无需串行循环 |
| `T.Pipelined` | **不使用** | — | 每块一个 tile、无跨 tile 复用，无 stage 可叠（§6.3） |
| `T.Persistent` | **不使用** | — | 逻辑块由硬件分波排空（§5.5） |

### 6.2 循环伪代码

```python
with T.Kernel(nblocks) as bx, T.SimtVF(threads=256):     # 线程域在块级分支内打开
    # side 分支（块级、运行时；两分支的 tile 形状各自静态）
    #   query: rb = bx // nchunks_q,  cb = bx % nchunks_q
    #   key  : t  = bx - nq,        rb = t // nchunks_k, cb = t % nchunks_k
    for i in T.Parallel(bt):                     # 行号（clamp 后窄化）
    for i, k in T.Parallel(bt, half):            # cs_ub（A1）
    for i, k in T.Parallel(bt, half):            # sn_ub
    for i, c in T.Parallel(bt, cols):            # x_ub 载入（行谓词）
    for i, c in T.Parallel(bt, cols):            # y_ub = x_ub（A3）
    for i, p in T.Parallel(bt, hpc * half):      # 低半/偶数（行谓词）
    for i, p in T.Parallel(bt, hpc * half):      # 高半/奇数（行谓词）
    for i, c in T.Parallel(bt, cols):            # y_ub 写回（行谓词）
```

pass 之间的顺序即数据依赖顺序：表 → 载入 → 复制 → 旋转 → 写回；`x_ub` 从载入后只读、`y_ub` 在旋转前只写一次拷贝、旋转只写 `y_ub`、写回只读 `y_ub`（无 WAR 冲突）。

### 6.3 流水线优化

**不使用 `T.Pipelined`**，理由三条：

1. **无 stage 可叠**：流水线的前提是同一块内「搬入 tile → 计算 → 搬出」可重叠；本设计的八条 pass 已把搬运与计算解耦到独立的并行 pass，自动调度（`allow_autoschedule` 默认开启）会在同一 VF 内按依赖排布并尝试重叠，无需显式 stage。
2. **无跨 tile 复用**：每块只处理一个 tile，不存在「下一 tile 预取」的对象。
3. **跨块重叠由硬件分波提供**：56 个 AIV 上不同块天然并行，块间无依赖 ⇒ 搬运与计算在设备级已重叠（`PL-1.6-copy-floor` 的实测结论同向：多 pass 链在大流量档被 MTE2 窗口隐藏）。

**Stage 4 显式性能杠杆**（本设计只登记，不在 Stage 1 采纳，均需真机测量）：

| 杠杆 | 改动 | 预期方向 | 依据 / 风险 |
|------|------|---------|------------|
| `units_per_block` 2048 → 4096（tile 8192，`bt=8`） | 块数减半、每块 32 元素/lane | 块级固定开销摊销一半（若块开销主导） | tile 规模 2× 于已验证点（§9 R4） |
| `threads` 256 → 128 | 每 lane 32 元素 | 减少 lane 域开销、增加每 lane 向量指令数 | 需重测 `threads×16` 之外的点 |
| 备选 B（块搬运 DMA，§3.3） | 三条 `T.copy` 替代 P 的载入/复制/写回三条 pass | 若逐 lane 向量写弱于块 DMA，则胜 | 需动态 extent copy 取证（§9 R8） |
| 备选 B_gptj（UB 去交织，§1.6.3 #5） | 仅 GPT-J 配对路径 | 若跨 lane 步长 2 被标量化，则胜 | 裁决阈值见 §1.6.3 T1/T2 |

### 6.4 尾块处理

| 尾块来源 | 是否可能 | 处理方式 | 代价 |
|---------|---------|---------|------|
| 行方向（`tokens % bt ≠ 0`） | **是**（唯一真实尾块；`tokens` 动态） | `nrb = ceildiv(tokens, bt)` 覆盖最后不满块；块内 `if row0 + i < tokens` 挡住读写；表读取用 `min(row0+i, tokens-1)` 钳位 | 最多浪费 `bt−1 = 3` 行，只影响 1 个行块（5120 块中的 1 块） |
| 列方向（`H_side % hpc ≠ 0`） | **否**（`hpc` 取因子） | `nchunks·cols = H_side·D` 精确成立 | 0（无运行期列判断） |
| `half` 非向量宽整数倍 | 本任务否（`half ∈ {16,32}`） | 按 `h` 展开为 `hpc` 条 `T.Parallel(bt, half)`（§0.6 R2 末） | 少一次跨 head 连续向量 |
| `tokens == 0` | 是 | wrapper 提前返回，**不 launch** | 0 |
| `tokens < bt` | 是（如 `tokens=1`） | 同「行方向」谓词；`nrb = 1` | 3 行空转 |
| `D == R`（无尾列） | 是（llama、gptj 的 head 内无尾列时） | 复制 pass 仍整块执行（无分支） | 一次 4096 元素的 UB 拷贝（等价于「全部列都要写」的必要代价） |

---

## 7. 同步策略

### 7.1 同步模式

**模式**: 自动同步（`ascend/pipeline.py` 的 `allow_autoschedule` 默认开启，`InsertSync` 按缓冲依赖插入事件；`references/programming.md` §同步与缓冲版本）。

### 7.2 同步点说明

| 位置 | 同步 API | 理由 |
|------|----------|------|
| pass 2（`cs/sn` 写）→ pass 5/6（读 `cs/sn`） | 自动（`InsertSync`） | 同一 UB 的 RAW |
| pass 3（`x_ub` 写）→ pass 4/5/6（读 `x_ub`） | 自动 | RAW |
| pass 4（`y_ub` 写）→ pass 5/6（覆写 `y_ub` 旋转区） | 自动 | WAR/WAW（不同列，仍需保序） |
| pass 5/6（`y_ub` 写）→ pass 7（读 `y_ub`） | 自动 | RAW |
| pass 1（`row_ub` 写）→ pass 2/3（读） | 自动 | RAW |
| 块间 | 无（不使用 `ascend_cross_core_set_flag/wait_flag`） | 块之间无数据依赖（读写目标互不重叠）；跨核旗标在本工具链只有接口说明（`references/programming.md`） |
| 若 Stage 3 发现某对依赖未插同步 | `T.sync_threads()`（安装包导出；源 GPU staged 程序亦用 `T.sync_threads()`） | 兜底手段；进入 §9 R2 |

结论：**本设计不手写任何同步点**；`T.Pipelined`/`T.Persistent`/跨核旗标均不使用。

### 7.3 pass_configs 配置

```python
pass_configs = {}   # 默认流水线（AutoSchedule / InsertSync / MaterializeMultiBuffer）
```

若 Stage 3 出现多缓冲膨胀导致 UB 贴限（§4.5），在该路径上显式 `annotate_buffer_versions({x_ub: 1, y_ub: 1, cs_ub: 1, sn_ub: 1})` 固定单版本；不调用 `annotate_unlimit_memory`。

---

## 8. 验证方案

> **CV 融合**：不适用——本算子纯 Vector，无 Cube 段与 cube/vector 协作（§5.1）。**设计期未做设备验证**：本节只给计划，执行在 Stage 3。

### 8.1 Golden 函数

golden 以 **§0.1 的源算子语义**为唯一依据，移植 `examples/TileOPs/tileops/workloads/rope.py` 的 `rope_tables` + `rope_reference`（该文件是独立参考，GPU/NPU 共用），**不复刻** §0.6 的 NPU 算法（不用 1-D 网格折叠、不用 `T.SimtVF`、不做 int32 窄化、不依赖行谓词、不按 `bt/cols` 分块）。

```python
import torch


def rope_tables(params: dict, query: torch.Tensor):
    """调用方持有的 cos/sin 表：fp32 角度 → dtype 参数 → query 的 dtype/device。"""
    rotary, base = params["rotary_dim"], params["base"]
    pos_freqs = base ** (torch.arange(0, rotary, 2, dtype=torch.float32) / rotary)
    inv = 1.0 / pos_freqs
    rows, mscale = params["max_position_embeddings"], 1.0
    angles = torch.outer(torch.arange(rows, dtype=torch.float32), inv)
    return tuple(
        (t * mscale).to(params["dtype"]).to(device=query.device, dtype=query.dtype)
        for t in (angles.cos(), angles.sin())
    )


def _rotate_pairs(part, c, s, neox):
    if neox:                                   # 前后半配对
        x1, x2 = part.chunk(2, dim=-1)
        return torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1)
    x1, x2 = part[..., 0::2], part[..., 1::2]   # 相邻配对（GPT-J）
    return torch.stack((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1).flatten(-2)


def _rotate(x, index, cos, sin, head_size, rotary_dim, neox):
    """x[*S, H·D] → 同形状：每 head 前 rotary_dim 列旋转（fp32 计算，舍入一次）。"""
    tokens = index.numel()
    heads = x.reshape(tokens, -1, head_size)
    c = cos[index.reshape(-1)].unsqueeze(1).float()
    s = sin[index.reshape(-1)].unsqueeze(1).float()
    rotated = _rotate_pairs(heads[..., :rotary_dim].float(), c, s, neox)
    out = torch.cat((rotated.to(x.dtype), heads[..., rotary_dim:]), dim=-1)
    return out.reshape(x.shape)


def golden_rope_positions(positions, query, key, offsets, params):
    """§0.1 语义的独立参考：行号 positions+offsets，越界由调用方先 clamp。"""
    cos, sin = rope_tables(params, query)
    index = positions if offsets is None else positions + offsets
    hs, rd, neox = params["head_size"], params["rotary_dim"], params["is_neox_style"]
    q_out = _rotate(query, index, cos, sin, hs, rd, neox)
    k_out = None if key is None else _rotate(key, index, cos, sin, hs, rd, neox)
    return q_out, k_out
```

关键性质：**clamp 由测试侧表达**（`positions.clamp(0, P-1)`，与 `tests/ops/test_rope.py` 的 `ref_program(positions.clamp(0, 7), ...)` 一致），因此 golden 与 kernel 的 clamp 语义互相独立；fp32 计算后**只舍入一次**（G1 项已对 fp64 参照验证容差内一致，§1.6.1）；`key is None` 时只返回 query。

### 8.2 精度标准

| dtype | atol | rtol | 说明 |
|-------|------|------|------|
| `float16` | 1e-3 | 1e-3 | 与 `tests/ops/test_rope.py::_tolerances` 一致 |
| `bfloat16` | 1.6e-2 | 1.6e-2 | 同上 |
| `float32` | 1e-5 | 1e-5 | 同上；FMA 因在该档位级不等价被否决（§11.1 X2） |

断言用 `torch.testing.assert_close(..., atol, rtol, equal_nan=True)`；G1 证据（§1.6.1）表明 fp32 单次舍入与 fp64 参照在上述容差下 0 违反，故该门槛可由本设计达成。

### 8.3 L0 / L1 / boundary 测试计划

**L0（冒烟门槛；每例都跑 3 种 dtype 的代表项，覆盖 6 条 workload 的关键分支）**：

| # | 用例 | 配置 | 覆盖点 |
|---|------|------|--------|
| L0-1 | `neox-full-head` | `tokens=64, Hq=4, Hk=2, D=64, R=64, neox, fp16` | 基本旋转 + clamp 边界的 `D==R` 路径 |
| L0-2 | `gptj-adjacent-full` | `tokens=64, Hq=4, Hk=2, D=64, R=64, gptj, fp16` | 相邻配对（跨 lane 步长 2）数值正确 |
| L0-3 | `neox-bf16-batched-positions` | `positions=[2,16]`、`query=[2,16,Hq·D]`、无 key、bf16 | `[*S]` 多维位置 + query-only 程序（`out_idx=[4]`）+ bf16 路径 |
| L0-4 | `fp32-partial-rotary` | `tokens=64, Hq=8, Hk=2, D=64, R=24, gptj, fp32` | fp32 无 cast 路径 + `R<D` 尾列直通（A3）+ `half=12` 的兜底形态 |
| L0-5 | `phi2-query-only-int32` | `tokens=1000, Hq=32, D=80, R=32, neox, bf16, position_dtype=int32` | manifest 第 6 条：窄头 + int32 位置 + 无 key |
| L0-6 | `clamp-outside-table` | `P=8`, positions `[-1, 8, 3]`, neox 与 gptj 各一 | 行号 clamp（A2 的运行时形态）；参考 `positions.clamp(0,7)` |
| L0-7 | `zero-head-key` | `Hq=3, Hk=0`，key 存在但 0 列 | wrapper 走 query-only 程序并返回 `empty_like(key)`（kernel 不参与） |
| L0-8 | `empty-tokens` | `tokens=0` | 不 launch、返回空张量（wrapper 早退） |
| L0-9 | `staged-both-values` | 同一调用分别用 `staged=True`/`False` 编译一次 | 两值编译同一程序且结果一致（§0.6 R6 的接口断言） |
| L0-10 | `roofline-accounting` | 任一 workload | `eval_roofline()` 的 `(flops, bytes)` 与 manifest 公式一致（`3·tokens·R·(Hq+Hk)`、`2·(q+k)+pos+cache`） |

**L1（完整功能，manifest 6 workload × 其 dtypes）**：`llama3-8b-decode-b64`(bf16)、`llama3-8b-prefill-4k`(bf16, fp16)、`llama3-8b-b2-2k`(fp32)、`gptj-6b-prefill`(fp16)、`gptj-6b-decode-b32`(fp16)、`phi-2-query-only`(bf16)。

**boundary（本设计新增的针对性用例）**：

| # | 用例 | 目的 |
|---|------|------|
| B-1 | `tokens=17, bt=4` ⇒ 5 个行块、尾块 1 行 | 验证行谓词（唯一运行期尾块） |
| B-2 | `tokens=1`（< `bt`） | 最小规模 + `min(row0+i, tokens-1)` 钳位不越界 |
| B-3 | `D=65, R=32`（奇数 head，neox 与 gptj 各一） | 对应源用例 `test_rope_rotates_the_leading_columns_in_either_convention[odd-head]`：列块起点非 32 B 对齐仍正确 |
| B-4 | `D=80, R=32, Hq=32`（phi-2 形态） | `cols=640`、`bt=6`、tile 3840 的非 2 的幂 tile |
| B-5 | `Hq=3, D=64, R=64` ⇒ `hpc=1`、`nchunks=3` | `hpc` 退化路径（头数 < `hpc_cap` 且为奇数） |
| B-6 | `P=2`（极小表）+ positions `[0,1,1,0]` | 表行 gather 的地址边界 |
| B-7 | 输入含 `±inf/NaN/±0/65504`（fp16/bf16 角点） | 与 golden 逐位/容差对照（`equal_nan=True`）；对应 X1/X3 的否决理由 |
| B-8 | `R=24`（`half=12`，非 8 的倍数） | `half` 非向量宽整数倍的兜底形态（§5.4）不被 6 条 workload 覆盖，需保证正确性 |

**分层执行**：L0 用 `--level L0`（平台 `execution.precision_command`：`<python> <kernel> --level L0`，cwd = 任务目录），L1 用 `--level L1`，boundary 用 `--level boundary`；Stage 3 的 `{op}.py` 按此提供四个入口。

---

## 9. 风险点与注意事项

### 9.1 已知约束

| # | 约束 | 触发条件 | 影响 | 缓解 / 降级 |
|---|------|---------|------|------------|
| R1 | **UB 变量作 GM 行下标**（`COS[row_ub[i], k]`）未在本平台验证 | 表行载入 pass | 若编译拒绝「UB 值作动态下标」，pass 2/3 编译失败 | 兜底：把行号改为 `T.alloc_local([1], "int32")` 并在 `T.Parallel(bt)` 内展开为逐行标量循环（`for i in T.serial(bt)`）——语义不变，仅循环形态变化；登记为待 Stage 3 首个编译验证项 |
| R2 | **跨 pass 的 UB 依赖同步**依赖 `InsertSync` 自动插入 | 全部 pass | 若漏插同步 ⇒ 读到未写完的 UB（数值错、可能非确定） | L0-1/L0-2 即可暴露（结果整体错位）；兜底为显式 `T.sync_threads()`（安装包导出、源 staged 程序亦用） |
| R3 | **2-D `T.Parallel` + 语句级 `if` 是否向量化** | tile 载入/写回/旋转 pass | 若退化为标量 ⇒ 性能坍塌（正确性不受影响） | 同平台 `paged_cache_write` 的语句级行谓词已通过精度验证；Stage 3/4 以 msprof `scalar_ratio` 判定（§1.6.3 T2 的同口径）；兜底：按行展开为 `bt` 条 `T.Parallel(cols)`（每行无谓词，谓词移到串行行循环上） |
| R4 | **4096 元素 tile / 16 元素每 lane 为 4× 外推** | 默认 `units_per_block` | 若超出已验证 envelope ⇒ 编译失败或退化 | 已验证点为 48–1024 元素与 `threads×16`；首个编译即验证；失败时 `units_per_block` 减半（tile 2048） |
| R5 | **UB 多缓冲膨胀**（910B2C 口径 20/26 B/elem） | 自动多缓冲生效时 | 手工 34.0 KB → 膨胀后 106.5 KB（fp32），仍 ≤ 248 KB 但余量降至 2.3× | 必要时 `annotate_buffer_versions({x_ub:1, y_ub:1})`；或 `units_per_block` 减半 |
| R6 | **GPT-J 跨 lane 步长 2 访问可能被标量化**（`traps-compiler.md` C10） | `neox=False` 的两条 workload | 该两条 workload 的向量化效率可能显著低于 NeoX | 进 §1.6.3 的实验裁决计划（T1/T2）；备选 B_gptj（UB 去交织）/ B_simd（`vld2` DINTLV） |
| R7 | **`_STAGED`/`wide` 两入参在 NPU 被忽略** | wrapper 的 `_compile` 调用 | 若审阅者认为接口语义丢失：两值编译同一程序 | 理由与断言见 §0.6 R6、L0-9；`staged=True` 的类不会被 Op 调度（Stage 0 冻结的 `_KERNEL_MAP` 只调度两个 `_STAGED=False` 类） |
| R8 | **动态 extent 的 `T.copy`** 未在本平台验证 | 仅备选 B | B 路径编译失败 | B 用钳位基址 + 静态 `bt` extent（`b0 = min(row0, max(0, tokens-bt))`），避免动态 extent；`tokens < bt` 时回退 P |
| R9 | **常数跨型号外推**（910B2C → 950PR） | §1.6.0 的 D-2 估算 | 绝对时延可能偏差 1.5–2×；不影响候选间 1.32× 的流量淘汰 | 估算行已标注「未实测假设」；Stage 4 用真机 msprof 校准并在 `opt_log.md` 记「估算 vs 实测」偏差 |
| R10 | **物理核数未在本次设备侧复核**（`torch.npu.get_device_properties` 未执行） | §5.5 | 若实际 AIV 数与平台记录不符，分核判定的波数会变 | 平台记录含运行期探针来源（2026-11-23）；Stage 3 首次真机运行补记该调用返回值并对账 |
| R11 | **输出行重叠写**（仅备选 B 的钳位基址） | B 路径最后行块 | 两个块写同一行的同一结果（逐字节相同、幂等） | 语义不变但依赖「同值写」；若审阅要求严格无重叠，B 路径改为 `T.copy` + 行谓词（回到动态 extent 的 R8） |

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| 把 `threads` 传给 `T.Kernel` | 沿用 GPU 源码 `T.Kernel(..., threads=128)` | Python 层直接报错（ascend 方言拒绝） | `T.SimtVF(threads=256)`（§3.2） |
| 在 2-D grid 上做块索引 | `T.Kernel(rows, cols)` | 编译失败（只支持 1-D） | `bx` 折叠（§0.6 R1） |
| 条件分配 UB buffer | 把 `T.alloc_shared` 写进 `if bx < nq` | `TRAP-tvm-parser-rules`：条件 alloc 非法 | 分支前统一分配（`cols_max` 定尺寸） |
| 原位旋转（读写同一 `x_ub`） | 省一个 buffer | 跨 lane RAW ⇒ 数值错 | 双缓冲（`x_ub` 只读、`y_ub` 只写） |
| 一个并行体内两次 store | 低半与高半写在同一循环 | 向量化退化（fp8 案例实测会写 0，16-bit 至少退化） | 两遍 pass（§0.6 R8） |
| 用 int32 直接累加位置与 offset | `POS[..] + OFF[..]` 均为 int32 | 溢出（`2^31-1 + 2^31-1`） | 先 `T.cast(..., "int64")` 求和再 clamp（§0.6 R7） |
| 在 kernel 内重算 cos/sin | 以为「省一次表读」 | X3：96%+ 元素失配（表的 dtype 舍入被绕过） | 只读调用方的表（§1.6.1） |
| 尾列用 `T.copy` 但跨 dtype | 误把 dtype 转换塞进 copy | `TRAP-C12`：隐式跨 dtype 转换与额外 vcast | 全部 copy 两侧同 dtype |

### 9.3 特殊场景处理

| 场景 | 处理 |
|------|------|
| `tokens == 0` | wrapper 早退，不构建/不启动；L0-8 断言 |
| `tokens < bt` | 行谓词 + 表读取钳位（§5.4）；B-2 |
| `key` 为 0 列 | Op 层返回 `empty_like(key)` + query-only 程序（L0-7） |
| positions 越界（负值 / ≥ P） | clamp 到 `[0, P-1]`（A2）；L0-6 |
| positions dtype 含超出 int32 的值 | int64 求和 + clamp 后再窄化（A2 反例项证明「先窄化」会回绕） |
| `R == D`（无尾列） | 复制 pass 照常执行（全部列需写出）；无额外分支 |
| `R < D` | 尾列由复制 pass 直通（A3） |
| `D` 为奇数 / `hpc=1` | 列块退化为单 head 宽度；对齐由元素级保证（B-3、B-5） |
| `half` 非向量宽整数倍 | 按 `h` 展开（§5.4 末、B-8） |
| `base` 很大（如 500000，llama3） | 表由调用方生成，kernel 不感知 `base`（无三角函数） |

---

## 10. 交付清单

### 10.1 目录结构

```
examples/rope/_rope_positions_kernel/
├── DESIGN.md            # 本设计文档（Stage 1）
├── verify_equiv.py      # §1.6.1 的等价性机器验证脚本（torch CPU，可重复执行）
└── RETROSPECTIVE.md     # Stage 1 复盘（每次产出 DESIGN.md 后追加）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | 已完成 | 本文件（§0–§11） |
| `verify_equiv.py` | 已完成 | 40 个采纳项对照 + 被否决项数字；输出 `ADOPTED_VERDICT: ALL_PASS` |
| `RETROSPECTIVE.md` | 已完成 | Stage 1 复盘章节（Phase 8） |
| `examples/TileOPs/tileops/kernels/rope/rope/_ro_p_e_fwd_kernels.py` | 待 Stage 3 重写 | 按 §3.3 的 P 程序实现 NPU 工厂（内层 `kernel(units_per_block, staged)`） |
| `{op}.py`（分层测试入口 + golden） | 待 Stage 3 实现 | 按 §8.1/§8.3；golden 移植 `workloads/rope.py` |
| TileOps 侧 kernel 类（`tileops/kernels/rope/rope/rope.py`） | Stage 0 已冻结 | `default_config` 的 `units_per_block` 取值按 §5.2（两 dtype 均 2048 对）由 Stage 3 落定 |

### 10.3 命名规范

- 算子目录：`_rope_positions_kernel`（对应 GPU 工厂名）
- 设计文档 `DESIGN.md`；复盘 `RETROSPECTIVE.md`；等价脚本 `verify_equiv.py`
- kernel 工厂：`_rope_positions_kernel`（签名与 GPU 一致，见 §3.3）

### 10.4 实现顺序

1. 已完成：设计文档（本文）+ 等价性机器验证
2. Stage 3：golden（§8.1）→ L0（§8.3）→ 主选 P 的 kernel（§3.3）→ 逐层精度 → boundary
3. Stage 4：按 §6.3 的杠杆与 §1.6.3 的裁决计划（T1/T2/T3）决定是否切 B_gptj/B，并按 `[DESIGN_LIMIT]` 回写

---

## 11. 附录：被否决优化的数值证据、备选路径与调研来源

### 11.1 被否决的数学化简（`verify_equiv.py` 的 rejected 项）

**X1 原生 dtype 逐 op 舍入**（`r(r(x_lo·c) − r(x_hi·s))`，每步舍回 fp16/bf16）：fp16 1760/6144 位差（2.87e-01，最大 1602 ulp）、bf16 1584/6144（2.58e-01，最大 14897 ulp）、fp32 0（fp32 下与基线同式）。容差违反数为 0（fp16 的 1e-3 容差吸收得住）。结论：**否决**——① 与契约的「fp32 中转、单次舍入」不是同一算术（1.6.1 的等价性是位级主张），② 平台 `PL-1.17-subfp32-fp32-transit` 记录 bf16 的 v-前缀算术**不支持**（`T.vmul` bf16 编译失败的运行时实证），bf16 路径本就必须先进 fp32，③ 统一路径降低 dtype 分支。

**X2 单次舍入 FMA**（`fma(x_lo, c, −x_hi·s)` 语义，fp64 模拟）：fp16/bf16 位差 0（fp32 中间差 1 ulp 在舍入到 16-bit 后消失）；**fp32 878–1924/6144 位差**（最大 4117 ulp，容差 1e-5 下 0 违反）。结论：**否决**——对 fp32 输出不是位级等价，而算术不是瓶颈（§1.6.0 的 ALU 项 11.7 µs ≪ 流量 60.7 µs），统一不采用可避免 dtype 相关分支。

**X3 表即时重算**（由 `position · base^(-2k/R)` 在 fp64 域重算 cos/sin 再舍入到 dtype）：fp16 15859/16384（9.68e-01）、bf16 15785/16384（9.63e-01）、fp32 15886/16384（9.70e-01），容差违反 15202–15871。结论：**否决**——契约把表作为**已按 dtype 舍入的输入**，重算绕过了这次舍入；且每元素两个超越函数、只省 ≤6% 流量（§1.6.0 R3 #4）。

**A2 反例（guard 证据）**：先窄化后 clamp（`int32(pos) → clamp`）在 `-70..P+70` 全枚举 + int32/int64 极端值域上 5/8341 回绕（`±2^32±1`、`±2^40`），证明「clamp 必须在 int64 域完成后才窄化」不是可选顺序。

### 11.2 备选路径 B（块搬运 DMA）的细节与代价

| 项 | 主选 P（逐 lane 向量 pass） | 备选 B（块搬运 DMA） |
|----|---------------------------|---------------------|
| 网格 / 折叠 | `nblocks`、`(side, rb, cb)` | 同构（同一 `nblocks`/折叠/`bt`/`cols`） |
| GM 流量 | 每元素一读一写 | 相同 |
| 搬运指令 | 8 条并行 pass（tile 相关 5 条：载入、复制、旋转×2、写回） | 3 条 `T.copy`（GM→UB、UB→UB、UB→GM）+ 2 条旋转 pass |
| 行谓词 | 语句级 `if row0+i < tokens` | 钳位基址 `b0 = min(row0, max(0, tokens-bt))`（重叠行幂等写，R11）+ `tokens < bt` 回退 P |
| 证据缺口 | 无（同平台语句级谓词有算子级证据） | 动态 extent / 钳位基址的 `T.copy` 未在本平台验证（R8）；`T.copy` 的区域语义见 `references/programming.md` |
| 何时切 | 默认 | 若 Stage 4 实测逐 lane 向量写显著弱于块 DMA（判据见 §6.3 杠杆表与 `perf_feedback.md` 的 `[DESIGN_LIMIT]` 路由） |

### 11.3 GPT-J 配对的备选路径与证据缺口

- **B_gptj（UB 去交织）**：把 `x_ub` 的旋转区按 pair 拆到 `ev_ub/od_ub`（两条 pass：连续写、跨 lane 步长 2 读），配对计算在两块连续缓冲上进行，再写回 `y_ub` 的交织位置（另一条 pass）。代价：+2 条 UB pass、+`2·bt·hpc·half·itemsize` UB（16-bit 为 512 B/块量级）。收益：仅在「步长 2 访问被标量化」时成立。
- **B_simd（`T.simd.vld2` DINTLV）**：`vld2(addr, dist="DINTLV_B16")` 一次载入 256 个 bf16 → 两个 128 lane 寄存器（偶/奇），配 `vintlv` 回到交织。证据状态：安装包 `simd.py` 有签名与 docstring 示例；`PL-950-simd-reduce` 只验证了 FP32 归约形态的 vld/vsts，**16-bit 逐元素算术与 vintlv 未取证**（「未查到」而非「不存在」）。启用前须补 16-bit `vmul/vsub/vadd/vintlv/vsts` 的编译与数值验证。
- **共同约束**：两者都不改数学语义（同算式、同 dtype 转换），只改访问形态；切换判据见 §1.6.3 实验裁决计划的 T1/T2/T3。

### 11.4 调研来源清单（可追溯）

| 来源 | 用途 |
|------|------|
| `.agents/skills/tilelang-op-design/references/algorithm-research.md` | 调研四问执行标准、深度分级、负向断言纪律 |
| `.agents/skills/tilelang-op-design/references/migration-analysis.md` | 三问解读、四态处置、重设计模式库、语义保持论证 |
| `.agents/skills/_shared/standards/hardware-cost-model.md` | D-2 估算行格式与常数引用纪律 |
| `.agents/skills/_shared/standards/core-split-strategy.md` | §5.5 三要素 |
| `.agents/skills/tilelang-op-optimize/references/pattern-library/constants.md` | `CONST-mte2-degradation`（1.26 TB/s @48 AIV + 4 ns/128 B 段）、`CONST-vector-launch-overhead`（0.5 µs/op）、`CONST-capacity-910B2C`（UB 上限口径） |
| `.agents/skills/tilelang-op-optimize/references/pattern-library/elementwise.md` | PL-1.6（copy-floor / MTE2 退化）、PL-1.10（多输入前置装载）、PL-1.17（sub-fp32 fp32 中转）、PL-1.19（列切片写放大） |
| `.agents/skills/tilelang-op-optimize/references/pattern-library/traps-compiler.md` | C10（跨步 gather）、C12（copy 跨 dtype）、TRAP-UB-multibuffer-inflation、TRAP-parallel2d-bm-ge2-scalarize、TRAP-tvm-parser-rules |
| `.agents/platforms/toolchains/ascend/knowledge/PL-950-vector-execution.md` | 950 向量执行 + 尾块（SIMD/SIMT 两条路径） |
| `.agents/platforms/toolchains/ascend/knowledge/PL-950-row-primitives.md` | SIMT 行内逐元素 / 广播 / 尾块填充 |
| `.agents/platforms/toolchains/ascend/knowledge/numerical-boundaries.md` | PL-950-conversion-layout（SIMT 转换 + 广播，`atol=rtol=0`）、PL-950-simd-reduce（vld/vsts 归约形态） |
| `.agents/platforms/toolchains/ascend/knowledge/capability-gaps.md` | CG-950-batch-barrier（跨 block barrier 未认证）、CG-950-simd-shared-reduce |
| `.agents/platforms/toolchains/ascend/knowledge/bottlenecks.md` | 「无 950 真机瓶颈常数」的取证边界 |
| `.agents/platforms/toolchains/ascend/references/programming.md` | API 规则（1-D `T.Kernel`、`SimtVF`、`SimdVF`、`alloc_shared`、`copy` 的区域/整块语义、同步与缓冲版本） |
| `.agents/platforms/toolchains/ascend/workflow.md` | 执行上下文、API 证据纪律、模式选择 |
| `.agents/platforms/toolchains/ascend/kernel_template.py` | 平台 kernel 骨架与分层测试形态 |
| `examples/paged_cache_write/_paged_cache_write_kernel/DESIGN.md` | 同平台同工具链的 1-D 折叠 / 谓词 / 分核 / UB 预算口径与 API 证据链 |
| `examples/paged_cache_write/_paged_cache_write_kernel/_paged_cache_write_kernel.py` | 同平台已通过精度验证的实现（`T.SimtVF` + 语句级行谓词 + 单 store/体） |
| `examples/TileOPs/tileops/workloads/rope.py` | golden 的独立参考实现（`rope_tables` / `rope_reference`） |
| `examples/TileOPs/tileops/manifest/rope.yaml` | 6 条 workload 与 roofline 口径 |
| `examples/TileOPs/tests/ops/test_rope.py` | 33 例语义边界与容差 |
| `examples/posembedding/rope_llama31/kernels/table_fp32.py` | 同族 RoPE 的数学组织（*另一后端方言，不作 API 依据*） |
| 安装包 `/workspace/venv-ascend950/lib/python3.12/site-packages/tilelang/ascend/language/*.py` | `Kernel`/`SimtVF`/`SimdVF`/`copy`/`dynamic`/`alloc_shared`/`cast`/`min`/`max`/`ceildiv`/`simd.vld2` 的签名与 docstring |
| 本目录 `verify_equiv.py` | §1.6.1 / §11.1 的全部数值结论（torch CPU + fp64 参照） |

> 附注：本阶段未使用互联网检索——本地源（参考表 / pattern-library / 平台知识 / 源算子 / 同平台同族实现）已覆盖该算子族的候选空间（直译、表复用、极坐标、即时重算、FMA、原位、SIMD 去交织、host permute），故无 URL 需登记；`kb_search "rope rotary position embedding"` 在本平台知识库返回空命中（本算子族无既有条目），已如实记录在 §1.6.0 的调研范围中。
