# paged_cache_write 算子设计文档

> **任务平台**：`ascend-950pr-9579`（Ascend950PR_9579，dav-3510）；后端 `ascend`（`import tilelang.ascend.language as T`、`tilelang.compile(..., target="ascend")`）；工具链 tilelang 0.1.15+git0efd65c8、torch 2.13.0+cpu、torch-npu 2.13.0rc1、CANN 9.2.0（编译 SDK）+ CANN 9.1.0（runtime/OPP），见 `.agents/platforms/toolchains/ascend/toolchain.lock.json`（profile `ascend950pr-9579-x86_64-v3`）。
>
> **设计期未做设备验证**：本阶段按任务约束拒绝一切编译、运行、benchmark、msprof 与设备侧诊断命令（也未设置或修改任何设备可见性环境变量）。本文所有 API 结论来自**静态源码核对**（选中工具链本机安装 `tilelang` 的 ascend language 模块，dev root `/home/l00970450/zuo/tilelang-fork/build`，源码 `/home/l00970450/zuo/tilelang-fork/tilelang/ascend/language`）与**外部仓 ascend 测试用例阅读**，凡未经设备实证的条目显式标注「待 Stage 3 用真实编译确认」。
>
> **编程模式**：SIMT（`T.SimtVF(threads=128)` + `T.Parallel`）；平台 `programming_modes.migration_default` 为 `simd`，偏离理由与备选路径见 §2.4。
>
> **迁移任务**：源算子 `PagedKVCacheWriteFwdOp`（GPU TileLang / CUDA target）。§0 为源算子解读与迁移决策，§1–§11 的全部设计基于 §0.6 重设计后的 NPU 算法。
>
> **修订记录 — revision 1**（2026-11-23，Stage 2 检视不通过后的路径 A 修订）：修复 REVIEW 的两个阻塞项。**阻塞项 1**：§1.6.3 的「转向判据」扩写为完整的**实验裁决计划**（代表 shape 清单 / 指标与未知量反解式 / 可机械判定的数值阈值 T1–T3 / Stage 4 回写路径 / 按 `cache_dtype` 分派评估），§11.4 的切换判据改为同一口径并互相引用。**阻塞项 2**：§3.3 的备选路径 B 改为与主选 P **同构的 1-D 折叠网格**（`nblocks = ceildiv(tokens, block_rows)·nchunks`、`rb = bx // nchunks`、`cb = bx % nchunks`、列区间 `[cb·cols, (cb+1)·cols)` 且源/目标区间都带 `c0 = cb·cols` 偏移），删除未使用的 `s_ub`，把 `T.copy` region 形状写成确定形式并给出替代写法；同步重算 §4.3/§4.5 的 UB 预算（12.06 KB → **12.0 KB**）与 §11.4 的每块/总块 `T.copy` 口径。本次修订**未改动** §0 迁移分析、§1.6.0 调研结论、§1.6.1 等价优化（`verify_equiv.py` 结论原样保留）、§2 模式选型与 §5 分核策略的结论；仅做与之必要的数字联动（删除 `s_ub` 后 UB 合计在各处同步为 12.0 KB）。

---

## 0. 源算子解读与迁移分析（迁移类任务必填；非迁移任务删除本章节）

> 本章按「三问框架 + 耦合性判定 + 重设计」组织：先彻底读懂源算子（0.1–0.4），再判定算法与优化手段的硬件耦合性（0.5），最后给出 NPU 算法设计决策（0.6）。本章结论驱动 §1–§11 的所有设计决策。

### 0.1 源算子语义（做什么）

**数学语义（in-place 散射）**：给定新 token 的 key/value 张量 `k, v`（`(tokens, heads_kv, dim)`）与两张 page pool `k_pages, v_pages`（`(num_pages, page_size, heads_kv, dim)`，按 `physical_tokens = num_pages * page_size` 行组织），以及每个 token 的落点行号 `slot_mapping`（`int64 (tokens,)`）：

- 行视图：把 4-D 张量按 `row_width = heads_kv * dim` 展平为 `K[tokens, row_width]`、`KPages[physical_tokens, row_width]`（`v`/`v_pages` 同理）。该展平是纯 view，不搬数据。
- 对 `n ∈ [0, tokens)`：`s = slot_mapping[n]`；若 `s < 0` 则**该 token 不读不写**；否则对 `c ∈ [0, row_width)`：
  - 16-bit cache（`K == KV` dtype）：`KPages[s, c] = K[n, c]`，`VPages[s, c] = V[n, c]`（纯拷贝，无算术）；
  - FP8 cache（`cache_dtype == float8_e4m3fn`）：`KPages[s, c] = cast_fp8(clamp(cast_f32(K[n, c]) / k_scale[0], -448, 448))`，`VPages[s, c] = cast_fp8(clamp(cast_f32(V[n, c]) / v_scale[0], -448, 448))`。
- **规约语义**：无归约，**累加顺序 N/A**——每个输出元素只被唯一一个 `(token, col)` 对触达（调用方保证非负 slot 互不相同），故不存在浮点累加顺序差异。
- **dtype 语义**：输入 `float16` / `bfloat16`；16-bit cache 为同 dtype 直拷；FP8 路径先把输入提升到 `float32` 再除 scale，最后饱和转换成 `float8_e4m3fn`（±448 饱和、NaN 保持 NaN、inf 饱和到 ±448）。
- **边界语义**：`slot == -1` 跳过（不读 `K/V`、不写池）；`slot ∉ [0, physical_tokens)` 属调用方违约（kernel 不做边界检查，源实现亦无）；`tokens == 0` 由 host 提前返回、**不发起 kernel**；未被任何 slot 命名的行**保持原有字节**；NaN/±inf/溢出按饱和 cast 语义处理；`±448` 与 ties（`1.0625 / 1.1875`，`scale=1`）必须精确（`Exact(rtol=0, atol=0)`）。
- **输出**：无返回值（`out_idx=[]`），结果全部体现在被原地修改的 `k_pages` / `v_pages` 上。

**语义保持基线**：§8.1 golden 函数以本节语义为唯一依据实现（移植源仓 `ref_program`），**不得**参考 §0.6 的 NPU 算法。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `k` | `(tokens, heads_kv, dim)`，行视图 `(tokens, row_width)` | `float16` / `bfloat16` | 新 key；`tokens` 为动态轴 |
| `v` | 同 `k` | 同 `k` | 新 value |
| `k_pages` | `(num_pages, page_size, heads_kv, dim)`，行视图 `(physical_tokens, row_width)` | 与输入同 dtype，或 `float8_e4m3fn` | key page pool，**原地写** |
| `v_pages` | 同 `k_pages` | 同 `k_pages` | value page pool，**原地写** |
| `slot_mapping` | `(tokens,)` | `int64` | `slot_mapping[n]` = token `n` 写入的行号，或 `-1` 表示跳过 |
| `k_scale` | `(1,)` | `float32` | key 的除数；**当且仅当** cache 为 fp8 时传入 |
| `v_scale` | `(1,)` | `float32` | value 的除数；同上 |
| 返回值 | — | — | 无（`out_idx=[]`，纯原地副作用） |

补充：`head_dim`/`dim` 与 `heads_kv`、`num_pages`、`page_size` 都是**编译期常量**（进 kernel 特化参数），只有 `tokens` 是运行期动态轴（`T.dynamic("tokens")`）。

### 0.3 实现算法解读（怎么算）

**计算步骤分解**（源码每个计算语句都归入某步骤，无遗漏）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式的部分 |
|------|------|------|------|-------------------|
| 1 | host：`if k.shape[0] == 0: return` | `k.shape[0]` | 提前返回 | `tokens == 0` 不发起 kernel |
| 2 | host：`k.view(tokens, row_width)` 等 4 次 view | 4 个 4-D 张量 | 4 个 2-D 行视图 | 行视图等价（`slot` 索引展平后的行） |
| 3 | grid 建立：`T.Kernel(ceildiv(tokens, block_rows), ceildiv(row_width, block_cols), threads=threads)` | `tokens`, `row_width`（编译期 `block_rows=16, block_cols=128, threads=128`） | block 索引 `(row_tile, col_tile)` | 任务划分（每块 = 16 token × 128 列） |
| 4 | 元素坐标：`token = row_tile*block_rows + i`，`col = col_tile*block_cols + c` | block 索引 + `(i, c)` | `token`, `col` | 元素映射 |
| 5 | 边界 guard：`if token < tokens and col < row_width` | `token`, `col` | 写使能 | 非整除尾块（`tokens` 动态、`row_width` 未必整除 128） |
| 6 | slot 载入：`slot = Slots[token]` | `Slots` | `slot`（int64） | `s = slot_mapping[n]` |
| 7 | 跳过 guard：`if slot >= 0` | `slot` | 写使能 | `s >= 0` 才写 |
| 8a | 16-bit 写：`KPages[slot, col] = K[token, col]`；`VPages[slot, col] = V[token, col]` | 行视图 | page pool 字节 | 16-bit 直拷 |
| 8b | FP8 提升+除法：`T.Cast("float32", K[token, col]) / KScale[0]`（V 同式） | 输入元素 + scale | fp32 中间值 | `k[n,c] / k_scale` |
| 9 | FP8 饱和 cast + 存：`KPages[slot, col] = <fp32 值>`（`KPages` 为 fp8 → 隐式饱和 cast） | fp32 中间值 | fp8 字节 | `cast_fp8(clamp(...))`（源码注释：TileLang 的 FP8 cast 饱和有限溢出与 inf，并保留 NaN） |

**数据流与内存访问模式**（源硬件视角，GPU）：

```
GM[K]   --lane 直读(隐式 L1/L2 缓存)--> register(fp32) --÷scale--> register --sat cast--> GM[KPages]
GM[V]   --lane 直读----------------> register(fp32) --÷scale--> register --sat cast--> GM[VPages]
GM[Slots] --lane 读--> 标量 slot（决定目标行地址，随机行）
```

- **源实现没有 SMEM 分配**（无 `T.alloc_shared` / `alloc_shared` 调用），也没有块内数据复用——这是一次「逐元素、寄存器直通」的 GM→register→GM 散射。因此不存在 tile 搬入/搬出的双缓冲与流水结构。
- 写目标地址由 `slot` 决定，属**随机行 + 行内连续列**：同一 lane 组内列连续（`col` stride = 1），但行与行之间在 GM 上跳跃，写流量无跨行合并机会。

**循环与并行结构**：2-D block 网格（`row_tile × col_tile`）+ 块内 `T.Parallel(block_rows, block_cols)`（= 2048 元素，映射到 128 线程 → 16 元素/线程）+ 每元素一层 `token`/`col`/`slot` guard。`block_cols=128` 的意义是：`row_width ∈ {128, 512, 1024}` 时列维被切成 1/4/8 段，每段 128 列连续，兼顾短行（`row_width=128` 时 1 段）与小 token 数的并行度（`block_rows=16`）。

**host 侧逻辑**：
1. `forward` 在 `tokens == 0` 时直接返回（零 token 不启动 kernel）；
2. `view()` 把 4 张 4-D 张量压成 2-D 行视图（纯元数据，无拷贝）；
3. 按 `(block_rows, block_cols)` 缓存编译产物（`_compiled_fn` / `_compiled_config`）；GPU 侧另有 `@target_cache(maxsize=32)` 装饰器缓存 factory；
4. 量化时把 `k_scale, v_scale` 追加进参数列表（非量化时不传）。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 硬件耦合性初判 |
|---|----------|------|------|-----------------|---------------|
| 1 | 直接 GM 逐元素散射（无 SMEM 中转） | 消除 UB/SMEM 双程搬运与容量预算 | 每 lane 直接 `vld` GM → 寄存器 → `vsts` GM | 无（GM 可被线程级随机寻址；依赖 L1/L2 缓存吸收行跳变） | 硬件弱相关（可移植） |
| 2 | token 级 slot 共用（K/V 共享一次 guard 与一次 slot 载入） | 索引流量减半、分支减半 | 同一 `(token, col)` 的两次 store 复用同一 `slot` 变量 | 编译器 CSE/LICM 把 `Slots[token]` 提到两次 store 之外 | 硬件无关（编译器层） |
| 3 | `slot >= 0` 跳过（稀疏写） | 语义必需：负 slot 不触碰池 | 逐 lane 谓词化写 | 逐线程分支/谓词存储 | 硬件相关（分支语义） |
| 4 | 动态 token extent | 免去「每步 token 数都不一样 → 每步重编译」 | `T.dynamic("tokens")` 生成符号化 grid | kernel 启动支持动态 grid | 硬件无关（框架层） |
| 5 | 列切分 `block_cols=128` | 宽行（512/1024）下保持 128 列连续段，短行下退化为 1 段 | 2-D grid 的列维 | 2-D block grid | 硬件相关（grid 维度） |
| 6 | 固定线程映射 `threads=128` | 2048 元素/块 = 128 线程 × 16 元素/线程，向量寄存器满载 | `T.Parallel` 二维展开 + 线程绑定 | CUDA threadIdx | 硬件相关（线程模型） |
| 7 | FP8 饱和 cast（含 NaN 保持） | 量化写安全：越界→±448、NaN→NaN | 依赖 cast 指令自带饱和与 NaN 传播 | CUDA `cvt` 的 saturate 语义 | 硬件强相关（指令语义） |
| 8 | `block_rows=16` 小块行 tile | 小 token 数（decode 64 token）也有并行度 | 行维切 16 | 无 | 硬件无关 |
| 9 | autotune（8 组配置） | 自动选 tile | TVM autotuner 计时搜索 | GPU 计时 + 搜索框架 | 硬件相关（搜索工具） |

> 识别不出来 ≠ 不存在。未列出的优化会在迁移中被静默丢弃，导致性能回退无法追溯。本算子源码规模小（226 行），上表已覆盖其全部显式优化意图（无 SMEM tiling、无 warp shuffle、无 swizzle、无 online 统计量）。

### 0.5 硬件耦合性分析与 NPU 适配决策

**判定问题**：实现算法和优化手段是硬件强相关吗？能用在 NPU 上吗？

| 条目 | 层级（语义/算法/优化） | 源硬件依赖 | NPU 有等价能力？ | 处置（保留/等价替换/重新设计/舍弃） | NPU 对应方案 / 依据 |
|------|----------------------|-----------|-----------------|-----------------------------------|---------------------|
| 1 计算语义（in-place 散射 + 负 slot 跳过 + 未命名行不变） | 语义 | 无 | — | 保留 | 语义层无条件保留（§0.1 为唯一基线） |
| 2 动态 token extent | 语义/接口 | 无 | 有：`T.dynamic("tokens")` + 动态 1-D grid | 保留 | 外部仓 `test_dynamic_outputs.py`（`ROWS = T.dynamic("rows")`、`T.Kernel(ROWS)`）设备验证 |
| 3 host 侧 view 展平、零 token 早退、按 config 缓存编译产物 | 语义/接口 | 无（纯 host） | 有（torch `view` / Python 早退） | 保留 | 不进 kernel；wrapper 已固化该 dispatch |
| 4 直接 GM 逐元素散射（无 SMEM 中转） | 算法 | 无（两侧都是普通 ND 张量） | 有：`T.SimtVF` 内寄存器 ↔ GM 直读直写 | **保留（承载方式换为 SimtVF）** | 外部仓 `test_dynamic_outputs.py`（`positive[row, i] = source[row, i] + bias` 设备验证 GM 直写）；`test_ascend_fractal_storage_integration.py`（`T.Parallel(rows, cols)` → GM 直写，设备验证） |
| 5 token 级 slot 共用（K/V 共享 guard 与载入） | 优化 | 编译器 CSE/LICM | 部分（不保证 CSE 行为） | 等价替换 | 显式把 `slot` 提升为块内标量（§1.6.1 的 O3「tile 级 slot 载入」），不依赖编译器 |
| 6 `slot >= 0` 跳过（逐 lane 谓词写） | 语义/优化 | 逐线程分支 | 部分：`T.Parallel` 内 lane 级 `if` 有 codegen 证据，但**谓词化 GM 写**未设备验证 | 保留 + 风险登记 | 外部仓 `test_simtvf_vectorized_select.py` 在 `T.SimtVF(threads=32)` 内使用 `if lane < active_threads:` 包裹 UB 存储（编译层证据）；GM 谓词写见 §9 风险 R1，备选路径 B 见 §3.3 |
| 7 fp32 提升 + 除法 | 算法 | 无（数值语义） | 有（`vcvt` / `vdiv`） | 等价替换 | 保留 fp32 除法、**不做倒数乘法**（§1.6.1 的 O1 与 X1 数值证据：`scale=7.0` 时 X1 有 60/65536 失配） |
| 8 cast 内建饱和 + NaN 保持 | 算法 | CUDA `cvt` 饱和语义 | 不确定：`vmaxs/vmins` 的 NaN 传播未文档化 | 重新设计 R2 | §0.6 R2（显式 `T.clamp` + NaN 兜底） |
| 9 2-D block 网格 `(row_tile, col_tile)` | 算法/调度 | CUDA 2-D grid | 无（ascend `T.Kernel` 只有 1-D core grid） | 重新设计 R1 | §0.6 R1（1-D 折叠 `rb = bx // nchunks`、`cb = bx % nchunks`） |
| 10 固定线程映射 `threads=128` | 调度 | CUDA threadIdx | 有：`T.SimtVF(threads=128)` | 等价替换 | 外部仓与平台模板一致（`.agents/platforms/toolchains/ascend/kernel_template.py` 用 `T.SimtVF(threads=256)`） |
| 11 `block_cols=128` 列切分 | 调度 | 2-D grid | 有（折叠进 1-D 网格） | 保留 | §0.6 R1 一并承载 |
| 12 `block_rows=16` 行 tile | 调度 | 无 | 有（1-D 网格的 token 维） | 保留 | §5.2（wrapper `default_config` 已固定 16/128） |
| 13 autotune（8 配置） | 优化 | GPU 计时 + 搜索框架 | 无（本次不引入；内部可调参数仅 2 个） | 舍弃 | wrapper 的 K8 已移除 `autotune_configs/tune`，只走 `init_config` 启发式；舍弃理由：搜索空间仅 2 维且基准配置已覆盖全部 workload 宽度 |
| 14 SMEM tiling / 双缓冲 / swizzle | 优化 | shared memory | 无源使用 | 舍弃（不适用） | 源码中不存在（§0.3），无优化意图可承接 |

> 每项处置依据：语义项引用 §0.1；API 项引用外部仓（`/home/l00970450/zuo/tilelang-fork`）的 ascend 测试用例与选中工具链 language 模块源码；舍弃项均写明理由。**未发现**「GPU warp shuffle / swizzle / 三维 Kernel / TMA」类源手段——本算子本体不含归约与矩阵运算。

### 0.6 NPU 算法重设计

对 §0.5 中判定为「重新设计」的条目逐项给出 NPU 新算法。三项重设计不改变数学语义，只改变映射与指令表达方式。

**重设计项 R1: 2-D block 网格 → 1-D core grid 折叠**

- **源方案**：`T.Kernel(ceildiv(tokens, block_rows), ceildiv(row_width, block_cols), threads=threads)`，GPU 的 `(row_tile, col_tile)` 二维 block 索引直接给出 tile 坐标，硬件负责二维调度。
- **NPU 新算法**：ascend 的 `T.Kernel` 只有 1-D core grid（源码注释：`The NPU launch is a 1-D grid of AI cores with no SIMT thread domain at kernel scope`），因此把二维块索引折叠成一维：
  `nchunks = ceildiv(row_width, block_cols)`（编译期常量）、`nblocks = ceildiv(tokens, block_rows) * nchunks`、`rb = bx // nchunks`、`cb = bx % nchunks`，块内元素坐标 `token = rb * block_rows + i`、`col = cb * block_cols + c` 与源实现逐字一致。
- **语义保持论证**：映射 `bx ↦ (rb, cb)` 是 `[0, nblocks)` 到 `[0, ceil(tokens/R)) × [0, nchunks)` 的双射（行主序展开），块集合与源实现完全相同 ⇒ 覆盖同一元素集合、无重叠、无遗漏；每块内仍是 `T.Parallel(block_rows, block_cols)` 的 2048 元素映射，元素级判据（步骤 4–7）不变。边界语义（`token < tokens`、`col < row_width`、`slot >= 0`）逐条保留，故空维度/非整除/NaN 行为不变。

**重设计项 R2: FP8 饱和与 NaN 语义 → 显式 clamp 前置 + cast**

- **源方案**：依赖 TileLang 的隐式 FP8 cast 自带饱和（有限溢出 → ±448、inf → ±448）并保留 NaN；源码注释明确把该语义当作契约。
- **NPU 新算法**：把饱和**显式化**——先 `d = T.cast(K[token, col], "float32") / ks`，再 `d = T.clamp(d, -448.0, 448.0)`（`T.clamp(dst, min_val, max_val)` 等价于 `T.min(T.max(dst, min), max)`，选中工具链 common 层实现），最后 `T.cast(d, cache_dtype)`。NaN 兜底（若 Stage 3 发现 `vmaxs/vmins` 不传播 NaN）：`d = T.if_then_else(d == d, T.clamp(d, -448.0, 448.0), d)`。
- **语义保持论证**：对有限值 `x`，`cast_fp8(clamp(x, -448, 448))` 与「饱和 cast」逐位相同（clamp 后无溢出可饱和；`±448` 是 e4m3 的最大有限值，ties 仍由 cast 的 round-to-nearest-even 决定，故 `1.0625/1.1875` 行为不变）——该等价性由 `verify_equiv.py` 的 G1 与 X3 项机器对照支撑：**只靠 cast、不做 clamp** 时在角点集上出现 bf16 4/16、fp16 2/16 失配（§11 X3），说明显式 clamp 是**语义必需**而非可选优化；clamp 之后失配为 0。`+inf → 448`、`-inf → -448`、`NaN → NaN`（`T.clamp` 的 NaN 传播为「跟随目标 min/max，无额外检查」，语义实现见 §9 R3 风险与兜底式）。注意：MSRA/e4m3 的 `-448` 与 `+448` 边界在 clamp 后不会引入额外舍入（clamp 只做比较与选择）。

**重设计项 R3: 线程映射与搜索空间 → 固定 `threads=128` + 取消 autotune**

- **源方案**：`threads` 作为 kernel 内部参数（默认 128），外层 8 组 autotune 配置在 GPU 上计时择优。
- **NPU 新算法**：`threads` 固化为 kernel 内部常量 128（wrapper 已从接口移除 `threads`，`_paged_cache_write_kernel(heads_kv, dim, physical_tokens, in_dtype, cache_dtype)` 是冻结签名），块内用 `T.SimtVF(threads=128)` 声明线程域；不引入 autotune，用 `block_rows=16 / block_cols=128` 单一基线（§5.2），把「宽块」作为 Stage 4 显式实验杠杆（§6.3）。
- **语义保持论证**：kernel 内**无跨线程归约**、无线程间共享状态、无 `atomic`——线程数只影响「一个 block 内元素如何分摊到 lane」，不改变任何元素的算式与写目标，故线程数变化对结果**恒等**（逐元素映射的线程无关性）。autotune 只影响性能选择，与数值无关。

> 结论：§0.5 中「重新设计」3 项（R1/R2/R3）已全部给出 NPU 算法与语义保持论证；「等价替换」3 项（slot 提升、fp32 除法、SimtVF 线程域）与「保留」7 项、「舍弃」2 项见上表。无遗留未处置条目。

### 0.7 标杆实现

| 用途 | 路径 | 说明 |
|------|------|------|
| 源算子（GPU TileLang） | `/home/l00970450/advanced-compiler/team/lvpanhao/xiangmu/TileOPs/src/tileops/_kernels/attention/paged_cache_write.py` | §0.1–§0.4 的解读对象；kernel factory 签名 `_paged_cache_write_kernel(heads_kv, dim, physical_tokens, in_dtype, cache_dtype)`（内部 `(block_rows, block_cols, threads)`） |
| golden 基准（独立参考） | `/home/l00970450/advanced-compiler/team/lvpanhao/xiangmu/TileOPs/workloads/attention/paged_cache_write.py` 的 `ref_program` | 逐 token gather + `clamp(-limit, limit)` + `flat[rows] = stored`，`Exact(rtol=0, atol=0)`；§8.1 直接移植 |
| L0 基准用例 | `examples/TileOPs/tests/ops/test_paged_kv_cache_write.py` | 饱和/NaN/±inf/ties、全负 slot、零 token、契约拒绝、roofline 断言 |
| 算子契约与 workload | `/home/l00970450/advanced-compiler/team/lvpanhao/xiangmu/TileOPs/src/tileops/manifest/spec/attention.yaml`（`PagedKVCacheWriteFwdOp`，6 个 workload，`outputs: {}`） | 输入规格与 `Exact` 精度要求的机器可读来源 |
| 冻结接口（Stage 0） | `examples/TileOPs/tileops/kernels/attention/paged_cache_write/paged_cache_write.py` | `default_config = {"block_rows": 16, "block_cols": 128}`、`forward` 的 view/早退/dispatch 逻辑；本文设计须与其接口一致 |

> golden 以 §0.1 语义为唯一依据（移植源仓 `ref_program`），**不复刻** §0.6 的 NPU 算法，保证验证独立性。
---

## 1. 概述

### 1.1 算子名称

`paged_cache_write`（`PagedKVCacheWriteFwdOp` 的 NPU 实现；kernel factory `_paged_cache_write_kernel`）。

### 1.2 功能描述

按 `slot_mapping` 把新 token 的 K/V 行**原地散射**写入 paged KV cache 的指定行；`slot == -1` 的 token 跳过，未被命名的行保持原字节；FP8 cache 在写入时除以 scale 并饱和量化到 e4m3。

### 1.3 数学公式

记 `row_width = heads_kv·dim`，`K[tokens, row_width]` 为 `k` 的行视图（`V`、`KPages`、`VPages` 同理），`s_n = slot_mapping[n]`，`k_s = k_scale[0]`，`v_s = v_scale[0]`，`\mathrm{sat}` 为 e4m3 的饱和舍入：

$$
K_{\text{pages}}[s_n, c] =
\begin{cases}
K[n, c], & \text{16-bit cache}\\[4pt]
\mathrm{sat}\!\left(\mathrm{clamp}\!\left(\dfrac{K[n, c]}{k_s},\,-448,\,448\right)\right), & \text{fp8 cache}
\end{cases}
\qquad \text{for } s_n \ge 0,\; 0 \le c < row_width
$$

$$
V_{\text{pages}}[s_n, c] =
\begin{cases}
V[n, c], & \text{16-bit cache}\\[4pt]
\mathrm{sat}\!\left(\mathrm{clamp}\!\left(\dfrac{V[n, c]}{v_s},\,-448,\,448\right)\right), & \text{fp8 cache}
\end{cases}
\qquad \text{for } s_n \ge 0,\; 0 \le c < row_width
$$

`s_n = -1` 时两式均不作用（该 token 的整行不读不写）；`c \ge row_width` 不存在。fp32 提升是除法前的必要精度步骤（`K[n,c]` 为 fp16/bf16）。

### 1.4 算法描述

本节描述**迁移决策后的 NPU 算法**（§0.5/§0.6 落地形态），与源算法的差异逐条标注来源：

1. **任务映射（差异 ← §0.6 R1）**：1-D core grid，`nblocks = ceil(tokens / block_rows) · nchunks`，块内 `rb = bx // nchunks`、`cb = bx % nchunks`；每块负责 `block_rows × block_cols` 个 (token, col) 元素对。
2. **线程域（差异 ← §0.6 R3）**：`T.SimtVF(threads=128)` 打开线程域，`T.Parallel(block_rows, block_cols)` 把 2048 个元素映射到 128 线程（16 元素/线程），逐元素计算全部在向量寄存器内完成。
3. **直写（保留 ← §0.5 条目 4）**：不做 UB 中转——每 lane 直接读 `K/V` 的 GM 元素、在寄存器内算完、直接写 `KPages/VPages` 的 GM 元素（与源实现同构，只是从 CUDA lane 换成 SimtVF lane）。
4. **行级 guard 与跳过（保留 + 提升 ← §0.5 条目 5/6）**：每行读一次 `slot = Slots[token]`（int64 → int32 窄化，§1.6.1 O3），`slot >= 0` 才写；`token < tokens` 保留为 lane 谓词（`tokens` 运行期动态）。
5. **FP8 饱和显式化（差异 ← §0.6 R2）**：`cast_f32 → ÷scale → clamp(-448, 448) → cast_fp8`，clamp 是**语义必需**（非性能优化，见 §1.6.1 的 X3 证据）。
6. **16-bit 直拷（差异 ← §1.6.1 O2）**：另一条 prim_func 分支，无任何算术。
7. **宽度自适应（保留 ← §0.5 条目 11）**：`cols = block_cols if (row_width % block_cols == 0 and block_cols <= row_width) else row_width`（`row_width ∈ {128, 512, 1024}` 时等价于 `min(block_cols, row_width)`），使 `col` 边界在编译期可证，无需运行期判断。

**与源算法的一致性核对**：元素级算式（步骤 3–5）与源实现逐字相同；被替换的只有任务映射（1-D 折叠）、线程域声明（SimtVF）与饱和表达方式（显式 clamp）。无归约、无跨块依赖，故不存在累加顺序差异。

### 1.5 数据流图

```
GM[K(tokens,row_width)]  ─┐
GM[V(tokens,row_width)]  ─┤ (SimtVF lane：vld，GM→寄存器；原地无 UB)
GM[Slots(tokens,)]       ─┼─► [块内：块索引折叠 → token/col → token<tokens → slot=Slots[token] → slot>=0]
GM[KScale(1,)]           ─┤                      │
GM[VScale(1,)]           ─┘                      ▼
                              16-bit 分支：直拷 ─────────────────────► GM[KPages/VPages(slot,col)]
                              fp8  分支：vcvt(f32) → vdiv(ks,vs) → vmaxs(-448) → vmins(448) → vcvt(fp8) ─► 同上
```

- 全链路**无片上暂存**（主选路径 P）：GM → 向量寄存器 → GM。
- 无公共同步点：读与写互不依赖（并发前提是「非负 slot 互不相同」，由调用方保证），同一 block 内不出现「先写后读」或「先读后写」同一地址的竞争。

### 1.6 算法调研与优化分析 ⭐

> **设计第一优先级**：先调研（1.6.0，四问：等价化简公式 / 在线算法 / 复杂度 / 硬件亲和，输入公式与源算法只是基线候选之一），再在数学等价前提下优化公式（1.6.1），再把保留下来的计算交给向量单元（1.6.2），最后回答「在哪个轴上向量化」（1.6.3）。本节结论是 §3.1 公式拆解与 §6 循环结构的输入；1.6.1–1.6.3 的分析对象是 §1.6.0 选定、经 §0.6 重设计落地的 NPU 算法。

#### 1.6.0 算法调研（Algorithm Research）⭐

> 本算子是单遍逐元素散射类（无归约、无窗口、无矩阵运算），按 `algorithm-research.md` 的轻量调研档执行：四问各给一条明确结论 + 依据，但四问不跳过。调研对象是「同一数学语义（索引散射写 + 可选量化）的算法族」，源 GPU 直译只是基线候选 #0。

**R1 等价化简公式候选**（基线候选在表内；正式等价论证与收益量化在 §1.6.1 完成，此处只做初判）：

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 | 是否纳入 R3 对比 |
|---|------|------------|-----------|---------|----------------|
| 0 | 基线：源算法直译（2-D 网格 + 逐元素 slot/scale 读） | `KPages[Slots[token], col] = cast(cast_f32(K[token,col]) / KScale[0])`，每元素重复读 `Slots` 与 `KScale` | — | — | ✅（基线） |
| 1 | 直写 + 行级索引提升（P，选定） | 同上，但 `ks = KScale[0]` 块内标量、`slot` 每行读一次 int32；1-D 网格折叠 | 逐位等价（标量只读一次、窄化有 guard） | 索引/标量访存 `N·W·8 B → 块·R·8 B`；无 UB | ✅ |
| 2 | UB 中转行级 DMA（B，备选） | `T.copy(GM→UB)` → UB 内算 → `T.copy(UB→GM)` 每行一次 | 逐位等价（同算式，仅多一次往返） | 换取行级连续的 DMA 段（若 GM 直写指令效率差） | ✅ |
| 3 | SIMD 寄存器路径（`T.simd.vcvt/vdiv/vmaxs/vmins`） | 用显式向量指令表达同一算式 | 数值等价（同算子） | 与 #1 同流量，多一层 UB | ✅（证据不足，见 R4） |
| 4 | host 侧 `index_copy_` / 高级索引 | 把散射交给 torch host 侧索引 kernel | 数值等价 | —— | ❌（非 kernel 方案：需额外 full-pool 读写与回写，且 `Exact` 校验的原位语义要求 host 侧不得额外拷贝；结构性代价 ≫ 本体流量） |
| 5 | 数学化简（除法→乘倒数、16-bit 域缩放、只靠 cast 饱和） | `x·(1/s)`；在 bf16 域先缩放；省略 clamp | **不等价**（§11 X1/X2/X3 机器证据：失配非零） | 少算 1–2 op/元素 | ❌（不满足 `Exact(0,0)`） |

**R2 在线算法**：**无在线变体**（结论「无」）。结构依据：本算子的输出元素 `(s_n, c)` 由单一输入元素 `(n, c)`（外加标量 scale 与行索引）唯一决定，**不存在跨 token / 跨列的中间统计量**（无 max/sum 归约），因此「分块增量更新 running 统计量」这类在线算法在数学上没有可压缩的对象——每个元素天然只读一次、只写一次，**单遍即最优**。对照：GLU/softmax/RMSNorm 的在线算法存在是因为其输出依赖整行归约；本算子的行内元素彼此独立（`c` 之间无耦合）。故 R3 中所有候选的扫描遍数下界都是 1，在线化无收益空间。

**R3 复杂度对比**（四口径；代表 workload = llama-8b-prefill：`tokens=8192`、`row_width=1024`、bf16→bf16，共 4096 块）：

| 算法候选 | FLOPs | 访存量 (Bytes) | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|-------|---------------|---------|-------------|---------------------|
| #0 基线（逐元素索引读） | 0（16-bit 分支无算术；fp8 分支 5 op/元素/张量） | 数据 67.11 MB + `Slots` 逐元素读 8192·1024·8 = 67.11 MB（fp8 再加 scale 67.11 MB）≈ **134.2 MB**（fp8 下 201.3 MB） | 1 | 0（无片上暂存） | 4096 块 / 56 AIV = 73.1 波 |
| #1 直写 + 行级提升（P，选定） | 同上 | 数据 67.11 MB + `Slots` 每块每行一次 4096·16·8 = 0.52 MB ≈ **67.63 MB** | 1 | 0（无片上暂存） | 同上 |
| #2 UB 中转行级 DMA（B） | 同上 | GM 侧数据同 67.11 MB（UB 往返不计 GM 流量）+ `Slots` 0.52 MB；**多 2R = 32 条/块的 store DMA 指令** | 1（块内 1 读 + 1 写两跳） | **12.0 KB/块**（k/v 的 in + fp8 输出；行级 slot 为寄存器标量，不占 UB） | 同上 + 每块 64 条 `T.copy`（store 侧 32 条） |
| #3 SIMD 寄存器路径 | 同上 | 与 #2 同（SIMD 路径只能操作 UB 区域 → 必须 UB 中转） | 1 | 同 #2 | 同上 |
| #4 host 侧索引 | 同上（torch kernel 内） | 额外需一次全池 gather/scatter 回写（≥ 134 MB 量级） | ≥ 2 | 额外 full-pool 临时 | 依赖 torch 侧 launch，不可控 |

fp8 代表 workload（qwen3-235b-fp8：`tokens=4096`、`row_width=512`、bf16→fp8，1024 块）的同口径对照：

| 算法候选 | FLOPs | 访存量 (Bytes) | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核代价 |
|---------|-------|---------------|---------|-------------|---------------------|
| #0 基线 | 4096·512·2 元素·5 op = 20.97 M | 数据 12.58 MB + `Slots` 4096·512·8 = 16.78 MB + scale 16.78 MB ≈ **46.14 MB** | 1 | 0 | 1024 块 / 56 = 18.3 波 |
| #1 直写 + 行级提升（P） | 20.97 M | 数据 12.58 MB + `Slots` 1024·16·8 = 0.13 MB ≈ **12.71 MB** | 1 | 0 | 同上 |
| #2 / #3 | 20.97 M | 同 #1 + 32 条 store/块 | 1 | 12.0 KB/块 | 同上 |

**R4 硬件亲和性评估**（逐候选对照清单 + 设计期 roofline 估算；估算口径与常数见本节末）：

| 算法候选 | 计算单元匹配 | 片上容量 | 对齐 / 整除 | 静态边界 | 流水 / 融合 | 结论 |
|---------|-------------|---------|------------|---------|------------|------|
| #0 基线 | ✅ 纯 Vector（无 Cube 需求） | ✅ 无片上缓冲 | ⚠️ 逐元素 8 B 索引读在 128 B 段内浪费（1 个 int64 索引换一整段） | ✅ 同 #1 | ❌ 无法融合（索引读与数据写互相拖累） | ❌ 淘汰：流量 134.2 MB（prefill）为 #1 的 2.0 倍，估算下界 91.3 µs vs 46.0 µs（引用 CONST-mte2-degradation） |
| #1 直写 + 行级提升（P） | ✅ 纯 Vector：`vcvt/vdiv/vmaxs/vmins` 全部有 ascend 用例 | ✅ 0 B UB（不占容量） | ✅ `row_width ∈ {128,512,1024}` 是 128 的整数倍；`cols` 归一化后无尾 lane | ✅ `col` 边界编译期可证；`tokens` 动态但由 grid 上界 + lane 谓词覆盖 | ⚠️ 单 tile/块，无 `T.Pipelined` 空间（见 §6.3） | **✅ 选定**：流量最低（67.63 MB），估算下界 46.0 µs |
| #2 UB 中转行级 DMA（B） | ✅ 纯 Vector（另用 GM↔UB `T.copy`） | ✅ 12.0 KB ≪ 248 KB | ✅ 行级连续段（`row_width·b_in ≥ 256 B` 时一次成段） | ✅ 行级 guard | ⚠️ 块内 5 跳（load→算→store），无跨块流水 | ❌ 备选（非主选）：同流量但多 2R 条 store 指令/块；且 UB→GM 逐元素/行级写代价缺 950PR 实测常数，无法证明优于 #1（该待定项由 §1.6.3 的实验裁决计划 T1–T3 在 Stage 4 收敛） |
| #3 SIMD 寄存器路径 | ⚠️ `T.simd.*` 的 fp8 目标 dtype `vcvt/vsts` 用例在外部仓 SIMD 测试集中**未取证**（证据缺口，非「不存在」） | ✅ 同 #2 | ✅ 同 #2 | ✅ | ⚠️ 同 #2 | ❌ 不采纳为主选：证据缺口 + 必须 UB 中转；作为 fp8 链的 Stage 4 备选保留（§2.4） |
| #4 host 侧索引 | ❌ 非 kernel 内计算，跨 launch | ❌ 额外全池临时 | ✅ | ✅ | ❌ | ❌ 淘汰：额外 ≥ 134 MB 级流量 + 额外 launch，且与原位语义冲突 |
| #5 数学化简（倒数乘法/16-bit 缩放/无 clamp） | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ 淘汰：**数值不等价**（`Exact(0,0)` 下 X1/X2/X3 均有失配，§11） |

**设计期估算（D-2）**：代表 workload = llama-8b-prefill（16-bit，4096 块，73.14 波）。

| workload | 块数 | 流量 (Bytes) | 流量下界 (µs) | 发射下界 (µs) | 估算下界 (µs) |
|----------|------|-------------|--------------|--------------|--------------|
| llama-8b-decode（64 tok, 1024 宽） | 32 | 532,480 | 0.36 | 0.0 | 0.4（含启动地板） |
| **llama-8b-prefill（8192 tok, 1024 宽）** | **4096** | **67,633,152** | **46.0** | **0.0** | **46.0** |
| qwen3-235b-fp8（4096 tok, 512 宽） | 1024 | 12,713,984 | 8.65 | 54.9 | 54.9（**发射主导**） |
| gpt-oss-120b（256 tok, 512 宽） | 64 | 1,056,768 | 0.72 | 0.0 | 0.72 |
| odd-tokens（7 tok, 128 宽） | 1 | 7,224 | 0.005 | 0.0 | 0.01（启动地板主导） |
| qwen3-235b-fp8-fp16（1024 tok, 512 宽） | 256 | 3,178,496 | 2.16 | 13.7 | 13.7（发射主导） |

**估算下界: 46.0 us（流量 46.0 + 发射 0.0 + 容量 0.0，引用 CONST-mte2-degradation / CONST-vector-launch-overhead）**

估算口径与常数使用（先查 `constants.md`，库内已有的实测数字直接引用，不外推冒充）：

- **流量项**：`流量 ÷ 带宽`。`CONST-mte2-degradation`（910B2C，tilelang 0.1.2+15ad002b3d / CANN 8.5.0 / 2026-09-21，48 vector core）给出混合读写流量地板 ≈ 1.26 TB/s；950PR 有 56 AIV，按核数线性外推 1.26 × 56/48 = **1.47 TB/s**——**这是未实测假设**（外推未经 950PR 实测，且该常数只覆盖 MTE2 读侧，不覆盖 UB→GM 逐元素写），已在 §9 风险 R9 登记。67,633,152 B ÷ 1.47 TB/s = 46.0 µs。
- **发射项**：`CONST-vector-launch-overhead`（910B2C，tilelang 0.1.2+3a214cde + CANN 8.5.0 / 2026-09-09）= 0.5 µs/op；16-bit 路径**无向量 ALU op**（纯 GM↔寄存器搬移），故发射项 0；fp8 路径 6 op/块（每张量 `vdiv + vmaxs + vmins`，`vcvt` 视为与 `vld/vsts` 折叠；若 Stage 3 发现 cast 不能折叠则 10 op/块，发射项 ×1.67，见 §9 R10）→ 1024 块 · 3.0 µs ÷ 56 = 54.9 µs。
- **容量项**：主选路径无 UB buffer，0；块内无归约缓冲。
- **常数版本绑定**：以上两条常数均为 **910B2C + tilelang 0.1.2 时代测量**，本任务工具链为 950PR + tilelang 0.1.15 + CANN 9.2.0（编译 SDK）/ 9.1.0（runtime），属「跨型号引用」——结论按「量级判断」使用（用于淘汰 2 倍流量差的候选），不作绝对时延承诺。

**调研结论**：选定算法族 **#1「直写 + 行级索引提升」（P：1-D 折叠网格 + `T.SimtVF(threads=128)` + `T.Parallel` 内直接 GM 读、寄存器算、直接 GM 写）**，关键依据三条：(a) R3 显示它是唯一把索引/标量访存压到 0.5 MB 级（vs 基线 67–134 MB）且**不需要任何片上缓冲**的候选；(b) R4 显示其计算全落在 `vcvt/vdiv/vmaxs/vmins` 四个**已有 ascend 用例**的算子上，无 Cube、无归约、无流水需求；(c) 与源算法结构同构（直写散射），语义漂移最小，唯一未取证环节是「lane 谓词化 GM 写」（§9 R1，可按 §3.3 的 B 方案降级）。调研范围：`algorithm-research.md` 的 elementwise/scatter 族参考表；`constants.md` 的 `CONST-capacity-910B2C / CONST-vector-launch-overhead / CONST-mte2-degradation / CONST-copy-floor-method` 四条；`examples/` 全库（无 `target="ascend"` 示例，见 §3.5.3）；外部仓 ascend 测试集（`test_tilelang_ascend_clamp.py`、`test_tilelang_ascend_simtvf_cast.py`、`test_simtvf_arithmetic.py`、`test_simtvf_vectorized_select.py`、`test_dynamic_outputs.py`、`test_tilelang_ascend_dma_copy.py`）；结构分析：无归约 ⇒ 无在线空间、单遍即最优。**源算法优化意图承接**：§0.4 的 9 项中，直写（#1 保留）、slot 共用（#5 → O3 显式提升）、动态 extent（#2 保留）、列切分/行 tile（#5/#11 保留）、线程映射（#10 → SimtVF）、饱和 cast（#8 → R2 显式化）全部承接；autotune（#12）与 SMEM（#14，源未使用）舍弃并已给理由。

#### 1.6.1 数学等价优化（公式级）

> 分析对象为 §1.6.0 选定的 #1 直写算法。三项采纳项全部经 `verify_equiv.py`（torch CPU + fp64 参照，本目录 `verify_equiv.py`）机器验证，结果为 `EQUIV_PASS`。

| # | 优化项 | 原式 | 优化后公式 | 等价性论证 | 收益估算 |
|---|--------|------|-----------|-----------|---------|
| O1 | scale 载入提升（保留 fp32 除法） | 每元素 `cast_f32(K[n,c]) / KScale[0]`（每元素一次标量读） | 块内标量 `ks = KScale[0]`，元素式 `cast_f32(K[n,c]) / ks` | 同一不可变标量读 1 次 vs 读 `R·C` 次，值恒等（无并发写者）⇒ 逐位相同；**除法不换成倒数乘法**（X1 数值不等价） | prefill：标量/索引读 134.2 MB → 0.52 MB（-99.6%）；fp8 workload 发射项从 8 op 降到 6 op |
| O2 | 16-bit 恒等路径（分派消去算术） | fp8 分支算式在 16-bit 下需 `cast→除→clamp→cast`（若误用） | 16-bit 分支 `KPages[s,c] = K[n,c]`，零算术 | 同 dtype 直拷，`float16→float16` / `bfloat16→bfloat16` 无舍入、无溢出（全部 bit pattern 恒等） | 16-bit 路径 0 个向量 ALU op/块（省 6 op = 3.0 µs/块发射） |
| O3 | slot 窄化 + tile 级载入 + 窄化 guard | 每元素 `Slots[token]`（int64 读、int64 地址运算） | 每行读一次并 `T.cast(Slots[token], "int32")`；**编译期 guard**：`physical_tokens >= 2**31` 时保持 int64 | 索引域 `[-1, NP·PS)` ⊂ int32（`NP·PS < 2^31` 时）；guard 覆盖反例域并提出报错/回退 ⇒ 在合法域内逐位等价 | 索引读 8192 次×8 B（理想）vs 每元素一次 67.1 MB；int32 地址运算减少 64 位运算对 |

**等价性机器验证（D-1，`verify_equiv.py` 执行结果）**：

| 项 | 类型 | 对照元素数 | 违反数 | 违反率 | 最大 ulp 差 | 结论 |
|----|------|-----------|-------|--------|------------|------|
| O1 scale 提升（保留 fp32 除法）`scale=1.25, bfloat16` | 采纳 | 65536 | 0 | 0.000e+00 | 0.000 | **EQUIV_PASS** |
| O1 scale 提升（保留 fp32 除法）`scale=0.75, bfloat16` | 采纳 | 65536 | 0 | 0.000e+00 | 0.000 | **EQUIV_PASS** |
| O1 scale 提升（保留 fp32 除法）`scale=1.25, float16` | 采纳 | 65536 | 0 | 0.000e+00 | 0.000 | **EQUIV_PASS** |
| O1 scale 提升（保留 fp32 除法）`scale=0.75, float16` | 采纳 | 65536 | 0 | 0.000e+00 | 0.000 | **EQUIV_PASS** |
| O2 16-bit 恒等路径 `bfloat16` | 采纳 | 65536 | 0 | 0.000e+00 | 0.000 | **EQUIV_PASS** |
| O2 16-bit 恒等路径 `float16` | 采纳 | 65536 | 0 | 0.000e+00 | 0.000 | **EQUIV_PASS** |
| O3 slot 窄化 int64→int32（域 `-1 .. 65536` 全枚举） | 采纳 | 65538 | 0 | 0.000e+00 | 0.000 | **EQUIV_PASS** |
| O3 窄化 guard（`physical_tokens >= 2**31` 时保持 int64）：4 个极端索引中 3 个在无 guard 时会回绕 | 采纳（guard 检查项） | 4 | 3（无 guard 时的反例计数） | 7.500e-01 | 0.000 | **EQUIV_PASS**（guard 生效后 0 失配） |
| G1 golden vs fp64 参照 `scale=1.25, bfloat16` | 检查项 | 65536 | 0 | 0.000e+00 | 0.000 | **EQUIV_PASS** |
| G1 golden vs fp64 参照 `scale=0.75, bfloat16` | 检查项 | 65536 | 0 | 0.000e+00 | 0.000 | **EQUIV_PASS** |
| G1 golden vs fp64 参照 `scale=1.25, float16` | 检查项 | 65536 | 0 | 0.000e+00 | 0.000 | **EQUIV_PASS** |
| G1 golden vs fp64 参照 `scale=0.75, float16` | 检查项 | 65536 | 0 | 0.000e+00 | 0.000 | **EQUIV_PASS** |

脚本收尾行：`adopted items: 8, failed: 0` / `ADOPTED_VERDICT: ALL_PASS`。验证域说明：O1/O2 对 fp16/bf16 的**全部 65536 个 bit pattern** 穷举（含 NaN/±inf/subnormal 角点），O3 对 `-1..65536` 全枚举 + 极端值；判据为「优化式与基线式在 fp64 参照下逐位一致（ulp 差 = 0）」。**被否决项的失配数字见 §11 附录**（X1 倒数乘法、X2 16-bit 域缩放、X3 只靠 cast 饱和）。

**优化结论**：采纳 3 项（O1/O2/O3）+ 1 项语义必需项（§0.6 R2 的显式 `clamp(-448, 448)`，非性能优化、不计入等价比价），全部 EQUIV_PASS。**优化后公式**（供 §3.1 拆解）：

```
fp8:    d = cast_f32(X[token, col]) / s ;  d = clamp(d, -448.0, 448.0) ;  Pages[slot, col] = cast_fp8(d)
16-bit: Pages[slot, col] = X[token, col]
其中 s ∈ {ks, vs} 为块内标量（O1），slot 为每行一次载入的 int32（O3），
16-bit 分支无算术（O2）；clamp 与 NaN 兜底为 §0.6 R2 的语义必需项。
```

无优化空间的方向已核实并写明结论：本算子**无公共子表达式**（除已提升的标量）、**无高代价算子可降代**（无 `sqrt/exp/rsqrt`）、**无归约可合并**、**访存已到「每元素一读一写」下界**（除索引项已被 O1/O3 消尽）。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

> 盘点实现方案（§3.3 伪代码）中的全部循环与标量计算点，逐项给出向量替代方案或不可替代的具体理由。

| # | 计算点 | 原实现形态 | 向量替代方案 | 是否替代 | 不可替代理由（不可替代时必填） |
|---|--------|-----------|-------------|---------|------------------------------|
| 1 | 逐元素散射写（K/V 两路，共 2048 元素/块） | 标量二重循环 `for n: for c:` | `T.Parallel(block_rows, block_cols)`（SimtVF 内，2048 元素 / 128 线程 = 16 元素/线程，自动向量化 + 直接 GM store） | ✅ | — |
| 2 | fp32 提升与除法 | 标量 `cast` + `/` | `T.Parallel` 内的逐元素表达式（`vcvt` + `vdiv`；`test_simtvf_arithmetic.py`/`test_tilelang_ascend_simtvf_cast.py` 用例） | ✅ | — |
| 3 | 饱和 clamp | 标量 `min(max(x, -448), 448)` | `T.clamp`（→ `vmaxs` + `vmins`，`test_tilelang_ascend_clamp.py` 覆盖 fp16/fp32、threads 32/128、±inf） | ✅ | — |
| 4 | fp8 转换写 | 标量 cvt + 存 | `T.cast(d, cache_dtype)` → `vcvt`（`test_tilelang_ascend_simtvf_cast.py` 对 fp32↔e4m3 断言 bit-exact） | ✅ | — |
| 5 | 16-bit 直拷 | 标量赋值 | 同 #1（`vld` → `vsts`，`T.cast` 退化为同 dtype 无 op） | ✅ | — |
| 6 | `slot >= 0` 行跳过 | 标量分支 | lane 级谓词化写（SimtVF 内 `if`，`test_simtvf_vectorized_select.py` 有 lane 级 `if lane < active_threads` 的 codegen 证据） | ✅（保留为向量化谓词，不降级为标量循环） | 设备侧谓词写待 Stage 3 确认（§9 R1）；**不可改用数值选择**：写回旧值需读 `Pages[-1, col]`（越界） |
| 7 | `token < tokens` 尾块 guard | 标量比较 | 无法向量替代 | ❌ | 动态边界：`tokens` 运行期才知（`T.dynamic`），grid 上界 `ceil(tokens/R)` 必然覆盖一个不满的尾块；只能 lane 谓词（同 #6） |
| 8 | `col < row_width` 列 guard | 标量比较 | 由 §5.4 的 `cols` 归一化在**编译期**消除 | ✅（消除，不需运行期判断） | — |
| 9 | 块索引折叠 `rb = bx // nchunks`、`cb = bx % nchunks` | 每块一次标量整除/取模 | 无 | ❌ | block 级索引/任务映射计算：每块 2 次整数运算，非逐元素热点；`nchunks` 是编译期常量（2 的幂时可退化为移位） |
| 10 | 数据相关目标地址 `Pages[slot, col]` | lane 内地址计算 | 无（`T.simd.vscatter` 的索引类型为 uint16，`physical_tokens` 量级 67 M 元素 > 65535 ⇒ 不可用） | ❌ | 所需向量 API 不可用（索引宽度不足，实测外部仓 SIMD scatter 签名）；scatter 地址本就是逐 lane 标量 → 由 #1 的向量化写天然承载，无需单独替代 |
| 11 | host：`view` 展平、零 token 早退、config 分发 | Python 元数据 | 无（不进 kernel） | ❌ | host 侧元数据计算，不在 kernel 内 |
| 12 | 块内行循环（B 备选路径的行级 `for i in T.serial(16)`） | 顺序行循环 | 主选路径 P 无此循环（行维已在 `T.Parallel` 内并行） | 不适用 | 仅 B 备选路径需要；弃选 B 即消去该循环（见 §1.6.3 布局决策） |

**向量化结论**：逐元素计算已**全部向量化**（#1–#5 走 `T.Parallel` + v 前缀算子，替代方案均有外部仓 ascend 用例佐证）；保留 4 处标量/循环点，理由类别分别为：**动态边界**（#7 `tokens` 尾块）、**block 索引**（#9 折叠运算）、**API 缺失**（#10 `vscatter` 索引宽度不足）、**host 元数据**（#11 展平/早退）；#6 保留为 lane 谓词（向量化执行、非标量循环），#8 已由编译期归一化消除。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

> I/O layout 是契约（`k/k_pages` 的 NCHW-式布局 `(tokens, heads_kv, dim)` 与 `(num_pages, page_size, heads_kv, dim)` 不可改），但**核内向量化轴与暂存布局是设计变量**。本算子有 2 个可并行维（token 行 / 行内列），必须枚举候选。

**轴质量评分**（对选定布局的候选轴）：

| 评分项 | 说明 | 判定 |
|--------|------|------|
| 整除性 | 轴长是否为向量宽度整数倍 | 行内列轴 `row_width ∈ {128, 512, 1024}`，均为 128（fp8 下 256B/128 元素）的整数倍 ✓；token 轴为动态值（无整除要求，由尾块 guard 覆盖） |
| 尾 lane 浪费率 | `(ceil(len/vw)·vw − len)/len` | 列轴：`cols` 归一化后 **0%**；token 轴：尾块最多浪费 15/16 行（`odd-tokens` 用例 7 token → 浪费 56%，但仅 1 块、绝对代价 0.01 µs） |
| 累加链形态 | 该轴是否为串行累加维 | 无归约 ⇒ 两轴皆无累加链；列轴 stride = 1（连续），token 轴 stride = `row_width`（跨行跳变） |
| repack 代价 | 改布局的重排代价 | **0**（选定方案不改 I/O 布局，无核内重排） |
| UB 容量影响 | 布局对驻留缓冲的影响 | **0**（主选路径无 UB buffer） |

**候选矩阵**（全部枚举，弃选给量化理由）：

| # | 布局方案 | 向量化轴 | repack 路径 | 预估收益/代价 | 是否采纳 |
|---|---------|---------|------------|--------------|---------|
| 1 | I/O 原生行主序；块内 `T.Parallel(block_rows, block_cols)` | 行内 `col`（stride = 1，向量主维）+ token 行（每 lane 一行） | 无 | 基线：46.0 µs（prefill）／54.9 µs（fp8，发射主导） | **✅ 采纳** |
| 2 | 转置布局（核内按列主序组织，把 token 做向量维） | token 轴（stride = `row_width`） | 需 UB 内转置（`vsts` 跨步写） | 写目标行由 `slot` 决定、行间地址跳跃 ⇒ 转置后向量维指向**不同池行**，等于把一次成段写拆成 `block_rows` 次单元素 scatter：指令数 ×16、无合并 | ❌ 明确更差 |
| 3 | 行主序 + `T.simd.vscatter` 逐元素索引写 | 列轴 + 显式 scatter | 需先构造 uint16 索引 UB | `vscatter` 索引类型为 uint16 → 上限 65535 元素，本算子池规模 67 M 元素 ⇒ 越界不可用；且额外索引构造流量 ≥ 2·cols B/行 | ❌ API 不可用（索引宽度） |
| 4 | UB 中转 + 行级 DMA（B，布局不变，仅加暂存） | 列轴（行级连续段） | UB 内无需重排 | 同流量 + 每块 64 条 `T.copy`（store 侧 32 条）+ 12.0 KB UB；仅在 §9 R1/R2 触发时作为降级路径（裁决阈值见 §1.6.3 T1–T3） | ❌ 非主选（备选保留，§3.3） |
| 5 | host 侧 permute 进出（把池改成列主序） | 任意 | 两次 full-pool permute | 全池拷贝量 ≥ 134 MB 量级 ≈ 91 µs（引用 CONST-mte2-degradation 口径），远超算子本体 46 µs | ❌ repack 代价 > 本体 |
| 6 | 核内 NHWC 式通道重排 | — | — | 本算子无「通道」维可重排：`row_width = heads_kv·dim` 已是一维连续，拆 `(heads_kv, dim)` 不产生更优向量维（`dim ∈ {64,128}` 亦连续） | ❌ 无收益 |

**布局决策结论**：选定方案 **#1**：核内布局 = **I/O 原生行主序**（`Pages[slot, col]` 与 `X[token, col]` 均为 `(rows, row_width)` 行视图，无任何重排），**向量化轴 = 行内 `col`（stride = 1）**，token 行作为 `T.Parallel` 的第一维分摊到 lane；**repack 路径 = 无**。该结论同步落入 §3.3 伪代码（`for i, c in T.Parallel(block_rows, block_cols)`，内层 `c` 是连续轴）与 §6 循环结构（无内层串行维）。弃选方案的量化理由见候选矩阵 #2–#6。

**主选 / 备选的结构位置**：主选 **P** = §3.3 主选路径（GM 直写 + lane 谓词，0 B 片上缓冲）；备选 **B** = §3.3 备选路径（UB 中转行级 DMA，与 P **同构的 1-D 折叠网格** `nblocks = ceildiv(tokens, R)·nchunks`、`rb = bx // nchunks`、`cb = bx % nchunks`，列区间 `[c0, c0+cols)` 且 `c0 = cb·cols`）。两路径的代价量化见 §11.4；**本节与 §11.4 的裁决口径完全一致并互相引用**（§11.4 只复述阈值编号，不另立标准）。

##### 实验裁决计划（Stage 4 执行；因 P/B 取舍依赖 §9 R1/R2 的未设备验证能力与未实测写代价比）

**(a) 代表 shape**（前两个为必测，覆盖 16-bit 与 fp8 两类写路径；第三个仅作小规模参照）：

| 代表 shape | `tokens` | `heads_kv × dim = row_width` | 输入 dtype | cache dtype | 逻辑块数（`block_rows=16, cols=128`） | 检验对象 |
|---|---|---|---|---|---|---|
| `llama-8b-prefill` | 8192 | 1024（8×128） | bfloat16 | bfloat16 | 4096 | R1：lane 谓词化 GM 直写的吞吐（16-bit 写路径） |
| `qwen3-235b-fp8` | 4096 | 512（4×128） | bfloat16 | float8_e4m3fn | 1024 | R2：fp8 单字节 GM 写的事务效率 |
| `llama-8b-decode` | 64 | 1024 | bfloat16 | bfloat16 | 32 | 参照：小规模启动开销，**不参与阈值判定** |

**(b) 指标与反解式**：同一 shape 下 P、B 各跑一次 kernel 级 msprof（`scope=kernel`、unit `us`），取 `L_P`、`L_B`；原始数据落 `perf_opt/` 的 `msprof_raw_*` 并在 `perf_feedback.md` 记 `raw_evidence_path`。写侧有效带宽未知量记 `BW_write_eff`，`W` 为该 shape 的 GM 写字节数（P/B 相同）。

- **16-bit shape（同 dtype、同块形）**：P 与 B 的 GM 读写流量相同（B 的 UB 往返不计 GM），故 `Δc = (L_P − L_B) / W`（单位 µs / GM 字节）只反映 B 多出的 `4·block_rows` 条 `T.copy` 常数代价；在 `Tr ∝ W` 下写带宽无法从同 dtype 对比中分离 ⇒ **该 shape 只用时延比判定（T1）**，绝对带宽需外部锚点。
- **fp8 shape（P 单字节直写 vs B 128 B 行段 DMA）**：两路径写字节数不同，令 `ΔB_w = write_bytes_P − write_bytes_B`（由 msprof 写事务数 × 128 B 事务粒度折算），以该 shape 的写字节数 `W` 与上一步的 `Δc` 反解

  `BW_write_eff = ΔB_w / ((L_P − L_B) − W·Δc)`，

  即「写字节差 ÷ 扣除常数指令代价后的时延差」。若 `ΔB_w > 0` 却测到 `L_P ≤ L_B`（符号矛盾），或解出的 `BW_write_eff` 落在物理区间之外（以 `CONST-mte2-degradation` 的 1.26 TB/s @48 AIV 按本平台 56 AIV 折算的 1.47 TB/s 为中心、±50% 为界，属未实测外推），判为模型不匹配 ⇒ **只采信与模型无关的时延比（T1），不采信 `BW_write_eff` 绝对值**。
- **两 shape 联立**：模型只有两个未知量——常数指令代价 `Δc` 与写带宽 `BW_write_eff`。16-bit shape（写字节数相同、`Tr ∝ W`）先解出 `Δc`，fp8 shape 用该 `Δc` 消去指令项后解出 `BW_write_eff`，构成 2 方程 2 未知量的封闭系统，无需外部标定常数。

**(c) 可机械判定的阈值**（逐 shape 独立判定，结果按 `cache_dtype` 落盘）：

- **T1（时延比）**：`L_P ≥ 1.15 × L_B` ⇒ 该 shape 切 B。系数 `1.15 = 1 + 2·CV + 5%`，`CV` 为同 shape 重复测量的变异系数（**设计假设 5%，未实测**）；实测 `CV > 5%` 时阈值按 `1 + 2·CV + 0.05` 上修后重判（保证判决裕度 > 2σ 测量噪声）。
- **T2（写事务碎片率）**：fp8 shape 且 `ratio_tx = write_transactions / tx_min ≥ 2` ⇒ 该 fp8 组合切 B，其中 `tx_min = 2 · nchunks · ceil(cols · out_bytes / 128)` 为 K/V 两路、按 128 B 事务粒度计的最小事务数（`out_bytes = 1`）；`ratio_tx ≥ 2` 表示单字节直写把行段碎片化到最小事务数的 2 倍以上。
- **T3（语义门，优先于 T1/T2）**：Stage 3 精度验证若出现 R1（负 slot 行被写）或 R2（fp8 直写结果错）失败 ⇒ 不等性能数据，直接切 B。
- **组合规则**：同一 shape 被 T1 或 T2 任一命中即切 B；T3 命中即对该 kernel 特化全局切 B；三者皆未命中则维持 P。

**(d) 按 `cache_dtype` 分派**：工厂 `_paged_cache_write_kernel(heads_kv, dim, physical_tokens, in_dtype, cache_dtype)` 已按 `cache_dtype` 特化出两个独立 `prim_func`（§3.3），故判定结果可**只对 fp8 组合生效**（16-bit 走 P、fp8 走 B）而**不增加编译产物数量**；B 变体的 UB 预算按 dtype 分别为 fp8 12,288 B（12.0 KB）/ 16-bit 8,192 B（8.0 KB），均 ≪ 253952 B（§4.5）；块数与块粒度与 P 完全相同 ⇒ §5.5 的分核三档判定原样适用。

**(e) 回写路径**：Stage 4 把 (a)–(c) 的实测与反解结果以 `[DESIGN_LIMIT]` 附录补记写入 `perf_opt/perf_feedback.md`（schema 与路由见 `.agents/skills/_shared/standards/perf-feedback.md`），由 conductor 受控路由为「附录补记 / 设计修订 / 不处理」三类之一；**本节与 §11.4 的阈值以该附录补记为准更新**，未回写前一律维持主选 P。
---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: SIMT（`T.SimtVF(threads=128)` 线程域 + `T.Parallel` 逐元素并行表达式）

### 2.2 选型理由

本算子是**纯逐元素散射写**：无 matmul（排除 Cube）、无归约（排除 `T.reduce_*` 需求）、无跨 block 依赖（排除跨核 barrier / persistent 结构）、每块只有一个 tile（排除流水线多 stage）。三个可选路径的取证情况对比：

| 路径 | 承载 API | 本算子所需算子的取证情况 | 判定 |
|------|---------|------------------------|------|
| SIMT | `T.SimtVF(threads=…)` + `T.Parallel` + 逐元素表达式 | 逐元素 v 前缀算子在 SimtVF 内被大量用例覆盖：`test_tilelang_ascend_clamp.py`（`vmaxs/vmins`，fp16/fp32，threads 32/128，含 ±inf）、`test_tilelang_ascend_simtvf_cast.py`（`vcvt`，fp32↔e4m3 **bit-exact**，n = threads×1/2/4/8/16）、`test_simtvf_arithmetic.py`（fp8 逐元素算术，n ≤ 2048）、`test_dynamic_outputs.py`（动态网格 + **GM 直接逐元素写**，设备验证）、`test_ascend_fractal_storage_integration.py`（2-D `T.Parallel(rows, cols)` → **GM 直接写**，设备验证） | **✅ 选定** |
| SIMD | `T.SimdVF()` + `T.simd.vcvt/vdiv/vmaxs/vmins/vsts` | 寄存器级指令可表达同一算式，但 fp8 目标 dtype 的 `vcvt/vsts` 组合在外部仓 SIMD 测试集**未取证**（证据缺口）；且 SIMD 路径只能操作 UB 区域，必须引入 UB 中转 | 作为 fp8 链的 Stage 4 备选（§2.4） |
| Cube | `T.gemm` / L1+L0 | 本算子无矩阵乘，无可用计算单元匹配 | ❌ 不适用 |
| 混合 | Cube + Vector | 同上，无 Cube 部分 | ❌ 不适用 |

补充依据：平台规则 `.agents/platforms/toolchains/ascend/workflow.md` 明确「SIMT 的逐元素表达式放在 `T.SimtVF` 内」「向量操作参考上游 elementwise 的 `T.SimdVF`、`T.simd.*` 或 `T.SimtVF`」，即**模式由实际执行路径选择**；本算子最接近的已验证执行形态是「1-D 动态网格 + `T.SimtVF` + GM 直写」（`test_dynamic_outputs.py`），故取 SIMT。

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | 主选路径**不分配任何片上缓冲**（无 `T.alloc_shared`、无 `T.alloc_l1/l0*`）；GM 张量按 ND 行视图直读直写。B 备选路径才会用 `T.alloc_shared((block_rows, cols), dtype)`（= UB，`scope="shared.dyn"`） |
| 计算方式 | `for i, c in T.Parallel(block_rows, block_cols)` 内的逐元素表达式（`T.cast` / `/` / `T.clamp` / 直接赋值），由 Ascend 管线向量化到 `vcvt / vdiv / vmaxs / vmins / vsts`；无显式 `T.simd.*` 手写指令 |
| 线程域 | `T.SimtVF(threads=128)` 在 kernel body 内打开真实 threadIdx 域（kernel 级 `T.Kernel` 无 `threads` 参数，传了会被 Python 直接拒绝） |
| 同步 | 全部走自动同步（§7）：无 `T.copy`、无共享缓冲、无跨线程通信 |

### 2.4 与平台迁移默认的偏离说明

平台 `programming_modes.migration_default = simd`，本设计选 **SIMT**，偏离理由与边界如下：

1. **默认值的适用面**：`simd` 默认面向「寄存器级手写向量指令」的路径（`T.SimdVF` + `T.simd.*`）。本算子需要的是**数据相关地址的散射写**（`Pages[slot, col]`），而 SIMD scatter 的唯一 API `T.simd.vscatter` 的索引类型为 uint16（上限 65535 元素），本算子池规模 `physical_tokens·row_width` 达 67 M 元素量级 ⇒ 索引宽度不匹配（§1.6.3 候选 #3）。
2. **取证面**：本算子 fp8 链所需的 `vcvt(fp32→e4m3)` + 逐元素 `vdiv/vmaxs/vmins` + GM 直写，全部在 **SimtVF** 用例中被设备验证（§2.2 表）；`T.simd.*` 的 fp8 目标 dtype 组合**未取证**。平台规则允许按实际执行路径选模式，故取证据更完整的一侧。
3. **偏离的代价**：SIMT 路径的线程域语义（`T.Parallel` → lane 摊分 → 16 元素/线程）由编译器决定向量宽度，设计上少一层手写指令控制；性能杠杆相应改为 `block_rows/block_cols` 与 `threads` 常量（§6.3）。
4. **保留的 SIMD 备选路径**（仅 fp8 链，Stage 4 若 SIMT 发射开销成为瓶颈时启用）：`T.SimdVF()` 内改用 `T.simd.vcvt(T.simd.vld(...)) → T.simd.vdiv(..., ks) → T.simd.vmaxs(-448.0) → T.simd.vmins(448.0) → T.simd.vsts(...)`，配合 B 备选路径的 UB 中转；启用前须先补齐 fp8 `vsts` 的取证（待 Stage 3/4 编译确认）。**该备选不改变数学语义**（同算式、同 dtype 转换），只改变指令表达。

---

## 3. API 映射设计

### 3.1 公式拆解

> 输入公式为 §1.6.1 的**优化后公式**；每步使用选定平台可用的向量 API 或线程并行表达式，与 §1.6.2 的向量化结论一致。

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | `nblocks = ceildiv(tokens, R) · nchunks`，`rb = bx // nchunks`，`cb = bx % nchunks` | 1-D 块任务映射（§0.6 R1）；`nchunks = ceildiv(row_width, cols)` 为编译期常量 |
| 2 | `token = rb·R + i`，`col = cb·cols + c`，`(i, c) ∈ [0,R) × [0,cols)` | 块内元素坐标（与源实现逐字一致） |
| 3 | `valid = (token < tokens)` | 尾块谓词（`col < row_width` 已由 `cols` 归一化消除） |
| 4 | `slot = cast_int32(Slots[token])`（每行一次，O3） | 行级索引载入 + 窄化（含 `physical_tokens ≥ 2^31` 时保持 int64 的编译期 guard） |
| 5 | `write = valid ∧ (slot ≥ 0)` | 写使能（负 slot 跳过） |
| 6 | `d_k = cast_f32(K[token, col]) / ks`；`d_v = cast_f32(V[token, col]) / vs` | fp32 提升 + 除法（O1：`ks/vs` 块内标量；**不用倒数乘法**） |
| 7 | `d_k = clamp(d_k, -448.0, 448.0)`（NaN 兜底：`if_then_else(d==d, clamp(d), d)`） | 饱和（§0.6 R2 语义必需） |
| 8 | `KPages[slot, col] = cast_fp8(d_k)`；`VPages[slot, col] = cast_fp8(d_v)` | fp8 存储 |
| 9 | 16-bit 分支：`KPages[slot, col] = K[token, col]`；`VPages[slot, col] = V[token, col]` | O2：零算术直拷 |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 1 | 1-D 块网格 | `T.Kernel(nblocks)`，`as bx` | `nblocks` 为动态表达式（含 `tokens`）；**不传 `threads`**（ascend 方言拒绝该参数） | SIMT |
| 1' | 线程域 | `T.SimtVF(threads=128)` | `threads=128`（`latency` 默认 0）；与 `T.Kernel` 同一 `with` 语句链式打开 | SIMT |
| 1'' | 动态轴 | `T.dynamic("tokens")` | 返回符号变量，用于 `T.Tensor((tokens, row_width), …)` 的 extents 与 `nblocks` | — |
| 2–3 | 元素坐标与尾块谓词 | `for i, c in T.Parallel(R, cols)` + `if token < tokens` | 二维并行迭代（2048 元素）+ lane 级谓词 | SIMT |
| 4 | slot 载入/窄化 | `slot = T.cast(Slots[token], "int32")` | 索引表达式直接读 GM 标量 | SIMT |
| 5 | 负 slot 跳过 | `if slot >= 0:` | lane 级条件语句（谓词化 store） | SIMT |
| 6 | fp32 提升 + 除法 | `T.cast(K[token, col], "float32") / ks` | `ks = KScale[0]` 在 kernel body 顶部 hoist（O1） | SIMT |
| 7 | 饱和 | `T.clamp(d, -448.0, 448.0)` | 等价 `T.min(T.max(d, -448.0), 448.0)`（common 层 `clamp(dst, min_val, max_val)`） | SIMT |
| 7' | NaN 兜底（条件启用） | `T.if_then_else(d == d, T.clamp(d, -448.0, 448.0), d)` | 仅当 Stage 3 证明 `vmaxs/vmins` 不传播 NaN 时启用（§9 R3） | SIMT |
| 8 | fp8 存储 | `KPages[slot, col] = T.cast(d, cache_dtype)` | `cache_dtype = "float8_e4m3fn"`；GM 直接写 | SIMT |
| 9 | 16-bit 直拷 | `KPages[slot, col] = K[token, col]` | 独立 `prim_func` 分支（无算术，O2） | SIMT |
| B1 | 备选：UB 分配 | `T.alloc_shared((R, cols), in_dtype)` | ascend 方言签名 `alloc_shared(shape, dtype, scope="shared.dyn")`；返回 UB Buffer | SIMT（B 路径） |
| B2 | 备选：GM↔UB 搬移 | `T.copy(K[token, c0:c0+cols], k_ub[i, :])`（`c0 = cb*cols`，其余三处同型） | 区域 copy（两侧元素数须相等，形状可不同）；1-D 长度 `cols` ↔ UB 行 `(cols,)`；替代写法见 §3.3 | SIMT（B 路径） |

### 3.3 计算伪代码

**主选路径 P（直写）**：

```python
import tilelang
import tilelang.ascend.language as T

def _paged_cache_write_kernel(heads_kv, dim, physical_tokens, in_dtype, cache_dtype):
    row_width = heads_kv * dim
    quantized = cache_dtype == "float8_e4m3fn"
    threads = 128                      # §0.6 R3：常量，非参数

    @tilelang.jit(out_idx=[])
    def _func(block_rows, block_cols):
        tokens = T.dynamic("tokens")
        # 编译期常量：宽度归一化（§1.4 步骤 7 / §5.4）
        cols = block_cols if (row_width % block_cols == 0 and block_cols <= row_width) else row_width
        nchunks = (row_width + cols - 1) // cols
        nblocks = T.ceildiv(tokens, block_rows) * nchunks

        if quantized:
            @T.prim_func
            def _fp8(
                K: T.Tensor((tokens, row_width), in_dtype),
                V: T.Tensor((tokens, row_width), in_dtype),
                KPages: T.Tensor((physical_tokens, row_width), cache_dtype),
                VPages: T.Tensor((physical_tokens, row_width), cache_dtype),
                Slots: T.Tensor((tokens,), "int64"),
                KScale: T.Tensor((1,), "float32"),
                VScale: T.Tensor((1,), "float32"),
            ) -> None:
                ks = KScale[0]                       # O1：标量 hoist
                vs = VScale[0]
                with T.Kernel(nblocks) as bx, T.SimtVF(threads=threads):
                    rb = bx // nchunks               # §0.6 R1：1-D 折叠
                    cb = bx % nchunks
                    for i, c in T.Parallel(block_rows, cols):
                        token = rb * block_rows + i
                        col = cb * cols + c
                        if token < tokens:           # 尾块谓词（动态边界）
                            slot = T.cast(Slots[token], "int32")   # O3
                            if slot >= 0:            # 负 slot 跳过
                                kd = T.cast(K[token, col], "float32") / ks
                                vd = T.cast(V[token, col], "float32") / vs
                                KPages[slot, col] = T.cast(T.clamp(kd, -448.0, 448.0), cache_dtype)
                                VPages[slot, col] = T.cast(T.clamp(vd, -448.0, 448.0), cache_dtype)
            return _fp8

        @T.prim_func
        def _16bit(
            K: T.Tensor((tokens, row_width), in_dtype),
            V: T.Tensor((tokens, row_width), in_dtype),
            KPages: T.Tensor((physical_tokens, row_width), cache_dtype),
            VPages: T.Tensor((physical_tokens, row_width), cache_dtype),
            Slots: T.Tensor((tokens,), "int64"),
        ) -> None:
            with T.Kernel(nblocks) as bx, T.SimtVF(threads=threads):
                rb = bx // nchunks
                cb = bx % nchunks
                for i, c in T.Parallel(block_rows, cols):
                    token = rb * block_rows + i
                    col = cb * cols + c
                    if token < tokens:
                        slot = T.cast(Slots[token], "int32")
                        if slot >= 0:
                            KPages[slot, col] = K[token, col]   # O2：零算术
                            VPages[slot, col] = V[token, col]
        return _16bit
```

**备选路径 B（UB 中转行级 DMA；仅在 §9 R1/R2 触发时启用，裁决阈值见 §1.6.3 T1–T3）**：

> 与主选 P **同构的 1-D 折叠网格**：`nblocks`、`nchunks`、`cols` 全部沿用上面的编译期常量；`rb = bx // nchunks`、`cb = bx % nchunks`、`c0 = cb * cols`。网格与行块划分和 P 完全一致 ⇒ 列段覆盖、尾块与 §5.5 分核判定均与 P 相同。

```python
        # ---- 备选 B / fp8 变体：4 个 UB buffer（k_ub, v_ub, k8_ub, v8_ub）----
        @T.prim_func
        def _fp8_b(
            K: T.Tensor((tokens, row_width), in_dtype),
            V: T.Tensor((tokens, row_width), in_dtype),
            KPages: T.Tensor((physical_tokens, row_width), cache_dtype),
            VPages: T.Tensor((physical_tokens, row_width), cache_dtype),
            Slots: T.Tensor((tokens,), "int64"),
            KScale: T.Tensor((1,), "float32"),
            VScale: T.Tensor((1,), "float32"),
        ) -> None:
            with T.Kernel(nblocks) as bx, T.SimtVF(threads=threads):
                ks = KScale[0]                                   # O1：标量 hoist
                vs = VScale[0]
                k_ub = T.alloc_shared((block_rows, cols), in_dtype)
                v_ub = T.alloc_shared((block_rows, cols), in_dtype)
                k8_ub = T.alloc_shared((block_rows, cols), cache_dtype)
                v8_ub = T.alloc_shared((block_rows, cols), cache_dtype)
                rb = bx // nchunks                               # §0.6 R1：1-D 折叠（同 P）
                cb = bx % nchunks
                c0 = cb * cols                                   # 列段基址
                for i in T.serial(block_rows):                   # 行级串行 = B 的代价
                    token = rb * block_rows + i
                    if token < tokens:                           # 行维尾块 guard
                        slot = T.cast(Slots[token], "int32")     # 行级 slot 为寄存器标量（不占 UB）
                        if slot >= 0:                            # 负 slot 行级跳过
                            T.copy(K[token, c0:c0 + cols], k_ub[i, :])   # GM -> UB（1-D 区间，长度 cols）
                            T.copy(V[token, c0:c0 + cols], v_ub[i, :])
                            for c in T.Parallel(cols):               # 列轴向量化（§1.6.3 选定轴）
                                k8_ub[i, c] = T.cast(
                                    T.clamp(T.cast(k_ub[i, c], "float32") / ks, -448.0, 448.0),
                                    cache_dtype)
                                v8_ub[i, c] = T.cast(
                                    T.clamp(T.cast(v_ub[i, c], "float32") / vs, -448.0, 448.0),
                                    cache_dtype)
                            T.copy(k8_ub[i, :], KPages[slot, c0:c0 + cols])  # UB -> GM（同列段，含 c0 偏移）
                            T.copy(v8_ub[i, :], VPages[slot, c0:c0 + cols])
        return _fp8_b

        # ---- 备选 B / 16-bit 变体：仅 2 个 UB buffer（无算术，UB 只作中转）----
        @T.prim_func
        def _16bit_b(
            K: T.Tensor((tokens, row_width), in_dtype),
            V: T.Tensor((tokens, row_width), in_dtype),
            KPages: T.Tensor((physical_tokens, row_width), cache_dtype),
            VPages: T.Tensor((physical_tokens, row_width), cache_dtype),
            Slots: T.Tensor((tokens,), "int64"),
        ) -> None:
            with T.Kernel(nblocks) as bx, T.SimtVF(threads=threads):
                k_ub = T.alloc_shared((block_rows, cols), in_dtype)
                v_ub = T.alloc_shared((block_rows, cols), in_dtype)
                rb = bx // nchunks
                cb = bx % nchunks
                c0 = cb * cols
                for i in T.serial(block_rows):
                    token = rb * block_rows + i
                    if token < tokens:
                        slot = T.cast(Slots[token], "int32")
                        if slot >= 0:
                            T.copy(K[token, c0:c0 + cols], k_ub[i, :])       # GM -> UB
                            T.copy(V[token, c0:c0 + cols], v_ub[i, :])
                            T.copy(k_ub[i, :], KPages[slot, c0:c0 + cols])   # UB -> GM（零算术，O2）
                            T.copy(v_ub[i, :], VPages[slot, c0:c0 + cols])
        return _16bit_b
```

**覆盖性与尾段论证**：`bx ∈ [0, nblocks)`，`nblocks = ceildiv(tokens, R)·nchunks`，故 `(rb, cb) = (bx // nchunks, bx % nchunks)` 是 `[0, ceildiv(tokens,R)) × [0, nchunks)` 上的**双射**；列段并集 `⋃_cb [cb·cols, (cb+1)·cols) = [0, nchunks·cols)`，而 `cols` 归一化保证 `nchunks·cols == row_width`（`row_width % cols == 0` 时；否则 `cols = row_width, nchunks = 1`）⇒ **每一列恰被一块覆盖、无遗漏无重叠**；末列段即 `cb = nchunks−1`（`row_width = 512/1024` 时为第 4/8 段），列方向无尾块（§5.4）。行方向唯一尾块由 `if token < tokens` 覆盖，与 P 相同。

**`s_ub` 已删除（不分配）**：行级 `slot` 在 `T.serial` 循环体内是该行的标量，直接由寄存器承载即满足「每行一次载入」，写入其到 UB 反而多一次 UB store + 一次 UB load 并占用 64 B，且 `T.copy` 的行级 DMA 本就不需要 slot 索引（目标行由标量地址 `KPages[slot, ...]` 表达）。故 B 路径 UB 仅 fp8 4 个 buffer / 16-bit 2 个 buffer（§4.5）。

**`T.copy` region 的确定形式与替代写法**：源 `K[token, c0:c0+cols]` 是「标量行号 + 标量列基址」的 **1-D 连续区间，长度 `cols`**，目标 `k_ub[i, :]` 是形状 `(cols,)` 的 UB 行 ⇒ 两侧元素数相等（与外部仓 `test_tilelang_ascend_dma_copy.py` 的 `T.copy(src[:n, 0, :], ub[:n, :])` 同型）。若 Stage 3 编译拒绝 1-D region，按以下顺序降级（语义不变）：① 2-D region 写法 `T.copy(K[token:token+1, c0:c0+cols], k_ub[i:i+1, :])`（形状 `(1, cols)`，元素数相同）；② 逐元素写法 `for c in T.Parallel(cols): k_ub[i, c] = K[token, c0 + c]`（牺牲 DMA 成段性，仍保持向量化读）。

**每块 / 总 `T.copy` 条数**：每行 K/V 各「读 1 + 写 1」⇒ 4 条/行；`block_rows = 16` ⇒ **64 条/块**（其中 store 侧 32 条），与 §11.4 一致。总条数 `= 4 · block_rows · nblocks = 4 · tokens · nchunks`（`tokens` 为 `block_rows` 整数倍时）：`llama-8b-prefill` 为 4 × 8192 × 8 = 262,144 条；`qwen3-235b-fp8` 为 4 × 4096 × 4 = 65,536 条。折叠只修复列覆盖、不改变每块条数。

### 3.4 API 可行性确认

| API | 证据来源 | 验证状态 |
|-----|---------|---------|
| `T.Kernel(nblocks)`（1-D、无 `threads`） | 选中工具链 `tilelang/ascend/language/kernel.py` 模块文档：*"The NPU launch is a 1-D grid of AI cores with no SIMT thread domain at kernel scope … passing `threads=` … is rejected by Python"*；外部仓 `test_dynamic_outputs.py`（`T.Kernel(ROWS)`） | ✅ 源码级确认 + 外部仓设备用例；本算子用法待 Stage 3 用真实编译确认 |
| `T.SimtVF(threads=128)` | `frame.py`：`def SimtVF(threads: int \| list[int] \| tuple = 128, latency: int = 0)`；`test_tilelang_ascend_clamp.py`（threads 32/128）、`test_dynamic_outputs.py`（threads=128） | ✅ 签名确认 + 用例；待 Stage 3 用真实编译确认 |
| `T.Parallel(R, cols)`（2-D，2048 元素 / 128 线程） | `test_tilelang_ascend_print.py`（16×16）、`test_ascend_fractal_storage_integration.py`（`T.Parallel(rows, cols)` 512–1024 元素，设备验证）；16 元素/线程由 `test_tilelang_ascend_simtvf_cast.py`（n = threads×16 = 2048）覆盖 | ✅ 用例级；**2048 元素/128 线程的组合为 2–4× 外推**，待 Stage 3 用真实编译确认 |
| `T.dynamic("tokens")` | `tilelang/language/symbolics.py`：`def dynamic(name, dtype="int32")`；外部仓 `ROWS = T.dynamic("rows")` | ✅ |
| `T.cast(x, dtype)` | 外部仓 ascend 后端深 gemm 示例的 `transform_sf.py`（多处以 `T.cast` 做 dtype 转换，文件位于 `/home/l00970450/zuo/tilelang-fork` 的 deepgemm kernels 目录）；`test_tilelang_ascend_simtvf_cast.py` 断言 fp32↔e4m3 bit-exact | ✅ |
| `T.clamp(dst, min_val, max_val)` | `tilelang/language/customize.py`：`def clamp(dst, min_val, max_val)`，docstring 注明 *"NaN behavior follows the target's min/max operations without additional checks"*；`test_tilelang_ascend_clamp.py` 覆盖 fp16/fp32/±inf（**未覆盖 NaN**） | ⚠️ 存在性确认；NaN 行为待 Stage 3 用真实编译确认（§9 R3） |
| `T.if_then_else(cond, a, b)` | common language 逐元素 select；`test_simtvf_vectorized_select.py` 验证 SimtVF 内 select 的向量化 codegen | ✅ 存在性；NaN 兜底语义待 Stage 3 用真实编译确认 |
| 直接 GM 逐元素写 `GM[i, j] = 表达式` | `test_dynamic_outputs.py`（`positive[row, i] = source[row, i] + bias`，设备验证、`Exact` 比对）；`test_ascend_fractal_storage_integration.py`（`C[...] = ub[...]`，设备验证） | ✅ 设备用例；**fp8 目标 dtype 的 GM 直写**为唯一未取证组合，待 Stage 3 用真实编译确认（§9 R2） |
| lane 级 `if` 谓词化写 | `test_simtvf_vectorized_select.py` 的 `_vector_condition_kernel` 中 `if lane < active_threads:` 包裹写入（仅生成代码断言，未做数值验证） | ⚠️ codegen 证据；设备语义待 Stage 3 用真实编译确认（§9 R1） |
| `T.alloc_shared(shape, dtype, scope="shared.dyn")`（= UB） | `allocate.py`：`def alloc_shared(shape, dtype, scope="shared.dyn")` | ✅（仅 B 备选路径使用） |
| `T.copy(GM_region, UB)` / `T.copy(UB, GM_region)` | 外部仓 `test_tilelang_ascend_dma_copy.py`（`T.copy(src[:n, 0, :], ub[:n, :])`、`T.copy(acc, out[0, 0])`）；区域 copy 要求两侧元素数相等 | ✅（仅 B 备选路径使用；1-D 区间 `[c0, c0+cols)` ↔ UB 行 `(cols,)` 为确定形式，另有 2-D region 与逐元素两种替代写法，§3.3） |

**静态查证方法说明**：按平台规则（`workflow.md`：*"仓库根的同名源码包会遮蔽安装包，静态查询不要在仓库根运行"*），全部源码核对在外部 dev root `/home/l00970450/zuo/tilelang-fork` 内执行，先确认 `tilelang.__file__` = `/home/l00970450/zuo/tilelang-fork/tilelang/__init__.py`（版本 `0.1.15+git0efd65c8`，`Loading tilelang libs from dev root: /home/l00970450/zuo/tilelang-fork/build`），再读 `tilelang/ascend/language/*.py` 的导出与签名。仓库根 `tilelang/language/` 与 `docs/Tilelang.language/` 属另一工具链，**未**作为本平台 API 依据。

## 3.5 技术约束确认

### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| `T.Kernel` 只有 1-D core grid，无 `threads` 参数 | **是** | §0.6 R1 折叠二维网格为 1-D；线程域由 `T.SimtVF(threads=128)` 在 body 内打开 |
| 逐元素表达式必须在 `T.SimtVF`（SIMT 模式）内 | **是** | §3.3 伪代码的 `T.Parallel` 全部位于 `T.SimtVF` frame 内 |
| UB 容量 253952 B（248 KB），按 `design_budget` 与 INI 来源核算 | **是（仅 B 备选）** | 主选路径 0 B；B 路径 12.0 KB（fp8 变体）/ 8.0 KB（16-bit 变体）≪ 248 KB（§4.5） |
| L0A/L0B/L0C 仅 Cube 使用 | 否 | 无 `T.gemm`，L0 全部不适用（§5.3） |
| `ascend.batch_schedule` 跨 block barrier 未认证 | **是（规避）** | 本设计不做跨 block 同步、不做 persistent 全局归约；每块独立（§5.5、§7） |
| 动态 grid（`T.dynamic`）仅支持 1-D extent | **是** | 只有 `tokens` 一个动态轴，且只进 1-D grid 与张量 extents |
| fp8 目标 dtype 的 GM 直写与 NaN clamp 语义未取证 | **是** | 登记为 §9 风险 R2/R3，并给出兜底式（`T.if_then_else`）与降级路径（B） |
| 无 GM→GM 直通 DMA（搬运必须经 UB） | **是（影响 B 路径）** | B 路径按 GM→UB→计算→UB→GM 组织（源码无 SMEM 直通） |
| `T.simd.vscatter` 索引宽度 uint16（≤ 65535） | **是（排除方案）** | §1.6.3 候选 #3 淘汰；散射地址由 lane 级标量地址承载 |
| SIMD 寄存器路径的 fp8 支持未取证 | **是（排除方案）** | 主选 SIMT；SIMD 保留为 §2.4 的 Stage 4 备选 |
| 精度门槛 `Exact(rtol=0, atol=0)`（逐字节） | **是** | 所有数学化简中被否决项（X1/X2/X3）均因逐位不等价被排除（§11）；fp32 除法与 clamp 保留 |

### 3.5.2 参考实现差异说明

| 差异项 | 参考实现（GPU / `target` 为 CUDA） | 本项目（Ascend `target="ascend"`） | 转换方案 |
|--------|----------------|-----------------|----------|
| Kernel 结构 | `T.Kernel(row_tiles, col_tiles, threads=threads)` 二维 grid + `threads` 参数 | `T.Kernel(nblocks)` 一维 grid，**无** `threads`；线程域靠 `T.SimtVF(threads=128)` | §0.6 R1 折叠 + R3 常量线程域 |
| 语言模块 | `import tilelang.language as T`（`T.alloc_shared` 等为 CUDA 语义） | `import tilelang.ascend.language as T`（`T.Kernel`/`T.alloc_shared` 被 ascend 方言影子化） | 换 import 命名空间；`alloc_shared` 语义变为 UB（本设计主选不用） |
| fp8 饱和 cast | 隐式 `T.Cast`（依赖 CUDA `cvt` 的 saturate + NaN 保留） | 显式 `T.clamp` 前后夹（`vmaxs/vmins` 语义未文档化） | §0.6 R2；兜底 `T.if_then_else` |
| 内存层级 | SMEM（源未使用）+ GM 直写 | UB（本设计主选不使用）+ GM 直写 | 无差异（两侧都直写 GM） |
| 动态 shape | `T.dynamic("tokens")`（与源相同） | 同（`T.dynamic` 在 common 层） | 一致 |
| 饱和/舍入的行为证据 | 源仓单测（GPU 侧） | 外部仓 ascend 用例 | 见 §3.4 表 |
| 网格维度语义 | 二维 block 索引直接可用 | 需 `//`、`%` 手工折叠 | §0.6 R1（`nchunks` 编译期常量，2 的幂时可退化移位） |
| 编译产物缓存 | `@target_cache(maxsize=32)` 装饰器 | wrapper 在 `forward` 内按 config 缓存 `_compiled_fn`（K8/K9 已改造） | 沿用冻结 wrapper，不改接口 |

### 3.5.3 本项目同类实现参考

**结论：本项目 `examples/` 下无 `target="ascend"` 的兼容示例**（全库检索 `target="ascend"` 与 `import tilelang.ascend.language` 均为 0 命中；现有 elementwise/indexer 示例全部是 NPUIR 后端 `target="npuir"`，使用 `T.alloc_ub`、`T.symbolic`、`T.Kernel(..., is_npu=True)`，与本平台 API 不同源）。下表列出**结构最接近的参考文件**，只借鉴任务映射与循环组织，API 一律按 §3.2 的 ascend 方言重写：

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/elementwise/vec_add_2d_dynamic_shape.py` | 中（结构参考） | 动态 shape 处理与尾块 `T.min` 边界（NPUIR 后端，`T.symbolic`）；借鉴其「动态轴 + 尾块保护」的组织方式，API 换为 `T.dynamic` |
| `examples/elementwise/vec_add_2d.py` | 中（结构参考） | 2-D 逐元素并行的 tile 划分与 `T.Parallel` 使用形态 |
| `examples/indexer/indexer_fwd.py` | 中（索引路径参考） | 索引/散射类算子的索引张量载入与 guard 组织方式 |
| `.agents/platforms/toolchains/ascend/kernel_template.py` | 高（模板） | 平台 kernel 骨架：`@T.prim_func` + `T.Kernel` + `T.SimtVF(threads=256)` 的 1-D 结构与本设计同构（线程数取 128） |
| 外部仓 `/home/l00970450/zuo/tilelang-fork`（ascend 后端测试集） | 高（API 证据） | `test_dynamic_outputs.py`（1-D 动态网格 + SimtVF + GM 直写）、`test_ascend_fractal_storage_integration.py`（2-D `T.Parallel` → GM 写）、`test_tilelang_ascend_clamp.py`、`test_tilelang_ascend_simtvf_cast.py`、`test_simtvf_arithmetic.py`、`test_simtvf_vectorized_select.py`、`test_tilelang_ascend_dma_copy.py` |
---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `k` | `(tokens, heads_kv, dim)`，kernel 内行视图 `(tokens, row_width)` | `float16` / `bfloat16` | 新 key；`tokens = T.dynamic("tokens")`，`row_width = heads_kv·dim` 为编译期常量 |
| `v` | 同 `k` | 同 `k` | 新 value |
| `k_pages` | `(num_pages, page_size, heads_kv, dim)`，kernel 内行视图 `(physical_tokens, row_width)` | 与 `k` 同 dtype（16-bit 组合）或 `float8_e4m3fn`（fp8 组合） | key page pool，**原地写**；`physical_tokens = num_pages·page_size` 为编译期常量 |
| `v_pages` | 同 `k_pages` | 同 `k_pages` | value page pool，**原地写** |
| `slot_mapping` | `(tokens,)` | `int64` | 目标行号；`-1` 表示该 token 跳过 |
| `k_scale` | `(1,)` | `float32` | 仅 fp8 组合传入（wrapper 在量化路径追加该参数） |
| `v_scale` | `(1,)` | `float32` | 同上 |

补注：4-D → 2-D 的展平是 `torch.view` 纯元数据操作（§0.3 host 侧逻辑 2），kernel 侧只见到 2-D 行视图；`k`/`v` 与 `k_pages`/`v_pages` 的行宽相同（都是 `row_width`），因此元素级索引 `X[token, col]` 与 `Pages[slot, col]` 使用同一列坐标。

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `k_pages` | 同 §4.1（原地） | 同 §4.1 | 被写入的行来自 `slot_mapping`；未命名行字节不变 |
| `v_pages` | 同 §4.1（原地） | 同 §4.1 | 同上 |
| 返回值 | — | — | **无返回张量**（`out_idx=[]`）；结果只体现为两个 page pool 的原地修改 |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| 主选路径 P | — | — | — | **不分配任何片上缓冲**：GM → 向量寄存器 → GM（§1.5、§4.4） |
| `k_ub`（仅 B 备选） | `(16, 128)` | `float16` / `bfloat16` | UB（`T.alloc_shared`，`scope="shared.dyn"`） | 每行 K 的 GM→UB 中转 |
| `v_ub`（仅 B 备选） | `(16, 128)` | 同 `k_ub` | UB | 每行 V 的 GM→UB 中转 |
| `k8_ub`（仅 B 备选，fp8） | `(16, 128)` | `float8_e4m3fn` | UB | 量化结果驻留，供 UB→GM 行级 DMA |
| `v8_ub`（仅 B 备选，fp8） | `(16, 128)` | `float8_e4m3fn` | UB | 同上 |

B 路径的 16-bit 变体只分配 `k_ub` / `v_ub` 两个中转 buffer（零算术直拷）；fp8 变体另加 `k8_ub` / `v8_ub`。行级 `slot` 为寄存器标量，**不分配 UB 空间**（原 `s_ub` 已删除，理由见 §3.3）。

### 4.4 内存搬运路径

**主选路径 P（直写，§3.3 伪代码）**：

```
GM[K(tokens,row_width)]  ──vld（lane 直读，无片上暂存）──► 向量寄存器（fp32 提升 / 除法 / clamp）
GM[Slots(tokens,)]       ──标量读（每行一次，int64→int32）──► 行地址 slot ──► 目标行基址
GM[KScale(1,)]/VScale(1,)──标量读（每块一次，O1 hoist）────► 块内标量 ks/vs
                                                          │
16-bit：寄存器值 ──vsts（同 dtype 直写）──────────────────► GM[KPages(slot,col)] / GM[VPages(slot,col)]
fp8   ：寄存器值 ──vcvt(fp32) → vdiv → vmaxs/vmins → vcvt(fp8) ──vsts──► 同上
```

- 全链路 **无 UB/L1/L0 中转**，无「搬入—计算—搬出」三跳；这是源实现的直写结构在 Ascend 上的承载方式（§0.5 条目 4）。
- 写侧为「行地址数据相关 + 行内列连续」：同一 lane 的 `col` 连续（stride 1），跨 `i` 行时目标地址跳到另一池行 ⇒ 行内可成段，跨行不可合并。

**备选路径 B（UB 中转行级 DMA，仅在 §9 R1/R2 触发时启用；网格与主选 P 同构）**：

```
GM[K[token, c0:c0+cols]]（1-D 区间，token = rb·R+i，c0 = cb·cols）
        ──T.copy──► UB[k_ub 第 i 行] ──T.Parallel(cols) 计算──► UB[k8_ub 第 i 行]
                                                              └──T.copy──► GM[KPages[slot, c0:c0+cols]]
GM[V[token, c0:c0+cols]] ──T.copy──► UB[v_ub 第 i 行] ──同上──► UB[v8_ub 第 i 行] └──T.copy──► GM[VPages[slot, c0:c0+cols]]
```

- 块索引来自与 P 相同的折叠：`rb = bx // nchunks`、`cb = bx % nchunks`；源与目标区间都用同一 `c0 = cb·cols` 偏移，故 `row_width ∈ {512, 1024}`（`nchunks = 4/8`）时**每个列段各由一个块写入**，无遗漏（覆盖性论证见 §3.3）。
- 每行 2 跳（读 1 + 写 1），每块 `4·block_rows = 64` 条 `T.copy`（store 侧 32 条）；`GM→GM` 直通 DMA 在本平台不存在（§3.5.1），故 B 路径必须经 UB。
- 两路径的 **GM 侧流量相同**（每元素一读一写），差异只在 B 多出 UB 往返与行级指令数（§1.6.0 R3、§11.4）。

### 4.5 UB 内存预算

> 预算只针对 **B 备选路径**（主选路径 P 的 UB 占用为 0 B，无需预算表）；取默认 tile `block_rows = 16`、`cols = 128`、16-bit 输入 + fp8 输出（B 路径的最坏 dtype 组合，buffer 数最多）。下表为 **UB** 层级（`T.alloc_shared`，`scope="shared.dyn"`）；L1 / L0A / L0B / L0C 在本算子中不使用（§5.3）。行级 `slot` 为寄存器标量，不占 UB（`s_ub` 已删除，§3.3）。

**UB**（B 备选路径的片上缓冲清单，fp8 变体）

| Buffer | Shape | dtype | 大小 (Bytes) |
|--------|-------|-------|-------------|
| k_ub | (16, 128) | bfloat16 / float16 | 4096 B |
| v_ub | (16, 128) | bfloat16 / float16 | 4096 B |
| k8_ub | (16, 128) | float8_e4m3fn | 2048 B |
| v8_ub | (16, 128) | float8_e4m3fn | 2048 B |
| total_bytes（合计） | — | — | 12288 B |

合计 12.0 KB（fp8 变体）；16-bit 变体只有 `k_ub` + `v_ub` ⇒ 8192 B（8.0 KB）。UB 容量 253952 B（248 KB）⇒ 最坏占用率 4.84% ≤ 100%。容量佐证：CANN 9.1.0 `data/platform_config/Ascend950PR_9579.ini` 的 `[AICoreSpec] ub_size`（文件 md5 `2f0eca1f7ca70f62ee7d6d33990c3be1`，读取日期 2026-11-23），经 `statectl platform --task-dir examples/paged_cache_write/_paged_cache_write_kernel` 的 `design_budget.capacity_bytes.UB` 读取；`ub_inflation` 为 `null`（本平台未配置自动多缓冲折算系数，故不预留放大倍数）。

补充：若 Stage 4 把 `block_rows` 提到 64（每块 64×128 = 8192 元素），B 路径 fp8 变体合计 49152 B（48.0 KB）、16-bit 变体 32768 B（32.0 KB），均 ≤ 248 KB；主选路径 P 与 `block_rows` 无关（0 B）。

### 4.6 动态轴定义

| 动态轴 | 声明方式 | 运行时范围 |
|--------|----------|-----------|
| `tokens` | `tokens = T.dynamic("tokens")`，用于 `K/V` 与 `Slots` 的 extents 及 `nblocks = T.ceildiv(tokens, block_rows)·nchunks` | 契约上为任意非负整数；`tokens == 0` 由 host 提前返回、不发起 kernel（§0.1 边界语义）。workload 实测范围 1 ~ 8192（decode 64 / prefill 8192 / fp8 4096 / gpt-oss 256 / odd 7），grid 上界随 `tokens` 线性变化 |

`physical_tokens`、`row_width = heads_kv·dim`、`cols`、`nchunks`、`block_rows`、`block_cols`、`threads` 全部是**编译期常量**（进 kernel 特化参数），不存在运行期分支；唯一运行期动态量是 `tokens`。

### 4.7 JIT 配置

```python
@tilelang.jit(
    out_idx=[],          # 无返回值：输出为 k_pages / v_pages 的原地写
)
```

wrapper 侧（Stage 0 冻结）以 `tilelang.compile(main, target="ascend", out_idx=[])` 构建，并按 `(block_rows, block_cols)` 缓存编译产物；本设计不引入 `pass_configs`（§7.3）。
---

## 5. Tiling 策略

### 5.1 计算类型

**类型**: 纯 Vector（SIMT 线程域内的逐元素计算）

**判定依据**: 算子只含 4 种逐元素运算——同 dtype 直拷（16-bit 分支）、fp32 提升（`vcvt`）、除法（`vdiv`）、饱和（`vmaxs`/`vmins`）与 fp8 转换（`vcvt`）——没有矩阵乘（无 `T.gemm`）、没有归约（无 `reduce_*`）、没有跨 block 依赖。因此 **Cube 路径与 L0A/L0B/L0C 全部不适用**，L1 也不需要（无 Cube 输入 tile），唯一涉及的可选层级是 UB（仅 B 备选路径，§4.5）。

### 5.2 Block 划分

下列数值全部是编译期常量（进 kernel 特化参数），`M` / `N` 取代表规模（llama-8b-prefill）：

```python
M = 8192          # 代表规模：llama-8b-prefill 的 tokens（动态轴 tokens 的上界实例）
N = 1024          # 代表规模：row_width = heads_kv·dim（llama-8b: 8 × 128）
block_M = 16      # = block_rows：每个逻辑块负责 16 个 token（行）
block_N = 128     # = cols：每块负责行内 128 列（列宽归一化后的值，§5.4）
block_num = (M // block_M) * (N // block_N)   # 512 × 8 = 4096 个逻辑块
```

- `block_rows = 16`：与 wrapper `default_config` 一致；行维取 16 使 decode 形态（`tokens = 64`）仍有 4 个逻辑块、`odd-tokens`（7 token）只需 1 块；同时 `block_rows × cols = 2048` 元素 = 128 线程 × 16 元素/线程，落在已验证的 SimtVF 元素/线程区间（§3.4）。
- `block_cols = 128`（归一化后即 `cols`）：`row_width ∈ {128, 512, 1024}` 全部是 128 的整数倍，故每行被切成 1 / 4 / 8 段；16-bit 下 128 列 = 256 B/行段，fp8 输出侧 128 列 = 128 B/行段，均满足 128 B 事务粒度。
- 列维**不引入尾块**（`cols` 归一化后 `col < row_width` 恒成立），尾块只可能出现在行维（`tokens` 动态）。

### 5.3 约束分析

- **对齐约束**: 写事务按 128 B 计——16-bit 输入行段 `cols·2 = 256 B`（2 个事务）、fp8 输出行段 `cols·1 = 128 B`（1 个事务）。`cols = 128` 是 128 元素宽度，对 2 B/1 B dtype 都保持 128 B 对齐；若 Stage 4 改 `block_cols = 64` 则 fp8 行段降到 64 B（半个事务），不推荐。
- **UB 容量**: 主选路径 **0 B**；B 备选路径 12.0 KB（§4.5 合计 12288 B，fp8 变体；16-bit 变体 8192 B）≤ 248 KB（253952 B，CANN 9.1.0 `Ascend950PR_9579.ini` `[AICoreSpec] ub_size`，md5 `2f0eca1f7ca70f62ee7d6d33990c3be1`）。两种规模（`block_rows` = 16 / 64）均满足。
- **L0 容量**: 不适用——无 `T.gemm`，不分配 L0A/L0B/L0C（§3.5.1）。
- **动态 grid**: `T.dynamic` 只支持 1-D extent，本算子只有 `tokens` 一个动态轴且只进 1-D grid（§3.5.1）。
- **线程域元素上限**: `block_rows × cols = 2048` 元素 / `threads = 128` ⇒ 16 元素/线程；外部仓 `test_tilelang_ascend_simtvf_cast.py` 覆盖 `n = threads × 16 = 2048`（设备验证），本设计正好处在该已验证点（§9 风险 R7）。

### 5.4 注意事项

**列维（编译期归一化，消去运行期边界）**：

```python
cols = block_cols if (row_width % block_cols == 0 and block_cols <= row_width) else row_width
nchunks = (row_width + cols - 1) // cols        # row_width ∈ {128,512,1024} 时为 1 / 4 / 8
```

- 当 `row_width` 不被 `block_cols` 整除时（例如未来出现 `row_width = 96`），`cols` 退化为 `row_width` 本身、`nchunks = 1`，`col < row_width` 在编译期可证，**不需要运行期列 guard**（§1.6.2 #8）。
- 代价：`row_width` 非 128 整数倍时行段不满一个事务（`96 × 2 = 192 B`），属形状本身的限制，不是分块引入的浪费。

**行维（唯一的运行期尾块）**：

- `nblocks = ceildiv(tokens, block_rows) · nchunks` 由动态 `tokens` 决定；最后一行块可能不满 16 行，块内用 lane 谓词 `if token < tokens` 保护（§1.6.2 #7）。
- `odd-tokens`（7 token、`row_width = 128`）：1 个逻辑块、尾块浪费 9/16 行（56%），绝对代价 0.01 µs（§1.6.0 D-2 表），不值得为它缩小 `block_rows`。

**稀疏写（负 slot）**：

- `slot == -1` 的行在 lane 级被谓词化跳过（主选路径）或块内行级 guard 跳过（B 路径）；**不能**改写成「写入时选择旧值」的数值 select——那需要读 `Pages[-1, col]`，属越界地址（§1.6.2 #6）。

### 5.5 分核策略（物理核数适配）⭐

> 三要素（① 核数计算 / ② 物理核数实查依据 / ③ 规模判定与分核方案）按 `.agents/skills/_shared/standards/core-split-strategy.md` §1 组织；物理核数按任务平台 `ascend-950pr-9579` 的记录取值。

- **物理核数**: **56 个 Vector 核（AIV）**。实查记录：`python3 .agents/tools/statectl.py platform --task-dir examples/paged_cache_write/_paged_cache_write_kernel` 返回 `hardware.observed_cores = {"cube": 28, "vector": 56}`，`hardware.hardware_id = "ascend-950pr-9579"`（架构 `dav-3510`）；其 `core_source` 为 `Ascend950PR_9579.ini [SoCInfo] ai_core_cnt=28, cube_core_cnt=28, vector_core_cnt=56` 加运行期探针 `rtGetAiCoreCount()=28`（读取日期 2026-11-23）。本算子无 Cube 计算、全部落在 Vector ⇒ **按 56 个 AIV 分核**（28 个 AIC 不参与）。
- **逻辑核数**: 4096（`num_logical_kernels = ceil(M / block_M) × ceil(N / block_N) = ceil(8192 / 16) × ceil(1024 / 128) = 512 × 8`，代表规模 llama-8b-prefill）。其余 workload：decode 64 token/1024 宽 = 4×8 = 32；qwen3-235b-fp8 4096 token/512 宽 = 256×4 = 1024；gpt-oss-120b 256 token/512 宽 = 16×4 = 64；odd-tokens 7 token/128 宽 = 1×1 = 1。
- **规模判定**:
  - **小规模（逻辑核数 ≤ 物理核数）**：`odd-tokens`（1 块）、`decode`（32 块 < 56）⇒ **无需适配**——每块独占一个 AIV，硬件不做分波，尾部空核由硬件自然闲置（绝对时间 0.4 µs 级，属启动地板）。
  - **中等规模（物理核数的个位数倍）**：`gpt-oss-120b`（64 块 = 56 × 1.14）、`qwen3-235b-fp8-fp16`（256 块 = 56 × 4.57）、`qwen3-235b-fp8`（1024 块 = 56 × 18.29）⇒ **整数倍邻域内的分波调度**，无需调整 block。
  - **极大规模（≫ 物理核数）**：`llama-8b-prefill`（4096 块 = 56 × 73.14）⇒ 走**硬件分波调度**（逻辑块超发，硬件按 56 个 AIV 逐波排空），**不做 `T.Persistent` 核内串行**。依据：单块工作量只有 2048 元素（约 6 条向量 ALU 指令 / 块），核内串行会把 73 个块串到一个核上并引入每迭代的地址解算开销，且失去硬件对不规则耗时的自然均衡；本算子无 UB 驻留压力（主选路径 0 B），没有「必须复用片上缓冲」的理由（§6.3）。
- **分核方案（负载均衡量化）**：4096 = 56 × 73 + 8 ⇒ 前 73 波满载、末波仅 8 个 AIV 工作，尾部浪费 `(56 − 8) / 4096 = 1.17%`，即约 `46.0 µs × 1.17% ≈ 0.54 µs`（§1.6.0 D-2 的 46.0 µs 已按 73.14 波计）。核数邻域核算：`block_rows = 32`（逻辑块 2048 = 56 × 36 + 32）的相对尾部浪费也是 `(56 − 32)/2048 = 1.17%`，与 16 持平；`block_rows = 64`（1024 = 56 × 18 + 16）退化为 `(56 − 16)/1024 = 3.9%`。故**维持 `block_rows = 16`**（更大的行块只会增加单块指令数与 B 路径 UB 占用，不改善均衡）；该结论属可复算的规模判定，不依赖实测。
- **不做跨核协同**：`ascend.batch_schedule` 的跨 block barrier 在本工具链未认证（§3.5.1），本算子也无需跨块同步——每个 `(token, col)` 元素的写目标互不重叠（调用方保证非负 slot 互不相同），块之间没有依赖。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| 逻辑块（token 行块 × 列段） | block 级并行 | `T.Kernel(nblocks) as bx` | 1-D core grid；`bx` 经 `rb = bx // nchunks`、`cb = bx % nchunks` 折叠成 (行块, 列段)（§0.6 R1） |
| 线程 / lane | 线程域声明 | `T.SimtVF(threads=128)` | 打开 128 lane 的真实线程域（kernel 级不接受 `threads`） |
| 元素级（`block_rows × cols` = 2048） | 二维并行（向量化） | `for i, c in T.Parallel(block_rows, cols)` | 2048 元素 / 128 lane = 16 元素/线程，内层 `c` 为连续轴（§1.6.3 选定轴） |
| 列段方向（`nchunks`） | 无显式循环 | 折叠进 `T.Kernel` 的 `bx` | `nchunks ∈ {1,4,8}` 是编译期常量，展开为独立逻辑块，不做块内串行 |
| 行方向（token） | 无显式循环 | `T.Parallel` 第 0 维 + `if token < tokens` 谓词 | 行维已在块内并行；尾块由 lane 谓词覆盖（`tokens` 动态） |
| `T.serial` | **不使用**（仅 B 备选路径行级 16 次 `T.serial`） | — | 主选路径无顺序依赖，行维并行度更高（§1.6.2 #12） |
| `T.Pipelined` | **不使用** | — | 每块只有一个 tile、无跨 tile 复用，无 stage 可叠（§6.3） |
| `T.Persistent` | **不使用** | — | 逻辑块由硬件分波排空（§5.5），无需核内常驻循环 |

### 6.2 循环伪代码

```python
# 主选路径 P：1-D 网格 + 线程域 + 二维并行（与 §3.3 一致）
with T.Kernel(nblocks) as bx, T.SimtVF(threads=128):
    rb = bx // nchunks                     # 行块号（编译期常量 nchunks）
    cb = bx % nchunks                      # 列段号
    for i, c in T.Parallel(block_rows, cols):   # 2048 元素 → 128 lane × 16 元素
        token = rb * block_rows + i        # 行坐标（动态 tokens 的上界内）
        col = cb * cols + c                # 列坐标（< row_width，编译期可证）
        if token < tokens:                 # 唯一的运行期边界谓词（尾块）
            slot = T.cast(Slots[token], "int32")
            if slot >= 0:                  # 稀疏写谓词
                <fp8 或 16-bit 算式>        # 见 §3.3；直接 GM 写 Pages[slot, col]
```

### 6.3 流水线优化

**不使用 `T.Pipelined`**，理由三条：

1. **无 stage 可叠**：流水线的前提是同一 block 内「搬入 tile → 计算 tile → 搬出 tile」可重叠。主选路径没有 UB 中转（无搬入/搬出），计算完的元素直接写 GM，块内只有一次数据相变。
2. **无跨 tile 复用**：每块只处理一个 tile，不存在「下一个 tile 预取」的对象；B 备选路径的 16 行串行 `T.copy` 之间也没有复用关系（每行独立读写），叠加 stage 只会增加 UB 版本数。
3. **自动调度已开启**：`ascend/pipeline.py` 的 `allow_autoschedule` 默认开启，会执行 AutoSchedule / InsertSync / MaterializeMultiBuffer（见 `.agents/platforms/toolchains/ascend/references/programming.md` §同步与缓冲版本）；主选路径无缓冲可多版本化，这些 pass 不产生额外结构。

**Stage 4 显式性能杠杆**（本设计只登记，不在 Stage 1 采纳，均需真机测量）：

| 杠杆 | 改动 | 预期方向 | 依据 / 风险 |
|------|------|---------|------------|
| `threads` 128 → 256 | `T.SimtVF(threads=256)`，`block_rows × cols` 不变 ⇒ 8 元素/线程 | fp8 链的指令条数按元素/线程数线性下降，发射项约减半（若 54.9 µs 发射项主导，估算 ≈ 27.5 µs，**未实测假设**，按 CONST-vector-launch-overhead 的线性口径外推） | 需重测 2048 元素 / 256 lane 的组合（§9 R7） |
| `block_rows` 16 → 32 | 逻辑块数减半（4096 → 2048），单块 4096 元素 | 分波尾部浪费由 1.17% 降至约 0.5%；每块指令数翻倍、UB 占用（B 路径）翻倍 | §5.5 分核量化 |
| 走 B 备选路径 | UB 中转 + 行级 `T.copy` | 若 GM 直写（尤其 fp8 单字节写）效率显著低于 DMA 成段写，可换回 | §9 R1/R2 的降级触发条件 |

### 6.4 尾块处理

| 尾块来源 | 是否可能 | 处理方式 | 代价 |
|---------|---------|---------|------|
| 行维（`tokens % block_rows ≠ 0`） | **是**（唯一真实尾块；`tokens` 动态） | `nblocks = ceildiv(tokens, block_rows)·nchunks` 覆盖最后一个不满块；块内 lane 谓词 `if token < tokens` 阻止越界读 `K/V/Slots` 与越界写 | 尾块最多浪费 `block_rows − 1 = 15` 行，且只影响 1 个逻辑块（4096 块中的 1 块）；`odd-tokens` 用例浪费 9/16 行但绝对代价 0.01 µs |
| 列维（`row_width % block_cols ≠ 0`） | **否**（由 §5.4 归一化消除） | `cols` 取 `row_width` 或 `block_cols`，使 `nchunks·cols = row_width` 精确成立 | 0（无运行期列判断） |
| 稀疏（`slot < 0`） | 是（契约内合法输入） | 主选：lane 谓词 `if slot >= 0`；B 路径：块内行级 guard | 被跳过的行不产生读写流量（语义要求） |
| `tokens == 0` | 是 | host 侧 `if k.shape[0] == 0: return`，**不发起 kernel**（wrapper 已固化） | 0（不 launch） |
| `tokens` 非 `block_rows` 倍数且 < `block_rows` | 是（`odd-tokens` 7 行） | 同上 lane 谓词；`nblocks = 1·nchunks` | 见 §5.5 小规模档 |
---

## 7. 同步策略

### 7.1 同步模式

**模式**: 自动同步（`allow_autoschedule` 默认开启，由 `InsertSync` 自动插入所需事件）

### 7.2 同步点说明

| 位置 | 同步 API | 理由 |
|------|----------|------|
| 主选路径 P：全 kernel | 无显式同步 | 无 `T.copy`、无 `T.alloc_shared`、无跨 lane 通信；`K/V/Slots/Scale` 的读与 `Pages` 的写互不依赖（不同张量、不同地址域），不存在「先写后读」或「先读后写」同一地址的竞争 |
| 主选路径 P：块间 | 无（不使用 `ascend_cross_core_set_flag/wait_flag`） | 每个 `(token, col)` 的写目标互不重叠（调用方保证非负 slot 互不相同），块之间无数据依赖；跨核 flag 在本工具链未认证（`.agents/platforms/toolchains/ascend/references/programming.md` §同步与缓冲版本） |
| B 备选路径：`T.copy(GM→UB)` 之后、`T.Parallel` 读 UB 之前 | 自动（`InsertSync`） | 需等待 MTE2 搬入完成，`T.Parallel` 才能读到有效 UB 数据 |
| B 备选路径：`T.Parallel` 写 `k8_ub/v8_ub` 之后、`T.copy(UB→GM)` 之前 | 自动（`InsertSync`） | 需等待向量写完成，DMA 才能搬出 |
| B 备选路径：下一行迭代覆盖同一 UB 行之前 | 自动（同 buffer 版本约束） | 16 行串行复用同一 UB 区域，自动调度插入反依赖等待 |

结论：**本设计不手写任何同步点**。`ascend_pipe_barrier` / `ascend_set_flag` / `ascend_wait_flag` / `ascend_get_buf` 等显式接口仅在 `PL-950-manual-sync`（单核单 tile 管线）场景有已执行的配对记录，本算子无手工流水需求，故不使用（避免引入未认证的跨核 flag 使用模板）。

### 7.3 pass_configs 配置

```python
pass_configs = {}   # 不额外配置：默认流水线已包含 AutoSchedule / InsertSync / MaterializeMultiBuffer
```

说明：

1. 构建走 `tilelang.compile(main, target="ascend", out_idx=[])`（与平台模板 `.agents/platforms/toolchains/ascend/kernel_template.py` 一致），不传自定义 pass 配置。
2. 主选路径无 UB/alloca 缓冲，`MaterializeMultiBuffer` 与 `annotate_buffer_versions` 无作用对象；不调用 `annotate_unlimit_memory`（它会解除自动调度的容量约束，本算子不需要）。
3. B 备选路径若在 Stage 3 出现多缓冲膨胀，可在该路径上显式 `annotate_buffer_versions({k_ub: 1, v_ub: 1})` 固定单版本（接口与语义见 `programming.md` §同步与缓冲版本：`{buf: 1}` 取消该 storage 的自动多缓冲资格）；主选路径不受影响。

---

## 8. 验证方案

> **设计期未做设备验证**：本节只给验证计划与 golden 草案，实际执行在 Stage 3。

### 8.1 Golden 函数

golden 以 **§0.1 的源算子语义**为唯一依据，直接移植源仓参考实现（`workloads/attention/paged_cache_write.py` 的 `ref_program`），**不复刻** §0.6 的 NPU 算法（不使用 1-D 网格、不使用 `SimtVF`、不做 int32 窄化、不依赖 lane 谓词），保证验证独立性。

```python
import torch


def golden_paged_cache_write(k, v, k_cache, v_cache, slot_mapping,
                             k_scale=None, v_scale=None):
    """§0.1 语义的独立参考：按 slot 散射写入，负 slot 跳过，未命名行不变。

    移植自源仓 ref_program：先 clone 再写（参考实现运行在共享输入上，
    若直接写会毁掉对比基准）；FP8 用 clamp 显式表达饱和（torch 的部分
    FP8 cast 会把溢出变成 NaN，clamp 保持 NaN）。
    """
    k_out, v_out = k_cache.clone(), v_cache.clone()
    written = (slot_mapping >= 0).nonzero().squeeze(1)      # 写使能（负 slot 跳过）
    rows = slot_mapping[written]                            # 目标行号
    for pages, new, scale in ((k_out, k, k_scale), (v_out, v, v_scale)):
        flat = pages.view(-1, *pages.shape[2:])             # (physical_tokens, row_width)
        stored = new[written]                               # (written, heads_kv, dim)
        if scale is not None:
            limit = torch.finfo(pages.dtype).max            # float8_e4m3fn -> 448
            stored = (stored.float() / scale[0]).clamp(-limit, limit).to(pages.dtype)
        flat[rows] = stored                                 # 原地散射
    return k_out, v_out
```

关键性质：`slot == -1` 的行既不读 `k/v` 也不写池；未被命名的行保持调用前的字节；`flat[rows] = stored` 是 PyTorch 的高级索引赋值（重复索引时最后写入者胜出，但调用方保证非负 slot 互不相同，故与本 kernel 的元素级映射一一对应）。

### 8.2 精度标准

| dtype | atol | rtol | 说明 |
|-------|------|------|------|
| `float16` | 0 | 0 | 16-bit cache 为逐位拷贝；断言 `equal_nan=True` |
| `bfloat16` | 0 | 0 | 同上 |
| `float8_e4m3fn` | 0 | 0 | fp8 cache 的每个字节都要与 golden 相同（饱和、舍入、NaN 保持全部逐位比对） |

来源：算子契约 `Exact(rtol=0, atol=0)`（`src/tileops/manifest/spec/attention.yaml` 的 `PagedKVCacheWriteFwdOp`，`workloads/attention/paged_cache_write.py` 的 `verification()`）。**任何**容差都不被接受：被跳过的行也必须字节不变，溢出/NaN 必须精确落在同一 bit pattern 上。

### 8.3 L0 / L1 / boundary 测试计划

**L0（冒烟门槛，逐字节比对两个 pool）**：

| # | 用例 | 输入（tokens, heads_kv, dim, page_size, num_pages, dtype→cache_dtype, slots） | 覆盖点 |
|---|------|------|--------|
| L0-1 | `fp16-odd-page-skip-and-page-edges` | 5, 2, 64, 7, 3, fp16→fp16, `[20, -1, 0, 6, 13]` | `row_width = 128`（1 段）；负 slot 跳过；页首/页尾行；slot 乱序 |
| L0-2 | `bf16-one-kv-head-last-row` | 4, 1, 32, 4, 2, bf16→bf16, `[7, 0, 3, 4]` | `row_width = 32 < block_cols` ⇒ `cols` 归一化为 32；写最后一个物理行 |
| L0-3 | `fp8-bf16-scaled` | 3, 2, 64, 16, 2, bf16→fp8, `[1, -1, 31]` | bf16 输入 + fp8 池；`k_scale=1.25 / v_scale=0.75` 不同除数；负 slot |
| L0-4 | `fp8-fp16-page-boundary` | 2, 4, 128, 16, 4, fp16→fp8, `[63, 0]` | `row_width = 512` ⇒ `nchunks = 4`；页边界行 63 |
| L0-5 | `saturates_fp8_overflow` | 1, 1, 16, 2, 1, fp16→fp8, `[1]`，16 个手工值 `1000, -1000, 448, -448, 1.2, -1.2, 1.0625, 1.1875, nan, inf, -inf, 0, -0.0, 1, -1, 0.5`，`scale = 1` | 饱和（±1000→±448）、NaN 保持、±inf、ties（`1.0625/1.1875`）、±0 符号；`row_width = 16` 的极窄行 |
| L0-6 | `skips_every_token` | 3, 2, 64, 7, 4, bf16, `[-1, -1, -1]`；pool 预填 `3.0 / -2.0` | 全负 slot：两个 pool 逐字节不变（用哨兵值检测误写） |
| L0-7 | `of_no_tokens_writes_nothing` | 0, 2, 64, 16, 2, bf16 | `tokens == 0` 不发起 kernel，pool 不变 |
| L0-8 | `refuses_an_invalid_tensor_contract` | 4 个变体（scale 缺失 / scale 多余 / 输入 dtype 非 16-bit / cache dtype 非 fp8 也非输入 dtype） | host 契约校验抛 `ValueError`（不进 kernel） |
| L0-9 | `dispatches_to_its_kernel` | 3, 2, 64, 7, 2, fp16, `[1, 0, 6]` | 首次调用构建 `PagedKVCacheWriteKernel` |
| L0-10 | `eval_roofline` | 4, 2, 64, 4, 2, bf16→(bf16 / fp8), `[7, 0, -1, 3]` | 只统计有 slot 的 token：fp8 时 `flops = 4·written·row`，`bytes = 2·written·row·(2 + cache_itemsize) + tokens·8 (+8 fp8)` |

**L1（完整功能，`full` 标记）**：`bf16-decode-step-distinct-slots`（64, 8, 128, 64, 8, bf16, slots 随机不重复）与 `fp16-prefill-sweep`（4096, 8, 128, 16, 256, fp16）。

**boundary（边界补充，本设计新增的针对性用例）**：

| # | 用例 | 目的 |
|---|------|------|
| B-1 | `tokens = 17`, `block_rows = 16` ⇒ 2 个逻辑块、尾块 1 行 | 验证 `if token < tokens` 谓词（唯一运行期尾块） |
| B-2 | `row_width = 16`（`heads_kv=1, dim=16`）与 `row_width = 96`（非 128 整数倍） | 验证 §5.4 的 `cols` 归一化分支（`cols = row_width`、`nchunks = 1`） |
| B-3 | `tokens = 1` 单 token 单行 | 最小规模：1 个逻辑块、1 行有效 |
| B-4 | `slot_mapping` 含 `physical_tokens - 1`（池最后一行） | 地址上界（int32 索引不越界） |
| B-5 | fp8 且 `k_scale = 7.0`（非 2 的幂） | 验证未采用倒数乘法（X1 在该 scale 下失配，§11）；除法路径必须精确 |
| B-6 | fp8 且输入含 `nan` 且 `scale = 1.25` | 验证 `T.clamp` 的 NaN 传播（§9 R3）；若失败则启用 `T.if_then_else(d == d, ...)` 兜底 |

**分层执行**：L0 用 `--level L0`（Stage 3 的 `precision_command`：`<python> <kernel> --level L0`），L1 用 `--level L1`，边界用 `--level boundary`；全部为逐字节 `assert_close(atol=0, rtol=0, equal_nan=True)`。

---

## 9. 风险点与注意事项

### 9.1 已知约束

| # | 约束 | 触发条件 | 影响 | 缓解 / 降级 |
|---|------|---------|------|------------|
| R1 | **lane 级谓词化 GM 直写未设备验证** | 主选路径的 `if slot >= 0:` 包裹 GM 写 | 若 codegen 把谓词写退化为标量分支或错误地无条件写，会出现「负 slot 行被写」或性能坍塌 | 现有证据仅 `test_simtvf_vectorized_select.py` 的 lane 级 `if` 包裹 **UB** 写（codegen 断言）；降级路径 B（§3.3 备选：行级 guard + `T.copy`），切换后负 slot 在块内行级跳过，语义相同；**切换阈值：Stage 3 精度失败即 T3，性能数据按 §1.6.3 T1/T2 判定（§11.4 同口径）** |
| R2 | **fp8 目标 dtype 的 GM 直写未取证** | fp8 路径 `KPages[slot, col] = T.cast(d, cache_dtype)` | 单字节写可能不成段或需要中间 UB | 先按主选路径实现并在 Stage 3 编译/精度验证；失败则走 B 路径（UB→GM 行级 DMA，128 B/行段）。**切换阈值：编译/精度失败即 T3，写事务碎片率按 §1.6.3 T2（`ratio_tx ≥ 2`）判定** |
| R3 | **`vmaxs/vmins` 的 NaN 传播未文档化** | 输入含 NaN 的 fp8 用例 | `T.clamp` 可能把 NaN 变成 ±448（clamp docstring：*NaN behavior follows the target's min/max operations without additional checks*） | L0-5 / B-6 直接覆盖；失败则启用 `T.if_then_else(d == d, T.clamp(d, -448.0, 448.0), d)`（§3.2 步骤 7'），语义不变 |
| R4 | **fp32 除法 vs 倒数乘法** | 有人为省 1 条指令改用 `x * (1/s)` | 非 2 的幂 scale 下出现 1 ulp 级差异（X1：`scale=7.0` 时 60/65536 失配），`Exact(0,0)` 直接失败 | 设计明确保留 `vdiv`（§1.6.1 O1）；B-5 用例把该风险显性化 |
| R5 | **int64→int32 slot 窄化依赖编译期 guard** | `physical_tokens ≥ 2**31` | 窄化后行号回绕，写到错误行 | O3 的编译期 guard：`physical_tokens < 2**31` 时用 int32，否则保持 int64（`verify_equiv.py` 的 guard 项给出 3/4 反例）；本任务全部 workload 的 `physical_tokens ≤ 4096`，走 int32 分支 |
| R6 | **数据相关行地址**（`Pages[slot, col]`） | 所有路径 | 行地址由运行期 slot 决定，编译器无法证明对齐/连续性 ⇒ 不能做段合并优化 | 已由 §1.6.3 布局分析接受（行内 stride 1 保证段内连续）；无替代 API（`T.simd.vscatter` 索引宽度 uint16 不足，§11） |
| R7 | **`block_rows × cols = 2048` / `threads = 128` 为 2–4× 外推** | 默认 tile | 若该组合超出已验证的向量化 envelope，可能编译失败或降级为标量 | 现有设备证据为 512–1024 元素（2-D `T.Parallel`）与 `n = threads×16` 的 cast 用例；Stage 3 首个编译即验证；失败时把 `block_cols` 降到 64（1024 元素）或将 `block_rows` 降到 8 |
| R8 | **`ascend.batch_schedule` 跨 block barrier 未认证** | 若有人尝试跨块合并/持久化 | 死锁或未定义行为 | 本设计**不使用**任何跨 block 同步与 persistent 结构（§5.5、§7.2） |
| R9 | **950PR 常数由 910B2C 外推** | D-2 估算 | 绝对时延可能偏差 1.5–2×；不影响「#0 基线比 #1 慢约 2 倍」的量级结论 | 所有估算标注「未实测假设」（§1.6.0 估算口径）；Stage 4 用真机 msprof 校准 |
| R10 | **cast 是否与 `vld/vsts` 折叠未知** | fp8 链 | 若 `vcvt` 不能折叠，每块 op 数由 6 升到 10，发射项 ×1.67（54.9 → 91.5 µs） | 发射项在 §1.6.0 已按 6 op 计并注明该敏感性；Stage 4 以实测指令数为准 |

补充口径说明（供 Stage 3 对齐）：算子 `eval_roofline` 把 fp8 记为 `4 FLOPs / written 元素对`（`div + saturate` 各 1，k/v 两路合 4），本设计的发射估算按 **6 条向量指令/块**（`vdiv + vmaxs + vmins` × 2 路）计——两者是不同口径（浮点运算计数 vs 向量指令条数），不矛盾，但 Stage 3 不得用前者反推后者。

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| 把 `threads` 传给 `T.Kernel` | 沿用 GPU 源码写法 `T.Kernel(..., threads=128)` | Python 层直接报错（ascend 方言拒绝该参数） | 用 `T.SimtVF(threads=128)` 打开线程域（§3.2 步骤 1'/§3.4） |
| 用 2-D grid | `T.Kernel(row_tiles, col_tiles)` | 编译失败（只支持 1-D） | `bx // nchunks`、`bx % nchunks` 折叠（§0.6 R1） |
| `T.alloc_shared` 当 SMEM 用 | 以为它和 CUDA 的 shared 一样 | 实际是 UB，容量 248 KB，且引入搬运 | 主选路径不分配；B 路径按 §4.5 预算 |
| 依赖隐式 FP8 cast 的饱和 | 省略 `T.clamp` | NaN/溢出行为不等价（X3：bf16 4/16、fp16 2/16 失配） | 显式 `T.clamp(d, -448, 448)`（§0.6 R2） |
| 用 `Pages[slot - 1, col]` 之类偏移做 select | 想把负 slot 变成数值选择 | 越界读、语义错误 | 用谓词（`if slot >= 0`）或 B 路径的行级 guard |
| 把 `slot` 留在 int64 | 直接用 `Slots[token]` | 64 位地址运算、额外寄存器压力 | `T.cast(Slots[token], "int32")` + 编译期 guard（O3） |
| 在 `T.Parallel` 外读 `KScale[0]` | 试图放到 host 侧 | kernel 参数是设备张量，host 侧不可读 | 在 kernel body 顶部 `ks = KScale[0]`（O1） |

### 9.3 特殊场景处理

| 场景 | 处理 |
|------|------|
| `tokens == 0` | wrapper 的 `if k.shape[0] == 0: return` 提前返回，**不构建/不启动 kernel**；L0-7 断言 pool 不变 |
| `tokens` 非 `block_rows` 整数倍 | 尾块由 `if token < tokens` 谓词保护（§6.4） |
| `row_width` 非 `block_cols` 整数倍（如 96） | `cols` 归一化为 `row_width`，`nchunks = 1`，无运行期列 guard（§5.4） |
| 全部 slot 为 `-1` | 所有块都不写；仍会启动 kernel（`nblocks > 0`），代价为纯启动开销；语义由 L0-6 覆盖 |
| `physical_tokens` 很小（如 2 行）而 slot 指向最后一行 | 行地址上界由调用方契约保证；B-4 覆盖 `physical_tokens - 1` |
| fp8 的 `k_scale` 与 `v_scale` 不同 | 两条独立的标量 hoist（`ks`、`vs`），不得互相复用；L0-3 覆盖（1.25 / 0.75） |
| `±inf`、NaN、ties | 显式 clamp + cast；L0-5 覆盖 16 个角点值 |
| 同一 pool 行被两个 token 命名 | 契约禁止（调用方保证非负 slot 互不相同）；kernel 不做检查，与源实现一致 |

---

## 10. 交付清单

### 10.1 目录结构

```
examples/paged_cache_write/_paged_cache_write_kernel/
├── DESIGN.md            # 本设计文档（Stage 1）
├── verify_equiv.py      # §1.6.1 的等价性机器验证脚本（torch CPU，可重复执行）
└── RETROSPECTIVE.md     # Stage 1 复盘（每次产出 DESIGN.md 后追加）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | 已完成 | 本设计文档（§0–§11） |
| `verify_equiv.py` | 已完成 | 8 个采纳项 + 3 个否决项的数值对照，输出 `ADOPTED_VERDICT: ALL_PASS` |
| `RETROSPECTIVE.md` | 已完成 | 本阶段复盘章节（Phase 8） |
| TileOps 侧 kernel 实现（`examples/TileOPs/tileops/kernels/attention/paged_cache_write/paged_cache_write.py`） | 待 Stage 3 实现 | Stage 0 已生成冻结骨架（接口 `_paged_cache_write_kernel(heads_kv, dim, physical_tokens, in_dtype, cache_dtype)` + `default_config`），Stage 3 按 §3.3 主选路径 P 填充 kernel 本体 |
| 测试（L0/L1/boundary） | 待 Stage 3 实现 | 按 §8.3 的用例表；TileOps 侧已有 `examples/TileOPs/tests/ops/test_paged_kv_cache_write.py` 作为基线 |

### 10.3 命名规范

- 算子目录：`_paged_cache_write_kernel`（Stage 0 已定，对应 kernel factory 名）
- 实现文件：`paged_cache_write.py`（TileOps 布局）
- 设计文档：`DESIGN.md`；复盘：`RETROSPECTIVE.md`；等价性脚本：`verify_equiv.py`

### 10.4 实现顺序

1. 已完成：设计文档（本文）
2. 已完成：等价性机器验证脚本与结论
3. 待 Stage 3：golden 落地（§8.1）+ L0 用例（§8.3）→ kernel 实现（§3.3 P）→ 分层精度验证
4. Stage 4：按 §6.3 的性能杠杆与 §9.1 的 R1/R2 触发条件、以 §1.6.3 (a)–(e) 的实验裁决计划（阈值 T1–T3）决定是否切 B 路径

---

## 11. 附录：被否决优化的数值证据与备选方案细节

### 11.1 被否决的数学化简（`verify_equiv.py` 的 `rejected` 项）

**X1 除法 → 乘倒数**（`d = cast_f32(x) * (1.0/s)`）：在 `s` 为 2 的幂时逐位相同，但非 2 的幂时出现 1 ulp 级差异——`scale = 7.0`：60/65536 失配（9.155e-04）；`scale = 1.7`：30/65536（4.578e-04）；`scale = 0.3`：2/65536（3.052e-05）；`scale ∈ {3.0, 0.1, 1/3, 1.25, 0.75}` 与 2.10 M 随机抽样：0 失配。结论：**否决**——`scale` 是运行期输入（服务端取值不受控），不能只对「已测过的 scale」保证等价；`Exact(0,0)` 要求全域等价。

**X2 16-bit 域缩放**（`cast_f16/bf16(x / s)` 在输入 dtype 内完成除法）：bf16 下 127/65536 失配（1.938e-03），fp16 下 0 失配。结论：**否决**——bf16 只有 8 位尾数，除法在 bf16 域内舍入两次（除法 + 缩放）导致与「fp32 除法后一次 cast」不同；fp16 偶然通过不构成等价性证明。

**X3 只靠 cast 饱和（省略显式 clamp）**：在 16 个角点值上 bf16 失配 4/16（2.500e-01）、fp16 失配 2/16（1.250e-01）。结论：**否决**——Ascend 的 fp8 cast 不保证「有限溢出 → ±448」与「NaN → NaN」，显式 `T.clamp` 是**语义必需**（§0.6 R2），不是可选优化；因此它不计入 §1.6.1 的收益比较，而是重设计的一部分。

### 11.2 数据相关散射写：`T.simd.vscatter` 不可用的依据

外部仓 SIMD 测试集中 `T.simd.vscatter` 的索引操作数为 **uint16** 类型（上限 65535 个元素地址）。本算子的池规模为 `physical_tokens × row_width`：qwen3-235b-fp8 为 4096 × 512 = 2.1 M、llama-8b-prefill 为 4096 × 1024 = 4.2 M、工作负载声明的池行数上限（`num_pages × page_size`）达 67 M 元素量级 ⇒ 索引宽度不匹配，`vscatter` 无法表达「写到第 slot 行」（§1.6.3 候选 #3）。替代方式即主选路径的「lane 级标量地址 + `T.Parallel` 内直接 GM 写」。

### 11.3 SIMD 寄存器路径（Stage 4 备选）的细节与证据缺口

若 SIMT 路径的发射开销成为瓶颈（fp8 workload 的估算 54.9 µs 中流量只占 8.65 µs），可在 B 路径的 UB 中转之上改用显式 SIMD 指令：

```
T.simd.vld(k_ub 行) → T.simd.vcvt(f32) → T.simd.vdiv(ks) → T.simd.vmaxs(-448.0)
→ T.simd.vmins(448.0) → T.simd.vcvt(fp8) → T.simd.vsts(k8_ub 行) → T.copy(UB→GM 行)
```

证据状态：`T.simd.vcvt/vdiv/vmaxs/vmins` 的存在性有源码签名；但 **fp8 作为目标 dtype 的 `vcvt` + `vsts` 组合在外部仓 SIMD 测试集中未取证**（这是「未查到」而非「不存在」），且 SIMD 只能操作 UB ⇒ 必须叠加 §4.5 的 12.0 KB 中转。启用前须先补齐该组合的编译与数值验证；该备选不改变数学语义（同算式、同 dtype 转换）。

### 11.4 B 备选路径（UB 中转行级 DMA）的代价量化

| 项 | 主选路径 P | 备选路径 B |
|----|-----------|-----------|
| 网格 / 块映射 | `nblocks = ceildiv(tokens, R)·nchunks`，`rb = bx//nchunks`，`cb = bx%nchunks` | **同构**（同一 `nblocks`/`rb`/`cb`/`cols`，§3.3） |
| GM 数据流量（llama-8b-prefill） | 67.63 MB | 67.63 MB（UB 往返不计 GM 流量） |
| 每块额外交代（`T.copy`） | 0 | **64 条/块**（K/V 读 + K/V 写 各 16 行；store 侧 32 条） |
| 总 `T.copy` 条数 | 0 | `4·block_rows·nblocks = 4·tokens·nchunks`（prefill 262,144；qwen3-235b-fp8 65,536） |
| 列覆盖 | 全部列段（`c0 = cb·cols`） | 全部列段（同一 `c0` 偏移，含末段；§3.3 覆盖性论证） |
| UB 占用 | 0 B | fp8 变体 12.0 KB / 16-bit 变体 8.0 KB（§4.5） |
| 块内结构 | 1 次数据相变（读→写） | 5 跳（load → 算 → store，行级循环，`T.serial`） |
| 负 slot 处理 | lane 谓词（R1 风险） | 行级 guard（无谓词写风险） |
| 列向量化轴 | `col`（同） | `col`（同，§1.6.3 不变） |

**切换判据（Stage 4；与 §1.6.3 的实验裁决计划同一口径，此处只复述编号）**：对 §1.6.3 (a) 的代表 shape，按 (b) 的指标与反解式采集 `L_P`/`L_B`/`write_transactions`，再按 (c) 的阈值判定——**T3（语义门）优先**：§9 R1/R2 在 Stage 3 精度验证中失败即直接切 B；**T1**：`L_P ≥ 1.15 × L_B` ⇒ 该 shape 切 B；**T2**：fp8 shape 的 `ratio_tx ≥ 2` ⇒ 该 fp8 组合切 B；均未命中则维持 P。结果可按 §1.6.3 (d) 的 `cache_dtype` 分派只对 fp8 组合生效（不增编译产物），并按 (e) 以 `[DESIGN_LIMIT]` 附录补记回写 `perf_opt/perf_feedback.md` 后更新本节与 §1.6.3 的阈值。切换只改搬运路径，不改任何算式与索引语义。

### 11.5 调研来源清单（可追溯）

| 来源 | 用途 |
|------|------|
| `.agents/skills/tilelang-op-design/references/algorithm-research.md` | 调研四问的执行标准与 elementwise/scatter 族参考表 |
| `.agents/skills/tilelang-op-optimize/references/pattern-library/constants.md` | `CONST-vector-launch-overhead`（0.5 µs/op）、`CONST-mte2-degradation`（1.26 TB/s @48 AIV）、`CONST-capacity-910B2C` |
| `.agents/skills/_shared/standards/hardware-cost-model.md` | D-2 估算行格式与常数引用纪律 |
| `.agents/skills/_shared/standards/core-split-strategy.md` | §5.5 三要素（逻辑核数 / 物理核数实查 / 规模判定） |
| `statectl platform --task-dir examples/paged_cache_write/_paged_cache_write_kernel` | `ascend-950pr-9579` 的容量预算、`observed_cores`（cube 28 / vector 56）、`core_source`、`execution` 上下文 |
| 外部仓 `/home/l00970450/zuo/tilelang-fork` 的 `tilelang/ascend/language/{kernel,frame,allocate,math}.py` | `T.Kernel`（1-D、无 `threads`）、`T.SimtVF(threads=128, latency=0)`、`T.alloc_shared(shape, dtype, scope)`、`T.dynamic(name, dtype="int32")` 的签名与 docstring |
| 外部仓 ascend 测试集（`test_dynamic_outputs.py`、`test_ascend_fractal_storage_integration.py`、`test_tilelang_ascend_clamp.py`、`test_tilelang_ascend_simtvf_cast.py`、`simtvf/test_simtvf_arithmetic.py`、`simtvf/test_simtvf_vectorized_select.py`、`test_tilelang_ascend_dma_copy.py`） | API 可用性与数值证据（详见 §3.4） |
| 本目录 `verify_equiv.py` | §1.6.1 / §11.1 的全部数值结论（torch CPU + fp64 参照） |

> 附注：本阶段未使用互联网检索（本地源已覆盖该算子族的全部候选：直写、UB 中转、SIMD 寄存器路径、host 侧索引、数学化简），故无 URL 需登记。
