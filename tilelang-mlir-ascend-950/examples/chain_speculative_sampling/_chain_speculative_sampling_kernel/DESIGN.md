# _chain_speculative_sampling_kernel 算子设计文档

> 任务：把 GPU 侧 `ChainSpeculativeSamplingFwdKernel` 的 TileLang 工厂
> `_chain_speculative_sampling_kernel` 迁移到 Ascend950PR（ascend 工具链，编程模式
> SIMT 向量执行区）。源算子路径由 conductor 给出：
> `/workspace/TileOPs/src/tileops/_kernels/sampling/chain_speculative_sampling.py`。
> 精度金标准是 `examples/TileOPs/tileops/workloads/sampling.py` 的
> `philox_uniform` / `chain_speculative_sampling`（torch 整数/浮点张量算术，设备无关）。
> 本次不做性能调优，设计目标是「语义（尤其 `num_accepted` 与 Philox 逐位）保持 + 结构可在本
> 平台编译 + 流量下界不退化」。

## 0. 源算子解读与迁移分析

方法论：`tilelang-op-design` skill 的 `references/migration-analysis.md`（三问 → 耦合性判定 →
NPU 重设计）。本章结论驱动 §1–§10；后续章节的分析对象是 §0.6 重设计后的 NPU 算法，不是源
GPU 结构的直译。

### 0.1 源算子语义（做什么）

**数学语义**（`B` 条请求，每条 `N` 个 draft token，词表 `V`）：

| 步骤 | 公式 | 说明 |
|------|------|------|
| 1 | `u[b,j] = (philox4x32_10(key=(seed_lo,seed_hi), counter=(j,b,off_lo,off_hi))[0] >> 8) · 2^-24` | `j = 0..N`，逐位由 `(seed, offset, j, b)` 决定 |
| 2 | `accept[b,j] = u[b,j] · D[b,j,id[b,j]] < T[b,j,id[b,j]]` | float32 乘 + float32 比较（严格小于） |
| 3 | `num[b] = min{ j : ¬accept[b,j] }`（全部接受则 `N`） | 接受前缀长度，不含 bonus |
| 4 | `w[b,i] = max(0, T[b,num[b],i] − (D[b,num[b],i] if num[b] < N else 0))` | 停止位置的残差行；全部接受时是 `T` 的第 `N` 行 |
| 5 | `token[b] = argmin{ k : CDF_w[k] > u[b,N]·Σw }`，跳过零权重；无解取本行最后一个正权重下标 | inverse-CDF 抽样（`CDF` 在 fp64 中左到右累加） |
| 6 | `out[b,j] = id[b,j] (j < num[b])；token[b] (j = num[b])；−1 (j > num[b])` | `[B, N+1]` int32 |
| 7 | `num_accepted[b] = num[b]` | `[B]` int32 |

**规约语义（累加顺序）**：第 5 步的 `CDF` 在参考实现里是 `weights.double().cumsum(-1)` ——
**float64 左到右顺序累加**；`point = u · cdf[-1]` 同为 fp64。第 2 步的乘法/比较是 float32。
第 4 步的残差逐元素 float32（减法后 `clamp_min(0)`）。**这三条语义强度不同**，迁移时逐条区分：

- 第 2 步（`num_accepted` 与接受前缀）：**契约要求逐位等于参考**（`tests/ops/test_sampling.py`
  用 `exact_compare(out_accepted, ref_accepted)` 机械比对），因此只能用同一算式、同 dtype、同舍入路径。
- 第 4 步（残差元素值）：逐元素 float32 运算，**逐位可复现**（无累加）。
- 第 5 步（被抽到的 token 下标）：参考实现的 fp64 顺序前缀是**实现细节**而非契约——命中位置对前缀的
  累加分组敏感。契约（`verify_chain_outputs` + `assert_chain_distribution`）要求：抽到的 token 必须
  在其停止位置的残差行上**权重为正**，且整列分布符合残差/`min(draft,target)`/接受长度。参考实现与
  测试文档同样声明「token 走同一分布但不必是参考的下标」。
- 第 1 步（uniform）：由 `(seed, offset, draw, row)` 唯一确定，**与 launch 切分无关**，必须逐位复现。

**dtype 语义**：概率 `float32`；`draft_token_ids` `int32`；`seed`/`offset` `int64`；两个输出 `int32`。
源实现在第 1 步用 `uint32` 位运算与 `__umulhi`，第 2 步用 float32，第 5 步用 float64。

**边界语义**：`N ≥ 1`；`V ≥ 1`（本任务 workload `V ∈ {64, 32000, 4099, 128256, 129280, 151936}`）；
调用方保证每个概率行有限、非负、归一，且 `id ∈ [0, V)`；零权重下标永不被抽中；`−1` 恰好出现在被抽
token 之后；`num_accepted ∈ [0, N]`。`N = 600`（fixture 的压力用例）也必须成立。

**host 侧语义**：无预处理/后处理；`forward` 只做 `.contiguous()`（元数据视图）与 shape/dtype 校验。

### 0.2 源算子输入输出

| 参数 | Shape | dtype | 说明 |
|------|-------|-------|------|
| `draft_probs` | `[B, N, V]` | float32 | draft 模型概率行 |
| `draft_token_ids` | `[B, N]` | int32 | draft token，每行取值 `[0, V)` |
| `target_probs` | `[B, N+1, V]` | float32 | target 模型概率行（多一行 bonus） |
| `seed` | `[1]` | int64 | Philox key |
| `offset` | `[1]` | int64 | Philox counter 的高两字 |
| `partial`（内部） | `[B·parts]` | float64 | 源实现的跨 block 分块和（`forward` 里 `torch.empty`），**非公开输出** |
| **输出** `output_token_ids` | **`[B, N+1]`** | int32 | 接受前缀 + 被抽 token + `−1` |
| **输出** `num_accepted` | **`[B]`** | int32 | 接受前缀长度（不含 bonus） |

迁移必需字段「输出 shape」：`output_token_ids [B, N+1]`、`num_accepted [B]`（与源 op 的
`forward` 返回顺序一致，`out_idx` 顺序为 `(output_token_ids, num_accepted)`）。

### 0.3 实现算法解读（怎么算）

源实现是**一条 kernel = 一行请求的一次流式扫描**（`_plan` 决定 `vec/threads/parts/pace`）：

| 步骤 | 计算 | 输入 | 输出 | 对应语义公式的部分 |
|------|------|------|------|-------------------|
| 1 | `draw_rounds = ceil((N+1)/threads)` 轮，每线程一个 draw：置 `key=(seed_lo,seed_hi)`、`counter=(j, line, off_lo, off_hi)`，跑 `mix()` 十轮（每轮两次 `__umulhi` + 异或 + 乘），取 `counter[0] >> 8` 转 float32 乘 `2^-24` | `seed, offset` | `token_uniform`（draw `N`）、`verdict[j]` | 步骤 1–3 |
| 2 | `verdict[j] = (u·D[b,j,id_j] < T[b,j,id_j]) ? N : j` | 2N 个标量 + `draft_token_ids` | `verdict[]` | 步骤 2 |
| 3 | `accepted = min_j verdict[j]`（串行 N 步） | `verdict[]` | `accepted` = `num[b]` | 步骤 3 |
| 4 | `target_at = (b(N+1)+accepted)·V`；`draft_at = (bN + min(accepted,N−1))·V`；第 `f` 轮 `stage()` 把 `pace` 个 16 字节向量分别读入寄存器 `held/other`，`fold()` 求 `max(0,t−d)` 的向量内和并 warp-shuffle 归约，每 (slot, warp) 写一个 fp64 `seg` 叶子 | 两行概率（draft 仅在 `accepted<N` 时读） | `seg[leaves]`（片内） | 步骤 4 |
| 5 | `parts>1` 时块内把 `seg` 再折成一个 fp64 标量写 `partial[bx]`，`T.sync_grid()` 后读回本行 `parts` 个分块和，用 `locate()` 下降选 `owner_part` 与残余 `left` | `seg`, `partial` | `owner_part`, `remainder` | 步骤 5 的第一层 |
| 6 | 拥有者块 `locate(seg, leaves, leaf_span, uniform=0, base=remainder)` → 选 (slot, warp) → 读回该 warp 的 32 个向量 → `fold` 得 `tail[]` → `locate(tail, 32, 1, …)` 选向量 → 单线程逐元素走 `vec` 步得 token | `seg`, 行内 32 向量 | `token` | 步骤 5 |
| 7 | 单线程写 `output_token_ids[b, 0..N]`（前缀 = id、停止位 = token、其余 −1）与 `num_accepted[b]` | `accepted`, `token`, `ids` | 两个输出 | 步骤 6–7 |

**数据流（源硬件视角）**

```
GM[draft_probs/target_probs 行] --load16_evict_first(16B 向量)--> register(held/other, pace=4 向量)
   --fold: max(0,t-d) + 向量内和--> register --shfl_xor×5--> 每 warp 一个和
   --lane0 写--> SMEM seg[(slot,warp) 个 fp64 叶子]
   --(parts>1) 再折一次--> SMEM lane[32] --lane0--> GM partial[bx] --T.sync_grid-->
   本行 parts 个分块和 --> SMEM chunks --locate--> owner_part + remainder
   --locate(seg)--> (slot,warp) --重读该 warp 的 32 个 16B 向量--> tail[32] --locate--> 向量下标
   --逐元素 vec 步--> token --> GM 两个输出
```

**循环与并行结构**：grid = `B · parts`（`parts = max(1, min(row_tiles, sm_count // B))`）；block =
`threads`（`512`，`B ≥ sm_count` 时 `256`）；warp = 32 lane；每 lane 每轮用 `pace=4` 个寄存器向量暂存
再折叠；`T.sync_grid()` 要求**整网格同时驻留**（跨 block 归并 `parts` 个分块和）；每行最多被 `parts`
个 block 分担，每 block 至少 `vec` 个元素宽。

**host 侧逻辑**：`_plan(vocab, batch, sm_count)` 定 `vec = (V·4 % 16 == 0) ? 4 : 1`、`threads`、`parts`、
`leaves`；`refusal()` 用 shared 预算否决过大 `leaves`；`forward` 分配 `partial`（fp64）与两个 int32 输出、
把 3 个概率张量 `view(-1)`、按 `(B,N,V,vec,threads,parts,pace)` 特化编译并缓存
（`@target_cache(maxsize=32)`）。**编译期常量**：`B, N, V, vec, threads, parts, pace`。

### 0.4 优化手段解读（为什么快）

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 | 耦合性初判 |
|---|----------|------|------|-----------------|-----------|
| 1 | 单遍流式折叠（一行只读一次） | 把 GM 读压到流量下界（残差需要两行各一遍） | 分块读 + 片内累积，不物化 CDF | 共享内存容量（片内叶子）+ 大寄存器堆 | 算法层可移植，容量依赖硬件 |
| 2 | 分层 inverse-CDF 下降（row → seg → warp → 32 向量 → 元素） | 免去 O(V) CDF 中间缓冲与二次全行读 | 每层只下降一个候选，命中层重读 32 个向量 | 无（纯算法结构） | 可移植 |
| 3 | warp shuffle 树形归约（`shfl_xor` × 5） | 低延迟跨 lane 归约，免 shared 往返 | 32 lane 蝶形异或 | warp = 32 个同步 lane | 硬件强相关 |
| 4 | 16 字节向量读 + `L2::evict_first` 缓存策略（`load16_evict_first`） | 单次读的数据不污染 L2 | `createpolicy` + `ld.global.L1::evict_first.L2::cache_hint` | CUDA 缓存策略指令 | 硬件强相关 |
| 5 | 寄存器暂存 `pace=4` 个向量再折叠 | 装载与折叠重叠，减少依赖停顿 | 每线程 4 个 `[vec]` 寄存器向量 | 大寄存器堆 | 硬件强相关（可移植其意图） |
| 6 | `__umulhi` 常数乘高位 | Philox 每轮取 32×32 的高 32 位 | CUDA intrinsic 一条指令 | `__umulhi` | 硬件强相关 |
| 7 | 行内跨 block 拆分 + `T.sync_grid()` 整网格屏障 | 小 batch 时把空转的 SM 用起来 | 全网格驻留 + 全局屏障交换 `partial` | cooperative launch / 整网格驻留 | 硬件强相关 |
| 8 | fp64 叶子驻留 shared（`seg`） | 保住参考的前缀精度、避免二次全行读 | 每 (slot,warp) 一个 fp64 和 | shared 容量 + fp64 ALU | 硬件强相关 |
| 9 | uniform 计数只含 `(draw, row, offset)` | 切分无关的逐位可复现性 | counter 不写 launch 事实 | 无（语义设计） | 保留 |
| 10 | 链测试只读 `2N` 个标量 | 接受判定不读行（省 `2N·V` 元素） | 按 `id` gather 两个标量 | 无 | 保留 |
| 11 | bonus 位置把 draft 读作 0 | 全接受时省一整行 draft（`4V` 字节） | `drafted = accepted < N` 谓词 | 无 | 保留 |
| 12 | 单 lane 写叶子（`tx % 32 == 0`） | 归约结果只落一份 | 谓词化写 | 无 | 保留（换形式） |
| 13 | 静态 shape 特化（`target_cache`） | 循环上界与分块全部编译期常量 | 工厂参数即常量 | 无 | 保留 |
| 14 | `fold` 的向量内和再跨 lane 归约的两级分组 | 少一次 shared 往返、少一批指令 | 寄存器内 `vec` 步 + 蝶形 | register + warp | 硬件强相关 |

> 识别不出来 ≠ 不存在：以上 14 项是源码显式出现的优化，§0.5 逐项处置，§0.6 逐项承接（无一静默丢弃）。

### 0.5 硬件耦合性分析与 NPU 适配决策

判定依据：本平台 ruleset `.agents/platforms/toolchains/ascend/workflow.md` +
`.agents/platforms/toolchains/ascend/references/programming.md`（`T.Kernel` 只有一维核网格、无 `threads`；无
warp 概念；跨 block barrier 未认证；`T.alloc_shared` = UB 248KB/核（本型号预算 253952B））+
本次三份设计期探针（§3.4）的编译证据。

| 条目 | 层级 | 源硬件依赖 | NPU 有等价能力？ | 处置 | NPU 对应方案 / 依据 |
|------|------|-----------|-----------------|------|---------------------|
| 语义：第 1/2/4 步（uniform、接受判定、残差） | 语义 | 无 | — | **保留** | 语义层无条件保留；`num_accepted` 逐位一致（§1.6.1 E1–E4） |
| 语义：第 5 步 fp64 顺序前缀 | 语义（精度） | fp64 ALU | **无**（AIV 无 fp64 算术证据；`T.float64` 只是类型系统里的 dtype） | **等价替换（容差内）** | 换 fp32 分层前缀 + 同一条「首个越点且正权重」规则；分布等价性机器验证（§1.6.1 E5），`num_accepted` 不受影响 |
| 算法：单遍流式折叠（#1） | 算法 | shared 容量 | 有（UB 248KB） | **保留** | 分块流式折叠，叶子换成 `ceil(V/tile)` 个 fp32 tile 和（§0.6 R1） |
| 算法：分层下降（#2） | 算法 | 无 | 有 | **保留（改层次）** | 层数由 5 层降为 4 层（行 → tile → lane → 元素），因 UB 放得下每 tile 一个和（§0.6 R2） |
| 优化：warp shuffle 归约（#3） | 优化 | warp = 32 同步 lane | 无 warp（AIV 是 128 线程 SIMT 域，无 `shfl_xor` 语义证据） | **等价替换** | block 级 reducer：`T.alloc_reducer` + `reducer_init/update/finalize` → 生成 `tl::AscendAllReduce<SumOp,128,1,0>`（探针 p2b 生成码） |
| 优化：`load16_evict_first` + 缓存策略（#4） | 优化 | CUDA `createpolicy` / `ld.global` 修饰 | **无** | **舍弃** | 本平台无对应缓存策略 API；读一次的行本来也只读一次，L2 污染不是本 op 的地板项。舍弃不影响正确性或流量（§0.6 R5） |
| 优化：`pace=4` 寄存器暂存（#5） | 优化 | 大寄存器堆 + 显式寄存器数组 | 部分（`T.alloc_local` 可用，但跨 GM→UB 的装载由编译器调度） | **舍弃** | 修改后的折叠直接 GM→UB 元素读 + UB 内残差/归约，不需要「先暂存再折叠」；意图（装载与计算重叠）由编译器流水与自动多缓冲承接（§6.3） |
| 优化：`__umulhi`（#6） | 优化 | CUDA intrinsic | **无**（探针 p1b 表明 int64 乘法虽能编译，但源写法 `(int64)a*m >> 32` 有符号溢出 + 算术右移，**数值错误**） | **重新设计** | 4 次 16×16 乘积拼 32×32 高位（只用 `&`/`<<`/`>>`/`+`/`*`，全部 32 位），逐位等价的机器验证 §1.6.1 E1–E2（§0.6 R3） |
| 优化：行内跨 block 拆分 + `T.sync_grid()`（#7） | 优化 | 整网格驻留 + 全局屏障 | **无**（平台能力缺口 `CG-950-batch-barrier`：`ascend.batch_schedule` 的跨 block barrier 依赖未认证，取得独立验证前须选已有支持路径；`programming.md`：手工跨核 flag 亦未认证） | **重新设计** | 一行一个核、**无跨核同步**：粗层分块和留在 UB（`ceil(V/tile)` 个数），不需要任何 block 间交换（§0.6 R4）。多核拆行降级为 §1.6.3 备选 B（两 kernel，Stage 4 裁决） |
| 优化：fp64 叶子驻留 shared（#8） | 优化（精度） | fp64 ALU + shared | fp64 无 | **等价替换（容差内）** | fp32 tile 和；容量从「每 (slot,warp) 一个 fp64」变成「每 tile 一个 fp32」，共 19 个 float（§4.5） |
| 优化：`vec=4` 的 16 字节向量读 | 优化 | 16B 对齐的 `ld.global.v4` | 有（编译器向量化） | **等价替换** | GM→UB 元素级 `T.Parallel` 读，由编译器按可证明性向量化（探针 p2f 生成码出现 `float2` 读）；**不假设 16B 对齐**（见 §10 风险 R2：fixture 故意传 `storage_offset=1` 的输入） |
| 优化：uniform 与 launch 无关（#9） | 优化（语义） | 无 | — | **保留** | counter 仍只含 `(draw, row, offset)` |
| 优化：链测试只读 2N 标量（#10） | 优化 | 无 | — | **保留** | 标量 gather |
| 优化：全接受时把 draft 读作 0（#11） | 优化 | 无 | — | **保留** | `if_then_else(accepted<N, …)`；探针生成码为**分支化条件加载**，谓词为假时不发访存 |
| 优化：单 lane 写（#12） | 优化 | 无 | — | **等价替换** | `if T.get_thread_binding() == 0` 守卫单点写（探针 p2a/p2f 生成码） |
| 优化：静态 shape 特化（#13） | 优化 | 无 | — | **保留** | 工厂参数 `(B,N,V,tile,threads,blocks)` 全是编译期常量 |
| 优化：向量内和 + 跨 lane 两级分组（#14） | 优化 | register + warp | 部分 | **等价替换** | 两级分组保留：每 lane 串行 `per_lane` 步 + block 全归约（折叠级用 reducer；命中 tile 级用显式 lane 部分和，探针 p2f） |
| 源 `_plan` 的 `sm_count // B` 拆分决策 | 优化 | SM 数 | 有（Vector 核数） | **等价替换** | 换成本平台核数查询与「行 → 核」折叠（§5.5） |

**判定统计**：保留 7 项 / 等价替换 6 项 / 重新设计 2 项（#6、#7）/ 舍弃 2 项（#4、#5）。语义层 3 项全部保留或容差内等价替换，无一项改变 `num_accepted` 的逐位契约。

### 0.6 NPU 算法重设计

重设计输入 = §0.5 的两个「重新设计」+ 三个「等价替换」的落地方案 + §1.6.0 的算法族调研结论。
共同约定的 NPU 侧主结构（下称 NPU 主算法）：

```
grid = min(B, 56 vector cores)，每核 for rr in serial(ceil(B/grid)): line = bx + rr·grid（line < B 守卫）
  阶段 A（标量，核内一次）：Philox 十轮 × (N+1) 个 draw（32 位拆分 mulhi）→ u[j]
        → verdict[j] → num[b] → target_at / draft_at / u_token
  阶段 B（SimtVF 向量）：for t in serial(ntile): 一个 T.Parallel(tile) pass
        读 T/（若 num<N）D → r = max(0, t−d) → reducer 累加 → thread0 写 tile_sum[t]
  阶段 C（SimtVF 向量 + 标量）：粗层 tile_sum 前缀（≤19 步）→ 命中 tile k
        → 重算该 tile 进 UB → lane 部分和（threads 个）→ lane 前缀（128 步）→ 逐元素走查（≤tile/threads 步）
  阶段 D（标量）：写 output_token_ids（N+1 个 int32）与 num_accepted
```

**R1 折叠与叶子的重设计（承接 #1/#3/#8）**

- **源方案**：`pace=4` 个 16B 向量进寄存器 → 向量内和 → warp 蝶形（5 步）→ 每 (slot,warp) 一个 fp64
  叶子写 shared；叶子数 `leaves = ceil(row_tiles/parts)·warps`。
- **NPU 新算法**：每个 `tile` 一个 `T.Parallel(tile)` pass，读两行、算残差、`reducer_update` 累加；
  reducer 完成 block 级全归约后由 thread 0 把该 tile 的和写进 UB 数组 `tile_sum[ntile]`。
  两级分组保留：lane 内串行 `tile/threads` 步（等价于源「向量内和」的一级），block 全归约
  （`tl::AscendAllReduce`，128 线程，等价于源 warp 蝶形的跨线程一级）。
- **语义保持论证**：残差元素值 `max(0,t−d)` 与源一样是 float32 逐元素运算（§1.6.1 E4 逐位一致）；
  求和分组与源不同（源：32 lane × (vec 内和) × fp64 上层；本文：128 lane × (per_lane 串行) × fp32 上层），
  因此**行总和**有 fp32 舍入差。该差只进入第 5 步的抽样点，不进入 `num_accepted`；E5 实测相对偏差
  `4.8e-08`、68 行（含 161k 词表）下标与参考零分歧、零权重命中 0 次，分布检验（512 bin × 20000 次抽样）
  与参考计数完全相同。
- **边界语义核对**：尾 tile 用 `i < V` 谓词补 0（等价于源 `element[0] < full·vec` 的零填充）；
  全接受（`num=N`）时不发 draft 行访存（源 #11 意图）；`V` 非 4 的倍数（4099）时无任何向量对齐假设。

**R2 分层下降的重设计（承接 #2，层数 5 → 4）**

- **源方案**：row → `chunks[parts]` → `seg[leaves]` → `tail[32]` → 元素，命中层「重读 32 个向量」。
- **NPU 新算法**：row → `tile_sum[ntile]`（粗层，≤19 项）→ 命中 tile 内 `lane_sum[threads]`（128 项）→
  该 lane 的 `per_lane`（≤64）个元素串行走查。命中 tile 需**重算一次**（一次 UB pass），重算量
  `tile ≤ 8192` 元素 ≪ `V`（流量的 `1/ntile`）。
- **语义保持论证**：每层只保留「首个使包含式前缀越过点且权重为正」的候选；层内累加分组与阶段 B 相同
  （同一段宏），保证粗/细层在边界上一致（源 #8 的同一意图：`locate` 与分块和用同一种累加）。参考实现的
  「零权重跳到下一个正权重 / 无解取最后一个正权重」规则完整保留（E5 角点用例：首/尾补零、`u=0`、
  `u=(2^24−1)/2^24`）。
- **边界语义核对**：`point ≥ total`（舍入）→ 命中层找不到 → 取本行最后一个正权重下标，与参考同；
  重算的 tile 与阶段 B 的分组相同，故 `left` 的符号判定与粗层前缀自洽。

**R3 `__umulhi` 的重设计（承接 #6）**

- **源方案**：`T.call_extern("uint32", "__umulhi", a, m)`，一条 CUDA 指令给出 32×32 高 32 位。
- **NPU 新算法**：把 `a = ah·2^16 + al`、`m = mh·2^16 + ml`（`m` 是编译期常数），取四个 16×16 乘积
  `p0=al·ml, p1=ah·ml, p2=al·mh, p3=ah·mh`（每个 < 2^32，uint32 精确），
  `lo = (p0 & 0xFFFF) | ((t & 0xFFFF) << 16)`，`hi = p3 + (p1>>16) + (p2>>16) + (t>>16)`，
  `t = (p1 & 0xFFFF) + (p2 & 0xFFFF) + (p0 >> 16)`（< 2^18，不溢出）。**全部算子都是 AIV 的 32 位
  整数指令**（`&`、`<<`、`>>`、`+`、`*`），不引入 64 位整数能力。
- **语义保持论证**：四个部分积之和等于精确的 64 位乘积（按 2^16 进制展开，无截断），`hi/lo` 取该乘积的
  高/低 32 位，与 `__umulhi` 定义相同 → **逐位等价**。机器验证 §1.6.1 E1（4 个乘数 × 20009 个样本，
  0 个不符）+ E2（完整十轮流，含 `seed/offset` 角点，0 位不符）。
- **边界语义核对**：`a` 取 `0/1/0xFFFF/0x10000/0x7FFFFFFF/0x80000000/0xFFFFFFFE/0xFFFFFFFF` 全覆盖；
  `m` 取两个真实乘数与 `1/0xFFFFFFFF`。**陷阱记录**：源写法改写成 `(int64)a·m >> 32` 在本平台虽能编译，
  但 `(int64_t)` 有符号溢出 + 算术右移会得到错误高位（探针 p1b 生成码可见 `int64_t` 乘法与 `>>`），
  故明确**不采用** 64 位路线。

**R4 取消网格屏障的重设计（承接 #7）**

- **源方案**：`parts = max(1, min(row_tiles, sm_count // B))` 个 block 分担一行，各写 `partial[bx]`，
  `T.sync_grid()` 后逐块读回本行 `parts` 个分块和再 `locate`。
- **NPU 新算法**：**一行一个核、整行在一个核内闭合**：分块和（`tile_sum`）留在本核 UB，粗/细层下降也在
  本核完成，**没有任何跨核数据交换，也不需要屏障**。核数按 §5.5 折叠：`grid = min(B, 56)`，
  `ceil(B/56)` 行串行处理。
- **语义保持论证**：`num_accepted` 与 uniform 都与「行怎么切」无关（源 #9 的设计目标，本文原样保留：
  counter 只含 `(draw,row,offset)`）；行内前缀的累加分组改变只影响第 5 步的下标选择，落在 §0.1 允许的
  容差内（E5）。取消屏障后不存在「部分和已写但未读到」的窗口，**正确性依赖更少**。
- **代价与备选**：`B < 56` 时核利用度 = `B/56`（qwen3 的 `B=32` → 57%）。多核拆行需要两个 kernel
  （分块和落 GM + 收尾 kernel 重读命中块），流量 `+1/parts`、多一次 launch —— 列为 §1.6.3 备选 B，
  由 Stage 4 按 `B < 56` 的 workload 实测裁决（判定阈值见 §1.6.3）。

**R5 舍弃项的说明（#4 缓存策略、#5 寄存器暂存）**

- **#4**：本平台没有 CUDA 的 `createpolicy` / `ld.global.L1::evict_first` 等价 API（`programming.md` 未列出
  任何缓存策略接口；`copy_op.py` 只有搬运方向与 padding 选项）。该手段的目的（只读数据不污染 L2）在本算子
  不构成地板项：每行只读一次（`1+1/ntile` 遍），且没有「后读复用」的数据。**舍弃不影响正确性，也不增加
  流量**；若 Stage 4 发现 L2 争用（多个算子并发），再回到本项补取证。
- **#5**：`pace=4` 的寄存器暂存是为「一个 16B 向量一条指令」的 CUDA 读法服务的；本平台 GM→UB 走
  `T.Parallel` 元素读（编译器可向量化）或 DMA，装载与计算的解耦由编译器流水/自动多缓冲承担（§6.3）。
  保留其意图（装载与计算重叠）而不保留其形态；若 Stage 4 实测 `mte2_ratio < 0.95`（对照
  `CONST-copy-floor-method` 的判据），再考虑显式多缓冲版本。

**重设计完成度自检**：每项都有「源方案 → NPU 新算法 → 语义保持论证」三要素；§0.4 的 14 项优化逐项在
§0.5 有处置、在 §0.6 或 §1 有承接（无静默丢弃）；NPU 主算法满足 §4.5 的 UB 预算与 §5.5 的分核约束。

### 0.7 标杆实现

- **金标准**：`examples/TileOPs/tileops/workloads/sampling.py` 的 `philox_uniform` +
  `chain_speculative_sampling`（torch 张量算术，CPU/GPU/NPU 同结果）。§8.1 的 golden 直接移植它，
  **不移植**本文的 NPU 分块/流式结构。
- **契约校验器**：同文件的 `verify_chain_outputs` / `assert_chain_distribution`（结构 + 分布）。
- **规格**：`examples/TileOPs/tileops/manifest/sampling.yaml`（8 个 workload + roofline 公式）；
  精度用例 `examples/TileOPs/tests/ops/test_sampling.py`；性能用例
  `examples/TileOPs/benchmarks/ops/bench_chain_speculative_sampling.py`。
- **性能对照**：NPU 上没有 FlashInfer / vLLM 的 rejection sampler，也不对 golden 做 profile（精度参考不
  参与性能验收）；性能口径取 manifest roofline（`flops` / `bytes`）与本设计 §1.6.0 的设计期估算下界。

## 1. 概述

### 1.1 算子名称

`_chain_speculative_sampling_kernel`（NPU 侧类名 `ChainSpeculativeSamplingFwdKernel`，
op 名 `ChainSpeculativeSamplingFwdOp`，manifest family `sampling`）。

### 1.2 功能描述

投机解码的**链式验证 + 残差抽样**：一条请求带 `N` 个 draft token，逐个用 target/draft 在同一 token 上的
概率比做接受判定；在第一个拒绝的位置，从残差分布 `normalize(max(0, target − draft))` 抽一个 token；
若全部接受，则从 target 的第 `N` 行抽 bonus token；`−1` 填在停止位之后。输出的
`num_accepted`（不含 bonus）与接受前缀必须与参考**逐位一致**，抽到的 token 只需服从同一分布且落在
正权重上（§0.1）。

### 1.3 数学公式

$$
\begin{aligned}
u_{b,j} &= 2^{-24}\cdot\big(\text{Philox4x32-10}_{(k_0,k_1)=(seed_{lo},seed_{hi}),\,
c=(j,b,off_{lo},off_{hi})}[0] \gg 8\big), \quad j=0..N\\
a_{b,j} &= \big[u_{b,j}\cdot D_{b,j,id_{b,j}} < T_{b,j,id_{b,j}}\big] \quad(\text{fp32})\\
n_b &= \min\{j : \neg a_{b,j}\}\\
w_{b,i} &= \max\big(0,\; T_{b,n_b,i} - (n_b<N\;?\;D_{b,n_b,i}:0)\big)\\
p_b &= u_{b,N}\cdot\textstyle\sum_i w_{b,i}\\
\tau_b &= \min\{k : \textstyle\sum_{i\le k} w_{b,i} > p_b,\; w_{b,k}>0\}\\
out_{b,j} &= \begin{cases} id_{b,j} & j<n_b\\ \tau_b & j=n_b\\ -1 & j>n_b\end{cases},
\qquad num\_accepted_b = n_b
\end{aligned}
$$

补充两条文字规则（不写成花括号集合）：① 若 `accept` 全为真，`n_b = N`，且集合 `j : ¬a_{b,j}` 为空；
② 若不存在满足条件的 `k`（`p_b` 已越过本行总质量），`tau_b` 取本行最后一个 `w > 0` 的下标。
`CDF` 的累加在参考实现里是 fp64 左到右；本文换 fp32 分层前缀（§0.5 / §1.6.1 E5）。第 2 步与第 4 步的
算术 dtype 与参考完全相同。

### 1.4 算法描述

**迁移决策后的 NPU 算法**（= §0.6 的 NPU 主算法，也是 §1.6.0 选定的算法族 C1）：

1. **核映射**：`grid = min(B, 56)`（实查 56 个 Vector 核，§5.5），每核 `for rr in serial(ceil(B/grid))`
   处理 `line = bx + rr·grid`，`line < B` 守卫。
2. **阶段 A（标量，一次）**：`for j in serial(N+1)` 用 32 位拆分 mulhi 跑 Philox 十轮，得 `u[j]`；
   对 `j<N` 用 `if_then_else(u[j]·D < T)` 记 `verdict[j]`（接受记 `N`，拒绝记 `j`）；`n = min_j verdict[j]`；
   算出 `target_at = (line·(N+1)+n)·V`、`draft_at`、`u_token = u[N]`。
3. **阶段 B（向量折叠）**：`ntile = ceil(V/tile)`，每 tile 一个 `T.Parallel(tile)` pass：读
   `T[target_at + t·tile + i]`（与 `n<N` 时的 `D[draft_at + …]`）→ `r = max(0, ·)` → `reducer_update`；
   归约后 thread 0 写 `tile_sum[t]`。
4. **阶段 C（定位）**：`point = u_token·total`；粗层串行走 `tile_sum`（≤19 步）得命中 tile；
   重算该 tile 进 `r_ub`；`lane_sum[l]`（每 lane 串行 `per_lane` 步）；串行走 `lane_sum`（≤128 步）得命中
   lane；该 lane 的 `per_lane`（≤64）个元素串行走查，取「首个越过残余点且权重为正」。两条**条件兜底**
   （概率为零的窗口，不进入常规路径）：lane 内无越点元素 → 一次 `op="max"` 归约取本 tile 最后一个正权重；
   `point ≥ total` → 一次全行 `op="max"` 归约取本行最后一个正权重（与参考的 `last_positive` 一致）。
5. **阶段 D（写出）**：`for j in serial(N+1)`：`j<n → id`、`j==n → token`、否则 `−1`；`num_accepted[line] = n`。

窗口参数：`tile = min(8192, ceil(V/threads)·threads)`，`threads = 128`，`per_lane = tile/threads`。
**与源的结构差异一句话**：源「一行多 block + 全局屏障 + fp64 叶子 + warp 蝶形」，本文「一行一核 +
无跨核同步 + fp32 tile 叶子 + block 全归约 + 单 tile 重算」。流量同为「两行各一遍」（全部接受时一行一遍）
外加 `1/ntile` 的重算，即**未退化**。

### 1.5 数据流图

```
GM[draft_token_ids 行 N 个 int32] --标量 gather N 个--> UB[ids]; UB[verdict]
GM[seed/offset int64] --int32 位视图--> UB 标量 --Philox×10--> u[j]（寄存器/UB 标量）
UB[verdict] --串行 min N 步--> n --→ target_at / draft_at / u_token（UB 标量）

for t in 0..ntile-1（每 tile 一个向量 pass，UB r 缓冲只在命中 tile 时保留）:
  GM[target_probs 行 + t·tile .. ] --T.Parallel 元素读--> 寄存器
  GM[draft_probs  行 + t·tile .. ] --（n<N 时）T.Parallel 元素读--> 寄存器（否则常量 0）
  --max(0, t−d)--> 寄存器 --reducer_update--> lane 部分和 --AscendAllReduce--> tile 和
  --thread0--> UB tile_sum[t]        （同时记录本 tile 内最后一个正权重位置）

UB[tile_sum ≤19] --串行前缀--> 命中 tile k; u_token·total = point
GM[命中 tile 的两行] --重算--> UB r_ub[tile]
UB r_ub --每 lane 串行 per_lane 步--> UB lane_sum[128]
UB lane_sum --串行前缀--> 命中 lane; 残余点 left
UB r_ub[该 lane 段] --串行 per_lane 步--> token 下标
UB[ids / n / token] --串行 N+1 步--> GM[output_token_ids 行]; GM[num_accepted]
```

UB 缓冲：`tile_sum`、`lane_sum`、`r_ub`、`ids`、若干标量。无 L1/L0 参与（纯 Vector 路径，无 Cube）。

### 1.6 算法调研与优化分析 ⭐

#### 1.6.0 算法调研（Algorithm Research）⭐

**调研对象**：同一数学语义（§0.1 的 7 步）的算法族。源算法只是**基线候选 C0**。调研深度：本算子是
「规约 + 随机抽样 + 多步」类，按 skill 要求做完整调研（R1 候选表 + R3 四口径 + R4 逐候选）。

**R1 等价化简公式 / 结构重排候选**

| # | 候选 | 公式 / 结构 | 等价性初判 | 收益方向 |
|---|------|------------|-----------|---------|
| **C0** | **基线（源 GPU 算法）** | 一行 `parts` 个 block 流式折叠出 fp64 `seg` 叶子 + 全局屏障交换分块和 + 分层下降重读 32 向量 | 数学恒等（fp64 前缀） | 流量已是下界；对齐/屏障/精度都绑 GPU |
| C1 | **分层 fp32 流式折叠 + 单 tile 重算**（本设计） | 每 tile 一个 fp32 和（`≤19` 个）驻留 UB，粗/细两级下降，命中 tile 重算一次 | 容差内等价（第 5 步分组变；第 2/4 步逐位） | 免 fp64、免屏障、免 O(V) 缓冲；流量 `1+1/ntile` |
| C2 | CDF 物化到 GM | pass1 读两行算残差并写 `CDF[V]`（fp32/fp64），pass2 用 `searchsorted` 读 CDF | 恒等（若 fp64） | **负收益**：多写 `4V`（fp32）/`8V`（fp64）+ 多读同量 |
| C3 | 参考实现直译：fp64 全行 `cumsum` | `cdf = weights.double().cumsum(-1)` 然后 `searchsorted` | 恒等（就是参考） | **负收益**：`8V` 中间缓冲 + 顺序依赖 + fp64 ALU（AIV 无） |
| C4 | 单遍加权 reservoir / Gumbel 竞速 | 一次扫描用 `V` 个指数竞速随机数取代 inverse-CDF | 分布等价（不同随机流） | **负收益**：`V` 次超越函数；破坏「draw `N` 的 uniform 决定 token」的源结构 |
| C5 | 多核拆行（两 kernel） | kernel A：`B·parts` 个核各算一段残差和；kernel B：`B` 个核收尾（读分块和 + 重读命中块 + 下降） | 容差内等价（第 5 步再分组一次） | 流量 `+1/parts`；换取 `B<56` 时的核利用度（备选） |
| C6 | 放弃折叠，两次全行读 + 元素级 CDF 比较 | pass1 求和，pass2 全行累加找越点 | 恒等（顺序 fp32，更接近参考的顺序前缀） | 流量 `2×`；只在「fp32 顺序前缀的精度」是硬需求时才有意义 |

R1 初判结论：**C1 相对基线不改第 1/2/4 步、只改第 5 步的分组**，是唯一同时满足「免 fp64 + 免屏障 +
流量不增」的结构；C2/C3 流量翻倍、C4 计算量爆炸、C6 流量翻倍。C5 作为核利用度备选保留到 R4/§1.6.3。

**R2 在线算法（单遍流式变体）**

**有，且采纳。** 依据：① 结构判据——`Σ_i w_i` 与「每 tile 和」都是可增量维护的 running 统计量
（和 / 分组和），属 skill §3 R2 明确列出的「分块归约 + 合并」形态；② 源算子自身已是在线形态（§0.4 #1
单遍折叠），这是「存在在线变体」的直接证据；③ 参考实现（C3）的 `cumsum` 需要 `O(V)` 中间缓冲与
`O(V)` 顺序依赖，是在线性的**反例基线**。收益口径：扫描遍数 `2 → 1+1/ntile`、中间缓冲
`O(V)`（8V 字节，GM）→ `O(ntile + threads)`（UB，≤19+128 个 float）、UB 驻留可行（§4.5）。
**唯一的不可在线化点**：「越过点」的下标只有在 `total` 已知后才能定（inverse-CDF 的本质），因此保留
一次 `O(tile)` 的重算，而不是 `O(V)` 的第二遍。

**R3 算法复杂度（四口径对比，per 行；`s = n_b`，`s<N` 时读两行）**

口径说明：FLOPs 只计算术（乘加=2、`max`/比较各 1；超越函数按向量单元折算，标注为估算）；Bytes 只计
GM 流量（输入读 + 输出写 + 中间 GM 往返，片上不计）；「遍数」= 同一行概率数据的 GM 重复读次数；
「中间缓冲」= 需要落地的最大缓冲（片上按 float 数、GM 按字节）。

| 候选 | FLOPs（估算） | Bytes（GM） | 扫描遍数 | 中间缓冲峰值 | 可并行度 | 跨核归并代价 |
|------|--------------|------------|---------|-------------|---------|-------------|
| C0 基线 | `3V + V` ≈ `4V` | `4V`（全接受）/ `8V`（拒绝） | 1 + 32·vec/V | fp64 leaves（shared，`leaves·8B`） | `B·parts` | 全局屏障一次（整网格驻留） |
| **C1 本设计** | `3V + V` ≈ `4V` | `(4V 或 8V)·(1+1/ntile)` | `1 + 1/ntile` | `tile` 个 float（UB）+ `ntile+threads` | `min(B,56)` | **无** |
| C2 CDF 物化 | `4V` | `8V`(读) + `4V`(写) + `4V`(读) = `16V` | 3 | `V` 个 float（GM，`4V`B） | `B·parts` | 屏障或两 kernel |
| C3 参考直译 | `4V`（fp64） | `16V`（fp64 CDF 写读各 `8V`） | 3 | `V` 个 double（GM，`8V`B） | `B` | 无（单核一行）× |
| C4 Gumbel 竞速 | `≈ 12V`（`V` 次 `exp`+`log` 折算） | `8V` | 1 | `O(1)` | `B` | 无 |
| C5 两 kernel 拆行 | `4V` | `(4V 或 8V)·(1+1/parts)` | `1 + 1/parts` | `B·parts` 个 float（GM，`4B·B·parts`≈`0.9KB`） | `B·parts` | 一次 kernel 边界（无屏障） |
| C6 两遍全行 | `4V` | `2·(4V 或 8V)` | 2 | `O(1)` | `B` | 无 |

主导项是 Bytes（本算子每个元素只做 3~4 次算术，带宽受限）：C2/C3 是 C1 的 2 倍流量，C6 是 2 倍，
C4 的 FLOPs 比 C1 高一个量级（`V` 次超越函数），C0 与 C1 同量级但 C0 依赖本平台没有的屏障/fp64/warp。
**C1 与 C0 在 R3 上并列**（流量同阶），决胜在 R4 的硬件亲和性。

**R4 算法硬件亲和性（逐候选）**

检查清单：计算单元匹配 / 片上容量 / 对齐整除 / 静态边界 / 流水可融合性 / 跨核结构。

| 候选 | 计算单元匹配 | 片上容量 | 对齐/整除 | 静态边界 | 流水可融合 | 跨核结构 | 结论 |
|------|-------------|---------|----------|---------|-----------|---------|------|
| C0 | 逐元素/归约本可走 AIV，但 fp64 前缀在 AIV **无 fp64 ALU 证据**、warp 蝶形需 32 同步 lane（AIV 是 128 线程 SIMT 域）、`__umulhi`/`createpolicy` 无 | `leaves·8B` 受 shared 预算约束（UB 248KB 装得下） | `ld.global.v4` 需 16B 对齐；`V=4099` 时 `vec=1` 退化 | 静态（编译期常量） | 可 | **需整网格驻留 + 全局屏障（本平台未认证）** | **淘汰**（3 项无等价能力，1 项未认证） |
| **C1** | 全 fp32 逐元素 + block 全归约 + 标量走查，全部落在 AIV 原生能力（探针 p2a/p2b/p2d/p2f 生成 `tl::AscendAllReduce`、`max`、`float2` 读） | UB 峰值 ≈ 33KB（`tile=8192`），含膨胀 ≤ 57KB ≪ 248KB | 无对齐假设（元素级读）；`V` 任意取值的尾 tile 由 `i<V` 谓词覆盖 | 全静态（`B,N,V` 编译期） | 可（每 tile 一个 pass，编译器可流水） | **无跨核** | **选定** |
| C2 | 同 C1，但多一次 GM 写读 | `V` 个 float 的 GM 缓冲 | 同 C1 | 静态 | 可 | 屏障或两 kernel | 淘汰（流量 2×，无正确性收益） |
| C3 | **fp64 ALU 缺失** → 淘汰主因 | `8V`GM | — | — | — | — | **淘汰** |
| C4 | `V` 次 `exp`/`log`：超越函数吞吐比逐元素低一个量级，`V=151936` 时变为计算受限（与 R3 的 `12V` 折算一致） | `O(1)` | — | 静态 | 可 | 无 | **淘汰**（把带宽受限算子变成 SFU 受限） |
| C5 | 同 C1（两 kernel 都是 AIV） | 同 C1 + `B·parts` 个 float 的 GM workspace | 同 C1 | 静态 | 可（两个 launch） | **一次 kernel 边界（无屏障，绕开 `CG-950-batch-barrier`）** | **保留为备选 B**（§1.6.3） |
| C6 | 同 C1（顺序 fp32 前缀；更接近参考的顺序性） | `O(1)` | 同 C1 | 静态 | 可 | 无 | 弃选（流量 2×，换来的只是「顺序前缀」这一非契约精度） |

**设计期 roofline 估算（D-2）**：常数取自
`.agents/skills/tilelang-op-optimize/references/pattern-library/constants.md`（全部标注 `device: 910B2C`，
**未在本型号实测**，按「未实测假设 + 估算依据」使用）：

- `CONST-mte2-degradation`（910B2C，2026-09-21 版本戳）：混合读写全流量地板 `≈1.26 TB/s`（聚合）→ 流量项
  `= bytes / 1.26e12`。
- `CONST-vector-launch-overhead`（910B2C，2026-08-24）：`≈0.5 µs`/向量 op 固定发射 → 发射项
  `= (ntile + 2) · 0.5 µs · 每核行数`（每 tile 一个折叠 pass + 命中 tile 一个重算 pass + 一个 lane pass）。
- 容量项：UB 峰值 `33 KB`（`tile=8192`）对 248KB → 不构成下界，记 0；L0/L1 不参与（无 Cube）。
- 估算下界 = `max(流量项, 发射项, 容量项)`；代表 workload（`tile=8192`，`grid=min(B,56)`）：

| workload | B | N | V | bytes (MB) | ntile | 每核行数 | 流量项 (µs) | 发射项 (µs) | 估算下界 (µs) |
|---|---|---|---|---|---|---|---|---|---|
| chain-sampling-smoke | 2 | 2 | 64 | 0.001 | 1 | 1 | 0.0 | 1.5 | **1.5** |
| llama-8b-b16-n4 | 16 | 4 | 128256 | 8.21 | 16 | 1 | 6.5 | 9.0 | **9.0** |
| llama-8b-b64-n4 | 64 | 4 | 128256 | 32.84 | 16 | 2 | 26.1 | 18.0 | **26.1** |
| ds-v3-mtp-n1 | 64 | 1 | 129280 | 33.10 | 16 | 2 | 26.3 | 18.0 | **26.3** |
| ds-v3-mtp-n3 | 64 | 3 | 129280 | 33.10 | 16 | 2 | 26.3 | 18.0 | **26.3** |
| qwen3-235b-eagle-n5 | 32 | 5 | 151936 | 19.45 | 19 | 1 | 15.4 | 10.5 | **15.4** |
| llama-8b-b17-n8 | 17 | 8 | 128256 | 8.72 | 16 | 1 | 6.9 | 9.0 | **9.0** |
| llama2-7b-b256-n1 | 256 | 1 | 32000 | 32.77 | 4 | 5 | 26.0 | 15.0 | **26.0** |

流量项按 manifest roofline 的「更便宜一侧」计（`chain_bytes = min(12N+4V, 4+8V)`，即全部接受时只读
一行 target），再乘 `1+1/ntile` 的重算系数；**不用 MFU**（本算子 FLOPs/byte ≈ 0.5，是带宽主导）。
估算下界: 代表 workload 下界 1.5–26.3 µs（流量 0.0–26.3 + 发射 1.5–18.0 + 容量 0，引用
`CONST-mte2-degradation` / `CONST-vector-launch-overhead`，均为 910B2C 实测值外推，本型号待 Stage 4 校准）。
发射项在小 V / 小 B 的 workload 上压过流量项，说明**减少每核 op 组数（增大 `tile`、`B<56` 时压行数）**
是 Stage 4 的第一杠杆；`tile` 的取值以 UB 预算为上界（§1.6.3 表）。op 组数按「每 tile 一个折叠 pass +
命中 tile 一个重算 pass + 一个 lane pass」计；若折叠 pass 必须拆成两个 `T.Parallel`（§3.3 的形状注），
发射项 ×1.7，`tile` 翻到 16384 即抵消，结论（流量与发射同阶、带宽为主导）不变。

**调研结论**：选定 **C1「分层 fp32 流式折叠 + 单 tile 重算」外加 §0.6 的 NPU 主结构（一行一核、无跨核
同步、32 位拆分 mulhi 的 Philox）**。关键依据三条：(a) R3 显示 C1 与基线 C0 在流量与 FLOPs 上同阶
（`4V` FLOPs、`1+1/ntile` 遍），而 C2/C3/C6 流量翻倍、C4 计算量翻十倍；(b) R4 显示只有 C1（与 C5）的
全部计算落在 AIV 原生能力上——fp64 ALU、warp 蝶形、整网格屏障、`__umulhi`、CUDA 缓存策略这五项源依赖
在本平台逐项无等价物（`references/programming.md` 与探针证据）；(c) 与源算法的**优化意图同构**：
单遍折叠（#1）、分层下降（#2）、两级分组归约（#3/#14）、uniform 与 launch 无关（#9）全部承接，源算法
被替换的只是硬件形态（fp64→fp32、warp→block reducer、屏障→无屏障、`__umulhi`→拆分乘法），依据见
§0.5/§0.6。**调研范围**：`algorithm-research.md` §5 参考表（实查：`python3 .agents/tools/kb_search.py
"sampling inverse cdf philox" --task-dir 任务目录` 返回 `no entries matched`；同工具查
`"row reduce simt reducer"` 只命中能力缺口 `CG-950-simd-shared-reduce`，与本算子无关——本平台
knowledge 与 pattern-library 对 sampling/抽样族**无命中条目**），故候选由源算子
（§0.3/§0.4）+ 数学恒等自查（C1/C2/C6）+ 抽样算法族已知变体（C4 reservoir/Gumbel 竞速，
模型知识，属公开已知算法）+ 平台 `knowledge/` 的向量执行条目（PL-950-row-primitives / PL-950-simd-reduce /
TRAP-950-simd-shared-reduce）+ 本算子 3 份设计期探针补足；pattern-library `constants.md` 的 4 条相关常数
全部引用。C5 未被淘汰而是降级为备选（R3/R4 与 C1 并列），按 skill 要求交 §1.6.3 的实验裁决，不纸面拍板。

#### 1.6.1 数学等价优化（公式级）

分析对象 = §1.6.0 选定的 C1（含 §0.6 的 NPU 主结构）。逐项给出「原式 → 优化后公式 → 等价性论证 →
收益量化」，并在末尾附机器验证结果表（`verify_equiv.py`，本目录，torch CPU + fp64 参照）。

**A1 Philox 的 32×32 高位（mulhi）**

- **原式**：`__umulhi(a, m) = floor((a·m) / 2^32)`，一条 CUDA intrinsic；源写法若改写成
  `(int64)a·m >> 32` 则因有符号溢出 + 算术右移**算错**（探针 p1b 生成码确认平台按 int64 乘法降级）。
- **优化后**：`a = ah·2^16 + al`、`m = mh·2^16 + ml`（`m` 为编译期常数），
  `p0=al·ml, p1=ah·ml, p2=al·mh, p3=ah·mh`；`t = (p1 & 0xFFFF) + (p2 & 0xFFFF) + (p0 >> 16)`；
  `lo = (p0 & 0xFFFF) | ((t & 0xFFFF) << 16)`；`hi = p3 + (p1>>16) + (p2>>16) + (t>>16)`。
- **等价性论证**：按 2^16 进制展开 `(ah·2^16+al)(mh·2^16+ml) = p3·2^32 + (p1+p2)·2^16 + p0`，
  `p0 < 2^32`、`p1/p2/p3 ≤ (2^16−1)^2 < 2^32`、`t < 3·2^16 < 2^18`，全部中间量在 uint32 内精确保存；
  取模 2^32 的高/低 32 位即为精确 64 位乘积的两半 → 与 `__umulhi` **逐位恒等**。机器验证 E1（4 个乘数
  × 20009 个样本 = 160072 个 32 位字，0 不符）、E2（完整十轮流逐位相同，含 `seed/offset` 角点）。
- **收益量化**：把一条不可用的 intrinsic 换成 10 条 AIV 原生 32 位指令；调用量 `B(N+1) ≤ 2304` 次
  （最大 workload `256×2`）→ 总代价 `≈ 2.3e4` 条标量指令 `≈ 13 µs`/2304 次 ≈ **26 ns/核一轮**，
  占 §1.6.0 估算下界（15.4–26.3 µs）的 **< 0.3%**。零流量代价。

**A2 接受前缀：用「首个拒绝」替代 `cumprod().sum()`**

- **原式（参考）**：`num = (u·d < q).int().cumprod(1).sum(1)`（逐元素串行乘法链）。
- **优化后**：`num = min{ j : ¬accept_j }`（全部接受则 `N`）。
- **等价性论证**：`cumprod(j) = ∏_{i≤j} accept_i ∈ {0,1}` 恰在首个拒绝后恒 0，故
  `Σ_j cumprod(j) = 首个拒绝的下标`（无拒绝时 `= N`）。恒等（布尔域），故**逐位恒等**。E3 验证
  （92 行 4 组 shape，0 不符）。
- **收益量化**：省 `N−1` 次乘（`N ≤ 8` 时 < 10 op；`N = 600` 的压力用例省 599 次乘），并把串行乘法链换成
  串行 min 链（同长度）。收益量级小，但为**零风险恒等变形**，采纳。

**A3 残差就地重算（不物化 target 行）**

- **原式**：先把 target tile 读进 UB，再 `r = max(0, held − other)`，`held` 被覆盖。
- **优化后**：读的两行直接进寄存器参与 `r = max(0, T[·] − (n<N ? D[·] : 0))`，只写 `r`，
  不保留 `t` 的 UB 副本；命中 tile 需要时**重算一次**。
- **等价性论证**：`r` 的每个元素仍是同一个 float32 减法 + `max`，操作数相同、舍入路径相同 → 逐位恒等
  （E4：92 行残差逐位一致）。重算不改变结果（同一输入、同一算式、确定性）。
- **收益量化**：省一个 `tile·4 B` 的 UB 缓冲（`tile=8192` 时 32KB）与每 tile 一次 UB 写（`tile` 个
  store）；代价是命中 tile 多读一次 GM（`4V/tile` 字节 → 流量的 `1/ntile ≈ 5%`），净收益为正（UB 压力
  减半、op 组数不变）。GM 流量已计入 §1.6.0 的 `1+1/ntile`。

**A4 CDF 前缀：fp32 分层前缀替代 fp64 顺序前缀**

- **原式（参考与源实现）**：`CDF = weights.double().cumsum(-1)`（左到右 fp64 累加），
  `point = u·CDF[-1]`，`searchsorted`。
- **优化后**：`tile_sum[t] = Σ_{i∈tile t} r_i`（fp32，lane 串行 + block 全归约），
  `total = Σ_t tile_sum[t]`（fp32，顺序），`point = u·total`（fp32），粗层/细层用同一分组前缀。
- **等价性论证**：**容差内等价**（不是恒等）。误差来源：fp32 分层求和的相对误差上界
  `≈ (per_lane + log2(threads) + ntile)·2^-24 = 90·2^-24 ≈ 5.4e-6`（`per_lane = 64`、`threads = 128`、
  `ntile ≤ 19`）；`total` 的 fp32 舍入实测相对偏差 `4.845e-08`，而**分层前缀本身**的误差效应由 E5 的
  下标与分布对照直接覆盖（0/68 分歧）。影响面仅限第 5 步「抽到哪个下标」：`u·total` 的扰动若跨过某个
  权重边界就会选邻近下标。E5 实测：68 行（含 `V=151936`）× 角点（首/尾补零、`u=0`、`u=(2^24−1)/2^24`）
  下标与参考**零分歧**，零权重命中 **0 次**；分布检验（`V=512`、20000 次抽样、512 bin）候选与参考的
  计数逐 bin 相同（最大差 0，6σ 界 77）。**`num_accepted` 与接受前缀完全不受影响**（它们只走 A2/A5，
  逐位一致）。舍入方向无偏（随机对称），不会系统性偏向某个下标。
- **收益量化**：中间缓冲 `8V` B（GM）→ `(ntile + threads + tile)·4 B`（UB，`tile=8192` 时 33KB）；
  GM 流量至少 `16V`（CDF 的 fp64 写 + 读，未计 `next_positive`/`argmax` 的整行临时量）→ `8V(1+1/ntile)`
  （`V=151936` 时每行少 1.2 MB 的 CDF 往返）；消除 `O(V)` 顺序依赖
  （fp64 cumsum 在 AIV 上根本无法实现，故这一项同时是**可行性前提**）。

**A5 全接受位置不读 draft 行**

- **原式（参考）**：`padded_draft = cat(draft, zeros[·, :1])`，bonus 位置减一个零行。
- **优化后**：`n == N` 时残差 `r_i = T[N,i]`，draft 访存由谓词 `n < N` 关掉（`if_then_else` 生成
  分支化条件加载，谓词为假不发访存）。
- **等价性论证**：减 0 与不减在 IEEE float32 下对 `x − 0.0` 恒等（`x = −0.0` 时 `−0.0 − 0.0 = −0.0`，
  `max(·,0)` 结果仍 0；与参考的 `clamp_min(0)` 一致），且 `max(0, x − 0) = max(0, x)`。E4 覆盖包含
  `n = N` 的行（`N=1/4/8` 的数据里 `n=N` 常出现），逐位一致。
- **收益量化**：省一整行 draft 读 = `4V` 字节/行；manifest roofline 的「更便宜一侧」正是这一项
  （`chain_bytes = min(12N+4V, 4+8V)`），最大 workload 上占该行流量的 50%。

**A6 抽样规则：一次「首个越点且正权重」走查替代参考的整行 `next_positive` + `argmax`**

- **原式（参考 `_draw`）**：`cdf.cumsum` → `searchsorted` → `next_positive = where(pos, col, V).flip.cummin.flip`
  → `last_positive = V−1−argmax(flip(pos))` → 三段 where。
- **优化后**：命中 tile/lane 内取「首个 `w > left` 且 `w > 0`」的下标；若不存在，取**本行最后一个正权重**
  （阶段 B 顺带用 `tile_sum` 之外的极值记录，或在 UB 里按 tile 记一个「本 tile 最后正权重位」再取最大）。
- **等价性论证**：`searchsorted(cdf, point, right=True)` 返回首个 `cdf[k] > point` 的下标；由
  `cdf[k] > point` 可推出 `w_k > 0`（否则 `cdf[k] = cdf[k−1] ≤ point` 矛盾），故「首个越点」自动落在正
  权重上，参考的 `next_positive` 跳步在该点永不触发；只有 `point ≥ total`（舍入或 `u → 1`）时才需要
  「取最后一个正权重」，两式的兜底相同。E5 覆盖这两种角点，结果与参考一致。
- **收益量化**：省 `O(V)` 的 `cummin` + `argmax` 两遍整行扫描（参考实现约 `+8V` 字节读 + 2 次全行写）
  与 `4V` 的 CDF 物化；本实现只走 `per_lane ≤ 64` 步。

**等价性机器验证（D-1）结果表**（脚本：本目录 `verify_equiv.py`；`/workspace/venv-ascend950/bin/python
verify_equiv.py`，2026-11-23 执行，耗时 34s，15/15 项通过）

| 项 | 对照 | 指标 | 结果 |
|----|------|------|------|
| A1 | 32 位拆分 mulhi vs 参考 `_mulhilo`（`m ∈ {M0, M1, 1, 0xFFFFFFFF}`，20009 样本含角点） | 不符的 32 位字 / 总数 | 0 / 160072 → **EQUIV_PASS** |
| A1 | 十轮 Philox 流（`256×9`, `1×1`, `2×2`）vs 参考 `philox_uniform` | 逐位不符数；最大偏差 | 0；0.0 → **EQUIV_PASS** |
| A1 | 十轮 Philox 流，`seed/offset` 角点（`0/0`、`−1/−1`、`2^63−1/2^63−1`、`1234/0`、`0/7`） | 逐位不符数 | 0 / 36（每组） → **EQUIV_PASS** |
| A2 | 「首个拒绝」vs `cumprod().sum()` | `num_accepted` 不符行数 | 0 / 92 → **EQUIV_PASS** |
| A3+A5 | 就地残差（含 `n=N` 的零 draft 行）vs 参考 fp32 残差 | 逐位不符行数 | 0 / 92 → **EQUIV_PASS** |
| A4+A6 | fp32 分层前缀 + 走查 vs fp64 `cumsum` + `searchsorted` + `next_positive` | 下标分歧行数；零权重命中数 | 0 / 68；0 → **EQUIV_PASS** |
| A4 | `\|u·total_fp32 − u·total_fp64\| / total` | 最大相对偏差 | 4.845e-08 → **EQUIV_PASS** |
| A6 | 分布：`V=512`，20000 次抽样，512 bin 的计数 vs 参考 | 最大逐 bin 差 / 6σ 界 | 0 / 77 → **EQUIV_PASS** |
| A2–A6 | 输出行（前缀 + 被抽 token + `−1` 尾）vs 参考 | 行不符数（允许被抽下标不同，实测相同） | 0 / 92 → **EQUIV_PASS** |

**优化结论**：采纳 A1–A6 共 6 项（4 项逐位恒等、2 项容差内等价），无一项使 `num_accepted` / 接受前缀 /
uniform 偏离参考；A4 同时是可行性前提（AIV 无 fp64 算术）。§3.1 的公式拆解与 §3.3 伪代码以本节
A1–A6 的优化后公式为唯一输入。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

盘点实现方案（§1.4 五阶段）中**全部**循环与标量计算点：

| # | 计算点 | 规模 | 判定 | 替代方案 / 保留理由（类别） |
|---|--------|------|------|---------------------------|
| V1 | 折叠 pass：两行元素读 + 残差 + 累加 | 每 tile `tile=8192` 元素 × `ntile` | **已向量化** | `T.Parallel(tile)`（`T.SimtVF(threads=128)` 内），生成码出现 `float2` 读与 `tl::AscendAllReduce`（探针 p2b/p2f） |
| V2 | lane 部分和（命中 tile） | `threads=128` lane × `per_lane=64` | **已向量化** | `T.Parallel(threads)` + lane 内 `T.serial(per_lane)`（探针 p2f 生成 `for(k=0..32)` 逐线程循环） |
| V3 | 尾 tile 的有效性判定 | 每元素一个谓词 | **已向量化** | `T.if_then_else(i < V − t·tile, ·, 0)`，谓词即掩码（探针 p2c 的 `V=4099` 用例） |
| V4 | 全接受时关掉 draft 读 | 每元素一个谓词 | **已向量化** | `T.if_then_else(n < N, D[·], 0)`，生成分支化条件加载（探针 p2b） |
| V5 | `per` 行循环（`ceil(B/grid)` 行） | ≤5 行 | **保留标量** | **block 级任务映射**：`line = bx + rr·grid` 是核标识，不是逐元素热点（每核 ≤5 次迭代） |
| V6 | tile 循环（`ntile ≤ 19`） | ≤19 次 | **保留标量** | **数据分块控制流**：每轮内部已是 V1 的整块向量 pass；循环体本身零元素级算术 |
| V7 | Philox 十轮 × (N+1) draw | `≤ 2·10·(N+1)` 整数 op ≈ 180（`N ≤ 8`） | **保留标量**（有已验证的向量化形式但不用） | 量化理由：成本 `≈26 ns/核`，占估算下界的 **< 0.3%**；lane 并行形式（探针 p1c）已编译通过，但**共享 key 的 Weyl 递增会被每个 draw lane 各执行一次**（生成码 `if (threadIdx.x < 9) kw[0] += W0` → 9 次自增），必须改为 lane 私有寄存器才能正确——引入竞态风险换 < 0.3% 收益不成立。Stage 4 若需再启用（备选，含私有 key 修法） |
| V8 | 链测试 verdict（`N` 步，每步 2 个 gather + 1 乘 1 比） | `N ≤ 8`（压力用例 600） | **保留标量** | **动态索引 gather**：`D[b,j,id_j]` 的地址由数据决定（每步不同位置），无法用连续向量读替代；`T.Parallel(N)` 也不会让 8 个稀疏 load 变快，反而多一条 op 组 |
| V9 | `n = min_j verdict[j]` | `N` 步 | **保留标量** | **tile 级顺序依赖**：前缀式最小值归约需先有全部 verdict；`N ≤ 8` 时 8 条标量 `min` ≈ 8 ns |
| V10 | 粗层前缀（`tile_sum` 越点） | ≤19 步 | **保留标量** | **顺序依赖（running point）**：前缀不是归约，无法用 `reduce_*` 表达；19 步 ≈ 20 ns |
| V11 | lane 前缀（`lane_sum` 越点） | ≤128 步 | **保留标量** | 同 V10（128 步 ≈ 0.1 µs）。若 Stage 4 需压缩，可改两级（4×32）；当前占比 <1% |
| V12 | 元素走查（命中 lane 的 `per_lane` 元素） | ≤64 步 | **保留标量** | **前缀依赖 + 短路语义**：需要「首个越点」且要按权重顺序跳过零权重，是带谓词的有序扫描；64 步 ≈ 50 ns |
| V13 | 输出写出（`N+1` 个 int32 三态选择） | ≤9 个 store | **保留标量** | **block 级小量标量 + 单点写**：用一个 `T.Parallel(N+1)` 反而多一条 op 组（0.5 µs）换 9 个 store；且 `j<n / j==n / j>n` 三态在单点写里是一条分支链 |
| V14 | `target_at` / `draft_at` / `min(n, N−1)` 地址算术 | 每行 O(1) | **保留标量** | **block 级索引计算**（非逐元素热点） |
| V15 | seed/offset 的 64→32 位拆分 | 每行 2 个字 | **在 host 侧视图完成** | **host 元数据**：`seed.view(torch.int32)` 给出 `[2]`（低位在前），核内只做 `T.reinterpret(·, "uint32")`；避免任何 64 位整数指令（§3.5） |
| V16 | `tile` / `ntile` / `per_core` / `per_lane` 的取值 | host 每个 shape 一次 | **host 元数据** | 编译期常量（工厂参数），不进 kernel |

**向量化结论**：逐元素热点（V1–V4）**已全部向量化**，所用 API 均在本平台有佐证——
`T.SimtVF` + `T.Parallel`（`knowledge/PL-950-row-primitives.md`、`examples/paged_cache_write/_paged_cache_write_kernel/_paged_cache_write_kernel.py`）、
`alloc_reducer`/`reducer_init`/`reducer_update`/`finalize_reducer`（同条目）、以及本次三份探针在
`V=64…151936` 上的编译证据；保留的 8 处标量点（V5–V14）理由类别分别为 **block 级任务映射**（V5）、
**数据分块控制流**（V6）、**收益量化后可忽略且替代方案有已实测缺陷**（V7）、**动态索引 gather**（V8）、
**tile 级顺序依赖**（V9–V12）、**block 级小量标量**（V13–V14），另有 2 处属 **host 元数据**（V15–V16）。
**没有以「实现简单」为由保留的标量循环**；V7 的向量化形式与 V11 的两级前缀压缩作为 Stage 4 备选。

#### 1.6.3 向量化轴与数据布局决策 ⭐（阻塞级）

**I/O layout 是契约**（`draft_probs [B,N,V]`、`target_probs [B,N+1,V]` 行连续，尾轴 `V` 连续），
**核内布局与 lane 映射是设计变量**。按「规约类」清单枚举「水平归约 vs 垂直扫描」两条正交候选，并加上
两个本算子特有的候选：

| # | 候选 | 向量轴 / lane 语义 | 轴长整除性（fp32 向量宽 4、AIV 向量宽 64 lane） | 累加链形态 | repack 代价 | UB 影响 | 评分 |
|---|------|-------------------|---------------------------------------------|-----------|------------|---------|------|
| **L1** | **水平归约（主选）**：lane ↔ tile 内相邻元素，`T.Parallel(tile)` + block reducer | V 内连续轴（尾轴） | `tile` 取 `threads` 的整数倍（8192 = 128×64）→ 尾 tile 浪费 ≤127 个 lane 位（`V=32000` 时 0.4%）；`V=4099` 尾浪费 125 元素 | 每 lane 串行 `per_lane = 64` 步 + `log2(128)` 级全归约 → 向量化良好、跨步系数 1 | 无（直读 GM 行） | `r_ub` 32KB + `tile_sum` 76B + `lane_sum` 512B ≈ 33KB | **选定** |
| L2 | 垂直扫描：lane ↔ 行（一条 lane 扫一整行） | 行轴 | 与 V 无关；`B < 128` 时大量 lane 空闲（smoke `B=2` → 1.6% 利用） | 每 lane 串行 `V` 步（151936）→ 无法向量化 | 无 | 极小（每 lane 一个寄存器） | **弃选**：串行步数 × 151936 直接违反 R3 的复杂度口径；GM 访问步长 `4V` 字节，无合并 |
| L3 | 2-D tile（`R` 行 × `tile` 列，一次处理多行） | V 内连续轴 + 行成批 | 同上；`R·tile ≤ 8192` 才放得下 → `R=2, tile=4096` | 需沿 dim=1 的**二维归约**（每行一个和） | 无 | `R·tile·4` ≤ 32KB | **弃选**：二维 shared→shared 归约在本平台有已确认的输出布局缺陷（`CG-950-simd-shared-reduce` / TRAP-950-simd-shared-reduce，FP32 (2,64) 第二行 sum 与 CPU 不符）；且 `R·tile ≤ 8192` 的 UB 上限只把 op 组数减半（`ntile` 不变、行数成批），收益 < 走备选 B。列为 B2（启用前须先补二维归约取证） |
| L4 | UB 内重排/转置（行搬进 UB 后按列处理） | 重排后的列 | — | 逐元素 | **需一次 UB 读写往返**（`4V` 字节 UB 流量 + 一次 `T.copy`） | `+tile·4` | **弃选**：本算子的归约轴与连续轴本来就一致（尾轴），重排无任何收益（量化：repack 至少 `+tile·4 B` UB 读 + 一次 op 组 0.5 µs，换来 0 收益） |

**评分口径**：L1 的尾 lane 浪费 `≤ threads/2` 个元素/tile（相对 `tile` 是 0.8%），累加链是「lane 内 64 步
连续 + 5 级蝶形」，与 AIV 的 64-lane 向量宽度对齐（`per_lane=64` 使一次 lane 迭代正好一条 64 元素向量
指令宽度）；L2 的串行深度是 L1 的 2374 倍（151936 vs 64），L3 撞已验证缺陷，L4 零收益。

**与下游的一致性**：主选布局（尾轴连续、lane ↔ 元素、`tile = threads·64`）已同步写入 §3.3 伪代码（`tile`
与 `per_lane` 常量、`T.Parallel(tile)` 的尾轴映射）与 §6.1/§6.2 的循环结构；§4.3/§4.5 的缓冲形状
（`r_ub[tile]`、`lane_sum[threads]`、`tile_sum[ntile]`）与本节一致。

**备选 B（多核拆行，= §1.6.0 的 C5，实验裁决模式）**

- **结构完整性**：kernel A（`grid = B·parts`）每核负责一行的一段 `V/parts` 列，写出 fp32 段和到
  `workspace[B, parts]`；kernel B（`grid = B`）读本行 `parts` 个段和做粗层定位，再重读命中段
  （`V/parts` 元素，`parts=2..4`）做 lane/元素两级下降，写两个输出。buffer：A 需要 `r_ub[tile]`（同主选）
  + workspace（`B·parts·4 B ≤ 4KB`）；B 需要 `chunk = V/parts` 的 UB 缓冲（`parts=4, V=151936` → 38KB）。
  循环结构与主选同构，只多一次 launch 与一次 workspace 往返。dtype 路径不变（fp32 概率、int32 输出）。
  分核三要素按新粒度重算：逻辑核 `B·parts`，物理核 56，`parts = clamp(56 // B, 1, 4)`。
- **代价**：流量 `+1/parts`（重读命中段），多一次 kernel launch（约 3–5 µs 量级，未在本型号实测）；
  收益：`B < 56` 时核利用度从 `B/56` 提到接近 1（qwen3 `B=32`：57% → 100%）。
- **实验裁决计划**：代表 shape = `qwen3-235b-eagle-n5 (32,5,151936)`、`llama-8b-b16-n4 (16,4,128256)`、
  `llama-8b-b17-n8 (17,8,128256)`（三者 `B < 56`）；指标 = `msopprof` 的 Task Duration(µs)（同
  `perf_records.jsonl` 口径），A/B 交错采集；**判定阈值**：备选 B 的时延 `< 主选 × (1 − 0.10)` 才翻转
  （10% 抵消多一次 launch 与 workspace 往返的不确定性）；同时记录单核有效带宽 `bytes / (t_us · B)` 以
  反解「每核带宽 vs 聚合带宽」这一未实测常数（用于校准 §1.6.0 的流量项）。`B ≥ 56` 的 workload 不参与翻转
  判定（主选已满载）。裁决结果回写 §1.6.3 与 §1.6.0 的估算行。
- **按 shape 分派的可行性**：kernel 工厂按 `(B,N,V)` 特化，wrapper 已按 `(B,N,V,dtype,device)` 缓存，
  因此「`B < 56` 走 B、`B ≥ 56` 走主选」可作为**每 shape 的分派**实现，不必全局二选一。

**布局决策结论**：选定 **L1 水平归约布局**（尾轴连续、`T.SimtVF(threads=128)`、`tile=min(8192,
ceil(V/128)·128)`、`per_lane = tile/128`、粗层 `tile_sum[ntile]` + 细层 `lane_sum[128]`），弃选 L2
（串行深度 ×2374）、L3（撞 `CG-950-simd-shared-reduce` 且 UB 受限、收益 < 备选 B）、L4（repack 零收益）；
备选 B（两 kernel 拆行）连同三件套（结构 / 代价 / 裁决计划 + 阈值 10%）交 Stage 4，A/B 变体随
DESIGN.md 冻结进入调优清单，并按 `B < 56` 做每 shape 分派评估。

## 2. 编程模式选型

### 2.1 模式结论

**结论：SIMT 向量执行区**——`T.Kernel(grid)`（一维核网格）+ 块内 `T.SimtVF(threads=128)` 打开 lane 域，
逐元素算术用 `T.Parallel` 表达式，块级归约用 `alloc_reducer` 系列，标量/索引计算留在 VF 内的单点守卫。
平台 `programming_modes.migration_default` 是 `simd`，本设计**显式改选 SIMT 并给证据**（见 §2.2）；
若任务要求必须落在 `T.SimdVF()` + `T.simd.*` 上，则 §1.6.3 的「L1-Simd」变体需先补取证（见 §2.3）。

### 2.2 选型理由

对照平台四模式（SIMD / SIMT / Cube / 混合）：

| 模式 | 本算子需要的三件事 | 本平台取证 | 判定 |
|------|------------------|-----------|------|
| **SIMT**（`T.SimtVF(threads=128)` + `T.Parallel`） | ① fp32 逐元素（`max(0,t−d)`、谓词掩码）；② 块级行和；③ 动态行基址的元素级 GM 读 | `knowledge/PL-950-row-primitives.md`（行内逐元素 + 行和 reducer，FP32/FP16/BF16 + 尾块，仿真通过）；`knowledge/numerical-boundaries.md` 的 PL-950-conversion-layout（SIMT 内 `T.cast` 特殊值）；`examples/paged_cache_write/_paged_cache_write_kernel/_paged_cache_write_kernel.py`（动态行基址 + 行谓词 + 分层精度通过）；本次探针 p2a/p2b/p2c/p2d/p2f 在 `V=64…151936`、`B=2…256` 上**编译通过**并生成 `tl::AscendAllReduce`、`max`、`float2` 读 | **选定** |
| SIMD（`T.SimdVF()` + `T.simd.vld/vsts/vcadd/vsub/vmax`） | 同上 | `knowledge/repro/PL-950-simd-reduce.py` 只验证了 FP32 `vld/vcadd/vcmax/vsts/update_mask` 的**归约**形态；本仓**无任何算子用 `T.simd.*` 做逐元素算术**；且高层二维归约另有已确认缺陷（`CG-950-simd-shared-reduce`） | 弃选（取证不足，非能力否定） |
| Cube（`T.Cube` / `T.gemm`） | 无矩阵乘 | — | 不适用 |
| 混合（`T.MixedKernel`） | 无 Cube 段 | — | 不适用 |

1. **默认值的适用面**：`simd` 面向「寄存器级手写向量指令」路径（`T.SimdVF` + `T.simd.*` + `access_ptr`
   地址算术）。本算子的访存形态是「按行基址的元素级读 + 谓词掩码 + 块级归约 + 标量走查」，用
   `T.Parallel` 的多维迭代与谓词表达最直接，而不是逐向量寄存器手写。
2. **取证面（决定性）**：模式选择按实际执行路径，不由默认值决定（`.agents/platforms/toolchains/ascend/workflow.md`：
   「向量操作参考上游 elementwise 的 `T.SimdVF`、`T.simd.*` 或 `T.SimtVF`」「根据实际执行路径选择」）。
   本算子要的三件事在 SIMT 路径都有本平台仿真/编译证据，在 `T.simd.*` 侧逐元素算术无算子级佐证。
3. **不改语义**：两种模式只改指令表达，不改算式、dtype 或分组（§1.6.2 的 V1–V4 与 §1.6.1 的 A1–A6 不变）。

### 2.3 模式影响

- `T.Kernel(grid)` 只有一维核网格、**签名不接受 `threads`**（`references/programming.md`），lane 域必须在
  核内用 `T.SimtVF(threads=…)` 打开；`T.Parallel` **不得出现在任何 VF 区之外**（本次探针实测报错：
  `T.Parallel loop 'j' must be inside a VF block`）。
- 跨 VF 区传递的状态必须是 UB 缓冲（`T.alloc_shared`）或 GM；**`T.alloc_local` 不能作为 VF 区参数**
  （探针实测编译错误：`pointer arguments to simt_vf functions must reside in '__gm__ or __ubuf__'
  address space`）。本设计因此把每行状态（`n`、`target_at`、`draft_at`、`u_token`、`ids`）放在 UB 标量里。
- VF 区内的标量代码由**全部 lane 冗余执行**（编译器的既定行为，探针 p1c/p2a 生成码可见）；因此单点写
  必须用 `if T.get_thread_binding() == 0` 守卫，且标量状态在所有 lane 上保持同值（本文的标量代码分支
  条件只依赖全 lane 相同的数据，不引入发散）。
- 若必须用 SIMD：需要补的证据是「`T.simd.vld → vsub/vmax → vcgadd → vsts` 在 `tile=8192`、谓词掩码、
  动态行基址下的编译与数值」；`PL-950-simd-reduce` 覆盖的寄存器归约形态可作为起点，但**本设计不据此
  切换**（§1.6.3 的 B_simd 需先取证，与主选混用两种执行区未获验证）。

## 3. API 映射设计

### 3.1 公式拆解

以 §1.6.1 优化后的公式为唯一输入（A1–A6 之后）：

| 序号 | 计算 | 优化后公式 | 对应 §1.3 |
|------|------|-----------|-----------|
| P1 | Philox 十轮 | `for step in 10: if step: k += W; (hi0,lo0)=mulhi(c0,M0); (hi1,lo1)=mulhi(c2,M1); c=(hi1^c1^k0, lo1, hi0^c3^k1, lo0)`（A1） | 第 1 步 |
| P2 | uniform | `u = cast32(c0 >> 8) · 2^-24`（`cast32` 是 uint32→float32，值域 `[0,2^24)` 精确） | 第 1 步 |
| P3 | 接受判定 | `verdict_j = if(u_j·D[b,j,id_j] < T[b,j,id_j]) N else j` | 第 2 步 |
| P4 | 前缀长度 | `n = min_j verdict_j`（A2） | 第 3 步 |
| P5 | 残差行基址 | `target_at = (b(N+1)+n)·V`；`draft_at = (bN+min(n,N−1))·V`；`use_d = n < N`（A5） | 第 4 步 |
| P6 | 每 tile 残差和 | `r_i = max(0, T[target_at+t·tile+i] − (use_d ? D[draft_at+t·tile+i] : 0))`；`tile_sum[t] = Σ_i r_i`（A3/A4） | 第 4 步 |
| P7 | 总质量与点 | `total = Σ_t tile_sum[t]`；`point = u_N · total` | 第 5 步 |
| P8 | 粗层 | `k = min{t : Σ_{t'≤t} tile_sum[t'] > point}`（A4） | 第 5 步 |
| P9 | 细层 lane | `lane_sum[l] = Σ_{k<per_lane} r_{k·tile+l·per_lane+k}`；`l* = min{l : Σ_{l'≤l} lane_sum[l'] > left}` | 第 5 步 |
| P10 | 元素 | `τ = min{i ∈ lane l* : Σ_{j≤i} r_j > left 且 r_i > 0}`，否则本行最后一个 `r>0` 的下标（A6） | 第 5 步 |
| P11 | 输出 | `out[b,j] = j<n ? id_j : (j==n ? τ : −1)`；`num_accepted[b] = n` | 第 6–7 步 |

### 3.2 TileLang API 映射

| 计算 | API（`tilelang.ascend.language`） | 参数 / 形态 | 证据 |
|------|-----------------------------------|------------|------|
| 核网格 | `T.Kernel(grid)` → `as bx` | 一维，`grid = min(B, 56)`（编译期常量） | `kernel.py:Kernel`；`references/programming.md`（无 `threads` 参数） |
| lane 域 | `T.SimtVF(threads=128)` | 每行一个（在行循环内） | `frame.py:SimtVF`；探针 p2a/p2d |
| 逐元素 pass | `for i in T.Parallel(tile)` | 尾轴映射，谓词 `i < V − t·tile` | `knowledge/PL-950-row-primitives.md`；探针 p2b/p2f |
| 块级归约 | `T.alloc_reducer((1,), "float32", op="sum")` → `T.reducer_init` → `T.reducer_update(can[0], r)` → `T.finalize_reducer(can, s)` | 每 tile 一次 init/finalize | 同上；探针 p2b 生成 `tl::AscendAllReduce<tl::SumOp,128,1,0>` |
| lane 内累加 | `T.Parallel(threads)` + 内层 `T.serial(per_lane)` + `T.alloc_local((1,), "float32")` | 每 lane 一个寄存器累加器 | 探针 p2f（生成 `for(k=0;k<32;++k) part[0] += ub[…]`） |
| UB 缓冲 | `T.alloc_shared((n,), dtype)` | `r_ub[tile]`、`lane_sum[threads]`、`tile_sum[ntile]`、`ids[N]`、标量 | `allocate.py`；`references/programming.md`（UB = `shared.dyn`） |
| GM 标量读 | 直接索引 `draft_token_ids[line·N+j]`、`D[row·V+id]` | 在 VF 内 / 核内标量 | 探针 p2b（生成带界守卫的条件加载） |
| GM 标量写 | `out[line·(N+1)+j] = …` | thread-0 守卫 | 探针 p2a/p2f |
| 位运算 | `T.reinterpret(x, "uint32")`、`T.uint32(·)`、`&`、`<<`、`>>`、`^`、`*`、`+` | 全 32 位，含回绕 | 探针 p1a（生成纯 32 位 C++ 整数表达式） |
| 选择 / 边界 | `T.if_then_else`、`T.max`、`T.min`、`T.cast` | 谓词掩码与类型转换 | 探针 p2a/p2b/p2c |
| 循环 | `T.serial`（行、tile、draw、走查）+ `T.Parallel`（元素、lane） | 见 §6 | 探针 |
| 无流水注解 | 不写 `T.Pipelined`（§6.3） | 由编译器 AutoSchedule | `references/programming.md`（`allow_autoschedule` 默认开） |

### 3.3 计算伪代码

```python
# 工厂参数（全部编译期常量）：B, N, V, tile, threads, blocks
#   tile   = min(8192, ceil(V/threads)*threads)      # per_lane = tile // threads
#   ntile  = ceil(V/tile);  blocks = min(B, 56);  per_core = ceil(B/blocks)
with T.Kernel(blocks) as bx:
    r_ub     = T.alloc_shared((tile,), "float32")     # 命中 tile 的残差
    lane_sum = T.alloc_shared((threads,), "float32")
    tile_sum = T.alloc_shared((ntile,), "float32")
    ids      = T.alloc_shared((N,), "int32")
    n_s      = T.alloc_shared((1,), "int32")          # 接受前缀长度
    at_t     = T.alloc_shared((1,), "int32")          # target 行基址
    at_d     = T.alloc_shared((1,), "int32")          # draft  行基址
    u_tok    = T.alloc_shared((1,), "float32")        # draw N 的 uniform
    tok      = T.alloc_shared((1,), "int32")          # 被抽 token

    for rr in T.serial(per_core):
        line = bx + rr * blocks
        if line < B:                                   # 行守卫（尾核）
            # ---- 阶段 A：Philox + 链测试（标量，核内一次） ----
            kw = T.alloc_local((2,), "uint32"); cw = T.alloc_local((4,), "uint32")
            bw = T.alloc_local((2,), "uint32"); u  = T.alloc_local((1,), "float32")
            n_s[0] = N
            for j in T.serial(N):  ids[j] = draft_token_ids[line * N + j]
            for j in T.serial(N + 1):                  # 含 draw N
                philox_10(cw, kw, bw, seed_w, offset_w, j, line)     # A1
                u[0] = T.cast(cw[0] >> T.uint32(8), "float32") * T.float32(2.0**-24)  # A1
                if j == N:
                    u_tok[0] = u[0]
                else:
                    if (u[0] * draft_probs[(line*N + j)*V + ids[j]]
                        >= target_probs[(line*(N+1) + j)*V + ids[j]]):
                        n_s[0] = T.min(n_s[0], j)      # A2：首个拒绝
            at_t[0] = (line*(N+1) + n_s[0]) * V
            at_d[0] = (line*N + T.min(n_s[0], N-1)) * V

            with T.SimtVF(threads=threads):
                # ---- 阶段 B：折叠（每 tile 一个 pass） ----
                can = T.alloc_reducer((1,), "float32", op="sum")
                s   = T.alloc_fragment((1,), "float32")
                for t in T.serial(ntile):
                    T.reducer_init(can)
                    for i in T.Parallel(tile):
                        T.reducer_update(can[0], residual(at_t[0], at_d[0], t, i))  # A3/A4/A5
                    T.finalize_reducer(can, s)
                    if T.get_thread_binding() == 0:
                        tile_sum[t] = s[0]
                # ---- 阶段 C：定位（粗层 → 命中 tile → lane → 元素） ----
                pt = T.alloc_local((1,), "float32");  pick = T.alloc_local((1,), "int32")
                pt[0] = T.float32(0);  pick[0] = ntile - 1
                for t in T.serial(ntile): pt[0] = pt[0] + tile_sum[t]
                pt[0] = u_tok[0] * pt[0]                                  # P7
                left = T.alloc_local((1,), "float32"); left[0] = pt[0]
                for t in T.serial(ntile):                                # P8 粗层
                    if pick[0] == ntile - 1:                             # 尚未命中（命中末 tile 时同值）
                        before = left[0]
                        left[0] = left[0] - tile_sum[t]
                        if left[0] < T.float32(0):
                            pick[0] = t; left[0] = before                # 恢复到该 tile 起点
                for i in T.Parallel(tile):                               # 重算命中 tile
                    r_ub[i] = residual(at_t[0], at_d[0], pick[0], i)
                for l in T.Parallel(threads):                            # P9 lane 部分和
                    acc = T.alloc_local((1,), "float32"); acc[0] = T.float32(0)
                    for k in T.serial(per_lane): acc[0] = acc[0] + r_ub[l*per_lane + k]
                    lane_sum[l] = acc[0]
                lpick = T.alloc_local((1,), "int32"); lpick[0] = threads - 1
                for l in T.serial(threads):                              # P9 lane 前缀
                    if lpick[0] == threads - 1:                          # 尚未命中
                        before = left[0]
                        left[0] = left[0] - lane_sum[l]
                        if left[0] < T.float32(0):
                            lpick[0] = l; left[0] = before               # 恢复到该 lane 起点
                if T.get_thread_binding() == 0:                          # P10 元素走查
                    base = pick[0]*tile + lpick[0]*per_lane
                    tok[0] = -1; last = -1
                    for k in T.serial(per_lane):
                        w = r_ub[base + k]
                        if w > T.float32(0):
                            last = base + k
                            if (tok[0] < 0) & (w > left[0]): tok[0] = base + k
                        left[0] = left[0] - w
                    # A6 兜底（罕见）：lane 内无越点元素 → 本 tile 最后一个正权重（一次条件归约）；
                    # point >= total（u·total 舍入到 total）→ 一次全行 max 归约取本行最后一个正权重
                    if tok[0] < 0: tok[0] = last_positive_in_tile(pick[0])
                # ---- 阶段 D：写出 ----
                if T.get_thread_binding() == 0:
                    for j in T.serial(N + 1):
                        output_token_ids[line*(N+1) + j] = T.if_then_else(
                            j < n_s[0], draft_token_ids[line*N + T.min(j, N-1)],
                            T.if_then_else(j == n_s[0], tok[0], -1))
                    num_accepted[line] = n_s[0]
```

折叠 pass 的**形状**：一个 `T.Parallel(tile)` 体内先 `r_ub[i] = residual(·)` 再
`T.reducer_update(can[0], r_ub[i])`（与 `knowledge/repro/PL-950-row-primitives.py` 的「逐元素赋值 + 同行
reducer_update」同形，一个 op 组）。若 Stage 3 发现同一 UB 地址的写后读被插入保守同步、性能不可接受，可拆成
两个 `T.Parallel`（残差写入 + 归约，2 个 op 组/ tile）——数值完全相同，只是 §1.6.0 的发射项要乘 ~1.7。

（`residual(at_t, at_d, t, i)` 是 §1.6.1 A3/A5 的谓词表达式：`i0 = t·tile + i`；
`T.if_then_else(i0 < V, T.max(T[at_t+i0] − T.if_then_else(n<N, D[at_d+i0], 0), 0), 0)`，其中
`at_d` 已用 `min(n, N−1)` 保证地址合法。`last_positive_in_tile` 是**条件执行的一次** `T.Parallel(tile)`
+ `op="max"` reducer（`T.if_then_else(r_ub[i] > 0, i, −1)` 取 max），只在 lane 走查无解时发生——被选中的
tile 之和必然越过 `point`，故该 tile 至少有一个正权重元素，这一次归约必然给出解；`point ≥ total` 的
分支另用一次全行 `op="max"` 归约（`ntile` 个 pass）取本行最后一个正权重，与参考的 `last_positive` 完全一致。
两条兜底都是**概率为零的窗口**（实测 68 行含角点从未触发），因此不进入常规路径的 op 组计数。）

### 3.4 API 可行性确认

| 断言 | 确认方式 | 结果 |
|------|---------|------|
| `T.Kernel` 一维、无 `threads` | 安装包 `ascend/language/kernel.py` + `references/programming.md` | 成立（传 `threads` 会被 Python 拒绝） |
| `T.Parallel` 必须在 VF 区内 | 本任务探针实测 | 成立（VF 外报 `must be inside a VF block`） |
| reducer 三件套可用 | `knowledge/PL-950-row-primitives.md` + 探针 p2b/p2f | 成立（生成 `tl::AscendAllReduce`） |
| `T.alloc_local` 不可跨 VF 边界 | 本任务探针实测 | 成立（`must reside in __gm__ or __ubuf__`）；故状态放 UB 标量 |
| 32 位拆分 mulhi 可编译且只生成 32 位整数指令 | 探针 p1a（生成码逐字检查） | 成立 |
| 完整行流水在 `V=64/300/4099/32000/128960/151936`、`B=2/8/32/256` 上可编译 | 探针 p2a/p2b/p2c/p2d/p2e | 成立（`T.copy` 动态基址在 VF 内**不可**编译 → 元素级读为主选，见 §10 R2） |
| 真机数值 | **未做**（卡被其他算子占用，设计期只做无卡编译） | 待 Stage 3 |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 约束（来源） | 涉及？ | 处理方案 |
|-------------|-------|---------|
| `T.Kernel` 只有一维核网格、无 `threads`（`references/programming.md`） | 是 | 一维 `grid = min(B,56)`；lane 域用 `T.SimtVF(threads=128)` |
| `T.Parallel` 只能在 VF 区内（探针实测） | 是 | 全部逐元素/lane pass 包在 `T.SimtVF` 内；标量代码在 VF 内/核内均可 |
| 跨 block barrier 未认证（`CG-950-batch-barrier`） | 是 | 一行一核、无跨核同步（§0.6 R4） |
| 手工跨核 flag 未认证（`references/programming.md`） | 是 | 不使用；备选 B 用两个 kernel 而不是 flag |
| 二维 shared→shared 归约输出布局缺陷（`CG-950-simd-shared-reduce`） | 是 | 只用一维 reducer 与一维 `T.Parallel`；L3 候选因此弃选（§1.6.3） |
| UB 容量 253952 B/核（平台 `design_budget`）、自动多缓冲膨胀 ~1.7×（`CONST-capacity-910B2C`，910B2C 口径） | 是 | 峰值 33KB，含膨胀 ≤ 57KB（§4.5） |
| L0C 256KB / L1 512KB | 否 | 本算子无 Cube、无 L1/L0 使用（纯 Vector） |
| 对齐：GM 行基址只保证 4 字节对齐（`V=4099`、fixture 的 `storage_offset=1`） | 是 | 元素级 `T.Parallel` 读，**不假设 16B 对齐**；不依赖 DMA 对齐（§10 R2） |
| `T.Parallel` / 区域搬入的 extent 一律用编译期常量 | 是 | `B,N,V,tile,ntile,threads` 全编译期常量；尾块用谓词补 0，不写 `ub[:rem]` 形式（探针 p2e 的动态区域搬入报错，主选不用） |
| fp64 算术 | 是 | 不使用（`references/programming.md` 与 `numerical-boundaries.md` 的算子级证据都只有 fp32/fp16/bf16）；fp64 仅出现在**非公开内部 tensors** `partial` 的源实现里，本设计取消该缓冲 |
| 64 位整数运算 | 是 | 核内不使用：`seed/offset` 由 host 转 `int32` 位视图（§1.6.2 V15）；mulhi 用 32 位拆分（§1.6.1 A1） |
| `T.alloc_local` 不能跨 VF 边界（探针实测） | 是 | 并行状态放 UB（`T.alloc_shared((1,), ·)`），VF 内私有累加用 `T.alloc_local` |

#### 3.5.2 参考实现差异说明

| 差异 | 源实现 | 本设计 | 语义影响 |
|------|--------|--------|---------|
| 前缀精度 | fp64（`seg`/`chunks`/`total`） | fp32 分层 | 仅第 5 步下标；`num_accepted` 不受影响（§1.6.1 A4） |
| 跨 block 归并 | `parts` 个 block + `T.sync_grid` | 一行一核，无归并 | 无（uniform 与链测试本就与切分无关） |
| 行内并行 | 每 block `threads=512/256` + warp 蝶形 | `T.SimtVF(threads=128)` + block 全归约 | 分组变化 → 只影响前缀值（同 A4） |
| 行拆分策略 | `parts = min(row_tiles, sm_count // B)` | `blocks = min(B, 56)`，每核多行串行 | 核利用度差异（§1.6.3 备选 B 处理） |
| 缓存策略 | `L2::evict_first` | 无 | 无（流量不变） |
| 内部 `partial` 缓冲 | `forward` 分配 fp64 `[B·parts]` | **不存在** | 无（非公开输出；wrapper 少一次分配） |
| 元素级读 vs 16B 向量读 | `load16`（要求对齐） | 元素级（任意对齐） | 无（值相同；性能见 §10 R2 备选） |
| golden | GPU `ref_program` | 直接移植同一 torch 实现 | 无（§8.1） |

#### 3.5.3 本项目同类实现参考

| 参考 | 借用的东西 | 不可照搬的部分 |
|------|-----------|---------------|
| `examples/rope/_rope_positions_kernel/_rope_positions_kernel.py` | `T.Kernel(nblocks) as bx` + `T.SimtVF(threads=256)` 的核结构、逐元素 GM 读 + 行谓词、动态行的折叠网格、`T.dynamic` 之外全静态 shape 的写法 | 它的表行 gather 与旋转算式；它的 SIMT/SIMD 取舍只作方法论参考 |
| `examples/paged_cache_write/_paged_cache_write_kernel/_paged_cache_write_kernel.py` | `with T.Kernel(nblocks) as bx, T.SimtVF(threads=128)` 紧凑写法、动态行基址 scatter 的行谓词、`slot < 0` 跳过的分支形态（对应本文的行守卫） | fp8 转换与 page 池布局 |
| `.agents/platforms/toolchains/ascend/kernel_template.py` | `functools.lru_cache` 的 shape 特化工厂、`full/tail` 两段静态尾块写法、`--level L0/all` 的验证骨架 | 单元素 abs 的简单结构 |
| `.agents/platforms/toolchains/ascend/knowledge/repro/PL-950-row-primitives.py` | 行内逐元素 + `alloc_reducer` 的行和模式、尾块填充值的选择（sum 补 0） | 它的 `width=64` 单 tile 规模 |
| `.agents/platforms/toolchains/ascend/knowledge/repro/PL-950-simd-reduce.py` | 只作「SIMD 归约已有验证」的对照（本设计未采用该路径） | `T.simd.*` 的寄存器写法（§2.2 弃选） |

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数 | Shape | dtype | layout / 对齐 | 备注 |
|------|-------|-------|--------------|------|
| `draft_probs` | `[B, N, V]` | float32 | 行连续（尾轴 `V` 连续）；wrapper `contiguous()`；**基址只保证 4 字节对齐**（fixture 会传 `storage_offset=1` 的视图） | kernel 收到一维视图 `[B·N·V]` |
| `draft_token_ids` | `[B, N]` | int32 | 连续 | 值域 `[0, V)`（调用方保证） |
| `target_probs` | `[B, N+1, V]` | float32 | 同 `draft_probs` | 一维视图 `[B·(N+1)·V]` |
| `seed` | `[1]` | int64 | 单独分配（对齐） | host 侧 `view(torch.int32)` → `[2]`（little-endian：`[0]`=低 32 位） |
| `offset` | `[1]` | int64 | 同上 | 同上 |

`B·(N+1)·V ≤ 2^31−1` 是 int32 下标的前提（scaffold 的既有 refusal；最大 workload `256·9·151936 = 3.5e8`
满足）。

### 4.2 输出张量

| 参数 | Shape | dtype | 写入方式 |
|------|-------|-------|---------|
| `output_token_ids` | `[B, N+1]` | int32 | 每行 `N+1` 个标量 store（thread 0），一维视图 `[B·(N+1)]` |
| `num_accepted` | `[B]` | int32 | 每行 1 个标量 store（thread 0） |

输出由 wrapper 预先 `torch.empty` 分配并作为 kernel 实参传入（`out_idx=[]`），与 scaffold 的
`_compiled_fn(...)` 调用形态一致；**不再分配**源实现的 fp64 `partial` 缓冲（§0.6 R4 取消跨核归并）。

### 4.3 中间缓冲区

| 缓冲 | 层级 | Shape / dtype | 大小（`tile=8192`, `threads=128`, `ntile=19`） | 生命周期 |
|------|------|--------------|------------------------------------------|---------|
| `r_ub` | UB | `[tile]` float32 | 32768 B | 命中 tile 的重算 → lane 部分和 → 元素走查 |
| `lane_sum` | UB | `[threads]` float32 | 512 B | lane 部分和 → lane 前缀 |
| `tile_sum` | UB | `[ntile]` float32 | 76 B | 折叠结果 → 粗层前缀 |
| `ids` | UB | `[N]` int32 | 32 B（`N=600` 时 2400 B） | 阶段 A 读入 → 阶段 D 写出 |
| `n_s`, `at_t`, `at_d`, `u_tok`, `tok` | UB | 各 `[1]` | 5 × ≤16 B | 跨阶段标量（VF 边界只能传 UB） |
| reducer 临时区 | UB | `[threads]` float32（编译器分配） | 512 B | 每 tile 的全归约 scratch |
| `acc`/`left`/`pick`/`lpick`/`pt`/`part` | register（`T.alloc_local`） | 各 `[1]` | 0 B（VF 内私有） | 前缀与走查状态 |
| 源实现的 `seg`/`chunks`/`tail`/`prefix`/`lane`/`total`/`left`/`verdict`（fp64，shared） | — | — | **取消** | 由 `tile_sum`/`lane_sum` 承接（§0.6 R1/R2） |

### 4.4 内存搬运路径

```
GM[draft_probs / target_probs 行] --T.Parallel 元素读（VF 内，与 AIV 向量寄存器直连）--> 寄存器
寄存器 --max(0,t−d)--> 寄存器 --reducer_update--> lane 部分和 --tl::AscendAllReduce--> 标量
标量 --thread0--> UB[tile_sum]           （无 GM 往返、无 L1/L0 参与）
GM[命中 tile 的两行] --T.Parallel 元素读--> UB[r_ub]   （一次重算，量 = tile）
UB[r_ub] --每 lane 串行--> UB[lane_sum] --串行前缀--> token --> GM 输出
```

- **无 L1 / L0A / L0B / L0C 使用**：本算子没有矩阵乘，全部数据走 `GM ↔ UB`（AIV 私有）。平台
  `design_budget` 表中的 `L0C 256KB / L1 512KB` 不在本设计的活跃预算内。
- **无 UB→GM 中间写**：`tile_sum`/`lane_sum`/`r_ub` 全在片上闭合，`workspace` 只在 §1.6.3 备选 B
  （两 kernel）里出现。
- **对齐策略**：GM 行基址 `at_t = (line·(N+1)+n)·V` 只保证 4 字节对齐（`V=4099` 为奇数、fixture 故意
  偏移 1 个元素）。因此**不依赖 16B 对齐的 DMA**：读都是 `T.Parallel` 元素级读（探针 p2b/p2c 生成
  带界守卫的标量/`float2` 读），任意 4 字节对齐都正确。写都是 ≤10 个标量 store。
- **无 TMA / 无 nd2nz**：不涉及矩阵布局变换。

### 4.5 UB 内存预算

容量：平台 `design_budget.capacity_bytes.UB = 253952 B`（248 KB/核，来源
`Ascend950PR_9579.ini [AICoreSpec] ub_size`，与型号无关的规则见 `references/programming.md`）。

| 项 | 字节 | 说明 |
|----|------|------|
| `r_ub` | 32768 | `tile=8192` × 4B |
| `lane_sum` + reducer scratch | 1024 | 2 × 128 × 4B |
| `tile_sum` | 76 | `ntile=19` × 4B（最大 V 的 workload） |
| `ids` | 32（`N=600` 时 2400） | `N` × 4B |
| UB 标量 5 个 | 80 | 各自 16B 对齐 |
| **手工峰值** | **33980**（`N=600` 时 36348） | 33.2 KB / 35.5 KB |
| 自动多缓冲膨胀 | ×1.7（`CONST-capacity-910B2C`，910B2C 口径；显式 alloc 结构实测可低至 ×1.12） | 57766 / 61792 |
| **对照上限** | 253952 | 峰值占 **22.7% / 24.3%**（按 ×1.7 最坏口径） → 容量项不构成下界 |

`tile` 的取值上界：`r_ub` 之外还有 1024+76+80+32 ≈ 1.2KB 固定开销，故
`tile ≤ (253952 / 1.7 − 1200) / 4 ≈ 37000` 元素；本设计取 `8192`（Stage 4 可试 `4096…32768`，
每翻倍把 §1.6.0 的发射项减半）。

### 4.6 动态轴定义

- **kernel 内无动态轴**：`B`、`N`、`V` 都是工厂参数（编译期常量），`tile`/`ntile`/`per_lane`/`per_core`
  由它们导出。manifest 的「动态轴 `B, V`」在 wrapper 层体现为**按 shape 特化 + 缓存**（scaffold 的
  `_kernel_cache[(batch,num_draft,vocab,dtype,device)]`，对应源实现的 `@target_cache(maxsize=32)`）。
- **唯一的运行期动态量是数据本身**（行基址 `at_t`/`at_d`、`n`、`total`），它们只影响地址与标量，不影响
  循环上界 → 满足「静态边界」。
- workload 覆盖：`B ∈ {2,16,17,32,64,256}`、`N ∈ {1,2,3,4,5,8,600}`、`V ∈ {64,4099,32000,128256,
  129280,151936}`。

### 4.7 JIT 配置

```python
@functools.lru_cache(maxsize=32)          # 按 (B, N, V, tile, threads, blocks) 特化
def _chain_speculative_sampling_kernel(batch, num_draft, vocab, tile, threads, blocks):
    ...                                    # 返回 @T.prim_func
    return tilelang.compile(main, target="ascend", out_idx=[])
```

- `target="ascend"`、`out_idx=[]`（输出由 wrapper 分配并以实参传入）。
- **不写 `compile_flags`**：源实现的 `csrc_include("streaming_load.h")` 是 nvcc 的预包含（`load16*` 定义），
  本平台没有对应实现（§0.5 #4/#6），随装饰器一并删除。
- 不写 `pass_configs`：使用默认管线（`allow_autoschedule` 开、自动 `InsertSync`、自动多缓冲），见 §7.3。

## 5. Tiling 策略

### 5.1 计算类型

**规约 + 抽样（多步，无矩阵）**：逐元素残差 → 块级归约（每 tile 一个和）→ 分层定位 → 标量写出。
计算全部落在 AIV 向量/标量单元，**无 Cube 段**（§2）。

### 5.2 Block 划分

| 维度 | 取值 | 依据 |
|------|------|------|
| 核网格 | `T.Kernel(blocks)`，`blocks = min(B, 56)` | 一行 = 一个逻辑任务；物理核 56（§5.5） |
| 每核行数 | `per_core = ceil(B/blocks)`（尾核 `line < B` 守卫） | `B ≤ 56` 时 = 1 |
| lane 域 | `T.SimtVF(threads=128)` 每行一个 | 128 与 `per_lane = tile/threads` 整除；reducer 的 `threads=128` 与探针一致 |
| 行内 tile | `tile = min(8192, ceil(V/128)·128)`，`ntile = ceil(V/tile)` | UB 22.7%（§4.5）；`tile` 是 128 的整数倍 ⇒ `per_lane` 恒为整数 |
| tile 内元素 | `T.Parallel(tile)`，尾 tile 用 `i < V − t·tile` 谓词 | 静态 extent + 谓词，不做动态 extent |
| 细层 lane | `T.Parallel(threads)`，每 lane `per_lane = tile/128` 个连续元素 | §1.6.3 L1 |

代表取值：`V=151936` → `tile=8192, ntile=19, per_lane=64`；`V=32000` → `tile=8192, ntile=4`；
`V=128256` → `tile=8192, ntile=16`；`V=4099` → `tile=4224, ntile=1, per_lane=33`；`V=64` →
`tile=128, ntile=1, per_lane=1`。

### 5.3 约束分析

| 约束 | 本设计取值 | 余量 |
|------|-----------|------|
| UB 253952 B | 33.2 KB（×1.7 = 56.4 KB） | 22.7% |
| 静态循环上界 | `per_core ≤ 6`（manifest 最大 `B=256` → 5；fixture `B=300` → 6）、`ntile ≤ 19`、`N+1 ≤ 601`、`per_lane ≤ 64` | 全为编译期常量 |
| `T.Parallel` extent 静态 | 是（`tile`/`threads`/`N` 常量） | — |
| int32 下标 | `B·(N+1)·V ≤ 2^31−1` | 3.5e8 < 2.1e9 |
| 无跨核同步 | 是 | 无需整网格驻留 |
| 非整除处理 | 尾 tile 谓词 + 行守卫（`line < B`） | 见 §5.4 |
| Cube/L0C | 不使用 | — |

### 5.4 注意事项

1. **尾 tile**：`V` 不是 `tile` 的整数倍时（`V=151936 → 19×8192 = 155648`、`V=4099`、`V=32000 →
   4×8192 = 32768`），尾 tile 的有效长度 `rem = V − (ntile−1)·tile` 是编译期常量，用 `i < rem` 谓词把
   无效 lane 的残差补 0（补 0 不改变和，与源实现 `element[0] < full·vec` 的零填充同义）。
2. **`V < threads`**：`V=64` 时 `tile=128`（`ceil(64/128)·128`），`per_lane=1`，一个 tile 覆盖整行；
   `T.Parallel(128)` 的 64 个无效 lane 全部补 0。
3. **行守卫**：`B=256, blocks=56 → per_core=5`，最后一核的第 5 行是 `line = 56·5−1 = 279 > 255` →
   被 `line < B` 跳过（避免越界写输出）；`B=300` 时 `per_core=6`，同样由守卫截断。
4. **`N=600` 的压力用例**：`ids[600]`（2400 B）+ 601 次 Philox + 600 步 verdict 都是标量，在该用例下
   是主导成本（约 601×200 ≈ 1.2e5 条标量指令 ≈ 65 µs/核）；正确性不变，只在 §9 R3 记录。
5. **不能按「对齐物理核整数倍」调 block**：block 的粒度是「一行」，而 `B` 由输入决定，无法通过改
   `block_M` 改变逻辑核数（§5.5 只走「无需适配 / 核内串行」两条）。

### 5.5 分核策略（物理核数适配）⭐

**① 逻辑核数计算**

输出网格是**一维行网格**（每条请求一行、彼此独立，行内不能再拆——行内粗层和留在本核 UB，见 §0.6 R4）：

```
num_logical_kernels = ceil(B / rows_per_block) × 1 = ceil(B / 1) = B
workload: B ∈ {2, 16, 17, 32, 64, 256} → 逻辑核数 2 / 16 / 17 / 32 / 64 / 256
```

**② 物理核数依据（实查，非假设）**

- 目标型号 `ascend-950pr-9579`（`Ascend950PR_9579`，dav-3510）。
- 查询代码（Stage 3 在空闲卡上复核时原样执行）：

```python
import torch, torch_npu
p = torch.npu.get_device_properties(torch.npu.current_device())
print(p.name, p.cube_core_num, p.vector_core_num)   # 期望 28 / 56
```

- 平台记录（`statectl platform --task-dir …` 的 `hardware` 字段，2026-11-23）：
  `observed_cores = {cube: 28, vector: 56}`，来源
  `Ascend950PR_9579.ini [SoCInfo] ai_core_cnt=28, cube_core_cnt=28, vector_core_cnt=56`，
  并附运行时探针 `rtGetAiCoreCount() = 28`（AI Core = 28，AIV = 2×AIC = 56）。
- 本设计用**向量核数 56**（纯 Vector 算子，不用 Cube 核）；`ascend-950pr-9579` 的 `get_device_properties`
  实查值须在 Stage 3 首次真机运行前用上面的代码写入本节（本次卡被其他算子占用，只在无卡模式下做编译
  探针，未做真机查询——这一点如实记录，不拿假设值代替）。

**③ 规模判定与分核方案**

| workload | B | 逻辑核数 | 与 56 比较 | 判定 | 方案 |
|----------|---|---------|-----------|------|------|
| smoke | 2 | 2 | ≤ 56 | 小规模 | **无需适配**：`blocks = B = 2`，一核一行 |
| llama-8b-b16-n4 | 16 | 16 | ≤ 56 | 小规模 | **无需适配**：`blocks = 16` |
| llama-8b-b17-n8 | 17 | 17 | ≤ 56 | 小规模 | **无需适配**：`blocks = 17` |
| qwen3-235b-eagle-n5 | 32 | 32 | ≤ 56 | 小规模 | **无需适配**：`blocks = 32`（利用度 57%，见 §1.6.3 备选 B） |
| llama-8b-b64-n4 / ds-v3-mtp-n1/n3 | 64 | 64 | > 56 | 极大规模 | **核内串行**：固定 `blocks = 56`，`per_core = 2`，第 2 行由 `line < B` 守卫 |
| llama2-7b-b256-n1 | 256 | 256 | > 56 | 极大规模 | **核内串行**：`blocks = 56`，`per_core = 5`（56×5 = 280 ≥ 256） |
| fixture `(300,3,32000)`（精度用例，非 manifest 行） | 300 | 300 | > 56 | 极大规模 | **核内串行**：`blocks = 56`，`per_core = 6`（56×6 = 336 ≥ 300） |
| fixture `(4,600,64)` / `(4,3,64)` | 4 | 4 | ≤ 56 | 小规模 | **无需适配**：`blocks = 4`（成本主导项是 `N=600` 的标量链，§9.1 R3） |

- 「保持逻辑核 = 物理核整数倍」这一选项在这里**不适用**：任务粒度是一行（`rows_per_block = 1`），
  要让核数变成 56 的整数倍只能改「每核行数」，那正是「核内串行」方案本身。
- 核内循环边界：`per_core ≤ 6`（编译期常量，manifest 的 `B=256` 取 5），`line = bx + rr·blocks < B` 是唯一的运行期守卫；
  核内 `T.serial` 嵌套深度 ≤ 5 层（行 → tile → lane → 元素 → 输出），无平台限制冲突。
- 依据：`.agents/skills/_shared/standards/core-split-strategy.md` §1 三要素；平台规则
  `.agents/platforms/toolchains/ascend/workflow.md`（「分核查询使用 `torch.npu.get_device_properties`
  分别记录 Cube/Vector」）。

## 6. 循环与调度结构

### 6.1 循环结构总结

| # | 循环 | 类型 | 上界 | 作用 |
|---|------|------|------|------|
| L1 | `for rr in T.serial(per_core)` | 串行（任务映射） | ≤6（编译期） | 每核处理多行 |
| L2 | `for j in T.serial(N+1)` | 串行（标量） | ≤601 | Philox draw + uniform（含 draw N） |
| L3 | `for j in T.serial(N)` | 串行（标量） | ≤600 | verdict / 首个拒绝 |
| L4 | `for t in T.serial(ntile)` { `for i in T.Parallel(tile)` } | 串行 × **向量** | ≤19 × 8192 | 折叠（V1） |
| L5 | `for i in T.Parallel(tile)` | **向量** | 8192 | 命中 tile 重算（V1） |
| L6 | `for l in T.Parallel(threads)` { `for k in T.serial(per_lane)` } | **向量（lane）× 串行** | 128 × 64 | lane 部分和（V2） |
| L7 | `for t in T.serial(ntile)` | 串行（标量前缀） | ≤19 | 粗层越点（V10） |
| L8 | `for l in T.serial(threads)` | 串行（标量前缀） | 128 | lane 层越点（V11） |
| L9 | `for k in T.serial(per_lane)` | 串行（标量前缀） | ≤64 | 元素走查（V12） |
| L10 | `for j in T.serial(N+1)` | 串行（标量） | ≤9（压力用例 601） | 输出写出（V13） |

**不使用 `T.Pipelined` / `T.Persistent`**：
- `T.Pipelined` 需要显式指定 stage 与 buffer 版本；本算子的 tile 循环**不写任何跨迭代 UB 数据**
  （折叠 pass 只写 `tile_sum[t]` 一个标量，`r_ub` 只在循环外填一次），跨迭代依赖只有一个 reducer scratch，
  由默认 `AutoSchedule` + 自动多缓冲处理即可（§6.3）。
- `T.Persistent` 的语义在本平台由「`blocks = min(B,56)` + 核内行循环」等价表达（§5.5 ③），不需要额外
  机制；`ascend.batch_schedule` 的跨 block barrier 不得使用（`CG-950-batch-barrier`）。

### 6.2 循环伪代码

见 §3.3（同一份伪代码；L1–L10 的嵌套关系即其缩进结构）。要点：
- 阶段 A（L2/L3）在 `T.SimtVF` **之外**（核内标量，每核执行一次），避免 128 个 lane 冗余执行 601 次
  Philox 的整数链；
- 阶段 B–D 在 `T.SimtVF(threads=128)` **之内**（L4–L10），标量前缀由全 lane 冗余执行（同值），单点写用
  `T.get_thread_binding() == 0` 守卫；
- `T.Parallel` 的**内层向量维是尾轴 `i`（V 内连续）**，与 §1.6.3 的主选轴一致。

### 6.3 流水线优化

- 默认 `allow_autoschedule=True` → `AutoSchedule` + `InsertSync` + `MaterializeMultiBuffer` 生效；
  `r_ub` 只在阶段 C 一次填充，自动多缓冲对它的版本化没有收益（也不会被误用超过 2 版）。
- 折叠循环（L4）的迭代之间只有 `tile_sum[t]`（单点写）与 reducer scratch 的复用构成依赖 →
  **Stage 4 可试的项**：给 reducer 显式两版 scratch（或把 `tile` 减半、循环体展开 2 次用两个 reducer），
  让相邻 tile 的装载与归约重叠；判据是 `mte2_ratio ≥ 0.95`（`CONST-copy-floor-method` 的标量削减判定式）。
- 本设计不写 `T.annotate_buffer_versions`（默认自动）；若 Stage 3 发现 `r_ub` 被多缓冲膨胀到 2 版以上，
  用 `{r_ub: 1}` 关闭该 storage 的多缓冲资格（`references/programming.md` 的「`{buf: 1}` 取消多缓冲资格」）。

### 6.4 尾块处理

| 尾块 | 处理 | 代价 |
|------|------|------|
| 尾 tile（`V % tile ≠ 0`） | `i < V − t·tile` 谓词，残差补 0 | 尾 tile 内 `tile − rem` 个无效 lane（`V=151936` 时 3712 个，占 0.4% 总流量） |
| 尾行（`B % blocks ≠ 0`） | `line < B` 守卫整行体 | 0（跳过的行零访存） |
| `V < tile`（smoke、`V=64`） | 一个 tile 覆盖整行，`per_lane = 1` | `tile − V` 个无效 lane（64 个元素与 128 个 lane 的差） |
| `N` 任意（含 600） | `ids[N]`、`N+1` 次 draw、`N` 步 verdict 都是编译期上界 | 无运行期分支 |
| 命中 tile 的部分 lane | `lane_sum` 用同一谓词口径（补 0） | 无 |

## 7. 同步策略

### 7.1 同步模式

**自动同步（默认管线）**：不写显式 `T.sync_threads` / `ascend_*_flag`；由 `InsertSync` 在 UB 写→读边界
插入 `asc_syncthreads()`（探针 p2a/p2b/p2f 的生成码中逐处可见）。本算子**不跨核通信**，因此不存在
需要手工登记的 flag 对（`CG-950-batch-barrier` 与手工 cross-core flag 都不使用）。

### 7.2 同步点说明

| # | 位置 | 为什么需要 | 由谁插入 |
|---|------|-----------|---------|
| S1 | 阶段 A 写 UB 标量（`n_s`/`at_t`/`at_d`/`u_tok`/`ids`）→ 阶段 B 在同一 VF 区内读 | GM→UB 标量写在 lane 间可见 | 自动（`asc_syncthreads`） |
| S2 | 每 tile 折叠 pass 结束、`tile_sum[t]` 由 thread 0 写入 → 下一 tile 复用 reducer scratch | scratch 是同一块 UB | 自动（reducer 前后的同步） |
| S3 | `tile_sum[*]` 全部写完之后读它做粗层前缀 | 跨线程可见 | 自动 |
| S4 | 重算 pass 写 `r_ub` → `T.Parallel(threads)` 读 `r_ub` | 跨线程可见 | 自动 |
| S5 | `lane_sum[*]` 写完 → lane 前缀读 | 跨线程可见 | 自动 |
| S6 | 标量前缀循环里读写 UB 标量（`left`/`pick`） | 编译器的保守同步（探针 p2f 生成码在每步插 `asc_syncthreads`） | 自动 |

**单点写的正确性**：所有 GM/UB 的「单点写」都由 `if T.get_thread_binding() == 0` 守卫；VF 内所有标量
分支条件只依赖全 lane 同值的数据（`tile_sum`/`lane_sum`/`n`），因此不产生 lane 间发散，冗余执行得到的
标量状态在 128 个 lane 上一致。

**前缀状态放在寄存器里**：探针 p2f 的前缀状态用 UB 标量（`T.alloc_shared((1,), ·)`），生成码在 128 步
前缀里每步插一次 `asc_syncthreads`（S6）。本设计因此把 `left`/`pick`/`lpick`/`pt`/`part` 放进 VF 内的
`T.alloc_local`（register）：语义不变（全 lane 冗余执行、同值），且不再每步同步；UB 只在缓冲边界
（`tile_sum`/`lane_sum`/`r_ub`）上被读写，同步点因此只剩 S3/S4/S5 三处。若 Stage 3 实测发现寄存器版
与 UB 版行为有差异，回退到探针已验证的 UB 标量版（多出 S6 的同步代价）。

### 7.3 pass_configs 配置

- **不设置任何 `pass_configs`**：保持默认 `allow_autoschedule=True`、自动 `InsertSync`、自动多缓冲。
- 显式关闭自动调度（`PL-950-manual-sync` 的单核手工配对）**不采用**：本算子无 Cube 段、无跨引擎流水，
  自动调度的保守同步已经足够，手工配对会引入未验证的 flag 依赖。
- 不 `annotate_unlimit_memory`（UB 余量 77%，无需解除容量约束）。

## 8. 验证方案

### 8.1 Golden 函数

**Golden 的唯一依据是 §0.1 的语义**，实现方式是把
`examples/TileOPs/tileops/workloads/sampling.py` 的 `philox_uniform` / `_draw` /
`chain_speculative_sampling` **原样移植**到算子文件（torch CPU/设备无关算术），**不复刻**本文的分块/
流式/分层结构：

（以下为骨架与签名；逐字实现以参考文件的同名函数为准，`verify_equiv.py` 已包含一份可直接运行的同源实现。）

```python
MASK = 0xFFFFFFFF; M0, M1 = 0xD2511F53, 0xCD9E8D57; W0, W1 = 0x9E3779B9, 0xBB67AE85
SEED, OFFSET = 1234, 7                      # workload 固定的 Philox 状态

def golden_mulhilo(a, m):                   # 参考的高/低 32 位（16 位拆分，无 64 位溢出）
    low = (a & 0xFFFF) * m; high = (a >> 16) * m
    mid = low + ((high & 0xFFFF) << 16)
    return ((high >> 16) + (mid >> 32)) & MASK, mid & MASK

def golden_philox_uniform(seed, offset, rows, draws):   # [rows, draws] float32 in [0,1)
    ...                                     # counter=(j, r, off_lo, off_hi)，key=(seed_lo, seed_hi)，十轮
    return (c0 >> 8).float() * 2.0**-24

def golden_draw(weights, u):                # [B,V] × [B] -> [B] int32
    cdf = weights.double().cumsum(-1)                        # fp64 左到右（语义基准）
    point = u.double() * cdf[:, -1]
    index = torch.searchsorted(cdf, point[:, None], right=True).clamp(max=V - 1)
    ...                                     # 零权重跳到下一个正权重；越界取最后一个正权重

def golden_chain_speculative_sampling(draft_probs, draft_token_ids, target_probs, seed, offset):
    """返回 (output_token_ids [B,N+1] int32, num_accepted [B] int32)。"""
    # 逐步照 §1.3：u → accept → num（cumprod.sum）→ 残差行 → golden_draw → 三态输出
```

**独立性要求**：golden 只用 `(seed, offset, B, N, V, 输入张量)`，不出现 `tile`、`lane_sum`、`blocks`、
`per_core` 等本文结构量；它可以在 CPU 上独立跑通（`verify_equiv.py` 已在 CPU 上用它做对照）。

### 8.2 精度标准

| 输出 / 用例 | 标准 | 依据 |
|-------------|------|------|
| `num_accepted` | **逐位精确**（`exact_compare`） | `tests/ops/test_sampling.py`；§1.6.1 A2 |
| `output_token_ids` 的接受前缀（`j < n`） | 逐位等于 `draft_token_ids` | `verify_chain_outputs` 的断言 |
| `output_token_ids` 的 `−1` 尾（`j > n`） | 逐位等于 `−1` | 同上 |
| 被抽 token（`j == n`） | 必须在该位置残差行上**权重为正**；整列分布通过 6σ 检验（`assert_chain_distribution`） | 同上 + §1.6.1 A4/A6 |
| 全部 token 的位置 | `−1` 恰好在被抽 token 之后 | 同上 |
| 同一输入两次调用 | 逐位一致（可复现） | `tests/ops/test_sampling.py` 的第二次调用断言 |
| 对齐无关性 | `storage_offset = 1` 的输入与对齐输入**逐位一致** | 同文件的 smoke 用例；本设计的元素级读不依赖对齐（§4.4） |
| dtype 守卫 | `draft_probs` 为整数 dtype 时 wrapper 抛 `ValueError`（含 `float32`） | 同文件；属 wrapper 行为，kernel 不参与 |
| 分布（65536 行 × `V=16`） | `assert_chain_distribution` 的 6σ 界 | 同文件 |

### 8.3 L0 门槛测试计划

| 层 | shape（B, N, V） | 覆盖的结构点 | 断言 |
|----|-----------------|-------------|------|
| **L0** | (2, 2, 64)（smoke，同 manifest 首行） | `blocks=2`、`tile=128`、`ntile=1`、`per_lane=1`、一核一行 | `num_accepted` 逐位；结构（`verify_chain_outputs`）；两次调用逐位一致 |
| **L0b** | (4, 3, 64) | 一行 = 一核、`N=3`（`n ∈ {0..3}` 都可能出现） | 同上 |
| L1 | (8, 3, 4099) | `V` 非 4 的倍数（行基址 4 字节对齐）、`tile=4224`、`ntile=1`、`per_lane=33` | `num_accepted` 逐位 + 结构 + 分布（小样本） |
| L1 | (2, 3, 151936) | `ntile=19` 的完整流程（粗层 >1 tile）、`per_lane=64` | 同上 |
| L1 | (64, 4, 128256) | `B > 56` → `blocks=56, per_core=2`（核内串行 + 行守卫） | 同上 |
| L2 | (300, 3, 32000)、(4, 600, 64)、(256, 1, 32000)、(32, 5, 151936) | 全部 workload 形状 + `N=600` 的压力链 | 同上 |
| boundary | (2, 2, 64) 的两行概率移到 `storage_offset=1` | 对齐无关性（元素级读） | 与对齐输入逐位一致 |
| boundary | `draft_probs` 转 int32 | wrapper 拒绝路径 | `ValueError` 含 `float32` |

执行方式：本平台 `execution.precision_command`（`<kernel> --level L0|L1|L2|boundary|all`，工作目录为任务
目录）。L0 通过才进 L1/L2；任一失败保留日志与最小复现（Stage 3 契约）。

## 9. 风险点与注意事项

### 9.1 已知约束

| # | 风险 | 影响 | 缓解 / 判据 |
|---|------|------|-----------|
| R1 | **真机未验证**：本次卡被其他算子占用，只做了无卡编译探针（`ASCEND_RT_VISIBLE_DEVICES=""`），reducer 的数值、`asc_syncthreads` 的开销、标量前缀循环的实际代价都未实测 | 若真机行为与编译产物不符，可能是失败面最大的地方 | Stage 3 先在 L0（(2,2,64)）上跑通，再上 L1；探针生成码已逐条核对（`tl::AscendAllReduce<SumOp,128,1,0>`、`float2` 读、thread0 守卫） |
| R2 | **元素级 GM 读的带宽**：不依赖 16B 对齐（`V=4099`、`storage_offset=1`），因此主选走 `T.Parallel` 元素读；编译器是否把它向量化到 128 位取决于可证明性（探针里只见到 `float2`） | 若单核有效带宽远低于 §1.6.0 的 1.26 TB/s 假设，时延会高于估算下界 | ① 估算已按聚合地板口径标注为「外推」；② Stage 4 备选：在 host 侧判 `(data_ptr % 16 == 0) and (V % 4 == 0)` 时编译一个 `T.copy` 区域搬入变体（两条路径的**数值路径完全相同**——读机制不改变 `r_ub` 的内容与归约顺序，故仍满足「对齐无关逐位一致」）；③ 若编译器不向量化，评估 `T.simd.vld`（需先补取证，§2.3） |
| R3 | `N=600` 的压力用例：601 次 Philox + 600 步 verdict 全是标量，约 65 µs/核 | 只影响该用例的时延（不在 manifest 的 8 个 workload 里，属 fixture 压力用例） | 正确性无影响；若 Stage 4 需优化，可把 draw 循环 lane 化（V7 的备选，须用 lane 私有 key） |
| R4 | 发射项（`ntile+2` 个 op 组 × 0.5 µs）在 `B ≤ 56`、`V ≤ 128256` 的 workload 上压过流量项 | 估算下界偏乐观的项会变成发射 | `tile` 从 8192 起可按 UB 上限翻倍到 32768（`ntile` 减半→发射项减半）；`tile` 是 Stage 4 的第一杠杆 |
| R5 | fp32 分层前缀 vs 参考 fp64：理论上可改变被抽 token 的下标 | 不影响 `num_accepted`；只影响「抽到哪个 token」 | §1.6.1 A4/E5 已机器验证（68 行 0 分歧、零权重命中 0、分布逐 bin 相同）；Stage 3 的分布用例（65536 行）作为第二道闸 |
| R6 | 编译器的界守卫（生成码里的 `0 <= … && … < total` 条件加载）在越界地址上不发访存，但**依赖 `min(·, V−1)` 与 `min(n, N−1)` 的钳位**才能保证地址合法 | 若 Stage 3 去掉了钳位，可能出现越界读（读相邻行，结果错但不崩） | 设计把两处钳位写成硬要求（§3.3 的 `residual` 宏与 `at_d`）；`draft_token_ids` 的 `[0,V)` 是调用方保证 |
| R7 | 单点写守卫遗漏（VF 内标量写由全 lane 执行） | 128 个同值 store → 语义正确但浪费；若写成非幂等操作则会错 | §7.2 的规则：所有 GM/UB 单点写必须 `thread 0` 守卫；标量状态必须全 lane 同值 |
| R8 | `T.alloc_local` 被误传到 VF 区外（或跨 VF 边界） | 编译失败（探针实测 `must reside in __gm__ or __ubuf__`） | 跨阶段状态一律 UB 标量；VF 私有累加才用 `T.alloc_local` |
| R9 | 自动多缓冲把 `r_ub` 膨胀到 2 版 + 平台另有 1.7× 膨胀口径 | UB 峰值从 33 KB 涨到 ~112 KB（仍 < 253952） | 若报 UB overflow，用 `annotate_buffer_versions({r_ub: 1})` 固定单版（§6.3） |
| R10 | `V < threads`（`V=64` → `tile=128`）时 `lane_sum` 有一半 lane 的全零段 | 无（和不受影响） | 已在 §5.4 记录；`lane` 前缀在 `left>0` 时不会选到零段的末尾（零段不会被越过） |
| R11 | 两条兜底规则（tile 内最后正权重 / 全行最后正权重）与参考在 1e-7 的舍入窗口内可能不同 | 只影响「抽到哪个 token」（且该 token 仍带正权重） | 已在 §3.3 注明；E5 的角点（`u=0`、`u=(2^24−1)/2^24`、首尾补零）覆盖 |
| R12 | 源的 `autotune_accepts_random_int_inputs = False`（`draft_token_ids` 越界会读越界） | autotune 若用随机 int 输入会读到行外 | 本任务不做 autotune（scaffold 的 `tune` 已在 NPU 侧移除）；Stage 4 若加 autotune 必须带合法 id 生成器 |

### 9.2 常见错误

1. **把 `T.Parallel` 写到 `T.SimtVF` 外面** → 编译期报 `must be inside a VF block`（本次探针首版即撞）。
2. **把 `T.alloc_local` 传给 VF 区** → 编译期报 `pointer arguments to simt_vf functions must reside in
   '__gm__ or __ubuf__'`（本次探针第二版即撞）。跨阶段状态必须 UB。
3. **用 `(int64)a·m >> 32` 算 mulhi** → 有符号溢出 + 算术右移，高位错（§1.6.1 A1 的陷阱记录）。
4. **在 lane 并行形式里共享 key** → `kw[0] += W0` 被每个 lane 各执行一次（探针 p1c 生成码可见）；
   Philox 的 key/counter 必须在 lane 私有寄存器里。
5. **忘记尾 tile 谓词** → 尾 tile 读到下一行（`V=4099` 时最明显），`num_accepted` 可能仍对但 token 错。
6. **`T.copy` 在 VF 区内做动态基址区域搬入** → 本次探针 p2e 编译报错（`allow_append_ is false`）；
   主选用元素级读绕开。
7. **在 `T.Parallel` 里用动态 extent**（`ub[:rem]`）→ TIR 要求静态 extent；用谓词补 0 代替。

### 9.3 特殊场景处理

| 场景 | 处理 |
|------|------|
| `n = N`（全部接受） | 不读 draft 行（A5）；残差 = target 第 `N` 行；`at_d` 仍用 `min(N, N−1)` 保持地址合法（但不发访存） |
| `n = 0`（首个 draft 就拒绝） | 残差 = max(0, `T[b,0]` − `D[b,0]`)，正常流程；`output_token_ids` 第 0 位就是被抽 token，其后全 −1 |
| 残差行全零（`target ≡ draft`） | 契约上不会发生（行归一 ⇒ 残差和 > 0 否则两行相同则 accept 必真→ `n=N` 且残差 = target 行 > 0）；若发生，`total = 0` ⇒ `point = 0` ⇒ 走「本行最后一个正权重」兜底（无正权重时取 0 号位，参考实现同样 `clamp(max=V−1)` 后取到 0） |
| `u = 0` | `point = 0` ⇒ 命中第一个正权重下标（与参考的 `searchsorted` 一致，E5 角点覆盖） |
| 零权重元素 | 永不被选中（走查条件含 `w > 0`） |
| `V` 非 4 的倍数 / 基址 4 字节对齐 | 元素级读，无对齐假设（§4.4） |
| `B·(N+1)·V > 2^31−1` | wrapper 的既有 refusal 抛错（scaffold 保留） |
| `dtype != float32` / `seed|offset != int64` | wrapper 的 `_validate` 抛 `ValueError`（scaffold 保留） |

## 10. 交付清单

### 10.1 目录结构

```
examples/chain_speculative_sampling/_chain_speculative_sampling_kernel/
├── DESIGN.md                                  # 本文档（Stage 1 交付）
├── verify_equiv.py                            # §1.6.1 的等价性机器验证（15/15 EQUIV_PASS，34s）
├── RETROSPECTIVE.md                           # Stage 1 复盘（Phase 8）
└── history_version/
    ├── design_probe_philox.py                 # 探针 #1：Philox 的三种表达式（p1a/p1b/p1c）
    ├── design_probe_rowfold.py                # 探针 #2：行折叠/定位结构（p2a–p2g）
    └── probe_p1a_gen.txt, probe_p1b_gen.txt, probe_p1c_gen.txt,
        probe_p2a_gen.txt … probe_p2f_gen.txt  # 各探针的设备代码（证据）
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | 本阶段产出 | 冻结后进入 Stage 2 检视 |
| `verify_equiv.py` | 本阶段产出 | Stage 2 维度 8 会重跑复核 |
| `history_version/design_probe_philox.py` | 本阶段产出 | 探针 #1（≤10 min，实测 ~10 s） |
| `history_version/design_probe_rowfold.py` | 本阶段产出 | 探针 #2（≤10 min，实测 ~60 s） |
| `_chain_speculative_sampling_kernel.py`（Stage 3） | 待 Stage 3 | kernel 工厂 + golden + 分层测试；实现位置是 scaffold 的 `examples/TileOPs/tileops/kernels/sampling/chain_speculative_sampling/chain_speculative_sampling.py` 内被重写的工厂（替换导入的 GPU 实现） |
| `RETROSPECTIVE.md` | 本阶段产出 | 复盘章节（追加写） |

### 10.3 命名规范

- 工厂：`_chain_speculative_sampling_kernel(batch, num_draft, vocab, tile, threads, blocks)`（对齐源实现的
  工厂签名风格，去掉 `vec/parts/pace`，换成 NPU 的 `tile/blocks`）。
- kernel 内：`tile_sum`、`lane_sum`、`r_ub`、`n_s`、`at_t`、`at_d`、`u_tok`、`tok`、`ids`（与 §3.3 一致）。
- golden：`golden_chain_speculative_sampling` / `golden_philox_uniform` / `golden_draw`（§8.1）。
- 测试入口：`run_L0` / `run_L1` / `run_L2` / `run_boundary` / `main --level`（平台 kernel 模板的既有约定）。

### 10.4 实现顺序

1. **骨架**：`T.Kernel(blocks)` + 行循环 + 空 VF 区 + 输出写出（只写 `ids`/`−1`/`num_accepted=0`），
   在 L0 上跑通「结构与输出 shape」。
2. **Philox + 链测试**（阶段 A）：接 `seed/offset` 的 int32 视图，用 `verify_equiv.py` 的 E1/E2 在同一
   seed 下对拍（先在 CPU 上验证算法，再把常量搬进 kernel），L0 上断言 `num_accepted` 逐位。
3. **折叠**（阶段 B）：加 `residual` 宏 + reducer，先把 `tile_sum` 写回 GM 做临时调试输出，逐位对拍
   「每 tile 和」与 golden 的同分组和（可选，定位用）。
4. **定位**（阶段 C）：粗层 → 重算 → lane → 元素走查 → 两条兜底。
5. **分层测试**：L0 → L0b → L1 → L2 → boundary；每层先 `--level L0` 再 `all`。
6. **收尾**：删调试输出、确认无处方的 `T.copy`/`T.alloc_local` 跨边界、跑 `verify_equiv.py` 复核。

---

## 附：相对上一版的关键调整

n/a（首次设计，无 `design_error_summary` 与历史备份）。
