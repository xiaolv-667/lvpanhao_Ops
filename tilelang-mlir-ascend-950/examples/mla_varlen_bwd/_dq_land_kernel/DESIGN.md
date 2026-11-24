# _dq_land_kernel 算子设计文档（packed 分支 / `MLAVarlenBwdOp` 第 3 个 launch）

> **迁移任务（harness 子模式），首次设计 v0**。源算子：
> `/workspace/TileOPs/src/tileops/_kernels/attention/mla/bwd.py` 第 79–106 行的 `_dq_land_kernel`
> （GPU TileLang 实现）。目标平台 `ascend-950pr-9579`（dav-3510，CANN 9.2.1，TileLang ascend 0.1.15），
> 编程模式按平台 `programming_modes.migration_default = simd` 选定 **SIMD**（§2），
> `import tilelang.ascend.language as T`。
>
> **本函数与 `MLADenseBwdOp` 的 `_dq_land_kernel` 逐字相同**（证据链见 §0.0）：源码里**只有一个**
> 模块级定义（第 79–106 行），`MLADenseBwdKernel` 与 `MLAVarlenBwdKernel` 都经
> `_MLABwdKernelBase._compile`（第 354 行）拿同一个工厂、在 `_run` 的同一行（第 346 行）调用
> `dq = land(dq_acc)`。它**没有** `dense_seqlen` 一类分支参数（对比同文件的
> `_delta_kernel(batch, heads, dim_v, dense_seqlen, dtype)`），因此 packed / fixed-length 的区分
> **根本不进入**这个工厂。
>
> **复用基准声明（本设计最重要的一条前提）**：本设计的复用基准**不是** dense 的 `DESIGN.md` 初版，而是
> **dense 已通过真机的实现** `examples/mla_dense_bwd/_dq_land_kernel/_dq_land_kernel.py`
> （L0/L1/L2/boundary 共 31 个用例全 PASS，`max_diff=0` / `bit_diff=0`）与其 Stage 3 实证。
> 原因是 dense 设计文档初版 §3.2/§3.3 写的 `vsts(..., "NORM_B16")` + `pset(16, "PAT_VL64")` 形态在真机上
> **只写了一半列**（见 §9.2 E1 与 `examples/mla_dense_bwd/_dq_land_kernel/repro/CONST-950-vcvt-narrow-store-pack-b32.py`
> 的实测 256/256 vs 128/256 vs 4/256），实现阶段改成了 `dist="PK_B32"` + `pset(32, "PAT_ALL")`。
> **本文只承载真机验证过的形态**，不复述已被证伪的初版写法。
>
> **设计期证据（本机无卡编译，`ASCEND_RT_VISIBLE_DEVICES=""`，从不 launch；加速卡正被另一算子的
> Stage 3 占用）**：
> - 平台配置实查 `python3 .agents/tools/statectl.py platform --task-dir examples/mla_varlen_bwd/_dq_land_kernel`
>   → `hardware_id = ascend-950pr-9579`，`capacity_bytes` UB 253952 / L1 524288 / L0C 262144 / L0A 65536 /
>   L0B 65536，`observed_cores` cube 28 / vector 56，`programming_modes.migration_default = simd`，
>   `cost_measurements` **为空**（本型号在知识库里没有实测带宽常数）。
> - 编译探针（本任务目录新增，D-3，1 个）：`history_version/design_probe_packed.py` → fp16 / bf16
>   **双 dtype COMPILE OK**，设备码各 41 行，含 `vlds_norm` / `vmuls` / `vcvt` / **`vsts_pack_b32`**；
>   快照 `history_version/probe_packed_gen_float16.txt`（sha256 前缀 `d0fde4920bf2c7ac`）与
>   `history_version/probe_packed_gen_bfloat16.txt`（`914c64f7abb2f600`）；`total_q` 在设备码里是
>   **kernel 形参**而非常量（证明 `T_q` 不参与特化）。
> - 等价性机器验证（本任务目录，torch CPU + fp64 参照）：
>   `examples/mla_varlen_bwd/_dq_land_kernel/verify_equiv.py` 已执行 → A1/A2/A3 全 `EQUIV_PASS`
>   （0 ulp / 0 违反），C1/C2/C3 三个对照项全被证伪，结果表见 §1.6.1。
> - 复用证据（非本次重跑）：dense 真机 31/31 与两项真机 repro 脚本（第三条实证——`out_idx=[1]` 动态输出可用——
>   由 dense 的 L0-1 覆盖）
>   `examples/mla_dense_bwd/_dq_land_kernel/repro/CONST-950-vcvt-narrow-store-pack-b32.py`、
>   `examples/mla_dense_bwd/_dq_land_kernel/repro/CONST-950-vcvt-rs-disable-ieee.py`。
>
> 三个 launch 的契约：`_delta_kernel`（head-major fp32 `delta`）→ `_bwd_kernel`（写 dk/dv、向 fp32
> `dq_acc` 原地累加）→ **`_dq_land_kernel`（本函数：`dq = dq_acc * scale` 并 cast 落盘）**。本文只设计第 3 个；
> 上下游契约不可改动：`dq_acc` 由 `_bwd_kernel` 以 fp32 写好，shape `[T_q, H, D_qk]`、行主序连续。

## 复用分区总览

**复用（引用 dense 真机证据，本文不再重推）**

| # | 复用项 | 证据 |
|---|---|---|
| U1 | 一维 `T.Kernel(56)` 持久网格 + `it → (h, t0)` 解码 + `cpt` 连续区间 | dense §0.6 RD1 / §5.5；真机 31/31 |
| U2 | 两级流水 `T.Pipelined(..., num_stages=2)` + `T.annotate_buffer_versions({acc: 2, out: 2})` | dense §6.3；真机同几何连跑 20 次逐位一致，且单版本对照逐位一致 |
| U3 | VF 段链 `vld → vmuls(scale) → vcvt(ROUND_R, sat=False) → vsts` | dense 实现第 199–207 行；真机 `bit_diff=0` |
| U4 | **落盘形态 `dist="PK_B32"` + `pset(32, "PAT_ALL")`**（真机实证，取代初版 `NORM_B16`） | `examples/mla_dense_bwd/_dq_land_kernel/repro/CONST-950-vcvt-narrow-store-pack-b32.py`：256/256 正确、0 sentinel |
| U5 | **`vcvt(sat=False)` 即 IEEE**（65519→65504、65520→±inf；`sat` 对本转换对惰性） | `examples/mla_dense_bwd/_dq_land_kernel/repro/CONST-950-vcvt-rs-disable-ieee.py`；本目录 `verify_equiv.py` A3 |
| U6 | 运行期行 clamp `n = min(R, T_q − t0)` 同时约束两条 `T.copy` 与两侧切片 | dense §0.6 RD3；设备码快照 `history_version/probe_e15_gen.txt` |
| U7 | `D_qk = 192 = 3×64` 满 lane、`NSEG = 3`、`R = 64`、UB 147456 B（58.1%） | dense §4.5 / §5.2 |
| U8 | A1–A5 五项数学等价优化 | dense §1.6.1；本目录 `verify_equiv.py` 全 `EQUIV_PASS` |
| U9 | `out_idx=[1]` JIT 动态输出 + `total = T.dynamic("total_q")` 符号复用 | 真机 L0-1（dense）；本目录探针复现同一形态 |
| U10 | 编译后**设备码断言**（防计算体静默丢弃）：`vlds_norm`/`vmuls`/`vcvt` 且 `vsts_pack_b32`/`vsts_norm` | dense `_simd_body_gate`；本目录探针 `[PASS]` |

**packed 需复核（逐项结论，证据见 §11）**

| # | 复核项 | 结论 |
|---|---|---|
| P1 | 调用路径不做 view（packed 张量直接进 `_run`；dense 侧有 `view`） | **无差异**（kernel 只读 `dq_acc`，而 `dq_acc = torch.zeros(q.shape, …)` 恒连续） |
| P2 | `T_q = Σ_b q_len[b]` 的来源与量级（dense 为 `B·S_q`） | **无差异**（运行期符号，不参与特化；设备码 `total_q` 为形参） |
| P3 | 输出契约 `dq [T_q, H, D_qk]`（无 batch/seq 维） | **无差异**（与本 kernel 的 `out_idx` 输出同 shape；dense 侧多一次 no-op `view`） |
| P4 | `cu_seqlens_q` / `cu_seqlens_kv` 是否进入本 launch | **无差异**（不进入：`land(dq_acc)` 只收一个张量） |
| P5 | `scale` 特化（packed 测试含 `sm_scale = 0.05` 用例） | **无差异**（工厂键含 `scale`，与 dense 同一机制） |
| P6 | dtype 契约（fp16 / bf16，`dim_qk % 16 == 0`） | **无差异**（两条路径共享 `_MLABwdKernelBase.refusal` / 同一实现契约） |
| P7 | 分核（逻辑 tile 数按 packed workload 重算） | **结构无差异**，数值重算（§5.5） |
| P8 | 退化 packed 几何（如 `q_lens=[64]` → 4 个逻辑 tile < 56 核） | **无差异**（空区间零迭代，与 dense 的 `T_q` 极小情形同构） |
| P9 | 数值路径（fp32 乘 + 单次 RNE、非饱和） | **无差异**（§1.6.1 机器验证，本目录独立重跑） |

> **一句话结论**：本 kernel 的编译产物是 `(heads, dim_qk, scale, dtype)` 的函数，`T_q` 只在运行期绑定；
> packed 与 fixed-length 的全部差异落在「调用方是否 view」与「`T_q` 的数值」上，
> **`_dq_land_kernel` 本体无差异**。本文档为「几乎全量复用」，差异只体现在设计与验证的**覆盖对象**
> （packed 几何、多请求、ragged 长度）上，而不是实现结构上。

## 修订说明

本版为 `_dq_land_kernel`（packed）的**首次设计**（`first_design`，非 revision），无「相对上一版的关键调整」条目。
其复用的 dense 设计已通过检视与真机（31/31）；本版相对 dense **实现的**唯一增量是：新增本任务目录的编译探针
（`history_version/design_probe_packed.py`）与等价性脚本，用它们把「packed 路径无差异」从推理变成本目录内的
可复现证据，并按 packed workload 重算复杂度、分核与 L0 计划。

---

## 0. 源算子解读与迁移分析

### 0.0 复用判定：本函数与 dense 版逐字相同（证据链）

| # | 证据 | 内容 |
|---|---|---|
| 1 | **源码只有一个定义** | `/workspace/TileOPs/src/tileops/_kernels/attention/mla/bwd.py` 第 79–106 行定义了唯一的模块级 `_dq_land_kernel`；两个 Kernel 类都在 `_MLABwdKernelBase._compile`（第 354 行）里调用它，`_run`（第 346 行）里调用 `dq = land(dq_acc)`。第 358 行的 `MLADenseBwdKernel` 与第 382 行的 `MLAVarlenBwdKernel` 的区别只在 `forward`（第 361–379 行 vs 第 385–387 行），**都在 `_run` 之前/之后**，不进本工厂。 |
| 2 | **抽取文件函数体逐字相同** | `examples/TileOPs/tileops/kernels/attention/mla_dense_bwd/_mla_dense_bwd_kernels.py` 与 `examples/TileOPs/tileops/kernels/attention/mla_varlen_bwd/_mla_varlen_bwd_kernels.py` 中 `_dq_land_kernel` 的函数体 `diff` 为空（两文件的差异只在模块头注释与 `GroupTiling` docstring）。 |
| 3 | **工厂签名无 packed 分支参数** | `_dq_land_kernel(heads, dim_qk, scale, dtype)`；对比 `_delta_kernel(batch, heads, dim_v, dense_seqlen, dtype)` 与 `_bwd_kernel(..., dense_seqlen_q, dense_seqlen_kv, ...)`——packed 信息在这两个 launch 里以参数形式存在，在本 launch 里**不存在**。 |
| 4 | **编译产物与 `T_q` 无关** | 探针设备码 `history_version/probe_packed_gen_float16.txt` 第 16 行：`_land_kernel(__gm__ half* dq, __gm__ float* dq_acc, int32_t total_q)`；第 23 行 `ntiles = ((total_q + 63) >> 6) * 8` 运行期算；第 25 行 `asc_copy_gm2ub_align(..., min(64, (total_q - ((it >> 3) * 64))), 768, ...)`。没有任何行数被烘进常量。 |
| 5 | **packed 调用路径的唯一差别** | dense：`q.view(B*S_q, H, -1)` 进 `_run`，返回 `dq.view(q.shape)`（源码第 370/379 行）；packed：`q` 已是 `(T_q, H, D_qk)`，直接进 `_run`，`dq` 直接返回（源码第 385–387 行）。`_run` 内部两者逐字相同（第 338–347 行）。 |

**推论**：既然两个任务喂给同一工厂的是同一份代码、且该代码的编译产物与 `T_q` 及调用方 view 无关，
则 **NPU 侧的迁移决策、算法、优化、Tiling、循环、同步全部可以整段承接 dense**，本设计的实质工作是
(a) 把承接的内容按本文档标准复述并逐条给出本任务内可核对的证据，(b) 复核 packed 调用路径是否引入差异
（§11 九项，结论全为「无差异」），(c) 按 packed workload 重算复杂度、分核与验证计划。

### 0.1 源算子语义（做什么）

`_dq_land_kernel(heads, dim_qk, scale, dtype)` 是 MLA prefill 反向的**第三段也是最后一段**
（源 docstring：“``dq = dq_acc * scale`` in the input dtype”）：把 `_bwd_kernel`（key-stationary，跨 CTA 以
fp32 累加的 dQ 部分和）留在 GM 里的 fp32 累加缓冲，乘上 softmax 的 `scale` 并按输入 dtype 落盘。

**数学语义**（逐元素映射，无规约、无跨元素依赖）：

```
dq[t, h, d] = to_dtype( fp32(dq_acc[t, h, d]) · float32(scale) )
   t ∈ [0, T_q)  —— 动态轴；packed 下 T_q = Σ_b q_len[b]（cu_seqlens_q[-1]）
   h ∈ [0, H)    —— 头，编译期常量（packed manifest 取 8 / 128 / 64）
   d ∈ [0, D_qk) —— 编译期常量 192
```

**语义要素**（逐项由源码核对，非臆测）：

- **规约语义**：**没有规约**。`T.Parallel(block, dim_qk)` 的每个 `(i, d)` 独立计算，元素之间无累加顺序问题
  ——这是本算子与同批次另两个 launch 的本质区别（`_delta_kernel` 有 `D_v` 归约、`_bwd_kernel` 有 K 维累加与
  跨 CTA 累加）。因此迁移侧**不存在**「累加顺序是否可换分组」的自由度，也不受容差内重排影响。
- **dtype 语义**（本算子最重要的一条）：输入 `dq_acc` 已经是 **float32**（源码签名
  `T.Tensor((total, heads, dim_qk), "float32")` 即证据）；`scale` 是 python `float` 闭包常量；乘积在
  **float32 域内完成并只做一次舍入**；随后**一次性**按 round-to-nearest-even（RNE）窄化到
  `float16 | bfloat16`。即「fp32 乘 + 单次 RNE 窄化」，不是「fp16 乘 + fp16 舍入」（两次舍入）。
  §1.6.1 的 A1/C1 用机器验证量化了这一条在 fp16 下可达 3857 ulp。
- **溢出语义**：`scale = D_qk ** -0.5`（manifest `sm_scale` 默认值）远小于 1，但 `dq_acc` 是累加值，
  量级不受约束。IEEE fp16 的溢出阈值是 **65520**（65504 与不可表示的 65536 的中点）：`|y| < 65520` 的
  乘积仍 RNE 到 65504，`|y| ≥ 65520` 才变成 `±inf`。golden（torch 的 `.to(dtype)`）就是这个语义；
  §1.6.1 的 A3/C2 据此裁决设备侧**必须关饱和**（真机已确认 `sat=False` 即 IEEE，见 U5）。
- **非有限值语义**：无特殊处理——`NaN·s = NaN`、`±inf·s = ±inf`（`scale > 0`，符号不变）；源码没有任何
  `if_then_else` / clamp，自然传播。真机补充事实：`vcvt` 窄化**不保留 NaN payload**（任意 NaN → `0x7FFF`），
  故逐位比对必须「NaN 只比位置」且 `equal_nan=True`（dense Stage 3 实证）。
- **边界语义**：① `T_q` 是 `T.dynamic("total_q")`（源码第 90 行），运行期才知道；packed 下它等于
  `Σ_b q_len[b]`，**请求边界本身不进入本算子**（`cu_seqlens_*` 只是 `_bwd_kernel` 的入参）；
  ② 头数与 `D_qk` 固定（`_MLABwdKernelBase.refusal` 要求 `dim_qk % 16 == 0`，源码第 277 行），
  packed manifest 全部取 192；③ 行块与 `T_q` 的非整除是**正常路径**（`T.ceildiv(total, block)`）；
  ④ 越界（`row ≥ total`）的行**既不读也不写**——源码没有显式谓词，靠 TileLang 对动态 extent 的隐式越界守卫；
  这一点是本设计 §0.6 RD3 必须**显式化**的地方（NPU 侧用运行期 `min()` clamp）。
- **host 侧语义（packed）**：`mla_varlen_bwd.py` 第 205 行以 `(heads, dim_qk, scale, dtype)` 特化编译，
  `_run`（第 219–228 行）里 `dq_acc = torch.zeros(q.shape, dtype=torch.float32, device=q.device)`
  （`q` 已是 packed `(T_q, H, D_qk)`，因此 `dq_acc` 连续、行主序、行 stride `H·D_qk·4` B），
  然后 `dq = land(dq_acc)` 由 `out_idx=[1]` 分配输出并**直接返回**（不 view）。
  这是本设计的**调用契约**：工厂参数只有 `(heads, dim_qk, scale, dtype)`，被调函数**只有一个入参**
  `dq_acc`，因此 `T_q` 只能在 kernel 内以动态维出现（不能改成编译期常量）。

**语义保持基线**：§8.1 的 golden 只依据本节写成 `(dq_acc.float() * scale).to(dtype)`，与源 docstring 逐字
对应，**不参考 §0.6 的 NPU 算法**，也与 packed / dense 无关（逐元素式对 `t` 的语义在两种布局下相同）。

### 0.2 源算子输入输出

| 参数 | 方向 | Shape | dtype | 说明 |
|---|---|---|---|---|
| `dq_acc` | in | `(total, heads, dim_qk)` | `float32` | `total = T_q`（动态，packed 下为 `Σ q_len`）；行主序 `(t, h, d)`，`d` 连续 |
| `dq` | out | `(total, heads, dim_qk)` | `float16 \| bfloat16` | 与输入同 shape；dtype = 输入 dtype |

三处契约一致，互相印证（无臆测）：

1. 源码 `_dq_land_main` 的两个张量注解（第 94–95 行）；
2. packed op 契约 `examples/TileOPs/tileops/manifest/attention.yaml`：`dq: {dtype: T, shape: "[T_q, H, D_qk]"}`；
3. packed workload `examples/TileOPs/tileops/workloads/attention.py` 的 `total_q = sum(q_lens)` 与
   `(T_q, H, D_qk)` 生成形状。

与 dense 的差别仅在**调用方**：dense 的 op 契约是 `dq: [B, S_q, H, D_qk]`，由 `dq.view(q.shape)`（源码第 379 行）
把同一块 `(T_q, H, D_qk)` 缓冲重新解释成 BSHD——`view` 不改变内存，因此对本 kernel 无影响（§11 P3）。

### 0.3 实现算法解读（怎么算）

**源码结构化阅读**（`_dq_land_kernel` 全文 26 行 + host 侧 `_compile` / `_run`）：

- **入口签名**：`_dq_land_kernel(heads, dim_qk, scale, dtype)` → 内部 `_dq_land_func()` 无参（几何与 `scale`
  全部是闭包常量），`total = T.dynamic("total_q")` 是唯一的运行期维度。
- **Kernel 结构**：`T.Kernel(T.ceildiv(total, block), heads, threads=128)` → **二维网格**（行块 × 头），
  每块 128 线程（4 warp）；`block = 64` 行/tile。
- **计算原语**：`T.copy`（GM→fragment，跨步 region `dq_acc[bx*64:(bx+1)*64, h, :]`，行 stride `H·D_qk`）→
  `T.Parallel(64, dim_qk)` 逐元素乘 → `T.copy(acc, dq[...])` 写回（**在 copy 里完成 fp32→16-bit 的 cast**）。
- **内存分配**：`acc = T.alloc_fragment([64, dim_qk], "float32")`——**纯寄存器/fragment，无 SMEM、无 fp16
  暂存缓冲**；中间结果从不落 GM。
- **循环与调度**：无 K 维循环、无 `T.Pipelined`；一个 block 处理一个 tile（一次搬运 → 一次乘 → 一次写回）。
- **线程映射**：`T.Parallel(64, 192) = 12288` 个元素摊到 128 线程 → 每线程 96 个元素，编译器据此向量化
  （float4 粒度 24 次/线程）。
- **host 逻辑（packed）**：`out_idx=[1]` 由 JIT 分配输出（免 host 侧 `torch.empty` 与 memset）；
  `@target_cache(maxsize=32)` 按 `(heads, dim_qk, scale, dtype)` 特化编译；`_run` 里 `dq = land(dq_acc)`，
  `dq_acc` 由 `torch.zeros(q.shape, float32)` 现场分配（第 223 行）。**`cu_seqlens_*` 不在本 launch 的参数表里**。

**数据流图（源硬件视角）**

```
dq_acc (GM, fp32, 跨步行 768 B)──T.copy(MTE2/LDG)──▶ acc (寄存器 fragment 64×192 fp32, 48 KB)
                                                        │ T.Parallel 逐元素 × scale
                                                        ▼
                                                    acc (同寄存器)
                                                        │ T.copy(写回, 同时 cast)
                                                        ▼
                                              dq (GM, fp16/bf16, 跨步行 384 B)
```

**计算步骤分解**（源码每条计算语句都归入某步，无臆造步骤）

| # | 步骤 | 源码 | 数据 |
|---|---|---|---|
| S1 | 行块/头解码 | `bx`, `h`（网格坐标） | — |
| S2 | 跨步搬入 | `T.copy(dq_acc[bx*block:(bx+1)*block, h, :], acc)` | GM→fragment |
| S3 | 逐元素乘标量 | `acc[i,d] = acc[i,d] * scale`（`T.Parallel`） | fragment 原地 |
| S4 | 搬出并 cast | `T.copy(acc, dq[bx*block:(bx+1)*block, h, :])` | fragment→GM |

**请求边界不产生任何步骤**：源码第 79–106 行没有 `cu_seqlens`、没有 `q_len`、没有 per-request 循环。
packed 的「哪个 token 属于哪个请求」只影响 `_bwd_kernel` 的 mask 与累加范围，对本步骤表**零影响**
（§11 P4；这是本算子与 `_delta_kernel` packed 版的最大不同——后者因输出 layout 从 `(B,H,S)` 变成 `(H,T)`
而必须重新推导取址，见 `examples/mla_varlen_bwd/_delta_kernel/DESIGN.md` §0.1）。

### 0.4 优化手段解读（为什么快）

逐项识别源码里的优化手段（“识别不出来 ≠ 不存在”；未识别的优化会被静默丢弃）。本函数与 dense 版逐字相同，
故清单与 dense 一致，逐项复核后**无遗漏、无新增**：

| # | 优化手段 | 目的 | 机制 | 依赖的源硬件特性 |
|---|---|---|---|---|
| 1 | 二维网格 `(ceildiv(T_q,64), H)` | 把 `T_q·H` 个 tile 全部暴露为独立 block | CUDA 的 block 级并行 + 波次调度 | SM 数量多、block 调度器廉价 |
| 2 | `block = 64` 行 tile | 一次搬运复用 48 KB，摊薄搬运/启动开销 | fragment 驻留 | 寄存器文件容量 |
| 3 | **copy 内融合 cast**（fp32 fragment → fp16 GM 直写，无 fp16 中间缓冲） | 省一次 fp16 暂存往返与一次 pass | `T.copy` 隐式 dtype 转换 | CUDA 逐线程 load/store 可任意 cast |
| 4 | `T.Parallel(64, 192)` 摊平 | 让 12288 个元素均匀落到 128 线程并被向量化 | 每线程 96 元素、float4 向量 | warp 内连续 lane 合并访存 |
| 5 | `threads = 128` | 4 warp/CTA，兼顾占用率与 tile 尺寸 | CUDA 线程域 | warp 调度 |
| 6 | `TL_ENABLE_FAST_MATH` | 放开 fast-math 让编译器收缩运算 | 编译器 flag | 无（对单次乘无实际收益） |
| 7 | `compile_flags=["-O3", "-DENABLE_BF16"]` | 优化级别 + 打开 bf16 支持 | nvcc flag | CUDA 工具链 |
| 8 | `out_idx=[1]` JIT 分配输出 | 省 host 侧显式分配与 memset | 框架分配输出 | 无（框架能力） |
| 9 | 元素级**无分支**（无谓词、无边界判断） | 热路径无控制流，全向量化 | tile 消费整块 | 越界由编译器隐式守卫 |

**packed 复核**：这 9 项里没有一项依赖 `T_q` 的取值或请求边界；唯一与 `T_q` 相关的是第 1 项（网格规模）
与第 9 项（越界守卫），两者都只取决于「`T_q` 是运行期量」这一性质，而该性质在 dense 与 packed 下**相同**。

**迁移三问小结（阶段结论，供 §0.5 使用）**

| 问 | 结论 |
|---|---|
| **语义（What）** | 逐元素 `dq = cast_dtype(fp32(dq_acc)·float32(scale))`：**无规约、无跨元素依赖**，fp32 域乘 + **单次** RNE 窄化（非饱和），`dq`/`dq_acc` 同 shape `(T_q,H,D_qk)`；边界只有「尾块行数由 `T_q` 决定」与「非有限值自然传播」。packed 下 `T_q = Σ q_lens`，**请求边界不进入本算子**。（详见 §0.1/§0.2） |
| **算法（How）** | 4 步：行块/头解码 → 跨步 DMA 搬入 fp32 fragment(64×192) → 逐元素乘 `scale`（`T.Parallel`）→ 搬出时 copy 内 cast 落盘；无 SMEM 中间缓冲、无 K 维循环、无 `T.Pipelined`，一 tile 一 block（二维网格 `(ceildiv(T_q,64), H)` × 128 线程）。（详见 §0.3） |
| **优化手段（Why fast）** | 9 项：二维全暴露网格 / 64 行 tile 复用 / **copy 内融合 cast（无 fp16 中间缓冲）** / `T.Parallel` 摊平向量化 / 128 线程 / fast-math / `-O3`+bf16 flag / `out_idx` JIT 输出 / 热路径无分支。其中 3 项与硬件强相关（#3 重新设计、#5 舍弃、#4 等价替换），其余保留或补强。（详见 §0.4/§0.5） |

### 0.5 硬件耦合性分析与 NPU 适配决策

判定输入 = Phase R 调研结论（§1.6.0）：调研选定的算法族是**「单遍流式向量映射 + 32 位域单次舍入 +
核内流水分层」**，它与源算法在「逐元素、单遍、无中间缓冲」三点上一致，因此源算法的优化意图**大部分可承接**；
分歧集中在「cast 放在哪一层」与「并行如何映射到核」。该判定与 packed / dense **无关**（§0.0 推论）。

**按三层判定模型归类**（判定口径见 `.agents/skills/tilelang-op-design/references/migration-analysis.md` §5.1）：
- **语义层**（数学映射 / I/O 契约 / 边界 / dtype 语义）：**无条件保留**，对应 §0.1/§0.2 与 §8.1 的 golden；
  下表不含语义层条目（因为它不参与取舍）。
- **算法层**（步骤分解 S1–S4、数据流、分块策略）：**全部保留**——S1–S4 是纯数学顺序（解码 → 搬入 → 逐元素乘 →
  搬出），不依赖源硬件的布局或并行假设；`block = 64` 的行块复用属「弱硬件相关」，见下表 #2。
- **优化层**（并行组织、搬运调度、存储层级、cast 落点）：下表 #1/#3/#4/#5/#6/#7/#8/#9/#10 逐项判定。

| # | 条目 | 硬件强相关？ | 处置 | 依据 |
|---|---|---|---|---|
| 1 | 二维网格 `(行块, 头)` | 否（并行分解本身与硬件无关，只有映射方式相关） | **等价替换** | NPU 侧 `T.Kernel` **只接受一维网格且无 `threads`**（`.agents/platforms/toolchains/ascend/references/programming.md` 第 13 行）；一维网格 + `it → (h, t0)` 解码保持同一分解（§0.6 RD1） |
| 2 | 64 行 tile 复用 | 弱（容量参数化） | **保留** | UB 253952 B 足够承载 64×192 fp32 ×2 版本 + fp16 输出 ×2 版本（§4.5） |
| 3 | copy 内融合 cast | **是**（依赖逐线程 load/store 可 cast） | **重新设计** | 平台语义检查**显式拒绝** DMA cast：`T.copy() from dtype 'float32' to dtype 'float16' outside a VF block is not allowed. DMA copies cannot perform type casting — perform the cast inside a VF block (T.SimdVF() or T.SimtVF()) instead.`（dense 探针 m5/m6，`examples/mla_dense_bwd/_dq_land_kernel/history_version/design_probe_simd.py`）。改为 VF 内 `vcvt`（§0.6 RD2） |
| 4 | `T.Parallel` 摊平 | 是（线程/lane 语义） | **等价替换** | 换成 SIMD 寄存器级映射：一行 192 fp32 = **3 个 64-lane 向量寄存器**（2048 bit / 32 bit），每行 3 组 `vld→vmuls→vcvt→vsts`（§0.6 RD2） |
| 5 | `threads = 128` | **是**（CUDA warp 域） | **舍弃** | ascend 的 `T.Kernel` 没有线程域（`programming.md` 第 13 行）；核级并行由 56 个 AIV + 每核向量寄存器提供 |
| 6 | `TL_ENABLE_FAST_MATH` | 是（编译器 flag） | **舍弃** | 本算子只有一次 fp32 乘，没有可收缩的运算链；fast-math 反而可能改变次正规/舍入行为（§8.2 要求逐位 RNE 一致）。§7.3 记录为**故意不启用** |
| 7 | `-O3` / `-DENABLE_BF16` | 是（nvcc flag） | **舍弃** | CUDA 专有；ascend 的 bf16 支持由工具链提供（`vcvt` 的 `bfloat16` 目标类型） |
| 8 | `out_idx` 分配输出 | 否 | **保留（需补符号绑定）** | 平台支持 `tilelang.compile(..., out_idx=...)`（`.agents/platforms/toolchains/ascend/workflow.md` 第 31 行）；但 `T_q` 是动态维，必须把符号绑定到外层 python 变量，否则编译期报 `Cannot infer symbolic variable total_q ... bind it from an input tensor or scalar parameter`（dense 探针 e14 负向 / e15 正向） |
| 9 | 无谓词热路径 | 是（源侧靠隐式越界守卫） | **保留并补强** | 保持「热路径无分支」（行循环全静态、无逐元素谓词），但**尾 tile 的搬运 extent 显式运行期 clamp**（§0.6 RD3），不依赖隐式行为 |
| 10 | packed 调用路径（host 侧无 view） | **否**（纯调用方布局，kernel 不可见） | **保留** | `_run` 第 223/227 行；`dq_acc` 恒连续、`land` 只收一个张量（§11 P1–P4） |

**统计：保留 4 项（#2/#8/#9，其中 #9 需补强；#10 为 packed 调用路径）、等价替换 2 项（#1/#4）、
重新设计 1 项（#3）、舍弃 3 项（#5/#6/#7）。**

### 0.6 NPU 算法重设计

源算法的优化意图逐项承接（§0.4 → 本节），三项重设计各自给出「源方案 → NPU 新算法 → 语义保持论证」。
三项与 §0.5 处置的对应关系：RD2 ↔ #3（唯一的“重新设计”处置）；RD1 ↔ #1/#5（映射层，兼承 #2 的 tile 复用）；
RD3 ↔ #9（保留但补强），并补上源侧依赖隐式守卫的那部分边界处理。
**RD2 的落盘形态以真机实证为准（`PK_B32` + 32 位掩码），不是 dense 设计初版的 `NORM_B16`。**

#### RD1 并行映射：二维 CUDA 网格 → 一维核网格 + 持久化核内串行

- **源方案**：`T.Kernel(ceildiv(total,64), heads, threads=128)`，每个 block = 一个 (行块, 头)，由 SM 波次调度。
- **NPU 新算法**：`T.Kernel(G)`，`G = 56`（= 本型号 AIV 核数，实查见 §5.5）。逻辑 tile 定义为
  `it → (h = it % H, t0 = (it // H) · R)`，`ntiles = ceildiv(T_q, R) · H`；第 `bx` 个核用
  **运行期连续区间** `[bx·cpt, min((bx+1)·cpt, ntiles))` 串行走完自己的 tile（`cpt = ceildiv(ntiles, G)`），
  区间内用 `T.Pipelined(..., num_stages=2)` 做 load→compute→store 的两级流水。
- **语义保持论证**：映射只决定「哪个核在什么时候算哪个 `(t, h)` 行块」，元素级公式不变；每个 `(t,h,d)`
  仍被**恰好一个**核写到一次（区间 `[bx·cpt, …)` 对 `bx ∈ [0,G)` 互斥且并集覆盖 `[0, ntiles)`，`h`/`t0`
  解码是双射）——所以输出与源算法逐元素相同。线程域消失不改变数学。
- **承接的优化意图**：#1（并行分解不变）、#5（线程域 → 核+向量寄存器）、#2（tile 复用不变）。
- **packed 复核**：`ntiles` 的表达式在 packed 下只是 `T_q` 取另一个值；请求边界不参与 tile 解码。

#### RD2 计算层：copy-cast + 128 线程 fragment → VF 内向量段链

- **源方案**：fp32 fragment 在寄存器里乘 `scale`，**由写回 copy 完成 cast**（无 fp16 缓冲）。
- **NPU 新算法**：UB 上两块缓冲 `acc (R, DW) fp32` 与 `out (R, DW) dtype`；GM→UB 用一次跨步 `T.copy`；
  在 `T.SimdVF()` 内逐行、逐 64-lane 段执行
  `v = vld(acc[i, s*64]) → vmuls(v, scale) → vcvt(v, dtype, round="ROUND_R", sat=False, part=0)`
  → `vsts(out[i, s*64], v, m, dist="PK_B32", extent=64)`（`m = pset(32, "PAT_ALL")`）。即把「乘 + cast」
  两级都放进向量寄存器，`out` 只是 `vsts` 的落点，最后由一次 UB→GM `T.copy` 搬出。
- **语义保持论证**：`vmuls` 是 fp32 标量乘（与源码 `acc*scale` 同一条 IEEE 运算，且 `scale` 与源码一样是
  编译期 `float` 字面量，设备码里就是 `float(0x1.279a74590331cp-4f)`）；`vcvt(ROUND_R, sat=False)` 是
  「单次 RNE 窄化、不饱和」。两条真机实证把这条论证从编译级提升到设备级：
  ① `sat=False` 即 IEEE（`65519→65504`、`65520→±inf`、`1e5→±inf`，且 `sat=True` 与 `sat=False` 逐项相同，
  见 U5 的 repro）；② 窄化结果落在 16 位寄存器的**偶 lane**，稠密写回必须 `PK_B32` + **32 位 lane 掩码**
  （`NORM_B16` / `PK_B16` / 16 位掩码都只写一半，见 U4 的 repro）。`out` 缓冲列宽
  `DW = roundup(D_qk, 64) = 192`（packed manifest 几何无 padding）。**没有引入任何新的舍入或饱和**。
- **承接的优化意图**：#3（cast 与缩放在同一 pass 内完成、无 fp16 GM 中间缓冲）、#4（摊平改为 lane 映射）。

#### RD3 边界层：隐式越界守卫 → 显式运行期行数 clamp

- **源方案**：`T.copy` 的 region 用 `block` 全 extent，`row ≥ total` 的行依赖编译器对动态 extent 的隐式守卫。
- **NPU 新算法**：两次搬运的 region 行数都写成运行期表达式 `n = T.min(R, T_q − t0)`：
  `T.copy(dq_acc[t0:t0+n, h, :], acc[0:n, 0:DQ])` 与 `T.copy(out[0:n, 0:DQ], dq[t0:t0+n, h, :])`。
  行循环仍是**静态 R**，因此尾 tile 会多算 `R−n` 行（读未初始化 UB、写 `out` 的尾部列），这些行**从不被
  搬出**（store copy 带同一 clamp）——与同批次 `_bwd_kernel` 设计「尾块多算一个满 tile」的既定取舍一致。
- **语义保持论证**：有效行集合 `[0, n)` 与 `min(R, T_q − t0)` 恒等；越界行既不读 GM 也不写 GM
  （生成的设备码里两条 `asc_copy_*` 都带 `min(64, (total_q - ((it >> 3) * 64)))`，见本目录
  `history_version/probe_packed_gen_float16.txt` 第 25/33 行）。冗余计算只落在 UB 的无效行上，
  不改变任何输出元素。
- **承接的优化意图**：#9（保留热路径无分支，但不依赖未文档化的隐式守卫）。

#### 重设计后与源算法的结构差异（一句话）

源 = 「二维 CUDA 网格 + 128 线程 + copy 内 cast」；NPU = 「一维 56 核持久网格 + 两级流水 + VF 内
`vld/vmuls/vcvt/vsts(PK_B32)` 段链 + 运行期行 clamp」。**逐元素公式、fp32 域单次舍入、无中间缓冲三点完全保留。**

#### 重设计项的验证方式（`migration-analysis.md` §6.3 第 3 条，逐项）

| 重设计项 | 数学等价 | 边界语义核对 | 验证方式 |
|---|---|---|---|
| RD1 并行映射 | 元素级公式不变；`[bx·cpt, …)` 区间互斥且覆盖 ⇒ 每个元素被恰好一个核写一次 | 空区间（`ntiles < 56`）→ 零迭代，不写任何元素；`T_q = 0` 时 `ntiles = 0`，全部核空转、无输出（与源侧一致） | L0-4（退化）、P7（同一 `T_q` 不同切分逐位相同）、P5（连跑 20 次一致 + 单版本对照） |
| RD2 计算层（VF 段链） | `vmuls` = 源码同一条 fp32 乘；`vcvt(ROUND_R, sat=False)` = 单次 RNE 窄化 | `±inf`/`NaN` 自然传播（NaN payload 不保留 → 只比位置）；`±0` 符号一致；次正规按 IEEE；溢出阈值 65519/65520 | §1.6.1 机器验证（A1/A2/A3 全 `EQUIV_PASS`）+ L0-6a/6b（角点、`torch.equal`）+ P3/P4 |
| RD3 边界层（行 clamp） | 有效行集合 `[0, n)` 与 `min(R, T_q−t0)` 恒等；越界行不读不写 GM | `T_q < R`（L0-4）、`T_q % 64 ∉ {0}`（L0-3 的 308）、整除（L0-5）三种情形 | L0-3/L0-4/L0-5 + P1（canary 验无越界写）+ P2（尾块行数） |

**packed 侧重设计项：无。** 判据：RD1/RD2/RD3 的每条论证都只用「`T_q` 是运行期量」这一个性质，
而该性质在 packed 与 dense 下相同（§0.0 证据 4/5）。作为对照，同 op 的 `_delta_kernel` 因 packed 改变了
**输出 layout**（`(B,H,S)` → `(H,T)`）而必须重推取址（`examples/mla_varlen_bwd/_delta_kernel/DESIGN.md` §0.6）；
本 launch 既不读写 per-request 统计量，也不改 layout，故无对应重设计。

### 0.7 标杆实现

| 参考 | 相似度 | 借鉴点 |
|---|---|---|
| `examples/mla_dense_bwd/_dq_land_kernel/_dq_land_kernel.py` | **最高（同一函数、真机 31/31、`bit_diff=0`）** | **本设计的复用基准**：一维 56 核 + 两级流水 + `PK_B32` 段链 + 运行期 clamp + `_simd_body_gate` + golden/分层测试 |
| `examples/mla_dense_bwd/_dq_land_kernel/DESIGN.md` | 高（同函数的设计与论证） | 迁移三问、调研四问、A1–A5、分核三要素、L0 计划；**但 §3.2/§3.3 的 `NORM_B16` 形态已被真机证伪，不承接** |
| `/workspace/TileOPs/src/tileops/_kernels/attention/mla/bwd.py`（第 79–106 行） | 本体 | 语义、`block=64`、`scale` 闭包、`out_idx` |
| `examples/TileOPs/tileops/kernels/attention/mla_varlen_bwd/_mla_varlen_bwd_kernels.py`（第 429–455 行） | 本体（抽取副本） | 证明 packed scaffold 里的函数体与 dense 逐字相同 |
| `examples/mla_varlen_bwd/_delta_kernel/DESIGN.md` | 中（同 op 的 launch 1，packed） | packed 调用路径的描述方式、动态 `total` 的取证口径、分核三要素写法 |
| `examples/mla_dense_bwd/_delta_kernel/_delta_kernel.py` | 中（同 op 的 launch 1 实现） | 一维 `T.Kernel`、`T.annotate_buffer_versions`、跨步 region 的目的端切片、`vcvt` 两半语义 |
| `examples/paged_cache_write/_paged_cache_write_kernel/_paged_cache_write_kernel.py` | 中（同平台逐元素写回先例） | 目标 GM 的跨步写回与 dtype 落盘 |
| `.agents/platforms/toolchains/ascend/kernel_template.py` | 中 | 一维 `T.Kernel` 骨架、尾块分支形式（本设计用运行期 clamp 取代其静态 tail） |
| `.agents/platforms/toolchains/ascend/knowledge/PL-950-row-primitives.md` | 中 | 行原语的 region 限定与尾块正确性口径（SIMT 路径） |
| `.agents/platforms/toolchains/ascend/knowledge/numerical-boundaries.md` | 高（转换语义口径） | fp32→fp16/bf16 与 torch 逐位对照的样本集；明确「不把 SIMT 结论用于 `T.simd.vcvt` 的任意模式」——故本设计对 `vcvt` 另取真机证据（U4/U5） |
## 1. 概述

### 1.1 算子名称

`mla_varlen_bwd._dq_land_kernel`（`MLAVarlenBwdOp` 的第三个 launch，dQ 落盘；与
`mla_dense_bwd._dq_land_kernel` 为**同一源函数**）。

### 1.2 功能描述

把 `_bwd_kernel` 写好的 fp32 dQ 累加缓冲乘 `scale` 并按输入 dtype（fp16/bf16）落盘。纯带宽算子：
每元素读 4 B、写 2 B，无 FLOPs 级工作（每元素 1 次乘 + 1 次窄化）。packed 下作用在整个 packed 行轴
`T_q = Σ_b q_len[b]` 上，**不需要请求边界信息**。

### 1.3 数学公式

```
dq[t, h, d] = cast_dtype( fp32(dq_acc[t, h, d]) · sf32 )      sf32 = float32(scale)
```

- `cast_dtype`：round-to-nearest-even 窄化，溢出按 IEEE（fp16 阈值 65520）给 `±inf`，**不饱和**。
- 逐元素、无规约、无跨元素依赖；`dq` 与 `dq_acc` 同 shape `(T_q, H, D_qk)`。
- `T_q` 动态（packed 下 `= Σ_b q_len[b]`）；`H`、`D_qk = 192` 编译期常量；
  `scale = D_qk ** -0.5`（manifest 默认，工厂参数；packed 测试另有 `sm_scale = 0.05` 用例）。

### 1.4 算法描述

按 §1.6.0 的调研选定算法（**单遍流式向量映射**，经 §0.6 重设计落地；与 dense 实现同一算法）：

1. **tile 解码**：逻辑 tile `it = t_block · H + h`，`h = it % H`、`t0 = (it // H)·R`（`R = 64`）。
   一维网格 `G = 56` 个核，每核负责 `[bx·cpt, min((bx+1)·cpt, ntiles))` 连续区间。
2. **搬入**：`T.copy(dq_acc[t0:t0+n, h, :], acc[0:n, 0:DQ])`，`n = min(R, T_q − t0)`，跨步行 768 B。
3. **向量 epilogue**（`T.SimdVF`）：每行 3 段（192 = 3×64），每段
   `vld → vmuls(scale) → vcvt(fp32→dtype, ROUND_R, sat=False) → vsts(PK_B32, pset(32, PAT_ALL))`。
4. **搬出**：`T.copy(out[0:n, 0:DQ], dq[t0:t0+n, h, :])`，跨步行 384 B。
5. **流水**：`T.Pipelined(cpt 区间, num_stages=2)` + `acc`/`out` 各 2 个版本，`load(k+1)` 与
   `compute(k)/store(k)` 重叠。

**与源算法的差异**（§0.6）：网格与线程映射不同、cast 由 copy 层下移到 VF 层、尾块显式 clamp；
**数学语义与「无中间缓冲」结构相同**。**packed 不引入差异**（§0.6 末、§11）。

### 1.5 数据流图

```
dq_acc (GM, fp32, 行 768 B, 行 stride = H·192·4)
   │  MTE2: T.copy(region n 行)  ── 运行期 clamp n = min(64, T_q − t0)
   ▼
acc (UB, fp32, 64×192×4 B = 49152 B × 2 版本)
   │  VF (SIMD): 每行 3 段 vld → vmuls → vcvt → vsts(PK_B32)
   ▼
out (UB, dtype, 64×192×2 B = 24576 B × 2 版本)
   │  MTE3: T.copy(region n 行)
   ▼
dq (GM, fp16/bf16, 行 384 B, 行 stride = H·192·2)
```

### 1.6 算法调研与优化分析

#### 1.6.0 算法调研（Algorithm Research）

调研对象 = **同一数学语义的算法族**：所有能算出 `dq = cast(dq_acc·scale)` 的结构。输入公式 / 源算法
只是**基线候选**。调研深度：Phase 1 判定为「单步逐元素」类 → 轻量调研（四问各给出明确结论 + 依据），
但访存 / 复杂度仍按要求量化（带宽受限算子，流量项必算）。**packed 与 dense 的算法族完全相同**
（同一函数、同一语义、同一 I/O 连续性），故调研结论一致，本节按 packed workload 重算量化部分。

**R1 等价化简公式候选**（含基线；正式等价论证在 §1.6.1，此处只给初判与收益方向）

| 候选 | 公式结构 | 等价性初判 | 收益方向 |
|---|---|---|---|
| **B0 基线（源算法）** | `cast0(fp32(x)·sf32)`，单遍逐元素 | — | 参照点 |
| R1-a 单次舍入（32 位域乘 + 一次 RNE） | 同 B0（本设计即取此式） | 逐位恒等（§1.6.1 A1 机器验证 0 ulp） | 无（基线已是该形） |
| R1-b 16 位域乘（先 cast 再乘） | `cast0(cast0(x)·cast0(s))` | **不等价**：两次舍入，fp16 下最大 3857 ulp、违反率 1.2e-3（§1.6.1 C1） | 「省」一次 fp32 乘，但输入本来就是 fp32，无收益且失精度 → 弃 |
| R1-c 饱和窄化 | `cast_sat(fp32(x)·sf32)` | **不等价**：`|y| ≥ 65520` 时 65504 vs ±inf（§1.6.1 C2） | 无收益 → 弃 |
| R1-d 截断窄化 | `trunc(fp32(x)·sf32)` | **不等价**：系统性偏小 1 ulp（§1.6.1 C3） | 无收益 → 弃 |
| R1-e **融合进 `_bwd_kernel` epilogue**（取消本 launch） | 由 `_bwd_kernel` 直接写 `dq` | 数学上 `scale·Σp_i = Σ(scale·p_i)`（该式只省「乘」，不省搬运）；**但 `dq_acc` 是跨 CTA 累加（源码第 254 行 `T.atomic_add(dq_acc[q_start + m0 + i, h, d], dq_f[i, d])`），最后一趟 dQ 部分和到达前无法定稿** | 若可行可省 4 B/elem 的写 + 4 B/elem 的读（`ds-v3-8x4k` 的 `8E = 6.44 GB` ≈ 6.4 ms）；**不可行**：需要设备级 barrier，而平台明确「`ascend.batch_schedule` 的跨 block barrier 依赖尚未通过工具链认证」（`.agents/platforms/toolchains/ascend/workflow.md` 第 29 行；能力缺口条目 `CG-950-batch-barrier`，见 `.agents/platforms/toolchains/ascend/knowledge/capability-gaps.md`）；改为 fp16 累加又会破坏精度 → 弃 |
| R1-f 省掉 `dq_acc` 的 fp32 落地（bwd 内先 cast） | bwd 直接写 16-bit 部分和 | **不等价**：部分和的 16 位舍入误差在跨 CTA 累加中不可控 | 弃（且属上游 launch 的设计域） |

**R2 在线算法**

- **结论：本算法本身就是在线（单遍流式）形式，且没有可省的「多遍」**。结构依据：算子是 `T_q·H·D_qk`
  个元素的**纯映射**（无跨元素依赖、无归约、无状态），任何「分遍」方案都只会重复搬运同一批字节；
  基线 B0 已经是「读一遍 + 写一遍」。**不存在**需要物化（materialize）的中间矩阵，因此 online/offline 的
  区分对本算子退化——这一点与同 op 的 `_delta_kernel`（行归约，需一遍）、`_bwd_kernel`（K 维累加，
  受 UB 限制才必须分块）不同。
- **packed 复核**：请求边界是「分段」信息，若把它当作分遍依据只会把同一批字节切成更多次搬运，无收益；
  逐元素映射对任何分段都只需要 1 遍。故 packed 不改变 R2 结论。
- 唯一可在线化的东西是**流水**：把「读/算/写」在 tile 维度上重叠（§6.3），这已纳入算法（persistent +
  `num_stages=2`），不改变扫描遍数（仍 1 遍）。

**R3 算法复杂度对比**（四口径；口径说明：`E = T_q·H·D_qk` 个元素，`B_e` = 元素字节；流量 = 读 4 B + 写 2 B）

| 候选 | FLOPs | 访存 Bytes | 扫描遍数 | 中间缓冲峰值 | 可并行度 / 跨核归并 |
|---|---|---|---|---|---|
| **B0 基线（源）** | `E` 次乘 + `E` 次窄化 | `6E`（4 读 + 2 写） | 1 | 0 个 GM 中间缓冲（寄存器 fragment，48 KB/tile） | `ceil(T_q/64)·H` 个 block，**无归并** |
| 本设计（§0.6） | 同 B0 | `6E`（同上，仅 DMA 形态变化） | 1 | 147456 B UB/核（2 版本 ×(49 KB+24.5 KB)） | 56 核 × 核内串行 tile，**无归并** |
| R1-b/c/d | 同 B0 | `6E` | 1 | 同 B0 | 同 B0（但数值不等价） |
| R1-e（融合） | `E` 次乘 | 跨两 launch `10E` → 融合后 `2E`（省 `8E`） | 1 | 同 B0 | 需要设备级 barrier（`CG-950-batch-barrier`，不可用） |

- **具体数（packed manifest workload，`examples/TileOPs/tileops/manifest/attention.yaml`）**：

  | workload | `T_q = Σ q_lens` | `H` | `E` | `6E` | 逻辑 tile 数 |
  |---|---|---|---|---|---|
  | `mla-varlen-bwd-smoke` | 640 | 8 | 983,040 | 5.90 MB | 80 |
  | `ds-v3-8x4k` | 32768 | 128 | 805,306,368 | **4831.8 MB** | 65536 |
  | `ds-v3-mixed` | 32256 | 128 | 792,723,456 | 4756.3 MB | 64512 |
  | `kimi-k2-4x8k` | 32768 | 64 | 402,653,184 | 2415.9 MB | 32768 |

- 结论：**访存 / 遍数是主导项**（FLOPs 可忽略）；唯一能减半流量的候选 R1-e 被设备级 barrier 的可用性否决。
- packed 侧的量级变化：packed manifest 的 `T_q` 最大 32768（8 个 4096 长请求），对应 `6E = 4.83 GB`，
  是同批次 dense manifest 最大 workload（`ds-v3-b2-8k`，2.42 GB）的 2 倍——**这是本节唯一需要重算的量**。

**R4 硬件亲和性**

| 候选 | 计算单元匹配 | 片上容量 | 对齐/整除 | 静态边界 | 流水可融合 | 判定 |
|---|---|---|---|---|---|---|
| **本设计：SIMD 段链（主选）** | AIV 向量单元：`vld/vmuls/vcvt/vsts` 均在 `simd.py` 导出；一路 64 lane = 一整个 2048-bit 寄存器 | 144 KB UB（2 版本）< 248 KB ✓ | `192 = 3×64` 整段；行 768 B/384 B 均 32 B 对齐 ✓ | tile 内全静态（R 行、3 段），只有行数与搬运 extent 是运行期 | `T.Pipelined` + 2 版本，编译期已生成 ping-pong flag ✓ | **采纳**（本目录探针双 dtype COMPILE OK；dense 真机 31/31） |
| SIMT 投影（`T.SimtVF` + `T.Parallel` + `T.cast`） | AIV 线程域：128 线程 × `float2` 向量（dense 探针 n5 生成 `__float22half2_rn`） | 同 | 同 | 同 | 同 | 备选（§1.6.3 候选 A3；转换语义已被 `numerical-boundaries` 的 torch 逐位对照覆盖） |
| DMA 直接 cast（copy 层 cast） | **不支持**：平台语义检查拒绝 | — | — | — | — | 淘汰（报错原文见 §0.5 #3） |
| Cube / MixCV（`T.gemm` / `T.MixedKernel`） | 不适用：零矩阵乘，28 个 Cube 核无可计算内容 | — | — | — | — | 淘汰（把带宽活派给 Cube 只会浪费 28 个核） |
| host 侧 `torch` 完成（`dq_acc*scale` 后 `.to()`） | 不适用：契约规定本 launch 是 kernel（wrapper 只调 `land(dq_acc)`），且 host 方案多一次 GM 往返 | — | — | — | — | 淘汰（越出算子边界，且更慢） |

**设计期估算（D-2，本节末的下界行）**

- 常数来源分两级，均须标注可信度：
  ① `cost_measurements` **为空**；pattern-library 的带宽条目（`CONST-mte2-degradation` / `CONST-fabric-bw`，
  见 `.agents/skills/tilelang-op-optimize/references/pattern-library/constants.md`）是
  **Ascend910B2C + tilelang 0.1.2 + CANN 8.5.0** 的实测值（fabric 1.73 TB/s；256M 混合读写地板 1.26 TB/s，
  48 vector core；每核 18.4 GB/s），**与本任务的 950PR / 0.1.15 / CANN 9.2.1 不是同一组合**，只能作量级参照
  （56 × 18.4 GB/s = 1.03 TB/s 的每核地板外推）。
  ② **本型号的实测锚点**（同型号、同工具链、兄弟算子）：dense 的 `_dq_land_kernel` Stage 3 L2 实测
  `ds-v3-4k` 604 MB → 0.580 ms（≈1.04 TB/s）、`kimi-k2-8k` 604 MB → 0.576 ms（≈1.05 TB/s）、
  `ds-v2-lite-3k` 226.5 MB → 0.195–0.255 ms（0.89–1.16 TB/s）、`ds-v3-b2-8k` 2415.9 MB → 3.231 ms
  （≈0.75 TB/s）。**这是本设计可用的最强常数**（同为 950PR / 0.1.15 / CANN 9.2.1、同一逐元素写回路径），
  但它是**兄弟算子的实测、尚未进知识库**，按「未实测假设 + 估算依据」标注。取有效带宽带 **0.75–1.05 TB/s**。
- 流量项 = `6E ÷ 带宽`：smoke 5.90 MB → **5.6–7.9 µs**；`ds-v3-8x4k` 4831.8 MB → **4.60–6.44 ms**；
  `ds-v3-mixed` 4756.3 MB → 4.53–6.34 ms；`kimi-k2-4x8k` 2415.9 MB → **2.30–3.22 ms**
  （该几何与 dense `ds-v3-b2-8k` 的字节数相同，后者的实测 3.231 ms 正落在本带的上端）。
- 发射项：每 64 元素段 4 条向量指令 → `E/64 × 4` 条；`ds-v3-8x4k` 50.3 M 条 ÷ 56 核 = 899 K 条/核，
  按 1 条/cycle @1.4 GHz 估 ≈ **642 µs**（被流量项支配）；smoke 仅 ≈ 0.8 µs。
- 容量项：147456 B < 253952 B（UB），✓ 不构成约束。

> **设计期估算：4.60–6.44 ms**（`ds-v3-8x4k`；流量项 4831.8 MB ÷ 0.75–1.05 TB/s，发射项 ≈ 642 µs 被流量项
> 支配，容量项 ✓；常数取自 dense Stage 3 在同型号的 L2 实测带宽带，**该常数尚未进知识库、本算子未实测**，
> Stage 4 须回填实测值）。

**调研结论**：选定**单遍流式向量映射算法族**（B0 的语义 + §0.6 的核/向量映射），关键依据：
① 纯映射算子的流量下界 `6E` 已由基线达到，R1 的四个数值变体全部不等价、R1-e 的唯一减半路径被设备级
barrier 的可用性否决（`CG-950-batch-barrier`）；② 硬件亲和上只有 SIMD 段链与 SIMT 投影两条路，前者
容量/对齐/流水全部成立且已在本目录双 dtype 编译验证、在 dense 真机逐位通过，后者作为备选（§1.6.3）；
③ 由此 §1.4、§1.6.1–§1.6.3、§6 的分析对象都是该算法，不回退到源的「copy 内 cast + 线程域」直译。
**与基线的结构差异（一句话）**：并行分解从「二维 CUDA 网格 + 128 线程」改为「一维 56 核持久网格 + 64-lane
向量段链」，cast 从写回 copy 层移入 VF（`vcvt`）；**逐元素公式、流量下界 `6E`、无中间缓冲三点不变**。
**「无更优替代」的调研范围**：上述 R1 六候选、R2 结构判据（含 packed 分段判据）、R3 四口径与 4 个 packed
workload 的量化、R4 五候选，加上平台知识条目 `.agents/platforms/toolchains/ascend/knowledge/algorithm-candidates.md`
（纯 elementwise 无专门条目）与本目录 §0.7 的十项同类实现。**packed 与 dense 的调研结论相同**，
差别只在 R3 的 workload 数值（本节已重算）。

#### 1.6.1 数学等价优化（公式级）

分析对象 = §1.6.0 选定的算法（即 §0.6 RD2 的段链）。五项全部承接 dense（同一函数、同一实现），
逐项四要素如下；**其中 A4/A5 的收益与 packed workload 的 tile 数相关，已按 §1.6.0 的表重算**。

**A1 单次舍入（乘与窄化合并为一次舍入）**

- 原式（若按 16 位域算）：`dq = cast_dtype( cast_dtype(x) · cast_dtype(s) )` —— 两次舍入。
- 优化后：`dq = cast_dtype( fp32(x) · sf32 )`，乘积留在 fp32、只舍入一次。
- 等价性论证：golden 就是后者（源 `T.copy` 也是 fp32 fragment → 一次 cast）。数学上两次舍入不等于一次
  舍入；机器验证（下表）在 fp16 上量到 **3857 ulp / 违反率 1.2e-3**，故 16 位域形态被否决。
- 收益量化：0 次额外运算（输入本来就是 fp32），收益是**精度保真**：消除一个最多 3857 ulp 的偏差源。

**A2 `scale` 以 fp32 字面量内联（`vmuls` 的标量操作数）**

- 原式：先构造全 lane 广播向量 `vdup(sf32)`，再 `vmul(v, vdup)`。
- 优化后：`vmuls(v, sf32)`（向量×标量指令，标量直接来自指令立即数/标量寄存器）。
- 等价性论证：两种写法乘的都是同一个 fp32 常量 `float32(0.07216878…)`，IEEE 乘法逐位相同
  （机器验证 A2：0 ulp、违反率 0）。本目录设备码里该常量就是 `float(0x1.279a74590331cp-4f)`。
- 收益量化：**每段少 1 条向量指令**（`E/64` 条 → `ds-v3-8x4k` 约 12.6 M 条）与 1 个向量寄存器。

**A3 关闭窄化饱和（`sat=False`，即设备码 `RS_DISABLE`）**

- 原式（默认 `sat=True`）：`|y| > 65504` 一律钉到 `65504`。
- 优化后：IEEE 语义——`|y| < 65520` RNE 到 65504，`|y| ≥ 65520` → `±inf`。
- 等价性论证：golden 的 `.to(dtype)` 是 IEEE；机器验证里饱和形态（C2）在 `|y| ≥ 65520` 的角点上与
  golden 分歧（违反率 9.7e-4 fp16 / 7.3e-4 bf16），非饱和形态（A3）0 违反、0 ulp、NaN 位置一致。
  **真机已进一步确认**：`sat=False` 与 `sat=True` 在 f32→f16 上逐项相同，两者都是 IEEE
  （`examples/mla_dense_bwd/_dq_land_kernel/repro/CONST-950-vcvt-rs-disable-ieee.py`），因此该 flag
  本身对本转换对惰性，无额外数值风险。
- 收益量化：非性能项，而是**语义保真**；避免 `dq_acc` 溢出时把 inf 静默改成有限最大值（会污染下游）。

**A4 段切分取 `DW = roundup(D_qk, 64)`，本 op 几何下 `192 = 3×64` 无 padding**

- 原式（若按 128-lane 对齐 pad）：行宽 pad 到 256 元素 → 每行 4 段，1 段全是无效 lane。
- 优化后：`DW = 192`，每行恰好 3 段，`vsts` 的 **32 位 lane 掩码**（`pset(32, "PAT_ALL")`，与 `PK_B32` 对齐）全程满 lane——64 个 32 位字恰好装下 64 个有效结果，无空槽。
- 等价性论证：pad 形态的无效 lane 不参与输出（mask 排除），两式输出逐位相同；区别只在指令数。
- 收益量化：向量指令 `4R → 3R` 条/行（**−25%**），UB 的 `out` 缓冲 `256/192 → 1.0`（省 25%）。
  packed manifest 全部 `D_qk = 192`，因此每个 workload 都吃到该收益。

**A5 尾块行裁剪（搬运 extent = `min(R, T_q − t0)`）**

- 原式：按 `R` 行整块搬运（越界行依赖隐式守卫）。
- 优化后：只搬 `n` 行；`out` 也只搬 `n` 行。
- 等价性论证：有效行集合恒等（§0.6 RD3）；越界行在 golden 里根本不存在。无浮点差异。
- 收益量化：尾 tile 的 DMA 字节数 `R → n`；对 `T_q % 64 ≠ 0` 的几何省 `(64−n)/64` 的搬运量，
  且**保证不越界写**（正确性收益 > 性能收益）。**packed 下的实际价值**：4 个 packed workload 中
  `ds-v3-mixed`（`T_q = 32256 = 504×64`，整除）、`ds-v3-8x4k`（32768 = 512×64）、`kimi-k2-4x8k`
  （32768 = 512×64）都整除，只有 `smoke`（640 = 10×64）也整除；**但请求级长度是 ragged 的
  （512/1024/…/16384），且测试用例含 `[200,17,91]`（`T_q = 308 = 4×64+52`）与 `[64]`**，
  因此 A5 仍是必需的正确性项，不能因 manifest 恰好整除而省略。

**优化结论**：采纳 A1/A2/A3/A4/A5（全部为「保语义、减指令或不减」的项，无数值风险）；
**否决** 16 位域乘（C1）、饱和窄化（C2）、截断窄化（C3）、融合进上游 launch（R1-e，设备级 barrier 不可用）。
本算子没有「除法转乘倒数 / rsqrt / 减 max 稳定化 / 公共子表达式消除 / 归约合并」类空间——它只有一个乘，
**无更多可省的计算**（这就是「无更优替代」的正面陈述）。**packed 复核**：五项都与 `T_q` 的取值无关，
packed 不新增可优化项（也没有因分段而出现的公共子表达式）。

**等价性机器验证（D-1：`verify_equiv.py`，torch CPU + fp64 参照，本任务目录内独立重跑）**

脚本：`examples/mla_varlen_bwd/_dq_land_kernel/verify_equiv.py`（本机已执行，退出码 0）。输入 = 8192 个随机
fp32 dq 值 + 31 个角点：±0、±0.5、±1、±inf、NaN、±1e-8/±1e-38/±6e-8（次正规与 flush 区）、
fp16 上界 65504、溢出阈值 65519/65520、7e4、3e38、1e5 及其负值（每个 dtype 8223 个样本）。
「max ulp」按符号-幅值位型的有序整数距离计算；「违反率」按本 op 的验收容差
fp16 (atol 5e-3, rtol 1e-3) / bf16 (atol 1.6e-2, rtol 1.6e-2) 计算；scale 取 `192 ** -0.5`。

| dtype | 项 | 最大 ulp 差 | 违反率 | NaN 位置差 | 结论 | 说明 |
|---|---|---|---|---|---|---|
| float16 | A1 | 0 | 0.0 | 0 | **EQUIV_PASS** | 单次舍入 = golden |
| float16 | A2 | 0 | 0.0 | 0 | **EQUIV_PASS** | scale 预舍入到 fp32 无差异 |
| float16 | A3 | 0 | 0.0 | 0 | **EQUIV_PASS** | 非饱和 = IEEE = golden（含 65519/65520/1e5 角点） |
| float16 | C1（否决项） | 3857 | 1.216e-03 | 0 | NOT_EQUIV | 16 位域两次舍入 |
| float16 | C2（否决项） | 1 | 9.729e-04 | 0 | NOT_EQUIV | 饱和窄化（inf → 65504） |
| float16 | C3（否决项） | 1 | 9.729e-04 | 0 | NOT_EQUIV | 截断窄化 |
| bfloat16 | A1 | 0 | 0.0 | 0 | **EQUIV_PASS** | 同上 |
| bfloat16 | A2 | 0 | 0.0 | 0 | **EQUIV_PASS** | 同上 |
| bfloat16 | A3 | 0 | 0.0 | 0 | **EQUIV_PASS** | 同上 |
| bfloat16 | C1（否决项） | 1 | 0.0 | 0 | WITHIN_TOL | bf16 只有 8 位尾数，两次舍入落在 bf16 容差内但仍非逐位 |
| bfloat16 | C2（否决项） | 1 | 7.297e-04 | 0 | NOT_EQUIV | 同上 |
| bfloat16 | C3（否决项） | 1 | 7.297e-04 | 0 | NOT_EQUIV | 同上 |

脚本另有 `[baseline]` 自检行：golden（fp32 乘后 cast）与其 fp64 前置像在两种 dtype 上均 `ulp=0 / 违反率 0`
（本次输出 `nan_mis=0`），说明「fp32 乘」与「fp64 乘后一次舍入」在这批样本上无双重舍入差异，基线自洽。
脚本对采纳项与对照项分别断言（采纳项必须 `EQUIV_PASS`、对照项必须被证伪），全部满足时打印
`[EQUIV_PASS]` 并返回 0——本目录本次运行返回 0。

#### 1.6.2 向量化替代分析（循环 / 标量消除）

盘点实现方案（§0.6 RD2 + §3.3）里**全部**循环与标量计算点。**packed 复核的关键结论**：本 kernel 的循环与
标量点集合与 dense **完全相同**——packed 不引入 per-request 循环、不引入 `cu_seqlens` 取址、不引入分段
边界判断（§0.3 末），因此下表既是 dense 的盘点结果，也是 packed 的盘点结果：

| # | 计算点 | 形态 | 向量替代 | 结论 | 依据 |
|---|---|---|---|---|---|
| V1 | 逐元素乘 `x·scale`（源 `T.Parallel(64,192)`，每线程 96 元素） | 元素级标量循环 | **`vmuls`（64 lane/指令）** | ✅ 已替代 | `simd.py:vmuls`；本目录设备码 `simd_inst::vmuls(...)` |
| V2 | 逐元素 cast（源由 copy 完成） | 元素级标量转换 | **`vcvt`（64→64 lane，ROUND_R/RS_DISABLE）** | ✅ 已替代 | `simd.py:vcvt` 的 f32→f16/bf16 支持；设备码 `vcvt<half>(..., ROUND_R, RS_DISABLE, PART_EVEN, MODE_ZEROING)` |
| V3 | 元素的搬入/搬出（源逐线程 load/store） | 元素级访存 | **`vld` / `vsts(PK_B32)`（256 B / 128 B 一次）+ 两条 DMA `T.copy`** | ✅ 已替代 | `simd.py:vld/vsts`；设备码 `vlds_norm` / `vsts_pack_b32` / `asc_copy_gm2ub_align` / `asc_copy_ub2gm_align` |
| V4 | 行循环 `for i in T.serial(R)` | 行级 IR 循环（R=64） | **不可替代**（每轮是 3 段向量指令，本身已是向量操作；把行摊平成 `R·3` 段并不减少指令数，只把 `i·192 + s·64` 换成一条等价索引） | ❌ 保留（合理） | 类别：tile 级结构索引；每轮 3 段的偏移是 `i·192 + s·64`（`s` 段循环编译期可展开），标量开销相对 12 条向量指令可忽略；设备码即为此形（`for (i=0;i<64;++i) for (s=0;s<3;++s)`） |
| V5 | 段循环 `for s in T.serial(NSEG)` | 段级 IR 循环 | **不可替代**（同理；且展开成 3 条语句反而会触发工具链缺陷，见 §9.2 E1） | ❌ 保留（合理，且有反例证据） | dense 探针 e11：把段循环在 python 层展开 → 整个 SIMD 体被静默丢弃（`history_version/probe_e11_gen.txt` 无 `vlds_norm`）；本目录探针（IR 段循环）→ 正常生成 |
| V6 | tile 解码 `h = it % H`、`t0 = (it // H)·R` | block 级标量 | **不可替代**（每 tile 一次，不是逐元素热点） | ❌ 保留（合理） | 类别：block 级索引计算；`H` 是 2 的幂，设备码退化为 `it >> 3` / `it & 7`（本目录 fp16 设备码） |
| V7 | 区间边界 `cpt = ceildiv(ntiles, G)`、`it1 = min(it0+cpt, ntiles)` | block 级标量 | **不可替代**（每核一次） | ❌ 保留（合理） | 类别：任务映射计算，`T_q` 动态使 `ntiles` 只能运行期算（静态化不可能，见 §0.1 契约）；packed 下 `ntiles` 由 `Σ q_len` 决定，同样只能运行期求 |
| V8 | 尾块行数 `n = min(R, T_q − t0)` | 运行期标量（用于两条 DMA extent） | **不可替代**（动态 shape 的边界处理；已论证不能静态化——工厂拿不到 `T_q`） | ❌ 保留（合理） | 类别：动态边界；`T.min` 进入 DMA 的 extent 参数，设备码为 `min(64, (total_q - ((it >> 3) * 64)))` |
| V9 | 逐元素边界谓词 | 无（热路径无分支） | 不需要 | ✅ 已消除 | 设备码的 VF 体内只有 `for (i) for (s)` 两层循环，无 `if` |
| V10 | head 维广播/复制 | 无（每个 `h` 独立 tile） | 不需要 | ✅ 无需处理 | 设计上按 `h` 分 tile，不存在广播 |
| V11 | packed 请求分段处理（逐请求循环 / `cu_seqlens` 边界判断） | **不存在**（本 kernel 无此计算点） | 不需要 | ✅ 无需处理 | 源码第 79–106 行 `land` 唯一入参是 `dq_acc`；边界只存在于 `_bwd_kernel`（§11 P4） |

**向量化结论**：所有**逐元素**计算点（V1/V2/V3）与边界谓词（V9）都已用向量操作或 DMA 替代，热路径里
**没有逐元素标量循环**；保留的 4 个点全部属于「block/tile 级索引与任务映射」（V4/V5/V6/V7）或
「动态边界」（V8），其中 V5 的保留还有一条**负向证据**（python 层展开会丢代码，dense 探针 e11）。
替代所用 API（`vld/vmuls/vcvt/vsts/pset`）全部取自 `tilelang.ascend.language.simd` 的**当前安装包导出**
并在本目录探针双 dtype 编译通过，不依赖记忆或其他工具链。**packed 不新增任何循环或标量点（V11）**——
这是「几乎全量复用」在向量化维度上的直接体现。

#### 1.6.3 向量化轴与数据布局决策

I/O layout 是契约（`dq_acc` / `dq` 都是 `(T_q, H, 192)` 行主序连续，行内 `d` 连续；packed 与 dense 同一契约，
只是 `T_q` 的取法不同）；**核内布局与 lane 映射是设计变量**。逐候选枚举（含逐元素类的「连续轴选择 ×
广播轴对齐」，本算子无广播轴，退化为「连续轴选择 × tile 覆盖形状」）：

| 候选 | 结构 | 段宽 | 向量指令/行 | UB 占用 | 结论 |
|---|---|---|---|---|---|
| **A1（主选）单 head × 64 行 tile，行内 3×64 lane** | `acc (64,192) fp32` / `out (64,192) dtype`，每行 3 段满 lane | GM 段 768 B / 384 B | 3 vld + 3 vmuls + 3 vcvt + 3 vsts = 12 | 147456 B（含 2 版本） | **采纳**（本目录双 dtype COMPILE OK；dense 真机 31/31） |
| A2 多 head tile（`R × HH head × 192` 连续） | 段宽 `HH·768 B`（更大连续块） | 1.5–6 KB | 同 A1（按行不变） | `R·HH·192·4·2` 随 HH 线性涨，`HH=2,R=64` 就 294912 B 超限 | 弃：容量否决（§4.5 预算）——收益只是段宽变大，而 768 B 已远超「数百字节」的判据 |
| A3 SIMT 投影（每线程 `float2`，128 线程） | `T.SimtVF(threads=128)` + `T.Parallel(64,192)` | 同 A1 | 每线程 48 次 `float2` + `__float22half2_rn`（dense 探针 n5） | 同 A1 | 备选（见下方实验裁决）——转换语义已被 `numerical-boundaries` 的 torch 逐位对照覆盖 |
| A4 行 = 128 lane fp32（跨 2 个向量寄存器） | 一条逻辑指令跨 2 寄存器 | — | — | — | 弃：硬件向量寄存器就是 64 lane fp32（2048 bit/32 bit），没有 128 lane fp32 形态 |
| A5 转置/重排布局（先转成 `(H, T_q, D)` 再算） | 核内 repack | — | — | 额外一份缓冲 | 弃：I/O 契约是 `(T,H,D)`，重排要多一次 GM 往返（`+6E` 流量），与带宽目标相反。**packed 下更不可取**：packed 行轴本身是多请求拼接，重排既不改语义又增加一次全量搬运 |

**评分要点**（轴长对向量宽度整除性 / 尾 lane 浪费 / 累加链 / repack 代价 / UB 容量）：
`192 = 3×64` 对 64 lane **零浪费**；无累加链（纯映射）；无 repack；`147456 B / 253952 B = 58%`。
A2 的段宽收益（768 B → 1.5 KB）落在「段宽 ≥ 数百字节即正常」的区间内
（`CONST-gm-to-ub-bw-dilution`：「段宽 ≥ 数百字节时性能正常」，**910B2C 口径、跨型号假设**），
而容量代价是硬否决。**packed 复核**：packed 的 tile 覆盖形状与 dense 相同（单 head × 64 行），
因为 tile 解码只依赖 `it / H`，与请求分段无关；因此 A1–A5 的五条评分依据在 packed 下逐条不变。

**布局决策结论**：主选 **A1**（单 head × 64 行 tile，行内 3 个 64-lane 满段，`T.Pipelined` 2 版本，
`vsts` 用 `PK_B32` + 32 位掩码），因为它对 64 lane 零浪费、UB 58%、段宽 768 B 已过「数百字节」判据、
且已在本目录双 dtype 编译验证并被 dense 真机逐位验证；**备选 A3（SIMT 投影）**
按实验裁决模式保留（结构完整：同样的 tile/UB/分核参数，只把 VF 体换成
`T.SimtVF(threads=128)` + `T.Parallel(R, DQ)` + `T.cast`，dense 探针 n5 已编译并生成 `float2` 向量代码）。

**实验裁决计划（Stage 4 执行，判定裕度落在未实证常数上）**

- 未知常数：① 950PR 的 `vld/vmuls/vcvt/vsts` 指令吞吐（910B2C 的常数不能直接套用）；② `vcvt` 窄化的
  lane 利用率（`PART_EVEN` 只写低 64 lane，是否存在半寄存器浪费）；③ `PK_B32` 存储相对 `NORM_B16`
  半写形态之外的额外搬运代价（后者已被真机否决，仅作参考点）；④ SIMT 投影的 `float2` 路径相对
  SIMD 段链的有效带宽差异。四者都**没有**本型号的公开实测常数（`cost_measurements` 为空、
  `.agents/platforms/toolchains/ascend/knowledge/bottlenecks.md` 明确「当前没有 950 真机验证的瓶颈模式或
  收益阈值」）；本型号唯一可用锚点是 dense Stage 3 的 L2 时延（0.75–1.05 TB/s 有效带宽）。
- 代表 shape：`ds-v3-8x4k`（4831.8 MB）、`ds-v3-mixed`（4756.3 MB）、`kimi-k2-4x8k`（2415.9 MB）、
  `smoke`（5.90 MB）。
- 判据：`latency(备选 A3) < latency(主选 A1) × (1 − 0.05)` 才翻转；同时记录 `mte2_ratio` 与
  `*_active_bw`（按 `CONST-gm-to-ub-bw-dilution` 的方法学：判搬运瓶颈用活跃带宽，不用总量/总时长）。
- 回写路径：实测数据经附录补记固化本节判定；若 A3 全局或按 shape 分片胜出，则在 `perf_opt/` 产出 A3
  变体并按平台 `perf-feedback.md` 口径产出 `perf_feedback.md` 交 conductor 受控路由（本算子的工厂按
  `(heads, dim_qk, scale, dtype)` 特化，**按 shape 分派主选/备选可行**：`heads` 与 `dtype` 已在特化键里；
  packed 的 `T_q` 不参与特化，故「按 packed 长度分派」不可行，只能按 `heads`/`dtype` 分派）。

## 2. 编程模式选型

### 2.1 模式结论

**编程模式：SIMD**（平台 `programming_modes.choices` 的 `simd` 键；与 `migration_default = simd` 一致）。
落到 API 上 = 一维 `T.Kernel` 核网格 + `T.SimdVF()` 执行区域 + `T.simd.*` 寄存器级指令
（`vld/vmuls/vcvt/vsts/pset`），GM↔UB 由 `T.copy`（MTE2/MTE3）承担。

### 2.2 选型理由

1. **算子形态匹配**：纯逐元素映射，没有矩阵乘 → 不需要 Cube；没有线程私有状态 → 不需要 SIMT 线程域。
   AIV 的向量寄存器一次处理 64 个 fp32（2048 bit），而本算子的自然工作单位恰好是「一行 192 fp32 = 3 个
   满寄存器」，SIMD 是唯一零浪费的映射。
2. **单次舍入可控**：`vcvt` 的 `round="ROUND_R"` 与 `sat=False` 是**显式参数**，直接对应 golden 的
   「RNE、不饱和」语义（§1.6.1 A1/A3），且已在真机确认即为 IEEE（U5）；SIMT 的 `T.cast` 虽然也有 torch
   逐位对照证据（`numerical-boundaries`），但舍入模式不可在调用点显式声明。
3. **落盘形态已在真机定案**：`PK_B32` + `pset(32, "PAT_ALL")` 是唯一能稠密写回 `vcvt` 窄化结果的形态
   （U4 的 repro 实测 256/256），本目录探针在同一形态下双 dtype 编译通过。
4. **同 op 先例**：同批次 `_delta_kernel` 已用 `T.SimdVF` + `T.simd.*` 实现（dense 真机通过、packed 设计
   已冻结），其 `vcvt` 拓宽用法与本次窄化用法互为镜像。

### 2.3 模式影响

- **并行度来源**：56 个 AIV 核 × 每核 64-lane 向量单元；核内靠 `T.Pipelined` 让 load/compute/store 重叠。
- **同步**：全部由 AutoSchedule 自动插入（`asc_sync_notify/wait` 的 ping-pong flag），不写手工同步（§7）。
- **禁用项**：`T.SimtVF` 只作为 §1.6.3 的备选 A3；`threads=` 参数在本平台不存在（`T.Kernel(N)` 只有
  网格 extent）。
- **packed 影响**：无。SIMD 模式的选择依据是「逐元素 + 一行 3 个满向量寄存器」，与该行属于哪个请求无关。

## 3. API 映射设计

### 3.1 公式拆解

输入 = §1.6.1 的优化后公式（A1+A2+A3+A4+A5 采纳后的形态）：

```
# 每个核 bx ∈ [0, G)，G = 56
cpt  = ceildiv(ntiles, G)                     # ntiles = ceildiv(T_q, R) · H
for it in [bx·cpt, min((bx+1)·cpt, ntiles)):  # 核内串行 + 两级流水
    h  = it % H                               # 头（2 的幂 → 移位/掩码）
    t0 = (it // H) · R                        # 行块起点
    n  = min(R, T_q − t0)                     # 有效行数（运行期 clamp）
    # ① 搬入
    acc[0:n, 0:DQ] ← dq_acc[t0 : t0+n, h, :]  # fp32, 行 768 B
    # ② 向量 epilogue（VF）
    for i in 0..R-1:                          # 静态行循环
      for s in 0..2:                          # 静态段循环（192 = 3·64）
        v   = vld  acc[i, s·64]               # 64 个 fp32
        v   = vmuls(v, sf32)                  # A2：标量乘
        c   = vcvt(v, dtype, ROUND_R, sat=False)   # A1/A3：单次 RNE、不饱和
        vsts out[i, s·64] ← c, mask=pset(32, PAT_ALL), dist=PK_B32  # A4 + 真机落盘形态
    # ③ 搬出
    dq[t0 : t0+n, h, :] ← out[0:n, 0:DQ]      # dtype, 行 384 B
```

### 3.2 TileLang API 映射

| 数学/结构步骤 | TileLang API（当前安装包 `tilelang.ascend.language`） | 参数要点 |
|---|---|---|
| 核网格（56 个 AIV） | `T.Kernel(G)` | **一维**，无 `threads`（`kernel.py:Kernel`，多维被拒） |
| 动态维声明 | `total = T.dynamic("total_q")`，在两个 `T.Tensor` 注解里**复用同一对象** | 否则 `out_idx` 报「Cannot infer symbolic variable」（§3.5.4 E2） |
| UB 缓冲 | `T.alloc_shared((R, DQ), "float32")` / `(R, DQ), dtype` | 默认 scope `shared.dyn` = UB（`allocate.py`） |
| 双缓冲声明 | `T.annotate_buffer_versions({acc: 2, out: 2})` | 正整数版本数（`annotations.py`） |
| 流水循环 | `T.Pipelined(it0, it1, num_stages=2)` | 动态 `it0/it1` 已编译验证（本目录探针设备码 `for (it = block_idx*cpt; it < min(...); ++it)`） |
| 行/段循环 | `T.serial(R)` / `T.serial(3)` | **必须写在 prim_func 体内**（§9.2 E1） |
| 搬入 | `T.copy(dq_acc[t0:t0+n, h, :], acc[0:n, 0:DQ])` | 目的端显式切片 |
| 搬出 | `T.copy(out[0:n, 0:DQ], dq[t0:t0+n, h, :])` | 源端（UB）显式切片 |
| VF 区域 | `with T.SimdVF():` | `frame.py:SimdVF` |
| lane 掩码 | `T.simd.pset(32, "PAT_ALL")` | **32 位元素宽**，与 `PK_B32` 的 lane 宽度对齐（U4） |
| 向量载入 | `T.simd.vld(acc[i, s·64])` | 返回 `float32x64` |
| 标量乘 | `T.simd.vmuls(v, sf32)` | 向量 × 标量（A2） |
| 窄化 | `T.simd.vcvt(v, dtype, round="ROUND_R", sat=False, part=0)` | f32→f16/bf16；`part=0` 写低 64 lane |
| 向量存储 | `T.simd.vsts(out[i, s·64], c, m32, "PK_B32", extent=64)` | `extent` = 64 元素 = 128 B；**`PK_B32` 不可换成 `NORM_B16`/`PK_B16`**（真实证据见 U4） |
| 运行期 clamp | `T.min`, `T.ceildiv` | 生成 `min(64, (total_q - …))` |
| 编译 | `tilelang.compile(main, target="ascend", out_idx=[1])` | 输出由 JIT 分配 |

### 3.3 计算伪代码

以下即本目录探针 `history_version/design_probe_packed.py::build` 的代码形态（COMPILE OK，双 dtype），
与 dense 真机实现（`examples/mla_dense_bwd/_dq_land_kernel/_dq_land_kernel.py` 第 150–212 行）同形：

```python
@functools.lru_cache(maxsize=32)          # 与源 @target_cache(maxsize=32) 等价的特化缓存
def _dq_land_kernel(heads, dim_qk, scale, dtype):
    R, SEG = 64, 64
    G = _vector_cores()                       # 56，见 §5.5
    DW = ((dim_qk + SEG - 1) // SEG) * SEG    # 192 → 无 pad
    NSEG = DW // SEG                          # 3
    total = T.dynamic("total_q")              # 复用到两个注解（§3.5.4 E2）

    @T.prim_func
    def _dq_land_main(
        dq_acc: T.Tensor((total, heads, dim_qk), "float32"),
        dq:     T.Tensor((total, heads, dim_qk), dtype),
    ) -> None:
        ntiles = T.ceildiv(total, R) * heads
        with T.Kernel(G) as bx:
            acc = T.alloc_shared((R, DW), "float32")
            out = T.alloc_shared((R, DW), dtype)
            T.annotate_buffer_versions({acc: 2, out: 2})
            cpt = T.ceildiv(ntiles, G)
            it0 = bx * cpt
            it1 = T.min(it0 + cpt, ntiles)
            for it in T.Pipelined(it0, it1, num_stages=2):
                h  = it % heads
                t0 = (it // heads) * R
                n  = T.min(R, total - t0)
                T.copy(dq_acc[t0 : t0 + n, h, :], acc[0:n, 0:dim_qk])
                with T.SimdVF():
                    m = T.simd.pset(32, "PAT_ALL")
                    for i in T.serial(R):
                        for s in T.serial(NSEG):
                            v = T.simd.vcvt(
                                T.simd.vmuls(T.simd.vld(acc[i, s * SEG]), scale),
                                dtype, part=0, sat=False,
                            )
                            T.simd.vsts(out[i, s * SEG], v, m, "PK_B32", extent=SEG)
                T.copy(out[0:n, 0:dim_qk], dq[t0 : t0 + n, h, :])

    kernel = tilelang.compile(_dq_land_main, target="ascend", out_idx=[1])
    _simd_body_gate(kernel, (heads, dim_qk, float(scale), dtype))
    return kernel
```

四点必须照抄的实现约束（否则「编译通过但静默算错」，见 §9.2）：

1. `T.serial(NSEG)` 的段循环、`vld/vmuls/vcvt/vsts` 语句**全部词法上位于 prim_func 体内**；
   不得抽到模块级 python 辅助函数里用 `for s in range(NSEG)` 展开（dense 探针 e11 的反例：SIMD 体被整体丢弃）。
2. `T.dynamic("total_q")` 先绑定到 python 变量再用于两个张量注解（dense 探针 e14 负向 / e15 正向）。
3. `n = T.min(R, total - t0)` 必须同时出现在**两条** `T.copy` 的 region 行数与两侧缓冲切片上
   （目的端 `acc[0:n, 0:dim_qk]`、`out[0:n, 0:dim_qk]` 都要切片）。**packed 下同样必需**：
   尾块由 `T_q % 64` 决定，与请求是否整除无关（§1.6.1 A5）。
4. **`vsts` 的 `dist` 必须是 `"PK_B32"` 且掩码必须是 `pset(32, ...)`**；编译后必须断言设备码里出现
   `vlds_norm`/`vmuls`/`vcvt` 与 `vsts_pack_b32`/`vsts_norm`（`_simd_body_gate`，10.4 实现顺序第 2 步）。

### 3.4 API 可行性确认

| API | 证据（本平台 / 本任务） | 状态 |
|---|---|---|
| `T.Kernel(N)` 一维、无 `threads` | `.agents/platforms/toolchains/ascend/references/programming.md` 第 13 行；`kernel.py:Kernel` 源码（多维被拒） | ✅ 已核对 |
| `T.SimdVF()` + `T.simd.*` | `programming.md` 第 15 行；`simd.py` 导出 `vld/vsts/vcvt/vmuls/vadd/vcadd/pset` 等 | ✅ 已核对 |
| `T.copy` 不做 dtype 转换 | dense 探针 m5/m6 报错原文（§0.5 #3） | ✅ 已核对（负向） |
| `T.copy` 支持动态 extent + 跨步 region | **本目录探针设备码**：`asc_copy_gm2ub_align(..., min(64, (total_q - ((it >> 3) * 64))), 768, ..., 6144, 768)`、`asc_copy_ub2gm_align(..., min(64, ...), 384, ...)` | ✅ 编译级（本任务） |
| `T.annotate_buffer_versions` | `annotations.py`；本目录设备码出现 `(... & 1) * 12288` 两版本寻址与 4 组版本化 flag | ✅ 编译级（本任务） |
| `T.Pipelined(start, stop, num_stages=2)` 动态界 | 本目录设备码 `for (it = block_idx*cpt; it < min((block_idx+1)*cpt, ((total_q + 63) >> 6) * 8); ++it)` | ✅ 编译级（本任务） |
| `T.simd.vcvt(f32→f16/bf16, ROUND_R, sat=False)` | 本目录设备码 `vcvt<half>(..., ROUND_R, RS_DISABLE, PART_EVEN, MODE_ZEROING)`；**数值语义已真机实测**（U5 的 repro：65520→±inf，`sat` 惰性） | ✅ 编译级 + 真机（dense） |
| `T.simd.vsts(..., "PK_B32", pset(32, PAT_ALL), extent=64)` | 本目录设备码 `vsts_pack_b32(v_2, ..., 0, m)` 且 `m = asc_create_mask_b32(PAT_ALL)`；**稠密性已真机实测**（U4 的 repro：256/256） | ✅ 编译级 + 真机（dense） |
| `tilelang.compile(..., out_idx=[1])` + 动态首维 | 本目录双 dtype **COMPILE OK**（`out_idx=[1]` 生效，输出 shape 由输入绑定的 `total_q` 推出）；真机 L0-1（dense） | ✅ 编译级 + 真机（dense） |

### 3.5 技术约束确认

#### 3.5.1 本项目已知限制检查

| 限制（来源） | 本设计的处置 |
|---|---|
| `T.Kernel` 只接受一维网格、无 `threads`（`programming.md:13`） | 一维 `T.Kernel(56)`；线程域概念用「核 × 向量 lane」替代 |
| DMA copy 不能 cast（语义检查，dense 探针 m5/m6） | cast 放在 `T.SimdVF` 内由 `vcvt` 完成 |
| 整 buffer ↔ 整 buffer 拷贝要求元素总数相等；局部拷贝用显式 region（`programming.md:37`） | 两条 copy 的 UB 侧都显式切片 `[0:n, 0:dim_qk]` |
| 自动多缓冲要求「在可调度的串行循环内、且先写后读」（AutoSchedule 警告文本） | `out` 由 VF 体写、被 copy 读，且循环体内**无 python 层展开的孤立语句** → 本目录探针无该警告 |
| `ascend.batch_schedule` 跨 block barrier 未认证（`workflow.md:29`；`CG-950-batch-barrier`） | R1-e（融合进上游 launch）据此否决 |
| UB 253952 B（`capacity_bytes`） | 147456 B（含 2 版本），58% ✓（§4.5） |
| `dim_qk % 16 == 0`（源码第 277 行 / packed wrapper 内联同一检查） | packed manifest 全部 192；本设计对任意 `dim_qk % 16 == 0` 用 `DW = roundup(dim_qk, 64)` + 切片 copy 统一处理（§9.3） |
| 行首地址 32 B 对齐（DMA 粒度） | fp32 行 192×4 = 768 B、fp16 行 384 B，均为 32 B 整数倍 ✓ |
| `TL_ENABLE_FAST_MATH` 会放开 fast-math | **不启用**（§7.3），保证次正规/舍入与 golden 一致 |
| packed 行轴可能极长（`T_q = 32768`） | GM 偏移在设备码里提升到 `int64_t`（本目录设备码 `((int64_t)it >> 3) * (int64_t)98304`）✓ |

#### 3.5.2 参考实现差异说明

| 项 | 源（CUDA） | 本设计（ascend） | 差异理由 |
|---|---|---|---|
| 网格 | `(ceildiv(T_q,64), H)` 二维 | `(56,)` 一维 + tile 解码 | 平台限制（§0.5 #1） |
| 并行单位 | 128 线程 × 96 元素 | 56 核 × 64-lane 向量 × 核内串行 tile | 硬件执行模型（§0.5 #5） |
| cast 位置 | 写回 copy 内 | `T.SimdVF` 内 `vcvt` | DMA 不做 cast（§0.5 #3） |
| 中间缓冲 | 无（fragment 直接写回） | `out (R,DW) dtype` UB 暂存（vsts 落点） | 向量寄存器不能直写 GM |
| 尾块 | 隐式越界守卫 | 显式运行期 `min()` clamp | 不依赖未文档化行为（§0.6 RD3） |
| 流水 | 无（一 tile 一 block） | `T.Pipelined` 2 阶段 + 2 版本 | 单核内 load/compute/store 重叠需要循环 |
| fast-math | 开 | 关 | 单次乘无可收缩运算；避免舍入/次正规改变 |
| 调用方 view | dense 侧 `view` 进出 BSHD | packed 侧**不 view** | 契约为 packed THD；对本 kernel 不可见（§11 P1/P3） |

#### 3.5.3 本项目同类实现参考

| 参考文件 | 相似度 | 借鉴点 | 差异 |
|---|---|---|---|
| `examples/mla_dense_bwd/_dq_land_kernel/_dq_land_kernel.py` | **最高（同一函数，真机 31/31）** | 全部实现形态：一维网格、双版本、`PK_B32` 段链、clamp、`_simd_body_gate`、golden 与分层测试 | 无（本任务即复用） |
| `examples/mla_varlen_bwd/_delta_kernel/DESIGN.md` | 高（同 op 的 packed launch） | packed 调用路径的取证口径与写作范式 | 它是行归约（`vcadd`）且 packed 改 layout，本算子是纯映射且 layout 不变 |
| `examples/mla_dense_bwd/_delta_kernel/_delta_kernel.py` | 高 | 一维 `T.Kernel`、`T.annotate_buffer_versions`、`vcvt` 的两半语义、目的端切片 | 它是行归约（`vcadd`），本算子是纯映射（`vmuls+vcvt`） |
| `examples/paged_cache_write/_paged_cache_write_kernel/_paged_cache_write_kernel.py` | 中 | 目标 GM 跨步写回 | 它走 SIMT 直写，本算子走 SIMD 段链 |
| `.agents/platforms/toolchains/ascend/kernel_template.py` | 中 | 一维网格骨架、尾块形式 | 它 N 静态、用 `if bx == full` 分支；本算子 N 动态、用运行期 clamp |

#### 3.5.4 编译探针与负向对照

**本任务目录内新增的探针（本轮执行，无卡）**

| 探针 | 形式 | 结果 |
|---|---|---|
| `history_version/design_probe_packed.py`（本任务） | dense 真机形态的 packed 复现：56 核持久 + `T.Pipelined` 动态界 + 2 版本 + 运行期 clamp + `PK_B32` + `out_idx=[1]`，**fp16 / bf16 各编译一次** | **双 dtype COMPILE OK**；设备码各 41 行；`compute=['vlds_norm','vmuls','vcvt']`、`store=['vsts_pack_b32']`；`total_q_symbolic=True`；快照 `history_version/probe_packed_gen_float16.txt`（sha256 前缀 `d0fde4920bf2c7ac`）、`history_version/probe_packed_gen_bfloat16.txt`（`914c64f7abb2f600`） |

**复用的负向/正向对照（dense 真机前的编译级证据，本任务未重跑，仅作设计依据）**

| 探针 | 形式 | 结果 |
|---|---|---|
| e15 | 56 核持久 + 动态界 + 2 版本 + 运行期 clamp + `out_idx=[1]`（初版 `NORM_B16` 落盘） | COMPILE OK；`history_version/probe_e15_gen.txt` |
| e10 | 动态网格 `ceildiv(total,R)·H`，一 tile 一 block（无核内循环） | COMPILE OK，SIMD 体完整——**结构备选**，但核内无重叠 |
| e11（反例 1） | 段循环抽到模块级 python 辅助函数、用 `range(3)` 展开 | COMPILE OK 但**整个 SIMD 体被静默丢弃**（设备码只剩两个 mask 创建）→ §9.2 E1 |
| e14（反例 2） | `T.dynamic("total_q")` 直接写在两个张量注解里 | COMPILE FAIL：`Cannot infer symbolic variable total_q required by callee-allocated output buffer dq` → §9.2 E2 |
| e12/e13 | `num_stages=2` / `num_stages=1` 对照 | 均 COMPILE OK，且两份**设备码逐行相同** → 本平台的流水由 `T.annotate_buffer_versions` 的显式版本声明驱动；Stage 4 的串行基线必须**去掉版本声明**（§6.3） |
| m5/m6 | `T.copy(acc_f32_ub, dq_f16_gm)`（copy 层 cast） | COMPILE FAIL：`DMA copies cannot perform type casting` |
| n5 | SIMT 投影（备选 A3） | COMPILE OK，设备码 `float2` 载入 + `__float22half2_rn` |

**真机证据（dense，本任务未重跑——加速卡被占用；D 类，带复现脚本）**

| 事实 | 结论 | 复现 |
|---|---|---|
| `vcvt` 窄化 lane 摆放 | `PK_B32` + 32 位掩码是唯一稠密形态（256/256），`PK_B32`+16 位掩码 128/256，`NORM_B16`+16 位掩码 4/256 | `examples/mla_dense_bwd/_dq_land_kernel/repro/CONST-950-vcvt-narrow-store-pack-b32.py` |
| `vcvt` 的 `sat` 标志 | f32→f16 上 `sat=True/False` 逐项相同，均为 IEEE（65519→65504、65520→±inf） | `examples/mla_dense_bwd/_dq_land_kernel/repro/CONST-950-vcvt-rs-disable-ieee.py` |
| 动态 `T_q` + 动态界流水 + 2 版本 + `out_idx` | 真机可用（同几何连跑 20 次逐位一致；单版本对照逐位一致） | dense `_dq_land_kernel.py --level L0`（L0-7 / P5） |
## 4. 数据规格与内存规划

### 4.1 输入张量

| 名称 | Shape | dtype | 布局 | 说明 |
|---|---|---|---|---|
| `dq_acc` | `(T_q, H, D_qk)`，`T_q` **动态** | `float32` | 行主序连续；行内 `D_qk` 连续，行 stride = `H·D_qk·4` B | 由 `_bwd_kernel` 写好；packed wrapper 以 `torch.zeros(q.shape, dtype=torch.float32, device=q.device)` 现场分配（`mla_varlen_bwd.py` 第 223 行），`q` 已是 packed `(T_q,H,D_qk)`，故恒连续 |

packed manifest workload 的 `(T_q, H, D_qk)`：`(640, 8, 192)`、`(32768, 128, 192)`、
`(32256, 128, 192)`、`(32768, 64, 192)`（`T_q = Σ_b q_len[b]`）。
测试用例另有 `T_q = 64`（单请求 `[64]`）、`308`（`[200,17,91]`）、`384`（`[128,256]` / `[320,64]`）。

### 4.2 输出张量

| 名称 | Shape | dtype | 布局 | 说明 |
|---|---|---|---|---|
| `dq` | `(T_q, H, D_qk)` | `float16` \| `bfloat16` | 行主序连续；行 384 B，行 stride = `H·D_qk·2` B | `out_idx=[1]` 由 JIT 分配（shape 由输入绑定的 `total_q` 推出），wrapper 直接返回、**不做 view** |

与 manifest 的 `dq: {shape: "[T_q, H, D_qk]"}` 逐字一致（§0.2）。

### 4.3 中间缓冲区

| 名称 | Shape | dtype | scope | 字节数 | 版本数 |
|---|---|---|---|---|---|
| `acc` | `(R, DW) = (64, 192)` | `float32` | UB（`shared.dyn`） | 49152 | 2 |
| `out` | `(R, DW) = (64, 192)` | `float16`/`bfloat16` | UB（`shared.dyn`） | 24576 | 2 |
| 掩码 `m` | 向量谓词寄存器（32 位 lane） | `vector_bool` | 寄存器 | — | — |

无 GM 级中间缓冲（这是本算子与同批次另两个 launch 的又一个区别）。

### 4.4 内存搬运路径

```
GM (dq_acc, fp32)
  └─MTE2: T.copy(region n×1×192)─▶ UB acc[v]（v = it & 1）
                                      └─V: vld → vmuls → vcvt → vsts(PK_B32) ─▶ UB out[v]
                                                                         └─MTE3: T.copy(region n×1×192) ─▶ GM (dq, dtype)
```

- 不经 L1、不经 L0A/L0B/L0C（纯 Vector kernel；`T.Kernel` 内无 Cube 段）。
- 搬运粒度：每 tile 一次 MTE2（本目录设备码 `asc_copy_gm2ub_align(..., row=min(64, total_q−…), 768, …, 6144, 768)`）
  与一次 MTE3（`asc_copy_ub2gm_align(..., row=min(64, total_q−…), 384, …, 3072, 384)`），段宽 768 B / 384 B。
- **packed 无关**：搬运 region 的形态由 `(t0, h, n)` 决定，与请求边界无关；`stride` 仍是 `H·D_qk·B_e`，
  因为 packed 张量在内存里就是连续的行轴（请求之间没有填充）。

### 4.5 UB 内存预算

| 项 | 计算 | 字节 |
|---|---|---|
| `acc` × 2 版本 | `64·192·4·2` | 98304 |
| `out` × 2 版本 | `64·192·2·2` | 49152 |
| **合计（逻辑）** | | **147456（144 KB）** |
| 按平台文档的自动分配膨胀系数上界 ×1.12 复核（口径见 `.agents/skills/tilelang-op-optimize/references/pattern-library/constants.md` 的 `CONST-capacity-910B2C`，**910B2C 口径、跨型号假设**） | `147456 × 1.12` | 165151（161 KB） |
| 型号 UB 容量（`Ascend950PR_9579.ini [AICoreSpec] ub_size`，平台 `capacity_bytes.UB`） | — | 253952（248 KB） |
| 占用率 | `147456 / 253952` | **58.1%** |

- 余量 106 KB；`R = 96` 会到 221184 B（×1.12 = 247726 B，逼近上限）→ **`R` 取 64 是为 Stage 4 留出膨胀余量**。
- 尾部列：`DW = 192 = D_qk`，无 pad 列，因此不需要 `T.clear` 两侧清零那类补丁。
- packed 无关：UB 预算只由 `(R, DW, dtype)` 决定，与 `T_q` 无关。

### 4.6 动态轴定义

| 轴 | 是否动态 | 处理 |
|---|---|---|
| `T_q`（张量第 0 维） | **动态**（`T.dynamic("total_q")`） | 只出现在：`ntiles = ceildiv(T_q, R)·H`、`n = min(R, T_q − t0)`、`it1 = min(...)` 三处标量与两条 `T.copy` 的 region extent |
| `H` | 静态（工厂参数） | tile 解码的除数（2 的幂 → 移位/掩码） |
| `D_qk` | 静态（工厂参数，192） | 决定 `DW/NSEG` 与向量段数 |
| 请求边界（`cu_seqlens_*`、`q_len`） | **不出现** | 本 launch 的参数表里没有它们（源码第 94–95 行、packed wrapper 第 227 行） |

注意：工厂签名 `(heads, dim_qk, scale, dtype)` **拿不到 `T_q`**（§0.1 契约），因此 `T_q` 只能是运行期符号；
这也意味着**不能**用「编译期尾块分支」（同批次 `_delta_kernel` 的做法）处理非整除，只能用运行期 clamp。
**packed 的一个直接推论**：由于 `T_q` 不参与特化，同一个编译产物服务**所有** packed 长度与**所有**请求切分
（§8.3 P7 会实测这一点）。

### 4.7 JIT 配置

```python
@functools.lru_cache(maxsize=32)          # 与源码 @target_cache(maxsize=32) 等价的特化缓存
def _dq_land_kernel(heads, dim_qk, scale, dtype): ...
    return tilelang.compile(_dq_land_main, target="ascend", out_idx=[1])
```

- 特化键 = `(heads, dim_qk, scale, dtype)`（与源一致；`scale` 是编译期常量，随调用方 `sm_scale` 变化——
  packed 测试的 `sm_scale = 0.05` 用例会命中另一份特化）。
- **不启用** `pass_configs`：源里的 `TL_ENABLE_FAST_MATH` 被舍弃（§0.5 #6、§7.3）；
  `-O3` / `-DENABLE_bf16` 一类 nvcc flag 在 ascend 侧无对应项。
- `out_idx=[1]` 需要符号绑定（§3.5.4 E2）。
- 编译后必须过 `_simd_body_gate`（§10.4 第 2 步）。

## 5. Tiling 策略

### 5.1 计算类型

**纯逐元素映射（单步）+ 带宽受限**：每元素 1 次 fp32 乘 + 1 次窄化，读 4 B、写 2 B。没有矩阵乘（Cube 无用）、
没有归约（无跨 lane 合并）、没有 host 侧预处理、**没有 per-request 处理**。

**GEMM 相关约束的豁免（显式声明，避免误判为遗漏）**：本算子**不含 `T.gemm`**（§3.3 伪代码无矩阵乘），因此
① 平台的矩阵分形/`block_M·block_N` 约束不适用；② `L0C` 容量约束（256 KB）不适用——`L0C` 是本设计**不使用**的层级
（§4.3 的中间缓冲全在 UB）；③ GEMM 非整除的「host padding + crop」策略不适用，本算子的非整除由运行期行
clamp 处理（§5.2）。同理，平台的「三维 Kernel」限制在本设计中不构成约束：本设计用一维 `T.Kernel(56)`（§3.5.1）。

### 5.2 Block 划分

| 参数 | 取值 | 依据 |
|---|---|---|
| `R`（行/tile） | **64** | UB 预算：`2·R·192·4 + 2·R·192·2 = 147456 B`（`R = 96` 会到 221184 B，逼近 248 KB 上限，§4.5） |
| `DW`（行宽） | `roundup(D_qk, 64) = 192` | 段宽；packed manifest 几何下无 padding |
| `NSEG`（段/行） | `DW / 64 = 3` | 一行 192 fp32 = 3 个满 64-lane 寄存器 |
| tile 覆盖 | 1 个 head × 64 行 | 段宽 768 B（GM 侧），已过「数百字节」判据（§1.6.3） |
| 逻辑 tile 数 | `ntiles = ceildiv(T_q, 64) · H`（**运行期**） | packed manifest：80 / 65536 / 64512 / 32768 |
| 核网格 | `G = 56`（静态） | §5.5 |

**非整除处理策略**（`T_q % 64 ≠ 0`）：**不做 host 侧 padding / crop**（会多一次 GM 往返，且契约里 `dq` 的
shape 由 JIT 按输入绑定，host 改 shape 会越界）。改为 kernel 内运行期 clamp：
`n = T.min(R, T_q − t0)`，两条 `T.copy` 的行数与 UB 切片同时用 `n`；行循环保持静态 `R`，多算的 `R − n` 行
落在 UB 的无效行上且**不被搬出**（§0.6 RD3）。设备码证据见
`examples/mla_varlen_bwd/_dq_land_kernel/history_version/probe_packed_gen_float16.txt` 第 25/33 行。
**packed 下的重要性**：manifest 四个几何的 `T_q` 恰好都是 64 的倍数，但**请求级长度是 ragged 的**
（`[512,1024,…,16384]`、`[256,384]`），测试用例还含 `T_q = 308`、`64`；因此该策略是正确性必需项，
不能因 manifest 巧合而省略（§1.6.1 A5）。

### 5.3 约束分析

| 约束 | 检查 |
|---|---|
| 段整除 | `192 % 64 == 0` ✓（`D_qk % 16 == 0` 由算子 refusal 保证；非 64 倍数时用 `DW` pad + 切片 copy 统一处理，见 §9.3） |
| 行对齐 | fp32 行 768 B、fp16 行 384 B，均 32 B 整数倍 ✓ |
| UB 容量 | 147456 B / 253952 B = 58.1%（×1.12 膨胀后 65%）✓ |
| 向量寄存器 | 每段 1 个 `float32x64`，同时活跃 1 个 + 1 个 `halfx128` 结果 ✓ |
| 索引宽度 | 设备码把 GM 偏移提升到 `int64_t`（本目录 fp16 设备码 `((int64_t)it >> 3) * (int64_t)98304`）→ 最大 packed workload 的 `dq_acc` 3.22 GB（`4E`）/ `dq` 1.61 GB（`2E`）仍安全 ✓；`ntiles` 与 `cpt` 是 int32，最大 65536 ✓ |
| 动态 extent | 只出现在两条 `T.copy` 与三个标量上（§4.6）✓ |
| packed 行轴 | 无额外约束：请求之间无填充，行 stride 恒定 ✓ |

### 5.4 注意事项

- **不要把 `R` 调大**：`R = 96/128` 会让 UB 预算逼近/超过上限（Stage 4 若想加版本数或加大 tile，必须先
  按 §4.5 重算）。
- **不要让 tile 跨 head**：跨 head 的 region 虽然更连续（`HH·768 B`），但 UB 随 `HH` 线性涨（§1.6.3 A2 已否决）。
- **不要让 tile 跨请求做特殊化**：packed 请求长度只在 `_bwd_kernel` 里有意义；在本 kernel 里引入
  「按请求对齐的 tile」既无收益又会破坏 `T_q` 单一动态符号的简洁性（且 `dq` 的 shape 契约是 `[T_q,H,D_qk]`，
  与请求无关）。
- **`H` 是 2 的幂时**模/除退化为移位与掩码（packed manifest 的 8/64/128 全部满足）；若未来出现非 2 幂的
  `H`，tile 解码会多两条整数除法（每 tile 一次，可接受，但应在 Stage 4 复核）。
- **尾 tile 的冗余行**：单次最多多做 63 行（≈48 KB 的向量运算），相对总量可忽略。

### 5.5 分核策略（物理核数适配）

> 三要素的定义与判定口径见 `.agents/skills/_shared/standards/core-split-strategy.md` §1（本文件是权威版本，本节按 §2.1
> 的 Stage 1 要求给出三要素；核内循环的边界限制按任务平台 ruleset，见本节 ③ 末）。

**① 逻辑核数计算**：本算子的逻辑并行单元是 `(行块, 头)` tile：

```
num_logical_kernels = ceildiv(T_q, R) · H = ceildiv(Σ q_lens, 64) · H
```

| workload（packed manifest / 测试） | `T_q = Σ q_lens` | `H` | 逻辑核数 |
|---|---|---|---|
| `mla-varlen-bwd-smoke`（`[256,384]`） | 640 | 8 | 80 |
| `ds-v3-8x4k`（8×4096） | 32768 | 128 | 65536 |
| `ds-v3-mixed`（`[512,…,16384]`） | 32256 | 128 | 64512 |
| `kimi-k2-4x8k`（4×8192） | 32768 | 64 | 32768 |
| 测试 `smoke-varlen-bwd-single-request`（`[64]`） | 64 | 4 | **4** |

**② 物理核数依据（实查记录）**：目标型号 **ascend-950pr-9579**，Vector 物理核数 **56**（Cube 28）。

- 平台记录（`python3 .agents/tools/statectl.py platform --task-dir examples/mla_varlen_bwd/_dq_land_kernel`）：
  `hardware.observed_cores = {cube: 28, vector: 56}`，`core_source` = 「Ascend950PR_9579.ini [SoCInfo]
  ai_core_cnt=28, cube_core_cnt=28, vector_core_cnt=56; runtime probe rtGetAiCoreCount()=28 on
  Ascend950PR_9579, 2026-11-23」。
- 实现期（Stage 3/4）在同一设备上应再按平台规则实查一次：

  ```python
  import torch, torch_npu  # noqa: F401
  props = torch.npu.get_device_properties(torch.npu.current_device())
  vector_cores = props.vector_core_num      # 本机返回值：56
  cube_cores   = props.cube_core_num        # 本机返回值：28
  ```

  工厂里 `G = props.vector_core_num`（纯 Vector 算子用 AIV 数，**不乘 2、不用 `NPUUtils`**——
  `.agents/platforms/toolchains/ascend/workflow.md` 第 55 行要求分别记录 Cube/Vector）。取不到设备时
  （例如 host 侧预编译，本设计的无卡探针就是这种情形）回退到本平台记录的常量 56，并在日志里注明来源。

**③ 规模判定与分核方案**：**极大规模 → persistent 核内串行**。

- 判定：packed manifest 的最小 workload 逻辑核数 **80** 已 `> 56`（≈1.43×），最大 65536（≈1170×）——
  没有任何 manifest workload 落在「逻辑核数 ≤ 物理核数」，也无法通过调整 `R` 把核数压到 56 附近
  （`R` 受 UB 限制最大 64，压核数就得把 `R` 提到上千行，UB 需要数 MB ≫ 248 KB）。
- 方案：**固定启动 `G = 56` 个核**，每核用运行期连续区间 `[bx·cpt, min((bx+1)·cpt, ntiles))`
  （`cpt = ceildiv(ntiles, 56)`）串行走多个逻辑 tile，摊薄核启动开销；区间内再叠 `T.Pipelined`
  两级流水。核内循环的边界限制按平台 ruleset：`T.Pipelined` 的动态界已编译验证（本目录探针），
  循环 API 只用 `T.serial` / `T.Pipelined`（不用未认证的跨 block barrier）。
- 负载均衡：`cpt` 上取整使各核 tile 数差 ≤ 1：

  | workload | `ntiles` | `cpt` | 平均 tile/核 | 最大失衡 |
  |---|---|---|---|---|
  | `mla-varlen-bwd-smoke` | 80 | 2 | 1.43 | **+40%**（`cpt = 2` → 40 核各 2 个 tile、16 核区间为空立即退出，最长路径 2 vs 理想 1.43；该几何是 launch-bound，见 §1.6.0 估算 5.6–7.9 µs） |
  | `ds-v3-8x4k` | 65536 | 1171 | 1170.3 | +0.06% |
  | `ds-v3-mixed` | 64512 | 1152 | 1152.0 | **0.00%**（整除） |
  | `kimi-k2-4x8k` | 32768 | 586 | 585.1 | +0.15% |
  | 测试 `[64]` | 4 | 1 | 0.07 | 52 核空转（退化，见下） |

- 退化场景：`ntiles < 56` 时（如测试 `[64]` 的 4 个 tile）只有 `ntiles` 个核有活，其余核立即退出
  （空区间 → `Pipelined` 零迭代），**不需要额外分支**；与 dense 的 `T_q = 1` 情形同构（§11 P8）。

## 6. 循环与调度结构

### 6.1 循环结构总结

| 层 | 循环 | API | 说明 |
|---|---|---|---|
| L0 | 核网格 | `T.Kernel(56)` | 静态一维，56 个 AIV |
| L1 | 核内 tile 循环 | `T.Pipelined(it0, it1, num_stages=2)` | **动态界**（`it0/it1` 运行期）；2 版本 ping-pong |
| L2 | 行循环 | `T.serial(R)` | 静态 64 次；每轮 3 段向量指令 |
| L3 | 段循环 | `T.serial(NSEG)` | 静态 3 次；**必须留在 prim_func 体内**（§9.2 E1） |
| — | GM→UB / UB→GM | `T.copy` × 2 | 由 AutoSchedule 排入流水（MTE2/MTE3 各 1 次/tile） |

- **`T.Parallel` 不用**：SIMD 模式下逐元素工作由 `T.simd.*` 的 64-lane 指令承担（`T.Parallel` 属于
  SIMT 域，混用未验证）。
- **`T.Persistent` 不用**：平台提供 `T.Persistent(domain, wave_size, index, group_size, num_stages)`
  （`tile_schedule.py` 家族），语义是二维 tile 的 swizzle 分组，本算子是一维 tile 流；手工连续区间
  与之等价且已编译验证。
- **`T.serial` 的段循环与 `T.Pipelined` 的 2 阶段是两条独立机制**：前者展开 192 列，后者重叠 tile。
- **packed 无新增循环**：没有 per-request 循环、没有按 `cu_seqlens` 做边界判断的循环（V11）。

### 6.2 循环伪代码

```python
with T.Kernel(56) as bx:                       # L0
    acc, out = alloc + annotate_versions(2)
    for it in T.Pipelined(it0, it1, 2):        # L1 动态界
        h, t0 = it % H, (it // H) * R
        n = T.min(R, total - t0)
        T.copy(dq_acc[t0:t0+n, h, :], acc[0:n, 0:DQ])          # MTE2
        with T.SimdVF():
            m = T.simd.pset(32, "PAT_ALL")
            for i in T.serial(R):              # L2
                for s in T.serial(NSEG):       # L3
                    v = T.simd.vcvt(T.simd.vmuls(T.simd.vld(acc[i, s*64]), scale),
                                    dtype, part=0, sat=False)
                    T.simd.vsts(out[i, s*64], v, m, "PK_B32", extent=64)
        T.copy(out[0:n, 0:DQ], dq[t0:t0+n, h, :])              # MTE3
```

### 6.3 流水线优化

- `num_stages=2` + `T.annotate_buffer_versions({acc: 2, out: 2})`：本目录设备码生成了 **4 组版本化 flag**
  （`PIPE_MTE3→PIPE_V`、`PIPE_V→PIPE_MTE2` 各 2 个 id，循环内另有 `MTE2→V`、`V→MTE3` 的版本化
  set/wait），说明 `load(k+1)` 可与 `compute(k)/store(k)` 重叠。
- **`num_stages` 与版本声明的关系（dense 的编译级结论，本目录形态相同）**：`num_stages=2` 与
  `num_stages=1` 的设备码**逐行相同**——只要写了 `T.annotate_buffer_versions({acc: 2, out: 2})`，
  AutoSchedule 就按 2 版本生成 ping-pong flag，`num_stages` 不再改变结构。因此 Stage 4 若要做「无重叠」
  的串行基线，必须**删掉版本声明**（或声明 1 版本），而不是把 `num_stages` 调成 1。
- 更深流水（3 版本）需要 216 KB UB（×1.12 ≈ 242 KB），无安全余量 → 不取。

### 6.4 尾块处理

- **行数**：`n = T.min(R, T_q − t0)`，两条 `T.copy` 的 region 行数与 UB 切片都随 `n`。
- **计算**：行循环固定 `R`，`[n, R)` 行读未初始化 UB、结果写进 `out` 的这批行，**永不搬出**。
  （这是有意的「尾块多算」，与同批次 `_bwd_kernel` 的既定取舍一致；不引入逐行谓词 → 热路径无分支。）
- **写回保护**：设备码的 store 侧带同一 `min(64, (total_q − ((it >> 3) * 64)))` extent
  （本目录 `probe_packed_gen_float16.txt` 第 33 行），与 load 侧对称。
- **`T_q < R`**（如测试 `[64]` → `T_q = 64`，或 `[200,17,91]` 的 `T_q = 308 → n = 52`）：尾 tile 分别
  只搬 64 行 / 52 行。
- **packed 边界与 `T_q` 的交互**：请求长度之和决定 `T_q`，但**任何一个请求的结尾都不会落在 tile 中间后
  造成错误**——因为本算子不读请求边界，任何行都独立地做同一次乘 + 窄化（§11 P4/P7）。

## 7. 同步策略

### 7.1 同步模式

**自动同步**：平台 `ascend/pipeline.py` 的 `allow_autoschedule` 默认开启，`InsertSync` 与
`MaterializeMultiBuffer` 自动插入 `asc_sync_notify` / `asc_sync_wait` 与版本化 flag；本设计**不写任何
手工同步**（`programming.md` 第 53 行：未验证的手工跨核 flag 配对不作为模板）。

### 7.2 同步点说明（本目录设备码）

| 位置 | 生成的同步 | 作用 |
|---|---|---|
| 循环前 | `notify(MTE3→V, id 0/1)`、`notify(V→MTE2, id 0/1)` | 预置两个版本的初始状态 |
| 每轮开头 | `wait(V→MTE2, it&1)` | 等上一轮该版本缓冲被 V 读完 |
| MTE2 之后 | `notify(MTE2→V, it&1)` | 告知 V 数据就绪 |
| V 之前 | `wait(MTE3→V, it&1)`、`wait(MTE2→V, it&1)` | 等 store 释放缓冲 + 等 load 完成 |
| V 之后 | `notify(V→MTE3, it&1)`、`notify(V→MTE2, it&1)` | 告知 store 可写、告知下一轮 load 可覆盖 |
| 循环尾 | `wait(MTE3→V, 0/1)`、`wait(V→MTE2, 0/1)` | 收尾 |

同步区间与双缓冲版本一一对应（`it & 1`），不需要额外的跨核同步（56 个核之间**没有数据依赖**：
每个 tile 的 `(t, h)` 只属于一个核）。**packed 无关**：同步只依赖 tile 循环的 trip count，不依赖请求边界。

### 7.3 pass_configs 配置

| 配置 | 取值 | 理由 |
|---|---|---|
| `TL_ENABLE_FAST_MATH` | **不设置（关）** | 源开了它，但本算子只有一次 fp32 乘，没有可收缩的运算链；fast-math 可能改变次正规/舍入（§8.2 要求与 golden 逐位一致） |
| `compile_flags` | **不设置** | `-O3` / `-DENABLE_BF16` 是 nvcc flag；ascend 的 bf16 由工具链支持 |
| `annotate_buffer_versions` | `{acc: 2, out: 2}` | 显式声明 2 版本（与 `num_stages=2` 对齐） |
| 其它 pass | 默认 | AutoSchedule / InsertSync / MaterializeMultiBuffer 全部默认开启 |

## 8. 验证方案

### 8.1 Golden 函数

Golden **只依据 §0.1 的语义**（不参考 §0.6 的 NPU 算法：不出现 `vld/vcvt/分块/流水/56 核`任何概念），
与源 docstring 逐字对应，且**与 packed / dense 无关**：

```python
def golden_dq_land(dq_acc, scale, dtype):
    """dq = (dq_acc.float() * scale).to(dtype), elementwise.

    dq_acc: (T_q, H, D_qk) float32 (CPU tensor, as produced by _bwd_kernel);
    returns (T_q, H, D_qk) of `dtype`.  Independent of the kernel structure and
    of the packed/dense call path: one float32 multiply, one round-to-nearest-
    even narrowing, no blocking, no vector ops, no request offsets.
    """
    return (dq_acc.detach().cpu().float() * float(scale)).to(dtype)
```

- 对 inf/NaN 的行为由 torch 的 cast 决定（`NaN→NaN`、`|y| ≥ 65520 → ±inf`），与 §0.1 的语义一致。
- 任务要求的形式 `(dq_acc.float() * scale).to(dtype)` 与上式等价（`scale` 已是 python float）。
- 复算独立性检查：golden 里没有 `vcvt`、没有 tile、没有 `T_q` 假设、没有 `R`、没有 `cu_seqlens`——
  即使把 §0.6 的 NPU 算法整体推翻，golden 不变。

### 8.2 精度标准

| dtype | atol | rtol | 来源 |
|---|---|---|---|
| `float16` | `5e-3` | `1e-3` | `examples/TileOPs/tests/ops/test_mla_varlen_bwd.py`（`_RTOL[float16] = 1e-3`、`_ATOL_FLOOR = 5e-3`，即 `atol = max(5e-3, rtol)`） |
| `bfloat16` | `1.6e-2` | `1.6e-2` | 同上（`_RTOL[bfloat16] = 1.6e-2`，atol 取 `max(5e-3, 1.6e-2)`） |

**附加的强判据（本设计建议 L0 同时断言）**：本算子是「一次乘 + 一次窄化」，在 `vcvt(ROUND_R, sat=False)`
与 golden 同为 IEEE RNE 的前提下，输出应**逐位相等**（0 ulp）。dense 在真机上已达成全部 31 个用例
`max_diff=0` / `bit_diff=0`，因此 packed 侧也应以此为标准：L0 除容差断言外，还断言 `torch.equal(got, gold)`；
若不相等但落在容差内，必须报告最大 ulp 与差异元素数，并按「先怀疑落盘分布 / 掩码」的顺序排查
（dense Stage 3 的教训：容差内但非逐位，几乎总是 `vsts` 分布问题）。

### 8.3 L0 门槛测试计划

**复用策略（本设计的实现建议）**：Stage 3 可直接复用 dense 的
`examples/mla_dense_bwd/_dq_land_kernel/_dq_land_kernel.py` 全套骨架（`golden_dq_land`、`_compare`、
`_make_input`、`_corner_fill`、`_simd_body_gate`、`run_case`），只把用例几何从 `(B, S, H)` 换成 packed
`q_lens` 列表；**唯一需要新写的是 P7（多切分同二进制）**。

| 编号 | 几何（packed） | `T_q` | dtype | 覆盖的设计点 |
|---|---|---|---|---|
| L0-1 | `q_lens=[256,384]`，H=8（manifest smoke） | 640 | fp16 | 主路径 + `out_idx` 动态输出 + **多请求 packed**（最先跑，风险最高：R1/R3） |
| L0-2 | 同上 | 640 | bf16 | bf16 窄化路径（`vcvt` 的 `bfloat16` 目标） |
| L0-3 | `q_lens=[200,17,91]`，H=4 | 308 | fp16 | **ragged 请求 + 尾块 clamp**（`308 = 4×64 + 52`，最后 tile `n = 52`）；三条不规则请求边界；canary 验不越界写 |
| L0-4 | `q_lens=[64]`，H=4 | 64 | fp16 | 退化 packed（单请求；`ntiles = 4 < 56`，52 核空转） |
| L0-5 | `q_lens=[320,64]`，H=4 | 384 | fp16 | `T_q` 整除 64（无尾块）的对照路径；`ntiles = 24` |
| L0-6a | `q_lens=[64,64]` + **角点注入**，H=4 | 128 | fp16 | `vcvt` 数值语义：`±inf`、`NaN`、`±0`、次正规（±1e-38/±1e-8）、上界 65504/65519/65520、1e5、7e4（对应 §1.6.1 的 A3/C2 角点）；断言 `torch.equal`（NaN 只比位置） |
| L0-6b | 同上 | 128 | bf16 | 同上；**溢出角点需 `scale = 1.0`**（bf16 有限域几乎等于 fp32 域，`mx*1.5/scale` 会先溢出 fp32——dense Stage 3 的实证教训） |
| L0-7 | `q_lens=[4096]×8`，H=128（`ds-v3-8x4k`） | 32768 | bf16 | 满载 65536 个逻辑 tile、每核 1171 tile 的稳态；同时作为 Stage 4 baseline |
| L0-8 | P0：设备码 SIMD 链 + 特化缓存 | — | — | **`_simd_body_gate`**：每个编译产物必须含 `vlds_norm`/`vmuls`/`vcvt` 且含 `vsts_pack_b32`/`vsts_norm`；同一 `(heads, dim_qk, scale, dtype)` 两次调用返回同一对象 |
| L0-9 | 非法契约：`dtype=float32`、`heads=0`、`dim_qk=190`、`scale=0`、`scale=-0.1` | — | — | 工厂拒绝（与 dense 同一组） |

**L0 通过判据**：`atol/rtol` 断言 + `torch.equal` 断言 + 以下定点检查：

| 编号 | 检查 | 方法 |
|---|---|---|
| P1 | 无越界写 | 输出缓冲前后各放一块 `torch.full(..., sentinel)` 的更大张量（dense 的 `canary`），跑两次后检查 sentinel 不变 |
| P2 | 尾块不丢行/不多行 | L0-3 / L0-5 与 golden 逐元素核对（含最后 `n` 行） |
| P3 | 单次舍入（非两次） | L0-6 里构造能让 16 位域两次舍入暴露差异的值（`verify_equiv.py` 的 C1 样本），断言与 golden 逐位一致 |
| P4 | 非饱和 | L0-6 的 1e5 / 3e38 角点必须输出 `±inf`（若得到 65504，说明落到了饱和路径：与 dense 的「`sat` 惰性」实测矛盾，须改用 `sat=True` 并在设计附录登记语义偏差，或退回 SIMT 投影备选 A3） |
| P5 | 流水正确性 | 同一几何连跑 ≥ 20 次结果一致（排除 2 版本 ping-pong 竞态）；对照变体 = **去掉 `annotate_buffer_versions`**（单版本串行形态，见 §6.3），两者逐位一致 |
| P6 | 退化几何 | L0-4 不崩、不越界、结果正确 |
| **P7** | **同一二进制服务多个 packed 切分（边界盲）** | 固定 `H=8, D_qk=192, scale, dtype`，分别用 `q_lens=[256,384]` 与 `q_lens=[100,200,340]`（`T_q` 同为 640）以及 `q_lens=[640]`（单请求）构造**同一个** `dq_acc` 张量，三次调用的输出必须**逐位相同**且与 golden 一致——这直接证明 `T_q` 只以数值进入、请求切分对本 kernel 不可见（§11 P2/P4/P7） |

**packed manifest 的精度预期**：与 dense 相同（fp16 / bf16 全 `bit_diff=0`）；若出现容差内非逐位，按
「`vsts` 分布 → 掩码宽度 → 尾块 extent」的顺序排查（dense Stage 3 的唯一根因就在这三处）。

## 9. 风险点与注意事项

### 9.1 已知约束与风险

| # | 风险 | 影响 | 证据 / 现状 | 处置（回退方案） |
|---|---|---|---|---|
| R1 | **packed 路径的运行期绑定未在本任务真机验证**（加速卡被另一算子 Stage 3 占用） | 若 packed 的 `dq_acc` 首维在 launch 时无法绑定 `total_q` → launch 失败；若 packed 与 dense 有隐藏差异 → 结果错 | 编译级已证（本目录双 dtype COMPILE OK，`total_q` 为形参）；**同一实现的 dense 已真机 31/31** | L0-1（最先跑）+ P7；失败则先按 dense 的已验证路径逐项对齐（`dq_acc` 连续性、`out_idx` 签名），再回报 conductor |
| R2 | **`vcvt` 的 NaN payload 不保留**（任意 NaN → `0x7FFF`） | 逐位比对时含 NaN 用例假失败 | dense 真机实证（fp16 dev `0x7fff` vs torch `0x7e00`；bf16 `0x7fff` vs `0xffff`） | 比对时「NaN 只比位置」+ `equal_nan=True`（dense `_compare` 已如此） |
| R3 | 设计期 roofline 的有效带宽取自**兄弟算子的实测带**（0.75–1.05 TB/s），未进知识库 | 估算可能与本算子实测偏差 | `cost_measurements` 为空；`.agents/platforms/toolchains/ascend/knowledge/bottlenecks.md` 明确无本型号真机瓶颈数据 | Stage 4 采集 baseline 后按 `hardware-cost-model.md` §3 回填「设计估算 vs 实测」偏差行；> 2× 时注明失准项 |
| R4 | **尾块读未初始化 UB** | 未文档化：若硬件对未初始化 UB 的读有异常行为 | 仅最后 1 个 tile 的 `R−n` 行；结果从不搬出；dense 真机通过 | 若 L0-3 出现异常，改为 `for i in T.serial(R): if i < n:` 的运行期守卫，或 `T.clear(acc)`（一次性，代价 48 KB 填充） |
| R5 | smoke 几何**负载不均**（`ntiles = 80`，`cpt = 2` → 40 核各 2 tile、16 核空转，+40%） | 该几何时延由 launch 与最长核决定 | §5.5 表；估算 5.6–7.9 µs，已接近 launch 地板 | 不改（smoke 非性能目标）；Stage 4 若要，可按 `heads` 分派更小的 `G` |
| R6 | 工具链缺陷：**python 层展开的 SIMD 语句被静默丢弃**（dense e11） | 若实现期「为了整洁」把段循环抽成辅助函数 → 编译通过但输出错误 | 探针 e11 vs e15 对照（`history_version/probe_e11_gen.txt` 无 `vlds_norm`，`probe_e15_gen.txt` 有） | §3.3 四条实现约束 + §9.2 E1/E8；L0-8 的 `_simd_body_gate` 在第一次编译就拦住 |
| R7 | **`vsts` 分布写错**（用 `NORM_B16`/`PK_B16`/16 位掩码） | 只写一半列，编译通过且在写到的槽上数值正确 | dense 真机 repro：256/256 vs 128/256 vs 4/256 | 固化 `PK_B32` + `pset(32, "PAT_ALL")`；L0-8 断言 `vsts_pack_b32`；L0-6 的 `torch.equal` 会立刻暴露 |
| R8 | `dq_acc` 的**数值量级不可控**（上游累加） | 大 `scale` 或大累加值会触发溢出/次正规 | golden 与内核在两侧都按 IEEE 处理（A3 已裁决非饱和） | 不需要特判；L0-6 覆盖 |
| R9 | `scale` 作为缓存键是 python float（`sm_scale = 0.05` 与 `192**-0.5` 是不同键） | 多一份特化（多一次编译），不影响正确性 | packed 测试含 `sm_scale = 0.05` 用例；源码 `@target_cache` 同机制 | 接受（与源一致）；Stage 4 若在意编译时间，可把 `scale` 量化到固定小数位（需评估语义影响，本设计不做） |

### 9.2 常见错误

| 编号 | 错误写法 | 后果 | 正确写法 |
|---|---|---|---|
| E1 | 把段循环/`vsts` 抽到模块级 python 辅助函数里、用 `for s in range(3)` 展开 | **编译通过但整个 SIMD 体被丢弃**（设备码只剩 mask 创建），输出未定义 | `for s in T.serial(NSEG)` 与 `vld/vmuls/vcvt/vsts` 语句全部**词法上写在 prim_func 体内** |
| E2 | `T.Tensor((T.dynamic("total_q"), H, DQ), …)` 在两个注解里各调一次 `T.dynamic` | `out_idx` 推理失败：`Cannot infer symbolic variable total_q` | 先 `total = T.dynamic("total_q")` 再在两个注解里复用 |
| E3 | 想靠 `T.copy(acc_f32_ub, dq_f16_gm)` 完成 cast | 语义检查直接报错：`DMA copies cannot perform type casting` | cast 放 `T.SimdVF` 内 `vcvt` |
| E4 | 只 clamp load、不 clamp store（或反之） | 尾 tile 越界写 GM / 漏行 | 两条 `T.copy` 与两侧切片**同时**用 `n` |
| E5 | 在 `T.SimdVF` 里用 `T.Parallel`，或在 `T.SimtVF` 里用 `T.simd.*` | 未验证的模式混用 | SIMD 模式只用 `T.simd.*`；换模式整段换（备选 A3） |
| E6 | 打开 `TL_ENABLE_FAST_MATH` | 可能改变次正规/舍入，破坏与 golden 的逐位一致 | 不开（§7.3） |
| E7 | 把 `R` 提到 96/128 让「tile 更大」 | UB 超限（221184 B 已逼近 253952 B，加膨胀必炸） | 保持 `R = 64`，先按 §4.5 复算 |
| E8 | `T.simd.vsts(..., dist="NORM_B16")` 或 `pset(16, …)` 配 `PK_B32` | 编译通过，但只写一半列（另一半保持旧值/未初始化） | `dist="PK_B32"` + `pset(32, "PAT_ALL")`，并用 `_simd_body_gate` + `torch.equal` 双重把关 |
| E9 | 试图在 packed 下引入「按请求对齐的 tile」或 per-request 循环 | 无收益，且让 `ntiles` 依赖 `cu_seqlens`（本 launch 拿不到），破坏单一动态符号 | 保持「行轴一维 tile 流」，请求边界留在 `_bwd_kernel` |

### 9.3 特殊场景处理

| 场景 | 处理 |
|---|---|
| `D_qk % 64 ≠ 0`（如 96/144） | `DW = roundup(D_qk, 64)`、`NSEG = DW/64`；`vsts` 仍按满 64 lane 写 `out`（pad 列落在 `out` 内、不会被搬出），两条 copy 的列切片用 `dim_qk`。packed manifest 无此几何，L0 不覆盖，标注为**未验证假设** |
| `H` 非 2 的幂 | tile 解码用普通 `%` / `//`（每 tile 一次），不影响正确性 |
| `T_q < R` | `n = T_q`，只搬 `T_q` 行；其余行冗余计算（L0-4） |
| `T_q` 极大（> 2^24 行） | GM 偏移在设备码里是 `int64_t`；`ntiles` 与 `cpt` 是 int32，`T_q ≤ 2^25` 时仍在 int32 内（packed manifest 最大 32768） |
| `dq_acc` 全 0 | 输出全 0（`0·s = 0`，`+0` 的符号由 `vmuls`/`vcvt` 决定，torch 亦为 `+0`）→ 建议 L0-4 顺带断言 |
| bf16 目标 | 与 fp16 同路径，仅 `dtype` 参数不同（`vcvt` 支持 `bfloat16`）；bf16 的容差更松，但本设计目标是逐位一致 |
| `scale = 0` 或 `scale < 0` | manifest 要求 `0 < sm_scale < inf`，故不会为 0；`vmuls` 对负数同样正确，符号随 `scale` 翻转（工厂按 §8.3 L0-9 拒绝 0 / 负值，与 dense 一致） |
| **多个请求共享同一 `T_q`** | 输出必须逐位相同（P7 断言）；这是 packed 与 dense 的**行为等价性**判据 |
| 单请求 `q_lens=[64]` | 与 dense 的 `T_q = 64` 完全相同（L0-4）；`ntiles = 4 < 56` 时 52 核空转 |

### 9.4 packed 专属结论（本任务新增的唯一实质章节）

本任务是 packed 分支，但 `_dq_land_kernel` **不接触任何 packed 特有信息**：

1. 参数表里没有 `cu_seqlens_q` / `cu_seqlens_kv`（源码第 94–95 行）；
2. 没有 per-request 循环、没有 `q_len` 读取、没有分段边界判断（§0.3 步骤表）；
3. 输出契约 `[T_q, H, D_qk]` 与 kernel 内部 shape 完全相同（不做 BSHD↔THD 的 view，§11 P3）。

因此 packed 分支的唯一「新」风险是**验证覆盖**（L0-1/3/4/7 与 P7），而不是实现差异。
若 Stage 3 在 packed 几何上发现问题，第一诊断顺序是：① `dq_acc` 是否连续（`torch.zeros(q.shape)` 保证）；
② `out_idx` 输出 shape 是否等于 `(T_q, H, D_qk)`；③ 尾块 extent；④ `vsts` 分布——其中 ③④ 与 dense 共用。

## 10. 交付清单

### 10.1 目录结构

```
examples/mla_varlen_bwd/_dq_land_kernel/
├── DESIGN.md                              # 本文（Stage 1）
├── RETROSPECTIVE.md                       # Stage 1 复盘（Phase 8）
├── verify_equiv.py                        # 等价性机器验证（§1.6.1，本目录已跑 [EQUIV_PASS]）
├── history_version/
│   ├── design_probe_packed.py             # 本任务探针：packed 形态双 dtype 编译 + 设备码断言
│   ├── probe_packed_gen_float16.txt       # 本任务设备码快照（41 行，含 vsts_pack_b32）
│   └── probe_packed_gen_bfloat16.txt      # 同上（bf16）
├── _dq_land_kernel.py                     # Stage 3：kernel + golden + 分层测试
└── perf_opt/_dq_land_kernel.py            # Stage 4：调优变体（含 §1.6.3 备选 A3 的 A/B）
```

### 10.2 文件清单

| 文件 | 阶段 | 内容 |
|---|---|---|
| `DESIGN.md` | Stage 1（本文件） | 迁移分析 + 复用分区 + 调研 + 优化 + API/内存/Tiling/循环/同步/验证/风险 |
| `verify_equiv.py` | Stage 1 | 3 采纳项 + 3 对照项的 CPU 等价性验证（已跑，`[EQUIV_PASS]`） |
| `history_version/design_probe_packed.py` | Stage 1 | 无卡编译探针（本任务新增，双 dtype） |
| `RETROSPECTIVE.md` | Stage 1 | Phase 8 复盘 |
| `_dq_land_kernel.py` | Stage 3 | kernel + golden + L0/L1/L2/boundary 分层测试（`python _dq_land_kernel.py --level L0`） |
| `perf_opt/_dq_land_kernel.py` | Stage 4 | 调优变体（含 §1.6.3 备选 A3 对照） |

### 10.3 命名规范

- kernel 工厂：`_dq_land_kernel(heads, dim_qk, scale, dtype)`（与源签名一致，Stage 5 集成靠它接线）。
- golden：`golden_dq_land(dq_acc, scale, dtype)`（§8.1）。
- 测试入口：`run_L0` / `run_L1` / `run_L2` / `run_boundary` + `main()`（`--level`），与平台 kernel 模板一致；
  packed 几何以 `q_lens` 列表表达。

### 10.4 实现顺序

1. 从 dense 复制 `_dq_land_kernel.py` 全套（golden + `_compare` + `case` 骨架 + `_simd_body_gate`），
   把几何参数换成 `q_lens`，先跑 L0-1（验证 packed 的 `out_idx` 动态输出，R1）。
2. 第 2 步（硬前置）：确认每次编译产物的设备码含 `vlds_norm`/`vmuls`/`vcvt` 且含
   `vsts_pack_b32`/`vsts_norm`（L0-8）——**不通过不得继续**（防计算体静默丢弃）。
3. L0-2（bf16）→ L0-6（`vcvt` 语义 + 非饱和，R2/payload）→ L0-3/L0-4/L0-5（尾块与退化）→
   L0-7（满载）→ P7（多切分同二进制）。
4. 单版本（去掉 `annotate_buffer_versions`）变体跑 P5 对照（§6.3：`num_stages=1` 不会改变结构）。
5. 全量 L0 通过后再跑 L1/L2/boundary（packed 代表性几何）与 L0-7 的 Stage 4 baseline。

## 11. 与 dense 已验证实现的一致性核对

本节逐项回答「packed 是否引入差异」，每项给出证据与结论；**九项全部为「无差异」**，因此本设计对实现的要求
是「照抄 dense 实现 + 换 packed 几何的用例」。

| # | 复核项 | 证据 | 结论 |
|---|---|---|---|
| P1 | **调用路径不 view**：dense `forward` 把 BSHD `view` 成 `(B·S, H, D)` 再进 `_run`，返回时 `view` 回 BSHD（源码第 370/379 行）；packed 直接把 THD 打包张量交给 `_run`（第 385–387 行） | 两者进入 `_run` 的张量都是 `(T, H, D_qk)` 行主序；`_run` 内部逐字相同（第 338–347 行）；`dq_acc = torch.zeros(q.shape, …, device=q.device)` 恒为连续新分配（第 342 行 dense / 第 223 行 packed） | **无差异**：`view` 不改变内存，且 kernel 只读 `dq_acc`（不读 `q`） |
| P2 | **`T_q` 的来源与量级**：dense `T_q = B·S_q`（manifest 最大 16384）；packed `T_q = Σ q_lens`（最大 32768） | 工厂签名不含 `T_q`（§0.0 证据 3）；`total = T.dynamic("total_q")` 在两个注解里复用后由 launch 时的张量首维绑定；本目录设备码 `total_q` 是 kernel 形参、`ntiles` 运行期算（`probe_packed_gen_float16.txt` 第 16/23 行） | **无差异**：只是同一符号取不同值；同一二进制服务所有长度（§8.3 P7 的实测计划） |
| P3 | **输出契约 `[T_q, H, D_qk]` 无 batch/seq 维** | manifest `dq: {shape: "[T_q, H, D_qk]"}`；kernel 的 `out_idx=[1]` 输出为 `(total, heads, dim_qk)`；packed wrapper 直接返回（第 227–228 行），dense 侧多一次 `view`（第 379 行） | **无差异**（对本 kernel）：shape 与 buffer 相同；差别只在调用方是否 reshape |
| P4 | **`cu_seqlens_*` 是否进入本 launch** | `_dq_land_main` 只有两个张量入参（源码第 94–95 行）；packed `_run` 调 `land(dq_acc)` 只传一个张量（第 227 行）；`cu_seqlens_q`/`cu_seqlens_kv` 只出现在 `main(...)` 的实参表（第 226 行） | **无差异**：本 launch 对请求边界**完全盲**；边界语义全部由 `_bwd_kernel` 消化 |
| P5 | **`scale` 特化** | 两条路径都用 `self.scale = dim_qk**-0.5 if sm_scale is None else sm_scale`（源码第 320 行 / packed wrapper 第 168 行）；工厂键含 `scale`（`@target_cache` / `lru_cache` 同机制） | **无差异**：packed 测试的 `sm_scale = 0.05` 只多一份特化，不改语义 |
| P6 | **dtype 与 head-dim 契约** | 两条路径共享 `_MLABwdKernelBase.refusal`（源码第 273–281 行：`ATTENTION_DTYPES`、`dim_qk % 16 == 0`）；packed wrapper 内联同一检查（第 157–162 行） | **无差异**：同 dtype 集合、同整除约束 |
| P7 | **分核（逻辑 tile 数）** | 逻辑核数 = `ceildiv(Σ q_lens, 64)·H`；packed manifest 80 / 65536 / 64512 / 32768，全部 `> 56`；`cpt` 失衡 ≤ 0.15%（smoke 除外，见 §5.5） | **结构无差异**（同一 persistent + `cpt` 区间方案）；数值按 packed workload 重算，结论仍为「极大规模 → 核内串行」 |
| P8 | **退化几何** | 测试 `q_lens=[64]`（H=4）→ `ntiles = 4 < 56`；dense 的 L0-4（`T_q=1, H=1` → `ntiles=1`）与 L0-5（`T_q=65, H=2` → `ntiles=4`）同样 < 56 | **无差异**：空区间使 `T.Pipelined` 零迭代，其余核立即退出，无需分支 |
| P9 | **数值路径** | §1.6.1 机器验证（本目录独立重跑，A1/A2/A3 全 `EQUIV_PASS`）；dense 真机 31/31 `bit_diff=0`；`PK_B32`/`sat=False` 的真机语义见 U4/U5 | **无差异**：同一式、同一指令链、同一舍入形态 |

**总判定**：`_dq_land_kernel` 的 packed 与 dense 实现**应当逐字相同**（事实上源就是同一个函数），
本设计的差异**只出现在验证覆盖**（packed 几何的 L0 用例与 P7）与**文档表述**（packed 契约、`T_q` 来源）上。
这是本任务「几乎全量复用」要求的直接结论，不是省略分析：**九项复核各自给出了独立证据，且没有一项被发现存在差异**。
