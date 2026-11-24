# _make_scatter_kernel 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息

- 文档: examples/moe_permute_align/_make_scatter_kernel/DESIGN.md（1089 行）
- 任务类型: 迁移（GPU TileLang → ascend，`MoEPermuteAlignFwdOp` 第二段 kernel）
- 源算子: /workspace/TileOPs/src/tileops/_kernels/moe/permute_align.py（`_make_scatter_kernel` 行 162–218，全文 472 行已读）
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 3（均落在非阻塞口径，逐条见维度 6「建议级」、维度 7「建议级勘误」、维度 8「建议级偏差」三处的内联标注）

## 机械复核（D-1/D-5）

- **verify_equiv 重跑**: **PASS**（`cd examples/moe_permute_align/_make_scatter_kernel && /workspace/venv-ascend950/bin/python verify_equiv.py` → `RESULT: EQUIV_PASS`）。逐项输出与 §1.6.1 内嵌表**完全一致**：O1 `0/24`（8 shape × 3 到达顺序，脚本自报 `route-order replays checked in O1: 24`）、O2 `0/6`（含尾块 `numel ∈ {1,8,255,1032,4096,32768}`）、O3 `0/4` 且最大容量值 `102131`（与 §1.6.1 O3、§4.1 最大 `P` 一致）、O4 `0/5`。脚本内容核对：`baseline_two_level()` 逐块 SMEM rank + 批量预留（= 源算法）、`candidate_single_level()` 支持 `None/reversed/shuffled` 三种合法到达顺序、`contract_violations()` 覆盖「每 expert 多重集 / 槽互异 / 槽区间 / 区段外哨兵 / 游标终值 = 起点 + count_e」，即验证对象与 §1.6.1 的论证口径一致，非空壳脚本。
- **design_calc_check**（`python3 .agents/tools/design_calc_check.py --design … --task-dir …`）JSON `checks` 摘要：
  - `ub_budget`: **skip**（"no per-level buffer tables parsed from the memory plan"）→ **人工已核**：主选路径不分配任何 `alloc_shared/alloc_l1/alloc_l0*`（§4.3/§4.5），UB 占用 0 B ≤ 253952 B；备选 B `3·E + 2·tile` int32 = `3·1100·4 + 2·640·4 = 18320 B` ≤ 253952 B（§1.6.3，复算一致）；L1/L0A/L0B/L0C 均不使用。
  - `l0c_budget`: **skip**（"no Ascend Cube GEMM call found"）→ **人工已核**：无 `T.gemm`、无 L0 分配（§5.1），L0C 约束不适用。
  - `core_split`: **skip**（"insufficient parsed dims {}"）→ **人工已核**（逐 workload 复算，见维度 3）：12 条 workload 的 `nblocks` 取值为 `{1, 2, 32, 52}`（t1=1、t32=2、t512=32、t4096=52，三族相同）；`numel=32768` 复算 `ceil(32768/56)=586 → ept=ceil(586/128)=5 → tile=640 → nblocks=52`；且 `tile = 128·ept ≥ ceil(numel/56)` ⇒ `nblocks = ceil(numel/tile) ≤ 56` 对任意 `numel` 成立，算术与 §5.5 自洽。
  - `r3_metrics`: **pass**（"all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak)"）。
- 补充独立复算（不计入机械工具，供人工口径核对）：GM 流量 `2·32768·4 + 4·385 = 263684 B ≈ 264 KB`（§4.4 一致）；`P` 全 12 条 workload（24263 / 24511 / 28351 / 57023 / 16199 / 16447 / 20287 / 48959 / 8135 / 8383 / 12223 / 40895）与最大 `102131 = 32768 + 1101·63` 全部复算一致；§1.6.0 估算 `630 路由 × 4 cyc = 2520`、`85 × 30 = 2550`、`5.1 Kcyc / 1.6 GHz ≈ 3.2 µs` 复算一致。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

检视方式：亲自 Read 源算子全文（472 行），逐项与 §0.1–§0.7 核对，并对 align→scatter 边界做专项核对。

| 检查项 | 结论 | 源码核对证据（位置 + 语句）↔ DESIGN.md |
|---|---|---|
| 语义理解正确性 | pass | 源 `_make_scatter_kernel` 行 162–218：`cumsum` 作每 expert 补齐起点游标（align 行 126–129 `cumsum[expert] = excl`），行 201–214 `rank=atomic_add(s_count[eid])` + `s_base=atomic_add(cumsum[e], s_count[e])` + `sorted[s_base[eid]+rank]=idx` ↔ §0.1 公式 `sorted[cumsum[flat[i]]+r_i]=i`、§0.1 边界（行 169–170 `numel<1` raise；行 198 `if_then_else(idx<numel, flat[min(idx,numel-1)], -1)`）、§0.1 dtype（全 int32，无 cast）、§0.1「只 guard `eid>=0`」（行 201/213）**逐条一致**。 |
| expert 内顺序未定义的判定（契约级关键项） | pass（本检视重点 1 复核通过） | 源码用 `atomic_add` 的到达顺序写槽（行 202/210/214），块间写序由 `cumsum` 原子顺序决定 ⇒ 顺序不可复现。契约侧证据：`examples/TileOPs/tests/ops/test_moe_permute_align.py` 文件头 docstring（行 22–27）明写 "the route order *within* an expert's range is unspecified"，`_assert_route_equivalent`（行 47–96）用 `_routed_slots` 归并 (expert, route index) 后比对；workload 参考 `workloads/moe.py::ref_permute_align`（行 44–65）给的是**索引升序这一种合法排布**。更强佐证（设计未引）：同文件 `_make_small_batch_kernel`（行 221–336，写槽行 325–332）按 worker 交错顺序 `idx = wid + i·WARP_LANES` 落槽，与 align+scatter 路径的指数升序**本就不同** ⇒ 两条 GPU 路径已互为反例，intra-expert 顺序确非契约。**故「两级计数→单级 bump」的契约级等价论证前提成立**。 |
| 算法解读完整性 | pass | 源全部计算语句均被 §0.3 归位：行 169–174（步骤 1）、194–197（2）、186+198（3）、201–202（4）、205–210（5）、213–214（6）、31/176（7）。无遗漏步骤、无臆造步骤；数据流图覆盖 `flat/s_count/s_base/cumsum/sorted_token_ids` 全部 buffer；host 侧（行 443–460 `forward`：`flatten().contiguous()`、`cumsum=torch.empty(E+1)`、align 后同流 `scatter(flat, sorted, cumsum)`）已列出（§0.1 末段 / §0.3 末段）。行 199/203/211 的三次 `T.sync_threads()` 未单列为表格步骤，但在 §0.4 #4、§0.6 R3、§7 中被显式讨论，属覆盖充分。 |
| 优化手段覆盖完整性 | pass | §0.4 八项逐一可在源码定位：#1 两级计数（行 189–210）、#2 一 lane 一路由（行 172/184–186）、#3 尾夹取（行 198）、#4 `s_count/s_base` 分离（行 187–190 注释原文 "updating s_count in place makes TileLang put a sync inside the expert guard, which deadlocks past one round"）、#5 `expert_rounds`（行 174/194–197）、#6 1-D grid（行 173/184）、#7 `-O3`+`target_cache`（行 31/176）、#8 两段式（行 164–167 docstring）。§0.4 末段的「无 `T.copy`/无 `T.gemm`/无 warp shuffle」经全文核对成立（`tvm_warp_shuffle_up` 仅出现在 align 行 100–108）。 |
| 耦合性判定合理性 | pass | §0.5 12 行均有四态处置与依据；与平台能力映射不矛盾：`T.Kernel` 1-D 无 `threads`（`ascend/language/kernel.py:139–149` docstring 原文）、`T.SimtVF(threads=…)`（`frame.py:107`）、跨 block barrier 未认证（`references/programming.md` §同步与缓冲版本「跨 block batch barrier 仍未认证」）、`asc_atomic_add(__gm__ int32_t*, int32_t)`（`/usr/local/Ascend/cann-9.2.1/asc/include/simt_api/device_atomic_functions.h:31`，我复核该行号精确命中 `__gm__ int32_t` 重载）。舍弃项（#5/#6）均写明理由，`重新设计 0` 项的判定有依据。仅「#4 两级计数不减少全局原子」这一量化论据有算术偏差，记入维度 8（建议级，不影响本维度 pass）。 |
| NPU 重设计可行性 | pass | §0.6 R1/R2/R3 三项均有「源方案 → NPU 新算法 → 语义保持论证」三要素：R1 双射论证 `(bid,i,tx) ↦ idx` 覆盖 `[0, nblocks·128·EPT)` 且每路由恰一次（复算成立）；R2 游标单调自增 ⇒ 每 expert 取槽恰覆盖 `[S_e, S_e+count_e)` 且互异（独立推演成立）；R3 删除 `s_count/s_base`/expert 循环不改变落槽集合（因该结构只承载两级计数的中间态）。新算法满足平台约束：1-D 核网格、无跨核同步、静态边界、UB 0 B。 |
| 设计与决策一致性 | pass | §1.4 明列 6 条差异并标注来源（R1/R2/R3/O2/O4），§3.1 以 §1.6.1 优化后公式为输入，§3.3 伪代码与 §6.2 同型，§4.3「无中间缓冲」与 §5.1「纯 Vector」一致；未发现「§0 判重设计、§3 照抄源方案」的脱节（对照：§3.5.2 逐项列出计数结构、E 维度、线程模型的替换）。 |
| golden 独立性 | pass | §8.1 golden 移植 `ref_permute_align`（`workloads/moe.py` 行 44–65，纯 host `slot[eid] += 1`），不含原子、不含 1-D 核网格、不含 `SimtVF`、不含单级 bump；以 §0.1 语义为唯一依据，未复刻 §0.6 的 NPU 数据流。比对口径为契约级（每 expert 多重集 + 槽互异 + 哨兵 + `cumsum` 推进），与 `test_moe_permute_align.py::_assert_route_equivalent` 同口径。 |
| align→scatter 边界描述（本检视专项） | pass | §0.1 末段分工与源码逐条吻合：① align 填哨兵 `sorted_token_ids[idx] = numel`（align 行 67–71，bx==1 分支）、② `cumsum[expert] = excl` 与 `cumsum[num_experts] = s_carry[0]`（行 126–139）、③ `num_tokens_post_pad[0]`（行 136–138）、④ `expert_ids`（行 142–155）；scatter 只消费 `flat` + `cumsum`（行 179–183 签名），只推进 `cumsum[e< E]`、不读写 `cumsum[E]`（行 201–214 仅经 `flat` 取值），`cumsum` 由 `forward` 分配并同流顺序 launch（行 456–460）。「`cumsum` 必须等全局直方图落定 ⇒ 两段分工」与 align 的 `T.Kernel(2, …)` + 后置 launch 结构一致。 |

### 1. API 可行性 — pass

- §3.2 的每条 API 均落到本机选中模块（`/workspace/venv-ascend950/.../tilelang/ascend/language/`，0.1.15），无跨后端借用：
  - `T.Kernel(*blocks)`：`kernel.py:139–149` docstring 原文「a 1-D grid of AI cores … this dialect has no `threads` parameter」——与 §2.3/§5.1 一致；
  - `T.SimtVF(threads=128, latency=0)`：`frame.py:107` 签名逐字匹配；
  - `T.atomic_add(dst, value, memory_order=None, return_prev=False, annotations=None)`：`tilelang/language/atomic.py:211–217` 签名逐字匹配（经 `ascend/language/__init__.py` 的 `from tilelang.language.common import *` 进入 `T.` 命名空间）；
  - **ascend 降低链路源码级佐证（强于设计自引的探针）**：`tilelang/src/ascend/codegen/codegen_ascend.cc:2502–2539` 明确把 `atomic_{add,max,min}_elem_op` 与 `*_ret_elem_op` 降为 `asc_atomic_add(ptr, value)`，且注释写明「Ascend SIMT atomic APIs only accept (ptr, value)」、`is_ret` 分支直接产出返回旧值的表达式；设备端声明 `device_atomic_functions.h:31` 提供 `int32_t asc_atomic_add(__gm__ int32_t*, int32_t)`（`__NPU_ARCH__==3510`，第 18 行）。即「int32 GM 原子 + 取旧值」在**编译层**有代码生成实现，与探针 K2/K6/K9 的降级码一致。
  - `T.if_then_else` / `T.min`：common 层表达式，探针 K6/K9 降级码可见（`design_probe_skeleton.py:85–104`、`design_probe_primitives.py:165–205`）。
- **未设备验证项已如实登记**：`asc_atomic_add` 运行时行为/争用、逐 lane 数据相关 GM store 运行时行为、原子到达顺序导致的 intra-expert 不确定，均列入 §3.4「未设备验证项汇总」＋§9.1 R1/R2/R3，并给出兜底顺序（备选 B → §11.3 确定性路径）；符合平台「接口签名存在不代表已执行验证」的口径（`programming.md` 首段）。
- 探测脚本真实存在且内容与自述一致（`history_version/design_probe_primitives.py` 的 `k1_scatter_store/k2_gm_atomic/k3_ub_atomic/k4_cumsum/k5_store_mode_atomic/k9_factory_form`；`design_probe_skeleton.py` 的 `k1b/k2b/k6/k7/k8`）。§11.1 对 K5「`T.Task supports exactly one Ascend hardware pipe … mask=80`」与 K4 `tl.cumsum` 无 ascend 实现的负向结论，是用于**规避**而非用于放行，方向正确。
- 微小保留（不影响结论）：`T.simd.vscatter/vgather2` 只作用于 UB 的判定，设计引 docstring（`simd.py:1166–1176`「Scatter-store: base[index[lane]] = src[lane]」，`simd.py:1141–1156`「Gather elements from base using per-lane offsets」）＋平台口径（`programming.md`「`T.simd.*` 提供寄存器级操作」）＋同类先例（`paged_cache_write` 结论「SIMD 只能操作 UB」），属符号级证据而非签名级 scope 限制；但该负向断言同时被 §1.6.3 候选 #3/#5 的容量否决独立兜住（UB 窗口需 ≥ `P·4 B` = 228 KB），结论不受影响。

### 2. 内存层级规划 — pass

- §4.4 搬运路径完整：`GM[flat] → 寄存器 → GM[sorted]`（逐 lane 标量 load/store），`GM[cumsum]` 为原子 RMW 热区（L2）；无 UB/L1/L0 参与，与 §4.3「无中间缓冲」、§5.1「纯 Vector」、§6.3「无可多版本化缓冲」三方一致。
- UB 预算 0 B ≤ 253952 B（平台 `design_budget.capacity_bytes.UB`，我经 `statectl platform --task-dir …` 复核同值）；备选 B 18320 B ≤ 253952 B（复算一致）。L0C 不适用（无 `T.gemm`）。
- 每个 buffer 的层级标注齐全（§4.1/§4.2 标 GM，§4.3 明确 UB 0 B），`cumsum` 原地读写语义在 §4.2/§5.4 双处声明。
- 机械工具 `ub_budget`/`l0c_budget` 为 skip，已按「机械跳过，人工已核」处理（见机械复核段）。

### 3. Tiling 策略 — pass

- **Block 划分明确**：`THREADS=128`、`ept = max(1, ceil(ceil(numel/56)/128))`、`tile = 128·ept`、`nblocks = ceil(numel/tile)`（§5.2），并给出代表/中等/小规模三档实例值。
- **对齐约束**：int32 标量访问，无向量宽度要求；`tile` 取 `THREADS` 整数倍使 `idx` 计算退化为常量乘加（§5.3）。
- **非整除处理**：纯 Vector 但明确给出尾 lane 策略（`eid = -1` 跳过 + 夹取索引不越界），并量化尾部空闲（`numel=32768` 为 512/33280 = 1.5%，复算一致）。
- **分核三要素齐全**（对照 `.agents/skills/_shared/standards/core-split-strategy.md` §1）：① 逻辑核数 = `nblocks`，逐 workload 给出 1/2/32/52；② 物理核数有实查记录（`statectl platform --task-dir …` 返回 `observed_cores = {"cube":28,"vector":56}`，我复核返回值一致；未用文档假设值，并说明卡被占用故未跑 `torch.npu.get_device_properties`）；③ 规模判定三档均给「无需适配」依据，且给出极大规模「固定 56 核 + 核内 `ept` 串行」的降级方案。
- **核数与取值自洽（独立复算）**：`numel=8 → 1`；`256 → ceil(256/56)=5, ept=1, tile=128, nblocks=2`；`4096 → 74, ept=1, nblocks=32`；`32768 → 586, ept=5, tile=640, nblocks=52`。**`nblocks ≤ 56` 的推导成立**：`tile = 128·max(1, ceil(ceil(numel/56)/128)) ≥ ceil(numel/56)` ⇒ `nblocks = ceil(numel/tile) ≤ ceil(numel/ceil(numel/56)) ≤ 56`（对 `numel=1` 及任意 `numel` 均成立）。
- 核内串行上限 `ept ≤ 5`（本任务）为编译期常量，不依赖动态 shape；`T.serial` + `T.Parallel` 的边界全部静态可证。

### 4. 技术约束检测 — pass

- 三维 Kernel：不涉及（1-D 核网格，`T.Kernel(nblocks)`）。
- GEMM 非整除 / L0C 溢出：不涉及（无 `T.gemm`、无 L0 分配）。
- 平台约束逐条对照（§3.5.1 表）经核实无遗漏：`T.Kernel` 无 `threads`（`kernel.py` docstring）、逐元素表达式须在 `T.SimtVF` 内（`workflow.md` 第 23 行）、`tl.cumsum` 无 ascend 实现（探针 K4/K8 报错原文，设计规避）、store-mode 原子不能与向量语句同 `T.Task`（探针 K5）、UB 容量、无跨核 barrier（`programming.md`）。
- 谓词内原子 + `else` 分支 `slot=0` 的陷阱已识别并写成实现约束（§9.2 第 1 条，探针 K6 降级码可见 `else { condval = 0; }`），把原子与写放入同一谓词，避免非法 lane 写坏槽 0——这是本算子最容易踩的编译/运行时坑，处理到位。

### 5. 循环与同步 — pass

- §6.1 逐维给出循环类型与理由：`T.Kernel(nblocks)`（block 级）、`T.SimtVF(threads=128)`（线程域）、`T.serial(ept)`（每线程路由数，编译期常量，`#pragma unroll`）、`T.Parallel(128)`（lane 并行），并显式声明不使用 `T.Pipelined`/`T.Persistent` 及理由（单遍、无 tile 复用、逻辑块数 ≤ 物理核数）。
- §7 同步策略与编程模式匹配：无块内 `sync_threads`、无跨核 flag，唯一并发点是 `cumsum` 的硬件原子 RMW；段间靠同流 kernel 边界。与 §4.3（无 UB 缓冲）自洽——无缓冲即无可被 `InsertSync` 插入依赖的对象。
- §6.4 尾块处理四类来源齐全（路由维 / expert 维 / `numel=0` / 契约外 id / 槽越界），其中路由维与槽越界给出具体机制（夹取、`eid=-1`、编译器值域 guard）。
- 自动调度（`allow_autoschedule` 默认开启）的影响已评估（§6.3/§7.3）：本 kernel 无 alloca，`MaterializeMultiBuffer` 无作用对象。

### 6. 验证方案 — pass（含 1 处建议级用例描述勘误）

- §8.1 给出完整 PyTorch golden 草案，且**构造了 align 段前置状态**（`cumsum_in` 补齐起点 + 哨兵 `sorted_token_ids`），符合本段作为「第二段」的验证前提；`assert_route_equivalent` 的断言集合（每 expert 多重集、槽互异、哨兵保持、`cumsum` 精确推进）与 §0.1 语义一一对应，并显式否定四类错误（§8.2）。
- 精度标准：全 int32 索引数组 ⇒ 精确判定，并说明「不是逐位相等」的理由与更强自检的可行形式（排序归一化后逐元素相等，对任意原子顺序成立）——口径正确、可执行。
- §8.3 L0/L1/boundary 用例覆盖规模、`E>THREADS(1100)`、`bs=1`、尾块、`E=1` 极端热点、skew、`numel=0`，并强调每用例重建 `cumsum`（对应 §5.4 的原地语义约束）。
- 建议级（不影响结论）：§8.3 中 `1032 / 128 / 64` 一行的「尾 lane 512 个」与分核规则不符——`ept=1, tile=128, nblocks=ceil(1032/128)=9`，尾 lane = `9·128 − 1032 = 120`（512 是 `numel=32768, tile=640` 的尾 lane 数）。该数字仅作用例说明，不改变用例的验证价值（仍覆盖 `numel % tile ≠ 0`），Stage 3 落地时按 120 描述即可。

### 7. 完整性与一致性 — warn（3 处引用/计数勘误，均不影响结论）

- 章节齐全：§0（0.1–0.7 全备）+ §1–§11，§1.6 含 1.6.0/1.6.1/1.6.2/1.6.3 四个子节；全文检索未命中占位符模式（公式中的花括号为 LaTeX 记法，非模板占位）。
- 同类引用真实：`examples/paged_cache_write/_paged_cache_write_kernel/DESIGN.md`、`.agents/platforms/toolchains/ascend/kernel_template.py`、`examples/TileOPs/tileops/kernels/moe/moe_permute_align/_moe_permute_align_fwd_kernels.py`、`examples/TileOPs/tests/ops/test_moe_permute_align.py`、`examples/TileOPs/tileops/workloads/moe.py`、`examples/elementwise/vec_add_2d_dynamic_shape.py`、`examples/TileOPs/tileops/manifest/moe.yaml` 全部存在（逐个 `ls` 核对）。「无同语义兼容示例」的声明成立（全库仅 `paged_cache_write` 为 ascend 后端用例）。
- 内部一致：API 映射（§3.2）/内存规划（§4）/Tiling（§5）/循环（§6）四方无矛盾；§4.1 的 12 行 `ept/tile/nblocks` 与 §5.2 规则复算逐行一致；`P` 12 行复算一致。
- **建议级勘误（不影响结论，供 Stage 3 引用时修正）**：
  1. §0.5 末段统计「保留 6 / 等价替换 4 / 舍弃 2 / 重新设计 0」与表体不符：逐行计数为 保留 6（#1/#2/#7/#8/#10/#11）、等价替换 3（#3/#4/#9）、舍弃 2（#5/#6）、不适用 1（#12），合计 12。
  2. §0.7 把源 `_make_scatter_kernel` 标注为「行 168–229」，实际为行 162–218（行 219 起为 `_make_small_batch_kernel` 的装饰器），引用区间越出函数边界。
  3. §4.1 末段「测试集另有 `numel ∈ {1,8,256,512,600,1032,2048,8192,16384}`」与 `test_moe_permute_align.py` 的参数化列表不符：实测集合含 `{0,1,2,8,32,200,512,600,1023,1024,1032,2048,8192}`，不含 256/16384。结论性断言「分核规则一律给出 `nblocks ≤ 56`」仍然成立（由维度 3 的代数推导保证，与具体集合无关）。

### 8. 算法优化分析 — warn（阻塞级要件全部成立：四问齐全、等价性独立推演 + 机器验证一致、向量化覆盖完整、裁决三件套齐备；1 处量化论据偏差记为建议级）

> 维度判定说明：本维度的 warn 对应「非阻塞建议」（技能规则：warn 不阻塞结论），非 fail。阻塞级检查项——调研四问完整性、负向断言复核、等价性独立推演与机器验证一致性、循环/标量点向量化覆盖、未实测常数的裁决三件套——逐项为 pass；唯一偏差是收益量化中的一处代数取值错误，不改变算法落地的正确性。

**§1.6 章节存在性**：pass。§1.6.0（调研四问）/§1.6.1（数学等价优化 O1–O4）/§1.6.2（向量化替代分析，8 个点 L1/L2/S1–S6）/§1.6.3（向量化轴与布局决策，5 候选矩阵 + 备选 B + 裁决计划）四节齐备。

**算法调研存在性与完整性**：pass。R1 候选表 6 行且**基线 #0（源算法两级计数）在表**；R2「有，且选定」并给出结构依据（唯一跨元素状态是 O(E) 运行游标 ⇒ 天然存在单遍流式变体，充要条件是契约允许 intra-expert 任意顺序）；R3 四口径表覆盖「整型/原子操作数（flops=0 的对应物）/GM Bytes/扫描遍数/中间缓冲峰值」，基线在表、口径写明「Bytes 只计 GM 流量，不含 L2 命中重读」；R4 逐候选六项评估（单元/容量/对齐整除/静态边界/流水/跨核）；调研范围声明逐条可追溯（§11.4）。

**调研结论正确性（独立复核）**：
- 等价性：pass——单级 bump 与两级计数的契约级等价**独立推演成立**：两级形式中块 `b` 的 `base_b = prev(atomic_add(cumsum[e], s_count_b[e]))` 给该块分配长度 `s_count_b[e]` 的连续互异区间，各块区间按预留顺序首尾相接，故并集 = `[S_e, S_e+count_e)`；单级形式下 `count_e` 次自增的旧值序列恰为该区间的全部整数（`prev` 单调 +1 ⇒ 双射）。两式的差异只在 intra-expert 排列，而契约允许任意排列（维度 0 已用测试 + 源 small_batch 路径双重确认）。机器验证 24 次重放一致（见机械复核）。
- 复杂度算术复算：pass（四口径）+ **1 处建议级偏差（见下）**：`numel=32768` 的 GM 流量 `264 KB`、扫描遍数 1、缓冲峰值 0 B、候选 #2 的 `56 × 131 KB = 7.3 MB` 与 `E·numel/64 = 196608` 比较组、候选 #3 的 `262144 B > 253952 B` 均复算一致，**未发现中间缓冲 GM 往返漏计**（候选 #2 已把 `nblocks × numel × 4B` 读放大计入）。
- 负向断言复核：pass——「算法候选库无本族条目」经我直接 Read `.agents/platforms/toolchains/ascend/knowledge/algorithm-candidates.md` 确认（仅 elementwise/reduction/softmax/norm/GEMM/attention 六族）；「`kb_search` 无命中」经我重跑同名查询确认（`no entries matched`）；「无 950 真机瓶颈/代价常数」经 `knowledge/bottlenecks.md` 与 `statectl platform` 的 `cost_measurements: []` 双重确认；「存在在线变体」与源码结构（源算法即两级两趟）一致，未出现「源已用 online 却断言无 online」的矛盾；经典候选（计数排序/基数排序/前缀压缩/单块串行/host 预计算）均已枚举并逐一给出淘汰依据。
- 并列候选裁决路径：pass——单级 vs 两级被明确识别为依赖未实测常数（GM 原子 vs UB 原子相对代价），因此按规则给出**三件套**：主选（§1.6.3 候选 #1 + O1）、结构完整的备选 B（1-D 网格 + 3 个 UB 缓冲 + 2 次 `syncthreads` + 批量预留，含 UB 预算复算 18320 B、`E>tile` 的外层 expert 轮限制、分核三要素同主选）、裁决计划（4 个代表 shape、`msprof` Task Duration 指标、**阈值「备选 < 主选 × 0.9」**、回写路径含 `perf_feedback.md`/`[DESIGN_LIMIT]`、按 shape 分派可行性），符合 core-split-strategy §2.4「A/B 实测为必做项」。
- **建议级偏差（唯一实质性问题，不阻塞）**：§0.5 #4 / §0.6 R2 / §1.6.0 R4 行 0 / §1.6.1 O1 的「收益量化」中「每块非空 expert 数上界 `min(tile,E) = tile` ⇒ 全局原子数上界 `nblocks·min(tile,E) ≈ numel`，两级不减少全局原子」在本任务的参数下**不成立**：`ept=5 ⇒ tile=640 > E ∈ {128,256,384}`，故 `min(tile,E) = E`，两级形式的全局原子数上界应为 `nblocks·E` = 52×384 = **19968**（E=384）/ 52×256 = **13312**（E=256）/ 52×128 = **6656**（E=128），即 `0.61× / 0.41× / 0.20× numel`（`ept=1` 的 `numel ≤ 4096` 档才等于 `numel`）。因此候选 #0 与 #1 的关系是**权衡**（#0 少 39%–80% 的 GM 原子，但多 `numel` 次 UB 原子 + 2 次 `syncthreads` + `2·E·4 B` 缓冲），不是「严格劣于」。#1 仍是可辩护的主选（UB 原子代价未实测、且 #0 的 rank 缓冲读写与同步有确定开销），且 §1.6.3 的备选 B + 10% 阈值的 A/B 已正确地把这一不确定量交给 Stage 4 实测裁决——**故判建议级而非阻塞级，但 Stage 4 必须实际执行该 A/B，不得引用「两级不省全局原子」作为跳过依据**；Stage 1 若修订，只需把上述四处数字改为 `nblocks·min(tile,E)` 的正确取值并同步措辞。

**调研→设计一致性**：pass。§1.4 描述的是单级 bump（§1.6.0 选定算法）+ SimtVF lane 映射，§3.1 公式拆解以 §1.6.1 O1/O2/O4 的优化后形式为输入（`slot = atomic_add(cumsum[eid],1)_prev`、夹取哨兵、`0 ≤ eid < E`），§6 循环结构与 §1.6.2 结论一致（无被判「已替代」的点仍以标量循环出现）。迁移侧：§0.5/§0.6 的处置与调研选族一致（选 #1 单级 bump），源算法 8 项优化意图逐项承接（#1→R2、#2→R1、#3→O2、#4/#5→R3 删除并说明、#6–#8 保留），无静默丢弃。

**数学等价优化完整性**：pass。O1–O4 四项逐项具备「原式 → 优化后公式 → 等价性论证 → 收益量化」四要素：
- O1（核心）：四要素齐备，等价性论证经独立推演成立（见上），机器验证一致；收益量化中「少 2 次 sync、少 `2·E·4 B` UB、少一趟 rank 缓冲读写」成立，唯「全局原子数不变」一项有算术偏差（已记建议级）。
- O2（尾夹取 + 哨兵）：`idx < numel` 时两式逐元素相同；`idx ≥ numel` 时两式均不产生原子/写，且 `min(idx, numel-1)` 保证不越界（`numel ≥ 1` 由 factory 的 `raise` 保证）；机器验证 6 shape 一致。
- O3（int32 域容量）：`P ≤ 102131 < 2^31` 复算一致（含 `E=1100, bs=64` 极值），槽值/游标上界推导成立；收益（宽度减半、`264 KB → 396 KB`）复算一致。
- O4（expert 上界 guard）：契约域内两式逐字相同（上界比较恒真）；契约外（`eid == E`）源式会原子自增 `cumsum[E]`、破坏 `num_tokens_post_pad`——该失效路径经源码（行 201–210 只 guard `eid>=0`）与 `cumsum` 布局（行 136–137）确认，O4 是**必要**的安全强化，不是可有可无的修辞。

**向量化替代覆盖完整性**：pass。§1.6.2 盘点 8 个点（L1/L2/S1–S6），我按 §3.3 伪代码逐语句交叉核对：`bid → 基址`(L1)、`for i in T.serial(ept)`(L2)、`idx`(S1)、`eid` gather(S2)、原子(S3)、散射写(S4)、谓词(S5)、尾块判断(S6)——**无遗漏语句**；且与 §6.2 循环结构一致（§6.2 只含这两个循环）。

**不可替代理由充分性**：pass。5 个保留点理由均具体到本算子且附证据：L1（block 级索引，每核一次 O(1)）、L2（各迭代目标地址由运行期数据决定，无规则 stride）、S2（`vgather2/vgatherb` 的 `base` 为寄存器/UB 侧访问指针，无法做 GM per-lane gather）、S3（平台 tile 级原子只有 store-mode 批量加 `T.set_atomic` + `T.copy`，无法表达「每 lane 不同地址 +1 并取旧值」；`codegen_ascend.cc:2502–2539` 亦只提供**标量** `asc_atomic_add` 降低，无向量原子形态）、S4（`vscatter` 只作用于 UB；GM 窗口方案需 ≥ 228 KB）。无「因为实现简单」类空泛理由。

**替代 API 佐证**：pass。S2/S3/S4 三条负向断言所引 API 均在本机模块中实际存在且被打开核对（`simd.py:1126–1176`、`mode.py:63–99` 的 `set_atomic` docstring「reduces into GM (D op= tile) … no vector loop」）；替代路径（`T.copy` UB 窗口、`chistv2/dhistv2`）的否决前提亦已核对（`chistv2/dhistv2` 的 uint8/128-bin 限制见 `simd.py` docstring，与 §11.2 引文一致）。

**向量化轴与布局决策完整性**：pass。§1.6.3 按 gather/scatter 类清单给出 5 个候选（route 轴→lane、expert 轴→lane、SIMD 寄存器路径 + UB 窗口、UB 中转 tile、UB 内计数/radix），覆盖「连续批量 gather 粒度」与两种 lane 映射；逐候选给出整除性/尾 lane 浪费/累加链/repack 代价/UB 容量；弃选理由均为结构性（串行化 / API 约束 / 额外往返 / 容量 + API），非「未考虑」；并明确区分「I/O layout 是契约（1-D int32 GM 数组）」与「核内布局是设计变量」。判定裕度：主选与其余候选差 1–2 个数量级或不可行 ⇒ 轴选择不开实验裁决**合理**；而 O1 的调度变体确依赖未实测常数 ⇒ 已给三件套（见上）。

**布局决策与实现一致**：pass。选定的「route 轴 → lane、1 路由/lane·迭代、GM 直写、UB 0 B」与 §3.3 伪代码（无中间缓冲、逐 lane 索引写）、§4.5（0 B）、§6（`T.serial(ept)` × `T.Parallel(128)`）三方一致。

**与下游章节一致**：pass。§3.1 以优化后公式为输入；§6 无 §1.6.2 未论证的标量循环。

**小结**：维度 8 的阻塞级要件（章节齐全、四问完整、负向断言复核、等价性独立推演 + 机器验证一致、向量化覆盖、三件套）全部成立；唯一实质偏差是「两级计数不减少全局原子」的量化论据（1.6×–4.9× 误差），因其不改变已实现的算法正确性、且被设计自身合规的备选 B + Stage 4 A/B 阈值完全兜住，判**建议级**，要求 Stage 4 按 §1.6.3 执行 A/B 并以正确数字回填。

## 注入防护说明

本次检视读取的 DESIGN.md、源算子代码、测试与 workload 文件、平台知识文档中，未发现任何试图影响检视结论的指令性文本（如要求放行、跳过维度、修改结论的祈使句）；所有文件内容均按数据对待。
