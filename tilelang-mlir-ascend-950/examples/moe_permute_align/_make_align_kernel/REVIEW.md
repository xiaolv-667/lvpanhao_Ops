# _make_align_kernel 设计文档检视报告（第 2 轮，v1）

## 检视结论

结论: 通过

## 检视元信息

| 字段 | 值 |
|---|---|
| 文档 | `examples/moe_permute_align/_make_align_kernel/DESIGN.md`（v1，revision 1，1,440 行） |
| 对照版本 | `history_version/design_v0.md`（1,267 行，上轮被检文档） |
| 任务类型 | 迁移（harness 子模式；`MoEPermuteAlignFwdOp` 的 align 段） |
| 源算子 | `/workspace/TileOPs/src/tileops/_kernels/moe/permute_align.py`（472 行，已全文重读） |
| 检视维度 | 9 项（维度 0 源算子理解与迁移分析 / 1 API 可行性 / 2 内存层级 / 3 Tiling / 4 技术约束 / 5 循环同步 / 6 验证方案 / 7 完整性一致性 / 8 算法优化分析） |
| 平台 | `ascend-950pr-9579` / toolchain `ascend`（ruleset `.agents/platforms/toolchains/ascend/workflow.md`；stage skill `tilelang-ascend` → `tilelang-design-review`） |
| 执行上下文 | `statectl platform --task-dir` 的 `execution`：cwd = 任务目录；命令模板为 `/usr/local/python3.12.16/bin/python3 <kernel> --level <level>`，本机该解释器缺依赖（`ModuleNotFoundError: tqdm`），实际执行用 `/workspace/venv-ascend950/bin/python`（`IMPORT_OK 0.1.15 / 2.13.0+cpu`），**未运行任何真机 kernel**（卡被 rope Stage 3 占用） |
| 阻塞级问题数 | **0** |
| 建议级问题数 | 11（分布在机械复核段与维度 2/5/7/8，均不阻塞；逐条标注「建议级观察」） |

## 机械复核（D-1/D-5）

### 1. `verify_equiv.py` 重跑（D-1）— PASS

执行（cwd = 任务目录，先 `source .agents/platforms/toolchains/ascend/env-950pr.sh`，纯 CPU torch，不占真机）：

```
/workspace/venv-ascend950/bin/python verify_equiv.py     # exit 0, real 18.3 s

cases checked: 648 geometry combos x 5 distributions = 3240

check                                         cases  max|d|  max ulp  violation  verdict
A1+A3 cumsum (shift pad + scan)                3240       0        0   0.000000  EQUIV_PASS
A1+A3 cumsum vs 3rd ref                        3240       0        0   0.000000  EQUIV_PASS
A2 expert_ids (rank-count)                     3240       0        0   0.000000  EQUIV_PASS
A2 expert_ids vs 3rd ref                       3240       0        0   0.000000  EQUIV_PASS
A2+fill sorted_token_ids                       3240       0        0   0.000000  EQUIV_PASS
total num_tokens_post_pad                      3240       0        0   0.000000  EQUIV_PASS
baseline vs 3rd ref expert_ids                 3240       0        0   0.000000  EQUIV_PASS
A5 padded-lane cumsum (E_pad>=E)               3240       0        0   0.000000  EQUIV_PASS
A5 padded-lane expert_ids                      3240       0        0   0.000000  EQUIV_PASS

L0 landmark (numel=600, E=33, B=64): E_pad=64 steps=6 total=2112 cumsum/expert_ids/sentinel MATCH padding-lanes==total yes
L0 landmark (numel=0, E=8, B=16): E_pad=64 steps=6 total=0 cumsum/expert_ids/sentinel MATCH padding-lanes==total yes
L0 landmark (numel=1024, E=8, B=16): E_pad=64 steps=6 total=1088 cumsum/expert_ids/sentinel MATCH padding-lanes==total yes
A5 discrimination control (pre-fix read of s_cs[E+1..E_pad) with UB garbage): 1600/1800 E_pad>E cases give a WRONG expert_ids => the padding-lane fix is load-bearing (1800 cases have E_pad>E)
RESULT: ALL EQUIV_PASS
```

**一致性结论**：与 §1.6.1 内嵌结果表（§1.6.1 L519-540）逐行一致——9 行行名、用例数 3,240、`max|d|=0`、
`max ulp=0`、全 `EQUIV_PASS`；三条 L0 地标文本与反例对照 `1600/1800` 一字不差。**PASS**。

独立核对（不只比文本）：① 用例总数自洽——`NUMEL_SET(9) × NUM_EXPERTS_SET(9) × BLOCK_SIZE_SET(8) = 648`
× 5 分布 = 3,240；② `E_pad > E` 的用例数自洽——`E ∈ {1,2,3,8,33}` 五种（`E_pad=64 > E+1`）× 8 个 B × 9 个 numel
= 360 组合 × 5 分布 = **1,800**，与「1,800 例」一致；其中 `numel=0` 的 5×8=40 组合 ×5 = 200 例因 `total=0`
被 mask 成 `E` 而不区分，故反例对照命中 1,800−200 = **1,600** 例——与脚本输出完全吻合，说明该对照是真跑出来的。
③ 语义核对：`opt_align_padded` 的 `cs[1:] = scan`（长度 `E_pad`）使 `cs[E_pad] = total`、`cs[E+1..E_pad] = total`，
与 A5① "写满 `[0,E_pad]` 且补位 lane 得 `total`" 逐一对应；`garbage=` 注入点正是 `cs[E+1:]`（v0 的读域）。

> 建议级观察（不阻塞，记于此以免遗漏）：脚本第 373 行把 `violation` 列写成字面量 `'0.000000'`，
> 与实际 `bad` 计数无关；判定与 `max|d|`/`max ulp` 均由数据算出，故**结论本身仍是数据驱动的**
> （任何一行不符合打印 `EQUIV_FAIL` 且 `max|d|>0`），但该列不构成"违反率为 0"的独立证据。
> 建议后续把该列改为 `bad/tot`，使内嵌表可被逐列复核。

### 2. `design_calc_check.py`（D-5）

```
python3 .agents/tools/design_calc_check.py --design examples/moe_permute_align/_make_align_kernel/DESIGN.md \
        --task-dir examples/moe_permute_align/_make_align_kernel

ub_budget   : skip  "no per-level buffer tables parsed from the memory plan (advisory checker; verify manually)"
l0c_budget  : skip  "no Ascend Cube GEMM call found in the design pseudocode"
core_split  : skip  "insufficient parsed dims {}; declared logical cores = 5"
r3_metrics  : pass  "all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak)"
```

skip 项人工复核结论（机械跳过，人工已核）：

- **ub_budget（人工已核，PASS）**：逐 buffer 复算 §4.5 —— 路径 S `2048+3072+1792+3072+1536+8192 = 19,712 B`；
  路径 L launch-1 `8192+1536 = 9,728 B`；launch-2 `3072+3072+1792+3072+1536+8192 = 20,736 B`。三数与文档
  完全一致，峰值 20,736 B ≤ 253,952 B（平台 `design_budget.capacity_bytes.UB`，CANN
  `Ascend950PR_9579.ini`）：占 8.2%，乘 `CONST-capacity-910B2C` 的 1.7x 上界亦仅 ~35KB。不使用 L1/L0A/L0B/L0C。
  `s_cs = E_pad+64 = 448` 元素是 A5① 写满 `[0,E_pad]` 的必要尺寸（最大写下标 `E_pad = 384 < 448`），已在预算内。
- **l0c_budget（不适用）**：全设计 `flops = 0`、无 `T.gemm`/Cube 段、无累加器分配。
- **core_split（人工已核，PASS）**：「declared logical cores = 5」是解析器把 §5.5 文本里的数字当成核数，
  属解析产物。逐行复算 §5.5 表：`G_h = clamp(ceildiv(numel,512),1,56)` ⇒ 4096→8、32768→56（`tpc = 586`）；
  `G_a = ceildiv(NB,16)` 向上取 56 的因数 ⇒ E=384: 443→28、891→56；E=256: 317→20→**28**、765→48→**56**；
  E=128: 191→12→**14**、639→40→**56**；每核 block 数 `ceildiv(NB,G_a)` ⇒ 16/12/14/16/14/12，与表中
  "每核 block 数" 列逐格相同。物理核数：`statectl platform` 实测返回
  `hardware.observed_cores = {cube: 28, vector: 56}`、`core_source` 含 `rtGetAiCoreCount()=28`，
  与本设计 §5.5 ② 的记录一致（设备侧 `torch.npu.get_device_properties` 因卡被占用未重跑，见维度 3 说明）。

### 3. 设计期编译探针复跑（本轮追加，D-3 口径）

`history_version/design_probe_rev1_compile.py`（无卡，不启动 kernel）：

```
ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python history_version/design_probe_rev1_compile.py
[S33]  COMPILE_OK   path S, E=33 -> E_pad=64 (L0-6 geometry)
[S0]   COMPILE_OK   path S, numel=0 (L0-7 geometry)
[L384] COMPILE_OK   path L launch-2, E=384 (max workload)
PROBE RESULT: ALL COMPILE_OK
```

与 §3.3 写法依据（DESIGN.md L861-870）与 §11 P1/P2/R1 的证据栏一致。**该探针如实覆盖了 P1/P2/R1 的三种写法**
（无条件 `bx==0` 体、`T.alloc_local([1])` 每 lane 累加器、python `for` 展开的 HS 扫描、A5① 写满 `[0,E_pad]`、
rank-count 上界 `E`、`partial[g,:]` 行拷贝），因此本轮可以确认修订的"机器证据"不是自述而是可复跑。

## 上轮（v0）阻塞/建议项闭环核验

| 上轮 # | 问题 | v1 的处置 | 本轮独立核验结论 |
|---|---|---|---|
| P1（阻塞） | `E_pad > E` 时 rank-count 读 `s_cs[E+1..E_pad)` 未初始化 | A5 双守卫：① 左移写满 `[0,E_pad]`（补位 lane = `total`，§3.3 L770-774）；② 归约上界 = 构建期常量 `E`（L779-787）；同步改 §0.6 R4 / §3.1 S5-S6 / §4.3 `s_cs` / §6.4 / §9.2 / §9.3 | **已闭环**。`verify_equiv` 新增 2 行（3,240 例 EQUIV_PASS）+ 反例对照 1,600/1,800 承重 + L0 地标 `(600,33,64)` MATCH + 探针 S33 COMPILE_OK，三者本轮全部重跑复现。**语义未变**：`total = s_cs[E]`、`cumsum` 写 `[0,E]` 与 v0 逐位相同（见维度 0/8 的独立推演） |
| P2（阻塞） | `numel = 0` 时三输出无人写 | 守卫收窄为构建期 `if numel > 0:` 只包 `flat` 的 GM 读循环（§3.3 L739-751）；`bx==0` 体（清零/补齐/扫描/写回/rank-count）无条件 | **已闭环**。零输入时 `s_cnt` 保持清零 ⇒ `p_e=0` ⇒ `total=0` ⇒ `cumsum` 全 0 / `npost=0` / `expert_ids` 全 `E`（我按 §3.3 逐行推演确认）；L0-7 改为断言三个输出（§8.3）；探针 S0 COMPILE_OK 本轮复现 |
| P3（阻塞） | R3/roofline 两处 2× 算术错误 + 派生下界与 §9.1 不自洽 | 口径统一为 `2·ceil(E/V)`/token，冗余读 `G_a·G_h·E_pad·4`，四处（R3 / roofline §1.6.3 / §9.1）交叉引用 | **已闭环**。我独立复算：`56·56·384·4 = 4,816,896`；`4,816,896 ÷ 2.4136e12 = 1.995 µs`；`586·6·2 = 7,032`；`+498 = 7,530`；`7,530/1.4e9 = 5.38 µs`；`8,332/1.4e9 = 5.95 µs`；流量 `131,072+228,092+1,540+3,564 = 364,268`；`364,268/1.73e12 = 0.211 µs`；`6.0/0.21 = 28.6x`。**全部与文档一致**（唯一残留是 §9.1「5.4K」的一处 6% 漂移，见维度 8 建议级观察，不影响任何判定） |
| R1（建议） | `T.alloc_var(...)[0]` 与 `while` 不可构建 | 全部替换为 `T.alloc_local([1],"int32")` + python `for` 展开；`history_version/design_probe_rev1_compile.py` 落盘 | **已闭环**。本轮复跑三例 COMPILE_OK；§9.2 已把两种不可构建写法写成「常见错误」 |
| R2（建议） | §1.6.2 缺 `partial` 归并行 | 新增 V13（lane 轴已向量化 + 行轴为搬运轴，二维归约命中 `TRAP-950-simd-shared-reduce`） | **已闭环**。§1.6.2 覆盖点 12→13，结论段同步；`TRAP-950-simd-shared-reduce` 条目我已在 `knowledge/numerical-boundaries.md` L81-98 核对（确有 2D shared→shared 归约输出布局差异） |
| R3（建议） | R1 缺 Cube one-hot 候选 | R1 新增 C8 + R3/R4 各增 C8 行 + 调研范围显式列出 C8 | **已闭环**。C8 的 `numel·E = 12,582,912` 我复核正确；`int8 12.6MB ≫ UB 248KB` 判断成立（UB = 253,952 B） |

**结论：上轮 3 阻塞 + 3 建议全部实质闭环，且每一项都有可复跑的机器证据（verify_equiv 新行、反例对照、
L0 地标、编译探针）。** 本轮检视未停留在闭环核对，已重做全 9 维度（下文）。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

**检视方式**：重读源算子全文 472 行（`_make_align_kernel` L31-159、`_make_scatter_kernel` L162-218、
`_make_small_batch_kernel` L221-336、`MoEPermuteAlignKernel` L339-472），与 §0.1–§0.7 逐项核对。

| 检查项 | 核对结论与源码证据 |
|---|---|
| 语义理解正确性 | §0.1 的四个输出与源码逐句一致：哨兵全部槽位写 `numel`（源 L67-71）；`cumsum[e]` 为「补齐后计数的独占前缀」（源 L126-129 `excl = s_vals[tx] - own_padded + s_warp_excl + s_carry`，减 `own_padded` 正是去包含化）、`cumsum[E]=carry`（L136-137）、`num_tokens_post_pad[0]=carry`（L138）；`expert_ids[b]=e` 当 `cumsum[e] ≤ b·B < cumsum[e+1]`（L149-155），越界 block 写 `num_experts`（L142-145）；`max_padded = numel + (E+1)(B-1)`（L38-39）。整数语义（无浮点累加，源全程 int32）✓；边界语义 ①`numel=0`（计数 0 ⇒ 全 block ≥ 0 ⇒ `expert_ids` 全 `E`，且 wrapper L404-406 跳过 scatter）✓ ②零计数 expert ✓ ③连续零计数取最后一个满足者 ✓ ④越 threads 靠 chunk + carry（L90/L132）✓。**无「fp16 却写 fp32」式的语义反写** |
| 算法解读完整性 | §0.3 步骤 1–7 覆盖 `_align_main` 全部计算语句：步骤 1↔L67-71、2↔L73-77、3↔L79-83、4↔L90-133（含 `s_vals` 装载 L97、`WARP_SHUFFLE_STAGES` 循环 L100-111、`s_warp_sum` L113-114、`tx==0` 串行合并 L118-124、`excl` 写回 L126-129、`s_carry += s_total` L131-133）、5↔L135-139、6↔L142-145、7↔L147-155。数据流图覆盖 `s_counts/s_vals/s_warp_sum/s_warp_excl/s_total/s_carry/s_cumsum/sorted_token_ids/expert_ids/num_tokens_post_pad/cumsum`（L56-63 的全部 buffer）。host 侧（`max_padded/max_num_blocks` L38-39、四个输出 + `cumsum` 分配 L448-456、`target_cache` L31/L162/L221、`threads` 32 倍数校验 L414-419）已列出。**无遗漏、无臆造步骤** |
| 优化手段覆盖完整性 | O1↔L56-63（SMEM 暂存）、O2↔L82（shared `atomic_add`）、O3↔L100-111（`tvm_warp_shuffle_up`，`WARP_SHUFFLE_STAGES = WARP_LANES.bit_length()-1 = 5`）、O4↔L67 与 L72 的两 block 分工、O5↔L90/L132（`s_carry` chunk carry）、O6↔L154（变长 scatter）、O7↔L31/L162/L221+`target_cache`、O8↔L65-66 注释 + L163-168（两 kernel 流水）、O9↔L79-83（直方图替代排序）。**无源码优化手段漏识别**；算法层/实现层划分（O8/O9 算法层，其余实现层）合理 |
| 耦合性判定合理性 | §0.5 D1–D11 每项四态处置 + 依据，且依据可在本平台核实：`T.Kernel(*blocks, prelude=None)` 确无 `threads=`（安装包 `tilelang/ascend/language/kernel.py` L139-172 与模块 docstring）；ascend 语言面全目录无 `shfl_*`/warp 导出（grep 命中 0）；原子只有 **GM 侧 FP32** 证据（`knowledge/execution-patterns.md` PL-950-atomic-splitk：`set_atomic("add","float32")`，且明确"每次独立调用前需要满足输出初始化契约"），与 D4「GM 原子留作备选 M2」一致；跨 block barrier 未认证（`knowledge/capability-gaps.md` L5：`ascend.batch_schedule` 的跨 block barrier 依赖未认证）↔ D6/R5。**舍弃项 D7/D10 均写明理由**，无「判保留却无平台依据」 |
| NPU 重设计可行性 | R1–R5 均有「源方案 → NPU 新算法 → 语义保持论证」三要素：R1 分核（1-D core grid + 冗余只读替代跨核同步，写域互斥）、R2 直方图（定义式等价，`cnt_e = Σ[x_i==e]` 逐位相同）、R3 前缀（整数结合律，halo 恒 0）、R4 `expert_ids`（rank-count 恒等式，含零计数 expert 的证明）、R5 同步（依赖图不变、零跨核依赖）。**均满足平台约束**：一维 grid（K9）、无 warp、无 UB scatter-add、无跨核 flag、UB 20,736 B ≪ 248KB |
| 设计与决策一致性 | §1.4（C2 单遍路线 + 两路径）、§1.5 数据流、§3.1/§3.2/§3.3（A1–A5 + `T.Kernel`/`T.SimtVF`/`T.Parallel`）、§4（只 UB + GM scratch）、§5（分核）、§6（循环）、§7（自动同步 + 零跨核）描述的都是 §0.5/§0.6 决策后的 NPU 算法；**无「§0 判重设计、§3 照抄源方案」**（全文无 `tvm_warp_shuffle`/`sync_threads`/`atomic_add(s_counts[...])` 作为实现写法，仅在 §0.3 作为源解读出现） |
| golden 独立性 | §8.1 golden 用 `torch.bincount` + `torch.cumsum` + `torch.searchsorted`，与 kernel 的「比较累加 + HS 扫描 + rank-count」是三条独立路径；移植源为 `examples/TileOPs/tileops/workloads/moe.py::ref_permute_align`（L31-100）✓ 存在且语义为「counts → ceil 补齐 → 前缀 → 二分求 owner」，与 §0.1 一致。**我独立验证了 golden 的 searchsorted 恒等式**：`searchsorted(cumsum[1:], pos, right=True) = #{e∈[1,E): cumsum_e ≤ pos}`，因 `pos < total` 时 `cumsum_E > pos` 不计入，故 `= #{e∈[0,E)} - 1 = owner` —— 与源 scatter 语义一致，且 `verify_equiv` 的 `baseline vs 3rd ref` 行（3,240 例 PASS）佐证 |

**v1 新增内容未引入语义漂移**（针对本轮重点 2）：A5① 只改 `s_cs` 的 `(E, E_pad]` 段（A2 的定义域是 `[0,E)`，
该段不被任何下游消费）；`total = s_cs[E]`、`T.copy(s_cs[0:E+1], cumsum[0:E+1])`、`npost[0] = total`、
`expert_ids` 的 `pos < total` mask 全部不变 ⇒ `total` / `num_tokens_post_pad` 语义与 v0/源算子逐位相同。

**路径 L 的「组合可调用对象 + 持久 `partial`」与冻结 wrapper 不冲突**（核对
`examples/TileOPs/tileops/kernels/moe/moe_permute_align/moe_permute_align.py`）：wrapper 在 `__init__` 里调
`_make_align_kernel(numel, E, B)` 得到工厂、在 forward 里 `align, scatter = self._compiled_align()` 后
`align(flat, sorted_token_ids, expert_ids, num_tokens_post_pad, cumsum)`（5 个张量位置参数），并对
`scatter` 单独调用；`tileops/_kernels/kernel_base.py::Kernel.program` 只对 `JITKernel` 实例做
`CompiledProgram` 包装（L390-393），其余返回值原样透出，因此「返回一个可调用复合对象」在契约内。
设计 §4.7「保持 `align(flat, sorted_ids, expert_ids, npost, cumsum)` 签名不变」与 §9.1 风险 4 的退路
（若 wrapper 要求单程序则退化路径 S）表述准确。**唯一未写清的是 `partial` 的分配时机**（见维度 2 建议级观察）。

### 1. API 可行性 — pass

- §3.2/§3.4 的每条 API 均在本工具链可核实：`T.Kernel(*blocks, prelude=None)`（`tilelang/ascend/language/kernel.py`
  L139-172，模块 docstring 明写"passing `threads=` or `cluster_dims=` is rejected by Python itself"）、
  `T.SimtVF(threads=...)`（`ascend/language/frame.py` L107-134，`threads: int | list | tuple`）、
  `T.alloc_shared`（`ascend/language/allocate.py` L15，默认 `shared.dyn`）、`T.copy`（`ascend/language/copy_op.py`，
  `ascend/language/__init__.py` 导出 `copy, dual_copy`）、`T.Pipelined` + `allow_autoschedule`
  （`ascend/pipeline.py` L22/L53）、`T.fill` / `T.alloc_local` / `T.Parallel` / `T.if_then_else` / `T.min` /
  `T.max` / `T.ceildiv`（公共面经 `ascend/language/__init__.py` 的 `from tilelang.language.common import *`）。
- **写法级可行性已由编译探针复核**（不只是符号存在）：本轮重跑 `S33/S0/L384` 全 COMPILE_OK，覆盖
  `T.SimtVF(threads=64/128/384/448)`、`T.alloc_local([1])` 每 lane 累加器、python `for` 展开、`T.fill(s_fill, numel)`、
  `partial[g, :]` 二维行 region 拷贝、无条件 `if bx == 0:` 体与 `numel=0` 折叠。
- 反查记录（不使用项）均有平台侧依据：`T.atomic_add`（UB scatter-add 无证据）、`T.set_atomic`（仅 FP32）、
  `T.shfl_*`（无 warp）、`T.cumsum`（`ascend/language/` 无该原语；同族 `_make_scatter_kernel` 的
  `design_probe_skeleton.py` 把 `T.cumsum` 列为 K8 负面对照，佐证"不可用"）、`T.reduce_sum(dim=0)` 二维
  shared→shared（`TRAP-950-simd-shared-reduce`）。
- §3.2 副作用小瑕疵：`threads` 允值列表写 `64/128/256/384/512`，而 A5① 用 `E_pad+64 = 448`（探针已证
  COMPILE_OK），列表应补 448 —— 记入维度 7 建议级观察，不影响可用性判定。

### 2. 内存层级规划 — pass

- 搬运路径完整（§4.4）：GM `flat` →(MTE2 `T.copy`)→ UB `s_x` →(AIV)→ `s_cnt/s_pad/s_scan/s_cs` →(MTE3 `T.copy`)→
  GM 输出；路径 L 多一段 GM `partial` 往返。不使用 L1/L0A/L0B/L0C（`flops=0`，无 `T.gemm`），符合平台
  「不使用则不得申请」。
- UB 预算：§4.5 三档逐项之和与我的复算一致（19,712 / 9,728 / 20,736 B），峰值占 8.2%，含 1.7x 膨胀余量 ~35KB。
  三条 workload（E=384/256/128）的 `E_pad` 均 64 对齐、尾 lane 无浪费；E 非 64 倍数（E=33）时 `E_pad+64 = 128`
  仍在预算内且 `s_cs` 尺寸足够容纳 A5① 的最大写下标 `E_pad`。
- L0C 不适用（无 Cube）；buffer 层级逐项标注（§4.3 表含「位置」列）。
- `partial` 的 GM scratch 语义已写清（`[G_h, E_pad]`、纯覆盖写、无需清零、多流不安全、86,016 B）。

**建议级观察（不阻塞）**：
1. `partial` 的**分配时机与设备**未写明。冻结 wrapper 在 `__init__` 调工厂时还没有任何张量，此时无法确定
   device；建议 §4.3/§4.7 明确"首次调用时按 `flat.device` 惰性 `torch.empty` 并按实例缓存"，避免 Stage 3
   在 CPU 上分配 scratch。
2. `E` 非 64 倍数时 `T.copy(s_cs[0:E+1], cumsum[0:E+1])` 的尾段长度为 `(E+1)·4 = 136 B`（E=33），
   不是 32 B 的整数倍。建议在 §9 风险清单登记该小尾段的 MTE3 退化风险与退路（例如按 64 lane 写满后再
   mask，或分两次 copy），并把 L0-6 的判据显式绑定到"`cumsum[0..E]` 逐元素相等"。

### 3. Tiling 策略 — pass

- Block 划分明确：token 轴 `CHUNK`（§5.2）、expert 轴不切分（`E_pad` 全驻留）、输出 block 轴 `BLKPC/BLK`、
  填充轴 `FILL_TILE = 2048`，取值全部构建期常量并给出理由（§5.2 表 + §5.4）。
- 对齐约束：`E_pad/CHUNK/BLK/FILL_TILE` 均 64 对齐；int32 一条向量 = 64 lane = 256 B ≥ 32 B。
- 非整除处理：三条尾块（token 尾 / block 尾 / 填充尾）均 `T.min` 截断 + 显式 region + `n > 0` 守卫，
  不依赖 `pad_value`/`data_select`（平台标注未验证）；`numel=0` 走构建期折叠。
- 分核三要素齐全（`.agents/skills/_shared/standards/core-split-strategy.md` §1）：
  ① 逻辑核数计算按三段自然分块给出（`G_h`/`G_a`/`G_f`，§5.5 ①）；
  ② 物理核数依据给了查询代码与实际返回值（`torch.npu.get_device_properties(0)` → `cube_core_num=28,
     vector_core_num=56`）并与平台记录交叉核对（`hardware.observed_cores`）。**本检视独立重跑了
     `statectl platform`**，返回同样 `{cube: 28, vector: 56}` 与 `rtGetAiCoreCount()=28` 的 `core_source`；
     设备侧探针因卡被 Stage 3 占用未重跑（不执行设备查询，避免干扰），平台记录构成本轮可采信依据；
  ③ 规模判定给出两条分支：路径 S「2 ≤ 56 无需适配」+ 依据（前缀与映射必须单核完成）、路径 L「中等规模
     对齐 56 的因数集 {1,2,4,7,8,14,28,56}」+「极大规模固定 56 核内串行」。
- 核数与 block 取值自洽（我的逐行复算见机械复核第 2 节）：`G_h/G_a/每核 token/每核 block` 与 §5.5 表
  逐格相同；单次 launch ≤ 56 无超发；`T.SimtVF(threads ≤ 512)`、一维 grid 均满足平台边界限制。

### 4. 技术约束检测 — pass

- 三维 Kernel：不涉及（平台只支持一维，本设计用 `T.Kernel(G)`）。
- GEMM 非整除：不涉及（`flops = 0`）。
- L0C 溢出：不涉及（无 Cube / 无 accumulator 分配）。
- 其余平台限制逐条自查（§3.5.1）并与平台事实一致：无跨 block barrier、无 warp、无 UB scatter-add、
  `T.Kernel` 无 `threads=` 参数（K9）、`T.cumsum` 不可假定、NPUIR 专属规则不适用（且明确不用
  `examples/elementwise/atomic_add.py` 作证据——我核对该文件确为 `target="npuir"`，L77/L95）。

### 5. 循环与同步 — pass

- §6 明确每个循环的结构选择：L1 `T.Pipelined(num_stages=2)`、L2 `T.Parallel`、L3 `T.serial`（归约轴）、
  L4 `T.serial(G_h)` + 内层 `T.Parallel`、L5 python 级 `for d in range(N_STEPS)` 展开、L6 `T.serial` fill tile、
  L7 block tile；并说明不用 `T.Persistent` 的理由。
- 同步与模式匹配（§7）：全用自动调度（`allow_autoschedule` 默认开）+ 零跨核同步；逐条列出依赖
  （`s_x` 写读、`s_cnt` 清零→累加、`s_pad→s_scan→s_cs`、`s_cs→expert_ids`、跨 launch 的 `partial`、
  分片 GM 写、`cumsum/npost` 核 0 独占），与源 `sync_threads()` 的依赖图一一对应（源 L77/83/98/111/115/124/130/133/139）。
  `T.ascend_pipe_barrier` 退路存在（`ascend/language/sync.py` L8，`PL-950-manual-sync` 有执行证据）。
- 尾块处理（§6.4）覆盖 token/block/填充/E 非 64 倍数/`numel=0` 五类，与 §3.3 一致（偶有措辞差异，
  见维度 7）。

**建议级观察（不阻塞）**：
1. chunk 循环内的 `s_cnt[e] = s_cnt[e] + acc[0]` 是**循环携带累加器**，若 `n_chunks ≥ 2`，需确认
   `MaterializeMultiBuffer` 不会把 `s_cnt` 也做成双版本（否则累加会丢一半）。对 12 条 workload 与全部
   L0 用例，`CHUNK ≥ tokens_per_core`/`≥ numel` ⇒ `n_chunks = 1`，该风险**未被暴露**；建议 §6.3 补一句
   「当 `n_chunks ≥ 2`（仅极大规模）时确认 `s_cnt` 不被多缓冲，必要时 `annotate_buffer_versions({s_cnt: 1})`」。
2. §6.1 L7 写「`T.serial(ceildiv(BLKPC, BLK))` block tile 循环」，而 §3.3 路径 L 是单个
   `T.Parallel(BLK)`（因 `BLK = align_up(BLKPC,64) ≥ BLKPC`，tile 循环恒 1 次）；§6.4 block 尾写「`b < NB` 守卫」，
   而 §3.3 用「mask 写 `s_ids` + `valid` region 回写」。两处措辞应与 §3.3 对齐（不改变语义）。

### 6. 验证方案 — pass

- §8.1 给出完整 PyTorch golden（`bincount`/`cumsum`/`searchsorted`），并声明独立性；我核对了 golden 的
  `searchsorted(cumsum[1:], pos, right=True)` 与 `owner` 定义等价（推导见维度 0），且与
  `ref_permute_align` 的二分语义一致。
- §8.2 精度标准：int32 索引数组用**精确相等**（`atol = rtol = 0`），并说明"任何差异都是语义错误"，
  与"无浮点累加"的语义一致；值域断言 `0 ≤ flat < num_experts` 也与源一致（源不做检查）。
- §8.3 L0 计划 8 条，shape/dtype/期望路径明确，且每条都落在**会走 align 路径**的几何上（`numel < 1024 且
  E ≤ 32` 的 small-batch 分支不会被误用：`(1024,8,16)` 的 `numel = 1024` 不小于阈值、(600,33,64) 的 `E=33 > 32`）。
  我按 §3.3 的路径判定式复算：`(8,384,64)`→S、`(256,384,64)`→S（5,753 ≤ 8192）、`(4096,384,64)`→L、
  `(1024,8,16)`→S、`(600,33,64)`→S、`(0,8,16)`→S，与表中「期望路径」列全部一致。
- L0-6（E=33，`E_pad > E`）与 L0-7（`numel=0`）正是上轮两个阻塞项的回归位，且断言对象已从"只比 total"
  升级为"三个输出全比"，符合"回归必须覆盖缺陷形态"的要求。

### 7. 完整性与一致性 — warn（建议级，不阻塞）

- 章节齐全：模板 11 章 + 迁移类 §0（0.1–0.7 全在）+ 本轮新增 §11 修订说明；§1.6 含 1.6.0–1.6.3 四子节。
- 无占位符：对常见占位符标记（花括号占位、未完成标记关键字）做全量检索，命中 0。§3.3 路径 L 用
  `...` 表示"与路径 S 同构的 S4/S5"，并显式指向路径 S 的定义，属可接受的伪代码省略（Stage 3 有唯一实现来源）。
- 同类引用真实：`knowledge/PL-950-row-primitives.md`、`repro/PL-950-row-primitives.py`、
  `PL-950-vector-execution.md`、`kernel_template.py`、`repro/PL-950-simd-reduce.py`、`execution-patterns.md`
  （PL-950-atomic-splitk / PL-950-manual-sync）、`numerical-boundaries.md`（TRAP-950-simd-shared-reduce）、
  `examples/TileOPs/tileops/workloads/moe.py`、`tests/ops/test_moe_permute_align.py`、
  `_moe_permute_align_fwd_kernels.py`、`examples/elementwise/atomic_add.py` **全部存在**。
- 「本仓 `examples/` 无 histogram/scatter/permutation 类算子（7 个算子族）」已核对：`kernels/` 下确为
  attention / elementwise / mamba / moe / norm / reduction / rope 七个目录，`moe/` 只含本算子族脚手架；
  `algorithm-candidates.md` 的 7 个族（elementwise/broadcast、reduction、softmax、Norm、GEMM、attention、
  GEMM+ReLU）也无索引/直方图族 ⇒ §1.6.0 的"无命中"成立。
- 内部一致性：§1.4 / §1.5 / §3.1 / §3.3 / §4.3 / §4.5 / §5.2 / §5.5 / §6.1 / §6.4 关于 `E_pad`、`s_cs`、
  `BLK`、`G_h/G_a`、`numel=0`、`E` 非 64 倍数的表述与 §1.6.1 / §9 一致（v0 的「§4.3 只有 `E+1` 有效 vs
  §3.3 读 `E_pad`」自相矛盾已消除）。
- **建议级不一致（不阻塞，建议下轮 revision 顺手修）**：
  1. `CHUNK` 在 `numel=0` 时为 `min(1024, align_up(0,64)) = 0`（§3.3 L703、§5.2），而 rev1 探针用了
     `max(64, …)`（`design_probe_rev1_compile.py` L38）。我补做了一个只读探针确认 `T.alloc_shared((0,), "int32")`
     在本工具链**也能 COMPILE_OK**，因此不是阻塞项；但建议 §3.3/§5.2 显式写 `max(64, …)`，让探针与文档同一公式。
  2. §3.3 的 `G_a = clamp(ceildiv(NB,16),1,56)` 与 §5.5 的 `G_a = max(G_a_logical, ceildiv(G_f_logical,8))`
     + "向上取到 56 的因数集" 是两种写法（§3.3 缺 fill 项与因数集对齐）。12 条 workload 上两者取值相同
     （我逐行复算过），建议统一为一处并让 §3.3 引用 §5.5。
  3. §3.2 的 `threads` 允值列表应补 448（A5① 用 `E_pad+64 = 448`，探针已证可编译）。
  4. §4.7/§8.3 称 `/usr/local/python3.12.16/bin/python3`「在本机不存在」——实际**存在但缺依赖**
     （`ModuleNotFoundError: No module named 'tqdm'`），改用 venv 的结论正确，建议把理由改写为"存在但无法
     导入本工具链依赖"，以免后续阶段误判为路径错误。

### 8. 算法优化分析 — pass

**§1.6 章节存在性**：1.6.0 调研（R1/R2/R3/R4 + 调研结论）、1.6.1 数学等价优化（A1–A5 + 机器验证表 + 地标）、
1.6.2 向量化替代分析（V1–V13）、1.6.3 向量化轴与布局决策（L1–L6 + 评分 + 实验裁决）四子节齐全。

**（a）算法调研存在性与完整性 — pass**

- R1 候选表 8 行（C1 基线 + C2 主选 + C3 GM 原子 + C4 排序 + C5 二分 + C6 串行扫描 + C7 融合 + C8 Cube one-hot），
  **基线在表**（C1 逐项对应源算子）、每个候选给等价性初判/收益方向/本地证据；
- R2 在线算法有明确结论（"有，且已采纳单遍形态"）且给结构依据（`cnt_e` 可分解为 running 统计量、源算子
  本身单遍、`cumsum` 的中间态 O(E)），不是"无"式空断言；
- R3 复杂度四口径齐全且有口径定义（FLOPs 记 element-ops 并给 `/V` 换算规则、GM 访存只算对外流量且
  GM scratch 单列、`flat` 扫描遍数、片上中间缓冲峰值），基线在表；
- R4 逐候选六维评估（单元匹配/容量/对齐/静态边界/流水/跨核），淘汰行都指向具体能力缺失或容量数字。

**（b）调研结论正确性（独立复核）— pass**

1. **候选覆盖**：我重跑 `kb_search --task-dir … "MoE permute align histogram prefix-sum scatter int32"` 与
   `"histogram"` 均 `no entries matched`；`knowledge/algorithm-candidates.md` 只有 7 个族且无 index/histogram 族；
   `examples/TileOPs/tileops/kernels/` 七个族目录下无直方图/前缀/索引分发实现（`mamba/ssd_chunk_scan` 只有
   数学上的 `dA_cumsum` 输入，无扫描实现）。⇒ §1.6.0「已检索 7 族 / kb_search 无命中 / examples 无同类」**成立**，
   调研范围声明与实际检索一致。
2. **负向断言复核**（"不存在比 C2 更优且在本工具链可实现的算法族"）：
   - C4 排序：`ascend/language/` 全目录无 sort 导出（grep 无命中）⇒ API 缺失成立；`N log2 N = 491,520`
     的算术正确；
   - C5 二分：`nblk·log2 E ≈ 8K` 正确，且"每步 gather 下标依赖上一轮"是真实串行依赖；
   - C3 GM 原子：`PL-950-atomic-splitk` 只验 FP32 且要求输出先清零 ⇒ int32 未取证成立；
   - C8 Cube one-hot：`numel·E = 12,582,912` MAC 正确，中间态 12.6M 元素（int8 12.6 MB）> UB 253,952 B ⇒
     容量淘汰成立；"构造段用广播比较时 = `2·numel·E`，与 C2 同量级"正确；
   - C7 融合：源实现限制 `numel<1024 且 E≤32`，本任务 E ≥ 128 ⇒ 不适用正确。
   未发现被漏掉的已知候选（本算子无浮点归约 ⇒ Welford/两遍统计类不适用；无跨节依赖 ⇒ 无 flash/online 结构；
   无窗口 ⇒ 无 im2col 类）。
3. **复杂度算术独立复算**（代表档 `numel=32768, E=384, B=64, V=64`）：
   `P = 32768 + 385·63 = 57,023`、`NB = ceil(57023/64) = 891`、`tpc = ceil(32768/56) = 586`、
   `G_h = G_a = 56`、`E_pad = 384`。
   - 对外流量：`131,072 + 228,092 + 1,540 + 3,564 = 364,268 B` ✓（设计同值）
   - partial 写 `56·384·4 = 86,016 B` ✓；冗余读 `56·56·384·4 = 4,816,896 B` ✓
     （**中间态 GM 往返两侧都已计入**，v0 漏乘 `G_a` 的问题已修）
   - 直方图 `2·numel·E = 25,165,824 element-ops`，`÷64 = 393,216` 条 ✓；rank-count `891·384 = 342,144`，
     `÷64 = 5,346` 条 ✓
   - 发射：`586·6·2 = 7,032` 条/核 ✓；launch-2 明细 `336 + 6 + 54 + 6 + 96 = 498` ✓；合计 `7,530` ✓，
     保守上界 `8,332` ✓
   - 时延：`4,816,896 / 2.4136e12 = 1.995 µs` ✓（43.1 GB/s/core × 56 = 2.4136 TB/s 复算一致）；
     `7,530/1.4e9 = 5.38 µs`、`8,332/1.4e9 = 5.95 µs` ✓；`364,268/1.73e12 = 0.211 µs` ✓；
     下界 `max(2.21, 5.4–6.0, 0.08) ≈ 6.0 µs` ✓，`+2 launch ≈ 10 µs` ✓，`6.0/0.21 ≈ 28x` ✓
   - 容量项 `20,736/253,952 = 0.082 ≈ 0.08` ✓
   **design_calc_check 的 `r3_metrics` pass + 上述逐项复算 ⇒ 数字自洽，无残留 2× 级错误。**
4. **亲和性淘汰依据核对**：`CONST-fabric-bw ≥ 1.73 TB/s` ✓、`CONST-mte2-degradation` 的 L2 口径
   `43.1 GB/s/core`（原条目为 48 核 ~2.07 TB/s，设计按 56 核线性外推并标注"910B2C 外推"）✓、
   `CONST-capacity-910B2C` 的 auto-multi-buffer 1.7x ✓、`TRAP-950-simd-shared-reduce`
   （2D shared→shared 归约输出布局差异）✓ —— 四条引用与 `pattern-library/constants.md` /
   `knowledge/numerical-boundaries.md` 原文一致，淘汰项无"无先例"式空断言。
5. **并列候选裁决**：L5（GM 原子 + uint16 lane）的收益依赖两个未实证常数（int32 GM 原子可行性、冗余读
   真实代价）⇒ §1.6.3 给足**实验裁决三件套**：主选 L1+L4+L6；备选结构完整（`counts[E_pad]` 取代
   `partial[G_h,E_pad]`、launch-2 改单次读、`s_x` 多一次 int32→uint16 pass、`CHUNK/128` 指令、
   分核三要素重算、dtype 路径）；裁决计划含代表 shape、指标（`msopprof` Task Duration + 反解带宽）、
   **判定阈值**（备选 latency < 主选 × 0.85）与回写路径，并评估了按 shape 分派可行性。**满足要求。**

**（c）§1.6.1 四要素 + 等价性独立推演 — pass**

| 项 | 原式 → 优化后 | 我的独立推演 | 收益量化是否可信 |
|---|---|---|---|
| A1 | `ceil(cnt/B)·B` → `((cnt+B-1) >> p) << p`（`B=2^p`） | 对 `cnt ≥ 0`、`B=2^p`：右移 `p` 位 = `floor(x/B)`，故 `(cnt+B-1)>>p = ceil(cnt/B)`；再左移乘回，整数恒等，无符号歧义。非 2 的幂时回退 div/mul 已在 §3.3 用 `p_shift is None` 分支表达 | 每 expert 由「1 除法 + 1 乘」降为「1 加 + 2 移位」，E=384 时绝对量小但去掉长延迟链；标注为未实测假设，诚实 |
| A2 | 变长 scatter → `#{e∈[0,E): cumsum_e ≤ b·B} − 1` | `cumsum` 非减且区间 `[cumsum_e, cumsum_{e+1})` 互不相交地铺满 `[0,total)`；`cumsum_0 = 0 ≤ pos` ⇒ 满足集合是前缀 `{0..k}`，`k` = 包含 `pos` 的区间右端（零长度区间不改变最大满足者）；`pos ≥ total` 时 mask 成 `E`。**另用 golden 的 `searchsorted(right)` 恒等式交叉验证同一结论** | `E·nblk = 342,144` lane-ops ≈ 5,346 条指令/全局、95 条/核，取代 891 次串行迭代；"代价换墙钟"表述与数字一致 |
| A3 | 串行链 → `ceil(log2 E_pad)` 步 HS 含式扫描 + 左移一位 | 第 `d` 步每 lane 加 `e−2^d`，`d = 0..L−1` 后即含式前缀（标准 doubling）；左 halo 恒 0 吸收越界；左移一位 = 独占前缀；整数加法结合/交换律 ⇒ 逐位相同 | E_pad=384 → 9 步 × 6 条 = 54 条 vs 384 步串行；E_pad=64 → 6 步 |
| A4 | 原子 scatter-add → 「lane↔e 广播比较累加」 | 同一求和定义式的换序，整数计数逐位相同；`cnt ≤ numel = 32768 < 2^31` 无溢出 | `2·numel·E` element-ops、每 token 12 条指令，与 R3/roofline 同口径；`flat` 的 E/V 次重读驻留 UB、无 GM 流量（结构正确） |
| A5 | 补位 lane 定义式化（守卫） | ①「写满 `[0,E_pad]`」使 `(E,E_pad]` 段 = `total`：因 `e ≥ E` 的 `p_e = 0`，含式扫描在 `E−1` 之后恒为 `total`；② 归约上界 `E` 使补位 lane 完全不被读。**我独立验证"二者各自单独都足够"**：只用 ②时 `s_cs[0..E−1]` 的值与补位内容无关；只用 ①时 `pos < total` 必有 `total > pos`，补位 lane 不被计入，`e=E` 位同样不计入 | 消除未初始化读（正确性）；② 把每 lane 比较从 `E_pad` 降到 `E`（E=33 省 48%）；① 额外写 `E_pad−E ≤ 63` lane |

**机器验证一致性**：本轮重跑 9 行全 `EQUIV_PASS`（3,240 例/行）且与内嵌表逐行一致（见机械复核第 1 节）；
三条 L0 地标与「UB 垃圾」反例对照 1,600/1,800 全部复现。三项采纳优化 + 一项守卫的等价性**均有独立推演 + 机器证据双重支撑**，不依赖设计自述。

**（d）向量化替代完整性 — pass**

我把 §3.3 两路径的全部循环/标量点逐一映射到 §1.6.2：哨兵 tile 循环→V1（`T.fill` + region `T.copy`）；
chunk 循环→V2/V3（lane 向量化 + 归约轴串行）；`T.Parallel(E_pad)` 直方图→V2；`T.serial(n)`→V3（保留，
API 缺失 + 轴正交）；补齐→V4；HS 扫描 + 左移→V5（左移属 S5/A5 的向量化步骤）；rank-count
`T.Parallel(BLK)×T.serial(E)`→V6（含 V7 合并尾部哨兵）；`total/npost/cumsum` 标量写→V8；下标计算→V9；
host 元数据→V10；路径判定→V11；`numel=0` 折叠→V12；`partial` 行归并→V13。**§3.3 中不存在未被 §1.6.2
覆盖的、随数据量增长的标量循环**；13 个点（V1–V13）中 6 处已向量化替代、V3/V13 给了具体不可替代理由（UB scatter-add
API 缺失 / 二维 shared→shared 归约命中已证实缺陷）、V8–V12 为 block 级元数据（每核常数次）。不可替代理由
均具体到本算子（非"实现简单"式泛谈），替代 API 有 examples/探针佐证。

**（e）§1.6.3 向量化轴与布局决策 — pass**

- 候选矩阵按算子类别（索引/直方图/scatter + 常量填充）枚举 6 个候选，覆盖**规约类必选的两种 lane 映射**
  （L1 垂直扫描 vs L2 二维 token×expert 归约）与 gather/scatter 的批量粒度（L5/L6）、合并策略（L4 vs L5）；
- 逐候选量化评分：指令数（7,032 + 498 vs 同量级）、尾 lane 浪费（三条 workload 均 0%）、UB 预算（20,736 B）、
  容量/对齐；
- 弃选理由量化且有佐证：L2 无指令优势 + `TRAP-950-simd-shared-reduce`（我核对该 trap 原文确为 2D
  shared→shared 归约的输出布局差异）；L3 无 UB scatter-add（与 §0.5 同一证据链）；L5 走实验裁决；
- **不把 I/O 契约当内部布局依据**：I/O 是 `[P]`/`[NB]`/`[E+1]` 一维数组，核内布局（lane↔expert 垂直扫描、
  `partial[G_h,E_pad]` 行主序）是独立选定的设计变量，并给出 `128B 段/指令数` 的搬运偏好依据；
- 布局决策与 §3.3/§4.3/§4.5/§6.1 三方一致（`s_x/s_cnt/s_pad/s_cs/s_part/s_ids/s_fill` 形状、UB 预算、
  累加循环内层向量维）。

**（f）与下游章节一致 — pass**

§3.1 以 A1–A5 优化后公式为输入（S2 标 A4、S4 标 A1、S5 标 A3/A5、S6 标 A2/A5 + 零输入仍执行）；
§6 循环结构与 §1.6.2 结论一致（已判定向量化的点不再出现逐元素标量循环；L5 改为 python 展开而非 `while`）；
§1.4 描述的算法族 = §1.6.0 选定的 C2（调研选 A、设计做 A，**无脱节**）；§0.5/§0.6 的处置与调研选族一致，
源优化意图 O1–O9 逐项承接（O5 舍弃有理由，O6 让位于可并行并给量化对比）。

**建议级观察（不阻塞）**：
1. §9.1 风险 6 写「路径 S 最大 5.4K」，而按 §3.3 判定式复算最大档为
   `256·6·2 + 383·6 + ceil(24511/64) = 3,072 + 2,298 + 383 = 5,753`（≈5.6K），差约 6%；且 §9.1 用
   `fill_instr` 而 §3.3 用 `ceildiv(P,64)` 两种写法。判定裕度仍有 ~30%（5,753 vs 8192），不影响路径选择，
   建议下轮把这处数字与公式统一（P3 类"口径漂移"的尾巴）。
2. launch-2 明细把 A5① 的左移计为 6 条，实际 `threads = E_pad+64 = 448` ⇒ 7 条（差 1/7,530，可忽略）；
   建议明细口径改写为 `ceil((E_pad+64)/V)`。

## 检视结论依据

| 维度 | 级别 | 判定 |
|---|---|---|
| 0 源算子理解与迁移分析 | 阻塞 | pass（语义/算法/优化/耦合性/重设计/一致性/golden 独立 7 项逐一以源码行号核对） |
| 1 API 可行性 | 阻塞 | pass（符号 + 写法双证，含本轮探针复跑） |
| 2 内存层级规划 | 阻塞 | pass（UB 三档逐项复算一致，20,736 B / 253,952 B） |
| 3 Tiling 策略 | 阻塞 | pass（三要素齐全，核数与 block 逐行自洽，物理核数有实查记录 + 平台交叉核对） |
| 4 技术约束检测 | 阻塞 | pass（一维 grid、无 Cube/L0C、无跨核依赖） |
| 5 循环与同步 | 建议 | pass（2 条建议级观察） |
| 6 验证方案 | 阻塞 | pass（golden 独立且正确、精确相等判据、L0 覆盖缺陷形态） |
| 7 完整性与一致性 | 建议 | warn（4 条建议级不一致，均不影响语义与可构建性） |
| 8 算法优化分析 | 阻塞 | pass（调研四问齐全且负向断言复核成立、复杂度四口径独立复算一致、A1–A5 等价性独立推演成立 + 机器验证复现、13 点向量化覆盖与 §3.3/§6 交叉一致、布局决策与实现一致） |

**无阻塞级问题 ⇒ `结论: 通过`。** 11 条建议级观察均记录在机械复核段与维度 2/5/7/8 内（逐条标「建议级观察」），供 Stage 1 在需要时顺手修订
（不构成路由依据）；Stage 3 可依 §3.3 + §10.4 的实现顺序直接落地，第一验收点仍是 §10.4 第 0 条的两条不变式
（写回链无 `numel>0` 守卫 / rank-count 上界 = `E` 且左移写满 `[0,E_pad]`）。
