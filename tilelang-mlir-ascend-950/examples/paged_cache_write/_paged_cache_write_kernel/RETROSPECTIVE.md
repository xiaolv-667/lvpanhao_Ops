# RETROSPECTIVE — paged_cache_write / _paged_cache_write_kernel

## Stage 1 (design) Retrospective — first_design — 2026-11-23

迁移任务（harness 模式：`TileOPs` GPU `PagedKVCacheWriteFwdOp` → Ascend950PR_9579 / `target="ascend"`）。任务约束：仅产出 `DESIGN.md`（+ 本复盘），全程禁止编译、禁止运行设备侧 kernel，`execution` 的 device 模式被该硬约束覆盖，设计期只做只读静态 API 查证与 CPU 数值等价实验。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Constraint-doc | `design_calc_check.py` 的 `_section()` 把任何以 `#` 开头的行（含 Python 代码块内的 `#` 注释）都当作章节标题，导致 §5.2 的代码块把 `Tiling` 章节截断，后续 UB / 分核解析落空。 | `DESIGN.md#5.2` 原稿代码块首行 `# 逻辑块折叠`；`.agents/tools/design_calc_check.py` `_section()` | 在 `_section()` 中维护 fenced-code 状态（跳过 ``` 之间的行）再判定标题。 | R |
| Constraint-doc | 分核检查的正则 `逻辑核数[^\d]*(\d+)` 取 §5 内**第一次**出现，若三要素清单文字（「① 逻辑核数计算 … 按 standard §1 组织」）先出现，则 `[^\d]*` 吞到 `§1` 得到 declared=1，与重算值 4096 冲突并误报 fail。 | `DESIGN.md#5.5` 原稿清单行 vs `.agents/tools/design_calc_check.py` `check_core_split()` | 让正则要求声明形态（`逻辑核数**:\s*(\d+)`），或要求 standard 明确「声明行必须以 `**逻辑核数**: N` 开头」。 | R |
| Constraint-doc | UB 预算表解析要求：`**UB**` 标记行、buffer 名以拉丁字母开头、末行 `合计` 不得出现「容量 / 参考 / 佐证 / ≤」字样且单元格不得用反引号包裹数字。这些约束未写入任何文档，只能靠反复试跑反推。 | `DESIGN.md#4.6` 表；`.agents/tools/design_calc_check.py` `check_ub_budget()` | 在 `design_calc_check.py` 顶部 docstring 或 `references/quality-checklist.md` 增一节「表格解析约定」。 | R |
| Info-source | 任务平台 `ascend-950pr-9579` 的 `examples/` 下不存在 `target="ascend"` 的示例（`examples/ascend` 目录不存在），ruleset 指向的 `kernel_template.py` 是唯一可运行骨架；API 存在性证据只能取外部 fork 源码。 | `examples/paged_cache_write/_paged_cache_write_kernel/DESIGN.md#3.4`（外部 fork basename 引用）；`.agents/platforms/toolchains/ascend/workflow.md` | ruleset 增列「本平台 `examples/` 覆盖空白」提示，并指定外部 fork 源码为允许的 API 证据源（含 basename 引用规则）。 | R |
| Design-flow | 平台 `programming_modes.migration_default = simd`，但本算子为逐元素 scatter + 负 slot 谓词跳过 + 运行期 token 边界，SIMD 寄存器路径需为每元素构造掩码；实际选定 SIMT，偏离默认值须在文档中论证并给出 SIMD 回退路径。 | `DESIGN.md#2.4` | `references/programming.md` 增「migration_default 可偏离的判定条件（谓词化稀疏写 / 动态边界 / 掩码不可表达）」小节。 | P |
| Design-flow | 迁移流程要求 §1.6.0 调研结论驱动 §0.6 重设计，但两份参考文档（`migration-analysis.md` / `algorithm-research.md`）未说明二者的先后顺序与回写关系；本次按「先调研 → 再重设计 → 调研结论回写 §1.6.0」执行。 | `DESIGN.md#0.6` 与 `#1.6.0` 交叉引用 | `migration-analysis.md` 增时序图（Phase M0 → R → M1，M1 引用 R 的选定算法）。 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| bf16/fp16 域「除法 → 乘倒数」不数值等价：`x * (1/s)` 在 scale=7.0 / 1.7 / 0.3 时出现 1 ulp 偏差（bf16 分别 60 / 30 / 2 个元素每 65536），逐元素 scale 除法应保留 `vdiv` 而非改写成 `vmul`。 | D | `examples/paged_cache_write/_paged_cache_write_kernel/verify_equiv.py`（X1 行）；`DESIGN.md#1.6.1` | `/home/l00970450/zuo/.environment/venv/bin/python examples/paged_cache_write/_paged_cache_write_kernel/verify_equiv.py`（CPU，torch 2.10.0+cpu） | tilelang 0.1.15+git0efd65c8；CPU 数值实验，无设备参与 | `pattern-library / numeric-pitfalls` |
| 依赖 cast 饱和不可靠：bf16 输入 4/16 个超出 ±448 的元素在转 fp8 时未饱和（fp16 2/16），fp8 存前必须显式 `clamp(±448)`；NaN 需单独保留。 | D | `DESIGN.md#1.6.1`（X3 行）与 `#3.3` 伪码；`verify_equiv.py` X3 行 | 同上脚本（X3 用例，16 元素角点集） | tilelang 0.1.15+git0efd65c8；CPU 数值实验 | `pattern-library / numeric-pitfalls` |
| 逐元素小算子被**向量发射开销**主导：qwen3-235b-fp8 形态计算 8.65 µs，而 1024 个逻辑块的发射开销 54.9 µs（6.3×）；此类算子在小规模时应与相邻算子融合，而非独立发射。 | D | `DESIGN.md#1.6.0` D-2 表（launch 0.5 µs/op × 逻辑块数） | 无设备复现；常数取 `CONST-vector-launch-overhead`（tilelang 0.1.2+3a214cde / CANN 8.5.0 / 910B2C / 2026-09-09） | 常数为他平台实测，本平台（Ascend950PR_9579 / CANN 9.1.0）**未实测** | `pattern-library / constants`（补 950PR 实测行） |
| GPU「fancy-index scatter + 布尔 mask」→ NPU「1-D 逻辑块 + lane 谓词」的映射模式：把行索引/列段折叠进 `T.Kernel(bx)`，`slot == -1` 用 lane 级谓词跳过而非「select 旧值」（后者需要读越界地址）。 | C | `DESIGN.md#0.6`（R1/R2 重设计项）与 `#1.6.2` 第 6 条 | none | tilelang 0.1.15+git0efd65c8 | `references/migration-analysis.md`（NPU 重设计模式库） |
| 单块工作量极小的批量写算子不应使用 `T.Persistent` 核内串行：每块仅 2048 元素（约 6 条向量 ALU 指令），串行化 73 个块会叠加地址解算开销并失去硬件自然均衡，且本路径无 UB 驻留压力。 | P | `DESIGN.md#5.5`（极大规模分波调度论证） | none | tilelang 0.1.15+git0efd65c8 | `references/decision-tree.md`（Persistent 适用条件） |

### Transferable Lessons（可迁移教训）

- 本平台无 `target="ascend"` 的 `examples/` 示例可引用；API 证据取 `/home/l00970450/zuo/tilelang-fork/tilelang/ascend/language/` 源码，引用时只写 fork 内相对路径（`tilelang/ascend/language/...`）或其 basename，**不得**拼出 `examples/` 或 `testing/` 前缀的假路径（`gate_lint` 会把它们当成仓库内引用并判失败）。
- 迁移分页写类算子时，把「行索引 × 列段」折叠成 1-D `T.Kernel(nblocks)` 的 `bx // nchunks`、`bx % nchunks`，可让 `nchunks` 成为编译期常量并消除块内串行；`cols` 是否整除 `block_cols` 必须在编译期判定后写死。
- 谓词化稀疏写（`slot == -1` 跳过）不要改写成「写时 select 旧值」——旧值读取地址是 `Pages[-1, col]`，属越界；用 lane 级 `if` 谓词或块级行 guard。
- fp8 输出路径：必须显式 `clamp(±448)`，且 NaN 用 `T.if_then_else(d == d, clamped, d)` 保留；仅当 `vmaxs/vmins` 被证实传播 NaN 时才可省略该保护（设计期按「未证实」处理，列入风险点）。
- 设计期所有计算量 / 访存 / 发射开销结论都须标注「未做设备验证」，并在常量引用处写清他平台来源与日期，避免下游把估算当实测。

## Stage 2 (review) Retrospective — 2026-11-23

检视结论：**不通过**（维度 0–7 pass，维度 8 fail；阻塞级 2 项 / 建议级 9 项）。范围限定 `REVIEW.md`；全程静态核查 + CPU-only 脚本，无设备命令，未修改 DESIGN.md / 源仓 / Stage 0 产物。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Review-flow | 维度 8 要求「判定依赖未实测常数时须有实验裁决计划」，但未定义三件套的最小字段（代表 shape / 指标与反解方法 / 数值阈值 / 回写路径），不同检视者对「定性判据是否够用」口径不一。 | `REVIEW.md` 问题 1；`DESIGN.md#1.6.3`、`#11.4` | 在 `references/algorithm-research.md §5` 与 review skill 维度 8 清单中写明实验裁决计划四要素及「缺阈值判阻塞」。 | R |
| Review-flow | 维度 7 只列「内部一致性」与「引用存在性」，未要求把数值型契约断言回查平台 manifest/spec，导致设计自述数字错误需检视者自行发现。 | `REVIEW.md` 问题 5（`DESIGN.md#9 R5` 写 `physical_tokens ≤ 4096`，manifest 实为 65536） | review skill 维度 7 增加一条「上限/形状/workload 数等数值断言须回查平台 manifest 或 spec」。 | R |
| Info-source | `design_calc_check.py` 在无 Cube 时对 `l0c_budget` 返回 `skip`，skill 未规定 skip 项的复核口径（人工确认不适用 vs 需补证据）。 | `REVIEW.md` 机械复核段第 2 条 | `references/quality-checklist.md` 增「机械检查 skip 项须写明人工判定依据与依据位置」。 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 设计文档引用的源仓内部符号必须回查源码：本任务 §0.3 的 `_compiled_fn` / `_compiled_config` 在源仓全库不存在，实际机制为基类 `Kernel.program` 缓存 + factory `@target_cache`。 | P | `REVIEW.md` 问题 3；`DESIGN.md#0.3` host 侧第 3 条 | `grep -rn "_compiled_fn\|_compiled_config" --include=*.py <TileOPs 源仓>` → 无匹配 | TileOPs 源仓（本任务 checkout）；无设备参与 | `references/migration-analysis.md`（§0 核对清单） |
| 带宽 / 复杂度复算须逐张量按自身 dtype 计字节宽度：`KScale`（float32）被按 8 B/元素计，使被淘汰基线的流量与下界偏高约 1.5 倍（prefill 134.2→约 100.8 MB）。 | P | `REVIEW.md` 问题 8；`DESIGN.md#1.6.0` R3 表 | none | tilelang 0.1.15+git0efd65c8；纯算术复算 | `references/algorithm-research.md`（R3 复算口径） |
| 决策依赖未实测写代价时，「显著低于」的定性判据不足以支撑主选/备选取舍：须给代表 shape + 指标与反解方法 + 数值阈值 + 回写路径，并优先评估按编译期特化 dtype 分派。 | P | `REVIEW.md` 问题 1；`DESIGN.md#1.6.3` 转向判据 | none | 常数缺 950PR 实测（设计自述） | `references/algorithm-research.md §5`（实验裁决模式） |
| 迁移设计的备选实现必须与主选网格同构并验算列覆盖：备选 B 漏 `nchunks` 折叠与 `cb*cols` 列偏移，`row_width ∈ {512,1024}` 时只写第 0 列段。 | C | `REVIEW.md` 问题 2；`DESIGN.md#3.3` B 路径（534-553） | 按 workload 最大 `row_width/cols = 8` 验算。 | tilelang 0.1.15+git0efd65c8 | `pattern-library / bottleneck-patterns`（备选路径结构核对） |

### Transferable Lessons（可迁移教训）

- 检视迁移设计 §0 时，源仓内部符号 / API 名必须回查源码：本函数 §0.3 的 `_compiled_fn` / `_compiled_config` 在源仓不存在，实际为 `Kernel.program(build, *facts)` + `@target_cache(maxsize=32)`。
- 数值型契约断言（`physical_tokens`、页数、workload 数）必须回查平台 manifest / spec：本任务 `cache_shape[0]=4096` 是 `num_pages`，`physical_tokens = 4096 × 16 = 65536`，设计曾写 `≤ 4096`。
- 带宽复算逐张量按 dtype 计宽（float32 scale 4 B、int64 索引 8 B、fp8 1 B）；整表统一按 8 B 会把被淘汰基线的下界抬高约 1.5 倍，虽不改结论但口径不可复算。
- 主选 / 备选取舍依赖未实测常数时，设计必须给「代表 shape + 指标与反解方法 + 数值阈值 + 回写路径」，否则按维度 8 判阻塞；先评估按编译期特化参数分派（本算子 `cache_dtype` 为编译期参数，可分派 16-bit 与 fp8 路径）。
- 检视备选路径代码时按 `max(row_width/cols)` 验算列覆盖：缺 `nchunks` 折叠 / `cb*cols` 偏移会在 `row_width > cols` 时静默漏写列段，此类错误不影响编译但直接造成精度失败。

## Stage 1 (design) Retrospective — revision 1 — 2026-11-23

路径 A 修订（Stage 2 检视不通过 → 只修 `REVIEW.md` 的两个阻塞项）。范围：`DESIGN.md` 的 §1.6.3 / §3.3 / §4.3–§4.5 / §3.2 / §3.4 / §9 R1,R2 / §10.4 / §11.3–§11.4 的对应段落；未改 §0 迁移分析、§1.6.0 调研结论、§1.6.1 等价优化（`verify_equiv.py` 结论原样保留）、§2 模式选型、§5 分核策略的结论，编程模式仍为 `simt`（`T.SimtVF(threads=128)`）。全程未编译、未运行 kernel、未改 `ASCEND_RT_VISIBLE_DEVICES`；仅执行 `statectl gate 1`、`design_calc_check.py` 与 grep/编辑类静态操作。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Design-flow | 修订模式的输入只有「旧 DESIGN + 检视意见」，缺少「本次修订会牵动哪些派生数字」的定位工具；删除 1 个 buffer 后必须人工 grep 出 8 处 UB 合计/占用率/规模推演并逐一同步，漏一处即内部不一致。 | `DESIGN.md#1.6.0 R3/R4`、`#3.5.1`、`#4.5`、`#5.3`、`#11.3`、`#11.4` 的 `12.06 KB → 12.0 KB` 联动 | design skill 增「结构性修订联动清单」：改动 buffer / tile / block 数后，强制重算并回改 UB、容量占用率、每块/总指令数、R3 表、附录代价表。 | R |
| Design-flow | 「实验裁决计划」在 skill 中只有字段名（代表 shape / 指标 / 阈值 / 回写），无模板；阈值的统计口径（噪声、变异系数）与反解式的未知量消元方式全靠自拟。 | `DESIGN.md#1.6.3 (a)–(e)`；`REVIEW.md` 问题 1 | `references/algorithm-research.md §5` 给出可复制模板：`shape 表 / L_P,L_B 定义 / 反解式 / 阈值 = f(测量噪声) / [DESIGN_LIMIT] 回写`。 | R |
| Constraint-doc | `gate_lint` 的 `S1-REF-PATHS` 会把带 `docs|examples|testing|.agents` 前缀的路径当仓库内引用校验存在性，写「未来将生成的 `perf_opt/...`」时必须用相对名，否则误报；该规则未在 design skill 中提示。 | `.agents/tools/gate_lint.py`；本稿 §1.6.3 (e) 写 `perf_opt/perf_feedback.md`（无前缀） | platform ruleset 或 `quality-checklist.md` 增「未来工件只能写相对名」一条。 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 主选/备选取舍的时延阈值应显式绑定测量噪声：`T1: L_P ≥ (1 + 2·CV + 5%)·L_B`，`CV` 为同 shape 重复测量变异系数；设计假设 5%，实测 >5% 时按 `1+2CV+0.05` 上修重判。定性表述「显著低于」不可作为裁决阈值。 | P | `DESIGN.md#1.6.3 (c)` | none | 常数缺 950PR 实测（设计自述） | `references/algorithm-research.md §5`（实验裁决模板） |
| 当两路径写字节数不同时，写带宽未知量可用「两类 shape 联立」消元：同 dtype shape 先解出单位字节常数代价 `Δc`（`Tr ∝ W`，带宽不可分），fp8 shape 再用 `Δc` 解 `BW_write_eff = ΔB_w / ((L_P−L_B) − W·Δc)`；符号矛盾或超出物理区间即判模型不匹配、只采信时延比。 | P | `DESIGN.md#1.6.3 (b)` R4 #2 | none | 同上 | `references/algorithm-research.md §5`、`_shared/standards/hardware-cost-model.md` |
| 小算子 UB 中转路径的「行级 slot」不需要 UB 暂存：`T.serial` 行循环内 slot 是该行标量，用寄存器承载即可（`T.copy` 的目标行由标量地址 `KPages[slot, ...]` 表达）；误分配会让 UB 预算虚高并在 `T.copy` 路径上多一次 store+load。 | P | `DESIGN.md#3.3`（`s_ub` 删除论证）、`#4.5` | none | tilelang 0.1.15+git0efd65c8 | `pattern-library / bottleneck-patterns`（UB 中转路径） |
| 备选路径改为与主选同构网格后，列覆盖应写成三命题并逐条验算：① `bx ↦ (bx // nchunks, bx % nchunks)` 是双射；② 列段并集 `⋃[cb·cols,(cb+1)·cols) = [0, nchunks·cols) = [0, row_width)`；③ 末段即 `cb = nchunks−1`，无列尾块。 | C | `DESIGN.md#3.3`「覆盖性与尾段论证」；`REVIEW.md` 问题 2 | 以 `row_width ∈ {512,1024}, cols=128 ⇒ nchunks ∈ {4,8}` 验算 | tilelang 0.1.15+git0efd65c8 | `pattern-library / bottleneck-patterns`（备选路径结构核对） |

### Transferable Lessons（可迁移教训）

- 修订「删除一个 buffer」类问题时，先确定它影响的所有派生数字（UB 合计、占用率、`block_rows` 放大后的规模推演、每块/总指令条数、R3/R4 代价表），再统一 grep 旧值替换；只改预算表会留下 8 处内部矛盾并直接被检视命中。
- 实验裁决计划的阈值必须写成可机械判定的不等式，并把测量噪声显式纳入系数（`1 + 2CV + 5%`），否则 Stage 4 仍无法收敛；回写路径要具体到 `[DESIGN_LIMIT]` 附录补记 + conductor 受控路由。
- 主选/备选网格必须同构：备选若另建 grid，极易漏 `nchunks` 折叠与 `cb·cols` 偏移，`row_width > cols` 时静默漏写列段；校验时按 `max(row_width/cols)` 的 `nchunks` 逐段验算覆盖。
- 依赖未实测写带宽的取舍，可用「两类 dtype shape 联立」把常数指令代价与写带宽两个未知量分离；若解出的带宽落在物理区间外，说明模型不匹配，此时只采信与模型无关的时延比。
- 迁移算子的备选路径同样要给出「UB 按 dtype 变体分别预算」（本算子 16-bit 2 buffer / fp8 4 buffer），并说明按编译期特化的 `cache_dtype` 分派不增加编译产物，避免下游误以为要维护两套 kernel 配置。

## Stage 2 (review) Retrospective — revision 1 复审 — 2026-11-23

检视结论：**通过**（维度 0–8 全 pass，阻塞 0 项，建议级 11 项；上轮 2 个阻塞项均闭环）。范围限定 `REVIEW.md` 与本章节；全程静态只读 + CPU-only 脚本（`verify_equiv.py`、`design_calc_check.py`、`statectl gate`、一次静态 `import tilelang.ascend.language`），无设备命令，未修改 `DESIGN.md` / 源仓 / Stage 0 产物。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Review-flow | 复审模板没有「上轮阻塞项闭环」的结构化载体：须在同一份 REVIEW.md 内给出「要求 → 证据行号 → 独立复核 → 判定」，本次全靠自拟小节。 | `REVIEW.md`「上轮阻塞项闭环核对」；`DESIGN.md#1.6.3`、`#3.3` | review skill 模板增「复审模式：逐条 上轮问题 → 判定 → 证据行号 → 独立复核」小节与字段。 | R |
| Review-flow | 维度 8 只要求阈值「可机械判定」，未要求检视者抽查阈值两侧的口径/量纲；本轮 T2 的 `tx_min` 缺聚合范围因子，阈值退化为恒真。 | `REVIEW.md` 建议 13；`DESIGN.md#1.6.3 (c)` | 维度 8 清单增「不等式两侧须同口径、量纲可复算，逐项写出分子/分母聚合范围」。 | R |
| Review-flow | 复审与原轮的建议项编号没有延续规则，重编号会让 Stage 1 无法对账。 | `REVIEW.md` 建议 3–13（沿用上一轮、新增追加尾号） | review skill 增「复审沿用原编号，新增项追加尾号」。 | R |
| Info-source | 平台 API 证据路径的基准未说明：`T.clamp` 在 `tilelang/language/customize.py:44`（相对 `tilelang/` 根）而非 `tilelang/ascend/language/`，按后者查找会误判「API 不存在」。 | `REVIEW.md` 维度 1；`tilelang/language/common.py` 114/260 行导出链 | quality-checklist 增「API 证据写相对 `tilelang/` 根的路径并注明导出链，最终以一次静态 `dir(T)` 确认」。 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 复审的「阻塞项闭环」应以可复算谓词判定而非「小节已存在」判定：每项固定 5 字段（要求 / 证据行号 / 独立复核 / 数字复算 / 结论）。 | P | `REVIEW.md`「上轮阻塞项闭环核对」 | none | tilelang 0.1.15+git0efd65c8 | `references/quality-checklist.md`（复审模式） |
| 阈值不等式必须显式声明聚合范围（per-kernel / per-block），否则分子分母口径不一致会让阈值退化为恒真/恒假：本任务 T2 的 `tx_min` 缺 `tokens`（kernel 口径）或 `block_rows`（块口径），`ratio_tx` 被放大约 4096 倍。 | D | `REVIEW.md` 建议 13；`DESIGN.md#1.6.3 (c)` | 以 `qwen3-235b-fp8`（tokens=4096, nchunks=4, cols=128, out_bytes=1）复算：现式 `tx_min=8` vs kernel 最小 32,768 | tilelang 0.1.15+git0efd65c8；纯算术 | `references/algorithm-research.md §5`（实验裁决模板） |
| 备选路径列覆盖可用三命题机械验收：`bx` 双射 / 段并集 `= [0, nchunks·cols)` / 末段即 `nchunks−1`；对 `row_width ∈ {128, 512, 1024, 96}` 逐值代入全部通过。 | C | `REVIEW.md`「上轮阻塞项闭环核对」问题 2；`DESIGN.md#3.3` 636 行 | 按 `nchunks = row_width/cols` 逐值枚举 | tilelang 0.1.15+git0efd65c8 | `pattern-library / bottleneck-patterns`（备选路径结构核对） |
| 结构性修订的数值传播可用「旧值全文零命中 + 新值逐处命中」双向 grep 验收：`12.06` 仅剩修订记录、`12352`/`49.4`/`4.75%`/`0:cols` 零命中，比逐节阅读更可靠。 | P | `DESIGN.md#4.5`、`#1.6.0 R3`、`#11.4`；`REVIEW.md` 维度 7 | `grep -n "12.06\|12352\|49.4\|4.75%\|0:cols" DESIGN.md` | tilelang 0.1.15+git0efd65c8 | `references/quality-checklist.md`（数值一致性） |

### Transferable Lessons（可迁移教训）

- 复审要区分「章节存在」与「内容可复算」：阈值、覆盖性、UB 预算都必须逐项代入数字复算，否则「已补小节」也会带着口径错误通过。
- 阈值不等式的两侧必须同口径同量纲：写 `tx_min` 这类归一化量时必须声明 per-kernel 还是 per-block，并检查因子（`tokens` / `block_rows`）是否齐全。
- 数值传播用双向 grep 验收（旧值零命中 + 新值逐处命中），并区分「修订记录中作为历史值保留」与「正文残留」。
- API 证据的路径基准要写清（相对 `tilelang/` 根），并以一次静态导入（`dir(T)`）作为最终确认，避免文件位置假设导致误判「API 不存在」。
- 机械检查的 skip 项（本轮 `l0c_budget`）必须在报告中写明人工判定依据与位置，否则 skip 会被误读为「未检查」。

## Stage 3 (develop) Retrospective — first_impl (attempt 1) — 2026-11-23

范围：`_paged_cache_write_kernel.py`（kernel + 内嵌 PyTorch golden + L0/L1/L2/boundary 四层套件 + main 入口），真机 ascend-950pr-9579（dav-3510，1 张 NPU，`ASCEND_RT_VISIBLE_DEVICES=0`），TileLang ascend 0.1.15 / CANN 9.2.1 / torch 2.13.0+cpu + torch_npu 2.13.0.rc1。结论 **`[PRECISION_PASS]`**：L0 10/10、L1 2/2、L2 3/3、boundary 10/10，max_diff = 0（逐字节比对；唯一规范化处理是 e4m3 NaN 的符号位，见 CONST-950-fp8-nan-sign）。

实现与 `DESIGN.md#3.3` 的偏差 3 项，全部由实测驱动、语义不变：

1. fp8 路径把 K/V 拆成两个 `T.Parallel` 循环（DESIGN 为单循环双 store）：单循环双 fp8 store 会让向量化放弃并退化为**标量 e4m3 store，写 0**（`repro/TRAP-950-fp8-scalar-store.py`）。
2. fp8 路径新增编译期 guard `block_rows·cols/threads ≥ 2`：每个线程少于 2 个元素时同样退化为标量 e4m3 store 并写 0；同时这也约束了 Stage 4 的 tile 搜索下界（`threads=128` 时 `block_rows·cols ≥ 256`）。
3. 精度比对对 e4m3 的 NaN 字节做符号位规范化（0xFF→0x7F）：设备转换把 NaN 统一写 0x7F，torch 保留输入 NaN 的符号位（`-nan → 0xFF`）；算子契约按 `equal_nan` 比较，非 NaN 字节仍逐字节比对。`DESIGN.md#8.3` 的 L0-8/9/10 是 host/wrapper 级用例（契约拒绝、dispatch、roofline 记账），kernel-only 文件中以 `_case_contract` / `_case_dispatch` / `_case_written_roofline` 三项等价复现（后者额外用「未被命中行逐字节不变」证明没有多余写）。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Develop-flow | 平台 `execution.precision_command` 写死 `/usr/local/python3.12.16/bin/python3`，本机不存在该解释器；站点脚本 `env-950pr.sh` 与 dispatch prompt 才给出可用 venv。按机器可读字段执行会直接失败。 | `statectl platform --task-dir …` 的 `execution.precision_command`；`.agents/platforms/toolchains/ascend/env-950pr.sh` 的 `TILEOPS_950PR_VENV` | 让 `execution` 模板接受 `${TILELANG_PYTHON}` 或直接生成站点解析后的解释器路径，和 `env-*.sh` 保持单一来源。 | R |
| Develop-flow | 门禁表只有「编译/运行/精度/占位符」四类，没有「编译通过但生成码静默算错」这一类；本任务的标量 e4m3 store 编译、运行、退出码全部正常，只写 0，只有逐字节精度断言能兜住。 | `repro/TRAP-950-fp8-scalar-store.py`；`_paged_cache_write_kernel.py` 的 fp8 双循环与 `per_thread` guard | develop skill Phase 4 增「新 dtype 目标写路径首验：dump 生成码确认 store 宽度（`fp8_e4_4_t` vs 标量 `fp8_e4_t`），标量宽度视为不可信」。 | R |
| Develop-flow | 迁移 harness 任务的 kernel-only 文件没有 wrapper，但 DESIGN 的 L0 计划里混有 wrapper/host 级用例（契约拒绝、dispatch、roofline），skill 未说明如何落地。 | `DESIGN.md#8.3` L0-8/9/10；`_paged_cache_write_kernel.py` 的 `_case_contract`/`_case_dispatch`/`_case_written_roofline` | develop skill 增「harness 迁移：kernel-only 层只保留设备可执行的语义断言，host 契约类用例用 in-scope 等价项替代并在返回中披露」。 | R |
| Info-source | 上游 Stage 0 产物里写着「Known site deviation: NaN is not carried through」，与实测相反（`T.clamp` 完全传播 NaN，只有 `T.min/T.max` 会丢）。上游工件的行为断言没有版本/证据戳，容易被后续阶段当事实继承。 | `examples/TileOPs/tileops/kernels/attention/paged_cache_write/_paged_k_v_cache_write_fwd_kernels.py` 注释；`repro/CONST-950-fp8-nan-sign.py` | 迁移脚手架产物的「site deviation」注释须带实测脚本路径与工具链戳，或标 `unverified`。 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 标量 e4m3 GM store 静默写 0：单线程单元素（`block_rows·cols == threads`）或同一 `T.Parallel` 体内两个 fp8 store（向量化放弃 → 标量回退）都会把结果写成 0；改成「每个 pool 一个 `T.Parallel` 循环」后逐字节正确（`kp[0,0]` 由 0.0 恢复为 1.5）。 | D | `_paged_cache_write_kernel.py#_build`（fp8 双循环 + `per_thread` guard）；生成码 `*(__gm__ fp8_e4_4_t*)` vs `KPages[...] = tl::float_e4m3_t` | `repro/TRAP-950-fp8-scalar-store.py`（断言四条路径） | TileLang 0.1.15；CANN 9.2.1；ascend-950pr-9579 / dav-3510；torch 2.13.0+cpu, torch_npu 2.13.0.rc1 | `pattern-library / numeric-pitfalls` |
| e4m3 转换的 NaN 语义：`T.clamp` 传播 NaN（+nan 与 -nan 均写 0x7F），`T.min/T.max` 组合把 NaN 变成 -448；torch 保留 NaN 符号位（`-nan → 0xFF`）。逐位契约下须按 `equal_nan` 比对并规范化 NaN 符号位，非 NaN 字节（±inf 饱和、ties、±0）完全一致。 | D | `_paged_cache_write_kernel.py#_nan_canonical_bytes`；L0-5 / B-6 用例 | `repro/CONST-950-fp8-nan-sign.py` | 同上 | `pattern-library / numeric-pitfalls` |
| DESIGN 的 O3「slot 整块载入」在该后端不成立：生成码在向量块内重新读 `Slots[...]` 4 次（guard、上界检查、store 地址各一次），两个循环下每元素读 2 遍；实测 payload 有效带宽 16-bit 8192×1024 为 67 MB / 0.417 ms ≈ 161 GB/s，fp8 4096×512 为 12.6 MB / 0.234 ms ≈ 54 GB/s，量级与「按元素读 Slots（8 B/元素）」一致，而非 O3 的 0.52 MB。Stage 4 的杠杆应改为行串行 + slot 常驻寄存器（或 slot 预载 UB，即 DESIGN 路径 B）。 | D | `_paged_cache_write_kernel.py` fp8/16-bit 循环；L2-1/L2-2 实测 | `probe10`（`Slots[` 在 4 元素向量块内出现 4 次）＋ L2 计时 | 同上 | `pattern-library / bottlenecks` |
| 动态 extent 的零 token 调用必须在 host 侧跳过：直接 launch 会报 `Check failed: (extent > 0) is false: Ascend launch grid must be positive`，而不是静默不写。 | D | `_paged_cache_write_kernel.py#_write`（`k.shape[0] == 0` 早退）；L0-7 | `probe11`（零 token launch 直接抛 RuntimeError） | 同上 | `pattern-library / kernel-patterns` |
| 原地写算子的精确验证套路：`assert_close(atol=0, rtol=0, equal_nan=True)` + 原始字节比对（fp8 需先规范化 NaN 符号位）+ 「未命中行逐字节不变」集合断言，三件套一起才能同时证明「写对了」与「没多写」。fp8 的 `isclose` 在本栈只能在 host 侧做（`aclnnIsClose` error 161002）。 | P | `_paged_cache_write_kernel.py#_assert_exact`、`#_case_written_roofline` | `python _paged_cache_write_kernel.py --level L0` | 同上 | `pattern-library / testing-patterns` |
| 同一 kernel 内多个目标 dtype 的写路径要分别验证向量化形态：16-bit 双 store 单循环完全正常（L0-1/L1-2 通过），fp8 双 store 单循环写 0；差异只由目标 dtype 决定，不能由「同结构 16-bit 已验证」外推。 | P | L0-1（16-bit 双 store 单循环 PASS）vs `repro/TRAP-950-fp8-scalar-store.py` | 两个用例各自的生成码 | 同上 | `references/quality-checklist.md`（dtype 维度的验证独立性） |

### Transferable Lessons（可迁移教训）

- 目标 dtype 是 fp8（e4m3）的 GM 直写路径，每个 `T.Parallel` 循环体内只保留一个 fp8 store；同一体内出现两个 fp8 store、或每线程元素数少于 2，都会退化为标量 e4m3 store 并静默写 0（编译、运行、退出码全部正常）。
- 判断 fp8 写路径是否安全，看生成码里的 store 宽度：`*(__gm__ fp8_e4_4_t*)`/`fp8_e4_2_t` 是向量化形态（可信），`KPages[...] = tl::float_e4m3_t(...)` 是标量形态（本后端写 0）。
- 饱和 + NaN 保持要用 `T.clamp`（设备上展开为 NaN 选择 + min/max），不要用 `T.min/T.max` 组合：后者把 NaN 变成 `-448`；只靠 `T.cast` 不饱和，`±inf/1000 → NaN`。
- e4m3 的 NaN 符号位在设备上被规范化成 0x7F，而 torch 保留输入符号（`-nan → 0xFF`）：逐字节比对前必须规范化 NaN 符号位，否则会在纯 NaN 用例上误报 1 字节差异；其余字节（±inf 饱和、ties、±0）必须严格相等。
- 动态 extent 的 kernel 不要在零 token 时 launch（`Ascend launch grid must be positive`）；host 侧早退是契约的一部分，测试套件也必须走同一条路径。
- 同一 `prim_func` 里「K/V 共用一个循环体」在 16-bit 下可省一半 slot 载入，但为换取 fp8 正确性必须拆成两个循环，代价是 slot 载入翻倍；性能收益要靠把 slot 提到寄存器/UB（行串行或路径 B），而当前后端不会自动提升 `Slots[token]`（生成码按元素重读 4 次）。
- 上游阶段（脚手架/设计）写在注释里的行为断言（如「NaN 不保留」）在实测前一律按 `unverified` 处理：本次该断言与实测相反，照抄会引入多余的 `if_then_else` 分支并掩盖真正的 fp8 store 缺陷。
- fp8 张量的 `torch.testing.assert_close` 在设备侧不可用（`aclnnIsClose` 161002），精度比对统一搬回 host 做，对 kernel 语义没有影响。
