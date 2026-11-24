# RETROSPECTIVE — moe_permute_align / _make_align_kernel（Stage 1，first_design，迁移）

## Run Meta

| 字段 | 值 |
|---|---|
| 任务目录 | `examples/moe_permute_align/_make_align_kernel` |
| 模式 | first_design（迁移，harness 子模式） |
| 平台 | `ascend-950pr-9579` / toolchain `ascend`（TileLang 0.1.15 + CANN 9.2.1，`target="ascend"`） |
| 源算子 | `/workspace/TileOPs/src/tileops/_kernels/moe/permute_align.py` 中的 `_make_align_kernel` |
| 产出 | `DESIGN.md`、`verify_equiv.py`（已执行，全 EQUIV_PASS）、本文件 |
| 日期 | 2026-11-23 |

## Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| gate lint | `S1-REF-PATHS` 把 `path::symbol` 整体当路径校验，导致「源文件+符号」的写法必然失败 | 首轮 `statectl gate 1` 报 `examples/TileOPs/tileops/workloads/moe.py::ref_permute_align` 不存在；改为「`path` 中的 `symbol`」后通过 | `gate_lint.py` 的 `REF_PATH_RE` 在匹配后剥掉 `::symbol` 后缀再判存在；或在 design 模板中统一写作 `path` + 反引号符号 | R |
| 执行上下文 | 平台 `execution.precision_command` 的 `/usr/local/python3.12.16/bin/python3` 在本机不存在，实际解释器是 `/workspace/venv-ascend950/bin/python` | 任务上下文实测 + §4.7 记录 | `statectl platform` 增加「本机解释器存在性校验」并把可用路径写入 `execution`；或 design 模板要求显式记录解释器实测 | R |
| 平台能力 | 跨 block barrier / 跨核 flag 未认证，直接影响算法结构选择（决定单 launch 还是两次 launch） | `references/programming.md` 同步章节 + `knowledge/execution-patterns.md` PL-950-atomic-splitk 明确「不是全局 barrier 示例」 | 在 `programming.md` 增加一节「无跨 block barrier 时的三种落地形态（单核 / 多 launch / 冗余只读 + 分片写）」 | P |
| examples 取证陷阱 | `examples/elementwise/atomic_add.py` 用 `target="npuir"` + `T.alloc_ub`，与本工具链（`target="ascend"`、`T.alloc_shared`）不是同一后端，容易被误当作 NPU 原子证据 | 该文件 import 与 `is_npu=True`；本平台 ruleset 明确仓库根 `tilelang/language/` 属另一工具链 | `info-sources.md` 增加「`examples/` 内 target 标签核对」检查项（本次已按此排除该证据） | R |

## Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 950PR 物理核数实测：cube=28 / vector=56（`Ascend950PR_9579`） | D | `torch.npu.get_device_properties(0)` → `cube_core_num=28, vector_core_num=56`（2026-11-23 本机）；平台 `hardware.observed_cores` 同值 | `python -c "import torch,torch_npu; p=torch.npu.get_device_properties(0); print(p.name,p.cube_core_num,p.vector_core_num)"` | ascend / CANN 9.2.1 / TileLang 0.1.15 | pattern-library `constants.md` 新增 950PR 条目（现表全部为 910B2C，设计期 roofline 只能外推） |
| 无 scatter-add 平台的直方图模式：expert-lane 比较累加 + 局部直方图 GM 归并 | P | 本设计 §1.6.0 C2 / §0.6 R2；`T.atomic_add` 在 ascend 仅在 GM 侧有 FP32 证据（PL-950-atomic-splitk），UB scatter-add 无证据 | `examples/moe_permute_align/_make_align_kernel/DESIGN.md` §3.3 伪代码；待 Stage 3 实测 | ascend / TileLang 0.1.15 | `knowledge/operator-patterns.md` 新条目（索引/直方图类首个案例） |
| 无跨 block barrier 时的「冗余只读 + 分片写」解耦模式 | P | 本设计 §0.6 R1 / §5.5：launch-2 每核读同一份 `partial` 并冗余求和（整数确定性 ⇒ 逐位相同），写域分片互不相交 | 同上 §3.3 `align_kernel` | ascend / TileLang 0.1.15 | `references/programming.md`（同步章节）+ `migration-analysis.md` §6.2 跨阶段融合模式 |
| `expert_ids` 的 rank-count 恒等式：`owner(b) = #{e: cumsum[e] ≤ b·B} − 1` | P | §1.6.1 A2 的等价性证明 + `verify_equiv.py` 3,240 例 EQUIV_PASS（含零计数 expert、E=1、B=1） | `examples/moe_permute_align/_make_align_kernel/verify_equiv.py` | torch CPU（与工具链无关） | `algorithm-candidates.md` 新增 MoE align 条目（R1 候选：变长 scatter vs rank-count vs 二分） |
| 迁移任务必须在 `examples/` 内核对后端标签（`target="npuir"` ≠ `target="ascend"`） | R | 见 Skill Flow Issues 第 4 行 | — | — | `info-sources.md` 强制步骤 0 增补检查项 |

## Transferable Lessons

1. **「NPU 没有 X」要落到结构，而不是落到 API 名**：本次三个关键判定（无 UB scatter-add、无 warp shuffle、
   无跨 block barrier）都改变了算法结构，而不是换个函数调用。设计文档里把它们写成「能力缺失 → 算法替代 →
   等价性论证」的三段式，Stage 3 才有可执行依据。
2. **整数算子是无风险并行的礼物**：`cnt/cumsum/expert_ids` 全为整数，任何并行分组、任何累加顺序都逐位等价，
   因此「多核冗余计算同一结果」「log 深度扫描替代串行链」都不需要精度论证，只需**正确性**论证（含零计数、
   `numel=0` 等边界）。这让 §1.6.1 的机器验证可以做到 `max ulp = 0`。
3. **代价换墙钟要算清是哪一项**：`expert_ids` 从「每 block 1 次写」换成「每 block E 次比较」表面看更贵
   （342K lane-ops vs 891 次写），但把数据相关 trip count 换成了可向量化/可分片的规整比较，墙钟反而更短。
   设计文档必须把「lane-ops」与「向量指令数 / 核」分开写，否则容易自证错误。
4. **离 roofline 的距离要写进设计**：本算子估算下界 3.4 µs vs 流量下界 0.21 µs（16x），根因是「无 scatter-add」
   而非 tiling 不好；把它写成 §9.1 风险项 1 + Stage 4 裁决计划，避免 Stage 4 把结构性代价误当参数问题反复调参。
5. **单次 launch 的契约会反向约束算法**：wrapper 的 `align(flat, sorted, expert_ids, npost, cumsum)` 只接一个
   可调用对象，逼出「组合可调用对象 + 持久 scratch」的形态；若没在 §4.7 写清，Stage 3 会在编译期才发现
   两个 `@T.prim_func` 无法塞进一个 launch。

## Stage 2 (review) Retrospective — design_review — 2026-11-23

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Review-flow | 维度 1（API 可行性）只要求"源码/文档/examples 佐证符号存在"，无法判定**伪代码写法**能否构建：本次两种写法在语言文档里完全合法，只有编译才暴露（`T.alloc_var(...)+acc[0]`、`@T.prim_func` 内 `while` 展开） | `REVIEW.md#机械复核3` 探针 K4/K2 = COMPILE_FAIL；DESIGN.md#§3.3 L660/L676 | 在 `tilelang-design-review` skill Phase 1 增加一步"compile-only 探针"（`ASCEND_RT_VISIBLE_DEVICES=""` + `tilelang.compile`，每 kernel 单独进程），至少覆盖伪代码里每类非平凡写法 | R |
| Info-source | 反向陷阱：用 C++ codegen 源码的"显式分支列表"否证 API 可用性会误判（`codegen_ascend.cc` 的 copy 分支只有 6 条，`ubuf→ubuf` 不在其中，但 UB→UB `T.copy` 实测可编译） | `REVIEW.md#机械复核3` K1/K5/K14 = COMPILE_OK；`src/target/codegen_ascend.cc` L410-471 | 在 `info-sources.md` 写明：API 可用性的最终裁决是编译探针，codegen/符号表只能作为"存在性线索" | R |
| Info-source | pattern-library `constants.md` 全部为 910B2C，无 950PR 条目 ⇒ 950PR 设计期 roofline 只能跨型号外推，检视只能复算其算术、无法独立验证代价常数本身 | DESIGN.md#§1.6.0「设计期 roofline」；`pattern-library/constants.md` 12 条均 `device: 910B2C` | 平台 `knowledge` 增补 950PR 的 fabric 带宽 / MTE2 退化 / 向量发射率实测条目（当前只存在于设计外推里） | P |
| Review-flow | 退化输入（`numel=0`）与补位（`E_pad > E`）两类缺陷在本流程里没有机械拦截位：`design_calc_check.py` 不查伪代码语义，`verify_equiv.py` 只验数学式不验 buffer 布局 | DESIGN.md#§3.3 L649/L693-699；REVIEW.md#问题 1/问题 2 | 在 `tilelang-op-design` 的 quality-checklist 增补"退化输入写回路径在守卫外""以 `align_up` 维度作循环上界时被读区域必须全写"两条自检 | P |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| ascend 前端「可构建形式」实测清单：UB→UB `T.copy`（含偏移 region）/ `T.fill`(UB) / `T.SimtVF(threads=384,448)` / `T.alloc_local([1])` 每 lane 累加 **可编译**；`T.alloc_var(...)[0]`（`TypeError: 'BufferLoad' object is not subscriptable`）与 `@T.prim_func` 内 python `while`（`RuntimeError: Infinite while loop detected`）**不可构建** | D | REVIEW.md#机械复核3 探针 K1/K3/K5/K7/K8/K13/K14 vs K2/K4 | 任务目录下 `source .agents/platforms/toolchains/ascend/env-950pr.sh` + `ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python <probe>.py`（探针源码内联于 REVIEW.md#机械复核3，可直接复制成独立脚本） | TileLang 0.1.15（venv-ascend950）+ CANN 9.2.1 / 950PR 编译期（无设备计算） | `.agents/platforms/toolchains/ascend/references/programming.md`（新增「可构建形式清单」一节） |
| 检视阶段的 compile-only 探针法：逐 kernel 单进程 + `ASCEND_RT_VISIBLE_DEVICES=""`，可在单卡被占用时仍裁决 API/写法可行性（本机 13 个探针 ≈2 min） | P | REVIEW.md#机械复核3 | 同上 | — | `tilelang-design-review` SKILL.md（Phase 1 新步骤） |
| rank-count / 补位类变换的机器验证须带 `E_pad > E` 覆盖：数学等价（`cumsum[:E]`）通过 ≠ 实现等价（读 `[0,E_pad)`） | C | REVIEW.md#问题 1；`verify_equiv.py#opt_align`（用 `cumsum[:E]` 求和） | `examples/moe_permute_align/_make_align_kernel/verify_equiv.py` | torch CPU（与工具链无关） | `tilelang-op-design` references/algorithm-research.md（§1.6.1 机器验证的覆盖要求） |

### Transferable Lessons（可迁移教训）

- 伪代码的"可构建形式"必须按本工具链实测，不能按 API 名或语言文档推断：每 lane 私有累加器用
  `T.alloc_local([1],"int32")`（`T.alloc_var` 返回标量，不可下标）；构建期展开禁用 `while`，用
  `for d in range(<常量>)`。
- UB→UB `T.copy`（含带偏移 region）、`T.fill`（UB 常量填充）、`T.SimtVF(threads=384/448)` 在本工具链**可用**；
  不要仅凭 codegen 的显式 copy 分支列表否证——以编译探针为准。
- 凡以 `align_up(N, 64)` 作 lane 数/循环上界，被读的 UB 区域必须整段写入（补位 lane 也要显式赋值），
  否则拿到的是 UB 残留值；数学等价验证（只看 `[0,N)`）覆盖不到这类缺陷。
- 退化输入（如 `numel=0`）的常量写回必须放在"数据量 > 0"守卫之外：构建期路径判定会把退化输入路由到
  最小工作量的路径，而最小路径的守卫最容易顺手包住全部写回。

## Stage 1 (revision 1) Retrospective — designer — 2026-11-23

### Run Meta

| 字段 | 值 |
|---|---|
| 任务目录 | `examples/moe_permute_align/_make_align_kernel` |
| 模式 | revision 1（路径 A：Stage 2 检视不通过） |
| 输入 | `REVIEW.md`（3 阻塞 + 3 建议）、`history_version/design_v0.md` |
| 产出 | `DESIGN.md`（v1，§11 逐项闭环）、`verify_equiv.py`（9 行全 PASS）、`history_version/design_probe_rev1_compile.py` |
| 门禁 | `statectl gate 1` pass（无 failure / 无 warning）；`design_calc_check.py` r3_metrics pass |

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| 设计流程 | 「语义正确性守卫」在 §1.6.1 里没有位置：v0 把补位 lane 与零输入守卫当成实现细节，导致同一份文档里 §4.3（`E+1` 有效）与 §3.3（读 `E_pad`）自相矛盾而无人拦截 | `REVIEW.md#问题1/#问题2`；v1 §1.6.1 A5 现已成为一个带四要素的采纳项 | `tilelang-op-design` 的 Phase 2 第 1 步表格增加一行「正确性守卫（未初始化读 / 退化输入写回）」，要求与优化项同等待遇（四要素 + 机器验证行） | R |
| 验证脚本 | `verify_equiv.py` 只验「数学式在逻辑长度上的等价」，天然覆盖不到 lane 几何（`align_up`）与 guard 范围两类缺陷；检视才发现 | v0 的 `opt_align` 用 `cumsum[:E]`；v1 新增 `opt_align_padded`（`E_pad` lane + 补位语义）与「UB 垃圾」反例对照 | 在 design skill Phase 2 第 5 步写明：**机器验证必须按实现的 buffer 形状（含 `align_up` 补位）建模**，并至少给一条反例对照（证明守卫承重） | R |
| roofline | R3/roofline 的数字没有"可复算"锚点：v0 的 3,500 / 2.4M / 1.0 µs 全是从别的行抄来的派生值，漏乘后无人发现 | `REVIEW.md#问题3`；v1 把口径写成 `2·ceil(E/V)`/token 并在四处（R3 / roofline / §1.6.3 / §9.1）交叉引用 | `hardware-cost-model.md` 增加「每 token 指令数必须先定义一次再派生」的检查项；`design_calc_check.py` 增加"派生数字可复算"校验（当前只查四口径存在性） | P |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 「补位 lane 未初始化」缺陷的机器化拦截法：在等价验证脚本里以 `E_pad = align_up(E,64)` 的 lane 几何重跑，并注入两种 UB 垃圾值做反例对照（本例 1,600/1,800 个 `E_pad>E` 用例会出错） | D | `verify_equiv.py` 的 `opt_align_padded(..., garbage=)` + 输出行 `A5 discrimination control`；DESIGN.md#§1.6.1 A5 | `cd examples/moe_permute_align/_make_align_kernel && ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python verify_equiv.py`（19 s，torch CPU） | torch CPU（与工具链无关） | `tilelang-op-design` references/algorithm-research.md §1.6.1 验证覆盖要求 |
| 退化输入（`numel=0`）守卫的收窄范式：守卫只包「数据依赖的 GM 读」，契约必需的写回一律无条件；构建期 `if numel>0:` 只折叠读循环 | P | DESIGN.md#§3.3 路径 S / §11 P2；探针 `S0`（`numel=0`）COMPILE_OK；verify_equiv L0 地标 `(0,8,16)` MATCH | `history_version/design_probe_rev1_compile.py S0` | TileLang 0.1.15 + CANN 9.2.1 / 950PR 编译期 | `knowledge/operator-patterns.md`（边界/退化输入条目） |
| 修订期编译探针应作为独立工件落盘（`history_version/design_probe_*.py`），供 Stage 2 复跑；检视期探针只内联在 REVIEW.md 里，复现要手工抄 | P | 本任务 `history_version/design_probe_rev1_compile.py`（S33/S0/L384，单例 ~4 s）；REVIEW.md#机械复核3 的探针源码为内联文本 | 同上脚本 | — | `tilelang-op-design` SKILL Phase 2 D-3 小节 + `tilelang-design-review` Phase 1 |

### Transferable Lessons（可迁移教训）

- **数学等价 ≠ 实现等价**：等价性验证必须在「实现的 buffer 形状」上做（`align_up` 后的 lane 数、
  补位 lane 的取值），并给一条"反例对照"证明守卫是承重的——否则验证通过只是证明了另一个函数。
- **派生数字必须留可复算锚点**：一个 `2·ceil(E/V)` 的定义写一次，四处引用；任何一处改动都能被其余三处交叉校验。
  漏乘 ×2 这类错误在文档里比在代码里更难发现。
- **退化输入是路径选择器的盲区**：路径判定式为「工作量 ≤ 阈值 ⇒ 走最小路径」，而最小路径的守卫最容易被写成
  「包住一切」；写守卫时先问「这个值在零输入下是否仍有契约值」。
- **`if <特化常量>:` 在 tracing 期是免费的**：把守卫换成构建期 Python 分支，既消除运行期分支，
  又让"只包 GM 读"这种细粒度收窄变得自然。

## Stage 2 (review) Retrospective — design_review round 2（v1 复检） — 2026-11-23

### Run Meta

| 字段 | 值 |
|---|---|
| 任务目录 | `examples/moe_permute_align/_make_align_kernel` |
| 模式 | review 第 2 轮（对象 = revision 1 的 v1，要求全 9 维度重做） |
| 输入 | `DESIGN.md`(v1)、`history_version/design_v0.md`、源算子 472 行、`verify_equiv.py`、`history_version/design_probe_rev1_compile.py` |
| 产出 | `REVIEW.md`（结论：通过，0 阻塞 / 6 建议）、本章节 |
| 机械复核 | `verify_equiv.py` 重跑 18.3 s（9 行 × 3,240 例全 EQUIV_PASS，与内嵌表逐行一致）；`design_calc_check.py` 3 skip + `r3_metrics` pass（skip 项人工复算）；`design_probe_rev1_compile.py` 复跑 S33/S0/L384 全 COMPILE_OK |

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Review-flow | 机械复核（D-1/D-3）只比对「脚本输出 vs 文档内嵌表」的文本一致性，**不校验设计期探针是否逐字镜像 DESIGN 的公式**：本轮探针给 `CHUNK` 加了 `max(64, …)`，而 §3.3/§5.2 写的是 `min(1024, align_up(numel,64))`（`numel=0` 时为 0）。于是「S0 COMPILE_OK」并不字面覆盖文档公式，检视只能另做补充探针 | `history_version/design_probe_rev1_compile.py` L38 vs `DESIGN.md` §3.3 L703 / §5.2；`REVIEW.md#维度7` 建议级观察 1 | 在 `tilelang-design-review` Phase 1 第 7 步增加「探针常量 ↔ 文档常量逐项对照」（探针须引用 DESIGN 的常量表，不得另立公式） | R |
| Review-flow | 机器验证脚本的**打印列**不等于数据：`verify_equiv.py` 的 `violation` 列是字面量 `'0.000000'`，与 `bad` 计数无关。若检视只搬运行文本，会把"违反率 0"当成独立证据 | `verify_equiv.py` L373（`{'0.000000':>10s}`）; `REVIEW.md#机械复核1` 的观察 | 在 review skill 的 D-1 步骤写明「只采信可由数据复算的列（verdict / max\|d\| / max ulp），字面量列须回读脚本源码」 | R |
| Review-flow | 维度 0/7 的清单没有「**冻结 wrapper 调用契约 + 持久 scratch**」这一项：本设计路径 L 返回复合可调用对象并持有 `partial`，是否可接受要打开 `kernel_base.py::program` 与 wrapper `forward` 才能判定（结论：可接受），且 `partial` 的设备/分配时机仍缺说明 | `DESIGN.md` §4.7 / §9.1 风险 4-5；`tileops/_kernels/kernel_base.py` L390-393；`examples/TileOPs/tileops/kernels/moe/moe_permute_align/moe_permute_align.py` forward | 在 `tilelang-design-review` 维度 0 的「设计与决策一致性」增加子项「返回对象形态是否满足冻结 wrapper 的调用契约；持久 scratch 的设备/生命周期是否写明」 | R |
| Info-source | 平台 `execution.precision_command` 的解释器**存在但不可用**（缺 `tqdm`），与上一轮复盘记录的"不存在"不符；两个阶段都靠"实测替换"绕过，说明 execution 上下文没有可用性校验 | `/usr/local/python3.12.16/bin/python3 -c "import tilelang"` → `ModuleNotFoundError: No module named 'tqdm'`；`/workspace/venv-ascend950/bin/python` → `IMPORT_OK 0.1.15 2.13.0+cpu` | `statectl platform` 生成 `execution` 时对 `precision_command` 的解释器做一次 import 可用性探测，把可用解释器写回该字段 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 零长度 UB 分配在 ascend 前端**可构建**：`T.alloc_shared((0,), "int32")`（`numel=0` 时 `CHUNK=0` 的字面形态）COMPILE_OK，与 `max(64, …)` 形态等价可编译 | D | 本轮补充探针（`/tmp/opencode/probe0/p_chunk0.py`，两种 `chunk` 形态各一例，均 `COMPILE_OK`）；`REVIEW.md#维度7` | 任务目录内 `source env-950pr.sh` + `ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python <probe>`，`tilelang.compile(build(0,8,16,mode), target="ascend", out_idx=-1)` | TileLang 0.1.15（venv-ascend950）+ CANN 9.2.1 / 950PR 编译期（无设备计算） | `.agents/platforms/toolchains/ascend/references/programming.md`（可构建形式清单补"零长度 alloc"一条） |
| 本机解释器事实校正：`/usr/local/python3.12.16/bin/python3` **存在**但缺依赖（`tqdm`）⇒ 平台 `execution.precision_command` 不可直接执行；`/workspace/venv-ascend950/bin/python` 可用（tilelang 0.1.15 / torch 2.13.0+cpu / torch_npu 2.13.0rc1） | D | 本轮两次 import 实测；`REVIEW.md#检视元信息` 执行上下文行 | `python3 -c "import tilelang"`（系统解释器）与 `/workspace/venv-ascend950/bin/python -c "import tilelang, torch, torch_npu"` | CANN 9.2.1 / 950PR 宿主 | 平台 `execution` 生成逻辑 + `workflow.md` 执行上下文一节 |
| 检视「反例对照/区分度」类证据的定量复算法：按脚本自身的几何集合复算**期望命中数**（本轮 `E_pad>E` = 5 种 E × 8 个 B × 9 个 numel × 5 分布 = 1,800，扣掉 `total=0` 的 200 例 ⇒ 应为 1,600，与脚本输出 `1600/1800` 一致），可低成本判定"对照是真跑出来的"而非叙述 | P | `verify_equiv.py` 的 `A5 discrimination control` 行；`REVIEW.md#机械复核1` 的 ①②③ 独立核对 | `cd examples/moe_permute_align/_make_align_kernel && /workspace/venv-ascend950/bin/python verify_equiv.py`（18.3 s） | torch CPU（与工具链无关） | `tilelang-design-review` SKILL Phase 1 第 7 步（负向/区分度证据的复核方法） |

### Transferable Lessons（可迁移教训）

- **探针与文档必须共用同一组常量**：设计期编译探针是"写法可构建"的唯一证据，一旦它在公式上偷偷加保护项
  （如 `max(64, …)`），检视侧的"COMPILE_OK"就不再覆盖文档里那行字。检视时逐项比对探针常量与 DESIGN 常量，
  差异处要么补探针，要么改文档。
- **机器证据要看生成它的代码，不要只看它的打印**：验证脚本的列可能是字面量；采信前确认该列由数据算出
  （`max|d|`、verdict 类），否则"违反率 0"只是排版。
- **"反例对照"的价值可用算术复核**：按脚本的几何集合算出"应当有多少例能区分"，与实际输出比对——数目对不上
  说明对照没跑或参数被改过；数目对上则说明该守卫确实承重。成本远低于重读全部脚本。
- **单程序形态不是硬约束，"工厂返回什么"由 wrapper 决定**：多 launch 的组合可调用对象在本项目 wrapper 里
  合法（`program()` 只包装 `JITKernel`），但**持久 scratch 必须惰性按首个张量的设备分配**——工厂在
  `__init__` 被调用时还没有 device，这一点必须在设计里写明，否则 Stage 3 会在一台无卡主机上分配错设备。

## Stage 3 (develop, first_impl) Retrospective — 2026-11-23

### Run Meta

| 字段 | 值 |
|---|---|
| 任务目录 | `examples/moe_permute_align/_make_align_kernel` |
| 模式 | first_impl（迁移，harness 子模式），attempt 1，design_revision_count 1 |
| 编程模式 | `simd`（`T.Kernel(G)` 一维核网格 + `T.SimtVF(threads=...)` + `T.Parallel`，UB 驻留，无跨核同步） |
| 平台 / 环境 | `ascend-950pr-9579` / toolchain `ascend`（TileLang 0.1.15 + CANN 9.2.1，`target="ascend"`）；`/workspace/venv-ascend950/bin/python`；`ASCEND_RT_VISIBLE_DEVICES=0` |
| 产出 | `_make_align_kernel.py`（kernel + `golden_make_align_kernel` + L0/L1/L2/Boundary + `main`）、`repro/TRAP-950-shared-shift-copy.py`、`repro/TRAP-950-core-slice-overread.py` |
| 判定 | `[PRECISION_PASS]`：L0 8/8、L1 12/12、L2 3/3、Boundary 6/6，全部 `max|diff| = 0`（int32 精确相等） |
| 耗时 | 冷 cache 全量 `--level all` 93.6 s（含 20 个 geometry 的首次编译）；热 cache 2.8 s；单次调用 0.1–1.2 ms |
| 复现命令 | `cd examples/moe_permute_align/_make_align_kernel && source .agents/platforms/toolchains/ascend/env-950pr.sh && ASCEND_RT_VISIBLE_DEVICES=0 /workspace/venv-ascend950/bin/python _make_align_kernel.py --level all` |

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| 设计期探针口径 | D-3 编译探针只证「可构建」，但 DESIGN §9.1 风险 7 把它当成「正确性不受影响」的下调依据。§3.3 的 HS 扫描形态（`T.copy(s_cs[0:E_pad], s_scan[E_pad:2*E_pad])` + 移位读 `s_scan[E_pad+e-k]`）在 S33/S0/L384 全 COMPILE_OK，真机上第一步扫描就把移位项丢掉，`cumsum` 全错、`npost` 384 vs 448 | 探针 `history_version/design_probe_rev1_compile.py`（COMPILE_OK）；本阶段首跑 `(8,384,64)` 的 `cumsum`/`npost` 错；`repro/TRAP-950-shared-shift-copy.py` | design skill 的 D-3 增加一条硬约束：「编译探针的结论只能写『可构建』，不得写进正确性/风险下调」；风险条目模板把两者分列 | R |
| REVIEW 建议的处置等级 | REVIEW 维度 5 观察 1 与 §7.3 处置顺序 ① 已预判「`s_cs` 兼作扫描 scratch 与最终前缀 ⇒ 写后读/WAR 隐患，必要时改双 buffer」，但定性为「若出现再改」。实测这是**首版必现**的缺陷（`s_cs` 复用 + UB→UB copy 双因） | `REVIEW.md#维度5` 建议级观察 1、§7.3；本阶段第一次 `expert_ids` 失配 `got=[55,85,113,...]` vs `want=[44,67,103,...]` | review skill 维度 5：把「同一 UB buffer 同时充当 scratch 与最终结果」列为**阻塞级**静态检查（而非建议），要求首版即分 buffer | R |
| 设计公式复核缺口 | §3.3 路径 L 的 chunk 长度 `n = T.min(CHUNK, numel - b2)` 漏了「本核切片上界」，而 `CHUNK = align_up(TPC,64) > TPC` 恒成立 ⇒ 每核多读 `CHUNK−TPC` 个下一核的 token，直方图重复计数。REVIEW 维度 3/8 复算了核数、TB 与指令数，但没有复算「每个 chunk 的读取区间是否落在本核写域」 | `DESIGN.md` §3.3 L801-805 vs §5.2；`repro/TRAP-950-core-slice-overread.py`（row sums 256/256/256/250，TPC=250） | design skill §5 分核模板增加「每核/每 chunk 读取区间 ⊆ 本核写域」的显式不等式；review 维度 3 增加同名核对项 | R |
| 执行上下文 | 平台 `execution.precision_command` 的解释器仍不可用（第三轮记录同一问题），本阶段继续按实测替换为 venv 解释器 | `statectl platform --task-dir` 返回 `/usr/local/python3.12.16/bin/python3`；实测该解释器缺 `tqdm` | 同前两轮：`statectl platform` 生成 `execution` 时做一次 import 可用性探测并回写可用路径 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **UB→UB `T.copy` 的非对齐目标偏移会产出 lane-wrong 数据**：`T.copy(src[0:N-k], dst[k:N])`（k=1,2,4,8,16,32）结果与参考移位逐元素不等；用它承载 doubling scan 的移位项时扫描「照常跑完」但已错（step 0 即错） | D | 真机实测（Ascend950PR_9579, int32）：copy-shift 6/6 档不匹配；in-buffer scan 对 `(8,384,64)` 给出 `cumsum` 错、`npost=384`（应为 448）；换双 buffer 后 `cumsum`/`expert_ids` 全对 | `examples/moe_permute_align/_make_align_kernel/repro/TRAP-950-shared-shift-copy.py`（exit 0） | TileLang 0.1.15 + CANN 9.2.1 / 950PR 真机 | `.agents/platforms/toolchains/ascend/knowledge/numerical-boundaries.md`（新增 TRAP 条目）+ `references/programming.md` 搬运章节 |
| **向量前缀扫描的正确形态：双 buffer 交替 + 零 halo，全程不发生 shared→shared copy**。数据放 `[N, 2N)`、`[0,N)` 预置 0；每步 `dst[N+e] = src[N+e] + src[N+e-k]`，奇偶步用 python 级分支选 buffer；末态左移一位写 `s_pref[0..N]`（补位 lane 得 `total`）。`E_pad=384`（9 步）与 `E_pad=64`（6 步）均精确 | P | 同上 repro 的 `two_buffer_scan` 段（n=384 与 n=64 的 inclusive/exclusive 全 OK）；本算子 L0-1/L0-6/Boundary(E=65) 全过 | `examples/moe_permute_align/_make_align_kernel/repro/TRAP-950-shared-shift-copy.py` | TileLang 0.1.15 + CANN 9.2.1 / 950PR 真机 | `references/programming.md`（扫描/归约章节补「移位用双 buffer，不用 copy」） |
| **多核 partial 直方图的 chunk 长度必须被「本核切片」与「张量末端」双重夹紧**：`n = max(0, min(min(TPC - c*CHUNK, CHUNK), numel - base))`。只夹末端时，`CHUNK=align_up(TPC,64)` 必然越过本核边界，直方图重复计数（TPC=250 ⇒ 每核多算 6 个 token） | D | `repro/TRAP-950-core-slice-overread.py`：未夹时 row sums `[256,256,256,250]`（应全 250），夹紧后与 `torch.bincount` 逐元素相等 | `examples/moe_permute_align/_make_align_kernel/repro/TRAP-950-core-slice-overread.py`（exit 0） | TileLang 0.1.15 + CANN 9.2.1 / 950PR 真机 | `tilelang-op-design` 的 §5 分核模板 + `tilelang-design-review` 维度 3 |
| 真机已验证的 int32 写法清单（本工具链）：`T.alloc_local([1],"int32")` 是**每 lane 私有**累加器（64 与 384 lane 两档语义均正确）；`T.serial(n)` 的动态区间可用；`T.copy(partial[g, :], s_part)` 二维行 region 双向可用；UB 标量读→GM 标量写（`num_tokens_post_pad[0] = s_pref[E]`）可用；非 32B 倍数长度的 copy（34 / 43 个 int32）可用；`T.fill` 真机可用 | D | 本阶段 `(8,384,64)`/`(0,8,16)`/`(600,33,64)`/`(7,3,5)`/`(15,1,1)` 与 12 条 manifest workload 全部 `max|diff|=0`；三条独立探针（copy/align、核心切片、shift）均 exit 0 | `_make_align_kernel.py`（`--level all`，29 例）+ `repro/` 两个脚本 | TileLang 0.1.15 + CANN 9.2.1 / 950PR 真机 | `references/programming.md`「已验证写法」与 `programming.md` 的 int32 支持面 |
| 路径 L 的持久 scratch 复用语义已验证：同一实例连续两次调用（第二次换数据分布）结果仍精确，说明「launch-1 全量覆盖每行、空核写全零行」的契约成立；因此 wrapper 的每实例缓存是安全的（多流并发仍不安全） | D | `_run_case` 的 2-rep 结构（29 例 × 2 次调用全过，`max|diff|=0`），含 `numel=65536` 的 2-chunk 与 `numel=0` 的零输入 | `_make_align_kernel.py --level all`（每例 `x2 reps`） | TileLang 0.1.15 + CANN 9.2.1 / 950PR 真机 | `tilelang-design-review` 维度 2（持久 scratch 生命周期）+ `tilelang-op-design` §4.7 |

### Transferable Lessons（可迁移教训）

- **前缀/扫描类 kernel 不要把「移位」交给 shared→shared `T.copy`**：本工具链上带非对齐目标偏移的 UB→UB copy 会
  产出 lane-wrong 数据，而扫描会「照常跑完」并给出看似合理的整数，只在最终输出比对时暴露。用双 buffer 交替 +
  零 halo，让每个操作数都从自己的 buffer base 起读。
- **多核分片的任务长度要夹两次**：只夹「张量末端」会在 `CHUNK = align_up(TPC, 64) > TPC` 时读进下一核的数据，
  直方图/归约类 kernel 会静默重复计数；正确形式是 `min(min(TPC - c*CHUNK, CHUNK), numel - base)`。
- **编译通过 ≠ 数值正确**：设计期探针只能证「可构建」。真机第一步应是最小 geometry（如 `numel=8`）跑通全链并与
  golden 精确比对，再放大到 manifest 规模——本任务若直接跑最大档，会把「扫描丢项」「每核越界读」两类缺陷混在一起。
- **每 lane 私有累加器用 `T.alloc_local([1], "int32")`**：在 64 lane 与 384 lane 两档真机上语义都正确，可放心用于
  「lane ↔ expert」的直方图与 rank-count；不要用 `T.alloc_var(...)[0]`（返回标量、不可下标）。
- **`if`/三元在 eager builder 里会被拦截**：`a if cond else b` 作用在 buffer 上会走 `if_then_else` 并报
  `Mismatched type ... Expected ir.PrimExpr but got Buffer`；按 buffer 切换必须写成 python `if` **语句** + 具体 buffer 名
  （条件为 python bool 时按构建期分支折叠）。
- **`numel=0` 与 `E_pad > E` 两类边界要作为独立回归位**：零输入时写回链不能被 `numel>0` 守卫吞掉；
  `E_pad>E` 时归约上界必须是逻辑长度 `E`，且左移一步必须写满 `[0, E_pad]`。本任务 L0-6/L0-7 两条都真跑通过，
  说明 v1 设计的两条不变式在实现层是可落地的。
- **scratch 复用可测**：同一编译实例连续两次调用（换数据）即可验证「全量覆盖、无跨调用残留」，成本仅一次多跑，
  比事后推断 scratch 契约可靠。

## Stage 3 patch (disclosed, post-completion) — 2026-11-23

### What Stage 5 found

`test_permute_align_op[experts-past-threads]` (`numel=512, E=1100, B=64`) failed to compile:
`Maximum number of SIMT threads returned by LAUNCH_BOUND must be an integer constant in range (0, 2048]`.
Root cause: `T.SimtVF(threads=...)` is capped at **2048** lanes, while the lane counts in this kernel are
`E_pad = align_up(E, 64)` (1152 at `E=1100`), `2 * E_pad` (2304 — the halo zeroing) and `align_up(NB, 64)`.
`_compile_single` (small path, reachable at `numel=100, E=1100, B=1`) carried the same `2 * E_pad` form.

### Fix written back into the Stage 3 artifact

| # | change | effect |
|---|---|---|
| 1 | `s_scan_b` halo zeroing: one `threads=2*E_pad` pass → two `E_pad` passes (`s_scan_b[e]`, then `s_scan_b[E_pad + e]`) | the Stage 5 fix, applied verbatim, same writes in the same order; both kernels (path S and launch 2) |
| 2 | new `SIMT_THREAD_CAP = 2048` + `_lane_spans(count)` helper, applied to **all 27** lane regions (`e_pad`, `e_pad + 64`, `blk`) | every region is emitted once per ≤2048-lane span with the span offset added to the lane index; for `count <= 2048` it is the single `(0, count)` span, i.e. the previous one-region form |
| 3 | two new L0 cases `(512, 1100, 64)` and `(100, 1100, 1)` | the defect geometry is now a gate case, next to the `E_pad > E` / `numel = 0` regressions |
| 4 | `repro/TRAP-950-simt-thread-cap.py` (exit 0) | pins the cap boundary (2048 OK, 2049/2304/2560 fail) and proves the span-split form writes the reference values |

### Evidence

| 检查 | 结果 |
|---|---|
| PRE-patch `(512,1100,64)` / `(256,2500,64)` compile | `COMPILE_FAIL` (`Ascend device compilation failed` ... LAUNCH_BOUND) |
| POST-patch same geometries | `COMPILE_OK`; both run exact vs golden (`max|diff| = 0`) |
| `threads` boundary probe | 1024 / 2048 `COMPILE_OK`; 2049 / 2304 / 2560 `COMPILE_FAIL (LAUNCH_BOUND)` |
| `--level all` (31 cases: L0 10, L1 12, L2 3, Boundary 6) | all PASS, `max|diff| = 0`, exit 0 |
| beyond-cap defensive path (`E_pad` = 2560 and 2112, 2 spans each) | exact vs golden — the span guard is load-bearing, not just a compile fix |

### Transferable Lessons

- **SIMT 线程数是硬上限（2048），不是"取多少都行"**：以 `align_up(E, 64)` 或 `2 * align_up(E, 64)` 作为 lane 数的
  kernel，在 `E > 1984` 时静默越界（CPU 侧一切正常，只在设备编译期报 `LAUNCH_BOUND`）。lane 数应统一经
  `_lane_spans(count)` 生成，跨度 ≤2048，其余部分作为第二趟，索引加跨度偏移。
- **"每 lane 一个实体"的映射会随模型规模失效**：expert lane 直接等于 `E_pad` 的写法在 `E ≤ 1984` 内可读性最好，
  一旦超过就必须退化为「多趟 + 偏移」；把这条写进设计模板的分核/布局章节，比 Stage 5 才暴露便宜得多。
- **门禁用例要按"资源上限"选，而不只按 shape 极值选**：`E=1100` 这种"专家数刚好把 lane 数推过 1024/2048"的
  几何，比 `numel` 的最大档更容易触发编译期上限，值得作为固定的边界用例。
