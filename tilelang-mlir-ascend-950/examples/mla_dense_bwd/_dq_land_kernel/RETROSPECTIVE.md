## Stage 1 (design) Retrospective — first_design — 2026-11-23

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Design-flow | `tilelang-op-design` 的 Phase 2/3 没有“**先最小化复现要用的 DSL 形态再写设计**”的门禁：本算子按 §0.6 写好的 SIMD 形态能编译（k1/k3/n1）但**整个计算体被静默丢弃**，若不做形态二分就会把一份“编译通过、算错”的设计冻结进 Stage 3 | `DESIGN.md#3.5.4`（e11 负向 / e15 正向）；`history_version/probe_e11_gen.txt`、`history_version/probe_e15_gen.txt` | 在 `tilelang-op-design` Phase 2 第 5 步（等价性机器验证）旁加一条“**代码形态探针**”：凡设计里出现 `T.simd.*` / 内联汇编级 API，必须用 ≤2 个探针确认目标形态**在设备码里真的出现**（不只是 COMPILE OK） | R |
| Design-flow | D-3 的“≤2 个探针”预算是按**性能常数标定**设计的，对“形态是否被编译器保留”这类**存在性**问题不够用（本次用了 3 个探针文件、近 40 个变体才定位到触发条件） | `DESIGN.md#3.5.4` 的探针表；`history_version/design_probe_compile.py`、`design_probe_simd.py`、`design_probe_dynamic.py` | 在 `hardware-cost-model.md`（D-2/D-3 协议）里区分“常数标定探针（≤2）”与“代码形态存在性探针（不计入，但须落盘 + 记设备码快照）” | R |
| Info-source | 平台 `knowledge/INDEX.md` 与 `references/programming.md` 都提到 `T.SimdVF` / `T.simd.*` 可用，但**没有一条**说明“语句必须词法位于 prim_func 体内”这一硬约束；该约束只在踩坑后才从设备码反推出来 | `DESIGN.md#9.2` E1；`history_version/probe_e11_gen.txt`（无 `vlds_norm`）vs `probe_e15_gen.txt`（有） | 在 `.agents/platforms/toolchains/ascend/references/programming.md` 的 SIMD 行补一条“**VF 语句的摆放约束**：`T.simd.*` 语句与承载它们的循环必须写在 prim_func 体内；由 python 辅助函数展开会在 VF outlining 时被丢弃（`__simd_vf__` 无 buffer 参数）” | D |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| SIMD 体静默丢弃：python 层展开 vs IR 段循环（负向发现，含设备码对照） | D | `examples/mla_dense_bwd/_dq_land_kernel/history_version/probe_e11_gen.txt`（VF 函数只剩两个 mask，无 `vlds_norm`）与 `probe_e15_gen.txt`（完整 `vlds_norm/vmuls/vcvt/vsts_norm`） | `cd examples/mla_dense_bwd/_dq_land_kernel && source ../../../.agents/platforms/toolchains/ascend/env-950pr.sh && ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python history_version/design_probe_dynamic.py e11 e15`（e11 应打印 `SIMD body present: False`，e15 为 True） | tilelang 0.1.15（venv-ascend950 安装包）+ CANN 9.2.1 + Ascend950PR_9579；编译级、无 launch | pattern-library / `.agents/platforms/toolchains/ascend/references/programming.md` |
| 平台 DMA cast 被语义检查硬拒（`T.copy` f32→f16 不允许） | D | `examples/mla_dense_bwd/_dq_land_kernel/DESIGN.md#0.5`（#3 行，报错原文）；探针 m5/m6 | `… /workspace/venv-ascend950/bin/python history_version/design_probe_simd.py m5 m6`（两者均 COMPILE FAIL） | 同上 | `.agents/platforms/toolchains/ascend/references/programming.md`（copy 条目） |
| 动态维 + `out_idx` 的符号绑定要求：`T.dynamic` 必须复用同一对象，否则 callee-allocated 输出无法推理 | D | `DESIGN.md#3.5.4`（e14 反例 / e15 正例）；报错原文 `Cannot infer symbolic variable total_q ... bind it from an input tensor or scalar parameter` | `… python history_version/design_probe_dynamic.py e14 e15` | 同上 | `.agents/platforms/toolchains/ascend/references/programming.md` 或 kernel_template |
| 纯带宽逐元素算子的“无融合”判据：跨 CTA fp32 累加 + 无设备级 barrier ⇒ 落盘 launch 不可省 | P | `DESIGN.md#1.6.0` R1-e / R3（省 8 B/elem = 805 MB 的量化）+ `workflow.md` 第 29 行（`ascend.batch_schedule` 跨 block barrier 未认证） | 无需复现（设计层论断，证据为文档限制 + 流量算术 `10E→2E`） | 同上 | `algorithm-candidates` / `hardware-cost-model` |
| 设计期 roofline 在本型号缺常数时的写法：跨型号常数 + 每核地板外推 + 显式“未实测假设”标注 | P | `DESIGN.md#1.6.0` 的「设计期估算」行（479–586 µs，常数 `CONST-mte2-degradation`/`CONST-fabric-bw` 均为 910B2C） | none（Stage 4 回填对账） | 本次工具链 0.1.15/9.2.1；常数来源为 910B2C 0.1.2/CANN 8.5.0 | `.agents/skills/_shared/standards/hardware-cost-model.md` |

### Transferable Lessons（可迁移教训）

- 在本平台的 `T.SimdVF` 里，**语句的物理位置决定它是否被编译**：`T.simd.*` 与承载它们的 `for` 必须直接写在 `@T.prim_func` 体内；一旦抽到模块级 python 辅助函数里用 `for s in range(n)` 展开，编译器仍报 COMPILE OK，但设备码里的 `__simd_vf__` 函数会只剩 mask 创建、没有任何 `vld/vmuls/vcvt/vsts`——**必须打开生成代码确认计算体存在**，不能只看编译返回。
- 本平台的 DMA `T.copy` **不做 dtype 转换**（前端语义检查直接拒绝 f32→f16）；cast 必须放进 `T.SimdVF`（`vcvt`）或 `T.SimtVF`（`T.cast`）里做。
- 形如 `T.Tensor((T.dynamic("total_q"), …))` 的张量注解，若要配 `out_idx`（JIT 分配输出），必须先把 `T.dynamic(...)` 绑定到 python 变量再在两处注解复用，否则输出 shape 推理失败（`Cannot infer symbolic variable …`）。
- 判断“某段向量代码是否真的生效”最省时的办法是**把生成代码里的指令名当断言**（`grep vlds_norm|vmuls|vcvt|vsts_norm`）：本任务的 e11/e15 对照用两行 grep 就定位了整类缺陷。
- 对“逐元素 + 输入本来就是 fp32”的落盘算子，**不要**为了省一次乘把 scale 拉到 16 位域做：两次舍入在 fp16 上可达 3857 ulp（`verify_equiv.py` 的 C1 对照），且没有任何性能收益。

## Stage 2 (review) Retrospective — review — 2026-11-23

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Review-flow | Phase 1 第 7 步写的是 `python examples/{project}/{op}/verify_equiv.py`，但脚本依赖 torch 且需要工具链环境；用系统 `python3` 直接跑会缺依赖。本次按任务 prompt 改用 `/workspace/venv-ascend950/bin/python` + `source env-950pr.sh` 才跑通 | 本次 `python verify_equiv.py` → `[EQUIV_PASS]`（venv 解释器）；系统 python3 无 torch | 在 `tilelang-design-review` Phase 1 第 7 步注明"解释器与环境由任务平台 `execution` / 任务 prompt 决定，先用平台 venv"，并给出「先用 `--help`/import 探测解释器」的一行做法 | R |
| Mechanical-check | `design_calc_check.py` 在本设计上 4 项里 3 项 `skip`（`ub_budget` 解析不到 §4.5 的 markdown 预算表；`core_split` 拿不到 `block_M/block_N` 形式的 dims，因为 persistent 设计用 `cpt`/`G` 表达分核）；对"逐元素 + persistent"这一大类，可机检子集几乎为空 | 本次 JSON：`ub_budget` skip / `l0c_budget` skip / `core_split` skip / `r3_metrics` pass | 在 skill 里明确"persistent / 非 GEMM 设计的机检子集可能整体为空，须全量人工复核并逐项写结论"，避免检视者把 skip 当 pass；同时给 checker 记一条改进项（识别"每级缓冲表"与 `cpt = ceildiv(ntiles, G)` 形式） | R |
| Review-flow | 迁移任务在"无卡、仅编译级证据"条件下，设计的**设备码断言**（"设备码显示 …"）只能靠 `history_version/*_gen.txt` 快照核对，但维度表没有把这一项列为显式检查点；本次在维度 5/7 顺手核对时发现 2 处引用串号（`§6.4` 的 store 守卫形态其实来自 e10/e11；`§6.3` 的流水重叠结论与 e15 程序序不符） | `history_version/probe_e15_gen.txt`（store 侧只有单层 `min`、无 `if`；无跨轮 load-ahead）vs DESIGN §6.3/§6.4 | 在维度 5 与维度 7 各加一行："设计里每条『设备码显示 X』的断言，必须回到快照文件逐条比对；摘要与快照不一致时按 warn 记并指明出处" | P |
| Info-source | 判"负向断言是否成立"时，平台 `knowledge/capability-gaps.md` 是最有效的第一跳（本次 `CG-950-batch-barrier` 直接印证"融合进上游 launch 不可行"），但 skill 的调研复核步骤只点名 `algorithm-research.md` 参考表与源码证据，没点名 capability-gaps | `knowledge/capability-gaps.md` 的 `CG-950-batch-barrier`；`workflow.md` 第 29 行 | 在维度 8「调研结论正确性 ④ 负向断言复核」的取证清单里补一项："先查 `knowledge.capability-gaps.md` 的匹配条目" | P |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 编译级证据类设计的检视协议：设备码断言逐条回快照 | P | 本次 REVIEW 维度 5/7 的两处引用串号（§6.3 流水重叠、§6.4 store 守卫） | 对照 `history_version/probe_e15_gen.txt` 与 `probe_e10_gen.txt` / `probe_e11_gen.txt` 三份快照逐行比对 | tilelang 0.1.15 + CANN 9.2.1 / Ascend950PR_9579（编译级） | `tilelang-design-review`（维度 5/7 检查项） |
| "静默丢弃类缺陷"应从设计风险升级为 Stage 3 硬前置（设备码指令名断言） | R | 本次 REVIEW 的独立复核：e11 的 `__simd_vf__` 只剩 mask 创建、e10（唯一差别是模块级 helper 展开）计算体完整 | 编译后 `kernel.get_kernel_source()` grep `vlds_norm|vmuls|vcvt|vsts_norm`，缺失即失败 | 同上 | `tilelang-op-develop`（L0 判据）+ `.agents/platforms/toolchains/ascend/knowledge/capability-gaps.md`（当前缺该条目） |
| `vcvt(sat=False)` 的语义在安装包 docstring 写 "wrap"，与 IEEE inf 的目标语义未定 | P（未验证假设，非实测） | `tilelang/ascend/language/simd.py:vcvt` docstring（sat: "No sat; out-of-range values wrap"）+ `knowledge/numerical-boundaries.md` 第 19 行（"不把 SIMT 结论用于 `T.simd.vcvt` 的任意模式"） | 需真机 L0 角点（65519/65520/1e5）实测后才能登记 D 类 | tilelang 0.1.15 + CANN 9.2.1；**未实测** | `.agents/platforms/toolchains/ascend/knowledge/numerical-boundaries.md`（补一条 `T.simd.vcvt` 语义待验证项） |

### Transferable Lessons（可迁移教训）

- 对 persistent / 逐元素类设计，`design_calc_check.py` 的可机检子集会整体退化为 skip；检视结论必须建立在**手工复算**上（本次逐项复算了 UB 字节、五个 workload 的逻辑核数、`cpt` 失衡、`6E` 流量与四条时延估算、A2 弃选的 294912 B）。
- 设计文档里"设备码显示 …"这类摘要**极易串号**（同一算子目录有多份快照）：核对时必须打开被引用的具体文件，而不是接受摘要里的括号注释。
- 复核"无更优替代"这类负向断言的最高性价比路径是两条独立证据交叉：平台 `capability-gaps` 的条目 + 源算子的结构事实（本次是 `_bwd_kernel` 的 `T.atomic_add` 跨 CTA 累加 ⇒ 最后一趟部分和到达前无法定稿）。
- 当"编译级已证、运行期未证"的项（运行时语义 / 数值语义 / 动态输出分配）在 L0 计划里都有对应用例且失败有具体回退方案时，应判 warn 而不是 fail——否则会把所有无卡条件的设计都判不通过。

## Stage 3 (develop) Retrospective — first_impl attempt 1 — 2026-11-23

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Develop-flow | 设计把 `vsts` 的分布定成 `NORM_B16`（来自**纯编译级**探针 e15），而 `vcvt` 窄化后的 lane 摆放只有真机才能观察：按设计原样实现后 96/192 列压根没被写过（`history_version/stage3_probe_map2.py`）。实现期为此花了 6 个探针才定位 | `_dq_land_kernel.py` 文件头第 1 条；`history_version/stage3_probe_map2.py`、`stage3_probe_emit.py`、`stage3_probe_sentinel.py` | 设计 skill 的探针表里，凡出现 `vcvt` 窄化 + `vsts` 的形态，必须标注「**store 分布未验证**」并列入 L0 判据；`vsts` 的 dist 不应由编译通过来背书 | R |
| Info-source | 平台 `programming.md` 只说「`vcvt` 的 dtype/舍入/饱和参数与 SIMD load/store 分布参数需要按具体路径核对」，没有给出「窄化结果落在 16 位寄存器的**偶 lane**、稠密落盘要用 `PK_B32` + **32 位 lane 掩码**」这条可用事实；`simd.py` 的 `vsts`/`vcvt` docstring 也没有 | `.agents/platforms/toolchains/ascend/references/programming.md`（SIMD 段）；`repro/CONST-950-vcvt-narrow-store-pack-b32.py`（实测断言） | 在 `programming.md` 的 SIMD 行补一条窄化落盘配方：`vcvt(..., part=0)` → `vsts(..., dist="PK_B32")` + `pset(32, "PAT_ALL")`（掩码宽度必须跟 `PK_B32` 的 32 位 lane 对齐，否则只写一半） | D |
| Develop-flow | **UB 内容在同一进程的多次 launch 之间会保留**，因此「上一版写对过 → 下一版半写也能读回正确值」：同一形态的两个探针给出互相矛盾的结论（probe #5 报 2/768 错，probe #8 报 384/768 错），白耗一轮 | `history_version/stage3_probe_store2.py`（2/768）vs `history_version/stage3_probe_emit.py`（384/768）；定论见 `history_version/stage3_probe_sentinel.py`（内核内 sentinel 预填） | 在调试 skill（`tilelang-debug-helper`）与平台 `programming.md` 各加一条：「**半写/漏写的存储实验必须在内核内先把目标缓冲填 sentinel**，跨 launch 的 UB 残留会让错误看起来正确」 | P |
| Develop-flow | L0 计划里的 `P4 非饱和` 用例在 bf16 上按 `mx*1.5/scale` 构造角点时**溢出 float32**（`RuntimeError: value cannot be converted to type float without overflow`），设计没注意到 bf16 的有限范围几乎就是 fp32 的范围 | `_dq_land_kernel.py::_make_input`（`special="overflow"` 分支的 clamp + 校验）；`DESIGN.md#8.3` L0-6 | 设计/实现模板里对「按 dtype 上界 / scale 反推角点」给出通式：`edge = min(mx*1.5/scale, fp32max)` 并断言 `edge*scale > mx`，否则该 dtype 在此 scale 下无法构造溢出角点（bf16 需 scale ≥ 1） | P |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| `vcvt` f32→16 位窄化结果的落盘分布：`PK_B32` + 32 位 lane 掩码是唯一稠密形态，`NORM_B16` / `PK_B16` / 16 位掩码都会丢一半（负向发现，含实测数字） | D | `history_version/stage3_probe_sentinel.py` 输出：`PK_B32 mask=32` 256/256 正确、0 sentinel；`PK_B32 mask=16` 128/256 正确、128 sentinel；`NORM_B16 mask=16` 仅 4/256 正确；`stage3_probe_map2.py` 给出第 0 行逐槽映射 `[0..31 正确, 32..63 未写, 64..95 正确, 96..127 未写, …]` | `cd examples/mla_dense_bwd/_dq_land_kernel && source ../../../.agents/platforms/toolchains/ascend/env-950pr.sh && /workspace/venv-ascend950/bin/python repro/CONST-950-vcvt-narrow-store-pack-b32.py`（三行断言全过，打印 256/256、128/256、4/256） | tilelang 0.1.15（venv-ascend950 安装包）+ CANN 9.2.1 + Ascend950PR_9579 真机 | `.agents/platforms/toolchains/ascend/references/programming.md` / pattern-library |
| `vcvt` 的 `sat` 标志在 f32→f16 上**无效**，且 `RS_DISABLE` 是 IEEE（65520 → ±inf），与 docstring 的 “wrap” 措辞相反（实测数字表） | D | `repro/CONST-950-vcvt-rs-disable-ieee.py` 输出表：65504→65504、65519→65504、65520→inf、1e5→inf，`sat=False` 与 `sat=True` 逐项相同；两版设备码分别含 `RS_DISABLE` / `RS_ENABLE`（同一脚本内 `get_kernel_source()` 对照） | `… /workspace/venv-ascend950/bin/python repro/CONST-950-vcvt-rs-disable-ieee.py`（断言 `on == off` 且越阈值全 inf） | 同上 | `.agents/platforms/toolchains/ascend/knowledge/numerical-boundaries.md`（把 REVIEW 的「待实测」项升级为已实测） |
| NaN payload：`vcvt` 把任意输入的 NaN 规范成 `0x7FFF`（fp16）而 torch 保留输入的 NaN 位型（fp16 `0x7E00` / bf16 `0xFFFF`），非 NaN 元素逐位一致 | D | L0-6a/L0-6b 角点用例的实测位型：dev `0x7fff` vs torch `0x7e00`（fp16）、dev `0x7fff` vs torch `0xffff`（bf16）；同一用例非 NaN 元素 `nonzero bits equal = True` | `… python _dq_land_kernel.py --level L0`（`L0-6a/6b` PASS；`_compare` 只断言 NaN **位置**一致并 `equal_nan=True`） | 同上 | `.agents/platforms/toolchains/ascend/knowledge/numerical-boundaries.md`（NaN payload 不保留） |
| 真机时延对账：设计期 479–586 µs 的 roofline 估算在本型号上 4 个 workload 里 3 个命中，最大 workload 偏低 1.4 倍 | D | `_dq_land_kernel.py --level L2` 输出：ds-v3-4k 0.580 ms、kimi-k2-8k 0.576 ms、ds-v2-lite-3k 0.195–0.255 ms（设计 479–586 µs / 180–220 µs ✓）；ds-v3-b2-8k 3.231 ms（设计 1917–2346 µs，实测 ≈1.4×，2415.9 MB ⇒ 约 0.75 TB/s） | `… python _dq_land_kernel.py --level L2` | 同上（`elapsed` 含单次 launch，未做 msprof 采集） | `.agents/skills/_shared/standards/hardware-cost-model.md`（Stage 4 回填「设计估算 vs 实测」偏差行） |
| 把「编译通过 ≠ 算对」做成工厂内硬断言（P0）的落地形态：`_simd_body_gate` 在每次 compile 后检查设备码，并把命中集合落进 L0 报告 | P | `_dq_land_kernel.py::_simd_body_gate` + `run_L0` 的 `L0-8` 输出（9 个几何全部 `compute=['vlds_norm','vmuls','vcvt']`、`store=['vsts_pack_b32']`） | `… python _dq_land_kernel.py --level L0`（`L0-8 P0-simd-body-and-cache` PASS） | 同上 | `tilelang-op-develop`（L0 判据）/ kernel_template |

### Transferable Lessons（可迁移教训）

- 本平台 `T.simd.vcvt` 的 **f32→16 位窄化**结果是「每个 32 位字装一个值」（偶 lane），稠密写回必须 `T.simd.vsts(..., dist="PK_B32")` **且掩码用 `T.simd.pset(32, ...)`**；用 `NORM_B16`、`PK_B16` 或 16 位掩码都能编译、都在自己写到的槽上正确，但只覆盖一半数据。
- `T.simd.vcvt` 的 `sat` 参数在 f32→f16 上**不改变结果**（`RS_ENABLE` 与 `RS_DISABLE` 实测逐项相同），两者都是 IEEE（65520 → ±inf）；因此不要靠 `sat=True` 去实现「钳到最大有限值」，也不要因为 docstring 写 “wrap” 就假设关饱和会回绕。
- 存储类（半写/漏写/越界）实验必须**在内核内先把目标 UB 缓冲填 sentinel** 再写：同一进程里 UB 内容跨 launch 保留，上一版写对过的数据会让下一版的漏写读回「正确」值——这是本任务里唯一让两个探针互相矛盾的根因。
- 用 torch 上界/scale 反推溢出角点时，bf16 的有限范围几乎等于 fp32 的范围（3.39e38 vs 3.40e38），所以 **bf16 的窄化溢出用例必须配 scale ≥ 1**，否则操作数在 fp32 里就已经越界、根本构造不出该角点。
- 纯带宽逐元素迁移的 L0 判据里，「与 golden 逐位相等（bit-exact）」是可达的（本算子全部 31 个用例 `max_diff=0` / `bit_diff=0`）：一旦出现容差内但不逐位的差异，应先怀疑落盘分布或掩码，而不是浮点误差。
- 本平台 `vcvt` 窄化**不保留 NaN payload**（任意 NaN 输入 → 输出 `0x7FFF`），而 torch 保留输入位型；逐位比对时必须「NaN 只比位置」并 `equal_nan=True`，否则含 NaN 的用例会假失败。
- 动态 `T_q` + `T.Pipelined(动态界)` + `T.annotate_buffer_versions({buf:2})` + `out_idx=[1]` 这套组合在本平台**真机可用**（设计 R2/R3 的编译级不确定项已闭环）：同一几何连跑 20 次结果逐位一致，且去掉版本声明的单版本对照也逐位一致。
