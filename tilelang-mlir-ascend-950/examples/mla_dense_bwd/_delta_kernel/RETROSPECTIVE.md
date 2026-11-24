# RETROSPECTIVE — mla_dense_bwd / _delta_kernel

## Stage 1 (design) Retrospective — first_design — 2026-11-23

迁移任务（harness 子模式：TileOPs GPU `_delta_kernel` → Ascend950PR_9579 / `target="ascend"`）。
任务约束：只产出 `DESIGN.md`（+ 本复盘 + `verify_equiv.py` + 设计期编译探针）；本机仅 1 张卡且被同批次
另一算子的 Stage 3 占用，设计期只做无卡编译/静态探针（`ASCEND_RT_VISIBLE_DEVICES=""`）与 CPU 数值实验。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Constraint-doc | `gate_lint.design_lint` 的 `S1-EQUIV-EXEC` 用 `"EQUIV_FAIL" in §1.6.1` 判失败，因此设计者连“无 EQUIV_FAIL 项 / 全部通过、无失败项”这种**否定式表述**都不能写（本次写作时被迫改写措辞）。规则意图是“存在失败项”，实现却对字面量敏感。 | `DESIGN.md#1.6.1`；`.agents/tools/gate_lint.py` 第 322–330 行 | 把判定改为按表格单元扫描（如 `re.search(r"\\|\\s*EQUIV_FAIL\\s*\\|", r161)`），或在 `S1-EQUIV-EXEC` 的消息里注明“不得以任何形式出现该字面量”。 | R |
| Info-source | D-2 设计期 roofline 要求引用匹配本型号/工具链的实测常数，但平台 `cost_measurements` 为空，`pattern-library/constants.md` 全部条目绑定 910B2C + `tilelang 0.1.2`，本任务（950PR / 0.1.15 / CANN 9.2.1）只能写“未实测外推”，估算下界的可信度受限。 | `DESIGN.md#1.6.0` 设计期估算段；`.agents/tools/statectl.py platform --task-dir …` 返回 `cost_measurements: []` | 平台侧为 950PR 补一条最小实测常数集（HBM 混合读写地板 / 向量发射），或在 design skill 的 D-2 协议里给出“zero 常数”时的统一降级模板。 | D |
| Design-flow | Phase 2 的向量化替代分析表预设“归约能用平台的高层归约 API 替代”，但 ascend 侧 `T.reduce_sum` 的二维 shared→shared 路径有已登记的 TRAP、SIMT reducer 路径每行一次 barrier，实际只能落到显式 SIMD。skill 的“不可替代理由类别”里没有“高层 API 存在但代价倒挂”这一类。 | `DESIGN.md#1.6.2` V9；`.agents/platforms/toolchains/ascend/knowledge/numerical-boundaries.md`（`TRAP-950-simd-shared-reduce`）；`.agents/platforms/toolchains/ascend/references/programming.md` 第 48–49 行 | 在 `references/algorithm-research.md §3` 或 design skill Phase 2 第 2 项增补理由类别“**高层 API 存在但结构性代价倒挂（附量化）**”。 | P |
| Design-flow | skill Phase 2 第 5 步的等价性机器验证只规定“fp64 参照 + 角点值”，未规定**判决容差口径**。对纯归约类算子，`atol + rtol·abs(ref)` 会把任何 fp32 累加判为违反（抵消行），本次首版脚本即因此全项 FAIL，需自行推导尺度化口径（误差 ∝ Σabs(p_d)）。 | `verify_equiv.py`（`RTOL_SCALED` 注释与首轮失败输出）；`DESIGN.md#1.6.1` 表注 | 在 design skill Phase 2 第 5 步写明“归约类算子的对照容差以 Σ|项| 为尺度（并附标准容差违反率作参考）”，避免每个任务重复踩。 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| ascend 上 3-D GM 切片的「固定中间索引」区域拷贝 `o[r0:r0+R, h, :] → UB[R,D]` 生成**单条跨步 DMA**（`asc_copy_gm2ub_align(dst, src, 64, 256, …, 2048, 256)` = 64 行 × 256 B、源间距 2048 B、目的连续），不是 R 条 1-D 拷贝；这让“head 固定 + 行块”布局在 NPU 上可行。 | D | `examples/mla_dense_bwd/_delta_kernel/history_version/design_probe_compile.py`（P1）；`DESIGN.md#3.4`、`#0.6` R3 | `cd examples/mla_dense_bwd/_delta_kernel && ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python history_version/design_probe_compile.py p1` | tilelang 0.1.15 + CANN 9.2.1 SDK；ascend-950pr-9579；仅编译（无设备） | `pattern-library / layout`（拷贝描述符形态） |
| UB→GM 的「压缩写」`T.copy(d_ub[:, 0:1], delta[b,h,r0:r0+R])`（源为 `(R,8)` fp32 的列）生成单条跨步 store（`asc_copy_ub2gm_align(…, 64, 4, …, 4, 32)` = 64 元素、目的连续、源间距 32 B），可把“每行一次 4 B 标量写”降为每 tile 一次描述符。 | D | 同上 P1；`DESIGN.md#1.6.1` A3 | 同上 | 同上 | `pattern-library / layout` |
| **负面发现**：`T.Pipelined(n, num_stages=2)` + `T.annotate_buffer_versions({x:2})` 在 ascend 上只生成**双版本 UB**，并**不把下一 iteration 的装载提前到本 iteration 的计算之前**（生成的 `for (step…)` 内 `asc_sync_wait(PIPE_V, PIPE_MTE2, step)` 与 compute 同迭代）。因此“流水带来装载/计算重叠”是**未证实收益**，必须以实测裁决，不能在设计期假定。 | D | 同上 P4（完整生成代码）；`DESIGN.md#6.3`、`#1.6.3` 裁决计划 | 同上（`… design_probe_compile.py p4`） | 同上 | `pattern-library / bottleneck-patterns`（流水形态的实测边界） |
| 16-bit 行点积的标准指令序列在 ascend 上成立：`vlds_norm<dtype>` → 2×`vcvt(PART_EVEN/PART_ODD)` → 2×`vmul` + `vadd` → `vcadd` → `vsts_norm(…, PAT_VL1)`，fp16 与 bf16 双 dtype 均可编译，每行 11 条向量指令（D_v=128 恰好 1 个 2048-bit 寄存器）。 | D | 同上 P2a/P2b；`DESIGN.md#1.6.0` 发射项、`#3.2` | 同上（`… p2`） | 同上 | `pattern-library / constants`（补 950PR 每行指令预算行） |
| 归约类算子的等价性判决须用**尺度化容差**（`abs(cand-ref) <= 1e-5·Σabs(p_d) + 1e-6`）：`atol + rtol·abs(ref)` 会把“大项相消行”（Σabs(p)≈1e5 而 abs(Σp)≈0）判为违反——基线自身也违反（本次 large/fp16 有 1/64 行），与等价性无关。 | D | `examples/mla_dense_bwd/_delta_kernel/verify_equiv.py`；`DESIGN.md#1.6.1` 表注 | `/workspace/venv-ascend950/bin/python examples/mla_dense_bwd/_delta_kernel/verify_equiv.py`（CPU） | tilelang 0.1.15；CPU 数值实验，无设备 | `references/algorithm-research.md`（D-1 容差口径） |
| 迁移模式：带宽受限的“逐行点积”算子——一寄存器一行 + 逐行 `vcadd` + 32 B 行间距暂存 + 压缩写出；以 `grid = batch·heads·nsp` 固定核数、块内 tile 流水，避免一 tile 一 block 的核超发。 | C | `DESIGN.md#0.6` R1–R3、`#1.6.3` 候选 1 | none | tilelang 0.1.15 + CANN 9.2.1；ascend-950pr-9579 | `references/migration-analysis.md`（NPU 重设计模式库） |

### Transferable Lessons（可迁移教训）

- 编译探针的价值在**生成代码**，不在“COMPILE: OK”：跨步拷贝是否单描述符、段长/间距参数、同步 flag 配对、流水是否 peel，都只能从 `get_kernel_source()` 的 `asc_copy_*_align` / `asc_sync_*` 行读出；只报成功率的探针等于没做。
- `gate_lint` 的 `S1-EQUIV-EXEC` 对字面量 `EQUIV_FAIL` 敏感（出现在 §1.6.1 任何位置即 fail），因此该章节里连“无 EQUIV_FAIL”这样的否定句都不要写，改用“全部机器验证通过”。
- ascend 的行归约不要用 `T.reduce_sum`：SimdVF 内二维 shared→shared 路径有已登记的输出布局 TRAP，SIMT reducer 路径每行一次 barrier；逐行显式 `vld → vcvt → vmul → vadd → vcadd → vsts` 是已验证且开销最低的形态。
- 归约类算子的数值等价性检查要用“Σ|项| 为尺度”的容差；否则大项相消的行会把任何 fp32 累加（含基线）判成不等价。
- 迁移带宽受限算子时，先按“每元素必读一次 + 每行必写一次”写下界，再逐候选估算；对角 GEMM、两遍物化、跨核切分这类“看起来更并行”的方案在本类算子上都是流量/精度/同步上的净亏。
- `T.Kernel` 在 ascend 上是一维且核级无 `threads`：任何 GPU 形态的 `T.Kernel(x, y, threads=N)` 都必须先改并行结构再谈 API 映射，不能逐 API 翻译。

## Stage 2 (design review) Retrospective — review_v1 — 2026-11-23

迁移任务 9 维度检视（结论：不通过，3 阻塞 + 7 建议）。设计期证据全部用无卡编译探针重跑
（`ASCEND_RT_VISIBLE_DEVICES=""`），未 launch 任何 kernel；`verify_equiv.py` 与
`design_calc_check.py` 均按 D-1/D-5 重跑。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Review-dimension gap | 维度 1 的检查项是「API 存在性 / 与模式匹配 / 规范」，**没有**「伪代码写法可构建性」。本次三条阻塞里两条（`range()` 循环携带向量累加器、整 buffer 对局部 region 的 `T.copy`）都是「API 名字存在、写法不可构建」，只能靠检视者自建探针发现，检查表本身抓不到。 | `REVIEW.md` 问题 1/2；`/tmp/opencode/probe_dv64c.py` C1/C3/C5 | 维度 1 增一行「**伪代码关键写法的可构建性**：对 §3.3 中带构建期分支/循环携带量/形状可变的搬入至少抽 1 例做无卡构建（`tilelang.compile(..., target="ascend", out_idx=[])`），仅 API 名存在于 `simd.py` 不构成写法可用证据」。 | P |
| Mechanical-check blind spot | `design_calc_check.py` 在本设计上 `ub_budget` / `core_split` / `l0c_budget` **全部 skip**（`"no per-level buffer tables parsed"`、`"insufficient parsed dims {}; declared logical cores = 3"`），D-5 的机械半场等于零覆盖，只能人工复算；且 `core_split` 把正文「三个 launch」的 3 当成了逻辑核数。 | `/tmp` 下的 `design_calc_check` JSON 输出（`REVIEW.md` 机械复核段） | ① `core_split` 的 dims 解析失败时输出它尝试匹配的 anchor（章节/表格名）并把「declared logical cores」标为「unparsed fallback」；② `skip` 时在 `detail` 里回显解析器期望的表格形态（`DESIGN.md` 的 §4.5/§5.5 表头样例），否则设计者与检视者都无法判断 skip 是格式问题还是内容缺失。 | R |
| Info-source (repeat of Stage 1) | 判定「215 µs 下界」是否保守时，平台 `cost_measurements` 仍为空、pattern-library 常数全是 910B2C/48 核口径，本型号（950PR/56 核）无实测常数。检视者只能做核数外推敏感性分析（48 核聚合 215 µs vs 每核 × 56 核 262 µs），无法给出真值判定。 | `DESIGN.md#1.6.0` D-2 段；`statectl platform --task-dir …` → `cost_measurements: []` | 与 Stage 1 同一条：平台补 950PR 最小实测常数集（混合读写地板 + 向量发射）；在此之前 review 侧对「未实测外推」类估算只判「口径是否标注 + 敏感性是否给出」，不判数值对错（本次即按此处理为建议 5）。 | D |
| Coverage-source mismatch | 检视「验证方案覆盖」时，skill 只要求「具体规则 shape」，未说明以哪一个几何清单为准。本次 `D_v=64` 只出现在 op 自己的测试文件（`test_mla_dense_bwd_general_kernel`，带 smoke 标记），manifest workloads 里没有——设计因此写了「本任务不触发」，而 Stage 5 会跑到。 | `REVIEW.md` 问题 2；`examples/TileOPs/tests/ops/test_mla_dense_bwd.py` | 维度 6 增一句「覆盖核对以**三处并集**为准：manifest `workloads` + 本仓 `tests/ops/test_*.py` 的全部 param + bench 用例；设计以 manifest 为准时必须显式列出被排除的测试几何及理由」。 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **负面发现**：TileLang ascend eager builder 把 python `range()` 提升为 IR ForFrame，**跨迭代 rebind 的向量值不可用**（`RuntimeError: Immutable variable 'X' is used outside its defining region`）；把初值前置的 loop-carried 变体同样失败。因此「按 `nvec` 分段累加同一行部分和」只能**构建期特化 + 手写展开**（生成单一表达式），不能用循环。 | P | `REVIEW.md` 问题 1；/tmp/opencode/probe_dv64c.py C1/C5 vs C2 | `cd /tmp/opencode && ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python probe_dv64c.py` | tilelang 0.1.15 + CANN 9.2.1；ascend-950pr-9579；仅编译 | `tilelang-op-design/references/algorithm-research.md`（R4/向量化落地的写法约束）或 design skill Phase 2「向量化替代」 |
| **负面发现**：`T.copy(src_region, dst_buffer)` 在源区域元素数与目的 buffer 不等时**编译期报错**（`PlanMTECopy: source and destination regions must contain the same number of elements`）。`D_v` 小于向量宽度（如 `D_v=64, DW=128`）时必须把目的写成显式切片 `o_ub[0:R, 0:D_v]`；且 pad 列若要「零填充等价」，**两侧操作数都要清零**（只清一侧时另一侧是未初始化 UB，`0·Inf/NaN → NaN`）。 | P | `REVIEW.md` 问题 2；probe_dv64c.py C3（FAIL）/C4（OK）；`references/programming.md`「整个 Buffer 拷贝要求元素总数相等…局部拷贝用显式 region 表达」 | 同上（脚本内 D=64 两 case） | 同上 | `pattern-library / layout`（UB 行宽 pad 的正确写法）+ design skill 的「非整除处理」小节 |
| **负面发现**：尾块（`S % R ≠ 0`）时「行循环上界静态化」不等于「搬入/写出 extent 静态化」——编译器**不拦截**超界 extent 的 `T.copy`（R-extent 与 nlast-extent 两种写法都 COMPILE OK），越界读/越界写会静默进入运行期。设计文档必须把 `nlast` 同时落到两条 copy。 | P | `REVIEW.md` 问题 3；/tmp/opencode/probe_tail.py（两写法均 OK） | `cd /tmp/opencode && ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python probe_tail.py` | 同上 | `tilelang-op-design` 的尾块处理条款 + `references/quality-checklist.md` |
| 迁移检视的「证据指向」核对法：`verify_equiv.py` 通过 ≠ 实现了被验证的那条公式。本次 §1.6.1 A2 验证的是「2×`vcadd` + 标量加」，实现是「`vadd` 两半 + 1×`vcadd`」；检视时用同一 harness 重算**实现分组**才确认等价性成立（12/12 违反率 0）。 | P | `REVIEW.md` 建议 1；/tmp/opencode/check_opt1.py 输出表 | `ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python /tmp/opencode/check_opt1.py` | tilelang 0.1.15；torch CPU 数值实验 | 本 skill 维度 8「等价性论证正确性」行（补一句「核对被验证的分组/公式与 §3.1/§3.3 实现是否同一式」） |

### Transferable Lessons（可迁移教训）

- 检视伪代码不能止步于「API 在 `simd.py` 里有导出」：必须挑**承载构建期边界的那几句**（循环携带量、可变 shape 搬入、尾块 extent）做一次无卡构建。本次三条阻塞全部是「API 存在 + 写法崩」，只看导出表 100% 漏检。
- 迁移任务的验证覆盖要以「manifest workloads ∪ tests/ops 的 param ∪ bench」并集核对；只按 manifest 写测试计划，会把 Stage 5 必然跑到的几何（本次 `D_v=64`）判成「本任务不触发」。
- 「生成代码不 peel 流水」这类结论（Stage 1 已登记）在检视时应复用同一探针复核：本次 dump P4 全文确认装载与计算同迭代，从而确认设计把流水收益送进 A/B 裁决是正确处置，而不是把未证实收益写进设计前提。
- 机械复核的 `skip` 不等于通过：`design_calc_check` 三项 skip 时必须人工复算并在 REVIEW.md 里写明「复算过程 + 结论」，否则 D-5 的机械半场归零而检视报告看起来仍是「已复核」。

## Stage 1 (design) Retrospective — revision 1 — 2026-11-23

迁移任务 `mode=revision`（路径 A：Stage 2 检视不通过，3 阻塞 + 7 建议）。本版把三条阻塞逐项闭环到
**可编译形态**，并顺手吸收 4 条建议；算法主线与 §0 迁移决策不变。设计期仍只做无卡编译探针
（`ASCEND_RT_VISIBLE_DEVICES=""`）与 CPU 数值实验。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Design-flow | Stage 2 的负面结论只到「必须构建期特化 + 手写展开」，没说清 `nvec > 1` 时**怎么写**才不被 eager builder 拒绝。本次实测：把展开逻辑放进**模块/工厂层 python 函数**（`row_dot`），其 `for v in range(nvec)` 在 trace 期执行、只拼接表达式，即可支持任意 `nvec`，且 `nvec=1/2` 均编译通过。design skill 的 Phase 2 与 `algorithm-research.md` 应把这条“构建期展开 helper”形态写成**推荐写法**（而非只说“手写展开”）。 | `DESIGN.md#3.3` 注 2；`history_version/design_probe_compile_v2.py` F6（D_v=192、`nvec=2` 经 helper）COMPILE OK vs F7 FAIL | 在 design skill Phase 2 第 2 项或 `references/algorithm-research.md` §4 增补「**构建期展开 helper**：段数/分段归约在 trace 期用 python 循环拼接单一表达式；禁止在 prim_func 体内用 `range()` 承载向量累加器」。 | P |
| Design-flow | 平台条款里「整个 Buffer 拷贝要求元素总数相等、局部拷贝用显式 region」（`references/programming.md:37`）是**操作性**规定，但 design skill 的「非整除处理」小节只讲 padding/crop，没有把它落到 `T.copy(src_region, dst_region)` 的写法。设计者容易写出 `T.copy(src, o_ub)`（源 (R,D_v) → 目的 (R,DW)）。 | `DESIGN.md#3.3` 注 3；F3 FAIL / F2 OK | 在 design skill 的「非整除处理」与 `ascend-constraints.md` 增一行：「UB 行宽 pad（`DW > D_v`）时，两条搬入的目的端必须写显式 region `[0:R, 0:D_v]`，且 pad 列两侧都要 `T.clear`」。 | R |
| Design-flow | 「尾块用静态 `nlast`」这条要求在本任务里逼出一个额外结构决策：`nsp` 必须退化为 1（否则满 tile 循环上界在核间不一致）。skill 的尾块条款没有提示“静态 extent ⇒ 分核形状也要静态”的耦合。 | `DESIGN.md#5.2`、`#3.3` 注 4 | 在尾块处理条款补一句：「要求尾段 extent 静态化时，若分核切片数与 tile 数相关，须同时把分核数固定为尾块几何（本次取 `nsp = 1`），否则 loop bound 会重新变成运行期值」。 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **正面结论（补全 Stage 2 的负面结论）**：`nvec` 段合并的正确写法是「模块层 python helper + trace 期循环 + 单一嵌套表达式」。`row_dot(o_ub, do_ub, i, nvec)` 在 `nvec=1`（F1/F2/F4）与 `nvec=2`（F6，D_v=192 → DW=256）下全部 COMPILE OK，生成 `vadd(vadd(simd_inst::vmul(...` 的单一表达式；同样的逻辑写在 prim_func 体内（F7）仍是 `Immutable variable` 失败。 | D | `history_version/design_probe_compile_v2.py` F6/F7；`DESIGN.md#3.3` 注 2 | `cd examples/mla_dense_bwd/_delta_kernel && source ../../../.agents/platforms/toolchains/ascend/env-950pr.sh && ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python history_version/design_probe_compile_v2.py f6`（另 f7 为负例） | tilelang 0.1.15 + CANN 9.2.1；ascend-950pr-9579；仅编译（无设备） | design skill Phase 2 / `references/algorithm-research.md` |
| 归约分组等价性的**结构性恒等**：`vcadd(vadd(PART_EVEN, PART_ODD))`（实现）与 `vcadd(p[0:64]) + vcadd(p[64:128])`（v1 公式）在 `nvec==1` 下**逐位恒等**——因为平衡二叉树在 128 = 2×64 处正好有子树边界，`tree128 = tree64(前半) + tree64(后半)`。机器证据：`verify_equiv.py` 的 A2 与 A2b 两行 12/12 用例逐格相同（322 / 2798 / `ulp(cand,base)=2560`）。这条把「证据指向不一致」从“需要重测”降级为“可用恒等式解释”。 | D | `verify_equiv.py`（`impl_grouping` + A2b）；`DESIGN.md#1.6.1` 表与其下方注 | `cd examples/mla_dense_bwd/_delta_kernel && ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python verify_equiv.py` | tilelang 0.1.15；torch CPU 数值实验 | `references/algorithm-research.md`（D-1 归约分组口径） |
| **负面发现**：尾块几何（`S % R ≠ 0`）会**关闭 UB 多版本**——因为尾段的 `T.copy` 写在流水循环之外、却作用在同一 `o_ub/do_ub` 上，AutoSchedule 提示 “did not enable multi-buffering … written before it is read”。尾块几何因此没有流水重叠（性能，非正确性）。要保住流水须为尾段单独分配 `(nlast, DW)` buffer（结构变更）。 | D | `history_version/design_probe_compile_v2.py` F4/F8 的 warning；`DESIGN.md#6.3`、`#9.1` R10 | 同上（`… design_probe_compile_v2.py f4`） | 同上 | `pattern-library / bottleneck-patterns`（多版本启用的前提条件） |
| 尾块「全静态」的可编译形态：`nfull = S // R` 个满 tile 走 `T.Pipelined`，尾段独立成构建期段并使用 `nlast`（两条 copy + 行循环 + 写回）；`nfull == 0`（`S=1`）时 `if tpb > 0` 在 trace 期折叠、只剩尾段。三例（S=65、S=1、以及满 tile 几何）全部 COMPILE OK，生成 `asc_copy_gm2ub_align(…, 1, 256, …)` / `asc_copy_ub2gm_align(…, 1, 4, …)`。 | D | `DESIGN.md#3.3` 注 4、`#6.4`；F4/F8 | 同上（`… f4`、`… f8`） | 同上 | design skill 尾块条款 + `references/quality-checklist.md` |

### Transferable Lessons（可迁移教训）

- 「不可构建」的结论要升级成「**可以这样写**」：Stage 2 只说“必须构建期特化 + 手写展开”，本版才验证出可推广的 helper 形态（trace 期循环 + 单一表达式），并顺手证明 `nvec=2` 也成立——负面发现只有配上可用写法才对下游（Stage 3/4）有交付价值。
- 论证“证据与实现不一致”时，先查两者是否**结构恒等**（本例平衡二叉树的子树边界），能用一条恒等式解释就不必重编设计；解释不了再改实现或改证据。
- 分核形状与尾块 extent 是耦合的：一旦要求 extent 静态化，`nsp` 也必须静态到“不跨尾段”的取值（本次 `nsp = 1`）。写尾块方案时必须同时写它把分核退化到什么程度。
- revision 模式下“为什么不会再犯”：把每条修正落成 §9.2 的一行**错误条目**（症状 + 触发 + 正确写法），比在正文里多说一句更有效——下游读到的是可执行的禁令，而不是叙述。

## Stage 2 (design review) Retrospective — review_v2（完整重检 revision 1） — 2026-11-23

对象为 v2 设计（三条 v1 阻塞的闭环版本）。结论 `通过`（0 阻塞 + 9 建议）。本轮复盘只记录**检视流程自身**的缺口与
本轮亲手查证到的 API/编译行为，设计内容问题已随 `REVIEW.md` 流转给 Stage 3。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Review-flow | skill Phase 1 第 7 步只要求「重跑 `verify_equiv.py`，结果与 §1.6.1 内嵌表一致」。但脚本**不打印基线指标**（`ulp(base,ref)` 计算了却未输出），设计引用的「基线 2238 ulp」无法从输出比对——必须另写脚本调 `verify_equiv` 的 `sequential_sum/ulp_diff` 复算。机械复核的“可比对集合”与设计引用集合不一致时会静默放过未核对数字。 | `REVIEW.md#机械复核`（基线 2238/3310/64/189 独立复算）；`verify_equiv.py:268-272`（`u_base` 未打印） | 在 skill Phase 1 第 7 步补一句：「设计引用的每个数值都必须在脚本输出里可定位；不可定位的（如基线 ulp）须用脚本函数独立复算并在 REVIEW 里标注复算值」。 | R |
| Review-flow | 模板规定「结论通过时删除检视问题列表章节」，但「检视元信息」又要求填「建议级问题数」。通过时建议往哪写没有定义——本轮先写成「检视问题列表（建议级，不阻断）」后自行改名，属模板歧义。 | `REVIEW.md#建议汇总`；skill §5 模板 `## 检视问题列表` 注释行 | 模板补一个通过态章节 `## 建议汇总（非阻塞）`，并注明「不得使用『检视问题列表』标题以免与不通过态混淆」。 | R |
| Review-flow | 检视“闭环核验”只看 `COMPILE: OK` 不够：本轮必须 dump 生成代码才能发现两处与设计描述不符的事实（`d_ub` 被自动加版本；`T.clear` 实际覆盖两版本全量、不是单版本）。skill 对「探针证据复核」没有要求 dump 源码参数。 | `REVIEW.md` 建议 1/5；F2 全源码（`(it & 1) * 512 + 16384` floats、`for i in 4096 { 8 B store }`） | Phase 1 第 7 步增一条：「设计的编译期证据若引用生成代码（描述符参数、版本偏移、清零点数），须 dump 全文逐参数核对，不得只核 `COMPILE: OK`」。 | R |
| Info-source | `statectl platform` 的 `platform.design_budget.ub_inflation` / `ub_inflation_factor` 均为 `null`，设计只能引用 pattern-library 的 910B2C 膨胀常数（1.7× / 1.10–1.12×）。检视侧无法从平台上下文判断该系数是否适用于当前型号，只能核对常数条目本身。 | `statectl platform --task-dir examples/mla_dense_bwd/_delta_kernel` 输出 `"ub_inflation": null`；`DESIGN.md#4.5` | 平台侧在 `design_budget` 填 `ub_inflation` 条目（或显式标 `not-measured-for-this-model`），使 review 侧有一条可引用的判定依据。 | R |
| Review-flow | 上一轮（review_v1 流转 → Stage 1 revision）登记的一条 D 类价值点「尾块几何会关闭 UB 多版本，AutoSchedule 输出 did not enable multi-buffering 警告」，本轮**全量捕获 stderr/warnings 后无法复现**；且 F4/F8 的 `nfull ≤ 1` 使多版本本就不可观测。历史复盘里的 D 类条目缺“可复现性”复核环节。 | 本轮回执：`design_probe_compile_v2.py f4`/`f8` 输出无该警告；`REVIEW.md` 建议 2；`RETROSPECTIVE.md` Stage 1 revision 行 89 | evolver 蒸馏 D 类条目时要求「至少一次由**非作者 Stage** 独立复现」；review skill 增加一步：复核上一轮遗留 D 类断言的复现性，不复现则在 REVIEW 的建议里更正归因。 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **正面实测**：`T.clear` 作用在 `annotate_buffer_versions` 标注过的 UB buffer 上时，展开为**全量 footprint 填充**（`(64,128)` fp16 双版本 → `for i in 4096 { uint2 store }` = 32768 B），因此「循环前清零一次」对多版本 buffer 的 pad 列**是安全的**；此前“只清到版本 0、另一个版本未初始化 → `0·garbage`”的担心不成立。同时 `T.copy(src_region, dst_region)` 只写 `[0:R, 0:D_v]`，pad 列在两版本中都保持 0。 | D | `examples/mla_dense_bwd/_delta_kernel/REVIEW.md#机械复核`（clear 展开核对）；F2 全源码 | `cd examples/mla_dense_bwd/_delta_kernel && source ../../../.agents/platforms/toolchains/ascend/env-950pr.sh && ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python history_version/design_probe_compile_v2.py f2`（再 dump `get_kernel_source()`） | tilelang 0.1.15 + CANN 9.2.1；ascend-950pr-9579；仅编译（无设备） | `.agents/platforms/toolchains/ascend/knowledge/numerical-boundaries.md` 或 pattern-library `traps-*`（UB pad 清零口径） |
| **负面/预算口径**：显式 `annotate_buffer_versions({o:2, do:2})` 之外，编译器（MaterializeMultiBuffer）会给**流水循环体内**的其它 UB 暂存 buffer **自动加第二版本**——`d_ub(R,8) fp32` 128→256 floats（2048→4096 B），设计期手算 UB 预算会漏计小 buffer × 版本数（本次 66 KB 实为 68 KB）。 | D | `REVIEW.md` 建议 1；F2 生成代码 `(it & 1) * 512 + 16384`（vsts 与 ub2gm 同一偏移） | 同上（f2，`tpb=4` 使版本化可见；`tpb=1` 时不可见） | 同上 | `pattern-library/constants.md`（UB 预算口径）/ `traps-*` |
| **构建期展开的边界形态**：`for v in range(nvec)` 即使 `nvec == 1` 也会被提升为 IR `ForFrame`（IR dump 可见 `for v in range(1)`），跨迭代重绑定的向量值报 `Immutable variable`；把展开逻辑放到**模块层 python 函数**（trace 期执行、只拼表达式）后 `nvec=1/2` 全部可构建。 | P | `REVIEW.md#1`；F7 失败 / F6 通过 | `… design_probe_compile_v2.py f7`（进程 exit=1，需单独跑）与 `… f6` | 同上 | design skill `references/algorithm-research.md` §4 / develop skill（SIMD 段合并写法） |

### Transferable Lessons（可迁移教训）

- 复核“清零/初始化”类安全性时，必须 dump 生成代码确认**清零的字节数**：多版本 buffer 的 `T.clear` 可能被展开成“全部版本”的
  全量填充，也可能只清一版——差别决定 pad 路径是「零项不改和」还是「`0·garbage` 污染整行」，只读伪代码无法判断。
- 手算 UB 预算时按“版本数 × 每 buffer 大小”逐项列，且**版本数要以生成代码里的偏移为准**：除显式 `annotate_buffer_versions`
  外，流水体内的暂存 buffer 也会被自动加版本（本例 `d_ub` 2048→4096 B），漏计会让设计表与实测长期不一致。
- 引用他人（或上一轮）探针的**告警文本**作为机制证据时，先独立重跑并全量捕获 stderr：告警可能来自被展开/未触发的代码路径，
  本次「尾块几何关闭多版本」的告警在本轮完全不可复现，机制归因应降级为“未观测到”。
- 分核/尾块的静态化结论要连带检查“该几何下还能不能观测到多版本”——`nfull ≤ 1` 的探针天然没有流水，用它证明“流水被关闭/不关闭”
  是无效证据；要验证版本化交互，必须构造 `nfull ≥ 2` 且带尾段的几何（如 `S=131, R=64`）。

## Stage 3 (develop) Retrospective — first_impl (attempt 1) — 2026-11-23

产物：`_delta_kernel.py`（kernel `_delta_kernel(batch, heads, dim_v, dense_seqlen, dtype)` + 内嵌 `golden_delta` +
L0/L1/L2/boundary 分层用例 + `main --level`），真机跑测 **23/23 全过**（L0 6/6、L1 4/4、L2 4/4、boundary 9/9，
总 max_diff **7.63e-06**，目标容差 atol=rtol=1e-4）。设计 v2 的三条禁令全部照做（`nvec` 构建期展开、目的端显式切片 +
两侧 `T.clear`、尾段三处静态 `nlast`），实现层没有出现需要回退设计的阻塞项 → `[PRECISION_PASS]`。
冷启动全量耗时 62 s（19 次全新编译），编译缓存命中后 5.9 s。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Info-source | 平台 `execution.precision_command` 记录的解释器 `/usr/local/python3.12.16/bin/python3` 在本机**没有 torch/tilelang**（`ModuleNotFoundError: No module named 'torch'`），照执行上下文原样跑必然失败；实际必须用任务 venv `/workspace/venv-ascend950/bin/python`。技能只说"按 execution 命令执行"，没有"命令不可用时的降级与上报"口径，只能由 conductor 在 prompt 里口头补充。 | `_delta_kernel.py#L1` 头注释外的运行命令；本机实测两个解释器的 import 结果（`torch 2.13.0+cpu / tilelang 0.1.15` 仅在 venv 内） | 平台侧让 `statectl platform --check-environment` 校验 `execution` 里的解释器能 import torch/tilelang，不通过就把 `execution` 的路径改写为实际 venv（或直接报错而非静默记录）；develop skill Phase 4 补一句"execution 命令不可用时先上报，不得自行换解释器且不留记录"。 | R |
| Develop-flow | 编译器告警（`Ascend AutoSchedule did not enable multi-buffering ...`）**只在全新编译时出现**：命中磁盘 kernel cache 后 AutoSchedule 根本不跑，日志干净。第一轮 boundary 打出 4 组告警、第二轮同样几何 0 组，差异仅来自缓存。技能没有任何"复现编译期告警需 `tilelang.disable_cache()`"的提醒，直接导致 Stage 2 复核 F4/F8 时得出"告警不可复现"的假阴性。 | `repro/CONST-950-tail-no-multibuffer.py`（显式 `tilelang.disable_cache()` 后稳定复现）；REVIEW.md 建议 2 的"不可复现"结论 | develop skill Phase 2 与 review skill「探针证据复核」各加一条："要观测 pass 行为/告警，先 `tilelang.disable_cache()`；否则结论只能是『缓存命中，未运行该 pass』"。 | R |
| Develop-flow | eager builder 在 trace 时按**行号**从磁盘重读 prim_func 源码：进程运行中修改 kernel 文件会让工厂函数被误当 prim_func 追踪，报 `TypeError: make_closure.<locals>._delta_kernel.<locals>._delta_kernel() missing 5 required positional arguments`，整轮全量作废（本次发生一次，纯自伤）。技能与调试参考都没有"运行中不得改被测文件"的提醒。 | 本次 `--level all` 第一次运行（编辑 `_delta_kernel.py` 期间启动）的 traceback | develop skill Phase 4 加一句："分层跑测期间不得修改 `{op}.py`；eager builder 按行号重读源码，改动会让 trace 落到错误函数上（表现为参数缺失/莫名 TypeError）。" | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **`T.clear` 覆盖整个多版本分配且是 pad 几何的编译热点**：`(64,128)` bf16（`annotate_buffer_versions=2`）每个操作数展开为 **4096 × 8 B = 32768 B** 的 `PIPE_S` 展开标量存储循环（= 2 版本 × 16384 B，另各一条 `asc_sync_notify(PIPE_S, PIPE_MTE2)`），不清零的变体**一条 store 都不生成**；因此该 clear 是 `D_v=64` 几何编译 **27.0 s** 对 `D_v=128` **1.8 s（15×）** 的全部来源。设计 §4.5/REVIEW 建议 5 的"两侧清零 → pad lane 为 0"由此得到机制级确认。 | D | `_delta_kernel.py#L197-L203`（两侧 clear）、`#L0-3/L0-4` 真机 1.9e-06/3.8e-06；`repro/CONST-950-clear-pad-all-versions.py` 输出 | `cd examples/mla_dense_bwd/_delta_kernel/repro && source ../../../../.agents/platforms/toolchains/ascend/env-950pr.sh && python CONST-950-clear-pad-all-versions.py`（编译级，PASS） | tilelang 0.1.15 + CANN 9.2.1 / Ascend950PR_9579 / 无设备 | pattern-library `constants.md`（UB 预算与清零口径）/ `traps-*` |
| **尾段写在 `T.Pipelined` 之外会整体关闭该 buffer 的多版本**：S=131（`nfull=2, nlast=3`）流水装载的 UB 目的基址恒为 `[0]`/`[8192]`（无 `it *` 项）并打出 `did not enable multi-buffering` 告警；同一 kernel 去掉尾段（S=128）则为 `it * 8192`（双版本）、无告警。这是设计 R10 机制的第一份**可复现**证据（设计期 F4/F8 因 `nfull ≤ 1` + 缓存命中未能观测），性质为纯性能：数值仍正确（真机 3.8e-06）。 | D | `_delta_kernel.py#L217-L228`（尾段）；B-3/B-4/B-5/B-6 全部打告警且全过；`repro/CONST-950-tail-no-multibuffer.py` | `… python CONST-950-tail-no-multibuffer.py`（编译级，PASS：no-tail 2 条 versioned load、tail 0 条 + 告警） | 同上 | `PL-950-vector-execution`（尾块 + 多版本）/ `traps-*` |
| **`D_v=64 < DW=128` 的 pad 路径数值已在真机成立**：目的端切片 `[0:R, 0:D_v]` + 两侧 `T.clear` 即足够，无需 `update_mask`；含 pad 与尾段组合（S=65, D_v=64）也通过（1.9e-06）。op 自身 smoke 几何（`test_mla_dense_bwd_general_kernel`）因此在 Stage 5 前已有数值闸门。 | D | `_delta_kernel.py#run_L0`(L0-3/L0-4)、`#run_boundary`(B-5) | `… python _delta_kernel.py --level L0`（真机 PASS） | 同上（真机 1 卡） | 迁移参考 / develop skill（SIMD pad 路径写法） |
| **`nvec=2`（`D_v=192 > 128`）真机数值通过**（3.8e-06），闭合设计 §9.1 R11 的"仅编译证据"缺口：构建期 helper 把两段 `vadd` 拼成一个表达式、只做一次 `vcadd` 的写法在真机上正确。 | D | `_delta_kernel.py#run_boundary`(B-7)，`_row_dot` | `… python _delta_kernel.py --level boundary`（真机 PASS） | 同上 | develop skill（`nvec>1` 段合并）/ DESIGN R11 复核 |
| **含 frame 的复用片段必须 `@T.macro`**：把 SIMD 行循环抽成普通 python 函数在 prim_func 体内调用会报 `TypeError: 'ForFrame' object is not iterable`（eager builder 只重写 prim_func 自身源码，不重写被调函数）；纯表达式拼接的 helper（`_row_dot`）可以作普通函数。 | P | `repro/CONST-950-tail-no-multibuffer.py` 的 `_rows`（`@T.macro` + docstring 记录该失败形态）；主文件 `_row_dot`（普通函数，仅拼表达式） | 同一 repro 脚本即为正例 | 同上 | develop skill（SIMD 行循环复用）/ `PL-950-vector-execution` |
| **实测有效带宽 1.44–1.51 TB/s**（manifest 三 workload：`ds-v3-4k` 0.186/0.185 ms、`ds-v3-b2-8k` 0.719 ms、`kimi-k2-8k` 0.188 ms；按 `2·N·D_v·2 B + N·4 B` 计），**高于**设计 §1.6.0 的 1.03–1.26 TB/s（910B2C 外推），说明 H=128 的 32 KB 源跨距没有出现 R1 担心的跨步描述符带宽断崖。口径为 launch 前后 wall clock、repeat=2，属**下界**。 | D | `_delta_kernel.py#run_L2` 四个用例 | repro-missing（需 `msprof` Task Duration 交叉，留给 Stage 4） | 同上（真机 1 卡） | `bottlenecks.md` / constants（本型号带宽实测，替换 910B2C 外推） |

### Transferable Lessons（可迁移教训）

- 本迁移后续两个函数（`_bwd_kernel`、`_dq_land_kernel`）：只要某条代码路径在 `T.Pipelined` **之外**也写同一个 UB buffer
  （尾段、收尾修正、二次累加），该 buffer 的多版本会被整体关闭（装载地址不再随迭代变化，且打 `did not enable multi-buffering`
  告警）。依赖流水重叠的函数应给这类收尾段**另分配小 buffer**，不要复用主 buffer；也不要指望 `annotate_buffer_versions` 强开。
- UB 行 pad 到向量宽度（`D_v < 128`）的算子：目的端必须显式切片、**两侧**操作数都要清零；同时注意 `T.clear` 的代价是
  "整份分配"的展开标量存储（编译期 15×、运行期每核一次全量写），pad 列占比大或几何特殊时应优先 `update_mask(D_v, 32)`。
- 含 `T.serial`/`T.Pipelined` 的复用片段必须写成 `@T.macro`；只有**纯表达式** helper（例如多段 `vadd` 拼接、`nvec` 展开）才能作普通
  python 函数在 prim_func 体内调用——搞错的表现是 `ForFrame object is not iterable`，而不是类型错误提示。
- 复现编译器 pass 行为/告警前必须 `tilelang.disable_cache()`：命中磁盘缓存时该 pass 不运行，日志"干净"会被误读成"现象不存在"。
- 分层跑测或任何长跑进程进行中**不要修改被测 kernel 文件**：eager builder 按行号重读源码，改动会让 trace 落到错误函数上
  （表现为 `missing N required positional arguments`），整轮结果作废。
- 0.2–0.7 ms 级 kernel 用 launch 前后 wall clock + repeat 得到的带宽只能当**下界**用来排除数量级错误（本次 1.44–1.51 TB/s 与
  roofline 估算一致），精确归因仍需 `msprof` Task Duration。
