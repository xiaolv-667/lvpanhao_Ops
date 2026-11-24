## Stage 1 (design) Retrospective — first_design — 2026-11-23

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Constraint-doc | 设计 skill 的 §1.6.1 模板默认「优化项 = 相对源实现的改进」，但没有区分「相对源的优化」与「源在 Python 工厂里本来就有的专用化」。本任务初稿把 `split_k==1` 的构建期消去写成 A2 优化项（收益 12→3 条指令/chunk），复核时发现源 `_slice_sum` 的 Python 循环**本来就专用化**，该收益相对源为 0，属「不变量保持」而非优化 | `DESIGN.md` §1.6.1 A2 行（v1 修正后已按「结构性不变量 + 回归守卫」重写）；源 `dense.py:2015-2022` 的 `_slice_sum` 为 Python 循环 | 在 `tilelang-op-design` 的 Phase 2 第 1 步「数学等价优化」表中加一行判定：采纳项必须区分「相对源算子的改进」与「迁移必须保持的性质」，后者记为不变量、不进收益列 | R |
| Info-source | 设计期 roofline 与实验裁决都要用「每核有效带宽 / 向量发射速率」，但 950PR 两台机型的平台 `cost_measurements` 为空、pattern-library `constants.md` 全部绑定 910B2C，导致 D-2 只能写「待核验」。本任务因此把设计重点从「指令数优化」转向「流量下界 + 单 launch」——这是对的，但每个 950PR 任务都要重复一次该推导 | `DESIGN.md` §1.6.0「设计期估算（D-2）」；`.agents/tools/statectl.py platform --task-dir` 返回 `cost_measurements: []` | 在 `_shared/standards/hardware-cost-model.md` 增加「无本机型实测常数时的标准降级模板」：① 流量项必算 ② 用 910B2C 常数外推并显式标注 ③ 结论只写「与常数无关的排序」 | P |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 950PR SIMD：`vcvt` 不支持 fp32→fp32（`ValueError: Unsupported vcvt conversion float32->float32`），fp32 输出必须跳过 vcvt、用 `vsts(dist="NORM_B32")` 直存 | D | `DESIGN.md` §3.4 表「`T.simd.vcvt(fp32→fp32)`」行；探针 P5 首轮 FAIL 后修正 | `cd examples/shared_expert_mlp/splitk_reduce_kernel && ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python history_version/design_probe_compile.py p5` | tilelang ascend 0.1.15 + CANN 9.2.0 SDK；device-free 编译；ascend-950pr-9579 | `.agents/platforms/toolchains/ascend/knowledge/numerical-boundaries.md` |
| 950PR SIMD：`vdiv` 默认降为 `vdiv_0ulp_ftz_true`（精确 0-ulp + 次正规 FTZ），即使未开 fast-math | D | 探针 P2 生成代码 `simd_inst::vdiv_0ulp_ftz_true(g[0], den, mask, MODE_ZEROING)`；`DESIGN.md` §9.1 R2 | 同上，`... design_probe_compile.py p2` | 同上 | `.agents/platforms/toolchains/ascend/knowledge/numerical-boundaries.md` |
| 950PR SIMD：`vsts(dist="PK_B32")` 写 16-bit UB 后**直接** `T.copy` 到 GM 是逐字节正确的（无需 `mem_bar`）；只有「向量写 → 向量读」才需要 `VST_VLD` 屏障。归约/epilogue 类 kernel 应刻意设计成「只写不读」以躲开该屏障 | P | `examples/engram_gate_conv/_engram_gate_conv_fwd_kernel/repro/TRAP-950-simd-ub-store-load.py` 断言 A；本设计 §7.2 / §3.3 | `repro_runner` 跑该 repro（engram 任务已固化） | tilelang ascend 0.1.15 + CANN 9.2.1；ascend-950pr-9579 | `.agents/platforms/toolchains/ascend/knowledge/numerical-boundaries.md`（与既有 TRAP 条目并列） |
| 迁移设计的「等价性机器验证」必须先把 ulp 度量口径写死：以**输出 dtype 的网格**度量、尺度下限取该 dtype 的**最小正规数**；否则 bf16 次正规区（1e-35 量级）会把指标放大成 1e8 ulp 的无意义值，把正确改写误判成 EQUIV_FAIL | P | 本任务 v1 的 `verify_equiv.py` 首轮输出（`max ulp = 100663296`、34 例误判 FAIL）→ 修正后 110 例 0 失败；`DESIGN.md` §1.6.1「ulp 以输出 dtype 的网格度量且尺度下限取最小正规数」 | `cd examples/shared_expert_mlp/splitk_reduce_kernel && /workspace/venv-ascend950/bin/python verify_equiv.py` | 无关工具链（torch CPU 口径） | `.agents/skills/tilelang-op-design/references/algorithm-research.md`（或 skill Phase 2 第 5 步） |

### Transferable Lessons（可迁移教训）

- 在 950PR 上写纯向量 kernel：`T.SimdVF` 块内**不能**访问 GM（`BufferLoad ... inside a SIMD_VF block is not allowed`），所有 GM 读写必须先 `T.copy` 到 UB；跨 `T.serial` 迭代携带向量累加值必须用 `T.simd.alloc_local`，不能用 Python 变量。
- `vcvt` 的 `part` 必须是构建期常量；`vcvt` **没有** fp32→fp32，fp32 输出直接 `vsts(dist="NORM_B32")`。
- 16-bit 输出存 UB 用 `vsts(dist="PK_B32")`、读用 `vld(dist="UNPK_B16")`；`PK_B32` 之后**直接** `T.copy` 到 GM 不需要 `mem_bar`，但**同一 UB 缓冲的向量写→向量读**必须 `T.simd.mem_bar("VST_VLD")`（run-to-run 不确定）。归约类 kernel 应把缓冲设计成单向往返以躲开它。
- 统计 `ulp` 差值时先在**目标 dtype 网格**上定义度量（尺度下限取该 dtype 的最小正规数），否则次正规区会把好的改写判成失败；fp32 层的「足迹」只能作观测列，不能作判据。
- 迁移「源在 Python 工厂里已专用化」的算子（`split_k==1`、无 bias、无激活等分支）时，专用化是**必须保持的不变量**而非优化项——收益列要写清是相对「直译构建」还是相对「源实现」。
- 上游 split-K 归约类 kernel 的 `partial` 布局（`[split_k, m, n]` fp32）是跨 kernel 契约，写设计时必须双向引用（本文件与上游 GEMM 的 DESIGN 互相标注），否则 Stage 3 的任何一侧改动都会静默破坏另一侧。

## Stage 2 (review) Retrospective — pass — 2026-11-23

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Review-flow | 机械复核规则要求「重跑结果与 §1.6.1 内嵌表**一致**」，但未区分**判据列**（max ulp / 违反率 / 非有限 / 对 fp64 误差 / verdict）与**观测列**（fp32 足迹）。本任务观测列有一格偏差（内嵌写 ≤2、重跑为 3、§11.2 正文写 ≤3），按字面应判「不一致 → 维度 8 fail」，但判据列全部复现、等价性结论不受影响——字面规则会产出阻塞级误判 | `examples/shared_expert_mlp/splitk_reduce_kernel/REVIEW.md`「机械复核」段；`DESIGN.md` §1.6.1 A1 行 vs §11.2 | 在 `tilelang-design-review` skill Phase 1 第 7 步把「一致」拆成两级：判据列不一致 → fail；观测列不一致 → 建议级且必须在机械复核段显式记录 | R |
| Review-flow | 维度 8 只检查「脚本重跑全 PASS」，不检查**脚本用例是否真的构成对照**。本任务 `verify_equiv.py:191-197` 的 A2 检查把 `generic` 与 `spec` 写成同一个表达式（`(w + bias.float()).to(dtype)`）、`copy_gen`/`copy_spec` 同为 `w.to(dtype)`，`torch.equal` 必然为真——`[EQUIV_ALL_PASS]` 对 A2 提供了虚假支撑，需读脚本正文才能发现 | `verify_equiv.py:182-203`；REVIEW.md 维度 8 建议 2 | 在维度 8「等价性论证正确性」增加子项「脚本用例有效性抽查：候选式与基线式是否为两个不同表达式/两个不同构建；`equal(x, x)` 型自比较判为证据缺失」 | R |
| Review-flow | `design_calc_check.py` 对**纯 Vector、无 Cube、无 per-level 表格**的设计 4 项里 skip 3 项（`ub_budget`/`l0c_budget`/`core_split`），实际只剩只验存在性的 `r3_metrics`——「可机检子集」在本设计类别上几乎为空，全部落到人工手算 | `design_calc_check.py --design …` 输出（REVIEW.md 机械复核表）；本设计 §4.5 是「配置 × buffer」行表、§5.5 是散文式三要素 | 给 `design_calc_check.py` 增加两种行解析：① §4.5 的「配置 \| g_ub \| u_ub \| b_ub \| o_ub \| 合计」行 → 逐行加总校验；② §5.2 的「BC 规则 + 每 workload 行」→ 重算 `ceil(grid/物理核)·BC` 并比对；skip 时输出「必做手算模板」清单 | P |
| Info-source | 本机 `/workspace/tilelang-mlir-ascend-950` 检出目录缺 `build/` 且无 `tilelang/ascend`，在该目录内 `import tilelang` 走 dev-root 必失败；探针必须在**仓库外 cwd**（如 `/tmp`）以 site-packages 的 tilelang 0.1.15 运行。任何「重跑探针」的检视者都会先踩一次 | `cd <repo> && python -c "import tilelang"` → `ImportError: TileLang's private TVM Python package is missing`；`cd /tmp` 后同命令成功（tilelang 0.1.15） | 在 `references/evidence.md`（或平台 notes 的「站点 profile 偏离声明」）记一条：无 dev build 时探针/复核的运行 cwd 约定 | P |
| Info-source | `vcvt(..., sat=False)` 对 **f32→f16/bf16** 的 ±inf / 溢出语义在 `simd.py` docstring 里只有一句「out-of-range values wrap」（该措辞更像描述整型窄化），float→float 的非饱和行为无明确条款；设计据「`sat=False` 保持 ±inf」立论，检视侧无法用文档证实或否证，只能靠 L0-9 兜底 | `tilelang/ascend/language/simd.py:1241-1246`；`DESIGN.md` §0.6 R5 / §9.1 R3 | 在 `knowledge/numerical-boundaries.md` 补一条待实测项：「`vcvt` f32→f16 的 `sat=True/False` 对 ±inf 与上溢的分别行为」 | P |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 检视迁移类 Vector kernel 时，**用探针生成代码做字节级交叉核对**能同时验证三件事并抓出证据归因错误：① UB 偏移（本任务 p1 的 `bias@half[8192]`/`o_ub@half[9216]` → 20 480 B，与 §4.5 最坏配置逐字节相同）；② DMA 描述符形态（`asc_copy_gm2ub_align(..., 1, 4096, ...)` 证实「单条 4096 B 描述符」而非 1024 次标量读）；③ 写后读路径（`o_ub` 只出现在 `vsts_pack_b32` 与 `asc_copy_ub2gm_align`） | P | 本任务 REVIEW.md 维度 2/5/8；`DESIGN.md` §4.5 的「annotate_buffer_versions 关掉多版本后 UB 可静态核算」被证明归因不准（探针未调用该 API 也已逐字节吻合） | `cd /tmp && ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python <repo>/examples/shared_expert_mlp/splitk_reduce_kernel/history_version/design_probe_compile.py p1`（同在仓库外 cwd，否则 dev-root 导入失败） | tilelang ascend 0.1.15（site-packages）+ CANN 9.2.1；device-free 编译；ascend-950pr-9579 | `.agents/skills/tilelang-design-review`（Phase 1 检视方法） |
| 判「是否存在 UB 向量写→向量读（VST→VLD）竞态」的判定式：在生成代码里检查同一 buffer 是否**既**是 `vsts*` 的目标**又**是 `vld*`/`vlds_*` 的参数；只有一侧出现（写侧 `vsts` + 读侧 `asc_copy_ub2gm_align`）即安全，因为 `PIPE_V→PIPE_MTE3` 事件由编译器插入。逐 chunk 写不重叠区域也不构成依赖 | P | 本任务 REVIEW.md 维度 5 写后读专项；`vsts_pack_b32` → `asc_sync_notify(PIPE_V, PIPE_MTE3)` → `asc_copy_ub2gm_align`；对照 `TRAP-950-simd-ub-store-load.py` 断言 A | `cd /tmp && … design_probe_compile.py p2`，在生成源码里检索 `vsts`/`vld` 与缓冲名 | tilelang ascend 0.1.15 + CANN 9.2.1；ascend-950pr-9579 | `.agents/skills/tilelang-design-review`（维度 5 检视点）+ platform `numerical-boundaries.md` |

### Transferable Lessons（可迁移教训）

- 检视报告的「机械复核」段必须写清**哪几项是 skip + 自己怎么手算的**：纯 Vector 设计的 `design_calc_check` 会 3/4 skip，UB 逐配置加总与「三候选 `ceil(grid/核数)·BC` 枚举」都得人工做，并把复算数字写进报告（否则门禁无从判断你是否只是转了设计的自述）。
- 内嵌结果表与重跑比对时，先把表的列分成「判据列 / 观测列」；只有判据列的不一致才是阻塞级，观测列的偏差记录为建议级——否则会为 1 ulp 的观测笔误拦下一个正确的设计。
- 等价性脚本要抽查**用例有效性**：`equal(expr, expr)` 型自比较（本任务 A2 就是这样）或候选式/基线式实际同构，都等于没有验证；读脚本正文 30 秒，比相信 `[EQUIV_ALL_PASS]` 可靠。
- 迁移检视中「无写后读 UB 路径」这类断言不要接受文字结论：直接在探针生成代码里查同一 buffer 是否同时出现在 `vsts` 目标与 `vld` 参数里，再看编译器事件对（`PIPE_V→PIPE_MTE3`）；本任务结论成立，但成立的理由是「编译器已插 V→MTE3 事件 + 无 V→V 回读」，两者都要说清。
- 检出目录无 dev build 时，跑任何 tilelang 探针都必须先 `cd` 到仓库外（本机 `/tmp`），否则 dev-root 导入会以「private TVM Python package is missing」失败——这是环境事实而非设计缺陷，别写进设计问题列表。
- 上下游两个 DESIGN 对同一约束（如「`m < 16` 时 `split_k=1` 由谁回落」）的表述要交叉读：一侧写「kernel 内强制」、另一侧写「调用方修复」，虽等价但会让 Stage 5 只改一侧，检视时应作为一致性建议提出。

## Stage 3 (develop) Retrospective — first_impl (attempt 1) — 2026-11-23

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Develop-flow | 本 skill 的 `{op}.py` 结构规范只要求「kernel 构建入口与源工厂同名」，但**没覆盖「`T.prim_func` 形参集合随构建期开关变化」**这一形态（源算子 `has_bias` 时 bias 才出现在签名，DESIGN §3.3/E8 也如此要求）。实现时想用 helper 复用函数体，被 eager builder 连拒两次（`'ForFrame' object is not iterable` / `'Buffer' object does not support item assignment`），只能按变体各写一份 ~70 行函数体；这条约束（含「值表达式 helper 可以、`T.simd.pset` 不行」的边界）在动手前无法从 skill 或 platform `programming.md` 得知 | `splitk_reduce_kernel.py:179-310`（两份 `main`），`repro/TRAP-950-prim-func-helper-body.py`（5 项断言复现），`RETROSPECTIVE.md` 本段 | 在 `tilelang-op-develop` 的 `{op}.py` 结构规范里加一行：形参随构建期开关变化时按变体各写一份 `@T.prim_func`，函数体不得抽到普通 helper（附 repro 路径） | R |
| Develop-flow | skill 模板（`templates/op_template.py` 与 platform `kernel_template.py`）的 `main` 只有 `--level L0 / all`，**没有确定性判据**；而本平台 SIMD kernel 的已知陷阱（`TRAP-950-simd-ub-store-load`）恰恰只会以「run-to-run 不确定」的形式暴露，普通 `assert_close` 单次跑测对它是盲的。本次是照 `engram_gate_conv` 已通过 kernel 的手工先例补的 `repeat=N + torch.equal` 位比较（7 个配置 × 3 次 launch） | `splitk_reduce_kernel.py:463-476`（repeat + `torch.equal`）；L0-1/L0-2/L0-6/L0-7/L1-2/L1-4/L1-10 的 `repeat=3` | 在 kernel 模板的 `run_case` 里内置 `repeat=3` 位比较（默认对 16-bit 输出开启），而不是留给实现者记忆 | P |
| Info-source | 「用哪个 Python 跑真机精度」有两个来源且不一致：platform `execution.precision_command` 给 `/usr/local/python3.12.16/bin/python3`，而任务环境（调度 prompt / 设计期探针记录）要求 `/workspace/venv-ascend950/bin/python`。两者绑定不同 CANN/tilelang 组合，选错会得到另一套编译行为，且本机只有 1 张卡、不适合试错 | `statectl platform --task-dir` 返回的 `execution.precision_command`；`DESIGN.md` §11.1 的探针运行方式；REVIEW.md「机械复核」的环境说明 | 让 platform `execution` 显式指向任务 venv（或在其 notes 里写明「精度命令仅示意，实际用 `<venv>`」），并在 develop skill Phase 4 引用该唯一来源 | P |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 950PR eager `@T.prim_func`：函数体**不能**抽到普通 Python helper——`for ... in T.serial()` → `'ForFrame' object is not iterable`；`with T.serial() as i` + 缓冲赋值 → `'Buffer' object does not support item assignment`；`T.simd.pset()` 在 helper 内 → `Unresolved call ir.Op("tl.simd.pset")`；但返回 `vld/vcvt/vmul/vadd` 值表达式的 helper 可以。形参随开关变化的 kernel 必须按变体各写一份函数体 | D | `splitk_reduce_kernel.py:150-153`（`_upconvert` 是允许的形态）与 `:179-310`（两份 `main`）；错误逐字见复现脚本断言 | `cd examples/shared_expert_mlp/splitk_reduce_kernel && source .agents/platforms/toolchains/ascend/env-950pr.sh && ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python repro/TRAP-950-prim-func-helper-body.py` → `TOTAL checks: 5, failures: 0` / `[TRAP_CONFIRMED]` | tilelang ascend 0.1.15（site-packages）+ CANN 9.2.1；device-free 编译；ascend-950pr-9579 | `.agents/platforms/toolchains/ascend/knowledge/numerical-boundaries.md`（与既有 TRAP 条目并列）或 `references/programming.md` |
| 950PR split-K reduce 的数值实测：A1（`g·σ(g)·u → (g/(1+e^{-g}))·u`）+ `vcvt(ROUND_R, sat=False)` + `vsts(PK_B32)` 使 **16-bit 输出与「升序左结合 + 链尾 bias + 一次舍入」的 CPU golden 逐位相同**（23 个 16-bit 阻塞用例中 21 个 0 diff / 0 ulp；例外为 bf16 m=512 门控 1 ulp、门控 fp32 输出 2 fp32 ulp = 4.77e-7，均远在 1.6e-2 / 1e-5 容差内）。manifest 几何的整 kernel 墙钟 launch 6–8 µs（1.6–4.1 MB 流量） | D | `DESIGN.md` §8.2/§1.6.1 A1；实测输出 `L0-1`…`L0-12`、`L1-1`…`L1-10`、`L2-2`、`B-8` 行 | `cd examples/shared_expert_mlp/splitk_reduce_kernel && source .agents/platforms/toolchains/ascend/env-950pr.sh && ASCEND_RT_VISIBLE_DEVICES=0 /workspace/venv-ascend950/bin/python splitk_reduce_kernel.py --level all`（`max_diff` / `ulp` / `launch=` 列） | tilelang ascend 0.1.15 + CANN 9.2.1 runtime；真机 1×Ascend950PR_9579（cube 28 / vector 56） | pattern-library `constants.md`（给 950PR 补 `CONST-vector-launch-overhead` 的本机型实测） |
| 只要每个 UB 缓冲「单向使用」（`g/u/b` = DMA 进 + `vld` 读；`o_ub` = `vsts` 写 + DMA 出），`splitk_reduce` 形态的 kernel **不需要** `T.simd.mem_bar("VST_VLD")`：7 个配置 × 3 次 launch 位比较全一致；生成代码里 `o_ub` 只出现在 `vsts_pack_b32`/`vsts_norm` 与 `asc_copy_ub2gm_align` 之间，且自动插入 `PIPE_MTE2→PIPE_V` / `PIPE_V→PIPE_MTE3` 各一对 | D | `splitk_reduce_kernel.py:463-476`（repeat 判据）；生成代码 marker 统计（`vsts_pack_b32` 1 次、`asc_copy_ub2gm_align` 1 次、`asc_sync_notify` 2 次）；对照 `engram_gate_conv/.../repro/TRAP-950-simd-ub-store-load.py` 断言 A | `cd examples/shared_expert_mlp/splitk_reduce_kernel && ASCEND_RT_VISIBLE_DEVICES=0 /workspace/venv-ascend950/bin/python splitk_reduce_kernel.py --level all`（L0-1/L0-2/L0-6/L0-7/L1-2/L1-4/L1-10 均 `repeat=3`） | tilelang ascend 0.1.15 + CANN 9.2.1 runtime；真机 ascend-950pr-9579 | `knowledge/numerical-boundaries.md`（作为 `TRAP-950-simd-ub-store-load` 的第二个正向证据） |

### Transferable Lessons（可迁移教训）

- 在 950PR 上，若 `T.prim_func` 的形参集合依赖构建期开关（如可选 bias），**按变体各写一份完整函数体**：eager builder 的函数体不能调用普通 helper 做循环或缓冲赋值（`'ForFrame' object is not iterable` / `'Buffer' object does not support item assignment`），`T.simd.pset` 也必须在被装饰函数的调用点创建；只有「拿到调用方传入的地址/掩码、返回 `vld/vcvt/vmul/vadd` 表达式」的值 helper 是安全的（repro：`repro/TRAP-950-prim-func-helper-body.py`）。
- `out_idx=[]` 的 kernel 把输出当普通形参：harness 必须自己 `torch.empty(..., device="npu")` 后追加到参数表调用，函数**不返回**任何张量；把 `.cpu()` 当作同步点之外还要 `torch.npu.synchronize()`，否则位比较可能读到未完成的写出。
- 归约/epilogue 类 SIMD kernel 应把每个 UB 缓冲设计成单向（输入侧 DMA 进 + `vld` 读，输出侧 `vsts` 写 + DMA 出），这样既不需要 `mem_bar("VST_VLD")`，也能让「同输入连跑 3 次位比较」稳定通过；把这条位比较写进 `run_case`，它是该平台唯一能廉价发现写后读竞态的判据。
- 16-bit 输出的 split-K 归约在「同序累加 + 链尾 bias + 一次 RNE 舍入」下能与 CPU golden 逐位相同；一旦出现非零 diff，先怀疑累加顺序/舍入点被改（而不是先放宽容差）。fp32 输出（跳过 `vcvt` + `NORM_B32`）允许几个 fp32 ulp 的差（实测门控路径 2 ulp），因为 `silu` 的除法改写只保证输出 16-bit 网格 0 ulp。
- 上游部分和布局 `[split_k, m, n]` fp32 + 「`C` 由调用方预分配」是跨 kernel 契约：reduce kernel 的测试必须直接构造该布局（不要复用 GEMM 的中间张量语义），并用 `golden` 的 `w[0]+w[1]+...` 升序左加复算，否则链序错误会被容差掩盖。
