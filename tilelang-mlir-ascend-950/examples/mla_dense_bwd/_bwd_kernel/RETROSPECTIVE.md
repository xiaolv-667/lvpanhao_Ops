## Stage 1 (design) Retrospective — first_design — 2026-11-23

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Info-source | 平台的 GEMM 已核验用例只覆盖显式 L0 路径，但 `gemm_mad.py` 同时存在 L1-操作数路径（NT-only，含 assert）；design skill 的「API 证据」清单没有要求区分「源码存在 / 已核验」的**路径级**差异，容易把前者当作后者 | DESIGN.md §3.4「T.gemm(A_l1,B_l1,C_l0c)」行；`.agents/platforms/toolchains/ascend/knowledge/PL-950-cube-tile.md` 与 `repro/PL-950-gemm-tiles.py` 均用 `alloc_l0a/l0b` | 在 `references/info-sources.md` 增一条：GEMM 类算子必须记录所走 lowering 分支（L0 / L1）及其核验状态 | R |
| Design-flow | Phase R 的 R4 要求 roofline 定量，但平台 `cost_measurements` 为空且 `bottlenecks.md` 明确禁止用 910B 常数补齐 → 只能写「待核验」；缺少「在常数缺失时至少给出**结构主导项**排序」的显式指引，本次自行补了「L1 端口操作数流是地板项」的结构结论 | DESIGN.md §1.6.0「设计期估算」段；`.agents/platforms/toolchains/ascend/knowledge/bottlenecks.md`「尚缺的真机材料」 | `_shared/standards/hardware-cost-model.md` 增加：常数缺失时的降级输出格式 = 流量/发射/容量的**公式 + 主导项排序**（而非仅「待核验」） | R |
| Design-flow | migration-analysis.md §5.3 要求「按任务平台 ruleset 核对能力映射」，但本平台 950 相对 910B 的一处关键能力变化（`store_fixpipe` 只能 L0C→GM → 950 有 L0C→UB 双目标搬运）只在 pattern-library 的 910B 常数条目里以反面形式出现，平台 knowledge 无正面条目 | DESIGN.md §1.6.3 候选 5 行；`.agents/skills/tilelang-op-optimize/references/pattern-library/constants.md` `CONST-store-fixpipe-gm-only`（device: 910B2C） | 平台 knowledge 增一条 950 条目：`L0C→UB dual_copy` 的能力边界（同 dtype、2:1 分片），并标注它取代了 910B 的 GM 往返约束 | D |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| L1-操作数 GEMM 路径的 NT-only 硬约束（`trans_A=False, trans_B=True`）+ 由此导出的「转置副本」设计模式 | D | `tilelang/ascend/op/gemm/gemm_mad.py:GemmMAD.lower`（assert 文本）；DESIGN.md §0.6 R3 | `grep -n "currently only supports" .../ascend/op/gemm/gemm_mad.py` | tilelang 0.1.15 / CANN 9.2.1 / ascend-950pr-9579 | pattern-library（Cube 布局条目） |
| 注意力反向的「单遍必有一侧原子放大」结构结论：key-stationary 的 `dq` 原子流量 ∝ `S_kv/block_n`，query-stationary 换成更宽的 `dK+dV`，两 kernel 拆分 FLOPs×2 —— 选 key-stationary 的定量依据 | P | DESIGN.md §1.6.0 R3 表（C0–C3 行）与 R4 | none（纸面推导，代表 workload ds-v3-b2-8k） | — | `_shared`? → pattern-library（attention 族的原子放大口径） |
| `T.exp` 已核验 / `T.exp2` 未核验 → 「log2 域折叠」在 NPU 上是**负收益**迁移项；改用自然域 `exp`（机器验证 1 ulp） | D | `verify_equiv.py` 输出（A1 两行 EQUIV_PASS）；DESIGN.md §1.6.1 A1 | `cd examples/mla_dense_bwd/_bwd_kernel && /workspace/venv-ascend950/bin/python verify_equiv.py` | tilelang 0.1.15 / CANN 9.2.1 | pattern-library（exp 指令选择）+ knowledge/PL-950-softmax-boundaries 交叉引用 |

### Transferable Lessons（可迁移教训）

- 混合（Cube+Vector）算子设计前，先确定**每个矩阵乘的 NT 形式**是否可用：本平台 L1-GEMM 只允许 `trans_A=False, trans_B=True`，凡需要转置操作数的乘法都要另造一份 L1 副本（GM→L1 的 `transpose=True`，或 SIMT 内转置写后 UB→L1），代价要提前计入 L1 预算与迭代流量。
- L0C 是唯一可累加层且容量 256 KB：**跨块常驻的累加器**（如 dK/dV）先按 `block_n × 宽 × 4 B` 定死，剩余容量才轮到每迭代的临时累加器；`dq` 这类每迭代清零的累加器是唯一可压缩项。
- 平台 knowledge 只核验过「显式 L0 路径的单 tile GEMM」与「单 tile 混合 epilogue」；任何「多阶段混合循环」都是未核验组合，设计必须自带分级探针（2-GEMM 链 → 单块全链 → 加原子 → 加 mask/尾块）与回退阶梯，而不是等 Stage 3 一次性试。
- `T.dual_copy` 是 L0C→UB 在混合 kernel 里的唯一通道（同 dtype、2:1 分片）；需要按 AIV 半区索引数据时用 `with T.Vector(vector=2) as sid` 取 subblock id —— 但「显式 sid + 自动调度」的组合未核验，需在最小探针里先验。
- 张量是 `(T,H,D)` packed 布局时，固定 head 的切片是**跨步 region**（行 stride = H·D）。这类 region 的 DMA 行为在本前端未核验，凡用它（装载/写回/原子 store）都要列为首要探针项。

## Stage 2 (design review) Retrospective — mla_dense_bwd/_bwd_kernel — 2026-11-23

结论：`不通过`（阻塞 2 项：L1-GEMM 动态 region extent、显式 `T.Vector` 的 `dup sid`）。下列为**检视流程自身**的问题与可复用产出，设计内容问题已在 REVIEW.md 流转。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Mechanical-check | `design_calc_check.py` 对本案跳过 `ub_budget` 与 `core_split`（"no per-level buffer tables parsed" / "insufficient parsed dims"），但 §4.5/§4.6 与 §5.5 都是规范 markdown 表（含「合计/逻辑核数」行）；两份产出全靠人工复算，机械拦截位形同虚设 | `design_calc_check.py` 输出 `checks[0].status=skip`、`checks[2].status=skip`；DESIGN.md §4.5（`Buffer｜Shape｜dtype｜大小 (Bytes)`）、§5.5①（`workload｜B｜H｜S_kv｜ceil(S_kv/64)｜逻辑核数`） | 解析器支持「带 `合计` 行的字节表」与「表名含 `逻辑核数` 的核数表」，按列名识别 Bytes/逻辑核数，避免每案人工兜底 | R |
| Dimension-gap | 维度 1「API 存在性」只按符号是否存在于模块判定；本案两个阻塞问题都是「符号存在、**形态**被 lowering/SSA 拒绝」（`T.gemm` + 符号 region extent；`T.Vector` + `dual_copy` + UB→L1），静态查签名永远发现不了 | REVIEW.md 问题 1/2 的探针输出；`gemm_op.py:90-92`、`pipeline.py:63-66` | 维度 1 拆三级：存在性 / 形态可用性（可用最小 `tilelang.compile` 探针）/ 运行期正确性；并把「主结构依赖未核验路径时，至少对每种风险形态跑一次编译探针」写入 Phase 1（`ASCEND_RT_VISIBLE_DEVICES=""`，只编译不执行） | P |
| Evidence-scope | 设计文档把「本文档未核验」写成风险登记即可通过检视，但风险登记**不等于**可行性证据；本次 §9.1 风险 #1/#3 的回退阶梯本身也依赖同一不可用形态（`pT_lo/pT_hi` + 半区 GEMM 仍需 sid），属「回退阶梯不可执行」这一新型错误形态，现有维度表无专门检查项 | DESIGN.md §9.1 风险 #1/#3 的「处置/回退」列；REVIEW.md 问题 2 | 维度 1/4 增一项：**回退阶梯可执行性**——每条回退必须给出与主选不同的、已被证据支撑的形态，且不得复用被证否的机制 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| `T.gemm` L1-操作数路径的 M/N/K 取自 **region extent** 且必须静态（符号量抛 `ValueError: T.gemm requires static tile dimensions`）；显式 L0 路径则读 allocation 形状、region extent 可符号（`prove_equal` 即可）——两条路径的静态性契约不同 | D | `ascend/language/gemm_op.py:77-92`（`_gemm_dense_slots`）、`gemm_op.py:27-45`（`_l0_gemm_call` 注释）、`op/gemm/gemm_mad.py:151-165` | `/tmp/opencode/rv/probe_dyn.py`（`COMPILE_FAIL dyn=True` / `COMPILE_OK dyn=False`） | tilelang 0.1.15 / CANN 9.2.1 / ascend-950pr-9579 | 平台 knowledge（gemm-tiles / cube-tile 条目补「L1 路径的静态 extent 契约」） |
| 显式 `T.Vector(vector=2) as sid` 与 `dual_copy` + UB→L1（或第二个 GEMM）组合在 0.1.15 抛 `InternalError … dup sid`；自动调度会自合成 cthread sid（用户不可见），整 tile `T.copy`(L0C→UB) + `SimtVF` 可替代「半区 + sid」写法 | D | `ascend/pipeline.py:63-66`（RewriteDualCopy 注释）、`language/copy_op.py:60-92`、`language/frame.py:35-63` | `/tmp/opencode/rv/probe_vec3.py`（`vec_ub_l1_dual_nogemm` FAIL vs `auto2` OK）、`probe_full.py`（整 tile OK） | 同上 | 平台 knowledge/execution-patterns（混合 kernel 的 sid 使用边界） |
| 风险优先检视法：主结构押在未核验 lowering 路径时，先写 3–5 个最小编译探针把「不可编译 / 可编译 / 运行期待验」分开，再决定阻塞等级——比逐句读设计自述+风险登记更快命中阻塞项 | P | 本次 7 个探针把 2 个阻塞项与 4 个「编译级已排除」的伪风险分开（风险 #1/#4 由此降级） | `/tmp/opencode/rv/probe_*.py` | — | 本 skill（Phase 1 增「编译探针」步骤） |

### Transferable Lessons

- 「未核验」与「不可编译」是两回事：设计文档写「本路径仅有源码证据、未运行核验」时，检视方应先用最小 `tilelang.compile` 探针把它落到「可编译 / 编译失败」二元，再决定是风险还是阻塞。
- 迁移设计的重设计项（如「双 AIV 按 key 轴二分」）往往把 GPU 的线程/分片概念直译成 NPU 的 frame API；凡是「按执行体 id 索引数据」的写法，都要先确认该 id 在本工具链的**用户可见性**与**可用组合**。
- 动态尾块在本平台有两条互斥路径：GEMM 侧要么静态 extent + 补 0 + 掩码（L1 路径唯一选择），要么显式 L0 路径 + 符号 region。设计若混写（GEMM 用动态 extent、copy 也动态），会同时丢掉两条路径的可编译性。

## Stage 1 (design) Retrospective — revision (v1) — 2026-11-23

结论：两个阻塞项（动态 GEMM extent、`T.Vector`/`dup sid`）以**形态替换 + 本机可复现编译探针**闭环，
8 条建议全部处理；门禁 `gate 1` pass，`design_calc_check` 的 `ub_budget` 由 `skip` 变为 `pass`。
本节记录「哪类设计判断被推翻、正确依据是什么」，并纠正本文件 Stage 1 first_design 小节中已失效的两条教训。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Design-flow | Phase M1 的「四态判定」允许把「源码存在 + 有 assert」当作可用性证据，但没有区分**源码存在 / 静态可用 / 形态可编译 / 运行期正确**四级；v0 因此在 `T.gemm` region extent 与 `T.Vector` 上连错两处，且两条「回退阶梯」也各自依赖被证否的机制 | DESIGN.md history_version/design_v0.md §3.3、§9.1 风险 #1/#3；REVIEW.md 问题 1/2 | `references/migration-analysis.md` §5 与 `references/info-sources.md`：每个关键 API 形态必须记录四级状态之一，并在写回退阶梯前对该阶梯本身跑一次编译探针 | R |
| Design-flow | 「尾块策略」是一条贯穿 GEMM / copy / 掩码 / 写回的**跨章节不变量**，模板却把它分散在 §3.3 注、§5.4、§6.4 三处各写一遍，v0 因此出现「§5.4 说截断、§6.4 说编译期判定」的自相矛盾且无人发现 | DESIGN.md v0 §5.4 第 1–2 条 vs §6.4；v1 §3.3 注 2 + §5.4 + §6.4 统一口径 | `templates/design-template.md`：把「尾块策略」提升为独立小节（§5.4 唯一权威），其它章节只允许引用，不得复述 | R |
| Mechanical-check | `design_calc_check.py` 的 `ub_budget` 解析依赖「章节内出现 `**<LEVEL>` 标记 + 表首列以字母开头 + 合计行」；v0 的表首列是反引号包裹的 buffer 名，整表被判 no-table 而静默 `skip`；v1 只需把 buffer 名去掉反引号并把容量标注成独立行就变成 `pass`（10 条 = 176.5 KB，与人工复算逐位一致） | `design_calc_check.py` 输出 `ub_budget: skip → pass`；DESIGN.md v1 §4.5 | `design_calc_check.py` 在 `skip` 时输出「已看到表格但缺少 X 标记」的**可操作提示**，而不是只说 no per-level buffer tables parsed | R |
| Design-flow | 「探针脚本落盘」在 D-3 只覆盖「常数标定型」探针；本案的**编译可行性探针**同样需要落盘（否则下一轮无法复跑、检视只能信自述）。v1 把 3 个正向探针 + 3 条反例合并成一个文件放进 `history_version/` | DESIGN.md §3.5.4；`history_version/design_probe_compile.py` | `references/algorithm-research.md` / `hardware-cost-model.md`：把 D-3 的预算边界写清——编译可行性探针属「证据」，不占 2 个常数探针名额，但必须落盘 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| `AscendInsertOOBPadding` 会把「DST 全 extent、SRC 越界尾部」的 DMA 自动改写为 `min()` clamp + GM→L1 补 0（`asc_fill_l1`），因此**尾块可以完全不进 GEMM 的 extent 讨论**：GEMM 保持全 extent，尾部由 clamp+fill+掩码三件套收口 | D | DESIGN.md v1 §5.4、§3.5.4 探针 B；`ascend/pipeline.py:97-101`、`src/ascend/op/oob_padding.h`（`ClampDMACopyTail` 注释：shrink any axis exceeding the buffer shape） | `cd /tmp && ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python examples/mla_dense_bwd/_bwd_kernel/history_version/design_probe_compile.py`（探针 B 设备码含 `min(64, 96 - n0)` 与 3 处 `asc_fill_l1`） | tilelang 0.1.15 / CANN 9.2.1 / ascend-950pr-9579 | 平台 knowledge（gemm-tiles / cube-tile 条目：「尾块 = 全 extent GEMM + OOB clamp + 掩码」，取代 L0/符号 extent 的二选一叙事） |
| 「补 0 ≠ 梯度 0」：OOB fill 只让 `s = 0`，而 `P = exp(0·scale − lse) = exp(−lse) ≠ 0`，故尾块的 `n0+i < S_kv` / `m0+j < S_q` 掩码**不可省**；这是「用 OOBPadding 替代区段截断」时必须一并保留的语义项 | D | DESIGN.md v1 §5.4 第 1 条、§9.2 末行；v0 §5.4 第 2 条（曾误以为「补 0 即可」） | 同上的探针 B + L0-3（`S=96`）的哨兵值用例 | 同上 | 平台 knowledge（attention 类尾块语义）+ `references/migration-analysis.md` 边界语义核对清单 |
| 整 tile `T.copy`(L0C→UB) 在 `__mix__(1,2)` 下由 AIC 发一次 fixpipe，随后两次 `asc_sync_intra_arrive(PIPE_FIX, e/e+16)` 通知两个副块；两个 AIV 冗余执行同一整 tile 向量段（正确、无 sid、向量有效吞吐按 1 AIV 计） | D | DESIGN.md v1 §1.6.3、§3.5.4 探针 A 设备码（`int32_t sid = asc_get_sub_block_id()` 自动合成且未被使用） | 同上探针 A（`/tmp/opencode/probe_A.txt` 的 AIV 段） | 同上 | 平台 knowledge/execution-patterns（混合 kernel 的 AIV 冗余执行口径） |
| 单遍反向算子的 L1 端口口径必须**逐项分解**（GEMM 操作数读 + UB→L1 写 + GM→L1 装载 + 常驻摊销）：v0 的「248 KB / 27 FLOP/字节」落在「不含 GM→L1」与「全含」之间不可复现；统一到全含口径后主选为 279544 B / 24.5，备选 128/32 只改善 5.1%（而非 15%） | D | DESIGN.md v1 §1.6.0 分解表、§1.6.3 备选行 | none（纸面，代表 workload ds-v3-b2-8k；可由公式反解 590.6 GB / 2 113 536 次迭代） | — | `_shared/standards/hardware-cost-model.md`：要求「每迭代片上端口流量」给出逐项分解表，并声明含/不含哪些项 |
| 设计文档的容量表要写成**机械可解析**形态（LEVEL 标记行 + 无装饰的 buffer 名 + 独立合计行 + Bytes 后缀），否则 `design_calc_check` 静默 skip，复算全靠人工 | P | `design_calc_check.py` 对 v0 `skip` / 对 v1 `pass` 的对照；DESIGN.md v1 §4.5 | `python3 .agents/tools/design_calc_check.py --design examples/mla_dense_bwd/_bwd_kernel/DESIGN.md --task-dir examples/mla_dense_bwd/_bwd_kernel` | — | `templates/design-template.md` §4.5/§4.6 提供可解析的表格样例 |

### Transferable Lessons（可迁移教训）

- **「API 存在」永远不等于「这个形态能编译」。** 本任务两次阻塞都不是符号缺失，而是 lowering 的形态契约
  （L1-GEMM 读 region extent 要求静态；显式 `T.Vector` 与 `dual_copy`/UB→L1 混用触发 SSA `dup sid`）。
  凡是「主结构押在某条未核验路径」的设计，先把该形态写成 20–60 行最小 kernel 跑
  `tilelang.compile(target="ascend")`，再用结果决定它是风险还是阻塞——本任务用 3 个正向 + 3 个反例探针
  在一小时内替代了「实机首跑失败」的路径。
- **回退阶梯本身必须是可编译的形态。** v0 的两条回退（半区 GEMM、`MixedKernel`+手工 sync）都复用了被
  证否的 `sid` 机制，等于没有回退。写回退前先问：这条阶梯依赖的每个 API 形态，是否已有本轮探针/已核验条目？
- **本文件 Stage 1 first_design 小节的两条教训已过时，按本节为准**：
  (a) 原「`T.dual_copy` 是 L0C→UB 的唯一通道 + 用 `T.Vector` 取 subblock id」——正确结论是
  **整 tile `T.copy`(L0C→UB) 可用，且用户侧不应出现 `sid`**（`dup sid` 不可编译）；
  (b) 原「跨步 region 的 DMA 行为未核验」——已由编译级探针排除（跨步 region + `transpose=True` +
  `set_atomic` + L0C→跨步 GM 均 COMPILE_OK），剩余风险只在运行期。
- **「补 0」只能解决 GEMM 的数值污染，不能解决语义。** 尾块处理要同时回答三个问题：GEMM 的 extent 从哪来
  （静态 allocation）、越界怎么搬（平台 clamp+fill）、被补 0 的位置如何保证不进输出（掩码必须保留
  `< S_kv` / `< S_q` 两项）。任何一环缺失都会在 Stage 3 变成静默的精度错。
- **容量表与端口流量表要能被别人复算。** 表格形态（可解析）、单位口径（B / KiB / KB 混用陷阱）、
  含/不含项声明，三者缺一就会被检视判为「不可复现」——这次两条 warn（UB 预算、L1 端口）都是这么来的。

## Stage 2 (review) Retrospective — revision v1 复检 — 2026-11-23

结论：`结论: 通过`（阻塞 0 / 建议 2）。本节只记**检视流程自身**的缺口与可复用方法，设计问题的处置已随
REVIEW.md 流转，不在此重复。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Review-flow | 机械复核要求跑 `design_calc_check.py --task-dir <任务目录>`，但 harness 子模式下 `examples/{op}/`（op 级）与 `examples/{op}/{func}/`（函数级）各有一份 `.stage_state.json`；传 op 级目录会触发 `PlatformError: selected task platform differs from the artifact's task`，本轮先报错才发现应传**算子目录** | REVIEW.md「机械复核」段首行；`.agents/tools/platform_context.py:320-332` | 本 skill Phase 1 第 7 步与 REVIEW.md 模板注明「harness 子模式 `--task-dir examples/{op}/{func}`（函数级状态所在目录）」 | R |
| Review-flow | 「设备码证据与文档一致」缺可复现口径：设备码里的**计数与事件 id** 会随探针脚本演化与调度分配漂移（本轮 §3.5.4/§1.6.2 的 `asc_fill_l1 ×3` 实测为 4；§7.2 的 `e=1/17` 实测 3/19、4/20），而结构性结论（clamp 存在、双 arrive 相隔 16）才是稳定证据 | `history_version/design_probe_compile.py` 本轮输出 vs DESIGN.md §3.5.4/§1.6.2/§7.2 | 本 skill Phase 1 第 7 步加一句：设备码证据只按「调用**种类** + 结构特征（参数形态/相邻关系/偏移）」比对，计数与调度分配的 id 不参与一致性判定；文档若写计数须同时给出探针脚本版本或落盘输出路径 | R |
| Review-flow | 「编译级已证 / 运行期未证」的**放行条件**没有成文清单，本轮用三条自定标准（① 有可判定标准 ② 有**可编译**回退 ③ 已排程为 Stage 3 首个探针）把 P1–P6 判为非阻塞；同类判断此前依赖个人尺度 | REVIEW.md「Stage 3 前置探针与遗留风险」表；DESIGN.md §8.3/§9.1 | 本 skill 加「运行期未知量放行三条件」小节，避免把「尚未实测」与「设计缺陷」混为一谈（v0 曾把可编译性缺陷记成风险 #3，v1 又需防反向放水） | R |
| Mechanical-check | `l0c_budget` 只核单个累加器（`block 64x64 x fp32 = 16 KB`），不核多累加器**合计**；本案 5 个累加器 160 KiB = 62.5% 全靠人工复算，而合计值才是溢出判据（128/64 组合 272 KiB 溢出正是合计问题） | 本轮 `design_calc_check` 的 `l0c_budget: pass` 与 §4.6 合计行对照；DESIGN.md §5.3 | `design_calc_check.py` 在 §4.6 出现多累加器行时输出合计与占比（与 `ub_budget` 同口径），并把「合计 ≤ 容量」作为 pass 条件 | R |
| Review-flow | 阻塞项闭环的核对方式：§11 的「检视问题逐项闭环」表是**自述**，本轮改为「把每条闭环项还原成可复跑命令」（探针脚本 3/3、verify_equiv、`design_calc_check`）才算证据；这一定式在 skill 里没写 | REVIEW.md「机械复核」段的 3/3 COMPILE_OK 与逐条结构比对；DESIGN.md §11.2 | 本 skill Phase 1 增一步：对上一版 REVIEW 的每个阻塞项，用**上一版给出的复现命令**重跑并逐条对照新形态 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 用设备码的实际分配偏移做预算表的**第三路验证**：`buf_dyn_l1` / `buf_dyn_l0c` / `buf_dyn_shmem` 的最大偏移可与 §4.5/§4.6 的 L1/L0C/UB 合计逐位对照（本案 L1 172032 B、L0C 163840 B、UB 172416 B ≤ 保守上界 180736 B），比纯算术复算更能识破「纸面预算与真实分配不同量级」 | P | REVIEW.md 维度 2；`/tmp/opencode/probe_A.txt`（`buf_dyn_l1 + 81920 half`、`buf_dyn_l0c + 36864 float + 16384`、`buf_dyn_shmem[86208]`） | `cd /tmp && ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python examples/mla_dense_bwd/_bwd_kernel/history_version/design_probe_compile.py`，再在输出设备码里取三类 buffer 的最大下标 | tilelang 0.1.15 / CANN 9.2.1 / ascend-950pr-9579 | 本 skill（维度 2 复核方法）+ `_shared/standards/hardware-cost-model.md` |
| 混合 kernel 里 `T.set_atomic("add","float32")` 会被 codegen 同时展开到 AIC（包住 `asc_copy_l0c2gm`）与 AIV（每迭代一对 set/none，窗口内夹着 `scatter`/`copy_ub2l1`）段——检视时须确认原子窗口是否覆盖非 GM 存储，否则运行期可能出现非预期副作用 | D | REVIEW.md「Stage 3 前置探针」P1 行；`/tmp/opencode/probe_A.txt` 的 AIV 段（`asc_set_atomic_add_float()` … 3×`ascend_nd2nz_scatter`/`asc_copy_ub2l1` … `asc_set_atomic_none()`） | 同上探针 A；运行期确认交 Stage 3 的 P1/P5 | 同上 | 平台 knowledge/execution-patterns（store-atomic 的作用域与 codegen 展开） |
| 「运行期未知量」的放行三条件：① 有可判定标准（逐元素/哨兵/对照实现）② 回退形态**本身可编译** ③ 已排程为 Stage 3 首个探针——三条齐备判非阻塞，缺任一判阻塞 | P | REVIEW.md「Stage 3 前置探针与遗留风险」；DESIGN.md §8.3 P1–P6 | none（方法；本案以 §8.3 探针表为实例） | — | 本 skill Phase 2 维度 0/1 的判定附注 + `tilelang-op-design` 的 migration-analysis |

### Transferable Lessons（可迁移教训）

- **上一版阻塞项的闭环要用「原复现命令重跑」验证，不要读 §11 的闭环自述**：自述只能证明作者认为改好了，
  重跑探针 + 等价性脚本 + 机械检查三项才能证明改后的形态真能编译/真等价。
- **预算表要三路对齐**：公式复算 → 机械检查脚本 → 编译产物的实际分配偏移。三者同量级才判维度 2 pass；
  只做前两路时，寄存器/L0C 混算或别名假设错误可以一直藏着。
- **「编译级可编译」与「运行期正确」必须分账**：平台 pass 自动生成的东西（OOB clamp、`clear_accum` 动态布尔、
  双副块 sync）都能编译通过，但语义只在设备上才成立；对这类点，设计侧的合格线是「有判定标准 + 回退本身可编译
  + 排程为首个探针」，而不是「已核验」。
- **不要复述文档里的设备码计数与事件 id 作为证据**：它们随探针脚本与调度分配变化（本轮 `fill_l1` 3→4、
  `PIPE_FIX` 事件 id 1/17→3/19），稳定的证据是「调用了什么、参数形态、相邻/相隔关系」。

## Stage 3 (develop) Retrospective — first_impl attempt 1 — 2026-11-23

结论：`[PRECISION_PASS]`（L0 13/13、L1 6/6；L2 3/4 与 boundary 10/11 各 1 条建议级告警，
均已定位为「dS 强制 cast 到输入 dtype + 抵消放大」的固有量，非实现缺陷）。

本轮最关键的事实：**DESIGN.md §3.3 的主选向量段形态（整 tile `T.copy(L0C→UB)` + 整 tile epilogue）
在本工具链上不可用**，实测必须切设计自己写好的回退（`dual_copy` 2:1 半区）。因此 kernel 与 DESIGN §3.3
的差异是「形状」而非「数学」：5 个 NT GEMM、L1 操作数全 extent、掩码语义、原子写回、`scale` 归属全部保持。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Develop-flow | 设计文档把「编译级 COMPILE_OK」当成形态可行的证据，但本次第一优先级探针 P1 在**运行期**推翻了主选形态：`T.copy(L0C→UB)` 编译成 `asc_copy_l0c2ub(..., dual_dst_mode=0, ...)`，只落一个 AIV 的 UB（实测 99.2% 元素错、25% 从未写）。开发 skill Phase 4 未要求「先跑最小运行期探针再写全量 kernel」，本轮靠 hand-written P1 才在 30 分钟内定位 | DESIGN.md §3.5.4（探针 A 结论「COMPILE_OK」）vs `_bwd_kernel.py` 的 `T.dual_copy` 形态；repro/TRAP-950-l0c2ub-single-aiv.py | tilelang-op-develop Phase 2 加一步：设计文档里标「编译级已验证、运行期未证」的形态，必须先用**最小运行期探针**（单 GEMM + 单次跨层搬运 + 单段向量）验证再写全量，禁止把 compile-only 形态直接铺开到 5 个 GEMM 的完整 kernel | R |
| Develop-flow | 测试 harness 自身会伪造「精度失败」：`repeat>1` 的计时循环对同一个 `dq_acc` 反复 launch，而 `dq_acc` 是 store-atomic 的 in-out 累加器 → 结果被叠加 3 次，6 个用例全 FAIL（每例最大误差 ~82 = 2×max|dq_acc|）。开发 skill 没有「先取结果、再计时」的顺序要求 | `_bwd_kernel.py::run_case`（修正前的 repeat 分支）；对照 `/tmp/opencode/L1_run.log`（全 FAIL，误差 82.9）与 `/tmp/opencode/L1_run2.log`（全 PASS） | tilelang-op-develop 的 run_case 模板加注：任何 in-out / 原子累加输出，repeat 计时必须在**结果读取之后**，或每轮前置清零，并在报告里区分「首调（含 module load ~127 ms）」与「稳态」时延 | R |
| Info-source | 平台 `execution.precision_command` 的解释器 `/usr/local/python3.12.16/bin/python3` 在本机存在但**无 torch**；实际可跑的解释器是 `/workspace/venv-ascend950/bin/python`（调度方已在 prompt 里手工纠正）。若照抄 platform 模板会得到 ImportError 级「环境问题」误判 | `statectl platform --task-dir examples/mla_dense_bwd/_bwd_kernel` 的 execution 段 vs 本机实测 | platform 的 `execution` 段在站点 profile 偏离时应落库实际解释器（与 env-950pr.sh 的 `$TILEOPS_950PR_VENV/bin/python` 同源），而不是保留参考机路径 | R |
| Develop-flow | 「A2 两臂逐元素一致」若用 `torch.equal` 判 dq 会误报：`dq` 走跨核 store-atomic，两次 launcher 的累加顺序不同（实测最大 4.77e-07）。dk/dv 是单属主写入，可以 bitwise 判等 | `_bwd_kernel.py::_case_arm_equivalence`（L0-12 首轮 FAIL → 改为 dk/dv bitwise + dq 1e-5） | 开发 skill 的「双实现等价」检查清单加一条：归约/原子输出的对照判等只能按该输出的**累加契约**取容差，不能一律 bitwise | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| `T.copy(L0C→UB)`（`dual_dst_mode=0`）在混合 kernel 中只写一个 AIV 的 UB，整 tile 向量段必然读到脏数据；必须用 `T.dual_copy` 2:1 半区分片（M 分片 [M,N]↔[M/2,N]、N 分片 [M,N]↔[M,N/2] 均实测可用），L1 目标仍可保持整 tile | D | repro/TRAP-950-l0c2ub-single-aiv.py（dual 0/4096 错；plain 3839/4096 错、1024 未写）；`_bwd_kernel.py` §2 注释 | `ASCEND_RT_VISIBLE_DEVICES=0 /workspace/venv-ascend950/bin/python examples/mla_dense_bwd/_bwd_kernel/repro/TRAP-950-l0c2ub-single-aiv.py` | tilelang 0.1.15 / CANN 9.2.1 / ascend-950pr-9579 | 平台 knowledge/repro + `PL-950-mixed-epilogue` 条目的「适用边界」段（现文只说「不推广到 N 分片/多阶段」，未说「plain copy 不可用」） |
| 混合 kernel 取 AIV 副块号：`T.call_extern("int64", "asc_get_sub_block_id")` 写在 kernel body（自动调度会落进 `if ASC_IS_AIV` 并作为 VF 实参传入），值等于 `dual_copy` 的分片 id；`T.Vector(vector=2) as sid` 仍报 `dup sid`，`asc_get_sub_block_id()` 不能写在 `T.SimtVF` 内（`__simt_vf__` 无匹配重载） | D | repro/CONST-950-aiv-sid-capture.py（rows[0,32) 得 sid=0、rows[32,64) 得 1000；`T.Vector` 原文报错被断言） | 同上脚本 | 同上 | 平台 knowledge/repro + programming.md「API 与执行区域」的 subblock id 段 |
| 半区分片后，**mask 的全局索引必须随 AIV 变**：`dual_copy` 的 2:1 切分让每个 AIV 只持有被测维的一半，任何依赖该维全局下标的表达式（因果掩码 `gi <= gj + shift`、尾部 `gi < S_kv`）都必须加 `sid * 半区`，否则对角块掩码错半区（编译能过、精度全错） | P | `_bwd_kernel.py::_bwd_main` 掩码臂 `gi = n0 + T.cast(sid,"int32")*_HN + i`；L0-4/L0-5/L0-6 均覆盖对角块 | `python _bwd_kernel.py --level L0` | 同上 | tilelang-op-develop / tilelang-mixcv-skill 的「dual_copy 分片」注意事项 |
| 混合 kernel 的 GM 3-D 固定 head 切片（`dk[t0:t0+64, h, :]`，区域形状 `[64,1,192]`）不能作为 `dual_copy` 的 dst（`Cannot infer dual_copy split direction`，中轴单例既非 2:1 也不在被检的两条轴上）；改用 `T.copy` + `sid` 偏移的显式区域即可（半区偏移运行期动态也成立，尾块仍由 OOB clamp 收窄） | D | `_bwd_kernel.py` 块尾 `T.copy(out_k, dk[koff_own : koff_own + _HN, h, :])`；P2 哨兵检查（全用例 0 越界写） | 同上 + 哨兵行机制（`_OOB_PAD`） | 同上 | 平台 knowledge/repro（dual_copy 的 dst 形态边界） |
| store-atomic 的 in-out 语义实测成立：`dq_acc` 预填 2.0 后运行，结果 = 2.0 + dq_acc_true（误差与清零版逐位一致），证明是加法而非覆盖；调用方必须 `torch.zeros` 前置 | D | `_bwd_kernel.py` L0-11 的两个用例；原 PL-950-atomic-splitk 的调用契约 | `python _bwd_kernel.py --level L0` | 同上 | 平台 knowledge（store-atomic 契约） |
| bf16 下 `atol=5e-3` 对**强抵消元素**不够：已知 bf16 `dS` 的 cast 舍入（源语义）乘上 `sum|terms|/|value| ≈ 17×` 后，dk 单元素误差可达 7.8e-3（1e6 元素里 1 个）。判「实现错」还是「容差错」的判定手法：用 fp64 复算同一元素、再把 `dS` 单独 cast 成 bf16 重算，若 kernel 与后者差 ~1e-4 即为固有舍入 | D | `_bwd_kernel.py` L2-3 / B-4 的 WARN；诊断脚本 `/tmp/opencode/diag_dk.py` 输出（kernel -0.0917969 / bf16-ds 仿真 -0.0919166 / fp32-ds -0.0934249 / golden -0.0932617） | `python _bwd_kernel.py --level L2`（L2-3 单元素 7.8e-3 > atol 5e-3） | 同上 | 平台 knowledge（numerical-boundaries 的容差判定）+ TileOPs `numerics.py` 容差口径注 |

### Transferable Lessons（可迁移教训）

- **compile-only 证据不能推出形式可用**：凡是跨执行体（L0C↔UB、AIC↔AIV）的搬运，先用「单 GEMM + 单次搬运 +
  单段向量 + 回写 GM」的最小运行期探针验形态，再铺到完整 kernel；本函数按此在首个探针就发现了主选形态不可用。
- **混合 kernel 里 AIV 的 UB 是各副块私有的**：任何「AIC 写一次、两个 AIV 都读整块」的写法都不成立；跨 L0C→UB
  一律 `dual_copy` 2:1，且下游 L1/GM 目标保持整 tile 由两个半区拼回。
- **分片维度的全局下标要显式补 `sid * 半区`**：半区化之后，掩码/边界判断不再能用 tile 局部下标，编译不会报错，
  只会在对角块上静默错一半。
- **in-out 累加输出（原子加）不能用于计时重复**：先读结果、再计时；否则会把「精度失败」造在自己身上。
- **判断「精度差」归属时，用「把参考实现改成与 kernel 同样的舍入」做二分**：把 dS 单独 cast 到输入 dtype 后重算，
  若与 kernel 吻合到 1e-4，问题在容差而不在实现。这一手法可把 bf16/fp16 的告警从「疑似 bug」降级为「已知固有量」。
