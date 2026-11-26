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

## Stage 1 (design) Retrospective — revision v2（降 Vector 依赖 / 提 Cube 占比）— 2026-11-25

结论：`DESIGN_COMPLETED`，但**目标本身需要重写**——调度方给的「MFU 67% → ≥80%（Cube 口径）」在本算子的物理地板
上不可达（§1.6.4 F：Cube 0.88 + 填充 1.18 + 向量 1.24 = 3.30 µs/迭代 vs 目标 1.00），本版把它改成
「Stage 3 ≤227 ms（原型实测外推 175 ms）+ Stage 4 ≤115 ms / cube ≥40%」，并把「80% 不可达」作为
可复算的边界结论交付。本版另一个关键产出是**用三个探针把 v1 的 6.03 µs/迭代拆成可归因的分段单价**，
从而把「为什么 5 个 GEMM 没让 Cube 成为瓶颈」定位到一句向量跨步 scatter 写（占 63%）。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Design-flow | Phase R 的 R4 要求定量 roofline，但没有一步要求**把调用方给的验收目标与 roofline 对账**：本次调度要求「Cube MFU ≥80%（≈50.6 ms）」，而 R4 一算就知三项地板之和 >3×该目标。若照字面承诺，Stage 3/4 必然反复；本版靠人工对账才发现 | DESIGN.md §1.6.4 F 表；调度 prompt 的「MFU ≥80%」；`.agents/skills/_shared/standards/hardware-cost-model.md` | Phase R 的第 4 步（产出结论）增一条：**若调用方给出量化验收目标，必须先与「估算下界」对账**；目标 < 下界时在 §1.6.0 显式给出「不可达 + 需要的结构变化」，并把它写进交付摘要（而不是默默按目标写设计） | R |
| Design-flow | R4 的「设计期估算」在平台 `cost_measurements` 为空时只允许写「待核验」，但**本平台的常量其实可用探针在 10 分钟内自测**（分段单价、单核 MTE2、每 score 向量代价）。skill 把 D-3 探针限定为「标定未知常数」且「≤2 个」，容易让人以为「revision 里不能再测」——本次实际用了 2 个常数探针 + 2 个形态/原型探针才拿到根因 | DESIGN.md §1.6.4；`design_probe_bwd_cost.py` 的 9 个分段单价；v1 的「估算下界: 待核验」 | `references/algorithm-research.md` / `hardware-cost-model.md` 明确：**常数探针的 2 个名额按「revision」重置**（每版都可重新标定一次），并补一句「形态/编译/端到端原型探针属证据，不占常数名额」（v1 已就后半句提过，需正式落入标准文件） | R |
| Mechanical-check | `design_calc_check.py` 的 `core_split` 只支持二维 GEMM 网格 `ceil(M/bm)×ceil(N/bn)`；本算子的任务网格是**三轴**（key 块 × head × batch），于是输出 `skip`（不是 fail），检视方需要额外读 §5.5 才能确认核数 | 本次 `design_calc_check` 输出 `core_split: skip, insufficient parsed dims {'bm': 64, 'bn': 96}`；DESIGN.md §5.5① 的闭式口径段 | `core_split` 支持「显式闭式表达式」：若 §5 出现 `逻辑核数 = ceil(X/a) × Y × Z` 形态的可解析行，按该式复算并比对声明值；否则在 `detail` 里明确「三轴网格，跳过」以免被当成缺陷 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 显式 L0 路径（`alloc_l0a/l0b` + `T.copy(L1→L0)`）支持 `trans_A=True` 与 `trans_B=False`，**且转置由编译器生成的 `asc_copy_l12l0a_transpose` 落在 MTE1**——因此「一份 dS 同时服侍 dK 与 dQ」在 NPU 上可直接保留，不需要向量侧转置副本 | D | DESIGN.md §0.5 #6、§0.6 R3、§1.6.4 D；`gemm_mad.py:126-134` 的 MN-major 布局推断；原型设备码 `asc_copy_l12l0a_transpose` + `asc_mmad` | `/workspace/venv-ascend950/bin/python .../history_version/design_probe_l0_gemm.py`（(T,T)/(F,F)/(F,T) 三组全 PASS，rel 1.1e-7） | tilelang 0.1.15 / CANN 9.2.1 / ascend-950pr-9579 | 平台 knowledge（gemm-tiles 条目补「L0 路径的四类 trans 组合 + 转置落在 MTE1」）；并用于修正所有「NPU 只能 NT」的旧结论 |
| **同一 kernel 内混用「L1 操作数 GEMM」与「显式 L0 缓冲」会互相覆盖 L0A/L0B**，结果是静默数值错而非编译错 | D | DESIGN.md §1.6.4 E1、§3.5.1、§9.2 首行；原型 bisect：`dk` 错 4.8e-1 / `dv` 3.1e-2，全改 L0 后同一测试 PASS | 同上原型（把 dQ 单独切 L0 的中间版本） | 同上 | 平台 knowledge（execution-patterns 的「GEMM 路径不可混用」条目）+ tilelang-op-develop 的常见错误表 |
| v1 内层迭代的**分段单价**（`mte2` 1.176 / `gemm5` 0.587 / `l0c2ub` 0.173 / `ub2l1` 0.231 / `epi0` 0.825 / `epi1` 4.302 / `chain` 5.950 µs）与「与生产 6.03 µs 差 1.3%」的对齐方法：把慢内核按阶段拆成 9 个微内核、同 grid/R 计时、用「整链」一项校准 | D | DESIGN.md §1.6.4 A；`design_probe_bwd_cost.py` | `.../history_version/design_probe_bwd_cost.py`（R=400，grid=28，取 3 次最小值） | 同上 | `_shared/standards/hardware-cost-model.md`（增「内层迭代分段分解」模板）+ 本 skill 的 Phase 2/Stage 4 入口 |
| **向量侧的跨步转置写（`ds_u[j,i]`，步长 32 个 f16）单句占内层迭代 63%**：同一条 epilogue 去掉该句从 5.950 → 2.196 µs；两趟（跨步读）方案仍要 2.469 µs | D | DESIGN.md §1.6.4 A/B；探针 `epi0/epi1/epi2` | 同上 | 同上 | 平台 knowledge（PL-950-vector-execution 补「SIMT 内转置写的实测代价」）+ tilelang-vector-skill |
| **L0C 的实测可用上界约 244 KiB 而非 262144 B**：`249856 B`（95.3%）端到端触发 `device error 507015`，`221184 B`（84.4%）稳定 | D | DESIGN.md §4.6「L0C 上界证据」、§9.1 风险 2 | `.../history_version/design_probe_bwd_proto.py`（把 NK/MQ 改成 128/48 即复现） | 同上 | 平台 knowledge（gemm-tiles / cube-tile 补「L0C 多累加器合计的实测红线 ≈240 KiB」） |
| `T.copy(UB, L1, transpose=True)` 不可用（`Ascend device compilation failed`），UB→L1 的转置只能走 `trans_A` 的 L1→L0A 搬运 | D | DESIGN.md §1.6.4 E2、§3.4 末行 | 同上探针 1 的 `u2l1_tr` 变体 | 同上 | 平台 knowledge（copy 能力边界）+ 本 skill 的 §1.6.3 候选表模板 |
| 「先测分段单价、再选结构」的成本极低：本次三个探针合计约 25 分钟，把「Vector 受限」这一现象定位到可修改的具体语句，并使 revision 一轮收敛（v1 是两轮编译探针才发现形态不可用） | P | DESIGN.md §1.6.4；本轮从 `design_probe_bwd_cost.py` 到 §1.6.0 结论的推导链 | 三个探针脚本（见 §10.1） | — | 本 skill 的 Phase 2 第 3 步（裁决计划）+ tilelang-op-optimize 的「机制探针」小节 |

### Transferable Lessons（可迁移教训）

- **「这个形态不可用」的结论必须带一次本机探针。** v1 从 `gemm_mad.py:162` 的 L1 路径断言推出「NPU 没有
  `trans_A`」，并据此引入了一个占 63% 的向量 scatter；本次用 3 行 GEMM 探针就把四种 trans 组合各验一遍，
  代价 4 分钟、收益是整条主结构。**负向断言的成本远高于正向验证**。
- **一个 kernel 里只能有一条 GEMM 路径。** L1 操作数路径的模板内部管理 L0A/L0B，与用户显式分配的 L0 缓冲
  互相覆盖，症状是静默数值错。凡是「只把某一个 GEMM 换成更合适的形态」的局部优化想法，先问它会不会
  违反这条不变量。
- **容量红线用实测上界，不用标称值。** L0C 标称 256 KiB，95.3% 时设备直接报错；84% 稳定。分块设计要留
  ≥15% 余量，且每个新配置先用最小原型跑一次（比纸面预算可靠）。
- **revision 的第一件事是把上一版的耗时拆开，而不是重写结构。** 本次 6.03 µs → 分段单价 → 一句 scatter 占
  63%，使「换 Cube」这个模糊目标变成「删掉这句 + 把转置挪到 MTE1 + 放大 Nk」三个可验证动作。
- **调用方给的验收目标要先与 roofline 对账。** 80% Cube MFU 对本算子意味着 50.6 ms，而三项地板之和是
  3.30 µs/迭代（≈167 ms）；照字面写设计会在 Stage 3/4 反复。正确做法是把「不可达」量化成交付物的一部分，
  并给出「改什么才可能接近」的清单。
- **慢内核的分段计时必须用「整链」一项校准**：本次微内核 `chain` 5.950 µs vs 生产 6.03 µs（差 1.3%），
  这个 1.3% 是整套分解可信的前提；没有它，分段相加容易系统性偏低（各段各自有固定开销被重复计入或漏计）。

## Stage 2 (review) Retrospective — revision v2 复检 — 2026-11-25

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Review-flow | 维度 3「非整除处理」与维度 6「L0 测试计划」都按**整张张量**的越界来判，缺「batched 展平视图里的 request 段」这一检查项；本次「全 extent + OOB clamp + 掩码」的写法在两个维度下都看似通过，实际在 `B≥2 且 S_kv%BN≠0` 时越界写进下一个 request | DESIGN.md §5.4#1「同一 clamp 保证不写出 `T_kv`」vs 原型设备码 `asc_copy_ub2gm_align(..., min(48, max((S − sid*48) − n0, 0)))`；REVIEW.md 问题 1 | 在 tilelang-design-review §3 维度 3 的非整除行增：「若 I/O 为 `(B*S, H, D)` 展平视图，必须核对写回 clamp 的维度是 batch 内还是整张量，并要求设计给出 `B≥2 + S%block≠0` 用例（且重复 N 次，因为竞态）」 | R |
| Review-flow | 维度 8 的算术复算只逐表核数字，没有检查 **roofline/floor 的合成逻辑**（串行求和 vs 重叠取 max）。本次「三项之和 3.30 µs 是下界」的推导通过了 Stage 1 的自检与上一版复盘，实际与设计自己的重叠机制、以及 `chain_notr` 小于分段和 0.80 µs 的实测相矛盾 | DESIGN.md §1.6.4 F、§9.1 风险 13；§1.6.4 A 的 2.992 vs 2.190；REVIEW.md 问题 2 | 在 tilelang-design-review §3 维度 8「调研结论正确性」增一行：「floor/估算下界必须声明是串行和还是重叠 max；若设计自己测得 `整链 < Σ分段单价`，则不得用分段和作下界」 | P |
| Info-source | 平台 `execution.prototype_command` 指向 `/usr/local/python3.12.16/bin/python3`（无 torch/tilelang），`working_directory=<task_dir>` 又落在仓根内（dev-root `tilelang/` 遮蔽 site-packages，`import tilelang` 直接失败）——探针复跑必须先绕开平台默认上下文 | `statectl platform --task-dir …` 返回的 `execution`；本轮 `import tilelang` 报错；DESIGN.md §3.5.4 用 venv + 仓根外 cwd | `statectl platform` 的 `execution` 增 `interpreter`/`cwd_constraint` 字段（或平台 notes 增一条「dev-root 遮蔽 → 必须在仓根外、用 toolchain venv」），使各 Stage 不必自行试错 | R |
| Review-flow | `design_calc_check.py` 的 `l0c_budget` 只校验单个 `block_M×block_N` 累加器，多累加器合计（本算子 5 个，合计 84.4%，红线 240 KiB）只能人工核 | 工具 JSON：`"block 64x96 x fp32 accum = 24.0 KB vs L0C 256 KB"`；DESIGN.md §4.6 的 221,184 B | 给 `design_calc_check.py` 增 `l0c_budget_sum`：解析 §4.6 表的全部 L0C 行求和并对照平台容量，超出时按 pass/fail 输出 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **`AscendInsertOOBPadding` 只按 buffer 形状裁剪 DMA 尾块，不按 batch 内子区间**：`(B*S,H,D)` 展平视图下，`B≥2 且 S%block_n≠0` 时尾块的 `dk/dv` 普通 store 会覆盖下一个 request 的行（`dq` 的 store-atomic 因加 0 而良性；读侧越界被掩码丢弃而良性） | D | REVIEW.md 问题 1；DESIGN.md §5.4#1 / §0.6 R5；源算子 `bwd.py:255-260` 的逐行守卫 | 编译 `history_version/design_probe_bwd_proto.py`（B=1）后读 `device_kernel.asc`：`asc_copy_ub2gm_align(..., min(48, max((2048 − sid*48) − n0, 0)), ...)` —— 上界为整张量行数；把原型改成 B=2、`S_kv=512`（512=5×96+32）即可构造触发面 | tilelang 0.1.15 / CANN 9.2.1 / ascend-950pr-9579 | 平台 knowledge `traps-950pr-store-copy` + `tilelang-op-design` 的 migration-analysis（边界语义条目） |
| **分段单价之和不能当整链下界**：同一结构上 `Σ(mte2 1.176 + gemm5 0.587 + l0c2ub 0.173 + ub2l1 0.231 + epi0 0.825) = 2.992 µs` 而 `chain_notr = 2.190 µs`（整链比分段和低 27%），说明链内已有重叠/共同开销被分段探针重复计入；据此「三项地板之和 = 下界」的 roofline 结论会把可达目标判成不可达 | P | DESIGN.md §1.6.4 A/F；REVIEW.md 问题 2 | `/workspace/venv-ascend950/bin/python …/design_probe_bwd_cost.py chain`（本轮实测 chain 5.944 / chain_notr 2.190） | tilelang 0.1.15 / CANN 9.2.1 / ascend-950pr-9579 | `_shared/standards/hardware-cost-model.md` + tilelang-design-review 维度 8 |
| 变 tile 宽度前应枚举各 manifest workload 的 `S % BN`：v1 的 `BN=64` 对 5 条 workload 全部整除，掩盖了「B>1 尾块跨 request」的缺陷；改成 `BN=96` 后 4/5 条出现部分尾块 | R | REVIEW.md 问题 1；DESIGN.md §5.5 的 workload 表（8192/96=85.33、4096/96=42.67、512/96=5.33） | 手算 `S_kv % block_n` 即可 | — | tilelang-op-design 的 quality-checklist（tiling 变更检查项） |

### Transferable Lessons（可迁移教训）

- 展平 `(B*S, H, D)` 的 batched 视图里，「尾块越界」有两种：越过整张张量（哨兵/`min()` 能抓）和越过
  **本 request**（哨兵抓不到）。只有普通 store 有害，store-atomic 加 0、读侧被掩码丢弃都良性——所以
  这类缺陷只会在 `B≥2 且 S%block≠0` 时、以竞态形式出现，必须重复运行才能判定。
- **分段单价之和不是整链下界。** 若 `整链 < Σ分段`，说明存在重叠或共同开销被重复计入；任何以「和」
  为下界的 roofline / 「不可达」结论都必须先验证可加性（用设计自己的 chain 与分段数据即可）。
- 换 tile 宽度前先把每条 manifest workload 的 `S % block` 算一遍：整除与否决定了尾块路径是否被现有
  用例覆盖，也决定了「这条路径上一版为什么没暴露」。

## Stage 1 (design) Retrospective — revision v3（B1/B2/B3 + 5 建议）— 2026-11-25

结论：`DESIGN_COMPLETED`。**C2 结构不推翻**（检视方复跑 3.489 µs/176 ms，本轮 3-D/4-D 复跑 3.446–3.509 µs 同值），
只修论证/语义/算术。三个阻塞项的处理各有**新增实测**支撑：
① B1 写了 4-D vs 3-D 的对照探针，**非因果下 3-D 5/5 失败、4-D 5/5 通过**，并抓到两组设备码 clamp 上界
（1024 vs 512）作为机械证据；② B2 把「三项之和=下界」改成管道级 `max`，顺带发现**新的硬条件**
`MTE1 ≤0.20 µs`（否则 80% MFU 在任何流水下都不可达）；③ B3 逐项复算出 C2 GM 访存 190 GB（原 281）、
端口 217,088 B（原 221,184——原值其实是 L0C 容量，与同表合计自相矛盾）、C1 UB 47.2%（原 71.2% 无法复现）。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Design-flow | 「平台 pass 会处理尾块」这类依赖**编译器内部行为**的论断，skill 没有要求给出「该张量 rank 下」的设备码证据；v2 因此把 `AscendInsertOOBPadding` 的裁剪维度误当成语义维度 | DESIGN.md §5.4#1 / §1.6.4 H 的两组设备码；REVIEW.md 问题 1 | `references/migration-analysis.md` 的「边界语义」条目增一句：凡以平台 pass 作为边界论证的一方，必须贴**该形态（rank/布局）**的编译产物实参（`min(...)` 的完整表达式），并说明若 rank 变化该实参如何变 | R |
| Design-flow | skill 的 Phase 2「数学等价优化」只校验公式等价，没有「聚合量的算术自洽」检查：本次 v2 的端口表行（221,184）与同表合计（325,625）差 4,096、R3 的 C2 行沿用 C1 的迭代数——都是「单看每个数都像对的」 | DESIGN.md §1.6.0「R3 口径与出处」表；REVIEW.md 问题 3 | `hardware-cost-model.md` 增「聚合量三件套」：每个聚合数必须带 ①每迭代单价 ②迭代数 ③`÷/×核数` 因子，且**同表合计必须由行相加得出**（revision 时机械重算一遍） | R |
| Review-flow | 检视建议本身也可能带算术错误（建议 5 的「Nk=80 可留出 L0 2 版本余量」按 `2×(24,576+384·Nk) ≤ 65,536` 不成立），Stage 1 若照抄会把错误固化 | 本轮 §1.6.3 的容量复算；`L0B = 55,296 B`，2 版本需 110,592 B | tilelang-design-review 的建议条目要求「涉及容量的建议必须给出与容量比对的算式」（或标注为待核） | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **`AscendInsertOOBPadding` 的裁剪维度 = buffer 形状的行数**：同一份 C2 kernel，3-D `(B*S,H,D)` 视图下 `dk` 写回行数为 `min(48, max(B·S_kv − sid·48 − n0 − b·S_kv, 0))`，4-D `(B,S,H,D)` 视图下为 `min(48, max(S_kv − sid·48 − n0, 0))`；前者在 `B=2 且 S_kv%96≠0` 时把 0 写进下一个 request 的 64 行（普通 store ⇒ 竞态） | D | DESIGN.md §1.6.4 H、§5.4#1；`history_version/b1_device_code_bounds.txt` | `.../history_version/design_probe_b1_tailrace.py`（B=2/S=512/H=4，5 次 × 4 组合） | tilelang 0.1.15 / CANN 9.2.1 / ascend-950pr-9579 | 平台 knowledge `traps-950pr-store-copy`（把原条目从「按 buffer 形状」升级为「按 rank 给出实参样例 + 4-D 修法」） |
| **竞态类边界缺陷的判定方法**：因果掩码下尾块迭代数最少 ⇒ 越界 0 写总被真实值覆盖，5/5 侥幸通过；换非因果后 5/5 失败（0.33–0.40）。**「单次/单 mask 通过」不能否证竞态** | P | DESIGN.md §1.6.4 H、§8.3 L0-9/L0-10 | 同一探针的 causal/non-causal 两组 | 同上 | tilelang-op-design 的 quality-checklist + tilelang-op-develop 的 L0 用例模板（「B≥2 且尾块」必列，且重复 ≥5 次、逐 request 比对） |
| **管道级 roofline 清单（含「同一单元内才相加」规则）**：AIC 时间线 = `MTE1 + mad`（L0 单版本），其余管道（MTE2/AIV/FIX/MTE）取 `max`；据此 `80% MFU` 的必要条件变成 `MTE1 ≤0.20 µs`（L1→L0 ≥1.09 TB/s），这是 v2 完全没有暴露的约束 | P | DESIGN.md §1.6.4 F/G 的管道表与敏感性表 | 用 `design_probe_bwd_cost.py` 的分段价 + `gemm5` 与 mad 解析值之差反推 MTE1 上界 | 同上 | `_shared/standards/hardware-cost-model.md` 增「管道清单模板：单元 / 单价 / 是否与谁串行」；tilelang-design-review 维度 8 增「下界是否按执行单元取 max」检查项 |
| 保守量化的写法：MTE1 无平台常数时，用「同几何 v1 的 `gemm5 = 0.587 µs`（已含 L1→L0）按流量比 ×1.26 得上界 0.74 µs」+「`gemm5` 与 mad 解析值 0.534 只差 10% ⇒ 模板内高度重叠 ⇒ 点估计 0.1–0.3 µs」+「敏感性表把结论对 MTE1 的依赖显式化」，比单点假设更可审计 | P | DESIGN.md §1.6.4 G | `design_probe_bwd_cost.py gemm5` + `CONST-950-gemm-mte2-burst` 的 356–359 TF/s | 同上 | `hardware-cost-model.md` 的「未知常数」小节：允许「上界 + 点估计 + 敏感性表」三件套替代单点假设 |

### Transferable Lessons（可迁移教训）

- **「pass 会处理边界」必须落到该形态的实参**：`min()` 的第二个参数是 `S_kv` 还是 `B·S_kv`，取决于张量
  rank，而 rank 是设计自己的选择。设计期就能从编译产物读到这个实参——比任何散文论证都可靠。
- **竞态要换 mask 才测得出**：因果下「越界块更短」这一结构性偏置会掩盖缺陷。凡涉及跨 request /
  跨 block 的写，用例必须至少包含一个「两块等长」的形态（非因果或 `S_q≠S_kv`）+ 重复运行。
- **下界按执行单元取 max，同一单元内的串行链才相加**（本例 `MTE1 + mad`）；「三项之和」既不是下界
  也不是实测值，用它去否证一个目标会把可达目标判成不可达。
- **聚合量的自洽性要在 revision 时机械重算**：本轮靠「行相加 ≠ 合计」这一条就抓出了端口表的抄写错误；
  每个数带上「单价 × 迭代数 ÷ 核数」三要素后，同类错误无处可藏。

## Stage 2 (review) Retrospective — revision v3 复检（3 阻塞 + 5 建议）— 2026-11-25

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Review-flow | 我在上一轮检视里给出的建议「`Nk=80` 可留出 L0 2 版本余量」**按容量算术不成立**（`2×55,296 = 110,592 B > 64 KiB`，L0A 亦为 `2×40,960 > 64 KiB`），被设计方在 §12.2 更正。检视建议中的**配置类主张**没有「先自算容量/带宽」的强制步骤 | REVIEW.md(v2) 问题 5 的修改建议；DESIGN.md §1.6.3 / §12.2 #2 | 在 tilelang-design-review §6 输出规则增一条：「建议里给出的任何 tile/缓冲配置，必须附检视方自己的容量/带宽复算（≤ 平台容量），否则只能写成方向性建议」 | R |
| Review-flow | 「floor/roofline 的合成」检查项（上轮新增的 `sum vs max`）被设计采纳后，**下一层仍有缺口**：`max` 的各项必须核对**缓冲区版本**（单版本 ⇒ 同一单元内的依赖链要相加）。v3 把 `FIX 0.267 µs` 当独立管道，而 `st_c/dpt_c` 是单版本 L0C，`fix(i)` 与 `mad_st(i+1)` 有 WAR | DESIGN.md §1.6.4 F 的 max 式与 §1.6.4 G（只把 `MTE1+mad` 写成串行）；REVIEW.md(v3) 维度 8 的残留项 | 在 tilelang-design-review 维度 8 的「复杂度算术」行增：「管道级 max 的每一项须标注其缓冲区版本；单版本缓冲的写后读/WAR 必须并入同一单元的串行链」 | P |
| Info-source | 平台 knowledge 里**没有 MTE1（L1→L0）带宽常数**，导致 `80% MFU` 的关键量只能标「未核验假设 1.0–2.0 TB/s」。设计已排 P8 去标定 `aic_mte1_active_bw`，但常数库仍是空的 | DESIGN.md §1.6.4 G「平台 knowledge/constants.md 与 bottlenecks.md 中**没有 MTE1 带宽条目**」 | Stage 3/4 采到后按 `CONST-950-*` schema 登记一条「L1→L0 有效带宽 vs 操作数形状」，供后续 Cube 类算子直接引用 | D |
| Review-flow | 「rank 决定 OOB clamp 裁剪维度」这条硬约束，上一轮复盘已建议加入维度 3/6 的检查项，本轮确认它已被设计写进 §3.5.1/§5.4（闭环），但**检查项本身仍未落到 skill 文档** | 上轮复盘 Skill Flow Issues 第 1 行；本轮 DESIGN.md §3.5.1 新增行 | 与上轮同一改动，仍待落地到 `tilelang-design-review` §3 维度 3（batched 视图的写回边界） | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **「clamp 维度 = buffer 形状」的闭环实证**：同一份地址表达式，3-D 视图的写回行数上界是 `B·S_kv`、4-D 是 `S_kv`；只有 rank 变、算术不变。差分证据 = 「地址相同 + 上界不同」+ 「3-D 非因果 5/5 FAIL / 4-D 5/5 PASS」 | D | DESIGN.md §1.6.4 H、§5.4#1；`history_version/b1_device_code_bounds.txt`；REVIEW.md(v3) 机械复核 §5 | `TMPDIR=/workspace /workspace/venv-ascend950/bin/python …/history_version/design_probe_b1_tailrace.py`（我复跑：3-D 非因果 5/5 FAIL 0.33–0.40、4-D 5/5 PASS ≤2.1e-3；设备码 `min(48, max(1024−…−b·512,0))` vs `min(48, max(512−…,0))`） | tilelang 0.1.15 / CANN 9.2.1 / ascend-950pr-9579 | 平台 `traps-950pr-store-copy` + `tilelang-op-design` migration-analysis（边界语义） |
| **管道级 max 下界要同时核对缓冲版本**：`FIX` 被当作独立管道，但单版本 L0C 的 WAR 使其位于 AIC 链上（保守 `mad+MTE1+FIX = 1.25–1.45 µs ⇒ 63–73 ms`），`80% MFU` 还需「`st/dpt` 双版本（`Nk=80` 时 233,472 B = 89%）」这第 7 条 | P | DESIGN.md §1.6.4 F/G；REVIEW.md(v3) 维度 8「残留」 | 纸面推导：`0.88+0.20+0.267 = 1.35 µs`；双版本容量 `192,512+2×20,480 = 233,472 B` | 同上 | `_shared/standards/hardware-cost-model.md` + tilelang-design-review 维度 8 |
| 竞态类边界的用例必须选「两块迭代数相同」的形态才可确定性触发：因果下尾块 `loop_st` 最靠后（1 次迭代）总先完成 ⇒ 5/5 侥幸通过；非因果下 1 次 vs 8 次改成 8 vs 8 ⇒ 5/5 失败 | P | DESIGN.md §1.6.4 H 第 1 点；我复跑的逐次数据 | 同上探针（`causal=True/False` 两组对照） | 同上 | 平台 `traps-950pr-store-copy` + tilelang-op-develop 的门槛用例模板 |

### Transferable Lessons（可迁移教训）

- 边界类修复的**验收证据**要做成「同一份地址、只有裁剪上界变化」的差分，再加「缺陷形态确定性失败 / 修复形态全通过」的对照；只看「改完跑通了」无法区分修复与侥幸（因果掩码下 5/5 通过、换非因果即 5/5 失败）。
- 竞态用例要挑**两块工作量相等**的几何（如非因果）才可确定性触发；工作量差 8 倍的因果形态会让「先完成的块」单方面决定结果。
- 管道级 `max` 下界必须逐项核对**缓冲版本**：单版本缓冲的 WAR 会把「独立管道」拉进同一时间线（本例 `FIX` 0.267 µs 是否落在 AIC 链上取决于跨迭代流水是否让它与上一迭代尾部 mad 重叠）。
- 检视方给出的**配置类建议**（tile 尺寸、缓冲版本、回退方案）必须自己先做容量/带宽复算；`Nk=80` 那条建议就是漏了 `2×55,296 > 64 KiB`，被设计方更正。

## Stage 3 (develop) Retrospective — first_impl attempt 1（Cube 化重设计 / v3 冻结设计）— 2026-11-25

结论：`[PRECISION_PASS]`。L0 27/27、L1 6/6、L2 4/4、boundary 12/12；
生产 shape `ds-v3-b2-8k` **T_op 150.7 ms**（基线 455.3 ms，**3.02×**）、`aic_cube_ratio` **25.2%**、
`aiv_vec_ratio` **45.7%**（验收 ≤227 ms / ≥22% / ≤55% 全部达成，且有 1.5× 时延余量）。

本 attempt 的价值不在「把设计跑通」（§3.3 伪代码与 4-D 原型已由设计期验证），而在**三处只有真机才能给的结论**：
① 生产 shape 的单价是 **2.97 µs/迭代**（设计外推 3.42–3.49，高估 15%）；
② L1→L0 的实测有效带宽只有 **300 GB/s/核**（设计 1.6.4 G 的未核验假设是 1.0–2.0 TB/s），
   故 L0 单版本的 MTE1 暴露 **0.67 µs/迭代**，**越过设计写的 0.5 µs 回退阈值**；
③ 而设计预置的 `Nk=80` 回退**根本不可编译**（`nd2nz_copy.h:59 static_assert ROWS % 16 == 0`，
   2:1 半区 40 行）——该回退阈值在本结构下**无可用动作**，必须上报而不是照抄执行。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Develop-flow | 设计的容差口径（DESIGN §8.2「bf16 强抵消元素…判据沿用把参考实现的 dS 单独 cast 到输入 dtype 后重算」）只是散文，harness 必须自己把它落成代码；否则首个 L0 会把 **524288 个元素里的 1 个**（dv 0.00516 vs atol 0.005）判成 `[PRECISION_FAIL]` | `_bwd_kernel.py::run_case`（`round_inputs=True` 的 `golden_bwd_kernel` + 自动 fallback 断言）；`history_version` 对照：L0 首跑 rep0/rep2 FAIL（`/tmp/opencode/L0_cube.log`）→ 加 fallback 后 5/5 PASS（`L0_cube2.log`）；判别值 kernel-vs-same-rounding 2.3e-3、kernel-vs-fp64 7.8e-2 | tilelang-op-develop 的 `run_case` 模板内置「同舍入参考」二级判据：一级按设计容差比对，失败时用「P/dS 先 cast 到输入 dtype 再重算」的参考复核，并把越界元素数写进结果行 | P |
| Info-source | 设计预置的回退（§1.6.3 / §1.6.4 G 的 `Nk=80`，触发条件「MTE1 >0.5 µs/迭代」）**不可编译**：UB→L1 的 `dual_copy` 被 `InsertNd2Nz` 改写后要求每 AIV 行数 % 16 == 0，`Nk=80 ⇒ HN=40` 直接 static_assert 失败 | `repro/TRAP-950-nd2nz-rows-multiple16.py`（block_n=96 通过、80 报 `nd2nz_copy.h:59`）；`_bwd_kernel.py` 的 `T.dual_copy(p16_u, pT_s)` | 设计阶段对「每 AIV 半区行数」增加一条硬约束：凡 UB→L1 `dual_copy`，`block_n/2 % 16 == 0`；回退阶梯只能取 32/64/96/112/128，不能取 80 | D |
| Develop-flow | 设计的「每迭代缓冲 2 版本」预算（`T.annotate_buffer_versions`，§3.2 步骤 11 / §6.3）在本 kernel 上并未发生：该 API 面向 `T.Persistent + Z3` 自动调度器，本 kernel 用 `T.Pipelined`，Z3 报 `solver returned unsat, falling back to topological sort`，实测只有 `qT_s` 拿到 2 版本（L1 实占 229,376 B），L0A/L0B/L0C 全单版本（容量所限，本就无法多版本） | 设备码（编译缓存 `device_kernel.asc`）：`buf_dyn_l1 + ((kq - (n0>>6)) & 1) * 12288` 仅出现在 qT_s；L0A 24,576 elem / L0B 30,720 elem / L0C 55,296 elem 恰等于单版本预算；UB 116,992 B | tilelang-op-develop Phase 4 增一步「多版本核实」：从设备码的 buffer 偏移项反查每个片上缓冲的实际版本数，不要以 `num_stages`/预算表推定 | D |
| Info-source | 设计用「小 shape（S=2048/H=8）的每迭代单价 × 迭代数」外推生产 shape，得到 173–176 ms；实测 **150.7 ms**（2.97 vs 3.47 µs/迭代，高估 15%） | `repro/CONST-950-mla-bwd-cube-split.py`（T_op 150.637 ms）；`_bwd_kernel.py --level big`（150.686 / 150.753）；msprof Task Duration 151,006 / 150,926 µs | 设计期的外推口径加注：小 grid（<2 波）单价含尾波与 L2 冷启动，生产 shape 需按「单片 ≥10 波」重标定，或直接把外推标为保守上界 | D |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| MLA prefill 反向主 pass 的 Cube 化生产常数：`ds-v3-b2-8k` T_op 150.7 ms / 2.97 µs 每块迭代 / cube 25.2% / vec 45.7% / MTE1 0.668 µs 每迭代（299.7 GB/s per core）/ MTE2 118.5 GB/s per core；AIC 四管道忙时之和 2.74 µs ≈ 迭代 2.97 µs（**单 AIC 内无重叠**） | D | `examples/mla_dense_bwd/_bwd_kernel/evidence/msprof_pipe_capture1.csv`（22016 块均值，两次独立采集同值）；`repro/CONST-950-mla-bwd-cube-split.py` | `TMPDIR=/workspace /workspace/venv-ascend950/bin/python examples/mla_dense_bwd/_bwd_kernel/repro/CONST-950-mla-bwd-cube-split.py --time` | tilelang 0.1.15 / CANN 9.2.1 / ascend-950pr-9579 | 平台 knowledge/constants.md（新增 L1→L0 MTE1 有效带宽条目）+ bottlenecks.md（「AIC 内 cube/mte1/mte2/fixpipe 串行相加」条目） |
| 混合 kernel 的 UB→L1 `dual_copy`（`InsertNd2Nz` → `ascend_nd2nz_scatter`）要求每 AIV 行数 % 16 == 0：`block_n=80`（半区 40 行）**不可编译**，`block_n=96`（48 行）可 | D | `repro/TRAP-950-nd2nz-rows-multiple16.py`（两条断言：96 通过 / 80 报 `nd2nz_copy.h:59 static_assert ROWS % 16 == 0`） | 同上脚本 | 同上 | 平台 knowledge（nd2nz 形态边界）+ tilelang-mixcv-skill 的 dual_copy 约束段 |
| 4-D batch 视图把 `dk/dv` 的 store clamp 收到本 request：设备码 `min(48, max(((512 - sid*48) - (bid%6)*96), 0))`（**不是** `B*S_kv=1024`），与源算子的逐行守卫同义；5-D 以外任何 packed 视图都会跨 request | D | `repro/TRAP-950-nd2nz-rows-multiple16.py` 的结构断言（`"min(48, max(((512 - "` 且 `"1024" not in store`）；设计期证据 `history_version/b1_device_code_bounds.txt` | 同上脚本 | 同上 | 平台 knowledge（`AscendInsertOOBPadding` 的 clamp 维度 = buffer 形状） |
| bf16 强抵消输出的「1/5e5 元素」判据：kernel-vs-fp64 7.8e-2、kernel-vs-同舍入参考 2.3e-3 ⇒ 固有舍入而非缺陷；实现形态 = golden 的 `round_inputs=True` + 一级断言失败时自动二级复核并统计越界元素数 | P | `_bwd_kernel.py::golden_bwd_kernel(round_inputs=)` / `run_case` 的 `_checks` fallback；L0 首跑 2/5 FAIL → 复核后 5/5 PASS | `python _bwd_kernel.py --level L0`（L0-9 rep0/rep2 打出 `[1 element(s) over the fp64 atol: inherent bfloat16 rounding…]`） | 同上 | tilelang-op-develop 的 `run_case` 模板 + TileOPs `numerics.py` 容差口径注 |

### Transferable Lessons（可迁移教训）

- **4-D batch 视图是 `dk/dv` 写回正确性的前提，不是风格选择**：本平台 `AscendInsertOOBPadding` 的 clamp 维度是 buffer 形状的行数，
  3-D packed 视图只能裁到 `B*S_kv`（B≥2 且 `S_kv % block_n != 0` 时尾块覆盖下一个 request）。本实现全部 4-D，设备码与逐 request 比对双重验证。
- **每 AIV 半区行数必须 % 16 == 0**（UB→L1 dual_copy 经 nd2nz）：`block_n` 的可行集不是任意 16 的倍数，而是「半区也是 16 的倍数」，即 32 的倍数。
  设计阶段就应把这条写进分块约束，否则回退阶梯会给出不可编译的候选（本次 `Nk=80` 即如此）。
- **L1→L0 的单版本代价按 ~300 GB/s/核量级估**，不要用 1 TB/s 假设：217 KB/迭代 ≈ 0.67 µs，占 2.97 µs 迭代的 22%；
  在 Cube 只占 25% 的结构里，MTE1 与 cube 同量级，压 MTE1 与压 cube 同等重要。
- **小 shape 单价不能直接外推生产时延**：同一结构在 S=2048/H=8 是 3.47 µs/迭代，在 B=2/S=8192/H=128 是 2.97 µs/迭代（差 15%，
  尾波与派发摊薄方向相反）。外推值应标为保守上界。
- **容差判定要一次成型**：bf16 输出上「1 个元素超 atol」几乎必然是固有抵消舍入，先用「同舍入参考」复核再判 PASS/FAIL，
  否则每个新 shape 都会随机冒出一条假失败（本次 L0-9 的 rep0/rep2 即为实例）。
- **`num_stages` / `annotate_buffer_versions` 与 `Pipelined` 不是一回事**：本例 Z3 调度器 unsat 回退拓扑排序，只有 1 个 L1 缓冲被双版本；
  片上缓冲的版本数必须从设备码偏移项反查，不能由预算表推定。

## Stage 1 (design) Retrospective — revision v4（A5 旋转调度 / 多缓冲 epilogue）— 2026-11-26

结论：`DESIGN_COMPLETED`。本轮的输入不是检视意见，而是 Stage 4 的 `[DESIGN_LIMIT]` 归因：**结构无问题、
参数无空间，残差在语句顺序里**。v4 因此**只改时序**（`GB(k)={dV,dK,dQ}` 滞后一迭代），算法族、数学、
4-D 视图、grid、L0C/L0A/L0B 一律不动。

**本轮最关键的一条是「否决上游给你的处方、采纳它的归因」**：`perf_feedback.md` 给出的路径是
「把 `dP` 并入 `st_c` 以腾出 L0C 双版本」，那是**两级旋转（`GA` 也提前）**的必要条件；而 S1（只滞后 `GB`）
**只需要 L1 的两份装载缓冲 + `pT/dsT` 两份**，L0C 逐字节不动。若照抄处方，会白白引入 `GA` 内部的
fixpipe→mad 串行，并且把 L0C 推到 103.1%（越界）。区别的判据是依赖矩阵（哪条边是 RAW、哪条是 WAR）。

量化结果（本算子探针，不是外推）：`B=2/S=8192/H=8` **2.965 → 1.870 µs/迭代（−36.9%）**、
`B=2/S=4096/H=16` **3.069 → 1.968（−35.9%）**；`dk/dv` 与 v3 **逐位相同**（5/5）；
`T_op` 投影 **95–101 ms**（v3 154.4），cube **38–40%**（v3 25.1%）。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Design-flow | Stage 4 的 `[DESIGN_LIMIT]` 会附带「改设计建议」（本例：dP 并入 `st_c` + L0C 双版本），而 skill 的 revision 流程没有要求**先判定该建议对应哪一种调度方案（S1/S2）再做容量核算**。照抄会把「S2 的必要条件」误当成「S1 的必要条件」，代价是 L0C 103.1% 越界 + 新增串行 | `perf_feedback.md`「实测」第 3 条 vs 本设计 §6.3 E；本版实测 S1 只需 L1 +106,496 B、L0C 不变 | skill 的 revision 输入处理加一条：**上游处方必须与「依赖矩阵」逐边对齐**，并给出该处方在**最小改动方案**下的容量复算；处方与归因分开记录 | P |
| Design-flow | 「跨迭代软件流水」在 skill 里被当作 Stage 4 的调优项（`T.Pipelined`/`num_stages`），但没有提示「当调度器 unsat 时，重叠只能靠显式语句顺序」这一分支 | 本任务 13 个 `num_stages` 分支全在 ±3%；同族 varlen 与 dense 的 `z3_*.log` 均 `unsat` | skill 的 Phase 2/§6 增加分支：**若目标平台调度器对该 kernel `unsat`，跨迭代重叠必须写成「显式版本 + 语句重排」**，并给出「缓冲身份切断 WAR」的判据 | P |
| Design-flow | 数值等价「散文论证」已被 D-1 机检替代，但**调度级等价（只改语句顺序）**没有对应的机检形态；本轮沿用 varlen 的两级法（调度代数 + 设备逐位）才补上 | `verify_equiv.py::check_a5`（478 点/17,846 事件/selftest 3/3）；`design_probe_dense_rotation.py --parity`（5/5 逐位） | skill 的 D-1 增加第三类对象「调度级变更」的验证方式：① 事件流代数检查 + **反例自检**（必须能抓住已知缺陷）② 与冻结实现逐位对照 | D |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **S1 旋转（`GB` 滞后一迭代）只需要 L1 双版本，不需要 L0C 双版本**：`st_c/dpt_c/st_u/dp_u/p16_u/dsT_u` 的生存期都在同一迭代内；L1 = `98,304（常驻）+ 2×(81,920 + 24,576)` = 311,296 B（59.4%），L0C 221,184 B（84.4%）逐字节不变。两级旋转（S2）才需要 `st_c/dpt_c` 双版本（L0C 103.1% 越界） | D | 本轮探针：`rot` 与 `base` 的 L0C/L0A/L0B 声明逐字相同且编译通过；§1.6.4 I-1/I-4 | `TMPDIR=/workspace /workspace/venv-ascend950/bin/python examples/mla_dense_bwd/_bwd_kernel/history_version/design_probe_dense_rotation.py --time` | tilelang 0.1.15 / CANN 9.2.1 / ascend-950pr-9579 | 平台 knowledge/constants.md（MLA 反向 S1/S2 的缓冲增量）+ bottlenecks.md（「迭代内跨核 RAW 链」的诊断与处方分层） |
| **dense MLA 反向的旋转实测单价**：`B=2/S=8192/H=8` 2.965→**1.870 µs**（−36.9%）、`B=2/S=4096/H=16` 3.069→**1.968**（−35.9%）；同几何 v3 体探针 2.965 vs 生产 2.944（**+0.7%**，可直接外推）；`dk/dv` 与 v3 **逐位相同** | D | `design_probe_dense_rotation.py --time/--parity`；生产锚点 `perf_feedback.md` 实测第 1 条 | 同上（`--time`、`--parity`） | 同上 | 平台 knowledge/constants.md（dense MLA bwd 的 per-iteration 前后对照）+ `PL-950-mla-bwd-dense-chain-bound` 的状态更新（「链受限」→「S1 后可解，残差 0.42 µs 未分解」） |
| **A5 的两条结构性判据（可机械检查）**：① 生成代码里 `GB` 的三组 `L1→L0` + 三个 `asc_mmad` 之前**不得**有以 `PIPE_V` 为前驱的 `asc_sync_wait`；② `asc_mmad` 计数应为 `5×半体数 + 3×drain 分支数 = 16`、`__simt_vf__` 应为 `2 臂×2 半体+块尾 = 5`。二者可作为「旋转是否真的生效」的准入判据 | D | §1.6.4 I-1（`asc_mmad` 5→16、`simt_vf` 3→5、`l12l0a` 5→16）；varlen v5 的同一判据已在生产使用 | 同上 `--compile` + 生成源计数 | 同上 | 平台 knowledge/`PL-950-*` 新增「延迟 GEMM 旋转的准入判据」条目 |
| **代码体积是 A5 的确定代价**：生成源 20,341 → **50,879 B（×2.50）**；但同族的 varlen v5 是 61,913 B 且已上线 ⇒ icache 暴露有上界。判断「能不能接受」应看**绝对量**而不是倍数 | D | §1.6.4 I-1；varlen v5 生产在跑 | 同上 `--compile` | 同上 | tilelang-op-design 的「多缓冲/展开的代价」段 + 平台 knowledge（展开倍数 vs 绝对体积的口径） |
| **族内结论的方向不能外推**：varlen 是长序列收益**更小**（−26.5% vs −33.9%，因 packed 动态 DMA 在旋转后变成第一忙管道），dense 是长序列收益**更大**（−36.9% vs −35.9%，因静态 extent 的 MTE2 快 47% 且 L1 在 v3 已是 311 KB）；两者共享同一调度机制但**收益归因不同** | D | `mla_varlen_bwd/_bwd_kernel/RETROSPECTIVE.md`（mte2 0.86 share / 55.5 GB/s）vs 本设计 §1.6.4 I-4、风险 19 | 两族各自的 `--time` 脚本 | 同上 | 平台 knowledge（「同机制跨 workload 的收益归因必须逐族取证」）+ op-design 的复用规则 |

### Transferable Lessons（可迁移教训）

- **先看依赖边类型，再决定要不要加版本**：RAW（生产→消费）只能靠**换缓冲身份**（滞后 + 另一份），
  WAR（读者 vs 下一个写者）才靠加版本。把两者混为一谈会得出「多给几个版本就能重叠」的错误结论——
  这正是本任务 13 个 `num_stages` 分支全部无效的原因。
- **上游的「处方」要回到它的适用条件**：`perf_feedback.md` 的「dP 并入 `st_c`」在 S2 下必需、在 S1 下有害。
  判据是「该处方服务于哪一条依赖边」，而不是它在原报告里写得多具体。
- **探针必须能外推到生产**：本轮先在同一形状上比对 `base` 体与生产（+0.7%）再报 `rot` 的数字，
  这样 1.870 才有资格乘迭代数。varlen 的教训是同一步没做足（探针 −33.9% vs 生产 −17.8%），
  差异全部来自族内 DMA 行为的改变。
- **「逐位相同」是调度级重排最强的等价证据**：比 fp64 对照更敏感（fp64 容差会掩盖取错版本导致的小偏差，
  而取错版本在 `dk/dv` 上是成片差异）。凡「只改顺序不改算式」的改动，都应争取一个逐位对照。
- **验证器要有反例自检**：`check_a5` 必须能抓住「drain 奇偶取反 / 删 drain / 读当前版本」三种缺陷
  （实测 3/3 抓住）。没有反例自检的检查器，通过率没有意义。
- **容量收益可能来自「删掉不需要的版本」**：v3 给 `st_u/dp_u/p16_u/dsT_u` 也编了第二版（65.7%），
  A5 按生存期判定它们只需单版本 ⇒ UB 反而降到 51.2%。多缓冲的预算应按**生存期**算，不按旋钮算。

## Stage 2 (review) Retrospective — revision v4（A5 旋转）— 2026-11-26

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Review-flow | **结构判据的「区分力」没有跨任务回灌**：varlen v5 的检视已判定「`GB` 前无 `PIPE_V` 前驱 wait」在**未旋转**体上同样成立（无区分力、会假阳性），但 dense v4 仍把它作为唯一结构判据（P11/风险 18）。同族复用时这条教训没有进入设计/检视模板 | varlen `REVIEW.md` 建议 5-1；dense `DESIGN.md` §8.3 P11、§9.1 风险 18；本轮 REVIEW 建议 1 | tilelang-design-review §3 维度 6 增一条：「任何**结构判据**必须附一次**反例自检**（在已知缺陷体上必须失败），否则只能记为辅助」；tilelang-op-design 的验证方案模板同步 | R |
| Review-flow | 「未解释残差」可以直接用设计自己的分项数字做一次**可加性检验**，而不必一律推给 Stage 4 的 msprof：本轮 AIV 1.304 + fixpipe 0.570 = 1.874 ≈ 实测 1.870–1.874（≤0.2%），而设计归因的「块级固定开销」复算只有 ≈0.03 µs/iter（1.6%） | dense `DESIGN.md` §6.3 F、风险 17；perf_feedback 管道忙时；本轮 REVIEW 建议 2 | 维度 8 的「调研结论正确性」增一行：「每个『残差未分解』必须至少给一个**候选归因 + 证伪条件**；能用设计自己的分项数字凑出来的，不得写成未分解」 | P |
| Info-source | `design_calc_check.py` 的 `ub_budget` 因 §4.5 新增「版本」列而由 pass 变 **skip**（静默失去机检覆盖）。机检工具对文档表格的**列集合**脆弱 | 本轮机械复核 JSON：`ub_budget … "no per-level buffer tables parsed"`；v3 同一文档为 `pass (163.0 KB)` | `design_calc_check` 的 UB 表解析容忍额外列（按列名取值）；或 design skill 要求 §4.5 保留一行工具约定的「合计」行 | R |
| Review-flow | 跨族复用（varlen v5 → dense v4）时，「哪些能复用、哪些必须重算」的**差异表**是本轮质量最高的部分（§14.2 的 11 行）；但它目前是设计方的自发实践，检视侧没有对应检查项 | dense `DESIGN.md` §14.2；本轮 REVIEW 维度 8 | 维度 0/8 增一条：「复用同族结构时必须给出逐维度差异表，并标明每项是「直接复用 / 结构复用但参数重算 / 必须重写」，且每项有本算子的取证」 | P |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **A5 旋转在 dense 4-D / 静态 S / 无 slab 结构上实测 −36.8%**（2.965→1.874 µs/迭代，B=2/S=8192/H=8；`dk/dv` 与 v3 **逐位相同**、parity 5/5、numeric 5/5），且代码体积 20,461→51,239 B（×2.50）、`asc_mmad` 5→16、`__simt_vf__` 3→5 | D | dense `DESIGN.md` §1.6.4 I-1/I-3/I-4；`history_version/design_probe_dense_rotation.py`；`history_version/dense_rot_probe_run.log` | `cd /tmp && source …/env-950pr.sh && TMPDIR=/workspace /workspace/venv-ascend950/bin/python …/design_probe_dense_rotation.py --{compile,numeric,parity,time}`（我四种模式全跑，数值与设计日志逐位同值） | tilelang 0.1.15 / CANN 9.2.1 / ascend-950pr-9579 | 平台 knowledge（`execution-patterns` 增「跨迭代语句重排」条目）+ `tilelang-mixcv-skill` |
| **跨迭代旋转的可判定结构判据 = 读偏移身份**：反查 12 个每迭代 L1 缓冲基址，断言半体 A 的 `GB` 读版本 B 的基址、`F/EP` 写版本 A（半体 B 相反），`dq` 用 `(kq−1)` 行区间，drain 按 `n_it` 奇偶；「`GB` 前无 `PIPE_V` 前驱 wait」在两版体上都成立，**不可作判据** | P | varlen `REVIEW.md` 建议 5-1（设备码偏移已验）；dense `DESIGN.md` §7.2/§8.3 P11 | varlen 检视的 M3 行（12 个偏移：`49152/61440/…/149504`，末地址 `155,648×2 = 311,296 B`） | 同上 | tilelang-design-review 维度 6 + tilelang-mixcv-skill 的「旋转体验证」小节 |
| **「单版本缓冲的 WAR 会把两单元重新串起来」的指纹 = 残差 ≈ 对方忙时之和**：dense v4 的 AIV 1.304 + fixpipe 0.570 = 1.874 与实测 1.870–1.874 相差 ≤0.2%，而 `st_u/dp_u` 正是单版本；对应的旋钮只花 24,576 B UB（51.2%→60.9%） | P | dense `DESIGN.md` §6.3 F（残差 0.42「未分解」）、§7.2 第 3 行、风险 16/17；perf_feedback 管道忙时 | 纸面：`1.304+0.570 = 1.874` vs `§1.6.4 I-4` 的 1.870；`98,304 B ÷ 118.5 GB/s ÷ 64.5 = 0.013 µs/iter`（块级开销） | 同上 | `_shared/standards/hardware-cost-model.md` + tilelang-design-review 维度 8 |

### Transferable Lessons（可迁移教训）

- 跨迭代旋转（把 `GB(k)` 挪到 `EP(k+1)` 旁）的正确性判据是**读偏移身份**：`GB(k−1)` 的 L1→L0 源必须落在**另一版本**的基址，首迭代无 `GB`、末迭代由 drain 按 `n_it` 奇偶补，`clear_accum` 恰好一次、`dq` 用 `(kq−1)` 行区间。「`GB` 前没有 `PIPE_V` 等待」在未旋转体上同样成立，**不能**当判据；必须配「删 drain / 读当前版本」两个负反例自检。
- 只要某块缓冲是**单版本**，它的 WAR 就会把两个执行单元重新串起来：此时「残差 ≈ 对方忙时之和」往往是该 WAR 的指纹（dense: AIV 1.304 + fixpipe 0.570 = 1.874 ≈ 实测）。先做这个可加性检验，再决定要不要去 msprof 分解。
- 机检工具（`design_calc_check`）对文档表格的列集合敏感：给内存预算表加一列「版本」就会让 `ub_budget` 从 pass 静默变 skip。改表后必须重跑一次机检并确认没有 skip 项。

## Stage 3 (develop) Retrospective — first_impl attempt 1（A5 旋转落地 / v4 冻结设计）— 2026-11-26

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Develop-flow | **平台 `execution.prototype_command` 指向的解释器没有 torch/tilelang，且 cwd 在仓根会遮蔽 site-packages**；设计 §3.5.4 已登记偏离，但没有任何机检工具把「实际调用方式」交付给 Stage 3，只能靠读设计文档 | `statectl platform --task-dir` 返回 `/usr/local/python3.12.16/bin/python3` + `working_directory=<task_dir>`；实测 `import tilelang` 在仓根解析到 dev 源码（`TileLang's private TVM Python package is missing`），在 `/tmp` 解析到 site-packages 0.1.15 | 平台 profile 的 `execution` 增 `env_script` / `cwd_policy` 字段（本例应为 `env-950pr.sh` + 仓根之外），`statectl platform` 直接返回可执行命令 | R |
| Develop-flow | **设计文档的用例编号与本文件已有编号冲突**：v4 设计的 L0-12…L0-15 在本文件里已被 v3 的 golden-vs-naive / contract / atomic / arm-equivalence 占用，落地时只能另编 17…20，跨文档可追溯性变差 | `DESIGN.md` §8.3 L0-12…L0-15 vs `_bwd_kernel.py::run_L0` 的 L0-12…L0-16 | design skill 要求新增用例编号在 `DESIGN.md` 里声明「相对现有实现的偏移」，或由 develop 在返回摘要中给出映射（本轮已给） | P |
| Info-source | **设计里两个用例的 `n_it` 归属算错**：`L0-13 B=1,S=96,H=1` 被写成 `n_it=1（奇）`，但 `BLOCK_M=64` ⇒ `loop_ed=ceil(96/64)=2` ⇒ `n_it=2`（偶），即它与 L0-12 一样测的是偶数分支；奇数分支需要 `S=64` 或非因果几何 | `DESIGN.md` §8.3 L0-12/L0-13；本轮 `history_version` 反例探针：`S=96` 上删掉 drain 的**偶数**分支才失败（`S=64` 才是奇数） | design skill 的用例表增加「该用例触发的分支/奇偶」一列，并要求用 `loop_st/loop_ed` 的显式算式推导一次 | D |
| Develop-flow | 设计/检视都要求「`dk/dv` 与 v3 逐位」进 L1，但**没有说明 v3 体从哪来**：交付件被拷进 TileOPs 包后 `history_version/` 不存在 | 本轮实现：`_load_v3()` 按 `__file__` 旁路找冻结副本，缺失时记 `skipped`（L1 不因此失败） | develop skill 的「分层测试套件」模板增一条：跨版本逐位回归的门槛要写明**基线副本的落盘位置与缺失时的降级语义** | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **A5 旋转在 dense 生产几何上落地：T_op(`_bwd_main`) 150.7 → 95.19 ms（−36.8%），AIC 四管道由「串行相加」变「重叠 1.50×」**（pipe-ratio 和 0.914→1.501），cube 25.2%→**39.7%**、vec 45.7%→**86.0%**，`aic_mte1/mte2_active_bw` 282/121 GB/s，`icache_miss_rate 0.0` | D | `evidence/v4/`（`PipeUtilization.csv` 22016 块均值、`ArithmeticUtilization.csv`、`OpBasicInfo.csv` Task Duration 95,677 µs）、`evidence/v4/{v4_L0,v4_all,v4_big}.log`、`repro/CONST-950-mla-bwd-cube-split.py` | `cd /tmp && source …/env-950pr.sh && TMPDIR=/workspace msopprof --output=… --launch-count=1 --warm-up=0 --kernel-name=_bwd_main_kernel --aic-metrics=BasicInfo,PipeUtilization,ArithmeticUtilization /workspace/venv-ascend950/bin/python repro/CONST-950-mla-bwd-cube-split.py --time` | tilelang 0.1.15 / CANN 9.2.1 / ascend-950pr-9579 | 平台 knowledge `execution-patterns.md`（「跨迭代语句重排」条目补实测：AIC pipe 重叠 1.5×）+ `bottlenecks.md` |
| **旋转后墙从 AIC 换到 AIV：AIV vec 忙时 1.356 → 1.611 µs/迭代（+19%，源码级算式未变）**，vec 占空 86% 已越过 80% 警戒线；AIC 只剩 39.7% cube、管道重叠 1.5× 有余量 ⇒ `st_u/dp_u` 第二版（REVIEW 建议 2 的 Stage 4 首项）**不是**当前残差来源 | D | `evidence/v4/PipeUtilization.csv` vs `evidence/v4/recheck_v3_same_tool/PipeUtilization.csv`（同工具、同 shape、同 `--launch-count=1` 复采；v3 旧 `evidence/msprof_pipe_capture1.csv` 的 `aic_time` 375.9 µs/块是其 2× 口径伪影，与 150.7 ms 墙钟不自洽，本轮已用同条件重采 188.8 µs/块） | 同上，把 `--impl` 换成 `history_version/_bwd_kernel_impl_s3_v4_attempt1.py` 即 v3 体 | 同上 | 平台 knowledge `bottlenecks.md`（新增「旋转把 AIC 串行链切开后，AIV epilogue 成为新墙」）+ `tilelang-op-optimize` 的 A/B 排序 |
| **「删 drain / 读当前版本」两个负反例在设备上确有判别力**：删 drain（奇/偶两支各隔离）与半体 A 的 `GB` 改读 `_s0` 全部被 golden 抓到（69.4% / 93.1% / 97.0% / 78.5% 元素超差） | P | 本轮 `repro/PROBE-950-a5-rotation-negatives.py` 输出（`ALL_CAUGHT`）；`_bwd_kernel.py::_gradients`/drain 的补丁锚点 | `cd /tmp && source …/env-950pr.sh && TMPDIR=/workspace /workspace/venv-ascend950/bin/python …/repro/PROBE-950-a5-rotation-negatives.py`（对源码做文本替换后编译运行，自带断言） | 同上 | tilelang-design-review 维度 6（结构判据必须配反例自检） |
| **偏移身份判据可机检**：`device_kernel.asc` 里 12 个每迭代 L1 缓冲基址（A: 49152/61440/98304/106496/131072/137216；B: 73728/86016/114688/122880/143360/149504）——半体 A 的 `GB` 读 B 组、半体 B 读 A 组、drain `if`（奇）读 A 组 / `else`（偶）读 B 组；`dq` 行偏移 `(kq−1)·64`（半体 A 表达式带 `−98304`、B 不带）、drain `12484608 = 127·64·1536`；`asc_mmad=16`、`__simt_vf__=5`、`device_kernel.asc` 51,294 B | P | `evidence/v4/offset_identity.txt`；判据落地脚本 `history_version/v4_offset_identity.py`（5 条断言，退码即回滚触发） | `python history_version/v4_offset_identity.py <kernel-dir>/device_kernel.asc` | 同上 | tilelang-mixcv-skill 的「旋转体验证」小节 + 平台 knowledge `repro/`（可作为 PL-950 条目） |
| **`dk/dv` 与冻结 v3 体在 B=2/S=8192/H=8 上逐位相同（causal 与非因果各 3 次），且 v4 自身跨 launch 可复现**；`dq` 相对差 ≤1.9e-8 / 1.8e-7（原子到达序） | D | `evidence/v4/v4_all.log`（`L1-7`/`L1-8` 行）；实现 `_bwd_kernel.py::_case_rotation_parity` | `TMPDIR=/workspace /workspace/venv-ascend950/bin/python _bwd_kernel.py --level L1` | 同上 | tilelang-op-develop 的「跨版本逐位回归」模板 |

### Transferable Lessons（可迁移教训）

- 跨迭代旋转（把 `GB(k)` 挪到 `EP(k+1)` 旁）的判断要**先看偏移身份再看数值**：`device_kernel.asc` 里每迭代 L1 缓冲的基址是唯一能证明「版本真的交叉」的证据；数值通过也可能是版本根本没生效（同版本小 shape 也能碰巧对）。
- 同一 shape 的 msprof 数字**必须用同一 `--launch-count` / 同一工具版本重采后才能跨版本比较**：本任务的 v3 旧采集 `aic_time=375.9 µs/块` 与 150.7 ms 墙钟不自洽（2× 口径伪影），用同条件重采后是 188.8 µs/块；只用「ratio」跨采集比较同样会踩这个坑。
- 旋转削掉 AIC 的串行链之后，**被释放的那一侧不再是限制**：必须立刻重算「谁界定墙钟」（本轮 AIV vec 86% > AIC cube 39.7%），否则会把下一阶段预算花在有余量的那一侧（`st_u/dp_u` 第二版）。
- 「单版本缓冲的 WAR 是残差来源」这类归因，先用「两单元忙时对比 + 各自占空」证伪一次：本轮 AIC 空闲/占空比 AIV 低一半，说明残差不在 AIC 的 WAR 上。
- 用例的奇偶/分支归属必须用 `loop_st/loop_ed` 的显式算式推一遍再写进设计（`BLOCK_M=64` 时 `S=96 ⇒ n_it=2` 是偶数）；否则「测奇数分支」的用例实际测的是偶数分支，反例自检会给出错误的判别力结论。
- 逐位回归门槛里的「冻结基线」必须明确落盘路径与缺失降级语义（交付件被拷进集成包后基线目录通常不存在），否则这条门槛在 Stage 5 会静默失效或直接报错。
