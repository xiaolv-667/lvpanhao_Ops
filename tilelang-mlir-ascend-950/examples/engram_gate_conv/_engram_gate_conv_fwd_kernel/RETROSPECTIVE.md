# Stage 1 复盘 — `_engram_gate_conv_fwd_kernel`（Engram GateConv 前向，迁移）

> 追加式记录（Phase 8 自进化钩子）。每次 `DESIGN.md` 产出后追加一节；本节 = v1（first_design）。
> 字段规范：`tilelang-skill-evolution` skill 的 `references/retrospective-schema.md`。

## v1（first_design，2026-11-23，平台 ascend-950pr-9579 / tilelang 0.1.15 + CANN 9.2.1）

### 本次设计的结论摘要

- 结构：**单 launch**（源算子 / 950DT 参考均为两段 launch），每核连续 `RC` 行 + 3 行 halo 重算 + UB 4-slot 环形窗口；
  备选（两段 launch、每行一核）随文档冻结，交 Stage 4 A/B（阈值：备选 < 主选 × 0.9）。
- 公式：4 项等价优化（A1 门控点积融合、A2 `w_v` 折进 conv 权重、A3 conv 输入行每行一次、A4 `√d` 常量折叠），
  `verify_equiv.py` 72 用例/项全部 `EQUIV_PASS`（`Y` 1–3 ulp、`alpha` 9 ulp、容差预算 ≤ 0.18%）。
- 分核：`grid = M·(seq_len/RC)` = 32/32/32/48，全部 ≤ 56（无需适配，1 波）。
- UB：44.6 KB / 248 KB（18%，`d=768`）。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| tilelang-op-design / Phase 2 §1.6.3 | 迁移任务的「备选方案」要求「结构完整到 Stage 4 可直接实现」的深度，但未要求给出备选的**分核三要素重算**与 UB 预算——本设计主动补了（§1.6.3 备选段）。建议模板显式列出备选必填项 | DESIGN.md §1.6.3 备选段 | 在 design-template 的 §1.6.3 实验裁决模式里加一条「备选方案必含：布局/循环结构、UB 预算、分核三要素重算、dtype 路径」 | P |
| tilelang-ascend / 平台规则 | `references/programming.md` 未记载「SIMD_VF 块内禁止访问 GM」这条语义检查；它是编译器强制执行的前端检查（探针首轮即被拒），设计期若不知道会写出不可编译的伪代码 | `history_version/design_probe_compile.py` 首轮报错：`[TileLang Semantic Check] BufferLoad from global buffer 'WH' inside a SIMD_VF block is not allowed` | 在 `references/programming.md` 的「API 与执行区域」表加一行：`T.SimdVF()` 内禁止 GM `BufferLoad`（参数/输入必须先 `T.copy` 到 UB） | D |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| `T.clear` 对大 fp32 UB 缓冲降级为标量 `float2` 循环 | D | 探针首轮生成代码：`for (int32_t i = 0; i < 1536; ++i) *(float2*)(…) = *(float2*)(slot_local_cast)` ——12 KB 缓冲的清零变成 1536 次 8 B 标量存；改用向量 `vsts` 后生成 `vsts_norm(zero_p, …, full_p)` | `history_version/design_probe_compile.py`（改前/改后各跑一次，`grep float2` 从 1 行变 0 行） | tilelang 0.1.15 + CANN 9.2.1 / ascend-950pr-9579 / 2026-11-23 | `.agents/platforms/toolchains/ascend/knowledge/PL-950-vector-execution.md`（追加「清零用向量存」条目）或 `bottlenecks.md` |
| 融合（单 launch）替代跨 kernel 物化的可行性与判据 | P | 源算子与 950DT 参考都用「中间量落 GM + 第二 kernel」；本设计用「核内连续行所有权 + 环形窗口 + 3 行 halo 重算」消掉它，代价可量化（halo `3/RC`、无跨核 barrier 需求） | DESIGN.md §0.6 R3 / §1.6.0 R3（3.11 MB vs 4.66 MB） | 同上 | `.agents/skills/tilelang-op-design/references/algorithm-research.md` §5 参考表（「窗口/跨阶段融合」族加一行：跨行窗口依赖 → 核内连续所有权 + halo 重算） |
| 「`RC` 必须整除 `seq_len`」消除尾块代码路径 | P | 尾块（tail chunk）是这类 tile 规则的常见缺陷源（`examples/mla_dense_bwd/_delta_kernel/DESIGN.md` §9.2 记载了「尾块沿用 `R` extent」导致越界且编译器不报错）；本设计用「约数规则 + `RC=1` 兜底」让尾块**不存在** | `history_version/design_probe_compile.py` 的 `divisors()` 规则 + `prime-95` 几何 | 同上 | `.agents/skills/_shared/standards/core-split-strategy.md` §1 要素③（补一句「可用整除约束消除尾块路径」） |

### Transferable Lessons（可迁移教训）

1. **先跑编译探针再写伪代码**：本任务的两处结构错误（SIMD_VF 内读 GM、`T.clear` 的清零降级）都是「静态读文档看不到、
   编译器一跑就现形」的问题。探针（≤10 min，无卡）比事后 Stage 3 失败便宜一个数量级。
2. **ulp 指标的参照系必须写清楚**：`|cand−ref| / ulp(ref)` 在零/抵消元素上会给出 5–6 位数，看起来像灾难其实是分母问题；
   改为「按被评量 dtype 的网格」+ 同时给「尺度相对误差」与「容差预算占用」，结论才可读、可复核。
3. **迁移不是逐 API 翻译**：源算子的两段 launch 是 GPU「每 `(m,t)` 一 block」的产物，不是数学必需；把它当结论照抄会
   白白多付一次 launch 与一次 GM 往返。判据应回到「跨行依赖的宽度」（宽度 4 ⇒ 核内窗口足够）。
4. **`dtype` 舍入点要逐个点名**：本算子有 7 个舍入点（尤其 `r_v` 定义在**已舍入**的 `v̂` 上、残差用**未舍入**的
   `alpha·v`），任何一处颠倒都会被宽松容差（0.1）掩盖成「能过但语义错」。设计期把它们列成表（§0.1）比事后 debug 便宜。
5. **负向断言的证据要落到「查了什么」**：本任务 kb_search 无命中（如实记录）、`algorithm-candidates.md` 全表 6 族无命中、
   `examples/` 全量 ascend-target 算子扫过 —— 调研结论里的「无更优替代」才站得住。

---

## Stage 2 (review) Retrospective — first_review — 2026-11-23

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Review-flow | `design_calc_check.py` 对**非表格化的 UB 预算/分核规则**全部 skip（本设计 UB 表是 markdown 表、RC 规则是 python 代码块），复核只能靠人工；skip 项的「人工已核」又无强制记录位 | `REVIEW.md#机械复核`（`ub_budget/core_split = skip`，`r3_metrics = pass`） | 让 checker 至少解析 `\| Buffer \| Shape \| dtype \| Bytes \|` 表求和，并在 JSON 里给 `manual_required: true` 标记，便于机械核对「是否真的人工核过」 | R |
| Info-source | 检视 §0.3「源 kernel 语义」时，**源 kernel 与其自身 torch 参考的舍入点分歧**没有被维度 0 的任何检查项点名（现有 7 项只问「§0 与源码是否一致」），需要检视者自己发现 | `REVIEW.md#0` 建议级第 1 条；源 `gate_conv_fwd.py:108-113` vs `workloads/sequence_modeling.py:121` | 在 review skill 维度 0 增一行「源实现与源参考（ref_api/golden）的语义分歧是否已记录」；design skill 的 migration-analysis 同步加一条 | P |
| Info-source | 探针生成的 Ascend C 源码（`device_kernel.asc`）是复核「§0.6/§3.3/§7.2 引用的生成物」最直接证据，但 skill 未提它的位置与读法，检视者需自行从 `TILELANG_CACHE_DIR` 摸路径 | 本次按 `$TILELANG_CACHE_DIR/<ver>/linux-x86_64/kernels/<hash>/device_kernel.asc` 复核到 16 处 `asc_sync_*`、`(i_1 & 3)`、`6,4,…,4,32` | 在 review skill「Phase 1 读取与核对」加一步：探针声称的生成物证据可从缓存 `device_kernel.asc` 复核（含 grep 关键词示例） | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| `vsts(..., mask=PAT_VL1, extent=1)`（默认 NORM_B32）实际生成 `vsts_norm(src, ptr, 0, one)`，只写 4 B，不需要 `ONEPT_B32` 的 32 B 粒度契约 | D | `REVIEW.md#7` 建议级第 5 条；生成代码 `linux-x86_64/kernels/966f…/device_kernel.asc`（`vsts_norm(alpha_v[0], …, one)` 与 `asc_copy_ub2gm_align(..., 6, 4, …, 4, 32)`） | `cd <task_dir> && ASCEND_RT_VISIBLE_DEVICES= python history_version/design_probe_compile.py`，再 grep 缓存 `device_kernel.asc` 的 `vsts_norm` | tilelang 0.1.15 + CANN 9.2.1 / ascend-950pr-9579 / 2026-11-23（仅编译期生成物，无真机数值） | `.agents/platforms/toolchains/ascend/knowledge/PL-950-simd-reduce.md`（补「单 lane 标量落 UB：PAT_VL1 + NORM 即可，行距须 ≥32 B」） |
| 迁移检视的「源-参考分歧」检查项（源 kernel 与 ref_api/golden 在同一舍入点用不同中间值时，设计必须显式表态取哪一个） | P | `REVIEW.md#0`；源 `gate_conv_fwd.py:111`（未舍入）vs `workloads/sequence_modeling.py:121` / 950DT `gate_conv_fwd.py:103-107`（已舍入） | none（本次靠人工读三处源码比对） | 同上 | `.agents/skills/tilelang-design-review/SKILL.md` 维度 0 检查项 + `.agents/skills/tilelang-op-design/references/migration-analysis.md` |

### Transferable Lessons（可迁移教训）

- 检视迁移设计时，**源 kernel 的代码与源 reference（manifest `ref_api` / golden）可能不是同一语义**：凡「舍入点/dtype 转换点」都要三方对照
  （源 kernel、源 reference、已验证的他平台实现），确认设计与 golden 一致而不是与源 kernel 一致——否则实现能编译、却系统性偏离参考。
- 设计的「探针证据」是可复核的：`tilelang.compile` 成功后，生成代码落在 `$TILELANG_CACHE_DIR/<ver>/linux-x86_64/kernels/<hash>/device_kernel.asc`，
  grep `asc_sync`、`update_mask_b32`、`asc_copy_ub2gm_align`、`& 3` 等即可把设计引用的「生成 XX 条指令/XX 处同步」逐条对账，比重新写探针便宜。
- 复杂度表的字节口径要按**方向**分别核：halo/重算系数只乘读侧，写侧恒为 `输出张量数 × M·T·d·elem`；一处系数误乘会让所有候选同时低算同一常量，
  使「排序结论正确、绝对量全错」——这类偏差用「基线与候选同时复算」最容易发现。
- 同一份设计里「表格结论」与「正文汇总」的计数要交叉核对（本次 §1.6.2 表 4 个保留、正文写 5）；这类不一致不影响实现，但会误导下游按错误数量估算工作量。

---

## Stage 3 (develop) Retrospective — first_impl — 2026-11-23

### 本次实现的结论摘要

- 产物：`_engram_gate_conv_fwd_kernel.py`（686 行，kernel + 内嵌 golden + L0/L1/L2/boundary + `--level` 入口），
  编程模式 **SIMD（单次 AIV launch）**，与 DESIGN §2/§0.6 R3 的结构一致。
- 真机结果：**L0 11/11、L1 5/5、L2 4/4、Boundary 5/5 全过**，`--level all` 连跑 3 次退出 0（第 2/3/4 次为最终文件；
  第 1 次是我在编译中途改文件导致的 `TypeError`，见下方 Value Point 4）。
  6 输出最大误差：`Y` 4.9e-4、`vhat` 2.4e-4、`alpha` 1.8e-7、`rrms_h` 1.2e-4、`rrms_k` 1.9e-6、`rrms_v` 4.7e-3（绝对值，`rrms_v` 量级 ~50–1000）。
- UB 占用：manifest 四几何 15.0 / 35.9 / 18.4 / 59.6 KB（最大 24%，因窗口改为 `RC+3` 行，见下条）。
- 与 DESIGN 的三处偏离（均为实现层，数学语义不变）：
  1. **窗口 `4` 槽环形 → `RC+3` 行仿射窗口**（`slot[i]` 写、`slot[i-3+p]` 读，无取模）。等价，UB 从 45.7 KB 升到 59.6 KB（24%）。
  2. **`g = vhat·r_v` 的 vhat 由 `v_ub` 重算**，不从 16-bit `vh_ub` 回读（少一次 UB 读后写依赖）。
  3. **手工 `T.simd.mem_bar("VST_VLD")` 两处/行**（`st` 单 lane 存 → BRC 读；slot 窗写 → conv 读）——DESIGN §7 假设「同 pipe 顺序依赖不需要 flag」被实测否证。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| tilelang-op-develop / Phase 4 分层执行 | L0 门槛默认「跑一次即判定」，但**SIMD 内核里的 UB 读后写是并发可见性问题，单次运行大概率偶然通过**：本任务首轮 L0 的 6 个输出里只有 `Y` 错、且 `Y` 的误差每次运行不同（同二进制、同输入） | `/tmp/all_run*.log` 与首次 `l0_run1.log`（Y 全错）→ 修好后 3 次全过；N 次同输入不同结果见 `repro/TRAP-950-simd-ub-store-load.py` 的 B 变体（无 barrier 时 `nonzero_lanes` 在 0/2/64 之间变化） | 在 Phase 4 加一条硬性要求：**L0 至少连跑两遍并比对 6 个输出的 max_diff**（不一致即判「可见性/竞态」失败），对含「UB 写后读」的内核再额外跑一次 | R |
| 同上的判据形态 | 「6 个输出里只有 1 个错」被技能文档当作普通精度问题；本任务里它其实是**结构信号**：`vhat/alpha/rrms_*` 都是 lane 置换不变量或单 lane 输出，只有 `Y`（唯一消费 UB 写后读结果的输出）暴露问题 | 首轮 L0：`vhat=0.0` 但 `Y=2.0e-1`；改为 `slot←vhat` 后 `tap0/1/2` 全对、`tap3` 全错 | 在 skill 的失败处理表加一行「部分输出精确、部分输出错 ⇒ 先查该输出的数据来源是否为共用/回读缓冲，再怀疑算法」 | P |
| tilelang-op-design / 迁移分析 | DESIGN §9.1 R7 把「`vsts` 默认 `NORM_B32` 的分布语义」列为风险，**真实故障却是「写到读的可见性」**——风险清单命中了位置但写错了机制，导致实现期没有按它设计防御 | `repro/TRAP-950-simd-ub-store-load.py` A/B/B2：写入字节正确（DMA 拷贝逐位相同），无 barrier 的向量回读为 0/部分/正确（运行间变化），加 `mem_bar` 后逐位正确 | 在设计的「风险点」模板里把「UB 缓冲自读」单列一类：凡同一 kernel 内先写后读 UB 缓冲，风险应写「可见性需显式 mem_bar」而不是「dist 语义」 | P |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **UB 向量存 → 向量读需要显式 `T.simd.mem_bar("VST_VLD")`**；无 barrier 时（i）`vsts(PK_B32)` 写的 16-bit 行 `vld(UNPK_B16)` 读出 0/部分/正确（运行间变化），（ii）单 lane `vsts(PAT_VL1, extent=1)` 之后的 `vld(..., dist="BRC_B32")` 读到陈旧标量（把整行 `g` 均匀放大了 1.38×），（iii）fp32 窗口写后读给出逐 lane 错值。加 barrier 后三种情形全部逐位正确 | D | ①`repro/TRAP-950-simd-ub-store-load.py`（A/B/B2/C/D 五路断言，实跑 `REPRO_RESULT: ALL_ASSERTIONS_HOLD`）；②真机证据：同一二进制同输入 `Y` 的 max_diff 在「0.0 / 1.5e-1 / 6.98」之间变化，加两处 `mem_bar` 后 `--level all` 连跑 3 次结果一致；③生成代码里 V→V 依赖没有任何 `asc_sync_*`（只有 MTE2/V/MTE3/S 之间才有） | `examples/engram_gate_conv/_engram_gate_conv_fwd_kernel/repro/TRAP-950-simd-ub-store-load.py` | tilelang 0.1.15 + CANN 9.2.1 / ascend-950pr-9579 / 2026-11-23（真机） | 新增 `knowledge/numerical-boundaries.md` 条目 `TRAP-950-simd-ub-raw`（并在 `PL-950-simd-reduce.md` 的「统计量暂存」段引用：`st` 单 lane 存后必须 `mem_bar` 再 `BRC_B32` 读） |
| 单 lane 标量的广播读取用 `vld(st[k,i,0], dist="BRC_B32")`（配 mem_bar），而不是复用「先前算出的 `vector_f32` 局部」；本任务 `rv_v[0]` 跨 loop nest 复用时结果随数据变化，改为 BRC 后稳定 | D | 生成代码 `auto rvb = vlds_brc_elem<float>(ub + (i*8+736), 0)`；常向量用例 `g = 1.0` 逐位正确；随机向量用例改前 1.38× 均匀偏大、改后逐位正确 | 同上 repro 的 D 变体 + `probe_cases`（三组数据：随机 v / 等幅随机符号 / 周期性 v，改后 Y 全部 0.0） | 同上 | `knowledge/PL-950-simd-reduce.md`（「统计量单 lane 落 UB 后的广播读法」） |
| 用「**恒等数据 vs 随机数据**」与「**按 tap 拆权重**」两个正交探针定位 lane 混淆/可见性缺陷：恒等数据下置换与标量放大都不可见，随机数据下暴露；`cw=e_k`（单 tap）把 conv 拆成「窗口某一行」从而区分「窗口写错」与「conv 累加写错」 | P | `tap0/1/2 → Y==vhat`（halo 为零正确）与 `tap3 → 只剩 slot[3]`；同一代码在 `v=const` 时 `g=1.0` 正确、`v=random` 时 1.38× 偏大 | `repro/`（本任务把最小复现转正；调试脚本留在 `/tmp/probe_cases.py`，结论已折进 repro 的 D 变体） | 同上 | `.agents/skills/tilelang-op-develop/SKILL.md` Phase 4（「先用结构化输入把 6 个输出缩到 1 个可疑缓冲」） |
| 编译期改写会**保留**仿射化的 `floormod`（生成 `(i*64)+(p*64)+64` 与 `base+(i-3+p)*64` 等价），所以「环形窗口」本身可用；但每轮改动后必须用生成代码核对索引，而不是靠肉眼推断取模是否被丢弃 | D | 首轮我误判「编译器丢了 `% 4`」(见本节「教训 1」)，实为等价改写；生成代码 `slot` 写 `(i*64)+256`、读 `(i*64)+(p*64)+64`（base 256、4 行窗口）逐项核对后成立 | none（`$TILELANG_CACHE_DIR` 的 `device_kernel.asc` 可复核；脚本为 `/tmp` 一次性） | 同上 | `knowledge/PL-950-vector-execution.md`（补「索引化简后必须复核 `device_kernel.asc`」） |

### Transferable Lessons（可迁移教训）

- **先判「多次运行是否一致」，再判「精度」**：SIMD 内核里 UB 写后读的可见性问题是**非确定性的**，一次 L0 通过不构成证据。任何「同一编译产物、同一输入、两次结果不同」都直接判可见性缺陷，不要再按数值误差调。
- **「只有一个输出错」优先怀疑该输出的数据来源**：本任务 `vhat/alpha/rrms_*` 都是 lane 置换不变量或单 lane 输出（对错位/陈旧值不敏感），`Y` 是唯一消费 UB 回读结果的输出，所以它是唯一报警的那个。给每个输出标注「对 lane 置换是否敏感」能一步定位。
- **生成代码是唯一裁判**：我一度从 `(i*64)+(p*64)+64` 反推「取模被丢弃」，实际它与 `base+(i-3+p)*64`（base=256、窗口 4 行）完全等价——误判让我白追一轮。凡怀疑 codegen，先把地址表达式代数化简成「基址 + 行×步长」再下结论。
- **`BRC_B32` 是「标量 → 全 lane」唯一被验证可靠的通道**（配 `mem_bar`）；`vector_f32` 局部跨 loop nest 复用即使在同一 `SimdVF` 帧内也不要依赖。
- **16-bit 行的写与读用同一 barrier 规则**，因此 `vhat` 的 16-bit 行 + `UNPK_B16` 回读在加了 barrier 后可用（`repro` 的 B2 证明）；本实现仍选择「从 `v_ub` 重算」，因为 `v_ub` 是 DMA 填的、不涉及 UB 自读，风险更低——两条路都成立，优先选依赖更少的。
- **halo 重算确定性成立**：`RC+3` 行窗口下，halo 行与 owned 行共用同一段 python 代码（只有 `if r >= 0` 包裹不同），L0-2/L0-5 的 `RC=4/6` 用例（窗口跨 7/9 行滚动）逐位吻合，DESIGN §0.6 R3 的「同一条构建期展开的指令序列」论证在实现层落实了。
- **`st` 的 32 B 行距契约成立**：`(4, RC+3, 8)` fp32 + `vsts(PAT_VL1, extent=1)` + `T.copy(st[k, 3:3+RC, 0:1], Out[b, t0:t0+RC])` 在 `RC=1..6` 全部正确（含 `RC=1` 的 4 B 压缩写），DESIGN §9.1 R11 的风险可以下调。
