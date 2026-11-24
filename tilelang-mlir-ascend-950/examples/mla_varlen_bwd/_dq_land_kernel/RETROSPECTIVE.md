## Stage 1 (design) Retrospective — first_design — 2026-11-23

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Design-flow | `tilelang-op-design` 的 Phase M0/M1 假定「一个算子目录 = 一个自有源函数」，没有覆盖「**多个 op 共用一个模块级工厂**」的情形：`_dq_land_kernel` 在 `bwd.py` 里只有一个定义，`MLADenseBwdKernel` 与 `MLAVarlenBwdKernel` 通过 `_MLABwdKernelBase._compile` 共用它。skill 没有给「先证明逐字相同，再决定复用什么」的步骤，本次只能自行设计「源码单一定义 / 抽取文件函数体 `diff` / 工厂签名无分支参数 / 设备码符号」四段取证（DESIGN.md §0.0） | `DESIGN.md#0.0`（四段证据）；`bwd.py:79-106`（唯一定义）、`bwd.py:354`（共用调用点）、`bwd.py:385-387`（packed forward 差异只在 view） | 在 `references/migration-analysis.md` 增加「共用工厂分支」：先做**逐字相同判定**（含 `diff` 抽取文件函数体 + 签名参数表），判定为相同时把源算子解读（§0.1–§0.4）标注为「承接兄弟任务、仅复核」，并要求给出 packed/dense 的差异清单 | R |
| Design-flow | 复用型迁移没有「分区」章节模板，导致「完全复用项」与「需复核项」只能散落在 §0.5/§11。本次按任务 prompt 手写「复用分区总览 + §11 九项核对」两张表才把结论收敛 | `DESIGN.md` 的「复用分区总览」与 §11 | 在 `templates/design-template.md` 增加可选章节「复用分区（apply: 源函数被多 op 共用 / 同 op 多 launch 共用）」：一表列复用项+证据，一表列复核项+结论（允许全为「无差异」） | R |
| Info-source | 平台知识仍未收录 dense Stage 3 的两条真机事实（`vsts` 稠密落盘必须 `PK_B32`+32 位掩码；`vcvt(sat=False)` 即 IEEE）。`references/programming.md` 仍只写「SIMD load/store 分布参数需要按具体路径核对」，`knowledge/numerical-boundaries.md` 仍只声明「不把 SIMT 结论用于 `T.simd.vcvt` 的任意模式」。因此本设计只能引用**兄弟任务的 repro 脚本**作为真机证据，而不能引用平台条目 | `.agents/platforms/toolchains/ascend/references/programming.md`（SIMD 行）；`.agents/platforms/toolchains/ascend/knowledge/numerical-boundaries.md`（第 19 行）；对照 `examples/mla_dense_bwd/_dq_land_kernel/repro/CONST-950-vcvt-narrow-store-pack-b32.py` 与 `.../CONST-950-vcvt-rs-disable-ieee.py` | 把 dense Stage 3 的两条 D 类实证合入 `programming.md`（配方行）与 `numerical-boundaries.md`（`vcvt` 语义行）；合入前，任何 `vcvt` 窄化 + `vsts` 的设计都必须引用 repro 脚本并注明「平台条目未收录」 | D |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 复用型迁移的「逐字相同」四段取证：源码单一定义 / 抽取函数体 `diff` 为空 / 工厂签名无 packed 分支参数 / 设备码把 `T_q` 当形参 | P | `DESIGN.md#0.0` 五条证据；`DESIGN.md#11`（P1–P9 九项核对） | `diff <(sed -n '/^def _dq_land_kernel/,/^    return _dq_land_func()/p' examples/TileOPs/tileops/kernels/attention/mla_dense_bwd/_mla_dense_bwd_kernels.py) <(sed -n '/^def _dq_land_kernel/,/^    return _dq_land_func()/p' examples/TileOPs/tileops/kernels/attention/mla_varlen_bwd/_mla_varlen_bwd_kernels.py)`（无输出） | tilelang 0.1.15 + CANN 9.2.1 / Ascend950PR_9579（静态 + 编译级） | `tilelang-op-design/references/migration-analysis.md` 或 design template |
| packed 形态的无卡编译探针：fp16/bf16 各 41 行设备码、`vsts_pack_b32` 在场、`total_q` 为形参；两个源码 sha256 可作为「同一特化 → 同一设备码」的比对锚点 | D | `history_version/design_probe_packed.py`；`history_version/probe_packed_gen_float16.txt`（sha256 前缀 `d0fde4920bf2c7ac`）、`history_version/probe_packed_gen_bfloat16.txt`（`914c64f7abb2f600`） | `cd examples/mla_varlen_bwd/_dq_land_kernel && source ../../../.agents/platforms/toolchains/ascend/env-950pr.sh && ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python history_version/design_probe_packed.py`（打印 `[PASS] packed path keeps the SIMD body`） | tilelang 0.1.15（venv-ascend950）+ CANN 9.2.1 + Ascend950PR_9579；编译级、无 launch | pattern-library / `.agents/platforms/toolchains/ascend/references/programming.md` |
| 「同一二进制服务所有 packed 切分」应作为 packed 逐元素算子的标准 L0 判据（P7）：固定 `(H, D, scale, dtype)`，同一 `dq_acc` 用不同 `q_lens` 切分三次调用，输出必须逐位相同 | P | `DESIGN.md#8.3` P7 行；`DESIGN.md#4.6`（`T_q` 不参与特化） | 由 Stage 3 的 `_dq_land_kernel.py --level L0` 实施（本阶段无卡，未执行） | tilelang 0.1.15 + CANN 9.2.1 / Ascend950PR_9579 | `tilelang-op-develop`（L0 判据） |
| 知识库 `cost_measurements` 为空时，用**同型号兄弟算子的实测带宽带**（dense Stage 3 L2：0.75–1.05 TB/s）作为 D-2 常数锚点，并显式标注「兄弟算子实测、未进知识库」 | P | `DESIGN.md#1.6.0` 的「设计期估算」行与两级常数来源 | `examples/mla_dense_bwd/_dq_land_kernel/_dq_land_kernel.py --level L2`（dense 侧已跑：604 MB → 0.580/0.576 ms，2415.9 MB → 3.231 ms） | 同型号同工具链（0.1.15 / CANN 9.2.1 / 950PR） | `.agents/skills/_shared/standards/hardware-cost-model.md` |

### Transferable Lessons（可迁移教训）

- 迁移前先查「源函数是否被多个 op 共用」：若 `grep` 只命中**一个** `def` 而两个 Kernel 类都调它，则「packed 差异」几乎一定落在
  调用方（是否 `view`、传什么长度），而不在 kernel 内——此时正确做法是先证明逐字相同（源码单一定义 + 抽取文件函数体 `diff` + 工厂签名），
  再整段承接兄弟任务的设计，把分析预算花在**验证覆盖**（新几何的 L0）而不是重新推导。
- 判断「packed 是否引入差异」的高性价比三问：① kernel 参数表里有没有 `cu_seqlens_*` / per-request 参数；② 设备码里 `T_q`
  是常量还是 kernel 形参；③ 输出 shape 与 kernel 内部 shape 是否相同（是否多一次 `view`）。三问都为「无/形参/相同」时，
  可以判定该 launch 对打包边界**完全盲**，差异只剩验证覆盖。
- 复用兄弟任务的**实现**而不是兄弟任务的**设计文档**：设计文档可能保留已被真机证伪的形态（本任务：`vsts` 的 `NORM_B16`）。
  复用时必须核对「设计文本 vs 已通过真机的实现 vs 真机 repro 脚本」三者，以 repro 脚本为准。
- 交叉引用必须在**本任务目录**内留下可执行证据：即使结论是「与兄弟任务完全相同」，也应在本目录跑一次探针 + 等价性脚本
  （本次各 1 次，编译级/CPU 级，成本 < 1 分钟），否则下游检视无法区分「复核过」与「假设过」。

## Stage 2 (review) Retrospective — first_review — 2026-11-23

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Review-flow | Phase 1 第 7 步的机械复核命令写成 `python examples/{project}/{op}/verify_equiv.py`，未指定解释器与工作目录：系统 `python3` 无 torch，而在 **repo 根**运行会让 `import tilelang` 命中 dev root 并因缺私有 TVM 直接 ImportError | 本次在 `examples/mla_varlen_bwd/_dq_land_kernel` 下用 `/workspace/venv-ascend950/bin/python` + `source env-950pr.sh` 跑通 `[EQUIV_PASS]`；在 repo 根 `import tilelang` 报 `TileLang's private TVM Python package is missing` | 在 `tilelang-design-review` Phase 1 第 7 步注明：解释器取任务平台 venv，工作目录必须是任务目录（cwd 在 repo 根会遮蔽安装包），并给出 `statectl platform --task-dir` 的 `execution` 作为唯一口径 | R |
| Review-flow | `design_calc_check.py` 对本类「纯 Vector + 表格化预算」的设计**三项核心检查全 skip**（`ub_budget`/`core_split` 解析不出 §4.3 名称/shape/dtype/版本表与 §5.5 分核表，`l0c_budget` 因无 `T.gemm`），机制上等于机械拦截全部落空 | 本次 JSON：`ub_budget` skip("no per-level buffer tables parsed")、`core_split` skip("insufficient parsed dims {}; declared logical cores = 64")、`l0c_budget` skip、仅 `r3_metrics` pass | 让 `ub_budget` 解析 §4.3（名称/shape/dtype/scope/字节/版本）与 §4.5 合计行后复算总和；让 `core_split` 从 §5.5 的 `ntiles`/物理核数表复算 `cpt`、末核余量与失衡百分比。解析不到时必须输出 `warn` 而不是静默 `skip` | R |
| Review-flow | 维度 0 只有「亲自 Read 源算子」这一条方法，**没有覆盖「复用性主张」**（「本函数与兄弟算子逐字相同 / packed 无差异」）的检视手段，该类主张恰恰是复用型迁移最需要独立核对的一项 | 本报告维度 0 自行补做三段取证：`diff` 两个抽取副本的 `_dq_land_kernel` 函数体（输出为空）、核对工厂签名无 packed 分支参数、看设备码里 `total_q` 是形参 | 在 `tilelang-design-review` 维度 0 检查表加一行「复用性主张：须亲自 `diff` 抽取副本函数体 + 核对工厂签名/参数表 + 核对设备码里动态维是否形参；三项任一不成立即按两算子分别解读源算子」 | R |
| Info-source | conductor 调度建议的「diff 两个 Stage 3 实现」在 Stage 2 时序上不可执行——packed 侧 Stage 3 实现尚未产出 | 本目录无 `_dq_land_kernel.py`（DESIGN §10.1 标注为 Stage 3 产物），`ls` 仅有 `DESIGN.md`/`verify_equiv.py`/`history_version/`/`RETROSPECTIVE.md` | 在 review skill 或 conductor 调度口径写明：Stage 2 的等价替代证据 = 源函数 + 抽取副本 `diff` + 本目录探针的同特化设备码；两份实现互 `diff` 留作 Stage 3 自检项（写进 §10.4 第 1 步即可） | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 「同特化 → 同设备码」可作复用型迁移的审计锚点：同一工厂重复编译得到 sha256 完全相同的设备码（41 行），使设计文档引用的设备码快照可被第三方复现核对 | D | `REVIEW.md#机械复核`（重跑得 `d0fde4920bf2c7ac` / `914c64f7abb2f600`、`lines=41`、`total_q_symbolic=True`，与 `DESIGN.md#0.3`/`#3.5.4` 记录逐字相同）；探针 `history_version/design_probe_packed.py` | `cp history_version/design_probe_packed.py /tmp/… && cd /tmp/… && source …/env-950pr.sh && ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python design_probe_packed.py`（打印 `[PASS]`，两次 hash 一致） | tilelang 0.1.15（venv-ascend950）+ CANN 9.2.1 + Ascend950PR_9579；编译级、无 launch、codegen 确定性成立 | `.agents/platforms/toolchains/ascend/references/programming.md`（探针取证行）/ `tilelang-design-review` |
| `T.Pipelined(..., num_stages=2)` + `T.annotate_buffer_versions({buf:2})` 在 ascend 生成码里**不产生 load/compute 跨轮重叠**：每轮 load 之后立即 `wait(MTE2→V, it&1)`，可见重叠只有 `MTE2(k+1) ∥ MTE3(k)`，也没有 prologue 预取 | P | `REVIEW.md#5`（S1）；`history_version/probe_packed_gen_float16.txt` 第 24–34 行（`wait(V→MTE2)` → `copy gm2ub` → `notify(MTE2→V)` → `wait(MTE3→V)` → **`wait(MTE2→V)`** → VF → … → `copy ub2gm`） | `cd examples/mla_varlen_bwd/_dq_land_kernel && … history_version/design_probe_packed.py`，读生成的 `probe_packed_gen_float16.txt` 第 24–34 行的 flag 顺序 | tilelang 0.1.15 + CANN 9.2.1；静态生成码证据（**性能性结论，未经实测时延**） | `.agents/platforms/toolchains/ascend/references/programming.md`（流水语义行）/ `tilelang-op-design`（流水设计）|

### Transferable Lessons（可迁移教训）

- 检视「复用型迁移」时，最便宜也最硬的三项证据是：`diff` 两个抽取副本的函数体、核对工厂签名/参数表有没有兄弟分支参数、看设备码里动态维是不是 kernel 形参。
  三者都指向「相同」时，才可以把源算子解读标为「承接」；只读设计自述无法区分「复核过」与「抄过来」。
- 机械检查器的 `skip` **不等于** `pass`：纯 Vector、预算写成表格的设计常让 `ub_budget`/`core_split` 解析失败，必须逐项人工复算
  （UB 总和、`ntiles`/`cpt`/末核余量/失衡百分比），并把「机械跳过，人工已核」的具体数字写进 REVIEW.md，否则该维度等于没检。
- 设计文档里的**性能性**表述（流水是否重叠、收益几条指令、带宽是否触顶）必须拿生成码或 pattern-library 实测条目核对；
  **数值等价性**主张则拿 `verify_equiv`/真机 repro 核对——两类证据不可互相替代，混淆会同时放过性能虚报与等价性错误。
- 「负向断言」要逐字核对它点名的那份文件：本次设计写「平台候选库无 elementwise 专门条目」，而该文件首段就有
  `elementwise / broadcast` 行。实质候选没有漏，但结论的可审计性依赖引用准确，检视时应把「文件名 + 行内容」对上。

## Stage 3 (develop) Retrospective — first_impl attempt 1 — 2026-11-23

结果：`[PRECISION_PASS]`，L0 20/20、L1 6/6、L2 4/4、boundary 12/12，全部 `max_diff=0` / `bit_diff=0`（逐位等于 golden）。
实现文件 `_dq_land_kernel.py`（727 行）在第一次运行即通过，无精度修复轮次、无编译失败轮次。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Develop-flow | 设计 §8.3 的 L0 计划按「枚举数量」分层（满载几何只 1 例 `q_lens=[4096]×8`，非有限值与空 batch 推到 boundary），而任务口径要求 L0 覆盖 manifest 全部 3 条 workload 的 `T_q` + `T_q=0` + 非有限值。**实测证明设计的分层是过保守的**：把 manifest 三例（32768/32256/32768 行）+ `T_q=0` + 两个非有限值用例全部放进 L0 后，L0 总耗时 **33.0 s**（含 8 次编译），比 L1(24.9 s)+L2(48.0 s) 还短 | 本目录 `_dq_land_kernel.py --level L0`：`[L0] 20/20 cases passed`、`[TIME] 33.0 s`；`--level L2`：`[TIME] 48.0 s`（4 例，repeat3 + 分块比对）；`L0-manifest-ds-v3-8x4k launch=3.297 ms` | 在 `tilelang-op-develop` 的 L0/L1 判据里把分层依据写成「**按单例成本**而非按枚举数量」：单例（含 golden 比对）< 1 s 的用例一律进 L0，即使它是 manifest 满载几何；只有 repeat≥20 的稳定性/时延基线才留 L1/L2 | P |
| Develop-flow | 设计 §10.4 第 1 步要求「两份实现（dense vs packed）再 diff 一次」，但没有给 **diff 粒度**口径。本次实测三种粒度：整文件 diff 被 harness 差异（~120 行）淹没、**kernel 体 diff 只剩 1 处报错字符串 + 注释**、生成码 diff 必须先归一化 `prim_func` 名。粒度选错会得出相反结论（整文件 →「两个实现不同」，kernel 体 →「逐字相同」） | 本目录实测：`diff <(sed -n '/^def _vector_cores/,/^@functools.lru_cache(maxsize=32)/p' dense/…) <(… varlen/…)` 仅 2 处（`MLADenseBwdOp`→`MLAVarlenBwdOp` 报错串、注释），与 `REVIEW.md` 维度 0 的「逐字相同」主张一致；`repro/CONST-950-packed-impl-vs-design-devicecode.py` 归一化后 `normalised_identical=True` | 在 `tilelang-op-develop`（或 migration-analysis）写明复用型迁移的验收 diff 口径：**以 kernel 体 diff + 归一化生成码 diff 为准，整文件 diff 不作判据**；并给出 `sed -n '/^def <factory>/,/^@functools.lru_cache/p'` 这类按工厂体截取的一行命令 | R |
| Info-source | 平台 `bottlenecks.md` / `cost_measurements` 仍为空（设计期只能借兄弟算子 0.75–1.05 TB/s 的带），因此设计的 roofline 对最大 workload 估 4.60–6.44 ms，**实测 3.297/3.303 ms**（三 workload 一致 1.448–1.471 TB/s），常数偏低 1.4–1.9× | 本目录 `--level L2`/`L1`/`L0-manifest`：ds-v3-8x4k 3.297 ms（4831.8 MB → 1.466 TB/s）、ds-v3-mixed 3.233 ms（4756.3 MB → 1.471 TB/s）、kimi-k2-4x8k 1.668 ms（2415.9 MB → 1.448 TB/s） | 本算子 Stage 4 采集 msprof 后按 `hardware-cost-model.md` §3 回填「设计估算 vs 实测」偏差行；在那之前，平台知识里应把「0.75–1.05 TB/s」标注为**兄弟算子低负载口径**，不得当作本型号上限 | D |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 复用型迁移的验收 diff 口径：kernel 体 diff「逐字相同」（唯一差异 1 个报错字符串 + 注释），整文件 diff 因 harness 差异不可用 | P | 本目录 kernel 体与 `examples/mla_dense_bwd/_dq_land_kernel/_dq_land_kernel.py` 的 `_vector_cores.._dq_land_kernel` 段 diff 只有 2 处非代码差异；`REVIEW.md` 维度 0 的「逐字相同」主张由此从设计级升级为实现级 | `diff <(sed -n '/^def _vector_cores/,/^@functools.lru_cache(maxsize=32)/p' examples/mla_dense_bwd/_dq_land_kernel/_dq_land_kernel.py) <(sed -n '/^def _vector_cores/,/^@functools.lru_cache(maxsize=32)/p' examples/mla_varlen_bwd/_dq_land_kernel/_dq_land_kernel.py)`（仅 2 处 hunk） | tilelang 0.1.15 + CANN 9.2.1 + Ascend950PR_9579 | `tilelang-op-develop` / `migration-analysis.md` |
| 实现的生成设备码与设计冻结快照**逐行一致**（归一化 prim_func 名后，双 dtype 各 41 行、`compute=['vlds_norm','vmuls','vcvt']`、`store=['vsts_pack_b32']`）；归一化是必须的，否则 `_land_kernel` 与 `_dq_land_main_kernel(_simd_vf_0)` 让 sha256 必然不同 | D | 实现 sha256 `5e9df8aad8c39124`(fp16) / `acb93f398bf4d606`(bf16) vs 设计快照前缀 `d0fde4920bf2c7ac` / `914c64f7abb2f600`；`normalised_identical=True` 两 dtype | `cd examples/mla_varlen_bwd/_dq_land_kernel && source ../../../.agents/platforms/toolchains/ascend/env-950pr.sh && ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python repro/CONST-950-packed-impl-vs-design-devicecode.py`（打印 `[PASS] implementation matches the frozen design device code`，退出码 0） | tilelang 0.1.15（venv-ascend950）+ CANN 9.2.1 + Ascend950PR_9579；编译级、无 launch、codegen 确定性成立 | `tilelang-op-develop`（L0 判据）/ `references/programming.md` |
| packed 边界盲的实证：同一编译产物、同一 `dq_acc`、5 种切分（含 `[0,640]` / `[0,0,0,640]` 的 0 长度请求）输出**逐位相同**且与 golden 逐位一致 | D | `--level L0` 的 `L0-11 P7-one-program-multi-splitting`：`T_q=640 H=8 D_qk=192 float16: 5 splicings of one program are bit-identical`；`bit_diff=0` | `cd examples/mla_varlen_bwd/_dq_land_kernel && source ../../../.agents/platforms/toolchains/ascend/env-950pr.sh && /workspace/venv-ascend950/bin/python _dq_land_kernel.py --level L0`（L0-11 PASS） | 同上，真机 | `tilelang-op-develop`（packed 算子的标准 L0 判据） |
| manifest 满载几何进 L0 的实测成本：3 条 workload（`T_q`=32768/32256/32768，H=128/128/64）各含 golden 分块比对，L0 合计 33.0 s（含 8 次编译），故「满载只放 L2」是过保守分层 | D | `--level L0`：`[L0] 20/20 cases passed`、`[TIME] 33.0 s`；三例 launch 3.297/3.233/1.668 ms、`bit_diff=0` | 同上（`--level L0`） | 同上 | `tilelang-op-develop`（L0/L1 分层判据） |
| 本型号落盘类逐元素算子的实测有效带宽：1.448 / 1.466 / 1.471 TB/s（6E 口径，6E = 读 fp32 + 写 16-bit），显著高于设计期借用的兄弟算子 0.75–1.05 TB/s 带 | D | `--level L2`/`--level L1`：ds-v3-8x4k 4831.8 MB / 3.297–3.303 ms；ds-v3-mixed 4756.3 MB / 3.233–3.245 ms；kimi 2415.9 MB / 1.656–1.668 ms | `… /workspace/venv-ascend950/bin/python _dq_land_kernel.py --level L2`（打印 launch ms，含 warm-up 后的单次 launch） | 同上（`elapsed` 为 host 计时，未做 msprof 采集） | `.agents/skills/_shared/standards/hardware-cost-model.md` / `bottlenecks.md`（Stage 4 回填） |

### Transferable Lessons（可迁移教训）

- 复用兄弟任务的实现时，**diff 的正确粒度是 kernel 体**（`sed -n '/^def <factory>/,/^@functools.lru_cache/p'` 截取），不是整文件：整文件 diff 会被测试 harness 差异淹没（本任务 ~120 行），容易让人误判「实现不同」而重写已验证代码。
- 复用同一 SIMD 段链的 packed 变体时，**新增风险只在验证覆盖**：唯一需要新写的用例是「同一二进制服务多个 packed 切分」（P7），其余照抄 dense 用例只把 `(B,S)` 几何换成 `q_lens` 列表即可。
- 生成设备码可以直接当回归护栏，但必须先**归一化 prim_func 名**（`_x_kernel`、`_x_kernel_simd_vf_0`）：`__simd_vf__` 函数名带下划线后缀，`\b_land_kernel\b` 这类词边界替换**匹配不到** `_land_kernel_simd_vf_0`，归一化不全会把「完全一致」误报成「漂移」。
- 「满载几何太贵、只放 L2」是常见误判：本算子 manifest 最大 workload（`T_q`=32768, H=128, 805 M 元素）单例含 golden 分块比对也只 ~1 s，三条全进 L0 后 L0 仍是全量里最快的一层（33 s < L2 的 48 s）。分层应按**单例成本**，不按枚举数量或元素量直觉。
- packed 落盘算子的 `T_q=0` 必须在 L0 里跑：`ntiles = 0` 让 56 个核各跑一个空 `T.Pipelined`，只有真机能证明 `out_idx` 对空首维仍能分配输出、且不发生越界写。
- 设计期的跨型号带宽常数只能用于量级判断：本任务同一实现三个 workload 一致落在 1.45–1.47 TB/s，比设计借用的兄弟算子带高 1.4–1.9×；差异不在实现（实现与设计快照逐行一致），而在常数口径。
