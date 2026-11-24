# `_engram_project_kernel` Stage 复盘

## Stage 1 (design) Retrospective — first_design — 2026-11-23

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Design-flow | skill 的「资源分层加载」要求「设计期探针 ≤2 个、每探针 ≤10min」，但**编译探针的形态枚举**（同一结构 × 7 种 shape/dtype）在没有先例时是唯一的证据来源，一个探针脚本会自然膨胀到 10+ 变体；当前预算按「脚本数」计而不是按「编译次数/墙钟」计，容易让 agent 在脚本粒度上做取舍而不是在问题粒度上 | `examples/engram_decode/_engram_project_kernel/history_version/design_probe_shift_placement.py`（13 变体）+ 两个附加脚本 `design_probe_ktail_clear.py`、`design_probe_shift_3d.py` | 在 `tilelang-op-design` skill 的探针预算条目里把「≤2 个探针」明确为「≤2 个探针**主题**（可含多个 shape 变体），单主题墙钟 ≤10min」，并给出「失败形态也计入证据」的说明 | R |
| Info-source | 平台 ruleset 已说明 `examples/gemm/*.py` 属**另一工具链**（`tilelang.language` / npuir），但「同类实现」检索时它们仍是 `grep "T.gemm" examples/` 的最高频命中，需要额外一步才能筛出真正可用的 `examples/**/_*_kernel/*.py`（ascend 方言） | DESIGN.md §3.5.3；对照 `examples/gemm/matmul.py`（`tilelang.language`）与 `examples/shared_expert_mlp/_gemm_basic_kernel/DESIGN.md` | 在 `references/info-sources.md` 的「强制步骤 0」加一条可执行过滤：`grep -rl "tilelang.ascend.language" examples/` 为唯一合规同类实现来源 | R |
| Constraint-doc | `programming_modes.migration_default` 被写进 `op_requirements.programming_mode` 后，与「设计必须按证据选模式」存在张力：本算子主计算是矩阵乘，默认值 `simd` 与证据结论（混合）冲突，agent 需要自行判断该偏离并回报 | DESIGN.md §2.2；`statectl platform` 的 `programming_modes.migration_default = simd` | 在 `tilelang-op-design` 的选择章节注明「`migration_default` 只是字段默认值，不构成设计约束；偏离时须在 §2.2 写明并回报 conductor」 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 普通 `T.Kernel` 内同时放 L1/L0 GEMM 与 UB 代码会被编译器自动提升为 `__mix__(1,2)`（无需显式 `T.MixedKernel`）；后者在本结构下编译失败 `Need input to be in SSA form dup sid` | D | `examples/engram_decode/_engram_project_kernel/DESIGN.md` §11.1（探针 H1/H2 vs X1/X2） | `cd examples/engram_decode/_engram_project_kernel && source .agents/platforms/toolchains/ascend/env-950pr.sh && ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python history_version/design_probe_shift_placement.py` | TileLang ascend 0.1.15 + CANN 9.2.0 SDK / 9.2.1 本站 runtime；dav-3510（950PR，未上卡，compile-only） | pattern-library（新条目 PL-950-mixed-auto-promote）+ `.agents/platforms/toolchains/ascend/references/programming.md` |
| 自动提升后的 AIV 段**两个 sub-block 都执行**同一段代码（生成源码无 `sid` 守卫）→ 同字节值的重复 GM 写；用 `with T.Vector(vector=2) as sid` 显式拆分虽能编译，但会丢掉 UB 暂存的 `PIPE_MTE2→PIPE_MTE3` 同步对 | D | DESIGN.md §0.6 R4、§11.1（探针 H1 vs H7） | 同上（`A_b8_fp16` 与 `A_sidguard_b8` 两个变体各取 `get_kernel_source()` 对比 AIV 段） | 同上 | pattern-library + `knowledge/capability-gaps.md`（若确认为编译缺陷） |
| 平移（cache 左移）在本工具链的三条路线只有一条可用：3-D 区域拷贝 `>2D strided copy not yet supported`；L1 中转 `ScalarDcacheBypass: vector-valued buffer store not implemented`；逐 batch 行 GM→UB→GM ✅ | D | DESIGN.md §9.2 E3、§11.1（H1/H5/H6） | 同上 + `ASCEND_RT_VISIBLE_DEVICES="" ... history_version/design_probe_shift_3d.py` | 同上 | pattern-library（瓶颈/形态条目：shift-copy 的可用路径） |
| `T.clear` 对 L1 区域要求 C0（16 元素）对齐 → 「K 尾块就地补零」不可表达，必须改用「有效 K 区间」或 host 侧 pad | D | DESIGN.md §9.2 E1、§11.1（H3/H4） | `ASCEND_RT_VISIBLE_DEVICES="" ... history_version/design_probe_ktail_clear.py` | 同上 | `knowledge/gemm-tiles.md`（尾块条目补一行） |

### Transferable Lessons（可迁移教训）

- ascend 方言下 `T.Kernel` 只有 1 维网格、没有 `threads`；**不要**去找「三维 grid 的等价写法」，直接 1-D + 整数分解，并把每个轴的有效区间（`am/an/ak`）作为区域上界传下去。
- 同一 kernel 里既要有 Cube GEMM 又要有纯搬移时，先写普通 `T.Kernel`（L1/L0 + UB 混排）让它自动提升成 `__mix__(1,2)`；`T.MixedKernel` 在本工具链对复杂 body 会报 SSA `dup sid`，不要作为首选。
- 任何经 UB 中转的拷贝，**目的偏移恒为 0**（UB→UB 错位拷贝有已验证反例）；连续多行搬移用「逐行 2-D 拷贝 + `T.serial`」，3-D 区域拷贝在当前版本不被支持。
- 想在 L1 上「补零尾块」之前先试一次 `T.clear`：它要求 C0 对齐区域，动态尾长会直接编译失败；尾块优先用「有效区间进区域 + 让 `asc_mmad` 的 M/K/N 参数表达长度」。
- `T.annotate_buffer_versions` 只是「请求」，AutoSchedule 可以拒绝并打印警告——设计里写「2 版本」不等于运行时有 2 版本，容量预算要按两种情形都算一遍（本任务 L0B 恰好在 2 版本时打满 64 KiB）。

## Stage 2 (review) Retrospective — pass — 2026-11-23

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Review-flow | 维度 5（循环与同步）只有「同步策略与模式一致」的定性条款，**没有「design 对编译器行为的事实陈述须逐条对照生成代码」的检查项**。本次靠额外编译两次才发现 `§6.3/§9.1/§11.2#6` 的多缓冲论断张冠李戴（普通 `T.Kernel` 实际拿到 2 版本；警告只在 `T.MixedKernel` 结构出现），而这类陈述直接影响 Stage 4 的容量假设 | `examples/engram_decode/_engram_project_kernel/REVIEW.md` 维度 5 S2；对照 DESIGN.md §6.3/§9.1 与探针生成源码（`e_l1_version_counter_2 & 1`） | 在 `tilelang-design-review` 维度 5 增加一行检查项：「DESIGN 中对工具链行为（多缓冲是否生效 / 版本数 / 自动提升 / 报错文本）的事实陈述，须与 §11.1 探针的生成代码或本次复核复跑一致；不一致 → warn（若影响容量/正确性假设则 fail）」 | R |
| Review-flow | 维度 8 要求「弃选前提亲自核对」，但未提示**去语言模块里搜同义 API**（本次在 `copy_op.py` 发现文档化的 `T.copy(..., sub_blockid=…)`，是设计只评估 `T.Vector as sid` 时漏掉的官方 sub-block 路由入口）。弃选论证的完整性检查目前只核对「所引前提是否与文档冲突」，不核对「是否存在更直接的受支持入口」 | REVIEW.md 维度 8 S7；`/workspace/venv-ascend950/.../tilelang/ascend/language/copy_op.py:238,276` | 在维度 8「弃选论证证据规则」补一句：「弃选后须在对应语言模块 grep 同义关键字（`sub_block/blockid/broadcast/…`），确认没有更直接的受支持入口；发现即列为建议级（影响正确性时升为阻塞）」 | R |
| Review-flow | 机械复核只规定了 `verify_equiv.py` 与 `design_calc_check.py`，**未包含「设计期探针复跑」**。本次弃选前提（3-D 拷贝 / 显式 MixedKernel / L1 中转）只能靠手动挑脚本复跑，且 `design_calc_check` 的 `core_split` 在本设计上 skip（dims 解析不足），需人工兜底——流程里没有「skip 项必须人工复核并留证据」的显式落点（skill 有写，但机械段模板未给字段） | REVIEW.md「机械复核」段 core_split skip 行；`design_calc_check.py` 输出 `insufficient parsed dims {'bm':16,'bn':64}` | 在 REVIEW.md 模板的「机械复核」段增加两个必填子项：「探针复跑清单（脚本 + 复现结果）」与「skip 项人工复核结论」，便于 conductor/evolver 判断复核强度 | R |
| Info-source | 平台 `execution.prototype_command` 指定的解释器 `/usr/local/python3.12.16/bin/python3` **缺 `tqdm`**，`import tilelang` 直接失败，无法执行 design-time 探针；实际必须用 `/workspace/venv-ascend950/bin/python`。AGENTS.md 要求「按 execution 的命令执行」，两者冲突时没有降级口径 | `cd 任务目录 && /usr/local/python3.12.16/bin/python3 -c "import tilelang"` → `ModuleNotFoundError: No module named 'tqdm'`（同时该解释器解析到 repo dev root 的 `tilelang/`，非 venv 的 ascend 安装） | 在 `AGENTS.md` 的 Execution context 段加降级口径：「prototype/precision 命令的解释器不可用时，记录实际解释器与失败原因，改用工具链 venv 解释器，工作目录仍取 execution.working_directory」；并在平台 `execution` 里给 prototype 命令一个可用的 interpreter | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 普通 `T.Kernel`（L1/L0 Cube + UB 搬移）**确实拿到 2 个缓冲版本**（生成代码 `version_counter & 1` 索引 + 版本化 event id），AutoSchedule 的 `did not enable multi-buffering` 警告只在 `T.MixedKernel` 结构出现——不能把该警告当作「普通 kernel 多缓冲未生效」的证据 | D | `examples/engram_decode/_engram_project_kernel/REVIEW.md` 机械复核「补充一手复核」；对照 DESIGN.md §6.3/§9.1 | `cd examples/engram_decode/_engram_project_kernel && source .agents/platforms/toolchains/ascend/env-950pr.sh && ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python /tmp/opencode/rev_probe_plain.py`（§3.3 冻结结构的等价复刻；durable 等价形态见 history_version/design_probe_shift_placement.py 的 build_plain()）；对照 `history_version/design_probe_shift_placement.py` 的 `build_mixed()` 输出四条警告 | TileLang ascend 0.1.15 + CANN 9.2.0 SDK / 9.2.1 runtime；dav-3510 950PR；compile-only | `.agents/platforms/toolchains/ascend/knowledge/gemm-tiles.md`（多缓冲条目）+ pattern-library |
| `asc_mmad` 的 M/K 由 L0 region 直接表达：`d_mem=519` 尾段生成 `K=7`、`batch=17` 第二块生成 `M=1`、`clear_accum=(kt==0)` 降低为运行期布尔末参——即「有效区间代替谓词填零」在本工具链可编译且语义落点在 mmad 实参上；旁证 `gemm-tiles.md` 已验证 `(17,19,35)` 的 K=32+3 | D | REVIEW.md 机械复核（ktail 复跑）；`knowledge/gemm-tiles.md:17`；DESIGN.md §0.6 R2/§11.1 H2 | 同上 `/tmp/opencode/rev_probe_ktail.py`（batch=17, d_mem=519, d_pad=256, mcl=12） | 同上 | `knowledge/gemm-tiles.md`（尾块条目补 K=7 一例） |
| 自动提升的 `__mix__(1,2)` block 里，AIV 段的守卫由设计分支**正确重写**：`m0==0 and which==0` → `((bx & 7) >> 2) == 0 && (bx >> 3) == 0`；`sid` 被 `asc_get_sub_block_id()` 读取但**不参与任何守卫**（两 AIV 重复执行）——`T.copy(..., sub_blockid=…)`（`copy_op.py:276` 文档化）是比 `T.Vector as sid` 更直接的 sub-block 路由入口，可保留 copy lowering 的 MTE2→MTE3 同步 | D | REVIEW.md 维度 8 S7 + 机械复核；`tilelang/ascend/language/copy_op.py:238,276`；pattern-library `PL-1.13-aiv-dup-subid-split` | 同上 `/tmp/opencode/rev_probe_ktail.py`（打印 AIV 段守卫与 `sid` 用法） | 同上 | pattern-library `PL-1.13` update + `knowledge/capability-gaps.md`（若 `T.Vector as sid` 丢同步确认为编译缺陷） |

### Transferable Lessons（可迁移教训）

- 设计里写「编译器**没有**做某件事」（多缓冲未生效 / 自动提升不存在 / 某 API 不可用）时，必须自己重跑一次并把**生成代码**当证据；探针的警告文本会跟着结构走（`T.MixedKernel` 的警告不代表普通 `T.Kernel`），跨结构搬运警告会直接误导 Stage 4 的容量假设。
- 判断「有效区间代替谓词填零」是否成立，看 `asc_mmad` 的 M/K 实参（本次实测 `K=7`、`M=1`）比看设计叙述可靠；容量上限（如 L0B 恰好 = 容量）要用**编译通过**来证伪，不要用算术断言代替。
- 逐位等价断言遇到 `±0` 时必须显式说明：多数 ulp 工具把 `-0.0`/`+0.0` 归一（等价于 `float ==`），`0 + x ≡ x` 在 `x = -0` 时不成立；写「fp32 位型逐位相同」前先确认判据没有归一化。
- 迁移任务里如果 harness 的 torch 参考实现与 op 契约冲突（如短 cache 返回 `L+1` 行），golden 侧左填充规避只能救 Stage 3 的分段验证；Stage 5 用的是 harness 自己的 `ref_program`，仍会红——必须在进入 Stage 5 前把「修 Stage 0 工件」作为独立事项交回 conductor，而不是在 kernel 里绕。
- `design_calc_check.py` 的 `core_split` 在只解析到 `block_m/block_n` 的设计上会 skip；此时必须自己按 `mt × which × nt` 复算逻辑核数并与平台 `hardware.core_source` 的实查值比对，不能因为「机械 pass」就放过维度 3。
- 平台 `execution.prototype_command` 的解释器可能缺依赖（本次 `/usr/local/python3.12.16/bin/python3` 缺 `tqdm`，且会 shadow 到 repo dev root 的 `tilelang/`）；探针一律先验证解释器能 `import tilelang` 且版本是 venv 的 ascend 安装，再跑编译，并在报告里如实记录实际解释器。

## Stage 3 (develop) Retrospective — first_impl / attempt 1 — 2026-11-23

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Develop-flow | 平台 `templates.kernel`（`.agents/platforms/toolchains/ascend/kernel_template.py`）是**逐元素 abs** 形态：`@tilelang.jit(out_idx=-1)` + `T.SimtVF` + 单 buffer，没有 Cube 形态的最小骨架（`tilelang.compile(..., out_idx=[])` 预分配输出 + `lru_cache` 工厂 + `alloc_l1/l0a/l0b/l0c` + `annotate_buffer_versions`）。Cube/混合算子无法从模板直接起步，只能从 `examples/*/_*_kernel/*.py` 反推结构，模板的「从平台模板实现」条款对 Cube 类算子实际不可执行 | `examples/engram_decode/_engram_project_kernel/_engram_project_kernel.py`（工厂 + `T.Kernel` + L1/L0 分层）vs 模板全文 | 给平台 `templates.kernel` 增加第二个骨架（Cube tile：`alloc_l1/l0a/l0b/l0c` + `T.gemm(transpose_B=True, clear_accum=)` + `annotate_buffer_versions` + `out_idx=[]` 预分配输出的 `_run_case`），或在 develop skill 写明「Cube 类算子以 `examples/{moe_permute_align,shared_expert_mlp,paged_cache_write}/*_kernel/*.py` 为模板」 | R |
| Develop-flow | skill 的 L0 门槛来自 DESIGN §8.3（本设计给 3 个 L0 用例），但 conductor 的调度 prompt 要求 L0 另含 serving-b4 / `d_mem=519` / `L=0` / `L<mcl` / `d=256/512/1024`（8 例）。两份「L0 计划」在 Stage 1 冻结时没有人比对，Stage 3 只能按更严的一方补齐——若 conductor 未显式列出，`d_mem=519`（本设计唯一的 K 尾形态）会落到 L1，L0 门槛就失去对 E1 风险的判别力 | DESIGN.md §8.3（3 条 L0）vs 本 Stage 调度 prompt 的 L0 覆盖清单；实现见 `_engram_project_kernel.py` 的 `run_L0`（8 例） | 在 `tilelang-op-design` 的 §8.3 条款加「L0 必须覆盖：每个 manifest workload 至少一例 + 每个尾块/边界形态（M/K/N 尾、cache 极端长度、最小收缩轴）至少一例」，使 L0 计划可被 conductor 直接复用而非补丁 | R |
| Info-source | harness 侧的 project 调用面是 `_engram_project_kernel(batch, d_mem, d_padded, max_conv_len, dtype)(block_m, block_n, block_k, num_stages)`（`examples/TileOPs/tileops/kernels/sequence_modeling/engram_decode/engram_decode.py` 内 `project(...)`），而 DESIGN §10.3 只冻结了 `_build_project(...)` 的直接调用形态。Stage 3 必须自造一层同形 shim，否则 Stage 5 集成时 wrapper 与 kernel 的入口名不一致 | `examples/engram_decode/_engram_project_kernel/_engram_project_kernel.py` 的 `_engram_project_kernel()`；对照 `engram_decode.py` 的 `_engram_project_kernel(...)(block_m, block_n, block_k, num_stages)` | 在 `tilelang-op-design` 的交付清单（§10.3）增加一行「必须给出与 harness 调用形态一致的工厂入口（含 configure 参数表）」，避免每个迁移任务的 Stage 3 各自发明入口名 | R |
| Develop-flow | 分层测试的**运行时长口径**没有约定：`--level L0` 冷启动（8 次编译）实测 25.3s、热缓存 2.6s，二者差 10×。Stage 5/4 若按热缓存估时会低估编译开销（真机首次集成每 shape 都要编译） | 本次实测：冷 25.3s（`time ... --level L0`）；`--level all` 20 例热缓存 8.2s | 在 develop skill 的报告模板里把「冷/热」分别记录（编译次数 + 墙钟），并注明 `TILELANG_CACHE_DIR` 命中的判断方式 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| `asc_mmad` 按有效 K 掩蔽尾 lane：收缩长 K ∈ {1, 7, 16, 17, 263, 519}（含「短于一个 K 块」与「256 块 + 7 尾」）在真机上与 fp32 参考一致（max_diff ≤ 3.1e-5，atol/rtol 1e-3）→ `d_mem=519` 不需要 host 补零到 16 的整数倍（DESIGN §9.2 E1 / §11.2 #2 判为已解决） | D | `examples/engram_decode/_engram_project_kernel/_engram_project_kernel.py` 的 L0-ktail-519（max_diff 4.578e-05）、Boundary-dmem17（9.537e-07）、Boundary-dmem1（0.0）；生成代码 `asc_mmad(..., M=(17-max(...)), K=(519-max(kt*256,263)), N=64, ..., (kt==0))` | `cd examples/engram_decode/_engram_project_kernel && source ../../../.agents/platforms/toolchains/ascend/env-950pr.sh && /workspace/venv-ascend950/bin/python repro/CONST-950-mmad-ktail-nonaligned.py`（自包含，K=1/7/16/17/263/519 六例全 PASS） | TileLang ascend 0.1.15 + CANN 9.2.1；Ascend950PR_9579（dav-3510，真机） | `.agents/platforms/toolchains/ascend/knowledge/gemm-tiles.md`（尾块条目补「单段 K<16 与非 16 倍尾」两例）+ pattern-library |
| L0B 的 2 版本在**运行期**就是 100 %：生成代码 `w_l0 + ((e_l1_version_counter_2 & 1) * 16384)`（half 元素 = 32 KiB/版本 × 2 = 65536 B），且 `asc_mmad` 的 L0A/L0B 实参同时用该计数器索引、20 例真机全部通过 → REVIEW §5 S2 的「多缓冲已生效」修正被运行代码证实；任何 block_n/block_k/num_stages 上调都会越界 | D | `_engram_project_kernel.py` 全部 20 例（L0-serving-b4 K 段数 4 仍 PASS）；生成代码实读（b8/fp16 与 ktail-519 两种形状同一计数器形态） | `cd examples/engram_decode/_engram_project_kernel && source ../../../.agents/platforms/toolchains/ascend/env-950pr.sh && /workspace/venv-ascend950/bin/python -c "import _engram_project_kernel as M,re; s=M._build_project(8,512,256,18,'float16',16,64,256,2).get_kernel_source(); print(re.findall(r'w_l0 \+ \([^)]*\) \* 16384', s)[:1]); print(re.findall(r'asc_mmad\([^;]*;', s)[0])"` | TileLang ascend 0.1.15 + CANN 9.2.1；950PR 真机 | `knowledge/gemm-tiles.md`（多缓冲条目）+ `knowledge/PL-950-cube-tile.md`（L0B 100% 配置） |
| `max_conv_len == 1` 时用 **Python 级** `if nshift > 0` 守卫平移，整段 AIV 代码在编译期消失：生成代码既无 `__mix__` 也无 `asc_copy_gm2ub/ub2gm`（纯 AIC kernel，长度 5701 vs 混合形态 7211），且 `clear_accum` 在 `k_tiles == 1` 时折叠成常量 `(bool)1` | D | `_engram_project_kernel.py` 的 `_build_project` 中 `if nshift > 0: if m0 == 0:`；Boundary-mcl1 / Boundary-mcl1-ktail 两例真机 PASS；生成代码对比（`repro` 命令同上一行，改 `(1,256,256,1,'float16')`） | `cd examples/engram_decode/_engram_project_kernel && source ../../../.agents/platforms/toolchains/ascend/env-950pr.sh && /workspace/venv-ascend950/bin/python _engram_project_kernel.py --level boundary` | TileLang ascend 0.1.15 + CANN 9.2.1；950PR 真机 | pattern-library（混合 kernel 的分支消减写法） |
| 自动提升的 `__mix__(1,2)` block 里两个 AIV sub-block **重复执行**同一段 GM→UB→GM 平移（`sid = asc_get_sub_block_id()` 被读取但不参与守卫），实测**逐位安全**：`torch.equal` on int16 view 在全部 cache 用例（含 NaN/±inf/−0.0 载荷）成立，且末行 NaN 哨兵未被触碰 | D | `_engram_project_kernel.py` 的 `_run_case` 断言 2/3（20 例）+ `run_boundary` 的非有限载荷用例；生成代码 AIV 段（`if ((block_idx >> 2) == 0) { for b ... }`，无 sid 守卫） | `cd examples/engram_decode/_engram_project_kernel && source ../../../.agents/platforms/toolchains/ascend/env-950pr.sh && /workspace/venv-ascend950/bin/python _engram_project_kernel.py --level L0` | TileLang ascend 0.1.15 + CANN 9.2.1；950PR 真机 | `knowledge/capability-gaps.md`（把「重复写是否安全」从未决项改为已实测无害）+ pattern-library `PL-1.13` |

### Transferable Lessons（可迁移教训）

- 本 launch 的跨函数契约是**硬**的：`kv` 恒为 fp32 `(2, batch, d_padded)` 且不得提前舍入到输入 dtype；`new_conv_state` 只有 `0 .. max_conv_len-2` 行由投影段写、**末行必须由 step kernel 写**，step kernel 写入前不得假设末行内容（投影段既不读也不写它，实测末行预填的 NaN 哨兵在全部用例中保持）。
- `max_conv_len == 1` 时投影段的平移整段编译期消失（生成代码无 AIV 段）；step kernel 此时是该 cache 唯一写者，必须写第 0 行——不要按「投影段已搬过」来处理。
- 收缩轴 `d_mem` 允许非 16 倍数、甚至小于 16（实测 K=1 精确），因为 `asc_mmad` 按有效 K 掩蔽尾 lane；**不要**为对齐做 host 侧补零（会引入新张量并改变 kernel 参数）。
- harness 的 `default_config` 仍是 GPU 派生值（`block_k = _projection_block_k(d_mem) ∈ {64,32,16}`、`num_stages = 4`），与本平台冻结配置（`block_k=256, num_stages=2`）不同；集成时取 `_engram_project_kernel.py` 的 `default_config(d_pad)`，不要沿用 GPU 默认值。
- 同一 `EngramDecodeFwdOp` 内另一个 launch 若也用 Cube tile，注意 L0B 已被投影段在 2 版本下打满 64 KiB 的**同一张卡上**是每核私有资源（不冲突），但该 launch 自己的 `block_n·block_k·itemsize·num_stages` 必须独立反解，不能照抄 64/256/2。
