# RETROSPECTIVE — chain_speculative_sampling / _chain_speculative_sampling_kernel

## Stage 1 (design) Retrospective — first_design — 2026-11-23

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Design-flow | 设计期探针预算写「≤2 个探针、每探针 ≤10min」，但本任务需要定夺的结构点有 6 类（Philox 表达式 / VF 边界能传什么 / 折叠+归约形态 / 多行分核对 / 尾块与奇数 V / 命中 tile 的 lane 层）。实际做法是「2 个脚本 + 9 个 case」，每个 case 编译 ~2s（总 <2min），预算是按脚本数还是按 case 数计不明确 | `history_version/design_probe_philox.py`（3 case）、`history_version/design_probe_rowfold.py`（6 case），`design_probe_rowfold.py` 的 p2a–p2g | 在 `tilelang-op-design` SKILL.md Phase 2 第 3 项的 D-3 段落把预算口径写成「≤2 个脚本、脚本内 case 数不限、总墙钟 ≤20min」 | R |
| Constraint-doc | `references/programming.md` 列了 `T.SimtVF`/`T.SimdVF` 的接口，但没有一条写「`T.Parallel` 只能出现在 VF 区内」与「`T.alloc_local` 不可跨 VF 边界」——这两条是本次两个探针首版的失败原因，属结构性硬限制 | 探针 p2a 首版报 `T.Parallel loop 'j' must be inside a VF block`；第二版报 `pointer arguments to simt_vf functions must reside in '__gm__ or __ubuf__' address space` | 在 `.agents/platforms/toolchains/ascend/references/programming.md` 的「API 与执行区域」表后加两行硬限制（VF 区归属 / 跨边界实参地址空间） | R |
| Info-source | 平台 `cost_measurements` 为空，pattern-library `constants.md` 的 8 条常数全部绑定 `device: 910B2C`；对 950PR 的 roofline 只能按「未实测外推」标注。设计期没有任何针对 950PR 的实测带宽/发射常数可用，估算下界只能给区间并标注外推 | DESIGN.md §1.6.0 的「设计期估算」行与常数引用段 | 在 constants.md 增加一个「设备外推」小节，明确跨型号外推的允许口径（如「只用于量级判断，判据阈值另行实测」），避免每个任务重复论证 | R |
| Design-flow | 契约里「哪些输出逐位、哪些容差」只写在测试文件（`exact_compare(out_accepted, …)` + `verify_chain_outputs`）与 workload 文档字符串里，`op_requirements` 层没有字段承载。设计期必须读测试全文才能确定精度路径（本任务 `num_accepted` 逐位、token 只结构+分布） | `examples/TileOPs/tests/ops/test_sampling.py` 的 `test_chain_speculative_sampling_fwd`；DESIGN.md §0.1 的「规约语义」三条 | 在 conductor 的 `op_requirements` 契约里加一个可选的 `precision_contract` 字段（逐位 / 容差 / 仅分布），迁移场景由脚手架从测试文件抽取 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| Ascend AIV 无 `__umulhi`：用 4 次 16×16 乘积在 32 位内拼出 32×32 高/低字，逐位等价（160072 个 32 位字 0 不符，含 seed/offset 角点的完整 Philox 十轮流逐位相同） | D | DESIGN.md#§1.6.1 A1 与结果表；`history_version/probe_p1a_gen.txt` 生成纯 32 位整数表达式 | `/workspace/venv-ascend950/bin/python verify_equiv.py`（E1/E2，34s） | TileLang ascend 0.1.15 / CANN 9.2 编译 SDK / 950PR 无卡编译 / 2026-11-23 | `.agents/platforms/toolchains/ascend/knowledge/numerical-boundaries.md`（新条目 PL-950-mulhi-split） |
| 陷阱：`(int64)a * m >> 32` 求 mulhi 在本工具链**能编译但算错**（有符号 64 位溢出 + 算术右移）；只有无符号 64 位或 16 位拆分才对 | D | DESIGN.md#§1.6.1 A1 的陷阱记录；`history_version/probe_p1b_gen.txt` 生成 `((int64_t)c[0] * (int64_t)3528531795) >> (int64_t)32` | `history_version/design_probe_philox.py p1b` | 同上 | `.agents/platforms/toolchains/ascend/knowledge/numerical-boundaries.md`（新条目 TRAP-950-int64-mulhi） |
| 陷阱：Philox 的 key 放在 UB 上、用 `T.Parallel` 一个 lane 一个 draw 时，Weyl 递增被每个 lane 各执行一次（9 lane → key 前进 9 次）；key/counter 必须放 lane 私有寄存器 | P | `history_version/probe_p1c_gen.txt` 的 `if (threadIdx.x < 9) { kw[0] += 2654435769; }` | `history_version/design_probe_philox.py p1c` | 同上 | `.agents/platforms/toolchains/ascend/knowledge/numerical-boundaries.md` |
| Ascend 结构限制两条：① `T.Parallel` 必须在 `T.SimtVF`/`T.SimdVF` 区内；② `T.alloc_local` 不能作为 VF 区实参（状态必须 UB/GM） | P | DESIGN.md#§3.4 与 §3.5.1；探针 p2a 两版的报错文本 | `history_version/design_probe_rowfold.py p2a` | 同上 | `.agents/platforms/toolchains/ascend/references/programming.md`「API 与执行区域」 |
| `T.copy` 在 VF 区内做**动态基址**的区域搬入会编译失败（`allow_append_ is false`）；元素级 `T.Parallel` 读可绕开，且天然支持 4 字节对齐（`V=4099`、`storage_offset=1` 用例） | P | DESIGN.md#§3.4 与 §4.4；`history_version/design_probe_rowfold.py p2e` 失败、p2c 成功 | `history_version/design_probe_rowfold.py p2e` / `p2c` | 同上 | `.agents/platforms/toolchains/ascend/references/programming.md`（搬运小节补一条动态区域限制） |
| 无卡编译探针（`ASCEND_RT_VISIBLE_DEVICES=""` + `tilelang.compile` + 读生成码）足够定夺 API/结构问题：9 个 case 总墙钟 <2 min，抓住 4 个结构性失败点，比纸面推演可靠 | P | DESIGN.md#§3.4 的可行性表；`history_version/probe_*_gen.txt` | `ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python history_version/design_probe_rowfold.py all` | 同上 | `tilelang-op-design` SKILL.md Phase 2 D-3（探针优先于纸面推断的判据） |
| 「同一分布、不必同下标」类随机算子（speculative sampling / 拒绝采样）的等价性判据：零权重命中率必须为 0 + 逐 bin 6σ 分布检验 + 点值相对偏差；不能用逐位或 atol 判定 | D | DESIGN.md#§1.6.1 A4/A6 结果表（68 行 0 分歧、零权重 0 次、512 bin × 20000 抽样逐 bin 相同） | `/workspace/venv-ascend950/bin/python verify_equiv.py`（E5，34s） | 同上 | `.agents/skills/_shared/standards/` 的等价性验证标准（补一个「分布契约」判定类别） |
| 迁移随机算子先读测试文件确定精度契约：本任务 `num_accepted` 必须逐位（`exact_compare`），token 只做结构 + 分布 → fp32 分层前缀可行；若两者都逐位则必须 fp64 前缀（AIV 不可行，需换策略） | C | DESIGN.md#§0.1 三条语义强度、§8.2 精度标准 | none | 同上 | `tilelang-op-design` references/migration-analysis.md §2.1（语义要素表补「精度契约的三级强度」） |

### Transferable Lessons（可迁移教训）

- **先定「逐位 / 容差」边界，再定精度路径**：同一算子内部不同输出可以有不同强度的精度契约（本任务
  `num_accepted` 逐位、被抽 token 仅分布）。把这条读错，要么白做 fp64（本平台根本没有），要么漏掉
  必须逐位的输出而反复返工。
- **能编译 ≠ 算对**：`int64` 乘法这条捷径在本工具链完整编译通过，但数值是错的（有符号溢出）。凡是
  「替代某个 intrinsic」的写法，都要有 CPU 侧的逐位对拍（本任务 `verify_equiv.py` E1/E2），不能只凭
  编译通过。
- **无卡探针的成本极低、判别力极高**：9 个结构 case 共 <2 min，抓住 4 个编译期硬限制（VF 归属、
  local 跨边界、动态区域搬入、lane 共享 key）。在「卡被占用」的窗口期，先跑探针再写设计文档，
  比写完再返工便宜一个数量级。
- **取消一个硬件依赖往往比替换它更省事**：源实现的整网格屏障（`T.sync_grid`）在本平台未认证；与其
  设计跨核 flag，不如把「一行一个核、行内闭合」作为默认，把多核拆行降级成需实测裁决的备选——正确性
  依赖更少，设计也更容易验证。
- **roofline 的常数必须绑定型号**：本平台所有实测常数都是 910B2C 的，950PR 只能外推。把「哪一项是外推、
  哪一项是实测」写清楚，比给一个精确但来源错误的数字有价值。

## Stage 2 (review) Retrospective — review — 2026-11-23

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Review-flow | 9 维度表没有「编程模式选型（§2）」的落点。偏离 `programming_modes.migration_default` 的取证只能在维度 1「API 与模式匹配」里附带判断，而该行的通过标准只要求「与选定模式一致」，装不下「偏离默认值的负向断言是否成立」。本次靠人工扩读 `examples/` 才发现 §2.2「本仓无任何算子用 `T.simd.*` 做逐元素算术」与 `examples/mla_dense_bwd` 三个已通过检视的算子矛盾 | `DESIGN.md#§2.2` vs `examples/mla_dense_bwd/_delta_kernel/_delta_kernel.py:135-147`、`_dq_land_kernel/_dq_land_kernel.py:189-193`（三者 `REVIEW.md` 均「结论: 通过」） | 在 `tilelang-design-review` SKILL.md §3 维度 1 增加一行「模式偏离默认值时的取证核对：负向断言（无先例/无算子用某写法）须对照本仓 `examples/` 全量 grep 复核」 | R |
| Review-flow | `design_calc_check.py` 对「§4.5 用表格列 UB 缓冲、§5.5 用 workload 表列分核」的写法全 skip（`ub_budget`/`core_split` 均 skip），而这恰是本设计最需要机械拦截的两项；检视只能人工复算，且人工结果没有回写位置 | 本次 `design_calc_check` 输出：`ub_budget` skip（no per-level buffer tables parsed）、`core_split` skip（insufficient parsed dims `{'N': 600}`）、`l0c_budget` skip | 扩展 `.agents/tools/design_calc_check.py` 的表格解析：接受「缓冲名 \| 层级 \| 字节」型表格行与「workload \| B \| 逻辑核数 \| 判定」型表格；解析失败时在 detail 里点名需要的表格形态 | R |
| Review-flow | 容差内等价（E5 类）的机器验证脚本内部自带 pass 阈值（`mismatch <= max(1, rows // 100)`，即允许 1% 下标分歧），而 DESIGN 内嵌表只报实测「0/68」。检视者若不读脚本，会误以为「0 分歧」是被断言的，实际是「≤1% 允许 + 实测恰好 0」 | `verify_equiv.py:386`（`"EQUIV_PASS" if zero_weight == 0 and mismatch <= max(1, rows // 100)`）vs `DESIGN.md#§1.6.1` 结果表 | 在 `tilelang-design-review` SKILL.md §4 Phase 1 第 7 步要求「记录脚本自身的判定阈值，不只记录实测值」；并建议 design 侧 verify_equiv 模板把阈值打印进结果行 | R |
| Constraint-doc | 「探针跑了但没有生成码 / 没写进正文」的证据缺口没有检查项：p2e 因编译失败无生成码属合理，p2g（`_locate_bisect`）却既无 `probe_p2g_gen.txt` 也未在 §1.6.2 的替代方案里出现，而 §10.1 的目录清单用省略号暗示 p2a–p2g 齐全 | `history_version/`（只有 p1a/p1b/p1c/p2a/p2b/p2c/p2d/p2f 生成码）vs `DESIGN.md#§10.1`、`#§1.6.2 V11` | 维度 7 的「同类引用真实」旁加一条「§10 列举的证据文件与 `history_version/` 实际内容逐一对应（缺文件须在正文说明原因）」 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| ascend 方言的 `T.reinterpret` 是**值在前**（`reinterpret(value, dtype)`），与 `tirx.reinterpret(dtype, value)` 相反；跨后端查签名时容易误判「该 API 不存在/用法不同」 | D | 任务目录下 `inspect.signature(T.reinterpret)` → `(value, dtype, span=None)`；`T.__file__` = 安装包 `tilelang/ascend/language/__init__.py`；探针 `probe_p1a_gen.txt` 生成 `(*(uint32_t*)(&v_))` | `cd <task_dir> && /workspace/venv-ascend950/bin/python -c "import inspect, tilelang.ascend.language as T; print(T.__file__, inspect.signature(T.reinterpret))"` | TileLang ascend 0.1.15 / CANN 9.2.1 / 950PR / 2026-11-23 | `.agents/platforms/toolchains/ascend/references/programming.md`（API 表补 `reinterpret` 的参数序） |
| 检视期复核「某写法编译失败」类断言的做法：把 `history_version/*.py` 探针复制到 `/tmp` 后再跑（脚本把生成码写在自己目录，复制后不会污染算子目录），配合 `ASCEND_RT_VISIBLE_DEVICES=""` 只做编译 | P | 本次复跑 `design_probe_rowfold.py p2e` → `tvm.error.InternalError: Check failed: (allow_append_) is false: at[T.FloorDiv(267008, 32)] shared.dyn`，与 `DESIGN.md#§9.2` 第 6 条一致 | `cp design_probe_rowfold.py /tmp/opencode/probe/ && cd /tmp/opencode/probe && ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python design_probe_rowfold.py p2e` | 同上 | `tilelang-design-review` SKILL.md §4 Phase 1（弃选论证查证段补「探针复跑」的操作口径） |
| 迁移「分布契约」类算子（speculative sampling / 拒绝采样）的检视判据不是 atol/逐位，而是三条同时成立：零权重命中 = 0、逐 bin 6σ 分布检验、点值相对偏差量级；且**必须读测试文件确认哪些输出是逐位**（本任务 `exact_compare` 只用在 `num_accepted`，token 只有结构 + 分布） | C | `DESIGN.md#§0.1/§8.2` vs `examples/TileOPs/tests/ops/test_sampling.py:79`、`workloads/sampling.py#verify_chain_outputs/assert_chain_distribution` | `grep -n "exact_compare" examples/TileOPs/tests/ops/test_sampling.py` | 同上 | `tilelang-design-review` SKILL.md §3 维度 6（精度标准：随机算子按「逐位/分布」分级，不套 atol/rtol） |

### Transferable Lessons（可迁移教训）

- **「本仓无 X」这类负向断言先 grep 再信**：本任务设计写「无任何算子用 `T.simd.*` 做逐元素算术」，
  实际 `examples/mla_dense_bwd` 三个已通过检视的算子都在用。检视 `examples/` 全量 `grep -rn "T.simd\."`
  的成本是一分钟，代价是可能放过一条支撑模式选型的错误依据。
- **机械复核 skip ≠ pass**：`design_calc_check` 的 `ub_budget`/`core_split` 对本类文档大概率 skip，
  必须人工复算并写进 REVIEW 的机械复核段，否则「机械复核已跑」是空头保证。
- **等价性脚本的 pass 阈值要单独看**：`verify_equiv` 的容差项可能允许 1% 分歧，「实测 0」与
  「断言 0」是两件事；检视报告应同时记录阈值与实测。
- **迁移随机算子先读测试断言类型**：同一算子内不同输出可以有不同精度强度（本任务 `num_accepted` 逐位、
  被抽 token 仅分布）。读错会把「分布等价」写成「逐位」，导致 Stage 3 白追一个不可能达成的目标。

## Stage 3 (develop) Retrospective — first_impl / attempt 1 — 2026-11-23

结果：`[PRECISION_PASS]`。`L0 13/13`、`L1 7/7`、`L2 5/5`、`boundary 7/7`（共 32 case，单进程
`--level all` exit 0）。全部约 6.7 万行上 `num_accepted` 与参考**逐位**一致，被抽 token 与参考的
**下标也逐位一致**（`drawn_token_mismatches=0`）——设计允许的「结构 + 分布」容差一次都没用上。
设备实查 `Ascend950PR_9579 cube=28 vector=56`（`run_L0` 内断言 = 设计的 56）。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Develop-flow | 平台 kernel 模板与 `references/programming.md` 都没有「核网格上限」与「`T.SimtVF` 区与串行行循环的嵌套关系」两条约束，而这两条是本任务真机失败的主因（网格 65536 直接 abort；一个 VF 区跨整个行循环在部分 shape abort） | `repro/TRAP-950-grid-over-65535.py`（65535 OK / 65536 abort）；`repro/TRAP-950-dynamic-base-tile-pass.py`（同一 pipeline 两种拼写：一种 exact、一种 abort） | 在 `.agents/platforms/toolchains/ascend/references/programming.md` 增加「核网格 ≤ 65535」硬上限，并在 VF 区一节写明「VF 区应闭合在行/任务粒度内，不要跨串行行循环复用」 | R |
| Develop-flow | 设计期探针全部是「无卡编译」（`DESIGN.md#§3.4` 最后一行「真机数值 未做」），因此 §9.1 R1 列的三类风险（reducer 数值、同步开销、标量前缀）在 Stage 3 才暴露；而一次 device task abort 会污染进程的 device 上下文，该 level 后续 case 全部连带失败（L0 第一次跑只剩 6/13） | 首次 `--level L0` 日志：L0-7 起全部 `device error type 3, error code is 507035`，直到进程退出 | 在 `tilelang-op-develop` SKILL.md Phase 4 增加一条：「device abort 会污染流，调试图层应按 case 分进程跑（或每 case 前重建 device 上下文），否则一个失败会伪装成一片失败」 | R |
| Info-source | `TILELANG_CACHE_DIR` 是跨模块共享的编译缓存；用「改写同一文件生成变体模块」的方式做 A/B 时，未改写的运行会命中变体编译出的二进制，导致「同一 shape 一次 PASS 一次 FAIL」的假象（本任务因此一度把失败集判错，浪费一轮） | 实测：(2,2,16000) 在共享 cache 下 PASS，`TILELANG_CACHE_DIR=<fresh>` 下 FAIL；随后用逐变体独立 cache 复测才稳定 | 在 `tilelang-debug-helper` / `tilelang-op-develop` 的调试段落写明：变体/拼写对比必须给每个变体独立的 `TILELANG_CACHE_DIR` | R |
| Constraint-doc | 设计把「被抽 token 用 fp32 分层前缀」标注为容差内等价（A4），而实现下来 6.7 万行 0 分歧：说明该等价在本 workload 分布下实际是逐位的，但下一阶段无从判断「容差是否还需要留」 | `_chain_speculative_sampling_kernel.py#run_L1 L1-7`（`assert_chain_distribution` 通过）+ 全套 `drawn_token_mismatches=0` | 在 `tilelang-op-design` 的精度分级段落补一条：「容差内等价项在设计阶段记录上界，Stage 3 若实测 0 分歧则回写实测值，供 Stage 4/5 决定是否收紧断言」 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 核网格 `T.Kernel(grid)` 的 `grid` 上限是 65535：65535 个 block 正常完成，65536 与 131072 触发 `device error type 3, error code is 507035`（AIV 位图全 1），kernel 体本身只读写 `X[bx]/Y[bx]`。多行折叠必须用「≤56 个 block + 串行行循环」，不能写成 `grid = batch` | D | `repro/TRAP-950-grid-over-65535.py`（driver 断言 65535 exit 0、65536 exit≠0）；实现侧 `_plan()` 的 `min(batch, _MAX_BLOCKS)` | `cd <task_dir> && source ../../../.agents/platforms/toolchains/ascend/env-950pr.sh && /workspace/venv-ascend950/bin/python repro/TRAP-950-grid-over-65535.py` | TileLang ascend 0.1.15 / CANN 9.2.1 / Ascend950PR_9579 (dav-3510) / 2026-11-23 | `.agents/platforms/toolchains/ascend/references/programming.md` |
| 跨行/跨任务的 tile 重算若写成「一次 `T.Parallel(tile)` + 运行期基址（`pick*tile`）+ 谓词」，在部分词表上 abort；改成「`for t in T.serial(ntile): if t == pick: T.Parallel(tile)`（基址是循环变量 `t*tile`，即折叠 pass 自己的形态）」后同一 shape 逐位正确。失败 shape：(2,2,16000)、(112,1,32000)；通过 shape：(2,2,8193)、(2,2,128256) | D | `repro/TRAP-950-dynamic-base-tile-pass.py`（两臂：shipped exact / flat abort）；`_chain_speculative_sampling_kernel.py#stage C` 注释 | `cd <task_dir> && … && /workspace/venv-ascend950/bin/python repro/TRAP-950-dynamic-base-tile-pass.py` | 同上 | `knowledge/repro/` + `programming.md`（`T.Parallel` 基址形态） |
| 「一个 `with T.SimtVF` 跨整个串行行循环（reducer 只分配一次）」在 (2,2,16000)、(112,1,32000)、(256,1,32000) 上 abort；把 VF 区闭合到「一行一个」（reducer 在区内分配）后通过。与上一条需**同时**满足：单独任一修改仍 abort | D | 任务内实测矩阵（`vfrowr1` = 两条都改：20 个 shape 全 PASS；`vfrow`（只改 VF）与 `r1`（只改重算）分别仍在 (2,2,16000)/(112,1,32000) 失败） | none（未缩到最小自包含内核；与上一条同属「运行期基址 + VF 跨行」类，复现见上一条的 flat 臂） | 同上 | `knowledge/repro/`（待补最小复现） |
| 元素级 GM 读不需要自己钳位：后端为每次 load 生成 `0 <= addr && addr < total` 守卫（失败取 0），因此谓词 `i0 < vocab` + 仿射下标 `at_t + i0` 已安全；生成的守卫在 8193/16000/21760/128256/151936 的折叠与重算 pass 中都可见。反过来在该结构上加 `T.min(i0, vocab-1)` 并非必要（当前结构下也不再触发 abort，但守卫已足够） | D | `history_version/`（生成码抽查）；`_residual()` docstring | `cd <task_dir> && /workspace/venv-ascend950/bin/python history_version/probe_s3_compile.py` 后查 `get_kernel_source()` 中 `condval` 守卫 | 同上 | `programming.md`（GM 元素读的界守卫语义） |
| SIMT 域内的标量代码由全部 lane 锁步执行，指令代价与单 lane 相同（不是 128 倍）：Philox 十轮 × (N+1) 个 draw、verdict、前缀与元素走查全部放在 `T.SimtVF` 内实现，`N=600` 压力用例稳态 2.18 ms，主流程无额外代价——设计「把阶段 A 放到 VF 外以免 128 倍冗余」的顾虑不成立 | D | `history_version/probe_s3_time.py`（`4 600 64`：steady 2.18 ms；`32 5 151936`：1.33 ms） | `cd <task_dir> && … && /workspace/venv-ascend950/bin/python history_version/probe_s3_time.py 4 600 64` | 同上 | `programming.md`（SIMT 执行模型：标量代码的代价模型） |
| 调试 device task abort 的定位手法：① 先看 CANN device log 的 `aivmap` 位图（全 1 = 系统性，不是某行数据）；② 每 case 独立进程 + 每拼写独立 `TILELANG_CACHE_DIR`；③ 按「VF 区位置 → 重算/折叠 pass 形态 → 步进标量循环」逐条回退，一次只回退一处 | P | `repro/TRAP-950-*.py`；本任务 L0 首次失败矩阵（L0-13 与 (2,1,32000) 等） | 见两条 repro 的 driver 结构（子进程隔离） | 同上 | `tilelang-debug-helper` SKILL.md（device abort 分支） |

### Transferable Lessons（可迁移教训）

- **核网格 ≤ 65535**：`T.Kernel(grid)` 的 grid 超过 65535 直接 abort（`507035`）。多行/多任务折叠
  一律「少量 block + 串行任务循环」，不要为了省一层循环写 `grid = batch`。
- **device task abort 会污染进程**：一次 `device error type 3` 之后同进程的所有 NPU 调用都会失败。
  分层测试要按 case 分进程（本任务 `run_case` 捕获异常并继续，但设备已死——最终 evidence run 必须
  一次 `--level all` 跑通，以证明没有先死过设备）。
- **变体对比必须隔离编译缓存**：`TILELANG_CACHE_DIR` 共享时，改写同一源码生成的变体会污染未改写
  运行的二进制，造成「同 shape 时通时不通」；每个拼写一个独立 cache 目录。
- **谓词化的 tile pass 不要换成运行期基址**：把 `for t in serial(ntile): if t == pick: T.Parallel(tile)`
  简化成 `T.Parallel(tile)` + `pick*tile` 基址，在部分词表（16000、32000）上 abort；基址留在
  循环变量上（与折叠 pass 同形）才稳定。
- **`T.SimtVF` 区闭合在任务粒度**：一个 VF 区跨整个串行行循环（reducer 只分配一次）会在部分 shape
  abort；一行（一个任务）一个区，reducer 在区内分配。
- **SIMT 内标量代码不按 lane 数计价**：锁步执行，标量链（Philox/verdict/前缀/走查）放在 VF 区内
  不需要额外的「防 128 倍」设计。
- **元素级 GM 读靠编译器界守卫**：生成码里每次 load 带 `0 <= addr && addr < total`（失败取 0），
  谓词 + 仿射下标即可，无需在索引上做 `T.min` 钳位；交付前抽查生成码确认守卫存在。
