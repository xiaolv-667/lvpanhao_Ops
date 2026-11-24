## Stage 1 (design) Retrospective — first_design — 2026-11-23

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Design-flow | 设计 skill 的「跨阶段融合」重设计模式（migration-analysis §6.2）只提示「按目标平台核对计算单元协作、阶段间数据路径和同步」，**未提示核查融合后两个阶段的 GM 写域是否重叠**——而写域重叠 + 无跨核 barrier 会让「把常量填充交给另一个核」这一看似显然的并行化变成间歇性正确性错误 | `examples/moe_permute_align/_make_small_batch_kernel/DESIGN.md` §0.5 D6 / §0.6 R1 / §9.1 风险 4 | 在 migration-analysis.md §6.2「跨阶段融合重设计」补一条检查项：「融合后各阶段的写域若有交集，则它们在单 kernel 内必须由同一核顺序执行（平台无跨核 barrier 时不可按写域分核）」 | R |
| Design-flow | Phase R 的 R2（在线算法）对「存不存在单遍变体」只要求结构判据，但**没有要求把单遍变体的实际降级码拿出来核对**——本次单遍 staging 变体结构上完全成立，实测降级码却在每轮 token 循环内插一个 `asc_syncthreads()`（600 次），若只做纸面结构分析会误判为「可采纳」 | `examples/moe_permute_align/_make_small_batch_kernel/DESIGN.md` §1.6.0 C3 行 / §1.6.0 R2；`history_version/design_probe_forms.py` K4、`history_version/design_probe_refined.py` M1 | 在 algorithm-research.md §3 R2 的「完成标准」补充：**单遍变体若依赖数据相关片上索引访问，须以编译探针的降级码（是否插入同步/barrier）为采纳依据**，不得只按结构判据裁决 | R |
| Info-source | 平台 `execution.precision_command` 指向的 `/usr/local/python3.12.16/bin/python3` 在本机不可用于本工具链，实际须用 `/workspace/venv-ascend950/bin/python`；同 op 前两个函数的设计也各自记录了同一条偏差（每个函数都要重新发现一次） | `examples/moe_permute_align/_make_small_batch_kernel/DESIGN.md` §4.7；`examples/moe_permute_align/_make_align_kernel/DESIGN.md` §4.7 | 在 `statectl platform` 的 `execution` 输出中按 task 记录实测可用的解释器路径（或由 conductor 在调度 prompt 中透传一次），避免逐函数重复发现 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 数据相关 UB 下标访问（`s[data_dep_idx]`）在该工具链上会**在每个循环迭代插入 `asc_syncthreads()`**，并在发散守卫上打印 `[ThreadSync] Hoisting sync out of an if ... the race remains` 告警 | D | `examples/moe_permute_align/_make_small_batch_kernel/history_version/design_probe_refined.py` M1（读 UB + UB 原子 → 19 个 `asc_syncthreads()` + 告警）vs M5（读 GM + UB 原子 → 0 个 barrier、0 告警）；`history_version/design_probe_forms.py` K4（staged 形态 600 次）与 K5 | `cd examples/moe_permute_align/_make_small_batch_kernel && bash -c 'source /workspace/tilelang-mlir-ascend-950/.agents/platforms/toolchains/ascend/env-950pr.sh; ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python history_version/design_probe_refined.py'`（比对 M1 与 M5 的 `main_kernel_simt_vf_*` 源码） | tilelang 0.1.15 + CANN 9.2.1（编译 SDK）+ Ascend950PR_9579（无卡编译） | `.agents/platforms/toolchains/ascend/knowledge/numerical-boundaries.md`（新 TRAP 条目）/ pattern-library `traps-compiler.md` |
| `T.cumsum` 在 ascend 后端不可用：报 `tl.cumsum requires a target-specific implementation, but no cumsum implementation is registered for {"kind":"ascend",...}` | D | `examples/moe_permute_align/_make_small_batch_kernel/history_version/design_probe_forms.py` K7（COMPILE_FAIL + 完整报错文本） | 同上脚本的 K7 段 | tilelang 0.1.15 + CANN 9.2.1 + Ascend950PR_9579（无卡编译） | `.agents/platforms/toolchains/ascend/knowledge/numerical-boundaries.md`（与同 op scatter 设计的 K8 合并为一条） |
| UB `int32` 原子自增可取旧值在本工具链**编译可用**：`T.atomic_add(shared[i], 1, return_prev=True)` → `asc_atomic_add(&s[i], 1)`，头文件声明 `asc_atomic_add(__ubuf__ int32_t*, int32_t)`（`__NPU_ARCH__ == 3510`） | D | `examples/moe_permute_align/_make_small_batch_kernel/history_version/design_probe_forms.py` K6；`/usr/local/Ascend/cann-9.2.1/asc/include/simt_api/device_atomic_functions.h:25` | 同上脚本的 K6 段 | tilelang 0.1.15 + CANN 9.2.1 + Ascend950PR_9579（无卡编译；真机语义未验证） | `.agents/platforms/toolchains/ascend/knowledge/execution-patterns.md`（补「UB 原子」小节，与 PL-950-atomic-splitk 的 GM 原子并列，标注真机未验证） |
| 融合 kernel 的「常量填充 + 散射」写域检查法：填充写满 `[0,P)`、散射写 `[0,total)` 的子集 ⇒ 两者必须同核（无跨核 barrier 平台）；本 op 因此把 GPU 的 fill/worker 双线程组降级为同核前序 | P | `examples/moe_permute_align/_make_small_batch_kernel/DESIGN.md` §0.5 D6 / §0.6 R1 / §9.1 风险 4；探针 M9 的 sync 链（`notify(PIPE_MTE3, PIPE_V)` → `wait` 落在散射 VF 之前） | `history_version/design_probe_refined.py` M9–M11 | tilelang 0.1.15 + CANN 9.2.1 + Ascend950PR_9579 | `.agents/skills/tilelang-op-design/references/migration-analysis.md` §6.2 或 `.agents/platforms/toolchains/ascend/knowledge/bottlenecks.md` |
| 单核 kernel 在多核 NPU 上的分核论证模板：当全部阶段构成一条顺序依赖链时，逻辑核数 = 1 是**正确结论**而非「未适配」，须同时给出「跨核归约不可用 / 写域重叠 / 量级 < 2 µs」三条依据 | P | `examples/moe_permute_align/_make_small_batch_kernel/DESIGN.md` §5.5 ③ | 无需复现（论证结构） | tilelang 0.1.15 + CANN 9.2.1 + Ascend950PR_9579 | `.agents/skills/_shared/standards/core-split-strategy.md` §1 要素③（补「单核链式算子」这一合法结论） |

### Transferable Lessons（可迁移教训）

- 该工具链上**不要**直译 GPU 的「每 lane 私有行」结构（`cnts[lane][eid] += 1`）：数据相关的 UB 下标
  读改写会让编译器在每个循环迭代插一个 `asc_syncthreads()`，并在发散守卫上打印 `[ThreadSync] ... the
  race remains`（正确性告警，不只是性能问题）。等价且干净的替代是 **lane↔expert 比较累加**
  （每 lane 一个私有累加器，热循环内零 UB 下标访问）。
- 若必须用 UB 原子（`atomic_add(shared[i],1,return_prev=True)`），**把同一循环内的 UB 读改成 GM 读**：
  实测只有这样编译器才不插逐轮 barrier（M1 读 UB → 19 个 barrier；M5 读 GM → 0 个）。
- 融合 kernel 内若有「全量写 + 子集重写」两个阶段（如哨兵填充 + 结果写），它们**必须同核**并按程序顺序
  由自动流水排序；跨核拆分在本工具链不可排序（`CG-950-batch-barrier`），错误形态表现为「结果偶尔被
  常量覆盖」。
- `T.cumsum` 在 ascend 后端不可用（无 target-specific 实现）；前缀和用 `log2(W)` 步 Hillis–Steele 扫描
  手写，步数用 **python `for d in range(N)`** 展开（`@T.prim_func` 内 `while` 会报
  `Infinite while loop detected`）。
- 前缀扫描/累加器形态的**补位 lane 纪律**：lane 数取 2 的幂（W=32）时，`e ≥ E` 的 lane 计数天然为 0，
  左移一步要写满 `[0, W]` 使补位值成为**定义值**，而所有下游归约的上界取逻辑长度 `E`（不是 `W`）——
  这条在本 op 的两个 kernel（align / small_batch）上都成立，且可用「补位 lane 填 UB 垃圾」的判别性对照
  验证它是否承重。
- 设计期探针要核对**降级码里的 GM 偏移表达式**而不只是 `COMPILE_OK`：本次第一版填充切片公式写错，
  因写入区越界被 dead-code 消除，`COMPILE_OK` 完全掩盖了它，是通读降级码才发现的。

## Stage 2 (review) Retrospective — 2026-11-23

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Info-source | 设计文档用裸编号引用实测常数（`CONST-mte2-degradation` / `CONST-capacity-910B2C` / `CONST-vector-launch-overhead`），但这些条目**不在平台 `knowledge/` 目录**（该目录无 `constants.md`），实际位于 `.agents/skills/tilelang-op-optimize/references/pattern-library/constants.md`；检视 D-2 要求「引用平台知识库中匹配型号和版本的常数条目」，却未给定位路径，检视方须全仓 grep 才能核对数字（43.1 / 18.4 GB/s/core、~1.7× 膨胀、0.5 µs/op） | `.agents/skills/tilelang-op-optimize/references/pattern-library/constants.md:8`（CONST-capacity-910B2C）/`:46`（CONST-vector-launch-overhead）/`:82`（CONST-mte2-degradation）；`examples/moe_permute_align/_make_small_batch_kernel/DESIGN.md` §1.6.0 R4 | 在 design skill 的「常数引用规则」中要求引用格式含**文件路径 + 条目 id**（如 `pattern-library/constants.md#CONST-mte2-degradation`），并在 review skill 的 D-2 步骤写明常数条目的默认位置（平台 `knowledge/` 无命中时转 pattern-library） | R |
| Review-flow | 「设计期探针的降级码」这一承重证据没有留存规范：DESIGN §3.4 只写结论（「M5/M6 无 barrier」「M9 的 notify/wait 落在散射前」），探针脚本只 `print(src[:7000])`，关键片段（M9 尾部的 `asc_sync_notify(PIPE_MTE3, PIPE_V)`、M1 的 `asc_syncthreads`）常被截断 ⇒ 检视方**必须重跑探针再 grep dump** 才能区分「真复现」与「自述」 | `history_version/design_probe_refined.py`（`print(src[:7000] …)`）；本次检视重跑 `/tmp/opencode/probe_refined.log` 后 grep 到 M1 的 `for (r<19) { asc_syncthreads(); … }` 与 M5 的 0 命中 | design skill 明确「承重编译结论须在 DESIGN 内贴**最小降级码片段**（≤10 行）+ 探针 check 名 + 复跑命令」；review skill 的机械复核清单加入「承重降级码片段是否存在」一项 | R |
| Review-flow | 机械复核只规定「verify_equiv 执行结果与内嵌表**一致**且全 EQUIV_PASS」，未规定**行标签对账**：本次设计表把「补位 lane 纪律」标为 A3、「numel=0」标为 A7，脚本输出同名两行为 `A6 idle lanes` / `A7`（设计表的 A6 另指哨兵填充）；用例数与数值 14/14 行全对，但标签错位会让后来者无法按编号对账 | `examples/moe_permute_align/_make_small_batch_kernel/verify_equiv.py` 输出 vs `DESIGN.md` §1.6.1 结果表 | 在 review skill 的机械复核项补一句：「数值与用例数一致而标签不一致 ⇒ 记建议级，不判 fail；并给出标签映射建议」 | R |
| Review-flow | 缺「本 kernel 的触发条件 ∩ manifest/bench workload」的检查项：本 op 12 条 manifest workload 的 `num_experts ∈ {128,256,384}`，而本 kernel 只在 `E ≤ 32 and numel < 1024` 触发 ⇒ **空交集**（bench 由 manifest 驱动，故本路径无任何 bench 覆盖），而设计 §8.3 误称「L1/L2 覆盖 manifest 中 E 小的档位」。维度 6 现状（golden + L0 计划 + 精度标准）拦不住这类「纸上覆盖」 | `examples/TileOPs/tileops/manifest/moe.yaml`（workloads 段）；`examples/TileOPs/benchmarks/ops/bench_moe_permute_align.py`（`workloads_to_params`）；`DESIGN.md` §8.3 末段 | review skill 维度 6 增加一项：「核对算子路径触发条件与 manifest/bench workload 集合的交集；空交集须由设计要求显式声明『性能验证需自建 shape』」 | R |
| Review-flow | 设计期「工作负载上界」缺少机械可查项：`block_size` 由调用方给定且无上界，`P = numel + (E+1)(B-1)` 随 `B` 线性增长，而 roofline/指令界只在 `B ≤ 64`（`P = 3,102`）档成立；本设计用「本路径最大 workload」表述但只在 §0.2 用「取 B = 64 时」半句限定 | `DESIGN.md` §0.2 / §0.6 R1 / §1.6.0 R4 / §5.4 | design skill 的 roofline 小节要求「代表 workload 的选取依据 + 该口径不成立的 shape 区域」两行；review skill 维度 8 增加「roofline 口径的适用边界是否声明」的检查 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 多路径工厂算子（同一 op 由多条构建路径共用输出契约）的**契约核对三角**：① 源码三处写点（哨兵填充 / `expert_ids` owner 与尾段 / `num_tokens_post_pad`）；② op 参考实现的**补齐值**（`ref_program` 用 `numel` 补 `sorted_token_ids`、用 `num_experts` 补 `expert_ids` 尾段）；③ 测试的**归一化断言**（`_assert_route_equivalent` 按 (expert, 路由索引) 归一）。只有三者齐备才能判定「expert 内顺序」与「尾段值」确实属契约自由度 | P | `examples/TileOPs/src/tileops/_kernels/moe/permute_align.py:265-332`；`examples/TileOPs/tileops/workloads/moe.py:139-152`；`examples/TileOPs/tests/ops/test_moe_permute_align.py:58-84`；本次 REVIEW.md 维度 0 的 7 项核对 | `cd examples/TileOPs && python -m pytest tests/ops/test_moe_permute_align.py -k routes-1023 --collect-only`（只收集，不需卡）即可定位归一化入口 | tilelang 0.1.15 + CANN 9.2.1 + Ascend950PR_9579 | `.agents/skills/tilelang-design-review/SKILL.md` 维度 0（新增「多路径契约核对三角」检查项） |
| 检视期「承重证据重跑」协议：凡 DESIGN 以「探针降级码」为唯一证据的结论（barrier 计数、sync 链位置、指令折叠形态），检视须**重跑探针 + grep dump** 并与 DESIGN 的定性描述逐条比对，才算核对通过；两探针在无卡环境合计 < 1 min（11/11 与 6/7 COMPILE_OK） | P | 本次实测：`design_probe_refined.py` 11/11 COMPILE_OK、`design_probe_forms.py` 6/7（K7 负对照），M1 复现 19 次 `asc_syncthreads`、M5 为 0、M9 的 `notify(PIPE_MTE3, PIPE_V)` 落在散射 VF 前 | `cd examples/moe_permute_align/_make_small_batch_kernel && bash -c 'source /workspace/tilelang-mlir-ascend-950/.agents/platforms/toolchains/ascend/env-950pr.sh; ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python history_version/design_probe_refined.py'` 然后 `grep -c asc_syncthreads` 分区段统计 | tilelang 0.1.15 + CANN 9.2.1（编译 SDK）+ Ascend950PR_9579（无卡编译） | `.agents/skills/tilelang-design-review/SKILL.md` Phase 1 第 7 步（机械复核） |
| `return_prev=True` 的原子在 ascend 后端的**落点**：走 `tl.atomic_add_ret_elem_op`（返回值为表达式形式）→ `asc_atomic_add(ptr, val)`；`tl.simd.vscatter` 的 `base` 被 codegen 强制强转为 `__ubuf__ … *`（故不能 GM 散射） | D | codegen 静态核对（观测值）：`/workspace/venv-ascend950/lib/python3.12/site-packages/tilelang/src/ascend/codegen/codegen_ascend.cc:2502-2540`（ret 形态）、`:2444-2457`（vscatter 的 `(__ubuf__ …*)` 强转）；探针 M9 的降级码 `int32_t slot = asc_atomic_add((&…), 1);`。口径：源码静态核对，**非**性能实测（与 Stage 1 已登记的 K6「编译可用 + 头文件」条目互补，标的物为 codegen 落点而非头文件声明） | `grep -n "atomic_add_ret_elem_op\|simd_vscatter" /workspace/venv-ascend950/lib/python3.12/site-packages/tilelang/src/ascend/codegen/codegen_ascend.cc` | tilelang 0.1.15 + CANN 9.2.1（编译 SDK）+ Ascend950PR_9579 | `.agents/platforms/toolchains/ascend/knowledge/execution-patterns.md`（UB 原子小节补「codegen 落点」）；同页补「`vscatter` 仅 UB」的限制条款 |

### Transferable Lessons（可迁移教训）

- 检视阶段**重跑设计期探针并从降级码 grep 观测值**，是区分「承重证据」与「设计自述」的唯一手段：本次 M1 的
  19 次 `asc_syncthreads`、M5 的 0 次、M9 的 `notify(PIPE_MTE3, PIPE_V) → wait` 位置，全部是我自己重跑后在 dump 里
  数出来的，DESIGN 只给了结论。两个探针在无卡环境合计 < 1 min，成本远低于误放行的代价。
- 迁移检视的契约核对要走**三角**：源码写法 + op 参考实现的补齐值 + 测试的归一化断言。只看源码会漏「顺序自由度」
  与「尾段值」的契约边界；只看测试会漏「源码是否真的能产出该值」。本任务的「expert 内顺序未定义」正是靠测试的
  `_assert_route_equivalent` 归一化才敢判 A5 为契约级等价。
- 平台的 `CONST-*` 实测常数在 `.agents/skills/tilelang-op-optimize/references/pattern-library/constants.md`，**不在**
  `knowledge/` 目录；D-2 的「匹配型号和版本」核对必须落到该文件，并同时核对「该常数测的是不是设计用的那个量」
  （本次发现设计把「向量算子发射 0.5 µs/op」外推成「kernel launch ≈2 µs」，属不同量的外推 → 建议级）。
- 按 shape 特化的工厂算子（多路径分派）要把「触发条件 ∩ manifest workload」当必查项：本 op 的 12 条 manifest
  workload 全部 `E > 32`，与 `E ≤ 32` 的本 kernel 空交集 ⇒ bench 完全覆盖不到本路径，任何「L1/L2/bench 覆盖了
  manifest 小档位」的措辞都是纸上覆盖。
- `block_size` 这类「调用方给定、无上界」的参数会让 roofline 的「最大 workload」失去封闭性：`P ∝ B`，填充次数与
  流量随 `B` 线性增长；检视时须要求设计显式声明 roofline 口径的适用 shape 区域（本次 `B ≤ 64`）。

## Stage 3 (develop) Retrospective — first_impl / attempt 1 — 2026-11-23

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Develop-flow | develop skill 与平台 `programming.md` 都只说「`@T.prim_func` 内禁用 `while`」，**未说明 eager builder 会把 body 里所有 `for` / `if` 都 trace 成 TIR 节点**：`for t in range(n)` 的循环变量是运行期 `PrimExpr`，于是 python 内建 `min()/max()/if a and b` 作用在它上面直接抛 `ValueError: Cannot use and / or / not operator to Expr`（首次编译即失败，报错文本不含行号语义） | `_make_small_batch_kernel.py` §3 `_compile_small`（首版用 `min(FILL_TILE, p_total-off)` 于 `for t in range(n_fill)` 内）；失败日志 `/tmp/opencode/smoke1.log:269` | 在 develop skill §4 的 `{op}.py` 结构规范补一条：「构建期常量必须在 traced body **之外**算好并绑定为局部名；`for ... in range(n)` 的变量是运行期表达式，故其派生量不得喂给 python `min/max/and`，改用 `T.min/T.serial` 或对齐 kernel 的『整块 range 循环 + 静态尾块』写法」 | R |
| Develop-flow | 「用常量 padding（`-1`）消掉热循环边界守卫」这类**设计期不变量**没有被要求按目标平台的搬运粒度复核：一个 `T.copy` 的部分块搬运会把长度上取整到 32 B，落在 padding 区，且因为两段写域不相交而**没有依赖边**，编译器不会排序 ⇒ padding 被搬运的尾块覆盖，热循环把垃圾计入直方图（本次两例静默错算：`total` 15→20、1→8） | `_make_small_batch_kernel.py` §3（`if tail > 0:` 段的注释）；`repro/TRAP-950-partial-gm2ub-load.py`；失败用例 L0-13 / L0-16（首轮 `[L0] 15/17`） | 在 design skill 的「尾块处理」小节与 develop skill 的失败分类里各补一条：「凡以『常量填充 + 固定循环上界』替代边界守卫的方案，须先实测该平台部分块搬运的实际写入粒度（是否上取整、目标缓冲是否留 guard band），并用 copy 与 fill 之间**是否存在依赖边**判断先后顺序」 | R |
| Info-source | 平台 `execution.precision_command` 的 `/usr/local/python3.12.16/bin/python3` 在本机不可用（实为 `ModuleNotFoundError: No module named 'tqdm'`，且从仓库根运行时 `tilelang/__init__.py` 会遮蔽安装包）；Stage 1 已在 DESIGN §4.7 登记该偏差，Stage 3 仍须自己再发现一次 | `DESIGN.md` §4.7；本轮实测 `/usr/local/python3.12.16/bin/python3 -c "import tilelang"` → `tqdm` 缺失 | 同 Stage 1 提案（在 `statectl platform` 的 `execution` 中按 task 记录实测可用解释器与 cwd），并补一条「禁止在仓库根做 API 静态查询/编译」的提示 | R |
| Develop-flow | review 的 warn「`BLK = align_up(NB,64)` 在 `B=1, numel→1023` 时可达 1024 lane，而 `T.SimtVF` 只验证过 32/64/256」在 develop skill 里没有对应的**落地形态建议**，实施者需自行决定改法（本轮改为固定 `IDS_LANES=64` + `ids_blk//64` 轮串行，避免开出未验证的线程宽度） | `REVIEW.md` 维度 4 warn；`_make_small_batch_kernel.py` §3 P5 段 | 在 develop skill 的「按平台规则生成 kernel」补一句：「`T.SimtVF(threads=)` 只用探针验证过的宽度（32/64/256）；lane 数需求超出时用『固定宽度 × serial 轮次』而不是加宽线程域」 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| GM→UB 的 `T.copy`（`asc_copy_gm2ub_align`）**把长度上取整到 32 B（8 个 int32）写入 UB**：n%8≠0 时 `n..ceil8(n)-1` 被 GM 越界读到的垃圾覆盖；而 **UB→GM 方向不越界写**（canary 探针：P=1/23/18 的尾块全部无溢出） | D | 探针输出：`n=1→7 个槽被覆盖, n=7→1, n=8→0, n=9→7, n=23→1`，值均 `117901063(0x07070707)`；本 kernel 因此把 `s_x` 加 32 元素 guard band 并用构建期 `tail` 作循环上界 | `cd examples/moe_permute_align/_make_small_batch_kernel && bash -c 'source /workspace/tilelang-mlir-ascend-950/.agents/platforms/toolchains/ascend/env-950pr.sh; ASCEND_RT_VISIBLE_DEVICES=0 /workspace/venv-ascend950/bin/python repro/TRAP-950-partial-gm2ub-load.py'`（断言：n%8==0 不溢出、否则溢出 `8-n%8` 个槽） | tilelang 0.1.15 + CANN 9.2.1 + Ascend950PR_9579（真机） | `.agents/platforms/toolchains/ascend/knowledge/PL-950-vector-execution.md`（尾块读写小节）或 `numerical-boundaries.md`（新 TRAP 条目） |
| **UB `int32` 原子「取旧值」在真机成立**：`T.atomic_add(s_cs[eid], 1, return_prev=True)` → `asc_atomic_add(&s_cs[eid], 1)`；1023 路由 / 32 游标争用下连跑 20 次（共 60+ 次 launch）**每次契约等价且逐位相同**；同一 kernel 全源码 `asc_syncthreads` 计数 = 0（P6 热循环无核内 barrier） | D | `_make_small_batch_kernel.py` §3 P6；降级码 `slot[0] = asc_atomic_add((&(buf_dyn_shmem[eid+2208])), 1)` 单次求值；`grep -c asc_syncthreads src_1023_32_64.cce` = 0；压力脚本 `/tmp/opencode/stress_timing.py`（(1023,32,64)/(1023,8,16)×{uniform,[7]*1023,[0]*1023} 各 20 次） | `cd examples/moe_permute_align/_make_small_batch_kernel && bash -c 'source …/env-950pr.sh; ASCEND_RT_VISIBLE_DEVICES=0 /workspace/venv-ascend950/bin/python _make_small_batch_kernel.py --level L0'`（L0-9 repeat=3、L0-14/B-7 全偏斜） | tilelang 0.1.15 + CANN 9.2.1 + Ascend950PR_9579（真机；Stage 1 只有编译+头文件证据） | `.agents/platforms/toolchains/ascend/knowledge/execution-patterns.md`（UB 原子小节：把「编译可用」升级为「真机语义已验证 + 与 GM 读同循环时零 barrier」） |
| 单 launch 融合小批 kernel 的**实测延时**（host 侧 warm 计时，含派发）：`(1023,32,64)` 24.5 µs、`(600,8,64)` 14.7 µs、`(200,8,16)` 7.1 µs、`(8,4,4)` 4.1 µs —— 与设计 §1.6.0 R4 的 4.4 µs 估算差 3–6×，差额主要为 host 派发/包装开销（设备内指令量按估算 ≈2.4 µs 量级） | D | 计时脚本输出（每 shape 200 次 warm launch 取均值）；`DESIGN.md` §1.6.0 R4 / §9.1 风险 5 | 同上 `--level all` 的 `launch=` 字段，或 `/tmp/opencode/stress_timing.py` 的 `launch pair mean` | tilelang 0.1.15 + CANN 9.2.1 + Ascend950PR_9579（真机） | `.agents/skills/tilelang-op-optimize/references/pattern-library/constants.md`（新增 `CONST-950-small-fused-launch`）→ Stage 4 baseline |
| 部分块热循环的**无守卫写法**：用构建期 `tail = m*UNROLL + r` 拆成「m 组 4 路累加链 + r 条 python 级尾语句」，代替「常量 padding + 固定上界」；既保住 UNROLL 的依赖链深度，又不依赖搬运的写入粒度（padding 会被部分块搬运覆盖） | P | `_make_small_batch_kernel.py` §3 `if tail > 0:` 段（`tail_groups` / `tail_rem`）；四种余数均由 L0 覆盖：r=0（L0-4, numel=32）、r=1（L0-13, numel=1）、r=2（L0-2, numel=2）、r=3（L0-16, numel=7） | `--level L0 --case 0,1,12,15`（4 个余数档） | tilelang 0.1.15 + CANN 9.2.1 + Ascend950PR_9579（真机） | `.agents/platforms/toolchains/ascend/knowledge/PL-950-vector-execution.md` |

### Transferable Lessons（可迁移教训）

- `@T.prim_func` body 里的 `for` / `if` **全部会被 eager builder trace 成 TIR 节点**：`for t in range(n)` 的 `t`
  是运行期表达式，任何 `min(t*a, c)` / `if a and b` 都会抛 `Cannot use and / or / not operator to Expr`。
  构建期常量（分块数、尾块长度、余数）必须先在 traced body 之外算成 python int，再作为循环上界/切片端点使用。
- GM→UB 的 `T.copy` 对**部分块**的写入长度上取整到 32 B（8 个 int32），会把你放在 `[n, chunk)` 的常量 padding
  覆盖成 GM 越界垃圾，且 copy 与 fill 写域不相交 ⇒ 无依赖边，编译器不会保证 fill 在后。结论：**不要用 padding
  被覆盖后可读的假设**——热循环上界取精确元素数，并在 UB 缓冲尾部留 ≥8 元素的 guard band（本 kernel 用 32）。
- UB→GM 方向的同类尾块搬运**不越界写**（canary 实测），所以输出面不必留 guard band；这与 GM→UB 方向相反，
  两个方向要分别实测，不能互推。
- `T.SimtVF(threads=)` 只用探针验证过的宽度（32/64/256）。lane 需求可能超出时（如 `BLK = align_up(NB,64)` 在
  `B=1,numel=1023` 时达 1024），改成「固定 64 lane × `serial(⌈BLK/64⌉)` 轮」，不要凭推测开更宽的线程域。
- Hillis–Steele 扫描的步进**不能用 `for d in range(5)` 再判 `d%2`**（`d` 是运行期表达式，`%2` 分支会被 trace 成
  运行期分支）。直接按步展开 5 段（数据在 buff 上半、下半为恒零 halo、两 buffer 交替），既避开 shared→shared
  copy，也让降级码可逐段核对。
- `T.alloc_local([1])` 接原子返回值（`slot[0] = T.atomic_add(...)`）是必要写法：若把原子表达式直接当索引，
  编译器加的值域守卫 `if (0<=slot && slot<P)` 会多次引用它，有重复求值风险。真机验证：单次 `asc_atomic_add` +
  20 次重跑逐位一致。
- 判据顺序建议（真机首跑）：① 先跑最小 shape 确认原子与编译链；② 再跑 `numel` 不被 8 整除的档位（本次正是这里
  暴露搬运上取整）；③ 最后跑最大档与偏斜档做争用/确定性验证。跳过 ② 会让「大 shape 全过、小 shape 静默错」的
  缺陷漏到后续层级。
