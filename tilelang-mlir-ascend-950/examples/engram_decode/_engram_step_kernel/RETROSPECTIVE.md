## Stage 1 (design) Retrospective — first_design — 2026-11-23

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Info-source | 平台 pattern-library 的实测常数全部绑定 Ascend910B2C（`CONST-vector-launch-overhead` / `CONST-mte2-degradation` 等），本任务是 Ascend950PR_9579，`cost_measurements` 为空，导致 §1.6.0 的 roofline 只能写「未实测假设」并削弱了 SIMT/SIMD 裁决的纸面判定力 | `examples/engram_decode/_engram_step_kernel/DESIGN.md`#§1.6.0 的「设计期估算」行；`.agents/skills/tilelang-op-optimize/references/pattern-library/constants.md` 的 front-matter `device: 910B2C` | 在 constants.md 增加「设备适用性」一列或按设备分节，让设计期能一眼判断常数可否外推 | R |
| Design-flow | skill 的资源分层表要求「轻量调研只读 algorithm-research §1–§4」，但本算子（纯 Vector 归约 + 逐元素，无矩阵）同时是「规约类」，按 §2 必须是**完整调研**；两条规则给出的深度不同，需自行取严 | `.agents/skills/tilelang-op-design/SKILL.md` §3 资源分层表 vs `.agents/skills/tilelang-op-design/references/algorithm-research.md` §2 | 在资源分层表补一句「按 algorithm-research §2 的调研深度分级取严」 | R |
| Constraint-doc | `T.Kernel` 内 Python 级 `if` 会被 eager builder 作为 `ctx_if` **两支都 trace**（不按布尔值剪枝），文档与 `frame.py` 均未记载 | `examples/engram_decode/_engram_step_kernel/history_version/design_probe_step.py`（w=1 变体的 `UnboundLocalError: tap_ub`）；DESIGN.md#§9.2 E5 | 在 `references/programming.md`「API 与执行区域」加一条：`@T.prim_func` 内的 Python `if` 不剪枝，结构分支须放在 prim_func 外或用零 extent 分配 | D |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 4 个并发 reducer epoch 可在同一 `T.Parallel` 内工作，编译为 4 条独立 `AscendAllReduce<SumOp, T, 1, 0>`（各自 UB scratch，偏移按 tile 递增） | D | `examples/engram_decode/_engram_step_kernel/DESIGN.md`#§11.1（S1 serving-b4，生成源码实读）；`history_version/design_probe_step.py` | `cd examples/engram_decode/_engram_step_kernel && ASCEND_RT_VISIBLE_DEVICES= /workspace/venv-ascend950/bin/python history_version/design_probe_step.py`（compile-only，7/7 通过） | tilelang ascend 0.1.15 + CANN 9.2.1；host 无卡编译（卡被占用），设备型号 Ascend950PR_9579 | `.agents/platforms/toolchains/ascend/knowledge/PL-950-row-primitives.md`（追加「多 reducer 并发」用例） |
| 零 extent 的 `T.alloc_shared((0, d_pad), dtype)` 在本工具链合法，可作为「该变体不需要此缓冲」的统一写法（避免 Python 级结构分支） | D | DESIGN.md#§9.2 E5 / §11.1 第六变体（w=1，5 156 字节源码） | 同上（S1 w=1 变体） | tilelang ascend 0.1.15 + CANN 9.2.1 / 950PR 无卡编译 | `references/programming.md`（分配接口的边界值说明） |
| 舍入路径单项隔离验证法：对每个数学等价优化项各跑「只开该项 / 只关该项」两版，其余项固定，才能把 ulp 差异归因到单项（否则多项叠加后无法区分） | P | DESIGN.md#§1.6.1 的验证说明与结果表（8 用例 × 6 采纳项 = 48 cell） | `cd examples/engram_decode/_engram_step_kernel && /workspace/venv-ascend950/bin/python verify_equiv.py` | torch 2.13.0+cpu（无设备依赖） | `.agents/skills/_shared/standards/`（D-1 验证协议补充「单项隔离」要求） |
| 950DT 已验证实现把 `kv → dtype → fp32` 往返再归一化，与本仓 torch 参考（直接 fp32）**语义不同**：实测写出 dtype 值仅 0.39–76 % 逐位一致、最大绝对偏差 1.56e-02 | D | DESIGN.md#§1.6.1 的 B1 行（informational 表）；`/workspace/TileOPs-ascend950/src/tileops/kernels/ascend/sequence_modeling/decode.py` 第 220、247 行 | `verify_equiv.py`（B1 项：`run_candidate(..., {"B1"})` vs 不带 B1） | torch 2.13.0+cpu | `.agents/platforms/toolchains/ascend/knowledge/migration-diagnostics.md`（参考实现差异需逐项核对语义，不默认照抄） |
| latency 型归约算子（batch=1–8、每核 1 行）不应为「提高核占用」而拆 `d`：每行计算 ≈0.02 µs，拆行要么付跨 block barrier（未认证）要么 +1 launch（≈5 µs） | P | DESIGN.md#§5.5 ③ 与 §1.6.0 R3 候选 4 | none（依据为 §1.6.0 的记账，非实测） | — | `pattern-library / bottleneck-patterns` |

### Transferable Lessons（可迁移教训）

- `@T.prim_func` 体内的 Python `if <bool>:` **不是编译期剪枝**：eager builder 会把两支都 trace，
  未分配的名字在另一支里会 `UnboundLocalError`。要按编译期常量分结构，要么把分支放在 prim_func 之外
  （工厂函数里选不同的 prim_func），要么用零 extent 分配 + 零次迭代循环统一形状。
- `T.cast(x, "float32") ** 2` 不被 eager builder 接受（`pow()` 只支持 PrimExpr 底数），平方一律写 `x * x`。
- 用 `T.alloc_reducer` 的算子，reducer 的跨 lane 中转在 UB 里占 `tile × 4 B` **每个 reducer**；
  并发 N 个 reducer 就要在 UB 预算里加 N 份（4 个 reducer + `d_pad=512` = 8 KiB，占 248 KiB 的 3 %，
  但 tile 更大时必须重算）。
- 「每行一个核」的算子里，`T.SimtVF(threads=)` 的线程数与每 lane 元素数是**成反比**的一对：
  `threads = tile` 时归约树最矮、每 lane 1 项；`threads = tile/4` 时生成 `#pragma unroll` 的 4 项串行累加。
  两者都正确，但累加分组不同——做数值对比时必须固定这一档，否则 ulp 差异无法归因。
- 迁移带「前一个 launch 产出的中间量」的算子时，先确认**中间量的 dtype 与归一化时机**是否与参考实现一致：
  「先 cast 到输入 dtype 再归一化」与「直接对 fp32 归一化」在本任务上差 1.6e-2（超过 fp16 的 1e-5 量级
  四个数量级），而且会静默通过 0.05 的宽松容差。

## Stage 2 (review) Retrospective — first_review — 2026-11-23

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Review-flow | 检视清单没有「核对 DESIGN 中声明的 API 行为事实」这一项：本次 §9.2 E5 断言「`@T.prim_func` 内 Python `if` 两支都 trace → `UnboundLocalError`」，与实读的 `language/eager/builder.py:449-474` 及四变体实测矛盾，而维度 1/4 只要求核对"设计用到的 API 是否存在"，不要求复核"设计声称的 API 失败模式" | `examples/engram_decode/_engram_step_kernel/REVIEW.md`#O-1-a；DESIGN.md#§9.2 E5；`/workspace/venv-ascend950/lib/python3.12/site-packages/tilelang/language/eager/builder.py:449` | 在 tilelang-design-review SKILL.md Phase 1 增加一步：对 §9.2 / §11.1 中可十行复现的 API 行为断言（结构性 Python `if`、运算符支持、零 extent 分配）亲自复现，不符者记建议级并给最小复现 | R |
| Review-flow | D-5 的 `design_calc_check.py` 在本 DESIGN 上 `ub_budget` skip（§4.5 是 5 列「Buffer/Shape/dtype/版本数/占用」表，解析器未识别），因此 §4.5 手工合计少 1024 B 这类加总错误没有被机械拦截，只能靠检视者逐行相加 | `examples/engram_decode/_engram_step_kernel/REVIEW.md`#O-2-a；DESIGN.md#§4.5；`python3 .agents/tools/design_calc_check.py --design …/DESIGN.md` 输出 `ub_budget: skip` | 让 `design_calc_check.py` 支持该 5 列表头（或退化为"表内占用列求和 vs 合计行"比对），skip 时在输出里提示需要手工加总 | R |
| Review-flow | D-1 的机械判据只有「执行结果与内嵌表一致 + 全 EQUIV_PASS」，无法发现脚本里某个采纳项的开关是**死代码**（本次 A4 的 item flag 未被 `run_candidate` 消费 → 自比较产生平凡 0 ulp，"48 cell 全实测"的覆盖度被高估） | `examples/engram_decode/_engram_step_kernel/REVIEW.md`#O-8-a；`verify_equiv.py` 的 `run_candidate`（无 `"A4"` 分支） | 在 Phase 1 第 7 步加一句：抽查每个采纳项的 item flag 是否真的改变被比较的表达式；退化项改按解析恒等复核并在维度 8 标注 | R |
| Info-source | 维度 8 要求「亲和性淘汰引用匹配型号和版本的常数条目」，但平台 `cost_measurements` 为空、pattern-library 常数全部绑定 910B2C（`CONST-vector-launch-overhead`），检视只能确认设计"把未实测量标注为假设"合规，**无法**独立确认 `AscendAllReduce`/`vcadd` 的代价倍数，因此 §1.6.3 的 0.90 阈值只能核"三件套齐全"而非"阈值合理" | 平台 `statectl platform --task-dir …` 的 `cost_measurements: []`；DESIGN.md#§1.6.3 (iii)、#§9.1 | pattern-library/constants.md 按设备分节或加"可外推性"列，使维度 8 能判断哪些常数可跨型号使用 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| `@T.prim_func` 体内 `if <Python bool>:` **是**编译期剪枝（`ctx_if` 对非 PrimExpr 条件 yield 后由 `ctx_then/ctx_else` 按真值择一执行），与 Stage 1 记录的"两支都 trace"相反；四变体实测 w=1 的 `if w>1` 版与零 extent 版源码长度同为 1168 B | D | `examples/engram_decode/_engram_step_kernel/REVIEW.md`#O-1-a；`tilelang/language/eager/builder.py:449-474` | none（脚本未落盘；重建方式见 REVIEW.md#O-1-a：prim_func 内用 `if w > 1:` 包住 `tap_ub` 分配与 `for p in range(w-1)` 拷贝，w∈{1,4} 各编一次，`ASCEND_RT_VISIBLE_DEVICES=""` compile-only） | tilelang ascend 0.1.15 + CANN 9.2.1（venv site-packages）/ Ascend950PR_9579 无卡编译 | `.agents/platforms/toolchains/ascend/references/programming.md`（纠正"Python if 不剪枝"的说法；Stage 1 retro 的同一断言需一并修） |
| 文档级 UB 预算可用 `get_kernel_source()` 的**最大 UB 偏移**机械反查：`((half*)buf_dyn_shmem)[11776] + 512` = 24 576 B 与逐行手工加总一致，可发现表内合计少算 1024 B 的口径错误 | P | `examples/engram_decode/_engram_step_kernel/REVIEW.md`#机械复核 与 #O-2-a | `cd examples/engram_decode/_engram_step_kernel && ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python -c "…build_s1(…).get_kernel_source()"`（读最大下标） | tilelang ascend 0.1.15 + CANN 9.2.1 / 无卡编译 | `.agents/skills/_shared/standards/hardware-cost-model.md`（容量项的复核方法：优先用生成源码偏移而非手算） |
| 等价性脚本的「单项隔离」可能退化为自比较：若候选构造器没消费该项开关，cand 与 baseline 是同一个函数，0 ulp 无信息量。检视时必须读脚本确认开关真的改变表达式 | P | `examples/engram_decode/_engram_step_kernel/REVIEW.md`#O-8-a；`verify_equiv.py#run_candidate` | `cd examples/engram_decode/_engram_step_kernel && grep -n '"A4"' verify_equiv.py`（无命中即退化） | torch 2.13.0+cpu | `.agents/skills/_shared/standards/`（D-1 验证协议补「开关有效性抽查」） |

### Transferable Lessons（可迁移教训）

- 复核 D-1 等价性脚本时，除了看"全 PASS"，还要确认每个采纳项的开关真的改变被比较的表达式：否则该项的 0 ulp
  只是 `f(x)` 与 `f(x)` 的自比较（本任务的 A4 即是），结论要改用解析恒等论证。
- 文档声明的 API 失败模式（"这里会报 UnboundLocalError"）与 API 存在性同等重要：十行脚本就能复现的断言要亲自跑，
  本次发现 eager builder 对 Python `if` 是**剪枝**的，与设计文档记载相反。
- 需要报 UB 预算时，除逐行加总外，用 `get_kernel_source()` 里的最大 UB 下标反查一次（`下标 × sizeof + 尾长`），
  两条路径一致才写进文档；本任务两条路径差 1024 B 的那个数字不该进正式表格。
- 迁移任务里"中间量的归一化时机"要同时对两份文档核对：本任务的 project 段设计文档残留了一句"step 会把 kv 舍回
  输入 dtype"，与 step 段设计的结论相反——跨文档的这类残句会误导 Stage 3，发现即记录给 conductor。
- 卡被其它 Stage 占用时，设计的物理核数依据可以引用平台 `statectl platform` 已落库的 `hardware.core_source`
  （含查询代码与返回值），并补一次 CPU 侧 INI 复核；这足以满足分核三要素的"实查记录"，不需要争抢设备。

## Stage 3 (develop) Retrospective — first_impl / attempt 1 — 2026-11-23

Toolchain: tilelang ascend 0.1.15 + CANN 9.2.1, Ascend950PR_9579 (dav-3510, 28 Cube / 56 Vector)
real device, card 0.  Result: 27 cases (13 L0 / 7 L1 / 2 L2 / 5 boundary), every case launched
3 times bit-identically, `max_y_diff = 7.6e-06`, `max_state_diff = 1.95e-03` (1 ulp of fp16 at
magnitude 2), verdict `[PRECISION_PASS]`.

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Info-source | Platform `execution.precision_command` records the interpreter `/usr/local/python3.12.16/bin/python3`, which cannot import tilelang at all (`ModuleNotFoundError: No module named 'tqdm'`); only `/workspace/venv-ascend950/bin/python` works. The ruleset says to follow `execution`, so Stage 3 had to override it from the conductor prompt | `statectl platform --task-dir examples/engram_decode/_engram_step_kernel` -> `execution.precision_command`; measured: that interpreter fails on `import tilelang`, venv succeeds | Validate `execution.*_command` at platform-record time by running `import tilelang` with that exact interpreter, and fall back to the toolchain venv otherwise | R |
| Develop-flow | Phase 4's test list never says the *test shapes* must satisfy the upstream shape rule. A "cache-length boundary" case with `max_conv_len=7 < dilation*(w-1)=9` silently produced a wrong answer (the kernel has no tap guard by design) and was within the fp16 tolerance, so it would have passed a careless suite | `repro/TRAP-950-step-tap-guard.py`; first version of `_run_nonfinite` in `_engram_step_kernel.py` (mcl=7) | Add to develop Phase 4: extract every precondition/shape rule from DESIGN and assert it in the case runner before launching | R |
| Develop-flow | A frozen DESIGN can carry factually wrong API claims that Stage 2 already refuted (DESIGN 9.2 E5 still says a Python `if` inside `@T.prim_func` traces both branches). Only reading REVIEW's 11 advisory items gives the correct form; following DESIGN alone would have avoided the legal compile-time branch | `REVIEW.md#O-1-a` vs `DESIGN.md#§9.2 E5` and `#§11.1` | In develop Phase 1: treat advisory items that correct a fact about the toolchain as errata to DESIGN and follow them first | R |
| Info-source | `PL-950-row-primitives` is verified on the 950DT *simulator* with 64 threads only; this task needed 256/512 lanes on 950PR hardware, so its "verified" status did not cover the production configuration and Stage 3 had to re-establish it | `PL-950-row-primitives.md` front-matter (`hardware_id: ascend-950dt-9582`, 64 threads) vs `_engram_step_kernel.py#default_config` (256/512) | Record the covered lane widths / hardware in the entry's verification-scope table so the gap is visible before Stage 3 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| A row scalar carried in a `(1,)` fragment written inside `T.Parallel(1)` is emitted as per-lane *unguarded* code inside `SimtVF`, so it can be broadcast straight into the next elementwise pass -- no UB staging, no `mem_bar`. Bit identical to the inline-PrimExpr form (4x256 RMSNorm, 2.38e-07 vs fp32) | D | `examples/engram_decode/_engram_step_kernel/repro/PL-950-simt-scalar-broadcast.py`; generated VF body of `_engram_step_kernel.py` (`alpha[0]`, `rv[0]`, no `threadIdx.x == 0` guard) | `cd examples/engram_decode/_engram_step_kernel && ASCEND_RT_VISIBLE_DEVICES=0 /workspace/venv-ascend950/bin/python repro/PL-950-simt-scalar-broadcast.py` -> `RESULT: scalar broadcast via T.Parallel(1) is per-lane and needs no barrier` | tilelang ascend 0.1.15 + CANN 9.2.1 / Ascend950PR_9579 device | `.agents/platforms/toolchains/ascend/knowledge/PL-950-row-primitives.md` (add "row scalar into the next pass") |
| Four concurrent reducer epochs (4 `reducer_update` in ONE `T.Parallel`) at 256 and 512 lanes on 950PR hardware are numerically sound: 27 cases pass, `max_y_diff=7.6e-06`, `max_state_diff=1.95e-03` (=1 ulp at 2.0), and each case is bit-identical across 3 launches -- the SIMT reducer + `T.Parallel` UB-write path needs **no** `mem_bar("VST_VLD")` (unlike the SIMD `vsts -> vld` trap) | D | `_engram_step_kernel.py#run_L0/#run_L1`; generated source has 4x `AscendAllReduce<tl::SumOp, 512, 1, 0>` with scratch at 2816/3328/3840/4352 floats, UB peak 24 576 B (matches DESIGN 4.5) | `cd examples/engram_decode/_engram_step_kernel && ASCEND_RT_VISIBLE_DEVICES=0 /workspace/venv-ascend950/bin/python _engram_step_kernel.py --level all` | tilelang ascend 0.1.15 + CANN 9.2.1 / Ascend950PR_9579 device | `.agents/platforms/toolchains/ascend/knowledge/PL-950-row-primitives.md` (extend the hardware/lane scope) |
| An illegal `max_conv_len < dilation*(conv_kernel_size-1)` makes a guard-free tap read row `-2`: no error, batch 0 reads before the tensor (NaN output) and batch 1 is off by 3.5e-03 -- **below** the fp16 tolerance 0.05, i.e. a silent semantic error. A witness run proves the mechanism exactly (feeding that row where tap 0 lands reproduces the kernel to 0.0) | D | `examples/engram_decode/_engram_step_kernel/repro/TRAP-950-step-tap-guard.py` | `cd examples/engram_decode/_engram_step_kernel && ASCEND_RT_VISIBLE_DEVICES=0 /workspace/venv-ascend950/bin/python repro/TRAP-950-step-tap-guard.py` -> `RESULT: TRAP CONFIRMED` | tilelang ascend 0.1.15 + CANN 9.2.1 / Ascend950PR_9579 device | `.agents/platforms/toolchains/ascend/references/migration-diagnostics.md` (shape-rule preconditions hide silent errors) |
| The masked trailer path is exact on device for a non-power-of-two padded width: `d=600 -> d_padded=768, tile=1024` gives `y_diff=0` against the golden at fp16, bf16 and both `threads in {256, 512}`; the tail lanes' `if_then_else` read past the row is harmless because it is masked and the write-back is guarded | D | `_engram_step_kernel.py#run_L0` (`L0-d600-masked-tail`) and `#run_boundary` (`Boundary-d600-bf16`, `Boundary-d600-threads256`) | `... _engram_step_kernel.py --level boundary` | tilelang ascend 0.1.15 + CANN 9.2.1 / Ascend950PR_9579 device | `.agents/platforms/toolchains/ascend/knowledge/numerical-boundaries.md` (masked trailer form) |
| Splitting a two-launch operator for per-launch tests fixes the assertion set: the step launch must be judged on `y_t` + `new_conv_state[:, mcl-1, :]` + "every other row is bit-identical to a prefill", never against the golden's shifted 12 rows (that is launch 1's write set) | P | `_engram_step_kernel.py#_run_case` (assertions 1-4); DESIGN-review item `O-6-a` | `... _engram_step_kernel.py --level L0` (sentinels in every case) | tilelang ascend 0.1.15 + CANN 9.2.1 / Ascend950PR_9579 device | `.agents/skills/tilelang-op-develop/SKILL.md` (per-launch assertion split) |

### Transferable Lessons（可迁移教训）

- Engram decode is two launches: the step launch writes only `new_conv_state[:, max_conv_len-1, :]`. A segmented test must compare `y_t` and that one row against the reference and compare all other rows against a prefill bit-for-bit; comparing them against the golden's shifted history is a guaranteed false failure (the shift belongs to the project launch).
- Before authoring test shapes, extract the upstream shape rules and assert them in the case runner. `max_conv_len >= dilation*(conv_kernel_size-1)` is what lets the step kernel drop its tap-index guard; break it and the kernel reads a negative cache row with a deviation of only ~3e-3 (inside the fp16 tolerance 0.05), so nothing complains.
- A row scalar (`alpha`, `rv`, `rrms`) can live in a `(1,)` fragment written inside `T.Parallel(1)`: inside `SimtVF` that loop is emitted unguarded in every lane, so it broadcasts for free and is bit identical to the inline-expression form. Do not add UB staging or `T.simd.mem_bar` for it.
- Kernels that only write UB from `T.Parallel` and then leave through `T.copy` need no explicit barrier (the compiler emits `notify(PIPE_V, PIPE_MTE3)`/`wait`); the `mem_bar("VST_VLD")` rule is specific to SIMD `vsts -> vld` readbacks. Still verify by launching one compiled binary three times and requiring bitwise equality.
- When a migrated reference implementation and the frozen DESIGN disagree on a semantics detail (here: whether `kv` is rounded back to the input dtype before the norms), the sampling of L0 must include a case that would expose the difference; the round trip deviates by 1.6e-02 and would still pass a 0.05 tolerance, so tolerance alone cannot arbitrate.

## Stage 3 (develop) Retrospective — gate fix, attempt 1 re-run — 2026-11-23

Gate `S3-GOLDEN-TORCH` rejected the first delivery: the `golden_*` body only delegated to
`engram_decode_step_torch`, so an AST check for a direct `torch.*` call in that function saw
no torch API. Fixed by re-implementing the same semantics with torch ops inside
`golden_engram_decode` (`torch.matmul`, `torch.mean`/`torch.square`/`torch.rsqrt`,
`torch.sigmoid`, `torch.sum`, `F.pad`/`F.silu`, `torch.cat`, `torch.zeros`); the repository
reference is now a **cross-check assertion in every case** instead of the implementation.
All 27 cases were re-run on device: identical numbers to the pre-fix run (the two references
are bit identical), `[PRECISION_PASS]`, and `statectl gate 3` returns `pass: true`.

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Develop-flow | DESIGN 8.1 froze "call the source reference directly" as the golden, but gate S3-GOLDEN-TORCH requires a direct `torch.*` call inside the `golden_*` body, so the frozen instruction cannot satisfy the gate as written. The cheapest compliant form is to re-implement the reference's op sequence and assert bitwise parity against it -- which the design skill should prescribe, since it also keeps the golden independent | `statectl gate 3 --dir examples/engram_decode/_engram_step_kernel` (first run: `S3-GOLDEN-TORCH` failure); `DESIGN.md#§8.1`; `_engram_step_kernel.py#golden_engram_decode` | In `tilelang-op-design` 8.1 (and the develop skill's Phase 3): a golden may not be a pure delegation to another reference; require "direct torch implementation + bitwise parity assertion against the reference" and say so in advance | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| A direct-torch golden can be bit-identical to the source repository's reference implementation when the op sequence is replicated exactly (same matmul, `.mean(-1).add(eps).rsqrt()`, `torch.sigmoid`, `F.pad`->fp32 view, per-tap fp32 multiply-add, `F.silu` + residual, `torch.cat` append): 6 shapes covering fp16/bf16, partial cache, `w=1/mcl=1`, `d=600` and `d=1` all compare equal on `int16` views | D | `_engram_step_kernel.py#golden_engram_decode` + the parity assertion in `_run_case` (runs for all 27 cases) | `cd examples/engram_decode/_engram_step_kernel && ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python -c "import _engram_step_kernel as M; ..."` -> parity `True`; full suite `--level all` -> `[PRECISION_PASS]` | torch CPU (no device needed for the golden itself); device run: tilelang ascend 0.1.15 + CANN 9.2.1 / Ascend950PR_9579 | `.agents/skills/tilelang-op-develop/SKILL.md` (Phase 3 golden recipe) |

### Transferable Lessons（可迁移教训）

- A `golden_*` function must contain torch operations in its own body: delegating to another
  module's reference fails gate S3-GOLDEN-TORCH. Re-implement the reference's exact op
  sequence with `torch.*` calls and keep the reference as a bitwise parity assertion per test
  case -- that satisfies the gate and leaves the golden independent.
