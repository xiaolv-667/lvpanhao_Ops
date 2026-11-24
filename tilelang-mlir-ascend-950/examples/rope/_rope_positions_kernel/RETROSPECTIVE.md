# RETROSPECTIVE — rope / `_rope_positions_kernel`

## Stage 1 (design) Retrospective — first_design — 2026-11-23

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Design-flow | `design_calc_check.py` 的 `core_split` 只认单一输出网格（`ceil(M/bm) × ceil(N/bn)`），而本类算子一个 launch 内服务两个张量（query + key），声明「合计启动块数」会被判 fail；需把「逻辑核数」写成单侧值、合计另述才能自洽 | `examples/rope/_rope_positions_kernel/DESIGN.md` §5.5 与 `design_calc_check.py` 的 `check_core_split`（recompute 4096 vs declared 4096，合计 5120 另述） | 在 `core-split-strategy.md` §1 要素①补一句「多张量单 launch 时按单侧网格声明逻辑核数，合计块数单列」 | R |
| Design-flow | 迁移任务的 `_STAGED` / `wide` 这类源侧启发式入参在 NPU 无等价设备事实，skill 未给「保留签名但两值编译同一程序」这一标准处置措辞，设计期需自行给出并论证 | `DESIGN.md` §0.6 R6、§9 R7、§8.3 L0-9 | `migration-analysis.md` §6.2 增加「启发式入参的接口保留处置」条目 | R |
| Info-source | 平台知识库对 rope 族零条目（`kb_search "rope rotary position embedding"` 返回空），`examples/` 下唯一的 RoPE 实现 `examples/posembedding/rope_llama31` 属另一后端方言（`tilelang.language` / `T.Kernel(is_npu=True)` / `T.alloc_ub`），不能作为本工具链 API 依据；设计期必须在「结构参考」与「API 证据」之间显式分栏 | `DESIGN.md` §3.5.3、§11.4；`examples/posembedding/rope_llama31/kernels/table_fp32.py`（import 行） | 在 `info-sources.md` 增一条「同族但异后端的实现只能作数学/结构参考」的判例 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| rope 的 fp32「单次舍入」契约：原生 dtype 逐 op 舍入在 fp16 上 1760/6144 位差、bf16 1584/6144（最大 14897 ulp），但容差违反 0——位级契约与容差契约必须分开判定 | D | `examples/rope/_rope_positions_kernel/DESIGN.md` §1.6.1 / §11.1（X1 行） | `python examples/rope/_rope_positions_kernel/verify_equiv.py`（torch CPU） | tilelang 0.1.15 / CANN 9.2.1 / 无设备（CPU 数值实验） | pattern-library / cases.md |
| rope 表流量：逐 head 读表比「每行读一次」多 6.4×（llama prefill 33.6 MB vs 5.2 MB，占算子总流量 29% vs 6%），是 RoPE 迁移最容易漏掉的流量项 | D | `DESIGN.md` §1.6.0 R3 两张行（#0 117.5 MB vs #1 89.3 MB） | 同上（§1.6.0 的算术在 `calc` 口径下可复算） | tilelang 0.1.15 / CANN 9.2.1 | pattern-library / elementwise.md |
| RoPE 的 GPT-J（相邻配对）在 SIMT `T.Parallel` 下是「跨 lane 步长 2」访问，是 SIMD 去交织（`T.simd.vld2(dist="DINTLV_B16")`）唯一可能的收益场景；注意本仓无任何 `T.simd.*` 算子级用例，该结论当前只是候选 | P | `DESIGN.md` §1.6.3 候选 #5/#6 + 实验裁决计划；安装包 `tilelang/ascend/language/simd.py::vld2` docstring | 需 Stage 4 真机 A/B（`perf_opt/`），当前无 repro | tilelang 0.1.15（仅静态源码） | pattern-library / layout.md（待 Stage 4 证据回填） |
| 分核声明陷阱：多张量单 launch 的算子（query+key）若把「合计块数」写成逻辑核数，`design_calc_check` 会判 fail；正确写法是单侧网格 + 合计单列 | R | `examples/rope/_rope_positions_kernel/DESIGN.md` §5.5 | `python3 .agents/tools/design_calc_check.py --design examples/rope/_rope_positions_kernel/DESIGN.md --task-dir examples/rope/_rope_positions_kernel` | — | `.agents/skills/_shared/standards/core-split-strategy.md` |

### Transferable Lessons（可迁移教训）

- **表驱动的逐元素算子先算「表复用比」再谈分块**：若每个 head（或每列块）都重读一次小表，表流量会与数据流量同量级；把表按行载入 UB 并让块内所有 head 复用，是零风险、机器可验证（逐位等价）的流量削减。
- **「位级等价」与「容差内等价」要分开主张**：`verify_equiv.py` 的 adopted 项按位级判（0 失配），golden-vs-fp64 的 G1 项按容差判——混用会把 fp32 的 1 ulp 正常差异误判成失败，也会把真正的语义漂移放过去。
- **源侧启发式入参（`staged`/`wide`/`units`）在目标平台可能无设备事实可依**：保留签名以兼容冻结 wrapper，但要显式写出「两值编译同一程序」并加一条接口断言用例，否则会被检视为静默丢弃。
- **相邻配对（interleaved）与前后半配对（neox）的向量化代价不同**：前者在 SIMT 下是跨 lane 步长 2 访问，后者是两段连续流；同一 kernel 里要分别评估，不能用一个结论覆盖两种约定。
- **`T.Kernel` 无 `threads`、只有 1-D grid** 是本后端的硬事实：迁移任何带 `threads=` 或多维 grid 的 GPU kernel，第一件事是把 grid/thread 折叠成 `bx` + `T.SimtVF(threads=…)`，否则设计无法落地。
- **动态 token 数 + 行谓词** 是本后端已验证的组合（`paged_cache_write` 同平台实现）；但「条件 `T.alloc` 非法」意味着所有 UB 缓冲必须在块级分支之前按最大形状分配。

## Stage 2 (design review) Retrospective — first_review — 2026-11-23

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Review-flow | 本 skill 没有「内嵌结果表 vs 脚本输出逐行 diff（含**被否决行**）」的机械位：D-1 只要求「执行结果与 §1.6.1 内嵌表一致」，人工若只核 adopted 行就会漏掉 rejected 行里的单元格错误。本次 X3 行的「最大 ulp」写 `≤3.4e4`，重跑实测 fp32 为 `2158330937`（16-bit 一侧 32935–34092 才与 3.4e4 吻合） | `examples/rope/_rope_positions_kernel/DESIGN.md` §1.6.1 X3 行 vs `verify_equiv.py` 输出 `X3 … float32 … 2158330937` | 在 §3 维度 8「与下游章节一致」前补一条：**内嵌表逐单元格 diff（adopted + rejected，ulp 列一并核）**；或让 `verify_equiv.py` 输出 JSON 供比对 | R |
| Review-flow | 维度 0 的「优化手段覆盖完整性」目前只能靠读源码比对，skill 未给「表读/表复用类断言」的核对提示。本设计的 §0.4 #4 把「表行每 block 读一次」记为源已有优化，而源的 `table_slots` 在 `move_unit` 内按**单元（head×slot）** 调用 ⇒ 表被每个 head 各读一次；该错误同时与同文 A1/R3 自相矛盾 | 源 `_kernels/rope.py` L262（`table_slots(..., first, ...)` 在 `move_unit` 内）、L353–357（staged 同）；`DESIGN.md` §0.4 #4 vs §1.6.1 A1 | 在 `migration-analysis.md` 三问解读加一条「性能性断言必须落到源码语句位置」：凡「每 block/每行读一次」类断言，须给出源语句行号并区分「scaler 读一次」与「向量/表行读一次」 | R |
| Review-flow | 「冻结接口事实核对」无门禁位：设计对 Stage 0 脚手架代码的断言（§9 R7「`_KERNEL_MAP` 只调度两个 `_STAGED=False` 类」）未被要求对照该文件核实；实际 map 声明 4 条，且 `_get_kernel` 取 `kernel_map["neox"]` 与 map 键 `rope_neox_positions` 不匹配（首调 KeyError） | `examples/TileOPs/tileops/ops/rope/rope.py` L68–73、L196–199；`ops/op_base.py` L53–80；`DESIGN.md` §9 R7 | 在检视清单加一项：**设计对 `examples/TileOPs` 冻结代码的事实性断言逐条 Read 核实**，并把「设计已预警的接口缺陷」与「设计自身的缺陷」分开路由 | R |
| Tool-usage | `kb_search.py` 不接受裸 query（要求 `--toolchain` 或 `--task-dir`），设计文档里记录的「`kb_search "rope …"` 返回空命中」无法照抄复现；复核者需自行补参数 | `python3 .agents/tools/kb_search.py "rope rotary position embedding"` → `error: select a toolchain explicitly or supply an existing task directory`；加 `--task-dir` 后 `no entries matched` | 在 skill 维度 8「负向断言复核」段写明命令形态 `kb_search.py --task-dir <task> "<query>"` | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 表流量「逐 head 读」的算术核对法：`(Hq+Hk) × R × itemsize × tokens` 与「每行一次」的 `rows × chunks × R × itemsize` 必须分开算；本次设计的基线只用了 `Hq` 漏掉 `Hk`（33.6 MB vs 应为 41.9 MB），但淘汰方向不变（1.32× → 1.41×） | P | `DESIGN.md` §1.6.0 R3 基线行 / §1.6.1 A1 收益栏；重算 `4096×40×128×2 = 41.9 MB` | 纯算术，可由 `design_calc_check` 的 R3 口径扩展；当前无脚本 | tilelang 0.1.15 / CANN 9.2.1（无设备） | `.agents/skills/tilelang-op-design/references/algorithm-research.md`（R3 复算示例） |
| 「十进制 KB / 二进制 KiB 混用」是 UB 风险核算的常见失准形态：`26 B/elem × 4096 = 106,496 B` 被写成「106.5 KB」（十进制），而同文档 §4.5 表用 KiB（34.0 KB = 34,832 B）——`design_calc_check` 的 ub_budget 只解析「合计」行，prose 里的膨胀估算不入检查，需人工核单位 | P | `DESIGN.md` §4.5 正文「16-bit 81.9 KB / fp32 106.5 KB」vs 表格 KiB；`design_calc_check` 输出 `UB: 5 buffer entries sum = 34.0 KB` | `python3 .agents/tools/design_calc_check.py --design … --task-dir …`（不覆盖 prose） | tilelang 0.1.15 | 本 skill（维度 2 检视要点）+ `design_calc_check.py` SIZE_RE 支持 `KiB` 区分 |
| 冻结 wrapper 的 `default_config` 与设计 Tiling 的耦合：`units_per_block` 的**语义**被设计重定义（wrapper 注释「threads*units」→ 设计「每块 pair 预算」）时，`bt`/块数/分核/roofline 全部随之变；本设计 §5 采用 2048 而冻结值为 128/512，二者给出的 `bt` 为 4 vs 1（llama） | R | `examples/TileOPs/tileops/kernels/rope/rope/rope.py` L189–200；`DESIGN.md` §5.2/§10.2 | 复算：`bt = max(1, min(upb//(hpc*half), 4096//cols))`，`upb=2048 ⇒ 4`，`upb=128 ⇒ 1` | — | 本 skill（维度 3 检视要点：配置参数语义与冻结 wrapper 默认值对账） |

### Transferable Lessons（可迁移教训）

- **检视要「读两份代码」**：DESIGN.md 的迁移章节 + 源算子 + （本平台特有的）Stage 0 冻结脚手架代码。只读源算子会漏掉「设计对冻结接口的事实断言」类问题（本次 `_KERNEL_MAP`/键不匹配）。
- **机械复核的盲区要么记「人工已核」，要么补脚本**：`core_split` 只算单侧网格、`ub_budget` 只算表格合计、D-1 只比 adopted 行——三处都在本次检视中需要人工补算，写成「机械跳过，人工已核 + 具体数字」比笼统写「已核」更有用。
- **负向断言的复核成本低但价值高**：`kb_search`（空命中）、`algorithm-research.md` §5（无 rope 行）、`examples/` grep `T.simd.*`（0 处）、`PL-950-simd-reduce` 的 dtype 边界（「只覆盖 FP32」）四条各一次命令即可确认设计的关键否决前提，属于每次都要做的固定动作。
- **数字型错误要区分「影响决策」与「只影响叙述」**：基线表流量少算 20%（漏 `Hk`）使被淘汰候选看起来更好，决策方向不变 ⇒ 记建议级；若少算的是选定候选，则必须阻塞。这一判据值得在检视时显式写出。

## Stage 3 (develop) Retrospective — first_impl — 2026-11-23

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Develop-flow | 平台 `execution.precision_command` 模板的解释器路径（`/usr/local/python3.12.16/bin/python3`）在本站点存在但**无 torch**，直接照抄会得到 `torch` ImportError；conductor 已在调度 prompt 里改正为 `/workspace/venv-ascend950/bin/python`，但平台配置本身仍是错的，后续尝试若只读 `statectl platform` 会再次踩坑 | `python3 .agents/tools/statectl.py platform --task-dir examples/rope/_rope_positions_kernel` → `execution.precision_command[0]`；实测 `/usr/local/python3.12.16/bin/python3 -c "import torch"` → `ModuleNotFoundError: No module named 'torch'`，`/workspace/venv-ascend950/bin/python` 为 `torch 2.13.0+cpu / tilelang 0.1.15` | 把平台配置 `execution` 的解释器改为 `$TILEOPS_950PR_VENV/bin/python`（或让 `env-950pr.sh` 导出 `TILEOPS_950PR_PY` 并由 execution 模板引用），否则每次 Stage 3/4 都要人工纠正 | R |
| Develop-flow | 冻结 wrapper（`kernels/rope/rope/rope.py`）的 `default_config` 已是 `{"units_per_block": 2048}`——REVIEW S1 担心的「2048 vs 128/512」在 Stage 0 现态中**并不存在**，S1 是用旧值检视的；设计 §5.2 的 `bt=4` 与之自洽（本次实测 6 条 workload 的 `bt/hpc/cols/nch/tile/块数` 与设计表逐个吻合） | `examples/TileOPs/tileops/kernels/rope/rope/rope.py` L189–200（`default_config` 返回 2048）vs `REVIEW.md` S1；本文件 Stage 3 的几何核算（见 Value Point `D-3`） | 在 review skill 维度 3 的 S1 类条目里补一句「冻结 wrapper 的 `default_config` 必须现读，不能引用 Stage 1 的转录值」 | R |
| Develop-flow | 迁移 harness 的 `{op}.py` 需要覆盖「同一 factory 的 3 种 `out_idx` 变体（query-only `[4]` / key `[5,6]` / offsets `[6,7]`）」，但 skill 的 `{op}.py` 结构与模板只演示单 `out_idx`；变体选择必须与冻结 wrapper 的 `forward` 三分支逐字对齐（含「key 存在但 0 头时必须走 4 参数程序」） | 本文件 `_launch`/`run_case` 与 `kernels/rope/rope/rope.py` L263–271；失败实证：L0-7 初次 `ValueError: Kernel expected 4 inputs, but 5 are provided` | 在 develop skill 的「harness 迁移」小节写明：**参数个数必须按程序变体（`out_idx`）决定，而不是按「调用方手里有几个张量」**；并为多 `out_idx` 变体给出「一份 factory + 三个 prim_func」的推荐结构 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| Ascend 950：**UB 标量缓冲的值可以直接作 GM 行下标**（`cs[i,k] = TBL[row_ub[i], k]`，`row_ub` 由更早的 `T.Parallel` pass 写入），不需要退回 `T.alloc_local` + `T.serial` 展开；同一条 kernel 里 `@T.macro` 可同时收**缓冲与 Python int** 参数、块级运行时 `if/else` 两支各开 `T.SimtVF`、`out_idx` + `T.dynamic` 输出可被 callee 正确分配 | D | `examples/rope/_rope_positions_kernel/_rope_positions_kernel.py`（`side` macro 的 `cs_ub[i,k] = COS[row_ub[i], k]`、`if bx < nq:` 双分支、`out_idx=[4]/[5,6]/[6,7]`）；设计风险 R1 因此**解除** | `python repro/CONST-950-ub-value-as-gm-row-index.py`（cwd = 任务目录，真机，逐位 `atol=rtol=0` 通过；一个 kernel 同时断言上述四件事） | tilelang 0.1.15 / CANN 9.2.1 / Ascend950PR_9579 / 1 卡 | `.agents/platforms/toolchains/ascend/references/programming.md` + knowledge/PL-950-row-primitives.md |
| rope 的「fp32 单次舍入」契约在本后端被**逐位复现**：22 个真机 16-bit 用例（fp16/bf16，含 `D=65`、`R=24`、`R<D`、`±inf/NaN/65504`、clamp、offsets）与 fp32 torch golden 的 **max_diff 全部为 0**；只有 fp32 输出出现 `≤4.77e-07`（1e-5 门槛内）——即设计担心的 FMA/舍入顺序差异在 16-bit 路径上不存在，fp32 上的差异也只是 1 ulp 级 | D | `_rope_positions_kernel.py` 各 `run_L0/L1/L2/run_boundary` 的 `diff=` 栏（L0 13/13、L1 7/7、L2 3/3、boundary 11/11） | `python _rope_positions_kernel.py --level all`（真机）；FAIL 判据为 `_TOLERANCE`（fp16 1e-3 / bf16 1.6e-2 / fp32 1e-5，与源 `tests/ops/test_rope.py` 同值） | tilelang 0.1.15 / CANN 9.2.1 / Ascend950PR_9579 | pattern-library / numerical (PL-1.17-subfp32-fp32-transit 的证据补强) |
| rope 的 tile 几何对「pair 预算 2048」不敏感、对 `hpc` 的选择敏感：`bt = max(1, min(2048 // (hpc·R/2), 4096 // cols))` 在 6 条 workload 上给出 `bt ∈ {4,4,4,4,4,6}`、`tile ∈ {4096,3840}`、块数 `{80,5120,5120,4096,64,668}`，与设计 §5.2 表**逐格吻合**（含 phi-2 的非 2 的幂 tile 3840、15 元素/lane） | D | `DESIGN.md` §5.2 表 vs 本次核算（`hpc=(8,8)/(4,4)/(8,0)`、`cols=(1024)/(1024)/(640)`） | 复算 `bt` 公式（纯算术，见本复盘正文）；真机侧由 `run_L1` 六条 workload 通过佐证 | tilelang 0.1.15 / CANN 9.2.1 / Ascend950PR_9579 | `.agents/skills/tilelang-op-optimize/references/pattern-library/elementwise.md`（tile 预算条款） |
| 本设计选定的 SIMT 主选路径在真机上**端到端**比设计期流量下界慢约 5×（prefill 298.6 µs vs 60.7 µs 估算；decode 14.1 µs 说明单次 launch 的地板约 10 µs 级），且 gptj（跨 lane 步长 2）与 neox 的每 token 时延同量级（237.9 µs/2048 tok vs 298.6 µs/4096 tok）⇒ Stage 4 应先测块级固定开销与 ALU/发射项，再决定是否走 `units_per_block=4096` 与 B_gptj 分支 | D | 本文件 Stage 3 的 indicative 计时（host 循环 + `torch.npu.synchronize()`，**不是 msprof**，仅供 Stage 4 基线对照） | `python _rope_positions_kernel.py --level L1`（各 `L1-x` 行的 `ms` 即每例端到端时延；Stage 4 须以 msprof `Task Duration` 复核） | tilelang 0.1.15 / CANN 9.2.1 / Ascend950PR_9579 | `bottlenecks.md`（950 真机常数缺位）+ perf_opt 的 `[DESIGN_LIMIT]` 回写 |

### Transferable Lessons（可迁移教训）

- **先探针再写 kernel**：把「宏收缓冲参数 / UB 值作 GM 行下标 / 块级 if-else 内开线程域 / 动态 extent 的 out_idx」四件未取证事实压进一个 10 行的小 kernel，一次编译（~2 s）+ 一次运行就把设计里 4 条「待 Stage 3 确认」变成结论——比在 900 行的正式 kernel 上逐个试错快一个量级。
- **多 `out_idx` 变体的 harness 迁移，参数个数按「程序变体」而不是「调用方手里的张量」决定**：源 wrapper 对 `key_heads == 0` 一律走 query-only 程序，即使调用方传了 0 列的 key 张量；照张量个数调用会得到 `Kernel expected 4 inputs, but 5 are provided`。
- **表驱动的逐元素算子，把「行号/表行」载入 UB 后每行只算一次**：本算子每个 block 只算 `bt`（4~6）个行号，块内所有 head 复用同一表行；这既是设计 A1 的流量收益，也让 `T.Parallel` 的谓词只需落在 tile 的读写 pass 上。
- **UB 双缓冲（只读 `x_ub` + 只写 `y_ub`）让「先读后写」变成结构性质**：不需要任何 `T.sync_threads()` 手写同步，`InsertSync` 自动插同步即可在真机上得到逐位正确的结果；代价是每块多一条 UB→UB 拷贝 pass。
- **16-bit 输出的「fp32 中转 + 一次舍入」在本后端是位级可信的**：改写成原生 dtype 逐步舍入会引入 2.6e-1 级相对误差（设计 X1 证据），而位级契约在真机上可直接断言 `atol=rtol=0` 通过，比容差断言更能防回归。
- **迁移 harness 的 `default_config` 必须现读冻结 wrapper**：本次 REVIEW S1 依设计期转录值（128/512）报警，而 Stage 0 现态已是 2048，几何与设计 §5.2 自洽——省掉了一轮无谓的配置返工。
