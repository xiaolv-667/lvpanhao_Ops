# RETROSPECTIVE — shared_expert_mlp / `_gemm_basic_kernel`（Stage 1）

> 追加式复盘（只追加、不覆盖历史章节）。Schema：
> `.agents/skills/tilelang-skill-evolution/references/retrospective-schema.md`。

## 2026-11-23 · Stage 1 · first_design · v1

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| design / Phase M0 | 源函数是**两层工厂**（外层 8 个编译期参数 → 内层 6 个可调参数 → `@T.prim_func`），skill 的 M0「三问」模板假设被读的是一个 `@tilelang.jit` 函数；两层签名如何拆分（哪层属于语义、哪层属于配置）在模板里没有位置 | DESIGN.md §0.1 的两层结构表；源 `dense.py:4669-4831` | 在 `migration-analysis.md` §3.1「入口签名」补一问：工厂嵌套时须分别记录外层（语义/契约）与内层（tiling 配置）的签名，并说明迁移后内层哪些参数被删除（本任务删 `threads`，K9） | P |
| design / Phase R | R3「扫描遍数」口径对 GEMM 这类**块级重复读**算子不直接适用：单块内 A/B 各 1 遍，但块间重复读 `nt`/`mt` 次，是否计入「遍数」或另设「块级放大因子」口径，参考文件没有规定 | DESIGN.md §1.6.0 R3 表首行的口径说明（本文自定：遍数按块内、放大单列一栏） | `algorithm-research.md` §3 R3 增加一句：矩阵类算子允许把「块级重复读放大」作为访存口径的附列项，避免与 L2 复用混淆 | P |
| design / Phase 2 + 校验 | `verify_equiv.py` 的「最大 ulp 差」在**改变累加分组**的优化（split-K）上必然非零；若直接把它当判据会把合法的 split-K 判成 FAIL。真正判据应是「目标 dtype 输出差」而非「fp32 中间差」 | 首版脚本把 fp32 ulp 阈值设成 64，实测 128–4864 → 全 FAIL；改为「目标 dtype ulp ≤ 1 + 违反率 ≤ 1e-3，fp32 足迹仅作观测」后全 PASS | `tilelang-op-design` Phase 2 第 5 步补一条判定口径：重结合类优化的判据落在**输出 dtype**，中间精度差只作观测项记录 | R |
| design / gate | `design_calc_check.py` 的 `_section()` 把代码块里**列首的 `#` 注释**当成 Markdown 标题，导致 §5 在 §5.1 处被截断 → `l0c_budget` / `core_split` 全部 skip；同时预算表的 buffer 名写成 `` `a1` ``（反引号开头）会被行过滤掉 | 首版 DESIGN `design_calc_check` 输出 4×skip；去掉代码块列首 `#`、去掉反引号后 7×pass | 工具侧：`_section()` 跳过 fenced code block；行过滤放宽为「去反引号后以字母开头」。设计侧可在 skill 模板里提醒「预算表首列不要加反引号、代码块不要用列首 `#` 注释」 | D |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 950 ascend 后端：L1 输入的 `T.gemm` **只支持 NT**，且**不接受符号化 region**（`assert not trans_A and trans_B`、`raise "T.gemm requires static tile dimensions"`） | D | DESIGN.md §0.5 O9 / §5.4；`tilelang/ascend/op/gemm/gemm_mad.py::lower`（`_is_l1_input` 分支断言） | 探针 D1（静态 region 通过）/ 早期探针（`c0[:am,:an]` → `ValueError: M is symbolic: am`）；`history_version/design_probe_compile.py` | TileLang ascend 0.1.15 + CANN 9.2.0 SDK / 9.1.0 runtime，2026-11-23，无卡编译 | `pattern-library` 新增 `TRAP-950-l1gemm-nt-static`（或并入 gemm-tiles 条目） |
| 950 ascend 后端：`L0C → GM` 的 `T.copy` **自带 fp32→fp16/bf16 的硬件量化**（`quant_mode` 16/1/0），`dual_copy` 反而**不支持** dtype 转换 → fp32 累加的 GEMM 不必绕 UB | D | DESIGN.md §0.6 R3、§1.6.1 A1；`tilelang/ascend/op/copy.cc::GetCCQuantPre`；`copy_op.py::dual_copy` 的 dav-950 警告注释 | 探针 D1/D2/D12（`asc_copy_l0c2gm(..., asc_quant_mode(16/1/0))`）；`verify_equiv.py` 的 A1 位级对照 | 同上 | `pattern-library` 新增条目（替代「GEMM 必须经 UB cast」的直觉） |
| 950 ascend 后端：不显式 `annotate_buffer_versions` 时，`T.Pipelined` 的 L1 版本数由调度器**逐 buffer** 决定（实测 `a1` 3 版本 / `b1` 2 版本），片上预算因此不可静态核算 | D | DESIGN.md §6.3、§4.5 | 探针对照（同配置 auto vs `annotate_buffer_versions({a1:3,b1:3})` 的生成代码 `step & 1` vs `step % 3`） | 同上 | `pattern-library` / `programming.md` 的「同步与缓冲版本」段 |
| 950 ascend 后端：`T.use_swizzle` 在本后端**无实现**（`threadblock_swizzle_pattern` 只有 CUDA codegen 处理），且 `T.Kernel` 只有一维 → L2 亲和必须由 `bid` 分解自己表达 | D | DESIGN.md §0.5 O6、§1.6.3 D4③ | 静态检索 `src/ascend/codegen/`（无该 attr 分支）+ `dir(tilelang.ascend.language)` 有符号但无后端实现 | 同上 | `programming.md` 的「分配、布局和搬运」表补一行 |
| 分核判定模板缺「多波次中等规模」的默认选项：`core-split-strategy.md` 三选一（≤物理核 / 对齐整数倍 / 核内串行）不覆盖「逻辑核 1–32 波、每块工作量≫派发开销」的常见情形，实践中只能落在「对齐整数倍」并附加论证 | P | DESIGN.md §5.5③ | — | 同上 | `core-split-strategy.md` §1 要素③ 增加第 4 种：「多波次超发 + 波次派发」及所需量化依据（每块工作量 vs 派发开销） |
| 迁移工厂的**内层配置契约**（哪些 block 形状组合在目标后端非法、调用方必须如何分档）应作为 Stage 1 的独立交付项，而不是留给 Stage 3 试错 | P | DESIGN.md §5.2 配置契约 + §10.4 第 2 条（调用方 `_split_k(T≤32)=4` + `block_m=128` 在 `m∈{1,32}` 上必抛 `ValueError`） | `_shared_expert_mlp_fwd_kernels.py` 的 `_gemm_basic_func` 前置 raise + 本设计 §5.2 规则 3/4 | 同上 | `tilelang-op-design` 模板 §5 增加「配置契约」小节；`tilelang-op-develop` 增加「按 design 的契约校对调用方 config」检查 |

### Transferable Lessons（可迁移教训）

1. **"源实现能跑" ≠ "源实现的配置组合在语义上自洽"**：本任务的 NPU 脚手架把 decode 段的
   `split_k=4` 与默认 `block_m=128` 组合在一起，对 `m∈{1,32}` 必然触发源实现自己的
   `ValueError`。Stage 1 读源码时必须把**前置断言**当作语义的一部分检验，并把
   「调用方参数 × 前置断言」的交叉表算出来。
2. **编译探针（无卡）是设计期最高性价比的证据**：12 个变体 ~2 s/个，直接否定了三条原本会
   写进设计的错误假设（L1-GEMM 支持符号化 region、dual_copy 的 N-split 可符号推断、
   `use_swizzle` 可用）。凡「API 存在性 / 形态可编译性」的判断，都该先用 compile-only 探针
   落证据，再写进 DESIGN。
3. **等价性验证的判据必须落在输出 dtype**：任何改变累加分组的优化（split-K、树形归约、
   Welford）都会让 fp32 中间位模式必然不同；把 fp32 差当判据会把正确方案判死，把 fp32 差
   当"无所谓"又会漏掉真实精度回退。正确做法是「输出 dtype 差 ≤ 1 ulp + 违反率」+ 中间差作观测。
4. **后端硬约束要从安装包源码取证，不要从同仓 `examples/` 推**：本仓 `examples/gemm/` 的
   两份实现是 **NPUIR 后端**写法（`T.load_nd2nz`/`T.store_fixpipe`/`initC=`），照抄会得到
   `AttributeError`；ascend 后端的约束（NT-only、16 倍数、TILE_K_SUB、一维 grid）都在
   `tilelang/ascend/op/*` 与 `tl_templates/ascend/gemm.h` 里。
5. **n-fastest vs m-fastest 的取舍由 L2 容量与足迹决定，与"先例怎么写"无关**：源实现的
   `T.use_swizzle(10)` + n-fastest 在 128 MiB L2、A 面板 ≤ 58.7 MB 的 NPU 上是错的
   （t4096 时约 32 倍权重流量）。这类判断要算出来，不要继承。

## Stage 2 (review) Retrospective — first_review · 2026-11-23

> 复盘对象 = 检视流程自身（设计问题已随 REVIEW.md 流转给 Stage 1，不在此重复）。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Review-flow / D-1 机械复核 | 维度 8 的机械项只要求「重跑 `verify_equiv.py` 一致且全 `EQUIV_PASS`」，但脚本可以用**恒真/同构构造**凑 PASS（本任务 `A2_fp64_exact = torch.equal(ref64, ref64)`；A1/A3 两侧由同一批 fp32 值构造）。重跑一致 ≠ 判据有效，检视清单里没有「判据有效性抽检」这一位 | `verify_equiv.py:135`（恒真项）、`113-122`（A1/A3 同构构造）；REVIEW.md §8 建议 7 | 在 `tilelang-design-review` Phase 1 第 7 步增加一条：重跑通过后，抽检脚本判据是否含恒真式 / 两侧是否共用同一输入构造；命中则在维度 8 记「证据强度 warn」并要求改写判据 | R |
| Review-flow / 维度表缺口 | 三处内部矛盾都落在 **DESIGN 文本 ↔ 跨文件接口**（`§10.4` 的「已内建兜底」 vs 伴随 `splitk_reduce_kernel` 的 `[split_k,m,n]` 契约；bias 分支 extent vs §0.6 R5），而维度 7 只检「API 映射/内存规划/Tiling 三处相互一致」，没有「交付说明（§10.x）与接口契约一致」检查项 | DESIGN.md §10.4-2、§5.2 注、§4.2；`shared_expert_mlp.py:289-298` | 维度 7 增一行：「§10 交付说明 + §5.2 配置契约与 §4.2 输出契约、伴随 kernel 接口一致」；并明确「工厂内静默改配置参数」属禁项 | P |
| Review-flow / 探针证据留存 | 探针只留脚本不留产物（`history_version/` 无 `probe_src_*.cce`），且脚本自带的事实提取正则只匹配 `step % N`，`ns=4` 的 `step & 3` 被记成空 → 检视时必须自己重跑并解析生成代码才能判 D11/§6.3 | `history_version/design_probe_compile.py:146`（正则）；本次重跑结果 `step & 3` + 4 组事件 | `tilelang-op-design` 的探针模板要求「关键生成代码片段（或其摘要）随 DESIGN 留存」；`tilelang-design-review` 增加「探针产物缺失时重跑」的操作项 | P |
| Info-source / roofline 合规 | 「`cost_measurements` 为空时允许写 `估算下界: 待核验`、不得用 910B 常数补齐」这条只在 `standards/hardware-cost-model.md` 里，检视维度表（维度 1–8）没有位置，reviewer 需自行找到该标准才能判 | DESIGN.md §1.6.0「设计期估算」行；`hardware-cost-model.md` §2 | 维度 8 的「调研完整性」行补一句：`cost_measurements` 为空时校验「流量项已算 + 下界明确标注缺哪些常数 + 未使用其它型号常数」 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 用**生成代码里的地址常量**反解块遍历序（比读 DESIGN 的声明可靠）：`A[(bid & 31)*917504]` / `B[(bid >> 5)*1835008]` / `C[(bid & 31)*524288 + (bid >> 5)*256]`（mt=32、nt=16、m=n=k=4096/7168）直接证明 m-fastest | P | `examples/shared_expert_mlp/_gemm_basic_kernel/history_version/design_probe_compile.py` 的 D1；`REVIEW.md` 机械复核 3 | `ASCEND_RT_VISIBLE_DEVICES="" PROBE_OUT=/tmp python history_version/design_probe_compile.py`，看 `probe_src_D1-*.cce` 的 `A[`/`B[`/`C[` 下标 | TileLang ascend 0.1.15 + CANN 9.2.0 SDK / 9.1.0 runtime，2026-11-23，无卡编译 | `tilelang-design-review`（维度 3/8 的取证方法） |
| 950 `ascend_gemm_l1` 的 `TILE_K_SUB` 由 L0 容量反推而非配置：`_compute_tile_k_sub` 用 `l0_capacity=65536 / 2 stages`、`max_k = min(32768·8/(M·bits), 32768·8/(N·bits), K)` 取整到 `C0`；M=128/N=256/bf16 时得 64，恰使 L0B 两阶段 = 65536 B（用满不越界，模板按 `(sk & 1)` 索引） | D | `tilelang/ascend/op/gemm/gemm_mad.py:206-222`、`tl_templates/ascend/gemm.h:55-60,103`；DESIGN.md §4.5/§5.3；`design_calc_check` 的 `l0b_budget` pass（64/64 KB） | 探针 D1：`ascend_gemm_l1<128, 64, 256, 64, true, bfloat16_t>`（第 4 个模板参数 = TILE_K_SUB = 64） | 同上 | `gemm-tiles.md` / `pattern-library`（补「L0B 用满即合规」的判读规则） |
| 跨 kernel 契约禁项：**主循环 kernel 不得静默收敛由调用方同时传给伴随 kernel 的配置参数**（本任务 `split_k` 同时决定主循环 `output_shape`/`output_dtype` 与 `splitk_reduce_kernel` 的输入形状）。迁移脚手架丢配置选择时，正确做法是调用方按 workload 分档 + kernel 侧对非法组合显式 `ValueError` | P | DESIGN.md §3.3 vs §10.4-2；`_shared_expert_mlp_fwd_kernels.py:192-198`（`output_shape`/`output_dtype` 随 `split_k` 变）+ `shared_expert_mlp.py:289-298`（reduce 按同一 `split_k` 构造）；GPU 侧 `heuristics.py:940-966` 的 `"block_m": m` | 静态核对三处 + 用 m=1、split_k=4 构造配置即复现 `ValueError` | 同上 | `tilelang-op-design` 的「配置契约」小节 + `tilelang-design-review` 维度 7 |

### Transferable Lessons（可迁移教训）

- 同批次其它函数（`splitk_reduce_kernel`、`_silu_mul_fused_kernel`）检视时：**先画调用方 ↔ 主循环 ↔
  伴随 kernel 的参数传递链**（`shared_expert_mlp.py::_gemm_program` 是范本）。凡是既进主循环工厂又进
  伴随 kernel 构造的参数（本任务 `split_k`、`elems_per_cta`），主循环侧的任何「静默收敛」都会让两侧
  形状/语义分叉；回落只能由调用方完成。
- 迁移任务的「兼容性」问题要先分清**语义前置**（源实现正确性所必需，如 `k % block_k == 0`）与
  **实现前置**（源 GPU 路径的产物，如 split-K 的 `m % block_m == 0`——它在 GPU 上只为省 epilogue 守卫）。
  对后者要问「目标后端的新落盘路径是否已经使其不必要」，否则会把 GPU 的限制带进 NPU 设计并派生一堆补丁。
- 检视等价性证据时，`EQUIV_PASS` 重跑一致只是下限：还要看脚本每项判据是否**能失败**（恒真式、
  两侧同构构造、缩放不改变 Worst-case 的用例都要点出来）。

## Stage 3 (develop) Retrospective — first_impl · attempt 1 — 2026-11-23

> 覆盖整个 attempt 链：首跑 L0-1…L0-6 全过 → L0-7（M 尾块）暴露 1 个**运行期**后端缺陷
> → 改落盘形态后 L0/L1/L2/Boundary 全过（37/37，0 warn）。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Develop-flow / Phase 2–4 | 设计期探针的结论列只区分「编译通过」，没有「运行正确」位；本任务 `c0[:am,:an]` 落盘在 §3.4/§11.1 里是 ✅（D8/D9 编译通过），Stage 3 首跑才发现运行期错（尾块列 ≥16 全错）。skill 没有任何「新 copy/gemm 形态必须配一个最小运行期尾块用例」的硬性要求 | DESIGN.md §3.4（`T.copy(c0[:am,:an], …)` 行 ✅）、§11.1 D8/D9；`_gemm_basic_kernel.py` L0-7 首跑 3484/100000 元素超差 | `tilelang-op-develop` Phase 2 增加一条：**凡设计里首次出现的 copy/store 形态（尤其带符号 extent、带 dtype 转换、带 dual_dst），Phase 4 首跑必须先放一个最小运行期用例**（比 L0 全套更早）；`tilelang-op-develop` Phase 1 读 DESIGN 时把 §3.4 的「✅ 编译」一律当未验证处理 | R |
| Develop-flow / 四层语义 | 「L0/L1 必须全过、L2/Boundary 告警不阻塞」与 Boundary 层里「非法配置必须抛 ValueError」这类**硬断言**混在一起：边界层一旦承载硬契约检查，就不能整体吞异常。skill 未说明分层里哪些是 gating 内容 | `_gemm_basic_kernel.py::run_boundary`（6 个 `_expect_value_error` + 非阻塞 `_warn_case`）；stage prompt 第 2 条（L0/L1 gating） | `tilelang-op-develop` Phase 4/5 明确：Boundary 的**数值**用例可非阻塞，**契约/异常**用例（ValueError/形状契约）属 gating，失败须计入 exit code | R |
| Info-source / 执行上下文 | 平台 `execution.precision_command` 指向 `/usr/local/python3.12.16/bin/python3`，该解释器 `import tilelang` 直接 `ModuleNotFoundError`；真正可用的是 `env-950pr.sh` 里的 `$TILEOPS_950PR_VENV/bin/python`（= `/workspace/venv-ascend950/bin/python`）。子 stage 若照 `execution` 原样执行会得到「环境问题」误判 | `statectl platform --task-dir …` 的 `execution.precision_command`；`/usr/local/python3.12.16/bin/python3 -c "import tilelang"` → ModuleNotFoundError；`env-950pr.sh:19-20,79` | 平台配置修正 `execution.precision_command` 为 `"${TILEOPS_950PR_VENV}/bin/python"`（或 env 脚本里 `export` 后在命令中使用变量）；`AGENTS.md` 的 Execution context 段补一句「命令里的解释器以 env 脚本解析出的 venv 为准」 | R |
| Develop-flow / 计时口径 | skill 模板的 `check()` 只「调用一次 + 比对」，没有计时/预热约定；本任务首次调用新特化 ~106 ms（设备侧 kernel load），稳态 1.2 ms —— 直接单次计时会把性能低估 ~90× | `_gemm_basic_kernel.py::run_case`（warm-up + 3 次取均值）；手工对照：iter0 0.1065 s / iter1 0.0016 s / 5×均值 1.21 ms | `tilelang-op-develop` 模板的 `run_case` 加「首次调用只做预热，计时取 3 次均值」并打印 GFLOP/s，Stage 3 的数值才能当 Stage 4 基线 | D |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **L0C→GM（及 L0C→UB）部分区段的源 region 不可截断**：`T.copy(c0[:am,:an], C[...])` 在 `am` 不是整 16 行组的整数倍时，lowering 用 `compact_src_region.outer1->extent * row_frac` 当源行距（= ceil16(am)）而非 tile 行数，尾块从第 16 列起全错（m=48/bm=32：766 个元素超差、max_err 5.5e1）；改成**源 region 取整块、目标 region 截断**（`T.copy(c0, C[m0:m0+am, n0:n0+an])`）后 0 超差 | `tilelang/ascend/op/copy.cc:690`（`loop_src_stride`）与 `codegen_ascend.cc:1494`（`asc_copy_l0c2gm` 的第 6 个实参）；实测：本文件 L0-7/L0-11/L0-12/L1-t1-down 改前全错、改后全过 | `repro/TRAP-950-l0c-region-store-stride.py`（自含断言：整块源 0 违反 vs 截断源 766/190/191 违反；`block_m=16` 时两者都对，印证「ceil16(am)==block_m」条件） | TileLang ascend 0.1.15（`/workspace/venv-ascend950`）+ CANN 9.2.1 runtime，Ascend950PR_9579（28 Cube / 56 Vector），2026-11-23，真机 | `pattern-library` 新增 `TRAP-950-l0c-region-store-stride`；`gemm-tiles.md` 的「输出只写当前有效 M/N」一句补落盘形态 |
| 950 上 **L1 输入的 `ascend_gemm_l1` 模板**（`T.gemm(a1_l1, b1_l1, c0_l0c, transpose_B=True)`）不仅可编译，而且真机正确且高效：bf16 (4096,4096,7168) 128×256×64×3 阶段 = **1.21 ms ≈ 198 TF/s**（5 次背靠背均值），全 37 用例 0 精度超差；此前该形态只有 DESIGN 的「编译通过」证据 | `_gemm_basic_kernel.py` 各用例输出；手工计时脚本见 repro 说明；DESIGN.md §3.4/§11.1 原为纯编译证据 | `ASCEND_RT_VISIBLE_DEVICES=0 $TILEOPS_950PR_VENV/bin/python _gemm_basic_kernel.py --level all`（L1-t4096-gate-up 一行给 TF/s） | 同上 | `gemm-tiles.md` 增补「L1 操作数形态」的性能/正确性一位（原条目只有显式 L0 操作数） |
| fixpipe 的 fp32→bf16/fp16 量化（`quant_pre` 16/1/0）在真机上是**就近舍入**而非截断：k=128 的小 shape 上 bf16 输出与 `cast_bf16(fp32 matmul)` **逐位相等**（max_diff=0.0），fp16 为 1 个 fp16 ulp（7.8e-3 @ |C|≈8） | `_gemm_basic_kernel.py` L0-1（bf16 max_diff 0.0）/L0-2（fp16 7.812e-3）/L0-5 四布局全 0.0；DESIGN.md §9.1 第 2 条把它列为「未实测」 | `--level L0` 的 L0-1/L0-2 两行 | 同上 | `numerical-boundaries.md` 或 `gemm-tiles.md`：补「L0C→GM 的 dtype 转换是 RN」的实测结论 |
| **`has_bias` 的 Cube+Vector epilogue 形态真机可用**（`dual_copy(c0→cv)` M-split → `T.SimtVF` 加 bias 并 cast → `dual_copy(co→GM)`）：(32,64,128) bf16 与 `cast(fp32 matmul + fp32 bias)` **逐位相等**。DESIGN §9.3 第 1 条把它列为最大风险（「`T.copy(bias→bv)` 可能只有一个 AIV 执行」）——实测 GM→UB 的 1-D bias 切片在两个 AIV 上都拿到了数据，不需要 `sub_blockid`/`T.Vector` 回退 | `_gemm_basic_kernel.py` L0-6（max_diff 0.0）与 L0-8-k0-bias（strict equal）；DESIGN.md §9.3-1 | `--level L0` 的 `L0-6-bias`/`L0-8-k0-bias` | 同上 | `execution-patterns.md`（PL-950-mixed-epilogue）补「bias 广播：GM→UB 1-D 切片两 AIV 各得一份」 |
| **k == 0 的落盘不要依赖 L0C**：主循环零步时用「UB 填 0 / 广播 bias + 区段截断写 GM」（纯 Vector 特化，`__vector__`）即可，`T.clear(c0)` 这种未核验形态可以整段避开；语义等价且与源实现 strict 相等 | `_gemm_basic_kernel.py::_gemm_basic_program` 的 `k_tiles == 0` 分支；L0-8 / L0-8-k0-bias / L0-8b-k0-tails（三者 `torch.equal` 通过，含 am=16/an=44 双尾块）；DESIGN.md §9.3-5 的备选③ | `--level L0` 的 `L0-8-k0` / `L0-8-k0-bias` / `L0-8b-k0-tails`（最后一条同时证明 UB→GM 的区段截断没有 L0C 那个行距问题：同样形态下 strict 相等） | 同上 | `gemm-tiles.md`：k==0 边界用例的推荐落盘路径 |
| 工厂级「tile 前置 + 退化」策略（`block_m := min(block_m, align16(m))`；`split_k>1` 且不满足精确 tile / `m<16` 时**抛 ValueError** 而非静默改 `split_k`）：让 harness 默认 `block_m=128` 在 `m=1/32` 上可用，同时不破坏 `[split_k,m,n]` 输出契约 | `_gemm_basic_kernel.py::_clamp_block_m/_check_tiles/_split_k_for`；boundary 的 6 个 `_expect_value_error` 用例；DESIGN.md §5.2 规则 3/4 | `--level boundary` 的 `B-err-*` 六行 + `B-m32-clamp-bm128` | 同上 | `tilelang-op-develop` 的「harness 兼容」小节 + Stage 5 接线检查清单 |
| 单个 kernel 特化的**首次调用**要付 ~100 ms 设备侧加载（同一特化后续 1.2–1.6 ms）；跨 16 个特化的全量套件里这部分占 >90% 墙钟 | `_gemm_basic_kernel.py::run_case` 的注释与实现；手工计时 iter0 0.1065 s vs iter1 0.0016 s | `/tmp/opencode/time_check.py` 形态的 3 次循环（结论已固化到 run_case 的预热逻辑） | 同上 | `bottlenecks.md`：区分「kernel 稳态时延」与「特化首次加载」，Stage 4 基准须预热 |

### Transferable Lessons（可迁移教训）

- 本迁移后续函数（`splitk_reduce_kernel`、`_silu_mul_fused_kernel`）只要出现 **L0C 出数**
  （`T.copy(c0, …)` / `dual_copy(c0, …)`），源的 region 一律取**整块**、只在目的侧截断；把
  `c0[:am, :an]` 当源会静默算错尾块（第 16 列起全错），而且编译器不会报错。
- 任何「新 copy 形态 + 尾块 shape」的组合，先跑一个 `m % block_m != 0` 的最小用例（例如
  (48,64,128)/bm=32），再去跑大 shape；大 shape 多数整除，永远暴露不出尾块错误。
- `has_bias` 类 epilogue 若要复用本函数的结论：Cube→Vector 跨界必须用 `dual_copy` 的 2:1
  切分，而 **GM→UB 的 1-D 广播切片（bias/scale）两个 AIV 都会各拿到一份**，不需要
  `sub_blockid`；只有 L0C→UB 的整块 `T.copy` 才是单 AIV 的（见 mla_dense_bwd 的 TRAP 条目）。
- 计时必须先预热：一个特化的首次 launch 约 100 ms，稳态是 ms 级；Stage 3 报的错误时延会直接
  污染 Stage 4 的基线判断。
- 平台 `execution.precision_command` 里的解释器可能不可用；本任务实际解释器是
  `$TILEOPS_950PR_VENV/bin/python`，且必须在**任务目录**运行（仓库根的 `tilelang/` 会遮蔽安装包）。
