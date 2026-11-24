# RETROSPECTIVE — `_ssd_chunk_coupling_kernel` (ssd_chunk_coupling)

## Stage 1 (design) Retrospective — first_design — 2026-11-23

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Design-flow | 平台 `stage_skills.design`（tilelang-ascend）本身只有路由，Phase M0/R/M1 的实际执行标准在 `tilelang-op-design`；两处 skill 名称不同，容易漏掉 Phase R 的调研四问与 D-1/D-3 | `.agents/skills/tilelang-ascend/SKILL.md`（全文 12 行，只给路由）vs `.agents/skills/tilelang-op-design/SKILL.md:104-217` | 在 tilelang-ascend SKILL.md 的 Stage 1 段落显式列出 Phase M0 → R → M1 → 1 → 2 的顺序与两个强制工件（verify_equiv.py / design_probe_*.py） | R |
| Info-source | 平台 knowledge 无 `cost_measurements`、无 `CONST-*` 常数（只有容量），D-2 的 roofline 只能写"待核验 + 未实测假设"，估算行难以复核 | `statectl platform` 返回 `cost_measurements: []`；`.agents/platforms/toolchains/ascend/knowledge/bottlenecks.md`（"当前没有 950 真机验证的瓶颈模式"） | 为 ascend 平台补首批 D 类常数（HBM 有效带宽、Cube fp16 峰值、块启动代价），否则 D-2 在 ascend 上长期只能做容量项 | D |
| Design-flow | gate 1 的 `S1-PLACEHOLDER` 会把公式里的中文花括号（如 `Σ_{n∈K块 kk}`）判成未替换的模板变量，而公式推导天然会用到这种写法 | `statectl gate 1` 首次运行 → `疑似模板变量未替换：{n∈K块 kk}` | gate_lint 的 TEMPLATE_BRACE_RE 排除 `Σ_{`/`_{` 前缀的数学下标（或限定只检查行首模板槽位） | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| ascend：`T.MixedKernel(sids=2)` + 显式 `T.Cube()/T.Vector()` 在混合 epilogue 上编译失败；`T.Kernel` + 显式 `T.Vector(vector=2) as sid` 可用 | D | `examples/ssd_chunk_coupling/_ssd_chunk_coupling_kernel/history_version/design_probe_compile.py`（P3/P4 fail vs P5–P12 ok）；`probe_out/probe_fail_P3-mixed-mask-fp16.log` | `cd examples/ssd_chunk_coupling/_ssd_chunk_coupling_kernel && ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python history_version/design_probe_compile.py` | TileLang ascend 0.1.15 / CANN 9.2.1 / ascend-950pr-9579 | `.agents/platforms/toolchains/ascend/knowledge/execution-patterns.md`（新增混合 kernel 形态条目） |
| ascend：`dual_copy` 的 L0C→UB 必须同 dtype（fp32 中转），跨 dtype 前端直接拒绝；`ub2gm` 的 sid 偏置按 `64·Q` 元素计算 | D | `DESIGN.md#3.2`（步骤 5/7）；probe P5/P10 生成码 `asc_copy_l0c2ub(..., dual_dst_mode=1)` / `asc_copy_ub2gm_align(64, 行跨度 Q·2B, sid 偏置 64·Q)` | 同上 probe | 同左 | `.agents/platforms/toolchains/ascend/knowledge/execution-patterns.md`（PL-950-mixed-epilogue 补 dtype 与分片偏置条款） |
| ascend：GM→L1 `T.copy` 支持 4-D region 中"固定中间索引 + 两维切片"，行跨度正确推导为 `G·N·elem_bytes` | D | `DESIGN.md#3.4`；probe P1（G=8 跨度 2048 B）、P2/P6（G=1,N=64 跨度 128 B） | 同上 probe | 同左 | `.agents/platforms/toolchains/ascend/knowledge/gemm-tiles.md`（补 strided 4-D 源区域条款） |
| 设计手法：用**循环 trip 置 0** 代替"`if` 包住 pipelined GEMM"实现三角剪枝——前者前置 notify 与尾部 drain wait 数量在 0/2 迭代下平衡，后者可能失配挂死 | P | `DESIGN.md#0.6`(R4)、`#1.6.0`；probe P5（drain wait 在循环体外）vs P8/P10（trip=0 版本） | 同上 probe（对 P5 与 P8 生成码比较 `asc_sync_notify/wait` 计数） | 同左 | pattern-library（新增"条件化流水"反例条目） |
| 设计手法：被掩码丢弃的计算可用"垃圾累加器 + 统一 select 掩码"跳过，且必须以 fp64 参照 + nan/inf 角点机器验证（乘法掩码会因 `nan*0=nan` 破坏"上三角精确 0"契约） | P | `DESIGN.md#1.6.1`(O3 + 未采纳表)；`verify_equiv.py` E1/E3/E4 输出 | `cd examples/ssd_chunk_coupling/_ssd_chunk_coupling_kernel && /workspace/venv-ascend950/bin/python verify_equiv.py` | torch CPU（无设备依赖） | `.agents/skills/_shared/standards/`（等价性验证的反例集：掩码类算子必含 nan 角点） |
| 设计手法：Cube 三角 tile 剪枝的"理想下界"必须手推读+写两项（读 = 输入元素数·elem_bytes，写 = 输出元素数·elem_bytes）；本次检视发现设计把 wl3 下界误算成 33.6 读（应 37.7），说明"下界行"比"候选行"更容易出算术滑移 | P | `REVIEW.md` 维度 8.7（75.5 MB 应为下界，设计写 71.3 MB）；`DESIGN.md#1.6.0`(R3 表首行)、`#1.6.3`(#3 行) | `python3 .agents/tools/design_calc_check.py --design <path> --task-dir <task>`（该项不校验下界行，需人工复算） | 同左 | `.agents/skills/tilelang-op-design/references/algorithm-research.md` §3 R3（增一条"下界/参考行同样逐项复算"） |

### Transferable Lessons（可迁移教训）

- ascend 平台做"Cube + 逐元素尾部"时，**不要**用 `T.MixedKernel` 手写 `T.Cube()/T.Vector()` 双区域（分配在 launch 作用域会让标量任务同时触碰 UB 与 L1/L0，直接编译失败）；用 `T.Kernel`（自动调度）+ `with T.Vector(vector=2) as sid:` 只包住逐元素尾段，`T.dual_copy` 放在该 `Vector` 块内。
- L0C 结果要在 AIV 上做逐元素处理时，UB 中转缓冲必须与 L0C 同 dtype（fp32），cast 只能发生在 VF 内的第二次写入（另一块 dtype 缓冲）；`dual_copy` 的 L0C→UB 跨 dtype 是硬拒绝，不是 warning。
- 需要"跳过某类块的全部计算"时，优先把条件写成**循环 trip 数**（`T.if_then_else(cond, n, 0)`）而不是 `if` 包住流水循环体：本工具的 pipeline drain 同步在循环体外，trip=0 有前置 notify 兜底，`if` 则可能让 set/wait 失配。
- 三角/掩码类算子的掩码必须用 `if_then_else`（select），禁用乘法：`nan*0=nan` 会让"上三角精确 0"契约在非有限输入下失效；此类等价性用 fp64 参照 + nan/inf/±0/subnormal 角点做机器验证，成本几分钟。
- 设计期 compile-only 探针（`ASCEND_RT_VISIBLE_DEVICES=""` + `tilelang.compile(target="ascend")` + 检查 `get_kernel_source()` 的关键指令）能在无卡条件下拦截 API 形态错误，还能读出地址跨度/同步配对等语义事实，性价比远高于等到 Stage 3。

## Stage 2 (review) Retrospective — first_review — 2026-11-23

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Review-flow | 维度 1「API 形态可用性」只要求核对 DESIGN 自己的 probe，**没有**「与同 session 其他算子刚报告的失败形态做形态比对」的步骤。本任务是跨任务风险的实例：另一算子（`mla_dense_bwd/_bwd_kernel`）的检视刚把「显式 `T.Vector as sid`」判为阻塞级不可编译，而本设计通篇依赖同一 API；skill 无判定口径，只能检视方自建最小探针来区分「同一形态 / 不同形态」 | `REVIEW.md` 维度 1 第 4 条（证据 A–D）；对照 `examples/mla_dense_bwd/_bwd_kernel/REVIEW.md` 与 `DESIGN.md:858` 的反例行 | `tilelang-design-review` Phase 1 增一步：若同一 session 内已有其他算子报告过某 API 形态失败，必须（a）用最小可编译探针比对两处形态差异，（b）把「触发因子」而非「API 名」写进结论 | R |
| Review-flow | 占位符门禁会命中报告里**说明**无占位符的元语句（首版在「无占位符」条目里写出了关键字字面量，需二次改写才能过 scan） | `REVIEW.md` 维度 7 该条的两次改写（本任务 Stage 2 写盘过程） | review skill 的维度 7 检查项文案改为"以不含关键字字面量的方式陈述检出结果"，或在 gate_lint 中跳过反引号包裹的关键字 | R |
| Info-source | 维度 8 的复杂度复算缺「不可实测列」的口径：`cost_measurements` 为空时 R3 表的「含 L2 削减」「因果下界」两列既无法实测也无法反查，本次 5.6% 偏差正落在这两列（`design_calc_check` 的 `r3_metrics` 只查四口径存在性） | `REVIEW.md` 维度 8.7（75.5 MB vs 设计 71.3 MB）；`design_calc_check` 输出 `r3_metrics: pass（存在性）` | `algorithm-research.md` §3 R3 增一条：凡"下界 / 含 L2 / 未实测"列，设计必须写出推导式（读+写分项），reviewer 逐项复算而非只核候选行 | P |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| ascend：`T.Vector(vector=2) as sid` + `SimtVF` + GM 侧 dual_copy（L0C→UB 硬件双目的 + UB→GM）**可编译**；但同一 kernel 若另有 **UB→L1 的软件 dual_copy**（double 分片）+ 后续 GEMM，则抛 `InternalError: Check failed: (!var_idmap_.count(v)) is false: Need input to be in SSA form dup sid`，且**与是否写 `as sid` 无关**（`with_sid`/`no_sid` 两模式同时失败） | D | 检视方探针 `/tmp/opencode/rv2/probe_sid_bound.py`、`probe_sid_bound3.py`（variant `u2l1_outside_frame` FAIL、`design_form` OK）；MLA 检视遗留探针 `/tmp/opencode/rv/probe_sid.py` 今日重跑两模式均 FAIL；本设计 P5–P12 全 OK（`probe_out/probe_src_P10-final-fp16.cce` 含 `asc_get_sub_block_id()` ×2 且编译通过） | `cd /tmp/opencode/rv2 && ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python probe_sid_bound3.py`（无卡，仅编译） | TileLang ascend 0.1.15 / CANN 9.2.1 / ascend-950pr-9579 | `.agents/platforms/toolchains/ascend/knowledge/execution-patterns.md`（PL-950-mixed-epilogue 补「sid 作用域边界」：GM 侧 dual_copy 可共存，UB→L1 软件 dual_copy 不可） |
| ascend：动态 trip（`trips = 0`）的流水化 Kernel 其 notify/wait 在各事件通道上计数相等，可作「条件化流水是否悬挂」的静态判据：P10 生成码前期 6 条 `asc_sync_notify`（M→MTE1 ×2、MTE1→MTE2 ×4）与循环外尾部 6 条 `asc_sync_wait` 一一配对，循环内各通道 notify/wait 在 0/2 迭代下相等 | D | `probe_out/probe_src_P10-final-fp16.cce`（全文）；`REVIEW.md` 维度 5 第 4 条（检视方逐通道清点） | `grep -c asc_sync_notify\|asc_sync_wait probe_out/probe_src_P10-final-fp16.cce` 后逐行配对 | 同左 | pattern-library（新增"条件化流水"判定条目） |
| 检视手法：把 probe 设备码当「语义事实」核对（4-D region 的 nd2nz 行跨度、`dual_dst_mode`、ub2gm 的 sid 偏移与行跨度、VF 谓词逐 lane 展开），可在无卡条件下证明设计与生成码一致，比"编译通过"强一级 | P | `REVIEW.md` 维度 1 证据 C、维度 8.5；`probe_src_P11-final-G8-fp16.cce`（跨度 2048 B、`sid*16384` 偏移） | 同左 probe 产物 | 同左 | `.agents/skills/tilelang-design-review/` Phase 1 第 7 步（机械复核后加「生成码语义核对」子步） |

### Transferable Lessons（可迁移教训）

- 判定「某 API 形态不可用」必须绑定**触发因子**而非 API 名：本次「显式 `T.Vector as sid` 不可编译」的真实触发因子是「同一 kernel 里另有 UB→L1 的软件 `dual_copy`（2:1 分片）+ 后续 GEMM」，去掉该路径后同一写法在本工具链 0.1.15 上可编译。跨任务继承此类结论前，先用最小探针做形态比对，否则会把可用路径误判为阻塞。
- 同一设计里的多个 `dual_copy` 要按落点分类核对：L0C→UB（硬件双目的）与 UB→GM（按 sid 分片）可共存于「自动调度 Kernel + 显式 Vector 帧」；UB→L1（软件双拷贝 + 后续复用）是另一条路径，混入即 SSA 绑定冲突。
- 审阅动态 trip 的流水实现时，用「逐事件通道的 notify/wait 计数」做静态安全判据（0 迭代与满迭代各算一遍），比只看生成码能否编译更能排除 Stage 3 的挂死风险。
- 审阅算法调研的复杂度表时，**下界/参考行与候选行同权复算**：候选行的错误会立刻推翻选型，参考行的错误会被"结论没变"掩盖，但对 Stage 4 的 roofline 对账是直接误差源。

## Stage 3 (develop) Retrospective — first_impl — 2026-11-23

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Develop-flow | 平台把「API 形态可用性」的判据落在**编译证据**上（`references/programming.md`「API 与执行区域」列的 `T.Vector(vector=2)` 只有源码确认；设计期 D-3 探针也是 compile-only）。本次该形态**编译通过、运行出错值**，说明「id / 同步 / 帧类 API」的 compile 证据不能替代执行证据，而 skill 未区分这两类 API | `_ssd_chunk_coupling_kernel.py` 首版按 `DESIGN.md#3.3` 用 `T.Vector(vector=2) as sid` → `logs/run_L0_a1.log`（L0-3 对角错、L0-4..L0-7 大面积错）；`repro/TRAP-950-aiv-sid-vector-frame.py` 量化 5–7/8 次发射损坏 | 在 `references/programming.md`「API 与执行区域」增一列「证据级别＝仅编译 / 已执行」，并要求 id/帧/同步类 API 的 Stage 3 **首个上卡动作必须是带数值断言的冒烟** | R |
| Develop-flow | 四出口门禁只写「与 golden 对比精度正常」，没有给 **bf16 输出 1 ulp 舍入边界**的处理口径。本次 bf16 最大偏差 0.125 = 恰好 1 ulp（16/1.36M 元素），任何只按 `atol=rtol=1e-3` 判定的实现都会在 bf16 workload 上假失败；每个任务都要自己重推一遍容忍度 | `logs/run_L0_a2.log#L0-5`（16 elements, 1 ulp）；`_ssd_chunk_coupling_kernel.py#_compare`（`max(atol+rtol*|g|, 1 ulp_out)` + 报告 `differing` / `max_ulp`） | 在 `tilelang-op-develop` 的精度比对规范里给统一口径：允许 1 个输出 dtype ulp，并**必须**同时报告差异元素数与最大 ulp 距离（区分"舍入翻面"与"真错"） | R |
| Info-source | 平台 hardware 记录写 `cube_core_cnt` / `vector_core_cnt`，实际 `torch_npu` 属性名是 `cube_core_num` / `vector_core_num`（按 DESIGN §5.5 的复核代码直接跑会 `AttributeError`） | `_ssd_chunk_coupling_kernel.py` 复核输出 `device Ascend950PR_9579 cube_core_num 28 vector_core_num 56`；`DESIGN.md#5.5` 复核代码块 | 平台 hardware/spec 与 `DESIGN.md#5.5` 模板统一改用 `*_core_num`（或同时给出两个名字） | D |
| Develop-flow | 模板 `kernel_template.py` 的 `run_LX()` 直接计时，未规定 warm-up。本次未 warm-up 的计时把「新编译 kernel 的二进制加载 + 该尺寸首次输出分配」算进 kernel（50–120 ms vs 稳态 20–155 µs，差 3 个数量级），足以让人误判算子有性能问题 | `logs/run_L1_a1.log`（L1-1 launch=54 ms）vs `logs/run_L1_a2.log`（同一用例 0.051 ms，仅加了 warm-up + 逐次中位） | `kernel_template.py` 的 `run_case` 增加 warm-up 调用与"逐次计时取中位"，输出 `launch` 明确不含首次开销 | P |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| ascend TRAP：`with T.Vector(vector=2) as sid` 读取 AIV 子块号**编译通过但值错且非确定**（64 块网格下 8 次发射有 5–7 次损坏一个 64 行半块，8 次出现 5–7 种不同结果；严格上三角仍精确为 0 故不易察觉）；`sid = T.call_extern("int64","asc_get_sub_block_id")` 每次精确且确定 | D | `repro/TRAP-950-aiv-sid-vector-frame.py`（3 次重跑：extern 0/8、vector 5–7/8）；`_ssd_chunk_coupling_kernel.py#3)` epilogue；`logs/run_L0_a1.log` vs `logs/run_L0_a3.log` | `cd examples/ssd_chunk_coupling/_ssd_chunk_coupling_kernel && ASCEND_RT_VISIBLE_DEVICES=0 /workspace/venv-ascend950/bin/python repro/TRAP-950-aiv-sid-vector-frame.py` | TileLang ascend 0.1.15 / CANN 9.2.1 / Ascend950PR_9579 (dav-3510) | `.agents/platforms/toolchains/ascend/knowledge/execution-patterns.md`（PL-950-mixed-epilogue 增 sid 读取方式与 trap 条目，与 CONST-950-aiv-sid-capture 互链） |
| ascend 实测（kernel-only，含 launch+sync，warm 后逐次中位）：耦合算子 `grid=2` 11 µs、`grid=64`（780m fp16）20 µs、`grid=288` 74 µs、`grid=512` 130 µs、`grid=1152`（grouped-prefill）154–155 µs、`grid=2048` 316 µs；即 18–154 ns/块 | D | `logs/run_all_final.log`（各例 `launch=`）；`_ssd_chunk_coupling_kernel.py#run_case` 计时口径 | 同 repro 环境，`--level all` | 同左 | `.agents/platforms/toolchains/ascend/knowledge/bottlenecks.md`（首批 950 实测常数；可支撑 Stage 4 的 roofline 对账） |
| 定位 epilogue 分片错误的可解码编码法：令 C 行 `l = e_l`、B 行 `s = 16*(s%64) + n`，则耦合矩阵下三角必须逐位等于 `16*(s%64) + l`（值 < 2^11，fp16 精确表示，零容差）——一次比较即可指出"哪一半行、哪个块"读到陈旧数据 | P | `repro/TRAP-950-aiv-sid-vector-frame.py#operands/reference`；`_ssd_chunk_coupling_kernel.py#_case_mask_semantics` | 同上 | 同左 | pattern-library（调试方法：可解码编码 + 零容差逐元素比对，适用于任何 AIV 分片/sid 定位） |
| 三角掩码算子的三条断言组合（缺一即可能漏检）：① 严格上三角**逐元素精确 0**；② 对角线 == `C 行 l · B 行 l` 且非零；③ 跨 AIV 半块与跨 l-tile 的指定元素 == `C 行 l · B 行 s`（本题取 `(BL/2,BL/2)`、`(BL/2,0)`、`(BL,0)`、`(BL+BL/2,BL/2)`） | P | `_ssd_chunk_coupling_kernel.py#_case_mask_semantics`（L0-3 / L0-7 输出 `upper_nonzero=0/65280`、`diag_err≈1e-2`）；`verify_equiv.py` E1–E4 | `--level L0` | 同左 | `tilelang-op-develop` 的测试模板（掩码/因果类必带三条断言） |
| bf16 输出 1 ulp 边界的实测口径：fp32 累加后单次舍入的两种分组在 bf16 上会让 0.001% 元素落在舍入面两侧（本例 2p7b：16/1.36M 元素差 1 ulp，最大绝对差 0.125，相对 7e-3）；用 `max(atol+rtol*|g|, 1 ulp_out)` 判定并把 `max_ulp` 一并报告，可区分"翻面"与"真错"（本例 `max_ulp=1.00` 全级一致） | D | `logs/run_all_final.log`（L0-5 / L1-3 / L2-2 的 `differing`/`max_ulp`）；`_ssd_chunk_coupling_kernel.py#_compare` | `--level all` | 同左 | `.agents/skills/_shared/standards/`（精度比对标准：ulp 口径） |

### Transferable Lessons（可迁移教训）

- 读 AIV 子块号只能用 `sid = T.call_extern("int64", "asc_get_sub_block_id")` 再 `T.cast(sid, "int32")` 进 `T.SimtVF`；`with T.Vector(vector=2) as sid` 在本工具链**编译通过但运行值错且非确定**，且只在多块网格（`grid` 数十以上）暴露，单块冒烟会全部"通过"。
- 任何依赖 id / 帧 / 分片的写法，设计期 compile-only 探针不能证明其可用：首个上卡动作必须是带"可解码编码"数值断言的冒烟（行号与块号编进操作数，零容差逐元素比对），而不是先跑大 shape 的近似精度对比。
- 与 golden 比对时若 bf16 输出出现 1 ulp 级差异（相对 2^-8~2^-7），先算 `max_ulp`：等于 1 即为 fp32 累加分组造成的舍入翻面（可用 fp64 参照确认哪个更准），不要据此改 kernel 累加顺序。
- kernel 计时必须先 warm-up 再逐次取中位：新编译 kernel 的首次发射含二进制加载与该尺寸首次输出分配，量级可达 100 ms，与稳态（数十至数百 µs）差 3 个数量级。
- Cube+Vector 混合尾部做因果/三角掩码时，E 侧断言要覆盖"跨 AIV 半块"与"跨 l-tile"的指定元素（`(BL/2,BL/2)`、`(BL/2,0)`、`(BL,0)`），否则半块错位或 tile 解码错会与"掩码方向对"混为一谈（严格上三角仍全零，肉眼看不出来）。
- `T.dual_copy` 的 L0C→UB 只能同 dtype（fp32 中转）+ AIV 侧另开 dtype 缓冲做 cast；这条在 `BL=32/64/128` 三档均成立（Q=32/96/128/256 已实测通过）。
