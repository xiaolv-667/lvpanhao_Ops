# `_engram_project_kernel` 设计文档检视报告（Stage 2）

## 检视结论

结论: 通过

## 检视元信息

- 文档: `examples/engram_decode/_engram_project_kernel/DESIGN.md`
- 任务类型: 迁移（harness 子模式；`EngramDecodeFwdOp` 的第 1 个 launch）
- 源算子: `/workspace/TileOPs/src/tileops/_kernels/sequence_modeling/engram/decode.py`（`_engram_project_kernel` 第 25–111 行 + `EngramDecodeKernel.forward` 第 375–417 行）
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 8（逐条内嵌在各维度「建议级」条目里；本报告结论为通过，故不设独立章节）
- 平台: `ascend-950pr-9579` / `ascend`（dav-3510，28 Cube / 56 Vector，UB 253952 B / L1 524288 B / L0C 262144 B / L0A·L0B 65536 B），`statectl platform --task-dir` 实查落库一致

## 机械复核（D-1/D-5）

- **verify_equiv 重跑: PASS（全 EQUIV_PASS，与 §1.6.1 内嵌表逐项一致）**
  - 命令（cwd = 任务目录）：`ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python verify_equiv.py` → 末尾 `RESULT: ALL EQUIV_PASS` + `shift copy bit-exact (float16/bf16): True`。
  - 与内嵌表核对：A1（脚本标签 `A3`，clear_accum）12 行，最大 ulp **0**、>1ulp 率 0，最大相对误差 `9.06e-06 … 8.00e-04`（ktail-519/V）——与 §1.6.1 逐字一致；角点用例 fp16/bf16 均 0 ulp、NaN/±inf 位型一致；A2（脚本标签 `A4`）7 行（ktail-519 两个 m tile 都在），最大 `5.43e-02`（ktail-519 am=16/16）——一致；A3（脚本标签 `T1`）ktail-519 `1.50e-02`、其余 5 条无尾块 0 差异——一致；A4 平移逐位相等 `True`（fp16/bf16）——一致。脚本存在、执行成功、内嵌表与实跑输出无差异。
  - 复核注意（不影响 PASS 判定）：脚本的 `ulp_diff` 用有符号整数序映射，`-0.0` 与 `+0.0` 都映到 0（与 `float ==` 同口径），因此 A1 在 `P_0 = -0.0` 的角点上**不构成位型相等证据**；该点已在维度 8 单独推演（见 S4）。
- **design_calc_check（`python3 .agents/tools/design_calc_check.py --design … --task-dir …`）**
  - `ub_budget` **pass**：1 条 buffer，合计 2.4 KB，重算与声明一致，容量 248 KB。
  - `l1_budget` **pass**：2 条合计 80.0 KB，重算一致，容量 512 KB。
  - `l0a_budget` **pass**：16.0 KB / 64 KB。
  - `l0b_budget` **pass**：64.0 KB / 64 KB（= 100 %，确为紧约束）。
  - `l0c_budget` **pass**：`res` 4.0 KB / 256 KB；另 `block 16x64 x fp32 = 4.0 KB vs 256 KB` 亦 pass。
  - `core_split` **skip**（`insufficient parsed dims {'bm': 16, 'bn': 64}; declared logical cores = 1`）→ **人工复核已核**：逻辑核数 = `mt×2×nt`，按 §5.2/§5.5 的逐 workload 表复算为 8（decode-s1/serving-b1/b8/partial-cache）、16（serving-b4/ktail-519）；物理核数取平台实查记录（`statectl` 返回 `hardware.core_source`：`ai_core_cnt=28, cube_core_cnt=28, vector_core_cnt=56; rtGetAiCoreCount()=28`，与本报告头部实查值一致）；16 ≤ 28 → 单波、无需适配，判定与 `core-split-strategy.md` §1 三要素一致。
  - `r3_metrics` **pass**：四口径（FLOPs / 访存 Bytes / 扫描遍数 / 中间缓冲峰值）齐全。
- **补充一手复核（设计期探针重跑，compile-only，`ASCEND_RT_VISIBLE_DEVICES=""`，cwd = 任务目录，无设备计算；脚本放在 `/tmp/opencode/`，未改动任务目录）**
  - 按 §3.3/§5.6 冻结配置（`block_m=16, block_n=64, block_k=256, num_stages=2`）在 `batch=8, d_mem=512, d_pad=256, mcl=18, fp16` 上编译：**成功**；生成 `__mix__(1, 2)`；AIC 段含 `asc_copy_gm2l1_nd2nz / dn2nz / l12l0a / l12l0b / asc_mmad / asc_copy_l0c2gm`；`asc_mmad(..., 8, 256, 64, …, (kt == 0))`（M = `am` = 8、K = 256、末位即 `clear_accum` 运行期标志）；AIV 段含 `asc_copy_gm2ub_align` + `asc_copy_ub2gm_align` + `PIPE_MTE2↔PIPE_MTE3` 两向 notify/wait 对，行循环 `for b < 8`，**`sid` 被读取但未用于任何守卫**。
  - 同结构在 `batch=17, d_mem=519, d_pad=256, mcl=12` 上编译：**成功**；`asc_mmad(..., M=(17-max((bx>>3)*16,1)), K=(519-max(kt*256,263)), N=64, …, (kt==0))` → M = 16/1（对应 `am`）、K = 256/256/**7**（尾段）；AIV 守卫生成 `((bx & 7) >> 2) == 0 && (bx >> 3) == 0`，正好等价于 `which == 0 && m0 == 0`。
  - 弃选前提复跑：`history_version/design_probe_shift_3d.py` → `FAILED … PlanMTECopy: >2D strided copy not yet supported. src has 3 modes, dst has 1 modes after coalesce`（H6 复现）；`design_probe_shift_placement.build_mixed()` → `MIXEDKERNEL_FAILED: … Need input to be in SSA form dup sid`（X1/X2 复现）。
  - 报错文本在**已安装工具链**内定位：`tilelang/lib` 内 `PlanMTECopy: >2D strided copy not yet supported.`；`src/ascend/op/fill.cc:83` `Ascend L1 fill requires a C0-aligned region`；`src/ascend/transform/mark_scalar_dcache_bypass.cc:143` `ScalarDcacheBypass: vector-valued buffer store not implemented`。
  - 执行上下文说明：平台 `execution.prototype_command` 指定的解释器 `/usr/local/python3.12.16/bin/python3` 缺 `tqdm`（`ModuleNotFoundError`，`import tilelang` 失败），故探针沿用设计期实际可用的 `/workspace/venv-ascend950/bin/python`；未执行任何真机 kernel。

## 检视详情

### 0. 源算子理解与迁移分析 — pass（7 项判定：6 pass / 1 warn；warn 不阻塞）

> 检视方式：亲自 Read 源算子 `decode.py` 全文（第 1–417 行，含 `_engram_project_kernel` 第 25–111 行与 `EngramDecodeKernel.forward` 第 375–417 行），逐项与 DESIGN.md §0.1–§0.7 核对。下表「源码证据」列给 `decode.py` 行号，「DESIGN 证据」列给章节号。

| # | 检查项 | 判定 | 源码证据（`decode.py`） | DESIGN 证据 | 核对结论 |
|---|---|---|---|---|---|
| 0-1 | 语义理解正确性（数学语义 / 规约语义 / dtype 语义 / 边界语义） | **warn** | `_project` 第 46–82 行；`T.clear(acc)` 第 53 行、`T.gemm` 第 62/79 行、`accum_dtype="float"` 第 40 行；`kv` fp32 签名第 90 行 + `forward` 第 401 行 `torch.empty(..., dtype=torch.float32)`；平移第 101–105 行；边界第 57 行（行夹取）、第 81 行（写回守卫）、第 63–79 行（k 谓词填零）、第 99–101 行（`which` 与 `max_conv_len > 1` 守卫）；host 第 388–390 行（左填充）、第 391–399 行（d 维 0 填充）、第 408–409 行（crop） | §0.1（六条语义 bullet）、§0.2（I/O 表） | 数学语义（两组投影 + 左移一格）、规约语义（沿 `d_mem` 分 `block_k` 段串行累加、L0C fp32 累加器、循环前清零一次）、边界四条、host 侧填充/crop 均与源码一致。**唯一偏差**：§0.1「dtype 语义」写「step 段自己把 `kv` 舍入回输入 dtype 再 RMSNorm」，与第 167–169 行相反（`k_local`/`v_local` 是 fp32 fragment，直接读 `kv` 不舍入；只有第 202 行把 `vhat_local` cast 到输入 dtype 写历史末行）。紧随的括号已写出正确事实，且「project 段不得提前舍入」的结论正确、不影响任何设计决策 → 记 warn（建议级 S1） |
| 0-2 | 算法解读完整性（步骤覆盖 / 数据流 / host 逻辑） | **pass** | 索引分解第 97–98 行、e_t tile 装载第 56–57 行与第 65–71 行、W tile 装载第 58–61 行与第 72–78 行、K 段累加第 55/62 行与第 64/79 行、守卫写回第 80–82 行、平移第 102–105 行；`_projection_block_k` 第 246–253 行 | §0.3（6 步表 + 数据流图 + host 侧段落） | 第 46–109 行的全部计算语句都被归入 6 步之一，无遗漏、无臆造；数据流图覆盖 `e_shared`、`w_shared`、`acc`、`conv_state→new_conv_state` 全部 buffer；host 侧逻辑（左填充 / 右填充 / crop / block_k 选取）已单列 |
| 0-3 | 优化手段覆盖完整性（O1–O11） | **pass** | O1 第 50–51 行；O2 第 62/79 行；O3 第 52–53 行；O4 第 55/64 行（`T.Pipelined(num_stages=4)`）；O5 第 57 行（`block_m=16` + `T.min` 夹取）；O6 第 67–78 行；O7 第 99–105 行；O8 第 81 行；O9 第 94–96 行（三维 grid）；O10 第 102 行；`threads=128` 第 95 行 | §0.4（O1–O11 表） | O1–O10 全部可在源码定位并逐项标注硬件依赖；O11 明确标注「来自 950DT 参考、非源算子」，层级划分正确，未把实现层手段误判为算法层 |
| 0-4 | 耦合性判定合理性（四态处置 + 依据） | **pass** | 三维 grid 第 94–96 行；`threads=128` 第 95 行；平移逐元素并行第 102 行；行夹取第 57 行；k 谓词填零第 63–79 行 | §0.5（13 条处置表） | 13 条均有四态处置 + 依据，且与平台约束不矛盾：三维 grid / `threads` 判「重新设计 / 舍弃」有 `references/programming.md` 第 13 行（`T.Kernel(N)` 一维、签名无 `threads`）为证；L1 就地补零不可表达有 `fill.cc:83` 报错文本为证；4 个「重新设计」项 #8/#9/#10/#11 分别对应 §0.6 R2/R5/R1/R4；舍弃项（#7 夹取、#12 `threads`）均写理由 |
| 0-5 | NPU 重设计可行性（R1–R5 三要素） | **pass** | R1 承接 grid 第 94–96 行；R2 承接第 63–79 行；R3 承接第 57 行 + 第 80–82 行；R4 承接第 99–105 行；R5 承接第 80–82 行 | §0.6（R1–R5 各「源方案 → NPU 新算法 → 语义保持论证」） | 五项的语义保持论证在「有效区间 / 死行消除 / 区域写回 / 纯拷贝」四点上是精确等价（见维度 8 推演）。一手编译核对（本次，§3.3/§5.6 冻结配置）：生成 `asc_mmad` 实参 M=8（b8）、K=256/256/**7**（`d_mem=519` 尾段）、末位 `clear_accum=(kt == 0)`；AIV 守卫生成 `((bx & 7) >> 2) == 0 && (bx >> 3) == 0`，等价于 `which == 0 && m0 == 0` → 证明 R2/R3/R4 的边界与落点确由硬件参数与守卫表达，与文档叙述一致；R1 的块到 tile 映射为双射、写集互不重叠 |
| 0-6 | 设计与决策一致性（§1–§7 ↔ §0.5/§0.6） | **pass** | 与源码逐条对照：第 95 行 `threads=128` 已删（§3.2 第 1 行无 `threads`）、第 94 行三维 grid 改 1-D 整数分解（§3.3 第 686 行）、第 57 行夹取改为 `am` 区域（§3.3 第 698 行）、第 80–82 行逐元素写改为 L0C→GM 区域拷贝（§3.3 第 669 行）、第 102–105 行逐元素平移改为 `T.serial(batch)` 的 GM→UB→GM（§3.3 第 672–675 行） | §1.4（首句标注「与 §1.6.0 选定的 C1 一致」）、§3.2、§3.3、§4.3、§4.4、§6.1、§7.2 | §0.5/§0.6 的每一项处置都在 §1.4/§3.3/§4/§6/§7 落地，未出现「§0 判重设计、§3 照抄源方案」的脱节；§4.4 搬运路径与 §7.2 同步点与重设计后的混合结构一致 |
| 0-7 | golden 独立性（不复刻 NPU 算法） | **pass** | 语义出处第 46–79 行（两组投影）与第 102–105 行（平移）；源仓参考 `workloads/sequence_modeling.py::engram_decode_step_torch` 第 420–489 行 | §8.1（`golden_engram_project` / `golden_engram_decode_full`） | golden 用 `e_t.float() @ W_K.float()` + `state[:, 1:, :]` 直接移植源仓 torch 参考，不含区域截断、`clear_accum`、`shift_ub` 等 NPU 实现细节；末行 NaN 哨兵属契约断言（断言「本 kernel 未触碰末行」），不构成对 §0.6 算法的自证 |

**维度 0 小结**：7 项判定 6 pass / 1 warn（0-1 的 §0.1 措辞与第 167–169 行相反，建议级 S1，已在维度 0 内给出改法）；语义、算法、优化手段、耦合性、重设计、一致性、golden 独立性均有源码行号 + DESIGN 章节号双向证据，**无阻塞级问题**。

### 1. API 可行性 — pass

- 逐条核对**实际语言模块** `/workspace/venv-ascend950/lib/python3.12/site-packages/tilelang/ascend/language/`：`alloc_l1 / alloc_l0a / alloc_l0b / alloc_l0c / alloc_shared` 全部存在（`allocate.py:26/37/46/55/15`）；`annotate_buffer_versions` 存在且文档写明「`num_versions >= 2` 用固定版本数」（`annotations.py:19-31`）；`T.copy` 的 `transpose` 文档作用域确认是「Ascend GM→L1 only … Emit dn2nz」（`copy_op.py:270-272`），与 §3.2 第 4 行、§3.5.1「`transpose` 只对 GM→L1 有效」一致；`T.gemm` 文档确认「L0 operand regions specify the effective MAD M/N/K … trailing matrix dimensions must start at zero and describe a compact tile；L0 allocations and producer copies may be padded for hardware alignment（例：transposed FP32 load 可拷 K32 而 GEMM 只消费 `[:, :24]`）」（`gemm_op.py:186-190`）——正是 §3.2 第 5 行与 §0.6 R2 依赖的机制，文档层面成立。
- `T.Kernel(N)` 一维 + 无 `threads`、`T.MixedKernel` 属「源码确认」而非已验证路径，与 `references/programming.md` 第 13/17 行一致。
- §3.4 标 ✅ 的 API 均由探针与文档双证；标 ❌ 的两条（`T.MixedKernel`、3-D 区域拷贝）我各自复跑一次，报错原文复现（见机械复核）。
- 编程模式：`hybrid` 是平台 `programming_modes.choices` 的合法键；偏离 `migration_default = simd` 已在 §2.2 写明并回报 conductor，属 conductor 决策面而非设计缺陷。
- 建议级 S8（论据强弱，非可行性问题）：§2.2 的「≈16×」独立复算成立——`2·batch·d_mem·d_padded = 8.39 MFLOP`（b4），`378 / 23.6 TFLOP/s` 峰值假设下 `0.022 µs vs 0.36 µs`，比值 16.0 ✓；但相对 §1.6.0 自己给出的「发射项 ≈5 µs 主导」下界，0.34 µs 增量只占 ~7 %，单凭「16×」不足以否决。真正决定性的是 R4 已写的另两条——「无本地已验证的 rank-1 融合模式」与 batch=1/2 时纯向量路径只有 `d_padded/block_n` 个并行单元。建议把结论顺序调整为「并行度 + 已验证性为主、算力比为辅」，以免后续被质疑「用未实测峰值做否决」。

### 2. 内存层级规划 — pass

- 机械复算全 pass（UB 2.4 / L1 80 / L0A 16 / L0B 64 = 上限 / L0C 4 KB），且 L0B 的「100 %」不是纸面值：我按冻结配置编译成功，生成代码里 `w_l0 + (version_counter & 1) * 16384`（half 元素）= 每版本 32 KiB × 2 = 65536 B，确为容量上限且**编译通过**（`block_n=128`/`block_k=256` 的越界情形见 §9.2 E2）。
- 搬运路径 §4.4 四段（`GM→L1→L0A/L0B→mmad→L0C→GM`、`GM→UB→GM`）在生成代码中逐段对上（`asc_copy_gm2l1_nd2nz` / `dn2nz` / `l12l0a` / `l12l0b` / `asc_mmad` / `asc_copy_l0c2gm` / `gm2ub_align` / `ub2gm_align`）。
- 每个 buffer 都有层级标注（§4.3），`shift_ub` 在 `T.Kernel` 内分配、实际落在 AIV 私有 UB（生成代码中它是 AIV 段唯一 buffer）。
- 建议级（不阻塞、但需在 Stage 3 留意）：L0B 恰好等于容量上限，任何「版本数/`block_k`/`block_n` 上调」都会越界；设计已在 §5.3、§9.2 E2 写明，此处仅记录为 Stage 3 的编译首查项。

### 3. Tiling 策略 — pass

- Block 划分有具体值与理由（`block_m=16` 取 Cube M 分形粒度；`block_n=64` 由 L0B 与 grid 占用折中；`block_k=256` 由 L0B 反解 `min(65536/(64·2·2), 256)`——独立复算 = 256 ✓）。
- 对齐：`d_padded = align_up(d, ALIGNMENT)`，`ALIGNMENT = 256` 在抽取件 `_engram_decode_fwd_kernels.py:16` 实查确认，故 `d_padded % 16 == 0` 恒成立；`d_mem` 不做对齐而用 `ak` 表达（`d_mem=519`）——生成代码实测 `K=(519-max(kt*256,263))` = 7 ✓。
- 非整除处理：M/K/N 三向尾块都有方案（§6.4），且 M/N 尾块我是从生成代码看到实参的（`M=(17-max((bx>>3)*16,1))`、`N=64`、`l0c2gm` 的 `nburst=am`），不是纸面声明。
- **分核策略三要素齐全**（对照 `core-split-strategy.md` §1/§2.2）：① 逻辑核数 `mt×2×nt` 逐 workload 列表（8/16）；② 物理核数按平台口径实查并记录查询代码与返回值（`torch.npu.get_device_properties` 片段 + 平台落库 `core_source`，且与 `statectl` 当前返回值一致，非文档假设值）；③ 规模判定「逻辑 ≤ 物理 → 无需适配」并给出依据（最坏 16 块单波驻留 28 Cube 核），且说明了「为何不再增大并发」。核数与 block 取值自洽（`nt = ceil(d_padded/block_n)`），无核内串行循环、无动态 shape 依赖（§4.6 全编译期常量）。
- 建议级：`design_calc_check` 的 `core_split` 因 dims 解析不足 skip（人工已核如上）；若希望后续可机检，可在 §5.5 用 `mt/nt` 分行形式书写（不影响本结论）。

### 4. 技术约束检测 — pass

- 三维 Kernel：本设计主动改为 1-D（`programming.md:13`），故无 block_metadata 需求。
- GEMM 非整除：已在维度 3 处理；`T.clear` L1 补零不可用（`fill.cc:83`）→ 改有效区间，路径自洽。
- L0C 溢出：`res` 4 KB / 256 KB，无风险。
- 活跃缓冲、多缓冲版本与累加类型均有预算表（§4.5），且 2 版本是**实际生效**的（见维度 5 修正），预算与实际一致。

### 5. 循环与同步 — warn

- 循环结构齐全（§6.1：block 级并行 / K 收缩 / cache 行 / 元素级无），API 选择明确（`T.Kernel` / `T.Pipelined` / `T.serial`），尾块逻辑在 §6.4 单列；§6 与 §1.6.2 一一对应（无额外逐元素循环）。
- 同步与模式匹配：§7.1 选自动同步（`allow_autoschedule` 默认开启）、不写手工 flag；§7.2 的 7 类同步点我在生成代码中逐类对到：`PIPE_MTE2→PIPE_MTE1`、`PIPE_MTE1→PIPE_M`、`PIPE_M→PIPE_FIX`、`PIPE_M→PIPE_MTE1`（版本化 event），AIV 段 `PIPE_MTE2↔PIPE_MTE3` 两向配对均存在；AIC/AIV 之间确实无同步点。
- **建议级 S2（事实性修正，必须回写以免误导 Stage 4）**：§6.3、§9.1、§11.2 #6 声称「AutoSchedule 打印 `did not enable multi-buffering …`（探针 H1/H2 每次编译都出现）」「当前得到的是顺序 K 循环」「版本计数实际为 1（L1/L0 占用可能低于 §4.5 的 2 版本预算）」。我按 §3.3 冻结结构编译两次（b8 fp16、ktail-519），**均未出现该警告**，生成代码为 `e_l1_version_counter_2 = 0; … wait/notify(PIPE_*, event_t(counter & 1)) … counter++`，且 `w_l0 + (counter & 1) * 16384`、`buf_dyn_l1 + (counter & 1) * (16384/4096)`——即 **2 版本已生效、L0B 真实占用 64 KiB**。该警告可复现于 `T.MixedKernel` 结构（X1/X2 探针输出四条 buffer 警告），与本设计采用的普通 `T.Kernel` 不是同一结构，属**张冠李戴**。影响：不影响正确性；但「版本计数实际为 1」会让 Stage 4 误以为 L0B 有余量（实际 100 % 打满，`block_k` 不得上调），且 §11.2 #6 的「多缓冲是否可打开」是伪命题。建议改为「普通 `T.Kernel` 下 2 版本生效（生成代码版本计数器 &1 为证）；多缓冲拒绝警告只出现在 `T.MixedKernel` 结构；L0B 因此在运行期就是 64 KiB 满额，Stage 4 只能向下调（`num_stages=1`/手写 `order`/`stage`）」。

### 6. 验证方案 — pass

- §8.1 给出 PyTorch 参考草案两段（分段 golden + 整算子 golden），分段 golden 独立于 NPU 算法（见维度 0 golden 独立性），并给出 3 条分段断言（kv 数值 + 平移逐位 + 末行哨兵未被触碰）。
- §8.3 给出 L0 门槛计划：具体 shape（decode-s1 / serving-b1 / serving-b8）、dtype、覆盖点（累加链、bf16、`am=8`），以及 L1/boundary 清单（`max_conv_len=1`、`d_padded=272`、`d_mem<block_k`、ktail-519）。
- 精度标准按 dtype 给足：`kv` fp32 `1e-3/1e-3`；平移逐位 0；op 级 fp16 `0.05/0.05`、bf16 `0.1/0.05`——与 `examples/TileOPs/tests/ops/test_engram.py::_tol_decode`（第 67–70 行实测 `{"atol":0.05,"rtol":0.05}` / `{"atol":0.1,"rtol":0.05}`）逐值一致。
- **harness 参考缺陷（§9.2 E7）的归属判定：不是本设计的问题，须交回 conductor 修 Stage 0 工件。** 事实核对：`examples/TileOPs/tileops/workloads/sequence_modeling.py::engram_decode_step_torch` 在 `max_conv_len > L` 分支（第 **471–479** 行）返回 `torch.cat([conv_state, v_hat_norm], dim=1)`，即 **L+1 行**，与其自身 docstring（"`max_conv_len` long either way"）和 manifest 契约（`new_conv_state: [batch, max_conv_len, d]`）矛盾。后果：`test_engram_decode_partial_cache`（第 444 行起）第 **473** 行断言 `new_conv_state.shape == ref_state.shape == (batch, max_conv_len, d)`，在 `L=5, max_conv_len=12` 时 `ref_state.shape = (2, 6, 256)` → **无论 kernel 是否正确都会失败**，Stage 5 集成必然红。该缺陷位于 Stage 0 抽取的 harness 参考实现，设计侧无法修复（设计已在 §8.1 用「先左填充再调参考」规避，并在 E7 记录，处理正确）。**建议交回 conductor**：修 `engram_decode_step_torch` 的短 cache 分支为「先 `F.pad` 到 `max_conv_len`，再取 `cat([padded[:, 1:, :], v_hat_norm], dim=1)`」（与 kernel 写的 `0..mcl-2` 行 + step 写末行完全对齐），或同步修正第 473 行断言的期望；修好后 `test_engram_decode_partial_cache` 才具备判别力。
- 建议级 S5（引用行号，随 E7 一并修）：E7 记「第 465–470 行」实为 471–479；记「test_engram.py 第 464 行」实为 473。

### 7. 完整性与一致性 — warn

- 章节齐全（§0–§11，迁移类 §0.1–0.7 全在）；无残留占位标记（占位类关键词全文检索命中 0）；§3.5.3 引用的 6 条路径**全部真实存在**（`examples/shared_expert_mlp/_gemm_basic_kernel/DESIGN.md`、`knowledge/gemm-tiles.md`、`PL-950-cube-tile.md`、`PL-950-vector-execution.md`、`examples/moe_permute_align/.../TRAP-950-shared-shift-copy.py`、`/workspace/TileOPs-ascend950/.../decode.py`），§10.1 列出的 4 个探针脚本亦存在。
- 内部一致：API 映射（§3.2）↔ 内存规划（§4.3/4.5）↔ Tiling（§5.6）三处的 buffer 形状/版本数/`block_*` 相互一致，§1.6.3 的布局决策与 §3.3/§4.3/§6 一致。
- **建议级 S5（引用定位）**：两处行号/文件偏差（同一处见维度 0、维度 6）。二者都不改变结论，但属「同类引用真实」的可修项：① ktail-519 → `examples/TileOPs/tileops/workloads/sequence_modeling.py:338`（类 `EngramDecodeWorkload`）；② E7 的参考行 471–479、测试断言行 473。
- 建议级 S6（引用可加强）：§9.2 E1 把「`asc_mmad` 对非 16 倍数 K 的掩蔽」列为未证实风险，但平台已验证条目 `knowledge/gemm-tiles.md` 第 17 行明确记录了 `(17,19,35)` 组合「K 切成 32+3」并「每种组合运行两组 M/N/K」——非 16 倍数 K 尾块在该工具链（TileLang ascend 0.1.15 / CANN 9.2.0）已是**验证过的形态**。建议引用该条目把 E1 的风险等级下调（保留 L0 `ktail-519` 用例作为回归判据，回退方案不动）。

### 8. 算法优化分析 — pass

**§1.6 章节存在性 — pass**：1.6.0 / 1.6.1 / 1.6.2 / 1.6.3 四个子节齐全，且各自有实质内容（非只有标题）。

**算法调研存在性与完整性 — pass**：R1 候选表 9 行，**基线（源算子）在第 0 行**且入 R3 对比；R2 给出明确结论（「有，且它就是基线结构」——投影沿 `d_mem` 单遍 running 累加、中间缓冲 O(tile)）并以结构判据支撑，不是空白；R3 四口径齐全（FLOPs / 访存 Bytes / 扫描遍数 / 中间缓冲峰值），口径写清了「只计本 launch 的 GM 流量」「遍数 = 同一数据重复读」「峰值只计片上驻留」；R4 对 8 个候选逐项过 6 项检查清单；结论段写明选定 C1、关键依据、以及「无更优替代」的**调研范围**（参考表 R1 #1–#4、`gemm-tiles.md`、同平台 `_gemm_basic_kernel` DESIGN、`algorithm-candidates.md`）；源算法优化意图逐项承接（O1→L1、O2/O3→L0A/B/C、O4→Pipelined、O7→同 launch AIV 平移、O9→1-D grid；O5/O10 的意图由区域拷贝承接），无静默丢弃。

**调研结论正确性（独立复核）— pass**

- 复杂度复算（serving-b4）：FLOPs `2·4·1024·512·2 = 8.39 M` ✓；W 流量 `2·1024·512·2 = 2.10 MB` ✓；`e_t` 两遍 `2·4·1024·2 = 16 KB` ✓；`kv` `2·4·512·4 = 16 KB` ✓；平移 `4·19·512·2 = 78 KB` 读 + 78 KB 写 = 156 KB ✓（故 2.29 MB 合计自洽）。中间缓冲峰值 L1 80 K / L0B 64 K / L0C 4 K / UB 2.4 K 与 §4.5 一致，**无中间 GM 往返**（本 launch 只有 `kv`/`new_conv_state` 两个契约输出），故无漏计往返的问题。
- **建议级 S3（自洽性小偏差，可解释故不判 fail）**：R3 的 C1 行写「2.29 M（与行 0 同；A1–A3 只减无效工作量，不减有效流量）」，但 §1.6.1 A4 自己写明「重复写使平移流量 ×2（156 KB → 312 KB）」。两者相加后 C1 应为 **2.45 MB**（2.29 − 0.156 + 0.312）。差异 156 KB ≈ b4 流量的 6.8 %，不改变「发射项 ≈5 µs 主导」的结论与候选排序（C2 只差 16 KB、C1/C4 并列的判定不受影响），故不判 fail；建议在 R3 的 C1 行改成 2.45 M 或加脚注，避免与 A4 的收益量化对不上。
- 负向断言复核：「无更优替代」的限定集合可核对——`knowledge/algorithm-candidates.md`（平台 `knowledge.algorithm_candidates` 指定的候选表本体）只有 17 行，GEMM 条目为「显式 L1/L0 操作数、输入布局转换和 K 分块累加」，正是选定方案；`algorithm-research.md` §5 说明候选表本体已迁至该文件，故不存在被漏掉的命中行。`references/algorithm-research.md` §3 的 R1/R2 结构判据亦逐条过了一遍：本算子无除法/超越函数（投影段无归一化）、无可提取公共子表达式、收缩轴只有 running sum → 在线形式即 K 段累加，结论与文档一致。**未发现被漏掉的已知候选**（权重拼接、双累加器、跨 launch 融合、单元替换四种已覆盖；transposed-N tiling 属 §1.6.3 (ii) 的 N tile 维度权衡，且受 `kv` 的 `(2,batch,d_padded)` 契约与 L0C→GM 不能任意转置写入约束，不构成新的等价算法）。
- 并列候选裁决：C1 与 C4 在访存上并列（2.29 MB），设计未纸面拍板单一结论，而是给出 R4 的量化负向依据 + 「C4 保留为 Stage 3/4 小 shape 备选方向」；`block_n=64 vs 32` 的判定裕度落在未实测常数上，§1.6.3 末段给了**主选 + 备选 + 裁决计划**（代表 shape serving-b1/b8、指标 `Task Duration`、阈值 5 %、按 shape 分派可行性、回写路径）。实验裁决三件套成立。

**调研→设计一致性 — pass**：§1.4 首句即标「与 §1.6.0 选定的 C1 一致」，其 4 条与 R1–R5 对应；§3.1 的 6 个步骤以 §1.6.1 优化后公式为输入（步骤 3 即 A1 的「首段即初始化」）；§6 的循环表与 §1.6.2 结论一致（判定向量化的点不再出现逐元素标量循环）；§0.5/§0.6 的处置与调研选族（Cube + 同 launch AIV 搬移）一致。

**数学等价优化完整性 — pass**：A1–A4 每项四要素齐全（原式 → 优化后公式 → 等价性论证 → 收益量化），且分析了「无进一步优化空间」的四条依据（无除法/超越、无公共子表达式、无可降代换、访存已到候选下界）。A2/A3 的收益量化可复算（batch=1 省 `15·d_mem·2 B`；ktail-519 的 3 段 × 16 × 256 = 12288 次逐元素选择 ✓）。

**等价性论证独立推演 — pass（含 1 条建议级修正 S4）**

- A2（死行消除）：源实现的夹取仅为让越界地址合法、随后被写回守卫丢弃（`decode.py:57,81`）；本方案 M 向无越界地址，`am` 同时进装载/`gemm`/写回区域 → 精确等价 ✓（机器验证的死行填 `NaN/±inf/-0` 用例 7 行全 0 ulp，我复跑一致）。
- A3（K 尾块谓词 → 有效区间）：`Σ_{k<d_mem}` 与「补零到 `block_k` 边界后求和」在输入侧补零的情形下逐值相同（源两侧同时补零，`decode.py:67-78`）；`ak` 截断把乘积项集合精确限制到源的有效集合 → 精确等价 ✓。残余的硬件掩蔽假设已在 E1 披露，且见 S6（平台已验证 K=32+3）。
- A4（平移合并）：纯 dtype 拷贝、无算术 → 逐位等价 ✓（`torch.equal` on int16 view 复跑 True）。重复写的三条前提（本 launch 无读者 / `torch.empty` 新分配不与 `conv_state` 别名 / 两 AIV 写相同字节）经生成代码核对成立：AIV 段 `for b < batch` 只做 `gm2ub + ub2gm`，写的是 `new_conv_state`，本 launch 无任何读；`kv` 与 `new_conv_state` 均在 `forward` 内新建（`decode.py:401-405`）。
- A1（`clear_accum=(kt==0)` 代替显式零初始化）：**建议级 S4**——`0 + x ≡ x` 只在 `x ≠ -0` 时位型成立；`0 + (-0) = +0`，而「首段直接初始化」给出 `-0`，故 `P_0 = -0.0` 的角点上两者**位型不同**（设计的原文「`0 + (-0) = +0` 与「直接取 P_0」在 fp32 位型上一致」自相矛盾），且 `verify_equiv.py` 的 `ulp_diff` 把 `±0` 映到同一值（与 `float ==` 同口径），机器验证在这一点上不构成证据。**独立推演结论：数值等价、IEEE 下仅 `±0` 符号位可能不同，无 NaN/±inf 分歧**；该差异对本算子的所有消费者不可观测（step 段 `kv → k_local` 后平方/求和/`rsqrt`，±0 只会传播到零值结果；op 级比对为 `allclose`；§8.3 的逐位断言只作用于平移行，不作用于 `kv`）。因此 A1 的「采纳」结论成立、不影响精度门槛，建议把论证改写为「数值等价（±0 位型可能不同，不可观测）」或对 `kt==0` 保留一次 `T.clear`。**不判 fail 的理由**：等价性在可观测层面成立，缺陷仅在论证措辞与其对机器判据的描述（「fp32 位型逐位相同」被 `±0` 归一化削弱）。
- 收益量化：A1 `每 block 省 4 KiB L0C 清零`、A2 `batch=1 省 15/16 的 e_t 读与 L0C 写`、A3 仅 ktail 生效、A4 省一次 launch（≈5 µs，未实测假设）——量级与 §1.6.0 的发射项主导结论一致。

**向量化替代覆盖完整性 — pass（与 §3.3/§6 交叉核对）**：§3.3 伪代码的全部循环/标量点共 7 处（块索引分解、`am/an/ak`、K 段累加、e_t 装载、W 装载+转置、矩阵乘、平移行循环），§1.6.2 表逐点覆盖且编号可一一对上；其中 3 处已向量/DMA 化（装载/写回/平移暂存 = 3 条 DMA + Cube），4 处保留并给出类别化理由（block 级索引 ×2、tile 级顺序依赖 ×1、平台 API 缺失 + 逐元素退化 ×1）。§6.1 明确「元素级：无」，与 §1.6.2 的「0 处标量逐元素循环」一致，无遗漏点、无未论证的额外标量循环。

**不可替代理由充分性 — pass**：#1/#2 属 block 级索引（每 block/每 K 段一次）；#3 是循环携带累加链（改并行需 split-K + 跨核归并，§1.6.0 R3 已量化负向）；#7 的两条替代各有具体佐证（① API 缺失：我复跑 H6 得到原文 `>2D strided copy`；② 逐元素退化 4864 次标量读写 + UB→UB 错位拷贝反例 `TRAP-950-shared-shift-copy.py` 真实存在）。理由具体到本算子，非泛泛而谈。

**替代 API 可行性 — pass**：`T.copy(transpose=True)`、`T.copy` 区域、`T.gemm(clear_accum=…)`、`T.serial` 均在已安装模块的文档或生成代码中有佐证（生成代码分别出现 `asc_copy_gm2l1_dn2nz`、`asc_mmad(..., (kt==0))`）。

**向量化轴与布局决策完整性 — pass**：
- (i) 平移的轴决策 4 候选（I/O 原生 `d` 轴 / 核内 `d`-major / host permute / 逐元素），逐候选给量化（转置链 312 KB > 本体 156 KB；host 两次 permute 百 µs 级；逐元素退化），采纳项给了完整性/整除性（`an=64` fp16=128 B=4 个 32 B block，无尾 lane）、累加链形态、repack 代价 0、UB 影响 2432 B ≈ 1 %。**未把 I/O 契约当内部布局依据**：明确写「I/O 原生布局」是契约、核内布局是设计变量，且方案 2/3 是真正的布局变更候选。
- (ii) N tile 与 GEMM 布局 4 候选（NT 64/256、NT 32/256 备选、NT 128/128、NN），L0B 预算与 grid 占用逐行量化；弃选 NN 的「fp32 NN 必须走 NT」有 `gemm-tiles.md` 第 1 条与 `gemm_op.py` 文档支撑（我核对了 `gemm-tiles.md:21` 的 B=(K,N) 走 `transpose=True` + `transpose_B=True` 的已验证条目）。
- 实验裁决三件套：备选结构达到可实现深度（`e_l1(16,256)` 8 KiB、`w_l1(32,256)` 16 KiB、`w_l0b(16 KiB×2)`、`res(16,32)` 2 KiB、`shift_ub` ≤1.2 KiB、grid b1=16/b4=32 并指出 b4 需把 `block_k` 降到 128），阈值 5 %、代表 shape、回写路径、按 shape 分派可行性（`lru_cache` 键）均具备。
- 建议级 S7（可加强，不影响正确性）：设计只评估了 `with T.Vector(vector=2) as sid` 这一种拆分写法（并实测其丢 `PIPE_MTE2→PIPE_MTE3` 同步），但**已安装模块还提供 `T.copy(..., sub_blockid=…)`**——`copy_op.py:238/276` 文档写明「Sub-block id that routes the copy to one of the AIV sub-blocks」，是保留 copy lowering（因而可能保留同步）的官方入口。重复写本身无害（pattern-library `PL-1.13-aiv-dup-subid-split` 已把「双 AIV 默认重复执行 Vector 程序」记为 verified 行为，同条目实测 subid 分片收益 −28~−36 %），故建议把 `sub_blockid` 列为 Stage 4 的 A/B 项，而不是在 §0.6 R4/§11.2 #3 里只留 `T.Vector` 一条路。

**布局决策与实现一致 — pass**：§1.6.3 的 `d` 轴 + offset 0 的 GM↔UB 拷贝与 §3.3 的 `shift_ub[nshift, block_n]`、§4.3 的 buffer 表、§6.2 的行循环一致；GEMM 的 NT 布局与 §3.3 的 `T.copy(..., transpose=True)` + `T.gemm(transpose_B=True)` 一致；生成代码的 `dn2nz` 出现说明转置确在搬运通道内完成（非 GM 级 transpose）。

**与下游章节一致 — pass**：§3.1 步骤 3 用 A1 优化后形式、步骤 2 用 `am/an/ak`；§6 不再出现被判定向量化的逐元素循环；§6.4 的尾块表与 §0.6 R2/R3 同口径。

---

**检视小结**：维度 0–8 全部无阻塞级 fail；8 条均为建议级（S1 措辞、S2 多缓冲事实修正、S3 R3 字节自洽、S4 A1 的 ±0 论证、S5 两处引用定位、S6 E1 风险等级、S7 `sub_blockid` 备选、S8 §2.2 论据排序）。设计期探针的关键结论（自动 `__mix__(1,2)` 提升、L0B 满额可编译、K 尾 `ak=7` 进 `asc_mmad`、3-D 拷贝与显式 MixedKernel 不可用）经本次一手复跑全部成立；`verify_equiv.py` 与 `design_calc_check.py` 机械复核一致。**harness 参考缺陷（`engram_decode_step_torch` 短 cache 返回 L+1 行）不是设计问题**，已按维度 6 给出 conductor 侧的修复方案；它会让 Stage 5 的 `test_engram_decode_partial_cache` 必然失败，须在进入 Stage 5 前修 Stage 0 工件。
