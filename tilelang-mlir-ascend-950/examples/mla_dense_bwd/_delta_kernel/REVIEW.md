# _delta_kernel 设计文档检视报告（v2 / revision 1 完整重检）

## 检视结论

结论: 通过

## 检视元信息

- 文档: `examples/mla_dense_bwd/_delta_kernel/DESIGN.md`（1170 行，v2 / revision 1，2026-11-23）
- 上一版: `examples/mla_dense_bwd/_delta_kernel/history_version/design_v0.md`（v1 first_design，首轮检视 3 阻塞 + 7 建议）
- 任务类型: 迁移（TileOPs GPU `_delta_kernel` → `ascend-950pr-9579` / `target="ascend"`，harness 子模式：只迁移 `MLADenseBwdOp` 三个 launch 的第 1 个）
- 源算子: `/workspace/TileOPs/src/tileops/_kernels/attention/mla/bwd.py` 第 37–76 行 `_delta_kernel`；消费端契约第 134/148/217 行（`_bwd_kernel`）
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 平台: `platform.knowledge` 指向 `algorithm-candidates.md`；`ruleset = .agents/platforms/toolchains/ascend/workflow.md`；`stage_skills.review = tilelang-ascend`；
  `execution.mode = device`，本次按平台约定只做**无卡**编译探针与 CPU 数值实验（`ASCEND_RT_VISIBLE_DEVICES=""`，从不 launch）
- 阻塞级问题数: 0
- 建议级问题数: 9

## 机械复核（D-1/D-5）

- **verify_equiv 重跑**: `EQUIV_PASS`（**与 §1.6.1 内嵌表逐格一致，无漂移**）
  - 命令：`cd examples/mla_dense_bwd/_delta_kernel && /workspace/venv-ascend950/bin/python verify_equiv.py`（exit 0）
  - 逐项汇总（4 项 × 12 用例 = 48 行）：A1 `max ulp(cand,ref)` **3310** / A2 **2798** / A2b **2798** / A3 **2798**，
    `max violation`（尺度化口径）= **0.0000**，标准容差违反率 **0.0156** 仅出现在 `large/fp16`（= 该用例 1/64 行 = 0.015625，
    即 §1.6.1 表注的「抵消行」），`nf_mis` 全 0 —— 与 §1.6.1 表的四行数字**完全相同**。
  - **A2（实现分组）vs A2b（v1 引用的 2×`vcadd` + 标量加）逐格比对**：12/12 行**逐列相同**（含 `ulp(cand,base)` 最大值 2560、
    `random/fp16/D128` = 322/2560、`large/fp16` = 2798/512），与 §1.6.1 注的断言一致；本检视另用 `torch.equal` 比对两分组的
    int32 位模式 → **bitwise True**（12/12 用例），故「两分组在 `nvec==1` 下逐位恒等」有机器证据。
  - **基线数字独立复算**（脚本只打印 `ulp(cand,ref)`/`ulp(cand,base)`，未打印 `ulp(base,ref)`；§1.6.1 引用的「基线 2238 ulp」
    本次独立复算确认）：`random/fp16/D128` 基线 **2238** / 实现分组 **322**；`large/fp16` 基线 **3310** / 实现 **2798**；
    `random/bf16/D128` 64/64；`large/bf16` 189/67 —— 与 §1.6.1 A2 行「322 vs 基线 2238」「树形更接近真值」一致。
  - A1/A3 的 `ulp(cand,base)` 在 **12/12 用例全为 0**（逐位恒等），与「A1 只移动存放位置」「A3 纯数据搬运」的论证一致。
- **design_calc_check 复算**: `python3 .agents/tools/design_calc_check.py --design examples/mla_dense_bwd/_delta_kernel/DESIGN.md --task-dir examples/mla_dense_bwd/_delta_kernel`
  - `ub_budget`: **skip**（"no per-level buffer tables parsed from the memory plan (advisory checker; verify manually)"）
    → **机械跳过，人工已核**：设计口径 32768(`o_ub`×2)+32768(`do_ub`×2)+2048(`d_ub`)=67584 B；
    但本次 dump 的生成代码显示 `d_ub` 也被编译器加了第二版本（`(it & 1) * 512` floats）→ 实占 **69632 B（68 KB）**；
    两种口径均 ≤ 253952 B（1.7× 膨胀：114893 / **118374** ≤ 253952）→ 容量结论不变（见 §2 与建议 1）。
  - `l0c_budget`: **skip**（"no Ascend Cube GEMM call found in the design pseudocode"）→ **不适用**（纯 Vector，不分配 L0C）。
  - `core_split`: **skip**（"insufficient parsed dims {}; declared logical cores = 3" —— 解析器把「三个 launch」的 3 当逻辑核数）
    → **机械跳过，人工已核**：按 §5.2 规则对 §5.2 表内 11 行逐行复算 `R / nt / nsp / tpb / grid` 全部自洽
    （`ds-v3-4k` 64/64/2/32/256；`ds-v3-b2-8k` 128/1/128/256；`kimi-k2-8k` 128/4/32/256；`ds-v2-lite-3k` 48/4/12/256；
    manifest smoke 8/8/1/64；B=2,S=256,H=8 4/4/1/64；D_v=64 op smoke 4/4/1/16；S=200(R=50) 4/4/1/32；S=128 2/2/1/16；
    S=65 2/1/1(+tail)/8；S=1 1/1/0/1），行覆盖 `nsp·tpb·R = nfull·R = S` 全部成立。
  - `r3_metrics`: **pass**（"all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak)"）。
- **额外机械核对（本检视自行执行，非 skill 强制项）**
  - 重跑 `history_version/design_probe_compile_v2.py all`：**F1/F2/F4/F5/F6/F8 全部 COMPILE OK**；`f3` **COMPILE FAIL**、
    `f7` 单独跑 **FAIL**（`RuntimeError: Immutable variable `acc` is used outside its defining region!`，且工具链使进程以 exit=1 中止，
    与 §3.3 注 2「F7 需单独跑」的记录一致）→ 三条阻塞的**正向 + 负向对照证据全部复现**。
  - dump 生成代码逐参数核对（`F2`/`F4` 全源码）：装载 `asc_copy_gm2ub_align(dst, src, 64, 128, 0,0,0, mode, 512, 256)`（D=64,H=4）、
    `(…, 64, 256, …, 2048, 256)`（D=128,H=8）、尾段 `(…, 1, 256, …, 256, 256)`；写回 `asc_copy_ub2gm_align(…, 64, 4, mode, 4, 32)`、
    尾段 `(…, 1, 4, mode, 4, 4)` —— 与 §3.3 注 3/注 4、§3.4 表**逐参数一致**。
  - `T.clear` 展开核对：每个 buffer 被展开为 `for i in 4096 { 8 B store }` = **32768 B**，即**覆盖两个版本的全量 footprint**
    （`o_ub` 2×16 KB、`do_ub` 2×16 KB 都被清零）→ §3.3/§9.2「两侧清零」在本工具链下**真的成立**（两侧 × 两版本）。
  - 型号文件复核：`grep -n "core_cnt|ub_size|l1_size|l0_c_size" /usr/local/Ascend/cann-9.2.1/x86_64-linux/data/platform_config/Ascend950PR_9579.ini`
    → `ai_core_cnt=28`(20) / `cube_core_cnt=28`(21) / `vector_core_cnt=56`(22) / `l0_c_size=262144`(50) / `l1_size=524288`(51) /
    `ub_size=253952`(58)，与 §5.5② 记录的行号与数值**完全一致**；`statectl platform` 复跑 `observed_cores = {cube: 28, vector: 56}` 一致。
  - 事实源复核：`constants.md` 的 `CONST-mte2-degradation`（18.4 GB/s/core × 48 = ~1.26 TB/s 混合读写地板，绑定 910B2C /
    tilelang 0.1.2+15ad002b3d）、`CONST-capacity-910B2C`（1.7× / 1.10–1.12× 膨胀）条目**真实存在且数值与引用一致**；
    `algorithm-candidates.md` 的 `reduction` 行只有「单行 SIMT sum/max reducer」一条（设计已评估并否决）。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

检视方式：亲自 Read 源算子全文（`bwd.py` 1–387 行；重点 37–76 的 `_delta_kernel`、274–281 的 `refusal`、337–355 的 `_run`/`_compile`/
`forward`）并与 §0 逐项核对；消费端 134/148/217 行逐条对照。

- **语义理解正确**：源码 38–39 行 docstring「`delta[.., h, row] = sum_d o[row,h,d] * do[row,h,d]` in float32」↔ §0.1 数学式；
  `alloc_fragment([block, dim_v], "float32")`（源码 58–60）+ `T.copy`（61–62）证明「16-bit 输入搬入时精确提升 → fp32 乘 + fp32 累加」，
  与 §0.1「最重要的一条 dtype 语义」一致；`lse` 不在签名内（52–56 行）↔ 「与 lse 无关」；`T.reduce_sum(o_f, rows, 1)`（65 行）的
  树形分组判读与 §0.1「只要求同 dtype 语义下容差内等价」一致（非臆造，§1.6.1 A2 有机器证据）。
- **I/O 契约正确**：源码 49 行 `delta_shape = (batch, heads, dense_seqlen) if dense_seqlen else (heads, total)`；53–55 行三入参；
  134 行 `stat_shape = (batch, heads, dense_seqlen_q)` + `accum = "float32"`；148 行 `delta: T.Tensor(stat_shape, accum)`；
  217 行 `delta_s[i] = delta[b, h, m0 + i]` —— 四处与 §0.2 表及「输出契约不得偏离」结论**逐条吻合**。
- **算法步骤完整（无遗漏/无臆造）**：§0.3 的 6 步覆盖源码全部计算语句 —— 57 行二维 `T.Kernel(heads, ceildiv(total,32), threads=128)`、
  61–62 两条 `T.copy`、63–64 `T.Parallel` 乘、65 `T.reduce_sum`、66–72 谓词写回（`row // dense_seqlen` / `row % dense_seqlen`）；
  host 侧 274–281 `refusal`（`dim_qk % 16 == 0 and dim_v % 16 == 0`，§0.1 边界语义③的出处）、342 行 `prep(o, do)`、
  355 行 `q.view(batch * seqlen_q, heads, -1)` 均已列出。
- **优化手段覆盖完整**：§0.4 O1（63–65 原地乘积后立即归约、不落 GM）、O2（58–60 fp32 fragment）、O3（无 `alloc_shared`）、
  O4（40 行 `block = 32`）、O5（61 行固定 `h` 的 256 B 段）、O6（68 行 `if row < total`）、O7（65 行 `reduce_sum`）、
  O8（57 行二维网格 + 128 线程）、O9（37 行 `@target_cache(maxsize=32)`）逐条对得上源码；算法层/实现层划分与 §0.5 四态处置自洽。
- **耦合性判定合理（抽检 5 条 + 全表）**：`T.Kernel(N)` 一维、核级无 `threads` —— 本次 F1–F8 全部以 `T.Kernel(grid)` 编译通过，
  构成独立佐证；无 warp/`__shfl` → O7 判「重新设计」；O3/O4 判「等价替换」；O6 由运行期谓词改为**构建期静态尾段**（§6.4 / 探针 F4 的
  `(…, 1, 256, …, 256, 256)` 与 `(…, 1, 4, …, 4, 4)` 已实测生成）；舍弃 0 项且 O1–O9 全部有承接（§0.5 汇总段）。
- **NPU 重设计三要素齐全且已修到可构建**：R1（二维网格 + 128 线程 → 一维网格 + `R` 行 tile 流水）语义保持论证覆盖
  「行间无累加关系/并行重排不改每行内容」「尾段静态 `nlast`，越界行列既不读也不写」；R2（`reduce_sum` → 逐行 `vcadd`）覆盖
  「同一批 fp32 乘积、只改分组」「**pad 列两个操作数都清零**」「corner 用例 `nf_mis = 0`」；R3（fragment → UB 描述符 + 双缓冲）
  覆盖「目的端显式切片」「`d_ub` 行距只影响寻址」。v1 首轮指出的两条子论断（R2 只清一侧、R1 尾块仍取 `R` extent）**均已闭环**，
  且本次以生成代码独立验证：clear 覆盖两版本全量、尾段三处 extent 全为 1。
- **§1–§7 与迁移决策一致**：§1.4 五步（任务划分/搬入/融合乘/逐行 `vcadd`/批量写）、§3.2 映射、§4.4 搬运路径、§6.1 循环表
  描述的都是「一维网格 + `R` 行 UB tile + 逐行 SIMD `vcadd`」，无「§0 判重设计、§3 照抄源方案」的脱节。
- **golden 独立性成立**：§8.1 的 `prod = o.float() * do.float(); prod.sum(-1).transpose(1,2)` 只依据 §0.1/docstring 语义
  （朴素、无分块、无 `vcadd` 分组、无 `d_ub` 暂存），未复刻 §0.6 的 NPU 数据流，精度对比不自证。

### 1. API 可行性 — pass

- **API 存在性逐条复核（`tilelang 0.1.15` 实际模块 `tilelang/ascend/language/`）**：`Kernel`（`kernel.py`，一维、无 `threads`）、
  `SimdVF`（`frame.py`）、`simd.vld:324 / vsts:457 / vcvt / vmul:629 / vadd:554 / vcadd:957 / vcpadd:946 / vcgadd:983 / pset:111 /
  update_mask:125`、`annotate_buffer_versions`（`annotations.py`）、`Pipelined/serial` —— 全部存在，签名与设计一致
  （`vsts(addr, src, mask=None, dist="NORM_B32", extent=None)`、`vcadd` 文档「Pairwise add reduction」、`vcadd/vcgadd/vcpadd`
  均为**单寄存器内**归约 → 支撑 §1.6.2 V9「无跨多行批量归约 API」）。
  `T.clear` 来自 `tilelang/language/fill_op.py:48`（`fill` 显式处理 `tirx.BufferRegion` 分支，故区域形式在签名层面被支持）。
- **v1 两个不可构建形态已闭环（正向 + 负向对照）**：
  - 阻塞 1（`for v in range(nvec)` + `acc` 重绑定）：v2 改为**构建期 python helper 拼接单一表达式**，F1/F2/F4（`nvec=1`）、
    F5（D=256）、F6（D=192，`nvec=2` + pad）**全部 COMPILE OK**，生成 `auto prod = vadd(vmul(vcvt(vlds_norm<...` 单表达式；
    负向对照 F7 复现 `Immutable variable `acc` is used outside its defining region!`（IR 中可见 `for v in range(1)` 仍被提升为
    `script.ir_builder.tirx.ForFrame`）→ **v1 缺陷确证，v2 修正有效**。
  - 阻塞 2（`D_v < DW` 整 buffer ↔ 整 buffer 拷贝）：v2 改为目的端显式 region `o_ub[0:R, 0:D_v]` + 两侧 `T.clear`；
    F2（D=64、DW=128）**COMPILE OK** 且生成 `asc_copy_gm2ub_align(…, 64, 128, …, 512, 256)`（64 行 × 128 B、目的行距 256 B），
    负向对照 F3（不切片）**COMPILE FAIL**（`PlanMTECopy … got src=4096 and dst=8192`）。
  - 阻塞 3（尾块取 `R` extent）：v2 尾段独立成构建期代码段，三处 extent 全用静态 `nlast`；F4（S=65）生成 `(…, 1, 256, …)` /
    `(…, 1, 4, …, 4, 4)`，F8（S=1、`nfull=0`）同样 COMPILE OK —— 只读/只写 1 行，越界消失。
- **API 与模式匹配**：全部形态为 `T.Kernel` + `T.Pipelined` + 嵌套 `T.SimdVF` + `T.simd.*`，与 `programming_modes.migration_default = simd`
  及平台 `programming.md` 第 13–15 行条款一致；无 `T.Parallel` 误用、无二维网格、无 `threads=`。
- **写法可构建性**：§3.3 伪代码与探针代码形态一致（`T.Pipelined(tpb>0)` → `T.SimdVF` 内 `T.serial(R)`；目的端切片；尾段静态 extent），
  仅两处**实现细节差异**（不影响可构建性，见建议 3）：探针 `persistent()` 用 `nfull = nt`/`if nfull > 0`（等于设计在无尾块几何下的值），
  `tail()` 用 `if nlast != R`（设计用 `if tail` = `nlast != 0`；因探针只在 S=65/S=1 调用，未触发 0-extent 分支）。

### 2. 内存层级规划 — warn

- **搬运路径完整**（§4.4）：GM `o/do` →（单条跨步 DMA，源间距 `H·D_v·2 B`，目的 region `[0:R, 0:D_v]`）→ UB（2 版本）→ SIMD 寄存器
  （`vld/vcvt/vmul/vadd/vcadd`，乘积不物化）→ `d_ub[i,0]`（`vsts` + `PAT_VL1`、32 B 行距）→（单条压缩 DMA）→ GM `delta`；
  无 L1/L0C、无中间 GM 往返，与 §1.6.0 R3 的「每元素读一次、每行写 4 B」下界一致。生成代码逐参数吻合（见机械复核）。
- **UB 预算（warn 依据，非阻塞）**：设计口径 `32768 + 32768 + 2048 = 67584 B（66 KB）`，占用率 0.266；
  但本次 dump 的 F2 生成代码显示 `d_ub` 被自动加了第二版本（`vsts_norm(..., [(it & 1) * 512 + i*8 + 16384], ...)`、写回
  `asc_copy_ub2gm_align(..., [(it & 1) * 512 + 16384], ...)`）→ 实占 **69632 B（68 KB）**，与 §4.5、§6.3「`d_ub` **不**多版本」、
  §7.2「`d_ub` 单版本」三处描述不符。容量结论不受影响：69632 ≤ 253952，1.7× 膨胀 118374 ≤ 253952，1.10–1.12× 口径
  76595–77988 ≤ 253952。→ 建议 1。
- **L0C 容量**：不适用（无 `T.gemm`；`design_calc_check` 的 `l0c_budget` 亦 skip）。
- **Buffer 层级标注**：§4.3 每个 buffer 标注了层级（UB / SIMD 向量寄存器），寄存器中间量明确「不物化到 UB」（A1），`d_ub` 标注 32 B 行距。
- **pad 安全化已实测成立（v1 阻塞 2 的数值前提）**：`T.clear(o_ub)` / `T.clear(do_ub)` 各展开为 4096 次 8 B 存储 = **32768 B**，
  即**覆盖两个 UB 版本的全部 footprint**（`o_ub`/`do_ub` 各 2×16 KB）；两条 `T.copy` 只写 `[0:R, 0:D_v]`（列 ≥ `D_v` 不写），
  `vld` 读满 128 lane 的 pad 列在**两个版本中都是 0** → `0·x = 0`（含 `0·NaN` 被消除）严格成立，短行零填充不改和。
  （注：`T.clear` 的零值写入在生成代码里是 PIPE_S 的 8 B 存储循环，一次性、每 kernel 一次；见建议 5。）

### 3. Tiling 策略 — pass

- **Block 划分明确**：`block_M = R`（`R = S 在 [16,64] 内的最大因子，否则 64`）、`block_N = D_v`（整行一个向量寄存器，不切分），
  理由（`D_v=128` 恰好 1 个 16-bit 向量寄存器、行宽 256 B = 2 个 128 B 段）成立。
- **对齐约束**：行宽 256 B（D_v=64 时 128 B = 1 个完整段）；`vld` 2048-bit 整行；fp32 中间量 64 lane；`d_ub[R,8]` 32 B 行距与
  `PL-950-simd-reduce.py` 的 `(2,8)` 已验证布局一致（源码已核）。
- **非整除处理（v1 阻塞 2/3 的落点，全部闭环）**：
  - `S % R ≠ 0`：`nfull = S // R` 个满 tile 走 `T.Pipelined`，尾段是**独立构建期代码段**，两条 `T.copy` + `T.serial(nlast)` + 写回
    **三处 extent 全为静态 `nlast`**（F4/F8 生成代码已验证），**删除了** v1 的运行期谓词 `if s0 + i < S`，并取 `nsp = 1` 使循环上界
    与 extent 同时静态化。
  - `D_v % 128 ≠ 0`（D_v=64）：`DW = 128`、目的端显式切片 `[0:R, 0:D_v]`（元素数相等，F2 通过 / F3 负向失败）、两侧 `T.clear`（含两版本）。
  - `R` 选择规则在 §5.2 表 11 行上逐行复算成立（含 `R = 50`（S=200）与 `R = 64` 的尾块/无尾块两类）。
- **分核策略三要素（§5.5）齐备且为实查**：
  - ① 逻辑核数：`tile 数 = B·H·ceil(S/R)` 复算 = 8192 / 32768 / 8192 / 3072 / 64，与文内一致；分核后 `grid = B·H·nsp` = 256/256/256/256/64，
    `tpb = nfull/nsp` = 32/128/32/12/1，行覆盖 `nsp·tpb·R = S` 成立。
  - ② 物理核数依据**为实查**：`statectl platform --task-dir …` 复跑 `observed_cores = {cube: 28, vector: 56}`；型号 ini 行号/数值
    （20/21/22/50/51/58）与 §5.5② 记录**逐项一致**；`torch.npu` 未执行且写明理由（单卡被同批 Stage 3 占用），未以经验值替代；
    AIV/AIC 调度粒度假设已在 §9.1 R8 标注为未文档化假设。
  - ③ 规模判定「极大规模（3072–32768 tile ≫ 56 核）→ 固定启动核数 + 核内串行」与 `nsp` 取法（整除 `nt`、`grid ≥ 224`）自洽，
    核内循环上界全为构建期常量；备选形态（逻辑核超发，`grid = B·H·nt`）在 §1.6.3 与主选并列并给出裁决阈值。
- **波次说明**：256/56 = 4.57 → 5 波（91%）与 smoke 的 2 波均量化给出；`grid` 受 `B·H` 因子约束而不能强求 56 整数倍的说明正确。

### 4. 技术约束检测 — pass

- **三维 Kernel**：不涉及（一维 `T.Kernel(grid)`；F1–F8 全部以该形态编译通过）。
- **GEMM 非整除 / L0C 溢出**：不涉及（无 `T.gemm`、无 L0C 分配）。
- **平台条款逐条对上**（§3.5.1）：一维网格与核级无 `threads`、无 warp/`__shfl`、主动规避 `TRAP-950-simd-shared-reduce`、
  整 buffer ↔ 整 buffer 元素数相等条款（`programming.md`「整个 Buffer 对整个 Buffer 的拷贝要求元素总数相等；局部拷贝用显式 region 表达」）
  → 以目的端 region 满足、尾轴对齐、按 shape 特化编译、不启 fast-math。无「把其他工具链限制当作本平台要求」的迹象。

### 5. 循环与同步 — warn

- **循环结构（§6.1）** 明确且与生成代码一致：`T.Kernel(grid)` →（构建期分支）`T.Pipelined(tpb, num_stages=2)` → `T.SimdVF` 内
  `T.serial(R)` / `T.serial(nlast)`；行内 `d` 由向量 lane 承担；不使用 `T.Parallel`。
- **同步（§7.2）** 与 F2 生成代码对照**基本吻合**：block 起始 `asc_sync_notify(PIPE_V,PIPE_MTE2,0/1)`、`(PIPE_MTE3,PIPE_V,0/1)`；
  每迭代 `wait(PIPE_V,PIPE_MTE2,(it&1))` → 两条 `asc_copy_gm2ub_align` → `notify(PIPE_MTE2,PIPE_V,(it&1))` →
  `wait(PIPE_MTE3,PIPE_V,(it&1))` → `wait(PIPE_MTE2,PIPE_V,(it&1))` → SIMD VF → `notify(PIPE_V,PIPE_MTE3,(it&1))` / `(PIPE_V,PIPE_MTE2,(it&1))`
  → `wait(PIPE_V,PIPE_MTE3,(it&1))` → 压缩 DMA → `notify(PIPE_MTE3,PIPE_V,(it&1))`。自动同步策略（不写手工 flag）成立。
  **两处描述与实测不符（建议 1，非阻塞）**：① `d_ub` 实为双版本（事件也按 `(it&1)` 配对），§7.2「`d_ub` 单版本」应改写；
  ② 尾部收敛 wait 在 F2 为 **4 个**（MTE3→V 0/1、V→MTE2 0/1），§7.2 写「3 个」。
- **尾块（§6.4）** 已落到三处 extent（见 §3），无运行期谓词，与生成代码一致。
- **R10 判定（调度单点名）**：尾块几何取 `nsp = 1`、且尾段在流水外写同一 buffer —— 本检视判定为**纯性能降级，不影响正确性/契约**：
  ① 行覆盖与写回 extent 全为静态常量，`delta` 的数值与布局契约不变；② 尾段排在全部满 tile 之后顺序执行，无论编译器
  保留还是关闭多版本（两种情形分别落到「基址」或「版本 0」），「先写后读」的次序与 InsertSync 插入的跨 pipe wait 都成立，
  不存在读到旧版本的可能；③ manifest 与 op smoke 几何全部 `S % R == 0`，不进入该路径。
  但 **R10 的机制证据不可复现**：本次重跑 F4/F8 **没有**任何 "did not enable multi-buffering" 类警告，且 F4/F8 的 `nfull`
  分别为 1/0（流水循环被展开），**根本无从观测多版本**——故「因尾段在循环外写而关闭多版本」属未证实归因（建议 2）。
- **§6.3 的诚实标注**（装载未提前到上一 tile 计算之前、流水收益属未实测量、已入 §1.6.3 裁决计划）与生成代码一致，处理正确。

### 6. 验证方案 — pass

- **Golden 独立、可执行**（§8.1）：`prod.sum(dim=-1).transpose(1,2)`，与 §0.1 语义逐字对应，朴素无分块/无向量化/无 `d_ub` 映射，
  不构成自证；并说明设计期 `verify_equiv.py` 的 fp64 参照不替代 golden。
- **精度标准**（§8.2）：输出 fp32 → atol = rtol = 1e-4（与模板 fp32 标准一致）；并给出与 op 级容差的传递论证
  —— 本次核对 `examples/TileOPs/tests/ops/test_mla_dense_bwd.py:40-46` 的 `_RTOL = {fp16: 1e-3, bf16: 1.6e-2}` 与
  `_ATOL_FLOOR = 5e-3`，与 §8.2 引用**完全一致**。等价性实验亦佐证该标准可用：`random/fp16`、`random/bf16` 在 (1e-4, 1e-4) 下
  违反率 0；1.56% 的违反只出现在 ±512 量级的 `large/fp16` 抵消行（已登记 §9.1 R5，且源算子同 dtype 语义同样存在）。
- **L0/L1/L2/Boundary 计划具体**（§8.3）：规则 shape（`S=128` 双 dtype 主路径）、**`D_v=64` pad 路径**（与 op 自身
  `test_mla_dense_bwd_general_kernel` 同几何 —— 本次核对源码第 149–155 行 `MLADenseBwdTest(1, 256, 256, 4, 128, 64, torch.bfloat16)`
  即 batch=1/S=256/H=4/D_qk=128/**D_v=64**/bf16，与 §8.3 行完全对应）、`S=1` 极小、manifest smoke（`B=1,S=512,H=8`）、
  `ds-v3-b2-8k`（`tpb=128` 长核内循环）、`kimi-k2-8k`、Boundary（`S=200` `R=50`、`S=320`、`S=65` 尾段、`S=1`+`H=1`、全零、非有限 `equal_nan=True`），
  并新增「覆盖的设计点」列；「L0 两 dtype 全过才进 L1/L2」的分层契约明确。
- **覆盖缺口（建议 7，非阻塞）**：现有边界用例的尾块几何都只有 `nfull ≤ 1`（S=65/S=1），无法观测「流水多版本 + 尾段」
  的交互；建议补一条 `S = 131`（`R = 64`、`nfull = 2`、`nlast = 3`）与一条 `D_v = 64 + S % R ≠ 0` 组合。

### 7. 完整性与一致性 — warn

- **章节齐全**：§0（0.1–0.7）+ §1–§10 全覆盖模板 11 章；迁移任务的 9 维度所需章节（含 §1.6.0–§1.6.3、§5.5 三要素、§8.3）齐备。
- **无占位符**：全文扫描模板禁用词（三种占位标记）均 **0 命中**；§10.2 的 `⬜` 是文件状态列（不是内容占位）。
- **同类引用真实**：§3.5.3 的 6 条 + §0.5/§3.4 引用的 `PL-950-row-primitives.py`、`references/programming.md`、
  `knowledge/numerical-boundaries.md`、`core-split-strategy.md` 逐条 stat **全部存在**；所引条目内容与设计引用语义一致
  （`TRAP-950-simd-shared-reduce`、`PL-950-simd-reduce` 的 `update_mask(n,32)` + `(2,8)` 暂存 + `vsts(extent=1)` 形态均已核）。
- **内部一致性（warn 依据）**：结构层（API 映射 / 内存规划 / Tiling / 循环）相互一致，无矛盾；存在 4 处**数值/标签口径**偏差：
  ① `d_ub` 实为 2 版本 → 68 KB（§4.5/§6.3/§7.2）；② §0.6 表与 §3.3 注 2/§3.4 把 F5 标为「`nvec=2` 内联展开」，而探针 F5 与 F6
  走的是**同一个** `_row_dot_expr` helper（无内联形态被单独覆盖）；③ §3.2 步骤 6b 说 helper 返回 `vadd(...)` 链、§3.3 的 `row_dot`
  实际返回 `vcadd(part)`；④ §1.6.0 的 manifest 交叉核对把 `B*H*S_q*4`（2.1 MB）记为 delta 写出，而 `attention.yaml` 的注释
  写明该 4 B/行项是 **lse**（delta 未单列；量级核对结论不变）—— 见建议 1/3/6/8。

### 8. 算法优化分析 — pass

- **§1.6 章节存在性**：1.6.0 算法调研 / 1.6.1 数学等价优化 / 1.6.2 向量化替代分析 / 1.6.3 向量化轴与布局决策四节齐全，
  且「多步/规约类」按 `algorithm-research.md` §2 做了**完整调研**（R1 候选表 + R3 四口径 + R4 逐候选），深度匹配算子复杂度。
- **算法调研存在性与完整性**：R1 候选表**含基线候选 0**（源算法）+ 8 个候选并逐项初判（等价性 + 收益方向 + 是否入 R3）；
  R2 给出**明确结论「有」**（逐行点积天然单遍、状态 O(1)、`D_v>128` 的段合并在构建期展开、不选跨核 partial+merge）；
  R3 **四口径齐全**（FLOPs / 访存 Bytes / 扫描遍数 / 中间缓冲峰值，基线在表，口径写明不含启动与 16-bit 提升）；
  R4 逐候选六维评估（单元匹配 / 容量 / 对齐 / 静态边界 / 流水 / 跨核）；调研结论写明选定算法与「无更优替代」的**调研范围**
  （`algorithm-candidates.md` 全表、pattern-library `constants.md`/`INDEX.md`、`ALG-950-formulas.md`/`operator-patterns.md`/
  `PL-950-row-primitives.md`、本仓 ascend 向量算子）。
- **调研结论正确性（独立复核：成立）**：
  - **候选覆盖**：`algorithm-candidates.md` 的 `reduction` 行只有「单行 SIMT sum/max reducer」（`PL-950-row-primitives`），
    本设计**已评估并否决**（每行一次 `tvm_thread_allreduce` + barrier，§0.5/§1.6.1 A2/§3.4 三处引用），不存在漏掉的命中行候选；
    该表无本算子族的其它条目。
  - **负向断言复核**：§1.6.0 的负向论断只有 R1 候选 8「无代数化简」（结构依据：点积的乘加次数即信息量下界 D 乘 + D 加，
    行间无依赖、无共用子式）与候选 4/5/6 的**量化淘汰**（非「无先例」式断言），并**未**断言「无在线变体」（R2 明确判「有」）
    —— 与源算法证据（源码已是单遍融合）不矛盾，无被推翻的负向断言。
  - **复杂度算术独立复算（逐项）**：FLOPs `2·N·D` = 2·524288·128 = **134.2 M** ✓；基线访存 `2·N·D·2 + N·4` = 268.4 + 2.1 =
    **270.5 MB** ✓；候选 4 = 268.4×3 + 2.1 = **807.3 MB**（未漏计中间缓冲 GM 往返，✓；÷1.26 TB/s = 641 µs ✓）；
    候选 5 `2·R·R·D·(N/R)` = 2·N·R·D = **8.59 G**（64× ✓）；候选 6 原子流量 `2·4·524288` = **4.19 MB** ✓（v1 建议 2 已改对，
    §1.6.3 候选 3 引用同一数字，一致）；发射项 11×524288 = 5.77 M ÷ 56 ÷ 1.8 GHz = **57 µs** ✓；
    流量项 270.5 MB ÷ 1.26 TB/s = **214.7 µs**、÷ 1.03 TB/s = **262.3–262.5 µs** ✓（区间 215–262 µs 成立）。
    唯**收敛候选的中间缓冲峰值应为 68 KB（69632 B）而非 66 KB**（`d_ub` 被编译器加至 2 版本，见 §2）—— 差异 2 KB，
    R4 的占用率 0.27 与「不构成约束」的结论不变。
  - **亲和性淘汰依据**：容量/带宽常数条目（`CONST-capacity-910B2C` 的 1.7×/1.10–1.12×、`CONST-mte2-degradation` 的
    18.4 GB/s/core 与 ~1.26 TB/s 48 核聚合）**真实存在、数值一致**，且 **两处常数都显式标注「绑定 910B2C / tilelang 0.1.2，
    非 ascend-950pr-9579 实测」**；`statectl platform` 的 `cost_measurements: []` 与之一致（如实降级，未把外推当实测）。
    v1 首轮「215 µs 是保守下界」的口径问题已改为「上界 262 µs（每核地板 × 56）/ 下界 215 µs（48 核聚合直接复用）」并说明来源，
    且补了与 `manifest/attention.yaml`（本次核对：`flops = "B*H*(scores*per_score + rows*2*D_v)"`）的同口径交叉引用。
  - **并列候选裁决路径**：主选 vs 备选的分歧落在两个本型号未实测常数（跨步描述符带宽差、散段写放大），§1.6.3 给出
    **主选 + 结构完整备选（逻辑核超发，UB 34 KB、分核按新 `grid` 重算）+ 裁决计划**（3 代表 shape、`msprof` Task Duration、
    A/B ≥5 次、阈值「备选 < 主选 × 0.9 才翻转」、按 shape 分片回写）——符合实验裁决模式三件套。
- **调研 → 设计一致性**：§1.6.0 选定「候选 0 结构 + A1/A2/A3 实现级等价优化」，§1.4 五步、§3.1 公式拆解（明确「输入公式为
  §1.6.1 优化后公式」）、§6.1 循环表全部按该结论展开；迁移侧 §0.5 的 11 条四态处置与调研选族一致，O1–O9 的意图**逐项**有承接
  （§0.5 表 + §0.6 R2「源优化意图承接」段），无静默丢弃、无「调研选 A 设计做 B」。
- **等价性论证独立推演（三项均成立）**：
  - **A1（寄存器融合）**：两形态对每个 `d` 都做同一个 fp32 乘法并按 RNE 舍入到 fp32，再对同一批项求和，只移动存放位置 →
    **逐位恒等**。独立证据：重跑 `ulp(cand,base) = 0` 于 12/12 用例；生成代码为独立 `vmul` + `vadd`（无 FMA 收缩，
    且 §7.3 显式不开 `TL_ENABLE_FAST_MATH`）→ 论证成立。
  - **A2（实现分组：`vadd` 两半 + 1×`vcadd`）**：独立推演 —— `vcadd(blk[0::2] + blk[1::2])` 的第一步正是平衡二叉树
    对 128 元素的第一级配对，故 `tree128(x) ≡ tree64(x_even + x_odd)`；而 `tree64` 在**128 = 2×64**处正好有子树边界，
    故与「`tree64(p[0:64]) + tree64(p[64:128])`」**逐位相同**（fp32 每级一次舍入，两式舍入点一一对应）→ 与 §1.6.1 注的断言一致。
    机器证据：A2 与 A2b 12/12 逐格相同、int32 位模式 `torch.equal` 为 True；两者 `random/fp16/D128` = 322 ulp、基线 2238 ulp
    （本检视独立复算），`large/fp16` 2798 vs 基线 3310 —— **树形优于顺序**，容差内等价且无系统偏置。
    fp16/bf16 舍入影响评估正确：16-bit 只在入口无损提升，乘/加都在 fp32，分组差异不会被放大到 16-bit 量级。
  - **A3（暂存 + 压缩写出）**：索引映射 `d_ub[i,0] → out[i]` 数学恒等，独立证据 `ulp(cand,base) = 0`（12/12）；
    生成代码 `asc_copy_ub2gm_align(…, 64, 4, mode, 4, 32)`（64 元素、源间距 32 B、目的连续）与 §3.3 注 3/A3 收益口径吻合。
  - corner 用例（±inf、`inf·0`、NaN、全零、交替符号）`nf_mis = 0` → 非有限传播与基线逐位一致（含 pad 全零前提）。
- **数学等价优化完整性**：A1/A2/A3 均有四要素（原式 → 优化后公式 → 等价性论证 → 收益量化），收益栏已按 v1 建议拆分为
  「源基线 32 KB / NPU 物化则 `2·R·D·4 B` = **64 KB**」并复算一致；「其余空间无进一步优化」给出依据（乘/加/提升已是逐元素
  最小操作集，无公共子表达式、无除法/超越函数可降代换）。**建议 9（非阻塞）**：`T.simd.vmula`（`simd.py:635`）/`vmadd`（649）
  的融合乘加可把「2 `vmul` + 1 `vadd`」压到 2 条指令/行，属「算子降代换」候选；§7.3 已有「避 FMA 以保持乘积舍入位置」的论证，
  但 §1.6.1 未把它列为被评估项，建议补一行交叉引用（结论仍为不采纳：会破坏 A1 的逐位等价与源 fp32 乘语义）。
- **向量化替代覆盖完整性（与 §3.3/§6 交叉核对）**：V1–V11 覆盖 §3.3 伪代码与 §6.1 循环表中的**全部**计算点 ——
  行内乘（V1）、16→32 提升（V2）、两半合并（V3）、行内求和（V4）、暂存写（V5）、GM→UB（V6）、UB→GM（V7）、pad 清零（V8）、
  行循环（V9）、tile/核索引（V10）、尾段分支（V11），无遗漏点；V9 的「无跨多行批量归约 API」已由 `simd.py` 导出表核实
  （`vcadd:957 / vcgadd:983 / vcpadd:946` 均为单寄存器内归约），保留理由为 API 缺失且带量化代价（转置 ≥2 条/行 vs 省 9%），
  V10 为 block 级索引、V11 为静态边界 —— 无一处以「实现简单/照源码写」为由保留。两处口径需修正（建议 4/5，非阻塞）：
  ① 每行指令数 11 与生成代码不符 —— 表达式内 `vld(o)`/`vld(do)` 各出现两次（PART_EVEN/PART_ODD），生成代码为 4 `vld` + 4 `vcvt`
  + 2 `vmul` + 1 `vadd` + 1 `vcadd` + 1 `vsts` = **13 条**（除非后端 CSE 掉重复 `vld`）；发射项 57 µs 与 V9 的「9%」应注明该假设；
  ② `T.clear` 在生成代码中是 PIPE_S 的 4096×8 B 标量存储循环（不是向量指令），V8 标「向量化 ✅」不准确（一次性、非热点，
  对 D_v<DW 的极小几何可能与主体同量级）。
- **布局决策完整性（§1.6.3）**：候选矩阵 7 项，规约类**必选的两种 lane 映射**（候选 1/2 的水平归约 vs 候选 4 的垂直扫描）分别枚举，
  另含 D 轴跨核、Cube 对角 GEMM、host permute、核内转置；逐候选有量化评分（整除性 / 尾 lane 浪费率 / 累加链形态 / repack 代价 /
  UB 影响：`D_v=64` 的 100% lane 浪费但指令数不变、候选 2 的 32 KB/行 → 双缓冲只能 `R=1`）；弃选均有量化理由（候选 6 的 268 MB 级
  host permute、候选 7 的 ≥2 条转置 vs 1 条 `vcadd`）；**弃选前提已亲自核对**：候选 7 的转置 API 真实存在（`simd.py` 的
  `vintlv/pdintlv/vld2` 等），故「API 存在但代价倒挂」的表述与文档一致；候选 6 的 host permute 代价与平台「host 只允许 stride/shape
  视图」立场一致。**布局决策与实现三方一致**：§3.3 的 `vld(o_ub[i, 0])`（内层向量维 = `d`）、§4.3/§4.5 的 `(R, DW)` 形状与 66→68 KB
  预算、§6.1 的循环表同口径；`D_v > 128` 的 `nvec` 分段在构建期展开（§3.3 `row_dot`，F5/F6 已验证），不进入运行期循环。

## 结论依据与放行条件

- 三条 v1 阻塞全部闭环，且**正向（COMPILE OK）与负向（COMPILE FAIL）证据都由本检视独立复现**；伪代码与
  §3.3/§4.5/§5.2/§6.3/§6.4/§8.3/§9.2 的描述一致（除 §2 指出的 `d_ub` 版本数这一处数值/描述偏差）。
- v2 未引入正确性回归：尾段在流水外写同一 buffer（R10）判定为**纯性能降级**；pad 列的 `0·NaN` 已由「两侧 × 两版本清零」
  实质性消除（本次以生成代码的 32768 B ×2 填充佐证）；目的端切片后 pad 列**不会被 `T.copy` 写入**、也不会被清零以外的路径读脏。
- A2 证据对齐可信：实现分组与 A2b 在 `nvec==1` 下**逐位恒等**（数学上由「平衡二叉树在 128=2×64 处有子树边界」保证，
  经验上 12/12 逐格相同 + `torch.equal` 为真），且 §3.1 步骤 5–6 / §3.3 `row_dot` 正是实现分组。
- 性能口径自洽：215–262 µs 区间（1.26 TB/s 与 1.03 TB/s 两种外推）、+4.19 MB（候选 6）、64 KB（A1 物化口径）复算一致；
  66 KB 应按实测修正为 68 KB（不影响任何结论）；910B2C 外推在 §1.6.0、§4.5、R1/R2 处均已标注为**非本型号实测**。
- 9 条建议均为**建议级**（数值/标签/覆盖度），不构成放行阻断：按 skill Phase 3 判定表，全部阻塞级维度（0/1/2/3/4/6/8）
  均无 `fail`（维度 2 为 `warn`，warn 不阻断），故结论为 `通过`；建议在 Stage 3 实现时顺手修正建议 1/2，
  避免 Stage 3/4 依据错误描述做判断（Stage 3 的 L0 `D_v=64` 用例是这些假设的数值闸门）。

## 建议汇总（非阻塞，供 Stage 3 实施时顺手修正）

> 本节与上方各维度的 pass/warn 判定一一对应，**不含阻塞项**；9 条均为数值口径 / 证据标签 / 测试覆盖度类建议，
> 不影响 `结论: 通过`。

### 建议 1: `d_ub` 实际被编译器加至 2 版本，UB 预算应按 68 KB（69632 B）修正

- 严重程度: 建议
- 位置: DESIGN.md §4.5（UB 表）、§6.3（「`d_ub` **不**多版本」）、§7.2（「`d_ub` 单版本」）、§1.6.0 R3/R4（峰值 66 KB）
- 问题描述: 本次 dump 的 F2 生成代码显示 `d_ub` 也被 MaterializeMultiBuffer 加了第二版本
  （`vsts_norm(..., (__ubuf__ float*)(buf + ((it & 1) * 512 + i*8 + 16384)))`、写回同一 `(it & 1) * 512` 偏移），
  故实占 `16384+16384+4096 = 69632 B`，而非 §4.5 的 67584 B；同步事件亦按 `(it & 1)` 配对（§7.2 的「单版本 + 等上一次 DMA」
  描述与生成代码不符）。容量结论不受影响（69632 与 1.7× 膨胀值 118374 均 ≤ 253952）。
- 修改建议: §4.5 表加一行「`d_ub` × 2 版本（编译器自动多版本）2048×2 = 4096」并将总计改为 **69632 B（68 KB）**、占用率 0.274；
  §6.3/§7.2 改为「`d_ub` 亦按 `(it&1)` 双版本（实测），若需单版本可显式关闭该 buffer 的多版本化」；§1.6.0 R3 收敛候选行的峰值同步。

### 建议 2: R10 的「编译器因尾段在流水外写而关闭多版本」归因不可复现

- 严重程度: 建议
- 位置: DESIGN.md §3.3 注 4（末段「F4/F8 输出 "Ascend AutoSchedule did not enable multi-buffering for buffer `o_ub`" 警告」）、
  §6.3 尾段例外条、§9.1 R10
- 问题描述: 本次重跑 `design_probe_compile_v2.py f4` / `f8` 并全量捕获告警 → **无**任何 "multi-buffer / did not enable /
  written before" 文本；且 F4/F8 的 `nfull` 为 1/0，流水循环被完全展开，**多版本与否不可观测**。
  该归因属未证实推断（正确性结论不受影响：两种编译器行为下尾段都是「先写后读」顺序执行，见维度 5 的 R10 判定）。
- 修改建议: 删去括号内的警告引文，改写为「F4/F8 的 `nfull ≤ 1` 使多版本不可观测；若 `nfull ≥ 2` 且存在尾段，版本化与同步的交互
  未经探针验证」；并在 §8.3 增一条 `S = 131`（`R=64`、`nfull=2`、`nlast=3`）作为该交互的闸门（与建议 7 合并处理）。

### 建议 3: F5 被标为「内联展开」，但探针 F5 与 F6 走同一个 helper

- 严重程度: 建议
- 位置: DESIGN.md §0.6 R2、§3.3 注 2、§3.4（「F5（`nvec=2` 内联展开）、F6（`nvec=2` 经构建期 helper 展开）」）
- 问题描述: `design_probe_compile_v2.py` 的 F5（`persistent(256, 512, 8, 8)`）与 F6（`persistent(192, 512, 8, 8)`）**都**调用
  `_row_dot_expr` helper，没有真正的「内联手写 `vadd(vadd(p0,p1),p2)`」探针 → 「两种形态都编译通过」的说法超出证据。
- 修改建议: 把标签改为「F5：D=256（`nvec=2`，经 helper）、F6：D=192（`nvec=2` + pad，经 helper）」；若要保留「内联形态」结论，
  补一个真正手写内联的 `nvec=2` 探针（约 10 行）或删去该形态主张。

### 建议 4: 每行向量指令数 11 与生成代码不符（实为 13，除非后端 CSE）

- 严重程度: 建议
- 位置: DESIGN.md §1.6.0 发射项（「每行 11 条向量指令（2 `vld` + 4 `vcvt` + 2 `vmul` + 1 `vadd` + 1 `vcadd` + 1 `vsts`）」）、
  §1.6.2 V9（「每行 1 条 `vcadd` 占每行 11 条指令的 9%」）
- 问题描述: F1/F2/F6 生成代码中每行表达式含 **4 次 `vlds_norm`**（`o`/`do` 各在 PART_EVEN、PART_ODD 各一次，绑定到 python 变量
  并未阻止 IR 内联），即 4 `vld` + 4 `vcvt` + 2 `vmul` + 1 `vadd` + 1 `vcadd` + 1 `vsts` = 13 条/行。若后端不做 CSE，发射项应约
  67 µs（仍远低于 215–262 µs 的流量项），结论不变。
- 修改建议: 在发射项处注明「按 11 条估（假设后端 CSE 重复 `vld`）；生成代码字面为 13 条，Stage 4 以 `msprof` 指令计数核对」；
  V9 的「9%」改为以 13 条为分母（≈7.7%）或注明口径；如要降为 2 条 `vld`，可试 `pset/update_mask` 之后用 `T.simd.alloc_var`
  显式绑定寄存器（需探针验证）。

### 建议 5: `T.clear` 的 pad 清零是一次性 PIPE_S 标量存储循环，不是向量指令

- 严重程度: 建议
- 位置: DESIGN.md §1.6.2 V8（「pad 列清零 → `T.clear` ✅ 向量化替代」）、§9.1 R12
- 问题描述: 生成代码为 `for i in 4096 { 8 B store }`（每 buffer 32768 B，覆盖两版本），走 `PIPE_S`——属一次性 UB 初始化，
  不是向量化清零；对 `D_v < DW` 的极小几何（op smoke：B=1,S=256,H=4,D_v=64，总行 1024）该初始化可能与主体计算同量级。
- 修改建议: V8 的「向量替代方案」改为「构建期一次清零（生成代码为 PIPE_S 的 8 B 存储循环，覆盖全部版本）」并把「是否替代」
  标为「一次性初始化，非热点」；若 Stage 4 发现该路径开销显著，可改 `fill` 的 region 形式（`fill_op.fill` 支持 `BufferRegion`，
  本型号未验）只清 `[0:R, D_v:DW]`，或用 `PL-950-simd-reduce` 已验的 `T.simd.update_mask(D_v, 32)` 只算有效 lane。

### 建议 6: manifest 交叉核对里 `B*H*S_q*4` 被记为 delta 写出，但该 yaml 注释写的是 lse

- 严重程度: 建议
- 位置: DESIGN.md §1.6.0「设计期估算（D-2）」的 manifest 交叉核对段
- 问题描述: `examples/TileOPs/tileops/manifest/attention.yaml` 的 `bytes` 公式为
  `2*(q+qv+kv+kvv)*elem_bytes + B*H*S_q*4`，注释「lse is float32」；`delta` 的 `B*H*S_q*4` 写出项**未单列**。
  设计的「写出 `B*H*S_q*4` = 2.1 MB」与之量级一致但归属不精确（该项在 yaml 里是 lse）。
- 修改建议: 改为「manifest 的 `flops` 中 delta 项 = `B*H*rows*2*D_v` = 134.2 MFLOPs（逐项吻合）；`bytes` 的
  `B*H*S_q*4` 项注释为 lse，delta 的写出（同为 2.1 MB）在 yaml 中未单列，故本设计的 270.5 MB 与 yaml 口径同量级、不冲突」。

### 建议 7: 边界用例缺「尾段 + `nfull ≥ 2`」与「`D_v < DW` + 尾段」组合

- 严重程度: 建议
- 位置: DESIGN.md §8.3（Boundary 行）、§6.4、§9.1 R10
- 问题描述: 现有尾块用例（S=65 → `nfull=1`、S=1 → `nfull=0`）无法覆盖「流水循环 ≥ 2 次迭代 + 尾段写同一 buffer」的版本化/同步交互；
  `D_v=64` 与 `S % R ≠ 0` 的组合（pad 路径 + 尾段）也无用例。
- 修改建议: Boundary 增 `B=1, S=131, H=8, D_v=128`（`R=64`、`nfull=2`、`nlast=3`、`nsp=1`）与
  `B=1, S=65, H=4, D_v=64`（pad + 尾段）；用途列写「流水与尾段的 buffer 版本/同步交互」「pad + 尾段组合」。

### 建议 8: §3.2 步骤 6b 与 §3.3 `row_dot` 的返回值口径不一致

- 严重程度: 建议
- 位置: DESIGN.md §3.2 步骤 6b（helper「返回 `vadd(vadd(p₀,p₁),p₂)…` **单一表达式**」）vs §3.3 的 `row_dot`（返回 `vcadd(part)`）
- 问题描述: 一处说 helper 只拼 `vadd` 链、`vcadd` 在调用处；另一处把 `vcadd` 放进 helper。探针 `_row_dot_expr` 是前者。
- 修改建议: 统一为探针形态（helper 拼 `vadd` 链、调用处 `T.simd.vcadd(...)`），或明确写「helper 可返回 `vcadd` 结果，
  与探针形态等价（同一 trace 作用域）」。

### 建议 9: `vmula/vmadd` 融合乘加未被列为 §1.6.1 的被评估项

- 严重程度: 建议
- 位置: DESIGN.md §1.6.1（「其余空间」结论段）、§7.3（fast-math / FMA 论证）
- 问题描述: `simd.py:635 vmula` / `649 vmadd` 提供融合乘加，可把「2 `vmul` + 1 `vadd`」压到 2 条指令/行（约 −8% 发射）；
  §7.3 已给出「不收缩为 FMA 以保持乘积舍入位置」的理由，但 §1.6.1 的「无进一步优化空间」未提及该候选。
- 修改建议: §1.6.1 的结论段加一行「算子降代换已审视 `vmula/vmadd`：会改变乘积舍入位置、破坏 A1 的逐位等价与源
  「fp32 乘 + fp32 累加」语义，且收益仅约 1 条指令/行 → 不采纳（理由见 §7.3）」。
