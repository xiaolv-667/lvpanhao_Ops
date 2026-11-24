# _delta_kernel（packed 分支）设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息

- 文档: `examples/mla_varlen_bwd/_delta_kernel/DESIGN.md`（1397 行，v0 / first_design）
- 任务类型: 迁移（harness 子模式；`MLAVarlenBwdOp` 第 1 个 launch，`dense_seqlen == 0` 分支）
- 源算子: `/workspace/TileOPs/src/tileops/_kernels/attention/mla/bwd.py` 第 38–77 行
  （`_delta_kernel`；消费端 `_bwd_main` 第 122/137/147–148/222–224 行，host 侧第 382–387 行，contract 第 274–281 行）
- 检视维度: 9 项（0 源算子理解与迁移分析 / 1 API 可行性 / 2 内存层级 / 3 Tiling / 4 技术约束 /
  5 循环同步 / 6 验证方案 / 7 完整性 / 8 算法优化分析）
- 平台: `ascend-950pr-9579`（ascend / dav-3510，CANN 9.2.1，28 Cube / 56 Vector，UB 253952 B / L1 524288 B / L0C 262144 B），
  `stage_skills.review = tilelang-ascend`，`programming_modes.migration_default = simd`
- 阻塞级问题数: 0
- 建议级问题数: 5（均为文档级，逐项内嵌在对应维度，不影响结论；按模板要求，结论为通过时本文不设问题清单章节）

## 机械复核（D-1/D-5）

### 1. `verify_equiv.py` 重跑（D-1）

命令：`cd examples/mla_varlen_bwd/_delta_kernel && ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python verify_equiv.py`
→ 退出码 0，末行 `all items pass`。执行结果与 §1.6.1 内嵌结果表**逐格一致**：

| 采纳项 | 重跑用例数 | 重跑 max ulp | 重跑最大违反率 | §1.6.1 内嵌 | 一致 |
|---|---|---|---|---|---|
| A1 寄存器融合 | 12 | 3310 | 0.0000 | 12 / 3310 / 0.0000 / EQUIV_PASS | ✅ |
| A2 实现分组（`vadd` 两半 + 1×`vcadd`） | 12 | 2798 | 0.0000 | 12 / 2798 / 0.0000 / EQUIV_PASS | ✅ |
| A2b 对照分组 | 12 | 2798 | 0.0000 | 12 / 2798 / 0.0000 / EQUIV_PASS | ✅ |
| A3 批量写出 | 12 | 2798 | 0.0000 | 12 / 2798 / 0.0000 / EQUIV_PASS | ✅ |
| A4 packed head-major 写回（含运行期 clamp） | 8 | 0（逐位相同） | 0.0000 | 8 / 0（bit-exact）/ 0.0000 / EQUIV_PASS | ✅ |

A4 8 个几何的逐行输出与 §1.6.1 注完全对齐（`bit_equal=True`、`covered=True`、`clean_outside=True`、
`min_n≥1` 全部成立；静态 extent 越界诊断 `spill_static` = 252（T65-H4）/ 448（T200-H8）/ 63（T1-H1）/
128（T63-H128）/ 0（T128-H4、T256-H1、T4096-H128、T0-H4）——与 §1.6.1 A4 收益栏及 §8.3 的「越界写闸门」
数字一致）。另核对脚本逐行的 `ulp(cand,base)` 列：**A1 12/12 全为 0.0**（逐位相同），
`ulp(cand,ref)` 的 3310 为 fp32 累加对 fp64 的固有偏差（基线同享）。
结论：**EQUIV_PASS，机器验证与文档自述一致**。

### 2. `design_calc_check.py` 复算（D-5）

命令：`python3 .agents/tools/design_calc_check.py --design examples/mla_varlen_bwd/_delta_kernel/DESIGN.md --task-dir examples/mla_varlen_bwd/_delta_kernel`

| check | status | detail（原文） | 本检视处置 |
|---|---|---|---|
| `ub_budget` | **skip** | `no per-level buffer tables parsed from the memory plan (advisory checker; verify manually)` | **人工复核通过**：§4.5 与 P1 生成码逐字节闭合——`o_ub` 2 版本 = 2×8192 half = 32768 B，`do_ub` 基址在 +16384 half（=32768 B）再 32768 B，`d_ub` 基址在 +16384 float（=65536 B）、64×8×4 = 2048 B ⇒ **67584 B = 66 KB / 253952 = 0.266**（`D_v=64`：`d_ub` 双版本 4096 B ⇒ 69632 B = 68 KB；`D_v=192`（实测行 pitch 256 half）：65536+65536+4096 = 135168 B = 132 KB，×1.7 悲观膨胀 229786 B，余量 9.5%，与 §9.1 R8 一致）。三档均 ≤ 253952 B。 |
| `l0c_budget` | **skip** | `no Ascend Cube GEMM call found in the design pseudocode` | **人工复核通过（不适用）**：本算子无 Cube / L0A/B/C 使用（§0.5、§2.2、§5.1），无 GEMM 分形约束可查。 |
| `core_split` | **skip** | `insufficient parsed dims {}; declared logical cores = None` | **人工复核通过**：逻辑核 `ntiles = H·ceil(T_q/64)` = **65536 / 64512 / 32768**（复用 §5.2 表逐项复算：128·512、128·504、64·512）；物理核 **56**——本检视重跑 `statectl platform --task-dir …` 返回 `observed_cores = {"cube": 28, "vector": 56}`、`core_source` 含 `vector_core_cnt=56`；并复核 `/usr/local/Ascend/cann-9.2.1/x86_64-linux/data/platform_config/Ascend950PR_9579.ini` 第 20–22 行 `ai_core_cnt=28 / cube_core_cnt=28 / vector_core_cnt=56`、第 50/51/58 行 `l0_c_size=262144 / l1_size=524288 / ub_size=253952`（与文档记载逐项一致）。每核 tile 数 65536/56 = **1170.3**、64512/56 = **1152.0**、32768/56 = **585.1**、80/56 = 1.43、24/56 = 0.43、8/56 = 0.14（算术全部复核正确）；均衡划分 `ntiles·bx//G` 两界差 ≤1。int32 上界复算：`ntiles ≤ 2^31/(G−1) = 39.0e6` ⇒ `T_q·H ≤ 2.498e9`（§9.1 R4 的 2.5e9 正确）。 |
| `r3_metrics` | **pass** | `all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak)` | 与人工复算一致（见维度 8）。 |

### 3. 设计期编译探针复核（补充，无卡）

命令：`cd /tmp/dr_varlen && ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python .../history_version/design_probe_packed_v2.py p1 p2 p3 p4`

- P1/P2/P3/P4 均 **COMPILE OK**（与 §3.5.4 一致）；P1 生成码含 `(it − it0) & 1` 版本偏移（`*8192` half）、
  `asc_copy_gm2ub_align(…, min(64, total_q − t0), 256, …, 2048, 256)`、
  `asc_copy_ub2gm_align(&delta[(it>>3)*64 + (it&7)*total_q], …, min(64, total_q − t0), 4, mode(4), 4, 32)`；
  §7.2 描述的 `asc_sync_notify/wait(PIPE_V↔PIPE_MTE2/MTE3, event = (it−it0)&1 或 0/1)` 与尾部收敛 wait
  **在本次重跑中逐条可见**（§7.2 的表成立，见维度 5）。
- 归档日志 `history_version/probe_v2_tails.log` / `probe_v2_p5_p6_p7.log` / `probe_v2_neg.log` 只保留描述符行与
  N1–N3 结论（`asc_sync` 命中 0 行），故 §7.2 的同步证据不在归档件内——本检视以重跑补证成立，
  建议（非阻塞）把一次 `p1` 全量 stdout 归档，见维度 5 注。
- 文档记载的两处数字与重跑不符（不影响结论，见维度 7 注）：§3.5.4 P1 行「63 行源」实测 **37 行**（63 是 P2 的行数，
  P2=63、P3=61 与文档一致）；§3.3/§5.2/§1.5 的 `D_v=192 → DW=192` 注释与公式 `DW = ceil(D_v/VL)·VL = 256`
  及 §4.5 的 132 KB 口径（按 pitch 256 half）矛盾。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

**核对方式**：Read 源算子全文（`bwd.py` 第 38–77 行工厂 + 第 122/137/147–148/222–224 行消费端 +
第 274–281 行 contract + 第 382–387 行 host 侧），与 §0.1–§0.8 逐项比对。

**语义理解正确性 — pass**
- 源码第 49 行 `delta_shape = (batch, heads, dense_seqlen) if dense_seqlen else (heads, total)`；
  §0.1/§0.2 取后者 `(heads, total)` fp32 —— 与源码、与消费端 `T.Tensor(stat_shape, accum)`
  （第 137 行 `stat_shape = (heads, total_q)`、第 147–148 行声明、第 122 行 `accum = "float32"`）
  以及第 222–224 行 `delta[h, q_start + m0 + i]` 三处**一致**。
- dtype 语义：源码第 58–61 行 `alloc_fragment(..., "float32")` + `T.copy` 提升 ⇒ 「fp16/bf16 无损提升到 fp32，
  乘与加皆 fp32」。§0.1「这是本算子最重要的一条语义」与源码一致（不是「fp16 乘 + fp32 累加」）。
- 规约语义：源码第 65 行 `T.reduce_sum(o_f, rows, 1)` 是 fragment 分组归约（非严格串行）；§0.1 明确写「分组归约」
  并只要求容差内等价，未把它误述为顺序累加 —— 与源码一致。
- 边界语义：第 68–72 行 `if row < total: … delta[h, row] = rows[i]`；§0.1 ① 正确指出 packed 下
  「运行期谓词是唯一尾处理手段」；⑤ 关于 `batch` 不参与取址（第 57 行 grid 为 `(heads, ceildiv(total, block))`）正确。
- host 侧：`MLAVarlenBwdKernel.forward`（第 385–387 行）**不做 view**，直接 `self._run(...)`；§0.1/§0.3 的
  「packed 不 view、`delta` 由 `out_idx` 分配、`target_cache` 按 `(batch, heads, dim_v, 0, dtype)` 特化」与源码一致
  （第 45 行 `@target_cache`、第 46–48 行签名、第 51 行 `out_idx=[2]`）。
- 工厂拒绝语义：第 274–281 行 `refusal()` 要求 `dtype ∈ ATTENTION_DTYPES`（`call_spec.py:50`
  `(torch.float16, torch.bfloat16)`）且 `dim_qk % 16 == 0 and dim_v % 16 == 0`；§0.1 ③ 与 §4.7 的工厂校验
  （`float32` 拒绝、`dim_v=48` 拒绝）与源码一致。

**算法解读完整性 — pass**
§0.3 的 6 步分解表逐条对应源码计算语句，无遗漏、无臆造：
① grid（第 57 行）② `T.copy(o[rows,h,:], o_f)`（第 62 行）③ 同 `do`（第 63 行）④ `T.Parallel(block, dim_v)` 乘
（第 64–65 行）⑤ `T.reduce_sum(o_f, rows, 1)`（第 66 行）⑥ `T.Parallel(block)` 谓词写回（第 67–72 行）。
数据流图覆盖全部 buffer（`o_f`/`do_f`/`rows`/GM `o`/`do`/`delta`），并正确指出「无 SMEM、无中间 GM 往返」。
host 侧逻辑（无 view、`out_idx`、缓存键、`pass_configs`/`compile_flags`，第 51–54 行）已列出。

**优化手段覆盖完整性 — pass**
§0.4 的 O1–O10 与源码逐项对得上，且未把实现层误判为算法层：O1（乘完即归约不落 GM，第 64–66 行）、
O2（fp32 fragment 提升）、O3（`alloc_fragment` 无 SMEM）、O4（`block = 32`，第 40 行）、O5（固定 `h` 的 256 B 段、
源间距 `H·D_v·2 B`）、O6'（运行期谓词）、O7（`reduce_sum` 的 warp 树）、O8（二维 grid + `threads=128`）、
O9（`@target_cache`）、O10（packed 的 head-major 布局，第 49 行）。未遗漏源码中的调度/编译配置项
（`TL_ENABLE_FAST_MATH`、`-O3 -DENABLE_BF16` 已在 §0.3 列出并在 §7.3 给出「不迁移」理由）。

**耦合性判定合理性 — pass**
§0.5 的 12 个条目均有四态处置 + 依据，且与平台事实一致：`T.Kernel(N)` 一维、签名无 `threads`
（本检视实测 `inspect.signature(T.Kernel)` → `(*blocks, prelude=None)`）；`T.simd` 无 warp/`__shfl` 语义；
容量常数取自 `Ascend950PR_9579.ini`（本检视复读第 50/51/58 行一致）；高层 `T.reduce_sum` 在 `T.SimdVF` 内的
布局差异有 `TRAP-950-simd-shared-reduce` 依据；SIMD 归约可行性有 `PL-950-simd-reduce.py` 依据（已读：`(2,8)` fp32
暂存、`pset(32,"PAT_VL1")`、`vsts(..., extent=1)`，正是 §0.6 R2/A3 的原型）。舍弃项 0 项且写明。

**NPU 重设计可行性 — pass**
R1（静态 56 核 + 核内动态 tile 流）、R2（逐行 `vcadd`，承接 dense 且给出「只依赖整行驻留一个向量寄存器」的
结构性理由）、R3（跨步 DMA + 运行期 extent）、R4（运行期 clamp，并论证 dense 静态 `nlast` 在 packed 下
**构造性不可用**）、R5（head-major 取址 + 运行期 count）五项均有「源方案 → NPU 新算法 → 语义保持论证」三要素。
语义保持骨架（数学等价 / dtype 语义 / 边界）对本算子成立：行与行无累加关系，重排只改归约分组与数据驻留位置。
NPU 新算法满足平台约束：一维静态核、`T.SimdVF` 无线程域、UB 66 KB、无 Cube。

**设计与决策一致性 — pass**
§1.4/§3.1/§3.3/§5.2/§5.5/§6 描述的是 §0.5/§0.6 决策后的形态（静态 56 核 + 动态 tile 流 + `nvec` 构建期展开 +
三处运行期 extent + head-major 压缩写回），无「§0 判重设计、§3 照抄源方案」的脱节。§0.8/§11 显式分区
复用项与 packed 差异项，并把「dense 的哪条结论不能复用」逐条列出（§11.3），未见连带复制失效结论的情况。

**golden 独立性 — pass**
§8.1 `golden_delta` 按 §0.1 语义朴素实现（`o.float()*do.float()` 沿最后一维求和后转置），
无 UB 分块、不模拟 `vcadd` 分组、不模拟描述符/暂存缓冲；`chunk_rows` 只限制 fp32 临时量、不改变数值路径
（每行和与行分块无关，成立）。与源 docstring「`sum_d o[row,h,d]*do[row,h,d]` in float32」逐字对应，
未复刻 §0.6 的 NPU 数据流。

**建议（非阻塞）**：§0.7 把「smoke 几何 `q_lens [256,384]`、`H=8`、`D_v=128`」归到
`tests/ops/test_mla_varlen_bwd.py`；该文件的 `[256,384]` 用例实为 `heads=4, dim_qk=192, dim_v=128`
（第 151 行），`H=8` 来自 manifest workload（`attention.yaml` MLAVarlenBwdOp 首条）。
建议改写为「manifest smoke（H=8）/ op 测试用例（H=4）」，避免 Stage 3/5 取错几何。
`test_mla_varlen_bwd_general_kernel` 的 `[256,128] / H=4 / D_qk=128 / D_v=64`（第 158–160 行）记载正确。

### 1. API 可行性 — pass

- 存在性（本检视实测于 `/workspace/venv-ascend950`，TileLang ascend 0.1.15，`ASCEND_RT_VISIBLE_DEVICES=""`）：
  `T.Kernel / T.Pipelined / T.serial / T.annotate_buffer_versions / T.clear / T.dynamic / T.copy` 均存在；
  `tilelang.ascend.language.simd` 的 `vld / vsts / vcvt / vmul / vadd / vcadd / pset / update_mask`
  全部存在（`vcgadd / vcpadd / vdintlv / vintlv` 亦存在——后者支撑 §1.6.3 候选 7 的「API 存在但代价倒挂」论断）。
- 签名匹配：`T.Kernel(*blocks, prelude=None)` —— 一维静态 `T.Kernel(56)` 合法、无 `threads` 参数（与 §3.5.1 一致）；
  `T.Pipelined(start, stop, num_stages=...)` 接受运行期 `PrimExpr` 界（P1/P7 实测生成
  `for (it = ntiles·bx/G + …; it < …; ++it)`）；`vcadd(src, mask=None, mode=...)` 为单寄存器内归约
  （支撑 V9「无跨行批量归约 API」的理由）。
- 编译证据：§3.5.4 的 P1–P7 全部 COMPILE OK（本检视重跑 P1/P2/P3/P4 一致），N1 复现
  `PlanMTECopy: source and destination regions must contain the same number of elements, got
  src=T.min(64, total_q - it // 4 * 64) * 64 and dst=8192`（原文见 `probe_v2_neg.log`），
  N2 复现「编译通过但 5 个 SIMD 记号全缺 + `d_ub` read-before-write 告警」，N3 复现
  `min(64, max(total_q − t0, 0))` 的自行 clamp。
- API 规范：全部按平台 ruleset 的 `tilelang.ascend.language` 取证（`references/programming.md` +
  `ascend/language/*.py` + `knowledge/repro/*`），未借用其他后端导出。

### 2. 内存层级规划 — pass

- 搬运路径完整（§4.4）：GM `o`/`do` →（单条跨步 DMA，`n` 段 × 256 B（`D_v=64` 时 128 B），源间距
  `H·D_v·2 B`，目的 region `[0:n, 0:D_v]`，目的行距 `DW·2 B`）→ UB 双版本；UB → 向量寄存器（`vld`）→
  `d_ub[i,0]`（`vsts` + `PAT_VL1` + `extent=1`）→ GM `delta`（压缩 DMA，`n`×4 B、源间距 32 B、目的连续 4 B、
  基址 `4·(h·total+t0)`）。无 L1 / L0C / 中间 GM 往返，与生成码逐条吻合（P1/P2/P3/P5 描述符参数、
  P1 `o` 源间距 2048 B、P5 `H=128` 源间距 32768 B）。
- UB 预算：67584 B / 253952 = 0.266（`D_v=64` 69632 B、`D_v=192` 135168 B，1.7× 悲观膨胀 229786 B 余量 9.5%），
  与 P1/P2/P3 生成码的 shmem 偏移（版本步长 8192 half、`do_ub` @16384 half、`d_ub` @16384 float、
  `d_ub` 版本步长 512 float）逐字节闭合（机械 skip 项已人工复核，见「机械复核」§2）。
- L0C：不使用 Cube ⇒ 不适用；L0A/L0B 同样未使用。
- Buffer 层级标注：§4.3 逐个标注 `o_ub`/`do_ub`（UB 双版本）、`d_ub`（UB，1–2 版本由编译器决定）、
  向量寄存器中间量（各 1 寄存器，A1 不物化 fp32 tile），层级明确。

### 3. Tiling 策略 — pass

- Block 划分明确：`block_M = R = 64`（行）、`block_N = D_v`（整行一个向量寄存器，不切分），
  取值理由（UB 预算 + `R` 承袭 dense 已验证预算，Stage 4 可试 32/128）齐备（§5.2/§1.6.3）。
- 对齐约束：行宽 `D_v·2 B` = 256 B（`D_v=64` 为 128 B）= 2/1 个完整 128 B 段；`vld` 整行 = 1 个 2048-bit 寄存器；
  fp32 中间量 64 lane；`d_ub` 行间距 32 B；写回目的连续 4 B（`[H,T_q]` 最内轴 = `t`）——均满足。
- 非整除处理：`T_q % R ≠ 0` 无静态尾段，统一 `n = min(R, T_q − t0)` 约束两条装载与写回，行循环上界静态 `R`
  且多算行不写出；`D_v < DW` 走切片 + 两侧 `T.clear`；`nvec > 1` 构建期展开（§5.4/§6.4）。纯 Vector 算子
  该说明已给出，无遗漏。
- 分核策略三要素（对照 `.agents/skills/_shared/standards/core-split-strategy.md` §1/§2.2）：
  ① 逻辑核数 `H·ceil(T_q/R)` = 65536/64512/32768/80/24/8（逐项复算正确）；
  ② 物理核数**实查记录齐全**（`statectl platform --task-dir …` 的返回值 + INI 行号与数值 + 明确声明
  `torch.npu.get_device_properties` 未执行且不猜值），本检视重跑两路取证均一致；
  ③ 规模判定「极大规模」（585–1170 倍）→ 固定启动 56 核 + 核内分批，量化依据 1170×0.5 µs ≈ 585 µs ≈
  流量下界 1717–2101 µs 的 28–34%（算术复核正确），且 `G = 56` 与 `vector_core_cnt` 恰 1 倍 ⇒ 无波次损失。
  核数与 block 取值自洽（`R=64`、`G=56`、`ntiles` 运行期）；核内循环边界满足平台限制（`T.Pipelined` 运行期界
  P1/P7 实测可编译；`T.serial(R)` 静态）。小几何（`ntiles < 56`）由均衡划分保证 `it0 ≤ it1`，无负 trip count。
  与备选（P6 静态化）的核数/划分不变，已说明。

### 4. 技术约束检测 — pass

- 三维 Kernel：不涉及（一维 `T.Kernel(G)`），无需 `block_metadata`。
- GEMM 非整除：不涉及（无矩阵乘，§5.1 类型判定为纯 Vector，Cube 已在 §1.6.0 C6 量化淘汰）。
- L0C 溢出：不涉及（不使用 Cube）。
- 其余项目已知限制（§3.5.1 八项）逐项覆盖且判定正确：`T.Kernel` 一维/无线程域、UB 253952 B、
  无 warp/`__shfl`、高层二维 `T.reduce_sum` 陷阱、整 buffer ↔ 整 buffer 元素数相等（N1 复现）、
  尾轴对齐（`D_v` = 128/64/192）、动态 shape（仅行轴）、fast-math（不开）。

### 5. 循环与同步 — pass

- 循环结构明确（§6.1）：核网格 `T.Kernel(56)`（block 级并行、静态）；tile 流 `T.Pipelined(it0, it1, num_stages=2)`
  （核内串行 + 流水、**运行期界**）；行方向 `T.serial(R)`（静态界）；行内 `d` 方向向量化；不使用 `T.Parallel`
  （`SimdVF` 无线程域）。四类循环的 API 选择均有理由。
- `T.Pipelined` 运行期界 + 双缓冲**成立**（本检视实跑 P1 验证）：版本偏移 `(it − it0) & 1`（`*8192` half）
  出现在 `o_ub`/`do_ub` 的装载与 `vld` 地址中；每核循环界为
  `ntiles·bx/G + (ntiles·bx % G >> 31)` 到 `ntiles·(bx+1)/G + …` 形态（`it0 ≤ it1` 恒成立）。
  与 dense 的「尾段关闭多版本」风险**不同类**：本函数主选只有一条代码路径（无尾段、无第二写点），
  §6.3 据此论证任意 `T_q` 都保持双缓冲；P6 的告警
  （`did not enable multi-buffering for buffer "o_ub"/"do_ub"`，见 `probe_v2_p5_p6_p7.log`）恰好只出现在
  「残块段在流水外写同一 buffer」的备选形态上，构成该结论的负向证据。
- 同步与模式匹配：§7.1 选「自动同步」，不写手动 flag；§7.2 的 6 个同步点与生成码一致
  （`asc_sync_notify(PIPE_V, PIPE_MTE2, 0/1)`、`asc_sync_wait(PIPE_V, PIPE_MTE2, (it−it0)&1)`、
  `asc_sync_notify(MTE2→V)`、`asc_sync_wait(MTE3→V, 0)`、`asc_sync_notify(V→MTE3, 0)`、尾部 4 个收敛 wait）。
  `d_ub` 单/双版本（P1 单、P2 双）两形态都有对应 flag，且 §9.1 R6 已登记。
- 尾块处理（§6.4）：按 `T_q % R == 0` / `≠ 0` / `T_q < R` / `T_q = 0` / `D_v < DW` / P6 备选六种场景说明，
  与 §1.4 步骤 6、§5.4 一致；无运行期谓词、无静态尾段、无双路径。
- **建议（非阻塞）**：§7.2 的同步证据未归档（三份日志 `asc_sync` 命中 0 行；本检视重跑 P1 才补证）。
  建议把 `design_probe_packed_v2.py p1` 的完整 stdout 写入 `history_version/`，使 §7.2 的「实测生成代码」可复现，
  避免 Stage 3/Stage 4 再花一次编译去确认同步形态。

### 6. 验证方案 — pass

- Golden：§8.1 给出可直接运行的 `golden_delta`（PyTorch，fp32，head-major），语义依据 §0.1
  （`(o.float()*do.float()).sum(-1)` 转置），与源 docstring 一致；独立性见维度 0。
- L0 门槛计划：4 条 L0（主路径 `q_lens [256,384] T_q=640 H=8 D_v=128`；**pad 路径**
  `q_lens [256,128] T_q=384 H=4 D_v=64`（与 `test_mla_varlen_bwd_general_kernel` 同几何）；
  极小几何 `T_q=1 H=1`；工厂契约用例），含 dtype（fp16 + bf16 双过）、按算子类别的精度标准与
  「L0 必须两 dtype 全过才进 L1/L2」的闸门。L1/L2/Boundary 分层覆盖变长拼接、manifest 主力 workload、
  `T_q % R ≠ 0`（65/200/96/63）、全零、非有限值，以及「越界写间接探测器」。
- 精度标准：输出为 fp32，`atol=rtol=1e-4`（模板 fp32 口径），并说明与本任务 `_tolerance(dtype)`
  （`test_mla_varlen_bwd.py` 第 47–50 行：bf16 `atol=max(5e-3,1.6e-2), rtol=1.6e-2`）的关系——本算子更严格，
  不会成为下游瓶颈。§9.1 R11 亦登记了抵消行（`Σ|p|` 大而 `|Σp|` 小）的固有绝对误差，属源算子同 dtype 性质。

### 7. 完整性与一致性 — warn

- 章节齐全：§0（0.1–0.7 + 0.8）与 §1–§11 齐备，模板要求的信息全部有落点；§0.5 处置汇总、
  §0.8 复用/差异分区、§11 dense 对照索引使「复用」与「重推」边界可查。
- 无占位符：全文未出现模板占位标记（花括号占位、英文待办标记、中文待补类标记），检索零命中。
- 同类引用真实：§3.5.3 的 6 条路径**全部存在**（`PL-950-simd-reduce.py`、`PL-950-vector-execution.py`、
  `examples/mla_dense_bwd/_delta_kernel/DESIGN.md`、`examples/mla_varlen_bwd/_bwd_kernel/DESIGN.md`、
  `examples/paged_cache_write/…/_paged_cache_write_kernel.py`、`examples/moe_permute_align/…/_make_scatter_kernel.py`、
  `kernel_template.py`）；§0.7 的 manifest 条目（`attention.yaml` MLAVarlenBwdOp 的 4 个 workload：
  `[256,384]/H8`、`8×4096/H128`、`512…16384/H128`、`4×8192/H64`）与 §5.2/§8.3 的几何逐项吻合；
  roofline 的 delta 项（`H·Σq_len·2·D_v` = 1.074 GFLOP、写出 `H·Σq_len·4` = 16.8 MB）与 §1.6.0 R3 同口径。
- 内部一致：API 映射 / 内存规划 / Tiling 三处**基本自洽**（`R=64`、`G=56`、`DW`、`nvec`、`d_ub` 形状、
  三条描述符 extent 形态在 §3.2/§3.3/§4.3/§5.2/§6 一致），但有下列文档级不一致（非阻塞，建议修正）：
  1. **`DW` 注释算术错误**：§1.5「`DW = ceil(D_v/128)·128`（`D_v=192 → 192`）」、§3.3 与 §5.2 的同款注释
     均写 `D_v=192 → 192`，而 `ceil(192/128)*128 = 256`；§4.5 的 132 KB 预算按行 pitch 256 half（=512 B）
     计算、探针 P3 的装载目的行距亦为 512 B ⇒ 正确值是 **DW=256（`nvec = DW//VL = 2`）**。
     建议把三处注释改为 `D_v=192 → 256（nvec=2）`，并保留公式为唯一事实源（否则按注释实现会得 `nvec=1`，
     漏掉 128..191 列；该几何不在 manifest/smoke 中，故不影响本任务几何，但仍是 Stage 3 的实现陷阱）。
  2. **探针行数与重跑不符**：§3.5.4 P1 行写「63 行源」，本检视重跑为 37 行（P2=63、P3=61、N2=24 与文档一致）。
     建议订正为 37，或在日志中保留行数出处。
  3. **§0.7 的 smoke 几何归属**（见维度 0 建议）：`H=8` 属 manifest workload，测试文件用 `heads=4`。

### 8. 算法优化分析 — pass

**§1.6 章节存在性 — pass**：含 §1.6.0（调研四问）/ §1.6.1（数学等价优化 A1–A4 + A2b 对照）/
§1.6.2（向量化替代 V1–V13）/ §1.6.3（向量化轴与布局决策 + 实验裁决三件套）。

**算法调研存在性与完整性 — pass**
- R1 候选表含**基线 C0**，并覆盖参考表命中行（C7）+ 恒等变形空间（C1/C2/C3/C3b）+ 反例
  （C4 fp16 乘、C5 两遍、C6 对角 GEMM、C8 D 轴跨核切分）+ 结构排除（C9 无代数化简、C10 按请求分段）。
  本检视核对平台 `knowledge.algorithm_candidates`（`.agents/platforms/toolchains/ascend/knowledge/algorithm-candidates.md`
  第 9–11 行）：reduction 族确实只有「单行 SIMT sum/max reducer」（→ `PL-950-row-primitives`）一行，
  已被 C7 纳入评估 ⇒ **参考表命中行无遗漏**。
- R2 明确结论「**有**」在线算法，并给结构依据（逐行点积每行 `O(1)` 状态、行间无依赖、单遍 1 次扫描、
  `D_v>128` 段合并在构建期展开、`total` 动态只改「流有多长」不改流式结构），未作无依据的负向断言。
- R3 四口径齐全且基线在表（FLOPs / 访存量 / 扫描遍数 / 中间缓冲峰值 + 可并行度/跨核代价）。
- R4 逐候选六项检查（计算单元/容量/对齐/静态边界/流水融合/跨核），负向淘汰均有量化依据或已标注未实测外推；
  容量/带宽淘汰引用的 `CONST-capacity-910B2C`、`CONST-mte2-degradation`、`CONST-vector-launch-overhead`
  在 pattern-library 中真实存在，且均为 910B2C / tilelang 0.1.2 时代条目 —— 设计**显式标注「非本型号实测」**，
  本检视查阅 `constants.md`（224 行，13 个条目）确认其中**不存在** 950PR 实测常数 ⇒ 引用他型号常数并标注外推
  是当前知识库状态下的正确做法，不构成「漏查 pattern-library」。
- 调研结论含选定算法（C0 结构 + C1/C2/C3/C3b 实现级优化）、关键依据（访存 2.164 GB 为理论下界，
  C5/C6/C8 分别引入 ×2.98 流量、×64 FLOPs、跨核原子；C7 每行一次 barrier 在 4.19M 行规模不可接受）、
  「无更优替代」的调研范围（algorithm-candidates 全表、INDEX、ALG-950-formulas/operator-patterns/
  PL-950-row-primitives、pattern-library constants/cases、本仓 ascend 向量算子、`kb_search` 三组关键词）。
  `kb_search "row dot product reduce delta o*do"` 本检视重跑 → `no entries matched`（检索记录可复现）。

**调研结论正确性（独立复核）— pass**
- ① 等价性：见下「等价性论证独立推演」，四项采纳项推演均成立。
- ② 复杂度算术独立复算（全部对上）：`N = T_q·H = 32768·128 = 4.194304e6`；
  FLOPs `2·N·D = 1.073741824e9` = 1.074 G ✓；读 `2·N·D·2 = 2.147483648e9` = 2.147 GB，写 `N·4 = 16.777 MB`
  ⇒ 2.164 GB ✓；C5 = 2.147×3 + 0.0168 = 6.458 GB，比值 6.458/2.164 = 2.984（文档 ×2.98）✓；
  C6 = `2·R·N·D` = 68.7 G = 64×（`R=64`）✓；C8 原子流量 `2·4·4.194304e6` = 33.55 MB ✓；
  中间缓冲 66 KB = 32768+32768+2048 ✓（基线 32 KB = 2×32×128×4 ✓）；发射 `11×4.194304e6 = 46.14 M`、
  `/56 = 823.9 K`、`/1.8 GHz = 457.7 µs` ✓；描述符 `65536×3 = 196.6 K`、`/56 = 3511` ✓；
  流量项 `2.1643 GB / 1.03 TB/s = 2101 µs`、`/1.26 TB/s = 1717 µs` ✓；核启动 `1170×0.5 µs = 585 µs`
  = 下界的 27.8%–34.1% ✓。**未发现漏计中间缓冲 GM 往返**（本设计的中间量全在 UB/寄存器，
  C5 的 GM 往返已计入正表，属淘汰项）。
- ③ 亲和性判断与 ascend-constraints / pattern-library 无实质矛盾；负向淘汰有佐证
  （C7 每行 barrier 的仿真由 `PL-950-row-primitives.py` 的 SIMT 用法支撑；C6 的 fractal N≥16 与
  「每行 1 输出」冲突结构性成立）。
- ④ 负向断言复核：本设计**未作**「无在线变体 / 无化简公式」这类断言（R2 答「有」，C9 的「无代数化简」
  以信息量下界 `D` 乘 + `D` 加结构性论证）；「无更优替代」附调研范围 ⇒ 不构成可被推翻的负向断言。
- ⑤ 并列候选裁决：判定裕度的分歧点不在算法族选择，而在实现形态（运行期 count vs 静态 extent），
  已交 §1.6.3 实验裁决计划（非纸面拍板）。
- **建议（非阻塞）**：§9.1 R1 把「跨步描述符 MTE2 带宽低于连续描述符」列为风险项，但 pattern-library
  已实测条目 `CONST-gm-to-ub-bw-dilution`（910B2C，copy-only A/B：(M,tn) 跨步列块读 vs per-row 读墙钟
  0.98×，「读形态无差异」，前提是段宽 ≥ 数百字节 —— 本例 256 B 满足）与该前提相反。建议 R1 引用该条目、
  把「带宽偏低」降级为「已测不慢、仅段宽 < 数百字节时退化」，同时保留 L2 驻留口径（`CONST-mte2-degradation`
  的 sets=1 条目，§1.6.0 估算注已提及）作为乐观偏差来源。该改动只影响风险叙述，不改变任何候选裁决
  （§1.6.3 候选 1 的「256 B = 2 个完整 128 B 段 ✅」本就与该实测一致）。

**调研→设计一致性 — pass**
§1.6.0 选定算法实际驱动：§1.4（六步全部按「单遍流式 + 逐行 64 lane 树归约 + head-major 批量写回」描述）、
§3.1（步骤 1–8 以 §1.6.1 优化后公式为输入：`vld→2×vcvt→2×vmul→vadd→vcadd→vsts` + 单描述符写回）、
§6（tile 流 `T.Pipelined` / 行 `T.serial(R)` / 行内向量化，与 §1.6.2 结论逐项对应）。
迁移侧：§0.5 的处置（保留 7 / 等价替换 3 / 重新设计 2 / 舍弃 0）与 §1.6.0 选族一致；源优化手段 O1–O10
逐项写出「意图由什么结构承接」（§0.5 表 + §0.6 各 R 项「源优化意图承接」句），**无静默丢弃**。

**数学等价优化完整性 — pass（四项四要素齐备）**
A1 寄存器融合（原式 → 不物化 fp32 tile → 同一 fp32 乘法/同一批项 → 64 KB UB + 每 tile 两次片上往返归零）、
A2 实现分组（顺序/多级合并 → `vadd(vmul(EVEN),vmul(ODD))` + 1×`vcadd` → 容差内等价 + 无系统偏置 →
多级合并 → 1 条 `vcadd`，并量化 SIMT 路径的 barrier 代价）、A3 批量写出（`R` 次 4 B 标量写 → 32 B 行距暂存 +
1 条压缩描述符 → 纯数据搬运 → 指令数 N → N/R）、A4 packed head-major 写回 + 运行期 count
（逐行谓词写 → 单描述符写 `delta[h, t0:t0+n]` → 索引映射/覆盖/域外/非有限值五项机器验证 →
1 条描述符/tile，并量化静态 extent 的越界量 252/448/128/63）。其余空间（乘/加/提升/索引）给出
「无 CSE、无除法/超越函数可降代、每元素只读一次、行间独立」的「无进一步优化空间」结论与依据。

**等价性论证独立推演（逐项，不采信文档自述）**
- A1：两形态对每个 `d` 都执行同一个 fp32 乘法并四舍五入到 fp32，再对同一批项求和；唯一差异是值驻留位置
  ⇒ **逐位相同**。脚本逐行输出的 `ulp(cand,base)` 列 12 个用例**全为 0.0**（推演成立）；表头 `ulp(cand,ref)`
  的 3310 是 fp32 累加相对 fp64 的固有偏差（基线同享，`ulp(base,ref)` 同量级），不构成 A1 与基线的差异。
  故 §1.6.1「cand 与 base 0 ulp，12/12」的表述准确。
- A2：`vcadd` 64 lane 二叉树 = 同一批 fp32 项的重新分组，误差项仍是 fp32 舍入、无系统偏置；
  lane k = `p[2k]+p[2k+1]` 与 `vcvt` PART_EVEN/ODD 语义自洽（无论 PART_* 取「偶/奇 lane」还是「低/高半」，
  128 个乘积全部且仅被计入一次 ⇒ 和的期望不变，仅分组不同，故推演对所依托的具体语义稳健）。
  A2 与 A2b 在 `nvec == 1` 下逐位恒等的声明与脚本数字（两者 max ulp 均 2798、违反率均 0）一致。
- A3：纯数据搬运，索引映射 `d_ub[i,0] → out[i]` 重跑 0 ulp、违反率 0。
- A4：`(h, t0+i) ↔ d_ub[i,0]`（`i < n`）映射 + 覆盖性 + 域外清洁性 + `n ≥ 1` 由脚本 8 个几何全部
  `bit_equal / covered / clean_outside`；空输入（`T=0`）两形态皆空。推演成立，且方向性论据
  （静态 extent 会跨 head 污染）由 N3 + `spill_static` 量化支撑。

**向量化替代覆盖完整性（与 §3.3/§6 交叉核对）— pass**
§1.6.2 的 13 点覆盖 §3.3 伪代码与 §6.1 循环表的全部计算点：V1 乘 / V2 提升 / V3 两半合并 / V4 行归约 /
V5 暂存写 / V6 装载 / V7 写回 / V8 pad 清零（八项已替代，API 均有本平台证据：`simd.py` 导出 +
`PL-950-simd-reduce.py` 已验用法 + 本任务 P1/P2/P3 生成码）；保留五项 V9 行循环（API 缺失：`vcadd` 单寄存器，
`vcgadd/vcpadd` 同为单寄存器内归约，已核导出；跨行批量归约需 64×128 转置链 `vdintlv/vintlv`，代价 ≥2 条/行
换 1 条 `vcadd`）、V10/V12 block 级索引（每 tile 6–8 条标量整数指令 vs 64 行 × 11 条向量指令）、
V11 运行期 extent（动态 shape 边界，0 条逐元素指令）、V13 构建期分支不存在。**§6 中未出现 §1.6.2 未论证的
逐元素标量循环**：`T.serial(R)` 与三处 `T.copy` 的 extent 均有对应行，`T.Pipelined` 的 tile 流属任务并行
（其替代方向已被 R1 C8/C10 量化淘汰）。

**不可替代理由充分性 — pass**：五项理由分别归于「API 缺失（附查证过程与量化）」「block 级索引」
「依赖动态 shape 的边界处理」「不适用」，无「实现简单 / 照源码写」式空话。

**替代 API 可行性 — pass**：`T.simd.vld/vcvt/vmul/vadd/vcadd/vsts/pset`、`T.copy`、`T.clear`、
`T.annotate_buffer_versions`、`T.Pipelined(it0, it1, num_stages=2)` 均在本平台文件/实测中取证
（维度 1 已实测导出；P1–P7 编译通过）。

**向量化轴与布局决策完整性 — pass**
候选矩阵 8 行全部枚举，规约类必选的两种 lane 映射均在表内：候选 4「lane = 行、`D` 串行（垂直扫描）」、
候选 7「核内转置后批量归约」（水平归约的另一形态），并与「选哪条轴」（候选 1/2/6）正交枚举；
两遍/online 交互由 R1 C5（两遍物化）+ R2（单遍）覆盖。逐候选有量化评分（整除性 0%/100% 尾 lane 浪费、
累加链形态、repack 代价「无」的机制、UB 容量 66 KB vs 全头布局 32 KB/行 ⇒ 双缓冲只能 `R=1`）；
弃选均有量化理由（候选 2：写放大 8–32×；候选 3：原子次数 N、+33.6 MB；候选 4：向量化率 1/128、≥8× 慢；
候选 5：FLOPs ×64；候选 6：2.1 GB 级额外读写；候选 7：省 9% 指令 vs ≥2 条/行转置；候选 8：关闭双缓冲）。
**弃选前提亲自核对**：候选 7 依赖的 `vdintlv/vintlv` 确在 `tilelang.ascend.language.simd` 导出；
候选 4「垂直扫描」在本平台的劣势属结构判定；候选 6 禁止 host 改内容的前提与平台「host 只允许 stride/shape
视图」一致。I/O layout 被当作**契约**（消费端 `_bwd_kernel` 的 `[H,T_q]` 连续段）而非核内布局依据，
核内布局作为设计变量（`[R, D_v]`，向量轴 `d`）——口径正确。
**实验裁决三件套完整**：主选（候选 1 + 统一运行期 clamp + `T.Pipelined` + `G=56`）与备选（P6 静态化：
满段静态 `R` + 每 head 一段残块、buffer/UB/循环/repack/dtype/分核三要素均按新形态给出，达可直接实现深度）
并列，裁决计划含代表 shape（`ds-v3-8x4k`/`ds-v3-mixed`/`kimi-k2-4x8k`/smoke）、指标（`msprof` Task Duration，
A/B 交错 ≥5 次）、**判定阈值**（备选 < 主选 × 0.9 才翻转）、按 shape 分片回写路径，并评估了 shape 特化分派可行性。
判定依赖的未实测常数（运行期 count 的描述符代价；双缓冲 vs 静态 extent）已如实标注为未实测外推。

**布局决策与实现一致 — pass**：§3.3 的 `row_dot` 以整行 `vld`（`o_ub[i, v*VL]`）为向量轴 `d`，
buffer 形状 `(R, DW)`、`(R, 8)`、`nvec` 构建期展开与 §4.3 的层级标注、§4.5 的 UB 预算、§6.1 的循环结构三方一致；
重排路径「无」在 §3.3/§4.4 有对应实现（跨步描述符 + 运行期基址乘法）。唯一待修的是 `DW` 注释算术
（维度 7 建议 1），公式与预算本身一致。

**与下游章节一致 — pass**：§3.1 明示「输入公式为 §1.6.1 的优化后公式（A1+A2+A3+A4）」，8 个步骤与
优化后伪代码逐条对应；§6 循环结构与 §1.6.2 结论一致（V9 被判「不可替代」的行循环在 §6.1/§3.3 中
正是 `T.serial(R)`，无额外未论证的标量循环）。

## 结论依据（风险优先汇总）

- 会直接导致 Stage 3 编译/运行/精度失败的阻塞级问题：**0**。
  - 编译面：P1–P7 全 COMPILE OK（本检视重跑 P1–P4 一致），N1/N2/N3 三个反例已复现并给反制约束
    （目的端切片、语句必须在 prim_func 体内、写回 count 必须运行期）。
  - 运行面：一维静态核 + 运行期 tile 流的划分恒非负（`it0 ≤ it1`）；`n = min(R, total−t0) ≥ 1`
    由 `ntiles` 构造保证；int32 上界余量 >500×。
  - 精度面：A1/A3/A4 逐位等价，A2 容差内等价且树形更接近 fp64 真值；golden 独立、fp32 口径与下游
    `_tolerance` 相容。
- **A4 的「未文档化 clamp」风险判定：可接受，不构成设计阻塞，也不需要额外的前置探针。** 理由：
  ① 主选**不依赖**该行为——写回 count 显式写 `n`（N3 只作为反例登记，§3.3 注 5/§9.1 R5）；
  ② 该映射的正确性已有机器验证（8 几何逐位相同 + 覆盖恰好 + 域外清洁 + `n ≥ 1`）；
  ③ 越界风险已有数值闸门（§8.3 的 `T_q=65/200/96/63` boundary 用例 + 「越界写间接探测器」）。
  Stage 3 只需按 §3.3 写 `T.copy(d_ub[0:n, 0:1], delta[h, t0:t0+n])` 即可；若实现者改用静态 `R`，
  则由 N3 与 A4 诊断列已量化其后果（跨 head 污染）——该约束已在 §3.3 硬约束 ③ 与 §9.2 明确写出。
- 建议级 5 项（文档准确性/证据归档，不影响算法与实现决策）：§0.7 smoke 几何归属；§3.5.4 P1 行数 63→37；
  §1.5/§3.3/§5.2 的 `D_v=192 → DW=256` 注释订正；§7.2 同步证据归档；§9.1 R1 引用
  `CONST-gm-to-ub-bw-dilution` 实测条目后下调该风险。
