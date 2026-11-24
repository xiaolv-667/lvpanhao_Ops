# _dq_land_kernel 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息

- 文档: examples/mla_dense_bwd/_dq_land_kernel/DESIGN.md（1089 行）
- 任务类型: 迁移（MLADenseBwdOp 第三 launch，harness 子模式）
- 源算子: `/workspace/TileOPs/src/tileops/_kernels/attention/mla/bwd.py` 第 78–106 行 `_dq_land_kernel`（并核对 `_bwd_kernel` 的 `dq_acc` 契约 L151/L254/L342-346、host `_compile` L349-355）
- 目标平台: ascend-950pr-9579 / `ascend`（CANN 9.2.1，dav-3510，Cube 28 / Vector 56，UB 253952 B）
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 5（均列在对应维度说明与「补充」节内，不构成阻塞）

## 机械复核（D-1/D-5）

- **verify_equiv 重跑: PASS**。`cd examples/mla_dense_bwd/_dq_land_kernel && /workspace/venv-ascend950/bin/python verify_equiv.py` 返回 0，打印 `[EQUIV_PASS]`。逐行结果与 §1.6.1 内嵌表**完全一致**：fp16 A1/A2/A3 = `0 ulp / 0.0 / 0`，C1 = `3857 ulp / 1.216e-03 / NOT_EQUIV`，C2/C3 = `1 ulp / 9.729e-04 / NOT_EQUIV`；bf16 A1/A2/A3 = `0 ulp / 0.0 / 0`，C1 = `1 ulp / 0.0 / WITHIN_TOL`，C2/C3 = `1 ulp / 7.297e-04 / NOT_EQUIV`；`[baseline]` 两行均为 `ulp=0 viol=0`。脚本断言「采纳项全 EQUIV_PASS 且对照项全被证伪」成立（被证伪的 3857 ulp 对照项已按要求保留在表内）。
- **design_calc_check**: `python3 .agents/tools/design_calc_check.py --design examples/mla_dense_bwd/_dq_land_kernel/DESIGN.md --task-dir examples/mla_dense_bwd/_dq_land_kernel`
  - `ub_budget`: **skip** — "no per-level buffer tables parsed from the memory plan"；**人工已核**：`2·64·192·4 + 2·64·192·2 = 98304 + 49152 = 147456 B ≤ 253952 B`（58.06%），与 §4.5 一致。
  - `l0c_budget`: **skip** — "no Ascend Cube GEMM call"；**人工已核**：纯 Vector 算子，无 L0C 使用，n/a。
  - `core_split`: **skip** — "insufficient parsed dims {}; declared logical cores = 64"；**人工已核**：五个 workload 逻辑核数 64 / 8192 / 32768 / 8192 / 3072 复算正确，物理核 56（`statectl` 的 `hardware.observed_cores`）实查记录存在，方案为 persistent 固定 56 核 + 核内串行。
  - `r3_metrics`: **pass** — "all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak)"。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

检视方式：亲自 Read `bwd.py` 全文（L1–390）与 host 侧 `mla_dense_bwd.py`（L180–192、L202–208），逐项与 §0 核对。

- **语义理解正确性 pass**。源码 L81 docstring `dq = dq_acc * scale in the input dtype`；L94 `dq_acc: T.Tensor((total, heads, dim_qk), "float32")`；L100-101 `acc[i, d] = acc[i, d] * scale`（fp32 fragment 上做 fp32 乘）；L102 `T.copy(acc, dq[...])` 在 copy 内完成一次窄化。§0.1 的「fp32 乘 + 单次 RNE 窄化，不是两次舍入」与源码一致；「±inf/NaN 无特判」与 L100-102 无 `if_then_else`/clamp 一致；「`T_q` 动态」与 L90 `total = T.dynamic("total_q")` 一致；溢出阈值 65520（65504 与不可表示的 65536 的中点）为 IEEE fp16 事实，与 golden（torch `.to(dtype)`）一致。§0.2 的 I/O（`(total, heads, dim_qk)`、fp32 入 / `dtype` 出、同 shape）与 L94-95 及 manifest `outputs.dq: [B, S_q, H, D_qk]` 一致。
- **算法解读完整性 pass**。§0.3 的 S1–S4 覆盖 L95-102 的全部计算语句（网格坐标解码 / 跨步 region 搬入 / `T.Parallel(64,192)` 逐元素乘 / 搬出并 cast），无遗漏、无臆造；host 侧逻辑（`@target_cache(maxsize=32)` L78、`out_idx=[1]` L84、`_run` 的 `torch.zeros` L342 与 `land(dq_acc)` L346、`_compile` 的 `(heads, dim_qk, scale, dtype)` 特化 L354）均已列出，并核对 `mla_dense_bwd.py:186` 确为 `land = _dq_land_kernel(self.heads, self.dim_qk, self.scale, self.dtype_str)`，与 §0.1「调用契约」一致。§0.3 的「无归约、无 K 循环、无 Pipelined、无 SMEM」与 L95-102 一致。
- **优化手段覆盖完整性 pass**。§0.4 的 9 项逐条可回溯到源码：二维网格 L96、`block = 64` L87、copy 内融合 cast L102、`T.Parallel(64,192)` 摊平 L100、`threads=128` L96、`TL_ENABLE_FAST_MATH` L88、`compile_flags` L89、`out_idx` L84、无谓词热路径 L100-102。未发现被漏识别的源码优化（`@target_cache` 属框架缓存、已在 §0.3 提到，不计为算法优化）。`dq_acc` 的上下游契约（`_bwd_kernel` L151 声明 fp32、L254 `T.atomic_add` 跨 CTA 累加、L342 host 侧 `torch.zeros(q.shape, float32)` 连续）与 §0.1/§4.1 描述一致。
- **耦合性判定合理性 pass**。#1（二维网格→一维）经 `kernel.py:139-156` 实查确认（一维、无 `threads`、多维被拒）；#3（copy 内 cast）经 `ascend/analysis/vf_checker.py:105/148` 的拒绝文本实查确认（§0.5 引用的报错原文属实）；#5/#6/#7 舍弃有平台依据；#8（`out_idx`）经 `workflow.md:31` 确认。四态处置齐全、无「判保留却无平台依据」项。
- **NPU 重设计可行性 pass**。RD1/RD2/RD3 均有「源方案 → NPU 新算法 → 语义保持论证」三要素。独立复核 RD1 的互斥与覆盖：`cpt = ceildiv(ntiles,56)`、区间 `[bx·cpt, min((bx+1)·cpt, ntiles))` 对 `bx` 互斥（`it1 ≤ (bx+1)·cpt`）且并集覆盖 `[0, ntiles)`；`it → (h, t0)` 在 `t0` 为 64 倍数时双射 → 每元素恰被写一次，论证成立。RD3 的 `n = min(R, T_q - t0)` 双侧 clamp 与 e15 设备码一致（见维度 5），且 `t0 < T_q` 恒成立（`it < ntiles` ⇒ `it//H < ceildiv(T_q,64)`），故 `n ≥ 1`，无需零长度分支。
- **设计与决策一致性 pass**。§1.4/§1.5/§2/§3.2/§3.3/§4/§5/§6/§7 描述的全是 §0.6 决策后的形态（一维 56 核 + `T.Pipelined` 2 版本 + VF 段链 + 运行期 clamp），未出现「§0 判重设计、§3 照抄源方案」的脱节；§3.3 伪代码与探针 `history_version/design_probe_dynamic.py::make_e15`（L231-266）逐句对应（含 `total = T.dynamic("total_q")` 复用、`T.serial(NSEG)` 段循环、双侧 `n` 切片）。
- **golden 独立性 pass**。§8.1 的 `(dq_acc.detach().cpu().float() * float(scale)).to(dtype)` 只依据 §0.1 语义，无 `vcvt`/tile/核数/`T_q` 假设；独立复核该式与源码语义等价（fp32 乘 + 单次 RNE），因此即使推翻 §0.6 的 NPU 算法 golden 不变。

### 1. API 可行性 — warn

API 名称与参数全部在**当前安装包**（`/workspace/venv-ascend950/.../tilelang/ascend/language/`）实查到，无凭记忆：

| §3.2 的 API | 实查证据 | 结论 |
|---|---|---|
| `T.Kernel(G)` 一维、无 `threads` | `kernel.py:139-156`（"Exactly one dimension is allowed; a multi-dimensional launch is rejected"） | ✓ |
| `T.simd.pset(elem_width, dist)` | `simd.py:111`，docstring 列 `PAT_VL1..PAT_VL128` | ✓ |
| `T.simd.vld(addr, dist="NORM")` | `simd.py:324` | ✓ |
| `T.simd.vmuls(src, scalar, ...)` | `simd.py:1462`（A2 的「向量×标量」形态存在） | ✓ |
| `T.simd.vcvt(src, dtype, mask, round="ROUND_R", sat=True, part=0, mode)` | `simd.py` vcvt docstring：`ROUND_R` = RNE；`sat` 默认 True（印证 §1.6.1 A3「默认饱和」的前提）；`part` PART_EVEN=0 | ✓（参数成立，**数值语义未在真机核验**，见下） |
| `T.simd.vsts(addr, src, mask, dist, extent)` | `simd.py:457`（`NORM_B16` 与 `extent` 覆盖 access_ptr 足迹，与 §3.2 用法一致） | ✓ |
| `T.annotate_buffer_versions({buf: 2})` | `annotations.py:19-31`（1 = 退出多缓冲，≥2 = 固定版本数） | ✓ |
| `T.Pipelined(start, stop, num_stages=)` | `language/loop.py:112`（动态界为 PrimExpr） | ✓ |
| `tilelang.compile(..., out_idx=[1])` + 动态首维 | `workflow.md:31`；e15 **COMPILE OK** | ✓（编译级） |
| 「64 lane fp32 = 2048 bit」 | `simd.py:_vreg_lanes` = `2048 // elem_bits`（fp32→64、fp16→128） | ✓（§1.6.3 A4 的弃选前提成立） |

warn 原因（**建议 1**）：`vcvt` 的安装包 docstring 对 `sat=False` 写的是 "No sat; out-of-range values **wrap**"，而 §1.6.1 A3 把它表述为「非饱和 = IEEE，`|y| ≥ 65520` → ±inf」。设计已在 §9.1 R1 预料到这一分歧（"若 `RS_DISABLE` 不是 IEEE（例如把溢出'回绕'）"）并给了 L0-6/P4 判据与 A3（SIMT `T.cast`）完整备选，因此**不是阻塞**；但建议把 docstring 的 "wrap" 措辞显式引入 §1.6.1 A3 与 §3.4，并把 A3 的等价性结论限定为「模型层成立、设备层待 L0」，避免读者把它当作已证实的硬件等价。附带：§3.4 该行交叉引用写成「见 §9.1 风险 R2」，实际应为 R1（vcvt 语义）。

### 2. 内存层级规划 — pass

- 搬运路径完整（§4.4）：GM(fp32) —MTE2 `T.copy`→ UB `acc` —V `vld/vmuls/vcvt/vsts`→ UB `out` —MTE3 `T.copy`→ GM(dtype)；不经 L1/L0A/L0B/L0C，与纯 Vector kernel 一致。e15 设备码证实跨步 region DMA 形态：`asc_copy_gm2ub_align(..., min(64, total_q - …), 768, …, 1536, 768)`（行内 768 B、源行 stride 1536 B = `H·192·4`，`H=2` 探针）与 `asc_copy_ub2gm_align(..., min(64, …), 384, …, 768, 384)`。
- UB 预算（§4.5）：`acc` 49152×2 + `out` 24576×2 = **147456 B / 253952 B = 58.06%**（与设计 58.1% 一致）；余量 106 KB。`R=96` 时 221184 B（×1.12 = 247726 B）已逼近上限，§5.4「不要调大 R」的依据复算成立；3 版本 216 KB（×1.12 ≈ 242 KB）不取，复算成立。
- Buffer 层级标注齐全（§4.3：`acc`/`out` 均 `shared.dyn` = UB，掩码在寄存器）。L0C 容量检查 n/a（无 Cube）——机械检查 `l0c_budget` skip，人工已核。§4.5 的 ×1.12 膨胀系数来自 `CONST-capacity-910B2C`（910B2C 口径），设计已显式标注为跨型号假设；平台 `design_budget.ub_inflation` 本型号为 `null`，故 58.1% 用原始字节核算是正确的，膨胀行只作悲观上界，口径无误。

### 3. Tiling 策略 — pass

- Block 划分明确（§5.2）：`R = 64` 行/tile、`DW = roundup(D_qk,64) = 192`、`NSEG = 3`、tile = 1 head × 64 行，取值理由为 UB 预算（可复算，见维度 2）。
- 对齐约束：行内 fp32 768 B / fp16 384 B 均 32 B 整数倍；`192 = 3×64` 对 64-lane 零浪费。
- 非整除处理（§5.2/§6.4）：`n = min(R, T_q - t0)` 同时用于两条 copy 与两侧 UB 切片，行循环保持静态 R（多算行不搬出）；不依赖 host padding/crop，与「`out_idx` 由输入绑定 shape」的契约自洽。
- **分核策略三要素齐全**（§5.5，按 `.agents/skills/_shared/standards/core-split-strategy.md` §1 核对）：① 逻辑核数 `ceildiv(T_q,64)·H` 并给出五个 workload 的数值（我复算：512/8→64、4096/128→8192、16384/128→32768、8192/64→8192、12288/16→3072，全部正确）；② 物理核数**实查记录**存在（`statectl` 的 `core_source`：`Ascend950PR_9579.ini [SoCInfo] ai_core_cnt=28, cube_core_cnt=28, vector_core_cnt=56` + `rtGetAiCoreCount()=28`，`observed_cores.vector = 56`；我重新执行 `statectl platform --task-dir …` 得到同一值），并给出 Stage 3/4 的 `torch.npu.get_device_properties` 查询代码与「不乘 2、不用 NPUUtils」（合 `workflow.md:55`）；③ 规模判定为**极大规模**（最小 workload 逻辑核数 64 已 > 56，且 R 受 UB 限制最大 64、无法压核数——`585·192·12 ≈ 1.29 MB` 复算成立），方案为固定 `G = 56` + 核内串行连续区间 + `T.Pipelined`。
- 负载均衡：`cpt = ceildiv(ntiles,56)` 使 makespan = cpt，`ds-v3-4k` 147 / 均值 146.29（+0.5%）、`ds-v3-b2-8k` 586 / 585.14（+0.15%），复算一致；末核余量更少（107 / 538）不影响 makespan。退化路径 `ntiles < 56` 时 `it1 = min(it0+cpt, ntiles)` 产生空区间（如 ntiles=1 时 bx≥1 全部 `it0=1, it1=1`），无需额外分支——复算成立。
- 核内循环满足平台边界：只用 `T.serial` / `T.Pipelined`，动态界已由 e15 编译验证，不用未认证的 `ascend.batch_schedule` 跨 block barrier。

### 4. 技术约束检测 — pass

- 三维 Kernel：不涉及（一维网格，§6.1）；无 block_metadata 需求。
- GEMM 非整除 / L0C 溢出：不涉及（无 `T.gemm`、无 L0C），已在维度 2/3 说明。
- 逐条核对 §3.5.1 的 9 条限制：`T.Kernel` 一维（`kernel.py:139` ✓）、DMA 不能 cast（`vf_checker.py:105/148` ✓）、整 buffer↔整 buffer 拷贝要求元素数相等（`programming.md` 的 copy 条目 ✓，本设计全部用显式 region）、`dim_qk % 16 == 0`（`bwd.py:277` ✓）、行首 32 B 对齐（✓）、`TL_ENABLE_FAST_MATH` 不启用（§7.3，与 `A3`/`A1` 的逐位目标一致）、跨 block barrier 未认证（`workflow.md:29` ✓ 支撑 R1-e 否决）、索引宽度 `int64`（e15 设备码 `((int64_t)it) >> 1) * (int64_t)24576` ✓，最大 workload `dq_acc` 1.61 GB / `dq` 805 MB 复算一致）。
- 动态维 + `out_idx` 的前置条件（`T.dynamic` 符号复用）有 e14 反例 / e15 正例，属真实生效的约束，已写入 §3.3 与 §9.2 E2。

### 5. 循环与同步 — warn

- 循环结构完整（§6.1/§6.2）：L0 核网格 `T.Kernel(56)`、L1 `T.Pipelined(it0,it1,num_stages=2)`（动态界）、L2 `T.serial(R)`、L3 `T.serial(NSEG)`，并说明「不用 `T.Parallel`（SIMT 域混用未验证）」「不用 `T.Persistent`（语义是二维 tile swizzle，与一维 tile 流不符）」——理由具体，不是泛泛而谈。
- 同步与模式匹配：§7.1 声明全自动（AutoSchedule + InsertSync + MaterializeMultiBuffer），与 `programming.md` 的「未验证手工跨核 flag 配对」一致。我按 e15 设备码逐条核对了 flag 配对：循环前预置 `MTE3→V` / `V→MTE2` 的 id 0/1 → 每轮 `wait(V→MTE2, ver)`（缓冲再写保护）→ `notify(MTE2→V)` → `wait(MTE3→V)` + `wait(MTE2→V)` → VF → `notify(V→MTE3)`/`notify(V→MTE2)` → `wait(V→MTE3)` → store → `notify(MTE3→V)`，循环尾再 `wait` 四个 id。**无死锁**（每轮内的 wait 都有同轮或预置的 notify 对应），`acc`/`out` 两版本的读写危险均被覆盖；核间无数据依赖（tile 归属互斥）。
- 尾块处理：`n = min(R, T_q - t0)` 同时出现在 load 与 store 的 extent 上（e15 设备码两条 copy 均带 `min(64, (total_q - ((it >> 1) * 64)))`），有效行集恒等，越界行不读不写。

warn 原因（**建议 2** 与 **建议 3**，均不影响正确性，但会误导 Stage 3/4 的预期）：

1. **建议 2**：§6.3 与 §1.4 步骤 5 称「`load(k+1)` 与 `compute(k)/store(k)` 重叠」，但 e15 设备码里 load 在**同一轮**内先 issue 再被 `wait(MTE2→V)` 阻塞，之前只有 `wait(V→MTE2)`，**没有跨轮 load-ahead**；因此可证成的重叠只有 `MTE3(it-1) ∥ MTE2(it)`（不同版本、不同 pipe），V 与 MTE2 在同轮内是串行的。建议把 §6.3 的结论改写为「双缓冲已成立、MTE3/MTE2 重叠已由设备码证实、V∥MTE2 是否重叠留待 Stage 4 实测」，并把该差别并入 §9.1 R2（它现在只讲 flag 错序风险）。
2. **建议 3**：§6.4 写「store 侧带 `min(min(64, total_q - …), total_q_1 - …)` 与 `if (0 < total_q - …)` 双重守卫（e15）」——该形态实际出现在 `probe_e10_gen.txt` / `probe_e11_gen.txt`（两份都有 `total_q` 与 `total_q_1` 两个符号，源自内联 `T.dynamic` 反例写法）；`probe_e15_gen.txt` 只有单层 `min(64, total_q - …)`、**没有** `if` 守卫。正确性不受影响（`t0 < T_q` ⇒ `n ≥ 1`，无需零长度守卫），但证据归属要改。

（§6.3 关于「`num_stages=1` 与 `2` 的设备码逐行相同、故串行基线必须删版本声明」的结论我实际 `diff` 过 `probe_e12_gen.txt` 与 `probe_e13_gen.txt`：**完全相同**，该结论成立。）

### 6. 验证方案 — pass

- Golden（§8.1）以 §0.1 语义写成纯 torch 式，且给出独立性论证；`NaN→NaN`、`|y| ≥ 65520 → ±inf` 与 torch `.to()` 一致。
- L0 计划（§8.3）覆盖：主路径（L0-1，且明确「`out_idx` 动态输出分配最先跑，风险最高」）、bf16 路径（L0-2）、尾块 clamp（L0-3 `T_q=1000`，`1000 % 64 = 40` 复算正确；L0-5 `T_q=65`）、退化几何（L0-4 `T_q=1`）、角点注入（L0-6，含 ±0/±inf/NaN/次正规/65504/65519/65520/1e5，正好对应 A3/C2 的判据）、满载基线（L0-7 同时作 Stage 4 baseline）。
- 精度标准（§8.2）与源仓测试实查一致：`examples/TileOPs/tests/ops/test_mla_dense_bwd.py` 的 `_RTOL = {fp16: 1e-3, bf16: 1.6e-2}`、`_ATOL_FLOOR = 5e-3`、`atol = max(5e-3, rtol)` → fp16 (5e-3, 1e-3)、bf16 (1.6e-2, 1.6e-2)，与 §8.2 表一致。
- 附加判据合理：本算子是「一次乘 + 一次窄化」，在 `vcvt(ROUND_R, sat=False)` 与 golden 同为 IEEE RNE 的前提下应逐位相等，故声明 `torch.equal` 强断言并在不等时记录最大 ulp（正是 §9.1 R1 的证据采集手段）。P1–P6 定点检查覆盖越界写、尾块行数、单次舍入、非饱和、流水重复性、退化几何，可执行。
- 设计自述「动态界 / 2 版本流水 / `out_idx` 动态输出分配」三处只有编译级证据——**确认可在 L0 覆盖，不阻塞**：`out_idx` 由每个 L0 用例隐式行使（L0-1 首跑）、动态界与尾块由 L0-3/L0-5 行使、流水正确性由 P5（同一几何连跑 ≥ 20 次 + 「去掉版本声明」的单版本对照，结果逐位一致）；P5 选「删版本声明」而非「`num_stages=1`」也与 §6.3 的实测结论一致，是正确杠杆。

### 7. 完整性与一致性 — warn

- 章节齐全：§0（0.1–0.7 全）/ §1（1.1–1.6，含 1.6.0–1.6.3）/ §2 / §3（3.1–3.5，含 3.5.4 探针表）/ §4 / §5 / §6 / §7 / §8 / §9 / §10 全部存在。
- 无占位符：对 DESIGN.md 做了占位记号检索（占位符花括号变体 / 未完成标记 / 中文「待补」类措辞），全部零命中。
- 同类引用真实：§0.7 / §3.5.3 / §1.6.0 / §1.6.3 引用的 9 个路径全部存在，且内容对得上（`examples/TileOPs/tileops/kernels/attention/mla_dense_bwd/_mla_dense_bwd_kernels.py` 的 `_dq_land_kernel` 自第 428 行起、GPU 原样抽取；`examples/mla_dense_bwd/_delta_kernel/_delta_kernel.py`；`examples/paged_cache_write/...`；`kernel_template.py`；`PL-950-row-primitives.md`；`numerical-boundaries.md`；`examples/mla_dense_bwd/_bwd_kernel/DESIGN.md`；`constants.md`；`workflow.md`）。`history_version/` 下 e9–e15 / m2 / n5 设备码快照全部落盘，`probe_e15_gen.txt` 确为 41 行（与 §3.5.4 的「设备码 41 行」一致）。
- 内部一致：API 映射 / 内存规划 / Tiling / 循环四处的 56 核、`R=64`、`DW=192`、2 版本、clamp 口径互相一致，无矛盾。

warn 原因（**建议 4**，证据精度，不改变结论）：
1. §3.4 表内 `vcvt` 行把「数值语义待实测」交叉引用到「§9.1 风险 R2」，应为 **R1**。
2. §3.5.4 的探针表把 `e10` 称作「结构备选」，但 e10 用的是内联 `T.dynamic(...)` 反例写法（设备码出现 `total_q` 与 `total_q_1` 两个符号），**不能**直接配 `out_idx=[1]`（会触发 §9.2 E2）；作为「结构对照」可以，作为「备选实现」需先按 e15 方式绑定符号，建议加一句限定。
3. （提示，非缺陷）§4.5 的 ×1.12 膨胀行在本型号属跨型号假设（平台 `ub_inflation = null`），当前标注方式已是正确口径，Stage 4 若有实测再回填即可。

### 8. 算法优化分析 — pass

- **§1.6 章节存在性 pass**：1.6.0 调研 / 1.6.1 等价优化 / 1.6.2 向量化替代 / 1.6.3 向量化轴与布局决策四节齐全，且本算子被正确判定为「单步逐元素」轻量调研（四问仍逐问给结论）。
- **算法调研存在性与完整性 pass**：R1 候选表**含基线 B0**（源算法）且列出 6 个候选（单次舍入 / 16 位域乘 / 饱和 / 截断 / 融合进上游 / 省 fp32 落地）；R2 明确结论「本身即在线单遍，无多遍可省」并给结构依据；R3 覆盖**四口径**（FLOPs `E` 乘 + `E` 窄化、访存 `6E`、扫描 1 遍、中间缓冲 0 个 GM 缓冲 / 147456 B UB）且基线在表；R4 逐候选评估（SIMD 段链 / SIMT 投影 / DMA cast / Cube-MixCV / host torch），六维（单元匹配、容量、对齐、静态边界、流水可融合、判定）；结论给出选定算法与三条关键依据；「无更优替代」写明调研范围（R1 六候选 + R2 结构判据 + R3 四口径 + R4 五候选 + `algorithm_candidates` + `examples/` 七项）。
- **调研结论正确性（独立复核）pass**：
  - ① 等价性：选定算法与原语义的等价性在 1.6.1 被逐项证明（见下），无「容差内等价」隐患（本算子采纳项是**精确等价**，不是容差近似，故无 fp16/bf16 舍入近似风险）。
  - ② 复杂度**独立复算全部一致**：`E = 4096·128·192 = 100,663,296`；`6E = 603,979,776 B = 604.0 MB`；`604/1.26e12 = 479 µs`、`604/1.03e12 = 586 µs`；发射项 `E/64·4 = 6.29 M` 条、`/56 = 112.3 K`、`@1.4 GHz ≈ 80 µs`；smoke `786,432` 元素 → `4.72 MB`；`ds-v2-lite-3k = 37,748,736·6 = 226.5 MB`；`ds-v3-b2-8k = 402,653,184·6 = 2415.9 MB → 1917–2346 µs`；`R1-e` 省 `8E = 805.3 MB ≈ 640 µs`。**中间缓冲口径未漏计 GM 往返**（本设计确实无 GM 中间缓冲，UB 峰值 147456 B 已计入）。
  - ③ 亲和性淘汰有佐证：`CG-950-batch-barrier`（`knowledge/capability-gaps.md` + `workflow.md:29`）支撑 R1-e 否决；`vf_checker.py` 支撑 DMA cast 淘汰；`_vreg_lanes` 支撑 64-lane / A4 淘汰；`CONST-*` 均为 910B2C，设计已逐处标注「跨型号假设」，**未把 910B2C 常数当作本型号判定依据**。
  - ④ **负向断言逐条复核**：R2「无多遍可省」——结构判据成立（纯映射、无 running 统计量可物化，任何分遍都只重复搬同一批字节）；R1-e「融合不可行」——与源算法事实一致（`_bwd_kernel` L254 `T.atomic_add` 跨 CTA 累加，最后一趟部分和到达前无法定稿）+ 平台 barrier 未认证，两条独立支撑，不矛盾；`knowledge/algorithm-candidates.md` 的 elementwise 行（`PL-950-row-primitives` 的 SIMT 逐元素路径）**已在 R4 作为「SIMT 投影」评估**并保留为备选 A3，未出现「明显漏候选」。唯一措辞瑕疵：§1.6.0 称「本平台 `algorithm-candidates.md`，纯 elementwise 无专门条目」，实际该文件有 `elementwise / broadcast` 行（指向 `PL-950-row-primitives`）；候选实质已评估，仅措辞不准确，另 `PL-950-vector-execution.md`（950 DT 仿真上 SIMD + 尾块的 elementwise abs 已验条目）未被引用，建议补引以加强「同族先例」证据。
  - ⑤ 并列候选裁决路径：A1（SIMD 段链）与 A3（SIMT 投影）裕度落在未实测常数上，设计给了**实验裁决三件套**（主选 A1 + 结构完整的备选 A3（同 tile/UB/分核，仅换 VF 体，探针 n5 已编译生成 `float2` + `__float22half2_rn`）+ 裁决计划：三个代表 shape、`latency(A3) < latency(A1)×0.95` 阈值、`mte2_ratio`/`*_active_bw` 指标、回写路径、按 `heads`/`dtype` 分派可行性），符合要求。
- **调研→设计一致性 pass**：§1.4 的「单遍流式向量映射 + 56 核持久 + VF 段链」= §1.6.0 选定算法；§3.1 明确以 §1.6.1 优化后公式（A1+A2+A3+A4+A5）为输入；§6 的循环结构与该算法同构（persistent + 2 版本 + 静态行/段循环）；§0.5/§0.6 的处置与调研选族一致。**源优化手段的意图承接无静默丢弃**：§0.4 的 9 项在 §0.5 逐项有四态处置，并在 §0.6 的 RD1/RD2/RD3 与「承接的优化意图」里逐条点名（#1→RD1、#2→RD1、#3→RD2、#4→RD2、#5→RD1、#6/#7→§7.3 舍弃、#8→§3.5.4 补绑定、#9→RD3 补强）。
- **数学等价优化完整性 pass**：A1–A5 均有「原式 → 优化后公式 → 等价性论证 → 收益量化」四要素，收益栏有数字（A2 `E/64 ≈ 1.57 M` 条指令、A4 `4R→3R` = −25% 与 `out` 缓冲 −25%、A5 尾 tile 搬运 `R→n`、A1/A3 为精度保真并明确「非性能项」），无空收益栏；「无更多可省计算」给了正面依据（只有一个乘，无除法/rsqrt/稳定化/CSE/归约合并空间）。
- **等价性论证正确性（独立推演）pass**：
  - A1：`cast(fp32(x)·sf32)`（一次舍入）与 `cast(cast(x)·cast(s))`（两次舍入）数学上不等价——推演正确；机器验证给出 3857 ulp / 违反率 1.216e-3 的量化反例（我重跑一致），否决 16 位域形态成立；且输入本来就是 fp32，16 位域无收益。
  - A2：`vmuls(v,sf32)` 与 `vmul(v, vdup(sf32))` 乘的是同一个 fp32 常量——推演正确；e15 设备码里字面量确实按 fp32 落地（`float(0x1.6a09e667f3bcdp-4f)`，与 `SCALE` 的 fp32 值一致），而 §8.1 golden 的 `float(scale)` 在 fp32 张量上是同一个 fp32 标量 → 逐位相同。成立。
  - A3：**模型层**成立（非饱和窄化 = IEEE，65520 阈值正确）；**设备层**未证（见维度 1 的 warn），已由 §9.1 R1 + L0-6/P4 + A3 备选覆盖，论证边界被正确交代，不构成等价性推演错误。
  - A4：`192 = 3×64`，`vsts` 满 lane，pad 形态的无效 lane 不进输出 → 输出逐位相同，仅指令数差；成立。
  - A5：有效行集 `[0,n)` 与 `min(R, T_q-t0)` 恒等，越界行 golden 里不存在 → 无浮点差异；成立。
  - 另核 `[baseline]` 行：golden 与 fp64 前置像 0 ulp，说明「fp32 乘 + 一次窄化」在这批样本上无双重舍入，基线自洽（重跑一致）。
- **向量化替代覆盖完整性 pass**：§1.6.2 的 V1–V10 与 §3.3 伪代码、§6.1 循环表逐点交叉核对，**无遗漏**（L0 网格 / L1 `T.Pipelined` / L2 `T.serial(R)` / L3 `T.serial(NSEG)` / tile 解码 / 区间边界 / 尾块 n / 谓词 / 广播轴），热路径中不存在未论证的逐元素标量循环；V1/V2/V3 的替代均有 `simd.py` 导出 + e15 设备码双重佐证（`vlds_norm`/`vmuls`/`vcvt<half>(…ROUND_R, RS_DISABLE, PART_EVEN…)`/`vsts_norm`），V9 由 e15 的「VF 体内只有两层 for、无 if」佐证。
- **不可替代理由充分性 pass**：V4/V5（tile 级结构索引）给出「摊平不减少指令数、只等价替换索引」的具体理由；V5 还有负向证据（e11）；V6/V7（block 级标量，每 tile 一次）/V8（动态边界，工厂拿不到 `T_q`）分类正确、理由具体到本算子，符合「不得用『实现简单』」的要求。
- **替代 API 佐证 pass**：见维度 1，均为当前安装包导出，无跨后端同名 API 借用。
- **向量化轴与布局决策完整性 pass**：§1.6.3 对「逐元素类」按「连续轴选择 × tile 覆盖形状」枚举（无广播轴已说明），候选 A1/A2（多 head tile）/A3（SIMT 投影）/A4（128 lane）/A5（转置重排）齐全，每条有量化评分（段宽、每行向量指令数、UB 占用）；**弃选有量化理由且前提经实查**：A2 `HH=2,R=64 → 294912 B > 253952 B`（我复算：`2·64·384·4 + 2·64·384·2 = 196608 + 98304 = 294912` ✓）；A4「无 128 lane fp32 寄存器」经 `_vreg_lanes = 2048//32 = 64` 实查 ✓；A5 repack `+6E` 流量 ✓。I/O layout 被正确当作契约、核内布局当成设计变量，未出现「I/O 是 NCHW 所以内部也 NCHW」式论证。
- **布局决策与实现一致 pass**：§1.6.3 选定的 A1（单 head × 64 行、行内 3 个满段）与 §3.3 伪代码、§4.3 的 buffer 形状（`(64,192)` fp32 + `(64,192)` dtype，2 版本）、§4.5 的 UB 预算、§6.1 的双层 `T.serial` 三方一致；无重排路径需要落地。
- **与下游章节一致 pass**：§3.1 以优化后公式为输入（A1/A2/A3/A4/A5 全部出现在伪代码里）；§6 循环结构与 §1.6.2 结论一致（被判定「已向量化」的点 V1/V2/V3 在 §6 中不再出现逐元素标量循环）。

## 补充：对 conductor 三个重点问项的结论

1. **静默丢弃缺陷（设计 R8 / §9.2 E1）的独立复核：证据成立，设计最终形态已规避，且该缺陷应升级为 Stage 3 硬前置（实现级，不是设计阻塞）。**
   - 证据核对：`probe_e11_gen.txt` 的 `main_kernel_simd_vf_0` 体内只剩 `asc_create_mask_b16(PAT_VL64)` 与每行的 `asc_create_mask_b32/b16(PAT_ALL)`，**无 `vlds_norm`/`vmuls`/`vcvt`/`vsts_norm`**；`probe_e15_gen.txt` 同函数体内双层 `for(i<64) for(s<3)` + 完整 `vlds_norm → vmuls → vcvt<half>(ROUND_R, RS_DISABLE, PART_EVEN) → vsts_norm`。
   - 触发条件隔离成立：`design_probe_dynamic.py` 的 e10（动态 shape、段循环在体内）与 e11（**唯一差别**是 `_row_body(acc, out, i, …)` 这个**模块级** python 辅助函数用 `for s in range(3)` 展开）几何/副本/符号完全相同；`probe_e10_gen.txt` 计算体完整，e11 被丢弃 → 差异可归因于 python 层展开（与设计描述一致；`_row_body` 确在模块级）。补充一点精度说明：不是「整个 VF 块消失」，而是「`T.simd.*` 计算链被丢弃、mask 创建留下」，本质（编译通过但算错）与 §9.2 E1 的判断一致。
   - 规避是否到位：§3.3 的最终形态把段循环写成 `T.serial(NSEG)` 且所有 `T.simd.*` 语句词法位于 prim_func 体内，与 e15（COMPILE OK 且计算体完整）逐句对应 → **已规避**。
   - 是否写成 Stage 3 硬前置：**建议是（建议 5）**。成本极低（`kernel.get_kernel_source()` 里 grep `vlds_norm|vmuls|vcvt|vsts_norm`），能一次拦掉整类「编译 OK 但算错」。建议把这条加进 §8.3 的 L0 通过判据（例如 P0：编译后设备码必须含上述四类指令，否则直接判失败）与 §10.4 实现顺序第 2 步。另建议按根 `AGENTS.md`「Compiler capability gaps」把该缺陷登记到 `knowledge.capability-gaps.md`（当前该文件只有 `CG-950-batch-barrier` 与 `CG-950-simd-shared-reduce`，没有这条 SIMD 体丢弃条目）。
2. **动态界 + 2 版本流水 + `out_idx` 动态输出分配：可在 L0 覆盖，不阻塞。** `out_idx` 由任一 L0 用例行使（且 L0-1 被安排最先跑）；动态界与尾块由 L0-3/L0-5 行使（`n=40`、`n=1`）；2 版本流水由 P5（连跑 ≥ 20 次 + 单版本对照逐位一致）行使，对照杠杆（删版本声明而非 `num_stages=1`）与 §6.3 的实测结论一致，选对了。附带确认 §5.5 的 persitent 区间在 `ntiles < 56` 时会退化为空区间（L0-4/L0-5 正好覆盖），不需要额外分支。
3. **编程模式与预算：`simd` 与平台 `programming_modes.migration_default = simd` 一致，模式-API 匹配、禁用项写清（`threads=` 不存在，`T.Parallel` 不在 SIMD 域混用）；UB 144 KB = 58.06%（我按 253952 B 复算），56 个 AIV 与 `observed_cores.vector = 56` 实查一致。**
