# _engram_gate_conv_bwd_kernel 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息
- 文档: examples/engram_gate_conv_bwd/_engram_gate_conv_bwd_kernel/DESIGN.md（1284 行）
- 任务类型: 迁移（TileOPs harness 子模式，EngramGateConvBwdOp）
- 源算子: /workspace/TileOPs/src/tileops/_kernels/sequence_modeling/engram/gate_conv_bwd.py（448 行，已全文读取）
- 参照: examples/TileOPs/tileops/manifest/sequence_modeling.yaml（Bwd 4 workloads，`shape_rules: d>0 / d<=4096`）、
  examples/TileOPs/tests/ops/test_engram.py（`_tol_bwd`、Bwd 用例）、
  examples/TileOPs/tileops/workloads/sequence_modeling.py（`ref_engram_gate_conv_bwd`）、
  /workspace/TileOPs-ascend950/src/tileops/kernels/ascend/sequence_modeling/gate_conv_bwd.py（950DT 只读参考）
- 平台: ascend-950pr-9579 / ascend（CANN 9.2.1，dav-3510，28 Cube / 56 Vector，UB 253,952 B / L1 512 KiB / L0C 256 KiB；
  `statectl platform --task-dir examples/engram_gate_conv_bwd` 实读）
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 13（分布：维度 0 × 1、维度 2 × 1、维度 3 × 2、维度 6 × 2、维度 7 × 3、维度 8 × 4；全部为文档内可定点修正的表述/算术/伪代码守卫缺口，不改算法、不改语义、不改内存规划）

## 机械复核（D-1/D-5）

- **verify_equiv 重跑**: **PASS（与 §1.6.1 内嵌表逐格一致）**
  `cd examples/engram_gate_conv_bwd/_engram_gate_conv_bwd_kernel && ASCEND_RT_VISIBLE_DEVICES= /workspace/venv-ascend950/bin/python verify_equiv.py`
  → `EQUIV_SUMMARY ALL_PASS`；BASE/O1/O2/O3/O4/O5/ALL × float16/bfloat16 全部 `EQUIV_PASS`，每行 252 字段，
  尺度相对误差 0 / 2.633e-4 / 1.663e-4 / 0 / 4.496e-4 / 0 / 4.496e-4（fp16）与
  0 / 2.083e-3 / 2.083e-3 / 0 / 7.911e-4 / 0 / 2.083e-3（bf16）——与 §1.6.1 表**逐个数字相同**；
  per-field（ALL）dH/dk/dv/drms_w_h/drms_w_v/dconv_w = 4.50e-4/1.98e-4/2.90e-5/1.61e-6/0/0（fp16）、
  1.32e-3/2.08e-3/5.79e-5/2.44e-6/0/0（bf16）；基线 vs fp64 真值 = **8.883e-3（fp16）/ 5.393e-2（bf16）**，
  与 §1.6.1「8.88e-3 / 5.39e-2」及 §8.1「8.9e-3 / 5.4e-2」一致。
- **design_calc_check**（`python3 .agents/tools/design_calc_check.py --design … --task-dir examples/engram_gate_conv_bwd`）:
  - `r3_metrics`: **pass** — 「all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak)」
  - `ub_budget`: **skip** —「no per-level buffer tables parsed from the memory plan（advisory checker; verify manually）」
    → **人工已核**：按 §4.5 表逐行复算 7,680+3,072+6,144+36,864+6,144+12,288+12,288+6,144+12,288+1,536 = **104,448 B = 102.0 KiB**，
    与设计值逐字节相同；`134 B/channel` 复核 = 10+4+8+48+8+16+16+8+16 = 134 ✓；d=1024 → 137,216+8,192 = 145,408 B = 142.0 KiB ✓ < 253,952 B。
  - `l0c_budget`: **skip** — 无 Cube GEMM → **人工已核** n/a（§2 无 matmul，§4 不用 L0A/B/C）。
  - `core_split`: **skip** —「insufficient parsed dims {'M': 64}」（解析器把 §5.5 的示例 M 误取）
    → **人工已核**：独立复算 8 条几何 `nch/RC/nch/grid` = 32/44/43/48/54/54/8/2，`M·nch ≤ 56` 全部成立（见维度 3）。

## 检视详情

### 0. 源算子理解与迁移分析 — warn

**已亲自读源码逐语句核对**（`gate_conv_bwd.py:35-448`），结论：语义解读**正确且完整**，仅 launch 计数标注有误。

- **语义核对（§0.1 ↔ 源码）**: 12 条梯度公式逐条对上——`conv` 重算含 `if_then_else(src_t>=0)` 零填充（源码 90-106）↔ §0.1 line 31；
  `sig/d_silu = dy*(sig+conv*sig*(1-sig))`（110-112）↔ line 38；卷积转置 `dst_t=tid+3-p`（164-174）↔ line 40；
  `drms_w_v += d_vnorm*vhat*rrms_v`（182-186）↔ line 41；`dot_v`+`dvhat=rrms*w*dn-rrms³*vhat*dot/d+dY`（190-201）↔ line 42-43；
  `dalpha/dv/ddot=dalpha*alpha*(1-alpha)/sqrt(d)`（239-246）↔ line 44-45；`h_n/k_n`（248-250）↔ line 46；
  `dot_h/dH/drms_w_h(+h 项)`（254-270）↔ line 47-48/51；`dot_k/dk/drms_w_h(+k 项)`（273-289）↔ line 49-51。
  dtype：`accum_dtype="float"`、`dH=cast(dh_local,dtype)`（58/292-294）↔ §0.1「全 fp32 中间 + 一次 cast」✓。
  **保存态舍入**：源码 97/136 `T.cast(vhat[...])` 用的是保存的 `vhat`（非重算 `alpha·v`）↔ §0.1 line 59 与 manifest 注释
  「the forward's *rounded* vhat」✓。**eps**：工厂 `_engram_gate_conv_bwd_kernel(M,seq_len,d,dtype,run_len)` 无 eps（line 35）↔ §0.1 line 62 ✓。
  **边界**：`t-3+p<0` 零填充、batch 内独立 ✓（源码 95-104）。**host 侧**：`F.pad`→`align_up(d,256)` + 输出裁剪（411-448）↔ §0.3 line 137 ✓。
- **算法步骤覆盖（§0.3）**: 步骤表 0–13 覆盖源码全部计算语句，含易漏的三处——
  `token_runs>1` 时才清零 `dconv_w`（74-76）、`token_runs==1` 直达写（146-147）、stage-1a 的 `token_live/src_live` 双 guard（127-141），无遗漏、无臆造 ✓。
- **优化手段识别（§0.4）**: 12 项均真实存在（逐项对源码/`_run_len` 注释核对），耦合性初判与抽检一致：
  #2 原子、#5 `sm_count`（`4*sm_count//KS` = `sm_count` 上限，line 388-396）、#6 每 token 一 CTA 判「强」✓；#7/#9/#10 判「可移植」✓。
- **耦合性判定（§0.5）**: 四态处置齐全且依据可查。抽检两条最关键的负向依据，**均亲自打开平台知识库核对，与原文一致**：
  `PL-950-atomic-splitk`（knowledge/execution-patterns.md:48-68）确为「两个 **Cube** 核在 **L0C** 生成部分和 + `set_atomic("add","float32")` 写回 GM」，
  不含 UB→GM 原子 → §0.5 判 `T.atomic_add` 路径「未核验且顺序不定」成立；`CG-950-batch-barrier`（capability-gaps.md:3-5）
  确为未认证跨 block barrier → 「拆 2 次 launch」的依据成立。
- **重设计可行性（§0.6）**: R1–R6 每项都有「源方案 → NPU 新算法 → 语义保持论证」。独立复算其中两条最硬的：
  ① **R2 依赖距离**：窗口位置在迭代 `i'=pn-t0+3` 写入；step (b) 于 `i` 读 `pn ∈ [t0+i-6, t0+i-3]` = 迭代 `i-3..i` 的 4 次写；
  step (c) 于 `i` 读 `ds[tp..tp+3]`（`tp=t0+i-6`）同样是 `i-3..i`（含同迭代 (b) 刚写的 `tp+3`）→ **深度 4 恰为最小充分**，
  最早覆盖发生在 `i+4`（已过末次消费）✓；最前块 `pn=-3,-2,-1` 的零槽用 `floormod` 落到 slot 1/2/3 并在 `i=3` 被读到、`i=4..6` 才覆盖 ✓
  （关键：读写两侧都写 `T.floormod`，若误用 truncmod 则 `-3→-3` 槽错位；设计统一用 floormod，探针 202/245/258 实测同形）。
  ② **R6**：`/d` 用真实 `d`（源码 197/264/283 `float(d)`）↔ §0.6 R6「掩码 lane 零乘零不影响和」可推 ✓。
  `d_silu/vhat_norm/dY 不落 GM` 不改变语义成立：同一表达式同一 fp32 精度、每行只算一次；跨核 halo 多算 3 行只写本核 UB，
  `dconv_w` 由 `if i < RC+3` owner guard 去重（`pds=t0+i-3` 属本核 ⇔ `3 ≤ i < RC+3`，与 `i>=3` 外层合取后精确）✓。
- **§1–§7 与迁移决策一致**: §1.4/§3.3/§4/§6 实现的是 C2 + R1–R6（2 launch、单遍、per-core 部分和、设备掩码），
  未见「§0 判重设计、§3 照抄源方案」的脱节 ✓。
- **golden 独立性**: §8.1 与 `workloads/sequence_modeling.py:260-306` 的 `ref_engram_gate_conv_bwd` **逐行一致**
  （fp32 autograd 对 6 个可微输入求导、不使用 `vhat/alpha/rrms_*`、`conv1d(groups=d)` + 左补 `KS-1`、`set_multithreading_enabled(False)`），
  与 §0.6 的窗口/部分和/有序归约无关 ✓。

**⚠ 建议（不阻塞）：launch 计数标错**。源码有 **5** 个 `T.Kernel`（`70` 清零 / `78` pass1 / `119` pass1a / `151` pass1b / `206` pass2），
文档多处写成 4 次：header line 4「4 次 launch：清零 + 三段计算」、§0.4 #11「4 段 `T.Kernel`」、§0.6 R2「3 个计算 launch + 1 个清零」、
§1.4 line 345「源算子为 4 次」、§1.6.0 R3「C0（源算子：4 launch）」、§3.5.2「4 个 `T.Kernel` 段（清零 + 3 段）」。
改法：统一改为「5 次 = 清零 + 4 段计算（pass1 / 1a / 1b / pass2）」。该错误不改变任何结论（launch 5→2 比 4→2 收益更强），
但 §0.3 的步骤表已按 launch#2..#5 编号，与「4 次」自相矛盾，需一并订正。

### 1. API 可行性 — pass

- **API 存在性（逐名实查导出表）**: `/workspace/venv-ascend950/lib/python3.12/site-packages/tilelang/ascend/language/simd.py`
  实测导出 `vld`(324) / `vsts`(457) / `vcvt` / `vadd`(554) / `vsub`(623) / `vmul`(629) / `vmuls` / `vadds` / `vdiv`(792) /
  `vexp`(860) / `vneg` / `vcadd` / `vdupv` / `vdup` / `pset`(111) / `update_mask`(125) / `vmadd`(649) —— §3.2 用到的全部存在 ✓。
  `vexp2` 在该模块 **0 次命中** → §1.6.1 候选 9「API 缺失」的弃选依据成立（附注：导出名实为 `vexpdif`(1182)，文档两处写作 `vexpdiff`，属笔误）。
- **API 与模式匹配**: 纯 SIMD（`T.SimdVF` + 一维 `T.Kernel(grid)`），与 §2 选定模式一致；`T.Kernel` 签名无 `threads`
  见 `references/programming.md:13`（「一维核网格；签名没有 `threads`，多维网格在实现中拒绝」）✓，§0.5 对 `threads` autotune 的舍弃有依据。
- **刻意不用项的依据可查**: `T.reduce_sum` → `TRAP-950-simd-shared-reduce`（numerical-boundaries.md:81-100）原文即
  「FP32 (2,64) 二维 shared→shared 的 sum 第二行与 CPU 参考不一致」，替代路径正是 `PL-950-simd-reduce` 的显式寄存器归约（vadd 链+vcadd）✓；
  `T.atomic_add` → 见维度 0 的 atomic-splitk 核对 ✓。`T.reduce_sum`/`T.atomic_add` 的「刻意不用」均有平台证据，非主观规避。
- **编译证据**: `history_version/design_probe_compile.py`（580 行）7 几何（train-s96/s64/s128/prime-401/s512/short-1/pad-d192）
  × 2 kernel（fused + reduce）= **14 次编译**，与 §10.2「7 几何 × 2 kernel / ALL_OK」一致；探针内 `T.floormod`、
  `T.annotate_buffer_versions`、`update_mask`、动态 `off:off+cols`、`vld(part[row,col])` 链均实际使用。
- 未发现「凭记忆猜 API」：§3.4 的 11 条来源均指向本平台模块/文档/探针，与 ruleset（ascend）一致。

### 2. 内存层级规划 — pass

- **搬运路径完整**: §4.4 给出 GM→UB（`T.copy` 行/表）→ UB fp32（`vld(UNPK_B16)+vcvt`）→ 计算 → `vcvt(PK_B32)+vsts` → UB fp16 → GM 的完整链路，
  以及 `pc/ph/pv → PART → launch#2 → 输出` 的第二段；明确「无 L1/L0 路径、无 GM→L1→UB 中转」（一维网格下成立）✓。
- **UB 预算（人工复算，见机械复核）**: 104,448 B / 253,952 B = 41.1%，逐 buffer 与 §4.5 表**逐字节吻合**；
  d=1024 峰值 145,408 B = 142.0 KiB（余量 42.7%）< 容量 ✓；`sc` 条带 16·seq（seq=512 → 8 KiB）已计入 ✓。
- **Buffer 层级标注**: §4.3 全部标注 UB（含 dtype/形状/用途），PART 标注 GM workspace ✓；`part/tot` 归约 kernel 独立 launch，
  29,184 B 不与计算 kernel 叠加（理由正确）✓。
- **L0C/L0A/L0B**: 无 Cube 计算 → 不适用 ✓（`l0c_budget` 机械 skip 同样源于此）。
- 单版本缓冲理由（「滚动窗口语义要求同一槽 4 步内不被覆写」）与 `annotate_buffer_versions({…:1})` 的语义一致，容量充裕（41%）不需要多缓冲换流水 ✓。
- **⚠ 建议（不阻塞）**: §4.5 表「d=1024 … 余量 34%」实为 **seq=2048 列**的值（166/248 → 33.1%）；seq=512 列余量为 42.7%。建议把「余量」一列绑定到具体 seq 列，避免 Stage 3/4 误判容量上限。

### 3. Tiling 策略 — warn

- **Block 划分明确**: §5.2 给出 `dw=ceil(d/64)*64`、`nchunks=dw/64`、`nch=min(seq_len,max(1,56//M))`、`RC=ceil(seq_len/nch)`、
  `nch=ceil(seq_len/RC)`、`grid=M·nch`，取整与回填顺序正确（独立复算 8 条几何全部吻合：32/44/43/48/54/54/8/2，见表）✓。
  RC 取舍理由（并行度 vs halo 3/RC）量化 ✓。
- **对齐约束**: `dw` 64 lane + `update_mask(min(64,d-64·ch))` + 表尾清零 + 输出只写 `[:d]`；`d=192`（非 128 倍数）探针覆盖；
  `d` 非 64 倍数交 L0 `(1,16,100)` ✓（行首 32 B 对齐风险已显式登记，未夸大）✓。
- **非整除处理**: RC ∤ seq 用均匀标量 guard；`tokens<56` 空核；`d%64≠0` 用动态 extent —— 策略齐全（见下方 warn/建议的守卫缺口）。
- **分核策略三要素（`core-split-strategy.md` §1/§2.2 核对）**:
  ① 逻辑核数：给出 `M·nch ≤ 56` 与 `6·tiles ≤ 54` 两式并逐 shape 列值，由构造保证不超发 ✓；
  ② 物理核数依据：引用平台登记实查记录 `hardware.observed_cores = {cube:28, vector:56}`、
  `core_source = "Ascend950PR_9579.ini [SoCInfo] … + runtime probe rtGetAiCoreCount()=28, 2026-11-23"`（我已用
  `statectl platform --task-dir examples/engram_gate_conv_bwd` 复读，字段与值一致），并给出设备侧查询代码、
  如实声明本次未重跑（卡被其他算子 Stage 3 占用）→ 属「同型号平台实查记录」，非文档假设/经验值 ✓；
  ③ 规模判定：逻辑 ≤ 物理 → 无需适配，并论证 `M>56` 的分波安全（无跨块同步）✓。对脚手架「`sm_count→cube_core_num`(28)」的
  反驳有量化（train-s96：27 核/RC=11 vs 48 核/RC=6，halo 50% vs 27%）✓，取 `vector_core_num = 56` 合理。
- **⚠ 建议（不阻塞）：§5.4 的 `seq_len < 4` 一行写错，且 §8.3 #3 据此写了错误的断言**。原文「halo 全部越界 → 所有 tap 读零槽，
  `conv = 0`、`d_silu = dY·0.5`、`d_vnorm = 0`」与其括号「只有 `p=3` 的 tap 在 `t=0` 有效」自相矛盾，后者才对：
  源码 92-93 `src_t = tid-3+p`，`t=0` 时 **p=3 → src_t=0 有效**（读的是本行 `z[0]`），故 `conv[0] = conv_w[3]·z[0] ≠ 0`；
  源码 167 `dst_t = tid+3-p`，`t=0` 时仅 `p=3 → dst_t=0` → `d_vnorm[0] = conv_w[3]·d_silu[0] ≠ 0`。
  因此 §8.3 #3 的「`d_vnorm` 应为 0 → `dvhat = dY`」若被 Stage 3 写成断言会**误判**（`dvhat = rrms_v·w_v·g − rrms_v³·vhat·dot_v/d + dY ≠ dY`）。
  改法：§5.4 该行改为「3 个 tap 越界读零槽，仅 `p=3` 有效 → `conv = conv_w[3]·z[0]`；`d_vnorm[0] = conv_w[3]·d_silu[0]`」，
  §8.3 #3 删除该额外断言（主判据 `_tol_bwd` 不变），或改为断言「仅 `p=3` 有效」的等价关系。
- **⚠ 建议（不阻塞）：§3.3 step (c) 缺 `if tp < seq_len` 守卫**。`tp = t0+i-6`，`i` 循环到 `RC+5`；末块短尾时
  `tp` 可达 `t0+RC-1 ≥ seq_len`（例：train-s64 `RC=3`、`22×3=66>64` → 末块 1 token，`tp` 到 65；(3,401,256) `RC=23`、末块 10 token，`tp` 到 413）。
  §3.3 对 (c) 段是无条件 `T.copy(H/K/V → hrow)` 与 `T.copy(hrow → DH[b,tp,:])` → **越界读输入、越界写输出**。
  §5.4/§6.4 已声明「末块用均匀标量 guard」，但伪代码未体现。改法：step (c) 整体包 `if tp < seq_len:`（与 (b) 的 `pds < seq` 同形），
  并在 §6.4 明确「(a)/(b)/(c) 三段各自有独立 guard」。

### 4. 技术约束检测 — pass

- 三维 Kernel：不涉及，`T.Kernel` 一维（`programming.md:13`），任务索引 `b=bx//nch`、`c=bx%nch` 自解码 ✓。
- GEMM 非整除 / L0C 溢出：不涉及（无 matmul、不用 `alloc_l0*`）✓。
- 跨 block barrier：不需要（权重梯度走 2 次 launch + GM workspace）✓；不写任何 `ascend_set_flag`/跨核 flag ✓。
- `T.copy` 元素数相等约束：全部搬入/搬出带显式 region（`[:d]`、`[:, off:off+cols]`）✓（但归约输出的 region 上界需按维度 7 的建议收紧到 `d`）。
- 缓冲多版本：全部单版本并给出语义理由 ✓。

### 5. 循环与同步 — warn

- **循环结构**: §6.1 与 §1.6.2 的 14 点一一对应（token 推进 `T.serial(RC+2·KS-2)`、chunk `T.serial(dw//64)`、tap `T.unroll(KS)`、
  4 个 chunk pass、partial 搬出、归约 `T.serial(G)`、列 tile 走 grid）✓，无「1.6.2 判向量化却在 §6 出现对应标量循环」的情况 ✓。
- **同步与模式匹配**: 自动同步（`InsertSync`）+ 单版本 + 无显式 flag，与 SIMD 模式一致；两个 launch 之间靠 kernel 边界可见 ✓。
  4 个 chunk pass 共用 `gn/gz/hnf/knf` 是同址顺序读写，不需额外屏障 ✓。
- **尾块处理**: §6.4 覆盖 `d` 尾 lane / `seq` 末块 / `tokens<56` / 归约列尾块（缺口见维度 3 建议）✓。
- **⚠ 警告（不阻塞，已由设计自登记为 R-6）**: 探针编译日志出现 `V->MTE3: reverse ordering too loose to bound flag reuse` 与 `MTE2->V`
  同类告警（`(2,512,256)`、`(2,1,256)`、`(1,16,192)` 等几何）。判读：这是 InsertSync 对 flag 复用距离**无法给出上界**时的告警，
  不等于缺同步；但本设计的窗口是「同一 UB 槽跨 4 个迭代复用 + 每迭代独立 `T.copy` 行搬入」，若生成的同步被放宽到不足，
  表现为读到陈旧行（数值错而非崩溃）。**设计期无法证伪**（无真机 + 无同步 IR 证据），只能由 L0 数值用例判定。
  设计的处置（R-6：Stage 3 首跑即查对应几何，必要时插 `ascend_pipe_barrier`；不改变算法）是恰当的，故不判 fail。

### 6. 验证方案 — warn

- **Golden 函数**: §8.1 是 `workloads/sequence_modeling.py:260-306` 的忠实移植（fp32 autograd、6 个可微输入、
  `F.conv1d(groups=d)` + 左补 `KS-1`、`eps=1e-6`、`set_multithreading_enabled(False)`），不读 `vhat/alpha/rrms_*` ✓ 独立；
  「保存态已舍入 vs golden 未舍入」的有意差异已量化并作为 R-3 登记（fp16 8.9e-3 / bf16 5.4e-2，占 bf16 容差约 18%），
  与 `verify_equiv` 的 `fp64_gap` 实测值（8.883e-3 / 5.393e-2）**完全吻合** → 该差异确实远小于容差 0.2/0.3 ✓。
- **精度标准**: §8.2 = `_tol_bwd`（fp16 0.2/0.2、bf16 0.3/0.2），且**对每个输出分别施加**（test_engram.py:274 对 6 个输出逐个
  `allclose_compare(**_tol_bwd)`，含 3 个 fp32 权重梯度）→ 与 GPU manifest 容差一致 ✓，非自定标准。
- **确定性**: §8.2/§8.3 #7 要求 3 个权重梯度跨运行逐位相同（`torch.equal`），并给出归约顺序契约
  （核内 token 升序 + 跨核核号升序）；与 GPU 侧 `T.atomic_add` 顺序不定形成对照，**不追求与 GPU 逐位相同**的判断正确 ✓。
  误差上界独立复核：`n ≤ 2406`（= 2×(3×401)）→ `n·u = 2406×2^-24 = 1.43e-4` 相对，典型 `sqrt(n)·u ≈ 3e-6`，
  与实测 `drms_w_h` 2.44e-6 同量级；对 rtol 0.2 余量 ~1.4e3 ✓（顺序差异上界 ≤ 2·n·u 亦仅 2.9e-4）。
- **L0 计划**: 7 条用例覆盖 seq<4、单 chunk、素数 seq、`d%64≠0`、最大 UB、确定性 ✓，判据统一 `_tol_bwd` ✓。
- **⚠ 建议（不阻塞）**:
  (a) §8.3 #3 的附加断言错误（见维度 3），须删除或改正；
  (b) 缺仓库已有语义用例：`test_engram_gate_conv_bwd_dconv_w_keeps_nonfinite_like_torch`（`dY[0,-1,:]=±inf`、seq=9 分两个 run）
  正是 §0.6 R5「越界行不载入、只写零槽，杜绝 `0·inf`」的核心语义，L0 应显式加入（同输入比对 `dconv_w` 的非有限位置）；
  (c) `verify_equiv.py` 的 7 条几何中**没有 `d%64≠0`**（`(3,7,192)` 的注释「d not a multiple of 64」有误，192=3×64），
  而 §5.3 已把 `d` 非整除交 L0 `(1,16,100)` → 建议把 `(1,16,100)` 加入几何集（或在该节显式声明掩码路径不在等价验证范围内，现文已在 §5.3 声明，补一行为佳）。

### 7. 完整性与一致性 — warn

- **章节齐全 / 无占位符**: §0.1–0.7 + §1–§10 全齐，模板占位符零命中（§1.6.0「估算下界：待核验」与 §10.4「⬜」为如实登记的开放项，非模板占位）。
- **同类引用真实**: §3.5.3 的 6 条路径 + §0.7 的 3 条，逐个 Glob 均**存在** ✓（含前向兄弟目录的 `design_probe_compile.py`、源 kernel 副本、3 个同类 DESIGN.md）。
- **平台常数引用**: UB 253,952 B、56 Vector 核均与 `statectl platform` 实读一致；§0.5「950PR 与 DT 容量相同（UB 248 KiB）」经
  `.agents/platforms/hardware/ascend-950dt-9582.json` 复读成立（DT 亦为 253,952 B；DT 实现注释里的「216 KiB UB」是陈旧文字，不影响本设计）✓。
- **⚠ 建议（不阻塞）**:
  (i) **`out_idx` 与签名不一致**：§4.2 顺序为 12=dH,13=dk,14=dv,15=`drms_w_h`,16=`drms_w_v`,17=`dconv_w`,18=PART，
  而 §4.7 写 `out_idx=[12,13,14,15]` 并注释「15 = PART」→ 若照抄，3 个 fp32 权重梯度会变成**调用方必须传入的输入**、PART 也不被自动分配。
  改法：`out_idx=[12,13,14,15,16,17,18]`（或统一 §4.2 的顺序后在 §4.7 相应调整）。
  (ii) **归约 kernel 输出列越界**：§3.3/§6.4 的 `cols = min(width, dw - off)`，末 tile 在 `d%64≠0` 时 `off+cols > d`
  （例 d=100、dw=128：`off=64, cols=64` → 写 `DWH[64:128]`，越界 28 个 float；探针同形，但其几何的 d 全是 64 倍数故未暴露）。
  §3.5.1/§5.3/§0.6 R6 已要求「输出只写 `[:d]`」→ 改法：输出侧用 `out_cols = min(cols, d - off)`（`≤0` 时跳过该 tile），
  PART 读侧可保留 `cols`。L0 #6 `(1,16,100)` 是拦截点，务必按此实现。
  (iii) 笔误：§9.1 `sal=False` → `sat=False`；§1.6.1/§1.6.2 的 `vexpdiff` → `vexpdif`。
  (iv) 建议统一「launch 计数」（维度 0）与「§4.5 余量列」（维度 2）的表述。

### 8. 算法优化分析 — pass

**§1.6 章节存在性**: §1.6.0 / 1.6.1 / 1.6.2 / 1.6.3 / 1.6.4 五个子节齐全，1.6.1 有采纳与不采纳两栏，1.6.3 有候选矩阵 ✓。

**算法调研存在性与完整性（四问）**:
- R1 候选表 **10 行含基线 C0**（源算子 staged + 原子），覆盖恒等变形空间（确定性归约 / 融合窗口 / 物化复用 / 倒数乘 / 系数折叠 / CSE / 逐 tap 重算 / 无 fma 的 sigmoid 重排 / exp2 降代换），
  参考表（`knowledge/algorithm-candidates.md` 的 elementwise / reduction / softmax / RMSNorm 行）命中项已覆盖或 n/a（本算子无 GEMM/attention）✓。
- R2 **明确「有」** 且给结构判据（4 个行点积 = running 和、权重梯度 = 沿 token running 和、无需要重扫的全局量）+ 收益口径（扫描 3→1 遍、中间 26→8 B/lane）✓。
- R3 **四口径齐全且基线在表**：FLOPs（元素操作数/lane/token）、访存 GM 字节、扫描遍数、常驻中间缓冲，口径说明写清（各计 1、vcadd 不计、fp16 2 B / fp32 4 B）✓。
- R4 逐候选六维评估（单元匹配 / 容量 / 对齐 / 静态边界 / 流水融合 / 跨核）✓，C0 因原子无等价证据淘汰、C3 因 1.7× 计算换 12 KiB UB 淘汰，均有量。
- 调研范围显式列出（候选库 + operator-patterns + row-primitives + numerical-boundaries + execution-patterns + examples + 结构分析）✓；
  **源算子 12 项优化意图逐项承接**（#1→窗口、#2→有序归约、#3→不需要、#4→设备掩码、#5→56 核重导、#6→每核一 chunk、#7→保留、#8→vcadd、#9→R5、#10→保留、#11→2 launch、#12→舍弃）——我逐项对照 §0.4，**无静默丢弃** ✓。

**调研结论正确性（独立复核）**:
- 等价性：见「等价性论证独立推演」。
- 复杂度算术**独立复算**：84/97/117/145 × 221,184 lane（288×768）= 18.58M / 21.46M / 25.88M / 32.07M，与表内 18.6/21.5/25.9/32.1 M 一致；
  访存 29/48/50/29 B/lane × 221,184 = 6.41/10.62/11.06/6.41 MB，与表内一致；C2 的 29 B/lane 分解复核
  = 读 10×(1+3/RC=6 → 1.5)=15 + 写 6 + PART 往返 8 ✓。**中间缓冲 GM 往返未漏计**：C0/C1 计入 DS 4 B/lane 常驻与原子写、C2 只留 PART `O(G·dw)` ✓。
- 负向断言**逐条对照证据复核，无一条被推翻**：①「UB→GM 原子未核验」→ `PL-950-atomic-splitk` 原文限定 Cube L0C + `set_atomic` ✓；
  ②「高层二维 shared→shared 归约不可用」→ `TRAP-950/CG-950-simd-shared-reduce` 原文为 FP32 (2,64) 输出布局差异 ✓；
  ③「跨 block barrier 未认证」→ `CG-950-batch-barrier` ✓；④「`vexp2` 不存在」→ `simd.py` 实测 0 命中（仅有 `vexp`/`vexpdif`）✓；
  ⑤「`T.copy(transpose=True)` 的 UB 内转置未核验」→ `PL-950-conversion-layout` 原文「不是任意 stride Tensor 的支持声明，也未使用
  `T.copy(transpose=True)` 做 UB 内转置」✓（**弃选前提与实际代价机制一致，未被写成 GM 级 transpose**）。
- 并列候选有裁决路径：C2 vs C1 的取舍依赖两个未实测常数 → §1.6.4 给**三件套**（主选 C2 / 备选 B1 含 3 个 kernel 的 buffer 形状、循环结构、
  分核三要素 / 裁决计划含 6 条代表 shape、`msprof_task_duration`、A/B 交错 ≥5 轮、阈值「备选 < 主选×0.90 才翻转」、按 shape 分派、回写路径）✓，
  非纸面拍板。

**调研→设计一致性**: §1.4（2 launch / 单遍 / 窗口 / 有序归约）= §1.6.0 选定的 C2；§3.1 S0–S11 = 优化后公式；
§6 循环结构与 1.6.2 结论一致；§0.5/§0.6 的 R1–R6 处置与调研选族一致 ✓（无「选 A 做 B」）。

**数学等价优化完整性（四要素逐项）**: O1–O5 每项都有「原式 → 优化后 → 等价性论证 → 收益量化」，另列 2 项不采纳（候选 8 无收益、候选 9 API 缺失）✓。
**等价性论证独立推演**:
- O1（`x·fl(1/d)` vs `fl(x/d)`）：推导成立，二者差 ≤1 ulp（倒数一次舍入 + 乘一次舍入），设计判「容差内等价」而非「精确等价」——**判定正确**；
- O2（`rrms³·(1/d)` 折叠）：同上量级 ≤2 ulp ✓；
- O3（`zd=vhat·g` 复用、`direct=rrms·(w_v·g)`）：纯结合律重排，脚本实测**逐位 0 差** ✓；数学上 `Σ(vhat·w_v)·g ≡ Σ(vhat·g)·w_v` 成立 ✓；
- O4（`u_h=ddot·k_n`、`wuh=w_h·u_h`）：与源码 `dot_h=h*w_h*ddot*k_n`、`dH_dir=rrms*w_h*ddot*k_n`、`drms_w_h+=ddot*k_n*h*rrms_h` 同值，
  脚本实测 rel ≤7.9e-4（bf16 尺度，仍在 1 格点内）✓；
- O5（`vhat_norm` 物化）：同一表达式同一操作数 → 恒等（实测逐位 0）✓。
- 与 §8.2 精度标准一致（容差预算最大 0.0092 ≪ 1）✓。逐位/容差级的分层结论（O3/O5 逐位相同，O1/O2/O4 ≤1–2 ulp）**与内嵌表相符**。

**向量化替代完整性（与 §3.3 / §6 交叉核对）**: §1.6.2 盘点 14 个计算点，与 §3.3 伪代码、§6.1 循环表**逐一对应无遗漏**
（token 循环 #9、chunk 循环 #10、4 个 chunk pass、tap #1/#3、槽号/索引 #11、标量条带 #12、归约行序 #13、组选择 #14）；
9 处判「已替代」的 API 均在 `simd.py` 有导出且被探针使用；保留 5 处（#9、#10、#11、#13 行序、#14）理由**具体到本算子**
（依赖距离 3 的 tile 级顺序依赖 → 改 token 并行需把 `d_silu` 落回 GM 即备选 C1；chunk 循环每次迭代即一条 64 lane 指令；
每 token O(1) 标量索引；归约行序即确定性契约；block 级任务映射），**无「实现简单/照源码写」类空泛理由** ✓。

**替代 API 佐证**: 见维度 1（导出表 + 14 次编译）✓。

**向量化轴与布局决策完整性（§1.6.3）**: 7 个候选（原生 `[M,S,d]` d 轴 / 转置 `[M,d,S]` token 轴 / `d` 维跨核 / `[4][dw]` 窗口 /
`[dw][4]` / host padding / 标量条带）含**量化评分**（整除性、尾 lane 浪费率、累加链形态、repack 代价、UB 影响），
类别必选项「I/O 原生布局 + 最内连续轴」与「核内重排布局 + 高整除性轴」**一对齐备**（#1 vs #2），弃选理由量化
（token 轴：repack ≥16 B/lane ≈ 本体流量、行点积退化为每 lane 768 次标量；#5：向量宽度浪费 16×）✓；
`d=100` 尾 lane 28%、`d=7` 800% 的绝对量小已说明 ✓。**未把 I/O 契约当内部布局依据**（明确「I/O 是契约、核内布局是设计变量」）✓。

**布局决策与 §3.3/§4/§6 一致**: 窗口形状 `[KS,dw]`、`ch` 循环 `vld(…, ch*VL)`、累加循环内层向量维 = `d`、UB 预算按该布局计算（134 B/channel）✓ 三方一致。

**与 §3.1/§6 一致**: §3.1 明确「输入公式 = §1.6.1 优化后公式」，S0–S11 与 O1–O5 逐项可映射；§6 无 §1.6.2 未论证的标量循环 ✓。

**⚠ 建议（不阻塞，均不影响上述 pass 判定）**:
(i) §1.6.3 未把**卷积类**必选候选（UB 内 im2col 重排 / implicit GEMM）列入候选矩阵评分，只在 §2.2 与 §9 R-8 叙述性排除。
   对 4-tap 深度可分离卷积这两者显然劣（无通道混乘、im2col 展开 4× 无复用），建议补两行量化淘汰（FLOPs/访存/UB），使类别必选项齐全。
(ii) §1.6.0 R2 的术语：按 `references/algorithm-research.md` §3 R2 的定义，「partial kernel + merge kernel」本身不计入「在线算法」；
   本算子真正在线的是「token 串行 + 依赖距离内窗口」，per-core 部分和属跨核归约结构。结论（有在线变体、3 遍→1 遍）不受影响，措辞宜收紧。
(iii) §1.6.4 备选 B1 的 **d=4096 端 UB 算术偏高**：76 B/channel − 24 B/channel = 52 B/channel 等于把 3 张 fp32 表视为**零占位**；
   若表退化为 fp16 仍需 12 B/channel 常驻 → 64 B/channel → 4096×64 = 262,144 B > 253,952 B（超约 8 KiB），
   故 B1 实际覆盖约 `d ≤ 3900`。改法：或把 B1 上界写成 ~3900，或改为「表按 chunk 常驻（4×64 lane）」并重算（可到 4096），
   并在 §9 R-1 同步；这不影响主选（d ≤ 1024）。
(iv) §1.6.1 O1 的等价性论证末句「fp16/bf16 输出再舍入后差为 0（见验证表）」与该表自身矛盾（O1/fp16 最大 abs 4.883e-4 ≠ 0）；
   实测正确表述是「差 < 1 个输出格点（≤0.46 fp16 格点 / ≤0.5 bf16 格点）」。结论不变，文字需订正，避免 Stage 3 误以为可要求逐位。

**关于 `d` 上界（manifest `d ≤ 4096` vs 主选 `d ≤ 1024`）的处置判断**:
判定 **不阻塞**，但要求 Stage 3 落实「声明与实现一致」。理由：
1. 设计**未隐瞒**：§0.2/JIT 参数、§4.5 表、§9 R-1 三处明确主选上界 1024，并给出 UB 依据（142 KiB @ seq=512）；
2. 设计**给出了覆盖 1024<d≤4096 的备选**（§1.6.4 B1：3 launch，kernel A/B 的 buffer 形状、循环结构、分核三要素齐全，
   达 Stage 4 可直接实现深度）并给出**按 shape 分派**路径（「`d > 1024` 一律走 B1」）；
3. 实际可触发面为零风险：manifest 4 条 workload 的 `d` 为 256/512/768，`tests/ops/test_engram.py` 的 Bwd 用例 `d` 全为 256 → 当前无 `d>1024` 的合法性用例；
4. 剩余风险已转成明确的交付前确认项（§9 R-1「Stage 3/5 必须让 wrapper 的 `refusal` 与最终实现的上界一致，不得声明支持却编译失败」）。
**处置**：本检视要求（作为 Stage 3 的硬性检查项，不另设阻塞）：NPU 侧 wrapper/Op 的 `refusal()` 必须返回真实上界
（主选 1024，若实现 B1 则 4096），且不得沿用 manifest 的 `d <= 4096` 作为「已支持」声明；若 Stage 5 集成需要与 manifest `shape_rules`
字面一致，则按 §1.6.4 的 shape 分派实现 B1（其 d=4096 端算术需先按上文 (iii) 修正）。若 conductor/用户要求本迁移必须满足 `d ≤ 4096`
的可调用性，则应把「实现 B1 使上界达 4096」升级为 Stage 3 的验收项——这是产品口径决定，不是设计缺陷。

## 检视说明

按 skill Phase 3：所有**阻塞级**维度（0 源算子理解与迁移分析 / 1 API / 2 内存层级 / 3 Tiling / 4 技术约束 / 6 验证方案 / 8 算法优化分析）
均无 `fail`（维度 0、3、5、6、7 为 `warn`，warn 不阻塞），故判 **结论: 通过**。上述 13 条建议级问题已逐条给出
「位置 + 为什么 + 具体改法」，建议 Stage 3 在实现前先处理维度 3 的 `tp < seq_len` 守卫、维度 7 的 `out_idx` 与归约输出列裁剪
（三者若被照抄会分别导致越界读写与参数绑定错误），其余文字/算术订正可在实现中顺带完成。
