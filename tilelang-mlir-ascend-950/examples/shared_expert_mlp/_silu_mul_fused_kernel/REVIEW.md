# `_silu_mul_fused_kernel` 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息

- 文档: `examples/shared_expert_mlp/_silu_mul_fused_kernel/DESIGN.md`（1254 行，v1 首次设计）
- 任务类型: 迁移（migration-harness 子模式，`SharedExpertMLPFwdOp` 的第三段：gated activation）
- 源算子: `/workspace/TileOPs/src/tileops/_kernels/moe/shared_expert_mlp.py::_silu_mul_fused_kernel`
  （**实读**第 21–60 行；逐字抽取件 `examples/TileOPs/tileops/kernels/moe/shared_expert_mlp/_shared_expert_mlp_fwd_kernels.py`）
- 迁移后接线: `examples/TileOPs/tileops/kernels/moe/shared_expert_mlp/shared_expert_mlp.py`
  （`_decompose` / `_split_k` / `_silu_mul` / `default_config`，实读）
- 任务平台: `ascend-950pr-9579` / toolchain `ascend`（CANN 9.2.1，dav-3510，实查 28 Cube / 56 Vector，
  UB 253952 B / L1 524288 B / L0C 262144 B / L0A·L0B 各 65536 B；`statectl platform --task-dir` 2026-11-23 读取）
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 /
  验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 6（维度 3 ×2、维度 6 ×1、维度 8 ×3〔含 1 条口径 nit〕；**均不阻塞**，逐条附可执行改法，见各维度叙述）

## 机械复核（D-1/D-5）

### 1. `verify_equiv.py` 重跑（D-1，强制）

命令（与 §1.6.1 声明一致，torch CPU，无卡）：
`source .agents/platforms/toolchains/ascend/env-950pr.sh && cd examples/shared_expert_mlp/_silu_mul_fused_kernel && /workspace/venv-ascend950/bin/python verify_equiv.py`。

结果：**`EQUIV_RESULT: ALL_PASS`（8/8）**，且逐行数值与 §1.6.1 内嵌结果表**完全一致**：

| dtype | case | max_ulp | frac_gt1_ulp | max_rel | 非有限值不符 | tol 违反 | C2 max_ulp | C2 非有限值不符 |
|---|---|---|---|---|---|---|---|---|
| bfloat16 | random | 0 | 0.00e+00 | 0.000e+00 | 0 | 0.00e+00 | 0 | 0 |
| bfloat16 | wide-range | 0 | 0.00e+00 | 0.000e+00 | 0 | 0.00e+00 | 1 | 0 |
| bfloat16 | corner | 0 | 0.00e+00 | 0.000e+00 | 0 | 0.00e+00 | 0 | 5 |
| bfloat16 | exp-overflow | 0 | 0.00e+00 | 0.000e+00 | 0 | 0.00e+00 | 647 | 0 |
| float16 | random / wide-range / corner / exp-overflow | 0 | 0.00e+00 | 0.000e+00 | 0 | 0.00e+00 | 0 | 0 / 0 / 5 / 0 |

golden-form 对照行亦复现：bf16 `max_ulp=1 / max_rel=7.812e-03 / tol_violations=0`，fp16 `1 / 9.766e-04 / 0`
（与 §1.6.1 正文「bf16 1 ulp / 7.8e-3、fp16 1 ulp / 9.8e-4」一致）。→ 等价性机器验证**未缺失、未改动、
与文档一致**；C2 弃选反例（647 ulp / 5 处非有限值分类不符）为**可复现实测**，非估计。

### 2. `design_calc_check.py` 复算（D-5，强制）

命令：`python3 .agents/tools/design_calc_check.py --design examples/shared_expert_mlp/_silu_mul_fused_kernel/DESIGN.md --task-dir examples/shared_expert_mlp/_silu_mul_fused_kernel`

| check | status | detail（原文） | 人工复核结论 |
|---|---|---|---|
| `ub_budget` | **skip** | `no per-level buffer tables parsed from the memory plan (advisory checker; verify manually)` | **人工已核 pass**：§4.5 的千分位空格写法（`8 192`）未被 checker 正则识别。手算：`bm=16, bn=256` → 单 buffer/version `16·256·2 = 8192 B`，3 buffer × 2 version = **49 152 B ≤ 253 952 B（19.35%）**；`bm=1` 档 `3·2·256·2 = 3 072 B`；`bn=512` 档 `6·16·512·2 = 98 304 B（38.7%）`，×1.12 膨胀 = 110 101 B 仍 ≤ 253 952 B ✅ |
| `l0c_budget` | **skip** | `no Ascend Cube GEMM call found in the design pseudocode` | **人工已核 n/a**：本算子纯 Vector，无 Cube/L0C（§5.1、§4.4） ✅ |
| `core_split` | **fail** | `recomputed ceil(1/16) x ceil(1/256) = 1 vs declared 2` | **人工复核判定为解析伪阳性，非设计算术错误**：checker 从 §5 正文的 `M = 1`、`bn = min(block_n, 512)`、`bm = 16` 与表格串 `kimi-k2-t1`（取其 `2`）取到 `M=1, N=1, bm=16, declared=2`，故重算得 1。按 §5.5 表逐 workload 独立复算（`mt=ceil(M/bm)`、`nt=ceil(N/bn)`、`grid=mt·nsp`、`tpb=nt/nsp`）：t1 `1×72=72`（tpb 1）、t512 `32×6=192`（tpb 12）、ds-t32 `32×8=256`（tpb 1）、ds-t4096 `256×1=256`（tpb 8）、glm45 `32×6=192`（tpb 1）——**与 §5.5/§5.6 完全一致**；`nsp` 的「grid 最接近 4×56」规则亦自洽（含 t512 的 192/256 同距取小 192） ✅ |
| `r3_metrics` | pass | `all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak)` | 与 §1.6.0 R3 表一致（另做独立复算，见维度 8） |

### 3. 编译探针重跑（补充证据，非门禁项；决定维度 1/2/5 的结论强度）

`source .agents/platforms/toolchains/ascend/env-950pr.sh && ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python history_version/design_probe_compile.py`
→ **`PROBE_RESULT: ALL_OK`（4/4 COMPILE OK）**，几何与 §3.4 表逐项一致：
P1 `(4096,2048,bf16) (16,512,2) grid=512 tpb=2 UB=98304`、P2 `(32,1536,fp16) (1,256,3) grid=96 tpb=2 UB=3072`、
P3 `(1,1500,bf16) (1,512,3) grid=3 tpb=1 UB=6144`（双尾块 `am=1 / an=476` 动态 extent）、
P4 `(512,18432,bf16) (16,512,9) grid=288 tpb=4 UB=98304`。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

**核对方式：实读源算子全文（第 21–60 行）与迁移后 wrapper，逐项对照 §0.1–§0.7。**

| 检查项 | 结论 | 核对证据（源码位置 ↔ DESIGN 章节） |
|---|---|---|
| 语义理解正确性 | pass | 源 L50–L51 `g = gate_up[m_start+i, n_start+j]` / `u = gate_up[m_start+i, n_start+j+N]` ↔ §0.1（L29–34、L44）：gate 在**前半列**、up 在**后半列**、输出宽度减半，逐元素、零规约 ✅。源 L52–L56「cast f32 → `T.sigmoid(g_f32)` → `g_f32 * sigmoid_g * u_f32` → `T.cast(result, dtype)`」↔ §0.1 的「16-bit 精确加宽 → 全程 fp32 → 末尾单次舍入」四行代码块（L56–62）逐句对应 ✅。§0.1 的「无规约 → 无累加顺序舍入风险」与源码无累加轴一致 ✅。边界：源 L49 逐元素 `if` 守卫 ↔ §0.1「越界不写」，非有限值（`σ(-inf)=0` → `(-inf)·0=NaN`）与源码无特殊分支一致 ✅ |
| 算法解读完整性 | pass | §0.3 的 7 步表覆盖源全部计算语句：L43–L46（block 索引分解，步骤 1）、L50–L51（步骤 2/3）、L52–L53（步骤 4）、L54（步骤 5）、L55（步骤 6，左结合两次乘）、L56（步骤 7）；无遗漏、无臆造步骤 ✅。数据流图（§0.3 第 135–145 行）覆盖 `gate_up` 两个半区 + `out`，并指出「无 SMEM、无 Pipelined」与源 L48 直接读写 GM 一致 ✅。host 侧列出 `_silu_mul(cfg)` 取 `cfg["block_m"/"block_n"/"threads"]`（源 L~430）与 `_decompose` 的调用条件（`fused is None and not activated`）✅ |
| 优化手段覆盖完整性 | pass | §0.4 O1–O8 逐项标注硬件依赖：O1 融合 / O2 最小流量 / O3 fp32 中间 / O4 谓词尾块（算法与语义层，可移植）、O5 `T.Parallel` + 每线程向量化（warp/coalescing，L48）、O6 二维 tile 网格（L40–L46）、O7 `TL_ENABLE_FAST_MATH` + `-O3` + `-DENABLE_BF16`（源 L26–L30）、O8 `(128,256)` tile（源 L31 的 `threads` 默认与 wrapper `default_config`）✅。源实现无 SMEM / 无流水 / 无 warp 内建归约，故无「漏标」项；L48 的编译器向量化未被误判为算法层 ✅。行号引用有 2–4 行偏移（声明 24–64 行，实际 21–60 行，不影响逐项核对）——**建议（非阻塞）**：§0.7 与首页引用改为 `L21–60` |
| 耦合性判定合理性 | pass | §0.5 的 9 行每条都有「保留 / 等价替换 / 重新设计 / 舍弃」四态之一 + 依据：语义层（公式、fp32、单次舍入、I/O 契约）判「保留」✅；O4 判「等价替换」（区段 extent）且与源守卫语义等价 ✅；O5 判「重新设计」依据实查属实——`tilelang/ascend/language/kernel.py:139` 的 `T.Kernel(*blocks)` 文档明确「no ``threads`` parameter」「multi-dimensional launch is rejected」，与 §0.5 O5 行及 §2.3 描述一致 ✅；O7 判「舍弃」并写明数值方向（CANN `vexp` 默认精确变体，比 fast-math 更精确）✅；O8 判「重新设计」并给出夹取规则 ✅。无「无法判定」条目、无静默丢弃（§0.6 末段有 O1–O8 逐项承接核对）✅ |
| NPU 重设计可行性 | pass | R1/R2/R3 各含「源方案 → NPU 新算法 → 语义保持论证」三要素（§0.6 L210–248）。约束核验：一维 `T.Kernel` + 无 threads ✅（API 实查）；`T.SimdVF` 内不读 GM（所有 `T.copy` 在帧外，§3.3 L701–721 与设计一致）✅；UB 驻留 49 152 B（19.4%）✅；`bm/bn` 不整除走动态 extent 而非 host padding ✅；语义保持论证覆盖「fp32 中间 + 单次舍入 + 非有限值分类」（并由 `verify_equiv.py` 支撑）✅ |
| 设计与决策一致性 | pass | §1.4（L292–312）逐条标注源决策 → NPU 落地映射（R1/R2/R3）；§2.3 的线程/核模型与 §0.5 O5 处置一致；§3.3 伪代码即 R1/R2/R3 的落地形态（SIMD 链 + 区段 extent + `tpb` 流水），未出现「判重设计、§3 照抄源方案」的脱节 ✅。跨文件契约亦一致：`_gemm_basic_kernel(m=T, n=2S, k=H, trans_b=True)` 输出列 = `w_gate_up` 行序（gate 行在前）→ 本 kernel 取 `[:, :N]` 为 gate；本 kernel 输出 `[T,S]` = down GEMM（`m=T, n=H, k=S`）的 A ✅；与 `splitk_reduce_kernel` 的融合 epilogue（`gate = slice(…,j)`、`up = slice(…, out_n+j)`）同半区约定 ✅ |
| golden 独立性 | pass | §8.1 的 `golden_silu_mul` 只用「列切分 + `F.silu` + 乘法 + 一次 `.to(dtype)`」，不出现 `vdiv/vneg/UNPK_B16/PK_B32/(g·u)/(1+e^{-g})` 等 §0.6 实现痕迹，且以 §0.1 语义为唯一依据 ✅；`[:, :n]`/`[:, n:]` 显式固定 gate 在前，对应源仓负向对照 `test_shared_expert_mlp_keeps_the_gate_up_row_order`（实读 `tests/ops/test_fused_moe_shared_expert.py:145–162`：交换 `chunk(2,0)` 后 `moe_close` 必须失败）✅ |

**调用面核对（迁移后实际被调用的 workload）**：wrapper `forward` 在 `fused is None and not activated` 时才调
`_silu_mul`；`_decompose` 中 `gate_up_split = _split_k(num_tokens, hidden_size, block_k)`，`_split_k` 在
`num_tokens <= 32` 且 `k_tiles >= 2` 时取 2–4 片并让 `splitk_reduce_kernel` 融合 `silu_and_mul`。按
`_SPLIT_K_MAX_TOKENS=32`、`block_k=64` 逐条复算：kimi-k2-t1（T=1, k_tiles=112→split 4）、ds-v3-t32（T=32, 112→4）、
glm45-t32（T=32, 5120/64=80→4）走融合 epilogue；**kimi-k2-t512（T=512）与 ds-v3-t4096（T=4096）才调用本 kernel**
→ §0.3 L166–169 的「2/5 workload」断言**成立** ✅；调用签名 `_silu_mul` 传
`_silu_mul_fused_kernel(M, S, dtype)(block_m, block_n)`（2 参数，wrapper L~330）与 §3.3 / §10.5 的 K9「去掉 threads」
一致 ✅。

### 1. API 可行性 — pass

`§3.2/§3.4` 的每条 API 均在**本工具链实际模块**中找到签名佐证（`/workspace/venv-ascend950/lib/python3.12/site-packages/tilelang/ascend/language/`），而非其他后端：

- `T.Kernel(*blocks)`（一维、无 `threads`、多维被拒）：`kernel.py:139` ✅
- `T.SimdVF(latency=0)`：`frame.py:179` ✅；`T.simd.pset(elem_width=32, dist="PAT_ALL")`：`simd.py:111` ✅
- `T.simd.vld(addr, dist="NORM", *, post_inc=None)`，`"UNPK_B16"` 在允许集内：`simd.py:324` + dists 元组 ✅
- `T.simd.vcvt(src, target_dtype, mask=None, round="ROUND_R", sat=True, part=0, mode="MODE_ZEROING")`：`simd.py:1209` —— §3.2 步骤 5/7 的关键字写法（`part=0`、`round="ROUND_R"`、`sat=False`）与签名顺序一致 ✅
- `T.simd.vsts(addr, src, mask=None, dist="NORM_B32", extent=None)`：`simd.py:457` —— §3.3 的 `vsts(y_ub[…], v, full, dist="PK_B32")` 为位置参数 `mask=full` + 关键字 `dist` ✅（`PK_` 前缀自动把 `extent` 折半为 64 元素）
- `T.simd.vneg/vexp/vadds/vmul/vdiv`：`simd.py:887 / 860 / 1468 / 629 / 792`，调用形态 `(src[, scalar], mask)` 与 §3.3 一致 ✅
- `T.annotate_buffer_versions(dict)`：`annotations.py:19` ✅；`T.min` 二元 min：`language/tir/ir.py:181` ✅
- **精度变体核对**：`vdiv(..., precision=None)` 文档为「follows `tl.enable_fast_math`（fast math 关闭时为精确 fp32 除法）」，`TL_ENABLE_FAST_MATH` 默认 **False**（`transform/pass_config.py:83–87`），本设计 §4.7 又显式去掉了该开关 → 生成的即 §3.4 声明的 `vdiv_0ulp_ftz_true` ✅；`vexp` 同理走精确变体 ✅
- **已知陷阱一致性**：`examples/mla_dense_bwd/_dq_land_kernel/repro/CONST-950-vcvt-narrow-store-pack-b32.py` 实读确认「窄化 `vcvt` 的 64 个结果落在偶数 lane，稠密存储必须 `dist="PK_B32"`；pack 存储的谓词是 **32-bit lane 模式**，16-bit 掩码会静默漏写一半槽位」，与 §3.5.1 / §9.2 E2 / §1.6.2 #10 的写法一致 ✅；`sat=False` 与 `engram_gate_conv` 既有 3 处窄化调用（`_engram_gate_conv_fwd_kernel.py:344/389/433`）一致 ✅
- 无卡编译探针 4/4 重跑通过（见「机械复核」§3），`T.Pipelined + 双版本 + 动态 extent + bf16/fp16` 组合可编译 ✅

### 2. 内存层级规划 — pass

- **搬运路径完整**：§4.4 给出 `GM →(MTE2 T.copy)→ UB g_ub/u_ub →(vld/vcvt/…→vsts)→ UB y_ub →(MTE2 T.copy)→ GM`，无中间 GM 往返、无 L1/L0 参与（无 Cube）✅；每元素 GM 往返 = gate 读 1 + up 读 1 + out 写 1 = §1.6.0 R3 下界 ✅。
- **UB 预算（人工复核，机械项 skip）**：§4.5 数字逐一复算成立（49 152 B / 19.35%；`bm=1` 档 3 072 B；`bn=512` 档 98 304 B），远低于 253 952 B ✅。`assert 2*3*bm*bn*2 <= 253952`（§3.3 L685）与实际分配口径一致 ✅。
- **wrapper 128×256 的溢出与夹取**：`2·3·128·256·2 = 393 216 B > 253 952 B` → 若不夹取必然 `ub overflow`；§5.2 规则 1/2/5 的夹取（`bn=min(block_n,512)`、`bm=16 if M≥64 else 1`）使 wrapper 默认 `(128,256)` 落地为 `(16,256)` = 49 152 B ✅ —— **夹取规则对实际调用方 `default_config` 有效覆盖已核对**（wrapper 传 `block_m=128` 被忽略、`block_n=256` 保留；Stage 5 无需改 wrapper，§10.5 结论成立）。
- **Buffer 层级标注**：§4.3 三张 UB buffer + 向量寄存器（不物化）标注齐全 ✅；L0C/L1 明确 n/a ✅。
- **单元一致性**：字节/千字节口径自洽（`bm·bn·2` 单 version 8 192 B × 3 × 2 = 49 152 B），未混用 KB/KiB 两类口径（`253952 B` 与 `248 KiB` 混排处仅在结论句，数值同一）✅。

### 3. Tiling 策略 — warn

满足门槛项，存在两处**非阻塞**不自洽：

- **Block 划分明确**：§5.2 给出 `bn = min(block_n,512)` 取整到 64、`bm ∈ {16,1}` 两档及理由（`M ≥ 64` 与 decode 段饥饿）✅；§5.6 给出 5 条 workload 的 `bm/bn/nsp/tpb/grid`。
- **分核策略三要素（`_shared/standards/core-split-strategy.md` §1）**：① 逻辑核数计算齐全（§5.5 表 `grid = mt·nsp`，逐 workload 72/192/256/256/192）；② 物理核数依据**按平台规则实查**——引用 `statectl platform` 记录的 `Ascend950PR_9579.ini [SoCInfo] ai_core_cnt=28 / vector_core_cnt=56` + 运行时探测 `rtGetAiCoreCount()=28`，并写出 ruleset 规定的查询代码（`torch.npu.get_device_properties` 分别记录 Cube/Vector，与 `workflow.md:29/55` 一致）✅；设计期该卡被 Stage 3 占用，故引用平台落库探测值而非现场重跑，§9.1 风险 6 已声明 Stage 3 首跑前复核——符合「禁止文档假设/经验值」的要求（值来自探测记录）✅；③ 规模判定 = 中等规模（1.29–4.57 波），给 `nsp` 对齐 4 波的方案与利用率 64–91% 的取舍理由 ✅。
- **不整除处理**：§5.4 + §6.4 用 `am/an` 动态 extent；5 条 workload 实测全无尾块，尾路径由探针 P3（`am=1`、`an=476`）编译覆盖 ✅。
- **建议（非阻塞，warn 原因）1**：§5.4 第 3 条写「`N < 64`：`bn = max(64, …)` 后 `N < bn`」，但 §5.2 规则 1 与 §3.3 的实现是 `bn = min(block_n,512); bn -= bn % 64`，**没有 `max(64, …)`**。若 `block_n < 64`（例如 Stage 4 误传 32），`bn = 32` → `nchunk = bn//VL = 0` → 整个 chunk 循环不执行，输出保持 UB 垃圾值且**静默**写回（正确性隐患但编译通过）。改法：把 §5.2 规则 1 改为 `bn = max(64, min(block_n, 512)); bn -= bn % 64`（或在构建期加 `assert bn >= 64`），并同步 §3.3 伪代码与 §5.4 第 3 条。
- **建议（非阻塞，warn 原因）2**：§5.2 规则 2 的 `bm = 16 if M >= 64 else 1` **完全不读 `block_m`**，因此 §5.6 注「`bm = 32` … 列入 Stage 4 的调优清单」在现有夹取规则下**不可达**（Stage 4 改 config 不会生效）。改法：明确二选一——(a) 在 §5.6 注中把 `bm=32` 标为「需同时放开规则 2 上限（≤32）」，或 (b) 规则 2 改为 `bm = clamp(block_m, 1, 32)` 再按 `M` 下取整，并把 UB 断言保留为唯一护栏。

### 4. 技术约束检测 — pass

- 三维 Kernel：不涉及（一维 `T.Kernel(grid)`，`kernel.py:139` 明确多维被拒）✅
- GEMM 非整除：不涉及（无收缩轴，§6.4 已说明）✅
- L0C 溢出：不涉及（无 Cube；§5.3 明确 n/a）✅
- §3.5.1 本项目已知限制表 11 行逐项判定，命中项（一维 grid / SIMD 帧内禁读 GM / 整 buffer 拷贝需显式 region / 窄化 `vsts` 陷阱 / UB 容量 / 自动调度）均给出处理方案，且与 §3.2、§3.3、§9.2 的描述一致 ✅

### 5. 循环与同步 — pass

- **循环结构**：§6.1 表明确 `T.Kernel(grid)`（并行派发）/ `T.Pipelined(tpb, num_stages=2)`（软件流水）/ `T.serial(bm)` / `T.serial(bn//64)`（调度循环，且后者即向量化轴）/ `T.simd.*`（64 lane 向量），无逐元素标量循环，与 §1.6.2 的 10 点盘点结论一致 ✅（交叉核对：§3.3 伪代码里唯一的标量语句是 `im/sp/m0/n0/am/an/full`，全部属 §1.6.2 #7–#9 的 block 级索引/元数据）。
- **同步与模式匹配**：§7 采用自动同步（`allow_autoschedule`）+ `annotate_buffer_versions` 落实双版本，未手写 flag；§7.2 逐同步点列出依赖类型（MTE2→V、V→MTE2 反依赖 + 版本轮换、vsts→写回）✅；跨 block 无同步（各 block 独立）✅ 与平台「手工跨核 flag 未认证」一致。
- **尾块处理**：§6.4 说明动态 extent、死行/死 lane 不写回、无 K 方向；并指明本 5 条 workload 全无尾块、尾路径由 P3 保证可编译 ✅。
- **`tpb = 1` 退化**：3 条 decode workload 的 `T.Pipelined(1)` 已由探针 P3 编译验证 ✅。

### 6. 验证方案 — warn

- **Golden**：§8.1 给出可运行的 PyTorch 草案（fp32 上取整 → `F.silu(gate)*up` → 单次 `.to(dtype)`），与 §0.1 语义一致、与 §0.6 实现无关 ✅；并与源仓参考实现 `tileops/workloads/moe.py:gated_activation`（`chunk(2,-1)`）逐项对应 ✅；golden 形式 vs 原生 dtype 路径的 1 ulp 差异已实测并说明（`verify_equiv.py` golden-form 行）✅。
- **L0 计划**：4 例覆盖主路径（`bm=1`、`grid=256`、`tpb=1`）、fp16 dtype 路径、`M=1` 的 `am` 边界、`bm=16 + tpb=8` 流水分支；`M=64` 为非 manifest 形状并写明理由 ✅。
- **精度标准**：fp16/bf16 均 `atol=rtol=1e-2` + 归一化平方误差 ≤1e-4，与源仓 `moe_verification(1)`/`moe_close` 口径一致；并给出「fp32 中间 ⇒ ~10× 裕度」的依据 ✅。
- **建议（非阻塞，warn 原因）**：**溢出语义未列入风险/用例**。`vcvt(..., sat=False)` 的模块文档写「`False`: No sat; out-of-range values **wrap**」（`simd.py:1241–1247`），而 golden 的 `Tensor.to(float16)` 按 IEEE 溢出为 `inf`；fp16 路径（ds-v3-t32）的 `|y| = |silu(g)·u| > 65504` 时就可能出现「wrap vs inf」差异（bf16 输入同理，只是阈值更高）。manifest 的随机输入几乎不会触发，故不阻塞；改法：在 §9.1 增一条风险 + 在 §8.3 的 Boundary 清单加一个两元素用例（`u = fp16 max, g = 1`）确认设备侧实际语义；若确为 wrap，则需按 §0.1 的 IEEE 传播要求改用其它收窄路径或显式钳位到 `±inf`。

### 7. 完整性与一致性 — pass

- **章节齐全**：§0（0.1–0.7 齐全）+ §1–§10 共 11 章，模板要求项无缺 ✅
- **无占位符**：全文无模板占位符（花括号占位、未完成标记、待填写字样）✅
- **同类引用真实**：§3.5.3 的 5 条路径全部存在（`examples/engram_gate_conv/_engram_gate_conv_fwd_kernel/_engram_gate_conv_fwd_kernel.py`、`examples/mla_dense_bwd/_delta_kernel/_delta_kernel.py`、`examples/mla_dense_bwd/_dq_land_kernel/repro/CONST-950-vcvt-narrow-store-pack-b32.py`、`.agents/platforms/toolchains/ascend/kernel_template.py`、`examples/TileOPs/tileops/kernels/elementwise/mish/mish_kernel/mish.py`）✅；`pattern-library/constants.md` 的 `CONST-mte2-degradation`（18.4 / 43.1 GB/s·核）、`CONST-capacity-910B2C`（1.10–1.12 膨胀）、`CONST-gm-to-ub-bw-dilution`（段宽 ≥ 数百字节无惩罚）、`CONST-vector-launch-overhead` 条目均真实存在且数值与设计引用一致 ✅
- **内部一致**：§1.6.1 优化后公式 → §3.1 拆解 → §3.3 伪代码 → §6 循环结构 → §4.5 UB 预算 → §5.2 tile 五处口径一致 ✅；§5.5 与 §5.6 的 `bm/bn/nsp/tpb/grid` 表逐行相同 ✅；`nt/nsp = tpb` 全部整除 ✅
- 唯一内部张力是维度 3 列出的 `bn` 下限写法与维度 8 列出的 §1.6.3 #2 UB 数字，均不构成跨章节矛盾。

### 8. 算法优化分析 — warn

**§1.6 章节存在性**：1.6.0（调研四问）+ 1.6.1（等价优化）+ 1.6.2（向量化替代）+ 1.6.3（向量化轴与布局）+ 1.6.4（实验裁决）齐全 ✅。

**1) 算法调研存在性与完整性 — pass**

- R1 含**基线 C0**（源算法 `g·σ(g)·u`）+ C1–C7 七类结构候选，逐行给等价性初判与收益方向；平台 `knowledge/algorithm-candidates.md` 的 elementwise 命中行（「输入转 FP32，行标量广播后逐元素计算」）已评估（即 C0/C1 形态）✅。
- R2 给出**明确结论「有，且已是单遍」**，并以结构判据支撑（逐元素无跨元素依赖、无 running 统计量，故「两遍→单遍」无空间；唯一的「两遍」是算子级 gate/up GEMM 物化，属 C5 而非本函数可在线化对象）——判定正确，不存在与源算法矛盾（源本身已是融合单遍）✅。
- R3 四口径齐全且**基线在表**：向量算符/64 元素（12 vs 11）、访存 Bytes（50.33 / 56.62 MB）、扫描遍数（1）、中间缓冲峰值（49 152 B UB 驻留）、并附可并行度/跨核代价；口径句写明「不含 load/store 与 vcvt、超越函数按 1 条算」✅。
- R4 六项检查逐候选评估，负向淘汰附依据：C2（机器实测反例）、C3（exp2 多 1 条 `vmuls`）、C4（语义层变更）、C5（契约外，且记录为算子级后续优化项，非静默丢弃）、C6（契约/别名）、C7（+67% 流量）✅。
- 调研范围写明（候选库命中行 + PL-950 条目 + constants 条目 + 4 份本仓同后端实现 + 源仓两份 silu 实现 + 恒等变形自查），并如实记录「未使用互联网检索」（轻量调研允许）✅。

**2) 调研结论正确性（独立复核）— pass（含 1 项 warn 见下）**

- **等价性独立推演**：`g·(1/(1+e^{-g}))·u = (g·u)/(1+e^{-g})` 成立（乘法结合/交换 + 倒数定义，实数域精确）；舍入点 3 → 2，误差方向不增 ✅。我按设计同口径独立回放（fp32 逐步舍入 + 末尾 dtype 舍入）复算了 8 个用例的语义并与机械重跑结果一致；非有限值分类推演亦成立（`g=-inf → num=∓inf, d=inf → NaN`；`g=+inf → d=1 → ±inf`；`g=NaN → NaN`；`u=inf, g=0 → 0·inf=NaN`），与基线逐类相同 ✅。
- **C2 反例可信性复核（独立读码 + 复现）**：`y=(g·u)·e^{g}/(1+e^{g})` 在 `g > 88.7` 时 `e^g` fp32 溢出为 `inf` → `(g·u)·inf = inf`、`1+inf = inf` → `inf/inf = NaN`，而真值 `(g·u)/(1+e^{-g}) → g·u` 有限 → **非容差内等价**，反例成立 ✅。`verify_equiv.py` 中 `rejected_c2()` 确实实现该式（非误记），实测 bf16 exp-overflow 647 ulp、corner 5 处非有限值分类不符（两 dtype 各 5 处），**可重复** ✅。
- **复杂度独立复算（全部吻合）**：`3MN·2` → 50 331 648 B（ds-t4096）/ 56 623 104 B（t512）/ 110 592 B（t1）✅；C7 `5MN·2 = 83 886 080 B` 且 `83.9/50.3 = 1.67`（+67%）✅；C5 `83.9 → 16.8 MB`（−80%）✅；向量指令数 `8.39M/64×11 = 1.44M`、`9.44M/64×11 = 1.62M`、`18432/64×11 = 3.2k` ✅；发射项 `1.44M/56/1.8e9 = 14.3 µs`、`1.62M/56/1.8e9 = 16.1 µs` ✅；地板 `18.4 GB/s×56 = 1.03 TB/s → 48.9/55.0 µs`、`43.1×56 = 2.41 TB/s → 20.9/23.5 µs` ✅；中间缓冲 `3·16·256·2 = 24 576 B ×2 version = 49 152 B` ✅。**未发现漏计中间缓冲 GM 往返**（R3 明确「无 GM 中间缓冲」，与 §4.3 一致）✅。
- **常数来源与适用性**：`CONST-mte2-degradation` 实读确认 18.4 GB/s/核（HBM 地板，256M 混合读写）与 43.1 GB/s/核（L2 驻留、48 核口径）两个口径都真实存在，设计按 56 Vector Core 重算并**显式标注「910B2C 外推、本型号未实测」**，且说明 `cost_measurements` 为空、「1 IPC @1.8 GHz」为未文档化假设——**没有把外推数字当承诺** ✅。
- **并列候选裁决路径**：C1 vs C0 非并列（C1 严格少 1 条算符、流量相同、输出 0 ulp），可纸面裁决 ✅；布局 #1 vs #2 判定裕度依赖未实测常数 → 按实验裁决模式交 §1.6.4（三件套齐全：主选 #1 的结构/UB/分核、备选 #2 的完整结构 `bm=8, bn=N, nsp=1, versions=1` 与按新粒度重算的逻辑核数、裁决计划含代表 shape / `msopprof` 指标 / **0.97 判定阈值** / `perf_feedback.md` 回写路径 / 按 shape 分派可行性）✅。
- **warn ①（非阻塞）**：§1.6.1「无其余优化空间 ③ 无高代价算符可降代换（`vexp`/`vdiv` 已是 SFU）」这一负向断言**不完整**。同模块存在融合算符 `T.simd.vexpdif(src0, src1) = exp(src0 - src1)`（`simd.py:1182`，要求同宽 fp32 向量），`vexpdif(0, g)` 可直接得到 `exp(-g)`，省掉 1 条 `vneg`，使链从 5 条降为 4 条（R3 的 11 → 10）。设计未评估该候选。改法：在 §1.6.0 R1 增加候选 `C8: t = vexpdif(0, g)`（等价性：`exp(0-g)` 与 `exp(-g)` 在 fp32 上对全部有限/±inf/NaN 输入恒等；收益：−1 条算符/64 元素，但本算子带宽受限，收益仅在发射项，量级 ≤ 1/11 发射项），并在 §1.6.1 给出采纳或弃选理由后同步 §3.1/§3.3。**判定为非阻塞**：该优化不改变算法族结构、不影响任何布局/分核/UB 决策，且 §1.6.0 的结构性结论（带宽受限、流量已达下界）不受影响；另注：本平台**不存在倒数类 intrinsic**（`simd.py` 无 `vrec`/`vrecs`，`vdiv` 为唯一除法入口），故 skill 常举的「除法→乘倒数」项在本平台上不成立，设计未列该项不构成缺口。
- **warn ②（非阻塞）**：§1.6.3 候选 #2 与 §1.6.4 备选的 **UB 复算有误**：`3 buffers × (1×18432×2 B) = 110 592 B/version`，**2 version = 221 184 B < 253 952 B（×1.12 = 247 726 B 仍在限内）**，故「N=18432 时 … 2×110 KB > 248 KB **不可双缓冲**」不成立（216 KiB < 248 KiB）。实际排除 `N=18432` 的真正理由是 kimi-k2-t1 的 `mt=1 → grid=1`（设计已在同句写出），而 kimi-k2-t512 在该备选下 `bm=1` 时 `mt=512`、`grid=512`，双缓冲可行（余量仅 ~2.5% 含膨胀）。改法：把 §1.6.3 #2 的「不可双缓冲」改为「可双缓冲但含 1.12 膨胀后余量仅 ~2.5%」，并把 §1.6.4 备选的适用 shape 范围改为「`N ∈ {2048,1536}` 时 `bm=8`；`N=18432` 时 `bm=1` 仅 kimi-k2-t512 可用、kimi-k2-t1 因 `grid=1` 不可用」，避免 Stage 4 依据错误前提跳过 t512 的 A/B。

**3) 调研→设计一致性 — pass**

§1.6.0 选定 C1 后：§1.4 的算法描述（单遍、`(g·u)/(1+e^{-g})`、区段 extent、UB 双缓冲）即 C1 落地；§1.5 数据流图同样写 `num=vmul(g,u); y=vdiv(num,den)`；§3.1 公式拆解 7 步以 A1 为输入；§6 循环结构为单遍无第二遍扫描 ✅。迁移侧：§0.5/§0.6 的处置与调研选族一致（O1/O2 保留 = 本 kernel 即融合形态、无额外流量；O4 等价替换 = 区段 extent；O5→R1；O6/O8→R3），且 §0.6 末段对 O1–O8 逐项给出「意图承接 / 不再需要的理由」，**无静默丢弃** ✅。

**4) 数学等价优化完整性 — pass**

A1 四要素齐全（原式 6 条 → 优化后 5 条 → 有理式恒等 + 舍入点 3→2 → `12→11` 条/64 元素、流量不变）✅；并额外做了 5 项「无余空间」结论（无 CSE、无归约、无降代换、访存已在下界、稳定化不适用、C5 契约外）✅。算符计数独立复核：C0 = `vneg+vexp+vadds+vdiv(1/d)+vmul+vmul = 6`，C1 = `vneg+vexp+vadds+vmul+vdiv = 5`，与 R3 的 12/11 自洽 ✅。**口径注（非阻塞）**：C1 采用的 `precision=None` 精确 `vdiv` 内部为 Newton 精化序列而非单条 SFU 指令，故「−16.7% 算术链」是上限值；因 C0 与 C1 各含**恰好一次**除法，C1 优于 C0 的结论不受影响，建议在 §1.6.1 收益栏标注「精确除法按 1 条计，展开后相对收益更小」。

**5) 等价性论证独立推演 — pass**

逐项推演成立（见上文 2) 第一点）；容差内等价性未被滥用：C4 因破坏「fp32 中间 + 单次舍入」被正确排除在等价性论证之外（语义层变更），与 §8.2 的 1e-2 标准不冲突 ✅。**范围注**：0 ulp 结论是「torch fp32 回放」口径，不覆盖设备侧 `vexp` 的 SFU 近似误差；§8.2 的 1e-2 容差与 2 次舍入结构覆盖该差异，故不构成缺口（建议 §1.6.1 一句话点明该口径边界）。

**6) 向量化替代覆盖完整性（与 §3.3/§6 交叉核对）— pass**

§1.6.2 共 10 点，覆盖 §3.3 伪代码与 §6 循环中的**全部**计算点：`d/num/y`（#1）、两处加宽（#2）、一处收窄（#3）、元素级谓词（#4）已替代；保留 6 点（#5 `T.Pipelined` 列 tile、#6 `T.serial(bm)`+`T.serial(nchunk)`、#7 block 索引、#8 `T.min` 元数据、#9 `pset(32)`、#10 动态尾 lane）。交叉核对结果：§6.1 表中没有 §1.6.2 未论证的逐元素标量循环，§3.3 中除 #7–#9 外无标量语句 ✅。替代/保留计数与任务预期一致（4 替代 / 6 保留）✅。

**7) 不可替代理由充分性 — pass**

#5「多段 DMA + 多指令链无法整体化为一条向量指令，且承担双缓冲职责」、#6「每迭代已是 64-lane 指令，展开只增 `div/mod` 标量索引」、#7「每核一次 block 映射」、#8「每 tile 2 次标量 min，且它正是 #4 的替代物」、#9「循环不变编译期掩码」、#10「真实 workload `N % 64 == 0` 不触发；`PK_B32` 谓词为 32-bit lane 模式，强加 16-bit 掩码会静默漏写一半槽位」——**逐项具体到本算子**，无「实现简单/照源码写」式空话 ✅。

**8) 替代 API 佐证 — pass**

全部替代 API 有本工具链模块签名 + 同后端已验证用法（`engram_gate_conv` 的 silu 链 `vdiv(vadds(vexp(vneg(acc)),1))`）+ 编译探针生成代码（`vlds_unpack/vcvt/vmul/vadds/vexp/vneg/vdiv_0ulp_ftz_true/vsts_pack_b32`）三重佐证 ✅；`vld` 无 `stride` 参数的事实已实查（签名仅 `dist` + `post_inc`），#3 的「API 缺失」弃选前提成立 ✅。

**9) 向量化轴与布局决策完整性 — pass（含 1 项 nit）**

- 候选矩阵 6 行（行主序 chunk 64 / 全行 tile / 列主序向量轴=M / 扁平视图解交织 / host repack / 原地写回），逐候选给 repack 路径与量化代价（UB B 数、MTE2 段数、算符增量、流量增量、可覆盖核数）✅；I/O layout 被明确定性为契约、核内布局为设计变量（§1.6.3 首段），未出现「I/O 是行主序所以内部也行主序」式论证 ✅。
- 弃选理由均为量化/可查证：#3「`vld(addr, dist, post_inc)` 无 stride 参数 → 指令数 ×64」（API 实查成立）、#4「+2 算符（+18%）且段宽已 ≥ 1 KiB 无 MTE2 惩罚」（`CONST-gm-to-ub-bw-dilution` 实读确认「2D 跨列块 copy 在段宽 ≥ 数百字节时性能正常」）、#5「流量 50.3 → 117.4 MB（+133%）」、#6「0 收益 + 破坏别名契约」✅。
- 布局决策与 §3.3（`g_ub[r, c*VL]` chunk 基址）、§4（UB buffer 形状 `(bm,bn)`）、§6（内层循环维 = N 向 chunk）三方一致 ✅。
- **nit（非阻塞）**：#4 把元素解交织写成 `pdintlv`（`simd.py:226` 为**谓词**解交织；数据侧应为 `vdintlv` 或 `vld2(dist="DINTLV_*")`），若按实 API 复算，该候选的代价可能低于「+2 条/64 元素」。因 #4 的淘汰主论据是「段宽 ≥ 数百字节时 2-D 跨段拷贝无惩罚」（已实测条目），结论不变；建议仅更正算符名并按 `vdintlv`/`vld2(DINTLV)` 复算代价，以保持候选库口径可追溯。
- 实验裁决模式：三件套齐全（见 2) 末条）；**仅备选 UB 数字需按 warn ② 更正**。

**10) 与下游章节一致 — pass**

§3.1 以 A1 为输入（L628 显式声明）✅；§6 循环与 §1.6.2 结论一致（无未论证的标量循环）✅；§4.5 UB 按选定布局计算 ✅。

## 检视说明

- 本报告不设问题清单章节（结论为通过）；上述 6 条建议级问题已按维度就地给出「问题定位 + 为什么 + 具体改法」，供 Stage 3/Stage 4 与后续设计修订直接取用（不阻塞进入 Stage 3）。
- 最高风险项仍是设备侧数值/存储形态（`vsts(PK_B32)` + 32 位掩码、`vdiv` 精确变体、`vexp` 精度、`sat=False` 溢出语义）——这些均已由 §3.2/§3.5.1/§9.1/§9.2 固定写法并以 L0 双 dtype 用例暴露点覆盖，属 Stage 3 的验证职责，不影响设计可行性判定。
