# `_gemm_basic_kernel` 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息

- 文档: `examples/shared_expert_mlp/_gemm_basic_kernel/DESIGN.md`（1467 行，v1 首次设计）
- 任务类型: 迁移（migration-harness 子模式，`SharedExpertMLPFwdOp` 的 GEMM 段）
- 源算子: `/workspace/TileOPs/src/tileops/_kernels/gemm/dense.py::_gemm_basic_kernel`（4669–4831 行），
  逐字抽取件 `examples/TileOPs/tileops/kernels/moe/shared_expert_mlp/_shared_expert_mlp_fwd_kernels.py`
- 任务平台: `ascend-950pr-9579` / toolchain `ascend`（CANN 9.2.0 SDK / 9.1.0 runtime，dav-3510，
  实查 28 Cube / 56 Vector，UB 248 KiB / L1 512 KiB / L0C 256 KiB / L0A·L0B 各 64 KiB）
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 /
  验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 8（分布在维度 0 / 6 / 7 / 8，**均不阻塞**；逐条附可执行改法，见各维度叙述）

## 机械复核（D-1/D-5）

### 1. `verify_equiv.py` 重跑（D-1，强制）

命令与执行环境：`cd examples/shared_expert_mlp/_gemm_basic_kernel && source .agents/platforms/toolchains/ascend/env-950pr.sh && /workspace/venv-ascend950/bin/python verify_equiv.py`
（torch CPU，无卡；与 §1.6.1 声明的运行方式一致）。

结果：**10/10 `EQUIV_PASS`**，且逐行数值与 §1.6.1 内嵌结果表**完全一致**（抽检对齐：decode-splitk4-bf16
`128.0 / 0.42 / 0.00 / 0.00e+00 / 0.00 / 0.00`、decode-splitk4-fp16 `2048.0 / 0.46 / … / 1.00 / 1.00`、
prefill-splitk1-bf16/fp16 `0.0 / 0.00 …`、longk-splitk2-bf16 `144.0 / 0.65`、widerange-splitk4-bf16
`768.0 / 0.40`、widerange-splitk4-fp16 `576.0 / 0.44`、k7168-splitk4-bf16 `4608.0 / 0.83`、
k7168-splitk4-fp16 `4864.0 / 0.84`、corner `0.0 / 0.00`）。A1/A3 位级与非有限值模式在全部用例 pass；
A2 的 `A2_ulp_gap_vs_baseline = 0.00`、`A2_violation_rate_gt1ulp = 0.00e+00`。→ 等价性机器验证与文档一致，
脚本未缺失、未改动。

### 2. `design_calc_check.py` 复算（D-5，强制）

命令：`python3 .agents/tools/design_calc_check.py --design examples/shared_expert_mlp/_gemm_basic_kernel/DESIGN.md --task-dir examples/shared_expert_mlp/_gemm_basic_kernel`
→ **7/7 pass，无 skip 项**（故无「机械跳过，人工已核」条目）：

| check | status | detail |
|---|---|---|
| `l1_budget` | pass | L1: 2 buffer entries sum = 144.0 KB; capacity = 512.0 KB |
| `l0c_budget` | pass | L0C: 1 buffer entries sum = 128.0 KB; capacity = 256.0 KB |
| `l0a_budget` | pass | L0A: 1 buffer entries sum = 32.0 KB; capacity = 64.0 KB |
| `l0b_budget` | pass | L0B: 1 buffer entries sum = 64.0 KB; capacity = 64.0 KB（恰好用满，见维度 2） |
| `l0c_budget`（tile 口径） | pass | block 128x256 x fp32 accum = 128.0 KB vs L0C 256 KB |
| `core_split` | pass | recomputed ceil(4096/128) x ceil(4096/256) = 512 vs declared 512 |
| `r3_metrics` | pass | all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak) |

容量常数取自平台 `hardware.capacity_bytes`（CANN 9.1.0 `Ascend950PR_9579.ini [AICoreSpec]`，
md5 2f0eca1f…，2026-11-23），与 DESIGN §4.5/§5.3 的数字口径一致（L1 147456 B = 144 KiB、
L0C 131072 B = 128 KiB、L0A 32768 B、L0B 65536 B）。

### 3. 编译探针重跑（补充证据，非门禁项但决定维度 1/2/4 的结论强度）

`ASCEND_RT_VISIBLE_DEVICES="" PROBE_OUT=/tmp/probe_gemm_review /workspace/venv-ascend950/bin/python history_version/design_probe_compile.py`
→ **12/12 编译通过**，代码生成事实与 §11.1 逐条吻合，并额外解出三条设计未直说的证据：

- `D1`（prefill 主选）：`__global__ __cube__`；`ascend_gemm_l1<128, 64, 256, 64, true, bfloat16_t>`；
  `asc_copy_l0c2gm(..., asc_quant_mode(16))`；**无任何 `buf_dyn_ub` 分配**（主选路径真的不经 UB）；
  整核只有 1 个 `for` 循环（K 循环），无逐元素循环。
- `D1` 的地址表达式解出**遍历序为 m-fastest**：A 装载基址 `A[(block_idx & 31)*917504 + step*64]`
  （917504 = 128·7168，即 m tile 取自快轴），B 装载基址 `B[(block_idx >> 5)*1835008 + step*64]`
  （1835008 = 256·7168，n tile 取自慢轴），C 落盘基址 `C[(block_idx & 31)*524288 + (block_idx >> 5)*256]`
  ——与 §0.6 R4 / §1.6.3 D4② 声明的 `bx` 慢 / `by` 快一致（独立于设计自述的验证）。
- `D1` 的 L1 版本环 `step % 3`，b1 版本步长 16384 元素（=256·64）、a1 版本步长 8192 元素（=128·64），
  a1 基址偏移 49152 元素 → **L1 合计 3·32768 + 3·16384 = 147456 B**，与 §4.5 表逐字节相符。
- `D11`（ns=4）生成 `step & 3` 4 版本环 + `PIPE_MTE1→PIPE_MTE2` 事件 0/1/2/3 四组预置与排空
  （§6.3/§5.3 的「事件 0..3」声明成立；探针脚本自带的 `l1_ring` 正则只匹配 `step % N`，故其
  JSON 该字段为空——是探针统计口径问题，不是设计缺陷）。
- `D3/D4/D5` 的 `nd2nz/dn2nz` 计数 1/1/2；`D6/D12` `quant_mode 0`；`D2` `half + quant 1`；
  `D7` 夹取 128→32；`D8` `ascend_gemm_l1<16,…>` + `asc_copy_l0c2gm(..., 256, 1, 36864, 16, …)`
  （m=1 单行 burst）；`D9` 双尾块的 `an/am` 直接进入 DMA 参数（无逐元素守卫）；`D10`
  `__mix__(1, 2)` + `asc_copy_l0c2ub(..., asc_dual_dst_mode(1))`。全部与 §11.1 一致。

## 检视详情

### 0. 源算子理解与迁移分析 — warn

**本维度由本次检视亲自 Read 源算子核对（`dense.py:87-108 / 175-237 / 4669-4831 / 4834-4994`，抽取件
`_shared_expert_mlp_fwd_kernels.py:160-305`，包装层 `shared_expert_mlp.py:164-300`），逐项结论如下。**

| 检查项 | 结论 | 源码核对证据 ↔ DESIGN 章节 |
|---|---|---|
| 语义理解正确性 | pass | `accum_dtype = "float"`（4691）↔ §0.1「fp32 累加」；`T.clear(c_local)` 在循环前一次（4757），`T.gemm` 不再清零 ↔ §0.1/§0.3 步骤 6–7；`output_dtype = c_dtype if split_k == 1 else accum_dtype`（4722）↔ §0.1/§0.2 的 `(m,n)`/`(split_k,m,n)` fp32；bias 签名只在 `has_bias and split_k == 1` 出现（4828）↔ §0.1/§0.2；`T.min(start + j, n - 1)` 夹取 + fp32 上加（`_add_bias` 106 行）↔ §0.1「bias 在 fp32 累加器上加」；M/N 尾块守卫 `if m_start+i < m and n_start+j < n`（4810-4812）与 K 尾块禁止的注释（4763-4766）↔ §0.1 边界语义。四种布局 `a_shape/b_shape`（4708-4713）↔ §0.1 布局表。**无偏差** |
| 算法解读完整性 | pass | §0.3 步骤 1–11 逐条对得上：编译期校验与 shape 推导（4714-4722）、块索引（4758-4759）、K 段起点（4762）、A 装载两分支（4767-4776）、B 装载三分支含 NN 尾块掩码（4777-4796）、`T.gemm`（4797-4804）、流水（4761）、bias（4807-4808）、`c_local→c_smem`（4809）、带守卫写回（4810-4812）、split-K 直接落盘（4813-4821）；host 侧 `_main` 签名装配（175-237）与 `global_symbol` 命名（236）↔ §0.3 步骤 1 与 §10.3。**无遗漏步骤、无臆造步骤** |
| 优化手段覆盖完整性 | warn | 12 项（SMEM tiling / mma / fp32 fragment / Pipelined / 多版本 / swizzled layout / `use_swizzle(10)` / split-K / epilogue 守卫 / bias 融合 / `c_smem` 中转 / 精确 tile 校验）与源码一一对应。**缺 1 项**：`eviction_policy="evict_first" if split_k > 1 else None`（4781 行，B tile 装载的 L2 淘汰提示）未进 §0.4 表，因而 §0.5 也没有它的四态处置（建议 3） |
| 耦合性判定合理性 | pass | 抽检四条关键判定均在安装包源码复核成立：① NT-only——`ascend/op/gemm/gemm_mad.py:162` `assert not self.trans_A and self.trans_B, "Ascend L1 GEMM currently only supports … (NT)."`；② `policy` 失效——`src/ascend/op/gemm.cc:30-40` `ComputeWarpPartition` 恒写 `m_warp=n_warp=1` 返回 `{1,1}`，`SelectInst` 恒 `kAscendMAD`（25-27 行）；③ `use_swizzle` 不可用——`threadblock_swizzle_pattern` 只出现在 `src/{cuda,cuda/cutedsl,metal,rocm}/codegen/*.cc`，ascend codegen 无该分支；④ fixpipe 量化——`src/ascend/op/copy.cc:116-132` `GetCCQuantPre` 对 fp32→bf16/→fp16/同 dtype 返回 16/1/0。`gemm.h:48-53` 的 `FRAC_N=16 / TILE_K_SUB` 四个 `static_assert` 与 103 行 `(sk & 1)` L0 双缓冲亦与 §0.5 O1/O2/O3/O4/O5、§3.5.1 一致 |
| NPU 重设计可行性 | pass | R1–R6 每项都有「源方案 → NPU 新算法 → 语义保持论证」：R1 NT 装载归一（探针 D3/D4/D5）、R2 L1 多版本 + 模板内 L0 双缓冲（`_compute_tile_k_sub` 复算：`l0_capacity=65536/2 stage → max_k_b = 32768·8/(256·16) = 64`，与本设计 `TILE_K_SUB=64` 一致）、R3 L0C 累加 + quant 落盘（D1/D2/D12）、R4 1-D `bid` 的 m-fastest（本次从代码生成地址独立解出，见机械复核 3）、R5 bias 的 Cube+Vector（D10）、R6 split-K 契约（D6/D7）。均满足平台 Kernel/容量/对齐/布局约束 |
| 设计与决策一致性 | pass | §1.4 的 6 条、§3.1/§3.2/§3.3、§4.4 的六级搬运路径、§6 的循环表、§7 的自动同步表全部描述 §0.5/§0.6 决策后的形态，未出现「§0 判重设计、§3 照抄源方案」的脱节（例：§0.5 O7 舍弃 `c_smem` 中转 → §3.3 主选路径确无 `c_smem`，且本次代码生成 `buf_dyn_ub` 计数为 0 佐证） |
| golden 独立性 | pass | §8.1 直接以 §0.1 语义写 `torch.matmul`（`ah @ bh` + 列广播 bias + `acc.to(out_dtype or a.dtype)`），不预切块、不写部分和、不做硬件量化；`golden_gemm_basic_splitk` 按 `k // split_k` 切片返回 `[split_k,m,n]`，与 §1.3 第二式一致，未复刻 §0.6 的 NPU 数据流 |

**本维度的建议级（不阻塞）问题**

- **建议 1（契约表述自相矛盾，须在冻结前修正文字）**：§10.4 第 2 条称「本 kernel **已内建**
  `min(block_m, m)` 夹取与 `m < 16 → split_k = 1` 强制」，§5.2 注称「kernel 内也可按规则 4 强制
  `split_k = 1`，**两条路径等价**」。两处与本文其余部分冲突：
  ① §3.3 伪代码只有 `block_m = min(block_m, m)` + `assert`（`m=1` 时 `min(128,1)=1` → `assert block_m % 16 == 0`
  直接抛错，**没有**强制回落）；
  ② 工厂内改 `split_k` **不等价**——`split_k` 决定 `output_shape`/`output_dtype`（源 4721-4722 行，
  §3.3 同构），而调用方已按原 `split_k=4` 构造伴随 `splitk_reduce_kernel` 并调用 `reduce_(partial, out)`
  （`shared_expert_mlp.py:289-298`）。kernel 静默改 split 会让主循环返回 `(m,n)/c_dtype`，与 §4.2 声明的
  `(split_k,m,n)` fp32 硬契约不符 → 破坏跨 kernel 接口。
  **改法**：§10.4 第 2 条改为「`m < 16` 且 `split_k > 1` → 工厂**编译期显式 `ValueError`**（不得静默改
  split_k，输出 shape/dtype 是契约）」；把「调用方在 `m < 16` 时传 `split_k = 1`、`block_m` 取
  `min(cfg_block_m, m)`」从「建议」升格为**硬交付要求**，并点名改动点
  （`SharedExpertMLPKernel._split_k` 增加 `if num_tokens < 16: return 1`；`_gemm_program` 传
  `block_m=min(cfg["block_m"], m)`），同时说明该回落会切到非融合路径（`forward` 走 `self._silu_mul`，
  依赖同批次的 `_silu_mul_fused_kernel`）。
- **建议 2（影响面描述与实际不符）**：§5.2 注/§0.6 R6 称该兼容性问题「会影响 5 条 workload 中的 3 条」。
  按本设计自己的规则 3（`block_m := min(block_m, m)`，且与 GPU 侧 `small_m_splitk_config` 的
  `"block_m": m`（`heuristics.py:940-966`）同构），`m = 32` 的两条 workload（ds-v3-t32、glm45-t32）
  由夹取自愈；**实际只剩 1 条**（kimi-k2-t1，`m = 1`，2 次调用）必须由调用方改配置。
  建议按此修正，并把根因写明：NPU 脚手架在 Stage 0 丢了 GPU 的 `small_m_splitk_config` 与 `num_tokens == 32`
  的精确 gating（抽取件 92-93 行自述「GPU gates it at 32 tokens」），本任务 `_split_k` 对 `T ≤ 32`
  一律给 4，属继承缺陷，不是本 kernel 设计的缺陷。

### 1. API 可行性 — pass

§3.2/§3.4 的 14 条 API 全部有「安装包源码 + 兼容示例 + 本任务探针」三源佐证，本次独立复核：

- 存在性与签名：`T.Kernel(N)`（一维、无 `threads`）、`T.alloc_l1/alloc_l0c`、`T.copy`、`T.dual_copy`、
  `T.SimtVF`、`T.Pipelined`、`T.annotate_buffer_versions`、`T.clear`、`T.gemm`、`T.cast`、`T.min`
  在 `tilelang.ascend.language` 实际导出（本次 import 逐项核对，全部 `True`）。
- 规则条款：`references/programming.md` 明载「`T.Kernel(N)` 一维、签名没有 `threads`」；
  「`transpose` 的文档作用域为 GM→L1 的 DN→NZ 路径」；`copy_op.py::dual_copy` 的 M-split
  `[M,N] ↔ [M/2,N]`（`dual_dst_ctl=0b01`）、L0C→UB 为硬件双目标、完整 extent 必须为偶；
  `copy.cc:134-149` `CheckCompactL0Region` 只要求尾部矩阵维零起点（`c0[:am,:an]` 起点 0 满足）。
  均与 §3.2/§3.5.1/§9.1 一致。
- 编译事实：探针 12/12 覆盖 §3.3 的全部形态（NT/TN/NN/TT、split-K、m=1 尾块、m+n 双尾块、
  1 版本/3 版本/4 版本、out_dtype=fp32、has_bias 混合），`clear_accum=(step == 0)` 在生成代码里
  以 `(step == 0)` 实参传入 `ascend_gemm_l1` 模板。
- 唯一「编译通过、运行期语义未核验」项已在 §9.1/§8.3 显式登记：`quant_pre` 的**运行期舍入模式**
  （§3.4 状态栏写「编译已验证；运行期舍入由 Stage 3 的 L0-1/L0-2 核验」），信息如实，不构成
  虚假 API 主张。

### 2. 内存层级规划 — pass

- 搬运路径完整（§4.4 六级：`GM→L1`(MTE2 nz 归一) → `L1→L0A/L0B`(MTE1, 模板内) → `L0C`(MAD) →
  `L0C→GM`(fixpipe quant)），主选路径**不经 UB**；本次代码生成确认 D1 无 `buf_dyn_ub` 分配。
- 预算复算与机械检查一致（L1 147456 / 512 KiB、L0C 131072 / 262144、L0A 32768 / 65536、
  L0B 65536 / 65536），且从生成代码的版本步长独立复现（机械复核 3）。
- **L0B 恰好用满 64 KiB 不构成越界**的判断成立：`_compute_tile_k_sub` 取
  `l0_capacity // num_l0_stages = 32768 B`，`b0` 每阶段 `256·64·2 = 32768 B` × 2 阶段 = 65536 B，
  模板只按 `(sk & 1) * L0B_STAGE` 索引（`gemm.h:103`），无额外尾块；`design_calc_check` 亦判 pass。
- buffer 层级标注齐全（§4.3 表逐行标 L1/L0A/L0B/L0C/UB，含 `has_bias` 变体的 3 个 UB 缓冲 98816 B / 248 KiB = 38.9%）。
- decode 配置（`bm=32`）与 `num_stages=4` 的预算（110592 / 196608 B）也在表内，与 §4.5 注一致。

### 3. Tiling 策略 — pass

- block 取值明确且给了理由（`block_n=256` 使 n∈{4096,3072,7168,5120} 全整除；`block_k=64` 使
  `TILE_K_SUB=64` 刚好用满 L0B；`block_m=128` 使 L0C 用 50%）。
- 对齐：`block_m/n/k` 均 16 倍数并加编译期断言，`k % block_k == 0` 前置（5 个 workload 的
  k∈{7168,18432,2048,1536,5120} 全被 64 整除）。
- 非整除：M/N 尾块走「装载区段截断 + 存储区段截断」，GEMM 用完整静态 tile（与源语义
  「死行/死列被 epilogue 守卫排除」等价）；K 尾块显式不允许并给出 host 侧换 `block_k` 的回落链
  （64/32/16），与源 `default_config`/`autotune_configs` 的 `block_k` 选择同构（4940-4956）。
- 分核三要素齐全（§5.5）：① 逻辑核数逐 workload 列出，本次按 manifest
  （`moe.yaml` 的 5 条 workload）复算 144/28/576/112/64/28/512/896/48/80 **全部正确**
  （例：glm45 down `mt=1, nt=5120/256=20, split=4 → 80`）；② 物理核数引用平台实查记录
  （`hardware.core_source`：`ai_core_cnt=28, cube_core_cnt=28` + `rtGetAiCoreCount()=28`，2026-11-23）
  并给出 `torch.npu.get_device_properties` 查询代码、说明本卡被占用故引用平台落库值并在 Stage 3
  首跑复核；③ 规模判定为「中等规模（1.0–32 波）」，方案 = 逻辑块超发 + 波次派发，并给出
  ①对齐说明（`n=7168` 时 `nt=28` 恰为物理核数；glm45 gate/up 48 块 2 波 → 86% 利用率）
  ②不核内串行的量化依据（单块 134 MFLOP ≈ 13 µs ≫ 派发开销，按「未实测假设 10 TFLOP/s」标注）
  ③`T.Persistent` 无平台正确性条目的定级。
- 参考：本次核实 `knowledge/` 下确实无 `T.Persistent` 条目，`bottlenecks.md` 确实写「当前没有 950
  真机验证的瓶颈模式或收益阈值」，故 §5.5③(b) 的依据成立。
- 说明（不构成 fail）：标准 §1 要素③的「中等规模」典型做法是把核数对齐物理核整数倍，本设计选择
  「超发 + 波次派发」并给了量化依据（含尾波损失上限）。这是有依据的偏离，且 §5.5 已把 worst case
  （86%）写出来，Stage 4 还有 `block_n=128` 之类参数级旋钮。

### 4. 技术约束检测 — pass

- 一维 `T.Kernel` + 核内 `bid` 分解（已由代码生成的地址表达式验证 `bx` 慢 / `by` 快 / `bz` 最快）。
- GEMM 非整除、L0C 溢出、L1 溢出、`TILE_K_SUB` 整除、`m=1` 单行 burst、`k==0` 六类边界均有
  对应小节与探针/回退（§5.4、§9.1、§9.2 E1–E9、§9.3）。
- 未触碰未认证能力：不使用手工跨核 flag、`annotate_unlimit_memory`、`T.Persistent`；`num_stages ≤ 4`
  因“事件总数未文档化”设上限（本次 D11 实测 4 版本 = 4 组 MTE1↔MTE2 + M→FIX 事件 0，与声明一致）。

### 5. 循环与同步 — pass

- §6.1 的循环表只有 block 级与 tile 级两类，与 §1.6.2 结论一致；主选路径无逐元素循环
  （代码生成佐证：D1 只有 1 个 `for`）。
- §7 全自动同步，且 §7.2 的每一条都用生成代码事件佐证：`PIPE_MTE2→PIPE_MTE1`（版本号
  `step % 3`）、`PIPE_MTE1→PIPE_MTE2` 回写保护、循环前 3 个预置 / 循环后 3 个排空、`PIPE_M→PIPE_FIX`
  事件 0、`FIX→V` 与 `V→MTE3`（D10）、L0 双缓冲在模板内（`gemm.h` 的 Event ID contract）。本次重跑
  D1/D11 后事件形态与 §7.2 完全对应（D11 为 0..3 四组）。
- §7.3 给了三级回退阶梯（降 `num_stages` → 去版本注解 → 才手工 `T.Cube()/T.Vector()`），
  与平台「手工跨核未认证」的定级一致。
- 尾块（§6.4）与 §5.4 同源，不重复扣分。

### 6. 验证方案 — warn

- Golden（§8.1）、L0 计划（§8.3，9 条覆盖 quant16/1、m=1、split-K、四布局、bias、双尾块、k=0、满 K
  真实几何）、精度标准（fp16/bf16 `atol=rtol=1e-2` + NSE ≤ 1e-4；fp32 `1e-4/1e-4`）齐备；bf16/fp16
  的 1e-2 与 NSE 判据有任务侧出处（`test_fused_moe_shared_expert.py` 第 37-40 行明载
  「elementwise `atol = rtol = 1e-2` plus NSE」+ `moe_close`），引用真实。
- **建议 4（判据与 golden 实现不一致，会造成假失败）**：§8.3 的 L0-4/L0-9 判据写「fp32 ulp ≤ 1 vs
  fp64 参照」，但 §8.1 的 `golden_gemm_basic_splitk` 用 `ah @ bh`（`ah` 由 `.float()` 得来）给出的是
  **fp32 结果**，其自身 fp32 累加顺序与 kernel 不同。K=256/2048 时两个正确实现之间通常就有数个
  fp32 ulp 差，判据「≤ 1 ulp」会把正确实现判死。**改法（二选一）**：① golden 对 fp32 部分和改用
  fp64 累加（`ah.double() @ bh.double()` 后 `.float()`），判据保留 `≤ 1 ulp`；② 判据改为
  `atol = rtol = 1e-4`（与 §8.2 fp32 行一致），并把「≤1 ulp」只用于 `split_k == 1` 的目标 dtype
  输出（L0-1/L0-2，该处 fp32 累加相对目标 dtype 足够细，可行）。

### 7. 完整性与一致性 — warn

- 章节齐全：§0.1–§0.7 七小节 + §1–§11 全在；本次全文扫描未发现模板占位符或未完成标记
  （`statectl gate 1 --dir …` 复跑亦为 `{"ok": true, "pass": true, "failures": [], "warnings": []}`）。
- 引用真实：§3.5.3 与附录引用的 16 个路径逐个 `-e` 检查**全部存在**
  （`PL-950-gemm-tiles.py`、`PL-950-mixed-epilogue.py`、`gemm-tiles.md`、`PL-950-cube-tile.md`、
  `execution-patterns.md`、`bottlenecks.md`、`algorithm-candidates.md`、`kernel_template.py`、
  `core-split-strategy.md`、`hardware-cost-model.md`、`references/programming.md`、
  `examples/gemm/example_gemm.py`、`examples/gemm/matmul.py`、`examples/mla_dense_bwd/_bwd_kernel/DESIGN.md`、
  `tests/ops/test_fused_moe_shared_expert.py`、`benchmarks/ops/bench_fused_moe.py`）。
- 内部矛盾 3 处（**均建议级**）：
  1. **建议 1 同款**：§10.4/§5.2 与 §3.3/§4.2 关于「kernel 内强制 `split_k = 1`」的表述冲突（详见维度 0）。
  2. **建议 5（bias 分支 extent 冲突）**：§0.6 R5 与 §1.6.2 L5 声明「只装载 `[n0, n0+an)` 有效区间」，
     但 §3.2 第 6″ 行与 §3.3 写的是 `T.copy(bias[n0:n0+block_n], bv)` 与
     `T.dual_copy(co, C[m0:m0+block_m, n0:n0+block_n])`（全 `block_n`）。N 尾块 + `has_bias` 时前者
     会越界读 bias、后者会越界写 C。**改法**：在 `has_bias` 时补 `assert n % block_n == 0`（并说明
     `dual_copy` 的 2:1 分片要求使 `an` 不能直接代入），或真正改用 `an` 并把约束写清。本 op 的 10 次
     调用 `has_bias=False`、n 全整除，不触发。
  3. **建议 6（先例限定）**：§3.5.3 把 `PL-950-gemm-tiles.py` 记为 R2/R4 的「高度相似」先例。该条目
     实际走的是**显式 L0 路径**（`alloc_l0a/l0b` + L0 符号化 region 的 `T.gemm(a0[:am,:ak], …)`，
     见其 25-49 行），本设计走**模板内部 L0 路径**（L1 静态输入的 `T.gemm`）。作为「多版本 +
     `T.Pipelined` + `clear_accum=step==0` + 尾块区段拷贝」先例成立，作为 L0 流水先例不适用，
     建议加一句限定（否则 Stage 3 可能误以为模板内路径已有精度先例）。

### 8. 算法优化分析 — warn

**§1.6 章节存在性**：1.6.0/1.6.1/1.6.2/1.6.3 四子节齐全。

**调研四问（§1.6.0）**：R1 候选表含基线 C0 且 C1–C6 逐项评估（含“非候选”的说明）；R2 有明确结论；
R3 四口径齐全（机械 `r3_metrics` 亦 pass）；R4 逐候选六列评估 + 淘汰依据；调研范围写明（平台
`algorithm-candidates.md` GEMM 行、「显式 L1/L0 操作数、输入布局转换和 K 分块累加」——本次核对
该行确实如此，且其对应条目 `PL-950-gemm-tiles` 就是本设计的 C0 形态）；「无更优替代」路径未走，
而是给了 C0+C1 的选定与依据。

**调研结论的独立复核**：

- 复杂度算术复算（我自算）：`2mnk` decode = 2·32·4096·7168 = 1.879 G、prefill = 2·4096·4096·7168 =
  240.5 G ✓；ideal bytes `(mk+nk)·2 + mn·2`：decode 59.18 + 0.26 = **59.4 MB** ✓、prefill 117.4 + 33.55
  = **151.0 MB** ✓；C1 额外流量 `2·split·m·n·4`：decode 4.19 MB（**7.05%** ✓）、kimi-k2-t1 gate/up
  1.18 MB（0.22% ✓）；C4 `2·m·n·4` decode 1.05 MB（1.7% ✓）；C3 `1.75·m·n·4` ✓。
- 负向断言复核：① 「无在线变体空间」——结构判据成立：基线已是 K 方向单遍 running sum，状态
  `O(tile)` 且无第二统计量（无 max/rescale），结论与我独立判断一致；② C3 淘汰（容量 `1.75·m·n·4`
  超 L0C 一个数量级 + fp16 抵消误差「未文档化假设」已标注）成立；③ C4 淘汰依赖 A1 的一次舍入
  等价（见下）；④ C5 淘汰（破坏 `block_k | k`）与源前置一致；⑤ C2 persistent「平台无正确性条目」
  经我检索 `knowledge/` 确认（无 `T.Persistent` 条目），并有 `bottlenecks.md`「不使用 910B 数值补齐」佐证。
- 未发现明显漏掉的已知候选。可选观察：源实现的 `m % block_m == 0` 前置在 GPU 上是必要的（split-K
  epilogue 无守卫，4813-4821），而 NPU 的 split-K epilogue 已用 `c0[:am,:an]` 区段截断 → 技术上
  可以放宽 M/N 精确性，让 `m=1` 也能 split（`block_m=16, am=1, split=4`），从根上消掉维度 0 的
  建议 1/2，并给最带宽受限的 kimi-k2-t1 4× 核占用。**建议 7（可选，需探针）**：在 §5.2/§9 记为
  待评估项（需一个「split-K + M/N 尾块 + rank-3 目标」的编译/精度探针；当前 D6 是全静态 extent、
  D9 是 rank-2 目标，组合未验证）。不成立则维持现状，不影响本结论。

**调研→设计一致性**：§1.4 的算法 6 条、§3.1 公式拆解（以 §1.6.1 优化后公式为输入，含 6′/6″ 分支）、
§6 循环表均按 C0+C1 选定形态描述；§0.5/§0.6 的处置与调研结论同族（未替换为 Strassen 类/两遍类），
源实现 12 项优化意图逐项承接到 R1–R6 / 舍弃项（唯一未进表的 `evict_first` 见建议 3/维度 0）。

**数学等价优化（§1.6.1）四要素与独立推演**：

- **A1（融合量化落盘，两遍物化 → 一次 fixpipe 舍入）**：四要素齐全。独立推演：两条路径送进同一个
  fp32→c_dtype 舍入函数的都是**同一批 fp32 值**（fp32→GM→fp32 是恒等映射），故等价性只取决于
  ①「fp32 往返恒等」（成立）②「fixpipe 的 quant_pre 舍入与基准 cast 同模式」。②在硬件上**未实测**
  ——§9.1 已如实登记，并给出界限（若为截断，误差 ≤ 1 个目标 dtype ulp，fp16 ≈ 5e-4 相对、
  bf16 ≈ 4e-3 相对，均在 `rtol=1e-2` 内）与暴露用例（L0-1/L0-2）。因此「**位级**等价」的措辞应限定
  为「在 quant_pre 为 RN 时位级等价，否则 ≤1 目标 ulp」，**建议 8**：§1.6.1 A1 的等价性论证栏补这一句，
  收益量化不受影响。
- **A2（split-K 分片累加）**：独立推演：数学上加法结合律成立；fp32 重结合误差的量级为
  `O(√K · eps_fp32 · Σ|p_t|) ≈ 1e-6` 相对量级，远低于目标 dtype 的 1 ulp（fp16 5e-4 / bf16 4e-3）与
  `rtol=1e-2`，故「目标 dtype 0 ulp 差」的实测结论与理论一致；分片边界 `k_slice·block_k = k/split_k`
  与 golden 的 `k // split_k` 切片一致，归约按 `s = 0,1,…` 顺序，与伴随 kernel
  （抽取件 343 行 `for s in range(1, split_k)`）一致。**成立**。
- **A3（NT 装载归一）**：独立推演：装载期置换只改内存排布，被乘元素集合、K 分块序、累加序全不变
  → 精确等价（零舍入）。**成立**。
- **证据强度（建议 8，不阻塞）**：`verify_equiv.py` 中 `A2_fp64_exact = bool(torch.equal(ref64, ref64))`
  是**恒真断言**（同一对象自比，不构成验证）；A1/A3 的两侧也由同一批 fp32 值构造
  （`split32.clone().to(torch.float32).to(dt)`、`a.t().contiguous().t()`），检查虽都 pass，但信息量接近
  「构造性验证前提成立」。更有信息量的是 A2 的 `A2_ulp_gap_vs_baseline` / 违反率 / 与 fp64 参照的比较。
  **改法**：删除恒真项（或用 fp64 参照的两条累加路径之差替代），并在 §1.6.1 把 A1/A3 的「机器验证」
  表述限定为「验证前提（fp32 往返恒等 / 转置保值）成立」，把「位级等价」的证据归到 A2 的 ulp 对照。
  另：widerange 用例按 A×1e3、B×1e-3 同比缩放，各项量级同比缩放、**并未显著加剧抵消**，因此
  「刻意制造抵消」的说法偏强；建议补一个「同号大项互减」型近抵消用例（reassociation 最坏情形）。
  这些是证据表述问题，不改变 A1/A2/A3 的采纳结论。

**向量化替代（§1.6.2）覆盖核对**：11 个计算点（L1–L11）与 §3.3 伪代码、§6 循环结构交叉核对**无遗漏**
（K 主循环、输出逐元素写、M/N 越界判定、bias 加、bias 越界夹取、块索引、split-K 段起点、
`am/an/ak` 的 `T.min`、GM→L1 装载、L1→L0 分块、host 校验）。结论「6 消除/替代（L2/L3/L4/L5/L9/L10）
+ 5 保留（L1/L6/L7/L8/L11）」与本文自算一致；保留理由具体到本算子（顺序依赖 / block 索引 /
编译期常量边界 / host 元数据），非泛泛而谈。替代 API 均有佐证（`T.SimtVF` + `T.Parallel` 的 lane=n
生成 `float2` + `__float22bfloat162_rn`，D10 实测；`T.copy` 的 MTE2/MTE1 由 DMA 与模板承担）；
与 §6 一致（§6.1 表确无逐元素标量循环），也与代码生成一致（D1 无 UB、无逐元素循环）。

**§1.6.3 向量化轴与布局决策**：按 Cube/MixCV 类别枚举了 5 个决策点（GEMM 形态 / epilogue 归属 /
bias lane / 遍历序 / 重排位置），类别内必选候选（矩阵布局、装载重排 vs 显式转置、epilogue 归属侧）
齐全，逐候选有量化评分与弃选理由；抽检两条弃选论证成立：① D1③「`bn=256` 时 L0B 单版本 32 KB，
`num_stages ≥ 3` 会超 64 KB」——我复算 `256·64·2 = 32768 B`，×3 = 98304 > 65536 ✓；
② D4②「m-fastest 的 A 面板 ≤ L2（58.7 MB < 128 MiB）」——按 `l2_size=134217728`（型号档案原值，
本次核对 `ascend-950pr-9579-notes.md` 第 16 行确有该字段且注明“不用作 UB/L1 容量”）与可精确计算的
足迹，量级判断成立；且已被代码生成证实为实际遍历序。判定未落在未实证常数的裕度内（§1.6.3 末段说明
成立）；D2① 的备选 D2② 结构深度达标并给了切换阈值（L0 目标 dtype ulp > 1）。三方一致（§3.3 buffer
形状/§4 UB=0/§6 循环）已核对。

**与下游章节一致**：§3.1 以 §1.6.1 优化后公式为输入（含 A1 的一次舍入、A2 的分片求和、A3 的装载
归一）；§6 与 §1.6.2 结论一致。

**本维度建议级问题汇总**：建议 7（放宽 M/N 精确前置的可选优化，需探针）、建议 8（verify_equiv 恒真项 +
A1 位级措辞限定 + widerange 用例设计）。

### 关于「wrapper 配置兼容性」的归属判定（conductor 指定重点）

**判定：属 Stage 5 wrapper/配置层（根因是 Stage 0 脚手架丢失 GPU 的 small-m 配置选择），
不属 Stage 3 kernel 内的兜底职责，也不构成本 kernel 设计的阻塞项。**

依据与可执行处置：

1. **前置本身是源算子语义的一部分**，NPU 设计保留它是正确的：`dense.py:4716-4719` 的
   `raise ValueError("split-K basic GEMM requires exact M/N/K tiles")` 源于 split-K epilogue
   **无守卫**直接写 `c[bz, m_start:m_start+block_m, …]`（4813-4821）。
2. **`block_m := min(block_m, m)` 与 GPU 侧自身的做法同构**：`heuristics.py:940-966` 的
   `small_m_splitk_config` 返回 `"block_m": m`（且仅对 `m == 32` 生效）。因此 kernel 内夹取是
   **承接源语义**，不引入偏差 —— 也使 `m = 32` 的两条 workload 自愈。
3. **`m = 1` 无法用「kernel 内兜底」解决且保持契约**：`split_k` 决定 `output_shape`/`output_dtype`
   （源 4721-4722，§3.3 同构），而调用方已按 `split_k = 4` 构造伴随归约 kernel 并调用
   `reduce_(partial, out)`（`shared_expert_mlp.py:289-298`）。工厂内静默改 split 会破坏 §4.2 的
   `[split_k, m, n]` fp32 契约（形状/ dtype 不匹配）。**所以 kernel 侧唯一正确行为是显式报错**
   （§3.3 现有的 `assert` 即此，缺陷只在 §10.4/§5.2 的文字声称“已内建强制/两条路径等价”）。
4. **处置（交给 Stage 5，条件明确、可执行）**：
   - `SharedExpertMLPKernel._split_k`：`num_tokens < 16` 时返回 1（本任务即 `T = 1` 的 kimi-k2-t1，
     2 次调用），并知悉这会把 gate/up 切到非融合路径（`forward` 走 `_silu_mul`，依赖同批次
     `_silu_mul_fused_kernel`）；
   - `_gemm_program`：`block_m = min(cfg["block_m"], m)`（或直接按 §5.6 表分档 128/32/16）；
   - 其余 4 条 workload 按 §5.6 表即可（`m=32` 两条亦可依赖 kernel 夹取）。
   - DESIGN.md 侧需同步的只有文字（建议 1/2），不需要改结构、不需要回退设计。
5. **风险评级**：若 Stage 3 严格照 §3.3 实现，`m = 1` 的失败是**编译期显式报错**（`assert block_m % 16`
   或 `m % block_m`），不会产出静默错值；Stage 5 有 §5.6 表 + §9.2 E6 可依。若 Stage 3 反照 §10.4
   加上「静默强制 split_k = 1」，才会得到契约破裂（形状不符）——这正是建议 1 要求删除该表述的原因。
   conductor 若认为「交付说明的契约表述必须在冻结前消除」，可按建议 1/2 走一次轻量 revision；
   本检视不将其判为阻塞。

### roofline 合规性核对（conductor 指定重点）

平台 `cost_measurements = []`，`knowledge/` 下无匹配型号/版本的常数条目，`bottlenecks.md` 明载
「当前没有 950 真机验证的瓶颈模式或收益阈值」「这些条目不使用 910B 的数值补齐」。
DESIGN §1.6.0 的「设计期估算」行给出**已算出的流量项**（decode 63.6 MB、prefill 151.0 MB）与
`估算下界: 待核验（缺少 ascend-950pr-9579 的 HBM 有效带宽与 Cube 峰值吞吐实测常数 …）`，
并以「与常数无关的结论」（AI 29.6 → decode 带宽受限；AI 1593 → prefill 算力受限）导出后续设计重点。
这与 `hardware-cost-model.md` §2 允许的 `估算下界: 待核验（缺少 <指标和适用条件>）` 形式**一致**，
且**未用 910B 常数补齐**（全文唯一的算力数字 10 TFLOP/s 明确标注为「未实测假设」且只用于量级比较）。
→ 合规，不构成缺陷。
