# `_engram_gate_conv_fwd_kernel` 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息
- 文档: `examples/engram_gate_conv/_engram_gate_conv_fwd_kernel/DESIGN.md`
- 任务类型: 迁移（GPU TileLang 源算子 → ascend-950pr-9579）
- 源算子: `/workspace/TileOPs/src/tileops/_kernels/sequence_modeling/engram/gate_conv_fwd.py`（全文 254 行，已逐语句核对）
- 只读旁证: `/workspace/TileOPs-ascend950/src/tileops/kernels/ascend/sequence_modeling/gate_conv_fwd.py`（232 行，两次 AIV launch）、
  `examples/TileOPs/tileops/workloads/sequence_modeling.py`（golden）、`examples/TileOPs/tests/ops/test_engram.py`、
  `examples/TileOPs/tileops/manifest/sequence_modeling.yaml`、`.../engram/call_spec.py`
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 5（均为文档证据链/算术标注类，不阻塞 Stage 3；见各维度内「建议级」条目与文末汇总）

## 机械复核（D-1/D-5）

- **verify_equiv 重跑**: `PASS`。命令
  `cd <repo> && source .agents/platforms/toolchains/ascend/env-950pr.sh && /workspace/venv-ascend950/bin/python examples/engram_gate_conv/_engram_gate_conv_fwd_kernel/verify_equiv.py`
  退出 0，尾行 `FAILED: none`。重跑输出的 10 行汇总表（A1/A2/A3/A4/ALL × float16/bfloat16，每项 36 用例、ALL 216 用例）
  与 §1.6.1 内嵌表**逐字段一致**（例：A1/fp16 `9 ulp / 1.788e-07 / 预算 0.0010`；A2/bf16 `3 ulp / 3.815e-06 / 2.857e-03`；
  ALL/fp16 `1 ulp / 2.441e-04`），全部 `EQUIV_PASS`，`nf_mis = 0`。逐输出明细亦与 §1.6.1 注一致
  （`rrms_h/rrms_k` 0 ulp；`rrms_v` 224 ulp / 24.86% 预算，与设计写的 224 ulp / 24.9%、baseline 24.77% 同阶吻合）。
- **design_calc_check**: 命令
  `.agents/tools/design_calc_check.py --design <DESIGN.md> --task-dir <task>` → 输出 4 条 checks：
  - `ub_budget` **skip** — "no per-level buffer tables parsed"；**人工已核**（§4.5 表逐项复算，见维度 2）；
  - `l0c_budget` **skip** — "no Ascend Cube GEMM call"（本算子无 Cube，skip 正确）；
  - `core_split` **skip** — "insufficient parsed dims / declared logical cores = None"；**人工已核**（§5.2 规则重算，见维度 3）；
  - `r3_metrics` **pass** — "all four R3 metrics present"。
- **编译探针重跑（补充，非 D-1/D-5 要求）**: `cd <task_dir> && ASCEND_RT_VISIBLE_DEVICES= /workspace/venv-ascend950/bin/python history_version/design_probe_compile.py`
  → `PROBE_RESULT: ALL_OK`，6 几何均 `COMPILE OK`，几何值与 §5.2 表逐项吻合
  （train-s96 `RC=6,nch=16,grid=48,nchunks=12`；s64 `RC=4,grid=32`；s128 `RC=4,grid=32`；s32 `RC=1,grid=32`；prime-95 `RC=1,grid=95`；pad-d196 `DW=256,nchunks=4`）。
  并从 `TILELANG_CACHE_DIR` 的 `linux-x86_64/kernels/<hash>/device_kernel.asc` 取出该程序的生成代码，逐条核对 §0.6/§3.3/§7.2 引用的生成物：
  `if (1 <= (((block_idx & 15) * 2) + (i_1 / 3)))`、`if (3 <= i_1)`、slot 索引 `(i_1 & 3)`、行搬入 `asc_copy_gm2ub_align(..., 1, 1536, ...)`、
  标量回写 `asc_copy_ub2gm_align(..., 6, 4, ..., 4, 32)`、**16 处 `asc_sync_*`**、`vsts_norm(alpha_v[0], …, one)`（PAT_VL1）、
  `vdiv_0ulp_ftz_true`（未开 fast-math）、`update_mask_b32(min(64, 196 - ch*64))`（d=196 的 64/64/64/4）——**设计对探针证据的引用全部属实**。

## 检视详情

### 0. 源算子理解与迁移分析 — warn

已 Read 源算子全文并逐语句对照 §0.1–§0.6、§1–§7、§8.1。

- **语义理解（正确）**: §0.1 与源码逐点吻合——`h_norm`/`k_norm` 共用 `rms_w_h`（源 96–97）；`alpha = 1/(1+exp(-(dot/sqrt(d))))`（102）；
  `vhat_val = alpha·v`、`vhat_buf = cast(vhat_val, dtype)`（109–110）；`rrms_v = rsqrt(Σ/d + eps)`（113）；因果 4-tap 左补 0（130–144）；
  `Y = cast(conv_out*sigmoid(conv_out) + vhat_cur)`（146–148）；分母恒为 `float(d)`（90/113，不是 `d_padded`）；
  `alpha/rrms_*` fp32、`Y/vhat` 16-bit（`call_spec.EngramGateConvFwdInterface.forward` 返回契约，已读）。
  §0.1「本算子无 dilation」经 `call_spec.py` 复核：`EngramGateConvCall`（第 26–35 行）确无 `dilation`，该字段只属
  `EngramDecodeCall`（第 46 行）✓；`CONV_KERNEL_SIZE = 4`（第 23 行）✓；host 侧 `F.pad` 到 `align_up(d, 256)`
  （`tileops/_kernels/tiling.py: ALIGNMENT = 256`）✓，`out_idx=[6..11]` ✓。
- **算法解读（覆盖完整）**: §0.3 S1–S10 与源码 77–148 行逐段对应，无遗漏、无臆造（含 host 侧 `F.pad` 六张量 + 裁回两输出，源 242–254）。
- **优化手段（覆盖完整）**: §0.4 O1–O12 覆盖 fragment 驻留、`T.reduce_sum`、两段 launch 物化（O4）、`if_then_else`（O5）、
  `align_up(d,256)`（O6）、fp32 中间量（O7）、逐 tap 重乘（O8）、多输出（O9）、`@target_cache`（O10）、`threads` 自动调优（O11）、
  `compile_flags`/fast-math（O12），逐项标了硬件依赖与可移植性初判，未把实现层优化误记为算法层。
- **耦合性判定（合理）**: §0.5 四态处置齐全且有依据（二维 grid+threads → 重新设计；`reduce_sum` → `vcadd` 等价替换；
  `vhat` 落 GM → 重新设计为核内所有权+窗口；`if_then_else` → 块级分支+预清零；`align_up(d,256)` → 尾 lane 屏蔽；
  `threads`/fast-math → 舍弃并给理由）。抽检的判定与 `references/programming.md`（`T.Kernel(N)` 一维、`T.SimdVF` 执行区域）不矛盾。
- **NPU 重设计（可行且语义保持）**: §0.6 R1–R6 均有「源方案 → NPU 新算法 → 语义保持论证」三要素；R3 的 5 条论证覆盖数学等价、
  halo 确定性、因果边界、累加顺序（tap 仍 p=0..3 升序，与源 `T.serial(4)` 同序）、代价量化。
  我独立复核了 halo 确定性：`g_r` 只依赖 `H/k/v[r]`、`w_h/w_v`、`eps`、`d`、按 `ch` 的 `mask`，**与 `t0`/核号无关**，
  `w2h_ub`/`cwv_ub` 各核同输入同指令 → `g_r` 逐位可复现；`i % 4` 环与 tap 读下标 `(i-3+p) % 4` 自洽
  （`i` 迭代内先写 `slot[i%4]` 再读，p=3 读到的正是本行），`i-3+p ≥ 0` 对 `i ≥ 3` 恒成立 → 跨核重复计算不改变结果。
  `r_v` 每行只算一次（§0.5 R5(d)/§3.3 第二个 chunk 循环 + slot），4 tap 复用，符合 A3。
- **设计与决策一致**: §1.4/§1.5/§3.3/§6 描述的是 §0.6 决策后的单 launch + 环形窗口算法，无「§0 判重设计、§3 照抄源」的脱节；
  §3.5.2 给出了与源算子的逐项差异表。
- **golden 独立性（成立）**: §8.1 逐行等同 `examples/TileOPs/tileops/workloads/sequence_modeling.py` 的
  `_rmsnorm`/`engram_gate_conv_fwd_torch`（同 `mean(dim=-1)` + `add(eps).rsqrt()`、同 `F.pad(K-1,0)` + `conv1d(groups=d)`、
  残差用未舍入 `v_hat`），未复用 §0.6 的任何 NPU 结构（无 chunk/窗口/halo/`cwv`/`Σh·k·w²`）✓。
- **建议级（不阻塞）**: §0.1 舍入点表 ② 与 §0.3 S7 未记录一处**源 kernel 与其自身 torch 参考的分歧**：源 108–113 行
  `vhsq[0,j] = vhat_val * vhat_val` 用的是**未舍入** fp32 `vhat_val`，而 torch 参考（`workloads/sequence_modeling.py:121`
  `_rmsnorm(v_hat.to(H.dtype), …)`）与 950DT 参考（`gate_conv_fwd.py:103–107` 收窄→展宽后平方）都用**已舍入**值。
  设计选择已舍入（与 golden 一致，**正确**），但文档未点名该分歧，Stage 3 若"照源码对齐"会引入 ~2.5e-5 的系统偏差。
  建议在 §0.1 增一行「源 GPU kernel 在舍入点 ② 与 torch 参考不一致（附两处行号）；本设计取 manifest `ref_api` 的 torch 语义」，
  并把 §0.3 S7 明确为「源用未舍入值 / 本设计用已舍入值」。

### 1. API 可行性 — pass

- 逐条在**实际安装模块**（`/workspace/venv-ascend950/lib/python3.12/site-packages/tilelang/ascend/language/` 下的 `simd.py`
  与 `frame.py`，tilelang 0.1.15，`import tilelang.ascend.language` 打印导出表）核对导出与签名：`SimdVF`、`vld(dist="UNPK_B16")`、`vsts(addr, src, mask, dist, extent)`、
  `vcvt(src, dtype, mask, round, sat, part)`、`vmul/vadd/vdiv/vexp/vneg/vsqrt/vcadd/vdup/vdupv`、`update_mask(value, width=32)`、
  `pset(elem_width, dist="PAT_ALL")`（`"PAT_VL1"` 在合法 dist 表内）、`alloc_local`、`annotate_buffer_versions`、`T.copy`、
  `T.Kernel(N)`（无 `threads`）、`T.unroll`、`T.serial`、`T.clear`——**全部存在**，§3.2 的调用形态与签名一致
  （`vsts(..., one, extent=1)` 第三位为 mask、`extent=1` 覆盖 access_ptr footprint；`vcvt` 的 `part=0` 为构建期常量）。
- §1.6.0 R1 候选 12 的负向断言「本平台无 `vexp2`」经导出表复核成立（`simd` 导出无 `vexp2`）✓。
- §3.4 的「GM 访问禁止于 `T.SimdVF` 内」「`T.clear` 降级为 `float2` 循环」两条由探针实测支撑，且 §3.3 已按前者把
  三张参数张量与行数据 `T.copy` 到 UB（参数在 frame 外，行搬运在 `if r >= 0` 内但也在 frame 外），当前代码可编译（探针重跑 ALL_OK，
  生成代码里 `vld` 的基址全为 `buf_dyn_shmem`，无 GM 指针）✓。
- 重设计避让正确：`T.reduce_sum`（`TRAP-950-simd-shared-reduce`）与二维 grid/`threads` 均未使用。

### 2. 内存层级规划 — pass

`design_calc_check.ub_budget` 为 skip，已人工逐项复算（§4.5，`d=768, RC=6`）：
7×(768)×2 = 10 752 + `cw_ub` 4×768×2 = 6 144 + `w2h_ub` 768×4 = 3 072 + `cwv_ub` 4×768×4 = 12 288 + `slot` 4×768×4 = 12 288
+ `st` 4×9×8×4 = 1 152 = **45 696 B = 44.6 KiB / 253 952 B = 18.0%**，与 §4.5 一致；其余三几何复算 15 360 / 30 592 / 15 744 B
亦与表一致；`d=4096` 边界 238 720 B（233 KB）与表一致，×1.2 断言拒绝与 §9.1 R6 自述一致。
- 搬运路径完整（§4.4 七段，全部 GM↔UB，无 L1/L0A/L0B/L0C、无跨核 workspace）✓；每个 buffer 标注了层级 ✓；
- 多版本膨胀有据：`annotate_buffer_versions` 全 1（§3.2 第 13 行、§6.3），并给了 1.7×（auto multi-buffer）与 1.10–1.12×（显式 alloc）两种口径
  （`CONST-capacity-910B2C` 两条已核对，1.7× 与 1.10–1.12× 数值与该条目一致）✓；
- 编译期断言 `footprint × 1.2 ≤ 253 952` 与 §4.5 表的算式逐项对应（`4·DW·4 + 4·DW·4 + DW·4 + 7·DW·elem + 4·DW·elem + 4·(RC+3)·32`）✓。
- 说明：§1.6.0 R4 表把「44.6 KB 常驻 + 12 KB 窗口 / 248 KB = 18%」写得易误读（44.6 KB 已含 `slot`，重复计应为 22.8%），
  属表述问题，数值结论（18%）正确——并入文末建议级第 2 条。

### 3. Tiling 策略 — pass

`design_calc_check.core_split` 为 skip，已人工复算 §5.2 规则：`target = max(1, ceil(M·seq/56))`，`RC = min(seq 的 ≥target 约数)`，
`RC>1 且 M·(seq/RC) < 28` 时退回 `RC=1`：
- train-s32 `target=1 → RC=1 → grid=32`；train-s64 `target=3 → RC=4 → grid=32`；train-s128 `target=3 → RC=4 → grid=32`；
  train-s96 `target=6 → RC=6 → grid=48`——与 §5.2 表**完全一致**，且由探针几何输出独立复现（见机械复核）✓。
- 分核三要素齐全：① 逻辑核数 = `M·(seq/RC)` = 32/32/32/48；② **物理核数实查**（`statectl platform --task-dir` 的
  `observed_cores` 的 cube=28 / vector=56；型号 ini 复核：我实读 `/usr/local/Ascend/cann-9.2.1/x86_64-linux/data/platform_config/Ascend950PR_9579.ini`，
  第 20–22 行 `ai_core_cnt=28 / cube_core_cnt=28 / vector_core_cnt=56`、第 50 行 `l0_c_size=262144`、第 51 行 `l1_size=524288`、
  第 58 行 `ub_size=253952`——与设计记录的查询结果**逐项一致**，未使用经验值或文档假设）；③ 规模判定 ≤56 → 无需适配，
  并对 `grid<56` 的波内利用率取舍（0.57/0.86）给了量化与 Stage 4 分派路径 ✓。
- 对齐/非整除：行 DMA 用 `[:d]`（元素数相等，`d` 任意宽度，探针 d=196 的 392 B 已编译）；向量段 64 元素；`d % 64 != 0` 走
  `update_mask(min(64, d-ch*64), 32)`（生成代码实测 `min(64, 196 - ch*64)` → 64/64/64/4）✓；`seq_len % RC` 由约数规则消除
  （§6.4 有逐场景表，含 `seq_len ≤ 2`、质数 `seq_len`）✓。
- 说明：探针的 `geometry()` 还保留了未用的 `nfull/nlast` 字段（早期带尾块版本的残留），不影响结论，仅提示 Stage 3 复制规则时以 §5.2 为准。

### 4. 技术约束检测 — pass

三维 Kernel 不涉及（一维 `T.Kernel(grid)`，`references/programming.md` 条款一致）；无 GEMM → 无 L0 non-divisible 与 L0C 溢出问题
（`l0c_budget` skip 正确）；§3.5.1 的 12 条平台限制逐条有处理方案，其中「SIMD_VF 内禁 GM」「块级矩阵/循环上界静态」
「`vcvt.part` 构建期常量」「`d ≤ 4096` refusal」均与已查证的平台条款或源码一致。

### 5. 循环与同步 — pass

- §6.1 对每个循环/分支标明类型（`T.Kernel` 并行 / `T.serial(RC+3)` 行串行 / 两个块级 `if` / `T.serial(nchunks)` 向量化 /
  `T.unroll(4)` tap），与 §3.3 伪代码一致，且不使用 `T.Parallel`（`SimdVF` 无线程域）✓；
- §7 同步策略为全自动（不写手工 flag），与选定 SIMD 模式一致；§7.2 的「16 处 `asc_sync_*`」经**生成代码实测复核为 16 处** ✓，
  且生成的配对与设计描述的现象吻合（块起始 notify、参数/行搬入后 wait、`vh_ub/y_ub` 单版本导致的 V→MTE3 配对、
  tile 末 `st` 的 MTE3→V 等待）；`slot` 的跨行依赖为同 pipe（PIPE_V）顺序依赖，无需 flag 的判断成立；
- 单版本缓冲的理由（slot 跨行携带、行缓冲写后即读）在 §6.3/§3.2 给出，并把双版本流水列为 Stage 4 手段（不改结构）✓；
- 尾块：§6.4 表覆盖 5 类场景，与 §5.4 不矛盾。

### 6. 验证方案 — warn

- §8.1 golden 已与仓库权威参考逐行核对一致（见维度 0），6 输出 shape/dtype 与 §4.2 一致，`equal_nan` 与 float32 标量分开比对 ✓；
- §8.2 精度标准：`Y/vhat` fp16 1e-2、bf16 2e-2；fp32 标量 1e-3 —— 后者偏离技能默认的 1e-4，但有实测依据
  （`rrms_v` 对 golden 的固有偏差 224 ulp / 24.9% 预算，来源是语义舍入点 ② 而非本设计近似，§9.1 R5），属**有据放宽**，
  且仍比 op 测试的 0.1 严 100×；L0-1 两 dtype、L0-2 窗口、L0-3 `d=196`、L0-4 `seq_len<3`、L0-5 最大几何、
  Boundary（`d=200`、`seq_len=95`、全零、inf/NaN）覆盖了每个结构风险点，L0 全过才进 L1/L2 的门槛明确 ✓。
- **建议级（不阻塞）**: `vcvt(..., sat=False)` 的 f32→f16 溢出语义是「wrap（不饱和）」，而 golden 用 torch `.to(dtype)` → **±inf**。
  两者都不同于 golden（`sat=True` 会饱和到 ±65504），因此 §8.3/§9.3 的「inf/NaN 与 golden 同样传播」只对 bf16（指数域同 fp32）
  与 NaN 成立，**fp16 且 |x| > 65504 时不一致**。manifest 4 条 workload 与 L0 数值闸门都远离该区间（|α·v| ≤ |v|，实测 alpha ≤ 1），
  故不阻塞；建议 §8.3 的 Boundary 行把 inf 用例限定为 bf16，或把 §9.3 的断言改为「NaN 同传播；fp16 溢出为平台 cast 语义差异」。

### 7. 完整性与一致性 — warn

- 章节齐全：§0（0.1–0.7）+ §1–§11 全在；对三类未完成占位记号（待办标记 / 中文「待补」类措辞 / 花括号占位符）grep 全 0 命中 ✓；
- 引用真实性：§3.5.3 的 7 条路径全部存在（`examples/mla_dense_bwd/_delta_kernel/_delta_kernel.py` 与同目录 `DESIGN.md`、
  `examples/moe_permute_align/_make_scatter_kernel/DESIGN.md`、`kernel_template.py`、`knowledge/repro/PL-950-simd-reduce.py`、
  `PL-950-normalization.py`、`examples/TileOPs/docs/gpu_to_npu_adaptation.md`），950DT 参考与源算子路径存在 ✓；
- API 映射 / 内存规划 / Tiling 三处互不矛盾：buffer 名与形状在 §2.3/§4.3/§3.3 一致（`slot` 4×DW fp32、`st` 4×(RC+3)×8 fp32、
  `cwv_ub` 4×DW fp32），Tiling 的 `grid/RC/nchunks` 与 §5.5/§6/§3.3 一致 ✓；`M` 为构建期常量在 §4.6 与 §4.7/§5.2/§10.3 口径统一 ✓。
- **建议级（不阻塞）**: ① §1.6.2 结论段写「保留 **5** 处循环/标量」，但表内只有 4 个 ❌（V10/V11/V12/V14），且结论段自己也只列了 4 类理由
  （14 点、10 替代、4 保留）——数字应为 4；② 标量回写在 §1.6.2 V9 与 §3.2 第 11 行的说明列写 `vsts(..., ONEPT_B32, extent=1)`，
  而 §3.3 伪代码与实际生成代码是 `vsts_norm(..., PAT_VL1)`（`one`）+ `extent=1`，两处表述不一致。
  后者同时说明 §9.1 R11 的风险被高估：实际生成物是 NORM + 单 lane 谓词（只写 4 B），不是 32 B 粒度的 `ONEPT_B32`，
  唯一的布局要求（`st` 行距 32 B）已满足。

### 8. 算法优化分析 — warn

- **§1.6 结构齐全**: 1.6.0 调研四问 + 1.6.1 公式优化 + 1.6.2 向量化替代 + 1.6.3 布局决策，四个子节均实质内容（非空标题）✓。
- **调研存在性与完整性**: R1 候选表 12 行且**含基线候选 0**（源算法），参考表命中行已评估（我核对
  `knowledge/algorithm-candidates.md` 全表：elementwise/broadcast、reduction、softmax、RMSNorm/LayerNorm、GEMM、attention、GEMM+ReLU，
  **确无 gate-conv 融合族**，与设计「无命中」一致；`kb_search` 的 no-match 与 conductor 预注入一致）；R2 在线算法有**明确正面结论**
  （单遍 gate + 有限宽度窗口 ⇒ 环形窗口 + halo，并指出「行内两遍不可消」的边界）；R3 覆盖四口径且基线在表；R4 逐候选
  六项（单元/容量/对齐/静态边界/流水/跨核）评估，负向淘汰附依据 ✓；「无更优替代」写明了调研范围（候选库全表、INDEX 全部条目、
  constants.md、examples 内 ascend-target 算子、kb_search）✓。
- **调研结论正确性（独立复核）**:
  - 等价性：A1（`Σ(h·r_h·w)(k·r_k·w) = r_h r_k Σ h k w²`，分配律 + `r` 与 `d` 无关）、A4（`/√d` → `·(1/√d)`）、
    A2（`cw·(v̂·r_v·w_v) = (cw·w_v)·(v̂·r_v)`）、A3（同一确定性表达式复用）**我独立推演均成立**；A1/A4 属 fp32 重分组、
    A2 属舍入位置移动，均有机器验证的 ulp 与容差预算支撑（重跑一致）。R1 候选 6（用未舍入统计量求 `r_v`）被判**不等价**，
    与 golden 的 `v_hat.to(H.dtype)` 事实一致，判否正确；
  - 复杂度算术：FLOPs 6.39 M 与 manifest roofline 公式逐项复算一致（`3·(96·16142 + 1536·378) = 6 390 720`）；
    扫描遍数 1 / 1.5 / 4 与结构对应；中间缓冲 `O(4·DW)` 12 288 B 正确、且正确识别了被消除的 GM 往返（`2·M·T·d·elem`）。
    **发现一处算术瑕疵（建议级，不改变结论）**：写侧字节把 `Y + vhat` 记为 0.66 MB，正确值是 `2·M·T·d·elem = 884 736 B ≈ 0.88 MB`
    （0.66 疑似把 1.5× 的 halo 系数误乘到写侧）；该项在基线与主选行**同样**低算 0.22 MB，故基线 4.66 → 4.87、主选 3.11 → 3.33，
    排序、31% 的流量节省幅度与「单 launch 更优」的结论均不变。同口径交叉核对里 manifest `bytes` 应为 2.23 MB（式内含
    `2·M·T·d·elem`），设计写 2.00 MB 亦来自同一处。
  - 亲和性淘汰有佐证：Cube 求点积（fractal `N≥16` vs 每行 1 输出、`RC×RC` 对角浪费）、FFT/变换域（`simd.py` 无 FFT 原语，
    已查导出表）、D 轴跨核切分（lane 减半 + 每行一次跨核合并）、host permute（额外全张量搬运）均有具体量化或 API 证据，
    与 `ascend-constraints`/`programming.md` 不矛盾；
  - 负向断言复核：`vexp2` 缺失（成立，见维度 1）、`g_h`/`g_k` 无公共子式（成立：`Σh²`、`Σk²` 互相不可约）、
    「无更优算法族」（候选库确无窗口/融合族，且候选 2 的单 launch + RC=1 已被量化比较）——未发现被漏掉的已知候选。
  - 并列候选裁决：主选（单 launch+halo）与备选（两段 launch）裕度 13% 落在未实测常数（launch 固定开销 2–5 µs、
    950PR halo 重读代价）的不确定区间内，设计给了**三件套**：主选、结构完整的备选（`grid=288/128/128/32`、UB 21.5 KB、
    无 halo、带 A1/A2/A4、无块级 `r>=0`）、裁决计划（代表 shape 4 条、`msoprof` Task Duration、A/B 交错 ≥5 次、
    阈值「备选 < 主选 × 0.9」、按 shape 分片回写）✓；D-2 估算的常数来源（`CONST-mte2-degradation` 18.4 GB/s/core 与 L2 43.1 GB/s/core、
    `CONST-vector-launch-overhead`）已在 `pattern-library/constants.md` 核对到原条目，且全部标注「910B2C 外推、950PR 未实测」✓。
- **调研 → 设计一致性**: §1.4 的算法描述、§3.1 的公式拆解（输入即 A1–A4 优化后式子：`alpha = σ(shk·r_h·r_k·(1/√d))`、
  `g = v̂·r_v`、`c = Σ cwv·slot`）、§3.3 伪代码、§6 循环结构**四处同口径**，无「调研选 A 设计做 B」；
  §0.4 的优化意图承接逐项有交代（O1→UB 行缓冲+寄存器累加、O2→`vcadd`、O3→单次 chunk 循环算 3 个统计量、O4→窗口替代、
  O8→A2+A3 消除、O11/O12→舍弃）✓。
- **数学等价优化完整性**: §1.6.1 对选定算法做了系统审视（分配律/结合律/常量折叠/CSE/权重折叠/访存削减），4 个采纳项均有
  「原式 → 优化后 → 等价性论证 → 收益量化」四要素，收益有具体数字（每行省 3d/4d/3·2d 次乘、省 2 个 fp32 行向量、省 GM 往返 2.2 MB）；
  「无进一步优化空间」有结论且逐项给依据（3 个不可再约归约、`SiLU/σ` 已是最小原语、`v̂` 舍入往返是语义要求、`1/√` 三个输出无法合并）✓。
- **向量化替代完整性（与 §3.3/§6 交叉核对）**: §1.6.2 的 14 个点覆盖了 §3.3 的全部计算点——行内平方/乘积/累加（V1）、
  4 个归约（V2）、16↔32 位（V3）、`r` 的倒数平方根（V4）、`σ`/SiLU（V5）、4-tap 累加（V6）、因果零边界（V7）、
  行搬入写出（V8）、4 个 fp32 标量输出（V9）、行循环（V10）、chunk 循环（V11）、核/块索引（V12）、尾 lane（V13）、
  块级条件（V14）；另补的前置清零与 `w2h`/`cwv` 预置由其向量形态归入 V1/V3/V7。10 点已向量化，4 点保留且理由具体到本算子
  （tile 级顺序依赖 / API 不可变值约束 / block 级索引 / 静态边界），无「实现简单 / 照源码写」式理由 ✓。
  保留项与 §6 一致：§6 中不存在 §1.6.2 未论证的逐元素标量循环 ✓。
- **向量化轴与布局决策**: §1.6.3 给出 3 候选的轴质量评分（整除性、尾 lane 浪费、累加链形态、repack 代价、UB 影响）+ 5 行候选矩阵，
  类别必选候选在场（规约的水平 `vcadd` vs lane=行的垂直累积；卷积的直接跨步窗口 vs 隐式 GEMM vs 4 行转置；
  另含 host permute 与 D 轴切分），逐候选有量化弃选理由（如候选 2 省 0.6% 指令但转置 ≥300 条 + 18 KB + 跨 lane 取数 → 净亏），
  未把 I/O 布局当内部布局依据 ✓；选定「I/O 原生 (M,T,d) + 行内 chunk + 向量轴 d」与 §3.3（`h_ub[ch*64]`、`slot[i%4, ch*64]`）、
  §4.5（按该布局算 UB）、§6（归约轴 = 向量维、行循环为归约实例 + 窗口携带维）三方一致 ✓。

## 建议级观察（不阻塞，不构成修订要求；供 Stage 1 后续修订或 Stage 3 参考）

1. **§0.1/§0.3 源 kernel 与 torch 参考在舍入点 ② 上的分歧未记录**（源 108–113 用未舍入值；golden 与 950DT 参考用已舍入值）。
   设计取的语义**正确**，但证据链缺一条；建议补记并给出两处源码行号，防止 Stage 3「对齐源码」。
2. **§1.6.0 R3 写侧字节与 manifest bytes 口径**：`Y + vhat` 应为 0.88 MB（现写 0.66），基线 4.66→4.87、主选 3.11→3.33、
   manifest 2.00→2.23 MB；另 §1.6.0 R4 的「44.6 KB 常驻 + 12 KB 窗口 = 18%」重复计入 `slot`（正确总量 45.7 KB = 18%），
   中间缓冲标度宜写 `O(4·DW)` 而非 `O(RC·DW)`。所有结论不变。
3. **§1.6.2 结论段「保留 5 处」应为 4 处**（表内 V10/V11/V12/V14）。
4. **`vcvt(sat=False)` 与 golden 的 fp16 溢出语义差异**：建议把 §8.3 Boundary 的 inf 用例限定 bf16，或改写 §9.3 的等价断言。
5. **`ONEPT_B32` vs `PAT_VL1`+`NORM` 表述不一致**（§1.6.2 V9 / §3.2 第 11 行 vs §3.3 与生成代码）：以后者为准，
   并据此下调 §9.1 R11 的风险等级（实际生成 `vsts_norm(..., one)`，只写 4 B）。
