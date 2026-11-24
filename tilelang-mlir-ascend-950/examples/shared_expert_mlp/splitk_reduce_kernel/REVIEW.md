# splitk_reduce_kernel 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息
- 文档: `examples/shared_expert_mlp/splitk_reduce_kernel/DESIGN.md`
- 任务类型: **迁移**（migration-harness，`SharedExpertMLPFwdOp` 第二段；DESIGN.md 含 §0.1–0.7）
- 源算子: `/workspace/TileOPs/src/tileops/_kernels/gemm/dense.py` 的 `splitk_reduce_kernel`（第 1986–2053 行；抽取件 `examples/TileOPs/tileops/kernels/moe/shared_expert_mlp/_shared_expert_mlp_fwd_kernels.py:308-375` 经 diff 确认为逐字副本）
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 6（逐项写在对应维度末的「建议级…」条目内，均为不阻塞项）

## 机械复核（D-1/D-5）

- **verify_equiv 重跑**: `cd examples/shared_expert_mlp/splitk_reduce_kernel && /workspace/venv-ascend950/bin/python verify_equiv.py` → `TOTAL cases: 110, failures: 0` / `[EQUIV_ALL_PASS]`。
  与 §1.6.1 内嵌表的**判据列逐项一致**（A1 fp16/bf16：max ulp = 0、`>1ulp` 率 0.0、非有限不一致 0、对 fp64 误差 `cand == base`、verdict = EQUIV_PASS；A2 两 dtype `bitwise` 通过；O1 fp16/bf16/fp32 各 12 例 max ulp = 0）。
  用例数复算一致：A1 = 2 dtype × 3 split_k × 2 几何 × 6 模式 = 72；A2 = 2；O1 = 3 dtype × 2 split_k × 6 模式 = 36；合计 110。
  **一处观测列偏差（建议级，见维度 8 建议 1）**：A1/float16 行的「fp32 足迹」内嵌写「≤ 2」，重跑为 **3**（§11.2 正文写「≤ 3」，故文档自相矛盾）。该列在 §1.6.1 中明示为「观测项而非判据」，判据列全部复现，等价性结论不受影响。
- **design_calc_check**: `python3 .agents/tools/design_calc_check.py --design examples/shared_expert_mlp/splitk_reduce_kernel/DESIGN.md --task-dir examples/shared_expert_mlp/splitk_reduce_kernel`

  | check | status | detail | 处置 |
  |---|---|---|---|
  | `ub_budget` | **skip** | no per-level buffer tables parsed from the memory plan | **人工已核**：按 §4.3 形状逐配置复算 8 组全部与 §4.5 吻合（最坏 20 480 B = 248 KiB 的 8.1%），并用探针生成代码的 UB 偏移做字节级交叉核对（见维度 2） |
  | `l0c_budget` | **skip** | no Ascend Cube GEMM call found | **人工已核**：纯 Vector kernel，无 Cube/L0C（§5.1、§1.6.0 R4）→ 不适用 |
  | `core_split` | **skip** | insufficient parsed dims {}; declared logical cores = None | **人工已核**：按 §5.2 规则独立枚举三候选的 `ceil(grid/56)·BC` 复算，4 条 workload 取到的 `BC/grid/waves` 与 §5.2/§5.6 完全一致（见维度 3） |
  | `r3_metrics` | **pass** | all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak) | 四口径存在；算术另由维度 8 独立复算（发现 lane-ops 一列为建议级不一致） |

- **探针复核（我独立重跑，不引用设计自述）**: `history_version/design_probe_compile.py` 的 `p1 / p2 / p3 / p4a / p5` 全部 `COMPILE: OK`，生成代码与 DESIGN §11.1 的断言逐条吻合（`asc_copy_gm2ub_align(..., 1, 4096/2048, ...)` 单描述符、`for (s<3) vadd` 链、`vlds_unpack<bf16>+vcvt<float>` 的 bias 提升、`vdiv_0ulp_ftz_true`、`vcvt<half/bf16>(..., ROUND_R, RS_DISABLE, PART_EVEN)` + `vsts_pack_b32`、`p4a` 的 0 条 `vadd`、`p5` 的无 `vsts_pack_b32`、`p3` 的运行期 `min(1024,1536-c0)` 段长 + `update_mask`、`MTE2→V` 与 `V→MTE3` 各一对事件、`float2` 标量循环计数 0）。
  环境说明（供后续复现）：本机检出目录 `/workspace/tilelang-mlir-ascend-950` 缺 `build/` 且无 `tilelang/ascend`，在该目录内导入 tilelang 会走 dev-root 失败；探针须在**仓库外 cwd**（如 `/tmp`）下以 site-packages 的 tilelang 0.1.15 运行。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

检视方式：Read 源算子全文（`dense.py:1986-2053`）+ 抽取件 diff + 调用方 `shared_expert_mlp.py`，与 §0.1–0.7 逐项核对。

- **语义理解正确性 — pass（逐语句核对）**：`accum_dtype = "float"`（源 1998）↔ §0.1「fp32 累加」；`c_dtype = out_dtype or dtype` ↔ §0.1「c_dtype = out_dtype or dtype」；两处 Python 拒绝（`activation not in ("none","silu_and_mul")`、`gated and n % 2`）↔ §0.1 边界语义；`out_n = n // 2 if gated else n` ↔ §0.1/§0.2。
  **规约语义**：源 `_slice_sum`（2015-2022）为 `expr = w[0, gi, gj]; for s in range(1, split_k): expr = expr + w[s, gi, gj]; if has_bias: expr = expr + T.cast(bias[gj], accum_dtype)`，即**切片升序左结合、bias 在链尾、fp32 域**；§0.1 与 §1.3 的表述与之逐字对应，且明确「bias 在切片和之后加」「唯一舍入点在 `T.cast(..., c_dtype)`」。
  **门控**：源 `gate = _slice_sum(…, gj)`、`up = _slice_sum(…, out_n + gj)`（两半**各自**含 bias），`c[gi,gj] = T.cast(gate * T.sigmoid(gate) * up, c_dtype)` ↔ §0.1 门控分支（两半各含 bias、然后 `gate·σ(gate)·up`），无偏差。
  **边界**：`total = m*out_n` 平坦化 + 逐元素 `if idx < total` ↔ §0.1「越界元素不写」；`split_k == 1` 合法（循环 0 次）↔ §0.1/§9.3；非有限传播（`gate=-inf → -inf·0 = NaN`）与 §0.1/§9.1 R11 一致（我用源码表达式与 golden 的 `F.silu` 两路推演确认同为 NaN）。
- **I/O 契约 — pass**：源 `_main("_splitk_reduce", body, (((split_k, m, n), accum_dtype),), ((m, out_n), c_dtype), ((n,), dtype) if has_bias else None)` ↔ §0.2 表（w fp32 `[split_k,m,n]`、bias 输入 dtype 仅 has_bias 时出现、C `[m,out_n]` = `out_dtype or dtype`）；`C` 是参数而非 `out_idx` 输出（源 docstring「the caller allocates it before the mainloop」）↔ §0.1/§4.7 `out_idx=[]`。
  调用方实证：`shared_expert_mlp.py:289` 为 `splitk_reduce_kernel(split_k, m, n, self.dtype_str, activation)(_SPLIT_K_ELEMS_PER_CTA)`（**has_bias / out_dtype 取默认**），`:296` 为 `out = torch.empty((m, out_n), dtype=self.dtype)` → 印证 §0.2「本 op 中 has_bias=False、out_dtype=None」；`_split_k`（`:252-266`）逻辑为「`num_tokens > 32` 或 `k_tiles < 2` → 1，否则取 ≤ 4 的最大 K 整除因子」↔ §0.2 的 8 行 workload 表（我按 `k = hidden_size / ffn_size` 复算 4 条真实调用均得 split_k = 4）。
- **算法解读完整性 — pass**：§0.3 的 8 步表覆盖源全部计算语句（`base`、`gi/gj`、`idx<total`、`_slice_sum`、bias、门控第二流、`T.sigmoid`、`T.cast` 存），无遗漏步骤、无臆造步骤；数据流图覆盖全部 buffer（w/bias/C，无片上复用中间态）；host 侧逻辑（两种 `global_symbol` 签名装配、无 `out_idx`、编译期校验）已列出。
- **优化手段覆盖 — pass**：§0.4 O1–O10 覆盖源码显式优化并逐项标注硬件依赖：扁平 1-D 映射（源 2030-2034）、`elems_per_cta=1024`/`threads=256`、`C` 预分配（docstring）、fp32 单累加器 + 单次 cast、切片升序、门控同 kernel、`pass_configs={TL_ENABLE_FAST_MATH: gated}`（源 2010-2013）、`T.sigmoid` 内建、L2 驻留（docstring「the workspace of the shapes worth splitting fits in L2」）、`-O3 -DENABLE_BF16`。未遗漏主要手段，也未把实现层手段（O1/O2）误标为算法层。
- **耦合性判定合理性 — pass**：§0.5 十项每项有 语义层/硬件强相关/可用/四态处置/依据。抽检三条并亲自查证依据：① O7（fast-math）判「舍弃」，依据是平台 `vexp` 单指令 + `vdiv` 默认 `vdiv_0ulp_ftz_true` → 我用探针 p2 生成代码确证（`simd_inst::vexp`、`simd_inst::vdiv_0ulp_ftz_true`）；② C3/跨 block 融合淘汰引 `references/programming.md:73`「手工跨核 flag、循环复用下的复杂同步和跨 block batch barrier 仍未认证」→ 原文核对一致；③ C2 原子归约淘汰引 `knowledge/execution-patterns.md:PL-950-atomic-splitk`「不保证同一 kernel 内某个消费者可以在两个核写完前读取最终值」+ 需调用方预清零 + 仅 950DT 2 核用例 → 原文核对一致。统计行（保留 4 / 等价替换 3 / 重新设计 1 / 舍弃 2）与表体一致。O1 判「NPU lane 必须落连续列」与 `simd.py` 的 `vld/vsts` 以 `access_ptr`+固定向量宽度表达一致，可与 §1.6.3 D2/D3 的量化理由互证。
- **NPU 重设计可行性 — pass**：R1–R6 每项含「源方案 → NPU 新算法 → 语义保持论证」。新算法满足平台约束：一维 `T.Kernel(grid)`（`programming.md:13`「多维网格在实现中拒绝」）、`T.SimdVF` 内不访问 GM、跨迭代向量用 `T.simd.alloc_local`、`vcvt` 的 `part` 为构建期常量、无 f32→f32 的 `vcvt`（`simd.py:1265-1330` 的转换表不含该对）、`vcvt(..., part=0)` 一次落盘。R3 明确「累加分组与顺序与源一致 → 逐位相同」，未引入新的舍入分组；R5 明确「唯一舍入点仍在写回」「`ROUND_R` = 就近向偶」（`simd.py` docstring：「`ROUND_R` (default): Round to nearest, ties to even (banker's rounding)」——我独立查证，与 §0.6 R5 的「与 torch `.to()` 的 RNE 一致」相符）。R1–R6 的语义保持另由 §8.1 golden（§0.1 语义独立实现）+ §8.3 L0 分层覆盖兜底，论证链闭合。
- **§1–§7 与迁移决策一致性 — pass**：§1.4（6 步）、§3.1（10 步公式拆解，输入为 §1.6.1 优化后公式）、§3.3 伪代码、§4（UB 列块 `[split_k, BC]` + 输出列块）、§5.2（1 行 × BC 列）、§6.1（块级映射 + DMA 序列 + chunk 循环）描述的都是 §0.5/§0.6 决策后的 NPU 算法；全文无「§0 判重设计、§3 照抄源方案」的脱节（源侧的逐元素 `//`/`%`、`idx<total` 在 §3.3 已消失，与 §1.6.2 V3/V4 一致）。
- **golden 独立性 — pass**：§8.1 以 §0.1 语义独立实现（`acc = w[0] + w[1] + …` 升序左加、bias 在链尾、门控 `F.silu(gate) * up`、末次 `.to(out_dtype or dtype)`），不含 tile 切分/UB 缓冲/向量指令痕迹，语义与源逐项一致（我按源 `_slice_sum` 与 `T.sigmoid` 表达式逐步对照，无偏差）。§8.2 容差表引 `/workspace/TileOPs/workloads/numerics.py:reference_tolerance` 的 fp16 1e-3 / bf16 1.6e-2 / fp32 1e-5（`rtol = atol`）——已核对该文件实际值一致。

### 1. API 可行性 — pass

以实际语言模块 `tilelang.ascend.language`（site-packages，tilelang 0.1.15）核对导出与签名，并以**自己重跑的编译探针**作证据（不引用设计自述）：

- 导出面：`T.simd` 含 `vld / vsts / vadd / vmul / vdiv / vexp / vneg / vcvt / pset / update_mask / vdup / alloc_local / mem_bar`（`len(dir(T.simd)) = 100+`），`T.SimdVF` / `T.annotate_buffer_versions` / `T.Kernel` 均存在 → §3.2/§3.4 的 API 全部有导出佐证。
- 签名与参数：`vcvt(src, target_dtype, mask=None, round="ROUND_R", sat=True, part=0, mode="MODE_ZEROING")` —— 设计固定 `round="ROUND_R", sat=False, part=0` 合法；`sat` 语义 docstring 为「`True` 饱和到目标范围 / `False` 不饱和」，与 R5「`sat=False` 保持 ±inf」的设计意图一致（数值后果另由 L0-9 兜底）。`vsts(addr, src, mask=None, dist="NORM_B32", extent=None)`；`vld(addr, dist="NORM", *, post_inc=None)`，`UNPK_*` 前缀的 footprint 取 `lanes//2` → §3.2 的 `vld(b_ub[…], dist="UNPK_B16")` 与 `vsts(…, dist="PK_B32")` 用法成立。
- 关键负向事实：`vcvt` 的 Float→Float 分支只列 `f32→f16/bf16/fp8`，**无 f32→f32** → 设计的 E4/P5（fp32 输出跳过 vcvt，改 `vsts(dist="NORM_B32")`）有源码依据；我在 p5 重跑里复现了「无 `vsts_pack_b32`、`asc_copy_ub2gm_align(..., 1, 4096, ...)`」。
- 编译证据：p1/p2/p3/p4a/p5 全 `COMPILE: OK`，签名 `extern "C" __global__ __vector__ void main_kernel(...)` + `__simd_vf__ inline void main_kernel_simd_vf_0(...)`（纯 AIV 发射），与 §2.2 第 3 点一致。
- 与模式匹配：§2 选 SIMD、与平台 `programming_modes.migration_default = simd` 一致；无 Cube / 无 `T.Parallel` / 无手工 flag，符合 `references/programming.md` 的 SIMD 区域规则。
- 建议级备注（不阻塞）：§3.4 表格末行 `T.annotate_compile_flags(["-O3"])` 标注「可选 / 未使用」——该接口在本平台导出（`language/eager/builder.py:1220`），保留为 Stage 4 旋钮即可。

### 2. 内存层级规划 — pass

- **搬运路径完整**：§4.4 给出 GM→UB（MTE2，`asc_copy_gm2ub_align`）→ UB 内向量处理 → UB→GM（MTE3，`asc_copy_ub2gm_align`）的全链路，无 L1/L0A/L0B/L0C 参与（纯 Vector），与 §2.3 的模式影响一致。
- **Buffer 层级标注**：§4.3 四个 buffer 全部标 UB 且给版本数 1；`T.alloc_shared` 落 `buf_dyn_shmem`（探针生成代码证实）。
- **UB 预算（机械 skip → 人工复核）**：逐配置复算与 §4.5 全部一致——BC=256 门控 4096+4096+0+512 = 8 704 B；+bias 9 728 B；BC=1024 非门控 16 384+2 048 = 18 432 B；+bias **20 480 B（最坏，8.1%）**；BC=512 门控 17 408 / 19 456 B；退化 7 168 B。×1.7 上界 34 816 B = 13.7%（`TRAP-UB-multibuffer-inflation` 条目存在，verified）。
- **字节级交叉核对（我做的独立验证）**：探针 p1（BC=1024 + fp16 out + bias）生成代码中 `g_ub@float[0]`、`bias@half[8192]`（= 16 384 B）、`o_ub@half[9216]`（= 18 432 B）→ 总占用 18 432+2 048 = **20 480 B**，与 §4.5「ds down 非门控 + bias」逐字节相同；探针 p2（BC=512 bf16 门控 + bias）`g@float[0..2048)`、`u@float[2048..4096)`（= 16 384 B）、`bias@half[8192/8704]`、`o@half[9216]` → **19 456 B**，与 §4.5「glm gate/up 门控 + bias」逐字节相同。说明编译器未插入隐藏缓冲、UB 预算可静态核算。
- 建议级备注（见建议 4）：§4.5 把上述可静态核算归因于 `annotate_buffer_versions({…: 1})`，但 `history_version/design_probe_compile.py` 并未调用该 API，而我的重跑（无注解）已与表逐字节吻合——归因需订正。
- L0C/L1：不涉及，§4.5/§3.5.1 已声明；`l0c_budget` 机械 skip 的原因（无 Cube 调用）成立。

### 3. Tiling 策略 — pass

- **Block 划分明确**：§5.2 给出自由变量 `BC` 的构建期规则（`argmin_{BC∈{256,512,1024}} ceil(m·ceil(N/BC)/56)·BC`，并列取大）+ 每 workload 的 `BC/nc/grid/nch`。
- **对齐约束**：`BC` 全为 64（= 1 个 fp32 向量寄存器）的整数倍；4 条真实 workload 的 `N ∈ {2048, 7168, 1536, 5120}` 被选中的 `BC` 整除 → 真实路径无尾块。
- **非整除处理**：`bc = T.min(BC, N-c0)` 运行期 region + `update_mask`；探针 p3 复现（DMA 段长 `min(1024, 1536-c0)*4`，`update_mask` 1 处）→ 可编译。该形态与平台已验条目 `PL-950-vector-execution`（fp32 N=256/257「按有效元素数搬入、写回 GM」）同构，故运行期长度非 32 B 整数倍的顾虑不成立。
- **分核策略三要素（机械 skip → 人工复核）**：
  ① **逻辑核数**：`grid = m × ceil(N/BC)` = 256 / 224 / 96 / 160；启动核数 = grid（无核内多任务串行）。
  ② **物理核数依据（实查）**：§5.5 记录了 `statectl platform --task-dir` 的返回（`observed_cores = {cube: 28, vector: 56}`、`core_source` 含 runtime `rtGetAiCoreCount()=28`）与 ini 查询；我复跑 `grep -n "core_cnt|l0_c_size|l1_size|ub_size" /usr/local/Ascend/cann-9.2.1/x86_64-linux/data/platform_config/Ascend950PR_9579.ini` → 第 20–22 行 `ai_core_cnt=28 / cube_core_cnt=28 / vector_core_cnt=56`、第 50/51/58 行 `l0_c_size=262144 / l1_size=524288 / ub_size=253952`，与 §5.5 表格**逐项一致**。未跑 `torch.npu.get_device_properties` 的原因（单卡被同批 Stage 3 占用）已写明并给出 Stage 3 补测要求，符合 `core-split-strategy.md` §1② 的实查记录要求。
  ③ **规模判定与分核方案**：中等规模，波次 2/3/4/5，波内利用率 85.7% / 95.2% / 100% / 91.4%；不做核内串行的依据（块负载完全均匀 + 平台无 `T.Persistent` 正确性条目）成立。
- **自洽性（我独立复算）**：按 §5.2 规则枚举三候选的 `ceil(grid/56)·BC`——ds gate/up `1280 / 1536 / 2048 → 256`；ds down `4096 / 4096 / 4096 → 并列取大 1024`；glm gate/up `1024 / 1024 / 2048 → 并列取大 512`；glm down `3072 / 3072 / 3072 → 1024`，与 §5.2/§5.6 完全一致。`nch`、每块 DMA 条数、每块向量指令数（非门控 9、门控 21）与总指令数（21 504 / 32 256 / 16 128 / 23 040，每核 384/576/288/412）亦复算一致。

### 4. 技术约束检测 — pass

- **三维 Kernel**：不涉及（一维 grid，`programming.md:13`「多维网格在实现中拒绝」）；§9.2 E6 已记为陷阱。
- **GEMM 非整除**：不涉及（无 Cube/无 tile 前置）；§5.4 显式说明「本 kernel 沿切片轴只做顺序读取，不要求 `split_k` 整除任何量」，与上游 GEMM 的「`split_k | k_tiles`」前置区分正确。
- **L0C 溢出**：不涉及。
- §3.5.1 的已知限制检查表齐全且逐条给处理方案（SimdVF 禁 GM、`alloc_local`、`vcvt part` 常量、无 f32→f32、块级循环上界静态、`T.copy` 显式 region、UB 容量 + 多版本膨胀、无跨核同步、不用 `annotate_unlimit_memory`、无手工 flag/`mem_bar`、不开 fast-math）；其中「静态上界」「显式 region」「无 f32→f32」三项由我的探针重跑直接证实。

### 5. 循环与同步 — pass（含 1 项写后读专项判断）

- **循环结构**：§6.1 明确列出 核网格（`T.Kernel`）/ 块索引分解（块级标量）/ 切片装载（`T.serial(split_k)`，每迭代 1 条 DMA）/ 列 chunk（`T.serial(nch)` + 向量指令）/ 切片累加链（静态展开）/ silu（无循环），并显式声明**不用** `T.Parallel`、`T.Pipelined`、`T.Persistent` 及理由，与 §1.6.2 V12/V13/V14 的保留结论一致。
- **同步与模式匹配**：§7 选自动同步，不写任何 `ascend_set_flag/wait_flag`、不用 `T.simd.mem_bar`。生成代码证实编译器插入了 `asc_sync_notify/wait(PIPE_MTE2→PIPE_V)` 与 `(PIPE_V→PIPE_MTE3)` 各一对，与 §7.2 表格一致。
- **写后读（WAR / VST→VLD）专项判断 — 结论：确实不存在该路径，设计主张成立**。证据三条：
  1. 代码事实：p1/p2/p3 生成代码里 `o_ub` 只被 `vsts_pack_b32` 写、只被 `asc_copy_ub2gm_align`（MTE3 DMA）读，且读出前有 `asc_sync_notify(PIPE_V, PIPE_MTE3)`；`g_ub/u_ub/b_ub` 只被 MTE2 写、只被 `vld` 读，其前有 `PIPE_MTE2→PIPE_V` 事件。全程**没有**对同一 UB buffer 的「向量写 → 向量读」。
  2. 循环结构：chunk 循环各迭代写 `o_ub[ch·64 …)` 互不重叠，且循环体内无任何 `vld` 读 `o_ub`（`acc` 是 `alloc_local` 的寄存器，不用 UB）。
  3. 对照 engram 反例：`examples/engram_gate_conv/.../repro/TRAP-950-simd-ub-store-load.py` 的规则限定为「vsts 后再 **vld 同一 UB 缓冲**必须 `T.simd.mem_bar("VST_VLD")`」，其**断言 A 正是「`PK_B32` 存 + DMA 拷出逐字节正确、无需 barrier」**——即本 kernel 的写回形态落在该 repro 的正向结论内，不属 engram 踩到的竞态类别。§3.5.1/§7.2 的引用准确。
- **尾块处理**：§6.4 表覆盖 `N % BC == 0`、`N % BC != 0`、`out_n < BC`、`m` 小、`split_k` 与形状无整除关系五种情形，与 §5.4 一致；p3 生成代码证实运行期 `bc` 与 `update_mask`。

### 6. 验证方案 — pass

- **Golden 函数**：§8.1 给出可执行的 PyTorch 参考实现草案（签名、参数语义、累加顺序、bias 位置、门控分支、一次舍入），并声明独立性口径；实现正确性我逐项对照源语义确认（见维度 0）。
- **L0 测试计划**：§8.3 九例覆盖——真实几何（L0-1/2/3）+ fp16 输出（L0-4）+ 退化路径（L0-5 `split_k=1` 门控、L0-6 `split_k=1`+bias）+ 尾块与尾 lane（L0-7 `N=2000`，`2000 % 256 = 208`、`208 % 64 = 16`，确实触发两条尾路径）+ `out_dtype="float32"` + 门控 bias（L0-8）+ 非有限/次正规/边界值（L0-9）。我按 §5.2 规则复算 9 例的 `BC`（256/1024/512/256/256/256/256/256/256）全部与表一致，说明用例与 Tiling 自洽。
- **精度标准**：§8.2 按 dtype 给 atol/rtol（fp16 1e-3、bf16 1.6e-2、fp32 1e-5）并与 `TileOPs/workloads/numerics.py:reference_tolerance` 实际值一致（已核对该文件 273-283 行）；非有限值用 `equal_nan=True` + 位置模式一致。另给出「期望 ≤ 1 ulp」的附加判据与容差裕度说明（fp16 2 ulp、bf16 4 ulp），可用于回归检测，设计合理。
- 门槛定义（L0-1…L0-9 全过 = `[PRECISION_PASS]` 最低要求；L0-1/2 另记时延作 Stage 4 基线）清晰可执行。

### 7. 完整性与一致性 — pass

- **章节齐全**：§0（0.1–0.7 齐全）+ §1–§11 齐全；§1.6 含 1.6.0/1.6.1/1.6.2/1.6.3 四子节。
- **占位符检查**：全文无待补标记、无模板占位符；§3.3 的 `T.annotate_buffer_versions({...: 1})` 是伪代码省略写法（§3.2 已给出完整键集 `{g_ub: 1, u_ub: 1, b_ub: 1, o_ub: 1}`），不构成缺口。
- **同类引用真实**：§0.7/§3.5.3/附录引用的 18 个路径我逐个 stat，全部存在（`engram_gate_conv` 三个文件、`mla_dense_bwd/_delta_kernel/DESIGN.md`、`moe_permute_align/_make_scatter_kernel/DESIGN.md`、`examples/TileOPs/...`、`/workspace/TileOPs-ascend950/.../abs.py`、平台 knowledge 五条、`constants.md`、`kernel_template.py`、`ascend-950pr-9579-notes.md`、`programming.md`）；`references/ascend-constraints.md` 存在（位于 `tilelang-op-design/references/`）。
- **内部一致**：API 映射（§3.2）↔ 伪代码（§3.3）↔ 内存规划（§4.3/§4.5）↔ 循环表（§6.1）四处互证，且与探针生成代码的 UB 偏移、DMA 描述符、指令形态逐项吻合；上游契约（`_gemm_basic_kernel` §4.2 的 `[split_k, m, n]` fp32 + §9.2 E6）与本设计 §0.1/§4.1/§9.1 R6/R7 双向引用一致；调用方 `_decompose`（`shared_expert_mlp.py:233-250`）在 `gate_up_split > 1` 时确实把 `silu_and_mul` 融进 reduce（`activated=True`，独立激活 kernel 消失），与 §设计范围/§0.2 的陈述一致。
- 建议级备注（合并为建议 5）：§5.4 行标签写作「门控 `N` 为奇数」（应为 `n`，§9.1 R5 正确）；§1.6.0 称候选库「六族」而 `knowledge/algorithm-candidates.md` 实为 7 行；`kb_search` 实际返回 `top 1 of 2 entries`（第二命中未列）；§1.6.0 R3 的 C1 中间缓冲公式「`2·split_k·BC·4`」对非门控（单流）不成立（表内数值 16 KiB 是对的）。均为文字/口径精度问题，不影响任何设计判断。

### 8. 算法优化分析 — pass（含 3 项建议）

**§1.6 章节存在性** — pass：1.6.0/1.6.1/1.6.2/1.6.3 四子节齐备，且选定算法（C1 + A1 + A2）实际驱动 §1.4/§3.1/§6（见下「调研→设计一致性」）。

**算法调研存在性与完整性** — pass：
- R1 候选表 11 项（C0 基线 + C1–C10），含基线、等价性初判、收益方向、是否入 R3 列；
- R2 有**明确结论**（「有，且基线本身已是在线形态」）并给结构依据（每输出只依赖同位置 `split_k` 个输入；无 softmax max/sum、无 Welford 等可重缩放统计量；状态 O(tile)；源本身单遍），并显式把「跨 launch 的 GEMM 分片 + 归约」排除在本口径外；
- R3 四口径齐全（lane-ops / 访存 Bytes / 扫描遍数 / 中间缓冲峰值），**基线在表**，另加「可并行度 / 跨核代价」列，口径说明写清（lane-op 定义、访存按核实际 DMA 字节、遍数按每输入字节被读的 GM 次数、缓冲峰值指需额外物化量）；
- R4 逐候选六维评估（计算单元匹配 / 片上容量 / 对齐整除 / 静态边界 / 流水融合 / 跨核结构），负向淘汰均附依据；
- 调研深度与算子复杂度匹配（有完整候选表 + 四口径 + 逐候选亲和表 + 设计期 roofline）。

**调研结论正确性（独立复核）** — pass，逐条：
1. **等价性**：选定算法 C1 与原语义同构（同式、同分组、同顺序、同舍入点），A1 的等价性见下「等价性论证正确性」独立推演 → 成立。
2. **复杂度算术复算**：
   - 访存量：ds gate/up `4·32·4096·4 = 2 097 152 B + 32·2048·2 = 131 072 B = 2.228 MB ≈ 2.23 MB` ✓；ds down `3 670 016 + 458 752 = 4.13 MB` ✓；C2 `5.37 / 9.63 MB`（= 2.33–2.4×，与「2.3–2.4×」一致）✓；C4 `2.75 MB`（+23%，与「+24%」同位）✓；C3 只剩 out `0.13 / 0.46 MB` ✓。
   - 中间缓冲：C1 = 8 KiB（门控 BC=256，双流）/ 16 KiB（非门控 BC=1024，单流）→ 与生成代码 UB 占用一致 ✓；**本设计无任何 GM 中间往返**（C2/C4 的 `m·n·4` 物化被正确计入并淘汰）✓。
   - 扫描遍数：C1 = 1（每字节一次）✓；C2 = 2、C4 = 2 ✓。
   - D-2 roofline：`2.23/1.03 = 2.17 µs`、`4.13/1.03 = 4.01`、`2.23/2.41 = 0.93`、`4.13/2.41 = 1.71`、全部调用范围 `1.67–4.13 MB`（glm gate/up 1.573+0.098 = 1.67 ✓）、`1.62–4.01 µs`、`0.69–1.71 µs`、发射项 `384/1.8e9 = 0.213 µs` / `576 → 0.32 µs`、容量项 3.8%/7.3% —— 逐项复算一致 ✓。
   - **一处不一致（建议 3）**：R3 表 lane-ops 列的「gate/up = 0.92 M」与同表 down = 2.06 M 的口径不自洽——按 §5.6 自己的每 chunk 指令数（门控 21 / 非门控 9），lane-ops 应为 `65 536 × 21 = 1.38 M`（down `229 376 × 9 = 2.06 M` 与该口径吻合）；0.92 M 隐含 14 op/chunk，与 §1.6.1 A1 的「21 → 20（−4.8%）」互相矛盾。结论不受影响（AI 由 0.41 变 0.62，仍远低于任何机器平衡点，仍纯带宽受限）。
3. **亲和性判断**：与 `programming.md:73`（跨 block barrier 未认证）、`PL-950-atomic-splitk`（950DT 2 核 / 不保证同 kernel 内读到最终值 / 需预清零）、`CG-950-simd-shared-reduce`（二维高层 shared→shared 归约输出错误，已验替代路径 = 显式寄存器归约）逐条核对，设计的使用/淘汰结论与这些条目**不矛盾**，且「本设计不用 `T.reduce_sum` 任何二维形式、归约走 64 lane 内 `vadd`」正好落在该缺口给出的替代路径上。
4. **负向断言主动复核**：
   - 「候选库无 split-K 归约 epilogue 专族」→ 我读 `knowledge/algorithm-candidates.md` 全表（elementwise/broadcast、reduction、softmax、RMSNorm/LayerNorm、GEMM、attention、GEMM+ReLU 共 7 行），确实无该族 ✓；
   - 最接近的 `reduction`（单行 SIMT sum/max reducer）与 `elementwise/broadcast` 行是否被漏掉？—— 已由 §1.6.3 候选 2/D2（lane = 行，横向归约映射）与 §3.4（不用 `T.reduce_sum`）+ §1.6.0 C7（lane 上放切片 + `vcadd`）覆盖，**未漏** ✓；
   - 「不存在能同时减少读部分和 + 写 out 字节数的算法族」→ 几何下界论证成立（每字节至少读一次、输出至少写一次），C2/C4 的额外物化只会更多 ✓；
   - 我复跑 `python3 .agents/tools/kb_search.py "split-k reduction epilogue silu_and_mul vector reduce fp32 partial"` → top1 = `CG-950-simd-shared-reduce`（score 0.55，「top 1 of 2 entries」）→ 与设计「命中 1 条」基本一致（第二命中未列，建议级）。
5. **并列候选裁决路径**：`BC` 存在三候选并列的情形（ds down / glm up / glm down），设计**未纸面拍板**，而是进 §1.6.3 的实验裁决三件套（见下），符合要求。

**调研→设计一致性** — pass：§1.4 的 6 步、§3.1 的 10 步、§6 的循环表全部按 C1 + A1 + A2 的形态描述（块级行列映射、双列流、`(g/(1+e^{-g}))·u`、构建期专用化）；§0.5/§0.6 的处置与调研选族一致；**源算法 10 项优化手段逐项有承接说明**（O1→R1 块级映射（意图「一个 grid 覆盖任意 shape」由 `nc` 达成）、O2→`1×BC`、O3/O4/O5/O6→保留、O7→舍弃并给本平台替代、O8→展开、O9→保留为假设并在 D-2 双口径使用、O10→保留 `-O3`），无静默丢弃。

**数学等价优化完整性（§1.6.1 四要素）** — pass：
- **A1**：原式 `y = g·σ(g)·u`（取倒数再乘，两处 fp32 舍入）→ 优化后 `den = 1+e^{-g}`、`y = (g/den)·u`（一处舍入）；等价性论证（实数恒等 + fp32 舍入位置差异 + 输出 dtype 0 ulp）；收益量化（每 chunk 少 1 条 64-lane `vmul`，门控 21→20 = −4.8%，4 次调用合计少 1 792 条）——四要素齐全 ✓。
- **A2**：明确标为「结构性不变量 / 回归守卫」而非相对源的优化（因为源在 Python 工厂里本来就专用化），四要素结构相同但收益栏写「相对源为 0」——这个自我限定是准确的（源 `_slice_sum` 是 Python 循环，`if has_bias` 也是构建期判定）✓。
- 无进一步优化空间的结论有六条具体依据（加法次数下界、流量下界、舍入点下界、bias 必须先加、silu 只剩 1 `vexp`+1 `vdiv`、门控两链无公共子表达式）✓。

**等价性论证正确性（独立推演）** — pass：
- `g·(1/den)·u` 与 `(g/den)·u` 在实数域恒等成立：`den = 1 + e^{-g}`，对任意有限 `g` 有 `den ∈ (1, +∞)`，永不为 0，除法无定义点不存在（`g = -inf` 时 `den = inf`、`g/den = -inf/inf = NaN`，与源的 `-inf·σ(-inf) = -inf·0 = NaN` 同为 NaN；`g = +inf` 时 `den = 1`、结果 `±inf`，两式同）。前者的 fp32 固有差异只在「先取倒数再乘」vs「直接除」的舍入位置，属容差内等价，且**输出 dtype 网格上的差异由机器验证为 0 ulp**（我重跑确认 110 例 0 失败）。
- 设计对 `sat=False` 的语义依赖（保持 ±inf 而不夹到有限最大值）与 `vcvt` docstring 的 `sat` 语义一致，并有 L0-9 + §9.1 R3 的兜底与回退阶梯（退回「取倒数再乘」只多 1 条 `vmul`）。
- A2 的等价性：删去 0 次迭代的循环、不发射的 DMA、不求值的表达式，逐位无语义影响——平凡成立（但机器验证本身是自比较，见建议 2）。

**向量化替代完整性（与 §3.3/§6 交叉核对）** — pass：§1.6.2 列 15 个计算点，V1–V11 全部替代或整体消除（V3/V4 被块级映射消除、V5–V10 落 `T.simd.*`、V11 落 DMA），V12–V15 保留并给具体理由。与 §3.3 伪代码逐行对照无遗漏（伪代码里不存在逐元素标量循环）；与 §6.1 循环表交叉核对一致（核网格/块索引/切片装载/chunk/累加链/silu 六类全部落 V14/V12/V13/V2/V7）。生成代码的 `float2` 标量循环计数在我的 5 个探针重跑中均为 **0**，与结论互证。

**不可替代理由充分性** — pass：4 项保留的理由都具体到本算子且属可接受类别——V12 是「每迭代 1 条 DMA 的块级搬运序列」（非标量计算，展开/流水化是 Stage 4 旋钮）；V13 就是向量化本身（`BC/64` 次迭代 = 若干向量寄存器宽）；V14 是 block 级索引（每块 3 次整数运算，生成代码折成移位/掩码/取模）；V15 是 host 侧元数据（构建期判定，不进 kernel）。无「实现简单 / 照源码写」式空话。

**替代 API 可行性** — pass：与维度 1 同口径，全部有实际模块导出 + 我重跑的可编译证据（`vlds_norm/vlds_unpack/vcvt/vadd/vdiv_0ulp_ftz_true/vexp/vneg/vsts_pack_b32/vdup`），未使用不存在的向量 API；`T.sigmoid`（无 SIMD 证据）与 `T.reduce_sum`（`CG-950-simd-shared-reduce` 缺口）被明确列为「不使用」，处置正确。

**向量化轴与布局决策完整性（§1.6.3）** — pass：
- 类别候选矩阵按「逐元素/广播类 + 一个切片规约轴」枚举：D1（列轴，主选）、D2（行轴）、D3（扁平轴 = 源形态）、D4（两块分工 gate/up）、D5（切片轴放 lane），并给 7 维评分（整除性 / 尾 lane 浪费率 / 累加链形态 / repack 代价 / 广播轴对齐 / UB 影响）。
- 规约维的两种 lane 映射都评了：垂直扫描（D1，lane 持独立输出、切片维串行链）vs 水平归约（D5/C7，`vcadd` 跨 lane）——未漏。
- 弃选均有量化理由（D2：lane 只用 50% + bias 变跨步 gather + `m=32` 填不满 56 核；D3：恢复逐元素 `//`/`%` + 写回拆两段；D4：块数翻倍无收益；D5：4/64 lane 有效 = 94% 浪费 + 每 chunk 一次 `vcadd`）。**未把 I/O 契约当内部布局依据**：D1 的采纳理由是「`[S,m,n]` 行主序下 `w[s,row,c0:c0+bc]` 本身就是连续 `4·bc` 字节 → 单描述符 DMA」，我用 p1 生成的 `asc_copy_gm2ub_align(..., 1, 4096, ..., 4096, 4096)` 确证这是量化事实而非契约推论。
- 辅决策 S1–S4 逐项给采纳/弃选依据（S1 全切片驻留 vs 逐切片 `vsts`→`vld` 回读（并正确指出后者会引入 VST→VLD 依赖需 `mem_bar`）；S2 流水进 Stage 4；S3 写回必须经 UB；S4 `BC` 按规则绑定）。
- **实验裁决三件套**：主选（`BC` 规则 + §3.3 结构）+ 备选（统一 `BC=512`，我复算其分核三要素：`grid = 128/448/96/320` → 3/8/2/6 波、UB `2·4·512·4 + 2·512·2 + 512·2 = 19 456 B`、4 条 workload 的 `N` 全被 512 整除故无尾块代码）+ 裁决计划（4 个代表 shape、`msopprof` Task Duration A/B ≥5 次交错、**阈值「备选 < 主选 × 0.9 才翻转」**、按 shape 分派可行性评估、结果回写 §9.1）——三件套齐全且备选达到可直接实现深度 ✓。
- 判定所依赖的未实测常数（每块固定开销 `f`）确实未在平台 `cost_measurements`（`[]`，我已读 statectl 返回值）或 pattern-library 中实测（`CONST-vector-launch-overhead` / `CONST-mte2-degradation` 均绑定 910B2C + tilelang 0.1.2，我核对过 constants.md 的 12 条目 toolchain 字段），故「进实验裁决而非纸面单选」的处置正确。

**布局决策与实现一致** — pass：§1.6.3 选定 D1（核内 `[split_k, BC]` fp32 列块 + 门控第二同形流 + `[BC]` bias/输出块，向量化轴 = 列 `j`，repack = 无）；§3.3 伪代码的 `g_ub[s, ch*VL]`、§4.3 的 buffer 形状、§6 的「内层向量维 = 列」三方一致；§4.5 的 UB 预算按该布局计算，且与我重跑生成代码的字节偏移吻合。

**与下游章节一致** — pass：§3.1 明示「输入为 §1.6.1 的优化后公式（A1 + A2 已采纳）」，其步骤 8 就是 `y = (g/(1+e^{-g}))·u`；§6 未出现 §1.6.2 判定已向量化却仍保留的逐元素标量循环。

**建议级（不阻塞，建议在 §11.4 或下次 revision 一并订正）**：
1. §1.6.1 表 A1/float16 行的「fp32 足迹 ≤ 2 ulp」与脚本重跑（3）、§11.2 正文（≤ 3）不一致 → 订正为「≤ 3」。
2. `verify_equiv.py` 的 A2 检查是**自比较**（`generic` 与 `spec` 为同一表达式 `(w + bias.float()).to(dtype)`，`copy_gen` 与 `copy_spec` 同为 `w.to(dtype)`），未构成「无条件发射 vs 专用化」的证据 → 改为真正回放两版构建，或把该行降级为「人工推演：删除空操作，平凡成立」，避免给 `[EQUIV_ALL_PASS]` 提供虚假支撑。
3. §1.6.0 R3 的 lane-ops「gate/up = 0.92 M」与 §5.6 / §1.6.1 的每 chunk 指令数口径不自洽（同口径应为 1.38 M）→ 订正数字并同步 §1.6.0「AI ≈ 0.41」为 0.62（结论「纯带宽受限」不变）。
4. §4.5 把 UB 预算可静态核算归因于 `annotate_buffer_versions({…: 1})`，而 `design_probe_compile.py` 并未调用该 API（我的无注解答重跑已与表逐字节吻合）→ 或在探针中补上该注解并重跑，或改写为「默认无多版本 + 注解锁定」。
5. 文字/口径精度：§5.4 行标签「门控 `N` 为奇数」应为 `n`；§1.6.0「六族」应为 7 行；`kb_search` 实为 top-1 of 2（第二命中未列）；§1.6.0 R3 的 C1 中间缓冲公式「`2·split_k·BC·4`」对非门控（单流）不适用（表内 16 KiB 数值正确）。
6. §9.1 R7 写「上游 GEMM 报错」，而 `_gemm_basic_kernel` §9.2 E6 允许「kernel 内强制 `split_k=1`」或「调用方置 1」两条等价路径 → 建议统一表述为「上游 kernel 内强制或调用方回落，两条等价」，避免 Stage 5 只改一侧。
