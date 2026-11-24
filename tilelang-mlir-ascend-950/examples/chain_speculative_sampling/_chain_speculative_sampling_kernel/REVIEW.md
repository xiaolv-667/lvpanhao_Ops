# _chain_speculative_sampling_kernel 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息

- 文档: `examples/chain_speculative_sampling/_chain_speculative_sampling_kernel/DESIGN.md`
- 任务类型: 迁移（TileOPs 逐函数迁移，GPU `ChainSpeculativeSamplingFwdKernel` → ascend950PR）
- 源算子: `/workspace/TileOPs/src/tileops/_kernels/sampling/chain_speculative_sampling.py`（514 行，已全文阅读）
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 10
- 平台: `ascend-950pr-9579` / `ascend`（CANN 9.2.1，dav-3510，28 Cube / 56 Vector；UB 253952 B）
- 独立核对材料: `examples/TileOPs/tileops/workloads/sampling.py`（golden + 契约校验器）、
  `examples/TileOPs/tests/ops/test_sampling.py`、`examples/TileOPs/tileops/manifest/sampling.yaml`、
  `examples/TileOPs/benchmarks/ops/bench_chain_speculative_sampling.py`、
  `knowledge/{algorithm-candidates,capability-gaps,PL-950-row-primitives}.md`、
  `references/programming.md`、`history_version/probe_p*_gen.txt`、`verify_equiv.py`

## 机械复核（D-1/D-5）

- **verify_equiv 重跑**: `PASS` —— 在任务目录下
  `source .agents/platforms/toolchains/ascend/env-950pr.sh && /workspace/venv-ascend950/bin/python verify_equiv.py`，
  15/15 项 `EQUIV_PASS`（33.7 s），与 §1.6.1 内嵌结果表**逐项一致**：
  E1 `0/160072`；E2 `bit-mismatch=0`（256×9 / 1×1 / 2×2）+ 五个 seed/offset 角点各 `0/36`；
  E3 `0/92`；E4 `0/92`；E6 `0/92`；E5 `0/68`、zero-weight picks `0`、相对偏差 `4.845e-08`、
  分布 `max |Δcount|=0, bound=77`。
- **design_calc_check**（`python3 .agents/tools/design_calc_check.py --design … --task-dir …`）:
  - `ub_budget`: **skip**（"no per-level buffer tables parsed"）→ **人工已核**：§4.5 复算
    `32768+1024+76+32+80 = 33980 B`，`×1.7 = 57766 B = 253952 B 的 22.7%`（`N=600` 时
    36348 / 61792 / 24.3%）；逐项与 §4.3 的缓冲表、§3.3 的分配一致，无遗漏大项（`r_ub` 32768 +
    `lane_sum`+reducer scratch 1024 + `tile_sum` 76 + `ids` + 5 个 UB 标量）。
  - `l0c_budget`: **skip**（"no Ascend Cube GEMM call"）→ 正确，§2/§4.4 声明纯 Vector、无 L0C。
  - `core_split`: **skip**（"insufficient parsed dims"）→ **人工已核**：① 逻辑核数 `= B`
    ∈ {2,16,17,32,64,256}（§5.5①）；② 物理核 56 有平台实查记录（`statectl hardware.observed_cores
    = {cube:28, vector:56}`，来源 `Ascend950PR_9579.ini [SoCInfo] vector_core_cnt=56` + 运行时
    `rtGetAiCoreCount()=28`），并附 Stage 3 复核代码；③ `blocks=min(B,56)`、`per_core=ceil(B/blocks)`
    取值自洽（56×5=280≥256、56×2=112≥64、56×6=336≥300 的 fixture）。
  - `r3_metrics`: **pass**（"all four R3 metrics present"）。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

**语义核对**（§0.1/§0.2 ↔ 源码 + golden）：§0.1 的 7 步与源码及
`workloads/sampling.py` 的 `philox_uniform`/`_draw`/`chain_speculative_sampling` 一致，逐条核对：
(1) `u = (c0 >> 8)·2^-24` ↔ 源码 202–204 行 `counter[0] >> T.uint32(32-UNIFORM_BITS) *
float32(2**-24)`，`philox.py` 的 `UNIFORM_BITS=24` 与 golden 的 `(c0>>8).to(float32)*2.0**-24`
逐位同；(2) 接受判定 `u·D < T`（fp32）↔ 源码 210–218 行；(3) `num = min_j verdict_j` ↔ 源码
221–223 行；(4) 残差 `max(0, T − (n<N ? D : 0))` ↔ 源码 225–227（`min(n, N−1)` 基址）+ 361–368
行（`drafted` 谓词）；(5) 逆 CDF + 零权重跳过 + `last_positive` 兜底 ↔ golden `_draw`
（`searchsorted(right=True).clamp(V-1)` + `next_positive` + `last_positive`）；(6)(7) 输出与
`num_accepted` ↔ 377–383 行。dtype 语义（fp32 概率、int32 id/输出、int64 seed/offset、第 5 步
fp64 前缀）与源码 193–204、227、263、361 行一致。§0.2 的 I/O 与源码签名（142–149 行）一致，
并正确识别 `partial`（147、495 行）为 `forward` 内部分配的 fp64 跨块缓冲、非公开输出；
输出 shape `[B,N+1]`/`[B]` 正确。**无一处语义偏差**。

**逐位契约判断（专项核对）**：设计的分级判断与测试的真实断言一致，未把分布等价当逐位：

| 项 | 实现侧 | 参考侧 | 判定与证据 |
|---|---|---|---|
| `num_accepted` | 首个拒绝 `min_j verdict_j`（§3.1 P4） | `cumprod(1).sum(1)`（golden 第 84 行） | **逐位**：`cumprod` 在首个拒绝后恒 0 ⇒ 和 = 首个拒绝下标，整数域恒等（独立推演成立，E3 `0/92`）；测试用 `exact_compare(out_accepted, ref_accepted)`（`test_sampling.py:79`） |
| uniform / Philox | 十轮 `(j,b,off)` + 32 位拆分 mulhi（§3.1 P1/P2） | `philox_uniform`（golden 第 61–80 行） | **逐位**：key/counter/Weyl/轮序/mulhi/`>>8` 逐项一致；`cast(uint32<2^24→fp32)` 与 `·2^-24` 均精确；E2 全 0（含 `−1/−1`、`2^63−1` 角点） |
| 输出前缀 / `−1` 尾 | `j<n→id`、`j>n→−1`（§3.1 P11） | 同 | **逐位**（构造保证；`verify_chain_outputs` 断言前缀等于 `draft_token_ids`、`−1` 紧跟被抽 token） |
| 被抽 token | fp32 分层前缀 + 首个越点（§3.1 P8–P10） | fp64 `cumsum` + `searchsorted` | **结构 + 分布**（非逐位）：测试只做 `verify_chain_outputs`（正权重）与 `assert_chain_distribution`（6σ），**没有**对参考的 `exact_compare`；边界用例的 `exact_compare` 是"同一 kernel 的对齐 vs `storage_offset=1` 输入"，不是对参考 |
| fp64 → fp32 分层前缀是否改变 `num_accepted` | — | — | **不改变**：`num_accepted` 只由 A2（首个拒绝）与 uniform 决定，A4 只进入第 5 步的抽样点；E3 与 E5 分别独立验证 |

**算法解读完整性**（§0.3 ↔ 源码）：步骤 1–7 覆盖源码 190–383 行全部计算语句（draw 轮次
`190–219`、`verdict` 210–218、`min` 221–223、行基址 225–227、折叠 `229–252`（`stage`/`fold`
宏 55–78）、`parts>1` 分块和 + `T.sync_grid` + `locate` 275–297、命中 warp 重读 + 三段
`locate` 299–376、写出 377–383），未发现遗漏或臆造步骤；数据流图覆盖 `held/other/seg/partial/
chunks/tail/lane/prefix/total/left/pick/verdict/token_uniform` 全部缓冲；host 侧逻辑
（`_plan` 427–440、`refusal` 442–463、`forward` 479–510 的 `partial` 分配与 `view(-1)`、
`@target_cache(maxsize=32)`）均已列出。

**优化手段覆盖**（§0.4 ↔ 源码）：14 项逐项对应源码显式优化——单遍折叠（76–78、229–252）、
分层下降（81–138、299–376）、`shfl_xor` 蝶形（245–247、124–127）、`load16_evict_first`
（`load_vector(..., True)` 66–68 + `csrc_include("streaming_load.h")` 52）、`pace` 寄存器暂存
（168–169、230–243）、`__umulhi`（`philox.mix`）、`T.sync_grid`（274）、fp64 `seg` 叶子（153）、
切分无关的 counter（195–200）、2N 标量链测试（208–218）、bonus 位零 draft（67–68、241）、
单 lane 写（248–251）、静态特化（22–25）、两级分组（74–78+244–247）。算法层 / 实现层归类正确。

**耦合性判定**：§0.5 每条有四态处置 + 依据；抽检与源码/平台文档一致：warp 蝶形 → 无 warp 语义
（`programming.md` 无 `shfl_xor`；`T.SimtVF` 打开的是 SIMT 线程域）判"等价替换"合理；`T.sync_grid`
→ `CG-950-batch-barrier`（`knowledge/capability-gaps.md` 首条："跨 block barrier 依赖未认证…须选已有
支持路径"）判"重新设计"正确；`__umulhi` → 判"重新设计"正确（见下 R3 核对）；`load16_evict_first`
→ `programming.md` 未列任何缓存策略接口，判"舍弃"合理；`pace=4` 舍弃给出"意图由编译器多缓冲承接"+
Stage 4 判据（`mte2_ratio`），无静默丢弃。

**NPU 重设计可行性**：§0.6 R1–R5 均有「源方案 → NPU 新算法 → 语义保持论证」三要素：
- R3（32 位拆分 mulhi）：我独立推演成立——`a·m = p3·2^32 + (p1+p2)·2^16 + p0`，
  `t=(p1&0xFFFF)+(p2&0xFFFF)+(p0>>16) < 2^18`，`hi = p3+(p1>>16)+(p2>>16)+(t>>16)` 恰为精确高字
  （`a·m ≤ (2^32−1)^2` ⇒ 高字 < 2^32，无回绕），`lo` 取低字，与 `__umulhi` 定义逐位等；
  「`(int64)a·m>>32` 能编译但算错」的陷阱有生成码证据：`probe_p1b_gen.txt` 第 16 行
  `(((int64_t)c[0]) * (int64_t)3528531795) >> (int64_t)32`（有符号 64 位乘法溢出 + 算术右移），
  而 `probe_p1a_gen.txt` 显示拆分式生成的是纯 `uint32` 的 `&/<<>>/+/·` 表达式（53841/8019 即
  `M0` 的两个 16 位半字），与 §1.6.1 A1 的式子逐项同形。E1 `0/160072` 字、E2 十轮流逐位 0 不符。
- R4（取消屏障）：一行一核、`tile_sum` 留在 UB，正确性依赖严格更少（无"部分和已写未读"窗口）；
  与 §5.5 分核、§7.1 同步一致。
- R5（舍弃项）：给出"不影响正确性/不增加流量"的论证与 Stage 4 回归判据，非空泛理由。

**§1–§7 与 §0.5/0.6 一致**：§1.4 五阶段 / §1.5 数据流 / §3.3 伪代码 / §4 内存 / §5 分核 / §6 循环
全部描述的是 §0.6 的 NPU 主算法，无"§0 判重设计、§3 照抄源方案"的脱节；§3.5.2 的差异表逐项写明
语义影响。**golden 独立性**：§8.1 明确"移植 `workloads/sampling.py`、不复刻本文分块/流式结构"，
且 golden 只吃 `(seed, offset, B, N, V, 输入)`，不含 `tile/lane_sum/blocks/per_core`；
`verify_equiv.py` 内的 `ref_*` 系列是 golden 的同源移植（第 39–93 行），与候选实现（99–234 行）分离，
不存在自证。

**建议（建议级 S10）**：§9.3 的「残差行全零 → 参考同样 `clamp(max=V−1)` 后取到 0」与实际参考行为
不符——全零行时 `searchsorted(right=True)` 返回 V、clamp 后 `next_positive[V−1]=V`、`chosen==V`
走 `last_positive`，而 `positive` 全 False 时 `V−1−argmax(flip(pos))=V−1`，即参考返回 **V−1** 而非 0。
该分支已论证为不可达（拒绝位满足 `T ≤ u·D < D` ⇒ 至少有 `D−T>0`；`n=N` 时残差 = target 行 > 0），
但既然写进了文档，应改成"参考在此分支取 V−1，本设计不可达"以免误导 Stage 3 的兜底实现。

### 1. API 可行性 — pass

在**任务目录**下按 ruleset 做静态查证（`T.__file__ =
/workspace/venv-ascend950/lib/python3.12/site-packages/tilelang/ascend/language/__init__.py`，未在仓库根
用被遮蔽的源码包），§3.2/§3.3 用到的每个符号均存在且签名匹配：
`T.Kernel`（`kernel.py:139`，一维、无 `threads`）、`T.SimtVF(threads=…)`（`frame.py:107`）、
`T.Parallel`、`T.serial`、`T.alloc_shared/alloc_local/alloc_fragment`、`alloc_reducer`（`allocate.py:296`）、
`reducer_init`（签名 `(reducer, init=None)`，docstring 明确"整个 epoch 可置于线程一致循环/条件内，
init 与 finalize 共享同一作用域"——§3.3 的 per-tile `init/finalize` 合法）、`reducer_update`
（`acc[indices]` 形式，`can[0]` 用法正确）、`finalize_reducer(reducer, dst)`、
`get_thread_binding`、`if_then_else`、`reinterpret(value, dtype)`（**值在前**，§3.3/§1.6.2 V15 的
`T.reinterpret(·,"uint32")` 与探针 p1a 的 `T.reinterpret(seed_w[0],"uint32")` 均正确）、
`min/max/cast/uint32/float32/sync_threads`。
运行期行为的三条结构性断言均有本任务生成码佐证：`T.Parallel` 必须在 VF 区内（p2a 首版报错）、
`T.alloc_local` 不能作为 VF 实参（p2a 第二版报 `must reside in '__gm__ or __ubuf__'`）、
reducer 三件套生成 `tl::AscendAllReduce<tl::SumOp,128,1,0>`（`probe_p2b_gen.txt:72`）。
§3.4 的"`T.copy` 动态基址在 VF 内不可编译"我**亲自复跑确认**：把
`history_version/design_probe_rowfold.py` 复制到 `/tmp/opencode/probe` 后执行
`ASCEND_RT_VISIBLE_DEVICES="" … design_probe_rowfold.py p2e` → `COMPILE: FAIL`，
`tvm.error.InternalError: Check failed: (allow_append_) is false: at[T.FloorDiv(267008, 32)] shared.dyn`，
与 §9.2 第 6 条记录一致；主选改元素级 `T.Parallel` 读（p2c/p2b 成功）是正确规避。
模式与 ruleset 一致：`workflow.md` 允许"根据实际执行路径选择 SIMD、SIMT、Cube 或混合"，SIMT 侧有
`PL-950-row-primitives`（仿真正确性）+ `examples/paged_cache_write`（动态行基址 + 谓词）双重佐证。

**建议（建议级 S1）**：§2.2 判 SIMD "取证不足" 的两条事实性表述与仓库现状矛盾——原文"本仓**无任何
算子用 `T.simd.*` 做逐元素算术**"，但 `examples/mla_dense_bwd/_delta_kernel/_delta_kernel.py:135-147`
用 `T.simd.vld/vcvt/vmul/vadd` 做逐元素点积、`:212-228` 用 `T.simd.vsts/vcadd` 归约，
`_dq_land_kernel/_dq_land_kernel.py:189-193` 同类，三个算子的 `REVIEW.md` 均为「结论: 通过」。
**结论（选 SIMT）仍成立**（ruleset 允许按执行路径选择，且三件事在 SIMT 侧都有本平台证据），但理由需
改写为："`T.simd.*` 侧已具备寄存器级逐元素/归约实例，缺的只是本算子所需的『动态行基址 + 谓词掩码 +
块级 reducer + `tile=8192`』组合证据"，并把这 2–3 个 examples 列为参考（§3.5.3 可顺带补入），
避免"非能力否定"的依据被后续任务当作事实引用。

### 2. 内存层级规划 — pass

§4.4 给出完整搬运路径：`GM→寄存器（VF 内元素读）→ reducer → 标量 → UB[tile_sum]`、
`GM→UB[r_ub]（命中 tile 重算）→ lane_sum → token → GM`；无 L1/L0/TMA/nd2nz，声明与"无 Cube 段"自洽。
§4.5 UB 复算已如机械复核段所列（33.2 KB / ×1.7 = 56.4 KB，占 22.7%；`N=600` 24.3%），
`tile ≤ 37000` 的上界推导（`(253952/1.7 − 1200)/4`）正确。§4.3 每个缓冲标注层级（UB / register）与
生命周期，并显式列出被取消的源缓冲（`seg/chunks/tail/prefix/lane/total/left/verdict`）。
对齐策略（§4.4/§3.5.1）与实际约束一致：行基址只保证 4 字节对齐（`V=4099`、fixture `storage_offset=1`），
因此不依赖 16B DMA，元素级读由编译器按可证明性向量化（p2b/p2f 生成 `float2` 读）——不虚报 16B 对齐。
R9（自动多缓冲把 `r_ub` 膨胀到 2 版）给出 `annotate_buffer_versions({r_ub:1})` 的兜底，容量仍在预算内。

### 3. Tiling 策略 — pass

§5.2 给出 `blocks=min(B,56)`、`per_core=ceil(B/blocks)`、`tile=min(8192, ceil(V/128)·128)`、
`per_lane=tile/128` 的具体值与依据；代表取值经我复算全部自洽（`V=151936→tile 8192,ntile 19,
per_lane 64`；`V=4099→tile 4224(=33×128),ntile 1,per_lane 33`；`V=64→tile 128,per_lane 1`；
`V=32000→ntile 4`）。非整除用尾 tile 谓词 + `line<B` 行守卫（§5.4），满足"不做动态 extent"。
分核三要素齐全且数值自洽（见机械复核 `core_split` 人工核对）；核内串行边界 `per_core≤6`、
`ntile≤19`、`N+1≤601`、`per_lane≤64` 全为编译期常量，行覆盖 `line=bx+rr·blocks` 对
`B≤56·per_core` 完备。**核利用度已被量化并给裁决路径**：`B=16/17/32` 时利用度 28.6%/30.4%/57%，
`B=64` 满发 56 核（64 行/112 槽 57%），`B=256` 91%；对应降级方案为 §1.6.3 备选 B（两 kernel）+ 10%
阈值实验裁决，非纸面放行。

**建议（建议级 S5 之一）**：§5.5② 的"（AI Core = 28，AIV = 2×AIC = 56）"是固定乘二推断，
`workflow.md` 明确"不要使用 … 固定乘二替代本平台查询"。虽然同段已给出 `[SoCInfo] vector_core_cnt=56`
与 `statectl observed_cores.vector=56` 的实查记录（依据链成立），仍建议删去这一句内联推断，
直接引 `vector_core_cnt=56`，避免被后续任务当成"2×AIC"的通用规则。

### 4. 技术约束检测 — pass

无三维 Kernel、无 GEMM、无 L0C（`l0c_budget` skip 正确）。§3.5.1 把本仓已知限制逐条对上并给出处理：
`T.Kernel` 一维（§5.2/§3.2）、`T.Parallel` 只能在 VF 内（§3.3 全部包在 `T.SimtVF`）、
跨 block barrier 未认证（一行一核，回避）、手工跨核 flag 未认证（备选 B 用两个 kernel 而非 flag）、
二维 shared→shared 归约缺陷 `CG-950-simd-shared-reduce`（只用一维 reducer/`T.Parallel`，并据此弃选 L3）、
UB 253952 B、4 字节对齐、fp64 与 64 位整数不使用（`seed/offset` host 侧 `view(torch.int32)`）、
`T.alloc_local` 不可跨 VF 边界（跨阶段状态用 UB 标量）。未发现遗漏的平台约束。

### 5. 循环与同步 — pass

§6.1 用 L1–L10 逐条给出类型（`T.serial`/`T.Parallel`）、上界与作用的对应关系，与 §3.3 缩进结构一致；
`T.Pipelined`/`T.Persistent` 的不用给了具体理由（tile 循环不写跨迭代 UB 数据、跨核 barrier 禁用），
不是一句"不需要"。§7 的同步策略与选定模式一致：不写显式 flag，靠 `InsertSync`（p2a/p2b/p2f 生成码
逐处可见 `asc_syncthreads`），并把 6 个同步点 S1–S6 与"哪些落在缓冲边界"讲清；把前缀状态从 UB 标量
改进为 `T.alloc_local` 的依据（p2f 生成码在 128 步前缀里每步插 `asc_syncthreads`）是我在
`probe_p2f_gen.txt` 里亲眼确认的行为，且给出"若真机行为异常回退 UB 标量版"的退路。单点写守卫规则
（§7.2 "所有 GM/UB 单点写 `thread 0` 守卫"）与 p2f 生成码一致。§6.4 尾块四类（尾 tile / 尾行 /
`V<tile` / 任意 `N`）逐一给出处理与代价。

### 6. 验证方案 — pass

§8.1 golden 以 §0.1 语义为准、移植 `workloads/sampling.py`（骨架 + "逐字实现以参考同名函数为准"），
独立性要求写明；`verify_equiv.py` 第 39–93 行已包含一份可直接运行的同源参考实现，等于 golden 草案
已可执行。§8.2 精度标准按输出分级（`num_accepted` 逐位；前缀/`−1` 尾逐位；被抽 token 正权重 +
6σ 分布；两次调用逐位；对齐无关；dtype 守卫）——与我通读 `test_sampling.py` 的真实断言**逐条对应**
（`exact_compare` 只用于 accepted、参考可复现性与同 kernel 两次调用；token 只走
`verify_chain_outputs`/`assert_chain_distribution`）。§8.3 的 L0/L0b/L1/L2/boundary 计划覆盖
`ntile=19`、`B>56` 核内串行、`V=4099` 非整除、`N=600` 压力链、`storage_offset=1`、int dtype 拒绝，
并绑定 `execution.precision_command` 的 `--level` 约定；L0 门槛先行的顺序正确。

### 7. 完整性与一致性 — warn

模板 11 章节齐全，迁移类 §0.1–§0.7 全在；无占位符标记（已按门禁的三类占位串逐一 grep，均无命中）；
§3.5.3 引用的 5 处路径全部真实存在（`examples/rope/_rope_positions_kernel/`、
`examples/paged_cache_write/_paged_cache_write_kernel/`、`ascend/kernel_template.py`、
`knowledge/repro/PL-950-row-primitives.py`、`knowledge/repro/PL-950-simd-reduce.py`）。
但存在三处内部不一致（均不影响语义正确性，建议在冻结前修订）：

**建议（建议级 S3）折叠 pass 的形态自相矛盾**：§3.3 主伪代码（757–758 行）是
`T.reducer_update(can[0], residual(...))`（**不写 `r_ub`**），而紧随的"折叠 pass 的形状"注
（808–811 行）改成"一个 `T.Parallel(tile)` 体内先 `r_ub[i] = residual(·)` 再
`T.reducer_update(can[0], r_ub[i])`"（**每 tile 写 `r_ub`**）。后者与 §1.6.1 A3 的收益
（"省一个 `tile·4B` UB 缓冲与每 tile 一次 UB 写（`tile` 个 store）"）、§4.5 的 `r_ub` 生命周期
（"命中 tile 的重算"）、§6.3（"`r_ub` 只在阶段 C 一次填充"）直接冲突。**改法**：二选一并在四处
统一——(a) 保留主形式（`reducer_update` 吃表达式，`r_ub` 只写一次），把注改成"若编译器对表达式内
GM 读的向量化不理想，备选是先写 `r_ub` 再归约（此时 A3 的 UB 写收益作废、`r_ub` 可能被多缓冲，
UB 仍 < 253952）"；或 (b) 采用写 `r_ub` 的形式，同时删掉 A3 的"省每 tile 一次 UB 写"并更新 §6.3。

**建议（建议级 S4）residual 宏与 R6 的"两处钳位"不一致**：§3.3 注（813–815 行）的 `residual` 是
`T.if_then_else(i0 < V, T.max(T[at_t+i0] − …, 0), 0)`，`at_t+i0` **未做 `min(i0, V−1)` 钳位**；
而 §9.1 R6 说"依赖 `min(·, V−1)` 与 `min(n, N−1)` 的钳位才能保证地址合法"，探针 `_build` 用的也是
`target_probs[at[0] + T.min(t*tile+i, vocab-1)]`。若 Stage 3 照 §3.3 直写，而编译器对该谓词选择
"select + 预取"而非分支（p2b 的 draft 谓词是分支，但 `i0<V` 未在探针中单独验证），末行的尾 tile 会
读越界最多 `tile−1` 个元素。**改法**：把 `T.min(i0, V−1)` 写进 `residual` 宏（与探针一致，代价为一条
`min`），或把 R6 改写成"安全性依赖谓词生成分支，Stage 3 必须逐条核对生成码，否则加钳位"。

**建议（建议级 S2）自报计数有误**：§0.5 末尾"保留 7 项 / 等价替换 6 项 / 重新设计 2 项 / 舍弃 2 项"
合计 17，但该表实有 18 行、逐行统计为保留 7 / **等价替换 7**（语义第 5 步、#3、#8、`vec=4` 16B 读、
#12、#14、`_plan` 分核）/ 重新设计 2 / 舍弃 2；§1.6.2 末尾"保留的 **8 处**标量点（V5–V14）"实为
**10 处**（V5–V14，另有 V15/V16 的 2 处 host 元数据，这一句本身是对的）。修正数字即可，不影响任何决策。

### 8. 算法优化分析 — pass

**§1.6 章节存在性**：1.6.0 / 1.6.1 / 1.6.2 / 1.6.3 四个子节齐全，各有实质内容。

**调研四问完整性（逐项核对）**：
- R1：候选表含**基线 C0**（源 GPU 算法，逐字段描述 `parts`/屏障/fp64 叶子），另有 C1–C6 六个候选；
  每条有"公式/结构 + 等价性初判 + 收益方向"。命中行覆盖：`knowledge/algorithm-candidates.md`
  只有 elementwise / reduction / softmax / norm / GEMM / attention / mixed 七族，**无抽样族行**，
  R1 的"参考表无命中"成立。
- R2：结论是明确"**有，且采纳**"，依据三条（结构判据 + 源算子自身单遍折叠 + 参考 `cumsum` 为反例），
  并指出唯一不可在线化点（越点下标需先有 `total`，故保留 `O(tile)` 重算）。符合 R2 的"分块归约 +
  合并（同一 launch 内）"定义。
- R3：四口径齐全（FLOPs / GM Bytes / 扫描遍数 / 中间缓冲峰值）+ 可并行度 + 跨核归并代价，**基线 C0 在表**，
  口径说明写清"Bytes 只计 GM 流量、中间 GM 往返计入"。我独立复算：C0 `4V/8V`、C1 `(4V或8V)(1+1/ntile)`、
  C2 `8V+4V+4V=16V`、C6 `2×`、C4 `≈12V`、扫描遍数 `1+1/ntile` 与 `1+32·vec/V` 均与设计数字一致。
- R4：逐候选 6 项清单（计算单元 / 容量 / 对齐整除 / 静态边界 / 流水融合 / 跨核结构）全填，淘汰项有依据：
  C0 的 fp64/warp/`__umulhi`/`createpolicy` 无等价物 + 屏障未认证（§0.5 + `capability-gaps.md` 可核）；
  C3 fp64 ALU；C4 `V` 次超越函数；C2/C6 流量翻倍。容量淘汰引用平台 `design_budget`（950PR 自有值），
  未拿 910B2C 常数当能力判据。
- 调研结论：选定 C1 + §0.6 主结构，三条关键依据指向 R3/R4 与"优化意图同构"，**调研范围写明**（参考表 /
  examples / pattern-library / 结构自查 / 公开已知变体），不是空白或一句"无更优替代"。

**调研结论正确性（独立复核）**：① 等价性——C1 相对 C0 只改第 5 步分组，第 1/2/4 步逐位保持，推演成立
（见维度 0 的逐位契约表）；② 复杂度复算——见上，唯一偏差是 C3 行 Bytes 写成 `16V`（只计 fp64 CDF 写读），
而 C2 行计了输入行读；按同口径 C3 应为 `8V`（读两行）+`8V`（写）+`8V`（读）=`24V`（见建议 S9，结论不变）；
③ 负向断言复核——我**重跑 `kb_search.py`**：`"sampling inverse cdf philox"` → `no entries matched`，
`"inverse transform sampling cumsum prefix"` → `no entries matched`，`"row reduce simt reducer"` →
只命中 `CG-950-simd-shared-reduce`（与本算子无关，设计也如此引用），与 §1.6.0 的自述完全一致；
`algorithm-candidates.md` 全文亦无抽样族——"参考表无命中"不是借口而是事实；④ 并列候选 C1/C0 流量同阶，
已按 skill 要求把 C5（多核拆行）降级为备选交 §1.6.3 实验裁决，未纸面拍板。

**调研 → 设计一致性**：§1.4 的五阶段 = C1 + §0.6 主结构；§3.1 的 P1–P11 逐条标注 A1–A6；§6.1 的
L1–L10 与 §1.6.2 的 V5–V14 一一对应（行循环→V5、draw→V7、verdict→V8、折叠→V1/V6、重算→V1、
lane 部分和→V2、粗层→V10、lane 前缀→V11、元素走查→V12、写出→V13）；§0.5/§0.6 的处置与调研选族一致；
源算子的 14 项优化在 §0.5 有处置、在 §0.6 或 §1 有承接（逐项核对，无静默丢弃）。迁移侧的"优化意图承接"
（fp64→fp32、warp→block reducer、屏障→无屏障、`__umulhi`→拆分乘法）逐条有说明。

**数学等价优化完整性 / 等价性独立推演**：A1–A6 六项均有四要素。我逐项独立推演：
A1 精确（见维度 0，E1 `0/160072` 与内嵌表一致）；A2 在布尔域恒等（`cumprod` 在首个拒绝后恒 0 ⇒
和 = 下标；无拒绝 = N），E3 `0/92`；A3 元素值仍是同一 fp32 减法 + `max`，重算确定性，E4 `0/92`；
A5 `x−0` 与 `max(0,x−0)` 与参考 `clamp_min(0)` 同值（`−0.0` 不影响和与 `>0`），E4 覆盖 `n=N` 行；
A6 的"首个越点自动落在正权重上"推演成立（若 `w_k=0` 则 `cdf[k]=cdf[k−1] ≤ point` 与 `cdf[k]>point` 矛盾），
`point ≥ total` 的兜底与参考 `last_positive` 同规则，E5 覆盖 `u=0`、`u=(2^24−1)/2^24`、首尾补零；
A4 是**容差内**等价且明确标注，误差上界 `(per_lane+log2(threads)+ntile)·2^-24 = 90·2^-24 ≈ 5.4e-6` 与
实测 `4.845e-08` 量级自洽，影响面限定在非契约的 token 下标。**没有一项变形改变 `num_accepted`、
接受前缀或 uniform。** `verify_equiv.py` 重跑结果与 §1.6.1 内嵌表逐格相同（无"表是旧跑"的情况）。

**向量化替代完整性（与 §3.3/§6 交叉核对）**：§1.6.2 盘点 V1–V16，覆盖 §3.3 伪代码与 §6.1 的全部
循环/标量点（我逐条对照，无遗漏：折叠、lane 部分和、尾块谓词、draft 谓词、行循环、tile 循环、Philox、
verdict、`min`、粗层、lane 前缀、元素走查、写出、地址算术、seed 拆分、`tile` 取值）。V1–V4 已向量化
且有生成码佐证（`float2` 读、`AscendAllReduce`、`if/else` 条件加载、尾块谓词）。不可替代项逐项给
**具体到本算子**的理由类别（block 级任务映射 / 数据分块控制流 / 动态索引 gather / tile 级顺序依赖 /
block 级小量标量 / host 元数据），没有"实现简单"式理由；V7（Philox 标量化）的取舍还附了已实测缺陷
（lane 共享 key 的 Weyl 被每 lane 各加一次，`probe_p1c_gen.txt` 可见 `if (threadIdx.x < 9) kw[0] += W0`）。

**向量化轴与布局决策完整性**：按"规约类"清单枚举了**水平归约（L1）**与**垂直扫描（L2）**两条正交
lane 映射，并补 L3（2-D tile）、L4（UB 重排）；逐候选给整除性 / 尾 lane 浪费 / 累加链形态 / repack
代价 / UB 影响的量化（L1 8192=128×64、尾浪费 ≤127 lane 0.4%；L2 串行深度 2374 倍；L3 受
`CG-950-simd-shared-reduce` 限制且 `R·tile≤8192` 只减半 op 组；L4 `+tile·4B` UB 读换 0 收益）。
**弃选前提我已亲自核对**：`CG-950-simd-shared-reduce` 确为"高层二维 shared→shared 归约输出差异"且限定
于该路径（`capability-gaps.md` 原文）；L4 的 repack 代价写成 UB 级读写（未写成 GM transpose），与
`copy_op.py` 的语义不矛盾；I/O layout 只作契约、核内布局作设计变量的区分明确。
**实验裁决三件套**：主选 L1 + 备选 B（结构完整：kernel A/B 分块和落 `workspace` + 收尾 kernel 重读命中段，
buffer/UB 预算/分核三要素/dtype 路径按新粒度重算）+ 裁决计划（代表 shape 3 个、指标 `msopprof`
Task Duration、A/B 交错、**阈值 10%**、回写路径、按 `B<56` 分 shape 分派可行性）。判定裕度确实落在
未实测常数（每核带宽 / launch 3–5 µs）上，故三件套是必需的，设计已具备。
**布局 → 实现一致**：`tile=threads·64`、`per_lane`、`T.Parallel(tile)` 尾轴映射、`r_ub[tile]`/
`lane_sum[128]`/`tile_sum[ntile]` 在 §3.3/§4.3/§4.5/§6.1 四处一致，未出现"§1.6.3 选 A、§3.3 画 B"。
**与下游一致**：§3.1 以 A1–A6 优化后公式为唯一输入；§6 的循环结构与 §1.6.2 的判定一致（判定已向量化
的点不再出现对应标量循环）。

**建议（建议级 S5）设计期 roofline 的流量项未折算核利用度**：§1.6.0 的"设计期估算"用
`CONST-mte2-degradation` 的**聚合地板 1.26 TB/s**（该条目原文为 "48 vector core / 256M 3:1 混合读写"
的口径）除以每个 workload 的 bytes。但 `B=16`/`B=17`/`B=32` 三个 workload 只用 16/17/32 个核
（利用度 28.6%/30.4%/57%），单核带宽不会自动放大，故 `llama-8b-b16-n4` 的 6.5 µs、`b17-n8` 的 6.9 µs、
`qwen3` 的 15.4 µs 作为"下界"约乐观 2–3.5×（按 1.26 TB/s ÷ 48 核 × 16 核 ≈ 420 GB/s 估，8.21 MB ≈ 19.6 µs）。
虽然 §1.6.3 备选 B 已把"每核带宽 vs 聚合带宽"列为待反解常数，但表格本身没有折算，Stage 4 会把它当作
可达下界。**改法**：给流量项加一列"有效核数/56"，或对 `B<56` 的行注明"须乘 `56/B` 或按单核带宽实测校准"，
并在结论行把"1.5–26.3 µs"分成 `B≥56` 与 `B<56` 两组。另外同段的
`CONST-vector-launch-overhead`（910B2C，2026-08-24）版本戳与条目 front-matter（toolchain 2026-09-09）
不一致，请以条目为准。

**建议（建议级 S6）A1 的收益量化数字自相矛盾**：§1.6.1 A1 写"总代价 `≈ 2.3e4` 条标量指令
`≈ 13 µs`/2304 次 ≈ **26 ns/核一轮**，占估算下界的 **< 0.3%**"——`13 µs` 与 `<0.3%`（15.4 µs 的 0.3%
约 46 ns）互斥，`2.3e4` 条指令也不对应 13 µs；§1.6.2 V7 用的是"≈26 ns/核"，两者需统一。
**改法**：按"每核行数 × (N+1) × 10 轮 × ~10 条 32 位指令"重算（`B=256` 时每核 5 行 × 2 draw × 100 ≈
1000 条 ≈ 1 µs 量级），给一个口径自洽的数字并保留"< 1%"的结论（结论本身不受影响）。

**建议（建议级 S7）V11 的备选与已跑的探针不对应**：§1.6.2 V11（lane 前缀 ≤128 步）写"若 Stage 4 需压缩，
可改两级（4×32）"，但 `history_version` 里 p2g（`_locate_bisect`：掩码半窗 + 6 次 `op=max/sum` 全归约）
正是"把 128 步串行前缀换成归约"的候选，且**没有 `probe_p2g_gen.txt`**（p2e 因失败无生成码是合理的，
p2g 无生成码则缺少证据），而 §10.1 的目录清单用 `probe_p2a_gen.txt … probe_p2f_gen.txt` 的省略号
暗示 p2a–p2g 齐全。**改法**：在 V11 的"替代方案"栏改引 p2g（并补跑生成码 / 在 §3.4 记录其结果与
未采纳原因：多一次全 tile 归约 pass、`ntile` 不变但 op 组 +1），或删掉 p2g 探针与 §10.1 的对应表述。

**建议（建议级 S8）E5 的校验保真度与 kernel 分组不同**：`verify_equiv.py` 的 `cand_draw` 用
`tile=256/512/4096`、`threads=32`（`per_lane` 8–128），且 lane 层用**串行** `s = s + acc` 合并；
而 §3.3/§6 的 kernel 是 `tile=8192`、`threads=128`（`per_lane=64`）且 tile 和来自**树形**
`tl::AscendAllReduce`（`log2(128)` 级）。两者"规则"相同、**舍入分组不同**，所以 `0/68` 是强证据但
不是 kernel 分组的直接验证。**改法**：补一个与 kernel 同参数的 E5 变体（`tile=8192, threads=128`，
lane 和用成对树形合并），或在内嵌表里注明"分组不同、结论为量级证据"，并在 §9.1 R5 保留 Stage 3 的
65536 行分布用例作为第二道闸（设计要求已如此）。

**建议（建议级 S9）R3 表内 C3 的 Bytes 口径与 C2 不一致**：C2 计了输入行读（`8V`）+ CDF 写 `4V` +
CDF 读 `4V` = `16V`；C3 只计"fp64 CDF 写读各 `8V`"= `16V`，漏掉两行输入读 `8V`，同口径应为 `24V`
（A4 的收益量化行也写"至少 `16V`"，同样是漏计输入读）。**改法**：把 C3 改为 `24V`、A4 收益改为
"`24V` → `8V(1+1/ntile)`"；C3 的淘汰依据是 fp64 ALU 缺失与流量，改数字不改变结论。

## 检视旁注（不进维度判定，供 Stage 3 参考）

- **Stage A 的幂等性**：§3.3 把阶段 A 放在 `T.SimtVF` 之外（核内标量）。该写法在 p1a/p1b 上可编译，
  但"核内标量执行一次"没有真机证据（§9.1 R1 已如实记录）。有利的一点是阶段 A 的全部写都是**同值幂等**
  （`ids[]`、`n_s=min(n_s,j)`、`at_t/at_d`、`u_tok`），即使被 128 lane 冗余执行也不会算错，只是浪费——
  建议 Stage 3 在 L0 上用「打印每行 draw 次数 / 用 `ids` 计数」验证一次。
- **两条兜底里的"条件 `T.Parallel`"**（§3.3 注的 `last_positive_in_tile`）在探针中未单独验证
  （p2a–p2f 的 `T.Parallel` 都没有包在 VF 内的运行期 `if` 里）。属概率为零分支，但**必须能编译**；
  建议 Stage 3 的第一个编译探针就带上它，若被拒则改为"无条件算一次 `op=max` 归约"（多一个 op 组）。
- **`u*d >= T` 与参考 `!(u*d < T)` 的等价**：在"概率行有限、非负、归一"的契约下无 NaN，等价成立；
  Stage 3 不要改成 `>` 或加 epsilon。

