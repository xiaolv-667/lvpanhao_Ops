# _ssd_chunk_coupling_kernel 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息

- 文档: `examples/ssd_chunk_coupling/_ssd_chunk_coupling_kernel/DESIGN.md`（867 行）
- 任务类型: 迁移（TileOPs harness 子模式，family=mamba）
- 源算子: `/workspace/TileOPs/src/tileops/_kernels/mamba/ssd_chunk_coupling.py`（218 行，已全文 Read）
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 7

## 机械复核（D-1/D-5）

### 1) verify_equiv 重跑（D-1）

命令（工作目录 = 算子目录，平台 env 已 source）：

```
/workspace/venv-ascend950/bin/python verify_equiv.py
```

实测输出（2026-11-23，torch CPU，无设备）：

| 项 | 元素数 | 违反数 | max_ulp | verdict |
|---|---|---|---|---|
| E1（O1 融合 select+cast）/ float16（random+corner+garbage 各 256 = 768） | 768 | 0 | 0 | EQUIV_PASS |
| E1（O1）/ bfloat16（同上 768） | 768 | 0 | 0 | EQUIV_PASS |
| E2（O2 掩码判据强度缩减，BL∈16/32/64/128 × l0,s0 网格 × sid∈0/1） | 262 144 | 0 | 0 | EQUIV_PASS |
| E4（O3 被掩码 tile 的 K-trip 置 0）/ float16 | 256 | 0 | 0 | EQUIV_PASS |
| E4（O3）/ bfloat16 | 256 | 0 | 0 | EQUIV_PASS |
| E3（未采纳：乘法掩码，仅作反例） | 64 | 28 | — | 反例按预期失败 |
| 汇总行 | — | — | — | `"failed": []`，`all adopted items EQUIV_PASS` |

**与 §1.6.1 内嵌表一致性：完全一致**（768 / 768 / 262 144 / 512 元素、max_ulp 0、违反率 0；E3 的 28/64 亦一致）。结论：**EQUIV_PASS，脚本可重跑、无静默失败**。

### 2) design_calc_check 复算（D-5）

命令：

```
python3 .agents/tools/design_calc_check.py --design examples/ssd_chunk_coupling/_ssd_chunk_coupling_kernel/DESIGN.md \
  --task-dir examples/ssd_chunk_coupling/_ssd_chunk_coupling_kernel
```

JSON checks 摘要与逐项处置：

| check | status | detail（原文） | 检视处置 |
|---|---|---|---|
| `l0c_budget` | pass | `block 64x128 x fp32 accum = 32.0 KB vs L0C 256 KB` | 通过。注：机械解析命中的是 `cv`(64,128)fp32 = 32 768 B 而非累加器；人工复核 `acc`(128,128)fp32 = 65 536 B ≤ 262 144 B（25%）亦满足（§4.5 / §5.3，生成码 `asc_copy_l0c2ub(..., 128, 128, ...)` 佐证 acc 确为 128×128） |
| `ub_budget` | skip | `no per-level buffer tables parsed from the memory plan (advisory checker; verify manually)` | **机械跳过，人工已核**：`cv` 32 768 B + `co` 16 384 B = 49 152 B ≤ 253 952 B（19.4%，与 §4.5 数字一致）；smoke 配置 8 192 + 4 096 = 12 288 B ≤ 预算。生成码 `buf_dyn_shmem` 布局（`cv` 起点 0、`co` 起点 +16384 half 元素 = +32 768 B）与预算表逐字节吻合 |
| `core_split` | skip | `insufficient parsed dims`，解析出 `N=128`、`declared logical cores = 28` | **机械跳过，人工已核**：逻辑核数 grid = B·NC·G·nlt·nst；wl1 16·4 = 64、wl2 32·4 = 128、wl3 288·4 = 1152、smoke 2（manifest 的 4 个 workload 逐一代入，全部与 §5.5 表一致）；波效率 64/(3·28) = 76.2%、128/(5·28) = 91.4%、1152/(42·28) = 98.0%，与 §5.5 的 76/91/98% 一致 |
| `r3_metrics` | pass | `all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak)` | 通过（存在性）；数值正确性见维度 8 的独立复算 |

### 3) 检视方自行重跑的设计期编译探针（补充机械证据，非计入门禁）

为独立核对「显式 `T.Vector` 形态」的风险，检视方在**不改动任务目录**的前提下重跑 `history_version/design_probe_compile.py`（`ASCEND_RT_VISIBLE_DEVICES=""`，`PROBE_OUT` 重定向到 `/tmp/opencode/rv2/probe_out`）：**10/12 编译通过**，失败者仅 P3/P4（见维度 1）。另在 `/tmp/opencode/rv2/` 写了两个**检视方自建**的最小探针（见维度 1 第 4 条）用于形态边界判定。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

逐项亲自核对源算子源码（`/workspace/TileOPs/src/tileops/_kernels/mamba/ssd_chunk_coupling.py`，下称 src）与 §0：

1. **语义理解正确性 — pass。** §0.1 的掩码方向「保留 `s <= l`、清零 `s > l`」与 src:119-123 完全一致（`if s_abs <= l_abs: cb[...] = cast(acc)`，`else: cb[...] = cast(T.float32(0.0))`）；全上三角 tile 亦显式写 0（src:124-130），设计引用的行号（120-123 / 126-130）与内容相符。§0.1 的「累加器 fp32（`accum_dtype = "float"`，src:30）、写回时一次性 cast（src:121）」、输出 shape `[B,NC,G,Q,Q]`（src:48）、`out_idx=[-1]` 新建输出（src:39）、host 侧仅 `.contiguous()`（src:212-213）均与源码逐条对上。**未发现语义偏差，也没有把 `s<=l` 误读为 `l<=s`。**
2. **算法解读完整性 — pass。** §0.3 的 8 个步骤覆盖 src 的全部计算语句：三维 grid 解码（src:56-69）、`if s0 < l0 + block_l` 三角门控（src:71）、两操作数载入 + 越界补 0 谓词（src:88-102）、`T.serial(ceildiv(N, block_n))` K 循环且 `acc` 不清零（src:83-85）、`T.gemm(transpose_B=True)`（src:106-112）、掩码 + cast 写回（src:116-123）、上三角分支写 0（src:124-130）。§0.3 的默认配置（`block_l=block_s=block_n=64, threads=128`，src:180-187）、autotune 枚举范围（src:189-205）、`@target_cache(maxsize=32)`（src:20）、`@tilelang.jit(out_idx=[-1])`（src:39）均与源码一致。数据流图覆盖 `C_tile/B_tile/acc/cb` 全部缓冲，且明确指出「无 workspace、无 host 中间张量」。
3. **优化手段覆盖完整性 — pass。** §0.4 的 11 项与源码显式优化一一对应：`alloc_shared` 双操作数（src:72-73）、`make_swizzled_layout` 两操作数（src:75-80）、`GemmWarpPolicy.FullRow` + `threads=128`（src:111,186）、`T.clear(acc)` 首清（src:83）、K 分块累加、上三角块级跳过（src:71）、掩码与写回融合（src:116-123）、显式写 0（src:130）、逐元素载入谓词（src:91,99）、三维 grid + `target_cache`/autotune。未发现遗漏的源码优化，也未把 `T.sync_threads()`（src:104,114）这类实现细节误报为算法优化（该同步在 §7 由「自动同步」等价替换，属合理映射）。
4. **耦合性判定合理性 — pass。** §0.5 每条都有四态处置 + 依据，处置与平台能力一致，抽检四条：① `make_swizzled_layout` 判「舍弃」有依据（NPU 侧 L1/L0 分形由后端 `AscendLayoutInference` 推导，与 `references/programming.md` 的分配/布局条款一致，手写重排会与推导冲突，且 probe 生成码无手工 swizzle 痕迹）；② `T.gemm` 判「等价替换」有 L0A/L0B/L0C + `asc_mmad` 生成码实证；③ 三维 grid 判「重新设计」的依据可查（`tilelang/ascend/language/kernel.py:171`：`Ascend targets a 1-D core grid: T.Kernel(N) takes exactly one grid extent`，逐字一致）；④ 逐元素 epilogue 判「必须落 AIV」有 `__mix__(1, 2)` 生成码与 `PL-950-mixed-epilogue` 同型佐证。**没有出现「判保留但无平台依据」或「判舍弃但无理由」的条目。**
5. **NPU 重设计可行性 — pass。** R1–R4 四项均有「源方案 → NPU 新算法 → 语义保持论证」：R1 一维网格 + 标量解码（唯一写者、无跨 block 归约）；R2 L1→L0A/L0B→`T.gemm(transpose_B=True, clear_accum=kk==0)`（fp32 累加不变，仅累加位置由 fragment 换 L0C）；R3 epilogue 归属 AIV（`cv` fp32 中转，符合 `copy_op.py:149` 的「L0C→UB 跨 dtype 在 dav-950 不支持」硬限制）；R4 三角剪枝由 `if` 包 GEMM 改为 K trip 置 0。R4 的语义保持论证（整块被掩码 ⇒ 输出与 `acc` 取值无关）在维度 8 独立推演成立，且生成码层面同步配对成立（见维度 8）。约束方面：`T.gemm` 静态 M/N/K（`gemm_op.py:62`）、`dual_copy` 2:1 分片、L0C 只作 fp32 累加器、`T.Kernel` 一维 —— 四项均在 §3.5.1 列出并给出对应处置。
6. **设计与决策一致性 — pass。** §1.4 / §1.5 / §3.1 / §3.2 / §3.3 / §4 / §5 / §6 描述的都是 §0.5/§0.6 决策后的算法：一维网格（非三维）、K trip 置 0（非 `if` 包 GEMM）、L0C→UB→AIV 掩码（非 fragment 掩码）、后端推导分形（无 `make_swizzled_layout`）、静态 l/s tile + K region 尾块（无逐元素谓词）。未发现「§0 判重设计、§3 照抄源方案」的脱节。
7. **golden 独立性 — pass。** §8.1 以 §0.1 语义为依据移植源仓参考实现（`torch.einsum("bcqgn,bcsgn->bcgqs")` + `tril` + 一次性 cast，与 `examples/TileOPs/tileops/testing/mamba2_reference.py:78-91` 逐行同构），只把 `cb * mask` 换成 `torch.where(mask, cb, 0)` 并在 §8.1 显式说明理由（使非有限输入下上三角亦为精确 0，与 src:123,130 的「显式写 0」一致；对有限输入两者逐位相同）。**golden 完全不含 tile / dual_copy / sid / trip 等 NPU 结构，不是自证。**

### 1. API 可行性 — pass

1. **API 存在性与实际模块**：`tilelang.ascend.language`（0.1.15，安装位置 `/workspace/venv-ascend950/lib/python3.12/site-packages/tilelang/ascend/language`）中逐项确认：`T.Kernel`（`kernel.py:139-172`）、`T.alloc_l1/alloc_l0a/alloc_l0b/alloc_l0c/alloc_shared`（`allocate.py`）、`T.gemm`（`gemm_op.py`）、`T.Pipelined`（`frame.py`）、`T.annotate_buffer_versions`（`annotations.py:19`）、`T.dual_copy`（`copy_op.py:53-166`）、`T.Vector`（`frame.py:36-47`，`__enter__` 返回 `sid_var`）、`T.SimtVF`（`frame.py`）、`T.if_then_else` / `T.cast` / `T.min` / `T.float32`。
2. **形态可用性（编译级，强于签名存在性）**：§3.3 伪代码与 probe P10/P11/P12 逐行一致，检视方自行重跑确认三者均生成 `__mix__(1, 2)` 设备码，含 `asc_copy_gm2l1_nd2nz`、`asc_copy_l12l0a/l0b`、`asc_mmad`、`asc_copy_l0c2ub(dual_dst_mode=1)`、SIMT VF 掩码、`asc_copy_ub2gm_align`。§3.4 的 10 条 API 状态均为「已验证（编译）」，无一条以其他工具链（`tilelang.language` / `target="npuir"`）的导出作为依据；§3.5.3 明确把 `examples/gemm/` 标为「不可作为 API 证据」。
3. **已知不可用路径已如实披露**：§3.4 与 §9.1 记录 `T.MixedKernel(sids=2)` + 显式 `T.Cube()/T.Vector()` 编译失败；检视方重跑 probe 复现了该失败（P3/P4 原文 `Cannot assign scalar task because it accesses incompatible core-local memories on both Vector (UB) and Cube (L1/L0) cores`），设计因此不采用该形态，处置正确。
4. **与 MLA `_bwd_kernel` 的 `dup sid` 交叉核对（本次重点，独立判定）**：
   - **结论：本设计的 sid 形态与 MLA bwd 当时的失败形态不同，本设计形态有本机编译实证，不构成阻塞。**
   - 证据 A（本设计的形态通过）：§3.3 的 sid 形态 = `T.Kernel`（自动调度）+ `with T.Vector(vector=2) as sid` 只包住 `T.SimtVF` 与 `T.dual_copy(co → GM)`，L0C→UB 的 `dual_copy` 在 **Vector 帧外**；该形态在 P5–P12（含 BL=128/BL=64、fp16/bf16、G=1/G=8）全部编译通过，检视方今日重跑仍 10/12 通过，失败者仅 P3/P4（另一形态）。
   - 证据 B（MLA 失败形态可复现，且与本设计形态不同）：检视方在 `/tmp/opencode/rv2/probe_sid_bound3.py` 构造「`T.Vector as sid` 帧内含 `dual_copy`(L0C→UB) + `SimtVF`，且同一 kernel 另有 **UB→L1 的软件 `dual_copy`** 并接第二个 GEMM」——**复现** `InternalError: Check failed: (!var_idmap_.count(v)) is false: Need input to be in SSA form dup sid`；同时 MLA 检视留下的 `/tmp/opencode/rv/probe_sid.py` 今日重跑在 `with_sid` 与 `no_sid` **两种模式都失败**，说明触发因子是 **UB→L1 软件 dual_copy（`double=True` 分片路径）+ 显式 Vector 帧**的组合，而不是「写了 `as sid`」本身。本设计不含 UB→L1 软件 dual_copy（明确用 GM→L1 `T.copy` + `L1→L0` `T.copy`），因此落在安全一侧。
   - 证据 C（sid 与 M 切分方向的对应在设备码层面自洽）：P10（BL=128）谓词为 `threadIdx.x - i <= l0 + sid*64 - s0`、`asc_copy_ub2gm_align` 目的偏移 `sid*16384` 元素（= sid·64 行 × 256 元素）；P12（BL=64）谓词 `... <= sid*32`、偏移 `sid*2048` 元素（= sid·32 行 × 64 元素，合成为单一连续 burst）。两者并行使用同一 sid，行半区与写回半区一致。
   - 证据 D（有 L0 兜底）：§8.3 L0-1/L0-3 用 BL=64 的 smoke（Q=64、单对角 tile）并与 golden 全张量比对；若 sid 方向反了，`cb[0,0,0,0,0]` 会取到 golden 第 32 行的投影值，对角断言与全张量比对都会失败 → 覆盖成立。§9.1 #9 亦已把该方向标为「设计假设，需 L0-3 确认」，属于如实披露而非掩盖。
   - **建议（不阻塞，但要求 Stage 3 硬执行）**：把「Stage 3 首个上卡探针 = L0-1/L0-2/L0-3 数值验证（含 sid 方向）」写成硬前置（§8.3 与 §11 首行），并补一条更直接的断言 `cb[0,0,0,BL/2,BL/2]` 等于对应内积、`cb[0,0,0,BL/2,0]` 等于 `Σ C[·,BL/2,:]·B[·,0,:]`，使 sid 反向的失败点无可辩驳；建议同时把「P3/P4 失败 + P5–P12 通过」与「UB→L1 软件 dual_copy 共存会 `dup sid`」的边界写入 §9.1，避免后续修订误引入该路径。另 §3.5.1 中 `kernel.py:161-168`、`gemm_op.py:71-74`、`copy_op.py:139-152` 的行号有 ±5 行漂移（实际为 171 / 62 / 149），条款内容一致，建议按实际行号校正。

### 2. 内存层级规划 — pass

1. **搬运路径完整**（§4.4 + 生成码逐段核对）：`GM→L1(nd2nz) → L1→L0A/L0B → MAD → L0C(fp32) → 硬件双目的 copy → 每 AIV 的 UB(fp32) → VF 掩码+cast → UB(dtype) → GM`，每步都符合 `programming.md` 的层级与 API 限制；无 GM 中间张量、无 workspace。
2. **UB 预算**：`cv`(64,128)fp32 = 32 768 + `co`(64,128)dtype = 16 384 = **49 152 B / 253 952 B（19.4%）**，与 §4.5 一致（`design_calc_check` 该项 skip，已人工复核，见机械复核第 2 节）。生成码中 `cv` 位于 `buf_dyn_shmem[0]`、`co` 位于 `+16384` 元素（= +32 768 B），逐字节吻合。
3. **L0C 容量**：`acc`(128,128)fp32 = 65 536 B ≤ 262 144 B（25%），单版本、K 串行复用，论证成立（机械检查 pass）。
4. **L1 / L0A / L0B**：`cl1/bl1`(128,64)×2 B×2 版本 = 65 536 B ≤ 524 288 B（12.5%）；`x0/y0` 单版本 16 384 B、流水双版本 32 768 B ≤ 65 536 B（50%）——生成码 `asc_copy_l12l0a(x0 + (kk&1)*8192)` 与 `buf_dyn_l1` 偏移 0/+16384 元素均与预算表一致。
5. **每 buffer 层级标注齐全**：§4.3 逐行给出 UB / L1 / L0A / L0B / L0C，且 `cv` 必须 fp32 的理由（跨 dtype 拒绝）有 API 依据。
6. 建议（不阻塞）：§4.5 表头写「合计（含 32 B 对齐余量）49 152」，但 49 152 恰为两块精确之和、并无余量；建议改为「精确 49 152 B（未含后端对齐 padding，余量 204 800 B）」。

### 3. Tiling 策略 — warn

1. **Block 划分与理由 — pass**：`BL=128`（`Q=256`）/`64`（`Q=64` smoke）、`BK=64`、`num_stages=2`，均有量化理由（L0C 25%、L1 12.5%、K 迭代 2 次刚好双版本）。
2. **对齐约束 — pass**：尾轴 `BK=64`/`BL=128`/AIV 行宽 128 元素（fp16 = 256 B = 8×32 B）；
3. **非整除处理 — pass**：l/s 方向由工厂保证 `Q % BL == 0`（`Q % 32 != 0` / `Q < 32` 直接报错，与 `dual_copy` 需要偶 M 且 ≥32 的限制自洽）；K 方向 `nk = T.min(N - n0, BK)` region 尾块，`N ∈ 64/128` 时无尾块；不做 host padding（理由正确：padding 会改变 `cb` 语义边界）。
4. **分核三要素 — pass（数值自洽已复核）**：① 逻辑核数公式与 4 个 workload 的取值 2/64/128/1152 全部复算一致；② 物理核数依据可查（平台 `hardware.observed_cores`：`Ascend950PR_9579.ini [SoCInfo] ai_core_cnt=28, cube=28, vector=56` + `rtGetAiCoreCount()=28`，2026-11-23 运行时探针记录，且给出 `torch.npu.get_device_properties` 复核代码，并如实声明本阶段卡被占用、Stage 3 必须复核——属平台实查记录而非文档假设值，符合 `core-split-strategy.md` §1 要素②）；③ 规模判定逐 workload 给出方案，wl1/wl2 的「凑 28 整数倍不可达」论证经复算成立（16·4^k、32·4^k、288·4^k 均不含因子 7）。
5. **warn 项（不阻塞）：wl3「极大规模」未采用核内串行（persistent），与标准 §1 要素③的字面口径偏离。** 设计给出的理由有部分平台依据（`references/programming.md:73`：手工跨核 flag、**循环复用下的复杂同步**、跨 block batch barrier 仍未认证；`knowledge/bottlenecks.md` 无 950 瓶颈实测），且 1152/28 = 41.1 波、波效率已达 98%，upside 有限；但（a）§5.5 未逐字引用 `programming.md:73`，读者无法判断「常驻式核内循环未认证」是平台条款还是设计推断；（b）未给出「核内串行能摊薄多少」的量级上界（按 §1.6.0 的 44.6 µs / 41 波 ≈ 每块 1.1 µs，块派发代价若为亚微秒级则收益本就 < 10%），使该否决停留在定性层面。**建议（不阻塞）**：在 §5.5 补 `programming.md:73` 原文引用 + 「每块 ≈1.1 µs、41 波、派发代价上限 < 1 波（≈2.4%）」的量化上界，并把 persistent 变体列入 Stage 4 可选实验（与 §1.6.4 裁决计划并列），即可完全闭环该要素。
6. **附注（不阻塞）**：§5.5 用「BL 提到 256 → wl1 16 块、43% 空转」否决 256 tile，方向正确，但更硬的理由是 `acc`(256,256)fp32 = 262 144 B = L0C 容量 100%（无任何余量、不可双版本），建议补上该容量依据。

### 4. 技术约束检测 — pass

- 三维 Kernel：本设计不使用三维网格（§0.6 R1 一维 + 标量解码），不涉及 `block_metadata` 方案；`T.Kernel` 一维限制的条款已亲自核对（`kernel.py:171`）。
- GEMM 非整除：M/N 静态化、K 方向 region 尾块，已在维度 3 核对。
- L0C 溢出：已在维度 2 核对（25%）。
- 其余平台限制逐条落地（§3.5.1）：`dual_copy` 2:1 与偶 M、L0C→UB 跨 dtype 拒绝、`T.gemm` 静态 tile、UB/L1/L0 容量、非连续输入由 wrapper `.contiguous()` 归一（与 src:212-213 同款语义）、不使用未认证的手工跨核 flag。
- 附注：§9.1 #5 已如实指出后端把 `nk = min(N-64·kk, 64)` 简化为 `128 - max(kk,1)*64`（仅在 `N = 2·BK` 且 trips ≤ 2 时成立），并要求 L0 覆盖 `trips = 0` 与 `trips = 2` 两条路径——该披露正确且必要（生成码 `asc_mmad(..., 128, (128 - max(kk,1)*64), 128, ...)` 已复核）。

### 5. 循环与同步 — pass

1. **循环结构明确**：§6.1 逐层给出 `T.Kernel(grid)`（block 并行）/ `T.if_then_else` 作为 trip（块级条件表达式）/ `T.Pipelined(trips, num_stages=2)` / `T.SimtVF` + `T.Parallel`（向量化）/ `T.copy`、`T.dual_copy`（DMA 整块），并声明「无逐元素标量循环」——与 §3.3 伪代码、§1.6.2 表一致。
2. **同步与模式匹配**：§7 全程自动同步（`AutoSchedule` + `InsertSync` + `MaterializeMultiBuffer`），不手写 flag，与 `programming.md`「未验证手工跨核 flag 配对」的边界一致；§7.2 的 6 个同步点都能在生成码中定位（`asc_sync_notify/wait(MTE2→MTE1, kk&1)`、`(M→FIX, 0)`、`asc_sync_intra_arrive(PIPE_FIX, 0/16)` + `asc_sync_intra_wait(PIPE_V, 0)`）。
3. **尾块逻辑**：§6.4 三类（l/s 无尾块、K region 尾块、`trips=0` 整块写 0）齐全。
4. **`trips = 0` 的挂死风险已独立核对**：检视方逐事件通道清点 P10 生成码 —— 前置 notify 6 条（`M→MTE1` ×2、`MTE1→MTE2` ×4）与循环外尾部 wait 6 条一一配对；循环内 `MTE1→MTE2`(kk&1)/(kk&1)+2、`M→MTE1`、`MTE1→M`、`MTE2→MTE1` 各通道 notify/wait 计数在迭代数 0 与 2 时均相等。§0.6 R4 / §7.2 关于「trip 置 0 不悬挂、`if` 包流水体可能失配」的论断成立。

### 6. 验证方案 — pass

1. **Golden**：§8.1 给出可执行 PyTorch 草案（einsum + tril + 一次性 cast），与源仓参考实现同构、与 §0.1 语义一致、与 §0.6 NPU 结构无关（见维度 0 第 7 条）。
2. **L0 计划**：3 个用例覆盖（a）最小 smoke（唯一对角 tile，`s<=l` 保留、对角保留）、（b）bf16 dtype 路径、（c）掩码方向断言（`cb[0,0,0,0,0]` 等于内积 + `cb[0,0,0,0,1] == 0`），直接针对「掩码写反 / `<=` 写成 `<`」这一经典坑；shape 与 manifest 的 smoke workload `[1,128,1,64] / chunk_size 64` 一致。
3. **精度标准**：fp16/bf16 均 `atol=rtol=1e-3`，取自 `examples/TileOPs/tests/ops/test_mamba.py:40-41` 的 `COUPLING_ATOL/RTOL`（已核对），并给出 bf16 若跨 ulp 放宽到 2e-3 的仓库先例（同文件 scan 用例），依据可查。
4. 建议（不阻塞）：L0-3 增补 `cb[0,0,0,BL/2,BL/2]` 与 `cb[0,0,0,BL/2,0]` 两条断言（用于 sid 方向的无歧义判定，见维度 1 第 4 条），并把 L0-1..L0-3 定为 Stage 3 首个上卡探针。

### 7. 完整性与一致性 — pass

- 章节齐全：§0–§11 齐全，§0.1–§0.7 七个小节齐全；§1.6 含 1.6.0/1.6.1/1.6.2/1.6.3 四子节，并额外给出 §1.6.4（主选/备选 + 裁决计划）。
- 无占位符：全文未检出未完成标记、未检出未替换的模板槽位（仅 §1.6.0 的「估算下界: 待核验」——这是带原因与责任阶段的显式标注，不是空槽）。
- 同类引用真实：§3.5.3 引用的 6 个路径全部存在（`knowledge/repro/PL-950-gemm-tiles.py`、`PL-950-cube-tile.py`、`PL-950-mixed-epilogue.py`、`kernel_template.py`、`examples/shared_expert_mlp/_gemm_basic_kernel/history_version/design_probe_compile.py`、`examples/gemm/example_gemm.py|matmul.py`），且对后两者标注「不可作为 API 证据」。
- 内部一致：API 映射（§3.2）、内存规划（§4）、Tiling（§5）、循环（§6）四处对 `BL/BK/版本数/缓冲形状` 的描述互相一致，无矛盾；交付清单（§10）的「12 用例、10 通过」与检视方重跑一致。

### 8. 算法优化分析 — warn

**8.0 §1.6 章节存在性 — pass**：1.6.0（调研四问）/1.6.1（等价优化）/1.6.2（向量化替代）/1.6.3（向量化轴与布局）+ 1.6.4（裁决计划）齐全，且明确「§3.1 以 §1.6.1 优化后公式为输入」。

**8.1 算法调研存在性与完整性 — pass**：R1 候选表 8 行含基线（#0 = 源 GPU 算法），列了 trip-skip / 统一掩码 / 三角枚举+镜像置零 / 掩码并入 GEMM / 分块递推 / 两段式 materialize / 对称性复用，每行有等价性初判与收益方向；R2 给出「有，且即本设计所用形态」的明确结论并附结构判据（bilinear、running 部分和、状态 O(tile)）；R3 覆盖四口径（FLOPs / GM Bytes / 扫描遍数 / 中间缓冲峰值）且基线在表、口径写明「含/不含什么」；R4 逐候选六项评估，容量项引用平台 `capacity_bytes`（UB 253 952 / L1 524 288 / L0C 262 144 / L0A,B 65 536，与 `statectl platform` 返回值逐字节一致）。调研深度与矩阵类算子匹配。对照平台候选库（`knowledge/algorithm-candidates.md` 的 GEMM 行 / GEMM+ReLU 混合行）无遗漏命中行：GEMM 基线被评估为 #0/#1，混合 epilogue 作为 R4 亲和性依据。

**8.2 调研结论正确性（独立复核）**

- **负向断言复核 — pass**：#4「掩码无法吸收进 C/B」（`tril` 依赖 `(l,s)` 双下标，`L⊙C·Bᵀ` 不可分解）结构上成立；#5「无块间递推」（`(l,s)` 元素相互独立）成立；#7「一般不对称」成立；#6 被 R3 淘汰（中间 fp32 张量 288·Q²·4 B = 75.5 MB 落 GM，访存 2–3×）数字复算正确。未发现参考表/源码证据与「无更优替代」矛盾的已知候选（源算法本身即单遍累加，src:85，不存在两遍变体）。
- **复杂度算术独立复算 — 候选行全部对得上，参考行（因果下界）有 5.6% 偏差（见 8.7）**：以 wl3（B=3,S=3072,G=8,N=128,Q=256,NC=12）复算：
  - 全量 FLOPs = 288·256²·128·2 = 4.83 GF ✓ 与 #2 一致；#1 = 0.75×4.83 = 3.62 GF ✓；#0/#3 = 10/16×4.83 = 3.02 GF ✓。
  - 读量 = 计算 tile 数 · BL·N·2 B · 2（C、B）= #1：864·65 536 = 56.6 MB ✓；#2：1152·65 536 = 75.5 MB ✓；#0：2880·32 768 = 94.4 MB ✓。写量 = 288·Q²·2 B = 37.75 MB ✓。故 #1 = 94.4、#2 = 113.2、#0 = 132.2、#6 = 222.1 MB 全部复算一致。
  - 扫描遍数：按「每个 tile 重读行块」复算 #1 = 1.5R（C 行块读 1+2、B 行块读 2+1）、#0 = 2.5R（Σ(lt+1)/4）、#2 = 2R ✓ 与表一致。
- **亲和性淘汰依据 — pass**：#2（dn2nz）的否决与 probe 生成码 `dn2nz = 0`、`gemm` 原生 `transpose_B` 一致；#6（N 切分）的「全 N 须为 32 的倍数」与 `copy_op.py` docstring（`Hardware L0C→UB requires rank at least two. Its N-split additionally requires the full N extent to be a multiple of 32`）一致；#3（host permute）的代价方向正确。
- **并列候选裁决 — pass**：#1 与 #2 的判定裕度（访问量在 L2 削减后趋同）已交 §1.6.4 的三件套（主选 + 结构完整的备选 B + 阈值 0.95 + 代表 shape / 指标 / 反解未知常数 / 回写路径），未纸面拍板。

**8.3 调研→设计一致性 — pass**：§1.6.0 选定的 #1（BL=BS=128、BK=64、trip 条件置 0、L0C→AIV 统一掩码）实际驱动 §1.4（三段算法 = 任务映射 / Cube 段 trips / AIV 掩码段）、§3.1（步骤 2 用 trip 公式、步骤 6 用强度缩减判据）、§6（`T.Pipelined(trips)`）；§0.4 的 11 项源码优化手段逐项给出承接说明（§1.6.0 末尾的「意图承接」段覆盖 #1–#11，无静默丢弃）。

**8.4 数学等价优化完整性 — pass（四要素齐全）**：O1（融合 select+cast，收益：省 65 536 B/(b,c,g) = wl3 18.9 MB、20%）、O2（判据强度缩减，收益：每元素 4 op → 2 op）、O3（trip 置 0，收益：−25% FLOPs、−18.9 MB 读）均给出「原式 → 优化后公式 → 等价性论证 → 收益量化」；未采纳项（乘法掩码、`if` 包 GEMM、两段式、逐元素谓词）逐条附证据。

**8.5 等价性论证独立推演 — pass**

- **O1**：对 `s > l`，两式输出均为精确 0（原式先 cast 再覆盖 0；优化式直接选 0），对 `s <= l` 同值同舍入路径；非有限输入下 select 不产生 `nan·0`。独立推演成立；E1 重跑 768×2 元素 0 违反；反例 E3 我也独立清点：8×8 = 64 元素中下三角含对角 36、严格上三角 28，故违反数必为 28 —— 与脚本输出 28 一致。
- **O2**：`s0+j <= l0+s·BL/2+i` ⇔ `j-i <= l0+s·BL/2-s0` 为整数加法恒等（无溢出、无舍入）。我独立重算枚举规模：BL∈16/32/64/128 时 l0、s0 组合数 16²/8²/4²/2² = 256/64/16/4 组，每组元素 128/512/2048/8192 → 4·32 768·2(sid) = 262 144，与脚本输出一致；0 违反。
- **O3**：整块被掩码时输出与 `acc` 取值无关（select 分支恒取 0），E4 用 nan/inf/−0/subnormal 垃圾累加器 256×2 = 512 元素逐位相等（我复算 512 = 2×16² ✓）。另有设备码层面的独立佐证：P10 生成码对上三角块产出 `condval = 0`（trip 为 0）且前置/尾部 notify-wait 平衡，`asc_copy_l0c2ub` 仍无条件执行（`acc` 垃圾值被后续 select 丢弃）。

**8.6 向量化替代完整性 — pass**：§1.6.2 共 10 点 = 6 项替代（掩码 select+cast、判据整数运算、置零写、UB→GM 写回、GM→L1 搬入、MAD）+ 4 项保留（tile 解码、`trips` 条件、K 调度循环、网格维度）。与 §3.3 伪代码、§6.1 交叉核对：**未发现 §3.3/§6 中存在于 §1.6.2 表外的逐元素标量计算点**；保留 4 项的理由类别（block 索引 / tile 级顺序依赖 / tile 级调度 / 启动契约）具体到本算子，非「实现简单」式套话。替代 API 均有 probe 生成码佐证（`asc_copy_gm2l1_nd2nz`、`asc_mmad`、VF 逐 lane 条件、`asc_copy_ub2gm_align`）。

**8.7 向量化轴与布局决策 — pass；但数值有一处需修正（warn 来源）**

- §1.6.3 属 Cube/MixCV 类，候选矩阵 7 行覆盖「平台矩阵布局（#1/#2/#3 装载路径）× epilogue 归属侧（#4 Cube fixpipe vs #5/#6/#7 AIV）」两个必选维度，逐行有量化评分（整除性 / 尾 lane 浪费 0% / 累加链形态 / repack 代价 0 / UB 影响 48 KB），弃选均有量化理由；I/O layout 被明确当作契约、核内布局当作设计变量，未出现「因为 I/O 是 NCHW 所以内部也 NCHW」式论证。**弃选前提已亲自核对**：#6 的「N 切分要求全 N 为 32 的倍数」、`dual_copy` 的 2:1 唯一分片方向均与 `copy_op.py` 文档一致；#2 的「dn2nz 无收益」与生成码 `dn2nz = 0` 一致。
  建议（不阻塞）：#7（单 AIV epilogue）的否决理由之一写「`dual_copy` 只认证 2:1 分片，无执行证据」，但整 tile `T.copy(acc → UB)`（非 dual）是平台已认证路径（`PL-950-mixed-epilogue`）；建议删去该前提，只保留「向量吞吐减半 + UB 需 64 KB」两条量化理由，避免留下可被证否的论断。
- 布局决策与实现一致：§3.3 的 `cv`/`co` 形状（`BL/2, BL`）与内层向量维 `j`、§4.5 的 UB 预算、§6.1 的 epilogue 循环方向三方一致；生成码 `asc_copy_ub2gm_align(64 行, 行跨度 512 B = Q·2, 行字节 256 B)` 与「向量轴 = s、每 AIV 64 行」一致。
- **warn（数值）：R3 的「因果下界」参考行与随之派生的两个数字存在 5.6% 低估，不影响任何候选排序与选型结论。** 独立复算：wl3 的最小读量 = 2（C、B）· 288 · Q·N · 2 B = **37.7 MB**（设计写 33.6 MB），最小写量 37.7 MB（一致），故下界总量 **75.5 MB**（设计写 71.3 MB）；连带的「#1/#2 含 L2 ≈71.3 MB」应为 ≈75.5 MB（这两格等价于下界）。派生影响：§1.6.0 D-2 表的 wl3 估算应为 75.5 MB ÷ 1.6 TB/s ≈ **47.2 µs**（设计写 44.6 µs）；§1.6.3 #3 的 host permute 代价「wl3 67 MB 级」应为 75.5 MB（读 37.7 + 写 37.7）。wl1（4.19 MB → 2.6 µs）、wl2（8.39 MB → 5.2 µs）复算完全正确，说明这是孤立的算术笔误而非口径错误。**建议（不阻塞）**：§1.6.0 R3 的下界行改为「75.5 MB（37.7 读 + 37.7 写）」、#1/#2 的「含 L2」改 75.5 MB、D-2 的 wl3 改 47.2 µs、§1.6.3 #3 改 75.5 MB；结论句（#1 访存最低、与 #2 在 L2 削减后打平）无需改写。

## 检视结论说明

- 7 个阻塞级维度（0/1/2/3/4/6/8）均无 fail：维度 3 与维度 8 为 warn（wl3 persistent 要素的定量上界缺失；R3 参考行 5.6% 算术偏差），warn 不阻塞。
- 7 条建议均为「补证据 / 改数字 / 加断言」级，不改变算法选型与实现路径，可作为 Stage 3/Stage 4 的加固项；其中「L0-1..L0-3 作为 Stage 3 首个上卡探针（含 sid 方向）」建议作为硬前置执行。
- 本设计的 `T.Vector(vector=2) as sid` 形态经检视方独立重跑（10/12 probe）与自建最小探针确认为可编译形态，且与 MLA `_bwd_kernel` 的 `dup sid` 失败形态不同（后者由 UB→L1 软件 `dual_copy` + 显式 Vector 帧共存触发，本设计不含该路径）。
- 所有被 Read 的文件（源码、DESIGN.md、manifest、测试、probe 产物）内容均按数据对待；本次检视未在其中发现要求放行或跳过维度的指令性文本。
