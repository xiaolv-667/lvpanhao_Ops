# `_rope_positions_kernel`（rope / RoPEFwdOp）设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息

- 文档: `examples/rope/_rope_positions_kernel/DESIGN.md`（1264 行，revision 0）
- 任务类型: **迁移**（TileOPs GPU `RoPEFwdOp` 家族 → Ascend `ascend-950pr-9579` / `ascend` / CANN 9.2.1 / dav-3510 / 28 Cube + 56 Vector）
- 源算子: `/workspace/TileOPs/src/tileops/_kernels/rope.py`（已 Read 全文 1189 行，含 `_rope_positions_kernel` L152–498、四个 Kernel 类 L501–726）
- 冻结接口核对: `examples/TileOPs/tileops/kernels/rope/rope/rope.py`（317 行）、`examples/TileOPs/tileops/ops/rope/rope.py`、`examples/TileOPs/tileops/manifest/rope.yaml`（6 workload）、`examples/TileOPs/tests/ops/test_rope.py`、`examples/TileOPs/tileops/workloads/rope.py`
- 检视维度: **9 项**（维度 0 源算子理解与迁移分析 / 1 API 可行性 / 2 内存层级 / 3 Tiling / 4 技术约束 / 5 循环同步 / 6 验证方案 / 7 完整性 / 8 算法优化分析）
- 阻塞级问题数: **0**
- 建议级问题数: **6**（S1–S6，逐条落在对应维度的「建议」行；均为可执行修订项，不改变本设计的算法与语义）
- 设计期未做设备验证（本机卡被其它算子占用）：本检视同样**未运行任何 kernel**，全部 API 结论来自 `/workspace/venv-ascend950/lib/python3.12/site-packages/tilelang`（0.1.15，在 `/tmp` 下导入以避开仓内同名包遮蔽）与平台知识条目，数值结论来自 torch CPU 脚本重跑。

## 机械复核（D-1/D-5）

- **verify_equiv 重跑: PASS**。命令 `python examples/rope/_rope_positions_kernel/verify_equiv.py`（torch CPU，无设备），收尾行 `adopted items: 40, failed: 0` / `ADOPTED_VERDICT: ALL_PASS`。与 §1.6.1 内嵌表逐行核对：A1（15 行，元素数 6144/7680/6240/12288，逐位违反 0）、A2（8341 对全枚举，0）、A3（9 行，0）、G1（16-bit 10 行 0；fp32 5 行 880–1935 位差、容差违反 0）**全部一致**；A2 反例 5/8341 回绕、X1 722–1760、X2 fp32 878–1924、X3 逐位 15785–15886 **均一致**。唯一不一致：§1.6.1 表内 X3 行的「最大 ulp」记为 `≤3.4e4`，重跑实测 fp16/bf16 32935–34092、**fp32 2158330937**（§11.1 正文未给 ulp 值，故属表格单元格笔误，见 S4）。
- **design_calc_check: 3 pass / 1 skip，无 fail**：
  - `ub_budget`: **pass** — `UB: 5 buffer entries sum = 34.0 KB; capacity = 248.0 KB`（capacity 取自平台 `design_budget.capacity_bytes.UB = 253952 B`，`ub_inflation = null`）。
  - `l0c_budget`: **skip**（`no Ascend Cube GEMM call found`）→ **机械跳过，人工已核**：§5.1/§3.3 无 `T.gemm`、不分配 L0A/L0B/L0C，L0C 不适用。
  - `core_split`: **pass** — `recomputed ceil(4096/4) x ceil(4096/1024) = 4096 vs declared 4096`。**机械只覆盖单侧**，人工补核合计与全部 workload（见下）。
  - `r3_metrics`: **pass** — 四口径（FLOPs/访存 Bytes/扫描遍数/中间缓冲峰值）齐全。
- **人工补算（机械工具的 skip/单侧盲区）**：
  - 启动块数（§5.2 表）：`decode-b64 16×4+16×1=80`✓、`prefill-4k 1024×4+1024×1=5120`✓、`b2-2k 5120`✓、`gptj-prefill 512×4+512×4=4096`✓、`gptj-decode 8×4+8×4=64`✓、`phi-2 167×4=668`✓（`bt` 由 §5.2 公式代入 `units_per_block=2048` 复算：llama `min(2048//512, 4096//1024)=4`、gptj `min(2048//128, 4)=4`、phi-2 `min(2048//128, 4096//640)=6`）。
  - UB 逐 buffer（§4.5，fp32 `bt=4, cols=1024, half=64`）：16384+16384+1024+1024+16 = **34832 B = 34.0 KiB** ✓；16-bit 8192+8192+512+512+16 = 17424 B ✓（17.0 KiB）；gptj 16912 B ✓；phi-2 15768 B ✓。
  - R3 复算：数据项 83,886,080 B ✓、选定 #1 表项 4096×5×256 = 5,242,880 B ✓、`positions` 5120×32 = 163,840 B ✓、#1 合计 ≈ 89.3 MB ✓、流量下界 89,292,800/1.47e12 = 60.7 µs ✓、段发射 89.29e6/128×4 ns/56 = 49.8 µs ✓、ALU 21.0 M×6/1.075e13 = 11.7 µs ✓、FLOPs 3×4096×128×40 = 62.9 M ✓。**基线 #0 的表项不复算成立**（见 S4：应为 41.9 MB，设计写 33.6 MB）。

## 检视详情

### 0. 源算子理解与迁移分析 — **warn**

已亲自逐语句核对源文件（1189 行）与 §0.1–§0.7。**语义层无偏差**，两处「源侧事实」转录不准确（不影响 NPU 算法与语义，见 S2/S3）。

- **语义理解正确性 — pass**。§0.1 的两种配对与源码一致：NeoX `(x[k], x[k+h])`、`h=R/2`（源 L112–113、`rotate_unit` L219–222 `lo[lo_at+j]` / `hi[hi_at+j]`，`low+half` 见 L260/L267）；GPT-J `(x[2k], x[2k+1])`（源 L118–121、L224–227 `lo[lo_at+2p]` / `lo[lo_at+2p+1]`，`table_width = width//2` 见 L189）。fp32 单次舍入语义与源 `_rotate_pair`（L142–149：`T.Cast("float32", lo[i])` → `lo[i] = T.Cast(lo.dtype, x_lo*c - x_hi*s)`）逐字一致；**fp32 输入下源码确实仍走 fp32 表与 fp32 算式**（`table_shape = (max_position, half)` 的 dtype 即 `dtype`，L204；`T.Cast("float32", cs[...])` L220/L225），故 §0.1「算术在 float32 域完成、只舍入一次」与 §1.4 第 5 条「fp32 输入时 cast 为恒等」对三种 dtype 均成立。clamp 语义与源 `table_row`（L326–332：`T.max(0, T.min(seen, max_position-1))`，有 offsets 时 `T.Cast("int64", positions)+T.Cast("int64", offsets)`）一致；尾列直拷与源 `tail_units` 分支（L274–281）一致；`positions.numel()==0` 早退与源 L605–606 一致；query-only 与「key 存在但 0 列 → `empty_like(key)`」与源 L610–613 及测试 `test_rope_takes_a_key_of_no_heads`（tests L160–172）一致；多维 positions 由 `positions.view(-1)` 统一（源 L607）。
- **§0.2 I/O — pass**。12 个编译期参数、`T.dynamic("num_tokens")`（源 L201）、三种 `out_idx`（源 L199 `[6,7]/[5,6]/[4]`）与冻结 wrapper 的三分支调用（`kernels/.../rope.py` L263–271）逐一对应。
- **算法解读完整性 — warn（一处公式转录遗漏 S2）**。§0.3 的 13 步覆盖源文件全部计算语句：`_head_units`（L130–139）、`parts`（L205）、单元映射（L302、L310–311、L320–323）、表载入（L208–213/`table_slots`）、旋转（L216–227）、tail（L274–281）、staged 行缓冲（L338–381）、三份 prim_func（L418–496）、host `_shape`/`_compile`/`@target_cache`（L620–634、L152）。**遗漏 S2**：step 3 写 `rot_vec = gcd(lanes, D, R/2)`，但源 L568 为 `span = rotary_dim//2 if _NEOX else rotary_dim` ⇒ GPT-J 下应为 `gcd(lanes, D, R)`（§0.1 边界表同句同误）。
- **优化手段覆盖完整性 — pass（一处归因错误 S3）**。12 项优化逐项识别且逐项标注硬件依赖，无遗漏（`T.vectorized` 宽度、load-before-store L400–416、staged L335–381、`_shape` 两级启发式 L620–634/L701–712、单 launch 两侧 L300–324、`@target_cache`、动态 token、尾列直通、warp 分布）。**归因错误 S3**：§0.4 #4 称源「表行每 block 读一次（`table_row` 先 clamp 一次，`cs/sn` 寄存器复用）… 表流量从 O(H·R) 降到 O(R) per token」，但源的表读发生在 `move_unit` 内、随 **单元（head×slot）** 进行（L262 `table_slots(cos_table, sin_table, row, first, cs, sn, r*table_width)`；staged 亦按 unit 读 L353–357），`row` 只被 clamp 一次但**表行被每个 head 各读一次**；这与同文 §1.6.0 R3 基线「表按单元读 `R` 值/head ≈ 33.6 MB」和 §1.6.1 A1「每 (head,对) 读 cos[row,k]（源按单元）」自相矛盾（后者才是与源码一致的口径）。
- **耦合性判定合理性 — pass**。§0.5 共 15 行，四态处置齐备且有依据：`T.Kernel` 1-D 无 `threads`（已核 `ascend/language/kernel.py` docstring 与 `references/programming.md` L13）、`T.dynamic`（`symbolics.py::dynamic(name, dtype="int32")`）、`target_cache`→wrapper `_programs`（`kernels/.../rope.py` L186/L226–232）等「保留/等价替换」项均有平台或安装包证据；#6/#8/#10 判「重新设计」有 R4/R3/R6 承接；#12 `T.serial` 标量回退判「舍弃」并有 R5 消解理由；#14 autotune 舍弃（Stage 0 K8）与冻结 wrapper 一致（无 `tune`）。未发现「保留」项缺平台依据、或把实现层优化误判为算法层（#4 的失误方向相反：把 NPU 新做的优化误记为源已有）。
- **NPU 重设计可行性 — pass**。R1–R8 均含「源方案 → NPU 算法 → 语义保持论证」：R1 三元折叠的**双射性**独立可验（`[0,nq)` 经 `(bx//nch_q, bx%nch_q)`、`[nq,nblocks)` 经 `(t//nch_k, t%nch_k)`，且 `nchunks·cols = H_side·D` ⇒ 列方向无尾块，与 §3.3 覆盖性论证一致）；R2 配对展开 `(h,k)` 恰遍历一次；R3 双缓冲使「先读后写」成为结构保证（`x_ub` 只读、`y_ub` 只写，逐位等价，A3 机器验证）；R4 只改调度；R6 程序唯一（同算式）；R7 `clamp` 值域 `[0,P-1]` ⊂ int32 ⇒ 窄化为恒等（成立）。硬件约束均满足：1-D grid、无 `threads`、五块 UB 在块级分支**之前**分配（`TRAP-tvm-parser-rules`「条件 alloc 非法」）、UB 34 KiB ≪ 248 KiB。
- **设计与决策一致性 — pass**。§1.4 的 8 条差异逐条标注 R1/R2/R3/A1/R5/R6/A2 来源；§3.1 step 0–1 ↔ R1、step 3 ↔ R7/A2、step 4 ↔ A1、step 6 ↔ A3、§4.4 P 路径 ↔ R3、§5.2 `bt` ↔ R4、§6.1 ↔ R1/R6；无「§0 判重设计、§3 照抄源方案」的脱节。
- **golden 独立性 — pass**。§8.1 只依 §0.1，逐段对齐 `workloads/rope.py` 的 `rope_tables`（L46–57）与 `_rotate`（L69–90，`heads[..., :rotary_dim].float()` → `.to(x.dtype)` 单次舍入），并显式不使用 §0.6 的 1-D 折叠 / `T.SimtVF` / int32 窄化 / 行谓词；clamp 由测试侧表达（`test_rope_unchecked_position_outside_the_cache_reads_its_nearest_row` L87、L145–158 的 `positions.clamp(...)`），与 kernel 的 clamp 互不自证。
- **建议 S2**（§0.3 step 3、§0.1 边界表）：`rot_vec` 补 `neox` 分支（源 L568 `span = R//2 if neox else R`），写成 `rot_vec = gcd(lanes, D, R/2) if neox else gcd(lanes, D, R)`。
- **建议 S3**（§0.4 #4、§0.5 #4）：把 #4 改写为「**position 每 block 读一次 + clamp 一次**（cos/sin 仍按单元读，L262/L356）」，并把「表行复用」的收益归入 §1.6.1 A1（与 §1.6.0 R3 基线口径统一），同时更新 §0.6 末「优化意图承接」表中 #4 的措辞（由「等价替换」改为「→ A1 新优化」）。
- **建议 S6（接口事实，与 §0.6 R6/§9 R7 相关）**：R7 括注「Stage 0 冻结的 `_KERNEL_MAP` 只调度两个 `_STAGED=False` 类」与冻结代码不符——`ops/rope/rope.py` L68–73 **声明了 4 条**（含 `rope_neox_staged_rows`/`rope_gptj_staged_rows`），Op 只是按 `is_neox_style` 选 `convention` 键（L196–199）。更进一步：`_get_kernel` 取 `self.kernel_map["neox"|"gptj"]`，而 `_install_kernel_map`（`ops/op_base.py` L53–80）原样保留 `_KERNEL_MAP` 的键 `rope_neox_positions`/`rope_gptj`… ⇒ **Stage 3 首次 forward 会 KeyError**。这是 Stage 0 脚手架缺陷（非本设计的算法问题），但设计应在 §9 风险表登记该接口事实并注明由 conductor 决定修复归属（Stage 0 重跑脚手架或 Stage 3 对齐键名），否则 Stage 3 会在设计未预警处失败。

### 1. API 可行性 — **warn**

- **API 存在性 — pass**（逐条在安装包内核对，`tilelang.__version__ == 0.1.15`，从 `/tmp` 导入以避开仓内同名包）：
  `T.Kernel(*blocks, prelude=None)`（1-D，无 `threads`；docstring 明写「no SIMT thread domain at this scope」）✓；`T.SimtVF(threads: int|list|tuple = 128, latency: int = 0)` ✓；`T.Parallel(*extents, coalesced_width=None, loop_layout=None, annotations=None)` ✓；`T.alloc_shared(shape, dtype, scope="shared.dyn")` ✓；`T.dynamic(name, dtype="int32")` ✓；`T.cast(value, dtype, round='', sat=True, rbits=None, span=None)` ✓；`T.min/T.max(a,b)->PrimExpr`、`T.ceildiv(lhs,rhs,span=None)` ✓；`T.copy(src,dst,*,...)`（Ascend DMA lowering）✓；`T.If/T.Then/T.Else`、`T.sync_threads(barrier_id=None, arrive_count=None)`、`T.clamp`、`T.simd.vld2(addr, dist="DINTLV_B16", off=None)` ✓（docstring：`DINTLV_B16` = 256×bf16 → 两个 128-lane 寄存器，与 §11.3 的表述一致）。
- **API 与模式匹配 — pass**。SIMT 路径的形态在本平台有**算子级**先例：`kernel_template.py` L18–36（`T.Kernel(grid) as bx` + 运行时 `if bx < full:` 内两支 `T.SimtVF(threads=256)`，正是 §3.3「两支各一个 SimtVF 帧」的形态）；`examples/paged_cache_write/_paged_cache_write_kernel/_paged_cache_write_kernel.py` L133–172（`T.Kernel(nblocks) as bx, T.SimtVF(threads=128)` + `T.Parallel(block_rows, cols)` + 语句级行谓词 `if token < tokens` + 动态 tokens），且其 design 记录 4096 块 / 16 元素每线程的规模点（`block_rows=8/16, block_cols=128/256, _THREADS=128`）⇒ §5.2 的「4096 元素 / 16 元素每 lane」确有同形先例，绝对规模属 2–4× 外推并被 §9 R4 如实登记。
- **API 规范 — pass**。§3.4 明确在 site-packages 而非仓根核对（避开同名包遮蔽），我按同法复核；§3.5.3 显式区分「结构参考」与「API 证据」，并声明 `examples/posembedding/rope_llama31/*` 为另一工具链方言（`tilelang.language` / `T.Kernel(is_npu=True)` / `T.alloc_ub`）不作 API 依据——符合平台规则。
- **warn 项**：`COS[row_ub[i], k]`（UB 中的值作 GM 行下标）无直接佐证。最接近的证据是 `paged_cache_write` L146–147（**GM 读值** `slot = T.cast(Slots[token], ...)` 作 GM 行下标）。设计已在 §3.4 标 ⚠️、§9 R1 给出可执行兜底（改用 `T.alloc_local` 标量 + `T.serial(bt)` 逐行展开），风险可控，故记 warn 而非 fail。
- **建议**（并入 S5 的落地范围）：Stage 3 首个配置即验证该 gather 形态；若失败按 §9 R1 兜底，并把结论回写 §3.4。

### 2. 内存层级规划 — **pass**

- **搬运路径完整**：§4.4 给出 `GM[POS]→row_ub`、`GM[COS/SIN]→cs/sn_ub`、`GM[X 行块]→x_ub`、`x_ub→y_ub`、旋转两遍写 `y_ub`、`y_ub→GM[Y 行块]`，每步都是 UB 级（`T.alloc_shared` = `shared.dyn`），无 L1/L0 参与，与纯 Vector 判定（§5.1）一致；无跨块数据流。
- **UB 预算**：机械复核 `ub_budget: pass`（34.0 KiB / 248 KiB）；逐 buffer 手算复核一致（见「机械复核」）。最坏 dtype 选择正确（fp32 `bt=4, cols=1024` 字节数最大）；三处 workload 变体（16-bit / gptj / phi-2）均复算吻合。
- **L0C**：不适用（`l0c_budget: skip`，人工已核无 `T.gemm`）。
- **Buffer 层级标注**：§4.3 五块缓冲全部标注 `UB` 且说明无 fp32 中转缓冲（§1.6.3 #4 弃选理由：表达式级 `T.cast` 在寄存器内完成），与 `PL-1.17-subfp32-fp32-transit` 的「UB 同 dtype → cast → fp32 链」结构一致。
- **多缓冲膨胀核算**：引用 `TRAP-UB-multibuffer-inflation`（910B2C 结构类平台值 20/26 B/elem）得 81.9/106.5 「KB」≤ 248 KB，余量 ≥2.3×，并给出兜底（`annotate_buffer_versions` 固定单版本 / `units_per_block` 减半）。**口径小瑕（不阻塞）**：此处两个数是**十进制**口径（81,920 B / 106,496 B = 80.0/104.0 KiB），与 §4.5 表的 KiB 口径混用；结论（≤248 KiB）不受影响，建议后续统一为 KiB。

### 3. Tiling 策略 — **warn**

- **Block 划分明确**：`bt`/`hpc`/`cols`/`nchunks` 全部给值并给理由；六条 workload 的取值我已按 §5.2 公式独立复算，**全部吻合**（含 phi-2 的 `bt=6`、`cols=640`、tile 3840 与 `hpc=8`）。
- **对齐约束**：明确「元素级对齐成立、段级可能跨段」（`D=65, hpc=8 ⇒ cols·2 = 1040 B` 非 32 B 倍数），并区分「形状本身的性质」与「分块引入的浪费」；配对轴 `half ∈ {16,32}` 为 fp16/bf16 宽度 8、fp32 宽度 4 的整数倍 ⇒ 向量化前提成立（我复核：本任务 6 条 workload 的 `half` 仅 16/32，成立）。
- **非整除处理**：`hpc` 取 `H_side` 的因子 ⇒ `nchunks·cols = H_side·D` 恒精确，**列方向无运行期尾块**（证明成立）；行方向唯一尾块由 `if row0+i < tokens` + `r = min(row0+i, tokens-1)` 处理（越界行只读不写），`tokens < bt` 的退路（`nrb=1`、3 行空转）已写明。纯 Vector 算子已满足该维度要求。
- **分核策略三要素（`core-split-strategy.md` §1 / §2.2）— pass**：
  - ② 物理核数**实查**：给出 `statectl platform --task-dir …` 的实际返回（`observed_cores {cube:28, vector:56}`、`core_source` 含 `rtGetAiCoreCount()` 运行期探针，2026-11-23）。我独立重跑同命令，返回值与设计引用**逐字一致**；`programming_modes.migration_default = simd`、`stage_skills.review = tilelang-ascend` 亦一致。设备侧 `torch.npu.get_device_properties` 未复核的原因（卡被其它任务占用）与 §9 R10 的补记要求已写明。
  - ① 逻辑核数：单侧 4096（`ceil(4096/4)×ceil(4096/1024)`）+ key 侧 1024，合计 5120；其余 workload 逐个给值（复核一致）。写法满足 `design_calc_check` 的单侧口径。
  - ③ 规模判定：四档分类（≤物理核数 / 中 / 极大）＋每档结论＋尾部浪费量化（`5120 = 56×91+24` ⇒ 0.63% ≈ 0.4 µs）＋**明确不做 `T.Persistent` 的理由**（单块工作量仅 4096 元素、无跨块复用、91 波由硬件排空、核内串行会引入每迭代索引重算）。此选择偏离 `core-split-strategy.md` §1 ③ 对「极大规模」的 persistent 推荐，但属「三选一并给出依据」且量化充分、同平台 `paged_cache_write` 亦为逻辑超发形态，故接受（Stage 4 仍保留 persistent/`bt=8` 杠杆，§6.3）。
- **warn 项（S1）**：§5.2/§5.5/§1.6.0 的全部数值都建立在 **`units_per_block = 2048` 对**之上，而冻结 wrapper 的 `default_config` 是 `{"units_per_block": 128*(1 if fp32 else 4)}` 即 **128/512**（`kernels/rope/rope.py` L189–200，注释写作「threads*units」，语义是「每 block 的 head-单元数」）。§10.2 把改动推给 Stage 3（「按 §5.2 由 Stage 3 落定」），但 §0.5 #13、§3.5.2 又把该 wrapper 称为「已固化」。若 Stage 3 不动 `default_config`：llama 的 `bt = max(1, min(128//512, 4)) = 1` ⇒ tile 1024 元素、块数 16384，与 §5.2 表 / §5.5 逻辑核数 / §1.6.0 块数与 60.7 µs 估算**不可比**（正确性不受影响）。建议在 §5.2 或 §10.2 把「Stage 3 必须将 `default_config` 改为 `{"units_per_block": 2048}`（并同步 wrapper 注释中 `units_per_block` 的语义为『每块 pair 预算』）」写成硬性交付项；或改用 128/512 重算三处数字。

### 4. 技术约束检测 — **pass**

- **三维 Kernel**：不涉及（1-D core grid + `bx` 三元折叠，§3.1 step 0–1），不需要 `block_metadata`。
- **GEMM 非整除 / L0C 溢出**：不涉及（无 `T.gemm`）。
- **平台约束逐条命中**（§3.5.1）：1-D grid 无 `threads`（已核源码与 `references/programming.md` L13）、`T.SimtVF` 内逐元素、动态 grid 仅 1-D（同平台 `paged_cache_write` 用 `T.ceildiv(tokens,…)*nchunks` 的先例）、**条件 alloc 非法 ⇒ 五块 UB 在分支前分配**（`TRAP-tvm-parser-rules`，设计正确规避）、跨块 barrier 未认证（`CG-950-batch-barrier` 存在，设计不使用）、一个并行体两次 store 会退化（`paged_cache_write` L101–107 注释实证 fp8 写 0）⇒ §0.6 R8 拆两遍 pass、`TRAP-parallel2d-bm-ge2-scalarize`（910B2C/0.1.2）⇒ 用语句级 `if` 而非 `T.if_then_else` 融合并登记 §9 R3 的 `scalar_ratio` 核对、`TRAP-C12`（`T.copy` 跨 dtype 静默转换）⇒ 全部 copy 同 dtype、`TRAP-C10`（跨步 gather 向量指令缺失，截至 2026-08-28 build）⇒ GPT-J 走 §1.6.3 实验裁决。
- 与 `knowledge/capability-gaps.md`、`bottlenecks.md`（「当前没有 950 真机验证的瓶颈模式」）的取证边界一致，无与实测条目矛盾之处。

### 5. 循环与同步 — **warn**

- **循环结构 — pass**：§6.1 逐项给出 `T.Kernel`（block 并行）/`T.SimtVF`（lane 域）/`T.Parallel`（小规模行号、向量化 pass）的选择与理由，并明确 `T.serial`/`T.Pipelined`/`T.Persistent` **均不使用**及理由（无跨 tile 复用、无 stage 可叠、硬件分波）；§6.2 的 8 条 pass 顺序与 §1.6.2/§3.3 一致。
- **同步与模式匹配 — pass**：§7.2 逐对列出 RAW/WAR/WAW（`row_ub`→表、`cs/sn`→旋转、`x_ub`→载入后只读、`y_ub` 复制→覆写→写回），全部依赖自动同步（`allow_autoschedule` 默认开启，`references/programming.md` L53 已核），并给出 `T.sync_threads()` 兜底（安装包导出，签名已核）；无跨核旗标。
- **尾块处理 — pass**：§6.4 列出 7 种尾块来源，列方向/`D==R`/`tokens<bt`/`half` 非向量宽 全部给结论与代价。
- **warn 项（S5 之一）**：§3.3 的 `body(X, Y, POS, OFF, H, hpc, cols, nch, rb, cb)` 在 `main` 之外定义，却直接引用 `COS`、`SIN`、`row_ub`、`cs_ub`、`sn_ub`、`x_ub`、`y_ub` —— 这些名字只在 `main` 体内绑定，Python 闭包取不到，按 `TRAP-tvm-parser-rules`（未定义名报 Undefined variable / trace error）会直接失败。作为伪代码意图清楚（Stage 3 会传参），但建议把七个名字改成显式参数，避免 Stage 3 照抄。

### 6. 验证方案 — **pass**

- **Golden 函数**：§8.1 有可执行草案，与 `workloads/rope.py` 的 `rope_tables`（L46–57：`arange(0, rotary, 2)`、`base**(-…)`、`.to(params["dtype"]).to(query.dtype)`）和 `_rotate`（L69–90：fp32 计算后 `.to(x.dtype)` 一次舍入）逐段对应；`neox`/`gptj` 两种配对均实现；`key is None` 时返回 `None`。独立性见维度 0。
- **L0 计划**：10 条用例，覆盖 D==R、GPT-J 相邻配对、bf16 + 多维 positions + query-only、fp32 + `R<D`（`R=24`）、phi-2 形态（int32 位置 + `D=80`）、clamp、0 头 key、`tokens=0`、`staged` 两值、roofline 对账；boundary 另 8 条（`tokens=17/1`、`D=65`、`D=80`、`hpc=1`、`P=2`、角点 inf/NaN/65504、`half=12`）。分层入口（`--level L0/L1/boundary`）与平台 `execution.precision_command`（`<python> <kernel> --level <level>`，cwd=任务目录）一致。
- **精度标准**：fp16 1e-3、bf16 1.6e-2、fp32 1e-5（atol=rtol），与 `tests/ops/test_rope.py::_tolerances`（L59–65）**逐值一致**，并引用 G1（golden vs fp64，fp32 容差违反 0）作为可达性证据。

### 7. 完整性与一致性 — **warn**

- **章节齐全 — pass**：§0（0.1–0.7）+ §1–§11 齐全，迁移分析小节无缺失。
- **无占位符 — pass**：全文占位标记扫描 0 命中（未填充标记、待补章节、花括号变量名均不存在）。
- **同类引用真实 — pass**：§3.5.3/§0.7/§11.4 引用的 8 条路径全部存在（`examples/paged_cache_write/_paged_cache_write_kernel/{DESIGN.md,_paged_cache_write_kernel.py}`、`.agents/platforms/toolchains/ascend/kernel_template.py`、`examples/posembedding/rope_llama31/kernels/{table_fp32.py,table_low_precision.py}`、`examples/elementwise/vec_add_2d_dynamic_shape.py`、`.agents/platforms/toolchains/ascend/knowledge/{PL-950-row-primitives.md,numerical-boundaries.md}`，以及源算子与抽取文件）；引用的 `L101–107`（fp8 双 store）、`L129–172`（语句级行谓词）行号区间经核对**准确**。
- **内部一致 — warn**（三处，均已给建议，不影响算法）：
  1. `units_per_block` 2048（§5.2）vs 冻结 wrapper 128/512（S1）；
  2. §0.4 #4「源已每 block 读一次表」vs §1.6.0 R3/§1.6.1 A1「源按单元读表」（S3）；
  3. §1.6.2 #5 把尾列直通写成 `T.copy(x_ub, y_ub)`，而 §3.1 step 6/§3.3/§4.4 的主选 P 是「一条 `T.Parallel` 拷贝 pass」（`T.copy` 属备选 B，§1.6.1 A3 亦如此区分）（S5）；另 §1.6.1 表内 X3 行的「最大 ulp」与脚本不符（S4）。
- 其余交叉处（§1.4↔§3.1↔§4↔§5↔§6、§8 与 §0.1、§11 与 §1.6.1）一致。

### 8. 算法优化分析 — **warn**

**§1.6 章节存在性 — pass**：1.6.0/1.6.1/1.6.2/1.6.3 四子节齐备，且顺序符合「先调研 → 公式优化 → 向量化 → 布局」。

**算法调研存在性与完整性 — pass**
- R1 候选表**含基线 #0**（源直译），另 6 个候选（表行复用/整块 tile、三角恒等重写、极坐标重写、表即时重算、FMA 单次舍入、原位旋转），逐项给等价性初判与收益方向。
- R2 在线算法有**明确结论「无」**并给结构依据（输出元素只依赖同 token 的 2 输入 + 2 表值 + 1 行号，无跨元素统计量 ⇒ 单遍即最优）；我独立复核该结构判据成立（源 `rotate_token` 无任何归约/累计量）。
- R3 覆盖**四口径**（FLOPs / 访存 Bytes（注明含/不含）/ 扫描遍数 / 中间缓冲峰值）且基线在表。
- R4 逐候选六维评估（单元匹配/容量/对齐/静态边界/流水融合/跨核代价），并对「表驻留」给出容量算术（`8192×128×2 = 2.0 MB ≫ 248 KB`，复算 ✓）。
- 调研范围与负向断言复核：`algorithm-research.md` §5 无 rope 命中行（grep 全空）；平台 `knowledge/algorithm-candidates.md` 仅 7 个算子族且无 rope；我重跑 `kb_search --task-dir … "rope rotary position embedding"` 返回 `no entries matched` ⇒ 设计「本平台知识库对 rope 族零条目」属实；候选由源算法 + 三角恒等 + 同族实现补足，并**如实标注未使用互联网检索**（附注理由）。无被漏掉的已知在线/化简候选（该类算子在结构上不存在在线化对象；RoPE 的常见化简只有极性/重算/FMA 三项，均已评估并否决）。

**调研结论正确性（独立复核）— warn（数据项见 S4，结论方向不变）**
- 等价性：选定算法 #1 与基线同一算式、同一表值、同一配对，仅改变读取次数与暂存 ⇒ 逐位等价（A1/A3 机器验证 0 失配，我复核脚本为「两套独立寻址实现」对照，非自证）。R2 的「无在线变体」与源码结构一致。R1 #3（极坐标）以「4 个超越函数/对 + 中间舍入路径不同」淘汰，代价方向成立（等效算力 ~38×，属 ALU 侧量级淘汰；该候选未纳入 `verify_equiv.py`，不构成须机验的等价性主张）；#4（重算表）「不等价」成立（X3 逐位 96%+ 失配，容忍差 15202–15871）；#5（FMA）「fp32 位级不等价」成立（X2 fp32 878–1924 位差）；#6（原位）「RAW 冲突」成立（同 tile 内跨 lane 读另一列）。
- 复杂度复算：选定 #1 的所有数字复算精确吻合（见「机械复核」）；**基线 #0 的表项不成立**：`Hq+Hk = 40` 个 head 各读 `R = 128` 个表值 ⇒ `4096×40×128×2 B = 41.9 MB`，设计记 33.6 MB（只计了 query 的 32 个 head）。连带：基线总流量 117.5 → **125.8 MB**、淘汰比 1.32× → **1.41×**、A1 收益「-85%」→ **-87.5%**、「6.4×」→ **8.0×**。方向一致（基线更差），不影响选定，但属须改正的数字。
- 亲和性淘汰依据：`CONST-mte2-degradation`（1.26 TB/s @48 核、4 ns/128 B 段、2026-09-21 段数更正）、`CONST-vector-launch-overhead`（0.5 µs/op）、`CONST-capacity-910B2C`（UB 容量口径）三个条目 ID 与版本戳均在 `constants.md` 中核到；`PL-1.14-mte3-strided-ws` 的「band 化使 MTE3 3×」实测值核对一致（106→220 µs）；跨型号外推（910B2C→950PR，`1.26×56/48=1.47 TB/s`）已如实标注「未实测」并与 `bottlenecks.md`「无 950 真机瓶颈常数」口径一致。
- 并列候选裁决：GPT-J 配对访问是否被标量化**未纸面拍板**，进入 §1.6.3 实验裁决（T1/T2/T3），符合要求。

**调研→设计一致性 — pass**：§1.4 的 8 条差异、§3.1 step 0–9、§6 循环结构全部按 #1 落地（整块 tile + 表行复用 + 双缓冲 + 单程序）；迁移任务的 §0.5/§0.6 处置与调研选族一致，§0.4 的 12 项优化意图有逐项承接表（4 保留/4 等价替换/4 重设计/2 舍弃，合计 14 = 12 + §0.5 的 #12/#14），无静默丢弃（仅 #4 归因错误，见 S3）。

**数学等价优化完整性 — pass**：A1/A2/A3 三项均有四要素（原式 → 优化后 → 等价性论证 → 收益量化）；否定空间另给五条结论与依据（无代数化简下界、无跨 lane 公共子表达式、无可降代算子、访存已达一读一写下界、无归约可合并）。

**等价性论证独立推演 — pass**
- A1：`cs_ub[i,:] = COS[row_i,:]` 后被块内 `hpc` 个 head 复用，对每个 `(h,k)` 取到的 `cos[row,k]` 与逐单元读取的值**同一内存位置、同一 dtype 值** ⇒ 逐个算式输入相同 ⇒ 逐位相同。脚本的「两套独立寻址实现」对照（15 组，0 失配）排除「k 偏移写错」这一类错误。
- A2：`clamp` 在 int64 域完成后值域为 `[0,P-1]`，`P ≤ 2^31`（表在内存中）⇒ `int32` 窄化为恒等；先窄化则 `±2^32±1/±2^40` 回绕（反例 5/8341 复现），故顺序不可交换——论证成立，且与源 L329 的 `T.Cast("int64", …)` 相加口径一致。
- A3：同 dtype 整块拷贝（无转换、无舍入）+ 旋转区覆写 ⇒ 未覆写列即输出值；与源「尾列直拷」等价。`D=65/D=80/R<D` 共 9 组 0 失配。
- 与 §8.2 精度标准的关系一致：adopted 项按位级判、G1 按容差判（fp32 约 1 ulp 差异不误判），两套口径未混用。

**向量化替代完整性 — warn（覆盖完整，一处描述需统一 S5）**
- 覆盖核对：§3.3 主选 P 共 8 条 pass（1 行号 + 2 表 + 1 载入 + 1 复制 + 2 旋转 + 1 写回），§1.6.2 的 12 个点逐点对应，**无遗漏**（#1↔载入、#2↔两条表 pass、#3/#4↔两条旋转 pass、#5↔复制 pass、#6↔写回、#7↔行号、#8↔行谓词、#9↔块折叠、#10↔`p//half`、#11↔host、#12↔GPT-J 列索引）。
- 已替代项有 API 与平台证据（`PL-950-conversion-layout`、`PL-950-vector-execution`、`PL-950-row-primitives`、同平台 `paged_cache_write`）；未替代 4 项的理由具体到本算子（地址/索引计算、动态边界、block 级任务映射、host 元数据），**无「实现简单/照源码写」式理由**。
- 描述不一致（S5）：#5 的「向量替代方案」写 `T.copy(x_ub, y_ub)`，但主选 P 实际是 `T.Parallel(bt, cols)` 拷贝 pass（`T.copy` 只在备选 B）；应统一为 P 的形态，并把 `T.copy` 版本标注为 B。

**替代 API 可行性 — pass**：`T.Parallel`/`T.cast`/`T.copy`/`T.simd.vld2` 均在 0.1.15 安装包核对到签名；`vld2` 的 `DINTLV_B16` 语义与 §11.3 描述一致，且设计如实声明「本仓无任何 `T.simd.*` 算子级用例」（我 grep 全 `examples/` 得 0 处），`PL-950-simd-reduce` 结论明确「只覆盖 FP32 … 不覆盖其他 dtype」⇒ SIMD 备选留待取证、不作主选，判断正确。

**向量化轴与布局决策完整性 — pass**：§1.6.3 给出 7 个候选的矩阵（原生行主序 / lane 沿 token / 表值扩宽 / 物化 fp32 tile / GPT-J 去交织 / SIMD 去交织 / host permute），逐项有量化评分（流量、UB 增量、段数与 pass 数），弃选理由均为数量级（#2 引用 PL-1.14 实测 3×、#4 明确「UB 与指令都增加」且引 `PL-1.17` 的链结构、#7 host permute 成本 `167.8 MB ≈ +114 µs` 复算 ✓ 且 > 本体 2.8×）；明确指出「I/O layout 是契约、核内布局是设计变量」；repack 路径为「无」并在 §3.3/§4/§6 三方一致（buffer 形状 = `(bt, cols_max)`、内层列连续、无内层串行维）。
**实验裁决模式 — pass（备选结构深度见建议）**：GPT-J 的判定依赖未实测的编译器行为，设计给齐三件套——主选 P、备选 B_gptj（UB 去交织，含额外 2 条 pass 与 `bt·hpc·half·itemsize` 量级 UB）、裁决计划（代表 shape `gptj-6b-prefill`/`gptj-6b-decode-b32` + 对照锚点、msprof `scalar_ratio`/`vec_ratio` 反解、**可机械判定阈值 T1 `L_P ≥ 1.15×L_B`（含 CV 上修规则）/ T2 `scalar_ratio ≥ 0.5` / T3 语义门**、按 workload 分派、`[DESIGN_LIMIT]` 回写路径），并声明未回写前维持 P。CV=5% 与 ALU 吞吐为**未实测假设**，已显式标注且不承担最终判定。
- **建议（并入 S4/S5）**：① 修正 R3 基线表项与 A1 收益数字；② 统一 §1.6.2 #5 的替代形态描述；③ B_gptj 目前只到「两条 pass + 两个半区 buffer」量级，建议补到 buffer 形状 `(bt, hpc, half)`/UB 预算/分核三要素不变性逐条写出，以便 Stage 4 直接实现。

---

## 检视方法与环境记录（可复核）

- 源算子全文 Read（1189 行）+ 冻结 wrapper/Op/manifest/测试/workload 交叉核对；`examples/` 引用路径 8/8 存在性校验。
- 机械复核：`python examples/rope/_rope_positions_kernel/verify_equiv.py`（torch CPU）+ `python3 .agents/tools/design_calc_check.py --design … --task-dir …`（3 pass / 1 skip）。
- API 证据：`/workspace/venv-ascend950/lib/python3.12/site-packages/tilelang` 0.1.15，以 `inspect.signature` 逐个核对（在 `/tmp` 下导入，避开仓内 `tilelang/` 同名包遮蔽——这正是 §3.4 记录的证据纪律）。
- 平台事实：`python3 .agents/tools/statectl.py platform --task-dir examples/rope/_rope_positions_kernel`（`execution.mode = device`、`hardware.observed_cores {cube:28, vector:56}`、`capacity_bytes.UB = 253952`、`ub_inflation = null`、`programming_modes.migration_default = simd`、`stage_skills.review = tilelang-ascend`）。**未运行任何 kernel / msprof**（按调度约束）。
- 知识条目核对：`constants.md`（CONST-mte2-degradation / CONST-vector-launch-overhead / CONST-capacity-910B2C）、`elementwise.md`（PL-1.6 / PL-1.14 / PL-1.17 / PL-1.19）、`traps-compiler.md`（C10 / C12 / TRAP-UB-multibuffer-inflation / TRAP-parallel2d-bm-ge2-scalarize / TRAP-tvm-parser-rules）、`knowledge/{PL-950-row-primitives, PL-950-vector-execution, numerical-boundaries(PL-950-simd-reduce / PL-950-conversion-layout), capability-gaps(CG-950-batch-barrier), bottlenecks, algorithm-candidates}`、`core-split-strategy.md`、`hardware-cost-model.md`、`references/programming.md`。
