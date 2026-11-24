# Stage 1 复盘（mla_varlen_bwd / _bwd_kernel）

> 逐次追加。每次 `DESIGN.md` 产出后写一节，最新在最上。

## 2026-11-24 — 设计修订 v1（单次 launch grid 上界 65,535 的静默少算）

### 1. 触发与事实

- 实施后复核（`/tmp/opencode/MLA_VARLEN_BWD_8x4K_FINDINGS.md`，只读测量）证明 `ds-v3-8x4k`（B=8/H=128/L=4096）
  的 `grid = 66,560` 只跑了 1,024 个 CTA：20.75 ms 是**少算**（真实 ≈488 ms），request 1..7 的输出全 0。
- 关键事实（本任务补全）：`grid % 65536 != 0` 时**不报错**，只执行 `grid % 65536` 个块；整倍数时才 abort。
  v0 的设计只把「动态 grid + `live`」当静态边界处理，**没有把「块索引 16 位」列为技术约束**。

### 2. 为什么 v0 的验证没发现（结构教训）

1. 全部 correctness 用例 `H ≤ 32` ⇒ `grid ≤ 32×tiles`，物理上碰不到红线；manifest 的 H=128 case 恰好是唯一越界几何。
2. 该 bug 的**表现是「更快」**（截断），所以任何 latency 门槛/sanity 都不会失败；反而是「691 TFLOPS」这种
   不可能值应当直接判失败（G4 tripwire 的由来）。
3. 若参照解与待测解跑同一条大 grid 路径，两者会**一起错**——本轮用「B=1、grid 远低于上界」的同内核 launch
   作参照才定位到问题。**教训：边界 bug 的参照必须在不同 grid 量级上取。**

### 3. 处置（设计层）

- 采用 **head slab 多次 launch**：`grid = tiles*slab`，`slab = min(32, heads, 65535//tiles)`（编译期常量），
  `tile = bid//slab`、`h = h_off[i] + bid%slab`，尾 slab 用 `if h < heads` 守卫；一个编译产物 + 运行期
  `h_off`（int32[1]）服务全部 slab。
- **发射前断言**（`launch_plan.py`，纯 Python）：`grid ≤ 65,535`、head 覆盖恰好一次、越界 raise。
- 逐项否决「调大 `block_n`」「head 折进核内串行」「运行期除数」「两 kernel 拆分」「只断言不修」五种绕开方式
  （§5.6 f，均给了定量理由）。
- 明确声明：**不改算法/分块/dtype/精度路径/IO 契约，不做性能优化**。

### 4. 机器验证与探针（本次实跑）

- `verify_equiv.py` 新增 **A4**：8 组几何逐 bid 穷举 **568,904 个逻辑任务**，缺失 0 / 重复 0 / 越界 0，
  单次 `grid` 全部 ≤ 65,535；越界参数 2/2 raise；边界 2/2 通过（65,535 合法、自动缩小到 slab=31）。
  输出 `EQUIV_SUMMARY ALL_PASS`（A1×2 + A2 + A3 + A4 共 5 行）。
- 新增编译探针 **P6**（`history_version/design_probe_grid_slab.py`，heads=3/slab=2 的**不整除**几何）：
  `COMPILE_OK`，28,381 B，设备码 `block_idx >> 1`、`h_off[0] + (block_idx & 1)`、`if (h < 3)` 包住 body，
  13 个既有 mark 全保留 ⇒ slab 化没有改变核内形态。

### 5. 遗留（移交 Stage 3 / 集成）

1. Stage 3 第 0 步：内联 `plan_launch`、加 `h_off`/`head_slab`、跑 §8.4 **G5**（纯 host 侧）。
2. §8.4 G1–G4 必须在 Stage 3 通过（H=128 的逐 request 拆分对照是本次的唯一决定性回归）。
3. 同族 launch（`_delta_kernel` / `_dq_land_kernel`）的运行期 grid 也应在 wrapper 里复用同一断言（§9.1 风险 18）。
4. 性能代价（launch 边界 L2 冷启动、`head_slab` 是否提到 64）留 Stage 4 用 `msopprof` 实测裁决。


## 2026-11-23 — 首次设计 v0（packed 分支，复用 dense 已验证核心）

### 1. 任务形态与策略

- 迁移任务（harness 子模式），`_bwd_kernel` 的 **packed 分支**（`MLAVarlenBwdOp`）。
- 关键判断：读源码后确认 `_bwd_kernel`（`bwd.py:109-264`）是 dense/packed **共用工厂**，`dense` 只切换
  `stat_shape`、`num_tiles`、`(request, n0)` 解码、`lse/delta` 取址。dense 版已通过检视 + 真机（L0 13/13、
  L1 6/6），因此策略定为：**复用 dense 的已验证核心，只重推 3 处 packed 差异**（任务显式要求），
  而不是从零设计。这直接决定了文档结构（§0.8 复用/差异分区 + §11 对照索引）。

### 2. 三处差异的推导要点（价值最高处）

| 差异 | 关键发现 | 处置 |
|------|---------|------|
| ① 变长块枚举 | 源的 `GroupTiling.cumsum_offsets` 在 `num_groups > 8` 时走 **warp shuffle 前缀扫描**（`T.shfl_up` + kernel 级 thread extent），NPU 上根本不存在该原语；不能「照搬」 | 重推为**融合式 running-prefix 解码**：一次 `T.serial(batch)` 同时完成前缀和与 first-match decode，全部 local 标量、无共享块表；语义与 `decode` 的二分逐条对齐 |
| ② head-major `lse/delta` | 保持 `[H, T_q]` 反而最优：每次装载是 **256 B 连续段**（设备码 `asc_copy_gm2ub_align(..., 1, 256, ...)`）；转成 dense 的 `[B,H,S]` 只会多一次 host 侧重排 | 不转置 |
| ③ 逐 request 边界 | **最大的隐性坑**：平台 OOB clamp 只按**张量 extent**（设备码 `min(64, max((total_kv-kv_start)-n0,0))`），不认 request 边界 → `dk/dv` 整块写回会把**精确 0 覆盖到下一个 request 的行**（`dk/dv` 是赋值）。而 `dq` 因为掩码使越界列 `dS=0`，整块**原子加 0** 反而无害 | `dk/dv` 双分支（内部整块快路 / 尾半块逐 lane 守卫）；`dq` 保留整块原子并用 A3 把它变成可验证的不变式 |

### 3. 探针过程中的实际教训

1. **`@T.prim_func` 体内不能调用普通 Python 函数承载循环**：首版探针把计算体抽成普通函数，报
   `TypeError: 'ForFrame' object is not iterable`（`for ... in T.Pipelined(...)` 未被 AST 变换）。
   改 `@T.macro`（与源 `GroupTiling` 的写法一致）后一次通过。**这条已写进 §3.5.4 反例表**，避免 Stage 3 重犯。
2. **`T.clear` 不支持 L0C**（`InternalError: Unsupported scope shared.l0c`）：原本想给退化请求（`len_q=0`）
   做防御性清零，只能改设计——用 `loop_ed = max(ceil(len_q/64), loop_st+1)` 保证循环非空，让一次全掩码
   迭代产出精确 0。这是**设计因探针而改变**的实例（不是事后补说明）。
3. **动态 grid + 混合 kernel 居然直接编译通过**（P1/P2/P5）：这是开工时最大的未知项（平台先例只有纯 SIMT
   的动态 grid）。编译级通过不代表运行期通过，但足以让主方案成立，并把「静态 grid 上界」作为已编译的
   备用方案落纸（P5），而不是空口承诺。
4. **符号长度区段 copy 也能编译**（P3）：因此 `dk/dv` 尾块有两条可行路径，设计里把 P3 记为回退形态，
   主选仍是双分支（少一份「运行期长度参与 DMA」的未知量）。

### 4. 机器验证（verify_equiv.py）的两次迭代

- 第一版 A3 只枚举了**裁剪后**的块（`j` 用 `min(m0+64, len_q)`），于是 `spill_entries=0`、`acc_bad=1`
  （`-0.0 + 0.0 = +0.0` 的位模式差异被判成违反）→ `EQUIV_BLOCKED`。
- 修正两处：块按**完整 64 行**枚举（GEMM 实际就是整块跑），累加器比较改为**数值相等**并单独统计零号翻转。
  重跑 4/4 `EQUIV_PASS`（A2 1.157 亿元素、快路 13734/28261；A3 30528 个溢出差分元素 + 3816 个溢出 dq 行
  全为精确 0；累加器 4102 样本 1 个零号翻转，数值等价）。
- 教训：**不变式验证的枚举口径必须与 kernel 的分块口径一致**（整块 vs 裁剪块），否则验证会「假通过/假失败」。

### 5. 量化结论（供后续阶段对照）

- 代表 workload `ds-v3-mixed`：可见 score 对 22.9e9、FLOPs 38.2 T（分块实际 38.5 T，+0.9%）、
  迭代 5,623,296、live 块 64,512。
- 每迭代 L1 端口 **279280 B**（24.5 FLOP/字节）→ 全量 **1.571 TB**；GM 视图 **744 GB**
  （q/do 双读 460.7 + `dq` 原子 276.4 + k/v 4.2 + dk/dv 2.6），L2 驻留后 HBM ≈ 17.2 GB。
- 容量：L0C 62.5%、L1 32.8%、UB 35.7%（沿用 dense 真机通过的半区 buffer 清单，逐字节相同）。
- 分核：逻辑 33,024…66,560 ≫ 28 核 → 「逻辑块超发」，并用「最长块 0.0046%」量化长尾；
  偏离 `core-split-strategy.md` 的极大规模标准分支，写明了复评触发条件。

### 6. 留给后续阶段的重点（按优先级）

1. **R1/R2 探针**：动态 grid 的运行期行为 + host 侧 grid 实参传递（两个不同 `T_kv` 各编译一次）。
2. **R3/L0-9**：request 间污染回归（B=2 且尾块请求在前）——这是本函数最可能出错的地方。
3. **R4**：溢出零行的设备确认（同一数据 vs 「只有 request 1」逐位对照）。
4. 若动态 grid 失败 → 直接切 P5 的静态 grid（已编译），不要把时间花在别的方向上。
5. Stage 4 的流水：packed 的 `loop_ed` 无法静态化（`ceil(len_q/64)` 运行期），流水收益必然低于 dense，
   优先考虑 L1 侧双版本而不是 UB 侧。

## Stage 2 (design review) Retrospective — 2026-11-23 — 结论 通过（阻塞 0 / 建议 7）

> 本节复盘**检视流程自身**；设计内容的问题已随 `REVIEW.md` 流转给 Stage 1/Stage 3，不在此重复。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Mechanical-check | `design_calc_check.py` 只按「单块 `block_M×block_N×sizeof(accum)`」核 L0C（本案输出 `block 64x64 x fp32 accum = 16.0 KB vs L0C 256 KB`），**不核多累加器之和**；本案真正的约束是 5 个累加器合计 163840 B（62.5%），机械位完全没覆盖。L1 层则没有任何 check 项（本案 L1 172032 B 是必核项）。两处都靠人工逐 buffer 相加；若某设计把 5 个累加器配到超容量，机械检查仍会 pass | `/workspace/venv-ascend950/bin/python3 .agents/tools/design_calc_check.py --design … --task-dir …` 输出（`l0c_budget` detail 仅单块；无 `l1_budget` 项）；DESIGN.md §4.6（5 累加器合计）与 §4.5/§4.6（L1 表） | 解析器：① `l0c_budget` 改为累加 §4.6 表中 L0C 行的**全部**条目；② 增 `l1_budget` check（读 `design_budget.capacity_bytes.L1`）；③ `core_split` 支持「表名/列名含『逻辑核数』」的核数表（本案仍 skip） | R |
| Dimension-gap | 等价性机械位（D-1）只要求「重跑 + 与内嵌表一致」，**不要求审阅脚本是否真的验证了该命题**——一个把 candidate 与 baseline 写成同一表达式的脚本也能打印 `EQUIV_PASS`。本次靠人工读 `verify_equiv.py`（A2 的 `assert vis.all()` 是真判据、A3 是结构性推导而非自证）才确认非伪造 | `verify_equiv.py:182-196`（A2 快路断言）、`214-274`（A3 结构检查）；DESIGN.md §1.6.1 内嵌表 | 「机械复核（D-1）」段补一句：`verify_equiv.py` 必须审阅其**判据构造**（候选与基线是否独立求值、是否有 `assert` 型充分条件、容差定义是否可证否），只重跑不读脚本不算完成 | P |
| Evidence-scope | 设计文档的设备码引用是**表达式级原文**（如 `for (g=0; g<3; ++g) { ntg = … }`），本 Agent 先用 regex 按原文匹配 P2 源码，全部 0 命中（实际存在，但变量名/格式为 `for (int32_t g …)`、`cnt[0] += ntg`）；差点误判为「设备码证据不实」。dense Stage 2 复盘只沉淀了「计数/事件 id 会漂移」，**表达式文本本身也会漂移** | `/tmp/opencode/rv_varlen_p2.py`（按原文 regex → 0 命中）vs `/tmp/opencode/p2_src.cu:104-121`（结构存在） | 维度 0/1 的设备码核对改为固定三步：① 用**调用种类**（`asc_*` 符号）建索引；② 核**参数形态**（长度表达式是否含 `min()/max()/total_*`、相邻调用关系、是否落在原子窗口内）；③ 不比对变量名与嵌套格式。设计文档若引原文，检视时按 ①② 复核即可 | R |
| Reuse-evidence | "同源码工厂 + 复用兄弟设计已核验结论"这一形态在本任务出现（packed 复用 dense），但 skill 的维度 0/8 没有对应检查项：复用多少算够、哪些兄弟结论在变体下会失效、失效项是否被抓出。本次设计的 §0.8-A/B/C + §11.3 恰好是正确形态，检视方只能靠自建判据（亲读源工厂确认差异面 → 逐项核对 §11.3 是否覆盖全部失效结论） | DESIGN.md §0.8、§11.1-11.3；源 `bwd.py:124,133-139,176-184,214-224`（差异面只有 4 处） | 维度 0 增一子项「复用有效性」：① 复用的『同一源码工厂』主张须由源码差异面核对（列差异点并计数，与设计自述比对）；② 兄弟设计中被复用结论在变体下**必须逐条判存续**，「不再成立」的结论须有替代方案与机器证据（本案 R7/A3） | P |
| Dimension-gap | 兄弟设计（dense）在 Stage 3 实测得到的**已知容差边界**（bf16 `dS` cast 抵消放大 17×，单元素 7.8e-3 > atol 5e-3）没有随复用关系流入本设计的 §8.2/§9.1，检视时也无可循检查项；Stage 3 极可能重复诊断同一现象 | `examples/mla_dense_bwd/_bwd_kernel/RETROSPECTIVE.md` Stage 3 段（L2-3/B-4 WARN + 诊断手法）；本设计 §8.2 无该条 | 维度 6 增一子项：同一算法族的兄弟设计已有 Stage 3/4 结论时，§8.2 须承接其**容差边界与判定手法**（尤其 bf16/fp16 固有舍入），缺失记建议级 | P |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 设备码证据的「结构三元组」核对法：调用种类 → 参数形态（`min/max/total_*` 表达 + 相邻关系 + 原子窗口包夹）→ 不比对变量名/计数/id。可把「原文 regex 0 命中」这类假阴性一次排除 | P | 本次 `/tmp/opencode/rv_varlen_p2.py` 8 条 regex 全 0 命中（含必然存在的 `cnt +=`/`min(64, max((total_kv`），换「符号 + 参数形态」后 6 项一次确认（`asc_copy_l0c2gm` 的 `min(64, max((total_q-q_start)-kq*64,0))`、`asc_copy_ub2gm_align` 的 `min(32, …)`、`asc_copy_gm2ub_align` 的 `…*total_q + q_start` 等） | `cd /tmp && ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python /tmp/opencode/rv_varlen_p2.py` | tilelang 0.1.15 / CANN 9.2.1 / ascend-950pr-9579 | 本 skill（Phase 1 第 7 步 + 维度 0/1 设备码核对），并与 dense Stage 2 的「计数/事件 id 不参与判定」合并为一条 |
| 复用型迁移的检视判据（差异面计数 → 存续判定 → 失效项替代证据），避免「兄弟设计已通过」被当成免检通行证 | P | 本任务 §0.8-A/B/C、§11.3 + 源工厂 4 处差异面（`bwd.py:133-139,176-184,214-224`）的逐条核对 | 源文件 `bwd.py` 全文 Read + `grep -n "dense" bwd.py` | — | 本 skill（维度 0 迁移表增子项） |

### Transferable Lessons

- **机械检查的覆盖面要按「实际约束层」核对，不能只看它 pass**：本案 L1（32.8%）与「5 个 L0C 累加器之和」都不在机械位里，若只看 `pass` 就漏掉最该核的两个数。凡设计给出逐 buffer 表，检视方应亲手加一遍。
- **D-1 的可信度取决于脚本的判据构造**：重跑一致只证明「可复现」，不证明「验证了该命题」。看三件事——候选与基线是否独立求值、充分条件是否有 `assert`、容差定义是否可证否。
- **设备码引用按「调用种类 + 参数形态」核，不按原文串匹配**：变量名、格式化、`cnt[0] +=` 与 `cnt +=` 的差异会让 regex 全灭，制造假阴性；反过来也不会因计数/id 漂移制造假阳性。
- **「兄弟设计已通过」不是免检**：必须先亲读源码确认差异面有几处、与设计自述是否吻合，再把兄弟结论逐条判存续；失效项（本案：平台 clamp 写回、整块 L0C→UB、`T.clear` L0C）必须有替代方案 + 机器证据，而不是"借"过来。
- **Stage 3 的已知容差边界要随复用关系搬运**：同一数学的两个变体会撞同一堵墙（bf16 `dS` 抵消放大）；检视方有责任在 §8.2 里发现缺失并升级为建议。

## Stage 3 (develop) Retrospective — first_impl attempt 1 — 2026-11-23

结论：`[PRECISION_PASS]`（L0 20/20、L1 6/6、L2 4/4、Boundary 12/12；7 个运行期前置探针 R1–R7
全 PASS；`--level all` 423 s，含全部 16 个几何的编译）。设计的三处 packed 差异全部在真机上成立，
**§9.1 风险 1（动态 grid + 混合 kernel 的运行期行为）一次通过，无需回退到 P5 静态 grid**。

### 1. 设计文档里被真机确认的部分（不需要改，但值得记账）

| 设计断言 | 真机证据 |
|---|---|
| 动态 grid `(total_kv//64 + B) * H` + `live` 早退 | R1/R2：同一份编译产物在两个不同 `T_kv`（512 → grid 9、96 → grid 2）上都正确；L0-17/19 的 grid 实测 16640 / 8160 块全部正确、OOB 哨兵 0 次触碰 |
| 融合式 running-prefix decode（`T.serial(batch)` + `alloc_local`） | R1（B=2 不同长度、非整除）+ L0-6（`[65,1,200]` 三请求混合）+ L0-13（`[0,64]`）；`live==0` 块不发起任何写（哨兵 0 触碰） |
| `lse/delta` 保持 head-major `[H,T_q]`，不转置 | R5：正确的自然对数 head-major → dq 4.2e-2 通过；把它换成 log2 域 / token-major 重解释 / 错位 delta → 分别炸到 1.8e1 / 7.8e1 / 5.3e1，说明布局与对数域确实被消费而非静默忽略 |
| `dq` 整块 store-atomic + A3「溢出零行」 | R4：`[96,96]` 下 request 0 的尾块向 request 1 溢写 32 行，request 1 的 `dq_acc` 与「只跑 request 1」的结果**逐位相同（0.00e+00）**；L0-11 的「预填 2.0」两跑均与 golden 一致 ⇒ 是加不是覆盖 |
| `dk/dv` 双分支（内部整块 / 尾块逐 lane） | R3（`[1]/[1]`、`[300]/[700]`）+ L0-9（`[300,512]`，request 0 的 44 行尾块紧邻 request 1）+ B-11（`len_q=0` 夹在中间）逐请求干净 |
| 半区 `dual_copy` 2:1 + `asc_get_sub_block_id`（复用 dense） | R6：单块 `[64]/[64]` 的 rows[0,32) 与 rows[32,64) 各自与 golden 一致 ⇒ 两个 AIV 的半区没有错位 |
| `loop_ed = max(ceil(len_q/64), loop_st+1)` 覆盖 `len_q=0` | R7：`[0,64]/[64,64]` 不崩、request 0 的 `dk` 严格为 0、request 1 正确 |

### 2. 本轮唯一的实质发现：16-bit GEMM 操作数舍入的容差边界（D 类，已转正 repro）

`P`/`dS` 必须先 cast 到输入 dtype 才能进 16-bit L1 的 GEMM（源 GPU kernel 同样如此）。对一个真值
是小量抵消的 cell，该舍入被 `sum|terms|/|value|` 放大后**可以超出 atol 而 kernel 完全正确**：

- 实测：bf16 `[0,64]/[64,64]` 的 `dv[0,1,77]`：真值 1.21e-4、`sum|terms| = 70.1`（**5.8e5 倍抵消**），
  kernel 给 -5.920e-3，fp64 ideal 给 +1.21e-4 → 差 6.04e-3 > atol 5e-3；而把同一份 `P` cast 到 bf16
  再算的"源语义参照"给 -5.929e-3，**与 kernel 只差 7e-7**。
- 规模：L0-17/18/19（3.7e8…7.4e8 个比对 cell）分别只有 14 / 6 / 3 个 cell 越过 ideal 的 atol，
  且 100% 落在源语义参照的容差内；fp16（11 位尾数）在同几何上全部干净。
- 判定手法（可复用）：不要放宽 atol，而是加**第二个参照**（重复同样的 16-bit 舍入），
  `kernel == truth` 且 `kernel` 只在个位数 cell 上 != ideal ⇒ 固有量；若 `kernel != truth` 则是实现缺陷。
- 该现象 dense Stage 3 已实测（L2-3/B-4 WARN，7.8e-3），但诊断脚本只留在 `/tmp`，且
  Stage 2 检视只把它列为建议 ⇒ 本轮在 **L0 门槛上真实复现**（manifest smoke bf16 1/655360 cell），
  说明"建议级"的措辞低估了它的必然性。已转正为
  `repro/CONST-950-16bit-operand-cancellation.py`（自包含 + 4 组断言，实跑 exit 0）。

### 3. Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Develop-flow | 设计 §8.2 的精度门槛（fp64 ideal + `atol/rtol`）对 bf16 **不可达**：源语义强制把 `P/dS` 舍入到 8 位尾数，强抵消 cell 必然越界。开发 skill 的 `run_case` 模板只给了"单一参照 + assert_close"，没有"第二参照分类"这一档，本轮先撞 L0 FAIL 才建立判据 | `_bwd_kernel.py` 的 `run_case` 第二参照分支；repro/CONST-950-16bit-operand-cancellation.py；L0-1 首次 FAIL（1/655360 cell） | `tilelang-op-develop` 的精度模板：输入 dtype 的尾数 ≤ 11 位时，`{op}.py` 必须同时提供 `golden(quantize=True)` 的源语义参照，并把"越界 cell 数 ≤ 5e-5 且第二参照全过"写成通过条件；`atol` 本身不得放宽 | R |
| Develop-flow | 设计的 L0 计划按 manifest 原样（T=32256、H=128）要求 CPU golden，但本机 cgroup 上限 32 GiB：首版 golden 用全尺寸 fp64 累加器 + 全局 fp32 输入副本，**进程被 OOM kill**（组合脚本在 L0-2 后静默消失，`sys` 时间 4 min 是征兆） | 首跑日志（L0-1/L0-2 后无输出）；cgroup `memory.max = 34359738368`；改后峰值 RSS 13.7 GB | `tilelang-op-develop` 的 golden 模板加三条硬约束：① 输入按 request 切片后再 cast（不要全局 float()/double()）；② 累加器按 (request, head-slab) 粒度写进输出、不留全尺寸 fp64 累加器；③ score 平面原地复用（`s → p → ds`），长请求（>8192）用 fp32 累加；并给出"manifest 级用例可缩减 H"的显式许可 | R |
| Develop-flow | 平台 `execution.working_directory` 不是可选项：在仓库根 import tilelang 会命中仓库内 dev root 的 `tilelang/`，报 `TileLang's private TVM Python package is missing`——与 kernel 无关却像"环境坏了" | 在仓库根执行 `python -c "import tilelang"` 的 ImportError；在任务目录下正常（`tilelang 0.1.15 /workspace/venv-ascend950/...`） | `tilelang-ascend` 的 Stage 3 段：把"必须 cd 到任务目录再跑"从 ruleset 的 `working_directory` 抬成**首跑检查项**，并在报错分类表里加一行"repo-root ImportError = 工作目录错，不是环境缺依赖" | R |
| Review-flow | Stage 2 把 dense 的已知容差边界列为"建议 ②（非阻塞）"，但它在 packed 的 L0 门槛上必然复发（1/6.5e5）。建议级的判据应是"该现象在 Stage 3 的**门槛层**是否必然出现"，而不是"是否影响正确性主线" | REVIEW.md 建议 ②；L0-1 首跑 FAIL | `tilelang-design-review` 维度 6：兄弟实现的 Stage 3 已实测的固有精度边界，若在**本设计的 L0 计划里存在同类用例**，升级为阻塞（须在 §8.2 写明第二参照口径），否则仍为建议 | P |

### 4. Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 16-bit GEMM 操作数的**抵消放大容差边界**与"第二参照"判定法：`kernel == 16-bit-cast 参照` 且只有个位数 cell 越 ideal atol ⇒ 固有量；`kernel != 参照` ⇒ 实现缺陷。bf16 实测 5.8e5 倍抵消、6.04e-3 越界、与参照差 7e-7 | D | 本轮 L0-1（1/655360）、L0-17/18/19（14/6/3 个 cell 于 3.7e8–7.4e8 中）、L2-2；dense L2-3/B-4 的 7.8e-3 | `repro/CONST-950-16bit-operand-cancellation.py`（exit 0，4 组断言） | tilelang 0.1.15 / CANN 9.2.1 / ascend-950pr-9579 | 平台 knowledge（numerical-boundaries 条目 + `PL-950-softmax-boundaries` 交叉引用）+ develop skill 的精度模板 |
| **动态 grid 在 `__mix__(1,2)` 混合 kernel 上运行期成立**：`T.Kernel((total_kv//64 + B)*H)` 随运行期 `T_kv` 变化（9 / 2 / 8160 / 16640 块），同一份编译产物跨不同 `T_kv` 复用且结果正确；设计期把它列为"最大未核验项"，本轮 1 次通过 | D | 探针 R1/R2；L0-17（grid 16640）、L0-19（grid 8160）的 OOB 哨兵 0 触碰 | `history_version/stage3_probe.py`（R1/R2，assert 型） | 同上 | 平台 knowledge（execution-patterns / paged_cache_write 条目补"混合 kernel + 动态 grid 也成立"，此前先例只有纯 Vector） |
| packed 语义的三件套在真机上彼此独立成立：融合式 O(B) 前缀解码（无 `sync_grid`/无跨核 flag）、head-major `[H,T_q]` 统计量（256 B 连续段）、`dk/dv` 双分支 + `dq` 整块原子（溢写行为**逐位** 0） | D | R4（request1 vs solo = 0.00e+00）、R5（错布局炸到 1e1–1e2）、R3/L0-9/B-11、L0-6 | `history_version/stage3_probe.py` | 同上 | 平台 knowledge（attention 族的 packed 变体条目：clamp 只认张量 extent，request 边界必须自己收口） |
| 长 packed 请求的 **CPU 参照内存预算**：按 request 切片后 cast + score 平面原地复用（`s→p→ds`）+ 按 (request, head-slab) 直写输出，峰值 13.7 GB（改前 OOM @32 GiB）。H 缩减是可接受的旋钮（每 block 工作量与 H 无关） | P | 首跑 OOM；改后 `resource.ru_maxrss = 13.7 GB`（16384 token、H=16 的 mixed workload） | `_bwd_kernel.py --level L0`（L0-19 单例即可复现内存形态） | 同上 | develop skill 的 golden 模板 + `tilelang-op-design` 的 L0 计划（要求写"参照实现的内存预算"） |

### 5. Transferable Lessons

- 本迁移任务的后续函数（`_delta_kernel`、`_dq_land_kernel`）若也走 bf16/fp16，**精度门槛必须配"第二参照"**：
  先写 `golden(quantize=True)`（重复同样的输入 dtype 舍入），越界 cell 用它分类；直接放宽 atol 会把真 bug 一起放过去。
- packed 的 CPU 参照**不要**用"全尺寸 fp64 累加器 + 全局输入 float()"：32 GiB cgroup 下 4096×8 请求就会 OOM。
  按 request 切片 cast、score 平面原地复用、按 (request, head-slab) 直写输出是标准形态。
- 动态 grid 与 `__mix__(1,2)` 混合 kernel 在 950PR 上**运行期可用**（本轮一次性通过），所以 packed 类算子不必
  预先退化到"静态上界 + live"；但 `live` 必须包住**整个** body（含全部 DMA），否则非 live 块会读越界
  `cu_seqlens[req]` 并写错地址。
- 请求边界是 packed 唯一不能交给平台的语义：平台 OOB clamp 只认张量 extent（`min(64, max((total_kv-kv_start)-n0, 0))`），
  因此"赋值型"输出（`dk/dv`）必须双分支收口，而"累加型"输出（`dq`）可以靠"溢写恰为精确 0"整块原子——
  后者是可验证的不变式（本轮实测 request1-vs-solo 逐位相同），但前提是掩码保持 `if_then_else` 的选择语义。
- 同一份 kernel 在不同 seed 上会给出"这轮 FAIL、下轮 PASS"的表象：先查是否是高抵消 cell（比值 > 100 倍），
  再决定是不是实现问题；判据是第二参照，不是重跑。


## Stage 2 (design review) Retrospective — 2026-11-24 — 结论 不通过（阻塞 1 / 建议 10）

> 本节复盘**检视流程自身**。设计内容的问题（§8.4 的 `dq` 阈值不可达等）已随 `REVIEW.md` 的建议逐条流转给 Stage 1，
> 不在此重复；以下只记「检视维度/工具自身」暴露的缺口与被证实的判据形态。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Review-flow | 维度 6 的清单只有「Golden / L0 计划 / 按 dtype 的 atol-rtol」，**没有『同一份验证方案内两个阈值是否互相矛盾』这一项**。v0 早已写「`dq_acc` 不做逐位比对、只做 dtype 容差（atol 6.9e-2）」，v1 新加的 §8.4 对同一个量写 `≤1e-5`；`design_calc_check` 与全部 D-1 机械位都不涉阈值，只有人工把 `量级 × fp32 ulp` 算出来才暴露 | `REVIEW.md#问题 1`；`DESIGN.md#§8.2`(1762,1765) 与 `#§8.4 G1`(1816)；`§8.2` 在 v0/v1 间**逐字节相同**（`history_version/design_v0_pre_gridfix.md` vs `DESIGN.md` 区段对比 identical=True），故冲突是 v1 新增判据未与既存判据对齐 | `tilelang-design-review` 维度 6 增子项：「方案内每个绝对阈值须与对应量实测幅值的 ulp 及 §8.2 的 dtype 容差做一次量级自洽核算（`abs_tol ≥ 1 ulp(max|量|)`；若比 §8.2 更紧，必须写成 `k·ulp` 并说明 k 的来历）」 | P |
| Review-flow | 检视**修订版**时 skill 没有「增量面 × 不变面」的双栏工作法。v1 只改任务映射，`design_calc_check` 三个 pass 项（`ub_budget`/`l0c_budget`/`r3_metrics`）数字与 v0 **一字不差**，机械位等于没检；真正需要复核的变成设计新增的纯 Python 算术模块（`launch_plan.py`）与「未改动项」清单是否真的没动 | `REVIEW.md#机械复核 M2/M3`；`DESIGN.md#§12.3` 未改动项清单；`launch_plan.py:176-187`（自检主程序）；v0/v1 的 §4.5、§4.6、§8.1、§8.2 四段区段对比均 identical=True | Phase 1 第 7 步增一条：「若设计引入新的纯计算模块（launch/形状推导），机械复核 = ①直接执行该模块的自检与断言用例 ②对文档表格逐格比对 ③按『未改动项』清单抽 ≥3 项与上一版对比」，不得只贴 `design_calc_check` 的 JSON | R |
| Review-flow | 迁移任务的 Read 范围只规定「源算子文件」，未覆盖源算子的**依赖工具模块**。本函数 packed 分支的三条关键主张（warp shuffle 前缀扫描、`serial_groups=8` 退化分支、二分 decode 语义）全在同仓 `grouped_tiling.py`；设计 §0.4#15 / §0.6 R6 的正确性只能靠读该文件判定 | `REVIEW.md#维度 0`（优化手段行、R6 复核）；`/workspace/TileOPs/src/tileops/_kernels/grouped_tiling.py:74-84,93-104,166-200` | 维度 0 的「算法解读完整性」补齐 Read 范围：「源算子 import 的同仓 tiling/scan/布局工具模块须一并 Read（用 `grep -n "from tileops._kernels" bwd.py` 定位）」，否则『算法步骤覆盖』无法判定 | R |
| Mechanical-check | 平台知识条目被本任务补记后 `toolchain_stamp` 未更新，`kb_stale_check` 仍报 stale：检视方若只信 `kb_stale_check` 会把**已补全**的条目继续降级为「线索」，只信正文又会把**陈旧**条目当依据。skill 无「条目正文 vs 戳记一致性」核对项 | `REVIEW.md#问题 10`；`.agents/platforms/toolchains/ascend/knowledge/traps-950pr-execution.md:39-81`（facet 存在、`toolchain_stamp` 仍 `a010a95…`、当前 `7a52f2f1…`） | 维度 8 的信息源核对补一句：「引用知识条目时分别记录『正文首验日期』与『toolchain_stamp』；两者不一致按正文证据采信，并把戳记滞后作为建议级提出」 | P |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **验证阈值的量级自洽判据**：`abs_tol` 必须 ≥ 1 ulp(该量实测幅值)，且不得比 §8.2 的 dtype 容差更紧；更紧时必须写成 `k·ulp` 并说明 k。反例：fp32 累加量幅值 1.05e3 → 1 ulp = 1.22e-4，而设计写 1e-5（0.08 ulp）⇒ 必然假失败 | P | `DESIGN.md#§8.4`(1e-5) vs `#§8.2`(6.9e-2)；`/tmp/opencode/MLA_VARLEN_BWD_8x4K_FINDINGS.md#§D`（`|dq|≈76`、bf16 上差 2.0）；`REVIEW.md#问题 1` | 手算无需设备：`76×13.856=1.05e3`；`2^-23×2^10=1.22e-4`；与 findings §D 的 0.9999 逐位相同比例自洽 | tilelang 0.1.15 / CANN 9.2.1 / ascend-950pr-9579 | `tilelang-design-review` 维度 6 |
| **修订版的「不变面」抽检法**：把设计的「未改动项」清单当可证伪声明，抽 ≥3 项与上一版逐字对比（本次抽 §4.5 / §4.6 / §8.2 三段，全部 identical），据此才允许承袭上一版结论；新增量面做满维度检视 | P | `DESIGN.md#§12.3`；v0 与 v1 的 §4.5/§4.6/§8.1/§8.2 四段区段对比 identical=True；`REVIEW.md#维度 0` 第 3 段 | `python3 - <<'EOF'` 内按 `re.search(r"### 4\.5 …(?=### 4\.7)")` 取两文件区段后比较；本次四段均 True | — | 本 skill（Phase 1 增「修订版」分支）+ `tilelang-op-design` 的 revision 模式 |
| **块索引类平台上限修复的两层证法**：① 任务映射双射（纯算术，可逐 bid 穷举）② 设备码形态（静态除数 / 运行期原点 / 守卫包住 body）。两者都不覆盖运行期写集，故必须再配一条「逐 owner 拆分对照 + 逐 owner 非零能量检」的运行期用例（本次 G1 ④） | P | `verify_equiv.py#check_a4`（568,904 任务、`over_cap=0`）、`design_probe_grid_slab.py`（P6 设备码 `h_off[0] + (block_idx & 1)`）、`DESIGN.md#§8.4 G1` | `cd examples/mla_varlen_bwd/_bwd_kernel && TMPDIR=/workspace ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python history_version/design_probe_grid_slab.py`（本轮 COMPILE_OK，28,381 B，13 marks） | tilelang 0.1.15 / CANN 9.2.1 / ascend-950pr-9579 | 本 skill 维度 4/6 + `tilelang-op-design` 的验证方案段 |

### Transferable Lessons

- 修订版**新增的判据必须与既有判据对齐**：同一份设计里同一个量出现两个阈值（容差 6.9e-2 与 1e-5）时，先算「1 ulp 是多少」再判自洽；原子累加类输出只能做 dtype 容差，逐位比对只适用于单 owner 的赋值类输出（如 packed 的 `dk/dv`）。
- 「不读任何 GM / 不做任何 DMA」这类零成本声明要按设备码逐字核：运行期 launch 标量（如 head 原点 `h_off`）本身是一次 4 B GM 读，成本可忽略但声明不能写成「零」。
- 平台上限类修复的正确性证法分两层（映射双射 = 纯算术可穷举；设备码形态 = 编译级），两层都不覆盖运行期写集，必须再加一条「逐 owner 拆分对照 + 逐 owner 非零能量检」的真机用例，且参照必须是「同一内核、远离上界」的几何。
- 复用上一版结论前先验「未改动项」：预算表 / 容差 / 操作数方向抽 3 项逐字节对比，一致才允许承袭；当机械位数字一字不差时，它等于没检。
- 引用知识条目时同时记录正文首验日期与 `toolchain_stamp`：戳记滞后只说明元数据旧，不代表正文不可信；正文与戳记冲突按正文采信，并把戳记滞后作为建议级提出。

## 2026-11-24 — 设计修订 v2（验收判据修订：`dq ≤1e-5` 不可达）

> 触发：Stage 2 检视（`REVIEW.md`）对 v1 判「不通过」，**唯一阻塞项**是 §8.4 G1/G2/G3 与 §8.3 L0-12/L0-21 的
> `dq` 绝对阈值 `≤1e-5`。grid 修复本体（slab 方案、`h_off`、尾 slab 守卫、启动前断言、写集不交、A4、P6）经
> 检视独立复核**成立**，v2 **未改动**。本节只记「哪类设计判断被推翻、正确依据是什么」。

### 1. 被推翻的判断与正确依据

| 被推翻的判断（v1） | 正确依据 | 证据定位 |
|---|---|---|
| 「`dq` 的跨 launch 一致性可以用一个绝对阈值 `1e-5` 卡住」（想比 §8.2 更严） | 绝对阈值的下限是**该量的 1 个 ulp**：`\|dq_acc\| ≈ 1.05e3` ⇒ fp32 ulp `2^-23·2^10 = 1.221e-4`，`1e-5` 只有 **0.082 ulp** ⇒ 要求两次独立 store-atomic 累加逐位一致，不可达 | `REVIEW.md#问题 1`；`/tmp/opencode/MLA_VARLEN_BWD_8x4K_FINDINGS.md#§D`；`DESIGN.md#§8.2 ③` |
| 「§8.4 是 v1 新增的回归门，因此可以自带判据」 | **同一文档里同一个量只能有一个口径**：§8.2 在 v0/v1 间逐字节相同，早已规定 `dq_acc` 不做逐位比对、只做 dtype 容差（atol 6.9e-2 = 565 ulp）。新增判据必须**引用**它，不能重定义 | `DESIGN.md#§8.2`（v0/v1 identical）vs `#§8.4 G1`(v1)；`REVIEW.md#问题 1` |
| 「`dq` 的 bf16 输出差 `2.0` 说明 fp32 层也差 `2.0`」 | `dq = cast_bf16(dq_acc·scale)`：fp32 层 1 ulp 的差在 bf16 舍入边界上放大成 **1 个 bf16 ulp**（`ulp_bf16(76) = 0.5`、`ulp_bf16(256) = 2.0`）。`2.0` 是 `\|dq\| ∈ [256,512)` 上的 1 ulp，不是 4 ulp；反之若真是 4 ulp，`dq_acc` 层必须有 `≈27.7`（2.3e5 ulp）的差，与「0.9999 逐位相同 + 只差原子序」不相容 | `DESIGN.md#§8.2 ④`；findings §D |
| 「尾 slab 空块不读任何 GM」 | P6 设备码首行就是 `h = h_off[0] + (block_idx & 1)` ⇒ **1 次 4 B GM 标量读**；「零成本」声明必须按设备码逐字核 | `REVIEW.md#问题 5`；`history_version/design_probe_grid_slab.py` |
| 「计划与产物天然同源」（`block_n`） | `plan_launch` 的默认 `block_n=64` 与工厂 `cfg["block_n"]` 是**两个可独立变化的常量**；`block_n=32` + 131,072 token ⇒ 计划说 63,736、真实 grid `4,104×31 = 127,224` ⇒ 静默截断复发而断言仍通过 | `REVIEW.md#问题 2`；`DESIGN.md#§5.6 g`、`#§8.4 G5⑥` |
| 「`L2 = 128 MiB` 是硬件档案」 | 平台 `design_budget`/`knowledge` **无 L2 容量条目**；只有 UB/L1/L0C/L0A/L0B。引用不到条目就只能写成**未文档化假设**，且容量淘汰的结论应对数值不敏感 | `REVIEW.md#问题 6`；`DESIGN.md#§1.6.0 估算行` |
| 「弃选项 #5 的代价 `+5.7%`」 | 单迭代 MAC 合计 `3,407,872`；额外一次 `192×64×64 = 786,432 MAC` ⇒ **+23.1%**（`dK`/`dQ` 各一次则 +46.2%）。分母写出来才能核 | `REVIEW.md#问题 4`；`DESIGN.md#§1.6.1 弃选项` |

### 2. 处置（设计层，逐条落点）

- §8.2 新增**唯一口径**：① 判定形式 `atol + rtol·|ref|`；② **任何绝对阈值 ≥ 1 ulp 且不得比 §8.2 更紧**，
  要更紧只能写 `k·ulp(max|·|)`、`k ≥ 8`；③ 量级锚点；④ fp32→bf16 重量化关系；⑤ 诊断量 `k_ulp`（告警不判失败）。
- §8.4 目标改「四条」，G1/G2/G3 的 `dq` 判据全部指向 §8.2；`dk/dv` 保留逐位相同；G1 ④ 笔误 `dk/dk/dq` → `dk/dv/dq`；
  G5 增 ⑥ 同源后置断言（含反例算术）。
- §8.3 表补齐 L0-13…L0-19 与 L0-20…L0-23，新增 **L0-11b**（`H=64 / head_slab=32` ⇒ 2 次 launch，覆盖「slab 之间
  不得清零」），L0-1/L0-2 的几何统一到 manifest-smoke（`grid=96`）。
- §5.5 的 smoke 行改为 `tiles=12 / slab=8 / grid=96`；③ 改「96 ≈ 3.4 波」。
- §5.6 g / §4.8 增 `block_n` 同源 + 运行期后置断言；§9.1 增风险 19；§9.2 增两条反例（`block_n` 不同源、`dq` 固定阈值）。
- §1.6.0 的 C4/估算行撤掉 `128 MiB`，改未文档化假设 + 数值不敏感表述；风险 6 同步。
- 全文统一「尾 slab 空块 = 1 次 4 B `h_off` 读 + 2 个标量运算」；百分比复核：备选 L1 端口 `−5.8%`（两处 `−5.9%` 改正）、
  `dq` 原子流量 `−50%`（v1 误记 `−7.7%`）。

### 3. 机器验证（本次实跑）

- `verify_equiv.py`：`EQUIV_SUMMARY ALL_PASS`，5 行 `EQUIV_PASS`（A1 fp16/bf16、A2、A3、A4）；A4 的 smoke 行
  `slab=8 launches=1 max_grid=96`。
- 判据可达性：`atol 6.9e-2 = 565 ulp`、`rtol·1.053e3 = 1.053 = 8,627 ulp` ⇒ 总计 `1.122`（`dq` 单位 0.081）。
- `statectl gate --stage 1`：**0 fail / 0 warn**；`design_calc_check`：`ub_budget`/`l0c_budget`/`r3_metrics` pass。

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **「新增判据必须引用既有口径」自检项**：设计里出现第二个阈值时，先算幅度 ulp，再与既存 dtype 容差比较；紧于既有容差的绝对阈值一律改写成 `k·ulp`（k≥8）或直接引用既有容差 | R | `DESIGN.md#§8.2 ②`；`REVIEW.md#问题 1` | 手算：`76/0.07217=1.053e3`、`2^-23·2^10=1.22e-4`、`6.9e-2/1.22e-4=565` | tilelang 0.1.15 / CANN 9.2.1 / ascend-950pr-9579 | `tilelang-op-design` Phase 2/验证方案段 + `design_lint` 可加机械位（`abs_tol` 与 ulp 的比值扫描） |
| **高幅值累加量的「ulp 归一 + 重量化」解释模板**：`dq` 这类「fp32 累加 → cast 到低精度」的量，低精度侧的 1 ulp 差是正常现象；判定应落在 fp32 侧，低精度侧用 dtype 容差（其 rtol ≫ 2^-8，恒覆盖 1 ulp） | P | `DESIGN.md#§8.2 ④`；findings §D（`2.0` @ `mean\|ref\|≈76`） | `ulp_bf16 ∈ {0.5@76, 1@128, 2@256, 4@512}`；`5e-3+1.6e-2·128 = 2.05 > 2.0` | 同上 | 本 skill 的验证方案模板（bf16/fp16 输出 + fp32 中间量并存时） |
| **纯 Python 计划模块的「常量源」断言**：主机侧计划与 kernel 工厂共享 `block_n` 时，必须有运行期后置断言（`(total_kv//block_n+batch)*slab == max_grid`），否则断言对象可能与编译产物脱钩 | R | `DESIGN.md#§5.6 g`、`#§8.4 G5⑥`；`REVIEW.md#问题 2`；反例 `4,104×31 = 127,224 > 65,535` | 手算即可（`plan_launch(8, 131072, 128, block_n=32)` vs 默认 64） | 同上 | `tilelang-op-design` 的 Tiling/启动计划段 + develop 的 wrapper 模板 |

### Transferable Lessons

- **阈值是对物理量的声明，不是表达严格性的手段**：写绝对阈值前先算「1 ulp 是多少」，再算「该阈值是几个 ulp」；
  小于 8 ulp 的绝对阈值在跨核累加的场景里等价于要求逐位一致。
- **新增章节不得重定义既有口径**：先在全文档检索同一个量的所有阈值，逐一对齐；v0/v1 的 §8.2 逐字节相同，冲突
  只可能来自新章节。
- **低精度输出侧的「1 ulp 差」不需要慌张，但要用算式证明它是 1 ulp**：`ulp_bf16(|ref|)` 随幅值指数增长
  （76→0.5、256→2.0、512→4.0），结合「逐位相同比例」与「fp32 侧应有的差」可唯一判定读数。
- **主机侧计划与设备侧产物的常量必须同源，且要有运行期后置断言**：编译期常量一旦分叉，越界断言会「通过」而
  真实 grid 越界，且现象仍是「静默截断 + 变快」。
- **引用不到平台条目的常数一律写「未文档化假设」**，并让结论对该数值不敏感；不要把估算里的便利假设写成「硬件档案」。


## Stage 2 (design review) Retrospective #2 — 2026-11-24 — v2 复审 结论 通过（阻塞 0 / 建议 4）

> 本节复盘**检视流程自身**（第二轮：设计 v2 针对上一轮阻塞项 B1 的修订）。设计内容的问题已随 `REVIEW.md` 流转。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Review-flow | **上一轮的建议没有稳定编号**，导致「修订版是否逐条闭合」只能靠人工比对：我在 v1 的维度 3 里内嵌了两条「建议 2/3」（其中「auto-shrink 时 slab 取 heads 约数」从未进入问题列表编号），v2 的 §12.6 修订对照表只覆盖问题 2–10，**该条被静默丢弃**，本轮才发现 | `REVIEW.md#已闭合项` 末行（「v1 维度 3 内嵌」行）；`DESIGN.md#§12.6` 的「检视建议 9 条」表 | 本 skill Phase 4：REVIEW.md 的每条建议必须有**稳定 ID**（`S1..Sn`，含只写在维度内的），并在 Phase 5 之前输出一行 `SUGGESTION_IDS: S1,…`；revision 模式下第一步做 **ID 集合差**（缺哪条 = 未闭合） | R |
| Review-flow | **「未改动」声明不能只读 changelog**。v2 声称「grid 修复本体零改动」，§12.6 也列了清单；若照读，就无法排除「顺手改了一处 slab 公式」。本轮用 md5（`_bwd_kernel.py`/`launch_plan.py`/`verify_equiv.py`/P6 探针）＋ 四段逐字节区段比对（§3.5.4 P6 表、§5.6 d 写集论证、§5.6 f 备选裁决、§8.4 G4）才确认，并**发现** §1.6.1 A4 行有一处措辞差异（非结构差异） | `REVIEW.md#机械复核 M4` 末段；四段 `identical=True`、A4 段 1,192→1,214 B 仅末尾措辞 | Phase 1 增「修订版」分支：**凡「未改动项」声明，一律用 ① 产物哈希 ② 设计文档区段逐字节比对 双证**，并把「比对后发现的微小差异」如实列出（区分结构差异与措辞差异） | P |
| Review-flow | 第二轮出现了「**判据违反同一文档新立的元规则**」这一形态：v2 在 §8.2② 立了「任何绝对阈值 ≥ 1 ulp、不得比 dtype 容差更紧」，却把 L0-11/L0-11b 的判定留成「差值恒为 2.0 / 预填版 == 清零版 + 2.0（逐元素）」= **0 容差**，与 §8.2①「dq_acc 不做逐位比对」也冲突。skill 没有「新元规则是否回扫全文既有判据」的检查项，也没有「违反自家元规则时是否升为阻塞」的判据 | `REVIEW.md#维度 6 建议 1`；`DESIGN.md` 第 1861–1862 行（`== 2.0`）vs §8.2 ①②；对照 `_bwd_kernel.py:664-672` 已有的稳健实现 | 维度 6 增子项：「设计若新立数值判据元规则，须对全文既有判据做一次回扫（grep 固定阈值/相等断言），遗留违例按**该规则本身的严重度**处理」；并明确「有可用实现」不能作为降级理由（Stage 3 实现的是**文本**，不是上一轮的代码） | P |
| Mechanical-check | `statectl gate 1` 是本轮唯一可靠的 design_lint 入口（`.agents/tools/gate_lint.py` 无独立 CLI），但**检视 skill 的机械复核段只提名 `design_calc_check.py` 与 `verify_equiv.py`**，没提 `gate 1/2`。本轮两条都是我自己摸出来的：`gate 1` 验 DESIGN（0 fail/0 warn）、`gate 2` 验我自己的 REVIEW（同样 0 fail），后者对「结论字面量 + 维度章节」有机器兜底，价值很高 | `REVIEW.md#机械复核 M2`（`gate 1` 输出）；本轮额外跑 `gate 2` → `{"pass": true, "failures": [], "warnings": []}` | 本 skill Phase 1 第 7 步：把「`statectl gate <stage> --task-dir … --dir …`」写进机械复核清单（Stage 1 验待审 DESIGN、Stage 2 验自产 REVIEW），并规定输出 JSON 摘录格式 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **判据可达性的 ulp 锚定法**：给任何数值判据定阈值时，先用「同一 workload 的实测幅值」算 1 ulp，再检查「阈值 ≥ 1 ulp」；「变紧」必须写成 `k·ulp`（k≥8）。反例链：|dq_acc|≈1.05e3 → 1 ulp 1.22e-4；v1 的 1e-5 = 0.08 ulp（不可达）；v2 的 6.9e-2 = 565 ulp（可达） | P | `DESIGN.md#§8.2`③（锚点）、`#§8.4` G1（判据）；`/tmp/opencode/MLA_VARLEN_BWD_8x4K_FINDINGS.md#§D`（|dq|≈76） | 手算无需设备：`76/0.07217=1.053e3`；`2^-23·2^10=1.221e-4`；`0.069/1.221e-4=565` | tilelang 0.1.15 / CANN 9.2.1 / ascend-950pr-9579 | 本 skill 维度 6 + `tilelang-op-design` 的验证方案段 |
| **「放宽判据」时同步证明判别力**：阈值变松必须配「目标 bug 的偏差量级 ≫ 容差」的定量对照（本次：截断 → |Δ|≈76 vs 容差 1.12–1.22 = 60 倍余量），再加独立闸门（单 owner 输出逐位相同、per-owner 非零能量、host 断言、吞吐 tripwire）。只写「容差放宽到 dtype 标准」不足以给 pass | P | `REVIEW.md#维度 6` 第 6 条；`DESIGN.md#§8.4` G1 判定列 + G3「127 个 head 全部非零」+ G4 | 推演（无需设备）：截断使 request 1..7 输出恒 0，参照均量级 76 | — | 本 skill 维度 6 |
| **修订模式的双证法**：产物哈希（实现/脚本）+ 设计区段逐字节比对（声称未改的章节），并列出比对中发现的措辞级差异；配合「建议 ID 集合差」判定闭合率 | P | `REVIEW.md#机械复核 M4`（md5 + 四段 identical + A4 段 1,192→1,214 B）；`DESIGN.md#§12.6` 的对照表 | `md5sum _bwd_kernel.py launch_plan.py verify_equiv.py`；两版同名区段 `re.search` 截取后 `==` 比较 | — | 本 skill Phase 1「修订版」分支 |

### Transferable Lessons

- 统计量类判据（fp32 原子累加 / 归约输出）**只能做 dtype 容差或 `k·ulp` 归一界**；写固定绝对值前先算该量的 1 ulp —— 一次 `|值|≈1e3` 的量上 1e-5 只有 0.08 ulp，必然假失败。
- 「判据放宽」不是低风险动作：必须同时给出「目标 bug 的偏差 / 新容差」的倍数（本次 60×）与独立闸门清单，否则放宽会掩盖真 bug。
- 修订版声称「某部分未改动」时，先用哈希与区段逐字节比对证明，再读 changelog；措辞级差异也要列出（本次 §1.6.1 A4 一行有 22 B 差异，是措辞不是结构）。
- 设计新立的数值元规则（如「任何阈值 ≥ 1 ulp」）要立刻回扫全文既有判据：本次遗留的「== 2.0」严格相等判据就是元规则漏扫的产物，它属于「照字面实现会假失败」的隐患。
- 检视建议必须带稳定 ID（S1..Sn），且**只写在维度小节里的建议也要编号**；否则修订版只闭合「问题列表」里那几条，内嵌条目会静默丢失（本次 auto-shrink 的 slab 约数建议即如此）。

## Stage 3 (develop) Retrospective — first_impl, attempt 1 — 2026-11-24

> 本节复盘 **grid 越界修复的实现与验证**（DESIGN v2 §5.6 / §8.2 / §8.3 / §8.4 / §12.6 H1）。
> 设计判断本身未变更、`DESIGN.md` 未改；所有结论均有本机实测支撑。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Develop-flow | develop skill 的 L0 门槛模板按「代表 shape + 边界 + 多 dtype」组织，**不含「grid 含运行期量 → 越过 65,535」这一检查项**。本案（以及 `chain_speculative_sampling`）的 L0 把 H 缩到 ≤32 以适配 CPU 参考，于是 `grid ≤ 32×tiles` 物理上碰不到红线，bug 只能靠事后复核 manifest 大 shape 才发现。建议把「kernel 的 grid 是否含运行期量」列为 L0 模板的强制自检项 | `_bwd_kernel.py` 的 `run_L0()`（L0-17/18/19 的 H=32/16/16）；DESIGN §8.3 的 L0 表；`TRAP-950-grid-over-65535` 的两次独立命中 | `tilelang-op-develop/SKILL.md` Phase 4 增一条：「grid 依赖运行期 shape 时，L0 必须含一条跨 65,535 的 host 计划用例（或同族 split 对照）」 | R |
| Develop-flow | 平台的 `execution.precision_command` 指向 `/usr/local/python3.12.16/bin/python3`，本机该解释器**没有 torch/tilelang**（`import torch_npu` 直接 ERR99999）。Stage 3 全程改用 `/workspace/venv-ascend950/bin/python`（设计阶段已记录该偏离） | `statectl platform --task-dir …` 的 `execution.precision_command`；`env-950pr.sh` 的 `TILEOPS_950PR_VENV` | `workflow.md`/站点 profile 把 `execution` 的解释器指向 `env-950pr.sh` 解析出的 venv（或在 platform JSON 里给出 `python_fallback`） | R |
| Info-source | `T.Tensor((1,), "int32")` 的运行期实参必须是 **1 维**；把 `(n,1)` 张量的行切片 `h_off[i:i+1]` 传进去时，报错文本是 `kernel _bwd_main input h_off ndim mismatch, expected 1; expected: 2, got: 1`（自相矛盾，`expected` 出现两次且指向不同值），无法从文本判断到底是哪一维错 | 首跑 `/tmp/opencode/s3probe/p5.py`（全部 3 个 case 同错），改 `h_off` 为 1 维后一转通过 | `references/programming.md` 增「标量参数张量的 ndim 必须与声明一致；报错文本的 `expected … expected:` 双写是编译器已知缺陷」 | C |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| `T.Kernel(grid)` 的 grid 余数非 0 时**静默只执行 `grid % 65536` 块**（66,560 → 1,024），不是 abort；head-slab 分解（`grid=tiles*slab` + 运行期 `h_off` + `if h<head`）可把这 66,560 个任务分 4 次发射完整覆盖 | D | `_bwd_kernel.py --level G` 的 G1 行（4×16,640，8 个 request 的 dk/dv 逐位、dq_acc k_ulp 0.25–0.38）；`/tmp/opencode/MLA_VARLEN_BWD_8x4K_FINDINGS.md` §A/截断映射 | `repro/TRAP-950-grid-over-65535-silent-truncation.py`（设备实测：66,560 → 1,024 块；4×16,640 → 66,560 任务，每任务恰 1 次） | TileLang ascend 0.1.15 + CANN 9.2.1, Ascend950PR_9579 (dav-3510), 2026-11-24 | `.agents/platforms/toolchains/ascend/knowledge/traps-950pr-execution.md`（该 entry 的 `repro:` 字段补此文件） |
| 大几何（H=128）的 grid 上界回归用「**同一 kernel、逐 request B=1**（grid 2,080 ≪ 65,535）」作参照：`dk/dv` 逐位、`dq_acc` 0.25–0.5 ulp 收敛，且**逐 head 能量检**是截断的独立判据（截断 ⇒ 整 request/整 head 恒 0） | P | `_bwd_kernel.py` 的 `_split_reference`/`_compare_split`；G1/G2/G3a/G3b 的 per-request 表（dq_acc ≤ 3.8e-6，dq ≤ 2.0e-3，128/126/127 heads 全非零） | `_bwd_kernel.py --level G` | 同上 | `pattern-library`（大几何正确性参照形态）／`bottlenecks` |
| 精度/吞吐 tripwire 必须用**热**测量：新编译程序的首个 launch 在本栈上带 ~145 ms 一次性上传，冷跑 631.8 ms vs 热跑 483.3 ms（后者与 findings 的 488 ms 一致）；用冷跑会把 400–600 ms 的判据判成超限 | D | `_case_grid_cap_split` 的 G1 冷/热两行；G4 `latency 483.30 ms, 29.7 TFLOPS` | `_bwd_kernel.py --level G`（G1+G4） | 同上 | `bottlenecks` / optimize skill 的测量口径 |
| `dq_acc` 的 `k_ulp` 诊断必须用 **`max|Δ| / ulp(max|ref|)`**（DESIGN §8.2⑤ 的原式）；逐元素归一 `max_i |Δ_i|/ulp(|ref_i|)` 会被「理想值≈0」的单元支配（实测报 256，而 top-10 差值全为 0.0），失去诊断价值 | D | L0-12 首跑 `[WARN] k_ulp=256.00 top gaps [0.0 …]`；改为全局式后 `k_ulp max 0.50` | `_bwd_kernel.py --level L0`（L0-12） | 同上 | DESIGN/REVIEW 口径（§8.2⑤）→ 已由实现固化，无需改文档 |
| 「同一常量源 + 发射前事后断言」的落地形态：`factory_config()["block_n"]` 同时喂 `plan_launch` 与工厂，工厂把 `(head_slab, block_n)` **挂在编译产物上**，wrapper 每次发射前用产物自身的值重算真实 grid 与计划比对；两个故意不一致（plan slab 16 vs 产物 32；plan block_n 64 vs 产物 32 → 127,224 块）都抛 `ValueError` | P | `_bwd_kernel.py` 的 `factory_config`/`assert_launch_consistency`/`build_kernel`；`_case_launch_plan`（L0-20 ⑥）通过 | `_bwd_kernel.py --level L0`（L0-20） | 同上 | `pattern-library`（动态 grid 的 host 契约） |

### Transferable Lessons（可迁移教训）

- kernel 的 grid 若含运行期量（token 数、batch），**必须**在 host 侧发射前算并断言 `grid <= 65535`：余数为 0 的越界会 abort，余数非 0 的越界**不报错**、只跑 `grid % 65536` 块并把其余输出留成初值（表现为「更快 + 部分为 0」）。
- 把某一维切成**编译期 slab** 分多次 launch 时，逐 launch 的原点要用一元素 int32 GM 张量传入（`h = h_off[0] + bid % slab`），这样 `bid // slab` 仍是静态除数；尾 slab 用 `if h < heads` 守卫，空块只做一次 4 B 读即返回。
- 切分轴与原子累加器的**归约轴正交**时（head slab 之于 `dq` 的 query/key 归约），累加器只在第一次 launch 前清零一次；每次 launch 都清会丢掉前面 slab 的贡献。
- 大几何的边界回归**不能**用「同一个 grid 量级」的参照（会一起错）：用 B=1 逐 request 单跑（grid 低三个数量级）作参照，再补一条「逐 head 能量非零」的结构判据。
- 编译产物的元数据（本次 `head_slab` / `block_n`）挂在返回的 JIT kernel 对象上是可行的（`JITKernel` 允许 `setattr`），这比在 host 侧另存一份「我记得我传了什么」更能证明「计划 == 产物」。
- 端到端正确性的最后一个闸门是「wrapper 路径 == 直接 launch 路径逐位相同」：本次 8x4k 上 dq/dk/dv 三者 `bitwise=True`，说明多 launch 循环、`h_off` 传递与 `land` 的次序都没有引入偏差。
