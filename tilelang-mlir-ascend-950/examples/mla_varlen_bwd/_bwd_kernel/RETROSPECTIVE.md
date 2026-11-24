# Stage 1 复盘（mla_varlen_bwd / _bwd_kernel）

> 逐次追加。每次 `DESIGN.md` 产出后写一节，最新在最上。

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
