# 复盘（mla_varlen_bwd / _delta_kernel）

> 逐次追加。每次 `DESIGN.md` 产出后写一节，最新在最上。

## Stage 3 (develop) Retrospective — first_impl — 2026-11-23

**结论**：`[PRECISION_PASS]`，真机 L0 15/15、L1 4/4、L2 3/3、Boundary 10/10（共 32 例，
`max_diff = 1.144e-05`，atol/rtol = 1e-4），`--level all` exit 0，总计 54.3 s（含编译缓存命中）。
产物：`_delta_kernel.py`（523 行，kernel + `golden_delta` + `run_L0/L1/L2/boundary` + `main`）。

### 本轮的关键发现（价值最高处）

- **动态界流水仍保留双版本**（设计 §6.3 的核心待验项，dense 的 R10 结论在 packed 下不成立）：
  `T.Pipelined(it0, it1, num_stages=2)` 的界是运行期表达式，生成码仍是 `((it − it0) & 1) * 8192`
  （`do_ub` 再 `+16384` half），同步 event 也是 `(it − it0) & 1`，`D_v=64/128/192` 三档几何**均无**
  `did not enable multi-buffering` 告警；三条描述符 count 全为 `min(64, total_q − ((it>>k)*64))`。
  已转正为 `repro/PL-950-dynamic-pipeline-2versions.py`（源码断言 + 真机 T_q=65/200 数值断言，实跑 PASS）。
- **pad 路径的 `T.clear` 覆盖两版本**（dense R13 的假设在 packed 动态 extent 下仍成立）：`D_v=64` 生成码
  两段 `for i < 4096 { … ×2 vec }`（o_ub @0、do_ub @16384 half），各 8192 half/版本；该几何的 `d_ub`
  也带 `(it − it0) & 1) * 512` float 版本偏移，`D_v=128/192` 的 `d_ub` 是单版本（与设计 R6 一致）。
- **运行期 clamp 的三处 extent 在真机成立**：残块几何（`T_q = 1/63/64/65/96/200`、`D_v = 64/128/192`、
  `H = 4/8/128`）全部通过；`H=128 & T_q=63`（跨 head 污染最敏感的形态）与“先残块后全零”的间接探测器
  （B-8/B-9，全零用例 `max_diff = 0`）都没有脏写。
- **`T_q = 0` 可跑**：`ntiles = 0` 时 56 核全空循环，`out_idx=[2]` 分配的 `(H, 0)` 输出正常返回（0.014 ms），
  设计 §5.4/A4 只做了编译与等价性验证，本次补上真机证据。
- **真机时延基线**（warm，均含 host 侧 dispatch）：`ds-v3-8x4k` bf16 **1.424 ms** / fp16 **1.415 ms**、
  `ds-v3-mixed` bf16 **1.390 ms**、`kimi-k2-4x8k` bf16 **0.734 ms**；smoke（T_q=640,H=8）0.022 ms；
  空输入 0.014 ms。供 Stage 4 对账设计的流量下界（1717–2101 µs）。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Develop-flow（L0 分层） | 设计的 §8.3 把「运行期 clamp 的数值闸门」（T_q=63/65/200）与 `D_v=192`（nvec=2）放在 **Boundary**（不阻塞），但动态 extent 的越界写是**精度错误**而非告警项；conductor 也要求这些形状进 L0 | 设计 §8.3 表（Boundary 行「运行期 clamp 的数值闸门」）；本次把 63/64/65、`D_v=64`、`D_v=192`、多 request、确定性全部上提到 L0 后才 15/15 | 在 design/develop skill 的分层判据里固化：**「某条 extent 是新引入的运行期表达式 → 它的残块/短块用例必须是 L0 闸门」**，Boundary 只留非有限/全零/空输入这类语义边界 | R |
| Develop-flow（测试夹具正确性） | 设计 §8.3 的契约用例写「`dim_v=48`（非 16 倍数）应拒绝」，但 `48 = 3 × 16` **是** 16 的倍数 → 该用例不可满足，首次跑 L0 因此 14/15（`dim_v=48: no error raised`）；改为 `dim_v=40` 后通过 | 首跑日志 `[L0][FAIL] L0-6 … dim_v=48 (not a multiple of 16): no error raised`；`_delta_kernel.py` 现用 40 | 设计文档给「非法值」夹具时**同时给出算式与商**（如「40 = 16·2+8」），或直接写「最小非 16 倍数的正偶数 40」；reviewer 可加一条机械检查：`value % 16 != 0` | R |
| Develop-flow（计时口径） | 每编译出一个新 kernel，其**首次 launch** 在本机固定多付 ~105–137 ms（模块/二进制设备侧加载），直接读数会把 smoke 用例从 0.022 ms 误报为 106 ms；Stage 4 若沿用同类 run_case 会把常数噪声当成功耗差异 | 首跑 L0-1/L0-3/L0-5/L0-10 均 ~105–137 ms，同一 kernel 第二次起 0.016–0.24 ms；加 warm-up + `torch.npu.synchronize()` 后同一批用例为 0.017–0.243 ms | kernel 模板的 `run_case` 固定「先 warm 一次并 sync，再计时」，并在日志里区分 `first_launch`/`steady_launch` | P |
| Develop-flow（工厂缓存键） | packed 分支的 grid/取址都**不含** `batch`，但工厂键仍按源 `@target_cache` 含 `(batch, heads, dim_v, 0, dtype)`：本次 `ds-v3-8x4k`（reqs=8）与 `ds-v3-mixed`（reqs=6）几何相同却各编译一次（~27 s/次），L0 的 3 条 manifest 因此多付一次编译 | 跑测日志里 `Found kernel '_delta_main' in memory cache` 与两次 `begins to compile` 交替出现；`ds-v3-mixed` 复用的是**另一**实例 | 迁移说明里注明「键里的 `batch` 只为 host 侧调用点兼容而保留；上层若能保证同 `(heads, dim_v, dtype)` 传同一 `batch` 可省一次编译」，或在 harness 侧把 packed 的 `batch` 归一为 1 | P |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 运行期界的 `T.Pipelined(it0, it1, num_stages=2)` 仍保留 2 版本缓冲：生成码带 `((it − it0) & 1) * 8192`（第二操作数 `+16384`）与 event = `(it − it0) & 1` 的成对 `asc_sync_notify/wait`，`annotate_buffer_versions` 生效；同 loop body 内三条描述符的 count 可以是运行期 `min(64, total_q − t0)` | D | `_delta_kernel.py` 的 `ds-v3-8x4k/mixed/kimi` 三档几何（`D_v=64/128/192`）生成码与全量真机 PASS；`repro/PL-950-dynamic-pipeline-2versions.py` 实跑 `[OK] 2 versions + 3 dynamic extents + head-major base` + T_q=65/200 数值 PASS | `examples/mla_varlen_bwd/_delta_kernel/repro/PL-950-dynamic-pipeline-2versions.py` | TileLang ascend 0.1.15 / CANN 9.2.1 / ascend-950pr-9579（真机） | `.agents/platforms/toolchains/ascend/knowledge/PL-950-vector-execution.md`（动态界流水 + 多版本） |
| `D_v < DW` pad 路径在**动态 extent** 下的 `T.clear` 覆盖两版本：生成两段 `for i < 4096 { … ×2 vec }`（每段 8192 half = 一版本），且该几何 `d_ub` 亦被编译器双版本化（`* 512` float 偏移）；`D_v=128/192` 的 `d_ub` 为单版本 | D | `_delta_kernel.py` `D_v=64` 生成码；L0-3/L0-4、B-4（`D_v=64 & T_q=65`）数值 PASS（max_diff ≤ 3.8e-6） | `_delta_kernel.py`（`_geometry` 的 pad 分支 + `T.clear` 两侧） | TileLang ascend 0.1.15 / CANN 9.2.1 / ascend-950pr-9579（编译码 + 真机） | `.agents/platforms/toolchains/ascend/knowledge/repro/`（pad+动态 extent 形态）与 dense 的 R13 条目合并 |
| packed `_delta_kernel` 三档 manifest workload 的真机时延（warm，含 dispatch）：`ds-v3-8x4k` bf16 1.424 ms / fp16 1.415 ms、`ds-v3-mixed` bf16 1.390 ms、`kimi-k2-4x8k` bf16 0.734 ms；`T_q=32768,H=128` 的 `ntiles = 65536`（每核 1170 tile） | D | `_delta_kernel.py --level all`（L0/L2 各一次，repeat=3 的均值；`max_diff` 1.14e-5） | `_delta_kernel.py --level L2` | TileLang ascend 0.1.15 / CANN 9.2.1 / ascend-950pr-9579（真机） | `.agents/platforms/toolchains/ascend/knowledge/bottlenecks.md`（mla_varlen_bwd delta 项基线） |
| 新编译 kernel 的**首次 launch** 在本机固定多付 ~105–137 ms（设备侧模块加载），与本 kernel 的稳态时延（0.014–1.42 ms）差 2 个数量级：任何 ascend 计时脚本必须 warm-up 一次并 sync 后再计时 | P | 首跑 L0 的 4 个新 kernel 均 105–137 ms，第二次起 0.016–0.24 ms；加 warm-up 后 `--level all` 全程一致 | `_delta_kernel.py`（`run_case` 的 warm-up 段） | TileLang ascend 0.1.15 / CANN 9.2.1 / ascend-950pr-9579（真机） | `.agents/platforms/toolchains/ascend/kernel_template.py`（run_case 计时口径） |
| 动态行轴的「无静态尾段」实现范式：`n = min(R, total − t0)` 同时进两条装载与写回，行循环保持静态 `R`（多算行不写出），写回基址 `4·(h·total + t0)`；真机 32 例（`T_q = 0/1/63/64/65/96/200/640/3072/3584/32256/32768`）无越界、无跨 head 污染 | P | `_delta_kernel.py` 全量结果；`H=128 & T_q=63`、B-8/B-9 间接探测器、`T_q=0` 空输入 | `_delta_kernel.py --level all` | TileLang ascend 0.1.15 / CANN 9.2.1 / ascend-950pr-9579（真机） | `examples/mla_varlen_bwd/_bwd_kernel/DESIGN.md` 的实施参考（同 packed 动态维口径） |

### Transferable Lessons（可迁移教训）

- 本算子的 packed 分支**没有静态尾段**：`total` 是运行期量，`nfull/nlast` 不存在——不要照搬 dense 的
  「`S % R` 构建期分支 + 静态 `nlast` 三处 extent」，也不要照搬 dense 的 `_geometry`（`nsp/tpb/grid = B·H·nsp`）。
  正确形态是固定 56 核 persistent + 核内动态 tile 流 + 统一运行期 clamp。
- 写回的 count 必须**显式**写 `n`，不要写静态 `R`：静态 `R` 也能编译（编译器会自行 clamp 二维基址的最后轴），
  但那是未文档化行为，一旦 `delta` 变成一维扁平张量就会把残块溢写进下一个 head，且无诊断。
- 动态界的 `T.Pipelined` **不会**丢掉第二个版本（实测三档 `D_v` 均是 2 版本），所以不需要为了“保住双缓冲”
  去做任何手工 ping-pong；反之，若为了静态 extent 把残块段搬到流水外，才会丢多版本（dense 的 R10 形态）。
- head-major 输出的压缩写回只需多一次 **运行期乘法**（基址 `4·(h·total + t0)`），内层 `t` 仍是连续 4 B，
  描述符参数（count/4 B/源间距 32 B）与 dense 逐参数同构——下游 `_bwd_kernel` 按 `delta[h, qoff:qoff+64]`
  取数的 256 B 连续段假设成立。
- `D_v < DW`（本算子 `D_v=64`）必须**两侧** `T.clear` 且目的端切片 `[0:n, 0:D_v]`；动态 extent 下编译器
  仍把 clear 覆盖到两个版本（实测），但这条依赖会在换工具链时最先失效——L0 的 `D_v=64` 数值用例要一直保留。
- 计时先 warm-up：新 kernel 首次 launch 在本机多付 ~100 ms（设备侧模块加载），不 warm 会把 µs 级 kernel
  误读成百毫秒级；Stage 4/5 的任何 A/B 读数都要先排除这一段。

## Stage 2 (review) Retrospective — first_design — 2026-11-23

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Review-flow | `design_calc_check.py` 对本文的 `ub_budget` / `core_split` 均 **skip**（`insufficient parsed dims {}` / `no per-level buffer tables parsed`），而本设计的 UB 预算正是「三档几何 × 双版本」的表格 + 生成码偏移，机械检查完全未覆盖——reviewer 只能手工从 P1/P2/P3 的 shmem 偏移反推（8192 half 版本步长、`do_ub` @16384 half、`d_ub` @16384 float）才确认 67584 B | `REVIEW.md` 机械复核 §2（`checks[].status = skip` 原文）；`history_version/probe_v2_tails.log` 生成码偏移 | 给 `design_calc_check.py` 增加「按 `alloc_shared((R, DW), dtype)` 形式解析 buffer 声明并按 buffer 版本数折算」的兜底解析；解析失败时在 `detail` 里打印它看到的候选行，便于 Stage 1 直接修格式 | R |
| Review-flow | 设计期探针的 `_HITS` 过滤器**已包含** `asc_sync_notify/asc_sync_wait`，但归档的三份日志都来自「只打描述符行」的 dump 脚本（`asc_sync` 命中 0 行），导致 §7.2 的同步表在归档件里**不可复核**；本次靠重跑 `design_probe_packed_v2.py p1` 才补证 | `REVIEW.md` 机械复核 §3；`history_version/probe_v2_tails.log`、`probe_dump_tails_v2.py`（只筛 `asc_copy_*`） | 在 design skill 的探针模板里固定「每形态至少归档一份**过滤器全量输出**」，并在 `DESIGN.md` 证据表注明日志对应的探针名与运行时刻 | R |
| Review-flow | 维度表没有「§9 风险叙述与 pattern-library 已有实测条目的一致性」这一检查位：本设计 §9.1 R1 把「跨步描述符带宽低于连续描述符」列为风险，而 `CONST-gm-to-ub-bw-dilution` 已实测 copy-only A/B 为 0.98×（段宽 ≥ 数百字节时读形态无差异）——多出的保守前提会误导 Stage 4 归因，但现有维度只在「负向断言」口径下查 pattern-library | `REVIEW.md` 维度 8 建议（`CONST-gm-to-ub-bw-dilution`，910B2C 实测）；`DESIGN.md` §9.1 R1 | 在 tilelang-design-review 维度 8「调研结论正确性」③ 增子项：**§9.1 的每条风险/前提也须对照 pattern-library 实测条目，方向相反的保守前提同样记为偏差**（不止负向淘汰） | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 动态 tile 流（`ntiles` 运行期）+ `T.Pipelined(it0, it1, num_stages=2)` 仍保持双缓冲：生成码里 `o_ub`/`do_ub` 的读写地址带 `(it − it0) & 1` 版本偏移（8192 half 步长），并由编译器成对插入 `asc_sync_notify/wait(PIPE_V ↔ PIPE_MTE2/MTE3, event = (it − it0) & 1 或 0/1)` 与 block 尾部 4 个收敛 wait | D | `REVIEW.md` 机械复核 §3（2026-11-23 重跑 P1 的完整命中行：`COMPILE OK`、`37` 行源） | `cd /tmp && ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python /workspace/tilelang-mlir-ascend-950/examples/mla_varlen_bwd/_delta_kernel/history_version/design_probe_packed_v2.py p1` | tilelang ascend 0.1.15 + CANN 9.2.1 + ascend-950pr-9579（无卡编译） | `.agents/platforms/toolchains/ascend/knowledge/PL-950-vector-execution.md`（动态界流水 + 自动同步形态） |
| 设计期探针的行数/日志口径不可靠：同一脚本不同 dump 产物对同一形态给出行数差异（`probe_v2_tails.log` 无行数，全量 stdout 记 37 行，设计文档记 63 行 = P2 的行数），说明「行数」不能作为形态指纹使用；reviewer 若要复核形态应比对**具体生成行**（描述符参数、版本偏移、sync 事件）而非行数 | P | `REVIEW.md` 机械复核 §3（P1 37 / P2 63 / P3 61，文档 P1=63、P3=61）；`DESIGN.md` §3.5.4 | `design_probe_packed_v2.py p1 p2 p3` | tilelang ascend 0.1.15 + CANN 9.2.1 + ascend-950pr-9579 | `.agents/skills/tilelang-design-review`（证据核对口径）或 design skill 的探针模板 |

### Transferable Lessons（可迁移教训）

- 复核「编译器自动插入同步」这类论断时，不要只看设计文档与归档日志——先确认归档日志是否由**过滤器全量**导出；
  若日志只筛了描述符行，直接重跑一次探针（无卡、秒级）比推断便宜，且能顺带核对行数与版本偏移。
- 机械检查器 skip 不等于「无需检查」：UB 预算这类跳过的项，最快的机械复核其实是**生成码里的 shmem 偏移**
  （版本步长 × 版本数 + 各 buffer 基址偏移），比按文档表格重算更可靠——两者互相印证后再下结论。
- 审查风险叙述时把「风险」也当作可被证据推翻的断言：pattern-library 已有实测条目若与设计的保守前提方向相反
  （如跨步列块读实测不慢），应记录为偏差而不是放过，否则 Stage 4 会把归因错挂在该前提上。

## Stage 1 (design) Retrospective — first_design — 2026-11-23

### 任务形态与策略

- 迁移任务（harness 子模式），`_delta_kernel` 的 **packed 分支**（`MLAVarlenBwdOp` 第 1 个 launch）。
  上一轮 attempt 因环境瞬时 `database or disk is full`（`/root/ascend/log` 与 overlay 满）中断，未产出
  `DESIGN.md`，只留下探针 v1 与日志；本次在同目录续做并补齐证据。
- 读源码后确认：**本函数的 GPU 工厂与 dense 版逐字相同**，只有两处 `dense_seqlen` 分支（输出 shape 与写回
  下标）。dense 版已通过检视 + 真机 23/23，故策略定为“复用已验证核心 + 只重推 packed 差异”，文档结构上
  用 §0.8（复用/差异分区）+ §11（对照索引）把这两类内容显式分开。

### 本轮的关键发现（价值最高处）

| 发现 | 内容 | 处置 |
|------|------|------|
| **静态尾段在 packed 下构造性不可用** | dense 的“`S % R` 构建期分支 + 静态 `nlast`”依赖 `S` 为常量；packed 的 `total` 是运行期量 | 改运行期 clamp `n = min(R, total−t0)` 约束三条描述符；另给出**静态化备选 P6**（满 tile 静态 + 残块段）并实测其代价 = 关闭双缓冲 |
| **N3：静态 extent 居然“碰巧正确”** | 写回 extent 写静态 `R` 也能编译，且编译器对 `(heads, total)` 二维基址自带最后轴 clamp，count 变成 `min(64, max(total_q−t0,0))` | 这是**未文档化**行为（改成一维扁平输出即跨 head 污染）→ 设计仍显式写 `n`；把“静态 extent 的越界量”量化进 A4 诊断列（`T=65/H=4` → 252） |
| **`T.clear` 覆盖全部缓冲版本** | 之前担心“双版本 UB 的第二个版本 pad lane 未被清零 → `0·Inf=NaN`”；P2 生成码显示 clear 覆盖 offset 0 与 16384 half 各 16384 half = 两版本 | 解除该风险（§9.1 R13 仍登记为“依赖未文档化行为”，L0 `D_v=64` 为数值闸门） |
| **head-major 不改变写回步长语义** | 目的端最内轴仍是 `t`（4 B 连续），描述符参数与 dense 同构；只多一次 `h·total` 基址乘法 + count 变运行期 | §0.6 R5 给出 P5 设备码原文；消费端 `_bwd_kernel` 的 `[1,256]` 连续段契约因此得以保持 |
| **均衡划分 vs `cpt` 划分** | `cpt` 形态若不 clamp `it0`，`ntiles < (G−1)·cpt` 时会出现**负 trip count**（流水 prologue 下未定义） | 主选用 `ntiles·bx//G`（恒 `it0 ≤ it1`）；`cpt` 形态必须先 `min(bx·cpt, ntiles)`（P7 已编译） |

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Design-flow | 复用型迁移（同一 GPU 工厂的第二个分支）缺少“只需重推差异”的模板：skill 的 Phase M0/R/M1 与 §0 模板都是“整算子从零解读”，复用证据只能自建 §0.8/§11 章节承载 | `examples/mla_varlen_bwd/_delta_kernel/DESIGN.md` §0.8、§11；对照 `examples/mla_varlen_bwd/_bwd_kernel/DESIGN.md` §0.8 的同款自建结构（两次任务独立发明同一章节） | 在 `design-template.md` 增“同工厂多分支迁移”可选章节（复用表 + 差异表 + 未复用项），并在 Phase M0 增加“先判定是否同一源码工厂”的入口判据 | R |
| Info-source | `kb_search` 对“row dot / reduction / vcadd”三组关键词均无命中，而平台 `algorithm-candidates.md` 的 reduction 族只有一行 SIMT reducer——算法调研的参考表对本类“单点积”算子实际上为空，候选全靠结构判据枚举 | `kb_search.py` 三组查询输出 `no entries matched`；`.agents/platforms/toolchains/ascend/knowledge/algorithm-candidates.md` 第 10 行 | 在 `algorithm-candidates.md` 的 reduction 族补一行“逐行点积（row-dot）→ SIMD `vcadd` 或 SIMT reducer，按行数规模选择”，附本任务与 dense 版的实测指针 | D |
| Constraint-doc | 平台文档没有“运行期 extent 的 DMA 描述符”条款：count 可为运行期表达式、编译器另加最后轴 clamp，这两点都只在生成码里可见 | `history_version/probe_v2_tails.log`（count = `min(64, total_q−t0)`）；`history_version/probe_v2_neg.log` N3（clamp 形态） | 在 `.agents/platforms/toolchains/ascend/references/programming.md` 的 `T.copy` 小节补“动态 extent：允许、count 为运行期标量、平台另按最内轴 extent 兜底 clamp（未文档化，勿依赖）” | D |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| `T.clear` 覆盖全部缓冲版本（不止第一个）——`D_v<DW` 的 pad 安全化在双版本 UB 下成立 | D | `examples/mla_varlen_bwd/_delta_kernel/history_version/probe_v2_tails.log`（P2 生成码：offset 0 与 16384 half 各 16384 half）；DESIGN.md §3.4/§9.1 R13 | `cd /tmp && ASCEND_RT_VISIBLE_DEVICES="" python .../history_version/probe_dump_tails_v2.py` | tilelang ascend 0.1.15 + CANN 9.2.1 + ascend-950pr-9579 | `.agents/platforms/toolchains/ascend/knowledge/PL-950-vector-execution.md` 或 `numerical-boundaries.md` |
| 运行期 extent 写回：静态 `R` 被编译器按最内轴 clamp“掩盖”，一维扁平输出即跨 head 污染 | D | `history_version/probe_v2_neg.log`（N3：`min(64, max(total_q−t0,0))`）；DESIGN.md §3.3 注 5 | 同上的 `design_probe_packed_v2.py n3` | 同上 | `.agents/platforms/toolchains/ascend/references/programming.md`（`T.copy` 动态 extent 条款） |
| head-major 输出不改变压缩写回的描述符步长，只改基址与 count——`[H,T_q]` 与 `[B,H,S_q]` 的写回参数同构 | D | DESIGN.md §0.6 R5；P5 设备码 `delta[(it>>7)*64 + (it&127)*total_q] … , 4, mode(4), 4, 32` | `design_probe_packed_v2.py p5` | 同上 | `pattern-library/layout.md`（head-major 统计量写回形态） |
| 动态 tile 流的两种划分：均衡式 `ntiles·bx//G` 恒非负；`cpt` 形态必须先 `min(bx·cpt, ntiles)` 否则负 trip count | P | DESIGN.md §3.3 注 6；`probe_v2_p5_p6_p7.log`（P7 的 `min(...)` 循环界） | `design_probe_packed_v2.py p1 p7` | 同上 | `PL-950-vector-execution.py` / `execution-patterns.md` |
| 静态 extent 备选的代价是**关闭双缓冲**（残块段在流水外写同一 buffer） | D | `probe_v2_p5_p6_p7.log` P6 告警 `did not enable multi-buffering for buffer "o_ub"`；DESIGN.md §1.6.3 备选 | `design_probe_packed_v2.py p6` | 同上 | `pattern-library/traps-compiler.md` |

### Transferable Lessons（可迁移教训）

- 复用同一 GPU 工厂的另一个 layout 分支时，先回答“哪条 dense 结论依赖了静态 shape”——静态尾段、`nsp` 整除
  规则、按 shape 特化编译三项都可能在“行数运行期”的分支里失效，不要连带复制。
- 当输出是“最内轴 = 该分支自己的行轴”（如 head-major `[H,T]`）时，压缩写回的描述符步长与 dense 版同构，
  只有基址多一次 `h·T` 乘法与 count 变运行期；把这条写进设计可避免无谓地重新设计 epilogue。
- 编译器对 `T.copy` 的运行期 extent 会**自行**按最内轴 extent 兜底 clamp。它会让“静态 extent 写错”看起来
  能跑通——**不要依赖**，显式写 `min` 上界，并用“一维扁平输出会怎样”自检该依赖是否存在。
- 动态任务流的分核划分优先用 `ntiles·bx//G` 形态（恒非负、天然均衡）；任何 `bx·cpt` 形态都必须先 clamp
  `it0`，否则小规模输入会出现负 trip count。
- 缓冲版本数（`d_ub` 单/双）由编译器决定，不要假设；UB 预算按双版本上界计，同步 flag 由编译器成对插入。
- 设计期探针若在写日志时失败（磁盘满 / plog 目录不可写），先清 `/tmp` 的 msprof 残留并确认 `df` 再重跑；
  探针异常不要当作“编译失败”记入设计。
