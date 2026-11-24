## Stage 1 (design) Retrospective — first_design — 2026-11-23

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Design-flow | 「迁移任务必须先读源码」之外，同族**兄弟任务**(engram 前向)的 `history_version/design_probe_compile.py` 是本次最有价值的构造证据来源（滚动窗口 `slot[i%4]` 的 vld/vsts、`T.serial(rc+3)` halo 循环），但 skill 的 info-sources 只把「examples/ 同类实现」列为通用条目，没提示「同一算子族已在设计的兄弟目录（含 history_version 探针）」应优先检索 | `examples/engram_gate_conv/_engram_gate_conv_fwd_kernel/history_version/design_probe_compile.py`；本次 DESIGN.md §3.5.3 第 1 行 | 在 tilelang-op-design 的 `references/info-sources.md` 增加一条：迁移设计先 `find examples/<同族>/ -name 'design_probe*.py'` 复用已核验构造，避免重复探针 | P |
| Design-flow | R3 复杂度表的「访存 Bytes」极易把「常驻中间量」与「往返流量」混淆（本次首稿把源算子按 per-token partial 误记 80 B/lane，实为 50 B/lane：源算子用原子而非 partial 张量），会直接改变候选排序结论 | DESIGN.md §1.6.0 R3 表（终版 50/48/29/29 B/lane）；首稿旧值已被替换 | 在 `references/algorithm-research.md` §3 R3 附近加一句口径要求：「访存 = 每个候选的**实际**往返（含原子写），常驻缓冲单独一列」 | P |
| Design-flow | D-1 等价验证的 ulp 指标在「精确 0 元素」与「抵消元素」上无意义（spacing→1e-45 导致 ulp 爆到 1e19），首轮 4 项被误判 FAIL；改为「字段尺度相对误差 + 容差预算」双指标后全通过 | `verify_equiv.py` 的 `_stats`（终版）；首轮输出见本文件同目录脚本历史 | 在 skill Phase 2 第 5 步写明 ulp 口径：逐元素 ulp 需对 `|ref|` 设下限或改用「字段尺度 ulp + 容差预算」 | P |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 融合反向的「前导 3 行窗口」结构可在 TileLang ascend 上编译：一维 grid + `T.serial(RC+6)` 内三件事（写 nrm 窗口 / 写 ds 窗口 / stage-2 滞后 token）+ `floormod` 动态槽 + 互补标量 `if` 处理越界（无标量 `else`） | P | `examples/engram_gate_conv_bwd/_engram_gate_conv_bwd_kernel/history_version/design_probe_compile.py`（B1/B2/B3 注释与实现）；DESIGN.md §3.3 | `cd examples/engram_gate_conv_bwd/_engram_gate_conv_bwd_kernel && ASCEND_RT_VISIBLE_DEVICES= /workspace/venv-ascend950/bin/python history_version/design_probe_compile.py` → `PROBE_RESULT: ALL_OK`（7 几何 × 2 kernel） | TileLang ascend 0.1.15 + CANN 9.2.1 (ascend-950pr-9579, dav-3510)；编译期证据，无真机 | pattern-library（新条目：融合反向滚动窗口可编译形态） |
| 滚动窗口深度 = 依赖距离 + 1 的槽位证明（`ds[t]` 消费者 `t..t+3`、`nrm[p]` 消费者 `p..p+3` → 深度 4 且同一槽恰在末次消费后被覆盖） | P | DESIGN.md §0.6 R2「语义保持论证」；`history_version/design_probe_compile.py` 的 `slot = floormod(pn, KS)` | none（纸面证明 + 编译证据） | 同上 | pattern-library（卷积反向 / 滚动窗口设计规则） |
| 权重梯度确定性归约方案：per-core fp32 部分和（核内 token 升序）+ 第二个 launch 按核号升序求和，G ≤ 56 行 → 无需原子、无需清零 launch；fp32 误差上界 n·u（n ≤ 2406 → 1.4e-4）远小于 0.2 容差 | P | DESIGN.md §0.6 R1、§8.2；`PL-950-atomic-splitk`（原子仅 Cube 写回核验）作为反证 | none | 同上 | pattern-library（确定性跨核归约模式） |
| `T.simd` 无 `vexp2`：`exp→exp2` 降代换在本平台不成立（导出表实测只有 `vexp`/`vexpdif`） | C | DESIGN.md §1.6.1 末行（候选 9 弃选理由）；`/workspace/venv-ascend950/lib/python3.12/site-packages/tilelang/ascend/language/simd.py` 导出表 | `python -c "import tilelang.ascend.language as T; print(sorted(dir(T.simd)))"` | TileLang ascend 0.1.15 | pattern-library（API 能力表） |

### Transferable Lessons（可迁移教训）

- 兄弟任务的 `history_version/design_probe_*.py` 往往已核验本算子族需要的 SIMD 叶子构造，动手写新探针前先复用/裁剪它，能把探针数量压到 1 个。
- 复杂度对比表的「访存」一列必须先列清「往返（含原子/部分和写回）」与「常驻缓冲」两件事，否则会把会改变候选排序的常数写错（本次首稿把 50 B/lane 写成 80 B/lane）。
- 等价验证的 ulp 指标要对「精确 0 / 抵消」元素设下限，否则同一份通过的实现会被报成 1e19 ulp；判定以「容差预算 + 字段尺度相对误差」双指标为准。
- TileLang ascend 的 traced 前端对**标量 `else`** 不可靠：用互补的两个标量 `if`（`if a: if b: … / if not b: …`）替代，且越界行「不载入只写零槽」，可同时满足源算子语义与 `inf/nan` 边界（clamp 后乘 0 会让 `0·inf = nan` 污染结果）。

## Stage 2 (review) Retrospective — first_review — 2026-11-23

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Review-flow | 维度 0 的「算法解读完整性」只要求「步骤无遗漏/无臆造」，但**结构性计数**（launch 数、kernel 段数）错标不会被任何检查项拦截：本次 §0.3 步骤表按 launch#2..#5 编号正确，却在 header/§0.4/§0.6/§1.4/§1.6.0/§3.5.2 六处写成「4 次 launch（清零+三段）」 | `examples/engram_gate_conv_bwd/_engram_gate_conv_bwd_kernel/DESIGN.md` §0 line 4、§0.4 #11、§0.6 R2、§1.4 line 345、§1.6.0 R3、§3.5.2；源 `gate_conv_bwd.py` 的 5 个 `T.Kernel`（70/78/119/151/206） | 在 tilelang-design-review SKILL.md 维度 0「算法解读完整性」行补一条核对项：「§0.3 的 launch/阶段编号与源码 `T.Kernel` 块数一致（计数亦属事实）」 | R |
| Review-flow | 机械复核的 `design_calc_check.py` 在本设计上 3/4 项 `skip`（ub_budget 无 per-level 表、l0c 无 Cube、core_split 解析器把 §5.5 示例里的 `M=64` 误当算子维度）→ 可机检项实际全靠人工复算。跳过原因不写清时，Reviewer 容易把 skip 当 pass | `python3 .agents/tools/design_calc_check.py --design examples/engram_gate_conv_bwd/_engram_gate_conv_bwd_kernel/DESIGN.md --task-dir examples/engram_gate_conv_bwd` 输出 `{'M': 64}` | 在 design_calc_check.py 的 core_split skip detail 打印被解析到的 dims 来源行号；或在 skill 中要求 skip 时 Reviewer 显式记录人工复算的算式 | R |
| Review-flow | 「验证方案」维度的判据只查「有 L0 计划 / 有 golden / 有容差」，但 §8.3 里**附加断言写错**（「seq<4 时 d_vnorm 应为 0」）不会被现有检查项定位——这类断言错会在 Stage 3 变成误判失败的测试 | DESIGN.md §5.4 `seq_len<4` 行与 §8.3 #3；源 `gate_conv_bwd.py:167-174`（`dst_t=tid+3-p`，`t=0` 时仅 `p=3` 有效） | 在 tilelang-design-review SKILL.md 维度 6 增一条：「L0 用例若带语义断言（非容差断言），须逐个用源码语义复核其期望值」 | R |
| Review-flow | 伪代码级**守卫/裁剪缺口**（`if tp < seq_len`、归约输出 `cols` 越过 `d`）在维度表里没有明确落点：既不是 Tiling 的非整除策略（策略已给），也不是循环结构（结构已给），只能塞进维度 3/7 的建议 | DESIGN.md §3.3（step (c) 无 `tp<seq` 守卫；`copy(tot[:cols] -> DWH[off:off+cols])` 在 `d%64≠0` 时越界 28 float）；`history_version/design_probe_compile.py:531-541` 同形（其几何 d 全为 64 倍数未暴露） | 在 SKILL.md 维度 3 的「非整除处理」行加子项：核对「伪代码中每个动态索引的读/写两侧都有上界守卫（`pds<seq`、`tp<seq`、列裁剪到 `d`）」 | P |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 迁移检视的高收益核对手法：把 §0.1 的每条梯度公式与源码**逐语句**对齐（含结合律：`dot_h=Σ H·w_h·ddot·k_n` vs `Σ H·(w_h·u_h)`），比只看步骤表更容易发现语义偏差；本次 12 条全对上，仅计数标错 | P | DESIGN.md §0.1 line 36-51；`/workspace/TileOPs/src/tileops/_kernels/sequence_modeling/engram/gate_conv_bwd.py:90-294` | none（人工核对流程） | TileLang ascend 0.1.15 + CANN 9.2.1 (ascend-950pr-9579) | tilelang-design-review SKILL.md 维度 0 |
| 「用伪代码的循环边界 + 依赖距离反推窗口深度是否最小充分」的检视法：由 `pn=t0-3+i` 与 `tp=t0+i-6` 得读写迭代区间 `[i-3,i]`，直接证明 4 槽既必要又充分，并暴露「同一槽最早覆盖在 i+4」这一容易被写成 lead=2/4 的错误点 | P | DESIGN.md §0.6 R2 / §3.3；`history_version/design_probe_compile.py` 的 `slot = T.floormod(pn, KS)` | none | 同上 | pattern-library（滚动窗口设计规则）/ SKILL.md 维度 5 |
| 负向断言的复核必须打开被引 pattern 条目读**适用范围句**：`PL-950-atomic-splitk` 的限制句「两个 Cube 核 … FP32 L0C … set_atomic」正是「UB→GM 原子未核验」的依据；`TRAP-950-simd-shared-reduce` 的「范围限于二维 FP32 高层路径」正是「换显式寄存器归约」的依据。只复算数字不读适用范围会放过错误前提 | P | `.agents/platforms/toolchains/ascend/knowledge/execution-patterns.md:48-68`；`numerical-boundaries.md:81-100`；DESIGN.md §0.5/§1.6.0 R4 | none | TileLang ascend 0.1.15 + CANN 9.1 model 记录 | tilelang-design-review SKILL.md Phase 1 第 5 步 / 维度 8 |
| `tilelang.ascend.language.simd` 实测导出：`vexp`/`vexpdif`（**无 `vexp2`**）、`vcadd`/`vdupv`/`vdup`/`vadds`/`vmuls`/`vmadd` 全部存在 → `exp→exp2` 降代换在本平台不成立，且候选 9 的「API 缺失」弃选依据成立（文档将其拼作 `vexpdiff`，实为 `vexpdif`） | C | `/workspace/venv-ascend950/lib/python3.12/site-packages/tilelang/ascend/language/simd.py:860,1182`（`grep -c exp2` = 0） | `grep -n "^def v" /workspace/venv-ascend950/lib/python3.12/site-packages/tilelang/ascend/language/simd.py` | TileLang ascend 0.1.15 | pattern-library（API 能力表） |
| 本平台 950PR 与 950DT 的 UB/L1/L0C 容量**相同**（均为 253,952 / 524,288 / 262,144 B），950DT 参考实现注释里的「216 KiB UB」是陈旧文字 → 迁移容量结论不能照抄兄弟实现的注释，须回读平台 `capacity_bytes` | D | `.agents/platforms/hardware/ascend-950dt-9582.json` vs `ascend-950pr-9579.json`；`/workspace/TileOPs-ascend950/src/tileops/kernels/ascend/sequence_modeling/gate_conv_bwd.py:24` | `python3 .agents/tools/statectl.py platform --task-dir examples/engram_gate_conv_bwd` | CANN 9.1.0 model ini（md5 2f0eca1f…）+ CANN 9.2.1 工具链 | pattern-library（硬件常数）/ ascend-950dt-9582-notes.md |

### Transferable Lessons（可迁移教训）

- 迁移检视别只比「步骤表齐不齐」，要把每条梯度公式连**乘法结合顺序**一起对源码；顺序不同会改变末位，虽然通常无害，但必须能说清「逐位相同 vs 容差内等价」的边界。
- 滚动窗口设计的最小深度可以用「写迭代区间 ∩ 读迭代区间」直接反推（本次 `[i-3, i]` → 深度 4），这比读设计的文字论证可靠；顺手检查读写两侧用的是 `floormod` 还是 truncmod（负位置槽号会错位）。
- 复核「某 API 不支持」这类负向断言时，必须读被引 pattern 条目的**适用范围句**，而不是只对数字；同时用导出表直接验证 API 存在性（本次 `vexp2` 确认不存在）。
- 兄弟目标（950DT）实现的注释里的容量数字可能是陈旧的；容量一律回读任务平台的 `hardware.capacity_bytes`。
- 伪代码里每个动态索引（行 `pds/tp`、列 `off+cols`）都要读、写两侧各有上界守卫；只写一侧守卫，短尾 chunk 或 `d%64≠0` 就会越界。

## Stage 3 (develop) Retrospective — first_impl — 2026-11-23

### 本次实现的结论摘要

- 产物：`_engram_gate_conv_bwd_kernel.py`（1183 行：2 个 launch 的 kernel + 内嵌 golden + L0/L1/L2/boundary + `--level` 入口），
  编程模式 **SIMD**，结构 = DESIGN §1.4 的「融合单计算 kernel（4 槽滚动窗口 + per-core 部分和）+ 有序归约 kernel」。
- 真机结果（`--level all` 退出 0）：**L0 10/10、L1 6/6、L2 4/4、boundary 5/5**，共 25 例；
  6 输出最坏绝对误差：`dH` 1.22e-04、`dk` 9.77e-04、`dv` 1.95e-03、`drms_w_h` 3.60e-04、`drms_w_v` 4.45e-04、`dconv_w` 1.19e-02
  （`dconv_w` 最大值来自 `(3,401,256,bf16)`，其场尺度 5.57，占 rtol 0.2 的 1.1%）。
- 确定性：所有 `repeat>1` 用例（含 `(3,401,256)`、`(2,512,256)`、`(3,96,768)`、`(64,2,64)`）**6 个输出逐位相同**；
  `dY=0` 用例 6 个输出**精确为 0**。
- UB 占用（`_ub_footprint` 复算，152 B/通道 + 16 B/位置）：d=768/seq=96 → 115.5 KiB（46.6%）、
  d=1024/seq=512 → 160.0 KiB（64.5%）、d=256/seq=512 → 46.0 KiB（18.5%）。
- 单几何编译 6–9 s（fused + reduce 两个 kernel），`--level all` 全命中缓存时 4.6 s；
  单次调用（2 launch）0.04–1.05 ms。

### 与 DESIGN 的实现层偏离（均不改数学语义）

1. **槽号键在迭代号上**：`slot = floormod(i, 4)`，读侧 `floormod(i - k, 4)`；等价于设计的 `floormod(pn, 4)`（每核常数平移），
   但让所有槽表达式**可证非负**——`pn = t0-3+i` 在首个 chunk 的前 3 次迭代为负，`floormod(-3,4)` 是唯一不该依赖的行为。
2. **新增 `zraw` 4 槽窗口（+16 B/通道）**：设计 §3.3 伪代码用 `nrm`（=`vhat·rrms_v·w_v`）当 `zd = vhat·g` 的 vhat，
   与 §1.3/§1.6.1 O3 的公式（`dot_v = Σ vhat·w_v·g`、`drms_w_v = Σ g·vhat·rrms_v`、`corr = rrms³·vhat·dot_v/d`）
   不一致；实现按公式表取**原始 vhat**，因此多一个窗口。§3.1 的 S5/S6 文字须相应订正。
3. **去掉 `gz` 缓冲**（设计 §4.3 有）：`dalpha` 在 pass B 内随 `dv` 一起累加，`dvhat` 不再被后续 pass 复用 → −4 B/通道。
4. **输出行独立**：`dhrow/dkrow/dvrow` 三行，不复用 `hrow/krow/vvrow`。设计的探针在 pass B 把 `dv` 写进 `hrow`，
   而 pass C1 还要把 `hrow` 当 `H` 读——照抄会**破坏 gate 段全部结果**（编译不报错）。
5. **标量条带行距补齐到 32 B**：`seq_pitch = max(8, ceil(seq_len/8)*8)`（见下 Value Point 1）。
6. **`out_idx`**：launch#1 `[12,13,14,15]`（dH/dk/dv/PART），launch#2 `[1,2,3]`（三个 fp32 权重梯度），
   入口闭包按算子契约返回 6 个输出。REVIEW 维度 7(i) 指出的「12..18」是「单 kernel 带全部输出」的形态，
   本实现为两次 compile 的两套签名，按各自签名对齐即可（无参数绑定风险）。
7. **`d` 上界**：只实现主选，`MAX_HIDDEN_DIM = 1024`（`_check` 超界抛 `ValueError`）。
   **Stage 5 的 `refusal()` 必须写 1024，不得沿用 manifest 的 4096**（DESIGN §9.2 R-1）。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| develop / Phase 2 | 设计的**编译探针不覆盖语义**：探针（14 次编译全 OK）里有两处会算错/破坏结果的结构（`zd` 取 `nrm`、输出复用输入行），设计期与 Stage 3 首轮都只能靠人工重推公式发现 | `history_version/design_probe_compile.py:352-361`（`zd = vmul(nrmw[...], dn)`）与 `:387`（`dv` 写进 `hrow`）随后 `:406` 把 `hrow` 当 `H` 读；DESIGN §1.3 公式与 verify_equiv O3 均要求原始 `vhat` | 在 design skill 的探针要求里加一条：「探针除编译外，至少对**一个**几何跑通 torch 数值对照（或明确声明『仅编译』并列出未验证的语义点）」；Stage 3 skill Phase 2 加一条「伪代码的每个操作数回查 §1.3 公式表」 | P |
| develop / Phase 4 | 「一次 L0 通过」不足以判可见性正确；但本次**块内 mem_bar 已按兄弟任务补齐**，无需试错 | `examples/engram_gate_conv/_engram_gate_conv_fwd_kernel/RETROSPECTIVE.md` 的 Stage 3 VP 1；本实现 5 处 `mem_bar("VST_VLD")` 后所有 repeat 用例逐位一致 | Phase 4 的「L0 连跑两遍」建议已在兄弟任务登记，本次直接受益，无需新增 | D（沿用） |
| develop / Phase 4 | 失败信号是 **device error 507035（无栈、无诊断）**，被 `run_case` 归为 `RuntimeError` 记录，看起来像「运行失败」而不是「对齐契约违反」；日志里没有任何编译告警 | `(1,4,256)`/`(2,1,256)`/`(3,401,256)` 三种几何同错，`(1,8,256)` 同代码通过 → 定位到 `seq_len*4 B < 32 B` 与 `seq_len % 8 != 0` | 在 develop skill 的失败处理表加一行：「`device error type 3, code 507035` + 无编译诊断 ⇒ 先查 `T.copy` 目的区域的 **行距/地址 32 B 对齐**（2D UB 缓冲的 pitch 必须 8 个 fp32 的倍数且 ≥ 8）」 | D |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **`T.copy` 写入 2D UB 缓冲时，目的缓冲的每行必须 32 B 对齐**：fp32 行距须 ≥8 元素且为 8 的倍数。违反时编译无告警、真机直接 `device error type 3, error code 507035`。`T.copy` 的元素数不受此限（每行拷 1 个 fp32 也合法，只要行距 32 B） | D | 同一 kernel：`sc=(4,seq_len)` 时 `seq∈{1,2,4}` 与 `seq=401` 全部 507035，改为 `(4, max(8,ceil(seq/8)*8))` 后全部通过；最小复现里 pitch=1 必错、pitch=8 正确、pitch=408 正确 | `examples/engram_gate_conv_bwd/_engram_gate_conv_bwd_kernel/repro/TRAP-950-simd-strip-pitch.py`（A/B/C 断言 `REPRO_RESULT: ALL_ASSERTIONS_HOLD`；`--fault` 复现 507035） | tilelang 0.1.15 + CANN 9.2.1 / ascend-950pr-9579 / 2026-11-23（真机） | 新增 `knowledge/numerical-boundaries.md` 条目 `TRAP-950-simd-strip-pitch`；`PL-950-simd-reduce.md` 的「标量条带」段引用 |
| 「编译探针全绿 ≠ 伪代码语义正确」：本设计的探针把 `nrm`（`vhat·rrms_v·w_v`）当成 `zd = vhat·g` 的 vhat，并把 `dv` 写进 `hrow` 覆盖了随后要读的 `H`——两处都不会被编译期发现，只能在实现时逐操作数回查公式表 | P | `history_version/design_probe_compile.py:352-361, 387, 406`；DESIGN §1.3 line 328-329 与 verify_equiv `backward()` 的 `zd = z * d_vnorm`（z = 原始 vhat）；实现改为 `zraw` 窗口后 6 输出全部达容差 | none（人工回查；数值证据为本次 25 例） | 同上 | `.agents/skills/tilelang-op-develop/SKILL.md` Phase 2 + tilelang-op-design 的探针要求 |
| 环形窗口的槽号键在**迭代号**而非行号上，可消除「负行号 `floormod`」这一未验证行为，且与设计的 `floormod(pn,4)` 等价（每核常数平移）。判据：写 `floormod(i,4)`、读 `floormod(i-k,4)`，所有参数非负 | P | 本实现 `_build_fused` 的 `slot`/`T.floormod(i - p, KS)`；DESIGN §0.6 R2 的依赖距离论证不变；L0-6（`seq=1`）、L0-2（短尾）、L0-7（素数 401）逐位确定性通过 | none | 同上 | `knowledge/PL-950-vector-execution.md`（滚动窗口索引规范） |
| 权重梯度确定性归约在真机上成立：per-core fp32 部分和（核内 token 升序）+ 第二 launch 按核号升序求和，25 例中全部 `repeat` 用例 6 输出**逐位相同**，且 `drms_w_v/dconv_w` 与 golden 的相对差分别为 1.6e-3/2.1e-3（rtol 0.2 的 1%） | D | 本文件摘要；`run_case(repeat=2)` 的 `torch.equal` 分支；`(3,401,256,bf16)` 一次求和 1203 token | `cd <task dir> && ... python _engram_gate_conv_bwd_kernel.py --level all` | 同上 | `knowledge/execution-patterns.md`（`PL-950-atomic-splitk` 的确定性替代：有序归约模式实测成立） |

### Transferable Lessons（可迁移教训）

- **2D UB 缓冲做「按行条带」暂存时，先把行距按 32 B 补齐**（fp32：`max(8, ceil(n/8)*8)`）。症状是 device error 507035、
  编译无诊断、且只在 `n*4 B` 不是 32 B 倍数的几何上出现——先按 `seq_len` 与 8 的关系筛一遍再怀疑算法。
- **设计探针的「编译通过」只证明构造可编译**：凡探针里出现的操作数，实现时都要回查设计的公式表（本次两处：`zd` 用了
  `vhat_norm` 而非原始 `vhat`；`dv` 写进了随后要当 `H` 读的行）。回查成本 10 分钟，真机调试成本 1 小时。
- **复用输入行当输出行是陷阱**：融合 kernel 的 dtype 暂存行数量很便宜（d=1024 时一行 2 KiB），不要为省一行破坏后续 pass 的输入。
- **槽号用迭代号而非行号**：滚动窗口里 `floormod(行号, 4)` 在首个 chunk 会吃到负数，改用 `floormod(迭代号, 4)` 后
  数学等价而所有下标可证非负，编译期就能保证不越界。
- **device error 类失败要用「同代码不同几何」做二分**：`seq∈{1,2,4}` 错 / `seq=8` 对 / `seq=401` 错 / `seq=408` 对，
  一次就把「4 字节行距」与「非 8 倍数长度」两个成因锁定；比读 IR 快得多。
- **多波次（`M > 56`）不需要额外处理**：`(64,2,64)`（grid=64 > 56 个 AIV）实测正确且逐位确定，
  DESIGN §9.2 R-9 的「无跨块同步 → 波次安全」在实现层成立。
