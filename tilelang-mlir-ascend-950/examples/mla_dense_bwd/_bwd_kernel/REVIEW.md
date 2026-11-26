# _bwd_kernel 设计文档检视报告（revision v4 / A5 旋转调度）

## 检视结论

结论: 通过

## 检视元信息

- 文档: `examples/mla_dense_bwd/_bwd_kernel/DESIGN.md`（v4，2504 行，sha256 `b8bbf295…`，与 `.stage_state.json` 一致）
- 任务类型: 迁移（harness 子模式，`MLADenseBwdOp._bwd_kernel`）
- 源算子: `/workspace/TileOPs/src/tileops/_kernels/attention/mla/bwd.py`（`_bwd_kernel`，`bwd.py:109-264`）
- 对照工件: `history_version/design_v3_pre_multibuf.md`（v3，v4 的输入基线）、`examples/mla_varlen_bwd/_bwd_kernel/{DESIGN,REVIEW}.md`（A5 的来源与其检视教训）、`examples/mla_dense_bwd_perf/perf_opt/perf_feedback.md`（Stage 4 归因）、`history_version/design_probe_dense_rotation.py` + `dense_rot_probe_run.log`、`verify_equiv.py`
- 检视维度: **9 项**（0 源算子理解与迁移分析 / 1 API / 2 内存层级 / 3 Tiling / 4 技术约束 / 5 循环同步 / 6 验证方案 / 7 完整性 / 8 算法优化分析）
- 阻塞级问题数: **0**
- 建议级问题数: **6**（全部非阻塞，见末章「非阻塞建议」；其中 2 条来自 varlen v5 检视已发现的判据缺陷与 1 条来自我对其「残差」口径的复算）

## 机械复核（D-1/D-5）

### 1) `verify_equiv.py` 重跑（`/workspace/venv-ascend950/bin/python`，`ASCEND_RT_VISIBLE_DEVICES=""`，cwd=算子目录）

```
| A1 exp 域 P（float16）  | 1048592  | 1 | 0.000e+00 | 1 | 1 | EQUIV_PASS |
| A1 exp 域 P（bfloat16） | 1048592  | 1 | 0.000e+00 | 1 | 1 | EQUIV_PASS |
| A2 块级 mask 特化（64,64） | 76838912 | 0 | 0.000e+00 | - | - | EQUIV_PASS |
| A5 旋转调度代数（版本分配 / drain 奇偶 / clear_accum；96,64） | 71384 | - | 0.000e+00 | - | - | EQUIV_PASS |
A2 fast-path blocks: 9285/18761
A5 detail: schedule_points=478 gb_events=17846 selftest_caught=3/3
EQUIV_SUMMARY ALL_PASS
```

**与 §1.6.1 内嵌表逐字段一致**（含 A5 的 478 / 17,846 / 3-3）。我另读了 `check_a5` / `_a5_events` 的实现：事件流与 §3.3 伪代码**逐语句同构**（半体 A/B、`if kqA > loop_st`、`if kqB < loop_ed`、drain 奇偶），五条不变式覆盖「每 k 恰一次 GB 且递增 / `GB(k)` 读 `EP(k)` 的版本且未被更晚 EP 覆盖 / 填充版本未被 `F(k+2)` 提前覆盖 / `clear_accum` 恰一次 / `F→GA→EP→GB` 顺序」，且注入的三个缺陷（drain 奇偶取反、删 drain、半体 A 读当前版本）**全部被抓到**（`selftest 3/3`）⇒ 该检查器有判别力。

### 2) `design_calc_check.py` JSON

```json
{"check":"ub_budget",  "status":"skip", "detail":"no per-level buffer tables parsed from the memory plan (advisory checker; verify manually)"}
{"check":"l0c_budget", "status":"pass", "detail":"block 64x96 x fp32 accum = 24.0 KB vs L0C 256 KB"}
{"check":"core_split", "status":"skip", "detail":"insufficient parsed dims {'bm': 64, 'bn': 96}; declared logical cores = None"}
{"check":"r3_metrics", "status":"pass", "detail":"all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak)"}
```

- **`ub_budget` 由 v3 的 `pass` 变为 `skip`**：§4.5 新增「版本」列后工具解析不到分项 ⇒ **机械跳过，人工逐项复核**（见建议 6）：
  `st_u 12,288 + dp_u 12,288 + p16_u 6,144 + dsT_u 6,144 + lse_u 512 + dlt_u 512 = 37,888`；
  `tail_u 36,864 + tailv_u 24,576 + out_k 18,432 + out_v 12,288 = 92,160` ⇒ **合计 130,048 B = 51.2% / 253,952 ✓（余 123,904）**；保守兜底 `75,776 + 92,160 = 167,936 = 66.1%` ✓ 均不超容量。
- `l0c_budget` pass 只算单累加器；人工复核 5 累加器合计 `221,184 B = 84.4%` ✓（与 v3 逐字节相同）；L1 `2×106,496 + 98,304 = 311,296 = 59.4%` ✓；L0A 49,152（75%）/ L0B 61,440（93.75%）✓；V=3 旋钮 `3×81,920 + 2×24,576 + 98,304 = 393,216 = 75.0%` ✓。
- `core_split` skip（三维网格）→ 人工已核：`86×128×2 = 22,016` ✓。

### 3) `statectl gate 2`

```
python3 .agents/tools/statectl.py gate 2 --dir examples/mla_dense_bwd/_bwd_kernel
{"action":"gate","failures":[],"ok":true,"pass":true,"stage":2,"warnings":[]}
```

### 4) A5 探针重跑（`design_probe_dense_rotation.py`，四种模式全跑，我自己的输出）

```
[COMPILE] base (T.Pipelined num_stages=2) ... OK        [COMPILE] rot (T.serial + 2 versions + deferred GB) ... OK
[PASS] rot 1 key block/1 iter   B=1 S=96/96 H=1  : dk=4.355e-09 dv=0.000e+00 dq=5.958e-08
[PASS] rot 2 key blocks/n_it=2  B=1 S=128/128 H=1: dk=0.000e+00 dv=0.000e+00 dq=6.704e-08
[PASS] rot tail blocks          B=1 S=320/320 H=2: dk=1.766e-04 dv=0.000e+00 dq=1.672e-05
[PASS] rot 4-D B=2 + tail       B=2 S=512/512 H=4 causal    : dk=4.433e-04 dv=1.776e-04 dq=2.320e-04
[PASS] rot 4-D B=2 non-causal   B=2 S=512/512 H=4 non-causal: dk=2.336e-03 dv=2.538e-03 dq=3.379e-04
[PROTO_SUMMARY] ALL_PASS
[PASS] parity ×5：dk_bit_exact=True / dv_bit_exact=True；dq_max_abs ≤5.960e-08（rel ≤1.79e-07）
[PARITY_SUMMARY] ALL_PASS
[TIME] base B=2 S=4096 H=16 : 3.070 µs   [TIME] rot : 1.968 µs
[TIME] base B=2 S=8192 H=8  : 2.965 µs   [TIME] rot : 1.874 µs   （−36.8%）
```

- 数值/逐位**与设计 §1.6.4 I-2/I-3 的表逐位同值**（1.766e-04、4.433e-04、2.336e-03 …）；我这次 `rot` 的 S=8192 单价 **1.874**（设计 1.870，+0.2%，跑间波动内）。
- **探针的 `rot` 体与 §3.3 伪代码逐语句同构**（半体 A/B、`if kqA > loop_st`、`if kqB < loop_ed`、drain 奇偶、`(kq−1)` 偏移、4-D 块尾写回）——我逐行比对过，故上面这组数据就是「设计伪代码在设备上的实测」。
- `dk/dv` 逐位相同是 A5 的**决定性**证据：若滞后一迭代取错任一操作数版本，`dk/dv` 必然成片出错。

### 5) 代码体积与设备码（我自己从编译缓存量，不只看文档附表）

```
base: device_kernel.asc 20,461 B, asc_mmad=5,  __simt_vf__=3
rot : device_kernel.asc 51,239 B, asc_mmad=16, __simt_vf__=5      ⇒ ×2.504
```

与 §1.6.4 I-1 的 `×2.50 / mmad 16 / vf 5` **完全一致**（绝对字节数 20,341/50,879 与我的 0.6% 差属产物口径：生成源 vs `device_kernel.asc`）；`16 = 5+5+3+3`、`5 = 2 半体 × 2 臂 + 块尾` 的解释成立 ✓。

### 6) 引用核对（逐条查存在性与数值）

| 引用 | 结论 |
|---|---|
| `perf_feedback.md` 的管道忙时（mad 0.769 / mte1 0.680 / mte2 0.720 / fixpipe 0.570 / scalar 0.083 = 2.822 ≈ 墙钟 2.944） | **存在且一致**（我按 2.944/1.30/1.87 复算见维度 8） |
| z3 三份日志的 `unsat` + `manual order/stage … no effect` | **存在**（`logs/probes/z3_{dense,varlen,scheduler_probe}.log`，逐条 grep 到原文） |
| varlen v5 的 `61,913 B`、`3.167→2.092`、`4.267→3.134`、`AIV 1.95–2.10` | **存在且一致**（varlen DESIGN 第 16/1126 行等） |
| varlen v5 **检视**对「`GB` 前无 `PIPE_V`」判据的更正 | **存在**（varlen `REVIEW.md` 建议 5-1：该判据在 v4 生产体上同样成立、无区分力，需改「偏移身份」判据）→ 见建议 1 |
| `examples/mla_dense_bwd_perf/perf_opt/logs/probes/` 的 `bwd_base.generated.cpp` | 存在（19,695 B；与 §6.3 A 的 `(kq−loop_st)%3` 证据同源） |

### 7) 执行上下文

平台 `execution.prototype_command` 仍指向无 torch/tilelang 的解释器且 cwd 在仓根内；§3.5.4 的偏离登记表与 `cd /tmp && source env-950pr.sh && TMPDIR=/workspace venv-python` 固化为 Stage 3 首跑方式 —— 与我的实际复跑命令一致 ✓。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

- **源码核对**：已全文读源码 `/workspace/TileOPs/src/tileops/_kernels/attention/mla/bwd.py`（387 行，`_bwd_kernel` 在 `:109-264`、host 侧 `:338-379`），下列每条结论均附源码行号，并与 DESIGN §0.1–§0.7 逐项对照；本轮 A5 只改语句时序（不碰 `:229-260` 的算式与 `:255-260` 的逐行守卫），故源码侧无新增核对点，B1/golden 的源码核对结论沿用 v3。
- §0.1–§0.3（语义 / I/O / 算法步骤 / 优化手段）本轮未改，仍与源码 `bwd.py:229-260`（`s^T/dP^T/掩码/dS/5 GEMM/atomic`）、`:255-260`（块尾逐行守卫）、`:336`（默认 config）一致；§0.4 #11 的判据已在 v3 统一为「等价替换」，本轮无回归。
- **A5 不触碰语义**：§0.6 R2（Cube+Vector 同 kernel 交替）与 R3/R5/R6 全部保留；v4 只改「语句与谁的并发」，5 个 GEMM 的操作数集合、`clear_accum` 语义、`dV/dK` 的累加顺序、`dq` 的写回行区间逐字不变（§1.6.1 A5 (i)–(v)，我独立推演见「A5 机制复核」）；与源码 `bwd.py:226-251` 的 5 个 `T.gemm` 及其 `transpose_*`/`T.clear` 语义逐条对照无差异。
- B1（per-request 写回）的语义结论保持不变：4-D 视图 + `dk[b, kown:kown+HN, h, :]` 在旋转体里仍在循环之外（§3.3 末段），clamp 维度仍是 `S_kv` ✓。
- golden 独立性未变（§8.1 与 v3 相同，只读 `q,k,v,do`）✓。

### 1. API 可行性 — pass

- 新增形态全部有本轮**编译+运行**证据：`@T.macro`（4 个宏）、`T.serial(n_half)`（**运行期循环界**）、静态两份缓冲（`_s0/_s1`，无运行期版本索引）、`if kqA > loop_st` 与 `if kqB < loop_ed` 两个运行期块级分支、drain 的 `if (n_it % 2)` —— `--compile` 的两体均 OK，且 5 个数值用例 + 5 个逐位用例全过 ✓。
- 去掉了 `T.annotate_buffer_versions` / `num_stages`（§4.8 的 factory 只留 `n_versions` 作为 A/B 接口）——与 §6.3 A 的证据链（Z3 `unsat` 静默退化、版本索引绑定当前迭代、RAW 与版本无关、13 分支 ±3%）一致 ✓；`T.Pipelined` 与显式两版本混用被明确排除且给了理由（调度器会按当前迭代下标重写它管理的缓冲）✓。
- 其余 API 结论（L0 `trans_A`、`dual_copy`、`set_atomic`、`SimtVF`、`T.copy(UB,L1,transpose)` 不可用等）与 v3 相同，未变。

### 2. 内存层级规划 — pass

- 三套预算逐项人工复核通过（见机械复核 §2）：UB **130,048 B（51.2%，比 v3 少 36,864）**、L1 **311,296 B（59.4%，与 v3 同、来源改为显式两份）**、L0C/L0A/L0B **逐字节不变**（221,184 / 49,152 / 61,440）✓。
- **版本判定表（§4.3）是本轮最好的结构**：「哪块缓冲要两份」由**生存期**决定而非旋钮决定，逐行给「生产→消费 / 跨迭代? / 版本数 / 依据」。我复核四条：① `qT/doT` 被 `GB(k)` 在**下一迭代**读 ⇒ 必须 2 ✓；② `pT/dsT` 同 ⇒ 2 ✓；③ `st_u/dp_u/p16_u/dsT_u` 同迭代 ⇒ 1（**功能上成立**，但见建议 2：这是性能上的唯一单版本风险点）；④ `st_c/dpt_c/dq_c` 同迭代 ⇒ 1，因此 A5 **不需要** L0C 双版本 ✓（这是对 `perf_feedback.md` 处方「把 dP 并入 st_c 以腾 L0C」的正当否决）。
- 搬运路径（§4.4）与 v3 逐字相同（L0C/UB 侧无新增搬运），GM 字节数 189.8 GB 逐字节不变 ✓。

### 3. Tiling 策略 — pass

- tile 96/64、对齐、分核三要素、非整除处理（4-D 按 request clamp）全部沿用 v3 的已闭合结论 ✓；A5 不改 tiling、不改 grid（22,016 < 65,535）✓。
- `Nk=80` 回退与 `V=3` 旋钮（L1 393,216 = 75.0%）都有容量复算与阈值 ✓（我复算一致）。

### 4. 技术约束检测 — pass

一维 grid / 静态 extent / L0C 240 KiB 红线 / L0B 4 KiB 余量（`2×(24,576+384·Nk) ≤ 65,536` 仅 `Nk ≤ 21`，故 L0 单版本是容量硬结论，A5 也不需要双版本）/ 原子 dtype / 无手工跨核 flag —— 逐条有方案（§3.5.1）。新增「`AscendInsertOOBPadding` 的 clamp 维度 = buffer 形状」仍列为硬约束 ✓。

### 5. 循环与同步 — pass（1 条建议）

- 循环结构：`T.serial(n_half)` + 2-tile 静态展开，`T.Pipelined` 从设计中删除；§6.3 B 的机制论证（**RAW 边只能靠改语句顺序、版本只能放宽 WAR**）在机制层正确，且被「13 个 `num_stages`/版本分支全在 ±3%」的实测反证支持 ✓。
- §7.2 逐条列出 v4 新增的四条边：`EP(k) 写 pT[v]` ↔ `GB(k−1) 读 pT[1−v]`（**不同对象 ⇒ 无需同步**）、`EP(k+1) 写 pT[v]` ↔ `GB(k−1) 读 pT[v]`（WAR，距离 1，需 AIC→AIV 事件）、`F(k+2) 写 qT[v]` ↔ `GB(k) 读 qT[v]`（WAR，AIC 内）、`fixpipe(k+1)` ↔ `EP(k)`（`st_u/dp_u` WAR，唯一单版本风险点，风险 16 + P12）——**这四条正是 A5 的全部同步语义，没有遗漏** ✓。
- 建议（非阻塞）：P11 的判据不具区分力（varlen v5 检视已证），见建议 1。

### 6. 验证方案 — pass（1 条建议）

- golden / 容差 / L0-1…L0-11 不变；新增 **L0-12（n_it=2，drain 偶）/ L0-13（n_it=1，drain 奇）/ L0-14（B=2/S=8192/H=8–16，重复 3 次，与 v3 逐位）/ L0-15（同形非因果）** ✓ —— 覆盖了 varlen 首版踩过的「drain 偶数分支」坑。
- 新增 **P9（删 drain 必须失败）/ P10（GB 读当前版本必须失败）** 两个反例自检 —— 这类「检查器要有判别力」的做法正是本任务族上一轮的教训，**保留得好** ✓。
- 新增 P11（结构判据）/ P12（`st_u/dp_u` WAR 用 msprof AIC 空闲窗口）✓。
- 门槛口径「逐 request、逐次 + 重复 3 次」✓。
- 建议（非阻塞）：P11 需替换为偏移身份判据（建议 1）；性能验收表的区间/通过线需统一（建议 4）。

### 7. 完整性与一致性 — warn

- §0–§14 齐全、无占位符、§3.5.3/§3.5.4 引用真实存在 ✓；v3 的 6 处口径矛盾保持修复状态 ✓。
- 不一致项（建议 3/4，均不改变方案）：
  1. **§9.1 风险 13 仍是 v3 的表述**（「80% MFU … 但**原理上可及**（完全重叠下界 1.24，改进后 `0.80+MTE1`）」，处置栏还写着「档 B ≤62 ms / cube ≥60%」与「6 条条件含 `MTE1 ≤0.20`」）——与 §1.6.4 F 的 ⚠️ 横幅、§8.3 末行、§13.1 #11 的「实测 0.60–0.68 ⇒ 任何流水下不可达」**直接矛盾**。
  2. **§1.6.0 的 v3 结论段**（「…正确表述见 §1.6.4 F（当前串行结构下不可达，跨迭代流水后原理上可及）」）同样的残留；该句后面的「v4 roofline 重算」表是正确口径。
  3. 投影区间三处不一致：§8.3「93–96 ms（§14.3 的 93–101）」、§14.3「98.1（区间 94.5–105）」、§14.4「≤115 硬 / ≤105 期望」；按实测 1.874 复算为 **98.3 ms**，区间应是 `[1.80, 2.00] ⇒ 94.6–104.8`，即 94.5–105 那一组，**「93–96」不成立**。
  4. 同表「80% Cube MFU（T_op ≈ 48 ms）」与 §1.6.4 F 的 50.6 ms 用的是 378 vs 358 TF/s 两个基口径；「`aic_cube_ratio` 通过线 ≥40%」又写「38–39% 属投影保守端，不视为设计错误」——通过线被自我软化，验收口径不唯一。
  5. §4.5 「块尾缓冲不加版本：`92,160 × 2` 会上界到 `130,048 + 24,576 = 154,624`」——`+24,576` 是 `st_u/dp_u` 的第二版，不是块尾（块尾双版本应为 222,208 B）；该句把两个旋钮混写。
  6. §4.6 的 MTE2 三元组自相矛盾：`82,432 B ÷ 0.720 µs = 114.5 GB/s`，文中写 118.5 GB/s（3.4% 差；要么 0.696 µs，要么 114.5 GB/s）。
  7. §1.6.2 的表行序（#12 排在 #16 之后）与编号不连续，纯排版。

### 8. 算法优化分析 — pass（1 条建议）

**结构与四问**：R1 表新增 **#11（A5，采纳）/#12（S2，否决）** 两行，均有「原式/优化后/等价性/收益」四要素；R2 补充「旋转不改扫描遍数、不改中间缓冲」✓；R3 增加 **C2″** 行（FLOPs/访存/缓冲逐字节不变，仅墙钟 2.965→1.870）✓；R4 增加 C2″ 行与 `UB −36,864` ✓；D-2 有 **v4 重算表**（用 Stage 4 实测管道口径替换 v3 外推）✓。

**独立复算（抽查 10 项）**：

| 量 | 设计值 | 我的复算 | 判定 |
|---|---|---|---|
| A5 墙钟 | 1.870 µs（B=2/S=8192/H=8） | 我复跑 **1.874**；base 2.965（与生产 2.944 差 +0.7%） | ✓（−36.8%） |
| 投影 `_bwd_main` | 94.8 ms | `50,715 × 1.874/1000 = 95.0` | ✓ |
| `T_op` | 98.1（区间 94.5–105） | `95.0 + 3.3 = 98.3` | ✓ |
| cube 占比 | 38.8%（38.0 ms ÷ T_op） | `38.0/98.3 = 38.7%`（38.0 ms = 25.1% × 152.6 ms 实测忙时） | ✓ |
| AIV 占比 | ≈70% | `1.304/1.874 = 69.6%` | ✓ |
| 相对 v3 | −36.5%（T_op） | `154.4 → 98.3` = −36.3%（主因迭代数不变、单价 −36.8%、launch 3.3 恒定） | ✓ |
| S2 的 L0C | 270,336 B = 103.1% | `221,184 + 2×24,576 = 270,336/262,144 = 103.13%` | ✓ |
| `Nk=80`/`V=3` 容量 | L0B 55,296（84.4%）/ L1 393,216（75.0%） | 逐项一致 | ✓ |
| 首末迭代损失 | 22,016×2×1.0 µs ÷28 ≈ 1.6 ms ≈ 1.7% | `1.57 ms / 95 = 1.65%` | ✓ |
| `mte1+mad` 主链 | 1.449 µs ⇒ 73.2 ms 下限 | `(0.680+0.769) × 1.415e6/28 = 73.2` | ✓ |

**与 varlen 的比值自洽性**：varlen 旋转后 2.092 ≈ 其 AIV 1.95–2.10（由 AIV 界定）；dense 旋转后 1.874 vs 其 AIV 1.304（**不由 AIV 界定**，由 `mte1+mad` 链 + fixpipe 暴露界定）——两族的「谁界定」不同但各自的数字自洽，§14.2 的差异表把这一点列清楚了 ✓。dense 收益更大（−36.8% vs −33.9%/−26.5%）与「dense 的填充/装载形态不同」一致，且有本算子探针独立取证 ✓。

**A5 的等价性（我的独立推演）**：见下节表格，结论成立：版本取址、drain 奇偶、`clear_accum`、`dq` 行区间、累加顺序逐条正确；「`GB(k−1)` 与 `EP(k)` 作用在**不同缓冲对象**」这一点在 §3.3 的实参上成立（半体 A 传 `pT_s1`，本半体 `EP` 写 `pT_s0`），故该 WAR 边**在构造上不存在** —— 这是 A5 收益的机制层解释，我认可。

**否决 S2 的依据**：L0C 103.1% 越界 + 「A5 已把墙钟压到 `mte1+mad+fixpipe` 的 2.019 之下」⇒ 收益 ≈0 —— 算术与方向都成立；且与 `perf_feedback.md`「dP 并入 st_c」建议的适用条件（S2）区分清楚 ✓。**这是本轮最好的「否决也有量化」示例。**

**结论**：维度 8 pass。唯一的非阻塞意见是「残差 0.42 µs 的归属」（建议 2）——设计把它写成「未分解」，我认为它可由它自己的两个数字解释，并给出可检验的预测。

## A5 机制的逐项独立复核（本轮核心）

| # | 检查项 | 我的独立核对（不采信文档自述） | 结论 |
|---|---|---|---|
| 1 | **版本配对** | 定义 `u(k)=(k−loop_st)%2`；半体 A（`kqA=loop_st+2i`）⇒ `u=0`（`_s0`），半体 B ⇒ `u=1`（`_s1`）；`GB(k)` 读 `1−u(k)`。我按 `n_it=1,2,3,4` 手工推演，并对照 `verify_equiv` 的 478 点 / 17,846 事件枚举 | ✓ |
| 2 | **`GB(k−1)` 与 `EP(k)` 是否不同对象（WAR 边不存在）** | 半体 A：`EP(kqA)` 写 `pT_s0/dsT_s0`，`GB(kqA−1)` 读 `pT_s1/dsT_s1` ⇒ **不同 L1 对象**（`_s0/_s1` 是两个 `T.alloc_l1`）；半体 B 对称。→ 该 WAR **构造上不存在** ✓（这正是「AIC 不必等 AIV」的来源） | ✓ |
| 3 | **残留 WAR（距离 1）是否被正确插入** | `GB(k−1)` 读 `pT[v]` → `EP(k+1)` 写 `pT[v]`（AIV 侧）、`GB(k−1)` 读 `qT[v]` → `F(k+1)` 写 `qT[v]`（AIC 侧）：程序顺序保证队列次序，跨单元那条需 InsertSync 按缓冲身份插事件；设计在 §7.2 明确列出并承认「缺失会表现为随机坏块」；证据 = parity 5/5 逐位 + 复跑 2 次一致 | ✓（**证据强度见建议 1**） |
| 4 | **首迭代** | `if kqA > loop_st` 在 `i=0` 为假 ⇒ 无 `GB` ✓；`n_it=1` 时半体 B 整块被 `if kqB < loop_ed` 跳过 ✓（L0-13 覆盖） | ✓ |
| 5 | **末迭代 / drain** | `last=loop_ed−1`；`u(last)=(n_it−1)%2` ⇒ 奇 → `_s0`、偶 → `_s1`，与 §3.3/§14.2 的 `if (n_it%2)==1 else` 一致 ✓（我按 1/2/3/4 逐个验证）；`no_drain` 与 `drain_parity` 两个反例被 `check_a5` 抓到 ✓；P9 要求实现期再验一次 ✓ | ✓ |
| 6 | **`clear_accum=(kq−1==loop_st)`** | 首个 `GB` 恒为 `GB(loop_st)`：`n_it≥2` 时在 `i=0` 的半体 B（`kqB=loop_st+1 ⇒ kqB−1=loop_st` ✓）；`n_it=1` 时在 drain（`last==loop_st` ✓）。恰一次、`dk_c/dv_c` 正确初始化；`loop_ed ≥ loop_st+1` 保证非空 ✓ | ✓ |
| 7 | **`dq` 偏移与原子** | 半体 A `(kqA−1)*BM`、半体 B `(kqB−1)*BM`、drain `last*BM` ✓；写回仍是 4-D `dq_acc[b, m0:m0+64, h, :]` + `set_atomic`/`none` 紧包一次 `T.copy` ✓；每个 k 恰一次原子加（顺序跨块不确定，与源一致） | ✓ |
| 8 | **`dV/dK` 累加顺序** | `GB` 的调用次序按 k 递增（半体 A 的 `k−1` < 半体 B 的 `k`），故 `dk_c/dv_c` 仍按 query 块升序累加，与源算子契约一致 ✓；这也是 `dk/dv` 能**逐位**对齐 v3 的原因 | ✓ |
| 9 | **dense 专属差异** | `loop_st/loop_ed/n_it/n_half` 全在 dense 静态 S 上重算（无 `cu_seqlens`/slab）✓；`b/h/n0` 由 `bid` 在循环外一次解出、`GB` 内不重算 ✓（§3.3 注 4）；块尾 `dk/dv` 写回在循环外、4-D clamp 不变 ✓；无 head slab、单 launch ✓ | ✓ |
| 10 | **`st_u/dp_u` 单版本** | **功能上**成立（GA 写 / EP 读，同半体；跨半体只有 `fixpipe(k+1)` ↔ `EP(k)` 的 WAR）；**性能上**这是唯一把 AIC 与 AIV 重新耦合起来的边（设计自己在 §7.2/风险 16/P12 承认） | ✓（见建议 2） |
| 11 | **与 varlen 复用是否漏项** | §14.2 的 11 行差异表逐行核对：可复用（机制/2-tile/首末迭代/`clear_accum`）与必须重算（`loop_st`、`dq` 行偏移、块尾、grid 解码、无 slab、AIV/MTE2 归因、收益量级）**没有遗漏**；我另查：`lse/delta` 的 2 版本、`kT` 最后重灌、`sid` 半区偏移三处与 varlen 一致且 dense 同样需要 ✓ | ✓ |
| 12 | **4-D / B1 / grid 未回归** | 4-D 签名、per-request clamp、`dq` 4-D 原子、grid 22,016 全部与 v3 逐字相同 ✓（探针的 4-D B=2 用例 5/5 逐位通过） | ✓ |

## Stage 3 必做验证

1. **A5 偏移身份判据（取代 P11；P11 降级为辅助判据，见建议 1）**：反查 `device_kernel.asc` 的 12 个每迭代 L1 缓冲基址，断言
   ①半体 A 的 `GB` 三组 `L1→L0` 源偏移落在**版本 B** 基址、同半体 `F/EP` 落在**版本 A**（半体 B 相反）；
   ②`dq` 的 `asc_copy_l0c2gm` 行区间用 `(kq−1)*64`；③drain 的偏移按 `n_it` 奇偶落到 `_s0/_s1`；
   ④`GB` 的 3 个 `asc_mmad` 与 3 组 copy 之前**不得有以 `PIPE_V` 为前驱的 `asc_sync_wait`（保留为辅助，不作唯一判据）**；
   ⑤`asc_mmad=16`、`__simt_vf__=5`、生成源 ≈50–52 KB。**任一不符即回滚 v3 体**（风险 18 的判据改成这五条）。
2. **首/末迭代与 drain（P9 / P10）**：L0-13（`n_it=1`，drain 奇）、L0-12（`n_it=2`，drain 偶）与 golden 一致；再做两个**反例自检**——删 drain 必须失败、半体 A 的 `GB` 改读 `_s0` 必须失败（否则说明版本未生效）。
3. **4-D B=2 生产几何**：L0-14/L0-15（B=2/S=8192/H=8…16，causal/非因果）各**重复 3 次**；判据 = 与 golden 在 §8.2 容差内 **且** `dk/dv` 与 v3 冻结实现 **逐位相同**（`torch.equal`），`dq` 相对差 < 1e-3（fp32 原子到达序）。
4. **代码体积 / icache**：记录生成源字节与 `asc_mmad/__simt_vf__` 计数；若 `T_op` 显著差于 95 ms，先做「同 kernel 连跑两遍取第二次」的 icache 暖机判据，再按风险 14 的 ①②③ 缓解。
5. **同步与 `st_u/dp_u` WAR**：msprof 采 `AIC 空闲窗口`（P12）；>10% ⇒ 启用 `st_u/dp_u` 第二版（UB 154,624 B = 60.9%）。
6. **管道与验收**：`aic_cube_ratio`、`aiv_vec_ratio`（警戒 80%）、`aic_mte2_active_bw`（<85 GB/s 触发排查）、`aic_mte1_active_bw`（>0.75 µs 回看 V 旋钮）；**通过线以 `T_op ≤115 ms` 为唯一硬线**（cube ≥40% 作为期望值见建议 4）。
7. **逐位一致性的持久化**：把「`dk/dv` 与 v3 逐位」写进 L1 的门槛（不只是 L0-14），并在每次修订后复跑（A5 取错版本会成片出错，逐位比对是最便宜的探测器）。

## 非阻塞建议（6 条，供 Stage 3 落地 / Stage 4 排序）

1. **P11 换成「偏移身份」判据，并降级为辅助**（依据：varlen v5 的 **REVIEW** 已证明「`GB` 前无 `PIPE_V` 前驱 wait」在**未旋转**的 v4 生产体上同样成立 ⇒ 无区分力、会给假阳性）。dense 与 varlen 逐语句同构，本条**同样适用**。可判定的替代：见 Stage 3 验证第 1 项的 ①②③⑤（我在 varlen 的检视里已见过这套判据在设备码上被验证可行）。预期：P11 由「必跑判据」改为「辅助」，风险 18 的回滚条件改为「偏移身份不符 或 `T_op > 135 ms`」。
2. **残差 0.42 µs 的归属可再收紧，并把 `st_u/dp_u` 第二版提为 Stage 4 第一项**。按设计自己的两个数字：AIV 忙时 1.304（44.3% × 2.944）+ fixpipe 0.570 = **1.874 ≈ 实测 1.870–1.874（差 ≤0.2%）**；而 §6.3 F 归给「块级固定开销」的部分我复算只有 ≈0.03 µs/迭代（`98,304 B ÷ 118.5 GB/s ÷ 64.5 + 块尾 ≈1 µs ÷ 64.5`，占 1.6%）。故「未覆盖的 fixpipe」几乎就是全部残差，且正是 `st_u/dp_u` 单版本 WAR 的预测形态。**建议**：把风险 17 从「未分解」改为「已定位：`EP + fixpipe` 耦合」，A/B 顺序改为 ①`st_u/dp_u` 第二版（+24,576 B ⇒ 60.9%）②`V=3` ③epilogue 阶梯；**预期** 1.45–1.60 µs ⇒ `T_op` ≈ 77–84 ms、cube ≈ 45–49%（顺带把「cube ≥40%」从「不承诺」变成大概率达标）。若实测仍停在 1.87，则说明残差另有来源，再回 msprof 分解——这条建议自带证伪条件。
3. **清掉两处 v3 残留的「80% MFU 原理上可及」**：§9.1 风险 13 与 §1.6.0 的 v3 结论段（第 548–551 行一带）。v4 的正确口径是「`mte1+mad = 1.449 µs > 1.00` ⇒ **任何流水下都不可达**（§1.6.4 F 横幅、§8.3 末行、§13.1 #11 已如此写）」。改法：风险 13 的「原理上可及」→「不可达（实测 `MTE1 0.60–0.68` 否证了 `≤0.20` 的前提）」，处置栏删掉「档 B ≤62 ms / cube ≥60%」与「6 条条件」。
4. **统一投影区间与验收线**：①`T_op` 区间统一为 **94.5–105 ms（central 98）**（§8.3 的「93–96」按实测 1.874 不成立）；②「80% MFU ≈ 48 ms」改成与 §1.6.4 F 同基的 **50.6 ms（358 TF/s）** 或注明用 378 TF/s；③`aic_cube_ratio` 的**通过线**只保留一条（建议 `≥38%`，把 `≥40%` 写成目标）；④§4.6 的 MTE2 三元组二选一（`0.696 µs` 或 `114.5 GB/s`）。
5. **修 §4.3 版本判定表里 `lse/delta` 的理由**：`GB` **不读** `lse/delta`（它只读 `pT/dsT/doT/qT/kT`），故「下一迭代读」不是它们要两份的原因；真实原因是「`EP(k)` 在 AIV 读 `lse_u[v]` ↔ `F(k+1)` 在 MTE2 写 `lse_u[v]`」的 WAR——保留两份的决定不变（varlen 同），只把依据写对。
6. **保持 UB 预算可机检**：§4.5 加「版本」列后 `design_calc_check.py` 的 `ub_budget` 由 pass 变 **skip**。建议在同一节补一行工具能解析的合计（如 `| 合计 | — | — | — | 130048 B |`），或接受人工复核并在 §8.3 注明「UB 预算为人工复核项」。同时修 §4.5「块尾缓冲不加版本 ⇒ +24,576 = 154,624」这句（`+24,576` 是 `st_u/dp_u` 的第二版，不是块尾）。
