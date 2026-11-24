# _bwd_kernel 设计文档检视报告（v2：验收判据修订）

## 检视结论

结论: 通过

## 检视元信息

- 文档: `examples/mla_varlen_bwd/_bwd_kernel/DESIGN.md`（**v2**，2026-11-24，2,222 行；v1 = `history_version/design_v1_gridfix.md` 2,046 行，v0 = `history_version/design_v0_pre_gridfix.md`）
- 任务类型: **迁移**（harness 子模式），9 维度检视（含维度 0 与维度 8）
- 源算子: `/workspace/TileOPs/src/tileops/_kernels/attention/mla/bwd.py`（`_bwd_kernel`，`bwd.py:109-264`；packed 分支 `136-139 / 176-184 / 218-224`），**本轮重读全文 387 行**
- 本次检视对象: v2 = 对 v1 检视结论（**不通过，阻塞 1 = `dq` 判据不可达**）的修订。**逐项核实**声明 1–7（B1 闭合、`block_n` 同源、smoke 几何、L0 编号与 L0-11b、尾 slab 表述、L2/百分比、grid 修复零改动）
- 参照实现（未改动，md5 已核）: `_bwd_kernel.py` = `d3cf18dd5ae1f13d36c877b8a3b9fb8b`（mtime 2026-11-23_17:32）、`launch_plan.py` = `94471f4b57739c20e362bae8c3768582`（2026-11-24_05:38）、`verify_equiv.py` = `d7ffe16b8fa0a1182748b705c7664833`、`history_version/design_probe_grid_slab.py` = `c3d28860b23677c6623cd70719ad7dec`
- 平台: `ascend-950pr-9579` / `ascend`（dav-3510，CANN 9.2.1 / TileLang ascend 0.1.15）；`stage_skills.review = tilelang-ascend`
- 阻塞级问题数: **0**
- 建议级问题数: 4（均在对应维度内以「建议」标出，另在「留待 Stage 3」一节复述 3 条实现侧必做项）

## 机械复核（D-1/D-5）

### M1. verify_equiv 重跑（本轮重新执行，不是引用 v1 结果）

`cd examples/mla_varlen_bwd/_bwd_kernel && TMPDIR=/workspace ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python verify_equiv.py` → **exit 0**：

| 采纳项 | 样本数 | 候选 vs 基线 最大 ulp | 违反率 | 候选 vs fp64 最大 ulp | 基线 vs fp64 最大 ulp | 结论 |
|---|---|---|---|---|---|---|
| A1 exp 域 P（float16） | 1048592 | 1 | 0.000e+00 | 1 | 1 | EQUIV_PASS |
| A1 exp 域 P（bfloat16） | 1048592 | 1 | 0.000e+00 | 1 | 1 | EQUIV_PASS |
| A2 块级 mask 特化 · packed 运行时界（64/64） | 115696896 | 0 | 0.000e+00 | — | — | EQUIV_PASS |
| A3 packed 溢出零行不变式 | 38446 | 0 | 0.000e+00 | — | — | EQUIV_PASS |
| A4 head-slab 分解（任务集一一对应） | 568908 | 0 | 0.000e+00 | — | — | EQUIV_PASS |

附带：`A2 fast-path blocks: 13734/28261`；`A3 detail: spill_entries=30528 dq_rows=3816 acc=4102 zero_sign_flips=1`；`A4 detail: tasks=568904 over_cap=0 edge_checks=2/2 raised_on_overflow=2/2`；`EQUIV_SUMMARY ALL_PASS`。**与 DESIGN §1.6.1 内嵌表逐字段一致**，且 A4 明细的 smoke 行为 `slab=8 launches=1 max_grid=96`（= v2 统一后的 smoke 几何，见维度 3）。

判据构造复核（不只重跑）：A4 仍为「逐 launch 逐 bid 穷举 → 与 `{(t,h)}` 求差（缺失/重复/越界三项） + head 区间两两交叠 + 上界 65,535 / 越界 65,536 两侧 + auto-shrink 必须缩小而非 raise」，A2 仍有快路的 `assert vis.all()` 充分条件断言，A3 仍是结构性推导（越界列 `dS==0` 且 `ds16·k` fp32 累加为精确 0，并对 `-0.0`/次正规/`3.4e38` 验 `+0.0` 原子加）。脚本本身未改（md5 同上），判据未被 v2 触碰。

### M2. design_lint（0 fail / 0 warn）

`python3 .agents/tools/statectl.py gate 1 --task-dir examples/mla_varlen_bwd/_bwd_kernel --dir examples/mla_varlen_bwd/_bwd_kernel` →

```json
{"action": "gate", "failures": [], "ok": true, "pass": true, "stage": 1, "warnings": []}
```

→ **failures 0 / warnings 0**（该 group 覆盖 §-schema、调研四问+基线+结论字面量、core-split 三要素、占位符扫描、引用路径存在性）。与 §12.6「v2 的验证」声称的 `{"pass": true, "failures": [], "warnings": []}` 一致。

### M3. design_calc_check（`python3 .agents/tools/design_calc_check.py --design … --task-dir …`）

```json
"checks": [
  {"check": "ub_budget",  "status": "pass", "detail": "UB: 11 buffer entries sum = 88.5 KB; capacity = 248.0 KB"},
  {"check": "l0c_budget", "status": "pass", "detail": "block 64x64 x fp32 accum = 16.0 KB vs L0C 256 KB"},
  {"check": "core_split", "status": "skip", "detail": "insufficient parsed dims {'bm': 64, 'bn': 64}; declared logical cores = None"},
  {"check": "r3_metrics", "status": "pass", "detail": "all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak)"}
]
```

`skip` 项**人工已核**：§5.5 ① 逻辑核数逐格复算 —— `ds-v3-8x4k` 128×(512+8)=66,560；`ds-v3-mixed` 128×(504+6)=65,280；`kimi-k2-4x8k` 64×(512+4)=33,024；**smoke 128×(4+6)→8×12=96**（v2 已与 §5.6 c/A4/实现统一）；`H=126` 65,520；`H=127` 66,040。② 物理核数依据可实查（本轮 `statectl platform` 仍返回 `ai_core_cnt=28 / vector_core_cnt=56`）。③ 规模判定与分核方案（含对标准「极大规模」分支的显式偏离声明 + `96 ≈ 3.4 波`的判断）齐全。L0C 的真实约束是 §4.6 的 5 个累加器合计 163,840 B（62.5%）、L1 172,032 B（32.8%）、UB 90,624 B（35.7%）——三项与 v1 **逐字节相同**（§4.5/§4.6 区段 vs v1 比对 identical=True），人工已加总。

### M4. `launch_plan.py` 与 v2 新增/变更几何的复算（我实跑，逐格比对 DESIGN 表格）

| v2 位置 | DESIGN 声称 | 实跑 `plan_launch` | 一致 |
|---|---|---|---|
| §5.5 smoke / §5.6 c / L0-1-2 | tiles=12 slab=8 launches=1 grid=96 | tiles=12 slab=8 launches=1 max_grid=96 idle=0 | ✓ |
| §8.3 L0-3 | tiles=11 slab=4 grid=44 | 11 / 4 / 1 / 44 | ✓ |
| **§8.3 L0-11b（新增）** | tiles=17 slab=32 grid=544、2 次 launch、idle 0 | 17 / 32 / **launches=2** / 544 / idle=0 | ✓ |
| §8.3 L0-17 | H=32 ⇒ slab=32、1 次 launch、grid=16,640 | 520 / 32 / 1 / 16,640 | ✓ |
| §8.3 L0-18 | H=16 ⇒ slab=16、grid=8,256 | 516 / 16 / 1 / 8,256 | ✓ |
| §8.3 L0-19 | H=16 ⇒ slab=16、grid=8,160 | 510 / 16 / 1 / 8,160 | ✓ |
| §8.3 L0-20 / G5 ①–④ | 越界 raise / 上界 65,535 / 缩小到 slab=31 max_grid=63,736 | 两个越界均 `ValueError`；上界 `slab=1 max_grid=65535`；`slab=31 max_grid=63736` | ✓ |
| §8.3 L0-21 | slab=126 → grid 65,520；slab=32 → 4 次 launch、有效块相同 65,520 | 1 launch/65,520；4 launches/16,640、useful=65,520 | ✓ |
| §8.3 L0-22 / G3 | H=126 idle 1,040；H=127 idle 520；slab=63 → 2 次 launch | 1,040 / 520；63 → 2 launches grid 32,760 | ✓ |
| §8.4 G1 ② | B=1 单 request → tiles=65 slab=32 launches=4 max_grid=2,080 | 65 / 32 / 4 / 2,080 | ✓ |
| **§5.6 g ③ / G5 ⑥ 反例** | `block_n=32` + 131,072 token → `4,104×31 = 127,224 > 65,535` | 计划侧 `slab=31 max_grid=63,736`；按 block_n=32 重算真实 grid = **127,224** | ✓（反例成立） |

**grid 修复零改动的核对（声明 7）**：把 v1 与 v2 的同名区段逐字节比对 —— §3.5.4 的 P6 设备码证据表（3,673 B）、§5.6 d 写集不交论证（1,114 B）、§5.6 f 备选裁决（906 B）、§8.4 G4（309 B）**全部 identical=True**；§1.6.1 的 A4 行（1,192 → 1,214 B）**唯一差异是末尾代价措辞**（「只算 2 个标量」→「1 次 4 B `h_off` 读 + 2 个标量运算」），其 (i)–(vi) 等价性论证、双射主张、收益数字（16,640 / 25.4% / 余量 48,895 · 49,215）**逐字未变**；`slab = min(32, heads, 65535//tiles)`、`h_off`、守卫位置、`assert_launch_plan` 三检查、P6 全部原样。实现侧 `_bwd_kernel.py` / `launch_plan.py` / `verify_equiv.py` 的 md5 与 mtime 与 v1 时点一致 → **声明 7 成立**。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

v2 未改 §0 的实质内容（唯一变动是 R8 的尾 slab 措辞更正），故本轮按「迁移结论是否仍与源码一致 + 更正是否成立」复核；结论与 v1 相同，且更正**成立**。

| 检查项 | 结论 | 源码证据 ↔ DESIGN（本轮重读 `bwd.py` 387 行） |
|---|---|---|
| 语义理解 | pass | `bwd.py:226-228`（两个 `transpose_B=True` GEMM 给出 `s^T`、`dP^T`）、`229-243`（`exp2(st·scale·log2e − lse·log2e)` + `vis=(m0+j<len_q)&(n0+i<len_kv)&(n0+i<=m0+j+shift)`）、`245-251`（`dv_f+=gemm(p16,do_s)`、`ds16=st·(dpt−delta)`、`dk_f+=gemm(ds16,q_s)`、`dq_f=gemm(ds_s,k_s,transpose_A=True)`）↔ §0.1/§1.3；bottom-right 对齐、fp32 累加、`dk*scale`/`dv` 不乘/`dq` 只写 fp32（`254,257,260`）↔ §0.1 dtype 行 |
| I/O 契约 | pass | `bwd.py:141-154` 的 11 个形参顺序/形状/dtype ↔ §0.2 逐行；`stat_shape=(heads,total_q)`（`137`）↔ head-major；`dq_acc` fp32 in-out（`151`）；`o` 只被 `_delta_kernel` 消费（`341`）↔ §0.2 注 |
| 算法步骤覆盖 | pass | §0.3 S1–S18 覆盖 `bwd.py:176-260` 全部计算语句（含 packed 三件事：`cumsum_offsets` `181`、`live`/`decode` `182-184`、head-major `lse[h,q_start+m0+i]` `219-224`），无遗漏/无臆造 |
| 优化手段识别 | pass | §0.4 的 15 项逐条对得上源（key-stationary `200-201,245-248`、recompute、因果剪枝 `202`、tensor core、fragment 累加、`transpose_A` `251`、`exp2+LOG2E` `123,235`、`if_then_else` 掩码、`num_stages=2` `204,336`、atomic `254`、行统计、`FullRow`+`threads=256`、`GroupTiling` `138-139,181-184`、块表每 CTA 重算〔`grouped_tiling.py:99-104,166-200`：`serial_groups=8` 串行分支 / 否则 `T.shfl_up`×`WARP_SHUFFLE_STAGES` + `get_thread_extents()` + `sync_threads`〕）；#13 明确标注为 NPU 侧构造 |
| 耦合性判定 | pass | §0.5 的 23 行抽检：一维网格（`ascend/language/kernel.py:4,146` 明文）、NT-only（`ascend/op/gemm/gemm_mad.py:162` 断言原文）、L0C fractal（`gemm_mad.py:149`）、warp 扫描不可用（kernel 作用域无线程域）、65,535 行处置为 R8 |
| NPU 重设计 | pass | R1–R8 各有「源方案 → NPU 算法 → 语义保持论证」；**v2 只改 R8 的空块代价措辞**（「只算 2 个标量」→「1 次 4 B `h_off` GM 读 + 2 个标量运算 + 若干块级比较」），与 P6 设备码一致（见 M4 与维度 1）——v1 的「无 GM 读」确为笔误，v2 在 §0.6 / §5.6 b,c,e / §1.6.1 A4 / §1.6.2 #16 / §9.1 风险 15 七处**统一**，我 grep 确认「不读任何 GM」「只算 2 个标量」**已无残留在非更正语境** |
| 设计一致性 | pass | §1.4 / §3.1 / §3.3 / §6.1 描述的就是 slab 化 + 融合 decode + 双分支写回的 R1/R6/R7/R8；§12.6 的「v2 未改动项」清单我抽 3 项（slab 公式、写集论证、P6）逐字节比对一致 |
| golden 独立性 | pass | §8.1 只读 `q,k,v,do,cu_seqlens_*`，`lse`/`delta` 自算（`logsumexp` / `p*(dp-(p*dp).sum())`），按 request 物化 `P` 后 `einsum`，不复刻分块/半区/原子/双分支；`slab_heads=4` 仅为参照内存切分。v2 新增的 L0-15 用「按 request 逐对朴素循环」再互验 golden 自身（`_bwd_kernel.py:810-845` 已实现同形态），**独立性进一步加强** |

### 1. API 可行性 — pass

v1 已核的证据本轮未变（`T.Kernel` 一维/无 `threads`：`ascend/language/kernel.py:4,146`；`T.gemm` 仅 NT：`gemm_mad.py:162`；符号 extent 抛错：`gemm_op.py:62`；`dual_copy` 语义：`copy_op.py:53-83`）。v1 新增的两项（`h_off` 运行期标量、`bid // slab` 静态除模）在 v1 检视时我已独立 dump P6 设备码确认 `int32_t h = (h_off[0] + (((int32_t)block_idx) & 1))`（AIC/AIV 各一份）、`block_idx >> 1`、`if (h < 3)` 包住 body；v2 未动 §3.5.4 P6 证据表（区段 identical=True），故该结论继续成立。v2 新增的「`meta["head_slab"]/meta["block_n"]` 由工厂回传」是 **wrapper 层面的 Python 契约**（不是 TileLang API），不需另找 API 佐证。

### 2. 内存层级规划 — pass

v2 未改缓冲清单：机械位 `ub_budget` pass（88.5 KB / 248 KB），L1 172,032 B（32.8%）、L0C 163,840 B（62.5%）人工加总与 §4.5/§4.6 逐字节一致（并与 v1 identical=True）。`head_slab` 不改任何 buffer 形状（§5.2 末条）。

### 3. Tiling 策略 — pass（v1 建议 3 已闭合）

- `block_n=block_m=64`、`_HN=32`、对齐与非整除处置齐备；分核三要素齐备，物理核数依据可实查。
- **smoke 几何已统一（v1 建议 3 闭合）**：§5.5 表改为 `B=2 / lens 256,384（T=640）/ H=8 → Σceil=4+6=10 / 上界多算 2 / 逻辑核数 96 / slab=8 / grid=96 / 1 次 launch`，§5.5 ③ 改为「最小一条 smoke 96（≈3.4 波）」并单列「更小的 L0-3 门槛用例 44（≈1.6 波）」；§8.3 L0-1/L0-2 同步为 `[256,384]/H=8`，L0-3 保留 `[512,96]/H=4 → tiles=11/slab=4/grid=44`。四处（§5.5、§5.6 c、§8.3 L0-1、A4 明细、实现 `_bwd_kernel.py:914`）**同一几何**，我实跑 `plan_launch` 得到 12/8/1/96 与之一致。
- **建议 4（carry-over，非阻塞）**：`room < HEAD_SLAB_DEFAULT` 时 `slab` 仍取 `room` 而非 `heads` 的约数。auto-shrink 几何（tiles=2,056）取 `slab=31` → 末 launch 27/31 空块，`idle=55,512`（17.4% of 318,680）；若取 `heads=128` 的约数 **16** 则 `grid=32,896`、8 次 launch、`idle=0`、总块数少 17.4%（代价是 3 次额外 launch，µs 级）。**manifest 三个大 case 的 idle 均为 0，故不影响本期任何 workload**；建议在 §5.6 b 的「三个上界取 min」后补一句「当 `room < 32` 时优先取 `heads` 的约数（≥16）」，或登记为 Stage 4 调优项。

### 4. 技术约束检测 — pass

§3.5.1 的「单次 launch 块数 ≤ 65,535」行、§9.2 的反例（`grid = tiles × heads` 一次发射、按 slab 编译多份、slab 之间再清零）在 v2 中新增两条（`block_n` 不同源、给 `dq` 设固定绝对阈值），后者正是 v1 阻塞项的固化反例，价值高。L1-GEMM 静态 extent、L0C 唯一可累加层、L0C→UB 同 dtype、`dual_copy` 约束等行与源码一致（维度 0/1）。

### 5. 循环与同步 — pass

v2 未改 §6/§7 结构；§7.2 的「跨 head slab 的 launch 之间无需同步」仍以「写集按 head 不交 + 无跨 head 归约 + stream 序」为依据，与 §5.6 d（v1/v2 identical）一致。v1 建议的 L0-11b 补上了该论证的运行期覆盖（见维度 6）。

### 6. 验证方案 — pass（v1 阻塞项 B1 已闭合）

**B1 是否真的闭合 —— 逐条核**：

1. **无残留操作阈值**：我 grep 全文 `1e-5`，命中的 10 处**全部**是「解释 v1 的错误」「§12.6 的修订对照」「H1 移交说明」，无一处是可执行判据；`dq`/`dq_acc` 的可执行判据只剩 §8.2 的 `atol + rtol·|ref|`。
2. **§8.2 新增唯一口径 ①–⑤，算术我独立复算全部正确**：`|dq_acc| = 76/0.07217 = 1.053e3`；`ulp_fp32(1.053e3) = 2^-23·2^10 = 1.2207e-4`；`atol 6.9e-2 = 565 ulp`；`rtol·|ref| = 1e-3×1.053e3 = 1.053 = 8,627 ulp`；合计 `1.122`；折 dq 单位 `1.122×0.07217 = 0.0810` ✓；`ulp_bf16(76)=0.5`（76∈[64,128)，8 位有效位 ⇒ `2^(6-7)`）、`ulp_bf16(256)=2.0`、`ulp_bf16(512)=4.0` ✓；bf16 相对 ulp ∈ `(2^-8,2^-7]`、`rtol 1.6e-2 = 2.048 × 2^-7` ✓ ⇒「单靠 rtol 项即 ≥2 ulp，任何幅值下都覆盖 1 ulp」**成立**（`ulp(x)/|x| ≤ 2^-7 < rtol`，小幅值由 atol 兜住）——这正是可达性的**结构性**证明，不依赖任何未测量常数。
3. **变量 ④ 的判别力**：若 bf16 差达 4 ulp（`|dq|≈76` 上的 2.0），则 `dq_acc` 层必有 `2.0/scale = 27.7`（≈2.3e5 ulp）的差 ⇒ 只能是真实缺陷。复算 ✓。这条把「容差是否太松以致放过真 bug」的疑虑关掉：bf16 行只放过 ≤2 ulp 的重量化，3 ulp 以上必然对应 fp32 层的巨量偏差。
4. **`k_ulp` 只是诊断**：§8.2⑤ 明写「不参与判定」「>8 打印告警（不是失败判据）」；L0-12/L0-21 亦写「`>8` 告警、不作判据」——与 v2 §12.6 的「降级为诊断量」一致，没有把诊断量偷偷变成门槛。
5. **§8.3 / §8.4 全部对齐**：L0-12、L0-21 判定列改为「`dk/dv` 逐位相同；`dq_acc` 按 §8.2 口径」，并明写「禁用固定绝对值」；G1/G2/G3 判定列改为「`dk/dv` 逐位相同；`dq_acc`/`dq` 按 §8.2 容差」，G1 的量级锚点（1.05e3 / 1.22e-4 / 565 ulp）与 §8.2③ 一致。§8.4 另加「为什么 `dq` 不能要求逐位/绝对相等」段与「目标四条」，把口径写在门禁处。
6. **G1–G5 在新容差下仍抓得住 grid 截断（本项重点，我按判据逐条推演）**：
   - 截断时 request 1..7 的 `dk/dv/dq` **恒为 0**，而参照量级 ≈76 ⇒ `|Δ| ≈ 76`，容差 `5e-3 + 1.6e-2×76 = 1.22`（bf16）/ `6.9e-2+1e-3×1.05e3 = 1.12`（dq_acc）⇒ **超差 60 倍以上，必然失败**；
   - H=127 的**部分截断**（504/66,040 块）同理：request 0 只有前 4 个 key 块，`dk/dv` 缺失部分为 0、dv 累加不满 ⇒ 逐元素比对必然大面积超差；且 G3 额外要求「127 个 head 全部非零」，未执行到的 head 恒 0 ⇒ 直接失败；
   - 除数值判据外还有三重独立闸门：`dk/dv` **逐位相同**（截断必然破坏）、每 launch `grid ≤ 65,535` 的 host 断言、G4 的「异常快即失败」tripwire（截断只会变快）；
   - **反向**：容差放这么松会不会漏掉别的 bug？不会——A2 两臂、head 映射错误、h_off 传错等都会造成 ≥1 个 tile 的成片偏差（O(10)~O(100)），远超 1.12；而 ≤2 ulp 的 bf16 抖动按 §8.2④ 已证明必对应 ≤1 fp32 ulp 量级。**G1–G5 的判别力未见下降。**

**v1 建议 7 已闭合（L0-11b）**：新增 `H=64 / head_slab=32` ⇒ 2 次 launch（实跑 tiles=17/slab=32/grid=544/idle=0），用「预填 2.0 与清零各一次」区分「只清一次」与「每 launch 清一次」。**判据措辞有残留问题，见建议 1（非阻塞）**。

**v1 建议 6 已闭合（L0 编号）**：§8.3 现列出 L0-1…L0-12、L0-11b、L0-13…L0-23 共 24 行；我逐行与实现 `_bwd_kernel.py:914-955` 对照：L0-1/2 几何 `[256,384]/H=8` ✓、L0-3 `[512,96]/H=4` ✓、L0-9 `[300,512]`（v2 由 `[512]` 改正，与实现一致 ✓）、L0-13 `[0,64]/[64,64]` ✓、L0-14 `[1,1024]/[1024,1]` ✓、L0-15 `([6,8]/[8,4],causal)`＋`([5,3]/[5,7],非 causal)`、`H=2`、`D=8/6`、fp16 ✓（实现 `_case_golden_naive` 同形）、L0-16 契约 ✓、L0-17/18/19 的 H=32/16/16 ✓。§10.4 的「L0-1…L0-23（含 L0-11b）」现有对应内容。

**建议 1（非阻塞，需在 Stage 3 开工前改一句话）**：L0-11 与 L0-11b 的判定仍写成**严格相等** —— §8.3（第 1861-1862 行）「原子加而非覆盖（**差值恒为 2.0**）」「判定「预填版 **==** 清零版 + 2.0（逐元素）」」。这与同版新增的 §8.2 ①② 直接冲突：①明确「`dq_acc` 的累加顺序跨核不确定，不做逐位比对」，②要求「任何绝对阈值 ≥ 1 ulp」，而「== 2.0」是 **0 容差**（且是两次独立 launch 之间的逐位比较）。fp 加法不结合 ⇒ 预填 2.0 与后置 +2.0 的舍入路径不同（外加原子序可能不同），个别元素会偏离 2.0 约 1 ulp。**为何不阻塞**：现有实现 `_bwd_kernel.py:664-672` 已用等价且稳健的形态（`got_dq32 = got_dq32 - prefill` 后与参照按 dtype 容差比对），Stage 3 沿用即可通过；但这属「照字面实现会假失败」的隐患，且与本版刚立的规则相悖。**改法**：把两行判定改为「`|(预填版 − 2.0) − 清零版| ≤ atol + rtol·|ref|`（§8.2 口径）」，或直接引用实现的形态「预填后减去 2.0，再与参照按 §8.2 容差比对」；期望值：该差值应在 `≤1 ulp(|dq_acc|)` 量级，远小于 atol。

**建议 3（非阻塞，诊断量口径）**：§8.2⑤ 的 `k_ulp = max|Δ| / ulp(max|ref|)` 若「最大差元素」的幅值远小于 `max|ref|`，会**低估**抖动（例：|Δ|=1e-4 落在 |ref|=10 的元素上即 105 ulp，但分母用 ulp(1,053)=1.22e-4 会报 0.8）。改为逐元素归一再取最大 `max_i |Δ_i| / ulp(|ref_i|)` 更能反映「是否仍在 1-2 ulp 内」。同一建议顺带：§12.6「v2 的验证」把实测 `2.0` 归因为「`ulp_bf16=2.0` 出现在 `|dq| ∈ [256,512)` ⇒ 元素自身的 1 个 bf16 ulp」——该结论可由 §8.2④ 反推（若 2.0 落在 |dq|≈76 就是 4 ulp ⇒ fp32 层 2.3e5 ulp ⇒ 真实缺陷，与「weight requantization」的定性不符），但**文中未给出这一步推导**，而 `/tmp/opencode/MLA_VARLEN_BWD_8x4K_FINDINGS.md` §D 只给了「`|值|≈76`、≈2 bf16 ulp」的模糊表述。建议把 §8.2④ 的推导显式引用过来（一句话），否则读者会误以为设计凭空断言了元素幅值。

### 7. 完整性与一致性 — pass（v1 建议 2/5/8/9/10 已闭合）

- **建议 2（block_n 同源）闭合**：§5.6 g 新增「`block_n` 必须来自 `cfg["block_n"]`、`head_slab` 取自 `plan.slab`」+ **运行期后置断言**（`(total_kv//cfg_block_n + batch)·plan.slab == plan.max_grid`、`meta["head_slab"]/["block_n"]` 校验），并给出反例 `4,104×31 = 127,224 > 65,535`（我实跑复算 = 127,224 ✓）；同步落到 §4.8、G5 ⑥、L0-20、§9.1 风险 19、§9.2 反例 —— 五处闭环。
- **建议 5（L2 = 128 MiB）闭合**：C4 改为「≫ UB 248 KB + L1 512 KB + L0C 256 KB 之和（≈1.0 MB）的 ~4×10⁴ 倍以上」（数值不敏感，结论不受影响）；§1.6.0 估算行明确「**未文档化假设**」并给出「L2 ≥ ~34 MB 即驻留」的敏感性口径；风险 6 撤掉「硬件档案」引用。我 grep 确认「128 MiB」只剩解释性提及。
- **建议 9（笔误）闭合**：G1 ④ 现为 `dk/dv/dq` ✓。
- **建议 10（TRAP stamp）闭合为「有依据的处置」**：§12.5 新增 v2 状态段，说明 facet 已补、stamp 由 evolver 统一更新、结论不依赖条目状态。我实读 `.agents/platforms/toolchains/ascend/knowledge/traps-950pr-execution.md:64-81`：facet 内容与 findings 一致（66,560→1,024、H=127→504、H=126 正常），stamp 仍为 `a010a95…`（当前 `7a52f2f1…`）—— 与设计所述一致；该处置（不越权改知识库）**可接受**。
- 无占位符（design_lint 覆盖）、§3.5.3 引用的 11 个路径全部真实存在（我逐个 `test -e`）。
- 内部一致抽查全部通过：smoke 12/8/96、L0-11b 17/32/544、L0-17/18/19 的 grid、L0-21 的 65,520、G1 的 2,080、A4 的 `n=568908=568904+4`、`16640/65535=25.4%`、余量 48,895/49,215、`1040/66560=1.56%`、launch 边界 L2 上界 `3×5.37≈16 GB ≈2.2%`、`1669×4096+64×256=6,852,608`、`5,623,296×279,280=1.571 TB` —— 与各表数字逐格一致。
- **建议 2（本轮新增，非阻塞）**：§4.8 的 launch 片段用**裸 `assert`** 承载这两条安全断言（`assert (total_kv // cfg_block_n + batch) * plan.slab == plan.max_grid` / `assert meta[...]`），且首行仍写死 `cfg_block_n = 64`。`python -O` 会移除 `assert` ⇒ 一旦运行在优化模式，**这条防止静默截断的最后一道闸门会消失**；同时「写死 64」正是 §5.6 g 自己要禁止的形态。改法：把 §4.8 的两条 `assert` 换成与 §5.6 g ③ 一致的 `if …: raise ValueError(...)`，并把首行改为 `cfg = factory_config(...); cfg_block_n = cfg["block_n"]`（与 §5.6 g ③ 的代码块同形，避免两处示范互相矛盾）。

### 8. 算法优化分析 — pass（v1 建议 4 已闭合）

**§1.6 章节存在性**：1.6.0/1.6.1/1.6.2/1.6.3 齐全；v2 在 1.6.0 R3/R4（C4 表述、−5.8%）、1.6.1（#5 的 +23.1%）、1.6.3（备选 −5.8%）、1.6.2 #16（`h_off` 读）上做了**数值/措辞**修正，未删任何分析。

**调研完整性与结论正确性（独立复核）**：R1 候选表 13 行含基线、R2 有明确结论 + 结构依据、R3 四口径 C0–C7、R4 逐候选 6 维 —— 结构与 v1 相同。**v2 改过的三处百分比我全部独立复算**：
- **建议 4 闭合**：`delta` rank-1 拆分 `+192×64×64 = 786,432 MAC`；单迭代合计 `786,432(st)+524,288(dpt)+524,288(dv)+786,432(dk)+786,432(dq) = 3,407,872 MAC` ⇒ `786,432/3,407,872 = 23.08%` = **+23.1%** ✓，两项各一次 = **46.16%** ✓，与 v2 数字一致（v1 的 +5.7% 确为错记，否决结论不变且更强）。
- 备选 `block_n=128/block_m=32` 的 L1 端口 `1 − 263.2/279.3 = 5.76%` ⇒ **−5.8%** ✓（v1 的 −5.9% 已改正，§1.6.0、§5.2、§1.6.3、§5.6 f 四处统一）；其每迭代 FLOP/字节 `6,852,608/263,200 = 26.0` ✓，合计 `1.571×0.9423 = 1.480 TB` ✓。
- 风险 6 的「原子流量 −50%」：备选 `block_m=32` ⇒ 每次原子写 `32×192×4 = 24,576 B`（vs 49,152）⇒ 原子总量 `276 → 138 GB` ✓；且 `bn·bm = 128×32 = 64×64` ⇒ **迭代数不变**（`L²/(2·bn·bm)` 不变）✓，故 −50% 与「L1 端口同期仅 −5.8%」**可同时成立**（不是同一分母的错配）；总 GM `744 → 605.8`（−18.6%）✓。
- C4 的容量淘汰改为「≫ 在片容量之和 ≈1.0 MB 的 4×10⁴ 倍」（`45.8e9/1.0e6 = 4.6e4` ✓），对 L2 数值不敏感 ✓。

**调研→设计一致性**：§1.4 步骤 1 / §3.1 步骤 0–0'' / §6.1 首三行仍以 C1 + A4 为对象；§0.5/§0.6 处置与选族一致；源 §0.4 的 15 项在 §0.5 全有承接或明确舍弃理由。A4 的 (i)–(vi) 论证与 v1 逐字相同（我做了区段比对），其**双射命题我再次独立推演**：`bid ∈ [0,tiles·slab)` 经 `(bid//slab, bid%slab)` 到 `[0,tiles)×[0,slab)` 是双射；`H_i = [i·slab, min((i+1)·slab, heads))` 两两不交且并为 `[0,heads)`；过滤 `h<head` 后对 i 取并得 `[0,tiles)×[0,heads)` 每点恰一次 ⇒ 无缺失/无重复；写集按 head 不交（head 占行内连续 `D`）、无跨 head 归约 ⇒ 无需 launch 间同步。**成立**。

**等价性论证与机器验证**：A1/A2/A3 未改（脚本 md5 未变、5/5 重跑 ALL_PASS）；A4 只改了代价措辞。**判据构造**仍非自证（见 M1）。

**向量化替代覆盖（与 §3.3/§6 交叉）**：17 个计算点、12 项已向量化/换单元/由平台 pass 消除、5 项设备侧标量 + 1 项 host 编排；v2 只把 #16 的「2 个标量 op」补成「2 个标量运算 + 1 次 4 B `h_off` GM 读」，与设备码一致；§6.1 没有 §1.6.2 未论证的逐元素标量循环。**不可替代理由**具体到本算子（block 级控制流/寻址、host 元数据、kernel 作用域无线程域）。

**§1.6.3 布局决策**：候选矩阵 8 行、量化评分与弃选理由齐备；v2 只改了「−5.8%」，选定的候选 1 + 7 与 §3.3/§4.5/§5.6/§6.1 四方仍一致；实验裁决三件套（主选 + 结构完整的备选 + 阈值 0.9 + 回写路径）保留。

## 已闭合项（v1 阻塞 1 / 建议 10 → v2 逐条核对）

| v1 项 | 内容 | v2 处置 | 本轮核对结果 |
|---|---|---|---|
| **阻塞 1** | §8.4 G1/G2/G3 + §8.3 L0-12/L0-21 的 `dq ≤1e-5` 不可达且与 §8.2 矛盾 | 全部改为 §8.2 dtype 容差；§8.2 新增唯一口径 ①–⑤ + `k_ulp` 诊断；§8.4 增「目标四条」与「为什么不能逐位/绝对相等」 | **闭合**：全文无可执行 `1e-5`；§8.2③ 的 1 ulp / 565 ulp / 8,627 ulp / 0.081 dq 单位与我独立复算一致；bf16 行「恒覆盖 ≥1 ulp」有结构性证明；G1–G5 判别力未下降（截断 ⇒ 超差 60×） |
| 建议 2 | `block_n` 与工厂不同源 | 同源规则 + 运行期后置断言 + 反例 127,224；落 §4.8/§5.6 g/G5⑥/L0-20/风险 19/§9.2 | **闭合**（反例算术我实跑复算 ✓）；残留：「裸 `assert` + 写死 64」→ 本轮建议 2 |
| 建议 3 | smoke 几何两处不一致 | §5.5 改为 manifest-smoke（12/8/96），L0-3 单列为 44 | **闭合**（我实跑 96/44 均一致） |
| 建议 4 | 弃选项 `+5.7%` 应为 `+23.1%` | 改为 +23.1% / +46.2% 并写出分母；另复核 −5.8%、原子 −50% | **闭合**（三项均独立复算一致） |
| 建议 5 | 「空块不读任何 GM」与设备码不符 | 七处统一为「1 次 4 B `h_off` 读 + 2 个标量运算」 | **闭合**（非更正语境已无残留） |
| 建议 6 | L2 = 128 MiB 无溯源 | C4 改数值不敏感表述；标注未文档化假设 + 敏感性口径 | **闭合** |
| 建议 7 | §8.3 缺 L0-13…19 行 | 补齐 13–19 与 20–23，另加 L0-11b | **闭合**（逐行与实现一致） |
| 建议 8 | L0-11 单 launch 不覆盖「slab 间不清零」 | 新增 L0-11b（H=64/slab=32 ⇒ 2 次 launch） | **闭合**；判据措辞残留 → 本轮建议 1 |
| 建议 9 | G1 ④ 笔误 `dk/dk` | 改为 `dk/dv/dq` | **闭合** |
| 建议 10 | TRAP stamp 滞后 | 记为 evolver 统一更新、结论不依赖条目 | **闭合**（处置可接受） |
| （v1 维度 3 内嵌）| auto-shrink 的 slab 取约数 | 未处理 | 作为本轮**建议 4**重提（manifest 无影响） |

## 留待 Stage 3 的必做验证（设计已定义，实现期必须真跑并留存证据）

1. **G1｜H=128 / B=8 / L=4096 逐 request 拆分对照（v2 文字中的「v1 主线」，最高优先级）**
   ① 先断言 launch 计划 `tiles=520 slab=32 launches=4 max_grid=16640`、每次 `grid ≤ 65,535`；
   ② 对每个 `b` 用同一片数据构造 **B=1**（`tiles=65 slab=32 max_grid=2080`）作参照；
   ③ 全量结果的 `[cu_q[b]:cu_q[b+1]]` 行与单跑结果比对：`dk/dv` **逐位相同**、`dq_acc`/`dq` 按 §8.2 容差、`k_ulp` 作诊断；
   ④ **逐 head 能量检**：`dk/dv/dq` 的 128 个 head 切片 `|·|.sum() > 0`（截断时 request 1..7 恒 0 ⇒ 立刻失败）。
   参照几何必须远离上界（2,080 ≪ 65,535），否则参照会被同一个 bug 污染。
2. **G5 ⑥ / L0-20｜同源后置断言**：`cfg["block_n"]` 同时喂 `plan_launch` 与工厂；`block_n ∈ {64,128}` × `head_slab` 默认/显式，其中 **2 组故意不一致必须抛 `ValueError`**；另跑 G5 ①–⑤（65,536→raise、65,535 合法、131,072 自动缩小到 slab=31/63,736、内联副本与模块版逐字段一致）。
3. **L0-11b｜multi-launch 原子契约**：`H=64 / head_slab=32`（2 次 launch，`tiles=17/slab=32/grid=544`），预填 2.0 与清零各一次，验证「slab 之间不清零」；落在实现的稳健形态上（减 prefill 后按 §8.2 容差比对，见本轮建议 1）。
4. **G3｜H=126 / H=127 边界**：尾 slab `idle=1,040 / 520` 的计划断言、映射不变性（`head_slab ∈ {32,63}` 各跑一次，`dk/dv` 逐位相同）、H=127 的 127 个 head 全部非零。
5. **G4｜tripwire**：latency 落在 400–600 ms 量级、achieved ≤ 378 TFLOPS（截断只会「变快」，20 ms 级或 > 378 即失败）。
6. **H1（§12.6 移交项）**：现有 `_bwd_kernel.py:893` 的 `_case_arm_equivalence` 仍用 `dq > 1e-5`；Stage 3 重写时按 §8.3 L0-12 的新口径（`dk/dv` 逐位、`dq_acc` 走 §8.2 容差 + `k_ulp` 诊断），**不得沿用 1e-5**。
7. **h_off 传递**：逐 launch 记录 `head_off`（`describe`），结合 G3 的映射不变性与逐 head 非零检查确认漏 head 会暴露（风险 16）。

## 备注（供 conductor / Stage 3）

- 本次为**设计检视**，未运行任何设备 kernel；机械复核（verify_equiv 重跑、design_lint、design_calc_check、`launch_plan` 复算、v1↔v2 区段 diff、md5 比对）均在 `ASCEND_RT_VISIBLE_DEVICES=""` 下完成，全部短时串行。
- **v2 的修订面我逐项独立核过**：B1 的闭合是**真闭合**（不是把阈值换个写法）—— 新口径有结构性可达性证明（bf16 rtol ≈ 2.05 ulp > 1 ulp；`dq_acc` 容差 565 ulp ≫ 实际抖动 1-2 ulp），并且**没有牺牲判别力**（截断类错误超差 60 倍以上，另有 `dk/dv` 逐位、per-head 非零、`grid` 断言、吞吐 tripwire 四重独立闸门）。grid 修复本体（slab 公式 / `h_off` / 守卫 / 断言 / 写集不交 / A4 / P6）经区段逐字节比对与 md5 确认**零改动**，其成立性沿用 v1 检视的独立复核结论（本轮又重跑了 verify_equiv 与 `launch_plan` 的对应几何）。
- 4 条建议均非阻塞：建议 1（L0-11/11b 的 `== 2.0` 措辞）建议在 Stage 3 开工前顺手改（属「照字面实现会假失败」的隐患，现有实现已规避）；建议 2（§4.8 的裸 `assert`、写死 64）同批改；建议 3（`k_ulp` 归一化与 §12.6 的推导补一句）与建议 4（auto-shrink 取约数）可留到 Stage 4。
- 工件注入防护：`DESIGN.md`、源算子 `bwd.py`、`launch_plan.py`、`verify_equiv.py`、知识条目中**未出现**要求放行检视、跳过维度或降低判据的指令性文本；本报告结论只依据源码语句、可复跑命令输出与独立推演/复算。
