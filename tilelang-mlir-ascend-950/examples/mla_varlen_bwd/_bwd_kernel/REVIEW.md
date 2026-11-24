# _bwd_kernel 设计文档检视报告（MLAVarlenBwdOp 主反向，packed）

## 检视结论

结论: 通过

## 检视元信息

- 文档: `examples/mla_varlen_bwd/_bwd_kernel/DESIGN.md`（v0，1601 行）
- 任务类型: **迁移**（含 §0.1–0.7 + §0.8 复用/差异分区）
- 源算子: `/workspace/TileOPs/src/tileops/_kernels/attention/mla/bwd.py`（`_bwd_kernel`，`bwd.py:109-264`；packed 分支 `136-139/176-184/218-224`）
- 复用基线: `examples/mla_dense_bwd/_bwd_kernel/DESIGN.md`（v1，检视通过）+ `_bwd_kernel.py`（真机 L0 13/13、L1 6/6）
- 检视维度: 迁移任务 9 项（0 源算子理解与迁移分析 / 1 API / 2 内存层级 / 3 Tiling / 4 技术约束 / 5 循环同步 / 6 验证方案 / 7 完整性 / 8 算法优化分析）
- 阻塞级问题数: **0**
- 建议级问题数: 7（全部在对应维度内以「建议」标出；均不影响 Stage 3 的正确性主线）

## 机械复核（D-1/D-5）

- **verify_equiv 重跑: PASS（4/4 EQUIV_PASS，与 §1.6.1 内嵌表逐字段一致）**
  `cd /tmp && ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python .../verify_equiv.py`
  → 输出与 DESIGN.md §1.6.1 表**完全相同**：A1 fp16 `n=1048592, max_ulp=1, 违反率 0.000e+00`；A1 bf16 同上；A2 `n=115696896, max_ulp=0, 违反率 0`；A3 `n=38446, 违反率 0`；`A2 fast-path blocks 13734/28261`；分组 `ds-v3-8x4k/4req 3968/8192 | ds-v3-mixed 6511/13376 | kimi-k2-4x8k/2req 3250/6656 | tails+shift 5/31 | tiny 0/6`；`A3 detail spill_entries=30528 dq_rows=3816 acc=4102 zero_sign_flips=1`；`EQUIV_SUMMARY ALL_PASS`。脚本内容经逐行审阅：A2 用**真实块枚举**（manifest 长度集合 + 尾块 + shift 正负）验证「fast 谓词 ⇒ 块内 vis 全真」（`assert vis.all()`）与掩码臂逐元素一致；A3 检查越界列 `dS == 0` 且 `ds16·k` 的 fp32 累加为 0，并对 `-0.0/次正规/3.4e38` 验证 `+0.0` 原子加数值不变。**判定非伪造**。
- **design_calc_check: 3 pass / 1 skip**
  - `ub_budget`: pass — `UB: 11 buffer entries sum = 88.5 KB; capacity = 248.0 KB`
  - `l0c_budget`: pass — `block 64x64 x fp32 accum = 16.0 KB vs L0C 256 KB`
  - `r3_metrics`: pass — `all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak)`
  - `core_split`: **skip**（`insufficient parsed dims {'bm':64,'bn':64}; declared logical cores = None`）→ **人工已核**：§5.5① 四个 workload 的逻辑核数复算全部正确（8×64×128+8×128 = 66,560；504×128+6×128 = 65,280；128×4×64+4×64 = 33,024；smoke (8+2+1)×4 = 44）；§5.5② 物理核数取自平台落库实查记录（`core_source`: `Ascend950PR_9579.ini ai_core_cnt=28 …; rtGetAiCoreCount()=28`，本次 `statectl platform` 返回同值 28/56）；§5.5③ 有规模判定与分核方案。
- **人工复算（机械未覆盖项）**：UB = 8192+8192+3×4096+2×256+24576+16384+12288+8192 = **90624 B**（35.7%）；L1 = 24576+16384+24576+24576+24576+16384+16384+3×8192 = **172032 B**（32.8%）；L0C = 49152+32768+16384+16384+49152 = **163840 B**（62.5%）→ 三项与 §4.5/§4.6 逐字节一致，且与 dense 真机实现的 buffer 清单一字不差（逐 buffer 复核 `_bwd_kernel.py:250-260`）。
- **编译探针重跑（本 Agent 独立执行）**：`design_probe_compile.py` → `4/6 COMPILE_OK`，与 §3.5.4 表**逐条一致**：P1 40956 B / P2 27084 B / P3 25066 B / P5 27084 B 全 COMPILE_OK；P4、P2b 均 `InternalError: Unsupported scope shared.l0c`（反例成立）。并额外 dump P2 设备码逐项核对（见维度 0/1）。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

**逐项与源码核对（Read 全 387 行）**：

| 检查项 | 结论 | 源码证据 ↔ DESIGN |
|---|---|---|
| 语义正确性 | pass | `bwd.py:229-243`（`st=exp2(st·scale·log2e − lse·log2e)` + `(m0+j<len_q)&(n0+i<len_kv)&(n0+i<=m0+j+shift)`）↔ §0.1/§1.3 的 `P=exp(scale·q·k − lse)·1[…shift]`；`bwd.py:246-251`（`ds16=st·(dpt−delta)`、`dq=gemm(ds_s,k_s,transpose_A)`）↔ §0.3 S13/S16；**bottom-right 对齐 `n<=m+shift` 与源逐字一致**；规约语义（`T.clear(dk_f/dv_f)` 后逐块 fp32 累加 `bwd.py:200-201,248`，`dq` 逐元素原子加 `252-254`）↔ §0.1 规约行 |
| I/O 与转置布局 | pass | `bwd.py:141-154` 的 11 个参数顺序/形状/dtype ↔ §0.2 表逐行一致；`stat_shape=(heads,total_q)`（`bwd.py:137`）↔「head-major `[H,T_q]`」；`dq_acc` fp32 in-out（`151`）、`dk` 乘 scale（`257`）、`dv` 不乘（`260`）、`o` 只被 `_delta_kernel` 消费（`341`）↔ §0.1 host 侧行 |
| 算法解读完整性 | pass | §0.3 S1–S18 覆盖 `bwd.py:176-260` 全部计算语句（含 packed 三件事：`cumsum_offsets` `181`、`live`/`decode` `182-184`、head-major `lse[h,q_start+m0+i]` `219-224`），无遗漏步骤、无臆造步骤；host 侧三 launch（`338-347`，`dq_acc=torch.zeros` `342`）已列出 |
| 优化手段覆盖 | pass | §0.4 15 项：`T.clear` 显式清零（`200-201,225,227,250`）✔、`transpose_A=True`（`251`）✔、`exp2+LOG2E`（`235`）✔、`atomic_add` 逐元素（`252-254`）✔、`GroupTiling` warp 扫描（`138-139,181`）✔、`T.Pipelined(num_stages=2)`（`204`）✔、`policy=FullRow`（`226`）✔；**#13 `clear_accum` 明确标注为「NPU 适配构造，非源码手段」**（源码确无该参数）——未把实现层手段误报为算法层 |
| 耦合性判定 | pass | §0.5 22 行四态（保留 7 / 等价替换 8 / 重新设计 6 / 舍弃 1；**统计行写「21 行 / 保留 6」，与表实有 22 行、保留 7 差 1，见本节末尾建议**）。抽检关键判定均与平台资料一致：① `GroupTiling` warp 扫描判「NPU 无对应」——本次亲查 `grouped_tiling.py:141-150` 确用 `KernelLaunchFrame.Current().get_thread_extents()+T.shfl_up+WARP_SHUFFLE_STAGES`，而 `tilelang/ascend/language/kernel.py:4,146` 原文「no SIMT thread domain at kernel scope… this dialect has no `threads` parameter」→ 判定成立；② `transpose_A` 判「L1 路径禁止」——`gemm_mad.py:162` 原文 `assert not self.trans_A and self.trans_B`；③ L0C 唯一可累加 + `T.clear` 不支持——P4 本轮重跑复现 |
| 重设计可行性（R1–R7） | pass | R1 一维网格 + `tile=bid//H`（静态除数，P2 设备码 `block_idx >> 1`）；R2 L0C 累加 + 半区 UB（dense 真机）；R3 NT-only + `ds` 转置副本（`T.gemm` 全 extent，`gemm_op.py:62` 静态断言已被 P1/P2 规避）；R4 `SimtVF(128)` + `sid` 捕获（dense 真机）；R5 store-atomic + 溢出零行（A3）；R6 融合式 decode（P2 设备码逐句复现，见差异 ①）；R7 `dk/dv` 双分支（P2 设备码双路径，见差异 ③）。**每项都有源方案 → NPU 新算法 → 语义保持论证三要素**，且新算法不触碰平台 Kernel/存储/对齐约束 |
| 设计与决策一致性 | pass | §1.4 与 §0.6 同族（key-stationary + R1–R7）；§3.1 九步全为 NT 形式；§4/§5/§6 无「§0 判重设计、后文照抄源方案」的残留（对比 dense §11.3 明确列出三条**不再成立**的 dense 结论：平台 clamp 写回、整块 L0C→UB、`T.clear` L0C——这三条正是 packed 下必须替换的，处置正确） |
| golden 独立性 | pass | §8.1 golden 只读 `q,k,v,do,cu_seqlens_*`，**自算 `lse`/`delta`**（不用输入统计量）、按 request 物化 `P` 后 `einsum`、`nan_to_num(0)` 处理 `-inf` 行、fp32 只舍入一次 —— 与源 `mla_bwd_reference`/`ref_program` 同构，且与本设计的分块/半区/原子/双分支**无共享假设**；`dq_acc_true=dq/scale` 的中间量定义正确 |

**packed 三处差异逐项验证（本 Agent 独立核对，全部成立）**：

- **① 块枚举（R6）— pass**。`tile_upper_bound` 亲查 = `numel // block_m + num_groups`（`grouped_tiling.py:46`）↔ §0.3 与 grid 式一致；上界性独立证明：`Σceil(l_b/64) ≤ (T+63B)/64 < T//64 + B`，故「多算 ≤ B 个 tile」成立。`cnt` 在 `g` 末等于 `s_cum[g+1]`、first-match 与源二分解码同解（含 `nt=0` 空组两侧都取不到）、`n0=(tile−cnt)·64` 同式 —— 三条等价性论证经推演成立。**P2 设备码实测**：`for (g…) { ntg = ((cu_kv[g+1]+63) − cu_kv[g]) >> 6; if (cnt <= tile && tile < cnt+ntg) { req=g; n0=(tile*64 − cnt*64); } cnt += ntg; }` + `if (tile < cnt) live=1;`（`/tmp/opencode/p2_src.cu:104-121`）—— 与设计伪代码逐句对应。`len_q=0` 退化：源 `T.clear(dk_f/dv_f)`（`200-201`）→ 0 梯度，本设计用 `loop_ed=max(ceil,loop_st+1)` 保证一次全掩码迭代产出精确 0，**语义等价**（且额外获益：避免 L0C 未初始化）；`loop_st<loop_ed` 恒成立（`n0−shift ≤ len_q−1`）故该 max 不改变非退化行为。负载不均 512→16384 的量化（最长块 0.0046%）复算正确（5.62M 迭代、256 次迭代/最长块）。
- **② `lse`/`delta` 保持 head-major — pass**。源 `stat_shape=(heads,total_q)`（`bwd.py:137`）+ 取址 `lse[h, q_start+m0+i]`（`219-224`）↔ §1.6.3/§3.2/§4.1 未转置成 dense 的 `[B,H,S_q]`；**P2 设备码**：`asc_copy_gm2ub_align(… lse[kq*64 + (block_idx&1)*total_q + q_start] …, 1, (min(64, max((total_q−q_start)−kq*64,0))*4), …)` —— 单次 ≤256 B 连续段 + head 步长 `total_q`，与 §3.2 步骤 3 与 §11.2 差异 ② 的描述**逐字符一致**；lane=query 的 `lse_u[j]` 私有读不变。
- **③ 全偏移读写 — pass**。（a）`dq` 整块原子 + A3 溢出零行：独立推演成立 —— `gj≥len_q` 的列 `pv=0`（`T.if_then_else` 选择），`dp_u`/`dlt_u` 在该处只可能有限（下一 request 的真实值或补 0），故 `dsv=0*有限=精确 0`，`dq_c[j,:]=Σ_i 0·k[i,:]=0`，与源「只写有效行」数值等价（仅 `-0.0→+0.0` 符号翻转）；`dq` 的 DMA 长度按张量尾裁剪，P2 设备码 `asc_copy_l0c2gm(…, min(64, max((total_q−q_start)−kq*64,0)), 384, 64, …)` 确认；机器验证 30528 + 3816 全零、0 违反。（b）`dk` 乘 `scale`、`dv` 不乘 —— 与 `bwd.py:257/260` 一致，§3.1 步骤 6/9 与 §8.2 均正确。（c）`dk/dv` 双分支**不越 request 边界**：快路条件 `own+HN ≤ kv_end` 保证 32 行全在本 request（内部块两 AIV 均满足）；慢路 `own+i < kv_end` 是源逐元素守卫的直接对应，`own ≥ kv_end` 的 AIV 写 0 个元素；**边界穷举**（`len_kv−n0 = 20`：AIV0 走慢路写 20 行、AIV1 不写；`len_kv−n0 ≥ 64`：两 AIV 均走快路）无漏洞。P2 设备码同时出现 `if (own+32 <= kv_start+lkv) → asc_copy_ub2gm_align(...)` 与 `else → simt_vf_3`（其内 `if (... < kv_start+lkv) && 0 <= … && … < total_kv`）—— 双路径成立，且编译器额外补了 `0 ≤ idx < total_kv` 守卫。**§11.3「平台 clamp 只认张量 extent、不认 request 边界」的判定经 P2 设备码核实无误**（clamp 形如 `min(64, max((total_kv−kv_start)−n0, 0))`），这是 packed 下 dense「clamp 足够」结论确实失效的关键点，设计已单独重推 R7。

**复用有效性判定（同一源码工厂是否真的等价）— 成立**：亲读源工厂确认「packed/dense 只差 `stat_shape`、`num_tiles`、`request/n0` 解码、`lse/delta` 取址」四处（`bwd.py:133-139`、`176-184`、`214-224`），其余 5 个 GEMM 的语义/操作数方向/边界语义逐字相同；故 §0.8-A 的 9 项复用（算法族、5 个 NT 乘积与维度、全 extent GEMM+clamp+掩码、半区 `dual_copy`+`sid`、store-atomic、A1/A2、`num_stages=1`、UB/L1/L0C 预算、hybrid 模式）在 packed 下继续成立，而 §11.3 已把**不再成立**的三条 dense 结论显式撤销。三处预算逐字节复算与 dense 实现相符（90624/172032/163840），进一步印证「复用项由维度与平台约束决定、与索引空间无关」这一论证。

> 建议（非阻塞）：§0.5 统计行「21 行判定 / 保留 6 项」与表实有 22 行 / 保留 7 项（多出末行 `host torch.zeros(dq_acc)`）不符，建议改为 22 行 / 保留 7。§0.7 的 manifest 路径 `src/tileops/manifest/spec/attention.yaml` 应为 `src/tileops/manifest/attention.yaml`（抽取件为 `examples/TileOPs/tileops/manifest/attention.yaml`），且该处只列 3 个 workload，manifest 实有 **4** 个（另有 `mla-varlen-bwd-smoke`：`q_lens=[256,384]`、`H=8`、双 dtype），建议补入（该 smoke 的 B=2/H=8 也是 §5.5① 逻辑核数最小的真实用例）。

### 1. API 可行性 — pass

- **存在性**：§3.2 每条 API 均有平台侧佐证且本 Agent 复核：`T.Kernel(grid)` 一维无 `threads`（`ascend/language/kernel.py:4,146` 原文）；`T.gemm(A_l1,B_l1,C_l0c,transpose_B=True)` 仅 NT（`gemm_mad.py:162` 断言原文）；`T.gemm` 静态 extent（`gemm_op.py:62` `requires static tile dimensions`）；`T.dual_copy` 语义与路径集（`copy_op.py:53-65` 文档：M/N 分片、混合 kernel 专用）；`T.set_atomic` / `T.SimtVF(128)` / `T.exp` / `T.if_then_else` / `T.call_extern(asc_get_sub_block_id)` / `T.alloc_l1`/`alloc_l0c` 均有已核验 repro 条目（`PL-950-*`）或 dense 真机实现。
- **形态可用性（比符号存在更强）**：本轮**独立重跑** 5 个编译探针 + 1 反例：P1/P2/P3/P5 COMPILE_OK，P4/P2b 以 `Unsupported scope shared.l0c` 失败；并 dump P2 设备码核对到 `__mix__(1, 2)`、`ascend_gemm_l1 ×5`（含运行期布尔 `clear_accum` 实参 `kq == max(0,(n0+lq−lkv)>>6)`）、`asc_copy_l0c2ub`、`ascend_nd2nz_scatter`、`asc_copy_l0c2gm`（夹在 `asc_set_atomic_add_float`/`asc_set_atomic_none` 之间）、`asc_fill_l1 ×7`、`asc_copy_gm2l1_nd2nz/dn2nz`、`asc_copy_gm2ub_align`、4 个 `asc_vf_call<…simt_vf_0..3>`、`int64_t sid_1 = asc_get_sub_block_id()`。**§3.5.4 的设备码结论（按调用种类 + 参数形态比对，不计事件 id）全部成立**。
- **反例纪律**：`T.clear(L0C)`、符号 GEMM extent、`T.Vector(vector=2) as sid`、`@T.prim_func` 内直调普通函数 四条反例均为本机实测或 dense 实测，设计**未复用被证否的机制**，回退阶梯（P5 静态 grid、P3 符号长度 copy）本身也各自 COMPILE_OK —— 符合「回退阶梯可执行性」。

> 建议（非阻塞）：§3.5.4 与 §1.6.2 #11 有一处笔误（「`asc_copy_ub2gm_align` 与 `asc_copy_ub2gm_align` 的守卫分支」应为「快路 `asc_copy_ub2gm_align` 与慢路 `simt_vf_3`」）。另 `1.6.2 #9` 引用的 `T.shfl_up` 属通用 tilelang language，在 ascend 路径的 kernel 作用域无线程域可用，建议在 R6 的依据里补一句该 API 的**路径级**归属，避免后续误以为可在 `T.SimtVF` 外使用。

### 2. 内存层级规划 — pass

- §4.4 搬运路径完整（GM→L1/L0→L0C→UB→L1/L0C→GM 原子/UB→GM 双分支），每步标注层级与 API；§4.3 列出 19 个 buffer 且逐项标层级（L1 10 / L0C 5 / UB 8 + 两组标量）。
- **UB 90624 B / 253952（35.7%）**、**L1 172032 B / 524288（32.8%）**、**L0C 163840 B / 262144（62.5%）**：机械 pass + 本 Agent 逐 buffer 复算一致（见机械复核段）；单累加器最大 `dk_c = 64×192×4 = 49152 ≤ 262144` ✓。L0A/L0B 交给 GEMM lowering，并给出 L0 回退路径的 53248/61440 B 预算（93.8%，余量 4 KiB，只允许单版本）——量化完整。
- 与 dense 真机实现逐 buffer 相同（`_bwd_kernel.py:250-260`），故「预算可承袭」不是纸面推断而是已运行配置。

### 3. Tiling 策略 — pass

- §5.2 `block_n=block_m=64`、`_HN=32` 有理由（L0C 中 `dk_c/dv_c` 常驻 80 KiB；`dq_c` 48 KiB；16 倍数与 `D_qk=192=12×16`、`D_v=128=8×16` 整除）；`block_n=128 + block_m=64` 会使 L0C 到 272 KiB 溢出 → 只能配 `block_m=32`（216 KiB / 84.4%），已作为 §1.6.3 备选。
- 分核三要素齐全：① 逻辑核数表（四个 workload，本 Agent 复算 66,560 / 65,280 / 33,024 / 44 全对）；② 物理核数**实查记录**（平台 `core_source`：`ai_core_cnt=28, cube=28, vector=56` + `rtGetAiCoreCount()=28`，与本次 `statectl platform` 返回一致，非文档假设）；③ 规模判定「极大规模」+ 方案「1 逻辑块 = 1 次发射（逻辑块超发）」。
- **规则偏离判定**：方案偏离 `core-split-strategy.md` §1 的极大规模分支（该分支要求「固定启动内核数 = 物理核数，核内分批」），但设计**显式声明偏离**并给出三条依据：(a) 单块 597 MFLOP ≫ 派发开销、(b) packed 负载不均 8→256 次迭代必须动态派发、(c) 核内串行需把 Cube+Vector 混合体包进任务循环，而平台原文「手工跨核 flag、循环复用下的复杂同步和跨 block batch barrier 仍未认证」（`programming.md`，本 Agent 已查）。**且同形态（逻辑块超发、无核内串行）在 dense 真机 L0 13/13、L1 6/6 通过**，故偏离有实证支撑；同时给了三条复评触发条件。判定：三要素齐全、核数与 block 自洽、无「核内循环违反平台限制」→ pass。
- 非整除：§5.4/§6.4 统一口径（GEMM 全 extent；装载 clamp+fill；掩码保留区间比较；写回双分支）完整。

> 建议（非阻塞）：§5.5③「block_n=128 → ds-v3-mixed 仍有 ≈32,640 个逻辑块」按 `Σceil(len/128)×H + B×H` 复算为 **33,024**，建议订正数字（不影响「≫28」的结论）。

### 4. 技术约束检测 — pass

- 三维 Kernel 不涉及（一维 grid + 块内解码，§3.5.1 首行）。GEMM 非整除由「全 extent + clamp/fill + 掩码」收口、L0C 溢出由 §4.6 处置、`T.clear` 不支持 L0C 已用 `clear_accum=(kq==loop_st)` + 非空循环规避（P4 反例本轮复现）。
- **动态 grid + 混合 kernel**：判定为 **Stage 3 前置探针，非设计阻塞**。理由三条（与 dense Stage 2 复盘沉淀的放行三条件一致）：① 有可判定标准（§8.3 R1/R2：非整除长度 + B≥2 的三个梯度容差、哨兵查越界、两个 shape 的 grid 实参不同）；② 有**可编译**回退（P5 静态 grid 上界，本轮 COMPILE_OK，只需 host 契约 `total_kv ≤ TK_MAX`，`live` 与其余代码不变）；③ 已排程为 Stage 3 首个探针（§8.3 实施顺序 R1/R2 → R6 → R3/R4 → R5 → R7）。同机制先例 `examples/paged_cache_write`（设备验证）+ dense 同骨架真机通过；另需注意「动态 grid 是先例中的纯 Vector 形态、本函数是 `__mix__(1,2)`」这一未组合点，设计已在 §9.1 风险 1 明确。
- 另核一处与 packed 相关的平台行为：`set_atomic` 的原子窗口在 AIV 段也会展开包住 `nd2nz_scatter`/`copy_ub2l1`（P2 设备码 `p2_src.cu:350-354`）——dense 真机已证明无害，但 §7.2 未记录该展开形态。

> 建议（非阻塞）：§7.2 增一行「`set_atomic` 的 codegen 展开范围覆盖 AIV 段」，并引用 dense 真机结论 + Stage 3 探针确认，避免 Stage 3 把它当成新现象重复排查。

### 5. 循环与同步 — pass（含 2 条建议）

- §6.1 分层明确：block 级并行（`T.Kernel(grid)`）、block 级标量串行（`T.serial(batch)`）、动态界迭代（`T.Pipelined(loop_st, loop_ed, num_stages=1)`）、tile 级 `T.gemm`、半区 `T.SimtVF(threads=128)`+`T.Parallel(32,64)`、tile 级 `T.dual_copy`、运行时分支写回；**首行即声明「无逐元素标量循环」**，与 §1.6.2 一致。
- §7 全自动同步，逐同步点（MTE2→MTE1/M、M→FIX、FIX→V 两次 arrive 覆盖两副块、V→MTE、MTE→MTE1/M、V→MTE3）与 P2 设备码的 `asc_sync_*` 结构对应；不写手工 flag、不引入跨 block barrier，与平台「未认证」边界一致。
- 尾块：§6.4 四类（GEMM/装载/掩码/写回）与 §3.3 注、§5.4 一致。

> 建议（非阻塞）：① §3.3 的 `fast` 谓词把因果项写成了无条件（`(n0+63 <= m0+shift)` 恒在），而 dense 已核验形态是 `if is_causal: fast = fast & (…)`（`_bwd_kernel.py:283-286`），且本设计 §5.4 第 7 条声称「`is_causal=False` 时所有块都进入快路」——两者不一致。**正确性不受影响**（`is_causal=False` 下该额外项只会更保守，掩码臂也本就等价），但建议按 dense 形态加 `if is_causal` 守卫，使 §5.4 第 7 条与 L0-3 的预期成立。② §5.4 第 6 条的成品式 `loop_ed = max(ceil(len_q/64), loop_st+1)` 并未被 P2 编译过（P2 设备码为 `kq < ((lq+63)>>6)`），建议注明「P2 编译的是纯 `ceildiv` 界，`max(… )` 守卫由 Stage 3 R7 验证」。

### 6. 验证方案 — pass（含 1 条建议）

- §8.1 golden：fp64、按 request 物化 `P`、自算 `lse`/`delta`、`nan_to_num(0)`、`slab_heads` 分片控内存、返回 `dq_acc_true=dq/scale`，与源仓 `mla_bwd_reference`/`ref_program` 语义对齐且**不复刻**分块算法 → 独立性成立（维度 0 已核）。
- §8.2 精度标准按 dtype 给 atol/rtol 并**实查到源**（fp16 `5e-3/1e-3`、bf16 `5e-3/1.6e-2`，`MLAVarlenBwdWorkload.verification` 的 `max(5e-3, atol)`）；`dq_acc` 因跨核原子不确定而**不做逐位比对**、atol 按 `1/scale` 反比放大到 ≈6.9e-2（复算 5e-3/0.0722 = 0.069 ✅）。
- §8.3 L0 门槛 12 例，覆盖了 conductor 关心的三点：**L0-9 request 间污染**（B=2 + 逐请求单独比对 + 输出追加哨兵行）、**L0-11 原子契约**（`dq_acc` 预填 2.0 与清零两跑，差值恒 2.0）、**L0-12 A2 两臂**（`a2_fast=False` 强制掩码臂对照，判等口径为 `dk/dv` 逐位 + `dq` ≤1e-5 —— 与 dense 实测「原子序导致 4.77e-7 差异」的经验一致，口径正确）；另有 L0-3 非因果、L0-4/5 shift 正负、L0-6 极小/非对齐、L0-7 单块、L0-8 最小、L0-10 双尾块。**7 个运行期前置探针 R1–R7** 齐备且各带判定与顺序（含 R7 `[0,64]/[64,64]` 的 `len_q=0` 退化防御）。

> 建议（非阻塞）：dense Stage 3 已实测到一条**继承性容差边界**——bf16 下 `dS` 强制 cast 到输入 dtype 再乘 `sum|terms|/|value| ≈ 17×` 的抵消放大，会使 `dk` 单元素误差达 7.8e-3 > atol 5e-3（dense L2-3 / B-4 的 WARN，已定位为固有量非缺陷）。本设计的数学与之逐字相同，建议在 §8.2 或 §9.1 记录该已知边界与判定手法（用 fp64 + 单独 cast `dS` 的仿真做二分），避免 Stage 3 重复诊断。

### 7. 完整性与一致性 — pass

- 章节齐全：§0.1–0.7 + §0.8、§1.1–1.6（含 1.6.0–1.6.3）、§2–§11；全文无占位符（「估算下界: 待核验」是常数缺失下的合规降级表述，不属占位）。
- 同类引用真实：逐条 Glob 核对全部存在（`examples/mla_dense_bwd/_bwd_kernel/{DESIGN.md,_bwd_kernel.py}`、`examples/TileOPs/tileops/kernels/attention/mla_varlen_bwd/_mla_varlen_bwd_kernels.py`、`examples/paged_cache_write/…/DESIGN.md`、`examples/flash_attention/`、`examples/mixcv/`、`knowledge/repro/PL-950-*.py`、`kernel_template.py`、`history_version/design_probe_compile.py`）。
- 内部一致：API 映射（§3.2）/内存规划（§4.3–4.6）/Tiling（§5.2–5.3）/循环（§6）四处的形状、层级、分块逐一自洽；§11.1/§11.2 的复用与差异索引能落到具体章节。

> 建议（非阻塞）：见维度 0 的两条路径/计数建议（`manifest/spec` 路径、第 4 个 workload），以及 §3.5.3 中 `_mla_varlen_bwd_kernels.py` 的「逐行同源」宜注明抽取日期/commit，便于日后核对漂移。

### 8. 算法优化分析 — pass

**§1.6 章节存在性**：1.6.0 / 1.6.1 / 1.6.2 / 1.6.3 四子节齐全，含调研四问。

**调研存在性与完整性**：R1 候选 13 行**含基线 #0（源 packed 算法）**，覆盖恒等变形（自然域 exp、块级 mask、行统计提升、`scale` 折入、`delta` rank-1、`dV/dK` 合并的不等价反例、物化、query-stationary、两 kernel 拆分、flat tile 不等价反例、pad 分块、host 预算块表）；R2 有**明确结论**「有，且基线即在线变体」并给结构依据（`lse`/`delta` 外部提供 ⇒ 无需 running max/重缩放）+ packed 专属在线结论；R3 **四口径齐全**（FLOPs / GM Bytes / 扫描遍数 / 中间缓冲峰值）且基线 C0 在表、口径写清含与不含；R4 逐候选六维（单元匹配/容量/对齐/静态边界/流水/跨核）。调研范围写明（`algorithm-candidates.md` attention 族、`knowledge/` 950 条目、`examples/` 三类实现、源仓 golden 与 roofline），无独立可查的遗漏。

**调研结论正确性（独立复核）**：
- 复杂度**独立复算**全部吻合：可见 score 对 `Σ l(l+1)/2 ×128 = 22.9e9`；FLOPs `22.9e9×1669 = 38.2 T`；分块迭代 `(36+136+528+2080+8256+32896)×128 = 5,623,296`；每迭代 L1 端口 `172032+24576+81920+752 = 279,280 B`（本 Agent 逐项相加一致）→ `24.5 FLOP/B`、`1.571 TB`；GM 视图 744 GB（q/do 双读 81,920 B×5.62M = 460.7 GB、`dq` 原子 49,152 B×5.62M = 276.4 GB、k/v 4.2 GB、dk/dv 2.6 GB）——**无中间缓冲 GM 往返漏计**（P/dS 全程片上，与源一致）。
- 常数缺失的处置**合规**：平台 `cost_measurements` 为空，设计未拿 910B 常数补 950，只给量级指示并明确「不作判据」，同时给出**结构结论**（L1 端口操作数流是地板项，24.5 FLOP/B）——与 `bottlenecks.md` 的纪律一致。
- **负向断言逐条复核**：① 「无更优在线结构」的依据是**交换性结构分析**（输出必有一组沿另一轴跨块归并）+ 两条已证否的候选（R1#6 合并不等价、R1#10 flat tile 跨 request 漏写），不是经验断言 → 成立；② 「源 `GroupTiling` 的 warp 扫描在 NPU 无对应」→ 已亲查源码（`grouped_tiling.py:141-150`）与平台文档（`kernel.py:4`）成立；③ 「`transpose_A` 在 L1 路径禁止」→ `gemm_mad.py:162` 断言成立；④ 候选库 attention 命中行（`ALG-attention`：scale 折入、flash 在线结构、materialize vs 分块、Cube+Vector 流水、L0C/UB 定 tile）**全部被评估**（分别对 R1#4、R2、C4、C1、§4.5/4.6），未漏候选；⑤ 互联网来源未使用且已声明范围，符合「本地源已覆盖该族结构」的纪律。
- **并列候选裁决路径**：C1 与备选 `block_n=128/block_m=32` 的判定裕度（−5.8% L1 端口 vs L0C 84.4%）依赖未实测常数 → §1.6.3 给出**主选 + 备选结构 + 裁决计划三件套**（代表 shape、`msopprof` 指标、阈值「备选优于主选 ×0.9 才翻转」、按 `len_kv` 分片分派的回写路径），达到实验裁决要求。

**调研 → 设计一致性**：§1.4 明写「算法族见 §1.6.0：key-stationary 单遍」；§3.1 九步以 §1.6.1 优化后公式为输入（步骤 3 = `exp(s^T·scale − lse[j]) ⊙ mask` 且注明 A2 运行时特化）；§6 只有 block/tile 级循环（单遍、无两遍扫描）→ **无「调研选 A 设计做 B」**。迁移侧 §0.5/§0.6 与调研同族，源 15 项优化手段逐项有承接或明确舍弃（无静默丢弃；#14 从 dense 的「舍弃」升级为本函数主体并在 §1.6.0 结论里点名承接）。

**§1.6.1 数学等价优化（四要素逐项核对 + 独立推演）**：
- A1（自然域 `exp`）：恒等式 `exp2(x·log2e) ≡ exp(x)` 成立；收益量化正确（省 `lse·log2e` 的逐元素重算 =4096 次/迭代；`exp`/`exp2` 同属 SFU，指令数不减）；**加重跑的 ulp 表两行均 EQUIV_PASS 且候选不劣于基线**（各自 vs fp64 ≤1 ulp）。
- A2（块级 mask 特化）：谓词是「块内掩码恒真」的**可判定充分条件**（对块内最大 `n=n0+63` 与最小 `m=m0` 成立即全真）——推演成立；115,696,896 元素、违反率 0、快路 48.6% 与本 Agent 重跑一致；另有快路安全性论证（快路块不可能含 `lse=-inf` 空行）。
- A3（溢出零行不变式）：**新增不变式而非优化项**的口径正确（它是整块原子写回的前提）；推演见维度 0 差异 ③，机器验证 30528 + 3816 全零、`acc=4102` 中仅 1 个 `-0.0→+0.0` 零号翻转（数值等价），且设计在 §9.1 风险 5 明确「掩码必须保持选择语义、不得改成 `p*mask`」——依赖关系已闭环。
- 「已确认无需改动项」的自查（`p*(dpt−delta)` 已最优、`p` 单次 `exp` 被 pT/dsT/ds 共用、`lse/delta` 行私有读、`dk` 的 scale 只在块尾、packed 新增标量每 block 一次）经推演均成立，无遗漏的明显可优化项（逐元素除法不存在、无归约可合并）。
- 弃选定量依据可核：#4 `scale` 折入需 12,288 次预乘 > 省下的 4096 次/迭代且要两份副本；#5 rank-1 拆分 +≈5.7% 计算量（192×64×64 MAC/迭代 ≈ +5.7% 复算吻合）；#6 合并不等价（会求和）；#10 flat tile 跨 request 漏写（源 docstring 明示）。

**§1.6.2 向量化替代完整性（与 §3.3/§6 交叉核对）**：14 个计算点全部列入；逐点与 §3.3 伪代码对照无遗漏（`P` 的 exp、掩码、`dS`+列读、三份 cast、转置写、块尾 cast、K 维归约、`dq` 原子、块枚举、边界标量、双分支谓词、逐 lane 守卫、head-major 统计装载、尾块 clamp）；#1–#8、#11–#14 已有向量/换单元/由平台 pass 消除的**API 佐证**（`PL-950-row-primitives`/`softmax-boundaries`/`attention`/`conversion-layout`/`atomic-splitk`/`mixed-epilogue`/`gemm-tiles` + dense 真机 + P2 设备码）；保留 3 项（#9、#10 与 #11 的块级谓词）的理由具体到本算子（block 级控制流与全部 GM 地址、必须对 AIC 与两 AIV 一致、kernel 作用域无线程域），并给了成本量化（≤B 次 int32 读 + ≤3B 整数运算 ≈ 10⁻³ 量级）——**非「照源码写」式空话**；`§6.1` 首行的「无逐元素标量循环」与 §1.6.2 结论一致。

**§1.6.3 向量化轴与布局决策**：按 MixCV 类别给出候选矩阵（候选 1 lane=query / 候选 2 lane=key / 候选 3 UB 双转置 / 候选 4 纯 SIMT / 候选 5 中间块 GM 往返 / 候选 6 `h` 外层展平 / 候选 7 `h` 内层展平），逐候选量化（整除性、累加链形态、repack 副本数、UB 影响、动态除数与 L2 复用），弃选均有量化理由（2 份转置 + 4096 次跨步写 + 4 KB、纯 Vector 差 1–2 量级、每迭代 3×8 KB GM 往返 ≈ +276 GB）；**未把 I/O 契约 THD 当作内部布局依据**（明确「`D` 维是 Cube 的矩阵维，向量轴只能在 key/query 中选」，是与 dense 相同的正确处置）；决策与 §3.3 的 `T.Parallel(32,64)`（内层 query）、§4.5 半区预算、§6 循环结构三方一致；候选 5 的弃选前提（L0C→UB 取代 910 的 `store_fixpipe` GM 往返）与 `PL-950-mixed-epilogue` 实测一致。

> 建议（非阻塞）：§1.6.2 末尾「14 个计算点中 11 项已向量化（#1–#8、#11–#14）」——括号内枚举实为 12 项，且 #11 又被计入「保留 3 项」的第二重计数。建议改为「12 项已向量化/换单元/被 pass 消除（#1–#8、#11–#14），保留 2 个纯标量点（#9、#10）与 #11 的块级谓词」。（结构结论与 §6 一致，仅计数口径需订正。）

---

## 备注（供 conductor / Stage 3）

- 本次检视**未执行任何真机 kernel**（卡被 engram fwd Stage 3 占用）；机械复核、等价性重跑与 5 个编译探针均在 `ASCEND_RT_VISIBLE_DEVICES=""` 下完成。
- **7 条建议全部非阻塞**，但其中 3 条建议在 Stage 3 开工前顺手处理以省返工：① `fast` 谓词补 `if is_causal` 守卫（与 dense 已核验形态与 §5.4 第 7 条对齐）；② §8.2/§9.1 记录 bf16 抵消放大的已知容差边界（dense L2-3 实测 7.8e-3 > 5e-3）；③ §6.3/§5.4 注明 `loop_ed` 的 `max(…, loop_st+1)` 守卫尚属未编译形态，由 Stage 3 R7 验证。
- 工件注入防护：DESIGN.md 与源码中未出现要求放行检视或跳过维度的指令性文本；检视结论仅依据源码、设备码与可复跑命令得出。
