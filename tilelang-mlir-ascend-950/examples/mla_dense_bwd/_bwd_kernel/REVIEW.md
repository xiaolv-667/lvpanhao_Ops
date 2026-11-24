# _bwd_kernel（MLADenseBwdOp 主反向）设计文档检视报告（revision v1 复检）

## 检视结论

结论: 通过

## 检视元信息
- 文档: examples/mla_dense_bwd/_bwd_kernel/DESIGN.md（revision v1，1457 行）
- 上一版: examples/mla_dense_bwd/_bwd_kernel/history_version/design_v0.md（v0，检视不通过）
- 任务类型: 迁移（harness 子模式，源 `bwd.py` 的 `_bwd_kernel` 工厂）
- 源算子: /workspace/TileOPs/src/tileops/_kernels/attention/mla/bwd.py（`_bwd_kernel` 109–264；host `_run` 338–347；`MLADenseBwdKernel.forward` 361–379）
- 平台: ascend-950pr-9579 / ascend（dav-3510，CANN 9.2.1，TileLang ascend 0.1.15，28 Cube / 56 Vector；`design_budget.capacity_bytes`：UB 253952 / L1 524288 / L0C 262144 / L0A-L0B 各 65536）
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 2（均为 §3.5.4/§1.6.2/§7.2 的设备码**计数/编号**与本次重跑略有出入，不影响任何结构性结论）

## 机械复核（D-1/D-5）

**verify_equiv 重跑（本轮，2026-11-23，`ASCEND_RT_VISIBLE_DEVICES=""`）**：
`/workspace/venv-ascend950/bin/python examples/mla_dense_bwd/_bwd_kernel/verify_equiv.py` → `EQUIV_SUMMARY ALL_PASS`（exit 0），三行与 §1.6.1 内嵌表**逐字一致**：

| 采纳项 | 样本数 | 候选 vs 基线 最大 ulp | 违反率 | 候选 vs fp64 | 基线 vs fp64 | 结论 |
|---|---|---|---|---|---|---|
| A1 exp 域 P（float16） | 1048592 | 1 | 0.000e+00 | 1 | 1 | EQUIV_PASS |
| A1 exp 域 P（bfloat16） | 1048592 | 1 | 0.000e+00 | 1 | 1 | EQUIV_PASS |
| A2 块级 mask 特化（64/64） | 76838912 | 0 | 0.000e+00 | — | — | EQUIV_PASS |

`A2 fast-path blocks: 9285/18761` 与 §1.6.1 表注、R1#2 的「49.5%」一致。脚本结构已独立核对：候选/基线/ fp64 三路同口径比较（非自证），A2 为**全块逐元素**比对（非抽样），含 `lse=-inf`、±700、次正规、`S_q<S_kv`（(96,192)）与 `S_kv<S_q`（(128,64)）角点。

**design_calc_check**（`--task-dir examples/mla_dense_bwd/_bwd_kernel`；传给 `examples/mla_dense_bwd` 会因父目录存在平台戳不一致的 `.stage_state.json` 而报 `selected task platform differs`，按算子目录传入后正常）：

| check | status | detail | 人工复核结论 |
|---|---|---|---|
| `ub_budget` | **pass** | "UB: 10 buffer entries sum = 176.5 KB; capacity = 248.0 KB" | 与 §4.5 逐项复算一致：`st_u/dp_u 2×16384 + p16_u/dsT_u/ds_u 3×8192 + lse/dlt 512 + tail_u 49152 + tailv_u 32768 + out_k 24576 + out_v 16384 = 180736 B = 176.5 KiB = 71.2%` |
| `l0c_budget` | pass | "block 64x64 x fp32 accum = 16.0 KB vs L0C 256 KB" | 该检查只覆盖单累加器；5 个累加器总量人工复算 `49152+32768+16384+16384+49152 = 163840 B = 160 KiB = 62.5%`，与 §4.6 一致 |
| `core_split` | **skip** | "insufficient parsed dims … declared logical cores = None" | 人工复核（机械跳过，人工已核）：`H·B·ceil(S_kv/64)` = smoke 8·1·8=64、ds-v3-4k 128·1·64=8192、ds-v3-b2-8k 128·2·128=32768、kimi-k2-8k 64·1·128=8192、ds-v2-lite-3k 16·4·48=3072，五条**全部与 §5.5① 一致**；物理核 28 与 `statectl platform` 落库的 `core_source`（`ai_core_cnt=28, cube=28, vector=56; rtGetAiCoreCount()=28`）一致；64/28=2.29 波 |
| `r3_metrics` | pass | "all four R3 metrics present" | 见维度 8 的独立算术复算 |

**编译探针重跑（本轮，无设备，`cd /tmp/opencode/rv2` + `ASCEND_RT_VISIBLE_DEVICES=""`）**：
`history_version/design_probe_compile.py` → **3/3 COMPILE_OK**，且逐条与 §3.5.4 的设备码结论吻合：

| 探针 | 本轮实测 | 与文档一致性 |
|---|---|---|
| A `A_primary` | `COMPILE_OK source_bytes=13816 gemm_l1=5 l0c2ub=4 nd2nz_scatter=3`；设备码 `__mix__(1, 2)`；AIV 段 `int32_t sid = asc_get_sub_block_id();`（合成后**未被使用**）；`asc_set_atomic_add_float()` → `asc_copy_l0c2gm(...)` → `asc_set_atomic_none()` 相邻；两次 `asc_sync_intra_arrive(PIPE_FIX, ·)` 相隔 16；`asc_copy_l0c2ub(..., dual_dst_mode=0)`；L1 偏移上界 `81920 half` = 172032 B、L0C 上界 `36864+16384 float` = 163840 B、UB 最大 `86208 half` = 172416 B | 与 §3.5.4/§4.5/§4.6/**§7.2** 一致（源字节、5/4/3 计数、原子窗口、双 arrive、dual_dst_mode=0、UB 上界全部复现） |
| B `B_tail_oob` | `COMPILE_OK source_bytes=5781`；`asc_copy_gm2l1_nd2nz(..., min(64, max(96 - kq*64, 0)), ...)`、`asc_copy_gm2l1_dn2nz(..., 192, min(64, max(96-kq*64,0)))`、`asc_copy_ub2gm_align(..., min(64, (96 - n0*64)), ...)`；`asc_fill_l1(...)` **4 处**；GEMM 形参直接是运行期布尔 `(kq == (block_idx & 1))` | clamp/动态 `clear_accum` 与 §5.4/§6.4/§9.1#7 一致；`asc_fill_l1` 计数为 **4**（文档 §3.5.4/§1.6.2 写 3）→ 建议级问题 1 |
| C `C_a2_branch` | `COMPILE_OK source_bytes=4254`；两个 `__launch_bounds__(128)` 的 epilogue VF + 一个 `dim3(1)` 的辅助 VF，运行时 `if (...)` 二选一 | 与 §1.6.1 A2 机制、§6.4「运行时块级分支」一致（文档写「两个 SIMT VF」，实测另有一 `dim3(1)` 辅助函数，不影响结论） |

**结论**：上一版 2 个阻塞项（动态 GEMM region extent、显式 `T.Vector`/`sid`）在**形态层面**已闭环并有本机可复现证据；机械复核全 pass（1 项 skip 已人工核）。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

**语义核对（源码逐行，证据 = 源语句 + DESIGN 章节）**：
- 公式：`bwd.py:226` `T.gemm(k_s, q_s, st, transpose_B=True)` → §0.1/§0.3 S6 的 `s^T = k q^T` ✓；`228` `gemm(v_s, do_s)` → `dP^T` ✓；`235` `exp2(st*scale_log2 - lse_s[j]*LOG2E)` → §0.1 的 `P=exp(scale·qk−lse)`（`lse` 为**自然对数**，已由 `workloads/attention/mla.py:410-418` 的 `scores.logsumexp` 佐证）✓；`247` `ds16 = st*(dpt-delta_s)` ✓；`245` `gemm(p16, do_s, dv_f)` = `dV=Pᵀ dO` ✓；`248` `gemm(ds16, q_s, dk_f)` = `dK=Σ dS q` ✓；`251` `gemm(ds_s, k_s, dq_f, transpose_A=True)` = `dQ=Σ dS k` ✓。
- I/O：`bwd.py:141-154` 的 11 个张量（`q,k,v,do,lse,delta,cu_q,cu_kv,dq_acc,dk,dv`）与 `134` 的 `stat_shape=(batch,heads,dense_seqlen_q)` 全部出现在 §0.2；`o` 不在 kernel 签名内而由 `_run:341` 的 `prep(o, do)` 消费，§0.2 该行标注正确 ✓。
- 规约语义：`200-201` `T.clear(dk_f/dv_f)` + `204` 沿 `kq` 升序累加 → §0.1「`dV/dK` 沿 query 块升序 fp32 累加」✓；`254` 逐元素 `T.atomic_add` → §0.1「`dQ` 跨核顺序不确定、无逐位可复现契约」✓。
- dtype 语义：`122` `accum="float32"`、`166-170` `p16/ds16` 为输入 dtype、`257` `cast(dk_f*scale)`、`260` `cast(dv_f)`、`101`（`_dq_land_kernel`）才乘 `scale` → §0.1 dtype 表 6 行逐条对应 ✓；「本 kernel 不乘 `dq` 的 `scale`」与 §0.1 host 侧语义一致 ✓。
- 边界语义：`193-199` 越界补 0、`231-236` `if_then_else` 掩码（非乘掩码）、`216` `lse_s` 越界填 0 → §0.1(a)(b)(c) ✓；「空块不存在」的 bottom-right 论证我独立复算成立（最末 key 块 `loop_st = (n0−shift)//64 ≤ (S_q−64)//64 < loop_ed`，含 `S<64`、`S_q≠S_kv` 两种情形均非空，故 `clear_accum=(kq==loop_st)` 不会漏清零）✓。

**算法步骤覆盖**：§0.3 的 S1–S16 与 `bwd.py:176-260` 逐段对应（块解码 177-179、装载 192-199、清零 200-201、循环界 202-203、query 装载 206-224、5 GEMM 226-251、掩码 229-243、原子 252-254、写回 255-260），**无遗漏、无臆造**；host 侧 `view`/`arange`/`zeros`（363-379、342）已列 ✓。

**优化手段覆盖**：§0.4 14 项逐条可溯源（`Pipelined(num_stages=2)` 204/336、`GemmWarpPolicy.FullRow` 226/245/248、`threads=256` 155、`T.atomic_add` 254、key-stationary 156-201、recompute 不物化 `P/dS`、因果剪枝 202、packed 分支 133-139/180-184）。v0 的归属错误（把 `clear_accum` 记为源码手段）已修正为「NPU 适配构造」并给源码定位（`bwd.py:200-201,225,227,250` 只有 `T.clear`）✓。

**耦合性判定**：§0.5 18 行四态处置与源码/平台事实逐条抽检不矛盾：`T.Kernel` 一维（`kernel.py`）、L1-GEMM 只 NT（`ascend/op/gemm/gemm_mad.py:162` 原文 `assert not self.trans_A and self.trans_B, "Ascend L1 GEMM currently only supports trans_A=False, trans_B=True (NT)."`）、C 必须带 Ascend fractal layout（`gemm_mad.py:149`）、`dual_copy` 2:1 与 dst 单例中轴拒绝（`copy_op.py:107-134`，本轮重跑反例原文一致）、`policy` 不入 lowering（`gemm_mad.py` 全文无 policy 读取，仅 `gemm_op.py:144,210` 透传）✓；舍弃项（`FullRow`、packed 分支）均写明理由 ✓。

**重设计可行性（R1–R5）**：R1 一维网格 + div/mod 与 `(tile,h)` 双射 ✓；R2「L0C 是唯一可累加层」与 `gemm_mad.py:149` 断言一致，累加顺序/dtype 变化仅限 mad 的 K 维分组且已说明「只承诺 fp32 累加而非逐位一致」✓；R3 的 5 个 NT 改写我逐项重推：`k·qᵀ`、`v·doᵀ`、`pT·doTᵀ`、`dsT·qTᵀ`、`ds·kTᵀ` 的元素式与源 GEMM **逐元素恒等**（且与 §3.3 伪代码的 `transpose_B=True` 组合形状互洽：K 维分别 192/128/64/64/64）✓；R4 整 tile lane 映射的语义保持（逐元素、无跨 lane 归约）成立，实现形态（自动合成 `sid`、两 AIV 冗余）由探针 A 设备码佐证 ✓；R5 store-atomic 整块写与 `PL-950-atomic-splitk.py` 已核验形态一致 ✓。

**§1–§7 与 §0.5/§0.6 决策一致性**：§1.4 四步、§2.3、§3.1/§3.2/§3.3、§4.3、§6.1/§6.2 全部描述的是 R1–R5 之后的形态（全 extent GEMM + 整 tile epilogue + NT-only + store-atomic + 一维网格），未出现「§0 判重设计、§3 照抄源方案」；§3.3 伪代码与 §0.6 的 5 个乘积表逐项同名同形 ✓。

**golden 独立性**：§8.1 只读 `q,k,v,do`，`lse` 与 `delta` 由它自己重算（`mla.py:441-460` 的 `mla_bwd_reference` 同法），不消费输入 `lse/delta`、不物化分块、无原子；与 `ref_program`（`mla.py:506-518`）「fp32 梯度只舍入一次」口径一致 ✓；额外返回 `dq_acc_true = dq/scale` 与本 kernel 的 fp32 中间量对齐 ✓。**不复刻** §0.6 的 NPU 数据流 ✓。

判定：pass（v0 的 `clear_accum` 归属 warn 已消除，无新增偏差）。

### 1. API 可行性 — pass

§3.2/§3.4 的每条 API 本轮均由**可复现编译探针**落地（`tilelang 0.1.15` 实机工具链）：`ascend_gemm_l1`（5 处，含运行期布尔 `clear_accum`）、`asc_copy_l0c2ub`（4 处，整 tile 同 dtype fp32）、`ascend_nd2nz_scatter`（3 处 = `pT_s/dsT_s/ds_s`）、`asc_copy_l0c2gm`（位于 `asc_set_atomic_add_float`/`asc_set_atomic_none` 之间）、`asc_vf_call`（两段 128 线程 epilogue）、`ascopy_gm2l1_nd2nz/dn2nz`（含 `transpose=True` 的 `kT_s/qT_s/doT_s`）、`asc_copy_gm2ub_align`（`lse/delta` 1-D 区段）、动态界 `for (kq = <运行时下界>; ...)`。源码级佐证同步核对：`gemm_op.py:62` 的 `T.gemm requires static tile dimensions`（v0 阻塞 1 的根因）、`gemm_mad.py:162` 的 NT 断言、`gemm_op.py:27-45` 的 `_l0_gemm_call` 注释（L0 路径读 allocation 形状）、`copy_op.py:132` 的 dual_copy 分片推断报错、`pipeline.py:88-101` 的 `InsertNd2Nz` + `AscendInsertOOBPadding`。

两条上一版不可编译形态**已删除**且新形态已编译：全 extent GEMM（探针 A/B）、无 `T.Vector`/`sid` 的整 tile epilogue（探针 A/C）。§3.4 表对未核验项（`T.annotate_buffer_versions`、动态界 + `num_stages>1`）如实标注「未核验」并定量到首版取值（`num_stages=1`），不构成阻塞。

### 2. 内存层级规划 — pass

§4.3 表覆盖全部 25 个 buffer 且逐项标注层级；§4.4 搬运路径逐级完整（GM→L1/L0→L0C→UB→L1→GM，L0C→GM 原子）。预算**独立复算全部通过**（机械 `ub_budget` pass，`l0c_budget` pass）：
- UB 180736 B = 176.5 KiB = **71.2%**（每迭代活跃 57856 B、块尾 122880 B；不别名上界）；
- L1 172032 B = 168 KiB = **32.8%**，Stage 4 双版本 278528 B = 53.1%（`7 个每迭代缓冲 ×2 = 106496` 复算一致）；
- L0C 163840 B = 160 KiB = **62.5%**，单累加器上限 49152 B ≤ 262144 B；
- 备选 128/32：L0C 221184 B（84.4%）、L1 196608 B（37.5%）、UB 180480 B（71.1%，块尾必须分两段）——三处逐项复算一致。
**强佐证**：探针 A 设备码的实际分配与预算表**数值吻合**（L1 上界 172032 B、L0C 上界 163840 B、UB 上界 172416 B ≤ 180736 B 保守上界），说明表不是纸面数字。

### 3. Tiling 策略 — pass

block 划分（§5.2 `block_n=block_m=64`）理由与 L0C 主项（`dk_c/dv_c` 80 KiB）自洽；对齐（192=12×16、128=8×16、`dq_acc` 尾轴 768 B）成立；**非整除处理已统一为单一原则**（§3.3 注 2 / §5.4 / §6.4 三处同口径：GEMM 全 extent + OOB clamp/fill + 掩码收口），消除了 v0 的「§5.4 截断 vs §6.4 编译期判定」矛盾；尾块代价（多算一个满 tile，≤ 1/ceil(S/64)）已量化。分核三要素齐全：① 五条 workload 逻辑核数（64/8192/32768/8192/3072）复算一致；② 物理核数取平台已落库实查记录（`core_source`），并写明本卡被占用时 Stage 3 首跑前用 `torch.npu.get_device_properties` 复核——符合「不凭假设」；③ 极大规模判定 + 「逻辑块超发」方案 + **显式标注偏离 `core-split-strategy.md` §1** + 3 条复评触发条件（v0 建议 8 已闭环）。

### 4. 技术约束检测 — pass

§3.5.1 的 11 行限制检查逐条对照安装模块成立，且 v0 的错误行已改正：现行为「**L1-GEMM 的 M/N/K 取自 region extent，必须编译期静态**……GEMM 一律用 allocation 全 extent，动态 extent 只出现在 copy」（`gemm_op.py:62` 原文佐证）✓；L0C 唯一累加层 ✓；L0C→UB 不做 dtype 转换（探针 A 用 fp32 同 dtype 通过）✓；线程域只在 `T.SimtVF` ✓；不写手工 flag/跨 block barrier ✓；动态 shape 全静态（dense 工厂按 shape 编译）✓。§3.5.4 的 3 条反例（静态 extent ValueError、`dup sid` InternalError、dual_copy 分片 ValueError）本轮重跑与文档原文一致，且被写入 §9.2 常见错误表——**「为何不会再犯」的机制性防线成立**（v1 规定任何 GEMM 形态改动必须通过探针 A 形态编译）。

### 5. 循环与同步 — pass

循环结构（§6.1）分层清晰：block 级并行（一维静态 grid）、动态界 `T.Pipelined(loop_st, loop_ed, num_stages=1)`、tile 级 GEMM、整 tile 向量段、块尾 cast —— 与 §1.6.2 的「无逐元素标量循环」结论一致（§6.1 首行即声明该不变量）。§6.3 首版钉死单版本并给出 Stage 4 只展开 L1 的量化理由（UB 双版本 94.0% 不可行）；§6.4 尾块口径与 §3.3/§5.4 一致。§7 全自动同步（`AutoSchedule→AssignCore→PrepareMultiBuffer→ResolveCore→InsertSync`），设备码中出现 `asc_sync_notify/wait(PIPE_M, PIPE_FIX)`、两次 `asc_sync_intra_arrive(PIPE_FIX, e/e+16)` 与 AIV 侧 `asc_sync_intra_wait`、`asc_sync_intra_arrive(PIPE_MTE3, ·)` → `asc_sync_intra_wait(PIPE_MTE1, ·)`，与 §7.2 的同步点表逐行对应 ✓。P1 的「双 AIV 可见性」是运行期未知量，已在 §8.3 立为**第一个** Stage 3 探针并给出可编译回退，属正常的运行期验证排程，不是设计缺陷（详见下节）。

### 6. 验证方案 — pass

§8.1 golden 为 fp64 物化参考（独立于分块/原子/转置），与源 `mla_bwd_reference`（`mla.py:441-460`：recompute probs、`nan_to_num(0)` 处理 `-inf` 行）与 `ref_program`（`mla.py:506-518`：fp32 梯度只舍入一次）逐项对齐；§8.2 容差**实查有据**：`workloads/numerics.py:273-283` 给出 fp16 rtol 1e-3、bf16 rtol 1.6e-2，`MLADenseBwdWorkload.verification`（`mla.py:522-532`）取 `atol=max(5e-3, tol)` —— 与表中三行完全一致，并显式声明 `dq_acc` 因跨核原子顺序不确定而不做逐位比对 ✓。§8.3 的 L0-1…L0-7 覆盖 smoke、非因果、双尾块（96）、`shift>0`（96/192）、`shift<0`（128/64 的 `lse=-inf` 行）、单块单迭代、非 16 倍数边长（65），并新增 P1–P6 六个运行期前置探针（每个都有判定标准与回退），实施顺序明确 ✓。

### 7. 完整性与一致性 — warn

§0–§11 章节齐全（§0.1–0.7 完整，新增 §11 闭环表），全文**无占位符**（三类占位标记 0 命中；§1.6.0「估算下界: 待核验」是 D-2 要求的显式常量缺失标注，不是未完成项）；§3.5.3/§1.6.0 引用的 16 条路径经 Glob **全部存在**（含 `examples/flash_attention/flash_attn_npuir.py`、7 条 `knowledge/repro/PL-950-*`、`kernel_template.py`、`paged_cache_write` 设计范式）；内部一致（API 映射 / 内存规划 / Tiling / 伪代码 / 循环结构互不矛盾）；与 manifest roofline 公式一致（`formulas.py:644` 的 `per_score = 2(3·D_qk+2·D_v)+5 = 1669`，与 §1.6.0 逐字相同）。**warn 原因（2 条建议级）**：
1. §3.5.4 探针 B 行与 §1.6.2 #10 写「`asc_fill_l1(...)` ×3」，本轮重跑为 **4 处**（探针 B 装载 `k_s/v_s/q_s/qT_s` 四个张量，各需一次尾部补 0）。结构性结论（clamp + fill 存在）不变，但计数应与可复跑输出对齐。
2. §7.2 写「两次 arrive …… 探针 A 实测为 `1/17`」，本轮重跑为 **3/19 与 4/20**（事件 id 由调度器分配，随编译而变）。建议改为「形如 `e`/`e+16`，id 由调度器分配（本轮实测 3/19、4/20）」，避免下一轮把可变量当固定证据。

### 8. 算法优化分析 — pass

**§1.6 四子节齐全**（1.6.0/1.6.1/1.6.2/1.6.3）。

**调研存在性与完整性（独立复核）**：R1 候选表 10 行**含基线**（#0 源算法），覆盖恒等变形（自然域 exp、块级 mask、行统计提升、`scale` 折入、rank-1 拆分、合并 GEMM 的不等价反例、物化、query-stationary、两 kernel 拆分）；R2 **明确结论**「有，且基线即在线变体」，结构依据（`lse/delta` 由外部提供 → 无需 running max/重缩放）与源码 `bwd.py:226-251` 一致（源码确实不物化 `P/dS`，即已是流式单遍），且「无更优在线结构」附交换性论证（必有一个输出跨块归并）；R3 四口径齐全且**口径说明写清含/不含项**（不含 L1/UB 内部搬运，另表单列）；R4 逐候选六维评估。对 `knowledge/algorithm-candidates.md` 的 attention 命中行（仅「单头 SIMT QKᵀ/softmax/PV 前向」一条）逐条比对：本设计的候选覆盖**更广**，未发现被漏掉的已知候选（Welford 不适用——本算子不计算归一化统计量；`delta` 融合属跨 launch 结构，受三 launch 契约约束）。**负向断言逐条复核通过**（无「与参考表/源码矛盾」项）。

**调研结论正确性（独立推演 + 复算）**：
- 等价性：A1 是 `exp2(x·log2e) ≡ exp(x)` 恒等，机器复跑 1 ulp、违反率 0，且**候选 vs fp64 = 基线 vs fp64 = 1**（候选不劣于基线，非「仅搬移舍入」）；A2 的充分条件我独立推演：快路要求 `n0+BN ≤ S_kv ∧ m0+BM ≤ S_Q ∧ n0+BN−1 ≤ m0+shift`，则由 `i ≤ BN−1`、`j ≥ 0` 得 `n0+i ≤ n0+BN−1 ≤ m0+shift ≤ m0+j+shift`（掩码恒真），同时无填充行/列（故省 `gi< S_kv`/`gj< S_Q` 也安全），且这样的块**不可能**含 `lse=-inf` 行（每 query 至少见 BN 个 key）→ 不产生 inf/NaN；机器验证 6 组 shape 全覆盖 0 违反、快路 9285/18761。
- 复杂度算术复算（逐项与设计一致）：FLOPs `256×(33.55e6×1669 + 8192×256) = 14.34 T` ✓；迭代数 `256×128·129/2 = 2 113 536` ✓（与「每 (b,h) 8256 次」一致），分块量化开销 `2 113 536×4096/8.59e9 − 1 = 0.78%` ✓；GM 视图 `173.1 + 2.15 + 103.9 + 1.34 = 280.5 GB`（q/do 双份 81920 B/迭代、`dq` 原子 49152 B/迭代、`dk/dv` 按 bf16 输出 1.34 GB）✓；每迭代 L1 端口 `172032 + 24576 + 81920 + 1016 = 279544 B`、`6 852 608/279 544 = 24.5 FLOP/B`、全量 `590.6 GB` ✓ —— v0 不可复现的「248 KB/27」已修好；备选 128/32 独立复算：迭代 `256×64×(256−4kn)` 累加 = `2 129 920` ✓、L1 端口 `263164 B`（GM→L1 40960 + 操作数 196608 + UB→L1 24576 + 常驻摊销 ≈1020）✓、合计 `560.5 GB`、相对主选 **−5.1%** ✓、L0C 84.4% / L1 37.5% / UB 71.1% ✓。**中间缓冲未漏计**：C1 的上界含 L0C+L1+UB 三项，C4（物化）单列 `O(B·H·S²) ≈ 34 TB`（`2×128×8192²×2 B` 复算量级一致）并给出容量淘汰，未把 GM 往返藏在「片上峰值」里 ✓。
- 亲和性淘汰：C2/C3/C4/C5 的淘汰理由与 `programming.md`/`ascend-constraints` 不矛盾；容量类引用核对无误（`CONST-L1-port-bw`/`CONST-fabric-bw` 的 `device` 确为 `910B2C`，设计明确标注「不适用本型号，仅作量级参考」）；`CONST-store-fixpipe-gm-only`（910B「仅 L0C→GM」）被 950 的 L0C→UB 取代，有 `knowledge/repro/PL-950-mixed-epilogue.py:22-33`（`T.dual_copy(c0, cv)`）作 950 侧佐证 ✓。
- 并列候选裁决：主选/备选裕度依赖未实测常数（L1 端口带宽、Cube 吞吐），三件套齐全（主选 64/64；备选 128/32 含 L0C/L1/UB 逐项预算 + 块尾两段写回 + 核数影响；阈值「备选优于主选 ×0.9 才翻转」+ 指标/反解方法/回写路径 + 按 `S_kv` 分派可行性）✓ —— 满足「实验裁决模式」要求。

**调研→设计一致性**：§1.4 明确「算法族见 §1.6.0：key-stationary 单遍」，§3.1 以 §1.6.1 优化后公式为输入（`P=exp(s·scale−lse)` + 块级特化），§6 循环结构只有 block/tile 级（单遍、无两遍扫描）→ 无「选 A 做 B」；迁移维度上 §0.5/§0.6 与调研结论同族，源 14 项优化手段**逐项有承接或明确舍弃理由**（§0.5 处置列 + §1.6.0 末段声明），无静默丢弃 ✓。

**数学等价优化完整性**：A1/A2 各有完整四要素（原式 → 优化后 → 等价论证 → 收益量化：A1 每 score 少 1 次乘 = 4096/迭代；A2 快路 49.5% 块 × 每 score 省 2 比较 + 1 选择）；「已确认无需改动」项（`dS=p·(dpt−delta)` 无 CSE 空间、`p` 被 `pT/dsT` 共用、`dk` 的 `scale` 只在块尾）我逐条核对成立；弃选项定量（`scale` 折入 12288 次 vs 4096、rank-1 拆分 +5.7% MAC）复算一致 ✓。

**向量化替代完整性**：§1.6.2 盘点 11 个点，与 §3.3 伪代码、§6 循环结构**交叉核对无遗漏**（`exp/掩码/dS/三份 cast/转置写/dk cast/行统计/GEMM 换单元/store-atomic/尾块 clamp/任务索引` 一一对应）；10 项已替代或换单元/由平台 pass 消除，仅 #9（任务索引 `h/b/n0/loop_st/shift`）保留为标量，理由具体到「每 block 一次的 block 级映射，无逐元素等价 API」✓；v1 已删除上一版的 #12（`sid` 半区偏移）条目，§1.6.3 与 §4.5 同步改为整 tile ✓。替代 API 均有探针/已核验条目佐证 ✓。

**向量化轴与布局决策**：§1.6.3 给出 5 候选的量化矩阵（整除性/尾 lane/累加链形态/repack 代价/UB 影响），类别内必选候选（lane=key vs lane=query 两种 lane 映射、纯 SIMT、GM 往返）均已枚举；弃选有量化理由（lane=key 需 2 份转置副本 + 每元素一次列广播；纯 SIMT 差 1–2 数量级；GM 往返 +103 GB）；**未把 I/O 契约当内部布局依据**（明确「BSHD 是契约，布局自由度在 key×query 2-D 块」）；弃选代价机制经本人核对（`CONST-store-fixpipe-gm-only` 的 910 前提 vs 950 实测条目）✓。布局决策与 §3.3（`T.Parallel(64,64)` 内层 query）、§4.5（整 tile 预算）、§6（整 tile 向量段）**三方一致** ✓。

**与下游章节一致性**：§3.1 八步全部以优化后公式为输入（步骤 3 = `exp(s·scale − lse) ⊙ mask` 且注明 A2 运行时特化）；§6.1 明确「无逐元素标量循环」，与 §1.6.2 结论一致 ✓。

## Stage 3 前置探针与遗留风险（非阻塞，供 Stage 3 排程）

以下 6 项在设计中**已登记为运行期未知量**，并各自具备「判定标准 + 可编译回退」，故不构成设计阻塞（本检视不把它们当阻塞项判 fail）：

| # | 风险 | 为何可留到 Stage 3 | 本检视补充 |
|---|---|---|---|
| P1 | 整 tile `L0C→UB` 的双 AIV 可见性 | 设备码 `asc_copy_l0c2ub(..., dual_dst_mode=0)` + 两次 `asc_sync_intra_arrive(PIPE_FIX, e/e+16)` 表明编译器的调度模型**假定两个副块都消费同一次写**；且两 AIV 的 UB 偏移完全一致（`buf_dyn_shmem` 同一组地址）；回退（`dual_copy` 2:1 + 静态半区）可编译性单独验证 | 建议 P1 同时观察「AIV 段 `set_atomic_add` 窗口覆盖 UB→L1 拷贝」是否有副作用（设备码中 AIV 每迭代有 `asc_set_atomic_add_float()` … `asc_set_atomic_none()`，窗口内含 3 次 `scatter`/`copy_ub2l1`；若发现 L1 写异常，最小修法是把原子开关限制在 AIC 侧或改 `set_atomic` 为显式 store 属性） |
| P2/P3 | OOB 补 0 不泄漏进梯度 | 探针 B 已证 clamp + `asc_fill_l1` 生成；「补 0 ≠ 梯度 0」已写进 §5.4/§9.2 并用掩码收口；判定用哨兵值 | — |
| P4 | 动态 `clear_accum=(kq==loop_st)` 首块清零 | 编译级已验证运行期布尔透传；判定为与「提出首迭代 + 显式 `T.clear`」逐元素对照 | — |
| P5 | store-atomic 的 host 预清零契约 | 与 `PL-950-atomic-splitk` 同形态；`_run:342` 已 `torch.zeros` | — |
| P6 | A2 两臂一致 | 快路充分条件已证明且机器验证 0 违反；P6 是端到端复查 | — |
| #1/#2/#4/#5 | L1-GEMM 运行期正确性、多阶段混合未认证、跨步 region、UB 双版本不可行 | 均有首版规避（`num_stages=1`、单版本、全 extent）与量化回退（L0 路径 L0A 53248 B / L0B 61440 B = 93.8%，余量 4 KiB，仅单版本） | L0 回退的 93.8% 复算一致，实施时须逐 buffer 复用，不得再加缓冲 |

另：§3.4 标「未核验」的 `T.annotate_buffer_versions`（含对 L0C buffer 声明 1 版本）若在 Stage 3 被前端拒绝，直接删除该调用即可（默认单版本），不影响正确性。
