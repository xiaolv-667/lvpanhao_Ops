# paged_cache_write 设计文档检视报告（revision 1 复审）

## 检视结论

结论: 通过

## 检视元信息
- 文档: `examples/paged_cache_write/_paged_cache_write_kernel/DESIGN.md`（含 `修订记录 — revision 1`，第 11 行）
- 轮次: 第 2 轮（针对 revision 1；上一轮判定为不通过，阻塞 2 项 / 建议 9 项，本轮均已闭环）
- 任务类型: 迁移（TileOPs GPU `PagedKVCacheWriteFwdOp` → Ascend950PR_9579 / `target="ascend"`，harness 子模式）
- 源算子: `/home/l00970450/advanced-compiler/team/lvpanhao/xiangmu/TileOPs/src/tileops/_kernels/attention/paged_cache_write.py`（本轮重读全文 226 行）
- 检视维度: 9 项（0 源算子理解与迁移分析 / 1 API / 2 内存层级 / 3 Tiling / 4 技术约束 / 5 循环同步 / 6 验证方案 / 7 完整性 / 8 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 11（编号沿用上一轮 3–11，新增 12–13；均不影响结论）
- 检视方式: 静态只读核查 + CPU-only 脚本；未编译、未运行 kernel、未改 `ASCEND_RT_VISIBLE_DEVICES`；未修改 `DESIGN.md` / 源仓 / Stage 0 产物

## 机械复核（D-1/D-5）

1. `verify_equiv.py` 重跑: **PASS**。命令 `/home/l00970450/zuo/.environment/venv/bin/python examples/paged_cache_write/_paged_cache_write_kernel/verify_equiv.py`（torch 2.10.0+cpu，纯 CPU，无设备参与），收尾行 `adopted items: 8, failed: 0` / `ADOPTED_VERDICT: ALL_PASS`；8 个采纳项（O1 ×4：bf16/fp16 × scale 1.25/0.75；O2 ×2；O3 int32 域全枚举 + guard；G1 ×4）与 `DESIGN.md#1.6.1` 内嵌结果表逐行一致（违反数 0、违反率 0.000e+00、最大 ulp 差 0.000）。被否决项数字与 `#11.1` 一致：X1 `scale=7.0` → 60/65536（bf16 与 fp16 同值）、`1.7` → 30/65536、`0.3` → 2/65536；X2 bf16 → 127/65536、fp16 → 0；X3 bf16 4/16、fp16 2/16。
2. `design_calc_check`: `python3 .agents/tools/design_calc_check.py --design examples/paged_cache_write/_paged_cache_write_kernel/DESIGN.md --task-dir examples/paged_cache_write/_paged_cache_write_kernel`，`checks` 摘要：
   - `ub_budget` **pass** — "UB: 4 buffer entries sum = 12.0 KB (declared 合计 12.0 KB, recompute matches: True); capacity = 248.0 KB"
   - `l0c_budget` **skip** — "no Ascend Cube GEMM call found in the design pseudocode"；**人工复核**：本算子无 `T.gemm`、不分配 L0A/L0B/L0C（`#3.5.1`/`#5.1`/`#5.3`），L0C 预算不适用，skip 符合预期。
   - `core_split` **pass** — "recomputed ceil(8192/16) x ceil(1024/128) = 4096 vs declared 4096"
   - `r3_metrics` **pass** — "all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak)"
3. `statectl gate 1 --dir examples/paged_cache_write/_paged_cache_write_kernel`: `{"action": "gate", "failures": [], "ok": true, "pass": true, "stage": 1, "warnings": []}`

## 上轮阻塞项闭环核对

### 上轮问题 1（§1.6.3/§11.4 实验裁决三件套） — **pass（已闭环）**

| 要求 | 证据 | 独立复核 |
|------|------|---------|
| 代表 shape（含 bf16 prefill 与 fp8） | §1.6.3 (a) 389–395：`llama-8b-prefill`（tokens 8192，row_width 1024，bf16→bf16，4096 块，检验 R1 谓词直写）、`qwen3-235b-fp8`（4096，512，bf16→fp8，1024 块，检验 R2 单字节写）、`llama-8b-decode`（64，明确标注“不参与阈值判定”） | 两 dtype 写路径各有必测 shape；第三个参照 shape 已显式排除于判定，不会污染阈值 |
| 指标与未知量反解式 | §1.6.3 (b) 397–405：`L_P`/`L_B`（msprof `scope=kernel`、`us`）、原始数据落 `perf_opt/`；同 dtype 只解 `Δc = (L_P − L_B)/W`（单位 µs/GM 字节）；fp8 联立解 `BW_write_eff = ΔB_w / ((L_P − L_B) − W·Δc)` | 量纲成立：B ÷ µs = B/µs；符号自洽——B 多付 `4·block_rows` 条 `T.copy` 常数代价（§4.4 768 行），实测 `L_P − L_B` 含该项负贡献，故 `− W·Δc`（`Δc<0`）把常数项加回，恰为“纯写代价差”；两方程两未知量封闭，无外部标定 |
| 数值阈值 + 裕度 | §1.6.3 (c) 407–412：T1 `L_P ≥ 1.15 × L_B`（`1.15 = 1 + 2·CV + 5%`，`CV=5%` 假设；实测 `CV>5%` 按 `1+2·CV+0.05` 上修）；T2 fp8 `ratio_tx ≥ 2`；T3 语义门优先于 T1/T2；组合规则“任一命中即切 B，皆未命中维持 P” | T1 独立复算 `1 + 0.10 + 0.05 = 1.15` ✓，判决裕度 > 2σ+5%，升级规则同口径；T3 与 §9 R1/R2 一一对应（1051–1056 行），语义失败不等性能数据，正确优先；T2 的**归一化口径**存在因子缺失，见建议 13（不改变“已具备可机械判定阈值”的结论，且误判方向为保守切 B，不产生错误结果） |
| `cache_dtype` 分派 | §1.6.3 (d) 414：factory 已按 `cache_dtype` 特化两个 `prim_func`（§3.3 515–562），判定可只对 fp8 生效且**不增编译产物**；B 变体 UB 12,288 B / 8,192 B ≪ 253952 B；块粒度与 P 相同 ⇒ §5.5 分核三档原样适用 | 与 §4.5 785 行、§5.5 861–874 行一致；12288/253952 = 4.84% ✓ |
| 回写路径 | §1.6.3 (e) 416：Stage 4 以 `[DESIGN_LIMIT]` 附录补记写入 `perf_opt/perf_feedback.md`（schema/路由引 `_shared/standards/perf-feedback.md`），由 conductor 受控路由；未回写前维持 P | 路径具体、可执行、有归属方；§11.4 1166 行只复述编号、不另立标准，两处口径单一来源（385 行声明“本节与 §11.4 的裁决口径完全一致并互相引用”） |
| 物理合理性抽查 | — | 写带宽外推 1.26 TB/s × 56/48 = 1.47 TB/s ✓（设计已两处标注“未实测假设”，仅用于 ±50% 区间与 2× 流量量级淘汰）；符号矛盾（`ΔB_w>0` 但 `L_P ≤ L_B`）与越界（区间外）两分支均回退到模型无关的 T1，避免用坏模型反解 |

结论：三件套（shape / 指标与反解 / 阈值 / 回写 / 分派）齐全且互相自洽，上轮阻塞项 1 关闭。

### 上轮问题 2（§3.3 备选路径 B 结构缺列覆盖） — **pass（已闭环）**

| 要求 | 证据 | 独立复核 |
|------|------|---------|
| 与 P 同构的 1-D 折叠网格 | §3.3 565–567 声明 + 581/588–590（fp8）、618–623（16-bit）：`with T.Kernel(nblocks)`，`rb = bx // nchunks`、`cb = bx % nchunks`、`c0 = cb * cols`；`nblocks`/`nchunks`/`cols` 沿用 511–513 行同一编译期常量 | 与 P 路径 528–533 行逐项同构，无第二套网格定义 |
| 列覆盖（`row_width ∈ {128,512,1024}`） | §3.3 636 行三命题：`bx ↦ (bx//nchunks, bx%nchunks)` 是双射；`⋃_cb [cb·cols, (cb+1)·cols) = [0, nchunks·cols)`；`cols` 归一化（511 行）保证 `nchunks·cols == row_width`；末段 `cb = nchunks−1` 即第 nchunks 段 | 代入验算：`row_width=128` → `cols=128, nchunks=1`（单段、无 c0 偏移需求）；`512` → `nchunks=4`，段 [0,128)/[128,256)/[256,384)/[384,512) 并集 = [0,512) ✓；`1024` → `nchunks=8`，并集 = [0,1024) ✓；`row_width=96` → `cols=96, nchunks=1` ✓。`T.copy` 源 596/605 与目标 629/631 两处均带 `c0` 偏移，无“固定第 0 列”残留 |
| `s_ub` 删除或说明 | 638 行：`s_ub` 不分配，行级 `slot` 由寄存器承载，`T.copy` 目标行由标量地址 `KPages[slot, …]` 表达 | 残留检查：`s_ub` 仅出现在 11/638/740/773 行的“已删除/不占 UB”语境，无活引用；UB 清单 §4.5 779–783 行只有 4 个 buffer |
| `T.copy` region 确定形式（或两个可替代写法） | 640 行：源 `K[token, c0:c0+cols]` = “标量行号 + 标量列基址”的 1-D 连续区间（长度 `cols`），目标 `k_ub[i, :]` 形状 `(cols,)`，与 `test_tilelang_ascend_dma_copy.py` 的 `T.copy(src[:n, 0, :], ub[:n, :])` 同型；降级顺序 ① `(1, cols)` 2-D region ② 逐元素 `T.Parallel(cols)` 读 GM 到 UB | 三写法元素数均等于 `cols`，语义一致；不再有“待 Stage 3 确认”的开放形状 |
| §4.5/§11.4 计数自洽 | §4.5 783/785：4 项 4096+4096+2048+2048 = **12,288 B = 12.0 KB**，占用率 4.84%；§3.3 642：4 条/行 ⇒ **64 条/块**（store 侧 32），总数 `4·tokens·nchunks`：prefill **262,144**、qwen3-fp8 **65,536**；§11.4 1158–1159 行同值；787 行 `block_rows=64` → 49,152 B / 48.0 KB、16-bit 32,768 B / 32.0 KB | 逐项复算：12288 ✓；12288/253952 = 4.838% ✓；4·16 = 64、store 侧 2·16 = 32 ✓；4·8192·8 = 262,144 ✓（`nchunks = 1024/128 = 8`）；4·4096·4 = 65,536 ✓（`nchunks = 512/128 = 4`）；64×128×2×2 = 32,768 + 64×128×1×2 = 16,384 ⇒ 49,152 ✓。旧值 `12352` / `49.4` / `4.75%` / `0:cols` / `32 条额外交代` 全文零命中（`12.06` 仅存于第 11 行修订记录） |

结论：备选 B 已成为与 P 同构、覆盖全列、可直接实现的路径，上轮阻塞项 2 关闭。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

本轮重读源算子全文（226 行）并逐项核对 §0.1–§0.7，未发现语义、算法步骤、优化手段或 I/O 契约偏差。

| 检查项 | 结论 | 源码证据 ↔ DESIGN.md 证据 |
|--------|------|---------------------------|
| 语义理解正确性 | pass | 源码 22–24（docstring：`-1` 跳过；guard 与 slot 载入对两次 store 只付一次；fp8 饱和 cast 保 NaN）、36（`out_idx=[]`）、40（`T.dynamic("tokens")`）、62–68（fp8：`T.Cast("float32", K[token,col]) / KScale[0]`，隐式 cast 存 fp8）、88–92（16-bit 同 dtype 赋值）、209–210（`k.shape[0] == 0` 早退）↔ §0.1（19–28）、§0.2（34–45）。无归约、无返回值、dtype/边界描述逐条吻合 |
| 算法解读完整性 | pass | §0.3 表 1–9 覆盖全部计算语句：grid 54–58/80–84、坐标 59–61/85–87、边界 62/88、slot 63/89、跳过 64/90、16-bit 写 91–92、fp8 写 67–68；host 侧 209–221（早退 / 4 次 `view` / 量化追加 scale）已列出；无遗漏、无臆造 |
| 优化手段覆盖完整性 | pass | §0.4 九项逐项对应源码：直写（全文无 `alloc_shared`）、slot 共用（docstring 22–24）、谓词跳过（64/90）、动态 extent（40）、列切分/行 tile（56/82 + 158 的 `default_config`）、`threads` 映射（37/57/83）、饱和 cast（65–66 注释）、autotune 8 配置（161–174）；类别标注合理 |
| 耦合性判定合理性 | pass | §0.5 十四条均带四态处置 + 依据。抽检：#9「2-D grid → 1-D 折叠」与 `tilelang/ascend/language/kernel.py` 模块 docstring（3/6 行：1-D AI-core grid，`threads=` 由 Python 拒绝）一致；#8「饱和 cast → 显式 clamp」与 `tilelang/language/customize.py:44` 的 `clamp` 实现一致（本轮实测 `T.clamp` 可从 `tilelang.ascend.language` 取到）；#13 舍弃 autotune 与 Stage 0 冻结接口一致 |
| NPU 重设计可行性 | pass | §0.6 R1 的 `bx ↦ (rb, cb)` 双射独立复核成立；R2 的 `clamp + cast` 对有限值与 `±inf` 逐位等价，NaN 另有 R3 兜底，X3 机器证据支撑“clamp 语义必需”；R3 线程数无关性成立（源码 59–92 无跨线程归约/共享状态/atomic）。三项均满足平台 1-D Kernel、无 Cube、UB 容量约束 |
| 设计与决策一致性 | pass | §1.4（196–208）↔ §0.6 R1/R3/R2 + §1.6.1；§3.3 P 496–562 与 §6.2 891–905 结构一致；§5.4 `cols` 归一化与 §1.6.2 #8 对应；未见“§0 判重设计、§3 照抄源方案”的脱节 |
| golden 独立性 | pass | §8.1（970–999）为纯 torch `clone → (slot_mapping>=0).nonzero() → clamp(-limit, limit) → flat[rows] = stored`，无 1-D 网格、无 `SimtVF`、无 int32 窄化、无 lane 谓词；与源仓 `workloads/attention/paged_cache_write.py` 的 `ref_program` 逐字吻合，精度口径 `Exact(rtol=0, atol=0)` 一致 |

### 1. API 可行性 — pass

- 本轮实测导入 `tilelang.ascend.language as T`（纯静态，无设备）：`clamp` / `SimtVF` / `Kernel` / `dynamic` / `if_then_else` 均在 `dir(T)` 中；`SimtVF(threads=128, latency=0)` 见 `frame.py:107`；`T.Kernel` 1-D、不接受 `threads=`（`kernel.py` 3/6/143 行 docstring）。
- `T.clamp(dst, min_val, max_val)` 位于 `tilelang/language/customize.py:44`，经 `tilelang/language/common.py` 114/260 行导出（§3.4 的路径引用成立）。
- `T.simd.vscatter(src, base, index, mask=None)` docstring（`simd.py:1185–1189`）本轮复核：“index 为 per-lane 元素偏移（uint32 for 32-bit elements, uint16 for 8/16-bit）” ⇒ §1.6.3 #3 / §11.2 的“索引宽度不足”否决成立（67 M 元素池 ≫ 65535），表述细分见建议 10。
- `T.cast(..., sat=True)` 默认值不冲突（设计改用显式 `clamp`）；`T.dynamic(name, dtype="int32")` 见 `language/symbolics.py:13`。
- 编程模式：平台 `programming_modes.migration_default = simd`，设计选 SIMT 并在 §2.4（447–456）给出偏离理由（SIMD scatter 仅 `vscatter` 且索引宽度不匹配；fp8 目标 dtype 的 `vcvt/vsts` 组合未取证；SIMD 只能操作 UB），与 `workflow.md`“模式由实际执行路径选择”一致。
- 同构用法有设备侧测试佐证：`test_dynamic_outputs.py:24–27`（`T.Kernel(ROWS)` + `T.SimtVF(threads=WIDTH)` + `T.Parallel(WIDTH)` + GM 直写）、`test_ascend_fractal_storage_integration.py:36–37`（2-D `T.Parallel` 直写）、`test_tilelang_ascend_dma_copy.py`（B 路径 `T.copy` 同型）。

### 2. 内存层级规划 — pass

- §4.4（742–769）搬运路径完整：P = GM → 向量寄存器 → GM（0 B 片上缓冲，与源算子“无 SMEM”一致）；B = GM → UB → GM（762–764 行两跳，行级 `T.copy`）。
- §4.5（771–787）UB 预算仅针对 B：4 项合计 12,288 B（12.0 KB），容量 253,952 B（248 KB，`Ascend950PR_9579.ini [AICoreSpec] ub_size`，md5 已记录），占用率 4.84%；机械复算 `recompute matches: True`；16-bit 变体 8,192 B；`block_rows=64` 放大版 49,152 B / 32,768 B。
- L0A/L0B/L0C：无 Cube GEMM，不分配；`l0c_budget` skip 已人工确认不适用。
- §4.3（730–740）逐 buffer 标注层级；§4.6（789–795）明确唯一动态轴 `tokens` 只进 1-D grid 上界。

### 3. Tiling 策略 — pass

- §5.2 `block_M = 16`（`block_rows`）/ `block_N = 128`（`cols`）及量化理由（2048 元素 = 128 lane × 16 元素）；机械复算 `ceil(8192/16) × ceil(1024/128) = 4096` 与声明一致。
- 对齐：`row_width ∈ {128, 512, 1024}` 均为 128 整数倍，16-bit 行段 256 B、fp8 行段 128 B（后者恰为 1 个 128 B 事务，与 §1.6.3 (b) 400 行的“事务粒度折算”同口径）。
- 非整除：行维尾块由 `if token < tokens` lane 谓词覆盖（§6.4 927 行，最多浪费 15 行 / 1 个逻辑块）；列维由 §5.4 编译期归一化消除（928 行，运行期 0 判断）。
- 分核三要素齐全（§5.5 861–874）：① 逻辑核数 4096 及其余 workload 核数；② 物理核数 56 AIV 有实查记录（`statectl platform --task-dir …` 的 `hardware.observed_cores = {cube: 28, vector: 56}`，`core_source` 为 `Ascend950PR_9579.ini [SoCInfo]` + `rtGetAiCoreCount()=28`）；③ 分三档并量化尾浪费（顶格 4096 = 56×73+8，1.17% ≈ 0.54 µs；`block_rows = 32/64` 对照一致）。
- 建议级备注见建议 4（极大规模档未标注为对 `core-split-strategy.md §1 ③` 的偏离）。

### 4. 技术约束检测 — pass

- 三维 Kernel：不涉及（`T.Kernel` 1-D，`token × nchunks` 折叠为一维）。
- GEMM 非整除 / L0C 溢出：不涉及（无 `T.gemm`、无 L0*）。
- `T.dynamic` 限制：仅 `tokens` 一个动态轴且只进 1-D grid，符合“动态 extent 只支持 1-D”（§3.5.1 664–678）。
- 模式约束：`threads` 在 `T.SimtVF` 而非 `T.Kernel`；B 路径 UB 占用 ≪ 容量；两份 `prim_func` 由 `cache_dtype` 编译期特化，无运行期分支（§4.6 795 行）。

### 5. 循环与同步 — pass

- §6.1（878–890）逐层声明 API 与理由：`T.Kernel`（块并行）、`T.SimtVF`（线程域）、`T.Parallel(block_rows, cols)`（元素级二维并行）；显式声明**不使用** `T.serial`（P 路径）/`T.Pipelined`/`T.Persistent` 并给出理由。
- §7（934–964）同步策略与模式一致：P 无 `T.copy`、无共享缓冲 ⇒ 依赖 `InsertSync` 自动同步；B 的 `T.copy` 与 `T.Parallel` 之间由 pass 推导；不使用未认证的跨核 flag。
- §6.4（923–932）唯一真实尾块（行维）处理方式与代价写明；`slot < 0` 与 `tokens == 0` 两条“非尾块”路径均有明确处置。
- §6.3（907–913）三条“不使用 `T.Pipelined`”的理由与“单 tile/无复用/自动调度已开”一致；建议级备注见建议 12。

### 6. 验证方案 — pass

- §8.1（970–999）golden 以 §0.1 语义为唯一依据、移植源仓 `ref_program`，独立于 NPU 算法（1-D 网格 / 谓词 / 窄化均不出现）。
- §8.2（1001–1009）精度标准：`float16/bfloat16/float8_e4m3fn` 全部 `atol = rtol = 0`，`equal_nan=True`，与 `Exact(rtol=0, atol=0)` 契约一致；理由（饱和、舍入、NaN、被跳过行字节不变）写明。
- §8.3（1011–1043）覆盖度：L0 10 项 + L1 2 项 + boundary 6 项；对 NPU manifest `PagedKVCacheWriteFwdOp`（7 个 workload、4 组 `dtype_combos`）逐项对齐：fp16→fp16、bf16→bf16、bf16→fp8、fp16→fp8 全覆盖；`row_width` 16/32/96/128/512/1024（含 96 触发 `cols` 归一化分支）；页首/页尾/最后一行、全负 slot、零 token、饱和（±1000）/NaN/±inf/ties（1.0625/1.1875）/±0、`k_scale ≠ v_scale`、契约拒绝均有对应用例；分层命令与 Stage 3 `precision_command` 对齐。

### 7. 完整性与一致性 — pass

- 章节齐全：§0–§11 + §0.1–§0.7 完整；§1.6 含 1.6.0/1.6.1/1.6.2/1.6.3。
- 无占位符：全文对「花括号占位形 / 待办标记 / 补充标记」三类模式做 grep 自检，均零命中（与 `gate_lint` 的占位符模式表同口径）。
- 同类引用真实：§0.7/§3.5.3 引用的 `examples/TileOPs/{manifest,workloads,ops,tests,benchmarks,kernels/attention/paged_cache_write}`、`examples/elementwise/vec_add_2d*.py`、`examples/indexer/indexer_fwd.py`、`.agents/platforms/toolchains/ascend/kernel_template.py` 均存在；外部仓测试文件均在 `testing/ascend/` 下。
- 数值传播自洽（本轮专项抽查）：`12.06 → 12.0 KB`（§1.6.0 R3、§4.5 785、§11.3 1150、§11.4 1161）、`12352 → 12288 B`（§4.5 783）、`49.4 → 48.0 KB`（§4.5 787）三处联动全部落到新值，旧值零残留（`12.06` 仅在第 11 行修订记录中作为“修订前”说明保留）。
- 建议级备注见建议 5、6、12。

### 8. 算法优化分析 — pass

**（1）调研做了吗、结论对吗？（独立复核，非采信文档自述）**

- §1.6.0（225–417）四问齐全：R1 候选表（含基线 #0 = 源算法直译，共 6 个：直写行级提升 #1 / UB 中转 #2 / SIMD 寄存器 #3 / host 索引 #4 / 数学化简 #5-否决）；R2 在线算法；R3 复杂度四口径（FLOPs、访存字节、扫描遍数、峰值缓冲）；R4 硬件亲和逐候选六项评估。
- R2 负向断言“无在线变体”**复核成立**：源性语义中输出元素 `(s_n, c)` 仅由单一输入元素 + 标量 scale + 行索引决定，无 running 统计量（无 max/sum/计数），逐元素映射单遍即天然在线，“分块合并”无对象；对照 `algorithm-research.md §3 R2` 判据与 `algorithm-candidates.md` 命中行（reduction / softmax / RMSNorm / attention 均以归约为前提），未见被漏掉的已知候选。
- R3 复杂度**独立复算**：prefill 读 33.55 MB + 写 33.55 MB = 67.11 MB，加 `Slots` 4096 块 × 16 行 × 8 B = 0.52 MB ⇒ 67.63 MB；`67,633,152 ÷ 1.47e12 = 46.01 µs` ✓；fp8 12.58 + 0.13 = 12.71 MB ⇒ 8.65 µs ✓；发射项 1024 × 3.0 µs ÷ 56 = 54.86 µs ✓。**中间缓冲 GM 往返未漏计**（P 为 0 B；B 的 UB 往返不计 GM，额外 64 条 `T.copy`/块已在 §11.4 1158–1159 单列）。1.47 TB/s 属外推，§1.6.0 与 §9 R9 两处标注“未实测假设”，仅用于 2× 流量差的量级淘汰。
- R4 硬件亲和逐候选评估齐全；能力型负向论断（`vscatter` 索引宽度、fp8 `vcvt/vsts` 未取证）本轮由我重新打开 `simd.py` 与语言模块导入核对一致；#3 明确标注“未取证 ≠ 不存在”。
- 并列候选裁决路径：R3 中 #1/#2/#3 流量相同，设计以“无 UB + 少 32 条 store/块”单选 #1，且为 #1 与 #2 补上了 §1.6.3 的实验裁决计划（T1/T2/T3 + 回写），该前提即上轮阻塞项 1，现已闭环。

**（2）公式先优化了吗？（逐项独立推演等价性）**

| 优化项 | 原式 → 优化后 | 独立推演 | 收益 |
|--------|---------------|----------|------|
| O1（scale 载入提升，保留 fp32 `vdiv`） | 元素循环内 `KScale[0]`/`VScale[0]` → 块级标量 `ks`/`vs` | `KScale` 在 kernel 内无写者、与输出无别名 ⇒ `ks == KScale[0]` 每次成立；操作数与求值顺序完全不变 ⇒ 逐位相同。脚本对 fp16/bf16 全 65536 码位 × 2 个 scale 穷举 0 违反（我重跑确认） | 标量访存 67.11 MB → 0.52 MB；保留 `vdiv` 而非乘倒数的否决有机器证据（X1：非 2 的幂 scale 60/65536 失配） |
| O2（16-bit 恒等） | 同 dtype 赋值 → 同一表达式 | 源码 88–92 本身即同 dtype 赋值 ⇒ 恒等；65536 码位穷举 0 违反 | 消除无谓 cast |
| O3（slot 窄化 + tile 级载入 + guard） | `int64` 逐元素 → `T.cast(..., "int32")` | slot 域 `{-1} ∪ [0, NP·PS)`，`NP·PS ≤ 65536 < 2^31` ⇒ int32 精确保持；`≥ 2^31` 由编译期 guard 覆盖（脚本 4 个极端值中 3 个无 guard 时回绕，guard 后 0 失配） | 索引宽度减半、行级索引复用（依赖见建议 11） |
| G1（显式 `clamp(±448)`） | 依赖 cast 饱和 → `clamp` 后 cast | 有限值 clamp 后无溢出可饱和 ⇒ 与饱和 cast 逐位相同；`±inf → ±448`；NaN 由 clamp 传播（§9 R3）+ 兜底式。X3：仅靠 cast 在 16 个角点上 bf16 4/16、fp16 2/16 失配 ⇒ clamp 语义必需 | 正确性修正，无性能代价（2 条 min/max） |

四项均满足「原式 → 优化后公式 → 等价性论证 → 收益量化」；`verify_equiv.py` 重跑与内嵌表逐行一致，等价性结论成立。

**（3）循环 / 标量计算向量化了吗？（与 §3.3、§6 交叉核对）**

- P 路径的全部循环/标量点（§3.3 498–562）：元素级二维循环（`T.Parallel(block_rows, cols)`，已向量化）、`token < tokens` 谓词、`slot >= 0` 谓词、块索引折叠 `rb/cb`、host 元数据（`view`/早退/量化 scale 追加）。§1.6.2 #1–#12 逐点覆盖，与 §3.3/§6.2 一一对应，无遗漏点。
- 保留的标量点理由具体：动态边界（`tokens` 运行期才知道）、block 索引（grid 维度固有）、API 缺失（`vscatter` 索引宽度，`simd.py:1185–1189` 佐证）、host 元数据（不属于 kernel 内计算）。
- §6.1 887 行明确 P 路径不使用 `T.serial`，与“逐元素已全部向量化”一致；B 路径的 `T.serial(block_rows)` 是行级 DMA 结构所需，并在 §11.4 1162 行量化为“5 跳”。
- 替代 API 佐证：`vcvt/vdiv/vmaxs/vmins` 指向已存在的 ascend 测试（`test_tilelang_ascend_clamp.py`、`test_tilelang_ascend_simtvf_cast.py`、`simtvf/test_simtvf_arithmetic.py`、`simtvf/test_simtvf_vectorized_select.py`）；`T.clamp/T.cast/T.if_then_else/T.dynamic` 在语言模块确证。
- §3.1（460–474）明确以 §1.6.1 的优化后公式为拆解输入（每行先 `cast→f32`、再 `÷scale`、再 `clamp`、再 `cast→cache_dtype`），与 §1.6.1 的保留项（不改成乘倒数）一致；§1.6.3 选定的轴（行内 `col`）与 §3.3 内层 `c`、§6.1 884 行一致。

## 建议级事项（非阻塞，编号沿用上一轮）

> 以下 11 项均不影响本轮结论（无一项会改变语义、精度或编译可行性），供 Stage 3/4 或后续修订择机处理。

**建议 3（§0.3 第 3 条 + §3.5.2 表）**：`_compiled_fn` / `_compiled_config` 在源仓全库不存在（`grep -rn --include=*.py` 零命中；第 82 行与第 691 行两处均出现）；实际机制是基类 `Kernel.program(build, *facts)` 缓存编译产物（源码 222 行 + `kernel_base.py:351`）与 factory 的 `@target_cache(maxsize=32)`（源码 25 行）。建议两处一并改为真实机制名，不写不存在的变量。

**建议 4（§5.5 第三档）**：极大规模取“逻辑块超发 + 硬件分波”，与 `core-split-strategy.md §1 ③` 的“固定启动核数 + 核内分批”不同；理由量化且成立，但未标注为“偏离”。建议在该档开头加一句“本设计偏离 §1 ③，理由如下（量化）……”。

**建议 5（§9.1 R5，1055 行）**：`physical_tokens ≤ 4096` 与 manifest 不符（`cache_shape[0]` 是 `num_pages`，实际 `physical_tokens` 最大 65536）。结论（int32 远小于 2^31）不受影响。建议改为 `≤ 65536`，并把 §11.2 的“67 M 元素池”改为 `physical_tokens × row_width` 口径。

**建议 6（§1.6.0 调研结论段）**：承接段开篇说“§0.4 的 9 项”，引用编号却按 §0.5 行号且个别错位（如 `autotune（#12）`，§0.5 中 autotune 为 #13）。建议统一一套编号并标注对应关系。

**建议 7（§1.6.0 R1 表 / §11.5）**：未引用平台候选库 elementwise/broadcast 命中行 `PL-950-row-primitives`（“输入转 FP32、行标量广播、逐元素计算”，与候选 #1 同构）。实质已评估，缺可追溯链；建议补一行命中说明并把该文件加入 §11.5 来源清单。

**建议 8（§1.6.0 R3 基线表）**：`KScale`/`VScale` 为 `float32`，基线逐元素读应按 4 B/元素计，现按 8 B 计（与 int64 `Slots` 同口径），使被淘汰基线 #0 的流量与下界偏高约 1.5 倍；淘汰结论不变，但口径应可复算。建议改 4 B 重算或显式注明“保守按 8 B 估”。

**建议 9（§1.6.0 R1/R3/R4 #4）**：以“full-pool 回写 ≥ 134 MB”淘汰 host 侧索引不准确（`flat[rows] = stored` 只写被选中行）。真正的淘汰理由是契约类（非 kernel 方案 + launch/元数据开销）；建议改以契约理由为主，删除或修正该数字。

**建议 10（§1.6.2 #10 / §1.6.3 #3 / §11.2）**：`vscatter` 索引宽度应表述为“uint16（8/16-bit 元素）或 uint32（32-bit 元素）”（`simd.py:1185–1189`）；本算子 16-bit 路径走 uint16、池规模 67 M 元素仍远超 65535，否决成立。建议补全分 dtype 限定。

**建议 11（§1.6.1 O1/O3、§3.3 526–527 行、§9.2 末行）**：O1/O3 的收益以“块内只读一次 `KScale[0]`、每行只读一次 `Slots[token]`”为前提，但 `ks`/`vs` 绑定位于 `with T.Kernel(...)`（528 行）之外，`slot` 载入在 `T.Parallel` 元素循环内，最终是否提升由 codegen/LICM 决定；未被提升时流量回到 134 MB 量级（正确性不受影响）。建议 §9 增一条风险并以编译产物 IR 确认标量载入次数，同时把 `ks`/`vs` 移入 `T.Kernel` 作用域。

**建议 12（新增；§6.3 921 行）**：Stage 4 杠杆表仍用定性表述“若 GM 直写效率**显著低于** DMA 成段写，可换回”。§1.6.3 已把口径量化为 T1/T2/T3，此处应改为“按 §1.6.3 (c) T1/T2 判定”，避免同一决策出现第二套软判据（不影响结论，因 §1.6.3 已声明为唯一裁决口径）。

**建议 13（新增；§1.6.3 (c) T2 的 `tx_min`）**：`tx_min = 2 · nchunks · ceil(cols·out_bytes/128)` 缺少聚合范围的因子。`write_transactions` 在 (b) 中按全 kernel 的 msprof 写事务计数（400 行“写事务数 × 128 B 事务粒度”），则 kernel 口径的最小事务数应为 `2 · tokens · nchunks · ceil(cols·out_bytes/128)`；若按块口径则应含 `block_rows`（`2 · block_rows · ceil(...)`）。按现式，`qwen3-235b-fp8`（tokens 4096、nchunks 4、cols 128、out_bytes 1）的 `tx_min` 为 8，而 kernel 最小值为 32,768，`ratio_tx` 被放大约 4096 倍 ⇒ T2 退化为“恒真”，fp8 组合会无条件切 B。方向保守（B 语义等价、精度不受影响），且 T1 仍为主判据，故列为建议级；建议把 `tx_min` 改为与 `write_transactions` 同口径的显式表达式（kernel 或 per-block 二者择一并写明），并复核 T2 的 2 倍门限仍有区分度。
