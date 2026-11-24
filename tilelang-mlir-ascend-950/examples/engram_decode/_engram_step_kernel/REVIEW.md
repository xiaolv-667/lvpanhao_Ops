# `_engram_step_kernel` 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息

- 文档: `examples/engram_decode/_engram_step_kernel/DESIGN.md`
- 任务类型: 迁移（migration-harness；`EngramDecodeFwdOp` 的第二个 launch）
- 源算子: `/workspace/TileOPs/src/tileops/_kernels/sequence_modeling/engram/decode.py`
  （`_engram_step_kernel` 第 114–225 行 + host `EngramDecodeKernel.forward` 第 375–410 行）
- 对照文档: `examples/engram_decode/_engram_project_kernel/DESIGN.md`（第 1 个 launch，跨函数契约 §1.5）
- 检视维度: 9 项（维度 0 源算子理解与迁移分析 + 维度 1–8）
- 阻塞级问题数: 0
- 建议级问题数: 11（均记录在对应维度的「建议级观察」中，不阻塞 Stage 3）
- 检视环境: 平台 `ascend-950pr-9579` / `ascend`（dav-3510，28 Cube / 56 Vector，
  UB 253952 B / L1 524288 B / L0C 262144 B，CANN 9.2.1）；设计期编译探针以
  `ASCEND_RT_VISIBLE_DEVICES=""` 无卡执行（未占用被 Stage 3 征用的设备），未跑真机 kernel。

## 机械复核（D-1/D-5）

- **verify_equiv 重跑: PASS**。命令
  `cd examples/engram_decode/_engram_step_kernel && /workspace/venv-ascend950/bin/python verify_equiv.py`
  → `RESULT: ALL EQUIV_PASS`；采纳项 48 个 cell、0 失败、最坏 30.0 ulp。逐项 worst（脚本末尾汇总表）
  与 DESIGN §1.6.1 内嵌表**逐格一致**：A0 0 ulp / 100 %、A1 0 ulp / 100 %、A2 30 ulp / 100 %、
  A3 2 ulp / 100 %、A4 0 ulp / 100 %、A6 2 ulp / 99.9512 %（max_abs 9.54e-07）。
  informational 表也一致：B1 最坏 max_ulp 1.8278e+09、逐位一致率最低 0.3906 %、最大绝对偏差 **1.56e-02**
  （= DESIGN §1.6.1 B1 行所引证据）；B0 max_ulp 256、逐位 99.80–100 %。
- **design_calc_check: 1 pass + 3 skip（skip 项已人工复核）**。命令
  `python3 .agents/tools/design_calc_check.py --design examples/engram_decode/_engram_step_kernel/DESIGN.md --task-dir examples/engram_decode/_engram_step_kernel`
  - `r3_metrics`: **pass**（四口径 FLOPs / 访存 Bytes / 扫描遍数 / 中间缓冲峰值齐全）。
  - `ub_budget`: **skip**（未解析到分级 buffer 表）→ 人工复核：逐行相加
    `h 1024 + k 2048 + v 2048 + wh 1024 + wv 1024 + cw 4096 + tap 3072 + y 1024 + vn 1024 + scratch 8192
    = 24 576 B = 24 KiB ≤ 253 952 B`（9.7 %），且与生成源码的最大 UB 偏移
    `((half*)buf)[11776] + 512` = 24 576 B 完全吻合 → 结论成立（§4.5 表内「手工核算 23 552 B」一行
    少算 1024 B，见维度 2 的建议级观察）。
  - `l0c_budget`: **skip**（设计不含 Cube GEMM）→ 人工复核：全程无 `T.gemm`/L1/L0A/L0B/L0C，
    生成源码无 `asc_mmad`/`asc_copy_gm2l1` → 不适用，成立。
  - `core_split`: **skip**（解析到的维度不足；打印的 "declared logical cores = 56" 来自文中
    "56 Vector 核"，不是逻辑核数）→ 人工复核：逻辑核数 = `batch ≤ 17`、物理 Vector 核 56（实查记录见下），
    核数与 block 取值自洽 → 成立。
- **补充机械证据（本次独立重跑，compile-only）**：
  - `history_version/design_probe_step.py` → `RESULT: 7/7 compiled`，源码长度与 §11.1 记录一致
    （serving-b4 6000 B、decode-s1 5759 B、serving-b8 6113 B、masked tail 6777 B、
    threads=128 6384 B、w=1 5156 B、S2 5785 B）。
  - 抽读 serving-b4 生成源码（`get_kernel_source()`）：`__simt_vf__ __launch_bounds__(512)`、
    `float vf[1]`（每 lane 1 元素）、4 条 `AscendAllReduce<tl::SumOp, 512, 1, 0>` 且 scratch 偏移
    **2816 / 3328 / 3840 / 4352** float、9 条 `asc_copy_gm2ub_align` + 2 条 `asc_copy_ub2gm_align`、
    `notify(MTE2,V)` 落在 `p==2` 分支 + 一次 `wait`、VF 内 `asc_syncthreads()` 位于 `cof` 初始化与 tap
    累加之间、常量折叠为 `float(0x1p-9f)`（=1/512，精确）与 `float(0x1.6a09e667f3bccp-5f)`（=1/√512），
    tap 索引展开为 `conv_state[block_idx*10240 + p*2560 + 2560]`（row = 5/10/15 = `mcl-(w-1-p)δ`）、
    末行写 `new_conv_state[block_idx*10240 + 9728]`（row 19 = `mcl-1`）——与 §7.2/§11.1/§3.3 的自述
    **逐条吻合**。
  - `T.simd` 导出符号实测 94 个、无 `vexp2`（§1.6.2 #9 的 API 缺失论断成立）。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

亲自通读源文件（`decode.py` 全文 417 行）与 DESIGN §0.1–§0.7 逐条核对，**未发现语义偏差**：

| 检查项 | 结论 | 核对证据 |
|--------|------|---------|
| 语义理解正确性 | pass | §0.1 的 6 条公式 ↔ 源码第 166–221 行逐项对上：`k_local/v_local` 由 `kv[0/1,bid,j]` **fp32 直读**（源码 167–168 行无任何 cast，`k_local` 声明为 `accum_dtype="float"` 第 151–152 行），`h_local` 才 `T.cast`（169 行）；三个分母是 `/ float(d)`（174/181/196 行）而非 `d_padded`；`k_norm` 用 `rms_w_h`（183 行）与 §0.1「h/k 共用」一致；`alpha = 1/(1+exp(-(dot/T.sqrt(float(d)))))`（188 行）与 §0.1 显式写法一致；`new_conv_state[bid, max_conv_len-1, j] = cast(vhat_local)`（201–202 行）；`conv_out = cw[w-1]*vhat_norm` 后 `for p in T.serial(w-1)` 累加 `cw[p]*cast(conv_state[bid, mcl-(w-1-p)δ, j])`（207–214 行）；`y_t = cast(conv_out*σ(conv_out) + alpha*v_local)`（216–221 行）。§0.2 的 8 张量签名 ↔ 源码 141–148 行（`kv` 为 `accum_dtype`=fp32、`out_idx=[]` 由调用方预分配 ↔ `forward` 401–407 行） |
| 算法解读完整性 | pass | §0.3 的 10 步分解覆盖 150–221 行**全部**计算语句（load / hsq+reduce / h scale / ksq+reduce / k scale / hk+reduce+alpha / vhat / vsq+reduce / v scale / 写末行 / conv init + 3 tap / y），无遗漏、无臆造；数据流图覆盖 h、k、v、权重、tap → 4 标量 → 末行 + y 全部 buffer；host 侧（左填充、6 张量 0 填充、crop，388–409 行）已单列 |
| 优化手段覆盖完整性 | pass | §0.4 的 O1–O9 ↔ 源码实存手段（整行 fragment、`reduce_sum(dim=1)`×4、标量广播、`T.Kernel(batch)`、逐 pass、`T.serial` tap、`α·v` 重复、`rsqrt`、`threads=128`）；O10/O11 明确标注来自 950DT 只读参考而非源算子，并核实其出处：`_build_step` 四段 reducer epoch（S2 结构）与 `T.cast(T.cast(kf[i], dtype_str), "float32")`（950DT 文件**第 220 行**）、同款 v̂ 往返（**第 247 行**）——行号引用准确 |
| 耦合性判定合理性 | pass | §0.5 的 13 个条目均给四态处置 + 依据：#3 `reduce_sum(dim=1)`→**重新设计** 有缺陷条目（`TRAP-950-simd-shared-reduce`）+ 已认证替代（`PL-950-row-primitives`）；#6/#12 两项"重新设计"都在 §0.6 给了 NPU 新算法；#7 舍弃 dtype 往返有**机器证据**（B1 最大绝对偏差 1.56e-02、逐位一致率 0.39–76 %，本次重跑复现）；#11 舍弃"cache 左移挪进 step"有**归属依据**（与已冻结的 project §1.5 写者表一致：`new_conv_state[:, 0:mcl-1, :]` 归 project、末行归 step）；#13 舍弃 950DT 的 `_pick_chunk_rows`/`UB_BUDGET_BYTES` 有理由（本 kernel 无平移）。未出现"判保留却无平台依据"或"舍弃无理由"的条目 |
| NPU 重设计可行性 | pass | R1–R5 均含「源方案 → NPU 新算法 → 语义保持论证」：R1 的 pass 计数 15→2 与源码 15 个元素级 pass 逐一对上；R2 的 reducer 语义论证与模块文档一致（`tilelang/language/reduce_op.py` 的 `alloc_reducer` 原文："the physical layout can never change how many times a logical contribution is combined"——本次实读确认）；R3 的"4 reducer 并发"由探针复现（4 条独立 `AscendAllReduce` + 各自 scratch）；R4 的 `inv_d` 对 `d=2^k` 逐位等价成立、`1/√d` ≤2 ulp 已实测；R5 的 `T.SimtVF(threads=min(512,tile))` 已编译。均满足平台 1-D `T.Kernel`、UB 容量、尾轴对齐、静态边界约束 |
| 设计与决策一致性 | pass | §1.4 描述的就是 §1.6.0 选定的 C1（两遍 + 4 并发 reducer）；§3.1 以 §1.6.1 优化后公式为输入（S_h/S_k/P/S_v、`rh·rk·P`、`α²S_v`、`inv_d`）；§3.3/§6 无"§0 判重新设计、正文照抄源方案"的残留（无 `T.reduce_sum(dim=1)`、无逐 pass、无 `x/d`）；§4/§5/§7 与 §0.6 R1–R5 自洽 |
| golden 独立性 | pass | §8.1 **直接调用**源仓参考 `engram_decode_step_torch`（`examples/TileOPs/tileops/workloads/sequence_modeling.py:420-486`，本次核对：fp32 计算、左填充、恒返回 `max_conv_len` 行、末行写 `vhat_norm` 的 dtype 副本），并显式声明"不复刻因子提出 / 两遍结构 / reducer / 常量折叠"。golden 与 §0.1 语义同源，未复刻 §0.6 的 NPU 算法；`host_kv` 用 `e_t.float() @ W_K.float()` 与 golden 内部同表达式，L0 分段输入一致 |

**建议级观察（不阻塞）**：

- O-0-a：`examples/engram_decode/_engram_project_kernel/DESIGN.md` §0.1 有一句"step 段自己把 `kv` 舍入回输入
  dtype 再 RMSNorm"，与本源 §0.5 #7 / §0.1（直接 fp32 归一化）**相反**，也与该句自身后半段"GPU 源第 167–169 行
  直接读 fp32"矛盾。本源设计是对的（源码 167–168 行无 cast），跨 launch 契约（`kv` 为 fp32、project 不得提前
  舍入）两份文档一致，故不影响本设计；建议 conductor 记录该笔误，供 project 文档下次修订时删除这句
  （本 Agent 不改其他算子目录）。
- O-0-b：§0.7 与 §3.5.3 写"借用点（O10/O12）"，但 §0.4 的手段全集是 O1–O11、无 O12（应为 §0.5 #12 /
  R3「4 段 epoch → 1 段并发 4 reducer」）。悬空标签，建议改为 `O10 / §0.5 #12`。

### 1. API 可行性 — pass

- **API 存在性（逐条实读所选语言模块，`tilelang` 0.1.15 / venv site-packages）**：
  `T.alloc_reducer(shape, dtype, op)` ✓、`T.reducer_init/reducer_update(target, value)/finalize_reducer(reducer, dst)`
  ✓（v2 形态 `dst` 形参存在）、`T.SimtVF(threads=int|list|tuple, latency=0)` ✓、
  `T.Parallel(*extents, coalesced_width=...)` ✓、`T.rsqrt/T.exp/T.cast/T.if_then_else/T.int32/T.float32` ✓、
  `T.Kernel/alloc_shared/alloc_fragment/copy` ✓。与 §3.4 表格逐行一致。
- **API 与模式匹配**：同步/归约写法属 SIMT 线程域（`T.SimtVF` + `T.Parallel` + reducer epoch），与
  §2.1 选定模式一致；`T.Kernel` 不带 `threads`（平台拒绝），线程域由 `T.SimtVF` 表达 ✓。
- **API 规范/实证**：`history_version/design_probe_step.py` 本次重跑 **7/7 COMPILE OK**，覆盖
  serving-b4 / decode-s1 / serving-b8 / masked-tail / threads=128 / w=1·mcl=1 / S2 保守结构；
  生成源码标记（`asc_copy_gm2ub`/`asc_copy_ub2gm`）与 §11.1 记录一致。`§3.4` 中唯一标 ⚠️ 的
  `T.simd.*` 明确声明"仅备选、未编译"，与 §1.6.3 的定位一致，不构成主选的 API 风险。
- **建议级观察**：
  - O-1-a（文档准确性，**本次实测推翻**）：§9.2 E5 断言"`@T.prim_func` 内 Python `if w > 1:` 会被 eager builder
    两支都 trace → `UnboundLocalError: tap_ub`"。实读 `tilelang/language/eager/builder.py:449-474` 的 `ctx_if`：
    条件为 Python bool 时 `yield cond`、`ctx_then/ctx_else` 只在对应布尔值下进入，**不会两支都 trace**；实测
    （`/tmp/steprev/t_if.py`，四变体全部 COMPILE OK，w=1 的 `if w > 1:` 变体与零 extent 变体源码长度同为 1168 B）
    证明 Python 级 `if` 是编译期剪枝。因此"`w==1` 必须用零 extent 分配"不是硬约束（该写法本身合法且更好，可保留），
    但 E5 的**机制描述与根因**错误，且与 §3.3 自身"按 `exact` 在 Python 层选择是否生成谓词"（能在 Python 层分支）
    自相矛盾。建议 Stage 1 修订时把 E5 改成"用零 extent 分配统一形状（Python `if` 可用，但结构分支放在
    prim_func 外更清晰）"，或补上真实报错的完整最小复现。
  - O-1-b：探针 S1 的 SiLU 表达式写作 `cof/(1+exp(-cof))`（= §1.6.1 已拒绝的 B0 形态），而 §3.3/§6.2 的生产
    写法是 `cof*(1/(1+exp(-cof)))`。编译可行性结论不受影响（两者用同一批 API），但"编译证据 ≠ 冻结表达式"，
    建议 Stage 3 以 §3.3 为准并顺手把探针改成同款形态，免得日后误读。

### 2. 内存层级规划 — pass

- **搬运路径完整**：§4.4 的 GM→UB（9 条 MTE2）→ VF（2 段 + 4 归约）→ UB→GM（2 条 MTE3）路径与生成源码
  逐条对上（含 tap 的 3 条 `asc_copy_gm2ub_align` 与 `new_conv_state[bx, mcl-1, :]` 单条写回）；不涉及
  L1/L0A/L0B/L0C、无跨核 flag。
- **UB 预算**：逐行复算 24 576 B（24 KiB）≤ 253 952 B（9.7 %）；另以生成源码最大偏移
  `half[11776]+512 = 24 576 B` 交叉印证。`d_pad=256` 的 workload 为 12 288 B（12 KiB），与 §4.5 一致。
- **L0C 容量**：不适用（无 Cube）。
- **Buffer 层级标注**：§4.3 每个 buffer 均标注 UB / fragment(寄存器) / reducer scratch(UB，编译器追加)，
  并给出 scratch 4×`tile`×4 B = 8 KiB；与 §1.6.3 (i) 的"4 次/行归约"一致。
- **建议级观察**：O-2-a：§4.5 表内「合计（手工核算）23 552 B = 23 KiB」比逐行之和（24 576 B）**少 1024 B**
  ——正是 `design_calc_check.py` 的 `ub_budget` 想机械拦截的混合口径/加总类错误，但该检查本次 skip。
  由于全文实际使用的是下一行「探针实测 24 576 B = 24 KiB」，结论与预算余量（9.7 % vs 248 KiB）均不受影响；
  建议 Stage 1 修订时把这行改成 24 576 B，避免后续误引用。

### 3. Tiling 策略 — pass

- **Block 划分明确**：`grid = batch`（1 逻辑块 = 1 个 batch 行）、`tile = next_pow2(d_padded)`、
  `threads = min(512, tile)`、`reducers = 4`、`taps = w-1`（§5.2）；理由在 §1.6.3 (i)(ii) 与 §5.5 ③。
- **对齐约束**：`d_padded = align_up(d, 256)`（`ALIGNMENT = 256`，实读
  `/workspace/TileOPs/src/tileops/_kernels/tiling.py:9` 与 `examples/TileOPs/tileops/kernels/sequence_modeling/engram_decode/_engram_decode_fwd_kernels.py:16`）
  → `d_pad % 64 == 0`，fp32 向量宽 64 整除；5 条 workload `d_pad ∈ {256,512}` 时 `tile == d_pad`、无尾 lane。
- **非整除处理**：`tile > d_pad`（`d=600`→`d_pad=768`→`tile=1024`）走谓词分支（读与归约贡献置 0、写回守卫，
  §6.4），探针 masked-tail 变体编译通过（6777 B）。纯 Vector 算子无 GEMM 尾块。
- **分核策略三要素**：
  - ① 逻辑核数 = `batch`，逐 workload 列了 1/1/4/8/2/17/1（§5.5 表），与 manifest 5 条 workload + fixture
    两行核对一致（`examples/TileOPs/tileops/manifest/sequence_modeling.yaml` 的 `EngramDecodeFwdOp`：
    decode-s1 b=1/d=256/mcl=9/w=4/δ=3/fp16；serving-b1 b=1/d=256/mcl=12；serving-b4 b=4/d=512/mcl=20/δ=5；
    serving-b8 b=8/d=256/**bf16**/mcl=18；partial-cache b=2, L=5；fixture `EngramDecodeFixture`
    `test_engram.py` 的 17 行 d_mem=519 与 `mcl=1,w=1,δ=1` 两行）。
  - ② 物理核数依据 = **平台实查记录**（`statectl platform --task-dir` 的
    `hardware.core_source`："Ascend950PR_9579.ini [SoCInfo] ai_core_cnt=28, cube_core_cnt=28,
    vector_core_cnt=56; runtime probe rtGetAiCoreCount()=28"，本次查询复现同一字符串），并给出 ruleset 口径的
    查询代码（`torch.npu.get_device_properties` → `cube_core_num/vector_core_num` = 28/56）与 CPU 侧
    INI 复核（28/28/56）。设备被 Stage 3 占用时引用平台已落库探测记录，属"有查询代码 + 实际返回值"，
    满足 core-split 标准 §1 要素②。
  - ③ 规模判定：最坏 17 ≤ 56 → **无需适配**（单波、无 `T.Persistent` 核内串行、不做块数对齐），并给出
    "为何不往 `d` 方向拆核"的量化依据（每行 ≈20·d = 10 kFLOP ≈ 0.024 µs vs 拆行代价 ≥ 5 µs 或未认证 barrier）。
  - 机械 `core_split` 检查 skip（解析维度不足），以上为人工复核结论。

### 4. 技术约束检测 — pass

- **三维 Kernel**：不涉及（`T.Kernel(batch)` 本身 1-D；平台拒绝多维网格）。
- **GEMM 非整除 / L0C 溢出**：不涉及（无 `T.gemm`、无 L0 资源）；`d_pad % 16 == 0` 由 host `align_up(d,256)` 保证。
- **平台已知限制逐条核对**（§3.5.1）：1-D grid ✓、`T.Kernel` 无 `threads` ✓、高层 SIMD 二维归约缺陷已规避
  （用 reducer 而非 `T.reduce_sum(dim=1)`）✓、`ascend.batch_schedule` 跨 block barrier 未认证→本算子无需
  跨块通信 ✓、UB 容量 ✓、`max_conv_len >= δ(w-1)`（manifest shape rule + 上游 `__init__` 第 293–298 行）
  被正确用作"删除 tap 索引守卫"的前提 ✓、`w==1`/`mcl==1` 由零 extent + 空循环 + `mcl-1=0` 覆盖 ✓。
- 次要 API 事实（`T.cast(x,"float32") ** 2` 不被 eager builder 支持，E6）与本次实测一致（`pow()` 只接受
  PrimExpr 底数）。
- **建议级观察**：见维度 1 的 O-1-a（E5 的 eager builder 机制描述错误）与 O-1-b（探针 SiLU 形态）。

### 5. 循环与同步 — pass

- **循环结构**：§6.1 逐维列出 `T.Kernel`（block 并行）/ `T.Parallel(tile)`（2 段）/ `finalize_reducer`×4 /
  `T.Parallel(1)`（标量链）/ `range(w-1)`（Python 展开）/ 元素级"无"；与 §3.3 伪代码一一对应，无未申报的
  逐元素标量循环。
- **同步与模式匹配**：§7.1 声明自动同步（`allow_autoschedule`，不写手工 flag），§7.2 的同步点清单与生成源码
  **逐条吻合**（`notify(MTE2,V)` 在最后一个 tap copy 之后 + `wait` → `asc_vf_call`；VF 内 `asc_syncthreads()`；
  `notify(V,MTE3)`/`wait` 后两条写回；两次 UB→GM 同队列无额外同步）。`w==1` 时 1 条 notify + 1 条 wait
  （探针变体 6，5156 B）与 §7.2 末段一致。
- **尾块处理**：§6.4 覆盖 `tile > d_pad`（尾 lane 置 0 + 写回守卫）、`w==1`、`mcl==1`、batch 尾块（不存在）、
  归约域（跑满 `tile` 但分母恒为 `d`）。
- §6.3 的不流水论证（GM 全集中在开头 9 条预取、phase 2 依赖 phase 1 的标量、tap 无运行时循环、
  batch 是 block 并行）成立，且"只有 2 遍"与 §1.6.0 R2 的 inherent 2-phase 结论一致。

### 6. 验证方案 — pass

- **Golden 函数**：§8.1 直接调用源仓参考 `engram_decode_step_torch`（本次核对源码：fp32 中间量、左填充、
  恒返回 `max_conv_len` 行、`y_t = silu(conv_out)+v_hat`、末行 `v_hat_norm.to(dtype)`），并附 `host_kv`
  以给出与 project 段同语义的 fp32 `kv`（含 `d_pad` 零填充）；golden 独立于 NPU 算法。
- **L0 测试计划**：§8.3 给出 L0/L1/boundary/all 四层，L0 5 条用例覆盖 `mcl == δ(w-1)`（decode-s1 命中
  tap 索引 0）、`grid=1`、bf16、partial-cache（L=5）、`mcl=1/w=1`；L1 覆盖 `tile=512`/batch-17/12 步滚动；
  boundary 覆盖 `d=600` 掩码路径、两档 `threads`、角点输入。shape 来源与 manifest/fixture 实值一致
  （`EngramDecodeFixture` 实读：第 3 行 = `17, 519, 256, mcl=12, w=4, δ=3, fp16`（batch-17），
  第 4 行 = `1, 512, 256, mcl=1, w=1, δ=1, fp16`（one-tap）；manifest 5 条 workload 与 §0.2 表逐值一致）。
- **精度标准**：fp16 atol 0.05 / rtol 0.05、bf16 0.1 / 0.05，与 `examples/TileOPs/tests/ops/test_engram.py`
  的 `_tol_decode`（第 67–70 行）**逐值一致**；未修改行要求逐位不变（哨兵断言）；角点要求位型一致。
- **建议级观察**：O-6-a：§8.3 的 `decode-partial-cache` 行写"4 条断言；…；`new_conv_state` 12 行全检"，
  但 §8.1 的断言 3 要求"前 `mcl-1` 行逐位不变（哨兵）"。在 **step-only 分段测试**下，前 `mcl-1` 行若与 golden
  的平移结果比较必然不等（那不是本 kernel 的写集）；"12 行全检"只在整算子（project+step）口径成立。
  建议在 §8.3 注明两种口径的归属（分段 L0：y + 末行 + 哨兵不变；整算子：12 行全检），避免 Stage 3 误设断言。
  O-6-b：§8.1 的 golden 内部自行做投影（`e_t.float() @ W_K.float()`），与分段 L0 用的 `host_kv` 同表达式 →
  输入一致；整算子测试时 `kv` 来自 Cube 段（fp32 累加），与 torch matmul 的累加顺序不同，靠 atol 0.05 兜住，
  建议 Stage 3 在 L1 记录该跨 launch 差异的量级（§9.1 已把它列为"第一嫌疑"）。

### 7. 完整性与一致性 — pass

- **章节齐全**：§0（0.1–0.7 全）+ §1–§11 齐全；§1.6 含 1.6.0/1.6.1/1.6.2/1.6.3 四子节。
- **无占位符**：全文无占位符标记（对未完成标记类关键词与花括号模板变量做 grep，结果均为空；§10.2 的
  `⬜ 待 Stage 3` 是交付物状态标记，非占位符）。
- **同类引用真实**：§3.5.3 引用的 `examples/engram_gate_conv/_engram_gate_conv_fwd_kernel/DESIGN.md`、
  `examples/engram_decode/_engram_project_kernel/DESIGN.md`、`examples/norm/fused_add_rms_norm/`、
  `examples/mla_dense_bwd/_delta_kernel/DESIGN.md`、
  `examples/moe_permute_align/_make_align_kernel/repro/TRAP-950-shared-shift-copy.py`、
  `/workspace/TileOPs-ascend950/.../decode.py`、knowledge 下的 `PL-950-row-primitives.md` /
  `operator-patterns.md` / `numerical-boundaries.md` 全部**存在**；兄弟设计确实用 SIMD + `update_mask`、
  生成源码长度/结构引用（serving-b4 6000 B 等）与本次重跑一致。
- **内部一致**：API 映射（§3.2）× 内存规划（§4.3/§4.5）× Tiling（§5.2/§5.6）× 伪代码（§3.3）× 循环表（§6.1）
  五处描述无矛盾（buffer 名、reducer 数、`threads` 取值、tap 展开方式一致）。
- **建议级观察**：
  - O-7-a：§6.1/§6.2 称 tap "运行时无 tap 循环"，生成源码实际是 `for (int32_t p = 0; p < 3; ++p)` 的**常量边界
    C 循环**（3 条 copy 与 3 项累加被 canonicalize 回卷）。§1.6.2 #3 已如实写出这一点，建议 §6.1/§6.2
    改为"Python 展开后由编译器回卷为常量边界循环，运行时无动态索引算术"，消除同一文档内的两处不同表述。
  - O-7-b：见维度 0 的 O-0-b（"O12"悬空标签）。

### 8. 算法优化分析 — pass

**§1.6 章节存在性**：1.6.0 / 1.6.1 / 1.6.2 / 1.6.3 四子节齐全（§389–§691）。

**算法调研存在性与完整性（四问）**：pass。R1 候选表 10 行且**含基线**（候选 0 = 源算子，15 pass + 4×
`reduce_sum(dim=1)`，与源码逐条对上）；R2 明确结论"**有**在线变体并被采纳为 phase 1 单遍形式"，并给出
结构判据（可分解 running 统计量）+ 边界（`alpha` 依赖整行归约 → 2 遍是语义下界）；R3 四口径表覆盖
FLOPs / Bytes / 扫描遍数（并**区分 GM 遍数与行内元素级 pass 数**，口径写明）/ 中间缓冲峰值，**基线在表**，
并列了跨核代价与可并行度；R4 逐候选六项检查清单（单元匹配/容量/对齐/静态边界/流水融合/跨核），负向淘汰
各有证据或显式"未实测假设"标注。调研结论（选定 C1）与关键依据（唯一能把行内 pass 压到 2 的候选、访存已是
下界、只有 C1/C2/C3 满足单 launch+全静态+全片上）明确；"无更优替代"写明调研范围（algorithm-research.md
的替代空间 #1–#9 + 本仓同类实现 + `algorithm-candidates.md` 的 reduction/normalization 行 + pattern-library）。

**调研结论正确性（独立复核）**：pass。
- 复杂度**独立复算**：serving-b4 流量 = `4·512·2`（h）+ `2·4·512·4`（kv fp32）+ `4·3·512·2`（tap）+
  `6·512·2`（权重）+ `2·4·512·2`（写出）= 4096+16384+12288+6144+8192 = **47 104 B**（与 §1.6.0/§11.2 一致）；
  FLOPs `20 FLOP/元素 × d × batch = 20×512×4 ≈ 41 k` 的量级成立（逐元素 ~20 FLOP：h²/k²/hk·w²/v² 6、phase2 ~10、exp/div ~4）；
  行内 pass 计数 15 与源码 15 个元素级 `T.Parallel` 语句**逐一对应**；中间缓冲 GM 往返**未漏计**（C1 = 0 往返；
  候选 7 计入 `vhat_norm` 往返 4 K + tap 重读 48 K；候选 4 计入 0.4 K 归并 workspace）。四口径数值与设计数字一致。
- **负向断言逐条复核**（对照 `algorithm-research.md` §5 的替代空间 + `algorithm-candidates.md` 命中行 +
  源码证据 + 同类案例）：
  (a) "候选 6 深度卷积无公共收缩轴"→ 深度卷积权重按列独立、无公共收缩维，结构上成立（源实现也是逐 tap SAXPY）；
  (b) "无更优等价化简公式"→ reduction 类命中行（"单行 SIMT sum/max reducer"）与 normalization 类命中行
  （"FP32 统计量 + affine，输出转回输入 dtype"）本设计都已采用，未发现被漏掉的已知候选；
  (c) 在线类：设计**没有**误称"无在线变体"，反而采纳了单遍 running 统计（正确方向）；`alpha` 的两阶段依赖是
  真实 inherent 依赖（源码 187→188→191 行，`v̂` 依赖 `alpha`），A3 因子提出把它降级为标量乘法——本次独立
  推演确认（`Σ(αv)² = α²Σv²` 在实数域恒等，`α²·S_v` 与逐项平方和在 fp32 只差舍入）。
  (d) A3 的下溢边界论证独立复核成立：`||h_n|| ≤ √d`、`||k_n|| ≤ √d`（`w_h≡1` 且 `rrms≈√(d)/||·||`）→
  `|dot|/√d ≤ √d` → `alpha ≥ σ(-√d) = 1/(1+e^{22.6}) ≈ 1.5e-10`（d=512）→ `α² ≥ 2.3e-20 ≫ 1.18e-38` ✓；
  退化时两式同趋 `rsqrt(eps)`，行为一致。
- **并列候选裁决路径**：R3 中 C1 与候选 3（SIMD）属同一算法族、判定裕度落在两个未实测常数上，
  设计**没有纸面拍板**，而是交由 §1.6.3 (iii) 的三件套（主选 + 结构完整的备选 + 明确阈值的 A/B 计划）。

**调研→设计一致性**：pass。§1.4/§3.1/§6 实现的都是 C1（两遍 + 4 并发 reducer + A0–A4/A6）；
迁移侧 §0.5/§0.6 的处置与调研选族一致（#6/#12 的"重新设计"= R1/R3；#7 的"舍弃"= B1 语义偏差）；
源算法优化意图**逐项承接**（O1→UB/fragment 行驻留、O2→并发 reducer、O3→reducer `dst` 广播、O4→`T.Kernel(batch)`、
O5→两遍、O6→phase 2 内联、O7→A4、O8→保留 rsqrt + A6、O9→`T.SimtVF`），唯一舍弃项 O11 有理由，**无静默丢弃**。

**数学等价优化完整性（四要素）**：pass。A0（前提项）/A1/A2/A3/A4/A6 各有「原式 → 优化后公式 → 等价性论证 →
收益量化」。收益列非空且口径明确（pass 15→2、省 fragment/乘法/除法，并标注 0.5 µs/op 常数为 910B2C 实测、
950PR 未实测）；A6 明确"收益小、不夸大"。无优化空间部分给了 5 条显式结论 + 依据（无逐元素除法热点、
CSE 已尽、无可降代算子（`simd` 无 `vexp2`，本次实测 94 符号确认）、访存已是下界、归约遍数已是下界）。

**等价性论证正确性（独立推演）**：pass。逐项推演：
- A1：四条链的项集合与 `j` 遍历顺序不变，仅交错执行 → 每个累加器的中间值序列逐位不变（脚本用 numpy 逐 lane
  顺序累加模拟，实测 0 ulp，本次重跑一致）✓。
- A2：`(h·rh·w)(k·rk·w)` → `rh·rk·(h·k·w²)`；**不改变归约顺序**，只改每项的两处标量乘法位置。误差量级估计：
  每项 ≤ ~3 ulp 的乘序差，512 项同序累加后相对误差 ~几 ulp；实测最坏 30 ulp 落在 `alpha-small` 角点，
  且这是 `alpha`（≈σ(-22.6)≈1e-10）的相对 ulp 放大效应（`dα/α ≈ |Δx|`），传播到 `vn/y` 后仍 ≤ 3.6e-6 相对，
  比 op 容差 5e-2 低四个数量级；写出 dtype 值 100 % 逐位一致、max_abs = 0 → **不超差** ✓。
- A3：`Σ(αv)² = α²Σv²` 实数恒等，fp32 仅因子提位；实测 2 ulp、逐位 100 %；下溢边界见上 ✓。
- A4：源实现两处 `alpha*v_local[j]`（191/219 行）是同一 fp32 表达式，确定性保证逐位相同 → CSE 恒等 ✓（解析论证；
  机器 cell 的问题见建议级 O-8-a）。
- A6：`x·(1/d) ≡ x/d` 当 `d=2^k`（仅改指数，逐位）；`dot·(1/√d)` 误差 ≤1 ulp，实测最坏 2 ulp、逐位 99.9512 %、
  max_abs 9.54e-07 ≪ atol ✓。`d=600`（非 2 的幂）走同一公式时实测 1 ulp、写值仍 100 % 一致，无风险。
- 与 §8.2 精度标准的一致性：最坏 30 ulp ≈ 3.6e-6 相对 ≪ atol/rtol 0.05 ✓。

**向量化替代覆盖完整性（与 §3.3/§6 交叉核对）**：pass。§1.6.2 列出 11 个计算点，我按 §3.3 伪代码与 §6.1 循环表
逐一核对：块索引（#1）、编译期常量（#2）、tap 索引（#3，Python 展开）、phase 1 的 4 条贡献（#4）、4 次行归约（#5）、
tap 守卫删除（#6）、标量链（#7）、phase 2 复合链（#8）、`exp` 降代换（#9）、尾列谓词（#10）、9+2 条 DMA（#11）
——与 §3.3/§6.1 的元素一一对应，**无遗漏点**；§6.1 的"元素级：无逐元素标量循环"与 §1.6.2 结论一致（生成源码
即 lane 并行表达式，本次实读确认）。

**不可替代理由充分性**：pass。保留的 3 处均有具体理由：block 级索引（`T.Kernel(N)` 只给 1 维 `bx`）、编译期常量
（不产生运行时指令）、block 级标量链（一核一行、每行 O(1)，拼向量只得 1 有效 lane 属净负，且 `batch ≤ 17` 个标量
vs `tile×2` 元素）；API 缺失项（#9 `vexp2`）**本次实测复核成立**（`T.simd` 94 个导出符号无 `vexp2`，`T.exp` 已映射
`expf`，生成源码为证）。无"实现简单/照源码写"式理由。

**替代 API 可行性**：pass。所有采纳的替代 API 均有语言模块实读 + 编译探针（见维度 1）；`T.simd.*` 仅作备选并明确
未编译，与维度 1 口径一致。

**向量化轴与布局决策完整性**：pass。规约类必选候选齐备：水平归约（lane↔`j`）与垂直扫描（lane↔batch）**分别枚举**
（(i)#1/#2），另有 2-D lane 混合（#3）与 `d`-major 转置（#4）；逐候选有量化评分/否决理由（垂直扫描只用
8/512 = 1.6 % lane、`batch=1` 退化；2-D 混合归约次数 ×nchunk；转置链 `2·batch·d·2 B` = 8 KiB ≈ 本体流量 17 %）。
(i)#1 的轴质量评分（整除性 `tile/64 ∈ {4,8}`、累加链形态、repack 代价 0、UB 9.7 %）完整；(ii) lane 宽度三档
（512/128/64，含每 lane 元素数与生成形态）有默认选择与 A/B 依据；(iii) 执行单元给出主选 + **结构完整的备选**
（SIMD 的布局/nchunks/UB 16 KiB/寄存器累加器/结构/分核三要素重算/风险）+ **裁决计划**（代表 shape
serving-b1 与 serving-b4、`msopprof` kernel-only 时长、≥3 轮交错、阈值"备选 < 主选×0.90"、回写路径、
按 shape 分派可行性）——满足"判定裕度依赖未实证常数时必须三件套"的要求。弃选前提也经核对：`T.simd` 的
`update_mask` 用法与兄弟算子一致（`examples/engram_gate_conv/.../DESIGN.md` 的 `min(64, d-ch*64)`），
不是"未考虑"或"I/O 契约即内部布局"式论证。

**布局决策与实现一致**：pass。(i)#1 + (ii)#1 + (iii)① 与 §3.3（`(tile,)` fragment + `T.Parallel(tile,
coalesced_width=1)` + 4 reducer）、§4.3（9 个 UB + 4 scratch，24 KiB）、§6（两遍、无运行时 tap 循环）三方一致。

**与下游章节一致**：pass。§3.1 以 §1.6.1 优化后公式为输入（S_h/S_k/P/S_v、`rh·rk·P`、`α²S_v`、`inv_d/inv_sqrt_d`）；
§6 未出现 §1.6.2 已判向量化点的逐元素标量循环。

**建议级观察**：

- O-8-a（机器验证的诚实性）：`verify_equiv.py` 的 `run_candidate` **不消费** `"A4"` 开关（`av = alpha*v` 恒执行），
  因此 A4 的 48 个 cell 之一是 `f(cand) vs f(cand)` 的**自比较**，"实测 0 ulp"是平凡的。A4 的结论本身仍成立
  （191/219 行同一 fp32 表达式，确定性 ⇒ 逐位恒等），我的独立解析推演已覆盖；建议脚本给 A4 加真实基线
  （两处独立计算 `alpha*v`）或把该行标注为"解析恒等（非机器实测）"，避免"48 cell 全实测"的表述高估覆盖度。
- O-8-b（收益量化）：A4 的"省 1 次 `v` 的 UB 读"在最终结构下不成立——`v` 在 phase 1 已拷入寄存器 `vf`，
  phase 2 的两次 `alpha[0]*vf[i]` 都是寄存器读（生成源码 `vnf = ((alpha*vf)*rv)*wv` 与 `+ alpha*vf`），
  实际只可能省 1 次乘法（且依赖编译器 CSE）；同时 §3.1 的符号 `av` 未在 §3.3/§4.3 物化（内联两次），
  两处写法不一致。建议把 A4 的收益改为"1 次 `tile` 宽乘法（或由编译器 CSE 吸收）"，并在 §3.3 注明
  "`av` 以内联形式表达、数值上与物化等价"。
- O-8-c：A0 标注为"（前提）"是恰当的，但其"收益量化"（去掉每 lane 的 `j<d` 谓词、serving-b4 省 2048 次比较）
  是相对"掩码实现"而非相对源算子（源实现本来就跑满 `d_padded`）。建议在 A0 收益列注明参照对象，
  避免被误读为"相对基线的优化"。

## 检视小结

九个维度全部 pass，无阻塞级问题。设计的强项是：源算子语义（尤其 `kv` 直接 fp32 归一化、只写 history 末行、
分母恒为 `d`）与源码逐语句对齐且引用行号准确；两处刻意偏离 950DT 参考都有可复核证据（B1 的 1.56e-02 偏差
本次重跑复现，cache 平移归属与已冻结的 project 契约一致）；算法调研四问齐全、负向断言经复核成立、复杂度
四口径独立复算一致；等价性 6 项经独立推演 + 机器重跑双确认（48 cell、最坏 30 ulp、写值逐位 ≥ 99.95 %），
且 `alpha` 角点的 30 ulp 经推演确认来自相对 ulp 放大、不改变精度结论；API 与边界（4 并发 reducer、
`T.SimtVF(threads=min(512,tile))`、零 extent buffer、`w=1`）全部由 7/7 编译探针与生成源码实读佐证。
上述 11 条建议级观察均为**文档准确性/口径澄清**性质，不改变任何公式、布局或冻结决策，可在 Stage 1
下次修订时顺带修正，不阻塞 Stage 3。
