# _make_small_batch_kernel 设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息

- 文档: `examples/moe_permute_align/_make_small_batch_kernel/DESIGN.md`（1,416 行）
- 任务类型: 迁移（GPU TileLang/CUDA 源算子 → Ascend950PR_9579 / `ascend` 工具链，harness 子模式）
- 源算子: `/workspace/TileOPs/src/tileops/_kernels/moe/permute_align.py`（472 行；本设计对应 `_make_small_batch_kernel`
  第 221–336 行 + `forward` 第 426–466 行的分派逻辑；另核对 `_make_align_kernel` 31–159 行、`_make_scatter_kernel`
  162–218 行以确认三路输出契约）
- 检视维度: 9 项（源算子理解与迁移分析 / API 可行性 / 内存层级 / Tiling / 技术约束 / 循环同步 / 验证方案 / 完整性 / 算法优化分析）
- 编程模式: SIMT（`T.SimtVF(threads=32/64/256)` + `T.Parallel`），偏离 `programming_modes.migration_default = simd`
- 检视依据: 平台 `stage_skills.review = tilelang-ascend`；`statectl platform --task-dir <task>` 的 `ruleset`/`design_budget`/
  `knowledge`；`references/programming.md`；`knowledge/`（algorithm-candidates / PL-950-row-primitives / numerical-boundaries /
  execution-patterns / bottlenecks / capability-gaps）；`.agents/skills/tilelang-op-optimize/references/pattern-library/constants.md`
- 阻塞级问题数: 0
- 建议级问题数: 7（均为不阻塞的量化/措辞修正，见各维度 `warn` 说明）

## 机械复核（D-1/D-5）

### verify_equiv 重跑

命令（cwd = 任务目录，`source .agents/platforms/toolchains/ascend/env-950pr.sh`，`ASCEND_RT_VISIBLE_DEVICES=""`，不 launch 任何 kernel）：

```
/workspace/venv-ascend950/bin/python verify_equiv.py
```

结果：**PASS / 全部 EQUIV_PASS，退出码 0，0.53 s**；14 行检查的用例数与设计 §1.6.1 内嵌结果表**逐行一致**
（9 行 × 70 例 + A1 7,168 + A4 280 + 补位 lane 110 + 判别性对照 50 + `numel=0` 4），最大绝对差/最大 ulp 差全为 0。
L0 地标亦逐条复现：`(0,8,16) total=0`、`(100,8,48) total=384`、`(600,32,64) total=2048`、`(1023,8,16) total=1072`、
`(1000,3,1) total=1000`，判别性对照仍为「界 E 错 0/50、界 W 错 50/50」。

唯一差异是**行标签**（非数值）：脚本把补位 lane 行与 `numel=0` 行标为 `A6`/`A7`，而 DESIGN §1.6.1 表把同样两行标为
「A3 补位 lane 纪律」/「A7」且 DESIGN 的 `A6` 另指「哨兵填充」。用例数与数值完全对应，属文档标签不统一（建议级，不影响结论）。

### design_calc_check

```
python3 .agents/tools/design_calc_check.py --design examples/moe_permute_align/_make_small_batch_kernel/DESIGN.md \
        --task-dir examples/moe_permute_align/_make_small_batch_kernel
```

| check | status | detail | 复核处置 |
|---|---|---|---|
| `ub_budget` | skip | no per-level buffer tables parsed from the memory plan | **机械跳过，人工已核**：§4.5 逐 buffer 复算 `s_x 4096 + s_fill 4096 + s_cnt 128 + s_scan 256 + s_cs 132 + s_ids 256 = 8,964 B`；全用例峰值 `(1023,8,16)` 的 `BLK=128` ⇒ `s_ids 512` ⇒ 9,220 B = 3.63% × 253,952 B ✓。另用探针 M9/M11 降级码的 `buf_dyn_shmem` 偏移反证实际分配（M9: `s_fill[0..1023] + s_x[1024..1663](chunk=640) + … + s_cs[1792..1824]` ≈ 7.3 KB，无隐藏膨胀） |
| `l0c_budget` | skip | no Ascend Cube GEMM call found | 设计无 `T.gemm`/L0 分配（§4.4、§3.2），不适用 ✓ |
| `core_split` | skip | insufficient parsed dims {}; declared logical cores = 1 | **机械跳过，人工已核**：§5.5 三要素齐全（逻辑核数 1 / 实查 `cube_core_num=28, vector_core_num=56` 并附查询代码 / 规模判定），1 ≤ 56 ✓；`statectl platform` 的 `hardware.observed_cores = {cube:28, vector:56}` 与 `core_source` 交叉一致 ✓ |
| `r3_metrics` | pass | all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak) | 逐格复算见维度 8 |

### 补充自证（非 D-1/D-5 门禁项，检视方主动重跑）

设计把三条编译期事实作为**承重约束**，我按下述命令重跑其探针（无卡编译，不 launch）：

```
/workspace/venv-ascend950/bin/python history_version/design_probe_refined.py   # 11/11 COMPILE_OK
/workspace/venv-ascend950/bin/python history_version/design_probe_forms.py     # 6/7 COMPILE_OK（K7 为负对照）
```

- **M1 vs M5（承重）**：M1（原子循环内读 UB `s_x[idx]`）降级为 `for (r=0; r<19; ++r) { asc_syncthreads(); … }` ——
  19 次 barrier 与 DESIGN §0.5 D3「探针 M1/M2 实测 19 个」逐字复现；M5（同一循环改读 GM `flat[idx]`）的降级码
  **0 个 `asc_syncthreads`、0 条 `[ThreadSync]` 告警**。§0.6 R5 的「读 GM 而非 UB」是正确且承重的设计决定。
- **M9（最终单核骨架）**：填充 3 次 `asc_copy_ub2gm_align(..., min(1024, P-t*1024)*4, …)`，末块后
  `asc_sync_notify(PIPE_MTE3, PIPE_V, 0)`，对应 `asc_sync_wait` 恰落在散射 VF（`main_kernel_simt_vf_7`）之前 ——
  §6.3/§7.2 的「填充先于散射」排序链成立；散射体为 `slot = asc_atomic_add(&s_cs[eid], 1)` +
  编译器自动值域守卫 `if (slot < P && 0 <= slot) sorted[slot] = …`，与 §3.4/§1.6.2 V8 描述的降级形态一致。
- **原子 API**：`return_prev=True` 走 `tl.atomic_add_ret_elem_op` → 表达式形式 `asc_atomic_add(ptr, val)`
  （`tilelang/src/ascend/codegen/codegen_ascend.cc:2502-2540`）；CANN 头
  `/usr/local/Ascend/cann-9.2.1/asc/include/simt_api/device_atomic_functions.h:25`
  声明 `int32_t asc_atomic_add(__ubuf__ int32_t*, int32_t)`（`__NPU_ARCH__ == 3510`）。K6 降级码复现为
  `asc_atomic_add((&(s_count[key[…]])), 1)` ✓。**真机语义/吞吐仍无执行证据**（设计 §9.1 风险 1 已登记）。
- **`T.cumsum` 负对照**：K7 `COMPILE: FAIL` 且报错文本与 §3.4 引用逐字相同
  （`tl.cumsum requires a target-specific implementation … {"kind":"ascend",…}`）✓。
- **`T.simd.vscatter` 只能 UB 散射**：`simd.py:1166-1176` 的 docstring 为 `base[index[lane]] = src[lane]`，
  codegen 把 base 强转 `__ubuf__ … *`（`codegen_ascend.cc:2444-2457`），且 `simd.py` 内无任何 atomic 导出 ——
  §2.2 偏离 `simd` 的两条依据（GM 散射、寄存器面原子均无 API）**经亲自核对文档与 codegen 成立**。
- **`numel=0`（M11）**：填充 540 B（P=135×4）、`flat` 的 GM 拷入被 DCE（设备端参数被丢；host 侧 `k.params` 仍保留 4 参数是
  设计 §9.1 风险 8 的本机实测陈述，本次未复核该 host 侧细节）、`npost` 与 `expert_ids`（36 B = 9 块）照常写出、
  散射 VF 折叠为空体 —— 与 §9.1 风险 8、§1.6.1 A7 描述一致。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

核对方式：亲自读取 `permute_align.py` 全文（含 align/scatter 两段），逐条对照 §0.1–§0.7 与 §1–§8.1。

- **语义理解正确性（pass）**：§0.1 的三输出语义与源码逐项一致 —— `sorted_token_ids[P]` 先全写哨兵 `numel`（源 265–269 行）
  再把每条路由写进其 expert 区段（源 325–332 行）；`expert_ids[NB]` 的 owner（源 315–323 行）与「越过 `total` 的 block
  写 `num_experts`」（源 309–313 行）；`num_tokens_post_pad[0] = cumsum_s[num_experts]`（源 304 行）。
  `P = numel + (E+1)(B-1)`、`NB = ⌈P/B⌉`（源 241–242 行）与 §4.2 一致；`WARP_LANES = 32`
  （`/workspace/TileOPs/src/tileops/_utils/utils.py:9`）与 §0.1「E ≤ 32」一致。
  dtype（全 int32，无浮点累加）与 §0.1 的「无舍入、无累加顺序精度问题」判断成立。
  **边界语义**逐条与源码核对：① `E > 32` 抛 `ValueError`（源 236–240 行）↔ §9.1 风险 10 ✓；② `numel = 0` **确会**进入本路径
  （源 396–400 行的判据 `numel < 1024 and num_experts <= WARP_LANES` 不看 `numel > 0`）↔ §0.1 边界 ②、§6.4 末行 ✓，
  且层测 `no-routes-few-experts`（`tests/ops/test_moe_permute_align.py:126` 的 `(0,8,8,16)`）证实该分支存在；
  ③ 空 expert 区段不写 ✓（源 322 行变长循环 trip count = 0）；④ `numel % 32` 尾 lane 守卫 ✓（源 326–327 行）。
  **expert 内部顺序未定义**这一前提有独立证据：`_assert_route_equivalent`（测试 58–84 行）先按 `(expert, 路由索引)`
  归一化再比对，docstring 明写「up to route order within each expert」⇒ §0.1 / §1.6.1 A5 的「契约级等价」前提成立。
- **算法解读完整性（pass）**：§0.3 步骤表 1–8 与源码语句一一对应，无遗漏、无臆造。抽检三处易漏点均正确：
  ① 步骤 4 的行前缀是**列内串行**（源 288–293 行），`tokens_cnts[wid] = 0` 为列 `wid` 的零 halo，故「行 k = lane k 首 rank、
  行 32 = cnt_e」成立；② 步骤 8 的散射由**全部 32 个 worker**执行（源 325 行无 `wid < E` 守卫），每 worker 以自身行 `wid`
  为游标，行 `wid` 是 lane `0..wid-1` 的累计 ⇒ 各 worker 的槽区间互斥，§0.3 步骤 8 的写法与源码一致；③ 步骤 6 的越界 block
  写 `E` 在 `total` 已定之后（源 306 行 sync 之后）执行 ✓。host 侧逻辑（`max_padded`/`max_num_blocks`、三个 `torch.empty`、
  `target_cache`、`raise`）在 §0.3 步骤 9 完整列出，§4.7 与冻结 wrapper 的调用形态（`self._small_batch_fn()` 无 launch 参数，
  源 464–466 行）一致 ✓。
- **优化手段覆盖完整性（pass）**：§0.4 的 O1–O8 覆盖源码的全部显式优化意图（单 launch 融合、每 lane 私有行、行轴前缀
  转置、fill/worker 分工、rank 即游标、stride-32 分片、构建期特化、变长 `expert_ids` 写），并且「本 kernel 不含 warp shuffle /
  `T.copy` / 流水」的否定判断与源码一致（shuffle 只出现在 align 段 100–111 行）。算法层/实现层划分（O1/O5 算法层）
  与 §0.5 处置自洽。
- **耦合性判定合理性（pass）**：§0.5 十条（D1–D10）全部为四态处置 + 依据。抽检三条承重判定，均与平台事实一致：
  ① D3 舍弃「每 lane 私有行」的依据是「数据相关 UB 读改写会插核内 barrier」—— 我用探针 M1/M5 亲测复现（见机械复核）；
  ② D6「填充不能外移」依据 `CG-950-batch-barrier`（`knowledge/capability-gaps.md:3-5` 确载跨 block barrier 未认证）+
  写域重叠论证（`[0,P)` ⊃ `[0,total)`），逻辑成立；③ D10 舍弃 `threads=`/`sync_threads` 依据 `ascend/language/kernel.py`
  的注释（1-D 核网格、无 `threads`）与 `references/programming.md` API 表一致 ✓。舍弃项（D3/D10）均写明理由，无静默丢弃。
- **NPU 重设计可行性（pass）**：R1–R6 每项均有「源方案 → NPU 新算法 → 语义保持论证」三要素，且满足平台约束
  （一维 `T.Kernel(1)`、单单核、无跨核 flag、UB 8,964 B ≪ 253,952 B、全循环上界构建期常量）。语义保持论证经我独立复核：
  R2（`cnt_e = Σ[x_i == e]` 与私有行求和逐位相同，整数）、R3（整数结合律 + 左 halo 恒 0 + 补位 lane `p_e = 0` ⇒
  `s_cs[e] = total` 为定义值）、R4（`cs` 非减且区间铺满 `[0,total)` ⇒ `#{e: cs_e ≤ pos} − 1` 即最大满足者，零长度区间不改变结论）、
  R5（游标从 `cs_e` 单调 +1 恰 `cnt_e` 次 ⇒ 覆盖 `[cs_e, cs_e+cnt_e)`、两两互异、界内）**均成立**。
  R1 的「填充与散射同核」是唯一可行解（跨核无屏障），置换不改变结果。
- **设计与决策一致性（pass）**：§1.4 的 P0–P6、§3.1/§3.3 伪代码、§6 循环表均为 §0.6 重设计后的形态；未出现「§0.5 判重设计、
  §3 照抄源方案」的脱节（例如 §3.3 的 P2 是 lane↔expert 比较累加，而非源式 `tokens_cnts[(wid+1)*E+eid] += 1`；
  §3.3 P5 是 rank-count，而非变长 expert 主序写）。§3.3 伪代码与探针 `final_single_core()` 结构逐句对应 ✓。
  §0.6 末尾的 O1–O8 → R1–R6/D9 承接表逐项给出，量化交接（如 O8 → R4 的「少写让位于可并行」，§1.6.0 R3 有数字）✓。
- **golden 独立性（pass）**：§8.1 的 golden 移植 `examples/TileOPs/tileops/workloads/moe.py::ref_permute_align`（31–85 行：
  `bincount` 式计数 + `cumsum` 式补齐前缀 + **二分**求 owner + 索引升序放置），不复用本设计的任何技巧（无 lane↔expert 比较累加、
  无 4 路累加器、无 HS 扫描、无 rank-count、无原子游标、无分块填充）。我逐行核对了 golden 的 `torch.searchsorted(cs[1:], pos, right=True)`：
  `cs[0] = 0 ≤ pos` ⇒ 返回值 = `#{e ∈ [0,E] : cs_e ≤ pos} − 1` = 期望的 owner，与 `ref_permute_align` 的二分等价 ✓；
  尾部 `sorted_token_ids` 填 `numel`、`expert_ids` 填 `E` 与 `ref_program`（`workloads/moe.py:139-152` 的 `torch.full(…, numel)` / `torch.full(…, num_experts)`）一致 ✓。

### 1. API 可行性 — pass

§3.2/§3.4 的每条 API 我都在**本机选中工具链**（`/workspace/venv-ascend950/.../tilelang/ascend`，tilelang 0.1.15，
`target="ascend"`）核对，并区分「源码/头文件证据」与「执行证据」：

| API | 我的核对结果 |
|---|---|
| `T.Kernel(1)`（无 `threads=`） | `ascend/language/kernel.py` 模块 docstring 与 `__all__` 确认一维核网格、`threads=` 被 Python 层拒绝；`references/programming.md` API 表同述 ✓ |
| `T.SimtVF(threads=…)` + `T.Parallel` | `frame.py:82-122` 存在；探针 M9 降级为 `asc_vf_call<…>(cce::dim3(32/64/256))` ✓（int32 + 32/64/256 lane 形态均 COMPILE_OK；真机 lane 吞吐待 Stage 3） |
| `T.alloc_shared` / `T.alloc_local([k])` | `allocate.py`（UB `shared.dyn`）；探针降级为 `int32_t acc[4]` / `int32_t cnt[1]` ✓ |
| `T.copy(src_region, dst_region)`（运行期长度 region） | 探针 M9 降级 `asc_copy_ub2gm_align(…, min(1024, P-t*1024)*4, …)` ✓（`programming.md` 亦记录「局部拷贝用显式 region」） |
| `T.fill(buf_region, v)` | 探针 M8 降级为 `int2` 向量 store 的 `#pragma unroll` 循环 ✓ |
| `T.atomic_add(shared[i], 1, return_prev=True)`（int32） | `language/atomic.py:211-217` 有 `return_prev`；codegen `atomic_add_ret_elem_op` → `asc_atomic_add(ptr, val)` 表达式形式（`codegen_ascend.cc:2502-2540`）；CANN 头 `device_atomic_functions.h:25` 的 `__ubuf__ int32_t*` 重载（arch 3510）；探针 K6/M9 复现 ✓。**真机语义与吞吐无执行证据** ⇒ 与设计一致地保留为「编译/头文件已确认，真机待 Stage 3」，并已配 §1.6.3 备选与裁决计划 |
| `T.if_then_else` / `T.min` / `T.ceildiv` / `>>` `<<` | `tilelang.language.common` 导出；探针降级码可见 ✓ |
| `T.cumsum` | 探针 K7 `COMPILE_FAIL`（我重跑复现），报错文本与 §3.4 逐字相同 ⇒ 「不可用、自实现 HS 扫描」的否定论断成立 ✓ |

§3.4 另列出四条「不可用/不使用」的反查记录（`T.alloc_var(...)[0]`、`prim_func` 内 `while`、`T.reduce_sum(dim=0)`、GM 原子），
均指向本平台已有实证或本设计不需要，未拿其他后端同名 API 当证据 ✓。编程模式 `simt` 的偏离依据（§2.2）经核对成立：
`simd.py::vscatter` 的 codegen 强制 `__ubuf__` 基址（不能做 GM 散射），`simd.py` 无 atomic 导出（无寄存器面原子），
故 `T.alloc_var`/SIMD 路线无法表达本算子的两项核心操作；同 op 的 scatter 路径亦以 SIMT 落地（`_make_scatter_kernel/DESIGN.md` §2.2）。

### 2. 内存层级规划 — pass

- **搬运路径完整（pass）**：§4.4 给出 GM→UB（P1 `T.copy` 4·numel B）、UB→GM（P0 分块 4·2024 B 级 bulk store、P5 4·NB B、
  P4 单元素 `npost`）、以及 P6 的逐 lane 标量 store + GM `flat[idx]` 标量读；每步都与 `programming.md` 的 UB/GM 路径一致，
  无 L1/L0 申请（设计无矩阵运算，`T.alloc_l1/l0*` 未出现）✓。
- **UB 预算（pass）**：机械复核段已独立复算 8,964 B / 全用例峰值 9,220 B；用探针降级码的 `buf_dyn_shmem` 偏移反证实际分配
  约 7.3 KB（M9, chunk=640），无隐藏膨胀；即使按 `CONST-capacity-910B2C` 的 auto-multi-buffer 1.7× 上界（910B2C 实测、
  跨型号外推，设计已标注）也仅 ≈15.7 KB（6.2%）。容量非瓶颈 ✓。
- **L0C 容量**：不适用（无 Cube）✓。**Buffer 层级标注（pass）**：§4.3 六个缓冲全部标注 UB 并给出 shape/dtype/生命周期；
  §4.3 另明确「无 GM scratch」（`cumsum` 只在 UB）——这与两段路径（align 的 `cumsum[E+1]` GM 交接）形成正确对比 ✓。

### 3. Tiling 策略 — pass

- **Block 划分（pass）**：§5.2 给出五个轴的构建期取值（核网格 1 / expert 轴 W=32 / P2 迭代 `CHUNK/4 ≤ 256` / P6 轮数 ≤ 32 /
  P5 `BLK = align_up(NB,64)` / P0 `FILL_TILE=1024`）与理由；我按 §4.2 的十档 `(numel,E,B)` 复算了 `P/NB/CHUNK/BLK/填充块数/P6 轮数`
  **全部正确**（如 `(1023,32,64)` → P=3,102、NB=49、CHUNK=1,024、BLK=64、填充 4 块、32 轮；(1023,8,16) → P=1,158、NB=73、BLK=128）。
- **对齐约束（pass）**：本算子无尾轴 dtype 对齐问题（int32 标量 store + 1024 元素 tile 拷贝，4 B 元素天然满足）；
  `W=32`、`FILL_TILE=1024`、`BLK` 均 64/32 对齐 ✓。
- **非整除处理（pass）**：§5.2 给出六条（填充尾 `n = min(FILL_TILE, P-off)`、`s_x` 尾 `-1`、`BLK ≥ NB` + `b*B < total` 掩码 +
  `expert_ids[0:NB]` 写回、P6 `idx < numel`、`B` 非 2 的幂回退、`numel=0` 构建期折叠），不依赖 `pad_value`/`data_select`
  （`programming.md` 明确二者执行行为未验证）✓。探针降级码逐条印证（M9 的 `min(...)` 尾块、M11 的 `numel=0` 折叠）。
- **分核策略（pass，符合 `core-split-strategy.md` §1 三要素）**：① 逻辑核数 = `G_alg + G_sc + G_fill = 1`，并给出「三段各自
  不可分片」的理由（顺序依赖链 / 写域重叠 / 无跨核屏障）；② 物理核数**实查**（`torch.npu.get_device_properties` 的
  `cube_core_num=28 / vector_core_num=56`，并交叉 `statectl platform` 的 `observed_cores` 与 `core_source`，我复核了该 JSON）；
  本算子只用 AIV ⇒ 基准 56；③ 规模判定 1 ≤ 56 且给出「即使摊到 56 核也只省 < 2 µs、跨核归约不可用」的量级依据。
  「单核是正确结论而非未适配」的论证成立 ✓（此点亦被设计方写进 Stage 1 复盘的价值点）。

### 4. 技术约束检测 — pass

- **三维 Kernel**：不涉及（一维 `T.Kernel(1)`）✓。
- **GEMM 非整除 / L0C 溢出**：不涉及 ✓。
- **平台约束逐条（pass）**：§3.5.1 的 11 条限制核对与我按 `references/programming.md` 的复查一致 —— 一维核网格、无跨 block
  barrier/flag（`CG-950-batch-barrier`）、无 warp/shuffle、数据相关 UB 下标触发的核内 barrier（已规避且由探针证实）、
  UB 253,952 B、尾块显式 region、`T.cumsum` 不可假定可用；「NPUIR 专属规则不适用」亦正确（未用 NPUIR 规则，
  未以 `examples/elementwise/atomic_add.py` 之类他后端文件为证据）✓。
- **warn（建议级，不阻塞）**：`BLK = align_up(NB, 64)` 没有上界讨论。`block_size` 由调用方给定且 §0.2 明写「无上界」，
  当 `B` 很小（如 `B=1`）且 `numel` 接近 1023 时 `NB → numel`，`BLK` 可达 **1,024**，`T.SimtVF(threads=1024)` 是探针从未
  覆盖的宽度（32/64/256）。项目现有用例不会命中（`bs1` = `(64,8,128,1)` 因 `E=128 > 32` 走两段路径；本设计 L0-9 为 `(64,8,1)`
  ⇒ BLK=64），故不阻塞；建议在 §5.2 补一句「`BLK ≤ 1024`（B=1 时随 NB 增长）」并在 Stage 3 对 `BLK` 加上界（或改
  `Parallel(64) × serial(⌈BLK/64⌉)` 的 lane 分组），避免极端 `B` 触发未验证的线程宽度。

### 5. 循环与同步 — pass

- **循环结构（pass）**：§6.1 的 L1–L9 表把每个循环的构造（`T.serial` / `T.Parallel` / python 级展开 / `T.fill` region）写全，
  并对「不使用 `T.Pipelined`（无多 chunk 循环）」「不使用 `T.Persistent`（单核无任务循环）」「不用 `T.serial` 处理 lane 轴」
  给出理由 ✓。HS 的 5 步用 python `for d in range(5)` 展开（`@T.prim_func` 内 `while` 会报 `Infinite while loop detected`），
  探针降级为 `for (d=0; d<5; ++d) asc_vf_call<…>(…, d)` ✓。
- **同步与模式匹配（pass）**：§7.1/§7.2 声明「零显式同步、零跨核同步」，逐条列出 8 个依赖及自动流水事件；我用探针 M9 的
  降级码核对了三条关键链：`V(s_fill) → MTE3(填充)`、`MTE2(flat 载入) → V(P2)`、`MTE3(填充末块) → V(P6 散射)` —— 全部存在且
  落在正确位置（尤其填充→散射的写-写顺序，见机械复核）。P5→P6 的 `s_cs` 读→原子写依赖为同 pipe 程序顺序 ✓。
  §7.3 给出三层退路（恢复设计形态 → `ascend_pipe_barrier("PIPE_ALL")` → 关闭 auto_schedule 仅调试），与
  `execution-patterns.md` 的 PL-950-manual-sync 证据一致 ✓。
- **尾块处理（pass）**：§6.4 覆盖 8 类尾块（填充尾、路由尾 `-1`、散射尾、`NB % 64`、`B` 非 2 的幂、`B=1`、`E<W`、`numel=0`），
  逐条与 §3.3 伪代码一致；`E < W` 的补位 lane 纪律（比较永不成立 ⇒ `p_e = 0` ⇒ `s_cs[e] = total`，上游归约上界取 `E`）
  由 verify_equiv 的判别性对照（填 UB 垃圾时界 E 全对、界 W 全错）证明是承重不变式 ✓。
- **warn（建议级，不阻塞）**：§3.5.1 的措辞「热循环（P2/P6）**零 UB 下标访问**」不精确 —— P2 必然读 UB `s_x[i*UNROLL+j]`
  （循环下标，非数据相关），P6 才完全不读 UB。承重属性是「**零数据相关** UB 下标访问」（§9.1 风险 3 的表述是对的）。
  建议统一为「零数据相关 UB 下标访问」，避免 Stage 3 按字面理解而把 P2 的 `s_x` 读改掉/绕开。

### 6. 验证方案 — pass

- **Golden 函数（pass）**：§8.1 含完整 PyTorch 参考实现草案，且独立性论证成立（见维度 0 末项）；与冻结参考
  `ref_permute_align` + `ref_program` 的补齐语义（`sorted` 补 `numel`、`expert_ids` 补 `num_experts`）逐项一致 ✓。
- **L0 测试计划（pass）**：§8.3 的 9 例覆盖每条代码路径与每个边界：`numel < 32`（尾 lane）、`numel = 32`（整除）、
  `CHUNK=256`、`E=32=W`（无补位 lane）、`BLK=128`、`numel=0`、`B` 非 2 的幂（48）、`B=1`；每例都调用 golden 并用
  `_assert_route_equivalent` 判定，另加「非路由槽位恰为 `numel`」的额外断言 ✓。9 例中 8 例直接对应
  `tests/ops/test_moe_permute_align.py` 的真实用例（`tiny-bs4`/`single-token`/`small-bs16`/`sb-numel200`/
  `sb-numel600-maxexp`/`routes-1023`/`no-routes-few-experts`），`(100,8,48)` 与 `(64,8,1)` 为本路径自设边界
  （层测对应项 `bs-not-pow2`/`bs1` 因 `E=128` 走两段路径，设计未误称其覆盖本 kernel）✓。
- **精度标准（pass）**：§8.2 对所有输出取 `torch.equal`（`atol = rtol = 0`），依据是「全整型、无舍入，A1–A5 已机器验证」
  —— 对 int32 索引数组，零容差是正确口径，与 skill 的 dtype 默认容差规则不冲突（该默认针对浮点）。§8.2 另声明值域契约
  （`0 ≤ flat < E`）与 `E ≤ 32` ⇒ 原子不会越出 `[0,32)`，与源码一致（源算子同样不检查越界）✓。
- **warn（建议级，不阻塞）**：§8.3 末段「L1/L2 覆盖 manifest 12 条 workload 中 E 小的档位」与事实不符 ——
  `examples/TileOPs/tileops/manifest/moe.yaml` 的 12 条 workload 的 `num_experts` ∈ {384, 256, 128}，**全部 > 32**，
  `numel` 多为 8/256/4096，故**没有任何 manifest 或 benchmark 行会进入本路径**（`bench_permute_align` 由 manifest 驱动）。
  建议改为「manifest/bench 均不覆盖本路径；本路径的正确性由 L0/L1 的 test-derived shapes 覆盖，Stage 4 的性能 baseline
  须用 §1.6.3 裁决计划给定的 `(1023,32,64)`/`(600,8,64)`/`(200,8,16)` 自建 shape 采集」。

### 7. 完整性与一致性 — pass

- **章节齐全（pass）**：§0.1–§0.7、§1.1–§1.6.3、§2–§8（含 §8.1–§8.3）、§9–§11 齐全；§1.6 四个子节齐全 ✓。
- **无占位符（pass）**：全文检索无模板占位标记、待办标记或未完成标记（逐项检查占位符花括号变量、待办关键字、
  未完成字样，均 0 命中）✓。
- **同类引用真实（pass）**：Glob 核对 §3.5.3/§0.7/§1.6.0 引用的路径全部存在 ——
  `knowledge/PL-950-row-primitives.md`（+`repro/PL-950-row-primitives.py`）、`PL-950-vector-execution.md`、
  `execution-patterns.md`（+`repro/PL-950-atomic-splitk.py`）、`kernel_template.py`、
  `examples/TileOPs/tileops/kernels/moe/moe_permute_align/_moe_permute_align_fwd_kernels.py`、
  `examples/TileOPs/tileops/workloads/moe.py`、`examples/TileOPs/tests/ops/test_moe_permute_align.py`、
  同 op 的 `_make_align_kernel/`、`_make_scatter_kernel/DESIGN.md`。文档细节引用亦可核对：PL-950-row-primitives 的
  「64 线程 / 24 项」与原文一致；`bottlenecks.md` 确载「当前没有 950 真机验证的瓶颈模式或收益阈值」；
  `CG-950-batch-barrier`、`TRAP-950-simd-shared-reduce`、`PL-950-simd-reduce`、`PL-950-manual-sync` 均实存 ✓。
- **内部一致（pass）**：API 映射（§3.2）/ 内存规划（§4.3–4.5）/ Tiling（§5.2）/ 循环（§6.1）四处的 buffer 名、shape、
  迭代上界互相一致；§11.1 的三输出契约表与 §0.1、§4.2、§8.1 一致；`expert_ids` 尾段写 `E` 在三处（§4.2、§9.1 风险 9、§11.1）
  表述一致 ✓。
- **warn（建议级，不阻塞，两处量化/标签）**：
  ① §0.6 R1 的代价句「填充在单核上 **≤ 3 次** 4KB bulk store」与最大档 `P=3,102`（§4.2 表列「填充块数 = 4」）矛盾，
  应为 **4 次**（3,102 = 3×1024 + 30）；该数字被 §1.6.0 R3 的「填充 11 条指令」沿用，量级不受影响，建议直接改为 4 次。
  ② verify_equiv 的行标签与 §1.6.1 表不统一（脚本 `A6 idle lanes`/`A7` vs 设计「A3 补位 lane」/「A6 哨兵填充」），
  建议补齐标签映射，便于 Stage 3/4 复跑时对账。
  另记录（非问题）：设计方自述曾误读 `.migration_state.json` 的前 5 行；我检索全文未发现 DESIGN.md 对该文件的任何引用或
  依赖，判断不影响设计内容，仅作留痕。

### 8. 算法优化分析 — pass

**§1.6 章节存在性（pass）**：`1.6.0`（调研四问 + 候选表 + 复杂度表 + 亲和表 + 调研结论 + 「无更优替代」范围）、
`1.6.1`（A1–A6 四要素 + 机器验证表 + 结论）、`1.6.2`（V1–V13）、`1.6.3`（候选矩阵 + 评分 + 实验裁决 + 布局结论）
四个子节齐全且内容非空 ✓。

**算法调研存在性与完整性（pass）**：R1 候选表 C1–C8 **含基线 C1**；R2 在线算法给出**明确「有」**（C3 单遍 staging）并说明
淘汰依据（128 KB 桶 + 每 token 一个 barrier）；R3 复杂度表覆盖四口径（FLOPs/指令、GM Bytes、`flat` 扫描遍数、片上中间缓冲）
且基线在表；R4 逐候选六维评估（单元匹配/容量/对齐/静态边界/流水融合/跨核）+ 设计期 roofline；调研结论明确选 C2 并说明
「为何不是 C1/C3/C4/C5」；「无更优替代」写明五项检索范围（含 `knowledge.algorithm_candidates` 的 7 族）✓。

**调研结论正确性（独立复核，pass）**：

- **候选覆盖/负向断言复核**：我打开 `knowledge/algorithm-candidates.md`，其「已有最小实现」表恰为 elementwise/broadcast、
  reduction、softmax、RMSNorm-LayerNorm、GEMM、attention、GEMM+ReLU **7 族**（原文第 5–15 行），**无索引分发/直方图/前缀/
  排序类条目** ⇒ 设计「参考表无命中、候选集合由源码证据 + 结构判据构成」的陈述成立，不属于「漏掉已知候选」。
  进一步的负向复核：R2 未断言「无在线变体」而是明确「**有** C3、因容量与 barrier 淘汰」，我用探针 M1/M5 亲自复现了
  该淘汰依据（UB 读 + 原子 ⇒ 逐轮 `asc_syncthreads`；GM 读 ⇒ 无）；C6 的「无 sort API」与 `ascend/language/` 无 sort 导出
  一致；`T.cumsum` 不可用由 K7 复现。**未发现被漏掉的已知候选**。
- **复杂度算术复算（pass）**：我用代表 workload `numel=1023, E=32, B=64` 独立复算 ——
  GM 流量 `4,092×2(读 flat) + 12,408(哨兵 3,102×4) + 4,092(路由 1,023×4) + 196(ids 49×4) + 4(npost) = 24,884 B` ✓；
  C3/C5 少一遍 `flat` 读 ⇒ `20,792` ✓；指令 `P2 = (CHUNK/4=256)×4×3 = 3,072`、`P6 = 32×4 = 128`、`P5 = 32×49/32 = 49`
  （表列 64，含 select/add，差异可解释）、`P4 = 110`、`P0 = 11`，合计 `3,385` ✓（与「3.4K」一致）；
  发射项 `3,385 / 1.4e9 = 2.42 µs` ✓；流量项 `24,884 / 43.1 GB/s = 0.577 µs`、`/18.4 GB/s = 1.35 µs` ✓；
  容量项 `8,964 / 253,952 = 0.035` ✓；下界 `max(0.58, 2.42, 0.035) ≈ 2.4 µs` ✓。**中间缓冲未漏计 GM 往返**（本设计无 GM scratch，
  §4.3 已声明；对比 C8 的 `cumsum` 交接 132 B 已计入）。
- **亲和性/淘汰依据复核（pass，含亲自核对弃选前提）**：C3 淘汰引用的 128 KB × 1.7 = 217 KB 逼近 248 KB —— 我核对
  `CONST-capacity-910B2C`（pattern-library `constants.md:8-25`）确认 UB 192 KB @910B2C 与 auto-multi-buffer ~1.7× 膨胀条目
  **实际存在**，设计标注了「910B2C 实测、跨型号外推」✓；`CONST-mte2-degradation` 条目（`constants.md:82-101`）确认
  L2 驻留口径 **43.1 GB/s/core** 与 HBM 口径 **18.4 GB/s/core** 两个数字确为该条目所载，设计引用准确且标注了跨型号外推 ✓；
  C4 的「原子总数不比 C2 省」+「吞吐未实测」与 R4 判断自洽；C7 的 one-hot 构造代价（32K element-ops ≈ 1,024 条指令 + 32 KB
  中间态 + L1/L0 分形搬运）按 `gemm-tiles.md` 的路径约束成立 ✓。
- **并列候选裁决路径（pass）**：C2 vs C5 的裕度依赖两个未实测常数（UB int32 原子吞吐、1,023 步标量链每步代价），设计
  **没有纸面拍板**，而是给 §1.6.3 的主选 + 备选 + 裁决计划（见下）✓。

**调研→设计一致性（pass）**：§1.4 的七阶段（P0–P6）严格对应 §1.6.0 选定的 C2（lane↔expert 比较累加 + 5 步 HS + rank-count +
游标原子 + 同核前序填充）；§3.1 的公式拆解以 §1.6.1 的 A1–A5 为输入（表内每阶段直接引用 A 编号）；§6.1 的循环结构与
§1.6.2 结论一致（被判「已替代」的点在 §6 中不再出现逐元素标量循环）。迁移侧：§0.5/§0.6 的处置（D3 舍弃、D4 等价替换、
D5 保留+替换、D6 重新设计、D8 重新设计、D9 保留、D10 舍弃）与 §1.6.0 选族一致；源 O1–O8 的意图承接逐项列出，
**无静默丢弃**（O4 的并发意图让位于硬约束时给了量化说明）✓。

**数学等价优化完整性（pass，四要素逐项）**：A1 移位补齐、A2 rank-count、A3 HS 扫描、A4 四路累加、A5 游标原子，
每项都有「原式 → 优化后 → 等价性论证 → 收益量化」；A6（哨兵填充向量化）单列并指向 §1.6.2 V1；「无进一步公式化简空间」有
结论与依据（五处结构已处理，剩余交给 §1.6.2/§5）✓。

**等价性论证独立推演（pass，逐项我的结论）**：

- A1：`cnt ≥ 0`、`B = 2^p` ⇒ `(cnt + B − 1) >> p = ⌊(cnt+B−1)/B⌋ = ⌈cnt/B⌉`，再 `<< p` 乘回 ⇒ 恒等，无溢出（`cnt ≤ 1023`）✓。
- A2：`cs` 非减、`[cs_e, cs_{e+1})` 互斥铺满 `[0,total)`，`cs_0 = 0 ≤ pos` ⇒ 满足 `cs_e ≤ pos` 的 e 构成前缀 `{0..k}`，
  `k+1 = cnt`，而 `pos < total` 时 k 即包含 pos 的区间左端索引（**零长度区间不改变最大满足者**，我按 `cs = [0,0,64,64,128]`
  手算验证）；`pos ≥ total` 两式同判 `E` ✓。
- A3：我按 W=32、`p_e` 任意手推 5 步 HS：每步 `s[e] += s[e−2^d]`（`e < 2^d` 取左 halo 0），5 步后 `s_scan[W+e] = Σ_{e'≤e} p_{e'}`
  = `cs_{e+1}`；左移一位得 `cs_e`（`e=0` 特判 0），`cs_W = cs_E`（因 `e ≥ E` 的 `p_e = 0`）⇒ 与串行前缀逐位相同 ✓。
- A4：整数加法结合律；`s_x[i*UNROLL+j]` 覆盖 `[0, CHUNK)` 恰好一次；`cnt_e ≤ 1023 < 2^31` 无溢出 ✓。
- A5：契约级等价 —— 每个 expert 恰有 `cnt_e` 次原子自增（P6 对 `idx ∈ [0,numel)` 恰好一次 + 输入契约 `flat < E`），
  游标起始 `cs_e` ⇒ 槽集恰为 `[cs_e, cs_e+cnt_e)`、两两互异、界内；expert 内部顺序变为「轮次到达序」，而该顺序按
  `_assert_route_equivalent` 的归一化比对属**契约未定义** ⇒ 等价成立；未被占用的槽保持 P0 的哨兵 ✓。
- 与 §8.2 精度标准一致：全整数、`atol = rtol = 0` 口径自洽（非「容差内等价」）✓。

**向量化替代覆盖完整性（pass，与 §3.3/§6 交叉核对）**：我把 §3.3 伪代码与 §6.1 的 L1–L9 逐点对照 §1.6.2 的 V1–V13，
**13 个计算点全覆盖**（7 处替代 = V1/V2/V4/V5/V6/V7/V8；6 处保留 = V3/V9/V10/V11/V12/V13），且没有「§6 仍有未被
§1.6.2 涵盖的逐元素标量循环」的情况：`num_tokens_post_pad` 写（V9，block 级单元素）、`off/idx/b*B` 下标算术（V10）、
host 常量（V11）、`numel=0` 构建期分支（V12）、`s_x` 尾 `-1` 填（V13）逐项有据；P5 的内层 `serial(E)` 与 P6 的轮循环
分别落在 V6/V8 的「每 lane 归约轴 / lane 并行轮次」描述内（与 V3 的归约轴判据同构）✓。
**不残留随 `numel/E/NB` 增长的逐元素标量循环** —— 我用降级码复核了这一点（M9 的 P2 为 `for i<160 { for j<4 …}` 的
lane 并行体、P6 为 19 轮 lane 并行）。

**不可替代理由充分性（pass）**：V3（P2 归约轴）的理由具体到本算子：改 token 并行即需对 `cnt[eid]` 做数据相关 UB 读改写，
实测插逐轮 barrier + `[ThreadSync]` 告警（我用 M1 复现），且归约轴与 lane 轴只能一个占 lane —— 不是「实现简单」式空话；
V9–V13 的理由分别是 block 级元数据、任务映射算术、host 元数据、构建期折叠、以 ≤63 元素常量填换取热循环无守卫 ✓。

**替代 API 佐证（pass）**：`T.Parallel`/`T.copy`/`T.fill`/`T.alloc_local`/`T.atomic_add` 的佐证见维度 1（同口径，
本机选中工具链的源码 + 探针降级码）✓。

**向量化轴与布局决策完整性（pass）**：§1.6.3 按「gather/scatter/索引分发类」枚举 L1（lane↔expert 垂直扫描）、
L1'（lane↔token）、L2（lane↔expert 保序计数器，零原子）、L3（lane↔slot 反查）、L4（token×expert 二维比较 + `reduce_sum`）、
L5（源结构直译）、L6（常量 tile 填充），覆盖了「归约类水平/垂直 lane 映射」与「scatter 类批量粒度」两类必选候选；
逐候选有量化评分（指令数、尾 lane 浪费、UB 预算、深度）；弃选有量化理由（L3 多 1,023 步反查、L4 命中
`TRAP-950-simd-shared-reduce`、L5 64 个 barrier + ThreadSync 告警）——**无「未考虑」式论证**，也**没有把 I/O 契约
（`sorted_token_ids[sorted[slot]] = idx` 的写端散射）当作核内布局依据**，而是把「游标放在哪个轴」作为设计变量评估 ✓。
迁移侧对照了源码的隐式轴选择（源 lane↔token + 私有行），并明确其不可用的证据 ✓。
**弃选前提核对（pass）**：L4 的 `TRAP-950-simd-shared-reduce` 与 L5 的 barrier 前提我都亲自打开文档/探针核对（见维度 1、
机械复核），未发现与文档或实测矛盾；也没有把核内重排写成 GM 级 transpose 之类的前提错误。

**实验裁决模式完整性（pass）**：判定裕度依赖两个**未实测常数**（UB int32 原子吞吐、1,023 步标量链代价，设计已标注
「未实测假设」并说明设计期无法标定），因此须三件套 —— 设计提供：① 主选（L1+L1'+L6 = §3.3 形态）；② 备选 L2
**结构完整**（缓冲形状不变、`T.alloc_local([1])` 计数器、P6 改 `Parallel(W) × serial(numel)`、分核三要素不变、
dtype 路径不变、代价 ≈6,300 条指令 ≈1.9×），达到 Stage 4 可直接实现深度；③ 裁决计划含代表 shape（`(1023,32,64)`/
`(600,8,64)`/`(200,8,16)`）、指标（`msopprof` Task Duration + 两个未知常数的反解方法）、**明确判定阈值**
（备选 < 主选 × 0.85 才全局翻转）、按 shape 分派方案（`numel < 512` 走备选，工厂逐 `(numel,E,B)` 特化 ⇒ 天然可行）
与回写路径 ✓。

**布局决策与实现一致（pass）**：§1.6.3 的核内布局（`s_x[CHUNK]`、`s_cnt[32]`、`s_scan[64]` 左 halo 恒 0、`s_cs[33]`、
`s_ids[BLK]`、`s_fill[1024]`）与 §3.3 伪代码的 buffer 声明、§4.5 的 UB 预算、§6 的循环轴四方一致（探针 M9 的
`buf_dyn_shmem` 偏移亦与之对应：`s_fill @0`、`s_x @1024`、`s_cnt/s_ids @1664`、`s_scan @1728`、`s_cs @1792`）✓。

**与下游章节一致（pass）**：§3.1 每个阶段的公式均引用 A1–A5（优化后公式），无「仍用优化前公式」的残留；
§6 不再出现 §1.6.2 已判「替代」的逐元素标量循环 ✓。

**warn（建议级，不阻塞，两处口径）**：
① §1.6.0 C8 行把「多 1 次 launch ≈ 2 µs 级固定开销」归到 `CONST-vector-launch-overhead` —— 该条目
（`constants.md:46-61`）实测的是「向量算子发射 ~0.5 µs/op」，不是 kernel launch 固定开销，属**不同量的外推**。
设计已标注「常数未实测」，且 C8 的定性结论（两段路径要多一次 launch + `cumsum` GM 往返）不依赖该数字，故不阻塞；
建议改写为「launch 开销 ≈2 µs（无实测常数，与 `CONST-vector-launch-overhead` 的 0.5 µs/op 发射口径不同源）」。
② §1.6.0 R4 / §5.4 / §9.1 风险 5 的「≤3.4K 指令 ≈2.4 µs」只在项目 workload 的 `B ≤ 64`（`P ≤ 3,102`）下成立；
`block_size` 由调用方给定且无上界，`P`（因而填充次数与流量）随 `B` 线性增长。建议把该界显式限定为「`B ≤ 64` 的项目
workload 档」，并注明大 `B` 时延时由填充主导（正确性不受影响，因为 `N_FILL` 是构建期常量）。

## 检视备注（非阻塞、非问题）

- 设计方在交付说明中披露曾误读任务状态文件（`.migration_state.json`）的前 5 行。我在 DESIGN.md / RETROSPECTIVE.md
  中检索未发现对该文件的任何引用或结论依赖（§1–§11 的判据全部指向源码、平台 ruleset、knowledge 条目与探针降级码），
  故记录为留痕事项，不影响本次检视结论，也不需要修订。
- `numel=0` 的构建期折叠在降级码里保留了一个空体 `__simt_vf__ __launch_bounds__(1) {}` 与一次 `dim3(1)` 调用（M11）。
  无副作用，属编译器产物；Stage 3 若需进一步压缩可忽略。
- 本报告的全部"复核/推演"结论来自：「亲自读源码 + 亲自打开平台文档/知识条目 + 亲自重跑 verify_equiv/design_calc_check +
  亲自重跑两个编译探针（无卡，`ASCEND_RT_VISIBLE_DEVICES=""`）」。未做任何真机 kernel 执行（设备由同 op 的 Stage 3 占用）。
