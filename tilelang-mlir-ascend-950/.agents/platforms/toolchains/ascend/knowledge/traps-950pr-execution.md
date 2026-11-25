# 950PR kernel 结构与执行陷阱（batch2 实测）

本文件承载 ascend 工具链在 950PR 真机/编译期实测的 kernel 结构、网格、线程域与 AIV 标识类陷阱。
条目只陈述实测现象与可用形态；复现脚本在 `repro/`（自包含，头部记录条目 id 与首验版本戳）。

---
id: TRAP-950-simt-thread-cap
kind: trap
status: verified
origin_task: batch2-20261123 (moe_permute_align / _make_align_kernel Stage 3 patch；_make_small_batch_kernel 同族)
toolchain: TileLang ascend 0.1.15 + CANN 9.2.1, 2026-11-23
toolchain_stamp: a010a95909e4e752d0d0b602885eca4b6fab233edca2d3e540248ed71c2135dc
toolchain_id: ascend
hardware_id: ascend-950pr-9579
repro: repro/TRAP-950-simt-thread-cap.py
provenance: 溯源见 origin_task；失败/修复几何与 31 例全过记录在 moe_permute_align/_make_align_kernel/RETROSPECTIVE.md（Stage 3 patch）
apis: [T.SimtVF, T.Parallel]
dtype: [int32, float16]
---

# `T.SimtVF(threads=)` 上限 2048（LAUNCH_BOUND，编译期硬失败）

`T.SimtVF(threads=N)` 的 N 会变成 `LAUNCH_BOUND(N)` 常量，`N > 2048` 是编译期错误（不带源码
行号）：`Maximum number of SIMT threads returned by LAUNCH_BOUND must be an integer constant in
range (0, 2048]`。边界探针：1024 / 2048 `COMPILE_OK`；2049 / 2304 / 2560 `COMPILE_FAIL`。
触发几何：`E = 1100` ⇒ `E_pad = align_up(E,64) = 1152`、`2*E_pad = 2304`；`(512,1100,64)` 与
小路径 `(100,1100,1)` 同错。

可用形态：lane 数统一经 `_lane_spans(count)` 生成跨度 ≤ 2048 的多趟，索引加跨度偏移
（`count <= 2048` 时退化为单趟 `(0, count)`）。该形态在 `E_pad = 2560 / 2112` 的超限防御几何上
也逐位正确，说明跨度守卫是承重的而不只是编译修复。

分工：宽度是否「性能可用」另有一档（本批 `_make_small_batch_kernel` 只采用验证过的 32/64/256，
lane 需求超出时用「固定宽度 × serial 轮次」）；本条给的是**硬上限**。

边界：结论限于 `target="ascend"` 的 SIMT 线程域；不涉及 Cube 或 SIMD 通道宽度。

---
id: TRAP-950-grid-over-65535
kind: trap
status: verified
origin_task: batch2-20261123 (chain_speculative_sampling / _chain_speculative_sampling_kernel Stage 3)
toolchain: TileLang ascend 0.1.15 + CANN 9.2.1, 2026-11-23
toolchain_stamp: a010a95909e4e752d0d0b602885eca4b6fab233edca2d3e540248ed71c2135dc
toolchain_id: ascend
hardware_id: ascend-950pr-9579
repro: repro/TRAP-950-grid-over-65535.py
provenance: 溯源见 origin_task；失败首跑 L0 只剩 6/13（一次 abort 污染后续用例）记录在 chain_speculative_sampling/_chain_speculative_sampling_kernel/RETROSPECTIVE.md Stage 3
apis: [T.Kernel]
dtype: []
---

# `T.Kernel(grid)` 的 grid 上限 65535（65536 触发 507035）

950PR 上 `T.Kernel(grid)` 的 block 索引按 16 位携带：65535 个 block 正常完成，65536 与 131072
触发 `device error type 3, error code is 507035`，CANN 设备日志里 AIV 位图全 1。abort 与 kernel
体无关——测试体只读写 `X[bx]` / `Y[bx]`。

该限制反向约束折叠写法：本算子把 `B` 行折到 ≤ 56 个 vector core（`blocks = min(batch, 56)`，
每核 `ceil(batch/blocks)` 行）；写成 `blocks = batch` 在 `batch <= 65535` 合法、超过即 abort。
多任务/多行折叠的可用形态是「少量 block + 串行任务循环」。

**补全 facet（2026-11-24，origin_task `mla_varlen_bwd` / `_bwd_kernel`）：余数非 0 时不 abort，静默部分执行。**
上面两次 abort 的 grid（65536、131072）恰好都是 65536 的整倍数。当 `grid % 65536 != 0` 时设备**不报错**，
只执行 `grid % 65536` 个 block 并把结果写回——这是最危险的形态，因为它表现为「更快 + 部分输出为 0」：

- `mla_varlen_bwd/_bwd_kernel`，`grid = (total_kv//64 + batch) * heads`（含运行期量）：
  B=8/H=128/L=4096 ⇒ 66,560 块 → 只执行 1,024 块（`= 66,560 − 65,536`），request 1..7 的 `dq/dk/dv`
  全为 0，latency 20.75 ms（真实 ≈ 488 ms）；H=127（66,040）→ 504 块；H=126（65,520）正常。
- 逐 CTA 映射与 A/B（同内核只改 heads）记录在 `/tmp/opencode/MLA_VARLEN_BWD_8x4K_FINDINGS.md`；
  `grid % 65536 == 0` 的 abort 侧由 `repro/TRAP-950-grid-over-65535.py` 覆盖。

**可用形态**：grid 含运行期量时（如 `tiles * heads`，`tiles` 来自 token 数），编译期查不出、运行期不报错，
必须（a）host 侧发射前断言 `grid <= 65535`，越界 raise；（b）把其中一个维度切成编译期 **slab** 分多次 launch
（`grid = tiles*slab`，`h = h_off[0] + bid % slab`，`h_off` 用一元素 int32 张量逐 launch 传入，尾 slab 用
`if h < heads` 守卫使空块立即返回），这样 `bid // slab` 仍是静态除数。
参考实现：`examples/mla_varlen_bwd/_bwd_kernel/launch_plan.py`；编译级探针
`history_version/design_probe_grid_slab.py`（设备码 `block_idx >> 1` / `h_off[0] + (block_idx & 1)` /
`if (h < 3)`）。边界：slab 只切发射批次，不改变每 block 的计算体与写集（各 slab 的 head 区间不交，无需同步）。

边界：结论限于 950PR / CANN 9.2.1 的 grid 上界；不含 grid 与 AIC/AIV 映射比（grid 可超过物理核数）。
facet 补记于 2026-11-24（`mla_varlen_bwd/_bwd_kernel`），同版本戳（`a010a95…` → 当前 `7a52f2f…` 报 stale，
按 `kb_stale_check` 的降级口径复核；补记内容由该任务的设备实测支撑，见上）。

---
id: TRAP-950-aiv-sid-vector-frame
kind: trap
status: verified
origin_task: batch2-20261123 (ssd_chunk_coupling / _ssd_chunk_coupling_kernel Stage 3)
toolchain: TileLang ascend 0.1.15 + CANN 9.2.1, 2026-11-23
toolchain_stamp: a010a95909e4e752d0d0b602885eca4b6fab233edca2d3e540248ed71c2135dc
toolchain_id: ascend
hardware_id: ascend-950pr-9579
repro: repro/TRAP-950-aiv-sid-vector-frame.py
provenance: 溯源见 origin_task；3 次重跑（extern 0/8、vector 5–7/8）与生成码 `asc_get_sub_block_id()` 记录在 ssd_chunk_coupling/_ssd_chunk_coupling_kernel/RETROSPECTIVE.md Stage 3
apis: [T.Vector, T.call_extern, T.SimtVF, T.dual_copy]
dtype: [float16]
---

# 显式 `T.Vector(vector=2) as sid` 编译通过但运行值错且非确定

读取 AIV 子块号的两种写法在本工具链不等价：

- `with T.Vector(vector=2) as sid`：**编译通过、运行值错且非确定**。64 块网格下 8 次发射有
  5–7 次损坏一个 64 行半块，8 次出现 5–7 种不同结果；严格上三角仍精确为 0，因此「掩码方向对」
  的表象会掩盖它。单块冒烟（grid 个位数）全部通过，需要多块网格才暴露。
- `sid = T.call_extern("int64", "asc_get_sub_block_id")` 再 `T.cast(sid, "int32")` 进
  `T.SimtVF`：3 次重跑 0/8 失败，每次精确且确定。

同一 kernel 里若另有 **UB→L1 软件 `dual_copy`**（2:1 分片）+ 后续 GEMM，则抛 SSA 绑定冲突
`InternalError: Check failed: (!var_idmap_.count(v)) is false: Need input to be in SSA form dup sid`，
且与是否写 `as sid` 无关；GM 侧 `dual_copy`（L0C→UB 硬件双目的 / UB→GM 按 sid 分片）与自动调度
`T.Kernel` + 显式 `Vector` 帧可以共存。

边界：结论限于 950PR + ascending 0.1.15 的 AIV 子块号读取；id/帧/同步类 API 的编译证据不能替代
执行证据（本条是「编译通过但运行错」的原型）。

---
id: TRAP-950-prim-func-helper-body
kind: trap
status: verified
origin_task: batch2-20261123 (shared_expert_mlp / splitk_reduce_kernel Stage 3)
toolchain: TileLang ascend 0.1.15 + CANN 9.2.1, 2026-11-23
toolchain_stamp: a010a95909e4e752d0d0b602885eca4b6fab233edca2d3e540248ed71c2135dc
toolchain_id: ascend
hardware_id: ascend-950pr-9579
repro: repro/TRAP-950-prim-func-helper-body.py
provenance: 溯源见 origin_task；5 项断言复现（`TOTAL checks: 5, failures: 0` / `[TRAP_CONFIRMED]`）记录在 shared_expert_mlp/splitk_reduce_kernel/RETROSPECTIVE.md Stage 3
apis: [T.prim_func, T.serial, T.simd.pset]
dtype: []
---

# eager `@T.prim_func` 函数体不能抽普通 Python helper

950PR 的 eager builder 对被装饰函数体做 AST 变换；函数体里的循环与缓冲赋值不能移到普通 helper：

| 形态 | 结果 |
|---|---|
| helper 内 `for ... in T.serial()` | `TypeError: 'ForFrame' object is not iterable` |
| helper 内 `with T.serial() as i` + 缓冲赋值 | `'Buffer' object does not support item assignment` |
| helper 内 `T.simd.pset()` | `Unresolved call ir.Op("tl.simd.pset")` |
| helper 只接收调用方传入的地址/掩码并返回 `vld/vcvt/vmul/vadd` 表达式 | 可用（值表达式 helper） |

后果：`T.prim_func` 形参集合随构建期开关变化（如可选 bias）时，只能按变体各写一份完整函数体
（本算子两份 `main` 各约 70 行），`T.simd.pset` 必须在被装饰函数的调用点创建。

边界：结论限于 `target="ascend"` 的 eager `@T.prim_func`；`@T.macro` 是另一种（可用的）复用形态。

---
id: TRAP-950-scatter-eid-upper-bound
kind: trap
status: verified
origin_task: batch2-20261123 (moe_permute_align / _make_scatter_kernel Stage 3)
toolchain: TileLang ascend 0.1.15 + CANN 9.2.1, 2026-11-23
toolchain_stamp: a010a95909e4e752d0d0b602885eca4b6fab233edca2d3e540248ed71c2135dc
toolchain_id: ascend
hardware_id: ascend-950pr-9579
repro: repro/TRAP-950-scatter-eid-upper-bound.py
provenance: 溯源见 origin_task；`cumsum[8]` 16→19、3 个哨兵槽被覆写的实测与加界后 16/0 记录在 moe_permute_align/_make_scatter_kernel/RETROSPECTIVE.md Stage 3
apis: [T.atomic_add, T.Parallel, T.call_extern]
dtype: [int32]
---

# 编译器自动索引守卫按缓冲 extent 判定：`cumsum[E]` 需显式上界

多核 scatter 用 `cumsum`（extent = `E+1`）做 bump allocator 时，编译器自动插入的值域 guard 只保证
`eid < E+1`，因此源算子式 `if eid >= 0` 会让 `eid == num_experts` 的路由原子自增
`cumsum[num_experts]`（即 `num_tokens_post_pad`）并把路由索引写进 padding 区，**静默**破坏上一
阶段结果：实测未加界时 `cumsum[8]` 由 16 变 19、3 个哨兵槽被覆写；显式 `eid < num_experts` 后
`cumsum[8]` 保持 16、0 覆写。

判据：凡「缓冲区多出一个语义不同的元素」（`cumsum[E]`、`expert_ids` 尾块、
`num_tokens_post_pad`）都必须写显式上界，不能依赖自动 guard。

同一实测族的另一条：原子语句的 `else` 分支会回填 `0`，因此「原子」与「依赖其返回值的写」必须写在
**同一个** `if` 内，否则非法 lane 会写坏槽 0。

边界：结论限于 UB/GM int32 原子 + 逐 lane 数据相关索引写；不含 Cube 侧原子。

---
id: TRAP-950-core-slice-overread
kind: trap
status: verified
origin_task: batch2-20261123 (moe_permute_align / _make_align_kernel Stage 3)
toolchain: TileLang ascend 0.1.15 + CANN 9.2.1, 2026-11-23
toolchain_stamp: a010a95909e4e752d0d0b602885eca4b6fab233edca2d3e540248ed71c2135dc
toolchain_id: ascend
hardware_id: ascend-950pr-9579
repro: repro/TRAP-950-core-slice-overread.py
provenance: 溯源见 origin_task；未夹紧 row sums `[256,256,256,250]`（TPC=250）与夹紧后逐元素等于 bincount 记录在 moe_permute_align/_make_align_kernel/RETROSPECTIVE.md Stage 3
apis: [T.Kernel, T.SimtVF, T.atomic_add]
dtype: [int32]
---

# 多核分片的 chunk 长度需「本核切片」与「张量末端」双夹紧

多核 partial 直方图的 chunk 长度只夹张量末端时，`CHUNK = align_up(TPC, 64) > TPC` 必然越过本核
边界，读进下一核的 token，直方图重复计数：`TPC = 250` 时每核多算 6 个 token，row sums 得
`[256, 256, 256, 250]`（应全为 250）。

可用形态：`n = max(0, min(min(TPC - c*CHUNK, CHUNK), numel - base))` —— 本核切片与张量末端两次
夹紧；夹紧后与 `torch.bincount` 逐元素相等。

同族的设计侧要求：§5 分核模板应给出「每核/每 chunk 读取区间 ⊆ 本核写域」的显式不等式，检视维度 3
对应核对项；`align_up(TPC,64)` 形式的 CHUNK 是这类缺陷的高发形态。

边界：结论限于多核分片 + 跨核归并（partial → 第二段归并）的直方图/归约类 kernel。

---
id: TRAP-950-ub-atomic-vf-scope
kind: trap
status: verified
origin_task: moe_permute_align-moe_permute_align-20261124T171256Z (Stage 4)
toolchain: TileLang ascend 0.1.15 + CANN 9.2.1, 2026-11-25
toolchain_stamp: 7a52f2f1274df07d8f4c9c01353ce9d290f88c16e85047de0c425b6a009de4aa
toolchain_id: ascend
hardware_id: ascend-950pr-9579
repro: repro/TRAP-950-ub-atomic-vf-scope.py
provenance: 溯源见 origin_task；两次假失败（探针 `histatom`/`ubatom` 报「msprof produced no
  CSV」而真因是编译失败）记录在
  `examples/moe_permute_align_perf/perf_opt/logs/probe/mechanisms*.log` 与 `perf_opt/opt_log.md` §9.2
apis: [T.SimtVF, T.Kernel, T.atomic_add, T.alloc_local, T.Parallel]
dtype: [int32]
---

# per-lane UB 原子/数据相关访问必须写在「header 形式」的 `T.SimtVF` 内

想在 SIMT lane 上做 `T.atomic_add(s_cur[eid], 1, return_prev=True)`（或任何数据相关 UB 访问）时，
两种常见写法**编译期就失败**：

1. **局部缓冲声明在 VF 区域之外**（`with T.Kernel(g) as a:` 里 `T.alloc_local`，再另起
   `with T.SimtVF(...)`）：`error: pointer arguments to simt_vf functions must reside in
   '__gm__ or '__ubuf__' address space` —— 局部 buffer 被当作 `__simt_vf__` 形参传出。
2. **`T.Parallel` 不在任何 VF 块内**：`[TileLang Semantic Check] T.Parallel loop 'j' must be
   inside a VF block (T.SimdVF() or T.SimtVF()). Parallel loops outside VF blocks are not
   supported on Ascend NPU.`

可用形态（Stage 3 scatter 与本轮 tuned scatter 都用它，30 例 op 测试 + 综合 L0/L1 全过）：

```python
with T.Kernel(grid) as a, T.SimtVF(threads=128):     # 一个 header 同时起核与线程域
    s_cur = T.alloc_shared((e_pad,), "int32")
    eid = T.alloc_local([1], "int32")                 # 局部缓冲在 VF 内声明
    slot = T.alloc_local([1], "int32")
    for k in T.serial(n_load):
        for j in T.Parallel(128):
            s_cur[k * 128 + j] = cursor[a, k * 128 + j]
    for r in T.serial(rounds):
        for j in T.Parallel(128):
            ...
            slot[0] = T.atomic_add(s_cur[eid[0]], T.int32(1), return_prev=True)
```

配套坑（同一实测族）：`msprof`/`bench_kernel_msprof` 在**编译失败**时只报「failed to produce native
latency samples」并把子进程输出截断到 400 字符，看起来像「该 API 在本平台不可用」。判据：探针失败时
先在进程内直接 `prog(...)` 复跑一次拿完整编译栈，再下「API 不可用」的结论。

边界：结论限于 `target="ascend"` 的 SIMT 线程域（`T.SimtVF`）与 UB/GM int32 原子；不含 SIMD
（`T.SimdVF`）形态，也不含「同一 kernel 内多层 `with T.SimtVF` 嵌套」的语义（嵌套未认证）。
