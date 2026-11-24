# 950PR GM↔UB 存储与搬运陷阱（batch2 实测）

本文件承载 ascend 工具链（`target="ascend"`）在 950PR 真机上实测的 GM↔UB / L0C↔UB 存储与
region 搬运陷阱。条目只陈述实测现象与可用形态；数字与适用条件内嵌，复现脚本在
`repro/`（脚本自包含，头部记录条目 id 与首验版本戳）。工具链变更后按 front-matter
`status` 与 `kb_stale_check` 触发重验。

---
id: TRAP-950-fp8-scalar-store
kind: trap
status: verified
origin_task: batch2-20261123 (paged_cache_write / _paged_cache_write_kernel Stage 3)
toolchain: TileLang ascend 0.1.15 + CANN 9.2.1, 2026-11-23
toolchain_stamp: a010a95909e4e752d0d0b602885eca4b6fab233edca2d3e540248ed71c2135dc
toolchain_id: ascend
hardware_id: ascend-950pr-9579
repro: repro/TRAP-950-fp8-scalar-store.py
provenance: 溯源见 origin_task；设备码 dump（`*(__gm__ fp8_e4_4_t*)` vs `KPages[...] = tl::float_e4m3_t`）与 L0/L1/L2/boundary 逐字节比对记录在 paged_cache_write/_paged_cache_write_kernel/RETROSPECTIVE.md Stage 3
apis: [T.SimtVF, T.Parallel, T.copy]
dtype: [fp8_e4m3, float16, bfloat16]
---

# 标量 e4m3 GM store 静默写 0（向量化退化的判定）

在 950PR（dav-3510）+ CANN 9.2.1 上，fp8_e4m3 的 GM 直写只有向量化形态正确；两条强制标量化的
写法都写 0，而编译、运行、退出码全部正常：

1. 单个 store 的并行域恰好每线程 1 元素（`block_rows * cols == threads`）；
2. 同一 `T.Parallel` 循环体内出现两个 fp8 store（K/V 双池合并写）：向量化放弃，两个 store 均
   退化为标量 e4m3 store。

同结构的 16-bit（fp16/bf16）双 store 单循环不受影响（L0-1/L1-2 通过），因此该退化只由**目标
dtype** 决定，不能由「同结构 16-bit 已验证」外推。判定依据是生成码宽度：
`*(__gm__ fp8_e4_4_t*)`（向量，可信）vs `KPages[...] = tl::float_e4m3_t(...)`（标量，本后端写 0）。

可用形态：把两个池拆成同一 `T.SimtVF` 块内的两个 `T.Parallel` 循环（每体内一个 fp8 store），
并加编译期 guard `block_rows*cols/threads >= 2`（`threads=128` 时 `block_rows*cols >= 256`）。
该形态下 L0 10/10、L1 2/2、L2 3/3、boundary 10/10，逐字节 `max_diff = 0`。

边界：结论限于 fp8_e4m3 的 GM 直写路径与所列两种标量化触发条件；不包含 e5m2、UB 内 fp8 暂存或
经 L0C 的 fp8 落盘。

---
id: CONST-950-vcvt-narrow-store-pack-b32
kind: trap
status: verified
origin_task: batch2-20261123 (mla_dense_bwd / _dq_land_kernel Stage 3)
toolchain: TileLang ascend 0.1.15 + CANN 9.2.1, 2026-11-23
toolchain_stamp: a010a95909e4e752d0d0b602885eca4b6fab233edca2d3e540248ed71c2135dc
toolchain_id: ascend
hardware_id: ascend-950pr-9579
repro: repro/CONST-950-vcvt-narrow-store-pack-b32.py
provenance: 溯源见 origin_task；探针 history_version/stage3_probe_sentinel.py 与 stage3_probe_map2.py 的逐槽映射记录在 mla_dense_bwd/_dq_land_kernel/RETROSPECTIVE.md Stage 3
apis: [T.simd.vcvt, T.simd.vsts, T.simd.pset]
dtype: [float32, float16, bfloat16]
---

# vcvt 窄化落盘：PK_B32 + 32 位 lane 掩码是唯一稠密形态

`T.simd.vcvt(src_f32x64, "float16"|"bfloat16", part=0, sat=False)` 的结果寄存器有 128 lane，
64 个结果落在**偶 lane**（每个 32 位字一个值）。因此稠密写回必须 `dist="PK_B32"`，且掩码宽度
必须是 **32 位 lane**（`T.simd.pset(32, ...)`）。

实测（int32/fp16 落盘分布，内核内 sentinel 预填后计数）：

| 形态 | 正确槽位 | 未写（sentinel 残留） |
|---|---|---|
| `PK_B32` + mask=32 | 256/256 | 0 |
| `PK_B32` + mask=16 | 128/256 | 128 |
| `NORM_B16` + mask=16 | 4/256 | — |

逐槽映射为 `[0..31 正确, 32..63 未写, 64..95 正确, 96..127 未写, …]` —— 即 16 位掩码只启用 64 个
32 位 lane 中的 32 个。`NORM_B16` / `PK_B16` 均能编译、在写到的槽上数值也正确，只是漏写一半，
所以「同进程复跑读到上次残留的正确值」会掩盖缺陷（UB 内容跨 launch 保留）。

本现象在 `_dq_land_kernel` 上表现为 192 列中 96 列从未被写；设计期把 `vsts` 的分布取成
`NORM_B16` 的依据只是「编译通过」，实现期耗 6 个探针才定位。

边界：结论限于 f32→16 位窄化后的 **UB 落盘分布**；不覆盖 vcvt 的数值语义（见
CONST-950-vcvt-rs-disable-ieee / CONST-950-vcvt-sat-false-overflow-inf）。

---
id: TRAP-950-shared-shift-copy
kind: trap
status: verified
origin_task: batch2-20261123 (moe_permute_align / _make_align_kernel Stage 3)
toolchain: TileLang ascend 0.1.15 + CANN 9.2.1, 2026-11-23
toolchain_stamp: a010a95909e4e752d0d0b602885eca4b6fab233edca2d3e540248ed71c2135dc
toolchain_id: ascend
hardware_id: ascend-950pr-9579
repro: repro/TRAP-950-shared-shift-copy.py
provenance: 溯源见 origin_task；真机 `(8,384,64)` 首跑 cumsum/npost 失配与双 buffer 修复后的 29 例 max|diff|=0 记录在 moe_permute_align/_make_align_kernel/RETROSPECTIVE.md Stage 3
apis: [T.copy, T.fill, T.SimtVF]
dtype: [int32]
---

# UB→UB 非对齐目标偏移的 T.copy 产出 lane-wrong 数据

`T.copy(src[0:N-k], dst[k:N])` 在 950PR 真机上对 k = 1, 2, 4, 8, 16, 32 全部与参考移位逐元素
不等。把移位项交给这种 copy 承载的双调（Hillis–Steele）前缀扫描**照常跑完并给出看似合理的整数**，
但第 0 步即错：`(8,384,64)` 的 `cumsum` 全错、`npost` 得 384（应为 448）。

误差形态不保证是干净的 k 位移：相邻几何里观察到提前一个 lane，因此不能按「错位 k」去补。

可用形态：**双 buffer 交替 + 零 halo，全程不发生 shared→shared copy**。数据放 `[N,2N)`、
`[0,N)` 预置 0；每步 `dst[N+e] = src[N+e] + src[N+e-k]`，奇偶步用 python 级分支选 buffer；
末态左移一位写 `s_pref[0..N]`（补位 lane 得 total）。`E_pad=384`（9 步）与 `E_pad=64`（6 步）
的 inclusive/exclusive 结果均精确。

边界：结论限于 UB→UB（`shared` 域内）带非对齐目标偏移的 region copy；原址原地扫描
（`s[e] += s[e-k]`，无 copy）不在本条覆盖范围。

---
id: TRAP-950-l0c2ub-single-aiv
kind: trap
status: verified
origin_task: batch2-20261123 (mla_dense_bwd / _bwd_kernel Stage 3)
toolchain: TileLang ascend 0.1.15 + CANN 9.2.1, 2026-11-23
toolchain_stamp: a010a95909e4e752d0d0b602885eca4b6fab233edca2d3e540248ed71c2135dc
toolchain_id: ascend
hardware_id: ascend-950pr-9579
repro: repro/TRAP-950-l0c2ub-single-aiv.py
provenance: 溯源见 origin_task；P1 最小运行期探针与 L0 13/13 记录在 mla_dense_bwd/_bwd_kernel/RETROSPECTIVE.md Stage 3
apis: [T.copy, T.dual_copy]
dtype: [float32, float16, bfloat16]
---

# 整 tile `T.copy`(L0C→UB) 只落一个 AIV 的 UB（dual_dst_mode=0）

混合 kernel（`__mix__(1,2)`）里 `T.copy(c0, ub)` 编译为
`asc_copy_l0c2ub(..., dual_dst_mode=0, ...)`，只写一个 AIV 的 UB；随后两个 AIV 都执行整 tile
向量段，其中一个读到脏数据。实测：plain copy 3839/4096 元素错、1024 个元素从未被写；
换成 `T.dual_copy` 2:1 半区分片后 0/4096 错。

可用形态：L0C→UB 一律 `dual_copy` 2:1（M 分片 `[M,N]↔[M/2,N]` 与 N 分片
`[M,N]↔[M,N/2]` 均实测可用），L1/GM 目的侧仍可保持整 tile 由两个半区拼回。该形态要求同 dtype
（fp32 中转），跨 dtype 由前端硬拒绝。

边界：结论限于 AIC 写 L0C、两个 AIV 读 UB 的混合结构；纯 Vector kernel（无 AIC 生产 L0C）
不涉及该路径。

---
id: TRAP-950-l0c-region-store-stride
kind: trap
status: verified
origin_task: batch2-20261123 (shared_expert_mlp / _gemm_basic_kernel Stage 3)
toolchain: TileLang ascend 0.1.15 + CANN 9.2.1, 2026-11-23
toolchain_stamp: a010a95909e4e752d0d0b602885eca4b6fab233edca2d3e540248ed71c2135dc
toolchain_id: ascend
hardware_id: ascend-950pr-9579
repro: repro/TRAP-950-l0c-region-store-stride.py
provenance: 溯源见 origin_task；L0-7/L0-11/L0-12/L1-t1-down 改前全错（766 元素超差、max_err 5.5e1）改后 0 超差，记录在 shared_expert_mlp/_gemm_basic_kernel/RETROSPECTIVE.md Stage 3
apis: [T.copy]
dtype: [float32, bfloat16, float16]
---

# L0C→GM 源 region 截断在非整 16 行时用 ceil16(am) 当行距

`T.copy(c0[:am, :an], C[m0:m0+am, n0:n0+an])` 在 `am` 不是 16 行组整数倍时，lowering 以
`compact_src_region.outer1->extent * row_frac`（= `ceil16(am)`）作为**源行距**而非 tile 行数，
尾块从第 16 列起全部错：`m=48, bm=32` 时 766 个元素超差、`max_err = 5.5e1`，且编译器不报错。

可用形态：**源 region 取整块、目的 region 截断** —— `T.copy(c0, C[m0:m0+am, n0:n0+an])`，改后
0 超差。`block_m = 16` 时两种写法都对，因为此时 `ceil16(am) == block_m`，所以「小 tile 上验证过」
不构成安全性证据。

UB→GM 方向的区段截断不存在该行距问题（`L0-8b-k0-tails` 在 `am=16/an=44` 双尾块下 strict equal）。

边界：结论限于 L0C 出数（`T.copy` 与 `dual_copy` 的 L0C 源）在 16 行组不齐时的源侧截断；
目的侧截断在 L0C→UB 与 UB→GM 均安全。

---
id: TRAP-950-partial-gm2ub-load
kind: trap
status: verified
origin_task: batch2-20261123 (moe_permute_align / _make_small_batch_kernel Stage 3)
toolchain: TileLang ascend 0.1.15 + CANN 9.2.1, 2026-11-23
toolchain_stamp: a010a95909e4e752d0d0b602885eca4b6fab233edca2d3e540248ed71c2135dc
toolchain_id: ascend
hardware_id: ascend-950pr-9579
repro: repro/TRAP-950-partial-gm2ub-load.py
provenance: 溯源见 origin_task；首轮 L0 15/17 与修复后 31 例全过记录在 moe_permute_align/_make_small_batch_kernel/RETROSPECTIVE.md Stage 3
apis: [T.copy, T.fill]
dtype: [int32]
---

# GM→UB 部分块 T.copy 按 32 B 上取整并覆盖相邻槽

`asc_copy_gm2ub_align` 把长度上取整到 32 B（8 个 int32）写入 UB：`n % 8 != 0` 时
`n .. ceil8(n)-1` 被 GM 越界读到的垃圾覆盖。canary 探针实测（写入值 `0x07070707`）：

| n | 被覆盖槽数 |
|---|---|
| 1 | 7 |
| 7 | 1 |
| 8 | 0 |
| 9 | 7 |
| 23 | 1 |

**UB→GM 方向不越界写**（同探针 P=1/23/18 尾块无溢出），两个方向不可互推。

后果形态：以「常量 padding + 固定热循环上界」替代边界守卫时，padding 落在 `[n, chunk)`，而
copy 与 fill 写域不相交 ⇒ 两者无依赖边，编译器不保证 fill 在后，padding 被 GM 垃圾覆盖、热循环
把垃圾计入直方图（本算子实测 `total` 15→20、1→8 的静默错算）。

可用形态：热循环上界取精确元素数（构建期 `tail = m*UNROLL + r` 拆分），UB 缓冲尾部留至少 8 元素
guard band（本 kernel 用 32）。

边界：结论限于 GM→UB 的部分块搬运粒度；不含 UB→GM 写回路径。

---
id: TRAP-950-simd-strip-pitch
kind: trap
status: verified
origin_task: batch2-20261123 (engram_gate_conv_bwd / _engram_gate_conv_bwd_kernel Stage 3)
toolchain: TileLang ascend 0.1.15 + CANN 9.2.1, 2026-11-23
toolchain_stamp: a010a95909e4e752d0d0b602885eca4b6fab233edca2d3e540248ed71c2135dc
toolchain_id: ascend
hardware_id: ascend-950pr-9579
repro: repro/TRAP-950-simd-strip-pitch.py
provenance: 溯源见 origin_task；同代码不同几何二分（seq∈{1,2,4}/8/401/408）与修复后 25 例全过记录在 engram_gate_conv_bwd/_engram_gate_conv_bwd_kernel/RETROSPECTIVE.md Stage 3
apis: [T.copy, T.simd.vsts]
dtype: [float32, bfloat16]
---

# 2D UB 目的缓冲的行距必须 32 B 对齐（否则 507035）

`T.copy` 写入 2D UB 缓冲时，目的缓冲每行起始必须 32 B 对齐：fp32 行距需 ≥ 8 元素且为 8 的倍数。
违反时**编译无告警**，真机直接 `device error type 3, error code is 507035`。

实测（同一 kernel）：`sc = (4, seq_len)` 时 `seq ∈ {1,2,4}` 与 `seq = 401` 全部 507035，`seq = 8`
通过；改为 `(4, max(8, ceil(seq/8)*8))` 后全部通过。最小复现：pitch=1 必错、pitch=8 正确、
pitch=408 正确。每行拷贝的元素数不受此限（每行 1 个 fp32 也合法，只要行距 32 B）。

诊断形态：无编译诊断的 507035 + 「同代码不同几何」二分（`seq_len*4 B` 是否为 32 B 倍数）比读 IR
更快定位；该文件的对齐契约也是 `T.simd.vsts(PAT_VL1, extent=1)` 的 32 B 行距要求。

边界：结论限于 2D UB 目的缓冲（按行条带暂存）的行距；一维 UB 缓冲与 GM 目的缓冲不涉及。

---
id: TRAP-950-simd-ub-store-load
kind: trap
status: verified
origin_task: batch2-20261123 (engram_gate_conv / _engram_gate_conv_fwd_kernel Stage 3)
toolchain: TileLang ascend 0.1.15 + CANN 9.2.1, 2026-11-23
toolchain_stamp: a010a95909e4e752d0d0b602885eca4b6fab233edca2d3e540248ed71c2135dc
toolchain_id: ascend
hardware_id: ascend-950pr-9579
repro: repro/TRAP-950-simd-ub-store-load.py
provenance: 溯源见 origin_task；同一二进制同输入 Y 的 max_diff 在 0.0/1.5e-1/6.98 间变化与加 mem_bar 后 3 连跑一致，记录在 engram_gate_conv/_engram_gate_conv_fwd_kernel/RETROSPECTIVE.md Stage 3
apis: [T.simd.vsts, T.simd.vld, T.simd.mem_bar, T.copy]
dtype: [float32, float16, bfloat16]
---

# SIMD UB 写→读必须显式 `mem_bar("VST_VLD")`（非确定性）

同一 kernel 内先用向量存写 UB、再用向量 load 读同一缓冲，若无
`T.simd.mem_bar("VST_VLD")`，结果是**非确定的**（同一编译产物、同一输入、逐次不同）：

- `vsts(dist="PK_B32")` 写的 16-bit 行经 `vld(dist="UNPK_B16")` 读出 0 / 部分 / 正确；
- 单 lane `vsts(PAT_VL1, extent=1)` 后的 `vld(..., dist="BRC_B32")` 读到陈旧标量，把整行 `g`
  均匀放大 1.38×；
- fp32 窗口写后读给出逐 lane 错值。

生成代码里 V→V 依赖没有任何 `asc_sync_*`（只有 MTE2/V/MTE3/S 之间有），所以编译期不可见。
加 barrier 后上述三种情形逐位正确，`--level all` 连跑 3 次结果一致。

边界与互补：`vsts(dist="PK_B32")` 之后**直接** `T.copy` 到 GM 不需要该 barrier（不是 V→V 回读，
编译器插 `PIPE_V→PIPE_MTE3`）；`T.alloc_reducer`/`T.Parallel` 的 UB 写→`T.copy` 出栈路径同样
不需要（见 PL-950-simt-scalar-broadcast 与 PL-950-dynamic-pipeline-2versions 的正向证据）。
判据是「同一 buffer 是否既是 `vsts*` 目标又是 `vld*` 参数」。
