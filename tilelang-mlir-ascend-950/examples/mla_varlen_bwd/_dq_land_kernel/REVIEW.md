# _dq_land_kernel（MLAVarlenBwdOp 第 3 段，packed）设计文档检视报告

## 检视结论

结论: 通过

## 检视元信息

- 文档: `examples/mla_varlen_bwd/_dq_land_kernel/DESIGN.md`（1378 行）
- 任务类型: 迁移（harness 子模式，首次设计 v0）
- 源算子: `/workspace/TileOPs/src/tileops/_kernels/attention/mla/bwd.py` 第 79–106 行（`_dq_land_kernel`），host 侧第 267–387 行（`_MLABwdKernelBase` / 两个 Kernel 类）
- 目标平台: `ascend-950pr-9579`（dav-3510，CANN 9.2.1，TileLang ascend 0.1.15，28 Cube / 56 Vector，UB 253952 B / L1 524288 B / L0C 262144 B）
- 检视维度: 9 项（0 源算子理解与迁移分析 / 1 API 可行性 / 2 内存层级 / 3 Tiling / 4 技术约束 / 5 循环同步 / 6 验证方案 / 7 完整性 / 8 算法优化分析）
- 阻塞级问题数: 0
- 建议级问题数: 4（维度 5 一条、维度 8 三条；均已在本报告对应维度内就地给出定位与改法，不阻塞进入 Stage 3）

## 机械复核（D-1/D-5）

- **verify_equiv 重跑: PASS**。`cd examples/mla_varlen_bwd/_dq_land_kernel && source ../../../.agents/platforms/toolchains/ascend/env-950pr.sh && /workspace/venv-ascend950/bin/python verify_equiv.py` → 退出码 0，打印 `[EQUIV_PASS]`，逐行结果与 §1.6.1 内嵌表**完全一致**：fp16 A1/A2/A3 = `0 ulp / 0.0 / 0`、C1 = `3857 ulp / 1.216e-03 / NOT_EQUIV`、C2/C3 = `1 ulp / 9.729e-04 / NOT_EQUIV`；bf16 A1/A2/A3 = `0 ulp / 0.0 / 0`、C1 = `1 ulp / 0.0 / WITHIN_TOL`、C2/C3 = `1 ulp / 7.297e-04 / NOT_EQUIV`；两行 `[baseline]`（golden 与 fp64 前置像）均 `ulp=0 viol=0.00e+00 nan_mis=0`。脚本断言「采纳项全 EQUIV_PASS 且每个对照项都被证伪（NOT_EQUIV 或 WITHIN_TOL）」成立。
- **design_calc_check**: `python3 .agents/tools/design_calc_check.py --design … --task-dir …` 的 JSON `checks` 摘要：
  - `r3_metrics`: **pass** — "all four R3 metrics present (FLOPs/access bytes/scan passes/buffer peak)"。
  - `ub_budget`: **skip** — "no per-level buffer tables parsed from the memory plan (advisory checker; verify manually)"；**人工已核**：`acc` 64·192·4·2 = 98304 B + `out` 64·192·2·2 = 49152 B = **147456 B ≤ 253952 B（58.1%）**，§4.3/§4.5 表与算式自洽。
  - `l0c_budget`: **skip** — "no Ascend Cube GEMM call found"；**人工已核 n/a**：§3.3 伪代码与 §5.1 均无 `T.gemm`，本设计不使用 L0A/L0B/L0C。
  - `core_split`: **skip** — "insufficient parsed dims {}; declared logical cores = 64"（解析器把 `R=64` 当成了逻辑核数）；**人工已核**：`ntiles = ceildiv(T_q,64)·H` = 80 / 65536 / 64512 / 32768，`cpt = ceildiv(ntiles,56)` = 2 / 1171 / 1152 / 586，末核余量 1131 / 538，最大失衡 +40%（smoke）/ +0.06% / 0.00% / +0.15%，与 §5.5 表逐格相符。
- **补充机械复核（本检视自补，D-3 复现）**：把本目录探针复制到 `/tmp` 重跑（`ASCEND_RT_VISIBLE_DEVICES=""`，不 launch），双 dtype 均 `COMPILE OK`、`lines=41`、`compute=['vlds_norm','vmuls','vcvt']`、`store=['vsts_pack_b32']`、`total_q_symbolic=True`，两份设备码 sha256 前缀 `d0fde4920bf2c7ac` / `914c64f7abb2f600` 与 §0.3/§3.5.4 记录**逐字相同** ⇒ 该工具链的 codegen 对同一特化是确定性的，设计所引设备码快照可审计。

## 检视详情

### 0. 源算子理解与迁移分析 — pass

**复用的核心主张（与 dense 逐字相同 / packed 无差异）经独立核对成立**，五条证据全部亲手复核：

1. **源码唯一模块级定义**：`bwd.py:79–106` 是唯一 `def _dq_land_kernel(heads, dim_qk, scale, dtype)`；两个 Kernel 类都经 `_MLABwdKernelBase._compile`（实为第 349–355 行，`land = _dq_land_kernel(self.heads, self.dim_qk, self.scale, self.dtype_str)` 在 **354** 行）取同一工厂，`_run` 在 **346** 行 `dq = land(dq_acc)`；`MLADenseBwdKernel` 在第 **358** 行、`MLAVarlenBwdKernel` 在第 **382** 行，二者差异只在 `forward`（**361–379** 含 `q.view(...)`/`dq.view(q.shape)` vs **385–387** 直接 `return self._run(...)`）。DESIGN §0.0 证据 1 的行号全部核对无误。
2. **抽取副本函数体逐字相同**：`diff` `examples/TileOPs/tileops/kernels/attention/mla_dense_bwd/_mla_dense_bwd_kernels.py` 与 `…/mla_varlen_bwd/_mla_varlen_bwd_kernels.py` 的 `_dq_land_kernel` 函数体（自 `def _dq_land_kernel(heads` 至 `return _dq_land_func()`，varlen 侧 429–456 行）→ **输出为空**；两文件差异仅在模块头注释与一处 docstring/空行（已逐块比对）。DESIGN §0.0 证据 2 与 §0.7 「第 429–455 行」成立。
3. **工厂签名无 packed 分支参数**：对照 `_delta_kernel(batch, heads, dim_v, dense_seqlen, dtype)`（bwd.py:39）与 `_bwd_kernel(..., dense_seqlen_q, dense_seqlen_kv, ...)`（115–116）——packed 信息以参数存在于另两个 launch，本 launch 只有 4 个参数（80 行）。证据 3 成立。
4. **编译产物与 `T_q` 无关**：本目录设备码快照第 16 行 `_land_kernel(__gm__ half* dq, __gm__ float* dq_acc, int32_t total_q)`、第 23 行 `… < ((total_q + 63) >> 6) * 8`、第 25 行 `asc_copy_gm2ub_align(…, min(64, (total_q - ((it >> 3) * 64))), 768, …, 6144, 768)`、第 33 行 store 侧同一 `min(64, …)`——`total_q` 是 kernel 形参，没有任何行数被烘成常量。证据 4 成立（我重跑探针复现同一份 codegen）。
5. **packed 调用路径的唯一差别**：dense 侧 `_run` 收 `q.view(B*S_q, H, -1)` 并在返回时 `view` 回 BSHD（361–379）；packed 侧 `forward` 直接 `return self._run(q, k, v, o, do, lse, cu_seqlens_q, cu_seqlens_kv)`（385–387，实参已在 `examples/TileOPs/tileops/kernels/attention/mla_varlen_bwd/mla_varlen_bwd.py:230–248` 确认无 reshape）。证据 5 成立。

**P1–P9 九项复核抽查结论（全部成立）**：P1 `_run` 内 `dq_acc = torch.zeros(q.shape, dtype=torch.float32, device=q.device)`（`mla_varlen_bwd.py:223` / `bwd.py:342`）恒为连续新分配，kernel 只读 `dq_acc` 不读 `q`；P2 `T_q` 不参与特化（设备码见证据 4）；P3 `out_idx=[1]` 输出 `(total, heads, dim_qk)`，packed 侧直接返回、dense 侧多一次 no-op `view`；P4 本 launch 参数表无 `cu_seqlens_*`（`bwd.py:94–95` 两个张量入参），`cu_seqlens_*` 只出现在 `main(...)` 实参表（`mla_varlen_bwd.py:226`）与 **227** 行 `dq = land(dq_acc)` 无关；P5 `scale` 同源（`bwd.py:320` / `mla_varlen_bwd.py:168` 同为 `dim_qk**-0.5 if sm_scale is None else sm_scale`），packed 测试含 `sm_scale=0.05`（`tests/ops/test_mla_varlen_bwd.py:126–133`）只多一份特化；P6 dtype/整除契约一致（`ATTENTION_DTYPES = (float16, bfloat16)`（`_kernels/attention/call_spec.py:50`）与 wrapper 内联同集合，`_STEP = 16` 与 `dim_qk % 16 == 0`（`bwd.py:277`）同义）；P7 分核结构不变、数值按 packed workload 重算（见维度 3）；P8 `[64] → ntiles = 4 < 56` 空区间零迭代；P9 数值路径由本目录机器验证 + dense 真机 repro 双证。

**未发现遗漏的 packed 差异**。我另外主动核查了五个设计未逐条列出的旁路，均不构成差异：① `dq_acc` 的行 stride 恒为 `H·D_qk·4`（packed 张量在内存里是连续行轴，请求之间无填充，manifest `dq: [T_q,H,D_qk]` + workload `total_q = sum(q_lens)`（`workloads/attention.py:398–400`）确认）；② `heads` 在 packed workload 为 8/128/64、测试为 4，全为 2 的幂，tile 解码退化为 `it>>3`/`it&7` 的前提成立（非 2 幂情形 §5.4/§9.3 已声明用普通 `%`/`//`）；③ `dtype_str` 两条路径同源；④ 0 长度请求 / `T_q = 0` 时 `ntiles = 0`、全部核零迭代、无输出，与源侧「网格为 0」同构；⑤ 无 host 侧 per-request 预处理进入本 launch。

> **一处时序澄清（非设计缺陷）**：调度 prompt 建议「diff 两个 Stage 3 实现」，但本阶段 packed 侧 `_dq_land_kernel.py` 尚未产出（§10.1 标注为 Stage 3 产物），故本次以「源函数 + 抽取副本 `diff` + 本目录探针的同特化设备码」三段替代取证。Stage 3 产出后应按 §10.4 第 1 步对两份实现再 `diff` 一次（该项留作 Stage 3 自检项，不影响本检视结论）。

**语义理解正确性 pass**：§0.1「fp32 域乘 + 单次 RNE 窄化、无规约、无跨元素依赖」与源码一致——`acc = T.alloc_fragment([block, dim_qk], "float32")`（98 行，fp32 fragment）、`for i, d in T.Parallel(block, dim_qk): acc[i, d] = acc[i, d] * scale`（100–101，fp32 乘）、`T.copy(acc, dq[...])`（102，写回时一次 cast），`scale` 为 python float 闭包常量（80 行）。§0.2 的 I/O（`(total, heads, dim_qk)`、fp32 进 / `dtype` 出、同 shape）与源码注解 94–95 行、manifest `dq: {shape: "[T_q, H, D_qk]"}`（`manifest/attention.yaml:259`）、workload 生成形状三者一致。§0.1 的溢出阈值 65520、非有限值自然传播、边界「源码无显式谓词、靠隐式越界守卫」均可在 99/102 行的 block 全 extent region 上对应，且未夸大为「已验证」。

**算法步骤覆盖 pass**：源码全部计算语句只有 4 条（97 网格解码、99 搬入、100–101 逐元素乘、102 搬出），§0.3 的 S1–S4 一一对应，**无遗漏、无臆造**（本函数无 SMEM 暂存、无 K 维循环、无 `T.Pipelined`、无 cu_seqlens 算术，§0.3 也如实这么写）；host 侧 `_compile` / `_run` / 两个 `forward` 已列出。

**优化手段覆盖 pass**：§0.4 的 9 项逐条可回溯源码：二维网格(97)、`block = 64`(82)、copy 内融合 cast(98/102 的 dtype 差)、`T.Parallel(64,192)` 摊平(100)、`threads=128`(97)、`TL_ENABLE_FAST_MATH`(86)、`compile_flags=["-O3","-DENABLE_BF16"]`(87)、`out_idx=[1]`(85)、热路径无分支(99–102)。未识别的优化项经我反查不存在；`@target_cache(maxsize=32)`(79) 虽未列入 9 项，但 §4.7/§10.3 已按 `lru_cache(maxsize=32)` 显式承接，不构成静默丢弃。

**耦合性判定合理 pass**：§0.5 十条四态齐全（保留 4：`#2/#8/#9/#10`；等价替换 2：`#1/#4`；重新设计 1：`#3`；舍弃 3：`#5/#6/#7`），统计自洽（4+2+1+3=10）。关键依据我按平台文件实查：`programming.md:13`「`T.Kernel(N)` 一维核网格；签名没有 `threads`，多维网格在实现中拒绝」；DMA 不做 cast 的拒绝文本有 dense 历史探针 `design_probe_simd.py`（m5/m6）；`#5/#6/#7` 舍弃理由（无线程域 / fast-math 无可收缩链 / nvcc flag）与 §7.3、§9.2 E6 一致。

**NPU 重设计可行 pass**：RD1（区间互斥+覆盖：`cpt = ceildiv(ntiles,G)` ⇒ `G·cpt ≥ ntiles`；`it → (h = it%H, t0 = (it//H)·R)` 对 `[0, ntiles)` 是双射 ⇒ 每个 `(t,h,d)` 恰好被写一次）、RD2（`vmuls` = 同一 fp32 乘；`vcvt(ROUND_R, sat=False)` = 单次 RNE 非饱和；落盘 `PK_B32` + `pset(32,"PAT_ALL")`）、RD3（`n = min(R, T_q − t0)`，有效行集合恒等、越界行不读写 GM）三项都满足平台约束（一维网格、无 Cube、UB 147456 B），且论证只依赖「`T_q` 是运行期量」这一在 packed/dense 下相同的性质 ⇒ **packed 侧无新增重设计**这一判断成立（对照同 op `_delta_kernel` 因输出 layout 从 `(B,H,S)` 变 `(H,T)` 而必须重推，本 launch 不涉 layout 变化，`bwd.py:48` 的 `delta_shape` 分支本 launch 不具备）。

**设计与决策一致性 pass**：§1.4/§1.5/§2/§3.1/§3.3/§4/§5/§6/§7 描述的全是 RD1–RD3 之后的形态（一维 56 核 + `T.Pipelined` 2 版本 + VF 段链 + 运行期 clamp），未出现「§0 判重设计、§3 照抄源 copy-cast」的脱节；§3.3 伪代码与本目录探针 `history_version/design_probe_packed.py::build` 同形。

**golden 独立性 pass**：§8.1 的 `golden_dq_land` 只写 `(dq_acc.float() * scale).to(dtype)`，与源 docstring（81 行「``dq = dq_acc * scale`` in the input dtype」）逐字对应，不含 `vcvt`/tile/`R`/56 核/`cu_seqlens` 任何 NPU 或 packed 概念 ⇒ 即使 §0.6 的重设计整体被推翻 golden 仍不变。

### 1. API 可行性 — pass

按任务平台（`stage_skills.review = tilelang-ascend`，ruleset `workflow.md`）要求，在**当前安装包**（`/workspace/venv-ascend950/lib/python3.12/site-packages/tilelang`，0.1.15）逐条实查导出与签名，不采信记忆或其他工具链：

| §3.2 的 API | 实查结果 |
|---|---|
| `T.Kernel(G)` 一维、无 `threads` | 导出；`programming.md:13` 明示一维、无 `threads`、多维拒绝 |
| `T.dynamic` / `T.min` / `T.ceildiv` | 导出 |
| `T.alloc_shared((R,DW), dtype)` | 导出（本目录设备码用 `buf_dyn_shmem` ⇒ `shared.dyn` = UB） |
| `T.annotate_buffer_versions({acc:2, out:2})` | 导出（`annotations.py:19–31`：1 = 退出多缓冲，≥2 = 固定版本数） |
| `T.Pipelined(start, stop, num_stages=2)` | 导出；动态界在本目录探针设备码第 23 行成立 |
| `T.SimdVF()` | 导出（`frame.py:SimdVF`） |
| `T.simd.vld(addr)` | `vld(addr, dist='NORM', *, post_inc=None)` ⇒ 默认形态即设备码的 `vlds_norm` |
| `T.simd.vmuls(v, scalar)` | `vmuls(src, scalar, mask=None, mode='MODE_ZEROING')` |
| `T.simd.vcvt(v, dtype, round="ROUND_R", sat=False, part=0)` | `vcvt(src, target_dtype, mask=None, round='ROUND_R', sat=True, part=0, mode='MODE_ZEROING')` ⇒ 四个关键字都存在，且默认 `sat=True` 说明显式 `sat=False` 有实义 |
| `T.simd.vsts(out[...], c, m, "PK_B32", extent=64)` | `vsts(addr, src, mask=None, dist='NORM_B32', extent=None)`，源码对 `PK_` 前缀分支处理 ⇒ `"PK_B32"` 合法 |
| `T.simd.pset(32, "PAT_ALL")` | `pset(elem_width: int, dist: str = 'PAT_ALL')` |
| `tilelang.compile(main, target="ascend", out_idx=[1])` | 本目录探针双 dtype COMPILE OK，`out_idx=[1]` 生效、`total_q` 仍为形参 |

负向结论也有佐证：`T.copy` 跨 dtype 被语义检查拒绝（dense `design_probe_simd.py` m5/m6 的真实报错原文）；`T.Persistent` 确实存在但签名是 `(domain, wave_size, index, group_size=8, num_stages=0)`、语义是二维 tile 调度/swizzle（`tile_schedule.py` 的 `Ascend*TileScheduler` 族），§6.1 的弃用理由成立。**未发现凭记忆臆造的 API**。

### 2. 内存层级规划 — pass

- 搬运路径完整（§4.4）：`GM(dq_acc fp32) —MTE2 T.copy(region n×1×192)→ UB acc[v] —V: vld→vmuls→vcvt→vsts(PK_B32)→ UB out[v] —MTE3 T.copy(region n×1×192)→ GM(dq dtype)`；不经 L1/L0A/L0B/L0C，与纯 Vector kernel 一致，§4.4 已如实声明。
- UB 预算（人工复算，机械检查 skip）：`acc` 64·192·4·2 = 98304 B + `out` 64·192·2·2 = 49152 B = **147456 B**，占 253952 B 的 **58.1%**；按 ×1.12 上界 165151 B（161 KB，65%）仍在容限内。`R = 96` 会到 221184 B（×1.12 = 247726 B）、3 版本会到 221184 B（×1.12 ≈ 242 KB）——设计据此否决，算式复核无误。
- L0C：无 Cube 段 ⇒ n/a（§5.1 显式豁免声明，非遗漏）。
- Buffer 层级标注齐全（§4.3：`acc`/`out` 均 `shared.dyn` = UB，掩码在向量谓词寄存器），无 GM 级中间缓冲（§4.3 末明确）。
- 段宽/对齐：fp32 行 768 B、fp16 行 384 B，均 32 B 整数倍；UB 侧 `out[i, s*64]` 步长 128 B；GM 偏移在设备码提升到 `int64_t`（第 25/33 行 `((int64_t)it >> 3) * (int64_t)98304`），最大 packed workload 的 3.22 GB `dq_acc` 安全。
- 旁注（不阻塞）：§4.5 的 ×1.12 引 `CONST-capacity-910B2C`，该条目的 1.10–1.12 标定是「auto-multi-buffer=false」口径，且同条目另记一条显式 alloc 结构实测 ratio 0.981 的反例。设计已标注「910B2C 口径、跨型号假设」并把 58.1% 建立在原始字节上（平台 `design_budget.ub_inflation` 本型号为 `null`），口径正确、结论保守；同几何在 dense 真机 31/31 通过，实测无障碍。

### 3. Tiling 策略 — pass

- Block 划分：`R = 64` 行/tile、`DW = roundup(192,64) = 192`、`NSEG = 3`，取值理由（UB 预算与 Stage 4 膨胀余量）量化给出（§5.2/§5.4）。
- 对齐：`192 % 64 == 0`，尾轴无 padding；`dim_qk % 16 == 0` 由算子契约保证，非 64 倍数时 `DW` pad + 切片 copy 的处理在 §9.3 声明为**未验证假设**（诚实标注）。
- 非整除：运行期 `n = min(R, T_q − t0)`，两条 `T.copy` 的 region 与两侧 UB 切片同用 `n`，行循环保持静态 `R`；不做 host padding/crop（理由：会多一次 GM 往返且 `dq` shape 由 JIT 绑定）。packed 下该策略是正确性必需项（manifest 四个 `T_q` 恰整除，但测试含 `T_q = 308/64`），设计已明确「不能因 manifest 巧合省略」。
- **分核三要素齐全且数值自洽**（按 `.agents/skills/_shared/standards/core-split-strategy.md` §1/§2.2 核对）：
  ① 逻辑核数 `ceildiv(T_q,R)·H` = 80 / 65536 / 64512 / 32768 / 4（表）；
  ② 物理核数**实查记录**：我重跑 `python3 .agents/tools/statectl.py platform --task-dir examples/mla_varlen_bwd/_dq_land_kernel` 得 `hardware.observed_cores = {cube: 28, vector: 56}`、`core_source` 为 `Ascend950PR_9579.ini [SoCInfo] ai_core_cnt=28, cube_core_cnt=28, vector_core_cnt=56; runtime probe rtGetAiCoreCount()=28`；§5.5 另给了 Stage 3/4 的 `torch.npu.get_device_properties` 查询代码，我核对 `vector_core_num`/`cube_core_num` 确为 torch_npu 暴露的属性（`torch_npu/_inductor/config.py:36–37`、`torch_npu/npu/npu_config.py:255–256`），并符合 `workflow.md` 「分别记录 Cube/Vector、不用 NPUUtils、不乘二」；
  ③ 规模判定 = **极大规模 → persistent 核内串行**：`G = 56` 固定、`cpt = ceildiv(ntiles,56)`、区间 `[bx·cpt, min((bx+1)·cpt, ntiles))`，失衡表逐格复核通过（smoke `cpt=2` → 40 核各 2 tile、16 核空区间 +40%，其余 ≤0.15%），退化场景（`ntiles < 56`）空区间零迭代、无需分支；核内循环只用 `T.serial`/`T.Pipelined`（动态界已编译验证），不依赖未认证的跨 block barrier。

### 4. 技术约束检测 — pass

- 三维 Kernel：不涉及（一维 `T.Kernel(56)`；§5.1 显式声明本平台 `T.Kernel` 多维被拒，故三维 Kernel 的 block_metadata 方案 n/a）。
- GEMM 非整除：不涉及（无 `T.gemm`），已在维度 3 说明非整除由运行期 clamp 处理。
- L0C 溢出：不涉及（不使用该层级）。
- 其余平台限制逐条处置（§3.5.1）：UB 253952 B、`dw`/行对齐、`dim_qk % 16`、`T.copy` 元素数/局部 region、fast-math 关闭、int64 偏移、AutoSchedule 多缓冲前提——与 `programming.md`/`workflow.md` 的条款一致。

### 5. 循环与同步 — warn

- 循环结构明确（§6.1/§6.2）：L0 `T.Kernel(56)`（静态）、L1 `T.Pipelined(it0,it1,2)`（动态界）、L2 `T.serial(R)`、L3 `T.serial(NSEG)`；不用 `T.Parallel`（SIMT 域，混用未验证）、不用 `T.Persistent`（实查其为二维 tile 调度/swizzle，与一维 tile 流不符，理由具体）。
- 同步与模式匹配：AutoSchedule 自动插 `asc_sync_notify/wait`，无手工同步（`programming.md:53` 口径）；本目录设备码 19–39 行确有 4 组版本化 flag（`MTE3→V`、`V→MTE2` 各 0/1）与按 `it&1` 的 ping-pong wait/notify，与 §7.2 表逐行对应；56 核之间无数据依赖（每个 `(t,h)` tile 只属一核），无需跨核同步。
- 尾块（§6.4）：行数 clamp 双侧同 `n`、写回保护与 load 对称（设备码第 25/33 行）；`[n, R)` 行读未初始化 UB 但不搬出（风险 R4 已列处置）。
- **建议级观察 S1（不阻塞）**：§6.3 写「`load(k+1)` 与 `compute(k)/store(k)` 重叠」，但本目录设备码第 24–34 行的顺序是「`wait(V→MTE2, it&1)` → load → `notify(MTE2→V, it&1)` → `wait(MTE3→V)` → **`wait(MTE2→V, it&1)`** → VF → …」，即**本轮 load 之后立即等本轮 load 完成**，生成码里实际可见的重叠只有 `MTE2(k+1) ∥ MTE3(k)`，load 与 compute 没有跨轮重叠（也没有 prologue 预取）。**改法**：把 §6.3 该句改写为「2 版本使 store(k) 与 load(k+1) 的 DMA 可交错；load/compute 的跨轮重叠需 prologue 预取或更深流水，列为 Stage 4 的 A/B 候选」，并在 P5 的对照里把「是否真正重叠」纳入口径。该点纯属性能解释偏差（正确性由真机 20 次一致 + 单版本对照覆盖），故不阻塞。

### 6. 验证方案 — pass

- Golden：§8.1 给出完整 PyTorch 参考实现草案，独立性见维度 0；`NaN→NaN`、`|y| ≥ 65520 → ±inf` 与 §0.1 语义一致。
- L0 门槛计划（§8.3）9 条，覆盖：主路径 + `out_idx` 动态输出（L0-1，最先跑，风险最高）、bf16（L0-2）、ragged 尾块 clamp（L0-3，`308 = 4×64+52` 复算正确）、退化 `T_q = 64`（L0-4，`ntiles = 4`）、整除对照（L0-5，`ntiles = 24`）、角点注入双 dtype（L0-6a/6b，含 ±0/±inf/NaN/次正规/65504/65519/65520/1e5/7e4，且 bf16 溢出角点需 `scale = 1.0` 的理由正确——`mx/scale` 会先溢出 fp32）、满载基线（L0-7，`ntiles = 65536` 复算正确）、设备码硬前置 `_simd_body_gate`（L0-8）、非法契约拒绝（L0-9）。
- 精度标准：fp16 `rtol 1e-3 / atol 5e-3`、bf16 `1.6e-2 / 1.6e-2`，与 `tests/ops/test_mla_varlen_bwd.py:46–52`（`_RTOL`、`_ATOL_FLOOR = 5e-3`）逐字一致，非臆造。
- 附加判据合理：以 `torch.equal` 逐位为强判据（dense 真机 31 个用例 `max_diff=0`/`bit_diff=0` 支撑可达性），并给出排查顺序（落盘分布 → 掩码宽度 → 尾块 extent）；P1–P7 定点检查含 canary、单次舍入、非饱和、流水一致性、退化几何与 **P7「同一二进制服务多个 packed 切分」**（直接检验 `T_q` 只以数值进入）。
- 该计划能覆盖本设计的全部不确定项（`out_idx` 动态输出、动态界流水、2 版本、packed 几何），风险分级合理。

### 7. 完整性与一致性 — pass

- 章节齐全：§0（0.0–0.7）+ §1（1.1–1.6，含 1.6.0–1.6.3）+ §2 + §3（3.1–3.5.4）+ §4（4.1–4.7）+ §5（5.1–5.5）+ §6（6.1–6.4）+ §7（7.1–7.3）+ §8（8.1–8.3）+ §9（9.1–9.4）+ §10 + §11；另有「复用分区总览」「修订说明」两张有效前置表。
- 无占位符：对占位标记类关键词（未完成标记、花括号占位符、待补写标记）全文检索，0 命中。
- 同类引用真实：§0.7/§3.5.3/§3.5.4 引用的 16 个路径我逐个 `test -e` 全部存在（含 dense 的 `DESIGN.md`/`_dq_land_kernel.py`/两个 `repro/` 脚本、dense `history_version/design_probe_simd.py` 与 `probe_e10/11/12/13/15_gen.txt`、`probe_n5_gen.txt`、`_delta_kernel` 的 DESIGN 与实现、`paged_cache_write` 实现、`kernel_template.py`、`PL-950-row-primitives.md`、`numerical-boundaries.md`、`bottlenecks.md`、`capability-gaps.md`、`constants.md`、`core-split-strategy.md`、manifest 与 workload）；引用的源码行号抽查 20 处（79/82/90/97–102/277/320/338–347/349–355/354/346/358/382/361–379/385–387；wrapper 156–163/168/205/223/226/227/230–248）全部准确。
- 内部一致：UB 147456 B 在 §1.6.0 R4 / §1.6.3 / §4.5 / §5.2 / §5.3 六处一致；`E`/`6E`/`ntiles` 的 4 个 packed workload 数值我按 manifest 独立复算（E=983040/805306368/792723456/402653184；6E=5.90 MB/4831.8 MB/4756.3 MB/2415.9 MB；ntiles=80/65536/64512/32768）与 §1.6.0/§5.5 完全一致；估算时延带（smoke 5.6–7.9 µs、`ds-v3-8x4k` 4.60–6.44 ms、`ds-v3-mixed` 4.53–6.34 ms、`kimi` 2.30–3.22 ms）与 6E ÷ (0.75–1.05 TB/s) 逐项相符，发射项 50.3 M 条 ÷ 56 核 = 899 K 条/核 ≈ 642 µs @1.4 GHz 也算对；API 映射（§3.2）、内存规划（§4）、Tiling（§5）三处无矛盾。

### 8. 算法优化分析 — warn

**§1.6 章节存在性 pass**：1.6.0（调研四问）/1.6.1（数学等价优化）/1.6.2（向量化替代）/1.6.3（向量化轴与布局决策）四节齐全，无「只有标题无分析」。

**算法调研存在性与完整性 pass**：R1 候选表**含基线 B0**，六候选（B0 / R1-a 单次舍入 / R1-b 16 位域乘 / R1-c 饱和窄化 / R1-d 截断窄化 / R1-e 融合进上游 launch / R1-f 省掉 fp32 落地）逐行给出结构、等价性初判与收益方向；R2 有明确结论（「本算法本身即单遍在线形式，无可省多遍」）+ 结构依据（纯映射、无状态、无中间矩阵）；R3 四口径齐（FLOPs / 访存 Bytes / 扫描遍数 / 中间缓冲峰值）且**基线在表**、口径写清（`E = T_q·H·D_qk`，流量 = 读 4 B + 写 2 B）；R4 逐候选六维评估（单元匹配/容量/对齐/静态边界/流水/跨核）；调研深度与「单步逐元素」分级匹配（轻量调研但仍量化了带宽项，符合「带宽受限算子流量项必算」）。机械 `r3_metrics` pass 与人工核对一致。

**调研结论正确性（独立复核）pass（附三处需修正的表述，见 warn）**：

- ① 等价性：选定算法与源语义等价——源 `acc*scale`（fp32 fragment）→ 一次 cast，本设计 `vmuls` + `vcvt(ROUND_R, sat=False)` 保持同一「fp32 乘 + 单次 RNE」；**我独立推演**：A1 的「单次舍入」与 golden 同式（源 copy 也是单次 cast），C1 的两轮舍入在 fp16 上最大 3857 ulp（机器验证复跑同值）；A3 的非饱和 IEEE 语义在模型层与 golden 同，设备层另由 dense 真机 repro（65519→65504、65520→inf、1e5→inf，`sat` 惰性）支撑 ⇒ 无「容差内等价掩盖舍入风险」的问题。
- ② 复杂度复算：`6E` 为硬下界（fp32 进 GM 4E + 16-bit 出 GM 2E），本设计无 GM 中间缓冲 ⇒ 未漏计；R1-e 行把**跨两 launch 的 10E → 融合后 2E（省 8E）**显式写进表，`8E = 6.44 GB`（E = 805306368）复算正确——中间缓冲的 GM 往返**没有**被漏计。R2 的「1 遍」与 R3 的「扫 1 遍」自洽。
- ③ 亲和性淘汰有佐证：DMA cast 淘汰有真实拒绝文本（dense m5/m6）；Cube/MixCV 淘汰（零矩阵乘，28 个 Cube 核无可算内容）；host 侧 torch 淘汰（越出算子边界且多一次 GM 往返）；SIMD 主选与 SIMT 备选的容量/对齐/流水判断与 `programming.md`、`numerical-boundaries.md:19`（「不把 SIMT 结论用于 `T.simd.vcvt` 的任意模式」——本设计正是据此对 `vcvt` 另取真机证据）一致。
- ④ 负向断言复核：**「无在线变体」成立**（纯映射无跨元素状态，与候选库 `ALG-elementwise` 的「R2：逐元素映射天然单遍」一致）；**「R1-e 是唯一减半流量的路径且不可用」成立**——我核对 `knowledge/capability-gaps.md` 确有 `CG-950-batch-barrier`（「ascend.batch_schedule 的跨 block barrier 依赖」未认证）且 `workflow.md:29` 有同一未认证声明，另外该融合在数学上还要求「跨 CTA 累加到齐」，源第 254 行 `T.atomic_add(dq_acc[...], dq_f[...])` 证明确需设备级完成同步；**「无化简公式」成立**（候选库 ALG-elementwise 的 R1 三项：公共子表达式消除 / 除法乘倒数 / cast 链合并，前两项由「只有一个乘」结构性排除、第三项即 A1）。
- ⑤ 并列候选：A1（SIMD 段链）vs A3（SIMT 投影）不纸面拍板，§1.6.3 给了实验裁决三件套（主选 + 结构完整的备选 + 代表 shape/指标/判定阈值 `5%`/回写路径），并评估了按 shape 分派的可行性（可按 `heads`/`dtype`，不能按 `T_q`，因为 `T_q` 不参与特化）⇒ 符合要求。

> **建议级观察 S2（不阻塞）**：§1.6.0 末「「无更优替代」的调研范围」括号写「`knowledge/algorithm-candidates.md`（**纯 elementwise 无专门条目**）」，与该文件实际内容矛盾——该文件首段表内即有 `elementwise / broadcast | 输入转 FP32，行标量广播后逐元素计算 | PL-950-row-primitives` 一行；同时未引设计 skill 的候选库条目 `ALG-elementwise`（R1 三项、R2、R3「融合消除中间 GM 往返」、R4「单遍 Vector 流/dtype 路径」，`known_impl` = `elementwise/lerp_tensor/.../perf_opt/`，`kb_links` = `TRAP-fp16-opmath-golden`、`PL-1.6-copy-floor`）。**实质候选无遗漏**（该行的四项在本设计 §1.6.1 结论段与 R1-e 中均已逐项评估，SIMT 行原语也已作为 §1.6.3 备选 A3 评估并被 §0.7 引用），故不改判定，但**改法**：把该括号改为「候选库 `ALG-elementwise` 行与本平台 `algorithm-candidates.md` 的 `elementwise/broadcast` 行（→ `PL-950-row-primitives`，对应候选 A3）已逐项评估」，并补 `TRAP-fp16-opmath-golden`（逐元素 fp32 中转链对齐 torch golden ≤1 ulp——正是 A1 的独立佐证）与 `PL-1.6-copy-floor` 两条 kb_links 引用。

**调研→设计一致性 pass**：§1.4 描述的就是 §1.6.0 选定的「单遍流式向量映射」；§3.1 的拆解以 §1.6.1 的 A1–A5 优化后公式为输入（伪代码里 `vmuls(scale)`=A2、`vcvt(ROUND_R, sat=False)`=A1/A3、`DW=192/NSEG=3`=A4、`n=min(...)`=A5）；§6 的循环结构与 §1.6.2 结论一致（判定已替代的点没有留下逐元素标量循环）；迁移侧 §0.5/§0.6 与选族一致（SIMD 段链），源优化意图 #3（copy 内 cast）/#4（摊平向量化）的承接在 RD2 明确写出，**无静默丢弃**（9 项优化逐项承接/舍弃有据）。

**数学等价优化完整性 pass**：A1–A5 均有「原式 → 优化后公式 → 等价性论证 → 收益量化」四要素；无优化空间处给出正面陈述（只有一次乘，无除法/rsqrt/稳定化/CSE/归约合并空间）。**逐项独立推演**：A1 成立（golden 即一次舍入式，且 C1 被机器证伪）；A2 成立（两种写法乘同一 fp32 常量，IEEE 乘法逐位相同，设备码内联 `float(0x1.279a74590331cp-4f)`）；A3 成立（模型层 = IEEE，设备层已有真机 repro）；**A4 成立**（`DW = 192 = 3×64`，pad 形态的超出行不进入 `T.copy` 的 `dim_qk` 列 region ⇒ 输出逐位相同；指令 `4R → 3R` 与 UB `256→192` 的 −25% 复算正确）；**A5 成立**（有效行集合 `[0, min(R,T_q−t0))` 恒等，越界行不读写 GM，无浮点影响）。

> **建议级观察 S3（不阻塞）**：A2 的收益量化写「每段少 1 条向量指令（`ds-v3-8x4k` 约 12.6 M 条）」，但候选库 kb_link `PL-1.6-copy-floor` 的实测口径是「≥16M 档计算链被 MTE2 窗口完全隐藏、计算链优化 ROI≈0」，与 §1.6.0 自己的「发射项约 642 µs，被流量项支配」一致 ⇒ 该 12.6 M 条在性能上**不应**计入收益。**改法**：把 A2 的收益改写为「少 1 条向量指令/段 + 少 1 个向量寄存器；在本带宽受限算子中性能收益≈0（`PL-1.6-copy-floor`），保留该项是因为它消除了一个全 lane 广播的寄存器压力」。

**向量化替代覆盖完整性 pass**：§1.6.2 的 V1–V11 与 §3.3 伪代码、§6.1 循环表逐点交叉核对，**无遗漏**：V1 `vmuls`、V2 `vcvt`、V3 `vld`/`vsts`+两条 DMA 均已替代；V9 谓词已消除（设备码 VF 体内只有 `for i<64` / `for s<3`，无 `if`）；保留 4 个索引/映射点（V4 行循环、V5 段循环、V6 tile 解码、V7 `cpt`/`it1`，`ntiles` 归入该类）与 1 个动态边界点（V8 `n`），V10/V11 为「无需处理」。保留理由具体到本算子（tile 级结构索引 / block 级任务映射 / 动态 shape 使 `ntiles` 只能运行期算），且 **V5 的保留有负向证据**：我实查 dense `probe_e11_gen.txt` 的 SIMD 体只剩两个 mask 创建、`vlds_norm` 计数为 0，而 `probe_e15_gen.txt` 有 `vlds_norm`——与设计所述「python 层展开会丢代码」一致。替代 API 全部有安装包导出 + 本目录设备码双佐证。

> **建议级观察 S4（不阻塞，文字层面）**：① §1.6.2 结论段「保留的 **4** 个点全部属于…（V4/V5/V6/V7）或…（V8）」与表内 5 个保留点不自洽（应为「5 个：4 个索引/映射 + 1 个动态边界」）；② §1.6.1 A4 的「**64 个 32 位字**恰好装下 64 个有效结果」单位应为「64 个 32 位 **lane** 的低半字（共 128 B）」。两处均不影响已成立的结论，Stage 1 修订时顺手改正即可。

**向量化轴与布局决策完整性 pass**：§1.6.3 给出了候选矩阵 A1–A5（本算子属逐元素类，无广播轴 ⇒ 退化为「连续轴选择 × tile 覆盖形状」，已声明该退化理由），逐候选有量化评分（整除性 `192 = 3×64` 零尾 lane 浪费 / 无累加链 / 无 repack / UB 58%）；弃选均有量化理由——A2 多 head tile 容量否决（`HH=2, R=64` 即 294912 B 超 253952 B）、A4 硬件无 128-lane fp32 向量寄存器、A5 转置布局多一次 GM 往返 `+6E`（与带宽目标相反，packed 下更不可取）；I/O layout 被正确当作契约、核内布局当作设计变量；迁移侧对照了源码的隐式轴选择（源 `T.Parallel(block, dim_qk)` 的最内 `d` 轴 → 本设计 64-lane 段）；**弃选前提我亲自核对**：DMA cast 拒绝文本、`T.Persistent` 实为二维 swizzle 调度、段宽判据 `CONST-gm-to-ub-bw-dilution`「段宽 ≥ 数百字节时性能正常」原文存在（且设计标注了 910B2C 跨型号假设）。**实验裁决三件套齐全**：未实证常数逐条列出（950PR 的 4 类向量指令吞吐 / `PART_EVEN` lane 利用率 / `PK_B32` 额外搬运代价 / SIMT `float2` 有效带宽差异），并核对 `bottlenecks.md` 原文「当前没有 950 真机验证的瓶颈模式或收益阈值」、`cost_measurements` 为空（我实查 `statectl platform` 输出亦为 `[]`）⇒ 属「未实测」，故给备选 A3 的完整结构与 5% 判定阈值 + 回写路径，符合规则。

**布局决策与实现一致 / 与下游章节一致 pass**：§3.3 伪代码的 `for s in T.serial(NSEG)` 内层向量维 = 选定的 64-lane 段、`acc/out` 形状 = `(R, DW)` = 选定布局、UB 预算按该布局计算（§4.5）；§6 不出现 §1.6.2 未论证的标量循环；§3.1 用的是 A1–A5 之后的公式。

**综合判定**：维度 8 的实质结论（6E 下界、无更优替代、A1–A5 采纳、A3 备选）经独立推演与机械复核全部成立；S2/S3/S4 是**引用与表述**层面的缺陷（负向范围陈述与候选库文件矛盾、收益口径与 pattern-library 实测口径未对齐、两处计数/单位笔误），不推翻任何判定，故记 warn 而非 fail，建议 Stage 1 在下次修订（或 Stage 4 回写附录）时一并修正。
