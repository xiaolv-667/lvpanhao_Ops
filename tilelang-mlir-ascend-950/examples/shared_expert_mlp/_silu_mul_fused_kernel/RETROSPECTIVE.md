# `_silu_mul_fused_kernel` — Stage Retrospectives

## Stage 1 (design) Retrospective — first_design — 2026-11-23

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Design-flow | skill Phase M0 的「三问解读」清单里没有「本函数在迁移后是否仍会被调用」这一项；本次是靠读 NPU wrapper 的 `_decompose`/`_split_k` 才发现 5 条 workload 里只有 2 条真正调用本 kernel（其余走 splitk_reduce 的融合 epilogue）。若不查，§5.6 的配置表会按 5 条平铺而无法说明优先级。 | `DESIGN.md#0.3`（host 侧逻辑段）、`examples/TileOPs/tileops/kernels/moe/shared_expert_mlp/shared_expert_mlp.py#_decompose` | `references/migration-analysis.md` §3.1 的阅读协议第 6 条（host 侧逻辑）追加「读取迁移后 wrapper 的调用条件与 config 传递路径，确认本函数的运行时调用集合」。 | P |
| Design-flow | 逐元素算子的 §1.6.3 候选矩阵在 skill 参考里只列了「连续轴选择 × 广播轴对齐」；本次实际需要额外枚举「全行 tile / 扁平视图解交织 / 原地写回」三种结构候选才能把弃选论证写实。 | `DESIGN.md#1.6.3` 候选 #1–#6 | `tilelang-op-design/references/algorithm-research.md` §5 参考表补一行「elementwise / gated-activation」候选清单（含全行 tile 与视图重排两类）。 | P |
| Constraint-doc | 平台 `pattern-library/constants.md` 的全部条目绑定 Ascend910B2C，`cost_measurements` 为空；设计期 roofline 只能标「外推」。对 950PR 的 GM→UB 段开销/有效带宽没有任何条目，导致 §1.6.3 的主选 vs 备选必须走实验裁决而不能纸面判定。 | `DESIGN.md#1.6.0`（常数适用性声明）、`DESIGN.md#1.6.4` | `pattern-library/constants.md` 增补 ascend-950pr-9579 的 GM→UB 有效带宽与段开销两条 D 类条目（由首次 Stage 4 采集回填）。 | D |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 窄化 `vcvt`(f32→16bit) 必须配 `vsts(dist="PK_B32")` + `pset(32)`，否则静默漏写一半槽位 | D | `examples/shared_expert_mlp/_silu_mul_fused_kernel/DESIGN.md#3.2`、`examples/mla_dense_bwd/_dq_land_kernel/repro/CONST-950-vcvt-narrow-store-pack-b32.py` | `cd examples/mla_dense_bwd/_dq_land_kernel && ASCEND_RT_VISIBLE_DEVICES= /workspace/venv-ascend950/bin/python repro/CONST-950-vcvt-narrow-store-pack-b32.py`（需真机） | TileLang ascend 0.1.15 + CANN 9.2.1 runtime / Ascend950PR_9579 | pattern-library（已有条目，本次仅作引用确认） |
| silu 的 SIMD 最省形式是 `vdiv(g*u, 1+vexp(-g))`（5 条算符），与基线 `g*σ(g)*u`（6 条）输出逐位相同（0 ulp） | D | `examples/shared_expert_mlp/_silu_mul_fused_kernel/verify_equiv.py`（`EQUIV_RESULT: ALL_PASS`，bf16/fp16 × 4 用例 × 16k 元素） | `cd examples/shared_expert_mlp/_silu_mul_fused_kernel && /workspace/venv-ascend950/bin/python verify_equiv.py` | torch 2.13 CPU（fp32 回放 fp64 参照），不依赖设备 | pattern-library/constants 或 tilelang-op-design 参考表（elementwise 行） |
| `y=(g·u)·e^{g}/(1+e^{g})` 形式在 `g>88.7` 溢出为 NaN（实测 bf16 647 ulp / 5 处非有限值分类不符），是「数学恒等但 fp32 域不等价」的典型反例 | D | `examples/shared_expert_mlp/_silu_mul_fused_kernel/verify_equiv.py`（C2 列） | 同上 | 同上（纯 torch CPU） | tilelang-op-design/references/algorithm-research.md（R1 候选反例库） |
| 迁移后 wrapper 的 config（`block_m=128, block_n=256`）对逐元素 kernel 会直接 UB 溢出（393 KB > 248 KB），必须在工厂内夹取而不是改 wrapper | P | `DESIGN.md#5.2` 规则 1/2/5、`DESIGN.md#9.2 E1`；`examples/TileOPs/tileops/kernels/moe/shared_expert_mlp/shared_expert_mlp.py#default_config` | `ASCEND_RT_VISIBLE_DEVICES= /workspace/venv-ascend950/bin/python history_version/design_probe_compile.py`（P1 用 (16,512) 通过；(128,256) 的 UB 需求为手算） | TileLang ascend 0.1.15 + CANN 9.2.x 编译 SDK，无卡编译 | examples/TileOPs/.agents/skills/add-npu-op（K9/K11 适配条目） |

### Transferable Lessons（可迁移教训）

- 逐元素迁移的「tile」不是从源 kernel 抄的：GPU 的 `(block_m, block_n)` 是 SM 占用率取向，
  NPU 要按 UB 预算 × 56 Vector Core × MTE2 段宽重定；且必须先查 wrapper 实际传入的 config，
  否则会写出一个必然 UB 溢出的设计。
- 逐元素算子的 gap 论证只用三种料：流量（读了几份、写了几份）、每 64 元素的向量指令数、
  以及尾 lane/尾块浪费率。MFU/算术强度对这类算子没有意义。
- 源实现「数学恒等」的改写不一定在 fp32 域等价（例如把 `e^{-g}` 换成 `e^{g}` 会在大正数处
  `inf/inf`）；`verify_equiv.py` 的角点用例（±inf/NaN/subnormal/边界 shape）能在设计期
  机械拦住这类改写，成本 ~1 分钟。
- `T.SimdVF` 内不能读 GM：所有 `T.copy` 必须在帧外；`T.serial` 的行/chunk 循环是可用的，
  但循环里不能跨迭代携带向量值（需要累加时用 `T.simd.alloc_local`）。
- 平台 `T.Kernel` 只有一维且无 `threads`：GPU 的 `threads` 参数在迁移里应删掉并把并行度
  重新表达为「grid 个 block × 64 lane」，不要试图保留线程层级。

## Stage 2 (review) Retrospective — first_review — 2026-11-23

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Review-flow | 维度 8「数学等价优化完整性」的示例写的是「逐元素除法未考虑乘倒数」，但本平台 `simd` 模块**没有倒数 intrinsic**（`vrec`/`vrecs` 不存在），该示例会诱导 reviewer 判一个不存在的缺口；本平台真正需要查的是**同模块融合算符**（`vexpdif` = `exp(a-b)`，可省 1 条 `vneg`），而现有清单没有「按本平台 simd 导出清单枚举降代换候选」这一检查项。 | `REVIEW.md#8`（warn ①）；`/workspace/venv-ascend950/lib/python3.12/site-packages/tilelang/ascend/language/simd.py`（`def v` 清单无 `vrec`；`vexpdif` 在 1182 行） | `tilelang-design-review` SKILL.md §3 维度 8「数学等价优化完整性」增一行：负向断言「无高代价算符可降代换」须逐条比对本平台 `simd.py` 的**融合算符**导出集（`vexpdif`/`vabsdif`/`vmula`/`vmadd`/`vaxpy`），并注明平台是否有倒数类 intrinsic。 | R |
| Review-flow | `design_calc_check.py` 的 `core_split` 对「多 workload 平铺」的 §5 会产生伪 fail：它按正则从 §5 正文取 `M=1`/`bm=16`/`bn=256`，再从表格串 `kimi-k2-t1` 里取到 `declared=2`，于是重算得 1 ≠ 2。本次若按「fail → 维度 3 fail」机械执行会误判一个算术完全正确的设计。 | `REVIEW.md#机械复核` §2 `core_split` 行（detail 原文 `recomputed ceil(1/16) x ceil(1/256) = 1 vs declared 2`）；手算逐 workload `grid = mt·nsp` 与 §5.5/§5.6 全一致 | `.agents/tools/design_calc_check.py::check_core_split`：`declared` 只在「逻辑核数」同段落内匹配纯数字表格列（排除 `kimi-k2-t*` 等标签串），`M/N/bm/bn` 优先取 §5.6 表格行；无法唯一确定时降级为 `skip` 而非 `fail`。 | P |
| Review-flow | 本次发现「备选方案 UB 复算」类错误的通用形态：设计把「3 buffer 单 version 合计（110 592 B）」与「总占用（×2 version）」两口径混写，得出「2×110 KB > 248 KB」的假不等式。维度 8 的「备选方案 UB 预算按新任务粒度重算」没有机械位（`design_calc_check` 只查主选 `ub_budget`）。 | `DESIGN.md#1.6.3`（候选 #2）、`DESIGN.md#1.6.4`（备选 UB 108 KiB/version ×2 = 216 KiB < 248 KiB）；`REVIEW.md#8` warn ② | `tilelang-design-review` SKILL.md §3 维度 8「实验裁决模式完整性」增一句：备选方案的 UB 复算须给出「单 buffer/version → ×buffer 数 → ×version 数 → ×膨胀系数」的四步显式算式（禁写单一「2×N KB」比较式），reviewer 按四步逐项复算。 | R |
| Constraint-doc | `vcvt` 的 `sat=False`（RS_DISABLE）在模块文档里写「out-of-range values **wrap**」，而 golden 侧 `Tensor.to(float16)` 按 IEEE 溢出为 `inf`；两者在 `|y| > 65504` 时语义可能不一致，但 pattern-library 无任何 `sat`/溢出条目可对照。 | `/workspace/venv-ascend950/lib/python3.12/site-packages/tilelang/ascend/language/simd.py:1241–1247`（sat 段）；`REVIEW.md#6` 建议条 | `pattern-library/traps-runtime.md` 增条目「TRAP-vcvt-sat-false-overflow」：记录 f32→f16/bf16 `sat=False` 的溢出实测语义（inf / wrap / max）与对应 repro，供逐元素算子直接引用。 | D |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| `T.simd.vdiv(precision=None)` 的实际精度由 `tl.enable_fast_math` 决定：fast math 关闭 → `DivAlgo::PRECISION_0ULP_FTZ_TRUE`（多指令精化）；打开 → 裸 SFU（~2 ulp, ftz_true）。故「设计里删掉 `TL_ENABLE_FAST_MATH` 才拿到 0 ulp 除法」，且 `simd` 模块**无**倒数 intrinsic（`1/x` 也只能走 `vdiv`）。 | D | `simd.py:792–812`（vdiv docstring）、`tilelang/transform/pass_config.py:83–87`（`TL_ENABLE_FAST_MATH` 默认 False）、`REVIEW.md#8`（算符计数口径注） | `grep -n "def vdiv" -A 22 /workspace/venv-ascend950/lib/python3.12/site-packages/tilelang/ascend/language/simd.py` + `grep -n "TL_ENABLE_FAST_MATH" -A 5 .../transform/pass_config.py` | TileLang ascend 0.1.15 + CANN 9.2.1 SDK（无卡静态核对） | pattern-library/constants.md（向量算符代价条目）或 traps-runtime.md |
| 逐元素 SIMD 链的算符计数必须显式含「取负」：`exp(-g)` 在 CCE 是 `vneg + vexp` 两条，故 `g·σ(g)·u`=6 条、`(g·u)/(1+e^{-g})`=5 条，与 R3 的 12/11 才自洽；若用 `vexpdif(0,g)` 可再省 1 条（→10）。这是「迁移后比源算法少几条算符」类断言的核对口径。 | P | `DESIGN.md#1.6.1`（A1 原式/优化后各条数）、`DESIGN.md#1.6.0`（R3 表 12/11）；`simd.py:1182`（vexpdif） | `grep -n "def vexpdif" -A 16 .../ascend/language/simd.py`；等价性的 fp32 回放可复用 `verify_equiv.py` 的 `_f32` 逐步舍入写法 | TileLang ascend 0.1.15 + CANN 9.2.1 SDK | `tilelang-op-design/references/algorithm-research.md`（§3 R1 候选识别口径） |

### Transferable Lessons（可迁移教训）

- 逐元素迁移里「本 kernel 是否真的会被调用」必须查迁移后 wrapper 的分支条件（本例 `_split_k` 在
  `T ≤ 32 且 k_tiles ≥ 2` 时把激活融进 split-K epilogue，5 条 workload 只有 2 条走独立激活 kernel）。
  不查就会给不会执行的分支做优化取舍。
- 算符条数要按 **CCE 实际指令**数：`exp(-g)` 是 `vneg + vexp` 两条；`1+exp(-g)` 是 `vadds` 一条；
  一个「精确除」在关 fast-math 时内部是多指令精化，所以「少一条算符」类收益应写成上限值。
- 片上预算复算写成四步算式：单 buffer/version → ×buffer 数 → ×version 数 → ×膨胀系数；任何
  「2×N KB 与容量比大小」式的一步比较都可能把「单 version 合计」当「总占用」（本例 216 KiB 被写成
  > 248 KiB）。
- 收窄写 `vcvt(sat=False)` 的溢出语义（文档写 wrap）与 golden 的 IEEE inf 可能不同，只有
  `|y| > dtype max` 的用例能暴露；dtype 为 fp16 且 `y = 激活(x)·放大系数` 的算子应显式加一条
  这类 Boundary 用例，而不是依赖随机输入。

## Stage 3 (develop) Retrospective — first_impl / attempt 1 — 2026-11-23

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Develop-flow | REVIEW.md 里指向 Stage 3 的「开放数值问题」（`sat=False` 溢出是 wrap 还是 inf）在 develop skill 的 L0/Boundary 模板里没有落点：DESIGN.md §8.3 的 L0 计划四条全是随机输入用例，随机输入**永远**触发不到 `|y| > dtype max`，若照 DESIGN 的 L0 清单机械执行则该问题会带着「未知」结论进入 Stage 5。本次是靠人工把 REVIEW §6 的建议转成 B-10/B-11 两条用例才闭环。 | `REVIEW.md#6`（建议条原文「在 §8.3 的 Boundary 清单加一个两元素用例」）、`DESIGN.md#8.3`（L0-1..L0-4 无溢出用例）、本文件 `_silu_mul_fused_kernel.py::run_boundary` B-10/B-11 | `tilelang-op-develop` SKILL.md Phase 4 增一条：实现前逐条扫 REVIEW.md 的「建议级」条目，凡标注「交给 Stage 3 验证」的数值问题必须转成一条具名 Boundary 用例（不是只写在注释里）。 | R |
| Info-source | `statectl platform --task-dir` 的 `execution.precision_command` 在本机解析为 `/usr/local/python3.12.16/bin/python3`，该解释器 `import tilelang` 直接 `ModuleNotFoundError`（tilelang/torch_npu 只在 `venv-ascend950` 里）。若严格按 execution 字段启动，Stage 3 会在 import 阶段就失败并被误判成环境问题。 | 实测：`/usr/local/python3.12.16/bin/python3 -c "import tilelang"` → `ModuleNotFoundError: No module named 'tilelang'`；`/workspace/venv-ascend950/bin/python` → `tilelang 0.1.15 / torch 2.13`。`statectl platform --task-dir` 输出 `execution.precision_command` | `statectl.py platform` 在解析平台时对 `execution` 的两个命令做一次 import 探针（`python -c "import tilelang,torch_npu"`），失败则回落到锁定 venv 的解释器绝对路径并写进 `execution`；`env-950pr.sh` 也应把 venv 的 bin 放到 PATH 前面（当前 `python` 解析到 base 解释器）。 | D |
| Develop-flow | `{op}.py` 门禁要求 `golden_{op}(...)`，但 harness 迁移的 `op_name` 本身带前导下划线（`_silu_mul_fused_kernel`），机械拼接得到双下划线名 `golden__silu_mul_fused_kernel`；规范没有说明「是否保留 op 名原有的前导下划线」。 | 本文件 `golden_silu_mul` + 一行别名 `golden__silu_mul_fused_kernel = golden_silu_mul`；`tilelang-op-develop` SKILL.md §4「Structure」与 §「门禁校验」 | `tilelang-op-develop` SKILL.md §4 补一句：迁移 harness 的 golden 命名取 `golden_` + 去掉前导下划线的 op 名，并保留 `golden_{op}` 别名以兼容门禁。 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| `vcvt(f32→f16/bf16, sat=False)` 的溢出语义是 IEEE `±inf`，**不是**模块文档写的 wrap（128/128 位精确对齐 `Tensor.to(dtype)`，含 65519→65504、65520→inf、1e30、±inf、NaN） | D | `examples/shared_expert_mlp/_silu_mul_fused_kernel/repro/CONST-950-vcvt-sat-false-overflow-inf.py`（`[float16] bit-exact 128/128`、`[bfloat16] bit-exact 128/128`、`wrapped-to-finite=0`）；`REVIEW.md#6` 的开放问题由此关闭；本 kernel B-10/B-11 两条用例 diff=0 | `cd examples/shared_expert_mlp/_silu_mul_fused_kernel && ASCEND_RT_VISIBLE_DEVICES=0 /workspace/venv-ascend950/bin/python repro/CONST-950-vcvt-sat-false-overflow-inf.py` | TileLang ascend 0.1.15 + CANN 9.2.1 runtime / Ascend950PR_9579（真机） | `pattern-library/traps-runtime.md`（条目 `TRAP-vcvt-sat-false-overflow`，与 review 阶段提出的补录条目同一 key） |
| DESIGN A1 `(g·u)/(1+e^{-g})` 在设备侧与 golden 逐位一致（4096 元素/dtype，max_ulp=0），在 8.4M 元素的真实 workload 上最多 1 个 16-bit ulp —— 5 算符替代 6 算符的等价性由设备实测背书，而非仅 torch fp32 回放 | D | `examples/shared_expert_mlp/_silu_mul_fused_kernel/repro/CONST-950-silu-mul-a1-one-ulp.py`（`exact 4096/4096, max_ulp=0` ×2 dtype）；本 kernel L0-1/L0-2 的 `diff=3.125e-02`（= bf16 1 ulp @ 4.0–8.0）、`max_ulp=1 / frac_gt1ulp=0` | `cd examples/shared_expert_mlp/_silu_mul_fused_kernel && ASCEND_RT_VISIBLE_DEVICES=0 /workspace/venv-ascend950/bin/python repro/CONST-950-silu-mul-a1-one-ulp.py` | 同上 | `pattern-library/constants.md` 或 `tilelang-op-design/references/algorithm-research.md`（elementwise 行：A1 的设备侧 ULP 代价） |
| 逐元素迁移的 tile 必须由 kernel 侧夹取：wrapper 的 `(block_m=128, block_n=256)` 需要 `2·3·128·256·2 = 393 216 B > 253 952 B`；夹取规则 `bn = max(64, min(block_n,512)); bn -= bn % 64; bm = 16 if M ≥ 64 else 1` 落到 49 152 B。`bn < 64` 会让 `nchunk = 0` → 计算被静默跳过、把 UB 垃圾写回 GM，故 `max(VL, …)` 这一层不能省 | P | `_silu_mul_fused_kernel.py#geometry`（规则 1/2 + `assert`）；实测 hint 映射：`(128,256)→(bm16,bn256,49152 B)`、`(32,1024)→(16,512,98304 B)`、`(8,32)→(16,64,12288 B)`；`REVIEW.md#3` 建议 1/2 | `cd examples/shared_expert_mlp/_silu_mul_fused_kernel && /workspace/venv-ascend950/bin/python -c "import importlib.util as u;s=u.spec_from_file_location('m','_silu_mul_fused_kernel.py');m=u.module_from_spec(s);s.loader.exec_module(m);print([m.geometry(4096,2048,*h,'bfloat16') for h in ((128,256),(32,1024),(8,32))])"` | 无卡（纯构建期几何，torch CPU 可跑） | `examples/TileOPs/.agents/skills/add-npu-op`（K9/K11 适配条目）或 pattern-library 逐元素迁移条目 |
| `vsts(dist="PK_B32")` 之后接 `T.copy`（UB→GM）**不需要** `T.simd.mem_bar("VST_VLD")`：那条屏障只针对随后用 `vld` 读同一 buffer（VST→VLD）的情形。VST→MTE2 写回由自动同步覆盖，用 3 次重复启动的逐位一致性验证比"预防性加屏障"更可靠 | P | 本 kernel `run_L0` L0-9/L0-10、`run_L1` L1-1/2/4/5/7 的 `repeat=3` 全部 `det=True`（`torch.equal` 位比）；33 条用例 0 失败；对照 `examples/engram_gate_conv/.../repro/TRAP-950-simd-ub-store-load.py`（VST→VLD 才需 mem_bar） | `cd examples/shared_expert_mlp/_silu_mul_fused_kernel && ASCEND_RT_VISIBLE_DEVICES=0 /workspace/venv-ascend950/bin/python _silu_mul_fused_kernel.py --level all`（看 `det3` 两例与 L1 repeat 列无 `non-deterministic`） | 同上（真机） | `pattern-library/traps-runtime.md`（与 `TRAP-950-simd-ub-store-load` 互补的「不需要屏障」边界） |
| 5-op SIMD 链的实测墙钟：kimi-k2-t512（56.6 MB 读+写）0.14 ms、ds-v3-t4096（50.3 MB）0.12 ms、kimi-k2-t1（0.11 MB，grid=72）0.17 ms —— 聚合带宽约 400 GB/s，decode 段由发射/启动主导。口径：Python `perf_counter` 包住单次调用 + `torch.npu.synchronize()`，含 `out_idx` 输出分配，非 msprof | D | 本 kernel `--level L2`/`--level L1` 输出的 `launch=` 列（warm cache，逐用例；正式口径交 Stage 4 的 `msprof_task_duration`） | `cd examples/shared_expert_mlp/_silu_mul_fused_kernel && ASCEND_RT_VISIBLE_DEVICES=0 /workspace/venv-ascend950/bin/python _silu_mul_fused_kernel.py --level L2` | 同上（真机，CANN 9.2.1） | `pattern-library/constants.md`（待 Stage 4 用 msprof 复核后回填） |

### Transferable Lessons（可迁移教训）

- 16-bit 逐元素 kernel 的 golden 就写 `(F.silu(gate.float()) * up.float()).to(dtype)`，别为了"对齐设备"改成设备的 `(g*u)/(1+exp(-g))` 形式：本任务实测两者在 4096 元素上逐位相同、在 8.4M 元素上最多 1 个输出 ulp。
- 收窄 `vcvt(sat=False)` 的溢出是 IEEE `inf`（不是文档写的 wrap），所以 golden 的 `.to(dtype)` 是对的；但只有 `|y| > dtype max` 的用例（如 fp16 `silu(10)*65504`）能区分 inf 与 wrap，随机输入永远测不到——凡 `y = 激活(x) * 放大系数` 的算子必须显式加这条 Boundary。
- 逐元素迁移不要照搬 wrapper 的 `block_m = 128`：UB 预算 `2 version × 3 buffer × bm·bn·2 B ≤ 253 952 B` 要求 kernel 侧夹取；并且 `bn` 下限必须是 64（`bn < 64` 时 `nchunk = bn//64 = 0`，计算被静默跳过、UB 垃圾被写回 GM），一行 `bn = max(64, min(block_n, 512))` 同时消掉溢出与静默空算两个坑。
- `vsts(dist="PK_B32")` 后用 `T.copy` 写回 GM 不需要 `mem_bar("VST_VLD")`（那只管随后的 `vld` 读同一 buffer）；判据用 3 次重复启动的位一致，不要凭直觉加屏障，也不要因为别处的 VST→VLD 陷阱而误加。
- 迁移 harness 的 `op_name` 常带前导下划线，直接拼 `golden_{op}` 会得到双下划线函数名；实现时写可读的 `golden_<name>` 再加一行别名即可，不要让命名约定挡住门禁。
- `T.Kernel` 的一维 grid 与 `nsp/tpb` 几何对逐元素算子同样够用：`nsp` 取 `nt` 的因子中使 `mt·nsp` 最接近 `4 × vector_cores` 者，本任务 5 条 manifest workload 全部落在 1.29–4.57 波、与设计表逐行一致，无需 persistent 化。
