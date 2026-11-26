# RETROSPECTIVE -- mla_dense_bwd (migration-harness)

## Stage 5 (integrate) Retrospective — harness / attempt 1 — 2026-11-23

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Integrate-flow | `integrate_kernel.py::rewrite_wrapper_import` 只替换 `from .{extracted_module} import ...` 这一行，不清理 scaffold 在 import 前写的 “Part A: the GPU TileLang kernel factories …” 注释块，也不处理模块 docstring 里 “Pre-Stage-3 note: the imported factories are still the GPU (CUDA target) implementations … Running forward before that rewrite fails”。集成后 wrapper 顶部的两处文字与新切换块、与真实 import 目标自相矛盾，后续 Stage 4/6 读 wrapper 会被误导。 | `examples/TileOPs/tileops/kernels/attention/mla_dense_bwd/mla_dense_bwd.py#45-70`（原文见 `history_version/mla_dense_bwd_s5_attempt0.py`）；integrate_kernel.py 本次输出 `[wrapper] ... rewritten`，`[warn]` 无任何提示 | 让 `rewrite_wrapper_import` 在替换 import 时一并定位 scaffold 的 “Part A / Pre-Stage-3 note” 段落，标记为「已集成」或删除；至少在 `[warn]` 中提示残留注释需人工清理 | R |
| Integrate-flow | 已知修复目录（agent 内嵌表）缺一条高频项：「Kernel expected N inputs, but M are provided」= wrapper 仍按 GPU 二级 tiling 调用（`self.kernel(block_n, block_m, num_stages)`），而 Stage 3 冻结工厂已返回编译好的程序。该症状首次 report 即出现，8/10 用例同一条报错，但从报错文本无法直接看出是 wrapper 而非 kernel。 | `reports/tileops/20261123_113734_679663_MLADenseBwdOp/report.md#Failed Cases`；修复见 `mla_dense_bwd_kernel/integration_log.md#调试历史 attempt 1` | 在 integrator agent 的「已知修复目录」增加一行：`Kernel expected N inputs, but M are provided` → 检查 wrapper `_compile` 是否多套了一层 config 调用 / `default_config` 是否残留 GPU tiling 键 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 迁移 op 的 wrapper 必须按“集成后工厂返回形态”校准调用层级：Stage 3 冻结工厂返回 `tilelang.compile(...)` 编译好的程序（`_bwd_kernel` 11 输入、`_delta_kernel` `out_idx=[2]`、`_dq_land_kernel` `out_idx=[1]`），GPU wrapper 的 `self.kernel(block_n, block_m, num_stages)` 二级调用会把 3 个 int 当输入实参，报 `Kernel expected 11 inputs, but 3 are provided`；修法是 `default_config={}` + `_compile` 直接返回 `self.kernel` | P | `mla_dense_bwd_kernel/integration_log.md#调试历史 attempt 1`；`history_version/mla_dense_bwd_s5_attempt1.py` vs `mla_dense_bwd.py#_compile` | `pytest tests/ops/test_mla_dense_bwd.py -k smoke-bwd-bf16`（修前 8/10 fail，修后 10/10 pass） | tilelang 0.1.15, CANN9.2.1, Ascend950PR_9579 | pattern-library（migration-integration） |
| 多 launch（一次调用启动 3 个 kernel）算子的 TileOPs msprof 采集失效模式：`msprof op --kernel-name` 无法用单一名字过滤整个调用，未声明 `msprof_kernel_name` 时 capture 含 5 个 op（`Range_*`、`ZerosLike_*`、`_bwd_main_kernel`、`_delta_main_kernel`、`_dq_land_main_kernel`），报告把 `Range_` 的 208 us 记成算子延迟（真实 `events` 口径 ds-v3-b2-8k 为 455066 us），并标 `invalid tileops profile: unfiltered capture contains 5 distinct ops` | D | `examples/TileOPs/reports/tileops/20261123_114002_550215_MLADenseBwdOp/run.json#performance.cases[0..1]` 与 `#warnings` | `cd examples/TileOPs && TILEOPS_TASK_DIR=<task_dir> python -m tileops.reporting.cli run --op MLADenseBwdOp --prof-mode msprof`（5 case 中 2 条 msprof 无效、3 条回落 events） | tilelang 0.1.15, CANN9.2.1, Ascend950PR_9579, msprof 25.7.rc1.10 | pattern-library / bottleneck-patterns |
| 反例（C）：把 benchmark 的 “partial + CLI 非零退出码” 当集成失败会误判——本次 `status=partial` 纯由 profiling 口径造成（correctness 10/10、benchmark pytest 5/5 passed、0 failed），必须读 `run.json` 的 `summary.correctness_passed` 与 `invalid_profile_count` 区分 | C | `reports/tileops/20261123_114002_550215_MLADenseBwdOp/run.json#summary`（`correctness_passed=true`、`benchmark_passed=false`、`invalid_profile_count=2`） | 同上 report 命令，看 CLI `status=partial` 与 run.json summary 的对比 | tilelang 0.1.15, CANN9.2.1, Ascend950PR_9579 | 待定（integrator agent 判定规则已含此条，可提升为案例索引） |

### Transferable Lessons（可迁移教训）

- 迁移算子的 wrapper 里，凡是 GPU 侧存在 `self.kernel(block_n, block_m, num_stages)` 这类“二级 tiling 调用”的，集成后必须核对 Stage 3 工厂是否已把 tiling 冻结进 `tilelang.compile(...)`；残留二级调用会把 int 当 tensor 实参，报 `Kernel expected N inputs, but M are provided`，且 `default_config` 的 GPU tiling 键应清空。
- 三段式（delta → main → land）这一类多 launch 算子的 msprof 采集天然无效：要么按每个 `*_main_kernel` 分别采集后求和，要么在 kernel 侧合并；不要相信 report 里 msprof 行的延迟数字，先看 `run.json` 的 `profiler_valid`。
- 集成脚本改写 wrapper import 后，scaffold 留下的“仍是 GPU 实现 / forward 尚不可用”等历史注释会与新状态冲突，集成时顺手清理（只改注释，不动行为），否则后续 Stage 会误判当前实现状态。

## Stage 5 (integrate) Retrospective — harness / Cube 化重设计后的重新集成 — 2026-11-25

重组上下文：`_bwd_kernel` 已按 Cube 化重设计（5 个 GEMM 全显式 L0、`trans_A` 复用 `dsT`、`Nk=96`、
2-tile 合并写、`num_stages=2`、4-D batch 视图 per-request 写回），本轮只做重新集成 + 验证。
集成脚本幂等（`wrapper_rewritten=false`），首次 report 即 correctness 10/10 且全部 msprof 有效，
未消耗任何调试 attempt。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Integrate-flow | 算子的计算单元从 Vector 改到 Cube 后，`tileops/reporting/compute_units.json` 仍是重设计前的实测证据（`MLADenseBwdOp: unit=vector, aic_cube_ratio 8.3%/aiv_vec_ratio 73.5%`），导致 MFU 被除以 Vector 峰值 47 TFLOPS：93.0 TFLOP/s → 197.8%，触发 `physically-impossible: achieved compute exceeds the peak` 熔断，`invalid_acceptance_count=5`、`benchmark_passed=false`、report `status=partial`。该文件又被 `tests/test_reporting_acceptance.py::test_compute_unit_map_covers_every_migrated_operator` 钉死为 `"vector"`，Stage 5 的写权限（kernel + wrapper）无法同时修 map 与测试期望，因此 Cube 化算子在本 harness 下永远拿不到 `status=passed`。 | `reports/tileops/20261125_043620_849427_MLADenseBwdOp/run.json#summary`（`invalid_profile_count=0` 但 `invalid_acceptance_count=5`）与 `#warnings`；`tileops/reporting/compute_units.json#MLADenseBwdOp`；`tests/test_reporting_acceptance.py:67-80`；本轮 capture `artifacts/msprof/.../ArithmeticUtilization_*.csv`（`aic_cube_ratio 0.2553`、`aiv_vec_ratio 0.4426`） | 两条路：(a) 落地 `compute_units.json` 注释里已承诺的「Per-case msprof metadata overrides this map when present」——由 msprof 采集的 `aic_cube_ratio`/`aiv_vec_ratio` 推出 per-case `compute_unit`（`tileops/benchmark/msprof.py` 写入 metadata，`tileops/reporting/analyzer.py::_analyze_performance` 透传到 `case["compute_unit"]`），map 只作缺省；(b) 若维持静态 map，则把 «算子计算单元变更» 纳入 Stage 5 允许修改清单（map + 钉死测试同步改），并在 integrator agent 的「输出格式要求」里补一行 `acceptance_unit_updated` | R |
| Integrate-flow | integrator agent 的输入契约写下「确认提取/同步后的 `_mla_dense_bwd_kernels.py` 与任务目录 `_bwd_kernel.py` 一致」，但 `integrate_kernel.py` 从不重写该 GPU 抽取文件（它只 copy `{func}.py` 到 `{op_slug}_kernel/` 并生成聚合 `__init__.py`），wrapper 在 Round 1 后也已改为从 `.{op_slug}_kernel` 导入，因此该文件只是迁移记录、本就应当停留在 GPU 版本。按字面执行会误判「集成不一致」。 | `.agents/skills/add-npu-op/scripts/integrate_kernel.py:405-470`（无 extracted 文件写入）；`grep -rn _mla_dense_bwd_kernels`（仅 wrapper 注释与 `history_version/` 旧备份命中）；`mla_dense_bwd_kernel/integration_log.md#一致性核对` | 把该条改为「核对 `{op_slug}_kernel/{func}.py` 与 `examples/{op_slug}/{func}/{func}.py` 一致（md5）+ wrapper import 指向 `.{op_slug}_kernel`」，并注明 `{op_slug}_kernels.py` 是迁移记录、不参与一致性判定 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 多 launch 算子改用 `t_op_us`（一次调用所有 kernel 求和）后，msprof 采集从「无效」转为「有效」：`MLADenseBwdOp` 5 个 case 的 `invalid_profile_count` 由 2 → 0、`profiler_fallback_count` 由 3 → 0；`ds-v3-b2-8k` `T_op = 154251.2 us`（`_bwd_main_kernel` 150938.5 + `_delta_main_kernel` 715.9 + `_dq_land_main_kernel` 1639.0 = 153293.4，框架 helper `ZerosLike` 953.7 + `Range` 4.0），达 93.0 TFLOP/s；重设计前同口径 455066.6 us → **2.95x**（dev 自测 150.7 ms/3.02x 复现）；其余 workload 同比例：`kimi-k2-8k` 113774.4→38420.6（2.96x）、`ds-v2-lite-3k` 16621.7→6294.2（2.64x） | D | `reports/tileops/20261125_043620_849427_MLADenseBwdOp/{run.json,benchmark.json}`（`msprof.t_op_breakdown`）；旧值 `reports/tileops/20261123_114002_550215_MLADenseBwdOp/run.json` | `cd examples/TileOPs && TILEOPS_TASK_DIR=<task_dir> TMPDIR=/workspace python -m tileops.reporting.cli run --op MLADenseBwdOp --prof-mode msprof` | tilelang 0.1.15, CANN9.2.1, Ascend950PR_9579, msprof 25.7.rc1.10 | pattern-library / bottleneck-patterns（多 launch 采集口径） |
| 生产 shape 的 per-request 写回窗口要**从集成后的 op 路径**验证（op → wrapper → kernel），不是从 kernel 直接 launch 验证：复用开发阶段 `--level big` 的 chunked float32 前向（`_stats_npu` 同型循环，额外返回 `o`，因为 op 侧 `delta = rowsum(o*do)` 自行重算）+ `golden_head` 切片 fp64 参考，在 B=2/S=8192/H=128 上检查两路 request 的 dq/dk/dv 非零、request 1 的 key 0..63 全被填充（`dk` zeros 0/1572864、`dv` 0/1048576），并按 `(request, head)` 4 个切片比对 ~16.8M 元素 0 超差，最差 gap dq 4.9e-3 / dk 1.0e-2 / dv 1.8e-2；op 级实测 154.07 ms 与 msprof 154.25 ms 吻合 | P | `examples/mla_dense_bwd/_bwd_kernel/evidence/s5/{check_b2s8k_op.py,check_b2s8k_op.log}`；`mla_dense_bwd_kernel/integration_log.md#B=2 / S=8192 / H=128 non-silent check` | `/workspace/venv-ascend950/bin/python examples/mla_dense_bwd/_bwd_kernel/evidence/s5/check_b2s8k_op.py --heads 0,127 --reps 2` | tilelang 0.1.15, CANN9.2.1, Ascend950PR_9579 | pattern-library（migration-integration 验证手法） |
| 带有「前向统计量入参」（`lse`/`delta`）的 kernel，若探针直接拿「重算统计量」的 fp64 golden 比对，会把前向统计精度混进 kernel 误差：同一份输出 vs 重算 `lse` 的 golden 有 dq 5 / dk 17 / dv 9（共 16.8M）元素超 dev 底线（atol 5e-3），改为固定 kernel 实际消费的 `lse`/`delta` 再算参考后三者全部归 0。重设计前的 `run_big` 是靠「用 fp64 统计量覆盖 `lse`/`delta`」规避同一问题 | P | `check_b2s8k_op.py::golden_head_given_stats` + `check_b2s8k_op.log`（两行 `elements over ...` 对比） | 同上探针命令；`grep "elements over" evidence/s5/check_b2s8k_op.log` | tilelang 0.1.15, CANN9.2.1, Ascend950PR_9579 | pattern-library（精度诊断判据） |
| 反例（C）：`invalid_*` 里的 `invalid_profile` 与 `invalid_acceptance` 语义完全不同——Cube 化重设计后 `invalid_profile_count=0`（采集有效）却被 `invalid_acceptance_count=5`（算力超过记录峰值）判 `partial`。看到 "exceeds the physical peak" 先去读当次 capture 的 `aic_cube_ratio`/`aiv_vec_ratio`（本轮 25.5% / 44.3%，且 `aic_cube_fp_ratio == aic_cube_ratio`、int 0），确认是计算单元记录过期，而不是 grid 截断或 kernel 回归；不要因此改 kernel | C | `run.json#summary`、`#warnings`；`artifacts/msprof/.../_bwd_main_kernel/0/ArithmeticUtilization_*.csv` | 同上 report 命令后 `grep -i "exceed the physical peak" run.json` 与读取 capture CSV | tilelang 0.1.15, CANN9.2.1, Ascend950PR_9579 | 待定（建议并入 Migrate-flow 的 report 判定规则） |

### Transferable Lessons（可迁移教训）

- 多 launch 算子一律引用 `t_op_us`（一次调用所有 kernel 求和），不要引用 report 的 `latency_us`：后者是「过滤后单个 op」的中位数，本任务里它落在框架 helper `Range_*` 上（206-954 us），与 154 ms 的真实算子时间差 2-3 个数量级。
- 看到 acceptance `invalid` + "exceeds the physical peak" 而刚做过「Vector → Cube」重设计时，先读当次 msprof capture 的 `aic_cube_ratio`/`aiv_vec_ratio`：`tileops/reporting/compute_units.json` 的计算单元是**遗留实测值**（且被 `tests/test_reporting_acceptance.py` 钉死），这是报告元数据过期，不是 kernel 回归，也不是 grid 截断；Stage 5 无权同时改 map 与测试期望，应作为 R 提案交 conductor。
- 验证「per-request 写回窗口」（4-D batch 视图、避免 request b 的尾块覆盖 request b+1 的首行）必须在**集成后的 op 路径**上做：直接 launch kernel 会绕开 wrapper 的 `reshape(B, S, H, D)`；判据是 request 1 的 key 0..63 全非零 + 按 `(request, head)` 切片比对，而不是全局 max diff。
- kernel 若把前向统计量（`lse`/`delta`）当入参，精度探针要比对两次：一次用「重算统计量」的 golden（混合了统计精度），一次固定 kernel 实际消费的统计量（隔离 kernel 误差）；只有后者能判定 kernel 是否有问题。
- 集成包里的 `_mla_dense_bwd_kernels.py` 是 GPU 抽取记录，`integrate_kernel.py` 不会更新它、运行期也没有任何模块 import 它；一致性核对只针对 `{op_slug}_kernel/{func}.py` 与任务目录产物。

## Stage 5 (integrate) Retrospective — harness / A5 多缓冲轮换后的重新集成 — 2026-11-26

重组上下文：`_bwd_kernel` 已按 A5/v4 轮换重写（`GB(k-1)` 滞后、`T.serial` 两份静态半体、
`q/qT/do/doT/lse/delta/pT/dsT` 各持 `*_s0`/`*_s1` 两个静态版本、drain 按 `n_it` 奇偶、
`clear_accum=(kq-1==loop_st)`、`dq` 写 `(kq-1)*64`，`T.Pipelined`/`_STAGES` 删除）。本轮只做重新集成
+ 验证：集成脚本幂等（`wrapper_rewritten=false`、wrapper md5 前后一致），首次 report 即
correctness 10/10、5/5 msprof capture 有效、`invalid_*` 全 0，未消耗任何调试 attempt。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Integrate-flow | `integrate_kernel.py` 的 import smoke 只 `import_module` 聚合包并检查名字可调用，**不调用工厂**，也不比对「wrapper 的调用签名」与「集成后工厂参数表」；本轮 v4 删掉 `T.Pipelined`/`_STAGES` 并把工厂尾参改为 `n_versions=2`，兼容性全靠人工用 `ast` 读参数表确认（前 9 个位置参数与源顺序一致才安全）。若某轮出现「参数位移」（如把 `n_versions` 插在中段），smoke 仍会 pass，错误要到 pytest 才暴露，被误计入 Stage 5 调试 attempt。 | `examples/TileOPs/.agents/skills/add-npu-op/scripts/integrate_kernel.py#smoke_test`（仅 import 与 `hasattr`）；本轮核对记录 `mla_dense_bwd_kernel/integration_log.md#一致性核对`；工厂参数表 `_bwd_kernel.py:299-302` | 在 smoke 阶段加一条「wrapper 调用点 vs 工厂签名」的静态比对：对 `_run`/`_compile` 里每个 `func(...)` 取位置/关键字实参个数与前缀名，与 `inspect.signature(集成后工厂)` 的必填前缀匹配，不匹配即 `[error]`（比 import 强一个量级，成本仍是一次导入） | R |
| Integrate-flow | integrator agent 输出契约要求「record `{func}_DESIGN.md`」，但 `integrate_kernel.py` 每轮整文件覆盖它；当 DESIGN.md 在 Stage 3/4 被改写（本轮 v4 重写 242KB→新版本）后，**集成包内的 `*_DESIGN.md` 与报告/日志里引用的行号会一起漂移**，复盘与 evolver 用行号溯源时会指错。 | `mla_dense_bwd_kernel/_bwd_kernel_DESIGN.md` 本轮被覆盖（v4）；`integration_log.md#一致性核对` 只记 md5 不记版本号 | 集成日志侧记录「DESIGN.md md5 + 对应的 Stage 3 版本标记（如 DESIGN.md 6.3 B / 13.3 章节号）」而不是行号；规范文件里把「行号溯源」改为「章节号 + md5 溯源」 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| A5 轮换（`GB(k-1)` 滞后 + `T.serial` 双静态半体）在集成路径上完整复现收益：`ds-v3-b2-8k` `T_op` 由 154419.2 → **99091.7 us**（`_bwd_main_kernel` 95762.4 + `_delta_main_kernel` 715.4 + `_dq_land_main_kernel` 1640.5 = 98118.3，另 `ZerosLike` 968.9 + `Range` 4.4），cube 39.1%、MFU 38.3%、144.7 TFLOP/s，**-35.8%**；同口径 `kimi-k2-8k` 38421.1 → 24860.5（-35.3%，MFU 38.1%）、`ds-v3-4k` bf16 20571.7 → 13613.7（-33.8%，MFU 34.8%）、`ds-v2-lite-3k` 6296.7 → 3747.2（-40.5%，MFU 35.6%）；5/5 capture 有效、`invalid_profile_count=0`、`invalid_acceptance_count=0`、report `status=passed` | D | `examples/TileOPs/reports/tileops/20261126_040450_369928_MLADenseBwdOp/{run.json,benchmark.json}`（`performance.cases[*].msprof.t_op_breakdown` / `t_op_us`）；对照 `reports/tileops/20261125_044632_077346_MLADenseBwdOp/run.json`；`mla_dense_bwd_kernel/integration_log.md#验证结果（A5/v4，最终）` | `cd examples/TileOPs && TILEOPS_TASK_DIR=/workspace/tilelang-mlir-ascend-950/examples/mla_dense_bwd TMPDIR=/workspace /workspace/venv-ascend950/bin/python -m tileops.reporting.cli run --op MLADenseBwdOp --prof-mode msprof` | tilelang 0.1.15, CANN9.2.1, Ascend950PR_9579, msprof 25.7.rc1.10 | pattern-library / bottleneck-patterns（A5 轮换收益，含 per-workload 绝对数） |
| 工厂签名演化时先做「参数表前缀比对」再决定要不要动 wrapper：v4 把 `_bwd_kernel(...)` 尾参从 `num_stages` 改成 `n_versions=2`、删除 `T.Pipelined`/`_STAGES`，但前 9 个位置参数仍是源算子顺序（`batch, heads, dim_qk, dim_v, dense_seqlen_q, dense_seqlen_kv, is_causal, scale, dtype`），且 wrapper 的 `default_config={}` 从不传尾参，因此 `_run`/`_compile` 完全无需适配（wrapper md5 `367eeb05...` 前后不变）。判据用 `ast` 取 `FunctionDef.args` 与 wrapper 调用点的实参个数/名前缀，而不是试跑撞报错 | P | `mla_dense_bwd_kernel/_bwd_kernel.py:299-302`（参数表）、`mla_dense_bwd_kernel/mla_dense_bwd.py#_compile`（`_bwd_kernel(...)` 9 实参）、`mla_dense_bwd_kernel/integration_log.md#一致性核对`（md5 前后一致） | `python -c "import ast;t=ast.parse(open('mla_dense_bwd_kernel/_bwd_kernel.py').read());print([a.arg for n in ast.walk(t) if isinstance(n,ast.FunctionDef) and n.name=='_bwd_kernel' for a in n.args.args])"` | tilelang 0.1.15, CANN9.2.1, Ascend950PR_9579 | pattern-library（migration-integration：工厂演化时的 wrapper 核对法） |
| 旋转调度（`T.serial` + 静态双版本 + 滞后消费）会让 Z3 调度器报 `solver returned unsat, falling back to topological sort`，这是**设计预期**而非缺陷：本轮 `benchmark.log` 中该警告与 5/5 有效 msprof capture、正确性 10/10 共存，`invalid_*` 全 0。看到该行不必进入调试闭环 | D | `reports/tileops/20261126_040450_369928_MLADenseBwdOp/pytest/benchmark.log`（`Warning: Python Z3 scheduler failed ... falling back to topological sort`）；同一 run 的 `run.json#summary`（`invalid_profile_count=0`）；机制说明 `_bwd_kernel.py:47-52` | 同 report 命令后 `grep -c "Z3 scheduling failed" reports/tileops/<run>/pytest/benchmark.log` 并核对 `run.json#summary` | tilelang 0.1.15, CANN9.2.1, Ascend950PR_9579 | bottleneck-patterns（编译期警告判据） |
| 反例（C）：**不能拿上一轮 report 的绝对延迟当本轮验收期望值**——Cube 轮记录 `ds-v3-b2-8k` 154.4 ms，A5 轮真实值 99.1 ms；若照旧值卡阈值会把 -35.8% 的收益当噪声，反之若 A5 回归到 154 ms 而仍按旧值判定则会漏掉回归。验收必须引用**当次** `run.json`（`integration_report.json` 只指向本轮 run-id），跨轮比较只作信息项 | C | `integration_report.json`（本轮 run-id）；两轮 `run.json#performance.cases[*].msprof.t_op_us` 对比 | `diff <(python -c "…20261125_044632…") <(python -c "…20261126_040450…")`（见 integration_log 表格） | tilelang 0.1.15, CANN9.2.1, Ascend950PR_9579 | 待定（integrator 判定规则可并入） |

### Transferable Lessons（可迁移教训）

- 工厂签名演化（删 `T.Pipelined`/`_STAGES`、尾参换 `n_versions`）不等于 wrapper 必须改：先把集成后工厂的 `ast` 参数表与 wrapper 调用点的实参个数/名前缀比一遍，前 9 个位置参数一致且尾参有默认值时，wrapper 保持不动最省事也最不容易出错；改 wrapper 前一定先 `cp` 到 `history_version/`。
- 验收数字只认本轮 `run.json`：`integration_report.json` 必须指向本轮 run-id，跨轮绝对延迟只作信息项；旧轮数字（如 154.4 ms）既不能当期望值，也不能当回归基线之外的判据。
- 多个 kernel 的 op 依然引用 `t_op_us`（一次调用所有 kernel 求和），本任务的 `latency_us` 落在 helper `ZerosLike/Range` 上（子毫秒级），与真实算子时间差 1-2 个数量级。
- 旋转/静态多版本调度在编译日志里的 `Z3 ... unsat, falling back to topological sort` 是设计取舍的可见痕迹（版本化 buffer 的访问重写无法表达滞后消费），只要 capture 有效、`invalid_*` 为 0 就不属于编译失败，不要据此进调试闭环。
- per-request 写回窗口的验证要跑在集成后的 op 路径上（op → wrapper → kernel），判据是「request 1 的 key 0..63 全非零 + 按 (request, head) 切片比参考」，并且参考要固定 kernel 实际消费的 `lse`/`delta`（本轮 4 切片 ~16.8M 元素 0 超差，最差 gap dq 9.8e-3 / dk 1.2e-2 / dv 1.2e-2）。
