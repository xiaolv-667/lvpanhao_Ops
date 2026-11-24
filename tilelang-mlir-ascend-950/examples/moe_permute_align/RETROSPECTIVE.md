## Stage 5 (integrate) Retrospective — migration-harness, 3 kernels — 2026-11-23

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Integrate-flow | `integrate_kernel.py` 的 `rewrite_wrapper_import` 只替换 `from .{extracted} import ...` **语句本身**，不检测 scaffold wrapper 里已存在的 baseline/perf_opt 切换块（header 注释 + `# --- baseline` 行 + perf_opt 注释块），于是整块被写两次（header ×2、perf_opt 注释 ×2），需人工清理 | `moe_permute_align_kernel/history_version/moe_permute_align_wrapper_postintegrate.py`（第 63–101 行两块紧邻重复）；清理后 = 当前 `moe_permute_align.py` | 在 `integrate_kernel.py` 的 `rewrite_wrapper_import` 中先匹配既有切换块（如 `# --- baseline (Stage 3)` 或 `from \.{op_slug}_kernel import`）并整块替换，替代仅替换 import 语句；并在 add-npu-op SKILL 的 S5 步骤注明「已含切换块的 wrapper 需整块替换」 | R |
| Integrate-flow | 集成脚本对 kernel 文件与 `{func}_DESIGN.md` 都是**整文件覆盖**：集成期在集成包内做的 kernel 修复（本任务修了 `_make_align_kernel.py` 的 `threads=2*e_pad`）在下次重跑脚本时**静默回退**，没有任何 diff/保留提示 | 集成包 `_make_align_kernel.py`=修复版（16:47），conductor 产物 `examples/moe_permute_align/_make_align_kernel/_make_align_kernel.py` 仍是 `threads=2 * e_pad`（grep `2 \* e_pad` 两处）；`integrate_kernel.py#L444-L460` 为 `shutil.copy2` 无条件覆盖 | 集成脚本在目标文件已存在且内容与源不同时打印 `[warn] overwriting integrated <func>.py that diverged from the conductor product; Stage-5 fixes will be lost`；并在 integrator agent 说明中要求把集成期 kernel 修复回写 conductor 产物 | R |
| Integrate-flow | 「单算子 report」对**一次调用跑多个 kernel** 的算子无有效 profile 归因口径：msprof 未过滤 capture 含 2 个 distinct op 即判 invalid，12/12 benchmark 用例全废，但 report 仍写出 `latency_us`（取未过滤 capture 的 max），容易被误读为有效时延 | `reports/tileops/20261123_164821_903151_MoEPermuteAlignFwdOp/run.json#performance.cases[0]`：`profiler_valid=false`、`invalid_reason="unfiltered capture contains 2 distinct ops"`、`captured_op_names=["_scatter_main_kernel","main_kernel"]`、`latency_us=23.354`、`min_latency_us=2.622` | 在 add-npu-op SKILL 的 S5 benchmark 小节补一条：「多 kernel 算子（align+scatter 等）的 msprof capture 必然含多个 op，`kernel_filter` 无法自动消歧时按 partial 记录，`latency_us` 非有效单 kernel 时延，禁止当性能基线」；并给 benchmark 侧提多 kernel 归因（按 op 名分组求和）的改造提案 | R |
| Info-source | `statectl platform` 的 `integration.script` 值 `.agents/skills/add-npu-op/scripts/integrate_kernel.py` 缺 `examples/TileOPs/` 前缀，实际脚本在 TileOPs 工程内且必须在 `examples/TileOPs/` 下运行；直接按该相对路径从仓库根找不到文件 | `statectl platform --task-dir examples/moe_permute_align` 输出 `integration.script=.agents/skills/add-npu-op/scripts/integrate_kernel.py`；真实文件 `examples/TileOPs/.agents/skills/add-npu-op/scripts/integrate_kernel.py` | 把 `integration.script` 改为相对 TileOPs 工程根的路径（或写成 `examples/TileOPs/.agents/...`），与 `integration.scaffold_skill` 的写法（已带 `examples/TileOPs/` 前缀）保持一致 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **SIMT `T.SimtVF(threads=)` 上限 2048（LAUNCH_BOUND）实测**：`threads=2304` 编译期硬失败，报 `error: Maximum number of SIMT threads returned by LAUNCH_BOUND must be an integer constant in range (0, 2048]`；把超宽单趟拆成多趟 `threads<=2048` 等价写后编译通过且数值不变 | D | 失败：`reports/tileops/20261123_164406_365988_MoEPermuteAlignFwdOp/run.json#correctness.cases[experts-past-threads]`（`_compile_align_partial`，`E_pad=1152` ⇒ `2*e_pad=2304`）；修复后：同 shape 在 `reports/.../20261123_164821_903151_.../run.json` 通过 | `ASCEND_RT_VISIBLE_DEVICES=0 /workspace/venv-ascend950/bin/python /tmp/opencode/s5/repro_launch_bound.py`（`E=1100,B=64,numel=512`：修前 `[fail] align_partial ... LAUNCH_BOUND`，修后 `[ok]`）；端到端 `pytest tests/ops/test_moe_permute_align.py -k experts-past-threads` | tilelang 0.1.15 + CANN 9.2.1（bisheng 12.3.0） + Ascend950PR_9579 / dav-3510（真机） | `.agents/platforms/toolchains/ascend/knowledge/PL-950-vector-execution.md`（SIMT 线程宽度小节：**上限 2048，且线程数决定 LAUNCH_BOUND 常量**；与 Stage 3 small_batch 复盘的「只用验证过的 32/64/256」互补——这里给出的是硬上限而非性能建议） |
| 集成期修复手法：**超宽 SimtVF 单趟清零 → 等宽多趟清零**（写集合/写序不变，与既有「上/下半两趟」写法同构），用来在不动语义的前提下绕开 SIMT 宽度硬上限 | P | 集成包 `_make_align_kernel.py` 两处（`_compile_single` 与 `_compile_align_partial`）把 `with T.SimtVF(threads=2*e_pad): for e in T.Parallel(2*e_pad)` 拆成两个 `threads=e_pad` 趟 | 同上 repro 脚本 + `/tmp/opencode/s5/check_small_path_epads.py`（`numel=100,E=1100,B=1` 走 align 内部小路径 `path=S`，e_pad 同为 1152，pass） | tilelang 0.1.15 + CANN 9.2.1 + Ascend950PR_9579（真机） | `.agents/skills/tilelang-op-develop/references/`（失败分类：编译期 LAUNCH_BOUND →「降线程宽度、拆多趟」） |
| **align 内部小路径（`_use_small_path`）与 wrapper 选路是两个独立判据**：`numel<1024 且 E<=32` 只决定 wrapper 走 fused small_batch 还是 align+scatter；align 内部还会按 `hist+ids+fill <= PATH_S_INSTR_BUDGET` 再选单 launch 路径，因此 `E>1024` 的 `2*e_pad` 缺陷对**两条** align 路径都成立，只测 wrapper 的 large 路径会漏掉小路径 | C | `_make_align_kernel.py#_use_small_path`；`(numel=100,E=1100,B=1)` 实测 `align_path=S`（wrapper 侧因 E>32 仍走 align+scatter） | `/tmp/opencode/s5/check_small_path_epads.py` | tilelang 0.1.15 + CANN 9.2.1 + Ascend950PR_9579（真机） | `examples/moe_permute_align/_make_align_kernel/DESIGN.md`（路径选择小节复核） |
| 本算子 12 条 manifest workload 的**集成期实测时延**（align+scatter 两 launch，未过滤 capture，非有效 profile，仅数量级参考）：T=1/32/512/4096 → 约 12–45 µs；随 `E` 增大而升（384 > 256 > 128），随 T 增长平缓 | D | `reports/tileops/20261123_164821_903151_MoEPermuteAlignFwdOp/run.json#performance.cases[*].latency_us` | `cd examples/TileOPs && python -m tileops.reporting.cli run --op MoEPermuteAlignFwdOp --prof-mode msprof` | tilelang 0.1.15 + CANN 9.2.1 + Ascend950PR_9579（真机，msprof） | `.agents/skills/tilelang-op-optimize/references/pattern-library/constants.md`（Stage 4 baseline 参考；须标注 profile 无效） |

### Transferable Lessons（可迁移教训）

- **SIMT 线程数有硬上限 2048**：`T.SimtVF(threads=N)` 的 `N` 会变成 `LAUNCH_BOUND(N)` 常量，`N>2048` 是
  **编译期错误**（报错文本为 `Maximum number of SIMT threads returned by LAUNCH_BOUND must be an integer
  constant in range (0, 2048]`，不带源码行号）。任何按 shape 推导的线程宽度（如 `2*e_pad`、`align_up(NB,64)`）
  都必须先夹到 `<= 2048`；超宽时按「等宽多趟 + 不变写序」拆开，而不是加宽线程域。
- 清零/初始化「长度为 2L 的 halo 缓冲」不要写成 `threads=2L`：拆成两趟 `threads=L` 与既有「上半 + 下半」
  写法同构，语义逐位不变，且天然满足宽度上限。
- 一个 op 里有多条独立路径判据时（如 wrapper 的 fused-small 阈值 + kernel 内部的单/两趟阈值），
  **缺陷矩阵要按两条判据的笛卡尔积去覆盖**：本任务 E>1024 的 `2*e_pad` 缺陷同时存在于两条 align 路径，
  只测 wrapper 的 large 路径会把小路径的同一缺陷留到线上。
- 集成脚本对 kernel 是**整文件覆盖**：在集成包里改 kernel 只是让本次 report 通过，不会回流 conductor 产物。
  凡在 Stage 5 改过 kernel，必须在 `integration_log.md` 的遗留风险里写明「重跑脚本会静默回退」，
  并把补丁回写 Stage 3 产物作为显式后续动作。
- 多 kernel 算子（如 align + scatter）在 msprof 未过滤 capture 下必然 `invalid: contains N distinct ops`，
  report 仍会写 `latency_us`（取自该 capture 的 max）。这类数字只能当数量级参考，不能作为性能基线，
  也不能据此判定性能回归。
