# Integration Log -- MoEPermuteAlignFwdOp

- meta: `examples/TileOPs/tileops/kernels/moe/moe_permute_align/.migration_meta.json`
- op: `MoEPermuteAlignFwdOp` (op_slug `moe_permute_align`, family `moe`)
- task dir (`TILEOPS_TASK_DIR`): `/workspace/tilelang-mlir-ascend-950/examples/moe_permute_align`
- integration package: `examples/TileOPs/tileops/kernels/moe/moe_permute_align/moe_permute_align_kernel/`

## 集成结果

- integrated（`integrate_kernel.py`，一次运行，幂等，import 冒烟 pass）：
  - `_make_align_kernel` -> `_make_align_kernel.py`
  - `_make_scatter_kernel` -> `_make_scatter_kernel.py`
  - `_make_small_batch_kernel` -> `_make_small_batch_kernel.py`
- design_docs（Stage 1 交付件快照，未修改）：
  - `_make_align_kernel_DESIGN.md`
  - `_make_scatter_kernel_DESIGN.md`
  - `_make_small_batch_kernel_DESIGN.md`
  - 无缺失（无 `[warn] no DESIGN.md`）
- wrapper：`examples/TileOPs/tileops/kernels/moe/moe_permute_align/moe_permute_align.py`
  - baseline/perf_opt 双 import 切换块（baseline 生效，perf_opt 注释态）
  - `ascend_mode = None`（三 kernel 模式 align=`simd` / scatter=`simt` / small_batch=`simt`，
    `adapter.integration_mode` 对 ascend 统一返回 `None`；单 Kernel class 逐个模块自带模式）
  - 工厂层数核对：三段 Stage 3 产物均为一级调用
    `factory(numel, num_experts, block_size) -> builder()`，wrapper 的
    `_compiled_align()/_compiled_small()` 调用形态一致（无 GPU 时代两级调用），无需胶水改写
  - 调用形态核对：align `(flat, sorted, expert_ids, num_tokens_post_pad, cumsum)`、
    scatter `(flat, sorted, cumsum)`、small_batch `(flat, sorted, expert_ids, num_tokens_post_pad)` 全部一致
  - 选路核对：`numel < 1024 且 E <= 32` -> small_batch 融合；否则 align(+scatter)；与 `MAX_EXPERTS=32` 一致
- 集成脚本已知缺陷（已按规则清理，先备份）：
  - `rewrite_wrapper_import` 在 scaffold 已有切换块的 wrapper 上重复写入整块（header + baseline comment +
    perf_opt 注释各出现两次）-> 清理为单一 header + 单一 baseline import + 单一 perf_opt 注释块；
    同时更新 wrapper docstring 中过期的 "Pre-Stage-3 note ... forward fails（expected scaffold state）" 与
    `__init__` 内 "imported functions are the GPU implementations" 注释
  - `__all__` 未被写入不存在的名字（本次无该症状）
  - 清理后 import 冒烟复跑 pass：`exports ['_make_align_kernel', '_make_scatter_kernel', '_make_small_batch_kernel']`、
    `K.ascend_mode is None`、`tileops.kernels.moe.moe_permute_align.MoEPermuteAlignKernel is` 同一类
- 备份（`moe_permute_align_kernel/history_version/`）：
  - `moe_permute_align_wrapper_preintegrate.py`（集成前 scaffold wrapper）
  - `moe_permute_align_wrapper_postintegrate.py`（脚本改写后、人工清理前）
  - `_make_align_kernel_s5_attempt1.py`（compiler-fix 前的集成 kernel）

## 验证结果

- report_run_json: `examples/TileOPs/reports/tileops/20261123_164821_903151_MoEPermuteAlignFwdOp/run.json`
- report_md: `examples/TileOPs/reports/tileops/20261123_164821_903151_MoEPermuteAlignFwdOp/report.md`
- report_status: `partial`（correctness 通过，benchmark 采集到数值但 12/12 条 `profiler_valid=false`）
- correctness: **pass 30/30**（`tests/ops/test_moe_permute_align.py` 全量，含 small_batch 融合路径用例
  `sb-numel200`(numel=200,E=8)、`sb-numel600-maxexp`(numel=600,E=32)、`routes-1023`(numel=1023,E=8)）
- benchmark: **partial / 无效 profile（只记录不修复）**
  - `summary`: `benchmark_requested=true, benchmark_passed=false, case_count=12, msprof_case_count=12,
    invalid_profile_count=12, profiler_fallback_count=0, kernel_filter_fallback_count=0`
  - 每条用例 `invalid_reason = "unfiltered capture contains 2 distinct ops"`
    （`captured_op_names = ["_scatter_main_kernel", "main_kernel"]`，即 align + scatter 两次 launch 在同一 capture 内）；
    `kernel_name_attempts = [null]` ⇒ 框架无法自动消歧。`latency_us` 取自该未过滤 capture 的
    `max_latency_us`，**仅作参考，不是单 kernel / 单 op 有效时延**
  - 有效数值（latency_us，非有效 profile；本算子 roofline `flops=0`，ratio 无物理意义，只看 latency）：

    | label | T | E | latency_us |
    |---|---|---:|---:|
    | kimi-k2-t1 | 1 | 384 | 23.35 |
    | kimi-k2-t32 | 32 | 384 | 34.56 |
    | kimi-k2-t512 | 512 | 384 | 26.22 |
    | kimi-k2-t4096 | 4096 | 384 | 45.32 |
    | ds-v3-t1 | 1 | 256 | 17.57 |
    | ds-v3-t32 | 32 | 256 | 28.17 |
    | ds-v3-t512 | 512 | 256 | 24.14 |
    | ds-v3-t4096 | 4096 | 256 | 39.32 |
    | qwen3-235b-t1 | 1 | 128 | 12.02 |
    | qwen3-235b-t32 | 32 | 128 | 22.66 |
    | qwen3-235b-t512 | 512 | 128 | 22.18 |
    | qwen3-235b-t4096 | 4096 | 128 | 29.84 |

  - 形态观察（非结论）：latency 随 `E` 减小而下降（align 的 `e_pad` 与 `expert_ids` 扫描量随 E 增长），
    随 T 增长平缓；全部落 12–45 µs 区间，接近两次 launch 的派发量级，无明显 shape 爆炸
  - report CLI 因 `partial` 返回 1，不作为集成失败判据

## 调试历史

### attempt 1 -- `experts-past-threads` 编译失败（已修复）

- 症状：correctness 29/30，`test_permute_align_op[experts-past-threads]`（`numel=512, E=1100, B=64`）在
  编译期失败：
  `error: Maximum number of SIMT threads returned by LAUNCH_BOUND must be an integer constant in range (0, 2048]`
- 分类：编译失败（SIMT 线程宽度上限），非精度 / 非测试自身问题
- 定位（IR/几何最小复现 `/tmp/opencode/s5/repro_launch_bound.py`）：
  `_geometry(512,1100,64)` -> `E_pad = align_up(1100,64) = 1152`；`_compile_align_partial` 里
  `with T.SimtVF(threads=2 * e_pad)` ⇒ `threads = 2304 > 2048`（`_compile_hist` 的 `threads=e_pad=1152` 通过）。
  同一形态也在 `_compile_single`（align 内部小路径）第 221 行存在；该路径在 `numel<1024, E>1024, B` 小时可达
  （例如 `numel=100, E=1100, B=1`：`_use_small_path=True`），旧测试集未覆盖
- 修复（集成 kernel `_make_align_kernel.py`，两处同一改法）：把 `2*e_pad` 单趟清零拆成两个 `e_pad` 趟
  （`s_scan_b[e]=0` 与 `s_scan_b[e_pad+e]=0`），写集合与写序不变（与相邻 `s_scan_a` 上/下半两趟清零的既有写法同构）
- 验证：
  - 复现脚本由 `[fail] align_partial: ... LAUNCH_BOUND ...` 变为 `[ok] align_partial compiled`
  - 额外补测 e_pad>1024 的 align 内部小路径（`/tmp/opencode/s5/check_small_path_epads.py`）：
    `(numel=100,E=1100,B=1) align_path=S`、`(numel=200,E=1100,B=64) path=L`、`(numel=4096,E=1100,B=64) path=L` 全部 pass
  - 完整 report 重跑：correctness 30/30 pass（含 `experts-past-threads`）
- 备份：`history_version/_make_align_kernel_s5_attempt1.py`

## 遗留风险

1. **kernel 修复只在集成包内**，`examples/moe_permute_align/_make_align_kernel/_make_align_kernel.py`
   （conductor 产物）仍是 `threads=2*e_pad` 版本。重跑 `integrate_kernel.py` 会整文件覆盖，**静默回退该修复**；
   建议 conductor 把 2 处 `2*e_pad` 清零拆分回写进 Stage 3 产物（或由 Stage 3 修订），否则下次集成会重现失败。
2. **`e_pad > 2048` 仍不可用**：`threads=e_pad` / `threads=e_pad+64` / `threads=blk` 等仍直接用 e_pad 或
   `align_up(NB,64)` 派生宽度。本次测试集上限 E=1100（e_pad=1152）、manifest 上限 E=384，均在上限内；
   E>2048 或 `NB` 极大（`blk>2048`）时会再次触发同一 LAUNCH_BOUND 限制（未实测，标注为未验证推断）。
3. **benchmark 无有效 profile**：align + scatter 两次 launch 落在同一 msprof capture（2 distinct ops），
   12/12 用例 `profiler_valid=false`，改前改后都无法获得单 kernel 时延；按要求只记录不修复。
   若后续需要有效 latency，需在 benchmark/报告侧给出多 kernel 的归因口径（属 tests/benchmarks 范围，本 stage 不改）。
4. **small_batch 融合路径未被 benchmark 覆盖**：manifest 12 条 workload 全部 `E>32`，走 align(+scatter)；
   small_batch 只由 correctness 用例覆盖（`sb-numel200`、`sb-numel600-maxexp`、`routes-1023`、`skewed[few-experts]`，
   均 pass）。其真实时延只有 Stage 3 的 host 侧计时（24.5/14.7/7.1/4.1 µs 量级），无 msprof 证据。
5. 集成脚本缺陷仍在：wrapper 已有切换块时会重复写入整块 header/perf_opt 注释，需要人工清理；
   建议 `rewrite_wrapper_import` 先检测既有块。
