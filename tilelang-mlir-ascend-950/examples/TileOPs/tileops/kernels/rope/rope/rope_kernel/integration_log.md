# Integration Log -- RoPEFwdOp (rope)

- meta: `examples/TileOPs/tileops/kernels/rope/rope/.migration_meta.json`
- platform: ascend / ascend-950pr-9579 (CANN 9.2.1, GCC 12.3.0, dav-3510, 28 Cube / 56 Vector)
- task dir (`TILEOPS_TASK_DIR`): `/workspace/tilelang-mlir-ascend-950/examples/rope`
- interpreter: `/workspace/venv-ascend950/bin/python` (tilelang 0.1.15, torch 2.13.0+cpu, torch_npu 2.13.0rc1)
- integration script: `examples/TileOPs/.agents/skills/add-npu-op/scripts/integrate_kernel.py` (cwd `examples/TileOPs`)
- report entry: `python -m tileops.reporting.cli run --op RoPEFwdOp --prof-mode msprof` (cwd `examples/TileOPs`)
- integrated: `_rope_positions_kernel` -> `tileops/kernels/rope/rope/rope_kernel/_rope_positions_kernel.py`
  (md5 `4f355b11e418b67542f3597adabb8cc9`, byte-identical to the Stage 3 product)
- design_docs: `_rope_positions_kernel` -> `tileops/kernels/rope/rope/rope_kernel/_rope_positions_kernel_DESIGN.md`
  (md5 `599cf31f50bea728754ace836ed02914`, byte-identical to `examples/rope/_rope_positions_kernel/DESIGN.md`)
- wrapper: `tileops/kernels/rope/rope/rope.py` (baseline import active, `ascend_mode = None`)
- attempts: 4 (attempt 1 correctness failed -> glue fix; attempt 2 passed; attempt 3 comment-only cleanup, re-accepted; attempt 4 Stage 5 gate fixes -> report re-run, accepted)

## 验证结果

- report_run_json: `examples/TileOPs/reports/tileops/20261123_101035_110884_RoPEFwdOp/run.json`
  （门禁修正后复跑；`integration_report.json` 位于集成包目录并指向它）
- report_md: `examples/TileOPs/reports/tileops/20261123_101035_110884_RoPEFwdOp/report.md`
  (`report.html` and `benchmark.json` are in the same run directory; `latest.json` points here too)
- correctness: **pass, 34/34** (`tests/ops/test_rope.py`, 0 failed / 0 errors / 0 skipped;
  `summary.correctness_passed=true`; report `Avg Max Abs Error` 5.02e-08)
- benchmark: **pass, 7/7** msprof profiles valid (`invalid_profile_count=0`,
  `profiler_fallback_count=0`, `kernel_filter_fallback_count=0`, `run.json` `warnings: []`)
- run status: `passed` (CLI exit 0), `summary.benchmark_passed=true`

### Benchmark numbers (from the run.json of this run, msprof Task Duration)

| label | dtype | latency (us) | roofline ratio (%) |
|---|---|---:|---:|
| llama3-8b-decode-b64 | bfloat16 | 10.925 | 4.67 |
| llama3-8b-prefill-4k | bfloat16 | 430.361 | 8.32 |
| llama3-8b-prefill-4k | float16 | 429.756 | 8.32 |
| llama3-8b-b2-2k | float32 | 671.119 | 6.25 |
| gptj-6b-prefill | float16 | 330.012 | 8.39 |
| gptj-6b-decode-b32 | float16 | 10.302 | 4.17 |
| phi-2-query-only | bfloat16 | 141.328 | 2.93 |

Reported only, not tuned (Stage 4's scope): latency 10.3-671.1 us, i.e. 2.93 %-8.39 % of the
inlined roofline; decode shapes are launch-floor dominated (~10-11 us for 32-64 tokens).
The fp32 `llama3-8b-b2-2k` ratio jitters between runs (10.82 % -> 6.25 % at an unchanged
~670 us latency): the roofline's byte count divides by the *distinct* rows the random
positions name, which changes per run -- the latency is the stable number, not the ratio.

## 调试历史

- attempt 1 (report `20261123_095623_161890_RoPEFwdOp`, status=failed, 32 passed / 2 failed):
  症状: `test_rope_query_off_a_vector_boundary[neox]` 与 `[gptj]` 失败，报
  `NPU function error: device error type 3, error code is 507035`（traceback 落在
  `rope_tables`/`Op._tables` 的 host->device 拷贝上）。
  分类: 运行失败（设备错误），按 `diagnostics.debug_skill`（tilelang-ascend ->
  tilelang-error-fixer / tilelang-debug-helper）。
  定位（最小复现，每例独立进程，`/tmp/opencode/rope_s5_probe/`）:
  - 对齐 query + kernel：OK；未对齐 query + 纯 torch op：OK（排除后端不支持该视图）
  - 未对齐 query + kernel：507035；未对齐 -> `clone()` 再跑 kernel：OK
  - 元素偏移网格（fp16，每元素 2 B）: off=1(2 B) FAIL, 2(4 B) FAIL, 3(6 B) FAIL,
    4(8 B) OK, 8(16 B) OK, 10(20 B) FAIL, 12(24 B) OK, 16/32/128 OK
    => kernel 的逐 lane 向量化 GM 访问要求基址 8 B 对齐；测试用例的
    `torch.empty(n+1)[1:].view(shape)` 是 storage_offset=1 的连续视图，
    `contiguous()` 不会消除该偏移。
  修复（wrapper 胶水层，`_RoPEPositionsKernel._aligned`）: `data_ptr() % 16 != 0` 的
  positions/query/key/offsets 先 `clone()` 一次再交给 kernel（值/形状/dtype 不变，
  对齐输入零开销）。kernel 文件与 DESIGN.md 未改动（保持与 Stage 3 产物一致）。
  结果: 该两个用例单独复跑通过。
- attempt 1' 集成脚本自身失败（09:55:53）: wrapper 仍从 kernel 模块 import
  `_head_units`，而 Stage 3 产物把该算术折进 factory -> 首次集成 smoke
  `ImportError: cannot import name '_head_units'`，生成的 `__init__.py` 还把该名字写进
  `__all__`。修复: `_head_units`（纯算术，只喂已失效的占用率提示）在 wrapper 内本地定义，
  基线 import 收敛为 `_rope_positions_kernel` 单名；重跑集成脚本 -> `__init__.py` 重新生成、
  smoke `smoke-ok ['_rope_positions_kernel']`。（该问题属集成期接口不匹配，不是 kernel 精度/编译问题。）
- attempt 2 (report `20261123_100113_389981_RoPEFwdOp`, status=passed, 34/34, benchmark 7/7).
- attempt 3: 选源块里由集成脚本按*修复前* import 名单生成的过时注释行
  `# from .rope_kernel.perf_opt._head_units import _head_units` 已删除（会误导 Stage 4
  去找不存在的 perf_opt 模块），仅注释改动，重跑完整 report 复验 ->
  `20261123_100523_011640_RoPEFwdOp`, status=passed, 34/34, benchmark 7/7, 数值与 attempt 2 同量级（±0.5%）。
- attempt 4 (Stage 5 机械门禁 `statectl gate 5`): `S5-PKG`（report json 需在集成包内）与
  `S5-ASCEND-MODE`（`ascend_mode` 必须在直接继承 `Kernel` 的 class 上）两项失败 -> 修正后
  另触发 `S5-REPORT-STALE` -> 重跑完整 report -> `20261123_101035_110884_RoPEFwdOp`，
  status=passed, 34/34, benchmark 7/7；随后 `statectl gate 5` 通过（见下节）。

## 门禁修正（Stage 5 gate）

- `S5-PKG`：`integration_report.json` 移入集成包目录 `rope_kernel/`（内容不变，仍指向本次 run_json）；
  父目录 `rope/` 保留同名副本便于就近查阅。
- `S5-ASCEND-MODE`：`ascend_mode = None` 原被集成脚本写在 `RoPENeoxPositionIdsKernel` 上，而门禁只在
  **直接继承 `Kernel`** 的 class body 内查找该赋值（本 wrapper 直接继承 `Kernel` 的只有
  `_RoPEPositionsKernel`）⇒ `class_modes` 为空。改为在 `_RoPEPositionsKernel` 声明一次，四个变体继承；
  属性解析结果仍是 `None`（实测四个类的 `ascend_mode` 均为 `None`），行为不变。
- 两处均只动 wrapper 胶水与集成包元数据，**未触及 kernel 文件**（`_rope_positions_kernel.py`
  md5 仍为 `4f355b11e418b67542f3597adabb8cc9`）。
- 修完后门禁又报 `S5-REPORT-STALE`（`run.json` mtime 早于改动后的 wrapper —— 该规则比较
  run.json 与 wrapper/`__init__.py`/kernel 的 mtime）。按门禁要求**重跑了完整 report**
  （未改时间戳取巧）：`examples/TileOPs/reports/tileops/20261123_101035_110884_RoPEFwdOp/`，
  `status=passed`、正确性 34/34、benchmark 7/7（`invalid_profile_count=0`），
  latency 与上一轮同量级（decode 10.3/10.9 us，prefill 4k 429.8/430.4 us，fp32 b2-2k 671.1 us）；
  `integration_report.json` 已指向该 run.json。ascend_mode 从子类移到基类后行为不变
  （四个类的 `ascend_mode` 实测均为 `None`，34 个用例全过即为证据）。

## 遗留项（不阻塞集成，交后续阶段/进化机制）

1. `tileops/kernels/rope/rope/_ro_p_e_fwd_kernels.py`（Stage 0 提取的 GPU 版本）已无任何引用
   （全仓 grep 为 0）；它仍在包目录内且是 CUDA target 实现。不在本次允许编辑范围，未删除，
   建议 Stage 4/清理轮次移除，避免误 import。
2. `rope.py` 模块 docstring 的 "Part A ... extracted + imported" 与 "Pre-Stage-3 note"
   已过时（现 import 的是 Stage 3 NPU 产物）；按调度要求未改文档性内容。
3. 集成脚本 `integrate_kernel.py` 两个缺陷（已写入 op 级 RETROSPECTIVE 的 R 类提案）:
   `gen_init_py` 把未定义名字放进 `__all__`；`rewrite_wrapper_import` 的 perf_opt 注释块
   取自改造前的 import 名单。
4. benchmark 数值仅报告：SIMT 主选路径在 950PR 上为 roofline 的 2.82 %-10.82 %，
   decode 受 launch 地板限制（~10 us）。是否值得 Stage 4 重构（如 `units_per_block`
   上调、gptj 分支）留给调优阶段判定。
5. 本次只跑 `RoPEFwdOp`：`test_rope.py`/`bench_rope.py` 是 rope 家族裁剪后的共享文件，
   manifest `source.test/source.bench` 指向同一对文件；RoPELlama31 / YaRN / LongRoPE /
   RoPECosSin 未迁移，未做全族回归（按调度要求）。
