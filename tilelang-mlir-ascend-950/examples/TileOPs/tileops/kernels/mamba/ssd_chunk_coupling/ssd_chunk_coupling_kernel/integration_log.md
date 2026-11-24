# Integration Log -- SSDChunkCouplingFwdOp

- meta: `examples/TileOPs/tileops/kernels/mamba/ssd_chunk_coupling/.migration_meta.json`
- op / slug / family: `SSDChunkCouplingFwdOp` / `ssd_chunk_coupling` / `mamba`
- integrated (Stage 3 kernel): `_ssd_chunk_coupling_kernel` ->
  `examples/TileOPs/tileops/kernels/mamba/ssd_chunk_coupling/ssd_chunk_coupling_kernel/_ssd_chunk_coupling_kernel.py`
  (source `examples/ssd_chunk_coupling/_ssd_chunk_coupling_kernel/_ssd_chunk_coupling_kernel.py`)
- design_docs: `_ssd_chunk_coupling_kernel` ->
  `examples/TileOPs/tileops/kernels/mamba/ssd_chunk_coupling/ssd_chunk_coupling_kernel/_ssd_chunk_coupling_kernel_DESIGN.md`
  (byte-copy of `examples/ssd_chunk_coupling/_ssd_chunk_coupling_kernel/DESIGN.md`; no missing design doc)
- aggregate `__init__.py`: generated, re-exports `_ssd_chunk_coupling_kernel`
- wrapper: `examples/TileOPs/tileops/kernels/mamba/ssd_chunk_coupling/ssd_chunk_coupling.py`
  rewritten to the baseline/perf_opt selection block (baseline active,
  `# from .ssd_chunk_coupling_kernel.perf_opt...` commented); `ascend_mode = None`
  (programming mode `cube` / mixed tail; `sync_wrapper_ascend_mode` reported `unchanged`)
- attempts: 1

## 验证结果

- report_run_json: `examples/TileOPs/reports/tileops/20261123_135638_847395_SSDChunkCouplingFwdOp/run.json`
- report_md: `examples/TileOPs/reports/tileops/20261123_135638_847395_SSDChunkCouplingFwdOp/report.md`
- report.html: `examples/TileOPs/reports/tileops/20261123_135638_847395_SSDChunkCouplingFwdOp/report.html`
- correctness: **pass** — 8/8 cases (7 manifest-param cases + `test_ssd_chunk_coupling_fwd_noncontiguous`),
  `avg_max_abs_err = 1.58e-04`, `max_abs_err = 9.77e-04` (bf16 1 ulp, see Stage 3 retrospective),
  tolerance `atol = rtol = 1e-3` (unchanged)
- benchmark: **pass** — 3/3 non-smoke workloads, `prof_mode=msprof`, `profiler_valid=true`,
  `invalid_profile_count=0`, `profiler_fallback_count=0`, `kernel_filter_fallback_count=0`

| workload | dtype | B / NC / Q / G / N | latency (us) | roofline Ratio (%) |
|---|---|---:|---:|---:|
| mamba2-780m | float16 | 1 / 16 / 256 / 1 / 128 | 12.643 | 14.47 |
| mamba2-2p7b | bfloat16 | 4 / 8 / 256 / 1 / 128 | 22.910 | 15.24 |
| mamba2-grouped-prefill | float16 | 3 / 12 / 256 / 8 / 128 | 171.000 | 18.11 |

> 口径说明（避免误读，`run.json#performance.cases`）：`latency_us` 为 msprof 采集的
> kernel 稳态时延（`msprof.captured_op_names = ["main_kernel"]`，`sample_count = 1`）。
> `Ratio` 为框架 roofline 的 `GM Read + Write` 条目利用率；同条目里的
> `Bandwidth(TB/s) = 1.600` 是**模型参考带宽常量**（950PR GM 峰值），不是实测带宽。
> 由 `bytes/latency` 反推的有效 GM 带宽约 0.33 / 0.37 / 0.44 TB/s（派生值，非 profiler 上报）。
> `config = {block_l: 64, block_s: 64, block_n: 64}` 是 wrapper 缓存键；集成 kernel 的
> tile 形状由 `_tile_shape(chunk_len, d_state)` 内部决定（chunk 256 -> BL=128, BK=64），
> 这三个键**不会改变** tile，Stage 4 调优不要把它们当作可调 tile 参数。

## 未过滤（canonical）report 运行与既存阻塞

同一入口、不加 `-k` 的全文件运行（`--op SSDChunkCouplingFwdOp --prof-mode msprof`，无 pytest 过滤）：

- run: `examples/TileOPs/reports/tileops/20261123_135355_670687_SSDChunkCouplingFwdOp/run.json`
- summary: `correctness_tests = 12`, `passed = 8`, `failed = 4`, `benchmark skipped`, `status = failed`
- 本算子 8 例在该次运行中**逐例通过**（`run.json#correctness.cases`，8 条
  `test_ssd_chunk_coupling_fwd*` / `..._noncontiguous` 全为 `passed`）；
- 4 条失败全部来自**同文件内既存的 SSDChunkScanFwdOp**（早前已迁移集成，非本次改动）：

```
tests.ops.test_mamba::test_ssd_chunk_scan_fwd[1-2-64-4-64-32-1-dtype0]  failed
... AttributeError: type object 'PassConfigKey' has no attribute
    'TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION'
（另外 3 条 scan 参数化用例同一错误）
```

**既存性证据（与本次集成无关）**：

1. 隔离复现：`pytest tests/ops/test_mamba.py -k ssd_chunk_scan -x` → 2.53 s 内同一
   `AttributeError`，失败点在
   `tileops/kernels/mamba/ssd_chunk_scan/ssd_chunk_scan_kernel/perf_opt/_ssd_chunk_scan_fwd_kernel.py:213`，
   此时本算子代码尚未被导入/构造。
2. 工具链核对：安装的 `tilelang 0.1.15`（`/workspace/venv-ascend950/lib/python3.12/site-packages`）
   `PassConfigKey` 无该键（也无 `NPUIR_ENABLE_AUTO_MULTI_BUFFER`）；scan kernel 用的是
   `target="npuir"` + npuir 分支 pass key，需要仓库 dev 树（`/workspace/tilelang-mlir-ascend-950/tilelang/transform/pass_config.py:45`
   有该键）的 npuir 构建，而该 dev 树当前无编译好的私有 TVM（`import tilelang` 直接 ImportError）。
3. 工具链与锁一致：`toolchain.lock.json` 中 5 个 tilelang 文件 sha256 全部 OK（含
   `lib/libtilelang.so`、`lib/libtvm_compiler.so`），即环境**没有**从锁漂移；是本算子之外的
   既存 kernel 与锁定 `ascend` 版工具链不兼容。
4. 仅注释/文档级的既有 note：`tileops/kernels/attention/multi_head_attention/.../_gqa_prefill_fwd_kernel.py`
   使用同一缺失键，说明这是仓库内既存的 npuir/ascend 双形态混装问题，不是本次引入。

因此本次引用的是**算子作用域**运行（`-k ssd_chunk_coupling`，只剔除了上述另一算子的 4 例既存失败；
本算子 8 例在该过滤下与未过滤运行中完全相同且全部执行）。过滤条件原样记录在
`run.json#metadata.commands.correctness`（`... "-k", "ssd_chunk_coupling"`），未跳过任何本算子用例，
也未改动任何断言 / 容差 / 测试文件。

## 调试历史

- attempt 1 — 症状：首轮 fail-fast 运行在 scan 用例即中止（`-x`），未触及本算子；
  另据代码阅读发现 scaffold 遗留的 dispatch 与 Stage 3 工厂签名不一致：
  wrapper 调用 `_ssd_chunk_coupling_kernel(...)(block_l, block_s, block_n)(C, B)`，而 Stage 3 工厂
  `@functools.lru_cache` 直接返回 `tilelang.compile(main, target="ascend", out_idx=-1)` 的编译产物
  （`_launch()` 用法为 `_launch(...)(C, B)`），不存在内层 `(block_l, block_s, block_n)` 配置层。
  - 分类：wrapper 胶水错误（`wrapper 胶水错误`；仅改 wrapper 胶水层）
  - 修复：`ssd_chunk_coupling.py#_ssd_chunk_coupling_fwd_wrapped` 去掉内层配置调用，
    改为 `_ssd_chunk_coupling_kernel(batch, num_chunks, n_groups, chunk_len, d_state, dtype)(C_mat, B_mat)`；
    同步删除 scaffold 遗留的重复注释块（`# Part A: GPU TileLang kernel factory ... {gpu_repo_root}`）并
    更新已过期的 "Pre-Stage-3 note" 段落。`block_l/block_s/block_n` 保留为 custom_op 签名参数（缓存键）。
  - 备份：`ssd_chunk_coupling_kernel/history_version/ssd_chunk_coupling_s5_attempt1_pre_glue_fix.py`
    （另有集成前快照 `ssd_chunk_coupling_s5_pre_integrate.py`、`op_pkg_init_s5_pre_integrate.py`）
  - 结果：设备冒烟 fp16 `maxdiff = 0.0`、bf16 `maxdiff = 1.86e-09`；随后两次 report 运行本算子 8/8 通过。
- 未对 scan 用例做任何修改或重试（非本算子、且根因在锁定工具链缺 key，属不可修文件）。

## issues

1. **既存阻塞（非本算子）**：同文件 `SSDChunkScanFwdOp` 4 例在锁定 `ascend` 工具链
   （tilelang 0.1.15 site-packages）上必失败，根因 `PassConfigKey.TL_ENABLE_PLAN_AND_UPDATE_BUFFER_ALLOCATION`
   缺失（npuir 分支 key）。修复路径二选一：① 用仓库 dev 树（npuir 构建，需先
   `tools/prepare_tvm.py` 生成私有 TVM）跑 scan；② 为 scan 提供 `ascend` 版 pass_configs 或加
   `hasattr` 降级；③ 把 mamba 测试/benchmark 按算子拆分文件。在此之前，任何**全文件** report
   都会因这 4 例判 `failed`，与算子质量无关。
2. 集成脚本案头说明与实际不符：`integrate_kernel.py` 只把 `DESIGN.md` 复制为
   `{func}_DESIGN.md` 并生成 wrapper 双 import 块，**不校验** Stage 3 工厂的调用形状是否与 scaffold
   遗留的 `(block_l, block_s, block_n)` dispatch 匹配（本次因此产出一次必然失败的集成态）。建议脚本
   增加一条静态检查：若集成 kernel 顶层工厂返回的编译产物不含内层配置层，则提示 wrapper dispatch 需同步。
3. 集成包内保留了 scaffold 期的 `_ssd_chunk_coupling_kernels.py`（复数，GPU 抽取件）位于**算子包**层
   （`tileops/kernels/mamba/ssd_chunk_coupling/`），集成后已无任何 import 引用（wrapper 已切到
   `ssd_chunk_coupling_kernel` 聚合包）。本次未删除（不在允许改动范围内），建议后续清理以免误用。
4. 性能口径提醒：`default_config` 的 `block_l/block_s/block_n` 对集成 kernel 是**惰性**键
   （tile 由 `_tile_shape` 内部推导），见上文“口径说明”。
