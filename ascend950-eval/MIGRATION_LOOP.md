# Ascend 950 算子迁移与调优闭环（流程说明）

更新：2026-11-23。适用仓库：`xiangmu/TileOPs-ascend950`（分支 `ascend950`，HEAD `2362d1e9`），GPU 参考：`xiangmu/TileOPs`（`main`，HEAD `2485b53`）。覆盖范围：本批 11 个算子（清单见 `MIGRATION_PLAN_batch2.md`）。

## 0. 一句话

**不需要另建 Agent 框架。** 仓库本身就是一条 spec 驱动的生产环（`docs/design/architecture.md` 的 "agent production loop"），Ascend 侧的接入点只有一处 —— 在同一个 `*FwdInterface` 后面再挂一个 Ascend kernel。要做的是**按算子把这一圈跑够遍数**，并且保证每一圈的反馈信号是真的。

## 1. 闭环总图

```
manifest spec (权威)
      │  签名 / workloads / roofline 都从这里读，不许改
      ▼
  Op 层 (*Interface)  ──公开 API、调用契约、数值策略──┐
      │                                              │
      ├─ CUDA kernel（原有，不动）                    │
      └─ Ascend kernel（新增，register_kernel_type）──┘
                     │
                     ▼
        ① 正确性反馈：pytest --tileops-device=npu
                     │ 通过
                     ▼
        ② 性能反馈：benchmark_ascend + roofline → MFU 差距
                     │ 不达标
                     ▼
        ③ 调优：改 Ascend kernel → 回到 ①（并跑受波及算子的回归）
```

三个反馈信号缺一个，这一圈就是无效的：**过/不过**、**距 roofline 多远**、**波及了谁**。

## 2. 每一轮的四进四出

### 2.1 每轮必须喂进去的四样

| # | 输入 | 位置 | 缺了会怎样 |
|---|---|---|---|
| 1 | spec 条目（签名 + workloads + roofline） | `src/tileops/manifest/spec/<family>.yaml` | 不知道该支持哪些 shape/dtype，用例编不出来 |
| 2 | GPU 参考实现 | `src/tileops/kernels/<family>/<op>.py`（分支内已有） | 只能凭 op 文档猜语义，几乎必错 |
| 3 | 测试入口 | `tests/ops/test_<op>.py`，必要时 `test_<op>_ascend.py` | 没有判据，"跑通了"只能是主观判断 |
| 4 | roofline 公式 | spec 的 `roofline` 字段 | 无法判断性能差距是"实现档次问题"还是"没调参" |

> MLADenseBwdOp / MLAVarlenBwdOp 这四样**当前都不全**：spec 条目、Op 类、kernel、workload、测试需要先从 main 移植，见 §3.3。

### 2.2 每轮必须产出的四样

| # | 产物 | 说明 |
|---|---|---|
| 1 | `src/tileops/kernels/ascend/<family>/<op>.py` | 继承对应 `*FwdInterface`，`devices = frozenset({"npu"})`，实现 `refusal()`（或明确不限制），文件底部 `register_kernel_type("<OpName>", "ascend", <Kernel>)` |
| 2 | family 包登记 | `src/tileops/kernels/ascend/<family>/__init__.py` 里 import 该类并加入 `__all__` —— **这是注册生效的唯一通道**，漏了 `register_kernel_type` 永远不会执行 |
| 3 | 测试结果 | manifest 全部 case 的 passed/failed/skipped 计数，skipped 必须解释原因 |
| 4 | 性能记录 | case / dtype / 时延 / MFU / roofline 差额 / 口径标注（开发口径 or canonical） |

### 2.3 每轮不许碰的

- **manifest spec**：它是权威。唯一例外是 MLA 反向的**移植**（新增条目），不是修改已有条目。
- **原有 CUDA 实现**：`devices = frozenset({"npu"})` 保证 CUDA 设备不会选中 Ascend kernel，两边互不干扰。
- **其它 family 的 kernel**：除 §5 回归范围里点名的"波及文件"。

## 3. 三条起点各自的检查清单

### 3.1 路线 A：已有实现（6 个）→ 验收 + 调优

这 6 个的 Ascend 实现都在 HEAD `2362d1e9` 里，**来源是同一个 merge**。所以第 0 步不是写代码，是搞清楚"这活是不是已经有人做完了"。

| 步 | 动作 | 判据 |
|---:|---|---|
| 0 | 查这 6 个是否已被验收（PR / receipt / 台账），确认归属 | 明确是"调优"还是"接手补验收" |
| 1 | 环境：激活 venv，`pip install -e . --no-deps` | backend entry point 生效，算子能被 `npu` 后端选中 |
| 2 | 正确性：`pytest tests/ops/test_<op>.py --tileops-device=npu -q` | manifest 全部 case passed |
| 3 | 读 `refusal()`，确认覆盖边界；没有 `refusal` 的算子自己核对对齐/尾块 | 边界写进文档，不靠猜 |
| 4 | 性能：`scripts/benchmark_ascend.py --op <OpName>` | 得到时延 + MFU + 与 roofline 的差额 |
| 5 | 判断差距性质：实现档次（缺 Cube / 并行度不足）还是调参 | 出一句结论，再决定是否值得调 |
| 6 | 调优后回到第 2 步，并跑 §5 的全部波及算子 | 无回归 |

测试入口对照：

| 算子 | 测试命令 |
|---|---|
| EngramGateConvFwdOp / EngramGateConvBwdOp | `pytest tests/ops/test_engram.py tests/ops/test_engram_ascend.py --tileops-device=npu` |
| EngramDecodeFwdOp | 同上（`test_engram.py` 覆盖） |
| RoPEFwdOp | `pytest tests/ops/test_rope.py --tileops-device=npu` |
| SSDChunkCouplingFwdOp | `pytest tests/ops/test_mamba.py --tileops-device=npu` |
| ChainSpeculativeSamplingFwdOp | `pytest tests/ops/test_sampling.py --tileops-device=npu` |

### 3.2 路线 B：3 个有接口无 kernel → 只折一环

| 步 | 动作 |
|---:|---|
| 1 | 读接口：`kernels/moe/call_spec.py` 的 `PermuteAlignCall:59` / `PermuteAlignFwdInterface:211`、`SharedExpertMLPCall:68` / `SharedExpertMLPFwdInterface:236`；`kernels/attention/call_spec.py` 的 `PagedKVCacheWriteCall:848` / `PagedKVCacheWriteFwdInterface:879` |
| 2 | 读 GPU 参考：`kernels/moe/permute_align.py`、`kernels/moe/shared_expert_mlp.py`、`kernels/attention/paged_cache_write.py` |
| 3 | 新建 `kernels/ascend/<family>/<op>.py`，继承上表对应 Interface，设 `devices = frozenset({"npu"})`，实现 `forward` 与 `refusal` |
| 4 | 文件底部 `register_kernel_type("<OpName>", "ascend", <Kernel>)` |
| 5 | `kernels/ascend/<family>/__init__.py` import 该类并加入 `__all__`（**漏这步注册不生效**） |
| 6 | `pip install -e . --no-deps`，再 `pytest tests/ops/test_<op>.py --tileops-device=npu` |
| 7 | 性能与口径按 §3.1 的第 4 步起 |

三个算子的差异决定了"这一轮的难点在哪"：

| 算子 | 引擎 | 难点 | 判成败 |
|---|---|---|---|
| MoEPermuteAlignFwdOp | Vector / 标量 | 整型 histogram + 跨 expert 前缀和 + scatter；`num_experts` 128→384 动态；`block_size` padding 与无效 id 桶 | 正确性 + 时延（`flops=0`，无 MFU） |
| SharedExpertMLPFwdOp | Cube | 两段 GEMM；`[T,2S]` 中间量（T=4096,S=18432 时约 288 MiB）不能整块进 shared；`S=18432` 的 N 尾块 | MFU，且仅 T≥512 的 case 有意义 |
| PagedKVCacheWriteFwdOp | Vector / MTE | **无输出**，只能查 mutated 输入；`slot_mapping` 是 int64；`-1` 跳过；FP8 饱和 cast 且是除以 scale；`PS=7/NP=5` 奇数页 | 正确性（mutated 比对）+ 带宽 |

### 3.3 路线 C：MLADenseBwdOp / MLAVarlenBwdOp → 先移植，再折一环

第 0 步（移植）比写 kernel 更容易出错，清单如下（main → 分支，布局与命名都要改）：

| 内容 | main 位置 | 分支目标 |
|---|---|---|
| manifest spec 条目 | `attention.yaml:739`（MLADenseBwdOp）、`:776`（MLAVarlenBwdOp） | `attention.yaml` 新增两条 |
| Op 类 | `_ops/attention/mla.py:292`、`:385` | `ops/attention/mla.py` |
| 接口 + call_spec | `_kernels/attention/mla/bwd.py`、`bwd_ws.py`、`_kernels/attention/call_spec.py` | `kernels/attention/mla/`、`kernels/attention/call_spec.py` |
| workload | `workloads/attention/mla.py:536`、`:607` | `workloads/attention/mla.py` |
| 对外导出 | `attention.py`、`_ops/attention/__init__.py` | `attention.py`、`ops/attention/__init__.py`，以及 `ops/__init__.py` 的 name→module 映射与 lazy import 两处 |
| 测试 | `tests/ops/test_mla_bwd.py` | `tests/ops/test_mla_bwd.py` |

移植时**必须同步改**的三处：

1. **roofline 函数路径**：main 写 `tileops._perf.formulas.mla_dense_bwd_roofline` / `mla_varlen_bwd_roofline`，分支是 `tileops.perf.formulas.*`（对照 `attention.yaml:714`）。不改 → manifest 校验直接失败。
2. **命名分叉**：分支无 `_ops`/`_kernels` 前缀；MLA decode 在分支叫 `MLADecodeWithKVCacheFwdOp`（main 是 `MLADenseDecodeFwdOp`）。`call_spec` 的 Call/Interface 名要一起核对。
3. **参考实现**：`ref_api` 指向 `flash_attn_interface._flash_attn_backward`，NPU 上没有 flash-attn。移植后要先把"NPU 上用什么当参考"定下来，否则测试跑不起来（注意 NPU **无 float64**，不能随手拿 double 兜底）。

移植完成后，就退化成路线 B：写 ascend kernel → 登记 → 测试 → 性能。

语义上最容易写错的三点（写 kernel 前先记住）：`D_qk=192`（128 nope + 64 rope，非 2 的幂）、bottom-right 对齐 + causal（`S_q ≠ S_kv` 时对角块不等长）、`lse` 是**自然对数**且 Varlen 版是 head-major `[H,T_q]`。

## 4. 每一轮的验收判据

**功能轮**

- manifest 里该算子的**全部** case 都跑到，不许只跑 smoke 就宣布通过。
- `passed == 总数`；`skipped` 必须写明原因（典型原因：架构标记不符、参考实现不可用）。
- `refusal()` 的支持边界写进记录，不允许"没报错就算支持"。
- **不许为了让测试变绿而改断言、加 skip、放宽 tolerance 或改 manifest** —— 那是掩盖问题，不是修问题。

**性能轮**

- 同 case、同协议、同设备；记录 latency / MFU / 与 roofline 的差额。
- 必须标注口径：本地开发口径（无 `npu-guard`、未走 `benchmark_ascend.py` 正式协议）还是 canonical。
- 开发口径的数字**只能看趋势**，不能当"达标"结论，也不能写进验收报告。

**回归轮**

- §5 里点名的波及算子全部重跑，不看别人上次的记录。

## 5. 回归范围表（改这里会波及谁）

| 改动文件 | 波及算子 |
|---|---|
| `kernels/ascend/mamba/chunks.py` | SSDChunkCouplingFwdOp、SSDChunkStateFwdOp、SSDChunkScanFwdOp、SSDChunkCumsumFwdOp、SSDStatePassingFwdOp、SSDRecurrentFwdOp（及 Mamba2 组合 op） |
| `kernels/ascend/sampling/draws.py`（`_DrawKernel`） | ChainSpeculativeSamplingFwdOp、SamplingFromProbsFwdOp |
| `kernels/ascend/rope/rotation.py` | RoPEFwdOp、RoPELlama31FwdOp、YaRNFwdOp、LongRoPEFwdOp、RoPENeoxPositionIdsFwdOp |
| `kernels/ascend/sequence_modeling/*` | EngramGateConvFwdOp / EngramGateConvBwdOp、EngramDecodeFwdOp（若抽公共 helper 还会牵到 MHCPreFwdOp） |
| `kernels/moe/call_spec.py` | MoE family 的 Call/Interface（PermuteAlign、SharedExpertMLP 等） |
| `kernels/attention/call_spec.py` | attention family 全部算子的 Call/Interface |
| `manifest/spec/<family>.yaml` | 该 family 全族，**含 CUDA 侧** |
| `kernels/ascend/<family>/__init__.py` | 该 family 全部 Ascend 算子（注册靠它） |

## 6. 改哪个目录先读哪份域规则

照 `CLAUDE.md` 的表，读之前不要动手：

| 要改 | 先读 |
|---|---|
| `tests/` | `.claude/domain-rules/testing-budget.md` |
| `src/tileops/manifest/spec/*.yaml` | `.claude/domain-rules/manifest-spec.md` |
| `src/tileops/manifest/*.py`、`scripts/validate_manifest.py` | `.claude/domain-rules/manifest-validator.md` |
| `src/tileops/ops/`、`src/tileops/kernels/` | `.claude/domain-rules/ops-design.md` |
| `benchmarks/` | `.claude/domain-rules/benchmark.md` |
| `workloads/` | `docs/design/layer-boundaries.md` |
| `docs/design/` | `.claude/domain-rules/design-docs.md` |

两条额外约定：manifest 的程序化读取用 `from tileops.manifest import load_manifest, load_workloads`，不要自己重写 merge；编辑 yaml 用 `ruamel.yaml` 保证注释和键序不丢。

## 7. 环境与工具链（本机要点）

| 项 | 做法 |
|---|---|
| Python | `xiangmu/.venvs/ascend950`（py3.12）；`pip install -e . --no-deps` 让 backend entry point 生效 |
| 选卡 | `ASCEND_RT_VISIBLE_DEVICES=<n>`；本机 0/1/3 曾 Alarm、5 号性能异常、7 号不存在 |
| 编译 shim | `TILEOPS_SYNC_ATTR_SHIM=1`，`PYTHONPATH` 加 `xiangmu/.shims-py`，`pytest -p tileops_shim`；重试与超时用 `TILEOPS_COMPILE_RETRIES` / `TILEOPS_COMPILE_TIMEOUT`。它会修 `syn_instr_mode` 属性名并对 SIGSEGV / 超时重试 |
| a5 编译器 | 把 `xiangmu/.shims-a5` 前置到 `PATH`，`bishengir-compile/opt` 才会指向 `…/bin/bishengir-*-a5` |
| 编译缓存 | `TILELANG_CACHE_DIR` 每个实验用独立目录，避免不同 kernel 版本互相污染 |
| 对齐 | shared buffer 按 32B 对齐，否则 device error 507035 |

## 8. 口径红线（写进每份记录）

1. **功能通过 ≠ 性能达标**，这两件事分开判。
2. 本地跑出来的时延属"开发口径"，与 `ascend950-profiler-locked-v1` 的正式结果**不可互换**。
3. 当前设备是 28 核 950PR 样片（matmul ≈ 366.8 TFLOP/s），MFU 达标判定需要 canonical 32 核 950DT。
4. 上一批的历史数字不能直接当本批基线（协议、TileLang 版本、时钟策略都可能不同）。
5. NPU 无 float64：任何依赖 double 的参考实现都不可用，不能用它当"正确性基线"。

## 9. 卡住时的停止条件

同一个现象**连续 3 轮没有进展**就停下，不要继续盲试。停下来要给出：查了什么、拿到什么证据、最可能的两个假设、建议的下一步。禁止的"推进"手段包括：改测试断言、加 skip、放宽容差、改 manifest 去迁就实现、把失败用例排除在统计外。

## 10. 样板轮：建议先用 PagedKVCacheWriteFwdOp 跑通整圈

选它的理由：纯 scatter、**无输出**、6 个 case、不涉及 Cube，是 11 个里最容易一次跑通的，能把"写文件 → 登记 → 测试 → 性能记录"这条链上的流程问题一次性暴露出来。

一轮需要动的文件：

| 动作 | 文件 |
|---|---|
| 新建 | `src/tileops/kernels/ascend/attention/paged_cache_write.py`（继承 `PagedKVCacheWriteFwdInterface`，底部 `register_kernel_type`） |
| 修改 | `src/tileops/kernels/ascend/attention/__init__.py`（import + `__all__`） |

命令序列：

```bash
cd TileOPs-ascend950
source ../.venvs/ascend950/bin/activate
pip install -e . --no-deps
export ASCEND_RT_VISIBLE_DEVICES=<卡号>
export TILELANG_CACHE_DIR=<独立缓存目录>
pytest tests/ops/test_paged_kv_cache_write.py --tileops-device=npu -q
python scripts/benchmark_ascend.py --op PagedKVCacheWriteFwdOp --verify-only --run-dir <新建目录>
```

第二行之后正式性能再加 `--clock-config`（有的话）与 `--guard`，否则结论只能标"开发口径"。

## 11. 待定事项

1. **卡分配**：本批固定用哪张卡（避免和其它子任务撞卡）。
2. **6 个已有实现的归属**：确认是否已被验收，决定路线 A 是调优还是补验收。
3. **MLA 反向的参考实现**：NPU 无 flash-attn，正确性基线用什么。
4. **样板算子确认**：是否就从 PagedKVCacheWriteFwdOp 起（或改 MoEPermuteAlignFwdOp）。
5. **是否引入 autotune**：现有 Ascend kernel 里有 `*_tuned` 变体（如 `AscendFusedTopKTunedKernel`），要不要在本批统一走 autotune 路线，需要先定策略。

---
本文件只描述流程，不含代码改动。算子清单与现状见同目录 `MIGRATION_PLAN_batch2.md`，可行性判断见 `FEASIBILITY.md`。
