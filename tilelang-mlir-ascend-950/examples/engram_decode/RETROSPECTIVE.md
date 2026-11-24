# RETROSPECTIVE -- engram_decode (EngramDecodeFwdOp)

op 级集成复盘（migration-harness 模式）。逐函数 Stage 1/2/3/4 的复盘分别位于
`_engram_project_kernel/RETROSPECTIVE.md` 与 `_engram_step_kernel/RETROSPECTIVE.md`。

## Stage 5 (integrate) Retrospective — attempt 1 (0 repair attempts) — 2026-11-23

集成对象：`EngramDecodeFwdOp`（family `sequence_modeling`），2 个函数
`_engram_project_kernel` + `_engram_step_kernel`。结果 `INTEGRATE_COMPLETED`：
正确性 46/46（本算子 16/16）。门禁引用的是 `--prof-mode msprof` 报告（`status=partial`，
本算子 4/4 workload 因双 kernel 无有效 profile，仅记录）；补跑的 `events` 报告
（`status=passed`，10/10 case 有效）只作有效 latency 的补充证据。0 次修复 attempt。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Integrate-flow | `integrate_kernel.py` 只改写 wrapper 的 import 块与 `ascend_mode`，不核对 scaffold 的 `default_config`/dispatch 参数是否仍为 GPU 派生值；本算子 scaffold 保留 `block_k=_projection_block_k(d_mem)` / `num_stages=4`，而 Stage 3 冻结 tiling 是 `block_k=256` / `num_stages=2`，只能靠人读 Stage 3 文件发现 | `examples/TileOPs/.agents/skills/add-npu-op/scripts/integrate_kernel.py#rewrite_wrapper_import`、`#sync_wrapper_ascend_mode`；`tileops/kernels/sequence_modeling/engram_decode/engram_decode_kernel/history_version/engram_decode_s5_pre_integrate.py`（改前 `default_config`） | 在 `integrate_kernel.py` 末尾增加 config 漂移检查：若 wrapper `default_config` 的键值与该函数 Stage 3 模块的 `default_config(...)` 不一致，输出 `[warn] config drift: ...` 并给出建议值 | R |
| Integrate-flow | 集成与报告被拆成两轮调度（设备占用），`integration_report.json` 需先写 `pending_report` 再人工回写真实 `run_json`；流程规范未定义 pending 字段与回写步骤，字段形态由执行者自行约定 | `tileops/kernels/sequence_modeling/engram_decode/engram_decode_kernel/integration_report.json`（两轮内容）、`integration_log.md#验证结果` | 在 Stage 5 skill 中写明 `pending_report` 占位结构与回写时机（report 完成后把 `run_json`/`report_md`/`status` 覆盖为真实值） | R |
| Info-source | benchmark 侧 msprof 对多 launch 算子必然无效，但集成前无任何提示；本算子 4 个 workload 全 invalid 属结构必然，不是回归 | `reports/tileops/20261123_143920_807616_EngramDecodeFwdOp/run.json#warnings`（4 条 `unfiltered capture contains 2/3 distinct ops`） | 在 Stage 5 skill 的 report 验收段补一句：一次 forward 发多个 kernel 的算子预期 msprof 无效，改用 `--prof-mode events` 取有效 latency | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| wrapper `default_config` 委托集成模块的同名 `default_config`，避免 Stage 3 冻结 tiling 与胶水层漂移 | P | `tileops/kernels/sequence_modeling/engram_decode/engram_decode.py#default_config`、`#_engram_decode_wrapped`；实测 d∈{256,512,600} 三档 `Kernel.config` 均等于 `_engram_project_kernel.default_config(d_padded)`，step threads = `min(512, next_pow2(d_pad))` | `cd examples/TileOPs && python -c "import torch; from tileops.kernels.sequence_modeling.engram_decode.engram_decode import EngramDecodeKernel as K; print(K(4,519,600,20,4,5,1e-6,torch.float16).config)"` → `{'block_m':16,'block_n':64,'block_k':256,'num_stages':2}`（构造不编译 kernel，无 device 工作） | tilelang 0.1.15；Ascend950PR_9579；CANN 9.2.1 / compiler 12.3.0 | pattern-library（集成胶水：config 单一来源） |
| 多 launch 算子的 msprof 无效实证与 events 兜底数值（同一算子 10x 量级差） | D | `reports/tileops/20261123_143920_807616_EngramDecodeFwdOp/run.json#performance.cases`（4/4 invalid：serving-b1 3.141us、serving-b4 4.579us、serving-b8 6.385us、decode-partial-cache 3.092us）；`reports/tileops/20261123_144235_457422_EngramDecodeFwdOp/run.json#performance.cases`（events 10/10 valid：45.90 / 46.63 / 46.16 / 54.95us） | `cd examples/TileOPs && TILEOPS_TASK_DIR=…/examples/engram_decode python -m tileops.reporting.cli run --op EngramDecodeFwdOp --prof-mode msprof`（partial，4 条 invalid）对比 `--prof-mode events`（passed） | tilelang 0.1.15；Ascend950PR_9579（28 cube / 56 vector）；CANN 9.2.1 / compiler 12.3.0；torch_npu 2.13.0rc1 | bottleneck-patterns（profiling 能力边界） |
| host 侧 `F.pad` 会作为独立 `PadV3` 设备算子进入 msprof capture，使"2 launch"变"3 distinct ops" | D | 同上 msprof run.json，`decode-partial-cache`（conv_len 5 < max_conv_len 12）`captured_op_names = [PadV3_e1345c33…, engram_decode_step_kernel, engram_project_kernel]` | 同上，观察 `decode-partial-cache` case 的 `msprof.captured_op_names` 长度 3 与其余 case 的 2 | tilelang 0.1.15；Ascend950PR_9579；CANN 9.2.1 | bottleneck-patterns（wrapper 胶水引入的额外设备算子） |
| report.md 算子表的 correctness 分母可能大于该算子真实用例数（analyzer 回落归属） | C | `reports/tileops/20261123_144235_457422_EngramDecodeFwdOp/report.md#Operator Analysis`（28/28）vs `run.json#summary`（46/46）与 `#correctness.cases` 中 `properties.op=EngramDecodeFwdOp` 仅 16 条；`tileops/reporting/analyzer.py#_case_operator`（无 `op` property 的多算子共享测试文件回落到 `default`） | `cd examples/TileOPs && python -c "import json;d=json.load(open('reports/tileops/20261123_144235_457422_EngramDecodeFwdOp/run.json'));print(d['summary']['passed_cases'])"` → 46；逐 case 读 `properties.op` | 不适用（纯报告逻辑） | pattern-library（报告解读陷阱） |

### Transferable Lessons（可迁移教训）

- Stage 5 集成后必须逐项核对 wrapper 的 `default_config` 与 dispatch 实参是否等于该函数 Stage 3 模块的 `default_config(...)`；scaffold 常残留 GPU 派生值（`block_k`/`num_stages`/`threads`），而 `integrate_kernel.py` 不会替你改。
- 胶水层取配置时优先 `from .{op_slug}_kernel.{func} import default_config` 委托，而不是把常量抄进 wrapper：重集成 Stage 3 产物时 wrapper 自动跟随，Stage 4 只需重指一处 import。
- 一次 forward 发多个 kernel 的算子，msprof 的单 kernel-name 过滤必然无效（`unfiltered capture contains N distinct ops`）；这属结构必然，按平台口径只记录，报告阶段用 `--prof-mode events` 拿有效 latency，且 events 与 msprof 数值不可直接比较（含 host 提交开销，量级可差 10x）。
- wrapper 里任何 torch 侧形状补齐（如 `F.pad` 把 `L < max_conv_len` 的 cache 左填充）都会以独立设备算子出现在 profile capture 中；需要单 kernel 归因时先消除或折叠这些 host 侧算子。
- 多算子共享一个 `tests/ops/test_*.py` 时，`report.md` 的算子级 correctness 分母会把无 `op` property 的共享用例回落到本算子；验收一律以 `run.json#summary`（全量）与逐 case `properties.op`（本算子）为准。
