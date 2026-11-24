# 950PR 算子交接计划（11 个算子 → TileOPs-ascend950）

> 依据：本批 11 个算子在框架 `examples/TileOPs` 内完成迁移与真机验收；
> 本文只规划如何落到交付仓 `TileOPs-ascend950`，不改代码。
> 核对日期：2026-11-24。交付仓布局：`src/tileops/{kernels/ascend,ops,manifest/spec}` + `tests/ops` + `benchmarks/ops`。

## 1. 交付仓现状（逐算子）

| # | 算子 | manifest | Op 类 | ascend kernel | test | bench | 本批需要做的 |
|--:|---|---|---|---|---|---|---|
| 1 | MLADenseBwdOp | ✗ | ✗ | ✗ | ✗ | ✗ | **先移植 spec/Op/接口/test/bench，再落 kernel** |
| 2 | MLAVarlenBwdOp | ✗ | ✗ | ✗ | ✗ | ✗ | 同上 |
| 3 | MoEPermuteAlignFwdOp | ✓ | ✓ | ✗ | ✓ | ✓ | 只落 kernel + 注册 |
| 4 | SharedExpertMLPFwdOp | ✓ | ✓ | ✗ | ✓ | ✓ | 只落 kernel + 注册 |
| 5 | PagedKVCacheWriteFwdOp | ✓ | ✓ | ✗ | ✓ | ✓ | 只落 kernel + 注册 |
| 6 | EngramGateConvFwdOp | ✓ | ✓ | ✓（950DT） | ✓ | ✓ | 用 PR 版替换/对齐后重验 |
| 7 | EngramGateConvBwdOp | ✓ | ✓ | ✓（950DT） | ✓ | ✓ | 同上 |
| 8 | EngramDecodeFwdOp | ✓ | ✓ | ✓（950DT） | ✓ | ✓ | 同上 |
| 9 | RoPEFwdOp | ✓ | ✓ | ✓（950DT） | ✓ | ✓ | 同上 |
| 10 | SSDChunkCouplingFwdOp | ✓ | ✓ | ✓（950DT） | ✓ | ✓ | 同上 |
| 11 | ChainSpeculativeSamplingFwdOp | ✓ | ✓ | ✓（950DT） | ✓ | ✓ | 同上 |

**结论**：5 个算子需要新建/适配 ascend kernel（#3/#4/#5）或整套移植（#1/#2）；6 个已有 DT 实现的需要“替换为 PR 验证版 + 重跑该仓回归”。

## 2. 映射规则（框架 → 交付仓）

| 框架产物（`examples/TileOPs`） | 交付仓目标 | 处理方式 |
|---|---|---|
| `tileops/kernels/{family}/{op}/{op}_kernel/{func}.py` | `src/tileops/kernels/ascend/{family}/{op}.py` | 把 `{func}` 工厂**内联或 import** 进新 Kernel 文件；不能直接复制 wrapper |
| 同目录 `{func}_DESIGN.md` | （不落仓） | 作为设计依据留存框架侧 |
| `tileops/kernels/{family}/{op}/{op}.py`（wrapper） | 新 `Ascend{Op}Kernel` 类 | 重写为 `Kernel + {Op}Interface`，实现 `entry_for/refusal/__init__/forward` |
| `tileops/manifest/{family}.yaml` | `src/tileops/manifest/spec/{family}.yaml` | 仅 #1/#2 需要新增条目（其余已存在） |
| `tileops/ops/{family}/{op}.py` | `src/tileops/ops/{family}/{op}.py` | 仅 #1/#2 需要移植 |
| `tests/ops/test_{slug}.py` | `tests/ops/…` | 仅 #1/#2 需要移植；共享文件注意只带走本算子用例 |
| `benchmarks/ops/bench_{slug}.py` | `benchmarks/ops/…` | 仅 #1/#2 需要移植 |

## 3. Kernel 适配模板（交付仓约定）

```python
"""Ascend <Op> — PR-validated TileLang kernel."""
from typing import ClassVar
import torch
from tileops.backend import register_kernel_type
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.<family>.<op>.call_spec import <Op>Call, <Op>Interface

class Ascend<Op>Kernel(Kernel, <Op>Interface):
    devices: ClassVar[frozenset[str]] = frozenset({"npu"})
    supported_archs = None

    @classmethod
    def entry_for(cls, call: <Op>Call) -> Entry:
        return call, lambda: cls(call)

    @classmethod
    def refusal(cls, call: <Op>Call):
        # 与 kernel 的真实能力一致（见 §4 的口径风险）
        return super().refusal(call)

    def __init__(self, call: <Op>Call):
        super().__init__(device_index=call.device.index if call.device is not None else None)
        # 调用框架产出的工厂（或内联的工厂），按 call 的 shape/config 构建
        self.kernel = <factory>(...)

    def forward(self, *args):
        self._require_devices(...)
        return self.kernel(*args)   # 注意多 launch 顺序、fp32 中间量、输出 dtype

register_kernel_type("<OpName>", "ascend", Ascend<Op>Kernel)
```

要点：
1. **接口/命名不通用**：交付仓用 `*Interface` + `*Call`（`call_spec.py`），框架包用 wrapper 的 `default_config/kernel_map`；落仓时必须按交付仓接口改写。
2. **多 launch 顺序**：MLA 三段（delta→bwd→dq_land）、Engram bwd 两段、MoE align→scatter、SharedExpert gemm→reduce→silu、Decode project→step —— `forward` 里必须保持顺序并复用同一 `split_k`/config。
3. **host 侧义务**：`SharedExpert` 的 `split_k/block_m/block_k`、MLA 的 `dq_acc` 预清零、EngramDecode 的 `align_up/pad` 必须在 `__init__`/`forward` 复现。
4. **生成码断言**：`_dq_land_kernel` 的 `vlds_norm|vmuls|vcvt` + `vsts_pack_b32` 断言建议保留为 `__init__` 后的编译期检查。

## 4. 已知口径风险（落仓前必须对齐）

| 风险 | 影响算子 | 处置 |
|---|---|---|
| refusal 上界不一致：kernel 真实 `d<=1024`，manifest/ops 写 4096 | EngramGateConvBwd | 落仓时 `refusal()` 写 1024；若要 4096 必须先实现设计 B1 |
| 共享测试文件 | Engram ×3（交付仓已允许 4096？需核对） | 落仓后按交付仓测试结构重跑 |
| 多 kernel 性能归因 | MLA dense/varlen、MoE、SharedExpert、Decode | 交付仓 benchmark 同样受单名过滤限制，先统一 T_op 协议 |
| DT 旧实现与 PR 新实现并存 | 6 个已有实现 | 二选一：替换并重跑回归，或保留 DT 并新增 `ascend_pr` kernel_type（不推荐，评测口径会分裂） |

## 5. 建议执行顺序（每步可独立验收）

1. **#5 PagedKVCacheWrite**（接口齐全、单 kernel、最快打通链路）→ 验证 `entry_for/refusal/forward` 模板。
2. **#3 MoEPermuteAlign**（3 kernel、有共享测试）→ 验证多 launch 接线。
3. **#4 SharedExpertMLP**（3 kernel + host 义务）→ 验证 split_k/block_m 适配。
4. **#1/#2 MLA 反向**：先移植 spec/Op/接口/test/bench（GPU main → 交付仓布局），再落 3 kernel。
5. **6 个已有实现**：逐个用 PR 版替换，跑该仓 pytest + benchmark，记录 950DT→950PR 差异。

## 6. 验收口径（与 MFU 文档一致）

- 正确性：交付仓 `pytest tests/ops/test_*.py` 全过（真机 950PR）。
- 性能：`scripts/benchmark_ascend.py` 同协议；`T_op` = 该调用所有 kernel 之和（见 `examples/batch2/` 的求和工具）。
- 判据：CV 计算型 MFU、访存型 MU、时延型对基线；峰值需在 950PR 标定（当前 378.47 TFLOPS 为框架自报值）。
- 6 个已有实现的“挑战目标”基线 = 交付仓当前 DT 版在同机同 shape 的实测。

## 6. 2026-11-24 复核补充（交付仓实际覆盖）

对 `TileOPs-ascend950` 做了只读核对（详见 `docs/950pr-p1-baseline-status.md`），§1 的表格需要按实际覆盖修正：

| 分类 | 算子 | 说明 |
|---|---|---|
| 1:1 可跑 | SSDChunkCoupling、EngramGateConvFwd/Bwd、EngramDecode、ChainSpecSampling | workload/shape/dtype 与我们逐张量一致 |
| 需换算 | RoPE | 交付仓无 positions 输入；等价物 `RoPENeoxPositionIdsFwdOp` |
| 有算子、无 NPU 实现 | SharedExpertMLP、MoEPermuteAlign | kernel `devices={"cuda"}` |
| 交付仓缺失 | MLADenseBwd、MLAVarlenBwd、PagedKVCacheWrite | 无 op/spec/kernel/bench；MLA 连 fwd 也未注册 ascend |

因此 §1 的“5 个需要新建/移植”要拆成：
- **3 个从零建**（MLA dense/varlen 整套 spec/op/test/bench + kernel；Paged 的 NPU kernel）；
- **2 个只补 NPU kernel**（SharedExpert、MoE permute，op/bench/test 已有）；
- **6 个替换/重验**不变。

方言注意：交付仓 kernel 用 `tilelang.ascend.language`（官方 wheel 0.1.15），本仓迁移版用 npuir fork，
**不能直接复制**；落仓时需要按交付仓模板重写，并保持两套 venv / `TILELANG_CACHE_DIR` 隔离。

性能对比前置：交付仓 `runner.py` 强制站点 `--clock-config`（仓库/本机均无），缺它只能 `--verify-only`；
且硬件不同源（950PR 28 Cube vs 950DT 32 Cube），时延不可直比，只能用 MFU 归一后比较。

## 7. 落仓试点（2026-11-24）

`MoEPermuteAlignFwdOp` 已作为第一个试点落到交付仓：

- 新增 `src/tileops/kernels/ascend/moe/permute_align.py`（`AscendMoEPermuteAlignKernel`，单 launch、
  纯 DMA + 标量，无 `T.simd`/`T.ascend_*`），并在 `ascend/moe/__init__.py` 注册
  `register_kernel_type("MoEPermuteAlignFwdOp", "ascend", ...)`；
- `pytest tests/ops/test_moe_permute_align.py --tileops-device npu --tileops-target detect` → **26 passed**；
- in-process `verify_only`：`kimi-k2-t4096` / `ds-v3-t4096` → `VERIFY_OK`，3/3 输出 `max_abs_err = 0.0`，
  与参考 `torch.equal` 全 True；
- 诊断时延 ~1.95 ms/call（t4096），比我们 fork 的 `t_op_us` 643 µs 慢 2–3×；正式计时受 site guard
  + 缺 clock-config 阻塞（见下）。

**结论**：`tilelang.ascend.language` 可以承载我们的实现；模板可复用。下一试点：`PagedKVCacheWriteFwdOp`。

### 交付仓正式计时的两个基础设施阻塞

1. `scripts/benchmark_ascend.py` 需站点 `--clock-config`（仓库/本机均无）；
2. `npu_guard` 把常驻的 `npu-exporter`（pid 543/622，持有 `/dev/davinci_manager`、`/dev/hisi_hdc`）
   判为“设备忙”，直接 exit 75。需要站点在 guard 中白名单该 pid，或在没有 exporter 的 host 上跑。

在此之前，落仓验证用 `benchmarks.ascend.runner.benchmark_case(..., verify_only=True)` 的 in-process 路径
（绕过 guard，仅证明正确性，不出正式时延）。

### 7.1 第二个试点：`PagedKVCacheWriteFwdOp`

- 新增 `src/tileops/kernels/ascend/attention/paged_cache_write.py`（`AscendPagedKVCacheWriteKernel`，
  DMA 行路径 + fp8 显式 clamp/RNE），注册 `register_kernel_type("PagedKVCacheWriteFwdOp", "ascend", …)`；
- 交付仓 pytest：**11 passed / 4 failed**，4 个失败**均不在 kernel**：
  1. `fp8-bf16-scaled`、`fp8-fp16-page-boundary`、`saturates_fp8_overflow`：参考侧
     `workloads/attention/paged_cache_write.py:129` 的 `aclnnIndexPutImpl` 不支持 fp8；
  2. `compiles_cold_to_its_operator`：`torch.compile(fullgraph=True)` 缺 npu Inductor backend
     （未装 `triton-ascend`）；图所有权检查本身已通过。
- 独立逐位核对（真实 op 分派）：13/13 exact、0 bad bytes，含主导 fp8 case（4096×4×128，scale 1.25/0.75）；
- 诊断性能：fp8-4096 **41.7 µs**（≈1.24 TB/s），prefill-8192 54.0 µs。

**需要决策**：是否允许修改交付仓的**共享测试/参考层**（不是 kernel）：
- `workloads/attention/paged_cache_write.py`：fp8 分散改为 host 侧执行（我们 fork 已验证同款）；
- `workloads/numerics.py`：npu-fp8 比对改 `view(torch.uint8)` 逐字节 + NaN 归一化。
这两处改完，上述 3 个 fp8 测试可转绿；`triton-ascend` 缺失属环境问题，无法在仓内修。
