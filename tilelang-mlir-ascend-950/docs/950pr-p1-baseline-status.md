# P1 · 交付仓基线对比现状（2026-11-24）

> 对象：`/workspace/TileOPs-ascend950`（交付仓，`ascend950` 分支，代号 DT_9582）
> 对照：本批 11 个已迁移算子（950PR_9579）
> 结论：**“和官方库比性能”目前有两个硬阻塞**（站点 clock-config 缺失、硬件不同源）；
> 正确性交叉验证只对 5 个算子可做，另有 3 个算子交付仓根本不存在、2 个是 CUDA-only。

## 1. 11 个算子在交付仓的覆盖

| 级别 | 算子 | 说明 |
|---|---|---|
| **1:1 可跑** | SSDChunkCoupling、EngramGateConvFwd、EngramGateConvBwd、EngramDecode、ChainSpeculativeSampling | workload 名/shape/dtype 与我们的主导 case 逐张量一致 |
| **需换算** | RoPE | 交付仓 `RoPEFwdOp` 无 positions 输入；等价物 `RoPENeoxPositionIdsFwdOp`（4096 token 对齐，dtype/base 不同） |
| **有算子无 NPU 实现** | SharedExpertMLP、MoEPermuteAlign | kernel `devices={"cuda"}`，npu 上选不到实现（不是慢，是跑不了） |
| **交付仓缺失** | MLADenseBwd、MLAVarlenBwd、PagedKVCacheWrite | 无 op/spec/kernel/bench；连 `MLAVarlenFwdOp` 也未注册 ascend |

静态验证：`benchmarks/ascend/coverage.py::registered_coverage()` 共 162 个 ascend 算子，
`load_failures` 为空；上述 7 个 `YES` / 4 个 `NO` 与上表一致。

## 2. 为什么现在拿不到“官方库时延”

| 阻塞 | 证据 | 影响 |
|---|---|---|
| **站点 clock-config 缺失** | `benchmarks/ascend/runner.py` 非 `--verify-only` 时强制 `--clock-config`；仓库内无该文件（只有 `clocks.py` 实现），站点 CI 脚本 `/opt/tileops-ci/...` 本机不存在 | **正式计时跑不了**，只能 `--verify-only` 拿正确性 |
| **host 锁目录缺失** | `/run/lock/ascend-npu/` 不存在；`npu_guard` 会自建（root + `libdrvdsmi_host.so` 齐全，可行） | 需在 guard 内启动整个进程 |
| **硬件不同源** | 我们 950PR：28 Cube / 378 TFLOPS；交付仓 950DT：32 Cube / 432 TFLOPS | **MFU 可比、裸时延不可直比**；Vector 峰值对方未公布，访存型算子连 MFU 也无法严格归一 |
| **工具链方言不同源** | 交付仓 kernel 用 `tilelang.ascend.language`（官方 0.1.15 wheel）；我们的 dev fork 无 `ascend` 子包 | 两仓 kernel **不能互搬**；必须两套 venv + 两套 `TILELANG_CACHE_DIR` |

补充：`einops==0.8.2` 缺失会让 `benchmarks/_cases` 的 eager import 整体失败（已安装）。
不要在 `/workspace/tilelang-mlir-ascend-950` 下启动交付仓 python（会遮蔽官方 wheel）。

## 3. 现在能做的正确性交叉验证（`--verify-only`）

| 算子 | case | 备注 |
|---|---|---|
| SSDChunkCoupling | `mamba2-grouped-prefill-float16` | 1:1 |
| EngramGateConvFwd | `train-s96-bfloat16` | 1:1，~10 µs 地板 |
| EngramGateConvBwd | `train-s96-bfloat16` | 1:1，注意 launch 次数 |
| EngramDecode | `serving-b4-float16` | 1:1，双 launch |
| ChainSpeculativeSampling | `ds-v3-mtp-n3` | 只比 `num_accepted`/分布，**不比 token** |
| RoPE | `prefill-4k-bfloat16-int32` | 语义等价物，需 reshape |

命令模板（工作目录 `/workspace/TileOPs-ascend950`）：

```bash
set +u; . /usr/local/Ascend/cann-9.2.1/set_env.sh; set -u
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.2.1
export ASCEND_NPU_ARCH=dav-3510
export PYTHONPATH=/workspace/TileOPs-ascend950/src:/workspace/TileOPs-ascend950
export TILELANG_CACHE_DIR=/tmp/opencode/tl-cache-delivery
/workspace/venv-ascend950/bin/python scripts/benchmark_ascend.py \
  --owner "$USER/p1-verify" --run-dir /tmp/opencode/runs/p1-verify-$(date +%s) \
  --op EngramDecodeFwdOp --case-regex '^serving-b4-float16$' --verify-only
```

## 4. 结论与需要的输入

1. **性能对比**：缺站点 `clock-config`，本机无法产出交付仓正式时延 → 需要项目方提供：
   `{"lock":[...], "restore":[...], "component_mhz":[a,b]}`，或提供一台 950DT 的正式跑数。
2. **正确性对比**：5 个算子可做 `--verify-only`；MLADense/MLAVarlen/Paged 无基线，
   只能以“交付仓无此实现”记录，不能虚构对比。
3. **交付落仓**：5 个需要新建/移植（MLA dense/varlen 整套、Paged/SharedExpert/MoE 仅 kernel），
   6 个已有 950DT 实现的按 `docs/950pr-handoff-plan.md` 替换后重跑回归。
