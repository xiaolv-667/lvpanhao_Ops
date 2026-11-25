# 950PR MFU v2（Stage 4 优化后，9/11）

> 基线 v1（未优化）已冻结：`docs/950pr-mfu-baseline-v1.md` / `examples/batch2/mfu_baseline_v1.json`。
> 本文件记录 Stage 4 生产采纳后的结果；逐轮细节见 `docs/950pr-mfu-optimization-log.md`。
> 日期：2026-11-25。硬件：Ascend950PR_9579（28 Cube / 56 Vector，128 GiB）。

## 1. 总览

| 算子 | 关键 case | v1 T_op | v2 T_op | 加速 | v1 指标 | v2 指标 |
|---|---|---:|---:|---:|---|---|
| **SharedExpertMLP** | 512×7168×18432 | 2147.0 µs | **1203.3 µs** | 1.78× | MFU 50.02% | **MFU 89.24%**（实测 91.1%） |
| **SSDChunkCoupling** | grouped-prefill | 171.0 µs | **83.2 µs** | 2.06× | MFU 30.06% | **MFU 61.79%** |
| **RoPE** | llama3-8b-b2-2k | 671.1 µs | **150.1 µs** | 4.47× | MBU 15.82% | **MBU 70.0%** |
| **ChainSpecSampling** | ds-v3-mtp-n3 | 2542.8 µs | **714.6 µs** | 3.56× | MBU 0.81% | **MBU 2.89%** |
| **PagedKVCacheWrite** | qwen3-235b-fp8 | 267.7 µs | **32.5 µs** | 8.24× | MBU 2.95% | **MBU 24.3%** |
| PagedKVCacheWrite | llama-8b-prefill | 884.7 µs | **41.8 µs** | 21.2× | — | 1.40 TB/s（模型口径） |
| **EngramGateConvFwd** | train-s96 | 11.0 µs | **8.8 µs** | 1.25× | MBU 12.59% | 15.75% |
| **EngramGateConvBwd** | train-s96（2 launch） | 15.0 µs | **12.4 µs** | 1.21× | MBU 14.71% | 18.01% |
| **EngramDecode** | serving-b8 | 9.5 µs | **6.1 µs** | 1.55× | MBU 16.35% | 7.09%（b8 小流量） |
| EngramDecode | serving-b4（主导） | 8.2 µs | 8.2 µs | 1.00× | 2-launch 地板 | 2.20% MFU |
| **MoEPermuteAlign** | kimi-k2-t4096 | 650.7 µs | **29.7 µs** | 21.9× | FLOPs=0 | 纯时延 |

- **9/11** 已完成 Stage 4 并生产采纳；**MLA dense/varlen 待设计修订**（换 Cube）。
- 逐算子加速 1.2×–21.9×，**几何平均约 3.1×**（按各自主导 case）。
- 全部 9 个：L0/L1 回归通过、输出逐位/容差不变、单算子报告 `status=passed`、`invalid_*` 为 0。

## 2. 口径与说明

- **MFU** 仍是唯一验收指标（Cube 378 / Vector 47），`MU` 即 **MBU**（带宽利用率）。
- `T_op` = 一次调用内所有 kernel 之和；多 launch 算子（Engram Bwd/Decode、MoE、MLA）必须求和。
- **L2-resident**：模型字节/T_op 可以超过 1.6 TB/s（输入跨 launch 驻 L2，例如 Paged prefill 1.40–1.62 TB/s）。
  这不再判 `invalid`，只记 `bandwidth-above-hbm: L2-resident`；`invalid` 仅用于“算力超过峰值”（grid 截断类）。
- **launch 地板**：Engram 系列（8–15 µs）与 MoE t1/t32 的收益受 1.1–2.7 µs/launch 固定成本限制；
  EngramDecode 主导 b4 已到 69% 地板，代码与 v1 逐字节相同。
- `SSDChunkCoupling` 与 `EngramDecode` 的 `[DESIGN_LIMIT]`/平台地板项已按归档处理，不改设计。

## 3. 产物索引

| 产物 | 路径 |
|---|---|
| 当前逐 case（69 行） | `examples/batch2/mfu_cases.csv` |
| 当前算子级（11 行） | `examples/batch2/mfu_operators.csv` |
| 完整表 | `examples/batch2/MFU_ALL.md` |
| v1 冻结基线 | `examples/batch2/mfu_baseline_v1.json` / `docs/950pr-mfu-baseline-v1.md` |
| 逐轮优化日志 | `docs/950pr-mfu-optimization-log.md`；各 `examples/{op}_perf/perf_opt/` |
| 生产报告 | `examples/TileOPs/reports/tileops/2026112{4,5}_*` |

## 4. 待办

1. **MLA dense/varlen**：当前 MFU 67.0% / 47.1%（Vector 口径）；换 Cube 的理论空间约 8×，
   属设计修订（Stage 1），不是 Stage 4 参数调优。
2. 可选：Vector 峰值实测标定、fp32/fp8 Cube 峰值标定。
3. 交付仓（`TileOPs-ascend950`）按约定不在本轮范围。
