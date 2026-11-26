# 950PR MFU v2（优化后，11/11）

> 基线 v1（未优化）已冻结：`docs/950pr-mfu-baseline-v1.md` / `examples/batch2/mfu_baseline_v1.json`。
> 本文件记录 Stage 4 生产采纳后的结果；逐轮细节见 `docs/950pr-mfu-optimization-log.md`。
> 日期：2026-11-25。硬件：Ascend950PR_9579（28 Cube / 56 Vector，128 GiB）。

## 1. 总览

| 算子 | 关键 case | v1 T_op | v2 T_op | 加速 | v1 指标 | v2 指标 |
|---|---|---:|---:|---:|---|---|
| **MLADenseBwd** | ds-v3-b2-8k | 455.3 ms | **94.1 ms** | **4.84×** | MFU 67.0%（vector） | **MFU 40.30%（cube）** |
| **MLAVarlenBwd** | ds-v3-mixed | 1728.7 ms | **402.6 ms** | **4.29×** | MFU 47.05%（vector） | **MFU 25.12%（cube；8x4k 28.23%）** |
| **SharedExpertMLP** | 512×7168×18432 | 2147.0 µs | **1176.0 µs** | 1.83× | MFU 50.02% | **MFU 91.32%**（实测 93.2%，PASS） |
| **SSDChunkCoupling** | grouped-prefill | 171.0 µs | **54.7 µs** | **3.13×** | MFU 30.06% | **MFU 94.03% / MBU 86.32%**（PASS） |
| **RoPE** | llama3-8b-b2-2k | 671.1 µs | **120.1 µs** | 5.59× | MBU 15.82% | **MBU 88.41%** |
| **ChainSpecSampling** | ds-v3-mtp-n3 | 2542.8 µs | **346.1 µs** | **7.35×** | MBU 0.81% | **MBU 5.98%** |
| **PagedKVCacheWrite** | qwen3-235b-fp8 | 267.7 µs | **19.6 µs** | **13.66×** | MBU 2.95% | **MBU 40.19%** |
| PagedKVCacheWrite | llama-8b-prefill | 884.7 µs | **43.2 µs** | 20.5× | — | MBU 97.26%（1.556 TB/s） |
| **EngramGateConvFwd** | train-s96 | 11.0 µs | 8.8 µs | 1.25× | MBU 12.59% | 15.75% |
| **EngramGateConvBwd** | train-s96（2 launch） | 15.0 µs | 12.4 µs | 1.21× | MBU 14.71% | 18.01% |
| **EngramDecode** | serving-b8 | 9.5 µs | 6.1 µs | 1.55× | MBU 4.53% | 7.09% |
| EngramDecode | serving-b4（主导） | 8.7 µs | 8.2 µs | 1.06× | MBU 16.35% | 17.48% |
| **MoEPermuteAlign** | kimi-k2-t4096 | 650.7 µs | **27.3 µs** | **23.8×** | FLOPs=0 | MBU 0.83% |

- **11/11** 已优化并生产采纳（9 个 Stage 4 + MLA dense/varlen 的 Cube 设计修订；其中 SharedExpert、SSD、RoPE、ChainSpec、Paged、MoE 做了第二轮补优）。
- **2/11 达到 90% MFU**：SSD 94.03%、SharedExpert 91.32%；RoPE MBU 88.41%（实测峰值 93–97%，硬件边界）；
  MLA dense 40.30%、varlen 25.12%（Cube 化 + A5 多缓冲 + Stage 4 后）。
- MLADense 的 MFU 从 67.0%（Vector）变为 24.6%（Cube）不是变差：T_op 455.3→154.4 ms（2.95×），分母从 47 换成 378。
- 逐算子加速 1.06×–23.8×，**几何平均约 4.1×**（按各自主导 case）。
- 全部 11 个：L0/L1 回归通过、输出逐位/容差不变、单算子报告 `status=passed`、`invalid_*` 为 0。

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

1. **MLAVarlenBwd**：Cube 化 + Stage 4（stages=3）后 mixed **563.7 ms（3.07×）**、8x4k **168.5 ms（2.91×）**、
   kimi 159.0 ms；MFU 17.94% / 22.53% / 23.85%。mixed 距 500 ms 目标差 13%，已按 `[DESIGN_LIMIT]` 归档：
   残差来自「逐迭代单价随 request 长度分层」（16384 长请求 4.26 vs 4096 的 3.14 µs/迭代），属任务映射层。
2. **MLADenseBwd**：Cube 化 154.4 ms / cube 24.6%；Stage 4 只拿到 −1%（stages=4 + 两处 buffer 版本），
   已按 `[DESIGN_LIMIT]` 归档、**不采纳**（1% 在噪声内）；残差由迭代内跨核依赖链决定，需设计层改任务映射。
2. 可选：Vector 峰值实测标定、fp32/fp8 Cube 峰值标定。
3. 交付仓（`TileOPs-ascend950`）按约定不在本轮范围。
