# Ascend950PR_9579 型号资料

适用型号为 `ascend-950pr-9579`，与 `ascend` 工具链组合。核验日期：2026-11-23。

## 型号配置证据

已逐项读取本机 CANN 9.1.0 SDK 的 `x86_64-linux/data/platform_config/Ascend950PR_9579.ini`（20374 B，md5 `2f0eca1f7ca70f62ee7d6d33990c3be1`）。以下为 SDK 的型号描述，不是运行时可分配资源或真机测量结果。

| 信息 | INI 字段与值 | 用法 |
|---|---|---|
| 架构 | `[version] NpuArch=3510` | 本工具链以 `ASCEND_NPU_ARCH=dav-3510` 编译 |
| 族名 | `[version] Short_SoC_version=Ascend950` | 用于把 SKU 归一化到族；容量按本档案取值，不按族名 |
| 执行单元 | `[SoCInfo] ai_core_cnt=28`、`cube_core_cnt=28`、`vector_core_cnt=56` | 型号规格；设备运行前仍由平台查询核对 |
| Cube/Vector 组织 | `[SoCInfo] cube_vector_combine=split` | 不据此推导具体 kernel 的分核或同步策略 |
| 主存大小字段 | `[SoCInfo] memory_size=137438953472` | 记录配置原值（128 GiB），不作为当前可用显存 |
| L2 大小字段 | `[SoCInfo] l2_size=134217728` | 记录配置原值（128 MiB），不作为 UB/L1 容量使用 |
| UB 块大小字段 | `[AICoreSpec] ubblock_size=32` | 与编译流水线 UB 分配的 32 字节对齐设置分别核对；不能推出所有 API 的长度都必须是 32 字节倍数 |

UB/L1/L0A/L0B/L0C 的设计预算以 [型号 JSON](ascend-950pr-9579.json) 为入口，取自 `[AICoreSpec]`：

| 资源 | INI 字段 | 字节 | 换算 |
|---|---|---:|---|
| UB | `ub_size` | 253952 | 248 KiB |
| L1 | `l1_size` | 524288 | 512 KiB |
| L0C | `l0_c_size` | 262144 | 256 KiB |
| L0A | `l0_a_size` | 65536 | 64 KiB |
| L0B | `l0_b_size` | 65536 | 64 KiB |

## 与相邻型号的关系

- `ascend-950dt-9582`（Ascend950DT_9582）：五项容量与此型号**完全一致**（UB 248 KiB / L1 512 KiB / L0C 256 KiB / L0A 64 KiB / L0B 64 KiB，`l2_size` 同为 128 MiB，`cube_freq` 同为 1650）。差别在核数（32/64 vs 28/56）与主存（96 GiB vs 128 GiB）。
- `Ascend950`（`tilelang/utils/npu_arch.py` 的 `CHIP_SPECS` 条目）：四项容量与本档案一致，但 `cores` 记 24（实测 28）、`L2` 记 112 MiB（INI 为 128 MiB）。该表不参与本平台层的容量解析。

## 资源核算

缓冲占用按实际分配 shape、dtype 字节数、padding 和版本数计算。例如，64 个 FP32 元素的一个缓冲占 256 字节；两个独立版本的逻辑占用为 512 字节。别名、生命周期复用和最终分配由编译结果确认，不能仅按源代码变量数相加，也不能把调度器的容量限制解除当作物理容量增加。

对应分配接口、版本注解与来源见 [编程与资源规则](../toolchains/ascend/references/programming.md)。

## 待核对项

- 本档案的容量是 SDK 型号描述；UB 在运行时扣除系统预留后可分配的实际额度仍需实测确认。
- `observed_cores` 与运行前 `--probe-cores` 的结果必须一致；不一致时以运行时查询为准并更新本文件。

## 站点 profile 偏离声明（相对官方 canonical `ascend950-arm64-v1`）

本站使用独立 profile `ascend950pr-9579-x86_64-v2`（见 `../toolchains/ascend/toolchain.lock.json`）。
参考基准为 `TileOPs-ascend950/config/ascend950-toolchain.json` 与 `docs/ascend950/toolchain.md`。

| 项 | 官方 canonical | 本站 | 状态 |
|---|---|---|---|
| 编译器 SDK | CANN 9.2.0 release | **CANN 9.2.0**（`/home/l00970450/zuo/Ascend/cann-9.2.0`） | ✅ 已对齐 |
| 运行时 / OPP | CANN 9.1.0 | **CANN 9.1.0**（`/usr/local/Ascend/cann-9.1.0`） | ✅ 已对齐 |
| host C++ | GCC 12.3.0 | **GCC 12.3.0**（`/home/l00970450/zuo/.environment/gcc-12.3.0`） | ✅ 已对齐 |
| torch / torch-npu / tilelang | 2.13.0+cpu / 2.13.0rc1 / 0.1.15 | 同 | ✅ 已对齐 |
| 机器架构 | aarch64 | x86_64 | ⚠️ 契约允许本地开发，但不受 ARM64 二进制指纹认证 |
| Python | 3.12.15 | **3.12.15**（`.venvs/ascend950-py31215`） | ✅ 已对齐 |
| 硬件 | Ascend950DT_9582（32 Cube / 64 Vector） | Ascend950PR_9579（28 / 56） | ⚠️ 按任务要求使用 950PR |
| A5 编译器变体 | — | 9.2.0 只提供 `bishengir-compile/opt` 与 `hivmc-a5`；`bishengir-compile-a5`、`bishengir-opt-a5` 仅存在于 9.1.0 | ✅ **已实测（2026-11-23）**：9.2.0 的统一编译器**可直接产出 A5 kernel**，与 `TILEOPS_950PR_A5_SHIM=1`（9.1.0 的 `-a5` 变体）路径产物**逐字节相同**（`executable.so` sha256 `1eaf0bce…`，编译 2.7 s / 2.2 s）。因此默认不需要 A5 shim，`TILEOPS_950PR_A5_SHIM` 仅作备选保留。冒烟全程 `ASCEND_RT_VISIBLE_DEVICES` 为空，**未占用任何卡** |
| 安装包摘要 | conda 包 sha256 记录在锁 | 本机为已解压目录，无安装包摘要（锁中记为 null 并说明） | ⚠️ 无法比对安装包来源 |

**结论**：容量与版本层面已与官方 canonical 一致；剩余偏离不阻塞功能迁移，但本站产出**不构成 canonical 认证证据**，性能数字只能作相对参考。认证仍需 ARM64 + 950DT 的 canonical CI。
