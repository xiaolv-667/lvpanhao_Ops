# Ascend950DT_9582 型号资料

适用型号为 `ascend-950dt-9582`，与 `ascend` 工具链组合。核验日期：2026-11-19。

## 型号配置证据

已逐项读取锁定 CANN 9.2.0 SDK 的 `data/platform_config/Ascend950DT_9582.ini`。文件指纹见 [核验来源](../toolchains/ascend/references/evidence.md)。以下为 SDK 的型号描述，不是运行时可分配资源或真机测量结果。

| 信息 | INI 字段与值 | 用法 |
|---|---|---|
| 架构 | `[version] NpuArch=3510` | 本工具链以 `ASCEND_NPU_ARCH=dav-3510` 编译 |
| 执行单元 | `[SoCInfo] cube_core_cnt=32`、`vector_core_cnt=64` | 型号规格；设备运行前仍由平台查询核对 |
| Cube/Vector 组织 | `[SoCInfo] cube_vector_combine=split` | 不据此推导具体 kernel 的分核或同步策略 |
| 主存大小字段 | `[SoCInfo] memory_size=103079215104` | 记录配置原值；本次未进一步核对该字段单位，不作为当前可用显存 |
| L2 大小字段 | `[SoCInfo] l2_size=134217728` | 记录配置原值；本次未进一步核对该字段单位，不作为 UB/L1 容量使用 |
| UB 块大小字段 | `[AICoreSpec] ubblock_size=32` | 与编译流水线 UB 分配的 32 字节对齐设置分别核对；不能推出所有 API 的长度都必须是 32 字节倍数 |

UB/L1/L0A/L0B/L0C 的设计预算继续以 [型号 JSON](ascend-950dt-9582.json) 为入口。本次已与上述 INI 对照，五项一致；本说明不另建容量数值表。

INI 中还含频率、MemoryRates、intrinsic dtype 等字段。本次没有将这些字段换算成实测带宽、指令成本或 TileLang API 支持表：硬件/SDK 的字段与具体编译路径、运行时行为是不同的证据。

## 资源核算

缓冲占用按实际分配 shape、dtype 字节数、padding 和版本数计算。例如，64 个 FP32 元素的一个缓冲占 256 字节；两个独立版本的逻辑占用为 512 字节。别名、生命周期复用和最终分配由编译结果确认，不能仅按源代码变量数相加，也不能把调度器的容量限制解除当作物理容量增加。

对应分配接口、版本注解与来源见 [编程与资源规则](../toolchains/ascend/references/programming.md)。
