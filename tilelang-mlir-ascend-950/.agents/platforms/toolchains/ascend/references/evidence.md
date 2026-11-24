# 950 资料核验来源与范围

核验日期：2026-11-19。资料基于项目锁定的实际安装环境，未把其他后端的 API 或实测数字作为本工具链结论。

## 环境

- 型号配置：Ascend950DT_9582；仿真模型模式：Ascend950DT。
- TileLang：0.1.15 ARM64 wheel；Python 3.12.15；PyTorch 2.13.0+cpu；torch-npu 2.13.0rc1。
- 编译 SDK：CANN 9.2.0 release；runtime/OPP 与仿真模型：CANN 9.1.0；GCC 12.3.0。
- 项目 `toolchain.lock.json` 检查通过；实际工具链戳：`fad39935f627e0ea6f2dca0d681f3682cf53972ef4bb0d74193fc820c1362d3d`。
- 仿真器沿用已有临时型号查询补丁：处理模型驱动返回 PR 名称与 DT 型号门禁的差异，未修改本次 kernel 或计算结果。本次结果不证明真机设备身份、频率或性能。

环境组合的上游出处为 [工具链说明](https://github.com/yyttt6/TileOPs/blob/ascend950/docs/ascend950/toolchain.md)，实际版本与组件由项目内的 [工具链锁](../toolchain.lock.json) 校验。外部工程不作为验证脚本的导入或运行依赖。

## 源码核对

`tilelang/` 表示实际导入的安装包目录，通过 `tilelang.__file__` 定位；`ASCEND_HOME_PATH/` 表示本任务选定的编译 SDK。以下指纹记录本次读到的文件，不另设一套环境锁。升级后需重新核对相关结论。

| 核对文件 | SHA256 |
|---|---|
| `tilelang/ascend/language/__init__.py` | `585bc20974a0e15a21ca7ad546eb09731db0171384d0328057cbffb060274c9c` |
| `tilelang/ascend/language/kernel.py` | `8bee793b8745ad8095e69cb9ffa1f15039d6c9e98b0222e17f61fcf5c197d80a` |
| `tilelang/ascend/language/frame.py` | `f18686d7833a9f47ba0170a8672b8f19f82704c18b95db2c23c4d7ac16c3e5ba` |
| `tilelang/ascend/language/allocate.py` | `e1d25d5e1807314006538f5b76fae853f3fe438c403a92026aef6831dc230ad4` |
| `tilelang/ascend/language/copy_op.py` | `a9c75a7ba672a790c850ef5e76d0ecbec62f0b870d3ae0bd5c3a3a39630dbec8` |
| `tilelang/ascend/language/gemm_op.py` | `b4b222ed7a98d657fdc2ab566024a83c723cc463b06808fd69a5c0cbb0e7a892` |
| `tilelang/ascend/language/reduce_op.py` | `3a4a7df7a65f6c187ba4832de2f024fd6760fccf271325fa160899c7fdc61d44` |
| `tilelang/ascend/language/sync.py` | `ba04d675e009492a2f36305854be1d4b65e885819528d03fa7e53f0e1c41a6c9` |
| `tilelang/ascend/language/annotations.py` | `be8d7a3d3162909f988dd52bb161e11dfd417e287269e5120864302c026336eb` |
| `tilelang/ascend/language/simd.py` | `89483579ddde5e36a352a853720fc02ba2b06badeab9eca74b31bcde6a04b57f` |
| `tilelang/ascend/pipeline.py` | `d099e7a08b792fc04f605c5ac88b47381a1426a42f8f55c578dc81e3bf37e1ab` |
| `ASCEND_HOME_PATH/data/platform_config/Ascend950DT_9582.ini` | `b8e7d61780056889e795639504530ecacb2b759ad1602e929b863d837fdf0099` |
| `tilelang/ascend/op/gemm/gemm_mad.py` | `0f2d7021330e6bf6e6602902e9e6fb3b714765e7da874b79716da7c45d8e8c88` |
| `tilelang/ascend/language/mode.py` | `904873c209809dd36c3edc7c46d2850363c403e5450f4eb869cebb2fb7cce66e` |

`programming.md` 中的接口、参数校验、默认值、scope、流水线顺序等结论，分别来自上述文件中注明的定义。27 个导出接口已用 `inspect.signature` 核对；示例注释与签名不一致时，以当前实现为准。

[API 参数检查脚本](check_api_contract.py) 在不执行设备计算的环境中通过 8 项检查：Kernel threads 参数、SIMT 四维/零线程、完整 Buffer copy 元素数不等、L0C dual-copy dtype 变化、barrier/set_flag 参数类型、零缓冲版本。脚本检查的是前端拒绝行为，不推出相应硬件能力缺失。

## 执行证据

| 自包含脚本 | 运行位置 | 用例与结果 |
|---|---|---|
| [PL-950-row-primitives.py](../knowledge/repro/PL-950-row-primitives.py) | 950 DT 仿真器 | 广播、求和、softmax、RMSNorm × 3 种输入 dtype × N=63/64，共 24 项通过 |
| [PL-950-cube-tile.py](../knowledge/repro/PL-950-cube-tile.py) | 950 DT 仿真器 | M/N/K=16/16/32，FP16/BF16 输入、FP32 输出，共 2 项通过 |
| [ALG-950-formulas.py](../knowledge/repro/ALG-950-formulas.py) | CPU | softmax、LayerNorm、RMSNorm、无 mask attention 的 FP64 公式对照，共 8 项通过 |

行原语最大绝对误差为 `7.62939453125e-06`，单 tile GEMM 为 `2.86102294921875e-06`。逐项输入、参考实现和误差阈值保存在脚本及对应知识条目中。两个仿真进程均正常退出；没有使用 LLM 调用。

CPU 公式与 API 参数检查可从任务目录直接运行。设备脚本按任务平台 `execution.prototype_command` 启动，使用绝对脚本路径和新建输出目录，仿真串行执行。这里的 `repro/` 脚本是独立知识验证用例，不是阶段交付 kernel，保留自身的运行入口。

## 扩展仿真验证

沿用上述锁定环境和 DT 模型，所有用例串行运行，无 LLM 调用。38 项正确性检查通过，另有 2 组输入复核了高层 SIMD 归约的已知差异；共 10 个脚本正常退出。TRAP 脚本以预期差异为断言，不能作为正确实现使用。各脚本是独立知识验证用例，未改动主流程。

| 自包含脚本 | 核对项数 |
|---|---|
| [PL-950-normalization.py](../knowledge/repro/PL-950-normalization.py) | 6 |
| [PL-950-attention.py](../knowledge/repro/PL-950-attention.py) | 6 |
| [PL-950-gemm-tiles.py](../knowledge/repro/PL-950-gemm-tiles.py) | 8 |
| [PL-950-mixed-epilogue.py](../knowledge/repro/PL-950-mixed-epilogue.py) | 2 |
| [PL-950-manual-sync.py](../knowledge/repro/PL-950-manual-sync.py) | 1 |
| [PL-950-atomic-splitk.py](../knowledge/repro/PL-950-atomic-splitk.py) | 2 |
| [PL-950-conversion-layout.py](../knowledge/repro/PL-950-conversion-layout.py) | 6 |
| [PL-950-simd-reduce.py](../knowledge/repro/PL-950-simd-reduce.py) | 2 |
| [PL-950-softmax-boundaries.py](../knowledge/repro/PL-950-softmax-boundaries.py) | 5 |
| [TRAP-950-simd-shared-reduce.py](../knowledge/repro/TRAP-950-simd-shared-reduce.py) | 2 |

输入、参考实现和事先设定的精度阈值见脚本与对应知识条目。多缓冲与同步只记录正确性，不据仿真时间判断性能。

## 未认证范围

已验证的 LayerNorm/attention、特殊值、混合计算和同步路径均有具体 shape、dtype、输入和配置边界；所有 dtype/API 组合、手工跨核 flag、复杂循环同步、跨 block batch barrier 仍未认证。本次没有真机性能结论。
