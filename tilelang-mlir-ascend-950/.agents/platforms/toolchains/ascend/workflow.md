# TileLang ascend 工具链规则

## 环境与硬件

编译环境参考 [TileOPs ascend950 工具链说明](https://github.com/yyttt6/TileOPs/blob/ascend950/docs/ascend950/toolchain.md)，版本和文件指纹以本平台的 toolchain.lock.json 为准。使用 ARM64、官方 TileLang 0.1.15 wheel、CANN 9.2.0 release 编译 SDK、独立 CANN 9.1.0 runtime/OPP，以及锁定的 Python、PyTorch、torch-npu 和 C++ 编译器。按上游文档设置 ASCEND_HOME_PATH、ASCEND_RUNTIME_ROOT、ASCEND_NPU_ARCH=<hardware.architecture> 以及动态库路径。

用以下命令在 CPU 上检查软件版本和文件指纹。真机上改用 --probe，查询 torch.npu.get_device_properties 并验证设备型号和工具链。正式运行前用同一组合初始化或验证任务状态。

    python3 .agents/tools/statectl.py platform --hardware ascend-950dt-9582 --toolchain ascend --check-environment

## 已核验参考资料

- [950 型号资料](../../hardware/ascend-950dt-9582-notes.md)：型号配置字段及资源核算边界。
- [编程与资源规则](references/programming.md)：API、布局、搬运、同步和缓冲版本。
- [迁移与诊断参考](references/migration-diagnostics.md)：原语映射、检查顺序和证据范围。
- [知识索引](knowledge/INDEX.md)：已验证用例与算法资料。
- [核验来源](references/evidence.md)：安装包源码指纹、验证范围和自包含脚本。

## API 证据与执行环境

API 命名空间是 `tilelang.ascend.language`。用激活环境中的 Python 在任务目录进行静态查询，先核对 `T.__file__`，再按该安装位置的导出、定义、签名与本平台 kernel 模板查证。仓库根的同名源码包会遮蔽安装包，静态查询不要在仓库根运行。仓库 `tilelang/language/` 和 `docs/Tilelang.language/` 属于另一工具链，不能作为本工具链 API 可用性、分形、同步或模式的依据。

编程模式使用本平台 SIMD/SIMT/Cube/混合选项；SIMT 的逐元素表达式放在 `T.SimtVF` 内。是否有某个符号或某种写法，以选定模块和当前版本为准。算法候选可以复用数学思路，API 与代价结论仍需当前组合的证据。

`statectl platform` 返回的 `execution` 选择设备或仿真启动方法。仿真时，任何创建 NPU 张量或执行 kernel 的脚本（包括设计期原型）均按其中 `prototype_command` 或 `precision_command` 启动；先进入任务目录，传绝对脚本路径，输出目录每次新建。仅导入模块和检查签名可直接 Python。主流程向所有阶段完整传递该执行上下文。

## 设计、实现与排错

从任务的平台配置读取容量预算。记录 torch.npu.get_device_properties 返回的 Cube 和 Vector 核数，根据实际执行路径选择 SIMD、SIMT、Cube 或混合模式。向量操作参考上游 elementwise 的 T.SimdVF、T.simd.* 或 T.SimtVF。Cube 设计对照上游 GEMM 检查布局、累加缓冲和数据搬运。ascend.batch_schedule 的跨 block barrier 依赖尚未通过该工具链认证。设计评审核对实际 tile 占用、双缓冲、尾块保护和同步区间。

使用 import tilelang.ascend.language as T、@T.prim_func 和 tilelang.compile(..., target="ascend", out_idx=...)。参照平台 kernel 模板。用 PyTorch 参考实现和 --level L0 / --level all 执行分层精度验证。排错先检查锁定环境与目标架构，再用最小 shape 复现并保留编译器日志、生成代码和异常输入。

## 性能采集

沿用项目原有 kernel-only Task Duration(us) 口径、A/B 交错采集、workload inventory、perf_records.jsonl 和 winner 协议。950 适配器调用 CANN 自带的 msopprof，读取 OpBasicInfo CSV，原始证据仍写 msprof_raw_path，主指标仍为 msprof_task_duration。warm-up、launch-count 和重试继续由原有流程控制。

使用原有 .agents/tools/ab_test.py 和实验 runner，传入 --task-dir <任务目录>；候选按平台规则先验精度，再比较同一 workload 的 Task Duration。

能力缺口及其证据见 knowledge.capability_gaps 指定的数据文件。

## TileOPs 工程迁移

沿用项目内 examples/TileOPs/ 的原有七文件脚手架、GPU 源函数提取、逐函数 Stage 1/2/3、baseline/perf_opt 切换和 Stage 5 报告流程。GPU 源仓使用任务提供的路径。

Stage 0 使用 integration.scaffold_skill 指定的现有 add-npu-op 技能。该技能中的 NPUIR 编程示例在本平台按上述 ascend API 和模板实现；manifest、wrapper、测试、benchmark 和 .migration_meta.json 保留原有布局和字段。

Stage 5 在 examples/TileOPs/ 下执行原有 integration.script --meta <meta_path> --task-dir <聚合任务绝对目录>。集成仍复制 kernel 和 DESIGN.md、生成聚合导出并接线原有 wrapper；ascend 后端不使用 NPUIR 的 Developer/Expert 模式声明，wrapper.ascend_mode 置为 None。

从同一目录运行 TILEOPS_TASK_DIR=<聚合任务绝对目录> python -m tileops.reporting.cli run --op {op_name} --prof-mode msprof。报告链路通过保存的任务选择调用 950 profiler，继续产出原有 run.json、report.md 和 report.html，integration_report.json 仍引用 run_json。正确性必须通过；benchmark 为 partial 时沿用原有记录与验收规则。会话耗时继续使用原有 Markdown 和 render --with-session-timing。

## 工程公共工具与资源规则

工程中所有预算查询、集成和 report 都设置 `TILEOPS_TASK_DIR=<保存的平台任务目录>`。纯数学 tiling 计算可用 `tileops.utils.tiling`，明确传入当前平台核验的 alignment 和 budget；NPUIR 的 reduction 宏、H200 列上限及其 8 KiB 分配预留不是本工具链的默认规则。向量、归约和同步的实际 API 仍按 `tilelang.ascend.language` 查证。

采集器为 `msopprof`，命令参数由本工具链适配器的 `profiler_args` 提供；使用原 A/B 与实验 runner 并传 task-dir，不照抄 NPUIR 的 `msprof op` 或 dump 参数。分核查询使用 `torch.npu.get_device_properties` 分别记录 Cube/Vector；不要使用 NPUUtils 或固定乘二替代本平台查询。

仿真执行顺序：所有模型进程串行运行，包括各层精度与主流程二次校验；不要并发启动多个模型争用 CPU 和内存。精度层可用单次 --level all 覆盖，以减少模型启动开销。
