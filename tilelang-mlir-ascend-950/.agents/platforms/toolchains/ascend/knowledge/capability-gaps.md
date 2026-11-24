# ascend 能力缺口

## CG-950-batch-barrier

工具链锁记录的未认证功能为 ascend.batch_schedule 的跨 block barrier 依赖。需要该依赖的设计在取得独立验证前应选择已有支持路径。

## CG-950-simd-shared-reduce

锁定 TileLang ascend 0.1.15 / CANN 9.2.0 SDK + 9.1.0 DT 仿真器上，高层 SIMD 的二维 shared→shared 归约出现输出差异：FP32 输入 (2,64)、输出 (2,)，第二行 sum 与 CPU 参考不一致。两组输入独立确认；生成代码的统计量写入间隔与 GM 拷出布局不一致。

本次工具链戳：fad39935f627e0ea6f2dca0d681f3682cf53972ef4bb0d74193fc820c1362d3d。来源为 20261119-ascend950-knowledge-extended，型号 ascend-950dt-9582。

完整证据与自包含反例为 [TRAP-950-simd-shared-reduce](numerical-boundaries.md)。已经验证的替代路径为同页 PL-950-simd-reduce 的显式寄存器归约，以及 [SIMT reducer](PL-950-row-primitives.md)。该结论限定于反例中的高层路径，不表示所有 SIMD 归约不支持。本次未修改工具链或主流程。
