---
id: PL-950-conversion-layout
kind: pattern
status: verified
origin_task: 20261119-ascend950-knowledge-extended
toolchain: TileLang ascend 0.1.15; CANN 9.2.0 SDK / 9.1.0 simulator; 2026-11-19
toolchain_stamp: fad39935f627e0ea6f2dca0d681f3682cf53972ef4bb0d74193fc820c1362d3d
toolchain_id: ascend
hardware_id: ascend-950dt-9582
repro: repro/PL-950-conversion-layout.py
---

# SIMT 转换特殊值与带广播的转置

## 转换

输入为 FP32，长度 17，分别转换为 FP32、FP16、BF16。样本包含 ±0、±Inf、NaN、±1e-38、±1e-8、65504/±65520、1.00390625/1.01171875、2/-2/0.5。实际用例与 torch 的 dtype 转换对照，atol=rtol=0、equal_nan=True，并额外核对前两个零的 signbit。

这里验证的是脚本中给定值的转换结果，包括舍入、部分溢出/下溢样本和特殊值分类；不声明 NaN payload 保留或所有舍入边界均等价，也不把 SIMT 结论用于 T.simd.vcvt 的任意模式。

## 广播与布局

FP32/FP16/BF16 各验证 X=(3,17)、B=(17,) 的 Y=(X+B)ᵀ，输出 (17,3)。UB 为 X 分配 (3,32)、输出分配 (17,16)，copy 显式限定有效 region；转置与列广播在 SIMT 中表达。

输入由小整数除以 4、bias 除以 8 构造，三种类型均精确对照。不是任意 stride Tensor 的支持声明，也未使用 T.copy(transpose=True) 做 UB 内转置。两个部分合计六项通过。

适用环境与启动方法见 [核验来源](../references/evidence.md)。此条目的正确性结论限定于所列用例，不包含真机性能。

---
id: PL-950-simd-reduce
kind: pattern
status: verified
origin_task: 20261119-ascend950-knowledge-extended
toolchain: TileLang ascend 0.1.15; CANN 9.2.0 SDK / 9.1.0 simulator; 2026-11-19
toolchain_stamp: fad39935f627e0ea6f2dca0d681f3682cf53972ef4bb0d74193fc820c1362d3d
toolchain_id: ascend
hardware_id: ascend-950dt-9582
repro: repro/PL-950-simd-reduce.py
---

# 显式 SIMD 寄存器行归约

FP32 输入为 (2,64)/(2,63)，值由 -3 到 -1 均匀生成。GM 搬入 UB 后，在 T.SimdVF 内逐行执行 vld、vcadd/vcmax，并用 PAT_VL1 mask 仅写统计结果的一个 lane。

每行计算使用 update_mask(N,32)，N=63 时排除第 64 个 lane，不让 padding 参与 sum/max。统计量 UB 分配为 (2,8)，每行相隔 32 字节；vsts 使用 extent=1，搬出时逐行复制该有效元素。

两个 shape 均通过：sum 的 atol=1e-4、rtol=1e-5，max 为精确比较。本例不使用高层 T.reduce_sum/T.reduce_max；当前高层二维 shared→shared 路径的差异见 TRAP-950-simd-shared-reduce。

结论只覆盖 FP32、上述 mask 和布局。不覆盖其他 dtype、其他归约维度、NaN 或跨核合并。

适用环境与启动方法见 [核验来源](../references/evidence.md)。此条目的正确性结论限定于所列用例，不包含真机性能。

---
id: PL-950-softmax-boundaries
kind: pattern
status: verified
origin_task: 20261119-ascend950-knowledge-extended
toolchain: TileLang ascend 0.1.15; CANN 9.2.0 SDK / 9.1.0 simulator; 2026-11-19
toolchain_stamp: fad39935f627e0ea6f2dca0d681f3682cf53972ef4bb0d74193fc820c1362d3d
toolchain_id: ascend
hardware_id: ascend-950dt-9582
repro: repro/PL-950-softmax-boundaries.py
---

# FP32 softmax 的有限极值与非有限输入

单行 64 元素、SIMT FP32 统计，采用减 max、exp、sum、除法的完整设备计算。一个已编译 kernel 依次处理五组输入：
- 从 -1000 到 1000 的有限值。
- 全部为 -10000 的常数行。
- 全部为负无穷。
- 一个正无穷，其余为零。
- 一个 NaN，其余为零。

与 torch.softmax 比较，atol=1e-6、rtol=1e-5、equal_nan=True。非有限输入的结果分类以脚本断言及实际日志为准；该条目只在全部五组通过后入库。

不覆盖 NaN payload、混合精度、所有下溢范围或其他编译精度选项，也不将这些结果推广到原始 SIMD SFU 指令。

适用环境与启动方法见 [核验来源](../references/evidence.md)。此条目的正确性结论限定于所列用例，不包含真机性能。

---
id: TRAP-950-simd-shared-reduce
kind: trap
status: verified
origin_task: 20261119-ascend950-knowledge-extended
toolchain: TileLang ascend 0.1.15; CANN 9.2.0 SDK / 9.1.0 simulator; 2026-11-19
toolchain_stamp: fad39935f627e0ea6f2dca0d681f3682cf53972ef4bb0d74193fc820c1362d3d
toolchain_id: ascend
hardware_id: ascend-950dt-9582
repro: repro/TRAP-950-simd-shared-reduce.py
---

# 高层 SIMD 二维 shared→shared 归约的输出差异

在锁定工具链和 DT 仿真器上，FP32 UB 输入 (2,64)，输出 UB 为 (2,)，在同一个 T.SimdVF 内依次调用 T.reduce_sum(x,sm,dim=1) 和 T.reduce_max(x,mx,dim=1)，随后分别拷出两个统计量。该写法能编译，但 sum 的第二行与 CPU 参考不一致。

确认输入有两组：128 个 [-3,-1] 等距数 reshape 为 (2,64)，以及 (2,64) 的全 -2。自包含脚本断言这一差异仍存在，并打印 actual/expected；这是已确认的问题证据，不是可用 kernel 示例。

初次失败对应的生成代码中，sum 写入 float 偏移 128+64*row，而 GM 拷出从 128 开始的连续两个 float。写入布局与读取布局不同；max 路径也出现 136+64*row 的生成写入。该观察定位到当前 wheel 的生成代码，没有修改编译器或 Agent 主流程。

范围限于上述二维 FP32 高层路径，不据此认定所有 SIMD 归约均不支持。可用参考为 PL-950-simd-reduce 的显式寄存器归约，或 PL-950-row-primitives 的 SIMT reducer 路径，二者分别经过精度验证。升级工具链后需重新检查，修复后本反例脚本的预期差异断言应失败。

适用环境与启动方法见 [核验来源](../references/evidence.md)。此条目的正确性结论限定于所列用例，不包含真机性能。
