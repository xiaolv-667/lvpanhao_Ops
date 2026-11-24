# ascend 迁移与诊断参考

适用版本和来源见 [evidence.md](evidence.md)。以下资料接入现有生成、迁移、精度与调优阶段，不改变阶段或产物契约。

## 从源实现映射到 ascend

| 源实现中的计算结构 | 当前可用的表达或证据 | 保留的边界 |
|---|---|---|
| block 网格 | `T.Kernel(N)` 表达一维核网格 | 多维索引按源语义展开；GPU 的 `threads` 不能原样传到该接口 |
| 线程级逐元素计算 | `T.SimtVF` 内的 `T.Parallel` 和表达式 | 行原语脚本验证了有限长度、类型和运算，不代表所有表达式 |
| 显式向量操作 | `T.SimdVF` 与 `T.simd.*` | 已有 abs 条目验证固定 SIMD 宽度和尾块；寄存器 mask/distribution 按 API 核对 |
| 临时存储 | UB/L1/L0A/L0B/L0C 分配接口 | GPU shared memory 不能整体机械替换为某一个 NPU 层级 |
| sum/max 与行标量广播 | SIMT reducer 和 `finalize_reducer` 产物的读取 | 已验证行归约与广播；两核求和另有 [原子 split-K](../knowledge/execution-patterns.md)，不等同于全局 barrier |
| 矩阵乘 | 显式 L1→L0A/B、`T.gemm`、L0C 输出 | FP16/BF16 的输入布局转换、K 分块和 M/N/K 尾块已有 [GEMM 用例](../knowledge/gemm-tiles.md) |
| 尾块 | 源/目标均使用有效 region；计算显式处理无效 lane | 无效 lane 的值取决于算法：sum 用 0，max 用负无穷；不能让 padding 参与有效归一化 |
| barrier、异步搬运、跨核协作 | 由 ascend 同步和编译调度规则重新表达 | API 名称相似不构成语义等价证据，不能照搬 GPU 或 NPUIR 同步代码 |

数学公式、实现证据和未覆盖范围见 [算法候选](../knowledge/algorithm-candidates.md)。迁移继续保持源算子的 shape、dtype、返回值、误差标准和边界语义。

## 诊断顺序与证据

1. **环境及导入**：现有 `statectl platform --hardware ascend-950dt-9582 --toolchain ascend --check-environment` 检查锁定环境。静态查询从任务目录运行，记录 `T.__file__`，确认导入的是已安装的 `tilelang.ascend.language`。工具链版本名相同不能替代文件指纹。
2. **Python/API 构造**：先核对当前签名和 region。已检查的错误包括 `T.Kernel(..., threads=...)`、SIMT 非正线程数/四维线程参数、整个 Buffer 元素数不等的 copy、L0C dual-copy 改变 dtype、缓冲版本数为零。这些失败发生在相应的参数或 IR 构造检查中，尚未证明 kernel 执行失败。
3. **编译**：保存完整异常、输入 shape/dtype、对应源码和生成代码。`pipeline.py` 给出了布局推导、VFChecker、调度、同步、搬运 lowering 等阶段顺序；根据实际堆栈确定失败阶段。接口存在不能代替编译证据。
4. **启动与运行**：读取任务平台的 `execution`，在任务目录按该命令启动仿真；模型进程串行运行。保留启动日志、SoC 配置、退出码和用户脚本输出。仿真器打印配置诊断行不等同于算子正确或错误，最终看进程结果和逐用例断言。
5. **精度**：参考实现使用实际输入量化后的值。FP16/BF16 输入转成 FP32 计算的示例，其参考也先量化输入再转 FP32；不能与未量化原始输入比较。误差阈值、输入范围、shape 和特殊值覆盖分别记录。
6. **性能**：只有完成精度校验后才进入现有 A/B 流程。仿真结果不能填成真机带宽、成本常数或优化收益。

## 尚未形成结论的范围

本次补充已覆盖具体的转换/特殊值、混合计算、原子跨核求和和单核手工同步用例；范围分别见 [数值边界](../knowledge/numerical-boundaries.md)、[执行模式](../knowledge/execution-patterns.md) 和 [完整算子参考](../knowledge/operator-patterns.md)。任意 dtype/API 组合、手工跨核 flag、复杂循环同步和跨 block batch barrier 仍未认证。能力缺口仍使用现有 [capability-gaps.md](../knowledge/capability-gaps.md) 记录。
