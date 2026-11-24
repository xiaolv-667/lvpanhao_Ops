# ascend 算法候选

本页分别列出已通过仿真的最小实现和已核对的 CPU 数学参考。适用范围按条目给定，不将数学参考标为 950 实现。

## 已有最小实现

| 算子族 | 当前实现候选 | 证据 |
|---|---|---|
| elementwise / broadcast | 输入转 FP32，行标量广播后逐元素计算 | [PL-950-row-primitives](PL-950-row-primitives.md)，`x+max(x)` |
| reduction | 单行 SIMT sum/max reducer | [PL-950-row-primitives](PL-950-row-primitives.md) |
| softmax | 行驻留，减 max、exp、sum、归一化 | [PL-950-row-primitives](PL-950-row-primitives.md) |
| RMSNorm / LayerNorm | FP32 统计量及 affine 参数，输出转回输入 dtype | [PL-950-normalization](operator-patterns.md)，(2,63)、三种 dtype |
| GEMM | 显式 L1/L0 操作数、输入布局转换和 K 分块累加 | [PL-950-gemm-tiles](gemm-tiles.md)，含 M/N/K 尾块 |
| attention | 单头 SIMT QKᵀ、softmax、PV，含 causal 分支 | [PL-950-attention](operator-patterns.md)，Q/K/V→输出单 kernel |
| GEMM+ReLU | Cube 与双 AIV 的混合计算 | [PL-950-mixed-epilogue](execution-patterns.md) |

数学参考见 [ALG-950-formulas](ALG-950-formulas.md)：softmax、LayerNorm、RMSNorm 和 dense attention 的公式对照。该条目只认证 CPU 数学参考，其实现覆盖范围与上述仿真条目分别说明。
