---
id: PL-950-normalization
kind: pattern
status: verified
origin_task: 20261119-ascend950-knowledge-extended
toolchain: TileLang ascend 0.1.15; CANN 9.2.0 SDK / 9.1.0 simulator; 2026-11-19
toolchain_stamp: fad39935f627e0ea6f2dca0d681f3682cf53972ef4bb0d74193fc820c1362d3d
toolchain_id: ascend
hardware_id: ascend-950dt-9582
repro: repro/PL-950-normalization.py
---

# 带权重的 LayerNorm 与 RMSNorm

验证形状为 (2,63)，输入、weight、bias、输出分别覆盖 FP32/FP16/BF16；中间统计量为 FP32。每行一个核，SIMT 线程数 64，UB 以 64 元素容纳 63 个有效元素。随机种子 1；第一行随机输入，第二行为常数 2，eps=1e-5。

LayerNorm 计算 mean，再归约 (x-mean)²，输出 (x-mean)/sqrt(var+eps)*weight+bias。RMSNorm 输出 x/sqrt(mean(x²)+eps)*weight，未使用 bias。参考函数对量化后的输入/参数转 FP32 计算，再转换为输出 dtype。

六项通过。FP32 阈值 atol=rtol=1e-5，FP16 为 2e-3，BF16 为 2e-2。常数行覆盖了 LayerNorm 方差为零时 eps 和 bias 的路径。

代码的尾块将无效 lane 的统计贡献设为 0，统计分母使用 63。输出仅写有效元素。本条目是有 affine 参数的端到端设备实现，未覆盖多轴归一化、大行分块、特殊值或所有低精度舍入边界。

适用环境与启动方法见 [核验来源](../references/evidence.md)。此条目的正确性结论限定于所列用例，不包含真机性能。

---
id: PL-950-attention
kind: pattern
status: verified
origin_task: 20261119-ascend950-knowledge-extended
toolchain: TileLang ascend 0.1.15; CANN 9.2.0 SDK / 9.1.0 simulator; 2026-11-19
toolchain_stamp: fad39935f627e0ea6f2dca0d681f3682cf53972ef4bb0d74193fc820c1362d3d
toolchain_id: ascend
hardware_id: ascend-950dt-9582
repro: repro/PL-950-attention.py
---

# 单头 dense attention 的 SIMT 正确性参考

Q=(3,16)、K/V=(5,16)，FP32/FP16/BF16 输入，FP32 输出；分别验证无 mask 和上三角 causal mask，共六项。参考为量化输入转 FP32 后的 PyTorch scaled_dot_product_attention，dropout=0，默认缩放 1/sqrt(16)。

每个 query 行使用一个核，SIMT 线程数 32；score 使用长度 8 的 fragment 容纳 5 个 key。计算完整执行 QKᵀ、缩放、减 max 的 softmax、PV。无效 key 或 causal 屏蔽位置在 max 前置为负无穷。因果条件为 key_index<=query_index，覆盖非方形 3×5 的上左对齐 mask。

阈值 atol=rtol=1e-4。此例从 Q/K/V 到输出全在一个设备 kernel 内完成，使用 SIMT 标量表达式与归约，没有调用主机 softmax 或矩阵乘替代设备计算。

这是一条便于核对语义的实现，不是 Cube 优化版或 FlashAttention。未覆盖多 batch/head、GQA、KV cache、任意 mask、dropout、全 mask 行、低精度输出和性能收益。

适用环境与启动方法见 [核验来源](../references/evidence.md)。此条目的正确性结论限定于所列用例，不包含真机性能。
