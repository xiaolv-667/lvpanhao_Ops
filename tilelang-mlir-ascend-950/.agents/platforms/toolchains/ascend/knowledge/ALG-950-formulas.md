---
id: ALG-950-formulas
kind: pattern
status: verified
origin_task: 20261119-ascend950-knowledge
toolchain: PyTorch 2.13.0+cpu formula checks; TileLang ascend profile; 2026-11-19
toolchain_stamp: fad39935f627e0ea6f2dca0d681f3682cf53972ef4bb0d74193fc820c1362d3d
toolchain_id: ascend
hardware_id: ascend-950dt-9582
repro: repro/ALG-950-formulas.py
family: [softmax, layernorm, rmsnorm, attention]
---

## CPU 数学参考，不是 950 kernel 认证

下列公式用 CPU FP64、固定随机种子与 PyTorch 对应函数逐项对照。行 shape 为 (2,8)/(2,17)，attention 的 Q/K/V shape 为 (1,1,4,8)/(1,1,4,17)；共 8 项，`atol=rtol=1e-12`。

| 算子族 | 已核对的数学表达 | 实现所需计算 |
|---|---|---|
| softmax | `m=max(x); e=exp(x-m); y=e/sum(e)` | max、exp、sum、广播与除法 |
| LayerNorm | `mu=mean(x); v=mean((x-mu)^2); y=(x-mu)/sqrt(v+eps)` | 两次统计量归约及逐元素计算 |
| RMSNorm | `y=x/sqrt(mean(x^2)+eps)` | 平方和归约及逐元素计算 |
| dense attention | `S=QK^T/sqrt(d); P=softmax(S); O=PV` | 矩阵乘、行 softmax、矩阵乘 |

Norm 的 `eps=1e-5`，无 affine weight/bias；attention 无 mask、无 dropout、Q/K/V 的序列长度和 head 维度相同。测试不覆盖所有浮点输入，也未认证 LayerNorm 或 attention 的 950 端到端实现。

对于长度 N 的一行，驻留输入/中间量的存储规模为 O(N)，统计量为 O(1)。上述显式 attention 参考物化 S/P，矩阵元素数各为 query_length × key_length；不能把它称为 online/FlashAttention。

数学表达可作为设计候选；具体切分、布局、同步和精度按当前工具链实现并验证。当前没有本平台在线 softmax、Welford、分块 attention 或其性能取舍的已验证条目。
