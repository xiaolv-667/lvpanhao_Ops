# ascend 编程与资源规则

适用：Ascend950DT_9582，项目锁定的 TileLang ascend 0.1.15、CANN 9.2.0 编译 SDK 与 9.1.0 runtime/仿真器。核验来源和指纹见 [evidence.md](evidence.md)。

本页把当前安装包的接口事实与实际执行证据分开。接口签名或源码分支存在，不代表任意 shape、dtype 或接口组合均已验证。

## API 与执行区域

使用 `import tilelang.ascend.language as T`，通过 `tilelang.compile(..., target="ascend", out_idx=...)` 编译。当前任务平台提供编译环境和执行命令。

| 接口 | 已核对的规则 | 证据 |
|---|---|---|
| `T.Kernel(N)` | 一维核网格；签名没有 `threads`，多维网格在实现中拒绝 | `ascend/language/kernel.py:Kernel`；`threads` 参数拒绝已检查 |
| `T.SimtVF(threads=128)` | 线程域位于 VF 内；参数支持正整数或最多三维的序列 | `frame.py:SimtVF`；零线程及四维参数拒绝已检查 |
| `T.SimdVF()` | SIMD 执行区域；`T.simd.*` 提供寄存器级操作 | `frame.py:SimdVF`、`simd.py`；已有向量条目覆盖 FP32 abs |
| `T.Cube()`、`T.Vector(vector=2)` | 源码分别定义 AIC/AIV 区域；Vector frame 提供 subblock ID | `frame.py`；单 tile 混合计算见 [执行模式](../knowledge/execution-patterns.md)，不推广到任意流水 |
| `T.MixedKernel(N, sids=2)` | 返回核索引和 subblock ID；实现限定一维网格、`sids` 为 1 或 2 | `kernel.py:MixedKernel`；属于源码确认 |

SIMT 行处理的已验证用法见 [PL-950-row-primitives](../knowledge/PL-950-row-primitives.md)。该条目使用 `T.cast`、加减乘除、`T.exp`、`T.sqrt`、reducer 和行标量广播，记录了具体输入类型及数值范围。

当前 `simd.py` 导出 `pset`、`vld`、`vsts`、`vcvt`、`vadd`、`vexp` 等接口。`vcvt` 的 dtype/舍入/饱和参数与 SIMD load/store 分布参数需要按具体路径核对；本次没有把其注释中的类型列表登记为完整的执行支持矩阵。

## 分配、布局和搬运

| 资源 | 分配接口 | 默认 scope |
|---|---|---|
| UB | `T.alloc_shared(shape, dtype)` | `shared.dyn` |
| L1 | `T.alloc_l1(shape, dtype)` | `shared.l1` |
| L0A | `T.alloc_l0a(shape, dtype)` | `shared.l0a` |
| L0B | `T.alloc_l0b(shape, dtype)` | `shared.l0b` |
| L0C | `T.alloc_l0c(shape, dtype)` | `shared.l0c` |

来源为 `allocate.py`。接口名中的 `l1/l0a/l0b/l0c` 为小写。L0A/B 的布局留给消费它们的 GEMM 推导；L0C 的默认 `layout=True` 会为至少二维的缓冲注解累加器布局。

`copy_op.py:copy` 中已确认：

- 整个 Buffer 对整个 Buffer 的拷贝要求元素总数相等，形状不要求逐维相同；元素数不同的调用会报错。局部拷贝用显式 region 表达。
- `transpose` 的文档作用域为 GM→L1 的 DN→NZ 路径，不能据此把它当成任意内存间的通用转置。
- `pad_value` 与 `data_select` 互斥，源码将其用于 GM→UB 的右侧 padding；本次未验证这两个选项的执行行为。已有尾块脚本使用显式有效区间，不依赖它们。
- `dual_copy` 定义的是混合 kernel 的分片搬运。源码列出 L0C→UB、GM→UB、UB→GM、UB→L1；分片方向由 shape 推导。L0C→UB 源目标 dtype 不同时，前端明确拒绝，已检查该拒绝行为。这里不把这些路径登记为本次均已运行。

执行证据还覆盖 [GEMM 分块和尾块](../knowledge/gemm-tiles.md)、[SIMT 转换与广播转置](../knowledge/numerical-boundaries.md) 和 [L0C→双 AIV UB→GM](../knowledge/execution-patterns.md)。具体 shape、dtype 和路径分别以条目为准。

## GEMM 与归约

`gemm_op.py:gemm` 的 `clear_accum` 默认是 `False`。计算全新的 C 时，本次单 tile 示例显式传 `clear_accum=True`；重复累加需要按算法决定初始化点。L0 输入路径检查 M/N/K 相容性，并要求分配 tile 的 M/N/K 为静态值。布局、padding 和 region 限制以该函数及编译结果为准，不从型号 INI 的一个分形值推导全部 dtype 的分块约束。

`reduce_op.py` 为 `reduce_sum/max/min` 提供 ascend 分派，默认 `dim=-1`、`clear=True`。SIMD VF 中 shared→shared 的归约有独立分支。该高层二维分支发现了 [输出布局差异](../knowledge/numerical-boundaries.md)；显式 SIMD 寄存器归约另有两个 shape 通过。已验证的 SIMT 路径为：
`alloc_reducer → reducer_init → reducer_update → finalize_reducer`。两种路径的执行证据不能互相替代。

## 同步与缓冲版本

`ascend/pipeline.py` 的 `allow_autoschedule` 默认开启。开启时流水线调用 AutoSchedule、InsertSync 和 MaterializeMultiBuffer 等 pass。除显式关闭自动调度的单核管线同步示例外，当前用例使用默认流水线；未验证手工跨核 flag 配对。

`sync.py` 提供以下显式接口：

| 类别 | 接口 | 源码中的参数含义 |
|---|---|---|
| 管线屏障 | `ascend_pipe_barrier(pipe_t)` | 管线名称字符串 |
| 管线事件 | `ascend_set_flag(hard_event, event_id)` / `ascend_wait_flag(...)` | 事件名称和 ID |
| 跨核事件 | `ascend_cross_core_set_flag(mode_id, pipe, flag_id)` / `ascend_cross_core_wait_flag(...)` | 模式、管线和事件 ID |
| 缓冲锁 | `ascend_get_buf(pipe, buf_id, mode=False)` / `ascend_rls_buf(...)` | 管线和缓冲槽 |

[PL-950-manual-sync](../knowledge/execution-patterns.md) 提供已执行的单核单 tile 管线配对；跨核 flag 和缓冲锁仍只有接口说明，不据此推导其使用模板或可用 flag 数量。NPUIR 的 flag 预算和同步例子不作为此组合的依据。

`annotations.py` 中已确认：

- `annotate_buffer_versions({buf: N})` 要求正整数版本数；零版本的拒绝已检查。版本模式为 `auto/iteration/counter`。
- `{buf: 1}` 在实现约定中取消该 storage 的自动多缓冲资格；`(1, "auto")` 与其语义不同。
- `annotate_manual_multi_buffer` 声明用户已经手动安排的版本，不负责扩展分配；访问版本索引仍由代码表达。
- `annotate_unlimit_memory` 解除自动调度的容量约束，不增加型号的物理存储容量。

两阶段 GEMM、单 tile 混合计算、单核手工管线配对和两核原子 split-K 分别已有正确性条目。手工跨核 flag、循环复用下的复杂同步和跨 block batch barrier 仍未认证；任何流水收益都需要真机测量。
