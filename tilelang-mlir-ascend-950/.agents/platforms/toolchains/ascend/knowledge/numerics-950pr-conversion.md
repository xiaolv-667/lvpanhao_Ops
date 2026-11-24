# 950PR 转换/舍入语义实测（batch2）

本文件承载 ascend 工具链在 950PR 真机上实测的 dtype 转换、饱和/NaN 语义与等价改写数值代价。
条目只陈述实测值与适用条件；复现脚本在 `repro/`（自包含，头部记录条目 id 与首验版本戳）。

---
id: CONST-950-vcvt-rs-disable-ieee
kind: trap
status: verified
origin_task: batch2-20261123 (mla_dense_bwd / _dq_land_kernel Stage 3)
toolchain: TileLang ascend 0.1.15 + CANN 9.2.1, 2026-11-23
toolchain_stamp: a010a95909e4e752d0d0b602885eca4b6fab233edca2d3e540248ed71c2135dc
toolchain_id: ascend
hardware_id: ascend-950pr-9579
repro: repro/CONST-950-vcvt-rs-disable-ieee.py
provenance: 溯源见 origin_task；两版设备码 `RS_DISABLE`/`RS_ENABLE` 对照与角点表记录在 mla_dense_bwd/_dq_land_kernel/RETROSPECTIVE.md Stage 3
apis: [T.simd.vcvt]
dtype: [float32, float16]
---

# `vcvt` 的 `sat` 标志在 f32→f16 上不改变结果，且默认是 IEEE（不是 wrap）

实测 f32→f16（真机，角点表）：

| 输入 | 输出 |
|---|---|
| 65504 | 65504 |
| 65519 | 65504 |
| 65520 | ±inf |
| 1e5 | ±inf |

`sat=False`（生成 `RS_DISABLE`）与 `sat=True`（生成 `RS_ENABLE`）逐项相同。安装包 docstring 对
`sat` 的措辞是 "out-of-range values wrap"，与实测不符：本路径是 IEEE 语义（>65520 溢出为 inf），
不是回绕。

后果：不能靠 `sat=True` 实现「钳到最大有限值」，也不能因 docstring 写 wrap 就假设关饱和会回绕；
需要饱和时用 `T.clamp`（见 CONST-950-fp8-nan-sign）。

边界：结论限于 f32→f16 的窄化转换与该工具链版本；bf16 一侧见
CONST-950-vcvt-sat-false-overflow-inf（同为 IEEE inf）。

---
id: CONST-950-vcvt-sat-false-overflow-inf
kind: constant
status: verified
origin_task: batch2-20261123 (shared_expert_mlp / _silu_mul_fused_kernel Stage 3)
toolchain: TileLang ascend 0.1.15 + CANN 9.2.1, 2026-11-23
toolchain_stamp: a010a95909e4e752d0d0b602885eca4b6fab233edca2d3e540248ed71c2135dc
toolchain_id: ascend
hardware_id: ascend-950pr-9579
repro: repro/CONST-950-vcvt-sat-false-overflow-inf.py
provenance: 溯源见 origin_task；`[float16] bit-exact 128/128`、`[bfloat16] bit-exact 128/128`、wrapped-to-finite=0 记录在 shared_expert_mlp/_silu_mul_fused_kernel/RETROSPECTIVE.md Stage 3
apis: [T.simd.vcvt, T.cast]
dtype: [float32, float16, bfloat16]
---

# `vcvt(sat=False)` 的溢出是 IEEE ±inf：128/128 位与 `Tensor.to(dtype)` 一致

f32→f16 与 f32→bf16 两条路径，128 个角点样本与 torch `Tensor.to(dtype)` **逐位一致**
（`wrapped-to-finite = 0`），样本含 65519→65504、65520→inf、1e30、±inf、NaN。因此 golden 侧直接
`(x.float() ...).to(dtype)` 与设备等价，不需要为「wrap 语义」写特例。

配套判据：只有 `|y| > dtype max` 的用例能区分 inf 与 wrap（随机输入永远触发不到）；凡
`y = 激活(x) * 放大系数` 的 fp16 kernel 应有这类具名 boundary 用例（本 kernel B-10/B-11
diff = 0）。`T.cast` 不饱和（±inf/1000 → NaN），饱和需显式 `T.clamp`。

边界：结论限于 f32→16 位的 `vcvt`；不覆盖整型窄化或 e4m3/e5m2。

---
id: CONST-950-fp8-nan-sign
kind: constant
status: verified
origin_task: batch2-20261123 (paged_cache_write / _paged_cache_write_kernel Stage 3)
toolchain: TileLang ascend 0.1.15 + CANN 9.2.1, 2026-11-23
toolchain_stamp: a010a95909e4e752d0d0b602885eca4b6fab233edca2d3e540248ed71c2135dc
toolchain_id: ascend
hardware_id: ascend-950pr-9579
repro: repro/CONST-950-fp8-nan-sign.py
provenance: 溯源见 origin_task；L0-5 / B-6 的 NaN 字节对照与 `_nan_canonical_bytes` 记录在 paged_cache_write/_paged_cache_write_kernel/RETROSPECTIVE.md Stage 3
apis: [T.clamp, T.min, T.max, T.cast, T.if_then_else]
dtype: [bfloat16, float16, fp8_e4m3]
---

# e4m3 的 NaN 符号位被规范化；饱和必须用 clamp 而非 min/max 组合

设备转换把 ±nan 统一写 `0x7F`，而 torch 保留输入符号位（`-nan → 0xFF`）；逐字节比对须按
`equal_nan` 比较并规范化 NaN 符号位，其余字节（±inf 饱和、ties、±0）完全一致。

饱和/NaN 传播的三条实测：

- `T.clamp` 传播 NaN（`+nan`/`-nan` 均写 `0x7F`）且做 ±448 饱和；
- `T.min`/`T.max` 组合把 NaN 变成 `-448`（丢 NaN）；
- 仅 `T.cast` 不饱和：bf16 输入中 4/16 个超出 ±448 的元素未被饱和（fp16 为 2/16），
  `±inf/1000 → NaN`。

可用形态：fp8 存前显式 `clamp(±448)`，NaN 用 `T.if_then_else(d == d, clamped, d)` 保留。

边界：结论限于 e4m3 与 950PR 的转换路径；NaN payload 不保留（任意 NaN → `0x7F`），不覆盖 e5m2。

---
id: CONST-950-silu-mul-a1-one-ulp
kind: constant
status: verified
origin_task: batch2-20261123 (shared_expert_mlp / _silu_mul_fused_kernel Stage 3)
toolchain: TileLang ascend 0.1.15 + CANN 9.2.1, 2026-11-23
toolchain_stamp: a010a95909e4e752d0d0b602885eca4b6fab233edca2d3e540248ed71c2135dc
toolchain_id: ascend
hardware_id: ascend-950pr-9579
repro: repro/CONST-950-silu-mul-a1-one-ulp.py
provenance: 溯源见 origin_task；`exact 4096/4096, max_ulp=0`（×2 dtype）与 8.4M 元素上 1 ulp 记录在 shared_expert_mlp/_silu_mul_fused_kernel/RETROSPECTIVE.md Stage 3
apis: [T.simd.vdiv, T.simd.vexp, T.simd.vmul, T.simd.vsub]
dtype: [float16, bfloat16, float32]
---

# gated-activation 的 5 算符等价式：(g·u)/(1+exp(−g)) 设备侧 0 ulp

`(g·u)/(1+e^{−g})`（5 条算符：vneg+vexp+vadds+vmul+vdiv）替代基线 `g·σ(g)·u`（6 条）的实测代价：

- 4096 元素 / dtype：与 CPU golden **逐位一致**，`max_ulp = 0`（fp16 与 bf16 各一档）；
- 8.4M 元素的真实 workload（kimi-k2-t512）：最多 1 个 16-bit ulp（`max_ulp = 1`，
  `frac_gt1ulp = 0`，绝对差 3.125e-02 = bf16 1 ulp @ 4.0–8.0）；
- fp32 输出（跳过 vcvt）允许 2 个 fp32 ulp（4.77e-07）：`silu` 的除法改写只保证输出 16-bit 网格
  0 ulp。

配套事实：本平台 `T.simd` 无倒数 intrinsic（`vrec/vrecs` 不存在），`1/x` 只能走 `vdiv`；
`vdiv(precision=None)` 的精度由 `tl.enable_fast_math` 决定（关 → `PRECISION_0ULP_FTZ_TRUE`
多指令精化，开 → 裸 SFU ~2 ulp ftz）；`T.simd` 无 `vexp2`（有 `vexpdif`，可再省 1 条 `vneg`）。
算符计数须按 CCE 实际指令数（`exp(−g)` 是 `vneg+vexp` 两条）。

反例：`(g·u)·e^{g}/(1+e^{g})` 在 `g > 88.7` 时 fp32 溢出为 NaN（bf16 实测 647 ulp / 5 处非有限值
分类不符），说明「数学恒等」不等于 fp32 域等价。

边界：结论限于逐元素 gated activation 与所列 dtype；不覆盖 tensor-wise 归约或 softmax 主力路径。
