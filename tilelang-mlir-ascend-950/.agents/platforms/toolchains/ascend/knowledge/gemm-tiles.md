---
id: PL-950-gemm-tiles
kind: pattern
status: verified
origin_task: 20261119-ascend950-knowledge-extended
toolchain: TileLang ascend 0.1.15; CANN 9.2.0 SDK / 9.1.0 simulator; 2026-11-19
toolchain_stamp: fad39935f627e0ea6f2dca0d681f3682cf53972ef4bb0d74193fc820c1362d3d
toolchain_id: ascend
hardware_id: ascend-950dt-9582
repro: repro/PL-950-gemm-tiles.py
---

# GEMM 输入布局、K 分块与尾块

FP16/BF16 输入、FP32 输出，分别验证 B 按 (K,N) 和 (N,K) 存放。每种组合运行两组 M/N/K：
- (16,16,64)：K 切成两个 32 元素块，单阶段配置。
- (17,19,35)：M/N 分成 16×16 tile，K 切成 32+3；两阶段配置和显式两版本 L1/L0 操作数。

共八项，阈值 atol=rtol=1e-4，参考为量化输入转 FP32 后的矩阵乘。随机种子 2。

B=(N,K) 时按 region 搬入；B=(K,N) 时 GM→L1 copy 使用 transpose=True，使后续 B 缓冲保持 (N,K) 布局。L0 GEMM 使用 transpose_B=True。A 保持 (M,K)，本例没有测试 transpose_A=True。

每次 copy 和 gemm 的 region 使用当前有效 M/N/K；物理分配仍按 16×32 操作数与 16×16 L0C。clear_accum 仅在第一个 K 块为 True；输出只写当前有效 M/N。跨 K 块保留累加器，避免每块清零丢失前面的部分和。

两阶段配置通过正确性验证，不代表发生了理想重叠或性能提升。配置参数与最终调度结果是不同证据。本条目不覆盖量化 GEMM、FP32 Cube 输入及所有分块组合。

适用环境与启动方法见 [核验来源](../references/evidence.md)。此条目的正确性结论限定于所列用例，不包含真机性能。

---
id: PL-950-gemm-l1-fill
kind: pattern
status: verified
origin_task: shared_expert_mlp-SharedExpertMLPFwdOp-20261124T132107Z (Stage 4)
toolchain: TileLang ascend 0.1.15 + CANN 9.2.1, 2026-11-24
toolchain_stamp: 7a52f2f1274df07d8f4c9c01353ce9d290f88c16e85047de0c425b6a009de4aa
toolchain_id: ascend
hardware_id: ascend-950pr-9579
repro: repro/PL-950-gemm-l1-fill.py
---

# L1 操作数 GEMM 的填充带宽判据：先加长 block_k，再把 tile 顶到 L0C 上限

判据：`PipeUtilization` 里 `aic_mte2_ratio ≈ 1` 而 `aic_cube_ratio ≈ 0.5` 时，GEMM 受
**GM→L1 填充**限制，此时每 K 步的每字节复用是

    reuse(FLOP/B) = 2·bm·bn·bk / ((bm+bn)·bk·2) = bm·bn/(bm+bn)

测得的效率上限 = `reuse × 聚合 L1 填充带宽`（本平台 ≈ 2.7–2.9 TB/s，见
[CONST-950-gemm-mte2-burst](constants.md)）。128×256 → 85.3 FLOP/B → ~196 TF/s ≈ 52% MFU，
与 Stage 3 基线实测 50.1% 吻合。

**两把杠杆，顺序重要**：

1. `block_k` 64 → 128（每行 DMA 段 128 B → 256 B）：单核 MTE2 有效带宽 70 → 101–105 GB/s，
   kimi-k2-t512 gate/up 1201.6 → 811.8 µs（−32%）。L1 预算
   `(bm+bn)·bk·2·ns ≤ 512 KiB` 要求 `ns ≤ 3`（bk=128、256×256）。再抬到 bk=256 需 ns=2，
   收益为 0（818.9 µs）。
2. tile 顶到 L0C 上限（`bm·bn·4 ≤ 256 KiB` → **256×256**，reuse 128 FLOP/B）：同样在
   bk=64 下只值 −13%（1382.9 → 1201.6 µs）。窄 N（256×128、128×256）无收益；128×512
   （reuse 102.4）在 bk=128 时与 256×256 基本并列（816.3 vs 811.8），是无 256×256 时
   （例如 `bm` 被 M 夹小）的次选。

**同族衍生形态**：把一个 tile 拆成两个 fp32 L0C 累加器（`bm·bn` 减半）可以做 Cube+Vector
epilogue（本例把 `silu(gate)*up` 融进 gate/up GEMM），reuse 只取决于 `1/bm + 1/(2·bn)`，
取 `bm=256, bn=128` 时与 256×256 持平（128 FLOP/B），并省掉独立激活 pass（t512 −1.9%、
decode 段 −11%）。前提：N tile 必须是 64 的倍数（epilogue 按 64 车道块遍历，无尾掩码），
且 `dual_copy` 要求 M/S tile 精确。反例见 repro 的 bn=96 拒绝断言。

边界：结论限于 bf16/fp16 NT（`transpose_B=True`）L1 操作数路径、`ascend_gemm_l1` 模板、
本工具链；未覆盖 fp32 输入、L0 操作数路径与 `T.Persistent`。
