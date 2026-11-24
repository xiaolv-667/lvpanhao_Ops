# 第二批算子迁移与优化说明（11 个算子）

更新：2026-11-23。

| 项 | 值 |
|---|---|
| 工作仓库 | `xiangmu/TileOPs-ascend950`（`yyttt6/TileOPs` 分支 `ascend950`，HEAD `2362d1e9`，2026-11-09） |
| GPU 参考仓库 | `xiangmu/TileOPs`（`tile-ai/TileOPs` 分支 `main`，HEAD `2485b53`，2026-11-20） |
| 环境口径 | CANN 9.2.0 编译 / 9.1.0 runtime、tilelang 0.1.15（含 `tilelang.ascend`）、torch 2.13.0+cpu / torch_npu 2.13.0rc1、Python 3.12、隔离 venv `xiangmu/.venvs/ascend950` |
| 历史文档 | 上一批（DSA / MHC / GQA / FusedTopK / FFT / NSA / MoE PrePermute / GLA / GDN / SSD ChunkScan / SamplingFromProbs）的计划与评估已按你的要求删除，两批算子无重叠，但共用同一套环境、卡和验收口径 |

> 注：上一批文档（`MIGRATION_PLAN.md`、`correctness-20261121.md`、`ops/*.md`）已删除，本文件不再引用它们；本批的流程与可行性另见同目录 `MIGRATION_LOOP.md` 与 `FEASIBILITY.md`。

## 0. 结论摘要

| 状态 | 个数 | 算子 |
|---|---:|---|
| ✅ 本分支已有 Ascend 实现（在 HEAD `2362d1e9` 里） | 6 | EngramGateConvFwdOp、EngramGateConvBwdOp、EngramDecodeFwdOp、RoPEFwdOp、SSDChunkCouplingFwdOp、ChainSpeculativeSamplingFwdOp |
| ⬜ 有 spec + Op + 接口，缺 Ascend kernel（需新写） | 3 | MoEPermuteAlignFwdOp、SharedExpertMLPFwdOp、PagedKVCacheWriteFwdOp |
| ❌ 本分支完全缺失（需先从 main 移植，再写 kernel） | 2 | MLADenseBwdOp、MLAVarlenBwdOp |

三条工作路线由此分开：**A 验收调优（6 个）/ B 写 kernel（3 个）/ C 先移植再写（2 个）**，见 §3。

> 11 个算子**在上游 `TileOPs` main 里全部存在**（spec + Python Op + GPU kernel + workload + 测试齐全），所以本批不存在"仓库里没有这个算子"的问题，问题只在分支同步度和 Ascend 实现。

## 1. 总览表

行号均为分支内文件行号；标 `main:` 的取自 GPU 参考仓库。

| # | 算子 | family | manifest spec | GPU 实现（分支内） | Ascend 实现 | 状态 | 下一步 |
|---:|---|---|---|---|---|---|---|
| 1 | MLADenseBwdOp | attention | **分支无**（`main:attention.yaml:739`） | `main:_kernels/attention/mla/bwd.py`、`bwd_ws.py` | 无 | ❌ 缺失 | 移植 spec+Op+接口+workload+测试，再写 kernel |
| 2 | MLAVarlenBwdOp | attention | **分支无**（`main:attention.yaml:776`） | 同上 | 无 | ❌ 缺失 | 同上，packed/varlen 版本 |
| 3 | MoEPermuteAlignFwdOp | moe | `moe.yaml:1` | `kernels/moe/permute_align.py`（`MoEPermuteAlignKernel`） | 无 | ⬜ 待写 | 纯整型索引 kernel，无 MFU 口径 |
| 4 | SharedExpertMLPFwdOp | moe | `moe.yaml:477` | `kernels/moe/shared_expert_mlp.py` | 无 | ⬜ 待写 | 两段 GEMM + SiLU 门控 |
| 5 | EngramGateConvFwdOp | sequence_modeling | `sequence_modeling.yaml:3` | `kernels/sequence_modeling/engram/gate_conv_fwd.py` | `kernels/ascend/sequence_modeling/gate_conv_fwd.py:232` | ✅ 已有 | 验收 + 调优 |
| 6 | EngramGateConvBwdOp | sequence_modeling | `sequence_modeling.yaml:44` | `kernels/sequence_modeling/engram/gate_conv_bwd.py` | `kernels/ascend/sequence_modeling/gate_conv_bwd.py:485` | ✅ 已有 | 验收 + 调优 |
| 7 | EngramDecodeFwdOp | sequence_modeling | `sequence_modeling.yaml:91` | `kernels/sequence_modeling/engram/decode.py` | `kernels/ascend/sequence_modeling/decode.py:326` | ✅ 已有 | 验收 + 调优（`development.md` 已有专章） |
| 8 | RoPEFwdOp | rope | `rope.yaml:3` | `kernels/rope.py` | `kernels/ascend/rope/rotation.py:194`（neox）、`:198`（interleaved） | ✅ 已有 | 验收 + 调优 |
| 9 | PagedKVCacheWriteFwdOp | attention | `attention.yaml:679` | `kernels/attention/paged_cache_write.py` | 无 | ⬜ 待写 | 原地 scatter，含 FP8 饱和 cast |
| 10 | SSDChunkCouplingFwdOp | mamba | `mamba.yaml:52` | `kernels/mamba/ssd_chunk_coupling.py` | `kernels/ascend/mamba/chunks.py:271` | ✅ 已有 | 验收 + 调优 |
| 11 | ChainSpeculativeSamplingFwdOp | sampling | `sampling.yaml:149` | `kernels/sampling/chain_speculative_sampling.py` | `kernels/ascend/sampling/draws.py:266` | ✅ 已有 | 验收 + 调优 |

已有的 6 个 Ascend 实现全部通过 `register_kernel_type("<OpName>", "ascend", <Kernel>)` 注册，与 `docs/ascend950/development.md` 描述的 backend 约定一致；未改 GPU 实现、未改公开签名。

### 1.1 与上游 main 的差异（迁移前必须知道）

1. **两条 git 历史无关**：`TileOPs` 属 `tile-ai/TileOPs`，`TileOPs-ascend950` 是 `yyttt6/TileOPs` 的 fork，`git merge-base` 为空 → 不能 cherry-pick，只能按文件手工移植。
2. **目录布局不同**：main 是 `_ops/` + `_kernels/`，分支是 `ops/` + `kernels/`。
3. **算子命名有分叉**：main `MLADenseDecodeFwdOp` ↔ 分支 `MLADecodeWithKVCacheFwdOp`；main `HyperConnectionPreFwdOp` ↔ 分支 `MHCPreFwdOp`。移植 MLA 反向时要连带核对 `call_spec` 的 Call/Interface 名。
4. **分支 MLA 只有前向**：`kernels/attention/mla/` 仅 `decode.py`、`paged.py`、`prefill_varlen.py`、`prefill_varlen_ws.py`；`attention.yaml` 只有 `MLADecodeWithKVCacheFwdOp` / `MLAPagedFwdOp` / `MLAVarlenFwdOp` / `MLAKVCacheWriteFwdOp`。全分支检索不到 `MLADenseBwd`/`MLAVarlenBwd`/`mla_bwd` 任何符号。
5. **时间差**：main 比分支 HEAD 晚 11 天，MLA 反向很可能是 fork 之后才进 main 的，因此分支缺这两条属正常缺口，不是被删。

## 2. 逐算子说明

每节口径：语义 / 签名要点 / manifest 用例 / roofline / Ascend 现状 / 验收要点。

### 2.1 MLADenseBwdOp（❌ 分支缺失）

- **语义**：MLA prefill 反向（latent 已解压）。每个头有独立的 `D_qk` 宽 query/key（共享 rope 部分展开到每一头）和独立的 `D_v` 宽 value。`o`、`lse` 取自前向；`lse` 是每行 scaled score 的自然对数 log-sum-exp。**query i 对应 key 位置 `i + S_kv - S_q`**（bottom-right 对齐）；看不到任何 key 的 query 其 `lse = -inf`、梯度为 0。`sm_scale` 默认 `D_qk ** -0.5`。
- **ref_api**：`flash_attn_interface._flash_attn_backward`。
- **签名**：输入 `do [B,S_q,H,D_v]`、`q [B,S_q,H,D_qk]`、`k [B,S_kv,H,D_qk]`、`v [B,S_kv,H,D_v]`、`o [B,S_q,H,D_v]`、`lse [B,H,S_q]`（fp32）；输出 `dq`、`dk`、`dv`；params `is_causal`、`sm_scale`。
- **manifest 用例**（4 条）：`ds-v3-4k`（B1 S4096 H128 D_qk192 D_v128，bf16+fp16）、`ds-v3-b2-8k`（B2 S8192 H128，bf16）、`kimi-k2-8k`（B1 S8192 H64，bf16）、`ds-v2-lite-3k`（B4 S3072 H16，bf16）。
- **roofline**：`tileops._perf.formulas.mla_dense_bwd_roofline`（按 bottom-right 对齐 + causal 掩码统计每行看到的 score 数）。
- **Ascend 现状**：无。分支内既无 spec 条目、无 Op 类、无 kernel。
- **验收要点**：
  1. `D_qk = 192` 不是 2 的幂（= 128 nope + 64 rope），分块要对 192 单独处理，不能用 GEMM 那套 128/256 对齐假设。
  2. bottom-right 对齐 + causal ⇒ 对角块不等长，`S_q ≠ S_kv` 时容易把掩码写错。
  3. `lse` 是 fp32 且为自然对数（不是 log2），attention 反向的标准公式里若用 log2 会引入 `ln2` 因子误差。
  4. 参考实现依赖 flash-attn，NPU 上不存在；正确性比对需在 `torch_npu` 上另找等价路径（fp32 朴素反向可作参考，但**注意 NPU 无 float64**）。

### 2.2 MLAVarlenBwdOp（❌ 分支缺失）

- **语义**：`MLADenseBwdOp` 的 packed 形式。每个 request 只看自己的 key，其 query i 对应 key 位置 `i + kv_len - q_len`；`lse` 是 **head-major `[H, T_q]`**（即 FA3 对 packed 请求的返回布局）。
- **ref_api**：同 `flash_attn_interface._flash_attn_backward`。
- **签名**：输入 `do [T_q,H,D_v]`、`q [T_q,H,D_qk]`、`k [T_kv,H,D_qk]`、`v [T_kv,H,D_v]`、`o [T_q,H,D_v]`、`lse [H,T_q]`、`cu_seqlens_q [B+1]`、`cu_seqlens_kv [B+1]`（int32）；params 另有 `validate_inputs`（kw_only）。
- **manifest 用例**（3 条）：`ds-v3-8x4k`（T_q=T_kv=32768，8×4096，bf16+fp16）、`ds-v3-mixed`（T=32256，`q_lens`/`kv_lens` = 512/1024/2048/4096/8192/16384，bf16）、`kimi-k2-4x8k`（T=32768，4×8192，H64，bf16）。
- **roofline**：`tileops._perf.formulas.mla_varlen_bwd_roofline`（按各 request 自己的掩码求和）。
- **Ascend 现状**：无。
- **验收要点**：除 §2.1 的 4 点外，额外要注意——`dq/dk/dv` 要按 `cu_seqlens` 偏移写回；长度极端不均（512 到 16384）会带来核间负载不均，按 request 均分 grid 会出现长尾；`lse` 是 head-major 而 dense 版是 `[B,H,S_q]`，转置点别弄反。

### 2.3 MoEPermuteAlignFwdOp（⬜ 待写 kernel）

- **语义**：vLLM `moe_align_block_size` —— 按 expert 把 token 分桶，每个 expert 的 token 数补齐到 `block_size` 的整数倍，产出排序后的 token id、每个块的 expert id、以及补齐后的总 token 数。
- **ref_api**：`vllm._custom_ops.moe_align_block_size`。
- **签名**：输入只有 `topk_ids [T,K]`（int32，取值 `[0, num_experts)`）；params `num_experts`、`block_size`（默认 64）；输出 `sorted_token_ids [P]`、`expert_ids [ceil_div(P, block_size)]`、`num_tokens_post_pad [1]`，其中 `P = T*K + (num_experts+1)*(block_size-1)`。
- **manifest 用例**（12 条）：E=384（Kimi K2）／E=256（DeepSeek-V3）／E=128（Qwen3），每个 E 下 T ∈ {1, 32, 512, 4096}，`K=8`、`block_size=64`。
- **roofline**：`flops: "0"` —— 纯整型索引计算。
- **Ascend 现状**：无 Ascend kernel；接口已存在 —— `PermuteAlignCall`（`kernels/moe/call_spec.py:59`）、`PermuteAlignFwdInterface`（同文件 `:211`），GPU 实现为 `MoEPermuteAlignKernel`（`kernels/moe/permute_align.py:342`）。
- **验收要点**：
  1. **roofline flops = 0 ⇒ MFU 无意义**，验收只看正确性 + 时延对比（GPU 侧 vLLM `moe_align_block_size`）。
  2. 本质是 histogram → prefix sum → scatter 三段，全整型，不需要 Cube；难点在 4096×8=32768 个 id 的一次性计数与跨 expert 前缀和，以及 `num_experts` 从 128 到 384 的动态范围。
  3. 易踩坑：无效 id 桶（`num_experts` 本身那一桶）也要参与 padding；`expert_ids` 长度是 `ceil_div(P, block_size)` 而非 `num_experts`。
  4. 与已实现的 `MoEPrePermuteFwdOp`（`kernels/ascend/moe/pre_permute.py`）**不是同一算子**，别照抄——前者是 align 分桶，后者是 permute 搬数据。

### 2.4 SharedExpertMLPFwdOp（⬜ 待写 kernel）

- **语义**：MoE 层 shared expert 的 dense gated MLP：`down(silu(gate) * up)`，gate 与 up 的行堆叠在同一份权重里。
- **签名**：输入 `hidden_states [T,H]`、`w_gate_up [2S,H]`、`w_down [H,S]`（均 contiguous，fp16/bf16）；输出 `output [T,H]`。
- **manifest 用例**（5 条）：`kimi-k2-t1`（T1 H7168 S18432，bf16）、`kimi-k2-t512`（T512 H7168 S18432，bf16）、`ds-v3-t32`（T32 H7168 S2048，bf16+fp16）、`ds-v3-t4096`（T4096 H7168 S2048，bf16）、`glm45-t32`（T32 H5120 S1536，bf16）。
- **roofline**：`6*T*S*H + 6*T*S`（gate/up 与 down 两个 GEMM，加门控逐元素）。
- **Ascend 现状**：无 Ascend kernel；接口已存在 —— `SharedExpertMLPCall`（`kernels/moe/call_spec.py:68`）、`SharedExpertMLPFwdInterface`（同文件 `:236`），GPU 实现 `kernels/moe/shared_expert_mlp.py`。
- **验收要点**：
  1. 结构清晰：`[T,H] × w_gate_upᵀ → [T,2S]` → SiLU(gate)·up → `[T,S] × w_downᵀ → [T,H]`，两段 GEMM 可直接复用 `gemm` 的调度经验（参考 `docs/ascend950/gemm.md`）。
  2. **只有 T=512 / T=4096 的用例谈 MFU 才有意义**；T=1 / T=32 是 decode 场景，latency-bound（gemm.md 里 M=1/2 的 case 峰值利用率仅 0.5–1%，属正常现象）。
  3. 中间张量 `[T,2S]` 在 T=4096、S=18432 时是 4096×36864×2B ≈ 288 MiB，别整块放 shared，要 tile 化或分段；S=18432 这类非 2 的幂宽度要处理 N 尾块。
  4. 有一个现成的对照 workload `FusedMoESharedExpertFwdOp`（`moe.yaml:506`）会调用本算子，别把它俩搞混。

### 2.5 PagedKVCacheWriteFwdOp（⬜ 待写 kernel）

- **语义**：把 token `n` 的 `k` / `v` 行按 `slot_mapping[n]` 散射进 `k_pages` / `v_pages` 的对应 slot；`-1` 表示跳过该 token；"非负 slot 互不相同"是调用方的义务，kernel 不查重。FP8 cache 存 `(k / k_scale)` 与 `(v / v_scale)`，用饱和 cast。
- **ref_api**：`vllm._custom_ops.reshape_and_cache_flash`。
- **签名**：`k [N,H_kv,D]`、`v [N,H_kv,D]`（fp16/bf16）、`k_pages`/`v_pages [NP,PS,H_kv,D]`（**mutated + contiguous**）、`slot_mapping [N]`（**int64**）、`k_scale`/`v_scale [1]`（fp32，optional，且 `v_scale` 仅当 `k_scale` 存在时存在）。**输出为空**（原地写）。
- **dtype 组合**：`fp16/fp16`、`bf16/bf16`、`fp16/fp8_e4m3fn`、`bf16/fp8_e4m3fn`；约束 `(KV == float8_e4m3fn) == present(k_scale)`。
- **manifest 用例**（6 条）：`llama-8b-decode`（N64 H8 D128 NP4096 PS16，bf16+fp16）、`llama-8b-prefill`（N8192 H8 D128）、`qwen3-235b-fp8`（N4096 H4 D128，fp8 + k_scale）、`gpt-oss-120b`（N256 H8 D64）、`odd-tokens-and-page`（N7 H2 D64 **NP5 PS7**，fp16）、`qwen3-235b-fp8-fp16`（N1024 H4 D128 **NP512 PS32**，fp8 + k_scale）。
- **roofline**：`tileops.perf.formulas.paged_kv_cache_write_roofline`（只统计有 slot 的 token 的读写）。
- **Ascend 现状**：无 Ascend kernel；接口已存在 —— `PagedKVCacheWriteCall`（`kernels/attention/call_spec.py:848`）、`PagedKVCacheWriteFwdInterface`（同文件 `:879`），GPU 实现 `kernels/attention/paged_cache_write.py`。
- **验收要点**：
  1. **11 个里唯一"没有输出"的算子**，正确性只能靠 mutated 输入的比对（manifest 的 `mutated: true` 语义），不要写成返回新张量。
  2. `slot_mapping` 是 **int64**，Ascend 侧索引宽度和 `int32` 混淆会静默写错位置；`-1` 必须跳过而不是回绕到最后一页。
  3. `PS=7`、`NP=5` 的奇页用例专门打非 2 的幂页尺寸；`qwen3-235b-fp8-fp16` 的 `PS=32` 又是另一档。
  4. FP8 分支要求**饱和 cast**（clamp 到 e4m3 可表示范围）而非普通截断；同时传的是 `k / k_scale`，是除不是乘。
  5. N=7 的小用例会暴露 launch 固定开销，别按大 N 的 tile 策略一把梭。

### 2.6 EngramGateConvFwdOp（✅ 已有，验收+调优）

- **语义**：post-projection 的 Engram GateConv 融合 —— 对 `H`/`k`/`v` 各做一次 RMSNorm，gate 由归一化后的点积经 sigmoid 得到，产出 `vhat` 和残差 `Y`，再叠一个 causal width-4 depthwise 卷积（tap 数固定 4，dilation 由用例给定）。
- **签名**：输入 `H`,`k`,`v` `[M,seq_len,d]`、`rms_w_h`,`rms_w_v` `[d]`、`conv_w` `[4,d]`（fp16/bf16）；输出 `Y`,`vhat` `[M,seq_len,d]` 与 `alpha`,`rrms_h`,`rrms_k`,`rrms_v`（**fp32** `[M,seq_len]`）。
- **manifest 用例**（4 条）：`train-s32`（M1 S32 d256 fp16）、`train-s64`（M2 S64 d512 fp16）、`train-s128`（M1 S128 d256 bf16）、`train-s96`（M3 S96 d768 bf16）。
- **roofline**：`M*(seq_len*(21d+14) + 2d*(seq_len + (seq_len-1) + (seq_len-2) + (seq_len-3)))`。
- **Ascend 现状**：`kernels/ascend/sequence_modeling/gate_conv_fwd.py:232`，**两次 AIV launch**（`_build_gate_conv_fwd_gate` 归约段 + `_build_gate_conv_fwd_conv` 卷积段）。`refusal`：仅 fp16/bf16、`m/seq_len/d > 0`、**`d ≤ 4096`**。
- **验收要点**：6 个输出里有 3 个 fp32 中间量，必须逐个比对；用例最大 d=768、M≤3，全是小规模，属 latency 场景，别用 MFU 判成败（roofline 由 `d` 线性项主导）。

### 2.7 EngramGateConvBwdOp（✅ 已有，验收+调优）

- **语义**：§2.6 的完整反向。输入除正向的 `dY` 外，还要前向保存的 `vhat`、`alpha`、`rrms_h`、`rrms_k`、`rrms_v`（即保存的是真实前向输出，含已舍入的 `vhat`）。
- **签名**：输出 `dH`,`dk`,`dv` `[M,seq_len,d]` 与 `drms_w_h`,`drms_w_v` `[d]`、`dconv_w` `[4,d]`（三个权重梯度均为 **fp32**）。
- **manifest 用例**：同 §2.6 的 4 条（`train-s32/s64/s128/s96`）。
- **roofline**：`M*(seq_len*(45d+10) + 6d*(seq_len + (seq_len-1) + (seq_len-2) + (seq_len-3)))`。
- **Ascend 现状**：`kernels/ascend/sequence_modeling/gate_conv_bwd.py:485`（`register` 在 `:563`，全文 563 行）。**staged SIMD AIV**：每个 token 自持激活梯度与 fp32 权重梯度 partial，再用**有序归约**合并，刻意避开浮点原子加。`refusal` 在 `:496`。
- **验收要点**：权重梯度是 fp32 跨 token 累加，归约顺序会影响末位，按 manifest 的数值容差判；`drms_w_h`/`drms_w_v`/`dconv_w` 三个都要单独断言，容易只查 `dH/dk/dv`。

### 2.8 EngramDecodeFwdOp（✅ 已有，验收+调优）

- **语义**：decode 单步。`e_t [batch,d_mem]` 经 `W_K`/`W_V [d_mem,d]` 投影 → RMSNorm → gate → **膨胀卷积**（读 `conv_state` 最近 `L` 步，左侧 padding 到 `max_conv_len`）→ 写 `y_t` 与 `new_conv_state`（长度恒为 `max_conv_len`）。
- **签名**：输入 `e_t [batch,d_mem]`、`h_t [batch,d]`、`conv_state [batch,L,d]`、`W_K`/`W_V [d_mem,d]`、`rms_w_h`/`rms_w_v [d]`、`conv_w [conv_kernel_size,d]`；输出 `y_t [batch,d]`、`new_conv_state [batch,max_conv_len,d]`。约束 `max_conv_len ≥ dilation*(conv_kernel_size-1)`、`L ≤ max_conv_len`。
- **manifest 用例**（3 条）：`serving-b1`（b1 d_mem512 d256 max_conv_len12 k4 dilation3 fp16）、`serving-b4`（b4 1024/512 20 4 5 fp16）、`serving-b8`（b8 512/256 18 4 3 bf16）。
- **roofline**：`4*batch*d_mem*d + batch*(16d + conv_kernel_size*2d) + 20*batch`。
- **Ascend 现状**：`kernels/ascend/sequence_modeling/decode.py:326`，**两次 launch** —— Cube 投影段（batch 按 16 行、输出维按 64 列分块，K/V 权重 tile 在 batch 行间共享，写 fp32 K/V 中间量）+ Vector 段（每核一行 batch，把整行 RMSNorm 归约、gate、膨胀卷积、SiLU、history 更新放在一起）。**无显式 `refusal`**（默认接受合法 call）。`docs/ascend950/development.md` 已有 Engram decode 专章，可直接引用其设计描述与 3 条 serving 用例表。
- **验收要点**：正确性要**同时**查 `y_t` 和完整更新后的 `new_conv_state`（history 写错是这类算子最常见的漏检）；性能必须覆盖整个算子调用并保留两次 launch 的 device 记录，分段测量只作诊断。

### 2.9 RoPEFwdOp（✅ 已有，验收+调优）

- **语义**：`neox` 让一个头的前后两半互旋，`interleaved` 让相邻元素成对旋转；两者频率表相同，只有配对方式不同。`base` 默认 10000。
- **签名**：`x` 为 1d `[S,D]` 或 2d `[B,S,H,D]`（由 `input_layout` 决定）；`T ∈ fp16/bf16/fp32`；约束 `D > 0 且 D % 2 == 0`。
- **manifest 用例**（6 条）：`neox-1d-2k`（S2048 D64，**fp16+fp32**）、`neox-1d-4k`（S4096 D128 bf16）、`neox-2d-2k`（B2 S2048 H32 D128 fp16）、`interleaved-1d-2k`（S2048 D64 fp16）、`interleaved-2d-2k`（B2 S2048 H32 D128 bf16）、`interleaved-1d-5k`（S5120 D128 **fp32**）。
- **roofline**：`3 * N`（每个被旋转元素两次乘一次加），cos/sin 表算作 op state 不计入。
- **Ascend 现状**：`kernels/ascend/rope/rotation.py`，注册了两个 kernel_type 名 —— `"ascend"` → `AscendRoPENeoxKernel`（`:137`，`:194` 注册）与 `"ascend_interleaved"` → `AscendRoPEInterleavedKernel`（`:141`，`:198` 注册），公共基类 `_AscendRoPEKernel` 设 `devices = frozenset({"npu"})`。同文件还让 `RoPELlama31`/`YaRN`/`LongRoPE` 复用同一 kernel 类，`RoPENeoxPositionIds` 单独（`AscendRoPEPositionsKernel:147`）。
- **验收要点**：**无 `refusal`**，所有合法 call 都会进这个 kernel，包括两个 **fp32** 用例；这是本批唯一的纯 elementwise/带宽型算子，roofline 应看有效带宽而不是 MFU；改这个文件会连带影响 Llama31/YaRN/LongRoPE 四个算子，回归要一起跑。

### 2.10 SSDChunkCouplingFwdOp（✅ 已有，验收+调优）

- **语义**：每个 chunk、每个 group 的 causal `C@B` 耦合矩阵：`cb[b,c,g,l,s] = Σ_n C[b,c*Q+l,g,n] * B[b,c*Q+s,g,n]`（`s ≤ l`，否则 0）。
- **签名**：`C_mat`/`B_mat` `[B,S,G,N]`（fp16/bf16）；输出 `cb [B,NC,G,chunk_len,chunk_len]`；`NC = S // chunk_len`；约束 `S % chunk_len == 0`。
- **manifest 用例**（3 条）：`mamba2-780m`（B1 S4096 G1 N128 chunk256 fp16）、`mamba2-2p7b`（B4 S2048 G1 N128 bf16）、`mamba2-grouped-prefill`（B3 S3072 G8 N128 fp16）。
- **roofline**：`B*NC*G*chunk_len*chunk_len*N`（因果掩码已折半）。
- **Ascend 现状**：`kernels/ascend/mamba/chunks.py:271`，`devices = frozenset({"npu"})`，**无显式 `refusal`**。同文件还有 `SSDChunkState`(`:305`)、`SSDChunkScan`(`:335`)、`SSDChunkCumsum`、`SSDStatePassing`、`SSDRecurrent` 的 Ascend 实现 —— 动这个文件会波及 mamba 整族。
- **验收要点**：本质是 `chunk_len×N` 乘 `N×chunk_len` 的 batched GEMM 加上三角掩码清零，适合走 Cube；用例的 `G` 从 1 到 8、`NC` 从 8 到 16，可并行维度本来就不大，注意核占用率；掩码方向搞反（`s ≤ l` 是对下三角保留）是这类算子经典错误。

### 2.11 ChainSpeculativeSamplingFwdOp（✅ 已有，验收+调优）

- **语义**：每条请求验证 `N` 个 draft token。第 `i` 个以 `min(1, target/draft)` 接受；首次拒绝处从 `normalize(max(0, target - draft))` 抽一个，该行其余位置填 `-1`；若全部接受，则从 `target` 的第 `N` 行抽一个 bonus token。`num_accepted` 是接受前缀长度，**不含** bonus。
- **签名**：`draft_probs [B,N,V]`、`target_probs [B,N+1,V]`（fp32）、`draft_token_ids [B,N]`（int32）、`seed`/`offset [1]`（int64）；输出 `output_token_ids [B,N+1]`（int32）、`num_accepted [B]`（int32）。
- **manifest 用例**（7 条）：`llama-8b-b16-n4`、`llama-8b-b64-n4`、`ds-v3-mtp-n1`、`ds-v3-mtp-n3`、`qwen3-235b-eagle-n5`（**V=151936**）、`llama-8b-b17-n8`、`llama2-7b-b256-n1`（V=32000）。
- **roofline**：flops 与 bytes 都取"链停在哪"的两个极端里**更便宜**的一侧 —— 这是下界估计，不是严格上界。
- **Ascend 现状**：`kernels/ascend/sampling/draws.py:266`，与 `SamplingFromProbsFwdOp` 共用 `_DrawKernel` 基类（`prepare`/`parts`/`select` 三段；`N ≤ 1024` 走 `_prepare_parallel`，否则走 `_prepare`；`width = min(8192, max(64, next_pow2(V)))`）。`refusal`（`:224`）：`B*(N+1)*V ≤ 2^31-1`、`N ≤ 8192`、`V ≤ 2^26`。
- **验收要点**：随机性来自 `seed`/`offset`，**必须固定 seed 才能逐位比对**；`V` 到 15 万，主导成本是整行概率的读写带宽；改 `_DrawKernel` 会同时影响 `SamplingFromProbsFwdOp`（上一批已验收的算子），回归必须两个一起跑。

## 3. 三条工作路线

### 3.1 路线 A —— 已有实现（6 个）：先确认归属，再验收调优

这 6 个的 Ascend 实现都在分支 HEAD `2362d1e9` 里（来源提交同为该 merge），**很可能不是本批要新写的**。动手前先确认是否已被验收过（谁做的、有无 receipt / PR 台账），否则容易重复劳动。

正确性回归（首次必须先 `pip install -e . --no-deps` 让 backend entry point 生效）：

```bash
pytest tests/ops/test_engram.py tests/ops/test_engram_ascend.py \
       tests/ops/test_rope.py tests/ops/test_mamba.py tests/ops/test_sampling.py \
       --tileops-device=npu -q
```

注意回归范围：`test_mamba.py` 覆盖 mamba 整族（`chunks.py` 里还有 State/Scan/Cumsum/Recurrent/StatePassing），`test_sampling.py` 同时覆盖 `SamplingFromProbsFwdOp`，`test_rope.py` 牵动 Llama31/YaRN/LongRoPE —— 改一个会波及一片。

调优记录方式参照 `docs/ascend950/gemm.md`：每个 case 记录 tile 形状、stage 数、以及和同轮 baseline 的比值，单变量对比，不写"调优过"就算完。

### 3.2 路线 B —— 3 个待写 kernel：接口都现成，只缺 Ascend 实现

| 算子 | 已有接口 | roofline 口径 | 判成败的主要指标 |
|---|---|---|---|
| MoEPermuteAlignFwdOp | `PermuteAlignCall`（`kernels/moe/call_spec.py:59`）+ `PermuteAlignFwdInterface`（`:211`） | `flops: "0"` | 正确性 + 时延（**无 MFU 概念**） |
| SharedExpertMLPFwdOp | `SharedExpertMLPCall`（`:68`）+ `SharedExpertMLPFwdInterface`（`:236`） | `6*T*S*H + 6*T*S` | MFU，但仅 T≥512 有意义 |
| PagedKVCacheWriteFwdOp | `PagedKVCacheWriteCall`（`kernels/attention/call_spec.py:848`）+ `PagedKVCacheWriteFwdInterface`（`:879`） | `paged_kv_cache_write_roofline` | 正确性（mutated 比对）+ 带宽 |

写法沿用 `docs/ascend950/development.md` 的 backend 约定：

1. 新建 `src/tileops/kernels/ascend/<family>/<op>.py`，类继承对应 `*FwdInterface`，`devices = frozenset({"npu"})`；
2. 实现 `refusal(cls, call)` 声明支持的 shape/dtype 边界（本批已有实现里 Engram 与 sampling 都写了，可作为范式）；
3. 模块底部 `register_kernel_type("<OpName>", "ascend", <Kernel>)`；
4. 在 `kernels/ascend/<family>/__init__.py` 的 `__all__` 与 import 中登记；
5. 复用分支内已有测试（`test_moe_permute_align.py`、`test_fused_moe_shared_expert.py`、`test_paged_kv_cache_write.py`），必要时仿 `test_engram_ascend.py` 的 Ascend 专用入口另加。

### 3.3 路线 C —— 2 个 MLA 反向：先移植，再写 kernel

前置移植清单（main → 分支，**布局和命名都要改**）：

| 内容 | main 位置 | 分支目标 |
|---|---|---|
| manifest spec 条目 | `attention.yaml:739`（Dense）、`:776`（Varlen） | `attention.yaml` 新增两条 |
| Op 类 | `_ops/attention/mla.py:292`、`:385` | `ops/attention/mla.py` |
| kernel 接口 + call_spec | `_kernels/attention/mla/bwd.py`、`bwd_ws.py`、`_kernels/attention/call_spec.py` | `kernels/attention/mla/`、`kernels/attention/call_spec.py` |
| workload | `workloads/attention/mla.py:536`、`:607` | `workloads/attention/mla.py` |
| 对外导出 | `attention.py`、`_ops/attention/__init__.py` | `attention.py`、`ops/attention/__init__.py`（含 `ops/__init__.py` 的 name→module 映射与 lazy import 两处） |
| 测试 | `tests/ops/test_mla_bwd.py` | `tests/ops/test_mla_bwd.py` |

移植时必须同步改的三处：

1. **roofline 函数路径**：main 是 `tileops._perf.formulas.mla_dense_bwd_roofline` / `mla_varlen_bwd_roofline`，分支是 `tileops.perf.formulas.*`（对比 `attention.yaml:714` 的 `paged_kv_cache_write_roofline`）—— 不改会让 manifest 校验直接失败。
2. **命名**：分支没有 `_ops`/`_kernels` 前缀；且 MLA decode 在分支叫 `MLADecodeWithKVCacheFwdOp`（main 是 `MLADenseDecodeFwdOp`），抄导入时要连同 `call_spec` 的 Call/Interface 名一起核对。
3. **参考实现**：`ref_api` 指向 `flash_attn_interface._flash_attn_backward`，NPU 上没有 flash-attn —— 正确性比对需另定路径（fp32 朴素反向可作参考，但**NPU 无 float64**；该坑已列入 §4 的口径红线）。

## 4. 环境与验收口径

| 项 | 口径 |
|---|---|
| 环境 | CANN 9.2.0 编译 / 9.1.0 runtime、tilelang 0.1.15（含 `tilelang.ascend`）、torch 2.13.0+cpu / torch_npu 2.13.0rc1、Python 3.12、venv `xiangmu/.venvs/ascend950` |
| 正式性能 | `scripts/benchmark_ascend.py --owner … --run-dir <新建目录> --clock-config … --op <OpName> [--case-regex …] [--guard …]`，协议 `ascend950-profiler-locked-v1`（3 次 warmup + 15 个独立 profiler 采样 + 每次采样前 1 GiB 驱逐 + 前后时钟读数）；run-dir 必须新建，无静默续跑 |
| 仅正确性 | 加 `--verify-only` 并省略 `--clock-config`；**不出性能结论** |
| 多实现对比 | `--implementation-factory package.module:function`，与 TileOPs 共用同一参考、输入、reset 顺序和计时器 |
| CI | `python scripts/ci/ascend950.py --phase cpu|device --output <新建目录>` |
| 目标 | 正常目标 稀疏 CV MFU > 50%、CV > 80%、其余 > 90%；挑战目标超过库上最新 Ascend 实现；不达标须按「结论摘要 / 改动文件 / 功能实现 / 实测性能 / 正确性证据 / 未达标原因」的结构出评估报告 |
| 已知坑 | ① NPU 无 float64，参考实现用 double 的用例会在 kernel 执行前就失败；② shared buffer 需 32B 对齐，否则 device error 507035；③ 本地无 `npu-guard` 时属"开发口径"，不是 canonical 验收口径；④ 当前设备是 28 核 950PR 样片（matmul ≈ 366.8 TFLOP/s），MFU 判定需 canonical 32 核 950DT |

## 5. 待确认事项

1. **卡分配**：本批用哪几张卡？本机 0/1/3 曾 Alarm、5 号性能异常、7 号不存在，需避开。
2. **6 个已有实现的归属**：是否已被正式验收（谁提交、有无 receipt/PR）？决定路线 A 是"调优"还是"接手补验收"。
3. **MLA 反向的参考实现**：NPU 无 flash-attn，正确性基线用什么？
4. **文档粒度**：当前是一份汇总。是否需要拆成 11 份单算子文档（每份含结论摘要 / 改动文件 / 功能实现 / 实测性能 / 正确性证据 / 未达标原因）？

---
本文件只做说明与现状核对，未改动任何算子实现、manifest 或分支代码。上一批历史文档已删除；本批的流程与可行性见同目录 `MIGRATION_LOOP.md` 与 `FEASIBILITY.md`。
