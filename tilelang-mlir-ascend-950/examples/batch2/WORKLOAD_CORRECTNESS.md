# 主导 Workload 正确性验证（P0-2）

目标：补上“正确性只在小 shape 上验证过”的缺口。对 11 个已迁移算子的**主导 workload**
（以 `examples/batch2/mfu_operators.csv` / `mfu_table.json` 为准）逐例做一次真机比对，
每例给出可复核的判据、误差与结论。

- 采集脚本：`examples/batch2/verify_workload_shapes.py`
- 结构化结果：`examples/batch2/workload_correctness.json`（逐 case 一行 JSON 的
  `workload_correctness.jsonl` 是落盘中继，脚本每跑完一例立即 append）
- 本报告：`examples/batch2/WORKLOAD_CORRECTNESS.md`
- **未修改任何算子 / kernel 代码**；只新增验证脚本、本报告与结果 JSON。

## 0. 结论摘要

| 算子 | 主导 case | 方法 | 判据 | max_abs_err | 结论 |
|---|---|---|---|---|---|
| EngramGateConvFwdOp | train-s96 `[3,96,768]` conv_w `[4,768]` bf16 | 参考对照 | atol 0.2 / rtol 0.1 | 1.95e-3 | **pass** |
| EngramGateConvBwdOp | train-s96 同上 bf16 | 参考对照（autograd） | atol 0.3 / rtol 0.2 | 6.50e-3 | **pass** |
| EngramDecodeFwdOp | serving-b4 `[4,1024]`,`[4,512]` cache 20 fp16 | 参考对照（含 cache 更新） | atol 0.05 / rtol 0.05 | 9.77e-4 | **pass** |
| MoEPermuteAlignFwdOp | ds-v3-t4096 `[4096,8]` E=256 blk 64 | 参考对照（路由等价精确） | 精确（含填充/块主） | 0 | **pass** |
| MoEPermuteAlignFwdOp | kimi-k2-t4096 `[4096,8]` E=384 blk 64 | 同上 | 精确 | 0 | **pass** |
| PagedKVCacheWriteFwdOp | qwen3-235b-fp8 `k[4096,4,128]`→cache`[4096,16,4,128]` fp8 | 参考对照 | 逐字节精确 | 0 | **pass** |
| RoPEFwdOp | llama3-8b-b2-2k `pos[2,2048]` q`[2,2048,4096]` fp32 | 参考对照 | atol 1e-5 / rtol 1e-5 | 4.77e-7 | **pass** |
| SSDChunkCouplingFwdOp | mamba2-grouped-prefill C/B`[3,3072,8,128]` chunk 256 fp16 | 参考对照 | atol 1e-3 / rtol 1e-3 | 2.44e-4 | **pass** |
| ChainSpeculativeSamplingFwdOp | ds-v3-mtp-n3 B=64 N=3 V=129280 fp32 | 参考对照（num_accepted 精确 + 契约） | num_accepted 精确 | 0 | **pass** |
| SharedExpertMLPFwdOp | 512×7168×18432 bf16 | 参考对照 | moe_close(atol=rtol=1e-2, nsq≤1e-4) | 9.77e-4 (nsq 6.6e-6) | **pass** |
| MLADenseBwdOp | ds-v3-b2-8k B=2 S=8192 H=128 bf16 | 契约对齐参考 + 切片自洽 + 活性 | bf16 5e-3 / 1.6e-2 | 3.13e-2 | **fail** ⁽¹⁾ |
| MLAVarlenBwdOp | ds-v3-mixed `[512..16384]` H=128 bf16 | 同上 | bf16 5e-3 / 1.6e-2 | 3.13e-2 | **fail** ⁽¹⁾ |
| MLAVarlenBwdOp | ds-v3-8x4k 8×4096 H=128 bf16（复核） | 同上 | bf16 5e-3 / 1.6e-2 | 3.13e-2 | **fail** ⁽¹⁾ |

（1）MLA 反向三例：**delta/lse 契约已完全对齐**（下面的“契约对齐”参考使用算子实际收到的
`lse` 与 `_delta_kernel` 的 `delta = rowsum(o·do)`），超差元素从 95/183/321 降到
**14 / 36 / 44**（占 1.3e-8 / 1.7e-8 / 2.0e-8），全部残差的 `k_ulp ≤ 1.0`（按 DESIGN §8.3
的 `max|Δ| / ulp(max|ref|)` 定义，即误差不超过最大量级的 1 个 bf16 舍入）；把算子设计文档
明文规定的 **P/dS → 输入 dtype 量化**（DESIGN §8.1 golden 语义）加进参考后，三例均
**0 超差**。见 §3。

13 行（11 个算子，MoE 与 MLA 各多一个 case）/ 13 个 case 全部在主导 shape 上真机跑过并比对，
无 `unverified` 项（覆盖边界见 §4）。**10 例通过；3 个 MLA 反向 case 按逐元素判据
（bf16 `atol 5e-3 + rtol 1.6e-2`）仍然 fail**，但残差已定位到 P/dS 量化的 1-ulp 尾部，
不是丢块/截断，也不是 delta 约定（§3）。

## 1. 环境、方法与判据

### 环境
- 硬件：Ascend950PR_9579（1 卡；`npu-smi` 单卡），NPU HBM 132 GB
- 工具链：`source .agents/platforms/toolchains/ascend/env-950pr.sh` → CANN 9.2.1，
  `torch 2.13.0+cpu` + `torch_npu`；解释器 `/workspace/venv-ascend950/bin/python`
- **容器 host 内存 cgroup 上限 32 GiB**（`/sys/fs/cgroup/memory/memory.limit_in_bytes`
  = 34359738368）。因此脚本的所有比对都在 NPU 侧做归约，只把标量搬回 host：早期版本在
  2.1e9 元素上做 host 拷贝会被 OOM killer 杀掉（SIGKILL，无 traceback）。见 §5。
- 每例固定 seed `20261124`（`torch.manual_seed` + `torch.npu.manual_seed_all`），
  单卡串行，逐例落盘。

### 两种方法
1. **参考实现对照**：用 `tileops/workloads/*` 里既有的 `gen_inputs` / `ref_program`
   （PyTorch fp32 / 整数语义参考）与算子输出逐输出比对，记录 `max_abs_err`、
   `max_rel_err`、超差元素数与 ULP 诊断。
2. **切片自洽**（MLA）：同一份输入先跑**全量**，再把张量沿**可精确分解的轴**
   （dense：batch、head；varlen：逐 request）拆开分别跑，拼回与全量输出比较；
   另外统计每个 request / head 的“活性”（整段 0 即截断嫌疑）。

### MLA 的三层参考口径
`MLADenseBwdWorkload.ref_program` / `MLAVarlenBwdWorkload.ref_program`（仓库既有 oracle）与
算子契约在 **`delta` 的来源**上不同；本轮把三者分开记录，**主判据用“契约对齐”一层**：

| 口径 | `lse` | `delta` | P/dS 精度 | 角色 |
|---|---|---|---|---|
| A. 既有 workload `ref_program` | 自行重算 logsumexp | `Σ_j P_j(do·v_j)`，fp32 | fp32 | 历史对照（§3.2） |
| **B. 契约对齐（主判据）** | **算子实际收到的 `lse`** | **`rowsum(o·do)`，fp32，`o` 为算子收到的 bf16** | fp32 | **判据用** |
| C. DESIGN §8.1 golden | 同上 | 同上 | **P、dS 舍入到输入 dtype**（kernel/源算子的语义） | 残差归因（§3.3） |

B 与 C 由本脚本实现（`mla_bwd_reference_contract`），C 只是在 B 上多一步量化。
**实现自检**：把 B 的 `delta_source` 换成 `"probs"`（即 A 的 delta 约定）后，与本仓库既有的
`mla_bwd_reference` 在 smoke 几何（q_lens=[256,384]，H=8，D=192/128）上**逐位相同**
（`max_abs_err = 0.0`，`--selfcheck`），说明 B/C 与既有 oracle 的差别只在上面两列契约输入，
不在反向算术本身。

### 判据（逐元素）
- 数值类：`|got-ref| <= atol + rtol*|ref|`，NaN 视为相等；
  bf16 → `atol=5e-3, rtol=1.6e-2`（`tests/ops/test_mla_*` 的 `_tolerance`，与 DESIGN §8.2
  的 `dk/dv` 行一致），fp16/fp32 用各自算子测试里的既有容差（见上表）。
  `dq_acc`（fp32 内部累加量，算子不对外返回）按 DESIGN §8.2 为 `atol 6.9e-2 + rtol 1e-3`；
  算子只输出 bf16 `dq`，故对外按 bf16 判据。
- 索引/拷贝类（MoE、Paged）：逐字节相等；MoE 额外允许“同一 expert 区间内顺序不定”，
  比较前把各 expert 区间排序（路由等价，且对参考自身做同样的自检）。
- ChainSpec：`num_accepted` 精确 + 结构/范围契约（`verify_chain_outputs`）；
  bonus token 索引按契约不要求与参考一致。
- ULP 诊断：`max_ulp`（逐元素按 `|ref|` 归一，见 JSON）与
  `k_ulp_global = max|Δ| / ulp(max|ref|)`（DESIGN §8.3 L0-12/L0-21 的定义）。

### 复现
```bash
cd examples/TileOPs
source ../../.agents/platforms/toolchains/ascend/env-950pr.sh
export TMPDIR=/workspace
/workspace/venv-ascend950/bin/python ../batch2/verify_workload_shapes.py --list
/workspace/venv-ascend950/bin/python ../batch2/verify_workload_shapes.py --selfcheck    # MLA 参考自检
/workspace/venv-ascend950/bin/python ../batch2/verify_workload_shapes.py --only mla_varlen_bwd
/workspace/venv-ascend950/bin/python ../batch2/verify_workload_shapes.py --merge         # -> workload_correctness.json
```
每例会向 stdout 打印 `RESULT {json}` 并 append 到 `workload_correctness.jsonl`。
总耗时（热缓存）：单进程跑完 13 例 64 s（各例 elapsed 合计 59.5 s）；MLA 三例分别
14 s / 33 s / 14 s，
NPU 峰值 37.5 / 79.1 / 75.5 GB。

## 2. 逐算子结果

### Engram（train-s96 / serving-b4）
- **Fwd** `M=3, seq=96, d=768, bf16`：6 个输出全部比对（`Y, v_hat, alpha, rrms_h, rrms_k,
  rrms_v`）。`Y` 1.95e-3（远低于 0.2；bf16 ULP 折算见 JSON）；`v_hat` 逐位相同；4 个 fp32
  中间量 `alpha/rrms_h/rrms_k/rrms_v` 的严格误差 1.2e-7–7.6e-6（判据只用 0.2，这里额外记录了严格值）。
- **Bwd** 同上 bf16：6 个梯度逐个比对，最大 `dconv_w` 6.50e-3，`dv` 1.95e-3，其余 ≤1e-3；
  均远低于 0.3。权重梯度是跨 token 归约，末位随归约顺序变化，符合预期。
- **Decode** `B=4, d_mem=1024, d=512, cache=20, taps=4, dilation=5, fp16`：
  `y_t` 3.05e-5、`new_conv_state` 9.77e-4（4 ulp）。`new_conv_state` 的**纯移位前缀**
  4×19×512 = 38912 个元素与参考**逐位相同**（历史错位类 bug 最容易藏在这里）。

### MoE permute/align（索引类）
`topk_ids [4096,8]`，block 64；E=256（ds-v3，主导表口径）与 E=384（kimi-k2，
`mfu_operators.csv` 的 dominant 行）各跑一次：
- `num_tokens_post_pad` = 40704 / 48768，与参考一致；
- 每个 expert 的 padded 区间内，路由 flat 索引集合（排序后）与参考完全相同，
  其余槽位为 `numel` 填充；`expert_ids` 的块主与参考完全相同；尾部填充正确。
- 参考自身也通过同一套路由等价检查（自检比对 harness）。

### PagedKVCacheWrite（qwen3-235b-fp8）
`tokens=4096, heads_kv=4, dim=128, page=16, pages=4096`，`bf16 → fp8_e4m3`，
`k_scale=1.25 / v_scale=0.75`（distinct 非 1 的比例，防止 kernel 丢掉或交换 scale）：
- 两个 pool 各 33 554 432 个元素与参考**逐字节相同**（含 `-1` 跳过、未被命名的行保持原值）；
- 实际被改写行数 = 4096 = tokens，slot 范围 `[3, 65511]`，确认不是“空跑也通过”。

### RoPE（llama3-8b-b2-2k，fp32）
`positions[2,2048]`，`q[2,2048,4096]`，`k[2,2048,1024]`，head 128，rotary 128，
base 5e5，NeoX。20 971 520 个元素全部在 `atol=rtol=1e-5` 内，最大绝对误差 4.77e-7。
（`max_rel_err` 0.48% 出现在接近 0 的小值元素上；该元素绝对误差 ≤4.8e-7。）

### SSDChunkCoupling（mamba2-grouped-prefill）
`C/B [3,3072,8,128]`，chunk 256，fp16：18 874 368 个元素全部在 1e-3/1e-3 内，
最大 2.44e-4；`max_rel_err` 7.7% 与 `k_ulp=1024` 都落在因相消而接近 0 的元素上
  （这些点的绝对误差仍 ≤2.44e-4，判据是 1e-3）。
额外校验：上三角（`s > l`）元素**恰好全为 0**（0 个非零），因果掩码没有漏写。

### ChainSpeculativeSampling（ds-v3-mtp-n3）
`B=64, N=3, V=129280`，fp32：`num_accepted` 64/64 **精确相等**；接受长度直方图
`[27,14,14,9]` 双方一致；结构契约（`-1` 位置、接受前缀=draft id、接受前缀的目标支撑、
绘制 token 的权重非零）通过；输出范围 `[-1, V)` 通过。bonus token 索引 255/256 相同
（契约允许不同，见 §4）。

### SharedExpertMLP（512×7168×18432，bf16）
`hidden [512,7168]`、`w_gate_up [36864,7168]`、`w_down [7168,18432]`：
`max_abs_err` 9.77e-4，归一化平方误差 `nsq = 6.6e-6`（判据 ≤1e-4），3 670 016 个元素
无一个超 `atol=rtol=1e-2`。该 shape 在 manifest 中已有 `test_shared_expert_mlp_matches_the_manifest_workloads`
覆盖；本次独立复跑确认。

### MLA 反向（3 个 case）
主判据（B：同一 `lse` + 同一 `delta`，fp32 反向）逐输出：

| case | dq | dk | dv | 超差元素 | 结论 |
|---|---|---|---|---|---|
| ds-v3-b2-8k | 1.56e-2 · k_ulp 1.00 · 0 | 1.56e-2 · 1.00 · 0 | 3.13e-2 · 1.00 · **14** | 14 / 1.07e9 (1.3e-8) | **fail** |
| ds-v3-mixed | 1.56e-2 · 0.50 · 0 | 1.56e-2 · 0.50 · **1** | 3.13e-2 · 1.00 · **35** | 36 / 2.11e9 (1.7e-8) | **fail** |
| ds-v3-8x4k | 1.56e-2 · 0.50 · **8** | 1.56e-2 · 1.00 · **1** | 3.13e-2 · 1.00 · **35** | 44 / 2.15e9 (2.0e-8) | **fail** |

（每格：max_abs_err · k_ulp_global · 超差元素数；判据 bf16 `atol 5e-3 + rtol 1.6e-2`。）

同一批输出在 C 口径（DESIGN §8.1 golden，加上 P/dS → 输入 dtype 量化）：
**三例全部 0 超差**，最大绝对误差 dq ≤3.9e-3 / dk ≤1.6e-2 / dv ≤3.1e-2，
`k_ulp_global` 0.12–1.00。
即：残差完全由“P/dS 用量化到输入 dtype 的值进 GEMM”这一条设计语义解释（§3.3）。

“非静默少算”的独立证据（与任何参考无关）：
- `ds-v3-mixed` kernel 自报
  `main pass for total_kv=32256: tiles=510 slab=32 launches=4 max_grid=16320 blocks=65280 useful=65280 idle=0`
  （4 次 launch × 16320 块 = 全部有效、按 32 head 分片、无空转块）；
- `ds-v3-8x4k`：
  `main pass for total_kv=32768: tiles=520 slab=32 launches=4 max_grid=16640 blocks=66560 useful=66560 idle=0`；
- 全量 vs 分片：dense 按 batch（2 份）+ head（4×32 份）、varlen 按 request 逐份重跑，
  `dk/dv` 与全量**逐位相同**，`dq` 最大差 4.88e-4（dense）/ 9.77e-4（varlen）；
- 活性：dense 128 个 head、varlen 6 个 request × 128 head 全部非零且无非有限值
  （dense 每 head 的 `|·|` 和集中在 69913–71253 / 55592–56471 / 45560–47652）。

## 3. MLA 反向：从“约定差异”到“契约对齐 + 残差归因”

### 3.1 主判据（B 口径）的可复现记录
- 固定 seed `20261124`；命令
  `python ../batch2/verify_workload_shapes.py --only mla_dense_bwd`（或 `mla_varlen_bwd`、
  `mla_varlen_bwd_8x4k`）。
- 参考：`mla_bwd_reference_contract()` —— `P = exp(qk·scale − lse)`，`lse` 用算子实参；
  `delta[t,h] = Σ_d o[t,h,d]·do[t,h,d]`（fp32，`o`/`do` 为算子实参的 bf16 张量，
  与 `_delta_kernel.golden_delta` 同定义）；`dS = P·(dP − delta)`；`dq = scale·dS·K`、
  `dk = scale·dSᵀ·Q`、`dv = Pᵀ·dO`；全程 fp32，最后一次性舍入到 bf16。
- 残差规模：`k_ulp_global` 0.5–1.0 —— 最大误差不超过**最大量级的 1 个 bf16 舍入**；
  逐元素 ULP（按 `|ref|` 归一）很大只是因为这些点落在相消后接近 0 的元素上（JSON 的
  `max_ulp`）。最差元素示例（dense，dv `[b=0, kv=1, h=78, d=16]`）：
  got −0.287109375 / ref −0.296875 / err 9.77e-3 / tol 9.75e-3（超 1.5e-5），
  `k_ulp` 折算 10 ulp-at-ref。

### 3.2 约定差异（历史）：原 bf16 判据下的 95 / 183 / 321
用既有 workload `ref_program`（A 口径）时，三个 case 分别有
**95 / 1.07e9（8.8e-8）、183 / 2.11e9（8.7e-8）、321 / 2.15e9（1.5e-7）** 个元素超差，
最大绝对偏差 3.13e-2。**这不是 kernel 缺陷**，证据链：

1. 超差元素按“可见 key 数”集中（bottom-right 因果，第 i 行看到 i+1 个 key）：
   dq 的超差元素中 dense 56/62、varlen 75/96、8x4k 161/203 只有 **2 个可见 key**。
2. fp64 逐行重算（dense 最差元素，request 1 / 第 1 行 / head 45）：

   | 量 | 值 |
   |---|---|
   | 算子输出（bf16） | −0.0311279296875 |
   | 既有参考（A：`delta = Σ_j P_j(do·v_j)`） | −0.01373291015625 |
   | host fp64（同一 A 公式） | −0.0137546 |
   | host fp64（B：`delta = Σ_d o·do`） | −0.03110874 |

   → 既有参考与 fp64 差 2.2e-5；算子与 B 公式只差 **1.9e-5**（0.3 bf16 ulp）。
3. 机理：只有 2 个可见 key 时 `P ≈ (½,½)`，`dS = P·(dP − delta)` 是两个几乎相等的大数相减，
   `delta` 的微小偏差被放大到接近 100% 相对误差；A 与 B 的 delta 数学等价但数值不同——
   A 用 fp32 的 `Σ_j P_j(do·v_j)`，B（算子）用 `Σ_d o·do`，其中 `o` 已是 bf16 前向输出。
4. 换成 B 口径后，这三批超差元素分别从 95→14、183→36、321→44（86%/80%/86% 消失），
   且剩余元素不再是 dq/dk 的“2-key 行”，而是 dv 的 1-ulp 尾部（§3.3）。

### 3.3 残差 14 / 36 / 44：P/dS 的输入 dtype 量化
- 残差与 `delta` 无关的方向：dense 的 14 个全在 `dv`，而 `dv = Pᵀ·dO` **不使用 delta**。
- 唯一还差的一步是 P（以及 dq/dk 用到的 dS）在进 GEMM 前被舍入到输入 dtype——这是
  源 GPU 算子与 DESIGN §8.1 golden 的明文语义（"fp32 物化 softmax + P 量化到输入 dtype"）：
  把这一步加进参考（C 口径），三例的 `dq/dk/dv` **全部 0 超差**（最大 3.13e-2，即 dv 的
  1 个最大量级 ulp，且落在 |ref| 足够大的元素上）。
- 量级自洽：`dv` 的误差来自 `Σ_i ε_ij P_ij do_i`，`ε` 为 bf16 相对舍入（2⁻⁹ ≈ 0.2%），
  在 `Σ_i|P_ij do_i| ≈ O(1–10)` 的相消背景下给出 ~1e-2 的绝对偏差，正好等于
  最大量级（|dv|max ≈ 6.3）的 1 个 bf16 ulp = 3.13e-2 的量级。
- 因此 C 口径的 `k_ulp ≤ 1` 与 DESIGN §8.3（L0-12/L0-21）的验收尺度一致；B 口径的超差
  是“1 ulp-of-max 的舍入落在 |ref| 很小的元素上”这一逐元素判据的固有性质，不是功能错误。

### 3.4 结论（不改代码）
- **算子与自身 `delta`/`lse` 契约一致**：把参考换成算子实参的 `lse` 与 `rowsum(o·do)` delta 后，
  dq/dk 的“2-key 行”类超差基本消失（dq 超差 dense 62→0、varlen 96→0、8x4k 203→8；
  dk 19→0 / 52→0 / 83→1），
  **原 fail 的 80–86% 是参考的 delta 约定差异**（A vs B），不是 kernel 缺陷，也不是截断。
- **仍按判据标 fail 的残差 14/36/44 个元素**（1.3e-8–2.0e-8）是 P/dS 量化到输入 dtype 造成的
  1-ulp 尾部；它属于算子/DESIGN 的明文语义（C 口径 0 超差），但不满足“fp32 反向参考 + 逐元素
  bf16 判据”。按任务约定**不放宽判据、如实标 fail**，并把三种口径的数字都留档。
- 可选后续（需评审，不在本任务范围）：① 若把 §8.2 的判据语义明确为“P/dS 以输入 dtype 计量”
  （即 C 口径），三例即为 pass；② 或在测试里对 attention 反向引入 `k_ulp` 判据/分位数判据；
  ③ 或让 kernel 的 PV/dS GEMM 使用 fp32 P（改 kernel，超出范围）。

## 4. 覆盖边界与“未验证”声明

13 个 case 全部真机比对过，**没有** `unverified` 项；但以下边界没有覆盖，不应由此报告
推断为已验证：

1. **单一 seed**：每例只用 `seed=20261124` 一次；未做多种子/多次抽样统计
   （MLA 的尾部超差比例 ~1e-8，多种子会改变具体元素，但结论形态预期不变）。
2. **ChainSpec 的 bonus token 索引**按契约不要求与参考逐位一致（Philox 流相同的是
   `num_accepted` 与被接受前缀）；本报告只对后者做了精确判据。另：该参考的 `float64`
   归约在 NPU 上被 `torch_npu` 静默降级为 fp32（运行时有 `Device do not support double
   dtype` 警告），所以实际是 fp32-vs-fp32 的比对——`num_accepted` 仍 64/64 精确。
3. **MLA 的输入 `o`/`lse`**：契约对齐口径把这两个张量当作**给定量**（这正是“给定相同统计量”
   的含义），因此本次校验的是**反向实现**，不覆盖 `_delta_kernel` 与前向 `o/lse` 本身的精度；
   后者的正确性由 `_delta_kernel` 的 L0 用例负责（其 golden 定义为 `Σ_d o·do`，见 §3.1）。
4. **MLA 的切片轴**：dense 只做 batch / head 两轴，varlen 只做逐 request；沿 seq 的
   切片对 causal attention backward 不满足“可拼接”语义，未做，也不适用。
5. **MoE** 只验证了 `top_k=8, block_size=64` 与 E∈{256, 384}；其他 E/block 组合只在
   小 shape 套件里覆盖。
6. **Paged** 只验证了 fp8 cache 的主导行；`llama-8b-prefill`（8192 token、bf16 cache）
   等其余 case 未在本次主导验证中重跑（其 k/cache 更大，判据相同）。
7. **非主导参数**（MLA 非 causal、`sm_scale` 覆盖、Engram 单 token / 跨 batch 边界等）
   仍依赖既有小 shape 套件；本报告不重复声明它们的正确性。
8. `dq_acc`（fp32 内部累加量）及其 DESIGN §8.2 容差（`atol 6.9e-2 + rtol 1e-3`）无法直接
   比对：算子只返回 bf16 `dq`，累加缓冲不对外暴露；本报告按 bf16 `dq` 的 §8.2 判据
   （5e-3 / 1.6e-2）判定。

## 5. 复现性备注（本次踩到的坑）

- **host 内存 cgroup 上限 32 GiB**：最初版本的比对把 `dq/dk/dv`（各 2–3 GB）拷到 host
  再做 fp32 归约，`ds-v3-mixed`（2.1e9 元素）被 OOM killer 直接 SIGKILL（exit 137，
  日志停在设备行、无 traceback）。修复方式是所有归约都在 NPU 上做、只回传标量；
  `memory.max_usage_in_bytes` 命中 34359738368 可作为该现象的证据。
- **不要从仓库根启动**：`examples/TileOPs` 作为工作目录，脚本自身把
  `examples/TileOPs` 插入 `sys.path`，避免 dev-root `tilelang/` 遮蔽。
- **参考实现的 delta 缓存**：B/C 参考对每个 request 重算 slab（8 head × 1024 行），
  峰值 NPU 内存 79 GB；不要试图把 `P` 全量物化（`O(B·H·S²)`）。
- 运行期临时目录已清理：`rm -f /tmp/tileops_msprof_*`（本次无 msprof 采集）。
