# Ascend950PR 算子迁移 —— 项目状态与任务说明

更新：2026-11-23　｜　工作目录：`/home/l00970450/advanced-compiler/team/lvpanhao/xiangmu`

## 1. 我们在做什么

把 TileOPs 的 **11 个 LLM 算子**从 GPU 参考实现迁移到**华为昇腾 Ascend950PR**，并使用**同一套自动化 Agent 框架**完成设计 → 检视 → 开发 → 调优 → 集成。

最终要交两样东西：

1. **11 个算子**（Ascend kernel + 注册 + 测试 + 性能记录）
2. **这套为 950PR 适配过的 Agent 框架**（含硬件档案、工具链锁、环境脚本、使用说明）

## 2. 四个目录的角色

| 目录 | 角色 | 能否写 |
|---|---|---|
| `TileOPs/` | 上游 GPU 主库（11 个算子的权威来源，迁移时的 `$GPU_REPO`） | 只读参考 |
| `TileOPs-ascend950/` | 交付目标仓（kernel 落点：`src/tileops/kernels/ascend/` + `register_kernel_type` + 验收） | 可写 |
| `tilelang-mlir-ascend-950/` | **框架执行环境**（分支 `950pr-adaptation`，基于 theBei 的 `ascend950-agent-adaptation`） | 可写 |
| `ascend950-eval/` | 说明文档（批计划、流程、可行性） | 可写 |

## 3. 框架怎么启动

```bash
cd /home/l00970450/advanced-compiler/team/lvpanhao/xiangmu/tilelang-mlir-ascend-950
export TILEOPS_950PR_CARDS=2                      # 本机唯一空闲卡
source .agents/platforms/toolchains/ascend/env-950pr.sh
check_950pr                                       # 四项自检
opencode                                          # 选 tilelang-op-conductor
```

发需求模板：

```
迁移 <算子名> 到 Ascend950PR。

$GPU_REPO=/home/l00970450/advanced-compiler/team/lvpanhao/xiangmu/TileOPs
平台：ascend-950pr-9579
工具链：ascend

先只做 Stage 0/1/2，不要编译、不要运行 kernel、不要用卡。
```

**阶段与用卡关系**：Stage 0 脚手架 / Stage 1 设计 / Stage 2 检视 → **不需要卡**；Stage 3 精度起 → **必须用卡**。

## 4. 环境状态

### 4.1 本机（x86_64 服务器）

| 项 | 值 |
|---|---|
| NPU | 7 张 Ascend950PR；0/1 被他人占用、3/4 Alarm、5/6 被 ACL 封禁 → **只有 2 号可用** |
| 编译器 | CANN **9.2.0**（`/home/l00970450/zuo/Ascend/cann-9.2.0`） |
| 运行时 | CANN **9.1.0**（`/usr/local/Ascend/cann-9.1.0`） |
| GCC | 12.3.0 |
| Python | 3.12.15（venv `xiangmu/.venvs/ascend950-py31215`） |
| 包 | torch 2.13.0+cpu / torch-npu 2.13.0rc1 / tilelang 0.1.15 |
| 硬件档案 | `.agents/platforms/hardware/ascend-950pr-9579.json`（UB 248K / L1 512K / L0C 256K / L0A-B 64K，cube 28 / vector 56，dav-3510） |
| 工具链锁 | `ascend950pr-9579-x86_64-v3` |

**已验证**：无卡编译可行（`ASCEND_RT_VISIBLE_DEVICES=""`，2–3 秒产出 `executable.so`）。

### 4.2 hidevlab 独立 950PR（在线开发环境）

| 项 | 值 |
|---|---|
| NPU | 1 张 Ascend950PR，**独占无进程**；`npu-smi -t board` 显示 `NPU Name: 9579`（同 SKU） |
| 磁盘 | `/workspace` 300 G |
| CANN | **9.2.1**（compiler 与 opp 统一，`/usr/local/Ascend/cann` → `cann-9.2.1`） |
| GCC | 11.4.0 |
| Python | 3.12.16 |
| 依赖 | **安装中**：torch 2.13.0+cpu 已下完，剩 `tilelang`、`torch-npu`（本地 wheel） |

## 5. 已完成的工作

| 项 | 结果 |
|---|---|
| `PagedKVCacheWriteFwdOp` Stage 0 脚手架 | ✅ 982 秒，产出 `examples/TileOPs/tileops/kernels/attention/paged_cache_write/` |
| Stage 1 设计 | ✅ `DESIGN.md`（1088 行）+ `verify_equiv.py` + `RETROSPECTIVE.md` |
| Stage 2 检视（第 1 轮） | ❌ 不通过：2 阻塞 / 9 建议 |
| 走修订路径 A | ✅ 两条阻塞项闭环（实验裁决三件套 + 备选 B 伪代码列偏移） |
| Stage 2 检视（第 2 轮） | ✅ **通过** |
| 无卡编译冒烟 | ✅ `COMPILE_OK 2.1s`，产出 `executable.so`（与 A5 shim 路径逐字节相同） |
| L0 精度用例预检 | ✅ 本地可收集 **15 个用例**（`tests/ops/test_paged_kv_cache_write.py`） |
| 调度链核对 | ✅ op → `PagedKVCacheWriteKernel` 映射与构造参数一致，远端 pytest 会命中新 kernel |

## 6. 11 个算子状态

| # | 算子 | 现状 | 下一步 |
|---|---|---|---|
| 1 | EngramGateConvFwdOp | 已有 ascend 实现（上游在 **950DT** 上做） | 重跑正确性 + 重测性能 |
| 2 | EngramGateConvBwdOp | 同上 | 同上 |
| 3 | EngramDecodeFwdOp | 同上 | 同上 |
| 4 | RoPEFwdOp | 同上（含 neox / interleaved 两个 kernel） | 同上 |
| 5 | SSDChunkCouplingFwdOp | 同上 | 同上 |
| 6 | ChainSpeculativeSamplingFwdOp | 同上 | 同上 |
| 7 | MoEPermuteAlignFwdOp | 无 ascend kernel（有接口） | 新写 kernel |
| 8 | SharedExpertMLPFwdOp | 无 ascend kernel（有接口） | 新写 kernel |
| 9 | PagedKVCacheWriteFwdOp | 无 ascend kernel（有接口） | **正在做**：Stage 0–2 已完成 |
| 10 | MLADenseBwdOp | 交付仓缺 spec/Op/kernel | 先从 `TileOPs` 主库移植 |
| 11 | MLAVarlenBwdOp | 同上 | 同上 |

## 7. 关键口径（避免误判）

1. **canonical 目标是 950DT（32 Cube / 64 Vector），我们是 950PR（28 / 56）**。即使 kernel 效率相同，MFU 天然低约 **12.5%**，不能用 DT 的达标线卡 PR 的数字。
2. 片上容量两者**完全一致**（UB 248 KiB / L1 512 KiB / L0C 256 KiB / L0A-B 64 KiB / L2 128 MiB），所以 DT 上写的 kernel 不需要重写，只需**重新验证与重测性能**。
3. 本机编译**不需要卡**；真机执行才需要。
4. 团队规则：卡 4–6 禁用；同一时间只允许一人跑 benchmark；不动其他人的工作目录。

## 8. 交付物清单

```
交付/
├─ framework/    .opencode（conductor + 6 子代理 + 4 场景）
│                .agents（platforms/hardware/ascend-950pr-9579.*、toolchains/ascend 锁与脚本、skills、tools）
│                docs/950pr-framework-usage.md
├─ ops/          11 个算子：kernel + 注册 + 测试 + 性能记录 + DESIGN/REVIEW/perf_opt
└─ reports/      正确性证据、性能记录（标注 950PR 站点口径，非 canonical）
```

## 9. 当前卡点与下一步

| 步 | 内容 | 状态 |
|---|---|---|
| 1 | hidevlab 装依赖（torch 已完，剩 tilelang + torch-npu） | 🔄 进行中 |
| 2 | 在该机跑无卡编译冒烟，期望 `COMPILE_OK` | 待做 |
| 3 | 在该机跑 15 个 L0 精度用例（**第一次真正用卡**） | 待做 |
| 4 | 用框架批量推进其余 10 个算子的 Stage 0–2 | 待做 |
| 5 | 全部就绪后集中做 Stage 3–4 | 待做 |

远端真机精度命令（第 3 步）：

```bash
cd /workspace
tar xzf hidevlab-precision-pack.tar.gz
export PYTHONPATH=/workspace/tilelang-mlir-ascend-950/examples/TileOPs:$PYTHONPATH
export ASCEND_RT_VISIBLE_DEVICES=0
/workspace/venv-ascend950/bin/python -m pytest \
  examples/TileOPs/tests/ops/test_paged_kv_cache_write.py -q
```

## 10. 需要对方（另一台机器）知道的事

- 这边的框架与产物路径是 `/home/l00970450/advanced-compiler/team/lvpanhao/xiangmu/`，`tilelang-mlir-ascend-950` 是唯一执行环境。
- 环境是**按机器锁定**的（toolchain.lock.json 含 machine / CXX / 11 个文件指纹），换机器必须重建锁或另建站点 profile，不能直接复制。
- 950PR 硬件档案可跨同类机器复用，但要求 `npu-smi -t board` 的 `NPU Name` 为 `9579`。
- `aclrtGetSocName` 在本驱动上返回失败，型号请用 `npu-smi info -t board -i 0` 获取。
- 不要在别人的工作目录里操作；卡号使用前先 `npu-smi info` 确认。
