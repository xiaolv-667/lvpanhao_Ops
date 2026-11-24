# 950PR 算子迁移 / 优化框架使用说明

本仓库（`tilelang-mlir-ascend-950`，工作分支 `950pr-adaptation`）是**唯一执行环境**：算子迁移与优化都在这套 Agent 工作流里完成，产出的 kernel 再交接给 `TileOPs-ascend950`。

## 0. 三个目录的角色

| 目录 | 角色 | 能不能写 |
|---|---|---|
| `xiangmu/tilelang-mlir-ascend-950` | **框架 + 执行环境**（本次适配对象） | 可写 |
| `xiangmu/TileOPs` | GPU 参考仓库（迁移的源，`$GPU_REPO`） | 只读参考 |
| `xiangmu/TileOPs-ascend950` | 最终 kernel 落点（`kernels/ascend/` + `register_kernel_type`） | 可写 |
| `xiangmu/ascend950-eval` | 本批说明文档（现状 / 流程 / 可行性） | 可写 |

## 1. 一次性准备

```bash
cd ~/advanced-compiler/team/lvpanhao/xiangmu/tilelang-mlir-ascend-950
source .agents/platforms/toolchains/ascend/env-950pr.sh
```

这个脚本固化本次锁定的环境，**不要手工改**：

| 变量 | 值 | 作用 |
|---|---|---|
| `ASCEND_HOME_PATH` / `ASCEND_RUNTIME_ROOT` | `/usr/local/Ascend/cann-9.1.0` | CANN 工具链与运行时 |
| `ASCEND_NPU_ARCH` | `dav-3510` | 必须与硬件档案的 `architecture` 一致，否则 `--probe` 直接失败 |
| `ASCEND_RT_VISIBLE_DEVICES` | 默认 `0,2` | 团队只允许 0–3；**3、4 号当前 Alarm**，避开 |
| venv | `xiangmu/.venvs/ascend950` | python 3.12.3 / torch 2.13.0+cpu / torch-npu 2.13.0rc1 / tilelang 0.1.15 —— 与工具链锁逐项对应 |
| `TILELANG_CACHE_DIR` | `xiangmu/.cache/tilelang-950pr` | 独立缓存，避免不同实验互相污染 |

## 2. 三条自检命令

```bash
check_950pr                                  # 一键：平台 + 知识库 + 规范 + agent
```

或分开跑：

```bash
# ① 平台是否被正确识别（应输出 ok: ascend-950pr-9579 / ascend）
python $TILEOPS_950PR_REPO/.agents/tools/statectl.py platform \
  --hardware ascend-950pr-9579 --toolchain ascend --probe >/dev/null && echo ok

# ② 知识库是否自洽（ascend 域应为 0 failures）
python $TILEOPS_950PR_REPO/.agents/tools/kb_lint.py \
  --toolchain ascend --hardware ascend-950pr-9579 | tail -1

# ③ agent 是否注册（应看到 tilelang-op-conductor 等 8 个）
opencode debug agents | grep '"id": "tilelang-'
```

## 3. 迁移一个算子的流程

**第 1 步：起交互式 Agent**（需要在真实终端里，不能在这种非交互 shell 里跑）

```bash
cd $TILEOPS_950PR_REPO
source .agents/platforms/toolchains/ascend/env-950pr.sh
opencode                 # 选 tilelang-op-conductor（primary）
```

**第 2 步：把 GPU 源仓库指对**。`migrate-op` 的默认值不是我们的仓，必须显式覆盖：

```text
# 在 opencode 会话里发：
$GPU_REPO=/home/l00970450/advanced-compiler/team/lvpanhao/xiangmu/TileOPs
迁移 PagedKVCacheWriteFwdOp 到 Ascend950PR
```

或者直接用命令文件 `.opencode/command/migrate-op.md`，它接受 `$ARGUMENTS`（算子名）与 `$GPU_REPO`。

**第 3 步：conductor 会自动做场景路由**。它在 `migration` 场景下先执行：

```bash
python3 .agents/tools/statectl.py migration-mode \
  --source-repo {gpu_repo_root} --op {op_name} \
  --hardware ascend-950pr-9579 --toolchain ascend
```

返回 `migration_mode=harness`（当 GPU 仓是 TileOPs 同构：`manifest` + `tests/ops` + `benchmarks/ops` 且 manifest 含该算子键）时，走完整 `Stage 0 →（1→2→3）×N 函数 → 5`；否则走 `plain`（Stage 1→2→3）。

**第 4 步：产物位置**（框架自己的宿主目录，**不是** TileOPs-ascend950）

```text
examples/TileOPs/tileops/kernels/{family}/{op_slug}/
├─ DESIGN.md            Stage 1 设计与源算子解读
├─ REVIEW.md            Stage 2 检视
├─ {op}.py              Stage 3 kernel
├─ perf_opt/            Stage 4 调优记录（perf_records.jsonl / opt_log.md）
└─ .migration_meta.json
```

## 4. 优化一个已实现的算子

```text
# 在 opencode 会话里发：
请优化 examples/TileOPs/tileops/kernels/<family>/<op_slug>/<op>.py，目标 <latency 或 ratio>
```

conductor 走 `optimize` 场景（`stage_plan=[4, 回归]`），产出 `perf_opt/` 并强制回归。
优化依据的硬件预算来自**平台档案**（950PR：UB 248 KiB / L1 512 KiB / L0C 256 KiB / L0A-B 64 KiB），不是旧的 910B 兜底值。

## 5. 交接给 TileOPs-ascend950

框架的 Stage 5 集成是集成到它自己的 `examples/TileOPs/`。要落到我们的交付仓，需要一步搬运：

1. 取 `{op_slug}/{op}.py` 的 kernel 主体；
2. 落到 `TileOPs-ascend950/src/tileops/kernels/ascend/<family>/<op>.py`；
3. 类继承对应 `*FwdInterface`、`devices = frozenset({"npu"})`、实现 `refusal`；
4. 文件底部 `register_kernel_type("<OpName>", "ascend", <Kernel>)`；
5. 在 `kernels/ascend/<family>/__init__.py` 的 import 与 `__all__` 中登记。

## 6. 本次为 950PR 做的改动（提交框架时的增量）

| 类别 | 文件 | 说明 |
|---|---|---|
| 硬件档案（新增） | `.agents/platforms/hardware/ascend-950pr-9579.json` | 容量取自 CANN `Ascend950PR_9579.ini [AICoreSpec]`；核数 28/56 |
| 硬件说明（新增） | `.agents/platforms/hardware/ascend-950pr-9579-notes.md` | 取证字段、md5、与 950DT 的异同、待核对项 |
| 工具链接入 | `.agents/platforms/toolchains/ascend/profile.json` | `supported_hardware` 增加 `ascend-950pr-9579` |
| 工具链锁（重建） | `.agents/platforms/toolchains/ascend/toolchain.lock.json` | 按 x86_64 实测重写：machine/python/CXX/包版本/11 个文件指纹 |
| 原锁存证 | `.agents/platforms/toolchains/ascend/toolchain.lock.950dt-arm64-v1.json` | ARM64 950DT 原锁保留 |
| 环境脚本（新增） | `.agents/platforms/toolchains/ascend/env-950pr.sh` | 固化环境 + `check_950pr` 自检 |
| 知识域（无改动） | `.agents/platforms/toolchains/ascend/knowledge/**` | 保持框架自带的 950 知识库（12 个主题 md + 14 个 repro）原样；**未引入任何 910B/legacy 知识** |

验证结果：`statectl platform --probe` → `ok: true`（device `Ascend950PR_9579`、cores 28/56、容量 248/512/256/64/64 KiB）；`kb_lint` 两域均 **0 failures**；`standards_check` ok；8 个 agent 全部注册。

## 7. 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| `ascend is not configured for ascend-950pr-9579` | 工具链 `supported_hardware` 没登记 | 已在 `profile.json` 登记；若换型号需再加 |
| `ASCEND_NPU_ARCH: expected dav-3510, got None` | 没 source 环境脚本 | `source .../env-950pr.sh` |
| `machine: expected aarch64, found x86_64` | 用了 ARM64 锁 | 已换为 x86_64 锁；不要回退 `toolchain.lock.json` |
| `unrecognized device model` | 设备名不在任何档案的 `device_names` | 用 `npu_utils.so` 的 `get_arch()` 查真实名，再补档案 |
| `AclrtGetDeviceInfo failed ... 507899` | 驱动/CANN 版本提示 | 目前不影响查询，可忽略 |
| `kb_lint: ... select a toolchain explicitly` | 缺参数 | 显式给 `--toolchain` 与 `--hardware` |

## 8. 还没做的事（下一步）

- 尚未用一个真实算子跑通整链。建议首个算子用 `PagedKVCacheWriteFwdOp`（纯 scatter、无输出、6 个用例），跑通后再复制到其余 10 个。
- MLA 两个反向算子在本分支里没有 spec/Op（需先从 `TileOPs` main 移植），属额外一环。
