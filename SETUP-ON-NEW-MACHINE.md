# 在这台机器上继续做算子迁移与优化

本文件配合 `setup-framework.sh` 使用。目标：**不再来回传文件，直接在本机跑完整的迁移/优化流程。**

## 0. 包里有什么

| 目录/文件 | 作用 |
|---|---|
| `tilelang-mlir-ascend-950/` | **框架本体**（conductor + 6 子代理 + 4 场景 + 平台层 + 950 知识库） |
| `TileOPs/` | GPU 上游客服库，迁移时的源（`$GPU_REPO`） |
| `TileOPs-ascend950/` | 交付目标仓（kernel 落点 + `register_kernel_type` + 验收） |
| `ascend950-eval/` | 批计划、流程、可行性文档 |
| `setup.sh` | 安装 python 依赖（torch / torch-npu / tilelang / pytest） |
| `setup-framework.sh` | **为本站点生成工具链锁**并验证平台 |
| `env-fingerprint.sh` | 环境指纹（判断两台机器是否等价） |
| `wheels/torch_npu-2.13.0rc1-*.whl` | torch-npu 的 wheel（PyPI 装不到时用） |
| `PROJECT-STATUS.md` | 项目全貌与 11 个算子状态 |

**不含**：`.venv`（依赖按机器重装）、`.git`（仓库历史，需要时重新 clone）、`extra-info/`（57 MB 数据转储）。

## 1. 安装 Python 依赖（只需一次）

```bash
cd /workspace
bash setup.sh
```

期望最后看到：

```
  torch      2.13.0+cpu
  torch-npu  2.13.0rc1
  tilelang   0.1.15
COMPILE_OK in ~2s
```

`COMPILE_OK` 说明工具链可用（这一步**不占卡**）。

## 2. 生成本站点的工具链锁（关键，只需一次）

框架的锁记录的是**机器指纹**（machine / python / CXX / 11 个文件 sha256）。换机器必须重新生成，否则 `statectl` 会报指纹不符。

```bash
cd /workspace
bash setup-framework.sh
```

期望最后一行出现 `"ok": true`。这一步会：

1. 采集本站点环境
2. 写入 `tilelang-mlir-ascend-950/.agents/platforms/toolchains/ascend/toolchain.lock.json`（原锁自动备份）
3. 跑 `statectl platform --probe` 验证

> 硬件档案 `ascend-950pr-9579` 不需要重建：`npu-smi -t board` 显示本机 `NPU Name: 9579`，与档案一致。

## 3. 跑真机精度（第一次真正用卡）

```bash
cd /workspace
export PYTHONPATH=/workspace/tilelang-mlir-ascend-950/examples/TileOPs:$PYTHONPATH
export ASCEND_RT_VISIBLE_DEVICES=0
/workspace/venv-ascend950/bin/python -m pytest \
  examples/TileOPs/tests/ops/test_paged_kv_cache_write.py -q
```

期望 `15 passed`。失败时区分两类：**逐字节比对不等**（kernel 逻辑问题）vs **设备/编译错误**（环境问题）。

## 4. 启动框架做迁移/优化

```bash
cd /workspace/tilelang-mlir-ascend-950
source .agents/platforms/toolchains/ascend/env-950pr.sh    # 若路径不同，按实际改
check_950pr
opencode
```

进去选 `tilelang-op-conductor`，然后发：

```
迁移 MoEPermuteAlignFwdOp 到 Ascend950PR。

$GPU_REPO=/workspace/TileOPs
平台：ascend-950pr-9579
工具链：ascend
```

## 5. 阶段与用卡

| 阶段 | 内容 | 用卡 |
|---|---|---|
| Stage 0 | 脚手架（7 个文件） | ❌ |
| Stage 1 | 设计 → `DESIGN.md` | ❌ |
| Stage 2 | 检视 → `REVIEW.md`（不通过会打回 Stage 1） | ❌ |
| Stage 3 | 写 kernel + 编译 + **L0/L1 精度** | ✅ 精度部分 |
| Stage 4 | 调优（msprof / A-B 实测） | ✅ 且需独占 |
| Stage 5 | 集成（pytest smoke→全量） | ✅ |

设计、检视、编译都可以无卡完成；只有精度执行和调优需要卡。

## 6. 这台机器与原先那台的差异（要写进验收口径）

| 项 | 原机器 | 本机 | 影响 |
|---|---|---|---|
| CANN | 编译 9.2.0 + 运行 9.1.0 | 9.2.1 统一 | 编译/运行组合不同，需各自的锁 |
| gcc | 12.3.0 | 11.4.0 | 宿主侧生成物可能不同 |
| Python | 3.12.15 | 3.12.16 | 无实质影响 |
| 硬件 | 950PR 9579（28/56 核） | 950PR 9579（28/56 核） | **相同**，片上容量一致 |

因此：**kernel 代码与设计结论可复用**，但编译要在本机重做，性能数字要本机重测。

## 7. 两个已知告警（不影响正确性）

1. `CXX: None` —— 编译走 `bishengir` 直接出设备二进制，不经过宿主编译器；涉及 host 侧编译时再设 `CXX`。
2. `mindstudio-kl 要求 numpy<=1.26.4，实际 2.5.5` —— 只影响 msprof/profiler（Stage 4 调优）。若调优时 msprof 报错，先查这条。

## 8. 常见问题

| 现象 | 处理 |
|---|---|
| `fingerprint differs from ...` | 说明锁不是本机生成的 → 重跑 `setup-framework.sh` |
| `Agent not found: "tilelang-op-conductor"` | 必须在 `tilelang-mlir-ascend-950/` 目录内启动 opencode，且中途不要换目录 |
| `unrecognized device model` | `npu-smi -t board -i 0` 看 `NPU Name`；若不是 9579，需要新增硬件档案 |
| `aclrtGetSocName` 返回失败 | 该驱动特性，型号统一用 `npu-smi -t board` 获取 |
| 卡被别人占用 | `npu-smi info` 查看进程；换一张空卡或等对方结束 |
