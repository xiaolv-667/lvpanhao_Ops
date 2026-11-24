# lvpanhao_Ops — Ascend 950PR 算子迁移工作区

把「能重跑的环境」做成仓库：**代码 + 脚本 + 环境清单**入库，环境本体、模型和编译产物按需重建。
新机器 `git clone` 之后跑一条 `restore.sh` 就能回到可用状态，不用再花半天配环境。

- 硬件：Ascend950PR（`npu-smi` 报 `NPU Name: 9579`，单片 28/56 核）
- 软件：CANN 9.2.1 / python 3.12 / torch 2.13.0+cpu / torch-npu 2.13.0rc1 / tilelang 0.1.15
- 完整版本指纹：`env/versions.txt`、`env/env-manifest.json`、`env/requirements-freeze.txt`

## 快速恢复（新机器）

```bash
git clone https://github.com/xiaolv-667/lvpanhao_Ops.git
cd lvpanhao_Ops
bash restore.sh            # 装依赖 + 生成本机工具链锁 + 无卡编译冒烟
```

`restore.sh` 依次做：前置检查 → `setup.sh`（创建 venv、装 torch / torch-npu / tilelang、编译冒烟）
→ `setup-framework.sh`（采集本机指纹、写 `toolchain.lock.json`）→ 与 `env/env-manifest.json` 比对指纹。

- 只检查不装包：`bash restore.sh --no-install`
- 还想源码编译框架（拉 tvm / AscendNPU-IR）：`bash restore.sh --submodules`

指纹不一致是**正常**的（换机器必然不同）：kernel 代码与设计结论可复用，但**编译产物和性能数字必须在本机重做**。

## 目录

| 路径 | 作用 |
|---|---|
| `tilelang-mlir-ascend-950/` | 框架本体（conductor + 子代理 + 场景 + 950 知识库 + `examples/TileOPs` 算子实验） |
| `TileOPs/` | GPU 上的上游算子库，迁移的源（`$GPU_REPO`） |
| `TileOPs-ascend950/` | 交付目标仓（kernel 落点 + `register_kernel_type` + 验收） |
| `ascend950-eval/` | 批计划、迁移流程、可行性文档 |
| `gpu_mirror_950pr/` | 950PR 侧的镜像/工作负载对照 |
| `wheels/` | `torch_npu-2.13.0rc1` 的 wheel（PyPI 拿不到时用） |
| `env/` | 环境指纹、依赖锁、版本记录 |
| `setup.sh` | 装 python 依赖 + 无卡编译冒烟 |
| `setup-framework.sh` | 生成本站点工具链锁并验证平台 |
| `env-fingerprint.sh` | 环境指纹采集（判断两台机器是否等价） |
| `restore.sh` | 上面几步的编排，克隆后一键跑 |
| `configure-deepseek.sh` | 把本机 atomcode 配成 DeepSeek provider |
| `compile_smoke_npu.py` | 不占卡的编译冒烟 |
| `SETUP-ON-NEW-MACHINE.md` | 迁移到新机器的完整说明与差异分析 |
| `PROJECT-STATUS.md` | 项目全貌与算子状态 |

## 哪些东西不在仓库里

仓库刻意**不存**大件，它们按下面的方式重建或单独同步：

| 内容 | 体积 | 处理方式 |
|---|---|---|
| `venv-ascend950/` | 1.4G | 不入库，`restore.sh` / `setup.sh` 现场创建 |
| `/usr/local/Ascend`（CANN 9.2.1） | 13G | 随驱动盘安装，不入库 |
| `shared_assets/`（模型 + 数据集） | ~6.3T | 不入库，见 [`ASSETS.md`](ASSETS.md) |
| `**/reports/`、msprof 产物、`kernel_meta/` | 数百 MB | 不入库（生成物，跑一次就有） |
| `.deepseek.key` / `.env` | 几十字节 | 不入库（密钥）；本机自行放置 |
| `3rdparty/{tvm,AscendNPU-IR,AscendNPU-IR-Dev}` | — | 空目录，仅源码编译时需要，`restore.sh --submodules` 拉取 |

## 真的要用卡

```bash
export ASCEND_RT_VISIBLE_DEVICES=0          # 选一张空卡，npu-smi info 看占用
export PYTHONPATH=/workspace/tilelang-mlir-ascend-950/examples/TileOPs:$PYTHONPATH
/workspace/venv-ascend950/bin/python -m pytest \
  /workspace/tilelang-mlir-ascend-950/examples/TileOPs/tests/ops/test_paged_kv_cache_write.py -q
# 期望 15 passed
```

编译和设计阶段不占卡（`ASCEND_RT_VISIBLE_DEVICES=` 置空即可），只有精度与调优阶段需要独占。

## 启动迁移框架

```bash
cd tilelang-mlir-ascend-950
source .agents/platforms/toolchains/ascend/env-950pr.sh
check_950pr
opencode            # 选 tilelang-op-conductor
```

## 常见问题

| 现象 | 处理 |
|---|---|
| `fingerprint differs from ...` | 锁不是本机生成的，重跑 `bash setup-framework.sh` |
| `torch-npu` 装不上 | 确认 `wheels/` 下有 cp312 的 wheel，且用 `--no-index --find-links wheels` |
| `unrecognized device model` | `npu-smi info -t board -i 0` 看 `NPU Name` 是否为 9579 |
| `aclrtGetSocName` 失败 | 该驱动特性，型号统一用 `npu-smi -t board` 取 |
| `Agent not found: "tilelang-op-conductor"` | 必须在 `tilelang-mlir-ascend-950/` 内启动 opencode，中途别换目录 |
| `CXX: None` 告警 | 编译走 `bishengir`，不经宿主编译器；涉及 host 编译时再设 `CXX` |
| msprof 报 numpy 版本 | `mindstudio-kl` 要 numpy<=1.26.4，实际 2.5.5，只影响 profiler |

更细的说明见 `SETUP-ON-NEW-MACHINE.md`。
