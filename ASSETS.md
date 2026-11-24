# 大件资产（不在仓库里）

仓库只放代码和配方。模型、数据集这类大件留在机器或对象存储上，**换机器时单独同步**，不要试图提交它们。

当前本机占用（`du -sh /workspace/shared_assets/*`）：

| 子目录 | 体积 | 内容 |
|---|---|---|
| `shared_assets/models/` | ~5.9T | `BAAI` `Qwen` `MiniMax` `OpenBMB` `Wan-AI` `stabilityai` `google` `nvidia` `Eco-Tech` `GAIR` 等权重 |
| `shared_assets/OssSGLangMeetup/` | ~236G | meetup 相关素材与数据 |
| `shared_assets/datasets/` | ~107G | 评测/训练数据集 |

## 为什么不能进 Git

- GitHub 单文件硬上限 100MB，仓库建议 <5GB；这里是 TB 量级。
- Git 对大二进制不做增量压缩，推一次基本等于把整个目录复制一遍，且历史不可回收。
- 权重可重新下载，数据集可重新生成，属于「可再生资产」，不适合纳入版本库。

## 换机器怎么补回来

任选一种，按现场条件：

```bash
# 1) 两台机器直连（局域网 / 跳板机打通时最快，支持断点续传）
rsync -avP --partial --info=progress2 user@old-host:/workspace/shared_assets /workspace/

# 2) 走对象存储（华为云 OBS / S3 等），先推后拉
obsutil cp -r -f /workspace/shared_assets obs://<bucket>/lvpanhao/shared_assets/
# 新机器：
obsutil cp -r -f obs://<bucket>/lvpanhao/shared_assets /workspace/

# 3) 只补需要的子集（大多数场景够用）
rsync -avP user@old-host:/workspace/shared_assets/models/Qwen /workspace/shared_assets/models/
```

## 如果确实要用 Git 管大件

那就别用普通 Git：`git lfs` 仍受仓库配额限制（免费 1GB 存储 / 1GB 月流量），TB 级要么买配额，
要么用专为大文件设计的存储（OBS/S3 + 清单文件）。**推荐做法是在本仓库只记录清单**：

```bash
# 生成资产清单（文件名 + 大小 + sha256 前 16 位）入库，体量小的多
find /workspace/shared_assets -type f -size +100M \
  -printf '%s\t%p\n' | sort -rn > assets-large-files.tsv
```

这样即使机器全丢，也能照着清单把该下的权重补齐。
