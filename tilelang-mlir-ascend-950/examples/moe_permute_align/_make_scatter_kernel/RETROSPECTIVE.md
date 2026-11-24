## Stage 1 (design) Retrospective — first_design — 2026-11-23

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Design-flow | Gate 1 的 `S1-EQUIV-EXEC` 用字面量 `EQUIV_FAIL` 判负，而 skill 模板要求写「`EQUIV_FAIL` 项：无」这类小结——如实记录「无失败项」反而被判失败 | `statectl gate 1` 首轮输出 `S1-EQUIV-EXEC`（DESIGN.md §1.6.1）；`gate_lint.py:322` 只看是否含该子串 | `tilelang-op-design/templates/design-template.md` 的 §1.6.1 结果表下方改为要求写「失败项：无」，并注明禁止出现该字面量；或让 `gate_lint.py` 只匹配「| EQUIV_FAIL |」表格行 | R |
| Info-source | `S1-REF-PATHS` 把 `path.py::function` 形式整体当作路径 token（`REF_PATH_RE` 不排除 `::`），导致引用函数级的正确佐证被判「路径不存在」 | `gate_lint.py:45-47`、`:72-79`；首轮 3 条 `S1-REF-PATHS` 失败 | `gate_lint.py` 的 `REF_PATH_RE` 追加在 `::` 处截断；或在 skill 中写明「引用函数请把路径与函数名分开写」 | R |
| Design-flow | 设计期无法跑真机（唯一卡被别的算子 Stage 3 占用），而 `hardware-core-split-strategy.md` 要求「物理核数实查须记录查询代码与返回值」——只能用平台记录 `hardware.observed_cores` | DESIGN.md §5.5 ② 引用 `statectl platform` 返回的 `core_source`（含 `rtGetAiCoreCount()=28` 与读取日期） | `_shared/standards/core-split-strategy.md` §1 要素② 补一句：设备被占用时以 `statectl platform` 的 `hardware.core_source`（含运行期探针记录）为实查凭据，并注明未二次探测的原因 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| ascend 后端 `tl.cumsum` **无实现**（编译期直接 raise），前缀和类算子必须手写或改算法 | D | `DESIGN.md` §3.4 / §11.1 K4、K8；报错原文 `tl.cumsum requires a target-specific implementation, but no cumsum implementation is registered for {"kind":"ascend",…}` | `cd examples/moe_permute_align/_make_scatter_kernel && ASCEND_RT_VISIBLE_DEVICES="" python history_version/design_probe_primitives.py`（K4/K8 段） | tilelang 0.1.15（site-packages）+ CANN 9.2.1 + Ascend950PR_9579（无卡编译） | `.agents/platforms/toolchains/ascend/knowledge/capability-gaps.md` |
| SIMT VF 内 `T.atomic_add(__gm__ int32, 1, return_prev=True)` 可编译为 `asc_atomic_add(&x, 1)`，是 Ascend 上做 bump-allocator 的可行原语（全局原子 G2S 有官方 SIMT API，`__NPU_ARCH__==3510`） | D | `DESIGN.md` §3.4 / §11.1 K2、K6、K9；`/usr/local/Ascend/cann-9.2.1/asc/include/simt_api/device_atomic_functions.h:31` | 同上探针 K2/K6；`python -c "import sys;sys.path.insert(0,'history_version');import design_probe_primitives as P;P.k9_factory_form()"` | 同上 | `.agents/platforms/toolchains/ascend/knowledge/execution-patterns.md`（新增 PL-950-simt-atomic-scatter 条目候选） |
| store-mode 原子（`T.set_atomic` + `T.copy`）**不能**与 `T.Task` 内的向量语句共存：`T.Task supports exactly one Ascend hardware pipe … mask=80` | D | `DESIGN.md` §11.1 K5 报错原文 | 同上探针 K5 段 | 同上 | `.agents/platforms/toolchains/ascend/knowledge/capability-gaps.md` |
| 原子 + 逐 lane 数据相关 GM 散射写可在同一 `T.Parallel` 体内编译；编译器会**自动**为 GM 索引插入值域 guard，且原子 `else` 分支回填 `0`——因此「写」必须与「原子」同处一个谓词内，否则非法 lane 会写坏槽 0 | P | `DESIGN.md` §9.2 第 1 条 / §11.1 K6、K9 降级码 | 同上探针 K6/K9 | 同上 | `pattern-library`（散列/散射类模式）或 `tilelang-vector-skill` 的 SIMT 陷阱清单 |
| MoE permute-align 类 op 的 expert 内顺序在契约层未定义（测试按 expert 段归一化后比对）⇒ 可用单遍原子 bump 取代「直方图+前缀+二趟写」，流量与遍数为下界 | P | `DESIGN.md` §0.1 / §1.6.0 R2 / §1.6.1 O1；`examples/TileOPs/tests/ops/test_moe_permute_align.py` 的 `_assert_route_equivalent` | `python verify_equiv.py`（O1：8 shape × 3 到达顺序，0 违反） | 同上 | `tilelang-op-design/references/algorithm-candidates.md`（新增「索引/散射类」章节候选） |

### Transferable Lessons（可迁移教训）

- 迁移前先探针实测（无卡编译即可）目标工具链的**原语可用性**：`tl.cumsum` 在 ascend 无实现、store-mode 原子不能与向量语句同 `T.Task`；把这两条当成「源算子有前缀和/批量原子 ⇒ NPU 照抄」会在 Stage 3 才炸。
- 逐元素数据相关原子的返回值语义（`prev`）是把「计数 + 定槽」两步合成一步的关键；只有当**契约允许元素内部顺序自由**时才成立，迁移前必须到测试的比对口径里确认这一点。
- 编译器自动插入的索引值域 guard 会静默丢弃越界写，但同时会把原子 `else` 分支的值当成合法 slot——所以「原子」和「写」必须写在同一个 `if` 内。
- 分核数量对这类「每核工作量为 O(1) 条路由」的算子应按物理 AIV 数摊平（`ept` 吸收余量），不要照搬 GPU 的 `threads=512` 网格——否则 `nblocks` 会随 numel 线性超越核数。
- `@tilelang.jit(out_idx=[], target="ascend")` 的 lazy 形式（函数返回 prim_func）可直接复用 GPU 源算子的 factory 结构，只需补 `target`；这是迁移 kernel factory 的最小改动路径。

## Stage 3 (develop) Retrospective — first_impl attempt 1 — 2026-11-23

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Info-source | 平台 `execution.precision_command` 的解释器 `/usr/local/python3.12.16/bin/python3` 在本机**存在但无工具链**（`import torch` → `ModuleNotFoundError: No module named 'torch'`；`import tilelang` → `No module named 'tqdm'`），任何阶段照该记录启动设备代码都会在 import 阶段失败；实际必须用 `/workspace/venv-ascend950/bin/python` | 本会话两次 `python3 -c "import torch"` / `import tilelang` 实测输出；可用解释器 `python 3.12.15 / torch 2.13.0+cpu / torch_npu 可用` | `statectl platform` 的 `execution` 改为记录**含工具链的 venv 解释器**；若无法自动探测，则 ruleset「执行环境」段落须写明「execution 模板解释器须按本机 venv 核实后再用」 | R |
| Develop-flow | 在仓库根 `import tilelang` 会命中仓库源码包并直接失败（`ImportError: TileLang's private TVM Python package is missing ...`），且 `T.__file__` 指向非安装包；`workflow.md` 有此提示，但 `tilelang-op-develop` skill 的 Phase 2/3/4 未复述 | 本会话于仓库根首次 import 的报错原文；在任务目录下 import 后 `tilelang.__file__` = `site-packages/tilelang/__init__.py` | `tilelang-op-develop` skill 的执行约束补一条：「所有 import / 编译 / 跑测命令先 `cd` 到任务目录，并核对 `tilelang.__file__` 指向 site-packages」 | R |
| Develop-flow | ruff `SIM300`（Yoda condition）会拒绝 DESIGN 伪代码通篇使用的 `0 <= eid` 写法；服从 lint 改写为 `eid >= 0` 后，降级设备码与设计探针**完全一致**（`if ((0 <= eid) && (eid < 384))`），说明这纯属 lint 摩擦而非语义差异 | `ruff check` 首轮输出 SIM300（`_make_scatter_kernel.py:130`）；改写后重新编译的降级码仍为 `0 <= eid` | design skill 的伪代码/模板统一改用 `eid >= 0` 形式（或 `tilelang-op-design/templates` 注明 lint 会拒 Yoda 形式），避免 Stage 3 出现「守设计原文就过不了 lint」的二选一 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| SIMT VF 内 `asc_atomic_add(__gm__ int32*, 1)` 取旧值做 bump-allocator **真机成立**：32768 路由 / E=8（单游标 4096 次原子）、32768/E=384、尾块 1092 路由，合计 70 次「每次 launch 后全量校验」均 0 违反，max_diff=0 | D | `examples/moe_permute_align/_make_scatter_kernel/_make_scatter_kernel.py`（L0-3、L1-4、B-3 与 `_run` 的 per-launch 重检）；设计期仅编译级证据见 `DESIGN.md` §11.1 K2/K6/K9 | `cd examples/moe_permute_align/_make_scatter_kernel && ASCEND_RT_VISIBLE_DEVICES=0 /workspace/venv-ascend950/bin/python _make_scatter_kernel.py --level all` | tilelang 0.1.15（site-packages）+ CANN 9.2.1 + Ascend950PR_9579（dav-3510，1 卡） | `.agents/platforms/toolchains/ascend/knowledge/execution-patterns.md`（PL-950-simt-atomic-scatter 条目） |
| TRAP：`cumsum`（extent = E+1）上编译器自动插入的值域 guard 只保证 `eid < E+1`，因此源算子式 `if eid >= 0` 会让 `eid == num_experts` 的路由原子自增 `cumsum[num_experts]`（= `num_tokens_post_pad`）并把路由索引写进 padding 区，**静默**破坏上一阶段结果；显式 `eid < num_experts` 是必需项而非修辞 | D | `examples/moe_permute_align/_make_scatter_kernel/repro/TRAP-950-scatter-eid-upper-bound.py`（实测：未加界 `cumsum[8]` 16→19、3 个哨兵槽被覆写；加界后 16 且 0 覆写）；`DESIGN.md` §1.6.1 O4 | `cd examples/moe_permute_align/_make_scatter_kernel && ASCEND_RT_VISIBLE_DEVICES=0 /workspace/venv-ascend950/bin/python repro/TRAP-950-scatter-eid-upper-bound.py` | 同上 | `.agents/platforms/toolchains/ascend/knowledge/capability-gaps.md` 或 SIMT 陷阱清单（`pattern-library`） |
| 尾 lane 的「夹取索引 + `-1` 哨兵」在 ascend 上被编译器折叠为**组谓词** `(block_idx*ept + i) < ceil(numel/THREADS)`，clamp 读 `flat[min(idx, numel-1)]` 被留在谓词分支内 ⇒ 无越界读、无多余分支；`idx` 本身降级为 `block_idx*640 + i*128 + threadIdx.x` | D | `_make_scatter_kernel.py` 于 `numel=32768/E=384` 的降级设备码（`get_kernel_source()`），谓词为 `((block_idx*5)+i) < 256` 且 `32767` 夹取出现于分支内 | 同「运行」命令 + `get_kernel_source()` dump（见下方 Transferable Lessons 的实证方式） | 同上 | `tilelang-vector-skill` 的 SIMT 尾块写法清单 |
| 散射/原子类 kernel 的验证方法：① 每个 expert 段**排序归一化后与稳定参考逐元素比对**，一次覆盖「多重集相等 / 槽互异 / 段区间 / 段外哨兵」四类错误；② `repeat` 的**每一次** launch 都重检（而不是只查最后一次），才能把「原子到达顺序随机」的竞态变成可证事实——本次 70 次全检 0 违反 | D | `_make_scatter_kernel.py::_assert_route_equivalent` 与 `_run` 的重检循环；`DESIGN.md` §8.1/§8.2 的契约级口径 | `--level all`（各层）与 stress（`32768/E384 x40`、`32768/E8 x10`、`1092/E384 x20`） | 同上 | `tilelang-op-develop` skill 的验证方案章节 / `pattern-library`（散射类验证模式） |

### Transferable Lessons

- 本迁移任务其余两个函数（`_make_align_kernel`、`_make_small_batch_kernel`）同样落在 SIMT 形态：`T.Kernel(nblocks)` 只有 1-D 核网格且**不接受 `threads`**，lane 号只能在 `T.SimtVF(threads=...)` 内的 `T.Parallel` 里得到；不要照搬 GPU 源的 `T.Kernel(blocks, threads=512)`。
- 编译器插入的索引值域 guard 用的是**缓冲区 extent**：`cumsum` 的 extent 是 `E+1`，所以 `eid == num_experts` 能穿过该 guard 并原子自增 `num_tokens_post_pad`（已实测，见 `repro/TRAP-950-scatter-eid-upper-bound.py`）。凡「缓冲区多出一个语义不同的元素」（`cumsum[num_experts]`、`expert_ids` 尾块、`num_tokens_post_pad`）都必须写**显式上界**，不能依赖自动 guard。
- 原子语句的 `else` 分支会回填 `0`：必须把「原子」与「依赖其返回值的写」放在**同一个** `if` 内，否则非法 lane 会写坏槽 0。
- 任何写 `sorted_token_ids` 的分支都必须「只写自己拥有的槽」：本段的契约是只覆写每 expert 段的前 `count_e` 槽，其余位置保持 align 段写入的 `numel` 哨兵；归一化比对时「段外必须仍为哨兵」是抓越界写最灵敏的一条断言。
- 真机上每个新 shape 的**首次**调用约 2.0 s（设备端二进制装载），同进程后续热态 10–40 ms；Stage 4 计时必须先 warm-up 再采集，否则会把装载开销记成算子时延。
- 该族 kernel 的 factory 形态可直接复用设计/源算子结构：`@functools.lru_cache` + lazy `@tilelang.jit(out_idx=[], target="ascend")`（内层再套 `@T.prim_func`），返回零参可调用、以 `(flat, sorted_token_ids, cumsum)` 调用；`out_idx=[]` 表示结果全在副作用上。
- 验证散射类输出的顺序不确定性时，**不要**只比对最后一次 launch：把归一化比对放进 `repeat` 循环的每一次（本次 70 次全检），才能证明原子路径没有偶发竞态。
