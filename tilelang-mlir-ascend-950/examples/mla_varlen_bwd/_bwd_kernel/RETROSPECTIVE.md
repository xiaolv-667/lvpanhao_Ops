# Stage 1 复盘（mla_varlen_bwd / _bwd_kernel）

> 逐次追加。每次 `DESIGN.md` 产出后写一节，最新在最上。

## Stage 3 (develop) Retrospective — first_impl attempt 1 — 2026-11-26（v5 epilogue 旋转落地，A5）

### 0. 结论与验收对照

| 判据（DESIGN §8.4） | 目标 | 实测 | 结论 |
|---|---|---|---|
| mixed `T_op` | ≤ 500 ms（预期 390–425） | **440.2 ms**（−21.9% vs v4 563.7） | 必达 ✅；预期差 +3.5% |
| 8x4k `T_op` | ≤ 150 ms（预期 111–121） | **139.5 ms**（−17.2% vs v4 168.5） | 必达 ✅；落"可接受须报告"档 135–152 |
| kimi `T_op` | 不回退（预期 110–122） | **140.4 ms**（−11.3% vs v4 158.3） | 不回退 ✅；落"可接受须报告"档 |
| 1×16384（H=128） | 记录 µs/it（设计 3.134） | **330.2 ms / 3.248 µs/it**（−23.6%，与探针 3.134 差 +3.6%） | ✅ 与探针一致 |
| cube（`aic_cube_share`） | 8x4k ≥33%、mixed ≥30% | 8x4k **31.4%**（v4 25.6）、mixed **32.6%**（v4 26.3） | mixed ✅；8x4k 差 1.6 pt |
| `aiv_vec_ratio` | ≤ 80% | 8x4k 76.8 / mixed 77.5 / kimi 76.5 / 16k 81.5 | ✅（AIV 已成墙钟界定者，符合设计预期） |
| MFU（378 TF/s 口径，比值换算） | 25–33.6% | 8x4k **27.2%**、mixed **23.0%**、kimi **26.9%** | 部分达成 |
| 精度 | V14 `dk/dv` 逐位 | **全部逐位**（6 组小几何 + mixed/8x4k/16k 全量；8x4k 重复 4 次逐位） | ✅ |
| 结构 | V13 偏移身份 | **A 半体 `GB` 读 B 版本（143360/149504/122880/86016），B 半体读 A 版本**；drain 奇偶分支各读一侧 | ✅（见 `repro/CONST-950-mla-bwd-v5-lag1-identity.py`） |

一句话：**设计正确、机制生效（`dk/dv` 逐位 + 偏移身份 + cube 20.8→31.4%），但收益只拿到 −11…−24%，未到投影的 −25…−34%**；mixed/16k（长 request 占主导）接近投影，8x4k/kimi（同质中短块）明显打折。按设计 §8.4 第三条如实报告，不改判据。

### 1. 做了什么（按冻结 DESIGN v5，结构零偏离）

- `_bwd_kernel` 的查询块循环：`T.Pipelined(num_stages=2)` → **`T.serial(n_half)` + 2-tile 静态展开**，半体 = `F(k) → GA(k) → EP(k) → GB(k-1)`；四个 `@T.macro`（`_fill/_scores/_epilogue/_gradients`）与原 6 段内联代码逐句对应。
- 两份**静态命名**缓冲：`q_s/qT_s/do_s/doT_s/pT_s/dsT_s` + `lse_u/dlt_u` 的 `_s0/_s1`；`k_s/v_s/kT_s`、L0A/L0B/L0C、`st_u/dp_u/p16_u/dsT_u` 仍单版本（**没有**照抄 dense 的 S2，避免 L0C 103.1%）。
- drain：`if (n_it % 2) == 1: GB(last, A) else: GB(last, B)`，`clear_accum=(k==loop_st)` 跟 `GB` 走；`dq` 行区间滞后一个迭代（设备码里 A 半体 store 带 `- 24576`）。
- packed 的 per-request 守卫 / head slab / `h_off` / `tail` 双分支 / `live` / 原子 `dq` / grid 断言**逐字未动**；`_delta_kernel` / `_dq_land_kernel` 未动。
- 新增：工厂里的 `cfg["num_stages"] != 2 → ValueError` 守卫（`num_stages` 语义改为"显式版本数"，见下 §2 的坑）；L0 新增 **L0-25**（drain/奇偶/`loop_st>0` 覆盖）。

### 2. 本轮踩的四个坑（都可迁移）

| 坑 | 现象 | 根因 | 教训 |
|---|---|---|---|
| **生成码按单元分节** | 第一版"偏移身份"检查器把 `GB` 源与 `F/EP` 目的按**行号**比，结果 A 半体写成 4 块、`EP` 的目的根本没进集合 | kernel 体先发 `if ASC_IS_AIC {...}` 再发 `if ASC_IS_AIV {...}`，**文本顺序只在同一单元内是程序顺序** | 读生成码做结构判据时必须**先按 AIC/AIV 分节**；跨单元的"谁先谁后"只能靠事件/偏移身份，不能靠行号 |
| **"`GB` 前无 `PIPE_V` 前驱 wait"是假阳性判据** | 全文件 `asc_sync_wait(PIPE_V, PIPE_M)`：v5 **9** 条、v4 **5** 条，看着像 v5 更差 | 这些 wait 全在 `ASC_IS_AIV` 节内（AIV 等自己的向量），**AIC 节两版都是 0 条** | REVIEW 5-1 的判断成立，但要按节计数；真正的判据是**偏移身份**（已转正为 repro 脚本，并用 v4 体做反例校准） |
| **profiler 原始输出写在 5 GB 根盘** | v5-mixed 的 `T_op` 先是 220.9 ms、kimi 70.0 ms（比 v4 快 3 倍，明显荒谬）；16k 直接 `Task Duration = NA` | `/tmp/opencode` 在 overlay 根盘上，`msopprof` dump 把盘写满 → 第 3/4 个 slab 的 `DeviceProf*.bin` 写失败 | profiler 的 `--output` 必须落在 `/workspace`（300 GB）；**`window_ok`/`captured_samples` 必须等于 `per_call_launches` 才能采信 `T_op`**（collect.py 里已有该字段，别只看 sum） |
| **"baseline" 是哪个产物** | 包内 `tileops/.../_bwd_kernel.py`（`_STAGES=2`）测出 8x4k `T_op` 203.8 ms / mixed 688.8，比生产记录 168.5/563.7 慢 21–22% | 生产采纳的是 `perf_opt/_bwd_kernel.py`（`_STAGES=3`），两者只差一行 | A/B 前先确认**被采纳的产物**路径；否则收益会被虚高（−32% 而不是 −17%）。本轮三个基线都测了并在报告里分开列 |

### 3. 数字（全部本机实测，证据在 `evidence/v5_perf/`、`evidence/v5_logs/`）

- 精度：L0 **29/29**、L1 **8/8**、L2 **4/4**、boundary **12/12**、G **8/8**；`[PRECISION_PASS]`。G1/G2/G3 的 dk/dv 与"逐 request 拆分"参考**逐位**，`dq_acc` ≤1.9e-6（0 cell 超 8.2）。V5 的 per-head `dq_acc` 干净。
- `T_op`（`msopprof` Task Duration 之和，roofline 指标集）：

  | workload | v4 生产（`_STAGES=3`） | **v5** | Δ | v4 的 Stage 3 文件（`_STAGES=2`） |
  |---|---|---|---|---|
  | 8x4k bf16 | 168,521 µs | **139,465** | −17.2% | 203,847 |
  | mixed bf16 | 563,670 | **440,178** | −21.9% | 688,786 |
  | kimi 4x8k bf16 | 158,324 | **140,376** | −11.3% | 193,125 |
  | 1×16384 H=128 | 432,326 | **330,244** | −23.6% | 529,356 |

  `us/iteration`：3.143→**2.579**（8x4k）、4.137→**3.220**（mixed）、3.055→**2.701**（kimi）、4.263→**3.248**（16k）。
- 管道（`--metrics full`，`aic_*_share` = 该 pipe 占本核自身时间）：
  8x4k：cube 0.2084→**0.3142**、mte2 0.3065→**0.8604**（`active_bw` 80.2→**55.5 GB/s**）、fixpipe 0.1614→0.3747、`aiv_vec_share` 0.370→**0.768**；
  mixed：cube 0.2635→**0.3260**、mte2 0.5593→0.8130（59.9→54.6 GB/s）、`aiv_vec_share` 0.448→**0.800**。
  **AIC 的空闲窗口确实消失了**（`aic_time` 117.1→77.7 ms），但换成 **MTE2 成为 AIC 的第一忙管道（86%）且实测带宽反而下降 13 GB/s** —— 这是"收益未达投影"的现场证据，也是 Stage 4 的第一线索（不是再来一次旋转）。
- 工程代价（V17）：生成源 **66,246 B**（v4 30,415，×2.18；≤70 KB 达标）、`asc_mmad` 5→**16**、`__simt_vf__` 4→**6**、`asc_sync_wait` 26→**78**、`l12l0a/b` 5→16、`gm2l1` 7→11；单次编译 ≈20 s（与 v4 同量级，≤2× 达标）。**未出现"4k 变快、16k 变慢"的 icache 分离**（16k 反而 −23.6%），故不启用"合一 VF"缓解。
- 预算（按生成码实际分配核对，修正设计表）：**L1 = 311,296 B（59.4%）**（12 块每迭代缓冲末地址 155,648 元素，与设计逐字节一致）；**UB = 155,136 B（61.1%）**（`buf_dyn_shmem` 最高字面偏移：float 38,784×4、bf16 77,056×2；`p16_u/dsT_u` 因 2-tile 展开各两套 = 24,576 B，设计表少算的正是这 24 KB，REVIEW 2-1 正确）；L0C/L0A/L0B 与 v4 逐字节相同。

### 4. 未闭合（交接 Stage 4）

1. **8x4k/kimi 只拿到 −11…−17%**：与 request 长度相关（16k 占主导的 mixed/1×16384 是 −22…−24%）。n_it 小的块里"首迭代无 `GB` + drain"的两次非重叠（设计 §6.3 E 已量化 3.6%）不足以解释全部差额，需按 `n_it` 分桶实测。
2. **AIC MTE2 带宽从 63.6 → 55.5 GB/s**（占比 0.54→0.86）：装载变"忙而慢"。候选解释（均**未验证**）：两版本把 L1 常驻集从 200 KB 抬到 311 KB 影响 L2 驻留；`lse/delta` 的 64 行小拷贝与 `q/do` 的 dn2nz 交错变密。这是 Stage 4 的第一杠杆（V=3 旋钮、填充粒度、`lse/delta` 合并）。
3. **8x4k cube 31.4% < 33%** 验收线，差 1.6 pt；与 (2) 同源（AIC 时间被 MTE2 占据）。
4. `factory_config()["num_stages"]` 现为"显式版本数=2"的**契约**（工厂里有守卫）；Stage 4 若要试 V=3，必须同时改缓冲声明与守卫，不能只改常量。

### 5. Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp |
|---|---|---|---|---|
| **判定"滞后一迭代的消费者"只能靠缓冲偏移身份，不能靠"无 `PIPE_V` 前驱 wait"**：后者要按 `ASC_IS_AIC`/`ASC_IS_AIV` 分节计数（AIC 节两版都是 0），且全局计数 v5=9 > v4=5 会给出反向结论。偏移身份 + v4 反例校准才可判定 | P | `evidence/v5_perf/v5_body.generated.cpp`、`v4_body.generated.cpp`；A 半体 `GB` 读 143360/149504/122880/86016（B 版本），B 半体读 61440/106496/131072/137216（A 版本） | `repro/CONST-950-mla-bwd-v5-lag1-identity.py`（compile-only，断言 4 组偏移身份 + drain 奇偶 + lagged `dq`，并断言拒绝 v4 体） | tilelang 0.1.15 / CANN 9.2.1 / ascend-950pr-9579 |
| **生成码的 AIC/AIV 分节会破坏"行号 = 程序顺序"的假设**：跨单元的先后只能靠事件/偏移/数据身份，做任何生成码静态判据前先分节 | P | 本文件 §2 第 1 条（第一版检查器把 A 半体写成 4 块） | 同上脚本的 `events()`/`groups()` | 同上 |
| **`msopprof` 的 `--output` 必须落在非根盘，且 `T_op` 采信前必须校验 `window_ok`**：根盘写满会让若干个 slab 的 `DeviceProf*.bin` 丢失，`T_op` 偏小 3 倍（mixed 220.9 ms vs 440.2）或解析出 `NA` | R | `evidence/v5_logs/perf_collect.log`；`collect.py` 的 `window_ok`/`captured_samples`/`per_call_launches` | `perf_opt/collect.py --out /workspace/...` | 同上 |
| **A/B 前必须确认"被采纳的产物"**：同目录下包默认 `_bwd_kernel.py`（`_STAGES=2`）与 `perf_opt/_bwd_kernel.py`（`_STAGES=3`）只差一行，`T_op` 差 21–22%，用错会把 −17% 报成 −32% | R | 本文件 §3 表的三列基线 | — | 同上 |
| **v5 旋转的实测收益表（4 workload × T_op/us-per-iteration/cube/AIV）**：长 request 主导 −22…−24%，同质中短块 −11…−17%；`dk/dv` 逐位不变（6 组小几何 + 3 个全量 workload，8x4k 重复 4 次） | D | `evidence/v5_perf/*.json`、`evidence/v5_logs/` | — | 同上 |
| **旋转把 AIC 空闲窗口消掉后，残差搬到 MTE2（0.86 share / −13 GB/s）**：AIC 时间 117.1→77.7 ms 但 mte2 忙时 35.9→66.9 ms，"忙而慢"；下一步是填充效率而非再旋转 | D | `ds-v3-8x4k-bfloat16__{prod3,v5}__full.json` 的 `aic_mte2_*` | — | 同上 |


## 2026-11-26 — 设计修订 v5（epilogue 旋转 / `GB` 滞后一迭代，revision_index=1）

### 1. 触发与事实

- 触发：Stage 4 的 `[DESIGN_LIMIT]`（`examples/mla_varlen_bwd_perf/perf_opt/perf_feedback.md`）把缺口归到
  **迭代内跨核串行**：每迭代 AIV 忙时 1.95–2.10 µs、AIC 五管道和 3.56–3.80 µs、墙钟 3.14 / 4.14 µs，
  且 AIC 各管道之间**已经重叠**（`CONST-950-cube-queryloop-stages3-knee`：3.87 > 3.14）⇒ 剩下的是
  `GA(k) → EP(k) → GB(k)` 这条**单元间**串行链。
- 生产基线（v4）：mixed 563.7 ms / 8x4k 168.6 / kimi 159.8；目标 mixed ≤500、8x4k ≤150、kimi 不回退。

### 2. 做了什么（本轮的真实路径）

1. **先读源码与证据，不先写方案**：读 `_bwd_kernel.py`（99 KB）的循环体、Stage 4 的 `z3_*.log` 三份、
   `bwd_c1_stages3.generated.cpp`。**关键发现**：`num_stages` 的版本下标 `(kq-loop_st)%3` 同时出现在
   **生产者与消费者两侧** ⇒ 它只加宽 WAR，`EP→GB` 的 RAW 边一点都不动。这一步把「加缓冲」和「改顺序」
   分开了，也直接决定了后面不能走 `annotate_buffer_versions`。
2. **调研四问用在"调度族"上**（不是算法族）：列 S0–S6 七个调度候选，逐条判等价/复杂度/硬件亲和，
   淘汰 S3（同迭代互换位置，无环转 ⇒ 无收益）、S4（warp 特化，平台无线程域/无跨核 flag）、
   S5/S6（拆 kernel / 每 tile 一 launch）；S2（两级旋转）用 L0C 算术淘汰（`Nk=96` 下 103.1%）。
   **这一步避免了照抄 dense 的 S2 结论**（dense 的 Stage 4 建议的正是 S2 + `dP` 折叠，见 §9.1 风险 17）。
3. **写探针、跑设备、拿到数**（约 40 min）：`design_probe_epilogue_rotation.py` 同进程实现 v4 体与 v5 体，
   `--compile/--numeric/--parity/--time` 四个入口。4k：**3.167 → 2.092 µs（−33.9%）**；
   16k：4.267 → 3.134（−26.5%）；且 `dk/dv` **逐位相同**（5 组几何）。
4. **拿反例校准检查器**：给 `verify_equiv.py` 新增 `check_a5()`（调度代数），并用三个**人为注入的 bug**
   （drain 版本错 / 删 drain / `GB` 读当前版本）验证它真的会 BLOCKED（160 / 320 / 14,880 处违反）。

### 3. 踩过的坑（供后续同类任务直接复用）

| 坑 | 现象 | 根因 | 教训 |
|---|---|---|---|
| **drain 只写了一侧** | `q=kv=[192,192]` 的第二个 key 块 dk=5.7e-2（首个 key 块正常） | 2-tile 展开后 `n_it` 可奇可偶，最后半体是 A 或 B；首版只在 `n_it` 奇数时补 drain | **"末迭代的滞后消费者"必须按迭代数奇偶分派**；并且检查器必须被已知 bug 校准过才算有效 |
| 以为 `annotate_buffer_versions` 能表达 lag | 一度把方案写成「加版本 + `T.Pipelined`」 | 版本下标绑定当前迭代（设备码证据） | 判定"某机制能否表达某种依赖"要落到**生成代码**，不能停在 API 名字 |
| 以为旋转需要 L0C 双版本 | 差点照抄 dense 的 `dP` 折叠 | 混淆 S1（只滞后 `GB`）与 S2（`GA` 提前） | 先把**依赖图**画出来再谈缓冲：S1 里 `st_c` 的读写同迭代 ⇒ 单版本 |
| 把「AIC 忙时之和」当墙钟 | 一开始算不出"谁在空闲"（3.87 > 3.14） | 多管道可并发，和可以大于墙钟 | 判断跨单元串行要用「AIV 忙时 vs 墙钟」+ 依赖链，不要用管道和 |

### 4. 数字（本轮，可复核）

- 单价（`--time`，3 次取最小）：4k 3.167→**2.092**、16k 4.267→**3.134**、512+4k 4.191→**2.783** µs/迭代。
- 代码体积：31,435 → **61,913 B（+97%）**；`__simt_vf__` 4→6；动态指令数/迭代不变。
- 预算：L1 204,800→**311,296**（59.4%）、UB 129,536→**130,048**（51.2%）、L0C/L0A/L0B **不变**。
- 投影：mixed **390–425 ms**、8x4k **111–121**、kimi 110–122；MFU 25.0 / 33.6 / 32.7%。
- 等价性：`verify_equiv` A1–A5 **ALL_PASS**；设备 parity `dk/dv` **5/5 逐位**。

### 5. 未闭合（交接给 Stage 3/4）

- **16k 的单价分层只消掉一部分**（3.134 仍是 4k 的 1.50×）——机理未定，方向是填充粒度/工作集驻留。
- **V=3 旋钮未实测**（装载缓冲三份，L1 75.0%）。
- **icache 风险**：+97% 的静态代码；判据 V17 + 逐 workload `T_op`，缓解阶梯见 §9.1 风险 15。
- `factory_config()["num_stages"]` 语义改为「显式版本数」，Stage 3 需同步测试期望（**不是契约变更**）。

### 6. Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp |
|---|---|---|---|---|
| **«加缓冲» ≠ «能重叠»：buffer versioning 是 WAR 手段，RAW 链只能改语句顺序**。`annotate_buffer_versions` / `num_stages` 在本族 kernel 上 Z3 `unsat` 且只按当前迭代下标加版本（生成码 `(kq-loop_st)%3` 在生产者与消费者两侧同式） | P | `mla_dense_bwd_perf/perf_opt/logs/probes/z3_{dense,varlen,scheduler_probe}.log`、`bwd_c1_stages3.generated.cpp`；v5 探针 `--time` 三点 | `history_version/design_probe_epilogue_rotation.py --compile/--time` | tilelang 0.1.15 / CANN 9.2.1 / ascend-950pr-9579 |
| **把「跨核 epilogue」的消费者滞后一迭代，是 5-GEMM 显式 L0 结构上最便宜的一次结构优化**：只花 L1 两版本（+52%），换 −26…−34% 墙钟；`dk/dv` 逐位不变 | D | v5 `--time` / `--parity`；§1.6.4 F | 同上 `--parity` | 同上 |
| **手写环转的正确性可用「调度代数模拟」在 CPU 上判定**（版本分配 / drain / `clear_accum`），且检查器必须用注入 bug 校准；设备侧再用 `torch.equal` 判 `dk/dv` | P | `verify_equiv.py::check_a5`（320 调度点 / 10,400 `GB` 事件；注入 bug 分别产生 160/320/14,880 处违反） | `python verify_equiv.py` | 同上（CPU，无设备） |


## 2026-11-25 — 设计修订 v4（Cube 化：L1 路径 → 显式 L0 + `trans_A` 复用，revision_index=1）

### 1. 触发与事实

- 生产基线：`ds-v3-mixed` 1728.7 ms（Vector MFU 47.05%）、`ds-v3-8x4k` 490.6 ms、`kimi-k2-4x8k` 470.5 ms；
  msprof：`aic_cube_ratio 8.3%` / `aiv_vec_ratio 73.5%`（同一 `_bwd_main_kernel`）。
- 根因分解（dense 同结构探针 + 本仓分段实测）：内层迭代 5.95 µs 中 **3.75 µs（63%）是向量侧跨步转置
  `ds_u[j,i]`**，`gemm5`（5 个 NT GEMM）只有 0.587 µs（10%）。也就是说 **Cube 不是在算，是在等**。
- 该 scatter 的来源是 v3 的一个**判定错误**：把源算子 `transpose_A=True` 判为「NPU 不可用（L1 路径只支持 NT）」，
  于是在向量段造第二份 `dS`。而 dense 的实测证明**显式 L0 路径支持 `trans_A`**（`asc_copy_l12l0a_transpose`）。

### 2. 本次做对的四件事（可复用）

1. **先定"以谁为基线"再优化**：dense `MLADenseBwd` 与本源算子同一工厂、同一设备码形态、已真机 + 生产验证
   （150.7 ms / cube 25.2%）。把「复用 dense C2 核」写成 §0.8 的 10 项复用 + 5 项必须改，比从零推一遍
   快一个数量级，也避免了「重新发明一个更差的形态」。
2. **packed 的差异必须先找出来再动手**：dense 用 4-D batch 视图根治 `dk/dv` 的 request 隔离，packed **没有
   `S` 维可用**（单张 `(T,H,D)`）⇒ 必须换成核内逐 request 守卫（R7 双分支）。抓住这一条，其余 4 处差异
   （解码、统计量 layout、`dq` 溢出零行、head slab）都是已知形态的重排。
3. **设备数值原型是最便宜的定心丸**：6 个用例（尾块 / `shift` 双向 / `lse=-inf` 空行 / head slab / fp16+bf16）
   在 10 分钟内跑完，误差 2.4e-05…2.6e-03（判据 3e-2），一次性覆盖了「Cube 核 + packed 语义」的全部交互点。
   比等到 Stage 3 再发现 `dq` 全错（`b_hi` 顺序）便宜得多。
4. **诚实更新既有结论**：`Nk=96` 让 grid 从 66,560 掉到 44,672，**manifest 的几何不再逼近 65,535**。
   设计里没有因此删掉 slab 与断言，而是补了「为何仍必要」——因为 `tiles` 是运行期量，`auto-shrink` 几何
   （2,100 tiles → slab=31）与越界 `ValueError` 仍然触发。这类"问题暂时消失"的地方最容易埋雷。

### 3. 踩到的坑（信息源）

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Probe 工程 | `from __future__ import annotations` 会让 `@T.prim_func` 的 `T.Tensor((total_q, H, D), dt)` 注解变成字符串，`get_type_hints` 在**模块全局域**求值 ⇒ `NameError: name 'total_q' is not defined`（闭包变量不可见） | 本版 `design_probe_packed_cube_num.py` 首跑；删掉该 future import 即通过 | 设计/开发 skill 的探针模板注明：prim_func 所在的模块**不要**用 `from __future__ import annotations` | C |
| 测量口径 | 单价测量若把 H2D 拷贝写进计时循环（`fn(to(q), …)`），会把 `3.19 µs/迭代` 测成 `9.6 µs`（3× 误差）；张量必须先上卡再计时 | 本版同一脚本两次运行的对比（9.614 → 3.194 µs） | optimize/develop skill 的计时模板：事件计时区间内只放 kernel 调用 | D |
| 信息源核实 | 调度简报提到的 `TRAP-950-nd2nz-rows-multiple16` / `CONST-950-mla-bwd-cube-split` **不是 knowledge 条目**（全库无此 id），实际是 dense 仓的 `repro/*.py` 可执行脚本 | `grep -rn` 全 `.agents/`+`examples/` 只在 dense `RETROSPECTIVE.md` 命中 | 设计期引用前先按 id 检索；把「条目」与「repro 脚本」分开登记 | R |
| 关键约束 | `block_n/2 % 16 == 0`（UB→L1 `dual_copy` 经 nd2nz）是**硬编译约束**，v3 文档里的 `Nk=80` 回退与 dense 的 `Nk=112` 备选都不可编译（半区 40/56 行） | dense `repro/TRAP-950-nd2nz-rows-multiple16.py`（`static_assert ROWS % 16 == 0`）；本轮 P7 用 96 通过 | 设计 skill 的分块约束清单加一条：**半区行数 % 16 == 0 ⇒ block_n ∈ 32 的倍数**（可行集 ≤ 128 时只有 32/64/96/128） | D |

### 4. 数字（本版产出，供后续引用）

- 迭代数（`Nk=96, Mq=64`）：mixed **3,770,496**、8x4k **1,441,792**、kimi **1,420,032**（C1 的 Nk=64 为
  5,623,296 / 2,129,920 / 2,113,536 ⇒ **−32.9%**）。
- GM 访存（mixed）：**503.1 GB**（C1 746.8 GB，**−32.6%**）；每迭代 L1/UB 端口 325,227 B、**31.5 FLOP/字节**
  （C1 25.2，+25%）。
- 片上：L0C **221,184 B（84.4%）**、L1 204,800 B（39.1%；实测口径 229,376 B）、UB 129,536 B（51.0%）/
  双版本 166,912 B（65.7%）、L0A 49,152 B（75.0%）、L0B 61,440 B（**93.75%，余 4 KiB ⇒ L0 单版本**）。
- 单价：本轮 packed 实测 **3.194 µs/迭代**（`q=kv=[4096,4096]`, H=16, 45,056 迭代；保守上界）；
  dense 生产 **2.97 µs/迭代**（同 tile）。
- 投影：mixed **400–430 ms（4.0–4.3×）**、8x4k **153–165 ms（3.0–3.2×）**、kimi 151–162 ms；
  cube 口径 **23–27%**。

#### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| packed（THD + `cu_seqlens`）的 MLA 反向可用 dense C2 的 Cube 核直接承接：需替换的只有 5 处（融合解码、head-major 统计量、逐 request 写回双分支、`dq` 溢出零行、head slab），head slab 与 5 GEMM 的 L0 复用**无耦合**（P9 `no_slab` 的 marks 与 P7 全同） | P | `history_version/design_probe_packed_cube.py`（P7/P8/P9 3/3 COMPILE_OK）；`design_probe_packed_cube_num.py`（6/6 PASS，逐 request 切片） | 两者均在本目录，命令见 DESIGN §3.5.4 | tilelang 0.1.15 / CANN 9.2.1 / ascend-950pr-9579，2026-11-25 | 迁移 skill 的 `references/migration-analysis.md`（「同工厂兄弟算子已迁移」的复用分区范式） |
| `block_n/2 % 16 == 0` 是 UB→L1 `dual_copy` 的**硬编译约束** ⇒ packed MLA 反向的 tile 阶梯只有 32/64/96/128（`Nk=80/112` 不可编译） | D | dense `repro/TRAP-950-nd2nz-rows-multiple16.py`（`nd2nz_copy.h:59 static_assert ROWS % 16 == 0`）；本轮 P7 以 96 通过 | 同上 | 同上 | 平台 knowledge（nd2nz 形态边界）+ design skill 的分块约束清单 |
| 单价测量必须把 H2D 拷出计时区间：同一 kernel、同一 shape，含拷贝 **9.614 µs/迭代** vs 纯 kernel **3.194 µs/迭代**（3.0×） | D | `design_probe_packed_cube_num.py --time` 的两次运行 | `TMPDIR=/workspace /workspace/venv-ascend950/bin/python history_version/design_probe_packed_cube_num.py --time` | 同上 | optimize/develop skill 的计时模板 |

#### Transferable Lessons（可迁移教训）（可迁移教训）

- **同工厂的兄弟算子若已迁移并验证，先做「复用分区」再设计**：把差异（本次 5 处）逐项列出并给证据，
  比从零推导整条设计快得多，也不会退回到已被否决的形态。
- **"某个约束暂时不再触发"不等于可以删掉它**：`Nk` 变大让 grid 变小，但 `tiles` 仍是运行期量，断言与
  slab 必须留着，并写明"为何仍必要"（否则下一版会有人删掉）。
- **判定 API 可用性时以「平台已核验的路径」为参照，不要以「自己上次的实现」为参照**：v3 把 `transpose_A`
  判死，代价是 63% 的向量时间；而证据（dense 探针 2 + 设备码）当时就已经存在。
- **数值原型要逐 request 切片比对**，不能只看全张量 max——跨 request 泄漏的误差会被真实数值掩盖
  （dense 的 3-D 视图在因果下 5/5「通过」就是这种假通过）。
- **探针脚本自身也是代码**：`from __future__ import annotations`、计时区间含 H2D 这两类坑都会让「证据」失真，
  所以探针的第一次运行结果要先自检量级合理性。


## 2026-11-24 — 设计修订 v1（单次 launch grid 上界 65,535 的静默少算）

### 1. 触发与事实

- 实施后复核（`/tmp/opencode/MLA_VARLEN_BWD_8x4K_FINDINGS.md`，只读测量）证明 `ds-v3-8x4k`（B=8/H=128/L=4096）
  的 `grid = 66,560` 只跑了 1,024 个 CTA：20.75 ms 是**少算**（真实 ≈488 ms），request 1..7 的输出全 0。
- 关键事实（本任务补全）：`grid % 65536 != 0` 时**不报错**，只执行 `grid % 65536` 个块；整倍数时才 abort。
  v0 的设计只把「动态 grid + `live`」当静态边界处理，**没有把「块索引 16 位」列为技术约束**。

### 2. 为什么 v0 的验证没发现（结构教训）

1. 全部 correctness 用例 `H ≤ 32` ⇒ `grid ≤ 32×tiles`，物理上碰不到红线；manifest 的 H=128 case 恰好是唯一越界几何。
2. 该 bug 的**表现是「更快」**（截断），所以任何 latency 门槛/sanity 都不会失败；反而是「691 TFLOPS」这种
   不可能值应当直接判失败（G4 tripwire 的由来）。
3. 若参照解与待测解跑同一条大 grid 路径，两者会**一起错**——本轮用「B=1、grid 远低于上界」的同内核 launch
   作参照才定位到问题。**教训：边界 bug 的参照必须在不同 grid 量级上取。**

### 3. 处置（设计层）

- 采用 **head slab 多次 launch**：`grid = tiles*slab`，`slab = min(32, heads, 65535//tiles)`（编译期常量），
  `tile = bid//slab`、`h = h_off[i] + bid%slab`，尾 slab 用 `if h < heads` 守卫；一个编译产物 + 运行期
  `h_off`（int32[1]）服务全部 slab。
- **发射前断言**（`launch_plan.py`，纯 Python）：`grid ≤ 65,535`、head 覆盖恰好一次、越界 raise。
- 逐项否决「调大 `block_n`」「head 折进核内串行」「运行期除数」「两 kernel 拆分」「只断言不修」五种绕开方式
  （§5.6 f，均给了定量理由）。
- 明确声明：**不改算法/分块/dtype/精度路径/IO 契约，不做性能优化**。

### 4. 机器验证与探针（本次实跑）

- `verify_equiv.py` 新增 **A4**：8 组几何逐 bid 穷举 **568,904 个逻辑任务**，缺失 0 / 重复 0 / 越界 0，
  单次 `grid` 全部 ≤ 65,535；越界参数 2/2 raise；边界 2/2 通过（65,535 合法、自动缩小到 slab=31）。
  输出 `EQUIV_SUMMARY ALL_PASS`（A1×2 + A2 + A3 + A4 共 5 行）。
- 新增编译探针 **P6**（`history_version/design_probe_grid_slab.py`，heads=3/slab=2 的**不整除**几何）：
  `COMPILE_OK`，28,381 B，设备码 `block_idx >> 1`、`h_off[0] + (block_idx & 1)`、`if (h < 3)` 包住 body，
  13 个既有 mark 全保留 ⇒ slab 化没有改变核内形态。

### 5. 遗留（移交 Stage 3 / 集成）

1. Stage 3 第 0 步：内联 `plan_launch`、加 `h_off`/`head_slab`、跑 §8.4 **G5**（纯 host 侧）。
2. §8.4 G1–G4 必须在 Stage 3 通过（H=128 的逐 request 拆分对照是本次的唯一决定性回归）。
3. 同族 launch（`_delta_kernel` / `_dq_land_kernel`）的运行期 grid 也应在 wrapper 里复用同一断言（§9.1 风险 18）。
4. 性能代价（launch 边界 L2 冷启动、`head_slab` 是否提到 64）留 Stage 4 用 `msopprof` 实测裁决。


## 2026-11-23 — 首次设计 v0（packed 分支，复用 dense 已验证核心）

### 1. 任务形态与策略

- 迁移任务（harness 子模式），`_bwd_kernel` 的 **packed 分支**（`MLAVarlenBwdOp`）。
- 关键判断：读源码后确认 `_bwd_kernel`（`bwd.py:109-264`）是 dense/packed **共用工厂**，`dense` 只切换
  `stat_shape`、`num_tiles`、`(request, n0)` 解码、`lse/delta` 取址。dense 版已通过检视 + 真机（L0 13/13、
  L1 6/6），因此策略定为：**复用 dense 的已验证核心，只重推 3 处 packed 差异**（任务显式要求），
  而不是从零设计。这直接决定了文档结构（§0.8 复用/差异分区 + §11 对照索引）。

### 2. 三处差异的推导要点（价值最高处）

| 差异 | 关键发现 | 处置 |
|------|---------|------|
| ① 变长块枚举 | 源的 `GroupTiling.cumsum_offsets` 在 `num_groups > 8` 时走 **warp shuffle 前缀扫描**（`T.shfl_up` + kernel 级 thread extent），NPU 上根本不存在该原语；不能「照搬」 | 重推为**融合式 running-prefix 解码**：一次 `T.serial(batch)` 同时完成前缀和与 first-match decode，全部 local 标量、无共享块表；语义与 `decode` 的二分逐条对齐 |
| ② head-major `lse/delta` | 保持 `[H, T_q]` 反而最优：每次装载是 **256 B 连续段**（设备码 `asc_copy_gm2ub_align(..., 1, 256, ...)`）；转成 dense 的 `[B,H,S]` 只会多一次 host 侧重排 | 不转置 |
| ③ 逐 request 边界 | **最大的隐性坑**：平台 OOB clamp 只按**张量 extent**（设备码 `min(64, max((total_kv-kv_start)-n0,0))`），不认 request 边界 → `dk/dv` 整块写回会把**精确 0 覆盖到下一个 request 的行**（`dk/dv` 是赋值）。而 `dq` 因为掩码使越界列 `dS=0`，整块**原子加 0** 反而无害 | `dk/dv` 双分支（内部整块快路 / 尾半块逐 lane 守卫）；`dq` 保留整块原子并用 A3 把它变成可验证的不变式 |

### 3. 探针过程中的实际教训

1. **`@T.prim_func` 体内不能调用普通 Python 函数承载循环**：首版探针把计算体抽成普通函数，报
   `TypeError: 'ForFrame' object is not iterable`（`for ... in T.Pipelined(...)` 未被 AST 变换）。
   改 `@T.macro`（与源 `GroupTiling` 的写法一致）后一次通过。**这条已写进 §3.5.4 反例表**，避免 Stage 3 重犯。
2. **`T.clear` 不支持 L0C**（`InternalError: Unsupported scope shared.l0c`）：原本想给退化请求（`len_q=0`）
   做防御性清零，只能改设计——用 `loop_ed = max(ceil(len_q/64), loop_st+1)` 保证循环非空，让一次全掩码
   迭代产出精确 0。这是**设计因探针而改变**的实例（不是事后补说明）。
3. **动态 grid + 混合 kernel 居然直接编译通过**（P1/P2/P5）：这是开工时最大的未知项（平台先例只有纯 SIMT
   的动态 grid）。编译级通过不代表运行期通过，但足以让主方案成立，并把「静态 grid 上界」作为已编译的
   备用方案落纸（P5），而不是空口承诺。
4. **符号长度区段 copy 也能编译**（P3）：因此 `dk/dv` 尾块有两条可行路径，设计里把 P3 记为回退形态，
   主选仍是双分支（少一份「运行期长度参与 DMA」的未知量）。

### 4. 机器验证（verify_equiv.py）的两次迭代

- 第一版 A3 只枚举了**裁剪后**的块（`j` 用 `min(m0+64, len_q)`），于是 `spill_entries=0`、`acc_bad=1`
  （`-0.0 + 0.0 = +0.0` 的位模式差异被判成违反）→ `EQUIV_BLOCKED`。
- 修正两处：块按**完整 64 行**枚举（GEMM 实际就是整块跑），累加器比较改为**数值相等**并单独统计零号翻转。
  重跑 4/4 `EQUIV_PASS`（A2 1.157 亿元素、快路 13734/28261；A3 30528 个溢出差分元素 + 3816 个溢出 dq 行
  全为精确 0；累加器 4102 样本 1 个零号翻转，数值等价）。
- 教训：**不变式验证的枚举口径必须与 kernel 的分块口径一致**（整块 vs 裁剪块），否则验证会「假通过/假失败」。

### 5. 量化结论（供后续阶段对照）

- 代表 workload `ds-v3-mixed`：可见 score 对 22.9e9、FLOPs 38.2 T（分块实际 38.5 T，+0.9%）、
  迭代 5,623,296、live 块 64,512。
- 每迭代 L1 端口 **279280 B**（24.5 FLOP/字节）→ 全量 **1.571 TB**；GM 视图 **744 GB**
  （q/do 双读 460.7 + `dq` 原子 276.4 + k/v 4.2 + dk/dv 2.6），L2 驻留后 HBM ≈ 17.2 GB。
- 容量：L0C 62.5%、L1 32.8%、UB 35.7%（沿用 dense 真机通过的半区 buffer 清单，逐字节相同）。
- 分核：逻辑 33,024…66,560 ≫ 28 核 → 「逻辑块超发」，并用「最长块 0.0046%」量化长尾；
  偏离 `core-split-strategy.md` 的极大规模标准分支，写明了复评触发条件。

### 6. 留给后续阶段的重点（按优先级）

1. **R1/R2 探针**：动态 grid 的运行期行为 + host 侧 grid 实参传递（两个不同 `T_kv` 各编译一次）。
2. **R3/L0-9**：request 间污染回归（B=2 且尾块请求在前）——这是本函数最可能出错的地方。
3. **R4**：溢出零行的设备确认（同一数据 vs 「只有 request 1」逐位对照）。
4. 若动态 grid 失败 → 直接切 P5 的静态 grid（已编译），不要把时间花在别的方向上。
5. Stage 4 的流水：packed 的 `loop_ed` 无法静态化（`ceil(len_q/64)` 运行期），流水收益必然低于 dense，
   优先考虑 L1 侧双版本而不是 UB 侧。

## Stage 2 (design review) Retrospective — 2026-11-23 — 结论 通过（阻塞 0 / 建议 7）

> 本节复盘**检视流程自身**；设计内容的问题已随 `REVIEW.md` 流转给 Stage 1/Stage 3，不在此重复。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Mechanical-check | `design_calc_check.py` 只按「单块 `block_M×block_N×sizeof(accum)`」核 L0C（本案输出 `block 64x64 x fp32 accum = 16.0 KB vs L0C 256 KB`），**不核多累加器之和**；本案真正的约束是 5 个累加器合计 163840 B（62.5%），机械位完全没覆盖。L1 层则没有任何 check 项（本案 L1 172032 B 是必核项）。两处都靠人工逐 buffer 相加；若某设计把 5 个累加器配到超容量，机械检查仍会 pass | `/workspace/venv-ascend950/bin/python3 .agents/tools/design_calc_check.py --design … --task-dir …` 输出（`l0c_budget` detail 仅单块；无 `l1_budget` 项）；DESIGN.md §4.6（5 累加器合计）与 §4.5/§4.6（L1 表） | 解析器：① `l0c_budget` 改为累加 §4.6 表中 L0C 行的**全部**条目；② 增 `l1_budget` check（读 `design_budget.capacity_bytes.L1`）；③ `core_split` 支持「表名/列名含『逻辑核数』」的核数表（本案仍 skip） | R |
| Dimension-gap | 等价性机械位（D-1）只要求「重跑 + 与内嵌表一致」，**不要求审阅脚本是否真的验证了该命题**——一个把 candidate 与 baseline 写成同一表达式的脚本也能打印 `EQUIV_PASS`。本次靠人工读 `verify_equiv.py`（A2 的 `assert vis.all()` 是真判据、A3 是结构性推导而非自证）才确认非伪造 | `verify_equiv.py:182-196`（A2 快路断言）、`214-274`（A3 结构检查）；DESIGN.md §1.6.1 内嵌表 | 「机械复核（D-1）」段补一句：`verify_equiv.py` 必须审阅其**判据构造**（候选与基线是否独立求值、是否有 `assert` 型充分条件、容差定义是否可证否），只重跑不读脚本不算完成 | P |
| Evidence-scope | 设计文档的设备码引用是**表达式级原文**（如 `for (g=0; g<3; ++g) { ntg = … }`），本 Agent 先用 regex 按原文匹配 P2 源码，全部 0 命中（实际存在，但变量名/格式为 `for (int32_t g …)`、`cnt[0] += ntg`）；差点误判为「设备码证据不实」。dense Stage 2 复盘只沉淀了「计数/事件 id 会漂移」，**表达式文本本身也会漂移** | `/tmp/opencode/rv_varlen_p2.py`（按原文 regex → 0 命中）vs `/tmp/opencode/p2_src.cu:104-121`（结构存在） | 维度 0/1 的设备码核对改为固定三步：① 用**调用种类**（`asc_*` 符号）建索引；② 核**参数形态**（长度表达式是否含 `min()/max()/total_*`、相邻调用关系、是否落在原子窗口内）；③ 不比对变量名与嵌套格式。设计文档若引原文，检视时按 ①② 复核即可 | R |
| Reuse-evidence | "同源码工厂 + 复用兄弟设计已核验结论"这一形态在本任务出现（packed 复用 dense），但 skill 的维度 0/8 没有对应检查项：复用多少算够、哪些兄弟结论在变体下会失效、失效项是否被抓出。本次设计的 §0.8-A/B/C + §11.3 恰好是正确形态，检视方只能靠自建判据（亲读源工厂确认差异面 → 逐项核对 §11.3 是否覆盖全部失效结论） | DESIGN.md §0.8、§11.1-11.3；源 `bwd.py:124,133-139,176-184,214-224`（差异面只有 4 处） | 维度 0 增一子项「复用有效性」：① 复用的『同一源码工厂』主张须由源码差异面核对（列差异点并计数，与设计自述比对）；② 兄弟设计中被复用结论在变体下**必须逐条判存续**，「不再成立」的结论须有替代方案与机器证据（本案 R7/A3） | P |
| Dimension-gap | 兄弟设计（dense）在 Stage 3 实测得到的**已知容差边界**（bf16 `dS` cast 抵消放大 17×，单元素 7.8e-3 > atol 5e-3）没有随复用关系流入本设计的 §8.2/§9.1，检视时也无可循检查项；Stage 3 极可能重复诊断同一现象 | `examples/mla_dense_bwd/_bwd_kernel/RETROSPECTIVE.md` Stage 3 段（L2-3/B-4 WARN + 诊断手法）；本设计 §8.2 无该条 | 维度 6 增一子项：同一算法族的兄弟设计已有 Stage 3/4 结论时，§8.2 须承接其**容差边界与判定手法**（尤其 bf16/fp16 固有舍入），缺失记建议级 | P |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 设备码证据的「结构三元组」核对法：调用种类 → 参数形态（`min/max/total_*` 表达 + 相邻关系 + 原子窗口包夹）→ 不比对变量名/计数/id。可把「原文 regex 0 命中」这类假阴性一次排除 | P | 本次 `/tmp/opencode/rv_varlen_p2.py` 8 条 regex 全 0 命中（含必然存在的 `cnt +=`/`min(64, max((total_kv`），换「符号 + 参数形态」后 6 项一次确认（`asc_copy_l0c2gm` 的 `min(64, max((total_q-q_start)-kq*64,0))`、`asc_copy_ub2gm_align` 的 `min(32, …)`、`asc_copy_gm2ub_align` 的 `…*total_q + q_start` 等） | `cd /tmp && ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python /tmp/opencode/rv_varlen_p2.py` | tilelang 0.1.15 / CANN 9.2.1 / ascend-950pr-9579 | 本 skill（Phase 1 第 7 步 + 维度 0/1 设备码核对），并与 dense Stage 2 的「计数/事件 id 不参与判定」合并为一条 |
| 复用型迁移的检视判据（差异面计数 → 存续判定 → 失效项替代证据），避免「兄弟设计已通过」被当成免检通行证 | P | 本任务 §0.8-A/B/C、§11.3 + 源工厂 4 处差异面（`bwd.py:133-139,176-184,214-224`）的逐条核对 | 源文件 `bwd.py` 全文 Read + `grep -n "dense" bwd.py` | — | 本 skill（维度 0 迁移表增子项） |

### Transferable Lessons

- **机械检查的覆盖面要按「实际约束层」核对，不能只看它 pass**：本案 L1（32.8%）与「5 个 L0C 累加器之和」都不在机械位里，若只看 `pass` 就漏掉最该核的两个数。凡设计给出逐 buffer 表，检视方应亲手加一遍。
- **D-1 的可信度取决于脚本的判据构造**：重跑一致只证明「可复现」，不证明「验证了该命题」。看三件事——候选与基线是否独立求值、充分条件是否有 `assert`、容差定义是否可证否。
- **设备码引用按「调用种类 + 参数形态」核，不按原文串匹配**：变量名、格式化、`cnt[0] +=` 与 `cnt +=` 的差异会让 regex 全灭，制造假阴性；反过来也不会因计数/id 漂移制造假阳性。
- **「兄弟设计已通过」不是免检**：必须先亲读源码确认差异面有几处、与设计自述是否吻合，再把兄弟结论逐条判存续；失效项（本案：平台 clamp 写回、整块 L0C→UB、`T.clear` L0C）必须有替代方案 + 机器证据，而不是"借"过来。
- **Stage 3 的已知容差边界要随复用关系搬运**：同一数学的两个变体会撞同一堵墙（bf16 `dS` 抵消放大）；检视方有责任在 §8.2 里发现缺失并升级为建议。

## Stage 3 (develop) Retrospective — first_impl attempt 1 — 2026-11-23

结论：`[PRECISION_PASS]`（L0 20/20、L1 6/6、L2 4/4、Boundary 12/12；7 个运行期前置探针 R1–R7
全 PASS；`--level all` 423 s，含全部 16 个几何的编译）。设计的三处 packed 差异全部在真机上成立，
**§9.1 风险 1（动态 grid + 混合 kernel 的运行期行为）一次通过，无需回退到 P5 静态 grid**。

### 1. 设计文档里被真机确认的部分（不需要改，但值得记账）

| 设计断言 | 真机证据 |
|---|---|
| 动态 grid `(total_kv//64 + B) * H` + `live` 早退 | R1/R2：同一份编译产物在两个不同 `T_kv`（512 → grid 9、96 → grid 2）上都正确；L0-17/19 的 grid 实测 16640 / 8160 块全部正确、OOB 哨兵 0 次触碰 |
| 融合式 running-prefix decode（`T.serial(batch)` + `alloc_local`） | R1（B=2 不同长度、非整除）+ L0-6（`[65,1,200]` 三请求混合）+ L0-13（`[0,64]`）；`live==0` 块不发起任何写（哨兵 0 触碰） |
| `lse/delta` 保持 head-major `[H,T_q]`，不转置 | R5：正确的自然对数 head-major → dq 4.2e-2 通过；把它换成 log2 域 / token-major 重解释 / 错位 delta → 分别炸到 1.8e1 / 7.8e1 / 5.3e1，说明布局与对数域确实被消费而非静默忽略 |
| `dq` 整块 store-atomic + A3「溢出零行」 | R4：`[96,96]` 下 request 0 的尾块向 request 1 溢写 32 行，request 1 的 `dq_acc` 与「只跑 request 1」的结果**逐位相同（0.00e+00）**；L0-11 的「预填 2.0」两跑均与 golden 一致 ⇒ 是加不是覆盖 |
| `dk/dv` 双分支（内部整块 / 尾块逐 lane） | R3（`[1]/[1]`、`[300]/[700]`）+ L0-9（`[300,512]`，request 0 的 44 行尾块紧邻 request 1）+ B-11（`len_q=0` 夹在中间）逐请求干净 |
| 半区 `dual_copy` 2:1 + `asc_get_sub_block_id`（复用 dense） | R6：单块 `[64]/[64]` 的 rows[0,32) 与 rows[32,64) 各自与 golden 一致 ⇒ 两个 AIV 的半区没有错位 |
| `loop_ed = max(ceil(len_q/64), loop_st+1)` 覆盖 `len_q=0` | R7：`[0,64]/[64,64]` 不崩、request 0 的 `dk` 严格为 0、request 1 正确 |

### 2. 本轮唯一的实质发现：16-bit GEMM 操作数舍入的容差边界（D 类，已转正 repro）

`P`/`dS` 必须先 cast 到输入 dtype 才能进 16-bit L1 的 GEMM（源 GPU kernel 同样如此）。对一个真值
是小量抵消的 cell，该舍入被 `sum|terms|/|value|` 放大后**可以超出 atol 而 kernel 完全正确**：

- 实测：bf16 `[0,64]/[64,64]` 的 `dv[0,1,77]`：真值 1.21e-4、`sum|terms| = 70.1`（**5.8e5 倍抵消**），
  kernel 给 -5.920e-3，fp64 ideal 给 +1.21e-4 → 差 6.04e-3 > atol 5e-3；而把同一份 `P` cast 到 bf16
  再算的"源语义参照"给 -5.929e-3，**与 kernel 只差 7e-7**。
- 规模：L0-17/18/19（3.7e8…7.4e8 个比对 cell）分别只有 14 / 6 / 3 个 cell 越过 ideal 的 atol，
  且 100% 落在源语义参照的容差内；fp16（11 位尾数）在同几何上全部干净。
- 判定手法（可复用）：不要放宽 atol，而是加**第二个参照**（重复同样的 16-bit 舍入），
  `kernel == truth` 且 `kernel` 只在个位数 cell 上 != ideal ⇒ 固有量；若 `kernel != truth` 则是实现缺陷。
- 该现象 dense Stage 3 已实测（L2-3/B-4 WARN，7.8e-3），但诊断脚本只留在 `/tmp`，且
  Stage 2 检视只把它列为建议 ⇒ 本轮在 **L0 门槛上真实复现**（manifest smoke bf16 1/655360 cell），
  说明"建议级"的措辞低估了它的必然性。已转正为
  `repro/CONST-950-16bit-operand-cancellation.py`（自包含 + 4 组断言，实跑 exit 0）。

### 3. Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Develop-flow | 设计 §8.2 的精度门槛（fp64 ideal + `atol/rtol`）对 bf16 **不可达**：源语义强制把 `P/dS` 舍入到 8 位尾数，强抵消 cell 必然越界。开发 skill 的 `run_case` 模板只给了"单一参照 + assert_close"，没有"第二参照分类"这一档，本轮先撞 L0 FAIL 才建立判据 | `_bwd_kernel.py` 的 `run_case` 第二参照分支；repro/CONST-950-16bit-operand-cancellation.py；L0-1 首次 FAIL（1/655360 cell） | `tilelang-op-develop` 的精度模板：输入 dtype 的尾数 ≤ 11 位时，`{op}.py` 必须同时提供 `golden(quantize=True)` 的源语义参照，并把"越界 cell 数 ≤ 5e-5 且第二参照全过"写成通过条件；`atol` 本身不得放宽 | R |
| Develop-flow | 设计的 L0 计划按 manifest 原样（T=32256、H=128）要求 CPU golden，但本机 cgroup 上限 32 GiB：首版 golden 用全尺寸 fp64 累加器 + 全局 fp32 输入副本，**进程被 OOM kill**（组合脚本在 L0-2 后静默消失，`sys` 时间 4 min 是征兆） | 首跑日志（L0-1/L0-2 后无输出）；cgroup `memory.max = 34359738368`；改后峰值 RSS 13.7 GB | `tilelang-op-develop` 的 golden 模板加三条硬约束：① 输入按 request 切片后再 cast（不要全局 float()/double()）；② 累加器按 (request, head-slab) 粒度写进输出、不留全尺寸 fp64 累加器；③ score 平面原地复用（`s → p → ds`），长请求（>8192）用 fp32 累加；并给出"manifest 级用例可缩减 H"的显式许可 | R |
| Develop-flow | 平台 `execution.working_directory` 不是可选项：在仓库根 import tilelang 会命中仓库内 dev root 的 `tilelang/`，报 `TileLang's private TVM Python package is missing`——与 kernel 无关却像"环境坏了" | 在仓库根执行 `python -c "import tilelang"` 的 ImportError；在任务目录下正常（`tilelang 0.1.15 /workspace/venv-ascend950/...`） | `tilelang-ascend` 的 Stage 3 段：把"必须 cd 到任务目录再跑"从 ruleset 的 `working_directory` 抬成**首跑检查项**，并在报错分类表里加一行"repo-root ImportError = 工作目录错，不是环境缺依赖" | R |
| Review-flow | Stage 2 把 dense 的已知容差边界列为"建议 ②（非阻塞）"，但它在 packed 的 L0 门槛上必然复发（1/6.5e5）。建议级的判据应是"该现象在 Stage 3 的**门槛层**是否必然出现"，而不是"是否影响正确性主线" | REVIEW.md 建议 ②；L0-1 首跑 FAIL | `tilelang-design-review` 维度 6：兄弟实现的 Stage 3 已实测的固有精度边界，若在**本设计的 L0 计划里存在同类用例**，升级为阻塞（须在 §8.2 写明第二参照口径），否则仍为建议 | P |

### 4. Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 16-bit GEMM 操作数的**抵消放大容差边界**与"第二参照"判定法：`kernel == 16-bit-cast 参照` 且只有个位数 cell 越 ideal atol ⇒ 固有量；`kernel != 参照` ⇒ 实现缺陷。bf16 实测 5.8e5 倍抵消、6.04e-3 越界、与参照差 7e-7 | D | 本轮 L0-1（1/655360）、L0-17/18/19（14/6/3 个 cell 于 3.7e8–7.4e8 中）、L2-2；dense L2-3/B-4 的 7.8e-3 | `repro/CONST-950-16bit-operand-cancellation.py`（exit 0，4 组断言） | tilelang 0.1.15 / CANN 9.2.1 / ascend-950pr-9579 | 平台 knowledge（numerical-boundaries 条目 + `PL-950-softmax-boundaries` 交叉引用）+ develop skill 的精度模板 |
| **动态 grid 在 `__mix__(1,2)` 混合 kernel 上运行期成立**：`T.Kernel((total_kv//64 + B)*H)` 随运行期 `T_kv` 变化（9 / 2 / 8160 / 16640 块），同一份编译产物跨不同 `T_kv` 复用且结果正确；设计期把它列为"最大未核验项"，本轮 1 次通过 | D | 探针 R1/R2；L0-17（grid 16640）、L0-19（grid 8160）的 OOB 哨兵 0 触碰 | `history_version/stage3_probe.py`（R1/R2，assert 型） | 同上 | 平台 knowledge（execution-patterns / paged_cache_write 条目补"混合 kernel + 动态 grid 也成立"，此前先例只有纯 Vector） |
| packed 语义的三件套在真机上彼此独立成立：融合式 O(B) 前缀解码（无 `sync_grid`/无跨核 flag）、head-major `[H,T_q]` 统计量（256 B 连续段）、`dk/dv` 双分支 + `dq` 整块原子（溢写行为**逐位** 0） | D | R4（request1 vs solo = 0.00e+00）、R5（错布局炸到 1e1–1e2）、R3/L0-9/B-11、L0-6 | `history_version/stage3_probe.py` | 同上 | 平台 knowledge（attention 族的 packed 变体条目：clamp 只认张量 extent，request 边界必须自己收口） |
| 长 packed 请求的 **CPU 参照内存预算**：按 request 切片后 cast + score 平面原地复用（`s→p→ds`）+ 按 (request, head-slab) 直写输出，峰值 13.7 GB（改前 OOM @32 GiB）。H 缩减是可接受的旋钮（每 block 工作量与 H 无关） | P | 首跑 OOM；改后 `resource.ru_maxrss = 13.7 GB`（16384 token、H=16 的 mixed workload） | `_bwd_kernel.py --level L0`（L0-19 单例即可复现内存形态） | 同上 | develop skill 的 golden 模板 + `tilelang-op-design` 的 L0 计划（要求写"参照实现的内存预算"） |

### 5. Transferable Lessons

- 本迁移任务的后续函数（`_delta_kernel`、`_dq_land_kernel`）若也走 bf16/fp16，**精度门槛必须配"第二参照"**：
  先写 `golden(quantize=True)`（重复同样的输入 dtype 舍入），越界 cell 用它分类；直接放宽 atol 会把真 bug 一起放过去。
- packed 的 CPU 参照**不要**用"全尺寸 fp64 累加器 + 全局输入 float()"：32 GiB cgroup 下 4096×8 请求就会 OOM。
  按 request 切片 cast、score 平面原地复用、按 (request, head-slab) 直写输出是标准形态。
- 动态 grid 与 `__mix__(1,2)` 混合 kernel 在 950PR 上**运行期可用**（本轮一次性通过），所以 packed 类算子不必
  预先退化到"静态上界 + live"；但 `live` 必须包住**整个** body（含全部 DMA），否则非 live 块会读越界
  `cu_seqlens[req]` 并写错地址。
- 请求边界是 packed 唯一不能交给平台的语义：平台 OOB clamp 只认张量 extent（`min(64, max((total_kv-kv_start)-n0, 0))`），
  因此"赋值型"输出（`dk/dv`）必须双分支收口，而"累加型"输出（`dq`）可以靠"溢写恰为精确 0"整块原子——
  后者是可验证的不变式（本轮实测 request1-vs-solo 逐位相同），但前提是掩码保持 `if_then_else` 的选择语义。
- 同一份 kernel 在不同 seed 上会给出"这轮 FAIL、下轮 PASS"的表象：先查是否是高抵消 cell（比值 > 100 倍），
  再决定是不是实现问题；判据是第二参照，不是重跑。


## Stage 2 (design review) Retrospective — 2026-11-24 — 结论 不通过（阻塞 1 / 建议 10）

> 本节复盘**检视流程自身**。设计内容的问题（§8.4 的 `dq` 阈值不可达等）已随 `REVIEW.md` 的建议逐条流转给 Stage 1，
> 不在此重复；以下只记「检视维度/工具自身」暴露的缺口与被证实的判据形态。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Review-flow | 维度 6 的清单只有「Golden / L0 计划 / 按 dtype 的 atol-rtol」，**没有『同一份验证方案内两个阈值是否互相矛盾』这一项**。v0 早已写「`dq_acc` 不做逐位比对、只做 dtype 容差（atol 6.9e-2）」，v1 新加的 §8.4 对同一个量写 `≤1e-5`；`design_calc_check` 与全部 D-1 机械位都不涉阈值，只有人工把 `量级 × fp32 ulp` 算出来才暴露 | `REVIEW.md#问题 1`；`DESIGN.md#§8.2`(1762,1765) 与 `#§8.4 G1`(1816)；`§8.2` 在 v0/v1 间**逐字节相同**（`history_version/design_v0_pre_gridfix.md` vs `DESIGN.md` 区段对比 identical=True），故冲突是 v1 新增判据未与既存判据对齐 | `tilelang-design-review` 维度 6 增子项：「方案内每个绝对阈值须与对应量实测幅值的 ulp 及 §8.2 的 dtype 容差做一次量级自洽核算（`abs_tol ≥ 1 ulp(max|量|)`；若比 §8.2 更紧，必须写成 `k·ulp` 并说明 k 的来历）」 | P |
| Review-flow | 检视**修订版**时 skill 没有「增量面 × 不变面」的双栏工作法。v1 只改任务映射，`design_calc_check` 三个 pass 项（`ub_budget`/`l0c_budget`/`r3_metrics`）数字与 v0 **一字不差**，机械位等于没检；真正需要复核的变成设计新增的纯 Python 算术模块（`launch_plan.py`）与「未改动项」清单是否真的没动 | `REVIEW.md#机械复核 M2/M3`；`DESIGN.md#§12.3` 未改动项清单；`launch_plan.py:176-187`（自检主程序）；v0/v1 的 §4.5、§4.6、§8.1、§8.2 四段区段对比均 identical=True | Phase 1 第 7 步增一条：「若设计引入新的纯计算模块（launch/形状推导），机械复核 = ①直接执行该模块的自检与断言用例 ②对文档表格逐格比对 ③按『未改动项』清单抽 ≥3 项与上一版对比」，不得只贴 `design_calc_check` 的 JSON | R |
| Review-flow | 迁移任务的 Read 范围只规定「源算子文件」，未覆盖源算子的**依赖工具模块**。本函数 packed 分支的三条关键主张（warp shuffle 前缀扫描、`serial_groups=8` 退化分支、二分 decode 语义）全在同仓 `grouped_tiling.py`；设计 §0.4#15 / §0.6 R6 的正确性只能靠读该文件判定 | `REVIEW.md#维度 0`（优化手段行、R6 复核）；`/workspace/TileOPs/src/tileops/_kernels/grouped_tiling.py:74-84,93-104,166-200` | 维度 0 的「算法解读完整性」补齐 Read 范围：「源算子 import 的同仓 tiling/scan/布局工具模块须一并 Read（用 `grep -n "from tileops._kernels" bwd.py` 定位）」，否则『算法步骤覆盖』无法判定 | R |
| Mechanical-check | 平台知识条目被本任务补记后 `toolchain_stamp` 未更新，`kb_stale_check` 仍报 stale：检视方若只信 `kb_stale_check` 会把**已补全**的条目继续降级为「线索」，只信正文又会把**陈旧**条目当依据。skill 无「条目正文 vs 戳记一致性」核对项 | `REVIEW.md#问题 10`；`.agents/platforms/toolchains/ascend/knowledge/traps-950pr-execution.md:39-81`（facet 存在、`toolchain_stamp` 仍 `a010a95…`、当前 `7a52f2f1…`） | 维度 8 的信息源核对补一句：「引用知识条目时分别记录『正文首验日期』与『toolchain_stamp』；两者不一致按正文证据采信，并把戳记滞后作为建议级提出」 | P |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **验证阈值的量级自洽判据**：`abs_tol` 必须 ≥ 1 ulp(该量实测幅值)，且不得比 §8.2 的 dtype 容差更紧；更紧时必须写成 `k·ulp` 并说明 k。反例：fp32 累加量幅值 1.05e3 → 1 ulp = 1.22e-4，而设计写 1e-5（0.08 ulp）⇒ 必然假失败 | P | `DESIGN.md#§8.4`(1e-5) vs `#§8.2`(6.9e-2)；`/tmp/opencode/MLA_VARLEN_BWD_8x4K_FINDINGS.md#§D`（`|dq|≈76`、bf16 上差 2.0）；`REVIEW.md#问题 1` | 手算无需设备：`76×13.856=1.05e3`；`2^-23×2^10=1.22e-4`；与 findings §D 的 0.9999 逐位相同比例自洽 | tilelang 0.1.15 / CANN 9.2.1 / ascend-950pr-9579 | `tilelang-design-review` 维度 6 |
| **修订版的「不变面」抽检法**：把设计的「未改动项」清单当可证伪声明，抽 ≥3 项与上一版逐字对比（本次抽 §4.5 / §4.6 / §8.2 三段，全部 identical），据此才允许承袭上一版结论；新增量面做满维度检视 | P | `DESIGN.md#§12.3`；v0 与 v1 的 §4.5/§4.6/§8.1/§8.2 四段区段对比 identical=True；`REVIEW.md#维度 0` 第 3 段 | `python3 - <<'EOF'` 内按 `re.search(r"### 4\.5 …(?=### 4\.7)")` 取两文件区段后比较；本次四段均 True | — | 本 skill（Phase 1 增「修订版」分支）+ `tilelang-op-design` 的 revision 模式 |
| **块索引类平台上限修复的两层证法**：① 任务映射双射（纯算术，可逐 bid 穷举）② 设备码形态（静态除数 / 运行期原点 / 守卫包住 body）。两者都不覆盖运行期写集，故必须再配一条「逐 owner 拆分对照 + 逐 owner 非零能量检」的运行期用例（本次 G1 ④） | P | `verify_equiv.py#check_a4`（568,904 任务、`over_cap=0`）、`design_probe_grid_slab.py`（P6 设备码 `h_off[0] + (block_idx & 1)`）、`DESIGN.md#§8.4 G1` | `cd examples/mla_varlen_bwd/_bwd_kernel && TMPDIR=/workspace ASCEND_RT_VISIBLE_DEVICES="" /workspace/venv-ascend950/bin/python history_version/design_probe_grid_slab.py`（本轮 COMPILE_OK，28,381 B，13 marks） | tilelang 0.1.15 / CANN 9.2.1 / ascend-950pr-9579 | 本 skill 维度 4/6 + `tilelang-op-design` 的验证方案段 |

### Transferable Lessons

- 修订版**新增的判据必须与既有判据对齐**：同一份设计里同一个量出现两个阈值（容差 6.9e-2 与 1e-5）时，先算「1 ulp 是多少」再判自洽；原子累加类输出只能做 dtype 容差，逐位比对只适用于单 owner 的赋值类输出（如 packed 的 `dk/dv`）。
- 「不读任何 GM / 不做任何 DMA」这类零成本声明要按设备码逐字核：运行期 launch 标量（如 head 原点 `h_off`）本身是一次 4 B GM 读，成本可忽略但声明不能写成「零」。
- 平台上限类修复的正确性证法分两层（映射双射 = 纯算术可穷举；设备码形态 = 编译级），两层都不覆盖运行期写集，必须再加一条「逐 owner 拆分对照 + 逐 owner 非零能量检」的真机用例，且参照必须是「同一内核、远离上界」的几何。
- 复用上一版结论前先验「未改动项」：预算表 / 容差 / 操作数方向抽 3 项逐字节对比，一致才允许承袭；当机械位数字一字不差时，它等于没检。
- 引用知识条目时同时记录正文首验日期与 `toolchain_stamp`：戳记滞后只说明元数据旧，不代表正文不可信；正文与戳记冲突按正文采信，并把戳记滞后作为建议级提出。

## 2026-11-24 — 设计修订 v2（验收判据修订：`dq ≤1e-5` 不可达）

> 触发：Stage 2 检视（`REVIEW.md`）对 v1 判「不通过」，**唯一阻塞项**是 §8.4 G1/G2/G3 与 §8.3 L0-12/L0-21 的
> `dq` 绝对阈值 `≤1e-5`。grid 修复本体（slab 方案、`h_off`、尾 slab 守卫、启动前断言、写集不交、A4、P6）经
> 检视独立复核**成立**，v2 **未改动**。本节只记「哪类设计判断被推翻、正确依据是什么」。

### 1. 被推翻的判断与正确依据

| 被推翻的判断（v1） | 正确依据 | 证据定位 |
|---|---|---|
| 「`dq` 的跨 launch 一致性可以用一个绝对阈值 `1e-5` 卡住」（想比 §8.2 更严） | 绝对阈值的下限是**该量的 1 个 ulp**：`\|dq_acc\| ≈ 1.05e3` ⇒ fp32 ulp `2^-23·2^10 = 1.221e-4`，`1e-5` 只有 **0.082 ulp** ⇒ 要求两次独立 store-atomic 累加逐位一致，不可达 | `REVIEW.md#问题 1`；`/tmp/opencode/MLA_VARLEN_BWD_8x4K_FINDINGS.md#§D`；`DESIGN.md#§8.2 ③` |
| 「§8.4 是 v1 新增的回归门，因此可以自带判据」 | **同一文档里同一个量只能有一个口径**：§8.2 在 v0/v1 间逐字节相同，早已规定 `dq_acc` 不做逐位比对、只做 dtype 容差（atol 6.9e-2 = 565 ulp）。新增判据必须**引用**它，不能重定义 | `DESIGN.md#§8.2`（v0/v1 identical）vs `#§8.4 G1`(v1)；`REVIEW.md#问题 1` |
| 「`dq` 的 bf16 输出差 `2.0` 说明 fp32 层也差 `2.0`」 | `dq = cast_bf16(dq_acc·scale)`：fp32 层 1 ulp 的差在 bf16 舍入边界上放大成 **1 个 bf16 ulp**（`ulp_bf16(76) = 0.5`、`ulp_bf16(256) = 2.0`）。`2.0` 是 `\|dq\| ∈ [256,512)` 上的 1 ulp，不是 4 ulp；反之若真是 4 ulp，`dq_acc` 层必须有 `≈27.7`（2.3e5 ulp）的差，与「0.9999 逐位相同 + 只差原子序」不相容 | `DESIGN.md#§8.2 ④`；findings §D |
| 「尾 slab 空块不读任何 GM」 | P6 设备码首行就是 `h = h_off[0] + (block_idx & 1)` ⇒ **1 次 4 B GM 标量读**；「零成本」声明必须按设备码逐字核 | `REVIEW.md#问题 5`；`history_version/design_probe_grid_slab.py` |
| 「计划与产物天然同源」（`block_n`） | `plan_launch` 的默认 `block_n=64` 与工厂 `cfg["block_n"]` 是**两个可独立变化的常量**；`block_n=32` + 131,072 token ⇒ 计划说 63,736、真实 grid `4,104×31 = 127,224` ⇒ 静默截断复发而断言仍通过 | `REVIEW.md#问题 2`；`DESIGN.md#§5.6 g`、`#§8.4 G5⑥` |
| 「`L2 = 128 MiB` 是硬件档案」 | 平台 `design_budget`/`knowledge` **无 L2 容量条目**；只有 UB/L1/L0C/L0A/L0B。引用不到条目就只能写成**未文档化假设**，且容量淘汰的结论应对数值不敏感 | `REVIEW.md#问题 6`；`DESIGN.md#§1.6.0 估算行` |
| 「弃选项 #5 的代价 `+5.7%`」 | 单迭代 MAC 合计 `3,407,872`；额外一次 `192×64×64 = 786,432 MAC` ⇒ **+23.1%**（`dK`/`dQ` 各一次则 +46.2%）。分母写出来才能核 | `REVIEW.md#问题 4`；`DESIGN.md#§1.6.1 弃选项` |

### 2. 处置（设计层，逐条落点）

- §8.2 新增**唯一口径**：① 判定形式 `atol + rtol·|ref|`；② **任何绝对阈值 ≥ 1 ulp 且不得比 §8.2 更紧**，
  要更紧只能写 `k·ulp(max|·|)`、`k ≥ 8`；③ 量级锚点；④ fp32→bf16 重量化关系；⑤ 诊断量 `k_ulp`（告警不判失败）。
- §8.4 目标改「四条」，G1/G2/G3 的 `dq` 判据全部指向 §8.2；`dk/dv` 保留逐位相同；G1 ④ 笔误 `dk/dk/dq` → `dk/dv/dq`；
  G5 增 ⑥ 同源后置断言（含反例算术）。
- §8.3 表补齐 L0-13…L0-19 与 L0-20…L0-23，新增 **L0-11b**（`H=64 / head_slab=32` ⇒ 2 次 launch，覆盖「slab 之间
  不得清零」），L0-1/L0-2 的几何统一到 manifest-smoke（`grid=96`）。
- §5.5 的 smoke 行改为 `tiles=12 / slab=8 / grid=96`；③ 改「96 ≈ 3.4 波」。
- §5.6 g / §4.8 增 `block_n` 同源 + 运行期后置断言；§9.1 增风险 19；§9.2 增两条反例（`block_n` 不同源、`dq` 固定阈值）。
- §1.6.0 的 C4/估算行撤掉 `128 MiB`，改未文档化假设 + 数值不敏感表述；风险 6 同步。
- 全文统一「尾 slab 空块 = 1 次 4 B `h_off` 读 + 2 个标量运算」；百分比复核：备选 L1 端口 `−5.8%`（两处 `−5.9%` 改正）、
  `dq` 原子流量 `−50%`（v1 误记 `−7.7%`）。

### 3. 机器验证（本次实跑）

- `verify_equiv.py`：`EQUIV_SUMMARY ALL_PASS`，5 行 `EQUIV_PASS`（A1 fp16/bf16、A2、A3、A4）；A4 的 smoke 行
  `slab=8 launches=1 max_grid=96`。
- 判据可达性：`atol 6.9e-2 = 565 ulp`、`rtol·1.053e3 = 1.053 = 8,627 ulp` ⇒ 总计 `1.122`（`dq` 单位 0.081）。
- `statectl gate --stage 1`：**0 fail / 0 warn**；`design_calc_check`：`ub_budget`/`l0c_budget`/`r3_metrics` pass。

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **「新增判据必须引用既有口径」自检项**：设计里出现第二个阈值时，先算幅度 ulp，再与既存 dtype 容差比较；紧于既有容差的绝对阈值一律改写成 `k·ulp`（k≥8）或直接引用既有容差 | R | `DESIGN.md#§8.2 ②`；`REVIEW.md#问题 1` | 手算：`76/0.07217=1.053e3`、`2^-23·2^10=1.22e-4`、`6.9e-2/1.22e-4=565` | tilelang 0.1.15 / CANN 9.2.1 / ascend-950pr-9579 | `tilelang-op-design` Phase 2/验证方案段 + `design_lint` 可加机械位（`abs_tol` 与 ulp 的比值扫描） |
| **高幅值累加量的「ulp 归一 + 重量化」解释模板**：`dq` 这类「fp32 累加 → cast 到低精度」的量，低精度侧的 1 ulp 差是正常现象；判定应落在 fp32 侧，低精度侧用 dtype 容差（其 rtol ≫ 2^-8，恒覆盖 1 ulp） | P | `DESIGN.md#§8.2 ④`；findings §D（`2.0` @ `mean\|ref\|≈76`） | `ulp_bf16 ∈ {0.5@76, 1@128, 2@256, 4@512}`；`5e-3+1.6e-2·128 = 2.05 > 2.0` | 同上 | 本 skill 的验证方案模板（bf16/fp16 输出 + fp32 中间量并存时） |
| **纯 Python 计划模块的「常量源」断言**：主机侧计划与 kernel 工厂共享 `block_n` 时，必须有运行期后置断言（`(total_kv//block_n+batch)*slab == max_grid`），否则断言对象可能与编译产物脱钩 | R | `DESIGN.md#§5.6 g`、`#§8.4 G5⑥`；`REVIEW.md#问题 2`；反例 `4,104×31 = 127,224 > 65,535` | 手算即可（`plan_launch(8, 131072, 128, block_n=32)` vs 默认 64） | 同上 | `tilelang-op-design` 的 Tiling/启动计划段 + develop 的 wrapper 模板 |

### Transferable Lessons

- **阈值是对物理量的声明，不是表达严格性的手段**：写绝对阈值前先算「1 ulp 是多少」，再算「该阈值是几个 ulp」；
  小于 8 ulp 的绝对阈值在跨核累加的场景里等价于要求逐位一致。
- **新增章节不得重定义既有口径**：先在全文档检索同一个量的所有阈值，逐一对齐；v0/v1 的 §8.2 逐字节相同，冲突
  只可能来自新章节。
- **低精度输出侧的「1 ulp 差」不需要慌张，但要用算式证明它是 1 ulp**：`ulp_bf16(|ref|)` 随幅值指数增长
  （76→0.5、256→2.0、512→4.0），结合「逐位相同比例」与「fp32 侧应有的差」可唯一判定读数。
- **主机侧计划与设备侧产物的常量必须同源，且要有运行期后置断言**：编译期常量一旦分叉，越界断言会「通过」而
  真实 grid 越界，且现象仍是「静默截断 + 变快」。
- **引用不到平台条目的常数一律写「未文档化假设」**，并让结论对该数值不敏感；不要把估算里的便利假设写成「硬件档案」。


## Stage 2 (design review) Retrospective #2 — 2026-11-24 — v2 复审 结论 通过（阻塞 0 / 建议 4）

> 本节复盘**检视流程自身**（第二轮：设计 v2 针对上一轮阻塞项 B1 的修订）。设计内容的问题已随 `REVIEW.md` 流转。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Review-flow | **上一轮的建议没有稳定编号**，导致「修订版是否逐条闭合」只能靠人工比对：我在 v1 的维度 3 里内嵌了两条「建议 2/3」（其中「auto-shrink 时 slab 取 heads 约数」从未进入问题列表编号），v2 的 §12.6 修订对照表只覆盖问题 2–10，**该条被静默丢弃**，本轮才发现 | `REVIEW.md#已闭合项` 末行（「v1 维度 3 内嵌」行）；`DESIGN.md#§12.6` 的「检视建议 9 条」表 | 本 skill Phase 4：REVIEW.md 的每条建议必须有**稳定 ID**（`S1..Sn`，含只写在维度内的），并在 Phase 5 之前输出一行 `SUGGESTION_IDS: S1,…`；revision 模式下第一步做 **ID 集合差**（缺哪条 = 未闭合） | R |
| Review-flow | **「未改动」声明不能只读 changelog**。v2 声称「grid 修复本体零改动」，§12.6 也列了清单；若照读，就无法排除「顺手改了一处 slab 公式」。本轮用 md5（`_bwd_kernel.py`/`launch_plan.py`/`verify_equiv.py`/P6 探针）＋ 四段逐字节区段比对（§3.5.4 P6 表、§5.6 d 写集论证、§5.6 f 备选裁决、§8.4 G4）才确认，并**发现** §1.6.1 A4 行有一处措辞差异（非结构差异） | `REVIEW.md#机械复核 M4` 末段；四段 `identical=True`、A4 段 1,192→1,214 B 仅末尾措辞 | Phase 1 增「修订版」分支：**凡「未改动项」声明，一律用 ① 产物哈希 ② 设计文档区段逐字节比对 双证**，并把「比对后发现的微小差异」如实列出（区分结构差异与措辞差异） | P |
| Review-flow | 第二轮出现了「**判据违反同一文档新立的元规则**」这一形态：v2 在 §8.2② 立了「任何绝对阈值 ≥ 1 ulp、不得比 dtype 容差更紧」，却把 L0-11/L0-11b 的判定留成「差值恒为 2.0 / 预填版 == 清零版 + 2.0（逐元素）」= **0 容差**，与 §8.2①「dq_acc 不做逐位比对」也冲突。skill 没有「新元规则是否回扫全文既有判据」的检查项，也没有「违反自家元规则时是否升为阻塞」的判据 | `REVIEW.md#维度 6 建议 1`；`DESIGN.md` 第 1861–1862 行（`== 2.0`）vs §8.2 ①②；对照 `_bwd_kernel.py:664-672` 已有的稳健实现 | 维度 6 增子项：「设计若新立数值判据元规则，须对全文既有判据做一次回扫（grep 固定阈值/相等断言），遗留违例按**该规则本身的严重度**处理」；并明确「有可用实现」不能作为降级理由（Stage 3 实现的是**文本**，不是上一轮的代码） | P |
| Mechanical-check | `statectl gate 1` 是本轮唯一可靠的 design_lint 入口（`.agents/tools/gate_lint.py` 无独立 CLI），但**检视 skill 的机械复核段只提名 `design_calc_check.py` 与 `verify_equiv.py`**，没提 `gate 1/2`。本轮两条都是我自己摸出来的：`gate 1` 验 DESIGN（0 fail/0 warn）、`gate 2` 验我自己的 REVIEW（同样 0 fail），后者对「结论字面量 + 维度章节」有机器兜底，价值很高 | `REVIEW.md#机械复核 M2`（`gate 1` 输出）；本轮额外跑 `gate 2` → `{"pass": true, "failures": [], "warnings": []}` | 本 skill Phase 1 第 7 步：把「`statectl gate <stage> --task-dir … --dir …`」写进机械复核清单（Stage 1 验待审 DESIGN、Stage 2 验自产 REVIEW），并规定输出 JSON 摘录格式 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| **判据可达性的 ulp 锚定法**：给任何数值判据定阈值时，先用「同一 workload 的实测幅值」算 1 ulp，再检查「阈值 ≥ 1 ulp」；「变紧」必须写成 `k·ulp`（k≥8）。反例链：|dq_acc|≈1.05e3 → 1 ulp 1.22e-4；v1 的 1e-5 = 0.08 ulp（不可达）；v2 的 6.9e-2 = 565 ulp（可达） | P | `DESIGN.md#§8.2`③（锚点）、`#§8.4` G1（判据）；`/tmp/opencode/MLA_VARLEN_BWD_8x4K_FINDINGS.md#§D`（|dq|≈76） | 手算无需设备：`76/0.07217=1.053e3`；`2^-23·2^10=1.221e-4`；`0.069/1.221e-4=565` | tilelang 0.1.15 / CANN 9.2.1 / ascend-950pr-9579 | 本 skill 维度 6 + `tilelang-op-design` 的验证方案段 |
| **「放宽判据」时同步证明判别力**：阈值变松必须配「目标 bug 的偏差量级 ≫ 容差」的定量对照（本次：截断 → |Δ|≈76 vs 容差 1.12–1.22 = 60 倍余量），再加独立闸门（单 owner 输出逐位相同、per-owner 非零能量、host 断言、吞吐 tripwire）。只写「容差放宽到 dtype 标准」不足以给 pass | P | `REVIEW.md#维度 6` 第 6 条；`DESIGN.md#§8.4` G1 判定列 + G3「127 个 head 全部非零」+ G4 | 推演（无需设备）：截断使 request 1..7 输出恒 0，参照均量级 76 | — | 本 skill 维度 6 |
| **修订模式的双证法**：产物哈希（实现/脚本）+ 设计区段逐字节比对（声称未改的章节），并列出比对中发现的措辞级差异；配合「建议 ID 集合差」判定闭合率 | P | `REVIEW.md#机械复核 M4`（md5 + 四段 identical + A4 段 1,192→1,214 B）；`DESIGN.md#§12.6` 的对照表 | `md5sum _bwd_kernel.py launch_plan.py verify_equiv.py`；两版同名区段 `re.search` 截取后 `==` 比较 | — | 本 skill Phase 1「修订版」分支 |

### Transferable Lessons

- 统计量类判据（fp32 原子累加 / 归约输出）**只能做 dtype 容差或 `k·ulp` 归一界**；写固定绝对值前先算该量的 1 ulp —— 一次 `|值|≈1e3` 的量上 1e-5 只有 0.08 ulp，必然假失败。
- 「判据放宽」不是低风险动作：必须同时给出「目标 bug 的偏差 / 新容差」的倍数（本次 60×）与独立闸门清单，否则放宽会掩盖真 bug。
- 修订版声称「某部分未改动」时，先用哈希与区段逐字节比对证明，再读 changelog；措辞级差异也要列出（本次 §1.6.1 A4 一行有 22 B 差异，是措辞不是结构）。
- 设计新立的数值元规则（如「任何阈值 ≥ 1 ulp」）要立刻回扫全文既有判据：本次遗留的「== 2.0」严格相等判据就是元规则漏扫的产物，它属于「照字面实现会假失败」的隐患。
- 检视建议必须带稳定 ID（S1..Sn），且**只写在维度小节里的建议也要编号**；否则修订版只闭合「问题列表」里那几条，内嵌条目会静默丢失（本次 auto-shrink 的 slab 约数建议即如此）。

## Stage 3 (develop) Retrospective — first_impl, attempt 1 — 2026-11-24

> 本节复盘 **grid 越界修复的实现与验证**（DESIGN v2 §5.6 / §8.2 / §8.3 / §8.4 / §12.6 H1）。
> 设计判断本身未变更、`DESIGN.md` 未改；所有结论均有本机实测支撑。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Develop-flow | develop skill 的 L0 门槛模板按「代表 shape + 边界 + 多 dtype」组织，**不含「grid 含运行期量 → 越过 65,535」这一检查项**。本案（以及 `chain_speculative_sampling`）的 L0 把 H 缩到 ≤32 以适配 CPU 参考，于是 `grid ≤ 32×tiles` 物理上碰不到红线，bug 只能靠事后复核 manifest 大 shape 才发现。建议把「kernel 的 grid 是否含运行期量」列为 L0 模板的强制自检项 | `_bwd_kernel.py` 的 `run_L0()`（L0-17/18/19 的 H=32/16/16）；DESIGN §8.3 的 L0 表；`TRAP-950-grid-over-65535` 的两次独立命中 | `tilelang-op-develop/SKILL.md` Phase 4 增一条：「grid 依赖运行期 shape 时，L0 必须含一条跨 65,535 的 host 计划用例（或同族 split 对照）」 | R |
| Develop-flow | 平台的 `execution.precision_command` 指向 `/usr/local/python3.12.16/bin/python3`，本机该解释器**没有 torch/tilelang**（`import torch_npu` 直接 ERR99999）。Stage 3 全程改用 `/workspace/venv-ascend950/bin/python`（设计阶段已记录该偏离） | `statectl platform --task-dir …` 的 `execution.precision_command`；`env-950pr.sh` 的 `TILEOPS_950PR_VENV` | `workflow.md`/站点 profile 把 `execution` 的解释器指向 `env-950pr.sh` 解析出的 venv（或在 platform JSON 里给出 `python_fallback`） | R |
| Info-source | `T.Tensor((1,), "int32")` 的运行期实参必须是 **1 维**；把 `(n,1)` 张量的行切片 `h_off[i:i+1]` 传进去时，报错文本是 `kernel _bwd_main input h_off ndim mismatch, expected 1; expected: 2, got: 1`（自相矛盾，`expected` 出现两次且指向不同值），无法从文本判断到底是哪一维错 | 首跑 `/tmp/opencode/s3probe/p5.py`（全部 3 个 case 同错），改 `h_off` 为 1 维后一转通过 | `references/programming.md` 增「标量参数张量的 ndim 必须与声明一致；报错文本的 `expected … expected:` 双写是编译器已知缺陷」 | C |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| `T.Kernel(grid)` 的 grid 余数非 0 时**静默只执行 `grid % 65536` 块**（66,560 → 1,024），不是 abort；head-slab 分解（`grid=tiles*slab` + 运行期 `h_off` + `if h<head`）可把这 66,560 个任务分 4 次发射完整覆盖 | D | `_bwd_kernel.py --level G` 的 G1 行（4×16,640，8 个 request 的 dk/dv 逐位、dq_acc k_ulp 0.25–0.38）；`/tmp/opencode/MLA_VARLEN_BWD_8x4K_FINDINGS.md` §A/截断映射 | `repro/TRAP-950-grid-over-65535-silent-truncation.py`（设备实测：66,560 → 1,024 块；4×16,640 → 66,560 任务，每任务恰 1 次） | TileLang ascend 0.1.15 + CANN 9.2.1, Ascend950PR_9579 (dav-3510), 2026-11-24 | `.agents/platforms/toolchains/ascend/knowledge/traps-950pr-execution.md`（该 entry 的 `repro:` 字段补此文件） |
| 大几何（H=128）的 grid 上界回归用「**同一 kernel、逐 request B=1**（grid 2,080 ≪ 65,535）」作参照：`dk/dv` 逐位、`dq_acc` 0.25–0.5 ulp 收敛，且**逐 head 能量检**是截断的独立判据（截断 ⇒ 整 request/整 head 恒 0） | P | `_bwd_kernel.py` 的 `_split_reference`/`_compare_split`；G1/G2/G3a/G3b 的 per-request 表（dq_acc ≤ 3.8e-6，dq ≤ 2.0e-3，128/126/127 heads 全非零） | `_bwd_kernel.py --level G` | 同上 | `pattern-library`（大几何正确性参照形态）／`bottlenecks` |
| 精度/吞吐 tripwire 必须用**热**测量：新编译程序的首个 launch 在本栈上带 ~145 ms 一次性上传，冷跑 631.8 ms vs 热跑 483.3 ms（后者与 findings 的 488 ms 一致）；用冷跑会把 400–600 ms 的判据判成超限 | D | `_case_grid_cap_split` 的 G1 冷/热两行；G4 `latency 483.30 ms, 29.7 TFLOPS` | `_bwd_kernel.py --level G`（G1+G4） | 同上 | `bottlenecks` / optimize skill 的测量口径 |
| `dq_acc` 的 `k_ulp` 诊断必须用 **`max|Δ| / ulp(max|ref|)`**（DESIGN §8.2⑤ 的原式）；逐元素归一 `max_i |Δ_i|/ulp(|ref_i|)` 会被「理想值≈0」的单元支配（实测报 256，而 top-10 差值全为 0.0），失去诊断价值 | D | L0-12 首跑 `[WARN] k_ulp=256.00 top gaps [0.0 …]`；改为全局式后 `k_ulp max 0.50` | `_bwd_kernel.py --level L0`（L0-12） | 同上 | DESIGN/REVIEW 口径（§8.2⑤）→ 已由实现固化，无需改文档 |
| 「同一常量源 + 发射前事后断言」的落地形态：`factory_config()["block_n"]` 同时喂 `plan_launch` 与工厂，工厂把 `(head_slab, block_n)` **挂在编译产物上**，wrapper 每次发射前用产物自身的值重算真实 grid 与计划比对；两个故意不一致（plan slab 16 vs 产物 32；plan block_n 64 vs 产物 32 → 127,224 块）都抛 `ValueError` | P | `_bwd_kernel.py` 的 `factory_config`/`assert_launch_consistency`/`build_kernel`；`_case_launch_plan`（L0-20 ⑥）通过 | `_bwd_kernel.py --level L0`（L0-20） | 同上 | `pattern-library`（动态 grid 的 host 契约） |

### Transferable Lessons（可迁移教训）

- kernel 的 grid 若含运行期量（token 数、batch），**必须**在 host 侧发射前算并断言 `grid <= 65535`：余数为 0 的越界会 abort，余数非 0 的越界**不报错**、只跑 `grid % 65536` 块并把其余输出留成初值（表现为「更快 + 部分为 0」）。
- 把某一维切成**编译期 slab** 分多次 launch 时，逐 launch 的原点要用一元素 int32 GM 张量传入（`h = h_off[0] + bid % slab`），这样 `bid // slab` 仍是静态除数；尾 slab 用 `if h < heads` 守卫，空块只做一次 4 B 读即返回。
- 切分轴与原子累加器的**归约轴正交**时（head slab 之于 `dq` 的 query/key 归约），累加器只在第一次 launch 前清零一次；每次 launch 都清会丢掉前面 slab 的贡献。
- 大几何的边界回归**不能**用「同一个 grid 量级」的参照（会一起错）：用 B=1 逐 request 单跑（grid 低三个数量级）作参照，再补一条「逐 head 能量非零」的结构判据。
- 编译产物的元数据（本次 `head_slab` / `block_n`）挂在返回的 JIT kernel 对象上是可行的（`JITKernel` 允许 `setattr`），这比在 host 侧另存一份「我记得我传了什么」更能证明「计划 == 产物」。
- 端到端正确性的最后一个闸门是「wrapper 路径 == 直接 launch 路径逐位相同」：本次 8x4k 上 dq/dk/dv 三者 `bitwise=True`，说明多 launch 循环、`h_off` 传递与 `land` 的次序都没有引入偏差。

## Stage 2 (design review) Retrospective — 2026-11-25 — v4（Cube 化）复审 结论 通过（阻塞 0 / 建议 13）

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Review-flow | 「文档中由脚本产出的表格」与「检视期重跑输出」之间**没有逐格比对步骤**：维度 7 只要求"内部一致"，于是 v4 里 smoke 的 `tiles=12/grid=96`（应为 8/64）、auto-shrink 的 `tiles=2,100`（应为 2,108）、V12 的 "560 tokens"（应为 640）三处旧版残留都"内部自洽"地留了下来，只有把它们与 A4 的实跑明细逐格 diff 才会暴露 | `REVIEW.md` M1 的 A4 明细 `smoke 8/1/64/64/0`、`auto-shrink 31/5/65348/326740` vs `DESIGN.md` §5.5 表 / §5.6 c 表 / §8.3 V12 | 维度 7 增一条检查项："凡文档声明由脚本产出（A4 明细、`plan_launch` 输出、单价表、案例表）的数字，必须与本轮重跑输出**逐格 diff**，不得只做互相一致抽查" | R |
| Review-flow | 维度 8 的「备选方案结构完整（UB 预算按新任务粒度重算）」**无机械辅助**：`design_calc_check.py` 只解析主选的内存章节，备选行写错时无信号。本轮备选（`128/32`）的 `st_u/dp_u 2×16384 + p16_u/dsT_u 2×8192` 实际是按 `Mq=64` 算的（`Mq=32` 应为 `2×8192+2×4096`，UB 147,968 B 而非 172,544 B），只能手算发现 | `REVIEW.md` 维度 8 建议 8-1；`DESIGN.md` §1.6.3「备选（唯一一个）」 | `design_calc_check.py` 增 `budget_variants`：解析 §1.6.3 的主选/备选行，按其**声明的** `(block_n, block_m, _HN)` 重推每个 buffer 字节数并与表内数字比对 | R |
| Review-flow | 维度 1「API 存在性」以"该 API 存在 / 支持四类 trans"为通过标准，但设计真正吃风险的是**具体参数组合**。本设计只用 `(F,T)×4 + (T,T)×1`，`(T,F)/(F,F)` 从未被本设计验证；文档写"四类齐全"会让人误以为四类都被本设计验过 | `DESIGN.md` §3.5.2/§3.4 vs §0.6 R3 的 5 乘积表；`REVIEW.md` 维度 1 | 维度 1 增："API 的**参数组合**（trans 组合/切片形态/dtype）须以本设计**实际使用**的组合为清单核对，存在性证据不能替代组合证据" | R |
| Review-flow | 「68k 块上限 + 静默截断」这类平台 trap 在 Stage 2 **无法独立复验**（要构造 65,536 块几何并观察输出恒 0），只能继承 v3 的设备实测 + 本轮 A4 的结构性验证；而对应知识条目 `TRAP-950-grid-over-65535` 已是 stale，`kb_stale_check` 也会警告新读者 | `REVIEW.md` M5（stale 三条）；`DESIGN.md` §5.6 a | 平台 knowledge 的 trap 条目补一个**最小可复跑脚本**字段（本仓 `examples/mla_varlen_bwd/_bwd_kernel/repro/TRAP-950-grid-over-65535-silent-truncation.py` 已存在），使检视期能 1 条命令复验而不用引用历史结论 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 同一 kernel、同一 tile 的每迭代单价随 **grid 波数**变化：1,376 块（49 波）3.196 µs/迭代 vs 784 块（28 波）4.255 µs/迭代（+33%），两者都远超 1 波 ⇒ 小 shape 外推的偏差来自波数与 L2 冷启动，**不是**"短 request 块摊薄不足" | D | `REVIEW.md` M3 的 `--time` 两行 + M4 的 blocks/iters 复算；`DESIGN.md` §1.6.4 C 的因果表述（本轮建议改口径） | `TMPDIR=/workspace /workspace/venv-ascend950/bin/python history_version/design_probe_packed_cube_num.py --time` | TileLang ascend 0.1.15 + CANN 9.2.1, Ascend950PR_9579 (dav-3510), 2026-11-25 | 平台 `bottlenecks`（单价外推口径：先算块数/28 得波数，≥10 波才可用于生产外推）/ optimize skill 计时模板 |
| 「备选方案的预算表按**另一个 tile** 的尺寸填写」是一种隐蔽且机械可检的错误形态：`128/32` 的 UB 行用了 `Mq=64` 的 `st_u/dp_u/p16_u/dsT_u`（49,664 B vs 25,088 B），并连带把 `36.5 FLOP/字节` 也写成了 `Mq=64` 的值；由于偏差方向是"高估占用"，人读起来仍显得保守 | P | `REVIEW.md` 维度 8 建议 8-1；`DESIGN.md` §1.6.3 备选段 | 手算：`(64,32)` fp32 ×2 + `(64,32)` f16 ×2 + 512 = 25,088 B；`Mq=32` 的端口 ≈270,700 B ⇒ 25.2 FLOP/B | 同上 | design skill 的「实验裁决模式」清单（备选结构须按声明 tile 重算，逐项列出 buffer 形状×dtype 的字节推导）+ `design_calc_check.py` |

### Transferable Lessons（可迁移教训）

- 检视"某约束暂时不再触发"的修订时，**同时**核对三件事：守卫/断言是否仍在、是否仍有几何会触发它、以及"为何仍必要"的论证 —— 本例 `Nk` 96 让 manifest 全部远离 65,535，但 `auto-shrink` 几何仍取 `slab=31`、越界几何仍抛 `ValueError`，删掉断言就会在下一个更长 token 的调用上静默少算。
- 检视变长（packed）算子时，**读侧与写侧要分开收口**：读侧越界由平台 clamp 到**张量** extent + 掩码收口（读到邻居 request 的真实行无害，因为会被掩码置 0）；写侧必须按输出语义逐个核对 —— **赋值类**输出（`dk/dv`）写出 request 边界就是破坏（必须逐 lane 守卫），**累加类**输出（`dq_acc`）写出边界只有在"贡献恰为 0"时无害（需要"快路不溢出 + 掩码臂 `select` 先于乘法"两条一起成立）。
- 文档里的"预期区间"若贴着整数上沿（如 `153–165 ms` 的 165），验收时必须以**必达线**（`≥2.5×`）为准，否则落在上界会被读成"刚好达标、没有余量"。
- 让探针脚本自己产出"可供 diff 的表"（本例 A4 的逐几何明细、`--time` 的 blocks/iters），检视期就能用**一次重跑 + 逐格比对**替代"读文档+读脚本+手算"的三重劳动；本轮这三处旧版残留正是靠这种 diff 才被抓到。

---

## Stage 3 (develop) Retrospective — first_impl attempt 1 — 2026-11-25（v4 Cube 化：显式 L0 + `trans_A` 复用）

### 0. 本轮做了什么（实施清单，全部按冻结的 DESIGN.md v4）

| # | 设计要求 | 落点（`_bwd_kernel.py`） | 状态 |
|---|---|---|---|
| 1 | 5 个 GEMM 全显式 L0（`alloc_l0a/l0b/l0c` + 手工 L1→L0 `T.copy`） | `a_k (96,192)`、`a_t (96,64)`、`b_lo (64,192)`、`b_hi (192,96)`；10 条 L1→L0 copy + 5 个 `T.gemm` | ✅ 设备码 `asc_mmad×5`、`l12l0a×5`、`l12l0b×5` |
| 2 | `dS` 单份，`dQ` 用 `transpose_A=True` 复用 `dsT_s`；删向量侧跨步 scatter | 删除 `ds_s (L1)` / `ds_u (UB)` 与三处 `ds_u[j,i]` 写入；`T.copy(dsT_s, a_t)` → `T.gemm(a_t, b_hi, dq_c, True, True)` | ✅ `asc_copy_l12l0a_transpose×1`；AIV vec 1.314 µs/迭代（dense 1.344） |
| 3 | `block_n=96`（`_HN=48`）、2-tile 合并写、`num_stages=2` | `BLOCK_N=96`、`_STAGES=2`、`num_stages=_STAGES`；向量段只写 `p16_u`/`dsT_u` 两个半区 tile | ✅ `ascend_nd2nz_scatter×2` |
| 4 | 保留 head slab / `h_off` / 尾 slab 守卫 / grid 断言；`dk/dv` R7 双分支；`dq_acc` store-atomic + 溢出零行 | 未改动（v3 成果原样保留）；仅 `block_n` 由 64→96 | ✅ 逐 request/逐 head 切片 0 超差 |
| 5 | packed 的 per-request 守卫、`cu_seqlens` 解码、`live` | 未改动 | ✅ |
| 6 | 不改 `_delta_kernel` / `_dq_land_kernel`；wrapper 三 launch 契约不变 | factory 签名与 `build_kernel`/`plan_launch` 契约未变（`head_slab` 参数沿用） | ✅ 无需适配 |

同时按 REVIEW 建议 7-1…7-6 更新了被测期望值与常量：smoke `8/64`、auto-shrink `tiles=2,108 / slab=31 / grid=65,348 / 5 launches`、`launch_plan.BLOCK_N=96`（并新增「plan 的 `block_n` 必须等于模块常量」的 `ValueError`）、L0-20 的越界/上界几何改为 `6,290,688` / `6,290,592`。

### 1. 精度：分层全过（真实设备）

| 层 | 结果 | 关键覆盖 |
|---|---|---|
| L0 | **28/28** | 新增 L0-0a 精确 2 个 key 块（192）、L0-0b 单 key 块 fp16、L0-0c `q` 尾+`k` 尾+跨 request 溢出同现（100/205）、L0-0d 慢路 `len_kv∈{1,47,48,49,95}`；smoke 640 tokens 两种 dtype；A2 双臂（`dk/dv` 逐位、`dq_acc` 1e-3/6.9e-2 内、k_ulp 0.00）；store-atomic 单/多 launch；`len_q=0`；grid 越界 2/2 且上界 65,535 合法 |
| L1 | **8/8** | 两 dtype、非因果、多 request、单/多 launch（H=64：`tiles=22/slab=32/2 launches/704`） |
| L2 | **4/4** | 2048/1536/非因果 1024×2048 |
| boundary | **12/12** | 1/63/65/129/200、双向 shift、zero-dout、tiny-scores、`len_q=0` 居中 |
| G（H=128 大 shape） | **8/8** | G1 8x4k 逐 request（8/8 全非零、`dk/dv` 与逐 request 参考**逐位相同**、`dq_acc` ≤1.4e-6、k_ulp ≤0.38）+ 重复 2 次（`dk/dv` 逐位、`dq_acc` 逐位 0 漂移）+ **逐 head 128×8 全清**；G2 mixed 6 request（512…16384）+ 逐 head 128×6；G3a H=126（slab 32/63/126 三种划分数值一致，空块 698）；G3b H=127（空块 349）；G4 tripwire；L0-20 断言 |

`[G] 8/8`、`max_diff=0`（大 shape 的 G 级判据是「逐 request/逐 head 与同一 kernel 的单 request 参考一致」+ 原子契约，不用全量 golden）。

### 2. 性能：实测 vs 设计投影（同一台机、同一轮）

| geometry | 基线 | 实测 T_op（3 次取中位） | 加速 | 设计投影 | 必达线 |
|---|---|---|---|---|---|
| ds-v3-mixed (H=128) | 1728.7 ms | **681.5 ms** | **2.54×** | 400–430 ms（4.0–4.3×） | ≤500 ms（2.8×）**未达** |
| ds-v3-8x4k (H=128) | 490.6 ms | **201.6 ms**（min 196.2） | **2.43×** | 153–165 ms（3.0–3.2×） | ≤200 ms（2.5×）**边缘未达** |
| kimi-k2-4x8k (H=64) | 470.5 ms | **197.6 ms** | **2.38×** | 151–162 ms | — |

利用率（msprof，逐块均值，`evidence/msprof_pipe_*_s3.csv`）：`aic_cube_ratio = 20.8%`（基线 8.3%、目标 ≥22% **未达**）、`aiv_vec_ratio = 35.7%`（基线 73.5%、目标 ≤55% **达标**）。每迭代单价 **3.62 µs**（dense 生产 2.93 µs）。

### 3. 根因（本轮最有价值的两条，均为设计层、非实现层）

### 3.1 `T.dynamic` extent 把每个热 DMA 的长度变成运行期值（+0.55 µs/迭代）

用本 kernel 跑 **dense 的生产几何**（B=2、S=8192、H=128，与 `ds-v3-b2-8k` 逐字同 tile/同迭代数）：**196.9 ms vs dense 150.7 ms**，同一时刻重跑 dense 仍是 150.685 ms（2.971 µs/迭代）。逐管道对比（`evidence/` 两份 capture）：

| 管道 | dense | 本实现 | Δ |
|---|---|---|---|
| cube | 0.756 | 0.756 | 0 |
| mte1 (L1→L0) | 0.668 | 0.692 | +0.02 |
| **mte2 (AIC, GM→L1)** | **0.697**（118.5 GB/s） | **1.107**（80.5 GB/s） | **+0.41** |
| **mte2 (AIV, lse/delta)** | **0.112** | **0.348** | **+0.24** |
| fixpipe | 0.562 | 0.589 | +0.03 |

设备码给出了机制：dense 的 `q[b, m0:m0+64, h, :]`（静态 `S_q`）生成**常量长度** `..., 64, 192, 0, 0)`；本 kernel 的 `q[qoff:qoff+64, h, :]`（`qoff` 来自 `cu_seqlens`、行数是 `T.dynamic`）生成 `min(64, max((total_q - q_start) - kq*64, 0))`，且地址算术升到 `int64_t`。四个热拷贝（`q/do` 及两份转置）全部继承。可执行复现：`repro/TRAP-950-packed-dynamic-dma-length.py`（编译级、确定性，`TRAP_CONFIRMED`）。

**含义**：设计把 dense 的 2.97 µs/迭代当生产常数直接外推到 packed，忽略了 packed 的 `T.dynamic` 使每迭代 4 个热 DMA 失去常量长度。这是"复用 dense 常数"的口径缺陷，不是 kernel body 写错（cube/mte1/fixpipe 三项完全对齐即可证明）。

### 3.2 packed 负载异质让混合几何损失 30% 利用率（mixed 5.06 vs 3.62 µs/迭代）

同一 kernel、同一 H、同一 token 总数（32,256）、仅 request 长度分布不同：

| 几何 | 迭代数 | T_op | µs/迭代 | 由块时长反推的利用率 |
|---|---|---|---|---|
| 6×5376（等长） | 1,849,344 | 261.5 ms | 3.96 | 91% |
| mixed（512…16384） | 3,770,496 | 681.2 ms | **5.06** | **71%** |

msprof 给了同一结论的独立算法：单次 launch 的 `Σ(aic_time)/28 = 119 ms`，实际 Task Duration 170.4 ms ⇒ 30% 空闲；8x4k 的空闲只有 5%。**块内迭代数 1→256 的方差并没有被"动态派发"吸收**（与 DESIGN §5.5 ③(b) 的论证相反），因此 mixed 比等长几何多付 22–30%。

**含义**：§5.5 的"逻辑块超发 + 硬件动态派发"在本设备上对**同质**负载有效（8x4k 95%），对**异质**负载无效。若要改，只能改任务映射/分核（例如按 query 块切任务、或把长 request 拆分），属于设计层修订，Stage 3 无权擅改。

### 4. 与 DESIGN 的偏离

- **无实现偏差**：kernel body 与 §3.3 伪代码逐句同构（含 `kT` 最后重灌 `b_hi`、`clear_accum=(kq==loop_st)`、`dual_copy` 半区、`sid` 捕获、R7 双分支、A3 溢出零行）。
- **测试期望值按 REVIEW 7-1…7-6 修正**（smoke 8/64、auto-shrink 2,108/31/65,348、`BLOCK_N=96` + `block_n` 一致性 `ValueError`、L0-20 几何改 96 刻度）。
- **tripwire 常数按新分块重标**：`_GRID_MAX_MS 400→120`（截断跑 ~20 ms，Nk=96 后 8x4k 只有 44,672 块）、`_TFLOPS_MAX 378→200`（~90 TFLOPS 的 2 倍）。
- 未改 `_delta_kernel` / `_dq_land_kernel` / 三 launch 契约。

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Develop-flow | `statectl gate 3 --task-dir <op_dir>` 仍去仓库根找 `.stage_state.json` 并报 `E-MISSING`，而正确用法是 `gate 3 --dir <op_dir>`；调度简报与 REVIEW 都写的是 `--task-dir` | `.stage_state.json`（op 目录）存在；`python3 .agents/tools/statectl.py gate 3 --task-dir examples/mla_varlen_bwd/_bwd_kernel` → `E-MISSING`，`gate 3 --dir …` → `{"pass": true}` | `statectl.py`：`--task-dir` 分支在父目录找不到状态文件时回退到 `<task-dir>` 自身，或在报错里提示 `--dir` | R |
| Constraint-doc | 平台 `execution.precision_command` 指向 `/usr/local/python3.12.16/bin/python3`，该解释器缺 `torch/torch_npu`，按原样无法执行 kernel | `statectl platform` 输出的 `execution.precision_command`；该解释器 `import tilelang` 失败（缺 `tqdm`/`torch`），`/workspace/venv-ascend950/bin/python` 正常 | 平台 `execution` 增加 venv 解释器（或 fallback 字段），并在 workflow.md 注明"解释器以 `runtime.toolchain_stamp` 的 venv 为准" | R |
| Info-source | 设计 §8.3 V5/V6 要求的「逐 head 切片」与「cube/vector 利用率」在既有测试模板里没有落点，本轮必须在 `_compare_split` / `--level G` 里新增 | `_bwd_kernel.py::_per_head_report`（新增）、`evidence/msprof_pipe_*_s3.csv`（新增） | tilelang-op-develop 的 `run_case` 模板：默认带 per-head 报告与 msprof 采集步骤 | P |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| packed（`T.dynamic` extent）算子的热 DMA 长度变成运行期 `min(rows, total-start)`：AIC MTE2 118.5→80.5 GB/s、AIV MTE2 0.112→0.348 µs/迭代，每迭代 +0.55 µs（2.93→3.62，cube/mte1/fixpipe 不变）。**dense 的生产单价常数不可直接外推到 packed** | D | `evidence/msprof_pipe_{dense,8x4k,mixed}_s3.csv`（逐块均值）；同几何 A/B：本 kernel 196.9 ms vs dense 150.7 ms（同一时刻重跑） | `repro/TRAP-950-packed-dynamic-dma-length.py`（静态 extent → 常量 `64`；动态 extent → `min(64, total_q-start)` + `int64_t`） | tilelang 0.1.15 / CANN 9.2.1 / ascend-950pr-9579 | 平台 knowledge（`TRAP-950`：动态 extent 的 DMA 长度）+ op-design 的"复用 dense 常数"口径注 |
| packed 变长 backward 的**任务异质性**吃掉 22–30% 利用率：等长 6×5376 = 3.96 µs/迭代（91% 利用率），mixed = 5.06（71%）；`Σ aic_time/28` 与 Task Duration 直接给出空闲比例。"逻辑块超发 + 硬件动态派发"只在同质负载上成立 | D | 同上两份 capture + `/tmp/opencode` 的 A/B 计时（261.5 vs 681.2 ms） | `repro/TRAP-950-packed-dynamic-dma-length.py`（同目录；异质 A/B 见 `_bwd_kernel.py --level G` 的 G2 与 L0-17/18） | 同上 | 平台 knowledge/bottlenecks + `core-split-strategy` 的「异质负载」分支 |

### Transferable Lessons（可迁移教训）

- **"同一源码工厂 + 同一 tile"不等于"同一单价"**：只要索引空间把静态 extent 换成 `T.dynamic`，热 DMA 的长度与地址宽度都会变，MTE2 可掉 30–40%。复用别族的生产常数前，先用**同一 kernel 跑对方的几何**做一次 A/B（本轮正是这样把 196.9 vs 150.7 定位到 MTE2，而不是怀疑 GEMM 或 epilogue）。
- **判定"cube 化是否成功"要按管道拆账，不看总时间**：cube 0.756 与 dense 逐位相同、mte1/fixpipe 也对齐，只有 mte2 异常 ⇒ 结论直接指向 DMA 而不是算法。若只看总时间，会误判成"GEMM 没生效"。
- **动态派发假设必须用"等长 vs 异质"两条几何验证**：单看 mixed 会觉得 kernel 慢，单看 8x4k 会觉得一切正常；两者相除才是调度损失。
- 逐块均值分析要**按 `sub_block_id` 分组**：直接对 CSV 全行求均值会把 `vector0/vector1` 行混进 AIC 列，本轮一开始就把 dense 的 `aic_time` 误读成 375.9 µs（正确值 188.8）。

## Stage 2 (design review) Retrospective — 2026-11-26 — v5（epilogue 旋转）复审 结论 通过（阻塞 0 / 建议 13）

### Skill Flow Issues

| area | issue | evidence | suggested_doc_change | vp_type |
|---|---|---|---|---|
| Review-flow | **生成源里"跨单元依赖是否已被插入"无法按现有资料判定**：`__mix__(1,2)` 的 AIC 分支中**没有**任何以 `PIPE_V`/`PIPE_MTE3` 为前驱的 `asc_sync_wait`（AIV 分支同理没有可解析到的 `PIPE_FIX→PIPE_V`），而 v4（生产、Stage 3/4 全过）与 v5 完全一样 ⇒ 我无法从码面确认 `AIV→AIC` 的 `pT_s/dsT_s` handoff 排序，只能以逐位 parity + 重复性替代 | 本轮 `rot_src.cpp` / `base_src.cpp`（`get_kernel_source()`）的 AIC/AIV 两分支；`REVIEW.md` M3 第 4 行与维度 5 建议 5-1 | 平台 `references/programming.md` 或 `knowledge/` 增一条「mix kernel 生成码的同步判读」：`asc_sync_*` / `asc_sync_intra_*` 各自的作用域（核内 vs sub-block），以及"一条 AIV→AIC 依赖被插入"在码面上的可判定形态 | R |
| Review-flow | **生成源偏移的单位陷阱**：`buf_dyn_shmem)[N]` / `buf_dyn_l1 + N` 的 N 是**元素**（bf16×2 / float×4），设计把 `73,984` 读成"字节"并据此称 UB 表为"保守上界"，实际 UB ≈155 KB（比表多 ~24 KB，因为 2-tile 展开把 epilogue 的 UB scratch 复制成两套）；L1 侧反而恰好正确（155,648 元素 × 2 B = 311,296 B） | `DESIGN.md` §4.5「编译侧旁证…最大字面偏移 73,984 B」vs 生成源 `…)[73984]` 为 bf16；`REVIEW.md` 维度 2 建议 2-1 | design skill 的「预算表口径」加一条：从生成源核对预算必须先确认偏移的 dtype 宽度，并区分"逻辑单版本"与"展开后多份 scratch" | R |
| Review-flow | **机械检查静默退化**：`design_calc_check.py` 的 `ub_budget` 在 §4.5 表新增「版本」列后由 v4 的 `pass` 退化为 `skip`（"no per-level buffer tables parsed"），输出里只差一个 status 值，检视者容易当成"无需检查"放过去 | 本轮 `design_calc_check --design …` JSON；`REVIEW.md` M2 | `design_calc_check.py` 的 skip 分支输出 `regressed: true` + "与上一版本相比未解析的新增列名"，或在 `statectl gate 2` 里把"由 pass 变为 skip"记为 warning | R |
| Info-source | **Stage 4 的逐管道每迭代常数没有结构化字段**：设计反复引用"AIV 忙时 1.95–2.10 µs"，我在 `perf_records.jsonl` / `opt_log.md` / `perf_feedback.md` 里都找不到该区间，只能由 `aiv_*_share × us_per_iteration` 反算（得 2.131 / 2.844 / 2.233），并发现 mixed 的物理地板应是 ≈383 ms 而不是 286 ms | `perf_records.jsonl` 的 `final-c3-stages3` 三行（`pipe.vector.*_share`）；`REVIEW.md` 建议 8-1 | optimize skill 的 `perf_records` schema 增 `per_iteration_pipe_us`（逐管道 µs/迭代，直接由 share × us/it 生成），避免下游设计引用无出处区间 | R |

### Value Point Proposals

| title | vp_type | evidence | repro | toolchain_stamp | target_doc |
|---|---|---|---|---|---|
| 「滞后消费者」的**版本身份**可以从生成源的 L1 偏移直接判定，且这是能区分改动的判据：A 半体 `GB` 读 `+143360(pT_B)/+149504(dsT_B)/+86016(qT_B)/+122880(doT_B)`，同半体 `F/EP` 写 `+131072/+137216/+61440/+106496`（B 半体相反），12 块在 `buf_dyn_l1` 中连续、末地址 155,648 元素 = 311,296 B | D | `DESIGN.md` §6.3 C 的 WAR 行、§1.6.4 F-3；`REVIEW.md` M3 第 4 行 | `history_version/design_probe_epilogue_rotation.py --compile` → `get_kernel_source()` 读偏移（本轮落 `/tmp/opencode/rot_src.cpp`） | TileLang ascend 0.1.15 + CANN 9.2.1, Ascend950PR_9579 (dav-3510), 2026-11-26 | design skill「结构判据清单」+ 平台 knowledge（mix kernel 的 L1 布局判读） |
| **结构判据必须先对"改动前的产物"跑一遍**：v5 把「`GB` 前无 `PIPE_V` 前驱 wait」当作"成立与否的判据"，但 v4 生产体的 AIC 分支同样没有该 wait ⇒ 判据不区分版本，可能给出假阳性 | D | `base_src.cpp` / `rot_src.cpp` 的 AIC 分支 grep（`PIPE_V` 只出现在 `if ASC_IS_AIV` 段）；`REVIEW.md` 建议 5-1 | 同上一行的探针（`mode=base` 与 `mode=rot` 各编译一次） | 同上 | design skill「结构判据」要求：判据须在 pre/post 两版产物上验证"能区分" |
| 旋转类收益的**同日 A/B 基线**是必需的：同一台机、隔一天，base 体（3.165/4.267/4.186）与设计值（3.167/4.267/4.191）逐点吻合，而 rot 体（2.123/3.240/2.709）与设计值差 +1.5% / +3.4% / −2.7% ⇒ 跨上下文抖动约 ±3.4%，只测 rot 会把抖动读成收益或损失 | D | `REVIEW.md` M3 第 2 行；设计 §1.6.4 F-2 表 | `TMPDIR=/workspace /workspace/venv-ascend950/bin/python -c "…time_case(...)"`（本轮的 `/tmp/opencode/rot_time_repro.py`） | 同上 | optimize skill 的计时模板（每次候选必须同场测基线）+ §8.4 的验收口径 |

### Transferable Lessons（可迁移教训）

- 写"结构判据"时，先把同一条判据应用到**改动前的产物**上：若同样成立，它只是必要条件而不是判据。本例「`GB` 前无 `PIPE_V` 等待」在 v4 生产体上就成立，真正有区分力的是「`GB` 读的缓冲偏移属于另一半体版本」。
- 从生成源核对片上预算时，先确认**偏移单位**（元素 vs 字节）与 dtype 宽度，再确认**展开是否复制 scratch**：两处形状展开（本例 2-tile）会把同一逻辑缓冲变成多份物理分配（UB 表因此少算 24 KB），而 L1 表恰好正确（元素 ×2 B）。
- "调度等价"的最强证据是对**冻结且已生产验证的实现**逐位对照（`dk/dv` bit-exact），而不是 fp64 容差对照：前者能抓住"读错版本"这类只在部分块上出现的错误，且给 Stage 3 一个可执行的回归门。
- 评估性能重设计时，必须当场同时测"改动前"与"改动后"：跨日/跨上下文有 ~3% 抖动，单侧测量会把抖动误读成收益；把"跑间抖动"写进验收口径，比写一个精确到小数点后三位的目标值更有用。
