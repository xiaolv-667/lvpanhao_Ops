# NPUIR 专用约束：profile-collection

以下保留原平台的 API、命令和参数说明；公共适配与采集流程仍按共享文档执行。

本文档服务 `tilelang-op-optimize` 的 Stage 4。性能调优对象是迁移后 benchmark 实际参数化的、可运行且非 smoke 的 workload；合并对象是它们实际调用的 Stage 3 kernel。`msprof op Task Duration(us)` 是唯一 kernel 时延口径，不使用 NPU event 或端到端耗时。

逐案例确认它实际调用的 Stage 3 kernel，记录可唯一定位实现文件及函数的 `kernel_id` 和用于 `msprof op --kernel-name` 的 `target_kernel_name`。一个案例若调用多个目标 kernel，分别登记 workload–kernel 对；无法确认映射时记录原因并停止该案例的性能采集。TileOPs 原有 op 层路由只用于查明映射，不是 Stage 4 调优单位，也不应修改。

## 3. 串行运行 msprof op

对每个 `kernel_id + workload_id` 逐个运行 `msprof op`，禁止并发 profiling。使用 benchmark workload 参数直接运行 Stage 3 kernel 或 `perf_opt/` 下实际候选文件，**不运行需要 Stage 5 wrapper 的正式 benchmark**。

命令模板：

```bash
msprof op \
  --kernel-name={target_kernel_name} \
  --output={output_dir} \
  --launch-count=20 \
  --warm-up=5 \
  --dump=off \
  --aic-metrics=BasicInfo,PipeUtilization,ArithmeticUtilization,Memory,MemoryUB,MemoryL0,L2Cache,ResourceConflictRatio \
  python {actual_kernel_file}.py {workload_args}
```

如果当前 CANN 版本不兼容逗号形式的 `--aic-metrics`，退回：

```bash
msprof op --kernel-name={target_kernel_name} --output={output_dir} --launch-count=10 --warm-up=5 --aic-metrics=Default python {actual_kernel_file}.py {workload_args}
```

新建 `msprof op` 输出目录后，先确保 group/other 不可写，例如 `chmod 700 {output_dir}`；否则某些环境会拒绝采集或写入失败。

如果 `msprof op` 在输出目录下生成 `OPPROF_xxx` 子目录，记录实际 `raw_profile_dir`。

注意：`msprof op` 多 launch 稳定性需要靠多次独立运行，不要用 CSV 记录条数推断实际 launch 次数。
