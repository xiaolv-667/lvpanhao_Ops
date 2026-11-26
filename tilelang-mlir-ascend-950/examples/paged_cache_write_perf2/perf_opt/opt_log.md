# paged_cache_write — Stage 4 round 2 (fp8 store-issue) tuning log

- Task: `paged_cache_write-paged_cache_write-20261125` (task dir
  `examples/paged_cache_write_perf2`), operator `PagedKVCacheWriteFwdOp`, mode `full`.
- Platform: `ascend-950pr-9579` / `ascend` (Ascend950PR_9579, 28 Cube / 56 Vector), TileLang
  ascend 0.1.15 + CANN 9.2.1, single card.
- Read-only baseline (round-1 winner, in production):
  `examples/paged_cache_write_perf/perf_opt/paged_cache_write.py`.
- Delivered: `perf_opt/PagedKVCacheWriteFwdOp.py` (self-contained: golden + kernel + layered
  tests).
- Round-2 goal: fp8 `qwen3-235b-fp8` **≤25 µs** (from 31.9), prefill ≤45 µs and every other
  case non-regressing, byte-exact semantics unchanged.

## 1. Measurement protocol

The acceptance metric is the framework's msprof path (`tileops/benchmark/msprof.py`):
`msopprof --warm-up=5 --launch-count=10`, latency = **median of the 10 measured launches** in one
process. `perf_opt/experiments/bench_one.py` reproduces exactly that (the child launches the op
`warm_up + launch_count` times; the 10 captured rows are reduced by median), so the numbers in
this log are directly comparable to the production report `20261125_095321_930234`
(fp8 31.9 µs, prefill 43.3 µs). `perf_opt/experiments/run_experiments.py` drives the workload set
and appends `perf_records.jsonl`. Reps: 3 independent captures per workload for baseline and the
delivered file, 2 for exploratory branches.

Round-1 numbers quoted from the previous task were taken with a colder protocol (one launch per
capture); this log only compares numbers measured with the protocol above.

## 2. Baseline (round-1 kernel, this task's `phase=baseline`, round 0)

| workload | Task Duration (µs) | samples | tile |
|---|---:|---|---|
| llama-8b-decode-bfloat16 | 3.045 | 3.046 / 2.791 / 3.045 | 2x1024 |
| llama-8b-decode-float16 | 2.822 | 2.822 / 2.725 / 3.123 | 2x1024 |
| llama-8b-prefill | 43.485 | 43.597 / 43.485 / 43.309 | 32x1024 |
| qwen3-235b-fp8 | **31.778** | 31.940 / 31.778 / 31.758 | 16x512 |
| gpt-oss-120b | 4.805 | 4.776 / 4.946 / 4.805 | 8x512 |
| odd-tokens-and-page | 2.083 | 2.083 / 1.829 / 2.162 | 1x128 |
| qwen3-235b-fp8-fp16 | **13.055** | 12.956 / 13.055 / 13.075 | 8x512 |

These reproduce the production report (31.9 / 43.3 / 13.1 / 4.7 / 3.1 / 3.4 / 2.1 µs), which is
the check that the local harness measures what acceptance measures.

## 3. Round 1 — which hypothesis paid

**Phenomenon.** With round 1's structure the fp8 case sat at 0.40 TB/s (25% MBU) while the
16-bit prefill ran at 1.4 TB/s. Both move the same kind of row tiles; the difference is the store
path. The fp8 build has to store one byte per element, so the scattered rows go out as one
UB→GM `T.copy` each (8192 descriptors for 4096 tokens x 2 pools), and the auto-scheduler emits
`asc_sync_pipe(PIPE_MTE3)` — a **drain of the whole store pipe** — before every one of them
(same-pipe producer/consumer pair, which it cannot prove disjoint). That is a fixed per-row cost
of core time, i.e. the case is store-*issue* bound, not bandwidth bound.

**Branch m1 (adopted): manual sync on the quantised path.** Compile the fp8 program with
`pass_configs={"tl.enable_auto_schedule": False}` and state the only two dependencies the kernel
has:

```python
T.ascend_set_flag("MTE2_V", eid);  T.ascend_wait_flag("MTE2_V", eid)   # loads -> conversion
...                                                                     # conversion pass
T.ascend_set_flag("V_MTE3", eid);  T.ascend_wait_flag("V_MTE3", eid)   # conversion -> stores
for i in T.unroll(rows): ... per-row T.copy ...                        # no sync between rows
T.ascend_pipe_barrier("PIPE_MTE3")                                     # one drain per block
```

Two row stores never target the same row (distinct non-negative slots are the operator's
contract), so no ordering is needed between them; the end-of-block drain makes the writes visible
before the kernel exits. `eid` is 0 for the K side and 1 for the V side so each macro's flags are
its own pair.

**Branch m2: 1024 lanes.** Measured globally it was a tie on fp8 (19.43 vs 19.55 µs, inside
noise) and slightly better on fp8-fp16 (7.33 vs 7.62 µs), so the lane count is raised to 1024
*only* for the quantised build where the conversion pass scales with lanes; the 16-bit build
(prefill at the HBM ceiling) stays at 512 and was not re-measured.

### Candidate vs current best (current best = round-1 kernel)

| workload | current best (µs) | m1 auto-sync control (µs) | m1 manual sync (µs) | Δ vs current best | L0 |
|---|---:|---:|---:|---:|---|
| qwen3-235b-fp8 (16 rows) | 31.778 | 37.56 (probe, same shape) | **20.72** | -34.8% | 10/10 |
| qwen3-235b-fp8-fp16 (8 rows) | 13.055 | — | **10.25** | -21.5% | 10/10 |
| llama-8b-prefill | 43.485 | — | 44.12 | +1.5% (noise, same binary) | 10/10 |
| llama-8b-decode-bf16 | 3.045 | — | 2.96 | -2.8% (noise) | 10/10 |
| gpt-oss-120b | 4.805 | — | 4.90 | +2.0% (noise) | 10/10 |
| odd-tokens-and-page | 2.083 | — | 2.17 | +4.2% (noise) | 10/10 |

The probe (`perf_opt/experiments/probe_manual_sync.py`) isolates the change on the fp8 shape only
and checks both builds against a host reference: auto-schedule **37.56 µs** vs manual sync
**19.99 µs**, both byte-exact, i.e. the drain removal alone is a 1.88x on the store path. The real
kernel starts lower (31.78) because it hoists the scales and keeps the round-1 tile policy, and
lands at 20.72 µs with the same tile.

L0 per branch (`perf_opt/logs/all/m1_*.log`): L0 10/10, L1 2/2, L2 3/3, boundary 10/10, all
byte-exact (`max_diff = 0`).

## 4. Round 2 — retune the tile around the cheaper store

With the per-row drain gone the block's fixed cost dominates again, so the row tile wants to be
taller. Sweeps (median of 2-3 captures, fp8 shape as in the manifest):

| build | rows per block | Task Duration (µs) |
|---|---:|---:|
| fp8, 4096 tokens, 512 lanes | 8 | 30.28 |
| | 16 | 20.72 |
| | 32 | 20.15 |
| | **48** | **19.55 / 19.43 / 19.50** |
| | 64 | 23.47 (UB pressure: 196 KB of 248 KB) |
| fp8, 4096 tokens, 1024 lanes | 48 | 19.43 (tie with 512) |
| fp8-fp16, 1024 tokens, 512 lanes | 8 | 10.25 |
| | 16 | 8.47 |
| | **32** | **7.62** |
| | 48 | 9.49 (too few blocks: 22 over 56 cores) |
| fp8-fp16, 1024 tokens, 1024 lanes | 32 | 7.33 |

Policy that follows: both builds tile `tokens / 32` rows per block, the quantised build caps at 48
rows (vs 32 for 16-bit) and uses 1024 SimtVF lanes once the tile holds ≥16384 elements (the
conversion pass scales with lanes; the 16-bit build is already at the HBM ceiling and stays at
512). The `rows <= tokens/4` and `rows*cols >= 256` guards from round 1 are kept.

## 5. Round 3 — final file and full-set verification

`perf_opt/PagedKVCacheWriteFwdOp.py` is the round-1 kernel with the manual-sync store path and the
round-2 tile policy, written as one self-contained module. L0 10/10, L1 2/2, L2 3/3, boundary
10/10 byte-exact (`perf_opt/logs/final/*.log`) plus the three smoke geometries
(`experiments/smoke_precision.py`, `max_diff = 0`). fp8 raw-byte comparison (clamp/RNE/NaN
canonicalisation), negative-slot skip and distinct-slot last-writer-wins are all covered by those
levels and unchanged.

**Repeat-launch determinism** (the manual-sync build owns its synchronization, so a race would
show as a per-launch difference): `experiments/determinism_check.py` runs the delivered module's
fp8 case three times on one set of buffers — launch 0 compared byte-exactly with the workload
reference, launches 1 and 2 compared with launch 0 — `[det] DETERMINISTIC_PASS` (3 launches,
fp8 4096x512).

## Final Performance Test Data

Same final candidate for every row (`final_manual_sync`, `perf_opt/PagedKVCacheWriteFwdOp.py`,
sha256 in `perf_records.jsonl`), framework msprof protocol, median of 3 captures of 10 launches.

| kernel_id | workload_id | Task Duration baseline (µs) | Task Duration final (µs) | final candidate_id |
|---|---|---:|---:|---|
| tileops/kernels/attention/paged_cache_write/paged_cache_write_kernel/_paged_cache_write_kernel.py::_paged_cache_write_kernel::main | llama-8b-decode-bfloat16-bfloat16 | 3.045 | 2.908 | final_manual_sync |
| tileops/kernels/attention/paged_cache_write/paged_cache_write_kernel/_paged_cache_write_kernel.py::_paged_cache_write_kernel::main | llama-8b-decode-float16-float16 | 2.822 | 2.970 | final_manual_sync |
| tileops/kernels/attention/paged_cache_write/paged_cache_write_kernel/_paged_cache_write_kernel.py::_paged_cache_write_kernel::main | llama-8b-prefill-bfloat16-bfloat16 | 43.485 | 44.069 | final_manual_sync |
| tileops/kernels/attention/paged_cache_write/paged_cache_write_kernel/_paged_cache_write_kernel.py::_paged_cache_write_kernel::main | qwen3-235b-fp8-bfloat16-float8_e4m3fn | 31.778 | 19.292 | final_manual_sync |
| tileops/kernels/attention/paged_cache_write/paged_cache_write_kernel/_paged_cache_write_kernel.py::_paged_cache_write_kernel::main | gpt-oss-120b-bfloat16-bfloat16 | 4.805 | 4.777 | final_manual_sync |
| tileops/kernels/attention/paged_cache_write/paged_cache_write_kernel/_paged_cache_write_kernel.py::_paged_cache_write_kernel::main | odd-tokens-and-page-float16-float16 | 2.083 | 1.955 | final_manual_sync |
| tileops/kernels/attention/paged_cache_write/paged_cache_write_kernel/_paged_cache_write_kernel.py::_paged_cache_write_kernel::main | qwen3-235b-fp8-fp16-float16-float8_e4m3fn | 13.055 | 7.381 | final_manual_sync |

## Derived metrics and goal check

| workload | baseline → final | change | final effective BW | MBU vs 1.6 TB/s |
|---|---|---:|---:|---:|
| qwen3-235b-fp8 | 31.778 → 19.292 µs | **-39.3% (1.65x)** | 0.654 TB/s | **40.9%** (was 25.3%) |
| qwen3-235b-fp8-fp16 | 13.055 → 7.381 µs | **-43.5% (1.77x)** | 0.427 TB/s | 26.7% (was 15.4%) |
| llama-8b-prefill | 43.485 → 44.069 µs | +1.3% (noise) | 1.52 TB/s | 95% |
| llama-8b-decode-bf16 | 3.045 → 2.908 µs | -4.5% | 0.180 TB/s | 11% |
| llama-8b-decode-fp16 | 2.822 → 2.970 µs | +5.2% (noise) | 0.177 TB/s | 11% |
| gpt-oss-120b | 4.805 → 4.777 µs | -0.6% | 0.220 TB/s | 14% |
| odd-tokens-and-page | 2.083 → 1.955 µs | -6.1% | launch floor | launch floor |

Goal: **fp8 ≤25 µs** → 19.29 µs (1.30x margin under the bar, 1.65x over the baseline);
**prefill ≤45 µs** → 44.07 µs; every other case at or below baseline within its run-to-run
spread. The two 3 µs cases are the only "increases", measured at ±8-13% spread; a 5-capture A/B
on the same binaries gave decode-bf16 3.113 (baseline) vs 3.052 (this kernel) and decode-fp16
2.896 vs 2.995 — i.e. run noise, and the 16-bit build is the same program in both files (same
tile, same store path, no manual sync).

## 7. Why the remaining headroom is not reachable from the kernel

fp8 now moves 12.6 MB of roofline traffic in 19.3 µs (0.65 TB/s, 41% MBU). The floor for that
payload at the measured streaming ceiling (~1.4 TB/s) is ~9 µs, so ~10 µs is still per-row store
issue: 8192 descriptors at ~65 ns of core time each (`8192 x 65 ns / 56 cores ≈ 9.5 µs`). The
only way to remove more is fewer descriptors, and the destination rows are a random permutation
subset of the pool:

- merging adjacent destination rows: with 8192 rows drawn from 65536, ~12% of consecutive pairs
  are adjacent, so even a free in-kernel sort would save ~12% of the descriptors;
- writing several rows per descriptor is impossible unless the destinations are contiguous, and
  the pools are separate tensors for K and V;
- the alternative store path (per-lane vector store) cannot widen a one-byte element: it measured
  157 µs on this shape in round 1 and is incorrect on small fp8 tiles
  (`TRAP-950-fp8-scalar-store`).

This is a property of "scatter one row at a time" on this backend, not of the algorithm: the
operator's own semantics (write exactly the named rows) forces row-granular destinations. Recorded
here as the remaining headroom; no `[DESIGN_LIMIT]` is claimed because the target was met with
margin and nothing in `DESIGN.md` is contradicted.

## 8. Smoke / skipped

| workload_id | kind | precision |
|---|---|---|
| paged-write-smoke-float16-float16 | smoke | PASS (byte-exact) |
| paged-write-smoke-bfloat16-bfloat16 | smoke | PASS (byte-exact) |
| paged-write-smoke-bfloat16-float8_e4m3fn | smoke | PASS (byte-exact, NaN canonicalised) |

Nothing skipped.

## 9. Reproduction

```bash
cd examples/paged_cache_write_perf2
source ../../.agents/platforms/toolchains/ascend/env-950pr.sh
export TMPDIR=/workspace TILELANG_CACHE_DIR=/tmp/opencode/tl-cache-opt-paged2
PY=/workspace/venv-ascend950/bin/python

# the manual-sync effect in isolation (byte-exact check + 5/10 msprof protocol)
$PY perf_opt/experiments/probe_manual_sync.py --variant both
# baseline / candidate / final over the workload set
$PY perf_opt/experiments/run_experiments.py --module perf_opt/PagedKVCacheWriteFwdOp.py \
    --candidate-id final_manual_sync --round 2 --phase final --reps 3
$PY perf_opt/PagedKVCacheWriteFwdOp.py --level all
```

## 10. Skill Retrospective

**What worked.** The previous round's diagnosis (store-issue bound, one drain per row) was
actionable once the *cause* was attributed to `insert_sync.cc`'s same-pipe rule rather than to
bandwidth. Reading the sync-insertion code and then expressing the kernel's real dependencies by
hand turned a 1.6x into a 1.9x on the store path in one experiment, and the first probe already
carried the byte-exact check, so the mechanism was never in doubt.

**What cost time.**
1. The template's `one()` macro gained a parameter (`eid`) for the manual flags, and two of the
   four call sites were missed on the first edit — caught immediately by the L0 run, but a macro
   signature change should always be followed by a grep for call sites.
2. The probe's first version inlined `KS[0]` inside the conversion loop instead of hoisting the
   scale, so its "auto" control measured 37.6 µs against the real kernel's 31.8 µs. Fine for a
   relative A/B, but it cost a detour to explain; probe scripts should mirror the kernel they
   claim to model.
3. Automatic scheduling is the *default* for a reason: switching it off means every future edit to
   that prim_func must re-derive the dependency list. This is noted in the module docstring and
   next to the flags.

**Value points.**
- **D** — *same-pipe drain is what makes row-granular DMA stores expensive; manual sync removes
  it*: fp8 4096x512 shape, auto 37.56 µs vs manual 19.99 µs, both byte-exact; the 16-bit prefill
  keeps its lane store and is unaffected. Evidence: `perf_opt/experiments/probe_manual_sync.py`
  (`{"variant": "manual", "byte_exact": true, "median_us": 19.986}`),
  `perf_records.jsonl` rows `r1_m1_manual_sync` / `final_manual_sync`; repro = that probe;
  toolchain_stamp `a010a95909e4e752d0d0b602885eca4b6fab233edca2d3e540248ed71c2135dc`. Suggested
  write-back: extend `TRAP-950-rowstore-mte3-drain` (written last round) with the manual-sync
  countermeasure and these numbers.
- **P** — *for a scatter whose destinations are provably distinct, replace the auto-scheduler's
  same-pipe conservatism with two explicit flag pairs plus one per-block pipe barrier*: the code
  shape and the boundary (needs a correctness re-run at every L0/L1/boundary level; per-tensor
  event ids so the two macro invocations do not share a stale flag). Tier 1 (needs a second
  independent task).
- **R** — *a stage-4 probe harness must measure with the acceptance protocol*: the whole first
  round's numbers (cold single-launch) were 15% below the framework's warm median, which is the
  difference between "meets the bar" and "is 8 µs under the bar". Propose adding to the Stage 4
  skill: when the acceptance report uses `warm_up/launch_count`, the tuning harness must mirror
  it. Tier 2.
- **C** — `CASE-paged-write-round2` (store-issue → manual sync) for `knowledge/cases.md`.

**Remaining headroom**: per-row descriptor issue (~10 µs of the 19.3 µs); would need multi-row
descriptors, which the scattered destination rules out (see §7).

## 11. Known limits of this change

- The manual-sync build must be re-verified (L0/L1/L2/boundary) after any edit to the fp8
  prim_func: with `enable_auto_schedule=False` the compiler no longer inserts the syncs, so a new
  buffer or phase needs its own flag pair.
- The flag ids (0/1) are per-block scoped pairs (set and wait around the same dependency), which is
  what the verified cases exercise; they are not a general event allocator.
- The 16-bit path is untouched (auto-schedule, lane store) and remains at ~1.4 TB/s.

## 12. Production adoption (round 2)

### Overwrite / backup

| file | action | sha256 |
|---|---|---|
| `…/paged_cache_write_kernel/perf_opt/_paged_cache_write_kernel.py` | **overwritten** with this task's `perf_opt/PagedKVCacheWriteFwdOp.py` | `c74187e1bd71deea34b296ba477beab60ad1d55654356ea80bfd4cccf3c29b66` |
| `…/paged_cache_write_kernel/history_version/_paged_cache_write_kernel_pre_round2.py` | **backup** of the round-1 integrated kernel | `99a43c0d94adc6829f4b058069379fe1af66b7995c7a188efe099922c1687685` |

Nothing else changed: the wrapper (`paged_cache_write.py`, which already routes the tile through
`pick_config`) is untouched, and so are the op layer, the manifest, the tests and the benchmark.
The factory signature and `pick_config(tokens, heads_kv, dim, physical_tokens, in_dtype,
cache_dtype)` are the same as round 1, so the wrapper needs no edit.

### Production regression (cwd `examples/TileOPs`)

1. `pytest tests/ops/test_paged_kv_cache_write.py -q` → **15 passed** (4.9 s).
2. `perf_opt/experiments/production_check.py` (op layer, no eager geometry, 3 launches per case):

| case | routed tile | check | result |
|---|---|---|---|
| qwen3-235b-fp8 (4096x512, bf16->e4m3) | `(48, 512)` | byte-exact vs reference (launch 0) + launches 1,2 vs launch 0 | PASS |
| llama-8b-prefill (8192x1024, bf16->bf16) | `(32, 1024)` | byte-exact vs reference + repeat-launch determinism | PASS |

   `[production] BYTE_EXACT_DETERMINISTIC_PASS`. The determinism part matters for this round
   specifically: the quantised build compiles with `tl.enable_auto_schedule=False` and owns its
   synchronization, so a race would appear as a per-launch difference rather than a compile error.

### Single-operator report (msprof)

```
TILEOPS_TASK_DIR=…/examples/paged_cache_write TMPDIR=/workspace \
  /workspace/venv-ascend950/bin/python -m tileops.reporting.cli run \
  --op PagedKVCacheWriteFwdOp --prof-mode msprof
```

Run **`20261125_181200_990273_PagedKVCacheWriteFwdOp`** — `status=passed`, correctness 100%
(15/15, avg max abs error 0.00e+00), **`invalid = 0`**:

| case | round-1 production (µs) | **round-2 production (µs)** | change | achieved BW | MBU vs 1.6 TB/s |
|---|---:|---:|---:|---:|---:|
| qwen3-235b-fp8-bfloat16-float8_e4m3fn | 31.9 | **19.621** | **-38.5%** | 0.643 TB/s | **40.2%** |
| qwen3-235b-fp8-fp16-float16-float8_e4m3fn | 13.1 | **7.315** | **-44.2%** | 0.431 TB/s | 27.0% |
| llama-8b-prefill-bfloat16 | 43.3 | **43.166** | -0.3% | 1.556 TB/s | 97.3% |
| gpt-oss-120b-bfloat16 | 4.7 | 4.889 | +4.0% (noise) | 0.215 TB/s | 13.4% |
| llama-8b-decode-bfloat16 | 3.1 | 3.117 | +0.5% | 0.168 TB/s | 10.5% |
| llama-8b-decode-float16 | 3.4 | 3.351 | -1.4% | 0.157 TB/s | 10.0% |
| odd-tokens-and-page-float16 | 2.1 | 2.212 | +5.3% (noise) | launch floor | launch floor |

The `config` column of the report now shows the tiles that actually ran — 2x1024, 32x1024,
48x512, 8x512, 1x128, 32x512 — i.e. the round-2 policy is in effect in the integration package.

Reading: the target case lands 0.3 µs off the tuning measurement (19.62 vs 19.29, inside the
run-to-run spread of the 10-launch medians) and 12.4 µs under the round's 25 µs bar; prefill is
unchanged within noise and stays at ~1.56 TB/s; the three small cases move by ±5%, which is their
own spread (they were also ±8-13% in the tuning captures). The two `F` marks in the report's
acceptance summary are the memory-bound MFU-vs-90% policy (capped at 2.26% by HBM bandwidth),
which is a property of the operator class, not of this change; `invalid` is 0 for every case — the
prefill case that tripped the "achieved > nominal peak" guard in one round-1 run now sits at
1.556 TB/s, comfortably under it.

### Records

`perf_records.jsonl` gained 7 rows (`phase=merged`, `candidate_id=production_integrated_round2`,
`parent_id=final_manual_sync`, round 4, artifact = the integrated module with its sha256,
`msprof_raw_path` = this run's artifact dir). The earlier rows are unchanged (append-only); the
`final` rows are still the ones the Final Performance Test Data table reconciles against.

### Rollback

Copying `history_version/_paged_cache_write_kernel_pre_round2.py` back over
`perf_opt/_paged_cache_write_kernel.py` restores round 1 exactly (same factory signature and
`pick_config` surface, so the wrapper needs no change either).
