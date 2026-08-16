# Task 1 data/model migration baseline

Date: 2026-08-15/16 UTC

Baseline start revision: `fe5b90545872d6155968140b1a88b4f92b00461f`

Corrected source revision: `cd858d87b6d6927ab05feeb2286522481007c00a`

Toolchain: `nightly-2026-04-07`, rustc `1.96.0-nightly (bcded3316 2026-04-06)`

Lockfile SHA-256: `a89d3b90c2c322c60dfc3803b53bb38f6de3a9760eb156a53f8ba5fadd5ab860`
Status: contract, build, quarantined-archive, real-data CPU indicator, CUDA-build, and one-card RTX 4090 execution evidence captured; the measured CUDA lane is still hybrid and CPU/GPU data readiness is not approved.

This document freezes the behavior that the Vortex/f64 migration must preserve or intentionally replace. It is baseline evidence, not a correctness or profitability approval. In particular, `.fstore` f32 output and Polars model inputs below describe the current implementation; they are not the desired end state.

## Fixed contract fixtures

| Contract | Fixture SHA-256 | Result |
|---|---|---|
| `.fstore` layout, selected columns, nested row windows, labels, and recorded f64-to-f32 loss | `1ca701a47ac7e34943145bdee9f85d4688df140b31701c2c02b3797eab6afdfd` | PASS |
| Polars-to-model frame ordering, strict null/non-finite rejection, labels, deterministic logistic predictions, metadata, and save/load | `9f502365f837239b66d191f7cb4345a06643e96121ad2b6e3f753aeb5554ee84` | PASS |

The data fixture records both the original f64 bit patterns and the current widened-f32 result. Every high-precision value changes bits across the current `f64 -> f32 -> f64` boundary. The model fixture independently records the current f32 probability bits so Polars removal cannot silently change the small deterministic path.

Focused commands:

```text
cargo +nightly-2026-04-07 test -p neoethos-data --test feature_store_contract -- --nocapture
cargo +nightly-2026-04-07 test -p neoethos-models --test model_frame_contract -- --nocapture
```

Both passed locally and on the isolated Linux baseline source. A 2026-08-16 refresh on the current working tree passed the data contract with no diagnostic. The first refreshed model build passed but exposed one real vendored dead-code warning for the unused placeholder `sklears_core::CacheAnalyzer::read_perf_counters`; the function had no caller and returned only zero counters, so it was deleted. The incremental rerun passed in 25.08 seconds with zero INFO/WARN/ERROR records. The complete RED-diagnostic and GREEN logs are under `task1-contract-refresh2/` and `task1-model-warning-green/`; their downloaded files match all 13 remote SHA-256 entries.

The broader `neoethos-data --all-targets` run also passed: 175 tests passed, three explicitly ignored baseline tests remained ignored, no test failed, and the complete compiler/test output contained no warning, error, or panic. It completed in 48.93 seconds with 3,848,476 KiB peak RSS. The eight downloaded files under `task1-cpu-alltargets/` match their remote SHA-256 manifest.

## Hardware and execution controls

The measured Linux host was a Vast.ai instance with:

- AMD EPYC 7282, 64 visible logical CPUs / 32 physical cores / 2 sockets;
- cgroup CPU quota `6143999/100000`, so Rust's effective parallelism is 61 and the declared automatic worker limit is 59;
- NVIDIA GeForce RTX 4090, compute capability 8.9, 24,564 MiB, driver 590.48.01;
- 1.1 TB overlay volume, with no other NeoEthos build/test/benchmark launched during each measured command;
- `CARGO_BUILD_JOBS=59`, the pinned lockfile, release profile, and independent target directories for clean builds.

The data/model runtime benchmarks used three explicit warmups followed by ten measured samples. Clean builds use three distinct empty target directories. Incremental builds reuse exactly the first clean target after three warmups. Full command output, `/usr/bin/time -v`, cgroup accounting, target byte counts, and periodic process/GPU/disk telemetry are retained under `target/audit-logs/task1/vast-47806434/` and are intentionally ignored by Git.

## Data runtime baseline

Fixture: 250,000 M1 rows, 64 f32 feature columns, eight selected GA columns. Page cache was warm only after the declared three warmups.

| Operation | Median | p95 | Notes |
|---|---:|---:|---|
| canonical Vortex OHLCV scan | 5,503,268 ns | 6,807,041 ns | about 45.43 million rows/s |
| `.fstore` eight selected columns | 1,885,206.5 ns | 1,919,452 ns | current f32 mmap path |
| `.fstore` 100,000-row x 64-column window | 47,889,856.5 ns | 48,237,193 ns | materializes the window |
| repeated GA selected-column access | 15,781,666.5 ns | 19,741,839 ns | fixed checksum |

Sizes and memory:

- `.fstore`: 64,000,000 bytes;
- Vortex OHLCV: 3,116,224 bytes;
- Vortex is 4.869% of the `.fstore` bytes for these different logical payloads; this is not an apples-to-apples speed or compression claim;
- runtime test peak RSS: 138,240 KiB;
- clean compile/run command peak RSS: 3,969,288 KiB.

The raw ten-sample arrays and checksums are in `task1-baseline-run2/data.log`. These values are the pre-migration reference; the migration gate compares equivalent selected-column/window and GA workloads after the Vortex feature store exists.

## Model runtime baseline

Fixture: nine rows, four named features, 64 deterministic logistic fit+predict operations per measured sample.

| Operation | Median | p95 | Peak runtime RSS | Checksum |
|---|---:|---:|---:|---:|
| 64 fits plus predictions | 2,210,871 ns | 2,220,542 ns | 7,192 KiB | 23,618,025,154,880 |

The successful fresh model compile plus benchmark took 3:05.69 wall time, used 3,067% CPU, and peaked at 4,759,736 KiB RSS. Its full evidence is under `task1-model-baseline-run3/`.

## Build baseline and diagnostic classification

All three fresh `neoethos-data` and all three fresh `neoethos-models` release builds passed in independent target directories.

| Clean release target | Run 1 | Run 2 | Run 3 | Median | p95 | Maximum RSS | Target bytes (range) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `neoethos-data` contract, no run | 117.93 s | 119.19 s | 118.38 s | 118.38 s | 119.19 s | 3,968,980 KiB | 2,307,905,574–2,307,915,935 |
| `neoethos-models` contract, no run | 186.91 s | 188.58 s | 187.04 s | 187.04 s | 188.58 s | 4,893,652 KiB | 5,097,368,920–5,097,375,672 |

The three clean data builds added no cgroup-throttled period. The clean model builds added 255/248/266 throttled periods and 54.99/50.95/59.62 seconds of aggregate throttled CPU time respectively, despite there being no competing NeoEthos job. This is direct evidence that Cargo's job count plus the independent `-Z threads=8` rustc frontend width can burst above the container's CPU quota; a successful build is not evidence of correct aggregate scheduling.

The reused-target protocol also completed three warmups followed by ten measured runs for each target. These were intended to establish a true incremental baseline, but the missing watched path described below forced `vector-ta` and its dependants to rebuild on every invocation.

| Reused release target | Median | p95 | Maximum RSS | Throttled periods/time across measured runs |
|---|---:|---:|---:|---:|
| `neoethos-data` contract, no run | 92.455 s | 93.48 s | 4,085,152 KiB | 0 / 0 s |
| `neoethos-models` contract, no run | 132.16 s | 133.77 s | 4,998,072 KiB | 0 / 0 s |

The two-second telemetry stream contains 1,856 samples. Its largest observed compiler/native-builder process population was 194 (`69 c++`, `69 cc1plus`, `22 cc`, `20 cc1`, `10 rustc`, `2 cmake`, `1 ninja`, `1 cargo`); this is process population, not proof that all 194 were simultaneously CPU-active. Peak one-minute load was 53.41, peak measured system memory use was 38,105,477,120 bytes, and GPU utilization remained 0% as expected for this CPU/build protocol. Combined with the clean-model cgroup throttling, this proves that independent Cargo, rustc-frontend, and native-builder widths are not one managed budget.

The copied evidence tree was independently matched to the remote tree: 131 files, 696,827 bytes, with SHA-256 `ea23ba9cc7ee6e88a2b55c97b87983d79b6be0f1b1aecfa60e8d1fcde6692c1e` over the sorted per-file SHA-256 manifest.

The original successful baseline logs contained no `INFO` or compile error. They did contain these warnings, which were treated as defects rather than accepted noise:

1. `vendor/vector-ta-0.2.9-patched/build.rs`: unused `find_vs_installation`.
2. `sklears-core`: unused `read_perf_counters`.

The current working tree has removed both dead-code diagnostics: the vector-ta Windows discovery helper is now compiled only where it is called, and the unused sklears placeholder was deleted after the refreshed model log reproduced it. The later full final-source warning matrix must prove they do not return through another feature/platform combination.

An earlier model baseline failed before compilation completed. `catboost-rust 0.3.8` downloaded CatBoost 1.2.8 headers and a binary library during its build script, then panicked because the custom `CARGO_TARGET_DIR` did not contain an ancestor component literally named `target`. Repeating with a path containing `/target/` passed, but that workaround does not make the dependency safe or portable. The current build also performs unverified network downloads. Both findings remain release blockers for the dependency experiment/supply-chain task.

The supposed incremental builds are currently not incremental. `vendor/vector-ta-0.2.9-patched/build.rs` unconditionally emits:

```text
cargo:rerun-if-changed=kernels/cubin
```

That path is absent. A minimal Cargo reproduction and the real reused target both report the missing watched path as dirty and rerun the build script/compiler. The real `cargo -vv` line is: `Dirty vector-ta ...: the file vendor/vector-ta-0.2.9-patched/kernels/cubin is missing`. That supposedly incremental probe took 94.57 s, 1,054% CPU and 3,967,040 KiB peak RSS. Its four-file evidence tree (38,623 bytes) has sorted-manifest SHA-256 `17d6569437608aac15718b83551427430ffed0698dfb8badd7555672c495f228`. The later portable-build task must remove or conditionally emit the watch only for exact existing inputs and prove that a second identical `cargo -vv` invocation starts neither the build script nor NVCC.

Verbose build-script output also surfaced the effective hard-coded flags through `aws-lc-sys`: `-Ctarget-cpu=x86-64-v3` and `-Zthreads=8`. Its other `warning:` records describe successful feature/compiler probes and are informational build-script messages; the vector-ta dead-code warning remains a real warning. There were zero actual INFO-level records and zero strict error/panic/fatal records in the fingerprint probe.

The CUDA builder was also serial inside one Cargo build-script job. The original strict-f64 source manifest called `compile_kernel` synchronously, waiting for PTX and then fatbin output before visiting the next unit, so `CARGO_BUILD_JOBS=59` could not parallelize work hidden inside the build script. The replacement now queues one source job per translation unit, uses the build script's implicit Cargo jobserver slot for one worker, acquires one inherited token for every additional live source job, and keeps PTX+fatbin for one source under the same permit. It rejects every non-empty `NVCC_ARGS` value before launching NVCC; the old trailing argument append loops are gone.

The final fake-NVCC integration proof used a real Cargo jobserver with width four. It observed four, never more, simultaneous NVCC children; 12 successful source jobs produced exactly 24 invocations; an injected failure stopped after seven of sixteen possible invocations; and `NVCC_ARGS=--use_fast_math` was rejected before the fake compiler ran. The test completed in 74.464 s with no skip. Its three-file evidence tree is under `target/audit-logs/task1/vast-47806434/vector-ta-fake-nvcc-scheduler-final/`; `test.log` SHA-256 is `c2bd33944de9998b0c15446eeeb4d87395717a2d69375facf0b833d631ddd507`.

The first real parallel CUDA build exposed a defect that compile success alone hid: the build script declared `vidya_kernel.cu -> vidya_kernel.ptx` twice, so two workers raced to the same output and 340 queued jobs yielded only 339 PTX plus 339 fatbins. A pre-launch uniqueness check and static declaration test now reject duplicate source/output mappings, and the superseded duplicate declaration was deleted. The corrected clean build queued 339 source jobs and completed all 678 NVCC commands, producing exactly 339 non-empty PTX and 339 non-empty fatbins with no duplicate output path. It finished in 135.36 s, peaked at 59 simultaneous NVCC children, used 3,055 MiB peak RSS by `/usr/bin/time` (6.906 GiB peak aggregate RSS in the process monitor), and exited zero. The full evidence is under `target/audit-logs/task1/vast-47806434/vector-ta-real-cuda-build-v2/`.

An identical verbose build against that same target then reported both `vector-ta` and `neoethos-data` as `Fresh`, launched neither the vector-ta build script nor NVCC, produced zero filesystem output, and finished in 0.76 s. This closes the prior missing-watched-path fingerprint defect for the measured graph. Its log SHA-256 is `ed201efd9806ec28e3e306175d77b9d1bbd78f31672433bf2b917b8c4b00e4e5` under `target/audit-logs/task1/vast-47806434/vector-ta-real-cuda-build-v2-fresh/`.

These are build-throughput and artifact-integrity results, not device-correctness approval. The measured build still used an explicit `CUDA_ARCHS=89` or the old multi-architecture default rather than the required shared automatic `BuildHostPlanV1`; it also still inherited the repository-wide `target-cpu=x86-64-v3` and `-Zthreads=8` flags. Those old selectors/flags remain blockers until the shared resolver is connected and their superseded paths are deleted.

At runtime, `CudaF64Indicators::new` eagerly loads the distinct modules named by every `F64Kernel::ALL` variant, not only the thirteen modules needed by the NeoEthos period sweep. A filtered build that compiles only the common module therefore fails correctly rather than proving the production path. This is also a measurable startup/module-load cost and must be addressed through a typed requested-module set or a separately proven fail-closed loader contract, not through a test-only constructor.

## Current migration blockers frozen by this baseline

- Polars is a live dependency and production API in both `neoethos-data` and `neoethos-models`; it was compiled in both baseline graphs.
- `.fstore` remains a production mmap feature store and narrows shared feature values to f32.
- The current model contract consumes Polars and produces f32 probabilities. The replacement must move the shared boundary to typed f64+validity and keep any intrinsically f32 model view inside an explicit adapter.
- The CUDA indicator claim is a partial sweep, not full-feature GPU execution. The measured RTX 4090 `RequireGpu` route forbids fallback and passes the current thirteen sweep indicators, but the remaining feature graph still executes on CPU and is reported as hybrid.
- `.cargo/config.toml` hard-codes `target-cpu=x86-64-v3` and `-Z threads=8`; neither is a portable automatic policy.
- CatBoost's build, device-selection, validation, fallback, and artifact identity paths require independent repair and official-source validation before `Verified` is truthful.
- The corrected vector-ta strict-f64 CUDA build has 339 unique translation units and now uses the inherited Cargo jobserver, reaching 59 simultaneous NVCC children on the measured 59-worker host. A late telemetry sample showing `nvcc=1` was the drained tail of that queue, not the configured build width. Automatic architecture selection through the shared `BuildHostPlanV1` and requested-module-only runtime loading are still pending; current runtime construction eagerly loads the full registered module set.

## Real-data and CUDA evidence

Archive preflight, CPU feature execution, and one RTX 4090 CPU/CUDA parity run are complete. The public EURUSD snapshot is used only as quarantined external OHLCV for schema, feature, validity, and performance testing. It has no synchronized historical Bid/Ask, exact cTrader symbol/account contract, or broker PnL/cost provenance, so no financial ledger, win-rate, 2RR, Risky, Prop Firm, or profitability claim may be derived from it.

- Release asset: `snapshot-2026-08-09/EURUSD.tar`, 188,528,640 bytes, SHA-256 `042aa66fde29e697bd30d93d85c4762076c06e292097e42680ed0067b06ca061`.
- Download, SHA verification, tar path/type validation, and extraction passed in 8.91 s with 17,584 KiB peak RSS and no INFO/WARNING/ERROR records.
- The archive has 21 timeframe members. Fourteen are in the documented cTrader set (`M1,M2,M3,M4,M5,M10,M15,M30,H1,H4,H12,D1,W1,MN1`); seven are outside it (`M6,M12,M20,H2,H3,H6,H8`) and are rejected from the canonical test route.
- Only seven `.complete` files contain `download complete\n`; fourteen are zero bytes. The marker is not a content hash or provenance proof, so even non-empty members remain quarantined.
- The selected M5 `data.vortex` is 33,774,512 bytes with SHA-256 `4b12214461f8abe09c93f656192fb6c4b71d5c9ca2ea03e37403c52ad5f3bcd0`.
- The release does not establish whether timestamps denote bar open versus close, does not bind the symbol to a cTrader environment/account/symbol ID, and does not supply a typed schema/version manifest. Grid shape or directory names must not invent those facts.

The CPU feature census executed on the first 200,000 retained M5 rows and is recorded in `target/audit-logs/task1/vast-47806434/task1-real-data-cpu-v3/`. It completed in 32.266 s and emitted 1,795 f64 columns with schema hash `fnv64:71c75ada9432284e`:

- the base plan admitted all 342 listed ids and all 782 planned base columns under the measured RAM budget;
- the historical 18-id period sweep produced all 130 planned columns;
- the extended sweep admitted 102 ids / 910 columns and deferred 48 ids under the current fixed 4,096-column ceiling;
- the final ledger found 115 bit-identical columns, 38 constant columns, four all-non-finite columns, and 1,513 columns containing warmup/gap invalidity;
- eight ids were unknown to the dispatcher, fifteen outputs across two ids contradicted the dispatcher, four ids lacked the claimed CPU-batch capability, and `geometric_bias_oscillator_7` failed because the requested length 7 violates its declared 10..=500 range;
- 324 of 1,054,644 source rows had non-positive prices and the current loader silently dropped them. That repair behavior is evidence of a canonical-import defect, not approval of the resulting dataset;
- dozens of registry ids declared window-shaped parameters that the extended sweep cannot currently drive. Those variants are unavailable to the search until their exact multi-parameter semantics are implemented; renaming an unchanged default output is not accepted as a period sweep;
- measured process utilization averaged only about 223% CPU against a 59-worker allowance, so the feature lane is substantially underutilizing the AMD host even though it fits memory.

The CPU run therefore passes as a truthful diagnostic baseline but fails readiness: invalid source rows are repaired instead of rejected, the searchable parameter space has known holes, duplicate/degenerate features waste search capacity, and the scalar formulas have not yet completed their independent oracle review.

### RTX 4090 strict-f64 execution result

The current working tree was rebuilt from source for `sm_89` with CUDA 12.8.93, `CUDA_FAST_MATH=0`, no ambient `NVCC_ARGS`, and effective NVCC precision flags `-prec-div=true -prec-sqrt=true -fmad=false -ftz=false`. The clean GPU build passed in 3:14.83, averaged 2,186% CPU, and peaked at 7,388,080 KiB RSS. The builder reached 59 simultaneous NVCC children, produced the expected sm89 SASS plus compute89 PTX artifacts, and emitted no actual compile error. The complete verbose log is retained because it also exposes the still-active `-Zthreads=8`, `target-cpu=x86-64-v3`, and upstream dependency warnings; compile success does not waive those blockers.

The focused real-device unit test forced the fatbin path and passed all thirteen claimed sweep indicators (`sma, ema, rsi, roc, mom, atr, adx, willr, cci, mfi, tsi, obv, vwap`) bit-identically to the scalar CPU reference on its 100-bar fixture. It loaded real sm89 CUDA modules and did not use a CPU fallback. The CPU reference suite passed 13/13 tests, including the new 250,000-row SMA exact-window oracle.

The long-run SMA review found that the old scalar/CUDA rolling recurrence and CPU-batch prefix-sum implementations did not share one numerically stable definition. Against independently summed, exactly representable adversarial windows, the old rolling route drifted by hundreds of ULP and the prefix route by thousands. The replacement uses the same compensated rolling update and operation order in scalar, streaming, batch-row, and CUDA paths; obsolete SMA AVX/prefix implementations were deleted. This is parity plus an independent arithmetic-window oracle, not proof for every other indicator formula. The formula definition used for the review is the arithmetic mean of the last N values documented by [VectorTA SMA](https://vectoralpha.dev/projects/ta/indicators/sma/).

The vendored crate is not a root-workspace member, so its standalone copy was also compiled with its own generated lockfile. An initial `sma` substring filter unintentionally selected 549 tests whose names included `mismatch`; 548 passed, while one unrelated Yang-Zhang test failed because the published vendor tree does not contain its hard-coded `src/data/...csv` fixture. That is preserved as a separate fixture/packaging diagnostic rather than attributed to SMA. The corrected fully-qualified `--exact` SMA test passed 1/1 with zero warning/error/panic records in 0.18 seconds and 55,468 KiB peak RSS. All five files under `task1-vector-sma-exact/` match the remote SHA-256 manifest.

The full `RequireGpu` test then scanned 200,000 retained M5 rows and compared 1,784 ordered columns / 344,925,030 finite cells. It passed in 68.58 seconds with `worst_absolute_delta=0` and `worst_relative_delta=0`; CPU construction took 32.771766421 seconds and the CUDA-policy construction 31.998929188 seconds. There were 4,943 unequal f64 bit patterns despite zero numeric delta; for finite IEEE-754 values this is exclusively `+0.0` versus `-0.0`, so strict bit identity for the whole hybrid frame is not yet claimed. The report explicitly records `execution_class=hybrid_cuda_sweep_plus_cpu_unclaimed_nodes`, `full_frame_executed_entirely_on_gpu=false`, and `performance_comparison_valid=false`.

The full run emitted 22 INFO, 127 WARN, and zero ERROR/panic/fatal records. All 127 warnings were classified: one records 324 non-positive source-price rows; 124 are the same 62 unrecognised coupled-window declarations emitted once during the CPU build and once during the hybrid build; and two report the 115 bit-identical kept columns in those two passes. The final census reports 328 producing ids, 1,653 produced columns, 77 dropped columns (`8 unknown_indicator`, `15 unknown_output`, `4 unsupported_capability`, `1 compute_failed`, `49 over_budget`), 115 duplicate columns, and 41 degenerate columns. These diagnostics are search-space/correctness blockers, not harmless noise.

The complete build, CPU reference, GPU unit, full Vortex parity, timing, telemetry, JSON report, and SHA-256 manifest are under `task1-real-data-gpu-sm89-sma-green/`. All 18 downloaded files were independently rehashed with zero mismatch. CPU/GPU data readiness remains **not approved** until the warning-producing search space is repaired or fail-closed, canonical import rejects rather than repairs invalid source data, every production transformation receives an independent semantic/validity review, and each advertised CUDA architecture/load path is promoted separately.
