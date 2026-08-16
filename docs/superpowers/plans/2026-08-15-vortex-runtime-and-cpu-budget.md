# Vortex Runtime and Unified CPU Budget Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Use `superpowers:test-driven-development` for every behavior change, `superpowers:systematic-debugging` for every unexpected failure, and `superpowers:verification-before-completion` before any completion claim.

**Goal:** Make Vortex the only runtime/persisted market-data and shared-feature engine while preserving CSV, TSV, JSON, JSONL, Parquet, Arrow IPC/Feather, and Vortex as explicit user import sources; keep canonical prices, indicators, shared features, strategy weights/thresholds, live-learning rows, and GPU evaluation contracts `f64`; prove and expand the custom vector-ta real-f64 CUDA feature lane without hidden f32/CPU work; remove Polars, shared f32 feature narrowing, JSONL live-feature storage, `.fstore`, and unsupported integrated/WGPU GPU execution; expose one truthful CPU-only/one-or-many-NVIDIA device policy; and enforce one portable CPU-work ceiling of `available_parallelism() - 2` across the root, app, desktop, MCP, mesh, import, search, model, native-library, GPU-feeder, and managed-child paths.

**Architecture:** A zero-dependency `neoethos-execution-budget` leaf crate owns pure capacity resolution, immutable process installation, and hierarchical RAII permits. Runtime market data and out-of-core/shared live-learning feature data use Vortex only. One shared, Polars-free importer converts supported source formats to strict `i64` timestamp / `f64` OHLCV Vortex in bounded batches, verifies an immutable generation, and publishes it with one atomic current-manifest swap. `FeatureFrame`, Vortex feature columns, search weights/thresholds, signals, live-experience rows, GPU contracts/kernels, and backtest inputs remain f64; an intrinsically f32 model may receive a named, range-checked adapter-local view, never a shared narrowed store. Model APIs consume `FeatureFrame` plus typed labels rather than DataFrame/Series. One typed device policy chooses CPU-only, all eligible NVIDIA discrete devices, or an explicit non-empty set canonical-sorted by stable UUID+PCI identity; native CUDA owns the supported GPU lane, while integrated/WGPU/Vulkan/ROCm execution surfaces are removed after baseline evidence. The custom vector-ta lane has a generated per-indicator/output capability manifest, one resident f64 OHLCV upload per assigned card/job, batched proven kernels, direct resident handoff where compatible, and explicit stage coverage; a hybrid CPU segment is never reported as full-GPU. CPU-heavy async work is admitted by a dedicated coordinator before `spawn_blocking`; synchronous CLIs acquire at the top level; managed children hold parent reservations for their full lifetime.

**Tech Stack:** Rust 2024 using the repository-pinned nightly for every primary resolve/build/test, Vortex 0.67 before the separately gated upgrade experiment, Arrow/Parquet aligned with the selected Vortex graph, ndarray, Rayon, Tokio, Tauri, cTrader Open API fixtures, PowerShell validation on Windows, and CUDA validation on a real NVIDIA 3090/4090 VPS. Manifest `rust-version` values are tested separately as packaging/MSRV claims and never switch this plan's main toolchain to stable.

**Truth boundary:** This plan proves data-engine and CPU-scheduling correctness plus worker-count determinism. It must not claim that the existing trading cost formulas are broker-real, that CUDA is validated without a card, or that profitability is achieved. Full historical Bid/Ask execution replay and broker formula reconciliation remain a separate reviewed implementation plan and release gate.

**Log rule:** Never judge a command by exit code or its final lines. Preserve complete command output under `target/audit-logs/`, read it from the first INFO/compile line through final status, and classify every INFO/WARN/ERROR/dead-code/unused diagnostic. No `tail`-only review and no new warning are acceptable.

**Replacement/deletion rule:** A replacement is not complete when it merely compiles beside the old path. First connect and prove the new path through every production caller and boundary; then, in the same compile-green migration, delete the superseded implementation, feature/config/env aliases, fallback branches, dependencies, tests, scripts, and active documentation. Add narrow source/call-path guards that make their return RED. Retain compatibility code only as an explicit versioned offline/protocol migration with a tested retirement condition; never retain unreachable code or a silent legacy runtime fallback.

**Required execution order:** Task numbering groups related work for review, but the implementation order is `1 -> 2 -> 10 -> 11 -> 13 (existing-entrypoint startup portion) -> 16C -> (3 + 4 as one compile-green dataset-contract/publication/import revision) -> 4A -> 5 -> (5B + 6 + 6A + 7 + 8 + 9, including scalar-validity preparation and the new migration-entrypoint startup work, as one compile-green commit) -> 12 (remaining non-import workloads) -> 13 (remaining blocking-call inventory) -> 14 -> 15 -> 16 -> 16A -> 16B -> 17 -> 18A -> 18 -> 19`. Task 16C installs the earliest typed fail-closed broker capability before any later import/precision/model/native/search/app all-target test or code path can execute current heuristic cost/PnL/risk arithmetic. Every pre-broker test after that point either inspects typed f64/layout/serialization/signal behavior before finance, supplies an exact synchronized broker-capable fixture, or expects typed unsupported before arithmetic; no temporary test-only heuristic route is allowed. Tasks 3 and 4 are one public commit because the new dataset-contract leaf adds official timeframe variants while `neoethos-core` and every exhaustive consumer must migrate to the exact re-exported type together. Tasks 4A and 5 are one compile-green commit: production import adapters are not exposed before importer admission and every dependency worker is classified. Tasks 5B–9 are also one compile-green commit because scalar validity, the public f64+validity `FeatureFrame`/`FeatureData` flip, and the crate-wide `ExpertModel` migration cross the data/model boundary: no public revision is recorded while a producer still emits sentinel zero/NaN or any direct model consumer still requires `Vec<f32>`/`Array2<f32>`. This installs the budget/admission and broker-truth authorities before production import and f64 Vortex access require leases, while Vortex/Polars migration still remains mandatory before the CPU-budget milestone can be accepted. Do not create a temporary unbudgeted production wrapper, shared f32 compatibility accessor, committed sentinel bridge, duplicate dataset/timeframe type, heuristic finance bypass, or intentionally broken intermediate revision merely to follow topical document order.

---

## Milestone 1 — Freeze contracts and build the budget authority

### Task 1: Capture pre-migration data/model behavior and performance baselines

**Files:**
- Create: `crates/neoethos-data/tests/fixtures/feature_store_contract_v1.json`
- Create: `crates/neoethos-data/tests/feature_store_contract.rs`
- Create: `crates/neoethos-models/tests/fixtures/model_frame_contract_v1.json`
- Create: `crates/neoethos-models/tests/model_frame_contract.rs`
- Modify: `crates/neoethos-data/tests/vocabulary_restoration_measured.rs`
- Create: `docs/audits/2026-08-15-vortex-polars-baseline.md`
- Modify: `.gitignore` only if `target/audit-logs/` is not already ignored

**Step 1: Write the fixed feature-store contract test**

Use a deterministic frame with real `i64` timestamps, named columns, finite values, row windows, nested windows, selected-column reads, and label alignment. Include one f32-exact set for old/new storage parity and one high-precision f64 indicator set. Record both the pre-narrowing indicator bits and the current `FeatureFrame`/`.fstore` result so the latter fixture exposes the existing f64-to-f32 loss; it is a RED target, not the desired oracle.

**Step 2: Run the feature contract and confirm GREEN on the old implementation**

Run: `cargo test -p neoethos-data --test feature_store_contract -- --nocapture`

Expected: PASS while documenting identical old in-memory/disk values for the f32-exact set and the known divergence from pre-narrowing f64 for the high-precision set.

**Step 3: Write the fixed model-frame contract test**

Exercise current DataFrame-to-model conversion with named features, strict null/non-finite rejection, labels, train/validation splits, deterministic predictions from the small deterministic model paths, and stable training metadata. Store the expected contract independently of Polars types so the same fixture survives their removal.

**Step 4: Run the model contract and confirm GREEN on the old implementation**

Run: `cargo test -p neoethos-models --test model_frame_contract -- --nocapture`

Expected: PASS and no warning.

**Step 5: Record measured baselines**

Freeze a reproducible benchmark protocol before measuring: same commit/lockfile, pinned nightly, fixture hashes, release profile, resolved worker limit, device policy, storage volume, cache state, and no competing compile/test job. Runtime/import/scan/query/training benchmarks use three warmups followed by ten measured runs; report median, p95, throughput, peak RSS, and peak scratch/disk bytes. Clean builds use three independent fresh target directories; incremental builds use three warmups plus ten measured runs. Preserve every raw sample and profiler/log path, not only the aggregate.

Record commands, toolchain, effective logical threads, wall time, peak RSS, output size, and complete-log locations for:

- clean `neoethos-data` and `neoethos-models` build;
- canonical Vortex OHLCV scan;
- `.fstore` selected-column and row-window access at its current f32 width;
- repeated GA selected-column access;
- one fixed end-to-end training fixture.
- a per-pair/timeframe indicator ledger and stable feature-schema hash covering every attempted/produced/expected-nonproducing/panic/warmup/truncated/duplicate/degenerate/budget-deferred column, plus CPU versus currently claimed f64-CUDA coverage. Preserve every INFO/WARN line; the current partial GPU ratio and any all-NaN ballast are baseline facts, not accepted end-state behavior.

Do not describe Polars or `.fstore` as faster/slower without measurements.

Predeclare migration gates. On the fixed selected-column/window and GA fixtures, Vortex median latency may regress at most 10% and p95 at most 15% versus `.fstore`; fixed end-to-end training median may regress at most 5% and p95 at most 10%. Peak feature-store RSS must stay within `configured_decoded_cache + 2 * max_decoded_chunk + writer_batch + 64 MiB` above the input baseline, and scratch disk within checked `final_vortex_bound + one_candidate_bound + 64 MiB`; import uses its separately checked `ImportLimits`. A breach blocks completion until profiling and repair, or an explicit operator-reviewed waiver records raw samples, confidence/noise analysis, profile/root cause, correctness necessity, and exact accepted regression. “Material” is never a subjective pass.

**Step 6: Commit**

```text
test: freeze data and model migration contracts
```

### Task 2: Create the zero-dependency execution-budget leaf crate

**Files:**
- Create: `crates/neoethos-execution-budget/Cargo.toml`
- Create: `crates/neoethos-execution-budget/src/lib.rs`
- Create: `crates/neoethos-execution-budget/tests/process_install.rs`
- Modify: `Cargo.toml`
- Modify: `Cargo.lock`
- Modify: `crates/neoethos-core/Cargo.toml`
- Modify: `crates/neoethos-core/src/lib.rs`
- Modify: `mcp/Cargo.toml`
- Modify: `mcp/Cargo.lock`
- Modify: `mesh/Cargo.toml`
- Modify: `mesh/Cargo.lock`

**Step 1: Write failing pure-resolution tests**

Table-test effective logical counts `1, 2, 3, 4, 12, 23, 96`; persistent, legacy, and parent caps; request `9999`; and source provenance. Expected automatic outputs are `1, 1, 1, 2, 10, 21, 94`.

**Step 2: Run RED**

Run: `cargo test -p neoethos-execution-budget resolution -- --nocapture`

Expected: FAIL because the crate/API does not exist.

**Step 3: Implement pure capacity resolution**

Add non-zero typed inputs and a `ResolvedExecutionBudget` containing host inventory when supplied, effective logical threads, reserved threads, automatic limit, each optional cap with provenance, final effective limit, and coordination scope. An invalid zero is an error; an oversized request only narrows/preserves the automatic ceiling.

**Step 4: Write failing install tests**

Use subprocess-isolated cases to prove equal reinstallation is idempotent, conflicting installation fails, detection failure falls back to one worker with a structured diagnostic, and only the final effective limit seeds the broker.

**Step 5: Implement immutable process installation**

Use `std::thread::available_parallelism`, `OnceLock`, and no runtime/library dependencies. Do not initialize Rayon, Tokio, tracing, GPU, or model code from the leaf crate.

**Step 6: Write failing permit-broker tests**

Cover immediate acquisition, blocking acquisition, FIFO within priority, child priority, split/transfer, drop return, cancellation, nested-acquisition rejection, panic unwind, and `live_reserved_sum <= installed_limit` for every interleaving.

**Step 7: Implement `CpuPermitBroker` and non-cloneable `CpuLease`**

Use `Mutex`/`Condvar` only in the leaf synchronous broker. A lease can split its existing width but code holding one cannot acquire fresh permits. Return permits by RAII on success, error, panic, or cancellation.

**Step 8: Verify all three workspaces can resolve the leaf crate**

Run: `cargo test -p neoethos-execution-budget --all-targets -- --nocapture`

Run: `cargo check --manifest-path mcp/Cargo.toml --all-targets`

Run: `cargo check --manifest-path mesh/Cargo.toml --all-targets`

Expected: PASS; the leaf crate has no dependencies in `cargo tree -p neoethos-execution-budget`.

**Step 9: Commit**

```text
feat: add unified CPU execution budget authority
```

---

## Milestone 2 — Make import-to-Vortex strict, bounded, and atomic

### Task 3: Add bounded Vortex scan/write primitives and verified publication

**Files:**
- Create: `crates/neoethos-dataset-contracts/Cargo.toml`
- Create: `crates/neoethos-dataset-contracts/src/lib.rs`
- Create: `crates/neoethos-dataset-contracts/src/identity.rs`
- Create: `crates/neoethos-dataset-contracts/src/temporal.rs`
- Create: `crates/neoethos-dataset-contracts/tests/golden_dataset_identity.rs`
- Create: `crates/neoethos-dataset-contracts/tests/golden_temporal_contract.rs`
- Modify in the same Task 3+4 revision: `crates/neoethos-core/src/contracts/temporal.rs`
- Create: `crates/neoethos-data/src/core/dataset_manifest.rs`
- Create: `crates/neoethos-data/src/core/dataset_generation_lease.rs`
- Create: `crates/neoethos-data/src/core/dataset_candidate_lease.rs`
- Create: `crates/neoethos-data/src/bin/migrate_legacy_dataset_layout.rs`
- Create: `crates/neoethos-data/tests/vortex_atomic_publish.rs`
- Create: `crates/neoethos-data/tests/dataset_producer_provenance.rs`
- Create: `crates/neoethos-data/tests/dataset_identity.rs`
- Create: `crates/neoethos-data/tests/legacy_dataset_layout_migration.rs`
- Create: `crates/neoethos-data/tests/vortex_generation_lease.rs`
- Create: `crates/neoethos-data/tests/vortex_candidate_lease.rs`
- Modify: `crates/neoethos-data/src/core/vortex_io.rs`
- Modify: `crates/neoethos-data/src/core/mod.rs`
- Modify: `crates/neoethos-data/src/lib.rs`
- Modify: `crates/neoethos-data/Cargo.toml`
- Modify: `crates/neoethos-core/Cargo.toml`
- Modify: root `Cargo.toml`
- Modify: `Cargo.lock`

**Step 1: Write failing shared-contract and streaming/publication tests**

Create RED golden vectors in `neoethos-dataset-contracts` for all 14 timeframes, bar-open convention, external and broker-bound canonical identities, path bytes, and malformed/collision cases. Core/data/mesh/GPU-facing test fixtures must import/re-export the exact leaf types and produce the same bytes; a duplicate local enum/identity type fails the source/dependency guard. Task 3 creates the leaf and generic publication types, but does not claim the duplicate-type guard or whole workspace is GREEN until Task 4 replaces `neoethos-core::contracts::temporal` with an exact re-export and migrates every exhaustive consumer in the same public revision.

Prove that multiple bounded `ArrayRef` chunks write one file without an all-file `ByteBufferMut`, projection and row-range scans return only requested data, reopen verification checks schema/row count/value invariants, interrupted writes leave no accepted dataset, and temporary files are removed on every failure path. Add generic producer-envelope RED fixtures for schema-id grammar/size, canonical payload/hash binding, manifest association, unknown-schema pass-through at ordinary runtime, and every tamper case; a source guard proves `dataset_manifest` contains no non-Vortex format vocabulary or producer-specific branch. Before implementation, add RED identity/path fixtures for separators, dot segments, absolute/rooted paths, Windows ADS/reserved/device names, Unicode and case distinctions, broker suffixes, long/control/NUL inputs, noncanonical encodings, and two distinct names that the old punctuation-stripping `canonical_symbol` collides. The typed identity must round-trip exactly, never collide in the fixture, contain cTrader live/demo + server/account scope + `symbolId` where broker-bound, and keep external/no-broker-id data in a separately marked non-broker-real namespace. Before implementing publication, add dataset-root concurrent-publisher RED fixtures for A/B interleavings, both completion/linearization orders, cancellation/crash, same and different roots, stale-owner recovery, and expected-generation conflict. Two same-root requests capturing one expected base must yield exactly one durable success and one typed conflict; the success result records new/previous generation plus durable commit id at its linearization point, without promising it remains current after a later request commits. Add a RED reader-liveness case where a scan pins generation N, publishers install N+1 and N+2, GC runs, and repeated lazy projections/reopens from N still succeed until the reader releases its pin on both Windows and Linux.

Add RED legacy-layout migration cases for human `symbol=.../timeframe=...`: absent verified broker mapping can produce only external/non-broker-real identity; verified account symbol metadata may produce the exact broker-bound identity; failure publishes nothing and never deletes source; and runtime rejects the old layout after cutover. The utility may not infer environment/account/server/`symbolId` from names.

Add a RED candidate-liveness fixture in which a slow writer's candidate is forced older than the sweep threshold while another process runs GC. The candidate must survive until publish or explicit failure cleanup; after the writer subprocess crashes, the released contained orphan becomes collectible. PID/age metadata and PID reuse are never liveness authority.

**Step 2: Run RED**

Run: `cargo test -p neoethos-dataset-contracts --all-targets -- --nocapture`

Run: `cargo test -p neoethos-data --test dataset_identity --test dataset_producer_provenance --test legacy_dataset_layout_migration --test vortex_atomic_publish --test vortex_generation_lease --test vortex_candidate_lease -- --nocapture`

Expected: FAIL because `write_vortex_array` materializes the whole encoded file and the manifest contract does not exist.

**Step 3: Implement bounded Vortex I/O**

Add a streaming writer backed by a real temporary file/async stream, plus scan helpers using Vortex projection and row-range pushdown. Keep `write_vortex_array` only as a bounded single-array wrapper; remove `ByteBufferMut` whole-file encoding from production.

**Step 4: Upgrade completion metadata**

Turn `data.vortex.complete` from an empty marker into the single versioned current-generation manifest/pointer. It names an immutable content-addressed or generation-scoped Vortex file and contains only canonical schema version, generation id, row count, timestamp range, Vortex byte hash, typed `CanonicalDatasetIdentity`, canonical bar convention, publication timestamp/version, and one bounded opaque `ProducerProvenanceEnvelopeV1 { schema_id, canonical_payload, payload_sha256 }`. The generic publication/runtime layer validates the envelope's size, namespaced schema-id grammar, canonical byte/hash agreement, and manifest binding but never decodes it, branches on it, or contains CSV/TSV/JSON/Parquet/Arrow format vocabulary. Task 4's import boundary owns `ImportSourceFormat` plus `ImportProvenanceV1` and is the only code that constructs/decodes an import envelope; broker and legacy-migration writers use separate typed producer schemas. Thus source format, exact staged-source SHA-256, captured source identity/size metadata, import timestamp, and importer version live inside the import-owned payload rather than canonical dataset fields. Tampering, an unknown envelope schema at an import-reporting decode boundary, or hash mismatch fails closed, while ordinary Vortex discovery can verify and load the generation without understanding producer-specific metadata. This JSON is provenance/control metadata, not a market-data/query engine. Encode identity path components reversibly as versioned base32hex over a manually specified length-prefixed binary format with fixed tags/order/integer endianness; never strip punctuation, normalize/lowercase Unicode, or use a hash-only/OS-native filename. Broker-bound identity includes environment, server/account scope, `symbolId`, exact name metadata, and exact timeframe; external identity is explicitly non-broker-real. Recompute the path component from manifest bytes and reject noncanonical/colliding decodes. The generation id is an opaque single path component under that canonical dataset root: reject empty, `.`, `..`, absolute/rooted names, separators, alternate streams, and any symlink/junction/reparse resolution outside the configured root before locking, hashing, reading, renaming, or deleting. Garbage collection accepts only canonicalized unreferenced children that pass the same full-root containment check and never follows a link outside the root.

Implement `migrate_legacy_dataset_layout` as the only reader of old human `symbol=.../timeframe=...` roots. It performs synchronous CPU-budget preflight, validates, decodes, and rewrites a specifically versioned/proven old schema through the same immutable generation/CAS protocol, publishes all-or-nothing, and leaves sources untouched. It never byte-copies a generation whose correctness depends on read-time timestamp magnitude inference, sorting, deduplication, missing-to-zero, or another legacy repair. A documented old timestamp unit may be converted with checked exact arithmetic only when the legacy schema/manifest declares it and every value satisfies that mapping; ambiguous units, duplicates, descending rows, off-grid rows, or an unknown/empty-marker schema reject. It accepts a broker-bound mapping only when operator-supplied environment/account/server + `symbolId` + exact name metadata is verified against a captured broker symbol list; otherwise output is explicitly external/non-broker-real. It never derives missing broker identity from folder punctuation. After its versioned cutover, ordinary discovery/load rejects old roots with a migration diagnostic.

Publish in this order: write candidate file -> flush/fsync -> rename it to an immutable generation/content-hash path -> fsync the containing directory -> reopen/hash/validate immutable bytes -> write and fsync a candidate manifest that references exactly those bytes -> atomically replace only `data.vortex.complete` -> fsync the directory. Never replace the currently referenced data file in place. Retain the old referenced generation until after the pointer swap and defer garbage collection, so a crash at every transition leaves either the old verified generation or the complete new verified generation discoverable. A first import remains undiscoverable until the one pointer swap. Existing empty completion markers are read only through the explicit legacy migration path and never emitted again. Pointer resolution acquires a cross-process `DatasetGenerationLease` before releasing the dataset-root coordination lock, and every scan/backtest/model/FeatureFrame holds that reader lease across all lazy projection/reopen work. Each generation has an OS-backed shared reader lock; GC takes the root lock and must obtain a nonblocking exclusive generation lock before deletion, so it skips every live reader. OS lock release on process death is the liveness authority; age/PID metadata alone never overrides a live pin.

The immutable manifest and `CanonicalDatasetIdentity` include the typed `BarTimestampConvention`; canonical generations are bar-open only. Grid alignment never infers that field. A legacy or imported source with unknown/close/end convention fails unless an explicit independently proven converter records source convention, canonical bar-open convention, and exact conversion rule.

From candidate creation through successful publication or complete failure cleanup, hold an OS-backed cross-process `DatasetCandidateLease`. GC must obtain its nonblocking exclusive candidate lock before deleting an unreferenced contained candidate. A live slow writer is preserved regardless of age/PID metadata; process death releases the lock for safe orphan collection.

**Step 5: Test replacement and crash states**

Add fault injection after every write, flush, rename, validation, pointer-swap, directory-sync, and cleanup transition. Cover old generation + new staging, orphan complete generation + old pointer, truncated candidate, stale partial marker, manifest hash/path mismatch, absolute/`..`/separator generation names, symlink escapes, Windows junction/reparse escapes, interrupted post-swap garbage collection, and Windows atomic replacement behavior. After every injected interruption, reopening must load either the complete old verified generation or the complete new verified generation—never neither and never mismatched bytes. Cleanup must preserve the current generation, at least the retained rollback generation, every generation with a live `DatasetGenerationLease`, and every candidate with a live `DatasetCandidateLease`; only validated, contained, unreferenced candidates whose nonblocking exclusive candidate lock succeeds are eligible for candidate garbage collection. Tests prove no tampered manifest can cause a hash/read/delete outside the dataset root, a live N reader survives N+1/N+2 publication plus GC and repeated projections, a forced-aged slow writer survives concurrent GC, and crashed reader/writer OS-released pins become safely collectible without an age-only liveness guess.

Implement and test one expected-generation CAS. The request captures its base generation before candidate work. CPU admission completes before acquiring the canonical-identity-scoped cross-process publication lock; under that lock reread and compare the pointer, replace/fsync only on equality, and hold through final directory fsync. Record owner/process identity and fail-closed stale-owner/crash recovery. Success returns `{generation, previous_generation, durable_commit_id}` identifying the atomic swap/fsync linearization point; it does not promise wall-clock currentness at response delivery. A mismatch returns a typed conflict and retry is a new request with a new expected base. Exercise A/B interleavings, both linearization orders, cancellation and crash, same-root and different-root targets; history must be linearizable, at most one request per base succeeds, and different roots remain concurrent under their CPU leases.

**Step 6: Run focused and existing Vortex tests**

Run: `cargo test -p neoethos-data vortex -- --nocapture`

Expected: PASS and no new warning.

**Step 7: Continue directly into Task 4 without committing**

Task 3 may make the leaf and publication-focused tests GREEN, but the current core-owned temporal type and its exhaustive consumers are migrated in Task 4. Preserve RED/GREEN logs and create no public revision until the combined Task 3+4 matrix is compile-green.

### Task 4: Replace conversion code with one Polars-free import service

**Files:**
- Modify: `crates/neoethos-core/src/contracts/temporal.rs`
- Modify: `crates/neoethos-dataset-contracts/src/temporal.rs`
- Modify: `crates/neoethos-dataset-contracts/src/identity.rs`
- Modify: `crates/neoethos-core/src/config.rs`
- Create: `crates/neoethos-core/tests/ctrader_timeframe_contract.rs`
- Create: `crates/neoethos-core/tests/bar_timestamp_convention.rs`
- Modify: `crates/neoethos-app/src/app_services/ctrader_messages.rs`
- Modify: `crates/neoethos-app/src/app_services/bootstrap_writer.rs`
- Modify: `crates/neoethos-app/src/app_services/ctrader_bootstrap.rs`
- Modify: `crates/neoethos-app/src/app_services/broker_api.rs`
- Modify: `crates/neoethos-app/src/app_services/ctrader_data.rs`
- Modify: `crates/neoethos-app/src/app_services/live_trading.rs`
- Modify: `crates/neoethos-app/src/app_services/validation.rs`
- Modify: `crates/neoethos-app/src/server/data_control.rs`
- Create: `crates/neoethos-app/tests/ctrader_timeframe_contract.rs`
- Modify: `desktop/src/components/filters.tsx`
- Modify: `desktop/src/components/Select.tsx`
- Modify: `desktop/src/screens/Markets.tsx`
- Modify: `config.yaml`
- Modify: `desktop/src-tauri/resources/config.yaml`
- Modify: `crates/neoethos-core/src/resolved_config.rs`
- Modify: `crates/neoethos-cli/src/main.rs`
- Modify: `crates/neoethos-search/tests/higher_timeframe_lane_measured.rs`
- Modify: `crates/neoethos-search/src/discovery.rs`
- Create: `crates/neoethos-data/src/core/import_service.rs`
- Create: `crates/neoethos-data/src/core/import_provenance.rs`
- Create: `crates/neoethos-data/src/core/import_limits.rs`
- Create: `crates/neoethos-data/src/core/source_snapshot.rs`
- Create: `crates/neoethos-data/tests/import_contract.rs`
- Create: `crates/neoethos-data/tests/import_provenance.rs`
- Create: `crates/neoethos-data/tests/import_source_seal.rs`
- Create: `crates/neoethos-data/tests/import_bounded_memory.rs`
- Create: `crates/neoethos-data/tests/import_adversarial_limits.rs`
- Create: `crates/neoethos-data/tests/canonical_vortex_timestamp_contract.rs`
- Create: `crates/neoethos-data/tests/canonical_timeframe_resample.rs`
- Create: `crates/neoethos-data/tests/timeframe_single_source_guard.rs`
- Create: `crates/neoethos-app/tests/broker_vortex_writer_contract.rs`
- Create: `crates/neoethos-data/tests/fixtures/import/README.md`
- Modify: `crates/neoethos-data/src/core/mod.rs`
- Modify: `crates/neoethos-data/src/core/timestamps.rs`
- Modify: `crates/neoethos-data/src/core/resample.rs`
- Modify: `crates/neoethos-data/src/core/feature_registry.rs`
- Modify: `crates/neoethos-data/src/lib.rs`
- Modify: `crates/neoethos-data/src/core/to_vortex.rs`
- Modify: `crates/neoethos-data/src/core/universal_importer.rs`
- Delete after callers move: `crates/neoethos-data/src/core/parquet_migration.rs`
- Delete after callers move: `crates/neoethos-data/src/core/to_vortex.rs`
- Delete after callers move: `crates/neoethos-data/src/core/universal_importer.rs`
- Modify: `crates/neoethos-data/Cargo.toml`
- Modify: `Cargo.toml`
- Modify: `Cargo.lock`

**Step 1: Add aligned import dependencies**

Use Arrow/Parquet `58.3.0`, matching the Vortex 0.67 dependency graph: `arrow-array`, `arrow-schema`, `arrow-csv`, `arrow-json`, `arrow-ipc`, and `parquet` with only required features. Add `sha2` for streaming provenance plus the narrow target-specific Windows/Linux system bindings needed by `SourceSealV1`; no portable advisory-lock abstraction may be presented as writer exclusion. Do not add Polars compatibility or retain unused readers.

**Step 2: Write the seven-format contract fixture**

Generate equivalent CSV, TSV, JSON array, JSONL/NDJSON, Parquet, Arrow IPC file/stream (Feather-compatible extension), and Vortex inputs containing:

- exact `i64` millisecond timestamps explicitly declared/evidenced as bar-open;
- high-precision `f64` prices including a value the old f32 round-trip gate rejected;
- optional volume represented consistently with exact raw integer/decimal provenance where applicable and a versioned checked f64 derivation for volume-dependent features;
- aliases that map uniquely;
- deterministic row order.

Binary f64 inputs must compare `to_bits()` exactly after reopening Vortex. Decimal text inputs must equal their direct Rust f64 parse with no intermediate f32 conversion. Add Parquet, Arrow IPC, and Vortex negative fixtures whose OHLC, optional volume, or shared-feature columns are Float32: they must fail as precision-unrecoverable rather than widen and appear canonical. Accepted binary prices/features/volume are Float64, or an explicitly documented decimal/scaled-integer physical type whose scale and exact checked f64 mapping are stored in provenance; implicit casts are forbidden. Preserve exact broker integer volume in its raw canonical physical column and derive f64 only under a versioned exact mapping. Include zero, `16_777_217` (beyond f32 exact integer precision), integer/decimal boundaries, and a value not exactly representable by the declared f64 mapping; the last remains raw/provenanced but makes Footprint/other f64 volume-dependent plans typed unsupported instead of rounded.

**Step 3: Write failing strictness tests**

Each malformed, missing-timestamp, null, non-finite, incoherent OHLC, duplicate/ambiguous alias, truncated, corrupt, interrupted, or Float32 binary price/volume/feature source must fail the whole file with source row/field/type context. OHLC prices are finite and strictly positive. Optional broker volume, when present, is finite and `>= 0`; a zero-volume bar is valid unless the exact documented source schema forbids it. The raw broker integer/decimal type and unit come only from official protocol/schema evidence, are preserved exactly, and have a provenance-bound checked f64 mapping; any nonexact f64 mapping blocks the volume-dependent plan without corrupting the raw generation. Canonical timestamps must already be valid-range, strictly increasing `i64` milliseconds. Reject duplicate, descending, ambiguous-unit, and out-of-range timestamps and declared/inferred symbol or timeframe disagreement with source row/context. Require exactly one declared `CanonicalTimeframe` from `M1,M2,M3,M4,M5,M10,M15,M30,H1,H4,H12,D1,W1,MN1`; path/header inference only corroborates. Delete ±5% median inference. Fixed minute/hour opens lie on the epoch-minute grid and adjacent gaps are positive integer multiples; missing multiples are allowed for cTrader no-tick/weekend gaps. D1/W1/MN1 use separately evidenced calendar grids or fail closed as unsupported. Add M1-as-M5, off-grid, valid missing-period/weekend, and calendar-boundary fixtures. No row may be dropped, sorted, deduplicated, repaired, or synthesized from row index, and no missing value becomes zero. Write `import_provenance` RED canonical-byte/round-trip fixtures for every source variant and exact staged hash, plus wrong schema id, unknown version, oversized payload, mutated format/source hash/convention, and envelope hash mismatch. Ordinary Vortex runtime must verify the outer binding without decoding source format; only the exact import reporting API may decode it.

Make `neoethos-dataset-contracts` the sole enum/identity/convention source and replace `neoethos-core::contracts::temporal` with an exact public re-export plus any non-owning compatibility module functions. Add the currently omitted official M2/M4/M10 variants exactly once in the leaf, preserve H2 as rejected, and update every duration/order map, broker request/response encoder, `/broker/timeframes`, import/discovery validator, resampler, feature registry, config/default, desktop fallback/filter/select/help, and fixture to consume that exact type rather than a private array. `ctrader_data` accepts all 14 exact protocol codes, and no broker/app path rejects M2/M4/M10. `resample.rs` accepts a typed `CanonicalTimeframe`, canonical i64-millisecond input, and no arbitrary `M<n>`/H6/H8 or unit inference. Fixed minute/hour frames use their exact grid; D1/W1/MN1 resampling uses separately proven calendar rules or fails closed—MN1 is never 43,200 minutes/30 days and broker paging never approximates it. Replace private app/live/validation/search duration/order maps with typed leaf methods where they express protocol/grid meaning. Prove all 14 official `ProtoOATrendbarPeriod` values round-trip to their exact protocol code once, no unknown/H2 value routes, and UI/API order is deterministic. Add a whole-source/dependency guard against duplicate temporal/identity types, private protocol-code/timeframe-minute match tables, 30-day MN1 constants, unit-inference resampling, and unsupported arbitrary timeframe parsing; allow exact user-selected subsets and display labels only when they call the shared typed contract. Dependency direction is `neoethos-dataset-contracts -> core re-export -> consumers`; data/app/search/CLI and later feature/mesh/GPU contracts depend downward and no reverse edge is allowed.

Replace every legacy canonical-layout constructor. Broker bootstrap/download writers build a broker-bound `CanonicalDatasetIdentity` from the authenticated environment/account server metadata and exact broker `symbolId`; imports/corpus paths without that evidence build external/non-broker-real identity. Update resolved config, CLI display/selection, data discovery, search fixtures, app/API/UI responses, and documentation to resolve through manifest identity instead of formatting `symbol=.../timeframe=...`. Add a whole-source guard for legacy path strings/direct joins, with allow-list only inside `migrate_legacy_dataset_layout` and explicit human reporting labels.

Before implementing the snapshot path, add the complete cross-platform `SourceSealV1` RED matrix. Windows opens the source read-only with only `FILE_SHARE_READ` and proves a pre-existing or later write/delete/rename conflict returns a sharing violation. Supported local Linux filesystems open a regular `O_NOFOLLOW` read-only source, acquire `F_SETLEASE/F_RDLCK` before any read, and monitor the already-open canonical parent directory for modify/move/delete/create/overflow events; install signal-aware cancellation before acquiring the lease. Test a pre-existing writable descriptor, later append/overwrite/truncate, rename-away, unlink, same-path replacement, same-inode/path ABA write-and-restore, lease break/timeout pressure, event overflow, and unsupported/network filesystem. A conflicting writer or unavailable/unproven primitive must fail before source read/staging/publication. While the seal is held, a mutation is either blocked/rejected or produces a break/event that cancels with no publication; equal metadata or a second equal read is never sufficient. Add a Linux two-source RED case: hold two path seals/copies concurrently, break only one, require only its token to cancel promptly while the other succeeds, then release/reacquire with fd reuse pressure and prove no delayed signal is lost or misrouted. Assert a request waiting for a seal signal slot holds no CPU permit. After staging seal/final identity-event verification and source-seal release, mutate during parsing and prove parsers consume only staging. Add a streamed-upload case with no mutable path. Add RED canonical runtime/broker-writer fixtures showing current timestamp magnitude inference, sort/dedup, ms-to-ns storage, and missing-volume-to-zero behavior are rejected rather than normalized.

Before configuring readers, write RED adversarial-limit fixtures for one oversized CSV/TSV line/field/count, deeply nested JSON and a giant string/token/object, huge Arrow IPC metadata/message/body/record batch, corrupt or oversized Parquet footer/row group/page/dictionary, an extreme compressed-to-decompressed ratio/cumulative output, and corrupt/oversized Vortex footer/chunk metadata. Every fixture must fail before its dangerous allocation, publish nothing, and clean up within bounded disk/RSS. Add checked-overflow cases for rows × columns × element/offset/layout bytes.

Add a typed versioned `BarTimestampConvention` to the import request/schema evidence. Use numerically and grid-identical fixtures declared as `bar_open`, `bar_close`, and `unknown`: only explicitly evidenced `bar_open` publishes directly. Grid/timeframe inference cannot choose a convention. `bar_close`/`bar_end` require a separately named converter with a declared source convention and independently proven fixed/calendar/session mapping; otherwise they fail. The canonical identity, manifest/provenance, and later feature-plan source node retain the convention. Broker fixtures accept only the exact official cTrader trendbar convention and checked protocol-unit mapping.

Also force `RLIMIT_SIGPENDING`/real-time queue exhaustion so the Linux kernel fallback `SIGIO` path is exercised. It must atomically cancel every active seal, publish none, and drain safely; startup must reject Linux path import if fallback `SIGIO` cannot be blocked and owned by the coordinator. This exhaustion case is distinct from the ordinary two-source test, where a queued per-slot real-time signal must cancel only the matching seal.

**Step 4: Run RED**

Run: `cargo test -p neoethos-data --test import_contract --test import_provenance --test import_source_seal --test canonical_vortex_timestamp_contract -- --nocapture`

Run: `cargo test -p neoethos-core --test ctrader_timeframe_contract -- --nocapture`

Run: `cargo test -p neoethos-core --test bar_timestamp_convention -- --nocapture`

Run: `cargo test -p neoethos-app --test ctrader_timeframe_contract -- --nocapture`

Run: `cargo test -p neoethos-app --test broker_vortex_writer_contract -- --nocapture`

Expected: FAIL because Arrow import is missing, Parquet/IPC use Polars, JSON paths materialize whole files, malformed rows are dropped, and the f32 gate rejects valid f64.

**Step 5: Implement `ImportSourceFormat`, typed import provenance, and shared schema mapping**

Keep source-format vocabulary inside the exact `import_service`/`import_provenance` boundary. `ImportProvenanceV1` canonically serializes the selected/detected `ImportSourceFormat`, exact verified staging SHA-256, captured stable source identity/size metadata, declared schema/timeframe/bar convention, importer version/time, and any exact decimal/scaled-integer mapping into the generic bounded `ProducerProvenanceEnvelopeV1`. Add canonical round-trip, unknown-schema, oversized-payload, and byte/hash-tamper tests. Only this import boundary may decode that schema for reporting/audit; canonical Vortex discovery/runtime validates the envelope binding but never branches on format. Validate/build the typed dataset identity, canonical timeframe, and explicitly evidenced `BarTimestampConvention` before constructing any path or dataset lock; arbitrary app/API `symbol`/`timeframe` strings never reach `Path::join`. Only `bar_open` may publish directly. Unknown/close/end stamps fail unless a separately named, independently proven converter receives the explicit source convention and records the exact fixed/calendar/session mapping plus source/canonical conventions in identity and import provenance.

Before format detection/parsing, acquire the platform `SourceSealV1` described by Step 3. On Windows hold the no-write/no-delete-sharing handle. On Linux, one process-wide `SourceSealCoordinator` OS thread owns a reserved real-time signal range and `signalfd`; each active seal receives a unique signal slot, sets `F_SETOWN_EX(F_OWNER_TID)` to that coordinator and `F_SETSIG` to its slot, and registers `{slot, si_fd, generation, cancellation}` before `F_SETLEASE`. The coordinator validates all three fields and cancels only that copy. Release performs `F_UNLCK` and an acknowledged unregister/drain barrier before the source fd is closed or the signal slot reused; handler/coordinator shutdown joins after all seals drain. Do not open or read a path route when the seal cannot be established, and do not silently fall back to advisory locking, metadata, or repeat-hash heuristics. While the seal is held, open content once and bounded-copy it into a private, non-aliased staging file while hashing bytes written; flush/fsync, reopen and verify the staged hash, compare final stable handle/path identity, require an empty mutation-event log, then release the source seal. A streamed API upload writes directly to private staging and publishes only after complete receipt, fsync, reopen, and hash verification. All format detection, decompression, validation, and provenance operate only on those staged bytes. A change after seal release cannot alter imported rows or provenance and is reported as source-after-snapshot metadata rather than silently changing the dataset.

CSV/TSV use bounded Arrow/direct record batches with delimiter/header detection; JSONL uses bounded Arrow JSON batches; JSON arrays use a streaming Serde sequence visitor that flushes fixed-size batches; Parquet uses `ParquetRecordBatchReaderBuilder` with projection and bounded batch size; IPC uses official file/stream readers. All routes feed one typed OHLCV validator and one streaming Vortex writer. Parquet/IPC/Vortex schema inspection rejects Float32 OHLC/volume/shared-feature columns before any cast; only Float64 or an exact provenance-bearing documented decimal/scaled-integer mapping can become canonical. Exact integer volume remains a raw typed column plus the declared checked f64 derivation metadata consumed by CPU/GPU volume-dependent features. `ImportProvenanceV1.source_sha256` is always the verified staging-snapshot hash parsed by these routes; the generic manifest carries only its opaque, hash-bound producer envelope.

Enforce one typed `ImportLimits` before allocation: total source/staging bytes plus a required free-disk reserve; CSV/TSV record/field bytes and field count; JSON nesting/token/string/object/record bytes; Arrow schema/metadata/message/body/buffer and batch rows/columns/bytes; Parquet footer/metadata/row-group/page/dictionary compressed and declared-uncompressed bytes, cumulative decompressed output, and compression ratio; Vortex footer/chunk/metadata/buffer bytes; and total row/column bounds. Use checked arithmetic for every layout calculation. Preflight footer/message headers through bounded I/O before constructing a reader. If the exact Arrow/Parquet/Vortex/codec API can allocate before these checks and offers no bounded hook, mark that untrusted route unsupported rather than relying on post-allocation sink batching.

The Linux coordinator additionally blocks and owns process-wide fallback `SIGIO`: real-time signal queue exhaustion can bypass the configured `F_SETSIG` slot, so any fallback `SIGIO`, unknown/overflow `signalfd` record, or loss of safe signal ownership makes attribution unknowable and cancels all active seals fail-closed. Probe `RLIMIT_SIGPENDING`, masks/ownership, and `signalfd` setup at startup; if fallback handling cannot be installed and drained, mark Linux path sealing unsupported before any source open. Recovery may admit new seals only after an acknowledged all-seals drain barrier.

**Step 6: Remove all f32 import gates**

Delete `F32_DOWNCAST_TOLERANCE`, `validate_f32_precision`, their tests/comments, and every import intermediate that narrows price/volume to f32. Removing the old f32 round-trip gate does not authorize widening a Float32 binary price, volume, or feature source: that input is rejected as precision-unrecoverable and cannot produce runnable/broker-real canonical data. Feature/model narrowing, where explicitly required later, is a separate typed boundary and cannot alter canonical prices/volume provenance.

Replace canonical Vortex read/write and broker bootstrap timestamp repair with one strict millisecond/bar-open contract. Delete or make unreachable `normalize_timestamps_to_inferred_millis` from canonical paths; writers/readers reject nanosecond/second/magnitude-ambiguous schemas, unknown/non-open timestamp convention, duplicates, descending rows, and off-grid timestamps and never sort/deduplicate. Convert an exact cTrader protocol field only from its officially declared unit and bar convention with checked arithmetic (for example trendbar UTC minutes to milliseconds), store canonical i64 bar-open milliseconds directly, preserve the officially evidenced raw volume type/unit plus checked f64 mapping, and preserve absent optional volume as absent rather than zero. Broker writers validate before publication and use the same typed timeframe/convention/identity/generation protocol. Run import -> Vortex -> CPU/GPU Footprint/volume-feature fixtures for zero, greater-than-f32 precision, integer boundaries, and non-f64-exact raw volume; the last must remain typed unsupported for those features.

**Step 7: Require a caller-owned CPU lease**

The public import entry point receives one admission grant containing its `CpuLease` and, for a Linux path route, one `SourceSealSlot`; the broker/coordinator grants them atomically, so a request waiting for an available real-time signal slot holds no CPU permit and a granted import cannot perform nested promotion. Hold the CPU lease through immutable source snapshot/copy and hashing, source-seal verification, parsing, decompression, validation, encode/write, reopen verification, publish, staging deletion, and failure cleanup; hold the seal slot only through verified staging and its acknowledged unregister/drain. Split the CPU reservation among pipeline stages; never acquire nested permits inside the service.

**Step 8: Prove bounded memory and source independence**

Run a generated large fixture with an instrumented batch sink, process peak-RSS/allocator-high-water sampling, and staging-disk high-water measurement; assert all remain under configured limits. Sink batch rows alone are not sufficient. Run every adversarial fixture and verify the typed limit names the rejected boundary before publication and bounded cleanup completes. Even for an already-Vortex source, publication must create an independently owned byte copy or a platform COW clone whose later writes cannot alias the canonical generation; a mutable hard link, reflink with unsafe sharing semantics, symlink, junction, reparse alias, or source path registered in place is forbidden. Run the complete `SourceSealV1` matrix from Step 3 for every source class. A pre-existing writer or unsupported seal fails before reading; a mutation attempted while sealed is blocked/rejected or generates a break/event and no generation, including same-inode ABA restore. No passing assertion relies only on unchanged final metadata/hash. After the source seal is released, inject the same operations during parsing and prove parsed values plus manifest hash describe only the sealed staging bytes. After a successful import, independently overwrite/truncate and then delete the original source and prove the canonical Vortex bytes, hash, manifest, and reopened values remain unchanged.

**Step 9: Verify import tests**

Run: `cargo test -p neoethos-data --test import_contract --test import_provenance --test import_source_seal --test import_bounded_memory --test import_adversarial_limits --test canonical_vortex_timestamp_contract --test canonical_timeframe_resample --test timeframe_single_source_guard -- --nocapture`

Run: `cargo test -p neoethos-core --test bar_timestamp_convention -- --nocapture`

Run: `cargo test -p neoethos-app --test broker_vortex_writer_contract -- --nocapture`

Expected: PASS for all source formats and failure fixtures.

**Step 10: Run the combined Task 3+4 compile-green matrix and commit once**

Rerun the dataset-contract leaf golden tests, core re-export/official-timeframe guards, all Task 3 publication/lease/CAS tests, all Task 4 import/broker-writer tests, and affected root workspace checks. The duplicate-type/dependency-direction guard must be GREEN and no exhaustive consumer may remain on the old core-owned enum. Only then record the first public Task 3+4 revision.

```text
feat: stream and verify canonical Vortex publication and import
```

### Task 4A: Classify and admit importer workers before production exposure

**Files:**
- Modify: `crates/neoethos-execution-budget/src/lib.rs`
- Create: `crates/neoethos-execution-budget/tests/composite_admission.rs`
- Create: `crates/neoethos-core/src/execution/backend_inventory.rs`
- Create: `crates/neoethos-core/tests/backend_threading_inventory.rs`
- Create: `crates/neoethos-app/tests/import_execution_admission.rs`
- Create: `crates/neoethos-cli/tests/import_execution_admission.rs`
- Modify: `crates/neoethos-core/src/lib.rs`
- Modify: `crates/neoethos-app/src/server/data_control.rs`
- Modify: `crates/neoethos-app/src/app_services/execution_admission.rs`
- Modify: `crates/neoethos-cli/src/main.rs`
- Modify: Vortex feature/import call sites from Tasks 3–4

**Step 1: Classify the exact locked importer graph**

Using current official documentation and the exact locked source, classify staging copy/source-stability verification, Vortex scan/decode/encode, Arrow CSV/JSON/IPC readers, Parquet readers/decompression, compression codecs, SHA hashing, and any Rayon/native workers as `lease_native_width`, `exclusive_global_pool`, `single_thread_under_partial_lease`, `device_only`, or `unsupported_concurrent`. Record whether caller/helper threads count toward each width. Before the first lease, build an `ImportAdmissionPlan` only from explicit request metadata/declared route plus the locked inventory; do not open/read the mutable source, footer, magic, or inner codec during preflight. A Linux path route includes one `SourceSealSlot` requirement; Windows path and streamed-upload routes do not. Extend the leaf/coordinator with a generic bounded composite resource grant so the CPU width and seal slot are granted atomically from one queued request, with the same child-priority/FIFO/cancellation rules; waiting holds neither resource and partial acquisition/upgrade is impossible. Auto-detect, compressed, or otherwise unknown routes take the worst-case union, and reserve the full process budget whenever any possible backend is exclusive or unclassified. Only after admission may the source be opened once and sealed; format/footer/codec detection then reads staging bytes exclusively and must be a subset of the reserved plan. A mismatch fails without lease promotion or publication. An unclassified or unbounded concurrent route fails before source parsing starts.

Implement the labels as admission behavior, not documentation: `lease_native_width(n)` reserves `n` permits for the whole declared-capacity lifetime and configures a maximum total execution width `<= n` (caller/helpers included as documented); a nontrivial saturation fixture asserts observed concurrently CPU-active workers never exceed `n` and separately proves the backend can use parallelism, while utilization/throughput are performance evidence rather than a correctness equality. `exclusive_global_pool` waits for and atomically reserves the entire installed process budget while holding zero permits and therefore serializes against every other CPU lease/backend before its one-time pool can run. `single_thread_under_partial_lease` is legal only after a probe proves a maximum of one active worker and always holds one permit; `device_only` is legal only for a proven host-noncomputing device launch/wait, with feeder/post-processing separately admitted; and `unsupported_concurrent` returns a typed error before initialization/allocation/work whenever production overlap is possible. An enabled/default backend that is unclassified, cannot configure its proven width before global initialization, or violates its class fails sealed startup; an optional requested backend fails that request before work.

**Step 2: Write RED admission/lifetime tests**

Prove the app obtains coordinator admission before constructing the import `spawn_blocking` task, the CLI synchronously obtains one top-level composite grant, cancellation before and after admission leaks neither CPU permit nor `SourceSealSlot`, and the transferred CPU lease remains alive through the one source open, snapshot copy/hash, parse, codec/decompression, validation, Vortex encode/reopen/verify/publish, and complete failure cleanup while the seal slot remains alive through its unregister/drain barrier. Prove pipeline stages split the parent CPU lease rather than nested-acquire. Add one executable fixture for each of the five backend classes, including cross-backend exclusion while a full-budget global pool is reserved and fail-before-work behavior for unclassified/unsupported paths. Saturate all seal slots and CPU permits in opposite orders and prove queued requests hold neither resource, child priority/cancellation remains responsive, and no partial-resource deadlock occurs. With partial permits already held elsewhere, queue an exclusive importer/backend request and prove it waits while holding zero permits, remains cancellable, does not defeat a higher-priority child, and after acquiring the full budget excludes every other backend type. A same-extension/different-inner-codec fixture mutates/replaces the source around admission and proves no pre-admission source-byte read occurs, parsing uses only staging, and detected staging requirements can never exceed the reserved plan; an unexpected class fails without promotion. If the locked importer graph contains no exclusive backend, preserve the dependency/source/probe evidence that establishes this.

**Step 3: Implement app and CLI admission adapters**

Route app requests through the Task 11 coordinator before `spawn_blocking` and request the complete `ImportAdmissionPlan`; for a Linux path route it returns one indivisible `ImportAdmissionGrant { cpu_lease, source_seal_slot }`, while Windows path and streamed-upload routes return the same typed grant with no slot. Make the CLI perform nightly-process budget preflight and synchronously queue/acquire that same complete grant before calling the shared importer. Neither caller may acquire CPU first and then wait for a slot. Dropped/cancelled slot waiters hold zero CPU permits, a later higher-priority child can overtake an earlier opportunistic importer, and success transfers both resource lifetimes into the blocking importer. The adapters select a source and report progress/result only; they do not parse, auto-convert, or create a second storage path.

**Step 4: Run overlap and classification probes**

Overlap two Linux source seals/copies, import parsing/hash/codecs, Vortex verification/publish, and a held local lease. Break one seal and prove only its combined grant is cancelled/released while the other completes. Assert admitted widths and measured dependency workers never exceed the installed budget, slot-wait cancellation/child priority remains responsive, and an unsupported worker or seal configuration fails closed.

**Step 5: Verify but do not commit yet**

Run: `cargo test -p neoethos-core --test backend_threading_inventory -- --nocapture`

Run: `cargo test -p neoethos-execution-budget --test composite_admission -- --nocapture`

Run: `cargo test -p neoethos-app --test import_execution_admission -- --nocapture`

Run: `cargo test -p neoethos-cli --test import_execution_admission -- --nocapture`

Expected: PASS with every importer worker classified and no unadmitted production call. Continue directly to Task 5; Tasks 4A and 5 are one compile-green commit.

### Task 5: Separate import discovery from Vortex-only runtime discovery

**Files:**
- Create: `crates/neoethos-data/src/core/import_discover.rs`
- Create: `crates/neoethos-data/tests/runtime_format_boundary.rs`
- Modify: `crates/neoethos-data/src/core/discover.rs`
- Modify: `crates/neoethos-data/src/core/loader.rs`
- Modify: `crates/neoethos-data/src/core/mod.rs`
- Modify: `crates/neoethos-data/src/lib.rs`
- Modify: `crates/neoethos-app/src/server/data_control.rs`
- Modify: `crates/neoethos-app/src/server/chart_cache.rs`
- Modify: `crates/neoethos-app/src/server/mod.rs`
- Modify: `crates/neoethos-cli/src/main.rs`
- Modify: `crates/neoethos-cli/src/gpu_bench_prepare.rs`
- Delete or rewrite: `scripts/gpu-bench/prepare_snapshot.py`
- Modify: `scripts/gpu-bench/run_rented.sh`
- Modify: `scripts/gpu-bench/README.md`
- Modify: `.github/workflows/agent-stage1.yml`
- Modify: `desktop/src-tauri/src/lib.rs`
- Modify: `desktop/src/screens/Discovery.tsx`
- Modify: `desktop/src/screens/Help.tsx`
- Modify: `desktop/test/apiContracts.test.ts`

**Step 1: Write failing boundary tests**

Runtime discovery must return only verified canonical Vortex datasets. Passing CSV/TSV/JSON/JSONL/Parquet/Arrow directly to runtime loading must fail with an import-first diagnostic. Import discovery must still recognize all seven formats and infer symbol/timeframe without registering the source as runnable.

**Step 2: Run RED**

Run: `cargo test -p neoethos-data --test runtime_format_boundary -- --nocapture`

Expected: FAIL because `DataFormat` and `resolve_path_to_vortex` currently auto-detect and auto-convert in the runtime loader.

**Step 3: Split the APIs**

Make runtime `DatasetDiscovery`/`DataFileEntry` Vortex-only. Move source extensions and recursive source walking into `ImportDiscovery`/`ImportSourceFormat`. Delete cache-on-load and `.forex-vortex-cache`; the runtime loader validates `.vortex` plus manifest and never converts.

**Step 4: Wire app and CLI to the same service**

Keep `/data/import` and the CLI `import` command, but have both call `import_service`. Remove the separate Parquet migration command or turn it into a thin alias for ordinary explicit import without source deletion. Do not quarantine/move/delete the user's original source; report errors and leave it untouched.

Remove the GPU benchmark's direct `--csv` parser. If that command accepts a non-Vortex source for user convenience, it must submit it through the same admitted shared importer, wait for verified publication, and reopen only the canonical Vortex generation; otherwise require an already imported Vortex dataset. It may not retain a private CSV reader or bypass provenance.

Delete the private CSV hashing/parsing path in `scripts/gpu-bench/prepare_snapshot.py`, or change it to accept only a verified canonical Vortex generation plus manifest. Update `run_rented.sh`, its README, and `agent-stage1.yml` so no Python/shell/workflow command drives a private `--csv` engine. A convenience source path must invoke the shared admitted import command and then pass the resulting verified Vortex identity. `collate.py` may continue emitting a summary CSV because that is a reporting output, never a runtime/import input.

**Step 5: Expose every supported source in the desktop picker/help**

The native picker must list CSV, TSV, JSON, JSONL/NDJSON, Parquet, Arrow/IPC/Feather, and Vortex. The UI result must show the canonical Vortex destination, verified row count, source hash, schema version, and failure details.

**Step 6: Verify the three surfaces**

Run: `cargo test -p neoethos-data --test runtime_format_boundary -- --nocapture`

Run: `cargo test -p neoethos-app server_contract_tests -- --nocapture`

Run in `desktop`: `npm test`

Expected: PASS; import accepts all formats, runtime accepts Vortex only.

**Step 7: Commit Tasks 4A and 5 together**

```text
refactor: isolate source import from Vortex runtime
```

---

### Task 5B: Establish scalar validity before the atomic shared-feature migration

**Files:**
- Create: `crates/neoethos-data/tests/feature_plan_producer_coverage.rs`
- Create: `crates/neoethos-data/tests/feature_plan_transform_truth.rs`
- Create: `crates/neoethos-data/tests/full_feature_validity_parity.rs`
- Create: `crates/neoethos-data/tests/feature_semantic_source_closure.rs`
- Modify: `crates/neoethos-data/src/core/features.rs`
- Modify: `crates/neoethos-data/src/core/feature_registry.rs`
- Modify: `crates/neoethos-data/src/core/smc.rs`
- Modify: `crates/neoethos-data/src/core/session_features.rs`
- Modify: `crates/neoethos-data/src/core/regime_detection.rs`
- Modify: `crates/neoethos-data/src/core/quant_features.rs`
- Modify: `crates/neoethos-data/src/core/footprint_features.rs`
- Modify: `crates/neoethos-data/src/core/normalization.rs`
- Modify: `crates/neoethos-data/src/core/resample.rs`
- Modify: `crates/neoethos-data/src/core/cross_pair_features.rs`
- Modify: `crates/neoethos-data/src/core/hpc_ta.rs`
- Modify: `crates/neoethos-data/src/lib.rs`

**Step 1: Inventory every scalar producer before changing validity**

Reconcile `feature_registry` with the compiler/current production chain in `neoethos-data/src/lib.rs`, including classic/vector-ta, SMC, quant, session, regime, and Footprint even though Footprint is currently absent from the registry. Write `feature_plan_producer_coverage` RED so every reachable value producer has exactly one typed manifest row and no registered row is unreachable/duplicated. Each row declares a provisional `SemanticSourceSetV1` closure containing the producer plus every transitively value-affecting rolling/statistical/time/calendar/alignment/dispatch helper, generated input/build generator, parameter table, and shared macro selected by the active compiler/features. It also declares a filtered `RelevantDependencySetV1` for every value-affecting external crate/library reached by that node: canonical package/source identity, resolved version, lockfile checksum or exact vendored/source hash, and enabled features. `feature_semantic_source_closure` reconciles compiler/source/Cargo-lock/feature reachability with both declarations and fails on any reachable unclassified or declared-but-unreachable source/dependency. Add mutation fixtures for each closure class: changing any declared helper or relevant dependency changes the row/plan semantic payload, while a proven unrelated helper/dependency does not. Record current warmup/first-valid, gap, zero-denominator, degenerate, alignment/staleness, and non-finite behavior; finite `0.0` placeholders are evidence of the current defect, not validity.

**Step 2: Write independent RED validity/causality fixtures**

For every producer family and transform, use hand-derived small vectors and append/future-perturbation tests to distinguish valid mathematical zero from undefined warmup/gap/stale/degenerate output. Prove no future or validation/test bar changes an earlier/train feature; HTF/cross-pair inputs use explicit causal/staleness semantics; normalization cannot fit outside the declared training partition. Run the three new tests and record each failing source before implementation.

Run: `cargo test -p neoethos-data --test feature_plan_producer_coverage --test feature_semantic_source_closure --test feature_plan_transform_truth --test full_feature_validity_parity -- --nocapture`

**Step 3: Add a parallel internal scalar value-plus-validity contract**

Introduce an internal f64 value-plus-validity result for every producer and transform. Replace finite-zero/NaN sentinels for undefined cells with explicit invalidity only after that producer's RED evidence exists; valid zero remains valid. Keep the old public route temporarily only as an uncommitted parity/compile bridge inside this atomic worktree while Tasks 6–9 move all consumers; do not add a shared f32 accessor, publish the bridge, or commit it. Record semantic versions and validity rules for later `FeaturePlanIdentity`. Formula correctness beyond the validity/causality evidence remains fail-closed for live/promotion until Task 16B's full independent review.

**Step 4: Continue without a commit**

Run the focused tests GREEN for validity/causality, retain all RED/GREEN logs, then continue directly through Tasks 6–9. The first public commit removes every legacy sentinel bridge and is the combined compile-green Task 5B–9 revision.

---

## Milestone 3 — Replace `.fstore` and remove Polars from model APIs

**Atomicity rule for Tasks 5B–9:** these tasks are one public, compile-green scalar-validity/f64/Vortex/model migration and produce one commit only after every affected crate and all-target surface passes. Task 5B writes independent RED validity/causality evidence before changing any producer sentinel, may add a parallel internal value+validity route, and cannot commit its temporary legacy bridge. Task 6 may add/test a private `VortexFeatureStore` while the legacy public f32 `FeatureFrame` remains temporarily available in the same dirty worktree; it must not flip public feature types, delete `.fstore`, or commit. Task 6A performs the coordinated public f64+validity flip and ABI/artifact migration, but direct model consumers such as ensemble bootstrap and genetic constructors cannot compile until Tasks 7–9 replace their `Vec<f32>`/`Array2<f32>`/Polars contracts. Therefore none of Tasks 5B–8 commits or claims all-target GREEN alone: continue through every producer, model family/orchestrator, Polars removal, and legacy-sentinel bridge deletion, then run the combined workspace/feature matrix and record the first public revision. No shared f32 compatibility accessor, committed sentinel bridge, or broken intermediate revision may enter history.

### Task 6: Migrate out-of-core features from `.fstore` to Vortex

**Files:**
- Create: `crates/neoethos-data/src/core/vortex_feature_store.rs`
- Create: `crates/neoethos-data/src/core/feature_run_lease.rs`
- Create: `crates/neoethos-data/tests/vortex_feature_store_contract.rs`
- Create: `crates/neoethos-data/tests/vortex_feature_store_crash.rs`
- Create: `crates/neoethos-data/tests/vortex_feature_run_lease.rs`
- Modify: `crates/neoethos-data/src/core/features.rs`
- Modify: `crates/neoethos-data/src/core/mod.rs`
- Modify: `crates/neoethos-data/src/lib.rs`
- Modify: `crates/neoethos-app/src/server/system_status.rs`
- Delete after parity passes: `crates/neoethos-data/src/core/feature_store.rs`

**Step 1: Write the old/new parity test while both implementations exist**

Add warmup, gap/NaN, zero-denominator, stale-alignment, degenerate, valid-zero, and invalid cells. Compare validity bitmaps through full/projected/window reads; the Vortex path retains validity distinctly and never encodes invalid as numeric zero.

Write the f32-exact Task 1 fixture through `.fstore` and through a Vortex `StructArray` containing timestamp/row identity and named f64 feature fields; compare names/order, projected columns, full columns, row windows, nested windows, feature values, label alignment, and fixed GA/model outputs. For the high-precision fixture, compare the new path to the pre-narrowing f64 CPU indicator bits and assert it intentionally differs from the old `.fstore` result. Before cleanup implementation, add a two-process RED fixture: one process holds a long-lived feature run and repeatedly projects it while another sweeps, then a crash releases the OS lease and permits cleanup. Force misleading age/PID metadata and PID reuse; none may delete a live run.

**Step 2: Run RED against the missing Vortex store**

Run: `cargo test -p neoethos-data --test vortex_feature_store_contract --test vortex_feature_run_lease -- --nocapture`

Expected: FAIL because `VortexFeatureStore` does not exist.

**Step 3: Implement projection/range-backed storage**

Store each named f64 field with its Arrow/Vortex validity buffer and immutable validity semantics. Projection, row windows, and nested windows preserve those bits without materializing invalid cells as zero or relying on NaN payloads.

Add `VortexFeatureStore` privately/alongside the current public API and store named f64 feature fields in bounded Vortex chunks atomically in a run-scoped scratch directory. Exercise Vortex and Vortex-window backing variants with immutable schema/range metadata through the new store's direct contract while the legacy `FeatureData` public type remains unchanged for this preparatory task. Disk-backed accessors accept an existing lease/read context and use projection plus row-range pushdown. Defer the public `FeatureData::InMemory`/accessor f64 flip and deletion of the common `val as f32` cube write to the coordinated Task 6A migration.

**Step 4: Add a weighted decoded-chunk cache**

Include validity buffers in cache keys/values and byte accounting; a cached projection must reproduce the exact value bits and validity bits.

Cache only decoded requested f64 chunks, keyed by file identity/schema/column/range, with an explicit byte cap derived from the hardware memory plan. Recalculate all memory estimates for eight-byte features. The cache is memory-only, run-scoped, observable, and cannot become another persisted format.

**Step 5: Add crash/cleanup tests**

Prove unfinished scratch files are never opened and normal RAII removes the run directory. Each active run owns an OS-backed cross-process `FeatureRunLease`; startup sweep takes a nonblocking exclusive cleanup lock and skips every live run, even when its timestamp is old or a PID has been reused. Process death releases the lock so a contained crashed run becomes collectible. Path validation prevents cleanup outside the configured scratch root. PID/age manifests are diagnostic only and never liveness authority.

**Step 6: Migrate all producers/consumers**

Add the new Vortex creation/open/cleanup path alongside `.fstore` in `neoethos-data/src/lib.rs` for direct parity tests and prepare status reporting. Do not yet change public `FeatureFrame::{feature_column,sample_window,row_slice,row_window}` signatures or production callers. Preserve bounded memory; do not materialize the full cube as an intermediate.

**Step 7: Run parity and performance checks**

Run: `cargo test -p neoethos-data --test vortex_feature_store_contract --test vortex_feature_store_crash --test vortex_feature_run_lease -- --nocapture`

Repeat the Task 1 selected-column/window benchmarks. Profile and repair material regressions in the Vortex path before deleting `.fstore`.

Apply the frozen Task 1 gates to the ten measured release runs: selected/window and GA median <= 1.10x old and p95 <= 1.15x; end-to-end training median <= 1.05x and p95 <= 1.10x; RSS/disk remain below their formula-derived hard ceilings. Preserve raw samples. A threshold miss blocks deletion/commit unless the operator explicitly approves the documented profile/root-cause waiver.

**Step 8: Defer the public switch and deletion**

Keep `.fstore` only long enough to serve as the in-worktree parity oracle. Do not commit this dual-path state. Continue directly to Task 6A, where all public consumers move to f64/Vortex and the writer, reader, extension, compatibility fallback, stale cleanup names, and production comments are deleted together.

**Step 9: No commit—continue directly to Task 6A**

Record the focused test output, but do not stage or commit Task 6 separately.

### Task 6A: Carry f64 features through CPU search and the supported native-CUDA evaluator

**Files:**
- Create: `crates/neoethos-feature-contracts/Cargo.toml`
- Create: `crates/neoethos-feature-contracts/build.rs`
- Create: `crates/neoethos-feature-contracts/src/lib.rs`
- Create: `crates/neoethos-feature-contracts/src/source_manifest.rs`
- Create: `crates/neoethos-feature-contracts/tests/golden_identity.rs`
- Create: `crates/neoethos-feature-contracts/tests/semantic_source_manifest.rs`
- Create: `crates/neoethos-feature-contracts/tests/semantic_source_closure.rs`
- Create: `crates/neoethos-data/tests/feature_artifact_provenance.rs`
- Create: `crates/neoethos-data/tests/feature_validity_parity.rs`
- Modify: root `Cargo.toml` and `Cargo.lock`
- Modify: `crates/neoethos-dataset-contracts/Cargo.toml`
- Modify: `crates/neoethos-core/Cargo.toml`
- Modify: `crates/neoethos-data/Cargo.toml`
- Modify: `crates/neoethos-gpu-contracts/Cargo.toml`
- Modify: `mesh/Cargo.toml` and `mesh/Cargo.lock`
- Modify: `crates/neoethos-data/src/core/cross_pair_features.rs`
- Modify: `crates/neoethos-data/src/core/features.rs`
- Modify: `crates/neoethos-data/src/core/normalization.rs`
- Modify: `crates/neoethos-data/src/lib.rs`
- Modify: `crates/neoethos-data/src/test_fixtures.rs`
- Modify: `crates/neoethos-models/src/runtime/prediction.rs`
- Create: `crates/neoethos-models/src/runtime/model_implementation_identity.rs`
- Modify: `crates/neoethos-models/src/runtime/training_artifact.rs`
- Modify: `crates/neoethos-models/src/runtime/profile.rs`
- Modify: `crates/neoethos-models/src/training_orchestrator.rs`
- Create: `crates/neoethos-models/tests/runtime_prediction_f64.rs`
- Create: `crates/neoethos-models/tests/model_implementation_identity.rs`
- Modify: `crates/neoethos-core/src/config.rs`
- Modify: `crates/neoethos-core/src/storage.rs`
- Modify: `crates/neoethos-core/src/contracts/primitives.rs`
- Modify: `crates/neoethos-core/src/contracts/envelope.rs`
- Create: `crates/neoethos-core/src/feature_plan_identity.rs`
- Create: `crates/neoethos-core/tests/feature_plan_identity.rs`
- Modify: `crates/neoethos-core/src/domain/prop_firm.rs`
- Modify: `crates/neoethos-core/src/domain/risk.rs`
- Modify: `crates/neoethos-core/src/domain/risky_mode.rs`
- Modify: `crates/neoethos-core/src/scheduler.rs`
- Modify: `crates/neoethos-core/tests/config_has_recipient.rs`
- Modify: `crates/neoethos-core/tests/shipped_config_matches_defaults.rs`
- Modify: `crates/neoethos-search/src/discovery.rs`
- Modify: `crates/neoethos-search/src/eval.rs`
- Modify: `crates/neoethos-search/src/checkpoint.rs`
- Modify: `crates/neoethos-search/src/batch_ledger.rs`
- Modify: `crates/neoethos-search/src/strategy_db.rs`
- Modify: `crates/neoethos-search/src/live_portfolio.rs`
- Modify: `crates/neoethos-search/src/validation.rs`
- Modify: `crates/neoethos-search/src/orchestration.rs`
- Modify: `crates/neoethos-search/src/parity.rs`
- Modify: `crates/neoethos-search/src/genetic/evolution_math.rs`
- Modify: `crates/neoethos-search/src/genetic/diversity.rs`
- Modify: `crates/neoethos-search/src/genetic/migration.rs`
- Modify: `crates/neoethos-search/src/genetic/regime_labels.rs`
- Modify: `crates/neoethos-search/src/genetic/runtime_overrides.rs`
- Modify: `crates/neoethos-search/src/genetic/search_engine.rs`
- Modify: `crates/neoethos-search/src/genetic/seed_templates.rs`
- Modify: `crates/neoethos-search/src/genetic/smc_indicators.rs`
- Modify: `crates/neoethos-search/src/genetic/strategy_gene.rs`
- Modify: `crates/neoethos-search/src/backend.rs`
- Modify: `crates/neoethos-search/src/cubecl_eval.rs`
- Modify: `crates/neoethos-search/src/gpu_fallback.rs`
- Modify: `crates/neoethos-search/src/gpu_native/benchmark.rs`
- Modify: `crates/neoethos-search/src/gpu_native/capability.rs`
- Modify: `crates/neoethos-search/src/gpu_native/cpu_strategy.rs`
- Modify: `crates/neoethos-search/src/gpu_native/device_trades.rs`
- Modify: `crates/neoethos-search/src/gpu_native/engine.rs`
- Modify: `crates/neoethos-search/src/gpu_native/instrumentation.rs`
- Modify: `crates/neoethos-search/src/gpu_native/mod.rs`
- Modify: `crates/neoethos-search/src/gpu_native/parity_hierarchy.rs`
- Modify: `crates/neoethos-search/src/gpu_native/population_fixture.rs`
- Modify: `crates/neoethos-search/src/gpu_native/prototype_a.rs`
- Modify: `crates/neoethos-search/src/gpu_native/prototype_a_engine.rs`
- Modify: `crates/neoethos-search/src/gpu_native/prototype_b_engine.rs`
- Modify: `crates/neoethos-search/src/gpu_native/prototype_b_mirror.rs`
- Modify: `crates/neoethos-search/src/gpu_native/prototype_b_population_eval.rs`
- Modify: `crates/neoethos-search/src/gpu_native/prototype_bc.rs`
- Modify: `crates/neoethos-search/src/gpu_native/prototype_c_engine.rs`
- Modify: `crates/neoethos-search/src/gpu_native/prototype_c_engine/device.rs`
- Modify: `crates/neoethos-search/src/gpu_native/prototype_c_engine/device_tests.rs`
- Modify: `crates/neoethos-search/src/gpu_native/prototype_c_gpu.rs`
- Modify: `crates/neoethos-search/src/gpu_native/prototype_population_oracle.rs`
- Modify: `crates/neoethos-search/src/gpu_native/prototype_population.rs`
- Modify: `crates/neoethos-search/src/gpu_native/ranking.rs`
- Modify: `crates/neoethos-search/src/gpu_native/scenario.rs`
- Modify: `crates/neoethos-search/src/gpu_native/semantics.rs`
- Modify: `crates/neoethos-search/src/gpu_native/signal_trace_gpu.rs`
- Modify: `crates/neoethos-search/src/gpu_native/snapshot_fixture.rs`
- Modify: `crates/neoethos-search/src/gpu_native/trade_invariants.rs`
- Modify: `crates/neoethos-search/src/gpu_native/trade_trace_gpu.rs`
- Modify: `crates/neoethos-gpu-contracts/src/lib.rs`
- Modify: `crates/neoethos-gpu-contracts/tests/layout.rs`
- Modify: `crates/neoethos-gpu-cuda/src/lib.rs`
- Modify: `crates/neoethos-gpu-cuda/src/population.rs`
- Modify: `crates/neoethos-gpu-cuda/native/neoethos_gpu_cuda.h`
- Modify: `crates/neoethos-gpu-cuda/native/layout_asserts.cpp`
- Modify: `crates/neoethos-gpu-cuda/native/prototype_b.cu`
- Modify: `crates/neoethos-gpu-cuda/native/prototype_b_population.cu`
- Modify: `crates/neoethos-gpu-cuda/native/smoke.cu`
- Modify: `crates/neoethos-gpu-cuda/native/stub.cpp`
- Modify: `crates/neoethos-trader/src/gene_signal.rs`
- Modify: `crates/neoethos-trader/src/data_replay.rs`
- Modify: `crates/neoethos-trader/src/blend_signal.rs`
- Modify: `crates/neoethos-app/src/app_services/experience_store.rs`
- Create: `crates/neoethos-app/src/app_services/live_experience_writer_lease.rs`
- Modify: `crates/neoethos-app/src/app_services/experience_train.rs`
- Modify: `crates/neoethos-app/src/app_services/mod.rs`
- Modify: `crates/neoethos-app/src/app_services/live_trading.rs`
- Modify: `crates/neoethos-app/src/app_services/supervisor.rs`
- Modify: `crates/neoethos-app/src/app_services/federation.rs`
- Modify: `crates/neoethos-app/src/server/federation.rs`
- Modify: `crates/neoethos-app/src/server/risk.rs`
- Create: `crates/neoethos-app/src/bin/migrate_live_experience.rs`
- Create: `crates/neoethos-app/tests/live_experience_vortex.rs`
- Create: `crates/neoethos-app/tests/federation_gene_schema.rs`
- Create: `crates/neoethos-app/tests/federated_portfolio_schema.rs`
- Modify: `crates/neoethos-autoresearch/src/runner.rs`
- Modify: `crates/neoethos-autoresearch/src/runner/streaming.rs`
- Modify: `crates/neoethos-autoresearch/src/runner/tests.rs`
- Modify: `crates/neoethos-autoresearch/src/contracts.rs`
- Modify: `crates/neoethos-autoresearch/src/goals.rs`
- Modify: `crates/neoethos-autoresearch/src/judge.rs`
- Modify: `crates/neoethos-autoresearch/tests/loop_no_signal.rs`
- Modify: `crates/neoethos-autoresearch/tests/support/mod.rs`
- Modify: `mesh/src/main.rs`
- Create: `mesh/tests/gene_migration_schema.rs`
- Modify: `crates/neoethos-cli/src/gpu_bench_prepare.rs`
- Modify: `crates/neoethos-search/examples/gpu_eval_bench.rs`
- Modify: `crates/neoethos-search/examples/htf_effective_n_probe.rs`
- Modify: `crates/neoethos-search/examples/htf_prefilter_probe.rs`
- Modify: `crates/neoethos-search/tests/higher_timeframe_lane_measured.rs`
- Modify: `crates/neoethos-search/src/discovery_tests.rs`
- Create: `crates/neoethos-search/tests/f64_feature_lane.rs`

**Step 1: Write the precision regression before changing types**

Before implementation, make `feature_validity_parity` RED across scalar output, in-memory `FeatureFrame`, Vortex write/reopen/projection/cache, normalization, row/model selection, trader/live gating, and CUDA metadata. Warmup, gaps, non-finite output, zero denominators, stale alignments, and degenerate cells stay invalid; valid mathematical zero remains valid and distinct. Assert the current normalization path's non-finite-to-`0.0` rewrite is a failure and that invalid cells cannot fit statistics, rank, threshold, signal, size, or enter a live-experience intent.

Construct an indicator value and gene threshold that are distinct in f64 but collapse to the same f32 value. Assert the pre-narrowing CPU f64 path produces the mathematically expected pre-financial signal while the current FeatureFrame/search path demonstrates the wrong/equalized result; it may produce a trade/ledger only when the fixture carries the exact synchronized broker capability, otherwise it must return typed unsupported before finance. Add f64 `to_bits()` checks across feature construction, normalization, higher-timeframe alignment, selected-column extraction, core SMC configuration, search input, every Gene-bearing persisted artifact, the public SQLite `SavedStrategy` record, prop-firm preset/API/risk serialization boundaries, autoresearch promotion evidence, federation/mesh migration, strategy snapshot serialization, the exact live-entry row persisted for experience training, the shared serialized `RuntimePrediction` class probabilities/confidence, live `MlDecision` veto/confidence/sizing before financial execution, GPU host DTO serialization, and the native ABI boundary. Financial eligibility tests either supply exact broker evidence or assert the Task 16C gate; they never execute a default/heuristic formula merely to test f64. Add layout/version tests that expect the new f64 GPU contract, reject ABI v3 bytes/inputs, and reject every previous f32-derived or unversioned artifact/wire version rather than silently widening it. Add scheduler capacity tests whose expected byte counts come from the actual f64 host/device layouts.

Before adding `FeaturePlanIdentity`, write RED tests whose two plans have identical feature names/types/order/length but different formula semantic version, parameter convention, warmup/validity rule, or implementation-manifest hash; every FeatureFrame/Vortex/artifact/live/federation/GPU consumer must reject the mismatch. Before changing live persistence, write RED crash/recovery tests that require an immutable pending Vortex generation containing the exact f64 row before order send, stable client correlation and request fingerprint, atomic binding of account/environment+order+position+deal identity, partial fills/closes, broker event before response, duplicate/out-of-order deals, publish-before-ack retry, rejected/expired orders, and atomic pending-to-completed reference reconciliation. Assert the lifecycle journal contains IDs/state/Vortex generation+hash references only and cannot serialize feature values as JSON, SQLite fields, `.fstore`, or bespoke binary. Inject crashes pre-send, post-send/pre-response, post-broker-ack/pre-bind, and at every bind/publish boundary. Add a two-process same-account/store test: only one process can own the writer lease and send; after owner crash, recovery acquires the released OS lease and reconciles without duplicate/lost records. An unproven/ambiguous correlation or retry must become an explicit integrity-error exclusion, never an automatic resend, reconstructed row, or training record.

**Step 2: Run RED**

Also make RED the identity/provenance split: unchanged transformation semantics and fitted state over two different canonical input generations must produce equal `FeaturePlanIdentity` but different `DatasetFeatureArtifactProvenance` and cache/artifact identity; changing a semantic node or fitted state changes the plan. Add unrelated-commit/same-relevant-blobs, relevant-blob-change, LF/CRLF, dirty-TDD, and packaged-no-`.git` manifest fixtures before implementation. Make `model_implementation_identity` RED for an artifact whose feature plan is unchanged while only trainer/inference source, a reachable Burn/Candle/tree/native dependency version/feature, serialization/class mapping, precision/determinism contract, or required device-capability contract changes; missing, `unknown-local-source`, legacy, and mismatched identities must fail before inference, while an unrelated dependency change preserves the identity.

Run: `cargo test -p neoethos-search --test f64_feature_lane -- --nocapture`

Run: `cargo test -p neoethos-models --test runtime_prediction_f64 -- --nocapture`

Run: `cargo test -p neoethos-models --test model_implementation_identity -- --nocapture`

Run: `cargo test -p neoethos-app --test live_experience_vortex -- --nocapture`

Run: `cargo test -p neoethos-gpu-contracts --test layout -- --nocapture`

Run: `cargo test -p neoethos-core --test feature_plan_identity -- --nocapture`

Run: `cargo test -p neoethos-feature-contracts --test golden_identity -- --nocapture`

Run: `cargo test -p neoethos-feature-contracts --test semantic_source_manifest -- --nocapture`

Run: `cargo test -p neoethos-feature-contracts --test semantic_source_closure -- --nocapture`

Run: `cargo test -p neoethos-data --test feature_artifact_provenance -- --nocapture`

Run: `cargo test -p neoethos-data --test feature_validity_parity -- --nocapture`

Expected: FAIL at the current `Array2<f32>`, `ArrayView2<f32>`, `Vec<f32>`, live `as f32`, JSONL experience store, ABI-v3, and f32 GPU buffer/layout boundaries.

**Step 3: Convert the shared CPU/search lane**

Make validity a first-class shared contract alongside f64 values. `FeatureData`, every accessor/view, Vortex/cache, normalization, search/model row selection, trader, and live capture carry it. Normalization computes immutable fitted state only from valid cells in the explicitly declared training partition and preserves invalidity on transform; delete non-finite-to-zero replacement and the implicit whole-series/80% fit heuristic. A row missing any required feature is explicitly ineligible with stable original index and reason telemetry; it is never silently dropped/imputed or allowed to influence thresholds, ranking, signal, sizing, or training. A model-specific imputer remains unavailable unless separately named, train-split-only, identity-versioned, and independently reviewed.

Flip `FeatureData::InMemory`, shared accessors, the now-proven Vortex store, common feature matrices/views, normalization, alignment, core configuration SMC schedules/weights/gates, GA indicators, strategy/gene weights, evaluation inputs, thresholds, trader signal/replay/blend consumers, every `PropFirmConstraints`/preset/API/risk/evaluation field, CLI GPU preparation, tests, and examples to f64 in one coordinated change. Remove the f32-widening rounding workaround and test exact daily-loss, total-drawdown, challenge-profit, and minimum-monthly-profit boundaries through core, app, search, and autoresearch. Convert `RuntimePrediction`'s shared serialized `[class; 3]` and optional confidence fields plus the shared/live `MlDecision` contract to f64; bump/reject the old prediction wire/artifact form where serialized. Every intrinsically f32 model must widen finite, range-checked outputs inside its exact named adapter before constructing `RuntimePrediction`, so f32 cannot reach ensemble aggregation, regime/anomaly gating, trader veto, confidence, or position sizing. Switch production out-of-core creation/open/cleanup to Vortex and delete `.fstore` only after every consumer has moved. Derive scheduler RAM/VRAM byte estimates from the synchronized f64 element/layout definitions, including actual padding/SoA rules, and add overflow/capacity regressions; do not leave f32 comments or guessed constants. Do not preserve old f32 results for the high-precision fixture; the pre-narrowing f64 value is the precision-preservation reference for this migration, while Task 16B independently establishes whether the indicator formula itself is correct.

Introduce one versioned `FeaturePlanIdentity` shared by data, search, models, trader, app, federation/mesh, and GPU contracts. A new `neoethos-feature-contracts` leaf is the sole constructor/hasher and also defines `DatasetFeatureArtifactProvenance`; it depends only on the earlier zero-external-dependency `neoethos-dataset-contracts` leaf for `CanonicalDatasetIdentity`, `CanonicalTimeframe`, `BarTimestampConvention`, and their canonical bytes. Core/data/mesh/GPU depend downward on the appropriate leaves; feature contracts never import core/data, dataset contracts never import feature contracts, and no crate duplicates either type family. Standalone mesh and low-level GPU contracts therefore do not import the engine graph or reimplement hashing. Add `cargo tree` cycle/direction guards and cross-crate/workspace golden vectors proving the exact re-exported dataset types and feature/provenance bytes agree. The plan identity hashes the complete ordered value-producing transformation DAG from canonical source timeframe/schema contracts to the final `FeatureFrame`, not final names, a concrete source generation, or the vector-ta registry alone. Every node carries a typed operation tag, semantic version, ordered input-node/output identities, exact parameters/conventions, warmup/first-valid/non-finite/validity behavior, and physical schema. Required nodes cover source type/timeframe and input-derivation semantics; raw/vector-ta formula manifest; fixed/calendar resample, higher-timeframe alignment and staleness; normalization enabled/method/fit window plus immutable fitted-preprocessor state hash only when it changes numeric values; cross-pair derivation/alignment semantics; and every other production derived/post-processing stage. Concrete dataset/live generation hashes and ranges belong only to the separately domain-separated provenance type. Canonical topological order is deterministic and rejects duplicate/missing/cyclic node references. Use SHA-256 with domain `neoethos.feature-plan.identity\0` and a fixed identity-version tag over manually canonicalized bytes: fixed field order/variant tags; u32 length prefixes/counts and all integers big-endian; exact UTF-8 bytes; ordered nodes, outputs, and parameter tuples; f64 parameters as big-endian `to_bits()` after rejecting non-finite and negative zero; and raw 32-byte formula/capability manifest, relevant semantic-implementation payload, and value-affecting fitted-state hashes. JSON map order, platform-native integers, Unicode/case normalization, `DefaultHasher`, and per-crate hashing are forbidden. Mesh/artifacts/Rust GPU and C/CUDA wire layouts carry typed plan identity plus artifact provenance; only the shared leaf constructs them.

Persist the identity in FeatureFrame/Vortex metadata, resident f64 GPU metadata, training artifacts, checkpoints/portfolios, `SavedStrategy`, live experience, autoresearch promotion evidence, federation, and mesh. Golden canonical-byte/hash vectors run from the leaf, core, data, mesh, and GPU-contract crates on Windows/Linux and round-trip through every wire format. Reject mismatches even when names, types, order, and length are identical, including cases where only normalization enablement/method/fit window/fitted-state hash, source timeframe, resample/alignment/staleness semantic version, or cross-pair derivation changes. The vector-ta manifest is one DAG node, not the whole identity. Task 16B formula repairs must bump the affected semantic version/manifest hash and therefore change this identity; a legacy artifact regenerates or uses an explicit migration only when semantic equivalence is independently proved.

Split semantic compatibility from concrete data provenance. `FeaturePlanIdentity` hashes the source timeframe/schema contract and complete value-producing transformation DAG, including immutable fitted-preprocessor state when it changes numeric values, but never a concrete dataset/live generation hash. `DatasetFeatureArtifactProvenance` carries exact input generation hashes/ranges; caches and trained artifacts may bind `{plan, provenance}`, while live rows carry both and training partitions by plan/fitted state rather than every append generation. Same plan over different generations keeps semantic identity but produces distinct provenance/cache identity.

The implementation-manifest portion of the plan hashes only exact relevant canonical source/blob/dependency payloads and the generated formula/capability manifest. Implement versioned `SemanticSourceManifestV1`: each row owns a canonical `SemanticSourceSetV1` transitive closure, and every source entry is an exact canonical repo-relative forward-slash UTF-8 path, explicit `utf8_text|raw_binary|generated(declared payload kind)` tag, and SHA-256; entries are bytewise path-sorted and length-prefixed. The closure includes the top-level producer plus every value-affecting rolling/statistical/time/calendar/alignment/dispatch helper, shared macro/table, generated input, and build generator reachable for that node under the locked compiler/features. Each row also owns a canonical filtered `RelevantDependencySetV1` entry per value-affecting external crate/library with package name, source kind, canonical registry URL+package or Git URL+immutable revision or repository-relative vendored/path identity, resolved version, exactly one lockfile checksum/source-manifest hash, and enabled features. Its bytes use domain `neoethos.relevant-dependencies.v1\0`, fixed version and big-endian `u32` counts/lengths, explicit field/source-kind tags, byte-sorted entries and byte-sorted feature names; absolute/local paths, branch-only Git refs, ambient aliases, missing fields, duplicates, conflicts, unknown kinds, overflow, and noncanonical order reject. Reconcile compiler-selected module/source and Cargo feature/lock/call-graph reachability with every declared set and reject reachable unclassified or declared-unreachable sources/dependencies. Mutation/golden tests cover registry/Git/vendored/path encodings and each closure class, proving only affected rows/identities change, including a relevant `chrono` time/calendar change versus an unrelated dependency update. Reject absolute/dot/backslash/duplicate/case-fold-colliding, symlink/reparse, and submodule source entries. Text validates UTF-8 and converts CRLF and lone CR to LF only—no BOM/whitespace/Unicode normalization; binary hashes raw bytes. Explicitly classify Rust, `.cu`, `.h`, `.c/.cc/.cpp`, build scripts, and text manifests as text; generated entries list generator/input identities and their payload kind. Domain-separate and length-prefix the ordered final bytes. Repository commit is provenance outside the plan identity. Dirty RED/GREEN worktrees generate/test against canonicalized current relevant source bytes without a forbidden intermediate commit; Task 17/18/release runs the identical canonicalizer over final committed Git blob payloads and rejects relevant stale/dirty/index mismatch. Runtime reads the embedded manifest and needs no `.git`. Unrelated commits with identical relevant blobs and LF/CRLF checkout conversion preserve identity; changing a relevant source/dependency payload changes it.

Construct and persist `ModelImplementationIdentityV1` separately from `FeaturePlanIdentity`. Hash the exact trainer and inference adapter source manifests, filtered canonical relevant dependencies/features, model/backend serialization schema, label/class mapping, f64-to-backend and backend-to-f64 precision boundary, deterministic-reduction contract, and required CPU/CUDA execution-capability contract. Replace the current bare/possibly `unknown-local-source` commit provenance in core artifact envelopes with typed source/build provenance plus the required model identity; keep repository commit outside the semantic hash. Training artifact creation, profile/orchestrator publication, checkpoint/promotion wrapping, resume, load, and inference all validate the identity before model bytes are accepted. Record actual stable training-device identity/launch proof only as artifact provenance. Reject missing, `unknown-local-source`, old-version, source/dependency/schema/precision/determinism/device-contract mismatch before inference or resumed training. Golden/mutation tests prove an accepted dependency upgrade changes only model identities whose reachable trainer/inference implementation changed.

Implement `DatasetFeatureArtifactProvenanceV1` in the shared leaf as a canonical source-node mapping. Each exact `FeaturePlanIdentity` source-node ID binds once to typed `CanonicalDatasetIdentity`, manifest schema/hash, generation ID, Vortex hash, `BarTimestampConvention`, and strictly ordered non-overlapping consumed half-open row ranges plus timestamp bounds. Canonicalize by source-node ID. Reject duplicate/missing/extra/unknown/swapped nodes, overlapping/out-of-range segments, and convention/schema/timeframe/hash inconsistency. Tests cover reordered equivalent mappings plus swapped HTF/cross-pair bindings and enforce the same validation through cache, live rows, federation, and `ResidentF64FeatureBuffer`.

**Step 4: Replace live-learning JSONL and preserve the exact f64 entry row**

Change `LiveExperience.features` to `Vec<f64>`, bump its schema version, and remove the live `Vec<f32>` annotation/cast. Before accepting an intent, acquire one OS-backed cross-process `LiveExperienceWriterLease` scoped by exact broker environment/account plus live-store identity and hold it for the coordinator lifetime; a second process fails or defers before persisting or sending any order. Before submitting an order, write/fsync/reopen/hash/schema-verify a versioned immutable pending Vortex generation containing the exact f64 row, `FeaturePlanIdentity`, typed `DatasetFeatureArtifactProvenance` with exact input generations/ranges, immutable portfolio hash, exact broker environment/account/symbol, request fingerprint, and stable client correlation id; only durable acknowledgement of that generation and its typed control reference permits network send. Missing, malformed, or plan/provenance-inconsistent input fails before send. A transactional lifecycle journal/manifest may contain IDs, state, and Vortex generation/hash references only—never feature values or another feature-row serialization. On response or broker event, atomically bind authoritative order, position, deal, fill, and lifecycle ids to that reference. Define partial-fill and partial-close completion semantics and make the rule collision-free under account/position reuse. On restart, first acquire the writer lease, then reconcile intent-without-response, event-before-response, response/fill-after-crash, rejected/expired orders, duplicate submission/events, pending/open positions, and `ProtoOAClosePositionDetail`/deal history. If official/captured cTrader evidence does not prove idempotent correlation for an ambiguous intent, do not resend or infer: persist an integrity error, exclude it from learning, and require explicit reconciliation.

Atomically/idempotently transition the pending reference into completed-manifest ownership of the same verified immutable Vortex row; never rewrite the feature row into the lifecycle journal, SQLite, `.fstore`, JSON, or custom binary. Each record carries its stable lifecycle id, immutable portfolio artifact hash, `FeaturePlanIdentity`, and the unchanged typed `DatasetFeatureArtifactProvenance`; a mutable portfolio path is metadata only. The cross-process writer lease plus bounded single-writer coordinator serializes manifest read/modify/swap, applies backpressure, acknowledges only durable accepted records, deduplicates retried ids, and drains/joins on shutdown. Multiple symbol loops or two processes sharing one account/store cannot last-writer-win, lose a record, or duplicate after publish-before-ack. The writer is admitted through the installed app coordinator and holds a CPU lease through pending Vortex encode/verify/publish, reconciliation transition, and failure cleanup; the trading loop performs no unbudgeted filesystem or encoding work. `count` and `train_from_experience` scan only the Vortex schema, group semantic compatibility by plan plus fitted state, retain/filter/report concrete provenance separately, reject missing/malformed provenance, equal-length rows with different names/order/semantics, incompatible provenance contracts, or overwritten portfolio bytes, require finite values, and surface malformed records instead of silently continuing. Tests inject crashes at entry acknowledgement, before and after close receipt, after publish-before-ack, restart/retry, partial closes, duplicate/out-of-order deal events, cross-process ownership handoff, and reconciliation, proving an unbiased exactly-once completed set plus durable unresolved pending set with bit-identical plan/provenance fields.

Provide `migrate_live_experience` as a separately named, offline, one-time JSONL-v1 archival utility: its synchronous main performs budget preflight before any worker/runtime, acquires one top-level lease, uses bounded memory, and holds the lease through JSONL parsing/validation, archive-Vortex encode/reopen/verify/publish, and failure cleanup. Because v1 rows were narrowed to f32 and have no provable `FeaturePlanIdentity`, every output record uses a distinct legacy archive envelope with `precision_unrecoverable=true` and `feature_identity=unknown_legacy`. It never derives a current identity from feature names/count/order or a mutable portfolio path. It archives every valid record and publishes once, or reports the exact bad record and publishes nothing; cancellation leaves no accepted partial generation. Current counter/trainer/promotion/live paths reject the archive schema even when shape/names match true f64 rows. Add it to the startup/backend-classification matrix. Production startup, counter, and trainer never auto-read JSONL.

**Step 5: Version every persisted and wire Gene/SMC artifact**

Bump and test checkpoint, selected-portfolio, live-portfolio, streaming batch-ledger, Prototype-A upload, snapshot-fixture, and autoresearch-promotion schemas. Replace the public SQLite `SavedStrategy` record's unversioned `JsonValue` weights/thresholds/SMC payloads with a typed, finite-validated f64 envelope and explicit schema/provenance column; migrate its database and the search strategy DB in transactions. Bump any config artifact schema that serializes the f64 SMC schedule/weights/gates. Old f32-derived or unversioned rows/artifacts fail with a regenerate diagnostic unless a separately named offline converter emits `precision_unrecoverable=true`; deserializing old decimal values straight into f64 is forbidden.

Replace app federation's bare `Vec<Gene>` and mesh's opaque `serde_json::Value` migrants with a typed versioned f64 envelope plus advertised capability. Replace the separate worker submission path's `portfolio_json: String`/`trades_json: String` API and `app_services/federation.rs` opaque `Value` parser/current-live-portfolio writer with a typed envelope containing schema/capability version, immutable portfolio artifact hash, `FeaturePlanIdentity`, typed `DatasetFeatureArtifactProvenance`, typed f64 genes and trades, run/config provenance, and finite-value validation. Validate the envelope and plan/provenance consistency before counting or writing anything; publish with the same versioned live-portfolio protocol, and never treat a raw legacy string as current. Old/unknown peers remain connected only as observers and receive/send no scheduled genes or portfolios. Test mixed versions, missing/malformed provenance, malformed/unknown fields, legacy raw strings, f32-derived versions, hash/order/semantic/provenance mismatch, replay/idempotence, and f64-only threshold bits end to end; only an explicit offline migration may emit a `precision_unrecoverable=true` replacement.

**Step 6: Version and synchronize the f64 GPU ABI**

Bump `ABI_VERSION` and `NEOETHOS_GPU_ABI_VERSION` together from 3 to 4. Convert `DatasetDto.features`, gene weights/thresholds/stop-volatility multipliers, scenario perturbations, `GeneDescriptor` numeric signal fields, native pointer element types, and snapshot/fixture schemas to f64. Update Rust `repr(C)`, C/CUDA structs, FFI signatures, allocation/copy byte counts, and static/runtime size, alignment, offset, element-width, and version assertions as one ABI change. Both the Rust host and every native create/upload/evaluate entry point reject ABI v3 before allocation or launch; no compatibility cast or silent reinterpretation is permitted. If a persisted v3 snapshot has an exact, testable semantic migration, run it only through an explicit offline converter; otherwise reject it with a regenerate diagnostic.

**Step 7: Convert native CUDA and fail-close retiring GPU evaluators**

Native-CUDA buffers, kernel parameters, comparisons, ordered accumulations, reductions, signals, prices used by device trade logic, and readback used by search/backtest must be f64 and follow the same deterministic semantics. Replace f32 CUDA math/intrinsics and byte counts deliberately, including the population transpose/synthesis and standalone first-hit kernel; do not perform a mechanical host-only type edit. Preserve the native CUDA `-fmad=false` correctness control. ABI/layout/signal tests may run without broker replay, but every native trade/cost/PnL/risk kernel remains unreachable behind the already-installed Task 16C capability gate unless the input is an exact synchronized broker fixture; external OHLC must return typed unsupported before launching financial device logic.

Treat backend capability as runtime evidence, not a generic Rust type claim. The locked CubeCL 0.10 source defines `CubeElement for f64`, but its CUDA `register_supported_types` deliberately omits `FloatKind::F64` with a `CUDA_ERROR_INVALID_VALUE` note, and its Metal registration has no f64 type. During the atomic public f64 flip, make the CubeCL shared-feature/search/backtest evaluator and every WGPU/Vulkan/Metal/ROCm entry fail at strict preflight before allocation/launch and add a no-fallback test; change only the host types needed to keep the transitional revision compile-green. Do not port those retiring shared/trading kernels because Task 16A removes their production feature/dependency/config/source surfaces after the captured baseline. Preserve only the exact-named intrinsically-f32 CUDA model-local CubeCL/Candle/tree adapters for Task 16A to classify, inject with `ResolvedDeviceAssignment`, and prove by real launch with no fallback. Native CUDA is the sole supported NVIDIA path for shared f64 indicators/search/backtest and requires the real compile/launch/readback plus end-to-end parity fixture.

**Step 8: Capture focused f64 evidence and the complete cross-crate migration inventory**

Run: `cargo test -p neoethos-search --test f64_feature_lane -- --nocapture`

Run: `cargo test -p neoethos-search --no-default-features --all-targets -- --nocapture`

Run: `cargo test -p neoethos-data --all-targets -- --nocapture`

Run: `cargo test -p neoethos-core --all-targets -- --nocapture`

Run: `cargo test -p neoethos-search --all-targets -- --nocapture`

Run: `cargo test -p neoethos-gpu-contracts --all-targets -- --nocapture`

Run: `cargo test -p neoethos-gpu-cuda --all-targets -- --nocapture`

Expected for only the focused, model-independent commands above: PASS with f64 bits preserved through data/core/search/GPU contracts and no new warning.

Then run and preserve complete compiler output for:

Run: `cargo check -p neoethos-models --all-targets`

Run: `cargo check -p neoethos-trader --all-targets`

Run: `cargo check -p neoethos-app --all-targets`

Run: `cargo check -p neoethos-cli --all-targets`

Run: `cargo check -p neoethos-autoresearch --all-targets`

Run: `cargo check --workspace --all-targets`

Expected: RED only for the direct `FeatureFrame`/`FeatureData`/`ExpertModel`/Polars consumers explicitly scheduled in Tasks 7–9, including bootstrap and genetic constructors. Save and classify every compiler error as the migration inventory. Any unrelated error is repaired before continuing, but neither a filtered test nor a partial crate is called compile-green.

**Step 9: Compile meaningful GPU feature combinations where toolchains permit**

Prepare the standalone `cargo check --all-targets` combinations and run only model-independent slices that can be meaningful at this intermediate boundary. Record the expected model-dependent failures in the same Tasks 7–9 migration inventory. Task 9 reruns every combination after the model migration; CUDA-toolkit/device-required cases remain for Task 18 rather than treating a card-less skip as proof.

**Step 10: Do not commit—continue directly through Tasks 7–9**

Preserve and classify the Task 6A focused evidence, but do not record a revision yet. Its public feature-type flip makes the remaining direct model consumers an expected compile migration inventory. Continue immediately through Tasks 7–9 without adding a shared f32 compatibility accessor; the combined Task 9 matrix is the first permitted GREEN/commit boundary.

**Continuation of the Tasks 6–9 atomicity rule:** `FeatureFrame`/`FeatureData` and `ExpertModel` are crate-wide contracts, so Cargo cannot produce a green filtered test while any implementation is unmigrated. After either public contract changes, intermediate whole-crate compiler failures are an explicit migration inventory, not a passing checkpoint. Do not commit the data f64 flip, foundational model trait, individual families, orchestration, or dependency removal separately; the first revision recorded in history must compile every direct feature consumer and model implementation and pass the full workspace feature/all-target matrix with Polars and shared f32 compatibility removed.

### Task 7: Replace the model DataFrame contract with `FeatureFrame` and typed labels

**Files:**
- Modify: `crates/neoethos-models/src/base.rs`
- Modify: `crates/neoethos-models/src/common.rs`
- Modify: `crates/neoethos-models/src/runtime/prediction.rs` as part of the same Task 6A–9 atomic f64 boundary
- Modify: `crates/neoethos-models/src/parallel_trainer.rs`
- Modify: `crates/neoethos-data/src/core/features.rs`
- Modify: `crates/neoethos-models/tests/model_frame_contract.rs`

**Step 1: Write failing typed-contract tests**

Define expected behavior for named feature selection, row ranges, timestamps, strict finite materialization, typed `&[i32]` labels, train/validation splits, prediction row count, and lease propagation. Assert there is one physical feature backing, not both DataFrame and dense ndarray copies.

**Step 2: Run RED**

Run: `cargo test -p neoethos-models --test model_frame_contract -- --nocapture`

Expected: FAIL against `ExpertModel::{fit,fit_with_validation,predict_proba}` DataFrame/Series signatures, duplicate `TrainingPayload` storage, and common f32 feature assumptions.

**Step 3: Implement the typed model input**

Change expert APIs to consume the f64+validity `FeatureFrame`, typed label slices/validity, explicit row/column views, stable eligible-row indices, and a caller-owned lease. Provide strict f64 materialization only for the explicitly selected rows whose required features/labels are valid; retain original row mapping and reason counts, with no silent drop or imputation. A concrete backend that intrinsically accepts only f32 gets an adapter-local conversion after validity selection that checks finite/range, records `input_dtype=f64` and `backend_dtype=f32` in capability/training metadata, and has a prediction-parity/error-bound test; it may not mutate/cache a common f32 frame or turn invalid into zero. Preserve feature names/order/validity and reject misaligned masks/indices at the boundary.

**Step 4: Remove DataFrame conversion utilities**

Delete `dataframe_to_float32_array`, `strict_numeric_column_values`, `feature_columns_from_dataframe`, and DataFrame construction in `TrainingPayload`. Do not replace them with an Arrow/Polars compatibility DataFrame.

**Step 5: Capture the crate-wide migration inventory**

Run: `cargo check -p neoethos-models --all-targets`

Expected: RED after the global trait change. Save and classify the complete compiler output; it must enumerate every unmigrated implementation/caller scheduled for Tasks 8 and 9. A test-name filter is not used as a false GREEN signal because Rust compiles the whole crate first.

**Step 6: No commit—continue directly to Tasks 8 and 9**

Keep the contract test and implementation changes in the dirty worktree until all model families, orchestration, manifests, and lockfile are migrated and the final Task 9 matrix passes.

### Task 8: Migrate every model family and ensemble adapter off Polars

**Files:**
- Modify: `crates/neoethos-models/src/anomaly/forest_impl.rs`
- Modify: `crates/neoethos-models/src/deep_models.rs`
- Modify: `crates/neoethos-models/src/ensemble.rs`
- Modify: `crates/neoethos-models/src/ensemble_tests.rs`
- Modify: `crates/neoethos-models/src/exit_agent.rs`
- Modify: `crates/neoethos-models/src/exit_agent_tests.rs`
- Modify: `crates/neoethos-models/src/genetic.rs`
- Modify: `crates/neoethos-models/src/forecasting/hmm_regime.rs`
- Modify: `crates/neoethos-models/src/forecasting/swarm_impl.rs`
- Modify: `crates/neoethos-models/src/forecasting/swarm_impl_tests.rs`
- Modify: `crates/neoethos-models/src/evolution/crfmnes_impl.rs`
- Modify: `crates/neoethos-models/src/evolution/neat_impl.rs`
- Modify: `crates/neoethos-models/src/rl/dqn_impl.rs`
- Modify: `crates/neoethos-models/src/rl/dqn_impl_tests.rs`
- Modify: `crates/neoethos-models/src/soft_actor_critic.rs`
- Modify: `crates/neoethos-models/src/soft_actor_critic_tests.rs`
- Modify: `crates/neoethos-models/src/runtime/hpo.rs`
- Modify: `crates/neoethos-models/src/statistical/common.rs`
- Modify: `crates/neoethos-models/src/statistical/bayesian_impl.rs`
- Modify: `crates/neoethos-models/src/statistical/linear_impl.rs`
- Modify: `crates/neoethos-models/src/streaming/adaptive_impl.rs`
- Modify: `crates/neoethos-models/src/tree_models/common.rs`
- Modify: `crates/neoethos-models/src/tree_models/catboost.rs`
- Modify: `crates/neoethos-models/src/tree_models/lightgbm.rs`
- Modify: `crates/neoethos-models/src/tree_models/sklears.rs`
- Modify: `crates/neoethos-models/src/tree_models/xgboost.rs`
- Modify: `crates/neoethos-models/src/ensemble_inference/bootstrap.rs`
- Modify: `crates/neoethos-models/src/ensemble_inference/deep_classification_adapters.rs`
- Modify: `crates/neoethos-models/src/ensemble_inference/deep_timeseries_adapters.rs`
- Modify: `crates/neoethos-models/src/ensemble_inference/evolution_adapters.rs`
- Modify: `crates/neoethos-models/src/ensemble_inference/meta_adapters.rs`
- Modify: `crates/neoethos-models/src/ensemble_inference/mixed_adapters.rs`
- Modify: `crates/neoethos-models/src/ensemble_inference/mod.rs`
- Modify: `crates/neoethos-models/src/ensemble_inference/rl_exit_adapters.rs`
- Modify: `crates/neoethos-models/src/ensemble_inference/soft_voting.rs`
- Modify: `crates/neoethos-models/src/ensemble_inference/swarm_adapter.rs`
- Modify: `crates/neoethos-models/src/ensemble_inference/tree_adapters.rs`
- Modify: `crates/neoethos-models/tests/tree_models_integration.rs`

**Step 1: Migrate one family at a time inside the atomic batch**

For each family, update its tests and implementation to the typed f64 frame/label/lease contract and remove its entries from the saved crate-wide compiler inventory. Preserve existing probability-column mapping, label mapping, feature order, validation routing, artifacts, and metadata. Where upstream kernels/models are intrinsically f32, test and expose the adapter-local narrowing rather than narrowing `FeatureFrame` globally. Focused behavior tests may be prepared now, but do not claim they pass until the complete crate compiles after Task 9.

**Step 2: Pass lease width to native libraries**

Do not read `num_cpus`, legacy Rayon settings, or a global default from model adapters. The exact native thread-semantics enforcement is completed in Task 12; until then adapters accept the lease width explicitly and tests assert propagation.

**Step 3: Prepare the independent feature matrix**

Enumerate CPU default, each standalone optional model family, and meaningful aggregate features with `--all-targets`. Execute the complete matrix only after Task 9 removes the last old orchestrator/Polars callers. Feature unification from the full workspace is not proof that a standalone feature works.

**Step 4: Reach zero family-level migration errors**

Re-run: `cargo check -p neoethos-models --all-targets`

Expected: any remaining RED output is limited to orchestration/dependency removal scheduled in Task 9. If a family implementation, adapter, test, or example remains in the error list, Task 8 is not ready to continue.

**Step 5: No family commits**

Keep every family migration in the same dirty working tree and continue directly to Task 9.

### Task 9: Migrate orchestration and remove Polars from the dependency graph

**Files:**
- Modify: `crates/neoethos-models/src/training_orchestrator.rs`
- Modify: `crates/neoethos-models/src/ensemble_inference/bootstrap.rs`
- Modify: `crates/neoethos-data/src/core/hpc_ta.rs`
- Modify: `crates/neoethos-data/Cargo.toml`
- Modify: `crates/neoethos-models/Cargo.toml`
- Modify: `Cargo.toml`
- Modify: `Cargo.lock`
- Create: `crates/neoethos-data/tests/vortex_only_workspace_guard.rs`

**Step 1: Write failing orchestrator parity tests**

Use the frozen fixture to compare timestamps, selected columns, value/validity bits, explicit warmup/gap/stale/degenerate ineligibility reasons, stable eligible-row indices, label alignment, holdout boundaries, predictions, failed/loaded/skipped model lists, and complete training metadata. Replace DataFrame boolean masks/slices with typed row-index/range plus validity operations; no non-finite/invalid row is silently dropped, zero-filled, imputed, fitted, ranked, or signalled.

**Step 2: Run RED**

Run: `cargo test -p neoethos-models training_orchestrator -- --nocapture`

Expected: FAIL while orchestrator/bootstrap still construct/filter DataFrames.

**Step 3: Implement typed orchestration**

Operate on `FeatureFrame`, row masks/ranges, typed labels, and selected-column views. Ensure bootstrap no longer converts FeatureFrame back into DataFrame. Keep all OOS/purge/holdout ordering identical to the frozen contract.

**Step 4: Remove the last data-crate Polars uses**

Replace any `hpc_ta.rs` or importer helper still using Series/DataFrame with typed arrays. Delete stale Polars comments/settings/wrappers.

**Step 5: Remove Polars manifests and refresh the lockfile**

Delete every direct/workspace `polars` dependency. Refresh only through Cargo; never edit lock entries manually.

**Step 6: Add the workspace guard**

The guard scans production Rust, manifests, lockfile, tests, and examples for Polars declarations/imports/types; `.fstore` readers/writers/extensions; production live-experience JSONL; runtime non-Vortex source branches; compatibility wrappers; old f32-derived artifact/wire versions; hard-coded f32 scheduler element-size assumptions; and shared f32 feature, strategy-weight, threshold, SMC/gate, prop-firm limit/API/evaluation, `RuntimePrediction`/live-decision, GPU DTO/ABI/buffer schemas or narrowing casts. Its required scope includes `neoethos-core` configuration, storage/`SavedStrategy`, prop-firm/risk domains, and scheduler; `neoethos-data`; `neoethos-models/src/runtime/prediction.rs` plus every model-to-runtime adapter; `neoethos-search` including validation, checkpoints, ledgers, DB, portfolios, every `gpu_native` module and example; `neoethos-trader` including `blend_signal`; `neoethos-app` live experience/training/federation/risk API; `neoethos-autoresearch`; mesh migration; `neoethos-gpu-contracts`; `neoethos-gpu-cuda` Rust/C/CUDA sources; and CLI GPU preparation. It allow-lists source-format vocabulary only in exact symbols inside `import_service`/`import_provenance`/`import_discover` and exact app/API/CLI/desktop import adapters/tests, plus the separately named offline legacy-experience migration binary and reporting exports such as validation CSV. `dataset_manifest` may name only the generic `ProducerProvenanceEnvelopeV1` fields and must contain no source-format variants/strings; ordinary runtime may validate its hash binding but never decode it. Adapter allow-list entries may select/submit/report a format but not parse, auto-convert, or store it. Probability/label types are exempt only when they provably cannot affect a live veto, confidence, size, trade, ledger, or promotion decision; every model-local f32 adapter is exact-named. Every allow-list entry names an exact path and symbol plus rationale; broad directory exemptions are forbidden. The guard also rejects bare federation `Vec<Gene>`, opaque mesh migrant JSON, untyped/unversioned `SavedStrategy` numerics, and unversioned/legacy precision artifacts accepted as current f64.

Define scan roots explicitly: root manifests/config/toolchain, `crates/**`, `mcp/**`, `mesh/**`, `desktop/**`, `.github/**`, `scripts/**`, production examples/tests/fixtures, and shipped `README.md`/`BUILDING.md`/help. Exclude `.git`, `target`, caches, generated temporary evidence, and captured logs. Removed/legacy vocabulary is documentary-only in immutable `docs/audits/**` plus the exact approved design and implementation-plan files; no other docs/help exemption exists. Guard-fixture literals are allow-listed by exact test path and symbol, never a whole test/docs tree.

The same guard rejects any pending/current production live-learning feature row serialized outside a versioned Vortex generation, including feature values embedded in a lifecycle journal, SQLite fields, `.fstore`, JSON, or bespoke binary. Only typed lifecycle state/IDs and immutable Vortex generation/hash references may appear in the control journal.

**Step 7: Verify graph removal and the now-complete model migration**

Run: `cargo test -p neoethos-data --test vortex_only_workspace_guard -- --nocapture`

Run: `cargo tree --workspace | rg -i "polars|fstore"`

Expected: the guard passes and the search returns no dependency/output match.

Run: `cargo test -p neoethos-models --all-targets -- --nocapture`

Expected: PASS against the original frozen contract.

Every root/model/search/app/GPU financial-path test in this full matrix must either carry an exact synchronized broker capability or assert typed unsupported before arithmetic. A formerly heuristic fixture is not grandfathered merely because this is a migration regression run.

Run: `cargo test -p neoethos-data --all-targets -- --nocapture`

Run: `cargo test -p neoethos-core --all-targets -- --nocapture`

Run: `cargo test -p neoethos-search --no-default-features --all-targets -- --nocapture`

Run: `cargo test -p neoethos-search --all-targets -- --nocapture`

Run: `cargo test -p neoethos-trader --all-targets -- --nocapture`

Run: `cargo test -p neoethos-app --all-targets -- --nocapture`

Run: `cargo test -p neoethos-cli --all-targets -- --nocapture`

Run: `cargo test -p neoethos-autoresearch --all-targets -- --nocapture`

Run: `cargo test -p neoethos-codex --all-targets -- --nocapture`

Run: `cargo test -p neoethos-mcp --all-targets -- --nocapture`

Run: `cargo test -p neoethos-desktop --all-targets -- --nocapture`

Run: `cargo test -p neoethos-gpu-contracts --all-targets -- --nocapture`

Run: `cargo test -p neoethos-gpu-cuda --all-targets -- --nocapture`

Run: `cargo test --manifest-path mesh/Cargo.toml --all-targets -- --nocapture`

Run every standalone and aggregate model/GPU feature command prepared in Tasks 6A and 8 with `--all-targets`, then run `cargo check --workspace --all-targets`. Every command must PASS compile-green before any Task 6–9 change is committed, with f64 bits preserved through core config, shared CPU/search/trader/live-experience/autoresearch/federation/mesh/GPU-contract paths, every previous artifact/wire schema rejected, `.fstore` and Polars removed, and no new warning. Preserve and inspect complete INFO/WARN/ERROR output rather than only exit status.

**Step 8: Re-measure build/runtime baselines**

Repeat Task 1 wall-time, peak-RSS, Vortex scan, projected feature-window, GA access, and training measurements. Report deltas honestly and profile any material regression.

**Step 9: Commit Tasks 6–9 together**

```text
refactor: migrate f64 Vortex features and remove Polars
```

---

## Milestone 4 — Install one budget before every runtime and CPU workload

### Task 10: Replace legacy CPU resolution/config mutation with typed inputs

**Files:**
- Modify: `crates/neoethos-core/src/config.rs`
- Modify: `crates/neoethos-core/src/system.rs`
- Modify: `crates/neoethos-core/src/resolved_config.rs`
- Modify: `crates/neoethos-core/src/lib.rs`
- Modify: `crates/neoethos-core/tests/hardware_derived_not_settable.rs`
- Modify: `crates/neoethos-core/tests/config_single_load_path.rs`
- Modify: `crates/neoethos-core/tests/shipped_config_matches_defaults.rs`
- Modify: `crates/neoethos-app/src/server/knob_catalog.rs`
- Create: `crates/neoethos-app/tests/cpu_budget_knob_catalog.rs`
- Modify: `config.yaml`
- Modify: `desktop/src-tauri/resources/config.yaml`

**Step 1: Write failing config/resolution tests**

Change 12-thread automatic expectations from 11 to 10, add `9999 -> 10`, reject zero for canonical and legacy keys with the exact key in the error, prove cap precedence by minimum, and prove parent assignment does not mutate persistent settings. Add an app API/knob-catalog RED contract proving `models.backtest_runtime.rayon_threads` is no longer exposed, `system.hardware.cpu_budget` is the only persistent CPU-width knob, and its auto help/value says effective logical threads minus the fixed two-thread reserve rather than all cores or physical-core pinning. Add a 64-effective-logical-thread fixture: automatic process capacity is 62, and sufficiently parallel DataIngestion or Inference demand may request all 62 rather than silently clamping at 8 or 16; no UI `2` constant is a second capacity authority. Small workloads request only their proven useful units and concurrent leases still sum to at most 62.

**Step 2: Run RED**

Run: `cargo test -p neoethos-core --test hardware_derived_not_settable -- --nocapture`

Run: `cargo test -p neoethos-app --test cpu_budget_knob_catalog -- --nocapture`

Expected: FAIL against `resolve_cpu_budget` and `apply_process_cpu_assignment`.

**Step 3: Implement `ExecutionBudgetInputs::from_settings_and_parent`**

Remove `Settings::apply_process_cpu_assignment`. Retain legacy `models.backtest_runtime.rayon_threads` for one read-only compatibility window, emit one structured WARN naming `system.hardware.cpu_budget`, omit the legacy key on save, and expose only the canonical knob. Delete the retired knob-catalog entry and its “all logical cores”/physical-pinning guidance; the canonical entry derives its displayed automatic value and help from the same installed resolver/provenance used by runtime, not a second core-count formula.

**Step 4: Separate host inventory, process capacity, and per-job demand**

Keep serialized `HardwareProfile.cpu_cores` as legacy inventory only. Use `available_parallelism()`/installed budget for process capacity. Remove the current DataIngestion `8`, Inference `16`, and UI `2` hard clamps as capacity authorities. Each CPU-heavy job declares a `WorkloadDemand` derived from actual parallelizable shards/batches/model/backend limits; admission grants at most the installed budget and may use every otherwise-idle permit when the work can saturate it. Lightweight/I/O UI control work is not assigned a magic CPU pool, and any real CPU-heavy UI work enters the same broker. QoS comes from queue priority/reservations, not hidden worker ceilings. Diagnostics separately name host logical-thread inventory, installed capacity, requested demand, granted workers, and coordination scope.

**Step 5: Verify settings round trips**

Run: `cargo test -p neoethos-core config system -- --nocapture`

Run: `cargo test -p neoethos-app --test cpu_budget_knob_catalog -- --nocapture`

Expected: PASS; old settings load with a warning and save without the retired key.

**Step 6: Commit**

```text
fix: resolve CPU limits from effective process capacity
```

### Task 11: Add a budgeted Rayon executor and async admission coordinator

**Files:**
- Create: `crates/neoethos-core/src/execution.rs`
- Create: `crates/neoethos-app/src/app_services/execution_admission.rs`
- Create: `crates/neoethos-app/tests/execution_admission.rs`
- Modify: `crates/neoethos-core/src/lib.rs`
- Modify: `crates/neoethos-core/Cargo.toml`
- Modify: `crates/neoethos-app/src/app_services/mod.rs`
- Modify: `crates/neoethos-app/src/app_state.rs`

**Step 1: Write failing executor tests**

Prove a Rayon pool can execute only while holding a matching lease, cached idle threads are not counted as admitted work, nested jobs split rather than reacquire, panic returns permits, and concurrent admitted widths never exceed the budget.

**Step 2: Implement `BudgetedCpuExecutor`**

Build/select a pool exactly matching the transferred lease width and retain the lease through scoped completion. Do not install a second global authority.

**Step 3: Write failing async saturation tests**

With a one-worker budget held, first queue an opportunistic local job and then a higher-priority child reservation, exercise heartbeat and cancellation before/after admission, and return the held lease. Assert no Tokio core or coordinator blocks on a request-specific `Condvar`, the later child starts before the earlier local waiter, FIFO is preserved inside each priority, a cancelled head cannot stall the queue, and stop/child-exit cleanup returns every permit.

**Step 4: Implement the dedicated coordinator**

One OS coordinator thread owns the complete priority/FIFO request queue and answers Tokio oneshots. It never performs a blocking acquire for one request: it drains enqueue/cancel/shutdown messages, uses broker `try_acquire` for the highest-priority live request, and retries on explicit lease-return/channel wakeups. Alternatively the leaf broker may own the same enqueue/cancel priority queue, but all requests must be visible and reprioritizable before any wait. Dropped/cancelled requests are removed or marked without requiring a permit; a later child can overtake an earlier opportunistic local waiter. The app state owns shutdown and joins the coordinator cleanly.

**Step 5: Verify**

Run: `cargo test -p neoethos-app --test execution_admission -- --nocapture`

Expected: PASS, including saturation/cancellation/child-exit cases.

**Step 6: Commit**

```text
feat: coordinate CPU admission without blocking async runtimes
```

### Task 12: Budget the remaining search, model, native, fallback, and feeder work

**Files:**
- Modify: `crates/neoethos-core/src/execution/backend_inventory.rs` created in Task 4A
- Modify: `crates/neoethos-core/tests/backend_threading_inventory.rs` created in Task 4A
- Modify: `crates/neoethos-search/src/eval.rs`
- Modify: `crates/neoethos-search/src/execution_profile.rs`
- Modify: `crates/neoethos-search/src/backend.rs`
- Modify: `crates/neoethos-search/src/gpu_fallback.rs`
- Modify: `crates/neoethos-search/src/cubecl_eval.rs`
- Modify: `crates/neoethos-search/src/genetic/search_engine.rs`
- Modify: `crates/neoethos-models/src/parallel_trainer.rs`
- Modify: `crates/neoethos-models/src/hardware.rs`
- Modify: `crates/neoethos-models/src/tree_models/config.rs`
- Modify: native model adapters listed in Task 8
- Modify: Vortex feature scan/cache call sites from Tasks 3–6; importer call sites were already completed in Task 4A
- Modify: `crates/neoethos-data/src/lib.rs`
- Modify: `crates/neoethos-data/src/core/hpc_ta.rs`
- Modify: `crates/neoethos-data/src/core/gpu_indicators.rs`
- Modify: `vendor/vector-ta-0.2.9-patched/src/cuda/module_loader.rs`
- Modify: `vendor/vector-ta-0.2.9-patched/src/cuda/runtime.rs`
- Modify: `vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs`
- Modify: `vendor/vector-ta-0.2.9-patched/Cargo.toml`
- Modify: `crates/neoethos-gpu-cuda/src/lib.rs`
- Modify: `crates/neoethos-gpu-contracts/src/lib.rs`
- Modify: `crates/neoethos-gpu-contracts/Cargo.toml`
- Modify: `Cargo.lock`
- Create: `crates/neoethos-data/tests/cpu_budget_feature_construction.rs`
- Create: `crates/neoethos-data/tests/cuda_module_admission.rs`
- Create: `crates/neoethos-data/tests/cuda_module_load_evidence.rs`
- Create: `crates/neoethos-data/tests/cuda_common_loader_reachability.rs`

**Step 1: Research and record dependency thread semantics**

Extend the Task 4A inventory using current official documentation/local locked source. Keep its existing Vortex, Arrow CSV/JSON/IPC, Parquet/codec, and hashing classifications unchanged unless new evidence and their focused tests justify a revision. Classify Vortex feature scans, the production `neoethos-data` nested `rayon::join`/`par_iter`/`into_par_iter` feature lane, ndarray-Rayon, DuckDB, Burn CPU, CubeCL CPU fallback while it still exists, BLAS/OpenMP, XGBoost, LightGBM, CatBoost, sklears, and CUDA host phases as `lease_native_width`, `exclusive_global_pool`, `single_thread_under_partial_lease`, `device_only`, or `unsupported_concurrent`. CUDA context creation, SASS/fatbin module load, PTX parse/JIT/link/driver-cache compilation, and concurrent per-card initialization each get a separate row; active driver/JIT work is never exempted merely because a later kernel runs on device. Record whether the caller/helper/driver threads count toward each library's parameter. Do not infer semantics from memory. Reuse the executable Task 4A meanings unchanged: exact-width calls hold that width, an exclusive global pool reserves every process permit across backend types, a proven single-thread call holds one, device-only is only proven host-noncomputing device launch/wait and excludes feeder/post-processing/init/JIT CPU work, and unsupported/unclassified paths fail before initialization/allocation/work.

**Step 2: Write failing classification/startup tests**

Every enabled CPU/backend host phase must have one classification. Requesting an unclassified or unbounded concurrent backend must fail before work starts. For every class, test configured width versus observed active workers, permit lifetime, cancellation/panic return, overlap with a different backend, full-budget serialization for global pools, one-permit accounting for proven single-thread paths, and fail-before-side-effect behavior for unsupported/unclassified paths. `cuda_module_admission` uses cold/warm driver-cache fixtures, forced SASS/fatbin and PTX/JIT routes, cancellation, failure cleanup, and simultaneous per-card initialization; it proves no active host/JIT worker runs under a `device_only` label. Define `CudaBuildManifestIdV1([u8; 32])` and the versioned `ModuleLoadEvidenceV1` contract in `neoethos-gpu-contracts`: exact build-manifest ID, stable CUDA UUID/PCI identity and session ordinal, compute capability, driver/runtime/toolkit, logical module stem and exact artifact hash/size/container, requested and actual SASS/PTX/JIT/cache route, load duration/error, and the subsequent launch/readback identity. Task 15 later defines the manifest whose canonical SHA-256 produces this already-versioned ID; it does not revise the load-evidence schema. Evidence is collected in release builds even when debug logging is off. The custom vector-ta loader consumes this single zero/small shared contract through an explicit path dependency; it may not handwrite a second wire encoder.

**Step 3: Replace independent Rayon/native widths**

Remove search `None => Rayon default`, model-owned free-standing pools, `num_cpus` fallbacks, and direct legacy setting reads. Make `compute_hpc_feature_frame*` and every production feature-construction entry receive the caller's existing lease/executor context. Route its nested `rayon::join`, `par_iter`, and `into_par_iter` through `BudgetedCpuExecutor` at that lease width; child branches split/work-steal within the same admitted pool and never acquire another lease or use Rayon's global pool. Split the parent lease across concurrent models; pass each child width into native adapters according to documented semantics.

**Step 4: Budget GPU feeders/fallbacks without weakening GPU truth**

Pre/post-processing and CPU fallbacks use leases. CUDA context/module initialization and PTX/JIT/cache work acquire their classified lease before touching the driver and hold it through completion/cancellation/cleanup. Device kernels do not consume CPU permits only while the host/driver threads are proven to be passively waiting. An unbounded PTX/JIT route is unsupported for concurrent/strict production rather than hidden behind `device_only`. Replace every production-reachable direct `Module::from_ptx` call with one common evidence-producing loader; the reachability guard fails on a bypass. Mutually exclusive force-SASS and force-PTX controls make both routes testable. A fatbin success is not labelled SASS unless PTX is disabled and image inspection proves a matching SASS image; PTX/JIT evidence requires a controlled cold-cache route. Clear stale prior failures after a verified success. Missing/zero-byte/unknown-prebuilt artifacts, contradictory force controls, and an unvalidated fallback fail before launch. A requested strict GPU path still runs a real kernel or fails loudly; CPU-budget code may not silently route it to CPU or an unpromoted PTX path.

**Step 5: Add overlap probes**

Overlap an already Task-4A-admitted non-Vortex import through Vortex publish, Vortex projected scan, nested `neoethos-data` feature construction, search, model training, Tokio CPU work, GPU feeder/fallback mocks, cold SASS module load, cold PTX/JIT/cache compilation when supported, concurrent per-card initialization, and managed-child reservations. Assert both admitted width and measured active dependency/driver/JIT workers stay at/below the installed limit and a test hook observes zero feature tasks on Rayon's global pool; this is integration coverage, not the first importer-admission implementation.

**Step 6: Verify focused crates**

Run: `cargo test -p neoethos-core --test backend_threading_inventory -- --nocapture`

Run: `cargo test -p neoethos-search execution_profile eval -- --nocapture`

Run: `cargo test -p neoethos-models parallel_trainer tree_models -- --nocapture`

Run: `cargo test -p neoethos-data --test cuda_module_admission --test cuda_module_load_evidence --test cuda_common_loader_reachability -- --nocapture`

Run: `cargo test -p neoethos-data --test cuda_module_admission -- --nocapture`

Run: `cargo test -p neoethos-data --test cpu_budget_feature_construction -- --nocapture`

Expected: PASS with no unmanaged worker source.

**Step 7: Commit**

```text
fix: route CPU workloads through unified permits
```

### Task 13: Install the budget before every Tokio/Tauri runtime

**Files:**
- Create: `crates/neoethos-app/tests/startup_cpu_budget.rs`
- Create: `crates/neoethos-app/tests/migrate_live_experience_startup.rs` as part of atomic Task 6A when that binary is introduced
- Create: `crates/neoethos-data/tests/migrate_legacy_dataset_layout_startup.rs` as part of Task 3 when that binary is introduced
- Create: `crates/neoethos-cli/tests/startup_cpu_budget.rs`
- Create: `crates/neoethos-mcp/tests/startup_cpu_budget.rs`
- Create: `mcp/tests/startup_cpu_budget.rs`
- Create: `mesh/tests/startup_cpu_budget.rs`
- Create: `desktop/src-tauri/tests/startup_cpu_budget.rs`
- Modify: `crates/neoethos-app/src/main.rs`
- Modify: `crates/neoethos-app/src/bin/migrate_live_experience.rs` as part of atomic Task 6A
- Modify: `crates/neoethos-data/src/bin/migrate_legacy_dataset_layout.rs` as part of Task 3
- Modify: `crates/neoethos-cli/src/main.rs`
- Modify: `crates/neoethos-mcp/src/main.rs`
- Modify: `crates/neoethos-mcp/src/backend.rs`
- Modify: `crates/neoethos-mcp/src/server.rs`
- Create: `crates/neoethos-mcp/tests/cpu_payload_budget.rs`
- Modify: `desktop/src-tauri/src/lib.rs`
- Modify: `desktop/src-tauri/src/main.rs`
- Modify: `mcp/src/main.rs`
- Modify: `mcp/src/lib.rs`
- Create: `mcp/tests/cpu_payload_budget.rs`
- Modify: `mesh/src/main.rs`
- Create: `mesh/tests/cpu_payload_budget.rs`

**Step 1: Write failing startup-order tests**

Capture startup events and assert config seed/load, parent-cap parse, budget resolve/install, Linux import-signal preflight, hardware/search/model/data settings install, Tokio builder when applicable, and application builder occur in that order. Extend `crates/neoethos-app/tests/startup_cpu_budget.rs` and `crates/neoethos-cli/tests/startup_cpu_budget.rs` with a Linux-only trace proving their initial synchronous thread blocks fallback `SIGIO` plus the reserved real-time set and initializes/verifies `SourceSealCoordinator`/`signalfd` (or records typed path-import unsupported) before any Tokio/Tauri/Rayon/other worker exists. After the multithreaded runtime starts, lower/exhaust `RLIMIT_SIGPENDING`, break active leases, and prove fallback `SIGIO` is consumed only by the coordinator, cancels every active seal, publishes none, drains, and recovers. Assert all diagnostic fields and coordination scope for each executable, including the synchronous `migrate_live_experience` and `migrate_legacy_dataset_layout` binaries when their atomic tasks create them.

**Step 2: Replace `#[tokio::main]` entry points**

Use small synchronous mains that do preflight then build a multi-thread Tokio runtime with the resolved effective worker limit. On Linux, `crates/neoethos-app/src/main.rs` and `crates/neoethos-cli/src/main.rs` perform the import-signal preflight on that initial thread before runtime construction; lazy first-import initialization is forbidden because signal masks are per-thread. Zero/malformed `--cpu-threads` fails before runtime creation. Both offline migrators stay synchronous: each performs preflight before reader/codec/backend initialization, acquires one top-level lease with the initial workload class/width, holds it through bounded read/validation/Vortex verify/CAS publish/failure cleanup, and creates neither a hidden app coordinator nor a Tokio runtime.

**Step 3: Fix Tauri ordering**

Desktop synchronously seeds/loads config, installs the budget, builds/retains a Tokio runtime, calls `tauri::async_runtime::set` exactly once, and only then calls `tauri::Builder::default`. Test packaged first-run seed identity and runtime lifetime.

**Step 4: Classify every blocking and inline CPU-heavy async task**

Replace raw production `spawn_blocking`/`tauri::async_runtime::spawn_blocking` in the following files with the CPU-admitted wrapper or an explicitly named I/O-blocking wrapper:

- `crates/neoethos-app/src/main.rs`
- `crates/neoethos-app/src/server/{autonomous,bridge,broker_control,chart,data_control,diagnostics,engines_control,hardware,indicators,intelligence,mod,orders,pending_actions,strategy_lab,system_status,watchlist}.rs`
- `crates/neoethos-app/src/app_services/{discovery,live_spots_streamer,live_trading,supervisor,training}.rs`
- `desktop/src-tauri/src/{broker,lib}.rs`

Add a source guard rejecting new raw calls outside the wrappers. Also inventory synchronous parsing/serialization/validation/hashing/compression and potentially large loops executed directly inside async handlers/tasks in root MCP, top-level MCP, mesh, app, and desktop; the guard cannot key only on `spawn_blocking`. Broker/network/file socket wait may use the I/O wrapper without a CPU lease only under hard payload/decompressed-output/depth/count limits and a measured short-control threshold. JSON/CBOR/tool/gene/message encode/decode, HTTP/TLS/decompression CPU work, feature computation, parsing, hashing, chart construction, discovery, training, and evaluation above that threshold must obtain the process-local async admission before moving off the Tokio core worker, and hold the lease through validation/serialization/failure cleanup. Add one-worker large tool-list/argument/result and mesh-migrant saturation fixtures proving heartbeat, cancellation, and permit return.

**Step 5: Verify startup and responsiveness**

Run the focused startup tests in root, MCP, mesh, desktop, and both offline migrators. On the local 12-logical-thread host, capture the automatic worker limit `10`.

Run: `cargo test -p neoethos-app --test startup_cpu_budget --test migrate_live_experience_startup -- --nocapture`

Run: `cargo test -p neoethos-data --test migrate_legacy_dataset_layout_startup -- --nocapture`

Run: `cargo test -p neoethos-mcp --test cpu_payload_budget -- --nocapture`

Run: `cargo test --manifest-path mcp/Cargo.toml --test cpu_payload_budget -- --nocapture`

Run: `cargo test --manifest-path mesh/Cargo.toml --test cpu_payload_budget -- --nocapture`

Run: `cargo test -p neoethos-cli --test startup_cpu_budget -- --nocapture`

Run: `cargo test -p neoethos-mcp --test startup_cpu_budget -- --nocapture`

Run: `cargo test --manifest-path mcp/Cargo.toml --test startup_cpu_budget -- --nocapture`

Run: `cargo test --manifest-path mesh/Cargo.toml --test startup_cpu_budget -- --nocapture`

Run in `desktop/src-tauri`: `cargo test --test startup_cpu_budget -- --nocapture`

Each separate-workspace RED test must first observe `#[tokio::main]` creating the runtime before cap/install; GREEN proves cap parse, budget resolve/install, and complete structured startup fields plus `coordination_scope` occur before any Tokio/global/backend initialization.

**Step 6: Commit**

```text
fix: install CPU budget before async runtimes
```

### Task 14: Give scheduler children fixed lifetime reservations

**Files:**
- Modify: `crates/neoethos-core/src/scheduler.rs`
- Modify: `crates/neoethos-cli/src/main.rs`
- Modify: `crates/neoethos-cli/src/tui/jobs.rs`
- Modify: `crates/neoethos-app/src/server/federation.rs`
- Create: `crates/neoethos-app/tests/federation_capacity_contract.rs`
- Modify: `desktop/src/api.ts`
- Modify: `desktop/src/screens/Advanced.tsx`
- Create: `desktop/src/screens/Advanced.test.tsx`
- Modify: `desktop/src-tauri/src/lib.rs`
- Modify: `mcp/src/lib.rs`
- Modify: `mcp/src/main.rs`
- Create: `mcp/tests/managed_stdio_budget.rs`
- Modify: `mesh/src/main.rs`
- Create: `mesh/tests/mixed_version_capacity.rs`

**Step 1: Write failing fixed-slot tests**

For every admission/completion/error/cancellation/crash order, prove deterministic slot widths sum to the parent budget, running children never resize, queued child priority prevents starvation, and the lifetime lease returns on every exit. Before implementation, make `mcp/tests/managed_stdio_budget.rs` RED for composite desktop reservation, exact local subdivision, multiple simultaneous declared stdio widths, excess queueing, spawn failure, duplicate replacement, cancel/crash/wait/shutdown, unbounded command rejection, independent-MCP process-local mode, and no double reservation.

**Step 2: Implement fixed slot partitioning**

At scheduler construction compute fixed widths from parent limit and allowed simultaneous children. `RunningItem` owns the lease and passes its exact width via `--cpu-threads`. Remove dynamic `cpu_cores / active_children` recomputation.

**Step 3: Reserve desktop sidecars**

Launch managed mesh through a one-permit lifetime reservation. Resolve one validated desktop `mcp_subtree_cpu_threads` composite width (default one only with no enabled stdio servers), acquire it atomically while holding no partial slot, hold that original parent reservation for the MCP lifetime, and pass it via `--cpu-threads` plus managed-tree scope. Every child computes `local_effective = min(parent_assignment, child_automatic_limit)` from its own pre-runtime `available_parallelism() - 2` resolver; it reports unused reserved width while the desktop continues holding the full original reservation. If the complete sidecar/subtree reservation is unavailable, defer with structured INFO rather than starting an unbudgeted child.

Treat each stdio server spawned by top-level MCP as a managed grandchild. Extend `ServerCfg` with a validated CPU capability/class and declared width. A managed MCP installs a local broker whose total is its `local_effective`, reserves the declared sidecar/control width, and subdivides only the remainder; excess launches queue with child priority. It does not perform another parent/process-tree acquisition. Each stdio child receives a fixed assignment from that local broker and independently re-clamps it with the same minimum rule before runtime/backend initialization. Independently launched MCP resolves one process-local `available_parallelism - 2` budget and then uses the same subdivision. Store each local lifetime lease in `RunningService` through successful spawn, duplicate replacement, cancellation, wait error, crash cleanup, and shutdown join. Propagate fixed args/environment only through a documented server contract. Reject an unbounded/uncooperative command before spawn unless proven OS containment enforces its width. For remote HTTP, only bounded socket wait and short orchestration are I/O-only; local tool-list/argument/result JSON traversal, encode/decode/validation, TLS, and decompression obey Task 13 limits/classification and use the MCP local admission broker when CPU-heavy.

Test multiple simultaneous stdio children, spawn failure, duplicate replacement, cancel, crash, shutdown join, an uncooperative/unbounded command, remote HTTP large/decompression-limited results, and live reservation sums across desktop -> MCP -> stdio grandchildren. Include parent assignment eight with child effective availability three for managed MCP, mesh, and stdio: each local broker/child uses at most three, reports the unused parent-reserved five, and neither double-reserves nor lends them. Every failure releases exactly its own permits, queued child priority remains intact, and the process-tree reserved sum never exceeds the desktop parent budget. Re-run the one-worker root-MCP/top-level-MCP/mesh large-payload heartbeat and cancellation probes under the composite reservation.

**Step 4: Fix mesh wire semantics**

Keep legacy `cpu_cores` as host inventory alias only. Add `host_logical_threads`, `effective_worker_limit`, capacity protocol version, and `capacity_state`. Old peers without a positive effective limit are observer-only and unschedulable; never invent one worker or sum legacy cores. Define capacity protocol v2 as the compatibility window: it accepts legacy `cpu_cores` for inventory display and emits the alias only before `2026-12-31T00:00:00Z`. Capacity protocol v3, and every v2 process at/after that deadline, stops emitting the alias while still decoding it as observer-only inventory; scheduling always requires the explicit current effective-worker field. Inject the clock in tests and document the retirement constant so the window cannot become permanent.

Replace the mesh/status API's ambiguous `totalCores` and `self.cores` with typed `totalHostLogicalThreads`, `totalSchedulableWorkers`, `unknownCapacityNodes`, and per-node `hostLogicalThreads`, `effectiveWorkerLimit`, and `capacityState`. Only current, positive, schedulable announcements contribute to `totalSchedulableWorkers`; legacy/unknown observers may contribute to the separately labelled inventory total but never capacity. Update the desktop API and Advanced screen to display “Reported host logical threads (inventory)” separately from “Schedulable workers” and visibly count unknown/observer nodes.

**Step 5: Test cgroup/old-profile behavior**

Load a profile with `cpu_cores=96`, inject live effective `23`, and assert assignments derive from `21`. Mixed-version mesh fixtures must deserialize but never receive work until upgraded capacity is announced. Test v1/v2/v3 announcements immediately before and at the retirement timestamp, prove the legacy alias emission stops exactly at the deadline, and prove API/UI totals and labels never count an observer's inventory as schedulable workers.

**Step 6: Verify**

Run: `cargo test -p neoethos-core scheduler -- --nocapture`

Run: `cargo test --manifest-path mesh/Cargo.toml --all-targets -- --nocapture`

Run: `cargo test --manifest-path mcp/Cargo.toml --test managed_stdio_budget -- --nocapture`

Expected: PASS and live reservations never exceed the parent.

**Step 7: Commit**

```text
fix: reserve fixed CPU slots for managed children
```

### Task 15: Make repository build parallelism and packaging prerequisites portable

**Files:**
- Modify: `.cargo/config.toml`
- Modify: `rust-toolchain.toml`
- Modify: `Cargo.toml`
- Modify: `.github/workflows/ci.yml`
- Modify: `.github/workflows/release.yml`
- Modify: `.github/workflows/release-desktop.yml`
- Modify: `BUILDING.md`
- Create: `scripts/package-desktop.ps1`
- Create: `scripts/package-desktop.sh`
- Modify: `desktop/src-tauri/tauri.windows.conf.json`
- Modify: `desktop/src-tauri/tauri.linux.conf.json`
- Modify: `scripts/card_run.sh`
- Modify: `scripts/gpu-bench/remote_bootstrap.sh`
- Create: `scripts/build/resolve_host.rs`
- Create: `scripts/build-host.ps1`
- Create: `scripts/build-host.sh`
- Create: `scripts/gpu-bench/test_build_host_plan.py`
- Modify: every other script that sets fixed Cargo/rustc/native thread counts
- Modify: `vendor/vector-ta-0.2.9-patched/build.rs`
- Modify: `vendor/vector-ta-0.2.9-patched/Cargo.toml`
- Modify: `crates/neoethos-gpu-cuda/build.rs`
- Modify: `crates/neoethos-gpu-cuda/Cargo.toml`
- Create: `scripts/gpu-bench/test_cuda_builder_jobserver.py`
- Create: `scripts/gpu-bench/test_cuda_build_contract.py`
- Create: `scripts/gpu-bench/test_cuda_artifact_inventory.py`
- Create: `scripts/gpu-bench/test_cuda_builder_fingerprint.py`
- Create: `docs/audits/2026-08-15-cuda-artifact-inventory.md`
- Modify: `crates/neoethos-gpu-contracts/src/lib.rs`
- Modify: `crates/neoethos-gpu-contracts/Cargo.toml`
- Create: `crates/neoethos-gpu-contracts/tests/cuda_build_manifest.rs`
- Modify: `Cargo.lock`
- Modify: exact retained CubeCL/Candle/native-tree CUDA build, packaging, and runtime artifact adapters selected by Task 16A
- Create: `docs/audits/2026-08-15-simd-dispatch-inventory.md`
- Create: `crates/neoethos-data/tests/simd_dispatch_parity.rs`
- Create: `crates/neoethos-search/tests/simd_dispatch_end_to_end.rs`
- Modify: every production SIMD dispatch site selected by the source/compiler-command inventory
- Modify if probes expose an escape: `vendor/xgboost_lib-sys/build.rs`
- Modify if probes expose an escape: `vendor/lightgbm3-sys/build.rs`

**Step 1: Add repository search assertions**

Search root, CI, MCP, mesh, scripts, manifests, build scripts, and reachable native-builder configuration for `-Zthreads`, `threads=8`, fixed large `-j`, `JOBS=8`, `CARGO_BUILD_JOBS`, `NUM_JOBS`, CMake/Ninja parallel flags, and stale explanations. Record every intentional external override separately. Replace all fixed `scripts/card_run.sh` job widths with inherited Cargo jobserver capacity or `available_parallelism - 2` (minimum one); do not preserve its current fixed 8 at any phase.

Add a whole-repository/compiler-command guard for reachable `target-cpu=native`, `target-cpu=x86-64-v3`, `-march=`, `-mtune=`, `/arch:AVX*`, AVX2/BMI2/FMA enablement, and injected `RUSTFLAGS`/`CFLAGS`/`CXXFLAGS`. Classify dormant vendored upstream benchmark examples separately from any flag reachable in a shipped build. The current `scripts/gpu-bench/remote_bootstrap.sh` default `RUSTFLAGS=-C target-cpu=x86-64-v3` is removed; the helper may request only the portable baseline, while profiled SIMD functions use runtime dispatch.

**Step 2: Configure Cargo jobs**

Add repository-default `[build] jobs = -2`. Remove fixed `-Z threads=8`, update `rust-toolchain.toml` so its pinned-nightly rationale no longer claims `-Zthreads` is required, and remove every reachable repository-wide `target-cpu=x86-64-v3`/native instruction requirement from Windows/Linux rustflags, CI, deployment helpers, and native compiler flags. Ship a portable x86-64 baseline that runs on the supported AMD x86-64 population instead of assuming Zen/AVX2/BMI2/FMA; performance-sensitive SIMD may use explicit runtime feature detection and a tested dispatched implementation, never a binary-wide instruction requirement tied to the build/current host. Inventory every reachable runtime-dispatched branch with a stable branch ID and test-only force selector. Compare portable scalar with every enabled AVX2/BMI2/FMA branch over constant, edge, NaN/gap, high-dynamic-range, threshold-adjacent, alignment, and reduction fixtures: validity and f64 features/signals must match bit-for-bit. Compare ordered trades/ledger/metrics only on an exact synchronized broker-capable fixture; otherwise every branch must stop at the same typed unsupported gate. Disable/fail-close any branch that uses contraction, fast math, or operation/reduction ordering that cannot satisfy the semantic contract. Execute the clean baseline artifact on an x86-64 CPU lacking v3 features, not only compile it there, and repeat on AMD Zen/v3; dispatched optimized functions have separate feature-detection/illegal-instruction regressions plus selected-branch telemetry.

Create one synchronous `BuildHostPlanV1` resolver and make every repository build/packaging/VPS entry point call it before Cargo starts. In host-auto mode it reads `std::thread::available_parallelism()` once, computes `max(1, available - 2)` once, queries the current visible NVIDIA inventory once, retains UUID/PCI/name/compute-capability evidence for every eligible discrete card, and canonical-sorts/deduplicates the complete architecture set. It exports the worker limit for Cargo/native scheduling and the only CUDA architecture input, `NEOETHOS_CUDA_ARCHS=<semicolon-separated numeric set>`, for every custom/third-party CUDA builder. It writes and hashes one sealed plan consumed by build scripts and embedded in generated runtime/artifact metadata; no builder reruns device selection or accepts a different architecture. CPU-only mode performs no NVIDIA query. If CUDA features are requested and host-auto cannot resolve an eligible card, fail before a compiler child starts. An explicitly supplied reviewed cross/release matrix remains possible, but it is a separate typed resolution mode and provenance field, never an automatic-host fallback. Add RED/GREEN tests for no GPU, 3090/sm86, 4090/sm89, reordered/duplicate inventory, mixed sm86+sm89, cgroup-limited CPU visibility, a conflicting builder environment, and emitted runtime metadata. Remove the superseded `CUDA_ARCH`, `CUDA_ARCHS`, singular `NEOETHOS_CUDA_ARCH`, per-builder defaults/probes, fixed card names, first-device selection, and duplicate host-probe helpers after the shared plan is connected.

**Step 3: Verify Cargo and representative native job behavior**

Capture `cargo -vv check -p neoethos-execution-budget` and confirm Cargo uses the negative-job default/jobserver without a multiplied eight-thread rustc frontend. Then perform clean verbose builds that actually execute the vendored XGBoost and LightGBM CMake/native builders, native-search CUDA builder, and custom vector-ta builder (the CUDA builders locally as far as the toolchain permits and fully on Task 18). Record Cargo jobserver descriptors/environment, acquired/released token counts, `NUM_JOBS`, CMake/Ninja/MSBuild widths, rustc/native/nvcc subprocess concurrency, and peak process count. Fix or explicitly fail every builder that ignores the inherited budget; a leaf crate with no build script is not proof.

Replace vector-ta's serial hundreds-of-kernels PTX-then-fatbin loop and any independent native-CUDA parallelism with a bounded scheduler that acquires one inherited Cargo jobserver token per live nvcc child. Document how the build-script coordinator/implicit token is counted; never read host cores or create a second width. Give every kernel/architecture unique outputs and capture stdout/stderr separately, then emit/classify results in deterministic manifest order independent of completion order. On first error or cancellation, terminate/reap every child and publish no partial capability manifest. `test_cuda_builder_jobserver.py` injects capacities `1,2,12,23,64`, completion reordering, one failing nvcc child, and cancellation; it asserts maximum aggregate token-consuming concurrency `1,1,10,21,62`, stable output hashes/log attribution, and cleanup.

Define one versioned `CudaBuildManifestV1` wire contract in `neoethos-gpu-contracts` and make both builders emit/embed it. Its sole canonical encoder/hash implementation produces the exact `CudaBuildManifestIdV1` already stored by `ModuleLoadEvidenceV1`; vector-ta and native-search build scripts consume it through build-dependencies, while runtime loaders consume the same leaf normally. No builder/loader duplicates its serialization or hashing. It records the exact `BuildHostPlanV1` identity/resolution mode, compiler/toolkit, canonical ordered `-gencode` entries, exact SASS targets, embedded and standalone PTX targets, ABI, mandatory precision/code-generation flags, semantic-source hashes, and for each module its logical stem, payload kind, SHA-256, size, and reproducible build provenance. In the same leaf define the shape of `CudaHardwarePromotionKeyV1`, which binds compute capability, reviewed driver/runtime/toolkit compatibility, ABI and precision flags, semantic source manifest, build-manifest identity, exact loaded artifact hash, and one validated module-load path; Task 15 creates no promoted values, and only Task 18 may populate them from real-device evidence. The native-search release builder replaces its anonymous default `compute_70` PTX/raw architecture string with the exact shared `NEOETHOS_CUDA_ARCHS` set. `test_cuda_build_contract.py` uses a fake nvcc to inspect the exact final argv and artifact manifest for both builders. It injects `--use_fast_math`, `-fmad=true`, `-ftz=true`, `-prec-div=false`, `-prec-sqrt=false`, duplicate/conflicting precision flags, `-arch`/`-code`/`-gencode`, output-path/type switches, relocatable-device-code/link switches, and equivalent joined/separate spellings through `NVCC_ARGS`; every semantic override rejects before a child starts. Only an exact diagnostic-only allow-list is accepted. Every build-affecting environment input, including the build-plan identity and `VECTOR_TA_PREBUILT_FATBIN_DIR`, is emitted as `rerun-if-env-changed`; missing/zero-byte/prebuilt-`unknown` metadata rejects. Run `cuobjdump`/PTX inspection fixtures and prove the manifest's SASS/PTX target sets and hashes describe the emitted bytes exactly.

Inventory every production-reachable CUDA binary/module/library, not only the two custom builders: vector-ta, native search, and each exact retained model-local CubeCL/Candle/native-tree CUDA adapter receive a stable `backend_artifact_id`. A packaging/runtime wrapper emits the same typed manifest for third-party/prebuilt artifacts from exact package/source/features, build configuration, binary hash/size, and inspected SASS/PTX architecture images; it may not invent missing compiler/precision/architecture facts. Every retained backend has its own artifact-scoped promotion key and `ModelImplementationIdentityV1` binding, so a vector-ta sm89/SASS proof cannot authorize a Candle/CubeCL/tree binary. `test_cuda_artifact_inventory.py` reconciles features, loaders, packaged binaries, and runtime adapters against the inventory and fails on an unclassified artifact. If a third-party backend cannot expose a truthful build manifest, exact loaded bytes, module path, or per-architecture real-launch evidence, it is explicitly unavailable to strict/auto production rather than covered by a generic card key.

Repair Cargo fingerprinting before measuring throughput. The current vector-ta script emits `rerun-if-changed=kernels/cubin` even though that path is absent, tracks misspelled `VECTOR_TA_PREBUILD_*` names while reading `VECTOR_TA_PREBUILT_*`, and omits at least the fatbin-directory input; Cargo documents a nonexistent `rerun-if-changed` path as a cause of unnecessary rebuilding. Reconcile every `env::var`/tool/path input with exactly one matching `rerun-if-env-changed`/existing `rerun-if-changed` declaration, including nvcc selection and every prebuilt PTX/fatbin route; never watch a missing or generated output path. `test_cuda_builder_fingerprint.py` runs the identical large CUDA command twice with `-vv` and asserts the second invocation starts no build script/nvcc child and preserves output/manifest hashes, then mutates each declared source/env/tool input one at a time and proves exactly the affected unit rebuilds. It also compares the all-target compile and focused-test Cargo unit/feature graphs; an identical graph must reuse artifacts, while a genuinely different graph is recorded rather than described as unexplained cache loss.

On the real CUDA host, compare a recorded serial clean-build baseline with at least three bounded-parallel clean builds over the same source/target/toolchain and report raw wall time, median, peak process count, CPU utilization, RSS, and disk. A capacity above one must exercise multiple independent nvcc children while remaining within the jobserver ceiling; unexplained near-serial utilization or regression is profiled/fixed or receives an explicit operator-reviewed waiver, not accepted because the build eventually completes.

Run: `cargo test -p neoethos-data --test simd_dispatch_parity -- --nocapture`

Run: `cargo test -p neoethos-search --test simd_dispatch_end_to_end -- --nocapture`

Expected: every inventoried branch is force-selected at least once, reports its stable ID, matches the portable scalar validity/features/signals bit-for-bit, and either matches the exact broker-capable tuple or returns the same pre-financial typed unsupported result. No divergent optimized branch remains selectable.

**Step 4: Separate source validation from packaging artifacts**

Write a clean-target regression showing `cargo check --workspace --all-targets` and Rust desktop tests do not require pre-existing `mcp/target/release/neoethos-mcp[.exe]`, `mesh/target/release/neoethos-mesh[.exe]`, or `target/release/*.dll`. Move those generated resource entries out of the automatically loaded platform configs into a packaging-only overlay used by one script. The script builds root native release libraries plus isolated MCP/mesh sidecars under the inherited job budget, verifies expected profile/architecture/path/fresh commit provenance and hashes, rejects missing/stale/glob-empty resources, then invokes Tauri. Update CI/release workflows and `BUILDING.md`; no developer check may pass only because an old release artifact happens to exist.

**Step 5: Verify scripts and nightly toolchain**

Run representative `card_run.sh` planning/build commands in dry-run/test mode on synthetic capacities `1, 2, 12, 23, 96` and assert `1, 1, 10, 21, 94`. Capture `rustc -Vv`/`cargo -Vv` and fail if any repository path silently selects stable. Treat stable/MSRV commands, if run, as separately labelled packaging checks.

**Step 6: Commit**

```text
build: reserve two logical threads by default
```

---

## Milestone 5 — Determinism, logs, hardware proof, and handoff

### Task 16: Prove worker-count gate determinism without heuristic broker arithmetic

**Files:**
- Create: `crates/neoethos-search/tests/fixtures/captured_broker_cpu_parity/README.md`
- Create: `crates/neoethos-search/tests/fixtures/captured_broker_cpu_parity/quotes.vortex`
- Create: `crates/neoethos-search/tests/fixtures/captured_broker_cpu_parity/contract.json`
- Create: `crates/neoethos-search/tests/cpu_worker_determinism.rs`
- Create: `crates/neoethos-models/tests/model_worker_determinism.rs`
- Modify: `crates/neoethos-models/src/runtime/training_artifact.rs`
- Modify: `crates/neoethos-models/src/runtime/model_implementation_identity.rs`
- Modify: `crates/neoethos-models/src/training_orchestrator.rs`
- Modify only if RED proves nondeterminism: deterministic reduction sites in `crates/neoethos-search/src/`
- Modify only if RED proves nondeterminism: exact model/backend reduction/training sites selected by the test inventory

**Step 1: Build an immutable sanitized fixture**

Include exact symbol/account metadata units, timestamped Bid/Ask events, conversion-leg events, commissions, swaps, and closed-deal truth from a read-only/demo capture. Document account/environment/date and redaction; never commit credentials or account identifiers.

Task 16C's earliest shared broker-truth gate is already installed. This fixture validates scheduling/gate invariance without authorizing fallback arithmetic. Until the separate broker-truth replay plan consumes and reconciles every required captured field, both worker configurations must return the same typed unsupported result before costs, PnL, risk, trades, or metrics are calculated. Do not claim that a production backtest consumes the fixture's full broker semantics merely because the fixture contains them.

**Step 2: Write the 1-vs-auto comparison**

First inject one missing/unproven exact broker input and prove workers `1` and `auto` both fail at the identical gate/state before any financial arithmetic. If and only if the separate broker replay capability is fully proven for the complete fixture, use one deterministic seed to compare ordered fills/ledger, gross/net PnL, spread, commission, swap, conversion fees, drawdown, win rate, profit factor, expectancy, Sharpe, Sortino, Calmar, monthly series, trade count, and promotion verdict bit-for-bit. Otherwise record the full-tuple comparison as an explicit broker-plan release-gate item, not a passing infrastructure result.

Enumerate every production model family and exact CPU/CUDA/native backend from the runtime registry. With one immutable f64+validity frame, labels/splits, configuration, seed, and resolved device, run training and inference at worker width one and auto. A reproducible backend must preserve eligible-row indices, model serialization bytes/artifact hash, predictions/probabilities, downstream signals, loaded/skipped/failed sets, and semantic training metadata; worker telemetry is stored separately. Bind the exact deterministic-reduction/width contract into `ModelImplementationIdentityV1`. Make every current `DeterminismPolicy::BestEffort` or differing output RED and typed non-promotable; such a backend cannot train/resume/promote for Risky, Prop-Firm, live, or profitability evidence until repaired. A separately labelled stochastic research artifact may exist only outside all promotion/live eligibility.

**Step 3: Run RED and diagnose**

Run: `cargo test -p neoethos-search --test cpu_worker_determinism -- --nocapture`

Run: `cargo test -p neoethos-models --test model_worker_determinism -- --nocapture`

If gate results diverge, identify the first scheduling/state defect. If the exact broker capability is available and a tuple diverges, identify the first divergent state/reduction. For models, identify the first backend whose model bytes, row mapping, prediction, signal, or metadata differs and preserve the exact worker/backend log. Do not relax to a tolerance, accept `BestEffort` as reproducible, or enable a fallback formula merely because floating-point reductions run in a different order.

**Step 4: Implement deterministic reductions only where required**

Define a stable reduction tree/order and preserve ledger ordering. Apply the same rule only to the exact model/backend sites exposed by RED; otherwise keep that backend explicitly nondeterministic and unavailable to promotion/live modes. Re-run until every reproducible backend is bit-identical and every remaining stochastic backend fails the eligibility gate by type.

**Step 5: Commit**

```text
test: prove CPU worker-count trading determinism
```

### Task 16A: Retire integrated/WGPU execution and unify CPU/NVIDIA selection

**Files:**
- Create: `crates/neoethos-core/src/execution/device_policy.rs`
- Create: `crates/neoethos-core/tests/device_policy.rs`
- Create: `crates/neoethos-core/tests/device_policy_runtime_install.rs`
- Create: `crates/neoethos-search/tests/multi_cuda_determinism.rs`
- Modify: `crates/neoethos-core/src/config.rs`
- Modify: `crates/neoethos-core/src/system.rs`
- Modify: `crates/neoethos-core/src/system/backends.rs`
- Modify: `crates/neoethos-core/src/env_overrides.rs`
- Modify: `crates/neoethos-core/src/resolved_config.rs`
- Modify: `crates/neoethos-search/src/backend.rs`
- Modify: `crates/neoethos-search/src/execution_profile.rs`
- Modify: `crates/neoethos-search/src/engine_identity.rs`
- Modify: `crates/neoethos-search/src/eval.rs`
- Modify: `crates/neoethos-search/src/cubecl_eval.rs`
- Modify: native GPU modules selected by the compiler/source guard
- Modify: `crates/neoethos-data/src/lib.rs`
- Modify: `crates/neoethos-data/src/core/hpc_ta.rs`
- Modify: `crates/neoethos-data/src/core/gpu_indicators.rs`
- Create: `crates/neoethos-data/tests/indicator_device_policy.rs`
- Modify: `crates/neoethos-models/src/burn_models.rs`
- Modify: `crates/neoethos-models/src/common.rs`
- Modify: `crates/neoethos-models/src/deep_models.rs`
- Modify: `crates/neoethos-models/src/evolution/neat_gpu.rs`
- Modify: `crates/neoethos-models/src/evolution/crfmnes_gpu.rs`
- Modify: `crates/neoethos-models/src/statistical/linear_gpu.rs`
- Modify: `crates/neoethos-models/src/rl/dqn_impl.rs`
- Modify: `crates/neoethos-models/src/runtime/capabilities.rs`
- Modify: `crates/neoethos-models/src/runtime/install.rs`
- Modify: `crates/neoethos-models/src/runtime/profile.rs`
- Modify: `crates/neoethos-models/src/runtime/training_artifact.rs`
- Modify: `crates/neoethos-models/src/training_orchestrator.rs`
- Modify: `crates/neoethos-models/src/tree_models/common.rs`
- Modify: `crates/neoethos-models/src/tree_models/catboost.rs`
- Modify: `crates/neoethos-models/src/tree_models/lightgbm.rs`
- Modify: `crates/neoethos-models/src/tree_models/config.rs`
- Modify: `crates/neoethos-models/src/tree_models/xgboost.rs`
- Modify: `crates/neoethos-gpu-cuda/src/lib.rs`
- Modify: `crates/neoethos-gpu-cuda/native/neoethos_gpu_cuda.h`
- Modify: `crates/neoethos-gpu-cuda/native/smoke.cu`
- Modify: `crates/neoethos-gpu-cuda/native/stub.cpp`
- Modify: `crates/neoethos-cli/src/main.rs`
- Modify: `crates/neoethos-cli/src/gpu_bench.rs`
- Modify: `crates/neoethos-cli/src/gpu_bench_population.rs`
- Modify: `crates/neoethos-cli/src/gpu_bench_snapshot.rs`
- Modify: `crates/neoethos-cli/src/tui/pages/config_view.rs`
- Create: `crates/neoethos-cli/tests/device_policy_roundtrip.rs`
- Modify: `crates/neoethos-app/src/server/settings.rs`
- Modify: `crates/neoethos-app/src/server/hardware.rs`
- Create: `crates/neoethos-app/tests/device_policy_api.rs`
- Modify: `desktop/src/api.ts`
- Modify: `desktop/src/screens/Advanced.tsx`
- Create: `desktop/src/devicePolicy.test.ts`
- Modify: root/core/data/search/models/app/CLI/desktop Cargo manifests and feature forwarding, including `crates/neoethos-app/Cargo.toml`, `crates/neoethos-cli/Cargo.toml`, and `desktop/src-tauri/Cargo.toml`
- Modify: `config.yaml`
- Modify: `desktop/src-tauri/resources/config.yaml`
- Modify/delete: WGPU/Vulkan/ROCm-only examples, tests, settings, help, and stale comments found by the source guard
- Modify: `.github/workflows/ci.yml`
- Modify: `.github/workflows/agent-stage1.yml`
- Modify: `.github/workflows/release.yml`
- Modify: `scripts/gpu-bench/test_stage1_ci_contract.py`
- Modify: `BUILDING.md`

**Step 1: Capture the final pre-removal baseline**

Preserve the current synthetic adapter-classification tests and the existing measured Ryzen 5 5675U iGPU result (10–20x slower plus fused-path OOM) in the audit with exact provenance; do not present a source comment as a fresh hardware run. Record every production feature/config/API/CLI/model path that accepts `IntegratedGpu`, `wgpu_integrated_gpu`, `DefaultDevice`, Vulkan/WGPU/ROCm, the stale deleted-override message, or a generic `gpu` selector that can land on an unsupported adapter. The locked WGPU 29 source is the classification oracle for this inventory: `Instance::enumerate_adapters` plus `AdapterInfo.device_type`, not vendor-name heuristics.

**Step 2: Write RED typed-policy tests**

Define `ExecutionDevicePolicy::{CpuOnly, AutoNvidiaDiscrete, NvidiaDevices(NonEmptySet<GpuDeviceId>)}` and a resolved inventory carrying CUDA UUID, PCI bus id, runtime ordinal, name, VRAM, compute capability, driver/runtime, ABI, f64 probe, health, and the exact `CudaHardwarePromotionKeyV1`/module-load path authorized for that card. Prove CPU-only performs no CUDA enumeration/context initialization, including through `neoethos-data`/vector-ta and every model adapter; auto includes only fully preflight-clean NVIDIA discrete cards whose exact compute-capability/artifact/load-path key was promoted by Task 18. Discovery may report an unpromoted NVIDIA card only as `build_compatible_unvalidated`; auto and strict selection reject it. Explicit one/many selection preserves membership but UI/input order is display-only, and both auto and explicit sets canonical-sort by stable UUID+PCI identity before scheduling and artifact hashing. Duplicate, missing, ambiguous, integrated, non-NVIDIA, virtual/software/unknown, ABI-old, f64-failed, unpromoted architecture/load-path, or unhealthy devices fail before work starts. Fake indicator/model-lane assignments prove a selected nonzero ordinal/UUID is propagated and concurrent jobs receive distinct assigned cards. Tests reject each current direct parser/default (`gpu:N`, `cuda:N`, `gpu:0`, `cuda:0`) in common, NEAT, CR-FM-NES, linear, DQN, training artifacts, native tree adapters, and CLI GPU-benchmark population/snapshot commands. Make app API, CLI parse/save/reload, and desktop UI -> typed DTO -> saved config -> runtime assignment tests RED for `CpuOnly`, auto, a nonzero explicit card, reordered and duplicate inputs, an unpromoted architecture, and a multi-card set; every surface must arrive at the same canonical `ResolvedDeviceAssignment` and artifact identity. Auto may fall back visibly only in a fallback-allowed mode; GPU-required never does.

**Step 3: Implement one selection authority across every surface**

Expose CUDA UUID and PCI bus id from the native API and treat runtime ordinal as session-local reporting only. Replace independent search/model/env/indicator selectors with the resolved typed policy in CLI, app/API, desktop, scheduler, search, every direct model trainer/adapter and training artifact, model orchestration, `neoethos-data`, and every GPU benchmark/population/snapshot entry. Delete the process-global `POLICY_OVERRIDE`/`IndicatorComputePolicy` authority in `hpc_ta.rs`; pass the resolved per-job assignment into the vector-ta `GpuIndicatorEngine` instead of `GpuIndicatorEngine::new(..., 0)`. Replace model-local and benchmark-local string/ordinal parsing/default-zero with an injected `ResolvedDeviceAssignment`; artifacts persist stable UUID/PCI identity and separately label the session ordinal, exact backend, precision boundary, promotion key, module-load evidence, launch evidence, and CPU fallback status. The hardware API returns the same typed discovery DTO consumed by desktop; the desktop saves the canonical set and the app/CLI runtime reopens it without an independent conversion. A settings update must preflight and atomically install the new assignment for subsequent jobs or return a typed `pending_restart` state alongside the still-active assignment; saving a value cannot silently leave a different runtime selection active. `CpuOnly` never constructs any CUDA engine/library context. Provide an actual CPU-only UI/CLI option and a discoverable multi-select for one or many eligible NVIDIA cards. Retire legacy selectors at sealed config load with one structured warning and write only the canonical policy on save.

`CpuOnly` must not load/initialize CUDA, WGPU, or a GPU model backend. Sort auto-selected cards by canonical identity. Make every selected card available to the scheduler and run independent symbol/timeframe jobs concurrently with exactly one native-CUDA lane/context per assigned card. Never silently use only `card_ids[0]` when the UI/config says multiple cards are selected; logs enumerate selected, assigned, initialized, used, idle, and rejected cards with reasons.

**Step 4: Remove unsupported production GPU surfaces**

Remove integrated-GPU opt-in/skip code, generic/default WGPU selectors, `gpu-vulkan`, `gpu-rocm`, `gpu-apple`, WGPU/Metal/ROCm production feature forwarding/dependencies from every root/core/data/search/models/app/CLI/desktop manifest, and model/runtime values such as `wgpu_integrated_gpu`, `wgpu_virtual_gpu`, and `wgpu_default`. Remove or rewrite examples/tests/help that claim those paths are supported. Keep AMD x86-64 CPU execution portable rather than tuning to one Ryzen, but label all other CPU vendors/architectures unvalidated for this release. Shared feature/search/backtest NVIDIA execution uses only native CUDA f64/ABI-v4; the CubeCL search evaluator and known multi-device panic/fallback route are deleted rather than revived. Exact-named intrinsically-f32 model-local CubeCL/Candle/native-tree CUDA adapters may remain only under the injected resolved assignment, no-fallback real-launch evidence, and adapter-local narrowing contract; no WGPU/HIP backend or shared f64 evaluator is retained. The CubeCL prerelease experiment therefore applies only to those exact CUDA model adapters and cannot re-enable the deleted search/trading lane.

Remove/update the ROCm and Vulkan jobs in `.github/workflows/ci.yml`, Vulkan commands in `agent-stage1.yml`, the corresponding `test_stage1_ci_contract.py` assertions, and Vulkan/ROCm wording in `BUILDING.md` and release workflows. Add a repository guard across `.github`, scripts, manifests, config, README/BUILDING, CLI/API/UI help, tests, and examples for backend-specific `gpu-vulkan|gpu-rocm|gpu-apple|WGPU|Vulkan|ROCm`, exact Apple-Metal dependency/feature names, `AcceleratorBackend::Metal`, and GPU-policy/device-selector values that choose Metal. Do not scan the bare word `Metal`: cTrader metals asset classes, symbol categories/names, capture/download code, strategy categories, and market UI are legitimate financial domain behavior and must remain. Add exact allow-list fixtures proving those paths survive while an injected Apple-Metal backend selector fails. Exclude `.git`, `target`, caches, temporary/generated logs. Allow documentary backend vocabulary only in immutable `docs/audits/**`, the exact approved design/plan files, and exact guard-fixture symbols; production docs/help and every executable/config surface remain scanned. Validate workflow syntax and prove no shipped/CI command requests a retired feature.

**Step 5: Add deterministic multi-card behavior**

For independent jobs, prove all selected cards can be busy concurrently and each result retains its device identity, including vector-ta indicator and model-training jobs resolved to their assigned UUID/PCI identity rather than every job using ordinal zero. Direct model selectors in common/runtime profile/training artifact, NEAT, CR-FM-NES, linear, DQN, and native tree configuration/adapters receive the typed per-job assignment; none parses a raw device string or probes/defaults a card independently. Do not fan the same feature frame or model job to every card unless profiling and deterministic parity establish a benefit. For a single population, enable native-CUDA sharding only if tests prove deterministic stable work-id shards, per-card capability/VRAM admission, bounded feeders, exactly-once index coverage, and merge by original work id independent of device completion order. Device loss in strict mode fails the affected run/checkpoint; no CPU/other-device recompute is hidden. If real sharding proof is unavailable, fail that single-population mode explicitly while still using the selected cards across independent jobs. Independent-job support for multiple selected cards is a release gate and requires a process with at least two real eligible devices in Task 18.

**Step 6: Verify locally and guard the removal**

Run all-target tests for core, data, models, search, app, CLI, GPU contracts/CUDA, and desktop. Run `cargo test -p neoethos-core --test device_policy_runtime_install -- --nocapture`, `cargo test -p neoethos-cli --test device_policy_roundtrip -- --nocapture`, `cargo test -p neoethos-app --test device_policy_api -- --nocapture`, and the exact desktop `devicePolicy.test.ts`; each proves UI/API/CLI saved policy reaches the same runtime assignment/artifact, and the runtime-install test distinguishes atomically installed versus explicit pending-restart state. Add a source/feature guard rejecting integrated/WGPU/Vulkan/ROCm and exact Apple-Metal GPU-backend execution symbols, CubeCL search/backtest execution, stale selectors, adapter-local `gpu:N`/`cuda:N` parsing, default-adapter/default-zero routing, the separate indicator-policy `OnceLock`, hard-coded CUDA ordinal zero, first-card-only truncation, and strict/auto use of an unpromoted compute-capability/artifact/load-path key, while preserving tested financial metals asset-class/symbol/category logic. Exact-named CUDA-only model-local CubeCL/Candle/tree adapters are the only CubeCL/third-party GPU allow-list and must accept `ResolvedDeviceAssignment`. Fake-inventory tests cover zero, one, multiple, promoted, and unpromoted NVIDIA cards plus every rejected class. CPU-only tests run on this no-NVIDIA host and prove zero GPU initialization through models and vector-ta. Real multi-card execution remains RED/pending until Task 18 and blocks integration if no host with at least two simultaneously visible eligible cards is tested.

**Step 7: Commit**

```text
refactor: support explicit CPU and NVIDIA device sets only
```

### Task 16B: Review the full feature plan and make vector-ta f64 CUDA coverage exhaustive and truthful

**Files:**
- Create: `docs/audits/2026-08-15-vector-ta-mathematical-review.md`
- Create: `crates/neoethos-data/tests/vector_ta_production_manifest.rs`
- Create: `crates/neoethos-data/tests/vector_ta_formula_truth.rs`
- Create: `crates/neoethos-data/tests/vector_ta_f64_reachability.rs`
- Create: `crates/neoethos-data/tests/vector_ta_feature_parity.rs`
- Modify: `crates/neoethos-data/tests/feature_plan_transform_truth.rs`
- Modify: `crates/neoethos-data/tests/full_feature_validity_parity.rs`
- Modify: `crates/neoethos-data/tests/feature_plan_producer_coverage.rs`
- Modify: `crates/neoethos-data/tests/feature_semantic_source_closure.rs`
- Modify: `crates/neoethos-data/src/core/features.rs`
- Modify: `crates/neoethos-data/src/core/feature_registry.rs`
- Modify: `crates/neoethos-data/src/core/normalization.rs`
- Modify: `crates/neoethos-data/src/core/resample.rs`
- Modify: `crates/neoethos-data/src/core/cross_pair_features.rs`
- Modify: `crates/neoethos-data/src/core/smc.rs`
- Modify: `crates/neoethos-data/src/core/session_features.rs`
- Modify: `crates/neoethos-data/src/core/regime_detection.rs`
- Modify: `crates/neoethos-data/src/core/quant_features.rs`
- Modify: `crates/neoethos-data/src/core/footprint_features.rs`
- Modify: `crates/neoethos-data/src/core/gpu_indicators.rs`
- Modify: `crates/neoethos-data/src/core/hpc_ta.rs`
- Modify: `crates/neoethos-data/src/core/indicator_ledger.rs`
- Modify: `crates/neoethos-data/src/core/indicator_telemetry.rs`
- Modify: `crates/neoethos-data/src/lib.rs`
- Modify: `crates/neoethos-gpu-contracts/src/lib.rs`
- Create: `crates/neoethos-gpu-contracts/tests/resident_f64_feature_buffer.rs`
- Modify: `crates/neoethos-gpu-cuda/src/lib.rs`
- Modify: `crates/neoethos-gpu-cuda/src/population.rs`
- Modify: `crates/neoethos-gpu-cuda/native/neoethos_gpu_cuda.h`
- Modify as required: `crates/neoethos-gpu-cuda/native/prototype_b.cu`
- Modify as required: `crates/neoethos-gpu-cuda/native/prototype_b_population.cu`
- Modify: `crates/neoethos-search/src/gpu_native/mod.rs`
- Modify: `crates/neoethos-search/src/gpu_native/engine.rs`
- Modify: `crates/neoethos-search/src/gpu_native/prototype_b_engine.rs`
- Modify: `crates/neoethos-search/src/gpu_native/prototype_b_population_eval.rs`
- Modify: `crates/neoethos-search/src/discovery.rs`
- Modify: `crates/neoethos-search/src/eval.rs`
- Modify: `crates/neoethos-search/src/validation.rs`
- Modify: `crates/neoethos-trader/src/gene_signal.rs`
- Modify: `crates/neoethos-trader/src/blend_signal.rs`
- Modify: `crates/neoethos-app/src/app_services/live_trading.rs`
- Create: `crates/neoethos-search/tests/resident_f64_feature_handoff.rs`
- Modify: `vendor/vector-ta-0.2.9-patched/build.rs`
- Modify: `vendor/vector-ta-0.2.9-patched/kernels/cuda/neoethos_f64_kernels.cu`
- Modify: `vendor/vector-ta-0.2.9-patched/src/cuda/device_types_f64.rs`
- Modify: `vendor/vector-ta-0.2.9-patched/src/cuda/module_loader.rs`
- Modify: `vendor/vector-ta-0.2.9-patched/src/cuda/runtime.rs`
- Modify: `vendor/vector-ta-0.2.9-patched/src/cuda/neoethos_f64_wrapper.rs`
- Modify: `vendor/vector-ta-0.2.9-patched/src/indicators/dispatch/cuda_f64.rs`
- Create: `crates/neoethos-data/tests/vector_ta_cuda_artifact_contract.rs`
- Create: `crates/neoethos-data/tests/vector_ta_module_path_strict.rs`
- Modify: exact scalar indicator sources and f64 lane tests selected by the generated manifest/compiler

**Step 1: Generate the production coverage manifest**

Enumerate every `ALL_INDICATORS` base output, multi-output name, period-sweep output, extended-sweep parameter tuple, and pattern-matrix output that can enter a production `FeatureFrame`. Treat the current scalar f64 implementation only as a parity reference until its mathematics is independently reviewed. For each row record its complete `SemanticSourceSetV1` plus filtered `RelevantDependencySetV1` closure and resulting production semantic-source hash, formula/implementation semantic version, independent formula evidence, hand-derived/adversarial test-vector evidence, invariants, input/source derivation, first-valid and warmup semantics, NaN/singularity behavior, parameter conventions, output shape/names/order, f64 CUDA entry point, assignment/device contract, reviewer/disposition, and one status: `cuda_candidate_unverified`, `real_f64_cuda(CudaHardwarePromotionKeyV1)`, `reviewed_cpu_only`, or `unsupported`. `real_f64_cuda` is never a global boolean: the same row may be promoted for sm89/SASS and remain fail-closed for sm86, sm90, or PTX/JIT. Reconcile compiler-selected producer/helper/generator/table/macro and external package/source/version/checksum/features reachability against every row; an unclassified reachable or unreachable declared source/dependency blocks review, and source/dependency mutation fixtures prove the affected manifest/plan identity changes while unrelated helpers/dependencies do not. During dirty RED/GREEN work, generate the embedded semantic manifest from canonicalized exact current relevant source/generator/dependency payloads; it never hashes checkout line endings and never includes the repository commit in `FeaturePlanIdentity`. Record commit, Git object IDs, index state, compiler, and artifact hash separately as build/review provenance. At Task 17/18/release, compare every embedded relevant payload hash with the exact final committed Git blob/dependency payload and reject stale/dirty/index/lock divergence; packaged runtime consumes the embedded manifest without `.git`. Thus unrelated commits/dependency updates and LF/CRLF checkout conversion preserve identity, while a relevant semantic payload change changes it. Task 16B may assign only `cuda_candidate_unverified` after CPU formula review plus static/compile/contract evidence; that state is fail-closed for strict GPU selection. Only Task 18 may promote a row for one exact `CudaHardwarePromotionKeyV1` after the exact current-card launch/readback/parity/telemetry proof. A row cannot receive any reviewed status until its CPU formula evidence passes. The canonicalized ordered semantic manifest hash is an input to `FeaturePlanIdentity`; changing formula, conventions, parameters, validity, outputs, implementation semantics, or a relevant dependency changes the identity even when column names/types/order do not, while concrete dataset generation/range remains only in `DatasetFeatureArtifactProvenance`. Reconcile this manifest against vector-ta's registry, `F64_KERNELS`, withheld tables, the native module's actual exported symbols, `CudaBuildManifestV1`, and `GPU_SWEEP_SPECS`; neither a table row, current CPU/GPU agreement, an `_f64` suffix, nor fatbin presence alone is capability proof.

For a standard indicator, use the original author/paper or a documented canonical definition where one exists and write a deliberately independent slow f64 reference/test vector that shares no production recurrence/helper. For project-specific indicators, derive small vectors by hand and assert mathematical invariants. Cover constant, monotone, alternating, impulse, zero-denominator, very small/large magnitude, NaN/gap, shortest valid length, warmup+1, parameter edges, recurrence initialization, multi-output ordering, and validity. Cross-library agreement is corroboration, not the sole oracle. Any ambiguous competing convention is versioned/named explicitly rather than silently choosing the current output.

Build a second, exhaustive transformation-node section in the same versioned feature-plan manifest. Reconcile both `feature_registry` and the compiler/current production call graph in `neoethos-data/src/lib.rs`; enumerate classic/vector-ta plus every production SmartMoneyConcept (`smc.rs`), Session (`session_features.rs`), Regime (`regime_detection.rs`), Quantitative (`quant_features.rs`), Footprint (`footprint_features.rs`, even though it is currently absent from the registry), source convention/derivation, fixed/calendar resample, HTF availability/staleness, cross-pair point-in-time alignment, normalization split/fitted-state behavior, and every derived/post-processing node. `feature_plan_producer_coverage` fails if any value-producing function reachable from production lacks exactly one manifest row or if a registry row is unreachable/duplicated. Each requested node has a semantic-source hash, formula/causality/split/validity evidence, adversarial fixtures, and `reviewed` or `unreviewed` status; missing/unreviewed nodes fail the requested CpuOnly/strict/live/promotion plan before work. Independent tests use hand-derived values/invariants per producer where possible and causality/leakage metamorphic checks: appending or perturbing future/validation/test bars cannot change earlier/train features or fitted state; cross-pair/HTF joins never consume future data and enforce explicit maximum staleness; gaps remain invalid; calendar boundaries follow the typed timeframe rule; bar timestamp convention is preserved. No hard-coded whole-series/80% normalization fit or timestamp fallback is accepted.

**Step 2: Write RED full-plan truth, reachability, and anti-fake tests**

Fail when a GPU-claimed production row reaches any f32 host/device buffer, cast, byte count, literal-driven precision loss, legacy f32 dispatcher, empty/no-op stub, host-compute-then-upload wrapper, wrong output id, unverified first-valid rule, missing kernel write, zero launch count, build-manifest/artifact mismatch, or module-load route not present in its exact promotion key. Seed output buffers with sentinels and require a real assigned-device launch to overwrite the exact expected range. Reject hard-coded ordinal zero and assert telemetry records CUDA UUID/PCI identity, entry point, launches, rows, outputs, transfers, CPU segments, `CudaBuildManifestV1` identity, exact loaded artifact hash, and `ModuleLoadEvidenceV1`. Add a whole-production-call-graph guard rejecting direct `Module::from_ptx` outside the common loader. Tests force fatbin/SASS and PTX/JIT, reject contradictory force controls, clear stale failure state after success, reject zero-byte/unknown prebuilt artifacts, and prove an unvalidated PTX fallback cannot inherit a SASS promotion. Retained upstream/Python f32 APIs are outside production only when an exact call-graph/source guard proves they are unreachable from every NeoEthos feature path.

Consume Task 5B's immutable pre-repair scalar-validity RED/GREEN ledger first. Before any additional scalar or CUDA repair in Task 16B, populate `vector_ta_formula_truth.rs` with the independent published/hand-derived formula vectors and invariants from Step 1 and run it together with reachability/manifest tests. Rows already formula-correct may begin GREEN and are recorded as such; every newly discovered convention/formula/validity mismatch must be captured as a focused RED before its implementation changes. Also make the same-name/type/order but changed-semantic-version `FeaturePlanIdentity` rejection RED before modifying manifest propagation. Add golden tests proving LF/CRLF worktrees, an unrelated commit with identical relevant payloads, dirty RED/GREEN generation from canonical current bytes, and packaged runtime without `.git` produce the expected same embedded semantic manifest/plan identity; one relevant payload change differs. Separately test the final-source guard rejects an embedded manifest that no longer matches the committed relevant blobs. Concrete input-generation changes must leave the plan identity equal and change `DatasetFeatureArtifactProvenance`/cache identity instead.

Run: `cargo test -p neoethos-data --test vector_ta_formula_truth --test vector_ta_f64_reachability --test vector_ta_production_manifest --test vector_ta_cuda_artifact_contract --test vector_ta_module_path_strict --test feature_plan_producer_coverage --test feature_semantic_source_closure --test feature_plan_transform_truth --test full_feature_validity_parity -- --nocapture`

Expected: existing correct rows PASS, each independently exposed scalar/kernel/identity defect has an explicit failing assertion, and no repair begins without its RED evidence.

**Step 3: Make the full feature plan device-resident**

Replace the current GPU-only subset of the multi-period sweep with a single versioned production feature plan. Upload timestamp/OHLCV and derived shared sources once per frame to its assigned card, schedule independent proven indicator batches across bounded streams, keep f64 outputs resident, and pass resident named buffers directly into compatible native-CUDA pre-financial signal/search stages instead of device-host-device bouncing. Financial trade/cost/PnL/risk evaluation is a separate consumer that cannot launch without the already-installed exact broker capability; external/non-broker-real input must stop before that consumer.

Define `ResidentF64FeatureBuffer` in the shared GPU contract with stable CUDA UUID/PCI identity, session/context identity, full `FeaturePlanIdentity` (not only a name/type schema hash), typed `DatasetFeatureArtifactProvenance` for the exact resident input generations/ranges, row/column counts, strides/layout, explicit f64 element width, validity-buffer layout, producing stream/readiness event, allocation generation, and an RAII owner/lifetime token. One `CudaDeviceSession`/primary-context owner for the resolved assigned device is shared or borrowed by vector-ta production and the native-CUDA consumer; neither side creates an independent ordinal-selected context. The producer owns the allocation and records a CUDA event after the last write; the consumer waits on that event in its bounded stream and borrows the buffer without outliving its allocation/session. A device, context, allocation-generation, feature-plan, provenance, layout, or ABI mismatch and a stale/destroyed session fail before launch with no host fallback or global synchronize masquerading as handoff.

Extend the ABI-v4 Rust/C/CUDA contracts and native search engine so a compatible pre-financial Prototype-B/native signal evaluation consumes the resident buffer pointer/validity metadata directly and records the exact allocation/session identity used. Contract tests prove ownership and stale-handle rejection; a real-device integration test proves one OHLCV upload, nonzero vector-ta and pre-financial search/signal kernel launches, zero intermediate D2H/H2D feature bytes, exact schema/order, and no fallback. A financial native kernel launch is tested only with an exact synchronized broker-capable fixture; otherwise the same integration asserts typed unsupported and zero financial launches. Materialize bounded host/Vortex chunks only when persistence, a CPU-only consumer, or an explicit parity probe requires it, and report that boundary and transfer bytes explicitly; strict end-to-end GPU mode rejects any such intermediate materialization before search. Preserve deterministic feature names/order/validity and bounded VRAM admission; never launch the same frame redundantly on every selected card without measured benefit.

**Step 4: Close real f64 gaps one indicator/output at a time**

For every production row not yet `cuda_candidate_unverified` or `real_f64_cuda`, first make the scalar lane pass its independent formula/vector/invariant review—including NaN/finite rules, warmup, parameter coupling, recurrence initialization, accumulator/reduction order, multi-output layout, and error semantics. Repair the scalar implementation when it disagrees; bump its formula/implementation semantic version and the manifest/`FeaturePlanIdentity` instead of forcing CUDA to copy the bug. Old Vortex features, genes, models, portfolios, checkpoints, `SavedStrategy`, live rows, federation/mesh payloads, and GPU resident handles then fail/regenerate unless an explicit migration proves semantic equivalence. Only then implement/test the actual f64 kernel and wrapper against both the independently reviewed formula result and the corrected scalar parity reference. Local implementation produces at most `cuda_candidate_unverified`; it cannot set `real_f64_cuda` or enable strict selection. Do not mechanically rename an f32 kernel. Fix vector-ta kernel panics at their source with a focused regression; `KernelPanic`, unexplained `Truncated`, `ComputeFailed`, or `OtherDispatchError` becomes a hard feature-build failure rather than an all-NaN success. Expected warmup/undefined cells carry explicit validity and cannot threshold as a numeric value. Permanently unsupported and duplicate outputs are removed from the versioned search vocabulary before allocation/transfer/scoring; frame-specific degenerate outputs remain counted and validity-masked so they cannot influence ranking.

**Step 5: Repair and prove every non-vector transformation node**

Consume and independently re-review Task 5B's recorded pre-repair RED/GREEN evidence and semantic versions; do not pretend the now-validity-aware source is the original pre-repair baseline. For each requested non-vector node whose formula remains unreviewed or for any newly exposed defect, capture a new focused RED before changing it, then repair against independent formula/causality/split/validity evidence. Replace normalization's implicit whole-series/80% fit with an explicit caller-supplied training partition if Task 5B has not already done so; fit only valid training cells, persist the immutable fitted state, and preserve invalidity during transform. Cross-pair and HTF alignment use causal as-of semantics with a typed maximum-staleness/no-tick rule, never a future observation or timestamp/value fallback. Review SmartMoneyConcept, Session, Regime, Quantitative, Footprint, resample/calendar, and every post node independently; bump its semantic version/`FeaturePlanIdentity` on correction. The full-plan selector stays fail-closed until all requested node statuses and source hashes are reviewed. Run the full transform/validity tests GREEN and prove append/future-split invariance plus stable row/reason telemetry through search, model adapters, trader, and live capture.

**Step 6: Define strict versus hybrid truth**

Strict full-GPU preflight rejects a requested feature/model/search graph containing any mathematically unreviewed transformation node, `cuda_candidate_unverified`, `reviewed_cpu_only`, or `unsupported` segment before allocation. Fallback-allowed mode may run only nodes with independently reviewed scalar f64 formula/causality/split/validity semantics under its CPU lease, logs every CPU node, and reports exact CUDA/CPU coverage; it is labelled hybrid and never end-to-end GPU. `CpuOnly` runs the same versioned plan only when every requested vector and non-vector node is reviewed, without loading CUDA. Live/promotion eligibility has the same full-plan math gate plus the separate broker-real gate. No mode silently drops a node, zero-fills invalidity, substitutes an f32 path, treats CPU/GPU agreement as mathematical proof, promotes a candidate from compile evidence, or reports a kernel that did not launch.

**Step 7: Verify locally and on Task 18 hardware**

Locally run every independent formula/vector/invariant and full-transformation causality/split/validity review first, then the complete manifest/reachability/scalar-reference tests, resident-contract ownership/ABI tests, native-consumer compile tests, and a no-CUDA `CpuOnly` full-feature fixture with complete logs. The fixture includes every requested vector-ta, SmartMoneyConcept, Session, Regime, Quantitative, Footprint, resample/calendar, HTF, cross-pair, normalization, and post node; it remains fail-closed if any node is absent or unreviewed. Add hand-calculated signal/threshold fixtures so downstream CPU signal truth is not established merely by reproducing current production output; broker fill/cost/PnL truth remains the separate cTrader gate. On the real card run every mathematically approved f64 row against the reviewed CPU result on adversarial NaN/warmup/gap/staleness/high-precision fixtures and the exact GitHub corpus hashes, then run the vector-ta-to-native-search resident-handoff integration. Require bit identity where the operation order is identical; any bounded ULP/relative exception needs a first-divergent-operation analysis and still must produce identical feature validity and signals. The external GitHub corpus must then produce the identical typed unsupported result before financial evaluation on CPU and GPU. Compare ordered trades, ledger, and metrics only on an exact synchronized broker-captured replay fixture bound to the same data identity and complete broker-truth capability. Measure one-upload resident throughput, launch count, D2H/H2D bytes by stage, allocation/session identities, occupancy, VRAM, CPU fallback count, and end-to-end time rather than isolated kernel time. A host `Vec<f64>` boundary or device-host-device bounce keeps strict end-to-end GPU RED.

**Step 8: Commit**

Commit only the reviewed scalar repairs, disabled CUDA candidates, contracts, and tests. No `real_f64_cuda` capability bit or strict-GPU enablement is committed here.

```text
feat: prepare reviewed f64 CUDA indicator candidates
```

### Task 16C: Install the earliest fail-closed broker-truth boundary

**Files:**
- Modify: `crates/neoethos-search/src/eval.rs`
- Modify: `crates/neoethos-core/src/config.rs`
- Modify: `crates/neoethos-trader/src/data_replay.rs`
- Modify: exact promotion/Risky/Prop-Firm/live capability consumers selected by the guard
- Create: `crates/neoethos-search/tests/broker_real_capability_gate.rs`
- Create: `crates/neoethos-app/tests/live_broker_truth_gate.rs`

**Step 1: Write the fail-closed tests first**

Before changing current heuristic routes, make focused tests RED for flat/hourly/session/typical spread, shipped `backtest_spread_pips`, `default_pip_size`, operator-estimated commission/cost, OHLC proxies, missing-to-zero behavior, and every `scenario=approximate` arithmetic entry. Each fixture attempts any historical financial evaluation/search/backtest, promotion, Risky, Prop-Firm, and live eligibility and must be rejected before cost/PnL/risk computation, trading, or artifact publication—not merely relabelled non-promotable. Test that live risk/PnL requires an authoritative broker response and that absent exact broker capability disables rather than guesses. Scenario labels may select among exact captured broker datasets/contracts only; they cannot synthesize spread, pip/tick value, commission, swap, conversion, PnL, or risk inputs.

Run: `cargo test -p neoethos-search --test broker_real_capability_gate -- --nocapture`

Run: `cargo test -p neoethos-app --test live_broker_truth_gate -- --nocapture`

Expected RED: at least one current heuristic/default route still executes cost/PnL/risk arithmetic (whether or not it later crosses a claimed-real/eligibility boundary).

**Step 2: Implement the earliest shared capability gate**

Install a typed, versioned broker-truth capability at the earliest shared evaluation/live boundary. The current flat/hourly/typical spread, default pip-size, operator-estimated commission/cost, OHLC proxy, missing-to-zero, and heuristic `scenario=approximate` routes cannot execute financial arithmetic at all. Historical search/backtest, promotion, Risky Mode, Prop-Firm Mode, and live eligibility remain disabled until synchronized historical Bid/Ask plus conversion legs, exact `ProtoOASymbol` units, broker unrealized-PnL, and close/deal reconciliation from the separate broker plan pass. Live PnL/risk decisions use authoritative broker responses or live trading is disabled. Scenario selection is permitted only over exact captured broker inputs with typed provenance; no estimate/default/proxy mode remains runnable.

**Step 3: Verify and guard**

Rerun both focused tests GREEN and add a whole-source/call-path guard proving heuristic/default/OHLC/missing-zero/operator-estimated/approximate routes cannot execute financial arithmetic at all, not only that they cannot create broker-real, promotion, Risky, Prop-Firm, or live eligibility. Exact captured broker-provenance paths are the only allow-list. Preserve and classify the complete logs. Task 16 then tests identical fail-closed worker behavior before arithmetic; Task 17 runs the full root/member/all-target and diagnostic matrix on this post-gate code; Task 18 repeats all affected CPU/CUDA parity/capability paths on real hardware before that VPS is destroyed.

**Step 4: Commit the fail-closed release boundary**

```text
fix: fail closed without broker-real evidence
```

### Task 17: Run complete local validation and classify every diagnostic

**Files:**
- Create: `docs/audits/2026-08-15-vortex-cpu-validation.md`
- Create if useful: `scripts/classify-audit-log.ps1`
- Modify implementation files only when a diagnostic reveals a real defect

**Step 1: Start from a clean intended worktree**

Verify branch, status, worktree path, ignored artifacts, and diff. Preserve unrelated user changes. Inventory current master dirty/ignored state and `.codex/` read-only; classify owner/purpose plus credential, generated, cache, temporary, or source-worthy status. Do not copy anything blindly, but do not pre-decide blanket exclusion either. Clean and test only intended project changes, preserve credentials/cache locally, and record every item selected for later integration or deliberately deferred.

**Step 2: Format and run root validation with full logs**

Run and tee complete output for:

```text
cargo fmt --all -- --check
cargo test -p neoethos-execution-budget --all-targets -- --nocapture
cargo test -p neoethos-data --all-targets -- --nocapture
cargo test -p neoethos-models --all-targets -- --nocapture
cargo test -p neoethos-core --all-targets -- --nocapture
cargo test -p neoethos-search --all-targets -- --nocapture
cargo test -p neoethos-trader --all-targets -- --nocapture
cargo test -p neoethos-app --all-targets -- --nocapture
cargo test -p neoethos-cli --all-targets -- --nocapture
cargo test -p neoethos-gpu-contracts --all-targets -- --nocapture
cargo test -p neoethos-gpu-cuda --all-targets -- --nocapture
cargo test -p neoethos-codex --all-targets -- --nocapture
cargo test -p neoethos-mcp --all-targets -- --nocapture
cargo test -p neoethos-desktop --all-targets -- --nocapture
cargo test -p neoethos-autoresearch --all-targets -- --nocapture
cargo check --workspace --all-targets
```

Then run meaningful standalone CPU and native-CUDA feature combinations with `--all-targets` so Cargo feature unification cannot hide missing imports/backends. Removed WGPU/Vulkan/ROCm features must be rejected by the source/manifest guard, not compiled as dormant support.

**Step 3: Validate isolated workspaces**

Run and preserve full output:

```text
cargo test --manifest-path mcp/Cargo.toml --all-targets -- --nocapture
cargo check --manifest-path mcp/Cargo.toml --all-targets
cargo test --manifest-path mesh/Cargo.toml --all-targets -- --nocapture
cargo check --manifest-path mesh/Cargo.toml --all-targets
```

**Step 4: Validate desktop**

Run in `desktop`:

```text
npm test
npm run lint
npm run build
```

Also run the Rust desktop tests/check from the root workspace.

**Step 5: Classify output in INFO -> WARN -> ERROR order**

For every log, record every relevant INFO/WARN/ERROR/dead-code/unused line with workspace, source, explanation, and resolution/owner. Zero new warnings is required. Unexpected dead code or an unused path is a failure even if compilation exits zero.

**Step 6: Run source/dependency guards**

Confirm no Polars, `.fstore`, production live-feature JSONL, shared feature/strategy/SMC/live-decision/GPU-contract f32 narrowing, old Gene-bearing artifact/wire version, bare federation/mesh Gene JSON, runtime auto-converter/cache, stale CPU authority, unmanaged pool construction, raw CPU-heavy `spawn_blocking`, `num_cpus` capacity decision, stale `-Zthreads`, legacy mesh scheduling, integrated/WGPU/Vulkan/ROCm or exact Apple-Metal GPU-backend execution, generic default-adapter selection, or first-card-only truncation remains. Do not flag legitimate broker metals asset/category/symbol behavior. Confirm GPU ABI v3 and every prior f32-derived artifact schema are rejected rather than interpreted, and every NVIDIA lane is capability-gated. Reconcile the generated vector-ta manifest against every production indicator/output and reject reachable f32 buffers/casts/dispatch, empty or host-fallback `_f64` kernels, zero-launch success, hard-coded device zero, silent indicator drops, and unexplained panic/truncation/duplicate/degenerate ledger buckets. The non-Vortex vocabulary allow-list names exact shared import/provenance symbols and exact app/API/CLI/desktop import adapters/tests; it does not exempt direct parsing or producer-specific runtime branching. Classify legitimate model-local f32 adapters and probabilities/labels/inventory/reporting values that provably cannot affect live veto/sizing/trades.

**Step 7: Run local runtime/concurrency probes**

On the 6-core/12-thread host prove automatic effective limit `10`, async responsiveness under saturation, import/search/model/child overlap under the ceiling, canonical Vortex-only runtime, and Task 1 performance deltas.

**Step 8: Commit any diagnostic fixes one coherent file/group at a time**

Never batch unrelated warning cleanup into a giant commit. Re-run the exact command that exposed each issue before moving on.

**Step 9: Commit validation record**

```text
docs: record Vortex and CPU validation evidence
```

### Task 18A: Run a pinned-nightly dependency-upgrade experiment

**Files:**
- Create: `docs/audits/2026-08-15-dependency-upgrade-experiment.md`
- Create before candidate resolution: `crates/neoethos-data/tests/vortex_067_backward_compat.rs`
- Create before candidate resolution: `crates/neoethos-data/tests/fixtures/vortex_0_67_compat/` frozen canonical OHLC, projected feature/validity, manifest/provenance, and crash-state fixtures
- Create before candidate resolution: `crates/neoethos-app/tests/live_vortex_067_backward_compat.rs`
- Create before candidate resolution: `crates/neoethos-app/tests/fixtures/vortex_0_67_live_compat/` frozen pending/completed live rows and lifecycle references
- Create before corpus use: `crates/neoethos-data/tests/github_snapshot_contract.rs`
- Create before corpus use: `crates/neoethos-data/tests/fixtures/github_snapshot_contract/` synthetic zero-byte-marker, unsupported-timeframe, missing-convention, and heuristic-metadata tar manifests
- Modify in isolated candidate worktrees only: root `Cargo.toml`/`Cargo.lock`, `mcp/Cargo.toml`/`mcp/Cargo.lock`, and `mesh/Cargo.toml`/`mesh/Cargo.lock`
- Modify source only where a candidate's documented API change requires an explicit, reviewed migration

**Step 1: Freeze toolchain, vendor, and data provenance**

Use the repository-pinned Rust nightly for every baseline/candidate resolve, compile, test, and benchmark; record exact `rustc -Vv` and `cargo -Vv`. Treat `rust-version` only as a separate packaging/MSRV check. Record exact Git commits, lockfile hashes, feature graphs, and `cargo tree` path proof for the custom LightGBM, XGBoost, sklears, rlkit, and project-modified f64 `vector-ta` vendors. Exact-pin their public wrapper versions as needed so Cargo cannot silently bypass a `[patch]`. Vortex is not a custom fork and remains eligible for upgrade.

Before changing Vortex, freeze current 0.67-produced canonical fixtures that represent data for which the original user source may no longer exist: OHLC generation, projected f64 features plus validity, live pending intent, completed live experience, manifests, typed producer/artifact provenance, generation references, and interrupted publication/lifecycle states. Record exact schema, hashes, identities, and `f64::to_bits()` values. These are backward-read durability fixtures, not newly re-imported candidate data.

Download the public `kosred/neoethos-data` release `snapshot-2026-08-09` into a quarantine input area: its 14 per-symbol Vortex tar assets plus `symbol_metadata.json`. Verify published SHA-256 digests, but do not treat a tar path, raw `data.vortex`, or `.complete` filename as schema/provenance proof. First make `github_snapshot_contract` RED against the observed XAUUSD shape: a zero-byte `data.vortex.complete`, legacy paths, unsupported H2/H3/H6/H8/M6/M12/M20 members, absent timestamp unit/`BarTimestampConvention`/generator schema/broker identity, and heuristic commission/swap metadata. Runtime/migration must reject or explicitly report every unsupported member and must never consume `symbol_metadata.json` financial estimates.

The release becomes an external regression corpus only if the exact original generator/source schema, timestamp physical unit, bar-open convention, OHLCV physical types, and source provenance are independently recovered and frozen with evidence. Then the strict offline migration accepts only members whose timeframe is in the canonical 14-variant set and whose complete schema/convention validates; it records each rejected extra and old/new manifest/generation hash. It maps accepted data only to external/non-broker-real identity unless exact broker environment/account/server/`symbolId` mapping is independently proven. If that evidence cannot be recovered, classify the published release unusable for canonical truth and substitute a newly generated provenance-complete external multi-pair fixture; do not weaken the importer or fabricate metadata to keep the full corpus mandate.

Use a representative accepted subset locally and the complete provenance-passing replacement/accepted corpus on the VPS. Use identical hashes, settings, seeds, worker counts, and device policy for every candidate to compare Vortex/schema/timestamps, features, validity, and signals; assert every candidate stops with the same typed unsupported before financial evaluation. A separate full-tuple fixture is acceptable only when it is an exact time-aligned cTrader capture for its own canonical bars and includes synchronized Bid/Ask, conversion legs, exact symbol/account contract, and authoritative deal/PnL truth. A detached or typical broker-cost fixture cannot be combined with external OHLCV.

**Step 2: Create isolated resolver candidates**

Keep four independent states:

1. current baseline lockfiles;
2. semver-compatible updates only;
3. full breaking stable-release updates for every non-custom dependency, including the current Vortex release and its aligned Arrow/Parquet graph;
4. explicitly selected prereleases such as CubeCL 0.11 only when relevant to the supported native-CUDA/model stack.

Run installed-nightly Cargo's dry-run/update commands first and capture the full solver output. Use Cargo's `--ignore-rust-version` for the nightly candidates where supported so `mcp`/`mesh` package MSRV 1.91 does not silently select only 1.91-compatible dependencies; keep and test that MSRV separately as packaging policy. Current upstream docs are checked against live crates.io and exact downloaded source/release history; ctx7 is supporting evidence only because it may lag. Never mix candidates into one lockfile or treat `cargo update --breaking` as selecting prereleases automatically.

**Step 3: Validate the compatible candidate locally**

For root, `mcp`, and `mesh`, run clean release builds, all affected `cargo test --all-targets`, standalone feature matrices, and full workspace checks under the CPU budget. Preserve and classify every INFO/WARN/ERROR/dead-code/unused line. Prove every vendor patch still resolves locally; an unused patch or registry replacement is a failure even if the build succeeds. Record API changes, clean/incremental time, peak RAM, output size, and dependency removals/additions.

**Step 4: Migrate and validate breaking/prerelease candidates one dependency family at a time**

Upgrade Vortex with its exact aligned Arrow/Parquet versions and rerun import, publication, scan/projection, f64 bit, bounded-memory, and crash tests before touching an unrelated family. Every Vortex candidate must also reopen all frozen 0.67 OHLC/feature-validity/live pending/completed/manifest/provenance/crash-state fixtures with bit-identical values and unchanged typed identities/references. A candidate that cannot read them is `hold`/`reject`; acceptance then requires a separately versioned bounded offline migration that embeds or retains a proven 0.67 reader, preserves originals, writes/verifies/publishes atomically, preserves all semantic/provenance identities, survives every crash point, and is tested before runtime drops old-reader support. Re-importing a new fixture or deleting the only old generation is not a migration. Upgrade GPU/model/runtime families separately. For every candidate, regenerate `backend_inventory` from that exact resolved source/documentation and compare worker/pool behavior with the locked baseline. Also regenerate every row's canonical `RelevantDependencySetV1` and every affected `ModelImplementationIdentityV1` from the candidate lock/source/features; a relevant package change must change only affected feature-plan/model identities, while an unrelated change must not. Rerun source/dependency/model-identity mutation golden tests plus Task 4A/12 fail-before-work, saturation, cancellation, classification, and full overlap probes before assigning `accept`; a stale semantic/model dependency fingerprint or new/changed/unclassified/unbounded Rayon/native/global pool is `hold` or `reject` even if value tests pass. Use compile errors as an inventory, add TDD regressions for semantic changes, and never make a mechanical API edit without reading the current upstream contract. Stop or split a candidate when independent attribution would be lost.

**Step 5: Compare mathematical and performance truth locally**

For every buildable candidate compare canonical Vortex byte/value identity where the format contract permits it, vector-ta indicator bits, normalized features/validity, and signals on the immutable external corpus, then require the same typed unsupported before any trade/fill/cost/ledger/metric computation. Compare ordered trades/fills/ledger, broker spread/commission/swap/conversion inputs and costs, and the entire metric/promotion tuple only on an exact synchronized broker-captured replay fixture whose typed capability binds those precise bars and broker contracts. Run worker `1` versus `auto` and fixed OOS/CPCV partitions within the supported scope. Any unexplained feature/signal change, unsupported-vs-executed boundary difference, or broker-capable trade/cost/result change is a blocker; higher profitability alone is not an acceptance signal. Benchmark scan/import/search/model end-to-end throughput and peak memory from repeated runs with warm-up reported separately. The explicit CubeCL prerelease is evaluated only for exact-named model-local CUDA adapters; it cannot restore the removed CubeCL search/backtest or any WGPU/HIP lane.

**Step 6: Select, do not wholesale-merge**

Produce a dependency-by-dependency table: current/candidate version, reason, API/security/performance effect, warnings, CPU evidence, GPU evidence pending, mathematical parity, and disposition (`accept`, `hold`, `reject`). Only validated dependencies become a clean selected-upgrade commit for Task 18 hardware testing; retain the other candidate branches/worktrees as experimental evidence, not master changes.

**Step 7: Commit the audit and selected candidate separately**

```text
docs: record pinned-nightly dependency upgrade experiment
build: update validated non-custom dependencies
```

### Task 18: Validate CUDA on a real AMD-CPU + RTX 3090/4090 VPS

**Files:**
- Modify: `docs/audits/2026-08-15-vortex-cpu-validation.md`
- Modify: `docs/audits/2026-08-15-cuda-artifact-inventory.md`
- Create: `crates/neoethos-gpu-contracts/src/cuda_capability_registry.rs`
- Create: `crates/neoethos-gpu-contracts/capabilities/cuda-promotions-v1.json`
- Create: `crates/neoethos-gpu-contracts/tests/cuda_capability_registry.rs`
- Modify CUDA/source code only when real-card evidence exposes a defect

**Step 1: Provision only when local/root validation is clean**

Use the already-authorized Vast.ai workflow to select an AMD CPU host with RTX 3090 or 4090, sufficient RAM/disk, verified reliability/network, and current price. A one-card host may be used first for dependency-candidate and single-device work, but the release gate also requires a host exposing at least two eligible NVIDIA cards concurrently to one process so explicit subset/all-card independent-job selection is real rather than simulated. If no suitable multi-card offer is available within the operator-approved cost, stop before integration and report the advertised multi-card capability as blocked; do not merge it as merely pending. Use the dedicated NeoEthos SSH identity; never print private-key contents. Destroy any stale/failed instance promptly and never leave a billed instance idle between phases.

**Step 2: Capture hardware/toolchain truth**

Record `nvidia-smi`, `nvcc --version`, Rust/Cargo versions, `available_parallelism`, host CPU inventory, cgroup quota, RAM, disk, driver, and every visible GPU's stable identity/name/compute capability. Run the common host-auto resolver once before Cargo; on a 3090 it must produce worker limit `available_parallelism - 2` plus `NEOETHOS_CUDA_ARCHS=86`, on a 4090 `...=89`, and on a mixed host the canonical sorted/deduplicated set such as `86;89`. The same sealed `BuildHostPlanV1`/hash must reach vector-ta, native search, XGBoost, LightGBM, every retained CUDA model artifact, and generated runtime metadata. Set `CUDACXX` only through the plan when nvcc discovery is ambiguous and force `CUDA_FAST_MATH=0`; native CUDA independently retains `-fmad=false`. Do not set an architecture manually for host validation, and reject any old `CUDA_ARCH`, `CUDA_ARCHS`, singular `NEOETHOS_CUDA_ARCH`, fixed-card, first-card-only, or per-builder probe/default path. A separately invoked release/cross build may use an explicit reviewed architecture matrix, but its manifest must say `cross_release_explicit`; packaged images remain only `build_compatible_unvalidated` until the complete suite promotes that exact compute capability and module path.

**Step 3: Build meaningful CUDA feature combinations with full logs**

Compile standalone and aggregate native-CUDA features with `--all-targets` for the baseline, compatible, selected-breaking, and explicit-prerelease dependency candidates that passed Task 18A locally. The explicit matrix includes `neoethos-gpu-cuda --features cuda`, `neoethos-search --features gpu-b-native`, `neoethos-data --features gpu-cuda`, search `gpu-cuda`, CLI/app `gpu-nvidia`, and the CLI benchmark aggregate `gpu-nvidia,gpu-bench-cuda`; Cargo/default workspace checks alone do not enable these lanes. `burn-cuda-backend` remains a separate measured A/B candidate rather than silently joining the production aggregate. Use the resolved automatic worker limit—which already equals effective logical threads minus the fixed two-thread reserve—for Cargo/native jobs, narrowed only by measured RAM/disk caps; never subtract the reserve twice or fall back to script defaults such as eight. Read every INFO/WARN/ERROR/dead-code/unused line; a successful link is not device validation. WGPU/Vulkan/ROCm are not release candidates.

**Step 4: Prove real kernel execution and numerical parity**

First inspect every exact `CudaBuildManifestV1` and the bytes it describes: compiler/toolkit, canonical gencode list, mandatory precision flags, ABI, semantic-source identity, SASS/PTX images from `cuobjdump`/PTX inspection, and each loaded artifact's actual SHA-256/size must match. Assert ABI v4 agreement and the selected device UUID/PCI identity. Set `NEOETHOS_RUN_CUDA_SMOKE=1` for the native ABI smoke and `NEOETHOS_REQUIRE_GPU=1` for every search/data/model device probe so a skip or CPU fallback is a hard failure; scan the logs for skip/fallback wording in addition to exit status. Run f64 device-required compile/launch/readback probes that fail when no kernel launches, the lane is unsupported, ABI v3 is accepted, fast-math/FMA correctness controls are missing, or CPU fallback occurs.

Exercise executable module routes separately rather than treating a fatbin load as proof of SASS. For every advertised compute capability/backend artifact, first force SASS with PTX/JIT disabled and prove a matching inspected SASS image was loaded; then, only when embedded or standalone PTX is intended to remain supported, force each PTX route with a controlled empty driver cache and repeat the complete manifest/formula/parity/launch suite, followed by a warm-cache run. Each run must emit `ModuleLoadEvidenceV1` containing the exact UUID/PCI/session ordinal, compute capability, driver/runtime/toolkit, build-manifest ID, loaded artifact bytes/hash/container, actual SASS or PTX/JIT/cache route, load result/timing, and matching launch/readback identity. Contradictory force controls, a stale failure after success, direct-loader bypass, zero-byte/unknown-prebuilt bytes, artifact/hash mismatch, or an unvalidated automatic fallback fails strict mode. If the PTX/JIT route cannot be admitted/bounded or fully validated, disable it in strict production instead of allowing it to inherit a SASS result.

Compare full CPU/GPU indicator bits, validity, and signals, including the f64-only threshold fixture, on the exact same GitHub corpus hashes for every dependency candidate and require identical typed unsupported before financial evaluation. Compare trade-ledger and metric tuples only on an exact synchronized broker-captured replay fixture with the complete broker-truth capability. Any remaining f32-only or nominal-but-unlaunchable device path that violates the approved f64 parity contract is a failure to remediate or disable, not an accepted tolerance. Before promotion, verify the clean commit and exact Git blob payload hashes used by the manifest. Create one `CudaHardwarePromotionKeyV1` per `{backend_artifact_id, compute capability, reviewed driver/runtime/toolkit contract, ABI/precision flags, semantic source/build manifest, exact artifact hash, module-load path}` and promote only the manifest rows/model backend proven under that key. A single-card 3090 can promote only sm86 and the routes actually forced through the full suite; a single-card 4090 can promote only sm89. Every other packaged image remains build-compatible/unvalidated until equivalent real-hardware evidence exists, and a mixed-architecture multi-card host promotes each architecture/path independently. A model-local CubeCL/Candle/tree artifact has its own key and cannot inherit vector-ta/native-search proof. Failed or unrun rows remain fail-closed candidates or are downgraded to reviewed CPU/unsupported; promotion is an explicit reviewed registry diff, never an automatic runtime cache.

Store reviewed promotions only in the tracked `cuda-promotions-v1.json` registry, parsed by the typed `cuda_capability_registry` module. Its canonical SHA-256 encoding uses a fixed domain/version tag, fixed enum tags, big-endian lengths/counts, byte-sorted backend/compute-capability/load-path entries, and canonical embedded build/key bytes; duplicate/conflicting keys, unknown fields/versions, missing artifacts, or absolute/local paths reject. Runtime recomputes every referenced build/artifact/semantic identity and fails stale keys closed. Tests mutate source, build flags, gencode, ABI, artifact bytes, driver/toolkit contract, compute capability, and load path one at a time and prove the old key rejects; an explicit Task-18 revalidation/regeneration is required before a changed key is committed.

Test `CpuOnly` on the GPU host and prove it creates no CUDA context through search, every direct model adapter, or vector-ta indicators. Test automatic and explicit one-card selection. On the required multi-card host, test explicit subsets, all-card concurrent independent search/data/indicator/model-training jobs, distinct UUID/PCI assignment (including a nonzero runtime ordinal), per-card promotion key and module-load evidence, training-artifact identity, completion reordering, device loss, and one-versus-many feature/validity/signal identity. Compare one-versus-many trades/ledger/metrics only on the exact synchronized broker-capable fixture; otherwise every configuration must return the same typed unsupported result before financial arithmetic. Every architecture/load path actually assigned on the multi-card host must have its own complete promotion key. Each retained CUDA model backend must show a real kernel launch from its own verified artifact on its assigned device and no CPU fallback. Test deterministic native single-population sharding/merge only if that optional mode is being enabled; otherwise prove its preflight rejects it explicitly. Reject integrated/non-NVIDIA/unknown/unpromoted architecture or module-path selectors at preflight. A candidate that changes trades, costs, or profitability without an explained reviewed semantic correction is rejected even when faster.

**Step 5: Prove VPS CPU limits and overlap**

Verify the effective worker limit follows process/cgroup `available_parallelism - 2`, not host inventory. Run the same import/Vortex/search/model/GPU-feeder/child overlap probe and observe active workers at/below the limit while GPU occupancy remains unconstrained by CPU permits. Include cold CUDA context creation, forced SASS load, forced PTX parse/JIT/link/cache work when supported, and simultaneous per-card initialization; each active host phase acquires its classified lease before touching the driver and holds it through completion/cancellation/cleanup. Only a proven passive launch/wait segment is `device_only`.

**Step 6: Measure performance honestly**

Record per-card kernel occupancy, batch width, streams, transfer time, preprocessing, end-to-end throughput, GPU memory, CPU utilization, peak RAM, candidate version, and selected device set. Optimize only from profiles. Do not describe compilation, allocation, a no-op kernel, or a CPU fallback as acceleration.

**Step 7: Stop/destroy the billed instance when evidence is captured**

Confirm termination and record the instance ID/time/cost without exposing credentials.

**Step 8: Commit fixes/evidence**

Use focused commits for real-card defects. Commit the exact capability-manifest promotions and strict enablement only after their evidence, then commit the audit record:

```text
feat: promote hardware-verified f64 CUDA capabilities
docs: record real NVIDIA Vortex and CPU evidence
```

### Task 19: Review, integrate, and preserve honest completion boundaries

**Files:**
- Review all branch diffs and validation records
- Update user-facing documentation/config examples as required

**Step 1: Request an initial independent integration review**

Review correctness, security, Vortex/import boundary, f64 import preservation (including prop-firm/SavedStrategy/scheduler layouts), Polars/`.fstore` removal, permit lifetimes, `neoethos-data` Rayon containment, cancellation, startup order, native worker classifications, unified vector-ta device assignment, mesh compatibility, warnings, and one-card plus real multi-card evidence. Fix every release blocker and rerun its proving command.

**Step 2: Verify branch state and commits**

Perform a separate read-only inventory/diff of current master dirty state, ignored files, and `.codex/` before staging integration. Classify owner/purpose and whether each item is a credential, cache, generated target, temporary probe, audit secret, unrelated user edit, or source-worthy project change. Preserve credentials/caches/generated output and unrelated user edits without staging them. Clean, test, and deliberately integrate only reviewed source-worthy changes—including reviewed `.codex/` or prior dirty-master work when appropriate—and record every included/deferred item; blanket inclusion and blanket exclusion are both forbidden.

**Step 3: Confirm the already-validated broker boundary**

Verify Task 16C's fail-closed `broker_real` gate, Task 17's complete post-gate local matrix, and Task 18's affected real-card checks are all present in the reviewed evidence. Infrastructure may be integrated behind disabled broker-real capabilities; it must not be called a complete bot release.

**Step 4: Revalidate the exact final pre-integration candidate**

Tasks 18A and 18 may change selected lockfiles, CUDA source, and capability-manifest rows after Task 17. Therefore rerun the entire Task 17 format, root/member/all-target, standalone-feature, MCP, mesh, desktop, source/dependency guard, and INFO/WARN/ERROR/dead-code classification matrix on the exact final source plus selected lockfiles and hardware-promotion commit. No “affected subset” may substitute for this clean final-source matrix. Fixes from the independent review trigger the same complete rerun. Confirm the focused broker gates remain GREEN.

**Step 5: Build an integration candidate with current master deliberately**

Before integration, require successful real use of at least two simultaneously visible eligible NVIDIA cards for the advertised independent-job multi-selection contract; one-card evidence cannot waive this gate. Fetch current master, inspect divergence and the classified dirty/`.codex/` inventory, then create a separate integration candidate/worktree through the approved merge/PR path. Resolve conflicts by behavior and tests, not by choosing one side wholesale; do not update the protected/master ref yet.

**Step 6: Revalidate the complete integrated result**

After conflict resolution, rerun the entire Task 17 matrix/guards/full-log classification on the integration candidate regardless of how small the diff appears. If any source, manifest, lockfile, CUDA flag, capability row, or generated input differs from the hardware-tested candidate, reprovision/retain an equivalent AMD+NVIDIA environment and rerun the complete relevant Task 18 real-device suite before merge; multi-card behavior changes require the real multi-card suite. A focused test or subjective “unaffected” judgment cannot replace these gates.

**Step 7: Report exact completion levels**

Separate:

- implemented;
- CPU-tested locally;
- full logs reviewed;
- real NVIDIA-validated;
- integrated/merged;
- still pending broker-truth replay/OOS/profitability work.

Never claim `>55%` win rate at `2R`, Risky Mode capital targets, prop-firm challenge success, or broker-real historical PnL unless the later broker-truth and OOS/CPCV gates produce that evidence.

**Step 8: Commit/merge only after all required gates pass**

```text
merge: integrate Vortex runtime and unified CPU budget
```

---

## Follow-up release gate — separate broker-truth plan required

Before profitability research or promotion claims, write and independently review a separate plan that:

- wires `ProtoOAGetTickData` into paginated <=7-day Bid and Ask downloads;
- performs and preserves a read-only retention probe against the exact cTrader account/environment, recording every requested range/window, raw response/page boundary, earliest/latest returned tick, empty/error result, and probe date; it reports only observed availability and never infers a retention duration from timestamp fields or a single successful request;
- stores versioned broker quote events and conversion legs directly in Vortex;
- derives chronological last-known Bid/Ask, quote-staleness limits, and any trading-session reset boundary from official protocol evidence plus captured raw broker behavior before implementing them; it does not invent an unconditional reset rule, and it fails on unproven or missing coverage;
- executes buy entries at Ask/exits at Bid and sells at Bid/exits at Ask on quote/tick replay, including ambiguous intrabar ordering;
- derives commission, min commission, conversion fees, swap/rollover, volume, pip/tick value, margin, and rounding from the exact cTrader symbol/account contract and reconciles them against captured raw responses and demo closed deals;
- captures the exact `ProtoOASymbol` fields and documented units used by every formula, treats the broker-supplied live unrealized-PnL response as authoritative for live reconciliation, and treats `ProtoOAClosePositionDetail` plus its deal records as authoritative closed-execution/PnL truth; locally reconstructed live unrealized/closed PnL, heuristic unit guesses, missing-field-to-zero, and remembered/default symbol values are forbidden and fail closed;
- marks unavailable historical contract/execution truth as unsupported and fails before financial evaluation; it does not offer a heuristic `scenario=approximate` route, while scenario labels may select only exact captured broker inputs with typed provenance;
- then reruns deterministic OOS/CPCV and reports the full metric tuple, not win rate alone.

That plan is intentionally not folded into CPU scheduling: changing scheduling must not silently change financial arithmetic, and the current source is not assumed to be the mathematical oracle merely because GPU results match it.
